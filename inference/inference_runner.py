"""CoppeliaInferenceRunner — run PPO/Dreamer policies on CoppeliaSim scenes.

Two modes:
  1. Single episode (demo/video): run_episode(), prints step-by-step like Lab5 main.py.
  2. Batch (data collection): run_batch(N), auto-resets between episodes, saves CSV.

Usage:
    from inference.inference_runner import CoppeliaInferenceRunner
    runner = CoppeliaInferenceRunner(algo="ppo", use_cbf=True)
    result = runner.run_episode()
    runner.close()
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

# ── Ensure project root is on sys.path for imports ─────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from inference.constants import (
    LIDAR_MAX_DIST, N_LIDAR, DIST_NORM_MAX,
    ROBOT_INIT_POS, GOAL_DIST, CRASH_DIST, MAX_STEPS,
    GOAL_BONUS, CRASH_COST, STEP_PENALTY,
    DEFAULT_PPO_CHECKPOINT, DEFAULT_DREAMER_CHECKPOINT, DEFAULT_DREAMER_CONFIG,
    DEFAULT_SCENE,
)
from inference.cbf import get_safe_action


# ── Default scene paths ────────────────────────────────────────────────────
_SCENES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes")


def _resolve_scene_path(scene: str) -> str:
    """Resolve a scene name/path to absolute .ttt path.

    Absolute → as-is. Relative with ./ or ../ → from cwd.
    Otherwise → look in inference/scenes/ first, then cwd.
    """
    if os.path.isabs(scene):
        return scene
    if scene.startswith("./") or scene.startswith("../"):
        return os.path.abspath(scene)
    in_scenes = os.path.join(_SCENES_DIR, scene)
    if os.path.isfile(in_scenes):
        return in_scenes
    return os.path.abspath(scene)


class CoppeliaInferenceRunner:
    """Run PPO or DreamerV3 policies on CoppeliaSim scenes.

    Preconditions:
        - CoppeliaSim running with target scene loaded and simulation started.
        - Scene has Lua script on /Floor with step_centralizzato function.
        - Scene has /limo_1 object.
    Postconditions:
        - run_episode() returns metrics dict.
        - run_batch(N) returns list of metrics dicts.
        - close() disconnects.
    """

    def __init__(
        self,
        algo: str = "ppo",
        use_cbf: bool = True,
        scene_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        verbose: bool = True,
    ):
        """Initialize the inference runner.

        Args:
            algo: "ppo" or "dreamer".
            use_cbf: Apply CBF safety filter after policy action.
            scene_path: Path to .ttt scene. None → DEFAULT_SCENE.
            checkpoint_path: Path to model checkpoint. None → default per algo.
            device: "cpu" or "cuda".
            verbose: Print step-by-step info (Lab5 main.py style).
        """
        self.algo = algo.lower()
        if self.algo not in ("ppo", "dreamer"):
            raise ValueError(f"algo must be 'ppo' or 'dreamer', got '{algo}'")
        self.use_cbf = use_cbf
        self.verbose = verbose

        # Resolve scene path
        if scene_path is None:
            scene_path = DEFAULT_SCENE
        self.scene_path = _resolve_scene_path(scene_path)

        # Resolve checkpoint path
        if checkpoint_path is None:
            if self.algo == "ppo":
                checkpoint_path = os.path.join(_PROJECT_ROOT, DEFAULT_PPO_CHECKPOINT)
            else:
                checkpoint_path = os.path.join(_PROJECT_ROOT, DEFAULT_DREAMER_CHECKPOINT)
        else:
            if not os.path.isabs(checkpoint_path):
                checkpoint_path = os.path.join(_PROJECT_ROOT, checkpoint_path)
        self.checkpoint_path = checkpoint_path

        self.device = device
        self.sim = None
        self.client = None
        self.limo_handle = None
        self.script_handle = None
        self._model = None

        # Metrics counters for batch
        self.episode_count = 0

        self._connect()

    # ── Public API ─────────────────────────────────────────────────────────

    def run_episode(self, max_steps: int = MAX_STEPS) -> Dict[str, Any]:
        """Run a single inference episode on the running CoppeliaSim scene.

        Preconditions:
            - CoppeliaSim simulation is running (started from GUI).
        Postconditions:
            - Robot ends at goal, crashed, or timed out.
            - Simulation continues running (for multi-episode batch).

        Args:
            max_steps: Maximum steps before timeout.

        Returns:
            dict with: outcome, n_steps, cumulative_reward, cbf_interventions,
            cbf_rate, final_distance, episode, algo, cbf,
            actions (list of [v,w]), trajectory (list of per-step values).
        """
        self.episode_count += 1
        ep_num = self.episode_count

        # ── Reset robot to initial position ────────────────────────────────
        self._reset_robot()

        # ── Initialize Dreamer states if needed ────────────────────────────
        recurrent_state = None
        latent_state = None
        action_tensor = None
        if self.algo == "dreamer":
            import torch
            dreamer = self._model
            recurrent_state = torch.zeros(1, dreamer.recurrentSize, device=self.device)
            latent_state = torch.zeros(1, dreamer.latentSize, device=self.device)
            action_tensor = torch.zeros(1, dreamer.actionSize, device=self.device)

        # ── Episode loop ───────────────────────────────────────────────────
        done = False
        step = 0
        cumulative_reward = 0.0
        cbf_interventions = 0
        outcome = "timeout"
        prev_distance = 0.0
        actions: List[List[float]] = []
        trajectory: List[List[float]] = []

        # Get initial distance
        raw_buffer = self.sim.callScriptFunction(
            "step_centralizzato", self.script_handle, [0.0, 0.0]
        )
        data_array = np.frombuffer(raw_buffer, dtype=np.float32)
        prev_distance = float(data_array[N_LIDAR])

        safe_action = [0.0, 0.0]

        while not done and step < max_steps:
            # ── Get observation from CoppeliaSim ───────────────────────────
            raw_buffer = self.sim.callScriptFunction(
                "step_centralizzato", self.script_handle, safe_action
            )
            data_array = np.frombuffer(raw_buffer, dtype=np.float32)

            lidar_norm_raw = data_array[:N_LIDAR]    # normalized [0,1] from CoppeliaSim
            distance = float(data_array[N_LIDAR])     # meters
            angle = float(data_array[N_LIDAR + 1])    # radians

            # ── Observation ────────────────────────────────────────────────
            # LiDAR from step_centralizzato is ALREADY normalized to [0,1].
            # Our PPO/dreamer expect [0,1] input (same as custom env).
            lidar_norm = np.clip(lidar_norm_raw, 0.0, 1.0).astype(np.float32)
            dist_norm = np.clip(distance / DIST_NORM_MAX, 0.0, 1.0)
            obs = np.concatenate([
                lidar_norm,
                [np.float32(dist_norm), np.float32(np.cos(angle)), np.float32(np.sin(angle))],
            ]).astype(np.float32)

            # Convert to meters for CBF and crash detection
            lidar_data_m = lidar_norm_raw.astype(np.float64) * LIDAR_MAX_DIST
            min_lidar_m = float(np.min(lidar_norm)) * LIDAR_MAX_DIST

            # ── Check termination ──────────────────────────────────────────
            terminated_goal = distance < GOAL_DIST
            terminated_crash = min_lidar_m < CRASH_DIST
            terminated = terminated_goal or terminated_crash

            if terminated:
                reward = self._compute_reward(
                    distance, prev_distance, lidar_norm,
                    terminated_goal, terminated_crash,
                )
                cumulative_reward += reward
                trajectory.append([
                    step, safe_action[0], safe_action[1],
                    min_lidar_m, distance,
                ])
                step += 1

                if terminated_goal:
                    outcome = "goal"
                    if self.verbose:
                        print(f"\n  ✅ [GOAL] Reached in {step} steps | "
                              f"reward={cumulative_reward:.2f}")
                else:
                    outcome = "crash"
                    if self.verbose:
                        print(f"\n  💥 [CRASH] at lidar={min_lidar_m:.3f}m | "
                              f"reward={cumulative_reward:.2f}")
                break

            # ── Get policy action ──────────────────────────────────────────
            if self.algo == "ppo":
                action_np, _ = self._model.predict(obs, deterministic=True)
                action_np = np.asarray(action_np).flatten()
                v_nom = float(np.clip(action_np[0], 0.0, 1.0))
                w_nom = float(np.clip(action_np[1], -1.0, 1.0))
            else:
                import torch
                action_out, recurrent_state, latent_state = self._dreamer_inference_step(
                    self._model, obs, recurrent_state, latent_state, action_tensor
                )
                action_np = action_out.cpu().numpy().reshape(-1)
                v_nom = float(np.clip(action_np[0], 0.0, 1.0))
                w_nom = float(np.clip(action_np[1], -1.0, 1.0))
                action_tensor = action_out

            # ── Apply CBF safety filter ────────────────────────────────────
            if self.use_cbf:
                # CBF expects meters; lidar_data_m computed above from normalized
                safe_action = get_safe_action(v_nom, w_nom, lidar_data_m)
                diff_v = abs(v_nom - safe_action[0])
                diff_w = abs(w_nom - safe_action[1])
                if diff_v > 0.01 or diff_w > 0.01:
                    cbf_interventions += 1
                    cbf_status = "⚠️ CBF ACTIVE"
                else:
                    cbf_status = "✅ CLEAR"
            else:
                safe_action = [v_nom, w_nom]
                cbf_status = "Policy only"

            # ── Compute reward ─────────────────────────────────────────────
            reward = self._compute_reward(
                distance, prev_distance, lidar_norm,
                terminated_goal=False, terminated_crash=False,
            )
            cumulative_reward += reward
            prev_distance = distance

            # ── Log ────────────────────────────────────────────────────────
            actions.append([safe_action[0], safe_action[1]])
            trajectory.append([
                step, safe_action[0], safe_action[1],
                min_lidar_m, distance,
            ])

            if self.verbose:
                print(
                    f"[{cbf_status}] ep={ep_num} step={step:>3} | "
                    f"dist={distance:.2f}m | "
                    f"nom=[{v_nom:.3f}, {w_nom:.3f}] → "
                    f"app=[{safe_action[0]:.3f}, {safe_action[1]:.3f}]"
                )

            step += 1
            time.sleep(0.05)

        # ── Handle timeout ─────────────────────────────────────────────────
        if step >= max_steps and not terminated:
            outcome = "timeout"
            if self.verbose:
                print(f"\n  ⏰ [TIMEOUT] {max_steps} steps | "
                      f"reward={cumulative_reward:.2f}")

        episode_metrics = {
            "episode": ep_num,
            "algo": self.algo,
            "cbf": self.use_cbf,
            "outcome": outcome,
            "n_steps": step,
            "cumulative_reward": cumulative_reward,
            "cbf_interventions": cbf_interventions,
            "cbf_rate": cbf_interventions / max(step, 1),
            "min_lidar": min_lidar_m if not terminated_goal else 0.0,
            "final_distance": distance,
            "actions": actions,
            "trajectory": trajectory,
        }

        return episode_metrics

    def run_batch(self, n_episodes: int = 10, max_steps: int = MAX_STEPS) -> List[Dict[str, Any]]:
        """Run multiple episodes with auto-reset (stop/start sim) between.

        Preconditions:
            - CoppeliaSim running with simulation started.
        Postconditions:
            - N episodes completed. Simulation stopped after last.
            - Robot position reset between episodes.

        Args:
            n_episodes: Number of episodes.
            max_steps: Max steps per episode.

        Returns:
            list of episode metrics dicts.
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  Batch inference: {n_episodes} episodes")
            print(f"  Algo: {self.algo.upper()} | CBF: {'ON' if self.use_cbf else 'OFF'}")
            print(f"  Scene: {self.scene_path}")
            print(f"{'='*60}\n")

        results = []
        for ep in range(n_episodes):
            if ep > 0:
                try:
                    self.sim.stopSimulation()
                    time.sleep(0.2)
                except Exception:
                    pass
            try:
                self.sim.startSimulation()
                time.sleep(0.1)
            except Exception as e:
                print(f"  ❌ startSimulation failed ep {ep+1}: {e}")
                break

            result = self.run_episode(max_steps=max_steps)
            results.append(result)

            if self.verbose:
                print(f"  → Ep {ep+1}: {result['outcome']} | "
                      f"{result['n_steps']} steps | "
                      f"reward={result['cumulative_reward']:.2f} | "
                      f"CBF rate={result['cbf_rate']:.1%}\n")

        try:
            self.sim.stopSimulation()
        except Exception:
            pass
        return results

    def close(self):
        """Clean up resources."""
        self._model = None
        if self.verbose:
            print("  Inference runner closed.")

    # ── Private ────────────────────────────────────────────────────────────

    def _connect(self):
        """Connect to running CoppeliaSim via ZMQ.

        Raises:
            ConnectionError: If connection fails.
            RuntimeError: If scene objects not found.
        """
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient

        if self.verbose:
            print("  Connecting to CoppeliaSim (localhost:23000)...")
        try:
            self.client = RemoteAPIClient()
            self.sim = self.client.getObject("sim")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to CoppeliaSim. Is it running?\n  Error: {e}")

        try:
            self.limo_handle = self.sim.getObject("/limo_1")
        except Exception as e:
            raise RuntimeError(f"Cannot find /limo_1 in scene: {e}")

        try:
            floor = self.sim.getObject("/Floor")
            self.script_handle = self.sim.getScript(self.sim.scripttype_childscript, floor)
            if self.script_handle == -1:
                self.script_handle = self.sim.getScript(
                    self.sim.scripttype_customizationscript, floor
                )
            if self.script_handle == -1:
                raise RuntimeError("No script found on /Floor")
        except Exception as e:
            raise RuntimeError(f"Cannot find script on /Floor: {e}")

        if self.verbose:
            print(f"  ✅ Connected | limo={self.limo_handle} | script={self.script_handle}")
        self._load_model()

    def _load_model(self):
        """Load trained model (PPO SB3 or DreamerV3)."""
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                f"  Provide a valid --checkpoint or train first."
            )
        if self.algo == "ppo":
            if self.verbose:
                print(f"  Loading PPO from: {self.checkpoint_path}")
            from stable_baselines3 import PPO
            self._model = PPO.load(self.checkpoint_path, device=self.device)
        else:
            if self.verbose:
                print(f"  Loading Dreamer from: {self.checkpoint_path}")
            import torch
            # Dreamer's internal imports (from networks import ...) are bare,
            # so dreamer/ dir must be on sys.path
            _dreamer_dir = os.path.join(_PROJECT_ROOT, "dreamer")
            if _dreamer_dir not in sys.path:
                sys.path.insert(0, _dreamer_dir)
            from dreamer import Dreamer
            from utils import loadConfig

            config = loadConfig(DEFAULT_DREAMER_CONFIG)
            obs_shape = (23,)
            action_size = 2
            action_low = [0.0, -1.0]
            action_high = [1.0, 1.0]
            self._model = Dreamer(obs_shape, action_size, action_low, action_high,
                                  self.device, config.dreamer)
            self._model.loadCheckpoint(self.checkpoint_path)
            for net in ["actor", "encoder", "decoder", "recurrentModel", "posteriorNet"]:
                getattr(self._model, net).eval()

        if self.verbose:
            print(f"  ✅ Model loaded ({self.algo.upper()})")

    def _reset_robot(self):
        """Teleport robot to initial position."""
        try:
            current_pos = self.sim.getObjectPosition(self.limo_handle, self.sim.handle_world)
            current_ori = self.sim.getObjectOrientation(self.limo_handle, self.sim.handle_world)
            new_pos = [ROBOT_INIT_POS[0], ROBOT_INIT_POS[1], current_pos[2]]
            new_ori = [current_ori[0], current_ori[1], ROBOT_INIT_POS[2]]
            self.sim.setObjectPosition(self.limo_handle, self.sim.handle_world, new_pos)
            self.sim.setObjectOrientation(self.limo_handle, self.sim.handle_world, new_ori)
        except Exception as e:
            print(f"  ⚠️  Robot reset failed: {e}")

    @staticmethod
    def _compute_reward(distance, prev_distance, lidar_norm,
                        terminated_goal, terminated_crash) -> float:
        """Matches reward.py compute_reward for consistent metrics."""
        reward = float(prev_distance - distance)
        min_lidar_m = float(np.min(lidar_norm)) * LIDAR_MAX_DIST
        if min_lidar_m < 0.3:
            reward -= (0.3 - min_lidar_m) / 0.3
        if not terminated_goal and not terminated_crash:
            reward += STEP_PENALTY
        if terminated_goal:
            reward += GOAL_BONUS
        if terminated_crash:
            reward -= CRASH_COST
        return reward

    @staticmethod
    def _dreamer_inference_step(dreamer, obs_numpy, recurrent_state, latent_state, action):
        """Matches evaluate_dreamer.py dreamer_inference_step exactly."""
        import torch
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_numpy).float().unsqueeze(0).to(dreamer.device)
            encoded = dreamer.encoder(obs_tensor)
            new_recurrent = dreamer.recurrentModel(recurrent_state, latent_state, action)
            new_latent, _ = dreamer.posteriorNet(
                torch.cat((new_recurrent, encoded.view(1, -1)), -1)
            )
            full_state = torch.cat((new_recurrent, new_latent), -1)
            return dreamer.actor(full_state), new_recurrent, new_latent

    @staticmethod
    def print_results(results: List[Dict[str, Any]], title: str = "Inference"):
        """Print clean summary table of episode results.

        Args:
            results: List of episode metrics from run_batch().
            title: Optional section title.
        """
        if not results:
            print("  No results.")
            return
        n = len(results)
        n_goal = sum(1 for r in results if r["outcome"] == "goal")
        n_crash = sum(1 for r in results if r["outcome"] == "crash")
        n_timeout = sum(1 for r in results if r["outcome"] == "timeout")
        lengths = [r["n_steps"] for r in results]
        rewards = [r["cumulative_reward"] for r in results]
        cbf_rates = [r["cbf_rate"] for r in results]

        sep = "-" * 52
        print(f"\n{sep}")
        print(f"  {title}: {n} episodes")
        print(f"  Algo: {results[0]['algo'].upper()} | "
              f"CBF: {'ON' if results[0]['cbf'] else 'OFF'}")
        print(sep)
        print(f"  Goal rate:          {n_goal / n:.2%} ({n_goal}/{n})")
        print(f"  Crash rate:         {n_crash / n:.2%} ({n_crash}/{n})")
        print(f"  Timeout rate:       {n_timeout / n:.2%} ({n_timeout}/{n})")
        print(f"  Sum:                {(n_goal + n_crash + n_timeout) / n:.2f}")
        print(sep)
        print(f"  Mean episode len:   {float(np.mean(lengths)):.1f}")
        print(f"  Std episode len:    {float(np.std(lengths)):.1f}")
        print(f"  Mean reward:        {float(np.mean(rewards)):.3f}")
        print(f"  Std reward:         {float(np.std(rewards)):.3f}")
        print(f"  Mean CBF rate:      {float(np.mean(cbf_rates)):.1%}")
        print(sep)

    @staticmethod
    def save_csv(results: List[Dict[str, Any]], output_path: str):
        """Save episode results to CSV.

        Args:
            results: List of episode metrics dicts.
            output_path: Output CSV path.
        """
        import csv
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fieldnames = [
            "episode", "algo", "cbf", "outcome", "n_steps",
            "cumulative_reward", "cbf_interventions", "cbf_rate",
            "min_lidar", "final_distance",
        ]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"  💾 Results saved to: {output_path}")
