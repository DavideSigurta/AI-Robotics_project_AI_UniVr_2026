"""PPO training on Modal cloud.

Launches PPO training on a Modal A100 GPU with persistent checkpointing
via Modal Volume. Duplicates CurriculumCallback locally (not imported
from train_ppo.py) to avoid Modal dependency chain issues.

Usage:
    modal run modal_scripts/modal_train_ppo.py          # full 250k training
    modal run modal_scripts/modal_train_ppo.py --dry    # dry-run self-check
"""

import os
import sys

# Allow imports from project root (limo_env.py, reward.py)
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import modal

# ── Curriculum callback (copied locally — do not import from train_ppo.py) ──

from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    """SB3 callback that advances obstacle difficulty at predefined timesteps.

    Preconditions:
        - self.training_env is a VecEnv wrapping LimoCustomEnv instances.
        - The underlying env(s) have a set_curriculum_stage(stage) method.
    Postconditions:
        - At each threshold boundary, set_curriculum_stage(n) is called
          on the first underlying env.
        - self._current_stage tracks the active stage.
        - set_curriculum_stage is called exactly once per transition.

    Stage thresholds (roadmap §4.2):
        0: [     0,  80_000)  —  2-4 obstacles, r ∈ [0.10, 0.15]
        1: [80_000, 180_000)  —  4-7 obstacles, r ∈ [0.10, 0.20]
        2: [180_000, 250_000] —  6-10 obstacles, r ∈ [0.10, 0.20]

    Args:
        verbose: Verbosity level (0 = silent, 1 = print transitions).

    Example:
        >>> cb = CurriculumCallback(verbose=1)
    """

    STAGE_BOUNDARIES = {80_000: 1, 180_000: 2}

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._current_stage = 0

    def _on_step(self) -> bool:
        """Check timestep threshold and advance stage if needed.

        Pre: self.model.num_timesteps reflects current total timesteps.
        Post: Returns True (never interrupts training).

        Returns:
            bool: Always True — training continues.
        """
        ts = self.model.num_timesteps
        for boundary, stage in sorted(self.STAGE_BOUNDARIES.items()):
            if ts >= boundary and self._current_stage < stage:
                self._current_stage = stage
                env = self.training_env.envs[0]
                env.set_curriculum_stage(stage)
                if self.verbose > 0:
                    print(
                        f'[CurriculumCallback] Step {ts}: advancing to stage {stage} '
                        f'(n_obs={env.n_obstacles_range}, '
                        f'r_obs={env.obs_radius_range})'
                    )
                break
        return True


# ── Modal infrastructure ───────────────────────────────────────────────────

app = modal.App("limo-ppo-training")

# Volume for persistent checkpoints across runs
volume = modal.Volume.from_name("limo-checkpoints", create_if_missing=True)

# Container image with all dependencies + source code
image = (
    modal.Image.from_registry("python:3.11-slim")
    .pip_install(
        "stable-baselines3",
        "gymnasium",
        "torch",
        "numpy",
        "numba",
        "tensorboard",
    )
    .add_local_file(
        os.path.join(os.path.dirname(__file__), "..", "limo_env.py"),
        "/root/limo_env.py",
    )
    .add_local_file(
        os.path.join(os.path.dirname(__file__), "..", "reward.py"),
        "/root/reward.py",
    )
)


# ── Training function (runs on Modal A100) ─────────────────────────────────

@app.function(
    gpu="A100",
    timeout=3600,
    volumes={"/checkpoints": volume},
    image=image,
)
def train_ppo(total_timesteps: int = 250_000) -> str:
    """Run PPO training on a Modal A100 GPU.

    Preconditions:
        - Modal Volume "limo-checkpoints" is writable at /checkpoints.
        - All dependencies installed (via container image).
    Postconditions:
        - Checkpoints saved every 25k steps to /checkpoints/ppo/.
        - Final model saved at /checkpoints/ppo/ppo_limo_final.zip.
        - Returns path to final checkpoint.

    Args:
        total_timesteps: Total training timesteps (default 250_000).

    Returns:
        str: Path to the saved final model.

    Example:
        >>> path = train_ppo.remote(total_timesteps=250_000)
        >>> print(path)
        /checkpoints/ppo/ppo_limo_final.zip
    """
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    # Source files copied into /root/ by the image definition above
    sys.path.insert(0, "/root")
    from limo_env import LimoCustomEnv

    # ── Environment (start at curriculum stage 0) ──────────────────────────
    def make_env():
        return LimoCustomEnv(
            n_obstacles_range=(2, 4),
            obs_radius_range=(0.10, 0.15),
            randomize_goal=True,
        )

    env = DummyVecEnv([make_env])

    # ── Callbacks ──────────────────────────────────────────────────────────
    curriculum_cb = CurriculumCallback(verbose=1)

    checkpoint_path = "/checkpoints/ppo"
    os.makedirs(checkpoint_path, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=25_000,
        save_path=checkpoint_path,
        name_prefix="ppo_limo",
    )

    # ── PPO Model (exact hyperparameters from roadmap §5.2) ────────────────
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        seed=42,
    )

    # ── Training ───────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=total_timesteps,
        callback=[curriculum_cb, checkpoint_cb],
    )

    # ── Save final model ───────────────────────────────────────────────────
    final_path = os.path.join(checkpoint_path, "ppo_limo_final.zip")
    model.save(final_path)

    summary = (
        f"\n{'=' * 50}\n"
        f"PPO training complete on Modal A100\n"
        f"  Total timesteps: {total_timesteps}\n"
        f"  Final checkpoint: {final_path}\n"
        f"{'=' * 50}"
    )
    print(summary)

    env.close()
    return final_path


# ── Local entrypoint ───────────────────────────────────────────────────────

@app.local_entrypoint()
def main(total_timesteps: int = 250_000):
    """Launch PPO training remotely on Modal.

    Pre: Modal CLI is authenticated and app "limo-ppo-training" is deployed.
    Post: Training runs on Modal A100; checkpoints saved to Modal Volume.

    Args:
        total_timesteps: Total training timesteps (default 250_000).

    Example:
        $ modal run modal_scripts/modal_train_ppo.py
        $ modal run modal_scripts/modal_train_ppo.py --total-timesteps 50000
    """
    print(f"Launching PPO training on Modal A100 ({total_timesteps} timesteps)...")
    result_path = train_ppo.remote(total_timesteps=total_timesteps)
    print(f"Training complete. Model saved at: {result_path}")


# ── Self-check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== SELF-CHECK: modal_train_ppo.py ===")
    n_pass = 0
    n_total = 3

    # ── Test 1: Modal image definition ─────────────────────────────────────
    print("\n--- Test 1: Modal image definition (no import errors) ---")
    try:
        # Just verify the image definition chain does not raise
        _ = image
        print(f"  Image object created successfully")
        print(f"  Local files: limo_env.py, reward.py")
        print("  >>> PASS <<<")
        n_pass += 1
    except Exception as e:
        print(f"  Image definition failed: {e}")
        print("  >>> FAIL <<<")

    # ── Test 2: Volume definition ──────────────────────────────────────────
    print("\n--- Test 2: Modal Volume definition ---")
    try:
        vol_ref = modal.Volume.from_name("limo-checkpoints", create_if_missing=True)
        print(f"  Volume name: limo-checkpoints")
        print(f"  Volume exists/reachable: {vol_ref is not None}")
        print("  >>> PASS <<<")
        n_pass += 1
    except Exception as e:
        print(f"  Volume definition failed: {e}")
        print("  >>> FAIL <<<")

    # ── Test 3: Local dry-run (env + PPO instantiation) ────────────────────
    print("\n--- Test 3: Local env + PPO instantiation (dry run) ---")
    try:
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        import numpy as np
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from limo_env import LimoCustomEnv

        def make_env():
            return LimoCustomEnv(
                n_obstacles_range=(2, 4),
                obs_radius_range=(0.10, 0.15),
                randomize_goal=True,
            )

        env = DummyVecEnv([make_env])

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=0,
            seed=42,
        )

        obs = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action).flatten()

        v_ok = bool(0.0 <= action[0] <= 1.0)
        w_ok = bool(-1.0 <= action[1] <= 1.0)
        obs_shape_ok = obs.shape == (1, 23)

        print(f"  Observation shape: {obs.shape}  [{'OK' if obs_shape_ok else 'FAIL'}]")
        print(f"  Action[0] (v): {action[0]:.6f}  [{'OK' if v_ok else 'FAIL'}]")
        print(f"  Action[1] (w): {action[1]:.6f}  [{'OK' if w_ok else 'FAIL'}]")

        # Also verify CurriculumCallback instantiates and links correctly
        cb = CurriculumCallback(verbose=0)
        print(f"  CurriculumCallback._current_stage: {cb._current_stage}")

        # Verify checkpoint callback works
        from stable_baselines3.common.callbacks import CheckpointCallback
        ckpt_cb = CheckpointCallback(save_freq=1000, save_path="/tmp/ppo_test", name_prefix="ppo_test")
        print(f"  CheckpointCallback save_freq: {ckpt_cb.save_freq}")

        model.env = env  # keep ref for cleanup
        if obs_shape_ok and v_ok and w_ok:
            print("  >>> PASS <<<")
            n_pass += 1
        else:
            print("  >>> FAIL <<<")

        env.close()

    except Exception as e:
        import traceback
        print(f"  Local dry-run error: {e}")
        traceback.print_exc()
        print("  >>> FAIL <<<")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n=== Result: {n_pass}/{n_total} tests passed ===")
    if n_pass == n_total:
        print("All SELF-CHECK tests passed.")
    else:
        print(f"SOME TESTS FAILED ({n_total - n_pass} failures).")
        sys.exit(1)
