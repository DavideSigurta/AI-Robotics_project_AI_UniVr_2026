"""PPO training script for the LIMO environment.

Trains a PPO agent (Stable-Baselines3) on LimoCustomEnv with curriculum
learning. Supports checkpointing, TensorBoard logging, and arg overrides.

Usage:
    python train_ppo.py                          # default: 250k steps
    python train_ppo.py --timesteps 50000        # quick test
    python train_ppo.py --seed 0 --tb-log ./tb   # custom
"""

import argparse
import os
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from limo_env import LimoCustomEnv


# ── Curriculum callback ────────────────────────────────────────────────────

class CurriculumCallback(BaseCallback):
    """SB3 callback that advances obstacle difficulty at predefined timesteps.

    Preconditions:
        - self.training_env is a VecEnv wrapping LimoCustomEnv instances.
        - The underlying env(s) have a set_curriculum_stage(stage) method.
    Postconditions:
        - At each threshold boundary, set_curriculum_stage(n) is called
          on the first underlying env (all envs in DummyVecEnv share params).
        - self._current_stage tracks the active stage.
        - set_curriculum_stage is called exactly once per transition.

    Stage thresholds (roadmap §4.2):
        0: [     0,  80_000)  —  2-4 obstacles, r ∈ [0.10, 0.15]
        1: [80_000, 180_000)  —  4-7 obstacles, r ∈ [0.10, 0.20]
        2: [180_000, 250_000] —  6-10 obstacles, r ∈ [0.10, 0.20]

    Args:
        verbose: Verbosity level (0 = silent, 1 = print transitions).

    Example:
        >>> callback = CurriculumCallback(verbose=1)
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
        # Use self.model.num_timesteps (live read from model) rather than
        # self.num_timesteps (stale copy set once during init_callback).
        ts = self.model.num_timesteps
        for boundary, stage in sorted(self.STAGE_BOUNDARIES.items()):
            if ts >= boundary and self._current_stage < stage:
                self._current_stage = stage
                # Access underlying env through VecEnv wrapper
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


# ── Argument parsing ───────────────────────────────────────────────────────

def parse_args(argv=None):
    """Parse command-line arguments for training configuration.

    Pre: None.
    Post: Returns namespace with all fields set to CLI values or defaults.

    Args:
        argv: Optional list of strings (for testing). Defaults to sys.argv[1:].

    Returns:
        argparse.Namespace with fields: timesteps, save_freq, checkpoint_dir,
        tb_log, seed.

    Example:
        >>> args = parse_args(['--timesteps', '50000'])
        >>> args.timesteps
        50000
    """
    parser = argparse.ArgumentParser(
        description='Train PPO agent on LimoCustomEnv with curriculum learning.'
    )
    parser.add_argument(
        '--timesteps', type=int, default=250_000,
        help='Total training timesteps (default: 250000).'
    )
    parser.add_argument(
        '--save-freq', type=int, default=25_000,
        help='Checkpoint frequency in timesteps (default: 25000).'
    )
    parser.add_argument(
        '--checkpoint-dir', type=str, default='checkpoints_ppo',
        help='Directory for checkpoints (default: checkpoints_ppo).'
    )
    parser.add_argument(
        '--tb-log', type=str, default='tb_ppo',
        help='TensorBoard log directory (default: tb_ppo).'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42).'
    )
    return parser.parse_args(argv)


# ── Main training function ─────────────────────────────────────────────────

def main():
    """Run PPO training on LimoCustomEnv with curriculum learning.

    Preconditions:
        - All required packages are installed (sb3, gymnasium, numpy).
        - Checkpoint directory does not conflict with existing files.
    Postconditions:
        - Checkpoints saved at `save_freq` intervals in `checkpoint_dir`.
        - Final model saved as `{checkpoint_dir}/ppo_limo_final.zip`.
        - TensorBoard logs written to `tb_log`.

    Pipeline:
        1. Create LimoCustomEnv wrapped in DummyVecEnv.
        2. Instantiate PPO with roadmap §5.2 hyperparameters.
        3. Train with CurriculumCallback + CheckpointCallback.
        4. Save final model.

    Example:
        >>> main()  # runs with default args (250k steps)
    """
    args = parse_args()

    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.tb_log, exist_ok=True)

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

    checkpoint_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.checkpoint_dir,
        name_prefix='ppo_limo',
    )

    # ── PPO Model (roadmap §5.2 — exact hyperparameters) ───────────────────
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=args.tb_log,
        seed=args.seed,
    )

    # ── Training ───────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=args.timesteps,
        callback=[curriculum_cb, checkpoint_cb],
    )

    # ── Save final model ───────────────────────────────────────────────────
    final_path = os.path.join(args.checkpoint_dir, 'ppo_limo_final.zip')
    model.save(final_path)
    print(f'Final model saved to {final_path}')

    env.close()


# ── Self-check ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== SELF-CHECK: train_ppo.py ===')
    n_pass = 0
    n_total = 4

    # ── Test 1: env passes SB3 env checker ─────────────────────────────────
    print('\n--- Test 1: SB3 env_checker compatibility ---')
    try:
        test_env = LimoCustomEnv(randomize_goal=True)
        check_env(test_env)
        print('  check_env(LimoCustomEnv) passed without errors.')
        print('  >>> PASS <<<')
        n_pass += 1
    except Exception as e:
        print(f'  check_env failed: {e}')
        print('  >>> FAIL <<<')

    # ── Test 2: PPO model instantiation ────────────────────────────────────
    print('\n--- Test 2: PPO model instantiation ---')
    try:
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='stable_baselines3')
        inst_env = DummyVecEnv([lambda: LimoCustomEnv(randomize_goal=True)])
        test_model = PPO(
            'MlpPolicy',
            inst_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=0,
            seed=42,
        )
        print(f'  PPO policy type: {type(test_model.policy).__name__}')
        print(f'  PPO observation space: {test_model.observation_space}')
        print(f'  PPO action space: {test_model.action_space}')
        inst_env.close()
        print('  >>> PASS <<<')
        n_pass += 1
    except Exception as e:
        print(f'  PPO instantiation failed: {e}')
        print('  >>> FAIL <<<')

    # ── Test 3: CurriculumCallback stage transitions ───────────────────────
    print('\n--- Test 3: CurriculumCallback stage transitions ---')
    try:
        cb_env = LimoCustomEnv(randomize_goal=True)
        cb_vec_env = DummyVecEnv([lambda: cb_env])

        class _MockModel:
            """Minimal model mock for SB3 callback init.

            Provides get_env() so init_callback sets training_env correctly.
            """
            def __init__(self, vec_env):
                self.num_timesteps = 0
                self._vec_env = vec_env

            def get_env(self):
                return self._vec_env

        mock = _MockModel(cb_vec_env)
        callback = CurriculumCallback(verbose=0)

        # init_callback sets self.model, self.locals, self.globals, training_env
        callback.init_callback(mock)

        def _sim_step(ts):
            callback.model.num_timesteps = ts
            return callback._on_step()

        results = {}

        # Stage 0 — stays 0 below 80k
        results['initial_stage'] = cb_env.curriculum_stage
        _sim_step(50_000)
        results['after_50000'] = cb_env.curriculum_stage

        # Stage 0 → 1
        _sim_step(80_001)
        results['after_80001'] = cb_env.curriculum_stage
        results['n_range_1'] = cb_env.n_obstacles_range
        results['r_range_1'] = cb_env.obs_radius_range

        # Stage 1 → 2
        _sim_step(180_001)
        results['after_180001'] = cb_env.curriculum_stage
        results['n_range_2'] = cb_env.n_obstacles_range
        results['r_range_2'] = cb_env.obs_radius_range

        expected = {
            'initial_stage': 0,
            'after_50000': 0,
            'after_80001': 1,
            'n_range_1': (4, 7),
            'r_range_1': (0.10, 0.20),
            'after_180001': 2,
            'n_range_2': (6, 10),
            'r_range_2': (0.10, 0.20),
        }

        all_ok = True
        for key, ev in expected.items():
            got = results[key]
            ok = got == ev
            all_ok = all_ok and ok
            print(f'  {key}: expected {ev}, got {got}  [{"PASS" if ok else "FAIL"}]')

        if all_ok:
            print('  >>> PASS <<<')
            n_pass += 1
        else:
            print('  >>> FAIL <<<')

        cb_vec_env.close()

    except Exception as e:
        import traceback
        print(f'  Curriculum callback test error: {e}')
        traceback.print_exc()
        print('  >>> FAIL <<<')

    # ── Test 4: 1000-step training smoke test ──────────────────────────────
    print('\n--- Test 4: 1000-step training produces valid actions ---')
    try:
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='stable_baselines3')

        smoke_env = DummyVecEnv([lambda: LimoCustomEnv(randomize_goal=True)])
        smoke_model = PPO(
            'MlpPolicy',
            smoke_env,
            learning_rate=3e-4,
            n_steps=512,       # reduced for fast smoke test
            batch_size=32,     # reduced for fast smoke test
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=0,
            seed=42,
            device='cpu',
        )

        smoke_model.learn(total_timesteps=1000, progress_bar=False)

        # Verify model produces valid actions for a sample observation
        sample_obs = np.zeros((1, 23), dtype=np.float32)
        action, _ = smoke_model.predict(sample_obs, deterministic=True)
        action = np.asarray(action).flatten()

        v_ok = bool(0.0 <= action[0] <= 1.0)
        w_ok = bool(-1.0 <= action[1] <= 1.0)
        shape_ok = action.shape == (2,)

        print(f'  Action shape: {action.shape}  [{"OK" if shape_ok else "FAIL"}]')
        print(f'  Action[0] (v): {action[0]:.6f}  range [0,1]  [{"OK" if v_ok else "FAIL"}]')
        print(f'  Action[1] (w): {action[1]:.6f}  range [-1,1] [{"OK" if w_ok else "FAIL"}]')

        if shape_ok and v_ok and w_ok:
            print('  >>> PASS <<<')
            n_pass += 1
        else:
            print('  >>> FAIL <<<')

        smoke_env.close()

    except Exception as e:
        import traceback
        print(f'  Training smoke test error: {e}')
        traceback.print_exc()
        print('  >>> FAIL <<<')

    # ── Summary ────────────────────────────────────────────────────────────
    print(f'\n=== Result: {n_pass}/{n_total} tests passed ===')
    if n_pass == n_total:
        print('All SELF-CHECK tests passed.')
    else:
        print(f'SOME TESTS FAILED ({n_total - n_pass} failures).')
        sys.exit(1)
