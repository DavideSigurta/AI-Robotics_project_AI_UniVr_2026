"""Evaluate a trained PPO checkpoint on LimoCustomEnv.

Loads a PPO model and runs N evaluation episodes to report goal rate,
crash rate, timeout rate, mean episode length, and mean cumulative
reward. No Modal, no CoppeliaSim — pure local eval on custom env.

Usage:
    python evaluate_ppo.py --checkpoint checkpoints_ppo/ppo_limo_final.zip
    python evaluate_ppo.py --checkpoint model.zip --episodes 50 --seed 0
    python evaluate_ppo.py --self-check                    # run tests only
"""

import argparse
import os
import sys
from typing import Any, Dict, List

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from limo_env import LimoCustomEnv
from reward import compute_reward


# ── Evaluation function ────────────────────────────────────────────────────

def evaluate(
    checkpoint_path: str,
    n_episodes: int = 100,
    curriculum_stage: int = 2,
    seed: int = 123,
    deterministic: bool = True,
    verbose_actions: bool = False,
) -> Dict[str, Any]:
    """Evaluate a trained PPO checkpoint over N episodes.

    Preconditions:
        - checkpoint_path points to a valid PPO .zip saved by SB3.
        - curriculum_stage in {0, 1, 2}.
        - Training seed was 42 (eval seed default 123 differs to test
          unseen configs).
    Postconditions:
        - Returns dict with aggregate metrics.
        - No files created or modified.
        - No side effects on global state.

    Args:
        checkpoint_path: Path to SB3 PPO .zip checkpoint.
        n_episodes: Number of evaluation episodes.
        curriculum_stage: Obstacle difficulty stage (0=easy, 2=hard).
        seed: Base RNG seed (episode i uses seed + i).
        deterministic: If True, model.predict uses deterministic=True.
        verbose_actions: If True, print actions at steps 0,50,100,200,400
                         for the first 3 episodes.

    Returns:
        dict with keys:
            goal_rate, crash_rate, timeout_rate (float in [0,1])
            mean_episode_length, std_episode_length (float)
            mean_reward, std_reward (float)
            n_episodes (int)
            results (list of dicts per episode)

    Example:
        >>> stats = evaluate('checkpoints_ppo/ppo_limo_final.zip',
        ...                  n_episodes=10, curriculum_stage=0)
        >>> stats['goal_rate']
        0.3
    """
    # Load model
    model = PPO.load(checkpoint_path, device='cpu')

    # Create env at requested curriculum stage
    env = LimoCustomEnv(randomize_goal=True)
    env.set_curriculum_stage(curriculum_stage)

    results: List[Dict[str, Any]] = []

    for ep in range(n_episodes):
        ep_seed = seed + ep
        obs, _ = env.reset(seed=ep_seed)
        done = False
        ep_reward = 0.0
        ep_length = 0
        outcome = 'timeout'

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            action = np.asarray(action).flatten()

            if verbose_actions and ep < 3 and ep_length in {0, 50, 100, 200, 400}:
                print(f'ep={ep} step={ep_length} '
                      f'action=[v={action[0]:.4f}, w={action[1]:.4f}]')

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            done = terminated or truncated

        # Determine outcome from info dict
        if info.get('goal_reached', False):
            outcome = 'goal'
        elif info.get('crashed', False):
            outcome = 'crash'
        else:
            outcome = 'timeout'

        results.append({
            'episode': ep,
            'outcome': outcome,
            'length': ep_length,
            'reward': ep_reward,
            'seed': ep_seed,
        })

    env.close()

    # Aggregate
    n_goal = sum(1 for r in results if r['outcome'] == 'goal')
    n_crash = sum(1 for r in results if r['outcome'] == 'crash')
    n_timeout = sum(1 for r in results if r['outcome'] == 'timeout')
    lengths = [r['length'] for r in results]
    rewards = [r['reward'] for r in results]

    stats = {
        'goal_rate': n_goal / n_episodes,
        'crash_rate': n_crash / n_episodes,
        'timeout_rate': n_timeout / n_episodes,
        'mean_episode_length': float(np.mean(lengths)),
        'std_episode_length': float(np.std(lengths)),
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'n_episodes': n_episodes,
        'results': results,
    }

    return stats


# ── Print helper ───────────────────────────────────────────────────────────

def print_summary(stats: Dict[str, Any]) -> None:
    """Print a clean summary table of evaluation metrics.

    Pre: stats dict from evaluate().
    Post: Metrics printed to stdout.

    Args:
        stats: Output dict from evaluate().

    Example:
        >>> st = evaluate('model.zip', n_episodes=5)
        >>> print_summary(st)
    """
    sep = '-' * 48
    print(sep)
    print(f'  Evaluation: {stats["n_episodes"]} episodes')
    print(sep)
    print(f'  Goal rate:          {stats["goal_rate"]:.2%}')
    print(f'  Crash rate:         {stats["crash_rate"]:.2%}')
    print(f'  Timeout rate:       {stats["timeout_rate"]:.2%}')
    print(f'  Sum (should be 1):  '
          f'{stats["goal_rate"] + stats["crash_rate"] + stats["timeout_rate"]:.2f}')
    print(sep)
    print(f'  Mean episode len:   {stats["mean_episode_length"]:.1f}')
    print(f'  Std episode len:    {stats["std_episode_length"]:.1f}')
    print(f'  Mean reward:        {stats["mean_reward"]:.3f}')
    print(f'  Std reward:         {stats["std_reward"]:.3f}')
    print(sep)


# ── Argument parsing ───────────────────────────────────────────────────────

def parse_args(argv=None):
    """Parse CLI args for evaluation.

    Pre: None.
    Post: Returns namespace with all fields set.

    Args:
        argv: Optional list of strings (for testing).

    Returns:
        argparse.Namespace.

    Example:
        >>> a = parse_args(['--checkpoint', 'model.zip', '--episodes', '10'])
        >>> a.checkpoint
        'model.zip'
    """
    parser = argparse.ArgumentParser(
        description='Evaluate trained PPO on LimoCustomEnv.'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to PPO .zip checkpoint (required unless --self-check).'
    )
    parser.add_argument(
        '--self-check', action='store_true',
        help='Run self-check tests instead of evaluation (default: False).'
    )
    parser.add_argument(
        '--verbose-actions', action='store_true',
        help='Print actions at steps 0,50,100,200,400 for first 3 episodes.'
    )
    parser.add_argument(
        '--episodes', type=int, default=100,
        help='Number of evaluation episodes (default: 100).'
    )
    parser.add_argument(
        '--curriculum-stage', type=int, default=2, choices=[0, 1, 2],
        help='Curriculum difficulty stage (default: 2).'
    )
    parser.add_argument(
        '--seed', type=int, default=123,
        help='Base RNG seed for evaluation (default: 123).'
    )
    parser.add_argument(
        '--deterministic', action='store_true', default=True,
        help='Use deterministic actions (default: True).'
    )
    parser.add_argument(
        '--no-deterministic', action='store_false', dest='deterministic',
        help='Use stochastic actions.'
    )
    return parser.parse_args(argv)


# ── Main entry point ───────────────────────────────────────────────────────

def main():
    """Run evaluation from CLI args.

    Pre: --checkpoint path exists and is a valid PPO model.
    Post: Summary table printed; program exits with 0.
    """
    args = parse_args()

    if args.checkpoint is None:
        print('Error: --checkpoint is required for evaluation. '
              'Use --self-check to run tests.')
        sys.exit(1)

    if not os.path.isfile(args.checkpoint):
        print(f'Error: checkpoint not found: {args.checkpoint}')
        sys.exit(1)

    stats = evaluate(
        checkpoint_path=args.checkpoint,
        n_episodes=args.episodes,
        curriculum_stage=args.curriculum_stage,
        seed=args.seed,
        deterministic=args.deterministic,
        verbose_actions=args.verbose_actions,
    )
    print_summary(stats)


# ── Self-check ─────────────────────────────────────────────────────────────

def run_self_check() -> int:
    """Run all self-check tests for the evaluation module.

    Pre: None.
    Post: Prints PASS/FAIL for each test. Returns number of passed tests.

    Tests:
        1. Argparse correctly parses all flags.
        2. evaluate() with freshly trained tiny model (10 ep, stage 0)
           returns dict with all expected keys, correct types, valid ranges.
        2b. Rates (goal + crash + timeout) sum to 1.0.

    Returns:
        int: Number of passed tests (0-3).

    Example:
        >>> n = run_self_check()
        >>> n == 3
        True
    """
    print('=== SELF-CHECK: evaluate_ppo.py ===')
    n_pass = 0
    n_total = 3

    # ── Test 1: argparse parsing ───────────────────────────────────────────
    print('\n--- Test 1: argparse argument parsing ---')
    try:
        args = parse_args([
            '--checkpoint', 'dummy.zip',
            '--episodes', '50',
            '--curriculum-stage', '1',
            '--seed', '999',
            '--no-deterministic',
        ])
        checks = [
            ('checkpoint', args.checkpoint == 'dummy.zip'),
            ('episodes', args.episodes == 50),
            ('curriculum_stage', args.curriculum_stage == 1),
            ('seed', args.seed == 999),
            ('deterministic', args.deterministic is False),
        ]
        all_ok = True
        for name, ok in checks:
            all_ok = all_ok and ok
            print(f'  {name}: {"OK" if ok else "FAIL"}')
        if all_ok:
            print('  >>> PASS <<<')
            n_pass += 1
        else:
            print('  >>> FAIL <<<')
    except Exception as e:
        print(f'  Argparse test error: {e}')
        print('  >>> FAIL <<<')

    # ── Test 2: evaluate() fresh tiny model, 10 episodes, stage 0 ──────────
    print('\n--- Test 2: evaluate() with freshly trained tiny model ---')
    try:
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)

        train_env_mon = DummyVecEnv([
            lambda: Monitor(LimoCustomEnv(randomize_goal=True))
        ])
        tiny_model = PPO(
            'MlpPolicy', train_env_mon,
            learning_rate=3e-4, n_steps=512, batch_size=32,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            verbose=0, seed=42, device='cpu',
        )
        tiny_model.learn(total_timesteps=2000, progress_bar=False)
        train_env_mon.close()

        tmp_path = '/tmp/ppo_eval_test.zip'
        tiny_model.save(tmp_path)

        stats = evaluate(
            checkpoint_path=tmp_path,
            n_episodes=10,
            curriculum_stage=0,
            seed=200,
            deterministic=True,
        )

        expected_keys = [
            'goal_rate', 'crash_rate', 'timeout_rate',
            'mean_episode_length', 'std_episode_length',
            'mean_reward', 'std_reward', 'n_episodes', 'results',
        ]
        key_ok = all(k in stats for k in expected_keys)
        print(f'  All expected keys present: {key_ok}')

        type_ok = (
            isinstance(stats['goal_rate'], float)
            and isinstance(stats['crash_rate'], float)
            and isinstance(stats['timeout_rate'], float)
            and isinstance(stats['mean_episode_length'], float)
            and isinstance(stats['n_episodes'], int)
        )
        print(f'  Value types correct: {type_ok}')

        n_ok = stats['n_episodes'] == 10
        print(f'  n_episodes == 10: {n_ok}')

        len_ok = len(stats['results']) == 10
        print(f'  results list length 10: {len_ok}')

        rates = [stats['goal_rate'], stats['crash_rate'], stats['timeout_rate']]
        range_ok = all(0.0 <= r <= 1.0 for r in rates)
        print(f'  All rates in [0,1]: {range_ok}')

        len_pos = stats['mean_episode_length'] > 0
        print(f'  mean_episode_length > 0: {len_pos}')

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        if key_ok and type_ok and n_ok and len_ok and range_ok and len_pos:
            print('  >>> PASS <<<')
            n_pass += 1
        else:
            print('  >>> FAIL <<<')

        # ── Test 2b: rates sum to 1.0 ─────────────────────────────────────
        print('\n--- Test 2b: rates sum to 1.0 ---')
        rate_sum = stats['goal_rate'] + stats['crash_rate'] + stats['timeout_rate']
        sum_ok = abs(rate_sum - 1.0) < 1e-9
        print(f'  goal_rate={stats["goal_rate"]:.4f}, '
              f'crash_rate={stats["crash_rate"]:.4f}, '
              f'timeout_rate={stats["timeout_rate"]:.4f}')
        print(f'  sum={rate_sum:.10f}  [{"OK" if sum_ok else "FAIL"}]')
        if sum_ok:
            print('  >>> PASS <<<')
            n_pass += 1
        else:
            print('  >>> FAIL <<<')

    except Exception as e:
        import traceback
        print(f'  evaluate() test error: {e}')
        traceback.print_exc()
        print('  >>> FAIL <<<')

    # ── Summary ────────────────────────────────────────────────────────────
    print(f'\n=== Result: {n_pass}/{n_total} tests passed ===')
    if n_pass == n_total:
        print('All SELF-CHECK tests passed.')
    else:
        print(f'SOME TESTS FAILED ({n_total - n_pass} failures).')

    return n_pass


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()
    if args.self_check:
        n = run_self_check()
        sys.exit(0 if n == 3 else 1)
    else:
        main()
