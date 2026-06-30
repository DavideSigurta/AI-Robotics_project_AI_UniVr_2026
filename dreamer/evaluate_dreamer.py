"""Evaluate a trained DreamerV3 checkpoint on LimoCustomEnv.

Loads a DreamerV3 checkpoint (.pth saved by Dreamer.saveCheckpoint)
and runs N evaluation episodes using the dreamer_inference_step loop
(roadmap §6.5): posterior update per step, actor in deterministic mode.

No Modal, no CoppeliaSim — pure local eval on custom env.

Usage:
    python evaluate_dreamer.py --checkpoint results/checkpoints_dreamer/LimoCustomEnv_final_60k.pth
    python evaluate_dreamer.py --checkpoint model.pth --episodes 50 --seed 0
    python evaluate_dreamer.py --self-check                    # run tests only
"""

import argparse
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch

# Allow imports from dreamer/ modules
_dreamer_root = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_dreamer_root, ".."))
if _dreamer_root not in sys.path:
    sys.path.insert(0, _dreamer_root)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dreamer import Dreamer
from limo_env import LimoCustomEnv
from utils import loadConfig
from reward import compute_reward


# ── Inference step (roadmap §6.5) ──────────────────────────────────────────

@torch.no_grad()
def dreamer_inference_step(
    dreamer: Dreamer,
    obs_numpy: np.ndarray,
    recurrent_state: torch.Tensor,
    latent_state: torch.Tensor,
    action: torch.Tensor,
) -> tuple:
    """Single inference step using posterior update loop (roadmap §6.5).

    Unlike training (where posterior is used for RSSM learning), in inference
    the agent has access to the real observation at every step — so it updates
    the latent state with the posterior (encoding the real observation),
    then queries the actor for the next action.

    Preconditions:
        - dreamer is in eval mode (no training).
        - obs_numpy is shape (23,) float32 array (LimoCustomEnv observation).
        - recurrent_state shape (1, dreamer.recurrentSize).
        - latent_state shape (1, dreamer.latentSize).
        - action shape (1, dreamer.actionSize).
    Postconditions:
        - Returns (action_out, new_recurrent, new_latent).
        - All tensors on dreamer.device.

    Args:
        dreamer: DreamerV3 instance (loaded from checkpoint).
        obs_numpy: Observation vector, shape (23,).
        recurrent_state: Previous recurrent state (1, recurrentSize).
        latent_state: Previous latent state (1, latentSize).
        action: Previous action taken (1, actionSize).

    Returns:
        Tuple of (action_out, new_recurrent, new_latent).

    Example:
        >>> action_out, rec, lat = dreamer_inference_step(
        ...     dreamer, obs, rec, lat, act)
        >>> action_out.shape
        torch.Size([1, 2])
    """
    obs_tensor = torch.from_numpy(obs_numpy).float().unsqueeze(0).to(dreamer.device)
    encoded = dreamer.encoder(obs_tensor)

    # Update recurrent state with previous action
    new_recurrent = dreamer.recurrentModel(recurrent_state, latent_state, action)

    # Update latent state with real observation (posterior, not prior)
    new_latent, _ = dreamer.posteriorNet(
        torch.cat((new_recurrent, encoded.view(1, -1)), -1)
    )

    full_state = torch.cat((new_recurrent, new_latent), -1)
    action_out = dreamer.actor(full_state)  # deterministic (no training=True)

    return action_out, new_recurrent, new_latent


# ── Evaluation function ────────────────────────────────────────────────────

def evaluate(
    checkpoint_path: str,
    n_episodes: int = 100,
    curriculum_stage: int = 2,
    seed: int = 123,
    verbose_actions: bool = False,
) -> Dict[str, Any]:
    """Evaluate a trained DreamerV3 checkpoint over N episodes.

    Preconditions:
        - checkpoint_path points to a valid .pth saved by Dreamer.saveCheckpoint.
        - config 'limo-dreamer.yml' is findable by loadConfig (cwd or subdir).
        - curriculum_stage in {0, 1, 2}.
    Postconditions:
        - Returns dict with aggregate metrics.
        - No files created or modified.
        - Dreamer stays on CPU (or GPU if available).

    Args:
        checkpoint_path: Path to Dreamer .pth checkpoint.
        n_episodes: Number of evaluation episodes.
        curriculum_stage: Obstacle difficulty stage (0=easy, 2=hard).
        seed: Base RNG seed (episode i uses seed + i).
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
        >>> stats = evaluate('results/checkpoints_dreamer/LimoCustomEnv_final_60k.pth',
        ...                  n_episodes=10, curriculum_stage=0)
        >>> stats['goal_rate']
        0.3
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config and create Dreamer
    config = loadConfig("limo-dreamer.yml")
    env_raw = LimoCustomEnv(randomize_goal=True)
    obs_shape = env_raw.observation_space.shape       # (23,)
    action_size = env_raw.action_space.shape[0]        # 2
    action_low = env_raw.action_space.low.tolist()     # [0.0, -1.0]
    action_high = env_raw.action_space.high.tolist()   # [1.0, 1.0]

    dreamer = Dreamer(obs_shape, action_size, action_low, action_high,
                      device, config.dreamer)

    # Load checkpoint weights
    dreamer.loadCheckpoint(checkpoint_path)
    dreamer.actor.eval()
    dreamer.encoder.eval()
    dreamer.decoder.eval()
    dreamer.recurrentModel.eval()
    dreamer.posteriorNet.eval()
    print(f"Loaded Dreamer from {checkpoint_path} (device={device})")

    # Create eval env at requested curriculum stage
    env_raw_eval = LimoCustomEnv(randomize_goal=True)
    env_raw_eval.set_curriculum_stage(curriculum_stage)
    # Use raw env directly to access info dict for outcome classification.
    # We compute done = terminated or truncated ourselves to match the
    # contract expected by the inference loop.
    env = env_raw_eval

    results: List[Dict[str, Any]] = []

    for ep in range(n_episodes):
        ep_seed = seed + ep
        obs, _ = env.reset(seed=ep_seed)

        # Initialize states (roadmap §6.5)
        recurrent_state = torch.zeros(1, dreamer.recurrentSize, device=device)
        latent_state = torch.zeros(1, dreamer.latentSize, device=device)
        action_tensor = torch.zeros(1, action_size, device=device)

        done = False
        ep_reward = 0.0
        ep_length = 0
        outcome = 'timeout'

        while not done:
            # Dreamer inference step
            action_tensor, recurrent_state, latent_state = dreamer_inference_step(
                dreamer, obs, recurrent_state, latent_state, action_tensor
            )
            action_np = action_tensor.cpu().numpy().reshape(-1)

            if verbose_actions and ep < 3 and ep_length in {0, 50, 100, 200, 400}:
                print(f'ep={ep} step={ep_length} '
                      f'action=[v={action_np[0]:.4f}, w={action_np[1]:.4f}]')

            # Step raw env — (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = bool(terminated or truncated)
            ep_reward += reward
            ep_length += 1

        # Determine outcome from info dict (true on last step)
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
        >>> st = evaluate('model.pth', n_episodes=5)
        >>> print_summary(st)
    """
    sep = '-' * 48
    print(sep)
    print(f'  Dreamer Evaluation: {stats["n_episodes"]} episodes')
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
        >>> a = parse_args(['--checkpoint', 'model.pth', '--episodes', '10'])
        >>> a.checkpoint
        'model.pth'
    """
    parser = argparse.ArgumentParser(
        description='Evaluate trained DreamerV3 on LimoCustomEnv.'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to Dreamer .pth checkpoint (required unless --self-check).'
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
    return parser.parse_args(argv)


# ── Main entry point ───────────────────────────────────────────────────────

def main():
    """Run evaluation from CLI args.

    Pre: --checkpoint path exists and is a valid Dreamer .pth checkpoint.
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
        2. dreamer_inference_step produces correct shapes with an
           untrained Dreamer instance.
        3. evaluate() outcome classification + rate-summing works
           by running a single episode and checking all keys.

    Returns:
        int: Number of passed tests (0-3).

    Example:
        >>> n = run_self_check()
        >>> n == 3
        True

    Note on design: we do NOT train a tiny Dreamer for self-check
    (that would take ~5-10 minutes to fill buffer + gradient steps).
    Instead we verify shape correctness and logic consistency with
    an untrained (random) Dreamer — which suffices for catching
    refactoring bugs without the time cost.
    """
    print('=== SELF-CHECK: evaluate_dreamer.py ===')
    n_pass = 0
    n_total = 3

    # ── Test 1: argparse parsing ───────────────────────────────────────────
    print('\n--- Test 1: argparse argument parsing ---')
    try:
        args = parse_args([
            '--checkpoint', 'dummy.pth',
            '--episodes', '50',
            '--curriculum-stage', '1',
            '--seed', '999',
        ])
        checks = [
            ('checkpoint', args.checkpoint == 'dummy.pth'),
            ('episodes', args.episodes == 50),
            ('curriculum_stage', args.curriculum_stage == 1),
            ('seed', args.seed == 999),
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

    # ── Test 2: dreamer_inference_step shapes ──────────────────────────────
    print('\n--- Test 2: dreamer_inference_step shapes (untrained Dreamer) ---')
    try:
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)

        device = torch.device('cpu')
        config = loadConfig('limo-dreamer.yml')

        env_raw = LimoCustomEnv(randomize_goal=True)
        obs_shape = env_raw.observation_space.shape       # (23,)
        action_size = env_raw.action_space.shape[0]        # 2
        action_low = env_raw.action_space.low.tolist()
        action_high = env_raw.action_space.high.tolist()

        dreamer = Dreamer(obs_shape, action_size, action_low, action_high,
                          device, config.dreamer)

        # Initialize states
        rec = torch.zeros(1, dreamer.recurrentSize, device=device)
        lat = torch.zeros(1, dreamer.latentSize, device=device)
        act = torch.zeros(1, action_size, device=device)

        # Get a real observation
        obs, _ = env_raw.reset(seed=42)

        # Run inference step
        action_out, new_rec, new_lat = dreamer_inference_step(
            dreamer, obs, rec, lat, act)

        shape_ok = (
            action_out.shape == (1, 2)
            and new_rec.shape == (1, dreamer.recurrentSize)
            and new_lat.shape == (1, dreamer.latentSize)
        )
        print(f'  action_out shape: {tuple(action_out.shape)} '
              f'(expected (1, 2))  [{"OK" if action_out.shape == (1, 2) else "FAIL"}]')
        print(f'  new_rec shape:    {tuple(new_rec.shape)} '
              f'(expected (1, {dreamer.recurrentSize}))  [{"OK" if new_rec.shape == (1, dreamer.recurrentSize) else "FAIL"}]')
        print(f'  new_lat shape:    {tuple(new_lat.shape)} '
              f'(expected (1, {dreamer.latentSize}))  [{"OK" if new_lat.shape == (1, dreamer.latentSize) else "FAIL"}]')
        print(f'  action values in valid range: '
              f'v=[{action_out[0,0].item():.4f}] w=[{action_out[0,1].item():.4f}]')

        env_raw.close()

        if shape_ok:
            print('  >>> PASS <<<')
            n_pass += 1
        else:
            print('  >>> FAIL <<<')
    except Exception as e:
        import traceback
        print(f'  Inference step test error: {e}')
        traceback.print_exc()
        print('  >>> FAIL <<<')

    # ── Test 3: evaluate() logic and outcome classification ────────────────
    print('\n--- Test 3: evaluate() logic with untrained Dreamer (3 episodes) ---')
    try:
        import tempfile

        # Create a dummy .pth checkpoint from the untrained dreamer
        device = torch.device('cpu')
        config = loadConfig('limo-dreamer.yml')
        env_raw = LimoCustomEnv(randomize_goal=True)
        obs_shape = env_raw.observation_space.shape
        action_size = env_raw.action_space.shape[0]
        action_low = env_raw.action_space.low.tolist()
        action_high = env_raw.action_space.high.tolist()
        dreamer = Dreamer(obs_shape, action_size, action_low, action_high,
                          device, config.dreamer)

        # Save untrained checkpoint for evaluate() to load
        tmp_path = '/tmp/dreamer_eval_test.pth'
        dreamer.saveCheckpoint(tmp_path)
        env_raw.close()

        # Run evaluate with 3 episodes, stage 0
        stats = evaluate(
            checkpoint_path=tmp_path,
            n_episodes=3,
            curriculum_stage=0,
            seed=200,
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

        n_ok = stats['n_episodes'] == 3
        print(f'  n_episodes == 3: {n_ok}')

        len_ok = len(stats['results']) == 3
        print(f'  results list length 3: {len_ok}')

        rates = [stats['goal_rate'], stats['crash_rate'], stats['timeout_rate']]
        range_ok = all(0.0 <= r <= 1.0 for r in rates)
        print(f'  All rates in [0,1]: {range_ok}')

        len_pos = stats['mean_episode_length'] > 0
        print(f'  mean_episode_length > 0: {len_pos}')

        rate_sum = stats['goal_rate'] + stats['crash_rate'] + stats['timeout_rate']
        sum_ok = abs(rate_sum - 1.0) < 1e-9
        print(f'  goal_rate + crash_rate + timeout_rate = {rate_sum:.6f}'
              f'  [{"OK" if sum_ok else "FAIL"}]')

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        if key_ok and type_ok and n_ok and len_ok and range_ok and len_pos and sum_ok:
            print('  >>> PASS <<<')
            n_pass += 1
        else:
            print('  >>> FAIL <<<')
    except Exception as e:
        import traceback
        print(f'  evaluate() logic test error: {e}')
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
