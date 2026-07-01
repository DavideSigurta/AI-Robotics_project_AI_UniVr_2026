#!/usr/bin/env python3
"""CLI entry point for CoppeliaSim inference — single episode and batch.

Usage:
    # Single episode (demo): open scene in GUI → Start Sim → run:
    python inference/run_inference.py --algo ppo --cbf on

    # Batch evaluation:
    python inference/run_inference.py --algo dreamer --cbf off --episodes 10

    # All 4 combos (PPO/Dreamer x CBF on/off) on a scene:
    python inference/run_inference.py --all --episodes 5 --scene scenes/limo_cbf.ttt

    # Custom checkpoint:
    python inference/run_inference.py --algo ppo --checkpoint my_model.zip
"""

import argparse
import os
import sys
import time
from typing import List, Optional

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from inference.inference_runner import CoppeliaInferenceRunner
from inference.constants import DEFAULT_SCENE


# ── Default output directory ───────────────────────────────────────────────
_DEFAULT_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "results", "inference")


def _combo_name(algo: str, cbf: bool) -> str:
    return f"{algo}_{'cbf' if cbf else 'nocbf'}"


def run_combo(
    algo: str,
    use_cbf: bool,
    scene_path: str,
    n_episodes: int,
    output_dir: str,
    verbose: bool,
    device: str = "cpu",
    checkpoint_path: Optional[str] = None,
) -> List[dict]:
    """Run inference for one algo x CBF combination.

    Args:
        algo: "ppo" or "dreamer".
        use_cbf: Apply CBF filter.
        scene_path: Path to .ttt scene.
        n_episodes: Number of episodes.
        output_dir: Directory to save results CSV.
        verbose: Print per-step output.
        device: "cpu" or "cuda".
        checkpoint_path: Optional override for checkpoint path.

    Returns:
        List of episode metrics dicts.
    """
    label = f"{algo.upper()} {'+CBF' if use_cbf else 'no-CBF'}"
    print(f"\n{'─'*60}")
    print(f"  {label} | {n_episodes} ep | {os.path.basename(scene_path)}")
    print(f"{'─'*60}")

    runner = CoppeliaInferenceRunner(
        algo=algo,
        use_cbf=use_cbf,
        scene_path=scene_path,
        checkpoint_path=checkpoint_path,
        device=device,
        verbose=verbose,
    )

    if n_episodes <= 1:
        result = runner.run_episode()
        results = [result]
    else:
        results = runner.run_batch(n_episodes=n_episodes)

    runner.print_results(results, title=label)
    runner.close()

    # Save CSV
    scene_name = os.path.splitext(os.path.basename(scene_path))[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_name = f"{_combo_name(algo, use_cbf)}_{scene_name}_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_name)
    runner.save_csv(results, csv_path)

    return results


def run_all_combos(
    scene_path: str,
    n_episodes: int,
    output_dir: str,
    verbose: bool,
    device: str = "cpu",
    checkpoint_ppo: Optional[str] = None,
    checkpoint_dreamer: Optional[str] = None,
):
    """Run all 4 combos: PPO no-CBF, PPO+CBF, Dreamer no-CBF, Dreamer+CBF.

    Args:
        scene_path: Path to .ttt scene.
        n_episodes: Episodes per combo.
        output_dir: Output directory.
        verbose: Print per-step output.
        device: "cpu" or "cuda".
        checkpoint_ppo: Optional PPO checkpoint override.
        checkpoint_dreamer: Optional Dreamer checkpoint override.
    """
    print(f"\n{'='*60}")
    print(f"  FULL INFERENCE BENCHMARK")
    print(f"  Scene: {scene_path}")
    print(f"  Episodes per combo: {n_episodes}")
    print(f"{'='*60}")

    combos = [
        ("ppo", False, checkpoint_ppo),
        ("ppo", True, checkpoint_ppo),
        ("dreamer", False, checkpoint_dreamer),
        ("dreamer", True, checkpoint_dreamer),
    ]

    all_results = {}
    for algo, use_cbf, ckpt in combos:
        label = f"{algo.upper()} {'+CBF' if use_cbf else 'no-CBF'}"
        print(f"\n{'#'*60}")
        print(f"  Running: {label}")
        print(f"{'#'*60}")
        try:
            results = run_combo(algo, use_cbf, scene_path, n_episodes,
                                output_dir, verbose, device, ckpt)
            all_results[_combo_name(algo, use_cbf)] = results
        except Exception as e:
            print(f"  ❌ {label} failed: {e}")

    # Final summary table
    print(f"\n\n{'='*60}")
    print(f"  FINAL SUMMARY — {os.path.basename(scene_path)}")
    print(f"{'='*60}")
    print(f"  {'Combo':<25} {'Goal':>6} {'Crash':>6} {'Time':>6} {'Reward':>8} {'CBF%':>6}")
    print(f"  {'─'*25} {'─'*6} {'─'*6} {'─'*6} {'─'*8} {'─'*6}")
    for combo_name, results in all_results.items():
        n = len(results)
        n_goal = sum(1 for r in results if r["outcome"] == "goal")
        n_crash = sum(1 for r in results if r["outcome"] == "crash")
        n_timeout = sum(1 for r in results if r["outcome"] == "timeout")
        mean_reward = sum(r["cumulative_reward"] for r in results) / n
        mean_cbf = sum(r["cbf_rate"] for r in results) / n
        print(f"  {combo_name:<25} {n_goal/n:.0%}     {n_crash/n:.0%}     "
              f"{n_timeout/n:.0%}     {mean_reward:>+6.2f}  {mean_cbf:>5.1%}")
    print(f"{'='*60}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run PPO/Dreamer inference on CoppeliaSim scenes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Demo: single episode with PPO + CBF
  python inference/run_inference.py --algo ppo --cbf on

  # Demo: single episode with Dreamer, no CBF
  python inference/run_inference.py --algo dreamer --cbf off

  # Batch: 10 episodes with PPO + CBF
  python inference/run_inference.py --algo ppo --cbf on --episodes 10

  # All combos: 5 episodes each
  python inference/run_inference.py --all --episodes 5

  # Custom scene
  python inference/run_inference.py --algo ppo --cbf off --scene my_scene.ttt

  # Custom checkpoint
  python inference/run_inference.py --algo dreamer --checkpoint my_dreamer.pth
        """,
    )

    # Mutually exclusive: specific combo or all combos
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--all", action="store_true",
        help="Run all 4 combos (PPO/Dreamer x CBF on/off). Overrides --algo and --cbf.",
    )
    group.add_argument(
        "--algo", type=str, default="ppo", choices=["ppo", "dreamer"],
        help="Algorithm to run (default: ppo). Ignored with --all.",
    )

    parser.add_argument(
        "--cbf", type=str, default="on", choices=["on", "off"],
        help="Apply CBF safety filter (default: on). Ignored with --all.",
    )
    parser.add_argument(
        "--scene", type=str, default=None,
        help="Path to .ttt scene (default: inference/scenes/limo_cbf.ttt).",
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Number of episodes (default: 1). >1 = batch mode with auto-reset.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Custom checkpoint path. Overrides default per algo.",
    )
    parser.add_argument(
        "--checkpoint-ppo", type=str, default=None,
        help="PPO checkpoint for --all mode (default: results/checkpoints_ppo/...).",
    )
    parser.add_argument(
        "--checkpoint-dreamer", type=str, default=None,
        help="Dreamer checkpoint for --all mode (default: results/checkpoints_dreamer/...).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=_DEFAULT_OUTPUT_DIR,
        help=f"Output directory for CSV results (default: {_DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Device for model inference (default: cpu).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-step output (only print summary).",
    )

    return parser.parse_args(argv)


def main():
    args = parse_args()

    # Resolve scene path
    scene_path = args.scene
    if scene_path is None:
        scene_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), DEFAULT_SCENE
        )
    elif not os.path.isabs(scene_path):
        # Try relative to cwd first, then inference/scenes/
        candidate = os.path.abspath(scene_path)
        if not os.path.isfile(candidate):
            candidate2 = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "scenes", scene_path
            )
            if os.path.isfile(candidate2):
                candidate = candidate2
        scene_path = candidate

    if not os.path.isfile(scene_path):
        print(f"❌ Scene not found: {scene_path}")
        print(f"   Copy limo_cbf.ttt from Lab5_shield to inference/scenes/")
        sys.exit(1)

    verbose = not args.quiet

    if args.all:
        if args.checkpoint:
            ckpt_ppo = args.checkpoint
            ckpt_dreamer = args.checkpoint
        else:
            ckpt_ppo = args.checkpoint_ppo
            ckpt_dreamer = args.checkpoint_dreamer
        run_all_combos(
            scene_path=scene_path,
            n_episodes=args.episodes,
            output_dir=args.output_dir,
            verbose=verbose,
            device=args.device,
            checkpoint_ppo=ckpt_ppo,
            checkpoint_dreamer=ckpt_dreamer,
        )
    else:
        use_cbf = args.cbf == "on"
        run_combo(
            algo=args.algo,
            use_cbf=use_cbf,
            scene_path=scene_path,
            n_episodes=args.episodes,
            output_dir=args.output_dir,
            verbose=verbose,
            device=args.device,
            checkpoint_path=args.checkpoint,
        )


if __name__ == "__main__":
    main()
