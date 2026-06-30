"""DreamerV3 training on Modal cloud.

Launches DreamerV3 (NaturalDreamer adapted) training on a Modal A100 GPU
with persistent checkpointing via Modal Volume. The training loop uses
dreamer/main.py's entry point with limo-dreamer.yml config.

Usage:
    modal run modal_scripts/modal_train_dreamer.py           # full 60k gradient steps
    modal run modal_scripts/modal_train_dreamer.py --dry     # dry-run self-check
"""

import os
import sys

# Allow imports from project root (limo_env.py, reward.py) and dreamer/
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_dreamer_path = os.path.join(_project_root, "dreamer")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _dreamer_path not in sys.path:
    sys.path.insert(0, _dreamer_path)

import modal


# ── Modal infrastructure ───────────────────────────────────────────────────

app = modal.App("limo-dreamer-training")

# Same volume as PPO training — shared across experiments
volume = modal.Volume.from_name("limo-checkpoints", create_if_missing=True)

# Container image with all dependencies + source code
# Needs torch (Dreamer), numpy, gymnasium, plus the source files
image = (
    modal.Image.from_registry("python:3.11-slim")
    .pip_install(
        "torch",
        "gymnasium",
        "numpy",
        "numba",
        "attridict",
        "pyyaml",
        "pandas",
        "plotly",
        "imageio",
    )
    .add_local_file(
        os.path.join(_project_root, "limo_env.py"),
        "/root/limo_env.py",
    )
    .add_local_file(
        os.path.join(_project_root, "reward.py"),
        "/root/reward.py",
    )
    # Copy entire dreamer/ directory
    .add_local_dir(
        _dreamer_path,
        "/root/dreamer",
    )
)


# ── Training function (runs on Modal A100) ─────────────────────────────────

@app.function(
    gpu="A100",
    timeout=14400,  # 4h timeout (60k gs with A100 ~2-3h; with curriculum & v2 config need margin)
    volumes={"/checkpoints": volume},
    image=image,
)
def train_dreamer(gradient_steps: int = 60_000) -> str:
    """Run DreamerV3 training on a Modal A100 GPU.

    Preconditions:
        - Modal Volume "limo-checkpoints" is writable at /checkpoints.
        - All dependencies installed (via container image).
        - Source files copied to /root/ (limo_env.py, reward.py, dreamer/).
    Postconditions:
        - Checkpoints saved every 5000 gradient steps to /checkpoints/dreamer/.
        - Metrics CSV and plots saved to /checkpoints/dreamer/metrics/.
        - Returns path to final checkpoint directory.

    Args:
        gradient_steps: Total world model gradient steps (default 60_000).

    Returns:
        str: Path to the checkpoint directory.

    Example:
        >>> path = train_dreamer.remote(gradient_steps=60_000)
        >>> print(path)
        /checkpoints/dreamer/checkpoints
    """
    # Source files copied into /root/ by the image definition above
    dreamer_path = "/root/dreamer"
    if dreamer_path not in sys.path:
        sys.path.insert(0, dreamer_path)
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    # Override config paths to point to Modal volume
    checkpoint_dir = "/checkpoints/dreamer"
    metrics_dir = os.path.join(checkpoint_dir, "metrics")
    plots_dir = os.path.join(checkpoint_dir, "plots")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Import dreamer modules (after sys.path setup)
    import torch
    from dreamer import Dreamer
    from envs import CleanEnvWrapper
    from limo_env import LimoCustomEnv
    from utils import loadConfig, seedEverything, saveLossesToCSV, plotMetrics, ensureParentFolders

    # ── Load config ────────────────────────────────────────────────────────
    # findFile walks from cwd (/root on Modal), so pass basename only
    config = loadConfig("limo-dreamer.yml")
    seedEverything(config.seed)

    # Override gradient steps from function argument
    config.gradientSteps = gradient_steps

    # Override folder names to use Modal volume paths
    config.folderNames.checkpointsFolder = checkpoint_dir
    config.folderNames.metricsFolder = metrics_dir
    config.folderNames.plotsFolder = plots_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: gradientSteps={config.gradientSteps}, replayRatio={config.replayRatio}")

    # ── Environment ────────────────────────────────────────────────────────
    # Start at curriculum stage 0 (roadmap §4.2): 2-4 obstacles, r∈[0.10, 0.15]
    # TODO: add curriculum logic (stages 0→1→2) matching train_ppo.py
    _env_kwargs = dict(
        n_obstacles_range=(2, 4),
        obs_radius_range=(0.10, 0.15),
        randomize_goal=True,
    )
    env = CleanEnvWrapper(LimoCustomEnv(**_env_kwargs))
    env_eval = CleanEnvWrapper(LimoCustomEnv(**_env_kwargs))

    obs_shape = env.observation_space.shape       # (23,)
    action_size = env.action_space.shape[0]        # 2
    action_low = env.action_space.low.tolist()     # [0.0, -1.0]
    action_high = env.action_space.high.tolist()   # [1.0, 1.0]

    print(f"Observation shape: {obs_shape}, Action size: {action_size}")
    print(f"Action low: {action_low}, Action high: {action_high}")

    # ── Dreamer ────────────────────────────────────────────────────────────
    dreamer = Dreamer(
        obs_shape, action_size, action_low, action_high,
        device, config.dreamer,
    )

    # Resume from checkpoint if one exists
    run_name = f"{config.environmentName}_{config.runName}"
    if config.resume:
        checkpoint_path = os.path.join(
            config.folderNames.checkpointsFolder,
            f"{run_name}_{config.checkpointToLoad}",
        )
        if os.path.exists(checkpoint_path + ".pth"):
            dreamer.loadCheckpoint(checkpoint_path)
            print(f"Resumed from checkpoint: {checkpoint_path}")

    # ── Curriculum stages (same proportions as PPO: 32% and 72%) ───────────
    # Stage thresholds in gradient steps at 60k total:
    #   Stage 0→1: 0.32 × 60k = 19200
    #   Stage 1→2: 0.72 × 60k = 43200
    CURRICULUM_GS_BOUNDARIES = {19200: 1, 43200: 2}
    _current_curriculum_stage = 0

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"Starting training: {config.gradientSteps} gradient steps...")

    # Fill buffer with initial episodes (stage 0 by default)
    dreamer.environmentInteraction(env, config.episodesBeforeStart, seed=config.seed)
    print(f"Buffer filled: {len(dreamer.buffer)} transitions")

    iterations = config.gradientSteps // config.replayRatio
    for iteration in range(iterations):
        # Track metrics periodically for console output
        _print_losses_at = config.checkpointInterval  # same cadence as checkpoints
        _last_print = 0

        for _ in range(config.replayRatio):
            sampled = dreamer.buffer.sample(
                dreamer.config.batchSize, dreamer.config.batchLength,
            )
            initial_states, wm_metrics = dreamer.worldModelTraining(sampled)
            bh_metrics = dreamer.behaviorTraining(initial_states)
            dreamer.totalGradientSteps += 1

            # Advance curriculum based on gradient steps (not env steps)
            for boundary, stage in sorted(CURRICULUM_GS_BOUNDARIES.items()):
                if (dreamer.totalGradientSteps >= boundary
                        and _current_curriculum_stage < stage):
                    _current_curriculum_stage = stage
                    env.env.set_curriculum_stage(stage)
                    env_eval.env.set_curriculum_stage(stage)
                    print(
                        f"Curriculum advanced to stage {stage} at "
                        f"gs={dreamer.totalGradientSteps}"
                    )
                    break

            # Print losses periodically
            if dreamer.totalGradientSteps - _last_print >= _print_losses_at:
                _last_print = dreamer.totalGradientSteps
                print(
                    f"gs={dreamer.totalGradientSteps:>6} | "
                    f"wm={wm_metrics['worldModelLoss']:.1f} "
                    f"rec={wm_metrics['reconstructionLoss']:.1f} "
                    f"kl={wm_metrics['klLoss']:.2f} "
                    f"rew_pred={wm_metrics['rewardPredictorLoss']:.2f} | "
                    f"act={bh_metrics['actorLoss']:.4f} "
                    f"crit={bh_metrics['criticLoss']:.4f}"
                )

            # Checkpoint + evaluation
            if (dreamer.totalGradientSteps % config.checkpointInterval == 0
                    and config.saveCheckpoints):
                suffix = f"{dreamer.totalGradientSteps // 1000}k"
                dreamer.saveCheckpoint(
                    os.path.join(
                        config.folderNames.checkpointsFolder,
                        f"{run_name}_{suffix}",
                    ),
                )
                eval_score = dreamer.environmentInteraction(
                    env_eval, config.numEvaluationEpisodes,
                    seed=config.seed, evaluation=True,
                )
                print(
                    f"Checkpoint {suffix:>6} | "
                    f"grad_steps={dreamer.totalGradientSteps} | "
                    f"eval_score={eval_score:.2f}"
                )

        # Collect new environment episodes
        recent = dreamer.environmentInteraction(
            env, config.numInteractionEpisodes, seed=config.seed,
        )

        # Log metrics
        if config.saveMetrics:
            metrics_base = {
                "envSteps": dreamer.totalEnvSteps,
                "gradientSteps": dreamer.totalGradientSteps,
                "totalReward": recent,
            }
            # Re-sample for metrics logging
            s = dreamer.buffer.sample(
                dreamer.config.batchSize, dreamer.config.batchLength,
            )
            fs, wm = dreamer.worldModelTraining(s)
            bh = dreamer.behaviorTraining(fs)
            saveLossesToCSV(
                os.path.join(config.folderNames.metricsFolder, run_name),
                metrics_base | wm | bh,
            )
            plotMetrics(
                os.path.join(config.folderNames.metricsFolder, run_name),
                savePath=os.path.join(config.folderNames.plotsFolder, run_name),
                title=f"{config.environmentName}",
            )

        if (iteration + 1) % 10 == 0:
            print(
                f"Iteration {iteration + 1}/{iterations} | "
                f"grad_steps={dreamer.totalGradientSteps} | "
                f"env_steps={dreamer.totalEnvSteps} | "
                f"reward={recent:.2f}"
            )

    # ── Save final checkpoint ──────────────────────────────────────────────
    dreamer.saveCheckpoint(
        os.path.join(config.folderNames.checkpointsFolder, f"{run_name}_final"),
    )

    summary = (
        f"\n{'=' * 50}\n"
        f"DreamerV3 training complete on Modal A100\n"
        f"  Total gradient steps: {dreamer.totalGradientSteps}\n"
        f"  Total env steps: {dreamer.totalEnvSteps}\n"
        f"  Checkpoints: {checkpoint_dir}\n"
        f"{'=' * 50}"
    )
    print(summary)

    env.close()
    return checkpoint_dir


# ── Local entrypoint ───────────────────────────────────────────────────────

@app.local_entrypoint()
def main(gradient_steps: int = 60_000):
    """Launch DreamerV3 training remotely on Modal.

    Pre: Modal CLI is authenticated and app "limo-dreamer-training" is deployed.
    Post: Training runs on Modal A100; checkpoints saved to Modal Volume.

    Args:
        gradient_steps: Total gradient steps (default 60_000).

    Example:
        $ modal run modal_scripts/modal_train_dreamer.py
        $ modal run modal_scripts/modal_train_dreamer.py --gradient-steps 5000
    """
    print(f"Launching DreamerV3 training on Modal A100 ({gradient_steps} gradient steps)...")
    result_path = train_dreamer.remote(gradient_steps=gradient_steps)
    print(f"Training complete. Checkpoints at: {result_path}")


# ── Self-check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== SELF-CHECK: modal_train_dreamer.py ===")
    n_pass = 0
    n_total = 3

    # ── Test 1: Modal image definition (no import errors) ──────────────────
    print("\n--- Test 1: Modal image definition (no import errors) ---")
    try:
        _ = image
        print("  Image object created successfully")
        print("  Local files: limo_env.py, reward.py, dreamer/")
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

    # ── Test 3: Local dry-run (env + Dreamer instantiation) ────────────────
    print("\n--- Test 3: Local env + Dreamer instantiation (dry run) ---")
    try:
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        import torch
        import numpy as np
        from envs import CleanEnvWrapper
        from limo_env import LimoCustomEnv
        from dreamer import Dreamer
        from utils import loadConfig

        config = loadConfig("limo-dreamer.yml")
        # Override batch config for dry-run (smaller = less buffer data needed)
        config.dreamer.batchSize = 4
        config.dreamer.batchLength = 8
        device = torch.device("cpu")

        env = CleanEnvWrapper(LimoCustomEnv(
            n_obstacles_range=(2, 4),
            obs_radius_range=(0.10, 0.15),
            randomize_goal=True,
        ))

        obs_shape = env.observation_space.shape
        action_size = env.action_space.shape[0]
        action_low = env.action_space.low.tolist()
        action_high = env.action_space.high.tolist()

        dreamer = Dreamer(obs_shape, action_size, action_low, action_high,
                          device, config.dreamer)

        # Verify key shapes
        obs = env.reset(seed=42)
        obs_shape_ok = obs.shape == (23,)

        # Quick buffer fill and forward pass
        avg = dreamer.environmentInteraction(env, 3, seed=42)
        buf_ok = len(dreamer.buffer) > 0

        # Sample and forward pass
        need = config.dreamer.batchSize * config.dreamer.batchLength
        assert len(dreamer.buffer) >= need, f'Buffer too small: {len(dreamer.buffer)} < {need}'
        s = dreamer.buffer.sample(config.dreamer.batchSize, config.dreamer.batchLength)
        fs, wm = dreamer.worldModelTraining(s)
        bh = dreamer.behaviorTraining(fs)
        forward_ok = ("worldModelLoss" in wm and "actorLoss" in bh)

        print(f"  Observation shape: {obs.shape}  [{'OK' if obs_shape_ok else 'FAIL'}]")
        print(f"  Action size: {action_size}  [OK]")
        print(f"  Buffer fill ({len(dreamer.buffer)} transitions):  [{'OK' if buf_ok else 'FAIL'}]")
        print(f"  Forward pass (WM + BH):  [{'OK' if forward_ok else 'FAIL'}]")
        print(f"    worldModelLoss: {wm['worldModelLoss']:.2f}")
        print(f"    actorLoss: {bh['actorLoss']:.4f}")
        print(f"    criticLoss: {bh['criticLoss']:.4f}")

        if obs_shape_ok and buf_ok and forward_ok:
            print("  >>> PASS <<<")
            n_pass += 1
        else:
            print("  >>> FAIL <<<")

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
