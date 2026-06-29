"""Inference runner for PPO / DreamerV3 models on the LIMO environment."""

# TODO: import torch / stable_baselines3, LimoCustomEnv


class InferenceRunner:
    """Load a trained model and run it in simulation."""

    def __init__(self, model_path, algo="ppo", scene_path=None):
        # TODO: implement
        pass

    def run_episode(self, render=True):
        """Run an inference episode."""
        # TODO: implement
        pass

    def record_video(self, output_path="videos/out.mp4"):
        """Record a video of the episode."""
        # TODO: implement
        pass
