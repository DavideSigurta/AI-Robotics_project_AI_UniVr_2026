import gymnasium as gym
import numpy as np
from gymnasium import spaces

from reward import (
    LIDAR_MAX_DIST, DIST_NORM_MAX, LIDAR_FOV, N_LIDAR,
    GOAL_BONUS, CRASH_COST, OBSTACLE_SAFE_DIST, CRASH_DIST,
    GOAL_DIST, DT, MAX_STEPS, WORKSPACE,
    ROBOT_INIT_POS, GOAL_FIXED_POS,
    compute_reward, check_crash, check_goal_reached,
)


class LimoCustomEnv(gymnasium.Env):
    """Custom LIMO environment for reinforcement learning.

    Observation space:  (23,)  — 20 lidar rays + (dx, dy, v, omega)
    Action space:       Box([0, -1], [1, 1])  — [linear velocity, angular velocity]
    """

    def __init__(self, scene_path=None, **kwargs):
        super().__init__()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([0, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        # TODO: initialize CoppeliaSim connection, robot, obstacles
        pass

    def reset(self, *, seed=None, options=None):
        """Reset the environment and return the initial observation."""
        # TODO: implement
        return np.zeros(23, dtype=np.float32), {}

    def step(self, action):
        """Execute a simulation step.

        Parameters
        ----------
        action : np.ndarray
            [v, omega] velocity commands.

        Returns
        -------
        obs : np.ndarray
        reward : float
        terminated : bool
        truncated : bool
        info : dict
        """
        # TODO: implement
        return np.zeros(23, dtype=np.float32), 0.0, False, False, {}

    def _get_obs(self):
        """Build the observation vector (lidar + goal + velocities)."""
        # TODO: implement
        return np.zeros(23, dtype=np.float32)

    def _sample_obstacles(self):
        """Generate a new obstacle configuration (for training)."""
        # TODO: implement
        pass
