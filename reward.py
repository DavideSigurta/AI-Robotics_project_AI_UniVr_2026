import numpy as np

# ── Scene constants ─────────────────────────────────────────────────────────
LIDAR_MAX_DIST     = 5.0
DIST_NORM_MAX      = 7.0
LIDAR_FOV          = 2 * np.pi / 3   # 120 degrees — verified from CoppeliaSim scene
N_LIDAR            = 20
GOAL_BONUS         = 10.0
CRASH_COST         = 10.0
OBSTACLE_SAFE_DIST = 0.3
CRASH_DIST         = 0.15
GOAL_DIST          = 0.1
DT                 = 0.05
MAX_STEPS          = 500
WORKSPACE          = (-2.0, 2.0)

# Real position in scene limo_cbf.ttt (read from CoppeliaSim GUI)
ROBOT_INIT_POS     = (-1.7, 0.74, 0.0)   # x, y, theta
GOAL_FIXED_POS     = (1.74584, -0.44267)  # used during inference on the real scene


def compute_reward(observation, action, info):
    """Compute the reward for the current step.

    Parameters
    ----------
    observation : np.ndarray
        Observation vector (lidar + goal + current velocities).
    action : np.ndarray
        Applied action [v, omega].
    info : dict
        Info dictionary containing crash/goal flags.

    Returns
    -------
    reward : float
    done : bool
    info : dict
    """
    # TODO: implement
    pass


def check_crash(lidar):
    """Check if the robot is colliding with an obstacle."""
    # TODO: implement
    pass


def check_goal_reached(robot_pos, goal_pos):
    """Check if the robot has reached the goal."""
    # TODO: implement
    pass
