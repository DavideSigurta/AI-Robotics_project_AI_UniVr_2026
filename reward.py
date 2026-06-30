import numpy as np

# ── Scene constants ─────────────────────────────────────────────────────────
LIDAR_MAX_DIST     = 5.0
DIST_NORM_MAX      = 7.0
LIDAR_FOV          = 2 * np.pi / 3   # 120 degrees — verified from CoppeliaSim scene
N_LIDAR            = 20
GOAL_BONUS         = 10.0
CRASH_COST         = 10.0
STEP_PENALTY       = -0.01  # per-step cost to discourage standing still
OBSTACLE_SAFE_DIST = 0.3
CRASH_DIST         = 0.15
GOAL_DIST          = 0.1
DT                 = 0.05
MAX_STEPS          = 500
WORKSPACE          = (-2.0, 2.0)

# Real position in scene limo_cbf.ttt (read from CoppeliaSim GUI)
ROBOT_INIT_POS     = (-1.7, 0.74, 0.0)   # x, y, theta
GOAL_FIXED_POS     = (1.74584, -0.44267)  # used during inference on the real scene


def obstacle_penalty(min_lidar_m: float) -> float:
    """Continuous obstacle penalty, grows linearly as robot approaches obstacle.

    Pre: 0 <= min_lidar_m <= LIDAR_MAX_DIST
    Post: returns 0.0 if min_lidar_m >= OBSTACLE_SAFE_DIST,
          returns value in [0, 1] otherwise (1.0 when min_lidar_m = 0).

    Args:
        min_lidar_m: Minimum LiDAR reading in meters (not normalized).

    Returns:
        float: Penalty value in [0, 1].

    Example:
        >>> obstacle_penalty(0.5)   # > OBSTACLE_SAFE_DIST (0.3) -> 0.0
        >>> obstacle_penalty(0.15)  # (0.3 - 0.15) / 0.3 -> 0.5
        >>> obstacle_penalty(0.0)   # (0.3 - 0.0) / 0.3 -> 1.0
    """
    if min_lidar_m >= OBSTACLE_SAFE_DIST:
        return 0.0
    return (OBSTACLE_SAFE_DIST - min_lidar_m) / OBSTACLE_SAFE_DIST  # in [0, 1]


def compute_reward(distance, prev_distance, lidar_norm,
                   terminated_goal: bool, terminated_crash: bool) -> float:
    """Compute reward for the current step.

    Preconditions:
        - distance >= 0.0, prev_distance >= 0.0
        - lidar_norm is shape (20,) with values in [0, 1]
        - terminated_goal and terminated_crash are mutually exclusive bools
    Postconditions:
        - No side effects (pure function)
        - Reward components: shaping + obstacle_penalty + terminal bonus/cost

    Args:
        distance: Current euclidean distance to goal (m).
        prev_distance: Previous step euclidean distance to goal (m).
        lidar_norm: Normalized LiDAR readings, shape (20,), values in [0, 1].
        terminated_goal: True if goal reached this step.
        terminated_crash: True if crash detected this step.

    Returns:
        float: Total reward for this step.

    Example:
        # Step toward goal, no obstacle nearby, no termination
        >>> compute_reward(distance=0.8, prev_distance=1.0,
        ...                lidar_norm=np.full(20, 1.0),
        ...                terminated_goal=False, terminated_crash=False)
        0.2  # = (1.0 - 0.8) - 0.0
    """
    reward = 0.0

    # Shaping: reward approach to goal
    reward += float(prev_distance - distance)

    # Obstacle penalty based on closest LiDAR reading
    min_lidar_m = float(np.min(lidar_norm)) * LIDAR_MAX_DIST
    reward -= obstacle_penalty(min_lidar_m)

    # Small per-step cost to discourage v=0 equilibrium.
    # Not applied on terminal steps (goal or crash) so that
    # the terminal bonus/cost dominates the step's total.
    if not terminated_goal and not terminated_crash:
        reward += STEP_PENALTY

    # Terminal outcomes
    if terminated_goal:
        reward += GOAL_BONUS
    if terminated_crash:
        reward -= CRASH_COST

    return reward


def check_crash(lidar_norm: np.ndarray) -> bool:
    """Check if the robot is colliding with an obstacle.

    Precondition: lidar_norm has shape (20,) with values in [0, 1].
    Postcondition: Returns True if closest LiDAR reading in meters < CRASH_DIST.

    Args:
        lidar_norm: Normalized LiDAR readings, shape (20,), values in [0, 1].

    Returns:
        bool: True if min(lidar_norm) * LIDAR_MAX_DIST < CRASH_DIST.

    Example:
        >>> check_crash(np.array([0.02, 0.5, 1.0, ...]))  # 0.02 * 5.0 = 0.1 < 0.15 -> True
        >>> check_crash(np.array([0.1, 0.5, 1.0, ...]))   # 0.1 * 5.0 = 0.5 >= 0.15 -> False
    """
    return float(np.min(lidar_norm)) * LIDAR_MAX_DIST < CRASH_DIST


def check_goal_reached(robot_pos: tuple, goal_pos: tuple) -> bool:
    """Check if the robot has reached the goal position.

    Precondition: robot_pos and goal_pos are (x, y) tuples.
    Postcondition: Returns True if euclidean distance < GOAL_DIST.

    Args:
        robot_pos: Current robot position (x, y).
        goal_pos: Target goal position (x, y).

    Returns:
        bool: True if distance between robot and goal < GOAL_DIST.

    Example:
        >>> check_goal_reached((1.74, -0.44), (1.74584, -0.44267))  # dist ~0.006 < 0.1 -> True
        >>> check_goal_reached((-1.0, 0.0), (1.74584, -0.44267))    # dist ~2.75 >= 0.1 -> False
    """
    dist = np.hypot(robot_pos[0] - goal_pos[0], robot_pos[1] - goal_pos[1])
    return dist < GOAL_DIST


if __name__ == '__main__':
    # ── Self-check: 5 test cases ──────────────────────────────────────────
    print('=== SELF-CHECK: reward.py ===')

    lidar_clear = np.full(20, 1.0, dtype=np.float32)          # all max range
    lidar_near_obs = np.full(20, 0.3, dtype=np.float32)       # min = 0.3 * 5.0 = 1.5 m
    lidar_crash = np.full(20, 0.02, dtype=np.float32)         # min = 0.02 * 5.0 = 0.1 m < 0.15

    # Test 1: normal step, far from obstacles, moving toward goal (non-terminal)
    r1 = compute_reward(distance=1.0, prev_distance=1.2,
                        lidar_norm=lidar_clear,
                        terminated_goal=False, terminated_crash=False)
    expected_1 = (1.2 - 1.0) - 0.0 + STEP_PENALTY  # shaping + step cost
    print(f'Test 1 - normal approach:       got {r1:.4f}, expected {expected_1:.4f}')
    assert abs(r1 - expected_1) < 1e-6, f'MISMATCH: {r1} != {expected_1}'

    # Test 2: closer to goal than previous step -> larger positive shaping (non-terminal)
    r2 = compute_reward(distance=0.3, prev_distance=0.9,
                        lidar_norm=lidar_clear,
                        terminated_goal=False, terminated_crash=False)
    expected_2 = (0.9 - 0.3) - 0.0 + STEP_PENALTY  # 0.6 shaping + step cost
    print(f'Test 2 - strong approach:       got {r2:.4f}, expected {expected_2:.4f}')
    assert abs(r2 - expected_2) < 1e-6, f'MISMATCH: {r2} != {expected_2}'

    # Test 3: obstacle nearby -> penalty applied (non-terminal)
    r3 = compute_reward(distance=0.8, prev_distance=0.8,
                        lidar_norm=lidar_near_obs,
                        terminated_goal=False, terminated_crash=False)
    min_lidar_m_3 = float(np.min(lidar_near_obs)) * LIDAR_MAX_DIST  # 0.3 * 5.0 = 1.5
    pen_3 = 0.0 if min_lidar_m_3 >= OBSTACLE_SAFE_DIST else (OBSTACLE_SAFE_DIST - min_lidar_m_3) / OBSTACLE_SAFE_DIST
    expected_3 = 0.0 - pen_3 + STEP_PENALTY  # penalty + step cost
    print(f'Test 3 - obstacle penalty:      got {r3:.4f}, expected {expected_3:.4f}')
    assert abs(r3 - expected_3) < 1e-6, f'MISMATCH: {r3} != {expected_3}'

    # Test 4: goal reached -> bonus applied (terminal, NO step penalty)
    r4 = compute_reward(distance=0.05, prev_distance=0.5,
                        lidar_norm=lidar_clear,
                        terminated_goal=True, terminated_crash=False)
    expected_4 = (0.5 - 0.05) + GOAL_BONUS  # shaping + bonus, no STEP_PENALTY
    print(f'Test 4 - goal reached (+bonus):  got {r4:.4f}, expected {expected_4:.4f}')
    assert abs(r4 - expected_4) < 1e-6, f'MISMATCH: {r4} != {expected_4}'

    # Test 5: crash -> penalty applied (terminal, NO step penalty)
    r5 = compute_reward(distance=0.5, prev_distance=0.5,
                        lidar_norm=lidar_crash,
                        terminated_goal=False, terminated_crash=True)
    min_lidar_m_5 = float(np.min(lidar_crash)) * LIDAR_MAX_DIST  # 0.02 * 5.0 = 0.1
    pen_5 = (OBSTACLE_SAFE_DIST - min_lidar_m_5) / OBSTACLE_SAFE_DIST  # (0.3 - 0.1) / 0.3
    expected_5 = 0.0 - pen_5 - CRASH_COST  # no STEP_PENALTY on terminal
    print(f'Test 5 - crash (-cost):          got {r5:.4f}, expected {expected_5:.4f}')
    assert abs(r5 - expected_5) < 1e-6, f'MISMATCH: {r5} != {expected_5}'

    # ── check_crash tests ─────────────────────────────────────────────────
    print()
    print(f'check_crash(clear lidar):  {check_crash(lidar_clear)} (expected False)')
    print(f'check_crash(crash lidar):  {check_crash(lidar_crash)} (expected True)')
    assert not check_crash(lidar_clear)
    assert check_crash(lidar_crash)

    # ── check_goal_reached tests ──────────────────────────────────────────
    print()
    pos_at_goal = (1.74584, -0.44267)
    pos_far     = (-1.0, 0.0)
    print(f'check_goal_reached(at goal): {check_goal_reached(pos_at_goal, GOAL_FIXED_POS)} (expected True)')
    print(f'check_goal_reached(far):     {check_goal_reached(pos_far, GOAL_FIXED_POS)} (expected False)')
    assert check_goal_reached(pos_at_goal, GOAL_FIXED_POS)
    assert not check_goal_reached(pos_far, GOAL_FIXED_POS)

    # Test 6: standing still (v=0), no progress, clear lidar -> reward = STEP_PENALTY
    r6 = compute_reward(distance=1.0, prev_distance=1.0,
                        lidar_norm=lidar_clear,
                        terminated_goal=False, terminated_crash=False)
    expected_6 = STEP_PENALTY  # no shaping (prev - dist = 0), no obstacle, no terminal
    print(f'Test 6 - standing still:        got {r6:.4f}, expected {expected_6:.4f}')
    assert abs(r6 - expected_6) < 1e-6, f'MISMATCH: {r6} != {expected_6}'

    print()
    print('All SELF-CHECK tests passed.')
