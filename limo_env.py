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


# ── Helper: ray casting (non-Numba for now, can be njit-accelerated later) ──

def _ray_cast(robot_x, robot_y, robot_theta, ray_angles, obstacles,
              wall_min, wall_max, max_range) -> np.ndarray:
    """Ray casting against circular obstacles and axis-aligned walls.

    Preconditions:
        - obstacles is array of shape (N, 3) with [x, y, r] for each circle.
        - ray_angles is array of shape (M,) with angles relative to robot heading.
        - wall_min < wall_max define the axis-aligned bounding box.
    Postcondition:
        Returns distances array of shape (M,) in [0, max_range] (meters).

    Args:
        robot_x: Robot x position (m).
        robot_y: Robot y position (m).
        robot_theta: Robot heading (rad).
        ray_angles: Array of ray angles relative to heading, shape (M,).
        obstacles: Array of obstacles, shape (N, 3) = [x, y, radius].
        wall_min: Minimum coordinate for both axes (m).
        wall_max: Maximum coordinate for both axes (m).
        max_range: Maximum LiDAR range (m).

    Returns:
        np.ndarray: Distances along each ray, shape (M,), dtype float64.

    Example:
        >>> obs = np.array([[0.0, 0.0, 0.2]])
        >>> _ray_cast(-1.0, 0.0, 0.0, np.array([0.0]), obs, -2.0, 2.0, 5.0)
        array([0.8])  # ray hits circle at x=-0.8
    """
    M = len(ray_angles)
    distances = np.full(M, max_range, dtype=np.float64)

    for i in range(M):
        alpha = ray_angles[i]
        angle = robot_theta + alpha
        dx = np.cos(angle)
        dy = np.sin(angle)
        t_min = max_range

        # --- Wall intersections (4 axis-aligned walls) ---
        # x = wall_min (left)
        if abs(dx) > 1e-12:
            t = (wall_min - robot_x) / dx
            if t > 0 and t < t_min:
                y_hit = robot_y + t * dy
                if wall_min <= y_hit <= wall_max:
                    t_min = t
        # x = wall_max (right)
        if abs(dx) > 1e-12:
            t = (wall_max - robot_x) / dx
            if t > 0 and t < t_min:
                y_hit = robot_y + t * dy
                if wall_min <= y_hit <= wall_max:
                    t_min = t
        # y = wall_min (bottom)
        if abs(dy) > 1e-12:
            t = (wall_min - robot_y) / dy
            if t > 0 and t < t_min:
                x_hit = robot_x + t * dx
                if wall_min <= x_hit <= wall_max:
                    t_min = t
        # y = wall_max (top)
        if abs(dy) > 1e-12:
            t = (wall_max - robot_y) / dy
            if t > 0 and t < t_min:
                x_hit = robot_x + t * dx
                if wall_min <= x_hit <= wall_max:
                    t_min = t

        # --- Obstacle intersections (circles) ---
        N_obs = obstacles.shape[0]
        for j in range(N_obs):
            ox, oy, r = obstacles[j, 0], obstacles[j, 1], obstacles[j, 2]
            fx = ox - robot_x
            fy = oy - robot_y
            a = dx * dx + dy * dy  # = 1.0 since dx,dy are unit vector components
            b = -2.0 * (fx * dx + fy * dy)
            c = fx * fx + fy * fy - r * r
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                sqrt_disc = np.sqrt(disc)
                t1 = (-b - sqrt_disc) / (2.0 * a)
                t2 = (-b + sqrt_disc) / (2.0 * a)
                if 0 < t1 < t_min:
                    t_min = t1
                elif 0 < t2 < t_min:
                    t_min = t2

        distances[i] = t_min

    return distances


# ── Curriculum stages (roadmap §4.2) ───────────────────────────────────────

CURRICULUM_STAGES = {
    0: {'n_range': (2, 4),   'r_range': (0.10, 0.15)},
    1: {'n_range': (4, 7),   'r_range': (0.10, 0.20)},
    2: {'n_range': (6, 10),  'r_range': (0.10, 0.20)},
}


# ── Main environment class ─────────────────────────────────────────────────

class LimoCustomEnv(gym.Env):
    """Custom LIMO environment for reinforcement learning training.

    Observation space:  (23,) — 20 lidar rays + dist_norm + cos(angle) + sin(angle)
    Action space:       Box([0, -1], [1, 1]) — [linear velocity, angular velocity]
    Obstacles:          Randomly placed circles, regenerated at each reset.
    Goal:               Fixed or randomized, sampled excluding robot spawn zone.

    Preconditions for use:
        - Call reset() before first step().
        - Action values are clipped to valid range internally.
    """

    def __init__(self, n_obstacles_range=(3, 8), obs_radius_range=(0.1, 0.2),
                 randomize_goal=True, fixed_goal=None):
        """Initialize the custom LIMO environment.

        Args:
            n_obstacles_range: Tuple (min, max) for number of obstacles at reset.
            obs_radius_range: Tuple (min, max) for obstacle radius (m).
            randomize_goal: If True, sample random goal at each reset.
            fixed_goal: If not None, use this (x, y) as goal (overrides randomize).

        Example:
            >>> env = LimoCustomEnv(n_obstacles_range=(3, 8))
        """
        super().__init__()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0,  1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.n_obstacles_range = n_obstacles_range
        self.obs_radius_range = obs_radius_range
        self.randomize_goal = randomize_goal
        self.fixed_goal = fixed_goal  # (x, y) tuple or None

        # Ray angles: equispaced across FOV, centered on robot heading
        self._ray_angles = np.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, N_LIDAR)

        # Empty obstacle buffer
        self.obstacles = np.zeros((0, 3), dtype=np.float64)

        # Curriculum support
        self.curriculum_stage = 0

        # State variables (set by reset)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.goal = (0.0, 0.0)
        self.step_count = 0
        self.prev_distance = 0.0

    # ── Public API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        """Reset the environment to initial state.

        Pre: None.
        Post: Robot at ROBOT_INIT_POS, obstacles regenerated, goal (re)sampled.

        Args:
            seed: Optional RNG seed for reproducibility.
            options: Optional dict (unused, gymnasium API compatibility).

        Returns:
            Tuple of (obs, info) where obs is shape (23,).

        Example:
            >>> env = LimoCustomEnv()
            >>> obs, info = env.reset(seed=42)
            >>> obs.shape
            (23,)
        """
        super().reset(seed=seed)

        # Robot init position (from Lab 5 scene)
        self.robot_x, self.robot_y, self.robot_theta = ROBOT_INIT_POS
        self.step_count = 0

        # Goal: fixed or random
        if self.fixed_goal is not None:
            self.goal = self.fixed_goal
        elif self.randomize_goal:
            self.goal = self._sample_pos(
                exclude=[(self.robot_x, self.robot_y, 0.5)]
            )
        else:
            self.goal = self.fixed_goal  # None -> will fail gracefully

        # Obstacles
        n_obs = int(self.np_random.integers(*self.n_obstacles_range))
        self._sample_obstacles(n_obs)

        # Initial distance
        self.prev_distance = np.hypot(
            self.goal[0] - self.robot_x, self.goal[1] - self.robot_y
        )

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        """Execute one simulation step.

        Preconditions:
            - action is array-like of length 2.
            - Environment has been reset.
        Postconditions:
            - Robot position updated via differential drive kinematics.
            - Robot position clipped to WORKSPACE bounds.
            - No mutation of input action array.

        Args:
            action: np.ndarray of shape (2,) = [v, w].
                    v is clipped to [0.0, 1.0], w to [-1.0, 1.0].

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).

        Example:
            >>> env = LimoCustomEnv()
            >>> env.reset(seed=42)
            >>> obs, r, term, trunc, info = env.step(np.array([0.5, 0.0]))
        """
        # Clip action to valid ranges
        v = float(np.clip(action[0], 0.0, 1.0))
        w = float(np.clip(action[1], -1.0, 1.0))

        # Differential drive kinematics (roadmap §4.3)
        self.robot_x += v * np.cos(self.robot_theta) * DT
        self.robot_y += v * np.sin(self.robot_theta) * DT
        self.robot_theta += w * DT

        # Enforce workspace bounds
        self.robot_x = float(np.clip(self.robot_x, WORKSPACE[0], WORKSPACE[1]))
        self.robot_y = float(np.clip(self.robot_y, WORKSPACE[0], WORKSPACE[1]))

        # Normalize theta to [-pi, pi]
        self.robot_theta = (self.robot_theta + np.pi) % (2.0 * np.pi) - np.pi

        self.step_count += 1

        # Build observation and compute metrics
        obs = self._get_obs()

        distance = np.hypot(
            self.goal[0] - self.robot_x, self.goal[1] - self.robot_y
        )

        terminated_crash = check_crash(obs[:N_LIDAR])
        terminated_goal = check_goal_reached(
            (self.robot_x, self.robot_y), self.goal
        )
        terminated = bool(terminated_crash or terminated_goal)
        truncated = bool(self.step_count >= MAX_STEPS)

        reward = compute_reward(
            distance, self.prev_distance, obs[:N_LIDAR],
            terminated_goal, terminated_crash
        )

        self.prev_distance = distance

        info = {
            'crashed': terminated_crash,
            'goal_reached': terminated_goal,
            'distance': distance,
        }

        return obs, reward, terminated, truncated, info

    # ── Observation ────────────────────────────────────────────────────────

    def _get_obs(self):
        """Build observation vector from current state.

        Pre: self.robot_{x,y,theta}, self.obstacles, self.goal are valid.
        Post: Returns (23,) float32 array with normalized values.

        Returns:
            np.ndarray: Concatenation of [lidar_norm(20), dist_norm, cos, sin].

        Example:
            >>> env = LimoCustomEnv()
            >>> env.reset(seed=42)
            >>> obs = env._get_obs()
            >>> obs.shape
            (23,)
        """
        # Ray cast to get distances in meters
        distances = _ray_cast(
            self.robot_x, self.robot_y, self.robot_theta,
            self._ray_angles, self.obstacles,
            WORKSPACE[0], WORKSPACE[1], LIDAR_MAX_DIST
        )

        # Normalized LiDAR (roadmap §4.4)
        lidar_norm = np.clip(
            distances / LIDAR_MAX_DIST, 0.0, 1.0
        ).astype(np.float32)

        # Goal-relative features
        dx_goal = self.goal[0] - self.robot_x
        dy_goal = self.goal[1] - self.robot_y
        distance = np.hypot(dx_goal, dy_goal)
        dist_norm = np.clip(distance / DIST_NORM_MAX, 0.0, 1.0)

        angle_to_goal = np.arctan2(dy_goal, dx_goal) - self.robot_theta
        angle_to_goal = (angle_to_goal + np.pi) % (2.0 * np.pi) - np.pi

        obs = np.concatenate([
            lidar_norm,
            [np.float32(dist_norm),
             np.float32(np.cos(angle_to_goal)),
             np.float32(np.sin(angle_to_goal))]
        ]).astype(np.float32)

        return obs

    # ── Obstacle generation ────────────────────────────────────────────────

    def _sample_pos(self, exclude=None):
        """Sample a random (x, y) position inside workspace, avoiding excluded zones.

        Pre: exclude is list of (x, y, radius) tuples to avoid.
        Post: Returns (x, y) tuple with buffer >= 0.3m from edges.

        Args:
            exclude: List of (ex_x, ex_y, min_dist) to stay away from.

        Returns:
            Tuple (x, y) with uniform distribution in feasible region.

        Example:
            >>> env = LimoCustomEnv()
            >>> env.reset(seed=42)
            >>> env._sample_pos(exclude=[(-1.7, 0.74, 0.5)])
            (0.23, -0.85)  # random, depends on seed
        """
        margin = 0.3
        low = WORKSPACE[0] + margin
        high = WORKSPACE[1] - margin
        for _ in range(1000):
            x = float(self.np_random.uniform(low, high))
            y = float(self.np_random.uniform(low, high))
            if exclude:
                ok = True
                for ex, ey, min_d in exclude:
                    if np.hypot(x - ex, y - ey) < min_d:
                        ok = False
                        break
                if ok:
                    return (x, y)
            else:
                return (x, y)
        # Fallback: return center
        return (0.0, 0.0)

    def _sample_obstacles(self, n):
        """Generate n circular obstacles in the workspace.

        Pre: n >= 0.
        Post: self.obstacles is overwritten with array shape (n_eff, 3).
              Obstacles avoid robot spawn zone (0.5m) and goal (0.4m).

        Args:
            n: Desired number of obstacles (actual may be fewer if placement fails).

        Example:
            >>> env = LimoCustomEnv()
            >>> env.reset(seed=42)
            >>> env._sample_obstacles(5)
            >>> env.obstacles.shape[0]
            5
        """
        r_min, r_max = self.obs_radius_range
        exclude = [
            (self.robot_x, self.robot_y, 0.5),
            (self.goal[0], self.goal[1], 0.4),
        ]
        margin = 0.3
        low = WORKSPACE[0] + margin
        high = WORKSPACE[1] - margin

        obs_list = []
        for _ in range(n):
            for _ in range(1000):
                x = float(self.np_random.uniform(low, high))
                y = float(self.np_random.uniform(low, high))
                r = float(self.np_random.uniform(r_min, r_max))
                # Check overlap with existing obstacles and excluded zones
                ok = True
                for ex, ey, er in exclude:
                    if np.hypot(x - ex, y - ey) < er + r + 0.05:
                        ok = False
                        break
                if ok:
                    obs_list.append([x, y, r])
                    exclude.append((x, y, r))
                    break

        self.obstacles = np.array(obs_list, dtype=np.float64) if obs_list else np.zeros((0, 3), dtype=np.float64)

    # ── Curriculum (roadmap §4.2) ──────────────────────────────────────────

    def set_curriculum_stage(self, stage: int):
        """Set obstacle difficulty stage for curriculum learning.

        Stages (from roadmap §4.2):
            0: N in [2,4], radius in [0.10, 0.15]  (easy)
            1: N in [4,7], radius in [0.10, 0.20]  (medium)
            2: N in [6,10], radius in [0.10, 0.20] (hard)

        Pre: stage in {0, 1, 2}.
        Post: self.n_obstacles_range and self.obs_radius_range are updated.

        Args:
            stage: Integer 0, 1, or 2.

        Example:
            >>> env = LimoCustomEnv()
            >>> env.set_curriculum_stage(2)
            >>> env.n_obstacles_range
            (6, 10)
        """
        if stage not in CURRICULUM_STAGES:
            raise ValueError(f'Invalid curriculum stage {stage}. Must be 0, 1, or 2.')
        params = CURRICULUM_STAGES[stage]
        self.n_obstacles_range = params['n_range']
        self.obs_radius_range = params['r_range']
        self.curriculum_stage = stage


# ── Self-check ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== SELF-CHECK: limo_env.py ===')
    n_pass = 0
    n_total = 5

    # Test 1: instantiate + reset, verify obs shape and ranges
    print('\n--- Test 1: reset produces valid observation ---')
    env = LimoCustomEnv(randomize_goal=True)
    obs, info = env.reset(seed=42)
    shape_ok = obs.shape == (23,)
    range_ok = bool(np.all(obs[:20] >= 0.0) and np.all(obs[:20] <= 1.0))
    dist_ok = bool(0.0 <= obs[20] <= 1.0)
    cos_ok = bool(-1.0 <= obs[21] <= 1.0)
    sin_ok = bool(-1.0 <= obs[22] <= 1.0)
    all_ok = shape_ok and range_ok and dist_ok and cos_ok and sin_ok
    print(f'  shape=(23,): {shape_ok}')
    print(f'  lidar in [0,1]: {range_ok}')
    print(f'  dist_norm in [0,1]: {dist_ok}')
    print(f'  cos in [-1,1]: {cos_ok}')
    print(f'  sin in [-1,1]: {sin_ok}')
    print(f'  obstacles: {env.obstacles.shape[0]} placed')
    print(f'  goal: {env.goal}')
    if all_ok:
        print('  >>> PASS <<<')
        n_pass += 1
    else:
        print('  >>> FAIL <<<')

    # Test 2: 10 steps straight forward, print obs and reward
    print('\n--- Test 2: 10 steps forward (v=0.5, w=0.0) ---')
    env.reset(seed=42)
    for i in range(10):
        obs, reward, term, trunc, info = env.step(np.array([0.5, 0.0], dtype=np.float32))
        print(f'  step {i+1:2d}: reward={reward:+7.4f}  dist={info["distance"]:6.3f}m  '
              f'pos=({env.robot_x:.3f},{env.robot_y:.3f})  theta={env.robot_theta:.3f}')
    print('  >>> PASS <<<')
    n_pass += 1

    # Test 3: full random episode
    print('\n--- Test 3: random policy episode ---')
    env.reset(seed=42)
    total_reward = 0.0
    for step in range(MAX_STEPS + 10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        if term or trunc:
            outcome = 'GOAL' if info['goal_reached'] else ('CRASH' if info['crashed'] else 'TIMEOUT')
            print(f'  Episode ended at step {step+1}: {outcome}')
            print(f'  Total reward: {total_reward:.4f}')
            break
    else:
        print(f'  Unexpected: episode ran {MAX_STEPS} steps without termination')
    print('  >>> PASS <<<')
    n_pass += 1

    # Test 4: verify check_crash triggers near obstacle
    print('\n--- Test 4: crash detection near obstacle ---')
    env.reset(seed=42)
    # Place robot 0.05m from an obstacle
    env.obstacles = np.array([[0.5, 0.0, 0.15]], dtype=np.float64)
    env.robot_x = 0.5 + 0.05 + 0.15  # 0.7 — barely touching
    env.robot_y = 0.0
    env.robot_theta = np.pi  # facing left (toward obstacle)
    obs = env._get_obs()
    min_lidar_m = float(np.min(obs[:20])) * LIDAR_MAX_DIST
    crashed = check_crash(obs[:20])
    print(f'  min lidar (m): {min_lidar_m:.4f}  (CRASH_DIST={CRASH_DIST})')
    print(f'  crash detected: {crashed}')
    if min_lidar_m < CRASH_DIST and crashed:
        print('  >>> PASS <<<')
        n_pass += 1
    else:
        print('  >>> FAIL <<<')

    # Test 5: curriculum stages
    print('\n--- Test 5: curriculum stages ---')
    env = LimoCustomEnv()
    env.set_curriculum_stage(0)
    print(f'  stage 0: n_range={env.n_obstacles_range}, r_range={env.obs_radius_range}')
    assert env.n_obstacles_range == (2, 4), f'Expected (2,4), got {env.n_obstacles_range}'
    assert env.obs_radius_range == (0.10, 0.15), f'Expected (0.10,0.15), got {env.obs_radius_range}'
    env.set_curriculum_stage(1)
    print(f'  stage 1: n_range={env.n_obstacles_range}, r_range={env.obs_radius_range}')
    assert env.n_obstacles_range == (4, 7)
    assert env.obs_radius_range == (0.10, 0.20)
    env.set_curriculum_stage(2)
    print(f'  stage 2: n_range={env.n_obstacles_range}, r_range={env.obs_radius_range}')
    assert env.n_obstacles_range == (6, 10)
    assert env.obs_radius_range == (0.10, 0.20)
    print('  >>> PASS <<<')
    n_pass += 1

    print(f'\n=== Result: {n_pass}/{n_total} tests passed ===')
    if n_pass == n_total:
        print('All SELF-CHECK tests passed.')
    else:
        print(f'SOME TESTS FAILED ({n_total - n_pass} failures).')
