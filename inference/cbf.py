"""Standalone CBF safety filter for LIMO navigation.

Extracted from Lab5 controller.py with LIDAR_FOV corrected to 120deg (2pi/3).
No dependency on SB3, torch, or any RL library — usable with any policy.

Usage:
    from inference.cbf import get_safe_action
    v_safe, w_safe = get_safe_action(v_nom, w_nom, lidar_data)
"""

import os
import sys

# Allow running as script: python inference/cbf.py
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import cvxpy as cp

from inference.constants import (
    LIDAR_MAX_DIST, LIDAR_FOV, N_LIDAR,
    CBF_LOOKAHEAD, CBF_R_SAFE, CBF_GAMMA,
    CBF_V_MIN, CBF_V_MAX,
)


def get_safe_action(
    v_nom: float,
    w_nom: float,
    lidar_data: np.ndarray,
    lidar_fov: float = LIDAR_FOV,
    lidar_max_dist: float = LIDAR_MAX_DIST,
    n_lidar: int = N_LIDAR,
    lookahead: float = CBF_LOOKAHEAD,
    r_safe: float = CBF_R_SAFE,
    gamma: float = CBF_GAMMA,
    v_min: float = CBF_V_MIN,
    v_max: float = CBF_V_MAX,
) -> list[float]:
    """Apply CBF safety filter to a nominal action.

    Solves a QP that minimally modifies [v_nom, w_nom] subject to CBF
    constraints for each LiDAR ray that sees a nearby obstacle.

    The CBF constraint for each ray at distance d and angle alpha:
        h = (x_obs - lookahead)^2 + y_obs^2 - r_safe^2  >= 0
        h_dot + gamma * h >= 0

    Preconditions:
        - lidar_data contains ACTUAL distances in meters (NOT normalized).
        - v_nom in [0.0, 1.0], w_nom in [-1.0, 1.0].
    Postconditions:
        - Returns [v_safe, w_safe] with v in [v_min, v_max], w in [-1, 1].
        - If QP is infeasible, returns [0.0, 0.0] (emergency stop).

    Args:
        v_nom: Nominal linear velocity from policy [0.0, 1.0].
        w_nom: Nominal angular velocity from policy [-1.0, 1.0].
        lidar_data: Array of actual LiDAR distances in meters, shape (n_lidar,).
        lidar_fov: LiDAR field of view in radians (default 120deg).
        lidar_max_dist: LiDAR max range in meters (default 5.0).
        n_lidar: Number of LiDAR rays (default 20).
        lookahead: Lookahead distance for CBF (default 0.35).
        r_safe: Safe radius for CBF (default 0.30).
        gamma: CBF decay rate (default 2.0).
        v_min: Minimum safe linear velocity (default 0.1).
        v_max: Maximum safe linear velocity (default 0.7).

    Returns:
        list[float]: [v_safe, w_safe] — the filtered action.

    Example:
        >>> lidar = np.random.uniform(0.5, 5.0, 20)
        >>> v_s, w_s = get_safe_action(0.8, -0.3, lidar)
        >>> 0.0 <= v_s <= 1.0 and -1.0 <= w_s <= 1.0
        True
    """
    angles = np.linspace(-lidar_fov / 2, lidar_fov / 2, n_lidar)

    A_list: list[list[float]] = []
    b_list: list[float] = []

    for d, alpha in zip(lidar_data, angles):
        # Skip rays that don't see an obstacle (at max range)
        if d >= lidar_max_dist - 0.05:
            continue

        # Obstacle position in robot frame
        x_obs = d * np.cos(alpha)
        y_obs = d * np.sin(alpha)

        # Barrier function: distance from lookahead point to obstacle edge
        h = (x_obs - lookahead) ** 2 + y_obs ** 2 - r_safe ** 2

        # CBF constraint coefficients.
        # For a differential-drive robot with lookahead point CBF:
        #   h_dot = -2*(x_obs-L)*v - 2*L*y_obs*w
        # CBF condition: h_dot + gamma*h >= 0
        # => -2*(x_obs-L)*v - 2*L*y_obs*w >= -gamma*h
        # => 2*(x_obs-L)*v + 2*L*y_obs*w <= gamma*h
        # So constraint is: A @ u <= b  where
        #   A_v = 2*(x_obs - lookahead)
        #   A_w = 2*lookahead*y_obs
        #   b   = gamma*h
        # (Verified against Lab5 controller.py CBF math.)
        A_v = 2.0 * (x_obs - lookahead)
        A_w = 2.0 * lookahead * y_obs
        b_val = gamma * h

        A_list.append([A_v, A_w])
        b_list.append(b_val)

    # No obstacles nearby — return nominal action unchanged
    if len(A_list) == 0:
        return [float(v_nom), float(w_nom)]

    A = np.array(A_list, dtype=np.float64)
    b = np.array(b_list, dtype=np.float64)
    n_constraints = len(A_list)

    # QP formulation: minimize ||u - u_nom||_Q
    # Q[1,1]=0.1 gives stronger preference to steer (change w) than brake (change v)
    u = cp.Variable(2)
    u_nom_arr = np.array([v_nom, w_nom], dtype=np.float64)
    Q = np.diag([1.0, 0.5])  # Lab5 originale

    cost = cp.quad_form(u - u_nom_arr, Q)
    objective = cp.Minimize(cost)

    constraints = [
        A @ u <= b,
        u[0] >= v_min,
        u[0] <= v_max,
        u[1] >= -1.0,
        u[1] <= 1.0,
    ]

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.OSQP, verbose=False)
        if prob.status == cp.OPTIMAL:
            return [float(u.value[0]), float(u.value[1])]
        else:
            return [0.0, 0.0]  # emergency stop
    except Exception:
        return [0.0, 0.0]  # emergency stop


# ── Self-check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== SELF-CHECK: cbf.py ===")
    n_pass = 0
    n_total = 3

    # Test 1: no obstacles nearby → action unchanged
    lidar_clear = np.full(20, 5.0, dtype=np.float64)
    result = get_safe_action(0.5, 0.3, lidar_clear)
    if abs(result[0] - 0.5) < 0.01 and abs(result[1] - 0.3) < 0.01:
        print("  Test 1 (clear): PASS")
        n_pass += 1
    else:
        print(f"  Test 1 (clear): FAIL — got {result}")

    # Test 2: obstacle straight ahead → must modify action
    lidar_obstacle = np.full(20, 5.0, dtype=np.float64)
    lidar_obstacle[10] = 0.5  # obstacle at 0.5m straight ahead (center ray)
    result = get_safe_action(0.8, 0.0, lidar_obstacle)
    # CBF should force v down or change w
    changed = abs(result[0] - 0.8) > 0.01 or abs(result[1]) > 0.01
    if changed:
        print(f"  Test 2 (obstacle ahead): PASS — [{result[0]:.3f}, {result[1]:.3f}]")
        n_pass += 1
    else:
        print(f"  Test 2 (obstacle ahead): FAIL — got {result}")

    # Test 3: output ranges are valid
    lidar_random = np.random.uniform(0.0, 5.0, 20)
    result = get_safe_action(0.5, 0.0, lidar_random)
    v_ok = 0.0 <= result[0] <= 1.0
    w_ok = -1.0 <= result[1] <= 1.0
    if v_ok and w_ok:
        print(f"  Test 3 (ranges): PASS — [{result[0]:.3f}, {result[1]:.3f}]")
        n_pass += 1
    else:
        print(f"  Test 3 (ranges): FAIL — v={v_ok}, w={w_ok}, got {result}")

    print(f"\n  {n_pass}/{n_total} tests passed")
