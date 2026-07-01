"""Shared constants for inference on CoppeliaSim.

ALL constants must match reward.py and limo_env.py exactly.
LIDAR_FOV is corrected to 120deg (2pi/3) — the real scene value,
NOT the old Lab5 value of np.pi*0.8 (144deg).
"""

import numpy as np

# ── LiDAR ──────────────────────────────────────────────────────────────────
LIDAR_MAX_DIST = 5.0       # max range in meters
LIDAR_FOV      = 2 * np.pi / 3   # 120 degrees — verified from CoppeliaSim scene
N_LIDAR        = 20        # number of rays

# ── Observation normalization ──────────────────────────────────────────────
DIST_NORM_MAX = 7.0        # max distance for goal distance normalization

# ── Robot ──────────────────────────────────────────────────────────────────
ROBOT_INIT_POS = (-1.7, 0.74, 0.0)   # x, y, theta — from Lab5 scene

# ── Termination ────────────────────────────────────────────────────────────
GOAL_DIST  = 0.1           # meters — robot within this distance = goal reached
CRASH_DIST = 0.05          # meters — min lidar reading below this = crash
                           # 0.05 = 5cm (permissive for CoppeliaSim, avoids false
                           # positives from grazing LiDAR readings; training
                           # env used 0.15 but real scene needs more tolerance)
MAX_STEPS  = 500           # max steps per episode before timeout

# ── Reward ─────────────────────────────────────────────────────────────────
GOAL_BONUS   = 10.0
CRASH_COST   = 10.0
STEP_PENALTY = -0.01

# ── CBF ────────────────────────────────────────────────────────────────────
# Larger lookahead + moderate safety: CBF sees obstacles earlier, steers not brakes.
# L=0.15 gives A_w=2*L*y_obs enough authority to steer around side obstacles.
# R=0.10 gives ~20cm stopping distance for frontal obstacles.
# Q_w=0.1 in QP cost (cbf.py) makes steering 10x cheaper than braking.
CBF_LOOKAHEAD = 0.35       # lookahead distance (m) — Lab5 originale
CBF_R_SAFE    = 0.30       # safe radius (m) — Lab5 originale
CBF_GAMMA     = 2.0        # decay rate — Lab5 originale
CBF_V_MIN     = 0.1        # Lab5 originale
CBF_V_MAX     = 0.7        # Lab5 originale

# ── Default checkpoint paths (relative to project root) ────────────────────
DEFAULT_PPO_CHECKPOINT     = "results/checkpoints_ppo/ppo_limo_final.zip"
DEFAULT_DREAMER_CHECKPOINT = "results/checkpoints_dreamer/LimoCustomEnv_final_60k.pth"
DEFAULT_DREAMER_CONFIG     = "limo-dreamer.yml"

# ── Default scene path (relative to inference/ dir) ────────────────────────
DEFAULT_SCENE = "scenes/easy.ttt"
