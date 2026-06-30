# SOURCE: NaturalDreamer (https://github.com/InexperiencedMe/NaturalDreamer) — da adattare
import gymnasium as gym
import numpy as np


def getEnvProperties(env):
    assert isinstance(env.action_space, gym.spaces.Box), "Sorry, supporting only continuous action space for now"
    observationShape = env.observation_space.shape
    actionSize = env.action_space.shape[0]
    actionLow = env.action_space.low.tolist()
    actionHigh = env.action_space.high.tolist()
    return observationShape, actionSize, actionLow, actionHigh


class GymPixelsProcessingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        observationSpace = self.observation_space
        newObsShape = observationSpace.shape[-1:] + observationSpace.shape[:2]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=newObsShape, dtype=np.float32)

    def observation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))/255.0
        return observation


class CleanGymWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return obs


class CleanEnvWrapper:
    """Adapts gymnasium Env (5-tuple step) to NaturalDreamer's 3-tuple contract.

    NaturalDreamer's environmentInteraction expects:
        obs = env.reset(seed=...)              # single obs, not (obs, info)
        next_obs, reward, done = env.step(a)   # 3 values, done = terminated|truncated

    Strips info dicts, combines terminated/truncated, exposes spaces.

    Preconditions:
        - Inner env follows gymnasium.Env interface (step → 5-tuple, reset → (obs, info)).
    Postconditions:
        - step returns (obs, reward, done) where done = terminated or truncated.
        - reset returns obs only (discards info).

    Args:
        env: A gymnasium.Env instance (e.g., LimoCustomEnv).

    Example:
        >>> from limo_env import LimoCustomEnv
        >>> raw = LimoCustomEnv()
        >>> env = CleanEnvWrapper(raw)
        >>> obs = env.reset(seed=42)
        >>> obs.shape
        (23,)
        >>> next_obs, reward, done = env.step(np.array([0.5, 0.0]))
    """

    def __init__(self, env):
        self.env = env

    def reset(self, seed=None):
        return self.env.reset(seed=seed)[0]

    def step(self, action):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        return obs, reward, bool(terminated or truncated)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space


if __name__ == '__main__':
    # ── Self-check: CleanEnvWrapper with LimoCustomEnv ─────────────────
    print('=== SELF-CHECK: envs.py (CleanEnvWrapper) ===')

    import sys as _sys
    import os as _os
    _root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from limo_env import LimoCustomEnv
    import numpy as _np

    # 1. Instantiate wrapped env
    raw = LimoCustomEnv(n_obstacles_range=(2, 4), obs_radius_range=(0.10, 0.15))
    env = CleanEnvWrapper(raw)

    # 2. reset() returns single obs array (not tuple)
    obs = env.reset(seed=42)
    passed_1 = isinstance(obs, _np.ndarray) and obs.shape == (23,)
    print(f'reset() returns obs shape {obs.shape} (expected (23,))'
          f' — {"PASS" if passed_1 else "FAIL"}')
    assert passed_1, f'reset() returned {type(obs)}, shape {obs.shape}'

    # 3. step() returns exactly 3 values, done is bool
    action = _np.array([0.5, 0.0], dtype=_np.float32)
    result = env.step(action)
    passed_2 = (isinstance(result, tuple) and len(result) == 3
                and isinstance(result[0], _np.ndarray)
                and isinstance(result[2], bool))
    print(f'step() returns {len(result)} values, done is {type(result[2]).__name__}'
          f' (expected 3, bool) — {"PASS" if passed_2 else "FAIL"}')
    assert passed_2, f'step returned {len(result)} values, done type={type(result[2])}'

    # 4. Property extraction matches expected spaces
    obs_shape = env.observation_space.shape
    act_size  = env.action_space.shape[0]
    act_low   = env.action_space.low.tolist()
    act_high  = env.action_space.high.tolist()
    passed_3 = (obs_shape == (23,)
                and act_size == 2
                and act_low == [0.0, -1.0]
                and act_high == [1.0, 1.0])
    print(f'spaces: obs={obs_shape}, act_size={act_size}, low={act_low}, high={act_high}')
    print(f'expected: obs=(23,), act_size=2, low=[0.0,-1.0], high=[1.0,1.0]'
          f' — {"PASS" if passed_3 else "FAIL"}')
    assert passed_3, f'Space mismatch'

    print(f'All tests: {"PASS" if all([passed_1, passed_2, passed_3]) else "FAIL"}')
