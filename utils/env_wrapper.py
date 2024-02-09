from typing import Tuple
import gym
from typing import Any, Dict
from gym.core import Env
from gym.wrappers.time_limit import TimeLimit
import numpy as np


class EnvGuard(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        lb: float = -1.0,
        ub: float = 1.0,
        disable_reset_wrapper: bool = False,
    ):
        super().__init__(env)
        self._env = env
        self._need_reset = True
        self._lb = lb
        self._ub = ub
        self._disable_reset_wrapper = disable_reset_wrapper

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, dict]:
        assert not self._need_reset
        assert action.shape == self._env.action_space.shape
        assert (self._lb <= action).all() and (action <= self._ub).all()

        (next_state, reward, done, info) = self._env.step(action)

        if done:
            self._need_reset = True

        return (next_state, reward, done, info)

    def reset(self, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        self._need_reset = False

        rlt = self._env.reset(*args, **kwargs)

        if self._disable_reset_wrapper:
            return rlt

        if isinstance(rlt, tuple):
            return rlt
        else:
            return (rlt, dict())


class TimeoutExtractor(gym.Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self._env = env
        _e = env

        # Check TimeLimitWrapper is alread enforced
        _hasTimeoutWrapper = False
        while _e is not None:
            _hasTimeoutWrapper = isinstance(_e, TimeLimit)
            if _hasTimeoutWrapper:
                break
            _e = getattr(_e, "env", None)
        assert _hasTimeoutWrapper, "TimeoutExtractor must be used with TimeLimitWrapper"

    def step(self, action):
        next_state, rwd, done, info = self._env.step(action)
        if not done:
            terminal = False
            timeout = False
        elif "TimeLimit.truncated" not in info:
            terminal = True
            timeout = False
        else:
            truncated = info["TimeLimit.truncated"]
            terminal = not truncated
            timeout = False if terminal else True

        return (next_state, rwd, timeout, terminal, info)
