import numpy as np
from gym.spaces import Box
from functools import wraps


def make_box_env(bounds, dtype=np.float32):
    low, high = zip(*bounds)
    return Box(low=np.array(low, dtype=dtype), high=np.array(high, dtype=dtype))

def slice_box_env(box, start=None, end=None):
    s = slice(start, end)
    return Box(low=box.low[s], high=box.high[s])

def concat_box_envs(*envs):
    return Box(low=np.concatenate([e.low for e in envs]),
               high=np.concatenate([e.high for e in envs]))

def duplicate_box_env(env, n):
    return Box(low=np.tile(env.low, n),
               high=np.tile(env.high, n))

def box_env_ndims(env):
    return len(env.low)

def cache_obs(method):
    """ Cache the observation of the BS, updated every step. """
    t = None
    cache = {}
    @wraps(method)
    def wrapper(self, other=None):
        nonlocal t
        if other is None:
            args = self,
        else:
            args = self, other
        if self._time != t:
            cache.clear()
            t = self._time
        elif args in cache:
            return cache[args]
        cache[args] = ret = method(*args)
        if len(args) == 2:  # cache for the other
            cache[args[1], args[0]] = ret[1], ret[0]
        return ret
    return wrapper
