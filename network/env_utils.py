import numpy as np
from gymnasium.spaces import Box
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
    def wrapper(self, *args):
        nonlocal t
        key = (self, *args)
        if self._time != t:
            cache.clear()
            t = self._time
        elif key in cache:
            return cache[key]
        ret = cache[key] = method(self, *args)
        return ret
    return wrapper
