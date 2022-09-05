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

def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c

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
        if len(args) == 2:  # observe_other is symmetric
            cache[args[1], args[0]] = ret
        return ret
    return wrapper
