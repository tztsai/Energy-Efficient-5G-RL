import io
import os
import sys
import time
import enum
import atexit
import logging
import argparse
import numpy as np
import pandas as pd
from config import DEBUG
from pathlib import Path
from functools import wraps
from tqdm import tqdm, trange
# import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
from typing import List, Tuple, Dict, Any, Callable, Union, Sequence, Optional

np.set_printoptions(precision=3)

logger = logging.getLogger('main')

debug = logger.debug
info = logger.info
warn = logger.warning

def set_log_level(level):
    logging.basicConfig(
        # format='%(levelname)s: %(message)s',
        format='%(message)s',
        level=level)
    logger.setLevel(level.upper())

def dB2lin(dB):
    return 10 ** (dB / 10)

def lin2dB(lin):
    return 10 * np.log10(lin) if lin > 1e-10 else -100

def kwds_str(**kwds):
    return ', '.join(f'{k}={v}' for k, v in kwds.items())

def onehot_vec(n, k):
    v = np.zeros(n, dtype=np.float32)
    v[k] = 1
    return v

def onehot_dict(name, n, k):
    return OrderedDict(('{}_{}'.format(name, i), 1 if i == k else 0) for i in range(n))

def first(iterable, k=None):
    if k is None:
        return next(iter(iterable))
    else:
        it = iter(iterable)
        return (next(it) for _ in range(k))


class Profile:
    debug_counts, debug_times = defaultdict(int), defaultdict(float)

    @classmethod
    def print_debug_exit(cls):
        print('\n{}  COUNT --- TIME COST'.format('-' * 47))
        for name, _ in sorted(cls.debug_times.items(), key=lambda x: -x[1]):
            print(f"{name:<45} : {cls.debug_counts[name]:>6} {cls.debug_times[name]:>10.2f} ms")
    
    def __init__(self, name=''):
        self.name = name

    def __enter__(self):
        self.st = time.time()

    def __exit__(self, *_):
        et = (time.time()-self.st)*1000.
        self.debug_counts[self.name] += 1
        self.debug_times[self.name] += et
        # debug(f"{self.name:>20} : {et:>7.2f} ms")

def timeit(fn, name=None):
    @wraps(fn)
    def wrapper(*args, **kwds):
        with Profile(name or fn_name(fn)):
            return fn(*args, **kwds)
    def fn_name(fn):
        s = str(fn)
        if s[0] == '<':
            s = s.split(None, 2)[1]
        return s
    return wrapper

if not DEBUG:
    timeit = lambda fn: fn
else:
    atexit.register(Profile.print_debug_exit)