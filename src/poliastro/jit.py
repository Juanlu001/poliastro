# coding: utf-8
"""Just-in-time compiler and accelerated utility functions.

Wraps numba if it is available as a module, uses an identity
decorator instead.

"""
import warnings
import inspect

import numpy as np


def ijit(first=None, *args, **kwargs):
    """Identity JIT, returns unchanged function.

    """
    def _jit(f):
        return f

    if inspect.isfunction(first):
        return first
    else:
        return _jit


try:
    import numba
    jit = numba.njit
except ImportError:
    warnings.warn("Could not import numba package. All poliastro "
                  "functions will work properly but the CPU intensive "
                  "algorithms will be slow. Consider installing numba to "
                  "boost performance")
    jit = ijit


@jit('f8(f8[:])')
def norm(vec):
    return np.sqrt(np.dot(vec, vec))


@jit('f8[:](f8[:], f8[:])')
def cross(vec1, vec2):
    """ Calculate the cross product of two 3d vectors. """
    result = np.zeros_like(vec1)
    result[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1]
    result[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2]
    result[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    return result
