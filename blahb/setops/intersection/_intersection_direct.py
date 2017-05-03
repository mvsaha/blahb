import numpy as np
import numba

from numblahb.utils import lex_less_Nd, eq_Nd


@numba.njit
def _intersection_direct_1d(a, b):
    na, ndim = a.shape
    ib, nb = 0, b.shape[0]
    assert b.shape[1] == ndim
    take_a = np.empty(na, dtype=numba.boolean)
    for ia in range(na):
        while b[ib, 0] < a[ia, 0]:
            ib += 1
            if ib == nb:
                take_a[ia:] = False
                return take_a
        if a[ia, 0] == b[ib, 0]:
            take_a[ia] = True
        else:
            take_a[ia] = False
    return take_a


@numba.njit
def _intersection_direct_2d(a, b):
    na, ndim = a.shape
    ib, nb = 0, b.shape[0]
    assert b.shape[1] == ndim
    take_a = np.empty(na, dtype=numba.boolean)
    for ia in range(na):
        while (b[ib, 0] < a[ia, 0] or
            (b[ib, 0] == a[ia, 0] and (b[ib, 1] < a[ia, 1]))):
            ib += 1
            if ib == nb:
                take_a[ia:] = False
                return take_a
        if (a[ia, 0] == b[ib, 0] and a[ia, 1] == b[ib, 1]):
            take_a[ia] = True
        else:
            take_a[ia] = False
    return take_a


@numba.njit
def _intersection_direct_3d(a, b):
    na, ndim = a.shape
    ib, nb = 0, b.shape[0]
    assert b.shape[1] == ndim
    take_a = np.empty(na, dtype=numba.boolean)
    for ia in range(na):
        while (b[ib, 0] < a[ia, 0] or
            (b[ib, 0] == a[ia, 0] and (b[ib, 1] < a[ia, 1] or
                (b[ib, 1] == a[ia, 1] and b[ib, 2] < a[ia, 2])))):
            ib += 1
            if ib == nb:
                take_a[ia:] = False
                return take_a
        if (a[ia, 0] == b[ib, 0] and a[ia, 1] == b[ib, 1] and
            a[ia, 2] == b[ib, 2]):
            take_a[ia] = True
        else:
            take_a[ia] = False
    return take_a


@numba.njit
def _intersection_direct_4d(a, b):
    na, ndim = a.shape
    ib, nb = 0, b.shape[0]
    assert b.shape[1] == ndim
    take_a = np.empty(na, dtype=numba.boolean)
    for ia in range(na):
        while (b[ib, 0] < a[ia, 0] or
            (b[ib, 0] == a[ia, 0] and (b[ib, 1] < a[ia, 1] or
                (b[ib, 1] == a[ia, 1] and (b[ib, 2] < a[ia, 2] or
                    (b[ib, 2] == a[ia, 2] and b[ib, 3] < a[ia, 3])))))):
            ib += 1
            if ib == nb:
                take_a[ia:] = False
                return take_a
        if (a[ia, 0] == b[ib, 0] and a[ia, 1] == b[ib, 1] and
            a[ia, 2] == b[ib, 2] and a[ia, 3] == b[ib, 3]):
            take_a[ia] = True
        else:
            take_a[ia] = False
    return take_a


@numba.njit
def _intersection_direct_Nd(a, b):
    """Find the set intersection between two IndexSets in linear time."""
    na, ndim = a.shape
    ib, nb = 0, b.shape[0]
    assert b.shape[1] == ndim
    take_a = np.empty(na, dtype=numba.boolean)
    for ia in range(na):
        while lex_less_Nd(b[ib], a[ia]):
            ib += 1
            if ib == nb:  # If we run out of b coordinates, then we are done
                take_a[ia:] = False
                return take_a
        if eq_Nd(b[ib], a[ia]):
            take_a[ia] = True
        else:
            take_a[ia] = False
    return take_a


