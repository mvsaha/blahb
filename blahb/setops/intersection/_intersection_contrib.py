import numpy as np
import numba

from numblahb.utils import lex_less_Nd, eq_Nd


@numba.njit
def _intersection_contrib_1d(a_loc, b_loc):
    """Find the contributing pixels from a_loc and b_loc to the
    intersection of a_loc and b_loc locations.
    
    Arguments
    ---------
    a_loc, b_loc : 2d array
        These must be lexsorted. Each row corresponds to a
        coordinates. Each column corresponds to a dimension.
    
    Returns
    -------
    a_contrib, b_contrib : 1d boolean arrays with a size matching
    the lowest dimension of a_loc and b_loc, respectively. It is
    set to True if the corresponding row in the coordinates
    contributes to the intersection of the locations. Both arrays,
    while different in size, will have the same number of True
    values.
    """
    n_a, ndim = a_loc.shape
    n_b = b_loc.shape[0]
    assert b_loc.shape[1] == ndim
    a_contrib = np.empty(n_a, dtype=numba.boolean)
    b_contrib = np.empty(n_b, dtype=numba.boolean)
    i_b = 0
    
    for i_a in range(n_a):
        while b_loc[i_b, 0] < a_loc[i_a, 0]:
            b_contrib[i_b] = False
            i_b += 1
            if i_b == n_b:
                a_contrib[i_a:] = False
                return a_contrib, b_contrib
        
        if a_loc[i_a, 0] == b_loc[i_b, 0]:
            a_contrib[i_a] = True
            b_contrib[i_b] = True
            i_b += 1
            if i_b == n_b:
                a_contrib[i_a + 1:] = False
                return a_contrib, b_contrib
        else:
            a_contrib[i_a] = False
    if i_b < n_b:
        b_contrib[i_b:] = False
    return a_contrib, b_contrib


@numba.njit
def _intersection_contrib_2d(a_loc, b_loc):
    """Find the contributing pixels from a_loc and b_loc to the
    intersection of a_loc and b_loc locations.

    Arguments
    ---------
    a_loc, b_loc : 2d array
        These must be lexsorted. Each row corresponds to a
        coordinates. Each column corresponds to a dimension.

    Returns
    -------
    a_contrib, b_contrib : 1d boolean arrays with a size matching
    the lowest dimension of a_loc and b_loc, respectively. It is
    set to True if the corresponding row in the coordinates
    contributes to the intersection of the locations. Both arrays,
    while different in size, will have the same number of True
    values.
    """
    n_a, ndim = a_loc.shape
    n_b = b_loc.shape[0]
    assert b_loc.shape[1] == ndim
    
    a_contrib = np.empty(n_a, dtype=numba.boolean)
    b_contrib = np.empty(n_b, dtype=numba.boolean)
    i_b = 0
    
    for i_a in range(n_a):
        while (b_loc[i_b, 0] < a_loc[i_a, 0] or
                   (b_loc[i_b, 0] == a_loc[i_a, 0] and
                        (b_loc[i_b, 1] < a_loc[i_a, 1]))):
            b_contrib[i_b] = False
            i_b += 1
            if i_b == n_b:
                a_contrib[i_a:] = False
                return a_contrib, b_contrib
        
        if a_loc[i_a, 0] == b_loc[i_b, 0] and a_loc[i_a, 1] == b_loc[i_b, 1]:
            a_contrib[i_a] = True
            b_contrib[i_b] = True
            i_b += 1
            if i_b == n_b:
                a_contrib[i_a + 1:] = False
                return a_contrib, b_contrib
        else:
            a_contrib[i_a] = False
    if i_b < n_b:
        b_contrib[i_b:] = False
    return a_contrib, b_contrib


@numba.njit
def _intersection_contrib_3d(a_loc, b_loc):
    """Find the contributing pixels from a_loc and b_loc to the
    intersection of a_loc and b_loc locations.

    Arguments
    ---------
    a_loc, b_loc : 2d array
        These must be lexsorted. Each row corresponds to a
        coordinates. Each column corresponds to a dimension.

    Returns
    -------
    a_contrib, b_contrib : 1d boolean arrays with a size matching
    the lowest dimension of a_loc and b_loc, respectively. It is
    set to True if the corresponding row in the coordinates
    contributes to the intersection of the locations. Both arrays,
    while different in size, will have the same number of True
    values.
    """
    n_a, ndim = a_loc.shape
    n_b = b_loc.shape[0]
    assert b_loc.shape[1] == ndim
    
    a_contrib = np.empty(n_a, dtype=numba.boolean)
    b_contrib = np.empty(n_b, dtype=numba.boolean)
    i_b = 0
    
    for i_a in range(n_a):
        while (b_loc[i_b, 0] < a_loc[i_a, 0] or
                   (b_loc[i_b, 0] == a_loc[i_a, 0] and
                        (b_loc[i_b, 1] < a_loc[i_a, 1] or
                             (b_loc[i_b, 1] == a_loc[i_a, 1] and
                                  (b_loc[i_b, 2] < a_loc[i_a, 2]))))):
            b_contrib[i_b] = False
            i_b += 1
            if i_b == n_b:
                a_contrib[i_a:] = False
                return a_contrib, b_contrib
        
        if (a_loc[i_a, 0] == b_loc[i_b, 0] and
                    a_loc[i_a, 1] == b_loc[i_b, 1] and
                    a_loc[i_a, 2] == b_loc[i_b, 2]):
            a_contrib[i_a] = True
            b_contrib[i_b] = True
            i_b += 1
            if i_b == n_b:
                a_contrib[i_a + 1:] = False
                return a_contrib, b_contrib
        else:
            a_contrib[i_a] = False
    if i_b < n_b:
        b_contrib[i_b:] = False
    return a_contrib, b_contrib


@numba.njit
def _intersection_contrib_4d(a_loc, b_loc):
    n_a, ndim = a_loc.shape
    n_b = b_loc.shape[0]
    assert b_loc.shape[1] == ndim
    
    a_contrib = np.empty(n_a, dtype=numba.boolean)
    b_contrib = np.empty(n_b, dtype=numba.boolean)
    i_b = 0
    
    for i_a in range(n_a):
        while (b_loc[i_b, 0] < a_loc[i_a, 0] or
                   (b_loc[i_b, 0] == a_loc[i_a, 0] and
                        (b_loc[i_b, 1] < a_loc[i_a, 1] or
                             (b_loc[i_b, 1] == a_loc[i_a, 1] and
                                  (b_loc[i_b, 2] < a_loc[i_a, 2] or
                                       (b_loc[i_b, 2] == a_loc[i_a, 2] and
                                                b_loc[i_b, 3] < a_loc[
                                                i_a, 3])))))):
            b_contrib[i_b] = False
            i_b += 1
            if i_b == n_b:
                a_contrib[i_a:] = False
                return a_contrib, b_contrib
        
        if (a_loc[i_a, 0] == b_loc[i_b, 0] and
                    a_loc[i_a, 1] == b_loc[i_b, 1] and
                    a_loc[i_a, 2] == b_loc[i_b, 2] and
                    a_loc[i_a, 3] == b_loc[i_b, 3]):
            a_contrib[i_a] = True
            b_contrib[i_b] = True
            i_b += 1
            if i_b == n_b:
                a_contrib[i_a + 1:] = False
                return a_contrib, b_contrib
        else:
            a_contrib[i_a] = False
    if i_b < n_b:
        b_contrib[i_b:] = False
    return a_contrib, b_contrib


@numba.njit
def _intersection_contrib_Nd(a_loc, b_loc):
    n_a, ndim = a_loc.shape
    n_b = b_loc.shape[0]
    assert b_loc.shape[1] == ndim
    
    a_contrib = np.empty(n_a, dtype=numba.boolean)
    b_contrib = np.empty(n_b, dtype=numba.boolean)
    i_b = 0
    
    for i_a in range(n_a):
        while lex_less_Nd(b_loc[i_b], a_loc[i_a]):
            b_contrib[i_b] = False
            i_b += 1
            if i_b == n_b:
                a_contrib[i_a:] = False
                return a_contrib, b_contrib
        
        if eq_Nd(a_loc[i_a], b_loc[i_b]):
            a_contrib[i_a] = True
            b_contrib[i_b] = True
            i_b += 1
            if i_b == n_b:
                a_contrib[i_a + 1:] = False
                return a_contrib, b_contrib
        else:
            a_contrib[i_a] = False
    
    if i_b < n_b:
        b_contrib[i_b:] = False
    
    return a_contrib, b_contrib