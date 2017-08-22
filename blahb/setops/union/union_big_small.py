import numpy as np
import numba

from ...utils import enlarge_array, lex_less_Nd, eq_Nd


@numba.njit(nogil=True)
def _union_big_small_1d(a, b):
    na, nb = a.shape[0], b.shape[0]
    big_capacity = na + (nb // 2) if na > nb else nb + (na // 2)
    big = np.empty(big_capacity, dtype=numba.boolean)
    small = np.empty(nb, dtype=numba.boolean)
    n = np.uint32(0)
    ia, ib = np.uint32(0), np.uint32(0)
    
    while ia < na:
        while ib < nb and b[ib, 0] < a[ia, 0]:
            small[ib] = True
            ib += 1
            if n == big_capacity:
                big = enlarge_array(big, min(na + nb, big_capacity * 2))
                big_capacity = big.size
            big[n] = False
            n += 1
            # assert np.sum(small[:ib]) + np.sum(big[:n]) == n
        
        if ib == nb:  # No more b values, write all remaining a
            n_a_left = na - ia
            if n + n_a_left > big_capacity:
                big_capacity = n + n_a_left
                big = enlarge_array(big, big_capacity)
            big[n:n + n_a_left] = True
            n = n + n_a_left
            return big[:n], small
        
        if n == big_capacity:
            big = enlarge_array(big, min(na + nb, big_capacity * 2))
            big_capacity = big.size
        
        big[n] = True
        n += 1
        if a[ia, 0] == b[ib, 0]:
            small[ib] = False
            ib += 1
        ia += 1
    
    if ib < nb:
        n_b_left = nb - ib
        if n + n_b_left > big_capacity:
            big = enlarge_array(big, n + n_b_left)
        big[n:n + n_b_left] = False
        n = n + n_b_left
        small[ib:] = True
    return big[:n], small


@numba.njit(nogil=True)
def _union_big_small_2d(a, b):
    na, nb = a.shape[0], b.shape[0]
    big_capacity = na + (nb // 2) if na > nb else nb + (na // 2)
    big = np.empty(big_capacity, dtype=numba.boolean)
    small = np.empty(nb, dtype=numba.boolean)
    n = np.uint32(0)
    ia, ib = np.uint32(0), np.uint32(0)
    
    while ia < na:
        while ib < nb and (b[ib, 0] < a[ia, 0] or
                (b[ib, 0] == a[ia, 0] and (b[ib, 1] < a[ia, 1]))):
            small[ib] = True
            ib += 1
            if n == big_capacity:
                big = enlarge_array(big, min(na + nb, big_capacity * 2))
                big_capacity = big.size
            big[n] = False
            n += 1
            # assert np.sum(small[:ib]) + np.sum(big[:n]) == n
        
        if ib == nb:  # No more b values, write all remaining a
            n_a_left = na - ia
            if n + n_a_left > big_capacity:
                big_capacity = n + n_a_left
                big = enlarge_array(big, big_capacity)
            big[n:n + n_a_left] = True
            n = n + n_a_left
            return big[:n], small
        
        if n == big_capacity:
            big = enlarge_array(big, min(na + nb, big_capacity * 2))
            big_capacity = big.size
        
        big[n] = True
        n += 1
        if a[ia, 0] == b[ib, 0] and a[ia, 1] == b[ib, 1]:
            small[ib] = False
            ib += 1
        ia += 1
    
    if ib < nb:
        n_b_left = nb - ib
        if n + n_b_left > big_capacity:
            big = enlarge_array(big, n + n_b_left)
        big[n:n + n_b_left] = False
        n = n + n_b_left
        small[ib:] = True
    return big[:n], small


@numba.njit(nogil=True)
def _union_big_small_3d(a, b):
    na, nb = a.shape[0], b.shape[0]
    big_capacity = na + (nb // 2) if na > nb else nb + (na // 2)
    big = np.empty(big_capacity, dtype=numba.boolean)
    small = np.empty(nb, dtype=numba.boolean)
    n = np.uint32(0)
    ia, ib = np.uint32(0), np.uint32(0)
    
    while ia < na:
        while ib < nb and (b[ib, 0] < a[ia, 0] or
               (b[ib, 0] == a[ia, 0] and
                    (b[ib, 1] < a[ia, 1] or
                         (b[ib, 1] == a[ia, 1] and
                              (b[ib, 2] < a[ia, 2]))))):
            small[ib] = True
            ib += 1
            if n == big_capacity:
                big = enlarge_array(big, min(na + nb, big_capacity * 2))
                big_capacity = big.size
            big[n] = False
            n += 1
            # assert np.sum(small[:ib]) + np.sum(big[:n]) == n
        
        if ib == nb:  # No more b values, write all remaining a
            n_a_left = na - ia
            if n + n_a_left > big_capacity:
                big_capacity = n + n_a_left
                big = enlarge_array(big, big_capacity)
            big[n:n + n_a_left] = True
            n = n + n_a_left
            return big[:n], small
        
        if n == big_capacity:
            big = enlarge_array(big, min(na + nb, big_capacity * 2))
            big_capacity = big.size
        
        big[n] = True
        n += 1
        if (a[ia, 0] == b[ib, 0] and
            a[ia, 1] == b[ib, 1] and
            a[ia, 2] == b[ib, 2]):
            small[ib] = False
            ib += 1
        ia += 1
    
    if ib < nb:
        n_b_left = nb - ib
        if n + n_b_left > big_capacity:
            big = enlarge_array(big, n + n_b_left)
        big[n:n + n_b_left] = False
        n = n + n_b_left
        small[ib:] = True
    return big[:n], small


@numba.njit(nogil=True)
def _union_big_small_4d(a, b):
    na, nb = a.shape[0], b.shape[0]
    big_capacity = na + (nb // 2) if na > nb else nb + (na // 2)
    big = np.empty(big_capacity, dtype=numba.boolean)
    small = np.empty(nb, dtype=numba.boolean)
    n = np.uint32(0)
    ia, ib = np.uint32(0), np.uint32(0)
    
    while ia < na:
        while ib < nb and (b[ib, 0] < a[ia, 0] or
            (b[ib, 0] == a[ia, 0] and (b[ib, 1] < a[ia, 1] or
                (b[ib, 1] == a[ia, 1] and (b[ib, 2] < a[ia, 2] or
                    (b[ib, 2] == a[ia, 2] and b[ib, 3] < a[ia, 3])))))):
            small[ib] = True
            ib += 1
            if n == big_capacity:
                big = enlarge_array(big, min(na + nb, big_capacity * 2))
                big_capacity = big.size
            big[n] = False
            n += 1
            # assert np.sum(small[:ib]) + np.sum(big[:n]) == n
        
        if ib == nb:  # No more b values, write all remaining a
            n_a_left = na - ia
            if n + n_a_left > big_capacity:
                big_capacity = n + n_a_left
                big = enlarge_array(big, big_capacity)
            big[n:n + n_a_left] = True
            n = n + n_a_left
            return big[:n], small
        
        if n == big_capacity:
            big = enlarge_array(big, min(na + nb, big_capacity * 2))
            big_capacity = big.size
        
        big[n] = True
        n += 1
        if (a[ia, 0] == b[ib, 0] and
            a[ia, 1] == b[ib, 1] and
            a[ia, 2] == b[ib, 2] and
            a[ia, 3] == b[ib, 3]):
            small[ib] = False
            ib += 1
        ia += 1
    
    if ib < nb:
        n_b_left = nb - ib
        if n + n_b_left > big_capacity:
            big = enlarge_array(big, n + n_b_left)
        big[n:n + n_b_left] = False
        n = n + n_b_left
        small[ib:] = True
    return big[:n], small


@numba.njit(nogil=True)
def _union_big_small_Nd(a, b):
    """Find locations in a or b.
    
    Arguments
    ---------
    a, b : 2d int32 arrays
        The coordinate locations to merge
    
    Returns
    -------
    (big, small)
    `big` is an array that is the final size of the union of a and b. It is
    True where values of the union result are found in a (they could also be
    found in b). It is False where a location is only found in b. `small` is
    an array of bool flags that that is True where b should contribute to the
    false locations in `big`.
    """
    na, nb = a.shape[0], b.shape[0]
    big_capacity = na + (nb // 2) if na > nb else nb + (na // 2)
    
    big = np.empty(big_capacity, dtype=numba.boolean)
    small = np.empty(nb, dtype=numba.boolean)  # Size won't change
    
    # The number of values written to big
    # (alternatively: the next position to write to big)
    n = np.uint32(0)
    
    # Cursors into a and b
    ia, ib = np.uint32(0), np.uint32(0)
    
    while ia < na:
        while ib < nb and lex_less_Nd(b[ib], a[ia]):
            small[ib] = True
            ib += 1
            if n == big_capacity:
                big = enlarge_array(big, min(na + nb, big_capacity * 2))
                big_capacity = big.size
            big[n] = False
            n += 1
            # assert np.sum(small[:ib]) + np.sum(big[:n]) == n
        
        if ib == nb:  # No more b values, write all remaining a
            n_a_left = na - ia
            if n + n_a_left > big_capacity:
                big_capacity = n + n_a_left
                big = enlarge_array(big, big_capacity)
            big[n:n + n_a_left] = True
            n = n + n_a_left
            # Invariants:
            # assert np.sum(~big[:n]) == np.sum(small)
            # assert np.sum(big[:n]) == na
            # assert np.sum(small) + np.sum(big[:n]) == n
            return big[:n], small
        
        if n == big_capacity:
            big = enlarge_array(big, min(na + nb, big_capacity * 2))
            big_capacity = big.size
        
        big[n] = True
        n += 1
        if eq_Nd(a[ia], b[ib]):
            small[ib] = False
            ib += 1
        ia += 1
    
    if ib < nb:
        n_b_left = nb - ib
        if n + n_b_left > big_capacity:
            big = enlarge_array(big, n + n_b_left)
        big[n:n + n_b_left] = False
        n = n + n_b_left
        small[ib:] = True
    
    # Invariants:
    # assert np.sum(~big[:n]) == np.sum(small)
    # assert np.sum(big[:n]) == na
    # assert np.sum(small) + np.sum(big[:n]) == n
    return big[:n], small


@numba.njit(nogil=True)
def union_big_small_(a_loc, b_loc):
    ndim = a_loc.shape[1]
    if not ndim == b_loc.shape[1]:
        raise AssertionError("a_loc and b_loc must have the same number of "
                             "columns")
    
    if ndim == 1:
        return _union_big_small_1d(a_loc, b_loc)
    elif ndim == 2:
        return _union_big_small_2d(a_loc, b_loc)
    elif ndim == 3:
        return _union_big_small_3d(a_loc, b_loc)
    elif ndim == 4:
        return _union_big_small_4d(a_loc, b_loc)
    else:
        return _union_big_small_Nd(a_loc, b_loc)


@numba.njit(cache=True)
def merge_union_big_small_results(a_loc, b_loc, big, small, out=None):
    n_total = big.size
    ndim = a_loc.shape[1]
    assert ndim == b_loc.shape[1]
    
    if out is None:
        together = np.empty((n_total, ndim), a_loc.dtype)
    else:
        together = out
    
    together[big] = a_loc
    i_small = 0
    for i_big in range(n_total):
        if not big[i_big]:
            while not small[i_small]:
                i_small += 1
            for dim in range(ndim):
                together[i_big, dim] = b_loc[i_small, dim]
            i_small += 1
    return together


