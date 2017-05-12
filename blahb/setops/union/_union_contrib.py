import numpy as np
import numba

from ...utils import enlarge_array, enlarge_mat, lex_less_Nd, eq_Nd
from ...setops.utils import append_remaining


@numba.njit
def _union_contrib_1d(a, b):
    n_a, ndim = a.shape
    n_b = b.shape[0]
    n_c = 0  # Number of elements written to sz_c
    i_b = 0
    
    # sz_c = n_a + (n_b // 2) if n_a > n_b else n_b + (n_a // 2)
    sz_c = max(n_a, n_b)
    a_contrib = np.zeros(sz_c, dtype=numba.boolean)
    b_contrib = np.zeros(sz_c, dtype=numba.boolean)
    
    for i_a in range(n_a):
        while b[i_b, 0] < a[i_a, 0]:
            if n_c >= sz_c:
                sz_c = min(sz_c * 2, n_a + n_b)
                a_contrib = enlarge_array(a_contrib, sz_c)
                b_contrib = enlarge_array(b_contrib, sz_c)
            
            a_contrib[n_c] = False
            b_contrib[n_c] = True
            
            i_b += 1
            n_c += 1
            
            # If we reach the end of b before a, then write the rest of a
            # Note: This is BEFORE we have written the current i_a values
            if i_b == n_b:
                n_left = n_a - i_a
                sz = n_c + n_left
                if a_contrib.size < sz:
                    a_contrib = enlarge_array(a_contrib, sz)
                    b_contrib = enlarge_array(b_contrib, sz)
                a_contrib[n_c:sz] = True
                b_contrib[n_c:sz] = False
                return a_contrib[:sz], b_contrib[:sz]
        
        if n_c >= sz_c:
            sz_c = min(sz_c * 2, n_a + n_b)
            a_contrib = enlarge_array(a_contrib, sz_c)
            b_contrib = enlarge_array(b_contrib, sz_c)
        
        a_contrib[n_c] = True
        
        if a[i_a, 0] == b[i_b, 0]:
            b_contrib[n_c] = True
            i_b += 1
            n_c += 1
            # If we reach the end of b before a, then write the rest of a
            if i_b == n_b:
                n_left = n_a - i_a - 1
                sz = n_c + n_left
                if a_contrib.size < sz:
                    a_contrib = enlarge_array(a_contrib, sz)
                    b_contrib = enlarge_array(b_contrib, sz)
                a_contrib[n_c:sz] = True
                b_contrib[n_c:sz] = False
                return a_contrib[:sz], b_contrib[:sz]
        else:
            b_contrib[n_c] = False
            n_c += 1
    
    n_left = n_b - i_b
    sz = n_c + n_left
    
    if sz > a_contrib.size:
        a_contrib = enlarge_array(a_contrib, sz)
        b_contrib = enlarge_array(b_contrib, sz)
    a_contrib[n_c:sz] = False
    b_contrib[n_c:sz] = True
    
    return a_contrib[:sz], b_contrib[:sz]


@numba.njit
def _union_contrib_2d(a, b):
    n_a, ndim = a.shape
    n_b = b.shape[0]
    n_c = 0  # Number of elements written to sz_c
    ib = 0
    
    # sz_c = n_a + (n_b // 2) if n_a > n_b else n_b + (n_a // 2)
    sz_c = max(n_a, n_b)
    a_contrib = np.zeros(sz_c, dtype=numba.boolean)
    b_contrib = np.zeros(sz_c, dtype=numba.boolean)
    
    for ia in range(n_a):
        while (b[ib, 0] < a[ia, 0] or
               (b[ib, 0] == a[ia, 0] and b[ib, 1] < a[ia, 1])):
            if n_c >= sz_c:
                sz_c = min(sz_c * 2, n_a + n_b)
                a_contrib = enlarge_array(a_contrib, sz_c)
                b_contrib = enlarge_array(b_contrib, sz_c)
            
            a_contrib[n_c] = False
            b_contrib[n_c] = True
            
            ib += 1
            n_c += 1
            
            # If we reach the end of b before a, then write the rest of a
            # Note: This is BEFORE we have written the current ia values
            if ib == n_b:
                n_left = n_a - ia
                sz = n_c + n_left
                if a_contrib.size < sz:
                    a_contrib = enlarge_array(a_contrib, sz)
                    b_contrib = enlarge_array(b_contrib, sz)
                a_contrib[n_c:sz] = True
                b_contrib[n_c:sz] = False
                return a_contrib[:sz], b_contrib[:sz]
        
        if n_c >= sz_c:
            sz_c = min(sz_c * 2, n_a + n_b)
            a_contrib = enlarge_array(a_contrib, sz_c)
            b_contrib = enlarge_array(b_contrib, sz_c)
        
        a_contrib[n_c] = True
        
        if a[ia, 0] == b[ib, 0] and a[ia, 1] == b[ib, 1]:
            b_contrib[n_c] = True
            ib += 1
            n_c += 1
            # If we reach the end of b before a, then write the rest of a
            if ib == n_b:
                n_left = n_a - ia - 1
                sz = n_c + n_left
                if a_contrib.size < sz:
                    a_contrib = enlarge_array(a_contrib, sz)
                    b_contrib = enlarge_array(b_contrib, sz)
                a_contrib[n_c:sz] = True
                b_contrib[n_c:sz] = False
                return a_contrib[:sz], b_contrib[:sz]
        else:
            b_contrib[n_c] = False
            n_c += 1
    
    n_left = n_b - ib
    sz = n_c + n_left
    
    if sz > a_contrib.size:
        a_contrib = enlarge_array(a_contrib, sz)
        b_contrib = enlarge_array(b_contrib, sz)
    a_contrib[n_c:sz] = False
    b_contrib[n_c:sz] = True
    
    return a_contrib[:sz], b_contrib[:sz]


@numba.njit
def _union_contrib_3d(a, b):
    n_a, ndim = a.shape
    n_b = b.shape[0]
    n_c = 0  # Number of elements written to sz_c
    ib = 0
    
    # sz_c = n_a + (n_b // 2) if n_a > n_b else n_b + (n_a // 2)
    sz_c = max(n_a, n_b)
    a_contrib = np.zeros(sz_c, dtype=numba.boolean)
    b_contrib = np.zeros(sz_c, dtype=numba.boolean)
    
    for ia in range(n_a):
        while (b[ib, 0] < a[ia, 0] or
               (b[ib, 0] == a[ia, 0] and
                    (b[ib, 1] < a[ia, 1] or
                         (b[ib, 1] == a[ia, 1] and
                              b[ib, 2] < a[ia, 2])))):
            if n_c >= sz_c:
                sz_c = min(sz_c * 2, n_a + n_b)
                a_contrib = enlarge_array(a_contrib, sz_c)
                b_contrib = enlarge_array(b_contrib, sz_c)
            
            a_contrib[n_c] = False
            b_contrib[n_c] = True
            
            ib += 1
            n_c += 1
            
            # If we reach the end of b before a, then write the rest of a
            # Note: This is BEFORE we have written the current ia values
            if ib == n_b:
                n_left = n_a - ia
                sz = n_c + n_left
                if a_contrib.size < sz:
                    a_contrib = enlarge_array(a_contrib, sz)
                    b_contrib = enlarge_array(b_contrib, sz)
                a_contrib[n_c:sz] = True
                b_contrib[n_c:sz] = False
                return a_contrib[:sz], b_contrib[:sz]
        
        if n_c >= sz_c:
            sz_c = min(sz_c * 2, n_a + n_b)
            a_contrib = enlarge_array(a_contrib, sz_c)
            b_contrib = enlarge_array(b_contrib, sz_c)
        
        a_contrib[n_c] = True

        if (a[ia, 0] == b[ib, 0] and
            a[ia, 1] == b[ib, 1] and
            a[ia, 2] == b[ib, 2]):
            b_contrib[n_c] = True
            ib += 1
            n_c += 1
            # If we reach the end of b before a, then write the rest of a
            if ib == n_b:
                n_left = n_a - ia - 1
                sz = n_c + n_left
                if a_contrib.size < sz:
                    a_contrib = enlarge_array(a_contrib, sz)
                    b_contrib = enlarge_array(b_contrib, sz)
                a_contrib[n_c:sz] = True
                b_contrib[n_c:sz] = False
                return a_contrib[:sz], b_contrib[:sz]
        else:
            b_contrib[n_c] = False
            n_c += 1
    
    n_left = n_b - ib
    sz = n_c + n_left
    
    if sz > a_contrib.size:
        a_contrib = enlarge_array(a_contrib, sz)
        b_contrib = enlarge_array(b_contrib, sz)
    a_contrib[n_c:sz] = False
    b_contrib[n_c:sz] = True
    
    return a_contrib[:sz], b_contrib[:sz]


@numba.njit
def _union_contrib_4d(a, b):
    n_a, ndim = a.shape
    n_b = b.shape[0]
    n_c = 0  # Number of elements written to sz_c
    ib = 0
    
    # sz_c = n_a + (n_b // 2) if n_a > n_b else n_b + (n_a // 2)
    sz_c = max(n_a, n_b)
    a_contrib = np.zeros(sz_c, dtype=numba.boolean)
    b_contrib = np.zeros(sz_c, dtype=numba.boolean)
    
    for ia in range(n_a):
        while (b[ib, 0] < a[ia, 0] or
            (b[ib, 0] == a[ia, 0] and
                (b[ib, 1] < a[ia, 1] or
                    (b[ib, 1] == a[ia, 1] and
                        (b[ib, 2] < a[ia, 2] or
                            (b[ib, 2] == a[ia, 2] and
                                b[ib, 3] < a[ia, 3])))))):
            if n_c >= sz_c:
                sz_c = min(sz_c * 2, n_a + n_b)
                a_contrib = enlarge_array(a_contrib, sz_c)
                b_contrib = enlarge_array(b_contrib, sz_c)
            
            a_contrib[n_c] = False
            b_contrib[n_c] = True
            
            ib += 1
            n_c += 1
            
            # If we reach the end of b before a, then write the rest of a
            # Note: This is BEFORE we have written the current ia values
            if ib == n_b:
                n_left = n_a - ia
                sz = n_c + n_left
                if a_contrib.size < sz:
                    a_contrib = enlarge_array(a_contrib, sz)
                    b_contrib = enlarge_array(b_contrib, sz)
                a_contrib[n_c:sz] = True
                b_contrib[n_c:sz] = False
                return a_contrib[:sz], b_contrib[:sz]
        
        if n_c >= sz_c:
            sz_c = min(sz_c * 2, n_a + n_b)
            a_contrib = enlarge_array(a_contrib, sz_c)
            b_contrib = enlarge_array(b_contrib, sz_c)
        
        a_contrib[n_c] = True
        
        if (a[ia, 0] == b[ib, 0] and
            a[ia, 1] == b[ib, 1] and
            a[ia, 2] == b[ib, 2] and
            a[ia, 3] == b[ib, 3]):
            b_contrib[n_c] = True
            ib += 1
            n_c += 1
            # If we reach the end of b before a, then write the rest of a
            if ib == n_b:
                n_left = n_a - ia - 1
                sz = n_c + n_left
                if a_contrib.size < sz:
                    a_contrib = enlarge_array(a_contrib, sz)
                    b_contrib = enlarge_array(b_contrib, sz)
                a_contrib[n_c:sz] = True
                b_contrib[n_c:sz] = False
                return a_contrib[:sz], b_contrib[:sz]
        else:
            b_contrib[n_c] = False
            n_c += 1
    
    n_left = n_b - ib
    sz = n_c + n_left
    
    if sz > a_contrib.size:
        a_contrib = enlarge_array(a_contrib, sz)
        b_contrib = enlarge_array(b_contrib, sz)
    a_contrib[n_c:sz] = False
    b_contrib[n_c:sz] = True
    
    return a_contrib[:sz], b_contrib[:sz]


@numba.njit
def _union_contrib_Nd(a_loc, b_loc):
    """Merge two IndexSets in linear time, reporting where the
    coordinates from each input contribute to the final IndexSet.

    Arguments
    ---------
    a_loc, b_loc : 2d arrays of ints
        Coordinates of the two IndexSets to merge.
        Both arrays must be lexicographically sorted along
        the first dimension. Each matrix of locations must have
        size >= 1 along the lowest dimension (cannot be empty).

    Returns
    -------
    (_c, a_contrib, b_contrib)
    Where :
    _c : 2d array of int32
        The merged coordinates in lexicogrpahical order and unique_.
    a_contrib, b_contrib : 1d array of bools
        The length of a_contrib and b_contrib are both the
        size of the final, merged locations. Each array
        is set to True in the positions where it contributes
        a coordinate.

    a_contrib and b_contrib may be True at the same locations
    if the locations in a and b overlap.
    """
    n_a, ndim = a_loc.shape
    n_b = b_loc.shape[0]
    n_c = 0  # Number of elements written to sz_c
    i_b = 0
    
    #sz_c = n_a + (n_b // 2) if n_a > n_b else n_b + (n_a // 2)
    sz_c = max(n_a, n_b)
    a_contrib = np.zeros(sz_c, dtype=numba.boolean)
    b_contrib = np.zeros(sz_c, dtype=numba.boolean)
    
    for i_a in range(n_a):
        while lex_less_Nd(b_loc[i_b], a_loc[i_a]):
            if n_c >= sz_c:
                sz_c = min(sz_c * 2, n_a + n_b)
                a_contrib = enlarge_array(a_contrib, sz_c)
                b_contrib = enlarge_array(b_contrib, sz_c)
            
            a_contrib[n_c] = False
            b_contrib[n_c] = True
            
            i_b += 1
            n_c += 1
            
            # If we reach the end of b before a, then write the rest of a
            # Note: This is BEFORE we have written the current i_a values
            if i_b == n_b:
                n_left = n_a - i_a
                sz = n_c + n_left
                if a_contrib.size < sz:
                    a_contrib = enlarge_array(a_contrib, sz)
                    b_contrib = enlarge_array(b_contrib, sz)
                a_contrib[n_c:sz] = True
                b_contrib[n_c:sz] = False
                return a_contrib[:sz], b_contrib[:sz]
        
        if n_c >= sz_c:
            sz_c = min(sz_c * 2, n_a + n_b)
            a_contrib = enlarge_array(a_contrib, sz_c)
            b_contrib = enlarge_array(b_contrib, sz_c)
        
        a_contrib[n_c] = True
        
        if eq_Nd(a_loc[i_a], b_loc[i_b]):
            b_contrib[n_c] = True
            i_b += 1
            n_c += 1
            # If we reach the end of b before a, then write the rest of a
            if i_b == n_b:
                n_left = n_a - i_a - 1
                sz = n_c + n_left
                if a_contrib.size < sz:
                    a_contrib = enlarge_array(a_contrib, sz)
                    b_contrib = enlarge_array(b_contrib, sz)
                a_contrib[n_c:sz] = True
                b_contrib[n_c:sz] = False
                return a_contrib[:sz], b_contrib[:sz]
        else:
            b_contrib[n_c] = False
            n_c += 1
    
    n_left = n_b - i_b
    sz = n_c + n_left
    
    if sz > a_contrib.size:
        a_contrib = enlarge_array(a_contrib, sz)
        b_contrib = enlarge_array(b_contrib, sz)
    a_contrib[n_c:sz] = False
    b_contrib[n_c:sz] = True
    
    return a_contrib[:sz], b_contrib[:sz]