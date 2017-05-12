import numpy as np
import numba
from concurrent.futures import ThreadPoolExecutor

from ..indexset import IndexSet, concat_sorted_nonoverlapping
from ..bits import (UNIQUE, SORTED)
from ..settings import parse_chunk_args
from ..chunk import gen_cochunks
from ..utils import enlarge_array
from ..utils import lex_less_Nd, eq_Nd


@numba.njit
def _symmetric_difference_1d(a, b):
    na, nb = a.shape[0], b.shape[0]
    ia = ib = 0
    
    take_a = np.zeros(na, dtype=numba.boolean)
    take_b = np.zeros(nb, dtype=numba.boolean)
    
    c_size = max((na + nb) // 2, 8)
    
    # True where values come from a, False where values come from b
    contrib = np.zeros(c_size, dtype=numba.boolean)
    
    # The number of rows unique_ to a or b encountered, so far.
    # Alternatively, the index of the next value to place in contrib
    nc = 0
    
    for ia in range(na):
        while ib < nb and b[ib, 0] < a[ia, 0]:
            take_b[ib] = True
            ib += 1
            if c_size <= nc:
                c_size = min(na + nb, c_size * 2)
                contrib = enlarge_array(contrib, c_size)
            contrib[nc] = False
            nc += 1
        
        if ib == nb:
            # Finish writing a to output
            na_left = na - ia
            take_a[ia:] = True
            
            if c_size < nc + na_left:
                contrib = enlarge_array(contrib, nc + na_left)
            contrib[nc:nc + na_left] = True
            nc = nc + na_left
            return take_a, take_b, contrib[:nc]
        
        if a[ia, 0] == b[ib, 0]:
            take_a[ia] = False
            take_b[ib] = False
            ib += 1
        else:
            take_a[ia] = True
            if c_size <= nc:
                c_size = min(c_size * 2, na + nb)
                contrib = enlarge_array(contrib, c_size)
            contrib[nc] = True
            nc += 1
    
    if ib < nb:
        nb_left = nb - ib
        take_b[ib:] = True
        
        if c_size < nc + nb_left:
            contrib = enlarge_array(contrib, nc + nb_left)
        contrib[nc:nc + nb_left] = False
        nc = nc + nb_left
    
    return take_a, take_b, contrib[:nc]


@numba.njit
def _symmetric_difference_2d(a, b):
    na, nb = a.shape[0], b.shape[0]
    ia = ib = 0
    
    take_a = np.zeros(na, dtype=numba.boolean)
    take_b = np.zeros(nb, dtype=numba.boolean)
    
    c_size = max((na + nb) // 2, 8)
    
    # True where values come from a, False where values come from b
    contrib = np.zeros(c_size, dtype=numba.boolean)
    
    # The number of rows unique_ to a or b encountered, so far.
    # Alternatively, the index of the next value to place in contrib
    nc = 0
    
    for ia in range(na):
        while ib < nb and (b[ib, 0] < a[ia, 0] or
               (b[ib, 0] == a[ia, 0] and b[ib, 1] < a[ia, 1])):
            take_b[ib] = True
            ib += 1
            if c_size <= nc:
                c_size = min(na + nb, c_size * 2)
                contrib = enlarge_array(contrib, c_size)
            contrib[nc] = False
            nc += 1
        
        if ib == nb:
            # Finish writing a to output
            na_left = na - ia
            take_a[ia:] = True
            
            if c_size < nc + na_left:
                contrib = enlarge_array(contrib, nc + na_left)
            contrib[nc:nc + na_left] = True
            nc = nc + na_left
            return take_a, take_b, contrib[:nc]
        
        if a[ia, 0] == b[ib, 0] and a[ia, 1] == b[ib, 1]:
            take_a[ia] = False
            take_b[ib] = False
            ib += 1
        else:
            take_a[ia] = True
            if c_size <= nc:
                c_size = min(c_size * 2, na + nb)
                contrib = enlarge_array(contrib, c_size)
            contrib[nc] = True
            nc += 1
    
    if ib < nb:
        nb_left = nb - ib
        take_b[ib:] = True
        
        if c_size < nc + nb_left:
            contrib = enlarge_array(contrib, nc + nb_left)
        contrib[nc:nc + nb_left] = False
        nc = nc + nb_left
    
    return take_a, take_b, contrib[:nc]


@numba.njit
def _symmetric_difference_3d(a, b):
    na, nb = a.shape[0], b.shape[0]
    ia = ib = 0
    
    take_a = np.zeros(na, dtype=numba.boolean)
    take_b = np.zeros(nb, dtype=numba.boolean)
    
    c_size = max((na + nb) // 2, 8)
    
    # True where values come from a, False where values come from b
    contrib = np.zeros(c_size, dtype=numba.boolean)
    
    # The number of rows unique_ to a or b encountered, so far.
    # Alternatively, the index of the next value to place in contrib
    nc = 0
    
    for ia in range(na):
        while ib < nb and (b[ib, 0] < a[ia, 0] or
               (b[ib, 0] == a[ia, 0] and
                    (b[ib, 1] < a[ia, 1] or
                         (b[ib, 1] == a[ia, 1] and
                              b[ib, 2] < a[ia, 2])))):
            take_b[ib] = True
            ib += 1
            if c_size <= nc:
                c_size = min(na + nb, c_size * 2)
                contrib = enlarge_array(contrib, c_size)
            contrib[nc] = False
            nc += 1
        
        if ib == nb:
            # Finish writing a to output
            na_left = na - ia
            take_a[ia:] = True
            
            if c_size < nc + na_left:
                contrib = enlarge_array(contrib, nc + na_left)
            contrib[nc:nc + na_left] = True
            nc = nc + na_left
            return take_a, take_b, contrib[:nc]
        
        if (a[ia, 0] == b[ib, 0] and
            a[ia, 1] == b[ib, 1] and
            a[ia, 2] == b[ib, 2]):
            take_a[ia] = False
            take_b[ib] = False
            ib += 1
        else:
            take_a[ia] = True
            if c_size <= nc:
                c_size = min(c_size * 2, na + nb)
                contrib = enlarge_array(contrib, c_size)
            contrib[nc] = True
            nc += 1
    
    if ib < nb:
        nb_left = nb - ib
        take_b[ib:] = True
        
        if c_size < nc + nb_left:
            contrib = enlarge_array(contrib, nc + nb_left)
        contrib[nc:nc + nb_left] = False
        nc = nc + nb_left
    
    return take_a, take_b, contrib[:nc]


@numba.njit
def _symmetric_difference_4d(a, b):
    na, nb = a.shape[0], b.shape[0]
    ia = ib = 0
    
    take_a = np.zeros(na, dtype=numba.boolean)
    take_b = np.zeros(nb, dtype=numba.boolean)
    
    c_size = max((na + nb) // 2, 8)
    
    # True where values come from a, False where values come from b
    contrib = np.zeros(c_size, dtype=numba.boolean)
    
    # The number of rows unique_ to a or b encountered, so far.
    # Alternatively, the index of the next value to place in contrib
    nc = 0
    
    for ia in range(na):
        while ib < nb and (b[ib, 0] < a[ia, 0] or
               (b[ib, 0] == a[ia, 0] and
                    (b[ib, 1] < a[ia, 1] or
                         (b[ib, 1] == a[ia, 1] and
                              (b[ib, 2] < a[ia, 2] or
                                   (b[ib, 2] == a[ia, 2] and
                                            b[ib, 3] < a[ia, 3])))))):
            take_b[ib] = True
            ib += 1
            if c_size <= nc:
                c_size = min(na + nb, c_size * 2)
                contrib = enlarge_array(contrib, c_size)
            contrib[nc] = False
            nc += 1
        
        if ib == nb:
            # Finish writing a to output
            na_left = na - ia
            take_a[ia:] = True
            
            if c_size < nc + na_left:
                contrib = enlarge_array(contrib, nc + na_left)
            contrib[nc:nc + na_left] = True
            nc = nc + na_left
            return take_a, take_b, contrib[:nc]
        
        if (a[ia, 0] == b[ib, 0] and
            a[ia, 1] == b[ib, 1] and
            a[ia, 2] == b[ib, 2] and
            a[ia, 3] == b[ib, 3]):
            take_a[ia] = False
            take_b[ib] = False
            ib += 1
        else:
            take_a[ia] = True
            if c_size <= nc:
                c_size = min(c_size * 2, na + nb)
                contrib = enlarge_array(contrib, c_size)
            contrib[nc] = True
            nc += 1
    
    if ib < nb:
        nb_left = nb - ib
        take_b[ib:] = True
        
        if c_size < nc + nb_left:
            contrib = enlarge_array(contrib, nc + nb_left)
        contrib[nc:nc + nb_left] = False
        nc = nc + nb_left
    
    return take_a, take_b, contrib[:nc]


@numba.njit
def _symmetric_difference_Nd(a, b):
    na, nb = a.shape[0], b.shape[0]
    ia = ib = 0
    
    take_a = np.zeros(na, dtype=numba.boolean)
    take_b = np.zeros(nb, dtype=numba.boolean)
    
    c_size = max((na + nb) // 2, 8)
    
    # True where values come from a, False where values come from b
    contrib = np.zeros(c_size, dtype=numba.boolean)
    
    # The number of rows unique_ to a or b encountered, so far.
    # Alternatively, the index of the next value to place in contrib
    nc = 0
    
    for ia in range(na):
        while ib < nb and lex_less_Nd(b[ib], a[ia]):
            take_b[ib] = True
            ib += 1
            if c_size <= nc:
                c_size = min(na + nb, c_size * 2)
                contrib = enlarge_array(contrib, c_size)
            contrib[nc] = False
            nc += 1
        
        if ib == nb:
            # Finish writing a to output
            na_left = na - ia
            take_a[ia:] = True
            
            if c_size < nc + na_left:
                contrib = enlarge_array(contrib, nc + na_left)
            contrib[nc:nc + na_left] = True
            nc = nc + na_left
            return take_a, take_b, contrib[:nc]
        
        if eq_Nd(a[ia], b[ib]):
            take_a[ia] = False
            take_b[ib] = False
            ib += 1
        else:
            take_a[ia] = True
            if c_size <= nc:
                c_size = min(c_size * 2, na + nb)
                contrib = enlarge_array(contrib, c_size)
            contrib[nc] = True
            nc += 1
    
    if ib < nb:
        nb_left = nb - ib
        take_b[ib:] = True
        
        if c_size < nc + nb_left:
            contrib = enlarge_array(contrib, nc + nb_left)
        contrib[nc:nc + nb_left] = False
        nc = nc + nb_left
    
    return take_a, take_b, contrib[:nc]


@numba.njit
def _fill_symmetric_difference_result(a, a_take, b, b_take, contrib):
    ndim = a.shape[1]
    nc = contrib.size
    c = np.empty((nc, ndim), dtype=a.dtype)
    
    ia = ib = 0
    for ic in range(nc):
        if contrib[ic]:
            while not a_take[ia]:
                ia += 1
            for dim in range(ndim):
                c[ic, dim] = a[ia, dim]
            ia += 1
        else:
            while not b_take[ib]:
                ib += 1
            for dim in range(ndim):
                c[ic, dim] = b[ib, dim]
            ib += 1
    return c


@numba.njit
def put_vals_where_true(vals, tf, put):
    """
    tf.size = put.shape[0
    np.sum(tf) == vals.shape[0]
    """
    iv = 0
    ndim = vals.shape[1]
    assert ndim == put.shape[1]
    for i in range(tf.size):
        if tf[i]:
            for dim in range(ndim):
                put[i, dim] = vals[iv, dim]
            iv += 1


@numba.njit
def put_vals_where_false(vals, tf, put):
    """
    tf.size = put.shape[0
    np.sum(tf) == vals.shape[0]
    """
    iv = 0
    ndim = vals.shape[1]
    assert ndim == put.shape[1]
    for i in range(tf.size):
        if not tf[i]:
            for dim in range(ndim):
                put[i, dim] = vals[iv, dim]
            iv += 1


@numba.njit
def put_a_if_true_else_put_b(a, b, tf, put):
    ndim = a.shape[1]
    assert ndim == b.shape[1] == put.shape[1]
    ia = ib = 0
    for i in range(tf.size):
        if tf[i]:
            for dim in range(ndim):
                put[i, dim] = a[ia, dim]
            ia += 1
        else:
            for dim in range(ndim):
                put[i, dim] = b[ib, dim]
            ib += 1


@numba.njit(nogil=True)
def symmetric_difference_(a, b):
    """Find the symmetric difference of two IndexSets.
    
    Arguments
    ---------
    a, b : IndexSets
    
    Returns
    -------
    c : IndexSet with the locations that are in a or b, but not both.
    
    Note
    ----
    The result of:
    >>>symmetric_difference_(a, b)
    is equivalent to, but faster than this:
    >>>asymmetric_difference_(union_(a, b), intersection_(a, b))
    
    No MERGE flag is needed for the flags because the result is always non-
    overlapping. Data from `a` goes to the `a` locations of the result and
    data from `b` goes to the `b` locations.
    """
    ndim = a.ndim
    if ndim == 1:
        a_take, b_take, contrib = _symmetric_difference_1d(a.loc, b.loc)
    elif ndim == 2:
        a_take, b_take, contrib = _symmetric_difference_2d(a.loc, b.loc)
    elif ndim == 3:
        a_take, b_take, contrib = _symmetric_difference_3d(a.loc, b.loc)
    elif ndim == 4:
        a_take, b_take, contrib = _symmetric_difference_4d(a.loc, b.loc)
    else:
        a_take, b_take, contrib = _symmetric_difference_Nd(a.loc, b.loc)
    
    c_loc = np.empty((contrib.size, ndim), dtype=a.loc.dtype)
    put_a_if_true_else_put_b(a.loc[a_take], b.loc[b_take], contrib, c_loc)
    c = IndexSet(c_loc, SORTED | UNIQUE)
    
    if a.data is None and b.data is None:
        c._data = None
    elif b.data is None:
        data = np.full((contrib.size, a.data.shape[1]), np.nan, np.float32)
        data[contrib] = a.data[a_take]
        #put_vals_where_true(a.data[a_take], contrib, data)
        c.data = data
    elif a.data is None:
        data = np.full((contrib.size, b.data.shape[1]), np.nan, np.float32)
        put_vals_where_false(b.data[b_take], contrib, data)
        c.data = data
    else:  # Both have data that does not overlap
        #data = np.full((contrib.size, a.data.shape[1]), np.nan, np.float32)
        data = np.empty((contrib.size, a.data.shape[1]), np.float32)
        put_a_if_true_else_put_b(a.data[a_take], b.data[b_take], contrib, data)
        c.data = data
    
    return c


def symmetric_difference(a, b, **chunk_args):
    """Find locations in exactly one of the input IndexSets
    
    Arguments
    --------
    a, b : IndexSets
    **chunk_args : Options to control chunk size and parallelization.
        Can be some of the following optional arguments:
        * n_workers - Number of worker threads to map chunked operations onto.
        This value will be passed to concurrent.futures.ThreadPoolExecutor as
        the `max_workers` argument.
        * max_chunk_size - Maximum number of locations
        * min_chunk_size - Minimum number of locations
        * n_chunks - Minimum number of chunks to break an IndexSet into.
    
    Returns
    -------
    The locations in either `a` or `b`, but not both.
    """
    n_workers, n_chunks, min_chunk_sz, max_chunk_sz = parse_chunk_args(
        **chunk_args)
    
    sym_diff_fn = lambda objs: symmetric_difference_(a, b)
    
    # Do not chunk
    if n_chunks == n_workers == 1 and max(a.n, b.n) <= max_chunk_sz:
        return symmetric_difference_(a, b)
    
    _gen = gen_cochunks((a, b), **chunk_args)
    if n_workers == 1:  # Chunk but do not use threads
        indexsets = tuple(map(sym_diff_fn, _gen))
    else:  # Chunk but do not use threads
        with ThreadPoolExecutor(max_workers=n_workers) as thread_pool:
            indexsets = tuple(thread_pool.map(sym_diff_fn, _gen))
    
    return concat_sorted_nonoverlapping(indexsets)

