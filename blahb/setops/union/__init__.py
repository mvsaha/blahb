import numpy as np
import numba
from concurrent.futures import ThreadPoolExecutor

from ...indexset import IndexSet, concat_sorted_nonoverlapping
from ...settings import parse_chunk_args
from ...chunk import gen_cochunks
from ...data import merge_data_direct, merge_data_indirect
from ...data import all_short_circuit_merges, order_does_not_matter
from ...flags import DATA_DEFAULT, SORTED, UNIQUE
from ...encoding import compatible_encoding
from .union_big_small import union_big_small_, merge_union_big_small_results
from .union_contrib import union_contrib_


@numba.njit(nogil=True)
def union_(a, b, MERGE=None):
    """Perform set union on two IndexSets
    
    Arguments
    ---------
    a, b : IndexSet instances
    MERGE : array of uint8 | None
        Per-dimension flag indicating how to merge data from locations present
        in both IndexSets. This affects the resulting data even if one of the
        inputs has None for data.
        
        `MERGE` may have a single value, in which case the merge action is
        applied to all data columns, or have a length equal to the number of
        data columns, each data column is merged according to the
        corresponding flag. This argument must be an array.
    
    Returns
    -------
    An IndexSet containing all of the locations in a or b and merged data.
    
    Notes
    -----
    The following flags are can be used in the MERGE array:
    * 2 DATA_NANSUM
        Add all data in corresponding locations, treating NaNs as zero
    * 3 DATA_SUM
        Add all data in corresponding locations, propagating NaNs.
    * 4 DATA_NANMAX
        Take the maximum contributing datum for each location, ignoring
        NaN values. If all contributors are NaN, the result is NaN.
    * 5 DATA_MAX
        Take the maximum datum at each location, propagating NaNs. If any
        contributing data are NaN, then the result is NaN
    * 6 _BLAHB_DATA_NANMIN
        Take the minimum contributing datum for each location, ignoring
        NaN values. If all contributors are NaN, the result is NaN.
    * 7 _BLAHB_DATA_MIN
        Take the minimum datum at each location, propagating NaNs. If any
        contributing data are NaN, then the result is NaN
    * 8 DATA_NANFIRST
        Find the first non-NaN data from the contributing IndexSets, in
        order of their input. If all data are NaN, then the result is NaN.
    * 10 DATA_NANLAST
        Find the last non-NaN data from the contributing IndexSets, in
        order of their input. If all data are NaN, then the result is NaN.
    
    The default merge mode is _BLAHB_DATA_FIRST will be applied if None
    is given instead of an array.
    """
    ndim = a.ndim
    
    if MERGE is None:
        MERGE = np.array([DATA_DEFAULT], dtype=np.uint8)
    
    if a.data is None and b.data is None:
        if b.n > a.n:  # Neither has data so order does not matter
            a, b = b, a

        if compatible_encoding(a, b):
            a_loc, b_loc = a._loc.view(np.uint32), b._loc.view(np.uint32)
            big, small = union_big_small_(a_loc, b_loc)
            temp = np.empty((big.size, a_loc.shape[1]), dtype=a_loc.dtype)
            merge_union_big_small_results(a_loc, b_loc, big, small, temp)
            c = IndexSet(temp.view(np.int32), SORTED | UNIQUE)
            c._encoding = a._encoding
            
        else:
            big, small = union_big_small_(a.loc, b.loc)
            temp2 = merge_union_big_small_results(a.loc, b.loc, big, small)
            c = IndexSet(temp2, SORTED | UNIQUE)
    
    elif (a.data is not None) and (b.data is not None):
        
        
        if compatible_encoding(a, b):
            #a_loc, b_loc = a._loc.view(np.uint32), b._loc.view(np.uint32)
            give_a, give_b = union_contrib_(a_loc, b_loc)
        
        else:
            give_a, give_b = union_contrib_(a.loc, b.loc)
        
        _c = np.empty((give_a.size, ndim), dtype=np.int32)
        _c[give_a] = a.loc
        _c[give_b] = b.loc
        c = IndexSet(_c, SORTED | UNIQUE)
        data = merge_data_indirect(a.data, give_a, None, MERGE)
        data = merge_data_indirect(b.data, give_b, data, MERGE)
        c.data = data
    
    else:  # Exactly one of {`a`, `b`} have associated data
        if a.data is None:
            a, b = b, a  # Now a is the one with data
        
        if compatible_encoding(a, b):
            a_loc, b_loc = a._loc.view(np.uint32), b._loc.view(np.uint32)
            big, small = union_big_small_(a_loc, b_loc)
            temp = np.empty((big.size, a_loc.shape[1]), dtype=a_loc.dtype)
            merge_union_big_small_results(a_loc, b_loc, big, small, temp)
            c = IndexSet(temp.view(np.int32), SORTED | UNIQUE)
            c._encoding = a._encoding
        
        else:
            big, small = union_big_small_(a.loc, b.loc)
            temp2 = merge_union_big_small_results(a.loc, b.loc, big, small)
            c = IndexSet(temp2, SORTED | UNIQUE)
        
        if all_short_circuit_merges(MERGE):
            c._data = None
        else:
            if not b.data is None:
                raise ValueError('b.data should be None')
            
            data = merge_data_indirect(a.data, big, None, MERGE)
            
            # `big` here is not used in the call to merge_data_direct
            # it is needed for typing (see comments) in `merge_data_direct`.
            data = merge_data_direct(b.data, small, data, MERGE)
            c.data = data
    
    # TODO: We can know the bounds here to give to `c`
    return c


@numba.njit(nogil=True)
def union_multi_(MERGE, *indexsets):
    """Merge more than two IndexSets in nopython mode.
    
    Arguments
    ---------
    MERGE : None | uint8 array of MERGE flags
    *indexsets : Variable number of IndexSet objects.

    Returns
    -------
    The union of the locations in all input IndexSets.

    Notes
    -----
    `MERGE` is the first argument, this order differs from both `union` and
    `union_`.
    
    This function will be recompiled every time it is called with a new number
    of IndexSets.
    """
    n = len(indexsets)
    if n == 0:
        raise ValueError("must have at least 1 input.")
    elif n == 1:
        return indexsets[0]
    u = indexsets[0]
    for i in range(1, n):
        u = union_(u, indexsets[i], MERGE)
    return u


def union_multi_unordered(MERGE, indexsets):
    """Merge more than two IndexSets in nopython mode.
    
    Arguments
    ---------
    MERGE : None | uint8 array of MERGE flags
    *indexsets : Variable number of IndexSet objects.

    Returns
    -------
    The union of the locations in all input IndexSets.

    Notes
    -----
    `MERGE` is the first argument, this order differs from both `union` and
    `union_`.

    This function will be recompiled every time it is called with a new number
    of IndexSets.
    """
    indexsets = list(indexsets)
    if len(indexsets) == 0:
        raise ValueError("must have at least 1 input.")
    
    if len(indexsets) == 1:
        return indexsets[0]
    
    while len(indexsets) > 1:
        indexsets.sort(key=lambda x: x.n)
        a = indexsets.pop(0)  # The two smallest items
        b = indexsets.pop(0)  # ...
        u = union_(a, b, MERGE)
        indexsets.append(u)
    
    assert len(indexsets) == 1
    return indexsets[0]


def union(objs, MERGE=None, **chunk_args):
    """Merge the locations and data from multiple IndexSets.
    
    Arguments
    ---------
    objs : sequence of IndexSet objects
    MERGE : array of uint8
        See `union_` documentation for more details.
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
    The set union of the rows in `objs`. Any data present is merged according
    to the MERGE flags provided.
    """
    n_workers, n_chunks, min_chunk_sz, max_chunk_sz = parse_chunk_args(
        **chunk_args)
    
    if order_does_not_matter(MERGE):
        union_fn = lambda indexsets: union_multi_unordered(MERGE, indexsets)
    else:
        union_fn = lambda indexsets: union_multi_(MERGE, *indexsets)
    
    # Do not chunk
    if n_chunks == n_workers == 1 and max(o.n for o in objs) <= max_chunk_sz:
        return union_fn(objs)
    
    filter_chunk = lambda x: x.n > 0
    _gen = gen_cochunks(objs, filter_chunk=filter_chunk, **chunk_args)
    if n_workers == 1:  # Chunk but do not use threads
        indexsets = tuple(map(union_fn, _gen))
    else:  # Chunk but do not use threads
        with ThreadPoolExecutor(max_workers=n_workers) as thread_pool:
            indexsets = tuple(thread_pool.map(union_fn, _gen))
    
    return concat_sorted_nonoverlapping(indexsets)