"""
Methods for performing binary set operations on IndexSets.

"direct" methods refer to building a new 2d array of locations that is the
result of a binary set operation. They are always lexicographically sorted
and contain only unique rows. These methods return nothing else.

"contrib" methods also build the final locations, but they additionally
provide information that indicates which locations from the parent IndexSets
contributed to the result. This is used for merging the data associated with
the input IndexSets.

"indirect" Like contrib, these provide the final result locations and extra
information. Indirect gives the locations in the final array that come from
a given IndexSet. They are the size of the final array with the number of
True values equal to the number of values in the IndexSet.
"""
import numpy as np
import numba
from concurrent.futures import ThreadPoolExecutor

from ...settings import parse_chunk_args
from ...chunk import gen_cochunks
from ...indexset import IndexSet, concat_sorted_nonoverlapping
from ...flags import SORTED, UNIQUE, DATA_DEFAULT
from ...data import all_short_circuit_merges
from ...encoding import compatible_encoding

from .intersection_contrib import intersection_contrib_
from .intersection_direct import intersection_direct_
from ...data import merge_data_direct


@numba.njit(nogil=True)
def intersection_(a, b, MERGE=None):
    """ Find the locations common to both a and b, merging data if necessary.
    
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
    IndexSet with the coordinates that are in both a and b. Data in the
    overlap will be merged according to MERGE flags.
    
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
    * 6 DATA_NANMIN
        Take the minimum contributing datum for each location, ignoring
        NaN values. If all contributors are NaN, the result is NaN.
    * 7 DATA_MIN
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
    if a.is_empty:
        return a
    elif b.is_empty:
        return b
    
    if MERGE is None:
        MERGE = np.array([DATA_DEFAULT], dtype=np.uint8)
    
    ndim = a.ndim
    assert ndim == b.ndim
    
    if (a.data is not None) and (b.data is not None):
        # Both data are present
        
        if compatible_encoding(a, b):
            take_a, take_b = intersection_contrib_(
              a._loc.view(np.uint32), b._loc.view(np.uint32))
            c = a.take(take_a)
        else:
            take_a, take_b = intersection_contrib_(a.loc, b.loc)
            c = IndexSet(a.loc[take_a], UNIQUE | SORTED)
        
        data = a.data[take_a]
        data = merge_data_direct(b.data, take_b, data, MERGE)
        c.data = data
    
    elif a.data is None and b.data is None:
        # Both have no data, result has None data
        # We should use the direct method with the smallest footprint
        if b.n < a.n:
            a, b = b, a

        if compatible_encoding(a, b):
            take_a, take_b = intersection_contrib_(
              a._loc.view(np.uint32), b._loc.view(np.uint32))
            c = a.take(take_a)
        else:
            take_a = intersection_direct_(a.loc, b.loc)
            c = IndexSet(a.loc[take_a], UNIQUE | SORTED)
        c._data = None
    
    else:
        # Exactly one of `a` or `b` has data: make `a` the one with data
        if b.data is not None:
            a, b = b, a  # a is now the one with data
        
        if compatible_encoding(a, b):
            take_a = intersection_direct_(
                a._loc.view(np.uint32), b._loc.view(np.uint32))
            c = a.take(take_a)
        else:
            take_a = intersection_direct_(a.loc, b.loc)
            c = IndexSet(a.loc[take_a], UNIQUE | SORTED)
        
        if all_short_circuit_merges(MERGE):
            c._data = None
        else:
            data = a.data[take_a]
            # Take a will be ignored because b.data is None
            data = merge_data_direct(b.data, take_a, data, MERGE)
            if data is None:
                c._data = None
            else:
                c.data = data
    
    return c


@numba.njit(nogil=True)
def intersection_multi_(MERGE, *indexsets):
    """Find the pixelsets common to any of the input objects.

    Arguments
    ---------
    MERGE : None | uint8 array of MERGE flags
    *indexsets : Variable number of IndexSet objects.
    
    Returns
    -------
    The union of the locations in all input IndexSets.

    Notes
    -----
    `MERGE` is the first argument, this order differs from both
    `intersection` and `intersection_`.

    This function will be recompiled every time it is called with a new number
    of IndexSets.
    """
    n = len(indexsets)
    if n == 0:
        raise ValueError("must have at least 1 input.")
    elif n == 1:
        return indexsets[0]
    
    result = indexsets[0]
    for i in range(1, n):
        if result.n == 0:
            return result
        result = intersection_(result, indexsets[i], MERGE)
    return result
 

def intersection(objs, MERGE=None, **chunk_args):
    """Merge the locations and data common to all IndexSets.

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
    
    intersection_fn = lambda objs: intersection_multi_(MERGE, *objs)
    
    # Do not chunk
    if n_chunks == n_workers == 1 and max(o.n for o in objs) <= max_chunk_sz:
        return intersection_fn(objs)
    
    filter_chunk = lambda x: x.n > 0
    _gen = gen_cochunks(objs, filter_chunk=filter_chunk, **chunk_args)
    if n_workers == 1:  # Chunk but do not use threads
        indexsets = tuple(map(intersection_fn, _gen))
    else:  # Chunk but do not use threads
        with ThreadPoolExecutor(max_workers=n_workers) as thread_pool:
            indexsets = tuple(thread_pool.map(intersection_fn, _gen))
    
    return concat_sorted_nonoverlapping(indexsets)