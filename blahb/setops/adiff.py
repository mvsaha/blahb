import numpy as np
import numba
from concurrent.futures import ThreadPoolExecutor

from ..flags import *
from ..settings import parse_chunk_args
from ..chunk import gen_cochunks
from ..indexset import IndexSet, concat_sorted_nonoverlapping
from ..indexset import is_indexset_subclass
from ..utils import lex_less_Nd, eq_Nd
from ..encoding import compatible_encoding

@numba.njit
def _asymmetric_difference_1d(a, b):
    """Return flags on locations in a that are not in b."""
    na, nb = a.shape[0], b.shape[0]
    take_a = np.empty(na, dtype=numba.boolean)
    ib = 0
    
    for ia in range(na):
        while ib < nb and b[ib, 0] < a[ia, 0]:
            ib += 1
        
        if ib == nb:
            take_a[ia:] = True
            return take_a
        
        if a[ia, 0] == b[ib, 0]:
            take_a[ia] = False
        else:
            take_a[ia] = True
    return take_a


@numba.njit
def _asymmetric_difference_2d(a, b):
    """Return flags on locations in a that are not in b."""
    na, nb = a.shape[0], b.shape[0]
    take_a = np.empty(na, dtype=numba.boolean)
    ib = 0
    
    for ia in range(na):
        while ib < nb and (b[ib, 0] < a[ia, 0] or
               (b[ib, 0] == a[ia, 0] and b[ib, 1] < a[ia, 1])):
            ib += 1
        
        if ib == nb:
            take_a[ia:] = True
            return take_a

        if a[ia, 0] == b[ib, 0] and a[ia, 1] == b[ib, 1]:
            take_a[ia] = False
        else:
            take_a[ia] = True
    return take_a


@numba.njit
def _asymmetric_difference_3d(a, b):
    """Return flags on locations in a that are not in b."""
    na, nb = a.shape[0], b.shape[0]
    take_a = np.empty(na, dtype=numba.boolean)
    ib = 0
    
    for ia in range(na):
        while ib < nb and (b[ib, 0] < a[ia, 0] or
               (b[ib, 0] == a[ia, 0] and
                    (b[ib, 1] < a[ia, 1] or
                         (b[ib, 1] == a[ia, 1] and
                              b[ib, 2] < a[ia, 2])))):
            ib += 1
        
        if ib == nb:
            take_a[ia:] = True
            return take_a

        if (a[ia, 0] == b[ib, 0] and
            a[ia, 1] == b[ib, 1] and
            a[ia, 2] == b[ib, 2]):
            take_a[ia] = False
        else:
            take_a[ia] = True
    return take_a


@numba.njit
def _asymmetric_difference_4d(a, b):
    """Return flags on locations in a that are not in b."""
    na, nb = a.shape[0], b.shape[0]
    take_a = np.empty(na, dtype=numba.boolean)
    ib = 0
    
    for ia in range(na):
        while ib < nb and (b[ib, 0] < a[ia, 0] or
               (b[ib, 0] == a[ia, 0] and
                    (b[ib, 1] < a[ia, 1] or
                         (b[ib, 1] == a[ia, 1] and
                              (b[ib, 2] < a[ia, 2] or
                                   (b[ib, 2] == a[ia, 2] and
                                        b[ib, 3] < a[ia, 3])))))):
            ib += 1
        
        if ib == nb:
            take_a[ia:] = True
            return take_a

        if (a[ia, 0] == b[ib, 0] and
            a[ia, 1] == b[ib, 1] and
            a[ia, 2] == b[ib, 2] and
            a[ia, 3] == b[ib, 3]):
            take_a[ia] = False
        else:
            take_a[ia] = True
    return take_a


@numba.njit
def _asymmetric_difference_Nd(a, b):
    """Return flags on locations in a that are not in b."""
    na, nb = a.shape[0], b.shape[0]
    take_a = np.empty(na, dtype=numba.boolean)
    ib = 0
    
    for ia in range(na):
        while ib < nb and lex_less_Nd(b[ib], a[ia]):
            ib += 1
        
        if ib == nb:
            take_a[ia:] = True
            return take_a
        
        if eq_Nd(a[ia], b[ib]):
            take_a[ia] = False
        else:
            take_a[ia] = True
    return take_a


@numba.njit(nogil=True)
def _adiff_helper(a_loc, b_loc):
    ndim = a_loc.shape[1]
    if ndim == 1:
        return _asymmetric_difference_1d(a_loc, b_loc)
    elif ndim == 2:
        return _asymmetric_difference_2d(a_loc, b_loc)
    elif ndim == 3:
        return _asymmetric_difference_3d(a_loc, b_loc)
    elif ndim == 4:
        return _asymmetric_difference_4d(a_loc, b_loc)
    else:
        return _asymmetric_difference_Nd(a_loc, b_loc)
    

@numba.njit(nogil=True)
def asymmetric_difference_(a, b):
    """Return the locations in a that are not in b.
    
    Arguments
    ---------
    a, b : IndexSet
        The variables to compare
    
    Returns
    -------
    c : IndexSet
        Contains locations of `a` that are not in `b`, as well as data from
        `a` at these locations.
    """
    if a.n == 0:
        return a
    ndim = a.ndim
    
    if compatible_encoding(a, b):
        take_a = _adiff_helper(a._loc.view(np.uint32), a._loc.view(np.uint32))
        temp = a._loc[take_a]
        c = IndexSet(temp.astype(np.int32), SORTED | UNIQUE)
    else:
        take_a = _adiff_helper(a.loc, b.loc)
        c = a.take(take_a)
    
    if a.data is not None:
        c.data = a.data[take_a]
    else:
        c._data = None
    return c


@numba.njit(nogil=True)
def asymmetric_difference_multi_(a, others):
    """Find the locations in an IndexSet that are not in any other.
    
    Arguments
    ---------
    a : IndexSet
    *others : tuple of IndexSet instances.

    Returns
    -------
    An IndexSet containing locations in `a` that do not appeard in any other
    IndexSet.
    
    This function will be recompiled every time it is called with a new number
    of IndexSets.
    """
    n = len(others)
    if n == 0:
        return a
    
    for i in range(0, n):
        if a.n == 0:
            return a
        a = asymmetric_difference_(a, others[i])
    return a


def asymmetric_difference(a, others, **chunk_args):
    """Find the asymmetric difference of two IndexSets.

    Arguments
    ---------
    a  : IndexSet
    others : IndexSet | sequence of IndexSets
    
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
    An IndexSet with the locations in `a` that are not in any of `others`.
    """
    n_workers, n_chunks, min_chunk_sz, max_chunk_sz = parse_chunk_args(
        **chunk_args)
    
    if is_indexset_subclass(others):
        together = (a, others)
    else:
        together = (a, ) + tuple(others)
    
    asym_diff_fn = lambda seq: asymmetric_difference_multi_(seq[0], seq[1:])
    
    max_obj_sz = max(o.n for o in together)
    if n_chunks == n_workers == 1 and max_obj_sz <= max_chunk_sz:
        return asym_diff_fn(together)
    
    filter_group = lambda objs: objs[0].n > 0
    _gen = gen_cochunks(together, filter_group=filter_group, **chunk_args)
    if n_workers == 1:  # Chunk but do not use threads
        indexsets = tuple(map(asym_diff_fn, _gen))
    else:  # Chunk but do not use threads
        with ThreadPoolExecutor(max_workers=n_workers) as thread_pool:
            indexsets = tuple(thread_pool.map(asym_diff_fn, _gen))
    
    return concat_sorted_nonoverlapping(indexsets)
