import numpy as np
import numba

from numblahb.utils import enlarge_array

@numba.njit
def update_breaks(known_breaks, coords):
    i_c = 0
    i_kb = 0
    
    new_breaks = np.empty(known_breaks.size, np.uint64)
    i_nb = 0
    
    for i_kb in range(known_breaks.size):
        while i_c < known_breaks[i_kb]:
            if coords[i_c] != coords[i_c - 1]:
                if i_nb >= new_breaks.size:
                    sz = min(new_breaks.size * 2, coords.size + 1)
                    new_breaks = enlarge_array(new_breaks, sz)
                new_breaks[i_nb] = i_c
                i_nb += 1
            i_c += 1
        
        if i_nb >= new_breaks.size:
            sz = min(new_breaks.size * 2, coords.size + 1)
            new_breaks = enlarge_array(new_breaks, sz)
        
        new_breaks[i_nb] = known_breaks[i_kb]
        i_nb += 1
        i_c += 1
    # assert np.unique(new_breaks[:i_nb]).size == i_nb
    return new_breaks[:i_nb]


@numba.njit
def sub_sort_breaks(breaks, coords, sort_order):
    for i_br in range(breaks.size - 1):
        br_start, br_end = breaks[i_br], breaks[i_br + 1]
        sub_sort_order = np.argsort(coords[br_start:br_end])
        coords[br_start:br_end] = coords[br_start:br_end][sub_sort_order]
        sort_order[br_start:br_end] = sub_sort_order + br_start


@numba.njit
def lexsort_inplace(loc, save_indirect=False):
    """Sort a tuple of coordinates in place.

    Arguments
    ---------
    loc : n X ndim array of ints
        Each row represents a coordinate in the universe.

    Returns
    -------
    breaks : array with starting indexes of runs of identical pixels
        in the now-sorted coordinates.
    indirect : The indirect sorting indices for the input coordinates, if
        save_indirect is True. Otherwise it will be None.
    """
    # Dimensions to sort
    n, ndim = loc.shape
    
    # First dimension is easy
    # sub_breaks = np.array([0, n], dtype=np.uint64)
    sort_order = np.empty(n, dtype=np.uint64)
    
    indirect = np.zeros(0, dtype=np.int64)
    if save_indirect:
        indirect = np.arange(n  )  # , dtype=np.int64)
    
    for dim in range(ndim):
        if dim == 0:
            breaks = np.array([0, n], dtype=np.uint64)
        else:
            breaks = update_breaks(breaks, loc[:, dim - 1])
        
        # Inplace modification of sort_order
        sub_sort_breaks(breaks, loc[:, dim], sort_order)
        
        # Sort the sub-dims
        loc[:, dim + 1:] = loc[sort_order, dim + 1:]
        
        if save_indirect:
            indirect[:] = indirect[sort_order]

    breaks = update_breaks(breaks, loc[:, ndim - 1])
    return breaks[:-1], indirect