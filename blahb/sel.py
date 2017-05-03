"""
Functions for selecting data based on location value.
"""

import numba
import numpy as np

from .utils import exponential_search, searchsorted


@numba.njit
def _sel_range_sorted(sel_obj, dim, selector):
    """Unset flags except for where the coordinates in dim are
    match the selector."""
    if not dim == 0:
        raise ValueError("Can only call `sorted` methods on dim=0.")
    coords = sel_obj.indexset.loc[:, dim]
    start, stop = sel_obj.start, sel_obj.stop
    first = exponential_search(coords, selector[0], start, stop)
    last = exponential_search(coords, selector[1] + 1, first, stop)
    if sel_obj.has_flags:
        sel_obj.flags = sel_obj.flags[first - start:last - start]
        sel_obj._num_flags = sel_obj.flags.sum()
    
    sel_obj.start = first
    sel_obj.stop = last
    return sel_obj


@numba.njit
def sel_int(sel_obj, dim, selector):
    # Quit early if there are currently no pixels in the selection
    if sel_obj.num_flags == 0:
        return sel_obj
    
    # These are the coords to match against
    coords = sel_obj.indexset.loc[:, dim]
    
    if dim == 0:  # This dimension is sorted
        start = exponential_search(
            coords, selector, sel_obj.start, sel_obj.stop)
        stop = exponential_search(coords, selector + 1, start, sel_obj.stop)
        
        # This SelResult object already has flags that must be cropped
        if sel_obj.has_flags:
            sel_obj.flags = sel_obj.flags[start - sel_obj.start:
                                          stop - sel_obj.start]
            sel_obj._num_flags = sel_obj.flags.sum()
            assert stop - start == sel_obj.flags.size
        sel_obj.start = start
        sel_obj.stop = stop
    
    else:  # coords is not sorted
        if sel_obj.has_flags:
            flags = sel_obj.flags
            n_flags = sel_obj.num_flags
        else:  # Create the flags if they do not already exist
            n_flags = sel_obj.stop - sel_obj.start
            flags = np.ones(n_flags, dtype=numba.boolean)
        
        first_set = False  # Have we encountered any variables?
        start = sel_obj.start
        
        # Loop through the set flags an check if they match
        # Don't want this first loop because it forces loading of coords
        #for i, c in enumerate(coords[start:sel_obj.stop]):
        for i, j in enumerate(range(start, sel_obj.stop)):
            if flags[i]:
                if coords[j] == selector:
                    if not first_set:
                        first = j
                        last = j
                        first_set = True
                    else:
                        last = j
                else:
                    flags[i] = False
                    n_flags -= 1
        if n_flags:
            assert first_set
            last = last + 1
            sel_obj.flags = flags[first - start:last - start]
            sel_obj.start = first
            sel_obj.stop = last
            sel_obj.has_flags = True
            sel_obj._num_flags = n_flags
        else:
            sel_obj._num_flags = 0
            sel_obj.start = sel_obj.stop = 0
            sel_obj.flags = flags[:0].copy()
    
    return sel_obj


@numba.njit
def sel_below(sel_obj, dim, selector):
    """Input selector is a tuple of the form (None, val)
    We want to keep pixels that have a values less than or equal to val.
    """
    # Quit early if there are currently no pixels in the selection
    if sel_obj.num_flags == 0:
        return sel_obj
    
    # These are the coords to match against
    coords = sel_obj.indexset.loc[:, dim]
    
    if dim == 0:  # This dimension is sorted
        stop = exponential_search(
            coords, selector[1] + 1, sel_obj.start, sel_obj.stop)
        
        # This SelResult object already has flags that must be cropped
        if sel_obj.has_flags:
            sel_obj.flags = sel_obj.flags[:stop - sel_obj.start]
            sel_obj._num_flags = sel_obj.flags.sum()
        sel_obj.stop = stop
    
    else:  # coords is not sorted, must look over all coords
        if sel_obj.has_flags:
            flags = sel_obj.flags
            n_flags = sel_obj.num_flags
        else:  # Create the flags if they do not already exist
            n_flags = sel_obj.stop - sel_obj.start
            flags = np.ones(n_flags, dtype=numba.boolean)
        
        first_set = False  # Have we encountered any variables?
        thresh = selector[1]
        start = sel_obj.start
        # Loop through the set flags an check if they match
        #for i, c in enumerate(coords[start:sel_obj.stop]):
        for i, j in enumerate(range(start, sel_obj.stop)):
            if flags[i]:
                if coords[j] <= thresh:
                    if not first_set:
                        first = j
                        last = j
                        first_set = True
                    else:
                        last = j
                else:
                    flags[i] = False
                    n_flags -= 1
        if n_flags:
            assert first_set
            last = last + 1
            sel_obj.flags = flags[first - start:last - start]
            sel_obj.start = first
            sel_obj.stop = last
            sel_obj.has_flags = True
            sel_obj._num_flags = n_flags
        else:
            sel_obj._num_flags = 0
            sel_obj.start = sel_obj.stop = 0
            sel_obj.flags = flags[:0].copy()
    
    return sel_obj


@numba.njit
def sel_above(sel_obj, dim, selector):
    """Input selector is a tuple of the form (val, None)
    We want to keep pixels that have a values greater than or equal to val.
    """
    # Quit early if there are currently no pixels in the selection
    if sel_obj.num_flags == 0:
        return sel_obj
    
    # These are the coords to match against
    coords = sel_obj.indexset.loc[:, dim]
    
    if dim == 0:  # This dimension is sorted
        first = exponential_search(
            coords, selector[0], sel_obj.start, sel_obj.stop)
        
        # This SelResult object already has flags that must be cropped
        if sel_obj.has_flags:
            sel_obj.flags = sel_obj.flags[first - sel_obj.start:]
            sel_obj._num_flags = sel_obj.flags.sum()
        sel_obj.start = first
    
    else:  # coords is not sorted, must look over all coords
        if sel_obj.has_flags:
            flags = sel_obj.flags
            n_flags = sel_obj.num_flags
        else:  # Create the flags if they do not already exist
            n_flags = sel_obj.stop - sel_obj.start
            flags = np.ones(n_flags, dtype=numba.boolean)
        
        first_set = False  # Have we encountered any variables?
        thresh = selector[0]
        start = sel_obj.start
        # Loop through the set flags an check if they match
        #for i, c in enumerate(coords[start:sel_obj.stop]):
        for i, j in enumerate(range(start, sel_obj.stop)):
            if flags[i]:
                if coords[j] >= thresh:
                    if not first_set:
                        first = j
                        last = j
                        first_set = True
                    else:
                        last = j
                else:
                    flags[i] = False
                    n_flags -= 1
        if n_flags:
            assert first_set
            last = last + 1
            
            sel_obj.flags = flags[first - start:last - start]
            sel_obj.start = first
            sel_obj.stop = last
            sel_obj.has_flags = True
            sel_obj._num_flags = n_flags
        else:
            sel_obj._num_flags = 0
            sel_obj.start = sel_obj.stop = 0
            sel_obj.flags = flags[:0].copy()
    
    return sel_obj


@numba.njit
def sel_range(sel_obj, dim, selector):
    """Input selector is a tuple of the form (lo, hi)
    We want to keep locations along each dim where lo <= val <= hi is True.
    """
    # Quit early if there are currently no pixels in the selection
    if sel_obj.num_flags == 0:
        return sel_obj
    
    # These are the coords to match against
    coords = sel_obj.indexset.loc[:, dim]
    lo, hi = selector
    
    if dim == 0:  # This dimension is sorted
        
        start = exponential_search(coords, lo, sel_obj.start, sel_obj.stop)
        stop = exponential_search(coords, hi + 1, start, sel_obj.stop)
        # This SelResult object already has flags that must be cropped
        if sel_obj.has_flags:
            sel_obj.flags = sel_obj.flags[start - sel_obj.start:
                                          stop - sel_obj.start]
            sel_obj._num_flags = sel_obj.flags.sum()
        sel_obj.start = start
        sel_obj.stop = stop
    
    else:  # coords is not sorted, must look over all coords
        if sel_obj.has_flags:
            flags = sel_obj.flags
            n_flags = sel_obj.num_flags
        else:  # Create the flags if they do not already exist
            n_flags = sel_obj.stop - sel_obj.start
            flags = np.ones(n_flags, dtype=numba.boolean)
        
        first_set = False  # Have we encountered any variables?
        thresh = selector[0]
        start = sel_obj.start
        # Loop through the set flags an check if they match
        #for i, c in enumerate(coords[start:sel_obj.stop]):
        for i, j in enumerate(range(start, sel_obj.stop)):
            if flags[i]:
                if lo <= coords[j] <= hi:
                    if not first_set:
                        first = j
                        last = j
                        first_set = True
                    else:
                        last = j
                else:
                    flags[i] = False
                    n_flags -= 1
        if n_flags:
            assert first_set
            last = last + 1
            sel_obj.flags = flags[first - start:last - start]
            sel_obj.start = first
            sel_obj.stop = last
            sel_obj.has_flags = True
            sel_obj._num_flags = n_flags
        else:
            sel_obj._num_flags = 0
            sel_obj.start = sel_obj.stop = 0
            sel_obj.flags = flags[:0].copy()
    
    return sel_obj


@numba.njit
def sel_from_sorted_with_ints(sel_obj, dim, selector):
    """Leave flagged coordinates along a dim 0 that are in a sorted array."""
    
    assert dim == 0
    
    # Quit early if there are currently no pixels in the selection
    if sel_obj.num_flags == 0:
        return sel_obj
    
    start, stop = sel_obj.start, sel_obj.stop
    
    # These are the coords to match against
    coords = sel_obj.indexset.loc[start:stop, dim]
    
    if sel_obj.has_flags:
        flags = sel_obj.flags
        n_flags = sel_obj.num_flags
    else:
        n_flags = stop - start
        flags = np.ones(n_flags, dtype=numba.boolean)
        # flags = np.ones(n_flags, dtype=bool)
    
    lo, hi = coords[0], coords[-1]
    
    sel_start = exponential_search(selector, lo)
    sel_stop = exponential_search(selector, hi + 1, start=sel_start)
    selector = selector[sel_start:sel_stop]
    
    if selector.size == 0:
        sel_obj._set_to_empty()
        return sel_obj
    
    starts = searchsorted(coords, selector)
    stops = searchsorted(coords, selector + 1)

    # Quit early if all of the selector values were out of range of coords
    if starts[0] >= coords.size or stops[-1] == 0:
        sel_obj._set_to_empty()
        return sel_obj
    
    first = last = 0
    first_set = False
    n_flags = 0
    for i in range(selector.size - 1):
        # Keep flags set in matching strips
        for j in range(starts[i], stops[i]):
            if flags[j]:
                if first_set:
                    last = j
                else:
                    first = j
                    last = j
                    first_set = True
                n_flags += 1
        # Unset flags between matching strips
        flags[stops[i]:starts[i + 1]] = False
    
    for j in range(starts[-1], stops[-1]):
        if flags[j]:
            if first_set:
                last = j
            else:
                first = j
                last = j
                first_set = True
            n_flags += 1
    
    last += 1
    if n_flags == 0:
        sel_obj._set_to_empty()
    else:
        sel_obj.has_flags = True
        sel_obj.flags = flags[first:last]
        sel_obj.start = start + first
        sel_obj.stop = start + last
        sel_obj._num_flags = n_flags
    return sel_obj


@numba.njit
def val_in_array(val, arr):
    """Brute force check for a value in an array."""
    for a in arr:
        if val == a:
            return True
    return False


@numba.njit
def sel_from_unsorted_with_ints(sel_obj, dim, selector):
    """Leave flagged coordinates along a dim 0 that are in a sorted array."""
    
    assert dim > 0
    
    # Quit early if there are currently no pixels in the selection
    if sel_obj.num_flags == 0:
        return sel_obj
    
    start, stop = sel_obj.start, sel_obj.stop
    
    # These are the coords to match against
    coords = sel_obj.indexset.loc[:, dim]
    
    if sel_obj.has_flags:
        flags = sel_obj.flags
        n_flags = sel_obj.num_flags
    else:
        n_flags = stop - start
        flags = np.ones(n_flags, dtype=numba.boolean)
        # flags = np.ones(n_flags, dtype=bool)
    
    first = last = 0
    first_set = False
    
    # Loop through the set flags an check if they match
    for i, j in enumerate(range(start, stop)):
        if flags[i]:
            if val_in_array(coords[j], selector):
                if not first_set:
                    first = j
                    last = j
                    first_set = True
                else:
                    last = j
            else:
                flags[i] = False
                n_flags -= 1
    last += 1
    if n_flags == 0:
        sel_obj._set_to_empty()
    else:
        sel_obj.has_flags = True
        sel_obj.flags = flags[first - start:last - start]
        sel_obj.start = first
        sel_obj.stop = last
        sel_obj._num_flags = n_flags
    return sel_obj




# TODO: All dim=0 calls should use _sel_range_sorted if selector is not array


@numba.njit
def _sel_int(sel_result, dim, selector):
    if dim == 0:
        return _sel_range_sorted(sel_result, dim, (selector, selector))
    else:
        return sel_int(sel_result, dim, selector)


@numba.njit
def _sel_range(sel_result, dim, selector):
    if dim == 0:
        return _sel_range_sorted(sel_result, dim, selector)
    else:
        return sel_range(sel_result, dim, selector)


@numba.njit
def _sel_above(sel_result, dim, selector):
    if dim == 0:
        hi = sel_result.indexset.loc[-1 ,0]
        return _sel_range_sorted(sel_result, dim, (selector[0], hi))
    else:
        return sel_above(sel_result, dim, selector)


@numba.njit
def _sel_below(sel_result, dim, selector):
    if dim == 0:
        lo = sel_result.indexset.loc[0, 0]
        return _sel_range_sorted(sel_result, dim, (lo, selector[1]))
    else:
        return sel_below(sel_result, dim, selector)


@numba.njit
def sel_arr(sel_result, dim, selector):
    if dim == 0:
        return sel_from_sorted_with_ints(sel_result, dim, selector)
    else:
        return sel_from_unsorted_with_ints(sel_result, dim, selector)


from numba.types import i1, i2, i4, i8, u1, u2, u4, u8
all_int_types = [i1, i2, i4, i8, u1, u2, u4, u8]

none_int = numba.types.Tuple.from_types((None, numba.int64))

none_int_types = {numba.typeof((None, ty(0))) for ty in all_int_types}
int_none_types = {numba.typeof((ty(0), None)) for ty in all_int_types}

int_int_types = {numba.typeof((a(0), b(0))) for a in all_int_types
                                            for b in all_int_types}


@numba.generated_jit(nopython=True)
def _sel(sel_result, dim, selector):
    if selector in all_int_types:
        return lambda sel_result, dim, selector: _sel_int(
            sel_result, dim, selector)
    elif selector in none_int_types:
        return lambda sel_result, dim, selector: _sel_below(
            sel_result, dim, selector)
    elif selector in int_none_types:
        return lambda sel_result, dim, selector: _sel_above(
            sel_result, dim, selector)
    elif selector in int_int_types:
        return lambda sel_result, dim, selector: _sel_range(
            sel_result, dim, selector)
    elif isinstance(selector, numba.types.Array):
        return lambda sel_result, dim, selector: sel_arr(
            sel_result, dim, selector)
    else:
        raise TypeError("Cannot use `sel` with the given arguments.")