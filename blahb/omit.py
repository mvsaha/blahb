import numba
import numpy as np

from .utils import exponential_search
from .sel import val_in_array


@numba.njit
def _set_false_get_count(arr, start, stop):
    """Set the arr[start:stop] to False, counting the number of True set."""
    count = 0
    for i in range(start, stop):
        if arr[i]:
            count += 1
            arr[i] = False
    return count


@numba.njit
def _omit_range_sorted(sel_obj, dim, selector):
    """Leave flags raised if they do not match the integer selector."""
    
    # Quit early if there are currently no pixels in the selection
    if sel_obj.num_flags == 0:
        return sel_obj
    
    # These are the coords to match against
    coords = sel_obj.indexset.loc[:, dim]
    
    start, stop = sel_obj.start, sel_obj.stop
    
    assert dim == 0
    
    # The indexes into coords of the matching range
    # We only want locations before `first`  or after `last`
    first = exponential_search(
        coords, selector[0], sel_obj.start, sel_obj.stop)
    last = exponential_search(coords, selector[1] + 1, first, sel_obj.stop)
    
    # No locations were affected
    if first == last:
        return sel_obj
    
    # Ommitted locations span all of `start:stop`
    elif first == start and last == stop:
        sel_obj._set_to_empty()
        return sel_obj
    
    # There are not locations to take before `first`, only after `stop`
    elif first == start:
        # We can take an unmodified slice of `flags` from the back
        if sel_obj.has_flags:
            sel_obj.flags = sel_obj.flags[last - start:]
            sel_obj._num_flags = sel_obj.flags.sum()
        sel_obj.start = last  # `stop` is unchanged
    
    # There are not locations to take after `stop`, only before `first`
    elif last == stop:
        # We can take an unmodified slice of `flags` from the front
        if sel_obj.has_flags:
            sel_obj.flags = sel_obj.flags[:first - start]
            sel_obj._num_flags = sel_obj.flags.sum()
        sel_obj.stop = first  # `stop` is unchanged
    
    else:
        if sel_obj.has_flags:
            flags = sel_obj.flags
            n_flags = sel_obj._num_flags
        else:
            n_flags = stop - start
            flags = np.ones(n_flags, dtype=numba.boolean)
        to_decr = _set_false_get_count(flags, first - start, last - start)
        sel_obj._num_flags = n_flags - to_decr
        sel_obj.flags = flags
        # Start and stop do not change
        sel_obj.has_flags = True
    
    return sel_obj


@numba.njit
def _omit_int_unsorted(sel_obj, dim, selector):
    """Leave flags raised if they do not match the integer selector."""
    
    # Quit early if there are currently no pixels in the selection
    if sel_obj.num_flags == 0:
        return sel_obj
    
    # These are the coords to match against
    coords = sel_obj.indexset.loc[:, dim]
    
    start, stop = sel_obj.start, sel_obj.stop
    
    if sel_obj.has_flags:
        flags = sel_obj.flags
        n_flags = sel_obj.num_flags
    else:
        n_flags = stop - start
        flags = np.ones(n_flags, numba.boolean)
    
    first_set = False
    n_unset = 0
    for i, j in enumerate(range(start, stop)):
        if flags[i]:
            if selector == coords[j]:
                flags[i] = False
                n_unset += 1
            else:
                if first_set:
                    last = j
                else:
                    first = j
                    last = j
                    first_set = True
    
    n_flags -= n_unset
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
def _omit_range_unsorted(sel_obj, dim, selector):
    """Leave flags raised if they are not in a doubly-inclusive range."""
    
    # Quit early if there are currently no pixels in the selection
    if sel_obj.num_flags == 0:
        return sel_obj
    
    # These are the coords to match against
    coords = sel_obj.indexset.loc[:, dim]
    
    start, stop = sel_obj.start, sel_obj.stop
    
    if sel_obj.has_flags:
        flags = sel_obj.flags
        n_flags = sel_obj.num_flags
    else:
        n_flags = stop - start
        flags = np.ones(n_flags, numba.boolean)
    
    lo, hi = selector
    first_set = False
    n_unset = 0
    for i, j in enumerate(range(start, stop)):
        if flags[i]:
            if lo <= coords[j] <= hi:
                flags[i] = False
                n_unset += 1
            else:
                if first_set:
                    last = j
                else:
                    first = j
                    last = j
                    first_set = True
    
    n_flags -= n_unset
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
def _omit_above_unsorted(sel_obj, dim, selector):
    """Leave flags raised if they are not greater than selector[0]."""
    
    # Quit early if there are currently no pixels in the selection
    if sel_obj.num_flags == 0:
        return sel_obj
    
    # These are the coords to match against
    coords = sel_obj.indexset.loc[:, dim]
    
    start, stop = sel_obj.start, sel_obj.stop
    
    if sel_obj.has_flags:
        flags = sel_obj.flags
        n_flags = sel_obj.num_flags
    else:
        n_flags = stop - start
        flags = np.ones(n_flags, numba.boolean)
    
    thresh = selector[0]
    first_set = False
    n_unset = 0
    for i, j in enumerate(range(start, stop)):
        if flags[i]:
            if thresh <= coords[j]:
                flags[i] = False
                n_unset += 1
            else:
                if first_set:
                    last = j
                else:
                    first = j
                    last = j
                    first_set = True
    
    n_flags -= n_unset
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
def _omit_below_unsorted(sel_obj, dim, selector):
    """Leave flags raised if they are not less than selector[1]."""
    
    # Quit early if there are currently no pixels in the selection
    if sel_obj.num_flags == 0:
        return sel_obj
    
    # These are the coords to match against
    coords = sel_obj.indexset.loc[:, dim]
    
    start, stop = sel_obj.start, sel_obj.stop
    
    if sel_obj.has_flags:
        flags = sel_obj.flags
        n_flags = sel_obj.num_flags
    else:
        n_flags = stop - start
        flags = np.ones(n_flags, numba.boolean)
    
    thresh = selector[1]
    first_set = False
    n_unset = 0
    for i, j in enumerate(range(start, stop)):
        if flags[i]:
            if thresh >= coords[j]:
                flags[i] = False
                n_unset += 1
            else:
                if first_set:
                    last = j
                else:
                    first = j
                    last = j
                    first_set = True
    
    n_flags -= n_unset
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
def _omit_array_unsorted(sel_obj, dim, selector):
    """Leave flags raised if they are not less than selector[1]."""
    
    # Quit early if there are currently no pixels in the selection
    if sel_obj.num_flags == 0:
        return sel_obj
    
    # These are the coords to match against
    coords = sel_obj.indexset.loc[:, dim]
    
    start, stop = sel_obj.start, sel_obj.stop
    
    if sel_obj.has_flags:
        flags = sel_obj.flags
        n_flags = sel_obj.num_flags
    else:
        n_flags = stop - start
        flags = np.ones(n_flags, numba.boolean)
    
    first_set = False
    n_unset = 0
    for i, j in enumerate(range(start, stop)):
        if flags[i]:
            if val_in_array(coords[j], selector):
                flags[i] = False
                n_unset += 1
            else:
                if first_set:
                    last = j
                else:
                    first = j
                    last = j
                    first_set = True
    
    n_flags -= n_unset
    if n_flags:
        assert first_set
        last += 1
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
def _omit_int(sel_result, dim, selector):
    if dim == 0:
        return _omit_range_sorted(sel_result, dim, (selector, selector))
    else:
        return _omit_int_unsorted(sel_result, dim, selector)


@numba.njit
def _omit_above(sel_result, dim, selector):
    if dim == 0:
        hi = sel_result.indexset.loc[-1, 0]  # Largest dim-0 coordinate
        return _omit_range_sorted(sel_result, dim, (selector[0], hi))
    else:
        return _omit_above_unsorted(sel_result, dim, selector)


@numba.njit
def _omit_below(sel_result, dim, selector):
    if dim == 0:
        lo = sel_result.indexset.loc[0, 0]  # Smallest dim-0 coordinate
        return _omit_range_sorted(sel_result, dim, (lo, selector[1]))
    else:
        return _omit_below_unsorted(sel_result, dim, selector)


@numba.njit
def _omit_range(sel_result, dim, selector):
    if dim == 0:
        return _omit_range_sorted(sel_result, dim, selector)
    else:
        return _omit_range_unsorted(sel_result, dim, selector)


@numba.njit
def _omit_array(sel_result, dim, selector):
    return _omit_array_unsorted(sel_result, dim, selector)


from .sel import all_int_types, none_int_types, int_none_types, int_int_types

@numba.generated_jit(nopython=True)
def _omit(sel_result, dim, selector):
    if selector in all_int_types:
        return lambda sel_result, dim, selector: _omit_int(
            sel_result, dim, selector)
    elif selector in none_int_types:
        return lambda sel_result, dim, selector: _omit_below(
            sel_result, dim, selector)
    elif selector in int_none_types:
        return lambda sel_result, dim, selector: _omit_above(
            sel_result, dim, selector)
    elif selector in int_int_types:
        return lambda sel_result, dim, selector: _omit_range(
            sel_result, dim, selector)
    elif isinstance(selector, numba.types.Array):
        return lambda sel_result, dim, selector: _omit_array(
            sel_result, dim, selector)
    else:
        raise TypeError("Cannot use `sel` with the given arguments.")