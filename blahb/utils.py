"""
Miscellaneous utility functions.
"""
import numpy as np
import numba


@numba.njit
def enlarge_array(arr, size=-1):
    """Enlarge an array to given size, or double the size
    if no size is given.
    """
    if size < 0:
        size = arr.size * 2
    elif size < arr.size:
        raise ValueError("final size less than current size.")
    ret = np.empty(size, dtype=arr.dtype)
    ret[:arr.size] = arr
    return ret


@numba.njit
def enlarge_mat(mat, size=-1):
    """Enlarge an 2d array along the lowest dimension to given size, or double
    the size if no size is given."""
    if size < 0:
        size = mat.shape[0] * 2
    ret = np.empty((size, mat.shape[1]), dtype=mat.dtype)
    ret[:mat.shape[0]] = mat
    return ret


@numba.njit
def widen_bounds(a_bounds, b_bounds):
    """Find the inclusive hyperrect that bounds both hyperrects.
    
    Arguments
    ---------
    a_bounds, b_bounds : ndim X 2 int32 arrays
        Index [dim, 0] gives the lower bounds along each dimension "dim".
        Index [dim, 1] gives the upper bounds along each dimension "dim".
        Each bounding value is inclusive.
    
    Note
    ----
    This function assumes that a_bounds can be overwritten. If this
    is not True you should copy a_bounds before passing it into
    this function.
    """
    assert a_bounds.ndim == b_bounds.ndim
    assert a_bounds.shape[0]  == b_bounds.shape[0]
    assert a_bounds.shape[1] == b_bounds.shape[1]
    
    for dim in range(a_bounds.shape[0]):
        a_bounds[dim, 0] = min(a_bounds[dim, 0], b_bounds[dim, 0])
        a_bounds[dim, 1] = max(a_bounds[dim, 1], b_bounds[dim, 1])
    return a_bounds


_brute_force_thresh = 64


@numba.njit(nogil=True)
def exponential_search(arr, target, start=0, stop=-1):
    """Cache friendlier binary search on sorted array subsets.

    Arguments
    ---------
    arr : array that is sorted ascending over the range [start, stop)
    target : scalar value to search for in the array
    start : int
        Optional offset to begin searching on. By default searching will
        begin at the beginning of the array.
    stop : int
        The last location to consider placing target, if it exceeds all
        values on the half-open range arr[start:stop]
    Returns
    -------
    index : int
        The insertion location: the location that the target value could be
         inserted, before all equal elements, so that the array subset
         remains sorted ascending.

    Note
    ----
    The input array does not need to be fully sorted, as long as the subset
    arr[start:stop] is sorted. The
    """
    
    if arr.size >= 2147483647:
        raise ValueError("Arrays with >2147483647 elements not supported.")
    
    # Stop is the maximum index this function will return
    if stop < 0:
        stop = arr.size
    
    # And start is the minimum
    if target <= arr[start]:
        return start
    
    i = np.int32(start)
    j = np.int32(min(start + _brute_force_thresh, stop))
    ramp = _brute_force_thresh * 2
    
    # Loop until j exceeds target
    while j < stop and arr[j] < target:
        # print(i, j, ' - ', arr[i], arr[j - 1], 'ramp', ramp)
        i, j = j, np.int32(min(j + ramp, stop))
        ramp *= 2
    
    # All values are less than the target, but don't search for this
    # value until we have ramped up to it.
    if arr[stop - 1] < target:
        return stop
    
    # Binary search
    while j - i > _brute_force_thresh:
        pivot = np.int32(((j - i) // 2) + i)
        
        if arr[pivot] >= target:
            i, j = i, pivot
        else:
            i, j = pivot, j
    
    # Brute force the final subset
    for k in range(i, j):
        if target <= arr[k]:
            return k
    return j


@numba.njit
def searchsorted(arr, search_for):
    m = 0
    ret = np.empty(search_for.size, np.uint32)
    for i, sf in enumerate(search_for):
        m = exponential_search(arr, sf, start=m)
        ret[i] = m
    return ret


@numba.njit
def to_tuple(*args):
    return args


def is_iterable(x):
    try:
        iter(x)
        return True
    except:
        return False


def repeat(x):
    while True:
        yield x


def expand_maybe(x, ndim):
    """Expands a value to a tuple of values with length ndim, or returns
    x if it already has len == ndim.
    Raises DimensionError if x is already an iterable, but len != ndim.
    """
    if is_iterable(x):
        if len(x) != ndim:
            raise ValueError('x must be a scalar or length {}, not length'
            '{}'.format(ndim, x))
        else:
            return tuple(x)
    else:
        return (x,) * ndim


@numba.njit
def lex_less_Nd(a, b):
    """Determine if array 'a' lexicographically less than 'b'."""
    for _a, _b in zip(a, b):
        if _a < _b:
            return True
        elif _a > _b:
            return False
    return False


@numba.njit
def eq_Nd(a, b):
    """Determine if all of the elements of 'a' are equal to 'b'."""
    for _a, _b in zip(a, b):
        if _a != _b:
            return False
    return True


@numba.njit
def passthrough(x):
    return x


@numba.njit
def convert_to_array(x):
    return np.array(x)


@numba.generated_jit
def to_array(x):
    """Convert a 1d tuple or numpy array to a numpy array."""
    if isinstance(x, numba.types.Array):
        return lambda x: passthrough(x)
    else:
        return lambda x: convert_to_array(x)