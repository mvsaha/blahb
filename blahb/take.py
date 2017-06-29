"""
Selecting locations in a IndexSet based on their sorted location.
"""

import numba
from .flags import *
from .indexset import IndexSet


@numba.njit
def take_with_slice(indexset, where):
    """Take from an IndexSet between a start and stop location."""
    return IndexSet(indexset.loc[where], SORTED | UNIQUE)


@numba.njit
def take_range(indexset, where):
    """Take from an IndexSet between a start and stop location."""
    loc = indexset._loc[where[0]:where[1]]
    ret = IndexSet(loc, SORTED | UNIQUE)
    ret._encoding = indexset._encoding
    if indexset.data is not None:
        ret._data = indexset.data[where[0]:where[1]]
    return ret


@numba.njit
def take_with_ints(indexset, where):
    """Take a from an IndexSet with a sorted, unique_ array of locations.
    
    Arguments
    ---------
    indexset : IndexSet instance
    locations : 1d array of ints
        The locations of coordinates to take. The indices must be sorted and
        and unique.
    
    Returns
    -------
    An IndexSet containing only the coordinates at `locations`.
    """
    if where[0] < 0 or where[-1] >= indexset.n:
        raise IndexError("locations to take are out of bounds.")
    
    loc = indexset._loc[where]
    ret = IndexSet(loc, SORTED | UNIQUE)
    ret._encoding = indexset._encoding
    if indexset.data is not None:
        ret._data = indexset.data[where]
    return ret


@numba.njit
def take_with_int(indexset, where):
    """Take a single integer location from an IndexSet."""
    loc = indexset._loc[where:where + 1]
    ret = IndexSet(loc, SORTED | UNIQUE)
    ret._encoding = indexset._encoding
    if indexset.data is not None:
        ret._data = indexset.data[where:where + 1]
    return ret


@numba.njit
def take_with_bools(indexset, where):
    loc = indexset._loc[where]
    ret = IndexSet(loc, SORTED | UNIQUE)
    ret._encoding = indexset._encoding
    if indexset.data is not None:
        ret._data = indexset.data[where]
    return ret


@numba.njit
def raise_take_typeerror():
    raise TypeError('Invalid array type with take.')


@numba.generated_jit(nopython=True)
def take_(indexset, where):
    """Extract locations from an IndexSet based on their sorted position.
    
    Argument
    --------
    where : slice | int | array of int
        The positions to extract.
    
    Returns
    -------
    An IndexSet with the locations at the given position.
    """
    if isinstance(where, numba.types.UniTuple):
        return lambda indexset, where: take_range(indexset, where)
    if isinstance(where, numba.types.SliceType):
        return lambda indexset, where: take_with_slice(indexset, where)
    elif isinstance(where, numba.types.Integer):
        return lambda indexset, where: take_with_int(indexset, where)
    elif isinstance(where, numba.types.Array):
        if isinstance(where.dtype, numba.types.Boolean):
            return lambda indexset, where: take_with_bools(indexset, where)
        elif isinstance(where.dtype, numba.types.Integer):
            return lambda indexset, where: take_with_ints(indexset, where)
        else:
            return lambda indexset, where: raise_take_typeerror()
    else:
        return lambda indexset, where: raise_take_typeerror()
    