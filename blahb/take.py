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
    coords = indexset.loc[where[0]:where[1]]
    ret = IndexSet(coords, SORTED | UNIQUE)
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
        The locations of coordinates to take. This must be sorted and unique_.
    
    Returns
    -------
    An IndexSet containing only the coordinates at `locations`.
    """
    if where[0] < 0 or where[-1] >= indexset.n:
        raise IndexError("locations to take are out of bounds.")
    
    coords = indexset.loc[where]
    ret = IndexSet(coords, SORTED | UNIQUE)
    if indexset.data is not None:
        ret._data = indexset.data[where]
    return ret


@numba.njit
def take_with_int(indexset, where):
    """Take a single integer location from a """
    loc = indexset.loc[where:where + 1]
    ret = IndexSet(loc, SORTED | UNIQUE)
    if indexset.data is not None:
        ret._data = indexset.data[where:where + 1]
    return ret


from numba.types import i1, i2, i4, i8, u1, u2, u4, u8


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
        return lambda indexset, where: take_with_ints(indexset, where)
    else:
        raise TypeError("Cannot use `take` with the given type.")
    