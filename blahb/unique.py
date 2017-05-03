"""
Tools for finding unique rows of an array.
"""
import numpy as np
import numba


@numba.njit
def _unique_locations_in_sorted(loc):
    """Find the starting position for each run of unique_ coordinates.

    Arguments
    ---------
    loc : 2d array of ints
        Each row represents an N-d coordinate. This array must be
        lexicographically sorted.

    Returns
    -------
    1d array of bools the size matching the first dimension size of
    loc, set to True at the first of each unique_ row of location.
    """
    n = loc.shape[0]
    flags = np.zeros(n, dtype=numba.boolean)
    flags[0] = True
    for i in range(1, n):
        if np.any(loc[i] != loc[i - 1]):
            flags[i] = True
    return flags


@numba.njit
def _unique_mapping_from_flags(flags):
    """Find the final location of each (possibly repeated) coordinate
    in a lex-sorted array if it were made unique_.

    Arguments
    ---------
    flags : 1d bool array
        The same length as the location array to be made unique_, set
        to True at the start of each run of repeated coordinates.

    Returns
    -------
    Mapping from each coordinate to the final coordinate.

    Example
    -------
    The inputs (only two dimensions shown, but any number are allowed)
    x = [1,1,1,1,1,2,2,2,2,2]
    y = [3,3,4,4,5,3,5,6,6,7]
    arrs = x, y

    The input of this function would be:
        flags = [1,0,1,0,1,1,1,1,0,1] (as bools)

    The unique_ version of this would be:
        x_unique = [1,1,1,2,2,2,2]
        y_unique = [3,4,5,3,5,6,7]

    The output of this function would be:
        mapping = [0,0,1,1,2,3,4,5,6,6,7]
    which specifies the position of each coordinate in the input to the
    corresponding position in the non-repeating output.
    """
    n = flags.size
    mapping = np.zeros(n, np.uint64)
    mapping[0] = 0
    map_index = 0
    for i in range(1, n):
        if flags[i]:
            map_index += 1
        mapping[i] = map_index
    return mapping


@numba.njit
def _unique_mapping_from_locs(indices, n):
    """Find the final location of each (possibly repeated) coordinate
    in a lex-sorted array if it were made unique_.

    Arguments
    ---------
    indices : 1d int array
        Indices giving the start of each run of repeated coordinates
    n : int
        The total size of the coord array to be made unique_

    Returns
    -------
    Mapping from each coordinate to the final coordinate.
    """
    map_index = 0
    mapping = np.zeros(n, np.uint64)
    for i in range(1, indices.size):
        mapping[indices[i - 1]:indices[i]] = map_index
        map_index += 1
    mapping[indices[-1]:] = map_index
    return mapping