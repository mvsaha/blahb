import numpy as np
import numba

from .utils import enlarge_array
from .border import _decrement_direction, _increment_direction
from .border import _invert_direction
from .flags import SORTED_UNIQUE, DATA_NANFIRST
from .indexset import IndexSet

from .setops import asymmetric_difference_, union_

@numba.njit(nogil=True)
def buffer_(indexset, offsets, with_original=True):
    """Find the buffer around an IndexSet given Neighborhood offsets."""
    
    if indexset.ndim is not 2:
        raise ValueError("buffer_: indexset.ndim does not equal 2.")
    
    loc = indexset.loc
    combined = IndexSet(loc[:0], SORTED_UNIQUE)  # Empty IndexSet
    
    if indexset.data is None:
        MERGE = None
    else:
        MERGE = np.full(indexset.data.shape[1], DATA_NANFIRST,
                        dtype=np.uint8)
    
    for tup in zip(*offsets):
        shift = np.array(tup, dtype=np.int32)
        neighborset = IndexSet(loc + shift, SORTED_UNIQUE)
        combined = union_(combined, neighborset, MERGE)
    
    if with_original:
        combined = union_(indexset, combined, MERGE)
    
    else:
        # Don't worry about data, there are not pixels with data
        combined = asymmetric_difference_(combined, indexset)
    
    return combined


@numba.njit
def buffer2d_(border, neighbors):
    """Add a border around True pixels in an image.
    
    Arguments
    ---------
    border : tuple(array, array)
        Tuple of arrays corresponding to the border pixels of
        a single contiguous region (blob). The border pixels
        must be in the form returned by `border2d_` and
        determined using the same neighbor offsets that are in
        `neighbor`. Border must start with the top-most, left-
        most pixel in the blob.
    neighbors : tuple(array, array)
        Tuple of arrays corresponding to the y and x offsets
        of the central pixel in rotational order. The central
        pixel of a neighborhood, (0, 0), should be omitted.
        The neighborhood should also be symmetric, so that the
        negation of each neighbor coorindate pair is also present.

    Returns
    -------
    Buffer pixels containing the outside buffer containing all
    neighbor pixels, but no internal pixels.
    """
    y_border, x_border = border
    n_border = len(y_border)
    
    y_neighbors, x_neighbors = neighbors
    n_neighbors = len(y_neighbors)
    
    if n_border == 1:
        return y_neighbors + y_border[0], x_neighbors + x_border[0]
    else:
        assert len(y_neighbors) >= 3
        assert (y_neighbors[0] != y_neighbors[-2] or
                x_neighbors[0] != x_neighbors[-2])
    
    # `direction` should correspond to the previous cell
    direction = 0
    y = y_border[0] + y_neighbors[direction]
    x = x_border[0] + x_neighbors[direction]
    
    while y != y_border[-2] or x != x_border[-2]:
        direction = _decrement_direction(direction, n_neighbors)
        y = y_border[0] + y_neighbors[direction]
        x = x_border[0] + x_neighbors[direction]
    
    # direction points from neighbors[-2] to neighbors[0]
    direction = _increment_direction(direction, n_neighbors)
    
    y_buffer = np.zeros(64, y_border.dtype)
    x_buffer = np.zeros(64, y_border.dtype)
    n_buffer = 0
    
    for i in range(n_border - 1):
        
        y = y_border[i]
        x = x_border[i]
        
        y_next = y_border[i + 1]
        x_next = x_border[i + 1]
        
        y_test = y + y_neighbors[direction]
        x_test = x + x_neighbors[direction]
        
        while y_test != y_next or x_test != x_next:
            
            # Write the last ones, expanding the buffer if necessary
            if n_buffer >= y_buffer.size:
                y_buffer = enlarge_array(y_buffer)
                x_buffer = enlarge_array(x_buffer)
            
            y_buffer[n_buffer] = y_test
            x_buffer[n_buffer] = x_test
            n_buffer += 1
            
            # Point to the next neighbor
            direction = _increment_direction(direction, n_neighbors)
            
            y_test = y + y_neighbors[direction]
            x_test = x + x_neighbors[direction]
        
        # One more increment and inversion brings us back to the
        # next border pixel pointing back at this one
        direction = _invert_direction(direction, n_neighbors)
        
        # Start off not pointing back at the last neighbor
        direction = _increment_direction(direction, n_neighbors)
    
    return y_buffer[:n_buffer], x_buffer[:n_buffer]