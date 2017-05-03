"""
Finding the border pixels of an IndexSet
"""
import numpy as np
import numba

from .utils import enlarge_array


@numba.njit
def _invert_direction(direction, n):
    return (direction + (n // 2)) % n


@numba.njit
def _increment_direction(direction, n):
    direction += 1
    if direction == n:
        return 0
    return direction


@numba.njit
def _decrement_direction(direction, n):
    direction -= 1
    if direction < 0:
        return n - 1
    return direction


@numba.njit
def border2d_(img, start, y_neighbors, x_neighbors, y_offset=0, x_offset=0):
    """img must be 2d and have more than one True value.
    start must be the (y,x) index of leftmost True pixel on the uppermost row
    """
    # TODO: Document
    y_saved = np.empty(2, dtype=np.int64)
    x_saved = np.empty(2, dtype=np.int64)
    
    n_neighbors = len(y_neighbors)
    assert len(x_neighbors) == n_neighbors
    y_start, x_start = start
    
    assert img[y_start, x_start]
    
    direction_start = 0
    y_test = y_start + y_neighbors[direction_start]
    x_test = x_start + x_neighbors[direction_start]
    
    # Search exhaustively for the start of the image.
    while not img[y_test, x_test]:
        direction_start = _decrement_direction(direction_start, n_neighbors)
        y_test = y_start + y_neighbors[direction_start]
        x_test = x_start + x_neighbors[direction_start]
    
    direction_start = _invert_direction(direction_start, n_neighbors)
    
    # Goal: Prove that incrementing direction is still valid for sparse
    # neighborhoods.
    direction = _increment_direction(direction_start, n_neighbors)
    y, x = y_start, x_start
    
    y_saved[0] = y + y_offset
    x_saved[0] = x + x_offset
    n = 1
    
    while y != y_start or x != x_start or direction != direction_start:
        direction = _invert_direction(direction, n_neighbors)
        direction = _increment_direction(direction, n_neighbors)
        y_test = y + y_neighbors[direction]
        x_test = x + x_neighbors[direction]
        while not img[y_test, x_test]:
            direction = _increment_direction(direction, n_neighbors)
            y_test = y + y_neighbors[direction]
            x_test = x + x_neighbors[direction]
        
        y, x = y_test, x_test
        
        if n >= y_saved.size:
            y_saved = enlarge_array(y_saved)
            x_saved = enlarge_array(x_saved)
            assert y_saved.size == x_saved.size
        assert y_saved.size > n
        y_saved[n] = y + y_offset
        x_saved[n] = x + x_offset
        n += 1
    
    return y_saved[:n], x_saved[:n]