import numpy as np
import numba

from .timsort import timsort_
from .take import take_


@numba.njit(nogil=True)
def blobs_(indexset, labels):
    """Generate contiguous features from a labeled MultiBlob.
    
    Arguments
    ---------
    multiblob : A MultiBlob instance
        Generated from Neighborhood.label.
    
    Yields
    ------
    IndexSet instances with the contiguity defined in `multiblob`.
    """
    #labels = multiblob.labels.copy()
    sort_order = timsort_(labels)  # Sorts labels too
    start = 0
    start_label = labels[start]
    for i in range(labels.size):
        if labels[i] != start_label:
            yield take_(indexset, sort_order[start:i])
            start = i
            start_label = labels[i]
    yield take_(indexset, sort_order[start:])