"""
Tools for converting from IndexSet (spatially explicit) to image (implicit).
"""

import numpy as np
import numba

from .utils import expand_maybe
from .indexset import IndexSet
from .bits import SORTED_UNIQUE

def image(indexset, padding=0, data_col=None):
    """Convert an indexset into a cropped boolean image.

    Arguments
    ---------
    indexset : IndexSet instance
    padding : int | sequence of ints
        Padding to apply along each axis of the image
    data_col : [None] | int
        Which column of data the image should be filled with. By default
        no data column is selected and the output is a boolean image that
        is True where the locations are True.

    Returns
    -------
    A boolean or floating point image array with the same number of
    dimensions as `indexset` that is has a shape matching the extent
    (bounds) of `indexset` (or larger if padding was specified).

    If `data_col` is None, then it is True at locations contained by
    `indexset` and False elsewhere.
    
    If `data_col` is specified, then the pixels contain data values
    from the `data_col`-th column of data and NaN elsewhere.
    """
    bnds = indexset.bounds
    if padding is not 0:
        padding = np.asarray(expand_maybe(padding, indexset.ndim))
        if not all(val >= 0 for val in padding):
            raise ValueError("Padding for images must be >=0.")
        shp = bnds[:, 1] - bnds[:, 0] + (2 * padding) + 1
        coords = tuple((indexset.loc - (bnds[:, 0] - padding)).T)
    else:
        shp = bnds[:, 1] - bnds[:, 0] + 1
        coords = tuple((indexset.loc - bnds[:, 0]).T)
    
    if data_col is not None:
        _img = np.full(shp, np.nan, dtype=np.float32)
        _img[coords] = indexset.data[:, data_col]
    else:
        _img = np.zeros(shp, dtype=np.bool)
        _img[coords] = True
    
    return _img


@numba.njit
def where_(img, offsets=0):
    """Build an IndexSet from the location of True pixels in an image.
    """
    loc = np.vstack(np.where(img)).T.copy().astype(np.int32)
    _offsets = np.array(offsets).T
    if _offsets.size and _offsets.ndim:
        loc += _offsets
    return IndexSet(loc, SORTED_UNIQUE)


@numba.njit
def where_data_(img, offsets=0):
    """Extract the non-NaN pixels of a floating point image.
    
    Arguments
    ---------
    img: array of floats
        The image to extract non-Nan pixels from.
    offsets :
        The origin (location of the "first" pixel) of the image.
    
    Returns
    -------
    IndexSet containing the locations of the non-NaN elements of the image
    with data containing the values of those non-NaN elements.
    """
    not_nan = np.isfinite(img)
    
    coords = np.where(not_nan)
    loc = np.vstack(coords).T.copy().astype(np.int32)
    
    img_flat = img.reshape(-1)
    not_nan_flat = not_nan.reshape(-1)
    data = img_flat[not_nan_flat].astype(np.float32)
    
    _offsets = np.array(offsets).T
    if _offsets.size and _offsets.ndim:
        loc += _offsets
    
    result = IndexSet(loc, SORTED_UNIQUE)
    result._data = data.reshape((-1, 1))
    return result