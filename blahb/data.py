"""
Functions that allow propagation of associated data through set operations.
"""
import numpy as np
import numba
from .flags import *


@numba.njit
def _merge_indirect_min(contrib_flag, data_in, data_out):
    """Take the minimum value of contributing pixels, propagating NaNs."""
    j = 0
    d = np.float32(0)
    for i in range(contrib_flag.shape[0]):
        if contrib_flag[i]:
            if np.isnan(data_out[i]) or np.isnan(data_in[j]):
                data_out[i] = np.nan
            else: # Both values are non-NaN
                data_out[i] = min(data_out[i], data_in[j])
            j += 1
        else:
            data_out[i] = np.nan
    assert j == data_in.shape[0]


@numba.njit
def _merge_indirect_nanmin(contrib_flag, data_in, data_out):
    """Take the minimum value of contributing pixels, ignoring NaNs."""
    j = 0
    d = np.float32(0)
    for i in range(contrib_flag.shape[0]):
        if contrib_flag[i]:
            # Uses property min(NaN, value) -> value
            data_out[i] =  min(data_out[j], data_in[j])
            j += 1
    assert j == data_in.shape[0]


@numba.njit
def _merge_indirect_max(contrib_flag, data_in, data_out):
    """Take the maximum value of contributing pixels, propagating NaNs."""
    j = 0
    d = np.float32(0)
    for i in range(contrib_flag.shape[0]):
        if contrib_flag[i]:
            if np.isnan(data_out[i]) or np.isnan(data_in[j]):
                data_out[i] = np.nan
            else: # Both values are non-NaN
                data_out[i] = max(data_out[i], data_in[j])
            j += 1
        else:
            data_out[i] = np.nan
    assert j == data_in.shape[0]


@numba.njit
def _merge_indirect_nanmax(contrib_flag, data_in, data_out):
    """Take the maximum value of contributing pixels, ignoring NaNs."""
    j = 0
    d = np.float32(0)
    for i in range(contrib_flag.shape[0]):
        if contrib_flag[i]:
            # Uses property max(NaN, value) -> value
            data_out[i] = max(data_out[j], data_in[j])
            
            j += 1
    assert j == data_in.shape[0]


@numba.njit
def _merge_indirect_first(contrib_flag, data_in, data_out):
    """Take the first encountered non-NaN value of contributing pixels."""
    j = 0
    d = np.float32(0)
    for i in range(contrib_flag.shape[0]):
        if contrib_flag[i]:
            d = data_out[i]
            if np.isnan(d):
                data_out[i] = data_in[j]
            j += 1
    if not j == data_in.shape[0]:
        raise ValueError("Wrong input.")


@numba.njit
def _merge_indirect_last(contrib_flag, data_in, data_out):
    """Take the last encountered non-NaN value of contributing pixels."""
    j = 0
    d = np.float32(0)
    for i in range(contrib_flag.shape[0]):
        if contrib_flag[i]:
            d = data_in[j]
            if not np.isnan(d):
                data_out[i] = d
            j += 1
    assert j == data_in.shape[0]


@numba.njit
def _merge_indirect_sum(contrib_flag, data_in, data_out):
    """Take the sum of contributing pixels, propagating NaNs"""
    j = 0
    d = np.float32(0)
    for i in range(contrib_flag.shape[0]):
        if contrib_flag[i]:
            data_out[i] += data_in[j]
            j += 1
        else:
            data_out[i] = np.nan
    assert j == data_in.shape[0]


@numba.njit
def _merge_indirect_nansum(contrib_flag, data_in, data_out):
    """Take the sum of contributing non-NaN pixels."""
    j = 0
    d = np.float32(0)
    for i in range(contrib_flag.shape[0]):
        if contrib_flag[i]:
            d = data_in[j]
            if not np.isnan(d):
                if np.isnan(data_out[i]):
                    data_out[i] = d
                else:
                    data_out[i] += d
            j += 1
    assert j == data_in.shape[0]


@numba.njit
def merge_data_column_indirect(contrib, data_in, data_out, MERGE_DATA):
    """Merge a single column of data using a MERGE_DATA rule.
    
    Modifies `data_out`."""
    if MERGE_DATA == DATA_NANFIRST:
        _merge_indirect_first(contrib, data_in, data_out)
    elif MERGE_DATA == DATA_NANLAST:
        _merge_indirect_last(contrib, data_in, data_out)
    elif MERGE_DATA == DATA_SUM:
        _merge_indirect_sum(contrib, data_in, data_out)
        # NOT equivaled to:
        #data_out[contrib] = np.add(data_in, data_out[contrib])
    elif MERGE_DATA == DATA_NANSUM:
        _merge_indirect_nansum(contrib, data_in, data_out)
    elif MERGE_DATA == DATA_MIN:
        _merge_indirect_min(contrib, data_in, data_out)
        # NOT equivaled to:
        #data_out[contrib] = np.minimum(data_in, data_out[contrib])
    elif MERGE_DATA == DATA_NANMIN:
        data_out[contrib] = np.fmin(data_in, data_out[contrib])
    elif MERGE_DATA == DATA_MAX:
        _merge_indirect_max(contrib, data_in, data_out)
        # NOT equivaled to:
        #data_out[contrib] = np.maximum(data_in, data_out[contrib])
    elif MERGE_DATA == DATA_NANMAX:
        data_out[contrib] = np.fmax(data_in, data_out[contrib])
    else:
        raise ValueError("Invalid MERGE_DATA flag.")


from .flags import DATA_NANFIRST
__DEFAULT_MERGE_ACTION = DATA_NANFIRST


@numba.njit(
    (numba.optional(numba.float32[:, :]),
     numba.boolean[:],
     numba.optional(numba.float32[:, :]),
     numba.optional(numba.uint8[:]))
)
def merge_data_indirect(data_in, contrib, existing_data, MERGE_ACTION=None):
    """
    Arguments
    ---------
    data_in : [None] | 2d float32 matrix
        Input dataset
    contrib : 1d bool array
        The size of this array must match the lowest dimension shape
        of existing_data, if existing_data is not None. This indicates where
        values from data in should be extracted. The number of True values
        in this array should be
        IF data_in is None then this
        argument is ignored. It must still be passed in (cannot be None) due
        to typing restrictions. (a numba indexer cannot be optional and this
        arguments indexes existing_data).
    existing_data : None | [2d float32 matrix]
        If this is not None, it should have a lowest dimension shape
        equal to the number of True values in contrib.
    MERGE_ACTION : [None] | 1d uint8 array
        The kind of merge to do on each column of existing_data to
        combine it with data_int. This will raise an error. See the
        definitions in numblahb.bits. The default when None is given
        is to keep the first non-NaN value in each position
        (_BLAHB_DATA_FIRST). If an array with a single value is passed in then
        it will be applied to all columns of the data_in. Due to typing
        restrictions this argument must be an array, it cannot be a scalar
        (uint8 or otherwise).
    
    Returns
    -------
    data : None | 2d float32 matrix
        The data extracted using contrib. If existing data was given,
        this array is the SAME array, and has been modified by this
        function. Otherwise a new array has been created. If the input
        data is None and no existing data was passed in then the return
        value is None.

    This is designed to be used sequentially on data from multiple
    IndexSets.
    """
    if data_in is None:
        if existing_data is not None:
            ndim = existing_data.shape[1]
            if MERGE_ACTION is None:
                if DATA_DEFAULT & DATA_NANS_PROPAGATE:
                    existing_data[:, :] = np.nan
                else:
                    pass # Existing data does not change
            elif (MERGE_ACTION.size == 1 and
                          MERGE_ACTION[0] & DATA_NANS_PROPAGATE):
                existing_data[:, :] = np.nan
            elif MERGE_ACTION.size == ndim:
                for col, M in enumerate(MERGE_ACTION):
                    if M & DATA_NANS_PROPAGATE:
                        existing_data[:, col] = np.nan
            else:
                raise ValueError("Number of elements in MERGE do not match"
                                 "the dimensionality of input PixelSets.")
            return existing_data
        else:
            return None
    
    if not data_in.ndim == 2:
        raise ValueError("data_in must be 2d.")
    
    ncol = data_in.shape[1]
    if existing_data is None:  # Form the output array
        existing_data = np.full((contrib.size, ncol), np.nan, dtype=np.float32)
        existing_data[contrib] = data_in
        return existing_data
    
    if MERGE_ACTION is None:
        for col in range(ncol):
            merge_data_column_indirect(contrib,
              data_in[:, col], existing_data[:, col], DATA_DEFAULT)
    elif MERGE_ACTION.size == 1:
        for col in range(ncol):
            merge_data_column_indirect(contrib, data_in[:, col],
              existing_data[:, col], MERGE_ACTION[0])
    elif MERGE_ACTION.size == ncol:
        for col in range(ncol):
            merge_data_column_indirect(contrib, data_in[:, col],
              existing_data[:, col], MERGE_ACTION[col])
    else:
        raise ValueError("Number of elements in MERGE do not match"
                         "the dimensionality of input PixelSets.")
    return existing_data


@numba.njit
def _merge_contrib_nanmin(contrib_flag, data_in, data_out):
    """Find the minimum of all contributing pixels, propagating NaN."""
    # assert contrib_flag.size == data_in.shape[0] # Should be True
    # assert np.sum(contrib_flag) == data_out.shape[0] # Should be True
    n_out = 0
    for i in range(contrib_flag.size):
        if contrib_flag[i]:
            d = data_out[n_out]
            if np.isnan(d):
                data_out[n_out] = data_in[i]
            elif not np.isnan(data_in[i]):
                data_out[n_out] =  min(d, data_in[i])
            # else: Both elements are NaN, no change to data_out needed.
            n_out += 1
    assert n_out == data_out.shape[0]


@numba.njit
def _merge_contrib_nanmax(contrib_flag, data_in, data_out):
    """Find the minimum of all contributing pixels, propagating NaN."""
    # assert contrib_flag.size == data_in.shape[0] # Should be True
    # assert np.sum(contrib_flag) == data_out.shape[0] # Should be True
    n_out = 0
    for i in range(contrib_flag.size):
        if contrib_flag[i]:
            d = data_out[n_out]
            if np.isnan(d):
                data_out[n_out] = data_in[i]
            elif not np.isnan(data_in[i]):
                data_out[n_out] =  max(d, data_in[i])
            # else: Both elements are NaN, no change to data_out needed.
            n_out += 1
    assert n_out == data_out.shape[0]


@numba.njit
def _merge_contrib_nansum(contrib_flag, data_in, data_out):
    """Find the minimum of all contributing pixels, propagating NaN."""
    # assert contrib_flag.size == data_in.shape[0] # Should be True
    # assert np.sum(contrib_flag) == data_out.shape[0] # Should be True
    n_out = 0
    for i in range(contrib_flag.size):
        if contrib_flag[i]:
            d = data_in[i]
            if not np.isnan(d):
                if np.isnan(data_out[n_out]):
                    data_out[n_out] = d
                else:
                    data_out[n_out] += d
            # else: Both elements are NaN, no change to data_out needed.
            n_out += 1
    assert n_out == data_out.shape[0]


@numba.njit
def _merge_contrib_first(contrib_flag, data_in, data_out):
    n_out = 0
    for i in range(contrib_flag.size):
        if contrib_flag[i]:
            if np.isnan(data_out[n_out]):
                data_out[n_out] = data_in[i]
            n_out += 1
    assert n_out == data_out.shape[0]


@numba.njit
def _merge_contrib_last(contrib_flag, data_in, data_out):
    n_out = 0
    for i in range(contrib_flag.size):
        if contrib_flag[i]:
            if not np.isnan(data_in[i]):
                data_out[n_out] = data_in[i]
            n_out += 1
    assert n_out == data_out.shape[0]


@numba.njit
def merge_data_column_direct(contrib, data_in, data_out, MERGE_DATA):
    if MERGE_DATA == DATA_NANFIRST:
        _merge_contrib_first(contrib, data_in, data_out)
    elif MERGE_DATA == DATA_NANLAST:
        _merge_contrib_last(contrib, data_in, data_out)
    elif MERGE_DATA == DATA_SUM:
        np.add(data_in[contrib], data_out, data_out)
    elif MERGE_DATA == DATA_NANSUM:
        _merge_contrib_nansum(contrib, data_in, data_out)
    elif MERGE_DATA == DATA_MIN:
        np.minimum(data_in[contrib], data_out, data_out)
    elif MERGE_DATA == DATA_NANMIN:
        np.fmin(data_in[contrib], data_out, data_out)
    elif MERGE_DATA == DATA_MAX:
        np.maximum(data_in[contrib], data_out, data_out)
    elif MERGE_DATA == DATA_NANMAX:
        np.fmax(data_in[contrib], data_out, data_out)
    else:
        raise ValueError("Invalid MERGE_DATA flag.")


@numba.njit(
    (numba.optional(numba.float32[:, :]),
    numba.boolean[:],
    numba.optional(numba.float32[:, :]),
    numba.optional(numba.uint8[:]))
)
def merge_data_direct(data_in, contrib, existing_data=None, MERGE_ACTION=None):
    """
    Arguments
    ---------
    data_in : [None] | 2d float32 matrix
        Input dataset
    contrib : 1d bool array
        The size of this array must match the lowest dimension shape
        of data_in. This indicates where values from data in should be
        extracted. IF data_in is None then this argument is ignored. It must
        still be passed in (cannot be None) due to typing restrictions.
        (a numba indexer cannot be optional and this arguments indexes data_in)
    existing_data : None | [2d float32 matrix]
        If this is not None, it should have a lowest dimension shape
        equal to the number of True values in contrib.
    MERGE_ACTION : [None] | 1d uint8 array
        The kind of merge to do on each column of existing_data to
        combine it with data_int. See the definitions in blahb.flags.
        The default when None is given is to keep the first non-NaN value in
        each position (_BLAHB_DATA_FIRST). If an array with a single value
        is passed in then it will be applied to all columns of the data_in.
        Due to typing restrictions this argument must be an array, it cannot
        be a scalar (uint8 or otherwise).
    
    Returns
    -------
    data : None | 2d float32 matrix
        The data extracted using contrib. If existing data was given,
        this array is the SAME array, and has been modified by this
        function. Otherwise a new array has been created. If the input
        data is None and no existing data was passed in then the return
        value is None.
    
    This is designed to be used sequentially on data from multiple
    IndexSets.
    """
    if data_in is None:
        if existing_data is not None:
            n_cols = existing_data.shape[1]
            if MERGE_ACTION is None:
                if DATA_DEFAULT & DATA_NANS_PROPAGATE:
                    existing_data[:, :] = np.nan
                else:
                    pass
            elif MERGE_ACTION.size == 1:
                if MERGE_ACTION[0] & DATA_NANS_PROPAGATE:
                    existing_data[:, :] = np.nan
                else:
                    pass # No change to existing data
            elif MERGE_ACTION.size == n_cols:
                for col, M in enumerate(MERGE_ACTION):
                    if M & DATA_NANS_PROPAGATE:
                        existing_data[:, col] = np.nan
            else:
                raise ValueError("Number of elements in MERGE does not match"
                                 " the dimensionality of input PixelSets.")
            return existing_data
        else:
            return None
    
    if data_in.ndim != 2:
        raise TypeError("Dimensionality of data must be 2d.")
    if existing_data is None:  # Form the array if all inputs
        existing_data = data_in[contrib]
        return existing_data
    
    n_cols = data_in.shape[1]
    if MERGE_ACTION is None:
        for col in range(n_cols):
            merge_data_column_direct(contrib, data_in[:, col],
              existing_data[:, col], DATA_DEFAULT)
    elif MERGE_ACTION.size == 1:
        for col in range(n_cols):
            merge_data_column_direct(contrib, data_in[:, col],
              existing_data[:, col], MERGE_ACTION[0])
    elif MERGE_ACTION.size == data_in.shape[1]:
        for col in range(n_cols):
            merge_data_column_direct(contrib, data_in[:, col],
              existing_data[:, col], MERGE_ACTION[col])
    else:
        raise ValueError("Number of elements in MERGE do not match"
                         "the dimensionality of input PixelSets.")
    return existing_data


@numba.njit( (numba.optional(numba.uint8[:]), ) )
def all_short_circuit_merges(MERGE=None):
    """Determine if all MERGE flags are can be skipped if data is None.
    
    This allows us to skip the data merge step if any of the contributing
    data is None.
    
    However, if any of the columns are True, then we must propagate all data
    columns.
    """
    if MERGE is None:
        return numba.boolean(DATA_DEFAULT & DATA_NANS_PROPAGATE)
    for M in MERGE:
        if not (M & DATA_NANS_PROPAGATE):
            return False
    return True


@numba.njit( (numba.optional(numba.uint8[:]), ) )
def order_does_not_matter(MERGE=None):
    """Determine if a the merge order matters from the input flags.
    
    In some cases, like taking the MAX or SUM of multiple data values at the
    same location, the order in which we introduce arguments does not change
    the result. In this case we are allowed to rearrange the input values
    for more efficient computation.
    """
    if MERGE is None:
        M = DATA_DEFAULT
        return M == DATA_NANFIRST or M == DATA_NANLAST
    for M in MERGE:
        if M == DATA_NANFIRST or M == DATA_NANLAST:
            return False
    return True