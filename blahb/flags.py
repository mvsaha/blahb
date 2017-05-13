"""
Bit flag conventions.
"""
import numpy as np
import numba

__all__ = (
    'NO_FLAGS',
    'UNIQUE', 'unique_',
    'SORTED_UNIQUE',
    'SORTED', 'sorted_',
    'COPY', 'copy_',
    'KEEP_SORT_ORDER', 'keep_sort_order_',
    'CONSUME', 'consume_',
    
    'DATA_NANS_PROPAGATE',  #'_BLAHB_DATA_NONE',
    'DATA_NANSUM', 'data_nansum',
    'DATA_SUM', 'data_sum',
    'DATA_NANMAX', 'data_nanmax',
    'DATA_MAX', 'data_max',
    '_BLAHB_DATA_NANMIN', 'data_nanmin',
    '_BLAHB_DATA_MIN', 'data_min',
    'DATA_NANFIRST', 'data_nanfirst',
    'DATA_NANLAST', 'data_nanlast',
    'DATA_DEFAULT', 'data_default',
    'BLAHB_DATA_DEFAULT_ARRAY')

# ==========================================================================#
# Flags used when creating IndexSets. Some of these indicate to skip
# potentially costly preprocessing (like sorting and making unique_).
NO_FLAGS = np.uint8(0)
no_flags = numba.njit(lambda: NO_FLAGS)

UNIQUE = np.uint8(1)
unique_ = numba.njit(lambda: UNIQUE)
_BLAHB_NOT_UNIQUE = ~UNIQUE

SORTED = np.uint8(2)
sorted_ = numba.njit(lambda: SORTED)
_BLAHB_NOT_SORTED = ~SORTED

COPY = np.uint8(4)
copy_ = numba.njit(lambda: COPY)

KEEP_SORT_ORDER = np.uint8(8)
keep_sort_order_ = lambda: KEEP_SORT_ORDER

CONSUME = np.uint8(16)
consume_ = lambda: CONSUME


# This is commonly used
SORTED_UNIQUE = SORTED | UNIQUE
sorted_unique = numba.njit(lambda: SORTED_UNIQUE)

# ==========================================================================#
# Action flags for taking associated data from multiple IndexSets.
# Lowest bit indicates that contributing IndexSets must have data
# in order for any result data to be non-NaN.


# First bit set indicates that a data column, if all NaN, when merged with any
# other value will result in NaN.
# This bit is set for _BLAHB_DATA_{MIN, MAX, SUM, NONE}
DATA_NANS_PROPAGATE = np.uint8(1)

# Set the resulting data to None, even if the contributing IndexSets have data.
#_BLAHB_DATA_NONE = np.uint8(3)
#BLAHB_DATA_NONE = lambda: _BLAHB_DATA_NONE

# Sum all non-NaN contributing values. If all contributing data is NaN the
# result is NaN
DATA_NANSUM = np.uint8(4)
data_nansum = lambda: DATA_NANSUM

# Sum all contributing values. If any contributing data is NaN the
# result is NaN.
DATA_SUM = DATA_NANSUM | DATA_NANS_PROPAGATE
data_sum  = lambda: DATA_SUM

# Find the maximum non-NaN value of all contributing data. If all contributing
# data is NaN the result is NaN.
DATA_NANMAX = np.uint8(6)
data_nanmax = lambda: DATA_NANMAX

# Find the maximum value of all contributing data. If any of the contributing
# data is NaN the result is NaN.
DATA_MAX = DATA_NANMAX | DATA_NANS_PROPAGATE
data_max = lambda: DATA_MAX

# Find the minimum non-NaN value of all contributing data. If all contributing
# data is NaN the result is NaN.
_BLAHB_DATA_NANMIN = np.uint8(8)
data_nanmin = lambda: _BLAHB_DATA_NANMIN

# Find the minimum value of all contributing data. If any of the contributing
# data is NaN the result is NaN.
_BLAHB_DATA_MIN = _BLAHB_DATA_NANMIN | DATA_NANS_PROPAGATE
data_min = lambda: _BLAHB_DATA_MIN

# Find the first non-Nan contributing data. If all contributing data is NaN
# then the result is NaN.
DATA_NANFIRST = np.uint8(10)
data_nanfirst = lambda: DATA_NANFIRST

# Find the last non-Nan contributing data. If all contributing data is NaN
# then the result is NaN.
DATA_NANLAST = np.uint8(12)
data_nanlast = lambda: DATA_NANLAST


DATA_DEFAULT = DATA_NANFIRST
data_default = lambda: DATA_DEFAULT

BLAHB_DATA_DEFAULT_ARRAY = np.array([DATA_DEFAULT], dtype=np.uint8)


