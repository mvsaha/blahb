import numpy as np
from numpy.testing import  assert_array_equal as AAE
from ..indexset import IndexSet, make_data, make_indexset

N_TESTS = 1000
na = np.nan


def T(x):
    x = np.array(x)
    if x.ndim == 1:
        return x[:, None]
    else:
        return x.T


def _to_set(_loc):
    """Convert an array of locations into a set of tuple locations."""
    return set(map(tuple, _loc))


def to_set(indexset):
    """Convert an IndexSet into a set of tuple locations."""
    return _to_set(indexset._loc)


def to_dict(indexset):
    if indexset.data is None:
        return {tuple(indexset.loc[i]): None for i in range(indexset.n)}
    else:
        return {tuple(indexset.loc[i]): indexset.data[i]
                for i in range(indexset.n)}