import numpy as np
from numpy.testing import assert_array_equal as AAE

from ..timsort import timsort_
from .utils import N_TESTS


def test_timsort_rand():
    for i in range(N_TESTS * 5):
        n = np.random.randint(1, 1000)
        labels = np.random.randint(-10, 10, size=n)
        result = timsort_(labels.copy())
        ref = np.argsort(labels, kind="mergesort")
        AAE(result, ref)