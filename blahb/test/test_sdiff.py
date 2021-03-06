import numpy as np

from ..flags import *
from ..setops import symmetric_difference_
from .utils import *


def test_symmetric_difference_with_data():
    a =  make_indexset([0,   1,  3,  4,  5,  6,  7])
    a_data = make_data([na, -3, -6, na, -9, na,  1])
    # Overlap                        |   |   |   |
    b =  make_indexset(            [ 4,  5,  6,  7,  9, 10, 11])
    b_data = make_data(            [na, na, -2, -5, na, -3,  7])
    
    # Both a and b have no data
    c = symmetric_difference_(a, b)
    AAE(c.loc, T([0, 1, 3, 9, 10, 11]))
    assert c.data is None
    
    c = symmetric_difference_(b, a)
    AAE(c.loc, T([0, 1, 3, 9, 10, 11]))
    assert c.data is None
    
    # Only a has data and is first
    a.data = a_data
    c = symmetric_difference_(a, b)
    AAE(c.loc,  T([ 0,  1,  3,  9, 10, 11]))
    AAE(c.data, T([na, -3, -6, na, na, na]))
    
    # Only b has data and is last
    a.reset_data()
    b.data = b_data
    c = symmetric_difference_(a, b)
    AAE(c.loc,  T([ 0,  1,  3,  9, 10, 11]))
    AAE(c.data, T([na, na, na, na, -3,  7]))
    
    # Both a and b have data, a is first
    a.data, b.data = a_data, b_data
    c = symmetric_difference_(a, b)
    AAE(c.loc,  T([ 0,  1,  3,  9, 10, 11]))
    AAE(c.data, T([na, -3, -6, na, -3,  7]))
    
    # Both a and b have data, b is first
    c = symmetric_difference_(b, a)
    AAE(c.loc,  T([ 0,  1,  3,  9, 10, 11]))
    AAE(c.data, T([na, -3, -6, na, -3,  7]))


def test_symmetric_difference_randomized():
    for ndim in range(1, 6):
        for _ in range(N_TESTS * 5):
            nx, ny = np.random.randint(1, 2 ** ndim, size=2)
            lo, hi = -2, 3
            n_bits = int(np.ceil(np.log2(max(abs(lo), abs(hi)))))
            x = np.random.randint(lo, hi, (nx, ndim), dtype=np.int32)
            y = np.random.randint(lo, hi, (ny, ndim), dtype=np.int32)
            a = IndexSet(x, NO_FLAGS)
            b = IndexSet(y, NO_FLAGS)

            use_encoding = np.random.randint(0, 2)
            if use_encoding:
                enc = np.array([n_bits] * ndim) * (-1 if lo < 0 else 1)
                a.encode(enc)
                b.encode(enc)

            c = symmetric_difference_(a, b)
            reference = to_set(a).symmetric_difference(to_set(b))
            result = to_set(c)
            assert reference == result
            assert len(result) == c.n  # Make sure it's unique_
