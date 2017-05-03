from ..bits import *
from ..setops import asymmetric_difference_
from .utils import *


def test_asymmetric_difference_with_data():
    a =  make_indexset([0,   1,  3,  4,  5,  6,  7])
    a_data = make_data([na, -3, -6, na, -9, na,  1])
    # Overlap                        |   |   |   |
    b =  make_indexset(            [ 4,  5,  6,  7,  9, 10, 11])
    b_data = make_data(            [na, na, -2, -5, na, -3,  7])
    
    # Both a and b have no data
    c = asymmetric_difference_(a, b)
    AAE(c.loc, T([0, 1, 3]))
    assert c.data is None
    
    c = asymmetric_difference_(b, a)
    AAE(c.loc, T([9, 10, 11]))
    assert c.data is None
    
    # Only a has data and is first
    a.data = a_data
    c = asymmetric_difference_(a, b)
    AAE(c.loc,  T([ 0,  1,  3]))
    AAE(c.data, T([na, -3, -6]))
    
    # Only b has data and is last
    a.reset_data()
    b.data = b_data
    c = asymmetric_difference_(a, b)
    AAE(c.loc,  T([0, 1, 3]))
    assert c.data is None
    
    # Both a and b have data, a is first
    a.data, b.data = a_data, b_data
    c = asymmetric_difference_(a, b)
    AAE(c.loc,  T([ 0,  1,  3]))
    AAE(c.data, T([na, -3, -6]))
    
    # Both a and b have data, b is first
    c = asymmetric_difference_(b, a)
    AAE(c.loc,  T([ 9, 10, 11]))
    AAE(c.data, T([na, -3,  7]))


def test_asymmetric_difference_randomized():
    for ndim in range(1, 6):
        for _ in range(N_TESTS):
            nx, ny = np.random.randint(1, 2 ** ndim, size=2)
            x = np.random.randint(0, 3, (nx, ndim), dtype=np.int32)
            y = np.random.randint(0, 3, (ny, ndim), dtype=np.int32)
            
            a = IndexSet(x, NO_FLAGS)
            b = IndexSet(y, NO_FLAGS)
            
            c = asymmetric_difference_(a, b)
            
            reference = to_set(a) - to_set(b)
            result = to_set(c)
            assert reference == result
            assert len(result) == c.n  # Ensures that k is unique_