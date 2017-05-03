import numpy as np

from ..bits import *
from ..setops import intersection_
from .utils import *


def test_intersection_NANFIRST():
    a =  make_indexset([0,   1,  3,  4,  5,  6,  7])
    a_data = make_data([na, -3, -6, na, -9, na,  1])
    # Overlap                        |   |   |   |
    b =  make_indexset(            [ 4,  5,  6,  7,  9, 10, 11])
    b_data = make_data(            [na, na, -2, -5, na, -3,  7])
    
    merge = np.array([DATA_NANFIRST], dtype=np.uint8)
    
    # Both a and b have no data
    c = intersection_(a, b, merge)
    AAE(c.loc, T([4, 5, 6, 7]))
    assert c.data is None
    
    c = intersection_(b, a, merge)
    AAE(c.loc, T([4, 5, 6, 7]))
    assert c.data is None
    
    # Only a has data and is first
    a.data = a_data
    c = intersection_(a, b, merge)
    AAE(c.loc,  T([4,   5,  6, 7]))
    AAE(c.data, T([na, -9, na, 1]))
    
    # Only b has data and is last
    a.reset_data()
    b.data = b_data
    c = intersection_(a, b, merge)
    AAE(c.loc,  T([ 4,  5,  6,  7]))
    AAE(c.data, T([na, na, -2, -5]))
    
    # Both a and b have data, a is first
    a.data, b.data = a_data, b_data
    c = intersection_(a, b, merge)
    AAE(c.loc,  T([ 4,  5,  6, 7]))
    AAE(c.data, T([na, -9, -2, 1]))
    
    # Both a and b have data, b is first
    c = intersection_(b, a, merge)
    AAE(c.loc,  T([ 4,  5,  6,  7]))
    AAE(c.data, T([na, -9, -2, -5]))


def test_intersection_NANLAST():
    a =  make_indexset([0,   1,  3,  4,  5,  6,  7])
    a_data = make_data([na, -3, -6, na, -9, na,  1])
    # Overlap                        |   |   |   |
    b =  make_indexset(            [ 4,  5,  6,  7,  9, 10, 11])
    b_data = make_data(            [na, na, -2, -5, na, -3,  7])
    
    merge = np.array([DATA_NANLAST], dtype=np.uint8)
    
    # Both a and b have no data
    c = intersection_(a, b, merge)
    AAE(c.loc, T([4, 5, 6, 7]))
    assert c.data is None
    
    c = intersection_(b, a, merge)
    AAE(c.loc, T([4, 5, 6, 7]))
    assert c.data is None
    
    # Only a has data and is first
    a.data = a_data
    c = intersection_(a, b, merge)
    AAE(c.loc,  T([4,   5,  6, 7]))
    AAE(c.data, T([na, -9, na, 1]))
    
    # Only b has data and is last
    a.reset_data()
    b.data = b_data
    c = intersection_(a, b, merge)
    AAE(c.loc,  T([ 4,  5,  6,  7]))
    AAE(c.data, T([na, na, -2, -5]))
    
    # Both a and b have data, a is first
    a.data, b.data = a_data, b_data
    c = intersection_(a, b, merge)
    AAE(c.loc,  T([ 4,  5,  6,  7]))
    AAE(c.data, T([na, -9, -2, -5]))
    
    # Both a and b have data, b is first
    c = intersection_(b, a, merge)
    AAE(c.loc,  T([ 4,  5,  6, 7]))
    AAE(c.data, T([na, -9, -2, 1]))


def test_intersection_MIN():
    a =  make_indexset([0,   1,  3,  4,  5,  6,  7,  8])
    a_data = make_data([na, -3, -6, na, -9, na,  1,  3])
    # Overlap                        |   |   |   |   |
    b =  make_indexset(            [ 4,  5,  6,  7,  8, 9, 10, 11])
    b_data = make_data(            [na, na, -2, -5, 10, na, -3,  7])
    
    merge = np.array([_BLAHB_DATA_MIN], dtype=np.uint8)
    
    # Both a and b have no data
    c = intersection_(a, b, merge)
    AAE(c.loc, T([4, 5, 6, 7, 8]))
    assert c.data is None
    
    c = intersection_(b, a, merge)
    AAE(c.loc, T([4, 5, 6, 7, 8]))
    assert c.data is None
    
    # Only a has data and is first
    a.data = a_data
    c = intersection_(a, b, merge)
    AAE(c.loc,  T([4,   5,  6,  7, 8]))
    assert c.data is None
    
    # Only b has data and is last
    a.reset_data()
    b.data = b_data
    c = intersection_(a, b, merge)
    AAE(c.loc,  T([ 4,  5,  6,  7, 8]))
    assert c.data is None
    
    # Both a and b have data, a is first
    a.data, b.data = a_data, b_data
    c = intersection_(a, b, merge)
    AAE(c.loc,  T([ 4,  5,  6,  7, 8]))
    AAE(c.data, T([na, na, na, -5, 3]))
    
    # Both a and b have data, b is first
    c = intersection_(b, a, merge)
    AAE(c.loc,  T([ 4,  5,  6,  7, 8]))
    AAE(c.data, T([na, na, na, -5, 3]))


def test_intersection_NANMIN():
    a =  make_indexset([0,   1,  3,  4,  5,  6,  7,  8])
    a_data = make_data([na, -3, -6, na, -9, na,  1,  3])
    # Overlap                        |   |   |   |   |
    b =  make_indexset(            [ 4,  5,  6,  7,  8, 9,  10, 11])
    b_data = make_data(            [na, na, -2, -5, 10, na, -3,  7])
    
    merge = np.array([_BLAHB_DATA_NANMIN], dtype=np.uint8)
    
    # Both a and b have no data
    c = intersection_(a, b, merge)
    AAE(c.loc, T([4, 5, 6, 7, 8]))
    assert c.data is None
    
    c = intersection_(b, a, merge)
    AAE(c.loc, T([4, 5, 6, 7, 8]))
    assert c.data is None
    
    # Only a has data and is first
    a.data = a_data
    c = intersection_(a, b, merge)
    AAE(c.loc,  T([4,   5,  6, 7, 8]))
    AAE(c.data, T([na, -9, na, 1, 3]))
    
    # Only b has data and is last
    a.reset_data()
    b.data = b_data
    c = intersection_(a, b, merge)
    AAE(c.loc,  T([ 4,  5,  6,  7,  8]))
    AAE(c.data, T([na, na, -2, -5, 10]))
    
    # Both a and b have data, a is first
    a.data, b.data = a_data, b_data
    c = intersection_(a, b, merge)
    AAE(c.loc,  T([ 4,  5,  6,  7, 8]))
    AAE(c.data, T([na, -9, -2, -5, 3]))
    
    # Both a and b have data, b is first
    c = intersection_(b, a, merge)
    AAE(c.loc,  T([ 4,  5,  6,  7, 8]))
    AAE(c.data, T([na, -9, -2, -5, 3]))


def test_intersection_MAX():
    a =  make_indexset([0,   1,  3,  4,  5,  6,  7,  8])
    a_data = make_data([na, -3, -6, na, -9, na,  1,  3])
    # Overlap                        |   |   |   |   |
    b =  make_indexset(            [ 4,  5,  6,  7,  8, 9, 10, 11])
    b_data = make_data(            [na, na, -2, -5, 10, na, -3,  7])
    
    merge = np.array([DATA_MAX], dtype=np.uint8)
    
    # Both a and b have no data
    c = intersection_(a, b, merge)
    AAE(c.loc, T([4, 5, 6, 7, 8]))
    assert c.data is None
    
    c = intersection_(b, a, merge)
    AAE(c.loc, T([4, 5, 6, 7, 8]))
    assert c.data is None
    
    # Only a has data and is first
    a.data = a_data
    c = intersection_(a, b, merge)
    AAE(c.loc,  T([4,   5,  6, 7, 8]))
    assert c.data is None
    
    # Only b has data and is last
    a.reset_data()
    b.data = b_data
    c = intersection_(a, b, merge)
    AAE(c.loc,  T([ 4,  5,  6,  7,  8]))
    assert c.data is None
    
    # Both a and b have data, a is first
    a.data, b.data = a_data, b_data
    c = intersection_(a, b, merge)
    AAE(c.loc,  T([ 4,  5,  6, 7,  8]))
    AAE(c.data, T([na, na, na, 1, 10]))
    
    # Both a and b have data, b is first
    c = intersection_(b, a, merge)
    AAE(c.loc,  T([ 4,  5,  6,  7, 8]))
    AAE(c.data, T([na, na, na, 1, 10]))


def test_intersection_NANMAX():
    a =  make_indexset([0,   1,  3,  4,  5,  6,  7,  8])
    a_data = make_data([na, -3, -6, na, -9, na,  1,  3])
    # Overlap                        |   |   |   |   |
    b =  make_indexset(            [ 4,  5,  6,  7,  8, 9,  10, 11])
    b_data = make_data(            [na, na, -2, -5, 10, na, -3,  7])
    
    merge = np.array([DATA_NANMAX], dtype=np.uint8)
    
    # Both a and b have no data
    c = intersection_(a, b, merge)
    AAE(c.loc, T([4, 5, 6, 7, 8]))
    assert c.data is None
    
    c = intersection_(b, a, merge)
    AAE(c.loc, T([4, 5, 6, 7, 8]))
    assert c.data is None
    
    # Only a has data and is first
    a.data = a_data
    c = intersection_(a, b, merge)
    AAE(c.loc,  T([4,   5,  6, 7, 8]))
    AAE(c.data, T([na, -9, na, 1, 3]))
    
    # Only b has data and is last
    a.reset_data()
    b.data = b_data
    c = intersection_(a, b, merge)
    AAE(c.loc,  T([ 4,  5,  6,  7,  8]))
    AAE(c.data, T([na, na, -2, -5, 10]))
    
    # Both a and b have data, a is first
    a.data, b.data = a_data, b_data
    c = intersection_(a, b, merge)
    AAE(c.loc,  T([ 4,  5,  6, 7,  8]))
    AAE(c.data, T([na, -9, -2, 1, 10]))
    
    # Both a and b have data, b is first
    c = intersection_(b, a, merge)
    AAE(c.loc,  T([ 4,  5,  6, 7,  8]))
    AAE(c.data, T([na, -9, -2, 1, 10]))


def test_intersection_SUM():
    a =  make_indexset([0,   1,  3,  4,  5,  6,  7,  8])
    a_data = make_data([na, -3, -6, na, -9, na,  1,  3])
    # Overlap                        |   |   |   |   |
    b =  make_indexset(            [ 4,  5,  6,  7,  8, 9, 10, 11])
    b_data = make_data(            [na, na, -2, -5, 10, na, -3,  7])
    
    merge = np.array([DATA_SUM], dtype=np.uint8)
    
    # Both a and b have no data
    c = intersection_(a, b, merge)
    AAE(c.loc, T([4, 5, 6, 7, 8]))
    assert c.data is None
    
    c = intersection_(b, a, merge)
    AAE(c.loc, T([4, 5, 6, 7, 8]))
    assert c.data is None
    
    # Only a has data and is first
    a.data = a_data
    c = intersection_(a, b, merge)
    AAE(c.loc,  T([4, 5, 6, 7, 8]))
    assert c.data is None
    
    # Only b has data and is last
    a.reset_data()
    b.data = b_data
    c = intersection_(a, b, merge)
    AAE(c.loc,  T([ 4,  5,  6,  7,  8]))
    assert c.data is None
    
    # Both a and b have data, a is first
    a.data, b.data = a_data, b_data
    c = intersection_(a, b, merge)
    AAE(c.loc,  T([ 4,  5,  6,  7,  8]))
    AAE(c.data, T([na, na, na, -4, 13]))
    
    # Both a and b have data, b is first
    c = intersection_(b, a, merge)
    AAE(c.loc,  T([ 4,  5,  6,  7,  8]))
    AAE(c.data, T([na, na, na, -4, 13]))


def test_intersection_NANSUM():
    a =  make_indexset([0,   1,  3,  4,  5,  6,  7,  8])
    a_data = make_data([na, -3, -6, na, -9, na,  1,  3])
    # Overlap                        |   |   |   |   |
    b =  make_indexset(            [ 4,  5,  6,  7,  8, 9,  10, 11])
    b_data = make_data(            [na, na, -2, -5, 10, na, -3,  7])
    
    merge = np.array([DATA_NANSUM], dtype=np.uint8)
    
    # Both a and b have no data
    c = intersection_(a, b, merge)
    AAE(c.loc, T([4, 5, 6, 7, 8]))
    assert c.data is None
    
    c = intersection_(b, a, merge)
    AAE(c.loc, T([4, 5, 6, 7, 8]))
    assert c.data is None
    
    # Only a has data and is first
    a.data = a_data
    c = intersection_(a, b, merge)
    AAE(c.loc,  T([4,   5,  6, 7, 8]))
    AAE(c.data, T([na, -9, na, 1, 3]))
    
    # Only b has data and is last
    a.reset_data()
    b.data = b_data
    c = intersection_(a, b, merge)
    AAE(c.loc,  T([ 4,  5,  6,  7,  8]))
    AAE(c.data, T([na, na, -2, -5, 10]))
    
    # Both a and b have data, a is first
    a.data, b.data = a_data, b_data
    c = intersection_(a, b, merge)
    AAE(c.loc,  T([ 4,  5,  6,  7,  8]))
    AAE(c.data, T([na, -9, -2, -4, 13]))
    
    # Both a and b have data, b is first
    c = intersection_(b, a, merge)
    AAE(c.loc,  T([ 4,  5,  6,  7,  8]))
    AAE(c.data, T([na, -9, -2, -4, 13]))


def test_intersection_randomized():
    for ndim in range(1, 6):
        for _ in range(N_TESTS):
            nx, ny = np.random.randint(1, 2 ** ndim, size=2)
            x = np.random.randint(0, 3, (nx, ndim), dtype=np.int32)
            y = np.random.randint(0, 3, (ny, ndim), dtype=np.int32)
            
            a = IndexSet(x, NO_FLAGS)
            b = IndexSet(y, NO_FLAGS)
            
            c = intersection_(a, b)
            result = (to_set(a) & to_set(b))
            assert to_set(c) == result
            assert len(result) == c.n

