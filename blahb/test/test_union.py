import numpy as np
from numpy.testing import  assert_array_equal as AAE

from ..indexset import IndexSet, make_indexset, make_data
from ..flags import *
from ..setops import union_


N_TESTS = 500
na = np.nan

from .utils import *


def test_union_NANFIRST():
    a =  make_indexset([0,   1,  3,  4,  5,  6,  7])
    a_data = make_data([na, -3, -6, na, -9, na,  1])
    # Overlap                        |   |   |   |
    b =  make_indexset(            [ 4,  5,  6,  7,  9, 10, 11])
    b_data = make_data(            [na, na, -2, -5, na, -3,  7])
    
    merge = np.array([DATA_NANFIRST], dtype=np.uint8)
    
    # Both a and b have no data
    c = union_(a, b, merge)
    AAE(c.loc, T([0,   1,  3,  4,  5,  6, 7,  9, 10, 11]))
    assert c.data is None
    
    c = union_(b, a, merge)
    AAE(c.loc, T([0, 1, 3, 4, 5, 6, 7, 9, 10, 11]))
    assert c.data is None
    
    # Only a has data and is first
    a.data = a_data
    c = union_(a, b, merge)
    AAE(c.loc,  T([ 0,  1,  3,  4,  5,  6, 7,  9, 10, 11]))
    AAE(c.data, T([na, -3, -6, na, -9, na, 1, na, na, na]))
    
    # Only b has data and is last
    a.reset_data()
    b.data = b_data
    c = union_(a, b, merge)
    AAE(c.loc,  T([ 0,  1,  3,  4,  5,  6,  7,  9, 10, 11]))
    AAE(c.data, T([na, na, na, na, na, -2, -5, na, -3,  7]))
    
    # Both a and b have data, a is first
    a.data, b.data = a_data, b_data
    c = union_(a, b, merge)
    AAE(c.loc, T([0,   1,  3,  4,  5,  6,  7, 9, 10, 11]))
    AAE(c.data ,T([na, -3, -6, na, -9, -2, 1, na, -3,  7]))
    
    # Both a and b have data, b is first
    c = union_(b, a, merge)
    AAE(c.loc, T([ 0,  1,  3,  4,  5,  6,  7,  9, 10, 11]))
    AAE(c.data ,T([na, -3, -6, na, -9, -2, -5, na, -3,  7]))


def test_union_NANLAST():
    a =  make_indexset([0,   1,  3,  4,  5,  6,  7])
    a_data = make_data([na, -3, -6, na, -9, na,  1])
    # Overlap                        |   |   |   |
    b =  make_indexset(            [ 4,  5,  6,  7,  9, 10, 11])
    b_data = make_data(            [na, na, -2, -5, na, -3,  7])
    
    merge = np.array([DATA_NANLAST], dtype=np.uint8)
    
    # Both a and b have no data
    c = union_(a, b, merge)
    AAE(c.loc, T([0, 1,  3,  4,  5,  6, 7,  9, 10, 11]))
    assert c.data is None
    
    c = union_(b, a, merge)
    AAE(c.loc, T([0, 1, 3, 4, 5, 6, 7, 9, 10, 11]))
    assert c.data is None
    
    # Only a has data and is first
    a.data = a_data
    c = union_(a, b, merge)
    AAE(c.loc,  T([ 0,  1,  3,  4,  5,  6, 7,  9, 10, 11]))
    AAE(c.data, T([na, -3, -6, na, -9, na, 1, na, na, na]))
    
    # Only b has data and is last
    a.reset_data()
    b.data = b_data
    c = union_(a, b, merge)
    AAE(c.loc,  T([ 0,  1,  3,  4,  5,  6,  7,  9, 10, 11]))
    AAE(c.data, T([na, na, na, na, na, -2, -5, na, -3,  7]))
    
    # Both a and b have data, a is first
    a.data, b.data = a_data, b_data
    c = union_(a, b, merge)
    AAE(c.loc, T([0,   1,  3,  4,  5,  6,  7,  9, 10, 11]))
    AAE(c.data ,T([na, -3, -6, na, -9, -2, -5, na, -3,  7]))
    
    # Both a and b have data, b is first
    c = union_(b, a, merge)
    AAE(c.loc, T([ 0,  1,  3,  4,  5,  6, 7,  9, 10, 11]))
    AAE(c.data ,T([na, -3, -6, na, -9, -2, 1, na, -3,  7]))


def test_union_MIN():
    a =  make_indexset([0,   1,  3,  4,  5,  6,  7,  8])
    a_data = make_data([na, -3, -6, na, -9, na,  1,  3])
    # Overlap                        |   |   |   |   |
    b =  make_indexset(            [ 4,  5,  6,  7,  8, 9, 10, 11])
    b_data = make_data(            [na, na, -2, -5, 10, na, -3,  7])
    
    merge = np.array([_BLAHB_DATA_MIN], dtype=np.uint8)
    
    # Both a and b have no data
    c = union_(a, b, merge)
    AAE(c.loc, T([0, 1,  3, 4, 5, 6, 7, 8, 9, 10, 11]))
    assert c.data is None
    
    c = union_(b, a, merge)
    AAE(c.loc, T([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
    assert c.data is None
    
    # Only a has data and is first
    a.data = a_data
    c = union_(a, b, merge)
    AAE(c.loc,  T([ 0,  1,  3,  4,  5,  6, 7, 8,  9, 10, 11]))
    assert c.data is None
    
    # Only b has data and is last
    a.reset_data()
    b.data = b_data
    c = union_(a, b, merge)
    AAE(c.loc,  T([ 0,  1,  3,  4,  5,  6,  7,  8,  9, 10, 11]))
    assert c.data is None
    
    # Both a and b have data, a is first
    a.data, b.data = a_data, b_data
    c = union_(a, b, merge)
    AAE(c.loc, T([ 0,  1,  3,  4,  5,  6,  7, 8,  9, 10, 11]))
    AAE(c.data ,T([na, na, na, na, na, na, -5, 3, na, na, na]))
    
    # Both a and b have data, b is first (same as last; order does not matter)
    c = union_(b, a, merge)
    AAE(c.loc, T([ 0,  1,  3,  4,  5,  6,  7, 8,  9, 10, 11]))
    AAE(c.data ,T([na, na, na, na, na, na, -5, 3, na, na, na]))


def test_union_NANMIN():
    a =  make_indexset([0,   1,  3,  4,  5,  6,  7,  8])
    a_data = make_data([na, -3, -6, na, -9, na,  1,  3])
    # Overlap                        |   |   |   |   |
    b =  make_indexset(            [ 4,  5,  6,  7,  8, 9,  10, 11])
    b_data = make_data(            [na, na, -2, -5, 10, na, -3,  7])
    
    merge = np.array([_BLAHB_DATA_NANMIN], dtype=np.uint8)
    
    # Both a and b have no data
    c = union_(a, b, merge)
    AAE(c.loc, T([0, 1,  3, 4, 5, 6, 7, 8, 9, 10, 11]))
    assert c.data is None
    
    c = union_(b, a, merge)
    AAE(c.loc, T([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
    assert c.data is None
    
    # Only a has data and is first
    a.data = a_data
    c = union_(a, b, merge)
    AAE(c.loc,  T([ 0,   1,  3,  4,  5,  6, 7, 8,  9, 10, 11]))
    AAE(c.data, T([ na, -3, -6, na, -9, na, 1, 3, na, na, na]))
    
    # Only b has data and is last
    a.reset_data()
    b.data = b_data
    c = union_(a, b, merge)
    AAE(c.loc,  T([ 0,  1,  3,  4,  5,  6,  7,  8,  9, 10, 11]))
    AAE(c.data, T([na, na, na, na, na, -2, -5, 10, na, -3,  7]))
    
    # Both a and b have data, a is first
    a.data, b.data = a_data, b_data
    c = union_(a, b, merge)
    AAE(c.loc,  T([ 0,  1,  3,  4,  5,  6,  7, 8,  9, 10, 11]))
    AAE(c.data, T([na, -3, -6, na, -9, -2, -5, 3, na, -3,  7]))
    
    # Both a and b have data, b is first (same as last; order does not matter)
    c = union_(b, a, merge)
    AAE(c.loc,  T([ 0,  1,  3,  4,  5,  6,  7, 8,  9, 10, 11]))
    AAE(c.data, T([na, -3, -6, na, -9, -2, -5, 3, na, -3,  7]))


def test_union_MAX():
    a =  make_indexset([0,   1,  3,  4,  5,  6,  7,  8])
    a_data = make_data([na, -3, -6, na, -9, na,  1,  3])
    # Overlap                        |   |   |   |   |
    b =  make_indexset(            [ 4,  5,  6,  7,  8, 9, 10, 11])
    b_data = make_data(            [na, na, -2, -5, 10, na, -3,  7])
    
    merge = np.array([DATA_MAX], dtype=np.uint8)
    
    # Both a and b have no data
    c = union_(a, b, merge)
    AAE(c.loc, T([0, 1,  3, 4, 5, 6, 7, 8, 9, 10, 11]))
    assert c.data is None
    
    c = union_(b, a, merge)
    AAE(c.loc, T([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
    assert c.data is None
    
    # Only a has data and is first
    a.data = a_data
    c = union_(a, b, merge)
    AAE(c.loc,  T([ 0,  1,  3,  4,  5,  6, 7, 8,  9, 10, 11]))
    assert c.data is None
    
    # Only b has data and is last
    a.reset_data()
    b.data = b_data
    c = union_(a, b, merge)
    AAE(c.loc,  T([ 0,  1,  3,  4,  5,  6,  7,  8,  9, 10, 11]))
    assert c.data is None
    
    # Both a and b have data, a is first
    a.data, b.data = a_data, b_data
    c = union_(a, b, merge)
    AAE(c.loc,  T([ 0,  1,  3,  4,  5,  6, 7,  8,  9, 10, 11]))
    AAE(c.data, T([na, na, na, na, na, na, 1, 10, na, na, na]))
    
    # Both a and b have data, b is first (same as last; order does not matter)
    c = union_(b, a, merge)
    AAE(c.loc,  T([ 0,  1,  3,  4,  5,  6, 7,  8,  9, 10, 11]))
    AAE(c.data, T([na, na, na, na, na, na, 1, 10, na, na, na]))


def test_union_NANMAX():
    a =  make_indexset([0,   1,  3,  4,  5,  6,  7,  8])
    a_data = make_data([na, -3, -6, na, -9, na,  1,  3])
    # Overlap                        |   |   |   |   |
    b =  make_indexset(            [ 4,  5,  6,  7,  8, 9,  10, 11])
    b_data = make_data(            [na, na, -2, -5, 10, na, -3,  7])
    
    merge = np.array([DATA_NANMAX], dtype=np.uint8)
    
    # Both a and b have no data
    c = union_(a, b, merge)
    AAE(c.loc, T([0, 1,  3, 4, 5, 6, 7, 8, 9, 10, 11]))
    assert c.data is None
    
    c = union_(b, a, merge)
    AAE(c.loc, T([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
    assert c.data is None
    
    # Only a has data and is first
    a.data = a_data
    c = union_(a, b, merge)
    AAE(c.loc,  T([ 0,   1,  3,  4,  5,  6, 7, 8,  9, 10, 11]))
    AAE(c.data, T([ na, -3, -6, na, -9, na, 1, 3, na, na, na]))
    
    # Only b has data and is last
    a.reset_data()
    b.data = b_data
    c = union_(a, b, merge)
    AAE(c.loc,  T([ 0,  1,  3,  4,  5,  6,  7,  8,  9, 10, 11]))
    AAE(c.data, T([na, na, na, na, na, -2, -5, 10, na, -3,  7]))
    
    # Both a and b have data, a is first
    a.data, b.data = a_data, b_data
    c = union_(a, b, merge)
    AAE(c.loc,  T([ 0,  1,  3,  4,  5,  6, 7,  8,  9, 10, 11]))
    AAE(c.data, T([na, -3, -6, na, -9, -2, 1, 10, na, -3,  7]))
    
    # Both a and b have data, b is first (same as last; order does not matter)
    c = union_(b, a, merge)
    AAE(c.loc,  T([ 0,  1,  3,  4,  5,  6, 7,  8,  9, 10, 11]))
    AAE(c.data, T([na, -3, -6, na, -9, -2, 1, 10, na, -3,  7]))


def test_union_SUM():
    a =  make_indexset([0,   1,  3,  4,  5,  6,  7,  8])
    a_data = make_data([na, -3, -6, na, -9, na,  1,  3])
    # Overlap                        |   |   |   |   |
    b =  make_indexset(            [ 4,  5,  6,  7,  8, 9, 10, 11])
    b_data = make_data(            [na, na, -2, -5, 10, na, -3,  7])
    
    merge = np.array([DATA_SUM], dtype=np.uint8)
    
    # Both a and b have no data
    c = union_(a, b, merge)
    AAE(c.loc, T([0, 1,  3, 4, 5, 6, 7, 8, 9, 10, 11]))
    assert c.data is None
    
    c = union_(b, a, merge)
    AAE(c.loc, T([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
    assert c.data is None
    
    # Only a has data and is first
    a.data = a_data
    c = union_(a, b, merge)
    AAE(c.loc,  T([ 0,  1,  3,  4,  5,  6, 7, 8,  9, 10, 11]))
    assert c.data is None
    
    # Only b has data and is last
    a.reset_data()
    b.data = b_data
    c = union_(a, b, merge)
    AAE(c.loc,  T([ 0,  1,  3,  4,  5,  6,  7,  8,  9, 10, 11]))
    assert c.data is None
    
    # Both a and b have data, a is first
    a.data, b.data = a_data, b_data
    c = union_(a, b, merge)
    AAE(c.loc,  T([ 0,  1,  3,  4,  5,  6,  7,  8,  9, 10, 11]))
    AAE(c.data, T([na, na, na, na, na, na, -4, 13, na, na, na]))
    
    # Both a and b have data, b is first (same as last; order does not matter)
    c = union_(b, a, merge)
    AAE(c.loc,  T([ 0,  1,  3,  4,  5,  6,  7,  8,  9, 10, 11]))
    AAE(c.data, T([na, na, na, na, na, na, -4, 13, na, na, na]))


def test_union_NANSUM():
    a =  make_indexset([0,   1,  3,  4,  5,  6,  7,  8])
    a_data = make_data([na, -3, -6, na, -9, na,  1,  3])
    # Overlap                        |   |   |   |   |
    b =  make_indexset(            [ 4,  5,  6,  7,  8, 9,  10, 11])
    b_data = make_data(            [na, na, -2, -5, 10, na, -3,  7])
    
    merge = np.array([DATA_NANSUM], dtype=np.uint8)
    
    # Both a and b have no data
    c = union_(a, b, merge)
    AAE(c.loc, T([0, 1,  3, 4, 5, 6, 7, 8, 9, 10, 11]))
    assert c.data is None
    
    c = union_(b, a, merge)
    AAE(c.loc, T([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
    assert c.data is None
    
    # Only a has data and is first
    a.data = a_data
    c = union_(a, b, merge)
    AAE(c.loc,  T([ 0,   1,  3,  4,  5,  6, 7, 8,  9, 10, 11]))
    AAE(c.data, T([ na, -3, -6, na, -9, na, 1, 3, na, na, na]))
    
    # Only b has data and is last
    a.reset_data()
    b.data = b_data
    c = union_(a, b, merge)
    AAE(c.loc,  T([ 0,  1,  3,  4,  5,  6,  7,  8,  9, 10, 11]))
    AAE(c.data, T([na, na, na, na, na, -2, -5, 10, na, -3,  7]))
    
    # Both a and b have data, a is first
    a.data, b.data = a_data, b_data
    c = union_(a, b, merge)
    AAE(c.loc,  T([ 0,  1,  3,  4,  5,  6,  7,  8,  9, 10, 11]))
    AAE(c.data, T([na, -3, -6, na, -9, -2, -4, 13, na, -3,  7]))
    
    # Both a and b have data, b is first (same as last; order does not matter)
    c = union_(b, a, merge)
    AAE(c.loc,  T([ 0,  1,  3,  4,  5,  6,  7,  8,  9, 10, 11]))
    AAE(c.data, T([na, -3, -6, na, -9, -2, -4, 13, na, -3,  7]))


def test_union_randomized():
    for ndim in range(1, 6):
        for _ in range(N_TESTS*5):
            nx, ny = np.random.randint(1, 2**ndim, size=2)
            x = np.random.randint(0, 3, (nx, ndim), dtype=np.int32)
            y = np.random.randint(0, 3, (ny, ndim), dtype=np.int32)
            a = IndexSet(x, NO_FLAGS)
            b = IndexSet(y, NO_FLAGS)
            ref = to_set(a) | to_set(b)
            c = union_(a, b, None)
            result = to_set(c)
            assert ref == result
            assert len(result) == c.loc.shape[0]
