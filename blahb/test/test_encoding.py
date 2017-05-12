import numpy as np
from ..encoding import encode, decode


def get_random_array_encoding(size, *, n_bits=None, signed=None):
    signed = bool(np.random.randint(0, 2)) if signed is None else signed
    n_bits = n_bits or np.random.randint(2, 31 if signed else 32)
    n_bits = min(32, n_bits)
    mn = -(2 ** n_bits) if signed else 0
    mx = (2 ** n_bits) - 1
    arr = np.random.randint(mn, mx, size)
    
    return (-1 if signed else 1) * n_bits, arr


def test_encoding_rand():
    for i in range(50000):
        ndim = np.random.randint(1, 10)
        size = np.random.randint(1, 1000)
        encoding, loc = zip(*(get_random_array_encoding(size) for _ in range(ndim)))
        encoding = np.asarray(encoding)
        loc = np.vstack(loc).T.copy()
        result = decode(encode(loc, encoding), encoding)
        np.testing.assert_array_equal(result, loc)