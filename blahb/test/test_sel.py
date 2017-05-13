import numpy as np
from numpy.random import randint
from numpy.testing import assert_array_equal as AAE

from ..indexset import IndexSet
from ..flags import *


rand_bool = lambda: bool(randint(0, 2))

lo, hi = -10, 10
n_tests = 10000


def gen_int(lo, hi):
    return randint(lo, hi)


def gen_int_none(lo, hi):
    return randint(lo, hi), None


def gen_none_int(lo, hi):
    return None, randint(lo, hi)


def gen_int_int(lo, hi):
    return randint(lo, hi), randint(lo, hi)


def gen_arr(lo, hi):
    n = randint(0, 10)
    return np.unique(randint(lo, hi, size=n))


gens = {0: gen_int, 1: gen_int_none, 2: gen_none_int, 3: gen_int_int, 4: gen_arr}


def gen_random_selector(lo, hi):
    typ = np.random.randint(0, 5) # 0: int, 0
    return gens[typ](lo, hi)


def eval_int_selector(selector, coords):
    return coords == selector


def eval_int_int_selector(selector, coords):
    return (selector[0] <= coords) & (coords <= selector[1])


def eval_int_none_selector(selector, coords):
    return selector[0] <= coords


def eval_none_int_selector(selector, coords):
    return coords <= selector[1]


def eval_arr_selector(selector, coords):
    return np.in1d(coords, selector)


def naive_sel(selector, coords):
    if type(selector) is tuple:
        if selector[0] is None:
            return eval_none_int_selector(selector, coords)
        elif selector[1] is None:
            return eval_int_none_selector(selector, coords)
        else:
            return eval_int_int_selector(selector, coords)
    elif np.isscalar(selector):
        return eval_int_selector(selector, coords)
    else:
        return eval_arr_selector(selector, coords)


def naive_omit(selector, coords):
    return ~naive_sel(selector, coords)


def test_sel_rand():
    for ndim in range(1, 6):
        print(ndim, end=' ')
        for _ in range(n_tests):
            # print(n, end=' ')
            n = int(ndim ** (ndim / 2)) * randint(0, hi - lo)
            
            locs = randint(lo, hi, size=(n, ndim), dtype=np.int32)
            i = IndexSet(locs, CONSUME)
            
            n_selectors = randint(1, 4)
            selectors = [
                (randint(0, ndim), gen_random_selector(lo, hi), rand_bool())
                for _ in range(n_selectors)]
            j = i
            for dim, selector, should_omit in selectors:
                if should_omit:
                    j = j.omit(dim, selector)
                else:
                    j = j.sel(dim, selector)
            
            result = j.fin().loc
            
            bools = np.ones(i.n, dtype=bool)
            for dim, selector, should_omit in selectors:
                if should_omit:
                    bools &= naive_omit(selector, i.loc[:, dim])
                else:
                    bools &= naive_sel(selector, i.loc[:, dim])
            
            reference = i.loc[bools]
            AAE(result, reference)