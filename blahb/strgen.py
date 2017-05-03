"""
Functions for parameterizing strings useful for code generation.

So far only used by `label`
"""

import numpy as np


def coord_name(dim):
    return "coords_{}".format(dim)


def coord_dim_names(ndim):
    return ', '.join([coord_name(i) for i in range(ndim)])


def indent_block(string, n=1, indentation="    ", first_line=None):
    if first_line is None:
        first_line = n
    
    split = string.splitlines()
    if len(split) is 0:
        return []
    
    first = (indentation * first_line) + split[0] + '\n'
    others = ''.join((indentation * n) + s + '\n' for s in split[1:])
    return first + others


def prepend_break(string):
    return '\n' + string


def shape(dim):
    return 'shape_{}'.format(dim)


def assign(lhs, rhs):
    return '{} = {}'.format(lhs, rhs)


def shape_init_strings(shp):
    return '\n'.join([assign(shape(i), size) for i, size in enumerate(shp)])


def shift(dim):
    return 'shift_{}'.format(dim)


def shift_init_strings(neigh_shape):
    """Get the initialization strings for the valid shift values (ndim - 1)"""
    return '\n'.join(
        [assign(shift(i), size) for i, size in enumerate(neigh_shape[:-1])])


def cursors(dim):
    return 'cursors_{}'.format(dim)


def zeros(shp, dtype):
    if np.isscalar(shp):
        return 'np.zeros({}, dtype={})'.format(shp[0], dtype_string(dtype))
    else:
        return 'np.zeros({}, dtype={})'.format(tuple(shp), dtype_string(dtype))


def dtype_string(typ):
    return 'np.{}'.format(np.dtype(typ))


def cursors_init_string(shp, dtype):
    return assign(cursors(len(shp) - 1), zeros(shp, dtype))


def cursors_init_strings(shp, dtype):
    return '\n'.join(
        [cursors_init_string(shp[:i], dtype) for i in range(1, len(shp))])


def ends(dim):
    return 'ends_{}'.format(dim)


def ends_init_string(shp, dtype):
    return assign(ends(len(shp) - 1), zeros(shp, dtype))


def ends_init_strings(shp, dtype):
    return '\n'.join(
        [ends_init_string(shp[:i], dtype) for i in range(1, len(shp))])


def _c(dim):
    return 'c{}'.format(dim)


def index(var, idx):
    return '{}[{}]'.format(var, idx)


def single_coord_init_string(dim):
    return assign(_c(dim), index(coord_name(dim), 0))


def coord_init_strings(ndim):
    return '\n'.join(
        assign(_c(dim), index(coord_name(dim), 0)) for dim in range(ndim))


def r(dim):
    return 'r{dim}'.format(dim)


def range_init_string(dim, size):
    return assign('r{}'.format(dim), size // 2)


def range_init_strings(shp):
    return '\n'.join(
        [range_init_string(dim, size) for dim, size in enumerate(shp)])


def for_loop(element, sequence, body):
    """Build a string nesting a statement in a for loop."""
    return 'for {} in {}:\n{}'.format(element, sequence, indent_block(body, 1))


def i_(dim):
    return 'i{}'.format(dim)


def loop_over_shape(shp, body):
    ret = body
    for i, s in zip(reversed(range(len(shp))), reversed(shp)):
        ret = for_loop(i_(i), 'range({})'.format(s), ret)
    return ret