import numba

from ..utils import enlarge_mat


@numba.njit
def append_remaining(loc_in, index_in, loc_out, index_out):
    """Write the coordinates loc_in[index_in:] to the end of
    loc_out[index_out:], expanding loc_out if necessary."""
    n_left = loc_in.shape[0] - index_in
    size_needed = index_out + n_left
    if loc_out.shape[0] < size_needed:
        loc_out = enlarge_mat(loc_out, size_needed)
    loc_out[index_out:size_needed] = loc_in[index_in:]
    return loc_out[:size_needed]