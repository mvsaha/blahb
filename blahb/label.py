import numba
from .utils import exponential_search
from .strgen import *


def _split_init_into_coords_init_str(dim):
    return "coords_{} = loc[:, {}]".format(dim, dim)


def split_init_into_coords_init_str(ndim):
    return '\n'.join([_split_init_into_coords_init_str(dim)
                      for dim in range(ndim)])


update_cursor_dim_0_base_string = """
if shift_0:
    left_edge_0 = c0 - r0
    if shift_0 >= {P_shape_0}:
        # Re-initialize first cursor when all spans are invalidated
        cursors_0[0] = exponential_search(
            coords_0, left_edge_0, start=ends_0[-1])

        ends_0[0] = exponential_search(
            coords_0, left_edge_0 + 1, start=cursors_0[0])
    else:
        # Shift the spans that are still valid, but cursors must be reset
        for sh in range({P_shape_0} - shift_0): #
            cursors_0[sh] = ends_0[sh + shift_0 - 1]
            ends_0[sh] = ends_0[sh + shift_0]

    # Initialize cursors/ends for positions that are not shifted
    shift_0 = min(shift_0, {P_shape_0})
    for sh in range({P_shape_0} - shift_0, {P_shape_0}):
        cursors_0[sh] = exponential_search(
            coords_0, left_edge_0 + sh, ends_0[sh - 1])

        ends_0[sh] = exponential_search(
            coords_0,  left_edge_0 + sh + 1, start=cursors_0[sh])

if shift_0:
    shift_1 = np.int64({P_shape_1})

shift_0 = np.int64(coords_0[i_coord + 1] - c0)
c0 = coords_0[i_coord + 1]
"""


def update_cursor_section_dim_0(neigh_shape):
    """Propagate shift should be true if there is more than one dimension"""
    
    if not len(neigh_shape) > 1:
        raise ValueError(
            "Use specialized 1d labeling function for 1d pixelsets.")
    
    return update_cursor_dim_0_base_string.format(
        P_shape_0=neigh_shape[0],
        P_shape_1=neigh_shape[1],
    )


init_loop_base_string = """
start = cursors_{dim_minus_1}[{lower_dim_index}]
stop = ends_{dim_minus_1}[{lower_dim_index}]

cursors_{dim}[{lower_dim_index}, 0] = exponential_search(
    coords_{dim}, left_edge_{dim}, start=start, stop=stop)

ends_{dim}[{lower_dim_index}, 0] = exponential_search(
    coords_{dim}, left_edge_{dim} + 1, start=cursors_{dim}[{lower_dim_index}, 0], stop=stop)
"""


def param_init_loop(shp, dim):
    assert dim <= len(shp)
    lower_dim_index = ', '.join(
        [i_(low_dim) for low_dim in range(dim)])  # 'i0, i1, ...'
    body = init_loop_base_string.format(
        dim=dim,
        dim_minus_1=dim - 1,
        lower_dim_index=lower_dim_index,
    )
    return loop_over_shape(shp[:dim], body)


shift_loop_base_string = """
for sh in range({dim_shape} - shift_{dim}): #
    cursors_{dim}[{lower_dim_index}, sh] = ends_{dim}[{lower_dim_index}, sh + shift_{dim} - 1]
    ends_{dim}[{lower_dim_index}, sh] = ends_{dim}[{lower_dim_index}, sh + shift_{dim}]
"""


def param_shift_loop(shp, dim):
    assert len(shp) > dim
    lower_dim_index = ', '.join(
        [i_(low_dim) for low_dim in range(dim)])  # 'i0, i1, ...'
    body = shift_loop_base_string.format(
        dim=dim,
        dim_shape=shp[dim],
        lower_dim_index=lower_dim_index,
    )
    return loop_over_shape(shp[:dim], body)


set_higher_shift_string = """shift_{dim_plus_1} = {dim_plus_1_shape}"""


def param_set_higher_shift(shp, dim):
    if len(shp) - dim < 2:
        return ''
    else:
        return set_higher_shift_string.format(
            dim_plus_1=dim + 1, dim_plus_1_shape=shp[dim + 1])


set_new_cursor_loop_base_exponential_search_string = """
start = cursors_{dim_minus_1}[{lower_dim_index}]
stop = ends_{dim_minus_1}[{lower_dim_index}]

for sh in range({dim_shape} - shift_{dim}, {dim_shape}):
    start = max(start, ends_{dim}[{lower_dim_index}, sh - 1])
    cursors_{dim}[{lower_dim_index}, sh] = exponential_search(
        coords_{dim}, left_edge_{dim} + sh, start=start, stop=stop)

    ends_{dim}[{lower_dim_index}, sh] = exponential_search(
        coords_{dim}, left_edge_{dim} + sh + 1,
        start=cursors_{dim}[{lower_dim_index}, sh], stop=stop)
"""

set_new_cursor_loop_base_linear_search_string = """
start = cursors_{dim_minus_1}[{lower_dim_index}]
stop = ends_{dim_minus_1}[{lower_dim_index}]

for sh in range({dim_shape} - shift_{dim}, {dim_shape}):
    start = max(start, ends_{dim}[{lower_dim_index}, sh - 1])

    for i in range(start, stop + 1):
        if coords_{dim}[i] >= left_edge_{dim} + sh or i == stop:
            cursors_{dim}[{lower_dim_index}, sh] = i
            break

    start = cursors_{dim}[{lower_dim_index}, sh]
    for i in range(start, stop + 1):
        if coords_{dim}[i] > left_edge_{dim} + sh or i == stop:
            ends_{dim}[{lower_dim_index}, sh] = i
            break
"""


def param_set_new_cursor_loop(shp, dim):
    assert len(shp) > dim
    lower_dim_index = ', '.join(
        [i_(low_dim) for low_dim in range(dim)])  # 'i0, i1, ...'
    
    if dim < 2:
        base_str = set_new_cursor_loop_base_exponential_search_string
    else:
        base_str = set_new_cursor_loop_base_linear_search_string
    
    body = base_str.format(
        dim=dim,
        dim_shape=shp[dim],
        dim_minus_1=dim - 1,
        lower_dim_index=lower_dim_index
    )
    return loop_over_shape(shp[:dim], body)


minimize_shift_string = """shift_{dim} = min(shift_{dim}, {dim_shape})"""


def minimize_shift(dim, dim_shape):
    return minimize_shift_string.format(dim=dim, dim_shape=dim_shape)


cursor_loops_string = """
if shift_{dim}:
    left_edge_{dim} = c{dim} - r{dim}
    right_edge_{dim} = c{dim} + r{dim}
    if shift_{dim} >= {dim_shape}:
        {init_loop}
    else:
        {shift_loop}
    {minimize_shift}
    {set_new_cursor_loop}
    {set_higher_shift}

shift_{dim} = np.int64(coords_{dim}[i_coord + 1] - c{dim})
c{dim} = coords_{dim}[i_coord + 1]

"""


def param_cursor_loops(shp, dim):
    return cursor_loops_string.format(
        dim=dim,
        dim_shape=shp[dim],
        init_loop=indent_block(param_init_loop(shp, dim), 2, first_line=0),
        shift_loop=indent_block(param_shift_loop(shp, dim), 2, first_line=0),
        minimize_shift=minimize_shift(dim, shp[dim]),
        set_new_cursor_loop=indent_block(param_set_new_cursor_loop(shp, dim),
                                         1, first_line=0),
        set_higher_shift=param_set_higher_shift(shp, dim)
    )


last_dim_loop_string = """
c{dim} = coords_{dim}[i_coord]
left_edge_{dim} = c{dim} - r{dim}
right_edge_{dim} = c{dim} + r{dim}

{do_something_with_central_pixel}

{low_dim_loop}"""

last_dim_loop_body_string_hyperrect = """
cursor = cursors_{dim_minus_1}[{lower_dim_index}]
while cursor < ends_{dim_minus_1}[{lower_dim_index}] and coords_{dim}[cursor] < left_edge_{dim}:
    cursor += 1
cursors_{dim_minus_1}[{lower_dim_index}] = cursor  # Save the position we reached along the shard
while cursor < ends_{dim_minus_1}[{lower_dim_index}] and coords_{dim}[cursor] <= right_edge_{dim}:

    {do_something_with_neighbors}

    cursor += 1"""

last_dim_loop_body_string_struct_el = """
cursor = cursors_{dim_minus_1}[{lower_dim_index}]
while cursor < ends_{dim_minus_1}[{lower_dim_index}] and coords_{dim}[cursor] < left_edge_{dim}:
    cursor += 1
cursors_{dim_minus_1}[{lower_dim_index}] = cursor  # Save the position we reached along the shard

_end = ends_{dim_minus_1}[{lower_dim_index}]
for i_final in range({last_dim_shape}):
    while cursor < _end and coords_{dim}[cursor] < left_edge_{dim} + i_final:
        cursor += 1
    if cursor == _end:
        break
    elif coords_{dim}[cursor] == left_edge_{dim} + i_final and struct_el[{lower_dim_index}, i_final]:
        {do_something_with_neighbors}"""


def param_last_dim_loop(shp, struct_el):
    """
    shp : Shape of the hyperrect around the central pixel to search for neighbors
    struct_el: True/False on whether a structuring element of shape shp will be used."""
    
    assert len(shp)
    last_dim = len(shp) - 1
    lower_dim_index = ', '.join(
        [i_(low_dim) for low_dim in range(last_dim)])  # 'i0, i1, ...'
    
    if struct_el:
        loop_body = last_dim_loop_body_string_struct_el.format(
            dim=last_dim,
            dim_minus_1=last_dim - 1,
            last_dim_shape=shp[-1],
            lower_dim_index=lower_dim_index,
            do_something_with_neighbors="{do_something_with_neighbors}"
        )
    else:
        loop_body = last_dim_loop_body_string_hyperrect.format(
            dim=last_dim,
            dim_minus_1=last_dim - 1,
            lower_dim_index=lower_dim_index,
            do_something_with_neighbors="{do_something_with_neighbors}"
        )
    
    loop = loop_over_shape(shp[:-1], loop_body)
    
    return last_dim_loop_string.format(
        dim=last_dim,
        low_dim_loop=loop,
        do_something_with_central_pixel="{do_something_with_central_pixel}",
    )


# Find the ancestors of neighbor index
find_central_ancestor_string = """
central_ancestor = labels[i_coord]
while labels[central_ancestor] != central_ancestor:
    prev_central_ancestor = central_ancestor
    central_ancestor = labels[central_ancestor]
    labels[prev_central_ancestor] = central_ancestor"""

find_neighbor_ancestor_string = """
#central_ancestor = labels[i_coord]

neighbor_ancestor = labels[cursor]

if neighbor_ancestor == central_ancestor:
    break

#while labels[central_ancestor] != central_ancestor:
#    prev_central_ancestor = central_ancestor
#    central_ancestor = labels[central_ancestor]
#    labels[prev_central_ancestor] = central_ancestor

while labels[neighbor_ancestor] != neighbor_ancestor:
    prev_neighbor_ancestor = neighbor_ancestor
    neighbor_ancestor = labels[neighbor_ancestor]
    labels[prev_neighbor_ancestor] = neighbor_ancestor

if neighbor_ancestor == central_ancestor:
    labels[cursor] = central_ancestor
    labels[i_coord] = central_ancestor

if neighbor_ancestor < central_ancestor:
    labels[cursor] = neighbor_ancestor
    labels[i_coord] = neighbor_ancestor
    labels[central_ancestor] = neighbor_ancestor
    central_ancestor = neighbor_ancestor
else:  # neighbor_ancestor > central_ancestor:
    labels[cursor] = central_ancestor
    labels[i_coord] = central_ancestor
    labels[neighbor_ancestor] = central_ancestor"""

finalize_labels_str = """
for i in range(labels.size-1, -1, -1):
    i = numba.int_(i)
    anc = i

    while anc != labels[anc]:
        anc = numba.int_(labels[anc])

    while labels[i] != anc:
        i_prev = i
        labels[i_prev] = anc
        i = numba.int_(labels[i])
"""

label_func_string = """
def label(loc, labels, {struct_el}):
    
    {split_loc_to_coords}
    
    # Number of coordinates
    n = coords_0.size

    {shift_init_strings}
    {cursors_init_strings}
    {ends_init_strings}
    {coord_init_strings}
    {range_init_strings}
    for i_coord in range(n):
        {coord_loop_body}
    {finish_up}
    return labels"""


def find_neighbors_func(neigh_shape, use_struct_el):
    """ Build a nopython function to label locations.
    
    Arguments
    ---------
    neigh_shape : ndim-tuple of ints
        Should all be odd numbers so that the central pixel remains well
        defined
    use_struct_el : bool
        Flag indicating that the structuring element is not a perfect
        hyperect neighborhood (i.e. np.all(struct_el) == False)
    
    Returns
    -------
    Numba nopython function that labels IndexSet locations that are neighbors.
    """
    ndim = len(neigh_shape)
    fn = label_func_string.format(
        struct_el='struct_el' if use_struct_el else '',
        
        split_loc_to_coords = indent_block(
            split_init_into_coords_init_str(ndim), 1, first_line=0),
        
        coord_dim_names=coord_dim_names(ndim),
        
        coord_init_strings=indent_block(coord_init_strings(ndim),
                                        first_line=0),
        
        shift_init_strings=indent_block(shift_init_strings(neigh_shape), 1,
                                        first_line=0),
        
        cursors_init_strings=indent_block(
            cursors_init_strings(neigh_shape, np.int64), first_line=0),
        
        ends_init_strings=indent_block(
            ends_init_strings(neigh_shape, np.int64), first_line=0),
        
        range_init_strings=indent_block(range_init_strings(neigh_shape),
                                        first_line=0),
        coord_loop_body=''.join(
            [indent_block(update_cursor_section_dim_0(neigh_shape), 2)] +
            [indent_block(param_cursor_loops(neigh_shape, i), 2) for i in
             range(1, ndim - 1)] +
            [indent_block(param_last_dim_loop(neigh_shape, use_struct_el), 2)]
        ),
        finish_up=indent_block(finalize_labels_str, 1, first_line=0),
    )
    
    indent_amount = ndim + 3 if use_struct_el else ndim + 2
    
    fn = fn.format(
        do_something_with_central_pixel=indent_block(
            find_central_ancestor_string, 2, first_line=0),
        do_something_with_neighbors=indent_block(find_neighbor_ancestor_string,
                                                 indent_amount, first_line=0),
    )
    
    return fn


__saved_neighbor_funcs = dict()


def build_label_func(shape, use_struct_el):
    if (shape, use_struct_el) in __saved_neighbor_funcs:
        return __saved_neighbor_funcs[(shape, use_struct_el)]
    fn_string = find_neighbors_func(shape, use_struct_el)
    _loc = dict()
    exec(fn_string, globals(), _loc)
    fn = numba.jit(_loc['label'], nopython=True, nogil=True)
    __saved_neighbor_funcs[(shape, use_struct_el)] = fn
    return fn


@numba.njit
def merge_chunked_labels(master_labels, chunk_labels, overlap_start,
                         overlap_stop):
    n_overlapping = overlap_stop - overlap_start
    
    for i_chunk, i_master in enumerate(range(overlap_start, overlap_stop)):
        # print(i_chunk, i_master)
        anc_master = master_labels[i_master]
        while master_labels[anc_master] != anc_master:
            anc_master_prev = anc_master
            anc_master = master_labels[anc_master]
            master_labels[anc_master_prev] = anc_master
        
        anc_chunk = chunk_labels[i_chunk] + overlap_start
        while master_labels[anc_chunk] != anc_chunk:
            anc_chunk_prev = anc_chunk
            anc_chunk = master_labels[anc_chunk]
            master_labels[anc_chunk_prev] = anc_chunk
        
        if anc_chunk < anc_master:
            master_labels[anc_master] = anc_chunk
        elif anc_master < anc_chunk:
            master_labels[anc_chunk] = anc_master
    
    fin = overlap_stop + chunk_labels.size - n_overlapping
    master_labels[overlap_stop:fin] = (
        chunk_labels[n_overlapping:] + overlap_start)


@numba.njit([numba.void(numba.uint8[:]), numba.void(numba.uint16[:]),
             numba.void(numba.uint32[:]), numba.void(numba.uint64[:])],
            nogil=True)
def finalize_labels(labels):
    """Ensure that labels are root or point to a root."""
    for i in range(labels.size - 1, -1, -1):
        i = numba.int_(i)
        anc = i
        
        while anc != labels[anc]:
            anc = numba.int_(labels[anc])
        
        while labels[i] != anc:
            i_prev = i
            labels[i_prev] = anc
            i = numba.int_(labels[i])