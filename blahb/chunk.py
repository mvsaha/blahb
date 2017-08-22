"""
Breaking up an IndexSet into smaller pieces.
"""

from .encoding import same_encoding
from .settings import parse_chunk_args


def gen_cochunks(objs, filter_chunk=None, filter_group=None, **chunk_args):
    """Generate sets of chunks from a sequence of IndexSets.

    Arguments
    ---------
    objs : sequence of IndexSets
    filter_chunk : Callable
        Takes a proposed IndexSet chunk and determines if it should
        be added to the round of co-chunked Indexset peers that will
        be yielded. Should return a bool where True means include
        and False means discard.
    filter_group : Callable
        Takes a group of co-chunked IndexSet chunks and determines
        whether or not the whole group of co-chunks should be yielded
        or not. Should return a bool where True means yield and False
        means discard.
    **chunk_args : Options to control chunk size and parallelization.
        Can be some of the following optional arguments:
        * n_workers - Number of worker threads to map chunked operations onto.
        * max_chunk_size - Maximum number of locations
        * min_chunk_size - Minimum number of locations
        * n_chunks - Minimum number of chunks to break an IndexSet into.

    Yields
    ------
    A sequence of objects that are subsets of the input sequence. Each
    object in each generated sequence is guaranteed to not have any
    locations in common with any object in any *other* generated sequence.

    Notes
    -----
    This function is useful for breaking up multiple, large IndexSets
    so that the pieces correspond.

    If we have a sequence of IndexSets objects:
        [a, b, c...]

    This function generates a sequences of chunked IndexSets:
        (a_0, b_0, c_0, ...), (a_1, b_1, c_1, ...), ..., (a_n, b_n, c_n, ...)

    The coordinates in each IndexSet sequence {a, b, c}_i have an
    overlapping domain and possibly the same coordinates. Any IndexSets
    with different subcripts, say <a_i, a_j>, or <a_i, b_j> (where i!=j), are
    guaranteed to have no locations in common.
    """
    
    n_workers, n_chunks, min_chunk_sz, max_chunk_sz = parse_chunk_args(
        **chunk_args)
    
    min_sz = max(min_chunk_sz, (max(o.n for o in objs) // n_chunks))
    chunk_sz = min(max_chunk_sz, min_sz)

    # If no filter functions are provided...
    filter_chunk = filter_chunk or (lambda x: True)
    filter_group = filter_group or (lambda x: True)
    take_max = lambda o: o.take(chunk_sz - 1).loc[0]
    
    while len(objs):  # Some coordinates left in the fixed_pixelsets
        temp = []
        to_yield = []
        
        if any(o.n >= chunk_sz for o in objs):
            
            # sz = min(max(o.n for o in objs), max_chunk_size)
            
            large_objs = (o for o in objs if o.n >= chunk_sz)
            coords = map(take_max, (o for o in large_objs))
            coords = map(tuple, coords)
            
            split_coord = min(coords)
            
            for o in objs:
                
                taken, left = o.split_at_coord(split_coord, True)
                
                if filter_chunk(taken):
                    to_yield.append(taken)
                
                temp.append(left)
        
        else:  # The final yield
            to_yield = objs
        
        if filter_group(to_yield):
            yield tuple(to_yield)
        objs = temp