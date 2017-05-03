
__global_settings = {
    
    # The number of threads to spin up for parallel operations
    'n_workers': 1,
    
    # The number of chunks to split into, after observing max_chunk_size
    'n_chunks' : -1,
    
    # The absolute maximum number of elements in a chunk. This argument takes
    # priority when chunking
    'max_chunk_size': 2 ** 20,
    
    # The minimum chunk size to observe.
    'min_chunk_size': 2 ** 13,
}


def get_options(x):
    return __global_settings[x]


def set_options(**kwargs):
    __global_settings.update(kwargs)


def parse_chunk_args(**chunk_args):
    if "n_workers" in chunk_args:
        n_workers = chunk_args["n_workers"]
    else:
        n_workers = get_options('n_workers')
    
    if "n_chunks" in chunk_args:
        n_chunks = chunk_args["n_chunks"]
    else:
        n_chunks = get_options('n_chunks')
    
    if n_chunks < 0:
        n_chunks = get_options('n_workers')
    
    if "min_chunk_size" in chunk_args:
        min_chunk_size = chunk_args["min_chunk_size"]
    else:
        min_chunk_size = get_options('min_chunk_size')
    
    if "max_chunk_size" in chunk_args:
        max_chunk_size = chunk_args["max_chunk_size"]
    else:
        max_chunk_size = get_options('max_chunk_size')
    
    return n_workers, n_chunks, min_chunk_size, max_chunk_size