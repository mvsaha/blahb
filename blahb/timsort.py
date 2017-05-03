"""
Modified implementation of indirect timsort.
"""
#TODO: Technically not yet (need to implement galloping/Exponential sort).

import numba
import numpy as np
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from numblahb.settings import parse_chunk_args
from .utils import enlarge_array

spec = OrderedDict()
spec['_n'] = numba.uint32  # Current size of the array
spec['_indexes'] = numba.uint32[:]

@numba.jitclass(spec)
class Stack:
    def __init__(self, capacity):
        self._indexes = np.empty(max(capacity, 2), np.uint32)
        self._n = 0
    
    def push(self, val):
        if self._n >= self._indexes.size:
            self._indexes = enlarge_array(self._indexes)
        self._indexes[self._n] = val
        self._n += 1
        return self
    
    def pop(self):
        self._n -= 1
        return self._indexes[self._n]
    
    def remove(self, position):
        idx = self._indexes
        for i in range(position, self.n - 1):
            idx[i] = idx[i + 1]
        self._n -= 1
    
    @property
    def n(self):
        return self._n
    
    @property
    def arr(self):
        return self._indexes[:self._n]


def print_stack(stack):
    print(stack._indexes[:stack._n])


@numba.njit
def find_minrun(n):
    """ Find the minimum size that we should
    Taken directly from:
    <http://svn.python.org/projects/python/trunk/Objects/listsort.txt>
    """
    r = 0
    if not n >= 0:
        raise ValueError("array_size must be >= 0")
    while n >= 64:
        r |= n & 1
        n >>= 1
    return n + r


@numba.njit
def assert_sorted(arr):
    if arr.size <= 1:
        return
    for i in range(1, arr.size):
        if arr[i - 1] > arr[i]:
            raise AssertionError("array is not sorted.")


@numba.njit
def insertion_sort_indirect(labels, isort, key=0):
    """The key is the index of the first unsorted run.
    
    Every element of labels is sorted ascending below the key.
    """
    if key < 1:
        key += 1
    #assert_sorted(labels[0:key])
    if labels.shape[0] <= 1:
        return
    while key < labels.size:
        j = key
        while j >= 1 and labels[j] < labels[j - 1]:
            labels[j], labels[j - 1] = labels[j - 1], labels[j]
            isort[j], isort[j - 1] = isort[j - 1], isort[j]
            j -= 1
        key += 1


@numba.njit
def merge_indirect(labels, sort_order, i, j, k):
    """Merge the sorted subarrays labels[i:j] and labels[j:k].

    labels : uint32 array
        The integer values to sort on.
    isort : uint32 arra
        The original positions of each label. Is manipulated identically
        to labels so that the final isort array is an indirect sorter.
    i, j, k : int
        These are the partitions of the labels array that need to be sorted.
        We get away with passing in three indices instead of four
        (i.e. start0, end0, start1, end1) because the runs are always
        adjacent and end0 will always equal start1
    """
    # TODO: Add galloping (AKA exponential search to initial merges site).
    #assert_sorted(labels[i:j])
    #assert_sorted(labels[j:k])
    orig_i = i
    sz_a, sz_b = j - i, k - j
    a, a_isort = labels[i:j].copy(), sort_order[i:j].copy()
    b, b_isort = labels[j:k], sort_order[j:k]

    ia, ib = 0, 0
    
    write_count = 0
    while ia < sz_a and ib < sz_b:
        if a[ia] <= b[ib]:
            labels[i] = a[ia]
            sort_order[i] = a_isort[ia]
            i += 1
            ia += 1
            write_count += 1
        else:  # a[ia] > b[ib]:
            labels[i] = b[ib]
            sort_order[i] = b_isort[ib]
            i += 1
            ib += 1
            write_count += 1
    
    if ia != sz_a:
        labels[i:i + sz_a - ia] = a[ia:sz_a]
        sort_order[i:i + sz_a - ia] = a_isort[ia:sz_a]
        #assert write_count + sz_a - ia == k - orig_i
    elif ib != sz_b:
        labels[i:i + sz_b - ib] = b[ib:sz_b]
        sort_order[i:i + sz_b - ib] = b_isort[ib:sz_b]
        #assert write_count + sz_b - ib == k - orig_i
    else:
        raise RuntimeError("This should never be reached")


@numba.njit
def sort_run(arr, sort_order, start_idx, min_run):
    """Starting at `index` of `arr`, make at least `min_run` elements sorted.

    Arguments
    ---------
    arr : 1d array
    idx : location of `arr` to start sorting at
    min_run : The minimum number of values to sort

    Returns
    -------
    The index up to which `arr` has been sorted (starting at `index` of
    course). The maximum value that will be returned is array.shape[0]
    """
    idx = start_idx
    n = arr.size
    n_1 = n - 1
    if idx >= n_1:
        return n
    
    # Put idx after the already sorted elements
    while idx < n_1 and arr[idx] <= arr[idx + 1]:
        idx += 1
    
    # If min_run elements are already sorted...
    if idx >= start_idx + min_run or idx == arr.size:
        #assert not idx > arr.size
        return idx
    
    # Otherwise: insertion_sort up to min_run elements after idx
    else:
        fin = min(start_idx + min_run, n)
        insertion_sort_indirect(
            arr[start_idx:fin], sort_order[start_idx:fin], key=idx - start_idx)
        #assert_sorted(arr[start_idx:fin])
        return fin


@numba.njit
def collapse_merge(stack, labels, sort_order):
    """Ensure that the timsort invariants are met on the final 3 runs.

    stack : IndexStack instance
    labels : uint32 array
    sort_order : uint32 array

    Returns True if a merge was performed.
    Returns False if no merging is done.
    """
    n = stack.n
    if n <= 2:
        return False
    b, c, d, = stack.arr[n - 3:]  # Last three merge positions
    
    # Invariant condition number 2
    if c - b <= d - c:  # Distance between
        merge_indirect(labels, sort_order, b, c, d)
        stack.remove(n - 2)  # Remove the `c` position from the stack
        return True
    
    # Invariant condition number 1
    elif n > 3:
        a = stack.arr[n - 4]
        if b - a < d - b:  # Invariant number 1 is not met
            if b - a < d - c:  # Merge B with A when A is smaller than B
                merge_indirect(labels, sort_order, a, b, c)
                stack.remove(n - 3)  # Removing `b`
            else:  # Merge B with C
                merge_indirect(labels, sort_order, b, c, d)
                stack.remove(n - 2)  # Removing `c`
            return True
    return False  # All of the invariants are met


@numba.njit
def arange(n):
    ret = np.empty(n, dtype=np.uint32)
    for i in range(n):
        ret[i] = i
    return ret


@numba.njit(nogil=True)
def timsort_(labels, sort_order=None):
    """Modified version of timsort that reports each sorted element's
    original position (indirect_sort).
    
    Arguments
    ---------
    labels : uint32 array
        Labels that we want to sort
    sort_order : uint32 array
        Previous indirect sorting on this array. This argument is for
        internal use only.
    
    Returns
    -------
    sort_order : The indices of
    
    Note
    ----
    `labels` will be sorted in place. Pass in a copy of `labels` if you do
    not want it to be mutated.
    """
    n = labels.size
    if sort_order is None:
        sort_order = arange(n)
    
    min_run = find_minrun(n)
    stack = Stack(20)
    stack.push(0)
    index = 0
    while index != n:
        index = sort_run(labels, sort_order, index, min_run)
        stack.push(index)
        while stack.n >= 3 and collapse_merge(stack, labels, sort_order):
            pass
    
    while stack.n > 2:
        i, j, k = stack.arr[stack.n - 3:]
        merge_indirect(labels, sort_order, i, j, k)
        stack.remove(stack.n - 2)
    
    assert stack.arr[0] == 0 and stack.arr[1] == n
    return sort_order


def is_odd(x):
    return x % 2 is not 0


def yield_pairs(seq, skip_first_if_odd=False):
    start = 1 if (is_odd(len(seq)) and skip_first_if_odd) else 0
    
    if start is 1:
        yield (seq[0:1])
    
    for i in range(start, len(seq), 2):
        yield (seq[i:i + 2])


def repeat(x):
    while True:
        yield x


def merge_wrapper(chunks, labels, sort_order):
    """
    Arguments
    ---------
    chunks : sequence of 1 or 2 slices
    labels :
    sort_order :

    Returns
    -------
    slice object : The merged chunk if only one was given or the
    """
    if len(chunks) is 1:
        return chunks[0]
    elif len(chunks) is 2:
        i, j, k = chunks[0].start, chunks[0].stop, chunks[1].stop
        assert j == chunks[1].start
        # print('merging', i, j, k)
        merge_indirect(labels, sort_order, i, j, k)
        return slice(i, k)  # A single widened chunk
    else:
        raise ValueError("chunksize should be 1 or 2")


def timsort(labels, **chunk_args):
    """Indirect sort an array of connected component labels.

    Arguments
    ---------
    labels : uint32 array
    **chunk_args : Chunking and parallelization named arguments

    Returns
    -------
    sort_order : An indirect sorter array. Same as output of `np.argsort`.

    Notes
    -----
    `labels` will be sorted in place. Make a copy of `labels` if you
    do not want to mutate it.
    """
    n_workers, n_chunks, min_chunk_sz, max_chunk_sz = parse_chunk_args(
        **chunk_args)
    # Chunking this when we are not using >1 worker doesn't make sense
    if n_workers == 1 and labels.size <= max_chunk_sz:
        return timsort_(labels)
    sort_order = arange(labels.size)
    ch = np.linspace(0, labels.size, n_chunks + 1).astype(int)
    ch = list(map(slice, ch[:-1], ch[1:]))
    ch = [c_ for c_ in ch if c_[0] < c_[1]]
    label_chunks = [labels[c_] for c_ in ch]
    sort_order_chunks = [sort_order[c_] for c_ in ch]
    toggle_odds = False
    with ThreadPoolExecutor(max_workers=n_workers) as thread_pool:
        # List-ify results to force completion before moving on
        _ = list(thread_pool.map(timsort_, label_chunks, sort_order_chunks))
        
        for lc in label_chunks:
            assert_sorted(lc)
        results = ch
        while len(results) > 1:
            results = list(thread_pool.map(merge_wrapper,
                                           yield_pairs(results, toggle_odds),
                                           repeat(labels), repeat(sort_order)))
            toggle_odds = not toggle_odds
    return sort_order