import numpy as np
import math

from concurrent.futures import ThreadPoolExecutor

from .settings import parse_chunk_args
from .utils import searchsorted
from .label import build_label_func, finalize_labels, merge_chunked_labels
from .utils import repeat
from .indexset import IndexSet
from .multiblob import MultiBlob

from .setops import union
from .flags import *


def is_odd(val):
    return val % 2 is not 0


_base_neighborhoods = dict()


class _BaseNeighborhood:
    """Representation of n-dimensional spatial contiguity."""
    
    def __init__(self, offsets, struct_el, ranges, symmetric):
        struct_el = struct_el.copy()
        struct_el.flags.writeable = False
        
        self._struct_el = struct_el
        
        self._offsets = tuple(map(np.array, zip(*offsets)))
        for o in self._offsets:
            o.flags.writeable = False
        
        self._ranges = ranges
        self._ranges.flags.writeable = False
        
        self._label_func = None
        self._border_func = None
        self._cc_neighbors = None
        
        self._symmetric = symmetric
        _base_neighborhoods[offsets] = self
    
    @property
    def cc_neighbors(self):
        if self._cc_neighbors is None:
            # Exclude the central pixel
            offsets = list(i for i in zip(*self._offsets) if i != (0, 0))
            distance = lambda y, x: y * y + x * x
            rotation = lambda y, x: math.atan2(y, x)
            key = lambda yx: (rotation(*yx), distance(*yx))
            srt = sorted(offsets, key=key)
            split = srt.index(min(srt))
            offsets = srt[split:] + srt[:split]
            y_neigh, x_neigh = zip(*offsets)
            self._cc_neighbors = np.array(y_neigh), np.array(x_neigh)
        return self._cc_neighbors
    
    @property
    def symmetric(self):
        R, ndim = self._ranges, self._struct_el.ndim,
        struct_el = self._struct_el
        
        for dim in range(ndim):
            
            t_rev = tuple(slice(None) if d is not dim else slice(R[dim])
                          for d in range(ndim))
            t = tuple(slice(None) if d is not dim else slice(-1, R[dim], -1)
                      for d in range(ndim))
            
            if not np.all(struct_el[t_rev] == struct_el[t]):
                return False
        return True
    
    @property
    def is_full(self):
        return np.all(self._struct_el)
    
    @property
    def label_func(self):
        if self._label_func is None:
            fn = build_label_func(self._struct_el.shape, not self.is_full)
            self._label_func = fn
        return self._label_func


class Neighborhood:
    def __init__(self, struct_el, symmetric=True):
        """Create a Neighborhood object.

        Arguments
        ---------
        struct_el : numpy.array(dtype=bool, ndim=ndim)
            A scipy.ndimage type structuring element that gives a
            spatial template for neighborhood membership. True elements are
            considered neighbors of the central element. The length along
            each dimension should be an odd number, so that the "central"
            (or "middle") element is well-defined. If the dtype
            of struct_el is not `bool`, then all nonzero elements will
            be considered neighbors of the central pixel.
        symmetric : [True] | False
            Flag indicating that the structuring element should be made
            symmetric. Default is True.
        """
        if type(struct_el) is _BaseNeighborhood:
            self._base_neigh = struct_el
            return
        
        if not all(is_odd(d) for d in struct_el.shape):
            raise ValueError("struct_el must have an odd number of elements"
                             " along each dimension.")
        ranges = np.array(struct_el.shape, dtype=int) // 2
        offsets = tuple(o - r for o, r in zip(np.where(struct_el), ranges))
        
        # Now, make symmetrical the struct_el and offsets
        if symmetric:
            struct_el = struct_el.copy()
            struct_el[tuple(r - o for o, r in zip(offsets, ranges))] = True
        
        offsets = ((o - r) for o, r in zip(np.where(struct_el), ranges))
        offsets = tuple(zip(*tuple(map(tuple, offsets))))
        
        if offsets in _base_neighborhoods:
            self._base_neigh = _base_neighborhoods[offsets]
        else:
            self._base_neigh = _BaseNeighborhood(
                offsets, struct_el, ranges, symmetric)
    
    def __repr__(self):
        return "Neighborhood(n={}, ndim={}, ranges={}, {}={:.2f}, {})".format(
            len(self.offsets[0]), self.ndim, self.ranges, chr(961), self.rho,
            '"complex"')
    
    # Static methods
    
    @staticmethod
    def from_rank(ndim, rank):
        """Construct a Neighborhood with a given dimensionality and rank.

        Arguments:
            ndim - int
                Dimensionality of the neighborhood

            rank - int
                The squared N-d Euclidean Distance from the central pixel,
                under which pixels are considered neighbors.
                Should be an integer in the inclusive range: [1, `ndim`].
                Values outisde of this range will raise an error.

        Note:
            This function is similar to the scipy.ndimage function
            `generate_binary_structure` [], although 'ndim' and 'rank' are
            called 'rank' and 'conn', respectively. The structuring element
            will have length 3 along each dimension, with the central pixel
            located at [1, 1, 1, ...].
        """
        if not ndim > 0:
            raise ValueError('`ndim` must be positive.')
        
        if rank < 1:
            raise ValueError('rank must be >= 1, not {}.'.format(rank))
        elif rank > ndim:
            raise ValueError('rank must be <= ndim, not {}.'.format(rank))
        
        D = np.meshgrid(*[range(-1, 2) for i in range(ndim)])
        struct_el = np.sum([a ** 2 for a in D], axis=0) <= rank
        return Neighborhood(struct_el)
    
    @staticmethod
    def from_offsets(*coords):
        """Construct a Neighborhood with offsets from the central pixel.

        Arguments
        ---------
        *values - sequence of numpy.arrays(size=n, ndim=1)
            Sequence of numpy arrays giving the coordinates along each
            axis. The number of input arrays determines the dimensionality
            of the neighborhood.

        Notes
        -----
        Per the definition of Neighborhood, a pixel is always considered a
        neighbor to itself (reflexivity). Therefore, the central pixel
        (0, 0, 0, ...) is automatically included in offsets, even if it is
        not a member of the input arguments.

        Neighborhoodship is also symmetric. If B is a neighbor of A, then A
        is also a neighbor of B for a given neighborhood N. Because of this
        the negation of any input coordinate will also be included in the
        neighborhood. For example, If (-1, 2, 0) is among the input
        coordinates, then -(-1, 2, 0), or (1, -2, 0) is also included in
        the neighborhood).
        """
        ndim = len(coords)
        # assert np.array(offsets).ndim == 2
        
        if not all(len(c) == len(coords[0]) for c in coords):
            raise ValueError("The the length of coordinate offsets must"
                             "be identical along each dimension.")
        
        coords = tuple(np.array(c) for c in coords)  # Ensure we have np arrays
        radii = np.max(np.abs(coords),
                       axis=1)  # 'radius' of the hyperrectangle
        
        struct_el = np.zeros(radii * 2 + 1, dtype=bool)
        struct_el[tuple(c + r for c, r in zip(coords, radii))] = True
        
        # Center pixel (a pixel is always a neighbor to self)
        struct_el[tuple(radii)] = True
        
        return Neighborhood(struct_el)
    
    @staticmethod
    def VonNeumann():
        return Neighborhood.from_rank(ndim=2, rank=1)
    
    @staticmethod
    def Moore():
        return Neighborhood.from_rank(ndim=2, rank=2)
    
    @staticmethod
    def Cube():
        return Neighborhood.from_rank(ndim=3, rank=3)
    
    # Properties
    
    @property
    def cc_neighbors(self):
        return self._base_neigh.cc_neighbors
    
    @property
    def label_func(self):
        return self._base_neigh.label_func
    
    @property
    def is_full(self):
        """Returns True if the neighborhood is a filled hyperrectangle."""
        return np.all(self.struct_el)
    
    @property
    def ndim(self):
        return self.struct_el.ndim
    
    @property
    def offsets(self):
        """Relative offsets of all neighbor pixels."""
        return self._base_neigh._offsets
    
    @property
    def ranges(self):
        """The 'radii' along each axis of the bounding hyperrectangle."""
        return self._base_neigh._ranges
    
    @property
    def rho(self):
        return self.struct_el.sum() / np.prod(self.struct_el.shape)
    
    @property
    def shape(self):
        return self.struct_el.shape
    
    @property
    def struct_el(self):
        """True where pixels neighbors of the central element."""
        return self._base_neigh._struct_el
    
    @property
    def symmetric(self):
        return self._base_neigh.symmetric
    
    # Methods
    
    def drop(self, dims):
        """Remove a dimension from this Neighborhood.
        
        Arguments
        ---------
        dims : sequence of integers specifying which dimensions to remove.
        """
        struct_el = self.struct_el.any(axis=tuple(dims))
        return Neighborhood(struct_el, self.symmetric)

    def buffer(neighborhood, indexset, **chunk_args):
        """Create a buffer of pixels around existing locations.

        Arguments
        ---------
        indexset : IndexSet object
        neighborhood : Neighborhood object
            Should have the same dimensionality as `indexset`.

        Returns
        -------
        An IndexSet object containing all of the points in `indexset`
        and their neighbors. If `indexset` has data, then the result
        will have `indexset`'s data with NaN in new locations.
        """
        ranges = neighborhood.ranges
        neighbors = neighborhood.offsets
        expanded_size = neighbors[0].size
        to_combine = []
        for neighs in zip(*neighbors):
            neighs = np.array(neighs)
            if np.all(neighs == 0):
                continue
            loc = indexset.loc.copy()
            loc += neighs
            to_combine.append(IndexSet(loc, SORTED_UNIQUE))
    
        new = union(to_combine, MERGE=None, **chunk_args)
    
        MERGE = np.array([DATA_NANFIRST], dtype=np.uint8)
        return union([indexset, new], MERGE=MERGE, **chunk_args)
    
    def label(neigh, indexset, **chunk_args):
    
        if neigh.ndim is not indexset.ndim:
            raise TypeError("Dimensionality of Neighborhood ({}) and "
                            "PixelSet ({}) do not match.".format(
                neigh.ndim, indexset.ndim))
    
        assert indexset.n < np.iinfo(np.uint32).max
        n_workers, n_chunks, min_chunk_sz, max_chunk_sz = parse_chunk_args(
            **chunk_args)
    
        struct_el = neigh.struct_el
        if n_workers == n_chunks == 1 and indexset.n < max_chunk_sz:
            labels = np.arange(indexset.n, dtype=np.uint32)
            if neigh.is_full:
                neigh.label_func(indexset.loc, labels)
            else:
                neigh.label_func(indexset.loc, labels, struct_el)
            return labels
    
        n = indexset.n
        n_chunks = max(n_workers, n // max_chunk_sz)
        chunk_size = indexset.n // n_chunks
        
        dim_0_coords = indexset.loc[:, 0]
    
        index = np.linspace(0, n, num=n_chunks + 2, dtype=int)
        vals = dim_0_coords[
            index[1:-1]]  # Without the caps (added back in later)
        vals = np.unique(vals)
        starts = [0] + list(searchsorted(dim_0_coords, vals))
        stops = list(
            searchsorted(dim_0_coords, vals + neigh.ranges[0])) + [n]
        slices = [slice(start, stop) for start, stop in zip(starts, stops)
                  if start != stop]
        splits = [indexset.take(s).loc for s in slices]
    
        # The arrays that will passed with each func
        label_list = [np.arange(s.shape[0], dtype=np.uint32) for s in splits]
    
        if neigh.is_full:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(neigh.label_func, splits, label_list))
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(
                    neigh.label_func, splits, label_list,
                    repeat(struct_el)))
    
        labels = np.empty(indexset.n, dtype=np.uint32)
        labels[:label_list[0].size] = label_list[0]
        for i in range(1, len(label_list)):
            overlap_start = slices[i].start
            overlap_stop = slices[i - 1].stop
            ll = label_list[i]
            merge_chunked_labels(labels, ll, overlap_start, overlap_stop)
    
        if len(label_list) > 1:
            assert overlap_start + ll.size == labels.size
        
        finalize_labels(labels)
        return labels
    
    def multiblob(self, indexset, **chunk_args):
        """Create a MultiBlob by labeling an IndexSet."""
        labels = self.label(indexset, **chunk_args)
        result = MultiBlob(indexset.loc, labels)
        result._data = indexset.data
        return result
        