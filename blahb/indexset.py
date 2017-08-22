"""
Main library class
"""

import numpy as np
import numba
from collections import OrderedDict

# NOTE! The following are at the bottom of this file to avoid circular import:
# from .take import take_
# from .sel_result import SelResult_

from .encoding import decode, encode, same_encoding
from .lexsort import lexsort_inplace
from .unique import _unique_locations_in_sorted
from .utils import exponential_search, lex_less_Nd, eq_Nd, to_array
from .flags import *
from .border import border2d_

# For indexsets with less elements that this we will not cache bounds
_bounds_threshold = 32

spec = OrderedDict()
arr = np.array([[]], dtype=np.int32)
int32_2d_arr_typ = numba.typeof(arr)

spec['_ndim'] = numba.uint8
spec['_loc'] = int32_2d_arr_typ
spec['_bounds'] = numba.optional(int32_2d_arr_typ)
spec['_sort_order'] = numba.optional(int32_2d_arr_typ)
spec['_data'] = numba.optional(numba.float32[:, :])
spec['_encoding'] = numba.optional(numba.int8[:])

@numba.jitclass(spec)
class IndexSet:
    """Representation of a set of N-dimensiontal pixel locations.
    
    Pixel values are stored in a (n-by-ndim) lexicographically sorted numpy
    array which allows linear computation of set operations, an other fast
    image processing operations.
    
    This is a numba jitclass and can be used in nopython functions.
    """
    def __init__(self, loc, FLAGS=NO_FLAGS):
        """Initialize an IndexSet with locations.
        
        Arguments
        ---------
        loc : n-by-ndim int32 array
            The index locations to be stored by this IndexSet.
        FLAGS : int8
            Bit flags indicating how to process the locations. The final .loc
            array stored in this object must be unique_ and lexicographically
            sorted, these flags indicate whether or not this is already True
            and whether or not we can mutate `loc` if it isn't.
        
        Notes
        -----
        Possible flags (found importing blahb.flags):
        
        NO_FLAGS : Make no assumptions about the ownership, sortedness or
            uniqueness of the input array.
        
        CONSUME : `loc` may be sorted in place. Other users of `loc`
            will observe changes in this array. This only applies to the
            case where SORTED is not set.
        
        SORTED : `loc` is already lexicographically sorted, but may
            have repeated rows. This will NOT mutate `loc`.
        
        UNIQUE : `loc` is already unique_ (has no repeated rows), but
            may not be lexicographically sorted. This will not mutate `loc`.
        
        Multiple flags can be combined with the `|` operator. For example:
        if the input `loc` is sorted and unique we would give the flag:
            SORTED | UNIQUE
        
        Only skip sorting and uniquing if you are absolutely sure that the
        input locations already have these properties. These requirements are
        class invariants and essentially every computation you do will be
        incorrect if these invariants are violated.
        
        If necessary, the dytpe of `loc` will be converted to int32
        """
        if not loc.ndim == 2:
            raise ValueError('`loc` must be 2 dimensional.')
        
        self._sort_order = None
        
        if (FLAGS & COPY) and (FLAGS & CONSUME):
            raise ValueError("copy_ and consume_ cannot "
                             "both be set together.")
        
        if (FLAGS & SORTED) and (FLAGS & UNIQUE):
            if FLAGS & COPY:
                loc = loc.copy()
        
        elif not (FLAGS & SORTED):
            # Sorting modifies the array, so if we don't want to consume it
            # (it's owned by some other object) we should copy it
            if FLAGS & KEEP_SORT_ORDER:
                raise ValueError("Not implemented")
            
            if not (FLAGS & CONSUME):
                loc = loc.copy()
            uniq_loc, sort_order = lexsort_inplace(loc, True)
            
            if not (FLAGS & UNIQUE):
                loc = loc[uniq_loc]
        else:  # Already sorted, must be made unique
            _uniq_loc = _unique_locations_in_sorted(loc)
            loc = loc[_uniq_loc, :]
        
        self._ndim = loc.shape[1]
        self._loc = loc  #.astype(np.int32) #convert_to_int32(loc)
        self._encoding = None
    
    # Properties
    
    @property
    def bounds(self):
        """Determine the lowest and highest values for each dimension."""
        if self._bounds is None:
            if self.n == 0:
                return None
            
            bounds = np.empty((self.ndim, 2), dtype=self._loc.dtype)
            bounds[:, 0] = self._loc[0, :]
            bounds[:, 1] = self._loc[0, :]
            bounds[0, 0] = self._loc[0, 0]
            bounds[0, 1] = self._loc[-1, 0]
            n, ndim = self.n, self.ndim
            for i in range(n):
                for dim in range(1, ndim):
                    bounds[dim, 0] = min(bounds[dim, 0], self._loc[i, dim])
                    bounds[dim, 1] = max(bounds[dim, 1], self._loc[i, dim])
            if self.n < 32:  # Arbitrary threshold for caching the array
                self._bounds = bounds
            return bounds
        else:
            return self._bounds
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, value):
        """Set the data member of this IndexSet.
        
        This checks that the value has the right size and dimensionality.
        If not already a float32 array, it will be converted to one.
        """
        if value.shape[0] != self.n:
            raise ValueError("nrows of data does not match the "
              "number of coordinates in the IndexSet.")
        self._data = value.astype(np.float32)
    
    @property
    def encoding(self):
        return self._encoding
    
    @property
    def n(self):
        """The number of rows in self.loc"""
        return self._loc.shape[0]
    
    @property
    def ndim(self):
        if self.is_encoded:
            return self.encoding.size
        else:
            return self._loc.shape[1]
    
    @property
    def loc(self):
        """Return the matrix of locations that, decoding if necessary."""
        if self.is_encoded:
            if self.n > 10000:
                print('decoding', self.n)
            return decode(self._loc.view(np.uint32),
                          self._encoding.view(np.int8))
        else:
            return self._loc
    
    @property
    def is_empty(self):
        return self.n == 0
    
    @property
    def is_encoded(self):
        return self._encoding is not None
    
    # Methods
    
    def border(self, neighbors, pad):
        """ Find the border pixels of a 2d blob given a neighborhood.
        
        Arguments
        ---------
        neighbors : array, array
            y and x coordinates of neighbor offsets, in rotational order as
            provided by `blahb.Neighborhood.cc_neighbors`
        pad : int, int
            The result of `Neighborhood.ranges` for the same Neighborhood as
            `neighbors`. Specifies the padding needed when creating an
             IndexSet image.
        
        Returns
        -------
        (y, x) : Two arrays of coordinates containing the border pixels in
        counter-clockwise order. The first and last coordinates are identical.
        Depending on the neighborhood, some coordinates other than the end may
        be repeated.
        
        Note
        ----
        This is non-sensical to call on an IndexSet where are the pixels
        are not strongly connected. You should label and extract blobs
        from an IndexSet *before* finding the border.
        
        """
        if self.ndim != 2:
            raise ValueError("IndexSet.border(): only defined for ndim == 2.")
        
        img = self.image2d(pad)
        
        # The 'lower' corner of the hypperectangle spanned by self.loc
        y_lower, x_lower = self.bounds[:, 0]
        
        # 'start' in absolute coordinates
        first_y, first_x = self.take(0).loc[0]
        
        # Offset to relative coordinates
        start = first_y - y_lower + pad[0], first_x - x_lower + pad[1]
        
        # Location of the upper-left pixel in the image
        offset = y_lower - pad[0], x_lower - pad[1]
        
        return border2d_(img, neighbors, start, offset)
    
    def copy(self):
        """Copy all of the data in this IndexSet into a completely new one."""
        FLAGS = UNIQUE | SORTED | CONSUME
        ret = IndexSet(self._loc.copy(), FLAGS)
        ret.data = self.data.copy()
        return ret
    
    def decode(self):
        if self.is_encoded:
            #e = np.asarray(self._encoding)
            if self._encoding is not None:
                self._loc = decode(self._loc.view(np.uint32), self._encoding)
            self._encoding = None
    
    def drop(self, dimension):
        """Drop a dimension from this IndexSet."""
        
        ndim = self.ndim
        
        if not dimension < self.ndim:
            raise ValueError("Indexset.drop(): dimension given exceeds "
              "number of dimensions in IndexSet")
        
        elif dimension < 0:
            raise ValueError(
                "Indexset.drop(): dimension given must be >= 0")
        
        if dimension == ndim - 1:
            new_loc = self.loc[:, :ndim - 1].copy()
            return IndexSet(new_loc, SORTED)
        
        loc = self.loc
        new_loc = np.empty((self.n, ndim - 1), dtype=np.int32)
        j = 0
        
        for i in range(ndim):
            if i == dimension:
                continue
            new_loc[:, j] = loc[:, i]
            j += 1
        
        return IndexSet(new_loc, NO_FLAGS)
    
    def encode(self, encoding):
        """Encode this IndexSet into a packed bit representation.
        
        Arguments
        ---------
        encoding : 1d int8 array
            Gives the number of bits that each dimensions should be packed in
            to. A negative number in this array means that negative numbers
            should be packed as well.
        
        Note
        ----
        This method makes no checks on your data that the values in a given
        dimension (column of .loc) can actually be represented by the number
        of bits specified and the sign. If you specify an insufficient number
        of bits then there will be overflow errors and your comparisons with
        similarly encoded arrays and decoding will give wrong results.
        """
        if self.is_encoded:
            raise ValueError("IndexSet is already encoded.")
        
        _encoding = encoding.astype(np.int8)
        if not _encoding.size == self.ndim:
            raise ValueError("Length of encoding must match number of dims.")
        self._loc = encode(self._loc, _encoding).view(np.int32)
        self._encoding = _encoding
    
    def find_loc(self, coord):
        """Locate a coordinate in this IndexSet
        
        Arguments
        ---------
        coord : sequence of with number of elements equal to self.ndim
        
        Returns
        -------
        (contains, location)
        contains : bool
            Flag indicating if the exact location exists in this IndexSet
        index: int
            The lowest row of self.loc where `coord` could be inserted to
            maintain lexicographical ordering. Will be 0 if `coord` is lex-
            less than all locations or self.n if `coord` is lex-greater than
            all locations.
        
        Notes
        -----
        Time complexity is O(ndim*log(n)) where ndim is is the number of
        dimensions and n is the number of locations in this IndexSet.
        """
        if self.is_encoded:
            _c = to_array(coord)#.resize()
            if _c.ndim > 1:
                raise AssertionError(
                    "IndexSet.find_loc: `coord` must be one-dimensional.")
            encoded_coord = encode(_c.reshape((1, _c.size)), self.encoding)[0]
            return search_coord(self._loc.view(np.uint32), encoded_coord)
        else:
            return search_coord(self.loc, coord)
    
    def image2d(self, pad):
        """The cropped 2d boolean image corresponding to the first two
        dimensions of `loc`."""
        if self.ndim < 2:
            raise ValueError(
                "Indexset.image(): indexset must have >=2 dimensions.")
        
        if pad[0] < 0 or pad[1] < 0:
            raise ValueError("Indexset.image(): padding must be >= 0.")
        
        y_bounds = self.bounds[0]
        x_bounds = self.bounds[1]
        
        y_pad, x_pad = pad
        shp = (y_pad * 2 + 1 + y_bounds[1] - y_bounds[0],
               x_pad * 2 + 1 + x_bounds[1] - x_bounds[0])
        
        img = np.zeros(shp, dtype=numba.boolean)
        loc = self.loc[:, :2]
        
        for i in range(self.n):
            img[loc[i, 0] - y_bounds[0] + y_pad,
                loc[i, 1] - x_bounds[0] + x_pad] = True
        
        return img
    
    def omit(self, dim, selector):
        """Remove matching locations from this IndexSet.
        
        Arguments
        ---------
        dim : int - The dimension to match on
        selector : int | (int, None) | (None, int) | (int, int) | array
            The pattern to match against. A two-tuple indicates a range
            to match against (None indicates an open interval). An integer or
            array of integers to match can also be provided.
        
        Returns
        -------
        A SelResult_ object containing the locations in this IndexSet that
        have not been omitted. Cal `fin()` or `split()` on an SelResult_ object
        to finalize the selection process and get back an IndexSet.
        """
        sr = SelResult_(self)
        return sr.omit(dim, selector)
    
    def reset_data(self):
        """Set the data in this object to None."""
        self._data = None
    
    def sel(self, dim, selector):
        """Keep matching locations from this IndexSet.

        Arguments
        ---------
        dim : int - The dimension to match on
        selector : int | (int, None) | (None, int) | (int, int) | array
            The pattern to match against the locations of a given dimension.
            A two-tuple indicates a range to match against (None indicates an
            open lower or higher interval). An integer or array of integers to
            match can also be provided.
        
        Returns
        -------
        A SelResult_ object containing the locations in this IndexSet that
        have not been omitted. Cal `fin()` or `split()` on an SelResult_ object
        to finalize the selection process and get back an IndexSet.
        """
        sr = SelResult_(self)
        return sr.sel(dim, selector)
    
    def split(self, index):
        """Split this IndexSet into two at the index."""
        if index >= self.n:
            return self, make_empty_like(self)
        
        a = self.take((0, index))
        b = self.take((index, self.n))
        
        if not a.n + b.n == self.n:
            print(self.n, a.n, b.n)
            raise AssertionError("Error in split.")
        
        #if not (same_encoding(self, b) and same_encoding(self, b)):
        #    raise AssertionError("IndexSet.split: encodings differ.")
        
        return a, b
    
    def split_at_coord(self, coord, coord_to_first):
        """Split this IndexSet at the given coordinate values.
        
        Argument
        --------
        coord : len(ndim) sequence
        coord_to_first : True | False
            If True `coord` will be included in the first result. If False,
            `coord` will be included in the second result. This argument only
            applies if `coord` is present in this IndexSet.
        
        Returns
        -------
        a, b : IndexSet objects
            Contains all coordinates lexicographically less than and greater
            than `coord`, respectively.
        """
        contains, index = self.find_loc(coord)
        
        if coord_to_first and contains:
            return self.split(index + 1)
        else:
            return self.split(index)
    
    def take(self, where):
        """Extract locations from an IndexSet based on their sorted position.
        
        Argument
        --------
        where : slice | int | array of int
            The positions to extract.
        
        Returns
        -------
        An IndexSet with the locations at the given position.
        """
        ret = take_(self, where)
        
        #if not same_encoding(self, ret):
        #    raise AssertionError('Result encoding differs in `IndexSet.take`')
        
        return ret


#@numba.njit
#def _to_int32(x):
#    return x.astype(np.int32)


@numba.generated_jit
def convert_to_int32(x):
    """Convert an array to int32 if it isn't already."""
    if isinstance(x, numba.types.Array):
        if x.dtype == numba.int32:
            return lambda x: x
        else:
            return lambda x: x.astype(numba.int32)
    
    raise AssertionError("Can only convert numpy arrays to int32.")


@numba.njit(nogil=True)
def make_empty(ndim):
    loc = np.zeros((0, ndim), dtype=np.int32)
    return IndexSet(loc, SORTED_UNIQUE)


@numba.njit(nogil=True)
def make_empty_like(indexset):
    loc = np.zeros((0, indexset._loc.shape[1]), dtype=np.int32)
    ret = IndexSet(loc, SORTED_UNIQUE)
    if indexset.is_encoded:
        ret._encoding = indexset._encoding
    return ret


def make_indexset(x):
    # Coerce a 1d or 2d array into an IndexSet
    loc = np.array(x).astype(np.int32)
    if loc.ndim == 1:
        loc =  loc[..., None]
    return IndexSet(loc, CONSUME)


def make_data(x):
    # Coerce a 1d or 2d array of floats into data for an IndexSet
    data = np.array(x).astype(np.float32)
    if data.ndim == 1:
        data =  data[..., None]
    return data


@numba.njit(nogil=True)
def concat_sorted_nonoverlapping(objs):
    """Concatenate non-overlapping IndexSets

    Arguments
    ---------
    objs : tuple of IndexSets
        All of the locations in each input IndexSet must be strictly
        lexicographically less than all of the objects after. Additionally,
        all of the objs must have identical encoding.

    Returns
    -------
    IndexSet with merged locations and data.
    
    Raises
    ------
    Nothing. It is up to the user to ensure that the objects are, ordered,
    non-overlapping and have identical encodings.
    """
    n_objs = len(objs)

    first_obj = objs[0]
    n_encoded_cols = objs[0]._loc.shape[1]
    
    n_total = 0
    for o in objs:
        n_total += o.n
    
    loc = np.empty((n_total, n_encoded_cols), dtype=np.int32)
    start = 0
    
    for i in range(n_objs):
        
        indexset = objs[i]
        # Ensure that vars are non-overlapping
        if start > 0 and indexset.n:
            
            if not same_encoding(first_obj, indexset):
                raise ValueError("concat_sorted_nonoverlapping: "
                  "all objects must have identical encoding.")
        
        stop = start + indexset.n
        loc[start:stop] = indexset._loc
        start = stop
    
    result = IndexSet(loc, SORTED_UNIQUE)
    result._encoding = first_obj.encoding
    
    # Determine if there is any data attached to the result
    n_data_col = 0
    for i in range(n_objs):
        if objs[i].data is not None:
            if n_data_col != 0 and objs[i].data.shape[1] != n_data_col:
                raise ValueError("Number of data columns varies between objs.")
            n_data_col = objs[i].data.shape[1]
    
    if n_data_col:
        data = np.full((n_total, n_data_col), np.nan, dtype=np.float32)
        start = 0
        for i in range(n_objs):
            indexset = objs[i]
            if indexset.data is not None:
                stop = start + indexset.n
                data[start:stop] = indexset.data
                start = stop
        result.data = data
    
    return result


@numba.njit(nogil=True)
def search_coord(loc, coord):
    """Find a row in an array of coordinates."""
    stop, ndim = loc.shape
    start = 0
    
    if not len(coord) == ndim:
        raise ValueError("search_coord: Length of `coord` must equal number "
                         "of columns in `loc`.")
    
    for dim in range(ndim):
        c = coord[dim]
        values = loc[:, dim]
        start = exponential_search(values, c, start, stop)
        stop = exponential_search(values, c + 1, start, stop)
        
        if start == stop or c != values[start]:
            return False, start
    
    return True, start


def is_indexset_subclass(obj):
    """Determine if an object is an instance of IndexSet."""
    return numba.typeof(obj) is IndexSet.class_type.instance_type


from .sel_result import SelResult_
from .take import take_
