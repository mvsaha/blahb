"""
Main library class
"""

import numpy as np
import numba
from collections import OrderedDict

# NOTE! The following are at the bottom of this file to avoid circular import:
# from .take import take_
# from .sel_result import SelResult_

from .indexset import IndexSet, make_empty
from .utils import exponential_search
from .flags import *

# For indexsets with less elements that this we will not cache bounds
_bounds_threshold = 32

spec = OrderedDict()
arr = np.array([[]], dtype=np.int32)
int32_2d_arr_typ = numba.typeof(arr)

spec['_loc'] = int32_2d_arr_typ
spec['_bounds'] = numba.optional(int32_2d_arr_typ)
spec['_sort_order'] = numba.optional(int32_2d_arr_typ)
spec['_data'] = numba.optional(numba.float32[:, :])
spec['_labels'] = numba.uint32[:]


@numba.jitclass(spec)
class MultiBlob:
    """Class representing a collection of individual Blobs"""
    
    def __init__(self, loc, labels):
        """Initialize an IndexSet with locations.
        
        Arguments
        ---------
        loc : n-by-ndim int32 array
            The index locations to be stored by this IndexSet. These should
            already be sorted and unique. No copy of this array will be made
            (it will be reference).
        labels : uint32 array
            Gives the label membership for each row of labels.
        label_sort : uint32
            Indirect stable sort of labels
        
        Raises
        ------
        TypingError : if loc.dtype is not int32
        """
        self._loc = loc
        self._labels = labels
    
    # Properties
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def bounds(self):
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
    def n(self):
        """The number of rows in self.loc"""
        return self._loc.shape[0]
    
    @property
    def ndim(self):
        return self._loc.shape[1]
    
    @property
    def loc(self):
        return self._loc
    
    @property
    def is_empty(self):
        return self.n == 0
    
    # Methods
    
    def copy(self):
        """Copy all of the data in this IndexSet into a completely new one."""
        FLAGS = UNIQUE | SORTED | CONSUME
        ret = IndexSet(self._loc.copy(), FLAGS)
        ret.data = self.data.copy()
        return ret
    
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
        dimensions and n is the number of locations.
        """
        if not len(coord) == self.ndim:
            raise ValueError("IndexSet.find_loc: len(coord) must equal ndim.")
        start, stop = 0, self.n
        for dim, c in enumerate(coord):
            coords = self.loc[:, dim]
            start = exponential_search(coords, c, start, stop)
            stop = exponential_search(coords, c + 1, start, stop)
            if c != coords[start]:
                return False, start
        return True, start
    
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
            return self, make_empty(self.ndim)
        
        a_loc, b_loc = self.loc[:index], self.loc[index:]
        a = IndexSet(a_loc, SORTED_UNIQUE)
        b = IndexSet(b_loc, SORTED_UNIQUE)
        if self.data is not None:
            a.data, b.data = self.data[:index], self.data[index:]
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
        return take_(self, where)


from .sel_result import SelResult_
from .take import take_