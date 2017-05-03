"""
Class for tracking the result of selection to allow for chaining.
"""

import numba
import numpy as np

from .bits import *
from .indexset import IndexSet
from .sel import _sel
from .omit import _omit

from collections import OrderedDict

sel_result_spec = OrderedDict()

indexset_type = numba.deferred_type()

sel_result_spec['start'] = numba.int64
sel_result_spec['stop'] = numba.int64
sel_result_spec['indexset'] = indexset_type
sel_result_spec['flags'] = numba.boolean[:]
sel_result_spec['_num_flags'] = numba.int64
sel_result_spec['has_flags'] = numba.bool_


@numba.jitclass(sel_result_spec)
class SelResult_:
    def __init__(self, indexset):
        self.indexset = indexset
        self.start = 0
        self.stop = indexset.n
        
        self.flags = np.zeros(0, dtype=numba.boolean)
        self.has_flags = False
    
    @property
    def num_flags(self):
        if self.has_flags:
            return self._num_flags
        else:
            return self.stop - self.start
    
    def _set_to_empty(self):
        self.start = 0
        self.stop = 0
        self.flags = np.zeros(0, dtype=numba.boolean)
        self.has_flags = False
        self._num_flags = 0
    
    def sel(self, dim, selector):
        """Extract coordinates from an IndexSet where coordinates along a
        given dimension match a selector.
        
        Arguments
        ---------
        dim : int
            The dimension of coordinates to compare against.
        selector :
            The values that are considered a 'match'. Can be one of the
            following patterns:
            
            * val - Extracts coordinates matching a single integer value
            * (None, val) - Extracts coordinates <= val
            * (val, None) - Extracts coordinates >= val
            * (lo, hi) - Extracts where lo <= val <= hi; lo & hi are integers
            * array - Extracts where val
        
        Returns
        -------
        This SelResult_ instance, which is modified. No copy is made.
        
        Notes
        -----
        Selection is a narrowing operation, it will not add points to the
        current selection of flags.
        """
        return _sel(self, dim, selector)
    
    def omit(self, dim, selector):
        """Extract coordinates from an IndexSet where coordinates along a
        given dimension do not match a selector.
        
        Arguments
        ---------
        dim : int
            The dimension of coordinates to compare against.
        selector :
            The values that are considered a 'match'. Can be one of the
            following patterns:

            * val - Extracts coordinates matching a single integer value
            * (None, val) - Extracts coordinates <= val
            * (val, None) - Extracts coordinates >= val
            * (lo, hi) - Extracts where lo <= val <= hi; lo & hi are integers
            * array - Extracts where val

        Returns
        -------
        This SelResult_ instance, which is modified. No copy is made, but
        returning the result allows chaining.

        Notes
        -----
        Omission is a narrowing operation, it will not add points to the
        current selection of flags.
        """
        return _omit(self, dim, selector)
    
    def fin(self):
        """Retrieve the results of coordinate selection."""
        loc = self.indexset.loc[self.start:self.stop]
        if self.has_flags:
            loc = loc[self.flags]

        ret = IndexSet(loc, SORTED | UNIQUE)
        
        if self.indexset.data is not None:
            data = self.indexset.data[self.start:self.stop]
            if self.has_flags:
                data = data[self.flags]
            ret.data = data
        return ret
    
    def split(self):
        """Split an IndexSet into matching an non-matching coordinates."""
        
        n = self.indexset.n
        data = self.indexset.data
        start, stop = self.start, self.stop
        loc = self.indexset.loc
        
        a_loc = self.indexset.loc[start:stop]
        if self.has_flags:
            a_loc = a_loc[self.flags]
        
        a = IndexSet(a_loc, SORTED_UNIQUE)
        
        if self.indexset.data is not None:
            if self.has_flags:
                a.data = data[start:stop][self.flags]
            else:
                a.data = data[start:stop]
        
        if self.has_flags:
            n_b = start + (n - stop) + (start - stop - self.num_flags)
            b_loc = np.concatenate((
                loc[:start], loc[start:stop][~self.flags], loc[stop:]))
            
            b = IndexSet(b_loc, SORTED_UNIQUE)
            if data is not None:
                b.data = np.concatenate((
                    data[:start], data[start:stop][~self.flags], data[stop:]))
        
        else:  # There are no flags, just a simple slice
            if self.start == 0 and self.stop == n:
                b_loc = np.zeros((0, self.indexset.ndim), loc.dtype)
                b = IndexSet(b_loc, SORTED_UNIQUE)
            
            elif start == 0: # None taken from front, we can have a ref slice
                b = IndexSet(loc[stop:], SORTED_UNIQUE)
                b._data = data[stop:] if data is not None else None
            
            elif stop == n:
                b = IndexSet(loc[:start], SORTED_UNIQUE)
                b._data = data[:start] if data is not None else None
        
        return a, b


indexset_type.define(IndexSet.class_type.instance_type)
