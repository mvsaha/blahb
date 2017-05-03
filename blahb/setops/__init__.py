"""
Set operations on sets of coordinates.

All set operations are O(n), where n is the total number of locations in
the input IndexSets.
"""

from .adiff import asymmetric_difference, asymmetric_difference_
from .sdiff import symmetric_difference, symmetric_difference_
from .union import union, union_, union_multi_
from .intersection import intersection, intersection_, intersection_multi_
