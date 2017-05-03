import numpy as np
import scipy.ndimage as ndi

from ..neighborhood import Neighborhood
from ..indexset import IndexSet
from ..bits import *
from ..image import image


def test_label():
    shape = [5, 5]
    lo, hi = -5, 10
    for ndim in range(2, 6):
        print('ndim', ndim)
        for rank in range(1, ndim):
            nx, ny = np.random.randint(1, 2 ** ndim, size=2)
            loc = np.random.randint(lo, hi, (ny, ndim), dtype=np.int32)
            indexset = IndexSet(loc, NO_FLAGS)
            
            neigh = Neighborhood.from_rank(ndim=ndim, rank=rank)
            
            labels = neigh.label(indexset)
            res = np.unique(labels).size
            img = image(indexset)
            ref = ndi.label(img, structure=neigh.struct_el)[1] # of labels
            assert res == ref
        
        shape += [5]