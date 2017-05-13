#`blahb` - A python3 library for object-based image analysis.

## What `blahb` does:
* Represents N-dimensional indexes into an image and associated pixel data
* Makes use of `numpy` and `numba` for efficiency and compatibility
* Provides performant set operations, connected component labeling with arbitrary Neighborhoods, and some common fixed-grid spatial operations such a buffering.

## What `blahb` doesn't do:
* Geographic transformations.
* Represent continuous coordinates
* Anything that isn't rectilinear pixels on a regular grid.
* Real-time image processing

## When should you use `blahb`?
* If you have a sparse N-dimensional image, perhaps with associated data, on which you want to perform spatial image processing techniques.
* You already use `numba` and accelerate your code with `nopython=True`.
* Your images consist of chunks of a larger global grid.
* You do not know how large your domain will be at the beginning of your analysis.

[Full Documentation](https://mvsaha.github.io/blahb)