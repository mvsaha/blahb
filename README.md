`blahb` - A python3 library for object-based image analysis.

###### What `blahb` does:
* Represents N-dimensional indexes into an image and associated pixel data
* Makes use of `numpy` and `numba` for efficiency and compatibility
* Provides performant set operations, connected component labeling with arbitrary Neighborhoods, and some common fixed-grid spatial operations such a buffering.

###### What `blahb` doesn't do:
* Geographic transformations.
* Represent continuous coordinates
* Anything that isn't rectilinear pixels on a regular grid.
* Real-time image processing

###### When should you use `blahb`?
* If you have a sparse N-dimensional image, perhaps with associated data, on which you want to perform spatial image processing techniques.
* You already use `numba` and accelerate your code with `nopython=True`.
* Your images consist of chunks of a larger global grid.
* You do not know how large your domain will be at the beginning of your analysis.

###### Motivation: How is this different than typical numpy/scipy image processing?

In typical `numpy` image processing we work with boolean images, a fixed
grid of `True` or `False`
```python
>>>img = np.array([[0,0,0,1,0],
                   [0,0,0,0,0],
                   [0,0,0,0,0],
                   [0,1,0,0,0],
                   [0,0,0,0,1]], dtype=bool)
```
This is an *implicit* representation of spatial data because we are representing points using a the convention that (0, 0) is the top-left-most pixel and the positions of our `True` values of interest can be computed by their offset relative to start pixel. When we do computations, such as [connected-component labeling](https://en.wikipedia.org/wiki/Connected-component_labeling), we operate in this *implicit* realm.

An alternative, the approach that `blahb` takes, is to represent our `True` pixels *explicitly* by the coordinates that they take. This is easy enough to do in `numpy`:
```python
>>>y, x = np.where(img)
>>>a = set(zip(*(y, x)))
>>>a
{(0, 3), (3, 1), (4, 4)}
```
In other words, the location of `True` values is the set {(0, 3), (3, 1), (4, 4)}

Why is this nicer? Conceptually, it is simpler to think about positive space, about pixels that we *have*. However, there are limited existing tools for performing computations on this representation. This is a gap that `blahb` aims to fill.

*As an added bonus, we have injected a measure of optimism into our workflow by dropping all of the negative baggage of the implicit representation.*
 

Consider another image:
```python
>>>img2 = np.array([[0,0,0,0],
                    [0,1,0,0],
                    [0,0,0,0],
                    [0,0,0,1],
                    [0,0,0,0]], dtype=bool)
```
but instead of (0, 0), the top pixel now represents (2, 3). `True` pixels in this new image are represented by the set `b`:
```python
>>>y_offset, x_offset = 2, 3
>>>b = set(zip(*(y + y_offset, x + x_offset)))
>>>b
{(3, 4), (5, 6)}
```
When we use the *explicit* form, we can perform operations like "*find the union of `True` pixels in both images*" quite easily:
```python
>>>c = a | b
>>>c 
{(0, 3), (3, 1), (3, 4), (4, 4), (5, 6)}
```
The aim of `blahb` is to provide:
- A `numba.jitclass` similar to the above set-of-coordinate tuples that scales well to real image sizes (i.e. tens of millions of pixels) with an arbitrary number of dimensions (such as an image stack).
- Operators like the above `|` for efficiently manipulating comparing and combining these sets.
- Effortless integration with both `numpy` and `numba.njit` nopython functions when you want to write your own extensions.

For the *implicit* spatial representation, this is more involved:
```python
>>>result_shape = img2.shape[0] + y_offset, img2.shape[1] + x_offset
>>>big_img = np.zeros(result_shape, dtype=bool)
>>>big_img[:img.shape[0], :img.shape[1]] = img
>>>big_img[y_offset:y_offset + img2.shape[0],
           x_offset:x_offset + img2.shape[1]] |= img2
>>>set(zip(*np.where(big_img)))  # We get the same result
{(0, 3), (3, 1), (3, 4), (4, 4), (5, 6)}
```
Not only is there more code to write, there is more code to reason about; the potential for errors increases as we move into higher dimensions.


Read more at 
