- [0. Motivation](index.md)
- (1) Getting Started
- [2. Set Operations](2_setops.md)
- [3. Associated Data](3_data.md)

# 1. Getting Started: The `IndexSet`

The basic unit of`blahb` is the `IndexSet`.
`IndexSet` is a `numba.jitclass` that stores the values of `True`, and only `True`, pixels in an image. If you are not familiar with `numba`, don't worry.
All of the functionality is available to you in regular python code (or we'll mention when this is not the case).

In each `IndexSet` there is a 2-d array of locations (accessed through the `.loc` attribute).
Here is an example of how locations are stored:
```python

loc = [ ( 1,  3,  4 ),    |
        ( 2,  2,  7 ),    |
        ( 2,  3,  5 ),    |
        ( 2,  4,  2 ),    n = 7 
        ( 3,  1,  4 ),    |
        ( 3,  1,  5 ),    |
        ( 3,  1,  6 ) ]   |

         ---ndim---- = 3
```
In this example we are storing n=7 locations into a 3-dimensional image. Each row represents a unique location, and column represents the locations along a single dimension.

The name of this class gives more hints about how data is represented:
- `Index`... - The values stored are integers, representing discrete positions along their respective dimension.
    You can think of the values stored as pixel locations.
- ...`Set` - The locations stored are unique; each row appears at most once.

Another property of the `loc` array is that it is always lexicographically sorted. 
This allows fast computations, for example set-operations: with complexity *O(n ndim)*.

Locations are always stored as 32 bit integers.
Indexes outside of the range [-2147483648, 2147483647] are not supported.

Finally, instances of `IndexSet` should be treated as immutable.
Although allowed through language mechanics, you should never assign to values of an `IndexSet` (particularly `loc`). Per, python convention, protected members usually have a prepended underscore (`_`), but in some cases references to un-copied internals (such as `loc`, `bounds`) are exposed. You should *NEVER EVER* assign to these values.

## `IndexSet` Properties
Before we go over how to build an `IndexSet`, let's go over some of the properties that will help you get a sense of exactly what indices are contained in your `IndexSet`.

### `.loc`
2-d array of locations. `loc.shape[0]` is the number of locations and `loc.shape[1]` is the number of dimensions.

### `.n`
The number of locations in the `IndexSet`. This shorthand for `.loc.shape[0]`.

### `.ndim`
The number of dimensions in the `IndexSet`.
This shorthand for `.loc.shape[1]`

### `.bounds`
The inclusive lower and upper bounds of each dimension. For each dimension, this gives the . For the above example 7x3 array, `bounds` would be:
```python
array([[1, 3],
       [1, 4],
       [2, 7]], dtype=int32)
```
Each row corresponds to each dimension of the `IndexSet`, giving the minimum and maximum values along that dimension.

## Creating an `IndexSet`
There are a few ways to create an `IndexSet`:
1. From a boolean image.
2. Manually, by specifying the coordinates that should be represented
3. By combining or operating on existing `IndexSet`s.
4. From a floating point image.

We will only talk about the first two ways right now. See sections [2. Set Operations](2_setops.md) and [3. Associated Data](3_data.md) more information on the last points.

While the examples below show instances with 2 dimensions, the `IndexSet` can handle an arbitrarily high number of dimensions.

### 1. From an image
```python
img = np.array([[0,0,0,1],
                [0,1,1,0],
                [1,0,0,0],
                [0,0,0,1]])
```
```python
>>>index_set = blahb.where(img)
>>>index_set
<numba.jitclass.boxing.IndexSet at 0x...>

>>>index_set.loc
array([[0, 3],
       [1, 1],
       [1, 2],
       [2, 0],
       [3, 3]], dtype=int32)
```
### 2. Manually
You can create an `IndexSet` by manually passing in the `loc` array. Remember: the `loc` array must be 32 bit signed integers. It is the user's responsibility to make sure that the array is in the proper format.
```python
import blahb as bl
from blahb.flags import NO_FLAG
loc = np.array([[1, 2], [3, 2], [2, 4]], dtype=np.int32)
index_set = bl.IndexSet(loc, NO_FLAG)
```
What does the `NO_FLAG` input mean in the above snippet?
We pass in flags to tell the constructor about the properties of the location array we are passing in.
Here are the possible flags and what you are asserting when you are asserting when you use them:

- `blahb.flags.SORTED`
    - The input array is already sorted.
   This avoids a potentially costly sorting operation (remember: an invariant of the `IndexSet` is that the `.loc` array is lexicographically sorted)
- `blahb.flags.UNIQUE`
    - Each row of the input array is unique.
- `blahb.flags.CONSUME`
    The input array has no external references and we can mutate it as we like.
    If there are external references, then the sorting and other changes made to this array will be visible.
    Any external references *MUST NOT* mutate `loc` under *ANY* conditions.

These flags are bit-flags that can be combined using the `|` operator. For example:
```
IndexSet(loc, blahb.flags.UNIQUE | blahb.flags.CONSUME)
```
tells the constructor that the input `loc` array is already unique and that there are no external references. The lack of a `SORTED` flag means that sorting must still be done, but we can do it in place (i.e. by mutating `loc`) because `CONSUME` is set.

*WARNING* - Do not use any of these flags unless you are sure that the conditions are met.
These requirements are class invariants and essentially every computation you do will be incorrect if these invariants are violated.
In many cases, having to use these flags for optimizations can be avoided by using other library methods.
Consider the following code:
```python
loc = np.where(img)  # We know this result is already lex-sorted and unique 
loc = np.vstack(loc).T.astype(np.int32)
index_set = bl.IndexSet(loc, bl.flags.SORTED | bl.flags.UNIQUE)
```
Instead, you should use `blahb.where`, which already uses the `SORTED | UNIQUE` flag to speed up `IndexSet` creation.

*DOUBLE WARNING* - Do no assign a value to any of these flags. If you want a safer way to set flags use the flag functions.
 These are functions with no arguments that return the similarly named flag:
 ```
 bl.flags.SORTED | bl.flags.UNIQUE
 ```
 becomes:
 ```
 bl.flags.sorted_() | bl.flags.unique_()
 ```
 They can also be found in the `blahb.flags` module.
 
#### Examples:
##### Ex. 1 - Simple `IndexSet` Creation
```python
>>>loc = np.array([[3, 2], [0, 3], [1, 2], [1, 2]], dtype=np.int32)
>>>i = bl.IndexSet(loc, NO_FLAGS)
>>>i.loc
array([[0, 3],
       [1, 2],
       [3, 2]], dtype=int32)
```
Notice how the `loc` array has been made unique (a row of `[1, 2]` has been removed) and sorted.

Because we did not set any flags (specifically the `CONSUME` flag), the input `loc` remains unchanged:
```python
>>>loc
array([[3, 2],
       [0, 3],
       [1, 2],
       [1, 2]], dtype=int32)
```

##### Ex. 2
```python
>>>external_loc = np.array([[3, 2], [0, 3], [1, 2]], dtype=np.int32)
>>>external_loc
array([[3, 2],
       [0, 3],,
       [1, 2]], dtype=int32)

>>>i = bl.IndexSet(external_loc, bl.flags.UNIQUE | bl.flags.CONSUME)
>>>i.loc
array([[0, 3],
       [1, 2],
       [3, 2]], dtype=int32)

>>>external_loc
array([[0, 3],
       [1, 2],,
       [3, 2]], dtype=int32)
```
In this example we see how a reference (`external_loc`) can see changes (namely sorting) that was done to `loc` because `CONSUME` was set.

This can lead to a dangerous scenario where you forget that `external_loc` refers to `i.loc` and you accidentally mutate it:
```python
# DON`T DO!
external_loc[1, 0] += 2
```
Now the `IndexSet` `i` is in an invalid state (it is no longer sorted) and the operations that you perform on `i` may fail silently, giving you the wrong answer. Be careful to consider all the ways in which you may invalidate `i.loc`. When in doubt, ensure safety with by using `NO_FLAGS`.

## Methods on an `IndexSet`
There are various methods

### `.take(positions)`
Take locations at the given position in the array.
#### With a `slice`
#### With a `range`
#### With a `int`
#### With an `array`

---
### `.sel(dim, criterion)`
Take locations have a matching value along a given dimension

#### With a `slice`
#### With a `range`
#### With a `int`
#### With an `array`

---
### `.omit(dim, criterion)`
Take locations have a non-matching value along a given dimension

---
### `.find_loc(coord)`
Return the position of a given location and whether or not it is contained in the object.

---
### `.split(location)`
Split into two `IndexSets` at the given position. 

---
### `.copy()`
Make a deep copy of an IndexSet.
All locations an associated data will be copied.