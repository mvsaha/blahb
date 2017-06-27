- [0. Motivation](index.md)
- (1) Getting Started
- [2. Set Operations](2_setops.md)
- [3. Associated Data](3_data.md)

# 1. Getting Started: The `IndexSet`

The basic unit of `blahb` is the `IndexSet`.
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
Although allowed through language mechanics, you should never assign to values of an `IndexSet` (particularly `loc`).
Per, python convention, protected members usually have a prepended underscore (`_`), but in some cases references to un-copied internals (such as `loc`, `bounds`) are exposed.
You should *NEVER EVER* assign to these values.

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
You can create an `IndexSet` by manually passing in the `loc` array. Remember: the `loc` array must be 32 bit signed integers.
It is the user's responsibility to make sure that the array is in the proper format.
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
    - The input array has no external references and we can mutate it as we like.
    If there are external references, then the sorting and other changes made to this array will be visible.
    Any external references *MUST NOT* mutate `loc` under *ANY* conditions.

These flags are bit-flags that can be combined using the `|` operator. For example:
```
IndexSet(loc, blahb.flags.UNIQUE | blahb.flags.CONSUME)
```
tells the constructor that the input `loc` array is already unique and that there are no external references.
The lack of a `SORTED` flag means that sorting must still be done, but we can do it in place (i.e. by mutating `loc`) because `CONSUME` is set.

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
Now the `IndexSet` `i` is in an invalid state (it is no longer sorted) and the operations that you perform on `i` will fail silently, giving you the wrong answer.
Be careful to consider all the ways in which you may invalidate `i.loc`.
When in doubt, ensure safety by using `NO_FLAGS`.

## Methods on an `IndexSet`
Here we list various methods on an `IndexSet`.

### `.take(positions)`
Take rows at the given positions in `loc`, returning an `IndexSet` with those rows.

Conceptually, you should think of `take` as a you would `[]`/`__getitem__` access (`numba` does not yet support magic methods on `jitclass`es).

The `positions` argument can have a few different forms:
##### With a `slice`
```python
>>>loc = np.array([[1, 2],
                   [3, 1],
                   [4, 0],
                   [4, 2],
                   [4, 3],
                   [5, 0]], dtype=np.int32)
>>>a = bl.IndexSet(loc, bl.flags.NO_FLAGS)
>>>b = a.take(slice(3, None))
>>>b
<numba.jitclass.boxing.IndexSet at 0x...>

>>>b.loc
array([[4, 2],
       [4, 3],
       [5, 0]], dtype=int32)
```

##### With a range of values
This is identical to using a `slice`, but instead of building a `slice` object, you can pass in a tuple of values giving the lower and upper bounds:
```python
>>>b = a.take((2, 7))  # Same `a` as above
>>>b.loc
array([[4, 0],
       [4, 2],
       [4, 3],
       [5, 0]], dtype=int32)
```

##### With an `int`
This selects a single row from an `IndexSet`.
```python
>>>a.take(2).loc
array([[3, 1]], dtype=int32)
```
This method raises an `IndexError` if the integer index is out of bounds.

##### With an `array`
This extracts specific rows at the specified indices.
```python
>>>a.take(np.array([0, 2, 3])).loc  # Extract the first, third and fourth rows
array([[1, 2],
       [4, 0],
       [4, 2]], dtype=int32)
```

If a `slice`, range, or `int` is used to identify postions along the first dimension, then this will create a view into the array being taken from.
This can be much faster than copying the rows into a new `loc` array.
If you do not want your `loc` to share memory (for instance if a ref-counted reference is keeping a large, un-needed parent alive) then you can call `copy()` on an `IndexSet` to sever any references.

---
### `.sel(dimension, criterion)`
Select locations have a matching value along a given dimension.

The general idea of `.sel` is this:

*The first time you call `.sel` on an `IndexSet`, an object of the type `SelResult` is returned.
You can then `.sel` on this object as well, repeatedly.
When you are done `.sel`-ing call `.fin()` to finish the selection process and return a `IndexSet` with only the criterion.* 


To select values, call `.sel(dim, criterion)` where `dim` is an integer dimension number (starting at 0 for the lowest dimension) and `criterion` can have one of the following forms:
#### With a `slice`
Select values that lie within an inclusive range.

The following example selects rows having a value between 2 and 4 (inclusive) in the second dimension (dimension index of 1):
```python
>>>loc = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]], np.int32)
>>>a = bl.IndexSet(loc, bl.flags.NO_FLAG)
>>>b = a.sel(1, slice(2, 4))
>>>b  # sel-ing returns an intermediate SelResult object
<numba.jitclass.boxing.SelResult at 0x...>
>>>c = b.fin()  # Call .fin to finish the selection process
>>>c.loc
array([[1, 2],
       [2, 3],
       [3, 4]], dtype=int32)
```

#### With a range of values
As with `.take`, this you can pass in a tuple of (low, high) values instead of passing a `slice` object.
```python
a.sel(0, (2, 4))
```

#### With a `int`
Selects rows that are equal to a value along a given dimension:
```python
a.sel(1, 4)
```

#### With an `array`
Selects rows that have values in a given array along a given dimension:
```python
a.sel(1, np.array([1, 3, 5]))
```

#### Example
An example for clarification. Assume we have the following `IndexSet` `a`:
```python
>>>loc = np.array([[-2, 1],
                   [-3, 3],
                   [ 2, 7],
                   [ 4, 5],
                   [ 3, 4],
                   [ 5, 7]], dtype=np.int32)
>>>a = bl.IndexSet(loc, bl.flags.NO_FLAG)
```
And we want to select all values in the inclusive range [0, 4] along the first dimension and any value in the array [1, 3, 5, 7] in the second. 
```python
>>>b = a.sel(0, (0, 4)).sel(2, np.array([1, 3, 5, 7])).fin()
>>>b.loc
array([[2, 7],
       [4, 5]], dtype=int32)
```

What if, instead of just building an `IndexSet` of the selected values, we want to partition the array into an `IndexSet` of matching values and an `IndexSet` of all other values? We can use `.split()` instead of `.fin()` on the final `SelResult` object and it will return a tuple of `IndexSet`s: the first containing rows that matched all of the criteria, and a second containing rows that did not match all of the criteria.


Why bother with the intermediate `SelResult` and `.fin()`/`.split()`?
This way we can iteratively narrow the scope of the query before actually extracting the values we want.
This greatly enhances the speed of multi-criteria queries.

*Performance Tips:*
- Always, *always* `.sel` in order of increasing dimensions if you have multiple criteria on different dimensions.
Selection is a linear time operation on all dimensions but the first, but roughly O(log n) in the first dimension because the first column of `.loc` is sorted ascending due to its lexicographical ordering (this is a class invariant).
- Selecting values using an array criterion is slow even on the lowest dimension.
If you can formulate your problem to avoid array-based selection you should.

---
### `.omit(dimension, criterion)`
Take locations have a non-matching value along a given dimension.

This is similar to `sel`, except the result *doesn't* contain any rows/dimensions matching the criteria.
The possible criteria are the same as in `.sel`:
- `slice` of (low, high) values.
- (2-`tuple`) of (low, high) values.
- An `int` for matching a specific value.
- An `np.array` containing a list of values to omit.

Pro Tip: You chain `.sel` and `.omit` in a single query.
For example, if we want all the rows with first-dimension value of 20, with second-dimension values not equal to 3, and last-dimension values in the set {1, 3, 4}:
```python
a.sel(0, 20).omit(1, 3).sel(2, np.array([1, 3, 4])).fin()
```


---
### `.copy()`
Make a deep copy of an `IndexSet`.
All locations and associated data will be copied.
This will sever any external references that an `IndexSet` has.


---
### `.find_loc(coord)`
Return the position of a given row and whether or not it is contained in the object.

Returns a 2-tuple containing:
 1. Boolean indicating whether of not the `coord` is in the `IndexSet`
 2. The lowest index where the coordinate would be inserted into the rows of the `IndexSet` to maintain lexicographical ordering.

```python
>>>a = bl.IndexSet(np.array([[1, 2], [3, 4]), bl.flags.NO_FLAGS)
>>>a.find_loc((3, 4))
(True, 1)
>>>a.find_loc((1, 0))
(False, 0)
```

---
### `.split(location)`
Split into two `IndexSets` at the given position.

---
### `.split_at_coord(coord)`
Split into two `IndexSets`, the first containing all rows lexicographically lower than `coord` and the second containing all rows lexicographically greater than or equal to `coord`.
