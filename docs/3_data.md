- [0. Motivation](index.md)
- [1. Getting Started](1_intro.md)
- [2. Set Operations](2_setops.md)
- (3) Associated Data

# Data
`blahb` offers the possibility to associate floating point data with each row of an `IndexSet`.
This data will "follow" the coordinate rows it is associated with, propagating through the slicing and dicing that are performed `IndexSet`.

All data is stored in the form of 32 bit floating point "rows" and can be accessed thropugh the `.data` attribute of an `IndexSet`.

```python
>>>a = bl.IndexSet(np.array([[1, 2], [3, 4], [5, 6]]), bl.flags.NO_FLAGS)
>>>print(a.data)
None
```
By default, when there is no data associated with an `IndexSet` it has the value of `None`.

To assign data to each row, simply assign to the data attribute of an `IndexSet`:
```python
>>>a.data = np.array([[1,2,3], [4,5,6], [7,8,9]], dtype=np.float32)
>>>a.data
array([[ 1.,  2.,  3.],
       [ 4.,  5.,  6.],
       [ 7.,  8.,  9.]], dtype=float32)
```
Here we assigned a 2d array of data to the data attribute.
There are two things to note here:
1. The number of rows of data must exactly match the number of rows in the `IndexSet`
2. The data can have any number of columns. Specific operations (such as set operations) may require that the number of data columns in each object must match.

Now when we perform operations on this object the data will propagate:
```python
>>>b = a.take(np.array([0, 2]))
>>>b.loc
array([[1, 2],
       [5, 6]], dtype=int32)
>>>b.data
array([[ 1.,  2.,  3.],
       [ 7.,  8.,  9.]], dtype=float32)
```

Similar propagation occurs when we `sel` and `omit` data.
Here we have selected data that has a value
```python
c = a.sel(3, slice()).fin()
```

Call `.reset_data()` to erase the data in an `IndexSet` (i.e. reset it back to `None`).

Because we are using floating point data, we can use `NaN` to represent rows for which associated data are missing.
When `.data` is `None` it is conceptually identical to having a `data` matrix of all `NaN`.

### Set Operations
One important class of operations where we need to consistently treat associated data are set operations.
Here there may be many different objects, each with associated data.

In this case we let the user decide how associated data is chosen from the contributing `IndexSet`s.

Illustrated below are two `IndexSet`s: `A` and `B`.
For simplicity, each `IndexSet` has one dimension of coordinates (in `[]`), and a single column of data values (shown here as decimals).
The rows are arranged in lexicographical order (as they are stored internally) and aligned so that matching coordinates from each row are on the same line.
```
   A            B
[1] 0.9      [1] 0.8
[2] NaN      [2] 0.7
   -         [3] 0.6
[4] 0.2      [4] NaN
[5] 0.4         -
   -         [6] 0.5
```

We know what the resulting coordinates of `C` should look like when we find the `union` of `A` and `B`, but finding the resulting data is more complicated:
```
   A            B       =>     C
[1] 0.9      [1] 0.8        [1] ?
[2] NaN      [2] 0.7        [2] ?
   -         [3] 0.6        [3] ?
[4] 0.2      [4] NaN        [4] ?
[5] 0.4         -           [5] ?
   -         [6] 0.5        [6] ?
```
For example, at coordinate `[1]` should we take the associated data value from `A` (0.9), or from `B` (0.8)?

This is resolved with DATA FLAGS.