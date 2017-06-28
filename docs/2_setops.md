- [0. Motivation](index.md)
- [1. Getting Started](1_intro.md)
- (2) Set Operations
- [3. Associated Data](3_data.md)

# Set Operations
This chapter reviews some of the ways in which you can combine and compare `IndexSet` objects:
- `intersection`
- `union`
- `symmetric_difference`
- `asymmetric_difference`

One of the use cases for `blahb` is to be able to create, manipulate and call functions on `IndexSet`s from within `nopython=True` code blocks.
After all, `IndexSet` is a `numba` `jitclass`.
For this reason, all of the set operations listed above have two versions:
- Regular version
  - No underscore after name (ex: `blahb.union`)
  - Cannot be called in a `nopython` block
  - Takes as arguments:
    - An iterable of `IndexSet` objects (`union`, `intersection`), or 
    - Two `IndexSet` objects (`symmetric_difference`, `asymmetric_difference`)
  - Easily deployed over multiple cores by setting the `n_workers` keyword argument
- Nopython version
  - Underscore after name (ex: `blahb.union_`)
  - Can be called in a `nopython` block
  - Takes two `IndexSet` objects and a required uint8 merge flag array as arguments
  - No parallelization

*NOTE:* In general, when you see a `blahb` function name with an appended underscore (`_`) it means that the function is decorated with `nopython=True` and `nogil=True` and you can call it in `nopython` mode.

## Intersection
Create an `IndexSet` containing the rows found in every input.

#### `intersection`
#### `intersection_`


## Union
Create an `IndexSet` containing the rows found in any of the inputs.

#### `union`
#### `union_`

## Asymmetric Difference
Create an `IndexSet` containing the rows found in the first argument that are *not* present in the second. 

#### `asymmetric_difference`
#### `asymmetric_difference_`

## Symmetric Difference
Create `IndexSet` containing the rows that exist in exactly one of the inputs.

#### `symmetric_difference`
#### `symmetric_difference_`