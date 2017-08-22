import numpy as np
import numba

"""
The idea of this compression is to pack multiple dimensions that each require
less than 32 bits of information to represent into the _loc array of an
IndexSet.

The compression should be structured such that a lexicographical comparion
between two compressed rows should return the same value as a lexicographical
comparison between the uncompressed versions of the rows. This allows us to
perform set operations on compressed IndexSets while avoiding potentially
costly unpacking.

In this module, the specification for how many bits each dimension gets
is given by an encoding, a 1d array with a single signed integer for each
dimension (column) of IndexSet.loc. If the value along a given dimension, `e`,
is positive, this asserts that all values in that corresponding dimension are
positive, and all are in the half-open range:
    [0, 2 ** e).
If `e` is negative, this asserts that all of the values lie in the range:
    [ -(2 ** abs(e)), 2 ** abs(e) )
(and therefore requires abs(e) + 1 bits of representational power).

Once encoded, a bit-packed array is stored in the IndexSet as the int32 `loc`
array as in a uncompressed IndexSet. However, the bit order must be
interpreted as an array of uint32 (accessed through array.view(np.uint32)) for
the purposes of lexicographical comparison.

The utilities below actually support encoding/decoding numbers >32 bits (i.e.
storing values >32 bits acrossed multiple columns of a packed 32 bit columns)
but the IndexSet is currently restricted to int32 indices, so this doesn't
come into play.
"""


@numba.njit
def bits_required(lo, hi):
    """Find the number of bits needed to represent all values in a range.

    Arguments
    ---------
    lo, hi : int
        The lower and upper (inclusive) range of values that we want to
        represent.

    Returns
    -------
    n_bits : signed int
    The number of bits needed to represent all of the values in the range
    if both `lo` and `hi` are >= 0.
    If `lo` is negative, the negation of the maximum number of bits that are
    needed to represent the positive range is returned.
        [ 0,  max(abs(lo), abs(hi)) )
    """
    if abs(lo) <= abs(hi):
        n_bits = int(np.ceil(np.log2(abs(hi) + 1)))
    else:
        n_bits = int(np.ceil(np.log2(abs(lo))))
    
    if lo < 0:
        n_bits += 1
    
    return n_bits

@numba.njit
def bit_code(lo, hi):
    """Return the encoding value that is needed to represent a range of values.
    
    These should be used as the integer values for the encoding array passed
    into IndexSet.encode().
    """
    b = bits_required(lo, hi)
    if lo < 0 or hi < 0:  # Need a negative representation
        return -(b - 1)
    return b

@numba.njit
def min_encoded_value(n_bits):
    """The minimum value encoded by a signed bit-width."""
    return 0 if n_bits >= 0 else -(2 ** (abs(n_bits)))


@numba.njit
def bits_needed(encoding):
    """The number of 32 bit columns needed for an encoding."""
    n_bits = 0
    for e in encoding:
        n_bits += e if e >= 0 else abs(e) + 1
    return n_bits


@numba.njit
def packed_columns_needed(encoding):
    n_bits_total = bits_needed(encoding)
    n_ints_needed = n_bits_total // 32
    if n_bits_total % 32:
        n_ints_needed += 1
    return n_ints_needed


@numba.njit
def drop_lsb_bits(val, n_bits):
    """Drop the lowest bits in a number."""
    return (val >> n_bits) << n_bits


@numba.njit
def keep_lsb_bits(val, bits_to_keep):
    """Unset all bits but the lowest."""
    return val & ~drop_lsb_bits(val, bits_to_keep)


@numba.njit
def write_encoded(vals, encoding, packed, start):
    """Pack values of a certain number of bits into an encoded array.

    Arguments
    ---------
    vals : 1d array
        This will be mutated
    encoding : int
        The
    packed : 2d int
        The compressed array
    start : int
        The number of bits already written to packed, with MSBs of
        each column of packed filling up first.
    """
    
    # Number of bits in each column of compressed
    column_bit_capacity = 32
    
    n = vals.size
    min_val = min_encoded_value(encoding)
    
    # Need an extra bit to encode negative
    val_bit_width = encoding if encoding >= 0 else abs(encoding) + 1
    
    bits_to_write = val_bit_width  #
    total_bits_written = 0
    
    while bits_to_write:
        
        column = start // column_bit_capacity  # The column index to write to
        column_bit_start = start - (column * column_bit_capacity)  #
        
        bits_left_in_column = column_bit_capacity - column_bit_start
        bits_taken = min(bits_left_in_column, bits_to_write)
        
        # The number of bits that we will be left in the
        n_lsb_bits_dropped = bits_to_write - bits_taken
        
        # The LSB of the bit range to be written
        shift = (column_bit_capacity - column_bit_start
                 - bits_taken - n_lsb_bits_dropped)
        
        if shift == 0:
            for i in range(n):
                # Take only the first `bits_taken` higher bits
                val_to_write = drop_lsb_bits(
                    vals[i] - min_val, n_lsb_bits_dropped)
                val_to_write = keep_lsb_bits(val_to_write, val_bit_width)
                packed[i, column] |= np.uint32(val_to_write)
        elif shift < 0:
            shift = abs(shift)
            for i in range(n):
                # Take only the first `bits_taken` higher bits
                val_to_write = drop_lsb_bits(
                    vals[i] - min_val, n_lsb_bits_dropped)
                val_to_write = keep_lsb_bits(val_to_write, val_bit_width)
                packed[i, column] |= np.uint32(val_to_write >> shift)
        else:
            for i in range(n):
                # Take only the first `bits_taken` higher bits
                val_to_write = drop_lsb_bits(
                    vals[i] - min_val, n_lsb_bits_dropped)
                val_to_write = keep_lsb_bits(val_to_write, val_bit_width)
                packed[i, column] |= np.uint32(val_to_write << shift)
        
        total_bits_written += bits_taken
        start += bits_taken
        bits_to_write -= bits_taken
    
    assert total_bits_written == val_bit_width
    assert bits_to_write == 0
    return start


@numba.njit
def read_encoded(packed, encoding, vals, start):
    """Unpack values from an encoded bit-packed array."""
    column_bit_capacity = 32
    n = packed.shape[0]
    min_val = min_encoded_value(encoding)
    val_bit_width = encoding if encoding >= 0 else abs(encoding) + 1
    bits_to_read = val_bit_width  #
    total_bits_read = 0
    vals_unsigned = vals.view(np.uint32)
    vals_signed = vals.view(np.int32)
    while bits_to_read:
        column = start // column_bit_capacity
        column_bit_start = start - (column * column_bit_capacity)  #
        
        bits_left_in_column = column_bit_capacity - column_bit_start
        n_bits_read = min(bits_left_in_column, bits_to_read)
        
        # The number of bits that we will not be read
        n_lsb_bits_dropped = bits_to_read - n_bits_read
        
        # The LSB location of the bit range to be read
        shift = (column_bit_capacity - column_bit_start
                 - n_bits_read - n_lsb_bits_dropped)
        
        if shift == 0:
            for i in range(n):
                val = keep_lsb_bits(packed[i, column], val_bit_width)
                vals_unsigned[i] |= val
        elif shift < 0:
            shift = abs(shift)
            for i in range(n):
                val = keep_lsb_bits(packed[i, column] << shift, val_bit_width)
                vals_unsigned[i] |= val
        else:
            for i in range(n):
                val = keep_lsb_bits(packed[i, column] >> shift, val_bit_width)
                vals_unsigned[i] |= val
        
        start += n_bits_read
        bits_to_read -= n_bits_read
    
    for i in range(n):
        vals_signed[i] = np.int32(vals_unsigned[i] + min_val)
    return start


@numba.njit
def encode(loc, encoding):
    """Encode locations into bit-packed array.

    Arguments
    ---------
    loc : 2d int32 array
    encoding : 1d array
        Should have a shape matching loc.shape[1], the dimensionality.
        Negative values in this array indicate that negative values
        should be considered as well.

    Returns
    -------
    packed : int32 array
        Contains the locations in `loc`, compressed so that
        lexicographical comparisons between rows are still valid.
    """
    if not (encoding.ndim == 1 and encoding.size == loc.shape[1]):
        raise ValueError("`encoding` must be 1d and have size equal to "
                         "dimensions of `loc`.")
    n, ndim = loc.shape[0], packed_columns_needed(encoding)
    packed = np.zeros((n, ndim), dtype=np.uint32)
    start = 0
    for dim in range(loc.shape[1]):
        start = write_encoded(loc[:, dim], encoding[dim], packed, start)
    return packed


@numba.njit
def read_encoded(packed, encoding, vals, start):
    """Unpack values from an encoded bit-packed array.
    
    Arguments
    ---------
    packed : 2d array
        A n-by-ndim array containing bit-packed 32 bit unsigned integers
    encoding : 1d array
    vals : 1d array
        The array to write the outputs to
    start : int
        The start bit from which to read (from the first LSB in the array)
    """
    column_bit_capacity = 32
    n = packed.shape[0]
    min_val = min_encoded_value(encoding)
    val_bit_width = encoding if encoding >= 0 else abs(encoding) + 1
    bits_to_read = val_bit_width  #
    total_bits_read = 0
    vals_unsigned = vals.view(np.uint32)
    vals_signed = vals.view(np.int32)
    
    while bits_to_read:
        column = start // column_bit_capacity
        column_bit_start = start - (column * column_bit_capacity)  #
        
        bits_left_in_column = column_bit_capacity - column_bit_start
        n_bits_read = min(bits_left_in_column, bits_to_read)
        
        # The number of bits that we will not be read
        n_lsb_bits_dropped = bits_to_read - n_bits_read
        
        # The LSB location of the bit range to be read
        shift = (column_bit_capacity - column_bit_start
                 - n_bits_read - n_lsb_bits_dropped)
        
        if shift == 0:
            for i in range(n):
                val = keep_lsb_bits(packed[i, column], val_bit_width)
                vals_unsigned[i] |= val
        elif shift < 0:
            shift = abs(shift)
            for i in range(n):
                val = keep_lsb_bits(packed[i, column] << shift, val_bit_width)
                vals_unsigned[i] |= val
        else:
            for i in range(n):
                val = keep_lsb_bits(packed[i, column] >> shift, val_bit_width)
                vals_unsigned[i] |= val
        
        start += n_bits_read
        bits_to_read -= n_bits_read
        total_bits_read += n_bits_read
    
    assert total_bits_read == val_bit_width
    assert bits_to_read == 0
    
    for i in range(n):
        vals_signed[i] = np.int32(vals_unsigned[i] + min_val)
    return start


@numba.njit
def decode(packed, encoding):
    loc = np.zeros((packed.shape[0], encoding.size), dtype=np.int32)
    start = 0
    for dim in range(encoding.size):
        start = read_encoded(packed, encoding[dim], loc[:, dim], start)
    return loc


@numba.njit
def compatible_encoding(a, b):
    """Test whether two IndexSets have compatible encoding.
    
    Arguments
    ---------
    a, b: IndexSet objects
    
    Returns
    -------
    True if the encoding arrays for `a` and `b` are identical. Returns False
    if either `a` or `b` are not encoded.
    """
    if (a.encoding is None or b.encoding is None or
      a.encoding.size != b.encoding.size):
        return False
    
    a_enc = a.encoding.view(np.int8)
    b_enc = b.encoding.view(np.int8)
    for i in range(a_enc.size):
        if a_enc[i] != b_enc[i]:
            return False
    return True


@numba.njit
def same_encoding(a, b):
    if a.encoding is None and b.encoding is None:
        return True
    elif a.encoding is None or b.encoding is None:
        return False
    else:
        if not a.encoding.size == b.encoding.size:
            return False
        
        for i in range(a.encoding.size):
            if a.encoding[i] != b.encoding[i]:
                return False
        return True