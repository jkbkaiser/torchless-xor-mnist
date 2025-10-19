# IDX File Format (MNIST)

The IDX format is a **binary format** for storing vectors and multidimensional arrays.  
It is used for datasets like MNIST (images and labels).

---

## File Structure

1. **Magic Number (4 bytes)**
   - The first 2 bytes are always `0`.
   - The 3rd byte indicates **data type**:
     | Value | Type       |
     |-------|------------|
     | 0x08  | unsigned byte |
     | 0x09  | signed byte   |
     | 0x0B  | short (2 bytes) |
     | 0x0C  | int (4 bytes) |
     | 0x0D  | float (4 bytes) |
     | 0x0E  | double (8 bytes) |
   - The 4th byte is the **number of dimensions**.

2. **Dimension Sizes**
   - 4 bytes per dimension (big-endian).
   - Examples:
     - **Images (idx3)**: `[num_images, rows, cols]`
     - **Labels (idx1)**: `[num_labels]`

3. **Data**
   - Stored in **row-major order**.
   - **Images**: unsigned bytes (0–255) for pixel values.  
   - **Labels**: unsigned bytes (0–9) for class labels.

---

## Example (MNIST)

| File                         | Magic Number | Dimensions                  | Data Type      |
|-------------------------------|--------------|----------------------------|----------------|
| `train-images-idx3-ubyte`    | 0x00000803   | `[60000, 28, 28]`          | unsigned byte |
| `train-labels-idx1-ubyte`    | 0x00000801   | `[60000]`                   | unsigned byte |
| `t10k-images-idx3-ubyte`     | 0x00000803   | `[10000, 28, 28]`          | unsigned byte |
| `t10k-labels-idx1-ubyte`     | 0x00000801   | `[10000]`                   | unsigned byte |

---

### Notes

- All integers in the header are stored in **big-endian** format.
- `idx3` → 3-dimensional array (images)  
- `idx1` → 1-dimensional array (labels)
- Simple and efficient format, primarily for MNIST and similar datasets.

---

### References

- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [IDX File Format Documentation](http://yann.lecun.com/exdb/mnist/)

