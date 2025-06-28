import os
import numpy as np
import pandas as pd
from numpy.f2py.auxfuncs import throw_error


def validate_csvs(
        orig_path: str,
        dec_path: str,
        precision: str = "float32"
) -> bool:
    """
    Binary-exact comparison of two CSV files.

    Parameters
    ----------
    orig_path : str
        Path to the original CSV.
    dec_path : str
        Path to the decoded CSV.
    precision : {"float32", "float64"}, default "float32"
        Floating type to load and compare.  Any element where either
        side is NaN is treated as equal; every other element must have
        identical bit patterns.
    """
    if precision not in ("float32", "float64"):
        raise ValueError("precision must be 'float32' or 'float64'")

    dtype = precision  # "float32" or "float64"
    bit_dtype = np.uint32 if dtype == "float32" else np.uint64

    # File-existence check
    if not (os.path.exists(orig_path) and os.path.exists(dec_path)):
        print(f"-- Validation skipped: missing files at {orig_path} or {dec_path}")
        return False

    # Load with the chosen dtype
    orig = pd.read_csv(orig_path, dtype=dtype)
    dec = pd.read_csv(dec_path, dtype=dtype)

    # Shape must match
    if orig.shape != dec.shape:
        throw_error(f"-- Validation FAILED: shape mismatch (orig {orig.shape}, dec {dec.shape})")

    a, b = orig.values, dec.values

    # Positions involving any NaN are automatically "OK"
    nan_mask = np.isnan(a) | np.isnan(b)

    # View the floats as their raw integer bit patterns
    a_bits = a.view(bit_dtype)
    b_bits = b.view(bit_dtype)

    # Mismatch: bits differ and it’s not a NaN position
    mismatch = (a_bits != b_bits) & ~nan_mask

    if mismatch.any():
        i, j = np.argwhere(mismatch)[0]
        col = orig.columns[j]
        throw_error(f"-- Validation FAILED at row {i}, column '{col}': "
                    f"{orig.iat[i, j]!r} (bits {a_bits[i, j]:#x}) "
                    f"!= {dec.iat[i, j]!r} (bits {b_bits[i, j]:#x})")

    print(f"-- Validation PASSED under {precision}: all non-NaN elements match bit-for-bit.")
    return True
