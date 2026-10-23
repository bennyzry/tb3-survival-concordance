"""Rank queries for concordance, with NumPy's exact floating tie arithmetic."""

import ctypes
from pathlib import Path

import numpy as np


_library = ctypes.CDLL(str(Path(__file__).with_name("_concordance_counts.so")))
_count = _library.concordance_counts
_count.argtypes = [ctypes.c_int64] * 3 + [ctypes.c_void_p] * 11
_count.restype = ctypes.c_int


def _tie_bounds(values, tied_tol):
    """Find each risk's tie interval, without rounding risk +/- tolerance.

    Subtraction is monotone on each side of a finite floating risk.  Searching
    with the actual subtraction and comparison preserves dtype rounding and
    inclusive tolerance boundaries, even when ties are not transitive.
    """
    size = len(values)
    ranks = np.arange(size, dtype=np.int64)
    # These searches can examine risks that are not comparable.  Arithmetic
    # warnings are instead generated from actual comparison partners below.
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        if not (np.zeros(1, dtype=values.dtype) <= tied_tol)[0]:
            return ranks, ranks
        if (np.full(1, np.inf, dtype=values.dtype) <= tied_tol)[0]:
            return np.zeros(size, dtype=np.int64), np.full(size, size, dtype=np.int64)

        lo = np.zeros(size, dtype=np.int64)
        hi = ranks.copy()
        while np.any(lo < hi):
            mid = (lo + hi) // 2
            tied = np.absolute(values[mid] - values) <= tied_tol
            hi = np.where(tied, mid, hi)
            lo = np.where(tied, lo, mid + 1)
        lower = lo

        lo = ranks.copy()
        hi = np.full(size, size, dtype=np.int64)
        while np.any(lo < hi):
            active = lo < hi
            mid = (lo + hi) // 2
            tied = np.absolute(values[np.minimum(mid, size - 1)] - values) <= tied_tol
            lo = np.where(active & tied, mid + 1, lo)
            hi = np.where(active & ~tied, mid, hi)
        return lower, lo


def count_pairs(estimate, order, events, starts, tied_tol):
    values, ranks = np.unique(estimate, return_inverse=True)
    ranks = np.ascontiguousarray(ranks[order], dtype=np.int64)
    lower, upper = _tie_bounds(values, tied_tol)
    counts = np.zeros((5, len(order)), dtype=np.int64)
    tied_time = np.zeros(1, dtype=np.int64)
    result = _count(
        len(order), len(values), len(starts),
        ranks.ctypes.data, events.ctypes.data, starts.ctypes.data,
        lower.ctypes.data, upper.ctypes.data,
        *(row.ctypes.data for row in counts), tied_time.ctypes.data,
    )
    if result:
        raise MemoryError("could not allocate concordance rank tree")

    # Extremal partners detect exactly whether any comparable subtraction
    # overflows.  Evaluating the comparison also retains warnings caused by
    # conversion of a Python tolerance to the risk dtype (including empty
    # comparison arrays).
    has_partners = events & (counts[2] != 0)
    event_risk = estimate[order[has_partners]]
    np.absolute(values[counts[3, has_partners]] - event_risk) <= tied_tol
    np.absolute(values[counts[4, has_partners]] - event_risk) <= tied_tol
    return counts[:3], tied_time[0]
