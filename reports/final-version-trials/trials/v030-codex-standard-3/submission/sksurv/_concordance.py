"""O(n log n) concordance counting for double-precision risk estimates.

The native helper only counts pairs. Validation, censoring weights, floating
point accumulation, and public exceptions retain the metrics module's NumPy
semantics. The helper has no Python or NumPy binary-ABI dependency.
"""

import ctypes
from pathlib import Path

import numpy as np

from .exceptions import NoComparablePairException


_library = ctypes.CDLL(str(Path(__file__).with_name("_concordance_native.so")))
_count = _library.concordance_counts
_index_array = np.ctypeslib.ndpointer(dtype=np.intp, ndim=1, flags="C_CONTIGUOUS")
_count_array = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
_double_array = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
_bool_array = np.ctypeslib.ndpointer(dtype=np.bool_, ndim=1, flags="C_CONTIGUOUS")
_count.argtypes = [
    ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.c_ssize_t,
    _index_array, _index_array, _bool_array,
    _double_array, ctypes.c_double,
    _count_array, _count_array, _count_array, _count_array,
]
_count.restype = ctypes.c_int


def estimate_concordance_index(event_indicator, event_time, estimate, weights, tied_tol):
    order = np.argsort(event_time)
    times = event_time[order]
    events = np.ascontiguousarray(event_indicator[order])
    n = len(order)

    # _iter_comparable does not visit a final singleton. A final group with
    # several events, however, is visited even if it has no comparable pairs:
    # that case returns NaN with a RuntimeWarning rather than raising.
    visited_events = events.copy()
    if n and (n == 1 or times[-1] != times[-2]):
        visited_events[-1] = False
    if not visited_events.any():
        raise NoComparablePairException("Data has no comparable pairs, cannot estimate concordance index.")

    ends = np.concatenate((np.flatnonzero(times[1:] != times[:-1]) + 1, [n])).astype(np.intp, copy=False)
    values, ranks = np.unique(estimate, return_inverse=True)
    ranks = np.ascontiguousarray(ranks[order], dtype=np.intp)
    con = np.empty(n, dtype=np.int64)
    ties = np.empty(n, dtype=np.int64)
    comparable = np.empty(n, dtype=np.int64)
    info = np.zeros(2, dtype=np.int64)

    status = _count(
        n, len(values), len(ends), ranks, ends, events,
        values, tied_tol, con, ties, comparable, info,
    )
    if status:
        raise MemoryError("Unable to allocate concordance counting workspace")

    if info[1]:
        # The native comparisons can overflow just as NumPy subtraction can.
        # Issue that operation through NumPy to honor its warning/error policy.
        np.subtract(estimate.max(), estimate.min())

    con = con[visited_events]
    ties = ties[visited_events]
    comparable = comparable[visited_events]
    w = weights[order[visited_events]]

    # cumsum, unlike sum's pairwise reduction, preserves the release's order
    # of accumulation. Separate operations also preserve its rounding and
    # warnings for extreme weights and zero denominators.
    numerator = np.cumsum(w * con + 0.5 * w * ties)[-1]
    denominator = np.cumsum(w * comparable)[-1]
    cindex = numerator / denominator
    concordant = con.sum(dtype=np.int64)
    tied_risk = ties.sum(dtype=np.int64)
    discordant = comparable.sum(dtype=np.int64) - concordant - tied_risk
    return cindex, concordant, discordant, tied_risk, info[0]
