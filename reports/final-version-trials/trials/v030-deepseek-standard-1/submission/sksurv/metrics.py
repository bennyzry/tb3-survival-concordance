# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_array, check_consistent_length, check_is_fitted

try:  # regular package layout
    from .exceptions import NoComparablePairException
    from .nonparametric import CensoringDistributionEstimator, SurvivalFunctionEstimator
    from .util import check_y_survival
except ImportError:  # pragma: no cover - module imported as top-level file
    from sksurv.exceptions import NoComparablePairException
    from sksurv.nonparametric import CensoringDistributionEstimator, SurvivalFunctionEstimator
    from sksurv.util import check_y_survival

__all__ = [
    "as_concordance_index_ipcw_scorer",
    "as_cumulative_dynamic_auc_scorer",
    "as_integrated_brier_score_scorer",
    "brier_score",
    "concordance_index_censored",
    "concordance_index_ipcw",
    "cumulative_dynamic_auc",
    "integrated_brier_score",
]


def _check_estimate_1d(estimate, test_time):
    estimate = check_array(estimate, ensure_2d=False, input_name="estimate")
    if estimate.ndim != 1:
        raise ValueError(f"Expected 1D array, got {estimate.ndim}D array instead:\narray={estimate}.\n")
    check_consistent_length(test_time, estimate)
    return estimate


def _check_inputs(event_indicator, event_time, estimate):
    check_consistent_length(event_indicator, event_time, estimate)
    event_indicator = check_array(event_indicator, ensure_2d=False, input_name="event_indicator")
    event_time = check_array(event_time, ensure_2d=False, input_name="event_time")
    estimate = _check_estimate_1d(estimate, event_time)

    if not np.issubdtype(event_indicator.dtype, np.bool_):
        raise ValueError(
            f"only boolean arrays are supported as class labels for survival analysis, got {event_indicator.dtype}"
        )

    if len(event_time) < 2:
        raise ValueError("Need a minimum of two samples")

    if not event_indicator.any():
        raise ValueError("All samples are censored")

    return event_indicator, event_time, estimate


def _check_times(test_time, times):
    times = check_array(np.atleast_1d(times), ensure_2d=False, input_name="times")
    times = np.unique(times)

    if times.max() >= test_time.max() or times.min() < test_time.min():
        raise ValueError(
            f"all times must be within follow-up time of test data: [{test_time.min()}; {test_time.max()}["
        )

    return times


def _check_estimate_2d(estimate, test_time, time_points, estimator):
    estimate = check_array(estimate, ensure_2d=False, allow_nd=False, input_name="estimate", estimator=estimator)
    time_points = _check_times(test_time, time_points)
    check_consistent_length(test_time, estimate)

    if estimate.ndim == 2 and estimate.shape[1] != time_points.shape[0]:
        raise ValueError(f"expected estimate with {time_points.shape[0]} columns, but got {estimate.shape[1]}")

    return estimate, time_points


def _iter_comparable(event_indicator, event_time, order):
    n_samples = len(event_time)
    tied_time = 0
    i = 0
    while i < n_samples - 1:
        time_i = event_time[order[i]]
        end = i + 1
        while end < n_samples and event_time[order[end]] == time_i:
            end += 1

        # check for tied event times
        event_at_same_time = event_indicator[order[i:end]]
        censored_at_same_time = ~event_at_same_time

        mask = np.zeros(n_samples, dtype=bool)
        mask[end:] = True
        # an event is comparable to censored samples at same time point
        mask[i:end] = censored_at_same_time

        for j in range(i, end):
            if event_indicator[order[j]]:
                tied_time += censored_at_same_time.sum()
                yield (j, mask, tied_time)
        i = end


def _estimate_concordance_index(event_indicator, event_time, estimate, weights, tied_tol=1e-8):
    """Estimate the concordance index in O(n log n) instead of O(n**2).

    Instead of iterating over all events and comparing each of them against all
    samples that are comparable to it, the comparable pairs are counted with
    order statistics.  To this end, samples are ranked by their risk score and
    the number of samples with a smaller risk score among the samples that come
    later in time is obtained from a wavelet-matrix like prefix counter.

    The comparator arithmetic (risk ties, concordant/discordant decisions,
    accumulation of the weighted sums and the returned counts) is identical to
    the reference implementation, i.e. all comparisons are carried out in the
    dtype of ``estimate``.
    """
    if not _supports_order_statistics(estimate):
        return _estimate_concordance_index_reference(event_indicator, event_time, estimate, weights, tied_tol)

    n_samples = event_time.shape[0]
    order = np.argsort(event_time)

    event_time_sorted = event_time[order]
    event_sorted = event_indicator[order]
    estimate_sorted = estimate[order]
    weights_sorted = weights[order]

    starts, ends, group_of = _sorted_time_groups(event_time_sorted)
    n_groups = starts.shape[0]

    censored = ~event_sorted
    event_positions = np.flatnonzero(event_sorted)
    # The reference implementation never visits a group that starts at the last
    # sample, hence such an event does not contribute to any count.
    if starts[-1] == n_samples - 1 and event_positions.size and event_positions[-1] == n_samples - 1:
        event_positions = event_positions[:-1]
    if event_positions.size == 0:
        raise NoComparablePairException("Data has no comparable pairs, cannot estimate concordance index.")

    event_counts = np.bincount(group_of[event_sorted], minlength=n_groups)
    censored_counts = np.bincount(group_of[censored], minlength=n_groups)

    event_groups = group_of[event_positions]
    # first sample that is observed after the event's time point
    suffix_start = ends[event_groups]
    event_risk = estimate_sorted[event_positions]
    # censored samples at the same time point are comparable, too
    n_partners = (n_samples - suffix_start) + censored_counts[event_groups]

    sorted_risks = np.sort(estimate_sorted)
    risk_order = np.argsort(estimate_sorted)
    rank_of_position = np.empty(n_samples, dtype=np.int64)
    rank_of_position[risk_order] = np.arange(n_samples, dtype=np.int64)

    n_less = np.searchsorted(sorted_risks, event_risk, side="left").astype(np.int64)

    cast_overflow = _tolerance_cast_overflows(estimate.dtype, tied_tol)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first_tie, last_tie, has_ties = _tie_bounds(sorted_risks, event_risk, n_less, tied_tol)
        k_lo = np.where(has_ties, first_tie, n_less)
        k_hi = np.where(has_ties, last_tie, n_less)

        rank_counter = _RankPrefixCounter(rank_of_position, n_samples)
        # count of samples after the event's time point with rank < k_lo / < k_hi
        after_concordant = rank_counter.count_less(suffix_start, k_lo)
        after_tied = rank_counter.count_less(suffix_start, k_hi) - after_concordant
        subtract_overflow = _subtract_overflows(
            estimate_sorted, censored, starts, event_groups, suffix_start, event_risk, n_partners
        )

    # censored samples that share the event's time point
    if censored.any():
        censored_positions = np.flatnonzero(censored)
        censored_groups = group_of[censored_positions]
        censored_ranks = rank_of_position[censored_positions]
        censored_order = np.lexsort((censored_ranks, censored_groups))
        tree = censored_ranks[censored_order] + censored_groups[censored_order] * n_samples
        group_offset = np.zeros(n_groups + 1, dtype=np.int64)
        np.cumsum(censored_counts, out=group_offset[1:])
        offset = event_groups * n_samples
        same_concordant = np.searchsorted(tree, k_lo + offset, side="left") - group_offset[event_groups]
        same_tied = np.searchsorted(tree, k_hi + offset, side="left") - group_offset[event_groups] - same_concordant
    else:
        same_concordant = np.zeros(event_positions.size, dtype=np.int64)
        same_tied = same_concordant

    concordant_per_event = after_concordant + same_concordant
    tied_per_event = after_tied + same_tied
    discordant_per_event = n_partners - concordant_per_event - tied_per_event

    concordant = int(concordant_per_event.sum())
    discordant = int(discordant_per_event.sum())
    tied_risk = int(tied_per_event.sum())
    tied_time = int(np.dot(event_counts, censored_counts))

    w_i = weights_sorted[event_positions]
    numerator_terms = w_i * concordant_per_event + (0.5 * w_i) * tied_per_event
    denominator_terms = w_i * n_partners
    if denominator_terms.dtype != np.float64:
        denominator_terms = denominator_terms.astype(np.float64)
    numerator = np.float64(np.cumsum(numerator_terms)[-1])
    denominator = np.float64(np.cumsum(denominator_terms)[-1])

    if cast_overflow:
        warnings.warn("overflow encountered in cast", RuntimeWarning, stacklevel=2)
    if subtract_overflow:
        warnings.warn("overflow encountered in subtract", RuntimeWarning, stacklevel=2)

    cindex = numerator / denominator
    return cindex, concordant, discordant, tied_risk, tied_time


def _estimate_concordance_index_reference(event_indicator, event_time, estimate, weights, tied_tol=1e-8):
    """Reference O(n**2) implementation, used for dtypes with non-monotone comparisons."""
    order = np.argsort(event_time)

    tied_time = None

    concordant = 0
    discordant = 0
    tied_risk = 0
    numerator = 0.0
    denominator = 0.0
    for ind, mask, tied_time in _iter_comparable(event_indicator, event_time, order):
        est_i = estimate[order[ind]]
        event_i = event_indicator[order[ind]]
        w_i = weights[order[ind]]

        est = estimate[order[mask]]

        assert event_i, f"got censored sample at index {order[ind]}, but expected uncensored"

        ties = np.absolute(est - est_i) <= tied_tol
        n_ties = ties.sum()
        # an event should have a higher score
        con = est < est_i
        n_con = con[~ties].sum()

        numerator += w_i * n_con + 0.5 * w_i * n_ties
        denominator += w_i * mask.sum()

        tied_risk += n_ties
        concordant += n_con
        discordant += est.size - n_con - n_ties

    if tied_time is None:
        raise NoComparablePairException("Data has no comparable pairs, cannot estimate concordance index.")

    cindex = numerator / denominator
    return cindex, concordant, discordant, tied_risk, tied_time


def _sorted_time_groups(event_time_sorted):
    """Split a sorted array of time points into groups of equal time."""
    n_samples = event_time_sorted.shape[0]
    new_group = np.empty(n_samples, dtype=bool)
    new_group[0] = True
    np.not_equal(event_time_sorted[1:], event_time_sorted[:-1], out=new_group[1:])
    starts = np.flatnonzero(new_group)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = n_samples
    group_of = np.cumsum(new_group)
    group_of -= 1
    return starts, ends, group_of


class _RankPrefixCounter:
    """Count ranks smaller than a threshold among positional suffixes.

    Builds a wavelet matrix over the rank of each sample (in time-sorted order)
    such that ``count_less(start, upper)`` returns the number of samples at
    positions ``>= start`` whose rank is smaller than ``upper``.
    """

    __slots__ = ("n_samples", "n_bits", "zero_prefix", "zero_count")

    def __init__(self, ranks, n_samples):
        n_bits = max(1, int(n_samples).bit_length())
        self.n_samples = n_samples
        self.n_bits = n_bits
        self.zero_prefix = []
        self.zero_count = []
        current = np.ascontiguousarray(ranks, dtype=np.int64)
        for level in range(n_bits - 1, -1, -1):
            is_zero = ((current >> level) & 1) == 0
            prefix = np.empty(n_samples + 1, dtype=np.int32)
            prefix[0] = 0
            prefix[1:] = np.cumsum(is_zero, dtype=np.int32)
            self.zero_count.append(int(prefix[-1]))
            self.zero_prefix.append(prefix)
            current = np.concatenate((current[is_zero], current[~is_zero]))

    def count_less(self, start, upper):
        left = start.astype(np.int64, copy=True)
        right = np.full(left.shape[0], self.n_samples, dtype=np.int64)
        result = np.zeros(left.shape[0], dtype=np.int64)
        for index in range(self.n_bits):
            prefix = self.zero_prefix[index]
            level = self.n_bits - 1 - index
            left_zero = prefix[left].astype(np.int64)
            right_zero = prefix[right].astype(np.int64)
            take_ones = ((upper >> level) & 1).astype(bool)
            np.add(result, right_zero - left_zero, out=result, where=take_ones)
            zeros = self.zero_count[index]
            left = np.where(take_ones, zeros + left - left_zero, left_zero)
            right = np.where(take_ones, zeros + right - right_zero, right_zero)
        return result


def _tie_bounds(sorted_risks, event_risk, n_less, tied_tol):
    """Determine the contiguous range of risk ranks that are tied to an event.

    Returns ``(first, last, found)`` where the tied ranks are ``[first, last)``
    and ``found`` indicates whether any tied rank exists.  Comparisons use the
    dtype of ``sorted_risks`` exactly like the reference implementation.

    ``|risk - event_risk| <= tied_tol`` is monotone on either side of the
    event's own rank ``n_less``, which is the only risk score that is trivially
    tied (if the tolerance is not negative).  Therefore the tied ranks form an
    interval that can be found with two binary searches.
    """
    n_values = sorted_risks.shape[0]
    n_queries = event_risk.shape[0]
    n_iter = max(1, int(n_values).bit_length())

    found = np.absolute(sorted_risks[n_less] - event_risk) <= tied_tol

    # first tied rank at or below the event's rank
    lo = np.zeros(n_queries, dtype=np.int64)
    hi = np.where(found, n_less + 1, 0)
    for _ in range(n_iter):
        active = lo < hi
        middle = np.where(active, (lo + hi) >> 1, 0)
        is_tie = np.absolute(sorted_risks[middle] - event_risk) <= tied_tol
        hi = np.where(active & is_tie, middle, hi)
        lo = np.where(active & ~is_tie, middle + 1, lo)
    first = lo

    # first rank above the event's rank that is no longer tied
    lo = np.where(found, n_less, 0)
    hi = np.full(n_queries, n_values, dtype=np.int64)
    for _ in range(n_iter):
        active = lo < hi
        middle = np.where(active, (lo + hi) >> 1, 0)
        is_tie = np.absolute(sorted_risks[middle] - event_risk) <= tied_tol
        lo = np.where(active & is_tie, middle + 1, lo)
        hi = np.where(active & ~is_tie, middle, hi)

    return first, lo, found


def _supports_order_statistics(estimate):
    """Whether the O(n log n) counting is valid for the estimate's dtype."""
    dtype = estimate.dtype
    if dtype in (np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64)):
        return True
    if np.issubdtype(dtype, np.integer) and estimate.shape[0] > 0:
        info = np.iinfo(dtype)
        low = int(estimate.min())
        high = int(estimate.max())
        if np.issubdtype(dtype, np.signedinteger):
            # subtraction must not wrap around, otherwise the comparison of
            # absolute differences is no longer monotone
            return (high - low) <= int(info.max)
        return high == low
    return False


def _tolerance_cast_overflows(dtype, tied_tol):
    """Whether converting the tolerance to dtype raises an overflow warning."""
    if dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        return False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        np.less_equal(np.zeros(1, dtype=dtype), tied_tol)
    return any(issubclass(w.category, RuntimeWarning) for w in caught)


def _subtract_overflows(estimate_sorted, censored, starts, event_groups, suffix_start, event_risk, n_partners):
    """Whether the reference's risk differences overflow for any event.

    The reference computes ``estimate[comparable] - estimate[i]`` for every
    event ``i``.  A difference overflows if and only if it overflows for the
    smallest or largest risk score among the comparable samples, which can be
    obtained from suffix minima/maxima and per group extremes.
    """
    dtype = estimate_sorted.dtype
    if not np.issubdtype(dtype, np.floating):
        return False
    if not np.any(n_partners > 0):
        return False
    finite_max = float(np.finfo(dtype).max)
    if float(np.max(np.absolute(estimate_sorted))) <= finite_max / 2:
        # |a - b| <= |a| + |b| <= finite_max cannot overflow
        return False

    n_samples = estimate_sorted.shape[0]
    neg_inf = np.array(-np.inf, dtype=dtype)
    pos_inf = np.array(np.inf, dtype=dtype)
    suffix_max = np.maximum.accumulate(estimate_sorted[::-1])[::-1]
    suffix_min = np.minimum.accumulate(estimate_sorted[::-1])[::-1]
    group_max = np.maximum.reduceat(np.where(censored, estimate_sorted, neg_inf), starts)
    group_min = np.minimum.reduceat(np.where(censored, estimate_sorted, pos_inf), starts)

    has_suffix = suffix_start < n_samples
    safe_start = np.where(has_suffix, suffix_start, 0)
    top = np.maximum(np.where(has_suffix, suffix_max[safe_start], neg_inf), group_max[event_groups])
    bottom = np.minimum(np.where(has_suffix, suffix_min[safe_start], pos_inf), group_min[event_groups])

    with np.errstate(over="ignore", invalid="ignore"):
        high = np.subtract(top, event_risk)
        low = np.subtract(bottom, event_risk)
    overflow = (np.isinf(high) | np.isinf(low)) & (n_partners > 0)
    return bool(np.any(overflow))


def concordance_index_censored(event_indicator, event_time, estimate, tied_tol=1e-8):
    """Measures the agreement between a predicted risk score and the actual time-to-event.

    The concordance index is a measure of rank correlation between predicted risk
    scores and observed time points. It is defined as the proportion of all comparable
    pairs in which the predictions and outcomes are concordant.
    A pair of samples is concordant if the sample with a higher risk score has a
    shorter time-to-event. A higher concordance index indicates better model performance.

    A pair of samples is considered comparable if the sample with a shorter
    survival time experienced an event. This means we can confidently say that
    the individual with the shorter time had a worse outcome. If both samples
    are censored, or if they experienced an event at the same time, they are
    not comparable.

    When predicted risks are identical for a pair, 0.5 rather than 1 is added to the count
    of concordant pairs.

    See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb>`
    and [1]_ for further description.

    Parameters
    ----------
    event_indicator : array-like, shape = (n_samples,)
        A boolean array where ``True`` indicates an event and ``False`` indicates
        censoring.
    event_time : array-like, shape = (n_samples,)
        Array containing the time of an event or time of censoring.
    estimate : array-like, shape = (n_samples,)
        The predicted risk score for each sample (e.g., from ``estimator.predict(X)``).
        A higher value indicates a higher risk of experiencing an event.
    tied_tol : float, optional, default: 1e-8
        The tolerance value for considering ties in risk scores. If the
        absolute difference between two risk scores is smaller than or equal to
        ``tied_tol``, they are considered tied.

    Returns
    -------
    cindex : float
        The concordance index.
    concordant : int
        The number of concordant pairs.
    discordant : int
        The number of discordant pairs.
    tied_risk : int
        The number of pairs with tied risk scores.
    tied_time : int
        The number of comparable pairs with tied survival times.

    Notes
    -----
    This metric expects risk scores, which are typically returned by ``estimator.predict(X)``.
    It *does not accept* survival probabilities.

    See also
    --------
    concordance_index_ipcw
        A less biased estimator of the concordance index.

    References
    ----------
    .. [1] Harrell, F.E., Califf, R.M., Pryor, D.B., Lee, K.L., Rosati, R.A,
           "Multivariable prognostic models: issues in developing models,
           evaluating assumptions and adequacy, and measuring and reducing errors",
           Statistics in Medicine, 15(4), 361-87, 1996.
    """
    event_indicator, event_time, estimate = _check_inputs(event_indicator, event_time, estimate)

    w = np.ones_like(estimate)

    return _estimate_concordance_index(event_indicator, event_time, estimate, w, tied_tol)


def concordance_index_ipcw(survival_train, survival_test, estimate, tau=None, tied_tol=1e-8):
    r"""Concordance index for right-censored data based on inverse probability of censoring weights.

    This is an alternative to the estimator in :func:`concordance_index_censored`
    that does not depend on the distribution of censoring times in the test data.
    By using inverse probability of censoring weights (IPCW), it provides an unbiased
    and consistent estimate of the population concordance measure.

    This estimator requires access to survival times from the training data to
    estimate the censoring distribution.
    Note that survival times in `survival_test` must lie within the range of
    survival times in `survival_train`. This can be achieved by specifying the
    truncation time `tau`.
    The resulting `cindex` tells how well the given prediction model works in
    predicting events that occur in the time range from 0 to `tau`.

    For time points in `survival_test` that lie outside of the range specified by
    values in `survival_train`, the probability of censoring is unknown and an
    exception will be raised::

        ValueError: time must be smaller than largest observed time point

    The censoring distribution is estimated using the Kaplan-Meier estimator, which
    assumes that censoring is random and independent of the features.

    See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb>`
    and [1]_ for further description.

    Parameters
    ----------
    survival_train : structured array, shape = (n_train_samples,)
        Survival times for the training data, used to estimate the censoring
        distribution.
        A structured array with two fields. The first field is a boolean
        where ``True`` indicates an event and ``False`` indicates right-censoring.
        The second field is a float with the time of event or time of censoring.
    survival_test : structured array, shape = (n_samples,)
        Survival times for the test data.
        A structured array with two fields. The first field is a boolean
        where ``True`` indicates an event and ``False`` indicates right-censoring.
        The second field is a float with the time of event or time of censoring.
    estimate : array-like, shape = (n_samples,)
        Predicted risk scores for the test data (e.g., from ``estimator.predict(X)``).
        A higher value indicates a higher risk of experiencing an event.
    tau : float, optional
        Truncation time. The survival function for the underlying
        censoring time distribution :math:`D` needs to be positive
        at `tau`, i.e., `tau` should be chosen such that the
        probability of being censored after time `tau` is non-zero:
        :math:`P(D > \tau) > 0`. If `None`, no truncation is performed.
    tied_tol : float, optional, default: 1e-8
        The tolerance value for considering ties in risk scores.
        If the absolute difference between two risk scores is smaller than
        or equal to ``tied_tol``, they are considered tied.

    Returns
    -------
    cindex : float
        The concordance index.
    concordant : int
        The number of concordant pairs.
    discordant : int
        The number of discordant pairs.
    tied_risk : int
        The number of pairs with tied risk scores.
    tied_time : int
        The number of comparable pairs with tied survival times.

    Notes
    -----
    This metric expects risk scores, which are typically returned by ``estimator.predict(X)``.
    It *does not accept* survival probabilities.

    See also
    --------
    concordance_index_censored
        A simpler, but potentially biased, estimator of the concordance index.
    as_concordance_index_ipcw_scorer
        A wrapper class that uses :func:`concordance_index_ipcw`
        in its ``score`` method instead of the default
        :func:`concordance_index_censored`.

    References
    ----------
    .. [1] Uno, H., Cai, T., Pencina, M. J., D’Agostino, R. B., & Wei, L. J. (2011).
           "On the C-statistics for evaluating overall adequacy of risk prediction
           procedures with censored survival data".
           Statistics in Medicine, 30(10), 1105–1117.
    """
    test_event, test_time = check_y_survival(survival_test)

    if tau is not None:
        mask = test_time < tau
        survival_test = survival_test[mask]

    estimate = _check_estimate_1d(estimate, test_time)

    cens = CensoringDistributionEstimator()
    cens.fit(survival_train)
    ipcw_test = cens.predict_ipcw(survival_test)
    if tau is None:
        ipcw = ipcw_test
    else:
        ipcw = np.empty(estimate.shape[0], dtype=ipcw_test.dtype)
        ipcw[mask] = ipcw_test
        ipcw[~mask] = 0

    w = np.square(ipcw)

    return _estimate_concordance_index(test_event, test_time, estimate, w, tied_tol)


def cumulative_dynamic_auc(survival_train, survival_test, estimate, times, tied_tol=1e-8):
    r"""Computes the cumulative/dynamic area under the ROC curve (AUC) for right-censored data.

    This metric evaluates a model's performance at specific time points.
    The cumulative/dynamic AUC at time :math:`t` quantifies how well a model can
    distinguish subjects who experience an event by time :math:`t` (cases) from
    those who do not (controls). A higher AUC indicates better model performance.

    This function can also evaluate models with time-dependent predictions, such as
    :class:`sksurv.ensemble.RandomSurvivalForest`
    (see :ref:`User Guide </user_guide/evaluating-survival-models.ipynb#Using-Time-dependent-Risk-Scores>`).
    In this case, ``estimate`` must be a 2D array where ``estimate[i, j]`` is the
    predicted risk score for the :math:`i`-th instance at time point ``times[j]``.

    The receiver operating characteristic (ROC) curve and the area under the
    ROC curve (AUC) are metrics to evaluate a binary classifier. Each point on
    the ROC denotes the performance of a binary classifier at a specific
    threshold with respect to the sensitivity (true positive rate) on the
    y-axis and the specificity (true negative rate) on the x-axis.

    ROC and AUC can be extended to survival analysis by defining cases and
    controls based on a time point :math:`t`. *Cumulative cases* are all
    individuals that experienced an event prior to or at time
    :math:`t` (:math:`t_i \leq t`), whereas *dynamic controls* are those
    with :math:`t_i > t`. Given an estimator of the :math:`i`-th individual's
    risk score :math:`\hat{f}(\mathbf{x}_i)`, the cumulative/dynamic AUC at
    time :math:`t` is defined as

    .. math::

        \widehat{\mathrm{AUC}}(t) =
        \frac{\sum_{i=1}^n \sum_{j=1}^n I(y_j > t) I(y_i \leq t) \omega_i
        I(\hat{f}(\mathbf{x}_j) \leq \hat{f}(\mathbf{x}_i))}
        {(\sum_{i=1}^n I(y_i > t)) (\sum_{i=1}^n I(y_i \leq t) \omega_i)}

    where :math:`\omega_i` are inverse probability of censoring weights (IPCW).

    To account for censoring, this metric uses inverse probability of censoring
    weights (IPCW), which requires access to survival times from the training
    data to estimate the censoring distribution. Note that survival times in
    ``survival_test`` must lie within the range of survival times in ``survival_train``.
    This can be achieved by specifying ``times`` accordingly, e.g. by setting
    ``times[-1]`` slightly below the maximum expected follow-up time.

    For time points in ``survival_test`` that lie outside of the range specified by
    values in ``survival_train``, the probability of censoring is unknown and an
    exception will be raised::

        ValueError: time must be smaller than largest observed time point

    The censoring distribution is estimated using the Kaplan-Meier estimator, which
    assumes that censoring is random and independent of the features.

    The function also returns a summary measure, which is the mean of the
    :math:`\mathrm{AUC}(t)` over the specified time range, weighted by the
    estimated survival function:

    .. math::

        \overline{\mathrm{AUC}}(\tau_1, \tau_2) =
        \frac{1}{\hat{S}(\tau_1) - \hat{S}(\tau_2)}
        \int_{\tau_1}^{\tau_2} \widehat{\mathrm{AUC}}(t)\,d \hat{S}(t)

    where :math:`\hat{S}(t)` is the Kaplan–Meier estimator of the survival function.

    See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb#Time-dependent-Area-under-the-ROC>`,
    [1]_, [2]_, [3]_ for further description.

    Parameters
    ----------
    survival_train : structured array, shape = (n_train_samples,)
        Survival times for the training data, used to estimate the censoring
        distribution.
        A structured array with two fields. The first field is a boolean
        where ``True`` indicates an event and ``False`` indicates right-censoring.
        The second field is a float with the time of event or time of censoring.
    survival_test : structured array, shape = (n_samples,)
        Survival times for the test data.
        A structured array with two fields. The first field is a boolean
        where ``True`` indicates an event and ``False`` indicates right-censoring.
        The second field is a float with the time of event or time of censoring.
    estimate : array-like, shape = (n_samples,) or (n_samples, n_times)
        Predicted risk scores for the test data (e.g., from ``estimator.predict(X)``.
        A higher value indicates a higher risk of experiencing an event.
        If a 1D array is provided, the same risk score is used for all time points.
        If a 2D array is provided, ``estimate[:, j]`` is used for the :math:`j`-th
        time point.
    times : array-like, shape = (n_times,)
        The time points at which to compute the AUC. Values must be within the
        range of follow-up times in ``survival_test``.
    tied_tol : float, optional, default: 1e-8
        The tolerance value for considering ties in risk scores. If the
        absolute difference between two risk scores is smaller than or equal to
        ``tied_tol``, they are considered tied.

    Returns
    -------
    auc : ndarray, shape = (n_times,)
        The cumulative/dynamic AUC estimates at each time point in ``times``.
    mean_auc : float
        The mean cumulative/dynamic AUC over the specified time range ``(times[0], times[-1])``.

    Notes
    -----
    This metric expects risk scores, which are typically returned by ``estimator.predict(X)``
    (for time-independent risks), or ``estimator.predict_cumulative_hazard_function(X)``
    (for time-dependent risks). It *does not accept* survival probabilities.

    See also
    --------
    as_cumulative_dynamic_auc_scorer
        A wrapper class that uses :func:`cumulative_dynamic_auc`
        in its ``score`` method instead of the default
        :func:`concordance_index_censored`.

    References
    ----------
    .. [1] H. Uno, T. Cai, L. Tian, and L. J. Wei,
           "Evaluating prediction rules for t-year survivors with censored regression models,"
           Journal of the American Statistical Association, vol. 102, pp. 527–537, 2007.
    .. [2] H. Hung and C. T. Chiang,
           "Estimation methods for time-dependent AUC models with survival data,"
           Canadian Journal of Statistics, vol. 38, no. 1, pp. 8–26, 2010.
    .. [3] J. Lambert and S. Chevret,
           "Summary measure of discrimination in survival models based on cumulative/dynamic time-dependent ROC curves,"
           Statistical Methods in Medical Research, 2014.
    """
    test_event, test_time = check_y_survival(survival_test)
    estimate, times = _check_estimate_2d(estimate, test_time, times, estimator="cumulative_dynamic_auc")

    n_samples = estimate.shape[0]
    n_times = times.shape[0]
    if estimate.ndim == 1:
        estimate = np.broadcast_to(estimate[:, np.newaxis], (n_samples, n_times))

    # fit and transform IPCW
    cens = CensoringDistributionEstimator()
    cens.fit(survival_train)
    ipcw = cens.predict_ipcw(survival_test)

    # expand arrays to (n_samples, n_times) shape
    test_time = np.broadcast_to(test_time[:, np.newaxis], (n_samples, n_times))
    test_event = np.broadcast_to(test_event[:, np.newaxis], (n_samples, n_times))
    times_2d = np.broadcast_to(times, (n_samples, n_times))
    ipcw = np.broadcast_to(ipcw[:, np.newaxis], (n_samples, n_times))

    # sort each time point (columns) by risk score (descending)
    o = np.argsort(-estimate, axis=0)
    test_time = np.take_along_axis(test_time, o, axis=0)
    test_event = np.take_along_axis(test_event, o, axis=0)
    estimate = np.take_along_axis(estimate, o, axis=0)
    ipcw = np.take_along_axis(ipcw, o, axis=0)

    is_case = (test_time <= times_2d) & test_event
    is_control = test_time > times_2d
    n_controls = is_control.sum(axis=0)

    # prepend row of infinity values
    estimate_diff = np.concatenate((np.broadcast_to(np.inf, (1, n_times)), estimate))
    is_tied = np.absolute(np.diff(estimate_diff, axis=0)) <= tied_tol

    cumsum_tp = np.cumsum(is_case * ipcw, axis=0)
    cumsum_fp = np.cumsum(is_control, axis=0)
    true_pos = cumsum_tp / cumsum_tp[-1]
    false_pos = cumsum_fp / n_controls

    scores = np.empty(n_times, dtype=float)
    it = np.nditer((true_pos, false_pos, is_tied), order="F", flags=["external_loop"])
    with it:
        for i, (tp, fp, mask) in enumerate(it):
            idx = np.flatnonzero(mask) - 1
            # only keep the last estimate for tied risk scores
            tp_no_ties = np.delete(tp, idx)
            fp_no_ties = np.delete(fp, idx)
            # Add an extra threshold position
            # to make sure that the curve starts at (0, 0)
            tp_no_ties = np.r_[0, tp_no_ties]
            fp_no_ties = np.r_[0, fp_no_ties]
            scores[i] = np.trapezoid(tp_no_ties, fp_no_ties)

    if n_times == 1:
        mean_auc = scores[0]
    else:
        surv = SurvivalFunctionEstimator()
        surv.fit(survival_test)
        s_times = surv.predict_proba(times)
        # compute integral of AUC over survival function
        d = -np.diff(np.r_[1.0, s_times])
        integral = (scores * d).sum()
        mean_auc = integral / (1.0 - s_times[-1])

    return scores, mean_auc


def brier_score(survival_train, survival_test, estimate, times):
    r"""The time-dependent Brier score for right-censored data.

    The time-dependent Brier score measures the inaccuracy of
    predicted survival probabilities at a given time point.
    It is the mean squared error between the true survival status
    and the predicted survival probability at time point :math:`t`.
    A lower Brier score indicates better model performance.

    To account for censoring, this metric uses inverse probability of censoring
    weights (IPCW), which requires access to survival times from the training
    data to estimate the censoring distribution. Note that survival times in
    ``survival_test`` must lie within the range of survival times in ``survival_train``.
    This can be achieved by specifying ``times`` accordingly, e.g. by setting
    ``times[-1]`` slightly below the maximum expected follow-up time.

    For time points in ``survival_test`` that lie outside of the range specified by
    values in ``survival_train``, the probability of censoring is unknown and an
    exception will be raised::

        ValueError: time must be smaller than largest observed time point

    The censoring distribution is estimated using the Kaplan-Meier estimator, which
    assumes that censoring is random and independent of the features.

    The time-dependent Brier score at time :math:`t` is defined as

    .. math::

        \mathrm{BS}^c(t) = \frac{1}{n} \sum_{i=1}^n I(y_i \leq t \land \delta_i = 1)
        \frac{(0 - \hat{\pi}(t | \mathbf{x}_i))^2}{\hat{G}(y_i)} + I(y_i > t)
        \frac{(1 - \hat{\pi}(t | \mathbf{x}_i))^2}{\hat{G}(t)} ,

    where :math:`\hat{\pi}(t | \mathbf{x})` is the predicted survival probability
    up to the time point :math:`t` for a feature vector :math:`\mathbf{x}`,
    and :math:`1/\hat{G}(t)` is a inverse probability of censoring weight.

    See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb#Time-dependent-Brier-Score>`
    and [1]_ for details.

    Parameters
    ----------
    survival_train : structured array, shape = (n_train_samples,)
        Survival times for the training data, used to estimate the censoring
        distribution.
        A structured array with two fields. The first field is a boolean
        where ``True`` indicates an event and ``False`` indicates right-censoring.
        The second field is a float with the time of event or time of censoring.
    survival_test : structured array, shape = (n_samples,)
        Survival times for the test data.
        A structured array with two fields. The first field is a boolean
        where ``True`` indicates an event and ``False`` indicates right-censoring.
        The second field is a float with the time of event or time of censoring.
    estimate : array-like, shape = (n_samples, n_times)
        Predicted survival probabilities for the test data at the time points
        specified by ``times``, typically obtained from
        ``estimator.predict_survival_function(X)``. The value of ``estimate[:, i]``
        must correspond to the estimated survival probability up to
        the time point ``times[i]``.
    times : array-like, shape = (n_times,)
        The time points at which to compute the Brier score. Values must be
        within the range of follow-up times in ``survival_test``.

    Returns
    -------
    times : ndarray, shape = (n_times,)
        The unique time points at which the Brier score was estimated.
    brier_scores : ndarray, shape = (n_times,)
        The Brier score at each time point in ``times``.

    Notes
    -----
    This metric expects survival probabilities, which are typically returned by
    ``estimator.predict_survival_function(X)``.
    It *does not accept* risk scores.

    Examples
    --------
    >>> from sksurv.datasets import load_gbsg2
    >>> from sksurv.linear_model import CoxPHSurvivalAnalysis
    >>> from sksurv.metrics import brier_score
    >>> from sksurv.preprocessing import OneHotEncoder

    Load and prepare data.

    >>> X, y = load_gbsg2()
    >>> # The tumor grade is recorded as the Roman numerals 'I', 'II', 'III';
    >>> # their string lengths (1, 2, 3) give the ordinal encoding.
    >>> X["tgrade"] = X.loc[:, "tgrade"].map(len).astype(int)
    >>> Xt = OneHotEncoder().fit_transform(X)

    Fit a Cox model.

    >>> est = CoxPHSurvivalAnalysis(ties="efron").fit(Xt, y)

    Retrieve individual survival functions and get probability
    of remaining event free up to 5 years (=1825 days).

    >>> survs = est.predict_survival_function(Xt)
    >>> preds = [fn(1825) for fn in survs]

    Compute the Brier score at 5 years.

    >>> times, score = brier_score(y, y, preds, 1825)
    >>> print(score)
    [0.20881843]

    See also
    --------
    integrated_brier_score
        Computes the average Brier score over all time points.

    References
    ----------
    .. [1] E. Graf, C. Schmoor, W. Sauerbrei, and M. Schumacher,
           "Assessment and comparison of prognostic classification schemes for survival data,"
           Statistics in Medicine, vol. 18, no. 17-18, pp. 2529–2545, 1999.
    """
    test_event, test_time = check_y_survival(survival_test)
    estimate, times = _check_estimate_2d(estimate, test_time, times, estimator="brier_score")
    if estimate.ndim == 1 and times.shape[0] == 1:
        estimate = estimate.reshape(-1, 1)

    # fit IPCW estimator
    cens = CensoringDistributionEstimator().fit(survival_train)
    # calculate inverse probability of censoring weight at current time point t.
    prob_cens_t = cens.predict_proba(times)
    prob_cens_t[prob_cens_t == 0] = np.inf
    # calculate inverse probability of censoring weights at observed time point
    prob_cens_y = cens.predict_proba(test_time)
    prob_cens_y[prob_cens_y == 0] = np.inf

    # Calculating the brier scores at each time point
    brier_scores = np.empty(times.shape[0], dtype=float)
    for i, t in enumerate(times):
        est = estimate[:, i]
        is_case = (test_time <= t) & test_event
        is_control = test_time > t

        brier_scores[i] = np.mean(
            np.square(est) * is_case.astype(int) / prob_cens_y
            + np.square(1.0 - est) * is_control.astype(int) / prob_cens_t[i]
        )

    return times, brier_scores


def integrated_brier_score(survival_train, survival_test, estimate, times):
    r"""Computes the integrated Brier score (IBS).

    The IBS is an overall measure of the model's performance across all
    available time points :math:`t_1 \leq t \leq t_\text{max}`.
    It is the average Brier score, integrated over time.
    A lower IBS indicates better model performance.

    The integrated time-dependent Brier score over the interval
    :math:`[t_1; t_\text{max}]` is defined as

    .. math::

        \mathrm{IBS} = \int_{t_1}^{t_\text{max}} \mathrm{BS}^c(t) d w(t)

    where the weighting function is :math:`w(t) = t / t_\text{max}`.
    The integral is estimated via the trapezoidal rule.

    See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb#Time-dependent-Brier-Score>`
    and [1]_ for further details.

    Parameters
    ----------
    survival_train : structured array, shape = (n_train_samples,)
        Survival times for the training data, used to estimate the censoring
        distribution.
        A structured array with two fields. The first field is a boolean
        where ``True`` indicates an event and ``False`` indicates right-censoring.
        The second field is a float with the time of event or time of censoring.
    survival_test : structured array, shape = (n_samples,)
        Survival times for the test data.
        A structured array with two fields. The first field is a boolean
        where ``True`` indicates an event and ``False`` indicates right-censoring.
        The second field is a float with the time of event or time of censoring.
    estimate : array-like, shape = (n_samples, n_times)
        Predicted survival probabilities for the test data at the time points
        specified by ``times``, typically obtained from
        ``estimator.predict_survival_function(X)``. The value of ``estimate[:, i]``
        must correspond to the estimated survival probability up to
        the time point ``times[i]``.
    times : array-like, shape = (n_times,)
        The time points at which to compute the Brier score. Values must be
        within the range of follow-up times in ``survival_test``.

    Returns
    -------
    ibs : float
        The integrated Brier score.

    Notes
    -----
    This metric expects survival probabilities, which are typically returned by
    ``estimator.predict_survival_function(X)``.
    It *does not accept* risk scores.

    Examples
    --------
    >>> import numpy as np
    >>> from sksurv.datasets import load_gbsg2
    >>> from sksurv.linear_model import CoxPHSurvivalAnalysis
    >>> from sksurv.metrics import integrated_brier_score
    >>> from sksurv.preprocessing import OneHotEncoder

    Load and prepare data.

    >>> X, y = load_gbsg2()
    >>> # The tumor grade is recorded as the Roman numerals 'I', 'II', 'III';
    >>> # their string lengths (1, 2, 3) give the ordinal encoding.
    >>> X["tgrade"] = X.loc[:, "tgrade"].map(len).astype(int)
    >>> Xt = OneHotEncoder().fit_transform(X)

    Fit a Cox model.

    >>> est = CoxPHSurvivalAnalysis(ties="efron").fit(Xt, y)

    Retrieve individual survival functions and get probability
    of remaining event free from 1 year to 5 years (=1825 days).

    >>> survs = est.predict_survival_function(Xt)
    >>> times = np.arange(365, 1826)
    >>> preds = np.asarray([[fn(t) for t in times] for fn in survs])

    Compute the integrated Brier score from 1 to 5 years.

    >>> score = integrated_brier_score(y, y, preds, times)
    >>> print(round(score, 4))
    0.1816

    See also
    --------
    brier_score
        Computes the Brier score at specified time points.

    as_integrated_brier_score_scorer
        Wrapper class that uses :func:`integrated_brier_score`
        in its ``score`` method instead of the default
        :func:`concordance_index_censored`.

    References
    ----------
    .. [1] E. Graf, C. Schmoor, W. Sauerbrei, and M. Schumacher,
           "Assessment and comparison of prognostic classification schemes for survival data,"
           Statistics in Medicine, vol. 18, no. 17-18, pp. 2529–2545, 1999.
    """
    # Computing the brier scores
    times, brier_scores = brier_score(survival_train, survival_test, estimate, times)

    if times.shape[0] < 2:
        raise ValueError("At least two time points must be given")

    # Computing the IBS
    ibs_value = np.trapezoid(brier_scores, times) / (times[-1] - times[0])

    return ibs_value


def _estimator_has(attr):
    """Check that meta_estimator has `attr`.

    Used together with `available_if`."""

    def check(self):
        # raise original `AttributeError` if `attr` does not exist
        getattr(self.estimator_, attr)
        return True

    return check


class _ScoreOverrideMixin:
    def __init__(self, estimator, predict_func, score_func, score_index, greater_is_better):
        if not hasattr(estimator, predict_func):
            raise AttributeError(f"{estimator!r} object has no attribute {predict_func!r}")

        self.estimator = estimator
        self._predict_func = predict_func
        self._score_func = score_func
        self._score_index = score_index
        self._sign = 1 if greater_is_better else -1

    def _get_score_params(self):
        """Return dict of parameters passed to ``score_func``."""
        params = self.get_params(deep=False)
        del params["estimator"]
        return params

    def fit(self, X, y, **fit_params):
        self._train_y = np.array(y, copy=True)
        self.estimator_ = self.estimator.fit(X, y, **fit_params)
        return self

    def _do_predict(self, X):
        predict_func = getattr(self.estimator_, self._predict_func)
        return predict_func(X)

    def score(self, X, y):
        """Returns the score on the given data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = (n_samples,)
            Target relative to X for classification or regression;
            None for unsupervised learning.

        Returns
        -------
        score : float
        """
        estimate = self._do_predict(X)
        score = self._score_func(
            survival_train=self._train_y,
            survival_test=y,
            estimate=estimate,
            **self._get_score_params(),
        )
        if self._score_index is not None:
            score = score[self._score_index]
        return self._sign * score

    @available_if(_estimator_has("predict"))
    def predict(self, X):
        """Call predict on the estimator.

        Only available if estimator supports ``predict``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        """
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    @available_if(_estimator_has("predict_cumulative_hazard_function"))
    def predict_cumulative_hazard_function(self, X):
        """Call predict_cumulative_hazard_function on the estimator.

        Only available if estimator supports ``predict_cumulative_hazard_function``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        """
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_cumulative_hazard_function(X)

    @available_if(_estimator_has("predict_survival_function"))
    def predict_survival_function(self, X):
        """Call predict_survival_function on the estimator.

        Only available if estimator supports ``predict_survival_function``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        """
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_survival_function(X)


class as_cumulative_dynamic_auc_scorer(_ScoreOverrideMixin, BaseEstimator):
    """Wraps an estimator to use :func:`cumulative_dynamic_auc` as ``score`` function.

    See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb#Using-Metrics-in-Hyper-parameter-Search>`
    for using it for hyper-parameter optimization.

    Parameters
    ----------
    estimator : object
        Instance of an estimator.
    times : array-like, shape = (n_times,)
        The time points at which to compute the AUC. Values must be within the
        range of follow-up times of the test data.
    tied_tol : float, optional, default: 1e-8
        The tolerance value for considering ties in risk scores. If the
        absolute difference between two risk scores is smaller than or equal to
        ``tied_tol``, they are considered tied.

    Attributes
    ----------
    estimator_ : estimator
        Estimator that was fit.

    See also
    --------
    cumulative_dynamic_auc
    """

    def __init__(self, estimator, times, tied_tol=1e-8):
        super().__init__(
            estimator=estimator,
            predict_func="predict",
            score_func=cumulative_dynamic_auc,
            score_index=1,
            greater_is_better=True,
        )
        self.times = times
        self.tied_tol = tied_tol


class as_concordance_index_ipcw_scorer(_ScoreOverrideMixin, BaseEstimator):
    r"""Wraps an estimator to use :func:`concordance_index_ipcw` as ``score`` function.

    See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb#Using-Metrics-in-Hyper-parameter-Search>`
    for using it for hyper-parameter optimization.

    Parameters
    ----------
    estimator : object
        Instance of an estimator.
    tau : float, optional
        Truncation time. The survival function for the underlying
        censoring time distribution :math:`D` needs to be positive
        at `tau`, i.e., `tau` should be chosen such that the
        probability of being censored after time `tau` is non-zero:
        :math:`P(D > \tau) > 0`. If `None`, no truncation is performed.
    tied_tol : float, optional, default: 1e-8
        The tolerance value for considering ties in risk scores.
        If the absolute difference between two risk scores is smaller than
        or equal to ``tied_tol``, they are considered tied.

    Attributes
    ----------
    estimator_ : estimator
        Estimator that was fit.

    See also
    --------
    concordance_index_ipcw
    """

    def __init__(self, estimator, tau=None, tied_tol=1e-8):
        super().__init__(
            estimator=estimator,
            predict_func="predict",
            score_func=concordance_index_ipcw,
            score_index=0,
            greater_is_better=True,
        )
        self.tau = tau
        self.tied_tol = tied_tol


class as_integrated_brier_score_scorer(_ScoreOverrideMixin, BaseEstimator):
    """Wraps an estimator to use the negative of :func:`integrated_brier_score` as ``score`` function.

    The estimator needs to be able to estimate survival functions via
    a ``predict_survival_function`` method.

    See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb#Using-Metrics-in-Hyper-parameter-Search>`
    for using it for hyper-parameter optimization.

    Parameters
    ----------
    estimator : object
        Instance of an estimator that provides ``predict_survival_function``.
    times : array-like, shape = (n_times,)
        The time points at which to compute the Brier score. Values must be
        within the range of follow-up times of the test data.

    Attributes
    ----------
    estimator_ : estimator
        Estimator that was fit.

    See also
    --------
    integrated_brier_score
    """

    def __init__(self, estimator, times):
        super().__init__(
            estimator=estimator,
            predict_func="predict_survival_function",
            score_func=integrated_brier_score,
            score_index=None,
            greater_is_better=False,
        )
        self.times = times

    def _do_predict(self, X):
        predict_func = getattr(self.estimator_, self._predict_func)
        surv_fns = predict_func(X)
        times = self.times
        estimates = np.empty((len(surv_fns), len(times)))
        for i, fn in enumerate(surv_fns):
            estimates[i, :] = fn(times)
        return estimates
