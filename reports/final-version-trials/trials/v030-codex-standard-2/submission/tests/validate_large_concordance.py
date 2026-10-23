"""Large-cohort checks with an independent histogram-based oracle."""

import math
from pathlib import Path
import sys
from time import monotonic
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv


def expected(events, times, risk_bins, values, tolerance, weights):
    n_times = int(times.max()) + 1
    n_risks = len(values)
    cells = times * n_risks + risk_bins
    event_hist = np.bincount(cells[events], minlength=n_times * n_risks).reshape(n_times, n_risks)
    censor_hist = np.bincount(cells[~events], minlength=n_times * n_risks).reshape(n_times, n_risks)
    later = (event_hist + censor_hist).sum(axis=0)
    concordant = discordant = tied_risk = tied_time = 0
    numerator = denominator = 0.0
    for t in range(n_times):
        later -= event_hist[t] + censor_hist[t]
        partners = later + censor_hist[t]
        n_partners = int(partners.sum())
        tied_time += int(event_hist[t].sum()) * int(censor_hist[t].sum())
        for risk_bin, event_count in enumerate(event_hist[t]):
            with np.errstate(over="ignore"):
                ties = np.absolute(values - values[risk_bin]) <= tolerance
            n_ties = int(partners[ties].sum())
            n_con = int(partners[(values < values[risk_bin]) & ~ties].sum())
            concordant += int(event_count) * n_con
            tied_risk += int(event_count) * n_ties
            discordant += int(event_count) * (n_partners - n_con - n_ties)
            numerator += weights[t] * int(event_count) * (n_con + 0.5 * n_ties)
            denominator += weights[t] * int(event_count) * n_partners
    return numerator / denominator, concordant, discordant, tied_risk, tied_time


def run():
    rng = np.random.default_rng(712389)
    n = 200_000
    train = Surv.from_arrays(rng.random(5000) < 0.7, rng.integers(0, 66, 5000))
    censoring = CensoringDistributionEstimator().fit(train)
    maximum = np.finfo(float).max
    scenarios = [
        (np.array([-2., -1., 0., 1., 2.]), 1.0, 64, None),
        (np.array([0., 1e-8, np.nextafter(1e-8, 0), np.nextafter(1e-8, np.inf), 2e-8]), 1e-8, 64, 42),
        (np.array([-maximum, maximum, -1., 0., 1.]), 1e-8, 64, 42),
        (np.array([-maximum, maximum, -1., 0., 1.]), np.inf, 64, None),
        (np.array([-2., -1., 0., 1., 2.]), 0.0, 1, None),
    ]
    for case, (values, tolerance, n_times, tau) in enumerate(scenarios):
        events = rng.random(n) < 0.65
        times = rng.integers(0, n_times, n)
        risk_bins = rng.integers(0, len(values), n)
        risks = values[risk_bins]
        test = Surv.from_arrays(events, times)
        grid = np.arange(n_times)
        ipcw_weights = np.square(censoring.predict_ipcw(Surv.from_arrays(np.ones(n_times, bool), grid)))
        if tau is not None:
            ipcw_weights[grid >= tau] = 0
        for weighted in [False, True]:
            reference = expected(events, times, risk_bins, values, tolerance, ipcw_weights if weighted else np.ones(n_times))
            start = monotonic()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if weighted:
                    actual = concordance_index_ipcw(
                        survival_train=train, survival_test=test, estimate=risks, tau=tau, tied_tol=tolerance
                    )
                else:
                    actual = concordance_index_censored(
                        event_indicator=events, event_time=times, estimate=risks, tied_tol=tolerance
                    )
            assert actual[1:] == reference[1:], (case, weighted, actual, reference)
            assert math.isclose(actual[0], reference[0], rel_tol=2e-12, abs_tol=2e-14), (case, weighted, actual, reference)
            categories = {warning.category.__name__ for warning in caught}
            assert categories == ({"RuntimeWarning"} if case in [2, 3] else set()), (case, categories)
            print(f"case={case} ipcw={weighted} seconds={monotonic() - start:.3f} pairs={sum(actual[1:4])}", flush=True)
    print("Passed all ten 200,000-record histogram checks.")


if __name__ == "__main__":
    run()
