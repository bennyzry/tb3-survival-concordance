// SPDX-License-Identifier: GPL-3.0-or-later
// Build: g++ -O3 -std=c++17 -fPIC -shared -fno-fast-math -ffp-contract=off
//        _concordance_native.cpp -o _concordance_native.so
// This library deliberately uses a plain C ABI, independent of Python/NumPy.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {
using Index = std::ptrdiff_t;
using Count = std::int64_t;

class RiskCounts {
    std::vector<Count> tree;

public:
    std::vector<Count> frequency;

    RiskCounts(Index m, Index n, const Index* ranks)
        : tree(m + 1, 0), frequency(m, 0) {
        for (Index i = 0; i < n; ++i) {
            ++frequency[ranks[i]];
        }
        for (Index i = 1; i <= m; ++i) {
            tree[i] += frequency[i - 1];
            const Index parent = i + (i & -i);
            if (parent <= m) tree[parent] += tree[i];
        }
    }

    void remove(Index rank) {
        --frequency[rank];
        for (Index i = rank + 1; i < static_cast<Index>(tree.size()); i += i & -i) {
            --tree[i];
        }
    }

    Count prefix(Index end) const {
        Count count = 0;
        for (Index i = end; i > 0; i -= i & -i) count += tree[i];
        return count;
    }
};
} // namespace

extern "C" int concordance_counts(
    Index n, Index m, Index ngroups, const Index* ranks, const Index* ends,
    const bool* events, const double* values, double tol,
    Count* concordant, Count* ties, Count* comparable, Count* info) noexcept {
    try {
        RiskCounts active(m, n, ranks);
        std::vector<Index> lower(m), upper(m);
        for (Index r = 0; r < m; ++r) {
            if (!(tol >= 0)) {
                // Negative and NaN tolerances never declare a risk tie, even
                // for identical estimates. Strict risk ordering still applies.
                lower[r] = upper[r] = r;
                continue;
            }
            // Search using the actual rounded subtraction, rather than
            // risk +/- tolerance. The latter can misclassify the inclusive
            // boundary, and treating close risks as equivalence classes
            // would incorrectly assume transitive ties.
            Index lo = 0, hi = r;
            while (lo < hi) {
                const Index mid = lo + (hi - lo) / 2;
                if (std::fabs(values[mid] - values[r]) <= tol) hi = mid;
                else lo = mid + 1;
            }
            lower[r] = lo;
            lo = r + 1;
            hi = m;
            while (lo < hi) {
                const Index mid = lo + (hi - lo) / 2;
                if (std::fabs(values[mid] - values[r]) <= tol) lo = mid + 1;
                else hi = mid;
            }
            upper[r] = lo;
        }

        Count remaining = n;
        Index min_rank = 0, max_rank = m - 1;
        Index start = 0;
        for (Index g = 0; g < ngroups && start < n - 1; ++g) {
            const Index end = ends[g];
            Count nevents = 0;
            // Same-time events must not be comparison partners. Same-time
            // censors stay in the tree until all these events are queried.
            for (Index i = start; i < end; ++i) {
                if (events[i]) {
                    active.remove(ranks[i]);
                    --remaining;
                    ++nevents;
                }
            }
            info[0] += nevents * (end - start - nevents);
            if (remaining) {
                while (!active.frequency[min_rank]) ++min_rank;
                while (!active.frequency[max_rank]) --max_rank;
            }
            for (Index i = start; i < end; ++i) {
                if (!events[i]) continue;
                const Index r = ranks[i];
                const Count below = active.prefix(lower[r]);
                concordant[i] = below;
                ties[i] = active.prefix(upper[r]) - below;
                comparable[i] = remaining;
                // Only subtraction for an actual comparable pair is allowed
                // to produce the overflow warning. Precomputed bounds may
                // have examined risks that are never comparable.
                if (remaining &&
                    (std::isinf(values[min_rank] - values[r]) ||
                     std::isinf(values[max_rank] - values[r]))) {
                    info[1] = 1;
                }
            }
            for (Index i = start; i < end; ++i) {
                if (!events[i]) {
                    active.remove(ranks[i]);
                    --remaining;
                }
            }
            start = end;
        }
        return 0;
    } catch (...) {
        return -1;
    }
}
