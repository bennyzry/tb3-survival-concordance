/* Ranked pair counting for metrics.py.  Build with:
 * cc -O3 -std=c99 -fPIC -shared sksurv/_concordance_counts.c \
 *    -o sksurv/_concordance_counts.so
 *
 * The Fenwick tree holds subjects at later times, plus the censors at the
 * current time.  Events at the current time are inserted only after all of
 * them have been queried.  All counts and indices are explicitly 64 bit.
 */
#include <stdint.h>
#include <stdlib.h>

static int64_t prefix(const int64_t *tree, int64_t end)
{
    int64_t result = 0;
    for (; end > 0; end -= end & -end)
        result += tree[end];
    return result;
}

static void insert(int64_t *tree, int64_t size, int64_t rank)
{
    for (int64_t j = rank + 1; j <= size; j += j & -j)
        ++tree[j];
}

int concordance_counts(int64_t n, int64_t size, int64_t n_groups,
                       const int64_t *ranks, const uint8_t *events,
                       const int64_t *starts, const int64_t *lower,
                       const int64_t *upper, int64_t *concordant,
                       int64_t *tied, int64_t *comparable,
                       int64_t *minimum, int64_t *maximum,
                       int64_t *tied_time)
{
    int64_t *tree = calloc((size_t)size + 1, sizeof(*tree));
    if (tree == NULL)
        return -1;

    int64_t total = 0, min_rank = size, max_rank = 0;
    *tied_time = 0;
    for (int64_t group = n_groups; group-- > 0;) {
        int64_t begin = starts[group];
        int64_t end = group + 1 < n_groups ? starts[group + 1] : n;
        int64_t censors = 0;

        for (int64_t i = begin; i < end; ++i) {
            if (!events[i]) {
                int64_t rank = ranks[i];
                insert(tree, size, rank);
                if (rank < min_rank) min_rank = rank;
                if (rank > max_rank) max_rank = rank;
                ++total;
                ++censors;
            }
        }
        for (int64_t i = begin; i < end; ++i) {
            if (events[i]) {
                int64_t rank = ranks[i];
                int64_t below = prefix(tree, lower[rank]);
                concordant[i] = below;
                tied[i] = prefix(tree, upper[rank]) - below;
                comparable[i] = total;
                minimum[i] = total ? min_rank : 0;
                maximum[i] = total ? max_rank : 0;
                *tied_time += censors;
            }
        }
        for (int64_t i = begin; i < end; ++i) {
            if (events[i]) {
                int64_t rank = ranks[i];
                insert(tree, size, rank);
                if (rank < min_rank) min_rank = rank;
                if (rank > max_rank) max_rank = rank;
                ++total;
            }
        }
    }
    free(tree);
    return 0;
}
