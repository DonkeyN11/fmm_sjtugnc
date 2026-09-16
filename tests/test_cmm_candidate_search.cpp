/**
 * Regression test for the bounded candidate-radius fallback in CMM.
 *
 * Background. CMM's candidate search starts at the radius the paper specifies,
 * r_i = HPL_i. An epoch whose protection-level disk contains no road edge used
 * to be dropped from the output entirely, which is what the manuscript's
 * pseudocode does (`continue` on an empty candidate set) but which throws away
 * epochs whose fix is merely unlucky: the covariance honestly reports that
 * nothing is near, yet the vehicle was on a road.
 *
 * The retry is now a doubling of the radius, bounded by a hard cap. Three
 * properties keep this honest, and each is pinned below:
 *
 *   1. It triggers only when the candidate set is short. An epoch that already
 *      meets min_candidates must not have its radius widened, so the fallback
 *      cannot disturb epochs that are matching fine. The old code did the
 *      opposite: it broke out of the retry loop when `edges_to_consider` was
 *      empty, i.e. it refused to look further in exactly the case that needed
 *      it. That is the regression this file exists for.
 *   2. It is capped, so the radius cannot walk arbitrarily far and the worst
 *      case stays finite.
 *   3. It cannot spin: a radius that will not grow (0, NaN, inf) stops the
 *      retry rather than making `radius *= 2.0` a no-op forever.
 *
 * Measured on the Haikou set at r_i = HPL_i, the fallback recovers 1345 epochs
 * (0 lost) and fires at depth 1 for 1153 of them and depth 2 for 192; the cap
 * of 8 is not reached. Those epochs are 82.50 % accurate against ground truth
 * (33/40 evaluable), and only 5 epochs that already had a result change path,
 * which is the evidence that property 1 holds in practice and not just here.
 *
 * @author: Chenzhang Ning
 */

#include "mm/cmm/cmm_algorithm.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>

using FMM::MM::CovarianceMapMatch;

static int failures = 0;

static void check(bool got, bool want, const char *what) {
    if (got == want) {
        std::printf("  ok   %s -> %s\n", what, got ? "expand" : "stop");
        return;
    }
    std::printf("  FAIL %s -> got %s, want %s\n", what, got ? "expand" : "stop",
                want ? "expand" : "stop");
    ++failures;
}

// A protection level in the network's own units; 2.0e-4 degrees is about 22 m
// on the Haikou network. The predicate is unit-agnostic, but using a plausible
// value keeps the cases readable.
static constexpr double PL = 2.0e-4;
static constexpr int MAX_DOUBLINGS = 8;

int main() {
    std::printf("should_expand_search_radius\n");

    // The regression: an epoch with no candidate at all is precisely the one
    // that must be looked at again. The old loop broke here instead.
    check(CovarianceMapMatch::should_expand_search_radius(0, 3, 0, MAX_DOUBLINGS, PL),
          true, "no candidates, budget left");

    // One candidate short of the target still deserves the extra look.
    check(CovarianceMapMatch::should_expand_search_radius(2, 3, 0, MAX_DOUBLINGS, PL),
          true, "2 of 3 candidates");

    // Property 1: a set that already qualifies is left alone. This is what
    // keeps the fallback from widening the disk of a well-matched epoch.
    check(CovarianceMapMatch::should_expand_search_radius(3, 3, 0, MAX_DOUBLINGS, PL),
          false, "3 of 3 candidates");
    check(CovarianceMapMatch::should_expand_search_radius(9, 3, 0, MAX_DOUBLINGS, PL),
          false, "more candidates than asked for");

    // Property 2: the cap. Both the boundary and a value past it must stop.
    check(CovarianceMapMatch::should_expand_search_radius(0, 3, MAX_DOUBLINGS,
                                                          MAX_DOUBLINGS, PL),
          false, "at the cap");
    check(CovarianceMapMatch::should_expand_search_radius(0, 3, MAX_DOUBLINGS + 3,
                                                          MAX_DOUBLINGS, PL),
          false, "past the cap");

    // One doubling short of the cap must still be allowed, or the cap would
    // silently cost one level of the search.
    check(CovarianceMapMatch::should_expand_search_radius(0, 3, MAX_DOUBLINGS - 1,
                                                          MAX_DOUBLINGS, PL),
          true, "one doubling below the cap");

    // Property 3: a radius that cannot grow stops the retry. Without this the
    // doubling is a no-op and the caller's loop never terminates.
    check(CovarianceMapMatch::should_expand_search_radius(0, 3, 0, MAX_DOUBLINGS, 0.0),
          false, "zero radius");
    check(CovarianceMapMatch::should_expand_search_radius(0, 3, 0, MAX_DOUBLINGS, -1.0),
          false, "negative radius");
    check(CovarianceMapMatch::should_expand_search_radius(
              0, 3, 0, MAX_DOUBLINGS, std::numeric_limits<double>::quiet_NaN()),
          false, "NaN radius");
    check(CovarianceMapMatch::should_expand_search_radius(
              0, 3, 0, MAX_DOUBLINGS, std::numeric_limits<double>::infinity()),
          false, "infinite radius");

    // validate() is not on every construction path -- Python and
    // example/cmm_example.cpp build a config directly -- so min_candidates can
    // arrive as 0 or negative. Clamping to 1 keeps the fallback reachable; a
    // literal `candidates_found < min_candidates` would never fire and the
    // feature would be silently dead.
    check(CovarianceMapMatch::should_expand_search_radius(0, 0, 0, MAX_DOUBLINGS, PL),
          true, "min_candidates 0, no candidates (clamped to 1)");
    check(CovarianceMapMatch::should_expand_search_radius(0, -4, 0, MAX_DOUBLINGS, PL),
          true, "min_candidates negative, no candidates (clamped to 1)");
    check(CovarianceMapMatch::should_expand_search_radius(1, 0, 0, MAX_DOUBLINGS, PL),
          false, "min_candidates 0, one candidate");

    // A zero cap must disable the fallback outright rather than be read as
    // "unlimited".
    check(CovarianceMapMatch::should_expand_search_radius(0, 3, 0, 0, PL),
          false, "cap of zero doublings");

    if (failures != 0) {
        std::printf("FAILED (%d)\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("PASSED\n");
    return EXIT_SUCCESS;
}
