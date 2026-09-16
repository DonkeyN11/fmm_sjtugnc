/**
 * Regression test for the sub-segment restart decision in CMM.
 *
 * Background. When an epoch fails to connect to the current sub-segment, CMM
 * either restarts (commit the sub-segment, re-seed the HMM at this epoch) or
 * carries the epoch as a gap and retries it later as a long jump. The restart
 * path used to be gated on `current_sub_indices.size() >= 2`, which meant a
 * sub-segment that failed on its very first transition could never restart:
 * the size cannot grow while the sub-segment is stuck, so every following
 * epoch was carried until the 180 s max_interval escape -- even epochs that
 * connected to each other perfectly well. On the Haikou set at a tight
 * protection level this dropped 167 recoverable epochs (traj 11: 23, traj 21:
 * 144) and cost 1.37 pp of segment accuracy (93.21 % -> 94.58 %).
 *
 * The sub-segment size must therefore NOT gate the restart. These cases fail
 * to compile or to pass if that condition is reintroduced.
 *
 * @author: Chenzhang Ning
 */

#include "mm/cmm/cmm_algorithm.hpp"

#include <cstdio>
#include <cstdlib>

using FMM::MM::CovarianceMapMatch;
using FMM::MM::CovarianceMapMatchConfig;

static int failures = 0;

static void check(bool got, bool want, const char *what) {
    if (got == want) {
        std::printf("  ok   %s -> %s\n", what, got ? "restart" : "carry");
        return;
    }
    std::printf("  FAIL %s -> got %s, want %s\n", what, got ? "restart" : "carry",
                want ? "restart" : "carry");
    ++failures;
}

int main() {
    std::printf("should_restart_sub_segment\n");

    // The regression: a sub-segment of any size may restart. There is no size
    // argument to pass, by design -- if one is reintroduced this file stops
    // compiling, which is the point.
    check(CovarianceMapMatch::should_restart_sub_segment(/*enable_gap_bridging=*/true,
                                                         /*next_epoch_has_candidates=*/true),
          true,
          "gap bridging on, epoch has candidates");

    // An epoch with no road candidate cannot seed a sub-segment: its layer
    // would be empty. It has to be carried and retried instead.
    check(CovarianceMapMatch::should_restart_sub_segment(true, false), false,
          "gap bridging on, epoch has no candidates");

    // Splitting is what enable_gap_bridging gates. With it off the whole
    // trajectory stays one sub-segment, so nothing may restart.
    check(CovarianceMapMatch::should_restart_sub_segment(false, true), false,
          "gap bridging off, epoch has candidates");
    check(CovarianceMapMatch::should_restart_sub_segment(false, false), false,
          "gap bridging off, epoch has no candidates");

    // The configuration default must keep the recovery path open.
    CovarianceMapMatchConfig config;
    if (!config.enable_gap_bridging) {
        std::printf("  FAIL default enable_gap_bridging is false; a default run "
                    "could not recover from a disconnected epoch\n");
        ++failures;
    } else {
        std::printf("  ok   default config enables gap bridging\n");
    }

    if (failures != 0) {
        std::printf("FAILED (%d)\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("PASSED\n");
    return EXIT_SUCCESS;
}
