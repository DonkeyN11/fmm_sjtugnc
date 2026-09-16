/**
 * Tests for the covariance validity pair in CMM: is_covariance_usable() and
 * fallback_covariance().
 *
 * Background. CMM used to carry a hardcoded MIN_SIGMA floor: any covariance
 * whose smaller standard deviation fell below 5.0e-5 degrees was rescaled up
 * until it reached the floor. On the Haikou set that fired on 99.98 % of epochs
 * with a median rescale factor of 7.22x, so the covariance the receiver
 * reported was almost never the covariance the algorithm used -- and because
 * the direction penalty is inversely proportional to sigma, that factor of
 * 7.22 propagated into the transition model too.
 *
 * The floor is gone. A covariance is now used exactly as given when it is a
 * usable 2x2 covariance, and replaced by an isotropic 5 m fallback when it is
 * not. The two tests below pin both halves, and in particular pin that the
 * real-data covariance which the floor used to reject is now accepted
 * unchanged -- that is the regression this file exists for.
 *
 * @author: Chenzhang Ning
 */

#include "mm/cmm/cmm_algorithm.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

using FMM::MM::CovarianceMapMatch;
using FMM::MM::CovarianceMatrix;

static int failures = 0;

static void check(bool got, bool want, const char *what) {
    if (got == want) {
        std::printf("  ok   %s -> %s\n", what, got ? "usable" : "rejected");
        return;
    }
    std::printf("  FAIL %s -> got %s, want %s\n", what, got ? "usable" : "rejected",
                want ? "usable" : "rejected");
    ++failures;
}

static void check_close(double got, double want, double tol, const char *what) {
    if (std::fabs(got - want) <= tol) {
        std::printf("  ok   %s\n", what);
        return;
    }
    std::printf("  FAIL %s -> got %.17g, want %.17g (tol %.3g)\n", what, got, want, tol);
    ++failures;
}

// Field order is {sde, sdn, sdu, sdne, sdeu, sdun}.
static CovarianceMatrix cov(double sde, double sdn, double sdu, double sdne) {
    return CovarianceMatrix{sde, sdn, sdu, sdne, 0.0, 0.0};
}

static constexpr double MIN_SIGMA_OLD = 5.0e-5;  // the deleted floor, in degrees

int main() {
    std::printf("is_covariance_usable\n");

    // The regression. This is a real row of data/real_vehicle/hainan_06/
    // cmm_input_points.csv, and its horizontal sigma (0.72 m) sits a factor of
    // ~7 below the old floor, so the floor rewrote it. It must now pass
    // untouched. If this case ever fails, a scale-to-a-floor mechanism has
    // come back in some other guise.
    const CovarianceMatrix real = cov(6.4850815946669765e-06, 6.256475451009389e-06,
                                      0.8078, 2.8415502005447694e-12);
    check(CovarianceMapMatch::is_covariance_usable(real), true,
          "real Haikou row, sigma ~0.72 m (the floor used to rescale this)");
    check(real.sde < MIN_SIGMA_OLD, true, "  ...and it really is below the old floor");

    // A well-formed covariance of any size is fine. The large case guards the
    // other direction: nothing may impose an upper bound either. 2 m / 0.5 m is
    // the anisotropic shape from example/cmm_example.cpp.
    check(CovarianceMapMatch::is_covariance_usable(cov(2.0, 0.5, 2.0, 0.1)), true,
          "large anisotropic covariance (2.0, 0.5, sdne 0.1)");
    check(CovarianceMapMatch::is_covariance_usable(cov(7.0e-6, 7.0e-6, 7.0e-6, 0.0)), true,
          "small isotropic covariance");

    // Exactly singular. sde * sdn == |sdne| is the Cauchy-Schwarz boundary and
    // det is exactly zero there, so the strict '>' is what separates this from
    // the next case. A '<=' or '>=' would let a non-invertible matrix through,
    // and Matrix2d::inverse() answers a singular input with the zero matrix
    // rather than failing, which silently turns every Mahalanobis distance into
    // zero.
    check(CovarianceMapMatch::is_covariance_usable(cov(1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3)),
          false, "exactly singular (det == 0)");

    // Beyond the boundary: no covariance matrix can have this correlation.
    check(CovarianceMapMatch::is_covariance_usable(cov(1.0e-3, 1.0e-3, 1.0e-3, 1.1e-3)),
          false, "correlation beyond Cauchy-Schwarz (det < 0)");

    // Degenerate standard deviations. Zero is rejected because it makes the
    // matrix singular; a negative standard deviation is not a standard
    // deviation at all, and squaring it would hide the sign.
    check(CovarianceMapMatch::is_covariance_usable(cov(0.0, 1.0e-5, 1.0e-5, 0.0)), false,
          "sde == 0");
    check(CovarianceMapMatch::is_covariance_usable(cov(1.0e-5, 0.0, 1.0e-5, 0.0)), false,
          "sdn == 0");
    check(CovarianceMapMatch::is_covariance_usable(cov(-1.0e-5, 1.0e-5, 1.0e-5, 0.0)),
          false, "sde negative");
    check(CovarianceMapMatch::is_covariance_usable(cov(1.0e-5, -1.0e-5, 1.0e-5, 0.0)),
          false, "sdn negative");

    // Non-finite inputs. NaN is the one that matters: every comparison against
    // NaN is false, so a test written as `if (sde > 0 && ...)` would return
    // false here anyway, but a test written as `if (sde < 0) reject` would not.
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    check(CovarianceMapMatch::is_covariance_usable(cov(nan, 1.0e-5, 1.0e-5, 0.0)), false,
          "sde NaN");
    check(CovarianceMapMatch::is_covariance_usable(cov(1.0e-5, nan, 1.0e-5, 0.0)), false,
          "sdn NaN");
    check(CovarianceMapMatch::is_covariance_usable(cov(1.0e-5, 1.0e-5, 1.0e-5, nan)), false,
          "sdne NaN");
    check(CovarianceMapMatch::is_covariance_usable(cov(inf, 1.0e-5, 1.0e-5, 0.0)), false,
          "sde +inf");
    check(CovarianceMapMatch::is_covariance_usable(cov(1.0e-5, 1.0e-5, 1.0e-5, inf)), false,
          "sdne +inf");

    // sdu / sdeu / sdun are never read by the 2D model, so a bad vertical
    // standard deviation must not disqualify an otherwise good horizontal
    // covariance.
    check(CovarianceMapMatch::is_covariance_usable(
              CovarianceMatrix{7.0e-6, 7.0e-6, nan, 0.0, nan, nan}),
          true, "horizontal fine, vertical NaN (unused)");

    std::printf("\nfallback_covariance\n");

    constexpr double FALLBACK_M = 5.0;
    constexpr double M_PER_DEG_LAT = 111132.0;

    // Projected network: the fallback is the isotropic 5 m directly, since the
    // network's own units are already metres.
    const CovarianceMatrix proj =
        CovarianceMapMatch::fallback_covariance(true, 19.98);
    check_close(proj.sde, FALLBACK_M, 0.0, "projected: sde == 5.0 exactly");
    check_close(proj.sdn, FALLBACK_M, 0.0, "projected: sdn == 5.0 exactly");
    check_close(proj.sdu, FALLBACK_M, 0.0, "projected: sdu == 5.0 exactly");
    check_close(proj.sdne, 0.0, 0.0, "projected: sdne == 0 (isotropic)");

    // Geographic network: the same 5 m expressed in degrees. Longitude degrees
    // are shorter than latitude degrees by cos(latitude), so the east sigma is
    // the larger number.
    const double lat = 19.98;
    const CovarianceMatrix geo = CovarianceMapMatch::fallback_covariance(false, lat);
    const double cos_lat = std::cos(lat * M_PI / 180.0);
    check_close(geo.sdn, FALLBACK_M / M_PER_DEG_LAT, 0.0, "geographic: sdn == 5/111132");
    check_close(geo.sde, FALLBACK_M / (M_PER_DEG_LAT * cos_lat), 1e-18,
                "geographic: sde == 5/(111132 cos lat)");
    check_close(geo.sdne, 0.0, 0.0, "geographic: sdne == 0 (isotropic)");
    check_close(geo.sdu, FALLBACK_M / M_PER_DEG_LAT, 0.0, "geographic: sdu == sdn");

    // The on-ground shape must be a circle: the east sigma times a degree of
    // longitude must equal the north sigma times a degree of latitude. This is
    // the property that fails if someone "simplifies" the cos(latitude) away.
    check_close(geo.sde * M_PER_DEG_LAT * cos_lat, geo.sdn * M_PER_DEG_LAT, 1e-9,
                "geographic: 5 m in both directions on the ground");

    // At the equator the two are equal.
    const CovarianceMatrix eq = CovarianceMapMatch::fallback_covariance(false, 0.0);
    check_close(eq.sde, eq.sdn, 1e-15, "geographic lat 0: sde == sdn");
    check_close(eq.sde, FALLBACK_M / M_PER_DEG_LAT, 1e-15, "geographic lat 0: == 5 m in degrees");

    // Cos is even, so the southern hemisphere must match.
    const CovarianceMatrix south = CovarianceMapMatch::fallback_covariance(false, -lat);
    check_close(south.sde, geo.sde, 0.0, "geographic: sde symmetric in latitude");

    // At the pole a degree of longitude is zero length and 5 m cannot be
    // written down in degrees. The clamped cosine keeps the result finite; the
    // alternative (dividing by cos -> 0) yields inf, which would then fail
    // is_covariance_usable and cascade.
    for (double pole_lat : {89.99, 90.0, 90.0 + 1e-9}) {
        const CovarianceMatrix p = CovarianceMapMatch::fallback_covariance(false, pole_lat);
        const bool finite = std::isfinite(p.sde) && std::isfinite(p.sdn);
        check(finite, true, "geographic near the pole stays finite");
    }

    // The invariant that ties the two functions together: whatever the fallback
    // produces must itself be usable. If this broke, the algorithm would
    // substitute an unusable covariance for an unusable covariance and the
    // validity test would have no repair path.
    int unusable = 0;
    for (double l : {-89.99, -45.0, -0.001, 0.0, 19.98, 45.0, 60.0, 89.99, 90.0}) {
        if (!CovarianceMapMatch::is_covariance_usable(
                CovarianceMapMatch::fallback_covariance(false, l))) {
            ++unusable;
        }
    }
    check(unusable == 0, true, "fallback_covariance always satisfies is_covariance_usable");
    check(CovarianceMapMatch::is_covariance_usable(
              CovarianceMapMatch::fallback_covariance(true, 0.0)),
          true, "projected fallback satisfies is_covariance_usable");

    if (failures != 0) {
        std::printf("\nFAILED (%d)\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("\nPASSED\n");
    return EXIT_SUCCESS;
}
