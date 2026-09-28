/**
 * Tests for the map-registration term in the emission: sigma_m_to_covariance()
 * and emission_covariances().
 *
 * Background. The residual d = z_i - x_ij is not the GNSS error alone. It also
 * carries the map's own error -- true road position against digitised centreline
 * -- plus the vehicle's lateral offset within the carriageway. The candidate
 * SEARCH already allowed for both: that is what delta_road = 13.5 m is, and the
 * manuscript's own decomposition of it is 3.5 m of lane width plus a 10 m map
 * allowance. The EMISSION did not, so Sigma_i claimed the residual was the
 * reported covariance when it was not. Measured on the Haikou set, correcting
 * that improved ECE, Brier skill, AUC, separation and accuracy at the same time.
 *
 * Two invariants are pinned here, and neither is visible from a single run:
 *
 * 1. The metres-to-network-units conversion. One degree of longitude is shorter
 *    than one degree of latitude by cos(latitude), so an isotropic metric sigma
 *    is NOT an isotropic matrix in degrees. Getting this wrong yields an ellipse
 *    whose axis ratio is cos(latitude) -- 6 % at Haikou, unbounded at the pole.
 *
 * 2. The direction penalty must NOT see the map term. Its concentration
 *    kappa = |v|^2 / (sde * sdn) is a heading uncertainty, and a misregistered
 *    map moves the road rather than the vehicle's heading. If the term leaked in,
 *    kappa would be attenuated by roughly (0.8/10)^2 -- switching the direction
 *    penalty off without touching its flag -- and the emission change could not
 *    be attributed separately.
 *
 * @author: Chenzhang Ning
 */

#include "mm/cmm/cmm_algorithm.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

using FMM::MM::CovarianceMapMatch;
using FMM::MM::CovarianceMatrix;

static int failures = 0;

static void check_close(double got, double want, double tol, const char *what) {
    if (std::fabs(got - want) <= tol) {
        std::printf("  ok   %s\n", what);
        return;
    }
    std::printf("  FAIL %s -> got %.17g, want %.17g (tol %.3g)\n", what, got, want, tol);
    ++failures;
}

static void check(bool got, bool want, const char *what) {
    if (got == want) {
        std::printf("  ok   %s\n", what);
        return;
    }
    std::printf("  FAIL %s -> got %s, want %s\n", what, got ? "true" : "false",
                want ? "true" : "false");
    ++failures;
}

// Field order is {sde, sdn, sdu, sdne, sdeu, sdun}.
static CovarianceMatrix cov(double sde, double sdn, double sdu, double sdne) {
    return CovarianceMatrix{sde, sdn, sdu, sdne, 0.0, 0.0};
}

static bool same(const CovarianceMatrix &a, const CovarianceMatrix &b) {
    return a.sde == b.sde && a.sdn == b.sdn && a.sdu == b.sdu &&
           a.sdne == b.sdne && a.sdeu == b.sdeu && a.sdun == b.sdun;
}

static constexpr double M_PER_DEG_LAT = 111132.0;

int main() {
    std::printf("sigma_m_to_covariance\n");

    // Projected network: the network's units ARE metres, so the conversion is
    // the identity and no latitude may enter.
    const CovarianceMatrix proj =
        CovarianceMapMatch::sigma_m_to_covariance(10.0, true, 19.98);
    check_close(proj.sde, 10.0, 0.0, "projected: sde == sigma_m exactly");
    check_close(proj.sdn, 10.0, 0.0, "projected: sdn == sigma_m exactly");
    check_close(proj.sdne, 0.0, 0.0, "projected: sdne == 0");
    // Latitude must be ignored entirely on a projected network. If the cosine
    // crept into this branch the two results would differ by cos(19.98).
    check(same(proj, CovarianceMapMatch::sigma_m_to_covariance(10.0, true, 0.0)), true,
          "projected: independent of latitude");

    // Geographic network: degrees, and the east axis divides again by cos(lat).
    const double lat = 19.98;
    const double cos_lat = std::cos(lat * M_PI / 180.0);
    const CovarianceMatrix geo =
        CovarianceMapMatch::sigma_m_to_covariance(10.0, false, lat);
    check_close(geo.sdn, 10.0 / M_PER_DEG_LAT, 0.0, "geographic: sdn == s/111132");
    check_close(geo.sde, 10.0 / (M_PER_DEG_LAT * cos_lat), 1e-18,
                "geographic: sde == s/(111132 cos lat)");
    check_close(geo.sdne, 0.0, 0.0, "geographic: sdne == 0");

    // The property that fails if someone "simplifies" the cosine away: the
    // shape on the ground must be a circle, not an ellipse.
    check_close(geo.sde * M_PER_DEG_LAT * cos_lat, geo.sdn * M_PER_DEG_LAT, 1e-9,
                "geographic: circle on the ground, not an ellipse");
    check(geo.sde > geo.sdn, true, "geographic: east sigma is the larger number");

    // Latitude 0 is the one place the two axes coincide.
    check_close(CovarianceMapMatch::sigma_m_to_covariance(10.0, false, 0.0).sde,
                10.0 / M_PER_DEG_LAT, 1e-15, "geographic lat 0: sde == sdn");

    // cos is even, so the southern hemisphere matches.
    check_close(CovarianceMapMatch::sigma_m_to_covariance(10.0, false, -lat).sde,
                geo.sde, 0.0, "geographic: sde symmetric in latitude");

    // At the pole a degree of longitude is zero length. The clamped cosine keeps
    // the result finite; dividing by cos -> 0 would yield inf, which would then
    // fail is_covariance_usable and cascade into the fallback path.
    for (double pole : {89.99, 90.0, 90.0 + 1e-9}) {
        const CovarianceMatrix p = CovarianceMapMatch::sigma_m_to_covariance(10.0, false, pole);
        check(std::isfinite(p.sde) && std::isfinite(p.sdn), true,
              "geographic near the pole stays finite");
    }

    // The shared helper must keep fallback_covariance() bit-for-bit: the
    // fallback was refactored to call this function, and any drift here would
    // silently move the covariance of every epoch that reaches it.
    constexpr double FALLBACK_M = 5.0;
    check(same(CovarianceMapMatch::fallback_covariance(false, lat),
               CovarianceMapMatch::sigma_m_to_covariance(FALLBACK_M, false, lat)),
          true, "fallback_covariance(false, lat) == sigma_m_to_covariance(5, false, lat)");
    check(same(CovarianceMapMatch::fallback_covariance(true, lat),
               CovarianceMapMatch::sigma_m_to_covariance(FALLBACK_M, true, lat)),
          true, "fallback_covariance(true, lat) == sigma_m_to_covariance(5, true, lat)");

    std::printf("\nemission_covariances\n");

    // A real Haikou row: anisotropic, in degrees, with the cross term.
    const CovarianceMatrix real = cov(6.4850815946669765e-06, 6.256475451009389e-06,
                                      0.8078, 2.8415502005447694e-12);

    // --- invariant 1: sigma <= 0 leaves BOTH fields untouched ----------------
    for (double off : {0.0, -1.0, -1e-12}) {
        const auto ec = CovarianceMapMatch::emission_covariances(real, off, false, lat);
        check(same(ec.emission, real) && same(ec.direction, real), true,
              "sigma <= 0 leaves emission and direction identical to input");
    }
    // A negative value must behave as off, not as |sigma|: the constructor
    // clamps it, and this function must not be the place where it gets used raw.
    check(same(CovarianceMapMatch::emission_covariances(real, -10.0, false, lat).emission,
               real),
          true, "negative sigma is inert (not treated as |sigma|)");

    // --- invariant 2: the direction penalty NEVER sees the map term ----------
    // This is the whole point of the split. It must hold at every sigma.
    for (double s : {0.0, 0.1, 3.5, 10.0, 10.6, 13.5, 1000.0}) {
        const auto ec = CovarianceMapMatch::emission_covariances(real, s, false, lat);
        check(same(ec.direction, real), true,
              "direction == reported covariance, whatever the map sigma");
    }
    // And the consequence that matters numerically: kappa = |v|^2/(sde*sdn) is
    // formed from the direction covariance, so it must not move. If the map term
    // leaked in, this ratio would collapse by ~156x at sigma = 10.6.
    {
        const auto ec = CovarianceMapMatch::emission_covariances(real, 10.6, false, lat);
        const double kappa_dir = ec.direction.sde * ec.direction.sdn;
        const double kappa_reported = real.sde * real.sdn;
        check_close(kappa_dir, kappa_reported, 0.0,
                    "kappa denominator unchanged by the map term");
        const double kappa_if_leaked = ec.emission.sde * ec.emission.sdn;
        check(kappa_if_leaked > 100.0 * kappa_dir, true,
              "  ...and it WOULD have changed: leaking gives >100x the denominator");
    }

    // --- invariant 3: the emission term is quadrature, not substitution ------
    {
        const auto ec = CovarianceMapMatch::emission_covariances(real, 10.0, false, lat);
        const CovarianceMatrix m = CovarianceMapMatch::sigma_m_to_covariance(10.0, false, lat);
        check_close(ec.emission.sde, std::hypot(real.sde, m.sde), 0.0,
                    "emission sde = hypot(reported, map)");
        check_close(ec.emission.sdn, std::hypot(real.sdn, m.sdn), 0.0,
                    "emission sdn = hypot(reported, map)");
        // The cross term is uncorrelated with the map error, so it survives the
        // widening untouched. Zeroing or rescaling it would rotate the ellipse.
        check_close(ec.emission.sdne, real.sdne, 0.0,
                    "cross term sdne carried through unchanged");
        // The reported anisotropy is preserved, not replaced: at sigma = 10 m
        // the map term dominates, but the ratio must still be the reported one
        // once that term is removed again.
        const double ratio_emis = ec.emission.sde / ec.emission.sdn;
        const double ratio_reported = std::sqrt(
            (real.sde * real.sde + m.sde * m.sde) / (real.sdn * real.sdn + m.sdn * m.sdn));
        check_close(ratio_emis, ratio_reported, 1e-15, "anisotropy is additive, not replaced");
    }

    // A map term far larger than the reported covariance must dominate it, which
    // also re-checks the conversion end to end on the ground.
    //
    // The reported sigma does not vanish, so the result is 1000.00026 m and not
    // 1000: hypot(0.72, 1000). That is why the tolerance is 1e-3 rather than
    // exact, and why the strict '>' below is the assertion that separates adding
    // in quadrature from subtracting -- the two differ by only 5e-4 m here, far
    // inside a loose band.
    {
        const auto ec = CovarianceMapMatch::emission_covariances(real, 1000.0, false, lat);
        const double sde_m = ec.emission.sde * M_PER_DEG_LAT * cos_lat;
        const double sdn_m = ec.emission.sdn * M_PER_DEG_LAT;
        check_close(sde_m, 1000.0, 1e-3, "large map sigma: sde -> 1000 m on the ground");
        check_close(sdn_m, 1000.0, 1e-3, "large map sigma: sdn -> 1000 m on the ground");
        check(sde_m > 1000.0 && sdn_m > 1000.0, true,
              "  ...and strictly above it, so the term ADDS rather than subtracts");
    }

    // --- invariant 4: the result is always a usable covariance ---------------
    // The widened emission still has to pass the validity test, or the algorithm
    // would hand a singular matrix to Matrix2d::inverse(), which answers a
    // singular input with the zero matrix rather than failing.
    int unusable = 0;
    for (double s : {0.0, 1e-9, 0.1, 5.0, 10.6, 1e6}) {
        for (double l : {0.0, 19.98, -19.98, 89.99}) {
            const auto ec = CovarianceMapMatch::emission_covariances(real, s, false, l);
            if (!CovarianceMapMatch::is_covariance_usable(ec.emission)) ++unusable;
            if (!CovarianceMapMatch::is_covariance_usable(ec.direction)) ++unusable;
        }
    }
    check(unusable == 0, true, "every produced covariance satisfies is_covariance_usable");

    // The widening must never make the covariance smaller. A sign error that
    // subtracted instead of adding would still pass a naive "differs" test.
    for (double s : {0.1, 3.5, 10.6, 100.0}) {
        const auto ec = CovarianceMapMatch::emission_covariances(real, s, false, lat);
        check(ec.emission.sde > real.sde && ec.emission.sdn > real.sdn, true,
              "widening is monotone: emission sigma strictly exceeds reported");
    }

    if (failures != 0) {
        std::printf("\nFAILED (%d)\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("\nPASSED\n");
    return EXIT_SUCCESS;
}
