/**
 * Tests for the two ablation-only knobs on CovarianceMapMatchConfig.
 *
 * Background. The TITS review asked for the calibration gain over the HMM
 * baseline to be attributed to individual components. That needs a factorial
 * ablation over four mechanisms, two of which had no way to be varied: the
 * search radius was hard-wired to r_i = HPL_i, and the emission shape was
 * whatever the input covariance was.
 *
 * These two knobs exist to make those two factors separable. They are NOT part
 * of the paper's Section III model, they default to 0.0 (= off, i.e. the
 * production behaviour), and a non-zero value logs a WARN on construction.
 *
 * What this file pins:
 *   1. the defaults, because "off" is what keeps every existing result valid;
 *   2. the radius override, including that a negative value means "off" rather
 *      than being passed through to the search;
 *   3. the isotropic sigma definition -- equal TOTAL VARIANCE, so that the
 *      isotropic and anisotropic arms of the ablation differ in the shape of
 *      the error ellipse and not in its size;
 *   4. validate() rejecting negative values, so a typo cannot silently select
 *      the modelled behaviour while looking like a configured ablation.
 *
 * @author: Chenzhang Ning
 */

#include "mm/cmm/cmm_algorithm.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

using FMM::MM::CovarianceMapMatch;
using FMM::MM::CovarianceMapMatchConfig;
using FMM::MM::CovarianceMatrix;

static int failures = 0;

static void check_close(double got, double want, double tol, const char *what) {
    if (std::fabs(got - want) <= tol) {
        std::printf("  ok   %s -> %.10g\n", what, got);
        return;
    }
    std::printf("  FAIL %s -> got %.10g, want %.10g\n", what, got, want);
    ++failures;
}

static void check(bool got, bool want, const char *what) {
    if (got == want) {
        std::printf("  ok   %s -> %s\n", what, got ? "true" : "false");
        return;
    }
    std::printf("  FAIL %s -> got %s, want %s\n", what, got ? "true" : "false",
                want ? "true" : "false");
    ++failures;
}

// A covariance with the given horizontal standard deviations, no cross term.
// Only sde, sdn and sdne reach the 2x2 model.
static CovarianceMatrix cov(double sde, double sdn, double sdne = 0.0) {
    CovarianceMatrix c;
    c.sde = sde;
    c.sdn = sdn;
    c.sdu = 0.0;
    c.sdne = sdne;
    c.sdeu = 0.0;
    c.sdun = 0.0;
    return c;
}

int main() {
    std::printf("CovarianceMapMatchConfig search-radius and ablation knobs\n\n");

    // ---------------------------------------------------------------------
    // 1. Defaults. k_search = 1 with delta_road = 0 gives r_i = HPL_i, which is
    //    the behaviour every reported number was produced under; the isotropic
    //    knob must stay off for the same reason. If any of these defaults ever
    //    changes, the project's existing results stop meaning what they say.
    // ---------------------------------------------------------------------
    std::printf("defaults\n");
    CovarianceMapMatchConfig def;
    check_close(def.k_search, 1.0, 0.0, "default k_search is 1.0");
    check_close(def.delta_road, 0.0, 0.0, "default delta_road is 0.0");
    check_close(def.isotropic_emission_scale, 0.0, 0.0,
                "default isotropic_emission_scale is 0.0");
    check(def.validate(), true, "default config validates");

    // ---------------------------------------------------------------------
    // 2. effective_search_radius: r_i = k_search * HPL_i + delta_road.
    // ---------------------------------------------------------------------
    std::printf("\neffective_search_radius\n");
    const double PL = 2.0459e-4;

    // The identity case. This is the one that must not drift: it is what ties
    // the default configuration to the results produced before the coefficients
    // existed.
    check_close(CovarianceMapMatch::effective_search_radius(PL, 1.0, 0.0),
                PL, 0.0, "k=1, delta=0 -> exactly the protection level");

    check_close(CovarianceMapMatch::effective_search_radius(PL, 2.0, 0.0),
                2.0 * PL, 0.0, "k=2 -> twice the protection level");

    // delta_road is an offset in network units, added after the scaling, so the
    // constant-radius arm of the ablation is k_search = 0 with delta_road = r.
    const double R_CONST = 2.0459e-4;
    check_close(CovarianceMapMatch::effective_search_radius(PL, 0.0, R_CONST),
                R_CONST, 0.0,
                "k=0 with a delta -> a radius independent of the epoch");
    check_close(CovarianceMapMatch::effective_search_radius(0.5 * PL, 0.0, R_CONST),
                R_CONST, 0.0,
                "the constant holds for a different protection level too");

    // Scaling then offsetting, in that order. Reversing them would give
    // k*(PL + delta), which is a different radius whenever k != 1.
    check_close(CovarianceMapMatch::effective_search_radius(PL, 2.0, 0.01),
                2.0 * PL + 0.01, 1e-18, "scale first, then offset");

    // A zero protection level with no offset gives a zero radius. The search
    // then admits nothing and the epoch is dropped, which is the documented
    // response to an epoch the network cannot explain.
    check_close(CovarianceMapMatch::effective_search_radius(0.0, 1.0, 0.0),
                0.0, 0.0, "PL 0, no offset -> 0");

    // ---------------------------------------------------------------------
    // 3. isotropic_sigma. Equal total variance: sde^2 + sdn^2 is preserved, so
    //    the ablation's emission factor changes the ellipse's SHAPE only. If
    //    this were defined as, say, sigma = sde, the isotropic arm would also
    //    change how much probability mass the emission lays down, and the two
    //    factors would no longer be separable.
    // ---------------------------------------------------------------------
    std::printf("\nisotropic_sigma\n");
    check(CovarianceMapMatch::isotropic_sigma(cov(0.6, 0.8), 0.0) < 0.0, true,
          "scale 0 -> knob off");
    check(CovarianceMapMatch::isotropic_sigma(cov(0.6, 0.8), -2.0) < 0.0, true,
          "negative scale -> knob off");

    // A circle must stay a circle.
    check_close(CovarianceMapMatch::isotropic_sigma(cov(0.5, 0.5), 1.0),
                0.5, 1e-15, "circle in -> same sigma out");

    // An ellipse at scale 1: sqrt((0.36 + 0.64) / 2) = sqrt(0.5).
    const double expected = std::sqrt(0.5);
    check_close(CovarianceMapMatch::isotropic_sigma(cov(0.6, 0.8), 1.0),
                expected, 1e-15, "ellipse 0.6/0.8 -> sqrt(0.5)");

    // The invariant itself, stated directly, so that a future rewrite of the
    // formula has to keep the quantity the ablation actually relies on. The
    // tolerance is relative because the two sides reach the same number by
    // different routes (sqrt then square, versus square then multiply) and
    // differ in the last bits.
    for (double scale : {0.25, 1.0, 4.0}) {
        const double s = CovarianceMapMatch::isotropic_sigma(cov(0.6, 0.8), scale);
        const double want = scale * scale * (0.6 * 0.6 + 0.8 * 0.8);
        check_close(2.0 * s * s, want, 1e-12 * want,
                    "total variance scales as scale^2");
    }

    // AM-GM: the isotropic sigma is never below the geometric mean of sde and
    // sdn. That is why the isotropic arm also gets a weaker direction penalty
    // (kappa = speed^2 / (sde * sdn)) -- a consequence the ablation has to
    // report, not a bug.
    {
        const double s = CovarianceMapMatch::isotropic_sigma(cov(0.3, 1.2), 1.0);
        check(s * s >= 0.3 * 1.2, true,
              "sigma^2 >= sde*sdn (AM-GM): isotropic kappa is the weaker one");
    }

    // A degenerate covariance must not produce a sigma. is_covariance_usable()
    // would already have replaced it upstream; this is the second line of
    // defence, because the knob is reachable from the Python binding and from
    // example/cmm_example.cpp, which do not call validate().
    check(CovarianceMapMatch::isotropic_sigma(cov(0.0, 0.8), 1.0) < 0.0, true,
          "sde 0 -> no sigma invented");
    check(CovarianceMapMatch::isotropic_sigma(cov(0.6, 0.0), 1.0) < 0.0, true,
          "sdn 0 -> no sigma invented");

    // ---------------------------------------------------------------------
    // 4. validate(). A negative coefficient would either invert the radius or
    //    silently mean "off", so it must be rejected rather than reinterpreted.
    //    NaN matters here for the same reason: NaN < 0 is false, so a plain
    //    less-than test lets it through, and effective_search_radius would then
    //    return a NaN radius and search nowhere without saying so. The negated
    //    comparisons catch both.
    // ---------------------------------------------------------------------
    std::printf("\nvalidate\n");
    CovarianceMapMatchConfig bad_k;
    bad_k.k_search = -1.0;
    check(bad_k.validate(), false, "negative k_search rejected");

    CovarianceMapMatchConfig bad_delta;
    bad_delta.delta_road = -1.0;
    check(bad_delta.validate(), false, "negative delta_road rejected");

    CovarianceMapMatchConfig nan_k;
    nan_k.k_search = std::numeric_limits<double>::quiet_NaN();
    check(nan_k.validate(), false, "NaN k_search rejected");

    CovarianceMapMatchConfig nan_delta;
    nan_delta.delta_road = std::numeric_limits<double>::quiet_NaN();
    check(nan_delta.validate(), false, "NaN delta_road rejected");

    CovarianceMapMatchConfig bad_scale;
    bad_scale.isotropic_emission_scale = -1.0;
    check(bad_scale.validate(), false, "negative isotropic_emission_scale rejected");

    CovarianceMapMatchConfig both_on;
    both_on.k_search = 1.0;
    both_on.delta_road = 1.4e-4;
    both_on.isotropic_emission_scale = 1.0;
    check(both_on.validate(), true, "the ablation arms are valid configs");

    if (failures != 0) {
        std::printf("FAILED (%d)\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("PASSED\n");
    return EXIT_SUCCESS;
}
