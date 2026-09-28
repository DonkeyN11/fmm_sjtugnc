/**
 * Tests for the direction-consistency penalty in CMM.
 *
 * Three things changed on 2026-09-27, all measured rather than assumed, and none of them
 * visible from a single run.
 *
 * 1. THE TANGENT IS LOCAL. It used to be the edge's start -> end chord. C1 and C2 are a
 *    pair: a hairpin edge whose return leg points opposite to its own chord. A vehicle
 *    driving the return leg is aligned with its own segment and anti-aligned with the
 *    chord, so the pair fails if the chord ever comes back.
 *
 *    Measured on the Haikou set this was not a corner case: on 白驹大道 (3.5 km) the
 *    chord differs from the local direction by enough to inflate the apparent heading
 *    residual at 25-35 m of travel from 7.82 deg to 0.72 deg, and the inflation grew
 *    with speed -- the opposite of the 1/|v| law the residual must follow.
 *
 * 2. KAPPA CARRIES THE INTER-EPOCH CORRELATION. kappa = |v|^2 / (c * n^T Sigma n) with
 *    c = 2(1-rho), not |v|^2 / (sde * sdn). Two corrections tested apart: the factor c
 *    (C3) and the perpendicular quadratic form (C4) -- a covariance stretched ALONG the
 *    direction of travel must not change the penalty, because it does not change the
 *    error ACROSS the travel direction that governs the heading, while sde * sdn does.
 *
 * 3. THE TERM IS ONE-SIDED BY DEFAULT, AND THAT IS A MEASURED CHOICE. The symmetric
 *    form, kappa(cos - 1), is the obvious way to give a diverged vehicle a low score,
 *    and it was implemented and measured. Against the one-sided term at the manuscript
 *    operating point it costs 1.43 pp of accuracy (1.14 pp excluding the 61 epochs with
 *    defective ground truth, so it is not a labelling artefact), worsens ECE from 0.0303
 *    to 0.0452, and raises the false-alarm rate on correctly-matched epochs from 2.04 %
 *    to 3.29 %. It does not fix the case it was added for either: the road the vehicle
 *    had diverged onto is not a candidate at all, so no re-ranking among candidates can
 *    recover it. C5 pins the shipped one-sided behaviour; C8 pins the symmetric maths
 *    under an explicit flag so the measurement stays reproducible.
 *
 * Also added: a cap on the per-epoch contribution (C9) and a guard against displacements
 * that span a recording gap (C10). The heading residual is heavier-tailed than any von
 * Mises mixture that fits it -- both a two- and a three-component fit under-predict the
 * measured 3-6 sigma exceedance by a factor of two to three -- so the tail is bounded
 * rather than modelled, and the run length rather than the extremity of one epoch becomes
 * the deciding quantity. The gap guard is a TIME bound, not a speed bound: the four
 * offending Haikou epochs carried 2450 m over 118 s, which is 20.8 m/s, an ordinary road
 * speed that any usable speed ceiling would admit.
 *
 * @author: Chenzhang Ning
 */

#include "mm/cmm/cmm_algorithm.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

using FMM::MM::CovarianceMapMatch;
using FMM::MM::CovarianceMatrix;
using FMM::CORE::LineString;
using FMM::CORE::Point;

static int failures = 0;

static void check(bool got, bool want, const char *what) {
    if (got == want) {
        std::printf("  ok   %s\n", what);
        return;
    }
    std::printf("  FAIL %s -> got %s, want %s\n", what, got ? "true" : "false",
                want ? "true" : "false");
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
static CovarianceMatrix iso(double sigma) {
    return CovarianceMatrix{sigma, sigma, sigma, 0.0, 0.0, 0.0};
}

static LineString line(std::initializer_list<std::pair<double, double>> pts) {
    LineString ls;
    for (const auto &p : pts) ls.add_point(Point(p.first, p.second));
    return ls;
}

static constexpr double SIG = 0.5;   // same abstract length unit as the coordinates

// Defaults are the SHIPPED behaviour: one-sided, uncapped, unit time step, guard off.
// kappa_floor is placed LAST here, out of the order the production function takes it,
// so that every existing call site keeps its positional meaning and this change cannot
// silently re-bind an argument. Default 0.0 = the one-parameter model.
static double pen(const Point &prev, const Point &curr, const LineString &geom,
                  const Point &cand, const CovarianceMatrix &cov, double kappa,
                  double lambda_max = 1e12, double dt = 1.0, double max_interval = 0.0,
                  bool symmetric = false, double kappa_floor = 0.0) {
    return CovarianceMapMatch::compute_direction_penalty(
        prev, curr, geom, cand, cov, kappa, kappa_floor, lambda_max, dt, max_interval, symmetric);
}

int main() {
    const LineString eastbound = line({{0.0, 0.0}, {3.0, 0.0}});
    const Point on_east(1.5, 0.0);

    // ==== 1. the local tangent ==================================================
    std::printf("C1: the tangent is the candidate's own SEGMENT, not the edge chord\n");
    // A hairpin: out along +x to (2,0), back along -x to (1,0.02). The chord runs +x;
    // the return leg runs -x. Travel is in -x, so the vehicle is aligned with its own
    // segment and anti-aligned with the chord.
    const LineString hairpin = line({{0.0, 0.0}, {2.0, 0.0}, {1.0, 0.02}});
    const Point prev(2.6, 0.03), curr(1.6, 0.02);
    const double hairpin_pen = pen(prev, curr, hairpin, Point(1.5, 0.01), iso(SIG), 1.0);
    check_close(hairpin_pen, 0.0, 0.0,
                "driving the return leg gets NO penalty (local tangent agrees)");

    std::printf("\nC2: ...and the chord WOULD have penalised it, so C1 is not vacuous\n");
    // The same two end points as the hairpin, as a plain two-point edge. Its only
    // tangent IS the chord. Same displacement, same covariance, same candidate.
    const double chord_pen = pen(prev, curr, line({{0.0, 0.0}, {1.0, 0.02}}),
                                 Point(1.5, 0.01), iso(SIG), 1.0);
    check_close(chord_pen, -4.0, 0.2,
                "the chord alone gives the full -kappa = -4, so C1 is not vacuous");

    // ==== 2a. the correlation factor ============================================
    std::printf("\nC3: kappa scales as 1/c\n");
    // The edge runs x=3 -> x=0, so its tangent is -x. Travel in +x is the reversed case:
    // v = (+1,0) against a tangent of (-1,0), cos = -1. kappa = |v|^2/sigma^2 = 4.
    const LineString westbound = line({{3.0, 0.0}, {0.0, 0.0}});
    const Point wc(1.0, 0.0);
    const double p_c1 = pen(Point(1.0, 0.0), Point(2.0, 0.0), westbound, wc, iso(SIG), 1.0);
    check_close(p_c1, -4.0, 1e-9, "the shipped one-sided form gives -kappa = -4");
    const double p_c05 = pen(Point(1.0, 0.0), Point(2.0, 0.0), westbound, wc, iso(SIG), 0.5);
    check_close(p_c05 / p_c1, 2.0, 1e-12, "halving c doubles the penalty magnitude");
    const double p_c036 = pen(Point(1.0, 0.0), Point(2.0, 0.0), westbound, wc, iso(SIG), 0.36);
    check_close(p_c036 / p_c1, 1.0 / 0.36, 1e-12,
                "the calibrated c = 0.36 gives 2.78x the penalty");

    // ==== 2b. the perpendicular variance ========================================
    std::printf("\nC4: the variance that matters is ACROSS the travel direction\n");
    // Travel is along +x, so n = (0,1) and sigma_perp^2 must be the NORTH variance.
    // Stretching the covariance ALONG x leaves the heading error untouched, so the
    // penalty must not move -- but sde * sdn is 0.5 for both, so the old formula could
    // not tell them apart at all.
    const CovarianceMatrix across = CovarianceMatrix{0.5, 1.0, 0.5, 0.0, 0.0, 0.0};
    const CovarianceMatrix along  = CovarianceMatrix{1.0, 0.5, 0.5, 0.0, 0.0, 0.0};
    const double p_across = pen(Point(1.0, 0.0), Point(2.0, 0.0), westbound, wc, across, 1.0);
    const double p_along = pen(Point(1.0, 0.0), Point(2.0, 0.0), westbound, wc, along, 1.0);
    check_close(p_along / p_across, 4.0, 1e-12,
                "stretching along travel gives 4x the penalty of stretching across");
    check_close((along.sde * along.sdn) / (across.sde * across.sdn), 1.0, 1e-15,
                "  ...while the old sde*sdn proxy is identical for the two (ratio 1)");
    check(std::fabs(p_across) < std::fabs(p_along), true,
          "  ...so a wide heading error is penalised LESS, as it should be");

    // ==== 3. the shipped one-sided behaviour ====================================
    std::printf("\nC5: SHIPPED DEFAULT is one-sided\n");
    // This is the behaviour the manuscript reports. If the symmetric flag is ever
    // defaulted on, this case fails and the 1.43 pp of accuracy comes back with it.
    check_close(pen(Point(0.0, 0.0), Point(1.0, 0.0), eastbound, on_east, iso(SIG), 1.0),
                0.0, 0.0, "aligned travel: exactly zero");
    check_close(pen(Point(0.0, 0.0), Point(1.0, 1.0), eastbound, on_east, iso(SIG), 1.0),
                0.0, 0.0, "45 degrees off: exactly zero (one-sided returns nothing)");
    check_close(pen(Point(2.0, 0.0), Point(1.0, 0.0), eastbound, on_east, iso(SIG), 1.0),
                -4.0, 1e-9, "reversed travel: -kappa = -4");

    // ==== degenerate inputs =====================================================
    std::printf("\nC6: degenerate inputs return zero rather than a number\n");
    check_close(pen(Point(1.0, 1.0), Point(1.0, 1.0), eastbound, on_east, iso(SIG), 1.0),
                0.0, 0.0, "zero displacement (stopped vehicle)");
    check_close(pen(Point(2.0, 0.0), Point(1.0, 0.0), line({{1.0, 0.0}, {1.0, 0.0}}),
                    Point(1.0, 0.0), iso(SIG), 1.0),
                0.0, 0.0, "zero-length edge");
    check_close(pen(Point(2.0, 0.0), Point(1.0, 0.0), eastbound, on_east, iso(0.0), 1.0),
                0.0, 0.0, "zero covariance");
    check_close(pen(Point(2.0, 0.0), Point(1.0, 0.0), eastbound, on_east, iso(SIG), 0.0),
                0.0, 0.0, "zero kappa_scale");

    std::printf("\nC7: the penalty is never positive\n");
    int positive = 0;
    for (double ang = 0.0; ang <= 180.0; ang += 7.5) {
        const double r = ang * M_PI / 180.0;
        for (int k = 0; k < 2; ++k) {
            const bool sym = (k == 1);
            if (pen(Point(0.0, 0.0), Point(std::cos(r), std::sin(r)), eastbound, on_east,
                    iso(SIG), 0.36, 3.0, 1.0, 0.0, sym) > 0.0) ++positive;
        }
    }
    check(positive == 0, true, "no angle produces a positive term, in either form");

    // ==== the symmetric form, under its flag ====================================
    std::printf("\nC8: SYMMETRIC form (opt-in; measured worse, see the header)\n");
    check_close(pen(Point(0.0, 0.0), Point(1.0, 0.0), eastbound, on_east,
                    iso(SIG), 1.0, 1e12, 1.0, 0.0, true),
                0.0, 0.0, "aligned travel is still exactly zero");
    const double p_22 = pen(Point(0.0, 0.0),
                            Point(std::cos(0.389), std::sin(0.389)), eastbound, on_east,
                            iso(SIG), 1.0, 1e12, 1.0, 0.0, true);
    // kappa = |v|^2/sigma^2 = 4 for a unit displacement, cos(22.3 deg) = 0.925.
    check_close(p_22, 4.0 * (std::cos(0.389) - 1.0), 1e-9,
                "22.3 deg off is penalised (the one-sided form returns exactly zero)");
    // The (1,1) displacement has |v|^2 = 2, so kappa = 8, not 4.
    check_close(pen(Point(0.0, 0.0), Point(1.0, 1.0), eastbound, on_east,
                    iso(SIG), 1.0, 1e12, 1.0, 0.0, true),
                8.0 * (std::cos(M_PI / 4.0) - 1.0), 1e-12, "45 deg gives (cos45 - 1) kappa, kappa = 8");
    check_close(pen(Point(2.0, 0.0), Point(1.0, 0.0), eastbound, on_east,
                    iso(SIG), 1.0, 1e12, 1.0, 0.0, true),
                -8.0, 1e-9, "reversal is -2 kappa = -8 (one-sided gives -kappa = -4)");

    // ==== the cap ===============================================================
    std::printf("\nC9: the cap bounds one epoch's influence (symmetric form)\n");
    check_close(pen(Point(2.0, 0.0), Point(1.0, 0.0), eastbound, on_east,
                    iso(SIG), 1.0, 0.0, 1.0, 0.0, true),
                -8.0, 1e-9, "lambda_max = 0 leaves the -8 nat penalty uncapped");
    check_close(pen(Point(2.0, 0.0), Point(1.0, 0.0), eastbound, on_east,
                    iso(SIG), 1.0, 3.0, 1.0, 0.0, true),
                -3.0, 1e-12, "lambda_max = 3 caps it at -3");
    check_close(pen(Point(0.0, 0.0), Point(1.0, 1.0), eastbound, on_east,
                    iso(SIG), 1.0, 3.0, 1.0, 0.0, true),
                8.0 * (std::cos(M_PI / 4.0) - 1.0), 1e-12, "a penalty inside the cap is left alone");

    // ==== the interval guard ====================================================
    std::printf("\nC10: a displacement spanning a recording gap is not a heading\n");
    // 118 s is the interval the four offending Haikou epochs actually spanned.
    check_close(pen(Point(2.0, 0.0), Point(1.0, 0.0), eastbound, on_east,
                    iso(SIG), 1.0, 1e12, 118.0, 2.0),
                0.0, 0.0, "dt = 118 s: skipped");
    check_close(pen(Point(2.0, 0.0), Point(1.0, 0.0), eastbound, on_east,
                    iso(SIG), 1.0, 1e12, 1.0, 2.0),
                -4.0, 1e-9, "dt = 1 s: kept");
    check_close(pen(Point(2.0, 0.0), Point(1.0, 0.0), eastbound, on_east,
                    iso(SIG), 1.0, 1e12, 2.0, 2.0),
                -4.0, 1e-9, "dt = 2 s (one dropped sample): kept");
    check_close(pen(Point(2.0, 0.0), Point(1.0, 0.0), eastbound, on_east,
                    iso(SIG), 1.0, 1e12, 118.0, 0.0),
                -4.0, 1e-9, "max_interval = 0 disables the guard");

    if (failures != 0) {
        std::printf("\nFAILED (%d)\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("\nPASSED\n");
    return EXIT_SUCCESS;
}
