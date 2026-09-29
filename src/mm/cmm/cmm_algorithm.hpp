/**
 * Covariance-based map matching.
 *
 * CMM algorithm implementation that uses GNSS covariance matrix
 * and protection levels for more accurate map matching.
 *
 * @author: Chenzhang Ning
 * @version: 2025.09.30
 */

#ifndef FMM_CMM_ALGORITHM_H_
#define FMM_CMM_ALGORITHM_H_

#include "network/network.hpp"
#include "network/network_graph.hpp"
#include "mm/transition_graph.hpp"
#include "mm/fmm/ubodt.hpp"
#include "python/pyfmm.hpp"
#include "config/gps_config.hpp"
#include "config/result_config.hpp"

#include <string>
#include <vector>
#include <cmath>
// #include <Eigen/Dense>
// Temporary simple matrix implementation for demonstration
#include <vector>
#include <cmath>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include "cxxopts/cxxopts.hpp"

// Simple 2x2 matrix implementation for CMM
struct Matrix2d {
    double m[2][2];

    Matrix2d() {
        m[0][0] = m[0][1] = m[1][0] = m[1][1] = 0.0;
    }

    Matrix2d(double a, double b, double c, double d) {
        m[0][0] = a; m[0][1] = b;
        m[1][0] = c; m[1][1] = d;
    }

    Matrix2d inverse() const {
        double det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        if (det == 0) return Matrix2d(); // Return zero matrix if singular
        double inv_det = 1.0 / det;
        return Matrix2d(m[1][1] * inv_det, -m[0][1] * inv_det,
                       -m[1][0] * inv_det, m[0][0] * inv_det);
    }

    double determinant() const {
        return m[0][0] * m[1][1] - m[0][1] * m[1][0];
    }
};

struct Vector2d {
    double v[2];

    Vector2d(double x, double y) {
        v[0] = x; v[1] = y;
    }

    double operator*(const Vector2d& other) const {
        return v[0] * other.v[0] + v[1] * other.v[1];
    }
};

namespace FMM {
namespace MM {

/**
 * GNSS covariance matrix structure
 *
 * Represents the 3x3 covariance matrix for GNSS observations.
 * Only 6 unique values are stored (matrix is symmetric).
 *
 * Component mapping for CSV input (JSON array per point):
 * [values[0], values[1], values[2], values[3], values[4], values[5]]
 * maps to:
 * [sde, sdn, sdu, sdne, sdeu, sdun]
 *
 * Matrix layout (ENU coordinates):
 * | sde^2   sdne    sdeu  |
 * | sdne    sdn^2   sdun  |
 * | sdeu    sdun    sdu^2 |
 *
 * For 2D map matching, only the horizontal components are used:
 * - sde (East std)
 * - sdn (North std)
 * - sdne (North-East covariance)
 */
struct CovarianceMatrix {
    // Units depending on the input data (usually lla)
    double sde;    // East standard deviation
    double sdn;    // North standard deviation
    double sdu;    // Up standard deviation
    double sdne;   // North-East covariance
    double sdeu;   // East-Up covariance
    double sdun;   // Up-North covariance 

    // Convert to 2D covariance matrix (horizontal plane)
    Matrix2d to_2d_matrix() const {
        return Matrix2d(sde * sde, sdne, sdne, sdn * sdn);
    }

    // `get_2d_uncertainty()` used to live here, returning sqrt(det Sigma).
    // That is sigma_1 * sigma_2, whose unit is m^2, not the "position
    // uncertainty in meters" it claimed. The emission model needs the
    // quadratic form d^T Sigma^-1 d instead, so the accessor was dropped.
};

/**
 * Configuration class for CMM algorithm
 */
struct CovarianceMapMatchConfig {
    /**
     * Constructor of CovarianceMapMatch configuration
     * @param k_arg the number of candidates
     * @param min_candidates_arg minimum number of candidates to keep
     * @param reverse_tolerance reverse movement tolerance
     */
    CovarianceMapMatchConfig(int k_arg = 8, int min_candidates_arg = 3,
                           double reverse_tolerance = 0.0,
                           bool use_mahalanobis_candidates_arg = true,
                           bool filtered_arg = true,
                           bool enable_gap_bridging_arg = true,
                           double max_interval_arg = 180.0, /* in seconds */
                           double trustworthiness_threshold_arg = 0.0, /* linear prob */
                           double phmi_arg = 1.0e-5,
                           double cumulative_reverse_pct_arg = 0.03,
                           bool direction_penalty_arg = true,
                           double k_search_arg = 1.0,
                           double delta_road_arg = 0.0,
                           double isotropic_emission_scale_arg = 0.0,
                           bool tw_use_viterbi_arg = false,
                           double map_error_sigma_arg = 0.0,
                           double direction_kappa_scale_arg = 0.36,
                           double direction_kappa_floor_arg = 0.0,
                           double direction_lambda_max_arg = 0.0,
                           double direction_max_interval_arg = 2.0,
                           bool direction_symmetric_arg = false,
                           bool tp_normalized_arg = true);

    int k;                          /**< Number of candidates */
    int min_candidates;             /**< Minimum number of candidates to keep */
    double reverse_tolerance;           /**< Reverse movement tolerance */
    bool use_mahalanobis_candidates;    /**< Whether to use Mahalanobis-based candidate search */
    bool filtered;                      /**< Whether to apply trustworthiness_threshold; false bypasses it */

    // --- Gap Handling & Integrity Parameters ---
    bool enable_gap_bridging;           /**< Enable skipping invalid points to bridge gaps */
    double phmi;                        /**< Probability of Hazardously Misleading Integrity information (default 1e-5) */

    double max_interval;                /**< Maximum time interval to split trajectory */
    double trustworthiness_threshold;   /**< Threshold to filter out low-confidence matches */

    double cumulative_reverse_pct;       /**< Maximum cumulative reverse travel as fraction of edge length (0.03 = 3%) before blocking same-edge transition. Only applied on one-way edges. */
    bool direction_penalty;              /**< Whether to apply direction-consistency von Mises penalty for reverse-direction candidates (default true). Set to false for ablation studies isolating the contribution of direction awareness. */

    // Row normalisation of the transition factors,
    //     t_{a->b} = w_{a->b} / sum_b' w_{a->b'},   w = min(1, d_gnss / d_road).
    // true (default) is the model this matcher describes. The divisor depends on the
    // SOURCE candidate, so it does not cancel between the forward and backward
    // recursions; what keeps alpha and beta messages of one chain is that both read this
    // same setting, which is why the flag is applied to the recursion and to the stored
    // row together and never to one of them alone. false stores and uses the raw w.
    //
    // false is an ABLATION, and the reason it exists is attribution rather than the model:
    // the FMM baseline's forward recursion uses the raw factor (see the long note in
    // fmm_algorithm.cpp's update_layer), so the baseline-vs-proposed comparison changes
    // the emission AND the transition factor at once. A run with tp_normalized=false
    // against the baseline's constant-sigma emission gives the missing cell of the 2x2
    // grid -- {constant sigma, per-epoch sigma} x {raw, row-normalised} -- and locates
    // which of the two changes each reported difference belongs to. It is not a
    // configuration the paper reports.
    bool tp_normalized;                  /**< true (default) = row-normalised transition factors; false = raw (attribution ablation only) */

    // The von Mises concentration is kappa = |v|^2 / (c * n^T Sigma n), where n is
    // the unit normal to the displacement. c = 2(1-rho) carries the inter-epoch
    // correlation of the receiver error, which a single epoch's covariance cannot
    // express: the displacement is a difference of two fixes, so its error is
    // 2(1-rho)Sigma rather than Sigma.
    //
    // Measured by one regression, sigma_theta^2 = c * (n^T Sigma n)/|v|^2 + b, over
    // 11,792 correctly-matched gap-free steps on the Haikou set:
    //
    //     c = 0.227,  b = 0.0084 rad^2,  rho = 0.89,  z(b) = 9.0
    //
    // (experiments/scripts/fit_direction_kappa.py). The regressor is
    // n^T Sigma n / |v|^2, which is what the model says the covariance term is.
    //
    // THE FIT IS A FIXED POINT, NOT A ONE-SHOT REGRESSION, and the difference is not
    // cosmetic. The sample is "epochs the matcher got right on a gap-free step", so it
    // depends on the matcher's parameters, which depend on these constants. Regressing
    // against the PREVIOUS configuration's output gives c = 0.254 / b = 0.0093;
    // re-running at those and regressing again gives 0.227 / 0.0084, and re-running at
    // THOSE reproduces them exactly. The iteration converges in one step and the numbers
    // above are the converged ones. Quoting the first-iteration values is how a constant
    // the shipped configuration does not reproduce gets written down as if it were the
    // model's -- input/config/paper1_cmm.xml carries the same warning for the same reason.
    //
    // AN EARLIER VALUE OF 0.36 IS ALSO SUPERSEDED, and it failed for a different reason:
    // it came from regressing on 1/|v|^2 alone, which is the same thing only if
    // n^T Sigma n is constant -- it is not, it varies epoch to epoch with the reported
    // covariance -- so that fit was biased by omitting the regressor. So 0.36 and 0.254
    // are not two estimates of one quantity either; only 0.227 is the definition.
    //
    // This constant is NOT derivable from the reported covariance: Sigma_i is the
    // marginal distribution of one epoch and rho is a property of the pair. It is
    // measured, not assumed, and the measurement is a regression on the heading
    // residual rather than a fit to any reported metric. 1.0 restores the
    // pre-2026-09-27 behaviour of an implicit rho = 0.5.
    double direction_kappa_scale;
    /**
     * b in sigma_theta^2 = c * (n^T Sigma n) / |v|^2 + b, in rad^2.
     *
     * The second term is the part of the heading variance the covariance cannot contain:
     * the vehicle's own heading change within the epoch, the digitised centreline's
     * disagreement with the road, and the reference truth's own noise. Fitting only c and
     * forcing b = 0 makes c absorb b, which is why a slope fitted on one dataset does not
     * transfer to another -- the corpus fitted 1.09 to 3.44 against Haikou's 0.227.
     *
     * With b > 0, kappa saturates at 1/b as |v| grows instead of growing without bound: a
     * long straight step cannot make the heading arbitrarily certain when the vehicle is
     * steering. Measured on Haikou, b = 0.0084 rad^2 -- a 5.2 deg floor, 29x the c term at
     * the median speed. 0.0 reproduces the one-parameter model exactly, term for term.
     * See experiments/scripts/fit_direction_kappa.py.
     */
    double direction_kappa_floor;

    // The heading residual is heavy-tailed well beyond what the Gaussian behind kappa
    // predicts, and no von Mises mixture fits the 3-6 sigma band (both a two- and a
    // three-component fit under-predict the measured exceedance by a factor of two to
    // three there; the cause is structural -- a mixture jumps from a narrow core to a
    // broad component while the physical cause, how far the vehicle moves laterally in
    // one epoch, is continuous). Measured on 11,740 correctly-matched epochs, 1.89 %
    // exceed 4 sigma, so an uncapped term would fire an alarm on 222 of them.
    //
    // Since the tail cannot be modelled it must not be allowed to drive the inference.
    // The per-epoch contribution is capped, which makes the RUN LENGTH rather than the
    // extremity of any one epoch the deciding quantity: a single epoch can move the
    // candidate/background odds by at most exp(lambda_max), while a divergence sustained
    // over N epochs contributes N * lambda_max.
    // Only meaningful with direction_symmetric. The one-sided form's penalty at a
    // reversal is -kappa, which reaches ~2000 on this data, and capping THAT to a few
    // nats would gut the wrong-way detection the shipped term exists for. Defaulted off
    // for that reason; the symmetric form is where the cap was measured to matter.
    double direction_lambda_max;         /**< Cap on the per-epoch heading penalty, in nats. 0 (default) = no cap. Only useful with direction_symmetric */

    // The heading is the DIRECTION of the displacement, and that is only a heading if
    // the interval is short enough that the vehicle cannot have turned in between. It is
    // a TIME bound, not a speed bound: the four offending Haikou epochs carried 2450 m
    // over 118 s, which is 20.8 m/s -- a perfectly ordinary road speed, and one that any
    // speed ceiling high enough to admit a motorway would let through. What is wrong
    // with those epochs is that the chord between two fixes two minutes apart is not the
    // direction of travel at either end.
    double direction_max_interval;       /**< Displacements spanning more than this many seconds are not headings; the term is skipped. 0 disables the guard */

    // Making the heading term symmetric -- kappa(cos-1), penalising any misalignment
    // rather than only a reversal -- is the obvious way to give a diverged vehicle a
    // low score. It was implemented and MEASURED on 2026-09-27 and it is a net negative
    // on this data. At the manuscript operating point, against the one-sided term:
    //
    //     accuracy   97.87 -> 96.44   (-1.43 pp; -1.14 pp excluding the 61 GT-defect
    //                                  epochs, so it is not a labelling artefact)
    //     ECE        0.0303 -> 0.0452
    //     BSS       -0.176  -> -0.510
    //     sep        0.435  -> 0.305
    //     false alarm, tw < 0.5, on correctly-matched epochs:  2.04 % -> 3.29 %
    //
    // The loss is concentrated on Traj 11 (-5.1 pp) and Traj 13 (-1.8 pp), and the
    // mechanism is the heavy tail documented above: on the 3.1 % of correctly-matched
    // epochs whose heading residual exceeds 3 sigma, the symmetric term pushes the
    // matcher OFF the correct edge. And it does not achieve the goal anyway -- Traj 11's
    // window stays 44/44 wrong under it -- because the road the vehicle was on is not a
    // candidate at all, so no re-ranking among candidates can recover it.
    //
    // Left in, defaulted off, so the measurement is reproducible rather than a claim.
    bool direction_symmetric;            /**< false (default, measured better) = one-sided, penalise reversals only; true = symmetric von Mises, penalise any misalignment */

    // --- Search radius -------------------------------------------------------
    // The paper's candidate search radius is
    //     r_i = k_search * HPL_i + delta_road,
    // in the network's own units (degrees on this dataset). k_search scales the
    // integrity bound and is 1 in the paper.
    //
    // delta_road is a fixed lateral allowance, and it exists because the two
    // quantities are not the same size. A protection level bounds the distance
    // from the fix to the *true* position; the candidate attached to a segment
    // is a point on that segment's centreline, so the distance the search has to
    // cover is from the fix to a centreline, which exceeds the positioning
    // error by the vehicle's lateral offset within the carriageway. On this
    // dataset that offset is larger than the positioning error itself, so
    // delta_road = 0 discards epochs whose road is unambiguously within reach.
    //
    // An epoch whose disk still admits no segment is dropped rather than
    // retried: its candidate set is empty, it contributes no HMM state, and
    // matching continues with the epochs that remain.
    double k_search;      /**< Scale on the protection level (default 1.0) */
    double delta_road;    /**< Fixed lateral allowance added to the scaled protection level, in network units (default 0.0) */

    // --- Ablation-only knob --------------------------------------------------
    // Not part of the paper's Section III model; the manuscript never describes
    // it. It exists so that the component ablation
    // (experiments/scripts/ablate_cmm_factors.py) can vary the emission shape
    // on its own and attribute the calibration result. It defaults to 0.0,
    // which selects the modelled behaviour, and is declared at the END of the
    // constructor's parameter list so that the positional constructions in
    // example/cmm_example.cpp and CMM_README.md keep their meaning.

    double isotropic_emission_scale; /**< > 0 makes the emission claim an isotropic covariance with the SAME TOTAL VARIANCE as the reported one: sigma_i = scale * sqrt((sde_i^2 + sdn_i^2) / 2). scale = 1 is the variance-matched operating point. The projection keeps using the reported covariance, so emission shape and projection metric stay separable factors. */

    bool tw_use_viterbi; /**< Ablation-only. false (default) scores trustworthiness with the forward probability, giving the filtering posterior of the paper's Eq. (filtering_posterior_tw). true scores it with the Viterbi score instead: the confidence implied by the best-path recursion, which never sees the mass of the competing paths and is therefore the natural "native" score of a Viterbi-only matcher. The two differ only from the second epoch of a sub-segment onward; at the first they are the same quantity by construction. */

    // --- Map-registration term in the emission --------------------------------
    // The residual d = z_i - x_ij is not the GNSS error alone. It also carries
    // the map's own error: the distance from the true road position to the
    // digitised centreline, plus the vehicle's lateral offset within the
    // carriageway. The candidate search already allows for both -- that is what
    // delta_road is -- but the emission does not, so Sigma_i currently claims
    // the residual is the reported GNSS covariance when it is not.
    //
    // map_error_sigma adds that term back, in metres, as Sigma_emis = Sigma + s^2 I
    // added in the VARIANCE: sde and sdn are combined in quadrature and the
    // cross term is left alone, the map error being uncorrelated with the GNSS
    // one. Isotropic is the honest shape for the map component: georeferencing
    // error has no preferred direction. The anisotropic structure of the
    // reported covariance is preserved by construction, because the term is
    // added to it rather than replacing it.
    //
    // The DIRECTION PENALTY deliberately does not see this term. Its
    // concentration kappa = |v|^2 / (sde * sdn) is a heading uncertainty, driven
    // by the displacement error between consecutive fixes; a misregistered map
    // shifts where the road is, not which way the vehicle is pointing. Letting
    // kappa follow the widened covariance would attenuate it by (0.8/10)^2 and
    // confound the emission change with a direction change, which is the same
    // confound the ablation sweeps already carry.
    //
    // default 0.0 --> the term is absent and every result is bit-identical.
    double map_error_sigma; /**< Isotropic map-registration standard deviation in METRES, added to the emission covariance in quadrature. 0 (default) disables it. */

    /**
     * Check if the configuration is valid or not
     * @return true if valid
     */
    bool validate() const;

    /**
     * Print information about this configuration
     */
    void print() const;

    /**
     * Load configuration from xml data
     * @param xml_data xml data read from an xml file
     * @return a CovarianceMapMatchConfig object
     */
    static CovarianceMapMatchConfig load_from_xml(
        const boost::property_tree::ptree &xml_data);

    /**
     * Load configuration from argument data
     * @param arg_data argument data
     * @return a CovarianceMapMatchConfig object
     */
    static CovarianceMapMatchConfig load_from_arg(
        const cxxopts::ParseResult &arg_data);

    /**
     * Register arguments to an option object
     */
    static void register_arg(cxxopts::Options &options);

    /**
     * Register help information to a string stream
                       */
    static void register_help(std::ostringstream &oss);
};

/**
 * Enhanced trajectory with covariance and protection level data
 *
 * CSV Input Format Specification:
 * When reading from aggregated CSV format, each row should contain:
 * - id: Trajectory ID (integer)
 * - geom: WKT LINESTRING geometry
 * - timestamps: JSON 1D array of timestamps [t1, t2, t3, ...]
 * - covariances: JSON 2D array where each point has 6 values [sde, sdn, sdu, sdne, sdeu, sdun]
 *               Example: [[0.68,0.69,0.81,0.033,0.0,0.0],[0.67,0.69,0.81,0.032,0.0,0.0],...]
 * - protection_levels: JSON 1D array of protection levels [pl1, pl2, pl3, ...]
 *
 * Covariance Matrix Components (per point):
 * - sde: East standard deviation
 * - sdn: North standard deviation
 * - sdu: Up standard deviation
 * - sdne: North-East covariance
 * - sdeu: East-Up covariance
 * - sdun: Up-North covariance
 */
struct CMMTrajectory {
    int id;                                          /**< Id of the trajectory */
    CORE::LineString geom;                          /**< Geometry of the trajectory */
    std::vector<double> timestamps;                 /**< Timestamps of the trajectory */
    std::vector<CovarianceMatrix> covariances;      /**< Covariance matrices for each point */
    std::vector<double> protection_levels;          /**< Protection levels for each point */

    CMMTrajectory() : id(0) {}

    CMMTrajectory(int id_arg, const CORE::LineString &geom_arg,
                  const std::vector<double> &timestamps_arg,
                  const std::vector<CovarianceMatrix> &covariances_arg,
                  const std::vector<double> &protection_levels_arg)
        : id(id_arg), geom(geom_arg), timestamps(timestamps_arg),
          covariances(covariances_arg), protection_levels(protection_levels_arg) {}

    /**
     * Check if trajectory has valid covariance and protection level data
     */
    bool is_valid() const {
        size_t num_points = geom.get_num_points();
        return covariances.size() == num_points &&
               protection_levels.size() == num_points;
    }
};

/**
 * Candidate search result containing candidates and their emission probabilities.
 */
struct CandidateSearchResult {
    Traj_Candidates candidates;
    std::vector<std::vector<double>> emission_probabilities;
};

/**
 * Covariance-based map matching algorithm/model.
 */
class CovarianceMapMatch {
public:
    /**
     * Constructor of Covariance map matching model
     * @param network road network
     * @param graph road network graph
     * @param ubodt Upperbounded origin destination table
     */
    CovarianceMapMatch(const NETWORK::Network &network,
                      const NETWORK::NetworkGraph &graph,
                      std::shared_ptr<UBODT> ubodt)
        : network_(network), ubodt_(ubodt) {
        // Kept in the signature for parity with FastMapMatch(network, graph,
        // ubodt); CMM resolves distances through the UBODT and never walks the
        // adjacency graph, so no member is stored.
        (void)graph;
    }

    /**
     * Match a trajectory to the road network
     * @param traj input trajectory data with covariance and protection levels
     * @param config configuration of map matching algorithm
     * @return a vector of map matching results (one for each split segment)
     */
    std::vector<MatchResult> match_traj(const CMMTrajectory &traj,
                                      const CovarianceMapMatchConfig &config);

    /**
     * Match a trajectory while optionally returning the filtered trajectory
     * after dropping epochs with no feasible candidates/transitions.
     * @return a vector of map matching results
     */
    std::vector<MatchResult> match_traj(const CMMTrajectory &traj,
                                      const CovarianceMapMatchConfig &config,
                                      CMMTrajectory *filtered_traj);

    /**
     * Match GPS data stored in a file with covariance and protection level data
     * @param gps_config GPS configuration including covariance and protection level files
     * @param result_config result configuration
     * @param config map matching configuration
     * @param input_epsg EPSG code of input trajectory CRS (e.g., 4326 for WGS84)
     * @param use_omp whether to use OpenMP
     * @return a string storing information about running time and statistics
     */
    std::string match_gps_file(
        const FMM::CONFIG::GPSConfig &gps_config,
        const FMM::CONFIG::ResultConfig &result_config,
        const CovarianceMapMatchConfig &config,
        int input_epsg,
        bool use_omp = true);

    /**
     * Decide how to handle an epoch whose transition from the current
     * sub-segment failed.
     *
     * True  -> restart: commit the current sub-segment and re-seed the HMM at
     *          this epoch, so the epoch is matched in its own right.
     * False -> carry: put the epoch on the gap list and retry it later as a
     *          long jump from the last connected epoch (`skipped_indices`).
     *
     * The decision deliberately does NOT depend on the size of the current
     * sub-segment. It used to require `current_sub_indices.size() >= 2`, which
     * is not a safety property: the size cannot change while the sub-segment is
     * stuck, so a sub-segment that failed on its very first transition -- a
     * trajectory that starts off-network, or a restart seeded at an
     * off-network epoch -- could never restart again. Every following epoch was
     * then carried as a gap until the 180 s `max_interval` escape, even when
     * those epochs connected to each other perfectly well. Measured on the
     * Haikou set at a tight protection level, that dropped 167 epochs that the
     * restart recovers (traj 11: 23 epochs, traj 21: 144), and it cost 1.37 pp
     * of segment accuracy (93.21 % -> 94.58 %).
     *
     * Public so that tests/test_cmm_gap_restart.cpp can pin the rule; the
     * function is pure and has no effect on its own.
     *
     * @param enable_gap_bridging whether the config allows splitting at all
     * @param next_epoch_has_candidates false if the epoch has no road candidate;
     *        an empty layer cannot seed a sub-segment, so the epoch must be
     *        carried and retried rather than restarted at.
     * @return true if a new sub-segment should start at this epoch
     */
    static bool should_restart_sub_segment(bool enable_gap_bridging,
                                           bool next_epoch_has_candidates);

    /**
     * Test whether a covariance matrix can be used as the paper's emission
     * model requires, i.e. whether it defines a genuine 2x2 Gaussian.
     *
     * This exists to replace an earlier hard-coded floor on the standard
     * deviation. That floor did not check validity: it rescaled *every* small
     * covariance up to a fixed minimum, so the emission model silently used a
     * different covariance from the one the receiver reported. Measured on the
     * Haikou set it fired on 99.98 % of epochs with a median scale factor of
     * 7.22, i.e. it was not a guard rail, it was the model. It also propagated
     * into the direction penalty, which is inversely proportional to the
     * standard deviation and therefore depended on the floor rather than on the
     * data.
     *
     * The rule now is: a usable covariance is used exactly as given, however
     * small; an unusable one is replaced by a documented isotropic fallback.
     * The two cases are mutually exclusive and there is no third path.
     *
     * Only sde, sdn and sdne are examined, because to_2d_matrix() uses only
     * those. The three components are checked for finiteness, for a strictly
     * positive standard deviation, and for positive definiteness. Note the
     * determinant must be strictly positive: at exactly zero the matrix is
     * singular and Matrix2d::inverse() returns the zero matrix, which would give
     * every candidate a Mahalanobis distance of zero and make them
     * indistinguishable.
     *
     * @param cov covariance as read from the input, before any adjustment
     * @return true if cov may be used directly
     */
    static bool is_covariance_usable(const CovarianceMatrix &cov);

    /**
     * Build the isotropic fallback covariance used when is_covariance_usable()
     * rejects the input.
     *
     * The fallback is a 5 m circle, matching the assumed SPP accuracy of the
     * real-vehicle data. A circle is the honest shape here: an unusable
     * covariance carries no information about the error's orientation, so
     * inventing an anisotropy would be to fabricate exactly the quantity that
     * was just found missing. Note this is not a floor and must not be applied
     * to a covariance that passes is_covariance_usable() -- a valid 0.6 m
     * covariance stays 0.6 m.
     *
     * Units follow the network: metric CRS gives sigma in metres, geographic
     * CRS in degrees. In a geographic CRS one degree of longitude is shorter
     * than a degree of latitude by cos(latitude), so a single scale for both
     * would make the "circle" an ellipse. The correction matters: at Haikou
     * (20 N) it is 6 %, and it grows with latitude.
     *
     * @param network_projected true if the network's CRS is projected (metric)
     * @param latitude_deg observation latitude, only used for geographic CRS
     * @return a usable isotropic covariance
     */
    static CovarianceMatrix fallback_covariance(bool network_projected,
                                                double latitude_deg);

    /**
     * Express an isotropic metric standard deviation as a covariance in the
     * network's own units.
     *
     * The conversion is the one fallback_covariance() documents: metric CRS
     * passes sigma through unchanged, geographic CRS divides by the WGS84 mean
     * meridian length, and the east axis divides again by cos(latitude) so that
     * the result is a circle in metres rather than in degrees. Both callers
     * share it so that the fallback and the map-error term cannot drift apart
     * in units -- the failure mode this code already carries a defect for.
     *
     * @param sigma_m isotropic standard deviation in metres, must be >= 0
     * @param network_projected true if the network's CRS is projected (metric)
     * @param latitude_deg observation latitude, only used for geographic CRS
     * @return a zero-cross-term covariance in network units
     */
    static CovarianceMatrix sigma_m_to_covariance(double sigma_m,
                                                  bool network_projected,
                                                  double latitude_deg);

    /**
     * The two covariances the emission step needs, kept apart because they
     * answer different questions.
     *
     * `emission` is the position likelihood's: the reported covariance with the
     * map-registration term added in quadrature. `direction` is the direction
     * penalty's: the reported covariance, never widened.
     *
     * The split is deliberate and is the thing to preserve. The penalty's
     * concentration kappa = |v|^2 / (sde * sdn) is a HEADING uncertainty, driven
     * by the displacement error between consecutive fixes. A misregistered map
     * moves where the road is, not which way the vehicle points, so the map term
     * has no business in kappa. Letting it in would attenuate kappa by roughly
     * (0.8/10)^2 -- silently switching the direction penalty off -- and would
     * change two mechanisms at once, which is exactly what makes the earlier
     * isotropic-emission sweep unattributable.
     *
     * @param base the emission covariance already resolved for the isotropic
     *             ablation, i.e. BEFORE the map term is added
     * @param map_error_sigma isotropic map-registration sigma in metres, 0 = off
     * @param network_projected true if the network's CRS is projected (metric)
     * @param latitude_deg observation latitude, only used for geographic CRS
     * @return the pair; both fields equal base when map_error_sigma <= 0
     */
    struct EmissionCovariances {
        CovarianceMatrix emission;  /**< position likelihood */
        CovarianceMatrix direction; /**< direction penalty; never widened */
    };
    static EmissionCovariances emission_covariances(
        const CovarianceMatrix &base, double map_error_sigma,
        bool network_projected, double latitude_deg);

    /**
     * The candidate search radius at one epoch: r_i = k_search * HPL_i + delta_road.
     *
     * There is deliberately no fallback here. An epoch whose disk admits no
     * road segment is dropped, not retried with a larger radius: retrying until
     * every epoch finds a candidate would guarantee a match for every epoch at
     * the price of matching epochs the observation does not support. The
     * allowance that keeps legitimately reachable epochs in the set is
     * delta_road, and it is fixed once for the whole evaluation rather than
     * grown per epoch.
     *
     * This is the one place the radius is computed, in either the Mahalanobis
     * or the Euclidean candidate branch, so the two cannot drift apart.
     *
     * Public so that tests/test_cmm_ablation_knobs.cpp can pin the arithmetic;
     * the function is pure and has no effect on its own.
     *
     * @param protection_level the epoch's HPL, in network units
     * @param k_search scale on the protection level (1.0 in the paper)
     * @param delta_road fixed lateral allowance, in network units
     * @return the radius to search with
     */
    static double effective_search_radius(double protection_level,
                                          double k_search,
                                          double delta_road);

    /**
     * The standard deviation of the isotropic covariance used by the emission
     * when isotropic_emission_scale is set.
     *
     * The scale is defined to preserve the TOTAL VARIANCE of the reported
     * covariance: sde^2 + sdn^2, so that an isotropic arm and an anisotropic arm
     * of the ablation differ in the shape of the error ellipse and not in its
     * size. That makes the two levels of the emission factor comparable, which
     * is the whole point of fixing an operating point.
     *
     * This is applied to the emission only. The projection keeps using the
     * reported covariance: it is tempting to produce the isotropic arm by
     * rewriting the input table instead, but the same matrix drives the
     * Mahalanobis projection, so zeroing its off-diagonal and equalising its
     * diagonal would change the projection metric at the same time and confound
     * two factors.
     *
     * Because sqrt((a+b)/2) >= sqrt(ab) for positive a, b, the isotropic sigma
     * is at least the geometric mean of sde and sdn. The direction penalty uses
     * kappa = speed^2 / (sde * sdn), so the isotropic arm also gets a weaker
     * penalty whenever the reported ellipse is elongated -- correctly, since
     * the penalty is meant to follow the covariance the model believes.
     *
     * @param cov the reported covariance
     * @param scale multiplier; <= 0 means "not set"
     * @return the isotropic sigma, or <= 0 to mean "keep the reported
     *         covariance" -- either because the knob is off, or because the
     *         input is not a covariance and has no variance to preserve
     */
    static double isotropic_sigma(const CovarianceMatrix &cov, double scale);

    /**
     * Compute the direction-consistency penalty for a candidate.
     *
     * Two things changed here on 2026-09-27, both measured rather than assumed.
     *
     * 1. THE TANGENT IS LOCAL. The tangent used to be the edge's start -> end chord.
     *    For a long edge that chord is not the road's direction where the vehicle is:
     *    on 白驹大道 (3.5 km) it differs from the local direction by enough to inflate
     *    the apparent heading residual from 0.72 deg to 7.82 deg at 25-35 m of travel.
     *    The inflation grew with speed -- exactly the opposite of the 1/|v| law the
     *    measurement should follow -- which is what exposed it. The tangent is now the
     *    direction of the polyline segment the candidate sits on.
     *
     * 2. KAPPA CARRIES THE INTER-EPOCH CORRELATION AND A FLOOR. The displacement is a
     *    DIFFERENCE of two fixes, so its error is Var(dv) = 2(1-rho) Sigma, not Sigma.
     *    The code used to form |v|^2 / (sde * sdn), i.e. c = 1, which silently asserts
     *    rho = 0.5. The model is
     *
     *        sigma_theta^2 = c * (n^T Sigma n) / |v|^2 + b
     *
     *    with c = 0.227 (rho = 0.89) and b = 0.0084 rad^2, fitted over 11,792
     *    correctly-matched gap-free steps; see the direction_kappa_scale comment for
     *    why the fit is a FIXED POINT (0.254 / 0.0093 are its first iteration, not a
     *    second estimate) and why the earlier 0.36 from a 1/|v|^2-only regression is
     *    superseded, and the direction_kappa_floor comment for why b is in the model
     *    rather than absorbed into c. "No residual floor" was written when the fit
     *    regressed on 1/|v|^2 alone, which cannot separate a floor from a mis-scaled
     *    slope; with the correct regressor the floor is non-zero at z = 9.0.
     *
     * The perpendicular variance n^T Sigma n is used rather than the product
     * sde * sdn: the product is the geometric mean of the two axis variances whatever
     * the direction of travel, while the heading error is governed by the variance
     * ACROSS the direction of travel.
     *
     * The penalty remains one-sided in the sense that only cos(theta) < 0 is
     * penalised -- see the note at the call site for what that does and does not catch.
     *
     * @param obs_prev     previous GNSS observation point (z_{i-1})
     * @param obs_curr     current GNSS observation point (z_i)
     * @param edge_geom    candidate edge geometry
     * @param cand_point   the candidate point on that geometry (the projection)
     * @param cov          reported covariance of the current epoch, NOT the widened
     *                     emission covariance: the map term is common-mode across
     *                     nearby epochs so it cancels in the displacement
     * @param kappa_scale  c = 2(1-rho); 1.0 reproduces the pre-2026-09-27 formula
     * @return             log penalty term (<= 0); 0 = no penalty
     */
    static double compute_direction_penalty(const CORE::Point &obs_prev,
                                            const CORE::Point &obs_curr,
                                            const CORE::LineString &edge_geom,
                                            const CORE::Point &cand_point,
                                            const CovarianceMatrix &cov,
                                            double kappa_scale,
                                            double kappa_floor,
                                            double lambda_max,
                                            double dt,
                                            double max_interval,
                                            bool symmetric);

protected:
    /**
     * Compute direction-consistency penalty for reverse-direction candidates.
     * Uses GNSS displacement velocity v = obs_i - obs_{i-1} and the candidate
     * edge's tangent direction to penalize wrong-way matches.
     * Only penalizes when cos(theta) < 0 (opposite direction); forward and
     * lateral directions receive no penalty.
     * @param obs_prev      previous GNSS observation point (z_{i-1})
     * @param obs_curr      current GNSS observation point (z_i)
     * @param edge_start    start point of candidate road segment
     * @param edge_end      end point of candidate road segment
     * @param cov           current epoch covariance matrix (for kappa estimation)
     * @return              log penalty term (≤0); 0 = no penalty, negative = penalty
     */

    /**
     * Search candidates based on protection level
     * @param geom trajectory geometry
     * @param covariances covariance matrices for each point
     * @param protection_levels protection levels for each point
     * @param config CMM configuration
     * @return trajectory candidates with log-space emission probabilities
     */
    CandidateSearchResult search_candidates_with_protection_level(
        const CORE::LineString &geom,
        const std::vector<CovarianceMatrix> &covariances,
        const std::vector<double> &protection_levels,
        const std::vector<double> &timestamps,
        const CovarianceMapMatchConfig &config) const;

    /**
     * Get shortest path distance between two candidates
     * @param ca from candidate
     * @param cb to candidate
     * @param reverse_tolerance reverse movement tolerance
     * @return shortest path value
     */
    double get_sp_dist(const Candidate *ca, const Candidate *cb, double reverse_tolerance);

    /**
     * Initialize the first layer probabilities (Log-Space Top-K Normalization)
     * @param layer first layer to initialize
     * @param config CMM configuration
     */
    void initialize_first_layer(TGLayer *layer, const CovarianceMapMatchConfig &config, double &log_prob_unconsidered);

    /**
     * Update probabilities between two layers in the transition graph (LOG-SPACE)
     * Performs L2 filtering and Top-K Normalization in Log-Space.
     * @param la_ptr layer a (previous layer)
     * @param lb_ptr layer b (current layer)
     * @param eu_dist Euclidean distance between two observed points
     * @param connected set to false if the layer is not connected with the next layer
     * @param config CMM configuration
     * @param tp_raw_out if non-null, stores the raw transition probability matrix
     *        tp_raw_out[a][b] = P(z_b|s_b) * P(s_b|s_a) before normalization
     */
    void update_layer_cmm(TGLayer *la_ptr, TGLayer *lb_ptr,
                         double eu_dist,
                         bool *connected,
                         const CovarianceMapMatchConfig &config,
                         double &log_prob_unconsidered);

    /**
     * Slice a CMMTrajectory into a sub-segment [start_idx, end_idx)
     * @param traj original trajectory
     * @param start_idx start index (inclusive)
     * @param end_idx end index (exclusive)
     * @return sub-trajectory
     */
    CMMTrajectory slice_trajectory(const CMMTrajectory &traj, int start_idx, int end_idx) const;

private:
    const NETWORK::Network &network_;
    std::shared_ptr<UBODT> ubodt_;
};

} // MM
} // FMM

#endif // FMM_CMM_ALGORITHM_H_
