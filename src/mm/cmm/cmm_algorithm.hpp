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
                           bool direction_penalty_arg = true);

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
     * Decide whether the candidate search radius should be doubled and retried.
     *
     * The search starts at the paper's radius, r_i = HPL_i. An epoch whose
     * protection-level disk contains no road edge would otherwise be dropped
     * from the output entirely. That is what the manuscript's pseudocode does
     * (`continue` on an empty candidate set), but it discards epochs whose fix
     * is merely unlucky: the covariance honestly reports that nothing is near,
     * yet the vehicle was on a road. Doubling recovers them.
     *
     * This is NOT the heuristic doubling criticised in the manuscript's
     * related-work discussion, and the differences are what keep it honest:
     *   - it triggers only when the candidate set is short, never to top up a
     *     set that already qualifies, so it cannot widen the disk of an epoch
     *     that is matching fine;
     *   - it is capped by a hard doubling limit, so the radius stays finite and
     *     the worst case is bounded;
     *   - every candidate it admits lies beyond the protection level, so the
     *     PHMI grouping files it under the integrity-invalid weight. The extra
     *     candidates are available to the HMM but are not promoted to the
     *     high-emission group simply because we looked further.
     *
     * Public so that tests/test_cmm_candidate_search.cpp can pin the stop
     * conditions; the function is pure and has no effect on its own.
     *
     * @param candidates_found how many candidates the last search produced
     * @param min_candidates how many the configuration asks for; values below 1
     *        are clamped to 1, because validate() is not on every construction
     *        path and a zero target would make the fallback silently unreachable
     * @param doublings_done doublings already performed for this epoch
     * @param max_doublings hard cap on doublings for one epoch
     * @param search_radius the radius used for the last search
     * @return true if the caller should double the radius and search again
     */
    static bool should_expand_search_radius(size_t candidates_found,
                                            int min_candidates,
                                            int doublings_done,
                                            int max_doublings,
                                            double search_radius);

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
    static double compute_direction_penalty(const CORE::Point &obs_prev,
                                            const CORE::Point &obs_curr,
                                            const CORE::Point &edge_start,
                                            const CORE::Point &edge_end,
                                            const CovarianceMatrix &cov);

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
