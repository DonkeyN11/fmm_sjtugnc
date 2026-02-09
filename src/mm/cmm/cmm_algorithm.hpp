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

    // Calculate position uncertainty (2D)
    double get_2d_uncertainty() const {
        Matrix2d cov = to_2d_matrix();
        return std::sqrt(cov.determinant());
    }
};

/**
 * Configuration class for CMM algorithm
 */
struct CovarianceMapMatchConfig {
    /**
     * Constructor of CovarianceMapMatch configuration
     * @param k_arg the number of candidates
     * @param min_candidates_arg minimum number of candidates to keep
     * @param protection_level_multiplier_arg multiplier for protection level
     * @param reverse_tolerance reverse movement tolerance
     * @param window_length_arg sliding window length for trustworthiness
     */
    CovarianceMapMatchConfig(int k_arg = 8, int min_candidates_arg = 3,
                             double protection_level_multiplier_arg = 1.0,
                           double reverse_tolerance = 0.0,
                           bool normalized_arg = true,
                           bool use_mahalanobis_candidates_arg = true,
                           int window_length_arg = 10,
                           bool margin_used_trustworthiness_arg = true,
                           bool filtered_arg = true,
                           bool enable_candidate_filter_arg = true,
                           double candidate_filter_threshold_arg = 15.0,
                           bool enable_gap_bridging_arg = true,
                           double max_gap_distance_arg = 2000.0, /* in meters */
                        //    double min_gps_error_degrees_arg = 1.0e-6,
                           double max_interval_arg = 180.0, /* in seconds */
                           double trustworthiness_threshold_arg = 0.0,
                           double map_error_std_arg = 5.0e-6, /* in degrees */
                           double background_log_prob_arg = -20.0);

    int k;                          /**< Number of candidates */
    int min_candidates;             /**< Minimum number of candidates to keep */
    double protection_level_multiplier; /**< Multiplier for protection level */
    double reverse_tolerance;           /**< Reverse movement tolerance */
    bool normalized;                    /**< Whether to normalize emission probabilities */
    bool use_mahalanobis_candidates;    /**< Whether to use Mahalanobis-based candidate search */
    int window_length;                  /**< Sliding window length for trustworthiness */
    bool margin_used_trustworthiness;   /**< If true, use margin (top1-top2); else use top1 */
    bool filtered;                      /**< Whether to filter out points with no candidates/disconnected transitions */

    // --- New Parameters for Log-Space Filtering & Gap Handling ---
    bool enable_candidate_filter;       /**< Enable L2 candidate filtering based on relative log-probability */
    double candidate_filter_threshold;  /**< Log-probability threshold for filtering (default 15.0 -> exp(-15) ≈ 3e-7) */
    bool enable_gap_bridging;           /**< Enable skipping invalid points to bridge gaps */
    double max_gap_distance;            /**< Maximum physical distance (meters) to attempt bridging */

    // --- Minimum GPS Error for Emission Probability ---
    double min_gps_error_degrees;       /**< Minimum GPS error in degrees to prevent over-confidence (default 1e-5 ≈ 1.1m) */
    double max_interval;                /**< Maximum time interval to split trajectory */
    double trustworthiness_threshold;   /**< Threshold to filter out low-confidence matches */

    // --- New Parameters for Additive Map Noise and Background Noise ---
    double map_error_std;               /**< Map error standard deviation in degrees (default 5e-5 ≈ 5m). Added to GPS variance. */
    double background_log_prob;         /**< Background noise log probability (default -20.0). Used as "null hypothesis" in normalization. */

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
        : network_(network), graph_(graph), ubodt_(ubodt) {}

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

protected:
    /**
     * Calculate emission probability using covariance matrix (LOG-SPACE)
     * Returns the log of emission probability to prevent numerical underflow.
     * @param point_observed observed GPS point
     * @param point_candidate candidate point on road network
     * @param covariance covariance matrix of GPS observation
     * @param config CMM configuration containing min_gps_error_degrees
     * @return log emission probability
     */
    double calculate_emission_log_prob(const CORE::Point &point_observed,
                                       const CORE::Point &point_candidate,
                                       const CovarianceMatrix &covariance,
                                       const CovarianceMapMatchConfig &config) const;

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
    void initialize_first_layer(TGLayer *layer, const CovarianceMapMatchConfig &config);

    /**
     * Update probabilities between two layers in the transition graph (LOG-SPACE)
     * Performs L2 filtering and Top-K Normalization in Log-Space.
     * @param la_ptr layer a (previous layer)
     * @param lb_ptr layer b (current layer)
     * @param eu_dist Euclidean distance between two observed points
     * @param connected set to false if the layer is not connected with the next layer
     * @param config CMM configuration
     */
    void update_layer_cmm(TGLayer *la_ptr, TGLayer *lb_ptr,
                         double eu_dist,
                         bool *connected,
                         const CovarianceMapMatchConfig &config);

    /**
     * Update probabilities in a transition graph using CMM emission probabilities
     * @param tg transition graph
     * @param traj raw trajectory with covariance data
     * @param config map match configuration
     */
    void update_tg_cmm(TransitionGraph *tg,
                      const CMMTrajectory &traj,
                      const CovarianceMapMatchConfig &config);

    /**
     * Compute sliding-window trustworthiness scores and top-N paths.
     * @return pair of (trustworthiness margin per epoch, top-N log scores per epoch)
     */
    std::pair<std::vector<double>, std::vector<std::vector<double>>> compute_window_trustworthiness(
        const Traj_Candidates &tc,
        const std::vector<std::vector<double>> &log_emission_probabilities,
        const CMMTrajectory &traj,
        const CovarianceMapMatchConfig &config);

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
    const NETWORK::NetworkGraph &graph_;
    std::shared_ptr<UBODT> ubodt_;
};

} // MM
} // FMM

#endif // FMM_CMM_ALGORITHM_H_
