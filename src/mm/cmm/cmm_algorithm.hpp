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
 */
struct CovarianceMatrix {
    double sdn;    // North standard deviation (m)
    double sde;    // East standard deviation (m)
    double sdu;    // Up standard deviation (m)
    double sdne;   // North-East covariance (m²)
    double sdeu;   // East-Up covariance (m²)
    double sdun;   // Up-North covariance (m²)

    // Convert to 2D covariance matrix (horizontal plane)
    Matrix2d to_2d_matrix() const {
        return Matrix2d(sdn * sdn, sdne, sdne, sde * sde);
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
     */
    CovarianceMapMatchConfig(int k_arg = 8, int min_candidates_arg = 3,
                           double protection_level_multiplier_arg = 2.0,
                           double reverse_tolerance = 0.0);

    int k;                          /**< Number of candidates */
    int min_candidates;             /**< Minimum number of candidates to keep */
    double protection_level_multiplier; /**< Multiplier for protection level */
    double reverse_tolerance;       /**< Reverse movement tolerance */

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
     * @return map matching result
     */
    MatchResult match_traj(const CMMTrajectory &traj,
                         const CovarianceMapMatchConfig &config);

    /**
     * Match GPS data stored in a file with covariance and protection level data
     * @param gps_config GPS configuration including covariance and protection level files
     * @param result_config result configuration
     * @param config map matching configuration
     * @param use_omp whether to use OpenMP
     * @return a string storing information about running time and statistics
     */
    std::string match_gps_file(
        const FMM::CONFIG::GPSConfig &gps_config,
        const FMM::CONFIG::ResultConfig &result_config,
        const CovarianceMapMatchConfig &config,
        bool use_omp = true);

protected:
    /**
     * Calculate emission probability using covariance matrix
     * @param point_observed observed GPS point
     * @param point_candidate candidate point on road network
     * @param covariance covariance matrix of GPS observation
     * @return emission probability
     */
    double calculate_emission_probability(const CORE::Point &point_observed,
                                        const CORE::Point &point_candidate,
                                        const CovarianceMatrix &covariance) const;

    /**
     * Search candidates based on protection level
     * @param geom trajectory geometry
     * @param covariances covariance matrices for each point
     * @param protection_levels protection levels for each point
     * @param config CMM configuration
     * @return trajectory candidates
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
     * Update probabilities in a transition graph using CMM emission probabilities
     * @param tg transition graph
     * @param traj raw trajectory with covariance data
     * @param config map match configuration
     */
    void update_tg_cmm(TransitionGraph *tg,
                      const CMMTrajectory &traj,
                      const CovarianceMapMatchConfig &config);

    /**
     * Update probabilities between two layers a and b in the transition graph
     * @param level the index of layer a
     * @param la_ptr layer a
     * @param lb_ptr layer b next to a
     * @param eu_dist Euclidean distance between two observed points
     * @param reverse_tolerance reverse movement tolerance
     * @param connected set to false if the layer is not connected with the next layer
     */
    void update_layer_cmm(int level, TGLayer *la_ptr, TGLayer *lb_ptr,
                         double eu_dist, double reverse_tolerance,
                         bool *connected,
                         const CMMTrajectory &traj,
                         const CovarianceMapMatchConfig &config);

private:
    const NETWORK::Network &network_;
    const NETWORK::NetworkGraph &graph_;
    std::shared_ptr<UBODT> ubodt_;
};

} // MM
} // FMM

#endif // FMM_CMM_ALGORITHM_H_
