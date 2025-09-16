//
// Created for CMM implementation
// Covariance-based map matching algorithm
//

#include "mm/cmm/cmm_algorithm.hpp"
#include "algorithm/geom_algorithm.hpp"
#include "util/util.hpp"
#include "util/debug.hpp"
#include "io/gps_reader.hpp"
#include "io/mm_writer.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

using namespace FMM;
using namespace FMM::CORE;
using namespace FMM::NETWORK;
using namespace FMM::PYTHON;
using namespace FMM::MM;

// Implementation of CovarianceMapMatchConfig
CovarianceMapMatchConfig::CovarianceMapMatchConfig(int k_arg, int min_candidates_arg,
                                                   double protection_level_multiplier_arg,
                                                   double reverse_tolerance)
    : k(k_arg), min_candidates(min_candidates_arg),
      protection_level_multiplier(protection_level_multiplier_arg),
      reverse_tolerance(reverse_tolerance) {
}

void CovarianceMapMatchConfig::print() const {
    SPDLOG_INFO("CMMAlgorithmConfig");
    SPDLOG_INFO("k {} min_candidates {} protection_level_multiplier {} reverse_tolerance {}",
                k, min_candidates, protection_level_multiplier, reverse_tolerance);
}

CovarianceMapMatchConfig CovarianceMapMatchConfig::load_from_xml(
    const boost::property_tree::ptree &xml_data) {
    int k = xml_data.get("config.parameters.k", 8);
    int min_candidates = xml_data.get("config.parameters.min_candidates", 3);
    double protection_level_multiplier = xml_data.get("config.parameters.protection_level_multiplier", 2.0);
    double reverse_tolerance = xml_data.get("config.parameters.reverse_tolerance", 0.0);
    return CovarianceMapMatchConfig{k, min_candidates, protection_level_multiplier, reverse_tolerance};
}

CovarianceMapMatchConfig CovarianceMapMatchConfig::load_from_arg(
    const cxxopts::ParseResult &arg_data) {
    int k = arg_data["candidates"].as<int>();
    int min_candidates = arg_data["min_candidates"].as<int>();
    double protection_level_multiplier = arg_data["protection_level_multiplier"].as<double>();
    double reverse_tolerance = arg_data["reverse_tolerance"].as<double>();
    return CovarianceMapMatchConfig{k, min_candidates, protection_level_multiplier, reverse_tolerance};
}

void CovarianceMapMatchConfig::register_arg(cxxopts::Options &options) {
    options.add_options()
        ("k,candidates", "Number of candidates",
         cxxopts::value<int>()->default_value("8"))
        ("min_candidates", "Minimum number of candidates to keep",
         cxxopts::value<int>()->default_value("3"))
        ("protection_level_multiplier", "Multiplier for protection level",
         cxxopts::value<double>()->default_value("2.0"))
        ("reverse_tolerance", "Ratio of reverse movement allowed",
         cxxopts::value<double>()->default_value("0.0"));
}

void CovarianceMapMatchConfig::register_help(std::ostringstream &oss) {
    oss << "-k/--candidates (optional) <int>: Number of candidates (8)\n";
    oss << "--min_candidates (optional) <int>: Minimum number of candidates to keep (3)\n";
    oss << "--protection_level_multiplier (optional) <double>: Multiplier for protection level (2.0)\n";
    oss << "--reverse_tolerance (optional) <double>: proportion of reverse movement allowed on an edge\n";
}

bool CovarianceMapMatchConfig::validate() const {
    if (k <= 0 || min_candidates <= 0 || min_candidates > k ||
        protection_level_multiplier <= 0 || reverse_tolerance < 0 || reverse_tolerance > 1) {
        SPDLOG_CRITICAL("Invalid CMM parameter k {} min_candidates {} "
                       "protection_level_multiplier {} reverse_tolerance {}",
                       k, min_candidates, protection_level_multiplier, reverse_tolerance);
        return false;
    }
    return true;
}

// Implementation of CovarianceMapMatch

double CovarianceMapMatch::calculate_emission_probability(
    const CORE::Point &point_observed,
    const CORE::Point &point_candidate,
    const CovarianceMatrix &covariance) const {

    // Extract coordinates
    double obs_x = boost::geometry::get<0>(point_observed);
    double obs_y = boost::geometry::get<1>(point_observed);
    double cand_x = boost::geometry::get<0>(point_candidate);
    double cand_y = boost::geometry::get<1>(point_candidate);

    // Calculate difference vector
    double dx = obs_x - cand_x;
    double dy = obs_y - cand_y;
    Eigen::Vector2d diff(dx, dy);

    // Get 2D covariance matrix
    Eigen::Matrix2d cov = covariance.to_2d_matrix();

    // Calculate inverse of covariance matrix
    Eigen::Matrix2d cov_inv = cov.inverse();

    // Calculate Mahalanobis distance squared
    double mahalanobis_dist_sq = diff.transpose() * cov_inv * diff;

    // Calculate emission probability using multivariate Gaussian distribution
    double det = cov.determinant();
    double normalization = 1.0 / (2.0 * M_PI * std::sqrt(det));
    double probability = normalization * std::exp(-0.5 * mahalanobis_dist_sq);

    // Return log probability to avoid numerical underflow
    return std::log(probability);
}

Traj_Candidates CovarianceMapMatch::search_candidates_with_protection_level(
    const CORE::LineString &geom,
    const std::vector<CovarianceMatrix> &covariances,
    const std::vector<double> &protection_levels,
    const CovarianceMapMatchConfig &config) const {

    SPDLOG_DEBUG("Search candidates with protection level for {} points", geom.get_num_points());

    Traj_Candidates all_candidates;
    int num_points = geom.get_num_points();

    for (int i = 0; i < num_points; ++i) {
        // Get current point and its covariance/protection level
        CORE::Point point = geom.get_point(i);
        const CovarianceMatrix &cov = covariances[i];
        double protection_level = protection_levels[i];

        // Calculate search radius based on protection level and uncertainty
        double uncertainty = cov.get_2d_uncertainty();
        double search_radius = protection_level * config.protection_level_multiplier + uncertainty;

        SPDLOG_TRACE("Point {}: uncertainty={}, protection_level={}, search_radius={}",
                     i, uncertainty, protection_level, search_radius);

        // Search candidates using network's KNN search
        Point_Candidates point_candidates = network_.search_cs_knn(point, config.k, search_radius);

        // Filter candidates based on Mahalanobis distance
        Point_Candidates filtered_candidates;
        for (const Candidate &cand : point_candidates) {
            double mahalanobis_dist_sq = 0.0;

            // Calculate Mahalanobis distance
            double obs_x = boost::geometry::get<0>(point);
            double obs_y = boost::geometry::get<1>(point);
            double cand_x = boost::geometry::get<0>(cand.point);
            double cand_y = boost::geometry::get<1>(cand.point);

            double dx = obs_x - cand_x;
            double dy = obs_y - cand_y;
            Eigen::Vector2d diff(dx, dy);

            Eigen::Matrix2d cov = cov.to_2d_matrix();
            Eigen::Matrix2d cov_inv = cov.inverse();
            mahalanobis_dist_sq = diff.transpose() * cov_inv * diff;

            // Keep candidate if within reasonable Mahalanobis distance
            double mahalanobis_threshold = protection_level * protection_level;
            if (mahalanobis_dist_sq <= mahalanobis_threshold) {
                filtered_candidates.push_back(cand);
            }
        }

        // Ensure minimum number of candidates
        if (filtered_candidates.size() < config.min_candidates) {
            // Sort by distance and keep closest ones
            std::sort(point_candidates.begin(), point_candidates.end(),
                     [](const Candidate &a, const Candidate &b) {
                         return a.dist < b.dist;
                     });

            int keep_count = std::min(config.min_candidates, (int)point_candidates.size());
            filtered_candidates.assign(point_candidates.begin(),
                                     point_candidates.begin() + keep_count);
        }

        SPDLOG_TRACE("Point {}: {} candidates found", i, filtered_candidates.size());
        all_candidates.push_back(filtered_candidates);
    }

    SPDLOG_DEBUG("Candidate search completed");
    return all_candidates;
}

MatchResult CovarianceMapMatch::match_traj(const CMMTrajectory &traj,
                                          const CovarianceMapMatchConfig &config) {
    SPDLOG_DEBUG("Count of points in trajectory {}", traj.geom.get_num_points());

    // Validate trajectory
    if (!traj.is_valid()) {
        SPDLOG_ERROR("Invalid trajectory: covariance and protection level data mismatch");
        return MatchResult{};
    }

    SPDLOG_DEBUG("Search candidates with protection level");
    Traj_Candidates tc = search_candidates_with_protection_level(
        traj.geom, traj.covariances, traj.protection_levels, config);

    SPDLOG_DEBUG("Trajectory candidate {}", tc);
    if (tc.empty()) return MatchResult{};

    SPDLOG_DEBUG("Generate transition graph");
    // Create transition graph with dummy GPS error (will be overridden by CMM emission probs)
    TransitionGraph tg(tc, 1.0);

    SPDLOG_DEBUG("Update cost in transition graph using CMM");
    update_tg_cmm(&tg, traj, config);

    SPDLOG_DEBUG("Optimal path inference");
    TGOpath tg_opath = tg.backtrack();
    SPDLOG_DEBUG("Optimal path size {}", tg_opath.size());

    MatchedCandidatePath matched_candidate_path(tg_opath.size());
    std::transform(tg_opath.begin(), tg_opath.end(),
                   matched_candidate_path.begin(),
                   [](const TGNode *a) {
        return MatchedCandidate{*(a->c), a->ep, a->tp, a->cumu_prob, a->sp_dist};
    });

    O_Path opath(tg_opath.size());
    std::transform(tg_opath.begin(), tg_opath.end(),
                   opath.begin(),
                   [](const TGNode *a) {
        return a->c->edge->id;
    });

    std::vector<int> indices;
    const std::vector<Edge> &edges = network_.get_edges();
    C_Path cpath = ubodt_->construct_complete_path(traj.id, tg_opath, edges,
                                                   &indices,
                                                   config.reverse_tolerance);

    SPDLOG_DEBUG("Opath is {}", opath);
    SPDLOG_DEBUG("Indices is {}", indices);
    SPDLOG_DEBUG("Complete path is {}", cpath);

    LineString mgeom = network_.complete_path_to_geometry(traj.geom, cpath);
    return MatchResult{
        traj.id, matched_candidate_path, opath, cpath, indices, mgeom};
}

double CovarianceMapMatch::get_sp_dist(const Candidate *ca, const Candidate *cb,
                                      double reverse_tolerance) {
    NodeIndex s = ca->edge->target;
    NodeIndex e = cb->edge->source;
    const UBODTRecord *r = ubodt_->look_up(s, e);
    double sp_dist = ubodt_->get_sp_dist(r);
    if (sp_dist < 0) {
        // No path exists, try reverse direction
        s = ca->edge->source;
        e = cb->edge->target;
        r = ubodt_->look_up(s, e);
        sp_dist = ubodt_->get_sp_dist(r);
        if (sp_dist >= 0) {
            // Path exists in reverse direction
            double total_length = ca->edge->length + cb->edge->length;
            double reverse_dist = total_length - ca->offset - cb->offset;
            if (reverse_dist <= sp_dist * (1 + reverse_tolerance)) {
                return reverse_dist;
            }
        }
        return -1;
    } else {
        // Path exists in forward direction
        double dist = ca->edge->length - ca->offset + cb->offset;
        return dist + sp_dist;
    }
}

void CovarianceMapMatch::update_tg_cmm(TransitionGraph *tg,
                                       const CMMTrajectory &traj,
                                       const CovarianceMapMatchConfig &config) {
    std::vector<TGLayer> &layers = tg->get_layers();
    int N = layers.size();
    if (N == 0) return;

    // Reset first layer
    tg->reset_layer(&layers[0]);

    for (int level = 0; level < N - 1; ++level) {
        TGLayer *la_ptr = &layers[level];
        TGLayer *lb_ptr = &layers[level + 1];

        // Calculate Euclidean distance between consecutive points
        CORE::Point point_a = traj.geom.get_point(level);
        CORE::Point point_b = traj.geom.get_point(level + 1);
        double eu_dist = BG::distance(point_a, point_b);

        bool connected = true;
        update_layer_cmm(level, la_ptr, lb_ptr, eu_dist, config.reverse_tolerance,
                         &connected, traj, config);

        if (!connected) {
            SPDLOG_WARN("Trajectory disconnected at level {}", level);
        }
    }
}

void CovarianceMapMatch::update_layer_cmm(int level, TGLayer *la_ptr, TGLayer *lb_ptr,
                                         double eu_dist, double reverse_tolerance,
                                         bool *connected,
                                         const CMMTrajectory &traj,
                                         const CovarianceMapMatchConfig &config) {
    for (auto &node_a : *la_ptr) {
        double final_prob = -std::numeric_limits<double>::infinity();
        TGNode *next_opt_node = nullptr;
        double sp_dist = 0;

        for (auto &node_b : *lb_ptr) {
            // Calculate shortest path distance
            double cur_sp_dist = get_sp_dist(node_a.c, node_b.c, reverse_tolerance);
            if (cur_sp_dist < 0) continue;

            // Calculate transition probability
            double tp = TransitionGraph::calc_tp(cur_sp_dist, eu_dist);

            // Calculate cumulative probability
            double cumu_prob = node_a.cumu_prob + std::log(tp);

            if (cumu_prob > final_prob) {
                final_prob = cumu_prob;
                next_opt_node = &node_b;
                sp_dist = cur_sp_dist;
            }
        }

        if (next_opt_node != nullptr) {
            // Update the optimal node
            if (final_prob > next_opt_node->cumu_prob) {
                next_opt_node->prev = &node_a;
                next_opt_node->tp = std::exp(final_prob - node_a.cumu_prob); // Convert back from log
                next_opt_node->sp_dist = sp_dist;
                next_opt_node->cumu_prob = final_prob;
            }
        } else {
            *connected = false;
        }
    }
}

std::string CovarianceMapMatch::match_gps_file(
    const FMM::CONFIG::GPSConfig &gps_config,
    const FMM::CONFIG::ResultConfig &result_config,
    const CovarianceMapMatchConfig &cmm_config,
    bool use_omp) {

    std::ostringstream oss;
    std::string status;
    bool validate = true;

    if (!gps_config.validate()) {
        oss << "gps_config invalid\n";
        validate = false;
    }
    if (!result_config.validate()) {
        oss << "result_config invalid\n";
        validate = false;
    }
    if (!cmm_config.validate()) {
        oss << "cmm_config invalid\n";
        validate = false;
    }

    if (!validate) {
        oss << "match_gps_file canceled\n";
        return oss.str();
    }

    // For now, return message indicating that GPS file with covariance data is needed
    oss << "CMM match_gps_file requires trajectory data with covariance and protection level information.\n";
    oss << "Please provide enhanced GPS data files that include covariance matrices and protection levels.\n";
    oss << "Current implementation requires CMMTrajectory objects with complete metadata.\n";

    return oss.str();
}