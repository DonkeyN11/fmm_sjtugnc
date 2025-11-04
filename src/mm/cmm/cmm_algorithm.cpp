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

// #include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <optional>
#include <limits>
#include <memory>

#include <ogrsf_frmts.h>

#include <boost/property_tree/json_parser.hpp>

using namespace FMM;
using namespace FMM::CORE;
using namespace FMM::NETWORK;
using namespace FMM::PYTHON;
using namespace FMM::MM;

namespace {

std::string trim_copy(const std::string &input) {
    const auto begin = input.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = input.find_last_not_of(" \t\r\n");
    return input.substr(begin, end - begin + 1);
}

std::string to_lower_copy(const std::string &input) {
    std::string lowered = input;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lowered;
}

std::vector<std::string> split_line(const std::string &line, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    if (!line.empty() && line.back() == delimiter) {
        tokens.emplace_back("");
    }
    return tokens;
}

std::optional<double> parse_double(const std::string &value) {
    try {
        size_t parsed = 0;
        double result = std::stod(value, &parsed);
        if (parsed != value.size()) {
            return std::nullopt;
        }
        return result;
    } catch (const std::exception &) {
        return std::nullopt;
    }
}

std::optional<long long> parse_integer(const std::string &value) {
    try {
        size_t parsed = 0;
        long long result = std::stoll(value, &parsed);
        if (parsed != value.size()) {
            return std::nullopt;
        }
        return result;
    } catch (const std::exception &) {
        return std::nullopt;
    }
}

template <typename T>
bool parse_numeric_array(const std::string &json_text, std::vector<T> *output) {
    output->clear();
    const std::string trimmed = trim_copy(json_text);
    if (trimmed.empty()) {
        return false;
    }
    try {
        boost::property_tree::ptree tree;
        std::stringstream ss(trimmed);
        boost::property_tree::read_json(ss, tree);
        output->reserve(tree.size());
        for (const auto &node : tree) {
            output->push_back(node.second.get_value<T>());
        }
    } catch (const std::exception &ex) {
        SPDLOG_ERROR("Failed to parse numeric array from '{}': {}", trimmed, ex.what());
        return false;
    }
    return true;
}

bool parse_covariance_array(const std::string &json_text,
                            std::vector<CovarianceMatrix> *output) {
    output->clear();
    const std::string trimmed = trim_copy(json_text);
    if (trimmed.empty()) {
        return false;
    }
    try {
        boost::property_tree::ptree tree;
        std::stringstream ss(trimmed);
        boost::property_tree::read_json(ss, tree);
        output->reserve(tree.size());
        for (const auto &row : tree) {
            std::vector<double> values;
            values.reserve(row.second.size());
            for (const auto &value_node : row.second) {
                values.push_back(value_node.second.get_value<double>());
            }
            if (values.size() < 6) {
                SPDLOG_WARN("Covariance row has {} values, expected at least 6", values.size());
                return false;
            }
            output->push_back(
                CovarianceMatrix{values[0], values[1], values[2],
                                 values[3], values[4], values[5]});
        }
    } catch (const std::exception &ex) {
        SPDLOG_ERROR("Failed to parse covariance array: {}", ex.what());
        return false;
    }
    return true;
}

Matrix2d multiply_matrices(const Matrix2d &lhs, const Matrix2d &rhs) {
    return Matrix2d(
        lhs.m[0][0] * rhs.m[0][0] + lhs.m[0][1] * rhs.m[1][0],
        lhs.m[0][0] * rhs.m[0][1] + lhs.m[0][1] * rhs.m[1][1],
        lhs.m[1][0] * rhs.m[0][0] + lhs.m[1][1] * rhs.m[1][0],
        lhs.m[1][0] * rhs.m[0][1] + lhs.m[1][1] * rhs.m[1][1]);
}

Matrix2d transpose_matrix(const Matrix2d &mat) {
    return Matrix2d(mat.m[0][0], mat.m[1][0], mat.m[0][1], mat.m[1][1]);
}

bool geometry_is_projected(const CORE::LineString &geom) {
    const int num_points = geom.get_num_points();
    for (int i = 0; i < num_points; ++i) {
        double x = geom.get_x(i);
        double y = geom.get_y(i);
        if (std::abs(x) > 180.0 || std::abs(y) > 90.0) {
            return true;
        }
    }
    return false;
}

bool transform_linestring(CORE::LineString *line,
                          OGRCoordinateTransformation *transform) {
    if (!line || transform == nullptr) {
        return false;
    }
    const int num_points = line->get_num_points();
    for (int idx = 0; idx < num_points; ++idx) {
        double x = line->get_x(idx);
        double y = line->get_y(idx);
        if (!transform->Transform(1, &x, &y)) {
            return false;
        }
        line->set_x(idx, x);
        line->set_y(idx, y);
    }
    return true;
}

struct TransformInfo {
    Matrix2d jacobian;
    double proj_x;
    double proj_y;
    double scale_lon;
    double scale_lat;
};

std::optional<TransformInfo> compute_transform_info(
    OGRCoordinateTransformation *transform,
    double lon_deg,
    double lat_deg) {
    if (transform == nullptr) {
        return std::nullopt;
    }
    constexpr double delta = 1e-6;

    double x0 = lon_deg;
    double y0 = lat_deg;
    if (!transform->Transform(1, &x0, &y0)) {
        return std::nullopt;
    }

    double x_lon = lon_deg + delta;
    double y_lon = lat_deg;
    if (!transform->Transform(1, &x_lon, &y_lon)) {
        return std::nullopt;
    }

    double x_lat = lon_deg;
    double y_lat = lat_deg + delta;
    if (!transform->Transform(1, &x_lat, &y_lat)) {
        return std::nullopt;
    }

    double dx_dlon = (x_lon - x0) / delta;
    double dy_dlon = (y_lon - y0) / delta;
    double dx_dlat = (x_lat - x0) / delta;
    double dy_dlat = (y_lat - y0) / delta;

    TransformInfo info;
    info.proj_x = x0;
    info.proj_y = y0;
    info.jacobian = Matrix2d(dx_dlon, dx_dlat, dy_dlon, dy_dlat);
    info.scale_lon = std::hypot(dx_dlon, dy_dlon);
    info.scale_lat = std::hypot(dx_dlat, dy_dlat);
    return info;
}

bool maybe_reproject_trajectories(std::vector<CMMTrajectory> *trajectories,
                                  const NETWORK::Network &network,
                                  bool convert_to_projected) {
    if (!convert_to_projected || trajectories == nullptr || trajectories->empty()) {
        return false;
    }
    if (!network.is_projected()) {
        SPDLOG_WARN("Coordinate conversion requested but network CRS is not projected; skip trajectory reprojection.");
        return false;
    }
    if (!network.has_spatial_ref()) {
        SPDLOG_WARN("Network CRS information unavailable; skip trajectory reprojection.");
        return false;
    }

    bool all_projected = true;
    for (const auto &traj : *trajectories) {
        if (!geometry_is_projected(traj.geom)) {
            all_projected = false;
            break;
        }
    }
    if (all_projected) {
        SPDLOG_INFO("Trajectories already appear to be in a projected CRS; no reprojection applied.");
        return false;
    }

    OGRSpatialReference target_sr;
    if (target_sr.importFromWkt(network.get_spatial_ref_wkt().c_str()) != OGRERR_NONE) {
        SPDLOG_WARN("Failed to import network CRS from WKT; skip trajectory reprojection.");
        return false;
    }
    target_sr.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    OGRSpatialReference source_sr;
    source_sr.SetWellKnownGeogCS("WGS84");
    source_sr.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    OGRCoordinateTransformation *transform = OGRCreateCoordinateTransformation(&source_sr, &target_sr);
    if (transform == nullptr) {
        SPDLOG_WARN("Failed to create coordinate transformation for trajectories.");
        return false;
    }

    bool transformed_any = false;
    try {
        for (auto &traj : *trajectories) {
            if (geometry_is_projected(traj.geom)) {
                continue;
            }
            const int num_points = traj.geom.get_num_points();
            for (int idx = 0; idx < num_points; ++idx) {
                double lon = traj.geom.get_x(idx);
                double lat = traj.geom.get_y(idx);
                auto info_opt = compute_transform_info(transform, lon, lat);
                if (!info_opt) {
                    throw std::runtime_error("Coordinate transformation failed for trajectory point.");
                }
                const TransformInfo &info = *info_opt;
                traj.geom.set_x(idx, info.proj_x);
                traj.geom.set_y(idx, info.proj_y);

                if (idx < static_cast<int>(traj.covariances.size())) {
                    CovarianceMatrix &cov = traj.covariances[idx];
                    Matrix2d cov2d = cov.to_2d_matrix();
                    Matrix2d temp = multiply_matrices(info.jacobian, cov2d);
                    Matrix2d cov_new = multiply_matrices(temp, transpose_matrix(info.jacobian));
                    cov.sdn = std::sqrt(std::max(cov_new.m[0][0], 0.0));
                    cov.sde = std::sqrt(std::max(cov_new.m[1][1], 0.0));
                    cov.sdne = cov_new.m[0][1];
                }

                if (idx < static_cast<int>(traj.protection_levels.size())) {
                    double scale_factor = std::max(info.scale_lon, info.scale_lat);
                    if (scale_factor > 0) {
                        traj.protection_levels[idx] *= scale_factor;
                    }
                }
            }
            transformed_any = true;
        }
    } catch (const std::exception &ex) {
        OCTDestroyCoordinateTransformation(transform);
        throw;
    }

    OCTDestroyCoordinateTransformation(transform);
    if (transformed_any) {
        SPDLOG_INFO("Reprojected {} trajectories to match projected network CRS.", trajectories->size());
    }
    return transformed_any;
}

} // namespace

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
    double protection_level_multiplier = xml_data.get("config.parameters.protection_level_multiplier", 1.0);
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
    oss << "--protection_level_multiplier (optional) <double>: Multiplier for protection level (1.0)\n";
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

    double obs_x = boost::geometry::get<0>(point_observed);
    double obs_y = boost::geometry::get<1>(point_observed);
    double cand_x = boost::geometry::get<0>(point_candidate);
    double cand_y = boost::geometry::get<1>(point_candidate);

    double dx = obs_x - cand_x;
    double dy = obs_y - cand_y;

    Matrix2d cov = covariance.to_2d_matrix();
    Matrix2d cov_inv = cov.inverse();
    double det = cov.determinant();
    if (det <= 0) {
        return 0.0;
    }

    double mahalanobis_dist_sq = cov_inv.m[0][0] * dx * dx +
                                 2 * cov_inv.m[0][1] * dx * dy +
                                 cov_inv.m[1][1] * dy * dy;

    double normalization = 1.0 / (2.0 * M_PI * std::sqrt(det));
    return normalization * std::exp(-0.5 * mahalanobis_dist_sq);
}

CandidateSearchResult CovarianceMapMatch::search_candidates_with_protection_level(
    const CORE::LineString &geom,
    const std::vector<CovarianceMatrix> &covariances,
    const std::vector<double> &protection_levels,
    const CovarianceMapMatchConfig &config) const {

    SPDLOG_DEBUG("Search candidates with protection level for {} points", geom.get_num_points());

    CandidateSearchResult result;
    int num_points = geom.get_num_points();
    result.candidates.reserve(num_points);
    result.emission_probabilities.reserve(num_points);

    for (int i = 0; i < num_points; ++i) {
        CORE::Point point = geom.get_point(i);
        const CovarianceMatrix &cov = covariances[i];
        double protection_level = protection_levels[i];

        // double uncertainty = cov.get_2d_uncertainty(); uncertainty is already included in the covariance
        // double search_radius = protection_level * config.protection_level_multiplier + uncertainty;
        double search_radius = protection_level * config.protection_level_multiplier;

        SPDLOG_TRACE("Point {}: protection_level={}, search_radius={}",
                     i, protection_level, search_radius);

        CORE::LineString single_point_geom;
        single_point_geom.add_point(point);
        Traj_Candidates traj_candidates = network_.search_tr_cs_knn(single_point_geom, config.k, search_radius);
        Point_Candidates point_candidates = traj_candidates.empty() ? Point_Candidates() : traj_candidates[0];

        Matrix2d cov_mat = cov.to_2d_matrix();
        Matrix2d cov_inv = cov_mat.inverse();
        double det = cov_mat.determinant();
        bool valid_covariance = det > 0;

        Point_Candidates selected_candidates;
        std::vector<double> raw_probabilities;
        selected_candidates.reserve(point_candidates.size());
        raw_probabilities.reserve(point_candidates.size());

        double mahalanobis_threshold = protection_level * protection_level;

        double obs_x = boost::geometry::get<0>(point);
        double obs_y = boost::geometry::get<1>(point);

        for (const Candidate &cand : point_candidates) {
            double cand_x = boost::geometry::get<0>(cand.point);
            double cand_y = boost::geometry::get<1>(cand.point);
            double dx = obs_x - cand_x;
            double dy = obs_y - cand_y;

            double mahalanobis_dist_sq = cov_inv.m[0][0] * dx * dx +
                                         2 * cov_inv.m[0][1] * dx * dy +
                                         cov_inv.m[1][1] * dy * dy;

            if (mahalanobis_dist_sq <= mahalanobis_threshold) {
                selected_candidates.push_back(cand);

                double probability = 0.0;
                if (valid_covariance) {
                    double normalization = 1.0 / (2.0 * M_PI * std::sqrt(det));
                    probability = normalization * std::exp(-0.5 * mahalanobis_dist_sq);
                }
                raw_probabilities.push_back(probability);
            }
        }

        if (selected_candidates.size() < static_cast<size_t>(config.min_candidates) && !point_candidates.empty()) {
            Point_Candidates sorted_candidates = point_candidates;
            std::sort(sorted_candidates.begin(), sorted_candidates.end(),
                      [](const Candidate &a, const Candidate &b) {
                          return a.dist < b.dist;
                      });

            size_t keep_count = std::min(static_cast<size_t>(config.min_candidates), sorted_candidates.size());
            for (size_t idx = 0; idx < keep_count; ++idx) {
                const Candidate &candidate = sorted_candidates[idx];
                auto already_selected = std::any_of(selected_candidates.begin(), selected_candidates.end(),
                                                    [&candidate](const Candidate &existing) {
                                                        return existing.index == candidate.index &&
                                                               existing.edge == candidate.edge &&
                                                               existing.offset == candidate.offset;
                                                    });
                if (!already_selected) {
                    selected_candidates.push_back(candidate);
                    raw_probabilities.push_back(0.0);
                }
            }
        }

        double prob_sum = std::accumulate(raw_probabilities.begin(), raw_probabilities.end(), 0.0);
        std::vector<double> normalized_probabilities;
        normalized_probabilities.reserve(raw_probabilities.size());

        if (prob_sum > 0.0) {
            for (double prob : raw_probabilities) {
                normalized_probabilities.push_back(prob / prob_sum);
            }
        } else if (!raw_probabilities.empty()) {
            double uniform_prob = 1.0 / raw_probabilities.size();
            normalized_probabilities.assign(raw_probabilities.size(), uniform_prob);
            SPDLOG_WARN("Point {}: covariance determinant non-positive or no candidates within PL, using uniform emission", i);
        }

        SPDLOG_TRACE("Point {}: {} candidates kept", i, selected_candidates.size());
        result.candidates.push_back(std::move(selected_candidates));
        result.emission_probabilities.push_back(std::move(normalized_probabilities));
    }

    SPDLOG_DEBUG("Candidate search completed");
    return result;
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
    CandidateSearchResult candidate_result = search_candidates_with_protection_level(
        traj.geom, traj.covariances, traj.protection_levels, config);

    const Traj_Candidates &tc = candidate_result.candidates;
    const std::vector<std::vector<double>> &emission_probabilities = candidate_result.emission_probabilities;

    SPDLOG_DEBUG("Trajectory candidate {}", tc);
    if (tc.empty()) return MatchResult{};

    SPDLOG_DEBUG("Generate transition graph");
    TransitionGraph tg(tc, emission_probabilities);

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
    auto *r = ubodt_->look_up(s, e);
    double sp_dist = r ? r->cost : -1;
    if (sp_dist < 0) {
        // No path exists, try reverse direction
        s = ca->edge->source;
        e = cb->edge->target;
        r = ubodt_->look_up(s, e);
        sp_dist = r ? r->cost : -1;
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
        double eu_dist = boost::geometry::distance(point_a, point_b);

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
    bool layer_connected = false;
    for (auto &node_a : *la_ptr) {
        if (!std::isfinite(node_a.cumu_prob)) {
            continue;
        }

        for (auto &node_b : *lb_ptr) {
            if (node_b.ep <= 0) {
                continue;
            }

            double cur_sp_dist = get_sp_dist(node_a.c, node_b.c, reverse_tolerance);
            if (cur_sp_dist < 0) {
                continue;
            }

            double tp = TransitionGraph::calc_tp(cur_sp_dist, eu_dist);
            if (tp <= 0) {
                continue;
            }

            double temp = node_a.cumu_prob + std::log(tp) + std::log(node_b.ep);
            if (temp > node_b.cumu_prob) {
                node_b.cumu_prob = temp;
                node_b.prev = &node_a;
                node_b.tp = tp;
                node_b.sp_dist = cur_sp_dist;
                layer_connected = true;
            }
        }
    }

    if (connected != nullptr) {
        *connected = layer_connected;
    }
}

std::string CovarianceMapMatch::match_gps_file(
    const FMM::CONFIG::GPSConfig &gps_config,
    const FMM::CONFIG::ResultConfig &result_config,
    const CovarianceMapMatchConfig &cmm_config,
    bool convert_to_projected,
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

    std::ifstream ifs(gps_config.file);
    if (!ifs.is_open()) {
        SPDLOG_CRITICAL("Failed to open GPS file {}", gps_config.file);
        oss << "Failed to open GPS file " << gps_config.file << '\n';
        oss << "match_gps_file canceled\n";
        return oss.str();
    }

    std::string header_line;
    if (!std::getline(ifs, header_line)) {
        SPDLOG_CRITICAL("GPS file {} is empty", gps_config.file);
        oss << "GPS file empty\n";
        oss << "match_gps_file canceled\n";
        return oss.str();
    }

    header_line = trim_copy(header_line);
    if (header_line.empty()) {
        SPDLOG_CRITICAL("GPS file {} header is empty", gps_config.file);
        oss << "GPS header empty\n";
        oss << "match_gps_file canceled\n";
        return oss.str();
    }

    const std::vector<std::string> headers = split_line(header_line, ';');
    std::unordered_map<std::string, int> header_index;
    for (int idx = 0; idx < static_cast<int>(headers.size()); ++idx) {
        header_index[to_lower_copy(trim_copy(headers[idx]))] = idx;
    }

    auto find_index = [&](const std::vector<std::string> &candidates) -> int {
        for (const auto &candidate : candidates) {
            const std::string key = to_lower_copy(trim_copy(candidate));
            auto it = header_index.find(key);
            if (it != header_index.end()) {
                return it->second;
            }
        }
        return -1;
    };

    const int id_idx = find_index({gps_config.id, "id"});
    if (id_idx < 0) {
        SPDLOG_CRITICAL("Required id column not found in {}", gps_config.file);
        oss << "id column not found\n";
        oss << "match_gps_file canceled\n";
        return oss.str();
    }

    const int geom_idx = find_index({gps_config.geom, "geom"});
    const int timestamp_idx = find_index({gps_config.timestamp, "timestamps", "timestamp"});
    int covariance_idx = find_index({"covariances", "covariance", "covariance_json"});
    int protection_idx = find_index({"protection_levels", "protection_level", "pl"});

    bool aggregated_format = geom_idx >= 0 &&
                             timestamp_idx >= 0 &&
                             covariance_idx >= 0 &&
                             protection_idx >= 0;

    const int x_idx = aggregated_format ? -1 : find_index({gps_config.x, "x"});
    const int y_idx = aggregated_format ? -1 : find_index({gps_config.y, "y"});
    const int sdn_idx = aggregated_format ? -1 : find_index({"sdn"});
    const int sde_idx = aggregated_format ? -1 : find_index({"sde"});
    const int sdne_idx = aggregated_format ? -1 : find_index({"sdne"});
    const int sdu_idx = aggregated_format ? -1 : find_index({"sdu"});
    const int sdeu_idx = aggregated_format ? -1 : find_index({"sdeu"});
    const int sdun_idx = aggregated_format ? -1 : find_index({"sdun"});

    if (!aggregated_format) {
        if (timestamp_idx < 0) {
            SPDLOG_CRITICAL("Timestamp column not found in {}", gps_config.file);
            oss << "timestamp column not found\n";
            oss << "match_gps_file canceled\n";
            return oss.str();
        }
        if (protection_idx < 0) {
            SPDLOG_CRITICAL("Protection level column not found in {}", gps_config.file);
            oss << "protection_level column not found\n";
            oss << "match_gps_file canceled\n";
            return oss.str();
        }
        if (x_idx < 0 || y_idx < 0 || sdn_idx < 0 || sde_idx < 0) {
            SPDLOG_CRITICAL("Point-based GPS file requires x, y, sdn, sde columns");
            oss << "point GPS format missing required covariance columns\n";
            oss << "match_gps_file canceled\n";
            return oss.str();
        }
    }

    std::vector<CMMTrajectory> trajectories;
    trajectories.reserve(1024);
    int invalid_records = 0;
    int line_number = 1;

    if (aggregated_format) {
        const int required_max_index = std::max(
            std::max(id_idx, geom_idx),
            std::max(timestamp_idx, std::max(covariance_idx, protection_idx)));

        std::string line;
        while (std::getline(ifs, line)) {
            ++line_number;
            line = trim_copy(line);
            if (line.empty()) {
                continue;
            }
            std::vector<std::string> columns = split_line(line, ';');
            if (static_cast<int>(columns.size()) <= required_max_index) {
                SPDLOG_WARN("Line {} skipped: insufficient columns ({})", line_number, columns.size());
                ++invalid_records;
                continue;
            }

            const std::string id_token = trim_copy(columns[id_idx]);
            const auto id_opt = parse_integer(id_token);
            if (!id_opt) {
                SPDLOG_WARN("Line {} skipped: invalid id '{}'", line_number, id_token);
                ++invalid_records;
                continue;
            }
            if (*id_opt > std::numeric_limits<int>::max() ||
                *id_opt < std::numeric_limits<int>::min()) {
                SPDLOG_WARN("Line {} skipped: id {} out of range", line_number, *id_opt);
                ++invalid_records;
                continue;
            }
            const int traj_id = static_cast<int>(*id_opt);

            CORE::LineString geom;
            try {
                boost::geometry::read_wkt(trim_copy(columns[geom_idx]), geom.get_geometry());
            } catch (const std::exception &ex) {
                SPDLOG_WARN("Line {} skipped: invalid geometry '{}': {}", line_number, columns[geom_idx], ex.what());
                ++invalid_records;
                continue;
            }

            std::vector<double> timestamps;
            if (!parse_numeric_array(columns[timestamp_idx], &timestamps)) {
                SPDLOG_WARN("Line {} skipped: unable to parse timestamps", line_number);
                ++invalid_records;
                continue;
            }

            std::vector<CovarianceMatrix> covariances;
            if (!parse_covariance_array(columns[covariance_idx], &covariances)) {
                SPDLOG_WARN("Line {} skipped: unable to parse covariances", line_number);
                ++invalid_records;
                continue;
            }

            std::vector<double> protection_levels;
            if (!parse_numeric_array(columns[protection_idx], &protection_levels)) {
                SPDLOG_WARN("Line {} skipped: unable to parse protection levels", line_number);
                ++invalid_records;
                continue;
            }

            const int num_points = geom.get_num_points();
            if (num_points == 0) {
                SPDLOG_WARN("Line {} skipped: geometry contains no points", line_number);
                ++invalid_records;
                continue;
            }
            if (!timestamps.empty() && static_cast<int>(timestamps.size()) != num_points) {
                SPDLOG_WARN("Line {} skipped: timestamps size {} does not match point count {}",
                            line_number, timestamps.size(), num_points);
                ++invalid_records;
                continue;
            }
            if (static_cast<int>(covariances.size()) != num_points ||
                static_cast<int>(protection_levels.size()) != num_points) {
                SPDLOG_WARN("Line {} skipped: metadata length mismatch (covariances {}, protection {}, points {})",
                            line_number, covariances.size(), protection_levels.size(), num_points);
                ++invalid_records;
                continue;
            }

            CMMTrajectory traj;
            traj.id = traj_id;
            traj.geom = std::move(geom);
            traj.timestamps = std::move(timestamps);
            traj.covariances = std::move(covariances);
            traj.protection_levels = std::move(protection_levels);

            if (!traj.is_valid()) {
                SPDLOG_WARN("Line {} skipped: trajectory metadata invalid", line_number);
                ++invalid_records;
                continue;
            }

            trajectories.push_back(std::move(traj));
        }
    } else {
        struct TrajectoryBuilder {
            CORE::LineString geom;
            std::vector<double> timestamps;
            std::vector<CovarianceMatrix> covariances;
            std::vector<double> protection_levels;
        };

        std::unordered_map<long long, size_t> id_to_index;
        std::vector<long long> insertion_order;
        std::vector<TrajectoryBuilder> builders;

        const int required_max_index = [&]() {
            int max_index = id_idx;
            max_index = std::max(max_index, timestamp_idx);
            max_index = std::max(max_index, protection_idx);
            max_index = std::max(max_index, x_idx);
            max_index = std::max(max_index, y_idx);
            max_index = std::max(max_index, sdn_idx);
            max_index = std::max(max_index, sde_idx);
            max_index = std::max(max_index, sdne_idx);
            max_index = std::max(max_index, sdu_idx);
            max_index = std::max(max_index, sdeu_idx);
            max_index = std::max(max_index, sdun_idx);
            return max_index;
        }();

        std::string line;
        while (std::getline(ifs, line)) {
            ++line_number;
            line = trim_copy(line);
            if (line.empty()) {
                continue;
            }
            std::vector<std::string> columns = split_line(line, ';');
            if (static_cast<int>(columns.size()) <= required_max_index) {
                SPDLOG_WARN("Line {} skipped: insufficient columns ({})", line_number, columns.size());
                ++invalid_records;
                continue;
            }

            const std::string id_token = trim_copy(columns[id_idx]);
            const auto id_opt = parse_integer(id_token);
            if (!id_opt) {
                SPDLOG_WARN("Line {} skipped: invalid id '{}'", line_number, id_token);
                ++invalid_records;
                continue;
            }
            if (*id_opt > std::numeric_limits<int>::max() ||
                *id_opt < std::numeric_limits<int>::min()) {
                SPDLOG_WARN("Line {} skipped: id {} out of range", line_number, *id_opt);
                ++invalid_records;
                continue;
            }
            const long long id_value = *id_opt;

            auto get_builder_index = [&]() -> size_t {
                auto it = id_to_index.find(id_value);
                if (it != id_to_index.end()) {
                    return it->second;
                }
                const size_t new_index = builders.size();
                id_to_index.emplace(id_value, new_index);
                insertion_order.push_back(id_value);
                builders.emplace_back();
                builders.back().geom = CORE::LineString();
                return new_index;
            };

            const auto timestamp_opt = parse_double(trim_copy(columns[timestamp_idx]));
            const auto x_opt = parse_double(trim_copy(columns[x_idx]));
            const auto y_opt = parse_double(trim_copy(columns[y_idx]));
            const auto sdn_opt = parse_double(trim_copy(columns[sdn_idx]));
            const auto sde_opt = parse_double(trim_copy(columns[sde_idx]));
            const auto sdne_opt = (sdne_idx >= 0) ? parse_double(trim_copy(columns[sdne_idx])) : std::optional<double>(0.0);
            const auto sdu_opt = (sdu_idx >= 0) ? parse_double(trim_copy(columns[sdu_idx])) : std::optional<double>(0.0);
            const auto sdeu_opt = (sdeu_idx >= 0) ? parse_double(trim_copy(columns[sdeu_idx])) : std::optional<double>(0.0);
            const auto sdun_opt = (sdun_idx >= 0) ? parse_double(trim_copy(columns[sdun_idx])) : std::optional<double>(0.0);
            const auto protection_opt = parse_double(trim_copy(columns[protection_idx]));

            if (!timestamp_opt || !x_opt || !y_opt || !sdn_opt || !sde_opt || !protection_opt || !sdne_opt || !sdu_opt || !sdeu_opt || !sdun_opt) {
                SPDLOG_WARN("Line {} skipped: invalid numeric values", line_number);
                ++invalid_records;
                continue;
            }

            const size_t builder_index = get_builder_index();
            TrajectoryBuilder &builder = builders[builder_index];

            CORE::Point pt((*x_opt), (*y_opt));
            builder.geom.add_point(pt);
            builder.timestamps.push_back(*timestamp_opt);
            builder.covariances.push_back(CovarianceMatrix{
                *sdn_opt, *sde_opt, *sdu_opt, *sdne_opt, *sdeu_opt, *sdun_opt
            });
            builder.protection_levels.push_back(*protection_opt);
        }

        trajectories.reserve(builders.size());
        for (const long long id_value : insertion_order) {
            const size_t builder_index = id_to_index[id_value];
            TrajectoryBuilder &builder = builders[builder_index];
            const int num_points = builder.geom.get_num_points();
            if (num_points == 0) {
                SPDLOG_WARN("Trajectory {} skipped: no points collected", id_value);
                ++invalid_records;
                continue;
            }
            if (static_cast<int>(builder.timestamps.size()) != num_points ||
                static_cast<int>(builder.covariances.size()) != num_points ||
                static_cast<int>(builder.protection_levels.size()) != num_points) {
                SPDLOG_WARN("Trajectory {} skipped: metadata length mismatch (timestamps {}, covariance {}, protection {})",
                            id_value, builder.timestamps.size(),
                            builder.covariances.size(), builder.protection_levels.size());
                ++invalid_records;
                continue;
            }

            if (id_value > std::numeric_limits<int>::max() ||
                id_value < std::numeric_limits<int>::min()) {
                SPDLOG_WARN("Trajectory {} skipped: id out of int range", id_value);
                ++invalid_records;
                continue;
            }

            CMMTrajectory traj;
            traj.id = static_cast<int>(id_value);
            traj.geom = std::move(builder.geom);
            traj.timestamps = std::move(builder.timestamps);
            traj.covariances = std::move(builder.covariances);
            traj.protection_levels = std::move(builder.protection_levels);

            if (!traj.is_valid()) {
                SPDLOG_WARN("Trajectory {} skipped: invalid metadata", id_value);
                ++invalid_records;
                continue;
            }

            trajectories.push_back(std::move(traj));
        }
    }

    SPDLOG_INFO("Loaded {} trajectories ({} skipped) from {}", trajectories.size(), invalid_records, gps_config.file);
    ifs.close();

    bool trajectories_reprojected = false;
    if (convert_to_projected) {
        try {
            trajectories_reprojected = maybe_reproject_trajectories(&trajectories, network_, convert_to_projected);
        } catch (const std::exception &ex) {
            SPDLOG_WARN("Trajectory reprojection failed: {}", ex.what());
        }
    }

    FMM::IO::CSVMatchResultWriter writer(result_config.file, result_config.output_config);
    std::unique_ptr<OGRCoordinateTransformation, decltype(&OCTDestroyCoordinateTransformation)>
        output_transform(nullptr, OCTDestroyCoordinateTransformation);
    if (convert_to_projected && network_.is_projected() && network_.has_spatial_ref()) {
        OGRSpatialReference projected_sr;
        if (projected_sr.importFromWkt(network_.get_spatial_ref_wkt().c_str()) == OGRERR_NONE) {
            projected_sr.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
            OGRSpatialReference geographic_sr;
            geographic_sr.SetWellKnownGeogCS("WGS84");
            geographic_sr.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
            if (OGRCoordinateTransformation *ct = OGRCreateCoordinateTransformation(&projected_sr, &geographic_sr)) {
                output_transform.reset(ct);
            } else {
                SPDLOG_WARN("Failed to create output coordinate transformation; results remain in projected CRS.");
            }
        } else {
            SPDLOG_WARN("Failed to reconstruct network CRS for output transformation.");
        }
    }
    OGRCoordinateTransformation *output_transform_ptr = output_transform.get();
    if (!trajectories_reprojected) {
        output_transform_ptr = nullptr;
    }
    auto apply_output_transform = [&](CORE::Trajectory *traj, MM::MatchResult *match) {
        if (output_transform_ptr == nullptr || traj == nullptr || match == nullptr) {
            return;
        }
        if (!transform_linestring(&traj->geom, output_transform_ptr)) {
            SPDLOG_WARN("Failed to transform trajectory geometry back to geographic CRS; keeping projected coordinates.");
            return;
        }
        if (!transform_linestring(&match->mgeom, output_transform_ptr)) {
            SPDLOG_WARN("Failed to transform matched geometry back to geographic CRS; keeping projected coordinates.");
        }
        for (auto &matched : match->opt_candidate_path) {
            double px = boost::geometry::get<0>(matched.c.point);
            double py = boost::geometry::get<1>(matched.c.point);
            if (output_transform_ptr->Transform(1, &px, &py)) {
                boost::geometry::set<0>(matched.c.point, px);
                boost::geometry::set<1>(matched.c.point, py);
            } else {
                SPDLOG_TRACE("Failed to transform candidate point back to geographic CRS.");
            }
        }
    };
    const int step_size = 1000;
    int progress = 0;
    int points_matched = 0;
    int total_points = 0;
    int traj_matched = 0;
    int total_trajs = 0;
    auto begin_time = UTIL::get_current_time();

    if (use_omp && trajectories.size() > 1) {
#ifdef _OPENMP
        const int trajectories_count = static_cast<int>(trajectories.size());
        #pragma omp parallel for schedule(dynamic)
        for (int idx = 0; idx < trajectories_count; ++idx) {
            const CMMTrajectory &trajectory = trajectories[idx];
            MM::MatchResult result = match_traj(trajectory, cmm_config);
            CORE::Trajectory simple_traj{trajectory.id, trajectory.geom, trajectory.timestamps};
            #pragma omp critical(writer_section)
            {
                CORE::Trajectory output_traj = simple_traj;
                MM::MatchResult output_result = result;
                apply_output_transform(&output_traj, &output_result);
                writer.write_result(output_traj, output_result);
            }
            const int points_in_tr = trajectory.geom.get_num_points();
            #pragma omp critical(progress_section)
            {
                ++progress;
                ++total_trajs;
                total_points += points_in_tr;
                if (!result.cpath.empty()) {
                    points_matched += points_in_tr;
                    ++traj_matched;
                }
                if (step_size > 0 && progress % step_size == 0) {
                    std::stringstream buf;
                    buf << "Progress " << progress << '\n';
                    std::cout << buf.rdbuf();
                }
            }
        }
#else
        use_omp = false;
#endif
    }

    if (!use_omp || trajectories.size() <= 1) {
        for (const auto &trajectory : trajectories) {
            if (progress % step_size == 0) {
                SPDLOG_INFO("Progress {}", progress);
            }
            MM::MatchResult result = match_traj(trajectory, cmm_config);
            CORE::Trajectory simple_traj{trajectory.id, trajectory.geom, trajectory.timestamps};
            CORE::Trajectory output_traj = simple_traj;
            MM::MatchResult output_result = result;
            apply_output_transform(&output_traj, &output_result);
            writer.write_result(output_traj, output_result);
            const int points_in_tr = trajectory.geom.get_num_points();
            total_points += points_in_tr;
            ++total_trajs;
            if (!result.cpath.empty()) {
                points_matched += points_in_tr;
                ++traj_matched;
            }
            ++progress;
        }
    }

    auto end_time = UTIL::get_current_time();
    double duration = UTIL::get_duration(begin_time, end_time);
    double speed = duration > 0 ? static_cast<double>(points_matched) / duration : 0.0;

    oss << "Status: success\n";
    oss << "Time takes " << duration << " seconds\n";
    oss << "Total points " << total_points << " matched " << points_matched << "\n";
    oss << "Trajectories processed " << total_trajs << " matched " << traj_matched << "\n";
    oss << "Map match speed " << speed << " points/s \n";
    oss << "Trajectories skipped " << invalid_records << "\n";

    return oss.str();
}
