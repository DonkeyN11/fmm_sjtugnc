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
#include <cstddef>
#include <cmath>
#include <numeric>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <limits>
#include <memory>
#include <utility>

#include <ogrsf_frmts.h>

#include <boost/property_tree/json_parser.hpp>

using namespace FMM;
using namespace FMM::CORE;
using namespace FMM::NETWORK;
using namespace FMM::PYTHON;
using namespace FMM::MM;

namespace {

// Remove leading/trailing whitespace characters to sanitize tokens read from CSV files.
std::string trim_copy(const std::string &input) {
    const auto begin = input.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = input.find_last_not_of(" \t\r\n");
    return input.substr(begin, end - begin + 1);
}

// Lowercase helper that avoids repeated string creation when
// normalizing column headers originating from user provided files.
std::string to_lower_copy(const std::string &input) {
    std::string lowered = input;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lowered;
}

// Split a delimited line while preserving trailing empty fields to stay aligned with CSV semantics.
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

// Attempt to parse a floating point number while rejecting partially parsed strings.
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

// Attempt to parse a signed integer while ensuring the whole token is consumed.
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

// Maintain a descending list of top-k scores.
void push_top_k(std::vector<double> *scores, double value, size_t k) {
    if (!std::isfinite(value)) {
        return;
    }
    auto &vec = *scores;
    auto it = vec.begin();
    for (; it != vec.end(); ++it) {
        if (value > *it) {
            break;
        }
    }
    vec.insert(it, value);
    if (vec.size() > k) {
        vec.pop_back();
    }
}

// LogSumExp utility for numerical stability in log-space calculations
// Computes log(sum(exp(x_i))) as: max(x) + log(sum(exp(x_i - max(x))))
// This prevents overflow/underflow when exponentiating large or small log values.
double log_sum_exp(const std::vector<double> &log_vals) {
    if (log_vals.empty()) {
        return -std::numeric_limits<double>::infinity();
    }
    // Find maximum value
    double max_val = *std::max_element(log_vals.begin(), log_vals.end());
    if (max_val == -std::numeric_limits<double>::infinity()) {
        return max_val;
    }

    // Compute sum of exp(x_i - max(x))
    double sum = 0.0;
    for (double v : log_vals) {
        sum += std::exp(v - max_val);
    }
    return max_val + std::log(sum);
}

template <typename T>
// Parse a JSON array (encoded inside a CSV field) into a numeric vector of type T.
//
// Format specification:
// Used for timestamps and protection_levels columns in CSV files.
// These columns should contain a JSON 1D array where:
// - Each value corresponds to one trajectory point
// - Values are parsed as type T (typically double for timestamps/protection_levels)
//
// Example format for timestamps:
// [1234567890.0,1234567891.0,1234567892.0,...]
//
// Example format for protection_levels:
// [1.38,1.38,1.39,1.37,...]
//
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

// Parse and validate covariance matrices encoded as flattened JSON arrays.
//
// Format specification:
// The covariances column in CSV should contain a JSON 2D array where:
// - Each row in the JSON array corresponds to one trajectory point
// - Each row contains exactly 6 numeric values: [sde, sdn, sdu, sdne, sdeu, sdun]
//   where:
//   - sde: East standard deviation
//   - sdn: North standard deviation
//   - sdu: Up standard deviation
//   - sdne: North-East covariance
//   - sdeu: East-Up covariance
//   - sdun: Up-North covariance
//
// Example format:
// [[0.68,0.69,0.81,0.033,0.0,0.0],[0.67,0.69,0.81,0.032,0.0,0.0],...]
//
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

// Compute squared Mahalanobis distance given a 2x2 covariance inverse.
double compute_mahalanobis_sq(const Matrix2d &cov_inv, double dx, double dy) {
    return cov_inv.m[0][0] * dx * dx +
           2 * cov_inv.m[0][1] * dx * dy +
           cov_inv.m[1][1] * dy * dy;
}

// Fallback Euclidean distance squared when covariance is not present.
double compute_euclidean_sq(double dx, double dy) {
    return dx * dx + dy * dy;
}

// Helper holder bundling a candidate with its ranking metric so that callers
// can keep additional bookkeeping alongside the Candidate instance.
struct CandidateWithMetric {
    Candidate candidate;
    double metric;
};

// Normalize trustworthiness values of a layer so that they sum to one when positive.
void normalize_layer_trust(TGLayer *layer) {
    if (layer == nullptr || layer->empty()) {
        return;
    }
    double trust_sum = 0.0;
    for (auto &node : *layer) {
        if (std::isfinite(node.trustworthiness) && node.trustworthiness > 0) {
            trust_sum += node.trustworthiness;
        }
    }
    if (trust_sum > 0) {
        for (auto &node : *layer) {
            if (node.trustworthiness > 0) {
                node.trustworthiness /= trust_sum;
            }
        }
        return;
    }
    size_t positive_count = 0;
    for (auto &node : *layer) {
        if (node.ep > 0) {
            positive_count++;
        }
    }
    if (positive_count == 0) {
        return;
    }
    // 没有正的trustworthiness，用emission probability代替
    for (auto &node : *layer) {
        node.trustworthiness = (node.ep > 0) ? node.ep : 0.0;
    }
}

// Project an observation onto every segment of an edge and return the best candidate
// according to Mahalanobis (or Euclidean) distance along with the score used to rank it.
std::optional<CandidateWithMetric> create_edge_candidate(NETWORK::Edge *edge,
                                                         const CORE::Point &obs_point,
                                                         bool valid_covariance,
                                                         const Matrix2d &cov_inv,
                                                         NodeIndex *next_candidate_index) {
    if (edge == nullptr || edge->geom.get_num_points() < 2) {
        return std::nullopt;
    }

    const double obs_x = boost::geometry::get<0>(obs_point);
    const double obs_y = boost::geometry::get<1>(obs_point);

    double best_metric = std::numeric_limits<double>::infinity();
    double best_eucl_sq = std::numeric_limits<double>::infinity();
    double best_offset = 0.0;
    CORE::Point best_point;
    bool found = false;

    const FMM::CORE::LineString &geom = edge->geom;
    const int num_points = geom.get_num_points();
    double accumulated = 0.0;

    // Walk along every segment of this edge geometry to locate the closest projection.
    for (int seg = 0; seg < num_points - 1; ++seg) {
        double sx = geom.get_x(seg);
        double sy = geom.get_y(seg);
        double tx = geom.get_x(seg + 1);
        double ty = geom.get_y(seg + 1);
        double dx = tx - sx;
        double dy = ty - sy;
        double seg_len_sq = dx * dx + dy * dy;
        if (seg_len_sq <= 0) {
            continue;
        }
        double seg_len = std::sqrt(seg_len_sq);
        double obs_minus_start_x = obs_x - sx;
        double obs_minus_start_y = obs_y - sy;
        double t = 0.0;
        if (valid_covariance) {
            // Prefer anisotropic projection when a covariance matrix is available.
            double numerator = dx * (cov_inv.m[0][0] * obs_minus_start_x + cov_inv.m[0][1] * obs_minus_start_y) +
                               dy * (cov_inv.m[1][0] * obs_minus_start_x + cov_inv.m[1][1] * obs_minus_start_y);
            double denominator = dx * (cov_inv.m[0][0] * dx + cov_inv.m[0][1] * dy) +
                                 dy * (cov_inv.m[1][0] * dx + cov_inv.m[1][1] * dy);
            if (std::abs(denominator) < 1e-12) {
                t = (obs_minus_start_x * dx + obs_minus_start_y * dy) / seg_len_sq;
            } else {
                t = numerator / denominator;
            }
        } else {
            // Fall back to Euclidean projection if no covariance was provided.
            t = (obs_minus_start_x * dx + obs_minus_start_y * dy) / seg_len_sq;
        }
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;

        double px = sx + t * dx;
        double py = sy + t * dy;
        double diff_x = obs_x - px;
        double diff_y = obs_y - py;
        double eucl_sq = compute_euclidean_sq(diff_x, diff_y);
        double metric = valid_covariance ? compute_mahalanobis_sq(cov_inv, diff_x, diff_y) : eucl_sq;

        if (metric < best_metric) {
            boost::geometry::set<0>(best_point, px);
            boost::geometry::set<1>(best_point, py);
            best_metric = metric;
            best_eucl_sq = eucl_sq;
            best_offset = accumulated + t * seg_len;
            found = true;
        }
        accumulated += seg_len;
    }

    if (!found) {
        return std::nullopt;
    }

    // Package the best projection and supporting metrics into a Candidate instance.
    Candidate candidate{};
    candidate.index = (*next_candidate_index)++;
    candidate.edge = edge;
    candidate.offset = best_offset;
    candidate.dist = std::sqrt(std::max(best_eucl_sq, 0.0));
    candidate.point = best_point;

    CandidateWithMetric result{candidate, best_metric};
    return result;
}

// Small helper for 2x2 matrix multiplication used during covariance propagation.
Matrix2d multiply_matrices(const Matrix2d &lhs, const Matrix2d &rhs) {
    return Matrix2d(
        lhs.m[0][0] * rhs.m[0][0] + lhs.m[0][1] * rhs.m[1][0],
        lhs.m[0][0] * rhs.m[0][1] + lhs.m[0][1] * rhs.m[1][1],
        lhs.m[1][0] * rhs.m[0][0] + lhs.m[1][1] * rhs.m[1][0],
        lhs.m[1][0] * rhs.m[0][1] + lhs.m[1][1] * rhs.m[1][1]);
}

// Convenience transpose for the light-weight Matrix2d structure.
Matrix2d transpose_matrix(const Matrix2d &mat) {
    return Matrix2d(mat.m[0][0], mat.m[1][0], mat.m[0][1], mat.m[1][1]);
}

// Heuristic that checks if a geometry already lives in projected coordinates.
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

// Apply an in place GDAL transformation to every point in the linestring.
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

// Store intermediate results of projecting a single WGS84 coordinate to the network CRS.
struct TransformInfo {
    Matrix2d jacobian;
    double proj_x;
    double proj_y;
    double scale_lon;
    double scale_lat;
};

// Compute projected coordinates and numerical Jacobian for a lon/lat input.
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

// Conditionally reproject GPS trajectories into the network CRS (when available) and
// update their covariance/protection metadata accordingly.
bool maybe_reproject_trajectories(std::vector<CMMTrajectory> *trajectories,
                                  const NETWORK::Network &network,
                                  int input_epsg) {
    // Check if network has spatial reference
    if (!network.has_spatial_ref()) {
        SPDLOG_WARN("Network CRS information unavailable; skip trajectory reprojection.");
        return false;
    }

    // Get network EPSG code
    int network_epsg = 0;
    OGRSpatialReference network_sr;
    if (network_sr.importFromWkt(network.get_spatial_ref_wkt().c_str()) != OGRERR_NONE) {
        SPDLOG_WARN("Failed to import network CRS from WKT; skip trajectory reprojection.");
        return false;
    }
    network_sr.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);

    const char *auth_name = network_sr.GetAuthorityName(nullptr);
    const char *auth_code = network_sr.GetAuthorityCode(nullptr);
    if (auth_name && auth_code && std::string(auth_name) == "EPSG") {
        network_epsg = std::stoi(auth_code);
    } else {
        SPDLOG_WARN("Network CRS is not EPSG; assuming EPSG:4326");
        network_epsg = 4326;
    }

    // Check if reprojection is needed
    if (input_epsg == network_epsg) {
        SPDLOG_INFO("Input EPSG ({}) matches network EPSG ({}); no reprojection needed.", input_epsg, network_epsg);
        return false;
    }

    // Check if trajectories are already in network CRS
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

    // Create coordinate transformation from input EPSG to network CRS
    OGRSpatialReference source_sr;
    std::ostringstream source_epsg_str;
    source_epsg_str << "EPSG:" << input_epsg;
    if (source_sr.importFromEPSG(input_epsg) != OGRERR_NONE) {
        SPDLOG_WARN("Failed to import source CRS from EPSG:{}; skip trajectory reprojection.", input_epsg);
        return false;
    }
    source_sr.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);

    OGRCoordinateTransformation *transform = OGRCreateCoordinateTransformation(&source_sr, &network_sr);
    if (transform == nullptr) {
        SPDLOG_WARN("Failed to create coordinate transformation (EPSG:{} -> Network CRS); skip trajectory reprojection.", input_epsg);
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
                    cov.sde = std::sqrt(std::max(cov_new.m[0][0], 0.0));
                    cov.sdn = std::sqrt(std::max(cov_new.m[1][1], 0.0));
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
        SPDLOG_INFO("Reprojected {} trajectories from EPSG:{} to network CRS (EPSG:{})",
                     trajectories->size(), input_epsg, network_epsg);
    }
    return transformed_any;
}

} // namespace

// Implementation of CovarianceMapMatchConfig
// Keep configuration construction centralized so both XML and CLI share defaults.
CovarianceMapMatchConfig::CovarianceMapMatchConfig(int k_arg, int min_candidates_arg,
                                                   double protection_level_multiplier_arg,
                                                   double reverse_tolerance_arg,
                                                   bool normalized_arg,
                                                   bool use_mahalanobis_candidates_arg,
                                                   int window_length_arg,
                                                   bool margin_used_trustworthiness_arg,
                                                   bool filtered_arg,
                                                   bool enable_candidate_filter_arg,
                                                   double candidate_filter_threshold_arg,
                                                   bool enable_gap_bridging_arg,
                                                   double max_gap_distance_arg,
                                                //    double min_gps_error_degrees_arg,
                                                   double max_interval_arg,
                                                   double trustworthiness_threshold_arg,
                                                   double map_error_std_arg,
                                                   double background_log_prob_arg)
    : k(k_arg), min_candidates(min_candidates_arg),
      protection_level_multiplier(protection_level_multiplier_arg),
      reverse_tolerance(reverse_tolerance_arg),
      normalized(normalized_arg),
      use_mahalanobis_candidates(use_mahalanobis_candidates_arg),
      window_length(window_length_arg),
      margin_used_trustworthiness(margin_used_trustworthiness_arg),
      filtered(filtered_arg),
      enable_candidate_filter(enable_candidate_filter_arg),
      candidate_filter_threshold(candidate_filter_threshold_arg),
      enable_gap_bridging(enable_gap_bridging_arg),
      max_gap_distance(max_gap_distance_arg),
    //   min_gps_error_degrees(min_gps_error_degrees_arg),
      max_interval(max_interval_arg),
      trustworthiness_threshold(trustworthiness_threshold_arg),
      map_error_std(map_error_std_arg),
      background_log_prob(background_log_prob_arg) {
}

// Dump runtime configuration for debugging or reproducibility.
void CovarianceMapMatchConfig::print() const {
    SPDLOG_INFO("CMMAlgorithmConfig");
    SPDLOG_INFO("k {} min_candidates {} protection_level_multiplier {} reverse_tolerance {}",
                k, min_candidates, protection_level_multiplier, reverse_tolerance);
    SPDLOG_INFO("normalized {} use_mahalanobis {} window_length {} margin_trust {} filtered {}",
                normalized, use_mahalanobis_candidates, window_length, margin_used_trustworthiness, filtered);
    SPDLOG_INFO("enable_filter {} filter_threshold {} gap_bridging {} max_gap_distance {}",
                enable_candidate_filter, candidate_filter_threshold, enable_gap_bridging, max_gap_distance);
    SPDLOG_INFO("min_gps_error_degrees {} max_interval {} trustworthiness_threshold",
                min_gps_error_degrees, max_interval, trustworthiness_threshold);
    SPDLOG_INFO("map_error_std {} background_log_prob", map_error_std, background_log_prob);
}

// Parse configuration fields from XML, falling back to hard-coded defaults when needed.
CovarianceMapMatchConfig CovarianceMapMatchConfig::load_from_xml(
    const boost::property_tree::ptree &xml_data) {
    int k = xml_data.get("config.parameters.k", 8);
    int min_candidates = xml_data.get("config.parameters.min_candidates", 3);
    double protection_level_multiplier = xml_data.get("config.parameters.protection_level_multiplier", 1.0);
    double reverse_tolerance = xml_data.get("config.parameters.reverse_tolerance", 0.0);
    bool normalized = xml_data.get("config.parameters.normalized", true);
    bool use_mahalanobis_candidates = xml_data.get("config.parameters.use_mahalanobis", true);
    int window_length = xml_data.get("config.parameters.window_length", 10);
    bool margin_used_trustworthiness = xml_data.get("config.other.margin_used_trustworthiness", true);
    bool filtered = xml_data.get("config.parameters.filtered", true);

    // New parameters for log-space filtering and gap bridging
    bool enable_candidate_filter = xml_data.get("config.parameters.enable_candidate_filter", true);
    double candidate_filter_threshold = xml_data.get("config.parameters.candidate_filter_threshold", 15.0);
    bool enable_gap_bridging = xml_data.get("config.parameters.enable_gap_bridging", true);
    double max_gap_distance = xml_data.get("config.parameters.max_gap_distance", 2000.0);

    // Minimum GPS error to prevent over-confidence
    double min_gps_error_degrees = xml_data.get("config.parameters.min_gps_error_degrees", 1.0e-4);

    double max_interval = xml_data.get("config.parameters.max_interval", 180.0);
    double trustworthiness_threshold = xml_data.get("config.parameters.trustworthiness_threshold", 0.0);

    // New parameters for additive map noise and background noise normalization
    double map_error_std = xml_data.get("config.parameters.map_error_std", 5.0e-5);
    double background_log_prob = xml_data.get("config.parameters.background_log_prob", -20.0);

    return CovarianceMapMatchConfig{k, min_candidates, protection_level_multiplier, reverse_tolerance,
                                    normalized, use_mahalanobis_candidates, window_length,
                                    margin_used_trustworthiness, filtered,
                                    enable_candidate_filter, candidate_filter_threshold,
                                    enable_gap_bridging, max_gap_distance, min_gps_error_degrees,
                                    max_interval, trustworthiness_threshold,
                                    map_error_std, background_log_prob};
}

// Parse configuration flags from CLI arguments.
CovarianceMapMatchConfig CovarianceMapMatchConfig::load_from_arg(
    const cxxopts::ParseResult &arg_data) {
    int k = arg_data["candidates"].as<int>();
    int min_candidates = arg_data["min_candidates"].as<int>();
    double protection_level_multiplier = arg_data["protection_level_multiplier"].as<double>();
    double reverse_tolerance = arg_data["reverse_tolerance"].as<double>();
    bool normalized = arg_data["normalized"].as<bool>();
    bool use_mahalanobis_candidates = arg_data["use_mahalanobis"].as<bool>();
    int window_length = arg_data["window_length"].as<int>();
    bool margin_used_trustworthiness = arg_data["margin_used_trustworthiness"].as<bool>();
    bool filtered = arg_data["filtered"].as<bool>();

    // Check if new args exist (assuming they are registered) or use defaults
    bool enable_filter = arg_data.count("enable_candidate_filter") ? arg_data["enable_candidate_filter"].as<bool>() : true;
    double filter_thresh = arg_data.count("candidate_filter_threshold") ? arg_data["candidate_filter_threshold"].as<double>() : 15.0;
    bool enable_gap = arg_data.count("enable_gap_bridging") ? arg_data["enable_gap_bridging"].as<bool>() : true;
    double max_gap = arg_data.count("max_gap_distance") ? arg_data["max_gap_distance"].as<double>() : 2000.0;

    // Minimum GPS error to prevent over-confidence
    double min_gps_error = arg_data.count("min_gps_error_degrees") ? arg_data["min_gps_error_degrees"].as<double>() : 1.0e-4;

    double max_interval = arg_data.count("max_interval") ? arg_data["max_interval"].as<double>() : 180.0;
    double trustworthiness_threshold = arg_data.count("trustworthiness_threshold") ? arg_data["trustworthiness_threshold"].as<double>() : 0.0;

    // New parameters for additive map noise and background noise normalization
    double map_error_std = arg_data.count("map_error_std") ? arg_data["map_error_std"].as<double>() : 5.0e-5;
    double background_log_prob = arg_data.count("background_log_prob") ? arg_data["background_log_prob"].as<double>() : -20.0;

    return CovarianceMapMatchConfig{k, min_candidates, protection_level_multiplier, reverse_tolerance,
                                    normalized, use_mahalanobis_candidates, window_length,
                                    margin_used_trustworthiness, filtered,
                                    enable_filter, filter_thresh, enable_gap, max_gap, min_gps_error,
                                    max_interval, trustworthiness_threshold,
                                    map_error_std, background_log_prob};
}

// Register all tunable knobs so the CLI help stays in sync with the structure.
void CovarianceMapMatchConfig::register_arg(cxxopts::Options &options) {
    options.add_options()
        ("k,candidates", "Number of candidates",
         cxxopts::value<int>()->default_value("8"))
        ("min_candidates", "Minimum number of candidates to keep",
         cxxopts::value<int>()->default_value("3"))
        ("protection_level_multiplier", "Multiplier for protection level",
         cxxopts::value<double>()->default_value("2.0"))
        ("reverse_tolerance", "Ratio of reverse movement allowed",
         cxxopts::value<double>()->default_value("0.0"))
        ("normalized", "Normalize emission probabilities",
         cxxopts::value<bool>()->default_value("true"))
        ("use_mahalanobis", "Use Mahalanobis-based candidate search",
         cxxopts::value<bool>()->default_value("true"))
        ("window_length", "Sliding window length for trustworthiness (points)",
         cxxopts::value<int>()->default_value("10"))
        ("margin_used_trustworthiness", "If true use margin (top1-top2) as trustworthiness, else use top1 score",
         cxxopts::value<bool>()->default_value("true"))
        ("filtered", "Filter out points with no candidates or disconnected transitions",
         cxxopts::value<bool>()->default_value("true"))
        ("enable_candidate_filter", "Enable L2 candidate filtering based on relative log-probability",
         cxxopts::value<bool>()->default_value("true"))
        ("candidate_filter_threshold", "Log-probability threshold for filtering (default 15.0)",
         cxxopts::value<double>()->default_value("15.0"))
        ("enable_gap_bridging", "Enable trajectory gap bridging",
         cxxopts::value<bool>()->default_value("true"))
        ("max_gap_distance", "Max distance for gap bridging (meters)",
         cxxopts::value<double>()->default_value("2000.0"))
        ("min_gps_error_degrees", "Minimum GPS error in degrees to prevent over-confidence (default 1e-4 ≈ 11m)",
         cxxopts::value<double>()->default_value("1.0e-4"))
        ("max_interval", "Maximum time interval (seconds) to split segments",
         cxxopts::value<double>()->default_value("180.0"))
        ("trustworthiness_threshold", "Threshold for filtering low-confidence matches",
         cxxopts::value<double>()->default_value("0.0"))
        ("map_error_std", "Map error standard deviation in degrees for additive noise (default 5e-5 ≈ 5m)",
         cxxopts::value<double>()->default_value("5.0e-5"))
        ("background_log_prob", "Background noise log probability for normalization (default -20.0)",
         cxxopts::value<double>()->default_value("-20.0"));
}

// Append a short textual description for the Python binding documentation.
void CovarianceMapMatchConfig::register_help(std::ostringstream &oss) {
    oss << "-k/--candidates (optional) <int>: Number of candidates (8)\n";
    oss << "--min_candidates (optional) <int>: Minimum number of candidates to keep (3)\n";
    oss << "--protection_level_multiplier (optional) <double>: Multiplier for protection level (1.0)\n";
    oss << "--reverse_tolerance (optional) <double>: proportion of reverse movement allowed on an edge\n";
    oss << "--normalized (optional) <bool>: whether to normalize emission probabilities (true)\n";
    oss << "--use_mahalanobis (optional) <bool>: whether to use Mahalanobis-based candidate search (true)\n";
    oss << "--window_length (optional) <int>: sliding window length for trustworthiness (10)\n";
    oss << "--margin_used_trustworthiness (optional) <bool>: if true use margin (top1-top2), else use top1 score (true)\n";
    oss << "--filtered (optional) <bool>: whether to filter out points with no candidates or disconnected transitions (true)\n";
    oss << "--enable_candidate_filter (optional) <bool>: Enable L2 candidate filtering (true)\n";
    oss << "--candidate_filter_threshold (optional) <double>: Log-probability threshold for filtering (15.0)\n";
    oss << "--enable_gap_bridging (optional) <bool>: Enable trajectory gap bridging (true)\n";
    oss << "--max_gap_distance (optional) <double>: Max distance for gap bridging in meters (2000.0)\n";
    oss << "--min_gps_error_degrees (optional) <double>: Minimum GPS error in degrees to prevent over-confidence (1e-4 ≈ 11m)\n";
    oss << "--max_interval (optional) <double>: Maximum time interval (seconds) to split segments (180.0)\n";
    oss << "--trustworthiness_threshold (optional) <double>: Threshold for filtering low-confidence matches (0.0)\n";
    oss << "--map_error_std (optional) <double>: Map error standard deviation in degrees for additive noise (5e-5 ≈ 5m)\n";
    oss << "--background_log_prob (optional) <double>: Background noise log probability for normalization (-20.0)\n";
}

// Quick sanity checks to guard against invalid user supplied parameters.
bool CovarianceMapMatchConfig::validate() const {
    if (k <= 0 || min_candidates <= 0 || min_candidates > k ||
        protection_level_multiplier <= 0 || reverse_tolerance < 0 ||
        window_length <= 0 || candidate_filter_threshold < 0 || max_gap_distance < 0 ||
        map_error_std < 0) {
        SPDLOG_CRITICAL("Invalid CMM parameter k {} min_candidates {} "
                       "protection_level_multiplier {} reverse_tolerance {} window_length {} "
                       "filter_threshold {} max_gap_distance {} map_error_std {}",
                       k, min_candidates, protection_level_multiplier, reverse_tolerance,
                       window_length, candidate_filter_threshold, max_gap_distance, map_error_std);
        return false;
    }
    if (max_interval < 0) return false;
    return true;
}

// Implementation of CovarianceMapMatch
// Evaluate log emission probabilities by respecting each observation's covariance model.
// Returns log(P) to prevent numerical underflow.
double CovarianceMapMatch::calculate_emission_log_prob(
    const CORE::Point &point_observed,
    const CORE::Point &point_candidate,
    const CovarianceMatrix &covariance,
    const CovarianceMapMatchConfig &config) const {

    double obs_x = boost::geometry::get<0>(point_observed);
    double obs_y = boost::geometry::get<1>(point_observed);
    double cand_x = boost::geometry::get<0>(point_candidate);
    double cand_y = boost::geometry::get<1>(point_candidate);

    double dx = obs_x - cand_x;
    double dy = obs_y - cand_y;

    Matrix2d cov = covariance.to_2d_matrix();

    // Apply minimum GPS error to prevent over-confidence
    // This ensures covariance matrix is not too small, which would cause
    // extremely low emission probabilities for reasonable map-matching deviations
    double min_var = config.min_gps_error_degrees * config.min_gps_error_degrees;
    if (cov.m[0][0] < min_var) cov.m[0][0] = min_var;
    if (cov.m[1][1] < min_var) cov.m[1][1] = min_var;

    Matrix2d cov_inv = cov.inverse();
    double det = cov.determinant();

    // Protection against singular matrices or extremely confident GPS
    if (det <= 1e-50) {
        return -std::numeric_limits<double>::infinity();
    }

    double mahalanobis_dist_sq = cov_inv.m[0][0] * dx * dx +
                                 2 * cov_inv.m[0][1] * dx * dy +
                                 cov_inv.m[1][1] * dy * dy;

    // Log Gaussian: -0.5 * (log(2*pi) + log(det) + dist^2)
    static const double log_2pi = std::log(2.0 * M_PI);
    return -0.5 * (log_2pi + std::log(det) + mahalanobis_dist_sq);
}

// Enumerate candidate projections per point by respecting both covariance ellipses
// and the provided protection levels that limit how far points can deviate.
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
    NETWORK::NodeIndex next_candidate_index = network_.get_node_count();

    for (int i = 0; i < num_points; ++i) {
        CORE::Point point = geom.get_point(i);
        // Create a copy to allow modification (scaling)
        CovarianceMatrix cov = covariances[i];
        
        // FIX: Enforce a minimum standard deviation to prevent probability underflow
        // while preserving anisotropy by scaling all components proportionally
        // The input data has sigma ~ 6e-6 (0.6m) which is too small/confident.
        // We enforce min sigma ~ 5e-5 (approx 5 meters) while maintaining the original ratio
        constexpr double MIN_SIGMA = 5.0e-5;
        double scale_factor = 1.0;

        // Find the smaller of sde and sdn
        double min_sigma = std::min(cov.sde, cov.sdn);

        if (min_sigma < MIN_SIGMA && min_sigma > 0) {
            // Scale factor needed to bring min_sigma to MIN_SIGMA
            // This preserves the original anisotropy ratio between sde and sdn
            scale_factor = MIN_SIGMA / min_sigma;
        } else if (min_sigma <= 0) {
            // Invalid data, set to minimum isotropic covariance
            cov.sde = MIN_SIGMA;
            cov.sdn = MIN_SIGMA;
            cov.sdne = 0.0;
        }

        if (scale_factor > 1.0) {
            // Scale all covariance components proportionally to preserve anisotropy
            // Standard deviations scale linearly
            cov.sde *= scale_factor;
            cov.sdn *= scale_factor;
            cov.sdu *= scale_factor;
            // Covariances scale with square of scale factor (Var(aX) = a²Var(X))
            cov.sdne *= (scale_factor * scale_factor);
            cov.sdeu *= (scale_factor * scale_factor);
            cov.sdun *= (scale_factor * scale_factor);
        }

        double protection_level = protection_levels[i];

        // double uncertainty = cov.get_2d_uncertainty();
        // double search_radius = protection_level * config.protection_level_multiplier + uncertainty;
        double search_radius = protection_level * config.protection_level_multiplier;

        SPDLOG_TRACE("Point {}: protection_level={}, search_radius={}",
                     i, protection_level, search_radius);

        Matrix2d cov_mat = cov.to_2d_matrix();
        double det = cov_mat.determinant();
        bool valid_covariance = det > 0;
        Matrix2d cov_inv;
        if (valid_covariance) {
            cov_inv = cov_mat.inverse();
        }

        double obs_x = boost::geometry::get<0>(point);
        double obs_y = boost::geometry::get<1>(point);

        Point_Candidates selected_candidates;
        std::vector<double> raw_probabilities;

        // Optionally refine the initial network search using Mahalanobis-aware projection.
        if (config.use_mahalanobis_candidates) {
            bool radius_expanded = false;

            while (true) {
                CORE::LineString single_point_geom;
                single_point_geom.add_point(point);
                Traj_Candidates traj_candidates = network_.search_tr_cs_knn(single_point_geom, config.k, search_radius);
                Point_Candidates base_candidates = traj_candidates.empty() ? Point_Candidates() : traj_candidates[0];

                // Debug log for first 122 points of trajectory 11
                // if (traj_id == "11" && i < 122) {
                //     std::cerr << "[DEBUG] Traj " << traj_id << " Point " << i << ": kNN returned "
                //               << base_candidates.size() << " candidates, search_radius="
                //               << search_radius << " deg (" << (search_radius * 111000) << "m)\n" << std::flush;
                // }

                std::vector<NETWORK::Edge *> edges_to_consider;
                edges_to_consider.reserve(base_candidates.size());
                std::unordered_set<NETWORK::EdgeIndex> seen_edges;
                for (const Candidate &seed : base_candidates) {
                    if (seed.edge == nullptr) {
                        continue;
                    }
                    NETWORK::EdgeIndex edge_idx = seed.edge->index;
                    if (seen_edges.insert(edge_idx).second) {
                        edges_to_consider.push_back(seed.edge);
                    }
                }

                std::vector<CandidateWithMetric> candidate_pool;
                candidate_pool.reserve(edges_to_consider.size() * 3);
                double search_radius_sq = search_radius * search_radius;

                for (NETWORK::Edge *edge : edges_to_consider) {
                    if (auto edge_candidate = create_edge_candidate(edge, point, valid_covariance, cov_inv, &next_candidate_index)) {
                        candidate_pool.push_back(*edge_candidate);
                    }

                    auto process_node = [&](NETWORK::NodeIndex node_idx, bool use_target) {
                        const CORE::Point &node_point = network_.get_vertex_point(node_idx);
                        double dx = obs_x - boost::geometry::get<0>(node_point);
                        double dy = obs_y - boost::geometry::get<1>(node_point);
                        double eucl_sq = compute_euclidean_sq(dx, dy);
                        if (search_radius > 0 && eucl_sq > search_radius_sq) {
                            return;
                        }
                        double metric = valid_covariance ? compute_mahalanobis_sq(cov_inv, dx, dy) : eucl_sq;
                        Candidate candidate{};
                        candidate.index = next_candidate_index++;
                        candidate.edge = edge;
                        candidate.offset = use_target ? edge->length : 0.0;
                        candidate.dist = std::sqrt(eucl_sq);
                        candidate.point = node_point;
                        candidate_pool.push_back(CandidateWithMetric{candidate, metric});
                    };

                    process_node(edge->source, false);
                    process_node(edge->target, true);
                }

                const size_t desired_candidates = std::max(config.k, config.min_candidates);
                if (!candidate_pool.empty()) {
                    std::sort(candidate_pool.begin(), candidate_pool.end(),
                              [](const CandidateWithMetric &lhs, const CandidateWithMetric &rhs) {
                                  return lhs.metric < rhs.metric;
                              });
                    if (candidate_pool.size() > desired_candidates) {
                        candidate_pool.resize(desired_candidates);
                    }
                }

                selected_candidates.clear();
                raw_probabilities.clear();
                selected_candidates.reserve(candidate_pool.size());
                raw_probabilities.reserve(candidate_pool.size());

                for (const auto &entry : candidate_pool) {
                    selected_candidates.push_back(entry.candidate);
                    double log_probability = -std::numeric_limits<double>::infinity();
                    if (valid_covariance) {
                        double dx = obs_x - boost::geometry::get<0>(entry.candidate.point);
                        double dy = obs_y - boost::geometry::get<1>(entry.candidate.point);

                        // Apply additive map noise: add map error variance to GPS variance
                        // This smooths the effect of small GPS errors and preserves anisotropy
                        double map_var = config.map_error_std * config.map_error_std;
                        double var_e_total = cov.sde * cov.sde + map_var;
                        double var_n_total = cov.sdn * cov.sdn + map_var;

                        // Build effective covariance matrix with additive map noise
                        Matrix2d cov_eff;
                        cov_eff.m[0][0] = var_e_total;
                        cov_eff.m[1][1] = var_n_total;
                        // Map error is assumed isotropic and independent, so covariance term remains unchanged
                        cov_eff.m[0][1] = cov_eff.m[1][0] = cov.sdne;

                        double det_eff = cov_eff.determinant();
                        if (det_eff > 1e-30) {
                            Matrix2d cov_inv_eff = cov_eff.inverse();
                            double mahal_sq = cov_inv_eff.m[0][0] * dx * dx +
                                             2 * cov_inv_eff.m[0][1] * dx * dy +
                                             cov_inv_eff.m[1][1] * dy * dy;
                            // Log Gaussian: -0.5 * (log(2*pi) + log(det) + mahal_sq)
                            log_probability = -0.5 * (std::log(2.0 * M_PI) + std::log(det_eff) + mahal_sq);
                        }
                    }
                    raw_probabilities.push_back(log_probability);
                }

                // Note: Removed debug logging for trajectory 11 as traj_id parameter was removed

                if (selected_candidates.size() >= static_cast<size_t>(config.min_candidates) ||
                    edges_to_consider.empty() || radius_expanded) {
                    break;
                }

                search_radius *= 2.0;
                radius_expanded = true;
            }
        } else {
            // Basic candidate search that directly relies on network_kNN results.
            // Note: Removed debug logging for trajectory 11 as traj_id parameter was removed

            CORE::LineString single_point_geom;
            single_point_geom.add_point(point);
            Traj_Candidates traj_candidates = network_.search_tr_cs_knn(single_point_geom, config.k, search_radius);
            Point_Candidates point_candidates = traj_candidates.empty() ? Point_Candidates() : traj_candidates[0];

            selected_candidates.reserve(point_candidates.size());
            raw_probabilities.reserve(point_candidates.size());

            for (const Candidate &cand : point_candidates) {
                double cand_x = boost::geometry::get<0>(cand.point);
                double cand_y = boost::geometry::get<1>(cand.point);
                double dx = obs_x - cand_x;
                double dy = obs_y - cand_y;

                double log_probability = -std::numeric_limits<double>::infinity();
                if (valid_covariance) {
                    // Apply additive map noise: add map error variance to GPS variance
                    // This smooths the effect of small GPS errors and preserves anisotropy
                    double map_var = config.map_error_std * config.map_error_std;
                    double var_e_total = cov.sde * cov.sde + map_var;
                    double var_n_total = cov.sdn * cov.sdn + map_var;

                    // Build effective covariance matrix with additive map noise
                    Matrix2d cov_eff;
                    cov_eff.m[0][0] = var_e_total;
                    cov_eff.m[1][1] = var_n_total;
                    // Map error is assumed isotropic and independent, so covariance term remains unchanged
                    cov_eff.m[0][1] = cov_eff.m[1][0] = cov.sdne;

                    double det_eff = cov_eff.determinant();
                    if (det_eff > 1e-30) {
                        Matrix2d cov_inv_eff = cov_eff.inverse();
                        double mahalanobis_dist_sq = cov_inv_eff.m[0][0] * dx * dx +
                                                      2 * cov_inv_eff.m[0][1] * dx * dy +
                                                      cov_inv_eff.m[1][1] * dy * dy;
                        log_probability = -0.5 * (std::log(2.0 * M_PI) + std::log(det_eff) + mahalanobis_dist_sq);
                    }
                }

                selected_candidates.push_back(cand);
                raw_probabilities.push_back(log_probability);
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
                        // Give small log probability instead of -inf
                        raw_probabilities.push_back(-25.0); // exp(-25) is approx 1e-11
                    }
                }
            }
        }

        std::vector<double> log_emission_probs;
        if (config.normalized) {
            // Top-K normalization in log-space with background noise
            std::vector<double> top_k_scores = raw_probabilities;
            size_t k = std::min(top_k_scores.size(), static_cast<size_t>(config.k));
            std::partial_sort(top_k_scores.begin(), top_k_scores.begin() + k, top_k_scores.end(), std::greater<double>());
            top_k_scores.resize(k);

            // Calculate log sum of top-k candidates
            double log_sum_candidates = log_sum_exp(top_k_scores);

            // Include background noise in the normalization (represents "null hypothesis")
            // This prevents "矮子里拔将军" (picking the best of poor candidates)
            // LogSumExp: log(exp(log_sum_candidates) + exp(background_log_prob))
            std::vector<double> norm_components = {log_sum_candidates, config.background_log_prob};
            double log_norm = log_sum_exp(norm_components);

            log_emission_probs.reserve(raw_probabilities.size());

            if (log_norm > -std::numeric_limits<double>::infinity()) {
                for (double log_prob : raw_probabilities) {
                    log_emission_probs.push_back(log_prob - log_norm);
                }
            } else if (!raw_probabilities.empty()) {
                double uniform_log_prob = -std::log(static_cast<double>(raw_probabilities.size()));
                log_emission_probs.assign(raw_probabilities.size(), uniform_log_prob);
                if (!valid_covariance) {
                    SPDLOG_WARN("Point {}: covariance determinant non-positive, using uniform emission", i);
                } else {
                    SPDLOG_WARN("Point {}: no valid candidates within PL, using uniform emission", i);
                }
            }
        } else {
            log_emission_probs = raw_probabilities;
        }

        SPDLOG_TRACE("Point {}: {} candidates kept", i, selected_candidates.size());
        result.candidates.push_back(std::move(selected_candidates));

        // Note: Removed debug logging for trajectory 11 as traj_id parameter was removed
        result.emission_probabilities.push_back(std::move(log_emission_probs));
    }

    SPDLOG_DEBUG("Candidate search completed");
    return result;
}

// Slice a CMMTrajectory into a sub-segment [start_idx, end_idx)
CMMTrajectory CovarianceMapMatch::slice_trajectory(const CMMTrajectory &traj, int start_idx, int end_idx) const {
    if (start_idx < 0 || end_idx > traj.geom.get_num_points() || start_idx >= end_idx) {
        return CMMTrajectory();
    }

    CMMTrajectory sub_traj;
    sub_traj.id = traj.id;  // Keep same ID, caller can modify if needed

    // Copy geometry
    for (int i = start_idx; i < end_idx; ++i) {
        sub_traj.geom.add_point(traj.geom.get_point(i));
    }

    // Copy timestamps if present
    if (!traj.timestamps.empty()) {
        sub_traj.timestamps.assign(traj.timestamps.begin() + start_idx,
                                   traj.timestamps.begin() + end_idx);
    }

    // Copy covariances
    if (!traj.covariances.empty()) {
        sub_traj.covariances.assign(traj.covariances.begin() + start_idx,
                                     traj.covariances.begin() + end_idx);
    }

    // Copy protection levels
    if (!traj.protection_levels.empty()) {
        sub_traj.protection_levels.assign(traj.protection_levels.begin() + start_idx,
                                          traj.protection_levels.begin() + end_idx);
    }

    return sub_traj;
}

// Execute the full map-matching pipeline for a single trajectory using covariance-aware search.
std::vector<MatchResult> CovarianceMapMatch::match_traj(const CMMTrajectory &traj,
                                                       const CovarianceMapMatchConfig &config,
                                                       CMMTrajectory *filtered_traj) {
    SPDLOG_DEBUG("Count of points in trajectory {}", traj.geom.get_num_points());

    // Validate trajectory
    if (!traj.is_valid()) {
        SPDLOG_ERROR("Invalid trajectory: covariance and protection level data mismatch");
        return {};
    }

    SPDLOG_DEBUG("Search candidates with protection level");
    CandidateSearchResult candidate_result = search_candidates_with_protection_level(
        traj.geom, traj.covariances, traj.protection_levels, config);

    const Traj_Candidates &tc_raw = candidate_result.candidates;
    const std::vector<std::vector<double>> &log_eps_raw = candidate_result.emission_probabilities;

    std::vector<int> valid_indices;
    valid_indices.reserve(tc_raw.size());
    for (int i = 0; i < tc_raw.size(); ++i) {
        if (!tc_raw[i].empty()) {
            valid_indices.push_back(i);
        }
    }

    if (valid_indices.size() < 3) {
        return {};
    }

    std::vector<MatchResult> final_results;

    // Split valid_indices into segments based on max_interval
    std::vector<std::vector<int>> segments;
    std::vector<int> current_segment;

    for (size_t i = 0; i < valid_indices.size(); ++i) {
        int curr_idx = valid_indices[i];
        if (current_segment.empty()) {
            current_segment.push_back(curr_idx);
        } else {
            int prev_idx = current_segment.back();
            double time_diff = 0.0;
            if (!traj.timestamps.empty() && traj.timestamps.size() > static_cast<size_t>(curr_idx) &&
                traj.timestamps.size() > static_cast<size_t>(prev_idx)) {
                time_diff = std::abs(traj.timestamps[curr_idx] - traj.timestamps[prev_idx]);
            }

            if (time_diff > config.max_interval) {
                if (current_segment.size() >= 3) {
                    segments.push_back(current_segment);
                }
                current_segment.clear();
                current_segment.push_back(curr_idx);
            } else {
                current_segment.push_back(curr_idx);
            }
        }
    }
    if (current_segment.size() >= 3) {
        segments.push_back(current_segment);
    }

    for (const auto& segment_indices : segments) {
        int start_real_idx = segment_indices.front();
        int end_real_idx = segment_indices.back();

        // Use a dummy empty trajectory to initialize TransitionGraph with just the first layer
        Traj_Candidates start_tc = {tc_raw[start_real_idx]};
        std::vector<std::vector<double>> start_log_eps = {log_eps_raw[start_real_idx]};
        TransitionGraph tg(start_tc, start_log_eps, true);
        initialize_first_layer(&tg.get_layers()[0], config);

        bool segment_failed = false;
        // Process rest of the segment
        for (size_t i = 1; i < segment_indices.size(); ++i) {
            int prev_real = segment_indices[i-1];
            int next_real = segment_indices[i];

            double dist = boost::geometry::distance(traj.geom.get_point(prev_real),
                                                   traj.geom.get_point(next_real));

            // Build next layer
            TGLayer next_layer;
            for (size_t k=0; k<tc_raw[next_real].size(); ++k) {
                next_layer.push_back(TGNode{&tc_raw[next_real][k], nullptr, log_eps_raw[next_real][k], 0,
                                           -std::numeric_limits<double>::infinity(), 0, 0});
            }

            bool connected = false;
            update_layer_cmm(&tg.get_layers().back(), &next_layer, dist, &connected, config);

            if (!connected) {
                segment_failed = true;
                break; // Disconnected, abort this segment
            }
            tg.get_layers().push_back(std::move(next_layer));
        }

        if (segment_failed) continue;

        // Backtrack and finalize segment result
        TGOpath tg_opath = tg.backtrack();
        if (!tg_opath.empty()) {
            CMMTrajectory segment_traj = slice_trajectory(traj, start_real_idx, end_real_idx + 1);

            // Sub-candidates and sub-emissions for trustworthiness
            Traj_Candidates sub_tc;
            std::vector<std::vector<double>> sub_log_eps;
            for (int idx : segment_indices) {
                sub_tc.push_back(tc_raw[idx]);
                sub_log_eps.push_back(log_eps_raw[idx]);
            }

            auto trustworthiness_results = compute_window_trustworthiness(
                sub_tc, sub_log_eps, segment_traj, config);

            MatchedCandidatePath matched_candidate_path;
            matched_candidate_path.reserve(tg_opath.size());

            // Filter vectors based on trustworthiness if filtering enabled
            std::vector<MatchedCandidate> filtered_path;
            std::vector<int> filtered_indices;
            std::vector<double> filtered_sp_dist;
            std::vector<double> filtered_eu_dist;
            std::vector<double> filtered_trust;
            std::vector<std::vector<CandidateEmission>> filtered_details;

            std::vector<double> sp_distances;
            std::vector<double> eu_distances;

            // Prepare candidate details
            std::vector<std::vector<CandidateEmission>> all_details;
            all_details.resize(sub_tc.size());
            for (size_t i=0; i<sub_tc.size(); ++i) {
                for (size_t j=0; j<sub_tc[i].size(); ++j) {
                    all_details[i].push_back({
                        boost::geometry::get<0>(sub_tc[i][j].point),
                        boost::geometry::get<1>(sub_tc[i][j].point),
                        std::exp(sub_log_eps[i][j])
                    });
                }
            }

            for (size_t i = 0; i < tg_opath.size(); ++i) {
                const TGNode *node = tg_opath[i];
                double sp = (i == 0) ? 0.0 : node->sp_dist;
                double eu = (i == 0) ? 0.0 : boost::geometry::distance(segment_traj.geom.get_point(i-1), segment_traj.geom.get_point(i));

                double trust = node->trustworthiness;
                if (config.margin_used_trustworthiness && i < trustworthiness_results.first.size()) {
                    trust = trustworthiness_results.first[i];
                }

                MatchedCandidate mc{*(node->c), std::exp(node->ep), node->tp, node->cumu_prob, node->sp_dist, trust};
                matched_candidate_path.push_back(mc);
                sp_distances.push_back(sp);
                eu_distances.push_back(eu);

                // Apply filtering logic，linear probability thresholding based on trustworthiness，default threshold is 0.0 which means only filter out points with non-positive trustworthiness
                if (!config.filtered || trust >= config.trustworthiness_threshold) {
                    filtered_path.push_back(mc);
                    filtered_indices.push_back(segment_indices[i]);
                    filtered_sp_dist.push_back(sp);
                    filtered_eu_dist.push_back(eu);
                    filtered_trust.push_back(trust);
                    filtered_details.push_back(all_details[i]);
                }
            }

            // If after filtering we have too few points, skip this result
            if (filtered_path.empty()) continue;

            O_Path opath;
            for (const auto& mc : filtered_path) opath.push_back(mc.c.edge->id);

            std::vector<int> indices;
            // Note: construct_complete_path uses the full tg_opath (optimal path), not the filtered one,
            // because cpath defines the route geometry. We use full path for geometry continuity.
            C_Path cpath = ubodt_->construct_complete_path(traj.id, tg_opath, network_.get_edges(), &indices, config.reverse_tolerance);
            LineString mgeom = network_.complete_path_to_geometry(segment_traj.geom, cpath);

            MatchResult res{traj.id, filtered_path, opath, cpath, indices, mgeom};
            res.sp_distances = std::move(filtered_sp_dist);
            res.eu_distances = std::move(filtered_eu_dist);

            // Filter nbest_trustworthiness
            std::vector<std::vector<double>> filtered_nbest;
            for (size_t i = 0; i < tg_opath.size(); ++i) {
                const TGNode *node = tg_opath[i];
                double trust = node->trustworthiness;
                if (config.margin_used_trustworthiness && i < trustworthiness_results.first.size()) {
                    trust = trustworthiness_results.first[i];
                }
                if (!config.filtered || trust >= config.trustworthiness_threshold) {
                    if (i < trustworthiness_results.second.size()) {
                        filtered_nbest.push_back(trustworthiness_results.second[i]);
                    } else {
                        filtered_nbest.push_back({});
                    }
                }
            }
            res.nbest_trustworthiness = std::move(filtered_nbest);
            res.original_indices = std::move(filtered_indices);
            res.candidate_details = std::move(filtered_details);

            final_results.push_back(std::move(res));
        }
    }

    if (filtered_traj != nullptr && !final_results.empty()) {
        // For simplicity, if split, we just return the first segment as filtered_traj
        *filtered_traj = slice_trajectory(traj, final_results[0].original_indices.front(),
                                         final_results[0].original_indices.back() + 1);
    }

    return final_results;
}

std::vector<MatchResult> CovarianceMapMatch::match_traj(const CMMTrajectory &traj,
                                                       const CovarianceMapMatchConfig &config) {
    return match_traj(traj, config, nullptr);
}

// Compute the shortest-path distance between two candidates and allow limited reverse travel.
double CovarianceMapMatch::get_sp_dist(const Candidate *ca, const Candidate *cb,
                                      double reverse_tolerance) {
    // Check for null candidates
    if (ca == nullptr || cb == nullptr || ca->edge == nullptr || cb->edge == nullptr) {
        return -1;
    }

    // Handle transitions along the same edge directly, consistent with FMM logic.
    if (ca->edge->id == cb->edge->id) {
        if (ca->offset <= cb->offset) {
            return cb->offset - ca->offset;
        }
        double reverse_limit = ca->edge->length * reverse_tolerance;
        if (reverse_tolerance > 0 && (ca->offset - cb->offset) < reverse_limit) {
            return 0.0;
        }
        // Same edge but offset decreased beyond reverse tolerance
        static int debug_count = 0;
        if (debug_count < 5) {
            SPDLOG_WARN("Same edge {} but reverse offset too large: offset_a={}, offset_b={}, reverse_limit={}",
                       ca->edge->id, ca->offset, cb->offset, reverse_limit);
            debug_count++;
        }
    }

    // Otherwise rely on UBODT lookup for forward path between successive edges.
    NodeIndex s = ca->edge->target;
    NodeIndex e = cb->edge->source;
    auto *r = ubodt_->look_up(s, e);
    double sp_dist = r ? r->cost : -1;

    static int ubodt_miss_count = 0;
    if (sp_dist < 0 && ubodt_miss_count < 20) {
        SPDLOG_WARN("UBODT lookup failed: source edge {}->target={}, dest edge {}->source={}, s={}, e={}",
                   ca->edge->id, ca->edge->target, cb->edge->id, cb->edge->source, s, e);
        ubodt_miss_count++;
    }
    if (sp_dist < 0) {
                // // No path exists, try reverse direction
        // // When forward lookup fails, try to see if reverse travel is short enough to allow.
        // s = ca->edge->source;
        // e = cb->edge->target;
        // r = ubodt_->look_up(s, e);
        // sp_dist = r ? r->cost : -1;
        // if (sp_dist >= 0) {
        //     // Path exists in reverse direction
        //     double total_length = ca->edge->length + cb->edge->length;
        //     double reverse_dist = total_length - ca->offset - cb->offset;
        //     if (reverse_dist <= sp_dist * (1 + reverse_tolerance)) {
        //         return reverse_dist;
        //     }
        // }    
        return -1;
    } else {
        // Path exists in forward direction
        double dist = ca->edge->length - ca->offset + cb->offset;
        return dist + sp_dist;
    }
}

// Initialize the first layer with Top-K normalization in log-space
void CovarianceMapMatch::initialize_first_layer(TGLayer *layer, const CovarianceMapMatchConfig &config) {
    if (!layer || layer->empty()) {
        return;
    }

    // emission probabilities are already in log-space (from search_candidates_with_protection_level)
    std::vector<double> log_eps;
    log_eps.reserve(layer->size());
    for (const auto &node : *layer) {
        log_eps.push_back(node.ep); // ep here is already log-prob
    }

    // Apply Top-K normalization
    size_t k = std::min(log_eps.size(), static_cast<size_t>(config.k));
    std::vector<double> top_k_eps = log_eps;
    std::partial_sort(top_k_eps.begin(), top_k_eps.begin() + k, top_k_eps.end(), std::greater<double>());
    top_k_eps.resize(k);

    // Calculate log normalization factor
    double log_norm_factor = log_sum_exp(top_k_eps);

    // Initialize cumu_prob as normalized emission probability in log-space
    for (size_t i = 0; i < layer->size(); ++i) {
        auto &node = (*layer)[i];
        if (log_eps[i] > -std::numeric_limits<double>::infinity()) {
            node.cumu_prob = log_eps[i] - log_norm_factor;  // Log-space: subtract normalization factor
            node.trustworthiness = std::exp(node.ep);  // Linear ep for trustworthiness
        } else {
            node.cumu_prob = -std::numeric_limits<double>::infinity();
            node.trustworthiness = 0;
        }
        node.tp = 1.0;  // No transition probability for first layer (store as linear 1.0)
        node.prev = nullptr;
    }
}

// ... update_tg_cmm remains same ...

// Update all transitions between two consecutive layers based on path feasibility and
// Mahalanobis-aware emission probabilities (LOG-SPACE with Two-Level Filtering)
void CovarianceMapMatch::update_layer_cmm(TGLayer *la_ptr, TGLayer *lb_ptr,
                                          double eu_dist,
                                          bool *connected,
                                          const CovarianceMapMatchConfig &config) {
    *connected = false;
    const size_t next_candidate_count = lb_ptr->size();

    // Vector to store unnormalized log posterior scores for each candidate in next layer
    std::vector<double> raw_scores(next_candidate_count, -std::numeric_limits<double>::infinity());

    // 1. Viterbi Recursion: Calculate max log score for each candidate in next layer
    for (size_t b = 0; b < next_candidate_count; ++b) {
        TGNode &node_b = (*lb_ptr)[b];

        // Check for null candidate pointer
        if (node_b.c == nullptr) {
            raw_scores[b] = -std::numeric_limits<double>::infinity();
            continue;
        }

        // ep is already log-space
        double log_ep = node_b.ep;

        double max_prev_score = -std::numeric_limits<double>::infinity();
        TGNode *best_prev = nullptr;
        double best_log_tp = -std::numeric_limits<double>::infinity();
        double best_sp_dist = 0.0;

        for (auto &node_a : *la_ptr) {
            if (node_a.cumu_prob == -std::numeric_limits<double>::infinity()) {
                continue;  // Skip invalid previous states
            }

            // Check for null candidate pointer
            if (node_a.c == nullptr) {
                continue;
            }

            // Calculate shortest path distance
            double sp_dist = get_sp_dist(node_a.c, node_b.c, config.reverse_tolerance);
            double log_tp = -std::numeric_limits<double>::infinity();

            if (sp_dist >= 0) {
                // Calculate transition probability in log space
                double linear_tp = TransitionGraph::calc_tp(sp_dist, eu_dist);
                if (linear_tp > 0) {
                    log_tp = std::log(linear_tp);
                }
            }

            // Viterbi: max(prev.cumu_prob + log(tp)) + log(ep)
            double score = node_a.cumu_prob + log_tp;

            if (score > max_prev_score) {
                max_prev_score = score;
                best_prev = &node_a;
                best_log_tp = log_tp;
                best_sp_dist = sp_dist;
            }
        }

        if (best_prev != nullptr) {
            node_b.prev = best_prev;
            node_b.tp = std::exp(best_log_tp);  // Store linear tp, not log
            node_b.sp_dist = best_sp_dist;
            raw_scores[b] = max_prev_score + log_ep;  // Unnormalized posterior
        }
    }

    // 2. Two-Level Filtering
    if (raw_scores.empty()) {
        return;
    }
    double max_score_in_layer = *std::max_element(raw_scores.begin(), raw_scores.end());

    // L1 Filter: Check if layer is connected (has at least one valid path)
    if (max_score_in_layer == -std::numeric_limits<double>::infinity()) {
        return;  // Layer disconnected
    }

    std::vector<int> kept_indices;
    std::vector<double> scores_to_normalize;

    // L2 Filter: Filter candidates with relatively low probability
    for (size_t b = 0; b < next_candidate_count; ++b) {
        double score = raw_scores[b];

        // Skip null candidates
        if ((*lb_ptr)[b].c == nullptr) {
            (*lb_ptr)[b].cumu_prob = -std::numeric_limits<double>::infinity();
            continue;
        }

        // L2 Candidate filtering: drop if score is too far from max
        if (config.enable_candidate_filter &&
            (score < max_score_in_layer - config.candidate_filter_threshold)) {
            (*lb_ptr)[b].cumu_prob = -std::numeric_limits<double>::infinity();
            continue;
        }

        if (score > -std::numeric_limits<double>::infinity()) {
            kept_indices.push_back(static_cast<int>(b));
            scores_to_normalize.push_back(score);
        }
    }

    // Check if all candidates were filtered out
    if (kept_indices.empty()) {
        return;  // All filtered, layer disconnected
    }

    *connected = true;

    // 3. Top-K Normalization in Log-Space
    std::vector<double> top_k_scores = scores_to_normalize;
    size_t k = std::min(top_k_scores.size(), static_cast<size_t>(config.k));
    std::partial_sort(top_k_scores.begin(), top_k_scores.begin() + k, top_k_scores.end(), std::greater<double>());
    top_k_scores.resize(k);

    // Calculate log normalization factor from Top-K scores
    double log_norm = log_sum_exp(top_k_scores);

    // Normalize cumu_prob (log-space: subtract normalization factor)
    for (int idx : kept_indices) {
        (*lb_ptr)[idx].cumu_prob = raw_scores[idx] - log_norm;
        (*lb_ptr)[idx].trustworthiness = std::exp((*lb_ptr)[idx].cumu_prob);
    }
}

// Compute sliding-window trustworthiness using top-N (N=3) path scores.
std::pair<std::vector<double>, std::vector<std::vector<double>>>
CovarianceMapMatch::compute_window_trustworthiness(
    const Traj_Candidates &tc,
    const std::vector<std::vector<double>> &log_emission_probabilities,
    const CMMTrajectory &traj,
    const CovarianceMapMatchConfig &config) {

    const size_t layer_count = tc.size();
    std::vector<double> trust_margins(layer_count, 0.0);
    std::vector<std::vector<double>> n_best(layer_count);
    if (layer_count == 0) {
        return {trust_margins, n_best};
    }

    const size_t window_length = static_cast<size_t>(std::max(1, config.window_length));
    std::vector<double> euclidean_distances;
    euclidean_distances.reserve(layer_count > 0 ? layer_count - 1 : 0);
    for (size_t i = 0; i + 1 < layer_count; ++i) {
        euclidean_distances.push_back(
            boost::geometry::distance(traj.geom.get_point(i), traj.geom.get_point(i + 1)));
    }

    const size_t k = 3;
    for (size_t end_idx = 0; end_idx < layer_count; ++end_idx) {
        size_t start_idx = (end_idx + 1 > window_length) ? end_idx + 1 - window_length : 0;

        std::vector<std::vector<double>> prev_scores(tc[start_idx].size());
        const auto *start_eps = (start_idx < log_emission_probabilities.size())
                                    ? &log_emission_probabilities[start_idx]
                                    : nullptr;
        for (size_t j = 0; j < tc[start_idx].size(); ++j) {
            double log_ep = (start_eps && j < start_eps->size()) ? (*start_eps)[j] : -std::numeric_limits<double>::infinity();
            if (log_ep > -std::numeric_limits<double>::infinity()) {
                push_top_k(&prev_scores[j], log_ep, k);
            }
        }

        for (size_t cursor = start_idx + 1; cursor <= end_idx; ++cursor) {
            size_t prev_idx = cursor - 1;
            std::vector<std::vector<double>> cur_scores(tc[cursor].size());
            double eu_dist = (prev_idx < euclidean_distances.size()) ? euclidean_distances[prev_idx] : 0.0;
            const auto *cur_eps = (cursor < log_emission_probabilities.size())
                                      ? &log_emission_probabilities[cursor]
                                      : nullptr;

            for (size_t b = 0; b < tc[cursor].size(); ++b) {
                double log_ep_b = (cur_eps && b < cur_eps->size()) ? (*cur_eps)[b] : -std::numeric_limits<double>::infinity();
                if (log_ep_b == -std::numeric_limits<double>::infinity()) {
                    continue;
                }
                for (size_t a = 0; a < tc[prev_idx].size(); ++a) {
                    const auto &paths_to_a = prev_scores[a];
                    if (paths_to_a.empty()) {
                        continue;
                    }
                    double sp_dist = get_sp_dist(&tc[prev_idx][a], &tc[cursor][b], config.reverse_tolerance);
                    if (sp_dist < 0) {
                        continue;
                    }
                    double tp = TransitionGraph::calc_tp(sp_dist, eu_dist);
                    if (tp <= 0) {
                        continue;
                    }
                    double log_tp = std::log(tp);
                    for (double prev_log : paths_to_a) {
                        push_top_k(&cur_scores[b], prev_log + log_tp + log_ep_b, k);
                    }
                }
            }
            prev_scores.swap(cur_scores);
        }

        std::vector<double> combined;
        for (const auto &scores : prev_scores) {
            for (double val : scores) {
                push_top_k(&combined, val, k);
            }
        }

        // Apply Top-K normalization to the window scores if they exist
        if (!combined.empty()) {
            double log_norm = log_sum_exp(combined);
            for (double &val : combined) {
                val = std::exp(val - log_norm); // Convert to linear probability [0, 1]
            }
            n_best[end_idx] = combined;
            if (combined.size() >= 2) {
                trust_margins[end_idx] = combined[0] - combined[1];
            } else if (combined.size() == 1) {
                trust_margins[end_idx] = combined[0];
            }
        } else {
            n_best[end_idx] = {};
            trust_margins[end_idx] = 0.0;
        }
    }

    return {trust_margins, n_best};
}

// Entry point used by CLI/Python binding: parse trajectories, optionally reproject them,
// run the matcher, and write the outputs requested in result_config.
std::string CovarianceMapMatch::match_gps_file(
    const FMM::CONFIG::GPSConfig &gps_config,
    const FMM::CONFIG::ResultConfig &result_config,
    const CovarianceMapMatchConfig &cmm_config,
    int input_epsg,
    bool use_omp) {
    std::ostringstream oss;
    std::string status;
    bool validate = true;

    // Fail fast if any configuration block is malformed.
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

    // Resolve column positions in a case-insensitive manner so user supplied
    // CSV files can use different header spelling without recompilation.
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

    // The parser supports both aggregated trajectories per row and point-by-point formats.
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
        // Aggregated format: each CSV row stores the full trajectory geometry and metadata.
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
        // Point-based format: accumulate samples per trajectory id to rebuild a LineString later.
        struct TrajectoryBuilder {
            CORE::LineString geom;
            std::vector<double> timestamps;
            std::vector<CovarianceMatrix> covariances;
            std::vector<double> protection_levels;
        };

        // Preserve insertion order so trajectories are emitted consistently with the input.
        std::unordered_map<long long, size_t> id_to_index;
        std::vector<long long> insertion_order;
        std::vector<TrajectoryBuilder> builders;

        // Track the maximum column index referenced to reject malformed rows early.
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

            // Lazily create a builder per trajectory id as rows arrive.
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
    // Align GPS observations with the network CRS if needed so the matcher
    // can operate in a consistent coordinate system.
    // Check if input CRS (input_epsg) differs from network CRS
    bool need_reprojection = false;
    if (network_.has_spatial_ref()) {
        // Get network EPSG code from WKT
        int network_epsg = 0;
        OGRSpatialReference network_sr;
        if (network_sr.importFromWkt(network_.get_spatial_ref_wkt().c_str()) == OGRERR_NONE) {
            const char *auth_name = network_sr.GetAuthorityName(nullptr);
            const char *auth_code = network_sr.GetAuthorityCode(nullptr);
            if (auth_name && auth_code && std::string(auth_name) == "EPSG") {
                network_epsg = std::stoi(auth_code);
            }
        }
        need_reprojection = (input_epsg != network_epsg);
        SPDLOG_INFO("Input EPSG: {}, Network EPSG: {}, Reprojection needed: {}",
                     input_epsg, network_epsg, need_reprojection);
    }

    if (need_reprojection) {
        try {
            // Create a reprojection function that uses the network's CRS
            // Note: maybe_reproject_trajectories will be refactored to use input_epsg
            trajectories_reprojected = maybe_reproject_trajectories(&trajectories, network_, need_reprojection);
        } catch (const std::exception &ex) {
            SPDLOG_WARN("Trajectory reprojection failed: {}", ex.what());
        }
    }

    // Prepare the CSV writer and the optional transformation back to input CRS.
    FMM::IO::CSVMatchResultWriter writer(result_config.file, result_config.output_config);
    std::unique_ptr<OGRCoordinateTransformation, decltype(&OCTDestroyCoordinateTransformation)>
        output_transform(nullptr, OCTDestroyCoordinateTransformation);

    // Only transform back to input CRS if reprojection was performed
    if (trajectories_reprojected && network_.has_spatial_ref()) {
        // Create transformation from network CRS back to input EPSG
        OGRSpatialReference network_sr;
        if (network_sr.importFromWkt(network_.get_spatial_ref_wkt().c_str()) == OGRERR_NONE) {
            network_sr.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
            OGRSpatialReference input_sr;
            if (input_sr.importFromEPSG(input_epsg) == OGRERR_NONE) {
                input_sr.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
                if (OGRCoordinateTransformation *ct = OGRCreateCoordinateTransformation(&network_sr, &input_sr)) {
                    output_transform.reset(ct);
                    SPDLOG_INFO("Created output transformation from network CRS back to EPSG:{}", input_epsg);
                } else {
                    SPDLOG_WARN("Failed to create output coordinate transformation; results remain in network CRS.");
                }
            } else {
                SPDLOG_WARN("Failed to create input CRS from EPSG:{}; results remain in network CRS.", input_epsg);
            }
        } else {
            SPDLOG_WARN("Failed to reconstruct network CRS for output transformation.");
        }
    }
    OGRCoordinateTransformation *output_transform_ptr = output_transform.get();
    if (!trajectories_reprojected) {
        output_transform_ptr = nullptr;
    }
    // Helper that restores the match results to the original CRS before writing them out.
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
        for (auto &cand_list : match->candidate_details) {
            for (auto &cand_info : cand_list) {
                double px = cand_info.x;
                double py = cand_info.y;
                if (output_transform_ptr->Transform(1, &px, &py)) {
                    cand_info.x = px;
                    cand_info.y = py;
                } else {
                    SPDLOG_TRACE("Failed to transform candidate detail back to geographic CRS.");
                }
            }
        }
    };
    // Maintain counters for logging throughput and matching success rates.
    const int step_size = 1000;
    int progress = 0;
    int points_matched = 0;
    int total_points = 0;
    int traj_matched = 0;
    int total_trajs = 0;
    auto begin_time = UTIL::get_current_time();

    // Parallel execution path guarded by OpenMP when multiple trajectories exist.
    if (use_omp && trajectories.size() > 1) {
#ifdef _OPENMP
        // Buffer for storing results to maintain output order
        // Since one trajectory can result in multiple MatchResults, we use a vector of vectors.
        std::vector<std::vector<std::pair<CORE::Trajectory, MM::MatchResult>>> result_buffer;
        const int trajectories_count = static_cast<int>(trajectories.size());
        result_buffer.resize(trajectories_count);
        #pragma omp parallel for schedule(dynamic)
        for (int idx = 0; idx < trajectories_count; ++idx) {
            const CMMTrajectory &trajectory = trajectories[idx];
            CMMTrajectory filtered_traj_base;
            std::vector<MM::MatchResult> results = match_traj(trajectory, cmm_config, &filtered_traj_base);
            
            for (auto &result : results) {
                // Slice the original trajectory based on the segment's indices
                CMMTrajectory segment_cmm_traj = slice_trajectory(trajectory, result.original_indices.front(), result.original_indices.back() + 1);
                CORE::Trajectory segment_traj{segment_cmm_traj.id, segment_cmm_traj.geom, segment_cmm_traj.timestamps};
                
                apply_output_transform(&segment_traj, &result);
                
                #pragma omp critical(buffer_section)
                {
                    result_buffer[idx].push_back(std::make_pair(segment_traj, result));
                }

                const int points_in_segment = segment_traj.geom.get_num_points();
                #pragma omp critical(progress_section)
                {
                    points_matched += points_in_segment;
                }
            }

            #pragma omp critical(progress_section)
            {
                ++progress;
                ++total_trajs;
                total_points += trajectory.geom.get_num_points();
                if (!results.empty()) {
                    ++traj_matched;
                }
                if (step_size > 0 && progress % step_size == 0) {
                    std::stringstream buf;
                    buf << "Progress " << progress << '\n';
                    std::cout << buf.rdbuf();
                }
            }
        }

        // Write results in order of original trajectory index
        for (int i = 0; i < trajectories_count; ++i) {
            for (const auto &item : result_buffer[i]) {
                writer.write_result(item.first, item.second);
            }
        }
#else
        use_omp = false;
#endif
    }

    // Serial fallback when OpenMP is disabled or there is just a single trajectory.
    if (!use_omp || trajectories.size() <= 1) {
        for (const auto &trajectory : trajectories) {
            if (progress % step_size == 0) {
                SPDLOG_INFO("Progress {}", progress);
            }
            CMMTrajectory filtered_traj_base;
            std::vector<MM::MatchResult> results = match_traj(trajectory, cmm_config, &filtered_traj_base);
            
            for (auto &result : results) {
                CMMTrajectory segment_cmm_traj = slice_trajectory(trajectory, result.original_indices.front(), result.original_indices.back() + 1);
                CORE::Trajectory segment_traj{segment_cmm_traj.id, segment_cmm_traj.geom, segment_cmm_traj.timestamps};
                
                apply_output_transform(&segment_traj, &result);
                writer.write_result(segment_traj, result);
                
                points_matched += segment_traj.geom.get_num_points();
            }

            total_points += trajectory.geom.get_num_points();
            ++total_trajs;
            if (!results.empty()) {
                ++traj_matched;
            }
            ++progress;
        }
    }

    auto end_time = UTIL::get_current_time();
    double duration = UTIL::get_duration(begin_time, end_time);
    double speed = duration > 0 ? static_cast<double>(points_matched) / duration : 0.0;

    // Report a concise summary so CLI users can inspect throughput and error counts.
    oss << "Status: success\n";
    oss << "Time takes " << duration << " seconds\n";
    oss << "Total points " << total_points << " matched " << points_matched << "\n";
    oss << "Trajectories processed " << total_trajs << " matched " << traj_matched << "\n";
    oss << "Map match speed " << speed << " points/s \n";
    oss << "Trajectories skipped " << invalid_records << "\n";

    return oss.str();
}
