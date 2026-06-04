//
// Created for CMM implementation
// Covariance-based map matching algorithm
//

#include "mm/cmm/cmm_algorithm.hpp"
#include "algorithm/geom_algorithm.hpp"
#include "util/util.hpp"
#include "util/debug.hpp"
#include "io/gps_reader.hpp"
#include <deque>
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
                                                   bool filtered_arg,
                                                   bool enable_gap_bridging_arg,
                                                   double max_gap_distance_arg,
                                                   double min_gps_error_degrees_arg,
                                                   double max_interval_arg,
                                                   double trustworthiness_threshold_arg,
                                                   double map_error_std_arg,
                                                   double background_prob_arg,
                                                   double phmi_arg,
                                                   int lag_steps_arg,
                                                   double phmi_pl_multiplier_arg,
                                                   double h0_prior_log_odds_arg)
    : k(k_arg), min_candidates(min_candidates_arg),
      protection_level_multiplier(protection_level_multiplier_arg),
      reverse_tolerance(reverse_tolerance_arg),
      normalized(normalized_arg),
      use_mahalanobis_candidates(use_mahalanobis_candidates_arg),
      filtered(filtered_arg),
      enable_gap_bridging(enable_gap_bridging_arg),
      max_gap_distance(max_gap_distance_arg),
      min_gps_error_degrees(min_gps_error_degrees_arg),
      max_interval(max_interval_arg),
      trustworthiness_threshold(trustworthiness_threshold_arg),
      map_error_std(map_error_std_arg),
      background_prob(background_prob_arg),
      phmi(phmi_arg),
      lag_steps(lag_steps_arg),
      phmi_pl_multiplier(phmi_pl_multiplier_arg),
      h0_prior_log_odds(h0_prior_log_odds_arg) {
}

// Dump runtime configuration for debugging or reproducibility.
void CovarianceMapMatchConfig::print() const {
    SPDLOG_INFO("CMMAlgorithmConfig");
    SPDLOG_INFO("k {} min_candidates {} protection_level_multiplier {} reverse_tolerance {}",
                k, min_candidates, protection_level_multiplier, reverse_tolerance);
    SPDLOG_INFO("normalized {} use_mahalanobis {} filtered {}",
                normalized, use_mahalanobis_candidates, filtered);
    SPDLOG_INFO("gap_bridging {} max_gap_distance {}", enable_gap_bridging, max_gap_distance);
    SPDLOG_INFO("min_gps_error_degrees {} max_interval {} trustworthiness_threshold {}",
                min_gps_error_degrees, max_interval, trustworthiness_threshold);
    SPDLOG_INFO("map_error_std {} background_prob {} phmi {} lag_steps {}", map_error_std, background_prob, phmi, lag_steps);
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
    bool filtered = xml_data.get("config.parameters.filtered", true);

    // Gap bridging and integrity parameters
    bool enable_gap_bridging = xml_data.get("config.parameters.enable_gap_bridging", true);
    double max_gap_distance = xml_data.get("config.parameters.max_gap_distance", 2000.0);
    double phmi = xml_data.get("config.parameters.phmi", 1.0e-5);

    // Minimum GPS error to prevent over-confidence
    double min_gps_error_degrees = xml_data.get("config.parameters.min_gps_error_degrees", 1.0e-4);

    double max_interval = xml_data.get("config.parameters.max_interval", 180.0);
    double trustworthiness_threshold = xml_data.get("config.parameters.trustworthiness_threshold", 0.0);

    // New parameters for additive map noise and background noise normalization
    double map_error_std = xml_data.get("config.parameters.map_error_std", 5.0e-5);
    double background_prob = xml_data.get("config.parameters.background_prob", 0.1);

    // Fixed-lag smoothing: 0 = realtime filtering, N = delay N steps
    int lag_steps = xml_data.get("config.parameters.lag_steps", 0);

    // PHMI integrity multiplier: decoupled from search radius
    double phmi_pl_multiplier = xml_data.get("config.parameters.phmi_pl_multiplier", 5.0);
    double h0_prior_log_odds = xml_data.get("config.parameters.h0_prior_log_odds", 0.0);

    return CovarianceMapMatchConfig{k, min_candidates, protection_level_multiplier, reverse_tolerance,
                                    normalized, use_mahalanobis_candidates,
                                    filtered,
                                    enable_gap_bridging, max_gap_distance, min_gps_error_degrees,
                                    max_interval, trustworthiness_threshold,
                                    map_error_std, background_prob, phmi, lag_steps,
                                    phmi_pl_multiplier, h0_prior_log_odds};
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
    bool filtered = arg_data["filtered"].as<bool>();

    // Check if new args exist (assuming they are registered) or use defaults
    bool enable_gap = arg_data.count("enable_gap_bridging") ? arg_data["enable_gap_bridging"].as<bool>() : true;
    double max_gap = arg_data.count("max_gap_distance") ? arg_data["max_gap_distance"].as<double>() : 2000.0;
    double phmi = arg_data.count("phmi") ? arg_data["phmi"].as<double>() : 1.0e-5;

    // Minimum GPS error to prevent over-confidence
    double min_gps_error = arg_data.count("min_gps_error_degrees") ? arg_data["min_gps_error_degrees"].as<double>() : 1.0e-4;

    double max_interval = arg_data.count("max_interval") ? arg_data["max_interval"].as<double>() : 180.0;
    double trustworthiness_threshold = arg_data.count("trustworthiness_threshold") ? arg_data["trustworthiness_threshold"].as<double>() : 0.0;

    // New parameters for additive map noise and background noise normalization
    double map_error_std = arg_data.count("map_error_std") ? arg_data["map_error_std"].as<double>() : 5.0e-5;
    double background_prob = arg_data.count("background_prob") ? arg_data["background_prob"].as<double>() : 0.1;
    int lag_steps = arg_data.count("lag_steps") ? arg_data["lag_steps"].as<int>() : 0;
    double phmi_pl_multiplier = arg_data.count("phmi_pl_multiplier") ? arg_data["phmi_pl_multiplier"].as<double>() : 5.0;
    double h0_prior_log_odds = arg_data.count("h0_prior_log_odds") ? arg_data["h0_prior_log_odds"].as<double>() : 0.0;

    return CovarianceMapMatchConfig{k, min_candidates, protection_level_multiplier, reverse_tolerance,
                                    normalized, use_mahalanobis_candidates,
                                    filtered,
                                    enable_gap, max_gap, min_gps_error,
                                    max_interval, trustworthiness_threshold,
                                    map_error_std, background_prob, phmi, lag_steps,
                                    phmi_pl_multiplier, h0_prior_log_odds};
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
        ("filtered", "Filter out points with no candidates or disconnected transitions",
         cxxopts::value<bool>()->default_value("true"))
        ("enable_gap_bridging", "Enable trajectory gap bridging",
         cxxopts::value<bool>()->default_value("true"))
        ("max_gap_distance", "Max distance for gap bridging (meters)",
         cxxopts::value<double>()->default_value("2000.0"))
        ("min_gps_error_degrees", "Minimum GPS error in degrees to prevent over-confidence (default 1e-4 ≈ 11m)",
         cxxopts::value<double>()->default_value("1.0e-4"))
        ("max_interval", "Maximum time interval (seconds) to split segments",
         cxxopts::value<double>()->default_value("180.0"))
        ("trustworthiness_threshold", "Threshold on trustworthiness posterior [0,1] to filter low-confidence epochs (0.0)",
         cxxopts::value<double>()->default_value("0.0"))
        ("map_error_std", "Map error standard deviation in degrees for additive noise (default 5e-5 ≈ 5m)",
         cxxopts::value<double>()->default_value("5.0e-5"))
        ("background_prob", "Background state linear probability for off-road/unmapped-road (default 0.1)",
         cxxopts::value<double>()->default_value("-20.0"))
        ("phmi", "Probability of Hazardously Misleading Integrity information (default 1e-5)",
         cxxopts::value<double>()->default_value("1.0e-5"))
        ("lag_steps", "Fixed-lag smoothing steps (0=realtime filtering, N=delay N steps)",
         cxxopts::value<int>()->default_value("0"));
}

// Append a short textual description for the Python binding documentation.
void CovarianceMapMatchConfig::register_help(std::ostringstream &oss) {
    oss << "-k/--candidates (optional) <int>: Number of candidates (8)\n";
    oss << "--min_candidates (optional) <int>: Minimum number of candidates to keep (3)\n";
    oss << "--protection_level_multiplier (optional) <double>: Multiplier for protection level (1.0)\n";
    oss << "--reverse_tolerance (optional) <double>: proportion of reverse movement allowed on an edge\n";
    oss << "--normalized (optional) <bool>: whether to normalize emission probabilities (true)\n";
    oss << "--use_mahalanobis (optional) <bool>: whether to use Mahalanobis-based candidate search (true)\n";
    oss << "--filtered (optional) <bool>: whether to filter out points with no candidates or disconnected transitions (true)\n";
    oss << "--enable_gap_bridging (optional) <bool>: Enable trajectory gap bridging (true)\n";
    oss << "--max_gap_distance (optional) <double>: Max distance for gap bridging in meters (2000.0)\n";
    oss << "--min_gps_error_degrees (optional) <double>: Minimum GPS error in degrees to prevent over-confidence (1e-4 ≈ 11m)\n";
    oss << "--max_interval (optional) <double>: Maximum time interval (seconds) to split segments (180.0)\n";
    oss << "--trustworthiness_threshold (optional) <double>: trustworthiness posterior [0,1] threshold for filtering (0.0)\n";
    oss << "--map_error_std (optional) <double>: Map error standard deviation in degrees for additive noise (5e-5 ≈ 5m)\n";
    oss << "--background_prob (optional) <double>: Background state linear probability for off-road/unmapped-road (0.1)\n";
    oss << "--phmi (optional) <double>: Probability of Hazardously Misleading Integrity information (1e-5)\n";
    oss << "--lag_steps (optional) <int>: Fixed-lag smoothing steps (0=realtime filtering, N=delay N steps)\n";
}

// Quick sanity checks to guard against invalid user supplied parameters.
bool CovarianceMapMatchConfig::validate() const {
    if (k <= 0 || min_candidates <= 0 || min_candidates > k ||
        protection_level_multiplier <= 0 || reverse_tolerance < 0 ||
        max_gap_distance < 0 ||
        map_error_std < 0 || phmi < 0 || phmi > 1.0 || lag_steps < 0) {
        SPDLOG_CRITICAL("Invalid CMM parameter k {} min_candidates {} "
                       "protection_level_multiplier {} reverse_tolerance {} "
                       "max_gap_distance {} map_error_std {} phmi {}",
                       k, min_candidates, protection_level_multiplier, reverse_tolerance,
                       max_gap_distance, map_error_std, phmi);
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
    // Add a small regularization term to determinant for numerical stability
    if (det <= 1e-50) {
        return -std::numeric_limits<double>::infinity();
    }

    double mahalanobis_dist_sq = cov_inv.m[0][0] * dx * dx +
                                 2 * cov_inv.m[0][1] * dx * dy +
                                 cov_inv.m[1][1] * dy * dy;

    // Log Gaussian: -0.5 * (log(2*pi) + log(det + eps) + dist^2)
    static const double log_2pi = std::log(2.0 * M_PI);
    return -0.5 * (log_2pi + std::log(det + 1e-12) + mahalanobis_dist_sq);
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
                Traj_Candidates traj_candidates;
                #pragma omp critical(knn_section)
                {
                    traj_candidates = network_.search_tr_cs_knn(single_point_geom, config.k, search_radius);
                }
                Point_Candidates base_candidates = traj_candidates.empty() ? Point_Candidates() : traj_candidates[0];

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

                // ── Deduplicate: keep only the best-metric candidate per edge ──
                // Without this, each edge generates up to 3 candidates (Mahalanobis
                // projection + source node + target node), inflating the effective k
                // and compressing softmax trust to ~1/K_effective ≈ 0.5.
                // Rule: one candidate per edge → keep best Mahalanobis metric.
                {
                    std::map<NETWORK::EdgeIndex, size_t> best_per_edge;
                    for (size_t i = 0; i < candidate_pool.size(); ++i) {
                        const auto &c = candidate_pool[i];
                        if (c.candidate.edge == nullptr) continue;
                        NETWORK::EdgeIndex eidx = c.candidate.edge->index;
                        auto it = best_per_edge.find(eidx);
                        if (it == best_per_edge.end()) {
                            best_per_edge[eidx] = i;
                        } else if (c.metric < candidate_pool[it->second].metric) {
                            best_per_edge[eidx] = i;
                        }
                    }
                    if (best_per_edge.size() < candidate_pool.size()) {
                        std::vector<CandidateWithMetric> deduped;
                        deduped.reserve(best_per_edge.size());
                        for (const auto &kv : best_per_edge) {
                            deduped.push_back(std::move(candidate_pool[kv.second]));
                        }
                        candidate_pool.swap(deduped);
                    }
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
                            // Log Gaussian: -0.5 * (log(2*pi) + log(det + eps) + mahal_sq)
                            log_probability = -0.5 * (std::log(2.0 * M_PI) + std::log(det_eff + 1e-12) + mahal_sq);
                        }
                    }
                    raw_probabilities.push_back(log_probability);
                }

                if (selected_candidates.size() >= static_cast<size_t>(config.min_candidates) ||
                    edges_to_consider.empty() || radius_expanded) {
                    break;
                }

                search_radius *= 2.0;
                radius_expanded = true;
            }
        } else {
            // Basic candidate search that directly relies on network_kNN results.
            CORE::LineString single_point_geom;
            single_point_geom.add_point(point);
            Traj_Candidates traj_candidates;
            #pragma omp critical(knn_section)
            {
                traj_candidates = network_.search_tr_cs_knn(single_point_geom, config.k, search_radius);
            }
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
                        // Log Gaussian: -0.5 * (log(2*pi) + log(det + eps) + mahalanobis_dist_sq)
                        log_probability = -0.5 * (std::log(2.0 * M_PI) + std::log(det_eff + 1e-12) + mahalanobis_dist_sq);
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
        log_emission_probs.reserve(raw_probabilities.size());

        double sum_ep_raw = 0.0;
        std::vector<double> ep_raw_list(raw_probabilities.size(), 0.0);

        // 1. 转为线性 EP 并求和
        for (size_t k = 0; k < raw_probabilities.size(); ++k) {
            if (raw_probabilities[k] > -std::numeric_limits<double>::infinity()) {
                ep_raw_list[k] = std::exp(raw_probabilities[k]);
                sum_ep_raw += ep_raw_list[k];
            }
        }

        // 2. EP 线性归一化并转回 Log 空间（PHMI 完好性条件化）
        //    PHMI (Probability of Hazardously Misleading Information) encodes
        //    the GNSS integrity risk: with probability P_HMI, the true position
        //    may lie outside the Protection Level (PL), i.e. the road network
        //    may be wrong or the vehicle is off-road.
        //
        //    For each candidate k with Euclidean distance d_k from GPS obs:
        //      d_k ≤ PL:  candidate is "integrity-valid" → weight = (1-PHMI)
        //      d_k > PL:  candidate exceeds integrity bound → weight = PHMI
        //
        //    Normalization: separately normalize inside-PL group by (1-PHMI)
        //    and outside-PL group by PHMI.  This ensures total mass = 1 and
        //    prevents overconfidence when the road network cannot explain the
        //    GNSS observation.
        //
        //    When PHMI = 0, the behaviour is identical to the original
        //    unconditional model.
        double one_minus_phmi = 1.0 - config.phmi;
        double phmi = config.phmi;

        // Effective Protection Level for PHMI integrity check:
        //   effective_PL = raw_PL * phmi_pl_multiplier
        // The search multiplier controls how far to look for candidates;
        // the PHMI multiplier controls the integrity threshold independently,
        // allowing discrimination even when many candidates are found.
        double phmi_effective_pl = protection_level * config.phmi_pl_multiplier;

        // Separate raw EP sums for inside-PL and outside-PL groups
        double sum_ep_in_raw = 0.0, sum_ep_out_raw = 0.0;
        for (size_t k = 0; k < raw_probabilities.size(); ++k) {
            if (ep_raw_list[k] <= 0) continue;
            if (selected_candidates[k].dist <= phmi_effective_pl) {
                sum_ep_in_raw += ep_raw_list[k];
            } else {
                sum_ep_out_raw += ep_raw_list[k];
            }
        }

        for (size_t k = 0; k < raw_probabilities.size(); ++k) {
            if (ep_raw_list[k] > 0 && sum_ep_raw > 0) {
                bool inside_pl = (selected_candidates[k].dist <= phmi_effective_pl);

                double integrity_factor, norm_sum;
                if (config.phmi > 0) {
                    // PHMI-enabled: normalize each group independently
                    if (inside_pl) {
                        integrity_factor = one_minus_phmi;
                        norm_sum = (sum_ep_in_raw > 0) ? sum_ep_in_raw : sum_ep_raw;
                    } else {
                        integrity_factor = phmi;
                        norm_sum = (sum_ep_out_raw > 0) ? sum_ep_out_raw : sum_ep_raw;
                    }
                } else {
                    // PHMI disabled: original unconditional normalization
                    integrity_factor = one_minus_phmi;
                    norm_sum = sum_ep_raw;
                }

                double linear_ep_norm = integrity_factor * (ep_raw_list[k] / norm_sum);
                log_emission_probs.push_back(std::log(
                    std::max(linear_ep_norm, std::numeric_limits<double>::min())));
            } else {
                log_emission_probs.push_back(-std::numeric_limits<double>::infinity());
            }
        }

        // ── 3. Background state: constant-discount off-road pseudo-candidate ──
        // Appends a background candidate representing "vehicle not on any mapped road."
        // Real candidates are scaled by (1 - bg_prob) to make room; bg gets bg_prob.
        // This preserves PHMI normalization (inside/outside PL split) and adds an
        // anti-label-bias guard: when all road candidates have low EP, the background
        // absorbs probability mass and prevents softmax overconfidence.
        if (config.background_prob > 0.0 && config.background_prob < 1.0) {
            double bg_log = std::log(config.background_prob);
            double real_scale = 1.0 - config.background_prob;
            for (auto &lep : log_emission_probs) {
                if (lep > -std::numeric_limits<double>::infinity()) {
                    lep = std::log(std::exp(lep) * real_scale);
                }
            }
            Candidate bg{};
            bg.index = next_candidate_index++;
            bg.edge = nullptr;  // null edge → skipped in TP computation (off-road)
            bg.offset = 0.0;
            bg.dist = std::numeric_limits<double>::infinity();
            selected_candidates.push_back(std::move(bg));
            log_emission_probs.push_back(bg_log);
        }

        SPDLOG_TRACE("Point {}: {} candidates kept", i, selected_candidates.size());
        result.candidates.push_back(std::move(selected_candidates));
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
    // Validate trajectory
    if (!traj.is_valid()) {
        SPDLOG_ERROR("Invalid trajectory: covariance and protection level data mismatch");
        return {};
    }

    CandidateSearchResult candidate_result;
    #pragma omp critical(search_candidates_section)
    {
        candidate_result = search_candidates_with_protection_level(
            traj.geom, traj.covariances, traj.protection_levels, config);
    }

    const Traj_Candidates &tc_raw = candidate_result.candidates;
    const std::vector<std::vector<double>> &log_eps_raw = candidate_result.emission_probabilities;

    // Pre-extract global candidate details to ensure they are available even on failure
    std::vector<std::vector<CandidateEmission>> global_candidate_details(tc_raw.size());
    for (size_t i = 0; i < tc_raw.size(); ++i) {
        for (size_t j = 0; j < tc_raw[i].size(); ++j) {
            global_candidate_details[i].push_back({
                boost::geometry::get<0>(tc_raw[i][j].point),
                boost::geometry::get<1>(tc_raw[i][j].point),
                std::exp(log_eps_raw[i][j])
            });
        }
    }

    // Lambda to create a fallback MatchResult for failed segments or points
    auto create_fallback_result = [&](int start_idx, int end_idx, MatchStatus status) -> MatchResult {
        MatchResult res;
        res.id = traj.id;
        res.status = status;
        for (int i = start_idx; i <= end_idx && i < static_cast<int>(tc_raw.size()); ++i) {
            res.candidate_details.push_back(global_candidate_details[i]);
            res.original_indices.push_back(i);
        }
        return res;
    };

    std::vector<int> valid_indices;
    valid_indices.reserve(tc_raw.size());
    for (int i = 0; i < tc_raw.size(); ++i) {
        if (!tc_raw[i].empty()) {
            valid_indices.push_back(i);
        }
    }

    if (valid_indices.size() < 3) {
        return {create_fallback_result(0, tc_raw.size() - 1, MatchStatus::FAILED_NO_CANDIDATE)};
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

    Traj_Candidates empty_tc;
    std::vector<std::vector<double>> empty_log_eps;

    auto process_sub_segment = [&](TransitionGraph* tg_ptr, const std::vector<int>& sub_indices,
                                     std::vector<double>* h0_lambda_vec = nullptr) {
        if (tg_ptr == nullptr || sub_indices.empty()) return;
        int start_real_idx = sub_indices.front();
        int end_real_idx = sub_indices.back();
        
        TGOpath tg_opath = tg_ptr->backtrack();

        if (tg_opath.empty()) {
            final_results.push_back(create_fallback_result(start_real_idx, end_real_idx, MatchStatus::FAILED_DISCONNECTED));
            return;
        }

        // Build a continuous trajectory for the sub-segment to ensure correct distance calculations
        CMMTrajectory segment_traj;
        segment_traj.id = traj.id;
        for (int idx : sub_indices) {
            segment_traj.geom.add_point(traj.geom.get_point(idx));
            if (!traj.timestamps.empty() && idx < static_cast<int>(traj.timestamps.size())) {
                segment_traj.timestamps.push_back(traj.timestamps[idx]);
            }
        }

        MatchedCandidatePath matched_candidate_path;
        matched_candidate_path.reserve(tg_opath.size());

        std::vector<MatchedCandidate> filtered_path;
        TGOpath filtered_tg_opath;
        std::vector<int> filtered_indices;
        std::vector<double> filtered_sp_dist;
        std::vector<double> filtered_eu_dist;
        std::vector<std::vector<CandidateEmission>> filtered_details;

        std::vector<std::vector<CandidateEmission>> all_details;
        for (int idx : sub_indices) {
            all_details.push_back(global_candidate_details[idx]);
        }

        // Pre-extract top-3 trustworthiness per layer from lag-smoothing softmax posterior
        const auto& layers = tg_ptr->get_layers();
        std::vector<std::vector<double>> layer_nbest;
        layer_nbest.reserve(layers.size());
        for (size_t li = 0; li < layers.size(); ++li) {
            const TGLayer& layer = layers[li];
            std::vector<double> tw_vals;
            tw_vals.reserve(layer.size());
            for (const TGNode& node : layer) {
                if (node.trustworthiness > 0.0) {
                    tw_vals.push_back(node.trustworthiness);
                }
            }
            std::sort(tw_vals.begin(), tw_vals.end(), std::greater<double>());
            std::vector<double> top3(tw_vals.begin(),
                tw_vals.begin() + std::min<size_t>(3, tw_vals.size()));
            layer_nbest.push_back(std::move(top3));
        }

        for (size_t i = 0; i < tg_opath.size(); ++i) {
            const TGNode *node = tg_opath[i];
            double sp = (i == 0) ? 0.0 : node->sp_dist;
            double eu = (i == 0) ? 0.0 : boost::geometry::distance(segment_traj.geom.get_point(i-1), segment_traj.geom.get_point(i));

            // TW = P(x_t = i* | z_{1:t}) — filtering posterior of the Viterbi winner.
            // Computed by the forward algorithm: α_t(i) / Σ_j α_t(j) = softmax(forward_cumu).
            // This is a per-epoch, per-candidate probability that varies with local evidence.
            double trust = node->trustworthiness;

            double h0_lambda_val = (h0_lambda_vec != nullptr && i > 0 && i - 1 < h0_lambda_vec->size())
                ? std::exp(std::max(-700.0, std::min(700.0, (*h0_lambda_vec)[i - 1]))) : 1.0;

            MatchedCandidate mc{*(node->c), std::exp(node->ep), node->tp, node->cumu_prob, node->sp_dist, trust, node->delta_entropy, node->posterior_entropy, h0_lambda_val};
            matched_candidate_path.push_back(mc);

            if (!config.filtered || trust >= config.trustworthiness_threshold) {
                filtered_path.push_back(mc);
                filtered_tg_opath.push_back(node);
                filtered_indices.push_back(sub_indices[i]);
                filtered_sp_dist.push_back(sp);
                filtered_eu_dist.push_back(eu);
                filtered_details.push_back(all_details[i]);
            }
        }

        if (filtered_path.empty()) {
            final_results.push_back(create_fallback_result(start_real_idx, end_real_idx, MatchStatus::FAILED_NO_CANDIDATE));
            return;
        }

        O_Path opath;
        for (const auto& mc : filtered_path) opath.push_back(mc.c.edge->id);

        std::vector<int> indices_mapping;
        C_Path cpath = ubodt_->construct_complete_path(traj.id, filtered_tg_opath, network_.get_edges(), &indices_mapping, config.reverse_tolerance);
        LineString mgeom = network_.complete_path_to_geometry(segment_traj.geom, cpath);

        MatchResult res;
        res.id = traj.id;
        res.status = (sub_indices.size() == traj.geom.get_num_points()) ? MatchStatus::SUCCESS : MatchStatus::PARTIAL;
        res.opt_candidate_path = std::move(filtered_path);
        res.opath = std::move(opath);
        res.cpath = std::move(cpath);
        res.indices = std::move(indices_mapping);
        res.mgeom = std::move(mgeom);
        res.sp_distances = std::move(filtered_sp_dist);
        res.eu_distances = std::move(filtered_eu_dist);

        std::vector<std::vector<double>> filtered_nbest;
        filtered_nbest.reserve(tg_opath.size());
        for (size_t i = 0; i < tg_opath.size(); ++i) {
            const TGNode *node = tg_opath[i];
            double trust = node->trustworthiness;
            if (!config.filtered || trust >= config.trustworthiness_threshold) {
                if (i < layer_nbest.size()) {
                    filtered_nbest.push_back(layer_nbest[i]);
                } else {
                    filtered_nbest.push_back({});
                }
            }
        }
        res.nbest_trustworthiness = std::move(filtered_nbest);
        res.original_indices = std::move(filtered_indices);
        res.candidate_details = std::move(filtered_details);

        final_results.push_back(std::move(res));
    };

    for (const auto& segment_indices : segments) {
        double log_prob_unconsidered = -std::numeric_limits<double>::infinity();

        std::vector<int> current_sub_indices; 
        int start_real_idx = segment_indices.front();
        current_sub_indices.push_back(start_real_idx);

        auto tg_ptr = std::make_unique<TransitionGraph>(empty_tc, empty_log_eps, true);
        // Pre-reserve layers to prevent reallocation which would invalidate pointers
        tg_ptr->get_layers().reserve(segment_indices.size() + 1); 

        // Manually build the first layer to point to stable tc_raw data
        TGLayer start_layer;
        start_layer.reserve(tc_raw[start_real_idx].size());
        for (size_t k = 0; k < tc_raw[start_real_idx].size(); ++k) {
            start_layer.push_back(TGNode{
                &tc_raw[start_real_idx][k], nullptr, log_eps_raw[start_real_idx][k], 0,
                -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), 0, 0
            });
        }
        tg_ptr->get_layers().push_back(std::move(start_layer));
        initialize_first_layer(&tg_ptr->get_layers()[0], config, log_prob_unconsidered);

        // ── Fixed-lag smoothing buffer ──────────────────────────────────────
        // Stores LagEntry (layer ptr, tp matrix, frac_inside_pl) for lag_steps+1 layers.
        std::deque<LagEntry> lag_buffer;
        if (config.lag_steps > 0) {
            lag_buffer.push_back({&tg_ptr->get_layers()[0], {}, 1.0});
        }

        // ── Sequential Bayesian H0 test: trajectory-global Λ accumulation ──
        // λ_t = λ₀ · Π_{τ=0}^{t} LR_τ   (cumulative likelihood ratio)
        // α_t = λ_t / (1 + λ_t)          (posterior P(H0 | z_{0:t}))
        std::vector<double> h0_log_lambdas;     // log(λ_t) per original index
        double h0_log_lambda = config.h0_prior_log_odds;  // current log(λ_t)

        TGLayer* last_valid_layer = &tg_ptr->get_layers()[0];
        int last_valid_real = start_real_idx;
        std::vector<int> skipped_indices;

        for (size_t i = 1; i < segment_indices.size(); ++i) {
            int next_real = segment_indices[i];

            double dist = boost::geometry::distance(traj.geom.get_point(last_valid_real),
                                                   traj.geom.get_point(next_real));

            // Physical constraints: check speed and interval
            double time_diff = 0.0;
            if (!traj.timestamps.empty() && traj.timestamps.size() > static_cast<size_t>(next_real) && 
                traj.timestamps.size() > static_cast<size_t>(last_valid_real)) {
                time_diff = std::abs(traj.timestamps[next_real] - traj.timestamps[last_valid_real]);
            }
            double speed = (time_diff > 0) ? (dist / time_diff) : std::numeric_limits<double>::infinity();
            constexpr double MAX_REASONABLE_SPEED = 40.0; // 144 km/h

            if (config.enable_gap_bridging && (speed > MAX_REASONABLE_SPEED || time_diff > config.max_interval)) {
                // Sub-trajectory boundary: flush smoothing buffer before finishing this segment
                flush_lag_buffer(lag_buffer, *this, config.lag_steps);
                process_sub_segment(tg_ptr.get(), current_sub_indices, &h0_log_lambdas);

                for (int skipped_idx : skipped_indices) {
                    final_results.push_back(create_fallback_result(skipped_idx, skipped_idx, MatchStatus::FAILED_DISCONNECTED));
                }
                skipped_indices.clear();

                current_sub_indices.clear();
                current_sub_indices.push_back(next_real);

                tg_ptr = std::make_unique<TransitionGraph>(empty_tc, empty_log_eps, true);
                tg_ptr->get_layers().reserve(segment_indices.size() + 1);

                TGLayer new_start_layer;
                new_start_layer.reserve(tc_raw[next_real].size());
                for (size_t k = 0; k < tc_raw[next_real].size(); ++k) {
                    new_start_layer.push_back(TGNode{
                        &tc_raw[next_real][k], nullptr, log_eps_raw[next_real][k], 0,
                        -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), 0, 0
                    });
                }
                tg_ptr->get_layers().push_back(std::move(new_start_layer));
                initialize_first_layer(&tg_ptr->get_layers()[0], config, log_prob_unconsidered);

                // Restart smoothing buffer for new sub-trajectory
                if (config.lag_steps > 0) {
                    lag_buffer.push_back({&tg_ptr->get_layers()[0], {}, 1.0});
                }
                // Reset H0 lambda for new sub-trajectory
                h0_log_lambdas.clear();
                h0_log_lambda = config.h0_prior_log_odds;

                last_valid_layer = &tg_ptr->get_layers()[0];
                last_valid_real = next_real;
                continue;
            }

            TGLayer next_layer;
            for (size_t k=0; k<tc_raw[next_real].size(); ++k) {
                next_layer.push_back(TGNode{&tc_raw[next_real][k], nullptr, log_eps_raw[next_real][k], 0, -std::numeric_limits<double>::infinity(),
                                           -std::numeric_limits<double>::infinity(), 0, 0});
            }

            bool connected = false;
            std::vector<std::vector<double>> tp_raw_smoothing;
            std::vector<std::vector<double>>* tp_raw_ptr =
                (config.lag_steps > 0) ? &tp_raw_smoothing : nullptr;
            update_layer_cmm(last_valid_layer, &next_layer, dist, &connected, config,
                            log_prob_unconsidered, tp_raw_ptr);

            bool should_restart = config.enable_gap_bridging &&
                                  !tc_raw[next_real].empty() &&
                                  current_sub_indices.size() >= 2;

            if (!connected) {
                if (!should_restart) {
                    skipped_indices.push_back(next_real);
                }

                if (should_restart) {
                    // 1. Flush smoothing buffer and commit current sub-segment
                    flush_lag_buffer(lag_buffer, *this, config.lag_steps);
                    process_sub_segment(tg_ptr.get(), current_sub_indices, &h0_log_lambdas);
                    for (int skipped_idx : skipped_indices) {
                        final_results.push_back(create_fallback_result(skipped_idx, skipped_idx, MatchStatus::FAILED_DISCONNECTED));
                    }
                    skipped_indices.clear();

                    // 2. 以 next_real 为起点重建 TransitionGraph
                    current_sub_indices.clear();
                    current_sub_indices.push_back(next_real);

                    tg_ptr = std::make_unique<TransitionGraph>(empty_tc, empty_log_eps, true);
                    tg_ptr->get_layers().reserve(segment_indices.size() - i + 1);

                    TGLayer restart_layer;
                    restart_layer.reserve(tc_raw[next_real].size());
                    for (size_t k = 0; k < tc_raw[next_real].size(); ++k) {
                        restart_layer.push_back(TGNode{
                            &tc_raw[next_real][k], nullptr, log_eps_raw[next_real][k], 0,
                            -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), 0, 0
                        });
                    }
                    tg_ptr->get_layers().push_back(std::move(restart_layer));
                    log_prob_unconsidered = -std::numeric_limits<double>::infinity();
                    initialize_first_layer(&tg_ptr->get_layers()[0], config, log_prob_unconsidered);

                    // Restart smoothing buffer
                    if (config.lag_steps > 0) {
                        lag_buffer.push_back({&tg_ptr->get_layers()[0], {}, 1.0});
                    }
                    // Reset H0 lambda for new sub-trajectory
                    h0_log_lambdas.clear();
                    h0_log_lambda = config.h0_prior_log_odds;

                    last_valid_layer = &tg_ptr->get_layers()[0];
                    last_valid_real = next_real;
                }
            } else {
                for (int skipped_idx : skipped_indices) {
                    final_results.push_back(create_fallback_result(skipped_idx, skipped_idx, MatchStatus::FAILED_DISCONNECTED));
                }
                skipped_indices.clear();

                tg_ptr->get_layers().push_back(std::move(next_layer));
                current_sub_indices.push_back(next_real);

                // CRITICAL: Get new pointer after push_back because it might have moved
                // though we called reserve, it is safer to update it.
                last_valid_layer = &tg_ptr->get_layers().back();
                last_valid_real = next_real;

                // ── Accumulate global H0 lambda (always, not gated by lag_steps) ──
                // Bayesian sequential test:
                //   LR_t = P(z_t | H0) / P(z_t | ¬H0) = frac_inside / PHMI
                //   λ_t = λ_{t-1} × LR_t
                //   α_t = λ_t / (1 + λ_t) discounts trust at output (line 1371).
                // Fix: moved outside lag_steps gate so it runs with lag=0 too.
                {
                    double raw_pl = (next_real < static_cast<int>(traj.protection_levels.size()))
                        ? traj.protection_levels[next_real] : 0.0;
                    double effective_pl = raw_pl * config.phmi_pl_multiplier;
                    int n_inside_pl = 0, n_total = 0;
                    for (size_t b = 0; b < next_layer.size(); ++b) {
                        TGNode& nb = next_layer[b];
                        if (nb.c == nullptr) continue;
                        n_total++;
                        if (nb.c->dist <= effective_pl) n_inside_pl++;
                    }
                    double frac_inside =
                        (n_total > 0) ? static_cast<double>(n_inside_pl) / n_total : 1.0;

                    double phmi_floor = std::max(config.phmi / 1000.0, 1.0e-10);
                    double f_clamp = std::max(phmi_floor,
                        std::min(1.0 - phmi_floor, frac_inside));
                    double h0_lr = f_clamp / config.phmi;
                    h0_log_lambda += std::log(h0_lr);
                    h0_log_lambdas.push_back(h0_log_lambda);
                }

                // ── Push to smoothing buffer and apply fixed-lag smoothing ──
                if (config.lag_steps > 0) {
                    double raw_pl = (next_real < static_cast<int>(traj.protection_levels.size()))
                        ? traj.protection_levels[next_real] : 0.0;
                    double effective_pl = raw_pl * config.phmi_pl_multiplier;
                    int n_inside_pl = 0, n_total = 0;
                    for (size_t b = 0; b < next_layer.size(); ++b) {
                        TGNode& nb = next_layer[b];
                        if (nb.c == nullptr) continue;
                        n_total++;
                        if (nb.c->dist <= effective_pl) n_inside_pl++;
                    }
                    double frac_inside =
                        (n_total > 0) ? static_cast<double>(n_inside_pl) / n_total : 1.0;

                    lag_buffer.back().tp_to_next = std::move(tp_raw_smoothing);
                    lag_buffer.back().frac_inside_pl = frac_inside;
                    lag_buffer.push_back({last_valid_layer, {}, 1.0});

                    if (lag_buffer.size() > static_cast<size_t>(config.lag_steps)) {
                        apply_lag_smoothing(lag_buffer);
                        lag_buffer.pop_front();
                    }
                }
            }
        }

        // ── Trajectory end: flush remaining smoothing buffer ──
        flush_lag_buffer(lag_buffer, *this, config.lag_steps);

        if (!current_sub_indices.empty()) {
            process_sub_segment(tg_ptr.get(), current_sub_indices, &h0_log_lambdas);
        }
        for (int skipped_idx : skipped_indices) {
            final_results.push_back(create_fallback_result(skipped_idx, skipped_idx, MatchStatus::FAILED_DISCONNECTED));
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
        // Offset decreased on the same edge.
        // First guard: small decreases (up to 15% of edge length) are treated as
        // projection artifacts rather than reverse travel. Perpendicular projection
        // near polyline vertices can produce offset oscillations even when the
        // vehicle moves forward. This threshold is independent of reverse_tolerance.
        double offset_diff = ca->offset - cb->offset;
        double pct = offset_diff / ca->edge->length * 100.0;
        if (offset_diff <= ca->edge->length * 0.15) {
            return 0.0;
        }
        // Second guard: if reverse_tolerance is configured, allow additional reverse.
        if (reverse_tolerance > 0) {
            double reverse_limit = ca->edge->length * reverse_tolerance + 1e-7;
            if (offset_diff <= reverse_limit) {
                return 0.0;
            }
        }
        // Offset decreased beyond both guards — genuine reverse travel.
        // Check if U-turn / reverse travel on this edge is possible:
        // look up the reverse path (target → source) in UBODT.
        Record *r_reverse = nullptr;
        #pragma omp critical(ubodt_section)
        {
            r_reverse = ubodt_->look_up(ca->edge->target, ca->edge->source);
        }
        if (r_reverse != nullptr) {
            return ca->offset - cb->offset;
        }
        return -1;
    }

    // If edges are directly connected (target == source), no UBODT lookup needed.
    if (ca->edge->target == cb->edge->source) {
        return ca->edge->length - ca->offset + cb->offset;
    }

    // Otherwise rely on UBODT lookup for forward path between successive edges.
    NodeIndex s = ca->edge->target;
    NodeIndex e = cb->edge->source;
    Record *r = nullptr;
    #pragma omp critical(ubodt_section)
    {
        r = ubodt_->look_up(s, e);
    }
    double sp_dist = r ? r->cost : -1;

    if (sp_dist < 0) {
        return -1;
    } else {
        // Path exists in forward direction
        double dist = ca->edge->length - ca->offset + cb->offset;
        return dist + sp_dist;
    }
}

void CovarianceMapMatch::initialize_first_layer(TGLayer *layer, const CovarianceMapMatchConfig &config, double &log_prob_unconsidered) {
    if (!layer || layer->empty()) return;

    std::vector<double> layer_log_probs;

    // 1. 初始化首层的 cumu_prob 和 forward_cumu
    //    δ₁(i) = π(i) · e₁(i)  with π(i) = 1/K (uniform prior over real road candidates)
    //    α₁(i) = π(i) · e₁(i)  (forward = same as Viterbi at first layer)
    //    Background pseudo-candidate (c==nullptr) gets no π multiplier — its prior
    //    is embedded in its emission probability.
    size_t num_real = 0;
    for (const auto &node : *layer) {
        if (node.c != nullptr && node.ep > -std::numeric_limits<double>::infinity())
            ++num_real;
    }
    double log_uniform_prior = (num_real > 0)
        ? -std::log(static_cast<double>(num_real)) : 0.0;

    for (size_t i = 0; i < layer->size(); ++i) {
        auto &node = (*layer)[i];
        if (node.ep > -std::numeric_limits<double>::infinity()) {
            if (node.c != nullptr) {
                // Real road candidate: δ₁(i) = log(1/K) + log(e₁(i))
                node.cumu_prob = log_uniform_prior + node.ep;
                node.forward_cumu = log_uniform_prior + node.ep;
            } else {
                // Background pseudo-candidate: no π multiplier
                node.cumu_prob = node.ep;
                node.forward_cumu = node.ep;
            }
            layer_log_probs.push_back(node.forward_cumu);
        } else {
            node.cumu_prob = -std::numeric_limits<double>::infinity();
            node.forward_cumu = -std::numeric_limits<double>::infinity();
        }
        node.tp = 1.0;
        node.prev = nullptr;
    }

    // 2. 计算逐节点的过滤后验概率 (filtering posterior) 和首层后验熵
    //    filtering posterior: p_i = exp(cumu_prob_i) / sum_j exp(cumu_prob_j)
    //    即 P(state_t=i | obs_{1:t}) 的归一化概率 ∈ [0,1]
    double layer_entropy = 0.0;
    if (!layer_log_probs.empty()) {
        double log_sum = log_sum_exp(layer_log_probs);
        double inv_log2 = 1.0 / std::log(2.0);

        for (size_t i = 0; i < layer->size(); ++i) {
            auto &node = (*layer)[i];
            if (node.cumu_prob > -std::numeric_limits<double>::infinity()) {
                double log_norm = node.forward_cumu - log_sum;
                double p_norm = std::exp(log_norm);
                node.trustworthiness = p_norm;  // filtering posterior
                if (p_norm > 0.0) {
                    layer_entropy -= p_norm * log_norm * inv_log2;
                }
            } else {
                node.trustworthiness = 0.0;
            }
        }
        if (layer_entropy < 0.0) layer_entropy = 0.0;
    }

    // 3. 计算首层的信息增益 (delta_entropy = H_prior - H_posterior)
    //    首层没有先验层传播，使用均匀先验: H_prior = log2(N_valid)
    double delta_layer_entropy = 0.0;
    if (!layer_log_probs.empty()) {
        double H_prior = std::log2(static_cast<double>(layer_log_probs.size()));
        delta_layer_entropy = H_prior - layer_entropy;
        if (delta_layer_entropy < 0.0) delta_layer_entropy = 0.0;
    }

    // 4. 赋值 posterior_entropy 与 delta_entropy（trustworthiness 已在步骤2中逐个赋值）
    for (size_t i = 0; i < layer->size(); ++i) {
        auto &node = (*layer)[i];
        if (node.cumu_prob > -std::numeric_limits<double>::infinity()) {
            node.posterior_entropy = layer_entropy;
            node.delta_entropy = delta_layer_entropy;
        } else {
            node.posterior_entropy = -std::numeric_limits<double>::infinity();
            node.delta_entropy = -std::numeric_limits<double>::infinity();
        }
    }

    // 初始化未考虑状态概率
    log_prob_unconsidered = std::log(config.phmi);
}

// ... update_tg_cmm remains same ...

void CovarianceMapMatch::update_layer_cmm(TGLayer *la_ptr, TGLayer *lb_ptr,
                                          double eu_dist, bool *connected,
                                          const CovarianceMapMatchConfig &config,
                                          double &log_prob_unconsidered,
                                          std::vector<std::vector<double>> *tp_raw_out) {
    if (!la_ptr || !lb_ptr) return;
    *connected = false;
    const size_t prev_count = la_ptr->size();
    const size_t next_count = lb_ptr->size();
    if (next_count == 0) return;

    double phmi = config.phmi;
    double one_minus_phmi = 1.0 - phmi;
    double log_phmi = std::log(phmi);

    std::vector<double> leak_probs;
    leak_probs.push_back(log_prob_unconsidered); // 收集之前所有历元遗留的漏检概率

    // 1. 预先计算 A 到 B 的单步原始 TP、SP 距离及总和
    std::vector<double> sum_tp_raw_A(prev_count, 0.0);
    std::vector<std::vector<double>> tp_raw_matrix(prev_count, std::vector<double>(next_count, 0.0));
    std::vector<std::vector<double>> sp_dist_matrix(prev_count, std::vector<double>(next_count, -1.0));
    std::vector<std::vector<double>> rev_dist_matrix(prev_count, std::vector<double>(next_count, 0.0));

    for (size_t a = 0; a < prev_count; ++a) {
        TGNode &node_a = (*la_ptr)[a];
        if (node_a.cumu_prob == -std::numeric_limits<double>::infinity()) continue;

        for (size_t b = 0; b < next_count; ++b) {
            if ((*lb_ptr)[b].c == nullptr) continue;
            const Candidate *ca = node_a.c;
            const Candidate *cb = (*lb_ptr)[b].c;
            double sp_dist = get_sp_dist(ca, cb, config.reverse_tolerance);

            // ── Cumulative reverse travel guard ──
            // Same-edge offset regression: accumulate across consecutive epochs.
            // Uses min(30m, 15% of edge length) as threshold — projection noise
            // at vertices is <5m, while 30m is clearly real reverse travel.
            // The 15% cap handles very short edges (<200m).
            double cumul_rev = 0.0;
            if (sp_dist >= 0 && ca->edge != nullptr && cb->edge != nullptr &&
                ca->edge->id == cb->edge->id && ca->offset > cb->offset) {
                double step_rev = ca->offset - cb->offset;
                cumul_rev = node_a.reverse_dist + step_rev;
                double max_reverse = std::min(30.0, ca->edge->length * 0.15);
                if (cumul_rev > max_reverse) {
                    sp_dist = -1.0;  // cumulative reverse exceeds threshold → block
                }
            }

            if (sp_dist >= 0) {
                double tp_raw = TransitionGraph::calc_tp(sp_dist, eu_dist);
                tp_raw_matrix[a][b] = tp_raw;
                sp_dist_matrix[a][b] = sp_dist;
                rev_dist_matrix[a][b] = cumul_rev;
                sum_tp_raw_A[a] += tp_raw;
            }
        }

        if (sum_tp_raw_A[a] > 0) {
            // A 节点有合法的转移路径，正常漏检概率为 PHMI
            leak_probs.push_back(node_a.cumu_prob + log_phmi);
        } else {
            // A 节点前方没有候选点（死胡同），该分支概率 100% 漏检
            leak_probs.push_back(node_a.cumu_prob);
        }
    }

    // 2. 更新全局未考虑状态概率 (前向求和)
    log_prob_unconsidered = log_sum_exp(leak_probs);

    bool has_valid_candidate = false;

    // 用于收集先验对数概率 (预测分布，尚未乘 EP)
    std::vector<double> layer_prior_log_probs;

    // 3. 对当前历元的每个候选点 B 进行全概率合并
    for (size_t b = 0; b < next_count; ++b) {
        TGNode &node_b = (*lb_ptr)[b];
        if (node_b.c == nullptr) continue;

        std::vector<double> incoming_log_probs; // 收集所有指向 node_b 的分支概率
        TGNode *best_prev = nullptr;
        double best_log_branch_prob = -std::numeric_limits<double>::infinity();
        double best_log_tp_norm = -std::numeric_limits<double>::infinity();
        double best_sp_dist = 0.0;
        double best_rev_dist = 0.0;

        for (size_t a = 0; a < prev_count; ++a) {
            TGNode &node_a = (*la_ptr)[a];
            if (node_a.cumu_prob == -std::numeric_limits<double>::infinity()) continue;

            double tp_raw = tp_raw_matrix[a][b];
            double sum_tp_raw = sum_tp_raw_A[a];

            if (tp_raw > 0 && sum_tp_raw > 0) {
                // Row-normalize: for source a, Σ_j a(a,j) = 1
                double linear_tp_norm = tp_raw / sum_tp_raw;
                double log_tp_norm = std::log(linear_tp_norm);

                // 在对数空间内安全相加
                double log_branch_prob = node_a.cumu_prob + log_tp_norm;

                incoming_log_probs.push_back(log_branch_prob);

                // 记录 Viterbi 最优前驱，用于 backtrack
                if (log_branch_prob > best_log_branch_prob) {
                    best_log_branch_prob = log_branch_prob;
                    best_prev = &node_a;
                    best_log_tp_norm = log_tp_norm;
                    best_sp_dist = sp_dist_matrix[a][b];
                    best_rev_dist = rev_dist_matrix[a][b];
                }
            }
        }

        // 4. 计算最终的 Cumu_Prob (结合已归一化到 Log 空间的 EP)
        if (!incoming_log_probs.empty() && node_b.ep > -std::numeric_limits<double>::infinity()) {
            double log_sum_prev_probs = log_sum_exp(incoming_log_probs);
            layer_prior_log_probs.push_back(log_sum_prev_probs);  // 记录先验

            // Viterbi max: single best path score (for backtrack / optimal path)
            node_b.cumu_prob = best_log_branch_prob + node_b.ep;
            // Forward sum: marginal over all incoming paths (for posterior / trust)
            node_b.forward_cumu = log_sum_prev_probs + node_b.ep;

            if (best_prev != nullptr) {
                node_b.prev = best_prev;
                node_b.tp = std::exp(best_log_tp_norm);
                node_b.sp_dist = best_sp_dist;
                node_b.reverse_dist = best_rev_dist;
            }
        } else {
            node_b.cumu_prob = -std::numeric_limits<double>::infinity();
            node_b.forward_cumu = -std::numeric_limits<double>::infinity();
        }
    }

    // 5. 收集当前历元所有有效候选点的对数概率，计算局部信息熵
    std::vector<double> layer_log_probs;
    for (size_t b = 0; b < next_count; ++b) {
        TGNode &node_b = (*lb_ptr)[b];
        if (node_b.forward_cumu > -std::numeric_limits<double>::infinity()) {
            layer_log_probs.push_back(node_b.forward_cumu);
        }
    }

    // 计算逐节点的过滤后验概率 (filtering posterior) 和本层后验熵
    double layer_entropy = 0.0;
    if (!layer_log_probs.empty()) {
        double log_sum = log_sum_exp(layer_log_probs);
        double inv_log2 = 1.0 / std::log(2.0);

        for (size_t b = 0; b < next_count; ++b) {
            TGNode &node_b = (*lb_ptr)[b];
            if (node_b.forward_cumu > -std::numeric_limits<double>::infinity()) {
                double log_norm = node_b.forward_cumu - log_sum;
                double p_norm = std::exp(log_norm);
                node_b.trustworthiness = p_norm;  // filtering posterior
                if (p_norm > 0.0) {
                    layer_entropy -= p_norm * log_norm * inv_log2;
                }
            } else {
                node_b.trustworthiness = 0.0;
            }
        }
        if (layer_entropy < 0.0) layer_entropy = 0.0;
    }

    // 5a. 计算先验分布熵: H(P(x_t | z_{1:t-1}))
    double layer_prior_entropy = 0.0;
    if (!layer_prior_log_probs.empty()) {
        double log_sum_prior = log_sum_exp(layer_prior_log_probs);
        double inv_log2 = 1.0 / std::log(2.0);
        for (double log_p : layer_prior_log_probs) {
            double log_norm = log_p - log_sum_prior;
            double p_norm = std::exp(log_norm);
            if (p_norm > 0.0) {
                layer_prior_entropy -= p_norm * log_norm * inv_log2;
            }
        }
        if (layer_prior_entropy < 0.0) layer_prior_entropy = 0.0;
    }

    // 5b. 信息增益: ΔH = H_prior - H_posterior
    double delta_layer_entropy = layer_prior_entropy - layer_entropy;
    if (delta_layer_entropy < 0.0) delta_layer_entropy = 0.0;

    // 6. 赋值 posterior_entropy 与 delta_entropy（trustworthiness 已在循环中逐个赋值）
    has_valid_candidate = false;
    for (size_t b = 0; b < next_count; ++b) {
        TGNode &node_b = (*lb_ptr)[b];
        if (node_b.forward_cumu > -std::numeric_limits<double>::infinity()) {
            node_b.posterior_entropy = layer_entropy;
            node_b.delta_entropy = delta_layer_entropy;
            has_valid_candidate = true;
        } else {
            node_b.posterior_entropy = -std::numeric_limits<double>::infinity();
            node_b.delta_entropy = -std::numeric_limits<double>::infinity();
        }
    }

    *connected = has_valid_candidate;

    // 7. Optionally export the raw transition probability matrix for fixed-lag smoothing.
    //    tp_raw_out[a][b] stores the linear transition probability from candidate a to b.
    //    Used by apply_lag_smoothing() to re-evaluate earlier-layer posteriors.
    if (tp_raw_out != nullptr && has_valid_candidate) {
        *tp_raw_out = std::move(tp_raw_matrix);
    }
}

// ── Fixed-lag smoothing with sequential Bayesian H0 test ─────────────────────
// Phases:
//   1. Viterbi-style forward pass through L-step window for future evidence.
//   2. Softmax normalization of smoothed cumu_prob for per-candidate trust.
//   3. Sequential Bayesian H0 test: accumulates likelihood ratios across the
//      smoothing window and applies multiplicative discount α = P(H0|z_{1:t}),
//      which cannot be canceled by softmax normalization.
//
// H0 test formulation:
//   λ_t = λ₀ · Π_τ LR_τ,    where LR_τ = frac_inside / max(P_HMI, 1-frac_inside)
//   α_t = λ_t / (1 + λ_t)   (posterior probability of H0)
//   trust'[i] = α_t × trust[i]
//
// When all epochs consistently show candidates outside PL (frac_inside → 0),
// λ_t decays exponentially, α_t → 0, and trustworthiness → 0 regardless of
// the softmax-normalized ranking.  This detects map errors and off-road driving.

void CovarianceMapMatch::apply_lag_smoothing(
    std::deque<LagEntry>& lag_data) const {

    size_t L = lag_data.size() - 1;
    if (L == 0) return;

    LagEntry& oldest_entry = lag_data[0];
    TGLayer* oldest_layer = oldest_entry.layer;
    const size_t n_old = oldest_layer->size();
    std::vector<double> smoothed_log_probs(n_old,
        -std::numeric_limits<double>::infinity());
    const double neg_inf = -std::numeric_limits<double>::infinity();

    // ── Part 1: Viterbi-style forward pass through L-step window ──
    for (size_t i = 0; i < n_old; ++i) {
        TGNode& start_node = (*oldest_layer)[i];
        if (start_node.cumu_prob <= neg_inf) continue;

        // Step 1: candidate i → layer[1]
        const auto& tp01 = oldest_entry.tp_to_next;
        TGLayer* l1 = lag_data[1].layer;
        size_t n1 = l1->size();
        std::vector<double> dp_cur(n1, neg_inf);

        for (size_t j = 0; j < n1; ++j) {
            TGNode& node_j = (*l1)[j];
            if (node_j.cumu_prob <= neg_inf) continue;
            if (i < tp01.size() && j < tp01[i].size() && tp01[i][j] > 0) {
                dp_cur[j] = std::log(tp01[i][j]) + node_j.ep;
            }
        }

        // Steps 2..L
        for (size_t step = 2; step <= L; ++step) {
            TGLayer* nl = lag_data[step].layer;
            const auto& tp = lag_data[step - 1].tp_to_next;
            size_t nn = nl->size();
            std::vector<double> dp_next(nn, neg_inf);

            for (size_t a = 0; a < dp_cur.size(); ++a) {
                if (dp_cur[a] <= neg_inf) continue;
                for (size_t b = 0; b < nn; ++b) {
                    TGNode& node_b = (*nl)[b];
                    if (node_b.cumu_prob <= neg_inf) continue;
                    if (a < tp.size() && b < tp[a].size() && tp[a][b] > 0) {
                        double val = dp_cur[a] + std::log(tp[a][b]) + node_b.ep;
                        if (val > dp_next[b]) dp_next[b] = val;
                    }
                }
            }
            dp_cur = std::move(dp_next);
        }

        double best_cont = neg_inf;
        for (double v : dp_cur) {
            if (v > best_cont) best_cont = v;
        }
        if (best_cont > neg_inf) {
            smoothed_log_probs[i] = start_node.cumu_prob + best_cont;
        }
    }

    // ── Part 2: Softmax normalization ──
    double log_sum = log_sum_exp(smoothed_log_probs);
    double inv_log2 = 1.0 / std::log(2.0);
    double layer_entropy = 0.0;
    int valid_count = 0;
    for (size_t i = 0; i < n_old; ++i) {
        if (smoothed_log_probs[i] > neg_inf) valid_count++;
    }

    for (size_t i = 0; i < n_old; ++i) {
        TGNode& node = (*oldest_layer)[i];
        if (smoothed_log_probs[i] > neg_inf) {
            double log_norm = smoothed_log_probs[i] - log_sum;
            double p_norm = std::exp(log_norm);
            node.trustworthiness = p_norm;
            if (p_norm > 0.0) {
                layer_entropy -= p_norm * log_norm * inv_log2;
            }
        } else {
            node.trustworthiness = 0.0;
        }
    }
    if (layer_entropy < 0.0) layer_entropy = 0.0;

    // ── Part 3: Recompute entropy and delta ──
    double delta_entropy_update = 0.0;
    if (valid_count > 0) {
        double H_prior = std::log2(static_cast<double>(valid_count));
        delta_entropy_update = H_prior - layer_entropy;
        if (delta_entropy_update < 0.0) delta_entropy_update = 0.0;
    }
    for (size_t i = 0; i < n_old; ++i) {
        TGNode& node = (*oldest_layer)[i];
        if (node.cumu_prob > neg_inf) {
            node.posterior_entropy = layer_entropy;
            node.delta_entropy = delta_entropy_update;
        }
    }

    SPDLOG_DEBUG("Lag-smoothing: L={} valid={} entropy={:.4f} delta={:.4f}",
                 static_cast<int>(L), valid_count, layer_entropy, delta_entropy_update);
}

// ── Smoothing-buffer flush (trajectory end / sub-trajectory boundary) ────────
void CovarianceMapMatch::flush_lag_buffer(
    std::deque<LagEntry>& lag_data,
    const CovarianceMapMatch& cmm,
    int lag_steps)
{
    if (lag_steps <= 0) return;
    while (lag_data.size() > 1) {
        cmm.apply_lag_smoothing(lag_data);
        lag_data.pop_front();
    }
    lag_data.clear();
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
            // Correctly pass input_epsg instead of need_reprojection
            trajectories_reprojected = maybe_reproject_trajectories(&trajectories, network_, input_epsg);
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

    const int trajectories_count = static_cast<int>(trajectories.size());

    #pragma omp parallel for if(use_omp) schedule(dynamic)
    for (int i = 0; i < trajectories_count; ++i) {
        const CMMTrajectory &trajectory = trajectories[i];
        CMMTrajectory filtered_traj_base;
        std::vector<MM::MatchResult> results = match_traj(trajectory, cmm_config, &filtered_traj_base);
        
        for (auto &result : results) {
            if (result.original_indices.empty()) continue;
            
            // CRITICAL: Do NOT slice the trajectory. 
            // The writer uses result.original_indices to align fields to the original GPS points.
            // If we slice it, result.original_indices (which contains absolute indices) 
            // will cause out-of-bounds access.
            CORE::Trajectory full_traj;
            full_traj.id = result.id;
            full_traj.geom = trajectory.geom;
            full_traj.timestamps = trajectory.timestamps;
            
            // Protect coordinate transformation and file writing
            #pragma omp critical(writer_section)
            {
                apply_output_transform(&full_traj, &result);
                writer.write_result(full_traj, result);
                points_matched += result.original_indices.size();
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
                SPDLOG_INFO("Progress {}", progress);
            }
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
