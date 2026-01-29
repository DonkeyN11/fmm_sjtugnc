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
    double uniform = 1.0 / positive_count;
    for (auto &node : *layer) {
        node.trustworthiness = (node.ep > 0) ? uniform : 0.0;
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
                                                   double reverse_tolerance,
                                                   bool normalized_arg,
                                                   bool use_mahalanobis_candidates_arg,
                                                   int window_length_arg,
                                                   bool margin_used_trustworthiness_arg,
                                                   bool filtered_arg)
    : k(k_arg), min_candidates(min_candidates_arg),
      protection_level_multiplier(protection_level_multiplier_arg),
      reverse_tolerance(reverse_tolerance),
      normalized(normalized_arg),
      use_mahalanobis_candidates(use_mahalanobis_candidates_arg),
      window_length(window_length_arg),
      margin_used_trustworthiness(margin_used_trustworthiness_arg),
      filtered(filtered_arg) {
}

// Dump runtime configuration for debugging or reproducibility.
void CovarianceMapMatchConfig::print() const {
    SPDLOG_INFO("CMMAlgorithmConfig");
    SPDLOG_INFO("k {} min_candidates {} protection_level_multiplier {} reverse_tolerance {} normalized {} use_mahalanobis {} window_length {} margin_used_trustworthiness {} filtered {}",
                k, min_candidates, protection_level_multiplier, reverse_tolerance, normalized, use_mahalanobis_candidates, window_length, margin_used_trustworthiness, filtered);
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
    return CovarianceMapMatchConfig{k, min_candidates, protection_level_multiplier, reverse_tolerance, normalized, use_mahalanobis_candidates, window_length, margin_used_trustworthiness, filtered};
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
    return CovarianceMapMatchConfig{k, min_candidates, protection_level_multiplier, reverse_tolerance, normalized, use_mahalanobis_candidates, window_length, margin_used_trustworthiness, filtered};
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
         cxxopts::value<bool>()->default_value("true"));
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
}

// Quick sanity checks to guard against invalid user supplied parameters.
bool CovarianceMapMatchConfig::validate() const {
    if (k <= 0 || min_candidates <= 0 || min_candidates > k ||
        protection_level_multiplier <= 0 || reverse_tolerance < 0 || reverse_tolerance > 1 ||
        window_length <= 0) {
        SPDLOG_CRITICAL("Invalid CMM parameter k {} min_candidates {} "
                       "protection_level_multiplier {} reverse_tolerance {} window_length {}",
                       k, min_candidates, protection_level_multiplier, reverse_tolerance, window_length);
        return false;
    }
    return true;
}

// Implementation of CovarianceMapMatch
// Evaluate emission probabilities by respecting each observation's covariance model.
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

// Enumerate candidate projections per point by respecting both covariance ellipses
// and the provided protection levels that limit how far points can deviate.
CandidateSearchResult CovarianceMapMatch::search_candidates_with_protection_level(
    const CORE::LineString &geom,
    const std::vector<CovarianceMatrix> &covariances,
    const std::vector<double> &protection_levels,
    const CovarianceMapMatchConfig &config,
    const std::string &traj_id) const {

    SPDLOG_DEBUG("Search candidates with protection level for {} points", geom.get_num_points());

    CandidateSearchResult result;
    int num_points = geom.get_num_points();
    result.candidates.reserve(num_points);
    result.emission_probabilities.reserve(num_points);
    NETWORK::NodeIndex next_candidate_index = network_.get_node_count();

    for (int i = 0; i < num_points; ++i) {
        CORE::Point point = geom.get_point(i);
        const CovarianceMatrix &cov = covariances[i];
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
                    double probability = 0.0;
                    if (valid_covariance) {
                        double dx = obs_x - boost::geometry::get<0>(entry.candidate.point);
                        double dy = obs_y - boost::geometry::get<1>(entry.candidate.point);
                        double mahal_sq = compute_mahalanobis_sq(cov_inv, dx, dy);
                        double normalization = 1.0 / (2.0 * M_PI * std::sqrt(det));
                        probability = normalization * std::exp(-0.5 * mahal_sq);
                    }
                    raw_probabilities.push_back(probability);
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
            Traj_Candidates traj_candidates = network_.search_tr_cs_knn(single_point_geom, config.k, search_radius);
            Point_Candidates point_candidates = traj_candidates.empty() ? Point_Candidates() : traj_candidates[0];

            selected_candidates.reserve(point_candidates.size());
            raw_probabilities.reserve(point_candidates.size());

            for (const Candidate &cand : point_candidates) {
                double cand_x = boost::geometry::get<0>(cand.point);
                double cand_y = boost::geometry::get<1>(cand.point);
                double dx = obs_x - cand_x;
                double dy = obs_y - cand_y;

                double probability = 0.0;
                if (valid_covariance) {
                    double mahalanobis_dist_sq = compute_mahalanobis_sq(cov_inv, dx, dy);
                    double normalization = 1.0 / (2.0 * M_PI * std::sqrt(det));
                    probability = normalization * std::exp(-0.5 * mahalanobis_dist_sq);
                }

                selected_candidates.push_back(cand);
                raw_probabilities.push_back(probability);
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
        }

        std::vector<double> emission_probs;
        if (config.normalized) {
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
                if (!valid_covariance) {
                    SPDLOG_WARN("Trajectory {} Point {}: covariance determinant non-positive, using uniform emission", traj_id, i);
                } else {
                    SPDLOG_WARN("Trajectory {} Point {}: no valid candidates within PL, using uniform emission", traj_id, i);
                }
            }
            emission_probs = std::move(normalized_probabilities);
        } else {
            emission_probs = raw_probabilities;
        }

        SPDLOG_TRACE("Point {}: {} candidates kept", i, selected_candidates.size());
        result.candidates.push_back(std::move(selected_candidates));
        result.emission_probabilities.push_back(std::move(emission_probs));
    }

    SPDLOG_DEBUG("Candidate search completed");
    return result;
}

// Execute the full map-matching pipeline for a single trajectory using covariance-aware search.
MatchResult CovarianceMapMatch::match_traj(const CMMTrajectory &traj,
                                          const CovarianceMapMatchConfig &config,
                                          CMMTrajectory *filtered_traj) {
    SPDLOG_DEBUG("Count of points in trajectory {}", traj.geom.get_num_points());
    SPDLOG_INFO("Trajectory {}: filtered mode is {}", traj.id, config.filtered ? "enabled" : "disabled");

    // Validate trajectory
    if (!traj.is_valid()) {
        SPDLOG_ERROR("Invalid trajectory: covariance and protection level data mismatch");
        return MatchResult{};
    }

    SPDLOG_DEBUG("Search candidates with protection level");
    CandidateSearchResult candidate_result = search_candidates_with_protection_level(
        traj.geom, traj.covariances, traj.protection_levels, config, std::to_string(traj.id));

    Traj_Candidates candidates = std::move(candidate_result.candidates);
    std::vector<std::vector<double>> emission_probabilities = std::move(candidate_result.emission_probabilities);

    SPDLOG_DEBUG("Trajectory candidate {}", candidates);
    if (candidates.empty()) {
        if (filtered_traj != nullptr) {
            *filtered_traj = CMMTrajectory{};
            filtered_traj->id = traj.id;
        }
        return MatchResult{};
    }

    // Prepare working trajectory and candidates
    Traj_Candidates working_candidates;
    std::vector<std::vector<double>> working_emissions;
    CMMTrajectory working_traj;
    std::vector<int> working_original_indices;

    const bool has_timestamps = !traj.timestamps.empty();
    working_traj.id = traj.id;
    working_traj.covariances = traj.covariances;
    working_traj.protection_levels = traj.protection_levels;
    if (has_timestamps) {
        working_traj.timestamps = traj.timestamps;
    }

    if (config.filtered) {
        // Filtering enabled: remove points with no candidates and disconnected transitions
        std::vector<CORE::Point> filtered_points;
        std::vector<double> filtered_timestamps;
        std::vector<CovarianceMatrix> filtered_covariances;
        std::vector<double> filtered_protection_levels;
        std::vector<int> filtered_original_indices;
        const size_t total_points = static_cast<size_t>(traj.geom.get_num_points());

        filtered_points.reserve(total_points);
        filtered_covariances.reserve(total_points);
        filtered_protection_levels.reserve(total_points);
        working_candidates.reserve(total_points);
        working_emissions.reserve(total_points);
        filtered_original_indices.reserve(total_points);
        if (has_timestamps) {
            filtered_timestamps.reserve(total_points);
        }

        // Step 1: Remove points with no candidates
        for (size_t idx = 0; idx < candidates.size(); ++idx) {
            if (candidates[idx].empty()) {
                continue;
            }
            filtered_points.push_back(traj.geom.get_point(static_cast<int>(idx)));
            if (has_timestamps) {
                filtered_timestamps.push_back(traj.timestamps[idx]);
            }
            filtered_covariances.push_back(traj.covariances[idx]);
            filtered_protection_levels.push_back(traj.protection_levels[idx]);
            working_candidates.push_back(std::move(candidates[idx]));
            if (idx < emission_probabilities.size()) {
                working_emissions.push_back(std::move(emission_probabilities[idx]));
            } else {
                working_emissions.emplace_back();
            }
            filtered_original_indices.push_back(static_cast<int>(idx));
        }

        size_t removed_empty = total_points - filtered_points.size();
        size_t removed_disconnected = 0;

        // Step 2: Remove points with no valid transitions
        auto has_valid_transition = [&](size_t prev_idx, size_t next_idx) -> bool {
            if (prev_idx >= working_candidates.size() || next_idx >= working_candidates.size()) {
                return false;
            }
            const Point_Candidates &prev_candidates = working_candidates[prev_idx];
            const Point_Candidates &next_candidates = working_candidates[next_idx];
            if (prev_candidates.empty() || next_candidates.empty()) {
                return false;
            }

            const auto *prev_eps = (prev_idx < working_emissions.size())
                                   ? &working_emissions[prev_idx]
                                   : nullptr;
            const auto *next_eps = (next_idx < working_emissions.size())
                                   ? &working_emissions[next_idx]
                                   : nullptr;

            const CORE::Point &point_prev = filtered_points[prev_idx];
            const CORE::Point &point_next = filtered_points[next_idx];
            double eu_dist = boost::geometry::distance(point_prev, point_next);

            for (size_t a = 0; a < prev_candidates.size(); ++a) {
                double ep_a = (prev_eps && a < prev_eps->size()) ? (*prev_eps)[a] : 0.0;
                if (ep_a <= 0.0 || !std::isfinite(ep_a)) {
                    continue;
                }
                for (size_t b = 0; b < next_candidates.size(); ++b) {
                    double ep_b = (next_eps && b < next_eps->size()) ? (*next_eps)[b] : 0.0;
                    if (ep_b <= 0.0 || !std::isfinite(ep_b)) {
                        continue;
                    }
                    double sp_dist = get_sp_dist(&prev_candidates[a], &next_candidates[b], config.reverse_tolerance);
                    if (sp_dist < 0.0) {
                        continue;
                    }
                    double tp = TransitionGraph::calc_tp(sp_dist, eu_dist);
                    if (tp <= 0.0 || !std::isfinite(tp)) {
                        continue;
                    }
                    return true;
                }
            }
            return false;
        };

        bool removed = true;
        while (removed && working_candidates.size() > 1) {
            removed = false;
            for (size_t idx = 0; idx + 1 < working_candidates.size(); ) {
                if (!has_valid_transition(idx, idx + 1)) {
                    auto erase_offset = static_cast<std::ptrdiff_t>(idx + 1);
                    working_candidates.erase(working_candidates.begin() + erase_offset);
                    working_emissions.erase(working_emissions.begin() + erase_offset);
                    filtered_points.erase(filtered_points.begin() + erase_offset);
                    filtered_covariances.erase(filtered_covariances.begin() + erase_offset);
                    filtered_protection_levels.erase(filtered_protection_levels.begin() + erase_offset);
                    filtered_original_indices.erase(filtered_original_indices.begin() + erase_offset);
                    if (has_timestamps) {
                        filtered_timestamps.erase(filtered_timestamps.begin() + erase_offset);
                    }
                    ++removed_disconnected;
                    removed = true;
                    continue;
                }
                ++idx;
            }
        }

        if (removed_empty > 0 || removed_disconnected > 0) {
            SPDLOG_INFO("Trajectory {}: skipped {} empty epochs and {} disconnected epochs",
                        traj.id, removed_empty, removed_disconnected);
        }

        // Build working trajectory from filtered data
        for (const auto &point : filtered_points) {
            working_traj.geom.add_point(point);
        }
        if (has_timestamps) {
            working_traj.timestamps = std::move(filtered_timestamps);
        }
        working_traj.covariances = std::move(filtered_covariances);
        working_traj.protection_levels = std::move(filtered_protection_levels);
        working_original_indices = std::move(filtered_original_indices);
    } else {
        // Filtering disabled: use all points directly
        const auto &points = traj.geom.get_geometry_const();
        for (const auto &point : points) {
            working_traj.geom.add_point(point);
        }
        working_candidates = std::move(candidates);
        working_emissions = std::move(emission_probabilities);
        // Generate original indices (identity mapping when no filtering)
        working_original_indices.reserve(static_cast<size_t>(traj.geom.get_num_points()));
        for (int i = 0; i < traj.geom.get_num_points(); ++i) {
            working_original_indices.push_back(i);
        }
    }

    if (filtered_traj != nullptr) {
        *filtered_traj = working_traj;
    }

    const Traj_Candidates &tc = working_candidates;
    const std::vector<std::vector<double>> &emission_probs = working_emissions;

    if (tc.empty()) {
        return MatchResult{};
    }

    // Store lightweight emission info for optional debug output or Python bindings.
    std::vector<std::vector<CandidateEmission>> candidate_details(tc.size());
    for (size_t idx = 0; idx < tc.size(); ++idx) {
        const Point_Candidates &cand_list = tc[idx];
        const std::vector<double> *prob_list = (idx < emission_probs.size())
                                               ? &emission_probs[idx]
                                               : nullptr;
        candidate_details[idx].reserve(cand_list.size());
        for (size_t j = 0; j < cand_list.size(); ++j) {
            CandidateEmission ce;
            ce.x = boost::geometry::get<0>(cand_list[j].point);
            ce.y = boost::geometry::get<1>(cand_list[j].point);
            ce.ep = (prob_list && j < prob_list->size()) ? (*prob_list)[j] : 0.0;
            candidate_details[idx].push_back(ce);
        }
    }

    SPDLOG_DEBUG("Generate transition graph");
    TransitionGraph tg(tc, emission_probs);

    // Populate transition costs and trustworthiness using the covariance-aware routine.
    SPDLOG_DEBUG("Update cost in transition graph using CMM");
    update_tg_cmm(&tg, working_traj, config);

    auto trustworthiness_results = compute_window_trustworthiness(
        tc, emission_probs, working_traj, config);
    const std::vector<double> &trust_margins = trustworthiness_results.first;
    std::vector<std::vector<double>> n_best_trust = std::move(trustworthiness_results.second);

    SPDLOG_DEBUG("Optimal path inference");
    TGOpath tg_opath = tg.backtrack();
    SPDLOG_DEBUG("Optimal path size {}", tg_opath.size());

    MatchedCandidatePath matched_candidate_path;
    matched_candidate_path.reserve(tg_opath.size());
    std::vector<double> sp_distances;
    std::vector<double> eu_distances;
    sp_distances.reserve(tg_opath.size());
    eu_distances.reserve(tg_opath.size());
    for (size_t idx = 0; idx < tg_opath.size(); ++idx) {
        const TGNode *a = tg_opath[idx];
        double sp_dist_value = -1.0;
        double eu_dist_value = -1.0;
        if (idx == 0) {
            sp_dist_value = 0.0;
            eu_dist_value = 0.0;
        } else if (a->prev != nullptr) {
            if (std::isfinite(a->sp_dist)) {
                sp_dist_value = a->sp_dist;
            }
            if (idx < static_cast<size_t>(working_traj.geom.get_num_points())) {
                CORE::Point point_prev = working_traj.geom.get_point(static_cast<int>(idx - 1));
                CORE::Point point_cur = working_traj.geom.get_point(static_cast<int>(idx));
                eu_dist_value = boost::geometry::distance(point_prev, point_cur);
            }
        }
        sp_distances.push_back(sp_dist_value);
        eu_distances.push_back(eu_dist_value);
        double trust_value = a->trustworthiness;
        if (config.margin_used_trustworthiness) {
            if (idx < trust_margins.size()) {
                trust_value = trust_margins[idx];
            }
        } else {
            if (idx < n_best_trust.size() && !n_best_trust[idx].empty()) {
                trust_value = n_best_trust[idx][0];
            }
        }
        matched_candidate_path.push_back(
            MatchedCandidate{*(a->c), a->ep, a->tp, a->cumu_prob, a->sp_dist, trust_value});
    }

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
    MatchResult match_result{
        traj.id, matched_candidate_path, opath, cpath, indices, mgeom};
    match_result.nbest_trustworthiness = std::move(n_best_trust);
    match_result.candidate_details = std::move(candidate_details);
    match_result.sp_distances = std::move(sp_distances);
    match_result.eu_distances = std::move(eu_distances);
    match_result.original_indices = std::move(working_original_indices);
    return match_result;
}

MatchResult CovarianceMapMatch::match_traj(const CMMTrajectory &traj,
                                          const CovarianceMapMatchConfig &config) {
    return match_traj(traj, config, nullptr);
}

// Compute the shortest-path distance between two candidates and allow limited reverse travel.
double CovarianceMapMatch::get_sp_dist(const Candidate *ca, const Candidate *cb,
                                      double reverse_tolerance) {
    // Handle transitions along the same edge directly, consistent with FMM logic.
    if (ca->edge->id == cb->edge->id) {
        if (ca->offset <= cb->offset) {
            return cb->offset - ca->offset;
        }
        double reverse_limit = ca->edge->length * reverse_tolerance;
        if (reverse_tolerance > 0 && (ca->offset - cb->offset) < reverse_limit) {
            return 0.0;
        }
    }

    // Otherwise rely on UBODT lookup for forward path between successive edges.
    NodeIndex s = ca->edge->target;
    NodeIndex e = cb->edge->source;
    auto *r = ubodt_->look_up(s, e);
    double sp_dist = r ? r->cost : -1;
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

// Recompute emission normalization, transition penalties, and layer trust for each point pair.
void CovarianceMapMatch::update_tg_cmm(TransitionGraph *tg,
                                       const CMMTrajectory &traj,
                                       const CovarianceMapMatchConfig &config) {
    std::vector<TGLayer> &layers = tg->get_layers();
    int N = layers.size();
    if (N == 0) return;

    // Reset first layer
    tg->reset_layer(&layers[0]);
    normalize_layer_trust(&layers[0]);

    // Sweep through each pair of consecutive layers to update transition likelihoods.
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

// Update all transitions between two consecutive layers based on path feasibility and
// Mahalanobis-aware emission probabilities.
void CovarianceMapMatch::update_layer_cmm(int level, TGLayer *la_ptr, TGLayer *lb_ptr,
                                          double eu_dist, double reverse_tolerance,
                                          bool *connected,
                                          const CMMTrajectory &traj,
                                          const CovarianceMapMatchConfig &config) {
    bool layer_connected = false;
    const size_t prev_candidate_count = la_ptr->size();
    const size_t next_candidate_count = lb_ptr->size();
    std::vector<double> trust_contrib(next_candidate_count, 0.0);
    double trust_total = 0.0;

    for (auto &node_a : *la_ptr) {
        if (!(std::isfinite(node_a.trustworthiness) && node_a.trustworthiness > 0) ||
            !std::isfinite(node_a.cumu_prob)) {
            continue;
        }

        for (size_t b_idx = 0; b_idx < next_candidate_count; ++b_idx) {
            auto &node_b = (*lb_ptr)[b_idx];
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

            double trust_val = node_a.trustworthiness * node_b.ep * tp;
            trust_contrib[b_idx] += trust_val;
            trust_total += trust_val;

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

    if (next_candidate_count > 0) {
        if (trust_total > 0) {
            for (size_t b_idx = 0; b_idx < next_candidate_count; ++b_idx) {
                (*lb_ptr)[b_idx].trustworthiness = trust_contrib[b_idx] / trust_total;
            }
        } else {
            double ep_sum = 0.0;
            for (auto &node_b : *lb_ptr) {
                if (node_b.ep > 0) {
                    ep_sum += node_b.ep;
                }
            }
            if (ep_sum > 0) {
                for (auto &node_b : *lb_ptr) {
                    node_b.trustworthiness = (node_b.ep > 0) ? node_b.ep / ep_sum : 0.0;
                }
            } else {
                double uniform = next_candidate_count > 0 ? 1.0 / next_candidate_count : 0.0;
                for (auto &node_b : *lb_ptr) {
                    node_b.trustworthiness = uniform;
                }
            }
        }
    }

    if (connected != nullptr) {
        *connected = layer_connected;
    }
}

// Compute sliding-window trustworthiness using top-N (N=3) path scores.
std::pair<std::vector<double>, std::vector<std::vector<double>>>
CovarianceMapMatch::compute_window_trustworthiness(
    const Traj_Candidates &tc,
    const std::vector<std::vector<double>> &emission_probabilities,
    const CMMTrajectory &traj,
    const CovarianceMapMatchConfig &config) {

    const size_t layer_count = tc.size();
    std::vector<double> trust_margins(layer_count, std::numeric_limits<double>::quiet_NaN());
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
        const auto *start_eps = (start_idx < emission_probabilities.size())
                                    ? &emission_probabilities[start_idx]
                                    : nullptr;
        for (size_t j = 0; j < tc[start_idx].size(); ++j) {
            double ep = (start_eps && j < start_eps->size()) ? (*start_eps)[j] : 0.0;
            if (ep > 0) {
                push_top_k(&prev_scores[j], std::log(ep), k);
            }
        }

        for (size_t cursor = start_idx + 1; cursor <= end_idx; ++cursor) {
            size_t prev_idx = cursor - 1;
            std::vector<std::vector<double>> cur_scores(tc[cursor].size());
            double eu_dist = (prev_idx < euclidean_distances.size()) ? euclidean_distances[prev_idx] : 0.0;
            const auto *cur_eps = (cursor < emission_probabilities.size())
                                      ? &emission_probabilities[cursor]
                                      : nullptr;

            for (size_t b = 0; b < tc[cursor].size(); ++b) {
                double ep_b = (cur_eps && b < cur_eps->size()) ? (*cur_eps)[b] : 0.0;
                if (ep_b <= 0) {
                    continue;
                }
                double log_ep_b = std::log(ep_b);
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
        n_best[end_idx] = combined;
        if (combined.size() >= 2) {
            trust_margins[end_idx] = combined[0] - combined[1];
        } else if (combined.size() == 1) {
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
        std::vector<std::pair<CORE::Trajectory, MM::MatchResult>> result_buffer;
        const int trajectories_count = static_cast<int>(trajectories.size());
        result_buffer.resize(trajectories_count);
        #pragma omp parallel for schedule(dynamic)
        for (int idx = 0; idx < trajectories_count; ++idx) {
            const CMMTrajectory &trajectory = trajectories[idx];
            CMMTrajectory filtered_traj;
            MM::MatchResult result = match_traj(trajectory, cmm_config, &filtered_traj);
            filtered_traj.id = trajectory.id;
            CORE::Trajectory simple_traj{filtered_traj.id, filtered_traj.geom, filtered_traj.timestamps};
            #pragma omp critical(writer_section)
            {
                CORE::Trajectory output_traj = simple_traj;
                MM::MatchResult output_result = result;
                apply_output_transform(&output_traj, &output_result);
                result_buffer[idx] = std::make_pair(output_traj, output_result);
            }
            const int points_in_tr = simple_traj.geom.get_num_points();
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

        // Sort results by trajectory ID and write in order
        std::sort(result_buffer.begin(), result_buffer.end(),
            [](const auto &a, const auto &b) {
                return a.first.id < b.first.id;
            });

        for (const auto &item : result_buffer) {
            writer.write_result(item.first, item.second);
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
            CMMTrajectory filtered_traj;
            MM::MatchResult result = match_traj(trajectory, cmm_config, &filtered_traj);
            filtered_traj.id = trajectory.id;
            CORE::Trajectory simple_traj{filtered_traj.id, filtered_traj.geom, filtered_traj.timestamps};
            CORE::Trajectory output_traj = simple_traj;
            MM::MatchResult output_result = result;
            apply_output_transform(&output_traj, &output_result);
            writer.write_result(output_traj, output_result);
            const int points_in_tr = simple_traj.geom.get_num_points();
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

    // Report a concise summary so CLI users can inspect throughput and error counts.
    oss << "Status: success\n";
    oss << "Time takes " << duration << " seconds\n";
    oss << "Total points " << total_points << " matched " << points_matched << "\n";
    oss << "Trajectories processed " << total_trajs << " matched " << traj_matched << "\n";
    oss << "Map match speed " << speed << " points/s \n";
    oss << "Trajectories skipped " << invalid_records << "\n";

    return oss.str();
}
