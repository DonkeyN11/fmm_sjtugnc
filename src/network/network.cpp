#include "network/network.hpp"
#include "util/debug.hpp"
#include "util/util.hpp"
#include "algorithm/geom_algorithm.hpp"

#include <ogrsf_frmts.h> // C++ API for GDAL
#include <cmath> // Calculating probability and coordinate transforms
#include <algorithm> // Partial sort copy
#include <stdexcept>
#include <limits>
#include <cstdlib>
#include <fstream>
#include <cstdint>
#include <cstring>

// Data structures for Rtree
#include <boost/geometry/index/rtree.hpp>
#include <boost/iterator/function_output_iterator.hpp>

#include <boost/format.hpp>
#include "spdlog/fmt/fmt.h"

using namespace FMM;
using namespace FMM::CORE;
using namespace FMM::MM;
using namespace FMM::NETWORK;

namespace {

constexpr char kNetworkCacheMagic[8] = {'F','M','M','N','E','T','1','\0'};
constexpr uint32_t kNetworkCacheVersion = 1;

template <typename T>
bool read_value(std::istream &input, T *value) {
  return static_cast<bool>(input.read(reinterpret_cast<char *>(value), sizeof(T)));
}

template <typename T>
bool write_value(std::ostream &output, const T &value) {
  return static_cast<bool>(output.write(reinterpret_cast<const char *>(&value), sizeof(T)));
}

bool read_string(std::istream &input, std::string *out) {
  if (!out) {
    return false;
  }
  uint32_t size = 0;
  if (!read_value(input, &size)) {
    return false;
  }
  if (size == 0) {
    out->clear();
    return true;
  }
  out->assign(size, '\0');
  return static_cast<bool>(input.read(&(*out)[0], size));
}

bool write_string(std::ostream &output, const std::string &value) {
  uint32_t size = static_cast<uint32_t>(value.size());
  if (!write_value(output, size)) {
    return false;
  }
  if (size == 0) {
    return true;
  }
  return static_cast<bool>(output.write(value.data(), size));
}

int extract_epsg_code(const OGRSpatialReference &sr) {
  const char *auth_code = sr.GetAuthorityCode(nullptr);
  if (auth_code) {
    return std::atoi(auth_code);
  }
  const char *proj_auth = sr.GetAuthorityCode("PROJCS");
  if (proj_auth) {
    return std::atoi(proj_auth);
  }
  const char *geog_auth = sr.GetAuthorityCode("GEOGCS");
  if (geog_auth) {
    return std::atoi(geog_auth);
  }
  return -1;
}

int determine_utm_epsg(double lon_deg, double lat_deg) {
  if (!std::isfinite(lon_deg) || !std::isfinite(lat_deg)) {
    return -1;
  }
  if (lat_deg <= -80.0 || lat_deg >= 84.0) {
    return -1;
  }
  int zone = static_cast<int>(std::floor((lon_deg + 180.0) / 6.0)) + 1;
  if (zone < 1) zone = 1;
  if (zone > 60) zone = 60;
  int base = (lat_deg >= 0.0) ? 32600 : 32700;
  return base + zone;
}

bool transform_linestring_inplace(FMM::CORE::LineString *line,
                                  OGRCoordinateTransformation *transform) {
  if (!line || !transform) {
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

void rebuild_vertex_points_from_edges(const std::vector<NETWORK::Edge> &edges,
                                      std::vector<FMM::CORE::Point> *vertex_points) {
  if (vertex_points == nullptr) {
    return;
  }
  std::vector<bool> assigned(vertex_points->size(), false);
  for (const auto &edge : edges) {
    if (edge.source < vertex_points->size() && !assigned[edge.source]) {
      (*vertex_points)[edge.source] = edge.geom.get_point(0);
      assigned[edge.source] = true;
    }
    if (edge.target < vertex_points->size() && !assigned[edge.target]) {
      int npoints = edge.geom.get_num_points();
      if (npoints > 0) {
        (*vertex_points)[edge.target] = edge.geom.get_point(npoints - 1);
        assigned[edge.target] = true;
      }
    }
  }
}

} // namespace

bool Network::candidate_compare(const Candidate &a, const Candidate &b) {
  if (a.dist != b.dist) {
    return a.dist < b.dist;
  } else {
    return a.edge->index < b.edge->index;
  }
}

Network::Network(const std::string &filename,
                 const std::string &id_name,
                 const std::string &source_name,
                 const std::string &target_name,
                 bool convert_to_projected)
    : convert_to_projected_(convert_to_projected),
      is_projected_(false),
      reprojected_(false),
      network_file_(filename),
      id_name_(id_name),
      source_name_(source_name),
      target_name_(target_name) {
  if (FMM::UTIL::check_file_extension(filename, "shp,gpkg,geojson,fgb")) {
    read_ogr_file(filename,id_name,source_name,target_name);
  } else {
    std::string message = (boost::format("Network file not supported %1%") % filename).str();
    SPDLOG_CRITICAL(message);
    throw std::runtime_error(message);
  }
};

Network::Network(const CONFIG::NetworkConfig &config,
                 bool convert_to_projected)
    : convert_to_projected_(convert_to_projected),
      is_projected_(false),
      reprojected_(false),
      network_file_(config.file),
      id_name_(config.id),
      source_name_(config.source),
      target_name_(config.target),
      cache_path_(config.cache),
      has_bbox_(config.has_bbox),
      bbox_minx_(config.bbox_minx),
      bbox_miny_(config.bbox_miny),
      bbox_maxx_(config.bbox_maxx),
      bbox_maxy_(config.bbox_maxy) {
  if (!cache_path_.empty()) {
    if (load_cache(cache_path_, config)) {
      SPDLOG_INFO("Loaded network cache from {}", cache_path_);
      return;
    }
    SPDLOG_WARN("Failed to load network cache; fall back to source file.");
  }
  read_ogr_file(config.file, config.id, config.source, config.target);
  if (!cache_path_.empty()) {
    if (write_cache(cache_path_, config)) {
      SPDLOG_INFO("Saved network cache to {}", cache_path_);
    } else {
      SPDLOG_WARN("Failed to write network cache to {}", cache_path_);
    }
  }
}

void Network::add_edge(EdgeID edge_id, NodeID source, NodeID target,
                       const FMM::CORE::LineString &geom){
  NodeIndex s_idx, t_idx;
  if (node_map.find(source) == node_map.end()) {
    s_idx = node_id_vec.size();
    node_id_vec.push_back(source);
    node_map.insert({source, s_idx});
    vertex_points.push_back(geom.get_point(0));
  } else {
    s_idx = node_map[source];
  }
  if (node_map.find(target) == node_map.end()) {
    t_idx = node_id_vec.size();
    node_id_vec.push_back(target);
    node_map.insert({target, t_idx});
    int npoints = geom.get_num_points();
    vertex_points.push_back(geom.get_point(npoints - 1));
  } else {
    t_idx = node_map[target];
  }
  EdgeIndex index = edges.size();
  edges.push_back({index, edge_id, s_idx, t_idx, geom.get_length(), geom});
  edge_map.insert({edge_id, index});
};

bool Network::load_cache(const std::string &filename,
                         const CONFIG::NetworkConfig &config) {
  std::ifstream input(filename, std::ios::binary);
  if (!input) {
    return false;
  }
  char magic[8] = {};
  if (!input.read(magic, sizeof(magic))) {
    return false;
  }
  if (std::memcmp(magic, kNetworkCacheMagic, sizeof(kNetworkCacheMagic)) != 0) {
    SPDLOG_WARN("Network cache magic mismatch");
    return false;
  }
  uint32_t version = 0;
  if (!read_value(input, &version) || version != kNetworkCacheVersion) {
    SPDLOG_WARN("Network cache version mismatch");
    return false;
  }
  uint32_t flags = 0;
  int32_t cache_srid = 0;
  if (!read_value(input, &flags) || !read_value(input, &cache_srid)) {
    return false;
  }
  bool cache_is_projected = (flags & 0x1u) != 0u;
  bool cache_reprojected = (flags & 0x2u) != 0u;
  bool cache_has_bbox = (flags & 0x4u) != 0u;
  double cache_minx = 0.0;
  double cache_miny = 0.0;
  double cache_maxx = 0.0;
  double cache_maxy = 0.0;
  if (!read_value(input, &cache_minx) ||
      !read_value(input, &cache_miny) ||
      !read_value(input, &cache_maxx) ||
      !read_value(input, &cache_maxy)) {
    return false;
  }
  std::string cache_file;
  std::string cache_id;
  std::string cache_source;
  std::string cache_target;
  std::string cache_wkt;
  if (!read_string(input, &cache_file) ||
      !read_string(input, &cache_id) ||
      !read_string(input, &cache_source) ||
      !read_string(input, &cache_target) ||
      !read_string(input, &cache_wkt)) {
    return false;
  }
  if (cache_file != config.file ||
      cache_id != config.id ||
      cache_source != config.source ||
      cache_target != config.target) {
    SPDLOG_WARN("Network cache metadata mismatch; skip cache");
    return false;
  }
  if (config.has_bbox) {
    if (!cache_has_bbox) {
      SPDLOG_WARN("Network cache lacks bbox; skip cache");
      return false;
    }
    auto mismatch = [](double a, double b) {
      return std::fabs(a - b) > 1e-9;
    };
    if (mismatch(cache_minx, config.bbox_minx) ||
        mismatch(cache_miny, config.bbox_miny) ||
        mismatch(cache_maxx, config.bbox_maxx) ||
        mismatch(cache_maxy, config.bbox_maxy)) {
      SPDLOG_WARN("Network cache bbox mismatch; skip cache");
      return false;
    }
  } else if (cache_has_bbox) {
    SPDLOG_WARN("Network cache is bbox-filtered but config is not; skip cache");
    return false;
  }
  if (convert_to_projected_) {
    if (!cache_is_projected) {
      SPDLOG_WARN("Network cache is not projected but projection requested; skip cache");
      return false;
    }
  } else {
    if (cache_reprojected) {
      SPDLOG_WARN("Network cache is reprojected but projection disabled; skip cache");
      return false;
    }
  }
  uint64_t node_count = 0;
  uint64_t edge_count = 0;
  if (!read_value(input, &node_count) || !read_value(input, &edge_count)) {
    return false;
  }
  node_id_vec.clear();
  edges.clear();
  node_map.clear();
  edge_map.clear();
  vertex_points.clear();
  node_id_vec.resize(node_count);
  for (uint64_t i = 0; i < node_count; ++i) {
    NodeID value = 0;
    if (!read_value(input, &value)) {
      return false;
    }
    node_id_vec[i] = value;
    node_map.insert({value, static_cast<NodeIndex>(i)});
  }
  edges.reserve(edge_count);
  for (uint64_t i = 0; i < edge_count; ++i) {
    EdgeID edge_id = 0;
    uint32_t source_idx = 0;
    uint32_t target_idx = 0;
    double length = 0.0;
    uint32_t num_points = 0;
    if (!read_value(input, &edge_id) ||
        !read_value(input, &source_idx) ||
        !read_value(input, &target_idx) ||
        !read_value(input, &length) ||
        !read_value(input, &num_points)) {
      return false;
    }
    CORE::LineString geom;
    for (uint32_t p = 0; p < num_points; ++p) {
      double x = 0.0;
      double y = 0.0;
      if (!read_value(input, &x) || !read_value(input, &y)) {
        return false;
      }
      geom.add_point(x, y);
    }
    EdgeIndex index = static_cast<EdgeIndex>(edges.size());
    edges.push_back({index, edge_id, source_idx, target_idx, length, geom});
    edge_map.insert({edge_id, index});
  }
  num_vertices = node_id_vec.size();
  vertex_points.assign(num_vertices, CORE::Point());
  rebuild_vertex_points_from_edges(edges, &vertex_points);
  srid = cache_srid;
  spatial_ref_wkt_ = cache_wkt;
  is_projected_ = cache_is_projected;
  reprojected_ = cache_reprojected;
  build_rtree_index();
  return true;
}

bool Network::write_cache(const std::string &filename,
                          const CONFIG::NetworkConfig &config) const {
  std::ofstream output(filename, std::ios::binary | std::ios::trunc);
  if (!output) {
    return false;
  }
  output.write(kNetworkCacheMagic, sizeof(kNetworkCacheMagic));
  if (!write_value(output, kNetworkCacheVersion)) {
    return false;
  }
  uint32_t flags = 0;
  if (is_projected_) {
    flags |= 0x1u;
  }
  if (reprojected_) {
    flags |= 0x2u;
  }
  if (config.has_bbox) {
    flags |= 0x4u;
  }
  if (!write_value(output, flags) ||
      !write_value(output, static_cast<int32_t>(srid)) ||
      !write_value(output, config.bbox_minx) ||
      !write_value(output, config.bbox_miny) ||
      !write_value(output, config.bbox_maxx) ||
      !write_value(output, config.bbox_maxy)) {
    return false;
  }
  if (!write_string(output, config.file) ||
      !write_string(output, config.id) ||
      !write_string(output, config.source) ||
      !write_string(output, config.target) ||
      !write_string(output, spatial_ref_wkt_)) {
    return false;
  }
  uint64_t node_count = node_id_vec.size();
  uint64_t edge_count = edges.size();
  if (!write_value(output, node_count) || !write_value(output, edge_count)) {
    return false;
  }
  for (NodeID node_id : node_id_vec) {
    if (!write_value(output, node_id)) {
      return false;
    }
  }
  for (const auto &edge : edges) {
    uint32_t source_idx = static_cast<uint32_t>(edge.source);
    uint32_t target_idx = static_cast<uint32_t>(edge.target);
    uint32_t num_points = static_cast<uint32_t>(edge.geom.get_num_points());
    if (!write_value(output, edge.id) ||
        !write_value(output, source_idx) ||
        !write_value(output, target_idx) ||
        !write_value(output, edge.length) ||
        !write_value(output, num_points)) {
      return false;
    }
    for (uint32_t p = 0; p < num_points; ++p) {
      double x = edge.geom.get_x(p);
      double y = edge.geom.get_y(p);
      if (!write_value(output, x) || !write_value(output, y)) {
        return false;
      }
    }
  }
  return static_cast<bool>(output);
}

void Network::read_ogr_file(const std::string &filename,
                            const std::string &id_name,
                            const std::string &source_name,
                            const std::string &target_name) {
  SPDLOG_INFO("Read network from file {}", filename);
  OGRRegisterAll();
  GDALDataset *poDS = (GDALDataset *) GDALOpenEx(
    filename.c_str(), GDAL_OF_VECTOR, NULL, NULL, NULL);
  if (poDS == NULL) {
    std::string message = "Open dataset failed.";
    SPDLOG_CRITICAL(message);
    throw std::runtime_error(message);
  }
  OGRLayer *ogrlayer = poDS->GetLayer(0);
  if (has_bbox_) {
    ogrlayer->SetSpatialFilterRect(bbox_minx_, bbox_miny_, bbox_maxx_, bbox_maxy_);
    ogrlayer->ResetReading();
    SPDLOG_INFO("Apply network bbox filter [{},{}] - [{},{}]",
                bbox_minx_, bbox_miny_, bbox_maxx_, bbox_maxy_);
  }
  int NUM_FEATURES = ogrlayer->GetFeatureCount();
  // edges= std::vector<Edge>(NUM_FEATURES);
  // Initialize network edges
  OGRFeatureDefn *ogrFDefn = ogrlayer->GetLayerDefn();
  OGRFeature *ogrFeature;

  // Fetch the field index given field name.
  int id_idx = ogrFDefn->GetFieldIndex(id_name.c_str());
  int source_idx = ogrFDefn->GetFieldIndex(source_name.c_str());
  int target_idx = ogrFDefn->GetFieldIndex(target_name.c_str());
  if (source_idx < 0 || target_idx < 0 || id_idx < 0) {
    std::string error_message = fmt::format(
      "Field not found: {} index {}, {} index {}, {} index {}",
      id_name, id_idx, source_name, source_idx,
      target_name, target_idx);
    SPDLOG_CRITICAL(error_message);
    GDALClose(poDS);
    throw std::runtime_error(error_message);
  }

  if (wkbFlatten(ogrFDefn->GetGeomType()) != wkbLineString) {
    std::string error_message = fmt::format(
      "Geometry type of network is {}, should be linestring",
      OGRGeometryTypeToName(ogrFDefn->GetGeomType()));
    SPDLOG_CRITICAL(error_message);
    GDALClose(poDS);
    throw std::runtime_error(error_message);
  } else {
    SPDLOG_DEBUG("Geometry type of network is {}",
                 OGRGeometryTypeToName(ogrFDefn->GetGeomType()));
  }
  const OGRSpatialReference *ogrsr = ogrFDefn->GetGeomFieldDefn(0)->GetSpatialRef();
  std::string original_wkt;
  bool original_is_projected = false;
  if (ogrsr != nullptr) {
    char *wkt = nullptr;
    if (ogrsr->exportToWkt(&wkt) == OGRERR_NONE && wkt != nullptr) {
      original_wkt.assign(wkt);
      CPLFree(wkt);
    }
    original_is_projected = ogrsr->IsProjected();
    srid = extract_epsg_code(*ogrsr);
    if (srid != -1) {
      SPDLOG_DEBUG("Detected SRID {}", srid);
    } else if (ogrsr->IsGeographic()) {
      srid = 4326;
      SPDLOG_WARN("SRID not found, assume EPSG:4326");
    } else {
      SPDLOG_WARN("SRID not found for projected network");
    }
  } else {
    srid = 4326;
    SPDLOG_WARN("Spatial reference not found, assume EPSG:4326");
  }
  spatial_ref_wkt_ = original_wkt;
  is_projected_ = original_is_projected;

  double min_x = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();

  // Read data from shapefile
  EdgeIndex index = 0;
  while ((ogrFeature = ogrlayer->GetNextFeature()) != NULL) {
    EdgeID id = ogrFeature->GetFieldAsInteger64(id_idx);
    NodeID source = ogrFeature->GetFieldAsInteger64(source_idx);
    NodeID target = ogrFeature->GetFieldAsInteger64(target_idx);
    OGRGeometry *rawgeometry = ogrFeature->GetGeometryRef();
    LineString geom;
    if (rawgeometry->getGeometryType() == wkbLineString) {
      geom = ogr2linestring((OGRLineString *) rawgeometry);
    } else if (rawgeometry->getGeometryType() == wkbMultiLineString) {
      SPDLOG_TRACE("Feature id {} s {} t {} is multilinestring",
                   id, source, target);
      SPDLOG_TRACE("Read only the first linestring");
      geom = ogr2linestring((OGRMultiLineString *) rawgeometry);
    } else {
      SPDLOG_CRITICAL("Unknown geometry type for feature id {} s {} t {}",
                      id, source, target);
    }
    NodeIndex s_idx, t_idx;
    if (node_map.find(source) == node_map.end()) {
      s_idx = node_id_vec.size();
      node_id_vec.push_back(source);
      node_map.insert({source, s_idx});
      vertex_points.push_back(geom.get_point(0));
    } else {
      s_idx = node_map[source];
    }
    if (node_map.find(target) == node_map.end()) {
      t_idx = node_id_vec.size();
      node_id_vec.push_back(target);
      node_map.insert({target, t_idx});
      int npoints = geom.get_num_points();
      vertex_points.push_back(geom.get_point(npoints - 1));
    } else {
      t_idx = node_map[target];
    }
    int npoints = geom.get_num_points();
    for (int i = 0; i < npoints; ++i) {
      double px = geom.get_x(i);
      double py = geom.get_y(i);
      if (std::isfinite(px) && std::isfinite(py)) {
        min_x = std::min(min_x, px);
        max_x = std::max(max_x, px);
        min_y = std::min(min_y, py);
        max_y = std::max(max_y, py);
      }
    }
    edges.push_back({index, id, s_idx, t_idx, geom.get_length(), geom});
    edge_map.insert({id, index});
    ++index;
    OGRFeature::DestroyFeature(ogrFeature);
  }
  bool needs_reprojection = convert_to_projected_ && !is_projected_;
  if (needs_reprojection) {
    bool has_bbox = std::isfinite(min_x) && std::isfinite(max_x) &&
                    std::isfinite(min_y) && std::isfinite(max_y);
    if (!has_bbox) {
      SPDLOG_WARN("Unable to determine network extent; skip reprojection.");
    } else {
      double center_lon = (min_x + max_x) * 0.5;
      double center_lat = (min_y + max_y) * 0.5;
      int target_epsg = determine_utm_epsg(center_lon, center_lat);
      if (target_epsg > 0) {
        OGRSpatialReference source_sr;
        OGRErr source_err = OGRERR_NONE;
        if (!spatial_ref_wkt_.empty()) {
          source_err = source_sr.importFromWkt(spatial_ref_wkt_.c_str());
        } else {
          source_err = source_sr.SetWellKnownGeogCS("WGS84");
        }
        source_sr.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
        if (source_err != OGRERR_NONE) {
          SPDLOG_WARN("Failed to prepare source CRS for reprojection (err={}); skip reprojection.", source_err);
        } else {
          OGRSpatialReference target_sr;
          if (target_sr.importFromEPSG(target_epsg) != OGRERR_NONE) {
            SPDLOG_WARN("Failed to create target CRS EPSG:{}; skip reprojection.", target_epsg);
          } else {
            target_sr.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
            OGRCoordinateTransformation *ct = OGRCreateCoordinateTransformation(&source_sr, &target_sr);
            if (ct == nullptr) {
              SPDLOG_WARN("Failed to create coordinate transformation; skip reprojection.");
            } else {
              std::vector<CORE::LineString> transformed_geoms;
              transformed_geoms.reserve(edges.size());
              bool success = true;
              for (const auto &edge : edges) {
                CORE::LineString geom_copy = edge.geom;
                if (!transform_linestring_inplace(&geom_copy, ct)) {
                  success = false;
                  break;
                }
                transformed_geoms.push_back(std::move(geom_copy));
              }
              if (!success) {
                SPDLOG_WARN("Network contains coordinates outside the target projection; keep original CRS.");
              } else {
                for (std::size_t idx = 0; idx < edges.size(); ++idx) {
                  edges[idx].geom = std::move(transformed_geoms[idx]);
                  edges[idx].length = edges[idx].geom.get_length();
                }
                rebuild_vertex_points_from_edges(edges, &vertex_points);
                char *target_wkt = nullptr;
                if (target_sr.exportToWkt(&target_wkt) == OGRERR_NONE && target_wkt != nullptr) {
                  spatial_ref_wkt_ = target_wkt;
                  CPLFree(target_wkt);
                } else {
                  spatial_ref_wkt_.clear();
                }
                srid = target_epsg;
                is_projected_ = true;
                reprojected_ = true;
                SPDLOG_INFO("Reprojected network to EPSG:{}", target_epsg);
              }
              OCTDestroyCoordinateTransformation(ct);
            }
          }
        }
      } else {
        SPDLOG_WARN("Unable to determine suitable projected CRS for network; skip reprojection.");
      }
    }
  }
  GDALClose(poDS);
  num_vertices = node_id_vec.size();
  SPDLOG_INFO("Number of edges {} nodes {}", edges.size(), num_vertices);
  SPDLOG_INFO("Field index: id {} source {} target {}",
              id_idx, source_idx, target_idx);
  build_rtree_index();
  SPDLOG_INFO("Read network done.");
}    // Network constructor

int Network::get_node_count() const {
  return node_id_vec.size();
}

int Network::get_edge_count() const {
  return edges.size();
}

// Get the edge vector
const std::vector<Edge> &Network::get_edges() const {
  return edges;
}

const Edge& Network::get_edge(EdgeID id) const {
  return edges[get_edge_index(id)];
};

const Edge& Network::get_edge(EdgeIndex index) const {
  return edges[index];
};

// Get the ID attribute of an edge according to its index
EdgeID Network::get_edge_id(EdgeIndex index) const {
  return index < edges.size() ? edges[index].id : -1;
}

EdgeIndex Network::get_edge_index(EdgeID id) const {
  return edge_map.at(id);
}

NodeID Network::get_node_id(NodeIndex index) const {
  return index < num_vertices ? node_id_vec[index] : -1;
}

NodeIndex Network::get_node_index(NodeID id) const {
  return node_map.at(id);
}

Point Network::get_node_geom_from_idx(NodeIndex index) const {
  return vertex_points[index];
}

// Construct a Rtree using the vector of edges
void Network::build_rtree_index() {
  // Build an rtree for candidate search
  SPDLOG_DEBUG("Create boost rtree");
  std::vector<Item> items(edges.size());
  const std::size_t total_edges = edges.size();
#pragma omp parallel for if (total_edges > 10000)
  for (std::int64_t i = 0; i < static_cast<std::int64_t>(total_edges); ++i) {
    Edge *edge = &edges[static_cast<std::size_t>(i)];
    double x1, y1, x2, y2;
    ALGORITHM::boundingbox_geometry(edge->geom, &x1, &y1, &x2, &y2);
    boost_box b(Point(x1, y1), Point(x2, y2));
    items[static_cast<std::size_t>(i)] = std::make_pair(b, edge);
  }
  rtree = Rtree(items.begin(), items.end());
  SPDLOG_DEBUG("Create boost rtree done");
}

Traj_Candidates Network::search_tr_cs_knn(Trajectory &trajectory, std::size_t k,
                                          double radius) const {
  return search_tr_cs_knn(trajectory.geom, k, radius);
}

Traj_Candidates Network::search_tr_cs_knn(const LineString &geom, std::size_t k,
                                          double radius) const {
  int NumberPoints = geom.get_num_points();
  Traj_Candidates tr_cs(NumberPoints);
  unsigned int current_candidate_index = num_vertices;
  for (int i = 0; i < NumberPoints; ++i) {
    // SPDLOG_DEBUG("Search candidates for point index {}",i);
    // Construct a bounding boost_box
    double px = geom.get_x(i);
    double py = geom.get_y(i);
    Point_Candidates pcs;
    boost_box b(Point(geom.get_x(i) - radius, geom.get_y(i) - radius),
                Point(geom.get_x(i) + radius, geom.get_y(i) + radius));
    std::vector<Item> temp;
    // Rtree can only detect intersect with a the bounding box of
    // the geometry stored.
    rtree.query(boost::geometry::index::intersects(b),
                std::back_inserter(temp));
    int Nitems = temp.size();
    for (unsigned int j = 0; j < Nitems; ++j) {
      // Check for detailed intersection
      // The two edges are all in OGR_linestring
      Edge *edge = temp[j].second;
      double offset;
      double dist;
      double closest_x, closest_y;
      ALGORITHM::linear_referencing(px, py, edge->geom,
                                    &dist, &offset, &closest_x, &closest_y);
      if (dist <= radius) {
        // index, offset, dist, edge, pseudo id, point
        Candidate c = {0,
                       offset,
                       dist,
                       edge,
                       Point(closest_x, closest_y)};
        pcs.push_back(c);
      }
    }
    SPDLOG_DEBUG("Candidate count point {}: {} (filter to k)",i,pcs.size());
    if (pcs.empty()) {
      SPDLOG_DEBUG("Candidate not found for point {}: {} {}",i,px,py);
      return Traj_Candidates();
    }
    // KNN part
    if (pcs.size() <= k) {
      tr_cs[i] = pcs;
    } else {
      tr_cs[i] = Point_Candidates(k);
      std::partial_sort_copy(
        pcs.begin(), pcs.end(),
        tr_cs[i].begin(), tr_cs[i].end(),
        candidate_compare);
    }
    for (int m = 0; m < tr_cs[i].size(); ++m) {
      tr_cs[i][m].index = current_candidate_index + m;
    }
    current_candidate_index += tr_cs[i].size();
    // SPDLOG_TRACE("current_candidate_index {}",current_candidate_index);
  }
  return tr_cs;
}

const LineString &Network::get_edge_geom(EdgeID edge_id) const {
  return edges[get_edge_index(edge_id)].geom;
}

LineString Network::complete_path_to_geometry(
  const LineString &traj, const C_Path &complete_path) const {
  // if (complete_path->empty()) return nullptr;
  LineString line;
  if (complete_path.empty()) return line;
  int Npts = traj.get_num_points();
  int NCsegs = complete_path.size();
  if (NCsegs == 1) {
    double dist;
    double firstoffset;
    double lastoffset;
    const LineString &firstseg = get_edge_geom(complete_path[0]);
    ALGORITHM::linear_referencing(traj.get_x(0), traj.get_y(0), firstseg,
                                  &dist, &firstoffset);
    ALGORITHM::linear_referencing(traj.get_x(Npts - 1), traj.get_y(Npts - 1),
                                  firstseg, &dist, &lastoffset);
    LineString firstlineseg = ALGORITHM::cutoffseg_unique(firstseg, firstoffset,
                                                          lastoffset);
    append_segs_to_line(&line, firstlineseg, 0);
  } else {
    const LineString &firstseg = get_edge_geom(complete_path[0]);
    const LineString &lastseg = get_edge_geom(complete_path[NCsegs - 1]);
    double dist;
    double firstoffset;
    double lastoffset;
    ALGORITHM::linear_referencing(traj.get_x(0), traj.get_y(0), firstseg,
                                  &dist, &firstoffset);
    ALGORITHM::linear_referencing(traj.get_x(Npts - 1), traj.get_y(Npts - 1),
                                  lastseg, &dist, &lastoffset);
    LineString firstlineseg = ALGORITHM::cutoffseg(firstseg, firstoffset, 0);
    LineString lastlineseg = ALGORITHM::cutoffseg(lastseg, lastoffset, 1);
    append_segs_to_line(&line, firstlineseg, 0);
    if (NCsegs > 2) {
      for (int i = 1; i < NCsegs - 1; ++i) {
        const LineString &middleseg = get_edge_geom(complete_path[i]);
        append_segs_to_line(&line, middleseg, 1);
      }
    }
    append_segs_to_line(&line, lastlineseg, 1);
  }
  return line;
}

const std::vector<Point> &Network::get_vertex_points() const {
  return vertex_points;
}

const Point &Network::get_vertex_point(NodeIndex u) const {
  return vertex_points[u];
}

LineString Network::route2geometry(const std::vector<EdgeID> &path) const {
  LineString line;
  if (path.empty()) return line;
  // if (complete_path->empty()) return nullptr;
  int NCsegs = path.size();
  for (int i = 0; i < NCsegs; ++i) {
    EdgeIndex e = get_edge_index(path[i]);
    const LineString &seg = edges[e].geom;
    if (i == 0) {
      append_segs_to_line(&line, seg, 0);
    } else {
      append_segs_to_line(&line, seg, 1);
    }
  }
  //SPDLOG_DEBUG("Path geometry is {}",line.exportToWkt());
  return line;
}

LineString Network::route2geometry(const std::vector<EdgeIndex> &path) const {
  LineString line;
  if (path.empty()) return line;
  // if (complete_path->empty()) return nullptr;
  int NCsegs = path.size();
  for (int i = 0; i < NCsegs; ++i) {
    const LineString &seg = edges[path[i]].geom;
    if (i == 0) {
      append_segs_to_line(&line, seg, 0);
    } else {
      append_segs_to_line(&line, seg, 1);
    }
  }
  //SPDLOG_DEBUG("Path geometry is {}",line.exportToWkt());
  return line;
}

void Network::append_segs_to_line(LineString *line,
                                  const LineString &segs, int offset) {
  int Npoints = segs.get_num_points();
  for (int i = 0; i < Npoints; ++i) {
    if (i >= offset) {
      line->add_point(segs.get_x(i), segs.get_y(i));
    }
  }
}
