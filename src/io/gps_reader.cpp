/**
 *  Implementation of GPS reader classes
 *
 * @author: Can Yang
 * @version: 2017.11.11
 */
#include "io/gps_reader.hpp"
#include "util/debug.hpp"
#include "util/util.hpp"
#include "config/gps_config.hpp"
#include <ogrsf_frmts.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <limits>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <cctype>

#include <boost/format.hpp>

using namespace FMM;
using namespace FMM::CORE;
using namespace FMM::IO;
using namespace FMM::UTIL;

namespace {

bool load_spatial_ref_from_dataset(const std::string &filename,
                                   OGRSpatialReference *out_sr,
                                   std::string *out_wkt) {
  if (!out_sr) {
    return false;
  }
  OGRRegisterAll();
  GDALDataset *dataset = static_cast<GDALDataset *>(
      GDALOpenEx(filename.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr));
  if (dataset == nullptr) {
    return false;
  }
  OGRLayer *layer = dataset->GetLayer(0);
  const OGRSpatialReference *layer_sr = layer ? layer->GetSpatialRef() : nullptr;
  if (!layer_sr) {
    GDALClose(dataset);
    return false;
  }
  char *wkt = nullptr;
  if (layer_sr->exportToWkt(&wkt) != OGRERR_NONE || wkt == nullptr) {
    GDALClose(dataset);
    return false;
  }
  out_sr->importFromWkt(wkt);
  out_sr->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
  if (out_wkt) {
    *out_wkt = wkt;
  }
  CPLFree(wkt);
  GDALClose(dataset);
  return true;
}

bool spatial_ref_from_string(const std::string &crs,
                             OGRSpatialReference *out_sr,
                             std::string *out_wkt) {
  if (!out_sr || crs.empty()) {
    return false;
  }
  std::string upper = crs;
  std::transform(upper.begin(), upper.end(), upper.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  OGRErr err = OGRERR_NONE;
  if (upper.find("EPSG:") == 0) {
    int epsg = std::atoi(upper.substr(5).c_str());
    err = out_sr->importFromEPSG(epsg);
  } else {
    err = out_sr->importFromWkt(crs.c_str());
  }
  if (err != OGRERR_NONE) {
    return false;
  }
  out_sr->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
  if (out_wkt) {
    char *wkt = nullptr;
    if (out_sr->exportToWkt(&wkt) == OGRERR_NONE && wkt != nullptr) {
      *out_wkt = wkt;
      CPLFree(wkt);
    }
  }
  return true;
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

bool transform_bbox(double minx, double miny, double maxx, double maxy,
                    OGRCoordinateTransformation *ct,
                    double *out_minx,
                    double *out_miny,
                    double *out_maxx,
                    double *out_maxy) {
  if (!ct || !out_minx || !out_miny || !out_maxx || !out_maxy) {
    return false;
  }
  double xs[4] = {minx, minx, maxx, maxx};
  double ys[4] = {miny, maxy, miny, maxy};
  double min_tx = std::numeric_limits<double>::infinity();
  double min_ty = std::numeric_limits<double>::infinity();
  double max_tx = -std::numeric_limits<double>::infinity();
  double max_ty = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < 4; ++i) {
    double x = xs[i];
    double y = ys[i];
    if (!ct->Transform(1, &x, &y)) {
      return false;
    }
    min_tx = std::min(min_tx, x);
    min_ty = std::min(min_ty, y);
    max_tx = std::max(max_tx, x);
    max_ty = std::max(max_ty, y);
  }
  *out_minx = min_tx;
  *out_miny = min_ty;
  *out_maxx = max_tx;
  *out_maxy = max_ty;
  return true;
}

bool get_layer_extent(const std::string &filename,
                      double *minx, double *miny,
                      double *maxx, double *maxy,
                      std::string *wkt,
                      bool *is_projected) {
  OGRRegisterAll();
  GDALDataset *dataset = static_cast<GDALDataset *>(
      GDALOpenEx(filename.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr));
  if (dataset == nullptr) {
    return false;
  }
  OGRLayer *layer = dataset->GetLayer(0);
  if (!layer) {
    GDALClose(dataset);
    return false;
  }
  OGREnvelope env;
  if (layer->GetExtent(&env, true) != OGRERR_NONE) {
    GDALClose(dataset);
    return false;
  }
  if (minx) *minx = env.MinX;
  if (miny) *miny = env.MinY;
  if (maxx) *maxx = env.MaxX;
  if (maxy) *maxy = env.MaxY;
  if (is_projected) {
    const OGRSpatialReference *layer_sr = layer->GetSpatialRef();
    *is_projected = layer_sr ? layer_sr->IsProjected() : false;
  }
  if (wkt) {
    std::string wkt_value;
    OGRSpatialReference sr;
    if (load_spatial_ref_from_dataset(filename, &sr, &wkt_value)) {
      *wkt = wkt_value;
    }
  }
  GDALClose(dataset);
  return true;
}

} // namespace

std::vector<Trajectory> ITrajectoryReader::read_next_N_trajectories(int N) {
  std::vector<Trajectory> trajectories;
  int i = 0;
  while (i < N && has_next_trajectory()) {
    trajectories.push_back(read_next_trajectory());
    ++i;
  }
  return trajectories;
}

std::vector<Trajectory> ITrajectoryReader::read_all_trajectories() {
  std::vector<Trajectory> trajectories;
  int i = 0;
  while (has_next_trajectory()) {
    trajectories.push_back(read_next_trajectory());
    ++i;
  }
  return trajectories;
}

GDALTrajectoryReader::GDALTrajectoryReader(const std::string &filename,
                                           const std::string &id_name,
                                           const std::string &timestamp_name) {
  SPDLOG_INFO("Read trajectory from file {}",filename);
  OGRRegisterAll();
  poDS = (GDALDataset *) GDALOpenEx(filename.c_str(),
                                    GDAL_OF_VECTOR, NULL, NULL, NULL);
  if (poDS == NULL) {
    std::string message = "Open data source fail";
    SPDLOG_CRITICAL(message);
    throw std::runtime_error(message);
  }
  ogrlayer = poDS->GetLayer(0);
  _cursor = 0;
  // Get the number of features first
  OGRFeatureDefn *ogrFDefn = ogrlayer->GetLayerDefn();
  NUM_FEATURES = ogrlayer->GetFeatureCount();
  // This should be a local field rather than a new variable
  id_idx = ogrFDefn->GetFieldIndex(id_name.c_str());
  if (id_idx < 0) {
    std::string message = (boost::format("Id column %1% not found") % id_name).str();
    SPDLOG_CRITICAL(message);
    GDALClose(poDS);
    throw std::runtime_error(message);
  }
  timestamp_idx = ogrFDefn->GetFieldIndex(timestamp_name.c_str());
  if (timestamp_idx < 0) {
    SPDLOG_WARN("Timestamp column {} not found", timestamp_name);
  }
  if (wkbFlatten(ogrFDefn->GetGeomType()) != wkbLineString) {
    std::string message = (boost::format("Geometry type is %1%, which should be linestring") %
            OGRGeometryTypeToName(ogrFDefn->GetGeomType())).str();
    SPDLOG_CRITICAL(message);
    GDALClose(poDS);
    throw std::runtime_error(message);
  } else {
    SPDLOG_DEBUG("Geometry type is {}",
                 OGRGeometryTypeToName(ogrFDefn->GetGeomType()));
  }
  SPDLOG_INFO("Total number of trajectories {}", NUM_FEATURES);
  SPDLOG_INFO("Finish reading meta data");
}

bool GDALTrajectoryReader::has_next_trajectory() {
  return _cursor < NUM_FEATURES;
}

bool GDALTrajectoryReader::has_timestamp() {
  return timestamp_idx > 0;
}

Trajectory GDALTrajectoryReader::read_next_trajectory() {
  OGRFeature *ogrFeature = ogrlayer->GetNextFeature();
  int trid = ogrFeature->GetFieldAsInteger(id_idx);
  OGRGeometry *rawgeometry = ogrFeature->GetGeometryRef();
  FMM::CORE::LineString linestring =
    FMM::CORE::ogr2linestring((OGRLineString *) rawgeometry);
  OGRFeature::DestroyFeature(ogrFeature);
  ++_cursor;
  return Trajectory{trid, linestring};
}

int GDALTrajectoryReader::get_num_trajectories() {
  return NUM_FEATURES;
}

void GDALTrajectoryReader::close() {
  GDALClose(poDS);
}

CSVTrajectoryReader::CSVTrajectoryReader(const std::string &e_filename,
                                         const std::string &id_name,
                                         const std::string &geom_name,
                                         const std::string &timestamp_name) :
  ifs(e_filename) {
  std::string line;
  std::getline(ifs, line);
  std::stringstream check1(line);
  std::string intermediate;
  // Tokenizing w.r.t. space ' '
  int i = 0;
  while (safe_get_line(check1, intermediate, delim)) {
    if (intermediate == id_name) {
      id_idx = i;
    }
    if (intermediate == geom_name) {
      geom_idx = i;
    }
    if (intermediate == timestamp_name) {
      timestamp_idx = i;
    }
    ++i;
  }
  if (id_idx < 0 || geom_idx < 0) {
    std::string message = (boost::format("Id %1% or Geometry column %2% not found") % id_name % geom_name).str();
    SPDLOG_CRITICAL(message);
    throw std::runtime_error(message);
  }
  if (timestamp_idx < 0) {
    SPDLOG_WARN("Timestamp column {} not found", timestamp_name);
  }
  SPDLOG_INFO("Id index {} Geometry index {} Timstamp index {}",
              id_idx, geom_idx, timestamp_idx);
}

std::vector<double> CSVTrajectoryReader::string2time(
  const std::string &str) {
  std::vector<double> values;
  std::stringstream ss(str);
  double v;
  while (ss >> v) {
    values.push_back(v);
    if (ss.peek() == ',')
      ss.ignore();
  }
  return values;
}

bool CSVTrajectoryReader::has_timestamp() {
  return timestamp_idx > 0;
}

Trajectory CSVTrajectoryReader::read_next_trajectory() {
  // Read the geom idx column into a trajectory
  std::string line;
  std::getline(ifs, line);
  std::stringstream ss(line);
  int trid = 0;
  int index = 0;
  std::string intermediate;
  FMM::CORE::LineString geom;
  std::vector<double> timestamps;
  while (std::getline(ss, intermediate, delim)) {
    if (index == id_idx) {
      trid = std::stoi(intermediate);
    }
    if (index == geom_idx) {
      // intermediate
      boost::geometry::read_wkt(intermediate, geom.get_geometry());
    }
    if (index == timestamp_idx) {
      // intermediate
      timestamps = string2time(intermediate);
    }
    ++index;
  }
  return Trajectory{trid, geom, timestamps};
}

bool CSVTrajectoryReader::has_next_trajectory() {
  return ifs.peek() != EOF;
}

void CSVTrajectoryReader::reset_cursor() {
  ifs.clear();
  ifs.seekg(0, std::ios::beg);
  std::string line;
  std::getline(ifs, line);
}
void CSVTrajectoryReader::close() {
  ifs.close();
}

CSVPointReader::CSVPointReader(const std::string &e_filename,
                               const std::string &id_name,
                               const std::string &x_name,
                               const std::string &y_name,
                               const std::string &time_name) :
  ifs(e_filename) {
  std::string line;
  std::getline(ifs, line);
  std::stringstream check1(line);
  std::string intermediate;
  // Tokenizing w.r.t. space ' '
  int i = 0;
  while (safe_get_line(check1, intermediate, delim)) {
    if (intermediate == id_name) {
      id_idx = i;
    }
    if (intermediate == x_name) {
      x_idx = i;
    }
    if (intermediate == y_name) {
      y_idx = i;
    }
    if (intermediate == time_name) {
      timestamp_idx = i;
    }
    ++i;
  }
  if (id_idx < 0 || x_idx < 0 || y_idx < 0) {
    if (id_idx < 0) {
      std::string message = (boost::format("Id column %1% not found") % id_name).str();
      SPDLOG_CRITICAL(message);
      throw std::runtime_error(message);
    }
    if (x_idx < 0) {
      std::string message = (boost::format("X column name %1% not found") % x_name).str();
      SPDLOG_CRITICAL(message);
      throw std::runtime_error(message);
    }
    if (y_idx < 0) {
      std::string message = (boost::format("Y column name %1% not found") % y_name).str();
      SPDLOG_CRITICAL(message);
      throw std::runtime_error(message);
    }
  }
  if (timestamp_idx < 0) {
    SPDLOG_WARN("Time stamp {} not found, will be estimated ", time_name);
  }
  SPDLOG_INFO("Id index {} x index {} y index {} time index {}",
              id_idx, x_idx, y_idx, timestamp_idx);
}

Trajectory CSVPointReader::read_next_trajectory() {
  // Read the geom idx column into a trajectory
  std::string intermediate;
  FMM::CORE::LineString geom;
  std::vector<double> timestamps;
  bool on_same_trajectory = true;
  bool first_observation = true;
  int trid = -1;
  int prev_id = -1;
  double prev_timestamp = -1.0;
  std::string line;
  while (on_same_trajectory && has_next_trajectory()) {
    if (prev_line.empty()) {
      std::getline(ifs, line);
    } else {
      line = prev_line;
      prev_line.clear();
    }
    std::stringstream ss(line);
    int id = 0;
    double x = 0, y = 0;
    double timestamp = 0;
    int index = 0;
    while (std::getline(ss, intermediate, delim)) {
      if (index == id_idx) {
        id = std::stoi(intermediate);
      }
      if (index == x_idx) {
        x = std::stof(intermediate);
      }
      if (index == y_idx) {
        y = std::stof(intermediate);
      }
      if (index == timestamp_idx) {
        timestamp = std::stof(intermediate);
      }
      ++index;
    }
    if (prev_id == id || first_observation) {
      geom.add_point(x, y);
      if (has_timestamp())
        timestamps.push_back(timestamp);
    }
    if (prev_id != id && !first_observation) {
      on_same_trajectory = false;
      trid = prev_id;
    }
    first_observation = false;
    prev_id = id;
    if (!on_same_trajectory) {
      prev_line = line;
    }
  }
  if (!has_next_trajectory()) {
    trid = prev_id;
  }
  return Trajectory{trid, geom, timestamps};
}

bool CSVPointReader::has_next_trajectory() {
  return ifs.peek() != EOF;
}

void CSVPointReader::reset_cursor() {
  ifs.clear();
  ifs.seekg(0, std::ios::beg);
  std::string line;
  std::getline(ifs, line);
}

void CSVPointReader::close() {
  ifs.close();
}

bool CSVPointReader::has_timestamp() {
  return timestamp_idx > 0;
}

GPSReader::GPSReader(const FMM::CONFIG::GPSConfig &config) {
  mode = config.get_gps_format();
  if (mode == 0) {
    SPDLOG_INFO("GPS data in trajectory shapefile format");
    reader = std::make_shared<GDALTrajectoryReader>
               (config.file, config.id,config.timestamp);
  } else if (mode == 1) {
    SPDLOG_INFO("GPS data in trajectory CSV format");
    reader = std::make_shared<CSVTrajectoryReader>
               (config.file, config.id, config.geom, config.timestamp);
  } else if (mode == 2) {
    SPDLOG_INFO("GPS data in point CSV format");
    reader = std::make_shared<CSVPointReader>
               (config.file, config.id, config.x, config.y, config.timestamp);
  } else {
    std::string message = "Unrecognized GPS format";
    SPDLOG_CRITICAL(message);
    throw std::runtime_error(message);
  }
};

namespace FMM {
namespace IO {

GPSBounds compute_gps_bounds(const FMM::CONFIG::GPSConfig &config) {
  GPSBounds bounds;
  GPSReader reader(config);
  double minx = std::numeric_limits<double>::infinity();
  double miny = std::numeric_limits<double>::infinity();
  double maxx = -std::numeric_limits<double>::infinity();
  double maxy = -std::numeric_limits<double>::infinity();
  while (reader.has_next_trajectory()) {
    Trajectory traj = reader.read_next_trajectory();
    int npoints = traj.geom.get_num_points();
    for (int i = 0; i < npoints; ++i) {
      double x = traj.geom.get_x(i);
      double y = traj.geom.get_y(i);
      if (std::isfinite(x) && std::isfinite(y)) {
        minx = std::min(minx, x);
        miny = std::min(miny, y);
        maxx = std::max(maxx, x);
        maxy = std::max(maxy, y);
      }
    }
  }
  reader.close();
  if (std::isfinite(minx) && std::isfinite(miny) &&
      std::isfinite(maxx) && std::isfinite(maxy)) {
    bounds.valid = true;
    bounds.minx = minx;
    bounds.miny = miny;
    bounds.maxx = maxx;
    bounds.maxy = maxy;
  }
  return bounds;
}

} // namespace IO
} // namespace FMM

GPSBounds FMM::IO::compute_gps_bounds_in_network_crs(
    const FMM::CONFIG::GPSConfig &gps_config,
    const std::string &network_file,
    bool convert_to_projected) {
  GPSBounds bounds = FMM::IO::compute_gps_bounds(gps_config);
  if (!bounds.valid) {
    return bounds;
  }

  OGRSpatialReference gps_sr;
  std::string gps_wkt;
  bool gps_has_sr = false;
  int gps_format = gps_config.get_gps_format();
  if (gps_format == 0) {
    gps_has_sr = load_spatial_ref_from_dataset(gps_config.file, &gps_sr, &gps_wkt);
  }
  if (!gps_has_sr) {
    if (!gps_config.crs.empty()) {
      gps_has_sr = spatial_ref_from_string(gps_config.crs, &gps_sr, &gps_wkt);
    } else {
      gps_has_sr = spatial_ref_from_string("EPSG:4326", &gps_sr, &gps_wkt);
      if (gps_has_sr) {
        SPDLOG_WARN("GPS CRS not specified; default to EPSG:4326 for bbox transform");
      }
    }
  }
  if (!gps_has_sr) {
    SPDLOG_WARN("Failed to determine GPS CRS; bbox stays in GPS coordinate space");
    return bounds;
  }

  OGRSpatialReference network_sr;
  std::string network_wkt;
  bool network_is_projected = false;
  double net_minx = 0.0;
  double net_miny = 0.0;
  double net_maxx = 0.0;
  double net_maxy = 0.0;
  bool got_extent = get_layer_extent(network_file, &net_minx, &net_miny,
                                     &net_maxx, &net_maxy,
                                     &network_wkt, &network_is_projected);
  if (!got_extent) {
    SPDLOG_WARN("Failed to inspect network CRS; bbox stays in GPS coordinate space");
    return bounds;
  }
  if (network_wkt.empty() && !convert_to_projected) {
    SPDLOG_WARN("Network CRS missing; default to EPSG:4326 for bbox transform");
    network_wkt = "EPSG:4326";
  }
  if (convert_to_projected && !network_is_projected) {
    double center_lon = (net_minx + net_maxx) * 0.5;
    double center_lat = (net_miny + net_maxy) * 0.5;
    int target_epsg = determine_utm_epsg(center_lon, center_lat);
    if (target_epsg > 0) {
      std::ostringstream epsg;
      epsg << "EPSG:" << target_epsg;
      if (!spatial_ref_from_string(epsg.str(), &network_sr, &network_wkt)) {
        SPDLOG_WARN("Failed to build target UTM CRS; bbox stays in GPS coordinate space");
        return bounds;
      }
    } else {
      SPDLOG_WARN("Unable to determine projected CRS from network; bbox stays in GPS coordinate space");
      return bounds;
    }
  } else {
    if (!spatial_ref_from_string(network_wkt, &network_sr, nullptr)) {
      SPDLOG_WARN("Failed to parse network CRS; bbox stays in GPS coordinate space");
      return bounds;
    }
  }
  network_sr.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);

  if (gps_sr.IsSame(&network_sr)) {
    bounds.has_spatial_ref = true;
    bounds.spatial_ref_wkt = network_wkt;
    return bounds;
  }

  OGRCoordinateTransformation *ct =
      OGRCreateCoordinateTransformation(&gps_sr, &network_sr);
  if (ct == nullptr) {
    SPDLOG_WARN("Failed to create CRS transform; bbox stays in GPS coordinate space");
    return bounds;
  }
  double out_minx = 0.0;
  double out_miny = 0.0;
  double out_maxx = 0.0;
  double out_maxy = 0.0;
  bool ok = transform_bbox(bounds.minx, bounds.miny, bounds.maxx, bounds.maxy,
                           ct, &out_minx, &out_miny, &out_maxx, &out_maxy);
  OCTDestroyCoordinateTransformation(ct);
  if (!ok) {
    SPDLOG_WARN("Failed to transform bbox; bbox stays in GPS coordinate space");
    return bounds;
  }
  bounds.minx = out_minx;
  bounds.miny = out_miny;
  bounds.maxx = out_maxx;
  bounds.maxy = out_maxy;
  bounds.has_spatial_ref = true;
  bounds.spatial_ref_wkt = network_wkt;
  SPDLOG_INFO("GPS bbox transformed to network CRS");
  return bounds;
}
