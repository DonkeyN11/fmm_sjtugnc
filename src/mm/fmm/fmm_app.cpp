//
// Created by Can Yang on 2020/4/1.
//

#include "mm/fmm/fmm_app.hpp"
#include "mm/fmm/ubodt_manager.hpp"
#include "io/gps_reader.hpp"
#include "io/mm_writer.hpp"
#include <ogrsf_frmts.h>
#include <omp.h>
#include <cmath>
#include <memory>

using namespace FMM;
using namespace FMM::CORE;
using namespace FMM::NETWORK;
using namespace FMM::MM;

namespace {

void destroy_ct(OGRCoordinateTransformation *ptr) {
  if (ptr != nullptr) {
    OCTDestroyCoordinateTransformation(ptr);
  }
}

using TransformPtr =
  std::unique_ptr<OGRCoordinateTransformation, decltype(&destroy_ct)>;

bool geometry_is_projected(const LineString &geom) {
  int num_points = geom.get_num_points();
  for (int i = 0; i < num_points; ++i) {
    double x = geom.get_x(i);
    double y = geom.get_y(i);
    if (std::abs(x) > 180.0 || std::abs(y) > 90.0) {
      return true;
    }
  }
  return false;
}

bool transform_linestring(LineString *line,
                          OGRCoordinateTransformation *transform) {
  if (line == nullptr || transform == nullptr) {
    return false;
  }
  int num_points = line->get_num_points();
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

TransformPtr make_wgs84_to_network_transform(const Network &network,
                                             bool convert_to_projected) {
  TransformPtr transform(nullptr, destroy_ct);
  if (!convert_to_projected) {
    return transform;
  }
  if (!network.is_projected()) {
    SPDLOG_WARN("convert_to_projected enabled but network CRS is not projected; skip trajectory reprojection.");
    return transform;
  }
  if (!network.has_spatial_ref()) {
    SPDLOG_WARN("Network CRS information unavailable; skip trajectory reprojection.");
    return transform;
  }
  OGRSpatialReference target_sr;
  if (target_sr.importFromWkt(network.get_spatial_ref_wkt().c_str()) != OGRERR_NONE) {
    SPDLOG_WARN("Failed to import network CRS; skip trajectory reprojection.");
    return transform;
  }
  target_sr.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
  OGRSpatialReference source_sr;
  source_sr.SetWellKnownGeogCS("WGS84");
  source_sr.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
  OGRCoordinateTransformation *ct =
    OGRCreateCoordinateTransformation(&source_sr, &target_sr);
  if (ct == nullptr) {
    SPDLOG_WARN("Failed to create coordinate transformation; skip trajectory reprojection.");
    return transform;
  }
  transform.reset(ct);
  SPDLOG_INFO("Trajectories will be reprojected to match network CRS.");
  return transform;
}

void maybe_reproject_trajectory(Trajectory *trajectory,
                                OGRCoordinateTransformation *transform) {
  if (trajectory == nullptr || transform == nullptr) {
    return;
  }
  if (trajectory->geom.get_num_points() == 0) {
    return;
  }
  if (geometry_is_projected(trajectory->geom)) {
    return;
  }
  if (!transform_linestring(&trajectory->geom, transform)) {
    SPDLOG_WARN("Failed to reproject trajectory id {}", trajectory->id);
  }
}

void maybe_reproject_trajectories(std::vector<Trajectory> *trajectories,
                                  OGRCoordinateTransformation *transform) {
  if (trajectories == nullptr || transform == nullptr) {
    return;
  }
  for (auto &trajectory : *trajectories) {
    maybe_reproject_trajectory(&trajectory, transform);
  }
}

} // namespace
void FMMApp::run() {
  auto start_time = UTIL::get_current_time();
  FastMapMatch mm_model(network_, ng_, ubodt_);
  const FastMapMatchConfig &fmm_config = config_.fmm_config;
  IO::GPSReader reader(config_.gps_config);
  IO::CSVMatchResultWriter writer(config_.result_config.file,
                                  config_.result_config.output_config);
  auto trajectory_transform = make_wgs84_to_network_transform(
    network_, config_.convert_to_projected);
  // Start map matching
  int progress = 0;
  int points_matched = 0;
  int total_points = 0;
  int step_size = 100;
  if (config_.step > 0) step_size = config_.step;
  SPDLOG_INFO("Progress report step {}", step_size);
  auto corrected_begin = UTIL::get_current_time();
  SPDLOG_INFO("Start to match trajectories");

  // Buffer for storing results when using parallel processing
  std::vector<std::pair<Trajectory, MM::MatchResult>> result_buffer;

  if (config_.use_omp){
    SPDLOG_INFO("Run map matching parallelly with buffered output");
    int buffer_trajectories_size = 100000;
    while (reader.has_next_trajectory()) {
      std::vector<Trajectory> trajectories =
        reader.read_next_N_trajectories(buffer_trajectories_size);
      int trajectories_fetched = trajectories.size();
      if (trajectories_fetched == 0) {
        continue;
      }
      maybe_reproject_trajectories(&trajectories, trajectory_transform.get());

      // Resize result buffer to match the number of trajectories
      result_buffer.resize(trajectories_fetched);

      #pragma omp parallel for
      for (int i = 0; i < trajectories_fetched; ++i) {
        Trajectory &trajectory = trajectories[i];
        int points_in_tr = trajectory.geom.get_num_points();
        MM::MatchResult result = mm_model.match_traj(
            trajectory, fmm_config);

        // Store result in buffer instead of writing immediately
        #pragma omp critical
        {
          result_buffer[i] = std::make_pair(trajectory, result);
          if (!result.cpath.empty()) {
            points_matched += points_in_tr;
          }
          total_points += points_in_tr;
          ++progress;
          if (progress % step_size == 0) {
            std::stringstream buf;
            buf << "Progress " << progress << '\n';
            std::cout << buf.rdbuf();
          }
        }
      }

      // Write all results in sorted order
      writer.write_results(result_buffer);
      result_buffer.clear();
    }
  } else {
    SPDLOG_INFO("Run map matching in single thread");
    while (reader.has_next_trajectory()) {
      if (progress % step_size == 0) {
        SPDLOG_INFO("Progress {}", progress);
      }
      Trajectory trajectory = reader.read_next_trajectory();
      maybe_reproject_trajectory(&trajectory, trajectory_transform.get());
      int points_in_tr = trajectory.geom.get_num_points();
      MM::MatchResult result = mm_model.match_traj(
          trajectory, fmm_config);
      writer.write_result(trajectory,result);
      if (!result.cpath.empty()) {
        points_matched += points_in_tr;
      }
      total_points += points_in_tr;
      ++progress;
    }
  }
  SPDLOG_INFO("MM process finished");
  auto end_time = UTIL::get_current_time();
  double time_spent = UTIL::get_duration(start_time,end_time);
  double time_spent_exclude_input = UTIL::get_duration(corrected_begin,end_time);
  SPDLOG_INFO("Time takes {}", time_spent);
  SPDLOG_INFO("Time takes excluding input {}", time_spent_exclude_input);
  SPDLOG_INFO("Finish map match total points {} matched {}",
              total_points, points_matched);
  SPDLOG_INFO("Matched percentage: {}", points_matched / (double) total_points);
  SPDLOG_INFO("Point match speed: {}", points_matched / time_spent);
  SPDLOG_INFO("Point match speed (excluding input): {}",
              points_matched / time_spent_exclude_input);
  SPDLOG_INFO("Time takes {}", time_spent);
};

// FMMApp constructor implementations
FMMApp::FMMApp(const FMMAppConfig &config) :
    config_(config),
    network_(config_.network_config, config_.convert_to_projected),
    ng_(network_) {

    // Check if UBODT is already loaded in memory
    auto &manager = UBODTManager::getInstance();

    if (config_.use_memory_cache && manager.is_loaded(config_.ubodt_file)) {
        SPDLOG_INFO("Using cached UBODT from memory");
        ubodt_ = manager.get_ubodt(config_.ubodt_file);
        // Enable auto-release so UBODT is released after use
        manager.set_auto_release(true);
    } else {
        if (UBODTManager::check_daemon_loaded(config_.ubodt_file)) {
            SPDLOG_INFO("UBODT is preloaded by ubodt_daemon. Using fast loading from OS cache.");
        } else {
            SPDLOG_INFO("UBODT not found in daemon. Loading from file.");
        }
        auto start_time = UTIL::get_current_time();
        ubodt_ = UBODT::read_ubodt_file(config_.ubodt_file);
        auto end_time = UTIL::get_current_time();
        double duration = UTIL::get_duration(start_time, end_time);
        SPDLOG_INFO("UBODT loaded in {:.2f}s", duration);
    }
}

FMMApp::FMMApp(const FMMAppConfig &config, std::shared_ptr<UBODT> preloaded_ubodt) :
    config_(config),
    network_(config_.network_config, config_.convert_to_projected),
    ng_(network_),
    ubodt_(preloaded_ubodt) {
    SPDLOG_INFO("Using provided pre-loaded UBODT");
}
