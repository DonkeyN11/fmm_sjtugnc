/**
 * Content
 * Definition of MatchResultWriter Class, which contains functions for
 * writing the results.
 *
 * @author: Can Yang
 * @version: 2017.11.11
 */

#include "io/mm_writer.hpp"
#include "util/util.hpp"
#include "util/debug.hpp"
#include "config/result_config.hpp"
#include <omp.h>

#include <boost/geometry.hpp>
#include <sstream>

namespace FMM {

namespace IO {

CSVMatchResultWriter::CSVMatchResultWriter(
    const std::string &result_file, const CONFIG::OutputConfig &config_arg) :
    m_fstream(result_file), config_(config_arg) {
  write_header();
}

void CSVMatchResultWriter::write_header() {
  std::string header = "id";
  if (config_.write_opath) header += ";opath";
  if (config_.write_error) header += ";error";
  if (config_.write_offset) header += ";offset";
  if (config_.write_spdist) header += ";spdist";
  if (config_.write_sp_dist) header += ";sp_dist";
  if (config_.write_eu_dist) header += ";eu_dist";
  if (config_.write_pgeom) header += ";pgeom";
  if (config_.write_cpath) header += ";cpath";
  if (config_.write_tpath) header += ";tpath";
  if (config_.write_mgeom) header += ";mgeom";
  if (config_.write_ep) header += ";ep";
  if (config_.write_tp) header += ";tp";
  if (config_.write_trustworthiness) header += ";trustworthiness";
  if (config_.write_n_best_trustworthiness) header += ";n_best_trustworthiness";
  if (config_.write_cumu_prob) header += ";cumu_prob";
  if (config_.write_candidates) header += ";candidates";
  if (config_.write_length) header += ";length";
  if (config_.write_duration) header += ";duration";
  if (config_.write_speed) header += ";speed";
  if (config_.write_timestamp) header += ";timestamp";
  m_fstream << header << '\n';
}

void CSVMatchResultWriter::write_result(
    const FMM::CORE::Trajectory &traj,
    const FMM::MM::MatchResult &result) {
  std::stringstream buf;
  buf << result.id;
  if (config_.write_opath) {
    buf << ";" << result.opath;
  }
  if (config_.write_error) {
    buf << ";";
    if (!result.opt_candidate_path.empty()) {
      // If original_indices is available, output aligned with original trajectory points
      if (!result.original_indices.empty()) {
        int total_points = *std::max_element(result.original_indices.begin(), result.original_indices.end()) + 1;
        std::vector<double> output_values(total_points, -1.0);
        for (size_t i = 0; i < result.original_indices.size(); ++i) {
          int original_idx = result.original_indices[i];
          if (original_idx >= 0 && original_idx < total_points && i < result.opt_candidate_path.size()) {
            output_values[original_idx] = result.opt_candidate_path[i].c.dist;
          }
        }
        for (int i = 0; i < total_points; ++i) {
          buf << output_values[i] << (i==total_points-1?"":",");
        }
      } else {
        // Fallback to original behavior when original_indices is not available
        int N = result.opt_candidate_path.size();
        for (int i = 0; i < N - 1; ++i) {
          buf << result.opt_candidate_path[i].c.dist << ",";
        }
        buf << result.opt_candidate_path[N - 1].c.dist;
      }
    }
  }
  if (config_.write_offset) {
    buf << ";";
    if (!result.opt_candidate_path.empty()) {
      // If original_indices is available, output aligned with original trajectory points
      if (!result.original_indices.empty()) {
        int total_points = *std::max_element(result.original_indices.begin(), result.original_indices.end()) + 1;
        std::vector<double> output_values(total_points, -1.0);
        for (size_t i = 0; i < result.original_indices.size(); ++i) {
          int original_idx = result.original_indices[i];
          if (original_idx >= 0 && original_idx < total_points && i < result.opt_candidate_path.size()) {
            output_values[original_idx] = result.opt_candidate_path[i].c.offset;
          }
        }
        for (int i = 0; i < total_points; ++i) {
          buf << output_values[i] << (i==total_points-1?"":",");
        }
      } else {
        // Fallback to original behavior when original_indices is not available
        int N = result.opt_candidate_path.size();
        for (int i = 0; i < N - 1; ++i) {
          buf << result.opt_candidate_path[i].c.offset << ",";
        }
        buf << result.opt_candidate_path[N - 1].c.offset;
      }
    }
  }
  if (config_.write_spdist) {
    buf << ";";
    if (!result.opt_candidate_path.empty()) {
      // If original_indices is available, output aligned with original trajectory points
      if (!result.original_indices.empty()) {
        int total_points = *std::max_element(result.original_indices.begin(), result.original_indices.end()) + 1;
        std::vector<double> output_values(total_points, -1.0);
        for (size_t i = 0; i < result.original_indices.size(); ++i) {
          int original_idx = result.original_indices[i];
          if (original_idx >= 0 && original_idx < total_points) {
            double value = -1.0;
            if (i == 0) {
              value = 0.0;
            } else if (i < result.opt_candidate_path.size()) {
              value = result.opt_candidate_path[i].sp_dist;
            }
            output_values[original_idx] = value;
          }
        }
        for (int i = 0; i < total_points; ++i) {
          buf << output_values[i] << (i==total_points-1?"":",");
        }
      } else {
        // Fallback to original behavior when original_indices is not available
        int N = result.opt_candidate_path.size();
        for (int i = 1; i < N; ++i) {
          buf << result.opt_candidate_path[i].sp_dist
              << (i==N-1?"":",");
        }
      }
    }
  }
  if (config_.write_sp_dist) {
    buf << ";";
    if (!result.opt_candidate_path.empty()) {
      // If original_indices is available, output aligned with original trajectory points
      if (!result.original_indices.empty()) {
        int total_points = *std::max_element(result.original_indices.begin(), result.original_indices.end()) + 1;
        std::vector<double> output_values(total_points, -1.0);
        for (size_t i = 0; i < result.original_indices.size(); ++i) {
          int original_idx = result.original_indices[i];
          if (original_idx >= 0 && original_idx < total_points) {
            double value = 0.0;
            if (i < result.sp_distances.size()) {
              value = result.sp_distances[i];
            } else if (i < result.opt_candidate_path.size() && result.opt_candidate_path[i].sp_dist >= 0) {
              value = result.opt_candidate_path[i].sp_dist;
            }
            output_values[original_idx] = value;
          }
        }
        for (int i = 0; i < total_points; ++i) {
          buf << output_values[i] << (i==total_points-1?"":",");
        }
      } else {
        // Fallback to original behavior when original_indices is not available
        int N = result.opt_candidate_path.size();
        for (int i = 0; i < N; ++i) {
          double value = -1.0;
          if (!result.sp_distances.empty() && i < static_cast<int>(result.sp_distances.size())) {
            value = result.sp_distances[i];
          } else if (i == 0) {
            value = 0.0;
          } else {
            value = result.opt_candidate_path[i].sp_dist;
          }
          buf << value << (i==N-1?"":",");
        }
      }
    }
  }
  if (config_.write_eu_dist) {
    buf << ";";
    if (!result.opt_candidate_path.empty()) {
      // If original_indices is available, output aligned with original trajectory points
      if (!result.original_indices.empty()) {
        int total_points = *std::max_element(result.original_indices.begin(), result.original_indices.end()) + 1;
        std::vector<double> output_values(total_points, -1.0);
        for (size_t i = 0; i < result.original_indices.size(); ++i) {
          int original_idx = result.original_indices[i];
          if (original_idx >= 0 && original_idx < total_points) {
            double value = 0.0;
            if (i < result.eu_distances.size()) {
              value = result.eu_distances[i];
            } else if (i == 0) {
              value = 0.0;
            } else if (i > 0 && i < static_cast<size_t>(traj.geom.get_num_points())) {
              const auto &prev_point = traj.geom.get_point(static_cast<int>(i - 1));
              const auto &cur_point = traj.geom.get_point(static_cast<int>(i));
              value = boost::geometry::distance(prev_point, cur_point);
            }
            output_values[original_idx] = value;
          }
        }
        for (int i = 0; i < total_points; ++i) {
          buf << output_values[i] << (i==total_points-1?"":",");
        }
      } else {
        // Fallback to original behavior when original_indices is not available
        int N = result.opt_candidate_path.size();
        for (int i = 0; i < N; ++i) {
          double value = -1.0;
          if (!result.eu_distances.empty() && i < static_cast<int>(result.eu_distances.size())) {
            value = result.eu_distances[i];
          } else if (i == 0) {
            value = 0.0;
          } else if (i < traj.geom.get_num_points()) {
            const auto &prev_point = traj.geom.get_point(i - 1);
            const auto &cur_point = traj.geom.get_point(i);
            value = boost::geometry::distance(prev_point, cur_point);
          }
          buf << value << (i==N-1?"":",");
        }
      }
    }
  }
  if (config_.write_pgeom) {
    buf << ";";
    if (!result.opt_candidate_path.empty()) {
      // If original_indices is available, output aligned with original trajectory points
      if (!result.original_indices.empty()) {
        int total_points = *std::max_element(result.original_indices.begin(), result.original_indices.end()) + 1;
        std::vector<std::string> output_values(total_points, "");
        for (size_t i = 0; i < result.original_indices.size(); ++i) {
          int original_idx = result.original_indices[i];
          if (original_idx >= 0 && original_idx < total_points && i < result.opt_candidate_path.size()) {
            const FMM::CORE::Point &point = result.opt_candidate_path[i].c.point;
            std::stringstream ss;
            ss << boost::geometry::get<0>(point) << " " << boost::geometry::get<1>(point);
            output_values[original_idx] = ss.str();
          }
        }
        // Build LineString WKT
        buf << "LINESTRING(";
        bool first = true;
        for (int i = 0; i < total_points; ++i) {
          if (!output_values[i].empty()) {
            if (!first) buf << ",";
            buf << output_values[i];
            first = false;
          }
        }
        buf << ")";
      } else {
        // Fallback to original behavior when original_indices is not available
        int N = result.opt_candidate_path.size();
        FMM::CORE::LineString pline;
        for (int i = 0; i < N; ++i) {
          const FMM::CORE::Point &point = result.opt_candidate_path[i].c.point;
          pline.add_point(point);
        }
        buf << pline;
      }
    }
  }
  // Write fields related with cpath
  if (config_.write_cpath) {
    buf << ";" << result.cpath;
  }
  if (config_.write_tpath) {
    buf << ";";
    if (!result.cpath.empty()) {
      // Iterate through consecutive indexes and write the traversed path
      int J = result.indices.size();
      for (int j = 0; j < J - 1; ++j) {
        int a = result.indices[j];
        int b = result.indices[j + 1];
        for (int i = a; i < b; ++i) {
          buf << result.cpath[i];
          buf << ",";
        }
        buf << result.cpath[b];
        if (j < J - 2) {
          // Last element should not have a bar
          buf << "|";
        }
      }
    }
  }
  if (config_.write_mgeom) {
    buf << ";" << result.mgeom;
  }
  if (config_.write_ep) {
    buf << ";";
    if (!result.opt_candidate_path.empty()) {
      // If original_indices is available, output aligned with original trajectory points
      if (!result.original_indices.empty()) {
        int total_points = *std::max_element(result.original_indices.begin(), result.original_indices.end()) + 1;
        std::vector<double> output_values(total_points, 0.0);
        for (size_t i = 0; i < result.original_indices.size(); ++i) {
          int original_idx = result.original_indices[i];
          if (original_idx >= 0 && original_idx < total_points && i < result.opt_candidate_path.size()) {
            output_values[original_idx] = result.opt_candidate_path[i].ep;
          }
        }
        for (int i = 0; i < total_points; ++i) {
          buf << output_values[i] << (i==total_points-1?"":",");
        }
      } else {
        // Fallback to original behavior when original_indices is not available
        int N = result.opt_candidate_path.size();
        for (int i = 0; i < N - 1; ++i) {
          buf << result.opt_candidate_path[i].ep << ",";
        }
        buf << result.opt_candidate_path[N - 1].ep;
      }
    }
  }
  if (config_.write_tp) {
    buf << ";";
    if (!result.opt_candidate_path.empty()) {
      // If original_indices is available, output aligned with original trajectory points
      if (!result.original_indices.empty()) {
        int total_points = *std::max_element(result.original_indices.begin(), result.original_indices.end()) + 1;
        std::vector<double> output_values(total_points, -1.0);
        for (size_t i = 0; i < result.original_indices.size(); ++i) {
          int original_idx = result.original_indices[i];
          if (original_idx >= 0 && original_idx < total_points && i < result.opt_candidate_path.size()) {
            output_values[original_idx] = result.opt_candidate_path[i].tp;
          }
        }
        for (int i = 0; i < total_points; ++i) {
          buf << output_values[i] << (i==total_points-1?"":",");
        }
      } else {
        // Fallback to original behavior when original_indices is not available
        int N = result.opt_candidate_path.size();
        for (int i = 0; i < N - 1; ++i) {
          buf << result.opt_candidate_path[i].tp << ",";
        }
        buf << result.opt_candidate_path[N - 1].tp;
      }
    }
  }
  if (config_.write_trustworthiness) {
    buf << ";";
    if (!result.opt_candidate_path.empty()) {
      // If original_indices is available, output aligned with original trajectory points
      if (!result.original_indices.empty()) {
        int total_points = *std::max_element(result.original_indices.begin(), result.original_indices.end()) + 1;
        std::vector<double> output_values(total_points, -999.0);
        for (size_t i = 0; i < result.original_indices.size(); ++i) {
          int original_idx = result.original_indices[i];
          if (original_idx >= 0 && original_idx < total_points && i < result.opt_candidate_path.size()) {
            output_values[original_idx] = result.opt_candidate_path[i].trustworthiness;
          }
        }
        for (int i = 0; i < total_points; ++i) {
          buf << output_values[i] << (i==total_points-1?"":",");
        }
      } else {
        // Fallback to original behavior when original_indices is not available
        int N = result.opt_candidate_path.size();
        for (int i = 0; i < N; ++i) {
          const auto &matched = result.opt_candidate_path[i];
          buf << matched.trustworthiness;
          if (i != N - 1) {
            buf << ",";
          }
        }
      }
    }
  }
  if (config_.write_n_best_trustworthiness) {
    buf << ";";
    const auto &nbest = result.nbest_trustworthiness;
    if (!nbest.empty()) {
      // If original_indices is available, output aligned with original trajectory points
      if (!result.original_indices.empty()) {
        int total_points = *std::max_element(result.original_indices.begin(), result.original_indices.end()) + 1;
        std::vector<std::string> output_values(total_points, "()");
        for (size_t i = 0; i < result.original_indices.size(); ++i) {
          int original_idx = result.original_indices[i];
          if (original_idx >= 0 && original_idx < total_points && i < nbest.size()) {
            std::stringstream ss;
            ss << "(";
            const auto &scores = nbest[i];
            for (size_t j = 0; j < scores.size(); ++j) {
              ss << scores[j];
              if (j + 1 < scores.size()) {
                ss << ",";
              }
            }
            ss << ")";
            output_values[original_idx] = ss.str();
          }
        }
        for (int i = 0; i < total_points; ++i) {
          buf << output_values[i] << (i==total_points-1?"":",");
        }
      } else {
        // Fallback to original behavior when original_indices is not available
        for (size_t i = 0; i < nbest.size(); ++i) {
          buf << "(";
          const auto &scores = nbest[i];
          for (size_t j = 0; j < scores.size(); ++j) {
            buf << scores[j];
            if (j + 1 < scores.size()) {
              buf << ",";
            }
          }
          buf << ")";
          if (i + 1 < nbest.size()) {
            buf << "|";
          }
        }
      }
    }
  }
  if (config_.write_cumu_prob) {
    buf << ";";
    if (!result.opt_candidate_path.empty()) {
      // If original_indices is available, output aligned with original trajectory points
      if (!result.original_indices.empty()) {
        int total_points = *std::max_element(result.original_indices.begin(), result.original_indices.end()) + 1;
        std::vector<double> output_values(total_points, -999.0);
        for (size_t i = 0; i < result.original_indices.size(); ++i) {
          int original_idx = result.original_indices[i];
          if (original_idx >= 0 && original_idx < total_points && i < result.opt_candidate_path.size()) {
            output_values[original_idx] = result.opt_candidate_path[i].cumu_prob;
          }
        }
        for (int i = 0; i < total_points; ++i) {
          buf << output_values[i] << (i==total_points-1?"":",");
        }
      } else {
        // Fallback to original behavior when original_indices is not available
        int N = result.opt_candidate_path.size();
        for (int i = 0; i < N - 1; ++i) {
          buf << result.opt_candidate_path[i].cumu_prob << ",";
        }
        buf << result.opt_candidate_path[N - 1].cumu_prob;
      }
    }
  }
  if (config_.write_candidates) {
    buf << ";";
    if (!result.candidate_details.empty()) {
      // If original_indices is available, output aligned with original trajectory points
      if (!result.original_indices.empty()) {
        int total_points = *std::max_element(result.original_indices.begin(), result.original_indices.end()) + 1;
        std::vector<std::string> output_values(total_points, "()");
        for (size_t i = 0; i < result.original_indices.size(); ++i) {
          int original_idx = result.original_indices[i];
          if (original_idx >= 0 && original_idx < total_points && i < result.candidate_details.size()) {
            std::stringstream ss;
            ss << "(";
            const auto &list = result.candidate_details[i];
            for (size_t j = 0; j < list.size(); ++j) {
              const auto &cand = list[j];
              ss << "(" << cand.x << "," << cand.y << "," << cand.ep << ")";
              if (j + 1 < list.size()) ss << ",";
            }
            ss << ")";
            output_values[original_idx] = ss.str();
          }
        }
        for (int i = 0; i < total_points; ++i) {
          buf << output_values[i] << (i==total_points-1?"":",");
        }
      } else {
        // Fallback to original behavior when original_indices is not available
        for (size_t i = 0; i < result.candidate_details.size(); ++i) {
          buf << "(";
          const auto &list = result.candidate_details[i];
          for (size_t j = 0; j < list.size(); ++j) {
            const auto &cand = list[j];
            buf << "(" << cand.x << "," << cand.y << "," << cand.ep << ")";
            if (j + 1 < list.size()) buf << ",";
          }
          buf << ")";
          if (i + 1 < result.candidate_details.size()) buf << "|";
        }
      }
    }
  }
  if (config_.write_length) {
    buf << ";";
    if (!result.opt_candidate_path.empty()) {
      // If original_indices is available, output aligned with original trajectory points
      if (!result.original_indices.empty()) {
        int total_points = *std::max_element(result.original_indices.begin(), result.original_indices.end()) + 1;
        std::vector<double> output_values(total_points, -1.0);
        for (size_t i = 0; i < result.original_indices.size(); ++i) {
          int original_idx = result.original_indices[i];
          if (original_idx >= 0 && original_idx < total_points && i < result.opt_candidate_path.size()) {
            output_values[original_idx] = result.opt_candidate_path[i].c.edge->length;
          }
        }
        for (int i = 0; i < total_points; ++i) {
          buf << output_values[i] << (i==total_points-1?"":",");
        }
      } else {
        // Fallback to original behavior when original_indices is not available
        int N = result.opt_candidate_path.size();
        SPDLOG_TRACE("Write length for {} edges",N);
        for (int i = 0; i < N - 1; ++i) {
          // SPDLOG_TRACE("Write length {}",i);
          buf << result.opt_candidate_path[i].c.edge->length << ",";
        }
        // SPDLOG_TRACE("Write length {}",N-1);
        buf << result.opt_candidate_path[N - 1].c.edge->length;
      }
    }
  }
  if (config_.write_duration) {
    buf << ";";
    if (!traj.timestamps.empty()) {
      // If original_indices is available, output aligned with original trajectory points
      if (!result.original_indices.empty()) {
        int total_points = *std::max_element(result.original_indices.begin(), result.original_indices.end()) + 1;
        std::vector<double> output_values(total_points, -1.0);
        for (size_t i = 0; i < result.original_indices.size(); ++i) {
          int original_idx = result.original_indices[i];
          if (original_idx >= 0 && original_idx < total_points) {
            double value = -1.0;
            if (original_idx == 0) {
              value = 0.0;
            } else if (original_idx > 0 && original_idx < static_cast<int>(traj.timestamps.size())) {
              value = traj.timestamps[original_idx] - traj.timestamps[original_idx - 1];
            }
            output_values[original_idx] = value;
          }
        }
        for (int i = 0; i < total_points; ++i) {
          buf << output_values[i] << (i==total_points-1?"":",");
        }
      } else {
        // Fallback to original behavior when original_indices is not available
        int N = traj.timestamps.size();
        SPDLOG_TRACE("Write duration for {} points",N);
        for (int i = 1; i < N; ++i) {
          // SPDLOG_TRACE("Write length {}",i);
          buf << traj.timestamps[i] - traj.timestamps[i-1]
              << (i==N-1?"":",");
        }
      }
    }
  }
  if (config_.write_speed) {
    buf << ";";
    if (!result.opt_candidate_path.empty() && !traj.timestamps.empty()) {
      // If original_indices is available, output aligned with original trajectory points
      if (!result.original_indices.empty()) {
        int total_points = *std::max_element(result.original_indices.begin(), result.original_indices.end()) + 1;
        std::vector<double> output_values(total_points, -1.0);
        for (size_t i = 0; i < result.original_indices.size(); ++i) {
          int original_idx = result.original_indices[i];
          if (original_idx >= 0 && original_idx < total_points) {
            double value = -1.0;
            if (original_idx == 0) {
              value = 0.0;
            } else if (original_idx > 0 && i < result.opt_candidate_path.size() &&
                       original_idx < static_cast<int>(traj.timestamps.size())) {
              double duration = traj.timestamps[original_idx] - traj.timestamps[original_idx - 1];
              double sp_dist = result.opt_candidate_path[i].sp_dist;
              value = (duration > 0.0) ? (sp_dist / duration) : 0.0;
            }
            output_values[original_idx] = value;
          }
        }
        for (int i = 0; i < total_points; ++i) {
          buf << output_values[i] << (i==total_points-1?"":",");
        }
      } else {
        // Fallback to original behavior when original_indices is not available
        int N = traj.timestamps.size();
        for (int i = 1; i < N; ++i) {
          double duration = traj.timestamps[i] - traj.timestamps[i-1];
          buf << (duration>0?(result.opt_candidate_path[i].sp_dist/duration):0)
              << (i==N-1?"":",");
        }
      }
    }
  }
  if (config_.write_timestamp) {
    buf << ";";
    if (!traj.timestamps.empty()) {
      // If original_indices is available, output aligned with original trajectory points
      if (!result.original_indices.empty()) {
        int total_points = *std::max_element(result.original_indices.begin(), result.original_indices.end()) + 1;
        std::vector<double> output_values(total_points, -1.0);
        for (size_t i = 0; i < result.original_indices.size(); ++i) {
          int original_idx = result.original_indices[i];
          if (original_idx >= 0 && original_idx < total_points && original_idx < static_cast<int>(traj.timestamps.size())) {
            output_values[original_idx] = traj.timestamps[original_idx];
          }
        }
        for (int i = 0; i < total_points; ++i) {
          buf << output_values[i] << (i==total_points-1?"":",");
        }
      } else {
        // Fallback to original behavior when original_indices is not available
        int N = traj.timestamps.size();
        for (int i = 0; i < N; ++i) {
          buf << traj.timestamps[i] << (i==N-1?"":",");
        }
      }
    }
  }
  buf << '\n';
  // Ensure that fstream is called corrected in OpenMP
  #pragma omp critical
  m_fstream << buf.rdbuf();
}

void CSVMatchResultWriter::write_results(
    const std::vector<std::pair<FMM::CORE::Trajectory, FMM::MM::MatchResult>> &results) {
  // Sort results by trajectory ID
  std::vector<std::pair<FMM::CORE::Trajectory, FMM::MM::MatchResult>> sorted_results = results;
  std::sort(sorted_results.begin(), sorted_results.end(),
    [](const std::pair<FMM::CORE::Trajectory, FMM::MM::MatchResult> &a,
       const std::pair<FMM::CORE::Trajectory, FMM::MM::MatchResult> &b) {
      return a.first.id < b.first.id;
    });

  // Write all results in sorted order
  for (const auto &item : sorted_results) {
    write_result(item.first, item.second);
  }
}

} //IO
} //MM
