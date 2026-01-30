//
// Created by Can Yang on 2020/4/1.
//

#include "mm/fmm/ubodt_gen_algorithm.hpp"
#include "mm/fmm/ubodt.hpp"
#include "util/debug.hpp"
#include <omp.h>
#include <atomic>
#include <vector>

using namespace FMM;
using namespace FMM::CORE;
using namespace FMM::NETWORK;
using namespace FMM::MM;

std::string UBODTGenAlgorithm::generate_ubodt(
  const std::string &filename, double delta,
  bool binary, bool use_omp) const {
  std::ostringstream oss;
  std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();
  if (use_omp){
    precompute_ubodt_omp(filename, delta, binary);
  } else {
    precompute_ubodt_single_thead(filename, delta, binary);
  }
  std::chrono::steady_clock::time_point end =
      std::chrono::steady_clock::now();
  double time_spent =
      std::chrono::duration_cast<std::chrono::milliseconds>
          (end - begin).count() / 1000.;
  oss<< "Status: success\n";
  oss<< "Time takes " << time_spent << " seconds\n";
  return oss.str();
};

void UBODTGenAlgorithm::precompute_ubodt_single_thead(
    const std::string &filename, double delta, bool binary) const {
  int num_vertices = ng_.get_num_vertices();
  int step_size = num_vertices / 10;
  if (step_size < 10) step_size = 10;

  // Use network center as heuristic target for A* algorithm
  NodeIndex heuristic_target = num_vertices / 2;
  SPDLOG_INFO("Using A* algorithm with heuristic target node {}", heuristic_target);

  std::ofstream myfile(filename);
  SPDLOG_INFO("Start to generate UBODT with delta {}", delta);
  SPDLOG_INFO("Output format {}", (binary ? "binary" : "csv"));
  if (binary) {
    boost::archive::binary_oarchive oa(myfile);
    for (NodeIndex source = 0; source < num_vertices; ++source) {
      if (source % step_size == 0)
        SPDLOG_INFO("Progress {} / {}", source, num_vertices);
      PredecessorMap pmap;
      DistanceMap dmap;
      ng_.single_source_upperbound_astar(source, delta, &pmap, &dmap, heuristic_target);
      write_result_binary(oa, source, pmap, dmap);
    }
  } else {
    myfile << "source;target;next_n;prev_n;next_e;distance\n";
    for (NodeIndex source = 0; source < num_vertices; ++source) {
      if (source % step_size == 0)
        SPDLOG_INFO("Progress {} / {}", source, num_vertices);
      PredecessorMap pmap;
      DistanceMap dmap;
      ng_.single_source_upperbound_astar(source, delta, &pmap, &dmap, heuristic_target);
      write_result_csv(myfile, source, pmap, dmap);
    }
  }
  myfile.close();
}

// Parallelly generate ubodt using OpenMP
void UBODTGenAlgorithm::precompute_ubodt_omp(
    const std::string &filename, double delta,
    bool binary) const {
  int num_vertices = ng_.get_num_vertices();
  int step_size = num_vertices / 20;
  if (step_size < 10) step_size = 10;

  // Use network center as heuristic target for A* algorithm
  NodeIndex heuristic_target = num_vertices / 2;
  SPDLOG_INFO("Using A* algorithm with heuristic target node {}", heuristic_target);

  // Set OpenMP to use all available cores
  omp_set_num_threads(omp_get_num_procs());
  int num_threads = omp_get_max_threads();
  SPDLOG_INFO("Using {} OpenMP threads", num_threads);

  SPDLOG_INFO("Start to generate UBODT with delta {}", delta);
  SPDLOG_INFO("Output format {}", (binary ? "binary" : "csv"));

  if (binary) {
    // For binary output, use thread-local buffers and serialize sequentially
    std::vector<std::vector<Record>> thread_records(num_threads);
    std::vector<int> thread_progress(num_threads, 0);
    std::atomic<int> completed_sources{0};

#pragma omp parallel
    {
      int thread_id = omp_get_thread_num();

#pragma omp for schedule(dynamic)
      for (int source = 0; source < num_vertices; ++source) {
        const NodeIndex source_idx = static_cast<NodeIndex>(source);
        PredecessorMap pmap;
        DistanceMap dmap;
        ng_.single_source_upperbound_astar(source_idx, delta, &pmap, &dmap, heuristic_target);

        // Collect records in thread-local storage
        for (auto iter = pmap.begin(); iter != pmap.end(); ++iter) {
          NodeIndex cur_node = iter->first;
          if (cur_node != source_idx) {
            NodeIndex prev_node = iter->second;
            NodeIndex v = cur_node;
            NodeIndex u;
            while ((u = pmap[v]) != source_idx) {
              v = u;
            }
            NodeIndex successor = v;
            double cost = dmap[successor];
            EdgeIndex edge_index = ng_.get_edge_index(source_idx, successor, cost);
            thread_records[thread_id].push_back(
                {source_idx,
                 cur_node,
                 successor,
                 prev_node,
                 edge_index,
                 dmap[cur_node],
                 nullptr});
          }
        }

        thread_progress[thread_id]++;
        completed_sources++;

        // Master thread reports progress
        if (thread_id == 0 && completed_sources % step_size == 0) {
          SPDLOG_INFO("Progress {} / {} ({:.1f}%)", completed_sources.load(),
                     num_vertices, 100.0 * completed_sources.load() / num_vertices);
        }
      }
    }

    // Write all records to file using boost serialization
    std::ofstream myfile(filename);
    boost::archive::binary_oarchive oa(myfile);
    for (int i = 0; i < num_threads; ++i) {
      for (Record &r : thread_records[i]) {
        NETWORK::NodeIndex source = r.source;
        NETWORK::NodeIndex target = r.target;
        NETWORK::NodeIndex first_n = r.first_n;
        NETWORK::NodeIndex prev_n = r.prev_n;
        NETWORK::EdgeIndex next_e = r.next_e;
        double cost = r.cost;
        oa << source << target << first_n << prev_n << next_e << cost;
      }
    }
    myfile.close();

  } else {
    // For CSV output, use thread-local buffers to avoid I/O contention
    std::vector<std::stringstream> thread_buffers(num_threads);
    std::vector<int> thread_progress(num_threads, 0);
    std::atomic<int> completed_sources{0};

#pragma omp parallel
    {
      int thread_id = omp_get_thread_num();

      // Only master thread writes header
      if (thread_id == 0) {
        thread_buffers[thread_id] << "source;target;next_n;prev_n;next_e;distance\n";
      }

#pragma omp for schedule(dynamic)
      for (int source = 0; source < num_vertices; ++source) {
        const NodeIndex source_idx = static_cast<NodeIndex>(source);
        PredecessorMap pmap;
        DistanceMap dmap;
        ng_.single_source_upperbound_astar(source_idx, delta, &pmap, &dmap, heuristic_target);

        // Write to thread-local buffer
        write_result_csv_buffer(thread_buffers[thread_id], source_idx, pmap, dmap);

        thread_progress[thread_id]++;
        completed_sources++;

        // Master thread reports progress
        if (thread_id == 0 && completed_sources % step_size == 0) {
          SPDLOG_INFO("Progress {} / {} ({:.1f}%)", completed_sources.load(),
                     num_vertices, 100.0 * completed_sources.load() / num_vertices);
        }
      }
    }

    // Write all buffers to file
    std::ofstream myfile(filename);
    for (int i = 0; i < num_threads; ++i) {
      myfile << thread_buffers[i].str();
    }
    myfile.close();
  }

  SPDLOG_INFO("UBODT generation completed successfully");
}

/**
   * Write the result of routing from a single source node
   * @param stream output stream
   * @param s      source node
   * @param pmap   predecessor map
   * @param dmap   distance map
   */
void UBODTGenAlgorithm::write_result_csv(
    std::ostream &stream, NodeIndex s,
                                   PredecessorMap &pmap, DistanceMap &dmap) const {
  std::vector<Record> source_map;
  for (auto iter = pmap.begin(); iter != pmap.end(); ++iter) {
    NodeIndex cur_node = iter->first;
    if (cur_node != s) {
      NodeIndex prev_node = iter->second;
      NodeIndex v = cur_node;
      NodeIndex u;
      while ((u = pmap[v]) != s) {
        v = u;
      }
      NodeIndex successor = v;
      // Write the result to source map
      double cost = dmap[successor];
      EdgeIndex edge_index = ng_.get_edge_index(s, successor, cost);
      source_map.push_back(
          {s,
           cur_node,
           successor,
           prev_node,
           edge_index,
           dmap[cur_node],
           nullptr});
    }
  }
#pragma omp critical
  for (Record &r:source_map) {
    stream << r.source << ";"
           << r.target << ";"
           << r.first_n << ";"
           << r.prev_n << ";"
           << r.next_e << ";"
           << r.cost << "\n";
  }
}

/**
   * Write the result of routing from a single source node to buffer (thread-safe)
   * @param stream output buffer
   * @param s      source node
   * @param pmap   predecessor map
   * @param dmap   distance map
   */
void UBODTGenAlgorithm::write_result_csv_buffer(
    std::ostream &stream, NodeIndex s,
    PredecessorMap &pmap, DistanceMap &dmap) const {
  for (auto iter = pmap.begin(); iter != pmap.end(); ++iter) {
    NodeIndex cur_node = iter->first;
    if (cur_node != s) {
      NodeIndex prev_node = iter->second;
      NodeIndex v = cur_node;
      NodeIndex u;
      while ((u = pmap[v]) != s) {
        v = u;
      }
      NodeIndex successor = v;
      double cost = dmap[successor];
      EdgeIndex edge_index = ng_.get_edge_index(s, successor, cost);
      stream << s << ";"
             << cur_node << ";"
             << successor << ";"
             << prev_node << ";"
             << edge_index << ";"
             << dmap[cur_node] << "\n";
    }
  }
}

/**
 * Write the result of routing from a single source node in
 * binary format
 *
 * @param stream output stream
 * @param s      source node
 * @param pmap   predecessor map
 * @param dmap   distance map
 */
void UBODTGenAlgorithm::write_result_binary(boost::archive::binary_oarchive &stream,
                                      NodeIndex s,
                                      PredecessorMap &pmap,
                                      DistanceMap &dmap) const {
  std::vector<Record> source_map;
  for (auto iter = pmap.begin(); iter != pmap.end(); ++iter) {
    NodeIndex cur_node = iter->first;
    if (cur_node != s) {
      NodeIndex prev_node = iter->second;
      NodeIndex v = cur_node;
      NodeIndex u;
      // When u=s, v is the next node visited
      while ((u = pmap[v]) != s) {
        v = u;
      }
      NodeIndex successor = v;
      // Write the result
      double cost = dmap[successor];
      EdgeIndex edge_index = ng_.get_edge_index(s, successor, cost);
      source_map.push_back(
          {s,
           cur_node,
           successor,
           prev_node,
           edge_index,
           dmap[cur_node],
           nullptr});
    }
  }
#pragma omp critical
  for (Record &r:source_map) {
    NETWORK::NodeIndex source = r.source;
    NETWORK::NodeIndex target = r.target;
    NETWORK::NodeIndex first_n = r.first_n;
    NETWORK::NodeIndex prev_n = r.prev_n;
    NETWORK::EdgeIndex next_e = r.next_e;
    double cost = r.cost;
    stream << source << target
           << first_n << prev_n << next_e << cost;
  }
}

/**
 * Write the result of routing from a single source node to binary buffer (thread-safe)
 * @param stream output buffer
 * @param s      source node
 * @param pmap   predecessor map
 * @param dmap   distance map
 */
void UBODTGenAlgorithm::write_result_binary_buffer(std::ostream &stream,
                                      NodeIndex s,
                                      PredecessorMap &pmap,
                                      DistanceMap &dmap) const {
  for (auto iter = pmap.begin(); iter != pmap.end(); ++iter) {
    NodeIndex cur_node = iter->first;
    if (cur_node != s) {
      NodeIndex prev_node = iter->second;
      NodeIndex v = cur_node;
      NodeIndex u;
      // When u=s, v is the next node visited
      while ((u = pmap[v]) != s) {
        v = u;
      }
      NodeIndex successor = v;
      // Write the result to buffer as serialized data
      double cost = dmap[successor];
      EdgeIndex edge_index = ng_.get_edge_index(s, successor, cost);

      // Write binary data directly to stream
      stream.write(reinterpret_cast<const char*>(&s), sizeof(s));
      stream.write(reinterpret_cast<const char*>(&cur_node), sizeof(cur_node));
      stream.write(reinterpret_cast<const char*>(&successor), sizeof(successor));
      stream.write(reinterpret_cast<const char*>(&prev_node), sizeof(prev_node));
      stream.write(reinterpret_cast<const char*>(&edge_index), sizeof(edge_index));
      stream.write(reinterpret_cast<const char*>(&dmap[cur_node]), sizeof(dmap[cur_node]));
    }
  }
}
