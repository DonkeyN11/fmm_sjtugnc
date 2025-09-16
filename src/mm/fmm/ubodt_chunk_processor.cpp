//
// Created by Claude for chunk-based parallel UBODT generation
//

#include "mm/fmm/ubodt_chunk_processor.hpp"
#include "mm/fmm/ubodt.hpp"
#include "util/debug.hpp"
#include <omp.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

using namespace FMM;
using namespace FMM::CORE;
using namespace FMM::NETWORK;
using namespace FMM::MM;

UBODTChunkProcessor::UBODTChunkProcessor(const Network &network,
                                         const NetworkGraph &graph,
                                         double delta)
    : network_(network), graph_(graph), delta_(delta) {}

BoundingBox UBODTChunkProcessor::get_network_bounds() const {
    double min_x = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double min_y = std::numeric_limits<double>::max();
    double max_y = std::numeric_limits<double>::lowest();

    int num_vertices = graph_.get_num_vertices();
    for (NodeIndex i = 0; i < num_vertices; ++i) {
        auto point = network_.get_vertex_point(i);
        min_x = std::min(min_x, boost::geometry::get<0>(point));
        max_x = std::max(max_x, boost::geometry::get<0>(point));
        min_y = std::min(min_y, boost::geometry::get<1>(point));
        max_y = std::max(max_y, boost::geometry::get<1>(point));
    }

    return BoundingBox(min_x, max_x, min_y, max_y);
}

std::vector<NodeIndex> UBODTChunkProcessor::find_nodes_in_bounds(
    const BoundingBox& bounds) const {
    std::vector<NodeIndex> nodes;
    int num_vertices = graph_.get_num_vertices();

    for (NodeIndex i = 0; i < num_vertices; ++i) {
        auto point = network_.get_vertex_point(i);
        if (bounds.expand(delta_).contains(point)) {
            nodes.push_back(i);
        }
    }

    return nodes;
}

void UBODTChunkProcessor::identify_boundary_nodes(NetworkChunk& chunk) const {
    chunk.boundary_nodes.clear();

    for (NodeIndex node : chunk.nodes) {
        auto point = network_.get_vertex_point(node);

        // Check if node is near the boundary of the original (unexpanded) chunk
        double margin = delta_ * 0.1;  // 10% of delta as boundary margin
        if (boost::geometry::get<0>(point) <= chunk.bounds.min_x + margin ||
            boost::geometry::get<0>(point) >= chunk.bounds.max_x - margin ||
            boost::geometry::get<1>(point) <= chunk.bounds.min_y + margin ||
            boost::geometry::get<1>(point) >= chunk.bounds.max_y - margin) {
            chunk.boundary_nodes.push_back(node);
        }
    }
}

std::vector<NetworkChunk> UBODTChunkProcessor::create_grid_chunks(
    int rows, int cols) {
    std::vector<NetworkChunk> chunks;
    auto bounds = get_network_bounds();

    double width = bounds.max_x - bounds.min_x;
    double height = bounds.max_y - bounds.min_y;
    double chunk_width = width / cols;
    double chunk_height = height / rows;

    SPDLOG_INFO("Network bounds: ({:.2f},{:.2f}) to ({:.2f},{:.2f})",
                bounds.min_x, bounds.min_y, bounds.max_x, bounds.max_y);
    SPDLOG_INFO("Creating {}x{} grid chunks, each {:.2f}x{:.2f} units",
                rows, cols, chunk_width, chunk_height);

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            double min_x = bounds.min_x + col * chunk_width;
            double max_x = bounds.min_x + (col + 1) * chunk_width;
            double min_y = bounds.min_y + row * chunk_height;
            double max_y = bounds.min_y + (row + 1) * chunk_height;

            BoundingBox chunk_bounds(min_x, max_x, min_y, max_y);
            NetworkChunk chunk(row * cols + col, chunk_bounds, delta_);

            // Find nodes in this chunk (with delta buffer)
            chunk.nodes = find_nodes_in_bounds(chunk.bounds);

            // Identify boundary nodes
            identify_boundary_nodes(chunk);

            SPDLOG_INFO("Chunk {}: {} nodes, {} boundary nodes",
                       chunk.chunk_id, chunk.nodes.size(), chunk.boundary_nodes.size());

            chunks.push_back(chunk);
        }
    }

    return chunks;
}

void UBODTChunkProcessor::process_chunk(const NetworkChunk& chunk,
                                      const std::string& temp_filename,
                                      bool binary) {
    SPDLOG_INFO("Processing chunk {} with {} nodes", chunk.chunk_id, chunk.nodes.size());

    std::vector<Record> all_records;
    int processed_nodes = 0;
    int step_size = std::max(10, static_cast<int>(chunk.nodes.size() / 20));

    for (NodeIndex source : chunk.nodes) {
        PredecessorMap pmap;
        DistanceMap dmap;

        // Use existing Dijkstra implementation
        graph_.single_source_upperbound_dijkstra(source, delta_, &pmap, &dmap);

        // Convert results to records
        for (auto iter = pmap.begin(); iter != pmap.end(); ++iter) {
            NodeIndex target = iter->first;
            if (target != source) {
                NodeIndex prev_node = iter->second;
                NodeIndex v = target;
                NodeIndex u;

                // Find the successor node
                while ((u = pmap[v]) != source) {
                    v = u;
                }
                NodeIndex successor = v;

                double cost = dmap[successor];
                EdgeIndex edge_index = 0; // Placeholder - edge ID not needed for UBODT

                all_records.push_back({
                    source, target, successor, prev_node,
                    edge_index, dmap[target], nullptr
                });
            }
        }

        processed_nodes++;
        if (processed_nodes % step_size == 0) {
            SPDLOG_INFO("Chunk {} progress: {} / {} ({:.1f}%)",
                       chunk.chunk_id, processed_nodes, chunk.nodes.size(),
                       100.0 * processed_nodes / chunk.nodes.size());
        }
    }

    // Write results to temporary file
    write_chunk_results(chunk, all_records, temp_filename, binary);
    SPDLOG_INFO("Chunk {} completed: {} records generated", chunk.chunk_id, all_records.size());
}

void UBODTChunkProcessor::write_chunk_results(const NetworkChunk& chunk,
                                             const std::vector<Record>& records,
                                             const std::string& filename,
                                             bool binary) const {
    std::ofstream myfile(filename);

    if (binary) {
        boost::archive::binary_oarchive oa(myfile);
        for (const Record& r : records) {
            oa << r.source << r.target << r.first_n << r.prev_n << r.next_e << r.cost;
        }
    } else {
        // Only write header if this is the first chunk (will be handled during merge)
        for (const Record& r : records) {
            myfile << r.source << ";" << r.target << ";" << r.first_n << ";"
                   << r.prev_n << ";" << r.next_e << ";" << r.cost << "\n";
        }
    }

    myfile.close();
}

void UBODTChunkProcessor::merge_chunk_results(
    const std::vector<std::string>& chunk_files,
    const std::string& final_filename,
    bool binary) {

    SPDLOG_INFO("Merging {} chunk files into {}", chunk_files.size(), final_filename);

    if (binary) {
        // For binary format, simply concatenate
        std::ofstream final_file(final_filename, std::ios::binary);
        for (const auto& chunk_file : chunk_files) {
            std::ifstream input_file(chunk_file, std::ios::binary);
            final_file << input_file.rdbuf();
            input_file.close();

            // Clean up temporary file
            std::filesystem::remove(chunk_file);
        }
        final_file.close();
    } else {
        // For CSV format, write header and concatenate
        std::ofstream final_file(final_filename);
        final_file << "source;target;next_n;prev_n;next_e;distance\n";

        for (const auto& chunk_file : chunk_files) {
            std::ifstream input_file(chunk_file);
            std::string line;
            while (std::getline(input_file, line)) {
                final_file << line << "\n";
            }
            input_file.close();

            // Clean up temporary file
            std::filesystem::remove(chunk_file);
        }
        final_file.close();
    }

    SPDLOG_INFO("Merge completed");
}

std::string UBODTChunkProcessor::generate_ubodt_chunked(
    const std::string& filename, int chunk_rows, int chunk_cols,
    int num_threads, bool binary) {

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    SPDLOG_INFO("Starting chunk-based UBODT generation");
    SPDLOG_INFO("Grid size: {}x{}, threads: {}", chunk_rows, chunk_cols, num_threads);

    // Create chunks
    auto chunks = create_grid_chunks(chunk_rows, chunk_cols);
    SPDLOG_INFO("Created {} chunks", chunks.size());

    // Set up OpenMP
    omp_set_num_threads(num_threads);

    // Process chunks in parallel
    std::vector<std::string> temp_files(chunks.size());
    std::atomic<int> completed_chunks{0};

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < chunks.size(); ++i) {
        // Create temporary filename
        temp_files[i] = filename + ".tmp_" + std::to_string(chunks[i].chunk_id);

        // Process chunk
        process_chunk(chunks[i], temp_files[i], binary);

        completed_chunks++;
        SPDLOG_INFO("Progress: {}/{} chunks completed ({:.1f}%)",
                   completed_chunks.load(), chunks.size(),
                   100.0 * completed_chunks.load() / chunks.size());
    }

    // Merge results
    merge_chunk_results(temp_files, filename, binary);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double time_spent = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0;

    std::ostringstream oss;
    oss << "Status: success\n";
    oss << "Time takes " << time_spent << " seconds\n";
    oss << "Chunks processed: " << chunks.size() << "\n";
    oss << "Threads used: " << num_threads << "\n";

    return oss.str();
}

// Chunked algorithm implementation
std::string UBODTGenAlgorithmChunked::generate_ubodt_chunked(
    const std::string& filename, double delta, bool binary, bool use_chunking,
    int chunk_rows, int chunk_cols, int num_threads) const {

    if (!use_chunking) {
        // Fall back to original implementation
        return generate_ubodt(filename, delta, binary, true);
    }

    UBODTChunkProcessor processor(network_, ng_, delta);
    return processor.generate_ubodt_chunked(filename, chunk_rows, chunk_cols, num_threads, binary);
}