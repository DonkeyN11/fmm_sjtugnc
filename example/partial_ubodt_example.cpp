/**
 * Example: Using Partial UBODT for Dynamic Loading
 *
 * This example demonstrates how to use PartialUBODT to load only
 * the relevant UBODT records based on trajectory extent, significantly
 * reducing memory usage and load time.
 */

#include "mm/fmm/ubodt_partial.hpp"
#include "network/network.hpp"
#include "io/gps_reader.hpp"
#include "config/fmm_config.hpp"
#include "util/debug.hpp"

#include <iostream>
#include <chrono>

using namespace FMM;
using namespace FMM::CORE;
using namespace FMM::NETWORK;
using namespace FMM::MM;
using namespace FMM::IO;

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0]
                  << " <network.shp> <ubodt.bin> <trajectories.csv>\n";
        std::cout << "\nDescription:\n"
                  << "  This program demonstrates partial UBODT loading based on trajectory extent.\n"
                  << "  It only loads UBODT records relevant to the provided trajectories.\n\n"
                  << "Arguments:\n"
                  << "  network.shp    - Road network shapefile\n"
                  << "  ubodt.bin      - UBODT binary file (full or indexed format)\n"
                  << "  trajectories.csv - GPS trajectory file\n";
        return 1;
    }

    std::string network_file = argv[1];
    std::string ubodt_file = argv[2];
    std::string trajectory_file = argv[3];

    // Initialize logging
    spdlog::set_level(spdlog::level::info);
    SPDLOG_INFO("=== Partial UBODT Loading Example ===");

    // Step 1: Load network
    SPDLOG_INFO("Step 1: Loading network from {}", network_file);
    CONFIG::NetworkConfig network_config(network_file, "id", "source", "target");
    Network network(network_config, true);
    SPDLOG_INFO("Network loaded: {} edges, {} vertices",
                network.get_edge_count(), network.get_node_count());

    // Step 2: Read trajectories
    SPDLOG_INFO("Step 2: Reading trajectories from {}", trajectory_file);
    std::vector<Trajectory> trajectories;
    TrajectoryReader reader(trajectory_file, "id", "geom");
    while (reader.has_next_trajectory()) {
        trajectories.push_back(reader.read_next_trajectory());
    }
    SPDLOG_INFO("Loaded {} trajectories", trajectories.size());

    // Step 3: Create partial UBODT from trajectories
    SPDLOG_INFO("Step 3: Creating partial UBODT from trajectory extent");

    auto start = std::chrono::high_resolution_clock::now();

    // Option A: Load from trajectories (automatic bbox calculation)
    auto partial_ubodt = make_partial_ubodt_from_trajectories(
        ubodt_file,
        network,
        trajectories,
        0.1  // 10% buffer around trajectory bounding box
    );

    auto end = std::chrono::high_resolution_clock::now();
    double load_time = std::chrono::duration<double>(end - start).count();

    if (!partial_ubodt->is_valid()) {
        SPDLOG_CRITICAL("Failed to load partial UBODT");
        return 1;
    }

    // Step 4: Report statistics
    SPDLOG_INFO("=== Partial UBODT Statistics ===");
    SPDLOG_INFO("Loaded {} records from {} source nodes",
                partial_ubodt->get_num_records(),
                partial_ubodt->get_num_sources());
    SPDLOG_INFO("Load time: {:.2f} seconds", load_time);

    // Step 5: Test lookups
    SPDLOG_INFO("Step 5: Testing UBODT lookups");
    if (partial_ubodt->get_num_records() > 0) {
        // Get some sample nodes from the loaded set
        const auto &edges = network.get_edges();
        if (!edges.empty()) {
            NodeIndex source = edges[0].source;
            NodeIndex target = edges[0].target;

            const Record *rec = partial_ubodt->look_up(source, target);
            if (rec) {
                SPDLOG_INFO("Sample lookup: {} -> {} distance: {}",
                           source, target, rec->cost);
            } else {
                SPDLOG_INFO("Sample lookup: {} -> {} not found (expected if not in bbox)",
                           source, target);
            }
        }
    }

    // Step 6: Compare with full loading (commented out to avoid long wait times)
    // SPDLOG_INFO("Step 6: Comparing with full UBODT loading");
    // auto start_full = std::chrono::high_resolution_clock::now();
    // auto full_ubodt = UBODT::read_ubodt_file(ubodt_file);
    // auto end_full = std::chrono::high_resolution_clock::now();
    // double full_load_time = std::chrono::duration<double>(end_full - start_full).count();
    // SPDLOG_INFO("Full UBODT: {} records, load time: {:.2f} seconds",
    //             full_ubodt->get_num_rows(), full_load_time);
    // SPDLOG_INFO("Reduction: {:.1f}% records, {:.1f}% faster",
    //             100.0 * (1.0 - double(partial_ubodt->get_num_records()) / full_ubodt->get_num_rows()),
    //             100.0 * (1.0 - load_time / full_load_time));

    SPDLOG_INFO("=== Example completed successfully ===");

    return 0;
}
