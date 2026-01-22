/**
 * Batch Matching Example with UBODT Manager
 *
 * Demonstrates how to use UBODTManager to avoid reloading UBODT
 * for multiple matching operations.
 *
 * Compile:
 *   g++ -std=c++17 -O3 batch_match_example.cpp \
 *       -o batch_match_example \
 *       -I../src \
 *       -L../build \
 *       -lFMMLIB \
 *       $(pkg-config --cflags --libs gdal boost)
 *
 * Usage:
 *   ./batch_match_example <network.shp> <ubodt.bin> <traj_file1> [traj_file2 ...]
 */

#include "mm/fmm/ubodt_manager.hpp"
#include "mm/fmm/fmm_algorithm.hpp"
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
using namespace FMM::CONFIG;

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0]
                  << " <network.shp> <ubodt.bin> <traj_file1> [traj_file2 ...]\n\n";
        std::cout << "This tool loads UBODT once and matches multiple trajectory files.\n";
        std::cout << "UBODT is kept in memory until all files are processed.\n\n";
        return 1;
    }

    std::string network_file = argv[1];
    std::string ubodt_file = argv[2];

    spdlog::set_level(spdlog::level::info);

    // ============================================================
    // Step 1: Load Network
    // ============================================================
    std::cout << "========================================\n";
    std::cout << "Batch Matching Tool with UBODT Manager\n";
    std::cout << "========================================\n\n";

    std::cout << "[1/3] Loading network...\n";
    auto net_start = std::chrono::high_resolution_clock::now();

    NetworkConfig network_config(network_file, "id", "source", "target");
    Network network(network_config, true);

    auto net_end = std::chrono::high_resolution_clock::now();
    double net_time = std::chrono::duration<double>(net_end - net_start).count();

    std::cout << "  Network loaded: " << network.get_edge_count() << " edges, "
              << network.get_node_count() << " vertices\n";
    std::cout << "  Time: " << net_time << " seconds\n\n";

    // ============================================================
    // Step 2: Load UBODT (once)
    // ============================================================
    std::cout << "[2/3] Loading UBODT (will be cached for all files)...\n";
    auto ubodt_start = std::chrono::high_resolution_clock::now();

    auto ubodt = UBODTHelper::load_ubodt(ubodt_file, 1, true);

    auto ubodt_end = std::chrono::high_resolution_clock::now();
    double ubodt_time = std::chrono::duration<double>(ubodt_end - ubodt_start).count();

    if (!ubodt) {
        std::cerr << "  ERROR: Failed to load UBODT from " << ubodt_file << "\n";
        return 1;
    }

    std::cout << "  UBODT loaded: " << ubodt->get_num_rows() << " rows\n";
    std::cout << "  Time: " << ubodt_time << " seconds\n";
    std::cout << "  Status: CACHED in memory (will be reused for all files)\n\n";

    // ============================================================
    // Step 3: Process each trajectory file
    // ============================================================
    std::cout << "[3/3] Processing " << (argc - 3) << " trajectory file(s)...\n\n";

    FMMConfig config;
    config.radius = 300;

    int total_files = 0;
    int total_matched = 0;
    double total_match_time = 0;

    for (int i = 3; i < argc; ++i) {
        std::string traj_file = argv[i];

        std::cout << "File [" << total_files + 1 << "/" << (argc - 3) << "]: " << traj_file << "\n";

        try {
            // Read trajectories
            TrajectoryReader reader(traj_file, "id", "geom");
            std::vector<Trajectory> trajectories;

            int count = 0;
            while (reader.has_next_trajectory() && count < 10000) {
                trajectories.push_back(reader.read_next_trajectory());
                ++count;
            }

            std::cout << "  Read " << trajectories.size() << " trajectories\n";

            // Match (UBODT is already in memory - no reload needed!)
            auto match_start = std::chrono::high_resolution_clock::now();

            FMMAlgorithm fmm_algo(network, ubodt);

            int matched = 0;
            for (const auto &traj : trajectories) {
                auto result = fmm_algo.match_traj(traj, config);
                if (!result.cpath.empty()) {
                    ++matched;
                }
            }

            auto match_end = std::chrono::high_resolution_clock::now();
            double match_time = std::chrono::duration<double>(match_end - match_start).count();

            std::cout << "  Matched: " << matched << "/" << trajectories.size() << " trajectories\n";
            std::cout << "  Time: " << match_time << " seconds\n";

            total_files++;
            total_matched += matched;
            total_match_time += match_time;

        } catch (const std::exception &e) {
            std::cerr << "  ERROR: " << e.what() << "\n";
        }

        std::cout << "\n";
    }

    // ============================================================
    // Summary
    // ============================================================
    std::cout << "========================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "========================================\n";
    std::cout << "Files processed: " << total_files << "\n";
    std::cout << "Total matched: " << total_matched << " trajectories\n";
    std::cout << "Total matching time: " << total_match_time << " seconds\n";
    std::cout << "Average time per file: " << (total_match_time / total_files) << " seconds\n";
    std::cout << "\n";
    std::cout << "Breakdown:\n";
    std::cout << "  Network load: " << net_time << " seconds (once)\n";
    std::cout << "  UBODT load: " << ubodt_time << " seconds (once, cached)\n";
    std::cout << "  Matching: " << total_match_time << " seconds (all files)\n";
    std::cout << "  Total: " << (net_time + ubodt_time + total_match_time) << " seconds\n\n";

    // Show UBODT status
    std::cout << "UBODT Status:\n";
    UBODTHelper::print_ubodt_status();

    // ============================================================
    // Cleanup
    // ============================================================
    std::cout << "\nReleasing UBODT from memory...\n";
    size_t released = UBODTHelper::release_all_ubodt();
    std::cout << "Released " << released << " UBODT(s)\n";
    std::cout << "Done!\n";

    return 0;
}
