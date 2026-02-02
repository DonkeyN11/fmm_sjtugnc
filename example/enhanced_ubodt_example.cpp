/**
 * Enhanced UBODT Usage Examples
 *
 * Demonstrates three optimization features:
 * 1. Query Cache (LRU)
 * 2. Batch Processing
 * 3. Incremental Loading
 */

#include "mm/fmm/ubodt_enhanced.hpp"
#include "mm/fmm/fmm_algorithm.hpp"
#include "network/network.hpp"
#include "io/gps_reader.hpp"
#include "config/fmm_config.hpp"
#include "util/debug.hpp"

#include <iostream>
#include <iomanip>

using namespace FMM;
using namespace FMM::CORE;
using namespace FMM::NETWORK;
using namespace FMM::MM;
using namespace FMM::IO;

// ============================================================================
// Example 1: Query Cache
// ============================================================================

void example_query_cache(const Network &network, std::shared_ptr<UBODT> ubodt) {
    SPDLOG_INFO("=== Example 1: Query Cache ===");

    // Wrap UBODT with cache (10,000 entries)
    CachedUBODT cached_ubodt(ubodt, 10000);

    // Simulate some queries
    const auto &edges = network.get_edges();
    if (edges.empty()) return;

    // Perform 1000 queries (with many duplicates to demonstrate caching)
    for (int i = 0; i < 1000; ++i) {
        NodeIndex source = edges[i % edges.size()].source;
        NodeIndex target = edges[i % edges.size()].target;
        cached_ubodt.look_up(source, target);
    }

    // Get cache statistics
    auto stats = cached_ubodt.get_stats();
    SPDLOG_INFO("Cache Statistics:");
    SPDLOG_INFO("  Hits: {}", stats.hits);
    SPDLOG_INFO("  Misses: {}", stats.misses);
    SPDLOG_INFO("  Size: {} entries", stats.size);
    SPDLOG_INFO("  Hit Rate: {:.2f}%", stats.hit_rate() * 100);
}

// ============================================================================
// Example 2: Batch Processing
// ============================================================================

void example_batch_processing(const Network &network,
                              const std::string &ubodt_file,
                              const std::vector<Trajectory> &trajectories) {
    SPDLOG_INFO("=== Example 2: Batch Processing ===");

    if (trajectories.empty()) {
        SPDLOG_WARN("No trajectories provided for batch processing");
        return;
    }

    // Create batch processor
    BatchUBODTProcessor processor(ubodt_file, network, 0.1);

    // Define processing function (e.g., count candidates per trajectory)
    auto process_func = [&network](const Trajectory &traj,
                                   std::shared_ptr<PartialUBODT> partial_ubodt) {
        // Simulate processing: count how many edges are in the loaded region
        const auto &edges = network.get_edges();
        int count = 0;
        for (const auto &edge : edges) {
            auto rec = partial_ubodt->look_up(edge.source, edge.target);
            if (rec) ++count;
        }
        return count;
    };

    // Option A: Process all trajectories in one batch
    auto results_single = processor.process_batch(trajectories, process_func);

    SPDLOG_INFO("Single batch results:");
    for (size_t i = 0; i < std::min(results_single.size(), size_t(5)); ++i) {
        SPDLOG_INFO("  Trajectory {}: {} edges in region", i, results_single[i]);
    }

    // Option B: Process in groups (for large datasets)
    if (trajectories.size() > 100) {
        auto results_groups = processor.process_groups(trajectories, 50, process_func);

        auto stats = processor.get_last_stats();
        SPDLOG_INFO("Group processing statistics:");
        SPDLOG_INFO("  Total trajectories: {}", stats.total_trajectories);
        SPDLOG_INFO("  Total groups: {}", stats.total_groups);
        SPDLOG_INFO("  Total load time: {:.2f}s", stats.total_load_time);
        SPDLOG_INFO("  Avg load time per group: {:.2f}s", stats.avg_load_time_per_group);
        SPDLOG_INFO("  Avg records per group: {:.0f}", stats.avg_records_per_group);
    }
}

// ============================================================================
// Example 3: Incremental Loading
// ============================================================================

void example_incremental_loading(const Network &network,
                                  const std::string &ubodt_file,
                                  const std::vector<Trajectory> &trajectories) {
    SPDLOG_INFO("=== Example 3: Incremental Loading ===");

    if (trajectories.empty()) return;

    // Create incremental UBODT
    IncrementalUBODT incremental_ubodt(ubodt_file, network, 5000);

    // Split trajectories into chunks to simulate streaming data
    size_t chunk_size = std::min(size_t(10), trajectories.size());

    for (size_t i = 0; i < trajectories.size(); i += chunk_size) {
        size_t end = std::min(i + chunk_size, trajectories.size());
        std::vector<Trajectory> chunk(
            trajectories.begin() + i,
            trajectories.begin() + end
        );

        SPDLOG_INFO("Loading chunk {}/{} ({} trajectories)",
                    i / chunk_size + 1,
                    (trajectories.size() + chunk_size - 1) / chunk_size,
                    chunk.size());

        // Add trajectories to incremental UBODT
        size_t new_nodes = incremental_ubodt.add_trajectories(chunk, 0.1);

        SPDLOG_INFO("  Added {} new nodes (total: {})",
                    new_nodes, incremental_ubodt.get_num_loaded_nodes());
        SPDLOG_INFO("  Total UBODT records: {}", incremental_ubodt.get_num_records());
    }

    // Get load statistics
    auto stats = incremental_ubodt.get_load_stats();
    SPDLOG_INFO("Incremental loading statistics:");
    SPDLOG_INFO("  Total loads: {}", stats.total_loads);
    SPDLOG_INFO("  Total nodes loaded: {}", stats.total_nodes_loaded);
    SPDLOG_INFO("  Total load time: {:.2f}s", stats.total_load_time);
    SPDLOG_INFO("  Avg load time: {:.2f}s", stats.total_load_time / stats.total_loads);
}

// ============================================================================
// Example 4: Combined Usage
// ============================================================================

void example_combined_usage(const Network &network,
                            const std::string &ubodt_file,
                            const std::vector<Trajectory> &trajectories) {
    SPDLOG_INFO("=== Example 4: Combined Usage ===");

    if (trajectories.empty()) return;

    // Create PartialUBODT
    auto partial_ubodt = make_partial_ubodt_from_trajectories(
        ubodt_file, network, trajectories, 0.1
    );

    if (!partial_ubodt->is_valid()) {
        SPDLOG_ERROR("Failed to create PartialUBODT");
        return;
    }

    // Wrap with cache
    CachedUBODT cached_ubodt(partial_ubodt->get_ubodt(), 5000);

    // Use cached UBODT for matching
    int matches = 0;
    for (const auto &traj : trajectories) {
        // Simulate matching: look up some paths
        const auto &edges = network.get_edges();
        for (size_t i = 0; i < std::min(size_t(10), edges.size()); ++i) {
            auto rec = cached_ubodt.look_up(edges[i].source, edges[i].target);
            if (rec) ++matches;
        }
    }

    auto stats = cached_ubodt.get_stats();
    SPDLOG_INFO("Combined usage results:");
    SPDLOG_INFO("  Total matches: {}", matches);
    SPDLOG_INFO("  Cache hit rate: {:.2f}%", stats.hit_rate() * 100);
    SPDLOG_INFO("  PartialUBODT records: {}", partial_ubodt->get_num_records());
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0]
                  << " <network.shp> <ubodt.bin> <trajectories.csv>\n";
        std::cout << "\nDemonstrates:\n"
                  << "  1. Query Cache (LRU)\n"
                  << "  2. Batch Processing\n"
                  << "  3. Incremental Loading\n"
                  << "  4. Combined Usage\n";
        return 1;
    }

    std::string network_file = argv[1];
    std::string ubodt_file = argv[2];
    std::string trajectory_file = argv[3];

    // Setup logging
    spdlog::set_level(spdlog::level::info);

    SPDLOG_INFO("=== Enhanced UBODT Examples ===");

    // Load network
    SPDLOG_INFO("Loading network from {}", network_file);
    CONFIG::NetworkConfig network_config(network_file, "id", "source", "target");
    Network network(network_config, true);
    SPDLOG_INFO("Network loaded: {} edges, {} vertices",
                network.get_edge_count(), network.get_node_count());

    // Load trajectories
    SPDLOG_INFO("Loading trajectories from {}", trajectory_file);
    std::vector<Trajectory> trajectories;
    TrajectoryReader reader(trajectory_file, "id", "geom");

    // Limit to 100 trajectories for demo
    int count = 0;
    const int MAX_TRAJECTORIES = 100;

    while (reader.has_next_trajectory() && count < MAX_TRAJECTORIES) {
        trajectories.push_back(reader.read_next_trajectory());
        ++count;
    }

    SPDLOG_INFO("Loaded {} trajectories", trajectories.size());

    // Load full UBODT for examples 1 and 2
    SPDLOG_INFO("Loading UBODT from {}", ubodt_file);
    auto ubodt = UBODT::read_ubodt_file(ubodt_file);
    if (!ubodt) {
        SPDLOG_CRITICAL("Failed to load UBODT");
        return 1;
    }
    SPDLOG_INFO("UBODT loaded: {} rows", ubodt->get_num_rows());

    // Run examples
    example_query_cache(network, ubodt);
    std::cout << "\n";

    example_batch_processing(network, ubodt_file, trajectories);
    std::cout << "\n";

    example_incremental_loading(network, ubodt_file, trajectories);
    std::cout << "\n";

    example_combined_usage(network, ubodt_file, trajectories);
    std::cout << "\n";

    SPDLOG_INFO("=== All Examples Completed ===");

    return 0;
}
