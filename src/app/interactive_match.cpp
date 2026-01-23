/**
 * Interactive Map Matching Tool
 *
 * Allows loading UBODT once and performing multiple matching operations
 * without reloading the UBODT each time.
 *
 * Usage:
 *   ./interactive_match --network network.shp --ubodt ubodt.bin --mode fmm
 *
 * Commands in interactive mode:
 *   match <traj_file>     - Match trajectories from file
 *   status                - Show UBODT status
 *   release               - Release loaded UBODT
 *   reload                - Reload UBODT
 *   exit                  - Exit program
 */

#include "mm/fmm/ubodt_manager.hpp"
#include "mm/fmm/fmm_algorithm.hpp"
#include "mm/cmm/cmm_algorithm.hpp"
#include "network/network.hpp"
#include "io/gps_reader.hpp"
#include "util/debug.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>

using namespace FMM;
using namespace FMM::CORE;
using namespace FMM::NETWORK;
using namespace FMM::MM;
using namespace FMM::IO;
using namespace FMM::CONFIG;

// Interactive matching session
class InteractiveMatcher {
public:
    InteractiveMatcher(const std::string &network_file,
                      const std::string &ubodt_file,
                      const std::string &mode)
        : network_file_(network_file)
        , ubodt_file_(ubodt_file)
        , mode_(mode)
        , ubodt_loaded_(false) {

        // Load network
        SPDLOG_INFO("Loading network from {}", network_file);
        NetworkConfig network_config(network_file, "id", "source", "target");
        network_ = std::make_unique<Network>(network_config, true);
        SPDLOG_INFO("Network loaded: {} edges, {} vertices",
                   network_->get_edge_count(), network_->get_node_count());
    }

    ~InteractiveMatcher() {
        // Release UBODT on exit
        if (ubodt_loaded_) {
            SPDLOG_INFO("Releasing UBODT on exit");
            UBODTHelper::release_all_ubodts();
        }
    }

    /**
     * Run interactive session
     */
    void run() {
        std::cout << "\n========================================\n";
        std::cout << "  Interactive Map Matching Tool\n";
        std::cout << "  Mode: " << mode_ << "\n";
        std::cout << "  Network: " << network_file_ << "\n";
        std::cout << "  UBODT: " << ubodt_file_ << "\n";
        std::cout << "========================================\n\n";

        std::cout << "Commands:\n";
        std::cout << "  load                      - Load UBODT into memory\n";
        std::cout << "  match <traj_file>         - Match trajectories\n";
        std::cout << "  batch <file_list.txt>     - Batch match multiple files\n";
        std::cout << "  status                    - Show UBODT status\n";
        std::cout << "  release                   - Release UBODT\n";
        std::cout << "  cache <on|off>            - Enable/disable query cache\n";
        std::cout << "  help                      - Show this help\n";
        std::cout << "  exit                      - Exit program\n\n";

        std::string line;
        while (true) {
            std::cout << (mode_ == "fmm" ? "fmm" : "cmm") << "> ";
            std::getline(std::cin, line);

            if (line.empty()) continue;

            std::istringstream iss(line);
            std::string cmd;
            iss >> cmd;

            if (cmd == "exit" || cmd == "quit") {
                break;
            } else if (cmd == "help") {
                show_help();
            } else if (cmd == "load") {
                load_ubodt();
            } else if (cmd == "match") {
                std::string traj_file;
                iss >> traj_file;
                if (traj_file.empty()) {
                    std::cout << "Error: Please specify trajectory file\n";
                } else {
                    match_trajectories(traj_file);
                }
            } else if (cmd == "batch") {
                std::string list_file;
                iss >> list_file;
                if (list_file.empty()) {
                    std::cout << "Error: Please specify file list\n";
                } else {
                    batch_match(list_file);
                }
            } else if (cmd == "status") {
                show_status();
            } else if (cmd == "release") {
                release_ubodt();
            } else if (cmd == "cache") {
                std::string state;
                iss >> state;
                if (state == "on") {
                    use_cache_ = true;
                    std::cout << "Query cache enabled\n";
                } else if (state == "off") {
                    use_cache_ = false;
                    std::cout << "Query cache disabled\n";
                } else {
                    std::cout << "Usage: cache <on|off>\n";
                }
            } else {
                std::cout << "Unknown command: " << cmd << "\n";
                std::cout << "Type 'help' for available commands\n";
            }
        }

        std::cout << "\nThank you for using Interactive Map Matching Tool!\n";
    }

private:
    std::string network_file_;
    std::string ubodt_file_;
    std::string mode_;
    std::unique_ptr<Network> network_;
    bool ubodt_loaded_;
    bool use_cache_ = false;

    void show_help() {
        std::cout << "\nAvailable Commands:\n\n";
        std::cout << "  load\n";
        std::cout << "      Load UBODT into memory (required before first match)\n\n";
        std::cout << "  match <traj_file>\n";
        std::cout << "      Match trajectories from file\n";
        std::cout << "      Example: match data/trajectories.csv\n\n";
        std::cout << "  batch <file_list.txt>\n";
        std::cout << "      Batch match multiple trajectory files\n";
        std::cout << "      Format: one trajectory file path per line\n\n";
        std::cout << "  status\n";
        std::cout << "      Show UBODT loading status and statistics\n\n";
        std::cout << "  release\n";
        std::cout << "      Release UBODT from memory\n\n";
        std::cout << "  cache <on|off>\n";
        std::cout << "      Enable or disable query cache (default: off)\n\n";
        std::cout << "  help\n";
        std::cout << "      Show this help message\n\n";
        std::cout << "  exit\n";
        std::cout << "      Exit the program\n\n";
    }

    void load_ubodt() {
        if (ubodt_loaded_) {
            std::cout << "UBODT already loaded. Use 'release' first to reload.\n";
            return;
        }

        std::cout << "Loading UBODT from " << ubodt_file_ << "...\n";
        auto start = std::chrono::high_resolution_clock::now();

        if (use_cache_) {
            ubodt_ = UBODTHelper::load_cached_ubodt(ubodt_file_, 10000, 1, true);
        } else {
            ubodt_ = UBODTHelper::load_ubodt(ubodt_file_, 1, true);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end - start).count();

        if (ubodt_) {
            ubodt_loaded_ = true;
            std::cout << "UBODT loaded successfully in " << duration << " seconds\n";
            std::cout << "Records: " << ubodt_->get_num_rows() << "\n";
        } else {
            std::cout << "Failed to load UBODT\n";
        }
    }

    void match_trajectories(const std::string &traj_file) {
        if (!ubodt_loaded_) {
            std::cout << "Error: UBODT not loaded. Use 'load' command first.\n";
            return;
        }

        std::cout << "Matching trajectories from " << traj_file << "...\n";

        try {
            // Read trajectories
            TrajectoryReader reader(traj_file, "id", "geom");
            std::vector<Trajectory> trajectories;

            int count = 0;
            while (reader.has_next_trajectory() && count < 10000) {
                trajectories.push_back(reader.read_next_trajectory());
                ++count;
            }

            std::cout << "Read " << trajectories.size() << " trajectories\n";

            auto start = std::chrono::high_resolution_clock::now();

            // Match
            std::vector<std::string> result;
            if (mode_ == "fmm") {
                result = match_fmm(trajectories);
            } else {
                result = match_cmm(trajectories);
            }

            auto end = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double>(end - start).count();

            std::cout << "Matching completed in " << duration << " seconds\n";
            std::cout << "Processed " << result.size() << " trajectories\n";

            // Save results
            std::string output_file = traj_file + ".matched.csv";
            std::ofstream out(output_file);
            out << "traj_id\n";
            for (const auto &id : result) {
                out << id << "\n";
            }
            out.close();
            std::cout << "Results saved to " << output_file << "\n";

        } catch (const std::exception &e) {
            std::cout << "Error: " << e.what() << "\n";
        }
    }

    void batch_match(const std::string &list_file) {
        if (!ubodt_loaded_) {
            std::cout << "Error: UBODT not loaded. Use 'load' command first.\n";
            return;
        }

        std::ifstream infile(list_file);
        if (!infile.is_open()) {
            std::cout << "Error: Cannot open file list: " << list_file << "\n";
            return;
        }

        std::string line;
        int file_count = 0;
        int total_traj_count = 0;

        std::cout << "\n=== Batch Matching ===\n";

        while (std::getline(infile, line)) {
            if (line.empty() || line[0] == '#') continue;

            std::cout << "\n[" << (file_count + 1) << "] Processing: " << line << "\n";
            match_trajectories(line);
            ++file_count;
        }

        infile.close();
        std::cout << "\n=== Batch Matching Completed ===\n";
        std::cout << "Total files processed: " << file_count << "\n\n";
    }

    std::vector<std::string> match_fmm(const std::vector<Trajectory> &trajectories) {
        FMMConfig config;
        config.radius = 300;  // 300m search radius

        FMMAlgorithm fmm_algo(*network_, ubodt_);

        std::vector<std::string> matched_ids;
        for (const auto &traj : trajectories) {
            auto result = fmm_algo.match_traj(traj, config);
            if (!result.cpath.empty()) {
                matched_ids.push_back(std::to_string(traj.id));
            }
        }

        return matched_ids;
    }

    std::vector<std::string> match_cmm(const std::vector<Trajectory> &trajectories) {
        CMMConfig config;
        // Configure CMM parameters...

        CMMAlgorithm cmm_algo(*network_, ubodt_);

        std::vector<std::string> matched_ids;
        for (const auto &traj : trajectories) {
            auto result = cmm_algo.match_traj(traj, config);
            if (!result.cpath.empty()) {
                matched_ids.push_back(std::to_string(traj.id));
            }
        }

        return matched_ids;
    }

    void show_status() {
        std::cout << "\n=== UBODT Status ===\n";
        std::cout << "Loaded: " << (ubodt_loaded_ ? "Yes" : "No") << "\n";
        std::cout << "File: " << ubodt_file_ << "\n";
        std::cout << "Cache: " << (use_cache_ ? "Enabled" : "Disabled") << "\n";

        if (ubodt_loaded_) {
            UBODTHelper::print_ubodt_status();
        }

        std::cout << "\n";
    }

    void release_ubodt() {
        if (!ubodt_loaded_) {
            std::cout << "UBODT not loaded.\n";
            return;
        }

        std::cout << "Releasing UBODT...\n";
        size_t released = UBODTHelper::release_all_ubodts();
        std::cout << "Released " << released << " UBODT(s)\n";

        ubodt_loaded_ = false;
        ubodt_.reset();
    }

    std::shared_ptr<UBODT> ubodt_;
};

// ============================================================================
// Main
// ============================================================================

void print_usage(const char *program_name) {
    std::cout << "Usage: " << program_name
              << " --network <network.shp> --ubodt <ubodt.bin> --mode <fmm|cmm>\n\n";
    std::cout << "Options:\n";
    std::cout << "  --network <file>    Road network shapefile\n";
    std::cout << "  --ubodt <file>      UBODT binary file\n";
    std::cout << "  --mode <fmm|cmm>    Matching algorithm (default: fmm)\n";
    std::cout << "  --help              Show this help message\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << program_name << " --network data/road.shp \\\n";
    std::cout << "                       --ubodt data/ubodt.bin \\\n";
    std::cout << "                       --mode fmm\n\n";
}

int main(int argc, char **argv) {
    std::string network_file;
    std::string ubodt_file;
    std::string mode = "fmm";

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--network" && i + 1 < argc) {
            network_file = argv[++i];
        } else if (arg == "--ubodt" && i + 1 < argc) {
            ubodt_file = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
            if (mode != "fmm" && mode != "cmm") {
                std::cerr << "Error: Invalid mode. Use 'fmm' or 'cmm'.\n";
                return 1;
            }
        } else {
            std::cerr << "Error: Unknown or incomplete argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate arguments
    if (network_file.empty() || ubodt_file.empty()) {
        std::cerr << "Error: Missing required arguments.\n\n";
        print_usage(argv[0]);
        return 1;
    }

    // Initialize logging
    spdlog::set_level(spdlog::level::warn);  // Reduce log noise in interactive mode

    // Create and run interactive matcher
    try {
        InteractiveMatcher matcher(network_file, ubodt_file, mode);
        matcher.run();
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
