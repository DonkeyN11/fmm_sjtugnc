/**
 * Fast map matching.
 *
 * UBODT pre-loading command line program
 *
 * @author: Generated for FMM optimization
 * @version: 2025.01.01
 */

#include "mm/fmm/ubodt_memory_manager.hpp"
#include <iostream>
#include <string>

using namespace FMM;
using namespace FMM::MM;

void print_help() {
    std::cout << "UBODT Read - Pre-load UBODT files into memory cache\n\n";
    std::cout << "Usage: ubodt_read [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --ubodt FILE       UBODT file to load (required)\n";
    std::cout << "  --max_memory MB    Maximum memory usage in MB (default: unlimited)\n";
    std::cout << "  --multiplier N     UBODT multiplier (default: 50000)\n";
    std::cout << "  --status           Show cache status\n";
    std::cout << "  --clear            Clear all cached UBODT files\n";
    std::cout << "  --unload FILE      Unload specific UBODT file\n";
    std::cout << "  --help             Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  ubodt_read --ubodt data/ubodt.bin\n";
    std::cout << "  ubodt_read --ubodt data/ubodt.bin --max_memory 2048\n";
    std::cout << "  ubodt_read --status\n";
    std::cout << "  ubodt_read --clear\n";
}

int main(int argc, char **argv) {
    std::string ubodt_file;
    size_t max_memory_mb = 0;
    int multiplier = 50000;
    bool show_status = false;
    bool clear_cache = false;
    std::string unload_file;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--ubodt" && i + 1 < argc) {
            ubodt_file = argv[++i];
        } else if (arg == "--max_memory" && i + 1 < argc) {
            max_memory_mb = std::stoul(argv[++i]);
        } else if (arg == "--multiplier" && i + 1 < argc) {
            multiplier = std::stoi(argv[++i]);
        } else if (arg == "--status") {
            show_status = true;
        } else if (arg == "--clear") {
            clear_cache = true;
        } else if (arg == "--unload" && i + 1 < argc) {
            unload_file = argv[++i];
        } else if (arg == "--help") {
            print_help();
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_help();
            return 1;
        }
    }

    auto& manager = UBODTMemoryManager::get_instance();

    // Set memory limit first
    if (max_memory_mb > 0) {
        manager.set_max_memory(max_memory_mb);
        std::cout << "Memory limit set to " << max_memory_mb << " MB\n";
    }

    // Handle clear cache
    if (clear_cache) {
        std::cout << "Clearing UBODT cache...\n";
        manager.clear_cache();
        std::cout << "Cache cleared successfully.\n";
        return 0;
    }

    // Handle unload file
    if (!unload_file.empty()) {
        std::cout << "Unloading UBODT file: " << unload_file << "\n";
        if (manager.unload_ubodt(unload_file)) {
            std::cout << "UBODT unloaded successfully.\n";
        } else {
            std::cerr << "Failed to unload UBODT file.\n";
            return 1;
        }
    }

    // Handle show status
    if (show_status) {
        std::cout << "UBODT Cache Status:\n";
        manager.print_status();
        return 0;
    }

    // Handle load UBODT
    if (!ubodt_file.empty()) {
        std::cout << "Loading UBODT file: " << ubodt_file << "\n";
        if (manager.load_ubodt(ubodt_file, multiplier, max_memory_mb)) {
            std::cout << "UBODT loaded successfully into memory cache.\n";
            std::cout << "\nCache Status:\n";
            manager.print_status();
        } else {
            std::cerr << "Failed to load UBODT file.\n";
            return 1;
        }
    } else if (unload_file.empty() && !show_status && !clear_cache) {
        std::cerr << "No action specified. Use --help for usage information.\n";
        return 1;
    }

    return 0;
}