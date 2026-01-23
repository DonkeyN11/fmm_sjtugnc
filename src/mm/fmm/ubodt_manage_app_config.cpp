//
// UBODT Manager command line program configuration implementation
//

#include "mm/fmm/ubodt_manage_app_config.hpp"
#include "util/util.hpp"
#include "util/debug.hpp"
#include "cxxopts/cxxopts.hpp"
#include <iostream>

using namespace FMM;
using namespace FMM::MM;

UBODTManageAppConfig::UBODTManageAppConfig(int argc, char **argv) {
    std::string app_name = argv[0];
    cxxopts::Options options(app_name, "UBODT Manager - Load and manage UBODT in memory");

    // Positional argument for operation
    options.add_options()
        ("command", "Operation to perform: load, release, release_all, status",
         cxxopts::value<std::string>())
        ("file", "UBODT file path (for load/release operations)",
         cxxopts::value<std::string>())
        ("h,help", "Print help")
        ("v,verbose", "Verbose output");

    // Optional arguments
    options.add_options("Optional")
        ("m,multiplier", "Hash table multiplier (default: 1)",
         cxxopts::value<int>()->default_value("1"))
        ("n,network", "Network file path for validation",
         cxxopts::value<std::string>())
        ("log_level", "Log level (0=trace, 1=debug, 2=info, 3=warn, 4=err, 5=critical, 6=off)",
         cxxopts::value<int>()->default_value("2"));

    options.parse_positional({"command", "file"});
    auto result = options.parse(argc, argv);

    // Parse operation
    if (result.count("command") == 0) {
        // If no command specified, check if help is requested
        if (result.count("help") > 0) {
            help_specified = true;
            return;
        }
        // Otherwise default to showing status
        operation = Operation::STATUS;
    } else {
        std::string cmd_str = result["command"].as<std::string>();

        if (cmd_str == "load") {
            operation = Operation::LOAD;
        } else if (cmd_str == "release") {
            operation = Operation::RELEASE;
        } else if (cmd_str == "release_all") {
            operation = Operation::RELEASE_ALL;
        } else if (cmd_str == "status") {
            operation = Operation::STATUS;
        } else {
            std::cerr << "Unknown operation: " << cmd_str << "\n";
            help_specified = true;
            return;
        }
    }

    // Parse file path
    if (result.count("file") > 0) {
        ubodt_file = result["file"].as<std::string>();
    }

    // Parse options
    if (result.count("help") > 0) {
        help_specified = true;
    }

    if (result.count("verbose") > 0) {
        verbose = true;
        log_level = 1; // debug level
    }

    if (result.count("multiplier") > 0) {
        multiplier = result["multiplier"].as<int>();
    }

    if (result.count("network") > 0) {
        network_file = result["network"].as<std::string>();
    }

    if (result.count("log_level") > 0) {
        log_level = result["log_level"].as<int>();
    }

    // Configure logging
    spdlog::set_pattern("[%^%l%$][%t] %v");
    spdlog::set_level(static_cast<spdlog::level::level_enum>(log_level));
}

bool UBODTManageAppConfig::validate() const {
    if (help_specified) {
        return false;
    }

    // Validate operation and required arguments
    switch (operation) {
        case Operation::LOAD:
            if (ubodt_file.empty()) {
                std::cerr << "Error: UBODT file path required for 'load' operation\n";
                return false;
            }
            if (!UTIL::file_exists(ubodt_file)) {
                std::cerr << "Error: UBODT file not found: " << ubodt_file << "\n";
                return false;
            }
            break;

        case Operation::RELEASE:
            if (ubodt_file.empty()) {
                std::cerr << "Error: UBODT file path required for 'release' operation\n";
                return false;
            }
            break;

        case Operation::RELEASE_ALL:
        case Operation::STATUS:
            // No additional validation needed
            break;

        default:
            std::cerr << "Error: Invalid operation\n";
            return false;
    }

    return true;
}

void UBODTManageAppConfig::print() const {
    SPDLOG_INFO("---------- UBODT Manager Configuration ----------");
    SPDLOG_INFO("Operation: {}",
                operation == Operation::LOAD ? "load" :
                operation == Operation::RELEASE ? "release" :
                operation == Operation::RELEASE_ALL ? "release_all" :
                operation == Operation::STATUS ? "status" : "unknown");

    if (!ubodt_file.empty()) {
        SPDLOG_INFO("UBODT file: {}", ubodt_file);
    }
    if (!network_file.empty()) {
        SPDLOG_INFO("Network file: {}", network_file);
    }
    SPDLOG_INFO("Multiplier: {}", multiplier);
    SPDLOG_INFO("Log level: {}", log_level);
    SPDLOG_INFO("-----------------------------------------------");
}

void UBODTManageAppConfig::print_help() {
    std::cout << "UBODT Manager - Load and manage UBODT in memory\n\n";
    std::cout << "Usage:\n";
    std::cout << "  ubodt_manager load <ubodt_file> [options]\n";
    std::cout << "  ubodt_manager release <ubodt_file>\n";
    std::cout << "  ubodt_manager release_all\n";
    std::cout << "  ubodt_manager status\n\n";
    std::cout << "Commands:\n";
    std::cout << "  load         Load UBODT file into memory and keep it there\n";
    std::cout << "  release      Release a specific UBODT file from memory\n";
    std::cout << "  release_all  Release all UBODT files from memory\n";
    std::cout << "  status       Show status of loaded UBODT files\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  ubodt_file   Path to UBODT file (supports .csv, .txt, .bin formats)\n\n";
    std::cout << "Options:\n";
    std::cout << "  -m, --multiplier   Hash table multiplier (default: 1)\n";
    std::cout << "  -n, --network      Network file path for validation\n";
    std::cout << "  -v, --verbose      Verbose output (debug level logging)\n";
    std::cout << "  --log_level        Log level: 0=trace, 1=debug, 2=info, 3=warn, 4=err, 5=critical, 6=off\n";
    std::cout << "  -h, --help         Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  # Load UBODT into memory\n";
    std::cout << "  ubodt_manager load data/ubodt.bin\n\n";
    std::cout << "  # Load UBODT with larger hash table\n";
    std::cout << "  ubodt_manager load data/ubodt.bin --multiplier 2\n\n";
    std::cout << "  # Show status of loaded UBODTs\n";
    std::cout << "  ubodt_manager status\n\n";
    std::cout << "  # Release specific UBODT\n";
    std::cout << "  ubodt_manager release data/ubodt.bin\n\n";
    std::cout << "  # Release all UBODTs\n";
    std::cout << "  ubodt_manager release_all\n\n";
    std::cout << "Notes:\n";
    std::cout << "  - Once loaded, UBODT stays in memory until explicitly released\n";
    std::cout << "  - FMM and CMM applications will automatically use cached UBODT if available\n";
    std::cout << "  - If no cached UBODT is found, applications will load and release it automatically\n";
}
