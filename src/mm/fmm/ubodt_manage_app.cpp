//
// UBODT Manager application implementation
//

#include "mm/fmm/ubodt_manage_app.hpp"
#include "util/util.hpp"
#include <iomanip>

using namespace FMM;
using namespace FMM::MM;

UBODTManageApp::UBODTManageApp(const UBODTManageAppConfig &config)
    : config_(config) {
    // Disable auto-release so UBODT stays in memory
    UBODTManager::getInstance().set_auto_release(false);
}

int UBODTManageApp::run() {
    switch (config_.operation) {
        case UBODTManageAppConfig::Operation::LOAD:
            return load_ubodt() ? 0 : 1;

        case UBODTManageAppConfig::Operation::RELEASE:
            return release_ubodt() ? 0 : 1;

        case UBODTManageAppConfig::Operation::RELEASE_ALL:
            return release_all() ? 0 : 1;

        case UBODTManageAppConfig::Operation::STATUS:
            return show_status() ? 0 : 1;

        default:
            SPDLOG_ERROR("Unknown operation");
            return 1;
    }
}

bool UBODTManageApp::load_ubodt() {
    SPDLOG_INFO("Loading UBODT from: {}", config_.ubodt_file);

    auto start_time = UTIL::get_current_time();

    // Check if already loaded
    auto &manager = UBODTManager::getInstance();
    if (manager.is_loaded(config_.ubodt_file)) {
        SPDLOG_INFO("UBODT already loaded in memory");
        std::cout << "\nUBODT is already loaded in memory.\n";
        std::cout << "Use 'ubodt_manager status' to see details.\n\n";
        return true;
    }

    // Load UBODT using manager
    auto ubodt = manager.get_ubodt(config_.ubodt_file, config_.multiplier, false);

    auto end_time = UTIL::get_current_time();
    double duration = UTIL::get_duration(start_time, end_time);

    if (!ubodt) {
        SPDLOG_ERROR("Failed to load UBODT from {}", config_.ubodt_file);
        std::cout << "\nError: Failed to load UBODT file.\n";
        std::cout << "Please check the file path and format.\n\n";
        return false;
    }

    // Success
    SPDLOG_INFO("UBODT loaded successfully in {:.2f}s", duration);
    std::cout << "\n✓ UBODT loaded successfully!\n";
    std::cout << "  File: " << config_.ubodt_file << "\n";
    std::cout << "  Rows: " << ubodt->get_num_rows() << "\n";
    std::cout << "  Load time: " << std::fixed << std::setprecision(2)
              << duration << "s\n";
    std::cout << "  Status: Kept in memory for reuse\n\n";
    std::cout << "The UBODT will now be automatically used by FMM and CMM applications.\n";
    std::cout << "Use 'ubodt_manager status' to view loaded UBODTs.\n";
    std::cout << "Use 'ubodt_manager release " << config_.ubodt_file
              << "' to release it from memory.\n\n";

    return true;
}

bool UBODTManageApp::release_ubodt() {
    SPDLOG_INFO("Releasing UBODT: {}", config_.ubodt_file);

    auto &manager = UBODTManager::getInstance();

    // Check if loaded
    if (!manager.is_loaded(config_.ubodt_file)) {
        std::cout << "\nUBODT is not currently loaded in memory.\n\n";
        return false;
    }

    // Release UBODT
    size_t released = manager.release_ubodt(config_.ubodt_file);

    if (released > 0) {
        std::cout << "\n✓ Released " << released << " UBODT instance(s)\n";
        std::cout << "  File: " << config_.ubodt_file << "\n\n";
        return true;
    } else {
        std::cout << "\nNo UBODT found matching: " << config_.ubodt_file << "\n\n";
        return false;
    }
}

bool UBODTManageApp::release_all() {
    SPDLOG_INFO("Releasing all UBODTs");

    auto &manager = UBODTManager::getInstance();
    size_t released = manager.release_all();

    std::cout << "\n✓ Released " << released << " UBODT instance(s) from memory\n\n";

    return true;
}

bool UBODTManageApp::show_status() {
    auto &manager = UBODTManager::getInstance();
    auto stats = manager.get_stats();

    std::cout << "\n========== UBODT Manager Status ==========\n";
    std::cout << "Total UBODTs loaded: " << stats.total_ubodts << "\n";
    std::cout << "Total references: " << stats.total_references << "\n";

    // Format memory size
    double memory_mb = stats.memory_estimated / (1024.0 * 1024.0);
    std::cout << "Estimated memory: ";
    if (memory_mb < 1024.0) {
        std::cout << std::fixed << std::setprecision(2) << memory_mb << " MB\n";
    } else {
        std::cout << std::fixed << std::setprecision(2) << (memory_mb / 1024.0) << " GB\n";
    }

    std::cout << "Auto-release: " << (manager.get_auto_release() ? "enabled" : "disabled") << "\n";

    // Show detailed status via logging
    manager.print_status();

    std::cout << "==========================================\n\n";

    // Show usage hints
    if (stats.total_ubodts > 0) {
        std::cout << "Loaded UBODTs will be automatically used by FMM and CMM.\n";
        std::cout << "Use 'ubodt_manager release <file>' or 'release_all' to free memory.\n\n";
    } else {
        std::cout << "No UBODTs currently loaded.\n";
        std::cout << "Use 'ubodt_manager load <file>' to load a UBODT into memory.\n\n";
    }

    return true;
}
