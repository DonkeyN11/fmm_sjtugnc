//
// Covariance-based map matching application entry
//

#include "mm/cmm/cmm_app.hpp"
#include "mm/fmm/ubodt_manager.hpp"

#include "util/debug.hpp"

#include <utility>

using namespace FMM;
using namespace FMM::MM;
using namespace FMM::NETWORK;

CMMApp::CMMApp(const CMMAppConfig &config)
    : config_(config),
      network_(config_.network_config, false),  // Network stays in original CRS
      graph_(network_) {
    initialize_matcher();
}

CMMApp::CMMApp(const CMMAppConfig &config, std::shared_ptr<UBODT> preloaded_ubodt)
    : config_(config),
      network_(config_.network_config, false),  // Network stays in original CRS
      graph_(network_),
      ubodt_(std::move(preloaded_ubodt)) {
    initialize_matcher();
}

void CMMApp::initialize_matcher() {
    if (!ubodt_) {
        // Check if UBODT is already loaded in memory
        auto &manager = UBODTManager::getInstance();

        if (config_.use_memory_cache && manager.is_loaded(config_.ubodt_file)) {
            SPDLOG_INFO("Using cached UBODT from memory");
            ubodt_ = manager.get_ubodt(config_.ubodt_file);
            // Enable auto-release so UBODT is released after use
            manager.set_auto_release(true);
        } else {
            if (UBODTManager::check_daemon_loaded(config_.ubodt_file)) {
                SPDLOG_INFO("UBODT is preloaded by ubodt_daemon. Using fast loading from OS cache.");
            } else {
                SPDLOG_INFO("UBODT not found in daemon. Loading from disk.");
            }
            ubodt_ = UBODT::read_ubodt_file(config_.ubodt_file);
        }
    } else {
        SPDLOG_INFO("Using provided pre-loaded UBODT");
    }
    cmm_ = std::make_unique<CovarianceMapMatch>(network_, graph_, ubodt_);
}

void CMMApp::run() {
    SPDLOG_INFO("CMM application started");
    if (!cmm_) {
        initialize_matcher();
    }
    const std::string status = cmm_->match_gps_file(
        config_.gps_config,
        config_.result_config,
        config_.cmm_config,
        config_.input_epsg,  // Use explicit input_epsg instead of convert_to_projected
        config_.use_omp);
    if (!status.empty()) {
        SPDLOG_INFO("{}", status);
    }
    SPDLOG_INFO("CMM application finished");
}
