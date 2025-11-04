//
// Covariance-based map matching application entry
//

#include "mm/cmm/cmm_app.hpp"

#include "util/debug.hpp"

#include <utility>

using namespace FMM;
using namespace FMM::MM;
using namespace FMM::NETWORK;

CMMApp::CMMApp(const CMMAppConfig &config)
    : config_(config),
      network_(config_.network_config, config_.convert_to_projected),
      graph_(network_) {
    initialize_matcher();
}

CMMApp::CMMApp(const CMMAppConfig &config, std::shared_ptr<UBODT> preloaded_ubodt)
    : config_(config),
      network_(config_.network_config, config_.convert_to_projected),
      graph_(network_),
      ubodt_(std::move(preloaded_ubodt)) {
    initialize_matcher();
}

void CMMApp::initialize_matcher() {
    if (!ubodt_) {
        SPDLOG_INFO("Loading UBODT from {}", config_.ubodt_file);
        ubodt_ = UBODT::read_ubodt_file(config_.ubodt_file);
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
        config_.convert_to_projected,
        config_.use_omp);
    if (!status.empty()) {
        SPDLOG_INFO("{}", status);
    }
    SPDLOG_INFO("CMM application finished");
}
