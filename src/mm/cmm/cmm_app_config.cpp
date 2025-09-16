//
// Created for CMM implementation
// CMM application configuration management
//

#include "mm/cmm/cmm_app_config.hpp"
#include "util/debug.hpp"

using namespace FMM;
using namespace FMM::CONFIG;
using namespace FMM::MM;

CMMAppConfig CMMAppConfig::load_from_xml(const std::string &xml_file) {
    boost::property_tree::ptree tree;
    boost::property_tree::read_xml(xml_file, tree);

    CMMAppConfig config;
    config.gps_config = GPSConfig::load_from_xml(tree);
    config.result_config = ResultConfig::load_from_xml(tree);
    config.cmm_config = CovarianceMapMatchConfig::load_from_xml(tree);
    config.use_omp = tree.get("config.other.use_omp", false);
    config.log_level = tree.get("config.other.log_level", 2);

    return config;
}

CMMAppConfig CMMAppConfig::load_from_arg(const cxxopts::ParseResult &arg_data) {
    CMMAppConfig config;
    config.gps_config = GPSConfig::load_from_arg(arg_data);
    config.result_config = ResultConfig::load_from_arg(arg_data);
    config.cmm_config = CovarianceMapMatchConfig::load_from_arg(arg_data);
    config.use_omp = arg_data["use_omp"].as<bool>();
    config.log_level = arg_data["log_level"].as<int>();

    return config;
}

void CMMAppConfig::register_arg(cxxopts::Options &options) {
    GPSConfig::register_arg(options);
    ResultConfig::register_arg(options);
    CovarianceMapMatchConfig::register_arg(options);

    options.add_options()
        ("use_omp", "Use OpenMP for parallel processing",
         cxxopts::value<bool>()->default_value("false"))
        ("log_level", "Log level (0=trace, 1=debug, 2=info, 3=warn, 4=error, 5=critical)",
         cxxopts::value<int>()->default_value("2"));
}

void CMMAppConfig::register_help(std::ostringstream &oss) {
    GPSConfig::register_help(oss);
    ResultConfig::register_help(oss);
    CovarianceMapMatchConfig::register_help(oss);
    oss << "--use_omp (optional): Use OpenMP for parallel processing (false)\n";
    oss << "--log_level (optional): Log level (0=trace, 1=debug, 2=info, 3=warn, 4=error, 5=critical) (2)\n";
}

void CMMAppConfig::print() const {
    SPDLOG_INFO("---- CMM Application Configuration ----");
    gps_config.print();
    result_config.print();
    cmm_config.print();
    SPDLOG_INFO("Log level {}", log_level);
    SPDLOG_INFO("Use omp {}", use_omp);
    SPDLOG_INFO("---- Configuration done ----");
}

bool CMMAppConfig::validate() const {
    if (!gps_config.validate()) {
        SPDLOG_CRITICAL("GPS configuration invalid");
        return false;
    }
    if (!result_config.validate()) {
        SPDLOG_CRITICAL("Result configuration invalid");
        return false;
    }
    if (!cmm_config.validate()) {
        SPDLOG_CRITICAL("CMM configuration invalid");
        return false;
    }
    if (log_level < 0 || log_level > 5) {
        SPDLOG_CRITICAL("Log level {} invalid", log_level);
        return false;
    }
    return true;
}

CMMApp::CMMApp(const CMMAppConfig &config,
               const NETWORK::Network &network,
               const NETWORK::NetworkGraph &graph,
               std::shared_ptr<UBODT> ubodt)
    : config_(config), network_(network), graph_(graph), ubodt_(ubodt) {
    cmm_ = std::make_unique<CovarianceMapMatch>(network, graph, ubodt);
}

void CMMApp::run() {
    // For now, just print information about CMM requirements
    SPDLOG_INFO("CMM application started");
    SPDLOG_INFO("CMM requires trajectory data with covariance matrices and protection levels");
    SPDLOG_INFO("This implementation provides the core CMM algorithm with covariance-based emission probability");
    SPDLOG_INFO("To use CMM, prepare GPS data files that include:");
    SPDLOG_INFO("1. Trajectory coordinates (as in standard FMM)");
    SPDLOG_INFO("2. Covariance matrices for each GPS point (sdn, sde, sdu, sdne, sdeu, sdun)");
    SPDLOG_INFO("3. Protection levels for each GPS point");

    // The actual GPS file processing would require enhanced GPS readers
    // that can parse covariance and protection level data
}