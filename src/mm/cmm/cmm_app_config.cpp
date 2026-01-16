//
// Created for CMM implementation
// CMM application configuration management
//

#include "mm/cmm/cmm_app_config.hpp"
#include "util/debug.hpp"
#include "util/util.hpp"
#include "io/gps_reader.hpp"

using namespace FMM;
using namespace FMM::CONFIG;
using namespace FMM::MM;
using namespace FMM::IO;

CMMAppConfig::CMMAppConfig()
    : ubodt_file(),
      use_omp(true),
      log_level(2),
      step(100),
      convert_to_projected(false),
      help_specified(false),
      network_bbox_from_gps(false),
      network_bbox_padding(0.0) {}

CMMAppConfig CMMAppConfig::load_from_xml(const std::string &xml_file) {
    boost::property_tree::ptree tree;
    boost::property_tree::read_xml(xml_file, tree);

    CMMAppConfig config;
    config.network_config = NetworkConfig::load_from_xml(tree);
    config.gps_config = GPSConfig::load_from_xml(tree);
    config.result_config = ResultConfig::load_from_xml(tree);
    config.cmm_config = CovarianceMapMatchConfig::load_from_xml(tree);
    config.ubodt_file = tree.get<std::string>("config.input.ubodt.file", "");
    config.use_omp = tree.get("config.other.use_omp", true);
    config.log_level = tree.get("config.other.log_level", 2);
    config.step = tree.get("config.other.step", 100);
    config.convert_to_projected = tree.get("config.other.convert_to_projected", false);
    config.network_bbox_from_gps = tree.get("config.other.network_bbox_from_gps", false);
    config.network_bbox_padding = tree.get("config.other.network_bbox_padding", 0.0);
    if (config.network_bbox_from_gps && !config.network_config.has_bbox) {
        auto bounds = IO::compute_gps_bounds_in_network_crs(
            config.gps_config, config.network_config.file, config.convert_to_projected);
        if (bounds.valid) {
            config.network_config.has_bbox = true;
            config.network_config.bbox_minx = bounds.minx - config.network_bbox_padding;
            config.network_config.bbox_miny = bounds.miny - config.network_bbox_padding;
            config.network_config.bbox_maxx = bounds.maxx + config.network_bbox_padding;
            config.network_config.bbox_maxy = bounds.maxy + config.network_bbox_padding;
            config.network_config.bbox_valid = true;
            SPDLOG_INFO("Network bbox set from GPS with padding {}", config.network_bbox_padding);
        } else {
            SPDLOG_WARN("Failed to derive GPS bbox; network bbox disabled");
        }
    }

    return config;
}

CMMAppConfig CMMAppConfig::load_from_arg(const cxxopts::ParseResult &arg_data) {
    CMMAppConfig config;
    config.network_config = NetworkConfig::load_from_arg(arg_data);
    config.gps_config = GPSConfig::load_from_arg(arg_data);
    config.result_config = ResultConfig::load_from_arg(arg_data);
    config.cmm_config = CovarianceMapMatchConfig::load_from_arg(arg_data);
    config.ubodt_file = arg_data["ubodt"].as<std::string>();
    config.use_omp = arg_data["use_omp"].as<bool>();
    config.log_level = arg_data["log_level"].as<int>();
    config.step = arg_data["step"].as<int>();
    config.convert_to_projected = arg_data["convert_to_projected"].as<bool>();
    config.network_bbox_from_gps = arg_data["network_bbox_from_gps"].as<bool>();
    config.network_bbox_padding = arg_data["network_bbox_padding"].as<double>();
    config.help_specified = arg_data.count("help") > 0;
    if (config.network_bbox_from_gps && !config.network_config.has_bbox) {
        auto bounds = IO::compute_gps_bounds_in_network_crs(
            config.gps_config, config.network_config.file, config.convert_to_projected);
        if (bounds.valid) {
            config.network_config.has_bbox = true;
            config.network_config.bbox_minx = bounds.minx - config.network_bbox_padding;
            config.network_config.bbox_miny = bounds.miny - config.network_bbox_padding;
            config.network_config.bbox_maxx = bounds.maxx + config.network_bbox_padding;
            config.network_config.bbox_maxy = bounds.maxy + config.network_bbox_padding;
            config.network_config.bbox_valid = true;
            SPDLOG_INFO("Network bbox set from GPS with padding {}", config.network_bbox_padding);
        } else {
            SPDLOG_WARN("Failed to derive GPS bbox; network bbox disabled");
        }
    }

    return config;
}

void CMMAppConfig::register_arg(cxxopts::Options &options) {
    NetworkConfig::register_arg(options);
    GPSConfig::register_arg(options);
    ResultConfig::register_arg(options);
    CovarianceMapMatchConfig::register_arg(options);

    options.add_options()
        ("ubodt", "UBODT file name",
         cxxopts::value<std::string>()->default_value(""))
        ("use_omp", "Use OpenMP for parallel processing",
         cxxopts::value<bool>()->default_value("false"))
        ("log_level", "Log level (0=trace, 1=debug, 2=info, 3=warn, 4=error, 5=critical)",
         cxxopts::value<int>()->default_value("2"))
        ("step", "Progress report step",
         cxxopts::value<int>()->default_value("100"))
        ("convert_to_projected", "Convert inputs to a projected CRS when necessary",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("network_bbox_from_gps", "Auto-crop network by GPS bounds",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("network_bbox_padding", "Padding for GPS-derived bbox",
         cxxopts::value<double>()->default_value("0.0"))
        ("h,help", "Print help information");
}

void CMMAppConfig::register_help(std::ostringstream &oss) {
    NetworkConfig::register_help(oss);
    GPSConfig::register_help(oss);
    ResultConfig::register_help(oss);
    CovarianceMapMatchConfig::register_help(oss);
    oss << "--ubodt (required) <string>: UBODT file name\n";
    oss << "--use_omp (optional): Use OpenMP for parallel processing (false)\n";
    oss << "--log_level (optional): Log level (0=trace, 1=debug, 2=info, 3=warn, 4=error, 5=critical) (2)\n";
    oss << "--step (optional): Progress report step (100)\n";
    oss << "--convert_to_projected (optional): Convert inputs to a projected CRS when necessary (false)\n";
    oss << "--network_bbox_from_gps (optional): Auto-crop network by GPS bounds (false)\n";
    oss << "--network_bbox_padding (optional): Padding for GPS-derived bbox (0.0)\n";
    oss << "-h/--help: Print this help information\n";
}

void CMMAppConfig::print() const {
    SPDLOG_INFO("---- CMM Application Configuration ----");
    network_config.print();
    gps_config.print();
    result_config.print();
    cmm_config.print();
    SPDLOG_INFO("UBODT file {}", ubodt_file);
    SPDLOG_INFO("Log level {}", log_level);
    SPDLOG_INFO("Use omp {}", use_omp);
    SPDLOG_INFO("Step {}", step);
    SPDLOG_INFO("Convert to projected {}", convert_to_projected);
    SPDLOG_INFO("Network bbox from GPS {}", (network_bbox_from_gps ? "true" : "false"));
    SPDLOG_INFO("Network bbox padding {}", network_bbox_padding);
    SPDLOG_INFO("---- Configuration done ----");
}

bool CMMAppConfig::validate() const {
    if (!network_config.validate()) {
        SPDLOG_CRITICAL("Network configuration invalid");
        return false;
    }
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
    if (ubodt_file.empty()) {
        SPDLOG_CRITICAL("UBODT file path not provided");
        return false;
    }
    if (!UTIL::file_exists(ubodt_file)) {
        SPDLOG_CRITICAL("UBODT file not exists {}", ubodt_file);
        return false;
    }
    return true;
}
