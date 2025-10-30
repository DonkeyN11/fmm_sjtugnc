/**
 * Covariance based map matching.
 *
 * cmm command line program main function
 *
 * @author: Chenzhang Ning
 * @version: 2025.09.29
 */

#include "mm/cmm/cmm_app.hpp"
#include "util/debug.hpp"
#include "util/util.hpp"

#include "cxxopts/cxxopts.hpp"

#include <iostream>
#include <sstream>

using namespace FMM;
using namespace FMM::MM;

namespace {

void print_help_and_exit() {
  std::ostringstream oss;
  oss << "cmm argument lists:\n";
  CMMAppConfig::register_help(oss);
  std::cout << oss.str();
}

} // namespace

int main(int argc, char **argv) {
  spdlog::set_pattern("[%^%l%$][%s:%-3#] %v");

  CMMAppConfig config;
  bool config_loaded = false;

  if (argc == 2 && UTIL::check_file_extension(argv[1], "xml,XML")) {
    config = CMMAppConfig::load_from_xml(argv[1]);
    config_loaded = true;
  } else {
    cxxopts::Options options("cmm", "Covariance-based map matching");
    CMMAppConfig::register_arg(options);

    try {
      auto result = options.parse(argc, argv);
      if (result.count("help") > 0 || argc == 1) {
        print_help_and_exit();
        return 0;
      }
      config = CMMAppConfig::load_from_arg(result);
      config_loaded = true;
    } catch (const std::exception &ex) {
      std::cerr << "Failed to parse arguments: " << ex.what() << "\n";
      print_help_and_exit();
      return 1;
    }
  }

  if (!config_loaded) {
    print_help_and_exit();
    return 0;
  }

  spdlog::set_level(static_cast<spdlog::level::level_enum>(config.log_level));

  if (!config.validate()) {
    return 1;
  }

  if (!config.help_specified) {
    config.print();
  }

  try {
    CMMApp app(config);
    app.run();
  } catch (const std::exception &ex) {
    SPDLOG_CRITICAL("CMM application failed: {}", ex.what());
    return 1;
  }

  return 0;
}
