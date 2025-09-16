#include "mm/fmm/ubodt_gen_app_config.hpp"
#include "util/util.hpp"
#include "util/debug.hpp"

using namespace FMM;
using namespace FMM::CORE;
using namespace FMM::NETWORK;
using namespace FMM::MM;
using namespace FMM::CONFIG;
UBODTGenAppConfig::UBODTGenAppConfig(int argc, char **argv) {
  spdlog::set_pattern("[%^%l%$][%s:%-3#] %v");
  if (argc==2) {
    std::string configfile(argv[1]);
    if (UTIL::check_file_extension(configfile,"xml,XML"))
      load_xml(configfile);
    else {
      load_arg(argc,argv);
    }
  } else {
    load_arg(argc,argv);
  }
  spdlog::set_level((spdlog::level::level_enum) log_level);
  if (!help_specified)
    print();
}

void UBODTGenAppConfig::load_xml(const std::string &file) {
  SPDLOG_INFO("Read configuration from xml file: {}",file);
  // Create empty property tree object
  boost::property_tree::ptree tree;
  boost::property_tree::read_xml(file, tree);
  network_config = NetworkConfig::load_from_xml(tree);
  delta = tree.get("config.parameters.delta", 3000.0);
  result_file = tree.get<std::string>("config.output.file");
  // 0-trace,1-debug,2-info,3-warn,4-err,5-critical,6-off
  log_level = tree.get("config.other.log_level", 2);
  use_omp = !(!tree.get_child_optional("config.other.use_omp"));
  use_chunking = !(!tree.get_child_optional("config.other.use_chunking"));
  chunk_rows = tree.get("config.other.chunk_rows", 4);
  chunk_cols = tree.get("config.other.chunk_cols", 4);
  chunk_threads = tree.get("config.other.chunk_threads", 8);
  SPDLOG_INFO("Read configuration from xml file done");
}

void UBODTGenAppConfig::load_arg(int argc, char **argv) {
  SPDLOG_INFO("Start reading ubodt configuration from arguments");
  cxxopts::Options options("config",
                           "Configuration parser of ubodt_gen");
  // Register options
  NetworkConfig::register_arg(options);
  options.add_options()
    ("delta", "Upperbound distance",
    cxxopts::value<double>()->default_value("3000.0"))
    ("o,output", "Output file name",
    cxxopts::value<std::string>()->default_value(""))
    ("l,log_level", "Log level", cxxopts::value<int>()->default_value("2"))
    ("h,help",   "Help information")
    ("use_omp","Use parallel computing if specified")
    ("use_chunking","Use chunk-based parallel processing if specified")
    ("chunk_rows","Number of chunk rows for grid partitioning",cxxopts::value<int>()->default_value("4"))
    ("chunk_cols","Number of chunk columns for grid partitioning",cxxopts::value<int>()->default_value("4"))
    ("chunk_threads","Number of threads for chunk processing",cxxopts::value<int>()->default_value("8"));
  if (argc==1) {
    help_specified = true;
    return;
  }
  // Parse options
  auto result = options.parse(argc, argv);
  // Read options
  result_file = result["output"].as<std::string>();
  network_config =  NetworkConfig::load_from_arg(result);
  log_level = result["log_level"].as<int>();
  delta = result["delta"].as<double>();
  use_omp = result.count("use_omp")>0;
  use_chunking = result.count("use_chunking")>0;
  if (result.count("chunk_rows")>0) {
    chunk_rows = result["chunk_rows"].as<int>();
  }
  if (result.count("chunk_cols")>0) {
    chunk_cols = result["chunk_cols"].as<int>();
  }
  if (result.count("chunk_threads")>0) {
    chunk_threads = result["chunk_threads"].as<int>();
  }
  if (result.count("help")>0) {
    help_specified = true;
  }
  SPDLOG_INFO("Finish with reading ubodt arg configuration");
}

void UBODTGenAppConfig::print() const {
  SPDLOG_INFO("----    Print configuration   ----");
  network_config.print();
  SPDLOG_INFO("Delta {}",delta);
  SPDLOG_INFO("Output file {}",result_file);
  SPDLOG_INFO("Log level {}",UTIL::LOG_LEVESLS[log_level]);
  SPDLOG_INFO("Use omp {}",(use_omp ? "true" : "false"));
  SPDLOG_INFO("Use chunking {}",(use_chunking ? "true" : "false"));
  if (use_chunking) {
    SPDLOG_INFO("Chunk grid {}x{}", chunk_rows, chunk_cols);
    SPDLOG_INFO("Chunk threads {}", chunk_threads);
  }
  SPDLOG_INFO("---- Print configuration done ----");
}

void UBODTGenAppConfig::print_help() {
  std::ostringstream oss;
  oss << "ubodt_gen argument lists:\n";
  NetworkConfig::register_help(oss);
  oss << "--delta (optional) <double>: upperbound (3000.0)\n";
  oss << "-o/--output (required) <string>: Output file name\n";
  oss << "-l/--log_level (optional) <int>: log level (2)\n";
  oss << "--use_omp: use OpenMP or not\n";
  oss << "--use_chunking: use chunk-based parallel processing\n";
  oss << "--chunk_rows (optional) <int>: chunk rows (4)\n";
  oss << "--chunk_cols (optional) <int>: chunk columns (4)\n";
  oss << "--chunk_threads (optional) <int>: chunk threads (8)\n";
  oss << "-h/--help: help information\n";
  oss << "For xml configuration, check example folder\n";
  std::cout<<oss.str();
}

bool UBODTGenAppConfig::validate() const {
  SPDLOG_INFO("Validating configuration for UBODT construction");
  if (!network_config.validate()) {
    return false;
  }
  if (UTIL::file_exists(result_file)) {
    SPDLOG_WARN("Overwrite result file {}", result_file);
  }
  std::string output_folder = UTIL::get_file_directory(result_file);
  if (!UTIL::folder_exist(output_folder)) {
    SPDLOG_CRITICAL("Output folder {} not exists", output_folder);
    return false;
  }
  if (log_level < 0 || log_level > UTIL::LOG_LEVESLS.size()) {
    SPDLOG_CRITICAL("Invalid log_level {}, which should be 0 - 6", log_level);
    SPDLOG_INFO("0-trace,1-debug,2-info,3-warn,4-err,5-critical,6-off");
    return false;
  }
  if (delta <= 0) {
    SPDLOG_CRITICAL("Delta {} should be positive");
    return false;
  }
  SPDLOG_INFO("Validating done.");
  return true;
}

bool UBODTGenAppConfig::is_binary_output() const {
  if (UTIL::check_file_extension(result_file,"bin")) {
    return true;
  }
  return false;
}
