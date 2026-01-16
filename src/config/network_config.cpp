#include "config/network_config.hpp"
#include "util/util.hpp"
#include "util/debug.hpp"

#include <cctype>

namespace {

std::string trim_copy(const std::string &input) {
  std::size_t start = 0;
  while (start < input.size() && std::isspace(static_cast<unsigned char>(input[start]))) {
    ++start;
  }
  std::size_t end = input.size();
  while (end > start && std::isspace(static_cast<unsigned char>(input[end - 1]))) {
    --end;
  }
  return input.substr(start, end - start);
}

bool parse_bbox_string(const std::string &bbox_str,
                       double *minx,
                       double *miny,
                       double *maxx,
                       double *maxy) {
  if (bbox_str.empty() || !minx || !miny || !maxx || !maxy) {
    return false;
  }
  auto parts = FMM::UTIL::split_string(bbox_str);
  if (parts.size() != 4) {
    return false;
  }
  try {
    *minx = std::stod(trim_copy(parts[0]));
    *miny = std::stod(trim_copy(parts[1]));
    *maxx = std::stod(trim_copy(parts[2]));
    *maxy = std::stod(trim_copy(parts[3]));
  } catch (...) {
    return false;
  }
  return *minx <= *maxx && *miny <= *maxy;
}

} // namespace

void FMM::CONFIG::NetworkConfig::print() const{
  SPDLOG_INFO("NetworkConfig");
  SPDLOG_INFO("File name: {} ",file);
  SPDLOG_INFO("ID name: {} ",id);
  SPDLOG_INFO("Source name: {} ",source);
  SPDLOG_INFO("Target name: {} ",target);
  if (!cache.empty()) {
    SPDLOG_INFO("Cache file: {} ", cache);
  }
  if (has_bbox) {
    SPDLOG_INFO("BBox: {},{}, {},{}",
                bbox_minx, bbox_miny, bbox_maxx, bbox_maxy);
  }
};

FMM::CONFIG::NetworkConfig FMM::CONFIG::NetworkConfig::load_from_xml(
  const boost::property_tree::ptree &xml_data){
  std::string file = xml_data.get<std::string>("config.input.network.file");
  std::string id = xml_data.get("config.input.network.id", "id");
  std::string source = xml_data.get("config.input.network.source","source");
  std::string target = xml_data.get("config.input.network.target","target");
  FMM::CONFIG::NetworkConfig config{file, id, source, target};
  config.cache = xml_data.get("config.input.network.cache", "");
  std::string bbox = xml_data.get("config.input.network.bbox", "");
  if (!bbox.empty()) {
    config.has_bbox = parse_bbox_string(bbox,
                                        &config.bbox_minx,
                                        &config.bbox_miny,
                                        &config.bbox_maxx,
                                        &config.bbox_maxy);
    config.bbox_valid = config.has_bbox;
  }
  return config;
};

FMM::CONFIG::NetworkConfig FMM::CONFIG::NetworkConfig::load_from_arg(
  const cxxopts::ParseResult &arg_data){
  std::string file = arg_data["network"].as<std::string>();
  std::string id = arg_data["network_id"].as<std::string>();
  std::string source = arg_data["source"].as<std::string>();
  std::string target = arg_data["target"].as<std::string>();
  FMM::CONFIG::NetworkConfig config{file, id, source, target};
  if (arg_data.count("network_cache") > 0) {
    config.cache = arg_data["network_cache"].as<std::string>();
  }
  if (arg_data.count("network_bbox") > 0) {
    std::string bbox = arg_data["network_bbox"].as<std::string>();
    if (!bbox.empty()) {
      config.has_bbox = parse_bbox_string(bbox,
                                          &config.bbox_minx,
                                          &config.bbox_miny,
                                          &config.bbox_maxx,
                                          &config.bbox_maxy);
      config.bbox_valid = config.has_bbox;
    }
  }
  return config;
};

void FMM::CONFIG::NetworkConfig::register_arg(cxxopts::Options &options){
  options.add_options()
  ("network","Network file name",
  cxxopts::value<std::string>()->default_value(""))
  ("network_id","Network id name",
  cxxopts::value<std::string>()->default_value("id"))
  ("source","Network source name",
  cxxopts::value<std::string>()->default_value("source"))
  ("target","Network target name",
  cxxopts::value<std::string>()->default_value("target"))
  ("network_cache","Network cache file (optional)",
  cxxopts::value<std::string>()->default_value(""))
  ("network_bbox","Network bbox filter minx,miny,maxx,maxy (optional)",
  cxxopts::value<std::string>()->default_value(""));
};

void FMM::CONFIG::NetworkConfig::register_help(std::ostringstream &oss){
  oss<<"--network (required) <string>: Network file name\n";
  oss<<"--network_id (optional) <string>: Network id name (id)\n";
  oss<<"--source (optional) <string>: Network source name (source)\n";
  oss<<"--target (optional) <string>: Network target name (target)\n";
  oss<<"--network_cache (optional) <string>: Network cache file\n";
  oss<<"--network_bbox (optional) <string>: Network bbox minx,miny,maxx,maxy\n";
};

bool FMM::CONFIG::NetworkConfig::is_shapefile_format() const {
  if (FMM::UTIL::check_file_extension(file,"shp,gpkg,geojson,fgb"))
    return true;
  return false;
};

bool FMM::CONFIG::NetworkConfig::validate() const {
  if (!UTIL::file_exists(file)){
    SPDLOG_CRITICAL("Network file not found {}",file);
    return false;
  }
  if (!bbox_valid) {
    SPDLOG_CRITICAL("Invalid network bbox format; expected minx,miny,maxx,maxy");
    return false;
  }
  bool shapefile_format = is_shapefile_format();
  if (shapefile_format){
    return true;
  }
  SPDLOG_CRITICAL("Network format not recognized {}",file);
  return false;
}
