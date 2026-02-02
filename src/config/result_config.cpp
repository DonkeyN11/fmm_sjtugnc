#include "config/result_config.hpp"
#include "util/util.hpp"
#include "util/debug.hpp"
#include <set>

void FMM::CONFIG::ResultConfig::print() const {
  std::ostringstream ss;
  if (output_config.write_opath)
    ss << "opath ";
  if (output_config.write_pgeom)
    ss << "pgeom ";
  if (output_config.write_offset)
    ss << "offset ";
  if (output_config.write_error)
    ss << "error ";
  if (output_config.write_spdist)
    ss << "spdist ";
  if (output_config.write_sp_dist)
    ss << "sp_dist ";
  if (output_config.write_eu_dist)
    ss << "eu_dist ";
  if (output_config.write_cpath)
    ss << "cpath ";
  if (output_config.write_tpath)
    ss << "tpath ";
  if (output_config.write_mgeom)
    ss << "mgeom ";
  if (output_config.write_ep)
    ss << "ep ";
  if (output_config.write_tp)
    ss << "tp ";
  if (output_config.write_trustworthiness)
    ss << "trustworthiness ";
  if (output_config.write_n_best_trustworthiness)
    ss << "n_best_trustworthiness ";
  if (output_config.write_cumu_prob)
    ss << "cumu_prob ";
  if (output_config.write_candidates)
    ss << "candidates ";
  if (output_config.write_length)
    ss << "length ";
  if (output_config.write_duration)
    ss << "duration ";
  if (output_config.write_speed)
    ss << "speed ";
  if (output_config.write_timestamp)
    ss << "timestamp ";
  if (output_config.write_seq)
    ss << "seq ";
  if (output_config.write_ogeom)
    ss << "ogeom ";
  if (output_config.point_mode)
    ss << "point_mode ";
  SPDLOG_INFO("ResultConfig");
  SPDLOG_INFO("File: {}",file);
  SPDLOG_INFO("Fields: {}",ss.str());
};

std::string FMM::CONFIG::ResultConfig::to_string() const{
  std::ostringstream oss;
  oss << "Result file : " << file << "\n";
  oss << "Output fields: ";
  if (output_config.write_opath)
    oss << "opath ";
  if (output_config.write_pgeom)
    oss << "pgeom ";
  if (output_config.write_offset)
    oss << "offset ";
  if (output_config.write_error)
    oss << "error ";
  if (output_config.write_spdist)
    oss << "spdist ";
  if (output_config.write_sp_dist)
    oss << "sp_dist ";
  if (output_config.write_eu_dist)
    oss << "eu_dist ";
  if (output_config.write_cpath)
    oss << "cpath ";
  if (output_config.write_tpath)
    oss << "tpath ";
  if (output_config.write_mgeom)
    oss << "mgeom ";
  if (output_config.write_ep)
    oss << "ep ";
  if (output_config.write_tp)
    oss << "tp ";
  if (output_config.write_trustworthiness)
    oss << "trustworthiness ";
  if (output_config.write_n_best_trustworthiness)
    oss << "n_best_trustworthiness ";
  if (output_config.write_cumu_prob)
    oss << "cumu_prob ";
  if (output_config.write_candidates)
    oss << "candidates ";
  if (output_config.write_length)
    oss << "length ";
  if (output_config.write_duration)
    oss << "duration ";
  if (output_config.write_speed)
    oss << "speed ";
  if (output_config.write_timestamp)
    oss << "timestamp ";
  if (output_config.write_seq)
    oss << "seq ";
  if (output_config.write_ogeom)
    oss << "ogeom ";
  if (output_config.point_mode)
    oss << "point_mode ";
  return oss.str();
};

FMM::CONFIG::ResultConfig FMM::CONFIG::ResultConfig::load_from_xml(
  const boost::property_tree::ptree &xml_data) {
  ResultConfig config;
  config.file = xml_data.get<std::string>("config.output.file");
  if (xml_data.get_child_optional("config.output.fields")) {
    // Fields specified
    // close the default output fields (cpath,mgeom are true by default)
    config.output_config.write_cpath = false;
    config.output_config.write_mgeom = false;
    auto fields = xml_data.get_child("config.output.fields");
    for (auto& v : fields) {
      if (v.first == "seq") {
        config.output_config.write_seq = true;
      } else if (v.first == "opath") {
        config.output_config.write_opath = true;
      } else if (v.first == "cpath") {
        config.output_config.write_cpath = true;
      } else if (v.first == "tpath") {
        config.output_config.write_tpath = true;
      } else if (v.first == "mgeom") {
        config.output_config.write_mgeom = true;
      } else if (v.first == "pgeom") {
        config.output_config.write_pgeom = true;
      } else if (v.first == "offset") {
        config.output_config.write_offset = true;
      } else if (v.first == "error") {
        config.output_config.write_error = true;
      } else if (v.first == "spdist") {
        config.output_config.write_spdist = true;
      } else if (v.first == "sp_dist") {
        config.output_config.write_sp_dist = true;
      } else if (v.first == "eu_dist") {
        config.output_config.write_eu_dist = true;
      } else if (v.first == "ep") {
        config.output_config.write_ep = true;
      } else if (v.first == "tp") {
        config.output_config.write_tp = true;
      } else if (v.first == "trustworthiness") {
        config.output_config.write_trustworthiness = true;
      } else if (v.first == "n_best_trustworthiness") {
        config.output_config.write_n_best_trustworthiness = true;
      } else if (v.first == "cumu_prob") {
        config.output_config.write_cumu_prob = true;
      } else if (v.first == "candidates") {
        config.output_config.write_candidates = true;
      } else if (v.first == "length") {
        config.output_config.write_length = true;
      } else if (v.first == "duration") {
        config.output_config.write_duration = true;
      } else if (v.first == "speed") {
        config.output_config.write_speed = true;
      } else if (v.first == "timestamp") {
        config.output_config.write_timestamp = true;
      } else if (v.first == "ogeom") {
        config.output_config.write_ogeom = true;
      } else if (v.first == "point_mode") {
        config.output_config.point_mode = true;
      } else if (v.first == "all") {
        config.output_config.write_opath = true;
        config.output_config.write_pgeom = true;
        config.output_config.write_offset = true;
        config.output_config.write_error = true;
        config.output_config.write_spdist = true;
        config.output_config.write_sp_dist = true;
        config.output_config.write_eu_dist = true;
        config.output_config.write_cpath = true;
        config.output_config.write_mgeom = true;
        config.output_config.write_tpath = true;
        config.output_config.write_ep = true;
        config.output_config.write_tp = true;
        config.output_config.write_trustworthiness = true;
        config.output_config.write_n_best_trustworthiness = true;
        config.output_config.write_cumu_prob = true;
        config.output_config.write_candidates = true;
        config.output_config.write_length = true;
        config.output_config.write_duration = true;
        config.output_config.write_speed = true;
        config.output_config.write_timestamp = true;
      }
    }
  }
  return config;
};

FMM::CONFIG::ResultConfig FMM::CONFIG::ResultConfig::load_from_arg(
  const cxxopts::ParseResult &arg_data) {
  FMM::CONFIG::ResultConfig config;
  config.file = arg_data["output"].as<std::string>();
  if (arg_data.count("seq") > 0 && arg_data["seq"].as<bool>()) {
    config.output_config.write_seq = true;
  }
  if (arg_data.count("output_fields") > 0) {
    config.output_config.write_cpath = false;
    config.output_config.write_mgeom = false;
    std::string fields = arg_data["output_fields"].as<std::string>();
    std::set<std::string> dict = string2set(fields);
    if (dict.find("seq") != dict.end()) {
      config.output_config.write_seq = true;
    }
    if (dict.find("opath") != dict.end()) {
      config.output_config.write_opath = true;
    }
    if (dict.find("cpath") != dict.end()) {
      config.output_config.write_cpath = true;
    }
    if (dict.find("mgeom") != dict.end()) {
      config.output_config.write_mgeom = true;
    }
    if (dict.find("tpath") != dict.end()) {
      config.output_config.write_tpath = true;
    }
    if (dict.find("pgeom") != dict.end()) {
      config.output_config.write_pgeom = true;
    }
    if (dict.find("offset") != dict.end()) {
      config.output_config.write_offset = true;
    }
    if (dict.find("error") != dict.end()) {
      config.output_config.write_error = true;
    }
    if (dict.find("spdist") != dict.end()) {
      config.output_config.write_spdist = true;
    }
    if (dict.find("sp_dist") != dict.end()) {
      config.output_config.write_sp_dist = true;
    }
    if (dict.find("eu_dist") != dict.end()) {
      config.output_config.write_eu_dist = true;
    }
    if (dict.find("ep") != dict.end()) {
      config.output_config.write_ep = true;
    }
    if (dict.find("tp") != dict.end()) {
      config.output_config.write_tp = true;
    }
    if (dict.find("trustworthiness") != dict.end()) {
      config.output_config.write_trustworthiness = true;
    }
    if (dict.find("n_best_trustworthiness") != dict.end()) {
      config.output_config.write_n_best_trustworthiness = true;
    }
    if (dict.find("cumu_prob") != dict.end()) {
      config.output_config.write_cumu_prob = true;
    }
    if (dict.find("candidates") != dict.end()) {
      config.output_config.write_candidates = true;
    }
    if (dict.find("length") != dict.end()) {
      config.output_config.write_length = true;
    }
    if (dict.find("duration") != dict.end()) {
      config.output_config.write_duration = true;
    }
    if (dict.find("speed") != dict.end()) {
      config.output_config.write_speed = true;
    }
    if (dict.find("timestamp") != dict.end()) {
      config.output_config.write_timestamp = true;
    }
    if (dict.find("ogeom") != dict.end()) {
      config.output_config.write_ogeom = true;
    }
    if (dict.find("point_mode") != dict.end()) {
      config.output_config.point_mode = true;
    }
    if (dict.find("all") != dict.end()) {
      config.output_config.write_opath = true;
      config.output_config.write_pgeom = true;
      config.output_config.write_offset = true;
      config.output_config.write_error = true;
      config.output_config.write_spdist = true;
      config.output_config.write_sp_dist = true;
      config.output_config.write_eu_dist = true;
      config.output_config.write_cpath = true;
      config.output_config.write_mgeom = true;
      config.output_config.write_tpath = true;
      config.output_config.write_ep = true;
      config.output_config.write_tp = true;
      config.output_config.write_trustworthiness = true;
      config.output_config.write_n_best_trustworthiness = true;
      config.output_config.write_cumu_prob = true;
      config.output_config.write_candidates = true;
      config.output_config.write_length = true;
      config.output_config.write_duration = true;
      config.output_config.write_speed = true;
      config.output_config.write_timestamp = true;
    }
  }
  return config;
};

std::set<std::string> FMM::CONFIG::ResultConfig::string2set(
  const std::string &s) {
  char delim = ',';
  std::set<std::string> result;
  std::stringstream ss(s);
  std::string intermediate;
  while (getline(ss, intermediate, delim)) {
    result.insert(intermediate);
  }
  return result;
};

bool FMM::CONFIG::ResultConfig::validate() const {
  if (UTIL::file_exists(file))
  {
    SPDLOG_WARN("Overwrite existing result file {}",file);
  };
  std::string output_folder = UTIL::get_file_directory(file);
  if (!UTIL::folder_exist(output_folder)) {
    SPDLOG_CRITICAL("Output folder {} not exists",output_folder);
    return false;
  }
  return true;
};

void FMM::CONFIG::ResultConfig::register_arg(cxxopts::Options &options){
  options.add_options()
    ("o,output","Output file name",
    cxxopts::value<std::string>()->default_value(""))
    ("output_fields","Output fields",
    cxxopts::value<std::string>()->default_value(""))
    ("seq","Output sequence number of each point",
    cxxopts::value<bool>()->default_value("false"))
    ("point_mode","Output in point mode (one row per matched point)",
    cxxopts::value<bool>()->default_value("true"));
};

void FMM::CONFIG::ResultConfig::register_help(std::ostringstream &oss){
  oss<<"--output (required) <string>: Output file name\n";
  oss<<"--output_fields (optional) <string>: Output fields\n";
  oss<<"  opath,cpath,tpath,mgeom,pgeom,ogeom,\n";
  oss<<"  offset,error,spdist,sp_dist,eu_dist,tp,ep,trustworthiness,n_best_trustworthiness,candidates,cumu_prob,length,duration,speed,timestamp,seq,point_mode,all\n";
};
