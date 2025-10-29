/**
 * Covariance-based map matching.
 *
 * CMM application configuration management
 *
 * @author: Chenzhang Ning
 * @version: 2025.09.29
 */

#ifndef FMM_CMM_APP_CONFIG_H_
#define FMM_CMM_APP_CONFIG_H_

#include "mm/cmm/cmm_algorithm.hpp"
#include "config/network_config.hpp"
#include "config/gps_config.hpp"
#include "config/result_config.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include "cxxopts.hpp"

#include <string>
#include <sstream>

namespace FMM {
namespace MM {

/**
 * Configuration class for CMM application
 */
struct CMMAppConfig {
    FMM::CONFIG::NetworkConfig network_config;
    FMM::CONFIG::GPSConfig gps_config;
    FMM::CONFIG::ResultConfig result_config;
    CovarianceMapMatchConfig cmm_config;
    std::string ubodt_file;
    bool use_omp;
    int log_level;
    int step;
    bool help_specified;

    /**
     * Constructor
     */
    CMMAppConfig();

    /**
     * Load configuration from XML file
     * @param xml_file XML configuration file path
     * @return CMMAppConfig object
     */
    static CMMAppConfig load_from_xml(const std::string &xml_file);

    /**
     * Load configuration from command line arguments
     * @param arg_data command line arguments
     * @return CMMAppConfig object
     */
    static CMMAppConfig load_from_arg(const cxxopts::ParseResult &arg_data);

    /**
     * Register command line arguments
     * @param options command line options object
     */
    static void register_arg(cxxopts::Options &options);

    /**
     * Register help information
     * @param oss output stream
     */
    static void register_help(std::ostringstream &oss);

    /**
     * Print configuration information
     */
    void print() const;

    /**
     * Validate configuration
     * @return true if valid
     */
    bool validate() const;
};

} // MM
} // FMM

#endif // FMM_CMM_APP_CONFIG_H_
