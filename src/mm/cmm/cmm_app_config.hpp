/**
 * Covariance-based map matching.
 *
 * CMM application configuration management
 *
 * @author: Generated for CMM implementation
 * @version: 2025.01.01
 */

#ifndef FMM_CMM_APP_CONFIG_H_
#define FMM_CMM_APP_CONFIG_H_

#include "mm/cmm/cmm_algorithm.hpp"
#include "config/gps_config.hpp"
#include "config/result_config.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include "cxxopts/cxxopts.hpp"

#include <string>

namespace FMM {
namespace MM {

/**
 * Configuration class for CMM application
 */
struct CMMAppConfig {
    FMM::CONFIG::GPSConfig gps_config;
    FMM::CONFIG::ResultConfig result_config;
    CovarianceMapMatchConfig cmm_config;
    bool use_omp;
    int log_level;

    /**
     * Constructor
     */
    CMMAppConfig() : use_omp(false), log_level(2) {}

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

/**
 * CMM application class
 */
class CMMApp {
public:
    /**
     * Constructor
     * @param config CMM application configuration
     * @param network road network
     * @param graph network graph
     * @param ubodt UBODT table
     */
    CMMApp(const CMMAppConfig &config,
           const NETWORK::Network &network,
           const NETWORK::NetworkGraph &graph,
           std::shared_ptr<UBODT> ubodt);

    /**
     * Run the CMM application
     */
    void run();

private:
    CMMAppConfig config_;
    const NETWORK::Network &network_;
    const NETWORK::NetworkGraph &graph_;
    std::shared_ptr<UBODT> ubodt_;
    std::unique_ptr<CovarianceMapMatch> cmm_;
};

} // MM
} // FMM

#endif // FMM_CMM_APP_CONFIG_H_