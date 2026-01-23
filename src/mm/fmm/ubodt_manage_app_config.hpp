/**
 * Fast map matching.
 *
 * UBODT Manager command line program configuration
 *
 * @version: 2025.01.23
 */

#ifndef FMM_SRC_MM_FMM_UBODT_MANAGE_APP_CONFIG_HPP_
#define FMM_SRC_MM_FMM_UBODT_MANAGE_APP_CONFIG_HPP_

#include <string>
#include <iostream>

namespace FMM {
namespace MM {

/**
 * Configuration class for UBODT Manager command line program
 */
class UBODTManageAppConfig {
public:
    /**
     * Constructor of the configuration from command line arguments.
     * @param argc number of arguments
     * @param argv raw argument data
     */
    UBODTManageAppConfig(int argc, char **argv);

    /**
     * Validate the configuration
     * @return true if valid
     */
    bool validate() const;

    /**
     * Print configuration data
     */
    void print() const;

    /**
     * Print help information
     */
    static void print_help();

    // Operation type
    enum class Operation {
        LOAD,          // Load UBODT into memory
        RELEASE,       // Release specific UBODT
        RELEASE_ALL,   // Release all UBODTs
        STATUS,        // Show status of loaded UBODTs
        UNKNOWN
    };

    Operation operation = Operation::UNKNOWN;
    std::string ubodt_file;        // UBODT file path
    std::string network_file;      // Network file path (for validation)
    int multiplier = 1;            // Hash table multiplier
    bool help_specified = false;   // Help is specified or not
    int log_level = 2;             // log level, 0-trace,1-debug,2-info,
                                   // 3-warn,4-err,5-critical,6-off
    bool verbose = false;          // Verbose output
};

} // namespace MM
} // namespace FMM

#endif // FMM_SRC_MM_FMM_UBODT_MANAGE_APP_CONFIG_HPP_
