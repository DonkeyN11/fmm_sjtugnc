/**
 * Covariance-based map matching.
 *
 * CMM command line application entry point
 * 
 * @author: Chenzhang Ning
 * @version: 2025.09.29
 */

#ifndef FMM_CMM_APP_HPP_
#define FMM_CMM_APP_HPP_

#include "mm/cmm/cmm_app_config.hpp"
#include "mm/cmm/cmm_algorithm.hpp"
#include "network/network.hpp"
#include "network/network_graph.hpp"
#include "mm/fmm/ubodt.hpp"

#include <memory>

namespace FMM {
namespace MM {

/**
 * Covariance-based map matching application.
 */
class CMMApp {
public:
    /**
     * Construct the application from configuration.
     * @param config Parsed application configuration
     */
    explicit CMMApp(const CMMAppConfig &config);

    /**
     * Construct the application from configuration with a preloaded UBODT.
     * @param config Parsed application configuration
     * @param preloaded_ubodt Shared UBODT instance to reuse
     */
    CMMApp(const CMMAppConfig &config, std::shared_ptr<UBODT> preloaded_ubodt);

    /**
     * Run covariance-based map matching workflow.
     */
    void run();

private:
    const CMMAppConfig &config_;
    NETWORK::Network network_;
    NETWORK::NetworkGraph graph_;
    std::shared_ptr<UBODT> ubodt_;
    std::unique_ptr<CovarianceMapMatch> cmm_;

    void initialize_matcher();
};

} // namespace MM
} // namespace FMM

#endif // FMM_CMM_APP_HPP_
