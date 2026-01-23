/**
 * Fast map matching.
 *
 * UBODT Manager command line program
 *
 * @version: 2025.01.23
 */

#ifndef FMM_SRC_MM_FMM_UBODT_MANAGE_APP_HPP_
#define FMM_SRC_MM_FMM_UBODT_MANAGE_APP_HPP_

#include "mm/fmm/ubodt_manage_app_config.hpp"
#include "mm/fmm/ubodt_manager.hpp"
#include "mm/fmm/ubodt.hpp"

namespace FMM {
namespace MM {

/**
 * UBODT Manager application class
 *
 * This application provides command-line interface to manage UBODT
 * instances in memory, allowing them to be loaded once and reused
 * across multiple FMM/CMM operations.
 */
class UBODTManageApp {
public:
    /**
     * Create application from configuration
     * @param config Application configuration
     */
    explicit UBODTManageApp(const UBODTManageAppConfig &config);

    /**
     * Run the application
     * @return 0 on success, non-zero on failure
     */
    int run();

private:
    const UBODTManageAppConfig &config_;

    /**
     * Load UBODT into memory
     * @return true on success
     */
    bool load_ubodt();

    /**
     * Release specific UBODT from memory
     * @return true on success
     */
    bool release_ubodt();

    /**
     * Release all UBODTs from memory
     * @return true on success
     */
    bool release_all();

    /**
     * Show status of loaded UBODTs
     * @return true on success
     */
    bool show_status();
};

} // namespace MM
} // namespace FMM

#endif // FMM_SRC_MM_FMM_UBODT_MANAGE_APP_HPP_
