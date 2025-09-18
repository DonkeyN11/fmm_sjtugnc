/**
 * Fast map matching.
 *
 * UBODT Cache Daemon - Background service for UBODT caching
 *
 * @author: Generated for FMM optimization
 * @version: 2025.01.01
 */

#ifndef FMM_UBODT_DAEMON_HPP_
#define FMM_UBODT_DAEMON_HPP_

#include "mm/fmm/ubodt_memory_manager.hpp"
#include <string>
#include <thread>
#include <atomic>
#include <memory>
#include <mutex>

namespace FMM {
namespace MM {

/**
 * UBODT Cache Daemon - Runs in background to provide persistent caching
 */
class UBODTCacheDaemon {
public:
    static UBODTCacheDaemon& get_instance() {
        static UBODTCacheDaemon instance;
        return instance;
    }

    /**
     * Start the daemon
     * @param socket_path Path for Unix domain socket
     * @return true if started successfully
     */
    bool start(const std::string& socket_path = "/tmp/fmm_ubodt_daemon.sock");

    /**
     * Stop the daemon
     */
    void stop();

    /**
     * Check if daemon is running
     * @return true if daemon is running
     */
    bool is_running() const;

    /**
     * Get daemon status
     * @return Status string
     */
    std::string get_status() const;

private:
    UBODTCacheDaemon();
    ~UBODTCacheDaemon();

    // Non-copyable
    UBODTCacheDaemon(const UBODTCacheDaemon&) = delete;
    UBODTCacheDaemon& operator=(const UBODTCacheDaemon&) = delete;

    /**
     * Main daemon loop
     */
    void daemon_loop();

    /**
     * Handle client connection
     * @param client_fd Client socket file descriptor
     */
    void handle_client(int client_fd);

    /**
     * Process client command
     * @param command Command string
     * @return Response string
     */
    std::string process_command(const std::string& command);

    /**
     * Load UBODT into cache
     * @param filename UBODT file path
     * @param multiplier UBODT multiplier
     * @param max_memory_mb Memory limit in MB
     * @return Response string
     */
    std::string load_ubodt(const std::string& filename, int multiplier, size_t max_memory_mb);

    /**
     * Get cache status
     * @return Response string
     */
    std::string get_cache_status();

    /**
     * Clear cache
     * @return Response string
     */
    std::string clear_cache();

    /**
     * Unload specific UBODT
     * @param filename UBODT file path
     * @return Response string
     */
    std::string unload_ubodt(const std::string& filename);

    /**
     * Setup signal handlers
     */
    void setup_signal_handlers();

    std::string socket_path_;
    int server_fd_;
    std::atomic<bool> running_;
    std::unique_ptr<std::thread> daemon_thread_;
    mutable std::mutex mutex_;
};

} // namespace MM
} // namespace FMM

#endif // FMM_UBODT_DAEMON_HPP_