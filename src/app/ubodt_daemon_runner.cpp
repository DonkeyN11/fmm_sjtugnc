/**
 * Fast map matching.
 *
 * UBODT Cache Daemon Runner - Simple daemon launcher for testing
 *
 * @author: Generated for FMM optimization
 * @version: 2025.01.01
 */

#include "mm/fmm/ubodt_daemon.hpp"
#include "util/debug.hpp"
#include <iostream>
#include <string>
#include <signal.h>

using namespace FMM;
using namespace FMM::MM;

int main(int argc, char **argv) {
    // Initialize SPDLOG for console output
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%l][%s:%# ] %v");

    std::string socket_path = "/tmp/fmm_ubodt_daemon.sock";

    if (argc > 1) {
        socket_path = argv[1];
    }

    std::cout << "Starting UBODT Cache Daemon on socket: " << socket_path << std::endl;
    std::cout << "Press Ctrl+C to stop the daemon." << std::endl;

    // Setup signal handlers
    signal(SIGINT, [](int signum) {
        std::cout << "\nReceived signal " << signum << ", stopping daemon..." << std::endl;
        UBODTCacheDaemon::get_instance().stop();
        exit(0);
    });

    signal(SIGTERM, [](int signum) {
        std::cout << "\nReceived signal " << signum << ", stopping daemon..." << std::endl;
        UBODTCacheDaemon::get_instance().stop();
        exit(0);
    });

    // Start daemon
    auto& daemon = UBODTCacheDaemon::get_instance();
    if (!daemon.start(socket_path)) {
        std::cerr << "Failed to start daemon." << std::endl;
        return 1;
    }

    // Keep main thread alive
    while (daemon.is_running()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}