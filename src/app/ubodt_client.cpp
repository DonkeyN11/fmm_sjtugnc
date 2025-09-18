/**
 * Fast map matching.
 *
 * UBODT Cache Client - Communicates with UBODT Cache Daemon
 *
 * @author: Generated for FMM optimization
 * @version: 2025.01.01
 */

#include "mm/fmm/ubodt_daemon.hpp"
#include "util/debug.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <cstdlib>

using namespace FMM;
using namespace FMM::MM;

void print_help() {
    std::cout << "UBODT Client - Control UBODT Cache Daemon\n\n";
    std::cout << "Usage: ubodt_client [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --load FILE          Load UBODT file into daemon cache\n";
    std::cout << "  --multiplier N       UBODT multiplier (default: 50000)\n";
    std::cout << "  --max_memory MB      Maximum memory usage in MB\n";
    std::cout << "  --status             Show cache status\n";
    std::cout << "  --clear              Clear all cached UBODT files\n";
    std::cout << "  --unload FILE        Unload specific UBODT file\n";
    std::cout << "  --daemon-status      Show daemon status\n";
    std::cout << "  --start              Start daemon\n";
    std::cout << "  --stop               Stop daemon\n";
    std::cout << "  --socket PATH        Socket path (default: /tmp/fmm_ubodt_daemon.sock)\n";
    std::cout << "  --help               Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  ubodt_client --load data/ubodt.bin\n";
    std::cout << "  ubodt_client --status\n";
    std::cout << "  ubodt_client --start\n";
    std::cout << "  ubodt_client --stop\n";
    std::cout << "  ubodt_client --daemon-status\n";
}

std::string send_command(const std::string& socket_path, const std::string& command) {
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
        return "ERROR: Failed to create socket\n";
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(sock);
        return "ERROR: Cannot connect to daemon. Is it running?\n";
    }

    if (write(sock, command.c_str(), command.length()) < 0) {
        close(sock);
        return "ERROR: Failed to send command\n";
    }

    char buffer[4096];
    ssize_t bytes_read = read(sock, buffer, sizeof(buffer) - 1);
    if (bytes_read < 0) {
        close(sock);
        return "ERROR: Failed to read response\n";
    }

    buffer[bytes_read] = '\0';
    close(sock);
    return std::string(buffer);
}

bool check_daemon_running(const std::string& socket_path) {
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
        return false;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

    bool connected = (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0);
    close(sock);
    return connected;
}

void start_daemon(const std::string& socket_path) {
    if (check_daemon_running(socket_path)) {
        std::cout << "Daemon is already running.\n";
        return;
    }

    // Fork daemon process
    pid_t pid = fork();
    if (pid < 0) {
        std::cerr << "Failed to fork daemon process.\n";
        return;
    }

    if (pid > 0) {
        // Parent process
        std::cout << "Starting daemon with PID " << pid << "...\n";
        sleep(1); // Give daemon time to start

        if (check_daemon_running(socket_path)) {
            std::cout << "Daemon started successfully.\n";
        } else {
            std::cerr << "Failed to start daemon.\n";
        }
        return;
    }

    // Child process (daemon)
    setsid();
    chdir("/");
    umask(0);

    // Close standard file descriptors
    close(STDIN_FILENO);
    close(STDOUT_FILENO);
    close(STDERR_FILENO);

    // Start daemon
    auto& daemon = UBODTCacheDaemon::get_instance();
    daemon.start(socket_path);

    exit(0);
}

void stop_daemon(const std::string& socket_path) {
    std::string response = send_command(socket_path, "STOP");
    if (response.find("OK") != std::string::npos) {
        std::cout << "Daemon stopping gracefully...\n";
    } else {
        std::cout << "Daemon status: " << response;
    }
}

int main(int argc, char **argv) {
    // Initialize SPDLOG for console output
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%l][%s:%# ] %v");

    std::string socket_path = "/tmp/fmm_ubodt_daemon.sock";
    std::string ubodt_file;
    size_t max_memory_mb = 0;
    int multiplier = 50000;
    bool show_status = false;
    bool clear_cache = false;
    bool daemon_status = false;
    bool start_daemon_cmd = false;
    bool stop_daemon_cmd = false;
    std::string unload_file;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--load" && i + 1 < argc) {
            ubodt_file = argv[++i];
        } else if (arg == "--max_memory" && i + 1 < argc) {
            max_memory_mb = std::stoul(argv[++i]);
        } else if (arg == "--multiplier" && i + 1 < argc) {
            multiplier = std::stoi(argv[++i]);
        } else if (arg == "--status") {
            show_status = true;
        } else if (arg == "--clear") {
            clear_cache = true;
        } else if (arg == "--unload" && i + 1 < argc) {
            unload_file = argv[++i];
        } else if (arg == "--daemon-status") {
            daemon_status = true;
        } else if (arg == "--start") {
            start_daemon_cmd = true;
        } else if (arg == "--stop") {
            stop_daemon_cmd = true;
        } else if (arg == "--socket" && i + 1 < argc) {
            socket_path = argv[++i];
        } else if (arg == "--help") {
            print_help();
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_help();
            return 1;
        }
    }

    // Handle daemon start/stop
    if (start_daemon_cmd) {
        start_daemon(socket_path);
        return 0;
    }

    if (stop_daemon_cmd) {
        stop_daemon(socket_path);
        return 0;
    }

    // For all other commands, check if daemon is running
    if (!check_daemon_running(socket_path)) {
        std::cerr << "Daemon is not running. Use --start to start it.\n";
        return 1;
    }

    // Handle commands
    if (!ubodt_file.empty()) {
        std::string cmd = "LOAD " + ubodt_file + " " + std::to_string(multiplier);
        if (max_memory_mb > 0) {
            cmd += " " + std::to_string(max_memory_mb);
        }
        std::string response = send_command(socket_path, cmd);
        std::cout << response;
    }

    if (show_status) {
        std::string response = send_command(socket_path, "STATUS");
        std::cout << response;
    }

    if (clear_cache) {
        std::string response = send_command(socket_path, "CLEAR");
        std::cout << response;
    }

    if (!unload_file.empty()) {
        std::string response = send_command(socket_path, "UNLOAD " + unload_file);
        std::cout << response;
    }

    if (daemon_status) {
        std::string response = send_command(socket_path, "DAEMON_STATUS");
        std::cout << response;
    }

    // If no command specified, show status by default
    if (ubodt_file.empty() && !show_status && !clear_cache && unload_file.empty() && !daemon_status) {
        std::string response = send_command(socket_path, "STATUS");
        std::cout << response;
    }

    return 0;
}