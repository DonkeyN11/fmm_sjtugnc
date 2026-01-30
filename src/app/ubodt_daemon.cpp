/**
 * UBODT Daemon - Persistently keeps UBODT in memory
 *
 * This daemon runs in the background and keeps UBODT loaded,
 * allowing multiple FMM/CMM processes to share the same UBODT.
 *
 * Usage:
 *   ubodt_daemon start --ubodt data/ubodt.bin
 *   ubodt_daemon status
 *   ubodt_daemon stop
 *
 * @version: 2025.01.23
 */

#include "mm/fmm/ubodt_manager.hpp"
#include "util/util.hpp"
#include "util/debug.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <csignal>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace FMM;
using namespace FMM::MM;

// Global flag for signal handling
static std::string PID_FILE = "/tmp/ubodt_daemon.pid";
static std::string STATUS_FILE = "/tmp/ubodt_daemon_status.txt";
static volatile bool keep_running = true;

/**
 * Signal handler for graceful shutdown
 */
void signal_handler(int signal) {
    SPDLOG_INFO("Received signal {}, shutting down...", signal);
    keep_running = false;
}

/**
 * Write daemon status to file
 */
void write_status(const std::string &ubodt_file, bool is_loaded, size_t num_rows) {
    std::ofstream out(STATUS_FILE);
    if (out.is_open()) {
        out << "UBODT_DAEMON_STATUS\n";
        out << "PID: " << getpid() << "\n";
        out << "UBODT_FILE: " << ubodt_file << "\n";
        out << "LOADED: " << (is_loaded ? "yes" : "no") << "\n";
        out << "NUM_ROWS: " << num_rows << "\n";
        out.close();
        SPDLOG_DEBUG("Status written to {}", STATUS_FILE);
    }
}

/**
 * Read daemon status from file
 */
bool read_status(std::string &ubodt_file, bool &is_loaded, size_t &num_rows, pid_t &pid) {
    std::ifstream in(STATUS_FILE);
    if (!in.is_open()) {
        return false;
    }

    std::string line;
    std::string pid_str;
    bool found_daemon = false;

    while (std::getline(in, line)) {
        if (line.find("PID:") == 0) {
            pid_str = line.substr(4);
            // Trim whitespace
            size_t start = pid_str.find_first_not_of(" \t");
            size_t end = pid_str.find_last_not_of(" \t");
            if (start != std::string::npos) {
                pid_str = pid_str.substr(start, end - start + 1);
                pid = std::stoi(pid_str);
            }
        } else if (line.find("UBODT_FILE:") == 0) {
            ubodt_file = line.substr(11);
            size_t start = ubodt_file.find_first_not_of(" \t");
            size_t end = ubodt_file.find_last_not_of(" \t");
            if (start != std::string::npos) {
                ubodt_file = ubodt_file.substr(start, end - start + 1);
            }
        } else if (line.find("LOADED:") == 0) {
            std::string loaded = line.substr(7);
            size_t start = loaded.find_first_not_of(" \t");
            size_t end = loaded.find_last_not_of(" \t");
            if (start != std::string::npos) {
                loaded = loaded.substr(start, end - start + 1);
                is_loaded = (loaded == "yes");
            }
        } else if (line.find("NUM_ROWS:") == 0) {
            std::string rows = line.substr(9);
            size_t start = rows.find_first_not_of(" \t");
            size_t end = rows.find_last_not_of(" \t");
            if (start != std::string::npos) {
                rows = rows.substr(start, end - start + 1);
                num_rows = std::stoull(rows);
            }
        } else if (line.find("UBODT_DAEMON_STATUS") == 0) {
            found_daemon = true;
        }
    }

    in.close();

    // Check if process is still running
    if (found_daemon && pid > 0) {
        if (kill(pid, 0) == 0) {
            // Process is running
            return true;
        } else {
            // Process is not running, clean up stale status file
            SPDLOG_WARN("Daemon PID {} is not running, removing stale status file", pid);
            std::remove(STATUS_FILE.c_str());
            return false;
        }
    }

    return found_daemon;
}

/**
 * Daemon mode: Keep UBODT loaded in memory
 */
int run_daemon(const std::string &ubodt_file, int multiplier) {
    SPDLOG_INFO("Starting UBODT daemon...");
    SPDLOG_INFO("UBODT file: {}", ubodt_file);

    // Setup signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Load UBODT
    auto &manager = UBODTManager::getInstance();
    manager.set_auto_release(false); // Keep UBODT in memory

    SPDLOG_INFO("Loading UBODT...");
    auto start_time = UTIL::get_current_time();
    auto ubodt = manager.get_ubodt(ubodt_file, multiplier, false);
    auto end_time = UTIL::get_current_time();
    double duration = UTIL::get_duration(start_time, end_time);

    if (!ubodt) {
        SPDLOG_ERROR("Failed to load UBODT from {}", ubodt_file);
        return 1;
    }

    size_t num_rows = ubodt->get_num_rows();
    SPDLOG_INFO("UBODT loaded successfully: {} rows in {:.2f}s", num_rows, duration);

    // Write PID file
    std::ofstream pid_file(PID_FILE);
    if (pid_file.is_open()) {
        pid_file << getpid() << "\n";
        pid_file.close();
        SPDLOG_INFO("PID file written: {}", PID_FILE);
    }

    // Write initial status
    write_status(ubodt_file, true, num_rows);

    // Keep running until signal received
    SPDLOG_INFO("Daemon is now running. Press Ctrl+C to stop.");
    std::cout << "\n========================================\n";
    std::cout << "UBODT Daemon Started\n";
    std::cout << "========================================\n";
    std::cout << "PID: " << getpid() << "\n";
    std::cout << "UBODT: " << ubodt_file << "\n";
    std::cout << "Rows: " << num_rows << "\n";
    std::cout << "Load time: " << std::fixed << std::setprecision(2) << duration << "s\n";
    std::cout << "========================================\n\n";

    while (keep_running) {
        // Update status periodically
        write_status(ubodt_file, true, num_rows);
        sleep(5);
    }

    // Cleanup
    SPDLOG_INFO("Shutting down daemon...");
    std::remove(PID_FILE.c_str());
    std::remove(STATUS_FILE.c_str());
    manager.release_all();

    SPDLOG_INFO("Daemon stopped.");
    return 0;
}

/**
 * Show daemon status
 */
int show_status() {
    std::string ubodt_file;
    bool is_loaded = false;
    size_t num_rows = 0;
    pid_t pid = 0;

    if (!read_status(ubodt_file, is_loaded, num_rows, pid)) {
        std::cout << "\nUBODT daemon is not running.\n\n";
        return 0;
    }

    std::cout << "\n========== UBODT Daemon Status ==========\n";
    std::cout << "Status: Running\n";
    std::cout << "PID: " << pid << "\n";
    std::cout << "UBODT file: " << ubodt_file << "\n";
    std::cout << "Loaded: " << (is_loaded ? "Yes" : "No") << "\n";
    if (is_loaded) {
        std::cout << "Rows: " << num_rows << "\n";
    }
    std::cout << "==========================================\n\n";

    return 0;
}

/**
 * Stop daemon
 */
int stop_daemon() {
    std::ifstream pid_file(PID_FILE);
    if (!pid_file.is_open()) {
        std::cout << "\nNo daemon PID file found. Daemon may not be running.\n\n";
        return 1;
    }

    pid_t pid;
    pid_file >> pid;
    pid_file.close();

    if (pid <= 0) {
        std::cout << "\nInvalid PID in PID file.\n\n";
        return 1;
    }

    std::cout << "Stopping daemon (PID: " << pid << ")...\n";

    if (kill(pid, SIGTERM) == 0) {
        std::cout << "Signal sent successfully.\n";

        // Wait for process to terminate
        int count = 0;
        while (count < 10) {
            if (kill(pid, 0) != 0) {
                std::cout << "Daemon stopped successfully.\n\n";
                return 0;
            }
            sleep(1);
            count++;
        }

        std::cout << "Daemon did not stop gracefully. Force killing...\n";
        kill(pid, SIGKILL);
        sleep(1);

        if (kill(pid, 0) != 0) {
            std::cout << "Daemon killed.\n\n";
        } else {
            std::cout << "Failed to kill daemon.\n\n";
            return 1;
        }
    } else {
        std::cout << "Failed to send signal to daemon. Process may not exist.\n\n";
        return 1;
    }

    // Cleanup files
    std::remove(PID_FILE.c_str());
    std::remove(STATUS_FILE.c_str());

    return 0;
}

/**
 * Print help
 */
void print_help() {
    std::cout << "UBODT Daemon - Keep UBODT loaded in background\n\n";
    std::cout << "Usage:\n";
    std::cout << "  ubodt_daemon start --ubodt <file> [options]\n";
    std::cout << "  ubodt_daemon status\n";
    std::cout << "  ubodt_daemon stop\n\n";
    std::cout << "Commands:\n";
    std::cout << "  start   Start daemon and load UBODT\n";
    std::cout << "  status  Show daemon status\n";
    std::cout << "  stop    Stop daemon and release UBODT\n\n";
    std::cout << "Options:\n";
    std::cout << "  --ubodt <file>      Path to UBODT file\n";
    std::cout << "  --multiplier <num>  Hash table multiplier (default: 1)\n";
    std::cout << "  -h, --help         Show this help\n\n";
    std::cout << "Examples:\n";
    std::cout << "  # Start daemon\n";
    std::cout << "  ubodt_daemon start --ubodt data/ubodt.bin\n\n";
    std::cout << "  # Check status\n";
    std::cout << "  ubodt_daemon status\n\n";
    std::cout << "  # Stop daemon\n";
    std::cout << "  ubodt_daemon stop\n\n";
}

/**
 * Main function
 */
int main(int argc, char **argv) {
    // Setup logging
    spdlog::set_pattern("[%^%l%$][%t] %v");
    spdlog::set_level(spdlog::level::info);

    if (argc < 2) {
        print_help();
        return 1;
    }

    std::string command = argv[1];

    if (command == "start") {
        if (argc < 3 || std::string(argv[2]) != "--ubodt") {
            std::cerr << "Error: --ubodt <file> is required for start command\n";
            print_help();
            return 1;
        }

        if (argc < 4) {
            std::cerr << "Error: UBODT file path is required\n";
            print_help();
            return 1;
        }

        std::string ubodt_file = argv[3];
        int multiplier = 1;

        // Parse optional arguments
        for (int i = 4; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--multiplier" && i + 1 < argc) {
                multiplier = std::stoi(argv[++i]);
            } else if (arg == "-h" || arg == "--help") {
                print_help();
                return 0;
            }
        }

        // Check if daemon is already running
        std::string dummy_ubodt;
        bool dummy_loaded;
        size_t dummy_rows;
        pid_t dummy_pid;

        if (read_status(dummy_ubodt, dummy_loaded, dummy_rows, dummy_pid)) {
            std::cout << "\nError: Daemon is already running (PID: " << dummy_pid << ")\n";
            std::cout << "Use 'ubodt_daemon stop' to stop it first.\n\n";
            return 1;
        }

        return run_daemon(ubodt_file, multiplier);

    } else if (command == "status") {
        return show_status();

    } else if (command == "stop") {
        return stop_daemon();

    } else if (command == "-h" || command == "--help") {
        print_help();
        return 0;

    } else {
        std::cerr << "Unknown command: " << command << "\n";
        print_help();
        return 1;
    }

    return 0;
}
