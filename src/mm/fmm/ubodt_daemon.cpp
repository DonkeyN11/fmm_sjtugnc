//
// Implementation of UBODT Cache Daemon
//

#include "mm/fmm/ubodt_daemon.hpp"
#include "util/util.hpp"
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sstream>
#include <algorithm>
#include <mutex>

using namespace FMM;
using namespace FMM::MM;

UBODTCacheDaemon::UBODTCacheDaemon()
    : socket_path_("/tmp/fmm_ubodt_daemon.sock"),
      server_fd_(-1),
      running_(false) {
}

UBODTCacheDaemon::~UBODTCacheDaemon() {
    stop();
}

bool UBODTCacheDaemon::start(const std::string& socket_path) {
    if (running_) {
        SPDLOG_WARN("Daemon is already running");
        return false;
    }

    socket_path_ = socket_path;
    setup_signal_handlers();

    // Create Unix domain socket
    server_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        SPDLOG_ERROR("Failed to create socket: {}", strerror(errno));
        return false;
    }

    // Set socket to non-blocking
    int flags = fcntl(server_fd_, F_GETFL, 0);
    fcntl(server_fd_, F_SETFL, flags | O_NONBLOCK);

    // Remove existing socket file if it exists
    unlink(socket_path_.c_str());

    // Bind socket to path
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

    if (bind(server_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        SPDLOG_ERROR("Failed to bind socket: {}", strerror(errno));
        close(server_fd_);
        return false;
    }

    // Listen for connections
    if (listen(server_fd_, 5) < 0) {
        SPDLOG_ERROR("Failed to listen on socket: {}", strerror(errno));
        close(server_fd_);
        return false;
    }

    // Set socket permissions
    chmod(socket_path_.c_str(), 0666);

    running_ = true;
    daemon_thread_ = std::make_unique<std::thread>(&UBODTCacheDaemon::daemon_loop, this);

    SPDLOG_INFO("UBODT Cache Daemon started on socket: {}", socket_path_);
    return true;
}

void UBODTCacheDaemon::stop() {
    if (!running_) {
        return;
    }

    running_ = false;

    if (daemon_thread_ && daemon_thread_->joinable()) {
        daemon_thread_->join();
    }

    if (server_fd_ >= 0) {
        close(server_fd_);
        server_fd_ = -1;
    }

    unlink(socket_path_.c_str());
    SPDLOG_INFO("UBODT Cache Daemon stopped");
}

bool UBODTCacheDaemon::is_running() const {
    return running_;
}

std::string UBODTCacheDaemon::get_status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::stringstream ss;
    ss << "Daemon Status: " << (running_ ? "Running" : "Stopped") << "\n";
    ss << "Socket: " << socket_path_ << "\n";
    ss << "PID: " << getpid() << "\n";
    return ss.str();
}

void UBODTCacheDaemon::daemon_loop() {
    while (running_) {
        // Accept new connection
        struct sockaddr_un client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);

        if (client_fd < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // No connection available, sleep briefly
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            } else {
                SPDLOG_ERROR("Failed to accept connection: {}", strerror(errno));
                break;
            }
        }

        // Handle client in separate thread
        std::thread([this, client_fd]() {
            handle_client(client_fd);
        }).detach();
    }
}

void UBODTCacheDaemon::handle_client(int client_fd) {
    char buffer[4096];
    ssize_t bytes_read;

    while ((bytes_read = read(client_fd, buffer, sizeof(buffer) - 1)) > 0) {
        buffer[bytes_read] = '\0';
        std::string command(buffer);

        // Process command
        std::string response = process_command(command);

        // Send response
        if (write(client_fd, response.c_str(), response.length()) < 0) {
            SPDLOG_ERROR("Failed to send response: {}", strerror(errno));
            break;
        }
    }

    close(client_fd);
}

std::string UBODTCacheDaemon::process_command(const std::string& command) {
    std::istringstream iss(command);
    std::string cmd;
    iss >> cmd;

    if (cmd == "LOAD") {
        std::string filename;
        int multiplier = 50000;
        size_t max_memory_mb = 0;

        iss >> filename;
        if (iss >> multiplier) {
            int mem;
            if (iss >> mem) {
                max_memory_mb = mem;
            }
        }

        return load_ubodt(filename, multiplier, max_memory_mb);
    }
    else if (cmd == "STATUS") {
        return get_cache_status();
    }
    else if (cmd == "CLEAR") {
        return clear_cache();
    }
    else if (cmd == "UNLOAD") {
        std::string filename;
        iss >> filename;
        return unload_ubodt(filename);
    }
    else if (cmd == "STOP") {
        running_ = false;
        return "OK\n";
    }
    else if (cmd == "DAEMON_STATUS") {
        return get_status();
    }
    else {
        return "ERROR: Unknown command\n";
    }
}

std::string UBODTCacheDaemon::load_ubodt(const std::string& filename, int multiplier, size_t max_memory_mb) {
    auto& manager = UBODTMemoryManager::get_instance();

    try {
        if (manager.load_ubodt(filename, multiplier, max_memory_mb)) {
            return "OK: UBODT loaded successfully\n";
        } else {
            return "ERROR: Failed to load UBODT\n";
        }
    } catch (const std::exception& e) {
        return "ERROR: " + std::string(e.what()) + "\n";
    }
}

std::string UBODTCacheDaemon::get_cache_status() {
    auto& manager = UBODTMemoryManager::get_instance();

    std::stringstream ss;
    ss << "Cache Status:\n";

    size_t total_memory = 0;
    size_t cache_size = 0;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        // This is a simplified version - we would need to add methods to UBODTMemoryManager
        // to get this information without deadlocks
        ss << "Cached files: " << cache_size << "\n";
        ss << "Total memory usage: " << (total_memory / (1024 * 1024)) << " MB\n";
    }

    return ss.str();
}

std::string UBODTCacheDaemon::clear_cache() {
    auto& manager = UBODTMemoryManager::get_instance();
    manager.clear_cache();
    return "OK: Cache cleared\n";
}

std::string UBODTCacheDaemon::unload_ubodt(const std::string& filename) {
    auto& manager = UBODTMemoryManager::get_instance();
    if (manager.unload_ubodt(filename)) {
        return "OK: UBODT unloaded\n";
    } else {
        return "ERROR: UBODT not found\n";
    }
}

void UBODTCacheDaemon::setup_signal_handlers() {
    signal(SIGINT, [](int signum) {
        SPDLOG_INFO("Received signal {}, shutting down daemon...", signum);
        UBODTCacheDaemon::get_instance().stop();
        exit(0);
    });

    signal(SIGTERM, [](int signum) {
        SPDLOG_INFO("Received signal {}, shutting down daemon...", signum);
        UBODTCacheDaemon::get_instance().stop();
        exit(0);
    });
}