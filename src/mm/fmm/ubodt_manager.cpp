//
// UBODT Manager implementation
//

#include "mm/fmm/ubodt_manager.hpp"
#include "util/util.hpp"

#include <chrono>
#include <fstream>
#include <csignal>
#include <limits.h>  // For PATH_MAX
#include <cstring>   // For string operations

using namespace FMM;
using namespace FMM::CORE;
using namespace FMM::NETWORK;
using namespace FMM::MM;

// ============================================================================
// UBODTManager Implementation
// ============================================================================

bool UBODTManager::check_daemon_loaded(const std::string &filename) {
    std::string status_file = "/tmp/ubodt_daemon_status.txt";
    std::ifstream in(status_file);
    if (!in.is_open()) {
        return false;
    }

    std::string line;
    std::string daemon_file;
    bool is_loaded = false;
    pid_t pid = 0;
    bool found_daemon = false;

    while (std::getline(in, line)) {
        if (line.find("PID:") == 0) {
            try {
                pid = std::stoi(line.substr(4));
            } catch (...) { pid = 0; }
        } else if (line.find("UBODT_FILE:") == 0) {
            daemon_file = line.substr(11);
            // Trim whitespace
            size_t start = daemon_file.find_first_not_of(" \t");
            size_t end = daemon_file.find_last_not_of(" \t");
            if (start != std::string::npos) {
                daemon_file = daemon_file.substr(start, end - start + 1);
            }
        } else if (line.find("LOADED:") == 0) {
            if (line.find("yes") != std::string::npos) {
                is_loaded = true;
            }
        } else if (line.find("UBODT_DAEMON_STATUS") == 0) {
            found_daemon = true;
        }
    }
    in.close();

    // Check if process is running
    if (found_daemon && pid > 0 && kill(pid, 0) == 0) {
        // Check if filenames match (simple string match)
        // Normalize paths could be better but this is a basic check
        if (is_loaded && daemon_file == filename) {
            return true;
        }
        // Also try to check if one is suffix of another to be more robust
        if (is_loaded && (daemon_file.find(filename) != std::string::npos || 
                          filename.find(daemon_file) != std::string::npos)) {
            return true;
        }
    }

    return false;
}

std::shared_ptr<UBODT> UBODTManager::get_ubodt(const std::string &filename,
                                                int multiplier,
                                                bool force_reload) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string key = make_key(filename, "full");

    // Check if already loaded
    if (!force_reload) {
        auto it = ubodt_map_.find(key);
        if (it != ubodt_map_.end()) {
            SPDLOG_DEBUG("Using cached full UBODT from {}", filename);
            return it->second;
        }
    }

    // Load new UBODT
    SPDLOG_INFO("Loading full UBODT from {}", filename);
    auto start_time = UTIL::get_current_time();

    auto ubodt = UBODT::read_ubodt_file(filename, multiplier);

    auto end_time = UTIL::get_current_time();
    double duration = UTIL::get_duration(start_time, end_time);

    if (ubodt) {
        ubodt_map_[key] = ubodt;
        SPDLOG_INFO("Full UBODT loaded: {} rows in {:.2f}s",
                    ubodt->get_num_rows(), duration);
    }

    return ubodt;
}

std::shared_ptr<PartialUBODT> UBODTManager::get_partial_ubodt(
    const std::string &filename,
    const Network &network,
    const std::unordered_set<NodeIndex> &nodes,
    bool force_reload) {

    std::lock_guard<std::mutex> lock(mutex_);

    // Create a unique key based on node set
    std::string params = "nodes_" + std::to_string(nodes.size());
    std::string key = make_key(filename, "partial", params);

    // Check if already loaded
    if (!force_reload) {
        auto it = partial_map_.find(key);
        if (it != partial_map_.end()) {
            SPDLOG_DEBUG("Using cached PartialUBODT from {} ({} nodes)",
                        filename, nodes.size());
            return it->second;
        }
    }

    // Load new PartialUBODT
    SPDLOG_INFO("Loading PartialUBODT from {} ({} nodes)", filename, nodes.size());
    auto start_time = UTIL::get_current_time();

    auto partial_ubodt = std::make_shared<PartialUBODT>(filename, network, nodes);

    auto end_time = UTIL::get_current_time();
    double duration = UTIL::get_duration(start_time, end_time);

    if (partial_ubodt && partial_ubodt->is_valid()) {
        partial_map_[key] = partial_ubodt;
        SPDLOG_INFO("PartialUBODT loaded: {} records from {} sources in {:.2f}s",
                    partial_ubodt->get_num_records(),
                    partial_ubodt->get_num_sources(),
                    duration);
    }

    return partial_ubodt;
}

std::shared_ptr<PartialUBODT> UBODTManager::get_partial_ubodt(
    const std::string &filename,
    const Network &network,
    const std::vector<Trajectory> &trajectories,
    double buffer_ratio,
    bool force_reload) {

    std::lock_guard<std::mutex> lock(mutex_);

    // Create a unique key based on trajectories
    std::string params = "traj_" + std::to_string(trajectories.size()) +
                        "_buf_" + std::to_string(buffer_ratio);
    std::string key = make_key(filename, "partial", params);

    // Check if already loaded
    if (!force_reload) {
        auto it = partial_map_.find(key);
        if (it != partial_map_.end()) {
            SPDLOG_DEBUG("Using cached PartialUBODT from {} ({} trajectories)",
                        filename, trajectories.size());
            return it->second;
        }
    }

    // Load new PartialUBODT
    SPDLOG_INFO("Loading PartialUBODT from {} ({} trajectories)",
                filename, trajectories.size());
    auto start_time = UTIL::get_current_time();

    auto partial_ubodt = std::make_shared<PartialUBODT>(
        filename, network, trajectories, buffer_ratio);

    auto end_time = UTIL::get_current_time();
    double duration = UTIL::get_duration(start_time, end_time);

    if (partial_ubodt && partial_ubodt->is_valid()) {
        partial_map_[key] = partial_ubodt;
        SPDLOG_INFO("PartialUBODT loaded: {} records from {} sources in {:.2f}s",
                    partial_ubodt->get_num_records(),
                    partial_ubodt->get_num_sources(),
                    duration);
    }

    return partial_ubodt;
}

std::shared_ptr<CachedUBODT> UBODTManager::get_cached_ubodt(
    const std::string &filename,
    size_t cache_size,
    int multiplier,
    bool force_reload) {

    std::lock_guard<std::mutex> lock(mutex_);

    std::string params = "cache_" + std::to_string(cache_size);
    std::string key = make_key(filename, "cached", params);

    // Check if already loaded
    if (!force_reload) {
        auto it = cached_map_.find(key);
        if (it != cached_map_.end()) {
            SPDLOG_DEBUG("Using cached CachedUBODT from {} (cache size: {})",
                        filename, cache_size);
            return it->second;
        }
    }

    // Load full UBODT first
    auto ubodt = get_ubodt(filename, multiplier, force_reload);
    if (!ubodt) {
        return nullptr;
    }

    // Wrap with cache
    SPDLOG_INFO("Creating CachedUBODT with cache size {}", cache_size);
    auto cached_ubodt = std::make_shared<CachedUBODT>(ubodt, cache_size);

    cached_map_[key] = cached_ubodt;

    return cached_ubodt;
}

bool UBODTManager::is_loaded(const std::string &filename) const {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check all maps
    for (const auto &kv : ubodt_map_) {
        if (kv.first.find(filename) != std::string::npos) {
            return true;
        }
    }
    for (const auto &kv : partial_map_) {
        if (kv.first.find(filename) != std::string::npos) {
            return true;
        }
    }
    for (const auto &kv : cached_map_) {
        if (kv.first.find(filename) != std::string::npos) {
            return true;
        }
    }

    return false;
}

size_t UBODTManager::release_ubodt(const std::string &filename) {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t released = 0;

    // Release from all maps
    for (auto it = ubodt_map_.begin(); it != ubodt_map_.end();) {
        if (it->first.find(filename) != std::string::npos) {
            SPDLOG_INFO("Releasing full UBODT: {}", filename);
            it = ubodt_map_.erase(it);
            ++released;
        } else {
            ++it;
        }
    }

    for (auto it = partial_map_.begin(); it != partial_map_.end();) {
        if (it->first.find(filename) != std::string::npos) {
            SPDLOG_INFO("Releasing PartialUBODT: {}", filename);
            it = partial_map_.erase(it);
            ++released;
        } else {
            ++it;
        }
    }

    for (auto it = cached_map_.begin(); it != cached_map_.end();) {
        if (it->first.find(filename) != std::string::npos) {
            SPDLOG_INFO("Releasing CachedUBODT: {}", filename);
            it = cached_map_.erase(it);
            ++released;
        } else {
            ++it;
        }
    }

    if (released > 0) {
        SPDLOG_INFO("Released {} UBODT(s) for {}", released, filename);
    }

    return released;
}

size_t UBODTManager::release_all() {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t total = ubodt_map_.size() + partial_map_.size() + cached_map_.size();

    SPDLOG_INFO("Releasing all UBODTs: {} total", total);

    ubodt_map_.clear();
    partial_map_.clear();
    cached_map_.clear();

    return total;
}

UBODTManager::ManagerStats UBODTManager::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    ManagerStats stats;
    stats.total_ubodts = ubodt_map_.size() + partial_map_.size() + cached_map_.size();
    stats.total_references = stats.total_ubodts; // Simplified

    // Rough memory estimation
    size_t memory = 0;
    for (const auto &kv : ubodt_map_) {
        if (kv.second) {
            memory += kv.second->get_num_rows() * sizeof(Record);
        }
    }
    stats.memory_estimated = memory;

    return stats;
}

void UBODTManager::print_status() const {
    std::lock_guard<std::mutex> lock(mutex_);

    SPDLOG_INFO("========== UBODT Manager Status ==========");
    SPDLOG_INFO("Full UBODTs: {}", ubodt_map_.size());
    SPDLOG_INFO("PartialUBODTs: {}", partial_map_.size());
    SPDLOG_INFO("CachedUBODTs: {}", cached_map_.size());
    SPDLOG_INFO("Total loaded: {}", ubodt_map_.size() + partial_map_.size() + cached_map_.size());
    SPDLOG_INFO("Auto-release: {}", auto_release_ ? "enabled" : "disabled");

    // Print details for each loaded UBODT
    if (!ubodt_map_.empty()) {
        SPDLOG_INFO("--- Full UBODTs ---");
        for (const auto &kv : ubodt_map_) {
            if (kv.second) {
                SPDLOG_INFO("  {} -> {} rows", kv.first, kv.second->get_num_rows());
            }
        }
    }

    if (!partial_map_.empty()) {
        SPDLOG_INFO("--- PartialUBODTs ---");
        for (const auto &kv : partial_map_) {
            if (kv.second) {
                SPDLOG_INFO("  {} -> {} records, {} sources",
                           kv.first,
                           kv.second->get_num_records(),
                           kv.second->get_num_sources());
            }
        }
    }

    if (!cached_map_.empty()) {
        SPDLOG_INFO("--- CachedUBODTs ---");
        for (const auto &kv : cached_map_) {
            if (kv.second) {
                auto stats = kv.second->get_stats();
                SPDLOG_INFO("  {} -> cache hit rate: {:.2f}%",
                           kv.first, stats.hit_rate() * 100);
            }
        }
    }

    SPDLOG_INFO("==========================================");
}
