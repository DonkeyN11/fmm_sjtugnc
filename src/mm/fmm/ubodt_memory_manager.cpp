//
// Implementation of UBODT memory manager
//

#include "mm/fmm/ubodt_memory_manager.hpp"
#include "util/util.hpp"
#include <sys/resource.h>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>

using namespace FMM;
using namespace FMM::MM;
using namespace FMM::NETWORK;

namespace fs = std::filesystem;

bool UBODTMemoryManager::load_ubodt(const std::string& filename, int multiplier,
                                   size_t max_memory_mb) {
    std::lock_guard<std::mutex> lock(mutex_);

    SPDLOG_INFO("Loading UBODT file: {}", filename);

    // Check if already loaded
    if (cache_.find(filename) != cache_.end()) {
        SPDLOG_INFO("UBODT {} already loaded", filename);
        return true;
    }

    // Set memory limit if provided
    if (max_memory_mb > 0) {
        max_memory_bytes_ = max_memory_mb * 1024 * 1024;
    }

    try {
        // Check if file exists first
        std::ifstream test_file(filename);
        if (!test_file.good()) {
            SPDLOG_ERROR("UBODT file does not exist or cannot be read: {}", filename);
            return false;
        }
        test_file.close();

        // Estimate memory requirements before loading
        long estimated_rows = UBODT::estimate_ubodt_rows(filename);
        if (estimated_rows > 0) {
            size_t estimated_memory = estimated_rows * sizeof(Record) * 2; // Rough estimate

            // Check memory availability
            if (!check_memory_availability(estimated_memory)) {
                SPDLOG_ERROR("Insufficient memory to load UBODT file: {}", filename);
                return false;
            }
        }

        // Load UBODT
        auto start_time = UTIL::get_current_time();
        auto ubodt = UBODT::read_ubodt_file(filename, multiplier);
        auto end_time = UTIL::get_current_time();

        if (!ubodt) {
            SPDLOG_ERROR("Failed to load UBODT file: {}", filename);
            return false;
        }

        // Create cached entry
        auto cached_ubodt = std::make_shared<CachedUBODT>();
        cached_ubodt->ubodt = ubodt;
        cached_ubodt->filename = filename;
        cached_ubodt->range = load_range_info(filename);
        cached_ubodt->memory_usage = estimate_memory_usage(ubodt);
        cached_ubodt->update_access();

        SPDLOG_INFO("UBODT loaded successfully in {} seconds, estimated memory usage: {} MB",
                   UTIL::get_duration(start_time, end_time),
                   cached_ubodt->memory_usage / (1024 * 1024));

        // Add to cache
        cache_[filename] = cached_ubodt;

        // Check if we need to clean up
        cleanup_if_needed();

        return true;

    } catch (const std::exception& e) {
        SPDLOG_ERROR("Exception while loading UBODT: {}", e.what());
        return false;
    }
}

std::shared_ptr<CachedUBODT> UBODTMemoryManager::get_ubodt_for_range(
    NodeIndex start_node, NodeIndex end_node) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& pair : cache_) {
        auto& cached = pair.second;
        if (cached->range.contains(start_node) && cached->range.contains(end_node)) {
            cached->update_access();
            return cached;
        }
    }

    return nullptr;
}

std::shared_ptr<CachedUBODT> UBODTMemoryManager::get_ubodt_for_point(double x, double y) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& pair : cache_) {
        auto& cached = pair.second;
        if (cached->range.contains_point(x, y)) {
            cached->update_access();
            return cached;
        }
    }

    return nullptr;
}

std::shared_ptr<CachedUBODT> UBODTMemoryManager::get_ubodt(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = cache_.find(filename);
    if (it != cache_.end()) {
        it->second->update_access();
        return it->second;
    }

    return nullptr;
}

bool UBODTMemoryManager::unload_ubodt(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = cache_.find(filename);
    if (it != cache_.end()) {
        size_t memory_freed = it->second->memory_usage;
        cache_.erase(it);
        SPDLOG_INFO("Unloaded UBODT {}, freed {} MB", filename,
                   memory_freed / (1024 * 1024));
        return true;
    }

    return false;
}

void UBODTMemoryManager::clear_cache() {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t total_memory = 0;
    for (const auto& pair : cache_) {
        total_memory += pair.second->memory_usage;
    }

    cache_.clear();
    SPDLOG_INFO("Cleared all cached UBODT data, freed {} MB",
               total_memory / (1024 * 1024));
}

size_t UBODTMemoryManager::get_total_memory_usage() const {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t total = 0;
    for (const auto& pair : cache_) {
        total += pair.second->memory_usage;
    }
    return total;
}

size_t UBODTMemoryManager::get_cache_size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.size();
}

void UBODTMemoryManager::set_max_memory(size_t max_memory_mb) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_memory_bytes_ = max_memory_mb * 1024 * 1024;
    cleanup_if_needed();
}

void UBODTMemoryManager::print_status() const {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t total_memory = 0;
    for (const auto& pair : cache_) {
        total_memory += pair.second->memory_usage;
    }

    std::cout << "UBODT Memory Manager Status:\n";
    std::cout << "  Cached files: " << cache_.size() << "\n";
    std::cout << "  Total memory usage: " << (total_memory / (1024 * 1024)) << " MB\n";
    std::cout << "  Memory limit: " << (max_memory_bytes_ / (1024 * 1024)) << " MB\n";

    for (const auto& pair : cache_) {
        const auto& cached = pair.second;
        std::cout << "  File: " << cached->filename << "\n";
        std::cout << "    Memory: " << (cached->memory_usage / (1024 * 1024)) << " MB\n";
        std::cout << "    Node range: " << cached->range.min_node << " - " << cached->range.max_node << "\n";
        std::cout << "    Geo range: [" << cached->range.min_x << ", " << cached->range.min_y
                  << "] to [" << cached->range.max_x << ", " << cached->range.max_y << "]\n";
    }
}

size_t UBODTMemoryManager::estimate_memory_usage(const std::shared_ptr<UBODT>& ubodt) const {
    if (!ubodt) return 0;

    // Estimate based on:
    // 1. Number of rows in UBODT
    // 2. Size of each record (28 bytes)
    // 3. Hash table overhead
    long long num_rows = ubodt->get_num_rows();
    size_t record_size = sizeof(Record);
    size_t hash_table_overhead = num_rows * 2; // Rough estimate for hash table overhead

    return num_rows * record_size + hash_table_overhead;
}

void UBODTMemoryManager::cleanup_if_needed() {
    if (max_memory_bytes_ == 0) return;

    size_t current_usage = get_total_memory_usage();
    if (current_usage <= max_memory_bytes_) return;

    SPDLOG_INFO("Memory limit exceeded ({} MB > {} MB), cleaning up cache...",
               current_usage / (1024 * 1024), max_memory_bytes_ / (1024 * 1024));

    // Sort by last access time (LRU)
    std::vector<std::shared_ptr<CachedUBODT>> entries;
    for (auto& pair : cache_) {
        entries.push_back(pair.second);
    }

    std::sort(entries.begin(), entries.end(),
              [](const std::shared_ptr<CachedUBODT>& a,
                 const std::shared_ptr<CachedUBODT>& b) {
                  return a->last_access < b->last_access;
              });

    // Remove oldest entries until under limit
    for (auto& entry : entries) {
        if (current_usage <= max_memory_bytes_) break;

        cache_.erase(entry->filename);
        current_usage -= entry->memory_usage;
        SPDLOG_INFO("Unloaded UBODT {} (LRU), freed {} MB",
                   entry->filename, entry->memory_usage / (1024 * 1024));
    }
}

UBODTRange UBODTMemoryManager::load_range_info(const std::string& filename) {
    UBODTRange range;

    try {
        // For now, estimate range from file size and structure
        // In a real implementation, you might want to scan the first few records
        // or store range information in a separate metadata file

        struct stat stat_buf;
        if (stat(filename.c_str(), &stat_buf) == 0) {
            // Estimate number of rows
            long estimated_rows = UBODT::estimate_ubodt_rows(filename);

            // For demonstration, set a conservative range
            // In practice, you'd want to analyze the actual data
            range.min_node = 0;
            range.max_node = estimated_rows > 0 ? estimated_rows / 10 : 1000;

            // Set geographic range to a large area by default
            range.min_x = -180.0;
            range.max_x = 180.0;
            range.min_y = -90.0;
            range.max_y = 90.0;
        }

    } catch (const std::exception& e) {
        SPDLOG_WARN("Cannot load range info for {}: {}", filename, e.what());
        // Set default large range
        range.min_x = -180.0;
        range.max_x = 180.0;
        range.min_y = -90.0;
        range.max_y = 90.0;
        range.min_node = 0;
        range.max_node = 1000000;
    }

    return range;
}

bool UBODTMemoryManager::check_memory_availability(size_t required_memory) const {
    // Get current memory usage
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    size_t current_memory = usage.ru_maxrss * 1024; // Convert to bytes

    // Get available memory
    long available_memory = sysconf(_SC_AVPHYS_PAGES) * sysconf(_SC_PAGE_SIZE);
    if (available_memory < 0) {
        SPDLOG_WARN("Cannot determine available memory, assuming sufficient");
        return true;
    }

    // Check system memory limits
    if (required_memory > (size_t)available_memory) {
        SPDLOG_ERROR("Required memory ({}) exceeds available system memory ({})",
                    required_memory, available_memory);
        return false;
    }

    // Check process memory limits
    struct rlimit rlim;
    if (getrlimit(RLIMIT_AS, &rlim) == 0) {
        if (rlim.rlim_cur != RLIM_INFINITY) {
            if (current_memory + required_memory > rlim.rlim_cur) {
                SPDLOG_ERROR("Memory limit would be exceeded: current {}, required {}, limit {}",
                            current_memory, required_memory, rlim.rlim_cur);
                return false;
            }
        }
    }

    // Check user-specified memory limit
    if (max_memory_bytes_ > 0) {
        if (current_memory + required_memory > max_memory_bytes_) {
            SPDLOG_WARN("Memory usage would exceed user limit: current {}, required {}, limit {}",
                       current_memory, required_memory, max_memory_bytes_);
            return false;
        }
    }

    // Safety margin: leave at least 10% of available memory free
    size_t safety_margin = available_memory / 10;
    if (current_memory + required_memory + safety_margin > (size_t)available_memory) {
        SPDLOG_WARN("Memory usage would leave insufficient safety margin");
        return false;
    }

    return true;
}

UBODTMemoryManager::UBODTMemoryManager() {
    SPDLOG_INFO("Initializing UBODT Memory Manager");
    cleanup_expired_cache_files();
    load_cache_state();
}

UBODTMemoryManager::~UBODTMemoryManager() {
    SPDLOG_INFO("Cleaning up UBODT Memory Manager");
    save_cache_state();
}

void UBODTMemoryManager::save_cache_state() const {
    std::lock_guard<std::mutex> lock(mutex_);

    try {
        // Create cache directory if it doesn't exist
        fs::path cache_dir = fs::temp_directory_path() / "fmm_ubodt_cache";
        fs::create_directories(cache_dir);

        // Save cache state
        fs::path state_file = cache_dir / "cache_state.txt";
        std::ofstream out(state_file);

        if (out.is_open()) {
            out << "# FMM UBODT Cache State\n";
            out << "# Format: filename|memory_usage|last_access_time\n";

            auto now = std::chrono::system_clock::now();
            for (const auto& pair : cache_) {
                const auto& cached = pair.second;
                auto access_time = std::chrono::duration_cast<std::chrono::seconds>(
                    cached->last_access.time_since_epoch()).count();
                out << cached->filename << "|" << cached->memory_usage << "|" << access_time << "\n";
            }
            out.close();
            SPDLOG_DEBUG("Cache state saved to {}", state_file.string());
        }
    } catch (const std::exception& e) {
        SPDLOG_WARN("Failed to save cache state: {}", e.what());
    }
}

void UBODTMemoryManager::load_cache_state() {
    std::lock_guard<std::mutex> lock(mutex_);

    try {
        fs::path cache_dir = fs::temp_directory_path() / "fmm_ubodt_cache";
        fs::path state_file = cache_dir / "cache_state.txt";

        if (fs::exists(state_file)) {
            std::ifstream in(state_file);
            std::string line;

            while (std::getline(in, line)) {
                if (line.empty() || line[0] == '#') continue;

                size_t pos1 = line.find('|');
                size_t pos2 = line.rfind('|');
                if (pos1 != std::string::npos && pos2 != std::string::npos && pos1 != pos2) {
                    std::string filename = line.substr(0, pos1);
                    // We don't actually reload the UBODT data here, just acknowledge the cache state
                    // The actual UBODT will be loaded on demand when requested
                    SPDLOG_DEBUG("Found cached UBODT file: {}", filename);
                }
            }
            in.close();
        }
    } catch (const std::exception& e) {
        SPDLOG_WARN("Failed to load cache state: {}", e.what());
    }
}

void UBODTMemoryManager::cleanup_expired_cache_files() {
    try {
        fs::path cache_dir = fs::temp_directory_path() / "fmm_ubodt_cache";

        if (fs::exists(cache_dir)) {
            auto now = std::chrono::system_clock::now();
            auto max_age = std::chrono::hours(24); // Clean files older than 24 hours

            for (const auto& entry : fs::directory_iterator(cache_dir)) {
                if (entry.is_regular_file()) {
                    auto ftime = fs::last_write_time(entry);
                    // Convert file time to system clock time (C++17 compatible)
                    auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                        ftime - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
                    auto file_age = now - sctp;

                    if (file_age > max_age) {
                        fs::remove(entry.path());
                        SPDLOG_DEBUG("Removed expired cache file: {}", entry.path().string());
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        SPDLOG_WARN("Failed to cleanup expired cache files: {}", e.what());
    }
}