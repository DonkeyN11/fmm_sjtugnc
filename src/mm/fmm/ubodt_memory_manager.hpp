/**
 * Fast map matching.
 *
 * UBODT memory manager for caching and reusing UBODT data
 *
 * @author: Generated for FMM optimization
 * @version: 2025.01.01
 */

#ifndef FMM_UBODT_MEMORY_MANAGER_HPP_
#define FMM_UBODT_MEMORY_MANAGER_HPP_

#include "mm/fmm/ubodt.hpp"
#include "network/type.hpp"
#include "util/debug.hpp"
#include <unordered_map>
#include <memory>
#include <mutex>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <filesystem>

namespace FMM {
namespace MM {

/**
 * UBODT memory range information for checking if a point is within cached area
 */
struct UBODTRange {
    double min_x;
    double max_x;
    double min_y;
    double max_y;
    NETWORK::NodeIndex min_node;
    NETWORK::NodeIndex max_node;

    UBODTRange() : min_x(0), max_x(0), min_y(0), max_y(0),
                   min_node(0), max_node(0) {}

    bool contains(NETWORK::NodeIndex node) const {
        return node >= min_node && node <= max_node;
    }

    bool contains_point(double x, double y) const {
        return x >= min_x && x <= max_x && y >= min_y && y <= max_y;
    }

    void expand(const UBODTRange& other) {
        min_x = std::min(min_x, other.min_x);
        max_x = std::max(max_x, other.max_x);
        min_y = std::min(min_y, other.min_y);
        max_y = std::max(max_y, other.max_y);
        min_node = std::min(min_node, other.min_node);
        max_node = std::max(max_node, other.max_node);
    }
};

/**
 * Cached UBODT data with metadata
 */
struct CachedUBODT {
    std::shared_ptr<UBODT> ubodt;
    std::string filename;
    UBODTRange range;
    size_t memory_usage;
    std::chrono::system_clock::time_point last_access;

    CachedUBODT() : memory_usage(0) {}

    void update_access() {
        last_access = std::chrono::system_clock::now();
    }
};

/**
 * Memory manager for UBODT files
 */
class UBODTMemoryManager {
public:
    static UBODTMemoryManager& get_instance() {
        static UBODTMemoryManager instance;
        return instance;
    }

    /**
     * Load UBODT file into memory cache
     * @param filename UBODT file path
     * @param multiplier UBODT multiplier
     * @param max_memory_mb Maximum memory to use (MB), 0 for unlimited
     * @return true if loaded successfully
     */
    bool load_ubodt(const std::string& filename, int multiplier = 50000,
                   size_t max_memory_mb = 0);

    /**
     * Check if UBODT is loaded for given node range
     * @param start_node Start node index
     * @param end_node End node index
     * @return CachedUBODT pointer if available, nullptr otherwise
     */
    std::shared_ptr<CachedUBODT> get_ubodt_for_range(NETWORK::NodeIndex start_node,
                                                     NETWORK::NodeIndex end_node);

    /**
     * Check if UBODT is loaded for given geographic range
     * @param x X coordinate
     * @param y Y coordinate
     * @return CachedUBODT pointer if available, nullptr otherwise
     */
    std::shared_ptr<CachedUBODT> get_ubodt_for_point(double x, double y);

    /**
     * Get cached UBODT by filename
     * @param filename UBODT file path
     * @return CachedUBODT pointer if available, nullptr otherwise
     */
    std::shared_ptr<CachedUBODT> get_ubodt(const std::string& filename);

    /**
     * Remove UBODT from cache
     * @param filename UBODT file path
     * @return true if removed successfully
     */
    bool unload_ubodt(const std::string& filename);

    /**
     * Clear all cached UBODT data
     */
    void clear_cache();

    /**
     * Get memory usage statistics
     * @return Total memory usage in bytes
     */
    size_t get_total_memory_usage() const;

    /**
     * Get number of cached UBODT files
     * @return Number of cached files
     */
    size_t get_cache_size() const;

    /**
     * Set maximum memory limit
     * @param max_memory_mb Maximum memory in MB
     */
    void set_max_memory(size_t max_memory_mb);

    /**
     * Print cache status
     */
    void print_status() const;

    /**
     * Save cache state to persistent storage
     */
    void save_cache_state() const;

    /**
     * Load cache state from persistent storage
     */
    void load_cache_state();

    /**
     * Clean up expired cache files
     */
    void cleanup_expired_cache_files();

private:
    UBODTMemoryManager();
    ~UBODTMemoryManager();
    UBODTMemoryManager(const UBODTMemoryManager&) = delete;
    UBODTMemoryManager& operator=(const UBODTMemoryManager&) = delete;

    std::unordered_map<std::string, std::shared_ptr<CachedUBODT>> cache_;
    mutable std::mutex mutex_;
    size_t max_memory_bytes_ = 0;

    /**
     * Estimate memory usage of UBODT
     * @param ubodt UBODT pointer
     * @return Estimated memory usage in bytes
     */
    size_t estimate_memory_usage(const std::shared_ptr<UBODT>& ubodt) const;

    /**
     * Clean up old entries if memory limit exceeded
     */
    void cleanup_if_needed();

    /**
     * Load range information from UBODT file header
     * @param filename UBODT file path
     * @return UBODTRange information
     */
    UBODTRange load_range_info(const std::string& filename);

    /**
     * Check if memory is sufficient for loading UBODT
     * @param required_memory Required memory in bytes
     * @return true if sufficient memory is available
     */
    bool check_memory_availability(size_t required_memory) const;
};

} // namespace MM
} // namespace FMM

#endif // FMM_UBODT_MEMORY_MANAGER_HPP_