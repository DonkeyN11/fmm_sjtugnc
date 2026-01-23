/**
 * Fast map matching.
 *
 * UBODT Manager for persistent UBODT storage and reuse
 *
 * This manager allows UBODT to be loaded once and reused across multiple
 * matching operations, significantly reducing load time for batch processing.
 *
 * @version: 2025.01.22
 */

#ifndef FMM_SRC_MM_FMM_UBODT_MANAGER_HPP_
#define FMM_SRC_MM_FMM_UBODT_MANAGER_HPP_

#include "mm/fmm/ubodt.hpp"
#include "mm/fmm/ubodt_partial.hpp"
#include "mm/fmm/ubodt_enhanced.hpp"
#include "network/network.hpp"
#include "util/debug.hpp"

#include <unordered_map>
#include <string>
#include <memory>
#include <mutex>

namespace FMM {
namespace MM {

/**
 * UBODT Manager - Singleton pattern for managing UBODT instances
 *
 * Features:
 * - Load UBODT once and reuse it
 * - Support for full UBODT, PartialUBODT, and CachedUBODT
 * - Automatic reference counting
 * - Thread-safe operations
 * - Manual release control
 */
class UBODTManager {
public:
    /**
     * Get singleton instance
     */
    static UBODTManager& getInstance() {
        static UBODTManager instance;
        return instance;
    }

    // Delete copy and move operations
    UBODTManager(const UBODTManager&) = delete;
    UBODTManager& operator=(const UBODTManager&) = delete;
    UBODTManager(UBODTManager&&) = delete;
    UBODTManager& operator=(UBODTManager&&) = delete;

    /**
     * Load or retrieve a full UBODT
     * @param filename Path to UBODT file
     * @param multiplier Multiplier for hash table (default: 1)
     * @param force_reload Force reload even if already loaded (default: false)
     * @return Shared pointer to UBODT
     */
    std::shared_ptr<UBODT> get_ubodt(const std::string &filename,
                                      int multiplier = 1,
                                      bool force_reload = false);

    /**
     * Load or retrieve a PartialUBODT from node set
     * @param filename Path to UBODT file
     * @param network Road network
     * @param nodes Required node set
     * @param force_reload Force reload even if already loaded
     * @return Shared pointer to PartialUBODT
     */
    std::shared_ptr<PartialUBODT> get_partial_ubodt(
        const std::string &filename,
        const NETWORK::Network &network,
        const std::unordered_set<NETWORK::NodeIndex> &nodes,
        bool force_reload = false);

    /**
     * Load or retrieve a PartialUBODT from trajectories
     * @param filename Path to UBODT file
     * @param network Road network
     * @param trajectories Vector of trajectories
     * @param buffer_ratio Buffer ratio for bbox expansion
     * @param force_reload Force reload even if already loaded
     * @return Shared pointer to PartialUBODT
     */
    std::shared_ptr<PartialUBODT> get_partial_ubodt(
        const std::string &filename,
        const NETWORK::Network &network,
        const std::vector<CORE::Trajectory> &trajectories,
        double buffer_ratio = 0.1,
        bool force_reload = false);

    /**
     * Load or retrieve a CachedUBODT
     * @param filename Path to UBODT file
     * @param cache_size Cache size for LRU (default: 10000)
     * @param multiplier Multiplier for hash table (default: 1)
     * @param force_reload Force reload even if already loaded
     * @return Shared pointer to CachedUBODT
     */
    std::shared_ptr<CachedUBODT> get_cached_ubodt(const std::string &filename,
                                                   size_t cache_size = 10000,
                                                   int multiplier = 1,
                                                   bool force_reload = false);

    /**
     * Check if a UBODT is loaded
     * @param filename Path to UBODT file
     * @return True if loaded
     */
    bool is_loaded(const std::string &filename) const;

    /**
     * Release a specific UBODT
     * @param filename Path to UBODT file
     * @return Number of UBODTs released
     */
    size_t release_ubodt(const std::string &filename);

    /**
     * Release all UBODTs
     * @return Number of UBODTs released
     */
    size_t release_all();

    /**
     * Get statistics
     */
    struct ManagerStats {
        size_t total_ubodts;        // Total number of UBODTs loaded
        size_t total_references;    // Total reference count
        size_t memory_estimated;    // Estimated memory usage (bytes)
    };

    ManagerStats get_stats() const;

    /**
     * Print status to log
     */
    void print_status() const;

    /**
     * Enable/disable auto-release (default: false)
     * When enabled, UBODTs are automatically released when no longer referenced
     */
    void set_auto_release(bool enable) { auto_release_ = enable; }
    bool get_auto_release() const { return auto_release_; }

    /**
     * Check if a UBODT file is loaded by the ubodt_daemon
     * @param filename Path to UBODT file
     * @return true if loaded by daemon
     */
    static bool check_daemon_loaded(const std::string &filename);

private:
    UBODTManager() = default;
    ~UBODTManager() = default;

    // Mutually exclusive storage for different UBODT types
    std::unordered_map<std::string, std::shared_ptr<UBODT>> ubodt_map_;
    std::unordered_map<std::string, std::shared_ptr<PartialUBODT>> partial_map_;
    std::unordered_map<std::string, std::shared_ptr<CachedUBODT>> cached_map_;

    // Mutex for thread safety
    mutable std::mutex mutex_;

    // Configuration
    bool auto_release_ = false;

    /**
     * Generate a key for caching
     */
    std::string make_key(const std::string &filename, const std::string &type) const {
        return type + ":" + filename;
    }

    std::string make_key(const std::string &filename, const std::string &type,
                        const std::string &params) const {
        return type + ":" + filename + ":" + params;
    }
};

/**
 * Convenience functions for using UBODTManager
 */
namespace UBODTHelper {
    // Load full UBODT
    inline std::shared_ptr<UBODT> load_ubodt(const std::string &filename,
                                              int multiplier = 1,
                                              bool keep = true) {
        auto &manager = UBODTManager::getInstance();
        manager.set_auto_release(!keep);
        return manager.get_ubodt(filename, multiplier);
    }

    // Load PartialUBODT from trajectories
    inline std::shared_ptr<PartialUBODT> load_partial_ubodt(
        const std::string &filename,
        const NETWORK::Network &network,
        const std::vector<CORE::Trajectory> &trajectories,
        double buffer_ratio = 0.1,
        bool keep = true) {
        auto &manager = UBODTManager::getInstance();
        manager.set_auto_release(!keep);
        return manager.get_partial_ubodt(filename, network, trajectories, buffer_ratio);
    }

    // Load CachedUBODT
    inline std::shared_ptr<CachedUBODT> load_cached_ubodt(const std::string &filename,
                                                           size_t cache_size = 10000,
                                                           int multiplier = 1,
                                                           bool keep = true) {
        auto &manager = UBODTManager::getInstance();
        manager.set_auto_release(!keep);
        return manager.get_cached_ubodt(filename, cache_size, multiplier);
    }

    // Release specific UBODT
    inline size_t release_ubodt(const std::string &filename) {
        return UBODTManager::getInstance().release_ubodt(filename);
    }

    // Release all UBODTs
    inline size_t release_all_ubodts() {
        return UBODTManager::getInstance().release_all();
    }

    // Check if loaded
    inline bool is_ubodt_loaded(const std::string &filename) {
        return UBODTManager::getInstance().is_loaded(filename);
    }

    // Print status
    inline void print_ubodt_status() {
        UBODTManager::getInstance().print_status();
    }
} // namespace UBODTHelper

} // MM
} // FMM

#endif // FMM_SRC_MM_FMM_UBODT_MANAGER_HPP_
