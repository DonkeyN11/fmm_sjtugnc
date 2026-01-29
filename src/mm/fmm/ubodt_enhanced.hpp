/**
 * Fast map matching.
 *
 * Enhanced UBODT with caching, batch processing, and incremental loading
 *
 * @version: 2025.01.22
 */

#ifndef FMM_SRC_MM_FMM_UBODT_ENHANCED_HPP_
#define FMM_SRC_MM_FMM_UBODT_ENHANCED_HPP_

#include "mm/fmm/ubodt.hpp"
#include "mm/fmm/ubodt_partial.hpp"
#include "network/network.hpp"
#include "core/gps.hpp"
#include "util/debug.hpp"

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>
#include <list>
#include <mutex>

namespace FMM {
namespace MM {

/**
 * Enhanced UBODT with LRU cache for hot queries
 */
class CachedUBODT {
public:
    /**
     * Cache key for (source, target) pairs
     */
    struct CacheKey {
        NETWORK::NodeIndex source;
        NETWORK::NodeIndex target;

        bool operator==(const CacheKey &other) const {
            return source == other.source && target == other.target;
        }
    };

    /**
     * Cache key hasher
     */
    struct CacheKeyHash {
        std::size_t operator()(const CacheKey &key) const {
            // Combine two hashes using XOR and bit shifting
            std::size_t h1 = std::hash<NETWORK::NodeIndex>{}(key.source);
            std::size_t h2 = std::hash<NETWORK::NodeIndex>{}(key.target);
            return h1 ^ (h2 << 1);
        }
    };

    /**
     * Constructor
     * @param ubodt Base UBODT to wrap
     * @param cache_size Maximum number of cached entries (default: 10000)
     */
    CachedUBODT(std::shared_ptr<UBODT> ubodt, size_t cache_size = 10000);

    /**
     * Look up a record with caching
     * @param source Source node index
     * @param target Target node index
     * @return Pointer to record if found, nullptr otherwise
     */
    const Record *look_up(NETWORK::NodeIndex source, NETWORK::NodeIndex target);

    /**
     * Look up shortest path (not cached, uses underlying UBODT)
     */
    std::vector<NETWORK::EdgeIndex> look_sp_path(NETWORK::NodeIndex source,
                                                  NETWORK::NodeIndex target) const;

    /**
     * Get cache statistics
     */
    struct CacheStats {
        size_t hits;
        size_t misses;
        size_t size;
        double hit_rate() const { return hits + misses > 0 ? (double)hits / (hits + misses) : 0.0; }
    };

    CacheStats get_stats() const;
    void clear_cache();
    void reset_stats();

    /**
     * Get underlying UBODT
     */
    inline std::shared_ptr<UBODT> get_ubodt() const { return ubodt_; }

private:
    std::shared_ptr<UBODT> ubodt_;
    const size_t max_cache_size_;

    // LRU cache implementation
    std::unordered_map<CacheKey, const Record*, CacheKeyHash> cache_;
    std::list<CacheKey> lru_list_;  // Most recently used at front

    // Statistics
    size_t cache_hits_;
    size_t cache_misses_;

    /**
     * Update LRU list - move key to front
     */
    void update_lru(const CacheKey &key);

    /**
     * Evict least recently used entry if cache is full
     */
    void evict_lru();

    /**
     * Insert into cache
     */
    void cache_insert(const CacheKey &key, const Record *record);
};

/**
 * Batch processor for multiple trajectories with shared PartialUBODT
 */
class BatchUBODTProcessor {
public:
    /**
     * Constructor
     * @param ubodt_file Path to UBODT file
     * @param network Road network
     * @param buffer_ratio Buffer ratio for bbox expansion
     */
    BatchUBODTProcessor(const std::string &ubodt_file,
                       const NETWORK::Network &network,
                       double buffer_ratio = 0.1);

    /**
     * Process a batch of trajectories
     * Creates a shared PartialUBODT and applies function to each trajectory
     *
     * @param trajectories Vector of trajectories
     * @param func Function to apply to each trajectory (receives trajectory and PartialUBODT)
     * @return Vector of results
     */
    template<typename ResultFunc, typename ResultType = typename std::result_of<ResultFunc(const CORE::Trajectory&, std::shared_ptr<PartialUBODT>)>::type>
    std::vector<ResultType> process_batch(
        const std::vector<CORE::Trajectory> &trajectories,
        ResultFunc func);

    /**
     * Process trajectories in groups (for very large datasets)
     *
     * @param trajectories All trajectories
     * @param group_size Number of trajectories per group
     * @param func Function to apply to each trajectory
     * @return Vector of results
     */
    template<typename ResultFunc, typename ResultType = typename std::result_of<ResultFunc(const CORE::Trajectory&, std::shared_ptr<PartialUBODT>)>::type>
    std::vector<ResultType> process_groups(
        const std::vector<CORE::Trajectory> &trajectories,
        size_t group_size,
        ResultFunc func);

    /**
     * Get statistics from last batch processing
     */
    struct BatchStats {
        size_t total_trajectories;
        size_t total_groups;
        double total_load_time;
        double avg_load_time_per_group;
        double avg_records_per_group;
    };

    BatchStats get_last_stats() const { return last_stats_; }

private:
    std::string ubodt_file_;
    const NETWORK::Network &network_;
    double buffer_ratio_;
    BatchStats last_stats_;
};

/**
 * Incremental UBODT with dynamic node set expansion
 */
class IncrementalUBODT {
public:
    /**
     * Constructor - starts with empty node set
     * @param ubodt_file Path to UBODT file
     * @param network Road network
     * @param initial_capacity Initial capacity for node set
     */
    IncrementalUBODT(const std::string &ubodt_file,
                     const NETWORK::Network &network,
                     size_t initial_capacity = 10000);

    /**
     * Add nodes to the loaded set
     * @param new_nodes Set of new nodes to load
     * @return Number of new nodes actually loaded
     */
    size_t add_nodes(const std::unordered_set<NETWORK::NodeIndex> &new_nodes);

    /**
     * Add nodes from a bounding box
     * @param bbox Bounding box
     * @param buffer_ratio Buffer ratio
     * @return Number of new nodes loaded
     */
    size_t add_bbox(const boost::geometry::model::box<CORE::Point> &bbox,
                   double buffer_ratio = 0.1);

    /**
     * Add nodes from trajectories
     * @param trajectories Vector of trajectories
     * @param buffer_ratio Buffer ratio
     * @return Number of new nodes loaded
     */
    size_t add_trajectories(const std::vector<CORE::Trajectory> &trajectories,
                           double buffer_ratio = 0.1);

    /**
     * Check if a node is loaded
     */
    bool has_node(NETWORK::NodeIndex node) const;

    /**
     * Look up a record
     */
    const Record *look_up(NETWORK::NodeIndex source, NETWORK::NodeIndex target) const;

    /**
     * Look up shortest path
     */
    std::vector<NETWORK::EdgeIndex> look_sp_path(NETWORK::NodeIndex source,
                                                  NETWORK::NodeIndex target) const;

    /**
     * Get statistics
     */
    inline size_t get_num_loaded_nodes() const { return loaded_nodes_.size(); }
    inline size_t get_num_records() const { return ubodt_ ? ubodt_->get_num_rows() : 0; }
    inline bool is_valid() const { return ubodt_ != nullptr; }
    inline std::shared_ptr<UBODT> get_ubodt() const { return ubodt_; }

    /**
     * Get load statistics
     */
    struct LoadStats {
        size_t total_loads;
        size_t total_nodes_loaded;
        double total_load_time;
    };

    LoadStats get_load_stats() const { return load_stats_; }

private:
    std::string ubodt_file_;
    const NETWORK::Network &network_;
    std::shared_ptr<UBODT> ubodt_;
    std::unordered_set<NETWORK::NodeIndex> loaded_nodes_;
    LoadStats load_stats_;

    /**
     * Load UBODT for current node set
     */
    void reload_ubodt();
};

} // MM
} // FMM

#endif // FMM_SRC_MM_FMM_UBODT_ENHANCED_HPP_
