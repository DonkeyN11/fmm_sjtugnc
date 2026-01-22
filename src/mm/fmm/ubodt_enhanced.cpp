//
// Enhanced UBODT with caching, batch processing, and incremental loading
//

#include "mm/fmm/ubodt_enhanced.hpp"
#include "util/util.hpp"

#include <algorithm>
#include <chrono>

using namespace FMM;
using namespace FMM::CORE;
using namespace FMM::NETWORK;
using namespace FMM::MM;

// ============================================================================
// CachedUBODT Implementation
// ============================================================================

CachedUBODT::CachedUBODT(std::shared_ptr<UBODT> ubodt, size_t cache_size)
    : ubodt_(ubodt)
    , max_cache_size_(cache_size)
    , cache_hits_(0)
    , cache_misses_(0) {

    if (!ubodt_) {
        SPDLOG_CRITICAL("Cannot create CachedUBODT with null UBODT");
        return;
    }

    SPDLOG_INFO("Created CachedUBODT with cache size {}", cache_size);

    // Reserve space for cache
    cache_.reserve(cache_size);
    lru_list_.reserve(cache_size);
}

const Record *CachedUBODT::look_up(NodeIndex source, NodeIndex target) {
    CacheKey key{source, target};

    // Check cache
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        // Cache hit
        ++cache_hits_;
        update_lru(key);
        return it->second;
    }

    // Cache miss - query underlying UBODT
    ++cache_misses_;
    const Record *record = ubodt_->look_up(source, target);

    // Insert into cache if found
    if (record) {
        cache_insert(key, record);
    }

    return record;
}

std::vector<EdgeIndex> CachedUBODT::look_sp_path(NodeIndex source, NodeIndex target) const {
    // Path reconstruction is not cached (could be added if needed)
    return ubodt_->look_sp_path(source, target);
}

CachedUBODT::CacheStats CachedUBODT::get_stats() const {
    return CacheStats{cache_hits_, cache_misses_, cache_.size()};
}

void CachedUBODT::clear_cache() {
    cache_.clear();
    lru_list_.clear();
    SPDLOG_DEBUG("Cleared UBODT query cache");
}

void CachedUBODT::reset_stats() {
    cache_hits_ = 0;
    cache_misses_ = 0;
    SPDLOG_DEBUG("Reset cache statistics");
}

void CachedUBODT::update_lru(const CacheKey &key) {
    // Move key to front of LRU list
    auto it = std::find(lru_list_.begin(), lru_list_.end(), key);
    if (it != lru_list_.end()) {
        lru_list_.erase(it);
    }
    lru_list_.push_front(key);
}

void CachedUBODT::evict_lru() {
    if (lru_list_.empty()) return;

    // Remove least recently used (last element)
    CacheKey lru_key = lru_list_.back();
    lru_list_.pop_back();
    cache_.erase(lru_key);

    SPDLOG_TRACE("Evicted LRU entry from cache (size: {})", cache_.size());
}

void CachedUBODT::cache_insert(const CacheKey &key, const Record *record) {
    // Check if cache is full
    if (cache_.size() >= max_cache_size_) {
        evict_lru();
    }

    // Insert into cache and LRU list
    cache_[key] = record;
    lru_list_.push_front(key);

    SPDLOG_TRACE("Cached entry ({}, {}) - cache size: {}",
                 key.source, key.target, cache_.size());
}

// ============================================================================
// BatchUBODTProcessor Implementation
// ============================================================================

BatchUBODTProcessor::BatchUBODTProcessor(const std::string &ubodt_file,
                                         const Network &network,
                                         double buffer_ratio)
    : ubodt_file_(ubodt_file)
    , network_(network)
    , buffer_ratio_(buffer_ratio)
    , last_stats_{} {
}

template<typename ResultFunc, typename ResultType>
std::vector<ResultType> BatchUBODTProcessor::process_batch(
    const std::vector<Trajectory> &trajectories,
    ResultFunc func) {

    SPDLOG_INFO("Processing batch of {} trajectories with shared PartialUBODT",
                trajectories.size());

    auto start_time = std::chrono::high_resolution_clock::now();

    // Create shared PartialUBODT for all trajectories
    auto partial_ubodt = make_partial_ubodt_from_trajectories(
        ubodt_file_,
        network_,
        trajectories,
        buffer_ratio_
    );

    if (!partial_ubodt->is_valid()) {
        SPDLOG_CRITICAL("Failed to create PartialUBODT for batch");
        return {};
    }

    auto load_time = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start_time).count();

    SPDLOG_INFO("Shared PartialUBODT loaded: {} records from {} sources in {:.2f}s",
                partial_ubodt->get_num_records(),
                partial_ubodt->get_num_sources(),
                load_time);

    // Process each trajectory with shared UBODT
    std::vector<ResultType> results;
    results.reserve(trajectories.size());

    for (const auto &traj : trajectories) {
        results.push_back(func(traj, partial_ubodt));
    }

    // Update statistics
    last_stats_ = {
        trajectories.size(),
        1,  // Single group
        load_time,
        load_time,
        (double)partial_ubodt->get_num_records()
    };

    SPDLOG_INFO("Batch processing completed: {} trajectories processed",
                trajectories.size());

    return results;
}

template<typename ResultFunc, typename ResultType>
std::vector<ResultType> BatchUBODTProcessor::process_groups(
    const std::vector<Trajectory> &trajectories,
    size_t group_size,
    ResultFunc func) {

    SPDLOG_INFO("Processing {} trajectories in groups of {}",
                trajectories.size(), group_size);

    size_t num_groups = (trajectories.size() + group_size - 1) / group_size;
    SPDLOG_INFO("Total groups: {}", num_groups);

    std::vector<ResultType> all_results;
    all_results.reserve(trajectories.size());

    double total_load_time = 0.0;
    double total_records = 0.0;

    // Process each group
    for (size_t i = 0; i < num_groups; ++i) {
        size_t start_idx = i * group_size;
        size_t end_idx = std::min(start_idx + group_size, trajectories.size());

        // Create subgroup
        std::vector<Trajectory> group(
            trajectories.begin() + start_idx,
            trajectories.begin() + end_idx
        );

        SPDLOG_INFO("Processing group {}/{} ({} trajectories)",
                    i + 1, num_groups, group.size());

        // Process this group
        auto group_results = process_batch(group, func);
        all_results.insert(all_results.end(),
                          group_results.begin(),
                          group_results.end());

        total_load_time += last_stats_.total_load_time;
        total_records += last_stats_.avg_records_per_group;
    }

    // Update statistics
    last_stats_ = {
        trajectories.size(),
        num_groups,
        total_load_time,
        total_load_time / num_groups,
        total_records / num_groups
    };

    SPDLOG_INFO("Group processing completed: {} groups, total load time: {:.2f}s",
                num_groups, total_load_time);

    return all_results;
}

// ============================================================================
// IncrementalUBODT Implementation
// ============================================================================

IncrementalUBODT::IncrementalUBODT(const std::string &ubodt_file,
                                   const Network &network,
                                   size_t initial_capacity)
    : ubodt_file_(ubodt_file)
    , network_(network)
    , ubodt_(nullptr)
    , load_stats_{0, 0, 0.0} {

    loaded_nodes_.reserve(initial_capacity);
    SPDLOG_INFO("Created IncrementalUBODT with initial capacity {}", initial_capacity);
}

size_t IncrementalUBODT::add_nodes(const std::unordered_set<NodeIndex> &new_nodes) {
    size_t added_count = 0;

    for (NodeIndex node : new_nodes) {
        if (loaded_nodes_.find(node) == loaded_nodes_.end()) {
            loaded_nodes_.insert(node);
            ++added_count;
        }
    }

    if (added_count > 0) {
        SPDLOG_INFO("Added {} new nodes to IncrementalUBODT (total: {})",
                    added_count, loaded_nodes_.size());
        reload_ubodt();
    }

    return added_count;
}

size_t IncrementalUBODT::add_bbox(
    const boost::geometry::model::box<Point> &bbox,
    double buffer_ratio) {

    // Extract nodes from bbox
    auto nodes_in_bbox = PartialUBODT::extract_nodes_in_bbox(
        network_,
        bbox,
        buffer_ratio
    );

    return add_nodes(nodes_in_bbox);
}

size_t IncrementalUBODT::add_trajectories(
    const std::vector<Trajectory> &trajectories,
    double buffer_ratio) {

    if (trajectories.empty()) return 0;

    // Calculate bbox
    auto bbox = PartialUBODT::calculate_trajectories_bbox(trajectories);

    // Extract and add nodes
    auto nodes = PartialUBODT::extract_nodes_in_bbox(network_, bbox, buffer_ratio);
    return add_nodes(nodes);
}

bool IncrementalUBODT::has_node(NodeIndex node) const {
    return loaded_nodes_.find(node) != loaded_nodes_.end();
}

const Record *IncrementalUBODT::look_up(NodeIndex source, NodeIndex target) const {
    if (!ubodt_) return nullptr;
    return ubodt_->look_up(source, target);
}

std::vector<EdgeIndex> IncrementalUBODT::look_sp_path(NodeIndex source,
                                                        NodeIndex target) const {
    if (!ubodt_) return {};
    return ubodt_->look_sp_path(source, target);
}

void IncrementalUBODT::reload_ubodt() {
    auto start_time = std::chrono::high_resolution_clock::now();

    SPDLOG_INFO("Reloading UBODT with {} nodes", loaded_nodes_.size());

    // Create PartialUBODT with current node set
    auto partial_ubodt = std::make_shared<PartialUBODT>(
        ubodt_file_,
        network_,
        loaded_nodes_
    );

    if (!partial_ubodt->is_valid()) {
        SPDLOG_ERROR("Failed to reload UBODT incrementally");
        return;
    }

    ubodt_ = partial_ubodt->get_ubodt();

    auto load_time = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start_time).count();

    // Update statistics
    ++load_stats_.total_loads;
    load_stats_.total_nodes_loaded = loaded_nodes_.size();
    load_stats_.total_load_time += load_time;

    SPDLOG_INFO("UBODT reloaded: {} records in {:.2f}s (total loads: {}, avg time: {:.2f}s)",
                ubodt_->get_num_rows(),
                load_time,
                load_stats_.total_loads,
                load_stats_.total_load_time / load_stats_.total_loads);
}
