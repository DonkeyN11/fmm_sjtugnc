/**
 * Fast map matching.
 *
 * Partial UBODT loader for dynamic loading based on trajectory extent
 *
 * This module loads only the UBODT records relevant to the trajectories
 * being matched, significantly reducing memory usage and load time.
 *
 * @author: Optimized for dynamic loading
 * @version: 2025.01.22
 */

#ifndef FMM_SRC_MM_FMM_UBODT_PARTIAL_HPP_
#define FMM_SRC_MM_FMM_UBODT_PARTIAL_HPP_

#include "mm/fmm/ubodt.hpp"
#include "mm/fmm/ubodt_mmap.hpp"
#include "network/network.hpp"
#include "core/gps.hpp"
#include "util/debug.hpp"

#include <unordered_set>
#include <vector>
#include <memory>
#include <algorithm>
#include <boost/geometry.hpp>

namespace FMM {
namespace MM {

// Shortcut for box type
typedef boost::geometry::model::box<FMM::CORE::Point> Box;

/**
 * Partial UBODT loader that loads only relevant records
 */
class PartialUBODT {
public:
    /**
     * Constructor - creates a partial UBODT from a full UBODT file
     * @param filename Path to the UBODT binary file
     * @param network Road network
     * @param required_nodes Set of node indices to load
     */
    PartialUBODT(const std::string &filename,
                 const NETWORK::Network &network,
                 const std::unordered_set<NETWORK::NodeIndex> &required_nodes);

    /**
     * Constructor - creates a partial UBODT from trajectory extent
     * @param filename Path to the UBODT binary file
     * @param network Road network
     * @param trajectories Vector of trajectories
     * @param buffer_ratio Buffer ratio to expand the bounding box (default: 0.1 = 10%)
     */
    PartialUBODT(const std::string &filename,
                 const NETWORK::Network &network,
                 const std::vector<CORE::Trajectory> &trajectories,
                 double buffer_ratio = 0.1);

    /**
     * Look up a record by source and target nodes
     * @param source Source node index
     * @param target Target node index
     * @return Pointer to record if found, nullptr otherwise
     */
    const Record *look_up(NETWORK::NodeIndex source, NETWORK::NodeIndex target) const;

    /**
     * Look up shortest path as edge indices
     * @param source Source node index
     * @param target Target node index
     * @return Vector of edge indices representing the path
     */
    std::vector<NETWORK::EdgeIndex> look_sp_path(NETWORK::NodeIndex source,
                                                  NETWORK::NodeIndex target) const;

    /**
     * Get the number of records loaded
     * @return Record count
     */
    inline size_t get_num_records() const { return num_records_; }

    /**
     * Get the number of source nodes loaded
     * @return Source node count
     */
    inline size_t get_num_sources() const { return num_sources_; }

    /**
     * Check if the loading was successful
     * @return True if successfully loaded
     */
    inline bool is_valid() const { return ubodt_ != nullptr; }

    /**
     * Get the underlying UBODT pointer
     * @return Shared pointer to UBODT
     */
    inline std::shared_ptr<UBODT> get_ubodt() const { return ubodt_; }

    /**
     * Calculate the bounding box of trajectories
     * @param trajectories Vector of trajectories
     * @return Bounding box as boost::geometry::model::box
     */
    static boost::geometry::model::box<CORE::Point> calculate_trajectories_bbox(
        const std::vector<CORE::Trajectory> &trajectories);

    /**
     * Extract nodes within a bounding box
     * @param network Road network
     * @param bbox Bounding box
     * @param buffer_ratio Buffer ratio to expand the box
     * @return Set of node indices within the box
     */
    static std::unordered_set<NETWORK::NodeIndex> extract_nodes_in_bbox(
        const NETWORK::Network &network,
        const boost::geometry::model::box<CORE::Point> &bbox,
        double buffer_ratio = 0.1);

private:
    std::shared_ptr<UBODT> ubodt_;
    size_t num_records_;
    size_t num_sources_;
    double load_time_;
};

/**
 * Factory function to create a partial UBODT from node set
 * @param filename Path to UBODT file
 * @param network Road network
 * @param required_nodes Set of required node indices
 * @return Shared pointer to PartialUBODT
 */
std::shared_ptr<PartialUBODT> make_partial_ubodt_from_nodes(
    const std::string &filename,
    const NETWORK::Network &network,
    const std::unordered_set<NETWORK::NodeIndex> &required_nodes);

/**
 * Factory function to create a partial UBODT from trajectories
 * @param filename Path to UBODT file
 * @param network Road network
 * @param trajectories Vector of trajectories
 * @param buffer_ratio Buffer ratio to expand bounding box
 * @return Shared pointer to PartialUBODT
 */
std::shared_ptr<PartialUBODT> make_partial_ubodt_from_trajectories(
    const std::string &filename,
    const NETWORK::Network &network,
    const std::vector<CORE::Trajectory> &trajectories,
    double buffer_ratio = 0.1);

} // MM
} // FMM

#endif // FMM_SRC_MM_FMM_UBODT_PARTIAL_HPP_
