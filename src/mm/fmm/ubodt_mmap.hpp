/**
 * Fast map matching.
 *
 * Memory-mapped UBODT reader for performance optimization
 *
 * @author: Ning
 * @version: 2024.09.16
 */

#ifndef FMM_SRC_MM_FMM_UBODT_MMAP_HPP_
#define FMM_SRC_MM_FMM_UBODT_MMAP_HPP_

#include "mm/fmm/ubodt.hpp"
#include "network/type.hpp"
#include "util/debug.hpp"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <memory>

namespace FMM {
namespace MM {

/**
 * Memory-mapped UBODT record structure (packed for efficiency)
 */
struct MmapRecord {
    NETWORK::NodeIndex source;
    NETWORK::NodeIndex target;
    NETWORK::NodeIndex first_n;
    NETWORK::NodeIndex prev_n;
    NETWORK::EdgeIndex next_e;
    double cost;
} __attribute__((packed));

/**
 * Memory-mapped UBODT reader for fast access
 */
class UBODT_MMap {
public:
    UBODT_MMap(const UBODT_MMap &) = delete;
    UBODT_MMap &operator=(const UBODT_MMap &) = delete;

    /**
     * Constructor - maps UBODT file into memory
     * @param filename Path to UBODT binary file
     */
    UBODT_MMap(const std::string &filename);

    ~UBODT_MMap();

    /**
     * Look up a record by source and target nodes
     * @param source Source node index
     * @param target Target node index
     * @return Pointer to record if found, nullptr otherwise
     */
    const MmapRecord *look_up(NETWORK::NodeIndex source, NETWORK::NodeIndex target) const;

    /**
     * Look up shortest path as edge indices
     * @param source Source node index
     * @param target Target node index
     * @return Vector of edge indices representing the path
     */
    std::vector<NETWORK::EdgeIndex> look_sp_path(NETWORK::NodeIndex source,
                                                  NETWORK::NodeIndex target) const;

    /**
     * Get the number of records in the UBODT
     * @return Record count
     */
    inline size_t get_num_records() const { return num_records_; }

    /**
     * Get the upperbound (delta) value
     * @return Delta value
     */
    inline double get_delta() const { return delta_; }

    /**
     * Check if the mapping is valid
     * @return True if successfully mapped
     */
    inline bool is_valid() const { return data_ != nullptr; }

private:
    /**
     * Build spatial index for faster lookups
     */
    void build_spatial_index();

    int fd_;                          // File descriptor
    size_t file_size_;                // File size in bytes
    const MmapRecord *data_;          // Mapped data pointer
    size_t num_records_;              // Number of records
    double delta_;                     // Upperbound delta value

    // Spatial indexing structure for faster lookups
    struct SourceIndex {
        NETWORK::NodeIndex source;
        size_t start_offset;
        size_t count;
    };

    std::vector<SourceIndex> source_index_;  // Index sorted by source node
};

/**
 * Factory function to create memory-mapped UBODT
 * @param filename Path to UBODT file
 * @return Shared pointer to UBODT_MMap instance
 */
std::shared_ptr<UBODT_MMap> make_mmap_ubodt(const std::string &filename);

} // MM
} // FMM

#endif //FMM_SRC_MM_FMM_UBODT_MMAP_HPP_