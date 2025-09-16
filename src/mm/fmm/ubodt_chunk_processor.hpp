/**
 * Fast map matching.
 *
 * UBODT chunk-based parallel processor
 *
 * @author: Enhanced by Claude for chunk-based parallel processing
 * @version: 2024.09.16
 */

#ifndef FMM_SRC_MM_FMM_UBODT_CHUNK_PROCESSOR_HPP_
#define FMM_SRC_MM_FMM_UBODT_CHUNK_PROCESSOR_HPP_

#include "mm/fmm/ubodt_gen_algorithm.hpp"
#include "mm/fmm/ubodt.hpp"
#include "network/network.hpp"
#include "network/network_graph.hpp"
#include "core/geometry.hpp"
#include "util/debug.hpp"

#include <vector>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <filesystem>

#ifdef BOOST_OS_WINDOWS
#include <boost/throw_exception.hpp>
#endif

namespace FMM {
namespace MM {

// Forward declaration
struct Record;

/**
 * Geographic bounding box for chunking
 */
struct BoundingBox {
    double min_x, max_x;
    double min_y, max_y;

    BoundingBox(double min_x_, double max_x_, double min_y_, double max_y_)
        : min_x(min_x_), max_x(max_x_), min_y(min_y_), max_y(max_y_) {}

    bool contains(const FMM::CORE::Point& point) const {
        return boost::geometry::get<0>(point) >= min_x && boost::geometry::get<0>(point) <= max_x &&
               boost::geometry::get<1>(point) >= min_y && boost::geometry::get<1>(point) <= max_y;
    }

    bool intersects(const BoundingBox& other) const {
        return !(max_x < other.min_x || min_x > other.max_x ||
                 max_y < other.min_y || min_y > other.max_y);
    }

    BoundingBox expand(double buffer) const {
        return BoundingBox(min_x - buffer, max_x + buffer,
                          min_y - buffer, max_y + buffer);
    }
};

/**
 * Network chunk definition
 */
struct NetworkChunk {
    int chunk_id;
    BoundingBox bounds;
    BoundingBox expanded_bounds;  // with delta buffer
    std::vector<NETWORK::NodeIndex> nodes;
    std::vector<NETWORK::NodeIndex> boundary_nodes;

    NetworkChunk(int id, const BoundingBox& bbox, double delta = 0.0)
        : chunk_id(id), bounds(bbox), expanded_bounds(bbox.expand(delta)) {}
};

/**
 * Chunk-based UBODT processor
 */
class UBODTChunkProcessor {
public:
    /**
     * Constructor
     * @param network Road network
     * @param graph Network graph
     * @param delta Upper bound distance for path search
     */
    UBODTChunkProcessor(const NETWORK::Network &network,
                        const NETWORK::NetworkGraph &graph,
                        double delta);

    /**
     * Generate UBODT using chunk-based parallel processing
     * @param filename Output filename
     * @param chunk_rows Number of chunks in row direction
     * @param chunk_cols Number of chunks in column direction
     * @param num_threads Number of parallel threads
     * @param binary Whether to use binary format
     */
    std::string generate_ubodt_chunked(const std::string &filename,
                                      int chunk_rows, int chunk_cols,
                                      int num_threads, bool binary = true);

    /**
     * Create grid-based chunks
     * @param rows Number of rows
     * @param cols Number of columns
     * @return Vector of chunks
     */
    std::vector<NetworkChunk> create_grid_chunks(int rows, int cols);

    /**
     * Process a single chunk
     * @param chunk Chunk to process
     * @param temp_filename Temporary output filename
     * @param binary Whether to use binary format
     */
    void process_chunk(const NetworkChunk& chunk,
                      const std::string& temp_filename,
                      bool binary);

    /**
     * Merge chunk results into final UBODT
     * @param chunk_files List of temporary chunk files
     * @param final_filename Final output filename
     * @param binary Whether to use binary format
     */
    void merge_chunk_results(const std::vector<std::string>& chunk_files,
                           const std::string& final_filename,
                           bool binary);

private:
    const NETWORK::Network &network_;
    const NETWORK::NetworkGraph &graph_;
    double delta_;

    /**
     * Get bounding box of the entire network
     */
    BoundingBox get_network_bounds() const;

    /**
     * Find nodes within a bounding box
     * @param bounds Bounding box
     * @return Vector of node indices
     */
    std::vector<NETWORK::NodeIndex> find_nodes_in_bounds(const BoundingBox& bounds) const;

    /**
     * Identify boundary nodes in a chunk
     * @param chunk Chunk to analyze
     */
    void identify_boundary_nodes(NetworkChunk& chunk) const;

    /**
     * Write chunk results to file
     * @param chunk Chunk results
     * @param filename Output filename
     * @param binary Whether to use binary format
     */
    void write_chunk_results(const NetworkChunk& chunk,
                           const std::vector<Record>& records,
                           const std::string& filename,
                           bool binary) const;
};

/**
 * Chunk-based UBODT generation algorithm
 */
class UBODTGenAlgorithmChunked : public UBODTGenAlgorithm {
public:
    UBODTGenAlgorithmChunked(const NETWORK::Network &network,
                            const NETWORK::NetworkGraph &graph)
        : UBODTGenAlgorithm(network, graph) {}

    /**
     * Generate UBODT using chunk-based parallel processing
     * @param filename Output filename
     * @param delta Upper bound distance
     * @param binary Whether to use binary format
     * @param use_chunking Whether to use chunk-based processing
     * @param chunk_rows Number of chunk rows
     * @param chunk_cols Number of chunk columns
     * @param num_threads Number of threads
     */
    std::string generate_ubodt_chunked(const std::string &filename, double delta,
                                     bool binary = true, bool use_chunking = true,
                                     int chunk_rows = 4, int chunk_cols = 4,
                                     int num_threads = 8) const;
};

} // MM
} // FMM

#endif //FMM_SRC_MM_FMM_UBODT_CHUNK_PROCESSOR_HPP_