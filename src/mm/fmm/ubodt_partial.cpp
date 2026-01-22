//
// Created for dynamic UBODT loading based on trajectory extent
//

#include "mm/fmm/ubodt_partial.hpp"
#include "util/util.hpp"

#include <fstream>
#include <limits>
#include <algorithm>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

using namespace FMM;
using namespace FMM::CORE;
using namespace FMM::NETWORK;
using namespace FMM::MM;

// Calculate bounding box from trajectories
boost::geometry::model::box<Point> PartialUBODT::calculate_trajectories_bbox(
    const std::vector<Trajectory> &trajectories) {

    if (trajectories.empty()) {
        SPDLOG_WARN("No trajectories provided for bbox calculation");
        return boost::geometry::make_inverse<Point>();
    }

    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();

    for (const auto &traj : trajectories) {
        for (const auto &point : traj.geom.line) {
            double x = boost::geometry::get<0>(point);
            double y = boost::geometry::get<1>(point);

            min_x = std::min(min_x, x);
            min_y = std::min(min_y, y);
            max_x = std::max(max_x, x);
            max_y = std::max(max_y, y);
        }
    }

    SPDLOG_INFO("Trajectory bounding box: ({}, {}, {}, {})", min_x, min_y, max_x, max_y);

    Point min_pt(min_x, min_y);
    Point max_pt(max_x, max_y);
    return boost::geometry::make<boost::geometry::model::box<Point>>(min_pt, max_pt);
}

// Extract nodes within a bounding box
std::unordered_set<NodeIndex> PartialUBODT::extract_nodes_in_bbox(
    const Network &network,
    const boost::geometry::model::box<Point> &bbox,
    double buffer_ratio) {

    // Expand the bounding box by buffer_ratio
    double min_x = boost::geometry::get<0>(bbox.min_corner());
    double min_y = boost::geometry::get<1>(bbox.min_corner());
    double max_x = boost::geometry::get<0>(bbox.max_corner());
    double max_y = boost::geometry::get<1>(bbox.max_corner());

    double width = max_x - min_x;
    double height = max_y - min_y;

    double buffer_x = width * buffer_ratio;
    double buffer_y = height * buffer_ratio;

    Point expanded_min(min_x - buffer_x, min_y - buffer_y);
    Point expanded_max(max_x + buffer_x, max_y + buffer_y);
    boost::geometry::model::box<Point> expanded_bbox(expanded_min, expanded_max);

    SPDLOG_INFO("Expanded bounding box with ratio {}: ({}, {}, {}, {})",
                buffer_ratio,
                boost::geometry::get<0>(expanded_bbox.min_corner()),
                boost::geometry::get<1>(expanded_bbox.min_corner()),
                boost::geometry::get<0>(expanded_bbox.max_corner()),
                boost::geometry::get<1>(expanded_bbox.max_corner()));

    std::unordered_set<NodeIndex> nodes_in_bbox;
    const std::vector<Edge> &edges = network.get_edges();

    // Iterate through all edges and check if they intersect with the bounding box
    for (const auto &edge : edges) {
        // Get the envelope (bounding box) of the edge geometry
        boost::geometry::model::box<Point> edge_bbox;
        boost::geometry::envelope(edge.geom.line, edge_bbox);

        // Check if edge's bounding box intersects with our query box
        if (boost::geometry::intersects(edge_bbox, expanded_bbox)) {
            nodes_in_bbox.insert(edge.source);
            nodes_in_bbox.insert(edge.target);
        }
    }

    SPDLOG_INFO("Found {} unique nodes in bounding box (from {} edges)",
                nodes_in_bbox.size(), edges.size());

    return nodes_in_bbox;
}

// Constructor from node set
PartialUBODT::PartialUBODT(const std::string &filename,
                           const Network &network,
                           const std::unordered_set<NodeIndex> &required_nodes)
    : num_records_(0), num_sources_(0), load_time_(0.0) {

    if (required_nodes.empty()) {
        SPDLOG_WARN("No required nodes specified, loading full UBODT");
        ubodt_ = UBODT::read_ubodt_file(filename);
        num_records_ = ubodt_ ? ubodt_->get_num_rows() : 0;
        num_sources_ = required_nodes.size();
        return;
    }

    SPDLOG_INFO("Loading partial UBODT from {} with {} required nodes",
                filename, required_nodes.size());

    auto start_time = UTIL::get_current_time();

    // Check if file exists
    struct stat sb;
    if (stat(filename.c_str(), &sb) != 0) {
        SPDLOG_CRITICAL("UBODT file not found: {}", filename);
        return;
    }

    // Open and memory map the file
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        SPDLOG_CRITICAL("Failed to open UBODT file: {}", filename);
        return;
    }

    size_t file_size = sb.st_size;
    const MmapRecord *mapped_data = static_cast<const MmapRecord *>(
        mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0));

    if (mapped_data == MAP_FAILED) {
        close(fd);
        SPDLOG_CRITICAL("Failed to mmap UBODT file: {}", filename);
        return;
    }

    size_t total_records = file_size / sizeof(MmapRecord);
    SPDLOG_INFO("Total records in file: {}", total_records);

    // First pass: build source index and count required records
    struct SourceOffset {
        NodeIndex source;
        size_t offset;
    };

    std::vector<SourceOffset> source_offsets;
    source_offsets.reserve(total_records);

    for (size_t i = 0; i < total_records; ++i) {
        source_offsets.push_back({mapped_data[i].source, i});
    }

    // Sort by source for binary search
    std::sort(source_offsets.begin(), source_offsets.end(),
              [](const SourceOffset &a, const SourceOffset &b) {
                  return a.source < b.source;
              });

    // Count records for required sources
    size_t required_count = 0;
    std::unordered_set<NodeIndex> found_sources;

    auto current_source = source_offsets.begin();
    for (NodeIndex node : required_nodes) {
        // Binary search for this source
        auto it = std::lower_bound(source_offsets.begin(), source_offsets.end(), node,
                                   [](const SourceOffset &so, NodeIndex n) {
                                       return so.source < n;
                                   });

        if (it != source_offsets.end() && it->source == node) {
            // Count records for this source
            size_t start = it->offset;
            size_t end = start;
            while (end < total_records && mapped_data[end].source == node) {
                ++end;
            }
            required_count += (end - start);
            found_sources.insert(node);
        }
    }

    SPDLOG_INFO("Found {} records for {} source nodes",
                required_count, found_sources.size());

    if (required_count == 0) {
        SPDLOG_WARN("No records found for required nodes");
        munmap(const_cast<MmapRecord *>(mapped_data), file_size);
        close(fd);
        return;
    }

    // Create UBODT with appropriate size
    int buckets = UBODT::find_prime_number(required_count / UBODT::LOAD_FACTOR);
    ubodt_ = std::make_shared<UBODT>(buckets, 1);

    // Second pass: load only required records
    size_t loaded_count = 0;
    for (NodeIndex node : required_nodes) {
        // Binary search for this source
        auto it = std::lower_bound(source_offsets.begin(), source_offsets.end(), node,
                                   [](const SourceOffset &so, NodeIndex n) {
                                       return so.source < n;
                                   });

        if (it != source_offsets.end() && it->source == node) {
            size_t idx = it->offset;
            while (idx < total_records && mapped_data[idx].source == node) {
                const MmapRecord &mmap_rec = mapped_data[idx];

                // Check if target is also in required nodes (optional optimization)
                // For now, load all records from required sources

                Record *r = (Record *) malloc(sizeof(Record));
                r->source = mmap_rec.source;
                r->target = mmap_rec.target;
                r->first_n = mmap_rec.first_n;
                r->prev_n = mmap_rec.prev_n;
                r->next_e = mmap_rec.next_e;
                r->cost = mmap_rec.cost;
                r->next = nullptr;

                ubodt_->insert(r);
                ++loaded_count;

                ++idx;
            }
        }
    }

    munmap(const_cast<MmapRecord *>(mapped_data), file_size);
    close(fd);

    auto end_time = UTIL::get_current_time();
    load_time_ = UTIL::get_duration(start_time, end_time);

    num_records_ = loaded_count;
    num_sources_ = found_sources.size();

    SPDLOG_INFO("Loaded {} records from {} sources in {:.2f} seconds",
                num_records_, num_sources_, load_time_);
    SPDLOG_INFO("Reduction: {:.1f}% of total records",
                100.0 * (1.0 - double(num_records_) / total_records));
}

// Constructor from trajectories
PartialUBODT::PartialUBODT(const std::string &filename,
                           const Network &network,
                           const std::vector<Trajectory> &trajectories,
                           double buffer_ratio)
    : num_records_(0), num_sources_(0), load_time_(0.0) {

    if (trajectories.empty()) {
        SPDLOG_WARN("No trajectories provided");
        return;
    }

    SPDLOG_INFO("Creating partial UBODT from {} trajectories", trajectories.size());

    // Calculate bounding box
    auto bbox = calculate_trajectories_bbox(trajectories);

    // Extract nodes in bbox
    auto required_nodes = extract_nodes_in_bbox(network, bbox, buffer_ratio);

    // Use node-based constructor
    *this = PartialUBODT(filename, network, required_nodes);
}

// Look up a record
const Record *PartialUBODT::look_up(NodeIndex source, NodeIndex target) const {
    if (!ubodt_) return nullptr;
    return ubodt_->look_up(source, target);
}

// Look up shortest path
std::vector<EdgeIndex> PartialUBODT::look_sp_path(NodeIndex source, NodeIndex target) const {
    if (!ubodt_) return {};
    return ubodt_->look_sp_path(source, target);
}

// Factory function from nodes
std::shared_ptr<PartialUBODT> make_partial_ubodt_from_nodes(
    const std::string &filename,
    const Network &network,
    const std::unordered_set<NodeIndex> &required_nodes) {

    return std::make_shared<PartialUBODT>(filename, network, required_nodes);
}

// Factory function from trajectories
std::shared_ptr<PartialUBODT> make_partial_ubodt_from_trajectories(
    const std::string &filename,
    const Network &network,
    const std::vector<Trajectory> &trajectories,
    double buffer_ratio) {

    return std::make_shared<PartialUBODT>(filename, network, trajectories, buffer_ratio);
}
