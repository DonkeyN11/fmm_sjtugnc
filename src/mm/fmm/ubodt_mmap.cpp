//
// Created by Claude for memory-mapped UBODT optimization
//

#include "mm/fmm/ubodt_mmap.hpp"
#include "util/util.hpp"

#include <algorithm>
#include <fstream>
#include <cstring>

using namespace FMM;
using namespace FMM::CORE;
using namespace FMM::NETWORK;
using namespace FMM::MM;

UBODT_MMap::UBODT_MMap(const std::string &filename) : fd_(-1), file_size_(0), data_(nullptr), num_records_(0), delta_(0.0) {
    SPDLOG_INFO("Creating memory-mapped UBODT from {}", filename);

    // Open the file
    fd_ = open(filename.c_str(), O_RDONLY);
    if (fd_ == -1) {
        SPDLOG_CRITICAL("Failed to open UBODT file: {}", filename);
        return;
    }

    // Get file size
    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
        SPDLOG_CRITICAL("Failed to get file size: {}", filename);
        close(fd_);
        fd_ = -1;
        return;
    }
    file_size_ = sb.st_size;

    // Check if file size is valid for MmapRecord structure
    if (file_size_ % sizeof(MmapRecord) != 0) {
        SPDLOG_WARN("File size {} is not aligned with record size {}", file_size_, sizeof(MmapRecord));
        num_records_ = file_size_ / sizeof(MmapRecord);
    } else {
        num_records_ = file_size_ / sizeof(MmapRecord);
    }

    SPDLOG_INFO("UBODT file contains {} records", num_records_);

    // Map the file into memory
    data_ = static_cast<const MmapRecord*>(mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0));
    if (data_ == MAP_FAILED) {
        SPDLOG_CRITICAL("Failed to mmap UBODT file: {}", filename);
        close(fd_);
        fd_ = -1;
        data_ = nullptr;
        return;
    }

    // Find the maximum delta value
    double max_delta = 0.0;
    for (size_t i = 0; i < num_records_; ++i) {
        if (data_[i].cost > max_delta) {
            max_delta = data_[i].cost;
        }
    }
    delta_ = max_delta;

    SPDLOG_INFO("UBODT delta value: {}", delta_);

    // Build spatial index for faster lookups
    build_spatial_index();

    SPDLOG_INFO("Memory-mapped UBODT loaded successfully");
}

UBODT_MMap::~UBODT_MMap() {
    if (data_ != nullptr) {
        munmap(const_cast<MmapRecord*>(data_), file_size_);
        data_ = nullptr;
    }
    if (fd_ != -1) {
        close(fd_);
        fd_ = -1;
    }
}

void UBODT_MMap::build_spatial_index() {
    SPDLOG_INFO("Building spatial index for {} records", num_records_);

    // Temporary vector for building source index
    std::vector<std::pair<NodeIndex, size_t>> source_offsets;
    source_offsets.reserve(num_records_);

    // Collect all source nodes and their offsets
    for (size_t i = 0; i < num_records_; ++i) {
        source_offsets.emplace_back(data_[i].source, i);
    }

    // Sort by source node
    std::sort(source_offsets.begin(), source_offsets.end());

    // Build compressed index
    NodeIndex current_source = std::numeric_limits<NodeIndex>::max();
    for (const auto& entry : source_offsets) {
        if (entry.first != current_source) {
            // New source node found
            if (!source_index_.empty()) {
                // Update count for previous source
                source_index_.back().count = entry.second - source_index_.back().start_offset;
            }
            // Add new source entry
            source_index_.push_back({entry.first, entry.second, 0});
            current_source = entry.first;
        }
    }

    // Update count for the last source
    if (!source_index_.empty()) {
        source_index_.back().count = num_records_ - source_index_.back().start_offset;
    }

    SPDLOG_INFO("Spatial index built with {} unique source nodes", source_index_.size());
}

const MmapRecord *UBODT_MMap::look_up(NodeIndex source, NodeIndex target) const {
    if (!data_) return nullptr;

    // Binary search in source index
    auto it = std::lower_bound(source_index_.begin(), source_index_.end(), source,
        [](const SourceIndex& idx, NodeIndex val) {
            return idx.source < val;
        });

    if (it == source_index_.end() || it->source != source) {
        return nullptr; // Source not found
    }

    // Linear search within the source's records
    for (size_t i = it->start_offset; i < it->start_offset + it->count; ++i) {
        if (data_[i].target == target) {
            return &data_[i];
        }
    }

    return nullptr; // Target not found for this source
}

std::vector<EdgeIndex> UBODT_MMap::look_sp_path(NodeIndex source, NodeIndex target) const {
    std::vector<EdgeIndex> edges;
    if (source == target) return edges;

    const MmapRecord *r = look_up(source, target);
    if (r == nullptr) return edges;

    // Reconstruct the path by following the records
    while (r->first_n != target) {
        edges.push_back(r->next_e);
        r = look_up(r->first_n, target);
        if (r == nullptr) {
            edges.clear(); // Path broken
            return edges;
        }
    }
    edges.push_back(r->next_e);

    return edges;
}

std::shared_ptr<UBODT_MMap> FMM::MM::make_mmap_ubodt(const std::string &filename) {
    auto start_time = UTIL::get_current_time();

    auto ubodt = std::make_shared<UBODT_MMap>(filename);

    auto end_time = UTIL::get_current_time();
    double duration = UTIL::get_duration(start_time, end_time);

    if (ubodt->is_valid()) {
        SPDLOG_INFO("Memory-mapped UBODT loaded in {} seconds", duration);
    } else {
        SPDLOG_ERROR("Failed to load memory-mapped UBODT in {} seconds", duration);
    }

    return ubodt;
}