//
// Created by Ning: memory-mapped UBODT optimization
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

UBODT_MMap::UBODT_MMap(const std::string &filename, bool is_indexed_format)
    : fd_(-1), file_size_(0), data_(nullptr), raw_data_(nullptr),
      num_records_(0), delta_(0.0), header_offset_(0),
      is_indexed_format_(is_indexed_format) {
    
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

    // Map the file into memory
    void* mapped_ptr = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped_ptr == MAP_FAILED) {
        SPDLOG_CRITICAL("Failed to mmap UBODT file: {}", filename);
        close(fd_);
        fd_ = -1;
        return;
    }
    raw_data_ = static_cast<const char*>(mapped_ptr);

    if (is_indexed_format_) {
        // Read header to determine offset and record count
        load_index_from_header();
        
        // Point data to the start of records
        if (raw_data_) {
            data_ = reinterpret_cast<const MmapRecord*>(raw_data_ + header_offset_);
        }
    } else {
        // Standard raw binary format
        if (file_size_ % sizeof(MmapRecord) != 0) {
            SPDLOG_WARN("File size {} is not aligned with record size {}", file_size_, sizeof(MmapRecord));
        }
        num_records_ = file_size_ / sizeof(MmapRecord);
        data_ = reinterpret_cast<const MmapRecord*>(raw_data_);
        
        // Need to build index and scan delta
        // Find the maximum delta value
        double max_delta = 0.0;
        for (size_t i = 0; i < num_records_; ++i) {
            if (data_[i].cost > max_delta) {
                max_delta = data_[i].cost;
            }
        }
        delta_ = max_delta;
        
        build_spatial_index();
    }

    if (is_valid()) {
        SPDLOG_INFO("Memory-mapped UBODT loaded successfully. Records: {}, Delta: {}", num_records_, delta_);
    }
}

UBODT_MMap::~UBODT_MMap() {
    if (raw_data_ != nullptr) {
        munmap(const_cast<char*>(raw_data_), file_size_);
        raw_data_ = nullptr;
        data_ = nullptr;
    }
    if (fd_ != -1) {
        close(fd_);
        fd_ = -1;
    }
}

void UBODT_MMap::load_index_from_header() {
    if (!raw_data_) return;

    // Header structure:
    // uint64_t num_records
    // uint64_t num_sources
    // Index entries: [source(uint32), start_offset(uint64), count(uint64)] * num_sources
    
    uint64_t num_sources = 0;
    
    // Read counts
    memcpy(&num_records_, raw_data_, sizeof(uint64_t));
    memcpy(&num_sources, raw_data_ + sizeof(uint64_t), sizeof(uint64_t));
    
    // Calculate index size
    // Note: NodeIndex is int (4 bytes) usually, check definition
    // In ubodt_converter it writes: source(NodeIndex), start(uint64), count(uint64)
    size_t index_entry_size = sizeof(NETWORK::NodeIndex) + 2 * sizeof(uint64_t);
    header_offset_ = 2 * sizeof(uint64_t) + num_sources * index_entry_size;
    
    SPDLOG_INFO("Loading pre-calculated index: {} records, {} sources", num_records_, num_sources);
    
    // Reserve space
    source_index_.reserve(num_sources);
    
    // Pointer to start of index
    const char* index_ptr = raw_data_ + 2 * sizeof(uint64_t);
    
    for (size_t i = 0; i < num_sources; ++i) {
        NETWORK::NodeIndex source;
        uint64_t start, count;
        
        memcpy(&source, index_ptr, sizeof(source));
        index_ptr += sizeof(source);
        memcpy(&start, index_ptr, sizeof(start));
        index_ptr += sizeof(start);
        memcpy(&count, index_ptr, sizeof(count));
        index_ptr += sizeof(count);
        
        // The start offset in the file index is relative to the record section
        source_index_.push_back({source, static_cast<size_t>(start), static_cast<size_t>(count)});
    }
    
    // We don't scan for delta in indexed mode to save time, 
    // unless we want to assume a default or read it if it was stored (it isn't currently)
    // Let's set a safe default or 0 (it's mostly used for metadata)
    delta_ = 5000.0; // Reasonable default or TODO: store delta in header
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

    // Check if indexed binary format
    bool is_indexed = false;
    // We can peek the file or just try to detect from extension/header
    // Using UBODT::is_indexed_binary_format which is robust
    // But that requires including ubodt.hpp which is already included
    
    // Simple check: Is it likely indexed?
    // If filename ends in .bin or .binary, we check the header
    uint64_t rc, sc, hs;
    if (UBODT::is_indexed_binary_format(filename, &rc, &sc, &hs)) {
        is_indexed = true;
    }

    auto ubodt = std::make_shared<UBODT_MMap>(filename, is_indexed);

    auto end_time = UTIL::get_current_time();
    double duration = UTIL::get_duration(start_time, end_time);

    if (ubodt->is_valid()) {
        SPDLOG_INFO("Memory-mapped UBODT loaded in {} seconds (Mode: {})", 
                   duration, is_indexed ? "Indexed" : "Raw");
    } else {
        SPDLOG_ERROR("Failed to load memory-mapped UBODT in {} seconds", duration);
    }

    return ubodt;
}