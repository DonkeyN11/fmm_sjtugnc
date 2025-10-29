
#include "mm/fmm/ubodt_converter.hpp"
#include "mm/fmm/ubodt_mmap.hpp"
#include "util/util.hpp"

#include <fstream>
#include <algorithm>
#include <unordered_map>
#include <sstream>

using namespace FMM;
using namespace FMM::CORE;
using namespace FMM::NETWORK;
using namespace FMM::MM;

bool UBODTConverter::csv_to_mmap_binary(const std::string &input_csv,
                                       const std::string &output_bin,
                                       int progress_interval) {
    SPDLOG_INFO("Converting CSV UBODT to memory-mappable binary format");
    SPDLOG_INFO("Input: {} -> Output: {}", input_csv, output_bin);

    auto start_time = UTIL::get_current_time();

    // Open input CSV file
    std::ifstream input(input_csv);
    if (!input.is_open()) {
        SPDLOG_CRITICAL("Failed to open input CSV file: {}", input_csv);
        return false;
    }

    // Open output binary file
    std::ofstream output(output_bin, std::ios::binary);
    if (!output.is_open()) {
        SPDLOG_CRITICAL("Failed to open output binary file: {}", output_bin);
        input.close();
        return false;
    }

    // Skip CSV header
    std::string header;
    std::getline(input, header);

    long long total_records = 0;
    long long processed_records = 0;
    char line[1024];

    // First pass: count total records
    while (input.getline(line, sizeof(line))) {
        total_records++;
    }

    // Reset file position
    input.clear();
    input.seekg(0);
    std::getline(input, header); // Skip header again

    SPDLOG_INFO("Total records to process: {}", total_records);

    // Second pass: convert and write binary records
    while (input.getline(line, sizeof(line))) {
        processed_records++;

        // Parse CSV line
        Record r;
        sscanf(line, "%d;%d;%d;%d;%d;%lf",
               &r.source, &r.target, &r.first_n,
               &r.prev_n, &r.next_e, &r.cost);

        // Write binary record (packed format)
        output.write(reinterpret_cast<const char*>(&r.source), sizeof(r.source));
        output.write(reinterpret_cast<const char*>(&r.target), sizeof(r.target));
        output.write(reinterpret_cast<const char*>(&r.first_n), sizeof(r.first_n));
        output.write(reinterpret_cast<const char*>(&r.prev_n), sizeof(r.prev_n));
        output.write(reinterpret_cast<const char*>(&r.next_e), sizeof(r.next_e));
        output.write(reinterpret_cast<const char*>(&r.cost), sizeof(r.cost));

        // Progress reporting
        if (processed_records % progress_interval == 0) {
            double progress = (double)processed_records / total_records * 100.0;
            SPDLOG_INFO("Progress: {:.1f}% ({}/{})",
                       progress, processed_records, total_records);
        }
    }

    input.close();
    output.close();

    auto end_time = UTIL::get_current_time();
    double duration = UTIL::get_duration(start_time, end_time);

    SPDLOG_INFO("Conversion completed in {} seconds", duration);
    SPDLOG_INFO("Converted {} records from CSV to binary format", processed_records);

    return true;
}

bool UBODTConverter::csv_to_indexed_binary(const std::string &input_csv,
                                           const std::string &output_bin,
                                           int progress_interval) {
    SPDLOG_INFO("Converting CSV UBODT to indexed binary format");

    auto start_time = UTIL::get_current_time();

    // Open input CSV file
    std::ifstream input(input_csv);
    if (!input.is_open()) {
        SPDLOG_CRITICAL("Failed to open input CSV file: {}", input_csv);
        return false;
    }

    // Skip CSV header
    std::string header;
    std::getline(input, header);

    // Read all records into memory for sorting and indexing
    std::vector<Record*> records;
    char line[1024];
    long long total_records = 0;

    SPDLOG_INFO("Reading records into memory for sorting...");

    while (input.getline(line, sizeof(line))) {
        Record* r = new Record();
        sscanf(line, "%d;%d;%d;%d;%d;%lf",
               &r->source, &r->target, &r->first_n,
               &r->prev_n, &r->next_e, &r->cost);
        r->next = nullptr;
        records.push_back(r);
        total_records++;

        if (total_records % progress_interval == 0) {
            SPDLOG_INFO("Read {} records", total_records);
        }
    }

    input.close();

    SPDLOG_INFO("Total records read: {}", total_records);

    // Sort records for optimal access patterns
    optimize_record_order(records);

    // Open output binary file
    std::ofstream output(output_bin, std::ios::binary);
    if (!output.is_open()) {
        SPDLOG_CRITICAL("Failed to open output binary file: {}", output_bin);
        for (auto* r : records) delete r;
        return false;
    }

    // Write header information
    uint64_t num_records = records.size();
    output.write(reinterpret_cast<const char*>(&num_records), sizeof(num_records));

    // Write index section
    build_spatial_index(output, records);

    // Write sorted records
    for (const auto* r : records) {
        output.write(reinterpret_cast<const char*>(&r->source), sizeof(r->source));
        output.write(reinterpret_cast<const char*>(&r->target), sizeof(r->target));
        output.write(reinterpret_cast<const char*>(&r->first_n), sizeof(r->first_n));
        output.write(reinterpret_cast<const char*>(&r->prev_n), sizeof(r->prev_n));
        output.write(reinterpret_cast<const char*>(&r->next_e), sizeof(r->next_e));
        output.write(reinterpret_cast<const char*>(&r->cost), sizeof(r->cost));
    }

    output.close();

    // Clean up
    for (auto* r : records) delete r;

    auto end_time = UTIL::get_current_time();
    double duration = UTIL::get_duration(start_time, end_time);

    SPDLOG_INFO("Indexed binary conversion completed in {} seconds", duration);

    return true;
}

void UBODTConverter::optimize_record_order(std::vector<Record*>& records) {
    SPDLOG_INFO("Optimizing record order for spatial locality...");

    // Sort by source node, then by target node
    std::sort(records.begin(), records.end(),
        [](const Record* a, const Record* b) {
            if (a->source != b->source) {
                return a->source < b->source;
            }
            return a->target < b->target;
        });

    SPDLOG_INFO("Records sorted by spatial locality");
}

void UBODTConverter::build_spatial_index(std::ofstream& output,
                                         const std::vector<Record*>& records) {
    SPDLOG_INFO("Building spatial index...");

    // Group records by source node
    std::unordered_map<NodeIndex, std::vector<size_t>> source_map;
    for (size_t i = 0; i < records.size(); ++i) {
        source_map[records[i]->source].push_back(i);
    }

    // Write index header
    uint64_t num_sources = source_map.size();
    output.write(reinterpret_cast<const char*>(&num_sources), sizeof(num_sources));

    // Write source index entries
    for (const auto& entry : source_map) {
        uint32_t source = entry.first;
        uint64_t start_offset = entry.second.front();
        uint64_t count = entry.second.size();

        output.write(reinterpret_cast<const char*>(&source), sizeof(source));
        output.write(reinterpret_cast<const char*>(&start_offset), sizeof(start_offset));
        output.write(reinterpret_cast<const char*>(&count), sizeof(count));
    }

    SPDLOG_INFO("Spatial index built with {} source nodes", num_sources);
}

bool UBODTConverter::compress_ubodt(const std::string &input_file,
                                    const std::string &output_file) {
    SPDLOG_INFO("Compressing UBODT file: {} -> {}", input_file, output_file);

    // This is a placeholder for future compression implementation
    // Could use zlib, lz4, or other compression libraries
    SPDLOG_WARN("UBODT compression not yet implemented");
    return false;
}

bool UBODTConverter::validate_ubodt(const std::string &filename) {
    SPDLOG_INFO("Validating UBODT file: {}", filename);

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        SPDLOG_ERROR("Cannot open file for validation: {}", filename);
        return false;
    }

    // Check if it's a binary file (indexed format)
    uint64_t num_records;
    if (file.read(reinterpret_cast<char*>(&num_records), sizeof(num_records))) {
        SPDLOG_INFO("Detected indexed binary format with {} records", num_records);
        return true;
    }

    // Check if it's a raw binary format
    file.clear();
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();

    if (file_size % sizeof(MmapRecord) == 0) {
        size_t record_count = file_size / sizeof(MmapRecord);
        SPDLOG_INFO("Detected raw binary format with {} records", record_count);
        return true;
    }

    // Assume CSV format
    file.close();
    std::ifstream csv_file(filename);
    if (!csv_file.is_open()) {
        SPDLOG_ERROR("Cannot reopen file as CSV: {}", filename);
        return false;
    }

    std::string line;
    int line_count = 0;
    int valid_records = 0;

    // Skip header
    std::getline(csv_file, line);
    line_count++;

    while (std::getline(csv_file, line)) {
        line_count++;
        Record r;
        int parsed = sscanf(line.c_str(), "%d;%d;%d;%d;%d;%lf",
                           &r.source, &r.target, &r.first_n,
                           &r.prev_n, &r.next_e, &r.cost);
        if (parsed == 6) {
            valid_records++;
        }
    }

    csv_file.close();

    SPDLOG_INFO("CSV validation: {} lines, {} valid records", line_count - 1, valid_records);

    return valid_records > 0;
}