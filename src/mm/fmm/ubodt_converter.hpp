/**
 * Fast map matching.
 *
 * UBODT format converter for optimization
 *
 * @author: Ning
 * @version: 2024.09.16
 */

#ifndef FMM_SRC_MM_FMM_UBODT_CONVERTER_HPP_
#define FMM_SRC_MM_FMM_UBODT_CONVERTER_HPP_

#include "mm/fmm/ubodt.hpp"
#include "network/type.hpp"
#include "util/debug.hpp"

namespace FMM {
namespace MM {

/**
 * UBODT format converter for optimizing performance
 */
class UBODTConverter {
public:
    /**
     * Convert CSV UBODT to memory-mappable binary format
     * @param input_csv Path to input CSV file
     * @param output_bin Path to output binary file
     * @param progress_interval Progress reporting interval (number of records)
     * @return True if conversion successful
     */
    static bool csv_to_mmap_binary(const std::string &input_csv,
                                   const std::string &output_bin,
                                   int progress_interval = 1000000);

    /**
     * Create indexed binary format for even faster access
     * @param input_csv Path to input CSV file
     * @param output_bin Path to output indexed binary file
     * @param progress_interval Progress reporting interval
     * @return True if conversion successful
     */
    static bool csv_to_indexed_binary(const std::string &input_csv,
                                     const std::string &output_bin,
                                     int progress_interval = 1000000);

    /**
     * Compress existing UBODT file
     * @param input_file Input UBODT file (CSV or binary)
     * @param output_file Output compressed file
     * @return True if compression successful
     */
    static bool compress_ubodt(const std::string &input_file,
                              const std::string &output_file);

    /**
     * Validate UBODT file integrity
     * @param filename Path to UBODT file
     * @return True if file is valid
     */
    static bool validate_ubodt(const std::string &filename);

private:
    /**
     * Sort records for optimal memory access patterns
     * @param records Vector of records to sort
     */
    static void optimize_record_order(std::vector<Record*>& records);

    /**
     * Build spatial index for binary file
     * @param output_file Output file stream
     * @param records Vector of records
     */
    static void build_spatial_index(std::ofstream& output_file,
                                   const std::vector<Record*>& records);
};

} // MM
} // FMM

#endif //FMM_SRC_MM_FMM_UBODT_CONVERTER_HPP_