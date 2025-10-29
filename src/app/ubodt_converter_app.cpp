//
// Simple UBODT converter demonstration
//

#include "mm/fmm/ubodt_converter.hpp"
#include "util/util.hpp"
#include "util/debug.hpp"

#include <iostream>
#include <string>

using namespace FMM;
using namespace FMM::MM;

void print_help() {
    std::cout << "UBODT Format Converter - Simple Version\n\n";
    std::cout << "Usage:\n";
    std::cout << "  ./build/ubodt_converter csv2mmap input.csv output.bin\n";
    std::cout << "  ./build/ubodt_converter csv2indexed input.csv output.bin\n";
    std::cout << "  ./build/ubodt_converter validate input_file\n\n";
    std::cout << "Operations:\n";
    std::cout << "  csv2mmap    - Convert CSV to memory-mappable binary\n";
    std::cout << "  csv2indexed - Convert CSV to indexed binary\n";
    std::cout << "  validate     - Validate UBODT file integrity\n";
}

int main(int argc, char **argv) {
    spdlog::set_pattern("[%^%l%$][%s:%-3#] %v");
    spdlog::set_level(spdlog::level::info);

    if (argc < 2) {
        print_help();
        return 1;
    }

    std::string operation = argv[1];

    try {
        if (operation == "csv2mmap" && argc == 4) {
            std::string input_file = argv[2];
            std::string output_file = argv[3];

            std::cout << "Converting CSV to memory-mapped binary format...\n";
            std::cout << "Input: " << input_file << "\n";
            std::cout << "Output: " << output_file << "\n";

            bool success = UBODTConverter::csv_to_mmap_binary(input_file, output_file);

            if (success) {
                std::cout << "Conversion completed successfully!\n";
                return 0;
            } else {
                std::cout << "Conversion failed!\n";
                return 1;
            }

        } else if (operation == "csv2indexed" && argc == 4) {
            std::string input_file = argv[2];
            std::string output_file = argv[3];

            std::cout << "Converting CSV to indexed binary format...\n";
            std::cout << "Input: " << input_file << "\n";
            std::cout << "Output: " << output_file << "\n";

            bool success = UBODTConverter::csv_to_indexed_binary(input_file, output_file);

            if (success) {
                std::cout << "Conversion completed successfully!\n";
                return 0;
            } else {
                std::cout << "Conversion failed!\n";
                return 1;
            }

        } else if (operation == "validate" && argc == 3) {
            std::string input_file = argv[2];

            std::cout << "Validating UBODT file: " << input_file << "\n";

            bool success = UBODTConverter::validate_ubodt(input_file);

            if (success) {
                std::cout << "UBODT file is valid!\n";
                return 0;
            } else {
                std::cout << "UBODT file is invalid!\n";
                return 1;
            }

        } else {
            print_help();
            return 1;
        }

    } catch (const std::exception &e) {
        SPDLOG_CRITICAL("Exception occurred: {}", e.what());
        std::cout << "Error: " << e.what() << "\n";
        return 1;
    }
}