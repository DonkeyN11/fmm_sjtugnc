#include "mm/fmm/ubodt.hpp"
#include "mm/fmm/ubodt_mmap.hpp"
#include "util/util.hpp"
#include <iostream>
#include <chrono>
#include <memory>

using namespace FMM;
using namespace FMM::MM;

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <ubodt_file>\n";
        return 1;
    }

    std::string filename = argv[1];
    spdlog::set_pattern("[%^%l%$][%s:%-3#] %v");
    spdlog::set_level(spdlog::level::info);

    std::cout << "Testing UBODT loading performance for: " << filename << "\n";

    // Test regular CSV loading
    std::cout << "\n=== Testing CSV UBODT loading ===\n";
    auto start_csv = std::chrono::high_resolution_clock::now();

    try {
        auto ubodt_csv = UBODT::read_ubodt_csv(filename);
        auto end_csv = std::chrono::high_resolution_clock::now();
        auto duration_csv = std::chrono::duration_cast<std::chrono::milliseconds>(end_csv - start_csv);

        std::cout << "CSV UBODT loaded in " << duration_csv.count() << " ms\n";
        std::cout << "Number of records: " << ubodt_csv->get_num_rows() << "\n";

        // Test lookup performance
        std::cout << "Testing lookup performance...\n";
        auto lookup_start = std::chrono::high_resolution_clock::now();
        int found = 0;
        for (int i = 0; i < 1000; ++i) {
            auto result = ubodt_csv->look_up(i % 100, (i + 50) % 100);
            if (result != nullptr) found++;
        }
        auto lookup_end = std::chrono::high_resolution_clock::now();
        auto lookup_duration = std::chrono::duration_cast<std::chrono::microseconds>(lookup_end - lookup_start);
        std::cout << "1000 lookups completed in " << lookup_duration.count() << " μs (" << found << " found)\n";

    } catch (const std::exception &e) {
        std::cout << "CSV loading failed: " << e.what() << "\n";
    }

    // Test if file can be opened as memory-mapped
    std::cout << "\n=== Testing memory-mapped UBODT loading ===\n";
    try {
        auto start_mmap = std::chrono::high_resolution_clock::now();
        auto ubodt_mmap = make_mmap_ubodt(filename);
        auto end_mmap = std::chrono::high_resolution_clock::now();
        auto duration_mmap = std::chrono::duration_cast<std::chrono::milliseconds>(end_mmap - start_mmap);

        if (ubodt_mmap->is_valid()) {
            std::cout << "Memory-mapped UBODT loaded in " << duration_mmap.count() << " ms\n";
            std::cout << "Number of records: " << ubodt_mmap->get_num_records() << "\n";

            // Test lookup performance
            std::cout << "Testing lookup performance...\n";
            auto lookup_start = std::chrono::high_resolution_clock::now();
            int found = 0;
            for (int i = 0; i < 1000; ++i) {
                auto result = ubodt_mmap->look_up(i % 100, (i + 50) % 100);
                if (result != nullptr) found++;
            }
            auto lookup_end = std::chrono::high_resolution_clock::now();
            auto lookup_duration = std::chrono::duration_cast<std::chrono::microseconds>(lookup_end - lookup_start);
            std::cout << "1000 lookups completed in " << lookup_duration.count() << " μs (" << found << " found)\n";
        } else {
            std::cout << "Memory-mapped UBODT is not valid (not in correct format)\n";
        }
    } catch (const std::exception &e) {
        std::cout << "Memory-mapped loading failed: " << e.what() << "\n";
    }

    return 0;
}