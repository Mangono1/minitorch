#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace minitorch {

struct HardwareInfo {
    std::string architecture;
    std::string processor;
    std::string cpu_model;
    std::string simd;
    std::size_t physical_cores = 0;
    std::size_t logical_threads = 0;

    std::size_t memory_total_mb = 0;
    std::size_t memory_available_mb = 0;

    double cpu_frequency_mhz = 0.0;
};

struct BenchmarkResult {
    std::string name;
    std::size_t elements = 0;
    double milliseconds = 0.0;
    double operations = 0.0;
    double gops = 0.0;
};

class HardwareProfiler {
public:
    static HardwareInfo detect();

    static BenchmarkResult benchmark_vector_add(
        std::size_t elements = 1000000,
        std::size_t iterations = 20
    );

    static BenchmarkResult benchmark_vector_multiply(
        std::size_t elements = 1000000,
        std::size_t iterations = 20
    );

    static BenchmarkResult benchmark_scalar_multiply(
        std::size_t elements = 1000000,
        std::size_t iterations = 20
    );

    static BenchmarkResult benchmark_matmul(
        std::size_t dimension = 128,
        std::size_t iterations = 3
    );

    static double calculate_score(
        const std::vector<BenchmarkResult>& results
    );

    static void print_report(
        const HardwareInfo& info,
        const std::vector<BenchmarkResult>& results
    );
};

} // namespace minitorch
