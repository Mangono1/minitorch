#include "minitorch/hardware_profiler.h"

#include <iostream>
#include <vector>

int main() {

    try {

        const auto hardware =
            minitorch::HardwareProfiler::detect();

        std::vector<
            minitorch::BenchmarkResult
        > results;

        results.push_back(
            minitorch::HardwareProfiler::
                benchmark_vector_add(
                    1000000,
                    10
                )
        );

        results.push_back(
            minitorch::HardwareProfiler::
                benchmark_vector_multiply(
                    1000000,
                    10
                )
        );

        results.push_back(
            minitorch::HardwareProfiler::
                benchmark_scalar_multiply(
                    1000000,
                    10
                )
        );

        results.push_back(
            minitorch::HardwareProfiler::
                benchmark_matmul(
                    128,
                    2
                )
        );

        minitorch::HardwareProfiler::
            print_report(
                hardware,
                results
            );

        std::cout
            << "\nHardware profiler: PASS\n";

        return 0;

    } catch (const std::exception& error) {

        std::cerr
            << "Hardware profiler: FAIL\n"
            << error.what()
            << '\n';

        return 1;
    }
}
