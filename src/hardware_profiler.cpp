#include "minitorch/hardware_profiler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <intrin.h>

#elif defined(__linux__) || defined(__ANDROID__)

#include <sys/sysinfo.h>
#include <unistd.h>

#endif

namespace minitorch {

namespace {

using Clock = std::chrono::steady_clock;


/*
 * ============================================================
 * COMMON HELPERS
 * ============================================================
 */

static std::string trim(
    const std::string& value
) {
    const std::size_t begin =
        value.find_first_not_of(" \t\r\n");

    if (begin == std::string::npos) {
        return "";
    }

    const std::size_t end =
        value.find_last_not_of(" \t\r\n");

    return value.substr(
        begin,
        end - begin + 1
    );
}


/*
 * ============================================================
 * LINUX / ANDROID CPUINFO
 * ============================================================
 */

#if defined(__linux__) || defined(__ANDROID__)

static std::string read_cpuinfo_value(
    const std::string& key
) {
    std::ifstream file("/proc/cpuinfo");

    if (!file) {
        return "";
    }

    std::string line;

    while (std::getline(file, line)) {

        if (line.rfind(key, 0) == 0) {

            const std::size_t colon =
                line.find(':');

            if (colon != std::string::npos) {
                return trim(
                    line.substr(colon + 1)
                );
            }
        }
    }

    return "";
}


static double read_cpu_frequency() {

    const std::string value =
        read_cpuinfo_value("cpu MHz");

    if (value.empty()) {
        return 0.0;
    }

    try {
        return std::stod(value);
    } catch (...) {
        return 0.0;
    }
}

#endif


/*
 * ============================================================
 * WINDOWS CPU MODEL
 * ============================================================
 */

#if defined(_WIN32)

static std::string read_windows_cpu_model() {

    HKEY key = nullptr;

    const LONG result =
        RegOpenKeyExA(
            HKEY_LOCAL_MACHINE,
            "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
            0,
            KEY_READ,
            &key
        );

    if (result != ERROR_SUCCESS) {
        return "";
    }

    char buffer[512] = {};
    DWORD buffer_size =
        static_cast<DWORD>(sizeof(buffer));

    DWORD type = 0;

    const LONG query =
        RegQueryValueExA(
            key,
            "ProcessorNameString",
            nullptr,
            &type,
            reinterpret_cast<LPBYTE>(buffer),
            &buffer_size
        );

    RegCloseKey(key);

    if (
        query != ERROR_SUCCESS ||
        type != REG_SZ
    ) {
        return "";
    }

    return trim(
        std::string(buffer)
    );
}


static double read_windows_cpu_frequency() {

    HKEY key = nullptr;

    const LONG result =
        RegOpenKeyExA(
            HKEY_LOCAL_MACHINE,
            "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
            0,
            KEY_READ,
            &key
        );

    if (result != ERROR_SUCCESS) {
        return 0.0;
    }

    DWORD value = 0;
    DWORD value_size =
        static_cast<DWORD>(sizeof(value));

    DWORD type = 0;

    const LONG query =
        RegQueryValueExA(
            key,
            "~MHz",
            nullptr,
            &type,
            reinterpret_cast<LPBYTE>(&value),
            &value_size
        );

    RegCloseKey(key);

    if (
        query != ERROR_SUCCESS ||
        type != REG_DWORD
    ) {
        return 0.0;
    }

    return static_cast<double>(value);
}


static std::size_t read_windows_logical_processors() {

    const WORD groups =
        GetActiveProcessorGroupCount();

    if (groups == 0) {
        return 0;
    }

    std::size_t total = 0;

    for (
        WORD group = 0;
        group < groups;
        ++group
    ) {
        total +=
            static_cast<std::size_t>(
                GetActiveProcessorCount(group)
            );
    }

    return total;
}


static std::size_t read_windows_physical_cores() {

    DWORD length = 0;

    GetLogicalProcessorInformationEx(
        RelationProcessorCore,
        nullptr,
        &length
    );

    if (length == 0) {
        return 0;
    }

    std::vector<std::uint8_t> buffer(
        length
    );

    auto* info =
        reinterpret_cast<
            PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX
        >(
            buffer.data()
        );

    if (
        !GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            info,
            &length
        )
    ) {
        return 0;
    }

    std::size_t count = 0;
    DWORD offset = 0;

    while (offset < length) {

        auto* current =
            reinterpret_cast<
                PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX
            >(
                buffer.data() + offset
            );

        if (
            current->Relationship ==
            RelationProcessorCore
        ) {
            ++count;
        }

        if (current->Size == 0) {
            break;
        }

        offset += current->Size;
    }

    return count;
}


static std::string detect_windows_simd() {

    int cpu_info[4] = {};

    __cpuid(
        cpu_info,
        0
    );

    const int max_leaf =
        cpu_info[0];

    if (max_leaf < 1) {
        return "x86-64";
    }

    __cpuidex(
        cpu_info,
        1,
        0
    );

    const bool sse =
        (cpu_info[3] & (1 << 25)) != 0;

    const bool sse2 =
        (cpu_info[3] & (1 << 26)) != 0;

    const bool sse3 =
        (cpu_info[2] & (1 << 0)) != 0;

    const bool ssse3 =
        (cpu_info[2] & (1 << 9)) != 0;

    const bool sse41 =
        (cpu_info[2] & (1 << 19)) != 0;

    const bool sse42 =
        (cpu_info[2] & (1 << 20)) != 0;

    const bool avx =
        (cpu_info[2] & (1 << 28)) != 0;

    bool avx2 = false;
    bool avx512f = false;

    if (max_leaf >= 7) {

        __cpuidex(
            cpu_info,
            7,
            0
        );

        avx2 =
            (cpu_info[1] & (1 << 5)) != 0;

        avx512f =
            (cpu_info[1] & (1 << 16)) != 0;
    }

    if (avx512f) {
        return "SSE + SSE2 + AVX + AVX2 + AVX-512";
    }

    if (avx2) {
        return "SSE + SSE2 + AVX + AVX2";
    }

    if (avx) {
        return "SSE + SSE2 + AVX";
    }

    if (sse42) {
        return "SSE + SSE2 + SSE3 + SSSE3 + SSE4.1 + SSE4.2";
    }

    if (sse41) {
        return "SSE + SSE2 + SSE3 + SSSE3 + SSE4.1";
    }

    if (ssse3) {
        return "SSE + SSE2 + SSE3 + SSSE3";
    }

    if (sse3) {
        return "SSE + SSE2 + SSE3";
    }

    if (sse2) {
        return "SSE + SSE2";
    }

    if (sse) {
        return "SSE";
    }

    return "x86-64";
}


static std::size_t read_windows_total_memory_mb() {

    MEMORYSTATUSEX status = {};

    status.dwLength =
        sizeof(status);

    if (!GlobalMemoryStatusEx(&status)) {
        return 0;
    }

    return static_cast<std::size_t>(
        status.ullTotalPhys /
        (1024ULL * 1024ULL)
    );
}


static std::size_t read_windows_available_memory_mb() {

    MEMORYSTATUSEX status = {};

    status.dwLength =
        sizeof(status);

    if (!GlobalMemoryStatusEx(&status)) {
        return 0;
    }

    return static_cast<std::size_t>(
        status.ullAvailPhys /
        (1024ULL * 1024ULL)
    );
}

#endif


/*
 * ============================================================
 * SIMD DETECTION
 * ============================================================
 */

static std::string detect_simd() {

#if defined(_WIN32)

    return detect_windows_simd();

#elif defined(__aarch64__)

    std::ifstream file("/proc/cpuinfo");

    if (file) {

        std::string line;

        while (std::getline(file, line)) {

            if (
                line.find("Features") !=
                std::string::npos
            ) {

                if (
                    line.find("sve") !=
                    std::string::npos
                ) {
                    return "NEON + SVE";
                }

                if (
                    line.find("asimd") !=
                    std::string::npos ||
                    line.find("neon") !=
                    std::string::npos
                ) {
                    return "NEON";
                }
            }
        }
    }

    return "AArch64";

#elif defined(__x86_64__)

    std::ifstream file("/proc/cpuinfo");

    if (file) {

        std::string line;

        while (std::getline(file, line)) {

            if (
                line.find("flags") !=
                std::string::npos
            ) {

                if (
                    line.find("avx512") !=
                    std::string::npos
                ) {
                    return "SSE + AVX + AVX2 + AVX-512";
                }

                if (
                    line.find("avx2") !=
                    std::string::npos
                ) {
                    return "SSE + AVX + AVX2";
                }

                if (
                    line.find("avx") !=
                    std::string::npos
                ) {
                    return "SSE + AVX";
                }

                return "SSE";
            }
        }
    }

    return "x86-64";

#else

    return "Unknown";

#endif
}


/*
 * ============================================================
 * PHYSICAL CORES
 * ============================================================
 */

static std::size_t detect_physical_cores() {

#if defined(_WIN32)

    const std::size_t cores =
        read_windows_physical_cores();

    if (cores > 0) {
        return cores;
    }

#elif defined(__linux__) || defined(__ANDROID__)

    std::ifstream file("/proc/cpuinfo");

    if (file) {

        std::vector<std::string> physical_ids;

        std::string current_physical;
        std::string current_core;

        std::string line;

        while (std::getline(file, line)) {

            if (
                line.rfind(
                    "physical id",
                    0
                ) == 0
            ) {

                const std::size_t colon =
                    line.find(':');

                if (colon != std::string::npos) {
                    current_physical =
                        trim(
                            line.substr(
                                colon + 1
                            )
                        );
                }
            }

            if (
                line.rfind(
                    "core id",
                    0
                ) == 0
            ) {

                const std::size_t colon =
                    line.find(':');

                if (colon != std::string::npos) {
                    current_core =
                        trim(
                            line.substr(
                                colon + 1
                            )
                        );
                }
            }

            if (line.empty()) {

                if (
                    !current_physical.empty() &&
                    !current_core.empty()
                ) {

                    const std::string key =
                        current_physical +
                        ":" +
                        current_core;

                    if (
                        std::find(
                            physical_ids.begin(),
                            physical_ids.end(),
                            key
                        ) == physical_ids.end()
                    ) {
                        physical_ids.push_back(key);
                    }
                }

                current_physical.clear();
                current_core.clear();
            }
        }

        if (!physical_ids.empty()) {
            return physical_ids.size();
        }
    }

#endif

    const unsigned int threads =
        std::thread::hardware_concurrency();

    return threads == 0 ? 1 : threads;
}


/*
 * ============================================================
 * TOTAL MEMORY
 * ============================================================
 */

static std::size_t detect_total_memory_mb() {

#if defined(_WIN32)

    return read_windows_total_memory_mb();

#elif defined(__linux__) || defined(__ANDROID__)

    struct sysinfo info {};

    if (sysinfo(&info) == 0) {

        const std::uint64_t bytes =
            static_cast<std::uint64_t>(
                info.totalram
            ) *
            static_cast<std::uint64_t>(
                info.mem_unit
            );

        return static_cast<std::size_t>(
            bytes / (1024ULL * 1024ULL)
        );
    }

#endif

    return 0;
}


/*
 * ============================================================
 * AVAILABLE MEMORY
 * ============================================================
 */

static std::size_t detect_available_memory_mb() {

#if defined(_WIN32)

    return read_windows_available_memory_mb();

#elif defined(__linux__) || defined(__ANDROID__)

    struct sysinfo info {};

    if (sysinfo(&info) == 0) {

        const std::uint64_t bytes =
            static_cast<std::uint64_t>(
                info.freeram
            ) *
            static_cast<std::uint64_t>(
                info.mem_unit
            );

        return static_cast<std::size_t>(
            bytes / (1024ULL * 1024ULL)
        );
    }

#endif

    return 0;
}


/*
 * ============================================================
 * ELAPSED TIME
 * ============================================================
 */

static double elapsed_ms(
    const Clock::time_point& start,
    const Clock::time_point& end
) {
    return std::chrono::duration<
        double,
        std::milli
    >(
        end - start
    ).count();
}


/*
 * ============================================================
 * PREVENT OPTIMIZATION
 * ============================================================
 */

static void prevent_optimization(
    float value
) {
    volatile float sink = value;
    (void)sink;
}

} // namespace


/*
 * ============================================================
 * HARDWARE DETECTION
 * ============================================================
 */

HardwareInfo HardwareProfiler::detect() {

    HardwareInfo info;

#if defined(__aarch64__)

    info.architecture = "AArch64";

#elif defined(__x86_64__) || defined(_M_X64)

    info.architecture = "x86-64";

#elif defined(__arm__)

    info.architecture = "ARM";

#elif defined(__riscv)

    info.architecture = "RISC-V";

#else

    info.architecture = "Unknown";

#endif


#if defined(__ANDROID__)

    info.processor = "Android";

#elif defined(__linux__)

    info.processor = "Linux";

#elif defined(_WIN32)

    info.processor = "Windows";

#elif defined(__APPLE__)

    info.processor = "macOS";

#else

    info.processor = "Unknown OS";

#endif


/*
 * CPU MODEL
 */

#if defined(_WIN32)

    info.cpu_model =
        read_windows_cpu_model();

#else

    info.cpu_model =
        read_cpuinfo_value("model name");

    if (info.cpu_model.empty()) {
        info.cpu_model =
            read_cpuinfo_value("Hardware");
    }

    if (info.cpu_model.empty()) {
        info.cpu_model =
            read_cpuinfo_value("Processor");
    }

#endif

    if (info.cpu_model.empty()) {
        info.cpu_model = "Unknown CPU";
    }


/*
 * SIMD
 */

    info.simd =
        detect_simd();


/*
 * LOGICAL THREADS
 */

#if defined(_WIN32)

    info.logical_threads =
        read_windows_logical_processors();

#else

    info.logical_threads =
        std::thread::hardware_concurrency();

#endif

    if (info.logical_threads == 0) {
        info.logical_threads = 1;
    }


/*
 * PHYSICAL CORES
 */

    info.physical_cores =
        detect_physical_cores();


/*
 * MEMORY
 */

    info.memory_total_mb =
        detect_total_memory_mb();

    info.memory_available_mb =
        detect_available_memory_mb();


/*
 * CPU FREQUENCY
 */

#if defined(_WIN32)

    info.cpu_frequency_mhz =
        read_windows_cpu_frequency();

#elif defined(__linux__) || defined(__ANDROID__)

    info.cpu_frequency_mhz =
        read_cpu_frequency();

#endif

    return info;
}


/*
 * ============================================================
 * VECTOR ADD
 * ============================================================
 */

BenchmarkResult HardwareProfiler::benchmark_vector_add(
    std::size_t elements,
    std::size_t iterations
) {
    std::vector<float> a(
        elements,
        1.0f
    );

    std::vector<float> b(
        elements,
        2.0f
    );

    std::vector<float> output(
        elements
    );

    const auto start =
        Clock::now();

    for (
        std::size_t iteration = 0;
        iteration < iterations;
        ++iteration
    ) {

        for (
            std::size_t i = 0;
            i < elements;
            ++i
        ) {
            output[i] =
                a[i] + b[i];
        }
    }

    const auto end =
        Clock::now();

    prevent_optimization(
        output[elements - 1]
    );

    const double milliseconds =
        elapsed_ms(
            start,
            end
        );

    BenchmarkResult result;

    result.name =
        "Vector Add";

    result.elements =
        elements;

    result.milliseconds =
        milliseconds /
        static_cast<double>(
            iterations
        );

    result.operations =
        static_cast<double>(
            elements
        );

    result.gops =
        result.operations /
        (
            result.milliseconds *
            1.0e6
        );

    return result;
}


/*
 * ============================================================
 * VECTOR MULTIPLY
 * ============================================================
 */

BenchmarkResult HardwareProfiler::benchmark_vector_multiply(
    std::size_t elements,
    std::size_t iterations
) {
    std::vector<float> a(
        elements,
        1.5f
    );

    std::vector<float> b(
        elements,
        2.5f
    );

    std::vector<float> output(
        elements
    );

    const auto start =
        Clock::now();

    for (
        std::size_t iteration = 0;
        iteration < iterations;
        ++iteration
    ) {

        for (
            std::size_t i = 0;
            i < elements;
            ++i
        ) {
            output[i] =
                a[i] * b[i];
        }
    }

    const auto end =
        Clock::now();

    prevent_optimization(
        output[elements - 1]
    );

    const double milliseconds =
        elapsed_ms(
            start,
            end
        );

    BenchmarkResult result;

    result.name =
        "Vector Multiply";

    result.elements =
        elements;

    result.milliseconds =
        milliseconds /
        static_cast<double>(
            iterations
        );

    result.operations =
        static_cast<double>(
            elements
        );

    result.gops =
        result.operations /
        (
            result.milliseconds *
            1.0e6
        );

    return result;
}


/*
 * ============================================================
 * SCALAR MULTIPLY
 * ============================================================
 */

BenchmarkResult HardwareProfiler::benchmark_scalar_multiply(
    std::size_t elements,
    std::size_t iterations
) {
    std::vector<float> input(
        elements,
        2.0f
    );

    std::vector<float> output(
        elements
    );

    const float scalar =
        3.0f;

    const auto start =
        Clock::now();

    for (
        std::size_t iteration = 0;
        iteration < iterations;
        ++iteration
    ) {

        for (
            std::size_t i = 0;
            i < elements;
            ++i
        ) {
            output[i] =
                input[i] * scalar;
        }
    }

    const auto end =
        Clock::now();

    prevent_optimization(
        output[elements - 1]
    );

    const double milliseconds =
        elapsed_ms(
            start,
            end
        );

    BenchmarkResult result;

    result.name =
        "Scalar Multiply";

    result.elements =
        elements;

    result.milliseconds =
        milliseconds /
        static_cast<double>(
            iterations
        );

    result.operations =
        static_cast<double>(
            elements
        );

    result.gops =
        result.operations /
        (
            result.milliseconds *
            1.0e6
        );

    return result;
}


/*
 * ============================================================
 * MATRIX MULTIPLY
 * ============================================================
 */

BenchmarkResult HardwareProfiler::benchmark_matmul(
    std::size_t dimension,
    std::size_t iterations
) {
    const std::size_t size =
        dimension *
        dimension;

    std::vector<float> a(
        size,
        1.0f
    );

    std::vector<float> b(
        size,
        2.0f
    );

    std::vector<float> output(
        size,
        0.0f
    );

    const auto start =
        Clock::now();

    for (
        std::size_t iteration = 0;
        iteration < iterations;
        ++iteration
    ) {

        std::fill(
            output.begin(),
            output.end(),
            0.0f
        );

        for (
            std::size_t i = 0;
            i < dimension;
            ++i
        ) {

            for (
                std::size_t k = 0;
                k < dimension;
                ++k
            ) {

                const float value =
                    a[
                        i *
                        dimension +
                        k
                    ];

                for (
                    std::size_t j = 0;
                    j < dimension;
                    ++j
                ) {

                    output[
                        i *
                        dimension +
                        j
                    ] +=
                        value *
                        b[
                            k *
                            dimension +
                            j
                        ];
                }
            }
        }
    }

    const auto end =
        Clock::now();

    prevent_optimization(
        output[dimension - 1]
    );

    const double milliseconds =
        elapsed_ms(
            start,
            end
        );

    BenchmarkResult result;

    result.name =
        "Matrix Multiply";

    result.elements =
        dimension;

    result.milliseconds =
        milliseconds /
        static_cast<double>(
            iterations
        );

    result.operations =
        2.0 *
        static_cast<double>(
            dimension
        ) *
        static_cast<double>(
            dimension
        ) *
        static_cast<double>(
            dimension
        );

    result.gops =
        result.operations /
        (
            result.milliseconds *
            1.0e6
        );

    return result;
}


/*
 * ============================================================
 * PERFORMANCE SCORE
 * ============================================================
 */

double HardwareProfiler::calculate_score(
    const std::vector<BenchmarkResult>& results
) {
    if (results.empty()) {
        return 0.0;
    }

    double total = 0.0;

    for (const auto& result : results) {
        total += result.gops;
    }

    const double average =
        total /
        static_cast<double>(
            results.size()
        );

    return average;
}


/*
 * ============================================================
 * REPORT
 * ============================================================
 */

void HardwareProfiler::print_report(
    const HardwareInfo& info,
    const std::vector<BenchmarkResult>& results
) {
    std::cout
        << "========================================\n"
        << " CPUTorch V0.6.0 Hardware Profiler\n"
        << "========================================\n\n";

    std::cout
        << "===== HARDWARE =====\n";

    std::cout
        << "OS           : "
        << info.processor
        << '\n';

    std::cout
        << "Architecture : "
        << info.architecture
        << '\n';

    std::cout
        << "CPU          : "
        << info.cpu_model
        << '\n';

    std::cout
        << "Physical     : "
        << info.physical_cores
        << '\n';

    std::cout
        << "Logical      : "
        << info.logical_threads
        << '\n';

    std::cout
        << "SIMD         : "
        << info.simd
        << '\n';

    if (
        info.cpu_frequency_mhz >
        0.0
    ) {

        std::cout
            << "CPU MHz      : "
            << std::fixed
            << std::setprecision(2)
            << info.cpu_frequency_mhz
            << '\n';
    }

    std::cout
        << "RAM Total    : "
        << info.memory_total_mb
        << " MB\n";

    std::cout
        << "RAM Available: "
        << info.memory_available_mb
        << " MB\n\n";


    std::cout
        << "===== BENCHMARK =====\n";

    std::cout
        << std::left
        << std::setw(22)
        << "Operation"
        << std::right
        << std::setw(14)
        << "ms"
        << std::setw(14)
        << "GOPS"
        << '\n';

    std::cout
        << "----------------------------------------\n";


    for (const auto& result : results) {

        std::cout
            << std::left
            << std::setw(22)
            << result.name
            << std::right
            << std::setw(14)
            << std::fixed
            << std::setprecision(3)
            << result.milliseconds
            << std::setw(14)
            << std::setprecision(3)
            << result.gops
            << '\n';
    }


    std::cout
        << "\n========================================\n";

    std::cout
        << " Performance Score: "
        << std::fixed
        << std::setprecision(3)
        << calculate_score(results)
        << '\n';

    std::cout
        << "========================================\n";
}

} // namespace minitorch