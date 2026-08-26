cmake_minimum_required(VERSION 3.18)

project(minitorch
    VERSION 0.5.7
    LANGUAGES CXX
)

# ============================================================
# CPUTorch V0.5.7
# CPU-first architecture
#
# Vulkan is OPTIONAL.
# The CPU core must build without Vulkan SDK.
# ============================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============================================================
# Options
# ============================================================

option(
    MINITORCH_ENABLE_VULKAN
    "Enable Vulkan backend when Vulkan SDK is available"
    ON
)

# ============================================================
# CPU CORE
# ============================================================

set(MINITORCH_CORE_SOURCES

    # Tensor / Device
    src/tensor.cpp
    src/device.cpp

    # CPU
    src/cpu/cpu_ops.cpp

    # Neural Network
    src/nn/parameter.cpp
    src/nn/module.cpp
    src/nn/linear.cpp
    src/nn/relu.cpp
    src/nn/mse_loss.cpp
    src/nn/sequential.cpp

    # Optimizer
    src/optim/sgd.cpp

    # Hardware profiler
    src/hardware_profiler.cpp
)

# ============================================================
# Optional Vulkan detection
# ============================================================

set(MINITORCH_VULKAN_ENABLED OFF)

if(MINITORCH_ENABLE_VULKAN)

    find_package(Vulkan QUIET)

    if(Vulkan_FOUND)

        message(STATUS "================================================")
        message(STATUS "CPUTorch Vulkan backend: ENABLED")
        message(STATUS "Vulkan include: ${Vulkan_INCLUDE_DIRS}")
        message(STATUS "Vulkan library: ${Vulkan_LIBRARIES}")
        message(STATUS "================================================")

        set(MINITORCH_VULKAN_ENABLED ON)

        list(APPEND MINITORCH_CORE_SOURCES

            src/vulkan/context.cpp
            src/vulkan/buffer.cpp
            src/vulkan/compute.cpp
            src/vulkan/pipeline.cpp
        )

    else()

        message(WARNING "================================================")
        message(WARNING "Vulkan SDK not found.")
        message(WARNING "CPUTorch will build in CPU-only mode.")
        message(WARNING "Install Vulkan SDK later to enable GPU backend.")
        message(WARNING "================================================")

    endif()

endif()

# ============================================================
# CORE LIBRARY
# ============================================================

add_library(
    minitorch_core
    STATIC
    ${MINITORCH_CORE_SOURCES}
)

target_include_directories(
    minitorch_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

# ============================================================
# Compiler warnings / optimization
# ============================================================

if(MSVC)

    target_compile_options(
        minitorch_core
        PRIVATE
            /O2
            /W4
            /EHsc
    )

elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")

    target_compile_options(
        minitorch_core
        PRIVATE
            -O2
            -Wall
            -Wextra
            -Wpedantic
)

endif()

# ============================================================
# Vulkan configuration
# ============================================================

if(MINITORCH_VULKAN_ENABLED)

    target_link_libraries(
        minitorch_core
        PUBLIC
            Vulkan::Vulkan
    )

    target_compile_definitions(
        minitorch_core
        PUBLIC
            MINITORCH_HAS_VULKAN=1
    )

else()

    target_compile_definitions(
        minitorch_core
        PUBLIC
            MINITORCH_HAS_VULKAN=0
    )

endif()

# ============================================================
# EXAMPLE HELPER
# ============================================================

function(add_minitortch_example TARGET SOURCE TEST_NAME)

    add_executable(
        ${TARGET}
        ${SOURCE}
    )

    target_link_libraries(
        ${TARGET}
        PRIVATE
            minitorch_core
    )

    target_include_directories(
        ${TARGET}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
    )

    add_test(
        NAME ${TEST_NAME}
        COMMAND ${TARGET}
    )

endfunction()

# ============================================================
# TESTING
# ============================================================

enable_testing()

# ============================================================
# CPU / Tensor examples
# ============================================================

add_minitortch_example(
    minitorch_basic
    examples/basic.cpp
    minitorch_basic_test
)

add_minitortch_example(
    minitorch_nn_basic
    examples/nn_basic.cpp
    minitorch_nn_basic_test
)

add_minitortch_example(
    minitorch_linear_basic
    examples/linear_basic.cpp
    minitorch_linear_basic_test
)

add_minitortch_example(
    minitorch_relu_basic
    examples/relu_basic.cpp
    minitorch_relu_basic_test
)

add_minitortch_example(
    minitorch_mse_basic
    examples/mse_basic.cpp
    minitorch_mse_basic_test
)

add_minitortch_example(
    minitorch_sgd_basic
    examples/sgd_basic.cpp
    minitorch_sgd_basic_test
)

add_minitortch_example(
    minitorch_training_basic
    examples/training_basic.cpp
    minitorch_training_basic_test
)

add_minitortch_example(
    minitorch_sequential_basic
    examples/sequential_basic.cpp
    minitorch_sequential_basic_test
)

add_minitortch_example(
    minitorch_mlp_training_basic
    examples/mlp_training_basic.cpp
    minitorch_mlp_training_basic_test
)

# ============================================================
# Hardware profiler
# ============================================================

add_minitortch_example(
    minitorch_hardware_profiler
    examples/hardware_profiler_basic.cpp
    minitorch_hardware_profiler_test
)

# ============================================================
# Vulkan examples
#
# Only compile these when Vulkan SDK exists.
# ============================================================

if(MINITORCH_VULKAN_ENABLED)

    add_executable(
        minitorch_vulkan_basic
        examples/vulkan_basic.cpp
    )

    target_link_libraries(
        minitorch_vulkan_basic
        PRIVATE
            minitorch_core
            Vulkan::Vulkan
    )

    target_include_directories(
        minitorch_vulkan_basic
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
    )

    add_test(
        NAME minitorch_vulkan_basic_test
        COMMAND minitorch_vulkan_basic
    )


    add_executable(
        minitorch_vulkan_buffer_basic
        examples/vulkan_buffer_basic.cpp
    )

    target_link_libraries(
        minitorch_vulkan_buffer_basic
        PRIVATE
            minitorch_core
            Vulkan::Vulkan
    )

    target_include_directories(
        minitorch_vulkan_buffer_basic
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
    )

    add_test(
        NAME minitorch_vulkan_buffer_basic_test
        COMMAND minitorch_vulkan_buffer_basic
    )


    add_executable(
        minitorch_vulkan_compute_basic
        examples/vulkan_compute_basic.cpp
    )

    target_link_libraries(
        minitorch_vulkan_compute_basic
        PRIVATE
            minitorch_core
            Vulkan::Vulkan
    )

    target_include_directories(
        minitorch_vulkan_compute_basic
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
    )

    add_test(
        NAME minitorch_vulkan_compute_basic_test
        COMMAND minitorch_vulkan_compute_basic
    )


    add_executable(
        minitorch_vulkan_shader_basic
        examples/vulkan_shader_basic.cpp
    )

    target_link_libraries(
        minitorch_vulkan_shader_basic
        PRIVATE
            minitorch_core
            Vulkan::Vulkan
    )

    target_include_directories(
        minitorch_vulkan_shader_basic
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
    )

    add_test(
        NAME minitorch_vulkan_shader_basic_test
        COMMAND minitorch_vulkan_shader_basic
    )

else()

    message(STATUS "Vulkan examples skipped because Vulkan SDK is unavailable.")

endif()

# ============================================================
# Build summary
# ============================================================

message(STATUS "")
message(STATUS "================================================")
message(STATUS "CPUTorch V0.5.7 BUILD CONFIGURATION")
message(STATUS "================================================")
message(STATUS "Project version : ${PROJECT_VERSION}")
message(STATUS "C++ standard    : C++17")
message(STATUS "Compiler        : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Vulkan enabled  : ${MINITORCH_VULKAN_ENABLED}")
message(STATUS "CPU backend     : ENABLED")
message(STATUS "NN backend      : ENABLED")
message(STATUS "Autograd        : ENABLED")
message(STATUS "Optimizer       : ENABLED")
message(STATUS "Profiler        : ENABLED")
message(STATUS "================================================")
message(STATUS "")
