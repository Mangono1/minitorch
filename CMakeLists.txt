cmake_minimum_required(VERSION 3.18)

project(minitorch
    VERSION 0.3.1
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(minitorch_core STATIC
    src/tensor.cpp
    src/device.cpp
    src/cpu/cpu_ops.cpp
    src/nn/parameter.cpp
    src/nn/module.cpp
    src/nn/linear.cpp
    src/nn/relu.cpp
    src/nn/mse_loss.cpp
    src/nn/sequential.cpp
    src/optim/sgd.cpp
    src/vulkan/context.cpp
    src/vulkan/buffer.cpp
)

target_include_directories(minitorch_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(minitorch_core PRIVATE
        -O2
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

add_executable(minitorch_basic
    examples/basic.cpp
)

target_link_libraries(minitorch_basic
    PRIVATE
        minitorch_core
)

target_include_directories(minitorch_basic
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

enable_testing()

add_test(
    NAME minitorch_basic_test
    COMMAND minitorch_basic
)


add_executable(minitorch_nn_basic
    examples/nn_basic.cpp
)

target_link_libraries(minitorch_nn_basic
    PRIVATE
        minitorch_core
)

target_include_directories(minitorch_nn_basic
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

add_test(
    NAME minitorch_nn_basic_test
    COMMAND minitorch_nn_basic
)


add_executable(minitorch_linear_basic
    examples/linear_basic.cpp
)

target_link_libraries(minitorch_linear_basic
    PRIVATE
        minitorch_core
)

target_include_directories(minitorch_linear_basic
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

add_test(
    NAME minitorch_linear_basic_test
    COMMAND minitorch_linear_basic
)

add_executable(minitorch_relu_basic
    examples/relu_basic.cpp
)

target_link_libraries(minitorch_relu_basic
    PRIVATE
        minitorch_core
)

target_include_directories(minitorch_relu_basic
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

add_test(
    NAME minitorch_relu_basic_test
    COMMAND minitorch_relu_basic
)

add_executable(minitorch_mse_basic
    examples/mse_basic.cpp
)

target_link_libraries(minitorch_mse_basic
    PRIVATE
        minitorch_core
)

target_include_directories(minitorch_mse_basic
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

add_test(
    NAME minitorch_mse_basic_test
    COMMAND minitorch_mse_basic
)

add_executable(minitorch_sgd_basic
    examples/sgd_basic.cpp
)

target_link_libraries(minitorch_sgd_basic
    PRIVATE
        minitorch_core
)

target_include_directories(minitorch_sgd_basic
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

add_test(
    NAME minitorch_sgd_basic_test
    COMMAND minitorch_sgd_basic
)


add_executable(minitorch_training_basic
    examples/training_basic.cpp
)

target_link_libraries(minitorch_training_basic
    PRIVATE
        minitorch_core
)

target_include_directories(minitorch_training_basic
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

add_test(
    NAME minitorch_training_basic_test
    COMMAND minitorch_training_basic
)


add_executable(minitorch_sequential_basic
    examples/sequential_basic.cpp
)

target_link_libraries(minitorch_sequential_basic
    PRIVATE
        minitorch_core
)

target_include_directories(minitorch_sequential_basic
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

add_test(
    NAME minitorch_sequential_basic_test
    COMMAND minitorch_sequential_basic
)

add_executable(minitorch_mlp_training_basic
    examples/mlp_training_basic.cpp
)

target_link_libraries(minitorch_mlp_training_basic
    PRIVATE
        minitorch_core
)

target_include_directories(minitorch_mlp_training_basic
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

add_test(
    NAME minitorch_mlp_training_basic_test
    COMMAND minitorch_mlp_training_basic
)

add_executable(minitorch_vulkan_basic
    examples/vulkan_basic.cpp
)

target_link_libraries(minitorch_vulkan_basic
    PRIVATE
        minitorch_core
        vulkan
)

target_include_directories(minitorch_vulkan_basic
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

add_test(
    NAME minitorch_vulkan_basic_test
    COMMAND minitorch_vulkan_basic
)

add_executable(minitorch_vulkan_buffer_basic
    examples/vulkan_buffer_basic.cpp
)

target_link_libraries(minitorch_vulkan_buffer_basic
    PRIVATE
        minitorch_core
        vulkan
)

target_include_directories(minitorch_vulkan_buffer_basic
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

add_test(
    NAME minitorch_vulkan_buffer_basic_test
    COMMAND minitorch_vulkan_buffer_basic
)
