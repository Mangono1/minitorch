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
