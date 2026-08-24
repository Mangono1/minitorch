cmake_minimum_required(VERSION 3.18)

project(minitorch
    VERSION 0.1.0
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
