#!/data/data/com.termux/files/usr/bin/bash

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

echo "======================================"
echo " MiniTorch V0.1 - Termux Build"
echo "======================================"
echo
echo "Project : $PROJECT_ROOT"
echo "Compiler:"
clang++ --version | head -n 1
echo

mkdir -p "$BUILD_DIR"

cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER=clang

cmake --build . -j2

echo
echo "======================================"
echo " Build completed"
echo "======================================"
echo

./minitorch_basic

echo
echo "Running CTest..."
ctest --output-on-failure
