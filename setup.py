from setuptools import setup, Extension
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

sources = [
    "python/cputorch/_core.cpp",
    "src/tensor.cpp",
    "src/device.cpp",
    "src/cpu/cpu_ops.cpp",
    "src/nn/parameter.cpp",
    "src/nn/module.cpp",
    "src/nn/linear.cpp",
    "src/nn/relu.cpp",
    "src/nn/mse_loss.cpp",
    "src/nn/sequential.cpp",
    "src/optim/sgd.cpp",
]

include_dirs = [
    str(ROOT / "include"),
]

extra_compile_args = [
    "-std=c++17",
    "-O2",
    "-Wall",
    "-Wextra",
]

if sys.platform == "win32":
    extra_compile_args = ["/std:c++17", "/O2"]

extension = Extension(
    "cputorch._core",
    sources=sources,
    include_dirs=include_dirs,
    language="c++",
    extra_compile_args=extra_compile_args,
)

setup(
    name="cputorch",
    version="0.5.7",
    description="A lightweight tensor and neural network framework with CPU and Vulkan backends",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=["cputorch"],
    package_dir={"": "python"},
    ext_modules=[extension],
    python_requires=">=3.9",
    author="Frandika Imam Arifin",
    license="MIT",
)
