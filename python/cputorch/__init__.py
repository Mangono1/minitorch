"""
CPUTorch
========

CPU-focused tensor and neural-network engine written in C++.

Version 0.5.7.
"""

from ._core import Tensor

__version__ = "0.5.7"

__all__ = [
    "Tensor",
    "__version__",
]
