"""
CPUTorch
========

CPU-focused tensor and neural-network engine written in C++.

Version 0.5.4.
"""

from ._core import Tensor

__version__ = "0.5.4"

__all__ = [
    "Tensor",
    "__version__",
]
