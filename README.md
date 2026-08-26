# CPUTorch

CPUTorch is a lightweight tensor and neural-network framework built from scratch, with a C++ numerical core, Python API, CPU execution, and Vulkan compute support.

The project is designed as a general-purpose computational foundation for machine learning and artificial intelligence systems.

## Author

**Frandika Imam Arifin**

## Repository

https://github.com/Mangono1/minitorch

## Architecture

- C++ numerical core
- Python API
- CPU backend
- Vulkan compute backend
- Automatic differentiation
- Tensor operations
- Neural-network modules
- Optimizers
- Hardware profiling
- Hardware-aware execution
- Adaptive execution architecture

## Current Version

**CPUTorch V0.5.3**

## V0.5.3

Current development includes:

- Tensor engine
- requires_grad
- Computational graph infrastructure
- Automatic differentiation infrastructure
- Neural-network modules
- Linear layers
- ReLU
- Sequential models
- MSE loss
- SGD optimizer
- Hardware profiler
- CPU backend
- Vulkan compute infrastructure
- Vulkan buffer management
- Vulkan compute pipeline
- Vector arithmetic backend operations
- Scalar multiplication operation
- Python package
- PyPI distribution
- Multi-platform wheel build automation

## Installation

Install the released package directly from PyPI:

    python -m pip install cputorch

Or install a specific version:

    python -m pip install cputorch==0.5.3

## Quick Example

    import cputorch

    x = cputorch.Tensor(
        [1.0, 2.0, 3.0, 4.0],
        [2, 2],
        requires_grad=True
    )

    y = x.multiply_scalar(2.0)

    print(y.data)

## Design Philosophy

CPUTorch is intended to remain a general computational framework.

It is not tied to a specific domain or dataset.

It can serve as a foundation for:

- Machine learning
- Deep learning
- Computer vision
- Scientific computing
- Robotics
- Simulation
- Reinforcement learning
- Custom AI systems
- Domain-specific AI frameworks

Higher-level frameworks and models can use CPUTorch as their computational backend.

## Roadmap

- V0.1 Tensor core
- V0.2 Automatic differentiation
- V0.3 Neural-network modules
- V0.4 Optimizers
- V0.5 Hardware profiler
- V0.5.x Vulkan compute development
- V0.6 Expanded Vulkan backend
- V0.7 GPU watchdog and adaptive scheduler
- V0.8 Transformer architecture
- V0.9 Tokenizer and language-model infrastructure
- V1.0 General training engine

## License

MIT License.
