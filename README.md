# MiniTorch

MiniTorch is an experimental neural-network framework built from scratch.

## Architecture

- C++ numerical core
- Python API
- CPU backend
- Future Vulkan backend
- Automatic differentiation
- Neural-network modules
- Optimizers
- Transformer support
- Hardware-aware execution
- GPU watchdog
- Adaptive fallback

## Current Version

V0.2.0

## V0.2 Features

- requires_grad
- Gradient storage
- Computational graph
- backward()
- Gradient accumulation
- zero_grad()
- Addition backward
- Subtraction backward
- Multiplication backward
- Matrix multiplication backward
- Sum backward

## Build on Termux

```bash
./scripts/build_termux.sh
```

## Roadmap

- V0.1 Tensor core
- V0.2 Automatic differentiation
- V0.3 Neural network modules
- V0.4 Optimizers
- V0.5 Hardware profiler
- V0.6 Vulkan compute backend
- V0.7 GPU watchdog and adaptive scheduler
- V0.8 Transformer
- V0.9 Tokenizer and TinyLLM
- V1.0 MiniTorch training engine
