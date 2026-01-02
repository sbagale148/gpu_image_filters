# GPU Image Filters

A collection of image processing filters implemented in C++ for CPU and CUDA for GPU acceleration. This project demonstrates various image processing techniques including grayscale conversion, blur filters, and edge detection, with both CPU and GPU implementations for performance comparison.

## Overview

GPU Image Filters provides implementations of common image processing operations, starting with basic CPU-based filters and progressing to GPU-accelerated versions using CUDA. The project showcases the performance benefits of GPU parallelism for computationally intensive image operations.

## Project Structure

```
gpu_image_filters/
├── src/
│   ├── cpu/          # CPU-based image filters
│   │   ├── README.md
│   │   ├── basics.cpp
│   │   ├── threshold.cpp
│   │   ├── grayscale.cpp
│   │   ├── blur.cpp
│   │   └── edge_detection.cpp
│   └── gpu/          # GPU-accelerated filters (CUDA)
│       ├── README.md
│       ├── blur.cu
│       └── sobel.cu
└── README.md
```

## Features

- **CPU Filters**: Basic image processing operations including threshold, grayscale conversion, blur, and Sobel edge detection
- **GPU Filters**: CUDA-accelerated versions of blur and edge detection filters
- **Performance Comparison**: Direct comparison between CPU and GPU implementations

## Prerequisites

- C++ compiler with C++17 support (g++, clang++, or MSVC)
- OpenCV library
- CUDA toolkit (for GPU programs)
- NVIDIA GPU with CUDA support (for GPU programs)

## Building

### CPU Programs

```bash
# Example: Compile blur filter
g++ -std=c++17 src/cpu/blur.cpp -o blur `pkg-config --cflags --libs opencv4`
```

### GPU Programs

```bash
# Example: Compile GPU blur
nvcc src/gpu/blur.cu -o gpu_blur `pkg-config --cflags --libs opencv4`
```

## Usage

Place a `sample.jpg` image in the project root, then run the compiled executables:

```bash
# CPU filters
./blur
./edge_detection

# GPU filters
./gpu_blur
./gpu_sobel
```

## Tech Stack

- **C++** - CPU implementations
- **OpenCV** - Image I/O and processing utilities
- **CUDA** - GPU acceleration

## License

MIT
