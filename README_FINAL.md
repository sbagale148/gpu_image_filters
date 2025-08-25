# GPU Image Filters Project

This project explores image processing from CPU-based filters to GPU acceleration using CUDA. It demonstrates step-by-step implementations of basic image filters, convolution operations, and edge detection.

## Milestones

**M1: C++ Basics**  
- Exercises to strengthen loops, functions, and vector usage in C++.

**M2: 2D Array Threshold Filter**  
- Manipulated 2D arrays and applied threshold filters to images.

**M3: Grayscale Conversion**  
- Loaded real images and converted them to grayscale.
- Explored pixel intensity manipulation.

**M4: CPU Blur & Sobel Edge Detection**  
- Implemented 3x3 box blur filter on CPU.
- Implemented Sobel edge detection manually to highlight edges.
- Gained experience with convolution and handling image borders.

**M5: GPU Filters (CUDA)**  
- Implemented GPU-accelerated versions of the box blur and Sobel filters.
- Optimized image processing using CUDA kernels.
- Learned device memory management, threads, and blocks in CUDA.

## Tech Stack
- **C++** for CPU implementations
- **OpenCV** for image handling
- **CUDA** for GPU acceleration
- **Git & GitHub** for version control

## Key Learnings
- Image processing fundamentals: grayscale, blur, edge detection  
- CPU vs GPU computation: parallelism for speedup  
- Writing CUDA kernels and managing GPU memory  
- Handling real-world issues like borders, normalization, and data types

## How to Run
1. Clone the repository:
   git clone https://github.com/sbagale148/gpu_image_filters.git
2. Compile and run CPU versions using g++ and OpenCV.
3. Compile and run GPU versions using nvcc (requires CUDA-enabled GPU).

## Project Outcome
This project provides a practical understanding of image processing on both CPU and GPU, along with hands-on experience in C++, OpenCV, and CUDA.
