# GPU Image Filters (CUDA)

GPU-accelerated image processing filters using CUDA.

## Programs

- **blur.cu** - GPU-accelerated box blur filter
- **sobel.cu** - GPU-accelerated Sobel edge detection

## Building

```bash
nvcc <program>.cu -o <output> `pkg-config --cflags --libs opencv4`
```

## Requirements

- CUDA toolkit
- OpenCV library
- NVIDIA GPU with CUDA support

## Performance

GPU implementations provide significant speedup for large images compared to CPU versions, especially for convolution-based operations like blur and edge detection.

