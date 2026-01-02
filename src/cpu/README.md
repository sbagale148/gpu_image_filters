# CPU Image Filters

CPU-based implementations of image processing filters.

## Programs

- **basics.cpp** - C++ fundamentals and exercises
- **threshold.cpp** - Threshold filter for binary image conversion
- **grayscale.cpp** - RGB to grayscale conversion
- **blur.cpp** - Box blur filter implementation
- **edge_detection.cpp** - Sobel edge detection filter

## Building

```bash
g++ -std=c++17 <program>.cpp -o <output> `pkg-config --cflags --libs opencv4`
```

## Requirements

- OpenCV library
- C++17 compatible compiler

