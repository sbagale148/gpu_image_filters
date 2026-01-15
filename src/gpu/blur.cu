#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

__global__ void blurKernel(const unsigned char* src, float* dst, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  int kernel = 15;
  float sum = 0.0f;
  int count = 0;

  for (int ky = - kernel/2; ky <= kernel/2; ky++) {
    for (int kx = -kernel/2; kx <= kernel/2; kx++) {
      int ix = x + kx;
      int iy = y + ky;

      if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
        sum += src[iy * width + ix];
        count++;
      }
    }
  }

  dst[y * width + x] = sum / count;
}

int main() {
  cv::Mat img = cv::imread("sample.jpg", cv::IMREAD_GRAYSCALE);
  if (img.empty()) {
    std::cerr << "Could not open image\n";
    return 1;
  }

  int width = img.cols;
  int height = img.rows;
  size_t imgSizeUChar = width * height * sizeof(unsigned char);
  size_t imgSizeFloat = width * height * sizeof(float);

  unsigned char *d_src;
  float *d_dst;
  cudaError_t err;
  
  err = cudaMalloc(&d_src, imgSizeUChar);
  if (err != cudaSuccess) {
    std::cerr << "CUDA malloc error for source: " << cudaGetErrorString(err) << "\n";
    return 1;
  }
  
  err = cudaMalloc(&d_dst, imgSizeFloat);
  if (err != cudaSuccess) {
    std::cerr << "CUDA malloc error for destination: " << cudaGetErrorString(err) << "\n";
    cudaFree(d_src);
    return 1;
  }

  err = cudaMemcpy(d_src, img.data, imgSizeUChar, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "CUDA memcpy error (H2D): " << cudaGetErrorString(err) << "\n";
    cudaFree(d_src);
    cudaFree(d_dst);
    return 1;
  }

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1)/block.x, (height + block.y - 1)/block.y);

  blurKernel<<<grid, block>>>(d_src, d_dst, width, height);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << "\n";
    cudaFree(d_src);
    cudaFree(d_dst);
    return 1;
  }

  cv::Mat blurred_float(height, width, CV_32F);
  err = cudaMemcpy(blurred_float.data, d_dst, imgSizeFloat, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "CUDA memcpy error (D2H): " << cudaGetErrorString(err) << "\n";
    cudaFree(d_src);
    cudaFree(d_dst);
    return 1;
  }

  cv::Mat blurred;
  cv::normalize(blurred_float, blurred, 0, 255, cv::NORM_MINMAX, CV_8U);
  
  if (!cv::imwrite("Blurred_GPU.jpg", blurred)) {
    std::cerr << "Could not write Blurred_GPU.jpg\n";
    return 1;
  }
  std::cout << "Saved GPU-blurred image as Blurred_GPU.jpg\n";

  cudaFree(d_src);
  cudaFree(d_dst);

  return 0; 
}
