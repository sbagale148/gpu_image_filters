#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cmath>

__global__ void sobelKernel(const unsigned char* src, float* dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float Gx = 0.0f;
    float Gy = 0.0f;

    int kx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int ky[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int ix = x + j;
            int iy = y + i;

            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                float val = src[iy * width + ix];
                Gx += val * kx[i+1][j+1];
                Gy += val * ky[i+1][j+1];
            }
        }
    }

    dst[y * width + x] = sqrtf(Gx*Gx + Gy*Gy);
}

int main() {
    cv::Mat img = cv::imread("sample.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Could not open image\n";
        return 1;
    }

    int width = img.cols;
    int height = img.rows;
    size_t imgSize = width * height * sizeof(unsigned char);
    size_t outSize = width * height * sizeof(float);

    unsigned char *d_src;
    float *d_dst;
    cudaMalloc(&d_src, imgSize);
    cudaMalloc(&d_dst, outSize);
    cudaMemcpy(d_src, img.data, imgSize, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1)/block.x, (height + block.y - 1)/block.y);
    sobelKernel<<<grid, block>>>(d_src, d_dst, width, height);
    cudaDeviceSynchronize();

    cv::Mat grad_float(height, width, CV_32F);
    cudaMemcpy(grad_float.data, d_dst, outSize, cudaMemcpyDeviceToHost);

    cv::Mat grad;
    cv::normalize(grad_float, grad, 0, 255, cv::NORM_MINMAX, CV_8U);
    if (!cv::imwrite("Sobel_GPU.jpg", grad)) {
        std::cerr << "Could not write Sobel_GPU.jpg\n";
        return 1;
    }
    std::cout << "Saved GPU Sobel edge-detected image as Sobel_GPU.jpg\n";

    cudaFree(d_src);
    cudaFree(d_dst);

    return 0;
}
