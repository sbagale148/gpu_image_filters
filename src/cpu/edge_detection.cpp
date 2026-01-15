
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

cv::Mat applyKernel(const cv::Mat &src, const cv::Mat &kernel) {
  int kRows = kernel.rows;
  int kCols = kernel.cols;
  int kCenterY = kRows / 2;
  int kCenterX = kCols / 2;

  cv::Mat dst = cv::Mat::zeros(src.size(), CV_32F);

  for (int y = 0; y < src.rows; y++) {
    for (int x = 0; x < src.cols; x++) {
      float sum = 0.0f;

      for (int ky = 0; ky < kRows; ky++) {
        for (int kx = 0; kx < kCols; kx++) {
          int iy = y + (ky - kCenterY);
          int ix = x + (kx - kCenterX);

          // Handle edge cases by clamping coordinates
          iy = std::max(0, std::min(iy, src.rows - 1));
          ix = std::max(0, std::min(ix, src.cols - 1));

          sum += src.at<uchar>(iy, ix) * kernel.at<float>(ky, kx);
        }
      }

      dst.at<float>(y, x) = sum;
    }
  }

  return dst;
}

int main() {
  cv::Mat img = cv::imread("sample.jpg");
  if (img.empty()) {
    std::cerr << "Could not open or find the image\n";
    return 1;
  }

  cv::Mat gray;
  cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

  cv::Mat sobelX = (cv::Mat_<float>(3,3) << 
                    -1, 0, 1,
                    -2, 0, 2,
                    -1, 0, 1);
  
  cv::Mat sobelY = (cv::Mat_<float>(3, 3) << 
                    -1, -2, -1,
                    0, 0, 0,
                    1, 2, 1);

  cv::Mat gradX = applyKernel(gray, sobelX);
  cv::Mat gradY = applyKernel(gray, sobelY);

  cv::Mat magnitude = cv::Mat::zeros(gray.size(), CV_32F);
  for (int y = 0; y < gray.rows; y++) {
    for (int x = 0; x < gray.cols; x++) {
      float gx = gradX.at<float>(y, x);
      float gy = gradY.at<float>(y, x);
      magnitude.at<float>(y, x) = std::sqrt(gx * gx + gy * gy);
    }
  }

  cv::Mat edges;
  cv::normalize(magnitude, edges, 0, 255, cv::NORM_MINMAX, CV_8U);

  if (!cv::imwrite("Original.jpg", img)) {
    std::cerr << "Error: Could not write Original.jpg\n";
    return 1;
  }
  std::cout << "Saved image as Original.jpg\n";

  if (!cv::imwrite("Grayscale.jpg", gray)) {
    std::cerr << "Error: Could not write Grayscale.jpg\n";
    return 1;
  }
  std::cout << "Saved image as Grayscale.jpg\n";

  if (!cv::imwrite("Edge_Detection_Sobel.jpg", edges)) {
    std::cerr << "Error: Could not write Edge_Detection_Sobel.jpg\n";
    return 1;
  }
  std::cout << "Saved image as Edge_Detection_Sobel.jpg";

  return 0;
}
