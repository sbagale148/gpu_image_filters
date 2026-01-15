#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

int main() {
  cv::Mat img = cv::imread("sample.jpg", cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cerr << "Error: could not load image\n";
    return 1;
  }

  int rows = img.rows;
  int cols = img.cols;

  std::vector<std::vector<int>> gray(rows, std::vector<int>(cols));

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      cv::Vec3b pixel = img.at<cv::Vec3b>(r,c);
      int grayVal = static_cast<int>(0.114*pixel[0] + 0.587*pixel[1] + 0.299*pixel[2]);
      gray[r][c] = grayVal;
    }
  }

  for (int r = 0; r < std::min(5, rows); r++) {
    for (int c = 0; c < std::min(5, cols); c++) {
      std::cout << gray[r][c] << " ";
    }
    std::cout << "\n";
  }

  cv::Mat grayMat(rows, cols, CV_8UC1);
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      grayMat.at<uchar>(r, c) = static_cast<uchar>(gray[r][c]);
    }
  }

  if (!cv::imwrite("M3_gray_saved.jpg", grayMat)) {
    std::cerr << "Error: could not save M3_gray_saved.jpg\n";
    return 1;
  }

  std::cout << "Saved grayscaled image as M3_gray_saved.jpg\n";
  return 0;
}
