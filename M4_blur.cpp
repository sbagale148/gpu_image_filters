#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

static inline int clampi(int v, int lo, int hi) {
  return std::max(lo, std::min(v, hi));
}

int main() {
  const std::string path = "sample.jpg";
  cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cerr << "Error: could not load " << path << "\n";
    return 1;
  }

  cv::Mat gray(img.rows, img.cols, CV_8UC1);
  for (int r = 0; r < img.rows; r++) {
    for (int c = 0; c < img.cols; c++) {
      auto px = img.at<cv::Vec3b>(r, c);
      int B = px[0], G = px[1], R = px[2];
      int Y = static_cast<int>(0.114 * B + 0.587 * G + 0.299 * R);
      gray.at<uchar>(r, c) = static_cast<uchar>(Y);
    }
  }

  cv::Mat blur(gray.rows, gray.cols, CV_8UC1);

  for (int r = 0; r < gray.rows; r++) {
    for (int c = 0; c < gray.cols; c++) {

      int kernel = 15;
      int sum = 0;

      for (int dr = -kernel/2; dr <= kernel/2; dr++) {
        for (int dc = -kernel/2; dc <= kernel/2; dc++) {
          int rr = clampi(r + dr, 0, gray.rows - 1);
          int cc = clampi(c + dc, 0, gray.cols - 1);
          sum += gray.at<uchar>(rr, cc);
        }
      }

      int avg = sum / (kernel * kernel);
      blur.at<uchar>(r, c) = avg;
    }
  }

  if (!cv::imwrite("M4_blur.jpg", blur)) {
    std::cerr << "Error: could not save M4_blur.jpg\n";
    return 1;
  }

  std::cout << "Saved M4_blur.jpg\n";
  return 0;
}
