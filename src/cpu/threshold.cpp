#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

//Threshold function
void threshold(std::vector<std::vector<int>>& img, int t) {
  for (size_t r = 0; r < img.size(); r++) {
    for (size_t c = 0; c < img[r].size(); c++) {
      if (img[r][c] < t) {
        img[r][c] = 0;
      } else {
        img[r][c] = 255;
      }
    }
  }
}
int main() {
  //Seed random number generator so random numbers are different every run
  std::srand(std::time(nullptr));

  int rows = 5, cols = 5;
  std::vector<std::vector<int>> img(rows, std::vector<int>(cols));

  //Fill with random pixels (0..255)
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      img[r][c] = rand() % 256;
    }
  }

  //Print original image
  std::cout << "Original Image:\n";
  for (auto &row : img) {
    for (auto &val : row) {
      std::cout << val << " ";
    }
    std::cout << "\n";
  }

  //Get threshold from user
  std::cout << "Enter threshold value: ";
  int t;
  std::cin >> t;

  //Apply threshold
  threshold(img, t);

  //Print thresholded image
  std::cout << "Thresholded Image:\n";
  for (auto &row : img) {
    for (auto &val : row) {
      std::cout << val << " ";
    }
    std::cout << "\n";
  }
  return 0;
}
