#include <opencv2/core.hpp>

#include <iostream>

cv::Mat applyBilateralFilter(const cv::Mat &in_img, const int FILTER_SIZE, const double simga_r, const double sigma_s);