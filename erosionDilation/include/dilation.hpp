#include <opencv2/core.hpp>

#include <string>

cv::Mat dilate(const cv::Mat &img, const int FILTER_SIZE, const std::string &SE_TYPE);