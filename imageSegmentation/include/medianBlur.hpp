#include <opencv2/core.hpp>

uchar calculateMedian(std::vector<uchar> &vec);

cv::Mat medianBlur(const cv::Mat &img, const int FILTER_SIZE);