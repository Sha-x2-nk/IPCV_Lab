#include "opening.hpp"
#include "erosion.hpp"
#include "dilation.hpp"

#include <opencv2/core.hpp>

#include <string>

cv::Mat openImg(const cv::Mat &img, const int FILTER_SIZE, const std::string &SE_TYPE){
    // first erosion, then dilation
    cv::Mat eroded = erode(img, FILTER_SIZE, SE_TYPE);
    cv::Mat dilated = dilate(eroded, FILTER_SIZE, SE_TYPE);
    return dilated;
}