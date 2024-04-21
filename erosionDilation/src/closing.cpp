#include "closing.hpp"
#include "erosion.hpp"
#include "dilation.hpp"

#include <opencv2/core.hpp>

#include <string>

cv::Mat closeImg(const cv::Mat &img, const int FILTER_SIZE, const std::string &SE_TYPE){
    // first dilation, then erosion
    cv::Mat dilated = dilate(img, FILTER_SIZE, SE_TYPE);
    cv::Mat eroded = erode(dilated, FILTER_SIZE, SE_TYPE);
    return eroded;
}