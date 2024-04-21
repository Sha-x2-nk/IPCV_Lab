#include "../include/medianBlur.hpp"

#include <opencv2/core.hpp>

#include <algorithm>

typedef unsigned char uchar;

uchar calculateMedian(std::vector<uchar> &vec){
    std::sort(vec.begin(), vec.end());

    int size = vec.size();
    if (size % 2 == 0)
        return (vec[size / 2 - 1] + vec[size / 2]) / 2;
    else
        return vec[size / 2];
}

cv::Mat medianBlur(const cv::Mat &img, const int FILTER_SIZE){
    int rows = img.rows;
    int cols = img.cols;
    const int FILTER_RADIUS = FILTER_SIZE/2;

    cv::Mat out_img = cv::Mat::zeros(img.size(), CV_8UC1);

    out_img.forEach<uchar>([&](uchar &value, const int *positions){
                                const int r = positions[0];
                                const int c = positions[1];
                                std::vector<uchar> pixels;
                                
                                for(int i = -FILTER_RADIUS; i<= FILTER_RADIUS; ++i){
                                    for(int j= -FILTER_RADIUS; j<= FILTER_RADIUS; ++j){
                                        if(i + r >= 0 && i + r < rows && j + c >= 0 && j + c < cols)
                                            pixels.push_back(img.at<uchar>(i + r, j + c));
                                    }
                                }
                                value = calculateMedian(pixels);
                            });
    return out_img;
}
