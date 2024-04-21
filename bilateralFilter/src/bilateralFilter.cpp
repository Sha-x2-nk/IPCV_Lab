#include "../include/bilateralFilter.hpp"
#include "../include/gaussian.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <iostream>
cv::Mat applyBilateralFilter(const cv::Mat &in_img, const int FILTER_SIZE, const double sigma_r, const double sigma_s){
    int height = in_img.rows;
    int width = in_img.cols;

    const int FILTER_RADIUS = FILTER_SIZE / 2;

    cv::Mat filtered_img(height, width, CV_8U);

    for(int y= 0; y < height; ++y){
        for(int x = 0; x < width; ++x){

            float tot_weight = 0;
            float pixel_sum = 0;
            for(int j = -FILTER_RADIUS; j <= FILTER_RADIUS; ++j){
                for(int i = -FILTER_RADIUS; i <= FILTER_RADIUS; ++i){
                    int newY = y + j;
                    int newX = x + i;

                    if(newY >= 0 && newY < height && newX >= 0 && newX < width){
                        float spatial_weight = gaussian(sqrt( (x - newX) * (x - newX) + 
                                                              (y - newY) * (y - newY)), sigma_s);

                        float intensity_weight = gaussian( abs(in_img.at<unsigned char>(y, x) - 
                                                            in_img.at<unsigned char>(newY, newX)),
                                                            sigma_r);
                        
                        float weight = spatial_weight * intensity_weight;
                        tot_weight += weight;
                        pixel_sum += in_img.at<unsigned char>(newY, newX) * weight;
                    }
                }
            }
            int filtered_pixel = std::min<int>(ceil(pixel_sum / tot_weight), 255);
            filtered_img.at<unsigned char>(y, x) = filtered_pixel;
        }
    }
    return filtered_img;
}