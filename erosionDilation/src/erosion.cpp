#include "erosion.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <iostream>

cv::Mat erode(const cv::Mat &img, const int FILTER_SIZE, const std::string &SE_TYPE){
    const int FILTER_RADIUS = FILTER_SIZE/2;
    cv::Mat out_img = img.clone();
    
    for(int r = FILTER_RADIUS; r < img.rows - FILTER_RADIUS; ++r)
        for(int c = FILTER_RADIUS; c < img.cols - FILTER_RADIUS; ++c){
            bool is_black = false;
            if(SE_TYPE == "TRIANGLE"){
                for(int i = 0; i< FILTER_RADIUS; ++i)
                    for(int j = 0; j<= i; ++j)
                        is_black = ( is_black || (img.at<uchar>(r + i, c + j) == 0) ? 1 : 0 );
            }
            else{
                for(int i = -FILTER_RADIUS; i <= FILTER_RADIUS; ++i)
                    for(int j = -FILTER_RADIUS; j <= FILTER_RADIUS; ++j){
                        // check if the point is in SE_TYPE shape
                        bool to_consider = false;
                        if(SE_TYPE == "SQUARE")
                            to_consider = true;
                        else if(SE_TYPE == "CIRCLE"){
                            if(sqrtf(static_cast<float>(i*i + j*j)) <= FILTER_RADIUS)
                                to_consider = true;
                        }
                        else{
                            std::cerr<<"SPECIFY SE TYPE!!"<<std::endl;
                            exit(1);
                        }
                                
                        if(to_consider == false)
                            continue;
                        is_black = ( is_black || (img.at<uchar>(r + i, c + j) == 0) ? 1 : 0);
                    }
            }   
            out_img.at<uchar>(r, c) = (is_black) ? 0 : 255;
        }
    return out_img;
}