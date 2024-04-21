#include "include/bilateralFilter.hpp"


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

int main(int argc, char *args[]){
    /*
        args[0] = program
        args[1] = img_path
        args[2] = apply median filter or not [1|0]
    */
    if(argc < 5){
        std::cout<<"\nUSAGE:";
        std::cout<<"\nmain.exe\t[IMAGE]\t[SALT PEPPER NOISE REMOVAL]\tSIGMA_R\tSIGMA_S";
        std::cout<<"\nmain.exe\texample.jpg\t1\t75\t75";
    }

    cv::Mat inp_img = cv::imread(args[1], cv::IMREAD_GRAYSCALE);
    

    // remove salt and pepper noise
    if(args[2][0] == '1')
        cv::medianBlur(inp_img, inp_img, 3);


    double sigma_r = std::atoi(args[3]);
    double sigma_s = std::atoi(args[4]);

    cv::Mat bilateral_filter_custom = applyBilateralFilter(inp_img, 9, sigma_r, sigma_r);
    cv::Mat bilateral_filter_opencv;
    cv::bilateralFilter(inp_img, bilateral_filter_opencv, 9, sigma_r, sigma_s, cv::BORDER_CONSTANT|0);

    int tot_diff = 0;
    int max_diff = 0;
    for(int y= 0; y< bilateral_filter_custom.rows; ++y)
        for(int x= 0; x< bilateral_filter_custom.cols; ++x){
            tot_diff += abs(bilateral_filter_custom.at<unsigned char>(y, x) - bilateral_filter_opencv.at<unsigned char>(y, x));
            max_diff = std::max<int>(max_diff, bilateral_filter_custom.at<unsigned char>(y, x) - bilateral_filter_opencv.at<unsigned char>(y, x));
        }

    std::cout<<"TOTAL DIFFERENCE BETWEEN OPENCV AND CUSTOM BILATERAL FILTER: "<<tot_diff<<std::endl;
    std::cout<<"MAX DIFFERENCE BETWEEN OPENCV AND CUSTOM BILATERAL FILTER: "<<max_diff<<std::endl;

    int k;
    do{
        cv::imshow("Org Img", inp_img);
        cv::imshow(std::string("Bilateral Custom sigma_r: ") + std::to_string(sigma_r) + std::string(" sigma_s: ") + std::to_string(sigma_s), bilateral_filter_custom);
        cv::imshow(std::string("Bilateral opencv sigma_s: ") + std::to_string(sigma_r) + std::string(" sigma_s: ") + std::to_string(sigma_s), bilateral_filter_opencv);

        k = cv::waitKey(0);
    }while(k!= 'q' && k!= 'Q');

    return 0;
}