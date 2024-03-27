#include "include/histogramEqualisation.hpp"

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
    if(argc < 3){
        std::cout<<"\nUSAGE:";
        std::cout<<"\nmain.exe\t[IMAGE]\t[SALT PEPPER NOISE REMOVAL]";
        std::cout<<"\nmain.exe\texample.jpg\t1";
    }

    cv::Mat inp_img = cv::imread(args[1], cv::IMREAD_GRAYSCALE);
    

    // remove salt and pepper noise
    if(args[2][0] == '1')
        cv::medianBlur(inp_img, inp_img, 3);


    cv::Mat out_img = histogramEqualisation(inp_img);

    int k;
    do{
        cv::imshow("Org Img", inp_img);
        cv::imshow("Hist Equalised Img", out_img);
        
        k = cv::waitKey(0);
    }while(k!= 'q' && k!= 'Q');

    cv::destroyAllWindows();

    return 0;
}