#include "erosion.hpp"
#include "dilation.hpp"
#include "opening.hpp"
#include "closing.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

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


    cv::Mat inp_binary;
    cv::threshold(inp_img, inp_binary, 128, 255, cv::THRESH_BINARY);
    std::vector<std::string> shapes;
    shapes.push_back("SQUARE");
    shapes.push_back("CIRCLE");
    shapes.push_back("TRIANGLE");
    
    for(auto &shape: shapes){
        cv::Mat eroded = erode(inp_binary, 9, shape);
        cv::Mat dilated = dilate(inp_binary, 9, shape);
    
        cv::Mat openedImg = openImg(inp_binary, 9, shape);
        cv::Mat closedImg = closeImg(inp_binary, 9, shape);

        int k;
        do{
            cv::imshow("org " + shape, inp_img);
            cv::imshow("binary " + shape, inp_binary);
            cv::imshow("eroded " + shape, eroded);
            cv::imshow("dilated " + shape, dilated);
            cv::absdiff(eroded, dilated, eroded);
            cv::imshow("eroded - dilated" + shape, eroded); 
            cv::imshow("Opened " + shape, openedImg);
            cv::imshow("closed " + shape, closedImg);

            k = cv::waitKey(0);
        }while(k!= 'Q' && k!= 'q');
        cv::destroyAllWindows();
    }

}