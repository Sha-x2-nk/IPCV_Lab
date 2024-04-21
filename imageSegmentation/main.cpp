#include "include/erosion.hpp"
#include "include/thinning.hpp"
#include "include/medianBlur.hpp"

#include <opencv2/core.hpp> // for cv::Mat
#include <opencv2/highgui.hpp> // for cv::imshow, cv::waitKey
#include <opencv2/imgproc.hpp> // for medianblur

#include <iostream>
#include <vector>
#include <queue>

void onMouse(int event, int x, int y, int flags, void* userdata);
cv::Mat regionGrowing(const cv::Mat& src, cv::Point seed, int threshold);
cv::Mat findBoundary(const cv::Mat& img);

typedef unsigned char uchar;
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
        inp_img = medianBlur(inp_img, 3);

    imshow("Image", inp_img);

    cv::setMouseCallback("Image", onMouse, reinterpret_cast<void*>(&inp_img));

    cv::waitKey(0);

    return 0;
}

void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        cv::Mat& image = *static_cast<cv::Mat*>(userdata);
        int intensity = image.at<uchar>(y, x);
        std::cout << "Intensity at (" << x << ", " << y << "): " << intensity << std::endl;

        cv::Mat region = regionGrowing(image, cv::Point(x, y), 15); 
        cv::Mat boundary = findBoundary(region);

        cv::Mat thinnedBoundary = boundary.clone();
        thinning(thinnedBoundary);

        imshow("Region", region);
        imshow("Boundary", boundary);
        imshow("Thinned Boundary", thinnedBoundary);
    }
}

cv::Mat regionGrowing(const cv::Mat& src, cv::Point seed, int threshold) {
    cv::Mat region = cv::Mat::zeros(src.size(), CV_8UC1);
    std::queue<cv::Point> points;
    points.push(seed);
    uchar seedIntensity = src.at<uchar>(seed.y, seed.x);

    while (!points.empty()) {
        cv::Point pt = points.front(); points.pop();
        if (src.at<uchar>(pt.y, pt.x) >= seedIntensity - threshold && src.at<uchar>(pt.y, pt.x) <= seedIntensity + threshold)
            for (int dx = -1; dx <= 1; dx++)
                for (int dy = -1; dy <= 1; dy++) {
                    cv::Point neighbor = pt + cv::Point(dx, dy);
                    if (neighbor.x >= 0 && neighbor.y >= 0 && neighbor.x < src.cols && neighbor.y < src.rows) {
                        if (region.at<uchar>(neighbor.y, neighbor.x) == 0 && src.at<uchar>(neighbor.y, neighbor.x) >= seedIntensity - threshold && src.at<uchar>(neighbor.y, neighbor.x) <= seedIntensity + threshold) {
                            points.push(neighbor);
                            region.at<uchar>(neighbor.y, neighbor.x) = 255;
                        }
                    }
                }
    }

    return region;
}

cv::Mat findBoundary(const cv::Mat& img) {
    cv::Mat eroded_img;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    eroded_img = erode(img, 7, "SQUARE");
    return img - eroded_img;
}
