#include "../include/thinning.hpp"

#include <opencv2/core.hpp>

typedef unsigned char uchar;

// precomputing to avoid branch conditions, check thinningIterations
static uchar lut_zhang_iter0[] = {
    1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
    0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
    1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1,
    1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0,
    1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
    1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1};

static uchar lut_zhang_iter1[] = {
    1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
    0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
    0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0,
    1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1,
    0, 1, 1, 1};

void thinningIteration(cv::Mat& img, const int iter) {
    cv::Mat marker = cv::Mat::ones(img.size(), CV_8UC1);
    int rows = img.rows;
    int cols = img.cols;

    // for each runs in parallel
    marker.forEach<uchar>([&](uchar &value, const int *position){
        const int r = position[0];
        const int c = position[1];

        if(r == 0 || c == 0 || r == rows - 1 || c == cols - 1)
            return;

        uchar p2 = img.at<uchar>(r - 1, c);
        uchar p3 = img.at<uchar>(r - 1, c + 1);
        uchar p4 = img.at<uchar>(r, c + 1);
        uchar p5 = img.at<uchar>(r + 1, c + 1);
        uchar p6 = img.at<uchar>(r + 1, c);
        uchar p7 = img.at<uchar>(r + 1, c - 1);
        uchar p8 = img.at<uchar>(r, c - 1);
        uchar p9 = img.at<uchar>(r - 1, c - 1);

        int neighbours = p9 | (p2 << 1) | (p3 << 2) | (p4 << 3) | (p5 << 4) | (p6 << 5) | (p7 << 6) | (p8 << 7);

        if (iter == 0)
            value = lut_zhang_iter0[neighbours];
        else
            value = lut_zhang_iter1[neighbours];

        //int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
        //         (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
        //         (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
        //         (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
        //int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
        //int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
        //int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);
        //if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0) value = 0;
    });

    img &= marker; // We did not complement, above only we change from 0 to 1. (performance engineering)
}

void thinning(cv::Mat &img){
    img/=255;
    cv::Mat prev = cv::Mat::zeros(img.size(), CV_8UC1);
    cv::Mat diff;

    do{
        thinningIteration(img, 0);
        thinningIteration(img, 1);
        cv::absdiff(img, prev, diff);
        img.copyTo(prev);
    }while(cv::countNonZero(diff) > 0);
    img*= 255;
}
