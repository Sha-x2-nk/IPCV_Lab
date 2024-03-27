#ifndef HIST_MATCH
#define HIST_MATCH

#include <opencv2/core.hpp>

#include <iostream>

typedef unsigned char uchar;
typedef unsigned int uint;

int *computeHistogram(cv::Mat &img);

float *computePDF(int *hist);

float *computeCDF(float *cdf);

cv::Mat histogramEqualisation(cv::Mat &img);

#endif