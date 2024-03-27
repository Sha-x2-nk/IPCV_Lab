#ifndef HIST_MATCH
#define HIST_MATCH


#include <opencv2/core.hpp>

#include <iostream>

typedef unsigned char uchar;
typedef unsigned int uint;

int *computeHistogram(cv::Mat &img);

float *computePDF(int *hist);

float *computeCDF(float *pdf);

float findNearestCDF(float cdfVal, float *ref_cdf);

cv::Mat CDFPlot(float *cdf);

cv::Mat histogramMatching(cv::Mat &img, cv::Mat &ref_img);

cv::Mat histogramMatching(cv::Mat &img, float *ref_pdf);

#endif