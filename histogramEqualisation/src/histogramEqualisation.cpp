#include "../include/histogramEqualisation.hpp"

#include <opencv2/core.hpp>

#include <iostream>

// count frequencies of each pixel value
int *computeHistogram(cv::Mat &img)
{
    int *hist = (int *)malloc(256 * sizeof(int));
    memset(hist, 0, 256 * sizeof(int));

    for (int r = 0; r < img.rows; ++r)
        for (int c = 0; c < img.cols; ++c)
        {
            uchar intensity = img.at<uchar>(r, c);

            ++hist[intensity];
        }

    return hist;
}

// devides each histogram value by total pixels to get PDF(probablity density function)
float *computePDF(int *hist)
{
    uint tot_pixels = 0;

    for(int i= 0; i< 256;++i)
        tot_pixels += hist[i];

    
    float *pdf = (float *)malloc(256 * sizeof(float));

    // iterating through histogram
    for (int i = 0; i < 256; ++i)
        pdf[i] = ( static_cast<float>(hist[i]) ) / tot_pixels;

    return pdf;
}

// computes CDF(comulative density function) by the formula cdf[i] = cdf[i - 1] + pdf[i]. as expected ranges from [0, 1]
float *computeCDF(float *pdf)
{
    float *cdf = (float *)malloc(256 * sizeof(float));

    cdf[0] = pdf[0];

    for (int i = 1; i < 256; ++i)
        cdf[i] = cdf[i - 1] + pdf[i];

    return cdf;
}

// finds value closest to cdfVal in ref_cdf
float findNearestCDF(float cdf_val, float *ref_cdf)
{
    float ans = -1;
    float dist = INT_MAX;

    for (int i = 0; i < 256; ++i)
        if (dist > abs(ref_cdf[i] - cdf_val))
        {
            dist = abs(ref_cdf[i] - cdf_val);
            ans = ref_cdf[i];
        }

    return ans;
}

cv::Mat histogramEqualisation(cv::Mat &img)
{
    int *hist = computeHistogram(img);
    float *pdf = computePDF(hist);

    float *cdf = computeCDF(pdf);

    cv::Mat outImg(img.rows, img.cols, CV_8UC1);

    for (int r = 0; r < img.rows; ++r)
        for (int c = 0; c < img.cols; ++c)
        {
            int pixIntensity = img.at<uchar>(r, c);
            float cdfIntensity = cdf[pixIntensity];
            outImg.at<uchar>(r, c) = cdfIntensity * 255;
        }

    free(hist);
    free(pdf);
    free(cdf);
    return outImg;
}
