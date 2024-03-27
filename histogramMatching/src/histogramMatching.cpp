#include "../include/histogramMatching.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

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

cv::Mat CDFPlot(float *cdf){
    int histWidth = 512;
    int histHeight = 400;
    int binWidth = cvRound((double)histWidth / 256);

    cv::Mat histImage(histHeight, histWidth, CV_8UC3, cv::Scalar(255, 255, 255));

    for (int i = 1; i < 256; ++i) {
        cv::line(histImage, cv::Point(binWidth * (i - 1), histHeight - cvRound(cdf[i - 1] * histHeight)),
                 cv::Point(binWidth * i, histHeight - cvRound(cdf[i] * histHeight)),
                 cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    return histImage;
}

cv::Mat histogramMatching(cv::Mat &img, cv::Mat &ref_img)
{

    int *ref_hist = computeHistogram(ref_img);
    float *ref_pdf = computePDF(ref_hist);

    free(ref_hist);

    return histogramMatching(img, ref_pdf);
}


cv::Mat histogramMatching(cv::Mat &img, float *ref_pdf)
{
    int *hist = computeHistogram(img);
    float *pdf = computePDF(hist);

    float *cdf = computeCDF(pdf);
    float *ref_cdf = computeCDF(ref_pdf);

    cv::Mat cdf_plot = CDFPlot(cdf);
    cv::Mat ref_cdf_plot = CDFPlot(ref_cdf);

    float *matched_cdf = (float *)malloc(256 * sizeof(float));

    cv::Mat outImg(img.rows, img.cols, CV_8UC1);

    for (int r = 0; r < img.rows; ++r)
        for (int c = 0; c < img.cols; ++c)
        {
            int pixIntensity = img.at<uchar>(r, c);
            float cdfInternsity = cdf[pixIntensity];
            matched_cdf[pixIntensity] = findNearestCDF(cdfInternsity, ref_cdf);
            outImg.at<uchar>(r, c) = matched_cdf[pixIntensity] * 255;
        }

    cv::Mat matched_cdf_plot = CDFPlot(ref_cdf);


    cv::imshow("Org Img CDF", cdf_plot);
    cv::imshow("Ref Img CDF", ref_cdf_plot);
    cv::imshow("Matched Img CDF", matched_cdf_plot);
        
    free(hist);
    free(pdf);
    free(cdf);
    free(ref_cdf);
    return outImg;
}
