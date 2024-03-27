#include<iostream>
#include<time.h>
#include<opencv2/opencv.hpp>
#define BLOCK_SIZE 32 // 32 * 32= 1024. 1024 max threads per block
__global__ void convert2Gray(unsigned char* img_in, unsigned char *img_out, int height, int width){
    int col= threadIdx.x + blockIdx.x*blockDim.x;
    int row= threadIdx.y + blockIdx.y*blockDim.y;

    if(col< width && row< height){
        int grayIdx= row*width + col;
        int rgbIdx= grayIdx*3;
        unsigned char r= img_in[rgbIdx    ];
        unsigned char g= img_in[rgbIdx + 1];
        unsigned char b= img_in[rgbIdx + 2];
        
        img_out[grayIdx]= 0.21f*r + 0.71f*g + 0.07f*b;
    }
    
}

int main(){
    cv::Mat img= cv::imread("./assets/flower.jpeg", cv::IMREAD_COLOR);
    cv::Size s= img.size();
    int height= s.height;
    int width= s.width;
    if(img.empty()){
        std::cout<<"\nERROR LOADING IMAGE.";
        return -1;
    }
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    unsigned char *img_h= img.data;

    unsigned char *img_in_d, *img_out_d;
    cudaMalloc((void **)&img_in_d, sizeof(unsigned char)*height*width*3);
    cudaMalloc((void **)&img_out_d, sizeof(unsigned char)*height*width);
    cudaMemcpy(img_in_d, img_h, sizeof(unsigned char)*height*width*3, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1)/BLOCK_SIZE, (height + BLOCK_SIZE - 1)/ BLOCK_SIZE);

    convert2Gray<<<grid, block>>>(img_in_d, img_out_d, height, width);
    cudaDeviceSynchronize();

    unsigned char *img_out_h= (unsigned char*)malloc(sizeof(char)*height*width);
    cudaMemcpy(img_out_h, img_out_d, sizeof(unsigned char)*height*width, cudaMemcpyDeviceToHost);

    cv::Mat grayImg= cv::Mat(height, width, CV_8UC1, (void *)img_out_h);
    cv::imwrite("flower_gray.jpg", grayImg);
    cv::imshow("IMG", grayImg);
    cv::waitKey(0);
    
    return 0;
}
