#include<iostream>
#include<opencv2/opencv.hpp>
#define BLOCK_SIZE 32 // 32x32= 1024 - max per block
#define BLUR_SIZE 16 // 16*2 + 1 size k square ka filter
// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

__global__ void blurKernel(unsigned char* img_in, unsigned char* img_out, int img_height, int img_width){
    int row= threadIdx.y + blockDim.y * blockIdx.y;
    int col= threadIdx.x + blockDim.x * blockIdx.x;

    if(row< img_height and col< img_width){
        int pix_val[3]={0};
        int pix_num= 0;
        for(int rowShift= -BLUR_SIZE; rowShift<= BLUR_SIZE; ++rowShift){
            for(int colShift= -BLUR_SIZE; colShift<= BLUR_SIZE; ++colShift){
                int curRow= row + rowShift; 
                int curCol= col + colShift;
                if(curRow< img_height and curCol< img_width){
                    int curIdx= (img_width*curRow + curCol)*3;
                    for(int ch= 0; ch< 3; ++ch){
                        pix_val[ch]+= img_in[curIdx + ch];
                    }
                    ++pix_num;
                }
            }
        }
        int outIdx= (row*img_width + col)*3;
        for(int ch= 0; ch< 3; ++ch){
            pix_val[ch]/= pix_num;
            img_out[outIdx + ch]= pix_val[ch];
        }
    }
}

int main(){
    cv::Mat img= cv::imread("assets/flower.jpeg", cv::IMREAD_COLOR); // opencv BGR
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::Size img_sz= img.size();
    int img_height= img_sz.height, img_width= img_sz.width;
    int arr_size= sizeof(unsigned char)*img_height*img_width*3;

    unsigned char *img_in_h, *img_out_h, *img_in_d, *img_out_d;
    img_in_h= img.data;
    img_out_h= (unsigned char*)malloc(arr_size);
    

    // allocate cuda memory
    cudaMalloc((void **)&img_in_d, arr_size);
    cudaMalloc((void **)&img_out_d, arr_size);
    // cudaCheckErrors("CUDA ERROR WHILE ALLOCATION MEM.");

    cudaMemcpy(img_in_d, img_in_h, arr_size, cudaMemcpyHostToDevice);
    // cudaCheckErrors("CUDA ERROR COPYING FROM HOST TO DEVICE.");

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((img_width + BLOCK_SIZE - 1)/BLOCK_SIZE, (img_height + BLOCK_SIZE - 1)/BLOCK_SIZE);

    blurKernel<<<grid, block>>>(img_in_d, img_out_d, img_height, img_width);
    cudaDeviceSynchronize();
    
    cudaMemcpy(img_out_h, img_out_d, arr_size, cudaMemcpyDeviceToHost);
    // cudaCheckErrors("CUDA ERROR COPYING FROM DEVICE TO HOST.");

    cv::Mat img_res= cv::Mat(img_height, img_width, CV_8UC3, (void *)img_out_h);
    cv::imshow("IMG", img_res);
    cv::waitKey(0);
    
    
    return 0;
}