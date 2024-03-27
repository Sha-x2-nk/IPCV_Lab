#include "include/histogramMatching.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <string.h>
#include <stddef.h>
#include <fstream>
#include <sstream>

inline bool isTxt(char *path){
    size_t len = strlen(path);

    if(len>= 4 && strcmp(path + len - 4, ".txt") == 0) 
        return 1;

    return 0;
}

int main(int argc, char *args[]){
    /*
        args[0] = program name
        args[1] = img1 path
        args[2] = SALT PEPPER NOISE REMOVAL [0|1]
        args[3] = img2 path / .txt file containing desired pdf
        args[4] = SALT PEPPER NOISE REMOVAL [0|1] (if args[3] is img)
    */
     if(argc < 4 || (isTxt(args[3]) == false && argc < 5)){
        std::cout<<"\nUSAGE:";
        std::cout<<"\nmain.exe\t[IMAGE]\t[SALT PEPPER NOISE REMOVAL]\t[IMAGE2]\t[SALT PEPPER NOISE REMOVAL]";
        std::cout<<"\n \t\tOR";
        std::cout<<"\nmain.exe\t[IMAGE]\t[SALT PEPPER NOISE REMOVAL]\t[TXT FILE CONTAINING HIST/PDF/CDF]";
    
        std::cout<<"\nmain.exe\texample.jpg\t1\texample1.jpg\t1";
        std::cout<<"\n \t\tOR";
        std::cout<<"\nmain.exe\texample.jpg\t1\tdata.txt";

        return 1;
    }

    cv::Mat inp_img = cv::imread(args[1], cv::IMREAD_GRAYSCALE);
    
    // remove salt and pepper noise
    if(args[2][0] == '1')
        cv::medianBlur(inp_img, inp_img, 3);
    

    cv::Mat out_img;

    // txt file
    if(argc == 4){
        std::cout<<"\nSPECIFYFILE TYPE: ";
        std::cout<<"\n1. HISTOGRAM (FREQUENCIES).";
        std::cout<<"\n2. PDF.";
        std::cout<<"\nENTER FILE TYPE: ";

        int ch;
        std::cin>>ch;

        std::ifstream inputFile(args[3]); // Open the input file
        if (!inputFile.is_open()){
            std::cerr << "Error opening the file.\n";
            return 1;
        }

        if(ch == 1){
            // will read file as histogram
            int *ref_hist = (int *)malloc(256 * sizeof(int));

            std::string line;
            while (getline(inputFile, line)) {
                std::istringstream iss(line);
                
                int idx, val;

                if (iss >> idx >> val) 
                    ref_hist[idx] = val;
                 
                else
                    std::cerr << "Error reading line: " << line << std::endl;
            }

            float *ref_pdf = computePDF(ref_hist);
            out_img = histogramMatching(inp_img, ref_pdf);

            inputFile.close(); // Close the input file
        }
        else if(ch == 2){
            // will read file as pdf
            float *ref_pdf = (float *)malloc(256 * sizeof(float));

            std::string line;
            while (getline(inputFile, line)) {
                std::istringstream iss(line);
                
                int idx;
                float val;

                if (iss >> idx >> val) 
                    ref_pdf[idx] = val;
                 
                else
                    std::cerr << "Error reading line: " << line << std::endl;
            }

            out_img = histogramMatching(inp_img, ref_pdf);

            inputFile.close(); // Close the input file
        }
            
    }
    // image file
    else{
        cv::Mat reference_img = cv::imread(args[3], cv::IMREAD_GRAYSCALE);
        
        // remove salt and pepper noise
        if(args[4][0] == '1')
            cv::medianBlur(inp_img, inp_img, 3);

        out_img = histogramMatching(inp_img, reference_img);

        cv::imshow("Ref Img", reference_img);
            
    }

    int k;
    do{
        cv::imshow("Org Img", inp_img);
        cv::imshow("Matched Img", out_img);
        
        k = cv::waitKey(0);
    }while(k!= 'q' && k!= 'Q');

    cv::destroyAllWindows();

    return 0;
}