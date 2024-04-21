// opencv libraries
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
// standard libraries
#include <filesystem>
#include <windows.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip> // for std::setw and std::setfill
#include <string>
#include<algorithm>

bool USE_CUDA = 0;

// we will use the YOLOv8 detector for training. Since training is a crucial process, we can afford to invest time to this
class YOLOv8_face
{
public:
    YOLOv8_face(std::string modelpath, float confThreshold, float nmsThreshold);
    void detect(cv::Mat& frame, std::vector<cv::Rect>& boxes, std::vector<float>& confidences, std::vector<std::vector<cv::Point>>& landmarks, bool return_largest);

private:
    cv::Mat resize_image(cv::Mat srcimg, int* newh, int* neww, int* padh, int* padw);
    const bool keep_ratio = true;
    const int inpWidth = 640;
    const int inpHeight = 640;
    float confThreshold;
    float nmsThreshold;
    const int num_class = 1;
    const int reg_max = 16;
    cv::dnn::Net net;
    void softmax_(const float* x, float* y, int length);
    void generate_proposal(cv::Mat out, std::vector<cv::Rect>& boxes, std::vector<float>& confidences, std::vector<std::vector<cv::Point>>& landmarks, int imgh, int imgw, float ratioh, float ratiow, int padh, int padw);
    void drawPred(float conf, int left, int top, int right, int bottom, cv::Mat& frame, std::vector<cv::Point> landmark);
};

static inline float sigmoid_x(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

YOLOv8_face::YOLOv8_face(std::string modelpath, float confThreshold, float nmsThreshold)
{
    this->confThreshold = confThreshold;
    this->nmsThreshold = nmsThreshold;
    this->net = cv::dnn::readNet(modelpath);
    if (USE_CUDA)
    {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else
    {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}

cv::Mat YOLOv8_face::resize_image(cv::Mat srcimg, int* newh, int* neww, int* padh, int* padw)
{
    int srch = srcimg.rows, srcw = srcimg.cols;
    *newh = this->inpHeight;
    *neww = this->inpWidth;
    cv::Mat dstimg;
    if (this->keep_ratio && srch != srcw)
    {
        float hw_scale = (float)srch / srcw;
        if (hw_scale > 1)
        {
            *newh = this->inpHeight;
            *neww = int(this->inpWidth / hw_scale);
            cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
            *padw = int((this->inpWidth - *neww) * 0.5);
            copyMakeBorder(dstimg, dstimg, 0, 0, *padw, this->inpWidth - *neww - *padw, cv::BORDER_CONSTANT, 0);
        }
        else
        {
            *newh = (int)this->inpHeight * hw_scale;
            *neww = this->inpWidth;
            cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
            *padh = (int)(this->inpHeight - *newh) * 0.5;
            copyMakeBorder(dstimg, dstimg, *padh, this->inpHeight - *newh - *padh, 0, 0, cv::BORDER_CONSTANT, 0);
        }
    }
    else
    {
        cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
    }
    return dstimg;
}

void YOLOv8_face::softmax_(const float* x, float* y, int length)
{
    float sum = 0;
    int i = 0;
    for (i = 0; i < length; i++)
    {
        y[i] = exp(x[i]);
        sum += y[i];
    }
    for (i = 0; i < length; i++)
    {
        y[i] /= sum;
    }
}

void YOLOv8_face::generate_proposal(cv::Mat out, std::vector<cv::Rect>& boxes, std::vector<float>& confidences, std::vector<std::vector<cv::Point>>& landmarks, int imgh, int imgw, float ratioh, float ratiow, int padh, int padw)
{
    const int feat_h = out.size[2];
    const int feat_w = out.size[3];
    // std::cout << out.size[1] << "," << out.size[2] << "," << out.size[3] << std::endl;
    const int stride = (int)ceil((float)inpHeight / feat_h);
    const int area = feat_h * feat_w;
    float* ptr = (float*)out.data;
    float* ptr_cls = ptr + area * reg_max * 4;
    float* ptr_kp = ptr + area * (reg_max * 4 + num_class);

    for (int i = 0; i < feat_h; i++)
    {
        for (int j = 0; j < feat_w; j++)
        {
            const int index = i * feat_w + j;
            int cls_id = -1;
            float max_conf = -10000;
            for (int k = 0; k < num_class; k++)
            {
                float conf = ptr_cls[k * area + index];
                if (conf > max_conf)
                {
                    max_conf = conf;
                    cls_id = k;
                }
            }
            float box_prob = sigmoid_x(max_conf);
            if (box_prob > this->confThreshold)
            {
                float pred_ltrb[4];
                float* dfl_value = new float[reg_max];
                float* dfl_softmax = new float[reg_max];
                for (int k = 0; k < 4; k++)
                {
                    for (int n = 0; n < reg_max; n++)
                    {
                        dfl_value[n] = ptr[(k * reg_max + n) * area + index];
                    }
                    softmax_(dfl_value, dfl_softmax, reg_max);

                    float dis = 0.f;
                    for (int n = 0; n < reg_max; n++)
                    {
                        dis += n * dfl_softmax[n];
                    }

                    pred_ltrb[k] = dis * stride;
                }
                float cx = (j + 0.5f) * stride;
                float cy = (i + 0.5f) * stride;
                float xmin = max((cx - pred_ltrb[0] - padw) * ratiow, 0.f); /// restore to the original image
                float ymin = max((cy - pred_ltrb[1] - padh) * ratioh, 0.f);
                float xmax = min((cx + pred_ltrb[2] - padw) * ratiow, float(imgw - 1));
                float ymax = min((cy + pred_ltrb[3] - padh) * ratioh, float(imgh - 1));
                cv::Rect box = cv::Rect(int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin));
                boxes.push_back(box);
                confidences.push_back(box_prob);

                // the below code is for landmarks and has been commented out
                // vector<Point> kpts(5);
                // for (int k = 0; k < 5; k++)
                // {
                // 	float x = ((ptr_kp[(k * 3)*area + index] * 2 + j)*stride - padw)*ratiow;  ///restore to the original image

                // 	float y = ((ptr_kp[(k * 3 + 1)*area + index] * 2 + i)*stride - padh)*ratioh;
                // 	///float pt_conf = sigmoid_x(ptr_kp[(k * 3 + 2)*area + index]);
                // 	kpts[k] = Point(int(x), int(y));
                // }
                // landmarks.push_back(kpts);
            }
        }
    }
}

void YOLOv8_face::detect(cv::Mat& srcimg, std::vector<cv::Rect>& boxes, std::vector<float>& confidences, std::vector<std::vector<cv::Point>>& landmarks, bool return_largest = false)
{
    int newh = 0, neww = 0, padh = 0, padw = 0;
    cv::Mat dst = this->resize_image(srcimg, &newh, &neww, &padh, &padw);
    cv::Mat blob;
    cv::dnn::blobFromImage(dst, blob, 1 / 255.0, cv::Size(this->inpWidth, this->inpHeight), cv::Scalar(0, 0, 0), true, false);
    this->net.setInput(blob);
    std::vector<cv::Mat> outs;

    this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

    /////generate proposals
    float ratioh = (float)srcimg.rows / newh, ratiow = (float)srcimg.cols / neww;

    generate_proposal(outs[0], boxes, confidences, landmarks, srcimg.rows, srcimg.cols, ratioh, ratiow, padh, padw);
    generate_proposal(outs[1], boxes, confidences, landmarks, srcimg.rows, srcimg.cols, ratioh, ratiow, padh, padw);
    generate_proposal(outs[2], boxes, confidences, landmarks, srcimg.rows, srcimg.cols, ratioh, ratiow, padh, padw);

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);

    // finding the largest face
    if (boxes.size() > 1 && return_largest)
    {
        int largest_area = 0;
        int largest_idx = 0;
        for (int i = 0; i < indices.size(); i++)
        {
            int idx = indices[i];
            cv::Rect box = boxes[idx];
            if (box.width * box.height > largest_area)
            {
                largest_area = box.width * box.height;
                largest_idx = idx;
            }
        }
        indices.clear();
        indices.push_back(largest_idx);
        auto largest_box = boxes[largest_idx];
        auto largest_conf = confidences[largest_idx];
        // auto largest_landmark = landmarks[largest_idx];
        boxes.clear();
        confidences.clear();
        // landmarks.clear();
        boxes.push_back(largest_box);
        confidences.push_back(largest_conf);
        // landmarks.push_back(largest_landmark);
    }
}

// Convert wide string to narrow string
std::string ConvertWideStringToNarrow(const wchar_t* wideString) {
    int bufferSize = WideCharToMultiByte(CP_UTF8, 0, wideString, -1, NULL, 0, NULL, NULL);
    if (bufferSize == 0) {
        return "";
    }

    std::vector<char> buffer(bufferSize);
    WideCharToMultiByte(CP_UTF8, 0, wideString, -1, buffer.data(), bufferSize, NULL, NULL);
    return std::string(buffer.data());
}

void ListDirectories(const std::wstring& folderPath, std::vector<std::string>& directories) {
    WIN32_FIND_DATAW findFileData; // Use WIN32_FIND_DATAW for Unicode (wide character)
    HANDLE hFind = FindFirstFileW((folderPath + L"\\*").c_str(), &findFileData); // Use FindFirstFileW

    if (hFind == INVALID_HANDLE_VALUE) {
        std::wcerr << L"Error opening directory: " << folderPath << std::endl;
        return;
    }

    do {
        if ((findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) &&
            wcscmp(findFileData.cFileName, L".") != 0 &&
            wcscmp(findFileData.cFileName, L"..") != 0) {
            directories.push_back(ConvertWideStringToNarrow(findFileData.cFileName));
        }
    } while (FindNextFileW(hFind, &findFileData) != 0); // Use FindNextFileW

    FindClose(hFind);
}

int main(int argc, char* args[])
{
    //if (argc != 2)
    //{
        //printf("\nUSAGE: SAVE 6 VIDEOS as v1.mp4, v2.mp4, v3.mp4, v4.mp4, v5.mp4, v6.mp4. then run ./main [ObjID]");
    //}
    //else if (argc == 4 && args[3] == "1") {
        //USE_CUDA = 1;
    //}
    

    std::vector<std::string> paths;
    paths.push_back("v1.mp4");
    paths.push_back("v2.mp4");
    paths.push_back("v3.mp4");
    paths.push_back("v4.mp4");
    paths.push_back("v5.mp4");
    paths.push_back("v6.mp4");

    wchar_t currentDir[MAX_PATH];
    if (GetCurrentDirectoryW(MAX_PATH, currentDir) == 0) { // Use GetCurrentDirectoryW
        std::wcerr << L"Error getting current directory." << std::endl;
        return 1;
    }

    std::vector<std::string> folders;
    std::wstring folderPath(currentDir);
    ListDirectories(folderPath, folders);
    std::string detector_model_path = "C:/Users/shash/Documents/IPCVInference1/models/yolov8n-face.onnx";


            for (int fldIdx = 0; fldIdx < folders.size(); ++fldIdx) {
                YOLOv8_face detector = YOLOv8_face(detector_model_path, 0.5, 0.96);
                std::string folder_name = folders[fldIdx];
                std::string regNo = folder_name;
                for (int vIdx = 0; vIdx < paths.size(); ++vIdx) {
                    std::string vid_path = folder_name + "/" + paths[vIdx];
                    cv::VideoCapture cap(vid_path);
                    int fps = cap.get(cv::CAP_PROP_FPS);
                    const int TOTAL_FRAMES = cap.get(cv::CAP_PROP_FRAME_COUNT);
                    const int FRAME_INTERVAL = TOTAL_FRAMES / 25; // 25 frames per video
                    std::cout << "\n VID FPS: " << fps << ". TOTAL FRAMES: " << TOTAL_FRAMES<<". FRAME INTERVAL: "<<FRAME_INTERVAL;
                    std::string vidNum;
                    char a = vid_path[vid_path.size() - 5];
                    if (a == '1') vidNum = std::to_string(1);
                    else if (a == '2') vidNum = std::to_string(2);
                    else if (a == '3') vidNum = std::to_string(3);
                    else if (a == '4') vidNum = std::to_string(4);
                    else if (a == '5') vidNum = std::to_string(5);
                    else if (a == '6') vidNum = std::to_string(6);

                    if (!cap.isOpened()) {
                        printf("Error: Unable to open the video file.\n");
                        return -1;
                    }
                    std::string op_folder1 = folder_name + "/output";

                    try {
                        // Check if the directory already exists
                        if (!std::filesystem::exists(op_folder1)) {
                            // Create the directory
                            std::filesystem::create_directory(op_folder1);
                        }
                    }
                    catch (const std::filesystem::filesystem_error& e) {
                        std::cerr << "Error creating directory: " << e.what() << std::endl;
                    }

                    std::string op_folder2 = folder_name + "/output/v" + vidNum;

                    try {
                        // Check if the directory already exists
                        if (!std::filesystem::exists(op_folder2)) {
                            // Create the directory
                            std::filesystem::create_directory(op_folder2);
                        }
                    }
                    catch (const std::filesystem::filesystem_error& e) {
                        std::cerr << "Error creating directory: " << e.what() << std::endl;
                    }


                    cv::Mat frame;
                    std::cout << "\n[..] STARTING ON " << vid_path;
                    for (int i = 1; i <= FRAME_INTERVAL * 25 && cap.read(frame); ++i) {
                        if( i % FRAME_INTERVAL != 0 ) continue;
                        std::ostringstream oss;
                        oss << std::setw(3) << std::setfill('0') << i; // Set width to 3 and fill with '0'
                        std::string txt_filename = folder_name + "/" + "output/v" + vidNum + "/" + regNo + "_v" + vidNum + "_f" + oss.str() + ".txt";
                        std::string jpg_filename = folder_name + "/" + "output/v" + vidNum + "/" + regNo + "_v" + vidNum + "_f" + oss.str() + ".jpg";

                        std::ofstream file(txt_filename, std::ios::app);

                        if (!file.is_open()) {
                            std::cerr << "Error opening the file!" << txt_filename << std::endl;
                            return 1; // Return error code
                        }
                        std::vector<cv::Rect> boxes;
                        std::vector<float> confidences;
                        std::vector<std::vector<cv::Point>> landmarks;
                        std::vector<int> indices;
                        detector.detect(frame, boxes, confidences, landmarks, true);
                        if (boxes.size() == 0) { --i; continue; }

                        double x     = boxes[0].x;
                        double y     = boxes[0].y;
                        double width    = boxes[0].width;
                        double height   = boxes[0].height;

                        double x_mid = (x + x + width) / 2;
                        double x_norm = (x_mid) / frame.cols;
                        double y_mid = (y + y + width) / 2;
                        double y_norm = (y_mid) / frame.rows;

                        double width_norm = (width) / frame.cols;
                        double height_norm = (height) / frame.rows;


                        std::string file_data = regNo + " " + std::to_string(x_norm) + " " + std::to_string(y_norm) + " " + std::to_string(width_norm) + " " + std::to_string(height_norm);
                        file << file_data << std::endl;

                        // to show cropped faces
                        frame = frame(boxes[0]);
                        cv::imshow("img", frame);
                        cv::waitKey(10);
                        
                        // Close the file
                        file.close();
                        cv::imwrite(jpg_filename, frame);
                    }
                    std::cout << "\n[+] " << vid_path << " DONE.";

                }
            }
    
    
    return 0;
}