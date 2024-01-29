#ifndef YOLOV5V8_DNN_H
#define YOLOV5V8_DNN_H

// Cpp native
#include <fstream>
#include <vector>
#include <string>
#include <random>

// OpenCV / DNN / Inference
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

struct Detection
{
    int class_id{ 0 };
    std::string className{};
    float confidence{ 0.0 };
    cv::Scalar color{};
    cv::Rect box{};
    cv::Mat boxMask; 
};

class Inference
{
public:
    Inference(const std::string& onnxModelPath, const cv::Size& modelInputShape = { 640, 640 }, const std::string& classesTxtFile = "", const bool& runWithCuda = true);
    std::vector<Detection> runInference(const cv::Mat& input);
    void DrawPred(cv::Mat& img, std::vector<Detection>& result);

private:
    void loadClassesFromFile();
    void loadOnnxNetwork();
    void LetterBox(const cv::Mat& image, cv::Mat& outImage,
        cv::Vec4d& params, //[ratio_x,ratio_y,dw,dh]
        const cv::Size& newShape = cv::Size(640, 640),
        bool autoShape = false,
        bool scaleFill = false,
        bool scaleUp = true,
        int stride = 32,
        const cv::Scalar& color = cv::Scalar(114, 114, 114));
    void GetMask(const cv::Mat& maskProposals, const cv::Mat& mask_protos, const cv::Vec4d& params, const cv::Size& srcImgShape, std::vector<Detection>& output);

private:
    std::string modelPath{};
    bool cudaEnabled{};
    cv::Size2f modelShape{};
    bool RunSegmentation = false;
    float modelConfidenceThreshold{ 0.25 };
    float modelScoreThreshold{ 0.45 };
    float modelNMSThreshold{ 0.50 };

    bool letterBoxForSquare = true;
    cv::dnn::Net net;
    std::string classesPath{};
    std::vector<std::string> classes{ "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
        "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };
};

#endif // YOLOV5V8_DNN_H
