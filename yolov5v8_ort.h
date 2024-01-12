#pragma once

#define    RET_OK nullptr
#define    USE_CUDA

#ifdef _WIN32
#include <Windows.h>
#include <direct.h>
#include <io.h>
#endif

#include <string>
#include <vector>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"

#ifdef USE_CUDA
#include <cuda_fp16.h>
#endif


enum MODEL_TYPE {
    //FLOAT32 MODEL
    YOLO_ORIGIN_V5 = 0,//support v5 detector currently
    YOLO_ORIGIN_V8 = 1,//support v8 detector currently
    YOLO_POSE_V8 = 2,
    YOLO_CLS_V8 = 3,
    YOLO_ORIGIN_V8_HALF = 4,
    YOLO_POSE_V8_HALF = 5,
    YOLO_CLS_V8_HALF = 6
};


typedef struct _DCSP_INIT_PARAM {
    std::string ModelPath;
    MODEL_TYPE ModelType = YOLO_ORIGIN_V8;
    std::vector<int> imgSize = { 640, 640 };
    float modelConfidenceThreshold=0.25;
    float RectConfidenceThreshold = 0.6;
    float iouThreshold = 0.5;
    bool CudaEnable = false;
    int LogSeverityLevel = 3;
    int IntraOpNumThreads = 1;
} DCSP_INIT_PARAM;


typedef struct _DCSP_RESULT {
    int classId;
    std::string className;
    float confidence;
    cv::Rect box;
    cv::Mat boxMask;       //矩形框内mask
    cv::Scalar color;
} DCSP_RESULT;


class DCSP_CORE {
public:
    DCSP_CORE();
    ~DCSP_CORE();

public:
    void DrawPred(cv::Mat& img, std::vector<DCSP_RESULT>& result);
    char* CreateSession(DCSP_INIT_PARAM& iParams);
    char* RunSession(cv::Mat& iImg, std::vector<DCSP_RESULT>& oResult);
    char* WarmUpSession();
    template<typename N>
    char* TensorProcess(clock_t& starttime_1, cv::Vec4d& params, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims,
        std::vector<DCSP_RESULT>& oResult);
    std::vector<std::string> classes{ "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };

private:
    Ort::Env env;
    Ort::Session* session;
    bool cudaEnable;
    Ort::RunOptions options;
    bool RunSegmentation = false;
    std::vector<const char*> inputNodeNames;
    std::vector<const char*> outputNodeNames;

    MODEL_TYPE modelType;
    std::vector<int> imgSize;
    float modelConfidenceThreshold;
    float rectConfidenceThreshold;
    float iouThreshold;

    
};

//void LetterBox(const cv::Mat& image, cv::Mat& outImage,
//    cv::Vec4d& params, //[ratio_x,ratio_y,dw,dh]
//    const cv::Size& newShape = cv::Size(640, 640),
//    bool autoShape = false,
//    bool scaleFill = false,
//    bool scaleUp = true,
//    int stride = 32,
//    const cv::Scalar& color = cv::Scalar(114, 114, 114));
