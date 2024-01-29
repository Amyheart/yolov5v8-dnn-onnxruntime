#include <iostream>
#include <getopt.h>
#include "yolov5v8_dnn.h"
#include "yolov5v8_ort.h"

using namespace std;
using namespace cv;

void main(int argc, char** argv)
{
    string img_path = "../img_test/bus.jpg";
    string model_path = "../weight_v5/yolov5s-fp16.onnx";
    string test_cls = "dnn";
    if (test_cls == "dnn") {
        // Input the path of model ("yolov8s.onnx" or "yolov5s.onnx") to run Inference with yolov8/yolov5 (ONNX)
        // Note that in this example the classes are hard-coded and 'classes.txt' is a place holder.
        Inference inf(model_path, cv::Size(640, 640), "classes.txt", true);
        cv::Mat frame = cv::imread(img_path);
        std::vector<Detection> output = inf.runInference(frame);
        if (output.size() != 0) inf.DrawPred(frame, output);
        else cout << "Detect Nothing!" << endl;
    }
   
    if (test_cls == "ort") {
        DCSP_CORE* yoloDetector = new DCSP_CORE;
        #ifdef USE_CUDA
        //DCSP_INIT_PARAM params{ model_path, YOLO_ORIGIN_V5, {640, 640},  0.25, 0.45, 0.5, true }; // GPU FP32 inference
        DCSP_INIT_PARAM params{ model_path, YOLO_ORIGIN_V5_HALF, {640, 640},  0.25, 0.45, 0.5, true }; // GPU FP16 inference
        #else
        DCSP_INIT_PARAM params{ model_path, YOLO_ORIGIN_V5, {640, 640},0.25, 0.45, 0.5, false };  // CPU inference
        #endif
        yoloDetector->CreateSession(params);
        cv::Mat img = cv::imread(img_path);
        std::vector<DCSP_RESULT> res;
        yoloDetector->RunSession(img, res);
        if (res.size() != 0) yoloDetector->DrawPred(img, res);
        else cout << "Detect Nothing!" << endl;
    }
}
