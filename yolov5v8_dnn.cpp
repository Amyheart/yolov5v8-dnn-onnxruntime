#include "yolov5v8_dnn.h"
using namespace std;

Inference::Inference(const std::string& onnxModelPath, const cv::Size& modelInputShape, const std::string& classesTxtFile, const bool& runWithCuda)
{
    modelPath = onnxModelPath;
    modelShape = modelInputShape;
    classesPath = classesTxtFile;
    cudaEnabled = runWithCuda;

    loadOnnxNetwork();
    // loadClassesFromFile(); The classes are hard-coded for this example
}

std::vector<Detection> Inference::runInference(const cv::Mat& input)
{
    cv::Mat SrcImg = input;
    cv::Mat netInputImg;
    cv::Vec4d params;
    LetterBox(SrcImg, netInputImg, params, cv::Size(modelShape.width, modelShape.height));

    cv::Mat blob;
    cv::dnn::blobFromImage(netInputImg, blob, 1.0 / 255.0, modelShape, cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    if (outputs.size() == 2) RunSegmentation = true;

    int rows = outputs[0].size[1];
    int dimensions = outputs[0].size[2];

    bool yolov8 = false;
    // yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
    // yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
    if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
    {
        yolov8 = true;
        rows = outputs[0].size[2];
        dimensions = outputs[0].size[1];

        outputs[0] = outputs[0].reshape(1, dimensions);
        cv::transpose(outputs[0], outputs[0]);
    }
    float* data = (float*)outputs[0].data;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<vector<float>> picked_proposals;

    for (int i = 0; i < rows; ++i)
    {
        int _segChannels;
        if (yolov8)
        {
            float* classes_scores = data + 4;
            cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double maxClassScore;
            minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
            if (maxClassScore > modelScoreThreshold)
            {
                if (RunSegmentation) {
                    _segChannels = outputs[1].size[1];
                    vector<float> temp_proto(data + classes.size() + 4, data + classes.size() + 4 + _segChannels);
                    picked_proposals.push_back(temp_proto);
                }
                confidences.push_back(maxClassScore);
                class_ids.push_back(class_id.x);
                float x = (data[0] - params[2]) / params[0];
                float y = (data[1] - params[3]) / params[1];
                float w = data[2] / params[0];
                float h = data[3] / params[1];
                int left = MAX(round(x - 0.5 * w + 0.5), 0);
                int top = MAX(round(y - 0.5 * h + 0.5), 0);
                if ((left + w) > SrcImg.cols) { w = SrcImg.cols - left;}
                if ((top + h) > SrcImg.rows) { h = SrcImg.rows - top; }
                boxes.push_back(cv::Rect(left, top, int(w), int(h)));
            }
        }
        else // yolov5
        {
            float confidence = data[4];
            if (confidence >= modelConfidenceThreshold)
            {
                float* classes_scores = data + 5;
                cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
                cv::Point class_id;
                double max_class_score;
                minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
                if (max_class_score > modelScoreThreshold)
                {
                    if (RunSegmentation) {
                        _segChannels = outputs[1].size[1];
                        vector<float> temp_proto(data + classes.size() + 5, data + classes.size() + 5 + _segChannels);
                        picked_proposals.push_back(temp_proto);
                    }
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id.x);
                    float x = (data[0] - params[2]) / params[0];
                    float y = (data[1] - params[3]) / params[1];
                    float w = data[2] / params[0];
                    float h = data[3] / params[1];
                    int left = MAX(round(x - 0.5 * w + 0.5), 0);
                    int top = MAX(round(y - 0.5 * h + 0.5), 0);
                    if ((left + w) > SrcImg.cols) { w = SrcImg.cols - left; }
                    if ((top + h) > SrcImg.rows) { h = SrcImg.rows - top; }
                    boxes.push_back(cv::Rect(left, top, int(w), int(h)));
                }
            }
        }
        data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);
    std::vector<Detection> detections{};
    std::vector<vector<float>> temp_mask_proposals;
    for (unsigned long i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(100, 255);
        result.color = cv::Scalar(dis(gen),
            dis(gen),
            dis(gen));

        result.className = classes[result.class_id];
        result.box = boxes[idx];
        if (RunSegmentation) temp_mask_proposals.push_back(picked_proposals[idx]);
        if (result.box.width != 0 && result.box.height != 0) detections.push_back(result);
    }
    if (RunSegmentation) {
        cv::Mat mask_proposals;
        for (int i = 0; i < temp_mask_proposals.size(); ++i)
            mask_proposals.push_back(cv::Mat(temp_mask_proposals[i]).t());
        GetMask(mask_proposals, outputs[1], params, SrcImg.size(), detections);
    }
    return detections;
}

void Inference::loadClassesFromFile()
{
    std::ifstream inputFile(classesPath);
    if (inputFile.is_open())
    {
        std::string classLine;
        while (std::getline(inputFile, classLine))
            classes.push_back(classLine);
        inputFile.close();
    }
}

void Inference::loadOnnxNetwork()
{
    net = cv::dnn::readNetFromONNX(modelPath);
    
    if (cudaEnabled)
    {
        std::cout << "\nRunning on CUDA" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else
    {
        std::cout << "\nRunning on CPU" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}

void Inference::LetterBox(const cv::Mat& image, cv::Mat& outImage, cv::Vec4d& params, const cv::Size& newShape,
    bool autoShape, bool scaleFill, bool scaleUp, int stride, const cv::Scalar& color)
{
    if (false) {
        int maxLen = MAX(image.rows, image.cols);
        outImage = cv::Mat::zeros(cv::Size(maxLen, maxLen), CV_8UC3);
        image.copyTo(outImage(cv::Rect(0, 0, image.cols, image.rows)));
        params[0] = 1;
        params[1] = 1;
        params[3] = 0;
        params[2] = 0;
    }

    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height,
        (float)newShape.width / (float)shape.width);
    if (!scaleUp)
        r = std::min(r, 1.0f);

    float ratio[2]{ r, r };
    int new_un_pad[2] = { (int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r) };

    auto dw = (float)(newShape.width - new_un_pad[0]);
    auto dh = (float)(newShape.height - new_un_pad[1]);

    if (autoShape)
    {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    }
    else if (scaleFill)
    {
        dw = 0.0f;
        dh = 0.0f;
        new_un_pad[0] = newShape.width;
        new_un_pad[1] = newShape.height;
        ratio[0] = (float)newShape.width / (float)shape.width;
        ratio[1] = (float)newShape.height / (float)shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
    {
        cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
    }
    else {
        outImage = image.clone();
    }

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    params[0] = ratio[0];
    params[1] = ratio[1];
    params[2] = left;
    params[3] = top;
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

void Inference::GetMask(const cv::Mat& maskProposals, const cv::Mat& mask_protos, const cv::Vec4d& params, const cv::Size& srcImgShape, std::vector<Detection>& output) {
    if (output.size() == 0) return;
    int _segChannels = mask_protos.size[1];
    int _segHeight = mask_protos.size[2];
    int _segWidth = mask_protos.size[3];
    cv::Mat protos = mask_protos.reshape(0, { _segChannels,_segWidth * _segHeight });
    cv::Mat matmulRes = (maskProposals * protos).t();
    cv::Mat masks = matmulRes.reshape(output.size(), { _segHeight,_segWidth });
    vector<cv::Mat> maskChannels;
    split(masks, maskChannels);

    for (int i = 0; i < output.size(); ++i) {
        cv::Mat dest, mask;
        //sigmoid
        cv::exp(-maskChannels[i], dest);
        dest = 1.0 / (1.0 + dest);

        cv::Rect roi(int(params[2] / modelShape.width * _segWidth), int(params[3] / modelShape.height * _segHeight), int(_segWidth - params[2] / 2), int(_segHeight - params[3] / 2));
        dest = dest(roi);
        cv::resize(dest, mask, srcImgShape, cv::INTER_NEAREST);

        //crop
        cv::Rect temp_rect = output[i].box;
        mask = mask(temp_rect) > modelScoreThreshold;
        output[i].boxMask = mask;
    }
}

void Inference::DrawPred(cv::Mat& img, vector<Detection>& result) {
    int detections = result.size();
    std::cout << "Number of detections:" << detections << std::endl;
    cv::Mat mask = img.clone();
    for (int i = 0; i < detections; ++i)
    {
        Detection detection = result[i];

        cv::Rect box = detection.box;
        cv::Scalar color = detection.color;

        // Detection box
        cv::rectangle(img, box, color, 2);
        mask(detection.box).setTo(color, detection.boxMask);

        // Detection box text
        std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

        cv::rectangle(img, textBox, color, cv::FILLED);
        cv::putText(img, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }
    // Detection mask
    if (RunSegmentation) cv::addWeighted(img, 0.5, mask, 0.5, 0, img); //将mask加在原图上面
    cv::imshow("Inference", img);
    cv::imwrite("out.bmp", img);
    cv::waitKey();
    cv::destroyWindow("Inference");
}