#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <openvino/openvino.hpp>
#include "rclcpp/rclcpp.hpp"

#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include <autoaim_interfaces/msg/detection.hpp>
#include <autoaim_interfaces/msg/detection_array.hpp>

enum COLOR { GRAY = 0, BLUE = 1, RED = 2, PURPLE = 3 };

enum TAG {
    SENTRY = 0,
    HERO = 1,
    ENGINEER = 2,
    INFANTRY_3 = 3,
    INFRANTRY_4 = 4,
    INFRANTRY_5 = 5,
};

struct Config {
    float confThreshold;
    float nmsThreshold;
    int numColor;
    int numTag;
    std::string onnxPath;
};

struct Resize {
    cv::Mat resizedImage;
    int dw;
    int dh;
};

struct Object {
    cv::Rect box;
    float conf;
    cv::Point kpts[4];
};

class OpenVINOInferEngine: public InferEngine {
public:
    // OpenVINOInferEngine() = default;
    OpenVINOInferEngine(Config config);
    ~OpenVINOInferEngine() = default;

    void img_preprocess() override;
    void infer() override;
    void img_postprocess() override;
    void set_input_image(const cv::Mat img) override;
    std::vector<autoaim_interfaces::msg::Detection> get_detection_vector() const override;
    cv::Mat debug_draw_armors() override;

    cv::Point2f offset;
    cv::Mat showImg;

private:
    float confThreshold;
    float nmsThreshold;
    int numColor;
    int numTag;
    int inputWidth;
    int inputHeight;
    int heightBias;
    float ratio;
    std::string onnxPath;
    Resize resize;
    cv::Mat img;
    ov::Tensor inputTensor;
    ov::InferRequest inferRequest;
    ov::CompiledModel compiledModel;
    autoaim_interfaces::msg::DetectionArray armorMsg;

    // 计算FPS
    std::chrono::time_point<std::chrono::steady_clock> start, end;

    void nms(std::vector<cv::Rect>& boxes_, float nmsThreshold_, std::vector<int>& results_);
    void model_initialize();
};

const std::vector<std::string> name = {
    "Sentry",     "Hero",    "Engineer",   "Infantry_3", "Infantry_4",
    "Infantry_5", "OutPost", "Small Base", "Big Base",
};
// gray blue red purple
const std::vector<cv::Scalar> colors = {
    cv::Scalar(114, 114, 114),
    cv::Scalar(255, 0, 0),
    cv::Scalar(0, 0, 255),
    cv::Scalar(255, 0, 255),
};
// 好看的点颜色 10个
const std::vector<cv::Scalar> pointColor = {
    cv::Scalar(255, 0, 0),     cv::Scalar(0, 255, 0),     cv::Scalar(0, 0, 255),
    cv::Scalar(255, 255, 0),   cv::Scalar(255, 0, 255),   cv::Scalar(0, 255, 255),
    cv::Scalar(255, 255, 255), cv::Scalar(128, 128, 128), cv::Scalar(0, 0, 0),
    cv::Scalar(255, 255, 255),
};

OpenVINOInferEngine::OpenVINOInferEngine(Config config) {
    confThreshold = config.confThreshold;
    nmsThreshold = config.nmsThreshold;
    numColor = config.numColor;
    numTag = config.numTag;
    onnxPath = config.onnxPath;
    model_initialize();
}

void OpenVINOInferEngine::model_initialize() {
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(onnxPath);
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
    ppp.input()
        .tensor()
        .set_element_type(ov::element::u8)
        .set_layout("NHWC")
        .set_color_format(ov::preprocess::ColorFormat::RGB);
    ppp.input()
        .preprocess()
        .convert_element_type(ov::element::f32)
        .convert_color(ov::preprocess::ColorFormat::BGR)
        .scale({ 255, 255, 255 }); // .scale({ 112, 112, 112 });
    ppp.input().model().set_layout("NCHW");
    ppp.output().tensor().set_element_type(ov::element::f32);
    model = ppp.build();
    compiledModel = core.compile_model(model, "GPU");
    inferRequest = compiledModel.create_infer_request();
    inputWidth = compiledModel.input().get_shape()[2];
    inputHeight = compiledModel.input().get_shape()[1];
}

void OpenVINOInferEngine::img_preprocess() {
    try {
        float width = img.cols;
        float height = img.rows;
        cv::Size new_shape = cv::Size(inputWidth, inputHeight);
        float r = float(new_shape.width / width);
        int new_unpadW = int(round(width * r));
        int new_unpadH = int(round(height * r));
        cv::resize(
            img,
            resize.resizedImage,
            cv::Size(new_unpadW, new_unpadH),
            0,
            0,
            cv::INTER_AREA
        );
        // 从图像下面裁剪成(new_shape.width, inputHeight)
        heightBias = resize.resizedImage.rows - inputHeight;

        resize.resizedImage = resize.resizedImage(
            cv::Rect(0, resize.resizedImage.rows - inputHeight, inputWidth, inputHeight)
        );
        ratio = 1 / r;
        showImg = img.clone()(
            cv::Rect(0, img.rows - ratio * inputHeight, ratio * inputWidth, ratio * inputHeight)
        );
        float* inputData = (float*)resize.resizedImage.data;
        inputTensor = ov::Tensor(
            compiledModel.input().get_element_type(),
            compiledModel.input().get_shape(),
            inputData
        );
        inferRequest.set_input_tensor(inputTensor);
    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "unknown exception" << std::endl;
    }
}

void OpenVINOInferEngine::infer() {
    inferRequest.infer();
}

void OpenVINOInferEngine::nms(
    std::vector<cv::Rect>& boxes_,
    float nmsThreshold_,
    std::vector<int>& results_
) {
    std::vector<int> idx(boxes_.size());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&boxes_](int i1, int i2) { return boxes_[i1].x < boxes_[i2].x; });
    for (int i = 0; i < boxes_.size(); i++) {
        cv::Rect box1 = boxes_[idx[i]];
        if (box1.area() == 0)
            continue;
        for (int j = i + 1; j < boxes_.size(); j++) {
            cv::Rect box2 = boxes_[idx[j]];
            if (box2.area() == 0)
                continue;
            int xx1 = std::max(box1.x, box2.x);
            int yy1 = std::max(box1.y, box2.y);
            int xx2 = std::min(box1.x + box1.width, box2.x + box2.width);
            int yy2 = std::min(box1.y + box1.height, box2.y + box2.height);
            int w = std::max(0, xx2 - xx1);
            int h = std::max(0, yy2 - yy1);
            float overlap = float(w * h) / (box1.area() + box2.area() - w * h);
            if (overlap > nmsThreshold_) {
                boxes_[idx[j]] = cv::Rect(0, 0, 0, 0);
            }
        }
    }
    for (int i = 0; i < boxes_.size(); i++) {
        if (boxes_[idx[i]].area() > 0) {
            results_.push_back(idx[i]);
        }
    }
}

/*autoaim_interfaces::msg::DetectionArray OpenVINOInferEngine::detect(cv::Mat& img_) {
    img = img_;
    start = chrono::steady_clock::now();
    img_preprocess();
    infer();
    img_postprocess();
    debug_draw_armors();
    end = chrono::steady_clock::now();
    chrono::duration<double> elapsed_seconds = end - start;
    // cout << "average time: " << elapsed_seconds.count() << "s" << endl;
    return armorMsg;
}*/

void OpenVINOInferEngine::img_postprocess() {
    // std::cout<<"img_postprocess"<<std::endl;
    const ov::Tensor& outputTensor = inferRequest.get_output_tensor();
    ov::Shape outputShape = outputTensor.get_shape();
    float* detections = outputTensor.data<float>();

    std::vector<cv::Rect> boxes;
    std::vector<Object> det;
    std::vector<int> colorIds, tagIds;
    std::vector<float> confidences;
    int outRows = outputShape[1];
    int outCols = outputShape[2];

    int areaThreshold = inputWidth * 0.1;

    const cv::Mat detOutput(outRows, outCols, CV_32F, (float*)detections);
    for (int i = 0; i < detOutput.rows; ++i) {
        // std::cout << "i " << i << " detOutput: " << detOutput.col(i).rowRange(0, outRows) << std::endl;
        const cv::Mat objectScore = detOutput.row(i).colRange(0, 1);
        if (objectScore.at<float>(0, 0) < confThreshold)
            continue;
        const cv::Mat colorScores = detOutput.row(i).colRange(1, 1 + numColor);
        const cv::Mat tagScores = detOutput.row(i).colRange(1 + numColor, 1 + numColor + numTag);
        cv::Point colorPoint, tagPoint;
        double colorScore, tagScore;
        minMaxLoc(colorScores, nullptr, &colorScore, nullptr, &colorPoint);
        minMaxLoc(tagScores, nullptr, &tagScore, nullptr, &tagPoint);
        // std::cout << "colorPoint " << colorPoint << " tagPoint " << tagPoint << std::endl;
        double score = std::min(colorScore, tagScore);
        const cv::Mat pose = detOutput.row(i).colRange(1 + numColor + numTag, outCols);
        if (score > confThreshold) {
            float tl_x = pose.at<float>(0, 0);
            float bl_x = pose.at<float>(0, 1);
            float br_x = pose.at<float>(0, 2);
            float tr_x = pose.at<float>(0, 3);
            float tl_y = pose.at<float>(0, 4);
            float bl_y = pose.at<float>(0, 5);
            float br_y = pose.at<float>(0, 6);
            float tr_y = pose.at<float>(0, 7);
            float cx = (tl_x + tr_x + bl_x + br_x) / 4;
            float cy = (tl_y + tr_y + bl_y + br_y) / 4;
            float ow = br_x - tl_x;
            float oh = br_y - tl_y;

            // 如果面积小于阈值，跳过
            cv::Rect box;
            box.x = static_cast<int>((cx - 0.5 * ow));
            box.y = static_cast<int>((cy - 0.5 * oh));
            box.width = static_cast<int>(ow);
            box.height = static_cast<int>(oh);

            boxes.push_back(box);
            colorIds.push_back(colorPoint.x);
            tagIds.push_back(tagPoint.x);
            confidences.push_back(score);

            Object detection;
            detection.box = box;
            detection.conf = score;
            detection.kpts[0] = cv::Point2f(tl_x, tl_y);
            detection.kpts[1] = cv::Point2f(bl_x, bl_y);
            detection.kpts[2] = cv::Point2f(br_x, br_y);
            detection.kpts[3] = cv::Point2f(tr_x, tr_y);
            det.push_back(detection);
        }
    }
    std::vector<int> result;
    nms(boxes, nmsThreshold, result); // Non-Maximum Suppression (NMS
    armorMsg.detections.clear();
    int colorMap[] = { COLOR::BLUE, COLOR::RED, COLOR::GRAY, COLOR::PURPLE };
    // std::cout<<"result.size "<<result.size()<<std::endl;
    for (int i = 0; i < result.size(); i++) {
        autoaim_interfaces::msg::Detection detection;

        detection.color = colorMap[colorIds[result[i]]];
        detection.label = tagIds[result[i]];
        // std::cout << "color " << detection.color << " label " << detection.label << std::endl;
        detection.confidence = confidences[result[i]];
        // std::cout << "ratio" << ratio << std::endl;

        detection.tl.x = det[result[i]].kpts[0].x * ratio;
        detection.tl.y = (det[result[i]].kpts[0].y + heightBias) * ratio;
        detection.bl.x = det[result[i]].kpts[1].x * ratio;
        detection.bl.y = (det[result[i]].kpts[1].y + heightBias) * ratio;
        detection.br.x = det[result[i]].kpts[2].x * ratio;
        detection.br.y = (det[result[i]].kpts[2].y + heightBias) * ratio;
        detection.tr.x = det[result[i]].kpts[3].x * ratio;
        detection.tr.y = (det[result[i]].kpts[3].y + heightBias) * ratio;
        armorMsg.detections.push_back(detection);
    }
}

cv::Mat OpenVINOInferEngine::debug_draw_armors() {
    for (int i = 0; i < armorMsg.detections.size(); i++) {
        autoaim_interfaces::msg::Detection& detection = armorMsg.detections[i];
        cv::Point2f kpts[4] { cv::Point2f(detection.tl.x, detection.tl.y),
                              cv::Point2f(detection.bl.x, detection.bl.y),
                              cv::Point2f(detection.br.x, detection.br.y),
                              cv::Point2f(detection.tr.x, detection.tr.y) };
        for (int j = 0; j < 4; j++) {
            kpts[j].y = kpts[j].y - ratio * heightBias;
        }
        // 不同颜色画出四个角点
        for (int j = 0; j < 4; j++) {
            circle(showImg, kpts[j], 3, pointColor[j], -1);
        }
        // 画出四个角点之间的两两连线
        for (int j = 0; j < 4; j++) {
            line(showImg, kpts[j], kpts[(j + 1) % 4], colors[detection.color], 2);
        }
        line(showImg, kpts[0], kpts[2], colors[detection.color], 2);
        line(showImg, kpts[1], kpts[3], colors[detection.color], 2);
        // 保留两位小数
        putText(
            showImg,
            name[detection.label] + " " + std::to_string(detection.confidence).substr(0, 4),
            cv::Point(kpts[0].x - 5, kpts[0].y - 15),
            cv::FONT_HERSHEY_TRIPLEX,
            0.8,
            cv::Scalar(255, 255, 255),
            1.5
        );
    }
    return showImg;
}

void OpenVINOInferEngine::set_input_image(const cv::Mat img) {
    this->img = img;
}

std::vector<autoaim_interfaces::msg::Detection> OpenVINOInferEngine::get_detection_vector() const {
    return armorMsg.detections;
}