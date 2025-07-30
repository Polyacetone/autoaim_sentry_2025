#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <hw_sentry_interfaces/msg/detections.hpp>

struct Detection {
    int color, label;
    float confidence;
    std::vector<cv::Point2f> keypoints;
};

class OpenVINOInferEngine {
public:
    OpenVINOInferEngine(
        const std::string& model_path,
        const std::string& device_name,
        const int num_colors,
        const int num_labels,
        const int num_keypoints,
        const float conf_threshold,
        const float nms_threshold
    );
    ~OpenVINOInferEngine() = default;

    std::vector<Detection> infer(const cv::Mat& input_image);

private:
    const int num_colors_, num_labels_, num_keypoints_;
    const float conf_threshold_, nms_threshold_;

    int input_image_width_;
    int input_image_height_;
    ov::InferRequest infer_request_;
    ov::CompiledModel compiled_model_;
};