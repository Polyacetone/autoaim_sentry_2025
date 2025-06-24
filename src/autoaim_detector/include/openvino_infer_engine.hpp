#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <hw_sentry_interfaces/msg/detections.hpp>

class OpenVINOInferEngine {
public:
    explicit OpenVINOInferEngine(
        const std::string& model_path,
        const std::string& device_name,
        const float conf_threshold,
        const float nms_threshold
    );
    ~OpenVINOInferEngine() = default;

    void preprocess();
    void infer();
    void postprocess();
    cv::Mat debug_draw_armors();

    cv::Mat input_image_;
    std::vector<hw_sentry_interfaces::msg::ArmorDetection> armor_detections_;

private:
    float conf_threshold_;
    float nms_threshold_;

    int input_image_width_;
    int input_image_height_;
    ov::InferRequest infer_request_;
    ov::CompiledModel compiled_model_;
};