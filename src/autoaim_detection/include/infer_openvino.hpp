#include <omp.h>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include "rclcpp/rclcpp.hpp"
#include "cv_bridge/cv_bridge.h"
#include <hw_sentry_interfaces/msg/detection_array.hpp>

#include <check_armor.hpp>

struct Config {
    std::string model_path;
    float conf_threshold;
    float nms_threshold;
};

class OpenVINOInferEngine: public InferEngine {
public:
    OpenVINOInferEngine(Config config);
    ~OpenVINOInferEngine() = default;

    void preprocess() override;
    void infer() override;
    void postprocess() override;
    void set_input_image(const cv::Mat image) override;
    std::vector<hw_sentry_interfaces::msg::Detection> get_detection_arr() const override;
    cv::Mat debug_draw_armors() override;

private:
    std::string model_path;
    float conf_threshold;
    float nms_threshold;
    int input_image_width;
    int input_image_height;
    cv::Mat image;
    ov::InferRequest infer_request;
    ov::CompiledModel compiled_model;
    std::vector<hw_sentry_interfaces::msg::Detection> detection_arr;
};

OpenVINOInferEngine::OpenVINOInferEngine(Config config) {
    conf_threshold = config.conf_threshold;
    nms_threshold = config.nms_threshold;
    model_path = config.model_path;

    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(model_path);
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
    ppp.input()
        .tensor()
        .set_element_type(ov::element::u8)
        .set_layout("NHWC")
        .set_color_format(ov::preprocess::ColorFormat::BGR);
    ppp.input()
        .preprocess()
        .convert_element_type(ov::element::f16)
        .convert_color(ov::preprocess::ColorFormat::RGB)
        .scale({255, 255, 255});
    ppp.input().model().set_layout("NCHW");
    ppp.output().tensor().set_element_type(ov::element::f32);
    model = ppp.build();
    compiled_model =
        core.compile_model(model, "GPU", {
            ov::hint::inference_precision(ov::element::f16), 
            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)
        }
    );
    infer_request = compiled_model.create_infer_request();
    input_image_height = compiled_model.input().get_shape()[1];
    input_image_width = compiled_model.input().get_shape()[2];
}

void OpenVINOInferEngine::preprocess() {
    try {
        if (input_image_height != image.rows || input_image_width != image.cols) {
            throw std::runtime_error("input image size does not match model requirements");
        }
        ov::Tensor input_tensor = ov::Tensor(
            compiled_model.input().get_element_type(),
            compiled_model.input().get_shape(),
            image.data
        );
        infer_request.set_input_tensor(input_tensor);
    } catch (const std::exception& e) {
        std::cerr << "Exception in preprocess: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception in preprocess" << std::endl;
    }
}

void OpenVINOInferEngine::infer() {
    infer_request.infer();
}

void OpenVINOInferEngine::postprocess() {
    const ov::Tensor& output_tensor = infer_request.get_output_tensor();
    const ov::Shape output_shape = output_tensor.get_shape();
    const int out_rows = output_shape[1];
    const int out_cols = output_shape[2];
    if (out_cols != 36) {
        std::cerr << "Error in postprocess: output columns != 36" << std::endl;
        return;
    }
    const cv::Mat output_mat(out_rows, out_cols, CV_32F, output_tensor.data<float>());
    std::vector<hw_sentry_interfaces::msg::Detection> detections_before_nms;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> indices;
    for (int i = 0; i < out_rows; i++) {
        // 输出向量格式：x1, y1, x2, y2, 24个类别的confidence, 8个key_pts.xy
        const cv::Mat row = output_mat.row(i).colRange(0, 36);

        const cv::Mat scores = row.colRange(4, 28);
        double max_score;
        cv::Point max_point;
        cv::minMaxLoc(scores, nullptr, &max_score, nullptr, &max_point);
        const float confidence = static_cast<float>(max_score);
        const int class_id = max_point.x;
        if (confidence < conf_threshold) {
            continue;
        }
        confidences.emplace_back(confidence);

        const float box_x = row.at<float>(0, 0);
        const float box_y = row.at<float>(0, 1);
        const float box_w = row.at<float>(0, 2);
        const float box_h = row.at<float>(0, 3);
        boxes.emplace_back(cv::Rect(box_x, box_y, box_w, box_h));

        hw_sentry_interfaces::msg::Detection detection;
        detection.confidence = confidence;
        detection.color = class_id % 3; // blue, red, gray
        detection.label = class_id / 3; // S, 1, 2, 3, 4, outpost, basesmall, basebig
        detection.tl.x = row.at<float>(0, 28);
        detection.tl.y = row.at<float>(0, 29);
        detection.bl.x = row.at<float>(0, 30);
        detection.bl.y = row.at<float>(0, 31);
        detection.br.x = row.at<float>(0, 32);
        detection.br.y = row.at<float>(0, 33);
        detection.tr.x = row.at<float>(0, 34);
        detection.tr.y = row.at<float>(0, 35);
        detections_before_nms.emplace_back(detection);
    }

    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);
    detection_arr.clear();
    for(const auto index: indices) {
        detection_arr.emplace_back(detections_before_nms[index]);
    }
}

void OpenVINOInferEngine::set_input_image(const cv::Mat image) {
    this->image = image;
}

std::vector<hw_sentry_interfaces::msg::Detection> OpenVINOInferEngine::get_detection_arr() const {
    return detection_arr;
}

cv::Mat OpenVINOInferEngine::debug_draw_armors() {
    const std::vector<std::string> name = {
        "Sentry",
        "1",
        "2",
        "3",
        "4",
        "Outpost",
        "Base small",
        "Base big",
    };
    // blue, red, gray
    const std::vector<cv::Scalar> colors =
        {cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255), cv::Scalar(114, 114, 114)};
    for (const auto& detection: detection_arr) {
        cv::Point2f kpts[4] {
            cv::Point2f(detection.tl.x, detection.tl.y),
            cv::Point2f(detection.bl.x, detection.bl.y),
            cv::Point2f(detection.br.x, detection.br.y),
            cv::Point2f(detection.tr.x, detection.tr.y)
        };
        for (int j = 0; j < 4; j++) {
            line(image, kpts[j], kpts[(j + 1) % 4], colors[detection.color], 1);
        }
        line(image, kpts[0], kpts[2], colors[detection.color], 1);
        line(image, kpts[1], kpts[3], colors[detection.color], 1);
        putText(
            image,
            name[detection.label] + " " + std::to_string(detection.confidence).substr(0, 4),
            cv::Point(kpts[0].x - 5, kpts[0].y - 15),
            cv::FONT_HERSHEY_TRIPLEX,
            0.8,
            cv::Scalar(255, 255, 255),
            1
        );
    }
    return image;
}