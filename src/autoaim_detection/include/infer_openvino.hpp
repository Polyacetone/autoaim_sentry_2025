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
    if (out_cols != 23) {
        std::cerr << "Error in postprocess: output columns != 23" << std::endl;
        return;
    }
    const cv::Mat output_mat(out_rows, out_cols, CV_32F, output_tensor.data<float>());
    using Point = geometry_msgs::msg::Point32;
    using DetectionOutput = std::tuple<int, int, Point, Point, Point, Point>; // color, label, key_points
    std::vector<cv::Rect> color_boxes, label_boxes;
    std::vector<float> color_confidences, label_confidences;
    std::vector<DetectionOutput> color_detections, label_detections;

    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < out_rows; i++) {
        // 输出向量格式：x1, y1, x2, y2, 3个颜色(b,r,g)的置信度分数, 8个类别的置信度分数, 8个key_pts.xy
        const cv::Mat row = output_mat.row(i).colRange(0, 23);

        const cv::Mat color_scores = row.colRange(4, 7);
        const cv::Mat label_scores = row.colRange(7, 15);
        double max_color_score, max_label_score;
        cv::Point max_color_point, max_label_point;
        cv::minMaxLoc(color_scores, nullptr, &max_color_score, nullptr, &max_color_point);
        cv::minMaxLoc(label_scores, nullptr, &max_label_score, nullptr, &max_label_point);
        if (max_color_score < conf_threshold && max_label_score < conf_threshold) {
            continue;
        }

        const float box_x = row.at<float>(0, 0);
        const float box_y = row.at<float>(0, 1);
        const float box_w = row.at<float>(0, 2);
        const float box_h = row.at<float>(0, 3);
        DetectionOutput detection;
        std::get<0>(detection) = max_color_point.x;
        std::get<1>(detection) = max_label_point.x;
        std::get<2>(detection).x = row.at<float>(0, 15);
        std::get<2>(detection).y = row.at<float>(0, 16);
        std::get<3>(detection).x = row.at<float>(0, 17);
        std::get<3>(detection).y = row.at<float>(0, 18);
        std::get<4>(detection).x = row.at<float>(0, 19);
        std::get<4>(detection).y = row.at<float>(0, 20);
        std::get<5>(detection).x = row.at<float>(0, 21);
        std::get<5>(detection).y = row.at<float>(0, 22);

        #pragma omp critical
        if (max_color_score > max_label_score) { // 是一个颜色框
            color_boxes.emplace_back(cv::Rect(box_x, box_y, box_w, box_h));
            color_confidences.emplace_back(max_color_score);
            color_detections.emplace_back(detection);
        } else { // 是一个类别框
            label_boxes.emplace_back(cv::Rect(box_x, box_y, box_w, box_h));
            label_confidences.emplace_back(max_label_score);
            label_detections.emplace_back(detection);
        }
    }

    std::vector<int> color_indices, label_indices;
    cv::dnn::NMSBoxes(color_boxes, color_confidences, conf_threshold, nms_threshold, color_indices);
    cv::dnn::NMSBoxes(label_boxes, label_confidences, conf_threshold, nms_threshold, label_indices);

    constexpr auto distance_squred = [](const Point& a, const Point& b) -> float {
        return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
    };
    constexpr auto middle_point = [](const Point& a, const Point& b) -> Point {
        Point p;
        p.x = (a.x + b.x) / 2;
        p.y = (a.y + b.y) / 2;
        return p;
    };
    detection_arr.clear();
    for (const int color_index: color_indices) {
        for (const int label_index: label_indices) {
            const float difference = distance_squred(
                std::get<2>(color_detections[color_index]),
                std::get<2>(label_detections[label_index])
            ) + distance_squred(
                std::get<4>(color_detections[color_index]),
                std::get<4>(label_detections[label_index])
            );
            if (difference < 20) {
                hw_sentry_interfaces::msg::Detection detection;
                detection.color = std::get<0>(color_detections[color_index]);
                detection.label = std::get<1>(label_detections[label_index]);
                detection.confidence = std::min(color_confidences[color_index], label_confidences[label_index]);
                detection.tl = middle_point(std::get<2>(color_detections[color_index]), std::get<2>(label_detections[label_index]));
                detection.bl = middle_point(std::get<3>(color_detections[color_index]), std::get<3>(label_detections[label_index]));
                detection.br = middle_point(std::get<4>(color_detections[color_index]), std::get<4>(label_detections[label_index]));
                detection.tr = middle_point(std::get<5>(color_detections[color_index]), std::get<5>(label_detections[label_index]));
                detection_arr.emplace_back(detection);
                break;
            }
        }
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