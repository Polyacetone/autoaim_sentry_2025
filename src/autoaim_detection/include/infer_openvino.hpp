#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include "rclcpp/rclcpp.hpp"
#include "cv_bridge/cv_bridge.h"
#include <autoaim_interfaces/msg/detection_array.hpp>

#include <check_armor.hpp>

struct Config {
    std::string onnx_path;
    float conf_threshold;
};

class OpenVINOInferEngine: public InferEngine {
public:
    OpenVINOInferEngine(Config config);
    ~OpenVINOInferEngine() = default;

    void preprocess() override;
    void infer() override;
    void postprocess() override;
    void set_input_image(const cv::Mat image) override;
    std::vector<autoaim_interfaces::msg::Detection> get_detection_arr() const override;
    cv::Mat debug_draw_armors() override;

private:
    std::string onnx_path;
    float conf_threshold;
    int input_image_width;
    int input_image_height;
    cv::Mat image;
    ov::InferRequest infer_request;
    ov::CompiledModel compiled_model;
    std::vector<autoaim_interfaces::msg::Detection> detection_arr;
};

OpenVINOInferEngine::OpenVINOInferEngine(Config config) {
    conf_threshold = config.conf_threshold;
    onnx_path = config.onnx_path;

    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(onnx_path);
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
    ppp.input()
        .tensor()
        .set_element_type(ov::element::u8)
        .set_layout("NHWC")
        .set_color_format(ov::preprocess::ColorFormat::BGR);
    ppp.input()
        .preprocess()
        .convert_element_type(ov::element::f32)
        .convert_color(ov::preprocess::ColorFormat::RGB)
        .scale({255, 255, 255});
    ppp.input().model().set_layout("NCHW");
    ppp.output().tensor().set_element_type(ov::element::f32);
    model = ppp.build();
    compiled_model = core.compile_model(model, "GPU", {ov::hint::inference_precision(ov::element::f32)});
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
    if (out_cols != 14) {
        std::cerr << "Error in postprocess: output columns != 14" << std::endl;
        return;
    }
    const cv::Mat output_mat(out_rows, out_cols, CV_32F, output_tensor.data<float>());
    detection_arr.clear();
    for (int row = 0; row < out_rows; row++) {
        // 输出向量格式：x1, y1, x2, y2, conf, class_id, key_pts
        const float confidence = output_mat.at<float>(row, 4);
        if (confidence < conf_threshold) {
            continue;
        }
        static int x = 0;
        x++;
        std::cout << x << ": ";
        for(int j=0; j<14; j++) {
            std::printf("%.2f ", output_mat.at<float>(row, j));
        }
        std::printf("\n");
        const int class_id = output_mat.at<float>(row, 5);
        autoaim_interfaces::msg::Detection detection;
        detection.confidence = confidence;
        detection.color = class_id % 3; // blue, red, gray
        detection.label = class_id / 3; // S, 1, 2, 3, 4, outpost, basesmall, basebig
        detection.tl.x = output_mat.at<float>(row, 6);
        detection.tl.y = output_mat.at<float>(row, 7);
        detection.bl.x = output_mat.at<float>(row, 8);
        detection.bl.y = output_mat.at<float>(row, 9);
        detection.br.x = output_mat.at<float>(row, 10);
        detection.br.y = output_mat.at<float>(row, 11);
        detection.tr.x = output_mat.at<float>(row, 12);
        detection.tr.y = output_mat.at<float>(row, 13);
        detection_arr.emplace_back(detection);
    }
}

void OpenVINOInferEngine::set_input_image(const cv::Mat image) {
    this->image = image;
}

std::vector<autoaim_interfaces::msg::Detection> OpenVINOInferEngine::get_detection_arr() const {
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