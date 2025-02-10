#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include "rclcpp/rclcpp.hpp"
#include "cv_bridge/cv_bridge.h"
#include <autoaim_interfaces/msg/detection_array.hpp>

enum COLOR { GRAY = 0, BLUE = 1, RED = 2, PURPLE = 3 };

struct Config {
    float conf_threshold;
    float nms_threshold;
    int num_colors;
    int num_tags;
    std::string onnx_path;
};

class OpenVINOInferEngine: public InferEngine {
public:
    OpenVINOInferEngine(Config config);
    ~OpenVINOInferEngine() = default;

    void preprocess() override;
    void infer() override;
    void postprocess() override;
    void set_input_image(const cv::Mat image) override;
    std::vector<autoaim_interfaces::msg::Detection> get_detection_vector() const override;
    cv::Mat debug_draw_armors() override;

private:
    float conf_threshold;
    float nms_threshold;
    int num_colors;
    int num_tags;
    int input_image_width;
    int input_image_height;
    std::string onnx_path;
    cv::Mat image;
    ov::Tensor input_tensor;
    ov::InferRequest infer_request;
    ov::CompiledModel compiled_model;
    autoaim_interfaces::msg::DetectionArray detection_arr_msg;

    void nms(std::vector<cv::Rect>& boxes, std::vector<int>& results);
    void model_initialize();
};

OpenVINOInferEngine::OpenVINOInferEngine(Config config) {
    conf_threshold = config.conf_threshold;
    nms_threshold = config.nms_threshold;
    num_colors = config.num_colors;
    num_tags = config.num_tags;
    onnx_path = config.onnx_path;
    model_initialize();
}

void OpenVINOInferEngine::model_initialize() {
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(onnx_path);
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
        .scale({255, 255, 255});
    ppp.input().model().set_layout("NCHW");
    ppp.output().tensor().set_element_type(ov::element::f32);
    model = ppp.build();
    compiled_model = core.compile_model(model, "GPU");
    infer_request = compiled_model.create_infer_request();
    input_image_width = compiled_model.input().get_shape()[2];
    input_image_height = compiled_model.input().get_shape()[1];
}

void OpenVINOInferEngine::preprocess() {
    try {
        if (input_image_height != image.rows || input_image_width != image.cols) {
            std::printf("输入图像长宽与模型要求不匹配。\n");
            std::printf(
                "输入图像大小：(%d, %d)，模型要求：(%d, %d)\n",
                image.cols,
                image.rows,
                input_image_width,
                input_image_height
            );
        }
        input_tensor = ov::Tensor(
            compiled_model.input().get_element_type(),
            compiled_model.input().get_shape(),
            (float*)image.data
        );
        infer_request.set_input_tensor(input_tensor);
    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "unknown exception" << std::endl;
    }
}

void OpenVINOInferEngine::infer() {
    infer_request.infer();
}

void OpenVINOInferEngine::nms(
    std::vector<cv::Rect>& boxes,
    std::vector<int>& results
) {
    std::vector<int> idx(boxes.size());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&boxes](int i1, int i2) { return boxes[i1].x < boxes[i2].x; });
    for (int i = 0; i < boxes.size(); i++) {
        cv::Rect box1 = boxes[idx[i]];
        if (box1.area() == 0)
            continue;
        for (int j = i + 1; j < boxes.size(); j++) {
            cv::Rect box2 = boxes[idx[j]];
            if (box2.area() == 0)
                continue;
            const int xx1 = std::max(box1.x, box2.x);
            const int yy1 = std::max(box1.y, box2.y);
            const int xx2 = std::min(box1.x + box1.width, box2.x + box2.width);
            const int yy2 = std::min(box1.y + box1.height, box2.y + box2.height);
            const int w = std::max(0, xx2 - xx1);
            const int h = std::max(0, yy2 - yy1);
            const float overlap = float(w * h) / (box1.area() + box2.area() - w * h);
            if (overlap > nms_threshold) {
                boxes[idx[j]] = cv::Rect(0, 0, 0, 0);
            }
        }
    }
    for (int i = 0; i < boxes.size(); i++) {
        if (boxes[idx[i]].area() > 0) {
            results.push_back(idx[i]);
        }
    }
}

void OpenVINOInferEngine::postprocess() {
    const ov::Tensor& output_tensor = infer_request.get_output_tensor();
    const ov::Shape output_shape = output_tensor.get_shape();
    const int out_rows = output_shape[1];
    const int out_cols = output_shape[2];
    const cv::Mat detOutput(out_rows, out_cols, CV_32F, output_tensor.data<float>());

    std::vector<cv::Rect> boxes;
    std::vector<autoaim_interfaces::msg::Detection> detections;

    for (int i = 0; i < detOutput.rows; i++) {
        const cv::Mat objectScore = detOutput.row(i).colRange(0, 1);
        if (objectScore.at<float>(0, 0) < conf_threshold)
            continue;
        const cv::Mat colorScores = detOutput.row(i).colRange(1, 1 + num_colors);
        const cv::Mat tagScores = detOutput.row(i).colRange(1 + num_colors, 1 + num_colors + num_tags);
        cv::Point colorPoint, tagPoint;
        double colorScore, tagScore;
        minMaxLoc(colorScores, nullptr, &colorScore, nullptr, &colorPoint);
        minMaxLoc(tagScores, nullptr, &tagScore, nullptr, &tagPoint);
        double score = std::min(colorScore, tagScore);
        if (score < conf_threshold)
            continue;
        const cv::Mat pose = detOutput.row(i).colRange(1 + num_colors + num_tags, out_cols);
        const float tl_x = pose.at<float>(0, 0);
        const float bl_x = pose.at<float>(0, 1);
        const float br_x = pose.at<float>(0, 2);
        const float tr_x = pose.at<float>(0, 3);
        const float tl_y = pose.at<float>(0, 4);
        const float bl_y = pose.at<float>(0, 5);
        const float br_y = pose.at<float>(0, 6);
        const float tr_y = pose.at<float>(0, 7);
        const float cx = (tl_x + tr_x + bl_x + br_x) / 4;
        const float cy = (tl_y + tr_y + bl_y + br_y) / 4;
        const float ow = br_x - tl_x;
        const float oh = br_y - tl_y;

        cv::Rect box;
        box.x = static_cast<int>(cx - 0.5 * ow);
        box.y = static_cast<int>(cy - 0.5 * oh);
        box.width = static_cast<int>(ow);
        box.height = static_cast<int>(oh);
        boxes.push_back(box);

        constexpr int colorMap[] = {COLOR::BLUE, COLOR::RED, COLOR::GRAY, COLOR::PURPLE};
        autoaim_interfaces::msg::Detection detection;
        detection.confidence = score;
        detection.label = tagPoint.x;
        detection.color = colorMap[colorPoint.x];
        detection.tl.x = tl_x;
        detection.tl.y = tl_y;
        detection.bl.x = bl_x;
        detection.bl.y = bl_y;
        detection.br.x = br_x;
        detection.br.y = br_y;
        detection.tr.x = tr_x;
        detection.tr.y = tr_y;
        detections.emplace_back(detection);
    }
    std::vector<int> result;
    nms(boxes, result);
    detection_arr_msg.detections.clear();
    for (const int index: result) {
        detection_arr_msg.detections.push_back(detections[index]);
    }
}

void OpenVINOInferEngine::set_input_image(const cv::Mat image) {
    this->image = image;
}

std::vector<autoaim_interfaces::msg::Detection> OpenVINOInferEngine::get_detection_vector() const {
    return detection_arr_msg.detections;
}

cv::Mat OpenVINOInferEngine::debug_draw_armors() {
    const std::vector<std::string> name = {
        "Sentry",
        "Hero",
        "Engineer",
        "Infantry_3",
        "Infantry_4",
        "Infantry_5",
        "OutPost",
        "Small Base",
        "Big Base",
    };
    // gray blue red purple
    const std::vector<cv::Scalar> colors = {
        cv::Scalar(114, 114, 114),
        cv::Scalar(255, 0, 0),
        cv::Scalar(0, 0, 255),
        cv::Scalar(255, 0, 255),
    };
    for (const auto& detection: detection_arr_msg.detections) {
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