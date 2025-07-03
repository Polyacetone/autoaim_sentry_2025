#include <openvino_infer_engine.hpp>

OpenVINOInferEngine::OpenVINOInferEngine(
    const std::string& model_path,
    const std::string& device_name,
    const float conf_threshold,
    const float nms_threshold
) {
    conf_threshold_ = conf_threshold;
    nms_threshold_ = nms_threshold;

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
    compiled_model_ = core.compile_model(
        model,
        device_name,
        {ov::hint::inference_precision(ov::element::f16),
         ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}
    );
    infer_request_ = compiled_model_.create_infer_request();
    input_image_height_ = compiled_model_.input().get_shape()[1];
    input_image_width_ = compiled_model_.input().get_shape()[2];
}

void OpenVINOInferEngine::preprocess() {
    try {
        if (input_image_height_ != input_image_.rows || input_image_width_ != input_image_.cols) {
            throw std::runtime_error("input image size does not match model requirements");
        }
        ov::Tensor input_tensor = ov::Tensor(
            compiled_model_.input().get_element_type(),
            compiled_model_.input().get_shape(),
            input_image_.data
        );
        infer_request_.set_input_tensor(input_tensor);
    } catch (const std::exception& e) {
        std::cerr << "Exception in preprocess: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception in preprocess" << std::endl;
    }
}

void OpenVINOInferEngine::infer() {
    infer_request_.infer();
}

void OpenVINOInferEngine::postprocess() {
    const ov::Tensor& output_tensor = infer_request_.get_output_tensor();
    const ov::Shape output_shape = output_tensor.get_shape();
    const int out_rows = output_shape[1];
    const int out_cols = output_shape[2];
    if (out_cols != 36) {
        std::cerr << "Error in postprocess: output columns != 36" << std::endl;
        return;
    }
    const cv::Mat output_mat(out_rows, out_cols, CV_32F, output_tensor.data<float>());
    std::vector<hw_sentry_interfaces::msg::ArmorDetection> detections_before_nms;
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
        if (confidence < conf_threshold_) {
            continue;
        }
        confidences.emplace_back(confidence);

        const float box_x = row.at<float>(0, 0);
        const float box_y = row.at<float>(0, 1);
        const float box_w = row.at<float>(0, 2);
        const float box_h = row.at<float>(0, 3);
        boxes.emplace_back(box_x, box_y, box_w, box_h);

        hw_sentry_interfaces::msg::ArmorDetection detection;
        detection.confidence = confidence;
        detection.color = class_id / 8; // blue, red, gray
        detection.label = class_id % 8; // S, 1, 2, 3, 4, outpost, basesmall, basebig
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

    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, indices);
    armor_detections_.clear();
    for (const auto index: indices) {
        const auto det = detections_before_nms[index];
        // 过滤角点回归到画面外的情况
        if (det.tl.x < 0 || det.tl.y < 0)
            continue;
        if (det.tr.x > input_image_width_ || det.tr.y < 0)
            continue;
        if (det.bl.x < 0 || det.bl.y > input_image_height_)
            continue;
        if (det.br.x > input_image_width_ || det.br.y > input_image_height_)
            continue;
        armor_detections_.emplace_back(det);
    }
}

cv::Mat OpenVINOInferEngine::debug_draw_armors() const {
    cv::Mat image = input_image_.clone();
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
    for (const auto& detection: armor_detections_) {
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
            cv::FONT_HERSHEY_COMPLEX,
            0.7,
            cv::Scalar(255, 255, 255)
        );
    }
    return image;
}