#include <autoaim_detector/openvino_infer_engine.hpp>

OpenVINOInferEngine::OpenVINOInferEngine(
    const std::string& model_path,
    const std::string& device_name,
    const int num_colors,
    const int num_labels,
    const int num_keypoints,
    const float conf_threshold,
    const float nms_threshold
):  
    num_colors_(num_colors), num_labels_(num_labels), num_keypoints_(num_keypoints),
    conf_threshold_(conf_threshold), nms_threshold_(nms_threshold) {
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
    const int output_cols = compiled_model_.output().get_shape()[2];
    if (output_cols != (4 + num_colors_ * num_labels_ + num_keypoints_ * 2)) {
        throw std::runtime_error("invalid output tensor shape");
    }
}

std::vector<Detection> OpenVINOInferEngine::infer(const cv::Mat& input_image) {
    if (input_image_height_ != input_image.rows || input_image_width_ != input_image.cols) {
        throw std::runtime_error("invalid input image size");
    }
    ov::Tensor input_tensor = ov::Tensor(
        compiled_model_.input().get_element_type(),
        compiled_model_.input().get_shape(),
        input_image.data
    );
    infer_request_.set_input_tensor(input_tensor);
    infer_request_.infer();
    const ov::Tensor& output_tensor = infer_request_.get_output_tensor();
    const ov::Shape output_shape = output_tensor.get_shape();
    const int out_rows = output_shape[1];
    const int out_cols = output_shape[2];
    std::vector<cv::Rect> bboxes;
    std::vector<float> confidences;
    std::vector<Detection> detections_before_nms;
    for (int i = 0; i < out_rows; i++) {
        const std::span<float> row(output_tensor.data<float>() + i * out_cols, out_cols);
        const auto max_conf = std::max_element(row.begin() + 4, row.begin() + 4 + num_colors_ * num_labels_);
        const cv::Rect bbox(row[0], row[1], row[2], row[3]);
        const float confidence = *max_conf;
        if (confidence < conf_threshold_) continue;
        const int color = (max_conf - row.begin() - 4) / num_labels_;
        const int label = (max_conf - row.begin() - 4) % num_labels_;
        std::vector<cv::Point2f> keypoints;
        for (int j = 0; j < num_keypoints_; j++) {
            keypoints.emplace_back(
                row[4 + num_colors_ * num_labels_ + j * 2],
                row[4 + num_colors_ * num_labels_ + j * 2 + 1]
            );
        }
        bboxes.emplace_back(bbox);
        confidences.emplace_back(confidence);
        detections_before_nms.emplace_back(color, label, confidence, keypoints);
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(bboxes, confidences, conf_threshold_, nms_threshold_, indices);
    std::vector<Detection> detections;
    for (const int index: indices) {
        const auto det = detections_before_nms[index];
        bool is_valid = true;
        for (int i = 0; i < num_keypoints_; i++) {
            if (det.keypoints[i].x > input_image_width_ || det.keypoints[i].x < 0) {
                is_valid = false;
                break;
            }
            if (det.keypoints[i].y > input_image_height_ || det.keypoints[i].y < 0) {
                is_valid = false;
                break;
            }
        }
        if (is_valid) detections.emplace_back(detections_before_nms[index]);
    }
    return detections;
}