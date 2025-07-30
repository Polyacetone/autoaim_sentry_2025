#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <hw_sentry_interfaces/msg/detections.hpp>
#include <hw_sentry_interfaces/msg/target_enemy.hpp>

#include <autoaim_common_definitions/common_definitions.hpp>
#include <autoaim_detector/openvino_infer_engine.hpp>

namespace autoaim_detector {
using namespace hw_sentry_interfaces::msg;

class DetectorNode: public rclcpp::Node {
public:
    explicit DetectorNode(const rclcpp::NodeOptions& options);

private:
    void img_callback(const sensor_msgs::msg::Image::SharedPtr msg);
    float get_framerate();
    cv::Mat draw_labeled_image(
        const cv::Mat& input_image,
        const std::variant<std::vector<ArmorDetection>, std::vector<BuffDetection>>& detections
    ) const;
    std::vector<ArmorDetection> to_armor_detections(const std::vector<Detection>& det) const;
    std::vector<BuffDetection> to_buff_detections(const std::vector<Detection>& det) const;

    std::string input_image_topic_;
    std::string target_enemy_topic_;
    std::string detections_topic_;
    std::string labeled_image_topic_;
    std::string armor_model_path_;
    std::string buff_model_path_;
    std::string device_name_;
    AutoaimMode mode_ = AutoaimMode::NONE;
    AutoaimMode default_mode_ = AutoaimMode::NONE;
    float decision_fallback_timeout_;
    bool enable_labeled_image_;
    bool enable_fps_;
    float confidence_threshold_;
    float nms_threshold_;

    std::chrono::high_resolution_clock::time_point prev_callback_time_;
    double last_recv_decision_time_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<TargetEnemy>::SharedPtr target_enemy_sub_;
    rclcpp::Publisher<Detections>::SharedPtr detections_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr labeled_image_pub_;

    std::unique_ptr<OpenVINOInferEngine> armor_infer_engine_;
    std::unique_ptr<OpenVINOInferEngine> buff_infer_engine_;
};

DetectorNode::DetectorNode(const rclcpp::NodeOptions& options): Node("autoaim_detector", options) {
    last_recv_decision_time_ = now().seconds();
    default_mode_ = static_cast<AutoaimMode>(declare_parameter<int>("default_mode"));
    decision_fallback_timeout_ = declare_parameter<float>("decision_fallback_timeout");
    input_image_topic_ = declare_parameter<std::string>("input_image_topic");
    target_enemy_topic_ = declare_parameter<std::string>("target_enemy_topic");
    detections_topic_ = declare_parameter<std::string>("detections_topic");
    labeled_image_topic_ = declare_parameter<std::string>("labeled_image_topic");
    armor_model_path_ = ament_index_cpp::get_package_share_directory("autoaim_detector")
        + "/model/" + declare_parameter<std::string>("armor_model_path");
    buff_model_path_ = ament_index_cpp::get_package_share_directory("autoaim_detector")
        + "/model/" + declare_parameter<std::string>("buff_model_path");
    device_name_ = declare_parameter<std::string>("device_name");
    enable_labeled_image_ = declare_parameter<bool>("enable_labeled_image");
    enable_fps_ = declare_parameter<bool>("enable_fps");
    confidence_threshold_ = declare_parameter<float>("confidence_threshold");
    nms_threshold_ = declare_parameter<float>("nms_threshold");

    armor_infer_engine_ = std::make_unique<OpenVINOInferEngine>(
        armor_model_path_,
        device_name_,
        3, 8, 4,
        confidence_threshold_,
        nms_threshold_
    );
    buff_infer_engine_ = std::make_unique<OpenVINOInferEngine>(
        buff_model_path_,
        device_name_,
        2, 2, 9,
        confidence_threshold_,
        nms_threshold_
    );
    
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        input_image_topic_,
        rclcpp::QoS(1),
        [&](const sensor_msgs::msg::Image::SharedPtr msg) { img_callback(msg); }
    );
    target_enemy_sub_ = create_subscription<TargetEnemy>(
        target_enemy_topic_,
        rclcpp::QoS(1),
        [&](const TargetEnemy::SharedPtr msg) {
            mode_ = static_cast<AutoaimMode>(msg->mode);
            last_recv_decision_time_ = now().seconds();
        }
    );
    detections_pub_ = create_publisher<Detections>(
        detections_topic_,
        rclcpp::QoS(1)
    );
    labeled_image_pub_ = create_publisher<sensor_msgs::msg::Image>(
        labeled_image_topic_,
        rclcpp::QoS(1)
    );
}

float DetectorNode::get_framerate() {
    auto current_time = std::chrono::high_resolution_clock::now();
    float duration = (current_time - prev_callback_time_).count() / 1e9;
    prev_callback_time_ = current_time;
    return 1 / duration;
}

void DetectorNode::img_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    if (now().seconds() - last_recv_decision_time_ > decision_fallback_timeout_) {
        mode_ = default_mode_;
    }
    if (enable_fps_) {
        RCLCPP_INFO(get_logger(), "Detection FPS: %.0f", get_framerate());
    }

    const auto cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    const cv::Mat img = cv_ptr->image;

    Detections detections;
    detections.mode = static_cast<int>(mode_);
    detections.label = -1;
    detections.header.stamp = msg->header.stamp;
    detections.header.frame_id = "autoaim_camera";

    if (mode_ == AutoaimMode::ARMOR) {
        const auto dets = armor_infer_engine_->infer(img);
        detections.armor_detections = to_armor_detections(dets);
        if (enable_labeled_image_) {
            sensor_msgs::msg::Image::SharedPtr labeled_image = cv_bridge::CvImage(
                msg->header,
                "bgr8",
                draw_labeled_image(img, detections.armor_detections)
            ).toImageMsg();
            labeled_image_pub_->publish(*labeled_image);
        }
    } else if (mode_ == AutoaimMode::SMALL_BUFF || mode_ == AutoaimMode::BIG_BUFF) {
        const auto dets = buff_infer_engine_->infer(img);
        detections.buff_detections = to_buff_detections(dets);
        if (enable_labeled_image_) {
            sensor_msgs::msg::Image::SharedPtr labeled_image = cv_bridge::CvImage(
                msg->header,
                "bgr8",
                draw_labeled_image(img, detections.buff_detections)
            ).toImageMsg();
            labeled_image_pub_->publish(*labeled_image);
        }
    }

    detections_pub_->publish(detections);
}

cv::Mat DetectorNode::draw_labeled_image(
    const cv::Mat& input_image,
    const std::variant<std::vector<ArmorDetection>, std::vector<BuffDetection>>& detections
) const {
    cv::Mat img = input_image.clone();
    const std::vector<std::string> armor_label =
        {"Sentry", "1", "2", "3", "4", "Outpost", "Base small", "Base big"};
    const std::vector<std::string> buff_label = {"Inactive", "Active"};
    const std::vector<cv::Scalar> colors =
        {cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255), cv::Scalar(114, 114, 114)};
    if (std::holds_alternative<std::vector<ArmorDetection>>(detections)) {
        const auto armor_detections = std::get<std::vector<ArmorDetection>>(detections);
        for (const auto& det: armor_detections) {
            cv::Point2f kpts[4] {
                cv::Point2f(det.tl.x, det.tl.y),
                cv::Point2f(det.bl.x, det.bl.y),
                cv::Point2f(det.br.x, det.br.y),
                cv::Point2f(det.tr.x, det.tr.y)
            };
            cv::line(img, kpts[0], kpts[1], colors[det.color], 2);
            cv::line(img, kpts[1], kpts[2], colors[det.color], 2);
            cv::line(img, kpts[2], kpts[3], colors[det.color], 2);
            cv::line(img, kpts[3], kpts[0], colors[det.color], 2);
            cv::line(img, kpts[0], kpts[2], colors[det.color], 1);
            cv::line(img, kpts[1], kpts[3], colors[det.color], 1);
            cv::drawMarker(img, kpts[0], cv::Scalar(255, 255, 0), cv::MARKER_DIAMOND, 4, 2);
            cv::drawMarker(img, kpts[1], cv::Scalar(255, 0, 255), cv::MARKER_DIAMOND, 4, 2);
            cv::drawMarker(img, kpts[2], cv::Scalar(0, 255, 255), cv::MARKER_DIAMOND, 4, 2);
            cv::drawMarker(img, kpts[3], cv::Scalar(0, 255, 0), cv::MARKER_DIAMOND, 4, 2);
            cv::putText(
                img,
                armor_label[det.label] + " " + std::to_string(det.confidence).substr(0, 4),
                cv::Point(kpts[0].x - 5, kpts[0].y - 15),
                cv::FONT_HERSHEY_TRIPLEX,
                0.7,
                cv::Scalar(255, 255, 255),
                1
            );
        }
    } else if (std::holds_alternative<std::vector<BuffDetection>>(detections)) {
        const auto buff_detections = std::get<std::vector<BuffDetection>>(detections);
        for (const auto& det: buff_detections) {
            cv::Point2f kpts[4] {
                cv::Point2f(det.t.x, det.t.y),
                cv::Point2f(det.l.x, det.l.y),
                cv::Point2f(det.b.x, det.b.y),
                cv::Point2f(det.r.x, det.r.y)
            };
            cv::line(img, kpts[0], kpts[1], colors[det.color], 2);
            cv::line(img, kpts[1], kpts[2], colors[det.color], 2);
            cv::line(img, kpts[2], kpts[3], colors[det.color], 2);
            cv::line(img, kpts[3], kpts[0], colors[det.color], 2);
            cv::drawMarker(img, kpts[0], cv::Scalar(255, 255, 0), cv::MARKER_DIAMOND, 4, 2);
            cv::drawMarker(img, kpts[1], cv::Scalar(255, 0, 255), cv::MARKER_DIAMOND, 4, 2);
            cv::drawMarker(img, kpts[2], cv::Scalar(0, 255, 255), cv::MARKER_DIAMOND, 4, 2);
            cv::drawMarker(img, kpts[3], cv::Scalar(0, 255, 0), cv::MARKER_DIAMOND, 4, 2);
            cv::putText(
                img,
                buff_label[det.label] + " " + std::to_string(det.confidence).substr(0, 4),
                cv::Point(kpts[0].x - 5, kpts[0].y - 15),
                cv::FONT_HERSHEY_TRIPLEX,
                0.7,
                cv::Scalar(255, 255, 255),
                1
            );
        }
    }
    return img;
}

std::vector<ArmorDetection> DetectorNode::to_armor_detections(const std::vector<Detection>& dets) const {
    std::vector<ArmorDetection> armor_dets;
    for (const auto& det: dets) {
        ArmorDetection armor_det;
        armor_det.color = det.color;
        armor_det.label = det.label;
        armor_det.confidence = det.confidence;
        armor_det.tl.x = det.keypoints[0].x;
        armor_det.tl.y = det.keypoints[0].y;
        armor_det.bl.x = det.keypoints[1].x;
        armor_det.bl.y = det.keypoints[1].y;
        armor_det.br.x = det.keypoints[2].x;
        armor_det.br.y = det.keypoints[2].y;
        armor_det.tr.x = det.keypoints[3].x;
        armor_det.tr.y = det.keypoints[3].y;
        armor_dets.emplace_back(armor_det);
    }
    return armor_dets;
}

std::vector<BuffDetection> DetectorNode::to_buff_detections(const std::vector<Detection>& dets) const {
    std::vector<BuffDetection> buff_dets;
    for (const auto& det: dets) {
        BuffDetection buff_det;
        buff_det.color = det.color;
        buff_det.label = det.label;
        buff_det.confidence = det.confidence;
        const cv::Point2f t = (det.keypoints[0] + det.keypoints[7]) / 2;
        const cv::Point2f l = (det.keypoints[1] + det.keypoints[2]) / 2;
        const cv::Point2f b = (det.keypoints[3] + det.keypoints[4]) / 2;
        const cv::Point2f r = (det.keypoints[5] + det.keypoints[6]) / 2;
        buff_det.t.x = t.x, buff_det.t.y = t.y;
        buff_det.l.x = l.x, buff_det.l.y = l.y;
        buff_det.b.x = b.x, buff_det.b.y = b.y;
        buff_det.r.x = r.x, buff_det.r.y = r.y;
        buff_dets.emplace_back(buff_det);
    }
    return buff_dets;
}
} // namespace autoaim_detector

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_detector::DetectorNode)