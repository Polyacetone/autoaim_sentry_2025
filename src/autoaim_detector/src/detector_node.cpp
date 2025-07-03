#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <hw_sentry_interfaces/msg/detections.hpp>
#include <hw_sentry_interfaces/msg/target_enemy.hpp>

#include <openvino_infer_engine.hpp>

namespace autoaim_detector {
class DetectorNode: public rclcpp::Node {
public:
    explicit DetectorNode(const rclcpp::NodeOptions& options);

private:
    void img_callback(const sensor_msgs::msg::Image::SharedPtr msg);
    float get_framerate();

    std::string input_image_topic_;
    std::string target_enemy_topic_;
    std::string detections_topic_;
    std::string labeled_image_topic_;
    std::string model_path_;
    std::string device_name_;
    int mode_ = -1;
    bool enable_labeled_image_;
    bool enable_fps_;
    float confidence_threshold_;
    float nms_threshold_;

    std::chrono::high_resolution_clock::time_point prev_callback_time_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<hw_sentry_interfaces::msg::TargetEnemy>::SharedPtr target_enemy_sub_;
    rclcpp::Publisher<hw_sentry_interfaces::msg::Detections>::SharedPtr detections_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr labeled_image_pub_;

    std::unique_ptr<OpenVINOInferEngine> ov_infer_engine_;
};

DetectorNode::DetectorNode(const rclcpp::NodeOptions& options): Node("autoaim_detector", options) {
    mode_ = declare_parameter<int>("default_mode");
    input_image_topic_ = declare_parameter<std::string>("input_image_topic");
    target_enemy_topic_ = declare_parameter<std::string>("target_enemy_topic");
    detections_topic_ = declare_parameter<std::string>("detections_topic");
    labeled_image_topic_ = declare_parameter<std::string>("labeled_image_topic");
    model_path_ = ament_index_cpp::get_package_share_directory("autoaim_detector")
        + "/model/" + declare_parameter<std::string>("model_path");
    device_name_ = declare_parameter<std::string>("device_name");
    enable_labeled_image_ = declare_parameter<bool>("enable_labeled_image");
    enable_fps_ = declare_parameter<bool>("enable_fps");
    confidence_threshold_ = declare_parameter<float>("confidence_threshold");
    nms_threshold_ = declare_parameter<float>("nms_threshold");

    ov_infer_engine_ = std::make_unique<OpenVINOInferEngine>(
        model_path_,
        device_name_,
        confidence_threshold_,
        nms_threshold_
    );
    
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        input_image_topic_,
        rclcpp::QoS(1),
        [&](const sensor_msgs::msg::Image::SharedPtr msg) { img_callback(msg); }
    );
    target_enemy_sub_ = create_subscription<hw_sentry_interfaces::msg::TargetEnemy>(
        target_enemy_topic_,
        rclcpp::QoS(1),
        [&](const hw_sentry_interfaces::msg::TargetEnemy::SharedPtr msg) {
            mode_ = msg->mode;
        }
    );
    detections_pub_ = create_publisher<hw_sentry_interfaces::msg::Detections>(
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
    if (enable_fps_) {
        RCLCPP_INFO(get_logger(), "Detection FPS: %.0f", get_framerate());
    }

    const auto cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    const cv::Mat img = cv_ptr->image;

    hw_sentry_interfaces::msg::Detections detections;
    detections.mode = mode_;
    detections.label = -1;
    detections.header.stamp = msg->header.stamp;
    detections.header.frame_id = "autoaim_camera";

    if (mode_ == 0) {
        ov_infer_engine_->input_image_ = img.clone();
        ov_infer_engine_->preprocess();
        ov_infer_engine_->infer();
        ov_infer_engine_->postprocess();
        detections.armor_detections = ov_infer_engine_->armor_detections_;
        if (enable_labeled_image_) {
            sensor_msgs::msg::Image::SharedPtr img_detected =
                cv_bridge::CvImage(msg->header, "bgr8", ov_infer_engine_->debug_draw_armors())
                    .toImageMsg();
            labeled_image_pub_->publish(*img_detected);
        }
    }

    detections_pub_->publish(detections);
}
} // namespace autoaim_detector

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_detector::DetectorNode)