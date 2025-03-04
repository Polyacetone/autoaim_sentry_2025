#include <infer_engine.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <autoaim_interfaces/msg/detection_array.hpp>

namespace autoaim_detection {
float get_fps() {
    static auto prev = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    float elapsed_sec = (now - prev).count() / 1e9;
    prev = now;
    return 1 / elapsed_sec;
}

class YoloDetectNode: public rclcpp::Node {
public:
    YoloDetectNode(const rclcpp::NodeOptions& options): Node("autoaim_detection", options) {
        get_parameters();

        RCLCPP_INFO(this->get_logger(), "初始化YOLO...");
        Config config = {model_path_, confidence_threshold_, nms_threshold_};
        infer_engine_ = create_infer_engine(config);
        RCLCPP_INFO(this->get_logger(), "初始化YOLO完成");
        // subscribe to image topic
        img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            img_topic_,
            10,
            std::bind(&YoloDetectNode::img_callback, this, std::placeholders::_1)
        );
        // pub
        detection_pub_ =
            this->create_publisher<autoaim_interfaces::msg::DetectionArray>(detection_topic_, 10);
        img_detected_pub_ =
            this->create_publisher<sensor_msgs::msg::Image>(img_detected_topic_, 10);
    }

private:
    std::string img_topic_;
    std::string detection_topic_;
    std::string img_detected_topic_;
    std::string model_path_;
    bool enable_detected_image_;
    bool enable_fps_;
    float confidence_threshold_;
    float nms_threshold_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
    rclcpp::Publisher<autoaim_interfaces::msg::DetectionArray>::SharedPtr detection_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_detected_pub_;

    std::unique_ptr<InferEngine> infer_engine_;

    void get_parameters() {
        img_topic_ = declare_parameter<std::string>("image_topic", "/camera/image_raw");
        detection_topic_ = declare_parameter<std::string>("detection_topic", "/detection");
        img_detected_topic_ =
            declare_parameter<std::string>("image_detected_topic", "/camera/color/image_detection");
        model_path_ = ament_index_cpp::get_package_share_directory("autoaim_detection") + "/model/"
            + declare_parameter<std::string>("model_path", "NULL");
        enable_detected_image_ = declare_parameter<bool>("enable_detected_image", false);
        enable_fps_ = declare_parameter<bool>("enable_fps", false);
        confidence_threshold_ = declare_parameter<float>("confidence_threshold", 0.5);
        nms_threshold_ = declare_parameter<float>("nms_threshold", 0.4);
    }

    void img_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        if (enable_fps_) {
            RCLCPP_INFO(get_logger(), "Detection FPS: %.0f", get_fps());
        }

        const auto cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        const cv::Mat img = cv_ptr->image;

        infer_engine_->set_input_image(img);
        infer_engine_->preprocess();
        infer_engine_->infer();
        infer_engine_->postprocess();
        auto detection_vec = infer_engine_->get_detection_arr();

        autoaim_interfaces::msg::DetectionArray detection_array;
        detection_array.detections = detection_vec;
        detection_array.header = msg->header;
        detection_pub_->publish(detection_array);

        if (enable_detected_image_) {
            sensor_msgs::msg::Image::SharedPtr img_detected =
                cv_bridge::CvImage(msg->header, "bgr8", infer_engine_->debug_draw_armors())
                    .toImageMsg();
            img_detected_pub_->publish(*img_detected);
        }
    }
};
} // namespace autoaim_detection

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_detection::YoloDetectNode)