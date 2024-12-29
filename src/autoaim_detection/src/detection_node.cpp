#include <infer_engine.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <autoaim_interfaces/msg/comm_recv.hpp>
#include <autoaim_interfaces/msg/detection_array.hpp>

namespace autoaim_detection {
class YoloDetectNode: public rclcpp::Node {
public:
    YoloDetectNode(const rclcpp::NodeOptions& options): Node("autoaim_detection", options) {
        // 从launch文件中获取参数
        this->declare_parameter<std::string>("image_topic", "/camera/image_raw");
        this->declare_parameter<std::string>("camera_info_topic", "/camera/camera_info");
        this->declare_parameter<std::string>("detection_topic", "/detection");
        this->declare_parameter<std::string>("enemy_info_topic", "/enemy_info");
        this->declare_parameter<std::string>(
            "image_detected_topic",
            "/camera/color/image_detection"
        );
        this->declare_parameter<std::string>("onnx_path", "111");
        this->declare_parameter<bool>("enable_debug", false);
        this->declare_parameter<int>("num_color", 2);
        this->declare_parameter<int>("num_tag", 6);
        this->declare_parameter<float>("confidence_threshold", 0.5);
        this->declare_parameter<float>("nms_threshold", 0.5);

        img_topic_ = this->get_parameter("image_topic").as_string();
        cam_info_topic_ = this->get_parameter("camera_info_topic").as_string();
        detection_topic_ = this->get_parameter("detection_topic").as_string();
        img_detected_topic_ = this->get_parameter("image_detected_topic").as_string();
        onnx_path_ = ament_index_cpp::get_package_share_directory("autoaim_detection") + "/model/"
            + this->get_parameter("onnx_path").as_string();
        num_colors_ = this->get_parameter("num_color").as_int();
        num_tags_ = this->get_parameter("num_tag").as_int();
        confidence_threshold_ = this->get_parameter("confidence_threshold").as_double();
        nms_threshold_ = this->get_parameter("nms_threshold").as_double();
        enemy_info_topic_ = this->get_parameter("enemy_info_topic").as_string();
        enable_debug_ = this->get_parameter("enable_debug").as_bool();
 
        if (enable_debug_) {
            RCLCPP_INFO(this->get_logger(), "img_topic: %s", img_topic_.c_str());
            RCLCPP_INFO(this->get_logger(), "cam_info_topic: %s", cam_info_topic_.c_str());
            RCLCPP_INFO(this->get_logger(), "detection_topic: %s", detection_topic_.c_str());
            RCLCPP_INFO(
                this->get_logger(),
                "Debug enabled. img_detected_topic: %s",
                img_detected_topic_.c_str()
            );
        }

        auto parameter_change_cb =
            std::bind(&YoloDetectNode::parameter_callback, this, std::placeholders::_1);
        reset_param_handler_ = this->add_on_set_parameters_callback(parameter_change_cb);

        RCLCPP_INFO(this->get_logger(), "初始化YOLO...");
        Config config = { confidence_threshold_,
                          nms_threshold_,
                          num_colors_,
                          num_tags_,
                          onnx_path_ };
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
        sub_enemy_info_ = this->create_subscription<autoaim_interfaces::msg::CommRecv>(
            enemy_info_topic_,
            10,
            std::bind(&YoloDetectNode::enemy_info_callback, this, std::placeholders::_1)
        );
    }
    rcl_interfaces::msg::SetParametersResult
    parameter_callback(const std::vector<rclcpp::Parameter>& parameters) {
        auto result = rcl_interfaces::msg::SetParametersResult();
        result.successful = true;
        for (const auto& parameter: parameters) {
            RCLCPP_INFO(
                this->get_logger(),
                "Setting parameter '%s' to '%s'",
                parameter.get_name().c_str(),
                parameter.value_to_string().c_str()
            );
            if (parameter.get_name() == "image_topic") {
                img_topic_ = parameter.as_string();
                RCLCPP_INFO(this->get_logger(), "image_topic changed to %s", img_topic_.c_str());
            } else {
                RCLCPP_ERROR(
                    this->get_logger(),
                    "Parameter '%s' not defined",
                    parameter.get_name().c_str()
                );
                result.successful = false;
            }
        }
        return result;
    }

private:
    std::string img_topic_;
    std::string cam_info_topic_;
    std::string enemy_info_topic_;
    std::string detection_topic_;
    std::string img_detected_topic_;
    std::string onnx_path_;
    bool enable_debug_;
    int num_colors_;
    int num_tags_;
    int num_balance_;
    float confidence_threshold_;
    float nms_threshold_;
    int target_color_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
    rclcpp::Subscription<autoaim_interfaces::msg::CommRecv>::SharedPtr sub_enemy_info_;
    OnSetParametersCallbackHandle::SharedPtr reset_param_handler_;

    rclcpp::Publisher<autoaim_interfaces::msg::DetectionArray>::SharedPtr detection_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_detected_pub_;

    std::unique_ptr<InferEngine> infer_engine_;

    void enemy_info_callback(const autoaim_interfaces::msg::CommRecv::SharedPtr msg) {
        target_color_ = msg->target_color;
        num_balance_ = msg->balance_target_list;
    }

    void filter(std::vector<autoaim_interfaces::msg::Detection>& detection_vec) {
        for (auto it = detection_vec.begin(); it != detection_vec.end();) {
            bool wrongColor = it->color == COLOR::GRAY || it->color == COLOR::PURPLE
                || it->color != target_color_;
            if (wrongColor) {
                it = detection_vec.erase(it);
            } else {
                ++it;
            }
        }
    }

    void img_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        const auto cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        const cv::Mat img = cv_ptr->image;

        infer_engine_->set_input_image(img);
        infer_engine_->img_preprocess();
        infer_engine_->infer();
        infer_engine_->img_postprocess();
        auto detection_vec = infer_engine_->get_detection_vector();

        filter(detection_vec);
        autoaim_interfaces::msg::DetectionArray detection_array;
        detection_array.armors = detection_vec;
        detection_array.header = msg->header;
        detection_pub_->publish(detection_array);

        if (enable_debug_) {
            sensor_msgs::msg::Image::SharedPtr imgDetected =
                cv_bridge::CvImage(msg->header, "bgr8", infer_engine_->debug_draw_armors())
                    .toImageMsg();
            img_detected_pub_->publish(*imgDetected);
        }
    }
};
} // namespace autoaim_detection

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_detection::YoloDetectNode)