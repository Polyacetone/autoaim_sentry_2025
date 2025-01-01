#include <string>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include <autoaim_interfaces/msg/detection_array.hpp>
#include <pnp.hpp>

using autoaim_interfaces::msg::DetectionArray;

namespace autoaim_prediction {
class PredictionNode: public rclcpp::Node {
public:
    explicit PredictionNode(const rclcpp::NodeOptions& options);
    ~PredictionNode() = default;

private:
    void get_parameters();
    void detection_callback(const DetectionArray::SharedPtr msg);
    void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

    bool enable_debug_;
    bool debug_mode_;
    bool debug_enable_predict_;
    int debug_target_color_;
    int debug_buff_mode_;
    float armor_dir_angle_;
    float filter_distance_;
    float imu_compensate_pitch_;
    float imu_compensate_yaw_;
    float t_delay_;
    std::string camera_info_topic_;
    std::string detection_sub_topic_;
    std::string comm_pub_topic_;
    std::string position_pub_topic_;

    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> broadcaster_;

    std::shared_ptr<rclcpp::Subscription<DetectionArray>> detection_sub_;
    std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>> camera_info_sub_;
};

PredictionNode::PredictionNode(const rclcpp::NodeOptions& options):
    Node("autoaim_prediction", options) {
    static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    get_parameters();

    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic_,
        10,
        [&](const sensor_msgs::msg::CameraInfo::SharedPtr msg) { camera_info_callback(msg); }
    );
    detection_sub_ = create_subscription<DetectionArray>(
        detection_sub_topic_,
        10,
        [&](const DetectionArray::SharedPtr msg) { detection_callback(msg); }
    );
}

void PredictionNode::get_parameters() {
    enable_debug_ = declare_parameter("enable_debug", false);
    debug_mode_ = declare_parameter("debug_mode", false);
    debug_enable_predict_ = declare_parameter("debug_enable_predict", false);
    debug_target_color_ = declare_parameter("debug_target_color", 0);
    debug_buff_mode_ = declare_parameter("debug_buff_mode", 1);
    armor_dir_angle_ = declare_parameter("armor_dir_angle", 0.2618);
    filter_distance_ = declare_parameter("filter_distance", 6000.0);
    imu_compensate_pitch_ = declare_parameter("imu_compensate_pitch", 1.3);
    imu_compensate_yaw_ = declare_parameter("imu_compensate_yaw", 0.3);
    t_delay_ = declare_parameter("t_delay", 98.0);
    camera_info_topic_ = declare_parameter("camera_info_topic", "/camera/color/camera_info");
    detection_sub_topic_ = declare_parameter("detection_sub_topic", "/detection");
    comm_pub_topic_ = declare_parameter("comm_pub_topic", "/serial/comm_send");
    position_pub_topic_ = declare_parameter("position_pub_topic", "/debug/position");

    const double cam_to_spindle_x = declare_parameter("cam_to_spindle_x", 65.8);
    const double cam_to_spindle_y = declare_parameter("cam_to_spindle_y", 0.0);
    const double cam_to_spindle_z = declare_parameter("cam_to_spindle_z", 63.9);

    geometry_msgs::msg::TransformStamped transform_to_fric;
    transform_to_fric.header.stamp = this->now();
    transform_to_fric.header.frame_id = "spindle";
    transform_to_fric.child_frame_id = "autoaim_camera";
    transform_to_fric.transform.translation.x = cam_to_spindle_x;
    transform_to_fric.transform.translation.y = cam_to_spindle_y;
    transform_to_fric.transform.translation.z = cam_to_spindle_z;
    transform_to_fric.transform.rotation.x = 0;
    transform_to_fric.transform.rotation.y = 0;
    transform_to_fric.transform.rotation.z = 0;
    transform_to_fric.transform.rotation.w = 1;
    static_broadcaster_->sendTransform(transform_to_fric);
}

void PredictionNode::detection_callback(const DetectionArray::SharedPtr msg) {
    if (enable_debug_) {
        for (const auto& detection: msg->detections) {
            if (debug_target_color_ == 0 || detection.color == debug_target_color_) {
                geometry_msgs::msg::TransformStamped transform_to_armor;
                transform_to_armor.header.stamp = this->now();
                transform_to_armor.header.frame_id = "autoaim_camera";
                transform_to_armor.child_frame_id = std::to_string(detection.color*10 + detection.label);
                pnp::solve_pnp(detection, transform_to_armor.transform);
                broadcaster_->sendTransform(transform_to_armor);
            }
        }
    }
}

void PredictionNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    pnp::cam_intrinsic = cv::Mat(3, 3, CV_64F, const_cast<double *>(msg->k.data())).clone();
    pnp::cam_distortion = cv::Mat(1, 5, CV_64F, const_cast<double *>(msg->d.data())).clone();
    // 相机内参和畸变在运行中不会改变，所以设置后即可取消camera_info订阅
    camera_info_sub_.reset();
    camera_info_sub_ = nullptr;
}
} // namespace autoaim_prediction

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_prediction::PredictionNode)