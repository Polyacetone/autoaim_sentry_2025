#include <rclcpp/rclcpp.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/convert.hpp>
#include <tf2/utils.hpp>
#include <tf2/time.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <sensor_msgs/msg/camera_info.hpp>
#include <hw_sentry_interfaces/msg/detections.hpp>
#include <hw_sentry_interfaces/msg/poses.hpp>

#include <pnp_solver.hpp>
#include <autoaim_common_utils/tf_utils.hpp>
#include <autoaim_common_utils/convert_utils.hpp>

namespace autoaim_locator {
using namespace hw_sentry_interfaces::msg;
using namespace sensor_msgs::msg;
using namespace geometry_msgs::msg;

class LocatorNode: public rclcpp::Node {
public:
    explicit LocatorNode(const rclcpp::NodeOptions& options);
    ~LocatorNode() = default;

private:
    void camera_info_callback(const CameraInfo::SharedPtr msg);
    void detections_callback(const Detections::SharedPtr msg);
    Poses solve_armor_detections(const Detections::SharedPtr msg);

    std::unique_ptr<PnPSolver> pnp_solver_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

    rclcpp::Subscription<CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Subscription<Detections>::SharedPtr selected_detections_sub_;
    rclcpp::Publisher<Poses>::SharedPtr poses_pub_;
};

LocatorNode::LocatorNode(const rclcpp::NodeOptions& options): Node("autoaim_locator", options) {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    pnp_solver_ = std::make_unique<PnPSolver>();
    
    std::string camera_info_topic = declare_parameter<std::string>("camera_info_topic");
    std::string selected_detections_topic = declare_parameter<std::string>("selected_detections_topic");
    std::string poses_topic = declare_parameter<std::string>("poses_topic");
    
    camera_info_sub_ = create_subscription<CameraInfo>(
        camera_info_topic,
        rclcpp::QoS(1),
        [&](const CameraInfo::SharedPtr msg) { camera_info_callback(msg); }
    );
    selected_detections_sub_ = create_subscription<Detections>(
        selected_detections_topic,
        rclcpp::QoS(1),
        [&](const Detections::SharedPtr msg) { detections_callback(msg); }
    );

    poses_pub_ = create_publisher<Poses>(
        poses_topic,
        rclcpp::QoS(1)
    );
}

void LocatorNode::detections_callback(const Detections::SharedPtr msg) {
    Poses poses;
    if (msg->mode == 0) {
        poses = solve_armor_detections(msg);
    } else if (msg->mode == 1) {
        // solve buff detections
    } else if (msg->mode == 2) {
        // solve dart detections
    }
    poses_pub_->publish(poses);
}

Poses LocatorNode::solve_armor_detections(const Detections::SharedPtr msg) {
    Poses poses;
    poses.mode = 0;
    poses.label = msg->label;
    poses.header.stamp = msg->header.stamp;

    tf2::Transform cam_to_chassis;
    if (!utils::try_lookup_tf(
        tf_buffer_,
        "chassis",
        "autoaim_camera",
        msg->header.stamp,
        cam_to_chassis,
        [&](const std::string& err) {
            RCLCPP_WARN(get_logger(), "Failed to lookup camera to chassis: %s", err.c_str());
        }
    )) return poses;

    tf2::Transform gimbal_to_chassis;
    if (!utils::try_lookup_tf(
        tf_buffer_,
        "chassis",
        "gimbal_pitch",
        msg->header.stamp,
        gimbal_to_chassis,
        [&](const std::string& err) {
            RCLCPP_WARN(get_logger(), "Failed to lookup gimbal to chassis: %s", err.c_str());
        }
    )) return poses;

    tf2::Transform chassis_to_basis;
    chassis_to_basis.setIdentity();
    if (utils::try_lookup_tf(
        tf_buffer_,
        "map",
        "chassis",
        {},
        chassis_to_basis,
        [&](const std::string& err) {
            RCLCPP_WARN(get_logger(), "Failed to lookup chassis to map: %s", err.c_str());
        }
    )) poses.header.frame_id = "map";
    else poses.header.frame_id = "chassis";

    auto gimbal_to_basis = chassis_to_basis * gimbal_to_chassis;
    auto gimbal_ypr =
        utils::to_euler_ypr(gimbal_to_basis.getRotation());
    
    for (const auto& detection: msg->armor_detections) {
        auto armor_to_cam = pnp_solver_->solve_pnp(detection, gimbal_ypr);
        auto armor_to_basis = chassis_to_basis * cam_to_chassis * armor_to_cam;
        auto armor_pose = utils::convert_to<geometry_msgs::msg::Pose>(armor_to_basis);
        poses.poses.emplace_back(armor_pose);
    }
    return poses;
}

void LocatorNode::camera_info_callback(const CameraInfo::SharedPtr msg) {
    pnp_solver_->set_cam_matrix(
        cv::Mat(3, 3, CV_64F, msg->k.data()),
        cv::Mat(1, 5, CV_64F, msg->d.data())
    );
    // 相机内参和畸变在运行中不会改变，所以设置后即可取消camera_info订阅
    camera_info_sub_.reset();
    camera_info_sub_ = nullptr;
}
} // namespace autoaim_locator

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_locator::LocatorNode)