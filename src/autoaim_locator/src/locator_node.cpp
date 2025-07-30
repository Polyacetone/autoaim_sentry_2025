#include <rclcpp/rclcpp.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/convert.hpp>
#include <tf2/utils.hpp>
#include <tf2/time.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <sensor_msgs/msg/camera_info.hpp>
#include <hw_sentry_interfaces/msg/detections.hpp>
#include <hw_sentry_interfaces/msg/poses.hpp>

#include <autoaim_locator/pnp_solver.hpp>
#include <autoaim_locator/lstm_pose_smoothing.hpp>
#include <autoaim_common_utils/tf_utils.hpp>
#include <autoaim_common_utils/convert_utils.hpp>

namespace autoaim_locator {
using namespace hw_sentry_interfaces::msg;
using namespace sensor_msgs::msg;
using namespace geometry_msgs::msg;

class LocatorNode: public rclcpp::Node {
public:
    explicit LocatorNode(const rclcpp::NodeOptions& options);

private:
    void camera_info_callback(const CameraInfo::SharedPtr msg);
    void detections_callback(const Detections::SharedPtr msg);
    Poses solve_armor_detections(const Detections::SharedPtr msg);
    Poses solve_buff_detections(const Detections::SharedPtr msg);

    std::string basis_frame_id_;

    ArmorType current_locating_label_ = ArmorType::NONE;

    std::unique_ptr<PnPSolver> pnp_solver_;
    std::shared_ptr<LSTMPoseSmoothing> lstm_pose_smoothing_ = nullptr;

    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

    rclcpp::Subscription<CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Subscription<Detections>::SharedPtr selected_detections_sub_;
    rclcpp::Publisher<Poses>::SharedPtr poses_pub_;
};

LocatorNode::LocatorNode(const rclcpp::NodeOptions& options): Node("autoaim_locator", options) {
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    pnp_solver_ = std::make_unique<PnPSolver>();
    
    basis_frame_id_ = declare_parameter<std::string>("basis_frame_id");
    if (basis_frame_id_ != "map" && basis_frame_id_ != "chassis") {
        throw std::invalid_argument("invalid basis frame id");
    }
    bool enable_lstm_smoothing = declare_parameter<bool>("enable_lstm_smoothing");
    if (enable_lstm_smoothing) {
        std::string model_name = declare_parameter<std::string>("lstm_smoothing_model");
        std::string model_path = ament_index_cpp::get_package_share_directory("autoaim_locator")
            + "/model/" + model_name;
        lstm_pose_smoothing_ = std::make_shared<LSTMPoseSmoothing>(model_path);
    }

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
    const AutoaimMode mode = static_cast<AutoaimMode>(msg->mode);
    Poses poses;
    if (mode == AutoaimMode::ARMOR) {
        poses = solve_armor_detections(msg);
    } else if (mode == AutoaimMode::SMALL_BUFF || mode == AutoaimMode::BIG_BUFF) {
        poses = solve_buff_detections(msg);
    } else if (mode == AutoaimMode::DART) {
        // solve dart detections
    }
    poses_pub_->publish(poses);
}

Poses LocatorNode::solve_armor_detections(const Detections::SharedPtr msg) {
    Poses poses;
    poses.mode = msg->mode;
    poses.label = msg->label;
    poses.header.stamp = msg->header.stamp;

    if (current_locating_label_ != static_cast<ArmorType>(msg->label)) {
        current_locating_label_ = static_cast<ArmorType>(msg->label);
        if (lstm_pose_smoothing_) lstm_pose_smoothing_->clear_hidden_states();
    }

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
    poses.header.frame_id = basis_frame_id_;
    if (basis_frame_id_ == "map") {
        utils::try_lookup_tf(
            tf_buffer_,
            "map",
            "chassis",
            {},
            chassis_to_basis,
            [&](const std::string& err) {
                RCLCPP_WARN(get_logger(), "Failed to lookup chassis to map: %s", err.c_str());
            }
        );
    } else if (basis_frame_id_ == "chassis") {
        chassis_to_basis.setIdentity();
    }

    auto gimbal_to_basis = chassis_to_basis * gimbal_to_chassis;
    auto gimbal_ypr = utils::to_euler_ypr(gimbal_to_basis.getRotation());
    auto armors_to_cam = pnp_solver_->solve_pnp(
        msg->armor_detections,
        gimbal_ypr,
        rclcpp::Time(msg->header.stamp).seconds(),
        lstm_pose_smoothing_
    );
    std::transform(armors_to_cam.begin(), armors_to_cam.end(), std::back_inserter(poses.poses),
        [&](const auto& armor_to_cam) {
            auto armor_to_basis = chassis_to_basis * cam_to_chassis * armor_to_cam;
            return utils::convert_to<geometry_msgs::msg::Pose>(armor_to_basis);
        }
    );

    return poses;
}

Poses LocatorNode::solve_buff_detections(const Detections::SharedPtr msg) {
    Poses poses;
    poses.mode = msg->mode;
    poses.label = msg->label;
    poses.header.frame_id = "gimbal_yaw";
    poses.header.stamp = msg->header.stamp;

    tf2::Transform cam_to_gimbal_yaw;
    if (!utils::try_lookup_tf(
        tf_buffer_,
        "gimbal_yaw",
        "autoaim_camera",
        msg->header.stamp,
        cam_to_gimbal_yaw,
        [&](const std::string& err) {
            RCLCPP_WARN(get_logger(), "Failed to lookup camera to gimbal yaw: %s", err.c_str());
        }
    )) return poses;
    
    auto buffs_to_cam = pnp_solver_->solve_pnp(msg->buff_detections, {});
    std::transform(buffs_to_cam.begin(), buffs_to_cam.end(), std::back_inserter(poses.poses),
        [&](const auto& buff_to_cam) {
            auto buff_to_basis = cam_to_gimbal_yaw * buff_to_cam;
            return utils::convert_to<geometry_msgs::msg::Pose>(buff_to_basis);
        }
    );
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