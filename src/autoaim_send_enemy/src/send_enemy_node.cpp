#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <sensor_msgs/msg/camera_info.hpp>
#include <hw_sentry_interfaces/msg/detections.hpp>
#include <hw_sentry_interfaces/msg/robot_color.hpp>
#include <hw_sentry_interfaces/msg/predictor_status.hpp>
#include <hw_sentry_interfaces/msg/enemy_position.hpp>

#include <autoaim_locator/pnp_solver.hpp>
#include <autoaim_predictor/armor_tracker.hpp>
#include <autoaim_predictor/tracker_status.hpp>
#include <autoaim_predictor/ema_filter.hpp>
#include <autoaim_common_definitions/common_definitions.hpp>

namespace autoaim_send_enemy {
using namespace hw_sentry_interfaces::msg;
using namespace sensor_msgs::msg;

class SendEnemyNode: public rclcpp::Node {
public:
    explicit SendEnemyNode(const rclcpp::NodeOptions& options);

private:
    void camera_info_callback(const CameraInfo::SharedPtr msg);
    void detections_callback(const Detections::SharedPtr msg);
    void predictor_status_callback(const PredictorStatus::SharedPtr msg);
    Eigen::Vector3f calc_center(const rclcpp::Time& time, const std::vector<ArmorDetection>& detections) const;

    float car_radius_;

    ArmorColor target_color_ = ArmorColor::NONE;
    ArmorType current_aiming_label_ = ArmorType::NONE;
    geometry_msgs::msg::Point32 current_aiming_center_;
    std::unique_ptr<PnPSolver> pnp_solver_;
    std::unique_ptr<TrackerStatus> car_status_[10];
    std::unique_ptr<EMAF<3>> emaf_center_[10];

    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

    rclcpp::Subscription<Detections>::SharedPtr detections_sub_;
    rclcpp::Subscription<CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Subscription<RobotColor>::SharedPtr robot_color_sub_;
    rclcpp::Subscription<PredictorStatus>::SharedPtr predictor_status_sub_;
    rclcpp::Publisher<EnemyPosition>::SharedPtr enemy_position_pub_;
};

SendEnemyNode::SendEnemyNode(const rclcpp::NodeOptions& options): Node("autoaim_send_enemy", options) {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    pnp_solver_ = std::make_unique<PnPSolver>();

    float emaf_center_filter_ratio = declare_parameter<float>("emaf_center_filter_ratio");
    unsigned max_temp_lost_frames = declare_parameter<int>("max_temp_lost_frames");
    unsigned max_converging_frames = declare_parameter<int>("max_converging_frames");
    car_radius_ = declare_parameter<float>("car_radius");
    for (int i = 0; i < 10; i++) {
        emaf_center_[i] = std::make_unique<EMAF<3>>(emaf_center_filter_ratio);
        car_status_[i] = std::make_unique<TrackerStatus>(max_temp_lost_frames, max_converging_frames);
    }

    target_color_ = static_cast<ArmorColor>(declare_parameter<int>("default_target_color"));
    std::string detections_sub_topic = declare_parameter<std::string>("detections_topic");
    std::string camera_info_sub_topic = declare_parameter<std::string>("camera_info_topic");
    std::string robot_color_sub_topic = declare_parameter<std::string>("robot_color_topic");
    std::string predictor_status_sub_topic = declare_parameter<std::string>("predictor_status_topic");
    std::string enemy_position_pub_topic = declare_parameter<std::string>("enemy_position_topic");

    detections_sub_ = create_subscription<Detections>(
        detections_sub_topic,
        rclcpp::QoS(1),
        [&](const Detections::SharedPtr msg) { detections_callback(msg); }
    );
    camera_info_sub_ = create_subscription<CameraInfo>(
        camera_info_sub_topic,
        rclcpp::QoS(1),
        [&](const CameraInfo::SharedPtr msg) { camera_info_callback(msg); }
    );
    robot_color_sub_ = create_subscription<RobotColor>(
        robot_color_sub_topic,
        rclcpp::QoS(1),
        [&](const RobotColor::SharedPtr msg) {
            const ArmorColor robot_color = static_cast<ArmorColor>(msg->robot_color);
            target_color_ = (robot_color == ArmorColor::BLUE) ? ArmorColor::RED : ArmorColor::BLUE;
        }
    );
    predictor_status_sub_ = create_subscription<PredictorStatus>(
        predictor_status_sub_topic,
        rclcpp::QoS(1),
        [&](const PredictorStatus::SharedPtr msg) { predictor_status_callback(msg); }
    );
    enemy_position_pub_ = create_publisher<EnemyPosition>(
        enemy_position_pub_topic,
        rclcpp::QoS(1)
    );
}

void SendEnemyNode::detections_callback(const Detections::SharedPtr msg) {
    std::unordered_map<ArmorType, std::vector<ArmorDetection>> map;
    if (static_cast<AutoaimMode>(msg->mode) != AutoaimMode::ARMOR) return;
    std::for_each(
        msg->armor_detections.begin(), msg->armor_detections.end(),
        [&](const auto& detection) {
            if (static_cast<ArmorColor>(detection.color) == target_color_) {
                map[static_cast<ArmorType>(detection.label)].emplace_back(detection);
            }
        }
    );
    EnemyPosition enemy_position;
    enemy_position.header.frame_id = "map";
    enemy_position.header.stamp = msg->header.stamp;
    enemy_position.enemy_color = static_cast<int>(target_color_);
    for (int i = static_cast<int>(ArmorType::SENTRY); i <= static_cast<int>(ArmorType::BASE); i++) {
        const ArmorType label = static_cast<ArmorType>(i);
        // 更新状态机和惯性滤波器
        if (map[label].size() == 1 || map[label].size() == 2) {
            Eigen::Vector3f center = calc_center(msg->header.stamp, map[label]);
            if (center == Eigen::Vector3f(0, 0, 0)) continue;
            if (car_status_[static_cast<int>(label)]->status() == StatusType::LOST) {
                emaf_center_[static_cast<int>(label)]->initialize(center);
            } else {
                emaf_center_[static_cast<int>(label)]->update(center);
            }
            car_status_[static_cast<int>(label)]->update(true);
        } else {
            car_status_[static_cast<int>(label)]->update(false);
        }
        // 填充enemy_position
        if (label == current_aiming_label_) {
            enemy_position.enemy_label.emplace_back(static_cast<int>(label));
            enemy_position.enemy_position.emplace_back(current_aiming_center_);
        } else if (
            car_status_[static_cast<int>(label)]->status() == StatusType::TRACKING ||
            car_status_[static_cast<int>(label)]->status() == StatusType::TEMP_LOST) {
            enemy_position.enemy_label.emplace_back(static_cast<int>(label));
            enemy_position.enemy_position.emplace_back(
                utils::convert_to<geometry_msgs::msg::Point32>(
                    emaf_center_[static_cast<int>(label)]->value()
                )
            );
        }
    }
    enemy_position_pub_->publish(enemy_position);
}

void SendEnemyNode::predictor_status_callback(const PredictorStatus::SharedPtr msg) {
    if (static_cast<AutoaimMode>(msg->mode) != AutoaimMode::ARMOR || msg->header.frame_id != "map") {
        current_aiming_label_ = ArmorType::NONE;
        return;
    }
    const StatusType tracker_status = static_cast<StatusType>(msg->tracker_status);
    if (tracker_status == StatusType::TRACKING || tracker_status == StatusType::TEMP_LOST) {
        current_aiming_label_ = static_cast<ArmorType>(msg->label);
        current_aiming_center_ = utils::convert_to<geometry_msgs::msg::Point32>(msg->center);
    } else {
        current_aiming_label_ = ArmorType::NONE;
    }
}

Eigen::Vector3f SendEnemyNode::calc_center(
    const rclcpp::Time& time,
    const std::vector<ArmorDetection>& detections
) const {
    tf2::Transform cam_to_chassis;
    if (!utils::try_lookup_tf(
        tf_buffer_,
        "chassis",
        "autoaim_camera",
        time,
        cam_to_chassis,
        [&](const std::string& err) {
            RCLCPP_WARN(get_logger(), "Failed to lookup camera to chassis: %s", err.c_str());
        }
    )) return {0, 0, 0};

    tf2::Transform gimbal_to_chassis;
    if (!utils::try_lookup_tf(
        tf_buffer_,
        "chassis",
        "gimbal_pitch",
        time,
        gimbal_to_chassis,
        [&](const std::string& err) {
            RCLCPP_WARN(get_logger(), "Failed to lookup gimbal to chassis: %s", err.c_str());
        }
    )) return {0, 0, 0};

    tf2::Transform chassis_to_map;
    if (!utils::try_lookup_tf(
        tf_buffer_,
        "map",
        "chassis",
        {},
        chassis_to_map,
        [&](const std::string& err) {
            RCLCPP_WARN(get_logger(), "Failed to lookup chassis to map: %s", err.c_str());
        }
    )) return {0, 0, 0};

    auto gimbal_to_map = chassis_to_map * gimbal_to_chassis;
    auto gimbal_ypr = utils::to_euler_ypr(gimbal_to_map.getRotation());
    
    const int armors_cnt = detections.size();
    Eigen::Vector3f center(0, 0, 0);
    std::for_each(
        detections.begin(), detections.end(),
        [&](const auto& detection) {
            auto armor_to_cam = pnp_solver_->solve_pnp(detection, gimbal_ypr);
            auto armor_to_map = chassis_to_map * cam_to_chassis * armor_to_cam;
            Armor armor(armor_to_map);
            center += (armor.translation + car_radius_ * armor.rotated_x) / armors_cnt;
        }
    );
    return center;
}

void SendEnemyNode::camera_info_callback(const CameraInfo::SharedPtr msg) {
    pnp_solver_->set_cam_matrix(
        cv::Mat(3, 3, CV_64F, msg->k.data()),
        cv::Mat(1, 5, CV_64F, msg->d.data())
    );
    // 相机内参和畸变在运行中不会改变，所以设置后即可取消camera_info订阅
    camera_info_sub_.reset();
    camera_info_sub_ = nullptr;
}

} // namespace autoaim_send_enemy

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_send_enemy::SendEnemyNode)