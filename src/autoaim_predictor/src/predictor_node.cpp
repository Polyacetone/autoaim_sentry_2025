#include <rclcpp/rclcpp.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/convert.hpp>
#include <tf2/utils.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <visualization_msgs/msg/marker_array.hpp>
#include <hw_sentry_interfaces/msg/poses.hpp>
#include <hw_sentry_interfaces/msg/bullet_speed.hpp>
#include <hw_sentry_interfaces/msg/shoot_pos.hpp>

#include <autoaim_common_utils/tf_utils.hpp>
#include <autoaim_common_utils/math_utils.hpp>
#include <autoaim_common_definitions/common_definitions.hpp>
#include <car_tracker.hpp>

namespace autoaim_predictor {
using namespace hw_sentry_interfaces::msg;

class PredictorNode: public rclcpp::Node {
public:
    explicit PredictorNode(const rclcpp::NodeOptions& options);

private:
    void poses_callback(const Poses::SharedPtr msg);
    std::tuple<Eigen::Vector3f, bool> predict_target(const rclcpp::Time& img_time) const;
    void send_shoot_pos(const rclcpp::Time& timestamp, const Eigen::Vector3f& target, bool can_shoot) const;
    std::vector<visualization_msgs::msg::Marker> construct_armor_markers(
        const rclcpp::Time& stamp,
        const std::vector<std::tuple<Eigen::Vector3f, Eigen::Quaternionf>>& armors,
        const ArmorType label
    ) const;
    visualization_msgs::msg::Marker construct_target_point_marker(
        const rclcpp::Time& stamp,
        const Eigen::Vector3f position
    ) const;

    bool enable_print_state_, enable_visualization_marker_;
    float bullet_speed_;
    float control_to_fire_time_;
    float shoot_compensate_pitch_, shoot_compensate_yaw_;

    std::unique_ptr<CarTracker> car_tracker_;
    std::string current_basis_frame_id;

    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    rclcpp::Subscription<Poses>::SharedPtr poses_sub_;
    rclcpp::Subscription<BulletSpeed>::SharedPtr bullet_speed_sub_;
    rclcpp::Publisher<ShootPos>::SharedPtr shoot_pos_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr visualization_marker_pub_;
};

PredictorNode::PredictorNode(const rclcpp::NodeOptions& options): Node("autoaim_predictor", options) {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    std::string car_tracker_params_path =
        ament_index_cpp::get_package_share_directory("autoaim_predictor")
        + "/config/car_tracker_params.yaml";
    car_tracker_ = std::make_unique<CarTracker>(car_tracker_params_path);

    enable_print_state_ = declare_parameter<bool>("enable_print_state");
    enable_visualization_marker_ = declare_parameter<bool>("enable_visualization_marker");
    bullet_speed_ = declare_parameter<float>("bullet_speed");
    control_to_fire_time_ = declare_parameter<float>("control_to_fire_time");
    shoot_compensate_pitch_ = declare_parameter<float>("shoot_compensate_pitch");
    shoot_compensate_yaw_ = declare_parameter<float>("shoot_compensate_yaw");
    std::string poses_topic = declare_parameter<std::string>("poses_topic");
    std::string bullet_speed_topic = declare_parameter<std::string>("bullet_speed_topic");
    std::string shoot_pos_topic = declare_parameter<std::string>("shoot_pos_topic");
    std::string visualization_marker_topic = declare_parameter<std::string>("visualization_marker_topic");
    poses_sub_ = create_subscription<Poses>(
        poses_topic,
        rclcpp::QoS(1),
        [&](const Poses::SharedPtr msg) { poses_callback(msg); }
    );
    bullet_speed_sub_ = create_subscription<BulletSpeed>(
        bullet_speed_topic,
        rclcpp::QoS(1),
        [&](const BulletSpeed::SharedPtr msg) {
            if (22.5 <= msg->bullet_speed && msg->bullet_speed <= 24.5) {
                bullet_speed_ = 0.5 * msg->bullet_speed + 0.5 * bullet_speed_;
            }
        }
    );
    shoot_pos_pub_ = create_publisher<ShootPos>(
        shoot_pos_topic,
        rclcpp::QoS(1)
    );
    visualization_marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        visualization_marker_topic,
        rclcpp::QoS(1)
    );
}

void PredictorNode::poses_callback(const Poses::SharedPtr msg) {
    const AutoaimMode mode = static_cast<AutoaimMode>(msg->mode);
    if (current_basis_frame_id != msg->header.frame_id) {
        if (current_basis_frame_id != "") {
            RCLCPP_WARN(
                get_logger(),
                "Basis frame id change from \"%s\" to \"%s\"",
                current_basis_frame_id.c_str(),
                msg->header.frame_id.c_str()
            );
        }
        current_basis_frame_id = msg->header.frame_id;
        car_tracker_->reset();
    }

    if (mode == AutoaimMode::ARMOR) {
        for (const auto& armor_pose: msg->poses) {
            car_tracker_->push(utils::convert_to<tf2::Transform>(armor_pose));
        }
        car_tracker_->update(rclcpp::Time(msg->header.stamp).seconds());
        if (car_tracker_->status() == TrackerStatus::LOST) return;
        if (enable_print_state_) {
            car_tracker_->print_colored_status_info();
            std::cout << "(Basis frame id: " << current_basis_frame_id << ")" << std::endl;
        }
        Eigen::Vector3f target;
        bool can_shoot;
        std::tie(target, can_shoot) = predict_target(msg->header.stamp);
        if (target != Eigen::Vector3f(0, 0, 0)) {
            send_shoot_pos(msg->header.stamp, target, can_shoot);
        }
        if (enable_visualization_marker_) {
            visualization_msgs::msg::MarkerArray marker_arr;
            marker_arr.markers = construct_armor_markers(
                msg->header.stamp,
                car_tracker_->get_all_armors(),
                static_cast<ArmorType>(msg->label
            ));
            marker_arr.markers.emplace_back(construct_target_point_marker(
                msg->header.stamp,
                target
            ));
            visualization_marker_pub_->publish(marker_arr);
        }
    }
}

std::tuple<Eigen::Vector3f, bool> PredictorNode::predict_target(const rclcpp::Time& img_time) const {
    tf2::Transform chassis_to_basis;
    if (current_basis_frame_id == "map") {
        if (!utils::try_lookup_tf(
            tf_buffer_,
            "map",
            "chassis",
            {},
            chassis_to_basis,
            [&](const std::string& err) {
                RCLCPP_WARN(get_logger(), "Failed to lookup chassis to map: %s", err.c_str());
            }
        )) return std::make_tuple(Eigen::Vector3f(0, 0, 0), 0);
    } else if (current_basis_frame_id == "chassis") {
        chassis_to_basis.setIdentity();
    } else {
        RCLCPP_ERROR(get_logger(), "Invalid basis frame id: %s", current_basis_frame_id.c_str());
        return std::make_tuple(Eigen::Vector3f(0, 0, 0), 0);
    }

    tf2::Transform gimbal_to_chassis;
    if (!utils::try_lookup_tf(
        tf_buffer_,
        "chassis",
        "gimbal_pitch",
        img_time,
        gimbal_to_chassis,
        [&](const std::string& err) {
            RCLCPP_WARN(get_logger(), "Failed to lookup gimbal to chassis: %s", err.c_str());
        }
    )) return std::make_tuple(Eigen::Vector3f(0, 0, 0), 0);
    auto gimbal_to_basis = chassis_to_basis * gimbal_to_chassis;
    auto gimbal_ypr = utils::to_euler_ypr(gimbal_to_basis.getRotation());

    tf2::Transform fake_fric_to_chassis;
    if (!utils::try_lookup_tf(
        tf_buffer_,
        "chassis",
        "fake_fric",
        img_time,
        fake_fric_to_chassis,
        [&](const std::string& err) {
            RCLCPP_WARN(get_logger(), "Failed to lookup fake_fric to chassis: %s", err.c_str());
        }
    )) return std::make_tuple(Eigen::Vector3f(0, 0, 0), 0);
    auto fake_fric_to_bassis = chassis_to_basis * fake_fric_to_chassis;

    float img_to_now_time = now().seconds() - img_time.seconds();
    float img_to_hit_time = car_tracker_->predict_img_to_hit_time(
        bullet_speed_,
        img_to_now_time + control_to_fire_time_,
        utils::convert_to<Eigen::Vector3f>(fake_fric_to_bassis.getOrigin())
    );

    Eigen::Vector3f target_to_basis_translation;
    bool can_shoot;
    std::tie(target_to_basis_translation, can_shoot) =
        car_tracker_->predict_shoot_pos(std::get<0>(gimbal_ypr), img_to_hit_time);
    tf2::Transform target_to_basis(
        {0, 0, 0, 1},
        utils::convert_to<tf2::Vector3>(target_to_basis_translation)
    );
    
    auto target_to_fake_fric = fake_fric_to_bassis.inverse() * target_to_basis;
    return std::make_tuple(
        utils::convert_to<Eigen::Vector3f>(target_to_fake_fric.getOrigin()),
        can_shoot
    );
}

void PredictorNode::send_shoot_pos(const rclcpp::Time& timestamp, const Eigen::Vector3f& target, bool can_shoot) const {
    ShootPos shoot_pos;
    shoot_pos.header.frame_id = "fake_fric";
    shoot_pos.header.stamp = timestamp;
    shoot_pos.shoot_flag = can_shoot ? 2 : 0;
    // 注意：发给电控的pitch是向上转为正。但我们在之前的计算中都是向下转为正
    shoot_pos.pitch = std::get<0>(trajectory::get_pitch_air_frac(
        std::hypot(target.x(), target.y()),
        target.z(),
        bullet_speed_
    )) - shoot_compensate_pitch_;
    shoot_pos.yaw = utils::rad_period_correction(atan2(target.y(), target.x())) + shoot_compensate_yaw_;
    shoot_pos_pub_->publish(shoot_pos);
}

std::vector<visualization_msgs::msg::Marker> PredictorNode::construct_armor_markers(
    const rclcpp::Time& stamp,
    const std::vector<std::tuple<Eigen::Vector3f, Eigen::Quaternionf>>& armors,
    const ArmorType label
) const {
    std::vector<visualization_msgs::msg::Marker> markers;
    for (unsigned i = 0; i < armors.size(); i++) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = current_basis_frame_id;
        marker.header.stamp = stamp;
        marker.ns = "predictor";
        marker.id = i;
        marker.type = marker.CUBE;
        marker.action = marker.ADD;
        marker.pose.position = utils::convert_to<geometry_msgs::msg::Point>(std::get<0>(armors[i]));
        marker.pose.orientation = utils::convert_to<geometry_msgs::msg::Quaternion>(std::get<1>(armors[i]));
        marker.scale.x = 0.01, marker.scale.y = defs::is_big_armor(label) ? 0.231 : 0.136, marker.scale.z = 0.05603;
        marker.color.r = (i + 1) & 4, marker.color.g = (i + 1) & 2, marker.color.b = (i + 1) & 1, marker.color.a = 0.5;
        marker.lifetime = rclcpp::Duration::from_seconds(1);
        markers.emplace_back(marker);
    }
    return markers;
}

visualization_msgs::msg::Marker PredictorNode::construct_target_point_marker(
    const rclcpp::Time& stamp,
    const Eigen::Vector3f position
) const {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "fake_fric";
    marker.header.stamp = stamp;
    marker.ns = "predictor";
    marker.id = 4;
    marker.type = marker.SPHERE;
    marker.action = marker.ADD;
    marker.pose.position = utils::convert_to<geometry_msgs::msg::Point>(position);
    marker.pose.orientation = utils::convert_to<geometry_msgs::msg::Quaternion>(Eigen::Quaternionf::Identity());
    marker.scale.x = marker.scale.y = marker.scale.z = 0.04;
    marker.color.r = 1, marker.color.g = 1, marker.color.b = 1, marker.color.a = 1;
    marker.lifetime = rclcpp::Duration::from_seconds(1);
    return marker;
}

} // namespace autoaim_predictor

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_predictor::PredictorNode)