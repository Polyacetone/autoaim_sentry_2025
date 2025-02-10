#include <string>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <autoaim_interfaces/msg/detection_array.hpp>
#include <autoaim_interfaces/msg/comm_send.hpp>

#include <pnp_solver.hpp>
#include <trisection_yaw.hpp>
#include <tracker.hpp>
#include <trajectory.hpp>

namespace autoaim_prediction {
using autoaim_interfaces::msg::Detection;
using autoaim_interfaces::msg::DetectionArray;

const geometry_msgs::msg::Transform EMPTY_TRANSFORM;

double to_sec(builtin_interfaces::msg::Time t) {
    return t.sec + t.nanosec * 1e-9;
}

std::string get_tf_armor_name(int color, int label, int index) {
    std::string name;
    char color_map[3] = {'G', 'B', 'R'}; // gray, blue, red
    name += color_map[color];
    name += static_cast<char>(label + '0');
    name += '-';
    name += static_cast<char>(index + '0');
    return name;
}

class PredictionNode: public rclcpp::Node {
public:
    explicit PredictionNode(const rclcpp::NodeOptions& options);
    ~PredictionNode() = default;

private:
    void get_parameters();
    void detection_callback(const DetectionArray::SharedPtr msg) const;
    void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

    // 获取最新且不重复的变换。有些小问题，不建议使用。
    [[deprecated]] geometry_msgs::msg::Transform
    get_lastest_transform(const std::string& target, const std::string& source) const;

    // 尝试获取指定时间点对应的变换。若尝试MAX_ATTEMPTS后仍没有找到，返回EMPTY_TRANSFORM
    geometry_msgs::msg::Transform try_get_transform(
        const std::string& target,
        const std::string& source,
        const rclcpp::Time& time_point
    ) const;

    // 选择目标颜色和标签的装甲板，并按照一定规则进行排序
    void select_armors(const std::vector<Detection> src, std::vector<Detection>& dst) const;

    bool enable_print_state_;
    bool enable_debug_;
    bool debug_mode_;
    int debug_target_color_;
    int debug_target_armor_;
    int debug_buff_mode_;
    float debug_prediction_time_;
    float debug_bullet_speed_;

    float armor_dir_angle_;
    float filter_distance_;
    float imu_compensate_pitch_;
    float imu_compensate_yaw_;
    float t_delay_;

    std::string camera_info_topic_;
    std::string detection_sub_topic_;
    std::string comm_pub_topic_;
    std::string position_pub_topic_;

    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_static_broadcaster_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

    std::shared_ptr<rclcpp::Subscription<DetectionArray>> detection_sub_;
    std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>> camera_info_sub_;
    std::shared_ptr<rclcpp::Publisher<autoaim_interfaces::msg::CommSend>> comm_send_;

    std::shared_ptr<PnPSolver> pnp_solver_;
    std::shared_ptr<TrisectionYaw> trisection_yaw_;
    std::shared_ptr<Tracker> tracker_;
};

PredictionNode::PredictionNode(const rclcpp::NodeOptions& options):
    Node("autoaim_prediction", options) {
    tf_static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    pnp_solver_ = std::make_shared<PnPSolver>();
    trisection_yaw_ = std::make_shared<TrisectionYaw>();
    std::string kf_params_path = ament_index_cpp::get_package_share_directory("autoaim_prediction")
        + "/config/kf_params.yaml";
    tracker_ = std::make_shared<Tracker>(kf_params_path);

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
    comm_send_ = create_publisher<autoaim_interfaces::msg::CommSend>(comm_pub_topic_, 10);
}

void PredictionNode::get_parameters() {
    enable_print_state_ = declare_parameter("enable_print_state", false);
    enable_debug_ = declare_parameter("enable_debug", false);
    debug_mode_ = declare_parameter("debug_mode", false);
    debug_prediction_time_ = declare_parameter("debug_prediction_time", 0.0);
    debug_target_color_ = declare_parameter("debug_target_color", 0);
    debug_target_armor_ = declare_parameter("debug_target_armor", 1);
    debug_buff_mode_ = declare_parameter("debug_buff_mode", 1);
    debug_bullet_speed_ = declare_parameter("debug_bullet_speed", 30.0);
    armor_dir_angle_ = declare_parameter("armor_dir_angle", 0.2618);
    filter_distance_ = declare_parameter("filter_distance", 6.0);
    imu_compensate_pitch_ = declare_parameter("imu_compensate_pitch", 1.3);
    imu_compensate_yaw_ = declare_parameter("imu_compensate_yaw", 0.3);
    t_delay_ = declare_parameter("t_delay", 0.098);
    camera_info_topic_ = declare_parameter("camera_info_topic", "/camera/color/camera_info");
    detection_sub_topic_ = declare_parameter("detection_sub_topic", "/detection");
    comm_pub_topic_ = declare_parameter("comm_pub_topic", "/serial/comm_send");
    position_pub_topic_ = declare_parameter("position_pub_topic", "/debug/position");

    const double cam_to_gimbal_x = declare_parameter("cam_to_gimbal_x", 0.0);
    const double cam_to_gimbal_y = declare_parameter("cam_to_gimbal_y", 0.0658);
    const double cam_to_gimbal_z = declare_parameter("cam_to_gimbal_z", 0.0639);
    geometry_msgs::msg::TransformStamped cam_to_gimbal;
    cam_to_gimbal.header.stamp = this->now();
    cam_to_gimbal.header.frame_id = "gimbal";
    cam_to_gimbal.child_frame_id = "autoaim_camera";
    cam_to_gimbal.transform.translation.x = cam_to_gimbal_x;
    cam_to_gimbal.transform.translation.y = cam_to_gimbal_y;
    cam_to_gimbal.transform.translation.z = cam_to_gimbal_z;
    // 这里认为相机坐标系向右是x，向下是y，向前是z。世界坐标系向右是x，向前是y，向上是z。
    cam_to_gimbal.transform.rotation.x = -0.7071068;
    cam_to_gimbal.transform.rotation.y = 0;
    cam_to_gimbal.transform.rotation.z = 0;
    cam_to_gimbal.transform.rotation.w = 0.7071068;
    tf_static_broadcaster_->sendTransform(cam_to_gimbal);
}

void PredictionNode::detection_callback(const DetectionArray::SharedPtr msg) const {
    if (enable_debug_ && debug_mode_ == false) {
        std::vector<Detection> target_armors;
        select_armors(msg->detections, target_armors);
        const int len = target_armors.size();
        for (int i = 0; i < len; i++) {
            const Detection& armor = target_armors[i];
            const std::string armor_name = get_tf_armor_name(armor.color, armor.label, i);
            geometry_msgs::msg::TransformStamped armor_to_cam;
            armor_to_cam.header.stamp = msg->header.stamp;
            armor_to_cam.header.frame_id = "autoaim_camera";
            armor_to_cam.child_frame_id = armor_name;
            // 计算装甲板相对于相机坐标系的位姿
            pnp_solver_->solve_pnp(armor, armor_to_cam.transform);
            trisection_yaw_->get_rotation(armor, armor_to_cam.transform);
            tf_broadcaster_->sendTransform(armor_to_cam);
            // 把装甲板的位姿转换到世界坐标系（原点为云台和小yaw转动的中心，方向为imu初始化时的方向）下进行滤波
            // 尽可能降低自己车的转动对跟踪器的影响。不过由于目前似乎没有有效的精准定位手段，还不能消除平动影响
            auto armor_to_world = try_get_transform("world", armor_name, msg->header.stamp);
            if (armor_to_world != EMPTY_TRANSFORM) {
                tracker_->push(armor_to_world);
            } else {
                RCLCPP_WARN(get_logger(), "Get transform failed.");
            }
        }
        tracker_->update();
        if (tracker_->tracker_status == TRACKER_STATUS::TRACKING) {
            const cv::Point3f predicted = tracker_->get_prediction(debug_bullet_speed_, t_delay_);
            geometry_msgs::msg::TransformStamped predicted_to_world;
            predicted_to_world.header.stamp = msg->header.stamp;
            predicted_to_world.header.frame_id = "world";
            predicted_to_world.child_frame_id = "predicted";
            predicted_to_world.transform.translation.x = predicted.x;
            predicted_to_world.transform.translation.y = predicted.y;
            predicted_to_world.transform.translation.z = predicted.z;
            tf_broadcaster_->sendTransform(predicted_to_world);

            // 目前弹道计算直接在世界坐标系下进行。虽然和摩擦轮有一定差距，不过感觉影响不大？
            float predicted_pitch, predicted_yaw;
            predicted_pitch = trajectory::calc_pitch(
                predicted_to_world.transform.translation.x,
                predicted_to_world.transform.translation.y,
                predicted_to_world.transform.translation.z,
                debug_bullet_speed_
            );
            predicted_yaw = math::rad_period_correction(
                atan2(
                    predicted_to_world.transform.translation.y,
                    predicted_to_world.transform.translation.x
                )
                - M_PI / 2
            );

            if (enable_print_state_) {
                tracker_->debug_print_state();
                RCLCPP_INFO(
                    get_logger(),
                    "Pitch: %4.1f  Yaw: %4.1f (degree)",
                    math::r2d(predicted_pitch),
                    math::r2d(predicted_yaw)
                );
            }

            /*autoaim_interfaces::msg::CommSend comm_send;
            comm_send.header.stamp = now();
            comm_send.found = true;
            comm_send.pitch = math::r2d(predicted_pitch);
            comm_send.yaw = math::r2d(predicted_yaw);
            comm_send_->publish(comm_send);*/
        }
    }
}

void PredictionNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    pnp_solver_->set_cam_matrix(
        cv::Mat(3, 3, CV_64F, msg->k.data()),
        cv::Mat(1, 5, CV_64F, msg->d.data())
    );
    trisection_yaw_->set_cam_matrix(
        cv::Mat(3, 3, CV_64F, msg->k.data()),
        cv::Mat(1, 5, CV_64F, msg->d.data())
    );
    // 相机内参和畸变在运行中不会改变，所以设置后即可取消camera_info订阅
    camera_info_sub_.reset();
    camera_info_sub_ = nullptr;
}

void PredictionNode::select_armors(
    const std::vector<Detection> src, 
    std::vector<Detection>& dst
) const {
    std::vector<Detection> filtered;
    // 筛选出目标颜色和标签的装甲板
    for (const auto& armor: src) {
        if (enable_debug_) {
            if ((debug_target_color_ == 0 || armor.color == debug_target_color_)
                && (armor.label == debug_target_armor_))
            {
                filtered.emplace_back(armor);
            }
        } else {
            // TODO
        }
    }
    if (filtered.empty()) {
        return;
    }

    constexpr auto get_center_x = [](const Detection& d) -> int {
        return (d.bl.x + d.br.x + d.tr.x + d.tl.x) / 4;
    };
    constexpr auto get_area = [](const Detection& d) -> int {
        return (d.br.x - d.tl.x) * (d.br.y - d.tl.y);
    };
    static int center_x_prev = 0;
    // 对目标装甲板进行排序
    if (filtered.size() == 1) {
        dst.push_back(filtered[0]);
    } else {
        // 根据击打面积和装甲板位置与正在瞄准位置间的差异排序
        std::sort(filtered.begin(), filtered.end(), [&](const Detection& a, const Detection& b) {
            if (center_x_prev == 0) {
                return get_area(a) > get_area(b);
            } else {
                return get_area(a) - abs(get_center_x(a) - center_x_prev)
                    > get_area(b) - abs(get_center_x(b) - center_x_prev);
            }
        });
        dst.emplace_back(filtered[0]);
        // 接下来选择击打面积次之，且和原来那个位置有较大差异的装甲板。
        // 虽然理论上detection中的nms已经能去除同一个装甲板的多个识别结果，但有时候还是会出现。
        const int len = filtered.size();
        const int center_x_first = get_center_x(filtered[0]);
        for (int i = 1; i < len; i++) {
            const int center_x_i = get_center_x(filtered[i]);
            if (abs(center_x_first - center_x_i) >= 50) {
                dst.push_back(filtered[i]);
                break;
            }
        }
    }
    center_x_prev = get_center_x(dst[0]);
}

[[deprecated]] geometry_msgs::msg::Transform
PredictionNode::get_lastest_transform(const std::string& target, const std::string& source) const {
    // 防止buffer没来得及更新使得找不到frame
    while (!tf_buffer_->canTransform(target, source, tf2::TimePointZero)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // 对每一次获取的时间戳进行记录，防止buffer没来得及更新使得找到旧的变换
    static std::unordered_map<std::string, double> timestamp_map;
    geometry_msgs::msg::TransformStamped tf_stamped;
    do {
        tf_stamped = tf_buffer_->lookupTransform(target, source, tf2::TimePointZero);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (timestamp_map[source] == to_sec(tf_stamped.header.stamp));
    timestamp_map[source] = to_sec(tf_stamped.header.stamp);
    return tf_stamped.transform;
}

geometry_msgs::msg::Transform PredictionNode::try_get_transform(
    const std::string& target,
    const std::string& source,
    const rclcpp::Time& time_point
) const {
    constexpr int MAX_ATTEMPTS = 1000;
    geometry_msgs::msg::Transform transform = EMPTY_TRANSFORM;
    for (int i = 0; i < MAX_ATTEMPTS; i++) {
        try {
            transform = tf_buffer_->lookupTransform(target, source, time_point).transform;
            break;
        } catch (const std::exception& ex) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    }
    return transform;
}
} // namespace autoaim_prediction

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_prediction::PredictionNode)