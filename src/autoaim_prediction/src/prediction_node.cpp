#include <string>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <hw_sentry_interfaces/msg/detection_array.hpp>
#include <hw_sentry_interfaces/msg/armor_distance.hpp>
#include <hw_sentry_interfaces/msg/shoot_pos.hpp>
#include <hw_sentry_interfaces/msg/decision_info.hpp>

#include <pnp_solver.hpp>
#include <trisection_yaw.hpp>
#include <tracker.hpp>
#include <trajectory.hpp>

namespace autoaim_prediction {
using hw_sentry_interfaces::msg::DecisionInfo;
using hw_sentry_interfaces::msg::Detection;
using hw_sentry_interfaces::msg::DetectionArray;

const geometry_msgs::msg::Transform EMPTY_TRANSFORM;

double to_sec(builtin_interfaces::msg::Time t) {
    return t.sec + t.nanosec * 1e-9;
}

std::string get_tf_armor_name(int color, int label, int index) {
    std::string name;
    char color_map[3] = {'B', 'R', 'G'}; // blue, red, gray
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
    void decision_callback(const DecisionInfo::SharedPtr msg);
    void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

    // 选择目标颜色和标签的装甲板，并按照一定规则进行排序
    void select_armors(const std::vector<Detection> src, std::vector<Detection>& dst) const;

    // 获取指定时间的自己云台的yaw, pitch, roll（相对于chassis）。
    // 之所以是ypr不是rpy，是因为我们采用的旋转顺序是yaw, pitch, roll。
    std::tuple<float, float, float> get_gimbal_ypr(const rclcpp::Time& time_point) const;

    bool enable_print_state_;
    bool enable_send_to_serial_;

    int target_color_;
    int target_armor_;
    float bullet_speed_;

    float control_to_fire_time_;
    float shoot_compensate_pitch_;
    float shoot_compensate_yaw_;

    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

    std::shared_ptr<rclcpp::Subscription<DetectionArray>> detection_sub_;
    std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>> camera_info_sub_;
    std::shared_ptr<rclcpp::Subscription<DecisionInfo>> decision_info_sub_;

    std::shared_ptr<rclcpp::Publisher<hw_sentry_interfaces::msg::ArmorDistance>> armor_dist_pub_;
    std::shared_ptr<rclcpp::Publisher<hw_sentry_interfaces::msg::ShootPos>> shoot_pos_pub_;

    std::shared_ptr<PnPSolver> pnp_solver_;
    std::shared_ptr<TrisectionYaw> trisection_yaw_;
    std::shared_ptr<Tracker> tracker_;
};

PredictionNode::PredictionNode(const rclcpp::NodeOptions& options):
    Node("autoaim_prediction", options) {
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    pnp_solver_ = std::make_shared<PnPSolver>();
    trisection_yaw_ = std::make_shared<TrisectionYaw>();
    std::string tracker_params_path =
        ament_index_cpp::get_package_share_directory("autoaim_prediction")
        + "/config/tracker_params.yaml";
    tracker_ = std::make_shared<Tracker>(tracker_params_path);

    get_parameters();
}

void PredictionNode::get_parameters() {
    enable_print_state_ = declare_parameter("enable_print_state", false);
    enable_send_to_serial_ = declare_parameter("enable_send_to_serial", true);
    target_color_ = declare_parameter("target_color", 0);
    target_armor_ = declare_parameter("target_armor", 1);
    bullet_speed_ = declare_parameter("bullet_speed", 30.0);
    control_to_fire_time_ = declare_parameter("control_to_fire_time", 0.098);
    shoot_compensate_pitch_ = declare_parameter("shoot_compensate_pitch", 0.0);
    shoot_compensate_yaw_ = declare_parameter("shoot_compensate_yaw", 0.0);

    std::string camera_info_topic =
        declare_parameter("camera_info_topic", "/camera/color/camera_info");
    std::string detection_sub_topic = declare_parameter("detection_sub_topic", "/detection");
    std::string decision_info_sub_topic = declare_parameter("decision_info_sub_topic", "/decision");
    std::string armor_distance_pub_topic =
        declare_parameter("armor_distance_pub_topic", "/prediction/armor_distance");
    std::string shoot_pos_pub_topic = declare_parameter("shoot_pos_pub_topic", "/serial/shoot_pos");
    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic,
        rclcpp::SensorDataQoS().keep_last(1),
        [&](const sensor_msgs::msg::CameraInfo::SharedPtr msg) { camera_info_callback(msg); }
    );
    detection_sub_ = create_subscription<DetectionArray>(
        detection_sub_topic,
        rclcpp::SensorDataQoS().keep_last(1),
        [&](const DetectionArray::SharedPtr msg) { detection_callback(msg); }
    );
    decision_info_sub_ = create_subscription<DecisionInfo>(
        decision_info_sub_topic,
        rclcpp::SensorDataQoS().keep_last(1),
        [&](const DecisionInfo::SharedPtr msg) { decision_callback(msg); }
    );
    armor_dist_pub_ = create_publisher<hw_sentry_interfaces::msg::ArmorDistance>(
        armor_distance_pub_topic,
        rclcpp::SensorDataQoS().keep_last(1)
    );
    shoot_pos_pub_ = create_publisher<hw_sentry_interfaces::msg::ShootPos>(
        shoot_pos_pub_topic,
        rclcpp::SensorDataQoS().keep_last(1)
    );
}

void PredictionNode::detection_callback(const DetectionArray::SharedPtr msg) const {
    // 这个thread把视野中所有装甲板的距离解算出来并发出去，用于决策
    if (!msg->detections.empty()) {
        auto thread = std::thread([=]() {
            hw_sentry_interfaces::msg::ArmorDistance armor_distance;
            armor_distance.header.stamp = msg->header.stamp;
            bool occurs[4][10] = {0};
            for (const auto& armor: msg->detections) {
                if (!occurs[armor.color][armor.label]) {
                    occurs[armor.color][armor.label] = 1;
                    geometry_msgs::msg::Transform armor_to_cam;
                    pnp_solver_->get_translation(armor, armor_to_cam);
                    armor_distance.colors.emplace_back(armor.color);
                    armor_distance.labels.emplace_back(armor.label);
                    const float x = armor_to_cam.translation.x;
                    const float y = armor_to_cam.translation.y;
                    const float z = armor_to_cam.translation.z;
                    armor_distance.distances.emplace_back(sqrt(x * x + y * y + z * z));
                }
            }
            armor_dist_pub_->publish(armor_distance);
        });
        thread.detach();
    }
    // 查询时间设置为图像时间减一定时间，是因为相机曝光、处理和传输有延迟
    // 相机曝光设置为2000时，图像时间（即msg->header.stamp）大概比串口节点发布的gimbal tf变换时间晚1.7ms (±0.3ms)
    // 不减掉这段时间的话，tf2查询会出现错误：Lookup would require extrapolation into the future.
    std::tuple<float, float, float> gimbal_ypr = get_gimbal_ypr(
        static_cast<rclcpp::Time>(msg->header.stamp) - std::chrono::microseconds(220)
    );
    float gimbal_yaw, gimbal_pitch, gimbal_roll;
    std::tie(gimbal_yaw, gimbal_pitch, gimbal_roll) = gimbal_ypr;
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
        pnp_solver_->get_translation(armor, armor_to_cam.transform);
        trisection_yaw_->get_rotation(armor, armor_to_cam.transform, gimbal_ypr);
        tf_broadcaster_->sendTransform(armor_to_cam);
        // 把装甲板的位姿转换到世界坐标系下进行滤波
        try {
            auto armor_to_chassis = tf_buffer_->lookupTransform(
                "chassis",
                armor_name,
                msg->header.stamp,
                std::chrono::milliseconds(1)
            ); // 设置1ms的timeout，因为tf发布后需要一段时间才能到tf_buffer里
            tracker_->push(armor_to_chassis.transform);
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN(
                get_logger(),
                "Failed to get transform from %s to chassis: %s",
                armor_name.c_str(),
                ex.what()
            );
        }
    }
    tracker_->update(to_sec(msg->header.stamp));
    if (tracker_->tracker_status != TRACKER_STATUS::LOST) {
        cv::Point3f prediction;
        bool can_shoot;
        std::tie(prediction, can_shoot) = tracker_->get_prediction(
            gimbal_yaw,
            bullet_speed_,
            to_sec(now()) - to_sec(msg->header.stamp) + control_to_fire_time_
            // img_to_fire_time = img_to_control_time + control_to_fire_time
        );

        geometry_msgs::msg::TransformStamped prediction_to_chassis;
        prediction_to_chassis.header.stamp = msg->header.stamp;
        prediction_to_chassis.header.frame_id = "chassis";
        prediction_to_chassis.child_frame_id = "prediction";
        prediction_to_chassis.transform.translation.x = prediction.x;
        prediction_to_chassis.transform.translation.y = prediction.y;
        prediction_to_chassis.transform.translation.z = prediction.z;
        tf_broadcaster_->sendTransform(prediction_to_chassis);

        geometry_msgs::msg::TransformStamped prediction_to_fric;
        try {
            // fake_fric是原点在摩擦轮系，但方向和大yaw相同的系。解出来的角度方便控车
            prediction_to_fric = tf_buffer_->lookupTransform(
                "fake_fric",
                "prediction",
                msg->header.stamp,
                std::chrono::milliseconds(1)
            );
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN(
                get_logger(),
                "Failed to get transform from prediction to fake_fric: %s",
                ex.what()
            );
            return;
        }

        // 注意：发给电控的pitch是向上转为正。但我们在之前的计算中都是向下转为正（因为符合右手定则）。
        const float predicted_pitch = trajectory::calc_pitch(
            prediction_to_fric.transform.translation.x,
            prediction_to_fric.transform.translation.y,
            prediction_to_fric.transform.translation.z,
            bullet_speed_
        ) + math::d2r(shoot_compensate_pitch_);
        const float predicted_yaw = math::rad_period_correction(
            atan2(
                prediction_to_fric.transform.translation.y,
                prediction_to_fric.transform.translation.x
            )
        ) + math::d2r(shoot_compensate_yaw_);
        
        if (enable_print_state_) {
            tracker_->debug_print_state();
            std::printf(
                "Pitch: %4.1f  Yaw: %4.1f (degree)    Shoot_flag: %s\n",
                math::r2d(predicted_pitch),
                math::r2d(predicted_yaw),
                (can_shoot ? "true" : "false")
            );
        }
        if (enable_send_to_serial_ && tracker_->tracker_status != TRACKER_STATUS::TEMP_LOST) {
            hw_sentry_interfaces::msg::ShootPos shoot_pos;
            shoot_pos.header.stamp = msg->header.stamp;
            // 发送给电控的shoot_flag中，0是不发弹，1是单发，2是连发。
            // 一般打人用连发，打符用单发。
            shoot_pos.shoot_flag = can_shoot ? 2 : 0;
            shoot_pos.pitch = predicted_pitch;
            shoot_pos.yaw = predicted_yaw;
            shoot_pos_pub_->publish(shoot_pos);
        }
    }
}

void PredictionNode::decision_callback(const DecisionInfo::SharedPtr msg) {
    target_color_ = msg->color;
    target_armor_ = msg->label;
    bullet_speed_ = msg->bullet_speed;
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

void PredictionNode::select_armors(const std::vector<Detection> src, std::vector<Detection>& dst) const {
    constexpr auto get_center_x = [](const Detection& d) -> int {
        return (d.bl.x + d.br.x + d.tr.x + d.tl.x) / 4;
    };
    constexpr auto get_area = [](const Detection& d) -> int {
        return (d.br.x - d.tl.x) * (d.br.y - d.tl.y);
    };
    static int center_x_prev = 0;

    std::vector<Detection> filtered;
    // 筛选出目标颜色和标签的装甲板
    for (const auto& armor: src) {
        if (armor.label == target_armor_) {
            if (target_color_ == armor.color) {
                filtered.emplace_back(armor);
            } else if (armor.color == 2) { // 特殊处理灰色装甲板
                if (abs(get_center_x(armor) - center_x_prev) <= 15) {
                    // 这里只根据灰色装甲板位置与上次瞄的位置差判断是否是被打成灰的
                    // 理论上需要累计计数判断是被打死了还是暂时灰色
                    // 不过哨兵决策应该会确保自瞄不会对着死人爽打，所以这块不写了
                    filtered.emplace_back(armor);
                }
            }
        }
    }
    if (filtered.empty()) {
        center_x_prev = 0;
        return;
    }

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
            if (abs(center_x_first - center_x_i) >= 15) {
                dst.push_back(filtered[i]);
                break;
            }
        }
    }
    center_x_prev = get_center_x(dst[0]);
}

std::tuple<float, float, float> PredictionNode::get_gimbal_ypr(const rclcpp::Time& time_point) const {
    geometry_msgs::msg::Transform transform;
    try {
        transform = tf_buffer_->lookupTransform("chassis", "gimbal_pitch", time_point).transform;
    } catch (const std::exception& ex) {
        RCLCPP_WARN(get_logger(),
            "Failed to get transform from gimbal_pitch to chasis: %s",
            ex.what()
        );
        return std::make_tuple(0, 0, 0);
    }
    double yaw, pitch, roll;
    tf2::Quaternion quat(
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
        transform.rotation.w
    );
    tf2::Matrix3x3 rot_mat(quat);
    rot_mat.getEulerYPR(yaw, pitch, roll);
    return std::make_tuple(yaw, pitch, roll);
}
} // namespace autoaim_prediction

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_prediction::PredictionNode)