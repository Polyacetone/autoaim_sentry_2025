#include <string>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <hw_sentry_interfaces/msg/detection_array.hpp>
#include <hw_sentry_interfaces/msg/shoot_pos.hpp>
#include <hw_sentry_interfaces/msg/robot_color.hpp>
#include <hw_sentry_interfaces/msg/enemy_priority.hpp>
#include <hw_sentry_interfaces/msg/comp_robots_hp.hpp>
#include <hw_sentry_interfaces/msg/bullet_speed.hpp>
#include <hw_sentry_interfaces/msg/debug_info.hpp>

#include <pnp_solver.hpp>
#include <tracker.hpp>
#include <trajectory.hpp>

namespace autoaim_prediction {
using namespace hw_sentry_interfaces::msg;
using sensor_msgs::msg::CameraInfo;

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
    void detection_callback(const DetectionArray::SharedPtr msg);
    void camera_info_callback(const CameraInfo::SharedPtr msg);
    void get_debug_info(DebugInfo& msg);

    // 接收机器人血量数据，更新is_enemy_can_shoot，用于筛去死掉或无敌的人
    void robots_hp_callback(const CompRobotsHp::SharedPtr msg);

    // 找出所有视野里的敌人，按照优先级和是否能打（不打死人和无敌），设置目标敌人编号
    void decide_target_armor(const DetectionArray::SharedPtr msg);

    // 选择目标颜色和标签的装甲板，并按照一定规则进行排序
    void select_armors(const std::vector<Detection>& src, std::vector<Detection>& dst) const;

    // 尝试获取指定时间点对应的变换。若尝试MAX_ATTEMPTS后仍没有找到，throw一个std::runtime_error
    geometry_msgs::msg::Transform try_get_transform(
        const std::string& target,
        const std::string& source,
        const rclcpp::Time& time_point
    ) const;

    // 获取指定时间的自己云台的yaw, pitch, roll（相对于chassis）。
    // 之所以是ypr不是rpy，是因为我们采用的旋转顺序是yaw, pitch, roll。
    std::tuple<float, float, float> get_gimbal_ypr(const rclcpp::Time& time_point) const;

    bool is_big_armor(int label) const {
        return (label == 1);
    }

    bool enable_print_state_;
    bool enable_send_to_serial_;
    bool enable_self_decision_;

    int target_color_ = -1;
    int target_armor_ = -1;
    float bullet_speed_;

    float control_to_fire_time_;
    float shoot_compensate_pitch_;
    float shoot_compensate_yaw_;

    std::vector<int> enemy_priority_;
    bool is_enemy_can_shoot_[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

    std::shared_ptr<rclcpp::Subscription<DetectionArray>> detection_sub_;
    std::shared_ptr<rclcpp::Subscription<CameraInfo>> camera_info_sub_;
    std::shared_ptr<rclcpp::Subscription<RobotColor>> robot_color_sub_;
    std::shared_ptr<rclcpp::Subscription<BulletSpeed>> bullet_speed_sub_;
    std::shared_ptr<rclcpp::Subscription<EnemyPriority>> enemy_priority_sub_;
    std::shared_ptr<rclcpp::Subscription<CompRobotsHp>> robots_hp_sub_;

    std::shared_ptr<rclcpp::Publisher<ShootPos>> shoot_pos_pub_;
    std::shared_ptr<rclcpp::Publisher<DebugInfo>> debug_info_pub_;

    std::shared_ptr<PnPSolver> pnp_solver_;
    std::shared_ptr<Tracker> tracker_;
};

PredictionNode::PredictionNode(const rclcpp::NodeOptions& options):
    Node("autoaim_prediction", options) {
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    pnp_solver_ = std::make_shared<PnPSolver>();
    std::string tracker_params_path =
        ament_index_cpp::get_package_share_directory("autoaim_prediction")
        + "/config/tracker_params.yaml";
    tracker_ = std::make_shared<Tracker>(tracker_params_path);

    get_parameters();
}

void PredictionNode::get_parameters() {
    enable_print_state_ = declare_parameter("enable_print_state", false);
    enable_send_to_serial_ = declare_parameter("enable_send_to_serial", true);
    enable_self_decision_ = declare_parameter("enable_self_decision", true);
    auto enemy_priority_int64 = declare_parameter("enemy_priority", std::vector<int64_t> {0, 1, 2, 3, 4});
    for (const auto item: enemy_priority_int64) {
        enemy_priority_.emplace_back(static_cast<int>(item));
    }
    target_color_ = declare_parameter("target_color", 0);
    bullet_speed_ = declare_parameter("bullet_speed", 30.0);
    control_to_fire_time_ = declare_parameter("control_to_fire_time", 0.098);
    shoot_compensate_pitch_ = math::d2r(declare_parameter("shoot_compensate_pitch", 0.0));
    shoot_compensate_yaw_ = math::d2r(declare_parameter("shoot_compensate_yaw", 0.0));

    std::string camera_info_topic = declare_parameter("camera_info_topic", "camera/color/camera_info");
    std::string detection_sub_topic = declare_parameter("detection_sub_topic", "detection");
    std::string decision_target_sub_topic = declare_parameter("decision_target_sub_topic", "decision");
    std::string robot_color_sub_topic = declare_parameter("robot_color_sub_topic", "serial/robot_color");
    std::string bullet_speed_sub_topic = declare_parameter("bullet_speed_sub_topic", "serial/bullet_speed");
    std::string enemy_priority_sub_topic = declare_parameter("enemy_priority_sub_topic", "decision/enemy_priority");
    std::string robots_hp_sub_topic = declare_parameter("robots_hp_topic", "serial/comp_robots_hp");

    std::string debug_info_pub_topic = declare_parameter("debug_info_pub_topic", "autoaim/debug_info");
    std::string shoot_pos_pub_topic = declare_parameter("shoot_pos_pub_topic", "serial/shoot_pos");

    camera_info_sub_ = create_subscription<CameraInfo>(
        camera_info_topic,
        rclcpp::SensorDataQoS().keep_last(1),
        [&](const CameraInfo::SharedPtr msg) { camera_info_callback(msg); }
    );
    detection_sub_ = create_subscription<DetectionArray>(
        detection_sub_topic,
        rclcpp::SensorDataQoS().keep_last(1),
        [&](const DetectionArray::SharedPtr msg) { detection_callback(msg); }
    );
    robot_color_sub_ = create_subscription<RobotColor>(
        robot_color_sub_topic,
        rclcpp::QoS(1),
        [&](const RobotColor::SharedPtr msg) {
            // 需要做一个处理避免裁判系统乱发东西吗？
            target_color_ = 1 - msg->robot_color;
        }
    );
    bullet_speed_sub_ = create_subscription<BulletSpeed>(
        bullet_speed_sub_topic,
        rclcpp::QoS(1),
        [&](const BulletSpeed::SharedPtr msg) {
            if (21 <= msg->bullet_speed && msg->bullet_speed <= 25) {
                // 对弹速进行惯性滤波
                bullet_speed_ = 0.5 * msg->bullet_speed + 0.5 * bullet_speed_;
            }
        }
    );
    enemy_priority_sub_ = create_subscription<EnemyPriority>(
        enemy_priority_sub_topic,
        rclcpp::QoS(1),
        [&](const EnemyPriority::SharedPtr msg) { enemy_priority_ = msg->enemy_priority; }
    );
    robots_hp_sub_ = create_subscription<CompRobotsHp>(
        robots_hp_sub_topic,
        rclcpp::QoS(1),
        [&](const CompRobotsHp::SharedPtr msg) { robots_hp_callback(msg); }
    );
    shoot_pos_pub_ = create_publisher<ShootPos>(
        shoot_pos_pub_topic,
        rclcpp::SensorDataQoS().keep_last(1)
    );
    debug_info_pub_ = create_publisher<DebugInfo>(
        debug_info_pub_topic,
        rclcpp::QoS(1)
    );
}

void PredictionNode::detection_callback(const DetectionArray::SharedPtr msg) {
    std::tuple<float, float, float> gimbal_ypr = get_gimbal_ypr(msg->header.stamp);
    float gimbal_yaw, gimbal_pitch, gimbal_roll;
    std::tie(gimbal_yaw, gimbal_pitch, gimbal_roll) = gimbal_ypr;
    std::vector<Detection> target_armors;
    decide_target_armor(msg);
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
        if (!pnp_solver_->solve_pnp(armor, gimbal_ypr, armor_to_cam.transform)) {
            continue;
        }
        tf_broadcaster_->sendTransform(armor_to_cam);
        // 把装甲板的位姿转换到世界坐标系下进行滤波
        try {
            auto armor_to_chassis = try_get_transform("chassis", armor_name, msg->header.stamp);
            tracker_->push(armor_to_chassis);
        } catch (const std::exception& ex) {
            RCLCPP_WARN(
                get_logger(),
                "Failed to get transform from %s to chassis: %s",
                armor_name.c_str(),
                ex.what()
            );
        }
    }
    tracker_->update(to_sec(msg->header.stamp), target_armor_);

    DebugInfo debug_info;
    debug_info.header.stamp = msg->header.stamp;
    this->get_debug_info(debug_info);
    tracker_->get_debug_info(debug_info);
    debug_info_pub_->publish(debug_info);

    if (tracker_->tracker_status != TRACKER_STATUS::LOST) {
        cv::Point3f target;
        bool can_shoot;
        std::tie(target, can_shoot) = tracker_->get_target_pos(
            gimbal_yaw,
            bullet_speed_,
            to_sec(now()) - to_sec(msg->header.stamp) + control_to_fire_time_
        );

        geometry_msgs::msg::TransformStamped target_to_chassis;
        target_to_chassis.header.stamp = msg->header.stamp;
        target_to_chassis.header.frame_id = "chassis";
        target_to_chassis.child_frame_id = "target";
        target_to_chassis.transform.translation.x = target.x;
        target_to_chassis.transform.translation.y = target.y;
        target_to_chassis.transform.translation.z = target.z;
        tf_broadcaster_->sendTransform(target_to_chassis);

        geometry_msgs::msg::Transform target_to_fric;
        try {
            // fake_fric是原点在摩擦轮系，但方向和大yaw相同的系。解出来的角度方便控车
            target_to_fric = try_get_transform("fake_fric", "target", msg->header.stamp);
        } catch (const std::exception& ex) {
            RCLCPP_WARN(
                get_logger(),
                "Failed to get transform from target to fake_fric: %s",
                ex.what()
            );
            return;
        }

        // 注意：发给电控的pitch是向上转为正。但我们在之前的计算中都是向下转为正（因为符合右手定则）。
        const float target_pitch = trajectory::calc_pitch(
            target_to_fric.translation.x,
            target_to_fric.translation.y,
            target_to_fric.translation.z,
            bullet_speed_
        ) - shoot_compensate_pitch_;
        const float target_yaw = math::rad_period_correction(
            atan2(
                target_to_fric.translation.y,
                target_to_fric.translation.x
            )
        ) + shoot_compensate_yaw_;

        if (enable_print_state_) {
            tracker_->debug_print_state();
            std::printf("Target color %d, label %d\n", target_color_, target_armor_);
            std::printf(
                "Pitch: %4.1f  Yaw: %4.1f (degree)    Shoot_flag: %s\n",
                math::r2d(target_pitch),
                math::r2d(target_yaw),
                (can_shoot ? "true" : "false")
            );
        }
        if (enable_send_to_serial_) {
            ShootPos shoot_pos;
            shoot_pos.header.stamp = msg->header.stamp;
            // 发送给电控的shoot_flag中，0是不发弹，1是单发，2是连发。
            // 一般打人用连发，打符用单发。
            shoot_pos.shoot_flag = can_shoot ? 2 : 0;
            shoot_pos.pitch = target_pitch;
            shoot_pos.yaw = target_yaw;
            shoot_pos_pub_->publish(shoot_pos);
        }
    }
}

void PredictionNode::decide_target_armor(const DetectionArray::SharedPtr msg) {
    static int current_target_lost_frames = 0;
    static int armor_appear_frames[5][15] = {0};
    bool occurred_armors[5][15] = {0};
    if (!msg->detections.empty()) {
        for (const auto& armor: msg->detections) {
            occurred_armors[armor.color][armor.label] = 1;
        }
    }
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 15; j++) {
            if (occurred_armors[i][j]) {
                armor_appear_frames[i][j]++;
            } else {
                armor_appear_frames[i][j] = 0;
            }
        }
    }
    if (target_armor_ != -1) {
        if (!occurred_armors[target_color_][target_armor_]) {
            current_target_lost_frames++;
        } else {
            current_target_lost_frames = 0;
        }
    }

    if (!enemy_priority_.empty()) {
        for (const int label: enemy_priority_) {
            if (label == -1) continue;
            if (label == target_armor_) {
                if (is_enemy_can_shoot_[label] && current_target_lost_frames <= 10) {
                    return;
                }
            } else {
                if (is_enemy_can_shoot_[label] && armor_appear_frames[target_color_][label] > 5) {
                    target_armor_ = label;
                    current_target_lost_frames = 0;
                    tracker_->tracker_status = TRACKER_STATUS::LOST;
                    return;
                }
            }
        }
    }

    target_armor_ = -1;
    tracker_->tracker_status = TRACKER_STATUS::LOST;
}

void PredictionNode::select_armors(const std::vector<Detection>& src, std::vector<Detection>& dst) const {
    constexpr auto get_center_x = [](const Detection& d) -> int {
        return (d.bl.x + d.br.x + d.tr.x + d.tl.x) / 4;
    };
    constexpr auto get_area = [](const Detection& d) -> int {
        return (d.br.x - d.tl.x) * (d.br.y - d.tl.y);
    };
    constexpr auto get_length_height_ratio = [](const Detection& d) -> float {
        return fabs((d.tl.x + d.bl.x - d.tr.x - d.br.x) / (d.tl.y + d.tr.y - d.bl.y - d.br.y));
    };
    static int center_x_prev = 0;
    std::vector<Detection> filtered;
    // 筛选出目标颜色和标签的装甲板
    for (const auto& armor: src) {
        // 用装甲板长宽比筛掉太斜的装甲板
        const bool yaw_too_large = get_length_height_ratio(armor) < (is_big_armor(armor.label) ? 2.0 : 1.6);
        if (!yaw_too_large && armor.label == target_armor_) {
            if (target_color_ == armor.color) {
                filtered.emplace_back(armor);
            } else if (armor.color == 2) { // 特殊处理灰色装甲板
                if (abs(get_center_x(armor) - center_x_prev) <= 15) {
                    // 这里只根据灰色装甲板位置与上次瞄的位置差判断是否是被打成灰的
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
    // 保存之前找过的ypr，在lookupTransform出现异常时返回
    static std::tuple<float, float, float> prev_ypr = std::make_tuple(0, 0, 0);
    geometry_msgs::msg::Transform transform;
    try {
        transform = tf_buffer_->lookupTransform("chassis", "gimbal_pitch", time_point).transform;
    } catch (const std::exception& ex) {
        RCLCPP_WARN(
            get_logger(),
            "Failed to get transform from gimbal_pitch to chassis: %s",
            ex.what()
        );
        return prev_ypr;
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
    prev_ypr = std::make_tuple(yaw, pitch, roll);
    return std::make_tuple(yaw, pitch, roll);
}

geometry_msgs::msg::Transform PredictionNode::try_get_transform(
    const std::string& target,
    const std::string& source,
    const rclcpp::Time& time_point
) const {
    constexpr int MAX_ATTEMPTS = 100;
    geometry_msgs::msg::Transform transform;
    for (int i = 0; i < MAX_ATTEMPTS; i++) {
        try {
            transform = tf_buffer_->lookupTransform(target, source, time_point).transform;
            return transform;
        } catch (const std::exception& ex) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    }
    throw std::runtime_error("try_get_transform failed after 100 attempts");
}

void PredictionNode::get_debug_info(DebugInfo& msg) {
    msg.target_color = target_color_;
    msg.target_armor = target_armor_;
    msg.bullet_speed = bullet_speed_;
    for (int i = 0; i < 10; i++) {
        msg.is_enemy_can_shoot.push_back(is_enemy_can_shoot_[i]);
    }
}

void PredictionNode::camera_info_callback(const CameraInfo::SharedPtr msg) {
    pnp_solver_->set_cam_matrix(
        cv::Mat(3, 3, CV_64F, msg->k.data()),
        cv::Mat(1, 5, CV_64F, msg->d.data())
    );
    // 相机内参和畸变在运行中不会改变，所以设置后即可取消camera_info订阅
    camera_info_sub_.reset();
    camera_info_sub_ = nullptr;
}

void PredictionNode::robots_hp_callback(const CompRobotsHp::SharedPtr msg) {
    constexpr double INVINCIBLE_TIME = 10;
    int enemy_hp[5];
    if (target_color_ == 1) {
        enemy_hp[0] = msg->red_7_robot_hp; // 电控那边认为7是哨兵
        enemy_hp[1] = msg->red_1_robot_hp;
        enemy_hp[2] = msg->red_2_robot_hp;
        enemy_hp[3] = msg->red_3_robot_hp;
        enemy_hp[4] = msg->red_4_robot_hp;
    } else {
        enemy_hp[0] = msg->blue_7_robot_hp; // 电控那边认为7是哨兵
        enemy_hp[1] = msg->blue_1_robot_hp;
        enemy_hp[2] = msg->blue_2_robot_hp;
        enemy_hp[3] = msg->blue_3_robot_hp;
        enemy_hp[4] = msg->blue_4_robot_hp;
    }

    static bool is_enemy_dead[5] = {0, 0, 0, 0, 0};
    static bool is_enemy_invincible[5] = {0, 0, 0, 0, 0};
    static double invincible_start_time[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < 5; i++) {
        if (enemy_hp[i] <= 0) {
            is_enemy_dead[i] = true;
        }
        if (is_enemy_dead[i] && enemy_hp[i] > 0) {
            is_enemy_dead[i] = false;
            is_enemy_invincible[i] = true;
            invincible_start_time[i] = now().seconds();
        }
        if (is_enemy_invincible[i]) {
            if (now().seconds() - invincible_start_time[i] < INVINCIBLE_TIME) {
                is_enemy_invincible[i] = true;
            } else {
                is_enemy_invincible[i] = false;
            }
        }
        if (!is_enemy_dead[i] && !is_enemy_invincible[i]) {
            is_enemy_can_shoot_[i] = true;
        }
    }
}
} // namespace autoaim_prediction

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_prediction::PredictionNode)