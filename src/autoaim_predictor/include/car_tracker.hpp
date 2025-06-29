#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <autoaim_common_utils/tf_utils.hpp>
#include <autoaim_common_utils/convert_utils.hpp>
#include <autoaim_common_utils/math_utils.hpp>
#include <termcolor/termcolor.hpp>
#include <kalman_filter.hpp>
#include <low_pass_filter.hpp>
#include <trajectory.hpp>

enum class TrackerStatus: unsigned { CONVERGING, TRACKING, TEMP_LOST, LOST };

struct Armor {
    explicit Armor(const tf2::Transform& armor_pose);
    explicit Armor(const Eigen::Vector3f& translation, const Eigen::Quaternionf& rotation);

    Eigen::Vector3f translation; // 直接从tf中拿来的位移
    Eigen::Quaternionf rotation; // 直接从tf中拿来的旋转
    Eigen::Vector3f rotated_x, rotated_y, rotated_z; // 绕着原y轴转15度后的各个方向向量
    float yaw; // 法向量在xy平面上投影与x轴的夹角
};

struct Car {
    explicit Car(const cv::FileNode& fn);

    float INITIAL_RADIUS, SWITCH_ARMOR_ANGLE;
    // 目前主要观测的装甲板编号。一般来说主要观测的是可视面积最大的那块，对应pushed_armors_[0]
    // 定义第一块看到的装甲板为0，车逆时针转（角速度>0）时看到的依次编号1、2、3
    unsigned main_observing_armor_id = 0;
    float accumulated_yaw; // 累计的0号装甲板yaw角，作为观测量更新kf_rotation_angle
    float prev_main_observing_yaw; // 上一帧作为主要观测装甲板的yaw角，用于逐差更新accumulated_yaw
    std::unique_ptr<LPF<1>> radius[4]; // 每个装甲板对应的半径
    std::unique_ptr<LPF<1>> height[4]; // 每个装甲板相对于0号装甲板的高度
    std::unique_ptr<LPF<3>> axis; // 车的旋转轴
    std::unique_ptr<KF<3>> kf_center; // 车中心的位置。车中心在转轴上的高度由0号装甲板确定，即认为0号装甲板对应的中心就是车的中心
    std::unique_ptr<KF<1>> kf_yaw; // 0号装甲板累计绕旋转轴转的角度
};

class Status { 
public:
    explicit Status(
        const cv::FileNode& fn,
        std::function<void(TrackerStatus from, TrackerStatus to)> status_change_handler,
        std::function<void(TrackerStatus current)> status_remain_handler
    );

    void reset();
    void update(bool is_valid);
    TrackerStatus status() const;
    unsigned status_frames() const;

private:
    void set_next_status(TrackerStatus status);

    // 两个handler会且仅会被调用其中一个
    const std::function<void(TrackerStatus from, TrackerStatus to)> status_change_handler; // 状态转移时调用
    const std::function<void(TrackerStatus current)> status_remain_handler; // 状态持续时调用
    unsigned MAX_TEMP_LOST_FRAMES, MAX_CONVERGING_FRAMES;

    TrackerStatus status_ = TrackerStatus::LOST;
    unsigned current_status_frames_ = 0;
};

class CarTracker {
public:
    explicit CarTracker(const std::string& params_path);

    void push(const tf2::Transform& armor_pose);
    void update(const double timestamp);
    void reset();
    TrackerStatus status() const;
    float predict_img_to_hit_time(
        const float bullet_speed,
        const float img_to_fire_time,
        const Eigen::Vector3f fric_to_basis
    ) const;
    std::tuple<Eigen::Vector3f, bool> predict_shoot_pos(
        const float gimbal_yaw_to_basis,
        const float img_to_hit_time
    ) const;

    void print_colored_status_info() const;
    std::vector<std::tuple<Eigen::Vector3f, Eigen::Quaternionf>> get_all_armors() const;

private:
    void status_change_handler(TrackerStatus from, TrackerStatus to);
    void status_remain_handler(TrackerStatus current);
    void update_antispin_mode();
    
    const unsigned ARMORS_COUNT = 4; // 一辆车有4个装甲板
    float ENTER_ANTISPIN_PALSTANCE, EXIT_ANTISPIN_PALSTANCE;
    float ANTISPIN_FOLLOW_ANGLE, ANTISPIN_SHOOT_ANGLE;
    std::vector<Armor> pushed_armors_;
    std::unique_ptr<Status> status_; // 状态机
    std::unique_ptr<Car> car_; // 整车观测器
    std::unique_ptr<KF<3>> kf_main_observing_armor_; // 当转速较小时不需要整车观测，只跟踪装甲板收敛速度更快
    double prev_update_time_, current_update_time_;
    bool is_antispin_mode_ = false;
};