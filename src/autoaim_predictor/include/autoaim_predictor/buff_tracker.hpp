#pragma once

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <hw_sentry_interfaces/msg/predictor_status.hpp>
#include <autoaim_common_utils/tf_utils.hpp>
#include <autoaim_common_utils/convert_utils.hpp>
#include <autoaim_common_utils/math_utils.hpp>
#include <autoaim_common_definitions/common_definitions.hpp>
#include <termcolor/termcolor.hpp>
#include <autoaim_predictor/kalman_filter.hpp>
#include <autoaim_predictor/ema_filter.hpp>
#include <autoaim_predictor/trajectory.hpp>
#include <autoaim_predictor/tracker_status.hpp>

struct Buff {
    explicit Buff(const tf2::Transform& buff_pose);

    Eigen::Vector3f translation; // 直接从tf中拿来的位移
    Eigen::Quaternionf rotation; // 直接从tf中拿来的旋转
    Eigen::Vector3f R_center; // R标位置
    float angle; // buff的旋转角
};

class SmallBuffObserver {
public:
    explicit SmallBuffObserver(const cv::FileNode& fn);
    void reset();
    void initialize(const std::vector<Buff>& buffs);
    void update(const std::vector<Buff>& buffs);
    std::tuple<Eigen::Vector3f, bool> predict_shoot_pos(
        const float bullet_speed,
        const float img_to_fire_time,
        const Eigen::Vector3f fric_to_gimbal_yaw
    ) const;

    void print_colored_status_info() const;
    void write_predictor_status(hw_sentry_interfaces::msg::PredictorStatus& status) const;

private:
    float SMALL_BUFF_SPEED; // 小buff的旋转速度
    unsigned QUEUE_SIZE; // 连续存储buff角度，达到收敛次数后判断旋转方向
    unsigned QUEUE_SAMPLE_INTERVAL; // 队列用于判断旋转方向的采样间隔

    std::deque<float> buff_angles_; // buff转角序列
    std::unique_ptr<EMAF<3>> R_center_;
    float theta_;

    enum class RotationDirection {UNKNOWN, CLOCKWISE, COUNTERCLOCKWISE};
    RotationDirection rotation_direction_ = RotationDirection::UNKNOWN; // 当前buff的旋转方向
};

class BuffTracker {
public:
    explicit BuffTracker(const cv::FileNode& fn);
    void push(const Buff& buff);
    void update(const double timestamp);
    void reset();
    void set_mode(AutoaimMode mode);
    StatusType status() const;
    std::tuple<Eigen::Vector3f, bool> predict_shoot_pos(
        const float bullet_speed,
        const float img_to_fire_time,
        const Eigen::Vector3f fric_to_gimbal_yaw
    ) const;

    void print_colored_status_info() const;
    void write_predictor_status(hw_sentry_interfaces::msg::PredictorStatus& status) const;

private:
    void status_change_handler(StatusType from, StatusType to);
    void status_remain_handler(StatusType current);

    AutoaimMode mode_ = AutoaimMode::NONE;

    std::vector<Buff> pushed_buffs_;
    double prev_update_time_, current_update_time_;

    std::unique_ptr<TrackerStatus> small_buff_status_;
    std::unique_ptr<SmallBuffObserver> small_buff_observer_;
};
