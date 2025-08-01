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
    void predict(const float time_elapsed) const;
    void update(const std::vector<Buff>& buffs);
    Eigen::Vector3f predict_shoot_pos(
        const float bullet_speed,
        const float img_to_fire_time,
        const Eigen::Vector3f fric_to_gimbal_yaw
    ) const;
    Eigen::Vector3f get_R_center() const; // 如果收敛中，则指向R_center

    void print_colored_status_info() const;
    void write_predictor_status(hw_sentry_interfaces::msg::PredictorStatus& status) const;

private:
    float SWITCH_BUFF_ANGLE; // 切换buff的角度阈值

    std::unique_ptr<KF<1>> kf_angle_;
    std::unique_ptr<EMAF<3>> R_center_;
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

    unsigned TEMP_LOST_RETURN_FRAMES;

    AutoaimMode mode_ = AutoaimMode::NONE;

    std::vector<Buff> pushed_buffs_;
    double prev_update_time_, current_update_time_;

    std::unique_ptr<TrackerStatus> small_buff_status_;
    std::unique_ptr<SmallBuffObserver> small_buff_observer_;
};
