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

struct Armor {
    explicit Armor(const tf2::Transform& armor_pose, const float pitch_to_basis);
    explicit Armor(
        const Eigen::Vector3f& translation,
        const Eigen::Quaternionf& rotation,
        const float pitch_to_basis
    );

    Eigen::Vector3f translation; // 直接从tf中拿来的位移
    Eigen::Quaternionf rotation; // 直接从tf中拿来的旋转
    Eigen::Vector3f rotated_x, rotated_y, rotated_z; // 转掉pitch后的各个方向向量
    float yaw; // 法向量在xy平面上投影与x轴的夹角
};

class CarObserver {
public:
    explicit CarObserver(const cv::FileNode& fn);
    void reset();
    void initialize(const std::vector<Armor>& armors);
    void predict(const float time_elapsed) const;
    void update(const std::vector<Armor>& armors);
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
    void write_predictor_status(hw_sentry_interfaces::msg::PredictorStatus& status) const;

    bool is_armor_switched_ = false; // 这次更新时装甲板是否切换了
    bool is_antispin_palstance_ = false; // 当前是否处于反陀螺角速度

private:
    const unsigned ARMORS_COUNT = 4; // 车有4个装甲板
    float INITIAL_RADIUS, SWITCH_ARMOR_ANGLE, DELTA_YAW_UPDATE_THRESHOLD;
    float ENTER_ANTISPIN_PALSTANCE, EXIT_ANTISPIN_PALSTANCE;
    float ANTISPIN_FOLLOW_ANGLE, ANTISPIN_SHOOT_ANGLE;

    // 目前主要观测的装甲板编号。一般来说主要观测的是可视面积最大的那块，对应pushed_armors_[0]
    // 定义第一块看到的装甲板为0，车逆时针转（角速度>0）时看到的依次编号1、2、3
    unsigned main_observing_armor_id_ = 0;
    float accumulated_yaw_; // 累计的0号装甲板yaw角，作为观测量更新kf_rotation_angle
    float prev_main_observing_yaw_; // 上一帧作为主要观测装甲板的yaw角，用于逐差更新accumulated_yaw
    std::unique_ptr<EMAF<1>> radius_[4]; // 每个装甲板对应的半径
    std::unique_ptr<EMAF<1>> height_[4]; // 每个装甲板相对于0号装甲板的高度
    std::unique_ptr<EMAF<3>> axis_; // 车的旋转轴
    std::unique_ptr<KF<3>> kf_center_; // 车中心的位置。车中心在转轴上的高度由0号装甲板确定，即认为0号装甲板对应的中心就是车的中心
    std::unique_ptr<KF<1>> kf_yaw_; // 0号装甲板累计绕旋转轴转的角度
};

class OutpostObserver {
public:
    explicit OutpostObserver(const cv::FileNode& fn);
    void reset();
    void initialize(const std::vector<Armor>& armors);
    void predict(const float time_elapsed) const;
    void update(const std::vector<Armor>& armors);
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
    void write_predictor_status(hw_sentry_interfaces::msg::PredictorStatus& status) const;

    bool is_armor_switched_ = false; // 这次更新时装甲板是否切换了

private:
    const unsigned ARMORS_COUNT = 3; // 前哨站有3个装甲板
    float RADIUS, SWITCH_ARMOR_ANGLE, DELTA_YAW_UPDATE_THRESHOLD;
    float OUTPOST_FOLLOW_ANGLE, OUTPOST_CAN_SHOOT_ANGLE;

    float accumulated_yaw_; // 第一次看到的装甲板yaw角，作为观测量更新kf_rotation_angle
    float prev_main_observing_yaw_; // 上一帧作为主要观测装甲板的yaw角，用于逐差更新accumulated_yaw
    std::unique_ptr<KF<3>> kf_center_; // 车中心的位置
    std::unique_ptr<KF<1>> kf_yaw_; // 第一次看到的装甲板累计绕旋转轴转的角度
};

class ArmorTracker {
public:
    explicit ArmorTracker(const cv::FileNode& fn);

    void set_target_label(ArmorLabel label);
    void push(const Armor& armor);
    void update(const double timestamp);
    void reset();
    StatusType status() const;
    std::tuple<Eigen::Vector3f, bool> predict_shoot_pos(
        const float bullet_speed,
        const float img_to_fire_time,
        const Eigen::Vector3f fric_to_basis,
        const float gimbal_yaw_to_basis
    );

    void print_colored_status_info() const;
    std::vector<std::tuple<Eigen::Vector3f, Eigen::Quaternionf>> get_all_armors() const;
    void write_predictor_status(hw_sentry_interfaces::msg::PredictorStatus& status) const;

private:
    void status_change_handler(StatusType from, StatusType to);
    void status_remain_handler(StatusType current);
    void update_kf_armor_pred_pos_history(const double time, const Eigen::Vector3f& kf_pred_pos);
    void update_car_pred_pos_history(const double time, const Eigen::Vector3f& car_pred_pos);
    void update_pred_accuracy();

    unsigned ERR_QUEUE_SIZE, APPROXIMATE_FRAMERATE;
    float AVG_ERR_THRESHOLD;
    
    ArmorLabel target_label_;
    std::vector<Armor> pushed_armors_;
    double prev_update_time_, current_update_time_;

    std::unique_ptr<KF<3>> kf_main_observing_armor_;
    std::unique_ptr<TrackerStatus> car_status_;
    std::unique_ptr<CarObserver> car_observer_;
    std::unique_ptr<TrackerStatus> outpost_status_;
    std::unique_ptr<OutpostObserver> outpost_observer_;

    std::deque<std::tuple<double, Eigen::Vector3f>> kf_armor_pred_pos_history_; // 单独装甲板的历史预测时间及预测位置
    std::deque<std::tuple<double, Eigen::Vector3f>> car_pred_pos_history_; // 整车的历史预测时间及预测位置
    std::deque<float> kf_armor_err_que_, car_err_que_; // 历史预测误差（由误差角度*距离算出）
    float kf_armor_avg_err_, car_avg_err_; // 由历史预测误差加权平均算出来的误差
};