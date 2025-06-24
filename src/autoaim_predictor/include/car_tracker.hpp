#pragma once

#include <opencv2/opencv.hpp>

#include <tf2/LinearMath/Matrix3x3.hpp>
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2/LinearMath/Transform.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/point32.hpp>

#include <autoaim_common_utils/tf_utils.hpp>
#include <autoaim_common_utils/convert_utils.hpp>
#include <autoaim_common_utils/math_utils.hpp>
#include <kalman_filters.hpp>
#include <trajectory.hpp>

enum class TrackerStatus { CONVERGING, TRACKING, TEMP_LOST, LOST };

struct Armor {
    cv::Point3f center; // 装甲板中心坐标
    float angle; // 装甲板向心方向在xy平面的投影向量与正前方（y轴）的夹角，逆时针为正
};

class CarTracker {
public:
    explicit CarTracker(const std::string& params_path);
    void push(const tf2::Transform& transform);
    void update(const double time_stamp, const int label);
    void debug_print_state();

    float get_img_to_hit_time(
        const float bullet_speed,
        const float img_to_fire_time,
        const cv::Point3f fric_to_basis
    );

    std::tuple<cv::Point3f, bool> get_target_pos(
        const float gimbal_yaw,
        const float img_to_hit_time
    );

    TrackerStatus tracker_status_ = TrackerStatus::LOST;
    std::unique_ptr<KFXYZ> kf_xyz_;
    std::unique_ptr<KFYaw> kf_yaw_;
    std::unique_ptr<UKFXY> ukf_;

private:
    float INITIAL_RADIUS = 0.26;
    float MIN_RADIUS = 0.2, MAX_RADIUS = 0.35;
    float OUTPOST_RADIUS = 0.22;
    float SWITCH_ARMOR_ANGLE = utils::d2r(50);
    float RADIUS_FILTER_RATIO = 0.7;
    float HEIGHT_FILTER_RATIO = 0.6;
    float ENTER_ANTITOP_PALSTANCE_THRESHOLD = utils::d2r(120);
    float EXIT_ANTITOP_PALSTANCE_THRESHOLD = utils::d2r(80);
    float ANTITOP_CAN_SHOOT_ANGLE = utils::d2r(30);
    float ANTITOP_FOLLOW_ANGLE = utils::d2r(30);
    float OUTPOST_CAN_SHOOT_ANGLE = utils::d2r(60);
    unsigned MAX_LOST_FRAMES = 5;
    unsigned CONVERGE_FRAMES = 5;
    unsigned OUTPOST_MAX_LOST_FRAMES = 40;

    unsigned current_status_frames_ = 0; // 当前tracker_status状态的持续帧数
    unsigned observing_armor_id_ = 0; // 正在观测的装甲板编号。定义第一块看到的装甲板为0，车逆时针转时看到的依次编号1、2、3
    float radius_[2]; // radius_[0]对应0、2装甲板半径，radius_[1]对应1、3
    float height_[4]; // 分别对应4个不同的装甲板高度
    float accumulated_yaw_ = 0; // 根据帧间差累计的yaw角，用于更新kf_yaw_

    double prev_update_time_ = 0;
    float prev_update_angle_ = 0;
    bool is_antitop_palstance_ = 0;
    
    int target_label_; // 当前正在跟踪的目标编号，用于特判前哨站
    std::vector<Armor> armors_;

    void load_params(const std::string& params_path);
    void update_radius();
    void update_height();
    bool is_outpost() const { return (target_label_ == 5); }
    bool decide_antitop_mode();
};