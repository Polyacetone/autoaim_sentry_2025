// 维护一个整车状态的跟踪器
// KFXYZ用于平动，KFYaw+UKFXY用于小陀螺，半径和高度采用惯性滤波
// 坐标系定义：前x，左y，上z
// yaw角方向定义：逆时针（即从x到y）为正
// 距离和时间均使用国际单位制（m、s），角度使用弧度制

#pragma once

#include <kalman_filters.hpp>
#include <math_utils.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <hw_sentry_interfaces/msg/debug_info.hpp>

enum class TRACKER_STATUS { CONVERGING, TRACKING, TEMP_LOST, LOST };

struct Armor {
    cv::Point3f center; // 装甲板中心坐标
    float angle; // 装甲板向心方向在xy平面的投影向量与正前方（y轴）的夹角，逆时针为正
};

class Tracker {
public:
    Tracker(const std::string& params_path);
    void push(const geometry_msgs::msg::Transform& transform);
    void update(const double time_stamp, const int label);
    void debug_print_state();
    void get_debug_info(hw_sentry_interfaces::msg::DebugInfo& debug_info);

    /*!
        @brief 获取预测击打坐标（世界系下）
        @param gimbal_yaw 云台相对世界系的yaw（用于计算面向我们的装甲板是哪个）
        @param bullet_speed 子弹速度
        @param img_to_fire_time 图像时间到开火时间的估计值
        @return 预测的击打坐标，和是否发弹（即shoot_flag）
        @attention 不应在tracker_status为lost时调用
    */
    std::tuple<cv::Point3f, bool> get_target_pos(
        const float gimbal_yaw,
        const float bullet_speed, 
        const float img_to_fire_time
    );

    TRACKER_STATUS tracker_status = TRACKER_STATUS::LOST;

private:
    float INITIAL_RADIUS = 0.26;
    float MIN_RADIUS = 0.2, MAX_RADIUS = 0.35;
    float OUTPOST_RADIUS = 0.22;
    float SWITCH_ARMOR_ANGLE = math::d2r(50);
    float RADIUS_FILTER_RATIO = 0.7;
    float HEIGHT_FILTER_RATIO = 0.6;
    float ANTITOP_PALSTANCE_THRESHOLD = math::d2r(50);
    float ANTITOP_CAN_SHOOT_ANGLE = math::d2r(30);
    float ANTITOP_FOLLOW_ANGLE = math::d2r(30);
    float OUTPOST_CAN_SHOOT_ANGLE = math::d2r(60);
    int MAX_LOST_FRAMES = 5;
    int CONVERGE_FRAMES = 5;
    int OUTPOST_MAX_LOST_FRAMES = 40;

    std::unique_ptr<KFXYZ> kf_xyz_;
    std::unique_ptr<KFYaw> kf_yaw_;
    std::unique_ptr<UKFXY> ukf_;
    unsigned current_status_frames_ = 0; // 当前tracker_status状态的持续帧数
    unsigned observing_armor_id_ = 0; // 正在观测的装甲板编号。定义第一块看到的装甲板为0，车逆时针转时看到的依次编号1、2、3
    float radius_[2]; // radius_[0]对应0、2装甲板半径，radius_[1]对应1、3
    float height_[4]; // 分别对应4个不同的装甲板高度
    float accumulated_yaw_ = 0; // 根据帧间差累计的yaw角，用于更新kf_yaw_

    double prev_update_time_ = 0;
    float prev_update_angle_ = 0;
    int target_label_; // 当前正在跟踪的目标编号，用于特判前哨站
    std::vector<Armor> armors_;

    void load_params(const std::string& params_path);
    void update_radius();
    void update_height();
    bool is_outpost() const { return (target_label_ == 5); }
};

Tracker::Tracker(const std::string& params_path) {
    load_params(params_path);
    kf_xyz_ = std::make_unique<KFXYZ>(params_path);
    kf_yaw_ = std::make_unique<KFYaw>(params_path);
    ukf_ = std::make_unique<UKFXY>(params_path);
}

void Tracker::push(const geometry_msgs::msg::Transform& transform) {
    Armor armor;
    armor.center =
        cv::Point3f(transform.translation.x, transform.translation.y, transform.translation.z);
    tf2::Quaternion quaternion(
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
        transform.rotation.w
    );
    tf2::Matrix3x3 rotation_mat(quaternion);
    double yaw, pitch, roll;
    rotation_mat.getEulerYPR(yaw, pitch, roll);
    armor.angle = yaw;
    armors_.emplace_back(armor);
}

void Tracker::update(const double time_stamp, const int label) {
    target_label_ = label;
    using TS = TRACKER_STATUS;
    const float time_elapsed = static_cast<float>(time_stamp - prev_update_time_); // 和上一帧比经过的时间

    if (armors_.empty() || armors_.size() > 2) {
        if (tracker_status != TS::LOST) { // 短暂失踪，只预测不更新
            if (tracker_status == TS::CONVERGING) {
                tracker_status = TS::LOST;
                current_status_frames_ = 0;
            } else if (tracker_status == TS::TRACKING) {
                tracker_status = TS::TEMP_LOST;
                current_status_frames_ = 0;
            } else if (tracker_status == TS::TEMP_LOST) {
                if (current_status_frames_ > (is_outpost() ? OUTPOST_MAX_LOST_FRAMES : MAX_LOST_FRAMES)) {
                    tracker_status = TS::LOST;
                    current_status_frames_ = 0;
                }
            }
            kf_xyz_->predict(time_elapsed);
            kf_yaw_->predict(time_elapsed);
            ukf_->predict(time_elapsed);
        }
        current_status_frames_++;
    } else {
        if (tracker_status == TS::LOST) { // 初始化
            tracker_status = TS::CONVERGING;
            current_status_frames_ = 0;
            kf_xyz_->initialize(armors_[0].center);
            kf_yaw_->initialize(armors_[0].angle);
            const float radius = is_outpost() ? OUTPOST_RADIUS : INITIAL_RADIUS;
            const cv::Point2f car_center(
                armors_[0].center.x + radius * cos(armors_[0].angle),
                armors_[0].center.y + radius * sin(armors_[0].angle)
            );
            ukf_->initialize(car_center);
            observing_armor_id_ = 0;
            radius_[0] = radius_[1] = radius;
            height_[0] = height_[1] = height_[2] = height_[3] = armors_[0].center.z;
            accumulated_yaw_ = prev_update_angle_ = armors_[0].angle;
        } else { // 正常预测并更新
            if (tracker_status == TS::TEMP_LOST) {
                tracker_status = TS::TRACKING;
                current_status_frames_ = 0;
            } else if (tracker_status == TS::CONVERGING) {
                if (current_status_frames_ > CONVERGE_FRAMES) {
                    tracker_status = TS::TRACKING;
                    current_status_frames_ = 0;
                }
            }
            current_status_frames_++;
            kf_xyz_->predict(time_elapsed);
            kf_yaw_->predict(time_elapsed);
            ukf_->predict(time_elapsed);
            float delta_angle = math::rad_period_correction(armors_[0].angle - prev_update_angle_);
            const unsigned armors_count = is_outpost() ? 3 : 4;
            accumulated_yaw_ += delta_angle;
            if (delta_angle < -SWITCH_ARMOR_ANGLE) {
                // 逆时针转（角速度大于0）时切换装甲板
                observing_armor_id_ += 1;
                observing_armor_id_ %= armors_count;
                accumulated_yaw_ += M_PI * 2 / armors_count;
                kf_xyz_->force_change_position(armors_[0].center);
            } else if (delta_angle > SWITCH_ARMOR_ANGLE) {
                // 顺时针转（角速度小于0）时切换装甲板
                observing_armor_id_ += 3;
                observing_armor_id_ %= armors_count;
                accumulated_yaw_ -= M_PI * 2 / armors_count;
                kf_xyz_->force_change_position(armors_[0].center);
            }
            kf_xyz_->update(armors_[0].center);
            kf_yaw_->update(accumulated_yaw_);
            update_radius();
            update_height();
            const float radius = is_outpost() ? OUTPOST_RADIUS : radius_[observing_armor_id_ % 2];
            const cv::Point2f car_center(
                armors_[0].center.x + radius * cos(armors_[0].angle),
                armors_[0].center.y + radius * sin(armors_[0].angle)
            );
            ukf_->update(car_center);
            prev_update_angle_ = armors_[0].angle;
        }
    }

    armors_.clear();
    prev_update_time_ = time_stamp;
}

std::tuple<cv::Point3f, bool> Tracker::get_target_pos(
    const float gimbal_yaw,
    const float bullet_speed, 
    const float img_to_fire_time
) {
    if (abs(kf_yaw_->palstance) < ANTITOP_PALSTANCE_THRESHOLD && !is_outpost()) { // 平动，只用KFXYZ预测
        // 理论上来说要精确求出这里的击打时间需要解一个方程，这里为了简化直接采用二阶近似
        const float img_to_hit_time_1 = math::get_distance(kf_xyz_->position) / bullet_speed;
        const float img_to_hit_time_2 = img_to_fire_time +
            math::get_distance(kf_xyz_->position + img_to_hit_time_1 * kf_xyz_->velocity) / bullet_speed;
        return std::make_tuple(
            kf_xyz_->position + img_to_hit_time_2 * kf_xyz_->velocity, 
            tracker_status != TRACKER_STATUS::CONVERGING
        );
    } else { // 转动，用KFYaw和UKFXY预测
        // 同上，img_to_hit_time为二阶近似
        const float img_to_hit_time_1 = math::get_distance(ukf_->position) / bullet_speed;
        const float img_to_hit_time_2 = img_to_fire_time +
            math::get_distance(ukf_->position + img_to_hit_time_1 * ukf_->velocity) / bullet_speed;
        const cv::Point2f pred_center = ukf_->position + ukf_->velocity * img_to_hit_time_2;
        // 0号装甲板在世界系下的预测yaw角
        const float pred_yaw_to_world = kf_yaw_->yaw + kf_yaw_->palstance * img_to_hit_time_2;
        // 0号装甲板在gimbal系下的预测yaw角
        const float pred_yaw_to_gimbal = math::rad_period_correction(pred_yaw_to_world - gimbal_yaw);
        // 车的装甲板数量，前哨站只有三个装甲板
        const unsigned armors_count = is_outpost() ? 3 : 4;
        // 最面向我们的装甲板在gimbal系下的预测角
        float target_angle_to_gimbal = M_PI * 2 / armors_count; 
        unsigned target_armor_id = 0;
        // 选择在img_to_hit_time之后，角度最小（即最面向我们）的那个装甲板
        for (unsigned i = 0; i < armors_count; i++) {
            const float pred_angle_to_gimbal =
                math::rad_period_correction(pred_yaw_to_gimbal + M_PI * i * 2 / armors_count);
            if (abs(pred_angle_to_gimbal) < abs(target_angle_to_gimbal)) {
                target_angle_to_gimbal = pred_angle_to_gimbal;
                target_armor_id = (observing_armor_id_ + i) % armors_count;
            }
        }
        const float follow_angle = is_outpost() ? OUTPOST_CAN_SHOOT_ANGLE : ANTITOP_FOLLOW_ANGLE;
        if (abs(target_angle_to_gimbal) < follow_angle) { // 跟随射击
            const float target_angle_to_world = 
                math::rad_period_correction(target_angle_to_gimbal + gimbal_yaw);
            const float radius = is_outpost() ? OUTPOST_RADIUS : radius_[target_armor_id % 2];
            const float height = is_outpost() ? height_[0] : height_[target_armor_id % 4];
            const cv::Point3f target = cv::Point3f(
                pred_center.x - cos(target_angle_to_world) * radius,
                pred_center.y - sin(target_angle_to_world) * radius,
                height
            );
            const float can_shoot_angle = is_outpost() ? OUTPOST_CAN_SHOOT_ANGLE : ANTITOP_CAN_SHOOT_ANGLE;
            const bool shoot_flag = abs(target_angle_to_gimbal) < can_shoot_angle;
            return std::make_tuple(target, shoot_flag && tracker_status != TRACKER_STATUS::CONVERGING);
        } else { // 去下一块装甲板出现位置准备射击
            const float next_follow_angle_to_world =
                math::rad_period_correction((kf_yaw_->palstance > 0 ? -1 : 1) * follow_angle + gimbal_yaw);
            // 这里要用target_armor_id而不是下一块的id取radius和height
            // 因为下一块装甲板即将进入击打范围时target_armor_id就是目标
            const float radius = is_outpost() ? OUTPOST_RADIUS : radius_[target_armor_id % 2];
            const float height = is_outpost() ? height_[0] : height_[target_armor_id % 4];
            const cv::Point3f target = cv::Point3f(
                pred_center.x - cos(next_follow_angle_to_world) * radius,
                pred_center.y - sin(next_follow_angle_to_world) * radius,
                height
            );
            return std::make_tuple(target, false);
        }
    }
}

void Tracker::update_radius() {
    if (!is_outpost() && armors_.size() == 2) {
        const unsigned index = observing_armor_id_ % 2;
        const float delta_x = armors_[1].center.x - armors_[0].center.x;
        const float delta_y = armors_[1].center.y - armors_[0].center.y;
        const float theta = armors_[0].angle;
        const float r_first = abs(delta_x * cos(theta) + delta_y * sin(theta));
        const float r_next = abs(-delta_x * sin(theta) + delta_y * cos(theta));
        if (MIN_RADIUS <= r_first && r_first <= MAX_RADIUS) {
            radius_[index] = RADIUS_FILTER_RATIO * radius_[index]
                + (1 - RADIUS_FILTER_RATIO) * r_first;
        }
    }
}

void Tracker::update_height() {
    const unsigned index = is_outpost() ? 0 : observing_armor_id_;
    height_[index] = HEIGHT_FILTER_RATIO * height_[index]
        + (1 - HEIGHT_FILTER_RATIO) * armors_[0].center.z;
}

void Tracker::load_params(const std::string& params_path) {
    cv::FileStorage fs(params_path, cv::FileStorage::READ);
    fs["Tracker"]["initial_radius"] >> INITIAL_RADIUS;
    fs["Tracker"]["min_radius"] >> MIN_RADIUS;
    fs["Tracker"]["max_radius"] >> MAX_RADIUS;
    fs["Tracker"]["switch_armor_angle"] >> SWITCH_ARMOR_ANGLE;
    fs["Tracker"]["radius_filter_ratio"] >> RADIUS_FILTER_RATIO;
    fs["Tracker"]["height_filter_ratio"] >> HEIGHT_FILTER_RATIO;
    fs["Tracker"]["antitop_palstance_threshold"] >> ANTITOP_PALSTANCE_THRESHOLD;
    fs["Tracker"]["antitop_follow_angle"] >> ANTITOP_FOLLOW_ANGLE;
    fs["Tracker"]["antitop_can_shoot_angle"] >> ANTITOP_CAN_SHOOT_ANGLE;
    fs["Tracker"]["max_lost_frames"] >> MAX_LOST_FRAMES;
    fs["Tracker"]["converge_frames"] >> CONVERGE_FRAMES;

    fs["Tracker"]["outpost_radius"] >> OUTPOST_RADIUS;
    fs["Tracker"]["outpost_max_lost_frames"] >> OUTPOST_MAX_LOST_FRAMES;
    fs["Tracker"]["outpost_can_shoot_angle"] >> OUTPOST_CAN_SHOOT_ANGLE;
    fs.release();
}

void Tracker::debug_print_state() {
    std::printf("----------\n");
    std::printf("current status: ");
    if (tracker_status == TRACKER_STATUS::CONVERGING) {
        printf("converging, %d\n", current_status_frames_);
    } else if (tracker_status == TRACKER_STATUS::TRACKING) {
        printf("tracking, %d\n", current_status_frames_);
    } else if (tracker_status == TRACKER_STATUS::LOST) {
        printf("lost, %d\n", current_status_frames_);
    } else if (tracker_status == TRACKER_STATUS::TEMP_LOST) {
        printf("temp_lost, %d\n", current_status_frames_);
    }
    std::printf(
        "kf xyz: [%3.0f, %3.0f, %3.0f] += [%3.0f, %3.0f, %3.0f] (cm)\n",
        kf_xyz_->position.x * 100,
        kf_xyz_->position.y * 100,
        kf_xyz_->position.z * 100,
        kf_xyz_->velocity.x * 100,
        kf_xyz_->velocity.y * 100,
        kf_xyz_->velocity.z * 100
    );
    std::printf(
        "kf yaw: %5.0f += %3.0f (degree)\n",
        math::r2d(kf_yaw_->yaw),
        math::r2d(kf_yaw_->palstance)
    );
    std::printf(
        "ukf center: [%3.0f, %3.0f] += [%3.0f, %3.0f] (cm)\n",
        ukf_->position.x * 100,
        ukf_->position.y * 100,
        ukf_->velocity.x * 100,
        ukf_->velocity.y * 100
    );
    std::printf("radius: %3.0f, %3.0f (cm)\n", radius_[0] * 100, radius_[1] * 100);
    std::printf("height: %3.0f, %3.0f, %3.0f, %3.0f (cm)\n",
        height_[0] * 100,
        height_[1] * 100,
        height_[2] * 100,
        height_[3] * 100
    );
}

void Tracker::get_debug_info(hw_sentry_interfaces::msg::DebugInfo& debug_info) {
    debug_info.tracker_status = static_cast<int>(tracker_status);
    debug_info.current_status_frames = current_status_frames_;
    debug_info.observing_armor_id = observing_armor_id_;
    constexpr auto cvpt3_to_tfpt = [](const cv::Point3f& p) {
        geometry_msgs::msg::Point32 ret;
        ret.x = p.x, ret.y = p.y, ret.z = p.z;
        return ret;
    };
    constexpr auto cvpt2_to_tfpt = [](const cv::Point2f& p) {
        geometry_msgs::msg::Point32 ret;
        ret.x = p.x, ret.y = p.y;
        return ret;
    };
    debug_info.kf_xyz_position = cvpt3_to_tfpt(kf_xyz_->position);
    debug_info.kf_xyz_velocity = cvpt3_to_tfpt(kf_xyz_->velocity);
    debug_info.ukf_xy_position = cvpt2_to_tfpt(ukf_->position);
    debug_info.ukf_xy_velocity = cvpt2_to_tfpt(ukf_->velocity);
    debug_info.kf_yaw = kf_yaw_->yaw;
    debug_info.kf_yaw_palstance = kf_yaw_->palstance;
    for (int i = 0; i < 2; i++) {
        debug_info.radius.emplace_back(radius_[i]);
        debug_info.height.emplace_back(height_[i]);
    }
}