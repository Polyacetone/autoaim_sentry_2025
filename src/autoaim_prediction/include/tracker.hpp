// 维护一个整车状态的跟踪器
// KFXYZ用于平动，KFYaw+UKFXY用于小陀螺，半径和高度采用惯性滤波
// 坐标系定义：除相机系（这里应该没涉及）外，其余都是向右x，向前y，向上z
// yaw角方向定义：逆时针（即从x到y）为正
// 距离和时间均使用国际单位制（m、s），角度使用弧度制

#pragma once

#include <kalman_filters.hpp>
#include <math_utils.hpp>

enum class TRACKER_STATUS { CONVERGING, TRACKING, TEMP_LOST, LOST };

struct Armor {
    cv::Point3f center; // 装甲板中心坐标
    float angle; // 装甲板向心方向在xy平面的投影向量与正前方（y轴）的夹角，逆时针为正
};

class Tracker {
public:
    Tracker(const std::string& params_path);
    void push(const geometry_msgs::msg::Transform& transform);
    void update(const double time_stamp);
    void debug_print_state();

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
    float SWITCH_ARMOR_ANGLE = math::d2r(50);
    float CLOSE_RADIUS_FILTER_RATIO = 0.7;
    float FAR_RADIUS_FILTER_RATIO = 0.8;
    float CLOSE_HEIGHT_FILTER_RATIO = 0.6;
    float FAR_HEIGHT_FILTER_RATIO = 0.7;
    float ANTITOP_PALSTANCE_THRESHOLD = math::d2r(50);
    float ANTITOP_CAN_SHOOT_ANGLE = math::d2r(30);
    float ANTITOP_FOLLOW_ANGLE = math::d2r(30);
    int MAX_LOST_FRAMES = 5;
    int CONVERGE_FRAMES = 5;

    std::shared_ptr<KFXYZ> kf_xyz_;
    std::shared_ptr<KFYaw> kf_yaw_;
    std::shared_ptr<UKFXY> ukf_;
    unsigned tracking_frames_ = 0; // 连续出现的帧数
    unsigned lost_frames_ = 0; // 消失的帧数
    unsigned observing_armor_id_ = 0; // 正在观测的装甲板编号。定义第一块看到的装甲板为0，车逆时针转时看到的依次编号1、2、3
    float radius_[2]; // radius_[0]对应0、2装甲板半径，radius_[1]对应1、3
    float height_[2]; // height_[0]对应0、2装甲板中心z坐标，height_[1]对应1、3
    std::vector<Armor> armors_;

    void load_params(const std::string& params_path);
    void update_radius();
    void update_height();
};

Tracker::Tracker(const std::string& params_path) {
    load_params(params_path);
    kf_xyz_ = std::make_shared<KFXYZ>(params_path);
    kf_yaw_ = std::make_shared<KFYaw>(params_path);
    ukf_ = std::make_shared<UKFXY>(params_path);
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

void Tracker::update(const double time_stamp) {
    using TS = TRACKER_STATUS;
    static double prev = time_stamp;
    const float time_elapsed = static_cast<float>(time_stamp - prev); // 和上一帧比经过的时间

    if (armors_.empty() || armors_.size() > 2) {
        tracking_frames_ = 0;
        lost_frames_++;
        if (tracker_status != TS::LOST) { // 短暂失踪，只预测不更新
            kf_xyz_->predict(time_elapsed);
            kf_yaw_->predict(time_elapsed);
            ukf_->predict(time_elapsed);
            if (lost_frames_ >= MAX_LOST_FRAMES) {
                tracker_status = TS::LOST;
            } else {
                tracker_status = TS::TEMP_LOST;
            }
        }
    } else {
        lost_frames_ = 0;
        tracking_frames_++;
        if (tracker_status == TS::LOST) { // 初始化
            kf_xyz_->initialize(armors_[0].center);
            kf_yaw_->initialize(armors_[0].angle);
            const cv::Point2f car_center(
                armors_[0].center.x + INITIAL_RADIUS * cos(armors_[0].angle),
                armors_[0].center.y + INITIAL_RADIUS * sin(armors_[0].angle)
            );
            ukf_->initialize(car_center);
            observing_armor_id_ = 0;
            radius_[0] = radius_[1] = INITIAL_RADIUS;
            height_[0] = height_[1] = armors_[0].center.z;
        } else { // 正常预测并更新
            kf_xyz_->predict(time_elapsed);
            kf_yaw_->predict(time_elapsed);
            ukf_->predict(time_elapsed);
            if (armors_[0].angle - kf_yaw_->yaw > M_PI) {
                armors_[0].angle -= M_PI * 2;
            } else if (armors_[0].angle - kf_yaw_->yaw < -M_PI) {
                armors_[0].angle += M_PI * 2;
            }
            if (armors_[0].angle - kf_yaw_->yaw < -SWITCH_ARMOR_ANGLE) { // 逆时针转（角速度大于0）时切换装甲板
                observing_armor_id_++;
                observing_armor_id_ %= 4;
                kf_yaw_->force_change_yaw(armors_[0].angle);
                kf_xyz_->force_change_position(armors_[0].center);
            } else if (armors_[0].angle - kf_yaw_->yaw > SWITCH_ARMOR_ANGLE) { // 顺时针转（角速度小于0）时切换装甲板
                observing_armor_id_ += 3;
                observing_armor_id_ %= 4;
                kf_yaw_->force_change_yaw(armors_[0].angle);
                kf_xyz_->force_change_position(armors_[0].center);
            }
            kf_xyz_->update(armors_[0].center);
            kf_yaw_->update(armors_[0].angle);
            update_radius();
            update_height();
            const cv::Point2f car_center(
                armors_[0].center.x + radius_[observing_armor_id_ % 2] * cos(armors_[0].angle),
                armors_[0].center.y + radius_[observing_armor_id_ % 2] * sin(armors_[0].angle)
            );
            ukf_->update(car_center);
        }
        if (tracking_frames_ >= CONVERGE_FRAMES) {
            tracker_status = TS::TRACKING;
        } else {
            tracker_status = TS::CONVERGING;
        }
    }
    
    armors_.clear();
    prev = time_stamp;
}

std::tuple<cv::Point3f, bool> Tracker::get_target_pos(
    const float gimbal_yaw,
    const float bullet_speed, 
    const float img_to_fire_time
) {
    if (abs(kf_yaw_->palstance) < ANTITOP_PALSTANCE_THRESHOLD) { // 平动，只用KFXYZ预测
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
        float target_angle_to_gimbal = M_PI / 2; // 最面向我们的装甲板在gimbal系下的预测角
        int target_armor_index = 0;
        // 选择在img_to_hit_time之后，角度最小（即最面向我们）的那个装甲板
        for (int i = 0; i < 4; i++) {
            const float pred_angle_to_gimbal = 
                math::rad_period_correction(pred_yaw_to_gimbal + i * M_PI / 2);
            if (abs(pred_angle_to_gimbal) < abs(target_angle_to_gimbal)) {
                target_angle_to_gimbal = pred_angle_to_gimbal;
                target_armor_index = (observing_armor_id_ + i) % 2;
            }
        }
        if (abs(target_angle_to_gimbal) < ANTITOP_FOLLOW_ANGLE) { // 跟随射击
            const float target_angle_to_world = 
                math::rad_period_correction(target_angle_to_gimbal + gimbal_yaw);
            const cv::Point3f target = cv::Point3f(
                pred_center.x - cos(target_angle_to_world) * radius_[target_armor_index],
                pred_center.y - sin(target_angle_to_world) * radius_[target_armor_index],
                height_[target_armor_index]
            );
            const bool shoot_flag = abs(target_angle_to_gimbal) < ANTITOP_CAN_SHOOT_ANGLE;
            return std::make_tuple(target, shoot_flag && tracker_status != TRACKER_STATUS::CONVERGING);
        } else { // 去下一块装甲板出现位置准备射击
            const float next_follow_angle_to_world = math::rad_period_correction(
                (kf_yaw_->palstance > 0 ? -1 : 1) * ANTITOP_FOLLOW_ANGLE + gimbal_yaw
            );
            const cv::Point3f target = cv::Point3f(
                pred_center.x - cos(next_follow_angle_to_world) * radius_[1 - target_armor_index],
                pred_center.y - sin(next_follow_angle_to_world) * radius_[1 - target_armor_index],
                height_[1 - target_armor_index]
            );
            return std::make_tuple(target, false);
        }
    }
}

void Tracker::update_radius() {
    if (armors_.size() == 2) {
        const int index = observing_armor_id_ % 2;
        const float delta_x = armors_[1].center.x - armors_[0].center.x;
        const float delta_y = armors_[1].center.y - armors_[0].center.y;
        const float theta = armors_[0].angle;
        const float r_first = abs(delta_x * cos(theta) + delta_y * sin(theta));
        const float r_next = abs(-delta_x * sin(theta) + delta_y * cos(theta));
        if (MIN_RADIUS <= r_first && r_first <= MAX_RADIUS) {
            radius_[index] = CLOSE_RADIUS_FILTER_RATIO * radius_[index]
                + (1 - CLOSE_RADIUS_FILTER_RATIO) * r_first;
        }
        if (MIN_RADIUS <= r_next && r_next <= MAX_RADIUS) {
            radius_[1 - index] = FAR_RADIUS_FILTER_RATIO * radius_[1 - index]
                + (1 - FAR_RADIUS_FILTER_RATIO) * r_next;
        }
    }
}

void Tracker::update_height() {
    const int index = observing_armor_id_ % 2;
    height_[index] = CLOSE_HEIGHT_FILTER_RATIO * height_[index]
        + (1 - CLOSE_HEIGHT_FILTER_RATIO) * armors_[0].center.z;
    if (armors_.size() == 2) {
        height_[1 - index] = FAR_HEIGHT_FILTER_RATIO * height_[1 - index]
            + (1 - FAR_HEIGHT_FILTER_RATIO) * armors_[1].center.z;
    }
}

void Tracker::load_params(const std::string& params_path) {
    cv::FileStorage fs(params_path, cv::FileStorage::READ);
    fs["Tracker"]["initial_radius"] >> INITIAL_RADIUS;
    fs["Tracker"]["min_radius"] >> MIN_RADIUS;
    fs["Tracker"]["max_radius"] >> MAX_RADIUS;
    fs["Tracker"]["switch_armor_angle"] >> SWITCH_ARMOR_ANGLE;
    fs["Tracker"]["close_radius_filter_ratio"] >> CLOSE_RADIUS_FILTER_RATIO;
    fs["Tracker"]["far_radius_filter_ratio"] >> FAR_RADIUS_FILTER_RATIO;
    fs["Tracker"]["close_height_filter_ratio"] >> CLOSE_HEIGHT_FILTER_RATIO;
    fs["Tracker"]["far_height_filter_ratio"] >> FAR_HEIGHT_FILTER_RATIO;
    fs["Tracker"]["antitop_palstance_threshold"] >> ANTITOP_PALSTANCE_THRESHOLD;
    fs["Tracker"]["antitop_follow_angle"] >> ANTITOP_FOLLOW_ANGLE;
    fs["Tracker"]["antitop_can_shoot_angle"] >> ANTITOP_CAN_SHOOT_ANGLE;
    fs["Tracker"]["max_lost_frames"] >> MAX_LOST_FRAMES;
    fs["Tracker"]["converge_frames"] >> CONVERGE_FRAMES;
    fs.release();
}

void Tracker::debug_print_state() {
    std::printf("----------\n");
    std::printf("current status: ");
    if (tracker_status == TRACKER_STATUS::CONVERGING) {
        printf("converging, tracking_frames: %d\n", tracking_frames_);
    } else if (tracker_status == TRACKER_STATUS::TRACKING) {
        printf("tracking, tracking_frames: %d\n", tracking_frames_);
    } else if (tracker_status == TRACKER_STATUS::LOST) {
        printf("lost, lost_frames: %d\n", lost_frames_);
    } else if (tracker_status == TRACKER_STATUS::TEMP_LOST) {
        printf("temp_lost, lost_frames: %d\n", lost_frames_);
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
    std::printf("height: %3.0f, %3.0f (cm)\n", height_[0] * 100, height_[1] * 100);
}