#include <car_tracker.hpp>

CarTracker::CarTracker(const std::string& params_path) {
    load_params(params_path);
    kf_xyz_ = std::make_unique<KFXYZ>(params_path);
    kf_yaw_ = std::make_unique<KFYaw>(params_path);
    ukf_ = std::make_unique<UKFXY>(params_path);
}

void CarTracker::push(const tf2::Transform& transform) {
    Armor armor;
    armor.center = utils::convert_to<cv::Point3f>(transform.getOrigin());
    auto ypr = utils::to_euler_ypr(transform.getRotation());
    armor.angle = std::get<0>(ypr);
    armors_.emplace_back(armor);
}

void CarTracker::update(const double timestamp, const int label) {
    target_label_ = label;
    using TS = TrackerStatus;
    const float time_elapsed = static_cast<float>(timestamp - prev_update_time_); // 和上一帧比经过的时间

    if (armors_.empty() || armors_.size() > 2) {
        if (tracker_status_ != TS::LOST) { // 短暂失踪，只预测不更新
            if (tracker_status_ == TS::CONVERGING) {
                tracker_status_ = TS::LOST;
                current_status_frames_ = 0;
            } else if (tracker_status_ == TS::TRACKING) {
                tracker_status_ = TS::TEMP_LOST;
                current_status_frames_ = 0;
            } else if (tracker_status_ == TS::TEMP_LOST) {
                if (current_status_frames_ > (is_outpost() ? OUTPOST_MAX_LOST_FRAMES : MAX_LOST_FRAMES)) {
                    tracker_status_ = TS::LOST;
                    current_status_frames_ = 0;
                }
            }
            kf_xyz_->predict(time_elapsed);
            kf_yaw_->predict(time_elapsed);
            ukf_->predict(time_elapsed);
        }
        current_status_frames_++;
    } else {
        if (tracker_status_ == TS::LOST) { // 初始化
            tracker_status_ = TS::CONVERGING;
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
            if (tracker_status_ == TS::TEMP_LOST) {
                tracker_status_ = TS::TRACKING;
                current_status_frames_ = 0;
            } else if (tracker_status_ == TS::CONVERGING) {
                if (current_status_frames_ > CONVERGE_FRAMES) {
                    tracker_status_ = TS::TRACKING;
                    current_status_frames_ = 0;
                }
            }
            current_status_frames_++;
            kf_xyz_->predict(time_elapsed);
            kf_yaw_->predict(time_elapsed);
            ukf_->predict(time_elapsed);
            float delta_angle = utils::rad_period_correction(armors_[0].angle - prev_update_angle_);
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
                observing_armor_id_ += (armors_count - 1);
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
    prev_update_time_ = timestamp;
}

bool CarTracker::decide_antitop_mode() {
    if (is_antitop_palstance_ && abs(kf_yaw_->palstance) < EXIT_ANTITOP_PALSTANCE_THRESHOLD) {
        is_antitop_palstance_ = 0;
    } else if (!is_antitop_palstance_ && abs(kf_yaw_->palstance) > ENTER_ANTITOP_PALSTANCE_THRESHOLD) {
        is_antitop_palstance_ = 1;
    }
    return is_antitop_palstance_ || is_outpost();
}

float CarTracker::get_img_to_hit_time(
    const float bullet_speed,
    const float img_to_fire_time,
    const cv::Point3f fric_to_basis
) {
    bool is_antitop_mode = decide_antitop_mode();
    const cv::Point3f target_to_basis = is_antitop_mode
        ? cv::Point3f(ukf_->position.x, ukf_->position.y, height_[0])
        : kf_xyz_->position;
    const cv::Point3f target_speed = is_antitop_mode
        ? cv::Point3f(ukf_->velocity.x, ukf_->velocity.y, 0)
        : kf_xyz_->velocity;
    float fly_time = 0;
    for (int i = 0; i < 5; i++) {
        const cv::Point3f pred_target_to_basis = target_to_basis + target_speed * (img_to_fire_time + fly_time);
        const cv::Point3f pred_target_to_fric = pred_target_to_basis - fric_to_basis;
        std::tie(std::ignore, fly_time) = trajectory::get_pitch_air_frac(
            std::hypot(pred_target_to_fric.x, pred_target_to_fric.y),
            pred_target_to_fric.z,
            bullet_speed
        );
    }
    return img_to_fire_time + fly_time;
}

std::tuple<cv::Point3f, bool> CarTracker::get_target_pos(
    const float gimbal_yaw,
    const float img_to_hit_time
) {
    const bool is_antitop_mode = decide_antitop_mode();
    if (!is_antitop_mode) {
        return std::make_tuple(
            kf_xyz_->position + kf_xyz_->velocity * img_to_hit_time, 
            tracker_status_ != TrackerStatus::CONVERGING
        );
    } else {
        const cv::Point2f pred_center = ukf_->position + ukf_->velocity * img_to_hit_time;
        // 0号装甲板在世界系下的预测yaw角
        const float pred_yaw_to_world = kf_yaw_->yaw + kf_yaw_->palstance * img_to_hit_time;
        // 0号装甲板在gimbal系下的预测yaw角
        const float pred_yaw_to_gimbal = utils::rad_period_correction(pred_yaw_to_world - gimbal_yaw);
        // 车的装甲板数量，前哨站只有三个装甲板
        const unsigned armors_count = is_outpost() ? 3 : 4;
        // 最面向我们的装甲板在gimbal系下的预测角
        float target_angle_to_gimbal = M_PI * 2 / armors_count; 
        unsigned target_armor_id = 0;
        // 选择在img_to_hit_time之后，角度最小（即最面向我们）的那个装甲板
        for (unsigned i = 0; i < armors_count; i++) {
            const float pred_angle_to_gimbal =
                utils::rad_period_correction(pred_yaw_to_gimbal - 2 * M_PI * i / armors_count);
            if (abs(pred_angle_to_gimbal) < abs(target_angle_to_gimbal)) {
                target_angle_to_gimbal = pred_angle_to_gimbal;
                target_armor_id = i;
            }
        }
        const float follow_angle = is_outpost() ? OUTPOST_CAN_SHOOT_ANGLE : ANTITOP_FOLLOW_ANGLE;
        if (abs(target_angle_to_gimbal) < follow_angle) { // 跟随射击
            const float target_angle_to_world = 
                utils::rad_period_correction(target_angle_to_gimbal + gimbal_yaw);
            const float radius = is_outpost() ? OUTPOST_RADIUS : radius_[target_armor_id % 2];
            const float height = is_outpost() ? height_[0] : height_[target_armor_id % 4];
            const cv::Point3f target = cv::Point3f(
                pred_center.x - cos(target_angle_to_world) * radius,
                pred_center.y - sin(target_angle_to_world) * radius,
                height
            );
            const float can_shoot_angle = is_outpost() ? OUTPOST_CAN_SHOOT_ANGLE : ANTITOP_CAN_SHOOT_ANGLE;
            const bool shoot_flag = abs(target_angle_to_gimbal) < can_shoot_angle;
            return std::make_tuple(target, shoot_flag && tracker_status_ != TrackerStatus::CONVERGING);
        } else { // 去下一块装甲板出现位置准备射击
            const float next_follow_angle_to_world =
                utils::rad_period_correction((kf_yaw_->palstance > 0 ? -1 : 1) * follow_angle + gimbal_yaw);
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

void CarTracker::update_radius() {
    if (!is_outpost() && armors_.size() == 2) {
        const unsigned index = observing_armor_id_ % 2;
        const float delta_x = armors_[1].center.x - armors_[0].center.x;
        const float delta_y = armors_[1].center.y - armors_[0].center.y;
        const float theta = armors_[0].angle;
        const float r_first = abs(delta_x * cos(theta) + delta_y * sin(theta));
        if (MIN_RADIUS <= r_first && r_first <= MAX_RADIUS) {
            radius_[index] = RADIUS_FILTER_RATIO * radius_[index]
                + (1 - RADIUS_FILTER_RATIO) * r_first;
        }
    }
}

void CarTracker::update_height() {
    const unsigned index = is_outpost() ? 0 : observing_armor_id_;
    height_[index] = HEIGHT_FILTER_RATIO * height_[index]
        + (1 - HEIGHT_FILTER_RATIO) * armors_[0].center.z;
}

void CarTracker::load_params(const std::string& params_path) {
    cv::FileStorage fs(params_path, cv::FileStorage::READ);
    fs["CarTracker"]["initial_radius"] >> INITIAL_RADIUS;
    fs["CarTracker"]["min_radius"] >> MIN_RADIUS;
    fs["CarTracker"]["max_radius"] >> MAX_RADIUS;
    fs["CarTracker"]["switch_armor_angle"] >> SWITCH_ARMOR_ANGLE;
    fs["CarTracker"]["radius_filter_ratio"] >> RADIUS_FILTER_RATIO;
    fs["CarTracker"]["height_filter_ratio"] >> HEIGHT_FILTER_RATIO;
    fs["CarTracker"]["enter_antitop_palstance_threshold"] >> ENTER_ANTITOP_PALSTANCE_THRESHOLD;
    fs["CarTracker"]["exit_antitop_palstance_threshold"] >> EXIT_ANTITOP_PALSTANCE_THRESHOLD;
    fs["CarTracker"]["antitop_follow_angle"] >> ANTITOP_FOLLOW_ANGLE;
    fs["CarTracker"]["antitop_can_shoot_angle"] >> ANTITOP_CAN_SHOOT_ANGLE;
    MAX_LOST_FRAMES = (int)fs["CarTracker"]["max_lost_frames"];
    CONVERGE_FRAMES = (int)fs["CarTracker"]["converge_frames"];

    fs["CarTracker"]["outpost_radius"] >> OUTPOST_RADIUS;
    fs["CarTracker"]["outpost_can_shoot_angle"] >> OUTPOST_CAN_SHOOT_ANGLE;
    OUTPOST_MAX_LOST_FRAMES = (int)fs["CarTracker"]["outpost_max_lost_frames"];
    
    fs.release();
}

void CarTracker::debug_print_state() {
    std::printf("----------\n");
    std::printf("current status: ");
    if (tracker_status_ == TrackerStatus::CONVERGING) {
        printf("converging, %d\n", current_status_frames_);
    } else if (tracker_status_ == TrackerStatus::TRACKING) {
        printf("tracking, %d\n", current_status_frames_);
    } else if (tracker_status_ == TrackerStatus::LOST) {
        printf("lost, %d\n", current_status_frames_);
    } else if (tracker_status_ == TrackerStatus::TEMP_LOST) {
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
        "kf yaw: [%5.0f] += [%3.0f] (degree)\n",
        utils::r2d(kf_yaw_->yaw),
        utils::r2d(kf_yaw_->palstance)
    );
    std::printf(
        "ukf center: [%3.0f, %3.0f] += [%3.0f, %3.0f] (cm)\n",
        ukf_->position.x * 100,
        ukf_->position.y * 100,
        ukf_->velocity.x * 100,
        ukf_->velocity.y * 100
    );
    std::printf("radius: [%3.0f, %3.0f] (cm)\n", radius_[0] * 100, radius_[1] * 100);
    std::printf("height: [%3.0f, %3.0f, %3.0f, %3.0f] (cm)",
        height_[0] * 100,
        height_[1] * 100,
        height_[2] * 100,
        height_[3] * 100
    );
    std::cout << std::endl;
}