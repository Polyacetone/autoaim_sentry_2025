#include <autoaim_predictor/buff_tracker.hpp>

using namespace Eigen;
using Scalarf = Vector<float, 1>;

static constexpr float BUFF_RADIUS = 0.7f; // Buff的半径
static constexpr float ADJACENT_LEAF_ANGLE = 2 * M_PI / 5; // 相邻符叶的夹角

/**********************************************************************************
***********************************    Utils    ***********************************
***********************************************************************************/

Vector3f calc_leaf_position(Vector3f R_center, float angle) {
    Vector3f leaf_position;
    leaf_position.x() = R_center.x();
    leaf_position.y() = R_center.y() - BUFF_RADIUS * std::cos(angle);
    leaf_position.z() = R_center.z() + BUFF_RADIUS * std::sin(angle);
    return leaf_position;
}

/**********************************************************************************
***********************************    Buff    ************************************
***********************************************************************************/

Buff::Buff(const tf2::Transform& buff_pose) {
    translation = utils::convert_to<Vector3f>(buff_pose.getOrigin());
    rotation = utils::convert_to<Quaternionf>(buff_pose.getRotation());
    R_center = rotation * Vector3f(0, 0, -BUFF_RADIUS) + translation;
    angle = std::atan2(translation.z() - R_center.z(), R_center.y() - translation.y());
}

/**********************************************************************************
******************************  SmallBuffObserver  ********************************
***********************************************************************************/

SmallBuffObserver::SmallBuffObserver(const cv::FileNode& fn) {
    SMALL_BUFF_SPEED = static_cast<float>(fn["small_buff_speed"]);
    SWITCH_BUFF_ANGLE = static_cast<float>(fn["switch_buff_angle"]);
    R_center_ = std::make_unique<EMAF<3>>(fn["R_center_filter_ratio"]);
    kf_angle_ = std::make_unique<KF<1>>(fn["kf_angle"]);
    reset();
}

void SmallBuffObserver::reset() {
    R_center_->reset();
    kf_angle_->reset();
}

void SmallBuffObserver::initialize(const std::vector<Buff>& buffs) {
    reset();
    kf_angle_->initialize(Scalarf(buffs[0].angle));
    R_center_->initialize(buffs[0].R_center);
}

void SmallBuffObserver::predict(const float time_elapsed) const {
    kf_angle_->predict(time_elapsed);
    if (kf_angle_->value().value() > M_PI) {
        kf_angle_->force_change_value(Scalarf(kf_angle_->value().value() - M_PI * 2));
    } else if (kf_angle_->value().value() < -M_PI) {
        kf_angle_->force_change_value(Scalarf(kf_angle_->value().value() + M_PI * 2));
    }
}

void SmallBuffObserver::update(const std::vector<Buff>& buffs) {
    const float buff_angle = utils::rad_period_correction(buffs[0].angle);
    if (std::abs(buff_angle - kf_angle_->value().value()) > SWITCH_BUFF_ANGLE) {
        kf_angle_->force_change_value(Scalarf(buff_angle));
    } else {
        kf_angle_->update(Scalarf(buff_angle));
    }
    R_center_->update(buffs[0].R_center);
}

Vector3f SmallBuffObserver::predict_shoot_pos(
    const float bullet_speed,
    const float img_to_fire_time,
    const Vector3f fric_to_gimbal_yaw
) const {
    float fly_time = 0;
    int direction_sign = (kf_angle_->derivative().value() < 0) ? -1 : 1;
    for (int i = 0; i < 5; i++) {
        const float pred_angle = kf_angle_->value().value() + direction_sign * SMALL_BUFF_SPEED * (img_to_fire_time + fly_time);
        Vector3f pred_leaf_position = calc_leaf_position(R_center_->value(), pred_angle);
        Vector3f target_to_fake_fric = pred_leaf_position - fric_to_gimbal_yaw;
        fly_time = std::get<1>(trajectory::get_pitch_air_frac(
            std::hypot(target_to_fake_fric.x(), target_to_fake_fric.y()),
            target_to_fake_fric.z(),
            bullet_speed
        ));
    }
    float img_to_hit_time = img_to_fire_time + fly_time;
    const float pred_angle = kf_angle_->value().value() + direction_sign * SMALL_BUFF_SPEED * img_to_hit_time;
    Vector3f pred_leaf_position = calc_leaf_position(R_center_->value(), pred_angle);
    return pred_leaf_position;
}

Vector3f SmallBuffObserver::get_R_center() const { return R_center_->value(); }

void SmallBuffObserver::print_colored_status_info() const {
    const auto print_vec = [](const char* format, Vector3f vec) {
        std::printf(format, vec.x(), vec.y(), vec.z());
    };
    std::cout << termcolor::bold << "SmallBuff.RCenter   " << termcolor::reset;
    print_vec("[% 4.0f, % 4.0f, % 4.0f]\n", R_center_->value() * 100);
    std::cout << termcolor::bold << "SmallBuff.Theta     " << termcolor::reset;
    std::printf(
        "[% 4.0f] += [% 4.0f] (%s)",
        utils::r2d(kf_angle_->value().value()),
        utils::r2d(kf_angle_->derivative().value()),
        kf_angle_->derivative().value() > 0 ? "counterclockwise" : "clockwise"
    );
    std::cout << std::endl;
}

void SmallBuffObserver::write_predictor_status(hw_sentry_interfaces::msg::PredictorStatus& status) const {
    status.r_center = utils::convert_to<geometry_msgs::msg::Point32>(R_center_->value());
    status.theta = kf_angle_->value().value();
}

/**********************************************************************************
*********************************  BuffTracker  ***********************************
***********************************************************************************/

BuffTracker::BuffTracker(const cv::FileNode& fn) {
    TEMP_LOST_RETURN_FRAMES = static_cast<int>(fn["temp_lost_return_frames"]);
    small_buff_status_ = std::make_unique<TrackerStatus>(
        fn["small_buff_status"],
        [this](StatusType from, StatusType to) { status_change_handler(from, to); },
        [this](StatusType curr) { status_remain_handler(curr); }
    );
    small_buff_observer_ = std::make_unique<SmallBuffObserver>(fn["small_buff_observer"]);
}

StatusType BuffTracker::status() const {
    if (mode_ == AutoaimMode::SMALL_BUFF) {
        return small_buff_status_->status();
    } else if (mode_ == AutoaimMode::BIG_BUFF) {
        // ...
    }
    return {};
}

void BuffTracker::push(const Buff& buff) {
    pushed_buffs_.push_back(buff);
}

void BuffTracker::set_mode(const AutoaimMode mode) {
    if (mode_ != mode) {
        mode_ = mode;
        reset();
    }
}

void BuffTracker::reset() {
    pushed_buffs_.clear();
    small_buff_status_->reset();
    small_buff_observer_->reset();
}

void BuffTracker::update(const double timestamp) {
    current_update_time_ = timestamp;
    bool is_valid = (pushed_buffs_.size() == 1);
    // 更新状态机，状态机会根据状态调用跟踪器的更新
    if (mode_ == AutoaimMode::SMALL_BUFF) {
        small_buff_status_->update(is_valid);
    } else if (mode_ == AutoaimMode::BIG_BUFF) {
        // ...
    }
    prev_update_time_ = current_update_time_;
    pushed_buffs_.clear();
}

void BuffTracker::status_change_handler(StatusType from, StatusType to) {
    if (from == StatusType::LOST && to == StatusType::CONVERGING) { // 初始化
        if (mode_ == AutoaimMode::SMALL_BUFF) {
            small_buff_observer_->initialize(pushed_buffs_);
        } else if (mode_ == AutoaimMode::BIG_BUFF) {
            // ...
        }
    }
}

void BuffTracker::status_remain_handler(StatusType current) {
    if (current != StatusType::LOST) { // 预测
        const float time_elapsed = static_cast<float>(current_update_time_ - prev_update_time_);
        if (mode_ == AutoaimMode::SMALL_BUFF) {
            small_buff_observer_->predict(time_elapsed);
        } else {
            // ...
        }
    }
    if (current == StatusType::CONVERGING || current == StatusType::TRACKING) { // 更新
        if (mode_ == AutoaimMode::SMALL_BUFF) {
            small_buff_observer_->update(pushed_buffs_);
        } else if (mode_ == AutoaimMode::BIG_BUFF) {
            // ...
        }
    }
}

std::tuple<Vector3f, bool> BuffTracker::predict_shoot_pos(
    const float bullet_speed,
    const float img_to_fire_time,
    const Vector3f fric_to_gimbal_yaw
) const {
    if (mode_ == AutoaimMode::SMALL_BUFF) {
        if (small_buff_status_->status() == StatusType::CONVERGING ||
            (small_buff_status_->status() == StatusType::TEMP_LOST &&
            small_buff_status_->current_status_frames() > TEMP_LOST_RETURN_FRAMES)) {
            return {small_buff_observer_->get_R_center(), false}; // 如果收敛中或者短暂丢失大于一定时间，则指向R_center
        } else {
            return {
                small_buff_observer_->predict_shoot_pos(bullet_speed, img_to_fire_time, fric_to_gimbal_yaw),
                small_buff_status_->status() != StatusType::TEMP_LOST
            };
        }
    } else if (mode_ == AutoaimMode::BIG_BUFF) {
        // ...
    }
    return {};
}

void BuffTracker::print_colored_status_info() const {
    if (mode_ == AutoaimMode::SMALL_BUFF) {
        std::cout << termcolor::bold << "Mode: Small buff" << termcolor::reset << std::endl;
        small_buff_status_->print_colored_status_info();
        small_buff_observer_->print_colored_status_info();
    } else {
        // ...
    }
}

void BuffTracker::write_predictor_status(hw_sentry_interfaces::msg::PredictorStatus& status) const {
    status.mode = static_cast<int>(mode_);
    status.label = -1;
    if (mode_ == AutoaimMode::SMALL_BUFF) {
        status.tracker_status = static_cast<int>(small_buff_status_->status());
        small_buff_observer_->write_predictor_status(status);
    } else {
        // ...
    }
}