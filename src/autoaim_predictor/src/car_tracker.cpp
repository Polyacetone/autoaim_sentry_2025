#include <car_tracker.hpp>

using namespace Eigen;
using Scalarf = Vector<float, 1>;

/**********************************************************************************
                                    Utils
***********************************************************************************/

std::tuple<float, float> calc_radius(const Armor& armor1, const Armor& armor2) {
    const Vector3f p1p2 = armor2.translation - armor1.translation;
    const float radius1_1 = std::abs(p1p2.dot(armor1.rotated_x.normalized()));
    const float radius1_2 = std::abs(p1p2.dot(armor2.rotated_y.normalized()));
    const float radius2_1 = std::abs(p1p2.dot(armor1.rotated_y.normalized()));
    const float radius2_2 = std::abs(p1p2.dot(armor2.rotated_x.normalized()));
    const float radius1 = (radius1_1 + radius1_2) / 2;
    const float radius2 = (radius2_1 + radius2_2) / 2;
    return std::make_tuple(radius1, radius2);
}

Vector3f calc_rotation_axis(const Armor& armor) {
    return armor.rotated_z.normalized();
}

Vector3f calc_rotation_axis(const Armor& armor1, const Armor& armor2) {
    const Vector3f axis = armor1.rotated_z + armor2.rotated_z;
    return axis.normalized();
}

float calc_relative_height(const Armor& reference, const Armor& target) {
    const Vector3f rt = target.translation - reference.translation;
    const float height1 = rt.dot(reference.rotated_z.normalized());
    const float height2 = rt.dot(target.rotated_z.normalized());
    return (height1 + height2) / 2;
}

Vector3f calc_car_center(
    const Armor& armor,
    const float radius,
    const Vector3f& axis,
    const float height
) {
    const Vector3f armor_pointing_center = armor.translation + armor.rotated_x.normalized() * radius;
    const Vector3f height_vec = axis.normalized() * height;
    return armor_pointing_center - height_vec;
}

Vector3f calc_armor_center(
    const Vector3f& car_center,
    const Vector3f& axis,
    const float height,
    const float radius,
    const float yaw
) {
    const Vector3f armor_pointing_center = car_center + axis.normalized() * height;
    const Vector3f armor_normal_on_xy = {std::cos(yaw), std::sin(yaw), 0};
    const Vector3f armor_normal = (
        axis.dot(axis) * armor_normal_on_xy
        - axis.dot(armor_normal_on_xy) * axis
    ).normalized();
    return armor_pointing_center - armor_normal * radius;
}

std::tuple<Vector3f, Quaternionf> calc_armor_pose(
    const Vector3f& car_center,
    const Vector3f& axis,
    const float height,
    const float radius,
    const float yaw
) {
    const Vector3f armor_pointing_center = car_center + axis.normalized() * height;
    const Vector3f armor_normal_on_xy = {std::cos(yaw), std::sin(yaw), 0};
    const Vector3f armor_normal = (
        axis.dot(axis) * armor_normal_on_xy
        - axis.dot(armor_normal_on_xy) * axis
    ).normalized();
    const Vector3f armor_center = armor_pointing_center - armor_normal * radius;
    Eigen::Matrix3f rmat;
    rmat.col(0) = armor_normal;
    rmat.col(1) = axis.cross(armor_normal).normalized();
    rmat.col(2) = axis.normalized();
    return std::make_tuple(armor_center, Quaternionf(rmat));
}

unsigned calc_another_armor_id(
    const Armor& reference,
    const Armor& target,
    const unsigned reference_armor_id,
    const unsigned armors_count
) {
    float yaw_diff = utils::rad_period_correction(reference.yaw - target.yaw);
    if (std::abs(yaw_diff - M_PI / 2) < M_PI / 2) { // yaw_diff接近pi/2
        return (reference_armor_id + 1) % armors_count;
    } else { // yaw_diff接近-pi/2
        return (reference_armor_id + armors_count - 1) % armors_count;
    }
}

/**********************************************************************************
                                    Armor
***********************************************************************************/

Armor::Armor(const tf2::Transform& armor_pose): Armor(
    utils::convert_to<Vector3f>(armor_pose.getOrigin()),
    utils::convert_to<Quaternionf>(armor_pose.getRotation())
) {}

Armor::Armor(const Eigen::Vector3f& translation, const Eigen::Quaternionf& rotation):
    translation(translation) {
    Vector3f original_x = rotation * Vector3f::UnitX();
    Vector3f original_y = rotation * Vector3f::UnitY();
    Vector3f original_z = rotation * Vector3f::UnitZ();
    AngleAxisf pitch_correction(utils::d2r(-15), original_y.normalized());
    rotated_x = pitch_correction * original_x;
    rotated_y = pitch_correction * original_y;
    rotated_z = pitch_correction * original_z;
    Vector2f rotated_x_on_xy = rotated_x.head<2>();
    rotated_x_on_xy.normalize();
    yaw = std::atan2(rotated_x_on_xy.y(), rotated_x_on_xy.x());
}

/**********************************************************************************
                                    Car
***********************************************************************************/

Car::Car(const cv::FileNode& fn) {
    INITIAL_RADIUS = static_cast<float>(fn["initial_radius"]);
    SWITCH_ARMOR_ANGLE = static_cast<float>(fn["switch_armor_angle"]);
    float radius_filter_ratio = static_cast<float>(fn["radius_filter_ratio"]);
    float height_filter_ratio = static_cast<float>(fn["height_filter_ratio"]);
    float axis_filter_ratio = static_cast<float>(fn["axis_filter_ratio"]);
    axis = std::make_unique<LPF<3>>(axis_filter_ratio);
    for (int i = 0; i < 4; i++) {
        radius[i] = std::make_unique<LPF<1>>(radius_filter_ratio);
        height[i] = std::make_unique<LPF<1>>(height_filter_ratio);
    }
    kf_center = std::make_unique<KF<3>>(fn["kf_center"]);
    kf_yaw = std::make_unique<KF<1>>(fn["kf_yaw"]);
}

/**********************************************************************************
                                    Status
***********************************************************************************/

Status::Status(
    const cv::FileNode& fn,
    std::function<void(TrackerStatus from, TrackerStatus to)> status_change_handler,
    std::function<void(TrackerStatus current)> status_remain_handler
): status_change_handler(status_change_handler), status_remain_handler(status_remain_handler) {
    MAX_TEMP_LOST_FRAMES = static_cast<int>(fn["max_temp_lost_frames"]);
    MAX_CONVERGING_FRAMES = static_cast<int>(fn["max_converging_frames"]);
}

void Status::set_next_status(TrackerStatus status) {
    if (status_ != status) {
        status_change_handler(status_, status);
        status_ = status;
        current_status_frames_ = 0;
    } else {
        status_remain_handler(status_);
        current_status_frames_++;
    }
}

void Status::reset() { set_next_status(TrackerStatus::LOST); }
TrackerStatus Status::status() const { return status_; }
unsigned Status::status_frames() const { return current_status_frames_; }

void Status::update(bool is_valid) {
    using TS = TrackerStatus;
    TS next_status = TS::LOST;
    if (is_valid) {
        switch (status_) {
            case TS::LOST: next_status = TS::CONVERGING; break;
            case TS::TEMP_LOST: next_status = TS::TRACKING; break;
            case TS::CONVERGING: {
                if (current_status_frames_ > MAX_CONVERGING_FRAMES) next_status = TS::TRACKING;
                else next_status = TS::CONVERGING;
                break;
            }
            case TS::TRACKING: next_status = TS::TRACKING; break;
        }
    } else {
        switch (status_) {
            case TS::LOST: next_status = TS::LOST; break;
            case TS::TEMP_LOST: {
                if (current_status_frames_ > MAX_TEMP_LOST_FRAMES) next_status = TS::LOST;
                else next_status = TS::TEMP_LOST;
                break;
            }
            case TS::CONVERGING: next_status = TS::LOST; break;
            case TS::TRACKING: next_status = TS::TEMP_LOST; break;
        }
    }
    set_next_status(next_status);
}

/**********************************************************************************
                                    CarTracker
***********************************************************************************/

CarTracker::CarTracker(const std::string& params_path) {
    cv::FileStorage fs(params_path, cv::FileStorage::READ);
    ENTER_ANTISPIN_PALSTANCE = static_cast<float>(fs["CarTracker"]["enter_antispin_palstance"]);
    EXIT_ANTISPIN_PALSTANCE = static_cast<float>(fs["CarTracker"]["exit_antispin_palstance"]);
    ANTISPIN_FOLLOW_ANGLE = static_cast<float>(fs["CarTracker"]["antispin_follow_angle"]);
    ANTISPIN_SHOOT_ANGLE = static_cast<float>(fs["CarTracker"]["antispin_shoot_angle"]);
    kf_main_observing_armor_ = std::make_unique<KF<3>>(fs["CarTracker"]["kf_main_observing_armor"]);
    status_ = std::make_unique<Status>(
        fs["Status"],
        [this](TrackerStatus from, TrackerStatus to) { status_change_handler(from, to); },
        [this](TrackerStatus curr) { status_remain_handler(curr); }
    );
    car_ = std::make_unique<Car>(fs["Car"]);
}

void CarTracker::reset() { pushed_armors_.clear(); status_->reset(); }
TrackerStatus CarTracker::status() const { return status_->status(); }
void CarTracker::push(const tf2::Transform& armor_pose) { pushed_armors_.emplace_back(armor_pose); }

void CarTracker::update(const double timestamp) {
    current_update_time_ = timestamp;
    bool is_valid = (pushed_armors_.size() == 1 || pushed_armors_.size() == 2);
    status_->update(is_valid);
    prev_update_time_ = current_update_time_;
    pushed_armors_.clear();
}

void CarTracker::status_change_handler(TrackerStatus from, TrackerStatus to) {
    if (from == TrackerStatus::LOST && to == TrackerStatus::CONVERGING) { // 初始化
        kf_main_observing_armor_->initialize(pushed_armors_[0].translation);
        car_->main_observing_armor_id = 0;
        car_->accumulated_yaw = pushed_armors_[0].yaw;
        car_->prev_main_observing_yaw = car_->accumulated_yaw;
        car_->kf_yaw->initialize(Scalarf(car_->accumulated_yaw));
        for (int i = 0; i < 4; i++) {
            car_->height[i]->initialize(Scalarf(0));
            car_->radius[i]->initialize(Scalarf(car_->INITIAL_RADIUS));
        }
        if (pushed_armors_.size() == 1) { // 只有一块装甲板的时候只能算旋转轴和中心
            Vector3f car_axis = calc_rotation_axis(pushed_armors_[0]);
            car_->axis->initialize(car_axis);
            Vector3f car_center = calc_car_center(
                pushed_armors_[0],
                car_->radius[0]->value().value(),
                car_axis,
                car_->height[0]->value().value()
            );
            car_->kf_center->initialize(car_center);
        } else if (pushed_armors_.size() == 2) { // 两块装甲板的时候可以算旋转轴、半径、中心、相对高度
            // armors[1]对应的装甲板编号
            unsigned another_armor_id = calc_another_armor_id(
                pushed_armors_[0],
                pushed_armors_[1],
                0, ARMORS_COUNT
            );
            Vector3f car_axis =
                calc_rotation_axis(pushed_armors_[0], pushed_armors_[1]);
            car_->axis->initialize(car_axis);
            std::tuple<float, float> radius =
                calc_radius(pushed_armors_[0], pushed_armors_[1]);
            car_->radius[0]->update(Scalarf(std::get<0>(radius)));
            car_->radius[another_armor_id]->update(Scalarf(std::get<1>(radius)));
            float another_armor_height =
                calc_relative_height(pushed_armors_[0], pushed_armors_[1]);
            car_->height[another_armor_id]->update(Scalarf(another_armor_height));
            Vector3f car_center0 = calc_car_center(
                pushed_armors_[0],
                car_->radius[0]->value().value(),
                car_axis,
                0
            );
            Vector3f car_center1 = calc_car_center(
                pushed_armors_[1],
                car_->radius[another_armor_id]->value().value(),
                car_axis,
                another_armor_height
            );
            car_->kf_center->initialize((car_center0 + car_center1) / 2);
        }
    }
}

void CarTracker::status_remain_handler(TrackerStatus current) {
    if (current != TrackerStatus::LOST) { // 预测
        const float time_elapsed = static_cast<float>(current_update_time_ - prev_update_time_);
        kf_main_observing_armor_->predict(time_elapsed);
        car_->kf_center->predict(time_elapsed);
        car_->kf_yaw->predict(time_elapsed);
    }
    if (current == TrackerStatus::CONVERGING || current == TrackerStatus::TRACKING) { // 更新
        float delta_yaw = utils::rad_period_correction(pushed_armors_[0].yaw - car_->prev_main_observing_yaw);
        if (delta_yaw < -car_->SWITCH_ARMOR_ANGLE) {
            // 逆时针转（角速度大于0）时切换装甲板
            car_->main_observing_armor_id += 1;
            car_->main_observing_armor_id %= ARMORS_COUNT;
            delta_yaw += M_PI * 2 / ARMORS_COUNT;
            kf_main_observing_armor_->force_change_value(pushed_armors_[0].translation);
        } else if (delta_yaw > car_->SWITCH_ARMOR_ANGLE) {
            // 顺时针转（角速度小于0）时切换装甲板
            car_->main_observing_armor_id += (ARMORS_COUNT - 1);
            car_->main_observing_armor_id %= ARMORS_COUNT;
            delta_yaw -= M_PI * 2 / ARMORS_COUNT;
            kf_main_observing_armor_->force_change_value(pushed_armors_[0].translation);
        } else {
            kf_main_observing_armor_->update(pushed_armors_[0].translation);
        }
        car_->accumulated_yaw += delta_yaw;
        car_->kf_yaw->update(Scalarf(car_->accumulated_yaw));
        if (pushed_armors_.size() == 1) { // 只有一块装甲板的时候只能算旋转轴和中心
            Vector3f car_axis = calc_rotation_axis(pushed_armors_[0]);
            car_->axis->update(car_axis);
            car_->axis->force_change_value(car_->axis->value().normalized());
            Vector3f car_center = calc_car_center(
                pushed_armors_[0],
                car_->radius[car_->main_observing_armor_id]->value().value(),
                car_->axis->value(),
                car_->height[car_->main_observing_armor_id]->value().value()
            );
            car_->kf_center->update(car_center);
        } else if (pushed_armors_.size() == 2) { // 两块装甲板的时候可以算旋转轴、半径、中心、相对高度
            // armors[1]对应的装甲板编号
            unsigned another_armor_id = calc_another_armor_id(
                pushed_armors_[0],
                pushed_armors_[1],
                car_->main_observing_armor_id,
                ARMORS_COUNT
            );
            // 更新半径
            std::tuple<float, float> radius =
                calc_radius(pushed_armors_[0], pushed_armors_[1]);
            car_->radius[car_->main_observing_armor_id]->update(Scalarf(std::get<0>(radius)));
            car_->radius[another_armor_id]->update(Scalarf(std::get<1>(radius)));
            // 更新车转轴方向向量
            Vector3f car_axis =
                calc_rotation_axis(pushed_armors_[0], pushed_armors_[1]);
            car_->axis->update(car_axis);
            car_->axis->force_change_value(car_->axis->value().normalized());
            // 更新装甲板相对高度
            float relative_height = calc_relative_height(pushed_armors_[0], pushed_armors_[1]);
            // 所有装甲板高度都相对0号装甲板，0号装甲板高度始终为0
            // 当相邻两块都不是0号装甲板时，只能基于其中一个更新下一个，造成误差累积
            // 为了尽可能避免误差累积需要让更新距离最短，按照0号和2号（对位）分类讨论
            if (car_->main_observing_armor_id == 0) { // 这个是0号，用这个更新另一个
                car_->height[another_armor_id]->update(Scalarf(relative_height));
            } else if (another_armor_id == 0) { // 另一个是0号，用另一个更新这个
                car_->height[car_->main_observing_armor_id]->update(Scalarf(-relative_height));
            } else if (car_->main_observing_armor_id == 2) { // 这个是2号，则这个离0号更远，用另一个更新这个
                car_->height[car_->main_observing_armor_id]->update(
                    car_->height[another_armor_id]->value() - Scalarf(relative_height)
                );
            } else if (another_armor_id == 2) { // 另一个是2号，则另一个离0号更远，用这个更新另一个
                car_->height[another_armor_id]->update(
                    car_->height[car_->main_observing_armor_id]->value() + Scalarf(relative_height)
                );
            }
            // 更新车中心
            Vector3f car_center0 = calc_car_center(
                pushed_armors_[0],
                car_->radius[car_->main_observing_armor_id]->value().value(),
                car_->axis->value(),
                car_->height[car_->main_observing_armor_id]->value().value()
            );
            Vector3f car_center1 = calc_car_center(
                pushed_armors_[1],
                car_->radius[another_armor_id]->value().value(),
                car_->axis->value(),
                car_->height[another_armor_id]->value().value()
            );
            car_->kf_center->update((car_center0 + car_center1) / 2);
        }
        update_antispin_mode();
        car_->prev_main_observing_yaw = pushed_armors_[0].yaw;
    }
}

void CarTracker::update_antispin_mode() {
    const float palstance = car_->kf_yaw->derivative().value();
    if (is_antispin_mode_ && std::abs(palstance) < EXIT_ANTISPIN_PALSTANCE) {
        is_antispin_mode_ = false;
    } else if (!is_antispin_mode_ && std::abs(palstance) > ENTER_ANTISPIN_PALSTANCE) {
        is_antispin_mode_ = true;
    }
}

float CarTracker::predict_img_to_hit_time(
    const float bullet_speed,
    const float img_to_fire_time,
    const Vector3f fric_to_basis
) const {
    const Vector3f target_to_basis = is_antispin_mode_
        ? car_->kf_center->value()
        : kf_main_observing_armor_->value();
    const Vector3f target_speed = is_antispin_mode_
        ? car_->kf_center->derivative()
        : kf_main_observing_armor_->derivative();
    float fly_time = 0;
    for (int i = 0; i < 5; i++) {
        const Vector3f pred_target_to_basis = target_to_basis + target_speed * (img_to_fire_time + fly_time);
        const Vector3f pred_target_to_fric = pred_target_to_basis - fric_to_basis;
        std::tie(std::ignore, fly_time) = trajectory::get_pitch_air_frac(
            std::hypot(pred_target_to_fric.x(), pred_target_to_fric.y()),
            pred_target_to_fric.z(),
            bullet_speed
        );
    }
    return img_to_fire_time + fly_time;
}

std::tuple<Vector3f, bool> CarTracker::predict_shoot_pos(
    const float gimbal_yaw_to_basis,
    const float img_to_hit_time
) const {
    if (!is_antispin_mode_) {
        return std::make_tuple(
            kf_main_observing_armor_->value() + kf_main_observing_armor_->derivative() * img_to_hit_time,
            status_->status() == TrackerStatus::TRACKING || status_->status() == TrackerStatus::TEMP_LOST
        );
    } else {
        const Vector3f pred_car_center =
            car_->kf_center->value() + car_->kf_center->derivative() * img_to_hit_time;
        // 0号装甲板在basis系下的预测yaw角
        const float pred_0_yaw_to_basis =
            car_->kf_yaw->value().value() + car_->kf_yaw->derivative().value() * img_to_hit_time;
        // 0号装甲板在gimbal系下的预测yaw角
        const float pred_0_yaw_to_gimbal =
            utils::rad_period_correction(pred_0_yaw_to_basis - gimbal_yaw_to_basis);
        // 最面向我们的装甲板在gimbal系下的预测角
        float target_angle_to_gimbal = M_PI * 2 / ARMORS_COUNT; 
        unsigned target_armor_id = 0;
        // 选择在img_to_hit_time之后，角度最小（即最面向我们）的那个装甲板
        for (unsigned i = 0; i < ARMORS_COUNT; i++) {
            const float pred_angle_to_gimbal =
                utils::rad_period_correction(pred_0_yaw_to_gimbal - 2 * M_PI * i / ARMORS_COUNT);
            if (abs(pred_angle_to_gimbal) < abs(target_angle_to_gimbal)) {
                target_angle_to_gimbal = pred_angle_to_gimbal;
                target_armor_id = i;
            }
        }
        if (abs(target_angle_to_gimbal) < ANTISPIN_FOLLOW_ANGLE) { // 跟随射击
            // 最面向我们的装甲板在basis下的预测yaw角
            const float target_angle_to_basis =
                utils::rad_period_correction(target_angle_to_gimbal + gimbal_yaw_to_basis);
            const Vector3f target = calc_armor_center(
                pred_car_center,
                car_->axis->value(),
                car_->height[target_armor_id]->value().value(),
                car_->radius[target_armor_id]->value().value(),
                target_angle_to_basis
            );
            const bool shoot_flag = std::abs(target_angle_to_gimbal) < ANTISPIN_SHOOT_ANGLE;
            return std::make_tuple(target, shoot_flag);
        } else { // 去下一块装甲板出现位置准备射击
            const float next_target_angle_to_basis = utils::rad_period_correction(
                (car_->kf_yaw->value().value() > 0 ? -1 : 1) * ANTISPIN_FOLLOW_ANGLE
                + gimbal_yaw_to_basis
            );
            // 这里要直接用target_armor_id而不是下一块的id取radius和height
            // 因为下一块装甲板即将进入击打范围时target_armor_id就是目标
            const Vector3f target = calc_armor_center(
                pred_car_center,
                car_->axis->value(),
                car_->height[target_armor_id]->value().value(),
                car_->radius[target_armor_id]->value().value(),
                next_target_angle_to_basis
            );
            return std::make_tuple(target, false);
        }
    }
}

void CarTracker::print_colored_status_info() const {
    const auto print_vec = [](const char* format, Vector3f vec) {
        std::printf(format, vec.x(), vec.y(), vec.z());
    };
    switch(status_->status()) {
        case TrackerStatus::TRACKING: {
            std::cout << termcolor::green << termcolor::bold << "TRACKING    " << termcolor::reset;
            break;
        }
        case TrackerStatus::CONVERGING: {
            std::cout << termcolor::white << termcolor::bold << "CONVERGING  " << termcolor::reset;
            break;
        }
        case TrackerStatus::TEMP_LOST: {
            std::cout << termcolor::red << termcolor::bold << "TEMP_LOST   " << termcolor::reset;
            break;
        }
        case TrackerStatus::LOST: {
            std::cout << termcolor::red << termcolor::bold << "LOST        " << termcolor::reset;
            break;
        }
    }
    std::cout << status_->status_frames() << std::endl;
    std::cout << termcolor::bold << "MainArmor   " << termcolor::reset;
    print_vec("[% 4.0f, % 4.0f, % 4.0f] += ", kf_main_observing_armor_->value() * 100);
    print_vec("[% 4.0f, % 4.0f, % 4.0f]\n", kf_main_observing_armor_->derivative() * 100);
    std::cout << termcolor::bold << "Car.ArmorID " << termcolor::reset;
    std::printf("%u\n", car_->main_observing_armor_id);
    std::cout << termcolor::bold << "Car.Center  " << termcolor::reset;
    print_vec("[% 4.0f, % 4.0f, % 4.0f] += ", car_->kf_center->value() * 100);
    print_vec("[% 4.0f, % 4.0f, % 4.0f]\n", car_->kf_center->derivative() * 100);
    std::cout << termcolor::bold << "Car.Yaw     " << termcolor::reset;
    std::printf(
        "[% 6.0f] += [% 4.0f]\n",
        utils::r2d(car_->kf_yaw->value().value()), utils::r2d(car_->kf_yaw->derivative().value())
    );
    std::cout << termcolor::bold << "Car.Axis    " << termcolor::reset;
    print_vec("[% 4.3f, % 4.3f, % 4.3f]", car_->axis->value());
    std::printf(" (φ: % 3.0f)\n", utils::r2d(utils::get_angle(
        car_->axis->value().cast<double>(),
        Vector3d::UnitZ()
    )));
    std::cout << termcolor::bold << "Car.Radius  " << termcolor::reset;
    std::printf(
        "[% 2.0f, % 2.0f, % 2.0f, % 2.0f]\n",
        car_->radius[0]->value().value() * 100, car_->radius[1]->value().value() * 100,
        car_->radius[2]->value().value() * 100, car_->radius[3]->value().value() * 100
    );
    std::cout << termcolor::bold << "Car.Height  " << termcolor::reset;
    std::printf(
        "[% 2.0f, % 2.0f, % 2.0f, % 2.0f]",
        car_->height[0]->value().value() * 100, car_->height[1]->value().value() * 100,
        car_->height[2]->value().value() * 100, car_->height[3]->value().value() * 100
    );
    std::cout << std::endl;
}

std::vector<std::tuple<Eigen::Vector3f, Eigen::Quaternionf>> CarTracker::get_all_armors() const {
    std::vector<std::tuple<Eigen::Vector3f, Eigen::Quaternionf>> armors;
    for (unsigned i = 0; i < ARMORS_COUNT; i++) {
        const float armor_yaw =  utils::rad_period_correction(
            car_->kf_yaw->value().value() - 2 * M_PI * i / ARMORS_COUNT
        );
        const std::tuple<Eigen::Vector3f, Eigen::Quaternionf> armor_pose = calc_armor_pose(
            car_->kf_center->value(),
            car_->axis->value(),
            car_->height[i]->value().value(),
            car_->radius[i]->value().value(),
            armor_yaw
        );
        armors.emplace_back(armor_pose);
    }
    return armors;
}