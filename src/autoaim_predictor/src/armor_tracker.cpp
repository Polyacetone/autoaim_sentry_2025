#include <autoaim_predictor/armor_tracker.hpp>

using namespace Eigen;
using Scalarf = Vector<float, 1>;

/**********************************************************************************
**********************************    Utils    ************************************
***********************************************************************************/

std::tuple<float, float> calc_radius(const Armor& armor1, const Armor& armor2) {
    const Vector3f p1p2 = armor2.translation - armor1.translation;
    const float radius1_1 = std::abs(p1p2.dot(armor1.rotated_x.normalized()));
    const float radius1_2 = std::abs(p1p2.dot(armor2.rotated_y.normalized()));
    const float radius2_1 = std::abs(p1p2.dot(armor1.rotated_y.normalized()));
    const float radius2_2 = std::abs(p1p2.dot(armor2.rotated_x.normalized()));
    const float radius1 = (radius1_1 + radius1_2) / 2;
    const float radius2 = (radius2_1 + radius2_2) / 2;
    return {radius1, radius2};
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
    Matrix3f rmat;
    rmat.col(0) = armor_normal;
    rmat.col(1) = axis.cross(armor_normal).normalized();
    rmat.col(2) = axis.normalized();
    return {armor_center, Quaternionf(rmat)};
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

float calc_img_to_hit_time(
    const float bullet_speed,
    const float img_to_fire_time,
    const Vector3f& target_to_fake_fric,
    const Vector3f& target_speed
) {
    float fly_time = 0;
    for (int i = 0; i < 5; i++) {
        const Vector3f pred_target_to_fake_fric =
            target_to_fake_fric + target_speed * (img_to_fire_time + fly_time);
        fly_time = std::get<1>(trajectory::get_pitch_air_frac(
            std::hypot(pred_target_to_fake_fric.x(), pred_target_to_fake_fric.y()),
            pred_target_to_fake_fric.z(),
            bullet_speed
        ));
    }
    return img_to_fire_time + fly_time;
}

/**********************************************************************************
**********************************    Armor    ************************************
***********************************************************************************/

Armor::Armor(const tf2::Transform& armor_pose, const float pitch_to_basis): Armor(
    utils::convert_to<Vector3f>(armor_pose.getOrigin()),
    utils::convert_to<Quaternionf>(armor_pose.getRotation()),
    pitch_to_basis
) {}

Armor::Armor(const Vector3f& translation, const Quaternionf& rotation, const float pitch_to_basis):
    translation(translation) {
    Vector3f original_x = rotation * Vector3f::UnitX();
    Vector3f original_y = rotation * Vector3f::UnitY();
    Vector3f original_z = rotation * Vector3f::UnitZ();
    AngleAxisf pitch_correction(-pitch_to_basis, original_y.normalized());
    rotated_x = pitch_correction * original_x;
    rotated_y = pitch_correction * original_y;
    rotated_z = pitch_correction * original_z;
    Vector2f rotated_x_on_xy = rotated_x.head<2>();
    rotated_x_on_xy.normalize();
    yaw = std::atan2(rotated_x_on_xy.y(), rotated_x_on_xy.x());
}

/**********************************************************************************
*********************************  CarObserver  ***********************************
***********************************************************************************/

CarObserver::CarObserver(const cv::FileNode& fn) {
    INITIAL_RADIUS = static_cast<float>(fn["initial_radius"]);
    SWITCH_ARMOR_ANGLE = static_cast<float>(fn["switch_armor_angle"]);
    DELTA_YAW_UPDATE_THRESHOLD = static_cast<float>(fn["delta_yaw_update_threshold"]);
    ENTER_ANTISPIN_PALSTANCE = static_cast<float>(fn["enter_antispin_palstance"]);
    EXIT_ANTISPIN_PALSTANCE = static_cast<float>(fn["exit_antispin_palstance"]);
    ANTISPIN_SHOOT_ANGLE = static_cast<float>(fn["antispin_shoot_angle"]);
    ANTISPIN_OUT_OF_SHOOT_ANGLE_THRESHOLD = static_cast<int>(fn["antispin_out_of_shoot_angle_threshold"]);
    ANTISPIN_IN_SHOOT_ANGLE_THRESHOLD = static_cast<int>(fn["antispin_in_shoot_angle_threshold"]);
    float radius_filter_ratio = static_cast<float>(fn["radius_filter_ratio"]);
    float height_filter_ratio = static_cast<float>(fn["height_filter_ratio"]);
    float axis_filter_ratio = static_cast<float>(fn["axis_filter_ratio"]);
    axis_ = std::make_unique<EMAF<3>>(axis_filter_ratio);
    for (int i = 0; i < 4; i++) {
        radius_[i] = std::make_unique<EMAF<1>>(radius_filter_ratio);
        height_[i] = std::make_unique<EMAF<1>>(height_filter_ratio);
    }
    kf_center_ = std::make_unique<KF<3>>(fn["kf_center"]);
    kf_yaw_ = std::make_unique<KF<1>>(fn["kf_yaw"]);
    reset();
}

void CarObserver::reset() {
    is_armor_switched_ = false;
    is_antispin_palstance_ = false;
    main_observing_armor_id_ = 0;
    accumulated_yaw_ = 0;
    prev_main_observing_yaw_ = 0;
    out_of_shoot_angle_count_ = 0;
    for (int i = 0; i < 4; i++) {
        height_[i]->reset();
        radius_[i]->reset();
    }
    axis_->reset();
    kf_center_->reset();
    kf_yaw_->reset();
}

void CarObserver::initialize(const std::vector<Armor>& armors) {
    reset();
    accumulated_yaw_ = armors[0].yaw;
    prev_main_observing_yaw_ = accumulated_yaw_;
    kf_yaw_->initialize(Scalarf(accumulated_yaw_));
    for (int i = 0; i < 4; i++) {
        radius_[i]->initialize(Scalarf(INITIAL_RADIUS));
    }
    if (armors.size() == 1) { // 只有一块装甲板的时候只能算旋转轴和中心
        Vector3f car_axis = calc_rotation_axis(armors[0]);
        axis_->initialize(car_axis);
        Vector3f car_center = calc_car_center(
            armors[0],
            radius_[0]->value().value(),
            car_axis,
            height_[0]->value().value()
        );
        kf_center_->initialize(car_center);
    } else if (armors.size() == 2) { // 两块装甲板的时候可以算旋转轴、半径、中心、相对高度
        // armors[1]对应的装甲板编号
        unsigned another_armor_id = calc_another_armor_id(
            armors[0],
            armors[1],
            0, ARMORS_COUNT
        );
        Vector3f car_axis =
            calc_rotation_axis(armors[0], armors[1]);
        axis_->initialize(car_axis);
        std::tuple<float, float> radius_measurements =
            calc_radius(armors[0], armors[1]);
        radius_[0]->update(Scalarf(std::get<0>(radius_measurements)));
        radius_[another_armor_id]->update(Scalarf(std::get<1>(radius_measurements)));
        float another_armor_height =
            calc_relative_height(armors[0], armors[1]);
        height_[another_armor_id]->update(Scalarf(another_armor_height));
        Vector3f car_center0 = calc_car_center(
            armors[0],
            radius_[0]->value().value(),
            car_axis,
            0
        );
        Vector3f car_center1 = calc_car_center(
            armors[1],
            radius_[another_armor_id]->value().value(),
            car_axis,
            another_armor_height
        );
        kf_center_->initialize((car_center0 + car_center1) / 2);
    }
}

void CarObserver::predict(const float time_elapsed) const {
    kf_center_->predict(time_elapsed);
    kf_yaw_->predict(time_elapsed);
}

void CarObserver::update(const std::vector<Armor>& armors) {
    float delta_yaw = utils::rad_period_correction(armors[0].yaw - prev_main_observing_yaw_);
    if (delta_yaw < -SWITCH_ARMOR_ANGLE) {
        // 逆时针转（角速度大于0）时切换装甲板
        main_observing_armor_id_ += 1;
        main_observing_armor_id_ %= ARMORS_COUNT;
        delta_yaw += M_PI * 2 / ARMORS_COUNT;
        is_armor_switched_ = true;
    } else if (delta_yaw > SWITCH_ARMOR_ANGLE) {
        // 顺时针转（角速度小于0）时切换装甲板
        main_observing_armor_id_ += (ARMORS_COUNT - 1);
        main_observing_armor_id_ %= ARMORS_COUNT;
        delta_yaw -= M_PI * 2 / ARMORS_COUNT;
        is_armor_switched_ = true;
    } else {
        is_armor_switched_ = false;
    }
    accumulated_yaw_ += delta_yaw;
    if (std::abs(delta_yaw) < DELTA_YAW_UPDATE_THRESHOLD) {
        kf_yaw_->update(Scalarf(accumulated_yaw_));
    }
    if (armors.size() == 1) { // 只有一块装甲板的时候只能算旋转轴和中心
        Vector3f car_axis = calc_rotation_axis(armors[0]);
        axis_->update(car_axis);
        axis_->force_change_value(axis_->value().normalized());
        Vector3f car_center = calc_car_center(
            armors[0],
            radius_[main_observing_armor_id_]->value().value(),
            axis_->value(),
            height_[main_observing_armor_id_]->value().value()
        );
        kf_center_->update(car_center);
    } else if (armors.size() == 2) { // 两块装甲板的时候可以算旋转轴、半径、中心、相对高度
        // armors[1]对应的装甲板编号
        unsigned another_armor_id = calc_another_armor_id(
            armors[0],
            armors[1],
            main_observing_armor_id_,
            ARMORS_COUNT
        );
        // 更新半径
        std::tuple<float, float> radius_measurements =
            calc_radius(armors[0], armors[1]);
        radius_[main_observing_armor_id_]->update(Scalarf(std::get<0>(radius_measurements)));
        radius_[another_armor_id]->update(Scalarf(std::get<1>(radius_measurements)));
        // 更新车转轴方向向量
        Vector3f car_axis =
            calc_rotation_axis(armors[0], armors[1]);
        axis_->update(car_axis);
        axis_->force_change_value(axis_->value().normalized());
        // 更新装甲板相对高度
        float relative_height = calc_relative_height(armors[0], armors[1]);
        // 所有装甲板高度都相对0号装甲板，0号装甲板高度始终为0
        // 当相邻两块都不是0号装甲板时，只能基于其中一个更新下一个，造成误差累积
        // 为了尽可能避免误差累积需要让更新距离最短，按照0号和2号（对位）分类讨论
        if (main_observing_armor_id_ == 0) { // 这个是0号，用这个更新另一个
            height_[another_armor_id]->update(Scalarf(relative_height));
        } else if (another_armor_id == 0) { // 另一个是0号，用另一个更新这个
            height_[main_observing_armor_id_]->update(Scalarf(-relative_height));
        } else if (main_observing_armor_id_ == 2) { // 这个是2号，则这个离0号更远，用另一个更新这个
            height_[main_observing_armor_id_]->update(
                height_[another_armor_id]->value() - Scalarf(relative_height)
            );
        } else if (another_armor_id == 2) { // 另一个是2号，则另一个离0号更远，用这个更新另一个
            height_[another_armor_id]->update(
                height_[main_observing_armor_id_]->value() + Scalarf(relative_height)
            );
        }
        // 更新车中心
        Vector3f car_center0 = calc_car_center(
            armors[0],
            radius_[main_observing_armor_id_]->value().value(),
            axis_->value(),
            height_[main_observing_armor_id_]->value().value()
        );
        Vector3f car_center1 = calc_car_center(
            armors[1],
            radius_[another_armor_id]->value().value(),
            axis_->value(),
            height_[another_armor_id]->value().value()
        );
        kf_center_->update((car_center0 + car_center1) / 2);
    }
    const float palstance = kf_yaw_->derivative().value();
    if (is_antispin_palstance_ && std::abs(palstance) < EXIT_ANTISPIN_PALSTANCE) {
        is_antispin_palstance_ = false;
    } else if (!is_antispin_palstance_ && std::abs(palstance) > ENTER_ANTISPIN_PALSTANCE) {
        is_antispin_palstance_ = true;
    }
    prev_main_observing_yaw_ = armors[0].yaw;
}

float CarObserver::predict_img_to_hit_time(
    const float bullet_speed,
    const float img_to_fire_time,
    const Vector3f fric_to_basis
) const {
    return calc_img_to_hit_time(
        bullet_speed,
        img_to_fire_time,
        kf_center_->value() - fric_to_basis,
        kf_center_->derivative()
    );
}

std::tuple<Vector3f, bool> CarObserver::predict_shoot_pos(
    const float gimbal_yaw_to_basis,
    const float img_to_hit_time
) {
    const Vector3f pred_car_center =
        kf_center_->value() + kf_center_->derivative() * img_to_hit_time;
    // 0号装甲板在basis系下的预测yaw角
    const float pred_0_yaw_to_basis =
        kf_yaw_->value().value() + kf_yaw_->derivative().value() * img_to_hit_time;
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
    if (abs(target_angle_to_gimbal) > ANTISPIN_SHOOT_ANGLE) {
        out_of_shoot_angle_count_++;
        in_shoot_angle_count_ = 0;
    } else {
        out_of_shoot_angle_count_ = 0;
        in_shoot_angle_count_++;
    }
    if (out_of_shoot_angle_count_ < ANTISPIN_OUT_OF_SHOOT_ANGLE_THRESHOLD
        && in_shoot_angle_count_ > ANTISPIN_IN_SHOOT_ANGLE_THRESHOLD) { // 跟随射击
        // 最面向我们的装甲板在basis下的预测yaw角
        const float target_angle_to_basis =
            utils::rad_period_correction(target_angle_to_gimbal + gimbal_yaw_to_basis);
        const Vector3f target = calc_armor_center(
            pred_car_center,
            axis_->value(),
            height_[target_armor_id]->value().value(),
            radius_[target_armor_id]->value().value(),
            target_angle_to_basis
        );
        return {target, true};
    } else { // 去下一块装甲板出现位置准备射击
        const float next_target_angle_to_basis = utils::rad_period_correction(
            (kf_yaw_->value().value() > 0 ? -1 : 1) * ANTISPIN_SHOOT_ANGLE
            + gimbal_yaw_to_basis
        );
        // 这里要直接用target_armor_id而不是下一块的id取radius和height
        // 因为下一块装甲板即将进入击打范围时target_armor_id就是目标
        const Vector3f target = calc_armor_center(
            pred_car_center,
            axis_->value(),
            height_[target_armor_id]->value().value(),
            radius_[target_armor_id]->value().value(),
            next_target_angle_to_basis
        );
        return {target, false};
    }
}

void CarObserver::print_colored_status_info() const {
    const auto print_vec = [](const char* format, Vector3f vec) {
        std::printf(format, vec.x(), vec.y(), vec.z());
    };
    std::cout << termcolor::bold << "Car.ArmorID " << termcolor::reset;
    std::printf("%u\n", main_observing_armor_id_);
    std::cout << termcolor::bold << "Car.Center  " << termcolor::reset;
    print_vec("[% 4.0f, % 4.0f, % 4.0f] += ", kf_center_->value() * 100);
    print_vec("[% 4.0f, % 4.0f, % 4.0f]\n", kf_center_->derivative() * 100);
    std::cout << termcolor::bold << "Car.Yaw     " << termcolor::reset;
    std::printf(
        "[% 6.0f] += [% 4.0f]\n",
        utils::r2d(kf_yaw_->value().value()), utils::r2d(kf_yaw_->derivative().value())
    );
    std::cout << termcolor::bold << "Car.Axis    " << termcolor::reset;
    print_vec("[% 4.3f, % 4.3f, % 4.3f]", axis_->value());
    std::printf(" (φ: % 3.0f)\n", utils::r2d(utils::get_angle(
        axis_->value().cast<double>(),
        Vector3d::UnitZ()
    )));
    std::cout << termcolor::bold << "Car.Radius  " << termcolor::reset;
    std::printf(
        "[% 2.0f, % 2.0f, % 2.0f, % 2.0f]\n",
        radius_[0]->value().value() * 100, radius_[1]->value().value() * 100,
        radius_[2]->value().value() * 100, radius_[3]->value().value() * 100
    );
    std::cout << termcolor::bold << "Car.Height  " << termcolor::reset;
    std::printf(
        "[% 2.0f, % 2.0f, % 2.0f, % 2.0f]",
        height_[0]->value().value() * 100, height_[1]->value().value() * 100,
        height_[2]->value().value() * 100, height_[3]->value().value() * 100
    );
    std::cout << std::endl;
}

std::vector<std::tuple<Vector3f, Quaternionf>> CarObserver::get_all_armors() const {
    std::vector<std::tuple<Vector3f, Quaternionf>> armors;
    for (unsigned i = 0; i < ARMORS_COUNT; i++) {
        const float armor_yaw =  utils::rad_period_correction(
            kf_yaw_->value().value() - 2 * M_PI * i / ARMORS_COUNT
        );
        const std::tuple<Vector3f, Quaternionf> armor_pose = calc_armor_pose(
            kf_center_->value(),
            axis_->value(),
            height_[i]->value().value(),
            radius_[i]->value().value(),
            armor_yaw
        );
        armors.emplace_back(armor_pose);
    }
    return armors;
}

void CarObserver::write_predictor_status(hw_sentry_interfaces::msg::PredictorStatus& status) const {
    for (unsigned i = 0; i < ARMORS_COUNT; i++) {
        status.radius.emplace_back(radius_[i]->value().value());
        status.height.emplace_back(height_[i]->value().value());
    }
    status.axis = utils::convert_to<geometry_msgs::msg::Point32>(axis_->value());
    status.center = utils::convert_to<geometry_msgs::msg::Point32>(kf_center_->value());
    status.velocity = utils::convert_to<geometry_msgs::msg::Point32>(kf_center_->derivative());
    status.yaw = kf_yaw_->value().value();
    status.palstance = kf_yaw_->derivative().value();
}

/**********************************************************************************
*******************************  OutpostObserver  *********************************
***********************************************************************************/

OutpostObserver::OutpostObserver(const cv::FileNode& fn) {
    RADIUS = static_cast<float>(fn["radius"]);
    SWITCH_ARMOR_ANGLE = static_cast<float>(fn["switch_armor_angle"]);
    DELTA_YAW_UPDATE_THRESHOLD = static_cast<float>(fn["delta_yaw_update_threshold"]);
    OUTPOST_SHOOT_ANGLE = static_cast<float>(fn["outpost_shoot_angle"]);
    OUTPOST_OUT_OF_SHOOT_ANGLE_THRESHOLD = static_cast<int>(fn["outpost_out_of_shoot_angle_threshold"]);
    OUTPOST_IN_SHOOT_ANGLE_THRESHOLD = static_cast<int>(fn["outpost_in_shoot_angle_threshold"]);
    kf_center_ = std::make_unique<KF<3>>(fn["kf_center"]);
    kf_yaw_ = std::make_unique<KF<1>>(fn["kf_yaw"]);
    reset();
}

void OutpostObserver::reset() {
    is_armor_switched_ = false;
    accumulated_yaw_ = 0;
    prev_main_observing_yaw_ = 0;
    kf_center_->reset();
    kf_yaw_->reset();
}

void OutpostObserver::initialize(const std::vector<Armor>& armors) {
    reset();
    accumulated_yaw_ = armors[0].yaw;
    prev_main_observing_yaw_ = accumulated_yaw_;
    kf_yaw_->initialize(Scalarf(accumulated_yaw_));
    Vector3f car_center = calc_car_center(armors[0], RADIUS, Vector3f::UnitZ(), 0);
    kf_center_->initialize(car_center);
}

void OutpostObserver::predict(const float time_elapsed) const {
    kf_center_->predict(time_elapsed);
    kf_yaw_->predict(time_elapsed);
}

void OutpostObserver::update(const std::vector<Armor>& armors) {
    float delta_yaw = utils::rad_period_correction(armors[0].yaw - prev_main_observing_yaw_);
    if (delta_yaw < -SWITCH_ARMOR_ANGLE) {
        // 逆时针转（角速度大于0）时切换装甲板
        delta_yaw += M_PI * 2 / ARMORS_COUNT;
        is_armor_switched_ = true;
    } else if (delta_yaw > SWITCH_ARMOR_ANGLE) {
        // 顺时针转（角速度小于0）时切换装甲板
        delta_yaw -= M_PI * 2 / ARMORS_COUNT;
        is_armor_switched_ = true;
    } else {
        is_armor_switched_ = false;
    }
    accumulated_yaw_ += delta_yaw;
    if (std::abs(delta_yaw) < DELTA_YAW_UPDATE_THRESHOLD) {
        kf_yaw_->update(Scalarf(accumulated_yaw_));
    }
    Vector3f car_center = calc_car_center(armors[0], RADIUS, Vector3f::UnitZ(), 0);
    kf_center_->update(car_center);
    prev_main_observing_yaw_ = armors[0].yaw;
}

float OutpostObserver::predict_img_to_hit_time(
    const float bullet_speed,
    const float img_to_fire_time,
    const Vector3f fric_to_basis
) const {
    return calc_img_to_hit_time(
        bullet_speed,
        img_to_fire_time,
        kf_center_->value() - fric_to_basis,
        kf_center_->derivative()
    );
}

std::tuple<Vector3f, bool> OutpostObserver::predict_shoot_pos(
    const float gimbal_yaw_to_basis,
    const float img_to_hit_time
) {
    const Vector3f pred_car_center =
        kf_center_->value() + kf_center_->derivative() * img_to_hit_time;
    // 0号装甲板在basis系下的预测yaw角
    const float pred_0_yaw_to_basis =
        kf_yaw_->value().value() + kf_yaw_->derivative().value() * img_to_hit_time;
    // 0号装甲板在gimbal系下的预测yaw角
    const float pred_0_yaw_to_gimbal =
        utils::rad_period_correction(pred_0_yaw_to_basis - gimbal_yaw_to_basis);
    // 最面向我们的装甲板在gimbal系下的预测角
    float target_angle_to_gimbal = M_PI * 2 / ARMORS_COUNT; 
    // 选择在img_to_hit_time之后，角度最小（即最面向我们）的那个装甲板
    for (unsigned i = 0; i < ARMORS_COUNT; i++) {
        const float pred_angle_to_gimbal =
            utils::rad_period_correction(pred_0_yaw_to_gimbal - 2 * M_PI * i / ARMORS_COUNT);
        if (abs(pred_angle_to_gimbal) < abs(target_angle_to_gimbal)) {
            target_angle_to_gimbal = pred_angle_to_gimbal;
        }
    }
    if (abs(target_angle_to_gimbal) > OUTPOST_SHOOT_ANGLE) {
        out_of_shoot_angle_count_++;
        in_shoot_angle_count_ = 0;
    } else {
        out_of_shoot_angle_count_ = 0;
        in_shoot_angle_count_++;
    }
    if (out_of_shoot_angle_count_ < OUTPOST_OUT_OF_SHOOT_ANGLE_THRESHOLD
        && in_shoot_angle_count_ > OUTPOST_IN_SHOOT_ANGLE_THRESHOLD) { // 跟随射击
        // 最面向我们的装甲板在basis下的预测yaw角
        const float target_angle_to_basis =
            utils::rad_period_correction(target_angle_to_gimbal + gimbal_yaw_to_basis);
        const Vector3f target = calc_armor_center(
            pred_car_center,
            Vector3f::UnitZ(),
            0,
            RADIUS,
            target_angle_to_basis
        );
        return {target, true};
    } else { // 去下一块装甲板出现位置准备射击
        const float next_target_angle_to_basis = utils::rad_period_correction(
            (kf_yaw_->value().value() > 0 ? -1 : 1) * OUTPOST_SHOOT_ANGLE
            + gimbal_yaw_to_basis
        );
        // 这里要直接用target_armor_id而不是下一块的id取radius和height
        // 因为下一块装甲板即将进入击打范围时target_armor_id就是目标
        const Vector3f target = calc_armor_center(
            pred_car_center,
            Vector3f::UnitZ(),
            0,
            RADIUS,
            next_target_angle_to_basis
        );
        return {target, false};
    }
}

void OutpostObserver::print_colored_status_info() const {
    const auto print_vec = [](const char* format, Vector3f vec) {
        std::printf(format, vec.x(), vec.y(), vec.z());
    };
    std::cout << termcolor::bold << "Outpost.Center  " << termcolor::reset;
    print_vec("[% 4.0f, % 4.0f, % 4.0f] += ", kf_center_->value() * 100);
    print_vec("[% 4.0f, % 4.0f, % 4.0f]\n", kf_center_->derivative() * 100);
    std::cout << termcolor::bold << "Outpost.Yaw     " << termcolor::reset;
    std::printf(
        "[% 6.0f] += [% 4.0f]",
        utils::r2d(kf_yaw_->value().value()), utils::r2d(kf_yaw_->derivative().value())
    );
    std::cout << std::endl;
}

std::vector<std::tuple<Vector3f, Quaternionf>> OutpostObserver::get_all_armors() const {
    std::vector<std::tuple<Vector3f, Quaternionf>> armors;
    for (unsigned i = 0; i < ARMORS_COUNT; i++) {
        const float armor_yaw =  utils::rad_period_correction(
            kf_yaw_->value().value() - 2 * M_PI * i / ARMORS_COUNT
        );
        const std::tuple<Vector3f, Quaternionf> armor_pose = calc_armor_pose(
            kf_center_->value(),
            Vector3f::UnitZ(),
            0,
            RADIUS,
            armor_yaw
        );
        armors.emplace_back(armor_pose);
    }
    return armors;
}

void OutpostObserver::write_predictor_status(hw_sentry_interfaces::msg::PredictorStatus& status) const {
    for (unsigned i = 0; i < ARMORS_COUNT; i++) {
        status.radius.emplace_back(RADIUS);
        status.height.emplace_back(0);
    }
    status.axis = utils::convert_to<geometry_msgs::msg::Point32>(Vector3f(0, 0, 1));
    status.center = utils::convert_to<geometry_msgs::msg::Point32>(kf_center_->value());
    status.velocity = utils::convert_to<geometry_msgs::msg::Point32>(kf_center_->derivative());
    status.yaw = kf_yaw_->value().value();
    status.palstance = kf_yaw_->derivative().value();
}

/**********************************************************************************
*********************************  ArmorTracker  **********************************
***********************************************************************************/

ArmorTracker::ArmorTracker(const cv::FileNode& fn) {
    ERR_QUEUE_SIZE = static_cast<int>(fn["err_queue_size"]);
    APPROXIMATE_FRAMERATE = static_cast<int>(fn["approximate_framerate"]);
    LOW_ACCURACY_ERR_THRESHOLD = static_cast<float>(fn["low_accuracy_err_threshold"]);
    kf_main_observing_armor_ = std::make_unique<KF<3>>(fn["kf_main_observing_armor"]);
    car_status_ = std::make_unique<TrackerStatus>(
        fn["car_status"],
        [this](StatusType from, StatusType to) { status_change_handler(from, to); },
        [this](StatusType curr) { status_remain_handler(curr); }
    );
    outpost_status_ = std::make_unique<TrackerStatus>(
        fn["outpost_status"],
        [this](StatusType from, StatusType to) { status_change_handler(from, to); },
        [this](StatusType curr) { status_remain_handler(curr); }
    );
    car_observer_ = std::make_unique<CarObserver>(fn["car_observer"]);
    outpost_observer_ = std::make_unique<OutpostObserver>(fn["outpost_observer"]);
    reset();
}

StatusType ArmorTracker::status() const {
    if (target_label_ == ArmorType::OUTPOST) {
        return outpost_status_->status();
    } else {
        return car_status_->status();
    }
}

void ArmorTracker::push(const Armor& armor) {
    pushed_armors_.emplace_back(armor);
}

void ArmorTracker::reset() {
    pushed_armors_.clear();
    kf_main_observing_armor_->reset();
    car_status_->reset();
    car_observer_->reset();
    outpost_status_->reset();
    outpost_status_->reset();
    kf_armor_pred_pos_history_.clear();
    car_pred_pos_history_.clear();
    kf_armor_err_que_.clear();
    car_err_que_.clear();
    kf_armor_avg_err_ = car_avg_err_ = 0;
}

void ArmorTracker::set_target_label(ArmorType label) {
    if (target_label_ != label) {
        reset();
        target_label_ = label;
    }
}

void ArmorTracker::update(const double timestamp) {
    current_update_time_ = timestamp;
    bool is_valid = (pushed_armors_.size() == 1 || pushed_armors_.size() == 2);
    // 更新状态机，状态机会根据状态调用跟踪器的更新
    if (target_label_ == ArmorType::OUTPOST) outpost_status_->update(is_valid);
    else car_status_->update(is_valid);
    // 统计单独装甲板预测以及整车预测的历史误差
    if (is_valid) update_pred_accuracy();
    prev_update_time_ = current_update_time_;
    pushed_armors_.clear();
}

void ArmorTracker::status_change_handler(StatusType from, StatusType to) {
    if (from == StatusType::LOST && to == StatusType::CONVERGING) { // 初始化
        kf_main_observing_armor_->initialize(pushed_armors_[0].translation);
        if (target_label_ == ArmorType::OUTPOST) {
            outpost_observer_->initialize(pushed_armors_);
        } else {
            car_observer_->initialize(pushed_armors_);
        }
    }
}

void ArmorTracker::status_remain_handler(StatusType current) {
    if (current != StatusType::LOST) { // 预测
        const float time_elapsed = static_cast<float>(current_update_time_ - prev_update_time_);
        kf_main_observing_armor_->predict(time_elapsed);
        if (target_label_ == ArmorType::OUTPOST) {
            outpost_observer_->predict(time_elapsed);
        } else {
            car_observer_->predict(time_elapsed);
        }
    }
    if (current == StatusType::CONVERGING || current == StatusType::TRACKING) { // 更新
        if (target_label_ == ArmorType::OUTPOST) {
            outpost_observer_->update(pushed_armors_);
        } else {
            car_observer_->update(pushed_armors_);
        }
        if (car_observer_->is_armor_switched_ || outpost_observer_->is_armor_switched_) {
            kf_main_observing_armor_->force_change_value(pushed_armors_[0].translation);
        } else {
            kf_main_observing_armor_->update(pushed_armors_[0].translation);
        }
    }
}

void ArmorTracker::update_kf_armor_pred_pos_history(
    const double time,
    const Vector3f& kf_pred_pos
) {
    kf_armor_pred_pos_history_.emplace_front(time, kf_pred_pos);
    while (kf_armor_pred_pos_history_.size() > 2 * APPROXIMATE_FRAMERATE) {
        kf_armor_pred_pos_history_.pop_back();
    }
}

void ArmorTracker::update_car_pred_pos_history(
    const double time,
    const Vector3f& car_pred_pos
) {
    car_pred_pos_history_.emplace_front(time, car_pred_pos);
    while (car_pred_pos_history_.size() > 2 * APPROXIMATE_FRAMERATE) {
        car_pred_pos_history_.pop_back();
    }
}

void ArmorTracker::update_pred_accuracy() {
    const auto calc_pitch = [](const Vector3f& v) {
        return std::atan2(v.z(), std::hypot(v.x(), v.y()));
    };
    const auto calc_yaw = [](const Vector3f& v) {
        return std::atan2(v.y(), v.x());
    };
    const auto update_err_que = [&](
        const std::deque<std::tuple<double, Eigen::Vector3f>>& history,
        std::deque<float>& err_que
    ) {
        auto iter = std::find_if(
            history.begin(),
            history.end(),
            [&](const auto& t) {
                return std::abs(std::get<0>(t) - current_update_time_) < 0.6 / APPROXIMATE_FRAMERATE;
            }
        );
        if (iter != history.end()) {
            auto pred_pos = std::get<1>(*iter);
            float pred_pitch = calc_pitch(pred_pos);
            float pred_yaw = calc_yaw(pred_pos);
            if (pushed_armors_.size() == 1) {
                float observation_pitch = calc_pitch(pushed_armors_[0].translation);
                float observation_yaw = calc_yaw(pushed_armors_[0].translation);
                float diff = std::hypot(
                    pred_pitch - observation_pitch,
                    pred_yaw - observation_yaw
                ) * pred_pos.norm();
                err_que.push_front(diff);
            } else if (pushed_armors_.size() == 2) {
                float observation_pitch0 = calc_pitch(pushed_armors_[0].translation);
                float observation_yaw0 = calc_yaw(pushed_armors_[0].translation);
                float observation_pitch1 = calc_pitch(pushed_armors_[1].translation);
                float observation_yaw1 = calc_yaw(pushed_armors_[1].translation);
                float diff0 = std::hypot(
                    pred_pitch - observation_pitch0,
                    pred_yaw - observation_yaw0
                ) * pred_pos.norm();
                float diff1 = std::hypot(
                    pred_pitch - observation_pitch1,
                    pred_yaw - observation_yaw1
                ) * pred_pos.norm();
                err_que.push_front(std::min(diff0, diff1));
            }
            while (err_que.size() > ERR_QUEUE_SIZE) err_que.pop_back();
        }
    };
    const auto calc_err_avg = [](const std::deque<float>& err_que) {
        float avg = 0;
        int size = err_que.size(), div = size * (size + 1) / 2;
        for (int i = 0; i < size; i++) {
            avg += err_que[i] * (size - i) / div;
        }
        return avg;
    };
    update_err_que(kf_armor_pred_pos_history_, kf_armor_err_que_);
    update_err_que(car_pred_pos_history_, car_err_que_);
    kf_armor_avg_err_ = calc_err_avg(kf_armor_err_que_);
    car_avg_err_ = calc_err_avg(car_err_que_);
}

std::tuple<Vector3f, bool> ArmorTracker::predict_shoot_pos(
    const float bullet_speed,
    const float img_to_fire_time,
    const Vector3f fric_to_basis,
    const float gimbal_yaw_to_basis
) {
    if (target_label_ == ArmorType::OUTPOST) {
        const float img_to_hit_time =
            outpost_observer_->predict_img_to_hit_time(bullet_speed, img_to_fire_time, fric_to_basis);
        return outpost_observer_->predict_shoot_pos(gimbal_yaw_to_basis, img_to_hit_time);
    } else {
        const float img_to_hit_time =
            car_observer_->predict_img_to_hit_time(bullet_speed, img_to_fire_time, fric_to_basis);
        auto kf_pred = std::make_tuple(
            kf_main_observing_armor_->value() + kf_main_observing_armor_->derivative() * img_to_hit_time,
            car_status_->status() == StatusType::TRACKING || car_status_->status() == StatusType::TEMP_LOST
        );
        auto car_pred = car_observer_->predict_shoot_pos(gimbal_yaw_to_basis, img_to_hit_time);
        if (std::get<1>(kf_pred)) {
            update_kf_armor_pred_pos_history(
                current_update_time_ + img_to_hit_time,
                std::get<0>(kf_pred)
            );
        }
        if (std::get<1>(car_pred)) {
            update_car_pred_pos_history(
                current_update_time_ + img_to_hit_time,
                std::get<0>(car_pred)
            );
        }
        if (car_observer_->is_antispin_palstance_) {
            return car_pred;
        } else {
            return kf_pred;
        }
    }
}

void ArmorTracker::print_colored_status_info() const {
    const auto print_vec = [](const char* format, Vector3f vec) {
        std::printf(format, vec.x(), vec.y(), vec.z());
    };
    if (target_label_ == ArmorType::OUTPOST) {
        outpost_status_->print_colored_status_info();
    } else {
        car_status_->print_colored_status_info();
    }
    std::cout << termcolor::bold << "MainArmor   " << termcolor::reset;
    print_vec("[% 4.0f, % 4.0f, % 4.0f] += ", kf_main_observing_armor_->value() * 100);
    print_vec("[% 4.0f, % 4.0f, % 4.0f]\n", kf_main_observing_armor_->derivative() * 100);
    if (target_label_ == ArmorType::OUTPOST) {
        outpost_observer_->print_colored_status_info();
    } else {
        car_observer_->print_colored_status_info();
    }
    if (target_label_ != ArmorType::OUTPOST) {
        std::cout << termcolor::bold << "MainArmorAvgErr " << termcolor::reset;
        std::printf("%4.1f", kf_armor_avg_err_ * 100);
        std::cout << std::endl;
        std::cout << termcolor::bold << "CarAvgErr       " << termcolor::reset;
        std::printf("%4.1f", car_avg_err_ * 100);
        std::cout << std::endl;
        if ((car_observer_->is_antispin_palstance_ ? car_avg_err_ : kf_armor_avg_err_) > LOW_ACCURACY_ERR_THRESHOLD) {
            std::cout << termcolor::yellow << "Low tracking accuracy!" << termcolor::reset << std::endl;
        }
    }
}

std::vector<std::tuple<Vector3f, Quaternionf>> ArmorTracker::get_all_armors() const {
    if (target_label_ == ArmorType::OUTPOST) {
        return outpost_observer_->get_all_armors();
    } else {
        return car_observer_->get_all_armors();
    }
}

void ArmorTracker::write_predictor_status(hw_sentry_interfaces::msg::PredictorStatus& status) const {
    status.mode = static_cast<int>(AutoaimMode::ARMOR);
    status.label = static_cast<int>(target_label_);
    if (target_label_ == ArmorType::OUTPOST) {
        status.tracker_status = static_cast<int>(outpost_status_->status());
        outpost_observer_->write_predictor_status(status);
    } else {
        status.tracker_status = static_cast<int>(car_status_->status());
        car_observer_->write_predictor_status(status);
    }
}