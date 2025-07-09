#include <autoaim_locator/pnp_solver.hpp>

tf2::Transform PnPSolver::solve_pnp(
    const hw_sentry_interfaces::msg::ArmorDetection& detection,
    const std::tuple<float, float, float>& gimbal_ypr
) const {
    const ArmorType label = static_cast<ArmorType>(detection.label);
    const auto& obj_pts = defs::is_big_armor(label) ? BIG_POINTS : SMALL_POINTS;
    const std::array<cv::Point2f, 4> img_pts {
        cv::Point2f {detection.tl.x, detection.tl.y},
        cv::Point2f {detection.bl.x, detection.bl.y},
        cv::Point2f {detection.br.x, detection.br.y},
        cv::Point2f {detection.tr.x, detection.tr.y}
    };
    std::array<cv::Mat, 2> rvecs, tvecs;
    std::array<float, 2> reproj_err;
    cv::solvePnPGeneric(
        obj_pts,
        img_pts,
        cam_intrinsic_,
        cam_distortion_,
        rvecs, tvecs,
        false,
        cv::SOLVEPNP_IPPE,
        cv::noArray(), cv::noArray(), reproj_err
    );

    // 左乘即可把opencv的相机系（右x，下y，前z）转成我们在tf2中的相机系（前x，左y，上z）
    const Eigen::Quaternionf cv_to_tf(-0.5, 0.5, -0.5, 0.5);

    auto [gimbal_yaw, gimbal_pitch, gimbal_roll] = gimbal_ypr;
    const Eigen::AngleAxisf roll_rotation(gimbal_roll, Eigen::Vector3f::UnitX());
    const Eigen::AngleAxisf pitch_rotation(gimbal_pitch, Eigen::Vector3f::UnitY());
    // 左乘即可把我们在tf2的相机系转掉云台的pitch和roll
    const Eigen::Quaternionf gimbal_pr(roll_rotation * pitch_rotation);

    std::tuple<float, float, float> corrected_ypr[2];
    Eigen::Quaternionf rotation[2];
    Eigen::Vector3f translation[2];
    for (unsigned i = 0; i < 2; i++) {
        Eigen::Vector3f tvec, rvec;
        cv::cv2eigen(tvecs[i], tvec);
        cv::cv2eigen(rvecs[i], rvec);
        rotation[i] = cv_to_tf * Eigen::AngleAxisf(rvec.norm(), rvec.normalized());
        translation[i] = cv_to_tf * tvec;

        Eigen::Quaternionf corrected_rotation = gimbal_pr * rotation[i];
        corrected_ypr[i] =
            utils::to_euler_ypr(utils::convert_to<tf2::Quaternion>(corrected_rotation));
    }

    // 从IPPE给出的两个解中选择最合适的解
    unsigned index = 0;

    const auto normalize = [](std::array<float, 2>& arr) {
        const float sum = arr[0] + arr[1];
        std::for_each(arr.begin(), arr.end(), [&](float& v) { v /= sum; });
    };
    normalize(reproj_err);

    std::array<float, 2> prior_angle_diff;
    for (unsigned i = 0; i < 2; i++) {
        prior_angle_diff[i] =
            std::abs(std::get<1>(corrected_ypr[i]) - defs::armor_pitch(label))
            + std::abs(std::get<2>(corrected_ypr[i]));
    }
    normalize(prior_angle_diff);

    index = (reproj_err[0] + prior_angle_diff[0]) > (reproj_err[1] + prior_angle_diff[1]);
    tf2::Transform armor_to_cam;
    armor_to_cam.setRotation(utils::convert_to<tf2::Quaternion>(rotation[index]));
    armor_to_cam.setOrigin(utils::convert_to<tf2::Vector3>(translation[index]));
    return armor_to_cam;
}

void PnPSolver::set_cam_matrix(const cv::Mat intrinsic, const cv::Mat distortion) {
    cam_intrinsic_ = intrinsic.clone();
    cam_distortion_ = distortion.clone();
}