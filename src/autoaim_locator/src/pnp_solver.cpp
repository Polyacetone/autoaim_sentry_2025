#include <pnp_solver.hpp>

tf2::Transform PnPSolver::solve_pnp(
    const hw_sentry_interfaces::msg::ArmorDetection& detection,
    const std::tuple<float, float, float>& gimbal_ypr
) const {
    const ArmorType label = static_cast<ArmorType>(detection.label);
    const auto& obj_pts = defs::is_big_armor(label) ? BIG_POINTS : SMALL_POINTS;
    const std::vector<cv::Point2f> img_pts {
        {detection.tl.x, detection.tl.y},
        {detection.bl.x, detection.bl.y},
        {detection.br.x, detection.br.y},
        {detection.tr.x, detection.tr.y}
    };
    std::vector<cv::Mat> rvecs(2), tvecs(2);
    cv::solvePnPGeneric(
        obj_pts,
        img_pts,
        cam_intrinsic_,
        cam_distortion_,
        rvecs, tvecs,
        false,
        cv::SOLVEPNP_IPPE
    );

    // 左乘即可把opencv的相机系（右x，下y，前z）转成我们在tf2中的相机系（前x，左y，上z）
    const Eigen::Quaternionf cv_to_tf(-0.5, 0.5, -0.5, 0.5);

    float gimbal_pitch, gimbal_roll;
    std::tie(std::ignore, gimbal_pitch, gimbal_roll) = gimbal_ypr;
    const Eigen::AngleAxisf roll_rotation(gimbal_roll, Eigen::Vector3f::UnitX());
    const Eigen::AngleAxisf pitch_rotation(gimbal_pitch, Eigen::Vector3f::UnitY());
    // 左乘即可把我们在tf2的相机系转掉云台的pitch和roll
    const Eigen::Quaternionf gimbal_pr(roll_rotation * pitch_rotation);

    float corrected_pitch[2];
    Eigen::Quaternionf rotation[2];
    Eigen::Vector3f translation[2];
    for (int i = 0; i < 2; i++) {
        Eigen::Vector3f tvec, rvec;
        cv::cv2eigen(tvecs[i], tvec);
        cv::cv2eigen(rvecs[i], rvec);
        rotation[i] = cv_to_tf * Eigen::AngleAxisf(rvec.norm(), rvec.normalized());
        translation[i] = cv_to_tf * tvec;

        Eigen::Quaternionf corrected_rotation = gimbal_pr * rotation[i];
        auto corrected_ypr =
            utils::to_euler_ypr(utils::convert_to<tf2::Quaternion>(corrected_rotation));
        corrected_pitch[i] = std::get<1>(corrected_ypr);
    }

    int index;
    if (defs::is_armor_pitch_negative(label)) {
        index = corrected_pitch[0] < corrected_pitch[1] ? 0 : 1;
    } else {
        index = corrected_pitch[0] > corrected_pitch[1] ? 0 : 1;
    }

    tf2::Transform armor_to_cam;
    armor_to_cam.setRotation(utils::convert_to<tf2::Quaternion>(rotation[index]));
    armor_to_cam.setOrigin(utils::convert_to<tf2::Vector3>(translation[index]));
    return armor_to_cam;
}

void PnPSolver::set_cam_matrix(const cv::Mat intrinsic, const cv::Mat distortion) {
    cam_intrinsic_ = intrinsic.clone();
    cam_distortion_ = distortion.clone();
}

float PnPSolver::get_reprojection_err(
    const std::vector<cv::Point2f>& ref_pts,
    const std::vector<cv::Point2f>& reprojected_pts,
    const float prior_yaw
) const {
    std::vector<Eigen::Vector2d> refs;
    std::vector<Eigen::Vector2d> pts;
    for (int i = 0; i < 4; i++) {
        refs.emplace_back(ref_pts[i].x, ref_pts[i].y);
        pts.emplace_back(reprojected_pts[i].x, reprojected_pts[i].y);
    }
    double cost = 0.0;
    for (int i = 0; i < 4; i++) {
        int p = (i + 1) % 4;
        // i - p 构成线段。过程：先移动起点，再补长度，再旋转
        Eigen::Vector2d ref_d = refs[p] - refs[i]; // 标准
        Eigen::Vector2d pt_d = pts[p] - pts[i];
        // 长度差代价 + 起点差代价 / 2（0 度左右应该抛弃）
        double pixel_dis = // dis 是指方差平面内到原点的距离
            (0.5 * ((refs[i] - pts[i]).norm() + (refs[p] - pts[p]).norm())
            + std::fabs(ref_d.norm() - pt_d.norm())) / ref_d.norm();
        double angular_dis = ref_d.norm() * utils::get_angle(ref_d, pt_d) / ref_d.norm();
        // 平方可能是为了配合 sin 和 cos
        // 弧度差代价（0 度左右占比应该大）
        double cost_i = utils::square(pixel_dis * std::sin(prior_yaw))
            + utils::square(angular_dis * std::cos(prior_yaw)) * DETECTOR_ERROR_PIXEL_BY_SLOPE;
        // 重投影像素误差越大，越相信斜率
        cost += std::sqrt(cost_i);
    }
    return static_cast<float>(cost);
}