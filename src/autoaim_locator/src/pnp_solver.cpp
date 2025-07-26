#include <autoaim_locator/pnp_solver.hpp>

std::vector<tf2::Transform> PnPSolver::solve_pnp(
    const std::vector<hw_sentry_interfaces::msg::ArmorDetection>& detections,
    const std::tuple<float, float, float>& gimbal_ypr,
    const double timestamp,
    const std::shared_ptr<LSTMPoseSmoothing> lstm
) {
    if (detections.size() != 1 && detections.size() != 2) {
        return {};
    }

    std::vector<std::array<cv::Mat, 2>> rvecs;
    std::vector<std::array<cv::Mat, 2>> tvecs;
    std::vector<std::array<float, 2>> reprojerrs;
    solve_pnp_cv(detections, rvecs, tvecs, reprojerrs);

    std::vector<std::array<Eigen::Quaternionf, 2>> rotations;
    std::vector<std::array<Eigen::Vector3f, 2>> translations;
    cvcoord_to_tfcoord(rvecs, tvecs, rotations, translations);

    std::vector<int> indexes;
    if (detections.size() == 1) {
        int index;
        if (lstm) {
            const float dt = static_cast<float>(std::clamp(timestamp - lstm_prev_update_time_, 0.0, 10.0));
            index = select_solution_lstm(rotations[0], reprojerrs[0], dt, lstm);
            lstm_prev_update_time_ = timestamp;
        } else {
            index = select_solution_prior_angle(
                rotations[0],
                reprojerrs[0],
                gimbal_ypr,
                defs::armor_pitch(static_cast<ArmorLabel>(detections[0].label))
            );
        }
        indexes.emplace_back(index);
    } else if (detections.size() == 2) {
        const auto index = select_solution_armors_relative_position(
            {rotations[0], rotations[1]},
            {translations[0], translations[1]}
        );
        indexes.emplace_back(index[0]);
        indexes.emplace_back(index[1]);
    }
    
    std::vector<tf2::Transform> armors_to_cam;
    for (size_t i = 0; i < indexes.size(); i++) {
        armors_to_cam.emplace_back(
            utils::convert_to<tf2::Quaternion>(rotations[i][indexes[i]]),
            utils::convert_to<tf2::Vector3>(translations[i][indexes[i]])
        );
    }
    return armors_to_cam;
}

void PnPSolver::solve_pnp_cv(
    const std::vector<hw_sentry_interfaces::msg::ArmorDetection>& detections,
    std::vector<std::array<cv::Mat, 2>>& rvecs,
    std::vector<std::array<cv::Mat, 2>>& tvecs,
    std::vector<std::array<float, 2>>& reprojerrs
) const {
    std::for_each(detections.begin(), detections.end(), [&](const auto& detection) {
        const ArmorLabel label = static_cast<ArmorLabel>(detection.label);
        const auto& obj_pts = defs::is_big_armor(label) ? BIG_POINTS : SMALL_POINTS;
        const std::array<cv::Point2f, 4> img_pts {
            cv::Point2f {detection.tl.x, detection.tl.y},
            cv::Point2f {detection.bl.x, detection.bl.y},
            cv::Point2f {detection.br.x, detection.br.y},
            cv::Point2f {detection.tr.x, detection.tr.y}
        };
        std::array<cv::Mat, 2> rvec, tvec;
        std::array<float, 2> reprojerr;
        cv::solvePnPGeneric(
            obj_pts,
            img_pts,
            cam_intrinsic_,
            cam_distortion_,
            rvec,
            tvec,
            false,
            cv::SOLVEPNP_IPPE,
            cv::noArray(),
            cv::noArray(),
            reprojerr
        );
        rvecs.emplace_back(rvec);
        tvecs.emplace_back(tvec);
        reprojerrs.emplace_back(reprojerr);
    });
}

void PnPSolver::cvcoord_to_tfcoord(
    const std::vector<std::array<cv::Mat, 2>>& rvecs,
    const std::vector<std::array<cv::Mat, 2>>& tvecs,
    std::vector<std::array<Eigen::Quaternionf, 2>>& rotations,
    std::vector<std::array<Eigen::Vector3f, 2>>& translations
) const {
    const size_t len = rvecs.size();
    rotations.reserve(len);
    translations.reserve(len);
    for (size_t i = 0; i < len; i++) {
        // 左乘即可把opencv的相机系（右x，下y，前z）转成我们在tf2中的相机系（前x，左y，上z）
        const Eigen::Quaternionf cv_to_tf(-0.5, 0.5, -0.5, 0.5);
        Eigen::Vector3f tvec, rvec;
        for (int j = 0; j < 2; j++) {
            cv::cv2eigen(tvecs[i][j], tvec);
            cv::cv2eigen(rvecs[i][j], rvec);
            rotations[i][j] = cv_to_tf * Eigen::AngleAxisf(rvec.norm(), rvec.normalized());
            translations[i][j] = cv_to_tf * tvec;
        }
    }
}

int PnPSolver::select_solution_lstm(
    const std::array<Eigen::Quaternionf, 2>& rotations,
    const std::array<float, 2>& reprojerrs,
    const float dt,
    const std::shared_ptr<LSTMPoseSmoothing> lstm
) const {
    const auto yaw0 = std::get<0>(utils::to_euler_ypr(rotations[0]));
    const auto yaw1 = std::get<0>(utils::to_euler_ypr(rotations[1]));
    const float result = lstm->infer(yaw0, yaw1, reprojerrs[0], reprojerrs[1], dt);
    const float diff0 = std::abs(result - yaw0);
    const float diff1 = std::abs(result - yaw1);
    return diff0 > diff1;
}

int PnPSolver::select_solution_prior_angle(
    const std::array<Eigen::Quaternionf, 2>& rotations,
    const std::array<float, 2>& reprojerrs,
    const std::tuple<float, float, float>& gimbal_ypr,
    const float prior_pitch
) const {
    auto [gimbal_yaw, gimbal_pitch, gimbal_roll] = gimbal_ypr;
    const Eigen::AngleAxisf roll_rotation(gimbal_roll, Eigen::Vector3f::UnitX());
    const Eigen::AngleAxisf pitch_rotation(gimbal_pitch, Eigen::Vector3f::UnitY());

    // 左乘即可把我们在tf2的相机系转掉云台的pitch和roll
    const Eigen::Quaternionf gimbal_pr(roll_rotation * pitch_rotation);

    std::tuple<float, float, float> corrected_ypr[2];
    for (unsigned i = 0; i < 2; i++) {
        Eigen::Quaternionf corrected_rotation = gimbal_pr * rotations[i];
        corrected_ypr[i] =
            utils::to_euler_ypr(utils::convert_to<tf2::Quaternion>(corrected_rotation));
    }

    std::array<float, 2> reproj_errs = reprojerrs;
    const auto normalize = [](std::array<float, 2>& arr) {
        const float sum = arr[0] + arr[1];
        std::for_each(arr.begin(), arr.end(), [&](float& v) { v /= sum; });
    };
    normalize(reproj_errs);

    std::array<float, 2> prior_angle_diff;
    for (unsigned i = 0; i < 2; i++) {
        prior_angle_diff[i] =
            std::abs(std::get<1>(corrected_ypr[i]) - prior_pitch)
            + std::abs(std::get<2>(corrected_ypr[i]));
    }
    normalize(prior_angle_diff);

    return (reproj_errs[0] + prior_angle_diff[0]) > (reproj_errs[1] + prior_angle_diff[1]);
}

std::array<int, 2> PnPSolver::select_solution_armors_relative_position(
    const std::array<std::array<Eigen::Quaternionf, 2>, 2>& rotations,
    const std::array<std::array<Eigen::Vector3f, 2>, 2>& translations
) const {
    std::array<int, 2> selected_indexes;

    const int left_armor = translations[1][0].y() > translations[0][0].y();
    const auto& ypr_left0 = utils::to_euler_ypr(rotations[left_armor][0]);
    const auto& ypr_left1 = utils::to_euler_ypr(rotations[left_armor][1]);
    selected_indexes[left_armor] = std::get<0>(ypr_left0) > std::get<0>(ypr_left1);

    const int right_armor = translations[1][0].y() < translations[0][0].y();
    const auto& ypr_right0 = utils::to_euler_ypr(rotations[right_armor][0]);
    const auto& ypr_right1 = utils::to_euler_ypr(rotations[right_armor][1]);
    selected_indexes[right_armor] = std::get<0>(ypr_right0) < std::get<0>(ypr_right1);

    return selected_indexes;
}

void PnPSolver::set_cam_matrix(const cv::Mat intrinsic, const cv::Mat distortion) {
    cam_intrinsic_ = intrinsic.clone();
    cam_distortion_ = distortion.clone();
}