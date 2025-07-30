#include <autoaim_locator/pnp_solver.hpp>
#include <variant>
#include <vector>

using namespace hw_sentry_interfaces::msg;

std::vector<tf2::Transform> PnPSolver::solve_pnp(
    const std::variant<std::vector<ArmorDetection>, std::vector<BuffDetection>>& detections,
    const std::tuple<float, float, float>& gimbal_ypr,
    const double timestamp,
    const std::shared_ptr<LSTMPoseSmoothing> lstm
) {
    if (std::holds_alternative<std::vector<BuffDetection>>(detections)) {
        const auto& buff_detections = std::get<std::vector<BuffDetection>>(detections);
        if (buff_detections.size() != 1) {
            return {};
        }
        
        std::vector<std::array<cv::Mat, 2>> rvecs;
        std::vector<std::array<cv::Mat, 2>> tvecs;
        std::vector<std::array<float, 2>> reprojerrs;
        solve_pnp_cv(buff_detections, rvecs, tvecs, reprojerrs);

        std::vector<std::array<Eigen::Quaternionf, 2>> rotations;
        std::vector<std::array<Eigen::Vector3f, 2>> translations;
        cvcoord_to_tfcoord(rvecs, tvecs, rotations, translations);
        
        std::vector<tf2::Transform> buffs_to_cam;
        for (size_t i = 0; i < rotations.size(); i++) {
            buffs_to_cam.emplace_back(
                utils::convert_to<tf2::Quaternion>(rotations[i][0]),
                utils::convert_to<tf2::Vector3>(translations[i][0])
            );
        }
        return buffs_to_cam;
    } else if (std::holds_alternative<std::vector<ArmorDetection>>(detections)) {
        const auto& armor_detections = std::get<std::vector<ArmorDetection>>(detections);
        if (armor_detections.size() != 1 && armor_detections.size() != 2) {
            return {};
        }

        std::vector<std::array<cv::Mat, 2>> rvecs;
        std::vector<std::array<cv::Mat, 2>> tvecs;
        std::vector<std::array<float, 2>> reprojerrs;
        solve_pnp_cv(armor_detections, rvecs, tvecs, reprojerrs);

        std::vector<std::array<Eigen::Quaternionf, 2>> rotations;
        std::vector<std::array<Eigen::Vector3f, 2>> translations;
        cvcoord_to_tfcoord(rvecs, tvecs, rotations, translations);

        if (armor_detections.size() == 1) {
            if (lstm) {
                const float dt = static_cast<float>(std::clamp(timestamp - lstm_prev_update_time_, 0.0, 10.0));
                const auto refined_rotation = refine_solution_lstm(rotations[0], reprojerrs[0], dt, lstm);
                lstm_prev_update_time_ = timestamp;
                return {tf2::Transform(
                    utils::convert_to<tf2::Quaternion>(refined_rotation),
                    utils::convert_to<tf2::Vector3>(translations[0][0])
                )};
            } else {
                const int index = select_solution_prior_angle(
                    rotations[0],
                    gimbal_ypr,
                    defs::armor_pitch(static_cast<ArmorType>(armor_detections[0].label))
                );
                return {tf2::Transform(
                    utils::convert_to<tf2::Quaternion>(rotations[0][index]),
                    utils::convert_to<tf2::Vector3>(translations[0][index])
                )};
            }
        } else if (armor_detections.size() == 2) {
            const auto indices = select_solution_armors_relative_position(
                {rotations[0], rotations[1]},
                {translations[0], translations[1]}
            );
            return {
                tf2::Transform(
                    utils::convert_to<tf2::Quaternion>(rotations[0][indices[0]]),
                    utils::convert_to<tf2::Vector3>(translations[0][indices[0]])
                ),
                tf2::Transform(
                    utils::convert_to<tf2::Quaternion>(rotations[1][indices[1]]),
                    utils::convert_to<tf2::Vector3>(translations[1][indices[1]])
                )
            };
        }
    }  
    return {};
}

void PnPSolver::solve_pnp_cv(
    const std::variant<std::vector<ArmorDetection>, std::vector<BuffDetection>>& detections,
    std::vector<std::array<cv::Mat, 2>>& rvecs,
    std::vector<std::array<cv::Mat, 2>>& tvecs,
    std::vector<std::array<float, 2>>& reprojerrs
) const {
    if (std::holds_alternative<std::vector<BuffDetection>>(detections)) {
        const auto& buff_detections = std::get<std::vector<BuffDetection>>(detections);
        std::for_each(buff_detections.begin(), buff_detections.end(), [&](const auto& detection) {
            const auto& obj_pts = BUFF_POINTS;
            const std::array<cv::Point2f, 4> img_pts {
                cv::Point2f {detection.t.x, detection.t.y},
                cv::Point2f {detection.l.x, detection.l.y},
                cv::Point2f {detection.b.x, detection.b.y},
                cv::Point2f {detection.r.x, detection.r.y}
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
    } else if (std::holds_alternative<std::vector<ArmorDetection>>(detections)) {
        const auto& armor_detections = std::get<std::vector<ArmorDetection>>(detections);
        std::for_each(armor_detections.begin(), armor_detections.end(), [&](const auto& detection) {
            const ArmorType label = static_cast<ArmorType>(detection.label);
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
}

void PnPSolver::cvcoord_to_tfcoord(
    const std::vector<std::array<cv::Mat, 2>>& rvecs,
    const std::vector<std::array<cv::Mat, 2>>& tvecs,
    std::vector<std::array<Eigen::Quaternionf, 2>>& rotations,
    std::vector<std::array<Eigen::Vector3f, 2>>& translations
) const {
    const size_t len = rvecs.size();
    rotations.resize(len);
    translations.resize(len);
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

Eigen::Quaternionf PnPSolver::refine_solution_lstm(
    const std::array<Eigen::Quaternionf, 2>& rotations,
    const std::array<float, 2>& reprojerrs,
    const float dt,
    const std::shared_ptr<LSTMPoseSmoothing> lstm
) const {
    const auto ypr0 = utils::to_euler_ypr(rotations[0]);
    const auto ypr1 = utils::to_euler_ypr(rotations[1]);
    const auto yaw0 = std::get<0>(ypr0);
    const auto yaw1 = std::get<0>(ypr1);
    const float pred_yaw = lstm->infer(yaw0, yaw1, reprojerrs[0], reprojerrs[1], dt);
    const float diff0 = std::abs(pred_yaw - yaw0);
    const float diff1 = std::abs(pred_yaw - yaw1);
    const auto refined_rotation =
        Eigen::AngleAxisf(diff0 < diff1 ? std::get<2>(ypr0) : std::get<2>(ypr1), Eigen::Vector3f::UnitX())
        * Eigen::AngleAxisf(diff0 < diff1 ? std::get<1>(ypr0) : std::get<1>(ypr1), Eigen::Vector3f::UnitY())
        * Eigen::AngleAxisf(pred_yaw, Eigen::Vector3f::UnitZ());
    return Eigen::Quaternionf(refined_rotation);
}

int PnPSolver::select_solution_prior_angle(
    const std::array<Eigen::Quaternionf, 2>& rotations,
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

    std::array<float, 2> prior_angle_diff;
    for (unsigned i = 0; i < 2; i++) {
        prior_angle_diff[i] =
            std::abs(std::get<1>(corrected_ypr[i]) - prior_pitch)
            + std::abs(std::get<2>(corrected_ypr[i]));
    }

    return prior_angle_diff[0] > prior_angle_diff[1];
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