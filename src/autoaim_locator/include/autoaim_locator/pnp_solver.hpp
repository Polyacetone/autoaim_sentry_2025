#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/convert.hpp>
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2/LinearMath/Matrix3x3.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <hw_sentry_interfaces/msg/detections.hpp>

#include <autoaim_locator/lstm_pose_smoothing.hpp>
#include <autoaim_common_utils/tf_utils.hpp>
#include <autoaim_common_utils/math_utils.hpp>
#include <autoaim_common_utils/convert_utils.hpp>
#include <autoaim_common_definitions/common_definitions.hpp>

class PnPSolver {
public:
    /*!
        @brief 设置相机的内参和畸变矩阵
        @attention 在使用solve_pnp(...)之前一定要先设置内参矩阵
    */
    void set_cam_matrix(const cv::Mat intrinsic, const cv::Mat distortion);

    std::vector<tf2::Transform> solve_pnp(
        const std::vector<hw_sentry_interfaces::msg::ArmorDetection>& detections,
        const std::tuple<float, float, float>& gimbal_ypr,
        const double timestamp = 0,
        const std::shared_ptr<LSTMPoseSmoothing> lstm = nullptr
    );

private:
    void solve_pnp_cv(
        const std::vector<hw_sentry_interfaces::msg::ArmorDetection>& detections,
        std::vector<std::array<cv::Mat, 2>>& rvecs,
        std::vector<std::array<cv::Mat, 2>>& tvecs,
        std::vector<std::array<float, 2>>& reprojerrs
    ) const;

    void cvcoord_to_tfcoord(
        const std::vector<std::array<cv::Mat, 2>>& rvecs,
        const std::vector<std::array<cv::Mat, 2>>& tvecs,
        std::vector<std::array<Eigen::Quaternionf, 2>>& rotations,
        std::vector<std::array<Eigen::Vector3f, 2>>& translations
    ) const;

    int select_solution_prior_angle(
        const std::array<Eigen::Quaternionf, 2>& rotations,
        const std::array<float, 2>& reprojerrs,
        const std::tuple<float, float, float>& gimbal_ypr,
        const float prior_pitch
    ) const;

    int select_solution_lstm(
        const std::array<Eigen::Quaternionf, 2>& rotations,
        const std::array<float, 2>& reprojerrs,
        const float dt,
        const std::shared_ptr<LSTMPoseSmoothing> lstm
    ) const;

    std::array<int, 2> select_solution_armors_relative_position(
        const std::array<std::array<Eigen::Quaternionf, 2>, 2>& rotations,
        const std::array<std::array<Eigen::Vector3f, 2>, 2>& translations
    ) const;

    static constexpr float DETECTOR_ERROR_PIXEL_BY_SLOPE = 2.0f;

    // 单位: 米
    static constexpr float HEIGHT = 0.05603f;
    static constexpr float BIG_WIDTH = 0.231f;
    static constexpr float SMALL_WIDTH = 0.136f;
    // 装甲板坐标系：前x，左y，上z
    const std::vector<cv::Point3f> SMALL_POINTS {
        {0, SMALL_WIDTH / 2, HEIGHT / 2},
        {0, SMALL_WIDTH / 2, -HEIGHT / 2},
        {0, -SMALL_WIDTH / 2, -HEIGHT / 2},
        {0, -SMALL_WIDTH / 2, HEIGHT / 2}
    };
    const std::vector<cv::Point3f> BIG_POINTS {
        {0, BIG_WIDTH / 2, HEIGHT / 2},
        {0, BIG_WIDTH / 2, -HEIGHT / 2},
        {0, -BIG_WIDTH / 2, -HEIGHT / 2},
        {0, -BIG_WIDTH / 2, HEIGHT / 2}
    };

    cv::Mat cam_intrinsic_ = (cv::Mat_<float>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    cv::Mat cam_distortion_ = (cv::Mat_<float>(1, 5) << 0.0, 0.0, 0.0, 0.0, 0.0);

    double lstm_prev_update_time_ = -1;
};