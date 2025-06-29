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

    tf2::Transform solve_pnp(
        const hw_sentry_interfaces::msg::ArmorDetection& detection,
        const std::tuple<float, float, float>& gimbal_ypr
    ) const;

private:
    // 计算实际的角点和重投影后的角点的差异
    float get_reprojection_err(
        const std::vector<cv::Point2f>& ref_pts,
        const std::vector<cv::Point2f>& reprojected_pts,
        const float prior_yaw
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
};