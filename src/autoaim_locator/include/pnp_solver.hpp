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

#include <autoaim_common_utils/math_utils.hpp>

class PnPSolver {
public:
    /*!
        @brief 设置相机的内参和畸变矩阵
        @attention 在使用solve_pnp(...)之前一定要先设置内参矩阵
    */
    void set_cam_matrix(const cv::Mat intrinsic, const cv::Mat distortion);

    /*!
        @brief 用opencv的solvepnp求位置，用几何法优化的三分法求yaw
        @return 返回false表示找解的时候有问题
    */
    std::variant<std::monostate, tf2::Transform> solve_pnp(
        const hw_sentry_interfaces::msg::ArmorDetection& detection,
        const std::tuple<float, float, float>& gimbal_ypr
    ) const;

private:
    // 取重投影后的上中点和下中点连线，根据斜率求yaw角
    float geometric_get_yaw(
        const Eigen::Quaternionf& gimbal_pr,
        const Eigen::Quaternionf& rotation,
        const Eigen::Vector3f& translation,
        const int label
    ) const;

    // 对重投影后的pts_cost进行三分法求最小值来求装甲板yaw角
    float trisection_get_yaw(
        const Eigen::Quaternionf& gimbal_pr,
        const Eigen::Vector3f& translation,
        const std::vector<cv::Point2f>& img_pts,
        const int label,
        const float prior_yaw
    ) const;

    // 计算实际的角点和重投影后的角点的差异，作为传入三分法的损失函数
    float get_pts_cost(
        const std::vector<cv::Point2f>& ref_pts,
        const std::vector<cv::Point2f>& rotated_pts,
        const float prior_yaw
    ) const;

    // 计算旋转角（armor_pitch是装甲板相对于世界系的pitch，是已知的；armor_yaw是三分法扔进来的变量）对应的装甲板角点坐标
    std::vector<Eigen::Vector3f> get_spinned_pts(
        const Eigen::Vector3f& armor_center, 
        const int armor_label,
        const float armor_pitch,
        const float armor_yaw
    ) const;

    // 三分法求函数的最小值
    std::pair<float, float> trisection_find_min(
        float left,
        float right,
        const std::function<float(float)>& cost_function,
        const int iterations
    ) const;

    float get_armor_pitch_to_world(int label) const;
    bool is_big_armor(int label) const;

    static constexpr int FIND_ANGLE_ITERATIONS = 12; // 三分法迭代次数，理想精度<1
    static constexpr float DETECTOR_ERROR_PIXEL_BY_SLOPE = 2.0;

    // 单位: 米
    static constexpr float HEIGHT = 0.05603;
    static constexpr float BIG_WIDTH = 0.231;
    static constexpr float SMALL_WIDTH = 0.136;
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