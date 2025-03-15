#pragma once

#include <math.h>
#include <Eigen/Dense>

namespace math {
// 把角度（弧度制）修正到-pi~pi之间
constexpr float rad_period_correction(const float rad) {
    return rad + round((-rad) / (2 * M_PI)) * (2 * M_PI);
}

// 弧度转角度
constexpr float r2d(const float rad) {
    return rad * 180.0 / M_PI;
}

// 角度转弧度
constexpr float d2r(const float deg) {
    return deg * M_PI / 180.0;
}

constexpr float square(const float x) {
    return x * x;
}

// 计算两个Eigen向量的夹角。返回值介于0~pi。
float get_angle(const Eigen::Vector2f& vec1, const Eigen::Vector2f& vec2) {
    if (vec1.norm() == 0.0 || vec2.norm() == 0.0) {
        return 0.0;
    }
    return std::acos(vec1.dot(vec2) / (vec1.norm() * vec2.norm()));
}

float get_distance(const cv::Point3f& point) {
    return sqrt(square(point.x) + square(point.y) + square(point.z));
}

float get_distance(const cv::Point2f& point) {
    return sqrt(square(point.x) + square(point.y));
}

float get_distance(const cv::Point2f& point1, const cv::Point2f& point2) {
    return get_distance(point2 - point1);
}

Eigen::Vector3d to_euler_angle(Eigen::Matrix3d rot_mat) {
    Eigen::Vector3d euler_angle;
    euler_angle(0) = std::atan2(-rot_mat(0, 1), rot_mat(1, 1));
    euler_angle(1) = std::atan2(
        rot_mat(2, 1),
        std::sqrt(rot_mat(0, 1) * rot_mat(0, 1) + rot_mat(1, 1) * rot_mat(1, 1))
    );
    euler_angle(2) = std::atan2(-rot_mat(2, 0), rot_mat(2, 2));
    return euler_angle;
}


} // namespace math