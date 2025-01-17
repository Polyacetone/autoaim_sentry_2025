#pragma once

#include <math.h>
#include <Eigen/Dense>

namespace math {
constexpr float rad_period_correction(const float rad) {
    return rad + round((-rad) / (2 * M_PI)) * (2 * M_PI);
}

constexpr float r2d(const float rad) {
    return rad * 180.0 / M_PI;
}

constexpr float d2r(const float deg) {
    return deg * M_PI / 180.0;
}

constexpr float squre(const float x) {
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
    return sqrt(squre(point.x) + squre(point.y) + squre(point.z));
}
} // namespace math