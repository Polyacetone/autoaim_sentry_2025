#pragma once

#include <cmath>
#include <Eigen/Dense>

namespace utils {

// 弧度转角度
static constexpr float r2d(const float rad) {
    return rad * 180.0 / M_PI;
}

// 角度转弧度
static constexpr float d2r(const float deg) {
    return deg * M_PI / 180.0;
}

// 平方
static constexpr float square(const float x) {
    return x * x;
}

// 把角度（弧度制）修正到-pi~pi之间
static constexpr float rad_period_correction(const float rad) {
    return rad + round((-rad) / (2 * M_PI)) * (2 * M_PI);
}

// 计算两个Eigen向量的夹角。返回值介于0~pi
static constexpr float get_angle(const Eigen::Vector2f& vec1, const Eigen::Vector2f& vec2) {
    if (vec1.norm() == 0.0 || vec2.norm() == 0.0) {
        return 0.0;
    }
    return std::acos(vec1.dot(vec2) / (vec1.norm() * vec2.norm()));
}

} // namespace math_utils