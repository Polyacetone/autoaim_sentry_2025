#pragma once

#include <cmath>
#include <eigen3/Eigen/Dense>
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2/LinearMath/Matrix3x3.hpp>

#include <autoaim_common_utils/convert_utils.hpp>

namespace utils {

// 弧度转角度
static constexpr double r2d(const double rad) {
    return rad * 180.0 / M_PI;
}

// 角度转弧度
static constexpr double d2r(const double deg) {
    return deg * M_PI / 180.0;
}

// 平方
static constexpr double square(const double x) {
    return x * x;
}

// 把角度（弧度制）修正到-pi~pi之间
static constexpr double rad_period_correction(const double rad) {
    return rad + round((-rad) / (2.0 * M_PI)) * (2.0 * M_PI);
}

// 计算两个Eigen向量的夹角。返回值介于0~pi
static constexpr double get_angle(const Eigen::VectorXd& vec1, const Eigen::VectorXd& vec2) {
    double cosval = vec1.dot(vec2) / (vec1.norm() * vec2.norm());
    if (vec1.norm() * vec2.norm() < 0.001 || cosval > 0.999) [[unlikely]] {
        return 0.0;
    }
    return std::acos(cosval);
}

// 四元数转ypr欧拉角
static constexpr std::tuple<float, float, float> to_euler_ypr(const tf2::Quaternion& quat) {
    double yaw, pitch, roll;
    tf2::Matrix3x3 rot_mat(quat);
    rot_mat.getEulerYPR(yaw, pitch, roll);
    return {yaw, pitch, roll};
}

// 四元数转ypr欧拉角
static constexpr std::tuple<float, float, float> to_euler_ypr(const Eigen::Quaternionf& quat) {
    return to_euler_ypr(utils::convert_to<tf2::Quaternion>(quat));
}

} // namespace math_utils