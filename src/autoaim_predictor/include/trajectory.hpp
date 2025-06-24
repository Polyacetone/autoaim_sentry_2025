#pragma once

#include <tuple>
#include <cmath>

namespace trajectory {

constexpr float G = 9.8; // 重力加速度
constexpr float AIR_DENSITY = 1.1691; // 空气密度 25摄氏度
constexpr float CD = 0.56; // 球的阻力系数
constexpr float MASS = 0.0032; // 质量 kg
constexpr float AREA = M_PI * (0.017 / 2) * (0.017 / 2); // 横截面积 m^2
constexpr float COEFF = 0.5 * AIR_DENSITY * CD * AREA / MASS; // 阻力系数

std::tuple<float, float> shoot_altitude(float theta, float v, float d);

/*!
    @param d: 目标到枪口的距离
    @param h: 目标相对于枪口的高度
    @param v: 弹速
    @return 射击角度和飞行时间。射击角度向上为正，单位rad
*/
std::tuple<float, float> get_pitch_air_frac(float d, float h, float v);

} // namespace trajectory