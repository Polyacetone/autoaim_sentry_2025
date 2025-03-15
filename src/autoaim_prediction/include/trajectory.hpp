#pragma once

#include <tuple>
#include <math.h>

#include <math_utils.hpp>

namespace trajectory {

constexpr float G = 9.8; // 重力加速度
constexpr float AIR_DENSITY = 1.1691; // 空气密度 25摄氏度
constexpr float Cd = 0.56; // 球的阻力系数
constexpr float MASS = 0.0032; // 质量 kg
constexpr float AREA = M_PI * (0.017 / 2) * (0.017 / 2); // 横截面积 m^2
constexpr float Coeff = 0.5 * AIR_DENSITY * Cd * AREA / MASS; // 阻力系数

std::tuple<float, float> shoot_altitude(float theta, float v, float d) {
    constexpr auto air_fraction_acc = [](float v) -> float {
        return Coeff * v * v;
    };
    float vd, vh, cur_d, cur_h, cur_v, fly_time;
    float dt = 1e-4; // 0.1ms
    cur_d = 0;
    cur_h = 0;
    vd = v * cos(theta);
    vh = v * sin(theta);
    fly_time = 0;
    for (int i = 0; i < 10000 && cur_d < d; i++) {
        cur_d += vd * dt;
        cur_h += vh * dt;
        cur_v = sqrt(pow(vd, 2) + pow(vh, 2));
        vd += -air_fraction_acc(cur_v) * vd / cur_v * dt;
        vh += -air_fraction_acc(cur_v) * vh / cur_v * dt - G * dt;
        fly_time += dt;
    }
    return std::make_tuple(cur_h, fly_time);
}

// d: 目标到枪口的距离, h: 目标相对于枪口的高度, v: 弹速
std::tuple<float, float> get_pitch_air_frac(float d, float h, float v) {
    constexpr auto get_pitch = [](float d, float h, float v) -> float {
        float pitch = atan(
            (1 - sqrt(1 - 2 * G / pow(v, 2) * (h + 0.5 * G * pow(d, 2) / pow(v, 2))))
            / (G * d / pow(v, 2))
        );
        return pitch;
    };
    float pitch = -1;
    float h_actual = h, h_cur;
    float fly_time;
    for (int i = 0; i <= 20; i++) {
        pitch = get_pitch(d, h, v);
        std::tie(h_cur, fly_time) = shoot_altitude(pitch, v, d);
        float err = h_cur - h_actual;
        if (abs(err) <= 1e-4) {
            break;
        }
        h -= err;
    }
    // 如果没有找到合适的pitch角度
    if (pitch == -1) {
        pitch = get_pitch(d, h, v);
        fly_time = d / (v * cos(pitch));
    }
    return std::make_tuple(pitch, fly_time);
}

float calc_pitch(float x, float y, float z, float speed) {
    float pitch, fly_time;
    std::tie(pitch, fly_time) = get_pitch_air_frac(sqrt(math::square(x) + math::square(y)), z, speed);
    return pitch;
}

} // namespace trajectory