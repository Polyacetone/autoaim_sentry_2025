#pragma once

#include <tuple>
#include <math.h>

namespace trajectory {

constexpr float G = 9.8; // 重力加速度
constexpr float AIR_DENSITY = 1.1691; // 空气密度 25摄氏度
constexpr float Cd = 0.56; // 球的阻力系数
constexpr float MASS = 0.0032; // 质量 kg
constexpr float AREA = M_PI * (0.017 / 2) * (0.017 / 2); // 横截面积 m^2
constexpr float Coeff = 0.5 * AIR_DENSITY * Cd * AREA / MASS; // 阻力系数

float r2d(float rad) {
    return rad * 180.0 / M_PI;
}
float d2r(float deg) {
    return deg * M_PI / 180.0;
}

float air_fraction_acc(float v) {
    return Coeff * v * v;
}

float get_pitch(float y, float z, float v) {
    float pitch_angle = atan(
        (1 - sqrt(1 - 2 * G / pow(v, 2) * (y + 0.5 * G * pow(z, 2) / pow(v, 2))))
        / (G * z / pow(v, 2))
    );
    return pitch_angle;
}

float yaw_period_correction(float yaw_current) {
    int round = int(yaw_current * 100.f); // 保留两位小数
    round = round % 36000;
    if (round < -18000)
        return float(round + 36000.f) / 100.f;
    else if (round > 18000)
        return float(round - 36000.f) / 100.f;
    else
        return round / 100.f;
}

std::tuple<float, float> shoot_altitude(float theta, float v0, float z) {
    float vz, vy, cur_z, cur_y, cur_v, fly_time;
    float dt = 1e-4; // 0.1ms
    cur_z = 0;
    cur_y = 0;
    vz = v0 * cos(theta);
    vy = v0 * sin(theta);
    fly_time = 0;
    for (int i = 0; i < 10000 && cur_z < z; i++) {
        cur_z += vz * dt;
        cur_y += vy * dt;
        cur_v = sqrt(pow(vz, 2) + pow(vy, 2));
        vz += -air_fraction_acc(cur_v) * vz / cur_v * dt;
        vy += -air_fraction_acc(cur_v) * vy / cur_v * dt - G * dt;
        fly_time += dt;
    }
    return std::make_tuple(cur_y, fly_time);
}

std::tuple<float, float> get_pitch_air_frac(float z, float y, float v) {
    float pitch = -1;
    float y_actual = y;
    float y_cur, fly_time;
    for (int i = 0; i <= 20; i++) {
        pitch = get_pitch(y, z, v);
        std::tie(y_cur, fly_time) = shoot_altitude(pitch, v, z);
        float err = y_cur - y_actual;
        if (abs(err) <= 1e-4) {
            break;
        }
        y -= err;
    }
    // 如果没有找到合适的pitch角度
    if (pitch == -1) {
        pitch = get_pitch(y, z, v);
        fly_time = z / (v * cos(pitch));
    }
    return std::make_tuple(r2d(pitch), fly_time);
}

std::tuple<float, float> get_pitch_yaw(float x, float y, float z, float speed) {
    float pitch, yaw, fly_time;
    std::tie(pitch, fly_time) = get_pitch_air_frac(y, z, speed);
    yaw = -yaw_period_correction(r2d(atan(x / y)));
    return std::make_tuple(pitch, yaw);
}

} // namespace trajectory