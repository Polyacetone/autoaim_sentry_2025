#include <trajectory.hpp>

namespace trajectory {

std::tuple<float, float> shoot_altitude(float theta, float v, float d) {
    constexpr auto air_fraction_acc = [](float v) -> float {
        return COEFF * v * v;
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
    return {cur_h, fly_time};
}

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
        if (fabs(err) <= 1e-4) {
            break;
        }
        h -= err;
    }
    // 如果没有找到合适的pitch角度
    if (pitch == -1) {
        pitch = get_pitch(d, h, v);
        fly_time = d / (v * cos(pitch));
    }
    return {pitch, fly_time};
}

} // namespace trajectory