// 过滤几何形状不符合条件的装甲板

#include <autoaim_interfaces/msg/detection.hpp>

namespace check_armor {
using Detection = autoaim_interfaces::msg::Detection;
using Point32 = geometry_msgs::msg::Point32;

// 检查装甲板几何形状是否符合条件
bool check_armor_shape(const Detection& det) {
    constexpr auto get_length = [](const Point32& p1, const Point32& p2) {
        return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
    };
    constexpr auto get_angle = [](const Point32& p1, const Point32& p2) {
        // 防止除0（尽管概率很小，但出现了会拉着所有东西一起死）
        if (abs(p1.y - p2.y) <= 0.1f) [[unlikely]] {
            return 0.0f;
        }
        return static_cast<float>(atan((p1.x - p2.x) / (p1.y - p2.y)) * 180 / M_PI);
    };
    const float left_length = get_length(det.tl, det.bl);
    const float right_length = get_length(det.tr, det.br);
    const float left_angle = get_angle(det.tl, det.bl);
    const float right_angle = get_angle(det.tr, det.br);
    const float angle_diff = abs(right_angle - left_angle);
    const float length_diff = abs(left_length - right_length);
    // 同上，防止除0
    if (left_length <= 0.1f || right_length <= 0.1f) [[unlikely]] {
        return false;
    }
    const float length_diff_ratio = length_diff / std::max(left_length, right_length);
    return angle_diff < 2.5 && // 灯条角度差需小于允许的最大角差
        length_diff_ratio < 0.15; // 左右灯条长度差不可偏大
}
} // namespace check_armor