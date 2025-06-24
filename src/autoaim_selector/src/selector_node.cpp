#include <rclcpp/rclcpp.hpp>

#include <hw_sentry_interfaces/msg/detections.hpp>
#include <hw_sentry_interfaces/msg/robot_color.hpp>
#include <hw_sentry_interfaces/msg/enemy_priority.hpp>
#include <hw_sentry_interfaces/msg/comp_robots_hp.hpp>
#include <hw_sentry_interfaces/msg/target_enemy.hpp>

namespace autoaim_selector {
using namespace hw_sentry_interfaces::msg;

class SelectorNode: public rclcpp::Node {
public:
    explicit SelectorNode(const rclcpp::NodeOptions& options);
    ~SelectorNode() = default;

private:
    void detections_callback(const Detections::SharedPtr msg);
    void select_armors(const std::vector<ArmorDetection>& src, std::vector<ArmorDetection>& dst) const;

    bool is_big_armor(int label) const {
        return (label == 1);
    }

    int mode_ = -1;
    int target_color_ = -1;
    int target_label_ = -1;

    rclcpp::Subscription<Detections>::SharedPtr detections_sub_;
    rclcpp::Subscription<RobotColor>::SharedPtr robot_color_sub_;
    rclcpp::Subscription<TargetEnemy>::SharedPtr target_enemy_sub_;

    rclcpp::Publisher<Detections>::SharedPtr selected_detections_pub_;
};

SelectorNode::SelectorNode(const rclcpp::NodeOptions& options): Node("autoaim_selector", options) {
    mode_ = declare_parameter<int>("default_mode");
    target_label_ = declare_parameter<int>("default_target_label");
    target_color_ = declare_parameter<int>("default_target_color");

    std::string detections_sub_topic = declare_parameter<std::string>("detections_topic");
    std::string target_enemy_sub_topic = declare_parameter<std::string>("target_enemy_topic");
    std::string robot_color_sub_topic = declare_parameter<std::string>("robot_color_topic");
    std::string selected_detections_pub_topic = declare_parameter<std::string>("selected_detection_topic");

    detections_sub_ = create_subscription<Detections>(
        detections_sub_topic,
        rclcpp::QoS(1),
        [&](const Detections::SharedPtr msg) { detections_callback(msg); }
    );
    robot_color_sub_ = create_subscription<RobotColor>(
        robot_color_sub_topic,
        rclcpp::QoS(1),
        [&](const RobotColor::SharedPtr msg) { target_color_ = 1 - msg->robot_color; }
    );
    target_enemy_sub_ = create_subscription<TargetEnemy>(
        target_enemy_sub_topic,
        rclcpp::QoS(1),
        [&](const TargetEnemy::SharedPtr msg) {
            mode_ = msg->mode;
            target_label_ = msg->label;
        }
    );

    selected_detections_pub_ = create_publisher<Detections>(
        selected_detections_pub_topic,
        rclcpp::QoS(1)
    );
}

void SelectorNode::detections_callback(const Detections::SharedPtr msg) {
    if (msg->mode != mode_) return;
    Detections selected_detections;
    selected_detections.header.stamp = msg->header.stamp;
    selected_detections.mode = mode_;
    selected_detections.label = target_label_;
    if (msg->mode == 0) {
        select_armors(msg->armor_detections, selected_detections.armor_detections);
        selected_detections_pub_->publish(selected_detections);
    }
}

void SelectorNode::select_armors(
    const std::vector<ArmorDetection>& src,
    std::vector<ArmorDetection>& dst
) const {
    constexpr auto get_center_x = [](const ArmorDetection& d) -> int {
        return (d.bl.x + d.br.x + d.tr.x + d.tl.x) / 4;
    };
    constexpr auto get_area = [](const ArmorDetection& d) -> int {
        return (d.br.x - d.tl.x) * (d.br.y - d.tl.y);
    };
    constexpr auto get_length_height_ratio = [](const ArmorDetection& d) -> float {
        return fabs((d.tl.x + d.bl.x - d.tr.x - d.br.x) / (d.tl.y + d.tr.y - d.bl.y - d.br.y));
    };
    static int center_x_prev = 0;
    std::vector<ArmorDetection> filtered;
    // 筛选出目标颜色和标签的装甲板
    for (const auto& armor: src) {
        // 用装甲板长宽比筛掉太斜的装甲板
        const bool yaw_too_large = get_length_height_ratio(armor) < (is_big_armor(armor.label) ? 2.0 : 1.6);
        if (!yaw_too_large && armor.label == target_label_) {
            if (target_color_ == armor.color) {
                filtered.emplace_back(armor);
            } else if (armor.color == 2) { // 特殊处理灰色装甲板
                if (abs(get_center_x(armor) - center_x_prev) <= 15) {
                    // 这里只根据灰色装甲板位置与上次瞄的位置差判断是否是被打成灰的
                    filtered.emplace_back(armor);
                }
            }
        }
    }
    if (filtered.empty()) {
        center_x_prev = 0;
        return;
    }

    // 对目标装甲板进行排序
    if (filtered.size() == 1) {
        dst.push_back(filtered[0]);
    } else {
        // 根据击打面积和装甲板位置与正在瞄准位置间的差异排序
        std::sort(filtered.begin(), filtered.end(), [&](const ArmorDetection& a, const ArmorDetection& b) {
            if (center_x_prev == 0) {
                return get_area(a) > get_area(b);
            } else {
                return get_area(a) - abs(get_center_x(a) - center_x_prev)
                    > get_area(b) - abs(get_center_x(b) - center_x_prev);
            }
        });
        dst.emplace_back(filtered[0]);
        // 接下来选择击打面积次之，且和原来那个位置有较大差异的装甲板。
        // 虽然理论上detection中的nms已经能去除同一个装甲板的多个识别结果，但有时候还是会出现。
        const int len = filtered.size();
        const int center_x_first = get_center_x(filtered[0]);
        for (int i = 1; i < len; i++) {
            const int center_x_i = get_center_x(filtered[i]);
            if (abs(center_x_first - center_x_i) >= 15) {
                dst.push_back(filtered[i]);
                break;
            }
        }
    }
    center_x_prev = get_center_x(dst[0]);
}
} // namespace autoaim_selector

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_selector::SelectorNode)