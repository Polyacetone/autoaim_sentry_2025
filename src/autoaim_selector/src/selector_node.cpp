#include <rclcpp/rclcpp.hpp>

#include <hw_sentry_interfaces/msg/detections.hpp>
#include <hw_sentry_interfaces/msg/robot_color.hpp>
#include <hw_sentry_interfaces/msg/enemy_priority.hpp>
#include <hw_sentry_interfaces/msg/comp_robots_hp.hpp>
#include <hw_sentry_interfaces/msg/target_enemy.hpp>

#include <autoaim_common_definitions/common_definitions.hpp>

namespace autoaim_selector {
using namespace hw_sentry_interfaces::msg;

class SelectorNode: public rclcpp::Node {
public:
    explicit SelectorNode(const rclcpp::NodeOptions& options);

private:
    void detections_callback(const Detections::SharedPtr msg);
    void select_armors(const std::vector<ArmorDetection>& src, std::vector<ArmorDetection>& dst) const;

    AutoaimMode mode_ = AutoaimMode::NONE;
    ArmorColor target_color_ = ArmorColor::NONE;
    ArmorType target_label_ = ArmorType::NONE;

    rclcpp::Subscription<Detections>::SharedPtr detections_sub_;
    rclcpp::Subscription<RobotColor>::SharedPtr robot_color_sub_;
    rclcpp::Subscription<TargetEnemy>::SharedPtr target_enemy_sub_;

    rclcpp::Publisher<Detections>::SharedPtr selected_detections_pub_;
};

SelectorNode::SelectorNode(const rclcpp::NodeOptions& options): Node("autoaim_selector", options) {
    mode_ = static_cast<AutoaimMode>(declare_parameter<int>("default_mode"));
    target_label_ = static_cast<ArmorType>(declare_parameter<int>("default_target_label"));
    target_color_ = static_cast<ArmorColor>(declare_parameter<int>("default_target_color"));

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
        [&](const RobotColor::SharedPtr msg) {
            const ArmorColor robot_color = static_cast<ArmorColor>(msg->robot_color);
            target_color_ = (robot_color == ArmorColor::BLUE) ? ArmorColor::RED : ArmorColor::BLUE;
        }
    );
    target_enemy_sub_ = create_subscription<TargetEnemy>(
        target_enemy_sub_topic,
        rclcpp::QoS(1),
        [&](const TargetEnemy::SharedPtr msg) {
            mode_ = static_cast<AutoaimMode>(msg->mode);
            target_label_ = static_cast<ArmorType>(msg->label);
        }
    );

    selected_detections_pub_ = create_publisher<Detections>(
        selected_detections_pub_topic,
        rclcpp::QoS(1)
    );
}

void SelectorNode::detections_callback(const Detections::SharedPtr msg) {
    if (static_cast<AutoaimMode>(msg->mode) != mode_) return;
    Detections selected_detections;
    selected_detections.header.stamp = msg->header.stamp;
    selected_detections.mode = static_cast<int>(mode_);
    selected_detections.label = static_cast<int>(target_label_);
    if (mode_ == AutoaimMode::ARMOR) {
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
    static int center_x_prev = 0;
    std::vector<ArmorDetection> filtered;
    // 筛选出目标颜色和标签的装甲板
    std::copy_if(src.begin(), src.end(), std::back_inserter(filtered),
        [&](const auto& armor) -> bool {
            return static_cast<ArmorType>(armor.label) == target_label_
                && (static_cast<ArmorColor>(armor.color) == target_color_
                    || (static_cast<ArmorColor>(armor.color) == ArmorColor::GRAY
                        && abs(get_center_x(armor) - center_x_prev) <= 15)); // 如果是灰色装甲板则根据和之前的位置差异判断是否是同一个
        }
    );
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