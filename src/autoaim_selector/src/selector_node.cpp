#include <rclcpp/rclcpp.hpp>

#include <hw_sentry_interfaces/msg/detections.hpp>
#include <hw_sentry_interfaces/msg/robot_color.hpp>
#include <hw_sentry_interfaces/msg/target_enemy.hpp>
#include <hw_sentry_interfaces/msg/comp_robots_hp.hpp>

#include <autoaim_common_definitions/common_definitions.hpp>

namespace autoaim_selector {
using namespace hw_sentry_interfaces::msg;

class SelectorNode: public rclcpp::Node {
public:
    explicit SelectorNode(const rclcpp::NodeOptions& options);

private:
    void detections_callback(const Detections::SharedPtr msg);
    void comp_robots_hp_callback(const CompRobotsHp::SharedPtr msg);
    void decide_target_label(const std::vector<ArmorDetection>& armors);
    void select_armors(
        const std::vector<ArmorDetection>& src,
        std::vector<ArmorDetection>& dst
    ) const;

    AutoaimMode default_mode_;
    std::vector<ArmorLabel> default_target_label_priority_;
    float decision_fallback_timeout_;
    unsigned current_target_max_lost_frames_, switch_new_target_appear_frames_;
    float big_armor_length_height_threshold_, small_armor_length_height_threshold_;
    
    ArmorLabel target_label_ = ArmorLabel::NONE;
    ArmorColor target_color_;
    AutoaimMode mode_;
    std::vector<ArmorLabel> target_label_priority_;
    double last_recv_decision_time_ = 0;

    unsigned current_target_lost_frames_ = 0;
    unsigned armor_appear_frames_[5][15] = {};
    bool is_enemy_dead_[15] = {};
    bool is_enemy_invincible_[15] = {};
    double invincible_start_time_[15] = {};

    rclcpp::Subscription<Detections>::SharedPtr detections_sub_;
    rclcpp::Subscription<RobotColor>::SharedPtr robot_color_sub_;
    rclcpp::Subscription<TargetEnemy>::SharedPtr target_enemy_sub_;
    rclcpp::Subscription<CompRobotsHp>::SharedPtr comp_robots_hp_sub_;
    rclcpp::Publisher<Detections>::SharedPtr selected_detections_pub_;
};

SelectorNode::SelectorNode(const rclcpp::NodeOptions& options): Node("autoaim_selector", options) {
    default_mode_ = static_cast<AutoaimMode>(declare_parameter<int>("default_mode"));
    mode_ = default_mode_;
    auto default_target_label_priority = declare_parameter<std::vector<int>>("default_target_label_priority");
    std::transform(
        default_target_label_priority.begin(), default_target_label_priority.end(),
        std::back_inserter(default_target_label_priority_),
        [](const auto& v) { return static_cast<ArmorLabel>(v); }
    );
    target_label_priority_ = default_target_label_priority_;
    target_color_ = static_cast<ArmorColor>(declare_parameter<int>("default_target_color"));
    decision_fallback_timeout_ = declare_parameter<float>("decision_fallback_timeout");
    current_target_max_lost_frames_ = declare_parameter<int>("current_target_max_lost_frames");
    switch_new_target_appear_frames_ = declare_parameter<int>("switch_new_target_appear_frames");
    big_armor_length_height_threshold_ = declare_parameter<float>("big_armor_length_height_threshold");
    small_armor_length_height_threshold_ = declare_parameter<float>("small_armor_length_height_threshold");
    last_recv_decision_time_ = now().seconds();

    std::string detections_sub_topic = declare_parameter<std::string>("detections_topic");
    std::string target_enemy_sub_topic = declare_parameter<std::string>("target_enemy_topic");
    std::string robot_color_sub_topic = declare_parameter<std::string>("robot_color_topic");
    std::string comp_robots_hp_topic = declare_parameter<std::string>("comp_robots_hp_topic");
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
            target_label_priority_.clear();
            std::transform(
                msg->label.begin(), msg->label.end(),
                std::back_inserter(target_label_priority_),
                [](const auto& v) { return static_cast<ArmorLabel>(v); }
            );
            last_recv_decision_time_ = now().seconds();
        }
    );
    comp_robots_hp_sub_ = create_subscription<CompRobotsHp>(
        comp_robots_hp_topic,
        rclcpp::QoS(1),
        [&](const CompRobotsHp::SharedPtr msg) { comp_robots_hp_callback(msg); }
    );
    selected_detections_pub_ = create_publisher<Detections>(
        selected_detections_pub_topic,
        rclcpp::QoS(1)
    );
}

void SelectorNode::detections_callback(const Detections::SharedPtr msg) {
    if (now().seconds() - last_recv_decision_time_ > decision_fallback_timeout_) {
        mode_ = default_mode_;
        target_label_priority_ = default_target_label_priority_;
        RCLCPP_WARN(get_logger(), "EnemyPosition not received for a while, fallback to default mode");
    }
    if (static_cast<AutoaimMode>(msg->mode) != mode_) return;
    Detections selected_detections;
    selected_detections.header.stamp = msg->header.stamp;
    selected_detections.mode = static_cast<int>(mode_);
    selected_detections.label = -1;
    if (mode_ == AutoaimMode::ARMOR) {
        decide_target_label(msg->armor_detections);
        selected_detections.label = static_cast<int>(target_label_);
        select_armors(msg->armor_detections, selected_detections.armor_detections);
        selected_detections_pub_->publish(selected_detections);
    }
}

void SelectorNode::decide_target_label(const std::vector<ArmorDetection>& armors) {
    bool occurred_armors[5][15] = {};
    if (!armors.empty()) {
        for (const auto& armor: armors) {
            occurred_armors[armor.color][armor.label] = true;
        }
    }
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 15; j++) {
            if (occurred_armors[i][j]) {
                armor_appear_frames_[i][j]++;
            } else {
                armor_appear_frames_[i][j] = 0;
            }
        }
    }
    if (target_label_ != ArmorLabel::NONE) {
        if (!occurred_armors[static_cast<int>(target_color_)][static_cast<int>(target_label_)]) {
            current_target_lost_frames_++;
        } else {
            current_target_lost_frames_ = 0;
        }
    }
    if (!target_label_priority_.empty()) {
        for (const ArmorLabel label: target_label_priority_) {
            if (label == ArmorLabel::NONE) continue;
            if (label == target_label_) {
                if (!is_enemy_dead_[static_cast<int>(label)] &&
                    !is_enemy_invincible_[static_cast<int>(label)] &&
                    current_target_lost_frames_ <= current_target_max_lost_frames_) {
                    return;
                }
            } else {
                if (!is_enemy_dead_[static_cast<int>(label)] &&
                    !is_enemy_invincible_[static_cast<int>(label)] &&
                    armor_appear_frames_[static_cast<int>(target_color_)][static_cast<int>(label)] > switch_new_target_appear_frames_) {
                    target_label_ = label;
                    current_target_lost_frames_ = 0;
                    return;
                }
            }
        }
    }
    target_label_ = ArmorLabel::NONE;
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
        const float length = std::hypot(
            (d.tr.x + d.br.x) / 2 - (d.tl.x + d.bl.x) / 2,
            (d.tr.y + d.br.y) / 2 - (d.tl.y + d.bl.y) / 2
        );
        const float height = (
            std::hypot(d.tl.x - d.bl.x, d.tl.y - d.bl.y)
            + std::hypot(d.tr.x - d.br.x, d.tr.y - d.br.y)
        ) / 2;
        return length / height;
    };
    static int center_x_prev = 0;
    std::vector<ArmorDetection> filtered;
    // 筛选出目标颜色和标签的装甲板
    std::copy_if(src.begin(), src.end(), std::back_inserter(filtered),
        [&](const auto& armor) -> bool {
            return static_cast<ArmorLabel>(armor.label) == target_label_ // 编号一样
                && get_length_height_ratio(armor) > (defs::is_big_armor(target_label_) // 过滤太斜的
                    ? big_armor_length_height_threshold_
                    : small_armor_length_height_threshold_)
                && (static_cast<ArmorColor>(armor.color) == target_color_ // 颜色一样
                    || (static_cast<ArmorColor>(armor.color) == ArmorColor::GRAY // 如果是被打成灰色的，即使颜色不一样也能进
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
        std::sort(filtered.begin(), filtered.end(), 
            [&](const ArmorDetection& a, const ArmorDetection& b) {
                if (center_x_prev == 0) {
                    return get_area(a) > get_area(b);
                } else {
                    return get_area(a) - abs(get_center_x(a) - center_x_prev)
                        > get_area(b) - abs(get_center_x(b) - center_x_prev);
                }
            }
        );
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

void SelectorNode::comp_robots_hp_callback(const CompRobotsHp::SharedPtr msg) {
    constexpr float INVINCIBLE_TIME = 10;
    int enemy_hp[5] = {};
    if (target_color_ == ArmorColor::RED) {
        enemy_hp[0] = msg->red_7_robot_hp;
        enemy_hp[1] = msg->red_1_robot_hp;
        enemy_hp[2] = msg->red_2_robot_hp;
        enemy_hp[3] = msg->red_3_robot_hp;
        enemy_hp[4] = msg->red_4_robot_hp;
    } else if (target_color_ == ArmorColor::BLUE) {
        enemy_hp[0] = msg->blue_7_robot_hp;
        enemy_hp[1] = msg->blue_1_robot_hp;
        enemy_hp[2] = msg->blue_2_robot_hp;
        enemy_hp[3] = msg->blue_3_robot_hp;
        enemy_hp[4] = msg->blue_4_robot_hp;
    } else return;
    for (int i = 0; i < 5; i++) {
        if (enemy_hp[i] <= 0) {
            is_enemy_dead_[i] = true;
        }
        if (is_enemy_dead_[i] && enemy_hp[i] > 0) {
            is_enemy_dead_[i] = false;
            is_enemy_invincible_[i] = true;
            invincible_start_time_[i] = now().seconds();
        }
        if (is_enemy_invincible_[i]) {
            if (now().seconds() - invincible_start_time_[i] < INVINCIBLE_TIME) {
                is_enemy_invincible_[i] = true;
            } else {
                is_enemy_invincible_[i] = false;
            }
        }
    }
}
} // namespace autoaim_selector

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_selector::SelectorNode)