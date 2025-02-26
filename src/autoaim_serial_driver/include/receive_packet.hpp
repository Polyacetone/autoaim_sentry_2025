#pragma once

#include <cstdint>

class ReceiveFromImu {
public:
    uint8_t header = 0xB5;
    int16_t w;
    int16_t x;
    int16_t y;
    int16_t z;
    uint8_t checksum;
} __attribute__((packed));

class ReceiveFromController {
public:
    uint8_t header = 0x1B;
    uint8_t length = 11;
    int16_t pitch;
    int16_t yaw;
    int16_t big_yaw;
    int16_t shootspeed;
    uint8_t checksum;
} __attribute__((packed));

class ReceivePacketRfrCompRobotsHp {
public:
    uint8_t header = 0x03;
    uint8_t length = 35;
    /*/< 红 1 英雄机器人血量。若该机器人未上场或者被罚下，则血量为 0 */
    uint16_t red_1_robot_HP;
    /*/< 红 2 工程机器人血量。若该机器人未上场或者被罚下，则血量为 0 */
    uint16_t red_2_robot_HP;
    /*/< 红 3 步兵机器人血量。若该机器人未上场或者被罚下，则血量为 0 */
    uint16_t red_3_robot_HP;
    /*/< 红 4 步兵机器人血量。若该机器人未上场或者被罚下，则血量为 0 */
    uint16_t red_4_robot_HP;
    uint16_t reserved_1; ///< 保留位
    /*/< 红 7 哨兵机器人血量。若该机器人未上场或者被罚下，则血量为 0 */
    uint16_t red_7_robot_HP;
    uint16_t red_outpost_HP; ///< 红方前哨站血量
    uint16_t red_base_HP; ///< 红方基地血量
    /*/< 蓝 1 英雄机器人血量。若该机器人未上场或者被罚下，则血量为 0 */
    uint16_t blue_1_robot_HP;
    /*/< 蓝 2 工程机器人血量。若该机器人未上场或者被罚下，则血量为 0 */
    uint16_t blue_2_robot_HP;
    /*/< 蓝 3 步兵机器人血量。若该机器人未上场或者被罚下，则血量为 0 */
    uint16_t blue_3_robot_HP;
    /*/< 蓝 4 步兵机器人血量。若该机器人未上场或者被罚下，则血量为 0 */
    uint16_t blue_4_robot_HP;
    uint16_t reserved_2; ///< 保留位
    /*/< 蓝 7 哨兵机器人血量。若该机器人未上场或者被罚下，则血量为 0 */
    uint16_t blue_7_robot_HP;
    uint16_t blue_outpost_HP; ///< 蓝方前哨站血量
    uint16_t blue_base_HP; ///< 蓝方基地血量
    uint8_t checksum;
} __attribute__((packed));

class ReceivePacketRfrCompStatus {
public:
    uint8_t header = 0x01;
    uint8_t length = 6;
    uint8_t game_progress; ///< 比赛进程
    uint16_t stage_remain_time; ///< 当前阶段剩余时间
    uint8_t checksum;
} __attribute__((packed));

class ReceivePacketRfrInterMapClientToRobot {
public:
    uint8_t header = 0x33;
    uint8_t length = 15;
    float target_position_x;
    float target_position_y;
    uint8_t cmd_keyboard;
    uint8_t target_robot_id;
    uint16_t cmd_source;
    uint8_t checksum;
} __attribute__((packed));

class ReceivePacketRfrRobotGroundPos {
public:
    uint8_t header = 0x2B;
    uint8_t length = 43;
    float hero_x;
    float hero_y;
    float engineer_x;
    float engineer_y;
    float standard_3_x;
    float standard_3_y;
    float standard_4_x;
    float standard_4_y;
    float reserved_1;
    float reserved_2;
    uint8_t checksum;
} __attribute__((packed));

class ReceivePacketRfrRobotPerformance {
public:
    uint8_t header = 0x21;
    uint8_t length = 6;
    uint16_t current_hp;
    uint8_t power_management_shooter_output;
    uint8_t checksum;
} __attribute__((packed));

class ReceivePacketRfrRobotPos {
public:
    uint8_t header = 0x23;
    uint8_t length = 15;
    float x;
    float y;
    float angle;
    uint8_t checksum;
} __attribute__((packed));

class ReceivePacketRfrRobotResource {
public:
    uint8_t header = 0x28;
    uint8_t length = 6;
    uint16_t allowance_17mm;
    uint16_t remaining_coin;
    uint8_t checksum;
} __attribute__((packed));

class ReceivePacketRfrRobotRfid {
public:
    uint8_t header = 0x29;
    uint8_t length = 7;
    uint8_t our_base : 1; ///< bit 0: 己方基地增益点
    uint8_t our_central_highland : 1; ///< bit 1: 己方中央高地增益点
    uint8_t opp_central_highland : 1; ///< bit 2: 对方中央高地增益点
    uint8_t our_trapezoid_highland : 1; ///< bit 3: 己方梯形高地增益点
    uint8_t opp_trapezoid_highland : 1; ///< bit 4: 对方梯形高地增益点
    uint8_t our_launch_front : 1; ///< bit 5: 己方飞坡增益点（靠近己方一侧飞坡前）
    uint8_t our_launch_back : 1; ///< bit 6: 己方飞坡增益点（靠近己方一侧飞坡后）
    uint8_t opp_launch_front : 1; ///< bit 7: 对方飞坡增益点（靠近对方一侧飞坡前）
    uint8_t opp_launch_back : 1; ///< bit 8: 对方飞坡增益点（靠近对方一侧飞坡后）
    uint8_t our_highland_bottom : 1; ///< bit 9: 己方地形跨越增益点（中央高地下方）
    uint8_t our_highland_top : 1; ///< bit 10: 己方地形跨越增益点（中央高地上方）
    uint8_t opp_highland_bottom : 1; ///< bit 11: 对方地形跨越增益点（中央高地下方）
    uint8_t opp_highland_top : 1; ///< bit 12: 对方地形跨越增益点（中央高地上方）
    uint8_t our_highway_bottom : 1; ///< bit 13: 己方地形跨越增益点（公路下方）
    uint8_t our_highway_top : 1; ///< bit 14: 己方地形跨越增益点（公路上方）
    uint8_t opp_highway_bottom : 1; ///< bit 15: 对方地形跨越增益点（公路下方）
    uint8_t opp_highway_top : 1; ///< bit 16: 对方地形跨越增益点（公路上方）
    uint8_t our_fort : 1; ///< bit 17: 己方堡垒增益点
    uint8_t our_outpost : 1; ///< bit 18: 己方前哨站增益点
    uint8_t our_restoration_1 : 1; ///< bit 19: 己方与兑换区不重叠的补给区/RMUL 补给区
    uint8_t our_restoration_2 : 1; ///< bit 20: 己方与兑换区重叠的补给区
    uint8_t our_big_resource_island : 1; ///< bit 21: 己方大资源岛增益点
    uint8_t opp_big_resource_island : 1; ///< bit 22: 对方大资源岛增益点
    uint8_t central_boost : 1; ///< bit 23: 中心增益点 @attention 仅RMUL适用
    uint8_t reserved; ///< 保留区域
    uint8_t checksum;
} __attribute__((packed));

class ReceivePacketRfrRobotSentryDecision {
public:
    uint8_t header = 0x2D;
    uint8_t length = 9;
    uint16_t allowance : 11; ///< 除远程兑换外，哨兵成功兑换的发弹量
    uint8_t remote_allowance : 4; ///< 哨兵成功远程兑换发弹量的次数
    uint8_t remote_hp : 4; ///< 哨兵成功远程兑换血量的次数
    uint8_t allow_free_resurrection : 1; ///< 哨兵当前是否可以确认免费复活
    uint8_t allow_redemption_resurrection : 1; ///< 哨兵当前是否可以兑换立即复活
    uint16_t redemption_resurrection_cost : 10; ///< 哨兵当前兑换立即复活需要花费的金币数
    uint8_t reserved1 : 1;
    uint8_t out_of_war_status : 1; ///< 哨兵当前是否处于脱战状态
    uint16_t remain_total_allowance : 11; ///< 队伍17mm允许发弹量的剩余可兑换数
    uint8_t reserved2 : 4;
    uint8_t checksum;
} __attribute__((packed));

class ReceivePacketRfrTeamEvent {
public:
    uint8_t header = 0x11;
    uint8_t length = 7;
    uint8_t restoration_1 : 1;
    uint8_t restoration_2 : 1;
    uint8_t supplier_area : 1;
    uint8_t small_power_rune : 1;
    uint8_t large_power_rune : 1;
    uint8_t central_highland : 2;
    uint8_t trapezoid_highland : 2;
    uint16_t dart_last_hit_us_time : 9;
    uint8_t dart_last_hit_us_type : 3;
    uint8_t center_buff_area : 2;
    uint16_t reserved : 9;
    uint8_t checksum;
} __attribute__((packed));
