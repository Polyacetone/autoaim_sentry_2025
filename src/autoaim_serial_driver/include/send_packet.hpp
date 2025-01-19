#pragma once

#include <cstdint>
#include <autoaim_interfaces/msg/comm_send.hpp>

#include <serial_utils.hpp>

class SendPacket {
public:
    uint8_t header = 0xB1;
    uint8_t length = 9;
    int16_t pitch_angle;
    int16_t yaw_angle;
    uint8_t shoot_flag;
    uint8_t fly_t_to_send;
    uint8_t checksum;
    void from_msg(const autoaim_interfaces::msg::CommSend::SharedPtr& msg) {
        pitch_angle = static_cast<int16_t>(msg->pitch * 100);
        yaw_angle = static_cast<int16_t>(msg->yaw * 100);
        shoot_flag = msg->shoot_flag;
        fly_t_to_send = msg->fly_t_to_send;
    }
    int get_size() {
        return length;
    }
}; // __attribute__((packed))?

class SendPacketInfantry {
public:
    uint8_t header1 = 0xaa;
    uint8_t header2 = 0xbb;
    uint8_t header3 = 0xcc;
    int16_t pitch_angle;
    int16_t yaw_angle;
    uint8_t target_num;
    uint8_t shoot_flag;
    uint8_t vtm_x;
    uint8_t vtm_y;
    uint8_t checksum;
    uint8_t tail = 0xff;
    void from_msg(const autoaim_interfaces::msg::CommSend::SharedPtr& msg) {
        pitch_angle = serial_utils::swap_bytes_of_int16(static_cast<int16_t>(msg->pitch * 100));
        yaw_angle = serial_utils::swap_bytes_of_int16(static_cast<int16_t>(msg->yaw * 100));
        checksum = ((pitch_angle&0xff00)>>8) ^ (pitch_angle&0x00ff) ^ 
            ((yaw_angle&0xff00)>>8) ^ (yaw_angle&0x00ff) ^ target_num ^
            shoot_flag ^ vtm_x ^ vtm_y;
    }
    int get_size() {
        return 13;
    }
} __attribute__((packed));