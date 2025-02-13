#pragma once

#include <cstdint>

class SendPacket {
public:
    uint8_t header = 0xB1;
    uint8_t length = 9;
    int16_t pitch_angle;
    int16_t yaw_angle;
    uint8_t shoot_flag;
    uint8_t fly_t_to_send;
    uint8_t checksum;
    int get_size() {
        return length;
    }
}; // __attribute__((packed))?