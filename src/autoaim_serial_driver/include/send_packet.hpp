#pragma once

#include <cstdint>

class SendToController {
public:
    uint8_t header = 0xA2;
    uint8_t length = 9;
    int16_t pitch_angle;
    int16_t yaw_angle;
    uint8_t shoot_flag;
    uint8_t fly_t_to_send;
    uint8_t checksum;
} __attribute__((packed));