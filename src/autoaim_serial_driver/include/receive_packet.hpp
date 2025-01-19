#pragma once

#include <cstdint>
#include <string>

class ReceivePacket {
public:
    uint8_t header = 0x1B; // 0
    uint8_t length = 13; // 1
    int16_t pitch; // 2-3
    int16_t yaw; // 4-5
    int16_t roll; // 6-7
    int16_t big_yaw; // 8-9
    uint16_t shoot_speed; // 10-11
    uint8_t checksum; // 12
} __attribute__((packed));

class ReceivePacketInfantry {
public:
    uint8_t header1;
    uint8_t header2;
    uint8_t cmd;
    int16_t shoot_speed;
    int16_t roll;
    int16_t pitch;
    int16_t yaw;
    uint8_t target_color;
    uint8_t checksum;
} __attribute__((packed));