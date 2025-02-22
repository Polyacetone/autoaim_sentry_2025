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