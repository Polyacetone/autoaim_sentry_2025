#pragma once

#include <cstdint>
#include <string>

class ReceivePacketImu {
public:
    uint8_t header = 0xB5; // 0
    int16_t w;
    int16_t x;
    int16_t y;
    int16_t z;
    uint8_t checksum;
} __attribute__((packed));
