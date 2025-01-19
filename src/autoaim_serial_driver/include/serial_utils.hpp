#pragma once

#include <vector>
#include <cstdint>

namespace serial_utils {

template<typename T>
T from_vector(const std::vector<uint8_t>& data) {
    T packet;
    std::copy(data.begin(), data.end(), reinterpret_cast<uint8_t*>(&packet));
    return packet;
}

template<typename T>
std::vector<uint8_t> to_vector(const T& packet) {
    return std::vector<uint8_t>(
        reinterpret_cast<const uint8_t*>(&packet),
        reinterpret_cast<const uint8_t*>(&packet) + sizeof(packet)
    );
}

int16_t swap_bytes_of_int16(int16_t x) {
    return ((x & 0x00ff) << 8) + ((x & 0xff00) >> 8);
}

} // namespace serial_utils