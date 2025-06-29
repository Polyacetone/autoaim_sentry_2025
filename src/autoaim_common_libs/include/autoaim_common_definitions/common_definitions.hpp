#pragma once

#include <stdexcept>

enum class ArmorColor {
    NONE = -1,
    BLUE = 0,
    RED = 1,
    GRAY = 2
};

enum class ArmorType {
    NONE = -1,
    SENTRY = 0,
    ONE = 1,
    TWO = 2,
    THREE = 3,
    FOUR = 4,
    OUTPOST = 5,
    BASE = 6
};

enum class AutoaimMode {
    NONE = -1,
    ARMOR = 0,
    BUFF = 1,
    DART = 2
};

namespace defs {

constexpr float armor_pitch(ArmorType t) {
    if (t == ArmorType::NONE) [[unlikely]] throw std::invalid_argument("invalid armor type: NONE");
    return (t == ArmorType::OUTPOST) ? -0.2618 : 0.2618;
}

constexpr bool is_armor_pitch_negative(ArmorType t) {
    if (t == ArmorType::NONE) [[unlikely]] throw std::invalid_argument("invalid armor type: NONE");
    return armor_pitch(t) < 0;
}

constexpr bool is_big_armor(ArmorType t) {
    if (t == ArmorType::NONE) [[unlikely]] throw std::invalid_argument("invalid armor type: NONE");
    return (t == ArmorType::ONE || t == ArmorType::BASE) ? true : false;
}

}