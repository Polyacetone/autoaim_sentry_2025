#pragma once

#include <stdexcept>

enum class ColorType: int {
    NONE = -1,
    BLUE = 0,
    RED = 1,
    GRAY = 2
};

enum class ArmorType: int {
    NONE = -1,
    SENTRY = 0,
    ONE = 1,
    TWO = 2,
    THREE = 3,
    FOUR = 4,
    OUTPOST = 5,
    BASE = 6
};

enum class BuffType: int {
    NONE = -1,
    INACTIVATE = 0,
    ACTIVATE = 1
};

enum class AutoaimMode: int {
    NONE = -1,
    ARMOR = 0,
    SMALL_BUFF = 1,
    BIG_BUFF = 2,
    DART = 3
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