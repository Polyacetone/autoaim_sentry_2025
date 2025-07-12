#pragma once

#include <stdexcept>

enum class ArmorColor: int {
    NONE = -1,
    BLUE = 0,
    RED = 1,
    GRAY = 2
};

enum class ArmorLabel: int {
    NONE = -1,
    SENTRY = 0,
    ONE = 1,
    TWO = 2,
    THREE = 3,
    FOUR = 4,
    OUTPOST = 5,
    BASE = 6
};

enum class AutoaimMode: int {
    NONE = -1,
    ARMOR = 0,
    BUFF = 1,
    DART = 2
};

namespace defs {

constexpr float armor_pitch(ArmorLabel t) {
    if (t == ArmorLabel::NONE) [[unlikely]] throw std::invalid_argument("invalid armor type: NONE");
    return (t == ArmorLabel::OUTPOST) ? -0.2618 : 0.2618;
}

constexpr bool is_armor_pitch_negative(ArmorLabel t) {
    if (t == ArmorLabel::NONE) [[unlikely]] throw std::invalid_argument("invalid armor type: NONE");
    return armor_pitch(t) < 0;
}

constexpr bool is_big_armor(ArmorLabel t) {
    if (t == ArmorLabel::NONE) [[unlikely]] throw std::invalid_argument("invalid armor type: NONE");
    return (t == ArmorLabel::ONE || t == ArmorLabel::BASE) ? true : false;
}

}