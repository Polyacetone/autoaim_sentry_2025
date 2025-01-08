#pragma once

#include <kf.hpp>
#include <geometry_msgs/msg/transform.hpp>

enum class TrackerStatus { CONVERGING, TRACKING, TEMP_LOST, LOST };

class Tracker {
public:
    void push(const geometry_msgs::msg::Transform& transform, const int armor_area);
    void update();
    cv::Point3f predict(const float prediction_time);

    TrackerStatus status = TrackerStatus::LOST;

private:
    const unsigned int MAX_LOST_FRAMES = 5;
    const unsigned int CONVERGE_FRAMES = 5;
    const geometry_msgs::msg::Transform EMPTY_ARMOR;

    geometry_msgs::msg::Transform target_armor_, another_armor_;
    unsigned target_armor_area_;
    unsigned int tracking_frames_;
    unsigned int lost_frames_;
    KF kf_;
};

void Tracker::push(const geometry_msgs::msg::Transform& transform, const int armor_area) {
    if (target_armor_ == EMPTY_ARMOR) {
        target_armor_ = transform;
        target_armor_area_ = armor_area;
    } else if (another_armor_ == EMPTY_ARMOR) {
        if (armor_area > target_armor_area_) {
            another_armor_ = target_armor_;
            target_armor_ = transform;
        } else {
            another_armor_ = transform;
        }
    }
}

void Tracker::update() {
    using TS = TrackerStatus;

    if (target_armor_ == EMPTY_ARMOR) {
        tracking_frames_ = 0;
        lost_frames_++;
        if (status != TS::LOST) {
            kf_.update();
        }
        if (lost_frames_ >= MAX_LOST_FRAMES) {
            status = TS::LOST;
        } else {
            status = TS::TEMP_LOST;
        }
        return;
    }

    lost_frames_ = 0;
    tracking_frames_++;
    const cv::Point3f armor_center(
        target_armor_.translation.x,
        target_armor_.translation.y,
        target_armor_.translation.z
    );

    if (status == TS::LOST) {
        kf_.initialize(armor_center);
    } else {
        kf_.update(armor_center);
    }
    if (tracking_frames_ >= CONVERGE_FRAMES) {
        status = TS::TRACKING;
    } else {
        status = TS::CONVERGING;
    }

    target_armor_ = EMPTY_ARMOR;
    another_armor_ = EMPTY_ARMOR;
}

cv::Point3f Tracker::predict(const float prediction_frames) {
    return kf_.position + kf_.velocity * prediction_frames;
}