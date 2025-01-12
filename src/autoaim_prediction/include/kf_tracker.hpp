#pragma once

#include <geometry_msgs/msg/transform.hpp>
#include <opencv2/opencv.hpp>

namespace kf_tracker {

enum class TRACKER_STATUS { CONVERGING, TRACKING, TEMP_LOST, LOST };

class KF {
public:
    KF() {
        cvkf_.transitionMatrix = (cv::Mat_<float>(6, 6) <<
            1, 0, 0, 1, 0, 0,
            0, 1, 0, 0, 1, 0,
            0, 0, 1, 0, 0, 1,
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1);
        cv::setIdentity(cvkf_.measurementMatrix);
        cv::setIdentity(cvkf_.processNoiseCov, cv::Scalar::all(1e-4));
        cv::setIdentity(cvkf_.measurementNoiseCov, cv::Scalar::all(5e-2));
        cv::setIdentity(cvkf_.errorCovPost, cv::Scalar::all(1e-1));
    }

    void initialize(const cv::Point3f& point) {
        cvkf_.statePost.at<float>(0) = point.x;
        cvkf_.statePost.at<float>(1) = point.y;
        cvkf_.statePost.at<float>(2) = point.z;
        cvkf_.statePost.at<float>(3) = 0;
        cvkf_.statePost.at<float>(4) = 0;
        cvkf_.statePost.at<float>(5) = 0;
    }

    void update(const cv::Point3f& point) {
        cvkf_.predict();
        cv::Mat estimated = cvkf_.correct((cv::Mat_<float>(3, 1) << point.x, point.y, point.z));
        position.x = estimated.at<float>(0);
        position.y = estimated.at<float>(1);
        position.z = estimated.at<float>(2);
        velocity.x = estimated.at<float>(3);
        velocity.y = estimated.at<float>(4);
        velocity.z = estimated.at<float>(5);
    }

    void update() {
        cv::Mat predicted = cvkf_.predict();
        position.x = predicted.at<float>(0);
        position.y = predicted.at<float>(1);
        position.z = predicted.at<float>(2);
    }

    cv::Point3f position, velocity;

private:
    cv::KalmanFilter cvkf_ = cv::KalmanFilter(6, 3, 0, CV_32F);
};

class Tracker {
public:
    void push(const geometry_msgs::msg::Transform& transform, const int armor_area);
    void update();
    cv::Point3f predict(const float prediction_time);

    TRACKER_STATUS tracker_status = TRACKER_STATUS::LOST;

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
    } else {
        // 识别到了多于2个同类装甲板
        // TODO...
    }
}

void Tracker::update() {
    using TS = TRACKER_STATUS;

    if (target_armor_ == EMPTY_ARMOR) {
        tracking_frames_ = 0;
        lost_frames_++;
        if (tracker_status != TS::LOST) {
            kf_.update();
        }
        if (lost_frames_ >= MAX_LOST_FRAMES) {
            tracker_status = TS::LOST;
        } else {
            tracker_status = TS::TEMP_LOST;
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

    if (tracker_status == TS::LOST) {
        kf_.initialize(armor_center);
    } else {
        kf_.update(armor_center);
    }
    if (tracking_frames_ >= CONVERGE_FRAMES) {
        tracker_status = TS::TRACKING;
    } else {
        tracker_status = TS::CONVERGING;
    }

    target_armor_ = EMPTY_ARMOR;
    another_armor_ = EMPTY_ARMOR;
}

cv::Point3f Tracker::predict(const float prediction_frames) {
    return kf_.position + kf_.velocity * prediction_frames;
}

}