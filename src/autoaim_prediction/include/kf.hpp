#pragma once

#include <opencv2/opencv.hpp>

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