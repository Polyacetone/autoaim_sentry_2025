#pragma once

#include <opencv2/opencv.hpp>

class PnPSolver {
public:
    cv::Point3f get_translation(const std::vector<cv::Point2f>& img_points) const {
        std::vector<cv::Mat> rvecs, tvecs;
        const int solutions = cv::solvePnPGeneric(
            WORLD_POINTS,
            img_points,
            cam_intrinsic_,
            cam_distortion_,
            rvecs,
            tvecs,
            false,
            cv::SOLVEPNP_IPPE
        );
        cv::Point3f solution;
        solution.x = tvecs[0].at<double>(2);
        solution.y = -tvecs[0].at<double>(0);
        solution.z = -tvecs[0].at<double>(1);
        return solution;
    }

    /*!
        @brief 设置相机的内参矩阵和畸变矩阵
        @attention 算PnP前一定要先设置这个
    */
    void set_cam_matrix(const cv::Mat intrinsic, const cv::Mat distortion) {
        cam_intrinsic_ = intrinsic.clone();
        cam_distortion_ = distortion.clone();
    }

    PnPSolver() = default;
    ~PnPSolver() = default;

    // 单位: 米
    static constexpr float RADIUS = 0.055 / 2;
    // 关键点：上，左，下，右；目标坐标系：前x，左y，上z
    const std::vector<cv::Point3f> WORLD_POINTS {
        {0, 0, RADIUS},
        {0, RADIUS, 0},
        {0, 0, -RADIUS},
        {0, -RADIUS, 0}
    };

    cv::Mat cam_intrinsic_ =
        (cv::Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    cv::Mat cam_distortion_ = (cv::Mat_<double>(1, 5) << 0.0, 0.0, 0.0, 0.0, 0.0);
};