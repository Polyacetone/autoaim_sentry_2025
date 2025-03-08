#pragma once

#include <tf2/convert.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <opencv2/opencv.hpp>

#include <hw_sentry_interfaces/msg/detection.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

class PnPSolver {
public:
    /*!
        @brief 使用IPPE解PnP获取位移translation
        @param detection 输入的4个角点位置，以及装甲板标签（用于判断装甲板大小）。
        @param transform 输出的装甲板坐标系到相机坐标系的变换（只写入translation，实际上就是装甲板中心在相机系下的位置）。
        @return 返回1表示找解的时候出现问题（不过这种情况好像不常出现？目前好像没处理）。
        @note 相机坐标系和装甲板坐标系方向都是向前x，向左y，向上z。
    */
    bool get_translation(
        const hw_sentry_interfaces::msg::Detection& detection,
        geometry_msgs::msg::Transform& transform
    ) const {
        const auto& world_points = detection.label == 1 ? BIG_POINTS : SMALL_POINTS;
        const std::vector<cv::Point2d> img_points {
            {detection.tl.x, detection.tl.y},
            {detection.bl.x, detection.bl.y},
            {detection.br.x, detection.br.y},
            {detection.tr.x, detection.tr.y}
        };
        std::vector<cv::Mat> rvecs, tvecs;
        const int solutions = cv::solvePnPGeneric(
            world_points,
            img_points,
            cam_intrinsic_,
            cam_distortion_,
            rvecs,
            tvecs,
            false,
            cv::SOLVEPNP_IPPE
        );
        // IPPE返回两组对称解，这里筛选z大于0（opencv系下的z，就是在相机前面）的返回
        int solution_index = 0;
        if (solutions >= 1 && tvecs[0].at<double>(2) > 0) {
            solution_index = 0;
        } else if (solutions >= 2 && tvecs[1].at<double>(2) > 0) {
            solution_index = 1;
        } else {
            return 1;
        }
        // opencv的solvePnP认为相机系是向右x，向下y，向前z。
        // 我们（在tf2中发布的）认为相机系是向前x，向左y，向上z。
        // 所以我们的(x, y, z)对应opencv的(z, -x, -y)。
        // PnP解出的旋转没有使用，因为后面用三分法算
        transform.translation.x = tvecs[solution_index].at<double>(2);
        transform.translation.y = -tvecs[solution_index].at<double>(0);
        transform.translation.z = -tvecs[solution_index].at<double>(1);
        return 0;
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
    static constexpr float HEIGHT = 0.055;
    static constexpr float BIG_WIDTH = 0.2253;
    static constexpr float SMALL_WIDTH = 0.135;
    // 装甲板坐标系向右是x，向前是y，向上是z。
    const std::vector<cv::Point3f> SMALL_POINTS {
        {0, SMALL_WIDTH / 2, HEIGHT / 2},
        {0, SMALL_WIDTH / 2, -HEIGHT / 2},
        {0, -SMALL_WIDTH / 2, -HEIGHT / 2},
        {0, -SMALL_WIDTH / 2, HEIGHT / 2}
    };
    const std::vector<cv::Point3f> BIG_POINTS {
        {0, BIG_WIDTH / 2, HEIGHT / 2},
        {0, BIG_WIDTH / 2, -HEIGHT / 2},
        {0, -BIG_WIDTH / 2, -HEIGHT / 2},
        {0, -BIG_WIDTH / 2, HEIGHT / 2}
    };

    cv::Mat cam_intrinsic_ =
        (cv::Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    cv::Mat cam_distortion_ = (cv::Mat_<double>(1, 5) << 0.0, 0.0, 0.0, 0.0, 0.0);
};