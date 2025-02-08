#pragma once

#include <tf2/convert.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <opencv2/opencv.hpp>

#include <autoaim_interfaces/msg/detection.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

class PnPSolver {
public:
    /*!
        @brief 使用IPPE解PnP
        @param detection 输入的4个角点位置，以及装甲板标签（用于判断装甲板大小）。
        @param transform 输出的装甲板坐标系到相机坐标系的变换。
        虽然translation和rotation都写入了，不过鉴于后面有个重投影会再算出rotation，所以写入的rotation目前是没用的。
        @return 返回1表示找解的时候出现问题（不过这种情况好像不常出现？目前好像没处理）。
        @attention 相机坐标系定义与opencv一致（向右是x，向下是y，向前是z），装甲板坐标系定义是正常的（向右是x，向前是y，向上是z）。
    */
    bool solve_pnp(
        const autoaim_interfaces::msg::Detection& detection,
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
        // IPPE返回两组对称解，这里筛选z大于0（即在相机前面）的返回
        int solution_index = 0;
        if (solutions >= 1 && tvecs[0].at<double>(2) > 0) {
            solution_index = 0;
        } else if (solutions >= 2 && tvecs[1].at<double>(2) > 0) {
            solution_index = 1;
        } else {
            return 1;
        }
        transform.translation.x = tvecs[solution_index].at<double>(0);
        transform.translation.y = tvecs[solution_index].at<double>(1);
        transform.translation.z = tvecs[solution_index].at<double>(2);

        cv::Mat rodrigues;
        cv::Rodrigues(rvecs[solution_index], rodrigues);
        tf2::Matrix3x3 rotation_matrix(
            rodrigues.at<double>(0, 0), rodrigues.at<double>(0, 1), rodrigues.at<double>(0, 2),
            rodrigues.at<double>(1, 0), rodrigues.at<double>(1, 1), rodrigues.at<double>(1, 2),
            rodrigues.at<double>(2, 0), rodrigues.at<double>(2, 1), rodrigues.at<double>(2, 2)
        );
        tf2::Quaternion quaternion;
        rotation_matrix.getRotation(quaternion);
        transform.rotation.x = quaternion.getX();
        transform.rotation.y = quaternion.getY();
        transform.rotation.z = quaternion.getZ();
        transform.rotation.w = quaternion.getW();
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
        {-SMALL_WIDTH / 2, 0, HEIGHT / 2},
        {-SMALL_WIDTH / 2, 0, -HEIGHT / 2},
        {SMALL_WIDTH / 2, 0, -HEIGHT / 2},
        {SMALL_WIDTH / 2, 0, HEIGHT / 2}
    };
    const std::vector<cv::Point3f> BIG_POINTS {
        {-BIG_WIDTH / 2, 0, HEIGHT / 2},
        {-BIG_WIDTH / 2, 0, -HEIGHT / 2},
        {BIG_WIDTH / 2, 0, -HEIGHT / 2},
        {BIG_WIDTH / 2, 0, HEIGHT / 2}
    };

    cv::Mat cam_intrinsic_ =
        (cv::Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    cv::Mat cam_distortion_ = (cv::Mat_<double>(1, 5) << 0.0, 0.0, 0.0, 0.0, 0.0);
};