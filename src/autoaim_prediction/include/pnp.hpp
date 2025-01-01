#pragma once

#include <tf2/convert.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_ros/transform_broadcaster.h>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include <autoaim_interfaces/msg/detection.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

namespace pnp {
// 单位: mm
constexpr float HEIGHT = 55;
constexpr float BIG_WIDTH = 225.3;
constexpr float SMALL_WIDTH = 135;
static const std::vector<cv::Point3d> SMALL_POINTS {
    {0, SMALL_WIDTH / 2, HEIGHT / 2},
    {0, SMALL_WIDTH / 2, -HEIGHT / 2},
    {0, -SMALL_WIDTH / 2, -HEIGHT / 2},
    {0, -SMALL_WIDTH / 2, HEIGHT / 2}
};
static const std::vector<cv::Point3d> BIG_POINTS {
    {0, BIG_WIDTH / 2, HEIGHT / 2},
    {0, BIG_WIDTH / 2, -HEIGHT / 2},
    {0, -BIG_WIDTH / 2, -HEIGHT / 2},
    {0, -BIG_WIDTH / 2, HEIGHT / 2}
};

static cv::Mat cam_intrinsic =
    (cv::Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
static cv::Mat cam_distortion = (cv::Mat_<double>(1, 5) << 0.0, 0.0, 0.0, 0.0, 0.0);

void solve_pnp(
    const autoaim_interfaces::msg::Detection& detection,
    geometry_msgs::msg::Transform& transform
) {
    const auto& world_points = detection.label == 1 ? BIG_POINTS : SMALL_POINTS;
    const std::vector<cv::Point2d> img_points {
        {detection.tl.x, detection.tl.y},
        {detection.bl.x, detection.bl.y},
        {detection.br.x, detection.br.y},
        {detection.tr.x, detection.tr.y}
    };
    cv::Mat rvec, tvec;
    cv::solvePnP(
        world_points,
        img_points,
        cam_intrinsic,
        cam_distortion,
        rvec,
        tvec,
        false,
        cv::SOLVEPNP_IPPE
    );
    transform.translation.x = tvec.at<double>(0);
    transform.translation.y = tvec.at<double>(1);
    transform.translation.z = tvec.at<double>(2);

    cv::Mat rodrigues;
    cv::Rodrigues(rvec, rodrigues);
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
}
} // namespace pnp