// 从 https://github.com/julyfun/rm.cv.fans 抄来的

#pragma once

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <autoaim_interfaces/msg/detection.hpp>
#include <geometry_msgs/msg/transform.hpp>

cv::Point3d eigen3d_to_cv(const Eigen::Vector3d& vec) {
    return {vec.x(), vec.y(), vec.z()};
}
Eigen::Vector3d cv3d_to_eigen(const cv::Point3d& point) {
    return {point.x, point.y, point.z};
}

namespace math {
template<typename T>
T squre(const T& x) {
    return x * x;
}

// 总是返回 0 ~ pi
double get_angle_diff(const Eigen::Vector2d& vec1, const Eigen::Vector2d& vec2) {
    if (vec1.norm() == 0.0 || vec2.norm() == 0.0) {
        return 0.0;
    }
    return std::acos(vec1.dot(vec2) / (vec1.norm() * vec2.norm()));
}

// 2 维向量 vec 逆时针旋转 angle
Eigen::Vector2d rotate(const Eigen::Vector2d& vec, const double& angle) {
    Eigen::Matrix2d rotation_mat;
    double sin_angle = std::sin(angle);
    double cos_angle = std::cos(angle);
    rotation_mat << cos_angle, -sin_angle, sin_angle, cos_angle;
    return rotation_mat * vec;
}

double rad_period_correction(double rad) {
    return rad + round((-rad) / (2 * M_PI)) * (2 * M_PI);
}

std::pair<double, double> trisection_find_min(
    double left,
    double right,
    const std::function<double(double)>& cost_function,
    const int iterations
) {
    double phi = (std::sqrt(5.0) - 1.0) / 2.0;
    double ml_cost = 0.0, mr_cost = 0.0;
    int reserved = -1;
    for (int i = 0; i < iterations; i++) {
        double ml = left + (right - left) * (1. - phi);
        double mr = left + (right - left) * phi;
        if (reserved != 0) {
            ml_cost = cost_function(ml);
        }
        if (reserved != 1) {
            mr_cost = cost_function(mr);
        }
        if (ml_cost < mr_cost) {
            right = mr;
            mr_cost = ml_cost;
            reserved = 1;
        } else {
            left = ml;
            ml_cost = mr_cost;
            reserved = 0;
        }
    }
    return std::make_pair((left + right) / 2.0, right - left);
}
} // namespace math

class TrisectionYaw {
public:
    void set_cam_matrix(const cv::Mat intrinsic, const cv::Mat distortion);
    void get_yaw(
        const autoaim_interfaces::msg::Detection& detection,
        geometry_msgs::msg::Transform& transform
    ) const;

private:
    double get_pts_cost(
        const std::vector<cv::Point2d>& ref_pts,
        const std::vector<cv::Point2d>& rotated_pts,
        const double& expected_yaw
    ) const;
    std::vector<cv::Point3d> spin_armor_3d(
        const cv::Point3d& armor_center, 
        const int armor_label, 
        const double& armor_yaw
    ) const;
    std::vector<cv::Point2d> project_3d_to_2d(const std::vector<cv::Point3d>& object_pts) const;

    static constexpr int FIND_ANGLE_ITERATIONS = 12; // 三分法迭代次数，理想精度 < 1
    static constexpr double SIMPLE_TOP_TRACK_AREA_RATIO = 2.0;
    static constexpr double DETECTOR_ERROR_PIXEL_BY_SLOPE = 2.0;
    static constexpr double ARMOR_PITCH = 15.0 / 180.0 * M_PI;

    // 单位: 米
    static constexpr double HEIGHT = 0.055;
    static constexpr double BIG_WIDTH = 0.2253;
    static constexpr double SMALL_WIDTH = 0.135;

    cv::Mat cam_intrinsic_ =
        (cv::Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    cv::Mat cam_distortion_ = (cv::Mat_<double>(1, 5) << 0.0, 0.0, 0.0, 0.0, 0.0);
};

void TrisectionYaw::set_cam_matrix(const cv::Mat intrinsic, const cv::Mat distortion) {
    cam_intrinsic_ = intrinsic.clone();
    cam_distortion_ = distortion.clone();
}

void TrisectionYaw::get_yaw(
    const autoaim_interfaces::msg::Detection& detection,
    geometry_msgs::msg::Transform& transform
) const {
    std::vector<cv::Point2d> image_pts;
    image_pts.push_back({detection.tl.x, detection.tl.y});
    image_pts.push_back({detection.bl.x, detection.bl.y});
    image_pts.push_back({detection.br.x, detection.br.y});
    image_pts.push_back({detection.tr.x, detection.tr.y});
    const cv::Point3d armor_center = {
        transform.translation.x,
        transform.translation.y,
        transform.translation.z
    };
    std::function cost_func = [&](double yaw) -> double {
        std::vector<cv::Point3d> spinned_armor_pts = spin_armor_3d(armor_center, detection.label, yaw);
        std::vector<cv::Point2d> spinned_armor_pts_2d = project_3d_to_2d(spinned_armor_pts);
        return get_pts_cost(image_pts, spinned_armor_pts_2d, M_PI / 4);
        // TODO: expect yaw
    };
    const double armor_yaw =
        math::trisection_find_min(M_PI / 2, M_PI * 3 / 2, cost_func, FIND_ANGLE_ITERATIONS).first - M_PI;
    tf2::Quaternion quaternion;
    // 相机坐标系向右是x，向下是y，向前是z。（与opencv一致）
    // setEuler旋转顺序：先绕Y，再绕X，最后绕Z
    quaternion.setEuler(-armor_yaw, M_PI / 2 - ARMOR_PITCH, 0); 
    transform.rotation.x = quaternion.getX();
    transform.rotation.y = quaternion.getY();
    transform.rotation.z = quaternion.getZ();
    transform.rotation.w = quaternion.getW();
}

double TrisectionYaw::get_pts_cost(
    const std::vector<cv::Point2d>& ref_pts,
    const std::vector<cv::Point2d>& rotated_pts,
    const double& expected_yaw
) const {
    std::size_t size = ref_pts.size();
    std::vector<Eigen::Vector2d> refs;
    std::vector<Eigen::Vector2d> pts;
    for (std::size_t i = 0u; i < size; ++i) {
        refs.emplace_back(ref_pts[i].x, ref_pts[i].y);
        pts.emplace_back(rotated_pts[i].x, rotated_pts[i].y);
    }
    double cost = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        std::size_t p = (i + 1) % size;
        // i - p 构成线段。过程：先移动起点，再补长度，再旋转
        Eigen::Vector2d ref_d = refs[p] - refs[i]; // 标准
        Eigen::Vector2d pt_d = pts[p] - pts[i];
        // 长度差代价 + 起点差代价 / 2（0 度左右应该抛弃）
        double pixel_dis = // dis 是指方差平面内到原点的距离
            (0.5 * ((refs[i] - pts[i]).norm() + (refs[p] - pts[p]).norm())
             + std::fabs(ref_d.norm() - pt_d.norm()))
            / ref_d.norm();
        double angular_dis = ref_d.norm() * math::get_angle_diff(ref_d, pt_d) / ref_d.norm();
        // 平方可能是为了配合 sin 和 cos
        // 弧度差代价（0 度左右占比应该大）
        double cost_i = math::squre(pixel_dis * std::sin(expected_yaw))
            + math::squre(angular_dis * std::cos(expected_yaw)) * DETECTOR_ERROR_PIXEL_BY_SLOPE;
        // 重投影像素误差越大，越相信斜率
        cost += std::sqrt(cost_i);
    }
    return cost;
}

std::vector<cv::Point3d> TrisectionYaw::spin_armor_3d(
    const cv::Point3d& armor_center,
    const int armor_label,
    const double& armor_yaw
) const {
    const double WIDTH = (armor_label == 1) ? BIG_WIDTH : SMALL_WIDTH;
    cv::Point3d width_vec = 
        cv::Point3d(cos(armor_yaw - M_PI), 0, sin(armor_yaw - M_PI)) * (WIDTH / 2);
    cv::Point3d height_vec = cv::Point3d(
        sin(ARMOR_PITCH) * sin(armor_yaw - M_PI),
        cos(ARMOR_PITCH),
        -sin(ARMOR_PITCH) * cos(armor_yaw - M_PI)
    ) * HEIGHT;
    std::vector<cv::Point3d> corners {
        armor_center - width_vec - height_vec,
        armor_center - width_vec + height_vec,
        armor_center + width_vec + height_vec,
        armor_center + width_vec - height_vec
    };
    return corners;
}

std::vector<cv::Point2d> 
TrisectionYaw::project_3d_to_2d(const std::vector<cv::Point3d>& object_pts) const {
    std::vector<cv::Point2d> image_pts;
    // 相机坐标系到平面的投影中，rvec和tvec都是0
    projectPoints(
        object_pts,
        cv::Mat::zeros(3, 1, CV_32F),
        cv::Mat::zeros(3, 1, CV_32F),
        cam_intrinsic_,
        cam_distortion_,
        image_pts
    );
    return image_pts;
}