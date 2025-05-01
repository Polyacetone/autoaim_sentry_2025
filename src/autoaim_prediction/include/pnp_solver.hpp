#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <tf2/convert.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <hw_sentry_interfaces/msg/detection.hpp>
#include <math_utils.hpp>

class PnPSolver {
public:
    /*!
        @brief 设置相机的内参和畸变矩阵
        @attention 在使用solve_pnp(...)之前一定要先设置内参矩阵
    */
    void set_cam_matrix(const cv::Mat intrinsic, const cv::Mat distortion);

    /*!
        @brief 用opencv的solvepnp求位置，用几何法优化的三分法求yaw
        @return 返回false表示找解的时候有问题
    */
    bool solve_pnp(
        const hw_sentry_interfaces::msg::Detection& detection,
        const std::tuple<float, float, float>& gimbal_ypr,
        geometry_msgs::msg::Transform& transform
    ) const;

private:
    // 取重投影后的上中点和下中点连线，根据斜率求yaw角
    float geometric_get_yaw(
        const Eigen::Quaternionf& gimbal_pr,
        const Eigen::Quaternionf& rotation,
        const Eigen::Vector3f& translation,
        const int label
    ) const;

    // 对重投影后的pts_cost进行三分法求最小值来求装甲板yaw角
    float trisection_get_yaw(
        const Eigen::Quaternionf& gimbal_pr,
        const Eigen::Vector3f& translation,
        const std::vector<cv::Point2f>& img_pts,
        const int label,
        const float prior_yaw
    ) const;

    // 计算实际的角点和重投影后的角点的差异，作为传入三分法的损失函数
    float get_pts_cost(
        const std::vector<cv::Point2f>& ref_pts,
        const std::vector<cv::Point2f>& rotated_pts,
        const float prior_yaw
    ) const;

    // 计算旋转角（armor_pitch是装甲板相对于世界系的pitch，是已知的；armor_yaw是三分法扔进来的变量）对应的装甲板角点坐标
    std::vector<Eigen::Vector3f> get_spinned_pts(
        const Eigen::Vector3f& armor_center, 
        const int armor_label,
        const float armor_pitch,
        const float armor_yaw
    ) const;

    // 三分法求函数的最小值
    std::pair<float, float> trisection_find_min(
        float left,
        float right,
        const std::function<float(float)>& cost_function,
        const int iterations
    ) const;

    float get_armor_pitch_to_world(int label) const {
        return math::d2r((label == 5) ? -15 : 15);
    }
    bool is_big_armor(int label) const {
        return (label == 1 || label == 7);
    }

    static constexpr int FIND_ANGLE_ITERATIONS = 12; // 三分法迭代次数，理想精度<1
    static constexpr float DETECTOR_ERROR_PIXEL_BY_SLOPE = 2.0;

    // 单位: 米
    static constexpr float HEIGHT = 0.05603;
    static constexpr float BIG_WIDTH = 0.231;
    static constexpr float SMALL_WIDTH = 0.136;
    // 装甲板坐标系：前x，左y，上z
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

    cv::Mat cam_intrinsic_ = (cv::Mat_<float>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    cv::Mat cam_distortion_ = (cv::Mat_<float>(1, 5) << 0.0, 0.0, 0.0, 0.0, 0.0);
};

bool PnPSolver::solve_pnp(
    const hw_sentry_interfaces::msg::Detection& detection,
    const std::tuple<float, float, float>& gimbal_ypr,
    geometry_msgs::msg::Transform& transform
) const {
    const std::vector<cv::Point2f> img_pts {
        {detection.tl.x, detection.tl.y},
        {detection.bl.x, detection.bl.y},
        {detection.br.x, detection.br.y},
        {detection.tr.x, detection.tr.y}
    };
    std::vector<cv::Mat> rvecs(2), tvecs(2);
    cv::solvePnPGeneric(
        is_big_armor(detection.label) ? BIG_POINTS : SMALL_POINTS,
        img_pts,
        cam_intrinsic_,
        cam_distortion_,
        rvecs,
        tvecs,
        false,
        cv::SOLVEPNP_IPPE
    );
    for (int i = 0; i < 2; i++) {
        rvecs[i].convertTo(rvecs[i], CV_32F);
        tvecs[i].convertTo(tvecs[i], CV_32F);
    }
    // 左乘即可把opencv的相机系（右x，下y，前z）转成我们在tf2中的相机系（前x，左y，上z）
    const Eigen::Quaternionf cv_to_tf(-0.5, 0.5, -0.5, 0.5);

    float gimbal_pitch, gimbal_roll;
    std::tie(std::ignore, gimbal_pitch, gimbal_roll) = gimbal_ypr;
    const Eigen::AngleAxisf roll_rotation(gimbal_roll, Eigen::Vector3f::UnitX());
    const Eigen::AngleAxisf pitch_rotation(gimbal_pitch, Eigen::Vector3f::UnitY());
    // 左乘即可把我们在tf2的相机系转掉云台的pitch和roll，不转yaw（因为后面获取的yaw都是相对于相机系的）
    const Eigen::Quaternionf gimbal_pr(roll_rotation * pitch_rotation);

    Eigen::Vector3f rvec;
    cv::cv2eigen(rvecs[0], rvec);
    // 把opencv pnp的rvec转成四元数，然后转到我们的tf2系下
    Eigen::Quaternionf rotation(Eigen::AngleAxisf(rvec.norm(), rvec.normalized()));
    rotation = cv_to_tf * rotation;

    // 把opencv pnp的tvec转到我们的tf2系下
    Eigen::Vector3f translation;
    cv::cv2eigen(tvecs[0], translation);
    translation = cv_to_tf * translation;

    // geo_yaw是几何法求的yaw角，数值上可能不够准确，但不会出现正负跳变
    const float geo_yaw = geometric_get_yaw(gimbal_pr, rotation, translation, detection.label);
    // tri_yaw相当于用三分法去优化上面的geo_yaw，使其更加准确。用geo_yaw限制三分法可以避免有时发生的跳变
    const float tri_yaw = trisection_get_yaw(gimbal_pr, translation, img_pts, detection.label, geo_yaw);

    tf2::Quaternion quaternion;
    const float armor_pitch_to_world = get_armor_pitch_to_world(detection.label);
    // 旋转相当于相机系，所以需要用装甲板在世界系下的roll（取0即可）和pitch减去云台的roll和pitch
    quaternion.setRPY(-gimbal_roll, armor_pitch_to_world - gimbal_pitch, tri_yaw);
    transform.translation.x = translation(0);
    transform.translation.y = translation(1);
    transform.translation.z = translation(2);
    transform.rotation.x = quaternion.getX();
    transform.rotation.y = quaternion.getY();
    transform.rotation.z = quaternion.getZ();
    transform.rotation.w = quaternion.getW();

    // 如果装甲板的x坐标小于0，说明解到了相机后方，很明显有问题
    if (transform.translation.x < 0) {
        return false;
    }
    return true;
}

float PnPSolver::geometric_get_yaw(
    const Eigen::Quaternionf& gimbal_pr,
    const Eigen::Quaternionf& rotation,
    const Eigen::Vector3f& translation,
    const int label
) const {
    const Eigen::Quaternionf cv_to_tf(-0.5, 0.5, -0.5, 0.5);
    // 在tf2中的相机系（前x，左y，上z）下转掉云台的pitch和roll，得到“世界系”（其实不是真正的世界系，因为yaw没有转掉）
    const Eigen::Quaternionf corrected_rotation(gimbal_pr * rotation);
    const Eigen::Vector3f corrected_translation(gimbal_pr * translation);
    // 把转掉了pitch和roll的“世界系”转回到opencv的相机系（右x，下y，前z），用于projectPoints重投影
    const Eigen::AngleAxisf corrected_rotation_cv(cv_to_tf.inverse() * corrected_rotation);
    const Eigen::Vector3f corrected_translation_cv(cv_to_tf.inverse() * corrected_translation);

    cv::Vec3f corrected_rvec_cv, corrected_tvec_cv;
    cv::eigen2cv(Eigen::Vector3f(corrected_rotation_cv.angle() * corrected_rotation_cv.axis()), corrected_rvec_cv);
    cv::eigen2cv(corrected_translation_cv, corrected_tvec_cv);
    std::vector<cv::Point2f> reprojected_pts;
    // 在转掉了云台pitch和roll的opencv相机系中重投影，得到装甲板角点
    projectPoints(
        is_big_armor(label) ? BIG_POINTS : SMALL_POINTS,
        corrected_rvec_cv,
        corrected_tvec_cv,
        cam_intrinsic_,
        cam_distortion_,
        reprojected_pts
    );
    const cv::Point2f vertical_line =
        (reprojected_pts[0] + reprojected_pts[3] - reprojected_pts[1] - reprojected_pts[2]) / 2;
    const float armor_pitch_to_world = get_armor_pitch_to_world(label);
    return asin(std::clamp(tan(vertical_line.x / vertical_line.y) / tan(armor_pitch_to_world), -0.7f, 0.7f));
}

float PnPSolver::trisection_get_yaw(
    const Eigen::Quaternionf& gimbal_pr,
    const Eigen::Vector3f& translation,
    const std::vector<cv::Point2f>& img_pts,
    const int label,
    const float prior_yaw
) const {
    const Eigen::Quaternionf cv_to_tf(-0.5, 0.5, -0.5, 0.5);
    // 在tf2中的相机系（前x，左y，上z）下转掉云台的pitch和roll，得到“世界系”（其实不是真正的世界系，因为yaw没有转掉）
    const Eigen::Vector3f corrected_translation(gimbal_pr * translation);

    std::function cost_func = [&](float yaw) -> float {
        const float armor_pitch_to_world = get_armor_pitch_to_world(label);
        // 在转掉pitch和roll的“世界系”下按给定的yaw转装甲板
        std::vector<Eigen::Vector3f> spinned_armor_pts_corrected =
            get_spinned_pts(corrected_translation, label, armor_pitch_to_world, yaw);
        std::vector<cv::Point3f> spinned_armor_pts_cam;
        for (const auto& corrected_pt: spinned_armor_pts_corrected) {
            // 把“世界系”下的装甲板转回去（先转云台，再转到opencv相机系）
            Eigen::Vector3f cam_pt = cv_to_tf.inverse() * gimbal_pr.inverse() * corrected_pt;
            spinned_armor_pts_cam.emplace_back(cv::Point3f(cam_pt(0), cam_pt(1), cam_pt(2)));
        }
        std::vector<cv::Point2f> spinned_armor_pts_2d;
        // 在opencv相机系（右x，下y，前z）下对转过yaw的装甲板的角点进行重投影
        cv::projectPoints(
            spinned_armor_pts_cam,
            cv::Mat::zeros(3, 1, CV_32F),
            cv::Mat::zeros(3, 1, CV_32F),
            cam_intrinsic_,
            cam_distortion_,
            spinned_armor_pts_2d
        );
        return get_pts_cost(img_pts, spinned_armor_pts_2d, prior_yaw);
    };
    return trisection_find_min(prior_yaw - M_PI / 10, prior_yaw + M_PI / 10, cost_func, FIND_ANGLE_ITERATIONS).first;
}

void PnPSolver::set_cam_matrix(const cv::Mat intrinsic, const cv::Mat distortion) {
    cam_intrinsic_ = intrinsic.clone();
    cam_distortion_ = distortion.clone();
}

float PnPSolver::get_pts_cost(
    const std::vector<cv::Point2f>& ref_pts,
    const std::vector<cv::Point2f>& rotated_pts,
    const float prior_yaw
) const {
    std::vector<Eigen::Vector2f> refs;
    std::vector<Eigen::Vector2f> pts;
    for (int i = 0; i < 4; i++) {
        refs.emplace_back(ref_pts[i].x, ref_pts[i].y);
        pts.emplace_back(rotated_pts[i].x, rotated_pts[i].y);
    }
    float cost = 0.0;
    for (int i = 0; i < 4; i++) {
        int p = (i + 1) % 4;
        // i - p 构成线段。过程：先移动起点，再补长度，再旋转
        Eigen::Vector2f ref_d = refs[p] - refs[i]; // 标准
        Eigen::Vector2f pt_d = pts[p] - pts[i];
        // 长度差代价 + 起点差代价 / 2（0 度左右应该抛弃）
        float pixel_dis = // dis 是指方差平面内到原点的距离
            (0.5 * ((refs[i] - pts[i]).norm() + (refs[p] - pts[p]).norm())
            + std::fabs(ref_d.norm() - pt_d.norm())) / ref_d.norm();
        float angular_dis = ref_d.norm() * math::get_angle(ref_d, pt_d) / ref_d.norm();
        // 平方可能是为了配合 sin 和 cos
        // 弧度差代价（0 度左右占比应该大）
        float cost_i = math::square(pixel_dis * std::sin(prior_yaw))
            + math::square(angular_dis * std::cos(prior_yaw)) * DETECTOR_ERROR_PIXEL_BY_SLOPE;
        // 重投影像素误差越大，越相信斜率
        cost += std::sqrt(cost_i);
    }
    return cost;
}

std::vector<Eigen::Vector3f> PnPSolver::get_spinned_pts(
    const Eigen::Vector3f& armor_center,
    const int armor_label,
    const float armor_pitch,
    const float armor_yaw
) const {
    const float WIDTH = is_big_armor(armor_label) ? BIG_WIDTH : SMALL_WIDTH;
    // 长度为装甲板宽度的一半，方向向左（装甲板系y轴正方向）
    const Eigen::Vector3f width_vec = Eigen::Vector3f(-sin(armor_yaw), cos(armor_yaw), 0) * (WIDTH / 2);
    // 长度为装甲板高度的一半，方向向上（装甲板系z轴正方向）
    const Eigen::Vector3f height_vec = Eigen::Vector3f(
        sin(armor_pitch) * cos(armor_yaw),
        sin(armor_pitch) * sin(armor_yaw),
        cos(armor_pitch)
    ) * (HEIGHT / 2);
    const std::vector<Eigen::Vector3f> corners {
        armor_center + width_vec + height_vec,
        armor_center + width_vec - height_vec,
        armor_center - width_vec - height_vec,
        armor_center - width_vec + height_vec
    };
    return corners;
}

std::pair<float, float> PnPSolver::trisection_find_min(
    float left,
    float right,
    const std::function<float(float)>& cost_function,
    const int iterations
) const {
    float phi = (std::sqrt(5.0) - 1.0) / 2.0;
    float ml_cost = 0.0, mr_cost = 0.0;
    int reserved = -1;
    for (int i = 0; i < iterations; i++) {
        float ml = left + (right - left) * (1. - phi);
        float mr = left + (right - left) * phi;
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