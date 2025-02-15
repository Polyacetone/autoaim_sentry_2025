// 从 https://github.com/julyfun/rm.cv.fans 抄来的

#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <autoaim_interfaces/msg/detection.hpp>
#include <geometry_msgs/msg/transform.hpp>

#include <math_utils.hpp>

class TrisectionYaw {
public:
    /*!
        @brief 设置相机的内参和畸变矩阵。
        @attention 在使用get_yaw(...)之前一定要先设置内参矩阵。
    */
    void set_cam_matrix(const cv::Mat intrinsic, const cv::Mat distortion);

    /*!
        @brief 使用降自由度的重投影求装甲板yaw角。
        @param detection yolo识别结果，需要知道装甲板的2D角点坐标，和装甲板标签（用于判断装甲板大小）。
        @param transform 需要从这里读取pnp得到的装甲板中心点坐标（相机系下），并向这里写入计算得到的旋转（装甲板系相对于相机系）。
        @param gimbal_rpy 自己云台相对于世界系的roll, pitch, yaw。用于估计装甲板相对于云台（gimbal）系的pitch，以降自由度。
        @note 相机坐标系定义与opencv一致（向右是x，向下是y，向前是z），装甲板坐标系和云台系定义是正常的（向右是x，向前是y，向上是z）。
        由于相机系是倒转的，云台和世界系方向都是正常的，且云台与相机的旋转是固连的，所以目标装甲板的旋转以云台系为参考更符合直觉。
        但由于云台系中心离相机有一定的距离，所以位移还是应当以相机系为参考。
    */
    void get_rotation(
        const autoaim_interfaces::msg::Detection& detection,
        geometry_msgs::msg::Transform& transform,
        const std::tuple<float, float, float>& gimbal_rpy
    ) const;

private:
    /*!
        @brief 计算实际的角点和重投影后的角点的差异。作为传入三分法的损失函数。
        @param ref_pts yolo实际识别出的角点。
        @param rotated_pts 旋转一个角度后，重投影后的角点。
        @param prior_yaw 先验估计的yaw角。
    */
    float get_pts_cost(
        const std::vector<cv::Point2f>& ref_pts,
        const std::vector<cv::Point2f>& rotated_pts,
        const float& prior_yaw
    ) const;

    /*!
        @brief 计算旋转角（armor_yaw）对应的装甲板角点坐标（3D）。
        @param armor_center 装甲板中心的3D坐标（基于opencv的相机系）。
        @param armor_label 装甲板的标签。用于判断装甲板大小。
        @param armor_pitch 装甲板在gimbal系下的pitch角，是已知的（因为云台相对于世界的pitch和装甲板相对于世界的pitch均已知）。
        @param armor_yaw 装甲板在gimbal系下的yaw角，作为未知量由三分法传入。
        @note 相机坐标系定义与opencv一致（向右是x，向下是y，向前是z）。
        yaw角定义为装甲板向心方向法向量的水平投影与云台正前方的水平投影的夹角，范围-pi/2~pi/2，逆时针为正。
    */
    std::vector<cv::Point3f> spin_armor_3d(
        const cv::Point3f& armor_center, 
        const int armor_label,
        const float armor_pitch,
        const float armor_yaw
    ) const;

    /*!
        @brief 将3D点投影到2D。
        @param object_pts 装甲板的四个角点的3D坐标（基于opencv的相机系）。
        @note 相机坐标系定义与opencv一致（向右是x，向下是y，向前是z）。
    */
    std::vector<cv::Point2f> project_3d_to_2d(const std::vector<cv::Point3f>& object_pts) const;

    // 三分法求函数的最小值。
    std::pair<float, float> trisection_find_min(
        float left,
        float right,
        const std::function<float(float)>& cost_function,
        const int iterations
    ) const;

    static constexpr int FIND_ANGLE_ITERATIONS = 15; // 三分法迭代次数，理想精度<1
    static constexpr float SIMPLE_TOP_TRACK_AREA_RATIO = 2.0;
    static constexpr float DETECTOR_ERROR_PIXEL_BY_SLOPE = 2.0;

    // 单位: 米
    static constexpr float HEIGHT = 0.055;
    static constexpr float BIG_WIDTH = 0.2253;
    static constexpr float SMALL_WIDTH = 0.135;

    cv::Mat cam_intrinsic_ = (cv::Mat_<float>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    cv::Mat cam_distortion_ = (cv::Mat_<float>(1, 5) << 0.0, 0.0, 0.0, 0.0, 0.0);
};

void TrisectionYaw::set_cam_matrix(const cv::Mat intrinsic, const cv::Mat distortion) {
    cam_intrinsic_ = intrinsic.clone();
    cam_distortion_ = distortion.clone();
}

void TrisectionYaw::get_rotation(
    const autoaim_interfaces::msg::Detection& detection,
    geometry_msgs::msg::Transform& transform,
    const std::tuple<float, float, float>& gimbal_ypr
) const {
    float gimbal_yaw, gimbal_pitch, gimbal_roll;
    std::tie(gimbal_yaw, gimbal_pitch, gimbal_roll) = gimbal_ypr;
    const std::vector<cv::Point2f> image_pts {
        {detection.tl.x, detection.tl.y},
        {detection.bl.x, detection.bl.y},
        {detection.br.x, detection.br.y},
        {detection.tr.x, detection.tr.y}
    };
    const cv::Point3f armor_center {
        static_cast<float>(transform.translation.x),
        static_cast<float>(transform.translation.y),
        static_cast<float>(transform.translation.z)
    };
    std::function cost_func = [&](float yaw) -> float {
        std::vector<cv::Point3f> spinned_armor_pts =
            spin_armor_3d(armor_center, detection.label, gimbal_pitch + math::d2r(15), yaw);
        std::vector<cv::Point2f> spinned_armor_pts_2d = project_3d_to_2d(spinned_armor_pts);
        return get_pts_cost(image_pts, spinned_armor_pts_2d, M_PI / 4);
    };
    const float armor_yaw =
        trisection_find_min(-M_PI / 2, M_PI / 2, cost_func, FIND_ANGLE_ITERATIONS).first;
    tf2::Quaternion quaternion;
    // 相机坐标系定义与opencv一致（向右是x，向下是y，向前是z），装甲板坐标系定义是正常的（向右是x，向前是y，向上是z）。
    // 这里用的pi/2 - pitch是为了把相机坐标系方向转成正常坐标系的方向。
    // setEuler旋转顺序：ros2的系（前x，左y，上z）中是XYZ，但在我们的系（右x，前y，上z）里是YXZ。
    quaternion.setEuler(-armor_yaw, M_PI / 2 - gimbal_pitch - math::d2r(15), 0);
    transform.rotation.x = quaternion.getX();
    transform.rotation.y = quaternion.getY();
    transform.rotation.z = quaternion.getZ();
    transform.rotation.w = quaternion.getW();
}

float TrisectionYaw::get_pts_cost(
    const std::vector<cv::Point2f>& ref_pts,
    const std::vector<cv::Point2f>& rotated_pts,
    const float& prior_yaw
) const {
    std::size_t size = ref_pts.size();
    std::vector<Eigen::Vector2f> refs;
    std::vector<Eigen::Vector2f> pts;
    for (int i = 0; i < size; i++) {
        refs.emplace_back(ref_pts[i].x, ref_pts[i].y);
        pts.emplace_back(rotated_pts[i].x, rotated_pts[i].y);
    }
    float cost = 0.0;
    for (int i = 0; i < size; i++) {
        int p = (i + 1) % size;
        // i - p 构成线段。过程：先移动起点，再补长度，再旋转
        Eigen::Vector2f ref_d = refs[p] - refs[i]; // 标准
        Eigen::Vector2f pt_d = pts[p] - pts[i];
        // 长度差代价 + 起点差代价 / 2（0 度左右应该抛弃）
        float pixel_dis = // dis 是指方差平面内到原点的距离
            (0.5 * ((refs[i] - pts[i]).norm() + (refs[p] - pts[p]).norm())
             + std::fabs(ref_d.norm() - pt_d.norm()))
            / ref_d.norm();
        float angular_dis = ref_d.norm() * math::get_angle(ref_d, pt_d) / ref_d.norm();
        // 平方可能是为了配合 sin 和 cos
        // 弧度差代价（0 度左右占比应该大）
        float cost_i = math::squre(pixel_dis * std::sin(prior_yaw))
            + math::squre(angular_dis * std::cos(prior_yaw)) * DETECTOR_ERROR_PIXEL_BY_SLOPE;
        // 重投影像素误差越大，越相信斜率
        cost += std::sqrt(cost_i);
    }
    return cost;
}

std::vector<cv::Point3f> TrisectionYaw::spin_armor_3d(
    const cv::Point3f& armor_center,
    const int armor_label,
    const float armor_pitch,
    const float armor_yaw
) const {
    const float WIDTH = (armor_label == 1) ? BIG_WIDTH : SMALL_WIDTH;
    const cv::Point3f width_vec = cv::Point3f(cos(armor_yaw), 0, sin(armor_yaw)) * (WIDTH / 2);
    const cv::Point3f height_vec = cv::Point3f(
        sin(armor_pitch) * sin(armor_yaw),
        cos(armor_pitch),
        -sin(armor_pitch) * cos(armor_yaw)
    ) * (HEIGHT / 2);
    const std::vector<cv::Point3f> corners {
        armor_center - width_vec - height_vec,
        armor_center - width_vec + height_vec,
        armor_center + width_vec + height_vec,
        armor_center + width_vec - height_vec
    };
    return corners;
}

std::vector<cv::Point2f> 
TrisectionYaw::project_3d_to_2d(const std::vector<cv::Point3f>& object_pts) const {
    std::vector<cv::Point2f> image_pts;
    // 相机坐标系到平面的投影中，rvec和tvec都是0。
    cv::projectPoints(
        object_pts,
        cv::Mat::zeros(3, 1, CV_32F),
        cv::Mat::zeros(3, 1, CV_32F),
        cam_intrinsic_,
        cam_distortion_,
        image_pts
    );
    return image_pts;
}

std::pair<float, float> TrisectionYaw::trisection_find_min(
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