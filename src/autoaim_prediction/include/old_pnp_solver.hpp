#pragma once

#include <vector>
#include <math_utils.hpp>
using namespace std;
#include <tf2/convert.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <hw_sentry_interfaces/msg/detection.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

class OldPnPSolver {

private:
    std::vector<cv::Point3f> world_points_;
    std::vector<cv::Point2f> img_points_;
    int  label_;
    std::vector<cv::Mat> rvecs_, tvecs_;

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

    /*!
        @brief 使用IPPE解PnP，返回两个解
        @param detection 输入的4个角点位置，以及装甲板标签（用于判断装甲板大小）。
        @param transform 输出的装甲板坐标系到相机坐标系的变换。
        虽然translation和rotation都写入了，不过鉴于后面有个重投影会再算出rotation，所以写入的rotation目前是没用的。
        @return 返回1表示找解的时候出现问题（不过这种情况好像不常出现？目前好像没处理）。
        @attention 相机坐标系定义与opencv一致（向右是x，向下是y，向前是z），装甲板坐标系定义是正常的（向右是x，向前是y，向上是z）。
    */
    std::tuple<bool , geometry_msgs::msg::TransformStamped, geometry_msgs::msg::TransformStamped> solve_pnp_generic(
        const hw_sentry_interfaces::msg::Detection& detection,
        geometry_msgs::msg::TransformStamped & transform //传入target，但是返回两个pnp解的transform
    ) {

        label_ = detection.label;

        world_points_ = detection.label == 1 ? BIG_POINTS : SMALL_POINTS;
        img_points_ = {
            {detection.tl.x, detection.tl.y},
            {detection.bl.x, detection.bl.y},
            {detection.br.x, detection.br.y},
            {detection.tr.x, detection.tr.y}
        };
        
        // IPPE返回两组解
        const int solutions = cv::solvePnPGeneric( //pnp结果camera转到目标
            world_points_,
            img_points_,
            cam_intrinsic_,
            cam_distortion_,
            rvecs_,
            tvecs_,
            false,
            cv::SOLVEPNP_IPPE
        );

        tf2::Quaternion camera2gimbal(0.5, -0.5, 0.5, -0.5);

        // 定义两组PNP解的frame，相机坐标系为参考系
        geometry_msgs::msg::TransformStamped pnp_solution_1;
        pnp_solution_1.header.stamp = transform.header.stamp;  
        pnp_solution_1.header.frame_id = "autoaim_camera";
        pnp_solution_1.child_frame_id = "pnp_solution_1"; 

        geometry_msgs::msg::TransformStamped pnp_solution_2;
        pnp_solution_2.header.stamp = transform.header.stamp;
        pnp_solution_2.header.frame_id = "autoaim_camera";
        pnp_solution_2.child_frame_id = "pnp_solution_2";

        pnp_solution_1.transform.translation.x = tvecs_[0].at<double>(2);
        pnp_solution_1.transform.translation.y = -tvecs_[0].at<double>(0);
        pnp_solution_1.transform.translation.z = -tvecs_[0].at<double>(1);

        cv::Mat rodrigues_1;
        cv::Rodrigues(rvecs_[0], rodrigues_1); //旋转向量转旋转矩阵
        tf2::Matrix3x3 rotation_matrix(
            rodrigues_1.at<double>(0, 0), rodrigues_1.at<double>(0, 1), rodrigues_1.at<double>(0, 2),
            rodrigues_1.at<double>(1, 0), rodrigues_1.at<double>(1, 1), rodrigues_1.at<double>(1, 2),
            rodrigues_1.at<double>(2, 0), rodrigues_1.at<double>(2, 1), rodrigues_1.at<double>(2, 2)
        );

        tf2::Quaternion quaternion_1;
        rotation_matrix.getRotation(quaternion_1);
        
        quaternion_1 = camera2gimbal * quaternion_1;
        
        pnp_solution_1.transform.rotation.x = quaternion_1.getX();
        pnp_solution_1.transform.rotation.y = quaternion_1.getY();
        pnp_solution_1.transform.rotation.z = quaternion_1.getZ();
        pnp_solution_1.transform.rotation.w = quaternion_1.getW();

        //pnp解第二个
        pnp_solution_2.transform.translation.x = tvecs_[1].at<double>(2);
        pnp_solution_2.transform.translation.y = -tvecs_[1].at<double>(0);
        pnp_solution_2.transform.translation.z = -tvecs_[1].at<double>(1);

        cv::Mat rodrigues_2;
        cv::Rodrigues(rvecs_[1], rodrigues_2); //旋转向量转旋转矩阵
        tf2::Matrix3x3 rotation_matrix1(
            rodrigues_2.at<double>(0, 0), rodrigues_2.at<double>(0, 1), rodrigues_2.at<double>(0, 2),
            rodrigues_2.at<double>(1, 0), rodrigues_2.at<double>(1, 1), rodrigues_2.at<double>(1, 2),
            rodrigues_2.at<double>(2, 0), rodrigues_2.at<double>(2, 1), rodrigues_2.at<double>(2, 2)
        );
        
        tf2::Quaternion quaternion_2;
        rotation_matrix1.getRotation(quaternion_2);
        
        quaternion_2 = camera2gimbal * quaternion_2;
        
        pnp_solution_2.transform.rotation.x = quaternion_2.getX();
        pnp_solution_2.transform.rotation.y = quaternion_2.getY();
        pnp_solution_2.transform.rotation.z = quaternion_2.getZ();
        pnp_solution_2.transform.rotation.w = quaternion_2.getW();

        transform.transform = pnp_solution_1.transform;

        return {true, pnp_solution_1, pnp_solution_2};   
    }

    float get_armor_rotate_angle(const std::tuple<float, float, float>& gimbal_ypr,
                                const geometry_msgs::msg::Transform& pnp_solution_1,
                                const geometry_msgs::msg::Transform& pnp_solution_2) //rotation_matrix转到世界坐标系
    {

        float gimbal_yaw, gimbal_pitch, gimbal_roll;
        std::tie(gimbal_yaw, gimbal_pitch, gimbal_roll) = gimbal_ypr;

        Eigen::AngleAxisd rollAngle(gimbal_roll, Eigen::Vector3d::UnitX());  
        Eigen::AngleAxisd pitchAngle(gimbal_pitch, Eigen::Vector3d::UnitY()); 
        Eigen::AngleAxisd yawAngle(gimbal_yaw, Eigen::Vector3d::UnitZ()); 

        // 组合旋转
        Eigen::Quaternion q = rollAngle * pitchAngle;

        Eigen::Quaternion q1(
            pnp_solution_1.rotation.w,
            pnp_solution_1.rotation.x,
            pnp_solution_1.rotation.y,
            pnp_solution_1.rotation.z
        );
        q1.normalize();

        Eigen::Quaternion q2(
            pnp_solution_2.rotation.w,
            pnp_solution_2.rotation.x,
            pnp_solution_2.rotation.y,
            pnp_solution_2.rotation.z
        );
        q2.normalize();

        Eigen::Quaternion gimbal2camera(0.5, 0.5, -0.5, 0.5);

        q1 = gimbal2camera * q * q1, q2 = gimbal2camera * q * q2;

        Eigen::Vector3d p1(pnp_solution_1.translation.x, pnp_solution_1.translation.y, pnp_solution_1.translation.z);
        Eigen::Vector3d p2(pnp_solution_2.translation.x, pnp_solution_2.translation.y, pnp_solution_2.translation.z);

        Eigen::Matrix3d trans = (gimbal2camera * q).toRotationMatrix();

        p1 = trans * p1, p2 = trans * p2;

        cv::Mat rotVec1, rotVec2, tVec1, tVec2;
        cv::eigen2cv(q1.toRotationMatrix(), rotVec1);
        Rodrigues(rotVec1,rotVec1);
        tVec1 = (cv::Mat_<double>(1, 3) << p1[0], p1[1], p1[2]);

        cv::eigen2cv(q2.toRotationMatrix(), rotVec2);
        Rodrigues(rotVec2,rotVec2);
        tVec2 = (cv::Mat_<double>(1, 3) << p2[0], p2[1], p2[2]);
    
        //接下来进行重投影
        std::vector<cv::Point2f> projectedPoints1, projectedPoints2;
        auto pw = world_points_;

        //将世界坐标系坐标重投影回平面，但已经转到IMU坐标系下了，所以是x、z轴平行于地面，y轴垂直地面的 ，reproject坐标系
        projectPoints(pw, rotVec1, tVec1, cam_intrinsic_, cam_distortion_, projectedPoints1);
        projectPoints(pw, rotVec2, tVec2, cam_intrinsic_, cam_distortion_, projectedPoints2);

        //下面就是根据几何方法解出yaw
        float dirAngle1, dirAngle2;
        vector<cv::Point2f> pts = img_points_;

        cv::Point2f vertical_delta1 = (projectedPoints1[0] - projectedPoints1[1] - projectedPoints1[2] + projectedPoints1[3]) / 2;
        cv::Point2f vertical_delta2 = (projectedPoints2[0] - projectedPoints2[1] - projectedPoints2[2] + projectedPoints2[3]) / 2;

        float rotateAngle1 = asin(min(1.0f, fabs(tan(vertical_delta1.x / vertical_delta1.y) / tan(math::d2r(15))))) * (vertical_delta1.x > 0 ? -1 : 1);
        float rotateAngle2 = asin(min(1.0f, fabs(tan(vertical_delta2.x / vertical_delta2.y) / tan(math::d2r(15))))) * (vertical_delta2.x > 0 ? -1 : 1);

        auto marker_rotateAngle = (rotateAngle1 + rotateAngle2) / 2;
        marker_rotateAngle = math::rad_period_correction(marker_rotateAngle);

        // printf("rotateAngle: %.2f\n", math::r2d(marker_rotateAngle));

        // return marker_rotateAngle;
        
        // 下面是BA优化尝试优化pnp解的代码
        Eigen::Matrix3d world2IMU_rotation_matrix;
        world2IMU_rotation_matrix = rollAngle * pitchAngle;
        
        // 进行BA优化，先获得原相机坐标系下的位姿
        cv::Mat origin_rotVec1 = rvecs_[0].clone();
        cv::Mat origin_rotVec2 = rvecs_[1].clone();
        //获得旋转向量
        cv::Mat tVec3 = tvecs_[0].clone();
        cv::Mat tVec4 = tvecs_[1].clone();
        // BA优化函数
        cv::solvePnPRefineVVS(pw, pts, cam_intrinsic_, cam_distortion_, origin_rotVec1, tVec3);
        cv::solvePnPRefineVVS(pw, pts, cam_intrinsic_, cam_distortion_, origin_rotVec2, tVec4);        

        // 将优化后的位姿转成欧拉角
        cv::Mat mat3, mat4;
        cv::Rodrigues(origin_rotVec1, mat3);
        cv::Rodrigues(origin_rotVec2, mat4);
        Eigen::Matrix3d eigen_mat3, eigen_mat4;
        cv2eigen(mat3, eigen_mat3);
        cv2eigen(mat4, eigen_mat4);

        Eigen::Quaternion camera2gimbal(-0.5, 0.5, -0.5, 0.5);

        Eigen::Matrix3d BA_mat1 = world2IMU_rotation_matrix * camera2gimbal * eigen_mat3;
        Eigen::Matrix3d BA_mat2 = world2IMU_rotation_matrix * camera2gimbal * eigen_mat4;
        Eigen::Vector3d euler_angles3 = math::to_euler_angle(BA_mat1);
        Eigen::Vector3d euler_angles4 = math::to_euler_angle(BA_mat2);

        // printf("pnp_yaw: %.2f %.2f\n", math::r2d(euler_angles3[2]), math::r2d(euler_angles4[2]));

        // 根据重投影解从正负两个解中选一个
        Eigen::Vector3d euler_angles5;
        if (fabs(math::rad_period_correction(marker_rotateAngle - euler_angles3[2])) <
            fabs(math::rad_period_correction(marker_rotateAngle - euler_angles4[2])))
        {
            euler_angles5 = euler_angles3;
        }
        else
        {
            euler_angles5 = euler_angles4;
        }

        // printf("selected: %.2f\n", math::r2d(euler_angles5[2]));

        // 对BA优化解和重投影解选择性相信
        float alpha = (fabs(math::r2d(marker_rotateAngle) / 2 + math::r2d(euler_angles5[2]) / 2) - 30) / 90 * 15;
        float belta = 1 / (1 + exp(-alpha));
        auto marker_pnpRotateAngle = (1-belta) * marker_rotateAngle + belta * euler_angles5[2];
        return marker_pnpRotateAngle;
    }

public:

    OldPnPSolver() = default;

    ~OldPnPSolver() = default;

    bool solve_pnp(const hw_sentry_interfaces::msg::Detection& detection,
                    geometry_msgs::msg::TransformStamped & transform,
                    const std::tuple<float, float, float>& gimbal_ypr) {

        auto pnp_solution = solve_pnp_generic(detection, transform);

        tf2::Quaternion q(
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        );
        tf2::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);

        // printf("pnp_yaw: %.2f\n", math::r2d(yaw));

        double real_yaw = get_armor_rotate_angle(gimbal_ypr, std::get<1>(pnp_solution).transform, std::get<2>(pnp_solution).transform);
        
        printf("geometry_yaw: %.2f\n", math::r2d(real_yaw));

        // float alpha = (fabs(math::r2d(geo_yaw) / 2 + math::r2d(yaw) / 2) - 30) / 90 * 15;
        // float belta = 1 / (1 + exp(-alpha));
        // auto real_yaw = (1-belta) * geo_yaw + belta * yaw;

        // printf("real_yaw: %.2f\n", math::r2d(real_yaw));

        q.setRPY(roll, pitch, real_yaw);

        transform.transform.rotation.x = q.getX();
        transform.transform.rotation.y = q.getY();
        transform.transform.rotation.z = q.getZ();
        transform.transform.rotation.w = q.getW();

        return true;
    }

    /*!
        @brief 设置相机的内参矩阵和畸变矩阵
        @attention 算PnP前一定要先设置这个
    */
    void set_cam_matrix(const cv::Mat intrinsic, const cv::Mat distortion) {
        cam_intrinsic_ = intrinsic.clone();
        cam_distortion_ = distortion.clone();
    }
};