// 目前有一个滤装甲板中心的KF（KFXYZ），一个滤整车角度的KF（KFYaw），还有一个滤整车中心的UKF（UKFXY）
// 装甲板中心的KF用于平动，整车角度KF+整车中心UKF用于小陀螺

#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

class KFXYZ {
public:
    explicit KFXYZ(const std::string& params_path);
    void initialize(const cv::Point3f& meas);
    void update(const cv::Point3f& meas);
    void predict(float delta_t);
    void force_change_position(const cv::Point3f& meas);
    cv::Point3f position, velocity;

private:
    bool ENABLE_ADAPTIVE = false;
    float ADAPTIVE_FORGET_FACTOR = 0.1;
    unsigned ADAPTIVE_START_FRAME = 10;

    Eigen::VectorXf x_;
    Eigen::MatrixXf P_, F_, Q_, R_, H_;
    unsigned frame_count_;

    void set_output_result();
    void ensure_positive_semi_definite(Eigen::MatrixXf& matrix);
    void load_params(const std::string& params_path);
};

class KFYaw {
public:
    explicit KFYaw(const std::string& params_path);
    void initialize(const float meas);
    void update(const float meas);
    void predict(const float time_elapsed);
    void force_change_yaw(const float meas);
    float yaw, palstance;

private:
    Eigen::VectorXf x_;
    Eigen::MatrixXf P_, F_, Q_, R_, H_;

    void set_output_result();
    void load_params(const std::string& params_path);
};

class UKFXY {
public:
    explicit UKFXY(const std::string& params_path);
    void initialize(const cv::Point2f& position);
    void predict(float time_elapsed);
    void update(const cv::Point2f& position);
    cv::Point2f position, velocity;

private:
    const int state_dim = 4;
    const int meas_dim = 2;
    const int n_sigma_points = 2 * state_dim + 1;
    const float lambda, alpha, beta, kappa;

    Eigen::VectorXf x;
    Eigen::MatrixXf P, Q, R;
    Eigen::VectorXf weights_mean, weights_cov;
    Eigen::MatrixXf sigma_points;

    void generate_sigma_points();
    Eigen::VectorXf process_model(const Eigen::VectorXf& x, float time_elapsed);
    Eigen::VectorXf measurement_model(const Eigen::VectorXf& x);
    void load_params(const std::string& params_path);
    void set_output_result();
};