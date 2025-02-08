// 目前有一个滤装甲板中心的KF（KFXYZ），一个滤整车角度的KF（KFYaw），还有一个滤整车中心的UKF（UKFXY）
// 装甲板中心的KF用于平动，整车角度KF+整车中心UKF用于小陀螺

#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

class KFXYZ {
public:
    KFXYZ(const std::string& params_path) {
        load_params(params_path);
        cvkf_xyz_.transitionMatrix = (cv::Mat_<float>(6, 6) <<
            1, 0, 0, 1, 0, 0,
            0, 1, 0, 0, 1, 0,
            0, 0, 1, 0, 0, 1,
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1
        );
        cv::setIdentity(cvkf_xyz_.measurementMatrix);
        cv::setIdentity(cvkf_xyz_.errorCovPost, cv::Scalar::all(1.0));
    }

    void initialize(const cv::Point3f& meas) {
        cvkf_xyz_.statePost.at<float>(0) = meas.x;
        cvkf_xyz_.statePost.at<float>(1) = meas.y;
        cvkf_xyz_.statePost.at<float>(2) = meas.z;
        cvkf_xyz_.statePost.at<float>(3) = 0;
        cvkf_xyz_.statePost.at<float>(4) = 0;
        cvkf_xyz_.statePost.at<float>(5) = 0;
    }

    void update(const cv::Point3f& meas) {
        cv::Mat estimated = 
            cvkf_xyz_.correct((cv::Mat_<float>(3, 1) << meas.x, meas.y, meas.z));
        position.x = estimated.at<float>(0);
        position.y = estimated.at<float>(1);
        position.z = estimated.at<float>(2);
        velocity.x = estimated.at<float>(3);
        velocity.y = estimated.at<float>(4);
        velocity.z = estimated.at<float>(5);
    }

    void predict(const float time_elapsed) {
        update_transition_mat(time_elapsed);
        cv::Mat predicted = cvkf_xyz_.predict();
        position.x = predicted.at<float>(0);
        position.y = predicted.at<float>(1);
        position.z = predicted.at<float>(2);
    }

    // 强制更新状态量中的位置信息
    void force_change_position(const cv::Point3f& meas) {
        cvkf_xyz_.statePost.at<float>(0) = meas.x;
        cvkf_xyz_.statePost.at<float>(1) = meas.y;
        cvkf_xyz_.statePost.at<float>(2) = meas.z;
    }

    cv::Point3f position, velocity;

private:
    cv::KalmanFilter cvkf_xyz_ = cv::KalmanFilter(6, 3, 0, CV_32F);

    void update_transition_mat(const float time_elapsed) {
        cvkf_xyz_.transitionMatrix.at<float>(0, 3) = time_elapsed;
        cvkf_xyz_.transitionMatrix.at<float>(1, 4) = time_elapsed;
        cvkf_xyz_.transitionMatrix.at<float>(2, 5) = time_elapsed;
    }

    void load_params(const std::string& params_path) {
        cv::FileStorage fs(params_path, cv::FileStorage::READ);
        fs["KFXYZ"]["process_noise_cov"] >> cvkf_xyz_.processNoiseCov;
        fs["KFXYZ"]["measurement_noise_cov"] >> cvkf_xyz_.measurementNoiseCov;
        fs.release();
    }
};

class KFYaw {
public:
    KFYaw(const std::string& params_path) {
        load_params(params_path);
        cvkf_theta_.transitionMatrix = (cv::Mat_<float>(2, 2) <<
            1, 1,
            0, 1
        );
        cv::setIdentity(cvkf_theta_.measurementMatrix);
        cv::setIdentity(cvkf_theta_.errorCovPost, cv::Scalar::all(1.0));
    }

    void initialize(const float meas) {
        cvkf_theta_.statePost.at<float>(0) = meas;
        cvkf_theta_.statePost.at<float>(1) = 0;
    }

    void update(const float meas) {
        cv::Mat estimated = cvkf_theta_.correct((cv::Mat_<float>(1, 1) << meas));
        yaw = estimated.at<float>(0);
        palstance = estimated.at<float>(1);
    }

    void predict(const float time_elapsed) {
        update_transition_mat(time_elapsed);
        cv::Mat predicted = cvkf_theta_.predict();
        yaw = predicted.at<float>(0);
    }

    float yaw, palstance;

private:
    cv::KalmanFilter cvkf_theta_ = cv::KalmanFilter(2, 1, 0, CV_32F);

    void update_transition_mat(const float time_elapsed) {
        cvkf_theta_.transitionMatrix.at<float>(0, 1) = time_elapsed;
    }

    void load_params(const std::string& params_path) {
        cv::FileStorage fs(params_path, cv::FileStorage::READ);
        fs["KFYaw"]["process_noise_cov"] >> cvkf_theta_.processNoiseCov;
        fs["KFYaw"]["measurement_noise_cov"] >> cvkf_theta_.measurementNoiseCov;
        fs.release();
    }
};

class UKFXY {
public:
    UKFXY(const std::string& params_path):
        lambda(3 - state_dim),
        alpha(1e-3),
        beta(2),
        kappa(0) {
        load_params(params_path);
        x = Eigen::VectorXf::Zero(state_dim);
        P = Eigen::MatrixXf::Identity(state_dim, state_dim);
        weights_mean = Eigen::VectorXf(n_sigma_points);
        weights_cov = Eigen::VectorXf(n_sigma_points);
        sigma_points = Eigen::MatrixXf(state_dim, n_sigma_points);

        weights_mean(0) = lambda / (lambda + state_dim);
        weights_cov(0) = weights_mean(0) + (1 - alpha * alpha + beta);
        for (int i = 1; i < n_sigma_points; i++) {
            weights_mean(i) = 1 / (2 * (state_dim + lambda));
            weights_cov(i) = weights_mean(i);
        }
    }

    void initialize(const cv::Point2f& position) {
        x = Eigen::VectorXf::Zero(state_dim);
        x << position.x, position.y, 0, 0;
    }

    void predict(float time_elapsed) {
        generate_sigma_points();

        for (int i = 0; i < n_sigma_points; ++i) {
            sigma_points.col(i) = process_model(sigma_points.col(i), time_elapsed);
        }

        x = sigma_points * weights_mean;
        P = Q;
        for (int i = 0; i < n_sigma_points; ++i) {
            Eigen::VectorXf diff = sigma_points.col(i) - x;
            P += weights_cov(i) * diff * diff.transpose();
        }
        
        set_output_result();
    }

    void update(const cv::Point2f& position) {
        Eigen::VectorXf meas = Eigen::VectorXf::Zero(meas_dim);
        meas << position.x, position.y;
        Eigen::MatrixXf Z = Eigen::MatrixXf(meas_dim, n_sigma_points);
        for (int i = 0; i < n_sigma_points; i++) {
            Z.col(i) = measurement_model(sigma_points.col(i));
        }

        Eigen::VectorXf z_pred = Z * weights_mean;
        Eigen::MatrixXf S = R;
        for (int i = 0; i < n_sigma_points; i++) {
            Eigen::VectorXf diff = Z.col(i) - z_pred;
            S += weights_cov(i) * diff * diff.transpose();
        }

        Eigen::MatrixXf cross_cov = Eigen::MatrixXf::Zero(state_dim, meas_dim);
        for (int i = 0; i < n_sigma_points; i++) {
            Eigen::VectorXf x_diff = sigma_points.col(i) - x;
            Eigen::VectorXf z_diff = Z.col(i) - z_pred;
            cross_cov += weights_cov(i) * x_diff * z_diff.transpose();
        }

        Eigen::MatrixXf K = cross_cov * S.inverse();
        x += K * (meas - z_pred);
        P -= K * S * K.transpose();

        set_output_result();
    }

    cv::Point2f position, velocity;

private:
    static constexpr int state_dim = 4;
    static constexpr int meas_dim = 2;
    static constexpr int n_sigma_points = 2 * state_dim + 1;
    float lambda, alpha, beta, kappa;

    Eigen::VectorXf x;
    Eigen::MatrixXf P, Q, R;
    Eigen::VectorXf weights_mean, weights_cov;
    Eigen::MatrixXf sigma_points;

    void generate_sigma_points() {
        Eigen::MatrixXf A = P.llt().matrixL();
        sigma_points.col(0) = x;

        for (int i = 0; i < state_dim; i++) {
            sigma_points.col(i + 1) = x + sqrt(lambda + state_dim) * A.col(i);
            sigma_points.col(i + 1 + state_dim) = x - sqrt(lambda + state_dim) * A.col(i);
        }
    }

    Eigen::VectorXf process_model(const Eigen::VectorXf& x, float time_elapsed) {
        Eigen::VectorXf x_new = x;
        x_new(0) += x(2) * time_elapsed;
        x_new(1) += x(3) * time_elapsed;
        return x_new;
    }

    Eigen::VectorXf measurement_model(const Eigen::VectorXf& x) {
        return x.head(meas_dim);
    }

    void load_params(const std::string& params_path) {
        cv::FileStorage fs(params_path, cv::FileStorage::READ);
        cv::Mat process_noise, measurement_noise;
        fs["UKFXY"]["process_noise_cov"] >> process_noise;
        fs["UKFXY"]["measurement_noise_cov"] >> measurement_noise;
        cv::cv2eigen(process_noise, Q);
        cv::cv2eigen(measurement_noise, R);
        fs.release();
    }

    void set_output_result() {
        position.x = x(0);
        position.y = x(1);
        velocity.x = x(2);
        velocity.y = x(3);
    }
};