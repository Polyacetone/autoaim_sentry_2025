// 目前有一个滤装甲板中心的KF（KFXYZ），一个滤整车角度的KF（KFYaw），还有一个滤整车中心的UKF（UKFXY）
// 装甲板中心的KF用于平动，整车角度KF+整车中心UKF用于小陀螺

#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

class KFXYZ {
public:
    KFXYZ(const std::string& params_path): x_(6), P_(6, 6), F_(6, 6), Q_(6, 6), R_(3, 3), H_(3, 6) {
        load_params(params_path);
        initialize(cv::Point3f(0, 0, 0));
    }

    void initialize(const cv::Point3f& meas) {
        x_ << meas.x, meas.y, meas.z, 0, 0, 0;
        P_.setIdentity();
        P_ *= 1.0;
        H_ << 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0;
        set_output_result();
    }

    void update(const cv::Point3f& meas) {
        Eigen::VectorXf z(3);
        z << meas.x, meas.y, meas.z;
        Eigen::VectorXf y = z - H_ * x_;
        Eigen::MatrixXf S = H_ * P_ * H_.transpose() + R_;
        Eigen::MatrixXf K = P_ * H_.transpose() * S.inverse();
        x_ += K * y;
        P_ = (Eigen::MatrixXf::Identity(6, 6) - K * H_) * P_;
        set_output_result();
    }

    void predict(float delta_t) {
        F_.setIdentity();
        F_(0, 3) = delta_t;
        F_(1, 4) = delta_t;
        F_(2, 5) = delta_t;
        x_ = F_ * x_;
        P_ = F_ * P_ * F_.transpose() + Q_;
        set_output_result();
    }

    // 强制更新状态量中的位置信息
    void force_change_position(const cv::Point3f& meas) {
        x_(0) = meas.x;
        x_(1) = meas.y;
        x_(2) = meas.z;
        set_output_result();
    }

    cv::Point3f position, velocity;

private:
    Eigen::VectorXf x_;
    Eigen::MatrixXf P_, F_, Q_, R_;
    Eigen::MatrixXf H_;

    void set_output_result() {
        position.x = x_(0);
        position.y = x_(1);
        position.z = x_(2);
        velocity.x = x_(3);
        velocity.y = x_(4);
        velocity.z = x_(5);
    }

    void load_params(const std::string& params_path) {
        cv::FileStorage fs(params_path, cv::FileStorage::READ);
        cv::Mat Q_cv, R_cv;
        fs["KFXYZ"]["process_noise_cov"] >> Q_cv;
        fs["KFXYZ"]["measurement_noise_cov"] >> R_cv;
        cv::cv2eigen(Q_cv, Q_);
        cv::cv2eigen(R_cv, R_);
        fs.release();
    }
};

class KFYaw {
public:
    KFYaw(const std::string& params_path): x_(2), P_(2, 2), F_(2, 2), Q_(2, 2), R_(1, 1), H_(1, 2) {
        load_params(params_path);
        initialize(0);
    }

    void initialize(const float meas) {
        x_ << meas, 0.0;
        P_.setIdentity();
        P_ *= 1.0;
        H_ << 1.0, 0.0;
        set_output_result();
    }

    void update(const float meas) {
        Eigen::VectorXf z(1);
        z << meas;
        Eigen::VectorXf y = z - H_ * x_;
        Eigen::MatrixXf S = H_ * P_ * H_.transpose() + R_;
        Eigen::MatrixXf K = P_ * H_.transpose() * S.inverse();
        x_ += K * y;
        Eigen::MatrixXf I = Eigen::MatrixXf::Identity(2, 2);
        P_ = (I - K * H_) * P_;
        if (x_(0) < -M_PI) {
            x_(0) += M_PI * 2;
        } else if (x_(0) > M_PI) {
            x_(0) -= M_PI * 2;
        }
        set_output_result();
    }

    void predict(const float time_elapsed) {
        F_.setIdentity();
        F_(0, 1) = time_elapsed;
        x_ = F_ * x_;
        P_ = F_ * P_ * F_.transpose() + Q_;
        set_output_result();
    }

    // 强制更新状态量
    void force_change_yaw(const float meas) {
        x_(0) = meas;
        set_output_result();
    }

    float yaw, palstance;

private:
    Eigen::VectorXf x_; // 状态量 [yaw, palstance]
    Eigen::MatrixXf P_; // 误差协方差矩阵
    Eigen::MatrixXf F_; // 状态转移矩阵
    Eigen::MatrixXf Q_; // 过程噪声协方差
    Eigen::MatrixXf R_; // 测量噪声协方差
    Eigen::MatrixXf H_; // 观测矩阵

    void set_output_result() {
        yaw = x_(0);
        palstance = x_(1);
    }

    void load_params(const std::string& params_path) {
        cv::FileStorage fs(params_path, cv::FileStorage::READ);
        cv::Mat Q_cv, R_cv;
        fs["KFYaw"]["process_noise_cov"] >> Q_cv;
        fs["KFYaw"]["measurement_noise_cov"] >> R_cv;
        cv::cv2eigen(Q_cv, Q_);
        cv::cv2eigen(R_cv, R_);
        fs.release();
    }
};

class UKFXY {
public:
    UKFXY(const std::string& params_path): lambda(3 - state_dim), alpha(1e-1), beta(2), kappa(0) {
        load_params(params_path);
        initialize(cv::Point2f(0, 0));
    }

    void initialize(const cv::Point2f& position) {
        x = Eigen::VectorXf::Zero(state_dim);
        x << position.x, position.y, 0, 0;
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

        set_output_result();
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