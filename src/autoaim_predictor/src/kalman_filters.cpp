#include <kalman_filters.hpp>

KFXYZ::KFXYZ(const std::string& params_path):
    x_(6),
    P_(6, 6),
    F_(6, 6),
    Q_(6, 6),
    R_(3, 3),
    H_(3, 6) {
    load_params(params_path);
    initialize(cv::Point3f(0, 0, 0));
}

void KFXYZ::initialize(const cv::Point3f& meas) {
    x_ << meas.x, meas.y, meas.z, 0, 0, 0;
    P_.setIdentity();
    P_ *= 1.0;
    H_ << 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0;
    set_output_result();
    frame_count_ = 0;
}

void KFXYZ::update(const cv::Point3f& meas) {
    Eigen::Vector3f z(meas.x, meas.y, meas.z);
    Eigen::VectorXf y = z - H_ * x_;
    Eigen::MatrixXf S = H_ * P_ * H_.transpose() + R_;
    Eigen::MatrixXf K = P_ * H_.transpose() * S.inverse();
    x_ += K * y;
    P_ = (Eigen::MatrixXf::Identity(6, 6) - K * H_) * P_;
    if (ENABLE_ADAPTIVE && frame_count_ >= ADAPTIVE_START_FRAME) {
        // Sage-Husa自适应更新测量噪声R
        Eigen::MatrixXf H_P_Ht = H_ * P_ * H_.transpose();
        Eigen::MatrixXf y_outer = y * y.transpose();
        Eigen::MatrixXf R_estimate = y_outer - H_P_Ht;
        R_estimate = (R_estimate + R_estimate.transpose()) / 2.0f;
        ensure_positive_semi_definite(R_estimate);
        R_ = (1 - ADAPTIVE_FORGET_FACTOR) * R_ + ADAPTIVE_FORGET_FACTOR * R_estimate;
        ensure_positive_semi_definite(R_);
    }
    set_output_result();
    frame_count_++;
}

void KFXYZ::predict(float delta_t) {
    F_.setIdentity();
    F_(0, 3) = delta_t;
    F_(1, 4) = delta_t;
    F_(2, 5) = delta_t;
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
    set_output_result();
}

void KFXYZ::force_change_position(const cv::Point3f& meas) {
    x_(0) = meas.x;
    x_(1) = meas.y;
    x_(2) = meas.z;
    set_output_result();
}

void KFXYZ::set_output_result() {
    position.x = x_(0);
    position.y = x_(1);
    position.z = x_(2);
    velocity.x = x_(3);
    velocity.y = x_(4);
    velocity.z = x_(5);
}

void KFXYZ::ensure_positive_semi_definite(Eigen::MatrixXf& matrix) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(matrix);
    if (solver.info() != Eigen::Success)
        return;
    Eigen::VectorXf eigenvalues = solver.eigenvalues();
    Eigen::MatrixXf eigenvectors = solver.eigenvectors();
    matrix = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
    matrix = (matrix + matrix.transpose()) / 2.0f;
}

void KFXYZ::load_params(const std::string& params_path) {
    cv::FileStorage fs(params_path, cv::FileStorage::READ);
    cv::Mat Q_cv, R_cv;
    fs["KFXYZ"]["process_noise_cov"] >> Q_cv;
    fs["KFXYZ"]["measurement_noise_cov"] >> R_cv;
    cv::cv2eigen(Q_cv, Q_);
    cv::cv2eigen(R_cv, R_);
    ENABLE_ADAPTIVE = (int)fs["KFXYZ"]["enable_adaptive"] != 0;
    ADAPTIVE_FORGET_FACTOR = (float)fs["KFXYZ"]["adaptive_forget_factor"];
    ADAPTIVE_START_FRAME = (int)fs["KFXYZ"]["adaptive_start_frame"];
    fs.release();
}

KFYaw::KFYaw(const std::string& params_path):
    x_(2),
    P_(2, 2),
    F_(2, 2),
    Q_(2, 2),
    R_(1, 1),
    H_(1, 2) {
    load_params(params_path);
    initialize(0);
}

void KFYaw::initialize(const float meas) {
    x_ << meas, 0.0;
    P_.setIdentity();
    P_ *= 1.0;
    H_ << 1.0, 0.0;
    set_output_result();
}

void KFYaw::update(const float meas) {
    Eigen::VectorXf z(1);
    z << meas;
    Eigen::VectorXf y = z - H_ * x_;
    Eigen::MatrixXf S = H_ * P_ * H_.transpose() + R_;
    Eigen::MatrixXf K = P_ * H_.transpose() * S.inverse();
    x_ += K * y;
    Eigen::MatrixXf I = Eigen::MatrixXf::Identity(2, 2);
    P_ = (I - K * H_) * P_;
    set_output_result();
}

void KFYaw::predict(const float time_elapsed) {
    F_.setIdentity();
    F_(0, 1) = time_elapsed;
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
    set_output_result();
}

void KFYaw::force_change_yaw(const float meas) {
    x_(0) = meas;
    set_output_result();
}

void KFYaw::set_output_result() {
    yaw = x_(0);
    palstance = x_(1);
}

void KFYaw::load_params(const std::string& params_path) {
    cv::FileStorage fs(params_path, cv::FileStorage::READ);
    cv::Mat Q_cv, R_cv;
    fs["KFYaw"]["process_noise_cov"] >> Q_cv;
    fs["KFYaw"]["measurement_noise_cov"] >> R_cv;
    cv::cv2eigen(Q_cv, Q_);
    cv::cv2eigen(R_cv, R_);
    fs.release();
}

UKFXY::UKFXY(const std::string& params_path):
    lambda(3 - state_dim),
    alpha(1e-1),
    beta(2),
    kappa(0) {
    load_params(params_path);
    initialize(cv::Point2f(0, 0));
}

void UKFXY::initialize(const cv::Point2f& position) {
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

void UKFXY::predict(float time_elapsed) {
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

void UKFXY::update(const cv::Point2f& position) {
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

void UKFXY::generate_sigma_points() {
    Eigen::MatrixXf A = P.llt().matrixL();
    sigma_points.col(0) = x;

    for (int i = 0; i < state_dim; i++) {
        sigma_points.col(i + 1) = x + sqrt(lambda + state_dim) * A.col(i);
        sigma_points.col(i + 1 + state_dim) = x - sqrt(lambda + state_dim) * A.col(i);
    }
}

Eigen::VectorXf UKFXY::process_model(const Eigen::VectorXf& x, float time_elapsed) {
    Eigen::VectorXf x_new = x;
    x_new(0) += x(2) * time_elapsed;
    x_new(1) += x(3) * time_elapsed;
    return x_new;
}

Eigen::VectorXf UKFXY::measurement_model(const Eigen::VectorXf& x) {
    return x.head(meas_dim);
}

void UKFXY::load_params(const std::string& params_path) {
    cv::FileStorage fs(params_path, cv::FileStorage::READ);
    cv::Mat process_noise, measurement_noise;
    fs["UKFXY"]["process_noise_cov"] >> process_noise;
    fs["UKFXY"]["measurement_noise_cov"] >> measurement_noise;
    cv::cv2eigen(process_noise, Q);
    cv::cv2eigen(measurement_noise, R);
    fs.release();
}

void UKFXY::set_output_result() {
    position.x = x(0);
    position.y = x(1);
    velocity.x = x(2);
    velocity.y = x(3);
}