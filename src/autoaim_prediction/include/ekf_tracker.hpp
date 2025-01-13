// 从 https://github.com/SnocrashWang/WMJAimer 抄来的

#pragma once

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

namespace ekf_tracker {

double rad_period_correction(double rad) {
    return rad + round((-rad) / (2 * M_PI)) * (2 * M_PI);
}
double r2d(double rad) {
    return rad * 180.0 / M_PI;
}
double d2r(double deg) {
    return deg * M_PI / 180.0;
}
double get_distance(const cv::Point2f& p) {
    return sqrt(p.x * p.x + p.y * p.y);
}
double get_distance(const cv::Point3f& p) {
    return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

enum class TRACKER_STATUS { CONVERGING, TRACKING, TEMP_LOST, LOST };

// 装甲板参数
class Armor {
public:
    cv::Point3f position; // 三维坐标信息
    double yaw_angle; // 按yaw轴旋转的角度。单位: rad
    int area; // 图像中的可视面积。用于排序装甲板
};
using Armors = std::vector<Armor>;

// 整车的状态量
class State {
public:
    State();

    virtual Armors predict_armors(const double predict_time) const = 0;
    virtual Armor predict_closest_armor(const double predict_time, const double switch_threshold) const = 0;
    virtual Armor predict_facing_armor(const double predict_time) const = 0;

    virtual void print(const std::string& tag) const = 0;

    cv::Point2d center; // 旋转中心位置
    double height[2]; // 装甲高度。0对应索引0、2，1对应索引1、3
    double radius[2]; // 旋转半径。0对应索引0、2，1对应索引1、3
    double phase; // 角度（并非定值）
    double palstance; // 角速度

    int total_armor_count; // 装甲板数量
};

// 普通车的状态量
class StandardState: public State {
public:
    StandardState();
    StandardState(const Eigen::Matrix<double, 10, 1>& X);

    /**
        @brief 获取该运动状态下所有装甲板
        @param predict_time 预测时间
    */
    Armors predict_armors(const double predict_time) const override;

    /**
        @brief 获取目标装甲板
        @param predict_time 预测时间
        @param switch_threshold 更新装甲板切换的最小距离差
    */
    Armor predict_closest_armor(const double predict_time, const double switch_threshold) const override;

    /**
        @brief 获取正对的装甲板
    */
    Armor predict_facing_armor(const double predict_time) const override;

    /**
        @brief 打印信息
    */
    void print(const std::string& tag) const override;

    cv::Point2d velocity; // 旋转中心速度
};

State::State() {
    height[0] = height[1] = 0;
    radius[0] = radius[1] = 0;
}

StandardState::StandardState(): StandardState(Eigen::Matrix<double, 10, 1>::Zero()) {}
StandardState::StandardState(const Eigen::Matrix<double, 10, 1>& X): State() {
    total_armor_count = 4;
    center = cv::Point2d(X(0, 0), X(2, 0));
    velocity = cv::Point2d(X(1, 0), X(3, 0));
    height[0] = X(4, 0);
    height[1] = X(5, 0);
    radius[0] = X(6, 0);
    radius[1] = X(7, 0);
    phase = X(8, 0);
    palstance = X(9, 0);
}

Armors StandardState::predict_armors(const double predict_time) const {
    Armors armors(total_armor_count);
    for (int i = 0; i < total_armor_count; i++) {
        double angle = rad_period_correction(phase + palstance * predict_time + i * M_PI / 2);
        cv::Point2d point = center + velocity * predict_time
            + cv::Point2d(radius[i % 2] * cos(angle), radius[i % 2] * sin(angle));
        armors[i].position = cv::Point3d(point.x, point.y, height[i % 2]);
        armors[i].yaw_angle = angle;
    }
    return armors;
}

Armor StandardState::predict_closest_armor(const double predict_time, const double switch_threshold) const {
    /**
        我们选择在给定预测时间后在相机系中yaw角度绝对值最小，即面朝我方向最正的一块装甲板作为击打目标
        为了防止数据抖动使得目标在两块角度相近的装甲板之间来回跳变，在取最小值时额外加一小段阈值，即需要比原有的最小值减去该阈值更小才认为是新的最小值
        但是，这样做会导致目标切换存在一定延迟，所以给求最小值时的时间也额外加一段提前量，即预测其在该一段时间后可能会切换目标，这个时间通过切换阈值和当前的角速度计算得到
        另外，为了防止角速度接近0时计算得一个过大的提前量，将上述结果与0.2取较小值
    */
    double switch_advanced_time = std::min(0.2, d2r(switch_threshold) / abs(palstance));
    Armors armors = predict_armors(predict_time + switch_advanced_time);
    cv::Point2d predict_center = center + velocity * predict_time;
    int min_index = 0;
    for (int i = 0; i < total_armor_count; i++) {
        if (abs(rad_period_correction(
                M_PI + armors[i].yaw_angle - atan2(predict_center.y, predict_center.x)
            )) + d2r(switch_threshold)
            < abs(rad_period_correction(
                M_PI + armors[min_index].yaw_angle - atan2(predict_center.y, predict_center.x)
            )))
        {
            min_index = i;
        }
    }
    return predict_armors(predict_time)[min_index];
}

Armor StandardState::predict_facing_armor(const double predict_time) const {
    Armor armor;
    Armors armors = predict_armors(predict_time);
    cv::Point2d predict_center = center + velocity * predict_time;
    int min_index = 0;
    for (int i = 0; i < total_armor_count; i++) {
        if (abs(rad_period_correction(
                M_PI + armors[i].yaw_angle - atan2(predict_center.y, predict_center.x)
            ))
            < abs(rad_period_correction(
                M_PI + armors[min_index].yaw_angle - atan2(predict_center.y, predict_center.x)
            )))
        {
            min_index = i;
        }
    }

    // 如果最近的装甲板处于远离状态，则击打下一装甲板
    if (abs(rad_period_correction(
            M_PI + armors[min_index].yaw_angle + (palstance / abs(palstance)) * d2r(-50)
            - atan2(predict_center.y, predict_center.x)
        ))
        < abs(rad_period_correction(
            M_PI + armors[min_index].yaw_angle - atan2(predict_center.y, predict_center.x)
        )))
    {
        min_index = (min_index + 1) % total_armor_count;
    }
    double angle = rad_period_correction(atan2(predict_center.y, predict_center.x) - M_PI);
    armor.position = cv::Point3d(
        predict_center.x + radius[min_index % 2] * cos(angle),
        predict_center.y + radius[min_index % 2] * sin(angle),
        height[min_index % 2]
    );
    armor.yaw_angle = angle;
    return armor;
}

void StandardState::print(const std::string& tag) const {
    std::string prefix = "[StandardState] " + tag + " ";
    std::cout << prefix << "center: " << center << " += " << velocity << std::endl;
    std::cout << prefix << "height: " << height[0] << " " << height[1] << std::endl;
    std::cout << prefix << "radius: " << radius[0] << " " << radius[1] << std::endl;
    std::cout << prefix << "angle: " << phase << " += " << palstance << std::endl;
}

class EKF {
public:
    virtual void initialize(const Armors& armors) = 0;
    virtual std::pair<Eigen::MatrixXd, Eigen::MatrixXd> predict(const Armors& armors) = 0;
    virtual std::shared_ptr<State> update(
        const Armors& armors,
        const Eigen::MatrixXd& X_,
        const Eigen::MatrixXd& P_,
        std::map<int, int>& match
    ) = 0;

    virtual void reset() = 0;

protected:
    virtual void load_params(const std::string& file_path) = 0;

    virtual Eigen::MatrixXd get_predictive_measurement(const Eigen::MatrixXd& X, int i) = 0;
    virtual Eigen::MatrixXd get_measurement_PD(const Eigen::MatrixXd& X, int i) = 0;
    virtual Eigen::MatrixXd get_measure_noise_PD(const Eigen::MatrixXd& X, int i) = 0;

    double m_dt; // 单位时间
    double m_init_radius; // 初始半径
    double m_gain; // 测距噪声关于角度的增益倍数

    double m_process_noise[4]; // 状态转移噪声系数
    double m_measure_noise[3]; // 观测噪声系数

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m_X; // 状态
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m_X_update; // 状态修正
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m_P; // 状态协方差
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m_F; // 状态转移矩阵
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m_Q; // 状态转移噪声
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m_R; // 观测噪声
};

class StandardEKF: public EKF {
public:
    StandardEKF(const std::string& param_path);

    /**
        @brief 取排序后第一块装甲板进行初始化
        @param armors 绝对坐标系装甲板序列
    */
    void initialize(const Armors& armors) override;

    /**
        @brief 先验预测
        @param armors 绝对坐标系下的装甲板序列
        @return std::pair<Eigen::Matrix<double, 10, 1>, Eigen::Matrix<double, 10, 10>>(X_, P_)
    */
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> predict(const Armors& armors) override;

    /**
        @brief 匹配装甲板序列和标准装甲板
        @param armors 绝对坐标系装甲板序列
        @param X_ Eigen::Matrix<double, 10, 1> 先验状态
        @param P_ Eigen::Matrix<double, 10, 10> 先验状态协方差
        @param match 装甲板关联
        @return 运动状态
    */
    std::shared_ptr<State> update(
        const Armors& armors,
        const Eigen::MatrixXd& X_,
        const Eigen::MatrixXd& P_,
        std::map<int, int>& match
    ) override;

    /**
        @brief 重置
    */
    void reset() override;

private:
    /**
        @brief 读取参数配置文件
        @param file_path 配置文件路径
    */
    void load_params(const std::string& file_path) override;

    /**
        @brief 获取先验观测量
        @param X Eigen::Matrix<double, 10, 1> 先验状态
        @param i 标准装甲板索引
        @return Eigen::Matrix<double, 4, 1> 先验观测矩阵
    */
    Eigen::MatrixXd get_predictive_measurement(const Eigen::MatrixXd& X, int i) override;

    /**
        @brief 获取观测方程偏导矩阵
        @param X Eigen::Matrix<double, 10, 1> 先验状态
        @param i 标准装甲板索引
        @return Eigen::Matrix<double, 4, 10> 观测偏导矩阵
    */
    Eigen::MatrixXd get_measurement_PD(const Eigen::MatrixXd& X, int i) override;

    /**
        @brief 获取观测噪声偏导矩阵
        @param X Eigen::Matrix<double, 10, 1> 先验状态
        @param i 标准装甲板索引
        @return Eigen::Matrix<double, 4, 4> 观测偏导矩阵
    */
    Eigen::MatrixXd get_measure_noise_PD(const Eigen::MatrixXd& X, int i) override;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m_RR; // 增广观测噪声
};

StandardEKF::StandardEKF(const std::string& params_path): EKF() {
    load_params(params_path);
    reset();

    m_F << 
        1, m_dt, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 1, m_dt, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 1, m_dt, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1;

    double dd = m_process_noise[0];
    double da = m_process_noise[1];
    double dz = m_process_noise[2];
    double dr = m_process_noise[3];
    double t4 = pow(m_dt, 3) / 3;
    double t3 = pow(m_dt, 2) / 2;
    double t2 = pow(m_dt, 1);

    m_Q << 
        t4 * dd, t3 * dd, 0, 0, 0, 0, 0, 0, 0, 0, 
        t3 * dd, t2 * dd, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, t4 * dd, t3 * dd, 0, 0, 0, 0, 0, 0, 
        0, 0, t3 * dd, t2 * dd, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, dz, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, dz, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, dr, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, dr, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, t4 * da, t3 * da, 
        0, 0, 0, 0, 0, 0, 0, 0, t3 * da, t2 * da;

    Eigen::VectorXd measurement_noise4(4);
    measurement_noise4 << m_measure_noise[0], m_measure_noise[0], m_measure_noise[1],
        m_measure_noise[2];
    m_R = measurement_noise4.asDiagonal();
    Eigen::VectorXd measurement_noise8(8);
    measurement_noise8 << measurement_noise4, measurement_noise4;
    m_RR = measurement_noise8.asDiagonal();
}

void StandardEKF::load_params(const std::string& file_path) {
    cv::FileStorage fs(file_path, cv::FileStorage::READ);
    fs["dt"] >> m_dt;

    fs["StandardEKF"]["init_radius"] >> m_init_radius;
    fs["StandardEKF"]["gain"] >> m_gain;

    fs["StandardEKF"]["process_noise"]["displace_high_diff"] >> m_process_noise[0];
    fs["StandardEKF"]["process_noise"]["anglar_high_diff"] >> m_process_noise[1];
    fs["StandardEKF"]["process_noise"]["height"] >> m_process_noise[2];
    fs["StandardEKF"]["process_noise"]["radius"] >> m_process_noise[3];

    fs["StandardEKF"]["measure_noise"]["pose"] >> m_measure_noise[0];
    fs["StandardEKF"]["measure_noise"]["distance"] >> m_measure_noise[1];
    fs["StandardEKF"]["measure_noise"]["angle"] >> m_measure_noise[2];
    fs.release();

    m_X.resize(10, 1);
    m_X_update.resize(10, 1);
    m_P.resize(10, 10);
    m_F.resize(10, 10);
    m_Q.resize(10, 10);
    m_R.resize(4, 4);
    m_RR.resize(8, 8);
}

void StandardEKF::reset() {
    m_P = Eigen::Matrix<double, 10, 10>::Identity() * 1e-5;
}

void StandardEKF::initialize(const Armors& armors) {
    reset();
    // 取排序后第一块装甲板计算初始化状态
    m_X << armors[0].position.x - m_init_radius * cos(armors[0].yaw_angle), 0,
           armors[0].position.y - m_init_radius * sin(armors[0].yaw_angle), 0, armors[0].position.z,
           armors[0].position.z, m_init_radius, m_init_radius, armors[0].yaw_angle, 0;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> StandardEKF::predict(const Armors& armors) {
    Eigen::MatrixXd X_ = m_F * m_X;
    Eigen::MatrixXd P_ = m_F * m_P * m_F.transpose() + m_Q;
    return std::make_pair(X_, P_);
}

std::shared_ptr<State> StandardEKF::update(
    const Armors& armors,
    const Eigen::MatrixXd& X_,
    const Eigen::MatrixXd& P_,
    std::map<int, int>& match
) {
    // 观测
    Eigen::MatrixXd Z; // 实际观测量
    Eigen::MatrixXd h; // 先验观测量
    Eigen::MatrixXd H; // 观测方程偏导矩阵
    Eigen::MatrixXd V; // 观测噪声偏导矩阵
    Eigen::MatrixXd K; // 置信度权重矩阵
    Eigen::MatrixXd R; // 时变观测噪声矩阵

    if (match.size() == 1) {
        int num = match.begin()->first;
        Z.resize(4, 1);
        Z << armors[num].position.x, armors[num].position.y, armors[num].position.z, armors[num].yaw_angle;
        h.resize(4, 1);
        h << get_predictive_measurement(X_, match[num]);
        H.resize(4, 10);
        H << get_measurement_PD(X_, match[num]);
        V.resize(4, 4);
        V << get_measure_noise_PD(X_, match[num]);
        R = m_R;
        R(2, 2) *= m_gain * abs(Z(3, 0));

        K = P_ * H.transpose() * ((H * P_ * H.transpose() + V * R * V.transpose())).inverse();
    } else if (match.size() == 2) {
        int num1 = match.begin()->first;
        int num2 = (++match.begin())->first;
        Z.resize(8, 1);
        Z << armors[num1].position.x, armors[num1].position.y, armors[num1].position.z, armors[num1].yaw_angle, 
             armors[num2].position.x, armors[num2].position.y, armors[num2].position.z, armors[num2].yaw_angle;
        h.resize(8, 1);
        h << get_predictive_measurement(X_, match[num1]), get_predictive_measurement(X_, match[num2]);
        H.resize(8, 10);
        H << get_measurement_PD(X_, match[num1]), get_measurement_PD(X_, match[num2]);
        V.resize(8, 8);
        V << get_measure_noise_PD(X_, match[num1]), Eigen::MatrixXd::Zero(4, 4),
             Eigen::MatrixXd::Zero(4, 4), get_measure_noise_PD(X_, match[num2]);
        R = m_RR;
        R(2, 2) *= m_gain * abs(Z(3, 0));
        R(6, 6) *= m_gain * abs(Z(7, 0));

        K = P_ * H.transpose() * ((H * P_ * H.transpose() + V * R * V.transpose())).inverse();
    } else {
        // 若匹配数目不为1或2，则认为丢识别或者误判，直接返回先验状态
        m_X = X_;
        m_P = P_;
        return std::make_shared<StandardState>(m_X);
    }

    // 更新
    Eigen::MatrixXd tmp = Z - h;
    for (int i = 0; i < match.size(); i++) {
        tmp(3 + i * 4, 0) = rad_period_correction(tmp(3 + i * 4, 0));
    }
    m_X_update = K * tmp;

    m_X = X_ + m_X_update;
    m_P = (Eigen::Matrix<double, 10, 10>::Identity() - K * H) * P_;

    return std::make_shared<StandardState>(m_X);
}

Eigen::MatrixXd StandardEKF::get_predictive_measurement(const Eigen::MatrixXd& X, int i) {
    Eigen::Matrix<double, 4, 1> h;
    h << X(0, 0) + X(6 + i % 2, 0) * cos(X(8, 0) + i * M_PI / 2),
         X(2, 0) + X(6 + i % 2, 0) * sin(X(8, 0) + i * M_PI / 2), 
         X(4 + i % 2, 0),
         X(8, 0) + i * M_PI / 2;
    return h;
}

Eigen::MatrixXd StandardEKF::get_measurement_PD(const Eigen::MatrixXd& X, int i) {
    Eigen::Matrix<double, 4, 10> H;
    H << 1, 0, 0, 0, 0, 0, ((i + 1) % 2) * cos(X(8, 0) + i * M_PI / 2), (i % 2) * cos(X(8, 0) + i * M_PI / 2), -X(6 + i % 2, 0) * sin(X(8, 0) + i * M_PI / 2), 0, 0,
         0, 1, 0, 0, 0, ((i + 1) % 2) * sin(X(8, 0) + i * M_PI / 2), (i % 2) * sin(X(8, 0) + i * M_PI / 2), X(6 + i % 2, 0) * cos(X(8, 0) + i * M_PI / 2), 0, 0,
         0, 0, 0, (i + 1) % 2, i % 2, 0, 0, 0, 0, 
         0, 0, 0, 0, 0, 0, 0, 0, 1, 0;
    return H;
}

Eigen::MatrixXd StandardEKF::get_measure_noise_PD(const Eigen::MatrixXd& X, int i) {
    Eigen::Matrix<double, 4, 1> h = get_predictive_measurement(X, i);
    double psi = atan2(h(1, 0), h(0, 0));
    double phi = atan2(h(2, 0), sqrt(pow(h(0, 0), 2) + pow(h(1, 0), 2)));
    double d = sqrt(pow(h(0, 0), 2) + pow(h(1, 0), 2) + pow(h(2, 0), 2));
    Eigen::Matrix<double, 4, 4> V;
    V << -d * cos(phi) * sin(psi), -d * sin(phi) * cos(psi), cos(phi) * cos(psi), 0, 
         d * cos(phi) * cos(psi), -d * sin(phi) * sin(psi), cos(phi) * sin(psi), 0, 
         0, d * cos(phi), sin(phi), 0, 
         0, 0, 0, 1;
    return V;
}

class StandardEKFTracker {
public:
    StandardEKFTracker(const std::string& params_path);
    void push(const geometry_msgs::msg::Transform& transform, const int armor_area);
    void update();
    cv::Point3f get_prediction(const double bullet_speed, const double t_delay);

    TRACKER_STATUS tracker_status = TRACKER_STATUS::LOST;

private:
    void load_params(const std::string& file_path);
    void ekf_initialize();
    void ekf_predict();
    void ekf_update();
    Eigen::MatrixXd get_score_mat(const Armors& detect_armors, const Armors& standard_armors);
    std::map<int, int> get_match(const Eigen::MatrixXd& score_mat, double score_thres, int m);
    void get_combination_numbers(
        std::vector<int>& input,
        std::vector<int>& tmp_v,
        std::vector<std::vector<int>>& result,
        int start,
        int k
    );
    const std::shared_ptr<State> get_state() const;

    const unsigned int MAX_LOST_FRAMES = 5;
    const unsigned int CONVERGE_FRAMES = 5;
    double m_switch_threshold_ = 15;
    double m_match_score_threshold_ = 1.0;

    unsigned int tracking_frames_;
    unsigned int lost_frames_;

    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> prior_mats_;
    std::shared_ptr<State> prior_state_;
    std::shared_ptr<State> posterior_state_;
    std::shared_ptr<StandardEKF> ekf_;

    Armors armors_;
};

StandardEKFTracker::StandardEKFTracker(const std::string& params_path) {
    load_params(params_path);
    ekf_ = std::make_shared<StandardEKF>(params_path);
}

void StandardEKFTracker::push(
    const geometry_msgs::msg::Transform& transform,
    const int armor_area
) {
    Armor armor;
    armor.position =
        cv::Point3f(transform.translation.y, -transform.translation.x, transform.translation.z);
    tf2::Quaternion quaternion(
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
        transform.rotation.w
    );
    tf2::Matrix3x3 rotation_mat(quaternion);
    tf2::Vector3 y_vec = rotation_mat.getColumn(1);
    armor.yaw_angle = rad_period_correction(-atan(y_vec.getX() / y_vec.getY()) + M_PI);
    armor.area = armor_area;
    armors_.emplace_back(armor);
}

void StandardEKFTracker::update() {
    using TS = TRACKER_STATUS;
    if (armors_.empty() || armors_.size() > 2) {
        tracking_frames_ = 0;
        lost_frames_++;
        if (tracker_status != TS::LOST) {
            ekf_predict();
            if (lost_frames_ >= MAX_LOST_FRAMES) {
                tracker_status = TS::LOST;
            } else {
                tracker_status = TS::TEMP_LOST;
            }
        }
    } else {
        lost_frames_ = 0;
        tracking_frames_++;
        if (tracker_status == TS::LOST) {
            ekf_initialize();
            ekf_predict();
        } else {
            ekf_predict();
            ekf_update();
        }
        if (tracking_frames_ >= CONVERGE_FRAMES) {
            tracker_status = TS::TRACKING;
        } else {
            tracker_status = TS::CONVERGING;
        }
    }
    armors_.clear();
}

cv::Point3f StandardEKFTracker::get_prediction(const double bullet_speed, const double t_delay) {
    const auto state = get_state();
    assert(state != nullptr); // 不应该在tracker_status为LOST时调用get_prediction
    // 一阶线性化，粗略估计击打时间。
    const double hit_time =
        get_distance(state->predict_closest_armor(0, 0).position) / bullet_speed + t_delay;
    const cv::Point3f p = state->predict_closest_armor(hit_time, m_switch_threshold_).position;
    return cv::Point3f(-p.y, p.x, p.z);
}

void StandardEKFTracker::load_params(const std::string& file_path) {
    cv::FileStorage fs(file_path, cv::FileStorage::READ);
    fs["StandardEKFTracker"]["switch_threshold"] >> m_switch_threshold_;
    fs["StandardEKFTracker"]["match_score_threshold"] >> m_match_score_threshold_;
}

void StandardEKFTracker::ekf_initialize() {
    ekf_->initialize(armors_);
}

void StandardEKFTracker::ekf_predict() {
    std::sort(armors_.begin(), armors_.end(), [&](const Armor& a, const Armor& b) {
        return a.area > b.area;
    });
    prior_mats_ = ekf_->predict(armors_);
    prior_state_ = std::make_shared<StandardState>(prior_mats_.first);
}

void StandardEKFTracker::ekf_update() {
    Eigen::MatrixXd score = get_score_mat(armors_, prior_state_->predict_armors(0));
    std::map<int, int> match =
        get_match(score, m_match_score_threshold_, prior_state_->total_armor_count);
    posterior_state_ = ekf_->update(armors_, prior_mats_.first, prior_mats_.second, match);
}

const std::shared_ptr<State> StandardEKFTracker::get_state() const {
    if (tracker_status == TRACKER_STATUS::LOST) {
        return nullptr;
    } else if (tracker_status == TRACKER_STATUS::TEMP_LOST) {
        return prior_state_;
    } else {
        return posterior_state_;
    }
}

Eigen::MatrixXd
StandardEKFTracker::get_score_mat(const Armors& detect_armors, const Armors& standard_armors) {
    int m = detect_armors.size();
    int n = standard_armors.size();
    // 计算两组装甲板之间的坐标差和角度差两个负向指标
    Eigen::Matrix<double, Eigen::Dynamic, 2> negative_score;
    negative_score.resize(m * n, 2);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            negative_score(i * n + j, 0) =
                get_distance(detect_armors[i].position - standard_armors[j].position);
            negative_score(i * n + j, 1) =
                abs(rad_period_correction(detect_armors[i].yaw_angle - standard_armors[j].yaw_angle));
        }
    }

    // 数据标准化
    Eigen::Matrix<double, Eigen::Dynamic, 2> regular_score;
    regular_score.resize(m * n, 2);
    for (int i = 0; i < regular_score.rows(); i++) {
        regular_score(i, 0) = (negative_score.col(0).maxCoeff() - negative_score(i, 0))
            / (negative_score.col(0).maxCoeff() - negative_score.col(0).minCoeff());
        regular_score(i, 1) = (negative_score.col(1).maxCoeff() - negative_score(i, 1))
            / (negative_score.col(1).maxCoeff() - negative_score.col(1).minCoeff());
    }

    // 计算样本值占指标的比重
    Eigen::Matrix<double, Eigen::Dynamic, 2> score_weight;
    score_weight.resize(m * n, 2);
    Eigen::VectorXd col_sum = regular_score.colwise().sum();
    for (int i = 0; i < score_weight.rows(); i++) {
        score_weight(i, 0) = regular_score(i, 0) / col_sum(0);
        score_weight(i, 1) = regular_score(i, 1) / col_sum(1);
    }

    // 计算每项指标的熵值
    Eigen::Vector2d entropy = Eigen::Vector2d::Zero();
    for (int i = 0; i < score_weight.rows(); i++) {
        if (score_weight(i, 0) != 0)
            entropy(0) -= score_weight(i, 0) * log(score_weight(i, 0));
        if (score_weight(i, 1) != 0)
            entropy(1) -= score_weight(i, 1) * log(score_weight(i, 1));
    }
    entropy /= log(score_weight.rows());

    // 计算权重
    Eigen::Vector2d weight = (Eigen::Vector2d::Ones() - entropy) / (2 - entropy.sum());

    // 计算匹配得分矩阵
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> score;
    score.resize(m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (i < detect_armors.size() && j < standard_armors.size()) {
                score(i, j) = negative_score.row(i * standard_armors.size() + j) * weight;
            }
        }
    }

    return score;
}

std::map<int, int>
StandardEKFTracker::get_match(const Eigen::MatrixXd& matrix, double score_thres, int m) {
    std::map<int, int> row_col; // 存储最终结果，row_col[i]=j表示矩阵的第i行第j列是要选取的数
    for (int i = 0; i < matrix.rows(); i++) {
        // 值为-1表示不选择该行中的任何数
        row_col[i] = -1;
    }

    double min = 0, tmp = 0;
    for (int i = 0; i < matrix.rows() && i < m; i++) {
        min += matrix(i, i);
        row_col[i] = i;
    }

    std::vector<int> row; // 数列中的每个数代表矩阵的每一行
    for (int i = 0; i < matrix.rows(); i++) {
        // 有多少行，行数列就有多少个数
        row.emplace_back(i);
    }
    std::vector<int> col; // 数列中的每个数代表矩阵的每一列
    for (int i = 0; i < m; i++) {
        col.emplace_back(i);
    }
    std::vector<int> tmp_v; // 存储C(n,k)的中间结果
    std::vector<std::vector<int>> result; // 存储C(n,k)的结果
    std::vector<std::vector<int>> nAfour; // 存储A(n,4)的结果
    std::vector<std::vector<int>> fourAfour; // 存储A(4,4)的结果

    // 用A(n,m)求出可能选择的所有行的组合
    // 因为C(n,m) * A(m,m) = A(n,m)，所以先计算C(n,m)，再计算A(m,m)，最后将结果存到result中
    // 计算C(n,m)
    if (matrix.rows() <= m)
        get_combination_numbers(row, tmp_v, result, 0, matrix.rows());
    else
        get_combination_numbers(row, tmp_v, result, 0, m);

    // 使用上一步的结果计算A(m,m)，最终得到A(n,m)
    for (auto it = result.begin(); it != result.end(); it++) {
        do {
            nAfour.emplace_back(*it);
        } while (std::next_permutation(it->begin(), it->end()));
    }

    // 用A(m,m)求出可能选择的所有列的组合计算A(m,m)
    do {
        fourAfour.push_back(col);
    } while (std::next_permutation(col.begin(), col.end()));

    // 用A(n,m)和A(m,m)求出可能选择的所有行列的组合的结果
    for (auto it = nAfour.begin(); it != nAfour.end(); it++) {
        for (auto it2 = fourAfour.begin(); it2 != fourAfour.end(); it2++) {
            // 对于每一种组合，都求一遍它们的值，与min比较，如果小于min，则更新min
            tmp = 0;
            for (int i = 0; i < it->size(); i++)
                tmp += matrix((*it)[i], (*it2)[i]);
            if (tmp < min) {
                min = tmp;
                // 重新初始化map
                row_col.clear();
                // 每选择矩阵中的一个数时，就将该位置记录下来存到map中
                for (int i = 0; i < it->size(); i++) {
                    row_col[it->at(i)] = it2->at(i);
                }
            }
        }
    }

    // 删除-1的键值对
    for (auto it = row_col.begin(); it != row_col.end();) {
        if (it->second == -1 || (it->second != -1 && matrix(it->first, it->second) > score_thres)) {
            it = row_col.erase(it);
        } else {
            ++it;
        }
    }
    return row_col;
}

void StandardEKFTracker::get_combination_numbers(
    std::vector<int>& input,
    std::vector<int>& tmp_v,
    std::vector<std::vector<int>>& result,
    int start,
    int k
) {
    for (int i = start; i < input.size(); i++) {
        tmp_v.emplace_back(input[i]);
        if (tmp_v.size() == k) {
            result.emplace_back(tmp_v);
            tmp_v.pop_back();
            continue;
        }
        // 使用递归计算
        get_combination_numbers(input, tmp_v, result, i + 1, k);
        tmp_v.pop_back();
    }
}

} // namespace ekf_tracker