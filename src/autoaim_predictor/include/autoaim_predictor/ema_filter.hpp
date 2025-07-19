#pragma once

#include <eigen3/Eigen/Dense>

template<unsigned D>
class EMAF {
public:
    explicit EMAF(const float filter_ratio);
    void reset();
    void initialize(const Eigen::Vector<float, D>& val);
    void update(const Eigen::Vector<float, D>& meas);
    void force_change_value(const Eigen::Vector<float, D>& val);
    Eigen::Vector<float, D> value();
    
private:
    Eigen::Vector<float, D> value_;
    const float filter_ratio_ = -1; // 介于0-1之间，越大越稳定
};