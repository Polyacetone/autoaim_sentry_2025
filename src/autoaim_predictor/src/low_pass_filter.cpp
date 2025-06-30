#include <low_pass_filter.hpp>

template<unsigned D>
LPF<D>::LPF(const float filter_ratio): filter_ratio_(filter_ratio) {}

template<unsigned D>
void LPF<D>::reset() {
    initialize(Eigen::Vector<float, D>::Zero());
}

template<unsigned D>
void LPF<D>::initialize(const Eigen::Vector<float, D>& val) {
    value_ = val;
}

template<unsigned D>
void LPF<D>::update(const Eigen::Vector<float, D>& meas) {
    value_ = filter_ratio_ * value_ + (1 - filter_ratio_) * meas;
}

template<unsigned D>
void LPF<D>::force_change_value(const Eigen::Vector<float, D>& val) {
    value_ = val;
}

template<unsigned D>
Eigen::Vector<float, D> LPF<D>::value() {
    return value_;
}

template class LPF<1>;
template class LPF<3>;