#include <autoaim_predictor/ema_filter.hpp>

template<unsigned D>
EMAF<D>::EMAF(const float filter_ratio): filter_ratio_(filter_ratio) {
    reset();
}

template<unsigned D>
void EMAF<D>::reset() {
    initialize(Eigen::Vector<float, D>::Zero());
}

template<unsigned D>
void EMAF<D>::initialize(const Eigen::Vector<float, D>& val) {
    value_ = val;
}

template<unsigned D>
void EMAF<D>::update(const Eigen::Vector<float, D>& meas) {
    value_ = filter_ratio_ * value_ + (1 - filter_ratio_) * meas;
}

template<unsigned D>
void EMAF<D>::force_change_value(const Eigen::Vector<float, D>& val) {
    value_ = val;
}

template<unsigned D>
Eigen::Vector<float, D> EMAF<D>::value() {
    return value_;
}

template class EMAF<1>;
template class EMAF<3>;