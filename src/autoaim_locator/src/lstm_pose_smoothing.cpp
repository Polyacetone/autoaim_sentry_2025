#include <autoaim_locator/lstm_pose_smoothing.hpp>

LSTMPoseSmoothing::LSTMPoseSmoothing(const std::string& model_path) {
    ov::Core core;
    compiled_model_ = core.compile_model(model_path, "CPU");
    infer_request_ = compiled_model_.create_infer_request();
    inputs_ = compiled_model_.inputs();
    outputs_ = compiled_model_.outputs();
    h_ = ov::Tensor(inputs_[1].get_element_type(), inputs_[1].get_shape());
    c_ = ov::Tensor(inputs_[2].get_element_type(), inputs_[2].get_shape());
    clear_hidden_states();
}

float LSTMPoseSmoothing::infer(float yaw0, float yaw1, float err0, float err1, float dt) {
    if (dt > 0.5) {
        clear_hidden_states();
        return yaw0;
    }
    std::array<float, 5> feature = {yaw0, yaw1, err0, err1, dt};
    ov::Tensor feature_tensor(inputs_[0].get_element_type(), inputs_[0].get_shape());
    std::copy(feature.begin(), feature.end(), feature_tensor.data<float>());
    infer_request_.set_input_tensor(0, feature_tensor);
    infer_request_.set_input_tensor(1, h_);
    infer_request_.set_input_tensor(2, c_);
    infer_request_.infer();
    ov::Tensor yaw_pred = infer_request_.get_output_tensor(0);
    h_ = infer_request_.get_output_tensor(1);
    c_ = infer_request_.get_output_tensor(2);
    return *yaw_pred.data<float>();
}

void LSTMPoseSmoothing::clear_hidden_states() {
    std::fill_n(h_.data<float>(), h_.get_size(), 0.0f);
    std::fill_n(c_.data<float>(), c_.get_size(), 0.0f);
}