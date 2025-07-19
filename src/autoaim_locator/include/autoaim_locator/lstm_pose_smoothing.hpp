#pragma once

#include <openvino/openvino.hpp>

class LSTMPoseSmoothing {
public:
    explicit LSTMPoseSmoothing(const std::string& model_path);
    ~LSTMPoseSmoothing() = default;
    float infer(float yaw0, float yaw1, float err0, float err1, float dt);

private:
    void clear_hidden_states();

    ov::InferRequest infer_request_;
    ov::CompiledModel compiled_model_;
    ov::Tensor h_, c_; // 隐藏状态
    std::vector<ov::Output<const ov::Node>> inputs_, outputs_; // 输入输出大小和类型
    float time_elapsed_ = 0;
};