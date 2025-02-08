#include <iostream>
#include <chrono>

#include "MvCameraControl.h"

#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/utilities.hpp>
#include <camera_info_manager/camera_info_manager.hpp>
#include <image_transport/image_transport.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <autoaim_interfaces/msg/comm_recv.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <opencv2/opencv.hpp>

enum MODE : int {
    ARMOR_MODE = 0,
    SMALL_BUFF_MODE = 1,
    BIG_BUFF_MODE = 2,
};

namespace fps {
static auto last_time = std::chrono::steady_clock::now();
static float fps = 100;
float get_fps() {
    auto now_time = std::chrono::steady_clock::now();
    auto durt = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last_time).count();
    fps = 5e6 / (durt + 4e6 / fps);
    last_time = now_time;
    return fps;
}
} // namespace fps

struct CameraParam {
    bool auto_exposure;
    std::string camera_name;
    int brightness;
    int exposure;
    float gain;
    bool enable_gamma;
};

namespace autoaim_camera {
class HikCameraNode: public rclcpp::Node {
public:
    explicit HikCameraNode(const rclcpp::NodeOptions& options);
    ~HikCameraNode() override;
    void set_mode(int mode);

private:
    void declare_parameters();
    void capture_thread();
    void publish_time();
    bool catch_error(int ret);
    void open_cam();
    void close_cam();
    rcl_interfaces::msg::SetParametersResult
    parameters_callback(const std::vector<rclcpp::Parameter>& parameters);

    sensor_msgs::msg::Image image_msg_;
    sensor_msgs::msg::CameraInfo camera_info_msg_;

    image_transport::CameraPublisher camera_pub_;
    std::unique_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;
    OnSetParametersCallbackHandle::SharedPtr params_callback_handle_;
    rclcpp::Subscription<autoaim_interfaces::msg::CommRecv>::SharedPtr sub_comm_;

    void* cam_handle_;
    MV_IMAGE_BASIC_INFO img_info_;
    MV_CC_PIXEL_CONVERT_PARAM stConvertParam;

    CameraParam param_;
    std::thread capture_thread_;

    std::string img_pub_topic_;
    std::string enemy_info_topic_;
    int current_mode_;
    int camera_offset_y_;
    float exposure_time_;
    float gain_;
    bool enable_debug_;
    bool use_sensor_data_qos_;
};

void HikCameraNode::set_mode(int mode) {
    if (current_mode_ == mode) {
        return;
    }
    if (mode == MODE::ARMOR_MODE) {
        if (false != param_.enable_gamma) {
            // gamma使能
            catch_error(MV_CC_SetBoolValue(cam_handle_, "GammaEnable", true));
            catch_error(MV_CC_SetEnumValue(cam_handle_, "GammaSelector", 2));
            // catch_error(MV_CC_SetFloatValue(cam_handle_, "Gamma", param_.camera.gamma))
        }
        catch_error(MV_CC_SetFloatValue(cam_handle_, "ExposureTime", exposure_time_));
        catch_error(MV_CC_SetFloatValue(cam_handle_, "Gain", gain_));
    } else if (mode == MODE::SMALL_BUFF_MODE || mode == MODE::BIG_BUFF_MODE) {
        catch_error(MV_CC_SetBoolValue(cam_handle_, "GammaEnable", false));
        catch_error(MV_CC_SetFloatValue(cam_handle_, "ExposureTime", param_.exposure));
        catch_error(MV_CC_SetFloatValue(cam_handle_, "Gain", param_.gain));
        RCLCPP_INFO(this->get_logger(), "Set exposure time to: %d", param_.exposure);
        RCLCPP_INFO(this->get_logger(), "Set gain to: %f", param_.gain);
    }
    current_mode_ = mode;
}

HikCameraNode::HikCameraNode(const rclcpp::NodeOptions& options): Node("hik_camera", options) {
    declare_parameters();
    sub_comm_ = this->create_subscription<autoaim_interfaces::msg::CommRecv>(
        enemy_info_topic_,
        10,
        [this](const autoaim_interfaces::msg::CommRecv::SharedPtr msg) {
            set_mode(msg->mode);
        }
    );
    open_cam();
    params_callback_handle_ = this->add_on_set_parameters_callback(
        std::bind(&HikCameraNode::parameters_callback, this, std::placeholders::_1)
    );
    capture_thread_ = std::thread(&HikCameraNode::capture_thread, this);
}

HikCameraNode::~HikCameraNode() {
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
    close_cam();
}

void HikCameraNode::close_cam() {
    catch_error(MV_CC_StopGrabbing(cam_handle_));
    catch_error(MV_CC_CloseDevice(cam_handle_));
    catch_error(MV_CC_DestroyHandle(cam_handle_));
}

bool HikCameraNode::catch_error(int ret) {
    if (ret != MV_OK) {
        RCLCPP_ERROR(this->get_logger(), "Error: %#x", ret);
        return true;
    }
    return false;
}

void HikCameraNode::open_cam() {
    MV_CC_DEVICE_INFO_LIST devices_list;
    memset(&devices_list, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
    int camera_idx = -1;
    while (rclcpp::ok()) {
        RCLCPP_INFO(this->get_logger(), "Looking for camera <%s>", param_.camera_name.c_str());
        catch_error(MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &devices_list));
        int camera_nums = devices_list.nDeviceNum;
        for (int i = 0; i < camera_nums; i++) {
            MV_CC_DEVICE_INFO* device_info_ptr = devices_list.pDeviceInfo[i];
            std::string name(reinterpret_cast<char const*>(
                device_info_ptr->SpecialInfo.stUsb3VInfo.chUserDefinedName
            ));
            if (name == param_.camera_name || param_.camera_name == "auto") {
                camera_idx = i;
                break;
            }
            RCLCPP_INFO(
                this->get_logger(),
                "Camera <%s> detected, but not <%s>",
                name.c_str(),
                param_.camera_name.c_str()
            );
        }
        if (camera_idx != -1) {
            RCLCPP_INFO(this->get_logger(), "Camera <%s> found", param_.camera_name.c_str());
            break;
        }
        RCLCPP_WARN(
            this->get_logger(),
            "Camera <%s> not found, retry in 1s",
            param_.camera_name.c_str()
        );
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    catch_error(MV_CC_CreateHandle(&cam_handle_, devices_list.pDeviceInfo[camera_idx]));
    if (catch_error(MV_CC_IsDeviceConnected(cam_handle_))) {
        RCLCPP_ERROR(this->get_logger(), "相机全死里面啦, 等我重开一下😡");
        catch_error(MV_CC_OpenDevice(cam_handle_));
        catch_error(MV_CC_CloseDevice(cam_handle_));
    }
    catch_error(MV_CC_OpenDevice(cam_handle_));

    MV_CC_GetImageInfo(cam_handle_, &img_info_);
    image_msg_.data.reserve(img_info_.nHeightMax * img_info_.nWidthMax * 3);

    // Init convert param_
    stConvertParam.nWidth = img_info_.nWidthValue;
    stConvertParam.nHeight = img_info_.nHeightValue;
    stConvertParam.enDstPixelType = PixelType_Gvsp_RGB8_Packed;
    MV_CC_SetEnumValue(cam_handle_, "PixelFormat", PixelType_Gvsp_BGR8_Packed);

    if (param_.enable_gamma) {
        catch_error(MV_CC_SetBoolValue(cam_handle_, "GammaEnable", true));
        catch_error(MV_CC_SetEnumValue(cam_handle_, "GammaSelector", 2));
        // catch_error(MV_CC_SetFloatValue(cam_handle_, "Gamma", param_.gamma));
    } else {
        catch_error(MV_CC_SetBoolValue(cam_handle_, "GammaEnable", false));
    }

    auto qos = use_sensor_data_qos_ ? rmw_qos_profile_sensor_data : rmw_qos_profile_default;
    camera_pub_ = image_transport::create_camera_publisher(this, img_pub_topic_, qos);

    // 连续采集模式
    catch_error(MV_CC_SetEnumValue(cam_handle_, "AcquisitionMode", 2));
    catch_error(MV_CC_SetEnumValue(cam_handle_, "TriggerMode", MV_TRIGGER_MODE_OFF));
    if (param_.auto_exposure) {
        // 自动模式
        RCLCPP_INFO(this->get_logger(), "开启的是自动模式哟");
        // 自动模式 0:关闭 1:一次 2:连续
        catch_error(MV_CC_SetEnumValue(cam_handle_, "GainAuto", 0));
        catch_error(MV_CC_SetFloatValue(cam_handle_, "Gain", 16));
        catch_error(MV_CC_SetIntValue(cam_handle_, "AutoExposureTimeLowerLimit", 1000));
        catch_error(MV_CC_SetIntValue(cam_handle_, "AutoExposureTimeUpperLimit", 3000));
        catch_error(MV_CC_SetEnumValue(cam_handle_, "ExposureAuto", 2));
        catch_error(MV_CC_SetIntValue(cam_handle_, "Brightness", param_.brightness));
        // 白平衡 0:关闭 1:连续 2:一次 海康🧠有点问题
        catch_error(MV_CC_SetEnumValue(cam_handle_, "BalanceWhiteAuto", 1));
    } else {
        // 手动模式
        RCLCPP_INFO(this->get_logger(), "开启的是手动模式哟");
        // 清除默认设置
        catch_error(MV_CC_SetEnumValue(cam_handle_, "ExposureAuto", 0)); // 取消自动曝光
        catch_error(MV_CC_SetEnumValue(cam_handle_, "GainAuto", 0));
        // 设置曝光
        catch_error(MV_CC_SetFloatValue(cam_handle_, "ExposureTime", 2000));
        // 设置增益
        catch_error(MV_CC_SetFloatValue(cam_handle_, "Gain", 16));
    }
    // 设置分辨率
    catch_error(MV_CC_SetIntValue(cam_handle_, "Width", 1440));
    catch_error(MV_CC_SetIntValue(cam_handle_, "Height", 864));
    // this->set_parameter(rclcpp::Parameter("camera_width", 1440));
    // this->set_parameter(rclcpp::Parameter("camera_height", 864));
    catch_error(MV_CC_SetIntValue(cam_handle_, "OffsetY", camera_offset_y_));
    // 开始取流
    catch_error(MV_CC_StartGrabbing(cam_handle_));
    // 给时间自动曝光，1s后固定曝光和增益值
    if (param_.auto_exposure) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        catch_error(MV_CC_SetEnumValue(cam_handle_, "ExposureAuto", 0));
        catch_error(MV_CC_SetEnumValue(cam_handle_, "GainAuto", 0));
        catch_error(MV_CC_SetEnumValue(cam_handle_, "BalanceWhiteAuto", 0));

        MVCC_FLOATVALUE float_val = { 0 };
        catch_error(MV_CC_GetFloatValue(cam_handle_, "ExposureTime", &float_val));
        exposure_time_ = float_val.fCurValue;
        RCLCPP_INFO(this->get_logger(), "ExposureTime: %f", exposure_time_);

        catch_error(MV_CC_GetFloatValue(cam_handle_, "Gain", &float_val));
        gain_ = float_val.fCurValue;
        RCLCPP_INFO(this->get_logger(), "Gain: %f", gain_);
    }
    RCLCPP_INFO(this->get_logger(), "相机开始工作啦😊");
}

void HikCameraNode::declare_parameters() {
    std::string camera_name = declare_parameter("camera_name", "auto");
    bool auto_exposure = declare_parameter("auto_exposure", true);
    int brightness = declare_parameter("brightness", 128);
    int exposure = declare_parameter("exposure", 2000);
    float gain = declare_parameter("gain", 16.0);
    bool enable_gamma = declare_parameter("enable_gamma", false);
    int default_mode = declare_parameter("default_mode", 0);
    std::string camera_info_url = declare_parameter(
        "camera_info_url",
        "package://autoaim_camera/config/camera_info.yaml"
    );

    img_pub_topic_ = declare_parameter("img_pub_topic", "camera/image_raw");
    camera_offset_y_ = declare_parameter("camera_offset_y", 216);
    enemy_info_topic_ = declare_parameter("enemy_info_topic", "/serial/comm_recv");
    enable_debug_ = declare_parameter("enable_debug", false);
    use_sensor_data_qos_ = declare_parameter("use_sensor_data_qos", false);

    set_mode(default_mode);
    param_ = { auto_exposure, camera_name, brightness, exposure, gain, enable_gamma };

    if (enable_debug_) {
        RCLCPP_INFO(this->get_logger(), "img_pub_topic: %s", img_pub_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "camera_name: %s", camera_name.c_str());
        RCLCPP_INFO(this->get_logger(), "auto_exposure: %d", auto_exposure);
        RCLCPP_INFO(this->get_logger(), "brightness: %d", brightness);
        RCLCPP_INFO(this->get_logger(), "exposure: %d", exposure);
        RCLCPP_INFO(this->get_logger(), "gain: %f", gain);
        RCLCPP_INFO(this->get_logger(), "enable_gamma: %d", enable_gamma);
        RCLCPP_INFO(this->get_logger(), "use_sensor_data_qos: %d", use_sensor_data_qos_);
    }

    // Load camera info
    camera_info_manager_ =
        std::make_unique<camera_info_manager::CameraInfoManager>(this, camera_name);

    if (camera_info_manager_->validateURL(camera_info_url)) {
        camera_info_manager_->loadCameraInfo(camera_info_url);
        camera_info_msg_ = camera_info_manager_->getCameraInfo();
    } else {
        RCLCPP_ERROR(this->get_logger(), "Invalid camera info URL: %s", camera_info_url.c_str());
    }
}

void HikCameraNode::capture_thread() {
    MV_FRAME_OUT out_frame;
    image_msg_.header.frame_id = "camera_optical_frame";
    image_msg_.encoding = "rgb8";
    while (rclcpp::ok()) {
        int ret_val = MV_CC_GetImageBuffer(cam_handle_, &out_frame, 1000);
        image_msg_.header.stamp = this->now();
        if (MV_OK == ret_val) {
            stConvertParam.pDstBuffer = image_msg_.data.data();
            stConvertParam.nDstBufferSize = image_msg_.data.size();
            stConvertParam.pSrcData = out_frame.pBufAddr;
            stConvertParam.nSrcDataLen = out_frame.stFrameInfo.nFrameLen;
            stConvertParam.enSrcPixelType = out_frame.stFrameInfo.enPixelType;

            MV_CC_ConvertPixelType(cam_handle_, &stConvertParam);

            image_msg_.height = out_frame.stFrameInfo.nHeight;
            image_msg_.width = out_frame.stFrameInfo.nWidth;
            image_msg_.step = out_frame.stFrameInfo.nWidth * 3;
            image_msg_.data.resize(image_msg_.width * image_msg_.height * 3);
            // log width and height
            // RCLCPP_INFO(this->get_logger(), "width: %d, height: %d", image_msg_.width, image_msg_.height);

            camera_info_msg_.header = image_msg_.header;
            camera_pub_.publish(image_msg_, camera_info_msg_);

            MV_CC_FreeImageBuffer(cam_handle_, &out_frame);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Get buffer failed! ret_val: [%x]", ret_val);
            MV_CC_StopGrabbing(cam_handle_);
            MV_CC_StartGrabbing(cam_handle_);
        }
        if (enable_debug_) {
            RCLCPP_INFO(this->get_logger(), "FPS: %f", fps::get_fps());
        }
    }
}

rcl_interfaces::msg::SetParametersResult
HikCameraNode::parameters_callback(const std::vector<rclcpp::Parameter>& parameters) {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    for (const auto& parameter: parameters) {
        if (parameter.get_name() == "auto_exposure") {
            param_.auto_exposure = parameter.as_bool();
            RCLCPP_INFO(this->get_logger(), "Set auto_exposure to: %d", param_.auto_exposure);
        }
    }
    return result;
}

} // namespace autoaim_camera

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_camera::HikCameraNode)
