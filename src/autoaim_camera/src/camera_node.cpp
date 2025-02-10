#include "MvCameraControl.h"

#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/utilities.hpp>
#include <camera_info_manager/camera_info_manager.hpp>
#include <image_transport/image_transport.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <opencv2/opencv.hpp>

namespace autoaim_camera {
float get_fps() {
    static auto prev = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    float elapsed_sec = (now - prev).count() / 1e9;
    prev = now;
    return 1 / elapsed_sec;
}

class CameraNode: public rclcpp::Node {
public:
    explicit CameraNode(const rclcpp::NodeOptions& options);
    ~CameraNode() override;

private:
    void get_parameters();
    void open_cam();
    void start_grabbing();
    void close_cam();
    void capture_thread();
    bool catch_error(int ret, const char* description);

    sensor_msgs::msg::Image image_msg_;
    sensor_msgs::msg::CameraInfo camera_info_msg_;
    image_transport::CameraPublisher camera_pub_;
    std::unique_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;

    void* cam_handle_;
    MV_IMAGE_BASIC_INFO img_info_;
    MV_CC_PIXEL_CONVERT_PARAM pixel_convert_param_;

    std::thread capture_thread_;

    bool enable_fps_;
    std::string camera_name_;
    float exposure_, gain_;
};

CameraNode::CameraNode(const rclcpp::NodeOptions& options): Node("autoaim_camera", options) {
    get_parameters();
    open_cam();
    start_grabbing();
    capture_thread_ = std::thread(&CameraNode::capture_thread, this);
}

CameraNode::~CameraNode() {
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
    close_cam();
}

void CameraNode::close_cam() {
    catch_error(MV_CC_StopGrabbing(cam_handle_), "stop grabbing");
    catch_error(MV_CC_CloseDevice(cam_handle_), "close device");
    catch_error(MV_CC_DestroyHandle(cam_handle_), "destroy handle");
}

bool CameraNode::catch_error(int ret, const char* description) {
    if (ret != MV_OK) {
        RCLCPP_ERROR(this->get_logger(), "Error in \"%s\": %#x", description, ret);
        return true;
    }
    return false;
}

void CameraNode::get_parameters() {
    std::string camera_info_url =
        declare_parameter("camera_info_url", "package://autoaim_camera/config/camera_info.yaml");
    std::string img_pub_topic_ = declare_parameter("img_pub_topic", "/camera/color/image_raw");
    camera_name_ = declare_parameter("camera_name", "auto");
    enable_fps_ = declare_parameter("enable_fps", false);
    exposure_ = declare_parameter("exposure", 2000.0);
    gain_ = declare_parameter("gain", 16.0);

    camera_info_manager_ =
        std::make_unique<camera_info_manager::CameraInfoManager>(this, camera_name_);
    if (camera_info_manager_->validateURL(camera_info_url)) {
        camera_info_manager_->loadCameraInfo(camera_info_url);
        camera_info_msg_ = camera_info_manager_->getCameraInfo();
    } else {
        RCLCPP_ERROR(this->get_logger(), "Invalid camera info URL: %s", camera_info_url.c_str());
    }
    camera_pub_ =
        image_transport::create_camera_publisher(this, img_pub_topic_, rmw_qos_profile_default);
}

void CameraNode::capture_thread() {
    MV_FRAME_OUT out_frame;
    image_msg_.header.frame_id = "camera_optical_frame";
    image_msg_.encoding = "rgb8";
    while (rclcpp::ok()) {
        const int ret_val = MV_CC_GetImageBuffer(cam_handle_, &out_frame, 1000);
        image_msg_.header.stamp = this->now();
        if (ret_val == MV_OK) {
            pixel_convert_param_.pDstBuffer = image_msg_.data.data();
            pixel_convert_param_.nDstBufferSize = image_msg_.data.size();
            pixel_convert_param_.pSrcData = out_frame.pBufAddr;
            pixel_convert_param_.nSrcDataLen = out_frame.stFrameInfo.nFrameLen;
            pixel_convert_param_.enSrcPixelType = out_frame.stFrameInfo.enPixelType;

            MV_CC_ConvertPixelType(cam_handle_, &pixel_convert_param_);

            image_msg_.height = out_frame.stFrameInfo.nHeight;
            image_msg_.width = out_frame.stFrameInfo.nWidth;
            image_msg_.step = out_frame.stFrameInfo.nWidth * 3;
            image_msg_.data.resize(image_msg_.width * image_msg_.height * 3);

            camera_info_msg_.header = image_msg_.header;
            camera_pub_.publish(image_msg_, camera_info_msg_);

            MV_CC_FreeImageBuffer(cam_handle_, &out_frame);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Get buffer failed! ret_val: [%x]", ret_val);
            MV_CC_StopGrabbing(cam_handle_);
            MV_CC_StartGrabbing(cam_handle_);
        }
        if (enable_fps_) {
            RCLCPP_INFO(this->get_logger(), "Camera FPS: %.0f", get_fps());
        }
    }
}

void CameraNode::open_cam() {
    MV_CC_DEVICE_INFO_LIST devices_list;
    memset(&devices_list, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
    int camera_idx = -1;
    while (rclcpp::ok()) {
        RCLCPP_INFO(this->get_logger(), "Looking for camera <%s>", camera_name_.c_str());
        catch_error(MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &devices_list), "enum devices");
        const int camera_nums = devices_list.nDeviceNum;
        for (int i = 0; i < camera_nums; i++) {
            MV_CC_DEVICE_INFO* device_info_ptr = devices_list.pDeviceInfo[i];
            std::string name(reinterpret_cast<char const*>(
                device_info_ptr->SpecialInfo.stUsb3VInfo.chUserDefinedName
            ));
            if (name == camera_name_ || camera_name_ == "auto") {
                camera_idx = i;
                break;
            }
            RCLCPP_INFO(
                this->get_logger(),
                "Camera <%s> detected, but not <%s>",
                name.c_str(),
                camera_name_.c_str()
            );
        }
        if (camera_idx != -1) {
            RCLCPP_INFO(this->get_logger(), "Camera <%s> found", camera_name_.c_str());
            break;
        }
        RCLCPP_WARN(this->get_logger(), "Camera <%s> not found, retry in 1s", camera_name_.c_str());
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    catch_error(MV_CC_CreateHandle(&cam_handle_, devices_list.pDeviceInfo[camera_idx]), "create handle");
    catch_error(MV_CC_OpenDevice(cam_handle_), "open device");
}

void CameraNode::start_grabbing() {
    // 合并相邻像素，1440*1080 -> 720*540
    catch_error(MV_CC_SetEnumValue(cam_handle_, "BinningHorizontal", 2), "set binning horizontal");
    catch_error(MV_CC_SetEnumValue(cam_handle_, "BinningVertical", 2), "set binning vertical");

    // 设置像素格式
    catch_error(MV_CC_SetEnumValue(cam_handle_, "PixelFormat", PixelType_Gvsp_BGR8_Packed), "set pixel format");

    // 启用自动gamma
    catch_error(MV_CC_SetBoolValue(cam_handle_, "GammaEnable", true), "set gamma enable");
    catch_error(MV_CC_SetEnumValue(cam_handle_, "GammaSelector", 2), "set gamma selector");

    // 连续触发模式
    catch_error(MV_CC_SetEnumValue(cam_handle_, "AcquisitionMode", 2), "set acquisition mode");
    catch_error(MV_CC_SetEnumValue(cam_handle_, "TriggerMode", MV_TRIGGER_MODE_OFF), "set trigger mode");

    // 手动设置曝光和增益
    catch_error(MV_CC_SetEnumValue(cam_handle_, "ExposureAuto", 0), "set auto exposure");
    catch_error(MV_CC_SetEnumValue(cam_handle_, "GainAuto", 0), "set auto gain");
    catch_error(MV_CC_SetFloatValue(cam_handle_, "ExposureTime", exposure_), "set exposure time");
    catch_error(MV_CC_SetFloatValue(cam_handle_, "Gain", gain_), "set gain");

    // 设置分辨率（适当裁切，与模型输入大小匹配）
    catch_error(MV_CC_SetIntValue(cam_handle_, "Width", 640), "set width");
    catch_error(MV_CC_SetIntValue(cam_handle_, "Height", 384), "set height");
    catch_error(MV_CC_SetIntValue(cam_handle_, "OffsetX", 40), "set offset x");
    catch_error(MV_CC_SetIntValue(cam_handle_, "OffsetY", 124), "set offset y");

    // 设置BGR转RGB
    catch_error(MV_CC_GetImageInfo(cam_handle_, &img_info_), "get image info");
    pixel_convert_param_.nWidth = img_info_.nWidthValue;
    pixel_convert_param_.nHeight = img_info_.nHeightValue;
    pixel_convert_param_.enDstPixelType = PixelType_Gvsp_RGB8_Packed;

    // 开始取流
    catch_error(MV_CC_StartGrabbing(cam_handle_), "start grabbing");
}
} // namespace autoaim_camera

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_camera::CameraNode)