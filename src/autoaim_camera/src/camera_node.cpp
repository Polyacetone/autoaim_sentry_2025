#include <MvCameraControl.h>

#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/utilities.hpp>
#include <camera_info_manager/camera_info_manager.hpp>
#include <image_transport/image_transport.hpp>
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

    image_transport::CameraPublisher camera_pub_;
    std::unique_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;

    void* cam_handle_;
    MV_IMAGE_BASIC_INFO img_info_;
    MV_CC_PIXEL_CONVERT_PARAM pixel_convert_param_;

    std::thread capture_thread_;

    bool enable_fps_;
    bool enable_imu_trigger_;
    std::string camera_name_;
    float exposure_, gain_, frame_rate_;
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
    enable_imu_trigger_ = declare_parameter("enable_imu_trigger", false);
    frame_rate_ = declare_parameter("frame_rate", 100.0);
    exposure_ = declare_parameter("exposure", 2000.0);
    gain_ = declare_parameter("gain", 16.0);

    camera_info_manager_ =
        std::make_unique<camera_info_manager::CameraInfoManager>(this, camera_name_);
    if (camera_info_manager_->validateURL(camera_info_url)) {
        camera_info_manager_->loadCameraInfo(camera_info_url);
    } else {
        RCLCPP_ERROR(this->get_logger(), "Invalid camera info URL: %s", camera_info_url.c_str());
    }
    rmw_qos_profile_t custom_qos = rmw_qos_profile_sensor_data;
    custom_qos.depth = 1;
    camera_pub_ = image_transport::create_camera_publisher(this, img_pub_topic_, custom_qos);
}

void CameraNode::capture_thread() {
    sensor_msgs::msg::Image image_msg;
    sensor_msgs::msg::CameraInfo camera_info_msg;
    MV_FRAME_OUT out_frame;
    image_msg.header.frame_id = "camera_optical_frame";
    image_msg.encoding = "bgr8";
    camera_info_msg = camera_info_manager_->getCameraInfo();
    while (rclcpp::ok()) {
        const int ret_val = MV_CC_GetImageBuffer(cam_handle_, &out_frame, 100);
        image_msg.header.stamp = this->now();
        camera_info_msg.header = image_msg.header;

        if (ret_val != MV_OK) {
            RCLCPP_ERROR(this->get_logger(), "Get buffer failed! ret_val: [%x]", ret_val);
            MV_CC_StopGrabbing(cam_handle_);
            MV_CC_StartGrabbing(cam_handle_);
            continue;
        }

        // 1440*864, BGR8
        const cv::Mat capture_frame(
            cv::Size(out_frame.stFrameInfo.nWidth, out_frame.stFrameInfo.nHeight),
            CV_8UC3
        );
        std::copy(
            out_frame.pBufAddr,
            out_frame.pBufAddr + out_frame.stFrameInfo.nFrameLen + 1,
            capture_frame.data
        );
        MV_CC_FreeImageBuffer(cam_handle_, &out_frame);

        // 1280*768 -> 640*384
        cv::Mat resized_img;
        cv::resize(capture_frame, resized_img, cv::Size(640, 384), 0, 0, cv::INTER_LINEAR);

        image_msg.height = resized_img.rows;
        image_msg.width = resized_img.cols;
        image_msg.step = image_msg.width * 3;
        image_msg.data.resize(image_msg.width * image_msg.height * 3);
        std::copy(resized_img.data, resized_img.data + image_msg.data.size() + 1, image_msg.data.data());
        
        camera_pub_.publish(image_msg, camera_info_msg);

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
    // 设置像素格式
    catch_error(MV_CC_SetEnumValue(cam_handle_, "PixelFormat", PixelType_Gvsp_BGR8_Packed), "set pixel format");

    // 设置分辨率（MV-CS016-10UC最大1440*1080，不过这里取1280*768）
    catch_error(MV_CC_SetIntValue(cam_handle_, "Width", 1280), "set width");
    catch_error(MV_CC_SetIntValue(cam_handle_, "Height", 768), "set height");
    catch_error(MV_CC_SetIntValue(cam_handle_, "OffsetX", (1440-1280)/2), "set offset x");
    catch_error(MV_CC_SetIntValue(cam_handle_, "OffsetY", (1080-768)), "set offset y");

    // 启用自动gamma
    catch_error(MV_CC_SetBoolValue(cam_handle_, "GammaEnable", true), "set gamma enable");
    catch_error(MV_CC_SetEnumValue(cam_handle_, "GammaSelector", 2), "set gamma selector");

    // 启用自动白平衡
    catch_error(MV_CC_SetEnumValue(cam_handle_, "BalanceWhiteAuto", 1), "set balance white auto");

    // 手动设置曝光、增益（从配置文件中读取）
    catch_error(MV_CC_SetEnumValue(cam_handle_, "ExposureAuto", 0), "set auto exposure");
    catch_error(MV_CC_SetEnumValue(cam_handle_, "GainAuto", 0), "set auto gain");
    catch_error(MV_CC_SetFloatValue(cam_handle_, "ExposureTime", exposure_), "set exposure time");
    catch_error(MV_CC_SetFloatValue(cam_handle_, "Gain", gain_), "set gain");

    if (enable_imu_trigger_) {
        // 硬触发模式
        catch_error(MV_CC_SetEnumValue(cam_handle_, "TriggerMode", MV_TRIGGER_MODE_ON), "set trigger mode on");
        catch_error(MV_CC_SetEnumValue(cam_handle_, "TriggerSource", MV_TRIGGER_SOURCE_LINE0), "set trigger source");
    } else {
        // 连续触发模式
        catch_error(MV_CC_SetEnumValue(cam_handle_, "AcquisitionMode", 2), "set acquisition mode");
        catch_error(MV_CC_SetEnumValue(cam_handle_, "TriggerMode", MV_TRIGGER_MODE_OFF), "set trigger mode off");

        // 设置采集帧率
        catch_error(MV_CC_SetBoolValue(cam_handle_, "AcquisitionFrameRateEnable", true), "set frame rate enable");
        catch_error(MV_CC_SetFloatValue(cam_handle_, "AcquisitionFrameRate", frame_rate_), "set frame rate");
    }

    // 开始取流
    catch_error(MV_CC_StartGrabbing(cam_handle_), "start grabbing");
}
} // namespace autoaim_camera

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_camera::CameraNode)