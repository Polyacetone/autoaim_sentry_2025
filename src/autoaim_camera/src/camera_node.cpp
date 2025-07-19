#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <MvCameraControl.h>
#include <camera_info_manager/camera_info_manager.hpp>
#include <image_transport/image_transport.hpp>

#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

namespace autoaim_camera {
double to_sec(builtin_interfaces::msg::Time t) { return static_cast<rclcpp::Time>(t).seconds(); }

class CameraNode: public rclcpp::Node {
public:
    explicit CameraNode(const rclcpp::NodeOptions& options);
    ~CameraNode() override;

private:
    void get_parameters();
    void open_cam();
    void start_grabbing() const;
    void close_cam() const;
    bool catch_error(int ret, const char* description) const;
    void capture_thread();
    float get_framerate();

    // 从imu_timestamp_buffer_中获取图像时间戳对应的imu时间戳
    rclcpp::Time get_corresponding_imu_timestamp(const rclcpp::Time& img_time) const;

    bool enable_fps_;
    bool enable_imu_trigger_;
    std::string camera_name_;
    float exposure_, gain_, gamma_, frame_rate_;

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr imu_timestamp_sub_;
    image_transport::CameraPublisher camera_pub_;

    void* cam_handle_;
    std::unique_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;
    std::deque<rclcpp::Time> imu_timestamp_buffer_;
    std::thread capture_thread_;
    std::chrono::high_resolution_clock::time_point prev_callback_time_;
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

void CameraNode::close_cam() const {
    catch_error(MV_CC_StopGrabbing(cam_handle_), "stop grabbing");
    catch_error(MV_CC_CloseDevice(cam_handle_), "close device");
    catch_error(MV_CC_DestroyHandle(cam_handle_), "destroy handle");
}

bool CameraNode::catch_error(int ret, const char* description) const {
    if (ret != MV_OK) {
        RCLCPP_ERROR(this->get_logger(), "Error in \"%s\": %#x", description, ret);
        return true;
    }
    return false;
}

float CameraNode::get_framerate() {
    auto current_time = std::chrono::high_resolution_clock::now();
    float duration = (current_time - prev_callback_time_).count() / 1e9;
    prev_callback_time_ = current_time;
    return 1 / duration;
}

void CameraNode::get_parameters() {
    std::string camera_info_url = declare_parameter<std::string>("camera_info_url");
    std::string image_raw_topic = declare_parameter<std::string>("image_raw_topic");
    std::string imu_timestamp_topic = declare_parameter<std::string>("imu_timestamp_topic");
    camera_name_ = declare_parameter<std::string>("camera_name");
    enable_fps_ = declare_parameter<bool>("enable_fps");
    enable_imu_trigger_ = declare_parameter<bool>("enable_imu_trigger");
    frame_rate_ = declare_parameter<float>("frame_rate");
    exposure_ = declare_parameter<float>("exposure");
    gamma_ = declare_parameter<float>("gamma");
    gain_ = declare_parameter<float>("gain");

    imu_timestamp_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        imu_timestamp_topic,
        rclcpp::QoS(1),
        [&](const sensor_msgs::msg::JointState::SharedPtr msg) {
            imu_timestamp_buffer_.emplace_front(msg->header.stamp);
            if (imu_timestamp_buffer_.size() > 10) {
                imu_timestamp_buffer_.pop_back();
            }
        }
    );

    camera_info_manager_ = std::make_unique<camera_info_manager::CameraInfoManager>(this, camera_name_);
    if (camera_info_manager_->validateURL(camera_info_url)) {
        camera_info_manager_->loadCameraInfo(camera_info_url);
    } else {
        RCLCPP_FATAL(get_logger(), "Failed to load camera info");
        throw std::runtime_error("invalid camera_info_url");
    }
    rmw_qos_profile_t custom_qos = rmw_qos_profile_default;
    custom_qos.depth = 1;
    camera_pub_ = image_transport::create_camera_publisher(this, image_raw_topic, custom_qos);
}

void CameraNode::capture_thread() {
    sensor_msgs::msg::Image image_msg;
    sensor_msgs::msg::CameraInfo camera_info_msg;
    MV_FRAME_OUT out_frame;
    image_msg.header.frame_id = "autoaim_camera";
    image_msg.encoding = "bgr8";
    camera_info_msg = camera_info_manager_->getCameraInfo();
    while (rclcpp::ok()) {
        const int ret_val = MV_CC_GetImageBuffer(cam_handle_, &out_frame, 100);
        const rclcpp::Time current_time = this->now();

        if (ret_val != MV_OK) {
            RCLCPP_FATAL(this->get_logger(), "Get buffer failed! ret_val: [%x]", ret_val);
            // 把自己杀死后让外面的launcher再拉起来
            std::exit(1);
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

        if (enable_imu_trigger_) {
            image_msg.header.stamp = get_corresponding_imu_timestamp(current_time);
        } else {
            image_msg.header.stamp = current_time;
        }
        camera_info_msg.header = image_msg.header;

        image_msg.height = resized_img.rows;
        image_msg.width = resized_img.cols;
        image_msg.step = image_msg.width * 3;
        image_msg.data.resize(image_msg.width * image_msg.height * 3);
        std::copy(
            resized_img.data,
            resized_img.data + image_msg.data.size() + 1,
            image_msg.data.data()
        );
        
        camera_pub_.publish(image_msg, camera_info_msg);

        if (enable_fps_) {
            RCLCPP_INFO(this->get_logger(), "Camera FPS: %.0f", get_framerate());
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
            std::string name(reinterpret_cast<char const*>(device_info_ptr->SpecialInfo.stUsb3VInfo.chUserDefinedName));
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

void CameraNode::start_grabbing() const {
    // 设置像素格式
    catch_error(MV_CC_SetEnumValue(cam_handle_, "PixelFormat", PixelType_Gvsp_BGR8_Packed), "set pixel format");

    // 设置分辨率（MV-CS016-10UC最大1440*1080，不过这里取1280*768）
    catch_error(MV_CC_SetIntValue(cam_handle_, "Width", 1280), "set width");
    catch_error(MV_CC_SetIntValue(cam_handle_, "Height", 768), "set height");
    catch_error(MV_CC_SetIntValue(cam_handle_, "OffsetX", (1440-1280)/2), "set offset x");
    catch_error(MV_CC_SetIntValue(cam_handle_, "OffsetY", (1080-768)), "set offset y");

    // 启用手动gamma
    catch_error(MV_CC_SetBoolValue(cam_handle_, "GammaEnable", true), "set gamma enable");
    catch_error(MV_CC_SetEnumValue(cam_handle_, "GammaSelector", 1), "set gamma selector");
    catch_error(MV_CC_SetFloatValue(cam_handle_, "Gamma", gamma_), "set gamma value");

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

rclcpp::Time CameraNode::get_corresponding_imu_timestamp(const rclcpp::Time& img_time) const {
    auto iter = std::find_if(
        imu_timestamp_buffer_.begin(), imu_timestamp_buffer_.end(),
        [&](const auto& imu_timestamp) -> bool {
            const double diff = to_sec(img_time) - to_sec(imu_timestamp);
            // offset = 曝光时间 + 相机处理时间 + 图像传输时间 - 串口传输时间
            // 相机传输线带宽约3000Mbps，所以大概需要8ms才能把图像传过来。相机处理时间减去串口传输时间大概是1ms左右
            const double offset = exposure_ / 1e6 + 9e-3;
            return offset - 2e-3 < diff && diff < offset + 2e-3;
        }
    );
    if (iter != imu_timestamp_buffer_.end()) return *iter;
    if (imu_timestamp_buffer_.empty()) {
        RCLCPP_WARN(
            get_logger(),
            "Failed to synchronize image timestamp with imu timestamp: empty imu timestamp buffer"
        );
    } else {
        RCLCPP_WARN(
            get_logger(), 
            "Failed to synchronize image timestamp with imu timestamp: newest timestamp diff: %6.4lf",
            to_sec(img_time) - to_sec(imu_timestamp_buffer_[0])
        );
    }
    return img_time;
}
} // namespace autoaim_camera

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_camera::CameraNode)