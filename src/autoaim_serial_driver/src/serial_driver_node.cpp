#include <string>

#include <rclcpp/rclcpp.hpp>
#include <tf2/convert.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <serial_driver/serial_driver.hpp>

#include <autoaim_interfaces/msg/detection_array.hpp>
#include <autoaim_interfaces/msg/shoot_pos.hpp>

#include <serial_utils.hpp>
#include <receive_packet.hpp>
#include <send_packet.hpp>
#include <crc_checksum.hpp>

namespace autoaim_serial_driver {

constexpr float d2r(const float deg) {
    return deg * M_PI / 180.0;
}

class SerialDriverNode: public rclcpp::Node {
public:
    explicit SerialDriverNode(const rclcpp::NodeOptions& options);
    ~SerialDriverNode() = default;

private:
    void get_parameters();
    void receive_imu_data();
    std::unique_ptr<IoContext> owned_ctx_;
    std::unique_ptr<drivers::serial_driver::SerialPortConfig> device_config_;
    std::unique_ptr<drivers::serial_driver::SerialDriver> serial_driver_;
    std::thread receive_thread_;

    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

SerialDriverNode::SerialDriverNode(const rclcpp::NodeOptions& options):
    Node("autoaim_serial_driver", options) {
    owned_ctx_ = std::make_unique<IoContext>(2);
    serial_driver_ = std::make_unique<drivers::serial_driver::SerialDriver>(*owned_ctx_);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    get_parameters();
    constexpr int MAX_ATTEMPTS = 5;
    for (int i = 0; i < MAX_ATTEMPTS && rclcpp::ok(); i++) {
        try {
            if (!serial_driver_->port()->is_open()) {
                serial_driver_->port()->open();
                RCLCPP_INFO(get_logger(), "Serial driver opened.");
                receive_thread_ = std::thread(&SerialDriverNode::receive_imu_data, this);
                break;
            }
        } catch (const std::exception& ex) {
            RCLCPP_ERROR(get_logger(), ex.what());
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    if (!serial_driver_->port()->is_open() && rclcpp::ok()) {
        RCLCPP_WARN(get_logger(), "Cannot open imu serial driver after %d attempts.", MAX_ATTEMPTS);
        RCLCPP_WARN(get_logger(), "Publishing fake imu data.");
        receive_thread_ = std::thread([&]() {
            while (rclcpp::ok()) {
                geometry_msgs::msg::TransformStamped gimbal_to_world;
                gimbal_to_world.header.stamp = now();
                gimbal_to_world.header.frame_id = "world";
                gimbal_to_world.child_frame_id = "gimbal";
                tf_broadcaster_->sendTransform(gimbal_to_world);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }
}

void SerialDriverNode::get_parameters() {
    std::string imu_device_name = declare_parameter("imu_device_name", "/dev/ttyACM0");
    int baud_rate = declare_parameter("baud_rate", 115200);

    device_config_ = std::make_unique<drivers::serial_driver::SerialPortConfig>(
        baud_rate,
        drivers::serial_driver::FlowControl::NONE,
        drivers::serial_driver::Parity::NONE,
        drivers::serial_driver::StopBits::ONE
    );
    serial_driver_->init_port(imu_device_name, *device_config_);
}

void SerialDriverNode::receive_imu_data() {
    std::vector<uint8_t> header(1);
    std::vector<uint8_t> data;
    while (rclcpp::ok()) {
        data.reserve(sizeof(ReceivePacketImu));
        data.resize(sizeof(ReceivePacketImu) - 1);
        serial_driver_->port()->receive(header);
        while (header[0] != 0xB5 && rclcpp::ok()) {
            serial_driver_->port()->receive(header);
        }
        serial_driver_->port()->receive(data);
        rclcpp::Time recv_time = now();
        data.insert(data.begin(), header[0]);
        ReceivePacketImu packet = serial_utils::from_vector<ReceivePacketImu>(data);
        bool crc_ok = crc::verify_crc8_checksum(reinterpret_cast<const uint8_t *>(&packet), sizeof(packet));
        if (!crc_ok) {
            RCLCPP_WARN(get_logger(), "CRC check failed");
            continue;
        }
        geometry_msgs::msg::TransformStamped gimbal_to_world;
        gimbal_to_world.header.stamp = recv_time;
        gimbal_to_world.header.frame_id = "world";
        gimbal_to_world.child_frame_id = "gimbal";
        gimbal_to_world.transform.rotation.x = serial_utils::swap_bytes_of_int16(packet.x) / 32768.0;
        gimbal_to_world.transform.rotation.y = serial_utils::swap_bytes_of_int16(packet.y) / 32768.0;
        gimbal_to_world.transform.rotation.z = serial_utils::swap_bytes_of_int16(packet.z) / 32768.0;
        gimbal_to_world.transform.rotation.w = serial_utils::swap_bytes_of_int16(packet.w) / 32768.0;;
        tf_broadcaster_->sendTransform(gimbal_to_world);
    }
}
} // namespace autoaim_serial_driver

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_serial_driver::SerialDriverNode)