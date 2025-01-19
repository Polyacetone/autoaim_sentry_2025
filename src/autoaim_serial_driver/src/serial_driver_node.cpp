#include <string>

#include <rclcpp/rclcpp.hpp>
#include <tf2/convert.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <serial_driver/serial_driver.hpp>

#include <autoaim_interfaces/msg/detection_array.hpp>
#include <autoaim_interfaces/msg/comm_recv.hpp>
#include <autoaim_interfaces/msg/comm_send.hpp>

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
    void receive_data_infantry();
    void send_data_infantry(const autoaim_interfaces::msg::CommSend::SharedPtr msg);
    std::unique_ptr<IoContext> owned_ctx_;
    std::unique_ptr<drivers::serial_driver::SerialPortConfig> device_config_;
    std::unique_ptr<drivers::serial_driver::SerialDriver> serial_driver_;
    std::thread receive_thread_;

    rclcpp::Subscription<autoaim_interfaces::msg::CommSend>::SharedPtr comm_send_;
    rclcpp::Publisher<autoaim_interfaces::msg::CommRecv>::SharedPtr comm_recv_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    int debug_mode_;
    bool enable_debug_;
};

SerialDriverNode::SerialDriverNode(const rclcpp::NodeOptions& options):
    Node("autoaim_serial_driver", options) {
    owned_ctx_ = std::make_unique<IoContext>(2);
    serial_driver_ = std::make_unique<drivers::serial_driver::SerialDriver>(*owned_ctx_);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    get_parameters();
    constexpr int MAX_ATTEMPTS = 10;
    for (int i = 0; i < MAX_ATTEMPTS && rclcpp::ok(); i++) {
        try {
            if (!serial_driver_->port()->is_open()) {
                serial_driver_->port()->open();
                RCLCPP_INFO(get_logger(), "Serial driver opened.");
                receive_thread_ = std::thread(&SerialDriverNode::receive_data_infantry, this);
                break;
            }
        } catch (const std::exception& ex) {
            RCLCPP_ERROR(get_logger(), ex.what());
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    if (!serial_driver_->port()->is_open() && rclcpp::ok()) {
        RCLCPP_WARN(get_logger(), "Cannot open serial driver after %d attempts.", MAX_ATTEMPTS);
        RCLCPP_WARN(get_logger(), "Publishing fake imu data.");
        receive_thread_ = std::thread([&]() {
            while (rclcpp::ok()) {
                geometry_msgs::msg::TransformStamped spindle_to_imu;
                spindle_to_imu.header.stamp = now();
                spindle_to_imu.header.frame_id = "world";
                spindle_to_imu.child_frame_id = "spindle";
                tf_broadcaster_->sendTransform(spindle_to_imu);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }
}

void SerialDriverNode::get_parameters() {
    std::string comm_send_topic_name = declare_parameter("comm_send_topic", "/serial/comm_send");
    std::string comm_recv_topic_name = declare_parameter("comm_recv_topic", "/serial/comm_recv");
    std::string device_name = declare_parameter("device_name", "/dev/ttyACM0");
    int baud_rate = declare_parameter("baud_rate", 115200);
    enable_debug_ = declare_parameter("enable_debug", true);
    debug_mode_ = declare_parameter("debug_mode", 0);

    device_config_ = std::make_unique<drivers::serial_driver::SerialPortConfig>(
        baud_rate,
        drivers::serial_driver::FlowControl::NONE,
        drivers::serial_driver::Parity::NONE,
        drivers::serial_driver::StopBits::ONE
    );
    serial_driver_->init_port(device_name, *device_config_);
    comm_recv_ = create_publisher<autoaim_interfaces::msg::CommRecv>(comm_recv_topic_name, 10);
    comm_send_ = create_subscription<autoaim_interfaces::msg::CommSend>(
        comm_send_topic_name,
        10,
        [&](const autoaim_interfaces::msg::CommSend::SharedPtr msg) { send_data_infantry(msg); }
    );
}

void SerialDriverNode::receive_data_infantry() {
    std::vector<uint8_t> header(1);
    std::vector<uint8_t> data;
    while (rclcpp::ok()) {
        data.reserve(sizeof(ReceivePacketInfantry));
        data.resize(sizeof(ReceivePacketInfantry) - 1);
        serial_driver_->port()->receive(header);
        while (header[0] != 0x3f) {
            serial_driver_->port()->receive(header);
        }
        serial_driver_->port()->receive(data);
        rclcpp::Time recv_time = now();
        data.insert(data.begin(), header[0]);
        ReceivePacketInfantry packet = serial_utils::from_vector<ReceivePacketInfantry>(data);
        autoaim_interfaces::msg::CommRecv comm_recv;
        comm_recv.header.stamp = recv_time;
        comm_recv.pitch = serial_utils::swap_bytes_of_int16(packet.pitch) / 100.0;
        comm_recv.yaw = serial_utils::swap_bytes_of_int16(packet.yaw) / 100.0;
        comm_recv.roll = serial_utils::swap_bytes_of_int16(packet.roll) / 100.0;
        comm_recv.shoot_speed = serial_utils::swap_bytes_of_int16(packet.shoot_speed) / 100.0;
        comm_recv.mode = 0;
        comm_recv_->publish(comm_recv);

        geometry_msgs::msg::TransformStamped gimbal_to_world;
        gimbal_to_world.header.stamp = recv_time;
        gimbal_to_world.header.frame_id = "world";
        gimbal_to_world.child_frame_id = "gimbal";
        tf2::Quaternion quat_gimbal;
        quat_gimbal.setRPY(d2r(comm_recv.pitch), d2r(comm_recv.roll), d2r(comm_recv.yaw));
        gimbal_to_world.transform.rotation.x = quat_gimbal.x();
        gimbal_to_world.transform.rotation.y = quat_gimbal.y();
        gimbal_to_world.transform.rotation.z = quat_gimbal.z();
        gimbal_to_world.transform.rotation.w = quat_gimbal.w();
        tf_broadcaster_->sendTransform(gimbal_to_world);

        /*geometry_msgs::msg::TransformStamped fric_wheel_to_world;
        fric_wheel_to_world.header.stamp = recv_time;
        fric_wheel_to_world.header.frame_id = "world";
        fric_wheel_to_world.child_frame_id = "fric_wheel";
        fric_wheel_to_world.transform.translation.x = -cos(d2r(comm_recv.pitch))
            * sin(d2r(comm_recv.yaw)) * distance_fric_wheel_to_world_;
        fric_wheel_to_world.transform.translation.y = cos(d2r(comm_recv.pitch)) 
            * cos(d2r(comm_recv.yaw)) * distance_fric_wheel_to_world_;
        fric_wheel_to_world.transform.translation.z =
            sin(d2r(comm_recv.pitch)) * distance_fric_wheel_to_world_;
        tf2::Quaternion quat_fric;
        quat_fric.setRPY(0, 0, d2r(comm_recv.yaw));
        fric_wheel_to_world.transform.rotation.x = quat_fric.x();
        fric_wheel_to_world.transform.rotation.y = quat_fric.y();
        fric_wheel_to_world.transform.rotation.z = quat_fric.z();
        fric_wheel_to_world.transform.rotation.w = quat_fric.w();
        tf_broadcaster_->sendTransform(fric_wheel_to_world);*/
    }
}

void SerialDriverNode::send_data_infantry(const autoaim_interfaces::msg::CommSend::SharedPtr msg) {
    SendPacketInfantry packet;
    packet.from_msg(msg);
    std::vector<uint8_t> packet_vector = serial_utils::to_vector(packet);
    serial_driver_->port()->send(packet_vector);
}
} // namespace autoaim_serial_driver

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_serial_driver::SerialDriverNode)