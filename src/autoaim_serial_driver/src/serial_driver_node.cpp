#include <rclcpp/rclcpp.hpp>
#include <tf2/convert.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_ros/static_transform_broadcaster.h>
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
class SerialDriverNode: public rclcpp::Node {
public:
    explicit SerialDriverNode(const rclcpp::NodeOptions& options);
    ~SerialDriverNode() = default;

private:
    void get_parameters();
    void receive_hard_trigger_imu_data();
    void receive_controller_imu_data();
    void send_shoot_pos(const autoaim_interfaces::msg::ShootPos::SharedPtr msg);
    void publish_fake_imu_data();

    std::unique_ptr<IoContext> owned_ctx_;
    std::unique_ptr<drivers::serial_driver::SerialDriver> serial_driver_;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_static_broadcaster_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::shared_ptr<rclcpp::Subscription<autoaim_interfaces::msg::ShootPos>> shoot_pos_sub_;
    std::thread receive_thread_;

    bool enable_hard_trigger_;
    bool enable_shoot_;
};

SerialDriverNode::SerialDriverNode(const rclcpp::NodeOptions& options):
    Node("autoaim_serial_driver", options) {
    owned_ctx_ = std::make_unique<IoContext>(2);
    serial_driver_ = std::make_unique<drivers::serial_driver::SerialDriver>(*owned_ctx_);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    tf_static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    get_parameters();
    constexpr int MAX_ATTEMPTS = 5;
    for (int i = 0; i < MAX_ATTEMPTS && rclcpp::ok(); i++) {
        try {
            if (!serial_driver_->port()->is_open()) {
                serial_driver_->port()->open();
                RCLCPP_INFO(get_logger(), "Serial driver opened.");
                if (enable_hard_trigger_) {
                    receive_thread_ = std::thread([&]() { receive_hard_trigger_imu_data(); });
                } else {
                    receive_thread_ = std::thread([&]() { receive_controller_imu_data(); });
                }
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
        receive_thread_ = std::thread([&]() { publish_fake_imu_data(); });
    }
}

void SerialDriverNode::get_parameters() {
    enable_hard_trigger_ = declare_parameter("enable_hard_trigger", false);
    enable_shoot_ = declare_parameter("enable_shoot", true);
    std::string shoot_pos_sub_topic_ =
        declare_parameter("shoot_pos_sub_topic", "/serial/shoot_pos");
    shoot_pos_sub_ = create_subscription<autoaim_interfaces::msg::ShootPos>(
        shoot_pos_sub_topic_,
        10,
        [&](const autoaim_interfaces::msg::ShootPos::SharedPtr msg) { send_shoot_pos(msg); }
    );

    if (enable_hard_trigger_) {
        // TODO
    } else {
        std::string controller_tty_name = declare_parameter("controller_tty_name", "/dev/ttyACM0");
        int baud_rate = declare_parameter("baud_rate", 115200);
        auto device_config_ = std::make_unique<drivers::serial_driver::SerialPortConfig>(
            baud_rate,
            drivers::serial_driver::FlowControl::NONE,
            drivers::serial_driver::Parity::NONE,
            drivers::serial_driver::StopBits::ONE
        );
        serial_driver_->init_port(controller_tty_name, *device_config_);
    }

    const double cam_to_gimbal_x = declare_parameter("cam_to_gimbal_x", 0.0);
    const double cam_to_gimbal_y = declare_parameter("cam_to_gimbal_y", 0.06);
    const double cam_to_gimbal_z = declare_parameter("cam_to_gimbal_z", 0.06);
    const double cam_to_gimbal_yaw = declare_parameter("cam_to_gimbal_yaw", 0.0);
    const double cam_to_gimbal_pitch = declare_parameter("cam_to_gimbal_pitch", 0.0);
    geometry_msgs::msg::TransformStamped cam_to_gimbal;
    cam_to_gimbal.header.stamp = this->now();
    cam_to_gimbal.header.frame_id = "gimbal";
    cam_to_gimbal.child_frame_id = "autoaim_camera";
    cam_to_gimbal.transform.translation.x = cam_to_gimbal_x;
    cam_to_gimbal.transform.translation.y = cam_to_gimbal_y;
    cam_to_gimbal.transform.translation.z = cam_to_gimbal_z;
    tf2::Quaternion rotation;
    // setRPY绕固定轴旋转。旋转顺序是绕XYZ。
    rotation.setRPY(cam_to_gimbal_pitch, 0, cam_to_gimbal_yaw);
    cam_to_gimbal.transform.rotation.x = rotation.x();
    cam_to_gimbal.transform.rotation.y = rotation.y();
    cam_to_gimbal.transform.rotation.z = rotation.z();
    cam_to_gimbal.transform.rotation.w = rotation.w();
    tf_static_broadcaster_->sendTransform(cam_to_gimbal);
}

void SerialDriverNode::receive_hard_trigger_imu_data() {
    std::vector<uint8_t> header(1);
    std::vector<uint8_t> data;
    while (rclcpp::ok()) {
        data.reserve(sizeof(ReceiveFromImu));
        data.resize(sizeof(ReceiveFromImu) - 1);
        serial_driver_->port()->receive(header);
        while (header[0] != 0xB5 && rclcpp::ok()) {
            serial_driver_->port()->receive(header);
        }
        serial_driver_->port()->receive(data);
        rclcpp::Time recv_time = now();
        data.insert(data.begin(), header[0]);
        ReceiveFromImu packet = serial_utils::from_vector<ReceiveFromImu>(data);
        bool crc_ok =
            crc::verify_crc8_checksum(reinterpret_cast<const uint8_t*>(&packet), sizeof(packet));
        if (!crc_ok) {
            RCLCPP_WARN(get_logger(), "CRC check failed");
            continue;
        }
        geometry_msgs::msg::TransformStamped gimbal_to_world;
        gimbal_to_world.header.stamp = recv_time;
        gimbal_to_world.header.frame_id = "world";
        gimbal_to_world.child_frame_id = "gimbal";
        gimbal_to_world.transform.rotation.x =
            serial_utils::swap_bytes_of_int16(packet.x) / 32768.0;
        gimbal_to_world.transform.rotation.y =
            serial_utils::swap_bytes_of_int16(packet.y) / 32768.0;
        gimbal_to_world.transform.rotation.z =
            serial_utils::swap_bytes_of_int16(packet.z) / 32768.0;
        gimbal_to_world.transform.rotation.w =
            serial_utils::swap_bytes_of_int16(packet.w) / 32768.0;
        tf_broadcaster_->sendTransform(gimbal_to_world);
    }
}

void SerialDriverNode::receive_controller_imu_data() {
    std::vector<uint8_t> header(1);
    std::vector<uint8_t> data;
    while (rclcpp::ok()) {
        data.reserve(sizeof(ReceiveFromController));
        data.resize(sizeof(ReceiveFromController) - 1);
        serial_driver_->port()->receive(header);
        while (header[0] != 0x1B && rclcpp::ok()) {
            serial_driver_->port()->receive(header);
        }
        serial_driver_->port()->receive(data);
        rclcpp::Time recv_time = now();
        data.insert(data.begin(), header[0]);
        ReceiveFromController packet = serial_utils::from_vector<ReceiveFromController>(data);
        bool crc_ok =
            crc::verify_crc8_checksum(reinterpret_cast<const uint8_t*>(&packet), sizeof(packet));
        if (!crc_ok) {
            RCLCPP_WARN(get_logger(), "CRC check failed");
            continue;
        }
        geometry_msgs::msg::TransformStamped gimbal_to_world;
        gimbal_to_world.header.stamp = recv_time;
        gimbal_to_world.header.frame_id = "world";
        gimbal_to_world.child_frame_id = "gimbal";
        tf2::Quaternion rotation;
        // setRPY绕固定轴旋转。旋转顺序是绕XYZ。
        rotation.setRPY(packet.pitch / 1e4, 0, (packet.yaw + packet.big_yaw) / 1e4);
        gimbal_to_world.transform.rotation.w = rotation.w();
        gimbal_to_world.transform.rotation.x = rotation.x();
        gimbal_to_world.transform.rotation.y = rotation.y();
        gimbal_to_world.transform.rotation.z = rotation.z();
        tf_broadcaster_->sendTransform(gimbal_to_world);
    }
}

void SerialDriverNode::publish_fake_imu_data() {
    while (rclcpp::ok()) {
        geometry_msgs::msg::TransformStamped gimbal_to_world;
        gimbal_to_world.header.stamp = now();
        gimbal_to_world.header.frame_id = "world";
        gimbal_to_world.child_frame_id = "gimbal";
        tf_broadcaster_->sendTransform(gimbal_to_world);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void SerialDriverNode::send_shoot_pos(const autoaim_interfaces::msg::ShootPos::SharedPtr msg) {
    SendToController packet;
    packet.pitch_angle = static_cast<int16_t>(msg->pitch * 1e4);
    packet.yaw_angle = static_cast<int16_t>(msg->yaw * 1e4);
    if (enable_shoot_) {
        packet.shoot_flag = msg->shoot_flag;
    } else {
        packet.shoot_flag = 0;
    }
    std::vector<uint8_t> packet_vector = serial_utils::to_vector(packet);
    crc::append_crc8_checksum(packet_vector.data(), packet_vector.size());
    serial_driver_->port()->send(packet_vector);
}
} // namespace autoaim_serial_driver

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_serial_driver::SerialDriverNode)