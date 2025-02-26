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

#include <autoaim_interfaces/msg/comp_robots_hp.hpp>
#include <autoaim_interfaces/msg/comp_status.hpp>
#include <autoaim_interfaces/msg/inter_map_client_to_robot.hpp>
#include <autoaim_interfaces/msg/robot_ground_pos.hpp>
#include <autoaim_interfaces/msg/robot_performance.hpp>
#include <autoaim_interfaces/msg/robot_pos.hpp>
#include <autoaim_interfaces/msg/robot_resource.hpp>
#include <autoaim_interfaces/msg/robot_rfid.hpp>
#include <autoaim_interfaces/msg/robot_sentry_decision.hpp>
#include <autoaim_interfaces/msg/team_event.hpp>
#include <autoaim_interfaces/msg/chassis_status.hpp>

#include <serial_utils.hpp>
#include <receive_packet.hpp>
#include <send_packet.hpp>
#include <crc_checksum.hpp>

namespace autoaim_serial_driver
{
    class SerialDriverNode : public rclcpp::Node
    {
    public:
        explicit SerialDriverNode(const rclcpp::NodeOptions &options);
        ~SerialDriverNode() = default;

    private:
        void get_parameters();
        void receive_data();
        bool is_header_valid(uint8_t header);
        void receive_hard_trigger_imu_data(uint8_t header);
        void receive_controller_imu_data(uint8_t header);
        void receive_rfr_comp_robots_hp_data(uint8_t header);
        void receive_rfr_comp_status_data(uint8_t header);
        void receive_rfr_inter_map_client_to_robot_data(uint8_t header);
        void receive_rfr_robot_ground_pos_data(uint8_t header);
        void receive_rfr_robot_performance_data(uint8_t header);
        void receive_rfr_robot_pos_data(uint8_t header);
        void receive_rfr_robot_resource_data(uint8_t header);
        void receive_rfr_robot_rfid_data(uint8_t header);
        void receive_rfr_robot_sentry_decision_data(uint8_t header);
        void receive_rfr_team_event_data(uint8_t header);
        void send_shoot_pos(const autoaim_interfaces::msg::ShootPos::SharedPtr msg);
        void send_vel(const geometry_msgs::msg::Twist::SharedPtr msg);
        void send_chassis_status(const autoaim_interfaces::msg::ChassisStatus::SharedPtr msg);
        void publish_fake_imu_data();
        void reopenPort();

        std::unique_ptr<IoContext> owned_ctx_;
        std::unique_ptr<drivers::serial_driver::SerialDriver> serial_driver_;
        std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_static_broadcaster_;
        std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
        std::shared_ptr<rclcpp::Subscription<autoaim_interfaces::msg::ShootPos>> shoot_pos_sub_;
        std::thread receive_thread_;

        std::shared_ptr<rclcpp::Subscription<geometry_msgs::msg::Twist>> vel_sub_;
        std::shared_ptr<rclcpp::Subscription<autoaim_interfaces::msg::ChassisStatus>>
            chassis_status_sub_;

        std::shared_ptr<rclcpp::Publisher<autoaim_interfaces::msg::CompRobotsHp>> rfr_comp_robots_hp_pub_;
        std::shared_ptr<rclcpp::Publisher<autoaim_interfaces::msg::CompStatus>> rfr_comp_status_pub_;
        std::shared_ptr<rclcpp::Publisher<autoaim_interfaces::msg::InterMapClientToRobot>>
            rfr_inter_map_client_to_robot_pub_;
        std::shared_ptr<rclcpp::Publisher<autoaim_interfaces::msg::RobotGroundPos>>
            rfr_robot_ground_pos_pub_;
        std::shared_ptr<rclcpp::Publisher<autoaim_interfaces::msg::RobotPerformance>>
            rfr_robot_performance_pub_;
        std::shared_ptr<rclcpp::Publisher<autoaim_interfaces::msg::RobotPos>> rfr_robot_pos_pub_;
        std::shared_ptr<rclcpp::Publisher<autoaim_interfaces::msg::RobotResource>> rfr_robot_resource_pub_;
        std::shared_ptr<rclcpp::Publisher<autoaim_interfaces::msg::RobotRfid>> rfr_robot_rfid_pub_;
        std::shared_ptr<rclcpp::Publisher<autoaim_interfaces::msg::RobotSentryDecision>>
            rfr_robot_sentry_decision_pub_;
        std::shared_ptr<rclcpp::Publisher<autoaim_interfaces::msg::TeamEvent>> rfr_team_event_pub_;

        bool enable_hard_trigger_;
        bool enable_shoot_;

        std::string vel_sub_topic_;
        std::string chassis_status_sub_topic_;

        uint8_t rx_header_list[12] =
            {0xB5, 0x1B, 0x03, 0x01, 0x33, 0x2B, 0x21, 0x23, 0x28, 0x29, 0x2D, 0x11};
        void (SerialDriverNode::*rx_function_list[12])(uint8_t) = {
            &SerialDriverNode::receive_hard_trigger_imu_data,
            &SerialDriverNode::receive_controller_imu_data,
            &SerialDriverNode::receive_rfr_comp_robots_hp_data,
            &SerialDriverNode::receive_rfr_comp_status_data,
            &SerialDriverNode::receive_rfr_inter_map_client_to_robot_data,
            &SerialDriverNode::receive_rfr_robot_ground_pos_data,
            &SerialDriverNode::receive_rfr_robot_performance_data,
            &SerialDriverNode::receive_rfr_robot_pos_data,
            &SerialDriverNode::receive_rfr_robot_resource_data,
            &SerialDriverNode::receive_rfr_robot_rfid_data,
            &SerialDriverNode::receive_rfr_robot_sentry_decision_data,
            &SerialDriverNode::receive_rfr_team_event_data};
    };

    SerialDriverNode::SerialDriverNode(const rclcpp::NodeOptions &options) : Node("autoaim_serial_driver", options)
    {
        owned_ctx_ = std::make_unique<IoContext>(2);
        serial_driver_ = std::make_unique<drivers::serial_driver::SerialDriver>(*owned_ctx_);
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        tf_static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
        get_parameters();
        constexpr int MAX_ATTEMPTS = 5;
        for (int i = 0; i < MAX_ATTEMPTS && rclcpp::ok(); i++)
        {
            try
            {
                if (!serial_driver_->port()->is_open())
                {
                    serial_driver_->port()->open();
                    RCLCPP_INFO(get_logger(), "Serial driver opened.");
                    receive_thread_ = std::thread([&]()
                                                  { receive_data(); });
                    break;
                }
            }
            catch (const std::exception &ex)
            {
                RCLCPP_ERROR(get_logger(), ex.what());
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        if (!serial_driver_->port()->is_open() && rclcpp::ok())
        {
            RCLCPP_WARN(get_logger(), "Cannot open imu serial driver after %d attempts.", MAX_ATTEMPTS);
            RCLCPP_WARN(get_logger(), "Publishing fake imu data.");
            receive_thread_ = std::thread([&]()
                                          { publish_fake_imu_data(); });
        }
    }

    void SerialDriverNode::get_parameters()
    {
        enable_hard_trigger_ = declare_parameter("enable_hard_trigger", false);
        enable_shoot_ = declare_parameter("enable_shoot", true);
        std::string shoot_pos_sub_topic_ =
            declare_parameter("shoot_pos_sub_topic", "/serial/shoot_pos");
        shoot_pos_sub_ = create_subscription<autoaim_interfaces::msg::ShootPos>(
            shoot_pos_sub_topic_,
            10,
            [&](const autoaim_interfaces::msg::ShootPos::SharedPtr msg)
            { send_shoot_pos(msg); });

        if (enable_hard_trigger_)
        {
            // TODO
        }
        else
        {
            std::string controller_tty_name = declare_parameter("controller_tty_name", "/dev/ttyACM0");
            int baud_rate = declare_parameter("baud_rate", 115200);
            auto device_config_ = std::make_unique<drivers::serial_driver::SerialPortConfig>(
                baud_rate,
                drivers::serial_driver::FlowControl::NONE,
                drivers::serial_driver::Parity::NONE,
                drivers::serial_driver::StopBits::ONE);
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

        rfr_comp_robots_hp_pub_ =
            create_publisher<autoaim_interfaces::msg::CompRobotsHp>("/info/rfr_comp_robots_hp", 10);
        rfr_comp_status_pub_ =
            create_publisher<autoaim_interfaces::msg::CompStatus>("/info/rfr_comp_status", 10);
        rfr_inter_map_client_to_robot_pub_ =
            create_publisher<autoaim_interfaces::msg::InterMapClientToRobot>(
                "/info/rfr_inter_map_client_to_robot",
                10);
        rfr_robot_ground_pos_pub_ =
            create_publisher<autoaim_interfaces::msg::RobotGroundPos>("/info/rfr_robot_ground_pos", 10);
        rfr_robot_performance_pub_ =
            create_publisher<autoaim_interfaces::msg::RobotPerformance>("/info/rfr_robot_performance", 10);
        rfr_robot_pos_pub_ = create_publisher<autoaim_interfaces::msg::RobotPos>("/info/rfr_robot_pos", 10);
        rfr_robot_resource_pub_ =
            create_publisher<autoaim_interfaces::msg::RobotResource>("/info/rfr_robot_resource", 10);
        rfr_robot_rfid_pub_ =
            create_publisher<autoaim_interfaces::msg::RobotRfid>("/info/rfr_robot_rfid", 10);
        rfr_robot_sentry_decision_pub_ = create_publisher<autoaim_interfaces::msg::RobotSentryDecision>(
            "/info/rfr_robot_sentry_decision",
            10);
        rfr_team_event_pub_ =
            create_publisher<autoaim_interfaces::msg::TeamEvent>("/info/rfr_team_event", 10);

        vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
            vel_sub_topic_,
            10,
            [&](const geometry_msgs::msg::Twist::SharedPtr msg)
            { send_vel(msg); });

        chassis_status_sub_ = create_subscription<autoaim_interfaces::msg::ChassisStatus>(
            chassis_status_sub_topic_,
            10,
            [&](const autoaim_interfaces::msg::ChassisStatus::SharedPtr msg)
            {
                send_chassis_status(msg);
            });

        vel_sub_topic_ = declare_parameter("vel_sub_topic", "/cmd/vel");
        chassis_status_sub_topic_ = declare_parameter("chassis_status_sub_topic", "/chassis_status");
    }

    void SerialDriverNode::receive_data()
    {
        std::vector<uint8_t> header(1);

        while (rclcpp::ok())
        {
            try
            {
                serial_driver_->port()->receive(header);
                while (is_header_valid(header[0]) && rclcpp::ok())
                {
                    serial_driver_->port()->receive(header);
                }
            }
            catch (const std::exception &ex)
            {
                serial_driver_->port()->receive(header);
                RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 20, "Error while receiving data: %s", ex.what());
                reopenPort();
            }
        }
    }

    bool SerialDriverNode::is_header_valid(uint8_t header)
    {
        for (int i = 0; i < sizeof(rx_header_list) / sizeof(rx_header_list[0]); i++)
        {
            if (header == rx_header_list[i])
            {
                // 如果帧头符合要求，调用对应的处理函数
                (this->*rx_function_list[i])(header);
                return true;
            }
        }
        return false;
    }

    void SerialDriverNode::receive_rfr_comp_robots_hp_data(uint8_t header)
    {
        std::vector<uint8_t> data;
        data.reserve(sizeof(ReceivePacketRfrCompRobotsHp));
        data.resize(sizeof(ReceivePacketRfrCompRobotsHp) - 1);
        serial_driver_->port()->receive(data);
        data.insert(data.begin(), header);
        ReceivePacketRfrCompRobotsHp packet =
            serial_utils::from_vector<ReceivePacketRfrCompRobotsHp>(data);
        bool crc_ok =
            crc::verify_crc8_checksum(reinterpret_cast<const uint8_t *>(&packet), sizeof(packet));
        if (crc_ok)
        {
            autoaim_interfaces::msg::CompRobotsHp comp_robots_hp;
            comp_robots_hp.red_1_robot_hp = packet.red_1_robot_HP;
            comp_robots_hp.red_2_robot_hp = packet.red_2_robot_HP;
            comp_robots_hp.red_3_robot_hp = packet.red_3_robot_HP;
            comp_robots_hp.red_4_robot_hp = packet.red_4_robot_HP;
            comp_robots_hp.red_7_robot_hp = packet.red_7_robot_HP;
            comp_robots_hp.red_outpost_hp = packet.red_outpost_HP;
            comp_robots_hp.red_base_hp = packet.red_base_HP;
            comp_robots_hp.blue_1_robot_hp = packet.blue_1_robot_HP;
            comp_robots_hp.blue_2_robot_hp = packet.blue_2_robot_HP;
            comp_robots_hp.blue_3_robot_hp = packet.blue_3_robot_HP;
            comp_robots_hp.blue_4_robot_hp = packet.blue_4_robot_HP;
            comp_robots_hp.blue_7_robot_hp = packet.blue_7_robot_HP;
            comp_robots_hp.blue_outpost_hp = packet.blue_outpost_HP;
            comp_robots_hp.blue_base_hp = packet.blue_base_HP;
            rfr_comp_robots_hp_pub_->publish(comp_robots_hp);
        }
        else
        {
            RCLCPP_WARN(get_logger(), "rfr_comp_robots_hp_data_ CRC check failed");
        }
    }

    void SerialDriverNode::receive_rfr_comp_status_data(uint8_t header)
    {
        std::vector<uint8_t> data;
        data.reserve(sizeof(ReceivePacketRfrCompStatus));
        data.resize(sizeof(ReceivePacketRfrCompStatus) - 1);
        serial_driver_->port()->receive(data);
        data.insert(data.begin(), header);
        ReceivePacketRfrCompStatus packet = serial_utils::from_vector<ReceivePacketRfrCompStatus>(data);
        bool crc_ok =
            crc::verify_crc8_checksum(reinterpret_cast<const uint8_t *>(&packet), sizeof(packet));
        if (crc_ok)
        {
            autoaim_interfaces::msg::CompStatus comp_status;
            comp_status.game_progress = packet.game_progress;
            comp_status.stage_remain_time = packet.stage_remain_time;
            if (comp_status.stage_remain_time < 999)
            {
                rfr_comp_status_pub_->publish(comp_status);
            }
        }
        else
        {
            RCLCPP_WARN(get_logger(), "rfr_comp_status_data_ CRC check failed");
        }
    }

    void SerialDriverNode::receive_rfr_inter_map_client_to_robot_data(uint8_t header)
    {
        std::vector<uint8_t> data;
        data.reserve(sizeof(ReceivePacketRfrInterMapClientToRobot));
        data.resize(sizeof(ReceivePacketRfrInterMapClientToRobot) - 1);
        serial_driver_->port()->receive(data);
        data.insert(data.begin(), header);
        ReceivePacketRfrInterMapClientToRobot packet =
            serial_utils::from_vector<ReceivePacketRfrInterMapClientToRobot>(data);
        bool crc_ok =
            crc::verify_crc8_checksum(reinterpret_cast<const uint8_t *>(&packet), sizeof(packet));
        if (crc_ok)
        {
            autoaim_interfaces::msg::InterMapClientToRobot inter_map_client_to_robot;
            inter_map_client_to_robot.target_position_x = packet.target_position_x;
            inter_map_client_to_robot.target_position_y = packet.target_position_y;
            inter_map_client_to_robot.cmd_keyboard = packet.cmd_keyboard;
            inter_map_client_to_robot.target_robot_id = packet.target_robot_id;
            inter_map_client_to_robot.cmd_source = packet.cmd_source;
            rfr_inter_map_client_to_robot_pub_->publish(inter_map_client_to_robot);
        }
        else
        {
            RCLCPP_WARN(get_logger(), "rfr_inter_map_client_to_robot_data_ CRC check failed");
        }
    }

    void SerialDriverNode::receive_rfr_robot_ground_pos_data(uint8_t header)
    {
        std::vector<uint8_t> data;
        data.reserve(sizeof(ReceivePacketRfrRobotGroundPos));
        data.resize(sizeof(ReceivePacketRfrRobotGroundPos) - 1);
        serial_driver_->port()->receive(data);
        data.insert(data.begin(), header);
        ReceivePacketRfrRobotGroundPos packet =
            serial_utils::from_vector<ReceivePacketRfrRobotGroundPos>(data);
        bool crc_ok =
            crc::verify_crc8_checksum(reinterpret_cast<const uint8_t *>(&packet), sizeof(packet));
        if (crc_ok)
        {
            autoaim_interfaces::msg::RobotGroundPos robot_ground_pos;
            robot_ground_pos.hero_x = packet.hero_x;
            robot_ground_pos.hero_y = packet.hero_y;
            robot_ground_pos.engineer_x = packet.engineer_x;
            robot_ground_pos.engineer_y = packet.engineer_y;
            robot_ground_pos.standard_3_x = packet.standard_3_x;
            robot_ground_pos.standard_3_y = packet.standard_3_y;
            robot_ground_pos.standard_4_x = packet.standard_4_x;
            robot_ground_pos.standard_4_y = packet.standard_4_y;
            rfr_robot_ground_pos_pub_->publish(robot_ground_pos);
        }
        else
        {
            RCLCPP_WARN(get_logger(), "rfr_robot_ground_pos_data_ CRC check failed");
        }
    }

    void SerialDriverNode::receive_rfr_robot_performance_data(uint8_t header)
    {
        std::vector<uint8_t> data;
        data.reserve(sizeof(ReceivePacketRfrRobotPerformance));
        data.resize(sizeof(ReceivePacketRfrRobotPerformance) - 1);
        serial_driver_->port()->receive(data);
        data.insert(data.begin(), header);
        ReceivePacketRfrRobotPerformance packet =
            serial_utils::from_vector<ReceivePacketRfrRobotPerformance>(data);
        bool crc_ok =
            crc::verify_crc8_checksum(reinterpret_cast<const uint8_t *>(&packet), sizeof(packet));
        if (crc_ok)
        {
            autoaim_interfaces::msg::RobotPerformance robot_performance;
            robot_performance.current_hp = packet.current_hp;
            robot_performance.power_management_shooter_output = packet.power_management_shooter_output;
            rfr_robot_performance_pub_->publish(robot_performance);
        }
        else
        {
            RCLCPP_WARN(get_logger(), "rfr_robot_performance_data_ CRC check failed");
        }
    }

    void SerialDriverNode::receive_rfr_robot_pos_data(uint8_t header)
    {
        std::vector<uint8_t> data;
        data.reserve(sizeof(ReceivePacketRfrRobotPos));
        data.resize(sizeof(ReceivePacketRfrRobotPos) - 1);
        serial_driver_->port()->receive(data);
        data.insert(data.begin(), header);
        ReceivePacketRfrRobotPos packet = serial_utils::from_vector<ReceivePacketRfrRobotPos>(data);
        bool crc_ok =
            crc::verify_crc8_checksum(reinterpret_cast<const uint8_t *>(&packet), sizeof(packet));
        if (crc_ok)
        {
            autoaim_interfaces::msg::RobotPos robot_pos;
            robot_pos.x = packet.x;
            robot_pos.y = packet.y;
            robot_pos.angle = packet.angle;
            rfr_robot_pos_pub_->publish(robot_pos);
        }
        else
        {
            RCLCPP_WARN(get_logger(), "rfr_robot_pos_data_ CRC check failed");
        }
    }

    void SerialDriverNode::receive_rfr_robot_resource_data(uint8_t header)
    {
        std::vector<uint8_t> data;
        data.reserve(sizeof(ReceivePacketRfrRobotResource));
        data.resize(sizeof(ReceivePacketRfrRobotResource) - 1);
        serial_driver_->port()->receive(data);
        data.insert(data.begin(), header);
        ReceivePacketRfrRobotResource packet =
            serial_utils::from_vector<ReceivePacketRfrRobotResource>(data);
        bool crc_ok =
            crc::verify_crc8_checksum(reinterpret_cast<const uint8_t *>(&packet), sizeof(packet));
        if (crc_ok)
        {
            autoaim_interfaces::msg::RobotResource robot_resource;
            robot_resource.allowance_17mm = packet.allowance_17mm;
            robot_resource.remaining_coin = packet.remaining_coin;
            rfr_robot_resource_pub_->publish(robot_resource);
        }
        else
        {
            RCLCPP_WARN(get_logger(), "rfr_robot_resource_data_ CRC check failed");
        }
    }

    void SerialDriverNode::receive_rfr_robot_rfid_data(uint8_t header)
    {
        std::vector<uint8_t> data;
        data.reserve(sizeof(ReceivePacketRfrRobotRfid));
        data.resize(sizeof(ReceivePacketRfrRobotRfid) - 1);
        serial_driver_->port()->receive(data);
        data.insert(data.begin(), header);
        ReceivePacketRfrRobotRfid packet = serial_utils::from_vector<ReceivePacketRfrRobotRfid>(data);
        bool crc_ok =
            crc::verify_crc8_checksum(reinterpret_cast<const uint8_t *>(&packet), sizeof(packet));
        if (crc_ok)
        {
            autoaim_interfaces::msg::RobotRfid robot_rfid;
            robot_rfid.our_base = packet.our_base;
            robot_rfid.our_central_highland = packet.our_central_highland;
            robot_rfid.opp_central_highland = packet.opp_central_highland;
            robot_rfid.our_trapezoid_highland = packet.our_trapezoid_highland;
            robot_rfid.opp_trapezoid_highland = packet.opp_trapezoid_highland;
            robot_rfid.our_launch_front = packet.our_launch_front;
            robot_rfid.our_launch_back = packet.our_launch_back;
            robot_rfid.opp_launch_front = packet.opp_launch_front;
            robot_rfid.opp_launch_back = packet.opp_launch_back;
            robot_rfid.our_highland_bottom = packet.our_highland_bottom;
            robot_rfid.our_highland_top = packet.our_highland_top;
            robot_rfid.opp_highland_bottom = packet.opp_highland_bottom;
            robot_rfid.opp_highland_top = packet.opp_highland_top;
            robot_rfid.our_highway_bottom = packet.our_highway_bottom;
            robot_rfid.our_highway_top = packet.our_highway_top;
            robot_rfid.opp_highway_bottom = packet.opp_highway_bottom;
            robot_rfid.opp_highway_top = packet.opp_highway_top;
            robot_rfid.our_fort = packet.our_fort;
            robot_rfid.our_outpost = packet.our_outpost;
            robot_rfid.our_restoration_1 = packet.our_restoration_1;
            robot_rfid.our_restoration_2 = packet.our_restoration_2;
            robot_rfid.our_big_resource_island = packet.our_big_resource_island;
            robot_rfid.opp_big_resource_island = packet.opp_big_resource_island;
            robot_rfid.central_boost = packet.central_boost;
            rfr_robot_rfid_pub_->publish(robot_rfid);
        }
        else
        {
            RCLCPP_WARN(get_logger(), "rfr_robot_rfid_data_ CRC check failed");
        }
    }

    void SerialDriverNode::receive_rfr_robot_sentry_decision_data(uint8_t header)
    {
        std::vector<uint8_t> data;
        data.reserve(sizeof(ReceivePacketRfrRobotSentryDecision));
        data.resize(sizeof(ReceivePacketRfrRobotSentryDecision) - 1);
        serial_driver_->port()->receive(data);
        data.insert(data.begin(), header);
        ReceivePacketRfrRobotSentryDecision packet =
            serial_utils::from_vector<ReceivePacketRfrRobotSentryDecision>(data);
        bool crc_ok =
            crc::verify_crc8_checksum(reinterpret_cast<const uint8_t *>(&packet), sizeof(packet));
        if (crc_ok)
        {
            autoaim_interfaces::msg::RobotSentryDecision robot_sentry_decision;
            robot_sentry_decision.allowance = packet.allowance;
            robot_sentry_decision.remote_allowance = packet.remote_allowance;
            robot_sentry_decision.remote_hp = packet.remote_hp;
            robot_sentry_decision.allow_free_resurrection = packet.allow_free_resurrection;
            robot_sentry_decision.allow_redemption_resurrection = packet.allow_redemption_resurrection;
            robot_sentry_decision.redemption_resurrection_cost = packet.redemption_resurrection_cost;
            robot_sentry_decision.out_of_war_status = packet.out_of_war_status;
            robot_sentry_decision.remain_total_allowance = packet.remain_total_allowance;
            rfr_robot_sentry_decision_pub_->publish(robot_sentry_decision);
        }
        else
        {
            RCLCPP_WARN(get_logger(), "rfr_robot_sentry_decision_data_ CRC check failed");
        }
    }

    void SerialDriverNode::receive_rfr_team_event_data(uint8_t header)
    {
        std::vector<uint8_t> data;
        data.reserve(sizeof(ReceivePacketRfrTeamEvent));
        data.resize(sizeof(ReceivePacketRfrTeamEvent) - 1);
        serial_driver_->port()->receive(data);
        data.insert(data.begin(), header);
        ReceivePacketRfrTeamEvent packet = serial_utils::from_vector<ReceivePacketRfrTeamEvent>(data);
        bool crc_ok =
            crc::verify_crc8_checksum(reinterpret_cast<const uint8_t *>(&packet), sizeof(packet));
        if (crc_ok)
        {
            autoaim_interfaces::msg::TeamEvent team_event;
            team_event.restoration_1 = packet.restoration_1;
            team_event.restoration_2 = packet.restoration_2;
            team_event.supplier_area = packet.supplier_area;
            team_event.small_power_rune = packet.small_power_rune;
            team_event.large_power_rune = packet.large_power_rune;
            team_event.central_highland = packet.central_highland;
            team_event.trapezoid_highland = packet.trapezoid_highland;
            team_event.dart_last_hit_us_time = packet.dart_last_hit_us_time;
            team_event.dart_last_hit_us_type = packet.dart_last_hit_us_type;
            team_event.center_buff_area = packet.center_buff_area;
            rfr_team_event_pub_->publish(team_event);
        }
        else
        {
            RCLCPP_WARN(get_logger(), "rfr_team_event_data_ CRC check failed");
        }
    }

    void SerialDriverNode::receive_hard_trigger_imu_data(uint8_t header)
    {
        std::vector<uint8_t> data;
        data.reserve(sizeof(ReceiveFromImu));
        data.resize(sizeof(ReceiveFromImu) - 1);

        serial_driver_->port()->receive(data);
        rclcpp::Time recv_time = now();
        data.insert(data.begin(), header);
        ReceiveFromImu packet = serial_utils::from_vector<ReceiveFromImu>(data);
        bool crc_ok =
            crc::verify_crc8_checksum(reinterpret_cast<const uint8_t *>(&packet), sizeof(packet));
        if (!crc_ok)
        {
            RCLCPP_WARN(get_logger(), "CRC check failed");
            return;
        }
        geometry_msgs::msg::TransformStamped gimbal_to_world;
        gimbal_to_world.header.stamp = recv_time;
        gimbal_to_world.header.frame_id = "world";
        gimbal_to_world.child_frame_id = "gimbal";
        gimbal_to_world.transform.rotation.x = serial_utils::swap_bytes_of_int16(packet.x) / 32768.0;
        gimbal_to_world.transform.rotation.y = serial_utils::swap_bytes_of_int16(packet.y) / 32768.0;
        gimbal_to_world.transform.rotation.z = serial_utils::swap_bytes_of_int16(packet.z) / 32768.0;
        gimbal_to_world.transform.rotation.w = serial_utils::swap_bytes_of_int16(packet.w) / 32768.0;
        tf_broadcaster_->sendTransform(gimbal_to_world);
    }

    void SerialDriverNode::receive_controller_imu_data(uint8_t header)
    {
        std::vector<uint8_t> data;
        data.reserve(sizeof(ReceiveFromController));
        data.resize(sizeof(ReceiveFromController) - 1);
        serial_driver_->port()->receive(data);
        rclcpp::Time recv_time = now();
        data.insert(data.begin(), header);
        ReceiveFromController packet = serial_utils::from_vector<ReceiveFromController>(data);
        bool crc_ok =
            crc::verify_crc8_checksum(reinterpret_cast<const uint8_t *>(&packet), sizeof(packet));
        if (!crc_ok)
        {
            RCLCPP_WARN(get_logger(), "CRC check failed");
            return;
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

    void SerialDriverNode::publish_fake_imu_data()
    {
        while (rclcpp::ok())
        {
            geometry_msgs::msg::TransformStamped gimbal_to_world;
            gimbal_to_world.header.stamp = now();
            gimbal_to_world.header.frame_id = "world";
            gimbal_to_world.child_frame_id = "gimbal";
            tf_broadcaster_->sendTransform(gimbal_to_world);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    void SerialDriverNode::send_shoot_pos(const autoaim_interfaces::msg::ShootPos::SharedPtr msg)
    {
        SendToController packet;
        packet.pitch_angle = static_cast<int16_t>(msg->pitch * 1e4);
        packet.yaw_angle = static_cast<int16_t>(msg->yaw * 1e4);
        if (enable_shoot_)
        {
            packet.shoot_flag = msg->shoot_flag;
        }
        else
        {
            packet.shoot_flag = 0;
        }
        std::vector<uint8_t> packet_vector = serial_utils::to_vector(packet);
        crc::append_crc8_checksum(packet_vector.data(), packet_vector.size());
        serial_driver_->port()->send(packet_vector);
    }

    void SerialDriverNode::send_vel(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        TransmitPacketVel packet;
        packet.fromMsg(msg);
        std::vector<uint8_t> packet_vector = serial_utils::to_vector(packet);
        crc::append_crc8_checksum(packet_vector.data(), packet_vector.size());
        serial_driver_->port()->send(packet_vector);
    }

    void SerialDriverNode::send_chassis_status(
        const autoaim_interfaces::msg::ChassisStatus::SharedPtr msg)
    {
        TransmitPacketChassisStatusPacket packet;
        packet.fromMsg(msg);
        std::vector<uint8_t> packet_vector = serial_utils::to_vector(packet);
        crc::append_crc8_checksum(packet_vector.data(), packet_vector.size());
        serial_driver_->port()->send(packet_vector);
    }

    void SerialDriverNode::reopenPort()
    {
        RCLCPP_WARN(get_logger(), "Attempting to reopen port");
        try
        {
            if (serial_driver_->port()->is_open())
            {
                serial_driver_->port()->close();
            }
            serial_driver_->port()->open();
            RCLCPP_INFO(get_logger(), "Successfully reopened port");
        }
        catch (const std::exception &ex)
        {
            RCLCPP_ERROR(get_logger(), "Error while reopening port: %s", ex.what());
            if (rclcpp::ok())
            {
                rclcpp::sleep_for(std::chrono::seconds(1));
                reopenPort();
            }
        }
    }

} // namespace autoaim_serial_driver

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_serial_driver::SerialDriverNode)