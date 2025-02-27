#pragma once

#include <cstdint>
#include <geometry_msgs/msg/twist.hpp>
#include <autoaim_interfaces/msg/chassis_status.hpp>

class SendToController {
public:
    uint8_t header = 0xA2;
    uint8_t length = 9;
    int16_t pitch_angle;
    int16_t yaw_angle;
    uint8_t shoot_flag;
    uint8_t fly_t_to_send;
    uint8_t checksum;
} __attribute__((packed));

class TransmitPacketVel {
public:
    uint8_t header = 0xA1;
    uint8_t length = 9;
    uint16_t vx;
    uint16_t vy;
    uint16_t wz;
    uint8_t checksum;

    void fromMsg(const geometry_msgs::msg::Twist::SharedPtr& msg) {
        vx = static_cast<int16_t>(msg->linear.x * 100);
        vy = static_cast<int16_t>(msg->linear.y * 100);
        wz = static_cast<int16_t>(msg->angular.z * 100);
    }

    int getSize() {
        return length;
    }
} __attribute__((packed));

class TransmitPacketChassisStatusPacket {
public:
    uint8_t header = 0xA3;
    uint8_t length = 4;
    uint8_t chassis_status;
    uint8_t checksum;

    void fromMsg(const autoaim_interfaces::msg::ChassisStatus::SharedPtr& msg) {
        chassis_status = msg->chassis_status;
    }

    int getSize() {
        return length;
    }
} __attribute__((packed));