std_msgs/Header header
# Robot data
# sensor_msgs/Imu imu
float32 roll                    # 机器人云台 roll (deg)   up
float32 pitch                   # 机器人云台 pitch (deg)  right
float32 yaw                     # 机器人云台 yaw (deg)    forward
float32 big_yaw                 # 机器人大yaw (deg)       forward
float32 shoot_speed             # 射速 (m/s)
int8 mode                       # 机器人模式
int8 target_num_list            # 需要攻击的敌方机器人列表
int8 target_color               # 敌方颜色
int8 balance_target_list        # 需要的敌方平衡机器人列表
