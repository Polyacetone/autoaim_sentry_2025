std_msgs/Header header
bool found                          # 是否找到目标
float32 pitch                       # 预测的云台 pitch (deg)   up
float32 yaw                         # 预测的云台 yaw (deg)    forward
uint8 shoot_flag                    # 射击标志
uint8 target_num                    # 目标机器人编号
uint8[] detect_target_num_list        # 识别到的的敌方机器人列表
geometry_msgs/Vector3[] detect_target_pos_list  # 识别到的敌方机器人位置列表
uint8 fly_t_to_send # 打符时间间隔 (ms)
uint8 active_buff_num 
