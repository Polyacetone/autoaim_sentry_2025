float64 theta                       # 装甲板朝向角 (degree)
float64 angle_speed                 # 自转角速度   (degree/ms)
float64 x                           # 平动: 装甲板中心x坐标 陀螺: 车中心x坐标 (mm/ms)
float64 y                           # 平动: 装甲板中心y坐标 陀螺: 车中心y坐标 (mm/ms)
float64 z                           # 平动: 装甲板中心z坐标 陀螺: 装甲板中心z坐标 (mm/ms)
float64 vy                          # 平动: 装甲板y轴速度 陀螺: 车y轴速度 (mm/ms)
float64 vz                          # 平动: 装甲板z轴速度 陀螺: 
float64 vx                          # 平动: 装甲板x轴速度 陀螺: 车x轴速度 (mm/ms)
float64 ax
float64 ay
int8 armor_count
float64 final_armor_position_x      # 预测的装甲板中心x坐标 (mm)
float64 final_armor_position_y      # 预测的装甲板中心y坐标 (mm)
float64 final_armor_position_z      # 预测的装甲板中心z坐标 (mm)
float64 pred_pitch                  # 预测的pitch角度 (degree)
float64 pred_yaw                    # 预测的yaw角度 (degree)
float64[] radius

# Buff
float64 this_buff_angle
int8 direction
float64 final_angle
int32 recognized_counts

# Big Buff
float64 a
float64 w 
float64 buff_theta
float64 b
