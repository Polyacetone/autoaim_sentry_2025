# 作为DetectionArray的成员类型，从autoaim_detection发送到autoaim_prediction。

int32 color # blue, red, gray
int32 label # S, 1, 2, 3, 4, outpost, basesmall, basebig
float32 confidence

geometry_msgs/Point32 tl
geometry_msgs/Point32 tr
geometry_msgs/Point32 bl
geometry_msgs/Point32 br