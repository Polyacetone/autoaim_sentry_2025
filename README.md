# autoaim_sentry_2025

本仓库存储2025赛季Hello World战队哨兵自瞄代码，代码以及本文档的作者均为@Polyacetone。

## 节点简介和相互关系

以下为所有节点的简介以及它们之间的话题通信关系。其中camera, detector, selector, locator, predictor是自瞄的关键节点。

### autoaim_camera

相机节点。负责打开海康相机、设置相机参数、取图，然后通过图像话题发布给autoaim_detector进行推理。

### autoaim_detector

推理节点。接收相机节点发送的图像，按照相应模式（装甲板或者符）进行推理，并把识别结果发送出去。

### autoaim_selector

推理结果选择节点。接收推理节点的识别结果，进行过滤后发送到后面。对于装甲板模式，需要筛选敌人颜色和目标编号。对于打符模式，需要筛选符的颜色和未激活类型。

### autoaim_locator

对选择后的推理结果进行PNP解算，获取对应的位姿，发送给后面的预测节点。

### autoaim_predictor

接收装甲板或者符的位姿，根据类型和模式维护相应跟踪器，发送最终云台的目标位置和射击标志。

---

### autoaim_recorder

内录节点。接收相机节点发送的图像，录制原始视频；接收推理节点发送的带标签图像和预测节点的跟踪器信息，录制画上详细信息的视频。

### autoaim_send_enemy

给决策发送视野中出现的所有敌人，用于决定合适的击打目标。直接接收推理节点的识别结果，筛选出敌方颜色的车，维护一个简单的跟踪器，然后发送给决策。这个节点功能和以上自瞄节点有部分重合，但独立出来是为了可扩展性，比如步兵等有操作手的兵种不需要自主决策，就直接删除这个包即可。

### autoaim_common_libs

包含自瞄需要的一些通用定义（装甲板类型、颜色类型等等）和小工具（类型转换工具、数学工具等）。目前为header-only包，不参与编译。

### autoaim_launcher

一个统一的启动节点。通过打开一个component_container_mt，将所有需要打开的节点（camera, detector, selector, locator, predictor, send_enemy, recorder）加载进去。使用可组合节点（composable node）是因为可以使用进程间通信（intra process comms），避免大负载（如图像）消息拥堵。值得注意的是这个launcher可以给所有话题加上namespace。

## 构建指南

本项目依赖ros-jazzy, opencv, Eigen3, openvino。除此之外，编译时还需要包含hw_sentry_interfaces，其中有一些自定义的ros2消息类型。

确保你拥有了上述环境之后，在项目根目录执行以下命令即可。

```bash
colcon build --symlink-install --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release
```

打开`CMAKE_EXPORT_COMPILE_COMMANDS`是为了配合clangd插件实现语法分析。

## 运行时依赖

autoaim_camera需要海康相机驱动来连接海康相机。autoaim_detector需要Intel的GPU和NPU来运行推理部分。autoaim_locator和autoaim_predictor等需要一个TF树来查询相机相对于世界系的位姿（这个TF树主要由导航部分维护，所以如果你需要完整功能的话还需要开启导航部分。不过调试的时候可以有替代办法，即开启hw_sentry_robot_descriptions和hw_sentry_serial_driver，让hw_sentry_serial_driver发送假的IMU信息）。

## 注意事项

本仓库是2025赛季的存档，理论上你不应该往这里提交，文档和注释错误除外。

在编写新赛季自瞄代码的时候（实际上编写任何代码时都应该这样）请务必注意**遵守已有的代码规范**（如果你看已有的代码不爽，其实也可以自己写一个新的）。例如：尽可能地使用新的C++特性，这通常会带来性能或者可读性的优化（一个例子是autoaim_common_libs里的convert_utils.hpp，利用了C++20的concept实现模板约束，提升可读性和可维护性）；对于一些可以复用的东西应适当抽象，以提升可读性。