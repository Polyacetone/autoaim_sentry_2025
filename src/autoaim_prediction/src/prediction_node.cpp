#include <string>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <autoaim_interfaces/msg/detection_array.hpp>

#include <pnp_solver.hpp>
#include <trisection_yaw.hpp>
#include <tracker.hpp>
#include <trajectory.hpp>

using autoaim_interfaces::msg::Detection;
using autoaim_interfaces::msg::DetectionArray;

double to_sec(builtin_interfaces::msg::Time t) {
    return t.sec + t.nanosec * 1e-9;
}

std::string get_tf_armor_name(int color, int label, int index) {
    std::string name;
    char color_map[3] = {'G', 'B', 'R'}; // gray, blue, red
    name += color_map[color];
    name += static_cast<char>(label + '0');
    name += '-';
    name += static_cast<char>(index + '0');
    return name;
}

namespace autoaim_prediction {
class PredictionNode: public rclcpp::Node {
public:
    explicit PredictionNode(const rclcpp::NodeOptions& options);
    ~PredictionNode() = default;

private:
    void get_parameters();
    void detection_callback(const DetectionArray::SharedPtr msg) const;
    void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    geometry_msgs::msg::Transform
    get_lastest_transform(const std::string& target, const std::string& source) const;
    void select_armors(const std::vector<Detection> src, std::vector<Detection>& dst) const;

    bool enable_debug_;
    bool debug_mode_;
    int debug_target_color_;
    int debug_target_armor_;
    int debug_buff_mode_;
    float debug_prediction_time_;
    float debug_bullet_speed_;

    float armor_dir_angle_;
    float filter_distance_;
    float imu_compensate_pitch_;
    float imu_compensate_yaw_;
    float t_delay_;
    std::string camera_info_topic_;
    std::string detection_sub_topic_;
    std::string comm_pub_topic_;
    std::string position_pub_topic_;

    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_static_broadcaster_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    std::shared_ptr<rclcpp::Subscription<DetectionArray>> detection_sub_;
    std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>> camera_info_sub_;

    std::shared_ptr<PnPSolver> pnp_solver_;
    std::shared_ptr<TrisectionYaw> trisection_yaw_;
    std::shared_ptr<Tracker> tracker_;
};

PredictionNode::PredictionNode(const rclcpp::NodeOptions& options):
    Node("autoaim_prediction", options) {
    tf_static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    pnp_solver_ = std::make_shared<PnPSolver>();
    trisection_yaw_ = std::make_shared<TrisectionYaw>();
    std::string kf_params_path = ament_index_cpp::get_package_share_directory("autoaim_prediction")
        + "/config/kf_params.yaml";
    tracker_ = std::make_shared<Tracker>(kf_params_path);

    get_parameters();

    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic_,
        10,
        [&](const sensor_msgs::msg::CameraInfo::SharedPtr msg) { camera_info_callback(msg); }
    );
    detection_sub_ = create_subscription<DetectionArray>(
        detection_sub_topic_,
        10,
        [&](const DetectionArray::SharedPtr msg) { detection_callback(msg); }
    );
}

void PredictionNode::get_parameters() {
    enable_debug_ = declare_parameter("enable_debug", false);
    debug_mode_ = declare_parameter("debug_mode", false);
    debug_prediction_time_ = declare_parameter("debug_prediction_time", 0.0);
    debug_target_color_ = declare_parameter("debug_target_color", 0);
    debug_target_armor_ = declare_parameter("debug_target_armor", 1);
    debug_buff_mode_ = declare_parameter("debug_buff_mode", 1);
    debug_bullet_speed_ = declare_parameter("debug_bullet_speed", 30.0);
    armor_dir_angle_ = declare_parameter("armor_dir_angle", 0.2618);
    filter_distance_ = declare_parameter("filter_distance", 6.0);
    imu_compensate_pitch_ = declare_parameter("imu_compensate_pitch", 1.3);
    imu_compensate_yaw_ = declare_parameter("imu_compensate_yaw", 0.3);
    t_delay_ = declare_parameter("t_delay", 0.098);
    camera_info_topic_ = declare_parameter("camera_info_topic", "/camera/color/camera_info");
    detection_sub_topic_ = declare_parameter("detection_sub_topic", "/detection");
    comm_pub_topic_ = declare_parameter("comm_pub_topic", "/serial/comm_send");
    position_pub_topic_ = declare_parameter("position_pub_topic", "/debug/position");

    const double cam_to_spindle_x = declare_parameter("cam_to_spindle_x", 0.0);
    const double cam_to_spindle_y = declare_parameter("cam_to_spindle_y", 0.0658);
    const double cam_to_spindle_z = declare_parameter("cam_to_spindle_z", 0.0639);
    geometry_msgs::msg::TransformStamped cam_to_spindle;
    cam_to_spindle.header.stamp = this->now();
    cam_to_spindle.header.frame_id = "spindle";
    cam_to_spindle.child_frame_id = "autoaim_camera";
    cam_to_spindle.transform.translation.x = cam_to_spindle_x;
    cam_to_spindle.transform.translation.y = cam_to_spindle_y;
    cam_to_spindle.transform.translation.z = cam_to_spindle_z;
    // 这里认为相机坐标系向右是x，向下是y，向前是z。世界坐标系向右是x，向前是y，向上是z。
    cam_to_spindle.transform.rotation.x = -0.7071068;
    cam_to_spindle.transform.rotation.y = 0;
    cam_to_spindle.transform.rotation.z = 0;
    cam_to_spindle.transform.rotation.w = 0.7071068;
    tf_static_broadcaster_->sendTransform(cam_to_spindle);

    const double fric_to_spindle_x = declare_parameter("fric_to_spindle_x", 0.0);
    const double fric_to_spindle_y = declare_parameter("fric_to_spindle_y", 0.06191);
    const double fric_to_spindle_z = declare_parameter("fric_to_spindle_z", 0.0);
    geometry_msgs::msg::TransformStamped fric_to_spindle;
    fric_to_spindle.header.stamp = this->now();
    fric_to_spindle.header.frame_id = "spindle";
    fric_to_spindle.child_frame_id = "fric_wheel";
    fric_to_spindle.transform.translation.x = fric_to_spindle_x;
    fric_to_spindle.transform.translation.y = fric_to_spindle_y;
    fric_to_spindle.transform.translation.z = fric_to_spindle_z;
    fric_to_spindle.transform.rotation.x = 0;
    fric_to_spindle.transform.rotation.y = 0;
    fric_to_spindle.transform.rotation.z = 0;
    fric_to_spindle.transform.rotation.w = 1;
    tf_static_broadcaster_->sendTransform(fric_to_spindle);
}

void PredictionNode::detection_callback(const DetectionArray::SharedPtr msg) const {
    if (enable_debug_ && debug_mode_ == false) {
        std::vector<Detection> target_armors;
        select_armors(msg->detections, target_armors);
        const int len = target_armors.size();
        for (int i = 0; i < len; i++) {
            const Detection& armor = target_armors[i];
            const std::string armor_name = get_tf_armor_name(armor.color, armor.label, i);
            geometry_msgs::msg::TransformStamped armor_to_cam;
            armor_to_cam.header.stamp = this->now();
            armor_to_cam.header.frame_id = "autoaim_camera";
            armor_to_cam.child_frame_id = armor_name;

            pnp_solver_->solve_pnp(armor, armor_to_cam.transform);
            trisection_yaw_->get_rotation(armor, armor_to_cam.transform);
            tf_broadcaster_->sendTransform(armor_to_cam);
            
            auto armor_to_spindle = get_lastest_transform("spindle", armor_name);
            tracker_->push(armor_to_spindle);
        }
        tracker_->update();
        if (tracker_->tracker_status == TRACKER_STATUS::TRACKING) {
            const cv::Point3f predicted = tracker_->get_prediction(debug_bullet_speed_, t_delay_);
            geometry_msgs::msg::TransformStamped predicted_to_cam;
            predicted_to_cam.header.stamp = this->now();
            predicted_to_cam.header.frame_id = "spindle";
            predicted_to_cam.child_frame_id = "target";
            predicted_to_cam.transform.translation.x = predicted.x;
            predicted_to_cam.transform.translation.y = predicted.y;
            predicted_to_cam.transform.translation.z = predicted.z;
            tf_broadcaster_->sendTransform(predicted_to_cam);

            auto predicted_to_fric = get_lastest_transform("fric_wheel", "target");
            float predicted_pitch, predicted_yaw;
            std::tie(predicted_pitch, predicted_yaw) = trajectory::get_pitch_yaw(
                predicted_to_fric.translation.x,
                predicted_to_fric.translation.y,
                predicted_to_fric.translation.z,
                debug_bullet_speed_
            );

            RCLCPP_INFO(
                this->get_logger(),
                "Pitch: %4.1f  Yaw: %4.1f (degree)",
                math::r2d(predicted_pitch),
                math::r2d(predicted_yaw)
            );
        }
    }
}

void PredictionNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    pnp_solver_->set_cam_matrix(
        cv::Mat(3, 3, CV_64F, msg->k.data()),
        cv::Mat(1, 5, CV_64F, msg->d.data())
    );
    trisection_yaw_->set_cam_matrix(
        cv::Mat(3, 3, CV_64F, msg->k.data()),
        cv::Mat(1, 5, CV_64F, msg->d.data())
    );
    // 相机内参和畸变在运行中不会改变，所以设置后即可取消camera_info订阅
    camera_info_sub_.reset();
    camera_info_sub_ = nullptr;
}

void PredictionNode::select_armors(const std::vector<Detection> src, std::vector<Detection>& dst)
    const {
    std::vector<Detection> filtered;
    // 筛选出目标颜色和标签的装甲板
    for (const auto& armor: src) {
        if (enable_debug_) {
            if ((debug_target_color_ == 0 || armor.color == debug_target_color_)
                && armor.label == debug_target_armor_)
            {
                filtered.emplace_back(armor);
            }
        } else {
            // TODO
        }
    }
    if (filtered.size() >= 2) {
        // 先按照击打面积排序
        std::sort(filtered.begin(), filtered.end(), [](const Detection& a, const Detection& b) {
            return (a.br.x - a.tl.x) * (a.br.y - a.tl.y) > (b.br.x - b.tl.x) * (b.br.y - b.tl.y);
        });
        dst.emplace_back(filtered[0]);
        // 接下来选择击打面积次之，且和原来那个位置有较大差异的装甲板。
        // 虽然理论上detection中的nms已经能去除同一个装甲板的多个识别结果，但有时候还是会出现。
        const int len = filtered.size();
        const int center_of_first =
            (filtered[0].bl.x + filtered[0].br.x + filtered[0].tr.x + filtered[0].tl.x) / 4;
        for (int i = 1; i < len; i++) {
            const int center_of_i =
                (filtered[i].bl.x + filtered[i].br.x + filtered[i].tr.x + filtered[i].tl.x) / 4;
            if (abs(center_of_first - center_of_i) >= 50) {
                dst.push_back(filtered[i]);
                break;
            }
        }
    } else if (filtered.size() == 1) {
        dst.push_back(filtered[0]);
    }
}

geometry_msgs::msg::Transform
PredictionNode::get_lastest_transform(const std::string& target, const std::string& source) const {
    // 防止buffer没来得及更新使得找不到frame
    while (!tf_buffer_->canTransform(target, source, tf2::TimePointZero)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // 对每一次获取的时间戳进行记录，防止buffer没来得及更新使得找到旧的变换
    static std::unordered_map<std::string, double> timestamp_map;
    geometry_msgs::msg::TransformStamped tf_stamped;
    do {
        tf_stamped = tf_buffer_->lookupTransform(target, source, tf2::TimePointZero);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (timestamp_map[source] == to_sec(tf_stamped.header.stamp));
    timestamp_map[source] = to_sec(tf_stamped.header.stamp);
    return tf_stamped.transform;
}
} // namespace autoaim_prediction

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_prediction::PredictionNode)