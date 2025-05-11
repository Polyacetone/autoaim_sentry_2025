#include <deque>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <message_filters/cache.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <hw_sentry_interfaces/msg/debug_info.hpp>
#include <hw_sentry_interfaces/msg/shoot_pos.hpp>

namespace autoaim_recorder {
using hw_sentry_interfaces::msg::DebugInfo;
using hw_sentry_interfaces::msg::ShootPos;

double to_sec(builtin_interfaces::msg::Time t) {
    return t.sec + t.nanosec * 1e-9;
}
float r2d(const float rad) {
    return rad * 180.0 / M_PI;
}
float d2r(const float deg) {
    return deg * M_PI / 180.0;
}

class RecorderNode: public rclcpp::Node {
public:
    explicit RecorderNode(const rclcpp::NodeOptions& options);
    ~RecorderNode() = default;

private:
    void get_parameters();
    void img_raw_callback(const sensor_msgs::msg::Image::SharedPtr msg);
    void img_detected_callback(const sensor_msgs::msg::Image::SharedPtr msg);
    void draw_info_on_img(const DebugInfo::SharedPtr msg, cv::Mat& img) const;
    void draw_info_on_img(const ShootPos::SharedPtr msg, cv::Mat& img) const;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_raw_sub_, img_detected_sub_;
    rclcpp::Subscription<DebugInfo>::SharedPtr debug_info_sub_;
    rclcpp::Subscription<ShootPos>::SharedPtr shoot_pos_sub_;
    std::deque<DebugInfo::SharedPtr> debug_info_que_;
    std::deque<ShootPos::SharedPtr> shoot_pos_que_;
    std::mutex debug_info_que_mtx_, shoot_pos_que_mtx_;

    cv::VideoWriter video_writer_raw_, video_writer_verbose_;
    std::mutex video_writer_raw_mtx_, video_writer_verbose_mtx_;
    double start_time_;

    float video_fps_;
    bool record_raw_, record_verbose_;
    std::string video_save_directory_;
};

RecorderNode::RecorderNode(const rclcpp::NodeOptions& options): Node("autoaim_recorder", options) {
    get_parameters();
    start_time_ = to_sec(now());

    auto now = std::chrono::system_clock::now();
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm = *std::localtime(&now_time_t);
    std::stringstream timestr;
    timestr << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S");

    if (record_raw_) {
        video_writer_raw_.open(
            video_save_directory_ + timestr.str() + " raw.mp4",
            cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
            video_fps_,
            cv::Size(640, 384)
        );
        if (!video_writer_raw_.isOpened()) {
            RCLCPP_ERROR(get_logger(), "Failed to open video writer!");
        }
    }
    if (record_verbose_) {
        video_writer_verbose_.open(
            video_save_directory_ + timestr.str() + " verbose.mp4",
            cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
            video_fps_,
            cv::Size(640, 384)
        );
        if (!video_writer_verbose_.isOpened()) {
            RCLCPP_ERROR(get_logger(), "Failed to open video writer!");
        }
    }
    if (!record_raw_ && !record_verbose_) {
        RCLCPP_INFO(get_logger(), "Recorder node has nothing to do.");
    }
}

void RecorderNode::get_parameters() {
    video_fps_ = declare_parameter("video_fps", 100.0);
    video_save_directory_ = declare_parameter("video_save_directory", "");
    record_raw_ = declare_parameter("record_raw", true);
    record_verbose_ = declare_parameter("record_verbose", true);
    if (record_raw_) {
        std::string img_raw_topic =
            declare_parameter("img_raw_topic", "autoaim/camera/image_raw");
        img_raw_sub_ = create_subscription<sensor_msgs::msg::Image>(
            img_raw_topic,
            rclcpp::SensorDataQoS().keep_last(1),
            [&](const sensor_msgs::msg::Image::SharedPtr msg) {
                std::thread(&RecorderNode::img_raw_callback, this, msg).detach();
            }
        );
    }
    if (record_verbose_) {
        std::string img_detected_topic =
            declare_parameter("image_detected_topic", "autoaim/detected_image");
        img_detected_sub_ = create_subscription<sensor_msgs::msg::Image>(
            img_detected_topic,
            rclcpp::QoS(1),
            [&](const sensor_msgs::msg::Image::SharedPtr msg) {
                std::thread(&RecorderNode::img_detected_callback, this, msg).detach();
            }
        );
    }

    std::string debug_info_topic = declare_parameter("debug_info_topic", "autoaim/debug_info");
    debug_info_sub_ = create_subscription<DebugInfo>(
        debug_info_topic,
        rclcpp::QoS(1),
        [&](const DebugInfo::SharedPtr msg) {
            debug_info_que_mtx_.lock();
            debug_info_que_.emplace_front(msg);
            if (debug_info_que_.size() >= 5) {
                debug_info_que_.pop_back();
            }
            debug_info_que_mtx_.unlock();
        }
    );
    std::string shoot_pos_topic = declare_parameter("shoot_pos_topic", "autoaim/shoot_pos");
    shoot_pos_sub_ = create_subscription<ShootPos>(
        shoot_pos_topic,
        rclcpp::SensorDataQoS().keep_last(1),
        [&](const ShootPos::SharedPtr msg) {
            shoot_pos_que_mtx_.lock();
            shoot_pos_que_.emplace_front(msg);
            if (shoot_pos_que_.size() >= 5) {
                shoot_pos_que_.pop_back();
            }
            shoot_pos_que_mtx_.unlock();
        }
    );
}

void RecorderNode::img_raw_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    const auto cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    if (cv_ptr->image.empty()) {
        RCLCPP_WARN(get_logger(), "img_detected_callback() get an empty frame, ignoring");
        return;
    }
    cv::Mat image = cv_ptr->image.clone();

    video_writer_raw_mtx_.lock();
    video_writer_raw_ << image;
    video_writer_raw_mtx_.unlock();
}

void RecorderNode::img_detected_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    const auto cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    if (cv_ptr->image.empty()) {
        RCLCPP_WARN(get_logger(), "img_detected_callback() get an empty frame, ignoring");
        return;
    }
    cv::Mat image = cv_ptr->image.clone();

    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    DebugInfo::SharedPtr debug_info_msg = nullptr;
    debug_info_que_mtx_.lock();
    for (auto it = debug_info_que_.begin(); it != debug_info_que_.end(); it++) {
        if (abs(to_sec((*it)->header.stamp) - to_sec(msg->header.stamp)) < 1e-3) {
            debug_info_msg = *it;
            debug_info_que_.erase(it);
            break;
        }
    }
    debug_info_que_mtx_.unlock();

    ShootPos::SharedPtr shoot_pos_msg = nullptr;
    shoot_pos_que_mtx_.lock();
    for (auto it = shoot_pos_que_.begin(); it != shoot_pos_que_.end(); it++) {
        if (abs(to_sec((*it)->header.stamp) - to_sec(msg->header.stamp)) < 1e-3) {
            shoot_pos_msg = *it;
            shoot_pos_que_.erase(it);
            break;
        }
    }
    shoot_pos_que_mtx_.unlock();

    if (debug_info_msg) {
        draw_info_on_img(debug_info_msg, image);
    }
    if (shoot_pos_msg) {
        draw_info_on_img(shoot_pos_msg, image);
    }

    video_writer_verbose_mtx_.lock();
    video_writer_verbose_ << image;
    video_writer_verbose_mtx_.unlock();
}

void RecorderNode::draw_info_on_img(const DebugInfo::SharedPtr msg, cv::Mat& img) const {
    using namespace std;
    using namespace cv;
    const string COLOR_MAP[3] = {"B", "R", "G"};
    const string LABEL_MAP[6] = {"S", "1", "2", "3", "4", "O"};
    const string STATUS_MAP[4] = {"CONVERGING", "TRACKING", "TEMP_LOST", "LOST"};
    const Scalar WHITE(255, 255, 255);
    const Scalar BLUE(255, 0, 0);
    const Scalar GREEN(0, 255, 0);
    const Scalar RED(0, 0, 255);
    const auto FONT = FONT_HERSHEY_DUPLEX;
    const auto text_left_align = [](const Mat& img, const string& s, const Point& p, const Scalar& c) {
        putText(img, s, Point(p.x, p.y), FONT, 0.5, c, 0.5);
    };
    const auto text_middle_align = [](const Mat& img, const string& s, const Point& p, const Scalar& c) {
        putText(img, s, Point(p.x - s.size() * 5, p.y), FONT, 0.5, c, 0.5);
    };
    char buf[128];

    // 左上角画时间
    std::sprintf(buf, "T: %.3lf", to_sec(msg->header.stamp) - start_time_);
    text_left_align(img, buf, Point(0, 10), WHITE);

    // 右上角画目标装甲板颜色和编号
    std::sprintf(buf, "%s%2d", COLOR_MAP[msg->target_color].c_str(), msg->target_armor);
    text_left_align(img, buf, Point(500, 10), WHITE);

    // 右上角画tracker_status
    const auto status_color = (msg->tracker_status == 0 ? WHITE : (msg->tracker_status == 1 ? GREEN : RED));
    text_middle_align(img, STATUS_MAP[msg->tracker_status], Point(590, 10), status_color);

    // 上面中间画可击打状态
    for (int i = 0; i < 6; i++) {
        const auto color = msg->is_enemy_can_shoot[i] ? GREEN : RED;
        text_left_align(img, LABEL_MAP[i], Point(260 + i * 20, 10), color);
    }

    // 左下角画跟踪器具体信息
    if (msg->tracker_status != 3) {
        if (msg->tracker_status == 0) { // converging
            std::sprintf(buf, "converging_cnt: %5d", msg->current_status_frames);
            text_left_align(img, buf, Point(0, 320), WHITE);
        } else if (msg->tracker_status == 1) { // tracking
            std::sprintf(buf, "tracking_cnt: %5d", msg->current_status_frames);
            text_left_align(img, buf, Point(0, 320), GREEN);
        } else if (msg->tracker_status == 1) { // temp_lost
            std::sprintf(buf, "temp_lost_cnt: %5d", msg->current_status_frames);
            text_left_align(img, buf, Point(0, 320), RED);
        }
    
        std::sprintf(
            buf,
            "KF: (%3.0f, %3.0f, %3.0f) += (%3.0f, %3.0f, %3.0f)",
            msg->kf_xyz_position.x * 100,
            msg->kf_xyz_position.y * 100,
            msg->kf_xyz_position.z * 100,
            msg->kf_xyz_velocity.x * 100,
            msg->kf_xyz_velocity.y * 100,
            msg->kf_xyz_velocity.z * 100
        );
        text_left_align(img, buf, Point(0, 335), WHITE);
    
        std::sprintf(
            buf,
            "UKF: (%4.0f, %4.0f) += (%4.0f, %4.0f)",
            msg->ukf_xy_position.x * 100,
            msg->ukf_xy_position.y * 100,
            msg->ukf_xy_velocity.x * 100,
            msg->ukf_xy_velocity.y * 100
        );
        text_left_align(img, buf, Point(0, 350), WHITE);
    
        std::sprintf(
            buf,
            "YAW: (%6.0f) += (%4.0f)",
            r2d(msg->kf_yaw),
            r2d(msg->kf_yaw_palstance)
        );
        text_left_align(img, buf, Point(0, 365), WHITE);
    
        std::sprintf(
            buf,
            "R: [%4.0f, %4.0f], H: [%4.0f, %4.0f, %4.0f, %4.0f]",
            msg->radius[0] * 100,
            msg->radius[1] * 100,
            msg->height[0] * 100,
            msg->height[1] * 100,
            msg->height[2] * 100,
            msg->height[3] * 100
        );
        text_left_align(img, buf, Point(0, 380), WHITE);
    }
}

void RecorderNode::draw_info_on_img(const ShootPos::SharedPtr msg, cv::Mat& img) const {
    using namespace std;
    using namespace cv;
    const Scalar WHITE(255, 255, 255);
    const auto FONT = FONT_HERSHEY_DUPLEX;
    const auto text_left_align = [](const Mat& img, const string& s, const Point& p, const Scalar& c) {
        putText(img, s, Point(p.x, p.y), FONT, 0.5, c, 0.5);
    };
    char buf[64];

    std::sprintf(buf, "Pitch: %4.1f", r2d(msg->pitch));
    text_left_align(img, buf, Point(530, 350), WHITE);
    std::sprintf(buf, "Yaw: %4.1f", r2d(msg->yaw));
    text_left_align(img, buf, Point(530, 365), WHITE);
    std::sprintf(buf, "Flag: %s", msg->shoot_flag ? "true" : "false");
    text_left_align(img, buf, Point(530, 380), WHITE);
}
} // namespace autoaim_recorder

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_recorder::RecorderNode)