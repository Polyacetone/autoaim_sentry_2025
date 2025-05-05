#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <hw_sentry_interfaces/msg/bullet_speed.hpp>
#include <hw_sentry_interfaces/msg/shoot_pos.hpp>

#include <pnp_solver.hpp>
#include <trajectory.hpp>

namespace autoaim_anti_dart {
constexpr float rad_period_correction(const float rad) {
    return rad + round((-rad) / (2 * M_PI)) * (2 * M_PI);
}
constexpr float r2d(const float rad) {
    return rad * 180.0 / M_PI;
}
constexpr float d2r(const float deg) {
    return deg * M_PI / 180.0;
}

class AntiDartNode: public rclcpp::Node {
public:
    explicit AntiDartNode(const rclcpp::NodeOptions& options);
    ~AntiDartNode() = default;

private:
    void get_parameters();
    std::vector<cv::Point2f> detect_points(const cv::Mat& img) const;
    geometry_msgs::msg::Transform try_get_transform(
        const std::string& target,
        const std::string& source,
        const rclcpp::Time& time_point
    ) const;

    void img_callback(const sensor_msgs::msg::Image::SharedPtr msg);
    void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<PnPSolver> pnp_solver_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Subscription<hw_sentry_interfaces::msg::BulletSpeed>::SharedPtr bullet_speed_sub_;
    rclcpp::Publisher<hw_sentry_interfaces::msg::ShootPos>::SharedPtr shoot_pos_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_detected_pub_;

    bool enable_anti_dart_, enable_detected_image_;
    float target_to_light_z_;
    float bullet_speed_, shoot_compensate_pitch_, shoot_compensate_yaw_;
};

AntiDartNode::AntiDartNode(const rclcpp::NodeOptions& options): Node("autoaim_anti_dart", options) {
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    pnp_solver_ = std::make_unique<PnPSolver>();
    get_parameters();
}

void AntiDartNode::get_parameters() {
    enable_anti_dart_ = declare_parameter("enable_anti_dart", true);
    enable_detected_image_ = declare_parameter("enable_detected_image", true);
    if (!enable_anti_dart_) return;

    target_to_light_z_ = declare_parameter("target_to_light_z", 0.08);
    bullet_speed_ = declare_parameter("bullet_speed", 24.0);
    shoot_compensate_pitch_ = d2r(declare_parameter("shoot_compensate_pitch", 0.0));
    shoot_compensate_yaw_ = d2r(declare_parameter("shoot_compensate_yaw", 0.0));

    std::string img_sub_topic = declare_parameter("image_topic", "autoaim/camera/image_raw");
    img_sub_ = create_subscription<sensor_msgs::msg::Image>(
        img_sub_topic,
        rclcpp::SensorDataQoS().keep_last(1),
        [&](const sensor_msgs::msg::Image::SharedPtr msg) { img_callback(msg); }
    );
    std::string camera_info_topic = declare_parameter("camera_info_topic", "camera/color/camera_info");
    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic,
        rclcpp::SensorDataQoS().keep_last(1),
        [&](const sensor_msgs::msg::CameraInfo::SharedPtr msg) { camera_info_callback(msg); }
    );
    std::string bullet_speed_sub_topic = declare_parameter("bullet_speed_sub_topic", "serial/bullet_speed");
    bullet_speed_sub_ = create_subscription<hw_sentry_interfaces::msg::BulletSpeed>(
        bullet_speed_sub_topic,
        rclcpp::QoS(1),
        [&](const hw_sentry_interfaces::msg::BulletSpeed::SharedPtr msg) {
            if (21 <= msg->bullet_speed && msg->bullet_speed <= 25) {
                bullet_speed_ = 0.5 * msg->bullet_speed + 0.5 * bullet_speed_;
            }
        }
    );
    std::string shoot_pos_pub_topic = declare_parameter("shoot_pos_pub_topic", "serial/shoot_pos");
    shoot_pos_pub_ = create_publisher<hw_sentry_interfaces::msg::ShootPos>(
        shoot_pos_pub_topic,
        rclcpp::SensorDataQoS().keep_last(1)
    );
    std::string img_detected_topic = declare_parameter("image_detected_topic", "autoaim/dart_detected_image");
    img_detected_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        img_detected_topic,
        rclcpp::QoS(1)
    );
}

void AntiDartNode::img_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    const auto cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    const std::vector<cv::Point2f> detected_pts = detect_points(cv_ptr->image);
    if (enable_detected_image_) {
        for (int i = 0; i < detected_pts.size(); i++) {
            cv::drawMarker(cv_ptr->image, detected_pts[i], cv::Scalar(0, 0, 255), 0, 4, 2);
        }
        sensor_msgs::msg::Image::SharedPtr img_detected =
            cv_bridge::CvImage(msg->header, "bgr8", cv_ptr->image).toImageMsg();
        img_detected_pub_->publish(*img_detected);
    }
    if (detected_pts.empty()) return;

    const cv::Point3f translation = pnp_solver_->get_translation(detected_pts);
    if (translation.x < 0) return;

    geometry_msgs::msg::TransformStamped dart_target_to_cam;
    dart_target_to_cam.header.stamp = msg->header.stamp;
    dart_target_to_cam.header.frame_id = "autoaim_camera";
    dart_target_to_cam.child_frame_id = "dart_target";
    dart_target_to_cam.transform.translation.x = translation.x;
    dart_target_to_cam.transform.translation.y = translation.y;
    dart_target_to_cam.transform.translation.z = translation.z + target_to_light_z_;
    tf_broadcaster_->sendTransform(dart_target_to_cam);

    geometry_msgs::msg::Transform dart_target_to_fric;
    try {
        dart_target_to_fric = try_get_transform("fake_fric", "dart_target", msg->header.stamp);
    } catch (const std::exception& ex) {
        RCLCPP_WARN(
            get_logger(),
            "Failed to get transform from target to fake_fric: %s",
            ex.what()
        );
        return;
    }

    const float target_pitch = trajectory::calc_pitch(
        dart_target_to_fric.translation.x,
        dart_target_to_fric.translation.y,
        dart_target_to_fric.translation.z,
        bullet_speed_
    ) - shoot_compensate_pitch_;
    const float target_yaw = rad_period_correction(
        atan2(
            dart_target_to_fric.translation.y,
            dart_target_to_fric.translation.x
        )
    ) + shoot_compensate_yaw_;

    hw_sentry_interfaces::msg::ShootPos shoot_pos;
    shoot_pos.header.stamp = msg->header.stamp;
    shoot_pos.shoot_flag = 0;
    shoot_pos.pitch = target_pitch;
    shoot_pos.yaw = target_yaw;
    shoot_pos_pub_->publish(shoot_pos);
}

std::vector<cv::Point2f> AntiDartNode::detect_points(const cv::Mat& img) const {
    using namespace cv;
    using namespace std;

    Mat hsv, green_mask;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    inRange(hsv, Scalar(40, 120, 60), Scalar(80, 255, 200), green_mask);
    morphologyEx(green_mask, green_mask, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(15, 15)));

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(green_mask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < contours.size(); i++) {
        // 是一个轮廓的子轮廓，但没有兄弟轮廓和和自己的子轮廓。为了识别绿色圆圈内的过曝部分
        if (hierarchy[i][0] == -1 && hierarchy[i][1] == -1 && hierarchy[i][2] == -1 && hierarchy[i][3] != -1) {
            float area = contourArea(contours[i]);
            if (area < 10) continue; // 面积要大于一定值

            float perimeter = arcLength(contours[i], true);
            float circularity = (4 * CV_PI * area) / (perimeter * perimeter);
            if (circularity < 0.7) continue; // 圆度要大于一定值

            RotatedRect ellipse = fitEllipse(contours[i]);
            Point2f center = ellipse.center;
            cv::Vec3b pixel = img.at<cv::Vec3b>(center.y, center.x);
            if (pixel[0] + pixel[1] + pixel[2] < 220 * 3) continue; // 中心点要很亮（过曝）

            const float radius = (ellipse.size.height + ellipse.size.width) / 4;
            if (center.x <= radius || center.y <= radius) continue;
            if (center.x >= img.cols - radius || center.y >= img.rows - radius) continue;

            Mat roi = img(Rect(center.x - radius, center.y - radius, radius * 2, radius * 2));
            resize(roi, roi, Size(), 5, 5);
            // 取roi放大后提取过曝的白色区域，可以让角点更准确
            Mat over_exposure;
            inRange(roi, Scalar(200, 200, 200), Scalar(255, 255, 255), over_exposure);
            morphologyEx(over_exposure, over_exposure, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
            vector<vector<Point>> over_exposure_contour;
            findContours(over_exposure, over_exposure_contour, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            if (over_exposure_contour.size() == 1) {
                RotatedRect ellipse_precise = fitEllipse(over_exposure_contour[0]);
                const float radius_precise = (ellipse_precise.size.height + ellipse_precise.size.width) / 4;
                const Point2f center_precise =
                    center - Point2f(radius, radius) + ellipse_precise.center / 5;

                Point2f u(center_precise.x, center_precise.y - radius_precise / 5);
                Point2f d(center_precise.x, center_precise.y + radius_precise / 5);
                Point2f l(center_precise.x - radius_precise / 5, center_precise.y);
                Point2f r(center_precise.x + radius_precise / 5, center_precise.y);
                return vector<Point2f>{u, l, d, r};
            }
        }
    }
    return vector<Point2f>();
}

geometry_msgs::msg::Transform AntiDartNode::try_get_transform(
    const std::string& target,
    const std::string& source,
    const rclcpp::Time& time_point
) const {
    constexpr int MAX_ATTEMPTS = 100;
    geometry_msgs::msg::Transform transform;
    for (int i = 0; i < MAX_ATTEMPTS; i++) {
        try {
            transform = tf_buffer_->lookupTransform(target, source, time_point).transform;
            return transform;
        } catch (const std::exception& ex) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    }
    throw std::runtime_error("try_get_transform failed after 100 attempts");
}

void AntiDartNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    pnp_solver_->set_cam_matrix(
        cv::Mat(3, 3, CV_64F, msg->k.data()),
        cv::Mat(1, 5, CV_64F, msg->d.data())
    );
    // 相机内参和畸变在运行中不会改变，所以设置后即可取消camera_info订阅
    camera_info_sub_.reset();
    camera_info_sub_ = nullptr;
}
} // namespace autoaim_anti_dart

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoaim_anti_dart::AntiDartNode)