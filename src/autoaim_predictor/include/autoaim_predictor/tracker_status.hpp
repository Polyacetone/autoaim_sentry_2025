#pragma once

#include <opencv2/opencv.hpp>
#include <termcolor/termcolor.hpp>

enum class StatusType : int { CONVERGING, TRACKING, TEMP_LOST, LOST };

class TrackerStatus {
public:
    explicit TrackerStatus(
        const cv::FileNode& fn,
        std::function<void(StatusType from, StatusType to)> status_change_handler =
            [](StatusType from, StatusType to) { (void)from, (void)to; },
        std::function<void(StatusType current)> status_remain_handler =
            [](StatusType current) { (void)current; }
    );
    explicit TrackerStatus(
        const unsigned max_temp_lost_frames,
        const unsigned max_converging_frames,
        std::function<void(StatusType from, StatusType to)> status_change_handler =
            [](StatusType from, StatusType to) { (void)from, (void)to; },
        std::function<void(StatusType current)> status_remain_handler =
            [](StatusType current) { (void)current; }
    );

    void reset();
    void update(bool is_valid);
    StatusType status() const;
    void print_colored_status_info() const;

private:
    void set_next_status(StatusType status);

    // 两个handler会且仅会被调用其中一个
    const std::function<void(StatusType from, StatusType to)>
        status_change_handler; // 状态转移时调用
    const std::function<void(StatusType current)> status_remain_handler; // 状态持续时调用
    unsigned MAX_TEMP_LOST_FRAMES, MAX_CONVERGING_FRAMES;

    StatusType status_ = StatusType::LOST;
    unsigned current_status_frames_ = 0;
};