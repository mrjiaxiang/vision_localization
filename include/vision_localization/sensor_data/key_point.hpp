#pragma once

#include <opencv2/opencv.hpp>

#include <Eigen/Core>

namespace vision_localization {
class Keypoint {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int lmid_;

    cv::Point2f px_;
    cv::Point2f unpx_;
    Eigen::Vector3d bv_;

    int scale_;
    float angle_;
    cv::Mat desc_;

    bool is3d_;

    bool is_stereo_;
    cv::Point2f rpx_;
    cv::Point2f runpx_;
    Eigen::Vector3d rbv_;

    bool is_retracked_;

    Keypoint()
        : lmid_(-1), scale_(0), angle_(-1.), is3d_(false), is_stereo_(false),
          is_retracked_(false) {}

    // For using kps in ordered containers
    bool operator<(const Keypoint &kp) const { return lmid_ < kp.lmid_; }
};
} // namespace vision_localization
