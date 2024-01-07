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

class MonoFeatureMeasurement {
  public:
    MonoFeatureMeasurement()
        : id(0), u(0.0), v(0.0), u_init(0.0), v_init(0.0), u_vel(0.0),
          v_vel(0.0), u_init_vel(0.0), v_init_vel(0.0) {}

    // id
    unsigned long long int id;
    // Normalized feature coordinates (with identity intrinsic matrix)
    double u; // horizontal coordinate
    double v; // vertical coordinate
    // Normalized feature coordinates (with identity intrinsic matrix) in
    // initial frame of this feature
    // # (meaningful if this is the first msg of this feature id)
    double u_init;
    double v_init;
    // Velocity of current normalized feature coordinate
    double u_vel;
    double v_vel;
    // Velocity of initial normalized feature coordinate
    double u_init_vel;
    double v_init_vel;
};

class StereoFeatureMeasurement {
  public:
    StereoFeatureMeasurement()
        : id(0), u(0.0), v(0.0), u_init(0.0), v_init(0.0), u_vel(0.0),
          v_vel(0.0), u_init_vel(0.0), v_init_vel(0.0) {}

    // id
    unsigned long long int id;
    // Normalized feature coordinates (with identity intrinsic matrix)
    double u; // horizontal coordinate
    double v; // vertical coordinate

    bool is_stereo;
    double ru;
    double rv;
    // Normalized feature coordinates (with identity intrinsic matrix) in
    // initial frame of this feature
    // # (meaningful if this is the first msg of this feature id)
    double u_init;
    double v_init;
    // Velocity of current normalized feature coordinate
    double u_vel;
    double v_vel;
    // Velocity of initial normalized feature coordinate
    double u_init_vel;
    double v_init_vel;
};

class MonoCameraMeasurement {
  public:
    double timeStampToSec;
    // All features on the current image,
    // including tracked ones and newly detected ones.
    std::vector<MonoFeatureMeasurement> features;
};

} // namespace vision_localization
