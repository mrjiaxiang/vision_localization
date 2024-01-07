#pragma once

#include <Eigen/Dense>

namespace vision_localization {

class PoseData {
  public:
    double time = 0.0;
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    Eigen::Vector3f vel = Eigen::Vector3f::Zero();

  public:
    Eigen::Quaternionf GetQuaternion();
};

} // namespace vision_localization