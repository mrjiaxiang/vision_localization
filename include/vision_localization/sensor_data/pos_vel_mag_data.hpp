#pragma once

#include <string>

#include <Eigen/Dense>

namespace vision_localization {

class PosVelMagData {
  public:
    double time = 0.0;

    Eigen::Vector3f pos = Eigen::Vector3f::Zero();
    Eigen::Vector3f vel = Eigen::Vector3f::Zero();
    Eigen::Vector3f mag = Eigen::Vector3f::Zero();
};

} // namespace vision_localization