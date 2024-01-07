#include "vision_localization/sensor_data/pose_data.hpp"

namespace vision_localization {

Eigen::Quaternionf PoseData::GetQuaternion() {
    Eigen::Quaternionf q;
    q = pose.block<3, 3>(0, 0);
    return q;
}

} // namespace vision_localization