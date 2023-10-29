#ifndef VISION_LOCALIZATION_PUBLISHER_TF_BROADCASTER_HPP_
#define VISION_LOCALIZATION_PUBLISHER_TF_BROADCASTER_HPP_

#include <Eigen/Dense>
#include <ros/ros.h>
#include <string>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

namespace vision_localization {
class TFBroadCaster {
  public:
    TFBroadCaster(std::string frame_id, std::string child_frame_id);
    TFBroadCaster() = default;
    void SendTransform(Eigen::Matrix4f pose, double time);

  protected:
    tf::StampedTransform transform_;
    tf::TransformBroadcaster broadcaster_;
};
} // namespace vision_localization
#endif