#ifndef VISION_LOCALIZATION_PUBLISHER_CLOUD_PUBLISHER_HPP_
#define VISION_LOCALIZATION_PUBLISHER_CLOUD_PUBLISHER_HPP_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

#include "vision_localization/sensor_data/cloud_data.hpp"

namespace vision_localization {
class CloudPublisher {
  public:
    CloudPublisher(ros::NodeHandle &nh, std::string topic_name,
                   std::string frame_id, size_t buff_size);
    CloudPublisher() = default;

    void Publish(CloudData::CLOUD_PTR &cloud_ptr_input, double time);
    void Publish(CloudData::CLOUD_PTR &cloud_ptr_input);

    bool HasSubscribers();

  private:
    void PublishData(CloudData::CLOUD_PTR &cloud_ptr_input, ros::Time time);

  private:
    ros::NodeHandle nh_;
    ros::Publisher publisher_;
    std::string frame_id_;
};
} // namespace vision_localization
#endif