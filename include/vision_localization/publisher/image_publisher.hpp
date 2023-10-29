#ifndef VISION_LOCALIZATION_PUBLISHER_CLOUD_PUBLISHER_HPP_
#define VISION_LOCALIZATION_PUBLISHER_CLOUD_PUBLISHER_HPP_

#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include <opencv2/opencv.hpp>

#include "vision_localization/sensor_data/image_data.hpp"

namespace vision_localization {
class ImagePublisher {
  public:
    ImagePublisher(ros::NodeHandle &nh, std::string topic_name,
                   std::string frame_id, size_t buff_size);

    ImagePublisher() = default;

    void Publish(ImageData &image_data_input, double time);
    void Publish(ImageData &image_data_input);

    bool HasSubscribers();

  private:
    void PublishData(ImageData &image_data_input, ros::Time time);

  private:
    ros::NodeHandle nh_;
    ros::Publisher publisher_;
    std::string frame_id_;
};

} // namespace vision_localization

#endif