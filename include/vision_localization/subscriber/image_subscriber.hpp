#ifndef VISION_LOCALIZATION_SUBSCRIBER_IMAGE_SUBSCRIBER_HPP_
#define VISION_LOCALIZATION_SUBSCRIBER_IMAGE_SUBSCRIBER_HPP_

#include <deque>
#include <mutex>
#include <thread>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include "vision_localization/sensor_data/image_data.hpp"

namespace vision_localization {
class ImageSubscriber {
  public:
    ImageSubscriber(ros::NodeHandle &nh, std::string topic_name,
                    size_t buff_size);
    ImageSubscriber() = default;
    void ParseData(std::deque<ImageData> &deque_cloud_data);
    cv::Mat GetImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg);

  private:
    void msg_callback(const sensor_msgs::Image::ConstPtr &image_msg_ptr);

  private:
    ros::NodeHandle nh_;
    ros::Subscriber subscriber_;

    std::deque<ImageData> new_image_data_;
    std::mutex buff_mutex_;
};

} // namespace vision_localization

#endif
