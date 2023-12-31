#ifndef VISION_LOCALIZATION_SUBSCRIBER_IMU_SUBSCRIBER_HPP_
#define VISION_LOCALIZATION_SUBSCRIBER_IMU_SUBSCRIBER_HPP_

#include <deque>
#include <mutex>
#include <thread>

#include "sensor_msgs/Imu.h"
#include <ros/ros.h>

#include "vision_localization/sensor_data/imu_data.hpp"

namespace vision_localization {
class IMUSubscriber {
  public:
    IMUSubscriber(ros::NodeHandle &nh, std::string topic_name,
                  size_t buff_size);
    IMUSubscriber() = default;
    void ParseData(std::deque<IMUData> &deque_imu_data);

  private:
    void msg_callback(const sensor_msgs::ImuConstPtr &imu_msg_ptr);

  private:
    ros::NodeHandle nh_;
    ros::Subscriber subscriber_;
    std::deque<IMUData> new_imu_data_;

    std::mutex buff_mutex_;
};
} // namespace vision_localization
#endif