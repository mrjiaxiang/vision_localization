#ifndef VISION_LOCALIZATION_PUBLISHER_ODOMETRY_PUBLISHER_HPP_
#define VISION_LOCALIZATION_PUBLISHER_ODOMETRY_PUBLISHER_HPP_

#include <string>

#include <Eigen/Dense>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>

#include "vision_localization/sensor_data/velocity_data.hpp"

namespace vision_localization {
class OdometryPublisher {
  public:
    OdometryPublisher(ros::NodeHandle &nh, std::string topic_name,
                      std::string base_frame_id, std::string child_frame_id,
                      int buff_size);
    OdometryPublisher() = default;

    void Publish(const Eigen::Matrix4f &transform_matrix, double time);
    void Publish(const Eigen::Matrix4f &transform_matrix);
    void Publish(const Eigen::Matrix4f &transform_matrix,
                 const VelocityData &velocity_data, double time);
    void Publish(const Eigen::Matrix4f &transform_matrix,
                 const VelocityData &velocity_data);
    void Publish(const Eigen::Matrix4f &transform_matrix,
                 const Eigen::Vector3f &vel, double time);
    void Publish(const Eigen::Matrix4f &transform_matrix,
                 const Eigen::Vector3f &vel);

    bool HasSubscribers();

  private:
    void PublishData(const Eigen::Matrix4f &transform_matrix,
                     const VelocityData &velocity_data, ros::Time time);

  private:
    ros::NodeHandle nh_;
    ros::Publisher publisher_;

    VelocityData velocity_data_;
    nav_msgs::Odometry odometry_;
};
} // namespace vision_localization
#endif