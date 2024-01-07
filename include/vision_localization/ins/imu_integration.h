#pragma once

#include <nav_msgs/Odometry.h>
#include <ros/ros.h>

#include <deque>
#include <iostream>

#include <fstream>
#include <jsoncpp/json/json.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "vision_localization/subscriber/imu_subscriber.hpp"

namespace vision_localization {
class IMUIntegration {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  public:
    IMUIntegration(void);
    void InitWithConfig(void);
    bool Run(void);

  private:
    bool ReadData(void);
    bool HasData(void);
    bool UpdatePose(void);
    bool PublishPose(void);

    Eigen::Vector3d GetUnbiasedAngularVel(const Eigen::Vector3d &angular_vel);

    Eigen::Vector3d GetUnbiasedLinearAcc(const Eigen::Vector3d &linear_acc,
                                         const Eigen::Matrix3d &R);

    bool GetAngularDelta(const size_t index_curr, const size_t index_prev,
                         Eigen::Vector3d &angular_delta);

    bool GetVelocityDelta(const size_t index_curr, const size_t index_prev,
                          const Eigen::Matrix3d &R_curr,
                          const Eigen::Matrix3d &R_prev, double &delta_t,
                          Eigen::Vector3d &velocity_delta);

    void UpdateOrientation(const Eigen::Vector3d &angular_delta,
                           Eigen::Matrix3d &R_curr, Eigen::Matrix3d &R_prev);

    void UpdatePosition(const double &delta_t,
                        const Eigen::Vector3d &velocity_delta);

  private:
    // node handler:
    ros::NodeHandle private_nh_;

    // subscriber:
    std::shared_ptr<IMUSubscriber> imu_sub_ptr_;
    ros::Publisher odom_estimation_pub_;

    // data buffer:
    std::deque<IMUData> imu_data_buff_;

    // config:
    bool initialized_ = false;

    // a. gravity constant:
    Eigen::Vector3d G_;
    // b. angular velocity:
    Eigen::Vector3d angular_vel_bias_;
    // c. linear acceleration:
    Eigen::Vector3d linear_acc_bias_;

    // IMU pose estimation:
    Eigen::Matrix4d pose_ = Eigen::Matrix4d::Identity();
    Eigen::Vector3d vel_ = Eigen::Vector3d::Zero();

    nav_msgs::Odometry message_odom_;

    std::ifstream ifs_;
    Json::Reader reader_;
    Json::Value value_;
};
} // namespace vision_localization