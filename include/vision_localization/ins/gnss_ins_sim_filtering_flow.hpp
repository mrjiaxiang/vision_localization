#pragma once

#include <ros/ros.h>

#include "vision_localization/filter/gnss_ins_sim_filtering.hpp"

#include "vision_localization/sensor_data/gnss_data.hpp"
#include "vision_localization/sensor_data/imu_data.hpp"
#include "vision_localization/sensor_data/velocity_data.hpp"

#include "vision_localization/subscriber/gnss_subscriber.hpp"
#include "vision_localization/subscriber/imu_subscriber.hpp"
#include "vision_localization/subscriber/velocity_subscriber.hpp"

#include "vision_localization/publisher/odom_publisher.hpp"

#include "vision_localization/filter/kalman_filter_interface.hpp"

#include "glog/logging.h"

#include <fstream>
#include <jsoncpp/json/json.h>

namespace vision_localization {

class GNSSINSSimFilteringFlow {
  public:
    GNSSINSSimFilteringFlow(ros::NodeHandle &nh,
                            const std::string &config_path);
    bool Run();

    // save odometry for evo evaluation:
    bool SaveOdometry(void);

  private:
    bool ReadData();

    bool HasInited() { return init_flag_; }

    bool HasImuData() {
        if (!imu_data_buff_.empty()) {
            double diff_filter_time =
                current_imu_data_.time - filtering_ptr_->GetTime();

            if (diff_filter_time <= 0.01) {
                return true;
            }
        }

        return false;
    }

    bool HasGNSSData() {
        return (!gnss_data_buff_.empty() && !sync_imu_data_buff_.empty() &&
                !imu_data_buff_.empty() && !vel_data_buff_.empty());
    }

    bool ValidIMUData();

    bool InitGNSS();

    bool InitINS();

    bool HasData();

    bool ValidData();

    bool UpdateLocalization();
    bool CorrectLocalization();

    bool PublishFusionOdom();

    bool UpdateOdometry(const double &time);
    /**
     * @brief  save pose in KITTI format for evo evaluation
     * @param  pose, input pose
     * @param  ofs, output file stream
     * @return true if success otherwise false
     */
    bool SavePose(const Eigen::Matrix4f &pose, std::ofstream &ofs);

  private:
    // subscriber:
    std::shared_ptr<IMUSubscriber> imu_sub_ptr_;
    std::deque<IMUData> imu_data_buff_;
    std::deque<IMUData> sync_imu_data_buff_;

    std::shared_ptr<GNSSSubscriber> gnss_sub_ptr_;
    std::deque<GNSSData> gnss_data_buff_;

    std::shared_ptr<VelocitySubscriber> vel_sub_ptr_;
    std::deque<VelocityData> vel_data_buff_;

    // publisher:
    std::shared_ptr<OdometryPublisher> fused_odom_pub_ptr_;

    // filtering instance:
    std::shared_ptr<GNSSINSSimFiltering> filtering_ptr_;

    IMUData current_imu_data_;
    IMUData current_sync_imu_data_;
    VelocityData current_vel_data_;
    GNSSData current_gnss_data_;

    // fused odometry:
    Eigen::Matrix4f fused_pose_ = Eigen::Matrix4f::Identity();
    Eigen::Vector3f fused_vel_ = Eigen::Vector3f::Zero();

    bool init_flag_{false};

    // trajectory for evo evaluation:
    struct {
        size_t N = 0;

        std::deque<double> time_;
        std::deque<Eigen::Matrix4f> fused_;
    } trajectory;
};
} // namespace vision_localization