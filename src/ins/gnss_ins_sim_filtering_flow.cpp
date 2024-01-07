#include "vision_localization/ins/gnss_ins_sim_filtering_flow.hpp"

#include "vision_localization/global_defination/global_defination.h"

namespace vision_localization {
GNSSINSSimFilteringFlow::GNSSINSSimFilteringFlow(
    ros::NodeHandle &nh, const std::string &config_file_path) {

    std::ifstream ifs(config_file_path, std::ios::binary);
    Json::Value value;
    Json::Reader reader;
    reader.parse(ifs, value);

    std::string imu_topic = value["imu_topic"].asString();
    std::string gps_topic = value["gps_topic"].asString();
    std::string vel_topic = value["vel_topic"].asString();

    imu_sub_ptr_ = std::make_shared<IMUSubscriber>(nh, imu_topic, 10000);
    vel_sub_ptr_ = std::make_shared<VelocitySubscriber>(nh, vel_topic, 1000000);
    gnss_sub_ptr_ = std::make_shared<GNSSSubscriber>(nh, gps_topic, 1000000);

    filtering_ptr_ = std::make_shared<GNSSINSSimFiltering>(
        value["error_state_kalman_filter"]);

    fused_odom_pub_ptr_ =
        std::make_shared<OdometryPublisher>(nh, "gnss_ins", "map", "map", 1000);
}

bool GNSSINSSimFilteringFlow::Run() {
    if (!ReadData())
        return false;

    if (!InitGNSS())
        return false;

    while (HasData()) {
        if (!HasInited()) {
            if (ValidData()) {
                InitINS();
            }
        } else {
            if (HasGNSSData() && ValidData()) {
                if (HasImuData()) {
                    while (HasImuData() && ValidIMUData() &&
                           current_imu_data_.time < current_gnss_data_.time) {
                        UpdateLocalization();
                    }

                    if (current_imu_data_.time >= current_gnss_data_.time) {
                        imu_data_buff_.push_back(current_imu_data_);
                    }
                }

                CorrectLocalization();
            }

            if (HasImuData() && ValidIMUData()) {
                UpdateLocalization();
            }
        }
    }

    return true;
}

bool GNSSINSSimFilteringFlow::InitINS() {
    Eigen::Vector3d vel(current_vel_data_.linear_velocity.x,
                        current_vel_data_.linear_velocity.y,
                        current_vel_data_.linear_velocity.z);

    filtering_ptr_->Init(vel, current_sync_imu_data_);
    LOG(INFO) << "gnss ins init successfull." << std::endl;

    init_flag_ = true;
    return true;
}

bool GNSSINSSimFilteringFlow::ReadData() {
    static std::deque<IMUData> unsynced_imu_;

    imu_sub_ptr_->ParseData(unsynced_imu_);

    imu_data_buff_ = unsynced_imu_;

    while (HasInited() && HasImuData() &&
           imu_data_buff_.front().time < filtering_ptr_->GetTime()) {
        imu_data_buff_.pop_front();
    }

    gnss_sub_ptr_->ParseData(gnss_data_buff_);
    vel_sub_ptr_->ParseData(vel_data_buff_);
    double gnss_time = gnss_data_buff_.front().time;
    IMUData::SyncData(unsynced_imu_, sync_imu_data_buff_, gnss_time);

    return true;
}

bool GNSSINSSimFilteringFlow::InitGNSS() {
    static bool gnss_inited = false;
    if (!gnss_inited) {
        if (gnss_data_buff_.empty())
            return false;

        GNSSData gnss_data = gnss_data_buff_.front();
        gnss_data.InitOriginPosition();
        gnss_inited = true;
    }

    return gnss_inited;
}

bool GNSSINSSimFilteringFlow::HasData() {
    if (gnss_data_buff_.size() == 0)
        return false;
    if (imu_data_buff_.size() == 0)
        return false;
    if (vel_data_buff_.size() == 0)
        return false;

    return true;
}

bool GNSSINSSimFilteringFlow::ValidData() {
    current_gnss_data_ = gnss_data_buff_.front();
    current_vel_data_ = vel_data_buff_.front();
    current_sync_imu_data_ = sync_imu_data_buff_.front();

    double diff_imu_time =
        current_gnss_data_.time - current_sync_imu_data_.time;
    double diff_vel_time = current_gnss_data_.time - current_vel_data_.time;

    if (diff_imu_time < -0.05 || diff_vel_time < -0.05) {
        gnss_data_buff_.pop_front();
        return false;
    }

    if (diff_imu_time > 0.05) {
        imu_data_buff_.pop_front();
        return false;
    }

    if (diff_vel_time > 0.05) {
        vel_data_buff_.pop_front();
        return false;
    }

    gnss_data_buff_.pop_front();
    sync_imu_data_buff_.pop_front();
    vel_data_buff_.pop_front();

    return true;
}

bool GNSSINSSimFilteringFlow::ValidIMUData() {
    current_imu_data_ = imu_data_buff_.front();

    imu_data_buff_.pop_front();

    return true;
}

bool GNSSINSSimFilteringFlow::UpdateLocalization() {
    if (filtering_ptr_->Update(current_imu_data_)) {
        return true;
    }

    return false;
}

bool GNSSINSSimFilteringFlow::CorrectLocalization() {
    KalmanFilterInterface::Measurement current_measurement;
    current_gnss_data_.UpdateXYZ();
    current_measurement.POSI =
        Eigen::Vector3d(current_gnss_data_.local_E, current_gnss_data_.local_N,
                        current_gnss_data_.local_U);
    current_measurement.time = current_gnss_data_.time;

    bool is_fusion_succeeded = filtering_ptr_->Correct(
        current_sync_imu_data_, KalmanFilterInterface::MeasurementType::POSI,
        current_measurement);

    if (is_fusion_succeeded) {
        PublishFusionOdom();

        // add to odometry output for evo evaluation:
        UpdateOdometry(current_gnss_data_.time);
        return true;
    }

    return false;
}

bool GNSSINSSimFilteringFlow::PublishFusionOdom() {
    filtering_ptr_->GetOdometry(fused_pose_, fused_vel_);

    fused_odom_pub_ptr_->Publish(fused_pose_, fused_vel_,
                                 current_imu_data_.time);

    return true;
}

bool GNSSINSSimFilteringFlow::UpdateOdometry(const double &time) {
    trajectory.time_.push_back(time);

    trajectory.fused_.push_back(fused_pose_);

    ++trajectory.N;

    return true;
}

bool GNSSINSSimFilteringFlow::SavePose(const Eigen::Matrix4f &pose,
                                       std::ofstream &ofs) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            ofs << pose(i, j);

            if (i == 2 && j == 3) {
                ofs << std::endl;
            } else {
                ofs << " ";
            }
        }
    }

    return true;
}

bool GNSSINSSimFilteringFlow::SaveOdometry(void) {
    if (0 == trajectory.N) {
        return false;
    }

    // // init output files:
    // std::ofstream fused_odom_ofs;
    // if (!FileManager::CreateFile(fused_odom_ofs,
    //                              WORK_SPACE_PATH +
    //                                  "/slam_data/trajectory/fused.txt")) {
    //     return false;
    // }

    // // write outputs:
    // for (size_t i = 0; i < trajectory.N; ++i) {
    //     SavePose(trajectory.fused_.at(i), fused_odom_ofs);
    // }

    return true;
}

} // namespace vision_localization