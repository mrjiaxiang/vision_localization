#include <cmath>

#include <glog/logging.h>

#include "vision_localization/global_defination/global_defination.h"
#include "vision_localization/ins/imu_integration.h"

namespace vision_localization {
IMUIntegration::IMUIntegration(void) : private_nh_("~"), initialized_(false) {}

void IMUIntegration::InitWithConfig(void) {
    std::string config_file_path =
        WORK_SPACE_PATH + "/" + "config/imu_integration.json";

    LOG(INFO) << "config file path : " << config_file_path << std::endl;
    ifs_.open(config_file_path, std::ios::binary);

    reader_.parse(ifs_, value_);

    std::string imu_topic = value_["imu"]["topic_name"].asString();
    imu_sub_ptr_ =
        std::make_shared<IMUSubscriber>(private_nh_, imu_topic, 10000);

    G_.x() = value_["imu"]["gravity"][0].asDouble();
    G_.y() = value_["imu"]["gravity"][1].asDouble();
    G_.z() = value_["imu"]["gravity"][2].asDouble();

    angular_vel_bias_.x() =
        value_["imu"]["bias"]["angular_velocity"][0].asDouble();
    angular_vel_bias_.y() =
        value_["imu"]["bias"]["angular_velocity"][1].asDouble();
    angular_vel_bias_.z() =
        value_["imu"]["bias"]["angular_velocity"][2].asDouble();

    linear_acc_bias_.x() =
        value_["imu"]["bias"]["linear_acceleration"][0].asDouble();
    linear_acc_bias_.y() =
        value_["imu"]["bias"]["linear_acceleration"][1].asDouble();
    linear_acc_bias_.z() =
        value_["imu"]["bias"]["linear_acceleration"][2].asDouble();

    std::string odom_topic = value_["odometry"]["topic_name"].asString();

    odom_estimation_pub_ =
        private_nh_.advertise<nav_msgs::Odometry>(odom_topic, 500);
}

bool IMUIntegration::Run(void) {
    if (!ReadData())
        return false;

    while (HasData()) {
        if (UpdatePose()) {
            PublishPose();
        }
    }

    return true;
}

bool IMUIntegration::ReadData(void) {
    // fetch IMU measurements into buffer:
    imu_sub_ptr_->ParseData(imu_data_buff_);

    if (static_cast<size_t>(0) == imu_data_buff_.size())
        return false;

    return true;
}

bool IMUIntegration::HasData(void) {
    if (imu_data_buff_.size() < static_cast<size_t>(3))
        return false;

    return true;
}

bool IMUIntegration::UpdatePose(void) {
    if (!initialized_) {
        // use the latest measurement for initialization:
        // OdomData &odom_data = odom_data_buff_.back();
        IMUData imu_data = imu_data_buff_.back();

        // 如果有初始位姿或者速度，使用这些进行初始化
        // pose_ = odom_data.pose;
        // vel_ = odom_data.vel;

        initialized_ = true;

        // odom_data_buff_.clear();
        imu_data_buff_.clear();

        // keep the latest IMU measurement for mid-value integration:
        imu_data_buff_.push_back(imu_data);
    } else {
        //
        // TODO: implement your estimation here
        // const size_t index_curr, const size_t index_prev,Eigen::Vector3d
        // &angular_delta
        // get deltas:
        size_t index_prev = 0;
        size_t index_curr = 1;
        Eigen::Vector3d angular_delta;
        GetAngularDelta(index_curr, index_prev, angular_delta);
        // update orientation:onst Eigen::Vector3d
        // &angular_delta,Eigen::Matrix3d &R_curr,Eigen::Matrix3d &R_prev
        Eigen::Matrix3d R_curr, R_prev;
        UpdateOrientation(angular_delta, R_curr, R_prev);
        // get velocity delta:
        double delta_t;
        Eigen::Vector3d velocity_delta;
        GetVelocityDelta(index_curr, index_prev, R_curr, R_prev, delta_t,
                         velocity_delta);
        // update position:
        UpdatePosition(delta_t, velocity_delta);
        // move forward --
        // NOTE: this is NOT fixed. you should update your buffer according
        // to the method of your choice:
        imu_data_buff_.pop_front();
    }

    return true;
}

bool IMUIntegration::PublishPose() {
    // a. set header:
    message_odom_.header.stamp = ros::Time::now();

    std::string frame_id = value_["odometry"]["frame_id"].asString();

    message_odom_.header.frame_id = frame_id;

    // b. set child frame id:
    message_odom_.child_frame_id = frame_id;

    // b. set orientation:
    Eigen::Quaterniond q(pose_.block<3, 3>(0, 0));
    message_odom_.pose.pose.orientation.x = q.x();
    message_odom_.pose.pose.orientation.y = q.y();
    message_odom_.pose.pose.orientation.z = q.z();
    message_odom_.pose.pose.orientation.w = q.w();

    // c. set position:
    Eigen::Vector3d t = pose_.block<3, 1>(0, 3);
    message_odom_.pose.pose.position.x = t.x();
    message_odom_.pose.pose.position.y = t.y();
    message_odom_.pose.pose.position.z = t.z();

    // d. set velocity:
    message_odom_.twist.twist.linear.x = vel_.x();
    message_odom_.twist.twist.linear.y = vel_.y();
    message_odom_.twist.twist.linear.z = vel_.z();

    odom_estimation_pub_.publish(message_odom_);

    return true;
}

Eigen::Vector3d
IMUIntegration::GetUnbiasedAngularVel(const Eigen::Vector3d &angular_vel) {
    return angular_vel - angular_vel_bias_;
}

Eigen::Vector3d
IMUIntegration::GetUnbiasedLinearAcc(const Eigen::Vector3d &linear_acc,
                                     const Eigen::Matrix3d &R) {
    return R * (linear_acc - linear_acc_bias_) - G_;
}

bool IMUIntegration::GetAngularDelta(const size_t index_curr,
                                     const size_t index_prev,
                                     Eigen::Vector3d &angular_delta) {
    if (index_curr <= index_prev || imu_data_buff_.size() <= index_curr) {
        return false;
    }

    const IMUData &imu_data_curr = imu_data_buff_.at(index_curr);
    const IMUData &imu_data_prev = imu_data_buff_.at(index_prev);

    double delta_t = imu_data_curr.time - imu_data_prev.time;

    Eigen::Vector3d curr_angular_velocity(imu_data_curr.angular_velocity.x,
                                          imu_data_curr.angular_velocity.y,
                                          imu_data_curr.angular_velocity.z);

    Eigen::Vector3d prev_angular_velocity(imu_data_prev.angular_velocity.x,
                                          imu_data_prev.angular_velocity.y,
                                          imu_data_prev.angular_velocity.z);

    Eigen::Vector3d angular_vel_curr =
        GetUnbiasedAngularVel(curr_angular_velocity);
    Eigen::Vector3d angular_vel_prev =
        GetUnbiasedAngularVel(prev_angular_velocity);

    std::string method = value_["method"].asString();

    if (method == std::string("median_integral")) {
        angular_delta = 0.5 * delta_t * (angular_vel_curr + angular_vel_prev);
    } else {
        angular_delta = delta_t * angular_vel_prev;
    }
    return true;
}

bool IMUIntegration::GetVelocityDelta(const size_t index_curr,
                                      const size_t index_prev,
                                      const Eigen::Matrix3d &R_curr,
                                      const Eigen::Matrix3d &R_prev,
                                      double &delta_t,
                                      Eigen::Vector3d &velocity_delta) {
    if (index_curr <= index_prev || imu_data_buff_.size() <= index_curr) {
        return false;
    }

    const IMUData &imu_data_curr = imu_data_buff_.at(index_curr);
    const IMUData &imu_data_prev = imu_data_buff_.at(index_prev);

    delta_t = imu_data_curr.time - imu_data_prev.time;

    Eigen::Vector3d curr_linear_acceleration(
        imu_data_curr.linear_acceleration.x,
        imu_data_curr.linear_acceleration.y,
        imu_data_curr.linear_acceleration.z);

    Eigen::Vector3d prev_linear_acceleration(
        imu_data_curr.linear_acceleration.x,
        imu_data_curr.linear_acceleration.y,
        imu_data_curr.linear_acceleration.z);

    Eigen::Vector3d linear_acc_curr =
        GetUnbiasedLinearAcc(curr_linear_acceleration, R_curr);
    Eigen::Vector3d linear_acc_prev =
        GetUnbiasedLinearAcc(prev_linear_acceleration, R_prev);

    std::string method = value_["method"].asString();

    if (method == std::string("median_integral"))
        velocity_delta = 0.5 * delta_t * (linear_acc_curr + linear_acc_prev);
    else {
        velocity_delta = delta_t * linear_acc_prev;
    }

    return true;
}

void IMUIntegration::UpdateOrientation(const Eigen::Vector3d &angular_delta,
                                       Eigen::Matrix3d &R_curr,
                                       Eigen::Matrix3d &R_prev) {
    // magnitude:
    double angular_delta_mag = angular_delta.norm();
    // direction:
    Eigen::Vector3d angular_delta_dir = angular_delta.normalized();

    // build delta q:
    double angular_delta_cos = cos(angular_delta_mag / 2.0);
    double angular_delta_sin = sin(angular_delta_mag / 2.0);
    Eigen::Quaterniond dq(angular_delta_cos,
                          angular_delta_sin * angular_delta_dir.x(),
                          angular_delta_sin * angular_delta_dir.y(),
                          angular_delta_sin * angular_delta_dir.z());
    Eigen::Quaterniond q(pose_.block<3, 3>(0, 0));

    // update:
    q = q * dq;

    // write back:
    R_prev = pose_.block<3, 3>(0, 0);
    pose_.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
    R_curr = pose_.block<3, 3>(0, 0);
}

void IMUIntegration::UpdatePosition(const double &delta_t,
                                    const Eigen::Vector3d &velocity_delta) {
    pose_.block<3, 1>(0, 3) += delta_t * vel_ + 0.5 * delta_t * velocity_delta;
    vel_ += velocity_delta;
}

} // namespace vision_localization
