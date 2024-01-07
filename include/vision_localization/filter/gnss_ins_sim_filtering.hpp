#pragma once

#include <deque>
#include <fstream>
#include <string>
#include <unordered_map>

#include <Eigen/Dense>

#include <jsoncpp/json/json.h>

#include "vision_localization/sensor_data/gnss_data.hpp"
#include "vision_localization/sensor_data/imu_data.hpp"
#include "vision_localization/sensor_data/velocity_data.hpp"

#include "vision_localization/filter/kalman_filter_interface.hpp"

namespace vision_localization {
class GNSSINSSimFiltering : public KalmanFilterInterface {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  public:
    GNSSINSSimFiltering(const Json::Value &node);

    void Init(const Eigen::Vector3d &vel, const IMUData &imu_data);

    Eigen::Vector3d GetUnbiasedLinearAcc(const Eigen::Vector3d &linear_acc,
                                         const Eigen::Matrix3d &R);

    Eigen::Vector3d GetUnbiasedAngularVel(const Eigen::Vector3d &angular_vel,
                                          const Eigen::Matrix3d &R);

    void UpdateProcessEquation(const Eigen::Vector3d &linear_acc_mid,
                               const Eigen::Vector3d &angular_vel_mid);

    bool SaveObservabilityAnalysis(const MeasurementType &measurement_type);

    void UpdateObservabilityAnalysis(const double &time,
                                     const MeasurementType &measurement_type);

    void ResetState(void);

    void ResetCovariance(void);

    void SetProcessEquation(const Eigen::Matrix3d &C_nb,
                            const Eigen::Vector3d &f_n,
                            const Eigen::Vector3d &w_b);

    bool Update(const IMUData &imu_data);

    bool Correct(const IMUData &imu_data,
                 const MeasurementType &measurement_type,
                 const Measurement &measurement);

    void UpdateErrorEstimation(const double &T,
                               const Eigen::Vector3d &linear_acc_mid,
                               const Eigen::Vector3d &angular_vel_mid);

    void UpdateOdomEstimation(Eigen::Vector3d &linear_acc_mid,
                              Eigen::Vector3d &angular_vel_mid);

    bool GetAngularDelta(const size_t index_curr, const size_t index_prev,
                         Eigen::Vector3d &angular_delta,
                         Eigen::Vector3d &angular_vel_mid);

    void UpdateOrientation(const Eigen::Vector3d &angular_delta,
                           Eigen::Matrix3d &R_curr, Eigen::Matrix3d &R_prev);

    bool GetVelocityDelta(const size_t index_curr, const size_t index_prev,
                          const Eigen::Matrix3d &R_curr,
                          const Eigen::Matrix3d &R_prev, double &T,
                          Eigen::Vector3d &velocity_delta,
                          Eigen::Vector3d &linear_acc_mid);

    void UpdatePosition(const double &T, const Eigen::Vector3d &velocity_delta);

    void CorrectErrorEstimation(const MeasurementType &measurement_type,
                                const Measurement &measurement);

    void CorrectErrorEstimationPosi(const Eigen::Vector3d &Posi,
                                    Eigen::VectorXd &Y, Eigen::MatrixXd &G,
                                    Eigen::MatrixXd &K);

    void EliminateError(void);

    bool IsCovStable(const int INDEX_OFSET, const double THRESH = 1.0e-5);

    void GetQPose(Eigen::MatrixXd &Q, Eigen::VectorXd &Y);

    void GetOdometry(Eigen::Matrix4f &pose, Eigen::Vector3f &vel);

    void GetCovariance(Cov &cov);

  private:
    static constexpr int kDimState{18};

    static constexpr int kIndexErrorPos{0};
    static constexpr int kIndexErrorVel{3};
    static constexpr int kIndexErrorOri{6};
    static constexpr int kIndexErrorAccel{9};
    static constexpr int kIndexErrorGyro{12};
    static constexpr int kIndexErrorG{15};

    static constexpr int kDimProcessNoise{15};

    static constexpr int kIndexNoiseAccel{0};
    static constexpr int kIndexNoiseGyro{3};
    static constexpr int kIndexNoiseBiasAccel{6};
    static constexpr int kIndexNoiseBiasGyro{9};
    static constexpr int kIndexNoiseG{12};

    // dimensions:
    static constexpr int kDimMeasurementPose{3};
    static const int kDimMeasurementPoseNoise{3};

    // state:
    using VectorX = Eigen::Matrix<double, kDimState, 1>;
    using MatrixP = Eigen::Matrix<double, kDimState, kDimState>;

    // process equation:
    using MatrixF = Eigen::Matrix<double, kDimState, kDimState>;
    using MatrixB = Eigen::Matrix<double, kDimState, kDimProcessNoise>;
    using MatrixQ = Eigen::Matrix<double, kDimProcessNoise, kDimProcessNoise>;

    // measurement equation:
    using MatrixGPose = Eigen::Matrix<double, kDimMeasurementPose, kDimState>;
    using MatrixCPose =
        Eigen::Matrix<double, kDimMeasurementPose, kDimMeasurementPoseNoise>;
    using MatrixRPose = Eigen::Matrix<double, kDimMeasurementPoseNoise,
                                      kDimMeasurementPoseNoise>;

    // measurement:
    using VectorYPose = Eigen::Matrix<double, kDimMeasurementPose, 1>;

    // Kalman gain:
    using MatrixKPose = Eigen::Matrix<double, kDimState, kDimMeasurementPose>;

    // state observality matrix:
    using MatrixQPose =
        Eigen::Matrix<double, kDimState * kDimMeasurementPose, kDimState>;

    // odometry estimation from IMU integration:
    Eigen::Matrix4d init_pose_ = Eigen::Matrix4d::Identity();

    Eigen::Matrix4d pose_ = Eigen::Matrix4d::Identity();
    Eigen::Vector3d vel_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro_bias_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d accl_bias_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d g_ = Eigen::Vector3d::Zero();

    // state:
    VectorX X_ = VectorX::Zero();
    MatrixP P_ = MatrixP::Zero();
    // process & measurement equations:
    MatrixF F_ = MatrixF::Zero();
    MatrixB B_ = MatrixB::Zero();
    MatrixQ Q_ = MatrixQ::Zero();

    MatrixGPose GPose_ = MatrixGPose::Zero();
    MatrixCPose CPose_ = MatrixCPose::Zero();
    MatrixRPose RPose_ = MatrixRPose::Zero();
    MatrixQPose QPose_ = MatrixQPose::Zero();

    // measurement:
    VectorYPose YPose_;
};
} // namespace vision_localization