#ifndef VISION_LOCALIZATION_SENSOR_DATA_IMU_DATA_HPP_
#define VISION_LOCALIZATION_SENSOR_DATA_IMU_DATA_HPP_

#include <Eigen/Dense>
#include <cmath>
#include <deque>

namespace vision_localization {
class IMUData {
  public:
    class Orientation {
      public:
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        double w = 0.0;

      public:
        void Normlize() {
            double norm =
                sqrt(pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0) + pow(w, 2.0));
            x /= norm;
            y /= norm;
            z /= norm;
            w /= norm;
        }
    };

    struct LinearAcceleration {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
    };

    struct AngularVelocity {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
    };

    struct AccelBias {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
    };

    struct GyroBias {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
    };

    double time = 0.0;

    Orientation orientation;

    LinearAcceleration linear_acceleration;
    AngularVelocity angular_velocity;

    AccelBias accel_bias;
    GyroBias gyro_bias;

  public:
    // 把四元数转换成旋转矩阵送出去
    Eigen::Matrix3f GetOrientationMatrix() const;
    static bool SyncData(std::deque<IMUData> &UnsyncedData,
                         std::deque<IMUData> &SyncedData, double sync_time);
};
} // namespace vision_localization
#endif