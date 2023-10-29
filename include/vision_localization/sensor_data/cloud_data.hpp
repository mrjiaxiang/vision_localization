#ifndef VISION_LOCALIZATION_SENSOR_DATA_CLOUD_DATA_HPP_
#define VISION_LOCALIZATION_SENSOR_DATA_CLOUD_DATA_HPP_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace vision_localization {
class CloudData {
  public:
    using POINT = pcl::PointXYZ;
    using CLOUD = pcl::PointCloud<POINT>;
    using CLOUD_PTR = CLOUD::Ptr;

  public:
    CloudData() : cloud_ptr(new CLOUD()) {}

  public:
    double time = 0.0;
    CLOUD_PTR cloud_ptr;
};
} // namespace vision_localization

#endif