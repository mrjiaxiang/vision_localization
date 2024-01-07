#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "vision_localization/extractor/orb_extractor.hpp"
#include "vision_localization/sensor_data/image_data.hpp"
#include "vision_localization/sensor_data/imu_data.hpp"
#include "vision_localization/sensor_data/key_point.hpp"

namespace vision_localization {
typedef unsigned long long int FeatureIDType;
class LarFratureTracker {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  public:
    enum ImageState { FIRST_IMAGE = 1, SECOND_IMAGE = 2, OTHER_IMAGES = 3 };

    LarFratureTracker(const std::string &config_file);

    LarFratureTracker(const LarFratureTracker &) = delete;

    LarFratureTracker operator=(const LarFratureTracker &) = delete;

    ~LarFratureTracker();

    bool initialize();

    bool processMonoImgae(const ImageData &image_data,
                          const std::vector<IMUData> &imu_data_buff,
                          MonoCameraMeasurement &features);

    std::string config_file_;
    ImageState image_state_;

    ImageData prev_image_data_;
    ImageData curr_image_data_;

    std::shared_ptr<ORBextractor> prev_orb_extractor_ptr_;
    std::shared_ptr<ORBextractor> curr_orb_extractor_ptr_;
    std::vector<cv::Mat> vorb_descriptors_;

    std::vector<cv::Mat> prev_pyramid_;
    std::vector<cv::Mat> curr_pyramid_;

    bool bfirst_img_;

    FeatureIDType next_feature_id_;
};
} // namespace vision_localization