#include "vision_localization/feature_tracker/lar_feature_tracker.hpp"

#include <glog/logging.h>

namespace vision_localization {
LarFratureTracker::LarFratureTracker(const std::string &config_file)
    : config_file_(config_file) {
    image_state_ = FIRST_IMAGE;
    next_feature_id_ = 0;

    if (prev_orb_extractor_ptr_ != nullptr) {
        prev_orb_extractor_ptr_.reset();
    }
    prev_orb_extractor_ptr_ = nullptr;

    if (curr_orb_extractor_ptr_ != nullptr) {
        curr_orb_extractor_ptr_.reset();
    }
    curr_orb_extractor_ptr_ = nullptr;
}

LarFratureTracker::~LarFratureTracker() {

    if (prev_orb_extractor_ptr_ != nullptr) {
        prev_orb_extractor_ptr_.reset();
        prev_orb_extractor_ptr_ = nullptr;
    }

    if (curr_orb_extractor_ptr_ != nullptr) {
        curr_orb_extractor_ptr_.reset();
        curr_orb_extractor_ptr_ = nullptr;
    }
}

bool LarFratureTracker::initialize() { return false; }

bool LarFratureTracker::processMonoImgae(
    const ImageData &image_data, const std::vector<IMUData> &imu_data_buff,
    MonoCameraMeasurement &features) {
    return false;
}

} // namespace vision_localization