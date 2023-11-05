#pragma once

#include <memory>

#include <ros/ros.h>

#include <fstream>
#include <jsoncpp/json/json.h>

#include <glog/logging.h>

#include "vision_localization/global_defination/global_defination.h"

//
// TF tree:
//
#include "vision_localization/publisher/tf_broadcaster.hpp"
#include "vision_localization/subscriber/tf_listener.hpp"

//
// subscribers:
//
#include "vision_localization/subscriber/gnss_subscriber.hpp"
#include "vision_localization/subscriber/image_subscriber.hpp"
#include "vision_localization/subscriber/imu_subscriber.hpp"
//
// publishers:
//
#include "vision_localization/publisher/image_publisher.hpp"
#include "vision_localization/publisher/odom_publisher.hpp"

// lk
#include "vision_localization/feature_tracker/lk_feature_tracker.hpp"

// params
#include "vision_localization/params/params.hpp"

namespace vision_localization {
class TrackerFlow {
  public:
    TrackerFlow(ros::NodeHandle &nh);

    bool Run();

  private:
    bool InitSubscribers(ros::NodeHandle &nh, const Json::Value &config_node);

    bool ReadData();
    bool HasData();
    bool ValidData();
    bool PublishData();

  private:
    std::shared_ptr<ImageSubscriber> image_left_sub_ptr_;
    std::shared_ptr<ImageSubscriber> image_right_sub_ptr_;

    std::shared_ptr<ImagePublisher> tracker_image_pub_ptr_;

    std::shared_ptr<LKFeatureTracker> lk_feature_tracker_ptr_;

    std::deque<ImageData> image_left_buff_;
    std::deque<ImageData> image_right_buff_;

    std::ifstream ifs_;
    Json::Reader reader_;
    Json::Value value_;
};

} // namespace vision_localization