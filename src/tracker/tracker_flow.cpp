#include "vision_localization/tracker/tracker_flow.hpp"

#include <glog/logging.h>

namespace vision_localization {
TrackerFlow::TrackerFlow(ros::NodeHandle &nh) {
    std::string config_file_path = WORK_SPACE_PATH + "/" + "config/config.json";

    LOG(INFO) << "config file path : " << config_file_path << std::endl;

    ifs_.open(config_file_path, std::ios::binary);
    reader_.parse(ifs_, value_);

    InitSubscribers(nh, value_["image"]);

    tracker_image_pub_ptr_ =
        std::make_shared<ImagePublisher>(nh, "tracking_image", "map", 100);

    lk_feature_tracker_ptr_ = std::make_shared<LKFeatureTracker>();
}

bool TrackerFlow::Run() {
    if (!ReadData()) {
        return false;
    }

    while (HasData()) {
        if (!ValidData()) {
            continue;
        }

        PublishData();
    }

    return true;
}

bool TrackerFlow::InitSubscribers(ros::NodeHandle &nh,
                                  const Json::Value &config_node) {
    image_left_sub_ptr_ = std::make_shared<ImageSubscriber>(
        nh, config_node["left_topic"].asString(),
        config_node["queue_size"].asInt());
    image_right_sub_ptr_ = std::make_shared<ImageSubscriber>(
        nh, config_node["right_topic"].asString(),
        config_node["queue_size"].asInt());
    std::cout << "topic : " << config_node["left_topic"].asString()
              << std::endl;
    return true;
}

bool TrackerFlow::ReadData() {
    image_left_sub_ptr_->ParseData(image_left_buff_);
    image_right_sub_ptr_->ParseData(image_right_buff_);

    if (image_left_buff_.empty() || image_right_buff_.empty()) {
        return false;
    }

    return true;
}

bool TrackerFlow::HasData() {
    if (image_left_buff_.size() == 0)
        return false;
    if (image_right_buff_.size() == 0)
        return false;

    return true;
}

bool TrackerFlow::ValidData() {
    cv::Mat image0, image1;
    double time = 0;

    if (!image_left_buff_.empty() && !image_right_buff_.empty()) {
        double time0 = image_left_buff_.front().getTime();
        double time1 = image_right_buff_.front().getTime();
        // 0.003s sync tolerance
        if (time0 < time1 - 0.003) {
            image_left_buff_.pop_front();
            LOG(INFO) << "throw left image" << std::endl;
        } else if (time0 > time1 + 0.003) {
            image_right_buff_.pop_front();
            LOG(INFO) << "throw right image" << std::endl;
        } else {
            time = image_left_buff_.front().getTime();
            image0 = image_left_buff_.front().getImage();
            image_left_buff_.pop_front();
            image1 = image_right_buff_.front().getImage();
            image_right_buff_.pop_front();
        }
    }
    if (!image0.empty() && !image1.empty()) {
        lk_feature_tracker_ptr_->trackImage(time, image0, image1);
        return true;
    }

    return false;
}

bool TrackerFlow::PublishData() {
    std::pair<double, cv::Mat> pair_data =
        lk_feature_tracker_ptr_->getTrackImage();

    ImageData image_data;
    image_data.setTime(pair_data.first);
    image_data.setImage(pair_data.second);

    tracker_image_pub_ptr_->Publish(image_data);

    return true;
}
} // namespace vision_localization