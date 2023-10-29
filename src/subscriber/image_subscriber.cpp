#include <cv_bridge/cv_bridge.h>

#include "vision_localization/subscriber/image_subscriber.hpp"

#include "glog/logging.h"

namespace vision_localization {
ImageSubscriber::ImageSubscriber(ros::NodeHandle &nh, std::string topic_name,
                                 size_t buff_size)
    : nh_(nh) {
    subscriber_ = nh_.subscribe(topic_name, buff_size,
                                &ImageSubscriber::msg_callback, this);
}

void ImageSubscriber::ParseData(std::deque<ImageData> &image_data_buff) {
    buff_mutex_.lock();

    if (new_image_data_.size() > 0) {
        image_data_buff.insert(image_data_buff.end(), new_image_data_.begin(),
                               new_image_data_.end());
        new_image_data_.clear();
    }

    buff_mutex_.unlock();
}

cv::Mat
ImageSubscriber::GetImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg) {
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1") {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    } else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat img = ptr->image.clone();
    return img;
}

void ImageSubscriber::msg_callback(
    const sensor_msgs::Image::ConstPtr &image_msg_ptr) {
    buff_mutex_.lock();

    cv::Mat image = GetImageFromMsg(image_msg_ptr);
    double time = image_msg_ptr->header.stamp.toSec();

    ImageData image_data(image, time);
    new_image_data_.push_back(image_data);

    buff_mutex_.unlock();
}
} // namespace vision_localization