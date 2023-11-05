
#include "vision_localization/publisher/image_publisher.hpp"

namespace vision_localization {
ImagePublisher::ImagePublisher(ros::NodeHandle &nh, std::string topic_name,
                               std::string frame_id, size_t buff_size)
    : nh_(nh), frame_id_(frame_id) {
    publisher_ = nh_.advertise<sensor_msgs::Image>(topic_name, buff_size);
}

void ImagePublisher::Publish(ImageData &image_data_input, double time) {
    ros::Time ros_time(time);
    PublishData(image_data_input, ros_time);
}

void ImagePublisher::Publish(ImageData &image_data_input) {
    ros::Time ros_time(image_data_input.getTime());
    PublishData(image_data_input, ros_time);
}

bool ImagePublisher::HasSubscribers() {
    return publisher_.getNumSubscribers() != 0;
}

void ImagePublisher::PublishData(ImageData &image_data_input, ros::Time time) {
    std_msgs::Header header;
    header.frame_id = frame_id_;
    header.stamp = time;

    cv::Mat image = image_data_input.getImage();

    sensor_msgs::ImagePtr msg =
        cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
    publisher_.publish(*msg);
}
} // namespace vision_localization