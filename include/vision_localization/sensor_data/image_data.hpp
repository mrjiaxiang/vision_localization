#ifndef VISION_LOCALIZATION_SENSOR_DATA_IMAGE_DATA_HPP_
#define VISION_LOCALIZATION_SENSOR_DATA_IMAGE_DATA_HPP_

#include "glog/logging.h"
#include <opencv2/opencv.hpp>

namespace vision_localization {
class ImageData {
  public:
    ImageData() : time(0.0) {}

    ImageData(const cv::Mat &mat, const double &image_time)
        : image(mat), time(image_time) {}

    ImageData(const ImageData &other)
        : image(other.image.clone()), time(other.time) {}

    ImageData &operator=(const ImageData &other) {
        if (this != &other) {
            image = other.image.clone();
            time = other.time;
        }
        return *this;
    }

    void setImage(const cv::Mat &img) { image = img; }

    void setTime(const double &t) { time = t; }

    cv::Mat getImage() const { return image; }

    double getTime() const { return time; }

  public:
    cv::Mat image;
    double time;
};
} // namespace vision_localization

#endif