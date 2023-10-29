#ifndef CAMERAFACTORY_H
#define CAMERAFACTORY_H

#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>

#include "vision_localization/camera_models/camera.hpp"

namespace vision_localization {

class CameraFactory {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CameraFactory();

    static boost::shared_ptr<CameraFactory> instance(void);

    CameraPtr generateCamera(Camera::ModelType modelType,
                             const std::string &cameraName,
                             cv::Size imageSize) const;

    CameraPtr generateCameraFromYamlFile(const std::string &filename);

  private:
    static boost::shared_ptr<CameraFactory> m_instance;
};

} // namespace vision_localization

#endif