#include "vision_localization/utils/camera_models.hpp"

CameraInternalParameters camera_internal_parameters;
DistortedParamaters distorted_paramaters;

void cam2World(const Eigen::Vector2d &cam, Eigen::Vector3d &world) {
    world(0) = (cam(0) - camera_internal_parameters.cx) /
               camera_internal_parameters.fx;
    world(1) = (cam(1) - camera_internal_parameters.cy) /
               camera_internal_parameters.fy;
    world(2) = 1;
}

void world2cam(const Eigen::Vector3d &world, Eigen::Vector2d &cam) {
    cam(0) = camera_internal_parameters.fx * world(0) +
             camera_internal_parameters.cx;
    cam(1) = camera_internal_parameters.fy * world(1) +
             camera_internal_parameters.cy;
}
void omnicam2world(const Eigen::Vector2d &cam, Eigen::Vector3d &world) {
    // 像素转换到球面
    double mx_u = (cam(0) - camera_internal_parameters.cx) /
                  camera_internal_parameters.fx;
    double my_u = (cam(1) - camera_internal_parameters.cy) /
                  camera_internal_parameters.fy;
    Eigen::Vector3d P;
    if (camera_internal_parameters.ksi == 1) {
        double lambda = 2.0 / (mx_u * mx_u + my_u * my_u + 1.0);
        P(0) = lambda * mx_u;
        P(1) = lambda * my_u;
        P(2) = lambda - 1.0;
    } else {
        double lambda = (camera_internal_parameters.ksi +
                         sqrt(1.0 + (1.0 - camera_internal_parameters.ksi *
                                               camera_internal_parameters.ksi) *
                                        (mx_u * mx_u + my_u * my_u))) /
                        (1.0 + mx_u * mx_u + my_u * my_u);
        P(0) = lambda * mx_u;
        P(1) = lambda * my_u;
        P(2) = lambda - camera_internal_parameters.ksi;
    }

    // 或者变换到归一化平面
    if (camera_internal_parameters.ksi == 1.0) {
        P(0) = mx_u;
        P(1) = my_u;
        P(2) = (1.0 - mx_u * mx_u - my_u * my_u) / 2.0;
    } else {
        // Reuse variable
        double rho2_d = mx_u * mx_u + my_u * my_u;
        P(0) = mx_u;
        P(1) = my_u;
        P(2) =
            1.0 - camera_internal_parameters.ksi * (rho2_d + 1.0) /
                      (camera_internal_parameters.ksi +
                       sqrt(1.0 + (1.0 - camera_internal_parameters.ksi *
                                             camera_internal_parameters.ksi) *
                                      rho2_d));
    }
}

void omniworld2cam(const Eigen::Vector3d &world, Eigen::Vector2d &cam) {
    // 相机坐标系转到球面
    Eigen::Vector3d Ps;
    double norm =
        world(0) * world(0) + world(1) * world(1) + world(2) * world(2);
    Ps(0) = world(0) / sqrt(norm);
    Ps(1) = world(1) / sqrt(norm);
    Ps(2) = world(2) / sqrt(norm);

    // 变换坐标系，新坐标系的原点位于Cp=(0,0,ksi)
    Ps = Ps + Eigen::Vector3d(0, 0, camera_internal_parameters.ksi);

    // 并转换到归一化平面
    Ps(0) = Ps(0) / Ps(2);
    Ps(1) = Ps(1) / Ps(2);
    Ps(2) = 1;

    cam(0) =
        camera_internal_parameters.fx * Ps(0) + camera_internal_parameters.cx;
    cam(1) =
        camera_internal_parameters.fy * Ps(1) + camera_internal_parameters.cy;
}

void equiModel(const Eigen::Vector3d &dis, Eigen::Vector3d &undis) {
    // 像素到归一化
    float ix = (dis(0) - camera_internal_parameters.cx) /
               camera_internal_parameters.fx;
    float iy = (dis(1) - camera_internal_parameters.cy) /
               camera_internal_parameters.fy;
    // 归一化计算r
    float r = sqrt(ix * ix + iy * iy);
    float theta = atan(r);
    float theta2 = theta * theta;
    float theta4 = theta2 * theta2;
    float theta6 = theta2 * theta4;
    float theta8 = theta4 * theta4;

    float thetad = theta * (1 + distorted_paramaters.k1 * theta2 +
                            distorted_paramaters.k2 * theta4 +
                            distorted_paramaters.k3 * theta6 +
                            distorted_paramaters.k4 * theta8);

    float scaling = (r > 1e-8) ? thetad / r : 1.0;

    undis(0) = camera_internal_parameters.fx * ix * scaling +
               camera_internal_parameters.cx;
    undis(1) = camera_internal_parameters.fy * ix * scaling +
               camera_internal_parameters.cy;
}

void radtanModel() {
    // 先经过内参变换，再进行畸变矫正
    cv::Point2f uv, px;
    const cv::Mat src_pt(1, 1, CV_32FC2, &uv.x);
    cv::Mat dst_pt(1, 1, CV_32FC2, &px.x);
    cv::Mat cvK, cvD;
    double xyz[3] = {0};
    cv::undistortPoints(src_pt, dst_pt, cvK, cvD);
    xyz[0] = px.x;
    xyz[1] = px.y;
    xyz[2] = 1.0;
}

void radtanModel(const Eigen::Vector3d &dis, Eigen::Vector3d &undis) {
    // 先经过镜头发生畸变，再成像过程，乘以内参
    double x, y, r2, r4, r6, a1, a2, a3, cdist, xd, yd;
    x = dis(0);
    y = dis(1);
    r2 = x * x + y * y;
    r4 = r2 * r2;
    r6 = r4 * r2;
    a1 = 2 * x * y;
    a2 = r2 + 2 * x * x;
    a3 = r2 + 2 * y * y;
    cdist = 1 + distorted_paramaters.k1 * r2 + distorted_paramaters.k2 * r4 +
            distorted_paramaters.k3 * r6;
    xd =
        x * cdist + distorted_paramaters.p1 * a1 + distorted_paramaters.p2 * a2;
    yd =
        y * cdist + distorted_paramaters.p1 * a3 + distorted_paramaters.p2 * a1;
    undis(0) =
        xd * camera_internal_parameters.fx + camera_internal_parameters.cx;
    undis(1) =
        yd * camera_internal_parameters.fy + camera_internal_parameters.cy;
}
