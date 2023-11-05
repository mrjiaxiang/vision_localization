#pragma once

#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include <eigen3/Eigen/Dense>

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "vision_localization/camera_models/CameraFactory.h"
#include "vision_localization/camera_models/CataCamera.h"
#include "vision_localization/camera_models/PinholeCamera.h"

namespace vision_localization {

using namespace std;

class Parameters {
  public:
    static void readParameters(std::string config_file);

    static void readIntrinsicParameter();

    static int EQUALIZE;

    static double FOCAL_LENGTH;
    static int WINDOW_SIZE;
    static int NUM_OF_F;
    // #define UNIT_SPHERE_ERROR

    static double INIT_DEPTH;
    static double MIN_PARALLAX;
    static int ESTIMATE_EXTRINSIC;

    static double ACC_N, ACC_W;
    static double GYR_N, GYR_W;

    static std::vector<Eigen::Matrix3d> RIC;
    static std::vector<Eigen::Vector3d> TIC;
    static Eigen::Vector3d G;

    static double BIAS_ACC_THRESHOLD;
    static double BIAS_GYR_THRESHOLD;
    static double SOLVER_TIME;
    static int NUM_ITERATIONS;
    static std::string EX_CALIB_RESULT_PATH;
    static std::string VINS_RESULT_PATH;
    static std::string OUTPUT_FOLDER;
    static std::string IMU_TOPIC;
    static double TD;
    static int ESTIMATE_TD;
    static int ROLLING_SHUTTER;
    static int ROW, COL;
    static int NUM_OF_CAM;
    static int STEREO;
    static int USE_IMU;
    static int MULTIPLE_THREAD;
    // pts_gt for debug purpose;
    static map<int, Eigen::Vector3d> pts_gt;

    static std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
    static std::string FISHEYE_MASK;
    static std::vector<std::string> CAM_NAMES;
    static int MAX_CNT;
    static int MIN_DIST;
    static double F_THRESHOLD;
    static int SHOW_TRACK;
    static int FLOW_BACK;

    static std::vector<CameraPtr> cameras;
    static bool stereo_cam;
};
} // namespace vision_localization