#include <glog/logging.h>

#include "vision_localization/global_defination/global_defination.h"
#include "vision_localization/params/params.hpp"

namespace vision_localization {

double Parameters::FOCAL_LENGTH = 460.0;
int Parameters::WINDOW_SIZE = 10;
int Parameters::NUM_OF_F = 1000;

double Parameters::INIT_DEPTH;
double Parameters::MIN_PARALLAX;
int Parameters::ESTIMATE_EXTRINSIC;

int Parameters::EQUALIZE;

double Parameters::ACC_N, Parameters::ACC_W;
double Parameters::GYR_N, Parameters::GYR_W;

std::vector<Eigen::Matrix3d> Parameters::RIC;
std::vector<Eigen::Vector3d> Parameters::TIC;
Eigen::Vector3d Parameters::G;

double Parameters::BIAS_ACC_THRESHOLD;
double Parameters::BIAS_GYR_THRESHOLD;
double Parameters::SOLVER_TIME;
int Parameters::NUM_ITERATIONS;
std::string Parameters::EX_CALIB_RESULT_PATH;
std::string Parameters::VINS_RESULT_PATH;
std::string Parameters::OUTPUT_FOLDER;
std::string Parameters::IMU_TOPIC;
double Parameters::TD;
int Parameters::ESTIMATE_TD;
int Parameters::ROLLING_SHUTTER;
int Parameters::ROW, Parameters::COL;
int Parameters::NUM_OF_CAM;
int Parameters::STEREO;
int Parameters::USE_IMU;
int Parameters::MULTIPLE_THREAD;
// pts_gt for debug purpose;
map<int, Eigen::Vector3d> Parameters::pts_gt;

std::string Parameters::IMAGE0_TOPIC, Parameters::IMAGE1_TOPIC;
std::string Parameters::FISHEYE_MASK;
std::vector<std::string> Parameters::CAM_NAMES;
int Parameters::MAX_CNT;
int Parameters::MIN_DIST;
double Parameters::F_THRESHOLD;
int Parameters::SHOW_TRACK;
int Parameters::FLOW_BACK;

std::vector<CameraPtr> Parameters::cameras;
bool Parameters::stereo_cam;

void Parameters::readParameters(std::string config_file) {
    std::ifstream ifs;
    ifs.open(config_file, std::ios::in);
    if (!ifs.is_open()) {
        LOG(WARNING) << "config_file dosen't exist; wrong config_file path."
                     << std::endl;
        return;
    }

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        LOG(ERROR) << "ERROR: Wrong path to settings." << std::endl;
        fsSettings.release();
        return;
    }

    fsSettings["equalize"] >> EQUALIZE;

    fsSettings["image0_topic"] >> IMAGE0_TOPIC;
    fsSettings["image1_topic"] >> IMAGE1_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    FLOW_BACK = fsSettings["flow_back"];

    MULTIPLE_THREAD = fsSettings["multiple_thread"];

    USE_IMU = fsSettings["imu"];
    LOG(INFO) << "USE_IMU : " << USE_IMU << std::endl;
    if (USE_IMU) {
        fsSettings["imu_topic"] >> IMU_TOPIC;
        LOG(INFO) << "IMU_TOPIC : " << IMU_TOPIC << std::endl;
        ACC_N = fsSettings["acc_n"];
        ACC_W = fsSettings["acc_w"];
        GYR_N = fsSettings["gyr_n"];
        GYR_W = fsSettings["gyr_w"];
        G.z() = fsSettings["g_norm"];
    }

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    fsSettings["output_path"] >> OUTPUT_FOLDER;
    VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
    LOG(INFO) << "result path " << VINS_RESULT_PATH << std::endl;
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2) {
        LOG(WARNING)
            << "have no prior about extrinsic param, calibrate extrinsic param"
            << std::endl;
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    } else {
        if (ESTIMATE_EXTRINSIC == 1) {
            LOG(WARNING) << " Optimize extrinsic param around initial guess!"
                         << std::endl;
            EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            LOG(WARNING) << " fix extrinsic param " << std::endl;

        cv::Mat cv_T;
        fsSettings["body_T_cam0"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    }

    NUM_OF_CAM = fsSettings["num_of_cam"];
    LOG(INFO) << "camera number : " << NUM_OF_CAM << std::endl;

    if (NUM_OF_CAM != 1 && NUM_OF_CAM != 2) {
        LOG(INFO) << "num_of_cam should be 1 or 2" << std::endl;
        assert(0);
    }

    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);

    std::string cam0Calib;
    fsSettings["cam0_calib"] >> cam0Calib;
    std::string cam0Path = configPath + "/" + cam0Calib;
    CAM_NAMES.push_back(cam0Path);

    if (NUM_OF_CAM == 2) {
        STEREO = 1;
        std::string cam1Calib;
        fsSettings["cam1_calib"] >> cam1Calib;
        std::string cam1Path = configPath + "/" + cam1Calib;
        // printf("%s cam1 path\n", cam1Path.c_str() );
        CAM_NAMES.push_back(cam1Path);

        cv::Mat cv_T;
        fsSettings["body_T_cam1"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    }

    readIntrinsicParameter();

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        LOG(INFO) << "Unsynchronized sensors, online estimate time offset, "
                     "initial td: "
                  << TD << std::endl;
    else
        LOG(INFO) << "Synchronized sensors, fix time offset: " << TD
                  << std::endl;

    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    LOG(INFO) << "ROW : " << ROW << " COL : " << COL << std::endl;

    if (!USE_IMU) {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        LOG(INFO)
            << "no imu, fix extrinsic param; no time offset calibration\n";
    }

    fsSettings.release();
}

void Parameters::readIntrinsicParameter() {
    for (size_t i = 0; i < CAM_NAMES.size(); i++) {
        LOG(INFO) << "reading paramerter of camera : " << CAM_NAMES[i];
        CameraPtr camera =
            CameraFactory::instance()->generateCameraFromYamlFile(CAM_NAMES[i]);
        cameras.push_back(camera);
    }
    if (CAM_NAMES.size() == 2)
        stereo_cam = true;
}

} // namespace vision_localization