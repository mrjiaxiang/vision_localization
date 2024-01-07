#pragma once

#include <csignal>
#include <cstdio>
#include <execinfo.h>
#include <iostream>
#include <queue>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace Eigen;

namespace vision_localization {
class LKFeatureTracker {
  public:
    LKFeatureTracker();

    void setMask();

    double distance(cv::Point2f &pt1, cv::Point2f &pt2);

    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
    trackImage(double cur_time, const cv::Mat &img);

    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
    trackImage(double cur_time, const cv::Mat &img_left,
               const cv::Mat &img_right);

    bool updateID(unsigned int i);

    void rejectWithF();

    vector<cv::Point2f> undistortedPoints(vector<cv::Point2f> &pts,
                                          int camera_num);

    vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts,
                                    map<int, cv::Point2f> &cur_id_pts,
                                    map<int, cv::Point2f> &prev_id_pts);

    void drawTrack(const cv::Mat &cur_img, vector<int> &ids,
                   vector<cv::Point2f> &curLeftPts,
                   map<int, cv::Point2f> &prevLeftPtsMap);

    void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight,
                   vector<int> &curLeftIds, vector<cv::Point2f> &curLeftPts,
                   vector<cv::Point2f> &curRightPts,
                   map<int, cv::Point2f> &prevLeftPtsMap);

    std::pair<double, cv::Mat> getTrackImage();

    cv::Ptr<cv::CLAHE> clahe_;

    cv::Mat mask_;
    cv::Mat prev_img_, cur_img_, right_img_;
    cv::Mat img_track_;
    vector<cv::Point2f> n_pts_;
    vector<cv::Point2f> predict_pts_;
    vector<cv::Point2f> prev_pts_, cur_pts_, cur_right_pts_;
    vector<cv::Point2f> prev_un_pts_, cur_un_pts_, cur_un_right_pts_;
    vector<cv::Point2f> pts_velocity_, right_pts_velocity_;
    vector<int> ids_, ids_right_;
    vector<int> track_cnt_;
    map<int, cv::Point2f> cur_un_pts_map_, prev_un_pts_map_;
    map<int, cv::Point2f> cur_un_right_pts_map_, prev_un_right_pts_map_;
    map<int, cv::Point2f> prev_left_pts_map_;
    double cur_time_;
    double prev_time_;

    int n_id_;
    bool has_prediction_;
};
} // namespace vision_localization
