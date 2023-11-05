#include "vision_localization/feature_tracker/lk_feature_tracker.hpp"
#include "vision_localization/params/params.hpp"
#include "vision_localization/time/tic_toc.h"
#include "vision_localization/tools/fundament_func.hpp"

#include "glog/logging.h"

namespace vision_localization {

LKFeatureTracker::LKFeatureTracker() {
    n_id_ = 0;
    has_prediction_ = false;
    clahe_ = cv::createCLAHE(3.0, cv::Size(8, 8));
}

void LKFeatureTracker::setMask() {
    mask_ = cv::Mat(Parameters::ROW, Parameters::COL, CV_8UC1, cv::Scalar(255));

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < cur_pts_.size(); i++)
        cnt_pts_id.push_back(
            make_pair(track_cnt_[i], make_pair(cur_pts_[i], ids_[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(),
         [](const pair<int, pair<cv::Point2f, int>> &a,
            const pair<int, pair<cv::Point2f, int>> &b) {
             return a.first > b.first;
         });

    cur_pts_.clear();
    ids_.clear();
    track_cnt_.clear();

    for (auto &it : cnt_pts_id) {
        if (mask_.at<uchar>(it.second.first) == 255) {
            cur_pts_.push_back(it.second.first);
            ids_.push_back(it.second.second);
            track_cnt_.push_back(it.first);
            cv::circle(mask_, it.second.first, Parameters::MIN_DIST, 0, -1);
        }
    }
}

double LKFeatureTracker::distance(cv::Point2f &pt1, cv::Point2f &pt2) {
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
LKFeatureTracker::trackImage(double cur_time, const cv::Mat &_img) {
    cur_time_ = cur_time;
    cur_img_ = _img;

    if (Parameters::EQUALIZE)
        clahe_->apply(cur_img_, cur_img_);

    cur_pts_.clear();

    if (prev_pts_.size() > 0) {
        vector<uchar> status;
        vector<float> err;
        if (has_prediction_) {
            cur_pts_ = predict_pts_;
            cv::calcOpticalFlowPyrLK(prev_img_, cur_img_, prev_pts_, cur_pts_,
                                     status, err, cv::Size(21, 21), 1,
                                     cv::TermCriteria(cv::TermCriteria::COUNT +
                                                          cv::TermCriteria::EPS,
                                                      30, 0.01),
                                     cv::OPTFLOW_USE_INITIAL_FLOW);

            int succ_num = 0;
            for (size_t i = 0; i < status.size(); i++) {
                if (status[i])
                    succ_num++;
            }
            if (succ_num < 10)
                cv::calcOpticalFlowPyrLK(prev_img_, cur_img_, prev_pts_,
                                         cur_pts_, status, err,
                                         cv::Size(21, 21), 3);
        } else
            cv::calcOpticalFlowPyrLK(prev_img_, cur_img_, prev_pts_, cur_pts_,
                                     status, err, cv::Size(21, 21), 3);

        // reverse check
        if (Parameters::FLOW_BACK) {
            vector<uchar> reverse_status;
            vector<cv::Point2f> reverse_pts = prev_pts_;
            cv::calcOpticalFlowPyrLK(cur_img_, prev_img_, cur_pts_, reverse_pts,
                                     reverse_status, err, cv::Size(21, 21), 1,
                                     cv::TermCriteria(cv::TermCriteria::COUNT +
                                                          cv::TermCriteria::EPS,
                                                      30, 0.01),
                                     cv::OPTFLOW_USE_INITIAL_FLOW);
            // cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts,
            // reverse_status, err, cv::Size(21, 21), 3);
            for (size_t i = 0; i < status.size(); i++) {
                if (status[i] && reverse_status[i] &&
                    distance(prev_pts_[i], reverse_pts[i]) <= 0.5) {
                    status[i] = 1;
                } else
                    status[i] = 0;
            }
        }

        for (int i = 0; i < int(cur_pts_.size()); i++)
            if (status[i] && !inBorder(cur_pts_[i]))
                status[i] = 0;
        reduceVector(prev_pts_, status);
        reduceVector(cur_pts_, status);
        reduceVector(ids_, status);
        reduceVector(track_cnt_, status);
    }

    for (auto &n : track_cnt_)
        n++;

    rejectWithF();

    setMask();

    int n_max_cnt = Parameters::MAX_CNT - static_cast<int>(cur_pts_.size());
    if (n_max_cnt > 0) {
        if (mask_.empty())
            cout << "mask is empty " << endl;
        if (mask_.type() != CV_8UC1)
            cout << "mask type wrong " << endl;
        cv::goodFeaturesToTrack(cur_img_, n_pts_,
                                Parameters::MAX_CNT - cur_pts_.size(), 0.01,
                                Parameters::MIN_DIST, mask_);
    } else
        n_pts_.clear();

    for (auto &p : n_pts_) {
        cur_pts_.push_back(p);
        ids_.push_back(n_id_++);
        track_cnt_.push_back(1);
    }

    cur_un_pts_ = undistortedPoints(cur_pts_, 0);
    pts_velocity_ =
        ptsVelocity(ids_, cur_un_pts_, cur_un_pts_map_, prev_un_pts_map_);

    if (Parameters::SHOW_TRACK)
        drawTrack(cur_img_, ids_, cur_pts_, prev_left_pts_map_);

    prev_img_ = cur_img_;
    prev_pts_ = cur_pts_;
    prev_un_pts_ = cur_un_pts_;
    prev_un_pts_map_ = cur_un_pts_map_;
    prev_time_ = cur_time_;
    has_prediction_ = false;

    prev_left_pts_map_.clear();
    for (size_t i = 0; i < cur_pts_.size(); i++)
        prev_left_pts_map_[ids_[i]] = cur_pts_[i];

    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> feature_frame;
    for (size_t i = 0; i < ids_.size(); i++) {
        int feature_id = ids_[i];
        double x, y, z;
        x = cur_un_pts_[i].x;
        y = cur_un_pts_[i].y;
        z = 1;
        double p_u, p_v;
        p_u = cur_pts_[i].x;
        p_v = cur_pts_[i].y;
        int camera_id = 0;
        double velocity_x, velocity_y;
        velocity_x = pts_velocity_[i].x;
        velocity_y = pts_velocity_[i].y;

        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        feature_frame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
    }

    return feature_frame;
}

map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
LKFeatureTracker::trackImage(double cur_time, const cv::Mat &img_left,
                             const cv::Mat &img_right) {
    cur_time_ = cur_time;
    cur_img_ = img_left;
    right_img_ = img_right;

    if (Parameters::EQUALIZE) {
        clahe_->apply(cur_img_, cur_img_);
        clahe_->apply(right_img_, right_img_);
    }

    cur_pts_.clear();

    if (prev_pts_.size() > 0) {
        vector<uchar> status;
        vector<float> err;
        if (has_prediction_) {
            cur_pts_ = predict_pts_;
            cv::calcOpticalFlowPyrLK(prev_img_, cur_img_, prev_pts_, cur_pts_,
                                     status, err, cv::Size(21, 21), 1,
                                     cv::TermCriteria(cv::TermCriteria::COUNT +
                                                          cv::TermCriteria::EPS,
                                                      30, 0.01),
                                     cv::OPTFLOW_USE_INITIAL_FLOW);

            int succ_num = 0;
            for (size_t i = 0; i < status.size(); i++) {
                if (status[i])
                    succ_num++;
            }
            if (succ_num < 10)
                cv::calcOpticalFlowPyrLK(prev_img_, cur_img_, prev_pts_,
                                         cur_pts_, status, err,
                                         cv::Size(21, 21), 3);
        } else
            cv::calcOpticalFlowPyrLK(prev_img_, cur_img_, prev_pts_, cur_pts_,
                                     status, err, cv::Size(21, 21), 3);
        // reverse check
        if (Parameters::FLOW_BACK) {
            vector<uchar> reverse_status;
            vector<cv::Point2f> reverse_pts = prev_pts_;
            cv::calcOpticalFlowPyrLK(cur_img_, prev_img_, cur_pts_, reverse_pts,
                                     reverse_status, err, cv::Size(21, 21), 1,
                                     cv::TermCriteria(cv::TermCriteria::COUNT +
                                                          cv::TermCriteria::EPS,
                                                      30, 0.01),
                                     cv::OPTFLOW_USE_INITIAL_FLOW);
            // cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts,
            // reverse_status, err, cv::Size(21, 21), 3);
            for (size_t i = 0; i < status.size(); i++) {
                if (status[i] && reverse_status[i] &&
                    distance(prev_pts_[i], reverse_pts[i]) <= 0.5) {
                    status[i] = 1;
                } else
                    status[i] = 0;
            }
        }

        for (int i = 0; i < int(cur_pts_.size()); i++)
            if (status[i] && !inBorder(cur_pts_[i]))
                status[i] = 0;
        reduceVector(prev_pts_, status);
        reduceVector(cur_pts_, status);
        reduceVector(ids_, status);
        reduceVector(track_cnt_, status);
    }

    for (auto &n : track_cnt_)
        n++;

    rejectWithF();

    setMask();

    int n_max_cnt = Parameters::MAX_CNT - static_cast<int>(cur_pts_.size());
    if (n_max_cnt > 0) {
        if (mask_.empty())
            cout << "mask is empty " << endl;
        if (mask_.type() != CV_8UC1)
            cout << "mask type wrong " << endl;
        cv::goodFeaturesToTrack(cur_img_, n_pts_,
                                Parameters::MAX_CNT - cur_pts_.size(), 0.01,
                                Parameters::MIN_DIST, mask_);
    } else
        n_pts_.clear();

    for (auto &p : n_pts_) {
        cur_pts_.push_back(p);
        ids_.push_back(n_id_++);
        track_cnt_.push_back(1);
    }

    cur_un_pts_ = undistortedPoints(cur_pts_, 0);
    pts_velocity_ =
        ptsVelocity(ids_, cur_un_pts_, cur_un_pts_map_, prev_un_pts_map_);

    ids_right_.clear();
    cur_right_pts_.clear();
    cur_un_right_pts_.clear();
    right_pts_velocity_.clear();
    cur_un_right_pts_map_.clear();
    if (!cur_pts_.empty()) {
        // printf("stereo image; track feature on right image\n");
        vector<cv::Point2f> reverseLeftPts;
        vector<uchar> status, statusRightLeft;
        vector<float> err;
        // cur left ---- cur right
        cv::calcOpticalFlowPyrLK(cur_img_, right_img_, cur_pts_, cur_right_pts_,
                                 status, err, cv::Size(21, 21), 3);
        // reverse check cur right ---- cur left
        if (Parameters::FLOW_BACK) {
            cv::calcOpticalFlowPyrLK(right_img_, cur_img_, cur_right_pts_,
                                     reverseLeftPts, statusRightLeft, err,
                                     cv::Size(21, 21), 3);
            for (size_t i = 0; i < status.size(); i++) {
                if (status[i] && statusRightLeft[i] &&
                    inBorder(cur_right_pts_[i]) &&
                    distance(cur_pts_[i], reverseLeftPts[i]) <= 0.5)
                    status[i] = 1;
                else
                    status[i] = 0;
            }
        }

        ids_right_ = ids_;
        reduceVector(cur_right_pts_, status);
        reduceVector(ids_right_, status);

        cur_un_right_pts_ = undistortedPoints(cur_right_pts_, 1);
        right_pts_velocity_ =
            ptsVelocity(ids_right_, cur_un_right_pts_, cur_un_right_pts_map_,
                        prev_un_right_pts_map_);
    }
    prev_un_right_pts_map_ = cur_un_right_pts_map_;

    if (Parameters::SHOW_TRACK)
        drawTrack(cur_img_, right_img_, ids_, cur_pts_, cur_right_pts_,
                  prev_left_pts_map_);

    prev_img_ = cur_img_;
    prev_pts_ = cur_pts_;
    prev_un_pts_ = cur_un_pts_;
    prev_un_pts_map_ = cur_un_pts_map_;
    prev_time_ = cur_time;
    has_prediction_ = false;

    prev_left_pts_map_.clear();
    for (size_t i = 0; i < cur_pts_.size(); i++)
        prev_left_pts_map_[ids_[i]] = cur_pts_[i];

    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> feature_frame;
    for (size_t i = 0; i < ids_.size(); i++) {
        int feature_id = ids_[i];
        double x, y, z;
        x = cur_un_pts_[i].x;
        y = cur_un_pts_[i].y;
        z = 1;
        double p_u, p_v;
        p_u = cur_pts_[i].x;
        p_v = cur_pts_[i].y;
        int camera_id = 0;
        double velocity_x, velocity_y;
        velocity_x = pts_velocity_[i].x;
        velocity_y = pts_velocity_[i].y;

        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        feature_frame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
    }

    for (size_t i = 0; i < ids_right_.size(); i++) {
        int feature_id = ids_right_[i];
        double x, y, z;
        x = cur_un_right_pts_[i].x;
        y = cur_un_right_pts_[i].y;
        z = 1;
        double p_u, p_v;
        p_u = cur_right_pts_[i].x;
        p_v = cur_right_pts_[i].y;
        int camera_id = 1;
        double velocity_x, velocity_y;
        velocity_x = right_pts_velocity_[i].x;
        velocity_y = right_pts_velocity_[i].y;

        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        feature_frame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
    }

    // printf("feature track whole time %f\n", t_r.toc());
    return feature_frame;
}

void LKFeatureTracker::rejectWithF() {
    // 当前被追踪到的光流至少8个点
    if (cur_pts_.size() >= 8) {
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts_.size()),
            un_prev_pts(prev_pts_.size());
        for (unsigned int i = 0; i < cur_pts_.size(); i++) {
            Eigen::Vector3d tmp_p;
            Parameters::cameras[0]->liftProjective(
                Eigen::Vector2d(cur_pts_[i].x, cur_pts_[i].y), tmp_p);
            tmp_p.x() = Parameters::FOCAL_LENGTH * tmp_p.x() / tmp_p.z() +
                        Parameters::COL / 2.0;
            tmp_p.y() = Parameters::FOCAL_LENGTH * tmp_p.y() / tmp_p.z() +
                        Parameters::ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            Parameters::cameras[0]->liftProjective(
                Eigen::Vector2d(prev_pts_[i].x, prev_pts_[i].y), tmp_p);
            tmp_p.x() = Parameters::FOCAL_LENGTH * tmp_p.x() / tmp_p.z() +
                        Parameters::COL / 2.0;
            tmp_p.y() = Parameters::FOCAL_LENGTH * tmp_p.y() / tmp_p.z() +
                        Parameters::ROW / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC,
                               Parameters::F_THRESHOLD, 0.99, status);

        reduceVector(prev_pts_, status);
        reduceVector(cur_pts_, status);
        reduceVector(cur_un_pts_, status);
        reduceVector(ids_, status);
        reduceVector(track_cnt_, status);
    }
}

vector<cv::Point2f>
LKFeatureTracker::undistortedPoints(vector<cv::Point2f> &pts, int camera_num) {
    vector<cv::Point2f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++) {
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        Parameters::cameras[camera_num]->liftProjective(a, b);
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    }
    return un_pts;
}

vector<cv::Point2f>
LKFeatureTracker::ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts,
                              map<int, cv::Point2f> &cur_id_pts,
                              map<int, cv::Point2f> &prev_id_pts) {
    vector<cv::Point2f> pts_velocity;
    cur_id_pts.clear();
    for (unsigned int i = 0; i < ids.size(); i++) {
        cur_id_pts.insert(make_pair(ids[i], pts[i]));
    }

    // caculate points velocity
    if (!prev_id_pts.empty()) {
        double dt = cur_time_ - prev_time_;

        for (unsigned int i = 0; i < pts.size(); i++) {
            std::map<int, cv::Point2f>::iterator it;
            it = prev_id_pts.find(ids[i]);
            if (it != prev_id_pts.end()) {
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                pts_velocity.push_back(cv::Point2f(v_x, v_y));
            } else
                pts_velocity.push_back(cv::Point2f(0, 0));
        }
    } else {
        for (unsigned int i = 0; i < cur_pts_.size(); i++) {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    return pts_velocity;
}

void LKFeatureTracker::drawTrack(const cv::Mat &cur_img, vector<int> &ids,
                                 vector<cv::Point2f> &curLeftPts,
                                 map<int, cv::Point2f> &prevLeftPtsMap) {
    img_track_ = cur_img.clone();
    cv::cvtColor(img_track_, img_track_, cv::COLOR_GRAY2RGB);

    for (size_t j = 0; j < curLeftPts.size(); j++) {
        double len = std::min(1.0, 1.0 * track_cnt_[j] / 20);
        cv::circle(img_track_, curLeftPts[j], 2,
                   cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }

    map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < ids.size(); i++) {
        int id = ids[i];
        mapIt = prevLeftPtsMap.find(id);
        if (mapIt != prevLeftPtsMap.end()) {
            cv::arrowedLine(img_track_, curLeftPts[i], mapIt->second,
                            cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }
}

void LKFeatureTracker::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight,
                                 vector<int> &curLeftIds,
                                 vector<cv::Point2f> &curLeftPts,
                                 vector<cv::Point2f> &curRightPts,
                                 map<int, cv::Point2f> &prevLeftPtsMap) {
    int cols = imLeft.cols;

    cv::hconcat(imLeft, imRight, img_track_);

    cv::cvtColor(img_track_, img_track_, cv::COLOR_GRAY2RGB);

    for (size_t j = 0; j < curLeftPts.size(); j++) {
        double len = std::min(1.0, 1.0 * track_cnt_[j] / 20);
        cv::circle(img_track_, curLeftPts[j], 2,
                   cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }

    for (size_t i = 0; i < curRightPts.size(); i++) {
        cv::Point2f rightPt = curRightPts[i];
        rightPt.x += cols;
        cv::circle(img_track_, rightPt, 2, cv::Scalar(0, 255, 0), 2);
    }

    map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); i++) {
        int id = curLeftIds[i];
        mapIt = prevLeftPtsMap.find(id);
        if (mapIt != prevLeftPtsMap.end()) {
            cv::arrowedLine(img_track_, curLeftPts[i], mapIt->second,
                            cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }
}

std::pair<double, cv::Mat> LKFeatureTracker::getTrackImage() {
    std::pair<double, cv::Mat> pair_data(cur_time_, img_track_);
    return pair_data;
}
} // namespace vision_localization
