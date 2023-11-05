#pragma once

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "vision_localization/params/params.hpp"

namespace vision_localization {
bool inBorder(const cv::Point2f &pt) {
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < Parameters::COL - BORDER_SIZE &&
           BORDER_SIZE <= img_y && img_y < Parameters::ROW - BORDER_SIZE;
}

// 根据状态位，进行“瘦身”
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status) {
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status) {
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}
} // namespace vision_localization