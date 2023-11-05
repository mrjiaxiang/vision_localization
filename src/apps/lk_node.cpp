#include <ros/ros.h>

#include <glog/logging.h>

#include "vision_localization/global_defination/global_defination.h"

#include "vision_localization/tracker/tracker_flow.hpp"

// params
#include "vision_localization/params/params.hpp"

using namespace vision_localization;

int main(int argc, char **argv) {

    google::InitGoogleLogging(argv[0]);

    FLAGS_log_dir = WORK_SPACE_PATH + "/Log";
    FLAGS_alsologtostderr = 1;

    std::string para_path =
        WORK_SPACE_PATH + "/config/kitti_raw/kitti_10_03_config.yaml";

    ros::init(argc, argv, "lk_node");
    ros::NodeHandle nh;

    std::shared_ptr<TrackerFlow> tracker_flow_ptr =
        std::make_shared<TrackerFlow>(nh);

    Parameters::readParameters(para_path);

    ros::Rate rate(10);
    while (ros::ok()) {
        ros::spinOnce();

        tracker_flow_ptr->Run();

        rate.sleep();
    }
    return 0;
}
