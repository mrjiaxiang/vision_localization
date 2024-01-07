#include <ros/ros.h>

#include "vision_localization/global_defination/global_defination.h"
#include "vision_localization/ins/gnss_ins_sim_filtering_flow.hpp"

#include <glog/logging.h>

using namespace vision_localization;

int main(int argc, char **argv) {

    google::InitGoogleLogging(argv[0]);

    FLAGS_log_dir = WORK_SPACE_PATH + "/Log";
    FLAGS_alsologtostderr = 1;

    std::string node_name{"gnss_ins_node"};
    ros::init(argc, argv, node_name);

    ros::NodeHandle nh;

    std::string config_file_path =
        WORK_SPACE_PATH + "/config/filter/gnss_ins.json";

    std::shared_ptr<GNSSINSSimFilteringFlow> gnss_ins_sim_filter =
        std::make_shared<GNSSINSSimFilteringFlow>(nh, config_file_path);

    // 100 Hz:
    ros::Rate loop_rate(100);
    while (ros::ok()) {
        ros::spinOnce();

        gnss_ins_sim_filter->Run();

        loop_rate.sleep();
    }

    return EXIT_SUCCESS;
}