#include <ros/ros.h>

#include "vision_localization/ins/imu_integration.h"

using namespace vision_localization;

int main(int argc, char **argv) {
    std::string node_name{"inertial_node"};
    ros::init(argc, argv, node_name);

    IMUIntegration imu_integration;

    imu_integration.InitWithConfig();

    // 100 Hz:
    ros::Rate loop_rate(100);
    while (ros::ok()) {
        ros::spinOnce();

        imu_integration.Run();

        loop_rate.sleep();
    }

    return EXIT_SUCCESS;
}