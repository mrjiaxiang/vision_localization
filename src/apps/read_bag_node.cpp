#include <ros/ros.h>

#include <glog/logging.h>

#include "vision_localization/global_defination/global_defination.h"

//
// TF tree:
//
#include "vision_localization/publisher/tf_broadcaster.hpp"
#include "vision_localization/subscriber/tf_listener.hpp"

//
// subscribers:
//
#include "vision_localization/subscriber/gnss_subscriber.hpp"
#include "vision_localization/subscriber/image_subscriber.hpp"
#include "vision_localization/subscriber/imu_subscriber.hpp"
//
// publishers:
//
#include "vision_localization/publisher/image_publisher.hpp"
#include "vision_localization/publisher/odom_publisher.hpp"

using namespace vision_localization;

int main(int argc, char **argv) {

    google::InitGoogleLogging(argv[0]);

    FLAGS_log_dir = WORK_SPACE_PATH + "/Log";
    FLAGS_alsologtostderr = 1;

    ros::init(argc, argv, "read_bag_node");
    ros::NodeHandle nh;

    //
    // get TF:
    //
    std::shared_ptr<TFListener> lidar_to_imu_tf_sub_ptr =
        std::make_shared<TFListener>(nh, "/imu_link", "/velo_link");
    std::shared_ptr<TFBroadCaster> lidar_to_map_tf_pub_ptr =
        std::make_shared<TFBroadCaster>("/map", "/velo_link");

    //
    // subscribe to topics:
    //
    std::shared_ptr<ImageSubscriber> image_sub_ptr =
        std::make_shared<ImageSubscriber>(
            nh, "/kitti/camera_gray_left/image_raw", 100000);
    std::shared_ptr<IMUSubscriber> imu_sub_ptr =
        std::make_shared<IMUSubscriber>(nh, "/kitti/oxts/imu", 1000000);
    std::shared_ptr<GNSSSubscriber> gnss_sub_ptr =
        std::make_shared<GNSSSubscriber>(nh, "/kitti/oxts/gps/fix", 1000000);

    //
    // register publishers:
    //
    std::shared_ptr<ImagePublisher> image_pub_ptr =
        std::make_shared<ImagePublisher>(nh, "current_image", "/map", 100);
    std::shared_ptr<OdometryPublisher> odom_pub_ptr =
        std::make_shared<OdometryPublisher>(nh, "lidar_odom", "/map",
                                            "velo_link", 100);

    std::deque<ImageData> image_data_buff;
    std::deque<IMUData> imu_data_buff;
    std::deque<GNSSData> gnss_data_buff;

    ros::Rate rate(100);

    while (ros::ok()) {
        ros::spinOnce();

        image_sub_ptr->ParseData(image_data_buff);
        imu_sub_ptr->ParseData(imu_data_buff);
        gnss_sub_ptr->ParseData(gnss_data_buff);

        while (!image_data_buff.empty() && !imu_data_buff.empty() &&
               !gnss_data_buff.empty()) {
            ImageData image_data = image_data_buff.front();
            IMUData imu_data = imu_data_buff.front();
            GNSSData gnss_data = gnss_data_buff.front();

            double d_time = image_data.time - imu_data.time;

            if (d_time < -0.05) {
                image_data_buff.pop_front();
            } else if (d_time > 0.05) {
                imu_data_buff.pop_front();
                gnss_data_buff.pop_front();
            } else {
                image_data_buff.pop_front();
                imu_data_buff.pop_front();
                gnss_data_buff.pop_front();

                image_pub_ptr->Publish(image_data);
            }
        }

        rate.sleep();
    }

    return 0;
}
