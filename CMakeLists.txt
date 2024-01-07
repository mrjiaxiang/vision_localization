cmake_minimum_required(VERSION 3.10.2)
project(vision_localization)

set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS "-std=c++14 -O3 -Wall")
SET(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -O3 -Wall")

# dbow3 
set( DBoW3_INCLUDE_DIRS "/usr/local/include" )
set( DBoW3_LIBS "/usr/local/lib/libDBoW3.a" )

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  pcl_ros
  tf
  eigen_conversions
  message_generation 
  std_srvs
  cv_bridge
)

# find_package(PkgConfig REQUIRED)
# pkg_check_modules(JSONCPP jsoncpp)
# include_directories(${JSONCPP_LIBRARIES})

## Generate messages in the 'msg' folder
# add_message_files(
#     FILES
#     # IMU:
#     IMUGNSSMeasurement.msg
#     # lidar:
#     LidarMeasurement.msg
#     # ESKF covariance monitor:
#     ESKFStd.msg
#     # EKF covariance monitor:
#     EKFStd.msg
#     # synced GNSS-odo measurement:
#     PosVel.msg
#     # synced GNSS-odo-mag measurement:
#     PosVelMag.msg
# )

# add_service_files(
#    FILES
#    saveScanContext.srv
#    saveMap.srv
#    optimizeMap.srv
#    saveOdometry.srv
# )

# generate_messages(
#    DEPENDENCIES
#    std_msgs
#    geometry_msgs
#    sensor_msgs
#    nav_msgs
# )

set(ALL_TARGET_LIBRARIES "")

include(cmake/opencv.cmake)
include(cmake/glog.cmake)
include(cmake/YAML.cmake)
include(cmake/geographic.cmake)
include(cmake/PCL.cmake)
include(cmake/sophus.cmake)
include(cmake/g2o.cmake)
include(cmake/ceres.cmake)
include(cmake/json.cmake)

include_directories(include ${catkin_INCLUDE_DIRS})

include(cmake/global_defination.cmake)
catkin_package()

file(GLOB_RECURSE ALL_SRCS "*.cpp")
file(GLOB_RECURSE NODE_SRCS "src/apps/*_node.cpp")
list(REMOVE_ITEM ALL_SRCS ${NODE_SRCS})

# add_executable(front_end_node src/apps/front_end_node.cpp ${ALL_SRCS})
# add_dependencies(front_end_node ${catkin_EXPORTED_TARGETS})
# target_link_libraries(front_end_node ${catkin_LIBRARIES} ${ALL_TARGET_LIBRARIES})

# add_executable(loop_closing_node src/apps/loop_closing_node.cpp ${ALL_SRCS})
# add_dependencies(loop_closing_node ${catkin_EXPORTED_TARGETS})
# target_link_libraries(loop_closing_node ${catkin_LIBRARIES} ${ALL_TARGET_LIBRARIES})

# add_executable(lio_back_end_node src/apps/lio_back_end_node.cpp ${ALL_SRCS})
# add_dependencies(lio_back_end_node ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
# target_link_libraries(lio_back_end_node ${catkin_LIBRARIES} ${ALL_TARGET_LIBRARIES})

add_executable(read_bag_node src/apps/read_bag_node.cpp ${ALL_SRCS})
target_link_libraries(read_bag_node ${catkin_LIBRARIES} ${ALL_TARGET_LIBRARIES})

add_executable(lk_node src/apps/lk_node.cpp ${ALL_SRCS})
add_dependencies(lk_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(lk_node ${catkin_LIBRARIES} ${ALL_TARGET_LIBRARIES})

add_executable(inertial_node src/apps/inertial_node.cpp ${ALL_SRCS})
add_dependencies(inertial_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(inertial_node ${catkin_LIBRARIES} ${ALL_TARGET_LIBRARIES})

add_executable(gnss_ins_node src/apps/gnss_ins_node.cpp ${ALL_SRCS})
add_dependencies(gnss_ins_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(gnss_ins_node ${catkin_LIBRARIES} ${ALL_TARGET_LIBRARIES})

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
install(TARGETS 
        # data_pretreat_node
        # front_end_node
        # lio_back_end_node
        # loop_closing_node
        read_bag_node
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark cpp header files for installation
install(DIRECTORY 
        include/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
    FILES_MATCHING PATTERN "*.h"
    PATTERN ".svn" EXCLUDE
)

## Mark other directories for installation:
install(DIRECTORY
        launch/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
    FILES_MATCHING PATTERN "*.launch"
)
install(DIRECTORY
        config/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
    FILES_MATCHING PATTERN "*.yaml"
)
install(DIRECTORY
        rviz/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/rviz
    FILES_MATCHING PATTERN "*.rviz"
)
