find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
include_directories(${JSONCPP_LIBRARIES})

list(APPEND ALL_TARGET_LIBRARIES ${JSONCPP_LIBRARIES})