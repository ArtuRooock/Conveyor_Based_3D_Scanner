cmake_minimum_required(VERSION 3.28)
project(st3_depthm)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

add_subdirectory(cnpy)

add_executable(optimize_camera_angle_baseline optimize_camera_angle_baseline.cpp)
target_link_libraries(optimize_camera_angle_baseline Eigen3::Eigen cnpy OpenMP::OpenMP_CXX)

add_executable(optimise_laser_angle optimise_laser_angle.cpp)
target_link_libraries(optimise_laser_angle Eigen3::Eigen cnpy OpenMP::OpenMP_CXX)

