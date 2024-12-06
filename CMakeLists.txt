cmake_minimum_required(VERSION 3.0.2)
project(lane_follower)

## C++11 지원
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  image_transport
  roscpp
  sensor_msgs
)

find_package(OpenCV REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

## 실행 파일 생성
add_executable(lane_follower_node src/lane_follower.cpp)

## 라이브러리 링크
target_link_libraries(lane_follower_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

## catkin_make 후 source 실행
