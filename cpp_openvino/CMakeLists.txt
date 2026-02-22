cmake_minimum_required(VERSION 3.20)
project(cpp_openvino_video CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(video_detect_ov src/video_detect_ov.cpp)
target_link_libraries(video_detect_ov PRIVATE openvino::runtime ${OpenCV_LIBS})
target_include_directories(video_detect_ov PRIVATE ${OpenCV_INCLUDE_DIRS})
if(WIN32)
  target_link_libraries(video_detect_ov PRIVATE psapi)
endif()
