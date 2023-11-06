cmake_minimum_required(VERSION 3.0)
project(simple)
set(CMAKE_BUILD_TYPE DEBUG)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)

include_directories(/usr/local/cuda-11.6/include)
link_directories(/usr/local/cuda-11.6/lib64)

include_directories(/wangjiadong/TensorRT-8.4.0.6/include/)
link_directories(/wangjiadong/TensorRT-8.4.0.6/lib)

find_package (OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(a main.cpp)
target_link_libraries(a nvinfer cudart ${OpenCV_LIBS})
