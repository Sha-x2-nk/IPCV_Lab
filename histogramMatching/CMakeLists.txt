cmake_minimum_required(VERSION 3.22)

project(HistMatch)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(WITH_CUDA OFF)
find_package(OpenCV REQUIRED)

set(SOURCES main.cpp src/histogramMatching.cpp)

include_directories({$OpenCV_INCLUDE_DIRS})

add_executable(main ${SOURCES})

target_link_libraries(main ${OpenCV_LIBS})

