cmake_minimum_required(VERSION 3.22)

project(ANNOTATE1)

set(WITH_CUDA OFF)
find_package(OpenCV REQUIRED)

set(SOURCES src/annotate1.cpp)

include_directories({$OpenCV_INCLUDE_DIRS})

add_executable(main ${SOURCES})

target_link_libraries(main ${OpenCV_LIBS})

