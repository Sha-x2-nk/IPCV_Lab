cmake_minimum_required(VERSION 3.22)

project(BilateralFilter)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(WITH_CUDA OFF)
find_package(OpenCV REQUIRED)

set(SOURCES main.cpp 
    src/BilateralFilter.cpp
    src/gaussian.cpp
)

add_executable(main ${SOURCES})

set_target_properties( main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/../bin
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../bin
)

target_link_libraries(main ${OpenCV_LIBS})