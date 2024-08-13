cmake_minimum_required(VERSION 3.22)
project(chessboardless)

find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)
find_package(OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(GTSAMCMakeTools)
include(GtsamBuildTypes) # Load build type flags and default to Debug mode
find_package(GTSAM REQUIRED)
include_directories(${GTSAM_INCLUDE_DIR})


add_executable(calibrate src/main.cpp 
                         src/graph.cpp 
                         src/feature_detection.cpp
                         src/feature_matching.cpp
                         src/reconstruction.cpp)
target_link_libraries(calibrate ${OpenCV_LIBS} 
                              Sophus::Sophus
                              Ceres::ceres
                              gtsam)

target_include_directories(calibrate
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>)


target_compile_options(calibrate PUBLIC -Wall -Wextra -pedantic)
target_compile_features(calibrate PUBLIC cxx_std_23)