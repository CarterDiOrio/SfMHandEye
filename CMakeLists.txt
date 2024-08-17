cmake_minimum_required(VERSION 3.22)
project(chessboardless)

find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)
find_package(OpenCV REQUIRED )
find_package(OpenMVG REQUIRED)
find_package(nlohmann_json REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(GTSAMCMakeTools)
include(GtsamBuildTypes) # Load build type flags and default to Debug mode
find_package(GTSAM REQUIRED)
include_directories(${GTSAM_INCLUDE_DIR})


add_executable(calibrate src/main.cpp 
                         src/calibration_data.cpp
                         src/features.cpp)
target_link_libraries(calibrate ${OpenCV_LIBS} 
                              Sophus::Sophus
                              Ceres::ceres
                              gtsam
                              nlohmann_json::nlohmann_json
                              OpenMVG::openMVG_sfm
                              OpenMVG::openMVG_matching
                              OpenMVG::openMVG_features
                              OpenMVG::openMVG_graph
                              OpenMVG::openMVG_image
                              OpenMVG::openMVG_robust_estimation
                              OpenMVG::openMVG_multiview)

target_include_directories(calibrate
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>)


target_compile_options(calibrate PUBLIC -Wall -Wextra -pedantic)
target_compile_features(calibrate PUBLIC cxx_std_23)