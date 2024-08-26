cmake_minimum_required(VERSION 3.22)
project(chessboardless)

find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)
find_package(OpenMVG REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs) 
find_library(MRCAL_LIB mrcal HINTS /usr/lib/x86_64-linux-gnu)



add_executable(calibrate src/main.cpp 
                         src/calibration_data.cpp
                         src/features.cpp
                         src/geometry.cpp)
target_link_libraries(calibrate 
                              Sophus::Sophus
                              nlohmann_json::nlohmann_json
                              OpenMVG::openMVG_camera
                              OpenMVG::openMVG_exif
                              OpenMVG::openMVG_features
                              OpenMVG::openMVG_geodesy
                              OpenMVG::openMVG_geometry
                              OpenMVG::openMVG_graph
                              OpenMVG::openMVG_image
                              OpenMVG::openMVG_linearProgramming
                              OpenMVG::openMVG_matching
                              OpenMVG::openMVG_matching_image_collection
                              OpenMVG::openMVG_multiview
                              OpenMVG::openMVG_numeric
                              OpenMVG::openMVG_robust_estimation
                              OpenMVG::openMVG_sfm
                              OpenMVG::openMVG_system
                              Ceres::ceres
                              ${MRCAL_LIB}
                              opencv_core 
                              opencv_imgproc
                              opencv_imgcodecs)

target_include_directories(calibrate
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>)


target_compile_options(calibrate PUBLIC -Wall -Wextra -pedantic)
target_compile_features(calibrate PUBLIC cxx_std_23)