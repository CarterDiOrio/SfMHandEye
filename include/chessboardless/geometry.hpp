#ifndef INC_GUARD_GEOMETRY_HPP
#define INC_GUARD_GEOMETRY_HPP

#include "chessboardless/calibration_data.hpp"
#include <format>
#include <openMVG/geometry/pose3.hpp>
#include <ranges>
#include <sfm/sfm_data.hpp>
#include <sophus/se3.hpp>
#include <tracks/tracks.hpp>
#include <opencv2/imgproc.hpp>
extern "C" {
  #include <mrcal/mrcal.h>
  #include <mrcal/mrcal-types.h>
}

/// @brief converts SE3 to OpenMVG pose3
/// @param T_world_camera the se3 transformation
/// @return the equivalent pose3
template<typename T>
openMVG::geometry::Pose3 se3_to_pose3(const Sophus::SE3<T> & T_world_camera)
{

  const Sophus::SE3<T> T_camera_world = T_world_camera.inverse();

  const Eigen::Vector3<T> center =
    -T_camera_world.rotationMatrix().transpose() *
    T_camera_world.translation();

  return openMVG::geometry::Pose3(T_camera_world.rotationMatrix(), center);
}

/// @brief converts pose3 to SE3
/// @param pose3 the pose3
/// @return the equivalent SE3 transformation from the origin
template<typename T>
Sophus::SE3<T> pose3_to_se3(const openMVG::geometry::Pose3 & pose3)
{
  const Sophus::SO3<T> rotation = pose3.rotation();
  const Eigen::Vector3<T> translation = pose3.translation();

  const Sophus::SE3<T> T_camera_world = Sophus::SE3<T>(rotation, translation);
  return T_camera_world.inverse();
}

/// @brief Creates the x and y reprojection maps from rich to lean mrcal model
/// @param from the rich model to convert from
/// @param to the lean model to convert to
std::pair<cv::Mat, cv::Mat> create_mrcal_reprojection_map(
  const mrcal_cameramodel_t & from,
  const mrcal_cameramodel_t & to);

/// @brief Performs hand eye calibration using SfM
/// AX = XB non linear optimization
/// @param sfm_data the SfM data
/// @param camera_set the camera set
/// @param initial_guess the initial guess for the hand eye calibration
/// @param K the camera intrinsics
/// @param regions_provider the regions provider
/// @return the hand eye calibration
Sophus::SE3d calibrate_hand_eye(
  openMVG::sfm::SfM_Data & sfm_data,
  CameraSet & camera_set,
  Sophus::SE3d initial_guess, Eigen::Matrix3d K,
  const openMVG::sfm::Regions_Provider & regions_provider);

/// @brief Refines the hand eye calibration using bundle adjustment
/// @param sfm_data the SfM data
/// @param camera_set the camera set
/// @param initial_guess the initial guess for the hand eye calibration
/// @param K the camera intrinsics
/// @param T_world_base_guess the initial guess for the world to base transformation
/// @param regions_provider the regions provider
/// @return the refined hand eye calibration
Sophus::SE3d ba_hand_eye(
  openMVG::sfm::SfM_Data & sfm_data,
  CameraSet & camera_set,
  const Sophus::SE3d & initial_guess, const Eigen::Matrix3d K,
  const Sophus::SE3d & T_world_base_guess,
  const openMVG::sfm::Regions_Provider & regions_provider);


void calculate_relative_error(
  const CameraSet & camera_set,
  const openMVG::sfm::SfM_Data & sfm_data,
  const Sophus::SE3d & T_hand_eye);

#endif
