#ifndef INC_GUARD_GEOMETRY_HPP
#define INC_GUARD_GEOMETRY_HPP

#include "chessboardless/calibration_data.hpp"
#include <format>
#include <openMVG/geometry/pose3.hpp>
#include <ranges>
#include <sfm/sfm_data.hpp>
#include <sophus/se3.hpp>
#include <tracks/tracks.hpp>

template <typename T>
openMVG::geometry::Pose3 se3_to_pose3(const Sophus::SE3<T> &T_world_camera) {

  const Sophus::SE3<T> T_camera_world = T_world_camera.inverse();

  const Eigen::Vector3<T> center =
      -T_camera_world.rotationMatrix().transpose() *
      T_camera_world.translation();

  return openMVG::geometry::Pose3(T_camera_world.rotationMatrix(), center);
}

template <typename T>
Sophus::SE3<T> pose3_to_se3(const openMVG::geometry::Pose3 &pose3) {
  const Sophus::SO3<T> rotation = pose3.rotation();
  const Eigen::Vector3<T> translation = pose3.translation();

  const Sophus::SE3<T> T_camera_world = Sophus::SE3<T>(rotation, translation);
  return T_camera_world.inverse();
}

/// @brief triangulates points in the track using known poses
/// @param tracks the feature matching tracks
/// @param sfm_data the sfm data containing the known poses and will contain the
/// triangulated structure
/// @param do_bundle_adjust if true performs bundle adjustment to optimize the
/// structure
void triangulate(const openMVG::tracks::STLMAPTracks &tracks,
                 openMVG::sfm::SfM_Data &sfm_data,
                 bool do_bundle_adjust = false);

/// @brief Performs hand eye calibration using SfM
/// @param sfm_data the
Sophus::SE3d calibrate_hand_eye(openMVG::sfm::SfM_Data &sfm_data,
                                CameraSet &camera_set,
                                Sophus::SE3d initial_guess, Eigen::Matrix3d K,
                                const openMVG::sfm::Regions_Provider& regions_provider);

Sophus::SE3d ba_hand_eye(openMVG::sfm::SfM_Data &sfm_data,
                                CameraSet &camera_set,
                                const Sophus::SE3d& initial_guess,  const Eigen::Matrix3d K,
                                const Sophus::SE3d& T_world_base_guess,
                                const openMVG::sfm::Regions_Provider& regions_provider);


void calculate_relative_error(const CameraSet &camera_set,
                              const openMVG::sfm::SfM_Data& sfm_data,
                              const Sophus::SE3d &T_hand_eye);

#endif