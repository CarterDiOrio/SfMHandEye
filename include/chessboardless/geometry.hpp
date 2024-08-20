#ifndef INC_GUARD_GEOMETRY_HPP
#define INC_GUARD_GEOMETRY_HPP

#include <openMVG/geometry/pose3.hpp>
#include <sophus/se3.hpp>

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

#endif