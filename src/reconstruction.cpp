#include "chessboardless/reconstruction.hpp"
#include "chessboardless/graph.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/triangulation.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <memory>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>
#include <optional>
#include <ranges>

namespace slam {

Sophus::SE3d estimate_motion_2d2d(const VisualSlamGraph &graph,
                                  size_t frame_id1, size_t frame_id2) {
  const auto shared_point_ids = get_shared_points(graph, frame_id1, frame_id2);

  std::vector<cv::Point2d> image_points1;
  std::vector<cv::Point2d> image_points2;

  for (const auto &point_id : shared_point_ids) {
    image_points1.push_back(
        graph.get_observation(frame_id1, point_id).feature.keypoint.pt);
    image_points2.push_back(
        graph.get_observation(frame_id2, point_id).feature.keypoint.pt);
  }

  cv::Mat intrinsics;
  cv::eigen2cv(graph.get_frame(frame_id1).intrinsics, intrinsics);

  cv::Mat inliers;
  const auto E = cv::findEssentialMat(image_points1, image_points2, intrinsics,
                                      cv::RANSAC, 0.99, 1.0, 1000, inliers);

  cv::Mat R, t;
  cv::recoverPose(E, image_points1, image_points2, intrinsics, R, t, inliers);

  // convert to SE3
  cv::Mat rotation_matrix;
  cv::Rodrigues(R, rotation_matrix);

  Eigen::Matrix3d rotation_eig;
  cv::cv2eigen(rotation_matrix, rotation_eig);

  return Sophus::SE3d(
      Sophus::SO3d::fitToSO3(rotation_eig),
      Eigen::Vector3d{t.at<double>(0), t.at<double>(1), t.at<double>(2)});
}

void add_frame_relative(VisualSlamGraph &graph,
                        const Eigen::Matrix3d &intrinsics, size_t base_frame_id,
                        const Sophus::SE3d &T_rf,
                        std::optional<size_t> new_frame_id = std::nullopt) {

  const auto &base_frame = graph.get_frame(base_frame_id);
  assert(base_frame.has_pose &&
         "Cannot add relative pose to a base frame without a pose");

  graph.insert_frame(std::make_unique<VisualFrame>(
                         intrinsics, base_frame.T_world_frame * T_rf),
                     new_frame_id);
}

void triangulate(VisualSlamGraph &graph, const std::vector<size_t> &frames) {

  // find the points to triangulate from the set
  std::set<size_t> to_be_triangulated = {};
  for (const auto &frame_id : frames) {
    for (const auto &point_id : graph.get_frame_points(frame_id)) {
      const auto &observers = graph.get_observers(point_id);
      if (!graph.get_point(point_id).has_been_triangulated &&
          !to_be_triangulated.contains(point_id) && observers.size() > 1) {
        to_be_triangulated.insert(point_id);
      }
    }
  }

  // create noise model
  const auto noise_model = gtsam::noiseModel::Isotropic::Sigma(3.0, 0.01);

  // triangulate each point
  for (const auto &point_id : to_be_triangulated) {
    auto &point = graph.get_point(point_id);

    // get all observers of point
    const auto &frame_ids = graph.get_observers(point_id);

    // get all 2d locations
    gtsam::Point3Vector measurements;
    for (const auto &feature :
         get_all_measurements(graph, frame_ids, point_id)) {
      measurements.push_back(gtsam::Point3{feature.pt.x, feature.pt.y, 1.0});
    }

    // camera poses
    gtsam::Pose3Vector poses;
    for (const auto &frame_id : frame_ids) {
      poses.emplace_back(graph.get_frame(frame_id).T_world_frame.matrix());
    }

    const auto triangulated_point =
        gtsam::triangulateLOST(poses, measurements, noise_model);

    point.has_been_triangulated = true;
    point.location = triangulated_point;
  }
}

} // namespace slam