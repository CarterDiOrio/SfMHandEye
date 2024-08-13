#include "chessboardless/feature_detection.hpp"
#include "chessboardless/feature_matching.hpp"
#include "chessboardless/graph.hpp"
#include "chessboardless/reconstruction.hpp"
#include "chessboardless/vslam_graph.hpp"
#include <Eigen/Dense>
#include <ceres/cost_function.h>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <ranges>
#include <string_view>

namespace ranges = std::ranges;

void display(const std::string_view name, cv::Mat img) {
  cv::Mat smaller;
  cv::resize(img, smaller, cv::Size(-1, -1), 0.5, 0.5);
  cv::imshow(std::string{name}, smaller);
}

int main(int argc, char **argv) {

  std::cout << std::format("Count: {}", argc) << std::endl;

  std::vector<cv::Mat> images;
  for (int i = 1; i < argc; i++) {
    std::puts(argv[i]);
    images.push_back(cv::imread(argv[i], cv::IMREAD_COLOR));
  }

  slam::VisualSlamGraph graph;

  cv::Ptr<cv::ORB> orb = cv::ORB::create(10000, 1.2f, 8);

  Eigen::Matrix3d intrinsics;
  intrinsics << 1381.17626953125, 0, 973.329956054688, 0, 1381.80151367188,
      532.698852539062, 0, 0, 1;

  for (const auto &[idx, image] : ranges::views::enumerate(images)) {
    auto features = slam::features::detect_features(image, *orb, 8, 1.2);
    features = slam::features::adaptive_non_maximal_suppression(features, 1000);

    // put frame into graph
    const auto fid =
        graph.insert_frame(std::make_unique<slam::VisualFrame>(intrinsics));

    std::cout << "ID: " << fid << std::endl;

    for (size_t feature_idx = 0; feature_idx < features.keypoints.size();
         feature_idx++) {
      graph.insert_observation(
          fid, slam::features::get_feature(features, feature_idx));
    }

    // cv::drawKeypoints(image, features.keypoints, image);
    // display("points", image);
    // cv::waitKey(0);
  }

  std::cout << std::format("Total Graph Size: {}", graph.num_points())
            << std::endl;

  const auto matches =
      slam::features::match_features(graph, 1, 2, images[0], images[1]);
  std::cout << std::format("Num Matches: {}", matches.size()) << std::endl;

  for (const auto &[pid1, pid2] : matches) {
    graph.merge_points(pid1, pid2);
  }

  std::cout << std::format("Total Graph Size: {}", graph.num_points())
            << std::endl;

  // estimate motion
  const Sophus::SE3d motion = slam::estimate_motion_2d2d(graph, 1, 2);
  std::cout << motion.translation() << std::endl;
  std::cout << motion.rotationMatrix() << std::endl;

  graph.get_frame(1).T_world_frame = Sophus::SE3d{};
  graph.get_frame(2).T_world_frame = motion.inverse();

  slam::triangulate(graph, {1, 2});

  for (const auto &pid : graph.get_frame_points(1)) {
    auto &point = graph.get_point(pid);

    if (point.has_been_triangulated) {
      std::cout << std::format("{} {} {}", point.location.x(),
                               point.location.y(), point.location.z())
                << std::endl;
    }
  }

  return 1;
}