#include "chessboardless/feature_matching.hpp"
#include "chessboardless/graph.hpp"
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <ranges>

namespace slam::features {

std::vector<std::pair<size_t, size_t>>
match_features(const VisualSlamGraph &feature_graph, size_t frame_id1,
               size_t frame_id2, cv::Mat img1, cv::Mat img2) {

  const auto &[key_points1, descriptors1] =
      get_frame_keypoints(feature_graph, frame_id1);
  const auto &[key_points2, descriptors2] =
      get_frame_keypoints(feature_graph, frame_id2);

  const auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);

  std::vector<cv::DMatch> dmatches;
  matcher->match(descriptors2, descriptors1, dmatches);

  std::vector<std::pair<size_t, size_t>> matches;
  for (const auto &match : dmatches) {
    matches.push_back(
        {feature_graph.get_frame_points(frame_id1)[match.trainIdx],
         feature_graph.get_frame_points(frame_id2)[match.queryIdx]});
  }

  cv::Mat img;
  cv::drawMatches(img1, key_points1, img2, key_points2, dmatches, img);
  cv::Mat smaller;
  cv::resize(img, smaller, cv::Size(-1, -1), 0.5, 0.5);
  cv::imshow("matches", smaller);
  cv::waitKey(0);
  return matches;
}

} // namespace slam::features
