#include "chessboardless/feature_detection.hpp"
#include <algorithm>
#include <limits>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <ranges>

namespace rv = std::ranges::views;

namespace slam::features {

Features detect_features(const cv::Mat &img, cv::Feature2D &detector,
                         double max_levels, double level_ratio) {
  Features detection{};
  detector.detectAndCompute(img, cv::noArray(), detection.keypoints,
                            detection.descriptors);
  detection.n_levels = max_levels;
  detection.level_ratio = level_ratio;
  return detection;
}

Features adaptive_non_maximal_suppression(const Features &features,
                                          size_t max_features) {

  const auto to_anms = [](const cv::KeyPoint &kp, int idx) {
    return AnmsPoint{.kp = kp, .idx = idx, .radius = 0.0};
  };

  std::vector<AnmsPoint> points;
  for (const auto &[idx, kp] : rv::enumerate(features.keypoints)) {
    points.push_back(to_anms(kp, idx));
  }

  const auto greater_response = [](const AnmsPoint &lhs, const AnmsPoint &rhs) {
    return lhs.kp.response > rhs.kp.response;
  };
  std::ranges::sort(points, greater_response);

  for (const auto &[idx, point] : rv::enumerate(points)) {
    double min = std::numeric_limits<double>::max();
    for (const auto &stronger : points | rv::take(idx)) {
      if (point.kp.response < 0.9 * stronger.kp.response) {
        min = std::min(min, cv::norm(point.kp.pt = stronger.kp.pt));
      }
    }
    point.radius = min;
  }

  std::ranges::sort(points, std::ranges::greater{}, &AnmsPoint::radius);

  Features detection;
  for (auto &point : points | rv::take(max_features)) {
    detection.keypoints.push_back(point.kp);
    detection.descriptors.push_back(features.descriptors.row(point.idx));
  }
  return detection;
}

Feature get_feature(const Features &features, size_t idx) {
  return {.keypoint = features.keypoints.at(idx),
          .descriptor = features.descriptors.row(idx),
          .n_levels = features.n_levels,
          .level_ratio = features.level_ratio};
};

double calculate_min_distance_ratio(const Feature &feature) {
  return 1.0 / std::pow(feature.level_ratio,
                        std::abs(feature.keypoint.octave -
                                 static_cast<int>(feature.n_levels)));
}

double calculate_max_distance_ratio(const Feature &feature) {
  return std::pow(feature.level_ratio, feature.keypoint.octave);
}

} // namespace slam::features