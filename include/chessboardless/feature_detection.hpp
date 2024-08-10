#ifndef INC_GUARD_FEATURE_DETECTION_HPP
#define INC_GUARD_FEATURE_DETECTION_HPP

#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>

namespace slam::features {

struct Features {
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  size_t n_levels;
  double level_ratio;
};

struct Feature {
  cv::KeyPoint keypoint;
  cv::Mat descriptor;
  size_t n_levels;
  double level_ratio;
};

struct AnmsPoint {
  cv::KeyPoint kp;
  int idx;
  double radius;
};

Features detect_features(const cv::Mat &img, cv::Feature2D &detector,
                         double max_levels, double level_ratio);

Feature get_feature(const Features &features, size_t idx);

Features adaptive_non_maximal_suppression(const Features &features,
                                          size_t max_features);

double calculate_min_distance_ratio(const Feature &feature);
double calculate_max_distance_ratio(const Feature &feature);

}; // namespace slam::features

#endif