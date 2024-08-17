#include "chessboardless/features.hpp"
#include <algorithm>
#include <execution>
#include <format>
#include <matching/indMatch.hpp>
#include <memory>
#include <stdexcept>

std::vector<std::unique_ptr<openMVG::features::Regions>>
describe_images(openMVG::features::Image_describer &describer,
                const CameraSet &camera_set) {
  std::vector<Image> images;
  for (const auto &camera : camera_set.cameras) {
    images.push_back(camera.image);
  }
  return describe_images(describer, images);
}

openMVG::matching::IndMatches match_pairs(
    const std::vector<std::unique_ptr<openMVG::features::Regions>> &regions) {

  std::vector<std::pair<size_t, size_t>> pairwise;
  for (size_t i = 0; i < regions.size(); i++) {
    for (size_t j = 0; j < i; j++) {
      pairwise.push_back({i, j});
    }
  }

  openMVG::system::LoggerProgress progress_bar(pairwise.size(),
                                               "- MATCHING FEATURES -");

  const auto match = [&regions,
                      &progress_bar](const std::pair<size_t, size_t> pair) {
    std::vector<openMVG::matching::IndMatch> matches;
    openMVG::matching::Match(
        openMVG::matching::EMatcherType::CASCADE_HASHING_L2,
        *regions[pair.first], *regions[pair.second], matches);
    ++progress_bar;
    return matches;
  };

  openMVG::matching::IndMatches matches(pairwise.size());

  std::transform(std::execution::par, pairwise.begin(), pairwise.end(),
                 matches.begin(), match);
  return matches;
}