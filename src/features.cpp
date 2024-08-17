#include "chessboardless/features.hpp"
#include <algorithm>
#include <execution>
#include <format>
#include <matching/indMatch.hpp>
#include <memory>
#include <openMVG/matching/indMatch.hpp>
#include <openMVG/matching_image_collection/Pair_Builder.hpp>
#include <stdexcept>

std::vector<std::unique_ptr<openMVG::features::Regions>>
describe_images(openMVG::features::Image_describer &describer,
                const CameraSet &camera_set) {
  std::vector<Image> images;
  for (const auto &camera : camera_set.cameras) {
    images.push_back(camera->image);
  }
  return describe_images(describer, images);
}

openMVG::matching::PairWiseMatches
match_pairs(const openMVG::matching_image_collection::Matcher &matcher,
            const openMVG::sfm::SfM_Data &sfm_data,
            std::shared_ptr<openMVG::sfm::Regions_Provider> region_provider) {

  const auto pair_set = openMVG::exhaustivePairs(sfm_data.views.size());
  openMVG::matching::PairWiseMatches matches;

  openMVG::system::LoggerProgress progress_bar(pair_set.size(),
                                               "- MATCHING FEATURES -");
  matcher.Match(region_provider, pair_set, matches, &progress_bar);

  return matches;
}