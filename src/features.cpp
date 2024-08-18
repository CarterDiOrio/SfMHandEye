#include <tracks/tracks.hpp>
#define OPENMVG_USE_OPENMP

#include "chessboardless/features.hpp"
#include <algorithm>
#include <execution>
#include <format>
#include <matching/indMatch.hpp>
#include <memory>
#include <openMVG/matching/indMatch.hpp>
#include <openMVG/matching_image_collection/E_ACRobust.hpp>
#include <openMVG/matching_image_collection/E_ACRobust_Angular.hpp>
#include <openMVG/matching_image_collection/Eo_Robust.hpp>
#include <openMVG/matching_image_collection/F_ACRobust.hpp>
#include <openMVG/matching_image_collection/GeometricFilter.hpp>
#include <openMVG/matching_image_collection/Pair_Builder.hpp>
#include <sfm/pipelines/sfm_regions_provider.hpp>
#include <sfm/sfm_data.hpp>
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

openMVG::matching::PairWiseMatches filter_matches(
    const openMVG::sfm::SfM_Data &sfm_data,
    const std::shared_ptr<openMVG::sfm::Regions_Provider> regions_provider,
    const openMVG::matching::PairWiseMatches &matches) {
  openMVG::matching::PairWiseMatches filtered_matches;

  std::cout << "Filtering matches via robust estimation..." << std::endl;

  // create image collection geometric filter
  auto filter = std::make_unique<
      openMVG::matching_image_collection::ImageCollectionGeometricFilter>(
      &sfm_data, regions_provider);

  openMVG::system::LoggerProgress progress_bar(matches.size(),
                                               "- Geometric Filtering -");

  filter->Robust_model_estimation(
      openMVG::matching_image_collection::GeometricFilter_EMatrix_AC(4.0, 2048),
      matches, false, 0.6, &progress_bar);
  filtered_matches = filter->Get_geometric_matches();

  return filtered_matches;
}

openMVG::tracks::STLMAPTracks
create_feature_tracks(const openMVG::matching::PairWiseMatches &matches) {
  openMVG::tracks::STLMAPTracks tracks;
  openMVG::tracks::TracksBuilder track_builder;

  std::cout << "Building tracks..." << std::endl;
  track_builder.Build(matches);
  track_builder.Filter();
  track_builder.ExportToSTL(tracks);

  return tracks;
}