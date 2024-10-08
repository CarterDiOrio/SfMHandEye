/// @file features.hpp
/// @brief utilities to make openmvg feature functionality easier to use

#ifndef INC_GUARD_FEATURES_HPP
#define INC_GUARD_FEATURES_HPP

#include "chessboardless/calibration_data.hpp"
#include <algorithm>
#include <concepts>
#include <execution>
#include <format>
#include <image/image_io.hpp>
#include <memory.h>
#include <memory>
#include <openMVG/features/image_describer.hpp>
#include <openMVG/features/sift/SIFT_Anatomy_Image_Describer.hpp>
#include <openMVG/image/image_container.hpp>
#include <openMVG/image/image_converter.hpp>
#include <openMVG/matching/indMatch.hpp>
#include <openMVG/matching/matcher_cascade_hashing.hpp>
#include <openMVG/matching/matcher_type.hpp>
#include <openMVG/matching/regions_matcher.hpp>
#include <openMVG/matching_image_collection/Matcher.hpp>
#include <openMVG/matching_image_collection/Matcher_Regions.hpp>
#include <openMVG/sfm/pipelines/sfm_regions_provider.hpp>
#include <openMVG/sfm/pipelines/sfm_regions_provider_cache.hpp>
#include <openMVG/system/loggerprogress.hpp>
#include <openMVG/tracks/tracks.hpp>
#include <sfm/sfm_data.hpp>
#include <string_view>

/// @brief creates and returns an image describer
/// @param the type of image describer
template<typename ... Params>
std::unique_ptr<openMVG::features::Image_describer>
create_image_describer(const std::string_view type, Params &&... args)
{
  if (type == "SIFT") {
    return std::make_unique<openMVG::features::SIFT_Anatomy_Image_describer>(
      openMVG::features::SIFT_Anatomy_Image_describer::Params (
        std::forward<Params>(args)...));
  } else {
    throw std::invalid_argument(
            std::format("{} describer is unsupported", type));
  }
}

/// @brief Describes images and finds features
/// @param describer the image describer to use
/// @param images the list of images to describe
/// @param workers the number of workers(threads) to use
template<typename T>
std::vector<std::unique_ptr<openMVG::features::Regions>>
describe_images(
  openMVG::features::Image_describer & describer,
  std::vector<openMVG::image::Image<T>> images)
{

  std::cout << "Extracting Features: " << std::endl;

  openMVG::system::LoggerProgress progress_bar(images.size(),
    "- EXTRACT FEATURES -");

  std::vector<std::unique_ptr<openMVG::features::Regions>> image_regions(
    images.size());

  const auto describe = [&describer,
      &progress_bar](const openMVG::image::Image<T> & image) {
    ++progress_bar;
    if constexpr (!std::same_as<T, unsigned char>) {
      openMVG::image::Image<unsigned char> grayscale;
      openMVG::image::ConvertPixelType(image, &grayscale);
      return describer.Describe(grayscale);
    } else {
      return describer.Describe(image);
    }
  };

  std::transform(
    std::execution::par, std::begin(images), std::end(images),
    std::begin(image_regions), describe);

  return image_regions;
}

/// @brief detects the features of the images in a camera set
/// @param camera_set the set of cameras
/// @return the vector of regions for each image, is aligned with the camera
/// indicies.
std::vector<std::unique_ptr<openMVG::features::Regions>>
describe_images(
  openMVG::features::Image_describer & describer,
  const CameraSet & camera_set);

/// @brief finds matches between all image pairs
/// @param matcher the collection matcher to use
/// @param sfm_data the sfm data structure containing meta data
/// @param region_provider the region provider
openMVG::matching::PairWiseMatches
match_pairs(
  const openMVG::matching_image_collection::Matcher & matcher,
  const openMVG::sfm::SfM_Data & sfm_data,
  std::shared_ptr<openMVG::sfm::Regions_Provider> region_provider);

/// @brief filters matches using robust essential matrix estimation
/// @param sfm_data the sfm data
/// @param regions_provider provides the regions
/// @param matches the pairwise matches to filter
openMVG::matching::PairWiseMatches filter_matches(
  const openMVG::sfm::SfM_Data & sfm_data,
  const std::shared_ptr<openMVG::sfm::Regions_Provider> regions_provider,
  const openMVG::matching::PairWiseMatches & matches);

/// @brief creates feature tracks from the pairwise matches
openMVG::tracks::STLMAPTracks
create_feature_tracks(const openMVG::matching::PairWiseMatches & matches);

#endif
