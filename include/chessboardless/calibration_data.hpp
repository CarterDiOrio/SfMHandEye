#ifndef INC_GUARD_CALIBRATION_DATA_HPP
#define INC_GUARD_CALIBRATION_DATA_HPP

#include <cstddef>
#include <memory>
#include <nlohmann/detail/macro_scope.hpp>
#include <nlohmann/json.hpp>
#include <openMVG/features/image_describer.hpp>
#include <openMVG/features/regions.hpp>
#include <openMVG/image/image_container.hpp>
#include <openMVG/image/pixel_types.hpp>
#include <openMVG/matching/indMatch.hpp>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>
#include <string_view>
#include <unordered_map>
#include <vector>

using Image = openMVG::image::Image<openMVG::image::RGBColor>;
using RegionsPtr = std::unique_ptr<openMVG::features::Regions>;

struct CameraJson {
  /// @brief the filename of the image
  std::string image;

  /// @brief a vector of [x, y, z, rot X, rot Y, rot Z] with rotations in
  /// degrees The convention of the mecha500 for rotation is Z * Y * X
  std::array<double, 6> pose;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CameraJson, image, pose);

struct CameraGroupJson {
  std::vector<CameraJson> cameras;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CameraGroupJson, cameras);

struct DataJson {
  std::vector<CameraGroupJson> groups;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DataJson, groups);

/// @brief Camera Group is a collection of cameras that underwent only
/// translation
struct CameraGroup {
  /// @brief images the images taken by the cameras
  std::vector<openMVG::image::Image<openMVG::image::RGBColor>> images;

  /// @brief the path to each of the images
  std::vector<std::filesystem::path> image_paths;

  /// @brief The hand pose of each camera
  std::vector<Sophus::SE3d> hand_poses;

  /// @brief camera poses` relative to the first camera
  std::vector<Sophus::SE3d> camera_poses;
};

struct Camera {
  /// @brief the id of the camera
  size_t id;

  /// @brief the id of the camera's group
  size_t group_id;

  /// @brief the camera's image
  Image image;

  /// @brief the image filepath
  std::string image_path;

  /// @brief pose of the camera relative to the mean of the group
  Sophus::SE3d pose;

  /// @brief the pose of the hand
  Sophus::SE3d T_base_hand;
};

/// @brief a collection of camera groups
struct CameraSet {

  std::vector<Camera> cameras;

  /// @brief image id to group id
  std::unordered_map<size_t, size_t> image_to_group;

  /// @brief group id to image ids
  std::unordered_map<size_t, std::vector<size_t>> group_to_images;
};

/// @brief interface for a region provider
class RegionProvider {
public:
  virtual openMVG::features::Regions &get_region(size_t idx) = 0;
};

/// @brief keeps all the regions in memory
class MemoryRegionProvider : RegionProvider {
  virtual openMVG::features::Regions &get_region(size_t idx) = 0;

private:
  std::vector<std::unique_ptr<openMVG::features::Regions>> features;
};

/// @brief caches the last requested regions in memory and stores the others on
/// disk
class CachedRegionProvider {
  /// @brief the directory containing the features
  CachedRegionProvider(std::filesystem::path directory);
};

/// @brief handles loading and storing any data in the pipeline
class CalibrationData {

public:
  CalibrationData(const CalibrationData &) = delete;

  /// @brief Constructor
  /// @param data_directory a string representing the path to the directory
  CalibrationData(const std::string_view &data_path);

  /// @brief validates the given path'
  /// @return a path object to the data directory
  static std::filesystem::path
  validate_data_directory(const std::string_view &data_directory);

  /// @brief loads the cameras data
  /// @return a camera set containing the camera data
  CameraSet &load_cameras();

  CameraSet &get_cameras();

  /// @brief checks if the data directory has detected featuers
  /// @return true if it contains a features directory
  bool has_features() const;

  /// @brief store detected features on disk
  /// @param features the vector of detected feature regions
  void store_features(const openMVG::features::Image_describer &describer,
                      const std::vector<RegionsPtr> &features) const;

  /// @brief loads detected feature regions from disk
  std::vector<RegionsPtr>
  load_features(const openMVG::features::Image_describer &describer);

  bool has_matches() const;

  // void store_matches(const openMVG::matches::IndMatches);

private:
  const std::filesystem::path data_directory;
  const std::filesystem::path feature_directory;
  CameraSet camera_set;
};

#endif