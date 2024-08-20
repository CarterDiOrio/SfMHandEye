#ifndef INC_GUARD_CALIBRATION_DATA_HPP
#define INC_GUARD_CALIBRATION_DATA_HPP

#include <cereal/cereal.hpp>
#include <cstddef>
#include <memory>
#include <nlohmann/detail/macro_scope.hpp>
#include <nlohmann/json.hpp>
#include <openMVG/cameras/Camera_Intrinsics.hpp>
#include <openMVG/features/image_describer.hpp>
#include <openMVG/features/regions.hpp>
#include <openMVG/image/image_container.hpp>
#include <openMVG/image/pixel_types.hpp>
#include <openMVG/matching/indMatch.hpp>
#include <openMVG/sfm/sfm_data.hpp>
#include <openMVG/sfm/sfm_view.hpp>
#include <opencv2/core.hpp>
#include <sfm/pipelines/sfm_regions_provider.hpp>
#include <sophus/se3.hpp>
#include <string_view>
#include <tracks/tracks.hpp>
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

struct Camera : public openMVG::sfm::View {
public:
  Camera() : openMVG::sfm::View() {}

  /// @brief the id of the camera's group
  size_t group_id;

  /// @brief the camera's image
  Image image;

  /// @brief pose of the camera relative to the mean of the group
  Sophus::SE3d pose;

  /// @brief the pose of the hand
  Sophus::SE3d T_base_hand;
};

/// @brief a collection of camera groups
struct CameraSet {

  std::vector<std::shared_ptr<Camera>> cameras;

  /// @brief the average transform hand to base transform of each group
  std::unordered_map<size_t, Sophus::SE3d> average_T_base_hand;

  /// @brief the average transform from the camera to the pase of each group
  std::unordered_map<size_t, Sophus::SE3d> average_T_base_camera;

  /// @brief image id to group id
  std::unordered_map<size_t, size_t> image_to_group;

  /// @brief group id to image ids
  std::unordered_map<size_t, std::vector<size_t>> group_to_images;
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

  /// @brief Gets the intrinsics of the camera
  /// @return camera intrinsics
  std::shared_ptr<openMVG::cameras::IntrinsicBase> get_intrinsics() const;

  /// @brief loads the cameras data
  /// @return a camera set containing the camera data
  CameraSet &load_cameras();

  /// @brief Gets the set of cameras
  /// @return a reference to the set of camera dat
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

  /// @brief gets the filesystem path to a matches file
  std::filesystem::path get_matches_path(bool raw = true);

  /// @brief saves the matches
  /// @param matches the pairwise matches to save
  /// @param filtered raw or filtered matches
  void store_matches(const openMVG::matching::PairWiseMatches &matches,
                     bool raw = true);

  /// @brief loads the matches if they exists from the disk
  /// @param raw raw or filtered matches
  std::optional<openMVG::matching::PairWiseMatches>
  load_matches(bool raw = true);

  inline std::filesystem::path get_data_directory() { return data_directory; }
  inline std::filesystem::path get_feature_directory() {
    return feature_directory;
  }
  inline std::filesystem::path get_matches_directory() {
    return matches_directory;
  }

private:
  std::filesystem::path data_directory;
  std::filesystem::path feature_directory;
  std::filesystem::path matches_directory;
  CameraSet camera_set;
};

/// @brief converts cameras to sfm data where the ids are aligned with those in
/// the camera set
/// @param cameras the camera set to create the SfM data from
/// @param intrinsics the intrinsic values to use for the cameras
openMVG::sfm::SfM_Data cameras_to_sfm_data(
    const CameraSet &cameras,
    std::shared_ptr<openMVG::cameras::IntrinsicBase> intrinsics);

openMVG::sfm::SfM_Data
group_to_sfm_data(const CameraSet &cameras, size_t group_id,
                  std::shared_ptr<openMVG::cameras::IntrinsicBase> intrinsics);

/// @brief Creates an sfm data instance with all of the cameras and landmarks
/// filled out
/// @param cameras the set of cameras
/// @param tracks all the feature tracks
openMVG::sfm::SfM_Data create_sfm_data(
    const CameraSet &cameras,
    std::shared_ptr<openMVG::cameras::IntrinsicBase> intrinsics,
    std::shared_ptr<openMVG::sfm::Regions_Provider> regions_provider,
    const openMVG::tracks::STLMAPTracks &tracks);

/// @brief Creates a new sfm data with only the data from the cameras in the
/// passed groups
/// @param cameras the set of cameras
/// @param sfm_data the sfm data to create the copy from
/// @param groups the group ids that contain the wanted cameras
/// @return a sfm_data object with only views from the given groups
openMVG::sfm::SfM_Data project_to_groups(const CameraSet &cameras,
                                         const openMVG::sfm::SfM_Data &sfm_data,
                                         std::vector<size_t> groups);

/// @brief updates the sfm_data with a projections data
/// @param sfm_data the sfm_data object to consolidate the changes into
/// @param projection a projection of the sfm_data object to consolidate the
/// changes into
void update_sfm_data(openMVG::sfm::SfM_Data &sfm_data,
                     const openMVG::sfm::SfM_Data &projection);

/// @brief Initializes the poses with each camera's relative pose
/// @param camera_set the camera set to pull the poses from
/// @param sfm_data the sfm_data to initialize the poses for
void initialize_poses_from_group(const CameraSet &camera_set,
                                 openMVG::sfm::SfM_Data &sfm_data);

/// @brief Initializes the view poses by using an initial guess
/// This initializes the poses for all views across all groups. Poses
/// within each group have accruate transforms due to only undergoing
/// translation. Between each group the registration is guessed via an initial
/// hand eye guess and the robot arms motion
/// @param camera_set the set of camera data
/// @param sfm_data the sfm data to initialize the poses for
void initialize_poses(const CameraSet &camera_set,
                      openMVG::sfm::SfM_Data &sfm_data);

#endif