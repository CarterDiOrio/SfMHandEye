#include "chessboardless/calibration_data.hpp"

#include "chessboardless/geometry.hpp"
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Geometry/AngleAxis.h>
#include <Eigen/src/Geometry/Quaternion.h>
#include <algorithm>
#include <cameras/Camera_Intrinsics.hpp>
#include <execution>
#include <features/regions.hpp>
#include <filesystem>
#include <format>
#include <fstream>
#include <geometry/pose3.hpp>
#include <iostream>
#include <matching/indMatch.hpp>
#include <matching/indMatch_utils.hpp>
#include <numeric>
#include <openMVG/cameras/Camera_Pinhole.hpp>
#include <openMVG/image/image_io.hpp>
#include <openMVG/system/loggerprogress.hpp>
#include <opencv2/imgcodecs.hpp>
#include <optional>
#include <sfm/pipelines/sfm_regions_provider.hpp>
#include <sfm/sfm_data.hpp>
#include <sfm/sfm_landmark.hpp>
#include <sophus/average.hpp>
#include <stdexcept>
#include <tracks/tracks.hpp>

static constexpr double deg2rad = M_PI / 180;
static const Eigen::AngleAxisd z90{-90.0 * deg2rad, Eigen::Vector3d::UnitZ()};
static const Sophus::SE3d T_hand_camera =
    Sophus::SE3d{z90.matrix(), Eigen::Vector3d::Zero()};

CalibrationData::CalibrationData(const std::string_view &data_path)
    : data_directory{CalibrationData::validate_data_directory(data_path)} {
  feature_directory = data_directory / "features";
  matches_directory = data_directory / "matches";

  load_cameras();
}

std::filesystem::path CalibrationData::validate_data_directory(
    const std::string_view &data_directory) {
  std::filesystem::path directory{data_directory};

  if (!std::filesystem::exists(directory)) {
    throw std::invalid_argument(
        std::format("Data directory {} does not exist", directory.string()));
  }

  if (!std::filesystem::is_directory(directory)) {
    throw std::invalid_argument(
        std::format("{} is not a directory", directory.string()));
  }
  return directory;
}

CameraSet &CalibrationData::load_cameras() {
  // load data file
  const auto data_filepath = data_directory / "data.json";
  std::ifstream data_file{data_filepath};
  const auto json = nlohmann::json::parse(data_file);
  const auto data = json.get<DataJson>();

  size_t camera_id = 0;

  // reset internal camera_set
  camera_set = CameraSet{};

  for (const auto &[group_id, camera_group_json] :
       std::ranges::views::enumerate(data.groups)) {

    camera_set.group_to_images[group_id] = {};
    std::vector<std::shared_ptr<Camera>> cameras(
        camera_group_json.cameras.size());

    // load in images and convert hand poses to SE3
    for (const auto &[idx, camera_json] :
         std::ranges::views::enumerate(camera_group_json.cameras)) {

      cameras[idx] = std::make_shared<Camera>();

      std::cout << std::format("Loading Image: {}\n", camera_id);

      const auto image_filepath = data_directory / camera_json.image;
      if (!std::filesystem::exists(image_filepath)) {
        throw std::invalid_argument(
            std::format("{} image does not exist. Malformed data file",
                        image_filepath.string()));
      }
      openMVG::image::ReadImage(image_filepath.c_str(), &cameras[idx]->image);

      // std::cout << std::format("{} {} {}\n", camera_json.pose[3],
      //                          camera_json.pose[4], camera_json.pose[5]);

      const Eigen::AngleAxisd x_rot{camera_json.pose[3] * deg2rad,
                                    Eigen::Vector3d::UnitX()};

      const Eigen::AngleAxisd y_rot{camera_json.pose[4] * deg2rad,
                                    Eigen::Vector3d::UnitY()};

      const Eigen::AngleAxisd z_rot{camera_json.pose[5] * deg2rad,
                                    Eigen::Vector3d::UnitZ()};

      const Sophus::SO3d rotation{(x_rot * y_rot * z_rot).toRotationMatrix()};

      const Eigen::Vector3d translation{
          camera_json.pose[0], camera_json.pose[1], camera_json.pose[2]};
      const Sophus::SE3d T_base_hand{rotation, translation};

      cameras[idx]->T_base_hand = Sophus::SE3d(rotation, translation);
      cameras[idx]->s_Img_path = image_filepath;
      cameras[idx]->group_id = group_id;
      cameras[idx]->id_view = camera_id++;
      cameras[idx]->ui_width = cameras[idx]->image.Width();
      cameras[idx]->ui_height = cameras[idx]->image.Height();

      camera_set.image_to_group[cameras[idx]->id_view] = group_id;
      camera_set.group_to_images[group_id].push_back(cameras[idx]->id_view);
    }

    // get the mean hand pose of the group
    std::vector<Sophus::SE3d> hand_poses;
    for (const auto &camera : cameras) {
      hand_poses.push_back(camera->T_base_hand);
    }
    const Sophus::SE3d T_base_hand_average = *Sophus::average(hand_poses);

    // get the mean camera pose of the group
    std::vector<Sophus::SE3d> camera_poses;
    for (const auto &camera : cameras) {
      camera_poses.push_back(camera->T_base_hand * T_hand_camera);
    };
    const Sophus::SE3d T_base_camerap = *Sophus::average(camera_poses);

    // add camera poses relative to the mean transform
    for (auto &camera : cameras) {
      camera->pose =
          T_base_camerap.inverse() * camera->T_base_hand * T_hand_camera;
    }

    // add cameras to camera set
    camera_set.cameras.insert(camera_set.cameras.end(), cameras.begin(),
                              cameras.end());
    camera_set.average_T_base_hand[group_id] = T_base_hand_average;
    camera_set.average_T_base_camera[group_id] = T_base_camerap;
  }

  return camera_set;
}

CameraSet &CalibrationData::get_cameras() { return camera_set; }

bool CalibrationData::has_features() const {
  return std::filesystem::exists(feature_directory);
};

void CalibrationData::store_features(
    const openMVG::features::Image_describer &describer,
    const std::vector<RegionsPtr> &features) const {
  // create the directory if it doesn't exist
  std::filesystem::create_directory(feature_directory);

  const auto save_features = [this, &describer](const auto &pair) {
    const std::filesystem::path img_path{
        camera_set.cameras[std::get<0>(pair)]->s_Img_path};
    const auto name = img_path.stem();
    const auto features_file =
        feature_directory / std::format("{}.feat", name.string());
    const auto descriptors_file =
        feature_directory / std::format("{}.desc", name.string());

    describer.Save(std::get<1>(pair).get(), features_file, descriptors_file);
  };

  const auto range = std::ranges::views::enumerate(features);
  std::for_each(std::execution::par, std::ranges::begin(range),
                std::ranges::end(range), save_features);
}

std::shared_ptr<openMVG::cameras::IntrinsicBase>
CalibrationData::get_intrinsics() const {
  const auto intrinsics = std::make_shared<openMVG::cameras::Pinhole_Intrinsic>(
      1920, 1080, 1381.17626953125, 973.329956054688, 532.698852539062);
  return intrinsics;
}

std::vector<RegionsPtr> CalibrationData::load_features(
    const openMVG::features::Image_describer &describer) {
  openMVG::system::LoggerProgress progress_bar(camera_set.cameras.size(),
                                               "- LOADING FEATURES -");

  const auto load_feature = [this, &describer, &progress_bar](
                                const std::shared_ptr<Camera const> &camera) {
    const std::filesystem::path img_path{camera->s_Img_path};
    const auto name = img_path.stem();
    const auto features_path =
        feature_directory / std::format("{}.feat", name.string());
    const auto descriptors_path =
        feature_directory / std::format("{}.desc", name.string());

    auto regions = describer.Allocate();
    describer.Load(regions.get(), features_path, descriptors_path);
    ++progress_bar;

    return regions;
  };

  std::vector<RegionsPtr> regions(camera_set.cameras.size());
  std::transform(std::execution::par, camera_set.cameras.begin(),
                 camera_set.cameras.end(), regions.begin(), load_feature);

  return regions;
}

std::filesystem::path CalibrationData::get_matches_path(bool raw) {
  return matches_directory /
         std::format("matches_{}.txt", (raw) ? "raw" : "filtered");
}

void CalibrationData::store_matches(
    const openMVG::matching::PairWiseMatches &matches, bool raw) {
  std::filesystem::create_directory(matches_directory);
  const auto pairs_path =
      matches_directory /
      std::format("matches_{}.txt", (raw) ? "raw" : "filtered");
  openMVG::matching::Save(matches, pairs_path);
}

std::optional<openMVG::matching::PairWiseMatches>
CalibrationData::load_matches(bool raw) {
  const auto pairs_path =
      matches_directory /
      std::format("matches_{}.txt", (raw) ? "raw" : "filtered");

  if (std::filesystem::exists(pairs_path)) {
    std::cout << "loading matches..." << std::endl;
    openMVG::matching::PairWiseMatches matches;
    openMVG::matching::Load(matches, pairs_path);
    return matches;
  } else {
    return std::nullopt;
  }
}

openMVG::sfm::SfM_Data cameras_to_sfm_data(
    const CameraSet &cameras,
    std::shared_ptr<openMVG::cameras::IntrinsicBase> intrinsics) {

  openMVG::sfm::SfM_Data sfm_data;
  sfm_data.intrinsics[0] = intrinsics;

  size_t pose_id = 0;
  for (const auto &camera : cameras.cameras) {
    camera->id_intrinsic = 0;
    sfm_data.views[camera->id_view] = camera;
    sfm_data.poses[pose_id] = openMVG::geometry::Pose3();
    camera->id_pose = pose_id;
    pose_id++;
  }

  return sfm_data;
}

openMVG::sfm::SfM_Data
group_to_sfm_data(const CameraSet &cameras, size_t group_id,
                  std::shared_ptr<openMVG::cameras::IntrinsicBase> intrinsics) {
  const auto &camera_ids = cameras.group_to_images.at(group_id);

  std::cout << "creating subset" << std::endl;

  openMVG::sfm::SfM_Data subset;
  subset.intrinsics[0] = intrinsics;

  size_t pose_id = 0;
  for (auto &view_id : camera_ids) {
    auto &camera = cameras.cameras.at(view_id);
    camera->id_intrinsic = 0;
    subset.poses[pose_id] = openMVG::geometry::Pose3(
        camera->pose.rotationMatrix(), camera->pose.translation());
    camera->id_pose = pose_id;
    subset.views[camera->id_view] = camera;
    pose_id++;
  };

  std::cout << "finished creating subset" << std::endl;

  return subset;
}

openMVG::sfm::SfM_Data create_sfm_data(
    const CameraSet &cameras,
    std::shared_ptr<openMVG::cameras::IntrinsicBase> intrinsics,
    std::shared_ptr<openMVG::sfm::Regions_Provider> regions_provider,
    const openMVG::tracks::STLMAPTracks &tracks) {
  openMVG::sfm::SfM_Data sfm_data;
  sfm_data.intrinsics[0] = intrinsics;

  // add all the cameras
  size_t pose_id = 0;
  for (const auto &camera : cameras.cameras) {
    camera->id_intrinsic = 0;
    sfm_data.views[camera->id_view] = camera;
    sfm_data.poses[pose_id] = openMVG::geometry::Pose3();
    camera->id_pose = pose_id;
    pose_id++;
  }

  // add all the track information
  auto &structure = sfm_data.structure;
  int idx(0);
  for (const auto &tracks_it : tracks) {
    structure[idx] = {};
    auto &obs = structure.at(idx).obs;
    for (const auto &track_it : tracks_it.second) {
      const auto imaIndex = track_it.first;
      const auto featIndex = track_it.second;
      const auto &pt =
          regions_provider->get(imaIndex)->GetRegionPosition(featIndex);
      obs[imaIndex] = {pt, featIndex};
    }
    ++idx;
  }

  return sfm_data;
}

openMVG::sfm::SfM_Data project_to_groups(const CameraSet &cameras,
                                         const openMVG::sfm::SfM_Data &sfm_data,
                                         std::vector<size_t> groups) {
  openMVG::sfm::SfM_Data projection;
  projection.intrinsics[0] = sfm_data.intrinsics.at(0);

  // get all the camera ids
  std::set<size_t> view_ids;
  for (const auto &group_id : groups) {
    const auto &group_views = cameras.group_to_images.at(group_id);
    view_ids.insert(group_views.begin(), group_views.end());
  }

  /// copy the views and poses over
  for (const auto &id_view : view_ids) {
    const auto &view = sfm_data.views.at(id_view);
    projection.views[id_view] = view;
    projection.poses[view->id_pose] = sfm_data.poses.at(view->id_pose);
  }

  // copy and prune the landmarks to only what is visible
  for (const auto &[id_landmark, landmark] : sfm_data.structure) {
    openMVG::sfm::Landmark new_landmark{};

    for (const auto &[observer_id, observation] : landmark.obs) {
      if (view_ids.contains(observer_id)) {
        new_landmark.obs[observer_id] = observation;
      }
    }

    if (!new_landmark.obs.empty()) {
      new_landmark.X = landmark.X;
      projection.structure[id_landmark] = new_landmark;
    }
  }

  return projection;
}

void update_sfm_data(openMVG::sfm::SfM_Data &sfm_data,
                     const openMVG::sfm::SfM_Data &projection) {
  // update view poses
  for (const auto &[id_view, view] : projection.views) {
    sfm_data.poses.at(view->id_pose) = projection.poses.at(view->id_pose);
  }

  // update landmark poses
  for (const auto &[id_landmark, landmark] : projection.structure) {
    sfm_data.structure.at(id_landmark).X = landmark.X;
  }
}

void initialize_poses_from_group(const CameraSet &camera_set,
                                 openMVG::sfm::SfM_Data &sfm_data) {
  for (const auto &[id, view] : sfm_data.views) {
    const Sophus::SE3d pose = camera_set.cameras.at(id)->pose;
    sfm_data.poses[view->id_pose] =
        openMVG::geometry::Pose3(pose.rotationMatrix(), pose.translation());
  }
}

void initialize_poses(const CameraSet &camera_set,
                      openMVG::sfm::SfM_Data &sfm_data) {
  const Sophus::SE3d T_base_v1 =
      camera_set.cameras
          .at(sfm_data.views.at((*sfm_data.views.begin()).first)->id_view)
          ->T_base_hand *
      T_hand_camera;

  const Sophus::SE3d t0 = camera_set.average_T_base_camera.at(0);

  for (const auto &[id, view] : sfm_data.views) {
    const auto &group_id = camera_set.image_to_group.at(id);

    const Sophus::SE3d pose =
        (camera_set.cameras.at(id)->T_base_hand * T_hand_camera);

    sfm_data.poses[view->id_pose] = se3_to_pose3(pose);
  }
}
