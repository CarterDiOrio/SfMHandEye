#include "chessboardless/calibration_data.hpp"

#include <Eigen/Dense>
#include <Eigen/src/Geometry/AngleAxis.h>
#include <algorithm>
#include <execution>
#include <features/regions.hpp>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <numeric>
#include <openMVG/image/image_io.hpp>
#include <openMVG/system/loggerprogress.hpp>
#include <opencv2/imgcodecs.hpp>
#include <ranges>
#include <sophus/average.hpp>
#include <stdexcept>

static constexpr double deg2rad = M_PI / 180;

CalibrationData::CalibrationData(const std::string_view &data_path)
    : data_directory{CalibrationData::validate_data_directory(data_path)},
      feature_directory{data_directory / "features"} {}

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
    std::vector<Camera> cameras(camera_group_json.cameras.size());

    // load in images and convert hand poses to SE3
    for (const auto &[idx, camera_json] :
         std::ranges::views::enumerate(camera_group_json.cameras)) {
      std::cout << std::format("Loading Image: {}\n", idx);

      const auto image_filepath = data_directory / camera_json.image;
      if (!std::filesystem::exists(image_filepath)) {
        throw std::invalid_argument(
            std::format("{} image does not exist. Malformed data file",
                        image_filepath.string()));
      }
      openMVG::image::ReadImage(image_filepath.c_str(), &cameras[idx].image);

      const Eigen::AngleAxisd x_rot{camera_json.pose[3] * deg2rad,
                                    Eigen::Vector3d::UnitX()};

      const Eigen::AngleAxisd y_rot{camera_json.pose[4] * deg2rad,
                                    Eigen::Vector3d::UnitY()};

      const Eigen::AngleAxisd z_rot{camera_json.pose[5] * deg2rad,
                                    Eigen::Vector3d::UnitZ()};

      const Sophus::SO3d rotation{(z_rot * y_rot * x_rot).toRotationMatrix()};
      const Eigen::Vector3d translation{
          camera_json.pose[0], camera_json.pose[1], camera_json.pose[2]};
      const Sophus::SE3d T_base_hand{rotation, translation};

      cameras[idx].T_base_hand = Sophus::SE3d(rotation, translation);
      cameras[idx].image_path = image_filepath;
      cameras[idx].group_id = group_id;
      cameras[idx].id = camera_id++;

      camera_set.image_to_group[camera_id] = group_id;
      camera_set.group_to_images[group_id].push_back(camera_id);
    }

    // get the mean hand pose of the group
    std::vector<Sophus::SE3d> hand_poses;
    for (const auto &camera : cameras) {
      hand_poses.push_back(camera.T_base_hand);
    };

    Sophus::SE3d T_base_hand = *Sophus::average(hand_poses);

    // add camera poses relative to the mean transform
    for (auto &camera : cameras) {
      camera.pose = T_base_hand.inverse() * camera.T_base_hand;
    }

    // add cameras to camera set
    camera_set.cameras.insert(camera_set.cameras.end(), cameras.begin(),
                              cameras.end());
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
        camera_set.cameras[std::get<0>(pair)].image_path};
    const auto name = img_path.stem();
    const auto ext = img_path.extension();

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

std::vector<RegionsPtr> CalibrationData::load_features(
    const openMVG::features::Image_describer &describer) {
  openMVG::system::LoggerProgress progress_bar(camera_set.cameras.size(),
                                               "- LOADING FEATURES -");

  const auto load_feature = [this, &describer,
                             &progress_bar](const Camera &camera) {
    const std::filesystem::path img_path{camera.image_path};
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
