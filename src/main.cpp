#include "chessboardless/calibration_data.hpp"
#include "chessboardless/features.hpp"
#include <Eigen/Dense>
#include <ceres/cost_function.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <openMVG/features/svg_features.hpp>
#include <openMVG/geometry/pose3.hpp>
#include <openMVG/sfm/pipelines/sfm_regions_provider_cache.hpp>
#include <openMVG/sfm/pipelines/structure_from_known_poses/structure_estimator.hpp>
#include <openMVG/sfm/sfm.hpp>
#include <openMVG/sfm/sfm_data.hpp>
#include <openMVG/sfm/sfm_data_io.hpp>
#include <openMVG/sfm/sfm_data_triangulation.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <ranges>
#include <sfm/pipelines/sfm_engine.hpp>
#include <sophus/se3.hpp>
#include <string_view>
#include <thread>

namespace ranges = std::ranges;

void display(const std::string_view name, cv::Mat img) {
  cv::Mat smaller;
  cv::resize(img, smaller, cv::Size(-1, -1), 0.5, 0.5);
  cv::imshow(std::string{name}, smaller);
}

int main(int argc, char **argv) {

  Eigen::Matrix3d intrinsics;
  intrinsics << 1381.17626953125, 0, 973.329956054688, 0, 1381.80151367188,
      532.698852539062, 0, 0, 1;

  auto calibration_data = CalibrationData{argv[1]};
  const auto camera_set = calibration_data.load_cameras();

  auto image_describer = create_image_describer("SIFT");
  image_describer->Set_configuration_preset(
      openMVG::features::EDESCRIBER_PRESET::ULTRA_PRESET);

  std::vector<Image> images;
  for (const auto &camera : camera_set.cameras) {
    images.push_back(camera.image);
  }

  const auto regions = [&calibration_data, &image_describer]() {
    if (calibration_data.has_features()) {
      std::cout << "Loading regions from disk" << std::endl;
      return calibration_data.load_features(*image_describer);
    } else {
      auto r =
          describe_images(*image_describer, calibration_data.get_cameras());
      calibration_data.store_features(*image_describer, r);
      return r;
    }
  }();

  match_pairs(regions);

  // openMVG::sfm::ReconstructionEngine;

  // const auto &camera = camera_set.cameras[10];

  // openMVG::features::Features2SVG(
  //     camera.image_path, {camera.image.Width(), camera.image.Height()},
  //     regions[10]->GetRegionsPositions(), "./features.svg");

  return 1;
}