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
#include <openMVG/graph/graph.hpp>
#include <openMVG/graph/graph_graphviz_export.hpp>
#include <openMVG/graph/graph_stats.hpp>
#include <openMVG/matching/regions_matcher.hpp>
#include <openMVG/matching/svg_matches.hpp>
#include <openMVG/sfm/pipelines/sfm_regions_provider.hpp>
#include <openMVG/sfm/pipelines/sfm_regions_provider_cache.hpp>
#include <openMVG/sfm/pipelines/structure_from_known_poses/structure_estimator.hpp>
#include <openMVG/sfm/sfm.hpp>
#include <openMVG/sfm/sfm_data.hpp>
#include <openMVG/sfm/sfm_data_io.hpp>
#include <openMVG/sfm/sfm_data_triangulation.hpp>
#include <openMVG/stl/stl.hpp>
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
  const auto camera_set = calibration_data.get_cameras();

  auto image_describer = create_image_describer("SIFT");
  image_describer->Set_configuration_preset(
      openMVG::features::EDESCRIBER_PRESET::ULTRA_PRESET);

  std::vector<Image> images;
  for (const auto &camera : camera_set.cameras) {
    images.push_back(camera->image);
  }

  auto sfm_data =
      cameras_to_sfm_data(camera_set, calibration_data.get_intrinsics());

  const auto regions = [&calibration_data, &image_describer, &sfm_data]() {
    if (!calibration_data.has_features()) {
      auto r =
          describe_images(*image_describer, calibration_data.get_cameras());
      calibration_data.store_features(*image_describer, r);
    }

    auto region_provider = std::make_shared<openMVG::sfm::Regions_Provider>();
    auto type = image_describer->Allocate();

    openMVG::system::LoggerProgress progress_bar(
        calibration_data.get_cameras().cameras.size(), "- LOADING FEATURES -");

    region_provider->load(sfm_data, calibration_data.get_feature_directory(),
                          type, &progress_bar);
    return region_provider;
  }();

  // openMVG::sfm::ReconstructionEngine;

  const auto &camera = camera_set.cameras[0];
  const auto &camera2 = camera_set.cameras[329];

  openMVG::features::Features2SVG(
      camera->s_Img_path, {camera->image.Width(), camera->image.Height()},
      regions->get(0)->GetRegionsPositions(), "./features.svg");

  auto matcher =
      std::make_shared<openMVG::matching_image_collection::Matcher_Regions>(
          0.8f, openMVG::matching::CASCADE_HASHING_L2);

  const auto matches = [&calibration_data, &matcher, &regions, &sfm_data]() {
    const auto maybe_matches = calibration_data.load_matches(true);
    if (maybe_matches.has_value()) {
      return maybe_matches.value();
    }
    const auto m = match_pairs(*matcher, sfm_data, regions);
    calibration_data.store_matches(m, true);
    return m;
  }();

  auto pairs = openMVG::matching::getPairs(matches);

  // -- export Putative View Graph statistics
  openMVG::graph::getGraphStatistics(sfm_data.GetViews().size(),
                                     getPairs(matches));

  //-- export view pair graph once putative graph matches has been computed
  {
    std::set<openMVG::IndexT> set_ViewIds;
    std::transform(sfm_data.GetViews().begin(), sfm_data.GetViews().end(),
                   std::inserter(set_ViewIds, set_ViewIds.begin()),
                   stl::RetrieveKey());
    openMVG::graph::indexedGraph putativeGraph(set_ViewIds, getPairs(matches));
    openMVG::graph::exportToGraphvizData("./graph.svg", putativeGraph);
  }

  {
    openMVG::matching::Matches2SVG(
        camera->s_Img_path, {camera->image.Width(), camera->image.Height()},
        regions->get(camera->id_view)->GetRegionsPositions(),
        camera2->s_Img_path, {camera2->image.Width(), camera2->image.Height()},
        regions->get(camera2->id_view)->GetRegionsPositions(),
        matches.at({camera->id_view, camera2->id_view}), "./matches.svg", true);
  }

  return 1;
}