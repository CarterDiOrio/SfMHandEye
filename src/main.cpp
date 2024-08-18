#include <multiview/translation_averaging_common.hpp>
#include <sfm/sfm_data_filters.hpp>
#include <sfm/sfm_landmark.hpp>
#include <types.hpp>
#define OPENMVG_USE_OPENMP

#include "chessboardless/SfMPlyHelper.hpp"
#include "chessboardless/calibration_data.hpp"
#include "chessboardless/features.hpp"
#include <Eigen/Dense>
#include <cameras/Camera_Common.hpp>
#include <ceres/cost_function.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <multiview/solver_resection.hpp>
#include <multiview/triangulation_method.hpp>
#include <nlohmann/json.hpp>
#include <numeric>
#include <openMVG/cameras/Camera_Intrinsics.hpp>
#include <openMVG/features/svg_features.hpp>
#include <openMVG/geometry/pose3.hpp>
#include <openMVG/graph/graph.hpp>
#include <openMVG/graph/graph_graphviz_export.hpp>
#include <openMVG/graph/graph_stats.hpp>
#include <openMVG/matching/regions_matcher.hpp>
#include <openMVG/matching/svg_matches.hpp>
#include <openMVG/multiview/triangulation_nview.hpp>
#include <openMVG/sfm/pipelines/sequential/SfmSceneInitializer.hpp>
#include <openMVG/sfm/pipelines/sfm_engine.hpp>
#include <openMVG/sfm/pipelines/sfm_features_provider.hpp>
#include <openMVG/sfm/pipelines/sfm_matches_provider.hpp>
#include <openMVG/sfm/pipelines/sfm_regions_provider.hpp>
#include <openMVG/sfm/pipelines/sfm_regions_provider_cache.hpp>
#include <openMVG/sfm/pipelines/structure_from_known_poses/structure_estimator.hpp>
#include <openMVG/sfm/sfm.hpp>
#include <openMVG/sfm/sfm_data.hpp>
#include <openMVG/sfm/sfm_data_colorization.hpp>
#include <openMVG/sfm/sfm_data_io.hpp>
#include <openMVG/sfm/sfm_data_io_ply.hpp>
#include <openMVG/sfm/sfm_data_triangulation.hpp>
#include <openMVG/stl/stl.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <ranges>
#include <sfm/pipelines/sequential/SfmSceneInitializerMaxPair.hpp>
#include <sfm/pipelines/sequential/sequential_SfM.hpp>
#include <sfm/pipelines/sequential/sequential_SfM2.hpp>
#include <sfm/pipelines/sfm_engine.hpp>
#include <sfm/sfm_data_BA.hpp>
#include <sophus/se3.hpp>
#include <string_view>
#include <thread>

namespace ranges = std::ranges;

void GetCameraPositions(const openMVG::sfm::SfM_Data &sfm_data,
                        std::vector<openMVG::Vec3> &vec_camPosition) {
  for (const auto &view : sfm_data.GetViews()) {
    if (sfm_data.IsPoseAndIntrinsicDefined(view.second.get())) {
      const openMVG::geometry::Pose3 pose =
          sfm_data.GetPoseOrDie(view.second.get());
      vec_camPosition.push_back(pose.center());
    }
  }
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

  auto raw_matches = [&calibration_data, &matcher, &regions, &sfm_data]() {
    const auto maybe_matches = calibration_data.load_matches(true);
    if (maybe_matches.has_value()) {
      return maybe_matches.value();
    }
    const auto m = match_pairs(*matcher, sfm_data, regions);
    calibration_data.store_matches(m, true);
    return m;
  }();

  std::cout << std::format("Processing {} matches\n", raw_matches.size());

  auto matches = [&calibration_data, &raw_matches, &regions, &sfm_data]() {
    const auto maybe_matches = calibration_data.load_matches(false);
    if (maybe_matches.has_value()) {
      return maybe_matches.value();
    }
    auto filtered_matches = filter_matches(sfm_data, regions, raw_matches);
    calibration_data.store_matches(filtered_matches, false);
    return filtered_matches;
  }();

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

  // const auto tracks = create_feature_tracks(matches);

  // auto group =
  //     group_to_sfm_data(camera_set, 0, calibration_data.get_intrinsics());

  // auto valid_ids = openMVG::sfm::Get_Valid_Views(group);

  // auto &structure = group.structure;
  // int idx(0);
  // for (const auto &tracks_it : tracks) {
  //   structure[idx] = {};
  //   auto &obs = structure.at(idx).obs;
  //   for (const auto &track_it : tracks_it.second) {
  //     const auto imaIndex = track_it.first;
  //     const auto featIndex = track_it.second;
  //     const auto &pt = regions->get(imaIndex)->GetRegionPosition(featIndex);

  //     if (valid_ids.contains(imaIndex)) {
  //       obs[imaIndex] = {pt, featIndex};
  //     }
  //   }

  //   if (obs.empty()) {
  //     structure.erase(idx);
  //   }

  //   ++idx;
  // }

  // openMVG::sfm::SfM_Data_Structure_Computation_Robust est(
  //     4.0, 3, 3,
  //     openMVG::ETriangulationMethod::INVERSE_DEPTH_WEIGHTED_MIDPOINT, true);
  // est.triangulate(group);

  auto type = image_describer->Allocate();
  auto feature_provider = std::make_shared<openMVG::sfm::Features_Provider>();
  feature_provider->load(sfm_data, calibration_data.get_feature_directory(),
                         type);

  auto match_provider = std::make_shared<openMVG::sfm::Matches_Provider>();
  match_provider->load(sfm_data, calibration_data.get_matches_path(false));

  auto scene_init = std::make_unique<openMVG::sfm::SfMSceneInitializerMaxPair>(
      sfm_data, feature_provider.get(), match_provider.get());

  std::unique_ptr<openMVG::sfm::ReconstructionEngine> sfm_engine;
  openMVG::sfm::SequentialSfMReconstructionEngine *engine =
      new openMVG::sfm::SequentialSfMReconstructionEngine(sfm_data, "./",
                                                          "./report.html");

  engine->SetFeaturesProvider(feature_provider.get());
  engine->SetMatchesProvider(match_provider.get());
  engine->SetTriangulationMethod(
      openMVG::ETriangulationMethod::INVERSE_DEPTH_WEIGHTED_MIDPOINT);
  engine->SetResectionMethod(openMVG::resection::SolverType::DEFAULT);
  engine->SetUnknownCameraType(openMVG::cameras::EINTRINSIC(
      openMVG::cameras::EINTRINSIC::PINHOLE_CAMERA));
  // engine->setInitialPair({0, 9});
  sfm_engine.reset(engine);

  sfm_engine->Set_Intrinsics_Refinement_Type(
      openMVG::cameras::Intrinsic_Parameter_Type::NONE);
  sfm_engine->Set_Extrinsics_Refinement_Type(
      openMVG::sfm::Extrinsic_Parameter_Type::ADJUST_ALL);
  sfm_engine->Set_Use_Motion_Prior(false);

  std::cout << sfm_engine->Process() << std::endl;

  Generate_SfM_Report(
      sfm_engine->Get_SfM_Data(),
      stlplus::create_filespec("./", "SfMReconstruction_Report.html"));

  std::vector<openMVG::Vec3> points, track_colors, camera_positions;
  if (openMVG::sfm::ColorizeTracks(sfm_data, points, track_colors)) {
    GetCameraPositions(sfm_data, camera_positions);
    openMVG::plyHelper::exportToPly(points, camera_positions, "./cloud.ply");
  }

  return 1;
}