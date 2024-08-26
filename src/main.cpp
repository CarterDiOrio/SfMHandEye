#define OPENMVG_USE_OPENMP
#include "chessboardless/geometry.hpp"
#include "chessboardless/SfMPlyHelper.hpp"
#include "chessboardless/calibration_data.hpp"
#include "chessboardless/features.hpp"
#include <Eigen/Dense>
#include <cameras/Camera_Common.hpp>
#include <ceres/cost_function.h>
#include <ceres/types.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <multiview/solver_resection.hpp>
#include <multiview/translation_averaging_common.hpp>
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
#include <ranges>
#include <sfm/pipelines/sequential/SfmSceneInitializerMaxPair.hpp>
#include <sfm/pipelines/sequential/sequential_SfM.hpp>
#include <sfm/pipelines/sequential/sequential_SfM2.hpp>
#include <sfm/pipelines/sfm_engine.hpp>
#include <sfm/sfm_data_BA.hpp>
#include <sfm/sfm_data_BA_ceres.hpp>
#include <sfm/sfm_data_filters.hpp>
#include <sfm/sfm_landmark.hpp>
#include <sophus/se3.hpp>
#include <string_view>
#include <thread>
#include <types.hpp>
#include <sophus/average.hpp>

namespace ranges = std::ranges;

void GetCameraPositions(
  const openMVG::sfm::SfM_Data & sfm_data,
  std::vector<openMVG::Vec3> & vec_camPosition)
{
  for (const auto & view : sfm_data.GetViews()) {
    if (sfm_data.IsPoseAndIntrinsicDefined(view.second.get())) {
      const openMVG::geometry::Pose3 pose =
        sfm_data.GetPoseOrDie(view.second.get());
      vec_camPosition.push_back(pose.center());
    }
  }
}

int main(int argc, char ** argv)
{
  Eigen::Matrix3d intrinsics;
  intrinsics << std::stod(argv[2]), 0, std::stod(argv[3]), 0, std::stod(argv[4]),
    std::stod(argv[5]), 0, 0, 1;

  const auto intrinsics_openmvg = std::make_shared<openMVG::cameras::Pinhole_Intrinsic>(
    1920, 1080, std::stod(argv[2]), std::stod(argv[3]), std::stod(argv[5]));

  auto calibration_data = CalibrationData{argv[1]};
  auto camera_set = calibration_data.get_cameras();

  auto image_describer = create_image_describer("SIFT");
  image_describer->Set_configuration_preset(
    openMVG::features::EDESCRIBER_PRESET::ULTRA_PRESET);

  std::vector<Image> images;
  for (const auto & camera : camera_set.cameras) {
    images.push_back(camera->image);
  }

  auto pre_sfm_data =
    cameras_to_sfm_data(camera_set, intrinsics_openmvg);

  const auto regions = [&calibration_data, &image_describer, &pre_sfm_data]() {
    if (!calibration_data.has_features()) {
      auto r =
        describe_images(*image_describer, calibration_data.get_cameras());
      calibration_data.store_features(*image_describer, r);
    }

    auto region_provider = std::make_shared<openMVG::sfm::Regions_Provider>();
    auto type = image_describer->Allocate();

    openMVG::system::LoggerProgress progress_bar(
      calibration_data.get_cameras().cameras.size(), "- LOADING FEATURES -");

    region_provider->load(
      pre_sfm_data, calibration_data.get_feature_directory(),
      type, &progress_bar);
    return region_provider;
  }();

  const auto & camera = camera_set.cameras[0];

  openMVG::features::Features2SVG(
    camera->s_Img_path, {camera->image.Width(), camera->image.Height()},
    regions->get(0)->GetRegionsPositions(), "./features.svg");

  auto matcher =
    std::make_shared<openMVG::matching_image_collection::Matcher_Regions>(
    0.8f, openMVG::matching::CASCADE_HASHING_L2);

  auto raw_matches = [&calibration_data, &matcher, &regions, &pre_sfm_data]() {
    const auto maybe_matches = calibration_data.load_matches(true);
    if (maybe_matches.has_value()) {
      return maybe_matches.value();
    }
    const auto m = match_pairs(*matcher, pre_sfm_data, regions);
    calibration_data.store_matches(m, true);
    return m;
  }();

  std::cout << std::format("Processing {} matches\n", raw_matches.size());

  auto matches = [&calibration_data, &raw_matches, &regions, &pre_sfm_data]() {
    const auto maybe_matches = calibration_data.load_matches(false);
    if (maybe_matches.has_value()) {
      return maybe_matches.value();
    }
    auto filtered_matches = filter_matches(pre_sfm_data, regions, raw_matches);
    calibration_data.store_matches(filtered_matches, false);
    return filtered_matches;
  }();

  // -- export Putative View Graph statistics
  openMVG::graph::getGraphStatistics(
    pre_sfm_data.GetViews().size(),
    getPairs(matches));

  openMVG::sfm::SfM_Data sfm_data;
  if (!openMVG::sfm::Load(sfm_data, "./sfm.bin", openMVG::sfm::ESfM_Data::ALL)) {

    auto type = image_describer->Allocate();
    auto feature_provider = std::make_shared<openMVG::sfm::Features_Provider>();
    feature_provider->load(
      pre_sfm_data, calibration_data.get_feature_directory(),
      type);

    auto match_provider = std::make_shared<openMVG::sfm::Matches_Provider>();
    match_provider->load(pre_sfm_data, calibration_data.get_matches_path(false));

    auto scene_init = std::make_unique<openMVG::sfm::SfMSceneInitializerMaxPair>(
      pre_sfm_data, feature_provider.get(), match_provider.get());

    std::unique_ptr<openMVG::sfm::ReconstructionEngine> sfm_engine;
    openMVG::sfm::SequentialSfMReconstructionEngine * engine =
      new openMVG::sfm::SequentialSfMReconstructionEngine(
      pre_sfm_data, "./",
      "./report.html");

    engine->SetFeaturesProvider(feature_provider.get());
    engine->SetMatchesProvider(match_provider.get());
    engine->SetTriangulationMethod(
      openMVG::ETriangulationMethod::INVERSE_DEPTH_WEIGHTED_MIDPOINT);
    engine->SetResectionMethod(openMVG::resection::SolverType::DEFAULT);
    engine->SetUnknownCameraType(
      openMVG::cameras::EINTRINSIC(
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
    openMVG::sfm::Save(sfm_engine->Get_SfM_Data(), "./sfm.bin", openMVG::sfm::ESfM_Data::ALL);
    sfm_data = sfm_engine->Get_SfM_Data();
  }

  std::cout << "loaded" << std::endl;

  for (size_t idx = 0; idx < 10; idx++) {
    std::cout << sfm_data.poses.contains(sfm_data.views[idx]->id_pose) << std::endl;
  }

  // get the scale factor for the world using the first group
  // with the origin being the first camera in the first group
  // this is a very naive way of calculating a scale transform between the two systems
  const Sophus::SE3d T_world_cam0 =
    camera_set.cameras.at(camera_set.group_to_images.at(0).at(0))->pose;

  double average_true_translation = 0.0;
  double average_sfm_translation = 0.0;
  size_t count = 0;
  for (const auto & camera_id: camera_set.group_to_images.at(0)) {
    if (camera_id != 0) {

      // make sure it was a valid sfm pose
      if (sfm_data.poses.contains(camera_id)) {

        // get the ground truth translation
        const auto & camk = camera_set.cameras.at(camera_id);
        const Sophus::SE3d T_cam0_camk = T_world_cam0.inverse() * camk->pose;
        average_true_translation += T_cam0_camk.translation().norm();

        // get the average sfm translation
        const auto & T_sfm_cam0 =
          pose3_to_se3<double>(sfm_data.poses[sfm_data.views.at(0)->id_pose]);
        const auto & T_sfm_camk =
          pose3_to_se3<double>(sfm_data.poses[sfm_data.views.at(camera_id)->id_pose]);
        average_sfm_translation += (T_sfm_cam0.inverse() * T_sfm_camk).translation().norm();
        count++;
      }
    }
  }

  average_true_translation /= count;
  average_sfm_translation /= count;

  const double scale = average_true_translation / average_sfm_translation;

  std::cout << std::format("SfM to True Scale: {}\n", scale);

  // scale all poses translations by scale factor
  for (auto & [_, pose]: sfm_data.poses) {
    Sophus::SE3d pose_scaled_se3 = pose3_to_se3<double>(pose);
    pose_scaled_se3.translation() *= scale;
    auto pose_scaled = se3_to_pose3(pose_scaled_se3);
    pose.rotation() = pose_scaled.rotation();
    pose.center() = pose_scaled.center();
  }

  // triangulate to scale the points
  openMVG::sfm::SfM_Data_Structure_Computation_Robust est(
    4.0, 2, 2, openMVG::ETriangulationMethod::INVERSE_DEPTH_WEIGHTED_MIDPOINT, true);
  est.triangulate(sfm_data);

  std::vector<openMVG::Vec3> points, track_colors, camera_positions;
  if (openMVG::sfm::ColorizeTracks(sfm_data, points, track_colors)) {
    GetCameraPositions(sfm_data, camera_positions);
    openMVG::plyHelper::exportToPly(
      points, camera_positions, "./cloud.ply",
      &track_colors);
  }

  Sophus::SE3d T_hand_eye = calibrate_hand_eye(
    sfm_data,
    camera_set,
    T_hand_camera,
    intrinsics,
    *regions);

  calculate_relative_error(
    camera_set,
    sfm_data,
    T_hand_eye
  );

  // update all camera poses using T_hand_eye
  // find the robots location
  std::vector<Sophus::SE3d> T_world_bases;
  for (const auto & view_id: openMVG::sfm::Get_Valid_Views(sfm_data)) {
    const Sophus::SE3d T_world_cam =
      pose3_to_se3<double>(sfm_data.poses[sfm_data.views[view_id]->id_pose]);
    const Sophus::SE3d T_world_base = T_world_cam * T_hand_eye.inverse() *
      camera_set.cameras[view_id]->T_base_hand.inverse();
    T_world_bases.push_back(T_world_base);
  }
  const Sophus::SE3d T_world_base = *Sophus::average(T_world_bases);

  for (const auto & view_id: openMVG::sfm::Get_Valid_Views(sfm_data)) {
    const Sophus::SE3d T_world_cam = T_world_base * camera_set.cameras[view_id]->T_base_hand *
      T_hand_eye;
    sfm_data.poses[sfm_data.views[view_id]->id_pose] = se3_to_pose3(T_world_cam);
  }

  T_hand_eye = ba_hand_eye(
    sfm_data,
    camera_set,
    T_hand_eye,
    intrinsics,
    T_world_base,
    *regions
  );

  calculate_relative_error(
    camera_set,
    sfm_data,
    T_hand_eye
  );

  return 1;
}
