#include "chessboardless/geometry.hpp"
#include "chessboardless/reprojection_cost_functions.hpp"
#include <ceres/ceres.h>
#include <ceres/loss_function.h>
#include <ceres/manifold.h>
#include <cstddef>
#include <format>
#include <sfm/sfm_data.hpp>
#include <sfm/sfm_data_BA.hpp>
#include <sfm/sfm_data_BA_ceres.hpp>
#include <sfm/sfm_data_filters.hpp>
#include <sfm/sfm_data_triangulation.hpp>
#include <sophus/ceres_manifold.hpp>
#include <sophus/se3.hpp>
#include <unordered_map>

Sophus::SE3d ba_hand_eye(
  openMVG::sfm::SfM_Data & sfm_data,
  CameraSet & camera_set,
  const Sophus::SE3d & initial_guess, const Eigen::Matrix3d K,
  const Sophus::SE3d & T_world_base_guess,
  const openMVG::sfm::Regions_Provider & regions_provider)
{
  Sophus::SE3d T_hand_eye = initial_guess;
  Sophus::SE3d T_world_base = T_world_base_guess;

  const auto valid_view_ids = openMVG::sfm::Get_Valid_Views(sfm_data);

  ceres::Problem problem;
  auto parameterization = new Sophus::Manifold<Sophus::SE3>;

  problem.AddParameterBlock(
    T_hand_eye.data(), Sophus::SE3d::num_parameters,
    parameterization);
  problem.AddParameterBlock(
    T_world_base.data(), Sophus::SE3d::num_parameters,
    parameterization);


  auto loss = new ceres::HuberLoss(1.0);
  for (auto &[landmark_id, landmark] : sfm_data.structure) {
    problem.AddParameterBlock(landmark.X.data(), 3, nullptr);

    for (const auto &[obs_id, observation] : landmark.obs) {

      //  const auto regions = regions_provider.get(camera_id);
      //   const auto* sio_regions = dynamic_cast<openMVG::features::SIFT_Regions*>(regions.get());
      //   const auto& feature = sio_regions->Features().at(observation.id_feat);
      //   const auto scale = feature.scale()


      // for each observation we add a reprojection cost function
      problem.AddResidualBlock(
        BasicHandEyeCostFunction::Create(
          K, observation.x, camera_set.cameras.at(obs_id)->T_base_hand),
        loss, landmark.X.data(), T_hand_eye.data(), T_world_base.data());
    }
  }

  // setup the solver
  ceres::Solver::Options ceres_solver_options;
  ceres_solver_options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
  ceres_solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  ceres_solver_options.minimizer_progress_to_stdout = true;
  ceres_solver_options.max_num_iterations = 100;
  ceres::Solver::Summary summary;
  ceres::Solve(ceres_solver_options, &problem, &summary);

  std::cout << " Initial RMSE: "
            << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
            << " Final RMSE: "
            << std::sqrt(summary.final_cost / summary.num_residuals) << "\n";

  // update the sfm with the new camera poses
  // and update the camera data
  for (const auto & view_id: valid_view_ids) {
    sfm_data.poses[sfm_data.views[view_id]->id_pose] = se3_to_pose3(
      T_world_base *
      camera_set.cameras[view_id]->T_base_hand * T_hand_eye);
  }

  std::cout << T_hand_eye.matrix() << std::endl;

  return T_hand_eye;
}


Sophus::SE3d calibrate_hand_eye(
  openMVG::sfm::SfM_Data & sfm_data,
  CameraSet & camera_set,
  Sophus::SE3d initial_guess, Eigen::Matrix3d K,
  const openMVG::sfm::Regions_Provider & regions_provider)
{
  Sophus::SE3d T_hand_eye = initial_guess;

  // find all camera ids that are valid within the sfm data
  std::set<size_t> valid_camera_ids;
  for (const auto & [view_id, view]: sfm_data.views) {
    if (sfm_data.poses.contains(view->id_pose)) {
      valid_camera_ids.insert(view_id);
    }
  }

  ceres::Problem problem;
  auto parameterization = new Sophus::Manifold<Sophus::SE3>;

  problem.AddParameterBlock(
    T_hand_eye.data(), Sophus::SE3d::num_parameters,
    parameterization);


  // for every pair of valid ids
  for (const auto & camera_id1: valid_camera_ids) {
    for (const auto & camera_id2: valid_camera_ids) {
      if (camera_id1 != camera_id2) {

        problem.AddResidualBlock(
          HandEyeCostFunction::Create(
            pose3_to_se3<double>(sfm_data.poses[sfm_data.views[camera_id1]->id_pose]),
            pose3_to_se3<double>(sfm_data.poses[sfm_data.views[camera_id2]->id_pose]),
            camera_set.cameras[camera_id1]->T_base_hand,
            camera_set.cameras[camera_id2]->T_base_hand
          ),
          nullptr,
          T_hand_eye.data()
        );
      }
    }
  }


  // setup the solver
  ceres::Solver::Options ceres_solver_options;
  ceres_solver_options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
  ceres_solver_options.linear_solver_type = ceres::DENSE_QR;
  ceres_solver_options.minimizer_progress_to_stdout = true;
  ceres_solver_options.max_num_iterations = 1000;
  ceres::Solver::Summary summary;
  ceres::Solve(ceres_solver_options, &problem, &summary);

  std::cout << " Initial RMSE: "
            << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
            << " Final RMSE: "
            << std::sqrt(summary.final_cost / summary.num_residuals) << "\n";


  std::cout << T_hand_eye.matrix() << std::endl;

  return T_hand_eye;
}

void calculate_relative_error(
  const CameraSet & camera_set,
  const openMVG::sfm::SfM_Data & sfm_data,
  const Sophus::SE3d & T_hand_eye)
{

  const auto valid_view_ids = openMVG::sfm::Get_Valid_Views(sfm_data);


  size_t count = 0;
  double translation = 0.0;
  for (const auto & view_id1: valid_view_ids) {
    for (const auto & view_id2: valid_view_ids) {

      if (view_id1 != view_id2) {

        // transform in sfm frame
        const Sophus::SE3d T_sfm_eye1 =
          pose3_to_se3<double>(sfm_data.poses.at(sfm_data.views.at(view_id1)->id_pose));
        const Sophus::SE3d T_sfm_eye2 =
          pose3_to_se3<double>(sfm_data.poses.at(sfm_data.views.at(view_id2)->id_pose));
        const Sophus::SE3d T_eye1_eye2 = T_sfm_eye1.inverse() * T_sfm_eye2;


        // calculate transform in hand frame
        const Sophus::SE3d T_base_hand1 = camera_set.cameras.at(view_id1)->T_base_hand;
        const Sophus::SE3d T_base_hand2 = camera_set.cameras.at(view_id2)->T_base_hand;
        const Sophus::SE3d T_hand1_hand2 = T_base_hand1.inverse() * T_base_hand2;

        // calculate error AX = XB
        const Sophus::SE3d T_eye1_hand2 = T_eye1_eye2 * T_hand_eye.inverse();
        const Sophus::SE3d T_eye1_hand2_hat = T_hand_eye.inverse() * T_hand1_hand2;

        // should be the identity matrix if no error
        const Sophus::SE3d T_err = T_eye1_hand2.inverse() * T_eye1_hand2_hat;

        translation += T_err.translation().norm();
        count++;
      }
    }
  }

  translation /= count;
  std::cout << std::format("Mean Translation Error (mm): {}\n", translation);
}
