#ifndef INC_GUARD_REPOREJECTION_COST_FUNCTION_HPP
#define INC_GUARD_REPOREJECTION_COST_FUNCTION_HPP

#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#include <ceres/cost_function.h>
#include <sophus/ceres_manifold.hpp>
#include <sophus/se3.hpp>

/// @brief Cost function for optimizing reprojection error based on hand eye
/// transform and structure A basic/naive cost function for doing SfM based hand
/// eye calibration:
///
/// Assumptions:
///
/// 1) The camera is pinhole.
///
/// 2) The hand position is absolute, does not model hand error in any way.
///
/// Does not assume:
/// - Anything about the relationship between camera poses
///     -> for example if your robot is accurate/precise enough (like the
///     Mecha500 or some linear stage system) within each group of only
///     translation you know the position of each camera relative to others
///     within the group as well as your arm can report the translation. This
///     could be used to provide further constraints.
struct BasicHandEyeCostFunction {

  BasicHandEyeCostFunction(const Eigen::Matrix3d &K,
                           const Eigen::Vector2d &image_point,
                           const Sophus::SE3d &T_base_hand)
      : K{K}, image_point{image_point}, T_base_hand{T_base_hand} {}

  /// @brief The operator() function for the cost function
  template <typename T>
  bool operator()(const T *world_point_param, 
                  const T *T_hand_eye_param,
                  const T *T_world_base_param,
                  T *residuals) const {
    // map the world point to a vector
    Eigen::Vector3<T> world_point =
        Eigen::Map<Eigen::Vector3<T> const>{world_point_param};

    // map the input se3 manifold to a Sophus type
    Sophus::SE3<T> T_hand_eye =
        Eigen::Map<Sophus::SE3<T> const>{T_hand_eye_param};
    Sophus::SE3<T> T_world_base =
        Eigen::Map<Sophus::SE3<T> const>{T_world_base_param};

    Sophus::SE3<T> T_eye_world = (T_world_base * T_base_hand * T_hand_eye).inverse();

    // transform the point to the eye frame
    Eigen::Vector<T, 4> point_eye = T_eye_world * world_point.homogeneous();
    // project the point to the image plane
    const auto x = K(0, 0) * point_eye.x() / point_eye.z() + K(0, 2);
    const auto y = K(1, 1) * point_eye.y() / point_eye.z() + K(1, 2);

    // calculate the residuals
    residuals[0] = x - T{image_point.x()};
    residuals[1] = y - T{image_point.y()};

    residuals[0] *= residuals[0];
    residuals[1] *= residuals[1];

    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Matrix3d &K,
                                     const Eigen::Vector2d &image_point,
                                     const Sophus::SE3d &T_base_hand) {
    return new ceres::AutoDiffCostFunction<
        BasicHandEyeCostFunction, 2, 3,
        Sophus::Manifold<Sophus::SE3>::num_parameters,
        Sophus::Manifold<Sophus::SE3>::num_parameters>(
        new BasicHandEyeCostFunction(K, image_point, T_base_hand));
  }

  const Eigen::Matrix3d K;
  const Eigen::Vector2d image_point;
  const Sophus::SE3d T_base_hand;
};

struct GroupToGroupCostFunction {

  GroupToGroupCostFunction(const Eigen::Matrix3d &K,
                           const Eigen::Vector2d &image_point,
                           const Sophus::SE3d &T_group_camera,
                           double scale)
      : K{K}, image_point{image_point}, T_group_camera{T_group_camera}, scale{scale} {}

  /// @brief The operator() function for the cost function
  template <typename T>
  bool operator()(const T *world_point_param, const T *T_world_group_param,
                  T *residuals) const {
    // map the world point to a vector
    Eigen::Vector3<T> world_point =
        Eigen::Map<Eigen::Vector3<T> const>{world_point_param};

    // map the input se3 manifold to a Sophus type
    Sophus::SE3<T> T_world_group =
        Eigen::Map<Sophus::SE3<T> const>{T_world_group_param};

    Sophus::SE3<T> T_eye_world = (T_world_group * T_group_camera).inverse();

    // transform the point to the eye frame
    Eigen::Vector<T, 4> point_eye = T_eye_world * world_point.homogeneous();
    // project the point to the image plane
    const auto x = K(0, 0) * point_eye.x() / point_eye.z() + K(0, 2);
    const auto y = K(1, 1) * point_eye.y() / point_eye.z() + K(1, 2);

    // calculate the residuals
    auto x_err = x - T{image_point.x()};
    auto y_err = y - T{image_point.y()}; 

    residuals[0] = (x_err * x_err) * ( 1 / (scale * scale));
    residuals[1] = (y_err * y_err) * ( 1 / (scale * scale));

    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Matrix3d &K,
                                     const Eigen::Vector2d &image_point,
                                     const Sophus::SE3d &T_group_camera,
                                     const double scale) {
    return new ceres::AutoDiffCostFunction<
        GroupToGroupCostFunction, 2, 3,
        Sophus::Manifold<Sophus::SE3>::num_parameters>(
        new GroupToGroupCostFunction(K, image_point, T_group_camera, scale));
  }

  const Eigen::Matrix3d K;
  const Eigen::Vector2d image_point;
  const Sophus::SE3d T_group_camera;
  const double scale;
};

struct HandEyeCostFunction {
  HandEyeCostFunction(
    const Sophus::SE3d& T_sfm_eye1,
    const Sophus::SE3d& T_sfm_eye2,
    const Sophus::SE3d& T_base_hand1,
    const Sophus::SE3d& T_base_hand2
  ): T_sfm_eye1{T_sfm_eye1}, T_sfm_eye2{T_sfm_eye2}, T_base_hand1{T_base_hand1}, T_base_hand2{T_base_hand2} {}
  
  template<typename T>
  bool operator()(const T* T_hand_eye_param, T* residuals_ptr) const {

    Sophus::SE3<T> T_hand_eye =
        Eigen::Map<Sophus::SE3<T> const>{T_hand_eye_param};

    const Sophus::SE3<T> T_eye1_eye2 = (T_sfm_eye1.inverse() * T_sfm_eye2).cast<T>();
    const Sophus::SE3<T> T_hand1_hand2 = (T_base_hand1.inverse() * T_base_hand2).cast<T>();

    const Sophus::SE3<T> T_hand1_eye2 = T_hand_eye * T_eye1_eye2;
    const Sophus::SE3<T> T_hand1_eye2_p = T_hand1_hand2 * T_hand_eye;

    const Sophus::SE3<T> T_err = T_hand1_eye2.inverse() * T_hand1_eye2_p;

    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals{residuals_ptr};
    residuals = T_err.log();

    return true;
  }

  static ceres::CostFunction *Create(const Sophus::SE3d& T_sfm_eye1,
    const Sophus::SE3d& T_sfm_eye2,
    const Sophus::SE3d& T_base_hand1,
    const Sophus::SE3d& T_base_hand2) {
      return new ceres::AutoDiffCostFunction<
        HandEyeCostFunction, 6, Sophus::Manifold<Sophus::SE3>::num_parameters>(
          new HandEyeCostFunction(T_sfm_eye1, T_sfm_eye2, T_base_hand1, T_base_hand2));

    }

  const Sophus::SE3d T_sfm_eye1;
  const Sophus::SE3d T_sfm_eye2;
  const Sophus::SE3d T_base_hand1;
  const Sophus::SE3d T_base_hand2;
};


#endif
