///
/// @file graph.hpp
/// @brief defines a feature graph for visual structure from motion
///

#ifndef INC_GUARD_GRAPH_HPP
#define INC_GUARD_GRAPH_HPP

#include "chessboardless/feature_detection.hpp"
#include <Eigen/Dense>
#include <concepts>
#include <cstddef>
#include <map>
#include <memory>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <optional>
#include <set>
#include <sophus/se3.hpp>
#include <span>
#include <unordered_map>

namespace slam {

struct ObservationEdge {
  /// @brief the feature detection of the point
  features::Feature feature;
};

struct Point {
  /// @brief the id of the frame that provided the best descriptor
  size_t best_frame_id;

  /// @brief the best descriptor for the point
  cv::Mat descriptor;

  /// @brief the level of the best descriptor
  int level;

  /// @brief the number of levels in the image pyramid
  size_t n_levels;

  /// @brief multiply this ratio by the detection distance of the descriptor
  /// to get the minimum detection distance
  double min_distance_ratio;

  /// @brief multiply this ratio by the detection distance to get the
  /// maximum detection distance
  double max_distance_ratio;

  /// @brief whether or not the point has been triangulated
  bool has_been_triangulated = false;

  /// @brief the 3d location of the point, only valid if has been triangulated
  /// is true
  Eigen::Vector3d location = Eigen::Vector3d::Zero();

  /// @brief Mean viewing direction
  Eigen::Vector3d viewing_direction = Eigen::Vector3d::Zero();
};

struct VisualFrame {
  VisualFrame(const Eigen::Matrix3d &intrinsics) : intrinsics(intrinsics) {}
  VisualFrame(const Eigen::Matrix3d &intrinsics, const Sophus::SE3d &pose)
      : intrinsics(intrinsics), has_pose(true), T_world_frame(pose) {}

  /// @brief the intrinsics of the camera
  Eigen::Matrix3d intrinsics;

  /// @brief whether or not the frame has a pose
  bool has_pose = false;

  /// @brief the pose of the camera
  Sophus::SE3d T_world_frame{Eigen::Matrix4d::Identity()};
};

/// @brief Models a feature graph between frames.
///
/// Contains all the visual information needed to perform sfm/reconstruction.
/// However this is not a pose structure. It just provides the functionality to
/// create a graph of features between frames.
///
/// Points will be automatically removed from the graph when no frame observes
/// them, since a point requires the keypoint from at least one from.
///
/// Frames will not be removed automatically.
///
/// Invariance 1, Point Invaraince:
/// 1) A point's descriptor will be chosen to minimize the distance to all the
/// other descriptors describing that point.
///
/// 2) A point's best frame id will be updated to match the frame that provided
/// the best descriptor.
///
/// 3) Max and Min distance ratios will be calculated using the best
/// descriptor's associated key point.
///
/// Invariance 2, Covisibility Graph:
/// 1) A covisibility graph will be maintained where the edges represent
/// connections between frames that share an observed point.
///
/// 2) The edge's value will be the number of shared observations.
///
class VisualSlamGraph {

public:
  /// @brief inserts a frame into the graph and returns its id
  /// @param frame a unqiue ptr to the frame, takes ownership
  /// @param id optionally specify the id directly.
  /// @return the id of the frame in the graph
  size_t insert_frame(std::unique_ptr<VisualFrame> frame,
                      std::optional<size_t> id = std::nullopt);

  /// @brief inserts a key point detection on a frame
  /// @post the point will ensure it has been updated to follow the invariance
  /// @param frame_id the id of the frame to link to
  /// @param feature the detected feature
  /// @param point_id a matched point to link to
  /// @return the id of the point representing the feature
  size_t insert_observation(size_t frame_id, const features::Feature &feature,
                            std::optional<size_t> point_id = std::nullopt);

  /// @brief Gets the frame from the graph
  /// @brief frame_id the id of the frame
  /// @return A const reference to the frame
  const VisualFrame &get_frame(size_t frame_id) const;
  VisualFrame &get_frame(size_t frame_id);

  /// @brief Gets the point from the graph
  /// @param point_id the id of the point
  /// @return A const reference to the point
  const Point &get_point(size_t point_id) const;
  Point &get_point(size_t point_id);

  /// @brief Gets the number of points in the graph
  /// @return the number of points;
  size_t num_points() const;

  /// @brief Gets the edge between the frame and observed point
  /// @param frame_id the id of the frame
  /// @param point_id the id of the point
  const ObservationEdge &get_observation(size_t frame_id,
                                         size_t point_id) const;

  /// @brief Gets ids of the observers of a point
  /// @param point_id the id of the point
  /// @return a vector of the ids of the observers
  const std::vector<size_t> &get_observers(size_t point_id) const;

  /// @brief Gets the ids of the points that the frame observes
  /// @param frame_id the id of the frame
  /// @return a span containing the ids of the points
  std::span<const size_t> get_frame_points(size_t frame_id) const;

  /// @brief Gets the ids of the frames that observe at least one point in
  /// common
  /// @param frame_id the frame to get the neighbors of
  /// @return a span containing the ids of the frames
  const std::vector<size_t> &get_covisibility_neighbors(size_t frame_id) const;

  /// @brief Gets the number of shared points between the two frames
  /// @param frame_id1 the first frame
  /// @param frame_id2 the second frame
  /// @return the number of shared observations
  size_t get_covisibility_count(size_t frame_id1, size_t frame_id2) const;

  /// @brief removes an observation of a point. If no observations of the point
  /// remain, the point is removed
  /// @param frame_id the id of the frame
  /// @param point_id the id of the point
  /// @returns true if the point was removed
  bool remove_observation(size_t frame_id, size_t point_id);

  /// @brief Removes observations of the point from the given frames
  /// @pre the point_id and frame_ids must exist within the graph
  /// @param frame_ids the ids of the frames we wants to remove the observation
  /// from
  /// @param point_id the observed point
  /// @return True if the point was removed due to have no more observers
  bool remove_observations(std::span<const size_t> frame_ids, size_t point_id);

  /// @brief removes a point from the graph
  /// @param point_id the id of the point to remove
  void remove_point(size_t point_id);

  /// @brief removes a frame from the graph
  /// @param the id of the frame to remove
  void remove_frame(size_t frame_id);

  /// @brief consolidates the two points into a single point with the id of the
  /// first Example Use Case: Two frames were inserted into the graph with their
  /// individual observed points, then matching was done and shared points need
  /// to be consolidated.
  /// @param point_id1 the first point to consolidate
  /// @param point_id2 the second point to consolidate
  void merge_points(size_t point_id1, size_t point_id2);

private:
  using adjacency_list = std::unordered_map<size_t, std::vector<size_t>>;

  enum class CovisibilityOperation { Added, Removed };

  /// @brief auto increment ids
  size_t frame_id_count = 0;
  size_t point_id_count = 100000;

  std::unordered_map<size_t, std::unique_ptr<VisualFrame>> frames;
  std::unordered_map<size_t, std::unique_ptr<Point>> points;

  adjacency_list frame_point_graph;

  /// @brief a map from a combination of [frame id, point id] to the edge
  /// between the two
  std::map<std::pair<size_t, size_t>, std::unique_ptr<ObservationEdge>> edges;

  /// @brief computed covisibility graph.
  /// Extra book keeping to make queries faster
  adjacency_list covisibility_graph;

  /// an order needs to be set for access, the smaller frame id goes first
  std::map<std::pair<size_t, size_t>, size_t> covisibility_edges;

  /// @brief Ensures the point invariance described in the class description is
  /// held
  /// @param the id of the point to update
  void update_point(size_t point_id);

  /// @brief maintains covisibility graph edges.
  /// If an observation was added we need to potentially add edges in the
  /// covisibility graph, and/or increase value in the edges. If an observation
  /// was removed, we need decrease edges value and if an edge value hits zero,
  /// remove it.
  /// @param frame_id the id of the frame
  /// @param point_id the id of the point
  /// @param operation whether or not the observation was added or removed
  void update_covisilbity_graph(size_t frame_id, size_t point_id,
                                CovisibilityOperation operation);

  /// @brief removes an observation without ensuring the point invariance
  /// This is for primarily batch updates
  /// @return whether or not the point was removed
  bool unsafe_remove_observation(size_t frame_id, size_t point_id,
                                 bool remove_point);

  /// @brief orders the ids to correctly query the graph
  std::pair<size_t, size_t> covisibility_pair(size_t frame_id1,
                                              size_t frame_id2) const;
};

/// @brief Get the neighbors of a frame that meet a covisibility threshold
/// @param feature_graph the feature graph
/// @param frame_id the id of the starting frame
/// @param covisibility_threshold the number of covisibile points required
/// @param max_depth the depth to limit the search to
/// @return The frame ids of the frames in the local covisibility graph
std::set<size_t> covisibility_query(const VisualSlamGraph &feature_graph,
                                    size_t frame_id,
                                    size_t covisibility_threshold,
                                    size_t max_depth);

/// @brief Assembles a keypoint vector and descriptor mat for all of a frame's
/// observations Was made for easy of use with opencv's matching functions
/// @param feature_graph the feature graph containing the frame
/// @param frame_id the id of the frame
/// @return a pair of the keypoints and a mat of the descriptors
std::pair<std::vector<cv::KeyPoint>, cv::Mat>
get_frame_keypoints(const VisualSlamGraph &feature_graph, size_t frame_id);

/// @brief Gets all the key point measurements of a point from the given frames
/// @param graph the graph containing the point
/// @param frame_ids the ids of the observing frames
/// @param point id the id of the point
std::vector<cv::KeyPoint>
get_all_measurements(const VisualSlamGraph &graph,
                     std::span<size_t const> frame_ids, size_t point_id);

/// @brief Gets the shared points between two frames
/// @param graph the graph containing the frames
/// @param frame_id1 the id of the first frame
/// @param frame_id2 the id of the second frame
std::vector<size_t> get_shared_points(const VisualSlamGraph &graph,
                                      size_t frame_id1, size_t frame_id2);

} // namespace slam

#endif
