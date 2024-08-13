#ifndef INC_GUARD_RECONSTRUCTION_HPP
#define INC_GUARD_RECONSTRUCTION_HPP

#include "chessboardless/graph.hpp"
#include <sophus/se3.hpp>

namespace slam {

/// @brief Performs 2D-2D motion estmiation between two frames using the
/// essential matrix
/// @param graph the graph containing the two frames with matched features
/// @param frame_id1 the id of the first frame
/// @param frame_id2 the id of the second frame
/// @return the transform T_frame2_frame1
Sophus::SE3d estimate_motion_2d2d(const VisualSlamGraph &graph,
                                  size_t frame_id1, size_t frame_id2);

/// @brief Adds a frame with a relative transform to the graph
/// @param graph the graph to add the frame to
/// @param frame the frame to add
/// @param base_frame_id the id of the frame that the pose is relative to
/// @param T_rf the transform from the new frame to the base frame
void add_frame_relative(VisualSlamGraph &graph,
                        std::unique_ptr<VisualFrame> frame,
                        size_t base_frame_id, const Sophus::SE3d &T_bf);

/// @brief triangulates points in the given frames
/// @param graph the graph containing the frames and points
/// @param frames the ids of the frames
void triangulate(VisualSlamGraph &graph, const std::vector<size_t> &frames);

// void reconstruct(VisualSlamGraph &graph, const std::vector<size_t> &frames);

} // namespace slam

#endif