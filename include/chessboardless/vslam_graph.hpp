///
/// @file vslam_graph.hpp
/// @brief defines a pose graph + feature graph
///

#ifndef INC_GUARD_VSLAM_GRAPH
#define INC_GUARD_VSLAM_GRAPH

#include "chessboardless/graph.hpp"
#include <ceres/autodiff_cost_function.h>
#include <ceres/cost_function.h>
#include <cstddef>
#include <memory>
#include <sophus/se3.hpp>
#include <unordered_map>
#include <utility>
#include <variant>

// vslam graph needs to hold camera poses and triangulated points and relate
// them to elements in the feature graph.

namespace slam {} // namespace slam

#endif