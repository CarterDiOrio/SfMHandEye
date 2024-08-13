#ifndef INC_GUARD_FEATURE_MATCHING_HPP
#define INC_GUARD_FEATURE_MATCHING_HPP

#include "chessboardless/feature_detection.hpp"
#include "chessboardless/graph.hpp"

namespace slam::features {

/// @brief matches features between the two frames
/// @param feature_graph the feature graph containing the frames
/// @param frame_id1 the id of the first frame
/// @param frame_id2 the id of the second frame
/// @return a list of the ids of matching points
std::vector<std::pair<size_t, size_t>>
match_features(const VisualSlamGraph &feature_graph, size_t frame_id1,
               size_t frame_id2, cv::Mat img1, cv::Mat img2);

} // namespace slam::features

#endif