#include "chessboardless/graph.hpp"
#include "chessboardless/feature_detection.hpp"
#include <algorithm>
#include <assert.h>
#include <deque>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <queue>
#include <ranges>
#include <utility>

namespace rv = std::ranges::views;

namespace slam {

size_t VisualSlamGraph::insert_frame(std::unique_ptr<VisualFrame> frame,
                                     std::optional<size_t> id) {
  if (id.has_value()) {
    const auto fid = id.value();

    // asserted because trying to insert a frame with the same id means
    // something has gone very wrong in the book keeping
    assert(!frames.contains(fid) &&
           "Feature Graph TRIED INSERT WITH FRAME ID THAT ALREADY EXISTS");

    if (frame_id_count < fid) {
      // update the count to not cause collisions if an automatic id frame is
      // inserted. yes this could be abused by setting fid very high for no
      // reason.
      frame_id_count = fid + 1;
    }

    frames.insert({fid, std::move(frame)});
    frame_point_graph.insert({fid, {}});
    covisibility_graph.insert({fid, {}});
    return fid;
  } else {
    frames.insert({++frame_id_count, std::move(frame)});
    frame_point_graph.insert({frame_id_count, {}});
    covisibility_graph.insert({frame_id_count, {}});
    return frame_id_count;
  }
};

size_t VisualSlamGraph::insert_observation(size_t frame_id,
                                           const features::Feature &feature,
                                           std::optional<size_t> point_id) {

  assert(frames.contains(frame_id) &&
         "TRIED TO INSERT DETECTION FOR FRAME THAT DOES NOT EXIST");

  if (point_id.has_value()) {
    const auto pid = point_id.value();
    assert(points.contains(pid) &&
           "TRIED TO lINK AGANIST POINT THAT DOES NOT EXIST");

    assert(!edges.contains({frame_id, pid}) &&
           "TRIED TO INSERT EDGE THAT ALREADY EXISTS INTO FRAME GRAPH");

    frame_point_graph.at(pid).push_back(frame_id);
    edges.insert({std::make_pair(frame_id, pid),
                  std::make_unique<ObservationEdge>(feature)});

    // maintain invariances
    update_point(pid);
    update_covisilbity_graph(frame_id, pid, CovisibilityOperation::Added);

    return pid;
  } else {
    const auto pid = ++point_id_count;

    points.insert(
        {pid, std::make_unique<Point>(frame_id, feature.descriptor,
                                      feature.keypoint.octave, feature.n_levels,
                                      calculate_min_distance_ratio(feature),
                                      calculate_max_distance_ratio(feature))});

    frame_point_graph.insert({pid, {frame_id}});
    frame_point_graph.at(frame_id).push_back(pid);
    edges.insert({std::make_pair(frame_id, pid),
                  std::make_unique<ObservationEdge>(feature)});

    return pid;
  }
};

const VisualFrame &VisualSlamGraph::get_frame(size_t frame_id) const {

  // higher level book keeping has gone very wrong for this to happen
  // and the program is in an invalid state.
  assert(frames.contains(frame_id) &&
         "QUERIED FEATURE GRAPH FOR FRAME ID THAT DOES NOT EXIST");
  return *frames.at(frame_id);
}

VisualFrame &VisualSlamGraph::get_frame(size_t frame_id) {

  // higher level book keeping has gone very wrong for this to happen
  // and the program is in an invalid state.
  assert(frames.contains(frame_id) &&
         "QUERIED FEATURE GRAPH FOR FRAME ID THAT DOES NOT EXIST");
  return *frames.at(frame_id);
}

const Point &VisualSlamGraph::get_point(size_t point_id) const {
  assert(points.contains(point_id) &&
         "FEATURE GRAPH QUERIED FOR POINT THAT DOES NOT EXIST");
  return *points.at(point_id);
}

Point &VisualSlamGraph::get_point(size_t point_id) {
  assert(points.contains(point_id) && "Graph: Does not contain point");
  return *points.at(point_id);
}

size_t VisualSlamGraph::num_points() const { return points.size(); }

const ObservationEdge &VisualSlamGraph::get_observation(size_t frame_id,
                                                        size_t point_id) const {
  assert(frames.contains(frame_id) &&
         "QUERIED FEATURE GRAPH FOR FRAME ID THAT DOES NOTE EXIST");
  assert(points.contains(point_id) &&
         "FEATURE GRAPH QUERIED FOR POINT THAT DOES NOT EXIST");
  return *edges.at({frame_id, point_id});
}

const std::vector<size_t> &
VisualSlamGraph::get_observers(size_t point_id) const {
  assert(points.contains(point_id) && "Graph: does not contain point");
  return frame_point_graph.at(point_id);
}

std::span<const size_t>
VisualSlamGraph::get_frame_points(size_t frame_id) const {
  // higher level book keeping has gone very wrong for this to happen
  // and the program is in an invalid state.
  assert(frames.contains(frame_id) &&
         "QUERIED FEATURE GRAPH FOR FRAME ID THAT DOES NOTE EXIST");
  assert(frame_point_graph.contains(frame_id) &&
         "FEATURE GRAPH ADJ LIST DOES NOT CONTAIN QUERIED FRAME");
  return frame_point_graph.at(frame_id);
}

const std::vector<size_t> &
VisualSlamGraph::get_covisibility_neighbors(size_t frame_id) const {
  assert(frames.contains(frame_id) &&
         "Feature Graph: Tried to query covisibility for frame that does not "
         "exist.");
  return covisibility_graph.at(frame_id);
};

size_t VisualSlamGraph::get_covisibility_count(size_t frame_id1,
                                               size_t frame_id2) const {
  assert(frames.contains(frame_id1) &&
         "Feature Graph: tried to query for frame that does not exist.");
  assert(frames.contains(frame_id2) &&
         "Feature Graph: tried to query for frame that does not exist.");

  const auto covisibility_edge =
      covisibility_edges.find(covisibility_pair(frame_id1, frame_id2));

  if (covisibility_edge == covisibility_edges.end()) {
    return 0;
  } else {
    return covisibility_edge->second;
  }
}

bool VisualSlamGraph::remove_observation(size_t frame_id, size_t point_id) {
  assert(!points.contains(point_id) &&
         "Feature Graph: Tried to remove observation from point that does not "
         "exist");

  assert(!frames.contains(frame_id) &&
         "Feature Graph: Tried to remove an observation from a frame that does "
         "not exist");

  assert(!edges.contains({frame_id, point_id}) &&
         "Feature Graph: Tried to remove edge that does not exist");

  if (!unsafe_remove_observation(frame_id, point_id, true)) {
    update_point(point_id);
    update_covisilbity_graph(frame_id, point_id,
                             CovisibilityOperation::Removed);
    return false;
  } else {
    return true;
  }
}

bool VisualSlamGraph::remove_observations(std::span<const size_t> frame_ids,
                                          size_t point_id) {
  assert(points.contains(point_id) &&
         "Feature Graph: Tried to remove observation from point that does not "
         "exist");

  for (const auto &frame_id : frame_ids) {
    assert(
        frames.contains(frame_id) &&
        "Feature Graph: Tried to remove an observation from a frame that does "
        "not exist");

    assert(edges.contains({frame_id, point_id}) &&
           "Feature Graph: Tried to remove edge that does not exist");

    unsafe_remove_observation(frame_id, point_id, false);

    // maintain invariance
    update_covisilbity_graph(frame_id, point_id,
                             CovisibilityOperation::Removed);
  }

  if (frame_point_graph.at(point_id).empty()) {
    frame_point_graph.erase(point_id);
    points.erase(point_id);
    return true;
  } else {
    // maintain invariance
    update_point(point_id);
    return false;
  }
}

void VisualSlamGraph::remove_point(size_t point_id) {
  assert(points.contains(point_id) &&
         "FEATURE GRAPH: TRIED TO REMOVE POINT THAT HAS ALREADY BEEN REMOVED");

  // remove all observations of point, this removes the point
  const bool point_removed =
      remove_observations(frame_point_graph.at(point_id), point_id);
  assert(
      point_removed &&
      "Feature Graph: point should be removed after all observations removed.");
}

void VisualSlamGraph::remove_frame(size_t frame_id) {
  assert(frames.contains(frame_id) &&
         "Feature Graph: cannot remove frame that is not in graph.");

  for (const auto &point_id : frame_point_graph.at(frame_id)) {
    remove_observation(frame_id, point_id);
  }

  frames.erase(frame_id);
  frame_point_graph.erase(frame_id);
}

void VisualSlamGraph::merge_points(size_t point_id1, size_t point_id2) {
  // we will be merging into point1, so find all the frames that observe point 2
  // but not 1
  const auto frames1 = get_observers(point_id1);
  std::set<size_t> old_frames;
  old_frames.insert(frames1.cbegin(), frames1.cend());

  std::vector<size_t> new_frames;
  for (const auto &frame_id : get_observers(point_id2)) {
    if (!old_frames.contains(frame_id)) {
      new_frames.push_back(frame_id);
    }
  }

  // get all the edges that will be lost
  std::vector<std::unique_ptr<ObservationEdge>> edges_to_merge;
  for (const auto &frame_id : new_frames) {
    edges_to_merge.push_back(std::move(edges.at({frame_id, point_id2})));
  }

  // remove point2
  remove_point(point_id2);

  // insert observations onto point 1
  for (const auto &[frame_id, edge] : rv::zip(new_frames, edges_to_merge)) {
    edges.insert({std::make_pair(frame_id, point_id1), std::move(edge)});

    std::erase(frame_point_graph.at(frame_id), point_id2);
    frame_point_graph.at(frame_id).push_back(point_id1);
    frame_point_graph.at(point_id1).push_back(frame_id);

    update_covisilbity_graph(frame_id, point_id1, CovisibilityOperation::Added);
  }

  // maintain point invariance
  update_point(point_id1);
}

void VisualSlamGraph::update_point(size_t point_id) {

  auto &point = *points.at(point_id);
  const auto &frame_ids = frame_point_graph.at(point_id);

  double min_distance = std::numeric_limits<double>::max();
  for (const auto frame_id1 : frame_ids) {
    const auto &desc1 = get_observation(frame_id1, point_id).feature.descriptor;

    double distance = 0.0;
    for (const auto frame_id2 : frame_ids) {
      const auto &desc2 =
          get_observation(frame_id2, point_id).feature.descriptor;
      distance += cv::norm(desc1, desc2, cv::NORM_HAMMING);
    }

    if (distance < min_distance) {
      min_distance = distance;
      point.best_frame_id = frame_id1;
    }
  }

  // get the best observation of the point
  const auto &observation = get_observation(point.best_frame_id, point_id);
  point.descriptor = observation.feature.descriptor;
  point.level = observation.feature.keypoint.octave;
  point.min_distance_ratio = calculate_min_distance_ratio(observation.feature);
  point.max_distance_ratio = calculate_max_distance_ratio(observation.feature);
}

void VisualSlamGraph::update_covisilbity_graph(
    size_t frame_id, size_t point_id, CovisibilityOperation operation) {

  assert(frames.contains(frame_id) && "Feature Graph: Does not contain frame");
  assert(points.contains(point_id) && "Feature Graph: Does not contain point");
  assert((operation == CovisibilityOperation::Removed ||
          edges.contains({frame_id, point_id})) &&
         "Feature Graph: No edge between frame and point");

  // find all the frames that observe the point
  const auto &frame_ids = frame_point_graph.at(point_id);

  for (const auto &other_frame : frame_ids) {

    const auto pair = covisibility_pair(frame_id, other_frame);
    if (operation == CovisibilityOperation::Added) {
      const auto edge = covisibility_edges.find(pair);

      if (edge != covisibility_edges.end()) {
        // edge already exists increment count
        edge->second++;
      } else {
        // new edge is needed
        covisibility_edges.insert({pair, 1});
        covisibility_graph.at(frame_id).push_back(other_frame);
        covisibility_graph.at(other_frame).push_back(frame_id);
      }

    } else if (operation == CovisibilityOperation::Removed) {
      assert(covisibility_edges.contains(pair) &&
             "Feature Graph: covisibility graph does not contain edge");

      const auto edge = covisibility_edges.find(pair);
      edge->second--;
      if (edge->second <= 0) {
        // no more shared features, remove
        covisibility_edges.erase(pair);
        std::erase(covisibility_graph.at(frame_id), other_frame);
        std::erase(covisibility_graph.at(other_frame), frame_id);
      }
    }
  }
}

bool VisualSlamGraph::unsafe_remove_observation(size_t frame_id,
                                                size_t point_id,
                                                bool remove_point) {
  // erase the connection between the point and frame
  std::erase(frame_point_graph.at(frame_id), point_id);
  std::erase(frame_point_graph.at(point_id), frame_id);
  edges.erase(edges.find({frame_id, point_id}));

  if (remove_point && frame_point_graph.at(point_id).empty()) {
    // point has nothing observing it, remove
    frame_point_graph.erase(point_id);
    points.erase(point_id);
    return true;
  } else {
    return false;
  }
}

std::pair<size_t, size_t>
VisualSlamGraph::covisibility_pair(size_t frame_id1, size_t frame_id2) const {
  if (frame_id1 > frame_id2) {
    return {frame_id2, frame_id1};
  } else {
    return {frame_id1, frame_id2};
  }
}

std::set<size_t> covisibility_query(const VisualSlamGraph &feature_graph,
                                    size_t frame_id,
                                    size_t covisibility_threshold,
                                    size_t max_depth) {

  // setup BFS queue
  std::set<size_t> frame_ids{frame_id};
  std::queue<size_t> frame_queue;
  frame_queue.push(frame_id);

  const auto add_to_queue = [&frame_queue, &frame_ids](const auto fid) {
    if (!frame_ids.contains(fid)) {
      frame_ids.insert(fid);
      frame_queue.push(fid);
    }
  };

  size_t current_depth = 0;

  while (!frame_queue.empty() && current_depth < max_depth) {

    // process current layer
    for (const auto &remaining [[maybe_unused]] :
         rv::iota(frame_queue.size())) {

      const auto &id = frame_queue.front();

      // if it isn't the start and we haven't visted it
      if (id != frame_id && !frame_ids.contains(id)) {

        const auto meets_threshold = [&feature_graph, covisibility_threshold,
                                      id](const size_t other) {
          return feature_graph.get_covisibility_count(id, other) >
                 covisibility_threshold;
        };

        std::ranges::for_each(feature_graph.get_covisibility_neighbors(id) |
                                  rv::filter(meets_threshold),
                              add_to_queue);

        frame_queue.pop();
      }
      current_depth++;
    }
  }

  return frame_ids;
}

std::pair<std::vector<cv::KeyPoint>, cv::Mat>
get_frame_keypoints(const VisualSlamGraph &feature_graph, size_t frame_id) {
  const auto &point_ids = feature_graph.get_frame_points(frame_id);

  std::cout << point_ids.size() << std::endl;

  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;

  for (const auto &pid : point_ids) {
    const auto &feature = feature_graph.get_observation(frame_id, pid).feature;
    keypoints.push_back(feature.keypoint);
    descriptors.push_back(feature.descriptor);
  }

  return {keypoints, descriptors};
}

std::vector<cv::KeyPoint>
get_all_measurements(const VisualSlamGraph &graph,
                     std::span<size_t const> frame_ids, size_t point_id) {
  std::vector<cv::KeyPoint> key_points;
  for (const auto &frame_id : frame_ids) {
    key_points.push_back(
        graph.get_observation(frame_id, point_id).feature.keypoint);
  }
  return key_points;
}

std::vector<size_t> get_shared_points(const VisualSlamGraph &graph,
                                      size_t frame_id1, size_t frame_id2) {
  const auto &frame1_points = graph.get_frame_points(frame_id1);
  const auto &frame2_points = graph.get_frame_points(frame_id2);

  std::set<size_t> points1(frame1_points.cbegin(), frame1_points.cend());
  std::set<size_t> points2(frame2_points.cbegin(), frame2_points.cend());

  std::vector<size_t> shared_points;
  std::set_intersection(points1.cbegin(), points1.cend(), points2.cbegin(),
                        points2.cend(), std::back_inserter(shared_points));

  return shared_points;
}

} // namespace slam
