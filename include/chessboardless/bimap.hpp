#ifndef INC_GUARD_BIMAP_HPP
#define INC_GUARD_BIMAP_HPP

#include <optional>
#include <unordered_map>

template <typename Left, typename Right> class unordered_bimap {
public:
  void insert_left(Left &&left, Right &&right) {
    left_to_right[left] = right;
    right_to_left[right] = left;
  }
  void insert_right(Right &&right, Left &&left) {
    right_to_left[right] = left;
    left_to_right[left] = right;
  }

  std::optional<const Right &> get_left(Left &&left) {
    const auto it = left_to_right.find(left);
    if (it == left.end()) {
      return std::nullopt;
    } else {
      return {*it};
    }
  }

  std::optional<const Left &> get_right(Right &&right) {
    const auto it = right_to_left.find(right);
    if (it == right.end()) {
      return std::nullopt;
    } else {
      return {*it};
    }
  }

private:
  std::unordered_map<Left, Right> left_to_right;
  std::unordered_map<Right, Left> right_to_left;
};

#endif