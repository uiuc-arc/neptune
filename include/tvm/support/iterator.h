/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#ifndef TVM_SUPPORT_ITERATOR_H_
#define TVM_SUPPORT_ITERATOR_H_

#include <tvm/runtime/container/array.h>

#include <optional>
#include <vector>

namespace tvm {
namespace support {

namespace details {

#define TVM_SET_ITER_FIELDS(V, Diff)                 \
  using iterator_category = std::input_iterator_tag; \
  using value_type = V;                              \
  using reference = V&;                              \
  using pointer = V*;                                \
  using difference_type = Diff;

template <typename Iter, typename Func>
struct MapIter {
  using U = typename Iter::value_type;
  using V = std::invoke_result_t<Func, U>;
  TVM_SET_ITER_FIELDS(V, std::ptrdiff_t)

  MapIter(Iter it, const Func& func) : current_(it), func_(func) {}

  V operator*() const { return func_(*current_); }
  MapIter& operator++() {
    ++this->current_;
    return *this;
  }

  bool operator==(const MapIter& other) const {
    return &func_ == &other.func_ && current_ == other.current_;
  }
  bool operator!=(const MapIter& other) const { return !(*this == other); }

 private:
  Iter current_;
  const Func& func_;
};

// Use this because std::optional<T> and Optional<T> have different APIs.
template <typename T>
bool OptHasValue(const std::optional<T>& opt) {
  return opt.has_value();
}
template <typename T>
bool OptHasValue(const Optional<T>& opt) {
  return opt.defined();
}

// A FilterMapIter either points to a defined value or the end of the container.
template <typename Iter, typename Func>
struct FilterMapIter {
  using U = typename Iter::value_type;
  using OptV = std::invoke_result_t<Func, U>;
  using V = std::remove_reference_t<decltype(OptV().value())>;
  TVM_SET_ITER_FIELDS(V, std::ptrdiff_t)

  FilterMapIter(Iter it, Iter end, const Func& func) : current_(it), end_(end), func_(func) {
    GoToNextValue();
  }

  V operator*() const { return *next_value_; }
  FilterMapIter& operator++() {
    ++current_;
    GoToNextValue();
    ICHECK(OptHasValue(next_value_) || current_ == end_);
    return *this;
  }

  bool GoToNextValue() {
    for (; current_ != end_; ++current_) {
      auto next_opt_v = func_(*current_);
      if (OptHasValue(next_opt_v)) {
        next_value_ = next_opt_v.value();
        return true;
      }
    }
    return false;
  }

  bool operator==(const FilterMapIter& other) const {
    return &func_ == &other.func_ && current_ == other.current_;
  }
  bool operator!=(const FilterMapIter& other) const { return !(*this == other); }

 private:
  Iter current_, end_;
  const Func& func_;
  std::optional<V> next_value_;
};

template <typename T>
struct RangeIter {
  TVM_SET_ITER_FIELDS(T, std::size_t)

  RangeIter(T start, T step, size_t position) : start_(start), step_(step), current_(position) {}

  T operator*() const { return start_ + current_ * step_; }
  RangeIter& operator++() {
    ++current_;
    return *this;
  }

  bool operator==(const RangeIter& other) const {
    return start_ == other.start_ && step_ == other.step_ && current_ == other.current_;
  }
  bool operator!=(const RangeIter& other) const { return !(*this == other); }

 private:
  T start_, step_;
  size_t current_;
};

template <typename Idx, typename Iter>
struct EnumerateIter1 {
  using V = typename Iter::value_type;
  using KV = std::pair<Idx, V>;
  TVM_SET_ITER_FIELDS(KV, std::size_t)

  EnumerateIter1(Iter it, Iter end, Idx idx = Idx(0)) : current_(it), end_(end), idx_(idx) {}

  KV operator*() const { return KV(idx_, *current_); }
  EnumerateIter1& operator++() {
    ++current_;
    ++idx_;
    return *this;
  }

  bool operator==(const EnumerateIter1& other) const { return current_ == other.current_; }
  bool operator!=(const EnumerateIter1& other) const { return !(*this == other); }

 private:
  Iter current_, end_;
  Idx idx_;
};

template <typename Idx, typename Iter>
struct EnumerateIter2 {
  using V = typename Iter::value_type;
  using KV = std::pair<V, Idx>;
  TVM_SET_ITER_FIELDS(KV, std::size_t)

  EnumerateIter2(Iter it, Iter end, Idx idx = Idx(0)) : current_(it), end_(end), idx_(idx) {}

  KV operator*() const { return KV(*current_, idx_); }
  EnumerateIter2& operator++() {
    ++current_;
    ++idx_;
    return *this;
  }

  bool operator==(const EnumerateIter2& other) const { return current_ == other.current_; }
  bool operator!=(const EnumerateIter2& other) const { return !(*this == other); }

 private:
  Iter current_, end_;
  Idx idx_;
};

template <typename Iter>
struct IteratorRange {
  IteratorRange(Iter begin, Iter end) : begin_(begin), end_(end) {}

  Iter begin() const { return begin_; }
  Iter end() const { return end_; }

  using Item = typename Iter::value_type;

  std::vector<Item> to_vector() const { return std::vector<Item>(begin(), end()); }

  template <class Container>
  auto to_container() const {
    return Container(begin(), end());
  }

  template <template <typename...> class Container, typename... Args>
  auto to_container(Args&&... args) const {
    return Container<Item>(begin(), end(), std::forward<Args>(args)...);
  }

 private:
  Iter begin_, end_;
};

}  // namespace details

template <typename Iter, typename Func>
auto map(const Iter& begin, const Iter& end, Func&& func) {
  auto ibegin = details::MapIter(begin, func), iend = details::MapIter(end, func);
  return details::IteratorRange(ibegin, iend);
}

template <typename Container, typename Func>
auto map(const Container& container, Func&& func) {
  return map(container.begin(), container.end(), std::forward<Func>(func));
}

template <typename Iter, typename Func>
auto filter_map(Iter begin, Iter end, Func&& func) {
  auto ibegin = details::FilterMapIter(begin, end, func),
       iend = details::FilterMapIter(end, end, func);
  return details::IteratorRange(ibegin, iend);
}

template <typename Container, typename Func>
auto filter_map(const Container& container, Func&& func) {
  return filter_map(container.begin(), container.end(), std::forward<Func>(func));
}

template <bool IdxFirst, typename Idx = size_t, typename Iter>
auto enumerate(Iter begin, Iter end, Idx start = Idx(0)) {
  using IterT = std::conditional_t<IdxFirst, details::EnumerateIter1<Idx, Iter>,
                                   details::EnumerateIter2<Idx, Iter>>;
  auto ibegin = IterT(begin, end, start), iend = IterT(end, end, start);
  return details::IteratorRange(ibegin, iend);
}

template <bool IdxFirst = true, typename Idx = size_t, typename Container>
auto enumerate(const Container& container, Idx start = Idx(0)) {
  return enumerate<IdxFirst, Idx>(container.begin(), container.end(), start);
}

template <typename T>
auto range(T start, T end, T step = T(1)) {
  auto ibegin = details::RangeIter(start, step, 0),
       iend = details::RangeIter(start, step, (end - start) / step);
  return details::IteratorRange(ibegin, iend);
}

template <typename Iter>
auto all_equal(const Iter& begin, const Iter& end) -> std::optional<typename Iter::value_type> {
  if (begin == end) {
    return std::nullopt;
  }
  auto value = *begin;
  if (std::all_of(begin + 1, end, [value](auto&& v) { return v == value; })) {
    return value;
  } else {
    return std::nullopt;
  }
}

template <typename Container>
auto all_equal(const Container& container) {
  return all_equal(container.begin(), container.end());
}

#define TVM_ITER_RANGE_REDUCE_METHODS(NAME)                                 \
  template <typename Iter, typename Func>                                   \
  auto NAME(const details::IteratorRange<Iter>& range, Func&& func) {       \
    return std::NAME(range.begin(), range.end(), std::forward<Func>(func)); \
  }

TVM_ITER_RANGE_REDUCE_METHODS(any_of)
TVM_ITER_RANGE_REDUCE_METHODS(all_of)
TVM_ITER_RANGE_REDUCE_METHODS(none_of)
TVM_ITER_RANGE_REDUCE_METHODS(for_each)
TVM_ITER_RANGE_REDUCE_METHODS(find_if)

}  // namespace support
}  // namespace tvm
#endif  // TVM_SUPPORT_ITERATOR_H_
