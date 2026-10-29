#pragma once

#include <numeric>

namespace ccpy {

namespace math {

template <typename Container> auto sum(const Container& c) -> typename Container::value_type {
    using T = typename Container::value_type;
    return std::accumulate(std::begin(c), std::end(c), T(0));
}

template <typename Container> auto product(const Container& c) -> typename Container::value_type {
    using T = typename Container::value_type;
    return std::accumulate(std::begin(c), std::end(c), T(1), std::multiplies<T>());
}


} // namespace math
} // namespace ccpy