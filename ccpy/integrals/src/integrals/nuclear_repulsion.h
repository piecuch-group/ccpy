#pragma once

#include <array>
#include <utility>
#include <vector>

namespace ccpy {

/// @brief Compute nuclear repulsion energy between the charges.
/// @param charges A vector of pairs of charges and Cartesian coordinates (in bohr).
/// @return The nuclear repulsion energy in atomic units.
double nuclear_repulsion(std::vector<std::pair<double, std::array<double, 3>>>& charges);

} // namespace ccpy