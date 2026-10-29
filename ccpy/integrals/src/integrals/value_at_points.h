#pragma once

#include <vector>
#include <array>

#include <libint2/shell.h>
#include "helpers/ndarray.h"

namespace ccpy {
class Basis;

/// @brief Evaluate the basis functions at the given points.
/// @param points a vector of points at which to evaluate the basis functions
/// @return a 2D array of shape (npoints, nbasis) containing the values of the basis
/// functions
np_matrix basis_at_points(const Basis& basis, const std::vector<std::array<double, 3>>& points);

/// @brief Evaluate the product of basis functions times a coefficient matrix at the given
/// points.
/// @param points a vector of points at which to evaluate the basis functions
/// @param C a 2D array of shape (nbasis, norb) containing the coefficients
/// @return a 2D array of shape (npoints, norb) containing the values of the basis
/// functions times the coefficients
/// @note The shape of C must be (nbasis, norb), where nbasis is the number of basis
/// functions in the basis set and norb is the number of orbitals.
np_matrix orbitals_at_points(const Basis& basis, const std::vector<std::array<double, 3>>& points,
                             np_matrix C);

void evaluate_shell(const libint2::Shell& shell, const std::array<double, 3>& point,
                    double* buffer);

std::vector<std::array<double, 3>> regular_grid(const std::array<double, 3> min,
                                                const std::array<size_t, 3> npoints,
                                                const std::vector<std::array<double, 3>> dr);

np_matrix orbitals_on_grid(const Basis& basis, np_matrix C, const std::array<double, 3> min,
                           const std::array<size_t, 3> npoints,
                           const std::vector<std::array<double, 3>> axis);

} // namespace ccpy