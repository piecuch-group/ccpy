#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "integrals/basis.h"

namespace ccpy {

/// @brief Validates the centers of derivative operators.
/// @param basis The basis set to against which we validate the centers.
/// @param charges The vector of pairs (charge, center) for each center in the system.
/// @param basis_name The basis set label.
/// @param caller_name The name of the function calling this validation.
/// @throws std::invalid_argument if the number of centers in the basis does not match the number of
/// centers in the charges, or if the centers in the basis do not match the centers in the charges.
inline void
validate_deriv_centers(const Basis& basis,
                       const std::vector<std::pair<double, std::array<double, 3>>>& charges,
                       const std::string& basis_name, const std::string& caller_name) {
    const auto atom_to_shell = basis.center_first_and_last(true);

    // Check that the number of centers in the basis matches the number of centers in the charges
    if (atom_to_shell.size() != charges.size()) {
        throw std::invalid_argument(
            caller_name + ": " + basis_name + " has " + std::to_string(atom_to_shell.size()) +
            " centers, but charges has " + std::to_string(charges.size()) + " centers");
    }

    // Check that the centers in the basis match the centers in the charges
    constexpr double center_tol = 1.0e-8;
    for (std::size_t A{0}, natom{atom_to_shell.size()}; A < natom; ++A) {
        const auto [first_shell, last_shell] = atom_to_shell[A];
        if (first_shell == last_shell) {
            continue;
        }

        const auto& shell_center = basis[first_shell].O;
        const auto& charge_center = charges[A].second;
        const double center_dist =
            std::hypot(shell_center[0] - charge_center[0], shell_center[1] - charge_center[1],
                       shell_center[2] - charge_center[2]);
        if (center_dist > center_tol) {
            throw std::invalid_argument(caller_name + ": " + basis_name + " center " +
                                        std::to_string(A) + " does not match charges center " +
                                        std::to_string(A) + "; distance is " +
                                        std::to_string(center_dist));
        }
    }
}

/// @brief Validates the shape of derivative weights for a given integral type.
/// @tparam Array The type of the weights array.
/// @tparam N The number of dimensions of the weights array.
/// @param weights The weights array to validate.
/// @param expected The expected shape of the weights array.
/// @param weight_name The name of the weights array.
/// @param caller_name The name of the function calling this validation.
/// @throws std::invalid_argument if the shape of the weights array does not match the expected
/// shape.
template <typename Array, std::size_t N>
inline void
validate_deriv_weight_shape(const Array& weights, const std::array<std::size_t, N>& expected,
                            const std::string& weight_name, const std::string& caller_name) {
    for (std::size_t i = 0; i < N; ++i) {
        if (weights.shape(i) != expected[i]) {
            std::ostringstream actual;
            std::ostringstream expect;
            actual << "(";
            expect << "(";
            for (std::size_t dim = 0; dim < N; ++dim) {
                if (dim != 0) {
                    actual << ", ";
                    expect << ", ";
                }
                actual << weights.shape(dim);
                expect << expected[dim];
            }
            actual << ")";
            expect << ")";
            throw std::invalid_argument(caller_name + ": " + weight_name + " has incorrect shape " +
                                        actual.str() + "; expected " + expect.str());
        }
    }
}

/// @brief Validates that the shell slices for derivative computations are valid.
/// @tparam N The dimensionality of the underlying integrals, e.g., 2 for two-center, 3 for
/// three-center
/// @param shell_slices The array of (start, end) indices for each basis's shells.
/// @param nshells The array of the number of shells in each basis.
/// @param caller_name The name of the function calling this validation.
/// @throws std::invalid_argument if any of the shell slices are not in the form of
template <std::size_t N>
inline void
validate_deriv_shell_slices(const std::array<std::pair<std::size_t, std::size_t>, N>& shell_slices,
                            const std::array<std::size_t, N>& nshells,
                            const std::string& caller_name) {
    for (std::size_t i = 0; i < N; ++i) {
        if (shell_slices[i].first >= shell_slices[i].second) {
            throw std::invalid_argument(caller_name +
                                        ": shell_slices indices must be in the form of "
                                        "(start, end) with start < end");
        }
        if (shell_slices[i].second > nshells[i]) {
            throw std::invalid_argument(caller_name +
                                        ": shell_slices indices must be within the number "
                                        "of shells in each basis");
        }
    }
}

/// @brief Returns the real part of a derivative weight.
/// @tparam T The type of the weight.
/// @param value The weight value.
/// @return The real part of the weight.
template <typename T> inline double real_deriv_weight(const T& value) {
    if constexpr (std::is_same_v<std::decay_t<T>, std::complex<double>>) {
        return value.real();
    } else {
        return static_cast<double>(value);
    }
}

} // namespace ccpy
