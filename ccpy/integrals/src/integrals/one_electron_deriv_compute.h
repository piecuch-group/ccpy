#pragma once

#include <stdexcept>
#include <string>

#include <libint2.hpp>

#include "integrals/basis.h"
#include "integrals/deriv_utils.h"

#include "helpers/ndarray.h"

namespace ccpy {

template <libint2::Operator Op, std::size_t M>
[[nodiscard]] auto
compute_one_electron_deriv(const Basis& basis1, const Basis& basis2, const np_matrix& dm,
                           const std::vector<std::pair<double, std::array<double, 3>>>& charges)
    -> np_vector {
    // Get the number of basis functions in each basis
    const std::size_t nb1 = basis1.size();
    const std::size_t nb2 = basis2.size();

    // Assert that dm has the correct shape
    if (dm.shape(0) != nb1 || dm.shape(1) != nb2) {
        throw std::invalid_argument(
            "compute_one_electron_deriv: density matrix has incorrect shape");
    }

    validate_deriv_centers(basis1, charges, "basis1", "compute_one_electron_deriv");
    validate_deriv_centers(basis2, charges, "basis2", "compute_one_electron_deriv");

    // Initialize libint2
    libint2::initialize();

    // Prepare engine
    const auto max_nprim = std::max(basis1.max_nprim(), basis2.max_nprim());
    const auto max_l = std::max(basis1.max_l(), basis2.max_l());
    libint2::Engine engine(Op, max_nprim, max_l, 1); // '1' means that we will get first derivatives

    // If the operator is position-dependent, set the charges as parameters
    constexpr bool op_is_pos_dep =
        (Op == libint2::Operator::nuclear || Op == libint2::Operator::opVop ||
         Op == libint2::Operator::erf_nuclear || Op == libint2::Operator::erfc_nuclear);
    if constexpr (op_is_pos_dep) {
        engine.set_params(charges);
    }

    const auto& results = engine.results();

    // Get arrays of indices of the first basis in a shell and the size of each shell
    const auto first_size1 = basis1.shell_first_and_size();
    const auto first_size2 = basis2.shell_first_and_size();

    const auto natoms = charges.size();
    // position-dependent operators also has result buffer lengths of 6 + 3N, where N is the number
    // of atoms, whereas position-independent operators has result buffer lengths of 6 (for the 6
    // derivative components)
    const size_t n_deriv_components = 6 + (op_is_pos_dep ? 3 * natoms : 0);

    const auto shell_to_atom1 = basis1.center_first_and_last(true);
    const auto shell_to_atom2 = basis2.center_first_and_last(true);
    // Allocate the result gradient vector
    np_vector grad = make_zeros<nb::numpy, double, 1>({natoms * 3});

    const auto dm_view = dm.view();
    auto grad_view = grad.view();

    for (std::size_t a1 = 0; a1 < natoms; ++a1) {
        const auto [first1, last1] = shell_to_atom1[a1];
        for (std::size_t a2 = 0; a2 < natoms; ++a2) {
            const auto [first2, last2] = shell_to_atom2[a2];
            for (std::size_t s1 = first1; s1 < last1; ++s1) {
                const auto& shell1 = basis1[s1];
                const auto [f1, n1] = first_size1[s1];

                for (std::size_t s2 = first2; s2 < last2; ++s2) {
                    const auto& shell2 = basis2[s2];
                    const auto [f2, n2] = first_size2[s2];

                    // Compute the integrals for this shell pair
                    engine.compute(shell1, shell2);

                    for (std::size_t k = 0; k < 6; ++k) {
                        auto buf = results[k];
                        if (!buf)
                            continue; // if the buffer is null, skip to the next component
                        const std::size_t atom_idx =
                            (k / 3 == 0)
                                ? a1
                                : a2; // first 3 components are for atom a1, next 3 are for atom a2
                        const std::size_t cart_idx = k % 3;
                        for (std::size_t i = 0, ij = 0; i != n1; ++i) {
                            for (std::size_t j = 0; j != n2; ++j, ++ij) {
                                const auto dm_val = dm_view(f1 + i, f2 + j);
                                grad_view(atom_idx * 3 + cart_idx) +=
                                    static_cast<double>(buf[ij]) * dm_val;
                            }
                        }
                    }
                    // automatically skipped for position-independent operators since their result
                    // buffer lengths are only 6, so the loop will not execute for k >= 6
                    for (std::size_t k = 6; k < n_deriv_components; ++k) {
                        auto buf = results[k];
                        if (!buf)
                            continue; // if the buffer is null, skip to the next component
                        const std::size_t atom_idx = (k - 6) / 3;
                        const std::size_t cart_idx = (k - 6) % 3;
                        for (std::size_t i = 0, ij = 0; i != n1; ++i) {
                            for (std::size_t j = 0; j != n2; ++j, ++ij) {
                                const auto dm_val = dm_view(f1 + i, f2 + j);
                                grad_view(atom_idx * 3 + cart_idx) +=
                                    static_cast<double>(buf[ij]) * dm_val;
                            }
                        }
                    }
                }
            }
        }
    }

    libint2::finalize();

    return grad;
}
} // namespace ccpy
