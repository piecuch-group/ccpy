#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <stdexcept>
#include <string>
#include <vector>

#include <libint2.hpp>

#include "helpers/ndarray.h"
#include "helpers/parallel.h"
#include "integrals/basis.h"
#include "integrals/deriv_utils.h"

namespace ccpy {

inline auto shell_to_center_index(const Basis& basis) -> std::vector<std::size_t> {
    const auto center_to_shell = basis.center_first_and_last(true);
    std::vector<std::size_t> shell_to_center(basis.nshells());
    for (std::size_t atom = 0; atom < center_to_shell.size(); ++atom) {
        const auto [first_shell, last_shell] = center_to_shell[atom];
        for (std::size_t shell = first_shell; shell < last_shell; ++shell) {
            shell_to_center[shell] = atom;
        }
    }
    return shell_to_center;
}

/// @brief Contracts a given set of weights with the derivative of the three-center
/// two-electron integrals (P| mu nu) with repect to nuclear coordinates, where P is a basis
/// function from basis1, and mu, nu are basis functions from basis2 and basis3 respectively.
/// @tparam Weights a contigous array type that can be viewed as a 3D array with shape
/// (nbasis1_slice, nbasis2_slice, nbasis3_slice)
/// @tparam Op the libint2 operator to compute the integral for
/// @param basis1 The first basis set (P)
/// @param basis2 The second basis set (mu)
/// @param basis3 The third basis set (nu)
/// @param shell_slices an array of pairs of (first,last) shell indices to compute for each basis.
/// @param weights the weights to contract with the integral derivatives, with shape (nbasis1_slice,
/// nbasis2_slice, nbasis3_slice),
/// @param charges the list of nuclear charges and coordinates, which must be in the same order as
/// the centers in the basis sets. This is only used if the operator is position-dependent (e.g.
/// nuclear attraction), and can be an empty vector for position-independent operators (e.g.
/// kinetic).
/// @return a vector of length 3N, where N is the number of atoms, containing the gradient of the
/// contracted integral with respect to the nuclear coordinates, ordered as (d/dx1, d/dy1,
/// d/dz1,...).
template <libint2::Operator Op, typename Weights>
[[nodiscard]] auto compute_two_electron_3c_deriv_by_shell(
    const Basis& basis1, const Basis& basis2, const Basis& basis3,
    const std::array<std::pair<std::size_t, std::size_t>, 3>& shell_slices, const Weights& weights,
    const std::vector<std::pair<double, std::array<double, 3>>>& charges) -> np_vector {
    validate_deriv_centers(basis1, charges, "basis1", "compute_two_electron_3c_deriv_by_shell");
    validate_deriv_centers(basis2, charges, "basis2", "compute_two_electron_3c_deriv_by_shell");
    validate_deriv_centers(basis3, charges, "basis3", "compute_two_electron_3c_deriv_by_shell");
    validate_deriv_shell_slices<3>(
        shell_slices,
        std::array<std::size_t, 3>{basis1.nshells(), basis2.nshells(), basis3.nshells()},
        "compute_two_electron_3c_deriv_by_shell");

    const auto [s1_begin, s1_end] = shell_slices[0];
    const auto [s2_begin, s2_end] = shell_slices[1];
    const auto [s3_begin, s3_end] = shell_slices[2];

    const auto offsets1 = basis1.shell_offsets();
    const auto offsets2 = basis2.shell_offsets();
    const auto offsets3 = basis3.shell_offsets();
    const std::size_t first1 = offsets1[s1_begin];
    const std::size_t first2 = offsets2[s2_begin];
    const std::size_t first3 = offsets3[s3_begin];
    const std::size_t nb1_slice = offsets1[s1_end] - first1;
    const std::size_t nb2_slice = offsets2[s2_end] - first2;
    const std::size_t nb3_slice = offsets3[s3_end] - first3;
    validate_deriv_weight_shape(weights,
                                std::array<std::size_t, 3>{nb1_slice, nb2_slice, nb3_slice}, "W3",
                                "compute_two_electron_3c_deriv_by_shell");

    libint2::initialize();

    const auto max_nprim = std::max({basis1.max_nprim(), basis2.max_nprim(), basis3.max_nprim()});
    const auto max_l = std::max({basis1.max_l(), basis2.max_l(), basis3.max_l()});

    const auto first_size1 = basis1.shell_first_and_size();
    const auto first_size2 = basis2.shell_first_and_size();
    const auto first_size3 = basis3.shell_first_and_size();
    const auto shell1_to_atom = shell_to_center_index(basis1);
    const auto shell2_to_atom = shell_to_center_index(basis2);
    const auto shell3_to_atom = shell_to_center_index(basis3);

    const std::size_t natoms = charges.size();
    auto grad = make_zeros<nb::numpy, double, 1>({3 * natoms});
    auto grad_data = grad.data();
    const auto weight_data = weights.data();
    // Find the number of threads to use (no more than the number of first basis slices)
    const std::size_t max_chunks = std::min(get_num_threads(), s1_end - s1_begin);
    std::vector<std::vector<double>> local_grads(max_chunks, std::vector<double>(3 * natoms, 0.0));
    // Atomic counter to assign a local gradient to each thread without overlap
    std::atomic<std::size_t> next_chunk{0};

    auto kernel = [&](std::size_t s1_chunk_begin, std::size_t s1_chunk_end) {
        libint2::Engine engine(Op, max_nprim, max_l, 1);
        engine.set(libint2::BraKet::xs_xx);
        const auto& results = engine.results();

        const std::size_t chunk_idx = next_chunk.fetch_add(1);
        auto& local_grad = local_grads[chunk_idx];

        for (std::size_t s1 = s1_chunk_begin; s1 < s1_chunk_end; ++s1) {
            const auto& shell1 = basis1[s1];
            const auto [f1, n1] = first_size1[s1];
            const std::size_t atom1 = shell1_to_atom[s1];

            for (std::size_t s2 = s2_begin; s2 < s2_end; ++s2) {
                const auto& shell2 = basis2[s2];
                const auto [f2, n2] = first_size2[s2];
                const std::size_t atom2 = shell2_to_atom[s2];

                for (std::size_t s3 = s3_begin; s3 < s3_end; ++s3) {
                    const auto& shell3 = basis3[s3];
                    const auto [f3, n3] = first_size3[s3];
                    const std::size_t atom3 = shell3_to_atom[s3];

                    engine.compute(shell1, shell2, shell3);

                    for (std::size_t component = 0; component < 9; ++component) {
                        const auto buf = results[component];
                        if (!buf) {
                            continue;
                        }
                        const std::size_t center_idx = component / 3;
                        const std::size_t atom_idx =
                            center_idx == 0 ? atom1 : (center_idx == 1 ? atom2 : atom3);
                        const std::size_t cart_idx = component % 3;

                        for (std::size_t i = 0, ijk = 0; i != n1; ++i) {
                            for (std::size_t j = 0; j != n2; ++j) {
                                for (std::size_t k = 0; k != n3; ++k, ++ijk) {
                                    const std::size_t w_idx =
                                        (f1 - first1 + i) * nb2_slice * nb3_slice +
                                        (f2 - first2 + j) * nb3_slice + (f3 - first3 + k);
                                    const double w = real_deriv_weight(weight_data[w_idx]);
                                    local_grad[atom_idx * 3 + cart_idx] +=
                                        static_cast<double>(buf[ijk]) * w;
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    parallel_for_chunked(s1_begin, s1_end, kernel);

    for (const auto& local_grad : local_grads) {
        for (std::size_t i{0}, maxi{local_grad.size()}; i < maxi; ++i) {
            grad_data[i] += local_grad[i];
        }
    }

    libint2::finalize();
    return grad;
}

/// @brief Convenience wrapper around compute_two_electron_3c_deriv_by_shell that computes the
/// derivative for all shells in each basis set and contracts with the given weights.
template <libint2::Operator Op, typename Weights>
[[nodiscard]] auto
compute_two_electron_3c_deriv(const Basis& basis1, const Basis& basis2, const Basis& basis3,
                              const Weights& weights,
                              const std::vector<std::pair<double, std::array<double, 3>>>& charges)
    -> np_vector {
    return compute_two_electron_3c_deriv_by_shell<Op>(
        basis1, basis2, basis3,
        std::array<std::pair<std::size_t, std::size_t>, 3>{
            std::pair<std::size_t, std::size_t>{0, basis1.nshells()},
            std::pair<std::size_t, std::size_t>{0, basis2.nshells()},
            std::pair<std::size_t, std::size_t>{0, basis3.nshells()}},
        weights, charges);
}

/// @brief Contract mixed second derivatives of three-center integrals over the first basis.
/// @details For I(P, mu, nu) = (P|mu nu), evaluates the scalar and three Cartesian
/// Pauli components of the momentum-dressed integral, using derivatives with respect
/// to the centers of mu and nu.
/// The returned order is scalar, x, y, z, matching one-electron `opVop`.
/// @param basis1 Potential-density basis whose functions are summed over.
/// @param basis2 Orbital basis for mu.
/// @param basis3 Orbital basis for nu.
/// @return Four matrices with shape `(basis2.size(), basis3.size())`.
template <libint2::Operator Op>
[[nodiscard]] auto compute_two_electron_3c_opVop(const Basis& basis1, const Basis& basis2,
                                                 const Basis& basis3) -> std::array<np_matrix, 4> {
#if !defined(LIBINT2_DERIV_ERI3_ORDER) || LIBINT2_DERIV_ERI3_ORDER < 2
    throw std::runtime_error(
        "Three-center second derivatives are not available in this Libint2 build.");
#else
    libint2::initialize();

    const auto max_nprim = std::max({basis1.max_nprim(), basis2.max_nprim(), basis3.max_nprim()});
    const auto max_l = std::max({basis1.max_l(), basis2.max_l(), basis3.max_l()});

    const std::size_t nb2 = basis2.size();
    const std::size_t nb3 = basis3.size();
    const std::size_t nshells1 = basis1.nshells();
    const std::size_t nshells2 = basis2.nshells();
    const std::size_t nshells3 = basis3.nshells();

    const auto first_size1 = basis1.shell_first_and_size();
    const auto first_size2 = basis2.shell_first_and_size();
    const auto first_size3 = basis3.shell_first_and_size();

    std::array<np_matrix, 4> ints;
    for (auto& component : ints) {
        component = make_zeros<nb::numpy, double, 2>({nb2, nb3});
    }

    if (nshells1 == 0 || nshells2 == 0 || nshells3 == 0) {
        libint2::finalize();
        return ints;
    }

    // Libint orders second geometrical derivatives as the flattened upper triangle of
    // the nine shell-center coordinates: P(xyz), mu(xyz), nu(xyz).
    constexpr auto deriv_index = [](std::size_t first, std::size_t second) {
        constexpr std::size_t ncoords = 9;
        return first * ncoords - first * (first + 1) / 2 + second;
    };
    constexpr std::array<std::size_t, 9> deriv_indices{
        deriv_index(3, 6), // mu_x, nu_x
        deriv_index(4, 7), // mu_y, nu_y
        deriv_index(5, 8), // mu_z, nu_z
        deriv_index(4, 8), // mu_y, nu_z
        deriv_index(5, 7), // mu_z, nu_y
        deriv_index(5, 6), // mu_z, nu_x
        deriv_index(3, 8), // mu_x, nu_z
        deriv_index(3, 7), // mu_x, nu_y
        deriv_index(4, 6), // mu_y, nu_x
    };

    std::array<double*, 4> output{};
    for (std::size_t component = 0; component < ints.size(); ++component) {
        output[component] = ints[component].data();
    }

    // Parallelize over rows of the retained orbital matrix. Each thread writes to a
    // disjoint set of rows, avoiding both synchronization and per-thread matrix copies.
    auto kernel = [&](std::size_t s2_begin, std::size_t s2_end) {
        libint2::Engine engine(Op, max_nprim, max_l, 2);
        engine.set(libint2::BraKet::xs_xx);
        const auto& results = engine.results();

        for (std::size_t s2 = s2_begin; s2 < s2_end; ++s2) {
            const auto& shell2 = basis2[s2];
            const auto [f2, n2] = first_size2[s2];

            for (std::size_t s3 = 0; s3 < nshells3; ++s3) {
                const auto& shell3 = basis3[s3];
                const auto [f3, n3] = first_size3[s3];

                for (std::size_t s1 = 0; s1 < nshells1; ++s1) {
                    const auto& shell1 = basis1[s1];
                    const auto n1 = first_size1[s1].second;

                    engine.compute(shell1, shell2, shell3);

                    std::array<const double*, 9> buffers{};
                    for (std::size_t d = 0; d < deriv_indices.size(); ++d) {
                        buffers[d] = results[deriv_indices[d]];
                    }

                    for (std::size_t i = 0, ijk = 0; i < n1; ++i) {
                        for (std::size_t j = 0; j < n2; ++j) {
                            for (std::size_t k = 0; k < n3; ++k, ++ijk) {
                                std::array<double, 9> d{};
                                for (std::size_t component = 0; component < d.size(); ++component) {
                                    if (buffers[component]) {
                                        d[component] = static_cast<double>(buffers[component][ijk]);
                                    }
                                }

                                const std::size_t offset = (f2 + j) * nb3 + f3 + k;
                                output[0][offset] += d[0] + d[1] + d[2];
                                output[1][offset] += d[3] - d[4];
                                output[2][offset] += d[5] - d[6];
                                output[3][offset] += d[7] - d[8];
                            }
                        }
                    }
                }
            }
        }
    };

    parallel_for_chunked(0, nshells2, kernel);

    libint2::finalize();
    return ints;
#endif
}

/// @brief Contracts a given set of weights with the derivative of the two-center
/// two-electron integrals (P|Q) with repect to nuclear coordinates, where P and Q are basis
/// functions from basis1 and basis2 respectively.
/// @tparam Weights a contigous array type that can be viewed as a 2D array with shape
/// (nbasis1_slice, nbasis2_slice)
/// @tparam Op the libint2 operator to compute the integral for
/// @param basis1 The first basis set (P)
/// @param basis2 The second basis set (Q)
/// @param shell_slices an array of pairs of (first,last) shell indices to compute for each basis.
/// @param weights the weights to contract with the integral derivatives, with shape (nbasis1_slice,
/// nbasis2_slice),
/// @param charges the list of nuclear charges and coordinates, which must be in the same order as
/// the centers in the basis sets. This is only used if the operator is position-dependent (e.g.
/// nuclear attraction), and can be an empty vector for position-independent operators (e.g.
/// kinetic).
/// @return a vector of length 3N, where N is the number of atoms, containing the gradient of the
/// contracted integral with respect to the nuclear coordinates, ordered as (d/dx1, d/dy1,
/// d/dz1,...).
template <libint2::Operator Op, typename Weights>
[[nodiscard]] auto compute_two_electron_2c_deriv_by_shell(
    const Basis& basis1, const Basis& basis2,
    const std::array<std::pair<std::size_t, std::size_t>, 2>& shell_slices, const Weights& weights,
    const std::vector<std::pair<double, std::array<double, 3>>>& charges) -> np_vector {
    validate_deriv_centers(basis1, charges, "basis1", "compute_two_electron_2c_deriv_by_shell");
    validate_deriv_centers(basis2, charges, "basis2", "compute_two_electron_2c_deriv_by_shell");
    validate_deriv_shell_slices<2>(shell_slices,
                                   std::array<std::size_t, 2>{basis1.nshells(), basis2.nshells()},
                                   "compute_two_electron_2c_deriv_by_shell");

    const auto [s1_begin, s1_end] = shell_slices[0];
    const auto [s2_begin, s2_end] = shell_slices[1];

    const auto offsets1 = basis1.shell_offsets();
    const auto offsets2 = basis2.shell_offsets();
    const std::size_t first1 = offsets1[s1_begin];
    const std::size_t first2 = offsets2[s2_begin];
    const std::size_t nb1_slice = offsets1[s1_end] - first1;
    const std::size_t nb2_slice = offsets2[s2_end] - first2;
    validate_deriv_weight_shape(weights, std::array<std::size_t, 2>{nb1_slice, nb2_slice}, "W2",
                                "compute_two_electron_2c_deriv_by_shell");

    libint2::initialize();

    const auto max_nprim = std::max(basis1.max_nprim(), basis2.max_nprim());
    const auto max_l = std::max(basis1.max_l(), basis2.max_l());

    const auto first_size1 = basis1.shell_first_and_size();
    const auto first_size2 = basis2.shell_first_and_size();
    const auto shell1_to_atom = shell_to_center_index(basis1);
    const auto shell2_to_atom = shell_to_center_index(basis2);

    const std::size_t natoms = charges.size();
    auto grad = make_zeros<nb::numpy, double, 1>({3 * natoms});
    auto grad_data = grad.data();
    const auto weight_data = weights.data();
    // Find the number of threads to use (no more than the number of first basis slices)
    const std::size_t max_chunks = std::min(get_num_threads(), s1_end - s1_begin);
    // Each thread will have its own local gradient to avoid race conditions
    std::vector<std::vector<double>> local_grads(max_chunks, std::vector<double>(3 * natoms, 0.0));
    // Atomic counter to assign a local gradient to each thread without overlap
    std::atomic<std::size_t> next_chunk{0};

    auto kernel = [&](std::size_t s1_chunk_begin, std::size_t s1_chunk_end) {
        libint2::Engine engine(Op, max_nprim, max_l, 1);
        engine.set(libint2::BraKet::xs_xs);
        const auto& results = engine.results();

        const std::size_t chunk_idx = next_chunk.fetch_add(1);
        auto& local_grad = local_grads[chunk_idx];

        for (std::size_t s1 = s1_chunk_begin; s1 < s1_chunk_end; ++s1) {
            const auto& shell1 = basis1[s1];
            const auto [f1, n1] = first_size1[s1];
            const std::size_t atom1 = shell1_to_atom[s1];

            for (std::size_t s2 = s2_begin; s2 < s2_end; ++s2) {
                const auto& shell2 = basis2[s2];
                const auto [f2, n2] = first_size2[s2];
                const std::size_t atom2 = shell2_to_atom[s2];

                engine.compute(shell1, shell2);

                for (std::size_t component = 0; component < 6; ++component) {
                    const auto buf = results[component];
                    if (!buf) {
                        continue;
                    }
                    const std::size_t atom_idx = component / 3 == 0 ? atom1 : atom2;
                    const std::size_t cart_idx = component % 3;

                    for (std::size_t i = 0, ij = 0; i != n1; ++i) {
                        for (std::size_t j = 0; j != n2; ++j, ++ij) {
                            const std::size_t w_idx =
                                (f1 - first1 + i) * nb2_slice + (f2 - first2 + j);
                            const double w = real_deriv_weight(weight_data[w_idx]);
                            local_grad[atom_idx * 3 + cart_idx] += static_cast<double>(buf[ij]) * w;
                        }
                    }
                }
            }
        }
    };

    parallel_for_chunked(s1_begin, s1_end, kernel);

    for (const auto& local_grad : local_grads) {
        for (std::size_t i{0}, maxi{local_grad.size()}; i < maxi; ++i) {
            grad_data[i] += local_grad[i];
        }
    }

    libint2::finalize();
    return grad;
}

/// @brief Convenience wrapper around compute_two_electron_2c_deriv_by_shell that computes the
/// derivative for all shells in each basis set and contracts with the given weights.
template <libint2::Operator Op, typename Weights>
[[nodiscard]] auto
compute_two_electron_2c_deriv(const Basis& basis1, const Basis& basis2, const Weights& weights,
                              const std::vector<std::pair<double, std::array<double, 3>>>& charges)
    -> np_vector {
    return compute_two_electron_2c_deriv_by_shell<Op>(
        basis1, basis2,
        std::array<std::pair<std::size_t, std::size_t>, 2>{
            std::pair<std::size_t, std::size_t>{0, basis1.nshells()},
            std::pair<std::size_t, std::size_t>{0, basis2.nshells()}},
        weights, charges);
}

/// @brief Contracts a four-index weight tensor W(mu,nu,rho,sigma) with the first derivatives of
/// the four-center integrals (mu nu|rho sigma) with respect to the nuclear coordinates.
/// @details Only symmetry-unique shell quartets are computed, so W must have the full 8-fold
/// permutational symmetry of the integrals: W(mn,rs) = W(nm,rs) = W(mn,sr) = W(rs,mn). This holds
/// for any symmetrized two-particle density. All four indices run over the same basis.
/// @return the gradient contribution, a vector of length 3 * natoms ordered (x1, y1, z1, ...).
template <libint2::Operator Op, typename Weights>
[[nodiscard]] auto
compute_two_electron_4c_deriv(const Basis& basis, const Weights& weights,
                              const std::vector<std::pair<double, std::array<double, 3>>>& charges)
    -> np_vector {
    validate_deriv_centers(basis, charges, "basis", "compute_two_electron_4c_deriv");
    const std::size_t nbf = basis.size();
    validate_deriv_weight_shape(weights, std::array<std::size_t, 4>{nbf, nbf, nbf, nbf}, "W4",
                                "compute_two_electron_4c_deriv");

    libint2::initialize();

    const auto max_nprim = basis.max_nprim();
    const auto max_l = basis.max_l();
    const auto first_size = basis.shell_first_and_size();
    const auto shell_to_atom = shell_to_center_index(basis);
    const std::size_t nshells = basis.nshells();

    // symmetry-unique shell pairs (s1 >= s2)
    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    pairs.reserve(nshells * (nshells + 1) / 2);
    for (std::size_t s1 = 0; s1 < nshells; ++s1) {
        for (std::size_t s2 = 0; s2 <= s1; ++s2) {
            pairs.emplace_back(s1, s2);
        }
    }

    const std::size_t natoms = charges.size();
    auto grad = make_zeros<nb::numpy, double, 1>({3 * natoms});
    auto grad_data = grad.data();
    const auto weight_data = weights.data();
    const std::size_t n2 = nbf * nbf, n3 = n2 * nbf;

    const std::size_t max_chunks = std::max<std::size_t>(1, std::min(get_num_threads(), pairs.size()));
    std::vector<std::vector<double>> local_grads(max_chunks, std::vector<double>(3 * natoms, 0.0));
    std::atomic<std::size_t> next_chunk{0};

    auto kernel = [&](std::size_t p_begin, std::size_t p_end) {
        libint2::Engine engine(Op, max_nprim, max_l, 1);
        const auto& results = engine.results();
        const std::size_t chunk_idx = next_chunk.fetch_add(1);
        auto& local_grad = local_grads[chunk_idx];

        for (std::size_t p12 = p_begin; p12 < p_end; ++p12) {
            const auto [s1, s2] = pairs[p12];
            const auto [f1, n1] = first_size[s1];
            const auto [f2, nn2] = first_size[s2];
            for (std::size_t p34 = 0; p34 <= p12; ++p34) {
                const auto [s3, s4] = pairs[p34];
                const auto [f3, nn3] = first_size[s3];
                const auto [f4, nn4] = first_size[s4];
                const double degeneracy = (s1 == s2 ? 1.0 : 2.0) * (s3 == s4 ? 1.0 : 2.0) *
                                          (p12 == p34 ? 1.0 : 2.0);
                const std::array<std::size_t, 4> atoms{shell_to_atom[s1], shell_to_atom[s2],
                                                       shell_to_atom[s3], shell_to_atom[s4]};

                engine.compute(basis[s1], basis[s2], basis[s3], basis[s4]);

                for (std::size_t component = 0; component < 12; ++component) {
                    const auto buf = results[component];
                    if (!buf) {
                        continue;
                    }
                    double sum = 0.0;
                    for (std::size_t i = 0, ijkl = 0; i != n1; ++i) {
                        for (std::size_t j = 0; j != nn2; ++j) {
                            const std::size_t ij = (f1 + i) * n3 + (f2 + j) * n2;
                            for (std::size_t k = 0; k != nn3; ++k) {
                                const std::size_t ijk = ij + (f3 + k) * nbf + f4;
                                for (std::size_t l = 0; l != nn4; ++l, ++ijkl) {
                                    sum += buf[ijkl] * real_deriv_weight(weight_data[ijk + l]);
                                }
                            }
                        }
                    }
                    local_grad[atoms[component / 3] * 3 + component % 3] += degeneracy * sum;
                }
            }
        }
    };

    parallel_for_chunked(std::size_t{0}, pairs.size(), kernel);

    for (const auto& local_grad : local_grads) {
        for (std::size_t i = 0; i < local_grad.size(); ++i) {
            grad_data[i] += local_grad[i];
        }
    }

    libint2::finalize();
    return grad;
}

} // namespace ccpy
