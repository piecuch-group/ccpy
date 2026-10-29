#pragma once

#include <libint2.hpp>

#include "integrals/basis.h"

#include "helpers/ndarray.h"

namespace ccpy {

// Define a no-params struct to use as a template parameter
struct NoParams {};

template <libint2::Operator Op, std::size_t M, typename Params = NoParams>
[[nodiscard]] auto compute_one_electron_multi(const Basis& basis1, const Basis& basis2,
                                              Params const& params = Params{})
    -> std::array<np_matrix, M> {

    // Initialize libint2
    libint2::initialize();

    // Prepare engine
    const auto max_nprim = std::max(basis1.max_nprim(), basis2.max_nprim());
    const auto max_l = std::max(basis1.max_l(), basis2.max_l());
    libint2::Engine engine(Op, max_nprim, max_l);

    if constexpr (not std::is_same_v<Params, NoParams>) {
        engine.set_params(params);
    }

    const auto& results = engine.results();

    // Get the number of basis functions in each basis
    const std::size_t nb1 = basis1.size();
    const std::size_t nb2 = basis2.size();

    // Loop over shell pairs and fill each buffer
    auto nshells1 = basis1.nshells();
    auto nshells2 = basis2.nshells();

    // Get arrays of indices of the first basis in a shell and the size of each shell
    const auto first_size1 = basis1.shell_first_and_size();
    const auto first_size2 = basis2.shell_first_and_size();

    // Allocate M separate flat buffers
    // Wrap each buffer into an ndarray with memory management
    std::array<np_matrix, M> ints;
    for (std::size_t k = 0; k < M; ++k) {
        ints[k] = make_zeros<nb::numpy, double, 2>({nb1, nb2});
    }

    for (std::size_t s1 = 0; s1 < nshells1; ++s1) {
        const auto& shell1 = basis1[s1];
        const auto [f1, n1] = first_size1[s1];

        for (std::size_t s2 = 0; s2 < nshells2; ++s2) {
            const auto& shell2 = basis2[s2];
            const auto [f2, n2] = first_size2[s2];

            // Compute the integrals for this shell pair
            engine.compute(shell1, shell2);

            // Loop over the components of this operator and fill the buffers
            for (std::size_t k = 0; k < M; ++k) {
                if (const auto buf = results[k]; buf) {
                    auto ints_view = ints[k].view();
                    for (std::size_t i = 0, ij = 0; i != n1; ++i) {
                        for (std::size_t j = 0; j != n2; ++j, ++ij) {
                            ints_view(f1 + i, f2 + j) = static_cast<double>(buf[ij]);
                        }
                    }
                }
            }
        }
    }

    libint2::finalize();

    return ints;
}
} // namespace ccpy