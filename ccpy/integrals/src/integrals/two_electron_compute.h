#pragma once
#include <future>
#include <libint2.hpp>

#include "helpers/ndarray.h"
#include "helpers/logger.h"
#include "helpers/parallel.h"

#include "integrals/basis.h"

namespace ccpy {

// Define a no-params struct to use as a template parameter
struct NoParams {};

template <libint2::Operator Op, typename Params = NoParams>
[[nodiscard]] auto compute_two_electron_4c_multi(const Basis& basis1, const Basis& basis2,
                                                 const Basis& basis3, const Basis& basis4,
                                                 Params const& params = Params{}) -> np_tensor4 {
    const auto start = std::chrono::high_resolution_clock::now();

    // Initialize libint2
    libint2::initialize();

    // Prepare engine
    const auto max_nprim = std::max(std::max(basis1.max_nprim(), basis2.max_nprim()),
                                    std::max(basis3.max_nprim(), basis4.max_nprim()));
    const auto max_l = std::max(std::max(basis1.max_l(), basis2.max_l()),
                                std::max(basis3.max_l(), basis4.max_l()));
    libint2::Engine engine(Op, max_nprim, max_l);

    if constexpr (not std::is_same_v<Params, NoParams>) {
        engine.set_params(params);
    }

    const auto& results = engine.results();

    // Get the number of basis functions in each basis
    const std::size_t nb1 = basis1.size();
    const std::size_t nb2 = basis2.size();
    const std::size_t nb3 = basis3.size();
    const std::size_t nb4 = basis4.size();

    // Get the number of shells in each basis
    auto nshells1 = basis1.nshells();
    auto nshells2 = basis2.nshells();
    auto nshells3 = basis3.nshells();
    auto nshells4 = basis4.nshells();

    // Get arrays of indices of the first basis in a shell and the size of each shell
    const auto first_size1 = basis1.shell_first_and_size();
    const auto first_size2 = basis2.shell_first_and_size();
    const auto first_size3 = basis3.shell_first_and_size();
    const auto first_size4 = basis4.shell_first_and_size();

    // Allocate a four index tensor
    auto ints = make_zeros<nb::numpy, double, 4>({nb1, nb2, nb3, nb4});
    auto v = ints.view();

    // Loop over shell quartets and fill each buffer
    for (std::size_t s1 = 0; s1 < nshells1; ++s1) {
        const auto& shell1 = basis1[s1];
        const auto [f1, n1] = first_size1[s1];

        for (std::size_t s2 = 0; s2 < nshells2; ++s2) {
            const auto& shell2 = basis2[s2];
            const auto [f2, n2] = first_size2[s2];

            for (std::size_t s3 = 0; s3 < nshells3; ++s3) {
                const auto& shell3 = basis3[s3];
                const auto [f3, n3] = first_size3[s3];

                for (std::size_t s4 = 0; s4 < nshells4; ++s4) {
                    const auto& shell4 = basis4[s4];

                    // Compute the integrals for this shell pair
                    engine.compute(shell1, shell2, shell3, shell4);

                    // Loop over the components of this operator and fill the buffers
                    if (const auto buf = results[0]; buf) {
                        const auto [f4, n4] = first_size4[s4];
                        for (std::size_t i = 0, ijkl = 0; i != n1; ++i) {
                            for (std::size_t j = 0; j != n2; ++j) {
                                for (std::size_t k = 0; k != n3; ++k) {
                                    for (std::size_t l = 0; l != n4; ++l, ++ijkl) {
                                        v(f1 + i, f2 + j, f3 + k, f4 + l) =
                                            static_cast<double>(buf[ijkl]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Finalize libint2
    libint2::finalize();

    const auto end = std::chrono::high_resolution_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    LOG_INFO2 << "[ccpy] Two-electron integrals timing: " << elapsed.count() << " ms\n";

    return ints;
}

template <libint2::Operator Op, typename Params = NoParams>
[[nodiscard]] auto compute_two_electron_3c_multi(const Basis& basis1, const Basis& basis2,
                                                 const Basis& basis3,
                                                 Params const& params = Params{}) -> np_tensor3 {
    const auto start = std::chrono::high_resolution_clock::now();

    // Initialize libint2
    libint2::initialize();

    const auto max_nprim =
        std::max(std::max(basis1.max_nprim(), basis2.max_nprim()), basis3.max_nprim());
    const auto max_l = std::max(std::max(basis1.max_l(), basis2.max_l()), basis3.max_l());

    const std::size_t nb1 = basis1.size();
    const std::size_t nb2 = basis2.size();
    const std::size_t nb3 = basis3.size();

    const std::size_t nshells1 = basis1.nshells();
    const std::size_t nshells2 = basis2.nshells();
    const std::size_t nshells3 = basis3.nshells();

    const auto first_size1 = basis1.shell_first_and_size();
    const auto first_size2 = basis2.shell_first_and_size();
    const auto first_size3 = basis3.shell_first_and_size();

    auto ints = make_zeros<nb::numpy, double, 3>({nb1, nb2, nb3});
    auto data = ints.data();

    /// This lambda function computes the integrals for a given range of shells
    /// in the first basis and fills the buffer.
    auto kernel_sym = [&](std::size_t s1_begin, std::size_t s1_end) {
        libint2::Engine engine(Op, max_nprim, max_l);
        engine.set(libint2::BraKet::xs_xx);
        if constexpr (!std::is_same_v<Params, NoParams>) {
            engine.set_params(params);
        }
        const auto& results = engine.results();

        // Loop over the given range of shells in basis1
        for (std::size_t s1 = s1_begin; s1 < s1_end; ++s1) {
            const auto& shell1 = basis1[s1];
            const auto [f1, n1] = first_size1[s1];

            // Loop over the shells in basis2 and basis3
            for (std::size_t s2 = 0; s2 < nshells2; ++s2) {
                const auto& shell2 = basis2[s2];
                const auto [f2, n2] = first_size2[s2];
                for (std::size_t s3 = 0; s3 <= s2; ++s3) {
                    const auto& shell3 = basis3[s3];

                    // Compute the integrals for this shell triplet
                    engine.compute(shell1, shell2, shell3);
                    auto buf = results[0];
                    if (!buf)
                        continue;

                    const auto [f3, n3] = first_size3[s3];

                    std::size_t offset = f1 * nb2 * nb3 + f2 * nb3 + f3;     // for s3 <= s2
                    std::size_t offset_sym = f1 * nb3 * nb2 + f3 * nb2 + f2; // for s3 > s2

                    // Loop over the components of this operator and fill the buffers
                    if (s2 != s3) {
                        for (std::size_t i = 0, ijk = 0; i < n1; ++i) {
                            for (std::size_t j = 0; j < n2; ++j) {
                                for (std::size_t k = 0; k < n3; ++k, ++ijk) {
                                    const auto val = static_cast<double>(buf[ijk]);
                                    data[offset + i * nb2 * nb3 + j * nb3 + k] = val;
                                    data[offset_sym + i * nb3 * nb2 + k * nb2 + j] = val;
                                }
                            }
                        }
                    } else {
                        for (std::size_t i = 0, ijk = 0; i < n1; ++i) {
                            for (std::size_t j = 0; j < n2; ++j) {
                                for (std::size_t k = 0; k < n3; ++k, ++ijk) {
                                    data[offset + i * nb2 * nb3 + j * nb3 + k] =
                                        static_cast<double>(buf[ijk]);
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    auto kernel_nosym = [&](std::size_t s1_begin, std::size_t s1_end) {
        libint2::Engine engine(Op, max_nprim, max_l);
        engine.set(libint2::BraKet::xs_xx);
        if constexpr (!std::is_same_v<Params, NoParams>) {
            engine.set_params(params);
        }
        const auto& results = engine.results();

        // Loop over the given range of shells in basis1
        for (std::size_t s1 = s1_begin; s1 < s1_end; ++s1) {
            const auto& shell1 = basis1[s1];
            const auto [f1, n1] = first_size1[s1];

            // Loop over the shells in basis2 and basis3
            for (std::size_t s2 = 0; s2 < nshells2; ++s2) {
                const auto& shell2 = basis2[s2];
                const auto [f2, n2] = first_size2[s2];
                for (std::size_t s3 = 0; s3 < nshells3; ++s3) {
                    const auto& shell3 = basis3[s3];

                    // Compute the integrals for this shell triplet
                    engine.compute(shell1, shell2, shell3);
                    auto buf = results[0];
                    if (!buf)
                        continue;

                    const auto [f3, n3] = first_size3[s3];

                    std::size_t offset = f1 * nb2 * nb3 + f2 * nb3 + f3;

                    for (std::size_t i = 0, ijk = 0; i < n1; ++i) {
                        for (std::size_t j = 0; j < n2; ++j) {
                            for (std::size_t k = 0; k < n3; ++k, ++ijk) {
                                data[offset + i * nb2 * nb3 + j * nb3 + k] =
                                    static_cast<double>(buf[ijk]);
                            }
                        }
                    }
                }
            }
        }
    };

    if (&basis2 == &basis3) {
        parallel_for_chunked(nshells1, kernel_sym);
    } else {
        parallel_for_chunked(nshells1, kernel_nosym);
    }

    // Finalize libint2
    libint2::finalize();

    const auto end = std::chrono::high_resolution_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    LOG_INFO2 << "[ccpy] Three-center two-electron integrals timing: " << elapsed.count() / 1000.0
              << " s\n";

    return ints;
}

template <libint2::Operator Op, typename Params = NoParams>
void compute_two_electron_3c_by_shell(
    const Basis& basis1, const Basis& basis2, const Basis& basis3,
    const std::array<std::pair<std::size_t, std::size_t>, 3>& shell_slices, np_tensor3_c& buffer,
    Params const& params = Params{}) {
    const auto start = std::chrono::high_resolution_clock::now();

    if (shell_slices[0].first >= shell_slices[0].second ||
        shell_slices[1].first >= shell_slices[1].second ||
        shell_slices[2].first >= shell_slices[2].second) {
        throw std::invalid_argument(
            "shell_slices indices must be in the form of (start, end) with start < end");
    }

    if (shell_slices[0].second > basis1.nshells() || shell_slices[1].second > basis2.nshells() ||
        shell_slices[2].second > basis3.nshells()) {
        throw std::invalid_argument(
            "shell_slices indices must be within the number of shells in each basis");
    }

    const auto [ish0, ish1] = shell_slices[0];
    const auto [jsh0, jsh1] = shell_slices[1];
    const auto [ksh0, ksh1] = shell_slices[2];

    const auto first_size1 = basis1.shell_first_and_size();
    const auto first_size2 = basis2.shell_first_and_size();
    const auto first_size3 = basis3.shell_first_and_size();

    const auto offsets1 = basis1.shell_offsets();
    const auto offsets2 = basis2.shell_offsets();
    const auto offsets3 = basis3.shell_offsets();

    // nb = index of first basis function in last shell - index of first basis function in first
    // shell + size of last shell
    const std::size_t nb1 = offsets1[ish1] - offsets1[ish0];
    const std::size_t nb2 = offsets2[jsh1] - offsets2[jsh0];
    const std::size_t nb3 = offsets3[ksh1] - offsets3[ksh0];

    const std::size_t first1 = first_size1[ish0].first;
    const std::size_t first2 = first_size2[jsh0].first;
    const std::size_t first3 = first_size3[ksh0].first;

    if (buffer.shape(0) < nb1 || buffer.shape(1) < nb2 || buffer.shape(2) < nb3) {
        std::string msg = "Buffer shape (" + std::to_string(buffer.shape(0)) + ", " +
                          std::to_string(buffer.shape(1)) + ", " + std::to_string(buffer.shape(2)) +
                          ") is too small for the given shell slices with sizes (" +
                          std::to_string(nb1) + ", " + std::to_string(nb2) + ", " +
                          std::to_string(nb3) + ")";
        throw std::invalid_argument(msg);
    }

    auto data = buffer.data();

    // Initialize libint2
    libint2::initialize();

    const auto max_nprim =
        std::max(std::max(basis1.max_nprim(), basis2.max_nprim()), basis3.max_nprim());
    const auto max_l = std::max(std::max(basis1.max_l(), basis2.max_l()), basis3.max_l());

    // strides to compute the offset in the buffer for a given shell triplet
    const std::size_t stride1 = buffer.shape(0);
    const std::size_t stride2 = buffer.shape(1);
    const std::size_t stride3 = buffer.shape(2);

    const std::size_t nish = ish1 - ish0;
    const std::size_t njsh = jsh1 - jsh0;

    /// This lambda function computes the integrals for a given range of shells
    /// in the first basis and fills the buffer.
    auto kernel = [&](std::size_t idx_begin, std::size_t idx_end) {
        libint2::Engine engine(Op, max_nprim, max_l);
        engine.set(libint2::BraKet::xs_xx);
        if constexpr (!std::is_same_v<Params, NoParams>) {
            engine.set_params(params);
        }
        const auto& results = engine.results();

        // Loop over the given range of composite ij indices
        for (std::size_t idx = idx_begin; idx < idx_end; ++idx) {
            const std::size_t s1 = ish0 + idx / njsh;
            const std::size_t s2 = jsh0 + idx % njsh;

            const auto& shell1 = basis1[s1];
            auto [f1, n1] = first_size1[s1];
            f1 -= first1; // adjust f1 to be the index in the sliced basis

            const auto& shell2 = basis2[s2];
            auto [f2, n2] = first_size2[s2];
            f2 -= first2; // adjust f2 to be the index in the sliced basis

            for (std::size_t s3 = ksh0; s3 < ksh1; ++s3) {
                const auto& shell3 = basis3[s3];

                // Compute the integrals for this shell triplet
                engine.compute(shell1, shell2, shell3);
                auto buf = results[0];
                if (!buf)
                    continue;

                auto [f3, n3] = first_size3[s3];
                f3 -= first3; // adjust f3 to be the index in the sliced basis

                const std::size_t offset = f1 * stride2 * stride3 + f2 * stride3 + f3;

                for (std::size_t i = 0, ijk = 0; i < n1; ++i) {
                    for (std::size_t j = 0; j < n2; ++j) {
                        for (std::size_t k = 0; k < n3; ++k, ++ijk) {
                            data[offset + i * stride2 * stride3 + j * stride3 + k] =
                                static_cast<double>(buf[ijk]);
                        }
                    }
                }
            }
        }
    };

    parallel_for_chunked(0, nish * njsh, kernel);

    // Finalize libint2
    libint2::finalize();

    const auto end = std::chrono::high_resolution_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    LOG_INFO2 << "[ccpy] Three-center two-electron integrals timing: " << elapsed.count() / 1000.0
              << " s\n";
}

template <libint2::Operator Op, typename Params = NoParams>
[[nodiscard]] auto compute_two_electron_3c_by_shell(
    const Basis& basis1, const Basis& basis2, const Basis& basis3,
    const std::array<std::pair<std::size_t, std::size_t>, 3>& shell_slices,
    Params const& params = Params{}) -> np_tensor3_c {
    if (shell_slices.size() != 3) {
        throw std::invalid_argument("shell_slices must have size 3");
    }

    if (shell_slices[0].first >= shell_slices[0].second ||
        shell_slices[1].first >= shell_slices[1].second ||
        shell_slices[2].first >= shell_slices[2].second) {
        throw std::invalid_argument(
            "shell_slices indices must be in the form of (start, end) with start < end");
    }

    if (shell_slices[0].second > basis1.nshells() || shell_slices[1].second > basis2.nshells() ||
        shell_slices[2].second > basis3.nshells()) {
        throw std::invalid_argument(
            "shell_slices indices must be within the number of shells in each basis");
    }

    const std::size_t ish0 = shell_slices[0].first;
    const std::size_t ish1 = shell_slices[0].second;
    const std::size_t jsh0 = shell_slices[1].first;
    const std::size_t jsh1 = shell_slices[1].second;
    const std::size_t ksh0 = shell_slices[2].first;
    const std::size_t ksh1 = shell_slices[2].second;

    const auto first_size1 = basis1.shell_first_and_size();
    const auto first_size2 = basis2.shell_first_and_size();
    const auto first_size3 = basis3.shell_first_and_size();

    // nb = index of first basis function in last shell - index of first basis function in first
    // shell + size of last shell
    const std::size_t nb1 =
        first_size1[ish1 - 1].first - first_size1[ish0].first + first_size1[ish1 - 1].second;
    const std::size_t nb2 =
        first_size2[jsh1 - 1].first - first_size2[jsh0].first + first_size2[jsh1 - 1].second;
    const std::size_t nb3 =
        first_size3[ksh1 - 1].first - first_size3[ksh0].first + first_size3[ksh1 - 1].second;

    const std::size_t first1 = first_size1[ish0].first;
    const std::size_t first2 = first_size2[jsh0].first;
    const std::size_t first3 = first_size3[ksh0].first;

    auto buffer = make_zeros<nb::numpy, double, 3, nb::c_contig>({nb1, nb2, nb3});
    compute_two_electron_3c_by_shell<Op>(basis1, basis2, basis3, shell_slices, buffer, params);
    return buffer;
}

template <libint2::Operator Op, typename Params = NoParams>
[[nodiscard]] auto compute_two_electron_2c_multi(const Basis& basis1, const Basis& basis2,
                                                 Params const& params = Params{}) -> np_matrix {
    const auto start = std::chrono::high_resolution_clock::now();

    // Initialize libint2
    libint2::initialize();

    // Prepare engine
    const auto max_nprim = std::max(basis1.max_nprim(), basis2.max_nprim());
    const auto max_l = std::max(basis1.max_l(), basis2.max_l());
    libint2::Engine engine(Op, max_nprim, max_l);
    engine.set(libint2::BraKet::xs_xs);

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
    const auto first_size_1 = basis1.shell_first_and_size();
    const auto first_size_2 = basis2.shell_first_and_size();

    auto ints = make_zeros<nb::numpy, double, 2>({nb1, nb2});
    auto v = ints.view();

    // Loop over the shell pairs and fill the buffer
    for (std::size_t s1 = 0; s1 < nshells1; ++s1) {
        const auto& shell1 = basis1[s1];
        const auto [f1, n1] = first_size_1[s1];

        // Loop over the shells in basis2
        for (std::size_t s2 = 0; s2 < nshells2; ++s2) {
            const auto& shell2 = basis2[s2];

            // Compute the integrals for this shell pair
            engine.compute(shell1, shell2);

            // Loop over the components of this operator and fill the buffers
            if (const auto buf = results[0]; buf) {
                const auto [f2, n2] = first_size_2[s2];
                for (std::size_t i = 0, ij = 0; i != n1; ++i) {
                    for (std::size_t j = 0; j != n2; ++j, ++ij) {
                        v(f1 + i, f2 + j) = static_cast<double>(buf[ij]);
                    }
                }
            }
        }
    }

    // Finalize libint2
    libint2::finalize();

    const auto end = std::chrono::high_resolution_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    LOG_INFO2 << "[ccpy] Two-center two-electron integrals timing: " << elapsed.count() / 1000.0
              << " s\n";

    return ints;
}

} // namespace ccpy