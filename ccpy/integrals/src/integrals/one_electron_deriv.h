#pragma once

#include <array>
#include <utility>
#include <vector>

#include "helpers/ndarray.h"

namespace nb = nanobind;

namespace ccpy {
class Basis;

/// @brief Contract first derivatives of overlap integrals with a density-like matrix.
/// @details Computes the gradient contribution
/// \f[
///   G_{A\alpha} =
///   \sum_{\mu\nu} D_{\mu\nu}
///   \frac{\partial S_{\mu\nu}}{\partial R_{A\alpha}},
/// \f]
/// where \f$S_{\mu\nu} = \langle \chi^1_\mu | \chi^2_\nu \rangle\f$.
/// Derivative buffers for both basis-function centers are mapped to the atom
/// indices in @p charges.
/// @param basis1 First orbital basis in the overlap integral.
/// @param basis2 Second orbital basis in the overlap integral.
/// @param dm Real contraction matrix with shape `(basis1.size(), basis2.size())`.
/// @param charges Nuclear charges and Cartesian centers.  Every basis center must match one
///        entry in this vector.
/// @return Real gradient vector of length `3 * charges.size()` in atom-major Cartesian order.
np_vector overlap_deriv(const Basis& basis1, const Basis& basis2, const np_matrix& dm,
                        const std::vector<std::pair<double, std::array<double, 3>>>& charges);

/// @brief Contract first derivatives of kinetic-energy integrals with a density-like matrix.
/// @details Computes the gradient contribution
/// \f[
///   G_{A\alpha} =
///   \sum_{\mu\nu} D_{\mu\nu}
///   \frac{\partial T_{\mu\nu}}{\partial R_{A\alpha}},
/// \f]
/// where \f$T_{\mu\nu} =
/// \langle \chi^1_\mu | -\frac{1}{2}\nabla^2 | \chi^2_\nu \rangle\f$.
/// Derivative buffers for both basis-function centers are mapped to the atom
/// indices in @p charges.
/// @param basis1 First orbital basis in the kinetic integral.
/// @param basis2 Second orbital basis in the kinetic integral.
/// @param dm Real contraction matrix with shape `(basis1.size(), basis2.size())`.
/// @param charges Nuclear charges and Cartesian centers.  Every basis center must match one
///        entry in this vector.
/// @return Real gradient vector of length `3 * charges.size()` in atom-major Cartesian order.
np_vector kinetic_deriv(const Basis& basis1, const Basis& basis2, const np_matrix& dm,
                        const std::vector<std::pair<double, std::array<double, 3>>>& charges);

/// @brief Contract first derivatives of nuclear-attraction integrals with a density-like matrix.
/// @details Computes the gradient contribution
/// \f[
///   G_{A\alpha} =
///   \sum_{\mu\nu} D_{\mu\nu}
///   \frac{\partial V_{\mu\nu}}{\partial R_{A\alpha}},
/// \f]
/// where \f$V_{\mu\nu}\f$ is the nuclear-attraction integral over @p charges.
/// The contraction includes derivative buffers for the two basis-function
/// centers and the explicit derivatives with respect to all nuclear charge
/// centers.
/// @param basis1 First orbital basis in the nuclear-attraction integral.
/// @param basis2 Second orbital basis in the nuclear-attraction integral.
/// @param dm Real contraction matrix with shape `(basis1.size(), basis2.size())`.
/// @param charges Nuclear charges and Cartesian centers.  Every basis center must match one
///        entry in this vector.
/// @return Real gradient vector of length `3 * charges.size()` in atom-major Cartesian order.
np_vector nuclear_deriv(const Basis& basis1, const Basis& basis2, const np_matrix& dm,
                        const std::vector<std::pair<double, std::array<double, 3>>>& charges);

} // namespace ccpy
