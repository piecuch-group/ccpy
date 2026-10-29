#pragma once

#include <array>
#include <utility>
#include <vector>

#include "helpers/ndarray.h"

namespace ccpy {
class Basis;

/// @brief Contract mixed second derivatives of three-center Coulomb integrals into `opVop`.
/// @details Computes scalar and Cartesian Pauli components of the momentum-dressed
/// three-center integral. The first basis is summed over, while the two orbital-basis
/// indices are retained.
/// @param basis1 Potential-density basis.
/// @param basis2 First orbital basis.
/// @param basis3 Second orbital basis.
/// @return Four real matrices in scalar, x, y, z order.
std::array<np_matrix, 4> coulomb_3c_opVop(const Basis& basis1, const Basis& basis2,
                                          const Basis& basis3);

/// @brief Contract first derivatives of three-center Coulomb integrals with real weights.
/// @details Computes the gradient contribution
/// \f[
///   G_{A\alpha} =
///   \sum_{P\mu\nu} W^P_{\mu\nu}
///   \frac{\partial (P|\mu\nu)}{\partial R_{A\alpha}},
/// \f]
/// where \f$P\f$ belongs to @p basis1 and \f$\mu,\nu\f$ belong to @p basis2 and
/// @p basis3, respectively.  The derivative buffers for all three centers in
/// the shell triplet are mapped to the atom indices in @p charges.
/// @param basis1 Auxiliary basis for the one-index side of \f$(P|\mu\nu)\f$.
/// @param basis2 First orbital basis for the two-index side of \f$(P|\mu\nu)\f$.
/// @param basis3 Second orbital basis for the two-index side of \f$(P|\mu\nu)\f$.
/// @param W3 Real weights with shape `(basis1.size(), basis2.size(), basis3.size())`.
/// @param charges Nuclear charges and Cartesian centers.  Every basis center must match one
///        entry in this vector.
/// @return Real gradient vector of length `3 * charges.size()` in atom-major Cartesian order.
np_vector coulomb_3c_deriv(const Basis& basis1, const Basis& basis2, const Basis& basis3,
                           const np_tensor3_c& W3,
                           const std::vector<std::pair<double, std::array<double, 3>>>& charges);

/// @brief Contract first derivatives of three-center Coulomb integrals with complex weights.
/// @details Uses the same contraction as the real overload, accumulating only
/// `real(W3)` because the derivative integrals and nuclear gradient are real.
/// @param basis1 Auxiliary basis for the one-index side of \f$(P|\mu\nu)\f$.
/// @param basis2 First orbital basis for the two-index side of \f$(P|\mu\nu)\f$.
/// @param basis3 Second orbital basis for the two-index side of \f$(P|\mu\nu)\f$.
/// @param W3 Complex weights with shape `(basis1.size(), basis2.size(), basis3.size())`.
/// @param charges Nuclear charges and Cartesian centers.  Every basis center must match one
///        entry in this vector.
/// @return Real gradient vector of length `3 * charges.size()` in atom-major Cartesian order.
np_vector coulomb_3c_deriv(const Basis& basis1, const Basis& basis2, const Basis& basis3,
                           const np_tensor3_complex_c& W3,
                           const std::vector<std::pair<double, std::array<double, 3>>>& charges);

/// @brief Contract first derivatives of two-center Coulomb metric integrals with real weights.
/// @details Computes the gradient contribution
/// \f[
///   G_{A\alpha} =
///   \sum_{PQ} W^M_{PQ}
///   \frac{\partial (P|Q)}{\partial R_{A\alpha}},
/// \f]
/// where \f$P\f$ belongs to @p basis1 and \f$Q\f$ belongs to @p basis2.  The
/// derivative buffers for both centers in the shell pair are mapped to the atom
/// indices in @p charges.
/// @param basis1 First auxiliary basis for the metric integral \f$(P|Q)\f$.
/// @param basis2 Second auxiliary basis for the metric integral \f$(P|Q)\f$.
/// @param W2 Real weights with shape `(basis1.size(), basis2.size())`.
/// @param charges Nuclear charges and Cartesian centers.  Every basis center must match one
///        entry in this vector.
/// @return Real gradient vector of length `3 * charges.size()` in atom-major Cartesian order.
np_vector coulomb_2c_deriv(const Basis& basis1, const Basis& basis2, const np_matrix_c& W2,
                           const std::vector<std::pair<double, std::array<double, 3>>>& charges);

/// @brief Contract first derivatives of two-center Coulomb metric integrals with complex weights.
/// @details Uses the same contraction as the real overload, accumulating only
/// `real(W2)` because the derivative integrals and nuclear gradient are real.
/// @param basis1 First auxiliary basis for the metric integral \f$(P|Q)\f$.
/// @param basis2 Second auxiliary basis for the metric integral \f$(P|Q)\f$.
/// @param W2 Complex weights with shape `(basis1.size(), basis2.size())`.
/// @param charges Nuclear charges and Cartesian centers.  Every basis center must match one
///        entry in this vector.
/// @return Real gradient vector of length `3 * charges.size()` in atom-major Cartesian order.
np_vector coulomb_2c_deriv(const Basis& basis1, const Basis& basis2, const np_matrix_complex_c& W2,
                           const std::vector<std::pair<double, std::array<double, 3>>>& charges);

/// @brief Contract a four-index weight tensor W with the nuclear first derivatives of the
/// four-center integrals (mn|rs). W must have 8-fold permutational symmetry.
np_vector coulomb_4c_deriv(const Basis& basis, const np_tensor4_c& W4,
                           const std::vector<std::pair<double, std::array<double, 3>>>& charges);

} // namespace ccpy
