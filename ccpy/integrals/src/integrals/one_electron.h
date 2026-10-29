#pragma once

#include <array>

#include "helpers/ndarray.h"

namespace nb = nanobind;

namespace ccpy {
class Basis;

/// @brief Compute the overlap integrals between two basis sets (<b1|b2>).
/// @param basis1 The basis set in the bra.
/// @param basis2 The basis set in the ket.
/// @return A 2D ndarray of shape (n1, n2), where n1 is the number of basis functions in
///         basis1 and n2 is the number of basis functions in basis2.
np_matrix overlap(const Basis& basis1, const Basis& basis2);

/// @brief Compute the kinetic energy integrals between two basis sets (<b1|-1/2 ∇^2|b2>).
/// @param basis1 The basis set in the bra.
/// @param basis2 The basis set in the ket.
/// @return A 2D ndarray of shape (n1, n2), where n1 is the number of basis functions in
///         basis1 and n2 is the number of basis functions in basis2.
np_matrix kinetic(const Basis& basis1, const Basis& basis2);

/// @brief Compute the nuclear attraction integrals between two basis sets (<b1|V|b2>).
/// @param basis1 The basis set in the bra.
/// @param basis2 The basis set in the ket.
/// @param charges A vector of pairs of charge and position.
/// @return A 2D ndarray of shape (n1, n2), where n1 is the number of basis functions in
///         basis1 and n2 is the number of basis functions in basis2.
np_matrix nuclear(const Basis& basis1, const Basis& basis2,
                  std::vector<std::pair<double, std::array<double, 3>>>& charges);

/// @brief Compute the multipolte integrals between two basis sets up to order 1
///        (<b1|op|b2>, with op = 1, x, y, z).
/// @param basis1 The basis set in the bra.
/// @param basis2 The basis set in the ket.
/// @param origin The origin of the multipole expansion.
/// @return A 4-element array of 2D ndarrays, each of shape (n1, n2), where n1 is the number
///         of basis functions in basis1 and n2 is the number of basis functions in basis2.
///         The elements of the array correspond to the operators
///         emultipole1[0] = <b1|b2>
///         emultipole1[1] = <b1|x|b2>
///         emultipole1[2] = <b1|y|b2>
///         emultipole1[3] = <b1|z|b2>
std::array<np_matrix, 4> emultipole1(const Basis& basis1, const Basis& basis2,
                                     std::array<double, 3>& origin);

/// @brief Compute the multipolte integrals between two basis sets up to order 2
///        (<b1|op|b2>, with op = 1, x, y, z, xx, xy, xz, yy, yz, zz).
/// @param basis1 The basis set in the bra.
/// @param basis2 The basis set in the ket.
/// @param origin The origin of the multipole expansion.
/// @return A 10-element array of 2D ndarrays, each of shape (n1, n2), where n1 is the number
///         of basis functions in basis1 and n2 is the number of basis functions in basis2.
///         The elements of the array correspond to the operators
///         emultipole2[0] = <b1|b2>
///         emultipole2[1] = <b1|x|b2>
///         emultipole2[2] = <b1|y|b2>
///         emultipole2[3] = <b1|z|b2>
///         emultipole2[4] = <b1|xx|b2>
///         emultipole2[5] = <b1|xy|b2>
///         emultipole2[6] = <b1|xz|b2>
///         emultipole2[7] = <b1|yy|b2>
///         emultipole2[8] = <b1|yz|b2>
///         emultipole2[9] = <b1|zz|b2>
std::array<np_matrix, 10> emultipole2(const Basis& basis1, const Basis& basis2,
                                      std::array<double, 3>& origin);

/// @brief Compute the multipolte integrals between two basis sets up to order 3
///        (<b1|op|b2>, with op = 1, x, y, z, xx, xy, xz, yy, yz, zz, xxx, xxy, xxz, xyy, xyz, xzz,
///        yyy, yyz, yzz, zzz).
/// @param basis1 The basis set in the bra.
/// @param basis2 The basis set in the ket.
/// @param origin The origin of the multipole expansion.
/// @return A 20-element array of 2D ndarrays, each of shape (n1, n2), where n1 is the number
///         of basis functions in basis1 and n2 is the number of basis functions in basis2.
///         The elements of the array correspond to the operators
///         emultipole3[0] = <b1|b2>
///         emultipole3[1] = <b1|x|b2>
///         emultipole3[2] = <b1|y|b2>
///         emultipole3[3] = <b1|z|b2>
///         emultipole3[4] = <b1|xx|b2>
///         emultipole3[5] = <b1|xy|b2>
///         emultipole3[6] = <b1|xz|b2>
///         emultipole3[7] = <b1|yy|b2>
///         emultipole3[8] = <b1|yz|b2>
///         emultipole3[9] = <b1|zz|b2>
///         emultipole3[10] = <b1|xxx|b2>
///         emultipole3[11] = <b1|xxy|b2>
///         emultipole3[12] = <b1|xxz|b2>
///         emultipole3[13] = <b1|xyy|b2>
///         emultipole3[14] = <b1|xyz|b2>
///         emultipole3[15] = <b1|xzz|b2>
///         emultipole3[16] = <b1|yyy|b2>
///         emultipole3[17] = <b1|yyz|b2>
///         emultipole3[18] = <b1|yzz|b2>
///         emultipole3[19] = <b1|zzz|b2>
std::array<np_matrix, 20> emultipole3(const Basis& basis1, const Basis& basis2,
                                      std::array<double, 3>& origin);

/// @brief Compute the small-component nuclear potential  (sigma p ) V (sigma p) between two basis
///        sets (<b1|op|b2>, with op = p.Vp, (p x Vp)_x, (p x Vp)_y, (p x Vp)_z).
/// @param basis1 The basis set in the bra.
/// @param basis2 The basis set in the ket.
/// @param origin The origin of the multipole expansion.
/// @return A 4-element array of 2D ndarrays, each of shape (n1, n2), where n1 is the number
///         of basis functions in basis1 and n2 is the number of basis functions in basis2.
///         The elements of the array correspond to the operators
///         opVop[0] = <b1|p.Vp|b2>
///         opVop[1] = <b1|(p x Vp)_x|b2>
///         opVop[2] = <b1|(p x Vp)_y|b2>
///         opVop[3] = <b1|(p x Vp)_z|b2>
std::array<np_matrix, 4> opVop(const Basis& basis1, const Basis& basis2,
                               std::vector<std::pair<double, std::array<double, 3>>>& charges);

/// @brief Compute the nuclear attraction integrals with the error function attenuation
///        between two basis sets (<b1|V|b2>, with
///             V = \sum_a Z_a erf(-omega * |r - R_a|) / |r - R_a|).
/// @param basis1 The basis set in the bra.
/// @param basis2 The basis set in the ket.
/// @param omega_charges A tuple of the form (alpha, charges), where alpha is the
///                      attenuation parameter and charges is a vector of pairs of charge
///                      and position.
/// @return A 2D ndarray of shape (n1, n2), where n1 is the number of basis functions in
///         basis1 and n2 is the number of basis functions in basis2.
np_matrix erf_nuclear(
    const Basis& basis1, const Basis& basis2,
    std::tuple<double, std::vector<std::pair<double, std::array<double, 3>>>>& omega_charges);

/// @brief Compute the nuclear attraction integrals with the complementary error function
///        between two basis sets (<b1|V|b2>, with
///             V = \sum_a Z_a erfc(-omega * |r - R_a|) / |r - R_a|).
/// @param basis1 The basis set in the bra.
/// @param basis2 The basis set in the ket.
/// @param omega_charges A tuple of the form (alpha, charges), where alpha is the
///                      attenuation parameter and charges is a vector of pairs of charge
///                      and position.
/// @return A 2D ndarray of shape (n1, n2), where n1 is the number of basis functions in
///         basis1 and n2 is the number of basis functions in basis2.
np_matrix erfc_nuclear(
    const Basis& basis1, const Basis& basis2,
    std::tuple<double, std::vector<std::pair<double, std::array<double, 3>>>>& omega_charges);

} // namespace ccpy