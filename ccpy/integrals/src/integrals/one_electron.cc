#include <span>

#include "integrals/one_electron.h"
#include "integrals/one_electron_compute.h"

namespace ccpy {
np_matrix overlap(const Basis& b1, const Basis& b2) {
    return compute_one_electron_multi<libint2::Operator::overlap, 1>(b1, b2)[0];
}

np_matrix kinetic(const Basis& b1, const Basis& b2) {
    return compute_one_electron_multi<libint2::Operator::kinetic, 1>(b1, b2)[0];
}

np_matrix nuclear(const Basis& basis1, const Basis& basis2,
                  std::vector<std::pair<double, std::array<double, 3>>>& charges) {
    return compute_one_electron_multi<libint2::Operator::nuclear, 1>(basis1, basis2, charges)[0];
}

std::array<np_matrix, 4> emultipole1(const Basis& basis1, const Basis& basis2,
                                     std::array<double, 3>& origin) {
    return compute_one_electron_multi<libint2::Operator::emultipole1, 4>(basis1, basis2, origin);
}

std::array<np_matrix, 10> emultipole2(const Basis& basis1, const Basis& basis2,
                                      std::array<double, 3>& origin) {
    return compute_one_electron_multi<libint2::Operator::emultipole2, 10>(basis1, basis2, origin);
}

std::array<np_matrix, 20> emultipole3(const Basis& basis1, const Basis& basis2,
                                      std::array<double, 3>& origin) {
    return compute_one_electron_multi<libint2::Operator::emultipole3, 20>(basis1, basis2, origin);
}

std::array<np_matrix, 4> opVop(const Basis& basis1, const Basis& basis2,
                               std::vector<std::pair<double, std::array<double, 3>>>& charges) {
    return compute_one_electron_multi<libint2::Operator::opVop, 4>(basis1, basis2, charges);
}

np_matrix erf_nuclear(
    const Basis& basis1, const Basis& basis2,
    std::tuple<double, std::vector<std::pair<double, std::array<double, 3>>>>& omega_charges) {
    return compute_one_electron_multi<libint2::Operator::erf_nuclear, 1>(basis1, basis2,
                                                                         omega_charges)[0];
}

np_matrix erfc_nuclear(
    const Basis& basis1, const Basis& basis2,
    std::tuple<double, std::vector<std::pair<double, std::array<double, 3>>>>& omega_charges) {
    return compute_one_electron_multi<libint2::Operator::erfc_nuclear, 1>(basis1, basis2,
                                                                          omega_charges)[0];
}

} // namespace ccpy