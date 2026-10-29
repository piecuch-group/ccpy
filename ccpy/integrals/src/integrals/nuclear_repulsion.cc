#include <array>
#include <utility>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <cassert>

namespace ccpy {

double nuclear_repulsion(std::vector<std::pair<double, std::array<double, 3>>>& charges) {
    double enuc = 0.0;
    for (size_t i = 0; const auto& [Zi, xyz_i] : charges) {
        for (size_t j = 0; const auto& [Zj, xyz_j] : charges) {
            if (j >= i) {
                continue;
            }
            const double r =
                std::hypot(xyz_i[0] - xyz_j[0], xyz_i[1] - xyz_j[1], xyz_i[2] - xyz_j[2]);
            if (r < 1e-10) {
                throw std::runtime_error("Nuclear repulsion: zero distance between charges");
            }
            ++j;
            // Coulomb's law
            enuc += Zi * Zj / r;
        }
        ++i;
    }
    return enuc;
}

} // namespace ccpy