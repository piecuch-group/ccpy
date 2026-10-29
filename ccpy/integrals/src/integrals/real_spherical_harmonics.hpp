#pragma once

namespace ccpy::ints {

// Values of square roots of small integers computed with Mathematica and 20 digits of precision
constexpr double sqrt_3 = 1.7320508075688772935;
constexpr double sqrt_5 = 2.2360679774997896964;
constexpr double sqrt_6 = 2.4494897427831780982;
constexpr double sqrt_7 = 2.6457513110645905905;
constexpr double sqrt_10 = 3.1622776601683793320;
constexpr double sqrt_14 = 3.7416573867739413856;
constexpr double sqrt_15 = 3.8729833462074168852;
constexpr double sqrt_21 = 4.5825756949558400066;
constexpr double sqrt_35 = 5.9160797830996160426;
constexpr double sqrt_42 = 6.4807406984078602310;
constexpr double sqrt_70 = 8.3666002653407554798;
constexpr double sqrt_105 = 10.246950765959598383;
constexpr double sqrt_154 = 12.409673645990856596;
constexpr double sqrt_210 = 14.491376746189438574;
constexpr double sqrt_231 = 15.198684153570663632;
constexpr double sqrt_429 = 20.712315177207979132;
constexpr double sqrt_462 = 21.494185260204677039;
constexpr double sqrt_6006 = 77.498387079990251885;

// Functions to evaluate the real spherical harmonics as defined in the libint2 library
// see https://en.wikipedia.org/wiki/Solid_harmonics#Real_form

void compute_real_spherical_harmonic_0(double x, double y, double z, double* buffer) {
    // Compute the spherical harmonic for l = 0
    buffer[0] = 1.0;
}

void compute_real_spherical_harmonic_1(double x, double y, double z, double* buffer) {
    // Compute the spherical harmonic for l = 1
    buffer[0] = y;
    buffer[1] = z;
    buffer[2] = x;
}

void compute_real_spherical_harmonic_2(double x, double y, double z, double* buffer) {
    // Compute the spherical harmonic for l = 2
    buffer[0] = sqrt_3 * x * y;
    buffer[1] = sqrt_3 * y * z;
    buffer[2] = (z * z - 0.5 * (x * x + y * y));
    buffer[3] = sqrt_3 * x * z;
    buffer[4] = 0.5 * sqrt_3 * (x * x - y * y);
}

void compute_real_spherical_harmonic_3(double x, double y, double z, double* buffer) {
    // Compute the spherical harmonic for l = 3
    buffer[0] = sqrt_10 * y * (0.75 * x * x - 0.25 * y * y);
    buffer[1] = sqrt_15 * x * y * z;
    buffer[2] = sqrt_6 * y * (z * z - 0.25 * (x * x + y * y));
    buffer[3] = z * (z * z - 1.5 * (x * x + y * y));
    buffer[4] = sqrt_6 * x * (z * z - 0.25 * (x * x + y * y));
    buffer[5] = 0.5 * sqrt_15 * z * (x * x - y * y);
    buffer[6] = sqrt_10 * x * (0.25 * x * x - 0.75 * y * y);
}

void compute_real_spherical_harmonic_4(double x, double y, double z, double* buffer) {
    // Compute the spherical harmonic for l = 4
    buffer[0] = 0.5 * sqrt_35 * x * y * (x * x - y * y);
    buffer[1] = 0.25 * sqrt_70 * y * z * (3 * x * x - y * y);
    buffer[2] = sqrt_5 * x * y * (-0.5 * x * x - 0.5 * y * y + 3.0 * z * z);
    buffer[3] = sqrt_10 * y * z * (-0.75 * x * x - 0.75 * y * y + z * z);
    buffer[4] = (4.375 * std::pow(z, 4) - 3.75 * z * z * (x * x + y * y + z * z) +
                 0.375 * std::pow(x * x + y * y + z * z, 2));
    buffer[5] = sqrt_10 * x * z * (-0.75 * x * x - 0.75 * y * y + z * z);
    buffer[6] = -1.0 / 30.0 * sqrt_5 * (x * x - y * y) * (7.5 * x * x + 7.5 * y * y - 45.0 * z * z);
    buffer[7] = 0.25 * sqrt_70 * x * z * (x * x - 3 * y * y);
    buffer[8] = sqrt_35 * (0.125 * std::pow(x, 4) - 0.75 * x * x * y * y + 0.125 * std::pow(y, 4));
}

void compute_real_spherical_harmonic_5(double x, double y, double z, double* buffer) {
    // Compute the spherical harmonic for l = 5
    buffer[0] =
        sqrt_14 * y * (0.9375 * std::pow(x, 4) - 1.875 * x * x * y * y + 0.1875 * std::pow(y, 4));
    buffer[1] = 1.5 * sqrt_35 * x * y * z * (x * x - y * y);
    buffer[2] = -1.0 / 840.0 * sqrt_70 * y * (3 * x * x - y * y) *
                (52.5 * x * x + 52.5 * y * y - 420.0 * z * z);
    buffer[3] = sqrt_105 * x * y * z * (-0.5 * x * x - 0.5 * y * y + z * z);
    buffer[4] = (1.0 / 15.0) * sqrt_15 * y *
                (39.375 * std::pow(z, 4) - 26.25 * z * z * (x * x + y * y + z * z) +
                 1.875 * std::pow(x * x + y * y + z * z, 2));
    buffer[5] = z * (7.875 * std::pow(z, 4) - 8.75 * z * z * (x * x + y * y + z * z) +
                     1.875 * std::pow(x * x + y * y + z * z, 2));
    buffer[6] = (1.0 / 15.0) * sqrt_15 * x *
                (39.375 * std::pow(z, 4) - 26.25 * z * z * (x * x + y * y + z * z) +
                 1.875 * std::pow(x * x + y * y + z * z, 2));
    buffer[7] = (1.0 / 210.0) * sqrt_105 * z * (x * x - y * y) *
                (-52.5 * x * x - 52.5 * y * y + 105.0 * z * z);
    buffer[8] = -1.0 / 840.0 * sqrt_70 * x * (x * x - 3 * y * y) *
                (52.5 * x * x + 52.5 * y * y - 420.0 * z * z);
    buffer[9] = 0.375 * sqrt_35 * z * (pow(x, 4) - 6 * x * x * y * y + std::pow(y, 4));
    buffer[10] =
        sqrt_14 * x * (0.1875 * std::pow(x, 4) - 1.875 * x * x * y * y + 0.9375 * std::pow(y, 4));
}

void compute_real_spherical_harmonic_6(double x, double y, double z, double* buffer) {
    // Compute the spherical harmonic for l = 6
    buffer[0] = sqrt_462 * x * y *
                (0.1875 * std::pow(x, 4) - 0.625 * x * x * y * y + 0.1875 * std::pow(y, 4));
    buffer[1] =
        0.1875 * sqrt_154 * y * z * (5 * std::pow(x, 4) - 10 * x * x * y * y + std::pow(y, 4));
    buffer[2] = -1.0 / 630.0 * sqrt_7 * x * y * (x * x - y * y) *
                (472.5 * x * x + 472.5 * y * y - 4725.0 * z * z);
    buffer[3] = (1.0 / 2520.0) * sqrt_210 * y * z * (3 * x * x - y * y) *
                (-472.5 * x * x - 472.5 * y * y + 1260.0 * z * z);
    buffer[4] = (1.0 / 210.0) * sqrt_210 * x * y *
                (433.125 * std::pow(z, 4) - 236.25 * z * z * (x * x + y * y + z * z) +
                 13.125 * std::pow(x * x + y * y + z * z, 2));
    buffer[5] = (1.0 / 21.0) * sqrt_21 * y * z *
                (86.625 * std::pow(z, 4) - 78.75 * z * z * (x * x + y * y + z * z) +
                 13.125 * std::pow(x * x + y * y + z * z, 2));
    buffer[6] = (14.4375 * std::pow(z, 6) - 19.6875 * std::pow(z, 4) * (x * x + y * y + z * z) +
                 6.5625 * z * z * std::pow(x * x + y * y + z * z, 2) -
                 0.3125 * std::pow(x * x + y * y + z * z, 3));
    buffer[7] = (1.0 / 21.0) * sqrt_21 * x * z *
                (86.625 * std::pow(z, 4) - 78.75 * z * z * (x * x + y * y + z * z) +
                 13.125 * std::pow(x * x + y * y + z * z, 2));
    buffer[8] = (1.0 / 420.0) * sqrt_210 * (x * x - y * y) *
                (433.125 * std::pow(z, 4) - 236.25 * z * z * (x * x + y * y + z * z) +
                 13.125 * std::pow(x * x + y * y + z * z, 2));
    buffer[9] = (1.0 / 2520.0) * sqrt_210 * x * z * (x * x - 3 * y * y) *
                (-472.5 * x * x - 472.5 * y * y + 1260.0 * z * z);
    buffer[10] = -1.0 / 2520.0 * sqrt_7 * (472.5 * x * x + 472.5 * y * y - 4725.0 * z * z) *
                 (pow(x, 4) - 6 * x * x * y * y + std::pow(y, 4));
    buffer[11] = 0.1875 * sqrt_154 * x * z * (pow(x, 4) - 10 * x * x * y * y + 5 * std::pow(y, 4));
    buffer[12] = sqrt_462 * (0.03125 * std::pow(x, 6) - 0.46875 * std::pow(x, 4) * y * y +
                             0.46875 * x * x * std::pow(y, 4) - 0.03125 * std::pow(y, 6));
}

void compute_real_spherical_harmonic_7(double x, double y, double z, double* buffer) {
    // Compute the spherical harmonic for l = 7
    buffer[0] = sqrt_429 * y *
                (0.21875 * std::pow(x, 6) - 1.09375 * std::pow(x, 4) * y * y +
                 0.65625 * x * x * std::pow(y, 4) - 0.03125 * std::pow(y, 6));
    buffer[1] = sqrt_6006 * x * y * z *
                (0.1875 * std::pow(x, 4) - 0.625 * x * x * y * y + 0.1875 * std::pow(y, 4));
    buffer[2] = -1.0 / 166320.0 * sqrt_231 * y *
                (5197.5 * x * x + 5197.5 * y * y - 62370.0 * z * z) *
                (5 * std::pow(x, 4) - 10 * x * x * y * y + std::pow(y, 4));
    buffer[3] = (1.0 / 6930.0) * sqrt_231 * x * y * z * (x * x - y * y) *
                (-5197.5 * x * x - 5197.5 * y * y + 17325.0 * z * z);
    buffer[4] = (1.0 / 1260.0) * sqrt_21 * y * (3 * x * x - y * y) *
                (5630.625 * std::pow(z, 4) - 2598.75 * z * z * (x * x + y * y + z * z) +
                 118.125 * std::pow(x * x + y * y + z * z, 2));
    buffer[5] = (1.0 / 126.0) * sqrt_42 * x * y * z *
                (1126.125 * std::pow(z, 4) - 866.25 * z * z * (x * x + y * y + z * z) +
                 118.125 * std::pow(x * x + y * y + z * z, 2));
    buffer[6] = (1.0 / 14.0) * sqrt_7 * y *
                (187.6875 * std::pow(z, 6) - 216.5625 * std::pow(z, 4) * (x * x + y * y + z * z) +
                 59.0625 * z * z * std::pow(x * x + y * y + z * z, 2) -
                 2.1875 * std::pow(x * x + y * y + z * z, 3));
    buffer[7] = z * (26.8125 * std::pow(z, 6) - 43.3125 * std::pow(z, 4) * (x * x + y * y + z * z) +
                     19.6875 * z * z * std::pow(x * x + y * y + z * z, 2) -
                     2.1875 * std::pow(x * x + y * y + z * z, 3));
    buffer[8] = (1.0 / 14.0) * sqrt_7 * x *
                (187.6875 * std::pow(z, 6) - 216.5625 * std::pow(z, 4) * (x * x + y * y + z * z) +
                 59.0625 * z * z * std::pow(x * x + y * y + z * z, 2) -
                 2.1875 * std::pow(x * x + y * y + z * z, 3));
    buffer[9] = (1.0 / 252.0) * sqrt_42 * z * (x * x - y * y) *
                (1126.125 * std::pow(z, 4) - 866.25 * z * z * (x * x + y * y + z * z) +
                 118.125 * std::pow(x * x + y * y + z * z, 2));
    buffer[10] = (1.0 / 1260.0) * sqrt_21 * x * (x * x - 3 * y * y) *
                 (5630.625 * std::pow(z, 4) - 2598.75 * z * z * (x * x + y * y + z * z) +
                  118.125 * std::pow(x * x + y * y + z * z, 2));
    buffer[11] = (1.0 / 27720.0) * sqrt_231 * z *
                 (-5197.5 * x * x - 5197.5 * y * y + 17325.0 * z * z) *
                 (pow(x, 4) - 6 * x * x * y * y + std::pow(y, 4));
    buffer[12] = -1.0 / 166320.0 * sqrt_231 * x *
                 (5197.5 * x * x + 5197.5 * y * y - 62370.0 * z * z) *
                 (pow(x, 4) - 10 * x * x * y * y + 5 * std::pow(y, 4));
    buffer[13] =
        0.03125 * sqrt_6006 * z *
        (pow(x, 6) - 15 * std::pow(x, 4) * y * y + 15 * x * x * std::pow(y, 4) - std::pow(y, 6));
    buffer[14] = sqrt_429 * x *
                 (0.03125 * std::pow(x, 6) - 0.65625 * std::pow(x, 4) * y * y +
                  1.09375 * x * x * std::pow(y, 4) - 0.21875 * std::pow(y, 6));
}

// a function pointer type for the spherical harmonic computation
using ComputeRealSphericalHarmonic = void (*)(double, double, double, double*);

// a map of function pointers for each l value
std::array<ComputeRealSphericalHarmonic, 8> compute_real_spherical_harmonic = {
    compute_real_spherical_harmonic_0, compute_real_spherical_harmonic_1,
    compute_real_spherical_harmonic_2, compute_real_spherical_harmonic_3,
    compute_real_spherical_harmonic_4, compute_real_spherical_harmonic_5,
    compute_real_spherical_harmonic_6, compute_real_spherical_harmonic_7};

} // namespace ccpy::ints
