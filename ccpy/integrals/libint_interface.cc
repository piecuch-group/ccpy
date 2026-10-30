// =====================================================================================
//  libint_interface.cc -- atomic-orbital integrals for ccpy (Python module `_libint`)
// =====================================================================================
//
//  Provenance and license
//  ----------------------
//  This file is adapted from the integral library of forte2, developed by the
//  Evangelista group (https://github.com/evangelistalab/forte2), which is distributed
//  under the GNU Lesser General Public License v3.0 (LGPL-3.0). The original forte2
//  code was ported into ccpy and then merged and simplified into this single file.
//  ccpy as a whole is distributed under the GNU General Public License v3.0 (GPL-3.0).
//
//  What is in here
//  ---------------
//  All integrals are computed with libint2 and returned to Python as numpy arrays
//  through nanobind. The file is organized in sections:
//
//    1. Helpers         numpy array aliases, logging, threading, libint2 setup
//    2. Basis           the Basis class, shell labels, evaluating a shell at a point
//    3. One-electron    overlap, kinetic, nuclear, multipoles, opVop, erf/erfc nuclear
//    4. Two-electron    2-, 3- and 4-center Coulomb (plus erf/erfc range-separated)
//    5. Derivatives     gradients of 1e/2e integrals contracted with weights, 3c opVop
//    6. Grids           basis functions and orbitals evaluated on points / grids
//    7. Bindings        the NB_MODULE(_libint, m) definition (the Python API)
//
//  Conventions: basis-function indices run over shells in Basis order; every array is
//  returned in C (row-major) order; coordinates are in bohr. "charges" is always a list
//  of (Z, [x, y, z]) pairs, one per atom, in the same order as the atoms in the basis.
// =====================================================================================

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <format>
#include <future>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <iterator>
#include <utility>
#include <vector>

#ifdef __linux__
#include <sched.h>
#endif

#include <libint2.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace ccpy {

using libint2::BraKet;
using libint2::Operator;
using libint2::Shell;

// =====================================================================================
//  1. Helpers
// =====================================================================================

// numpy array types. Without "_c" any memory layout is accepted (access via .view());
// with "_c" the array must be C-contiguous (we access the raw .data() pointer).
using np_vector = nb::ndarray<nb::numpy, double, nb::ndim<1>>;
using np_matrix = nb::ndarray<nb::numpy, double, nb::ndim<2>>;
using np_tensor3 = nb::ndarray<nb::numpy, double, nb::ndim<3>>;
using np_tensor4 = nb::ndarray<nb::numpy, double, nb::ndim<4>>;
using np_matrix_c = nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig>;
using np_tensor3_c = nb::ndarray<nb::numpy, double, nb::ndim<3>, nb::c_contig>;
using np_tensor4_c = nb::ndarray<nb::numpy, double, nb::ndim<4>, nb::c_contig>;
using np_matrix_complex_c = nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<2>, nb::c_contig>;
using np_tensor3_complex_c =
    nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<3>, nb::c_contig>;

// Point charges: a list of (Z, {x, y, z}).
using Charges = std::vector<std::pair<double, std::array<double, 3>>>;
// Range-separated point charges: (omega, charges).
using OmegaCharges = std::tuple<double, Charges>;
// Half-open shell ranges [first, last) for each of the three bases of a 3c integral.
using ShellSlices = std::array<std::pair<std::size_t, std::size_t>, 3>;

// Allocate a zero-filled numpy array of the given shape. numpy owns the memory and frees
// it when the array is garbage collected. Usage: auto S = zeros<np_matrix>({n1, n2});
template <typename ArrayType> ArrayType zeros(std::initializer_list<std::size_t> shape) {
    std::size_t size = 1;
    for (auto n : shape)
        size *= n;
    double* data = new double[size](); // the () zero-initializes
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    return ArrayType(data, shape.size(), shape.begin(), owner);
}

// The real part of a weight (weights can be real or complex arrays).
inline double real_part(double x) { return x; }
inline double real_part(const std::complex<double>& x) { return x.real(); }

// ---- Logging ------------------------------------------------------------------------
// Verbosity: 0 = silent, 1 = warnings, 2 = essential, 3 = info (default), 4 = timings,
// 5 = debug. Set from Python with set_log_level().
int log_level = 3;

template <typename... Args> void log_msg(int level, const Args&... args) {
    if (level > log_level)
        return;
    std::ostringstream os;
    (os << ... << args);
    std::cout << os.str() << std::endl;
}

// Milliseconds elapsed since `start` (used for the timing messages at log level 4).
using Clock = std::chrono::high_resolution_clock;
inline long long ms_since(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
}

// ---- Threading ----------------------------------------------------------------------

// Parse a positive integer from an environment variable value. Returns 0 if the variable
// is unset or not a valid positive integer. With allow_list, "4,2" is read as 4 (this is
// the OMP_NUM_THREADS nested-list syntax). Leading/trailing whitespace is ignored.
std::size_t parse_thread_limit(const char* value, bool allow_list = false) {
    if (value == nullptr)
        return 0;
    std::string s(value);
    const auto first = s.find_first_not_of(" \t\n\r\f\v");
    if (first == std::string::npos)
        return 0;
    const char* begin = s.data() + first;
    const char* end = s.data() + s.size();
    std::size_t n = 0;
    auto [ptr, ec] = std::from_chars(begin, end, n);
    if (ec != std::errc{})
        return 0;
    while (ptr != end && std::string(" \t\n\r\f\v").find(*ptr) != std::string::npos)
        ++ptr;
    if (ptr != end && !(allow_list && *ptr == ','))
        return 0;
    return n;
}

// Number of CPUs this process may run on (Linux affinity mask), or 0 if unknown.
std::size_t affinity_cpu_count() {
#ifdef __linux__
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0)
        return static_cast<std::size_t>(CPU_COUNT(&mask));
#endif
    return 0;
}

// Number of threads used by the threaded integral kernels.
// CCPY_NUM_THREADS_OVERRIDE wins if set. Otherwise we take the smallest of: the number of
// logical CPUs, the CPU affinity mask, OMP_NUM_THREADS, OMP_THREAD_LIMIT and
// SLURM_CPUS_PER_TASK (unset/invalid values are ignored). Always at least 1.
std::size_t get_num_threads() {
    if (std::size_t n = parse_thread_limit(std::getenv("CCPY_NUM_THREADS_OVERRIDE")))
        return n;
    std::size_t n = std::max(1u, std::thread::hardware_concurrency());
    for (std::size_t limit : {affinity_cpu_count(),
                              parse_thread_limit(std::getenv("OMP_NUM_THREADS"), true),
                              parse_thread_limit(std::getenv("OMP_THREAD_LIMIT")),
                              parse_thread_limit(std::getenv("SLURM_CPUS_PER_TASK"))}) {
        if (limit > 0)
            n = std::min(n, limit);
    }
    return n;
}

// Run func(thread, first, last) over [begin, end) split into one contiguous block per
// thread; `thread` is 0, 1, ... < get_num_threads() and can index per-thread scratch.
// Short loops (fewer than 3 items per thread) run serially as func(0, begin, end).
// Exceptions thrown inside a worker are re-thrown here.
template <typename Func> void parallel_for(std::size_t begin, std::size_t end, Func&& func) {
    if (end <= begin)
        return;
    const std::size_t count = end - begin;
    const std::size_t nthreads = get_num_threads();
    if (nthreads <= 1 || count < 3 * nthreads) {
        func(std::size_t{0}, begin, end);
        return;
    }
    const std::size_t block = (count + nthreads - 1) / nthreads;
    std::vector<std::future<void>> workers;
    for (std::size_t t = 0; t < nthreads; ++t) {
        const std::size_t first = begin + t * block;
        const std::size_t last = std::min(first + block, end);
        if (first < last)
            workers.push_back(std::async(std::launch::async,
                                         [&func, t, first, last] { func(t, first, last); }));
    }
    for (auto& w : workers)
        w.get();
}

// ---- libint2 setup ------------------------------------------------------------------

// libint2 must be initialized while engines are alive. Each public function creates one
// of these first: it initializes libint2 and finalizes it again when the function returns.
struct LibintSession {
    LibintSession() { libint2::initialize(); }
    ~LibintSession() { libint2::finalize(); }
    LibintSession(const LibintSession&) = delete;
    LibintSession& operator=(const LibintSession&) = delete;
};

class Basis;
libint2::Engine make_engine(Operator op, std::initializer_list<const Basis*> bases,
                            int deriv_order = 0, BraKet braket = BraKet::invalid);

// =====================================================================================
//  2. Basis
// =====================================================================================

// A list of contracted shells (one contraction per shell). Shells on the same atom must
// be added consecutively; atoms are detected by a change of shell center.
class Basis {
  public:
    void add(const Shell& shell) {
        if (shell.contr.size() != 1)
            throw std::runtime_error("Only single shells are supported");
        if (!shell.contr[0].pure)
            log_msg(1, "[ccpy] Warning: Cartesian Gaussians have limited support.\n"
                   "         It is recommended to use Solid-Harmonic Gaussians instead "
                   "(is_pure=True).");
        shells_.push_back(shell);
        max_l_ = std::max(max_l_, shell.contr[0].l);
        max_nprim_ = std::max(max_nprim_, shell.nprim());
        max_nbasis_ = std::max(max_nbasis_, shell.size());
        size_ += shell.size();
    }

    const Shell& operator[](std::size_t i) const {
        if (i >= shells_.size())
            throw std::out_of_range("Index out of range");
        return shells_[i];
    }

    void set_name(const std::string& name) { name_ = name; }
    std::string name() const { return name_; }
    std::size_t size() const { return size_; } // number of basis functions
    std::size_t nshells() const { return shells_.size(); }
    int max_l() const { return max_l_; }
    std::size_t max_nprim() const { return max_nprim_; }
    std::size_t max_nbasis() const { return max_nbasis_; }

    // (index of first basis function, number of functions) for each shell.
    std::vector<std::pair<std::size_t, std::size_t>> shell_first_and_size() const {
        std::vector<std::pair<std::size_t, std::size_t>> result;
        std::size_t first = 0;
        for (const auto& shell : shells_) {
            result.emplace_back(first, shell.size());
            first += shell.size();
        }
        return result;
    }

    // Index of the first basis function of each shell, plus the total size at the end.
    std::vector<std::size_t> shell_offsets() const {
        std::vector<std::size_t> offsets{0};
        for (const auto& shell : shells_)
            offsets.push_back(offsets.back() + shell.size());
        return offsets;
    }

    // [first, last) range per atom (center), counting shells or basis functions.
    std::vector<std::pair<std::size_t, std::size_t>> center_first_and_last(bool count_shell) const {
        std::vector<std::pair<std::size_t, std::size_t>> result;
        if (shells_.empty())
            return result;
        std::size_t first = 0, last = 0;
        auto center = shells_[0].O;
        for (const auto& shell : shells_) {
            const auto& O = shell.O;
            if (std::hypot(O[0] - center[0], O[1] - center[1], O[2] - center[2]) > 1e-8) {
                result.emplace_back(first, last); // a new center starts here
                center = O;
                first = last;
            }
            last += count_shell ? 1 : shell.size();
        }
        result.emplace_back(first, last);
        return result;
    }

    // Atom (center) index of each shell.
    std::vector<std::size_t> shell_to_center() const {
        std::vector<std::size_t> result(nshells());
        const auto centers = center_first_and_last(true);
        for (std::size_t atom = 0; atom < centers.size(); ++atom)
            for (std::size_t s = centers[atom].first; s < centers[atom].second; ++s)
                result[s] = atom;
        return result;
    }

  private:
    std::vector<Shell> shells_;
    std::size_t size_ = 0;       // total number of basis functions
    std::size_t max_nprim_ = 0;  // largest number of primitives in a shell
    std::size_t max_nbasis_ = 0; // largest number of functions in a shell
    int max_l_ = 0;              // largest angular momentum
    std::string name_ = "unnamed basis";
};

// Create a libint2 engine large enough for all the given bases.
libint2::Engine make_engine(Operator op, std::initializer_list<const Basis*> bases,
                            int deriv_order, BraKet braket) {
    std::size_t max_nprim = 0;
    int max_l = 0;
    for (const Basis* b : bases) {
        max_nprim = std::max(max_nprim, b->max_nprim());
        max_l = std::max(max_l, b->max_l());
    }
    libint2::Engine engine(op, max_nprim, max_l, deriv_order);
    if (braket != BraKet::invalid)
        engine.set(braket);
    return engine;
}

// Label of the idx-th function of a shell with angular momentum l ("px", "dz2", "g(3)").
std::string shell_label(int l, int idx) {
    static const std::vector<std::vector<std::string>> labels{
        {"s"},
        {"py", "pz", "px"},
        {"dxy", "dyz", "dz2", "dxz", "dx2-y2"},
        {"fy(3x2-y2)", "fxyz", "fyz2", "fz3", "fxz2", "fz(x2-y2)", "fx(x2-3y2)"}};
    static const std::vector<std::string> letters{"s", "p", "d", "f", "g", "h",
                                                  "i", "j", "k", "l", "m", "n"};
    if (l < 0)
        throw std::out_of_range("Invalid angular momentum index: " + std::to_string(l));
    if (idx < 0)
        throw std::out_of_range("Invalid index for angular momentum " + std::to_string(l) +
                                ": " + std::to_string(idx));
    if (l < static_cast<int>(labels.size()) && idx < static_cast<int>(labels[l].size()))
        return labels[l][idx];
    if (l >= static_cast<int>(letters.size()))
        throw std::out_of_range("Angular momentum " + std::to_string(l) +
                                " exceeds defined shell labels.");
    return letters[l] + "(" + std::to_string(idx) + ")";
}

// Real solid harmonics in libint2's order (m = -l ... l), for l = 0..7, written to out.
// See https://en.wikipedia.org/wiki/Solid_harmonics#Real_form
void real_solid_harmonics(int l, double x, double y, double z, double* out) {
    // square roots of small integers (20 digits)
    constexpr double sqrt_3 = 1.7320508075688772935, sqrt_5 = 2.2360679774997896964,
                     sqrt_6 = 2.4494897427831780982, sqrt_7 = 2.6457513110645905905,
                     sqrt_10 = 3.1622776601683793320, sqrt_14 = 3.7416573867739413856,
                     sqrt_15 = 3.8729833462074168852, sqrt_21 = 4.5825756949558400066,
                     sqrt_35 = 5.9160797830996160426, sqrt_42 = 6.4807406984078602310,
                     sqrt_70 = 8.3666002653407554798, sqrt_105 = 10.246950765959598383,
                     sqrt_154 = 12.409673645990856596, sqrt_210 = 14.491376746189438574,
                     sqrt_231 = 15.198684153570663632, sqrt_429 = 20.712315177207979132,
                     sqrt_462 = 21.494185260204677039, sqrt_6006 = 77.498387079990251885;
    using std::pow;
    const double r2 = x * x + y * y + z * z;
    switch (l) {
    case 0:
        out[0] = 1.0;
        break;
    case 1:
        out[0] = y;
        out[1] = z;
        out[2] = x;
        break;
    case 2:
        out[0] = sqrt_3 * x * y;
        out[1] = sqrt_3 * y * z;
        out[2] = (z * z - 0.5 * (x * x + y * y));
        out[3] = sqrt_3 * x * z;
        out[4] = 0.5 * sqrt_3 * (x * x - y * y);
        break;
    case 3:
        out[0] = sqrt_10 * y * (0.75 * x * x - 0.25 * y * y);
        out[1] = sqrt_15 * x * y * z;
        out[2] = sqrt_6 * y * (z * z - 0.25 * (x * x + y * y));
        out[3] = z * (z * z - 1.5 * (x * x + y * y));
        out[4] = sqrt_6 * x * (z * z - 0.25 * (x * x + y * y));
        out[5] = 0.5 * sqrt_15 * z * (x * x - y * y);
        out[6] = sqrt_10 * x * (0.25 * x * x - 0.75 * y * y);
        break;
    case 4:
        out[0] = 0.5 * sqrt_35 * x * y * (x * x - y * y);
        out[1] = 0.25 * sqrt_70 * y * z * (3 * x * x - y * y);
        out[2] = sqrt_5 * x * y * (-0.5 * x * x - 0.5 * y * y + 3.0 * z * z);
        out[3] = sqrt_10 * y * z * (-0.75 * x * x - 0.75 * y * y + z * z);
        out[4] = (4.375 * pow(z, 4) - 3.75 * z * z * r2 + 0.375 * pow(r2, 2));
        out[5] = sqrt_10 * x * z * (-0.75 * x * x - 0.75 * y * y + z * z);
        out[6] = -1.0 / 30.0 * sqrt_5 * (x * x - y * y) * (7.5 * x * x + 7.5 * y * y - 45.0 * z * z);
        out[7] = 0.25 * sqrt_70 * x * z * (x * x - 3 * y * y);
        out[8] = sqrt_35 * (0.125 * pow(x, 4) - 0.75 * x * x * y * y + 0.125 * pow(y, 4));
        break;
    case 5:
        out[0] = sqrt_14 * y * (0.9375 * pow(x, 4) - 1.875 * x * x * y * y + 0.1875 * pow(y, 4));
        out[1] = 1.5 * sqrt_35 * x * y * z * (x * x - y * y);
        out[2] = -1.0 / 840.0 * sqrt_70 * y * (3 * x * x - y * y) *
                 (52.5 * x * x + 52.5 * y * y - 420.0 * z * z);
        out[3] = sqrt_105 * x * y * z * (-0.5 * x * x - 0.5 * y * y + z * z);
        out[4] = (1.0 / 15.0) * sqrt_15 * y *
                 (39.375 * pow(z, 4) - 26.25 * z * z * r2 + 1.875 * pow(r2, 2));
        out[5] = z * (7.875 * pow(z, 4) - 8.75 * z * z * r2 + 1.875 * pow(r2, 2));
        out[6] = (1.0 / 15.0) * sqrt_15 * x *
                 (39.375 * pow(z, 4) - 26.25 * z * z * r2 + 1.875 * pow(r2, 2));
        out[7] = (1.0 / 210.0) * sqrt_105 * z * (x * x - y * y) *
                 (-52.5 * x * x - 52.5 * y * y + 105.0 * z * z);
        out[8] = -1.0 / 840.0 * sqrt_70 * x * (x * x - 3 * y * y) *
                 (52.5 * x * x + 52.5 * y * y - 420.0 * z * z);
        out[9] = 0.375 * sqrt_35 * z * (pow(x, 4) - 6 * x * x * y * y + pow(y, 4));
        out[10] = sqrt_14 * x * (0.1875 * pow(x, 4) - 1.875 * x * x * y * y + 0.9375 * pow(y, 4));
        break;
    case 6:
        out[0] = sqrt_462 * x * y * (0.1875 * pow(x, 4) - 0.625 * x * x * y * y + 0.1875 * pow(y, 4));
        out[1] = 0.1875 * sqrt_154 * y * z * (5 * pow(x, 4) - 10 * x * x * y * y + pow(y, 4));
        out[2] = -1.0 / 630.0 * sqrt_7 * x * y * (x * x - y * y) *
                 (472.5 * x * x + 472.5 * y * y - 4725.0 * z * z);
        out[3] = (1.0 / 2520.0) * sqrt_210 * y * z * (3 * x * x - y * y) *
                 (-472.5 * x * x - 472.5 * y * y + 1260.0 * z * z);
        out[4] = (1.0 / 210.0) * sqrt_210 * x * y *
                 (433.125 * pow(z, 4) - 236.25 * z * z * r2 + 13.125 * pow(r2, 2));
        out[5] = (1.0 / 21.0) * sqrt_21 * y * z *
                 (86.625 * pow(z, 4) - 78.75 * z * z * r2 + 13.125 * pow(r2, 2));
        out[6] = (14.4375 * pow(z, 6) - 19.6875 * pow(z, 4) * r2 + 6.5625 * z * z * pow(r2, 2) -
                  0.3125 * pow(r2, 3));
        out[7] = (1.0 / 21.0) * sqrt_21 * x * z *
                 (86.625 * pow(z, 4) - 78.75 * z * z * r2 + 13.125 * pow(r2, 2));
        out[8] = (1.0 / 420.0) * sqrt_210 * (x * x - y * y) *
                 (433.125 * pow(z, 4) - 236.25 * z * z * r2 + 13.125 * pow(r2, 2));
        out[9] = (1.0 / 2520.0) * sqrt_210 * x * z * (x * x - 3 * y * y) *
                 (-472.5 * x * x - 472.5 * y * y + 1260.0 * z * z);
        out[10] = -1.0 / 2520.0 * sqrt_7 * (472.5 * x * x + 472.5 * y * y - 4725.0 * z * z) *
                  (pow(x, 4) - 6 * x * x * y * y + pow(y, 4));
        out[11] = 0.1875 * sqrt_154 * x * z * (pow(x, 4) - 10 * x * x * y * y + 5 * pow(y, 4));
        out[12] = sqrt_462 * (0.03125 * pow(x, 6) - 0.46875 * pow(x, 4) * y * y +
                              0.46875 * x * x * pow(y, 4) - 0.03125 * pow(y, 6));
        break;
    case 7:
        out[0] = sqrt_429 * y *
                 (0.21875 * pow(x, 6) - 1.09375 * pow(x, 4) * y * y + 0.65625 * x * x * pow(y, 4) -
                  0.03125 * pow(y, 6));
        out[1] = sqrt_6006 * x * y * z *
                 (0.1875 * pow(x, 4) - 0.625 * x * x * y * y + 0.1875 * pow(y, 4));
        out[2] = -1.0 / 166320.0 * sqrt_231 * y * (5197.5 * x * x + 5197.5 * y * y - 62370.0 * z * z) *
                 (5 * pow(x, 4) - 10 * x * x * y * y + pow(y, 4));
        out[3] = (1.0 / 6930.0) * sqrt_231 * x * y * z * (x * x - y * y) *
                 (-5197.5 * x * x - 5197.5 * y * y + 17325.0 * z * z);
        out[4] = (1.0 / 1260.0) * sqrt_21 * y * (3 * x * x - y * y) *
                 (5630.625 * pow(z, 4) - 2598.75 * z * z * r2 + 118.125 * pow(r2, 2));
        out[5] = (1.0 / 126.0) * sqrt_42 * x * y * z *
                 (1126.125 * pow(z, 4) - 866.25 * z * z * r2 + 118.125 * pow(r2, 2));
        out[6] = (1.0 / 14.0) * sqrt_7 * y *
                 (187.6875 * pow(z, 6) - 216.5625 * pow(z, 4) * r2 + 59.0625 * z * z * pow(r2, 2) -
                  2.1875 * pow(r2, 3));
        out[7] = z * (26.8125 * pow(z, 6) - 43.3125 * pow(z, 4) * r2 +
                      19.6875 * z * z * pow(r2, 2) - 2.1875 * pow(r2, 3));
        out[8] = (1.0 / 14.0) * sqrt_7 * x *
                 (187.6875 * pow(z, 6) - 216.5625 * pow(z, 4) * r2 + 59.0625 * z * z * pow(r2, 2) -
                  2.1875 * pow(r2, 3));
        out[9] = (1.0 / 252.0) * sqrt_42 * z * (x * x - y * y) *
                 (1126.125 * pow(z, 4) - 866.25 * z * z * r2 + 118.125 * pow(r2, 2));
        out[10] = (1.0 / 1260.0) * sqrt_21 * x * (x * x - 3 * y * y) *
                  (5630.625 * pow(z, 4) - 2598.75 * z * z * r2 + 118.125 * pow(r2, 2));
        out[11] = (1.0 / 27720.0) * sqrt_231 * z * (-5197.5 * x * x - 5197.5 * y * y + 17325.0 * z * z) *
                  (pow(x, 4) - 6 * x * x * y * y + pow(y, 4));
        out[12] = -1.0 / 166320.0 * sqrt_231 * x * (5197.5 * x * x + 5197.5 * y * y - 62370.0 * z * z) *
                  (pow(x, 4) - 10 * x * x * y * y + 5 * pow(y, 4));
        out[13] = 0.03125 * sqrt_6006 * z *
                  (pow(x, 6) - 15 * pow(x, 4) * y * y + 15 * x * x * pow(y, 4) - pow(y, 6));
        out[14] = sqrt_429 * x *
                  (0.03125 * pow(x, 6) - 0.65625 * pow(x, 4) * y * y + 1.09375 * x * x * pow(y, 4) -
                   0.21875 * pow(y, 6));
        break;
    default:
        throw std::runtime_error("[ccpy] evaluate_shell with l > 7 is not supported");
    }
}

// Values of the (pure) functions of `shell` at `point`, written to out[0 .. shell.size()).
void evaluate_shell(const Shell& shell, const std::array<double, 3>& point, double* out) {
    const auto& contr = shell.contr[0];
    if (contr.l > 7)
        throw std::runtime_error("[ccpy] evaluate_shell with l > 7 is not supported");
    const double dx = point[0] - shell.O[0];
    const double dy = point[1] - shell.O[1];
    const double dz = point[2] - shell.O[2];
    const double r2 = dx * dx + dy * dy + dz * dz;
    double radial = 0.0; // contracted Gaussian
    for (std::size_t i = 0; i < shell.nprim(); ++i)
        radial += contr.coeff[i] * std::exp(-shell.alpha[i] * r2);
    real_solid_harmonics(contr.l, dx, dy, dz, out);
    for (std::size_t i = 0; i < contr.size(); ++i)
        out[i] *= radial;
}

// =====================================================================================
//  3. One-electron integrals
// =====================================================================================

// <b1|op|b2> for an operator with M components (e.g. M = 4 for emultipole1: 1, x, y, z).
// The engine must already be set up (operator and parameters). Returns M (n1 x n2) matrices.
template <std::size_t M>
std::array<np_matrix, M> one_electron(libint2::Engine& engine, const Basis& b1, const Basis& b2) {
    std::array<np_matrix, M> ints;
    for (auto& mat : ints)
        mat = zeros<np_matrix>({b1.size(), b2.size()});
    const auto off1 = b1.shell_offsets();
    const auto off2 = b2.shell_offsets();
    const auto& results = engine.results();

    for (std::size_t s1 = 0; s1 < b1.nshells(); ++s1) {
        for (std::size_t s2 = 0; s2 < b2.nshells(); ++s2) {
            engine.compute(b1[s1], b2[s2]);
            for (std::size_t c = 0; c < M; ++c) {
                const double* buf = results[c];
                if (!buf)
                    continue; // screened out: stays zero
                auto v = ints[c].view();
                for (std::size_t i = 0, ij = 0; i < b1[s1].size(); ++i)
                    for (std::size_t j = 0; j < b2[s2].size(); ++j, ++ij)
                        v(off1[s1] + i, off2[s2] + j) = buf[ij];
            }
        }
    }
    return ints;
}

np_matrix overlap(const Basis& b1, const Basis& b2) {
    LibintSession libint;
    auto engine = make_engine(Operator::overlap, {&b1, &b2});
    return one_electron<1>(engine, b1, b2)[0];
}

np_matrix kinetic(const Basis& b1, const Basis& b2) {
    LibintSession libint;
    auto engine = make_engine(Operator::kinetic, {&b1, &b2});
    return one_electron<1>(engine, b1, b2)[0];
}

// <b1| sum_A -Z_A / |r - R_A| |b2>
np_matrix nuclear(const Basis& b1, const Basis& b2, const Charges& charges) {
    LibintSession libint;
    auto engine = make_engine(Operator::nuclear, {&b1, &b2});
    engine.set_params(charges);
    return one_electron<1>(engine, b1, b2)[0];
}

// Nuclear attraction with erf(omega r)/r (long range) or erfc(omega r)/r (short range).
np_matrix erf_nuclear(const Basis& b1, const Basis& b2, const OmegaCharges& omega_charges) {
    LibintSession libint;
    auto engine = make_engine(Operator::erf_nuclear, {&b1, &b2});
    engine.set_params(omega_charges);
    return one_electron<1>(engine, b1, b2)[0];
}

np_matrix erfc_nuclear(const Basis& b1, const Basis& b2, const OmegaCharges& omega_charges) {
    LibintSession libint;
    auto engine = make_engine(Operator::erfc_nuclear, {&b1, &b2});
    engine.set_params(omega_charges);
    return one_electron<1>(engine, b1, b2)[0];
}

// Electric multipoles about `origin`, components in libint2 order:
//   emultipole1: 1, x, y, z
//   emultipole2: ... + xx, xy, xz, yy, yz, zz
//   emultipole3: ... + xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
template <Operator op, std::size_t M>
std::array<np_matrix, M> multipole(const Basis& b1, const Basis& b2,
                                   const std::array<double, 3>& origin) {
    LibintSession libint;
    auto engine = make_engine(op, {&b1, &b2});
    engine.set_params(origin);
    return one_electron<M>(engine, b1, b2);
}

// Small-component potential: <b1| p.Vp, (p x Vp)_x, (p x Vp)_y, (p x Vp)_z |b2>.
std::array<np_matrix, 4> opVop(const Basis& b1, const Basis& b2, const Charges& charges) {
    LibintSession libint;
    auto engine = make_engine(Operator::opVop, {&b1, &b2});
    engine.set_params(charges);
    return one_electron<4>(engine, b1, b2);
}

// Classical repulsion energy sum_{A>B} Z_A Z_B / R_AB.
double nuclear_repulsion(const Charges& charges) {
    double enuc = 0.0;
    for (std::size_t i = 0; i < charges.size(); ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            const auto& [Zi, Ri] = charges[i];
            const auto& [Zj, Rj] = charges[j];
            const double r = std::hypot(Ri[0] - Rj[0], Ri[1] - Rj[1], Ri[2] - Rj[2]);
            if (r < 1e-10)
                throw std::runtime_error("Nuclear repulsion: zero distance between charges");
            enuc += Zi * Zj / r;
        }
    }
    return enuc;
}

// =====================================================================================
//  4. Two-electron integrals
// =====================================================================================

// Copy a libint2 shell-triplet block (n1 x n2 x n3) into a C-ordered 3D array whose last
// two dimensions are (d2, d3), starting at index (f1, f2, f3).
inline void put_block3(const double* buf, double* out, std::size_t d2, std::size_t d3,
                       std::size_t f1, std::size_t n1, std::size_t f2, std::size_t n2,
                       std::size_t f3, std::size_t n3) {
    for (std::size_t i = 0, ijk = 0; i < n1; ++i)
        for (std::size_t j = 0; j < n2; ++j)
            for (std::size_t k = 0; k < n3; ++k, ++ijk)
                out[(f1 + i) * d2 * d3 + (f2 + j) * d3 + f3 + k] = buf[ijk];
}

// (b1 b2 | b3 b4), shape (n1, n2, n3, n4). Serial loop over all shell quartets.
np_tensor4 coulomb_4c(const Basis& b1, const Basis& b2, const Basis& b3, const Basis& b4) {
    const auto start = Clock::now();
    LibintSession libint;
    auto engine = make_engine(Operator::coulomb, {&b1, &b2, &b3, &b4});
    const auto& results = engine.results();

    auto ints = zeros<np_tensor4>({b1.size(), b2.size(), b3.size(), b4.size()});
    auto v = ints.view();
    const auto off1 = b1.shell_offsets(), off2 = b2.shell_offsets();
    const auto off3 = b3.shell_offsets(), off4 = b4.shell_offsets();

    for (std::size_t s1 = 0; s1 < b1.nshells(); ++s1) {
        const std::size_t f1 = off1[s1], n1 = b1[s1].size();
        for (std::size_t s2 = 0; s2 < b2.nshells(); ++s2) {
            const std::size_t f2 = off2[s2], n2 = b2[s2].size();
            for (std::size_t s3 = 0; s3 < b3.nshells(); ++s3) {
                const std::size_t f3 = off3[s3], n3 = b3[s3].size();
                for (std::size_t s4 = 0; s4 < b4.nshells(); ++s4) {
                    engine.compute(b1[s1], b2[s2], b3[s3], b4[s4]);
                    const double* buf = results[0];
                    if (!buf)
                        continue;
                    const std::size_t f4 = off4[s4], n4 = b4[s4].size();
                    for (std::size_t i = 0, ijkl = 0; i < n1; ++i)
                        for (std::size_t j = 0; j < n2; ++j)
                            for (std::size_t k = 0; k < n3; ++k)
                                for (std::size_t l = 0; l < n4; ++l, ++ijkl)
                                    v(f1 + i, f2 + j, f3 + k, f4 + l) = buf[ijkl];
                }
            }
        }
    }
    log_msg(4, "[ccpy] Two-electron integrals timing: ", ms_since(start), " ms\n");
    return ints;
}

// (b1 | op | b2 b3), shape (n1, n2, n3), threaded over the shells of b1.
// `prototype` is a fully set-up engine (operator, parameters); each thread uses a copy.
np_tensor3 three_center(const libint2::Engine& prototype, const Basis& b1, const Basis& b2,
                        const Basis& b3) {
    const auto start = Clock::now();
    const std::size_t nb2 = b2.size(), nb3 = b3.size();
    auto ints = zeros<np_tensor3>({b1.size(), nb2, nb3});
    double* data = ints.data();
    const auto off1 = b1.shell_offsets(), off2 = b2.shell_offsets(), off3 = b3.shell_offsets();

    // If b2 and b3 are the same object, (P|mn) = (P|nm): only compute shells s3 <= s2.
    const bool symmetric = (&b2 == &b3);

    parallel_for(0, b1.nshells(), [&](std::size_t, std::size_t s1_begin, std::size_t s1_end) {
        libint2::Engine engine = prototype; // engines are not thread-safe
        const auto& results = engine.results();
        for (std::size_t s1 = s1_begin; s1 < s1_end; ++s1) {
            const std::size_t f1 = off1[s1], n1 = b1[s1].size();
            for (std::size_t s2 = 0; s2 < b2.nshells(); ++s2) {
                const std::size_t f2 = off2[s2], n2 = b2[s2].size();
                const std::size_t s3_end = symmetric ? s2 + 1 : b3.nshells();
                for (std::size_t s3 = 0; s3 < s3_end; ++s3) {
                    engine.compute(b1[s1], b2[s2], b3[s3]);
                    const double* buf = results[0];
                    if (!buf)
                        continue;
                    const std::size_t f3 = off3[s3], n3 = b3[s3].size();
                    put_block3(buf, data, nb2, nb3, f1, n1, f2, n2, f3, n3);
                    if (symmetric && s2 != s3) { // also fill the transposed (P|nm) block
                        for (std::size_t i = 0, ijk = 0; i < n1; ++i)
                            for (std::size_t j = 0; j < n2; ++j)
                                for (std::size_t k = 0; k < n3; ++k, ++ijk)
                                    data[(f1 + i) * nb2 * nb3 + (f3 + k) * nb3 + f2 + j] = buf[ijk];
                    }
                }
            }
        }
    });
    log_msg(4, "[ccpy] Three-center two-electron integrals timing: ", ms_since(start) / 1000.0, " s\n");
    return ints;
}

np_tensor3 coulomb_3c(const Basis& b1, const Basis& b2, const Basis& b3) {
    LibintSession libint;
    auto engine = make_engine(Operator::coulomb, {&b1, &b2, &b3}, 0, BraKet::xs_xx);
    return three_center(engine, b1, b2, b3);
}

np_tensor3 erf_coulomb_3c(const Basis& b1, const Basis& b2, const Basis& b3, double omega) {
    LibintSession libint;
    auto engine = make_engine(Operator::erf_coulomb, {&b1, &b2, &b3}, 0, BraKet::xs_xx);
    engine.set_params(omega);
    return three_center(engine, b1, b2, b3);
}

np_tensor3 erfc_coulomb_3c(const Basis& b1, const Basis& b2, const Basis& b3, double omega) {
    LibintSession libint;
    auto engine = make_engine(Operator::erfc_coulomb, {&b1, &b2, &b3}, 0, BraKet::xs_xx);
    engine.set_params(omega);
    return three_center(engine, b1, b2, b3);
}

// (b1 | b2 b3) restricted to the shell ranges in `slices`, written into the top-left corner
// of `buffer` (which may be larger than needed). Threaded over (s1, s2) shell pairs.
void coulomb_3c_by_shell(const Basis& b1, const Basis& b2, const Basis& b3,
                         const ShellSlices& slices, np_tensor3_c& buffer) {
    const auto start = Clock::now();
    const std::array<std::size_t, 3> nshells{b1.nshells(), b2.nshells(), b3.nshells()};
    for (std::size_t x = 0; x < 3; ++x)
        if (slices[x].first >= slices[x].second)
            throw std::invalid_argument(
                "shell_slices indices must be in the form of (start, end) with start < end");
    for (std::size_t x = 0; x < 3; ++x)
        if (slices[x].second > nshells[x])
            throw std::invalid_argument(
                "shell_slices indices must be within the number of shells in each basis");

    const auto [ish0, ish1] = slices[0];
    const auto [jsh0, jsh1] = slices[1];
    const auto [ksh0, ksh1] = slices[2];
    const auto off1 = b1.shell_offsets(), off2 = b2.shell_offsets(), off3 = b3.shell_offsets();
    const std::size_t nb1 = off1[ish1] - off1[ish0];
    const std::size_t nb2 = off2[jsh1] - off2[jsh0];
    const std::size_t nb3 = off3[ksh1] - off3[ksh0];
    if (buffer.shape(0) < nb1 || buffer.shape(1) < nb2 || buffer.shape(2) < nb3)
        throw std::invalid_argument(
            "Buffer shape (" + std::to_string(buffer.shape(0)) + ", " +
            std::to_string(buffer.shape(1)) + ", " + std::to_string(buffer.shape(2)) +
            ") is too small for the given shell slices with sizes (" + std::to_string(nb1) +
            ", " + std::to_string(nb2) + ", " + std::to_string(nb3) + ")");

    double* data = buffer.data();
    const std::size_t d2 = buffer.shape(1), d3 = buffer.shape(2);
    const std::size_t njsh = jsh1 - jsh0;

    LibintSession libint;
    auto prototype = make_engine(Operator::coulomb, {&b1, &b2, &b3}, 0, BraKet::xs_xx);

    parallel_for(0, (ish1 - ish0) * njsh, [&](std::size_t, std::size_t first, std::size_t last) {
        libint2::Engine engine = prototype;
        const auto& results = engine.results();
        for (std::size_t idx = first; idx < last; ++idx) {
            const std::size_t s1 = ish0 + idx / njsh;
            const std::size_t s2 = jsh0 + idx % njsh;
            for (std::size_t s3 = ksh0; s3 < ksh1; ++s3) {
                engine.compute(b1[s1], b2[s2], b3[s3]);
                const double* buf = results[0];
                if (!buf)
                    continue;
                // positions relative to the start of each slice
                put_block3(buf, data, d2, d3, off1[s1] - off1[ish0], b1[s1].size(),
                           off2[s2] - off2[jsh0], b2[s2].size(), off3[s3] - off3[ksh0],
                           b3[s3].size());
            }
        }
    });
    log_msg(4, "[ccpy] Three-center two-electron integrals timing: ", ms_since(start) / 1000.0, " s\n");
}

// Same, but allocates and returns an array of exactly the slice size.
np_tensor3_c coulomb_3c_by_shell(const Basis& b1, const Basis& b2, const Basis& b3,
                                 const ShellSlices& slices) {
    const std::array<const Basis*, 3> bases{&b1, &b2, &b3};
    std::array<std::size_t, 3> n{};
    for (std::size_t x = 0; x < 3; ++x) {
        const auto [first, last] = slices[x];
        if (first >= last)
            throw std::invalid_argument(
                "shell_slices indices must be in the form of (start, end) with start < end");
        if (last > bases[x]->nshells())
            throw std::invalid_argument(
                "shell_slices indices must be within the number of shells in each basis");
        const auto off = bases[x]->shell_offsets();
        n[x] = off[last] - off[first];
    }
    auto buffer = zeros<np_tensor3_c>({n[0], n[1], n[2]});
    coulomb_3c_by_shell(b1, b2, b3, slices, buffer);
    return buffer;
}

// (b1 | op | b2), shape (n1, n2). The engine must already be set up.
np_matrix two_center(libint2::Engine& engine, const Basis& b1, const Basis& b2) {
    const auto start = Clock::now();
    auto ints = zeros<np_matrix>({b1.size(), b2.size()});
    auto v = ints.view();
    const auto off1 = b1.shell_offsets(), off2 = b2.shell_offsets();
    const auto& results = engine.results();
    for (std::size_t s1 = 0; s1 < b1.nshells(); ++s1) {
        for (std::size_t s2 = 0; s2 < b2.nshells(); ++s2) {
            engine.compute(b1[s1], b2[s2]);
            const double* buf = results[0];
            if (!buf)
                continue;
            for (std::size_t i = 0, ij = 0; i < b1[s1].size(); ++i)
                for (std::size_t j = 0; j < b2[s2].size(); ++j, ++ij)
                    v(off1[s1] + i, off2[s2] + j) = buf[ij];
        }
    }
    log_msg(4, "[ccpy] Two-center two-electron integrals timing: ", ms_since(start) / 1000.0, " s\n");
    return ints;
}

np_matrix coulomb_2c(const Basis& b1, const Basis& b2) {
    LibintSession libint;
    auto engine = make_engine(Operator::coulomb, {&b1, &b2}, 0, BraKet::xs_xs);
    return two_center(engine, b1, b2);
}

np_matrix erf_coulomb_2c(const Basis& b1, const Basis& b2, double omega) {
    LibintSession libint;
    auto engine = make_engine(Operator::erf_coulomb, {&b1, &b2}, 0, BraKet::xs_xs);
    engine.set_params(omega);
    return two_center(engine, b1, b2);
}

np_matrix erfc_coulomb_2c(const Basis& b1, const Basis& b2, double omega) {
    LibintSession libint;
    auto engine = make_engine(Operator::erfc_coulomb, {&b1, &b2}, 0, BraKet::xs_xs);
    engine.set_params(omega);
    return two_center(engine, b1, b2);
}

// =====================================================================================
//  5. Derivative integrals
// =====================================================================================
//
// Each *_deriv function returns the gradient G[3*A + x] = sum W * d(integral)/dR_{A,x}
// for all atoms A in `charges` (atom-major: x1, y1, z1, x2, ...). libint2 returns one
// buffer per (center, Cartesian direction) of the shells in the integral, e.g. 9 buffers
// for a 3-center integral: (center 1: x, y, z), (center 2: x, y, z), (center 3: x, y, z).
// Threaded versions accumulate into one gradient per thread and sum them at the end.

// Check that the atoms of `basis` are, in order, at the positions given in `charges`.
void check_centers(const Basis& basis, const Charges& charges, const std::string& basis_name,
                   const std::string& caller) {
    const auto atoms = basis.center_first_and_last(true);
    if (atoms.size() != charges.size())
        throw std::invalid_argument(caller + ": " + basis_name + " has " +
                                    std::to_string(atoms.size()) + " centers, but charges has " +
                                    std::to_string(charges.size()) + " centers");
    for (std::size_t A = 0; A < atoms.size(); ++A) {
        if (atoms[A].first == atoms[A].second)
            continue;
        const auto& O = basis[atoms[A].first].O;
        const auto& R = charges[A].second;
        const double dist = std::hypot(O[0] - R[0], O[1] - R[1], O[2] - R[2]);
        if (dist > 1.0e-8)
            throw std::invalid_argument(caller + ": " + basis_name + " center " +
                                        std::to_string(A) + " does not match charges center " +
                                        std::to_string(A) + "; distance is " +
                                        std::to_string(dist));
    }
}

// Check that a weight array has the expected shape.
template <typename Array>
void check_shape(const Array& w, const std::vector<std::size_t>& expected, const std::string& name,
                 const std::string& caller) {
    bool ok = true;
    std::string actual = "(", expect = "(";
    for (std::size_t d = 0; d < expected.size(); ++d) {
        ok = ok && (w.shape(d) == expected[d]);
        actual += (d ? ", " : "") + std::to_string(w.shape(d));
        expect += (d ? ", " : "") + std::to_string(expected[d]);
    }
    if (!ok)
        throw std::invalid_argument(caller + ": " + name + " has incorrect shape " + actual +
                                    "); expected " + expect + ")");
}

// Sum the per-thread gradients into a numpy vector.
np_vector sum_gradients(const std::vector<std::vector<double>>& per_thread, std::size_t n) {
    auto grad = zeros<np_vector>({n});
    for (const auto& g : per_thread)
        for (std::size_t i = 0; i < n; ++i)
            grad.data()[i] += g[i];
    return grad;
}

// ---- One-electron derivatives: sum_{mn} D_mn d<m|op|n>/dR ----------------------------
// For the nuclear attraction, libint2 also returns derivatives with respect to the
// position of every point charge (buffers 6, 7, ... = atom 0 x, y, z, atom 1 x, ...).
np_vector one_electron_deriv(Operator op, const Basis& b1, const Basis& b2, const np_matrix& dm,
                             const Charges& charges, const std::string& caller) {
    if (dm.shape(0) != b1.size() || dm.shape(1) != b2.size())
        throw std::invalid_argument(caller + ": density matrix has incorrect shape");
    check_centers(b1, charges, "basis1", caller);
    check_centers(b2, charges, "basis2", caller);

    LibintSession libint;
    auto engine = make_engine(op, {&b1, &b2}, 1);
    const bool depends_on_charges = (op == Operator::nuclear);
    if (depends_on_charges)
        engine.set_params(charges);
    const std::size_t ncomponents = 6 + (depends_on_charges ? 3 * charges.size() : 0);
    const auto& results = engine.results();

    const auto off1 = b1.shell_offsets(), off2 = b2.shell_offsets();
    const auto atom1 = b1.shell_to_center(), atom2 = b2.shell_to_center();
    const auto D = dm.view();
    auto grad = zeros<np_vector>({3 * charges.size()});
    auto g = grad.view();

    for (std::size_t s1 = 0; s1 < b1.nshells(); ++s1) {
        for (std::size_t s2 = 0; s2 < b2.nshells(); ++s2) {
            engine.compute(b1[s1], b2[s2]);
            for (std::size_t c = 0; c < ncomponents; ++c) {
                const double* buf = results[c];
                if (!buf)
                    continue;
                // buffers 0-2: center of shell 1, 3-5: center of shell 2, 6+: point charges
                const std::size_t atom = c < 3 ? atom1[s1] : (c < 6 ? atom2[s2] : (c - 6) / 3);
                const std::size_t xyz = c % 3;
                for (std::size_t i = 0, ij = 0; i < b1[s1].size(); ++i)
                    for (std::size_t j = 0; j < b2[s2].size(); ++j, ++ij)
                        g(3 * atom + xyz) += buf[ij] * D(off1[s1] + i, off2[s2] + j);
            }
        }
    }
    return grad;
}

np_vector overlap_deriv(const Basis& b1, const Basis& b2, const np_matrix& dm,
                        const Charges& charges) {
    return one_electron_deriv(Operator::overlap, b1, b2, dm, charges, "overlap_deriv");
}

np_vector kinetic_deriv(const Basis& b1, const Basis& b2, const np_matrix& dm,
                        const Charges& charges) {
    return one_electron_deriv(Operator::kinetic, b1, b2, dm, charges, "kinetic_deriv");
}

np_vector nuclear_deriv(const Basis& b1, const Basis& b2, const np_matrix& dm,
                        const Charges& charges) {
    return one_electron_deriv(Operator::nuclear, b1, b2, dm, charges, "nuclear_deriv");
}

// ---- Two-center Coulomb derivatives: sum_{PQ} W_PQ d(P|Q)/dR ---------------------------
// Weights may be real or complex (only the real part is used).
template <typename WeightArray>
np_vector coulomb_2c_deriv(const Basis& b1, const Basis& b2, const WeightArray& W2,
                           const Charges& charges) {
    check_centers(b1, charges, "basis1", "coulomb_2c_deriv");
    check_centers(b2, charges, "basis2", "coulomb_2c_deriv");
    const std::size_t nb2 = b2.size();
    check_shape(W2, {b1.size(), nb2}, "W2", "coulomb_2c_deriv");

    LibintSession libint;
    const auto prototype = make_engine(Operator::coulomb, {&b1, &b2}, 1, BraKet::xs_xs);
    const auto off1 = b1.shell_offsets(), off2 = b2.shell_offsets();
    const auto atom1 = b1.shell_to_center(), atom2 = b2.shell_to_center();
    const auto* w = W2.data();
    const std::size_t ngrad = 3 * charges.size();
    std::vector<std::vector<double>> grads(get_num_threads(), std::vector<double>(ngrad, 0.0));

    parallel_for(0, b1.nshells(), [&](std::size_t t, std::size_t s1_begin, std::size_t s1_end) {
        libint2::Engine engine = prototype;
        const auto& results = engine.results();
        auto& g = grads[t];
        for (std::size_t s1 = s1_begin; s1 < s1_end; ++s1) {
            for (std::size_t s2 = 0; s2 < b2.nshells(); ++s2) {
                engine.compute(b1[s1], b2[s2]);
                for (std::size_t c = 0; c < 6; ++c) {
                    const double* buf = results[c];
                    if (!buf)
                        continue;
                    const std::size_t atom = c < 3 ? atom1[s1] : atom2[s2];
                    for (std::size_t i = 0, ij = 0; i < b1[s1].size(); ++i)
                        for (std::size_t j = 0; j < b2[s2].size(); ++j, ++ij)
                            g[3 * atom + c % 3] +=
                                buf[ij] * real_part(w[(off1[s1] + i) * nb2 + off2[s2] + j]);
                }
            }
        }
    });
    return sum_gradients(grads, ngrad);
}

// ---- Three-center Coulomb derivatives: sum_{P mn} W^P_mn d(P|mn)/dR ---------------------
template <typename WeightArray>
np_vector coulomb_3c_deriv(const Basis& b1, const Basis& b2, const Basis& b3,
                           const WeightArray& W3, const Charges& charges) {
    check_centers(b1, charges, "basis1", "coulomb_3c_deriv");
    check_centers(b2, charges, "basis2", "coulomb_3c_deriv");
    check_centers(b3, charges, "basis3", "coulomb_3c_deriv");
    const std::size_t nb2 = b2.size(), nb3 = b3.size();
    check_shape(W3, {b1.size(), nb2, nb3}, "W3", "coulomb_3c_deriv");

    LibintSession libint;
    const auto prototype = make_engine(Operator::coulomb, {&b1, &b2, &b3}, 1, BraKet::xs_xx);
    const auto off1 = b1.shell_offsets(), off2 = b2.shell_offsets(), off3 = b3.shell_offsets();
    const auto atom1 = b1.shell_to_center(), atom2 = b2.shell_to_center(),
               atom3 = b3.shell_to_center();
    const auto* w = W3.data();
    const std::size_t ngrad = 3 * charges.size();
    std::vector<std::vector<double>> grads(get_num_threads(), std::vector<double>(ngrad, 0.0));

    parallel_for(0, b1.nshells(), [&](std::size_t t, std::size_t s1_begin, std::size_t s1_end) {
        libint2::Engine engine = prototype;
        const auto& results = engine.results();
        auto& g = grads[t];
        for (std::size_t s1 = s1_begin; s1 < s1_end; ++s1) {
            const std::size_t f1 = off1[s1], n1 = b1[s1].size();
            for (std::size_t s2 = 0; s2 < b2.nshells(); ++s2) {
                const std::size_t f2 = off2[s2], n2 = b2[s2].size();
                for (std::size_t s3 = 0; s3 < b3.nshells(); ++s3) {
                    const std::size_t f3 = off3[s3], n3 = b3[s3].size();
                    engine.compute(b1[s1], b2[s2], b3[s3]);
                    const std::array<std::size_t, 3> atoms{atom1[s1], atom2[s2], atom3[s3]};
                    for (std::size_t c = 0; c < 9; ++c) {
                        const double* buf = results[c];
                        if (!buf)
                            continue;
                        double& gc = g[3 * atoms[c / 3] + c % 3];
                        for (std::size_t i = 0, ijk = 0; i < n1; ++i)
                            for (std::size_t j = 0; j < n2; ++j)
                                for (std::size_t k = 0; k < n3; ++k, ++ijk)
                                    gc += buf[ijk] *
                                          real_part(w[(f1 + i) * nb2 * nb3 + (f2 + j) * nb3 + f3 + k]);
                    }
                }
            }
        }
    });
    return sum_gradients(grads, ngrad);
}

// ---- Four-center Coulomb derivatives: sum_{mnrs} W_mnrs d(mn|rs)/dR ---------------------
// Only symmetry-unique shell quartets are computed, so W must have the 8-fold symmetry of
// the integrals: W(mn,rs) = W(nm,rs) = W(mn,sr) = W(rs,mn) (true for a symmetrized 2-RDM).
np_vector coulomb_4c_deriv(const Basis& basis, const np_tensor4_c& W4, const Charges& charges) {
    check_centers(basis, charges, "basis", "coulomb_4c_deriv");
    const std::size_t nbf = basis.size();
    check_shape(W4, {nbf, nbf, nbf, nbf}, "W4", "coulomb_4c_deriv");

    LibintSession libint;
    const auto prototype = make_engine(Operator::coulomb, {&basis}, 1);
    const auto off = basis.shell_offsets();
    const auto atom = basis.shell_to_center();

    // symmetry-unique shell pairs (s1 >= s2)
    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    for (std::size_t s1 = 0; s1 < basis.nshells(); ++s1)
        for (std::size_t s2 = 0; s2 <= s1; ++s2)
            pairs.emplace_back(s1, s2);

    const double* w = W4.data();
    const std::size_t ngrad = 3 * charges.size();
    std::vector<std::vector<double>> grads(get_num_threads(), std::vector<double>(ngrad, 0.0));

    parallel_for(0, pairs.size(), [&](std::size_t t, std::size_t p_begin, std::size_t p_end) {
        libint2::Engine engine = prototype;
        const auto& results = engine.results();
        auto& g = grads[t];
        for (std::size_t p12 = p_begin; p12 < p_end; ++p12) {
            const auto [s1, s2] = pairs[p12];
            for (std::size_t p34 = 0; p34 <= p12; ++p34) {
                const auto [s3, s4] = pairs[p34];
                // how many equivalent quartets this unique one stands for
                const double degeneracy = (s1 == s2 ? 1.0 : 2.0) * (s3 == s4 ? 1.0 : 2.0) *
                                          (p12 == p34 ? 1.0 : 2.0);
                const std::array<std::size_t, 4> atoms{atom[s1], atom[s2], atom[s3], atom[s4]};
                engine.compute(basis[s1], basis[s2], basis[s3], basis[s4]);
                for (std::size_t c = 0; c < 12; ++c) {
                    const double* buf = results[c];
                    if (!buf)
                        continue;
                    double sum = 0.0;
                    for (std::size_t i = 0, ijkl = 0; i < basis[s1].size(); ++i)
                        for (std::size_t j = 0; j < basis[s2].size(); ++j)
                            for (std::size_t k = 0; k < basis[s3].size(); ++k) {
                                const std::size_t row =
                                    (((off[s1] + i) * nbf + off[s2] + j) * nbf + off[s3] + k) * nbf +
                                    off[s4];
                                for (std::size_t l = 0; l < basis[s4].size(); ++l, ++ijkl)
                                    sum += buf[ijkl] * w[row + l];
                            }
                    g[3 * atoms[c / 3] + c % 3] += degeneracy * sum;
                }
            }
        }
    });
    return sum_gradients(grads, ngrad);
}

// ---- Cholesky decomposition of (mn|rs) -------------------------------------------------
// The pivoted Cholesky decomposition needs only the diagonal (mn|mn) and the columns
// (mn|rs) of single shell pairs (rs). With the pivots J = (r s) held fixed it is a density
// fit with the pivot pair products as auxiliary functions, (mn|rs) ~ sum_JK (mn|J) M^-1_JK
// (K|rs), M_JK = (J|K); its nuclear derivatives are those of (mn|J) and (J|K).

// (mn|mn) for all function pairs, shape (nbf, nbf).
np_matrix coulomb_4c_diagonal(const Basis& basis) {
    LibintSession libint;
    const auto prototype = make_engine(Operator::coulomb, {&basis});
    const auto off = basis.shell_offsets();
    const std::size_t nbf = basis.size(), ns = basis.nshells();
    auto diag = zeros<np_matrix>({nbf, nbf});
    auto v = diag.view();
    parallel_for(0, ns, [&](std::size_t, std::size_t s_begin, std::size_t s_end) {
        libint2::Engine engine = prototype;
        const auto& results = engine.results();
        for (std::size_t s1 = s_begin; s1 < s_end; ++s1) {
            const std::size_t n1 = basis[s1].size();
            for (std::size_t s2 = 0; s2 < ns; ++s2) {
                const std::size_t n2 = basis[s2].size();
                engine.compute(basis[s1], basis[s2], basis[s1], basis[s2]);
                const double* buf = results[0];
                if (!buf)
                    continue;
                for (std::size_t i = 0; i < n1; ++i)
                    for (std::size_t j = 0; j < n2; ++j)
                        v(off[s1] + i, off[s2] + j) = buf[((i * n2 + j) * n1 + i) * n2 + j];
            }
        }
    });
    return diag;
}

// (mn|rs) for all m, n and the functions r, s of the shells s3, s4, shape (nbf, nbf, n3, n4).
np_tensor4 coulomb_4c_shell_column(const Basis& basis, std::size_t s3, std::size_t s4) {
    LibintSession libint;
    const auto prototype = make_engine(Operator::coulomb, {&basis});
    const auto off = basis.shell_offsets();
    const std::size_t nbf = basis.size(), ns = basis.nshells();
    const std::size_t n3 = basis[s3].size(), n4 = basis[s4].size();
    auto col = zeros<np_tensor4>({nbf, nbf, n3, n4});
    auto v = col.view();
    parallel_for(0, ns, [&](std::size_t, std::size_t s_begin, std::size_t s_end) {
        libint2::Engine engine = prototype;
        const auto& results = engine.results();
        for (std::size_t s1 = s_begin; s1 < s_end; ++s1) {
            const std::size_t n1 = basis[s1].size();
            for (std::size_t s2 = 0; s2 < ns; ++s2) {
                const std::size_t n2 = basis[s2].size();
                engine.compute(basis[s1], basis[s2], basis[s3], basis[s4]);
                const double* buf = results[0];
                if (!buf)
                    continue;
                for (std::size_t i = 0, ijkl = 0; i < n1; ++i)
                    for (std::size_t j = 0; j < n2; ++j)
                        for (std::size_t k = 0; k < n3; ++k)
                            for (std::size_t l = 0; l < n4; ++l, ++ijkl)
                                v(off[s1] + i, off[s2] + j, k, l) = buf[ijkl];
            }
        }
    });
    return col;
}

// Pivots (function pairs) grouped by shell pair: (shell r, shell s, [(P, r - off_r, s - off_s)]).
using PivotGroup = std::tuple<std::size_t, std::size_t,
                              std::vector<std::array<std::size_t, 3>>>;

std::vector<PivotGroup> group_pivots(const Basis& basis,
                                     const std::vector<std::pair<std::size_t, std::size_t>>& pivots) {
    const auto off = basis.shell_offsets();
    std::vector<std::size_t> shell_of(basis.size());
    for (std::size_t s = 0; s < basis.nshells(); ++s)
        for (std::size_t i = 0; i < basis[s].size(); ++i)
            shell_of[off[s] + i] = s;
    std::vector<PivotGroup> groups;
    for (std::size_t P = 0; P < pivots.size(); ++P) {
        const auto [r, s] = pivots[P];
        const std::size_t sr = shell_of[r], ss = shell_of[s];
        auto it = std::find_if(groups.begin(), groups.end(), [&](const PivotGroup& g) {
            return std::get<0>(g) == sr && std::get<1>(g) == ss;
        });
        if (it == groups.end()) {
            groups.emplace_back(sr, ss, std::vector<std::array<std::size_t, 3>>{});
            it = std::prev(groups.end());
        }
        std::get<2>(*it).push_back({P, r - off[sr], s - off[ss]});
    }
    return groups;
}

// sum_P sum_mn W3[P, m, n] d(J_P|mn)/dR for the pivot pairs J_P = (r_P s_P).
np_vector cholesky_3c_deriv(const Basis& basis, const np_tensor3_c& W3,
                                 const std::vector<std::pair<std::size_t, std::size_t>>& pivots,
                                 const Charges& charges) {
    check_centers(basis, charges, "basis", "cholesky_3c_deriv");
    const std::size_t nbf = basis.size(), ns = basis.nshells();
    check_shape(W3, {pivots.size(), nbf, nbf}, "W3", "cholesky_3c_deriv");
    LibintSession libint;
    const auto prototype = make_engine(Operator::coulomb, {&basis}, 1);
    const auto off = basis.shell_offsets();
    const auto atom = basis.shell_to_center();
    const auto groups = group_pivots(basis, pivots);
    const double* w = W3.data();
    const std::size_t ngrad = 3 * charges.size();
    std::vector<std::vector<double>> grads(get_num_threads(), std::vector<double>(ngrad, 0.0));

    parallel_for(0, groups.size(), [&](std::size_t t, std::size_t g_begin, std::size_t g_end) {
        libint2::Engine engine = prototype;
        const auto& results = engine.results();
        auto& g = grads[t];
        for (std::size_t ig = g_begin; ig < g_end; ++ig) {
            const auto& [sr, ss, members] = groups[ig];
            const std::size_t nsz = basis[ss].size();
            for (std::size_t s1 = 0; s1 < ns; ++s1) {
                const std::size_t n1 = basis[s1].size();
                for (std::size_t s2 = 0; s2 < ns; ++s2) {
                    const std::size_t n2 = basis[s2].size();
                    const std::array<std::size_t, 4> atoms{atom[sr], atom[ss], atom[s1], atom[s2]};
                    engine.compute(basis[sr], basis[ss], basis[s1], basis[s2]);
                    for (std::size_t c = 0; c < 12; ++c) {
                        const double* buf = results[c];
                        if (!buf)
                            continue;
                        double sum = 0.0;
                        for (const auto& [P, a, b] : members) {
                            const double* bufab = buf + (a * nsz + b) * n1 * n2;
                            for (std::size_t i = 0; i < n1; ++i) {
                                const double* wrow = w + (P * nbf + off[s1] + i) * nbf + off[s2];
                                for (std::size_t j = 0; j < n2; ++j)
                                    sum += bufab[i * n2 + j] * wrow[j];
                            }
                        }
                        g[3 * atoms[c / 3] + c % 3] += sum;
                    }
                }
            }

        }
    });
    return sum_gradients(grads, ngrad);
}

// sum_PQ W2[P, Q] d(J_P|J_Q)/dR for the pivot pairs.
np_vector cholesky_2c_deriv(const Basis& basis, const np_matrix_c& W2,
                                    const std::vector<std::pair<std::size_t, std::size_t>>& pivots,
                                    const Charges& charges) {
    check_centers(basis, charges, "basis", "cholesky_2c_deriv");
    check_shape(W2, {pivots.size(), pivots.size()}, "W2", "cholesky_2c_deriv");
    LibintSession libint;
    const auto prototype = make_engine(Operator::coulomb, {&basis}, 1);
    const auto atom = basis.shell_to_center();
    const auto groups = group_pivots(basis, pivots);
    const double* w = W2.data();
    const std::size_t npiv = pivots.size();
    const std::size_t ngrad = 3 * charges.size();
    std::vector<std::vector<double>> grads(get_num_threads(), std::vector<double>(ngrad, 0.0));

    parallel_for(0, groups.size(), [&](std::size_t t, std::size_t g_begin, std::size_t g_end) {
        libint2::Engine engine = prototype;
        const auto& results = engine.results();
        auto& g = grads[t];
        for (std::size_t ig = g_begin; ig < g_end; ++ig) {
            const auto& [sr, ss, members] = groups[ig];
            const std::size_t nsz = basis[ss].size();
            for (const auto& [su, sv, others] : groups) {
                const std::size_t nu = basis[su].size(), nv = basis[sv].size();
                const std::array<std::size_t, 4> atoms{atom[sr], atom[ss], atom[su], atom[sv]};
                engine.compute(basis[sr], basis[ss], basis[su], basis[sv]);
                for (std::size_t c = 0; c < 12; ++c) {
                    const double* buf = results[c];
                    if (!buf)
                        continue;
                    double sum = 0.0;
                    for (const auto& [P, a, b] : members)
                        for (const auto& [Q, u, x] : others)
                            sum += buf[((a * nsz + b) * nu + u) * nv + x] * w[P * npiv + Q];
                    g[3 * atoms[c / 3] + c % 3] += sum;
                }
            }
        }
    });
    return sum_gradients(grads, ngrad);
}

// ---- Three-center opVop ----------------------------------------------------------------
// For (P|mn), returns the scalar and the x, y, z Pauli components of the momentum-dressed
// integral sum_P (P| p_m . p_n |...), built from second derivatives with respect to the
// centers of m and n. Four (n2 x n3) matrices in the order scalar, x, y, z (as opVop).
std::array<np_matrix, 4> coulomb_3c_opVop(const Basis& b1, const Basis& b2, const Basis& b3) {
#if !defined(LIBINT2_DERIV_ERI3_ORDER) || LIBINT2_DERIV_ERI3_ORDER < 2
    throw std::runtime_error(
        "Three-center second derivatives are not available in this Libint2 build.");
#else
    const std::size_t nb3 = b3.size();
    std::array<np_matrix, 4> ints;
    for (auto& mat : ints)
        mat = zeros<np_matrix>({b2.size(), nb3});
    if (b1.nshells() == 0 || b2.nshells() == 0 || b3.nshells() == 0)
        return ints;

    // libint2 stores second derivatives as the upper triangle of the 9x9 matrix of
    // center coordinates P(x,y,z)=0-2, m(x,y,z)=3-5, n(x,y,z)=6-8.
    constexpr auto d2 = [](std::size_t a, std::size_t b) { return a * 9 - a * (a + 1) / 2 + b; };
    constexpr std::array<std::size_t, 9> which{
        d2(3, 6), d2(4, 7), d2(5, 8), // mx nx, my ny, mz nz  -> scalar
        d2(4, 8), d2(5, 7),           // my nz, mz ny         -> x
        d2(5, 6), d2(3, 8),           // mz nx, mx nz         -> y
        d2(3, 7), d2(4, 6),           // mx ny, my nx         -> z
    };

    LibintSession libint;
    const auto prototype = make_engine(Operator::coulomb, {&b1, &b2, &b3}, 2, BraKet::xs_xx);
    const auto off2 = b2.shell_offsets(), off3 = b3.shell_offsets();
    std::array<double*, 4> out{ints[0].data(), ints[1].data(), ints[2].data(), ints[3].data()};

    // Threads own disjoint rows (shells of b2), so no locking is needed.
    parallel_for(0, b2.nshells(), [&](std::size_t, std::size_t s2_begin, std::size_t s2_end) {
        libint2::Engine engine = prototype;
        const auto& results = engine.results();
        for (std::size_t s2 = s2_begin; s2 < s2_end; ++s2) {
            for (std::size_t s3 = 0; s3 < b3.nshells(); ++s3) {
                for (std::size_t s1 = 0; s1 < b1.nshells(); ++s1) {
                    engine.compute(b1[s1], b2[s2], b3[s3]);
                    std::array<const double*, 9> buf;
                    for (std::size_t d = 0; d < 9; ++d)
                        buf[d] = results[which[d]];
                    for (std::size_t i = 0, ijk = 0; i < b1[s1].size(); ++i)
                        for (std::size_t j = 0; j < b2[s2].size(); ++j)
                            for (std::size_t k = 0; k < b3[s3].size(); ++k, ++ijk) {
                                std::array<double, 9> d{};
                                for (std::size_t c = 0; c < 9; ++c)
                                    if (buf[c])
                                        d[c] = buf[c][ijk];
                                const std::size_t mn = (off2[s2] + j) * nb3 + off3[s3] + k;
                                out[0][mn] += d[0] + d[1] + d[2];
                                out[1][mn] += d[3] - d[4];
                                out[2][mn] += d[5] - d[6];
                                out[3][mn] += d[7] - d[8];
                            }
                }
            }
        }
    });
    return ints;
#endif
}

// =====================================================================================
//  6. Basis functions and orbitals on points / grids
// =====================================================================================

// Values of all basis functions at the points: shape (npoints, nbasis).
np_matrix basis_at_points(const Basis& basis, const std::vector<std::array<double, 3>>& points) {
    auto values = zeros<np_matrix>({points.size(), basis.size()});
    auto v = values.view();
    double buffer[16]; // enough for l <= 7
    std::size_t first = 0;
    for (std::size_t s = 0; s < basis.nshells(); ++s) {
        const Shell& shell = basis[s];
        if (!shell.contr[0].pure)
            throw std::runtime_error("[ccpy] basis_at_points: Only pure shells are supported");
        for (std::size_t p = 0; p < points.size(); ++p) {
            evaluate_shell(shell, points[p], buffer);
            for (std::size_t i = 0; i < shell.size(); ++i)
                v(p, first + i) += buffer[i];
        }
        first += shell.size();
    }
    return values;
}

// Orbital values sum_mu phi_mu(r) C[mu, j] at the points: shape (npoints, norb).
np_matrix orbitals_at_points(const Basis& basis, const std::vector<std::array<double, 3>>& points,
                             np_matrix C) {
    const std::size_t norb = C.shape(1);
    auto values = zeros<np_matrix>({points.size(), norb});
    auto v = values.view();
    auto c = C.view();
    double buffer[16];
    std::size_t first = 0;
    for (std::size_t s = 0; s < basis.nshells(); ++s) {
        const Shell& shell = basis[s];
        if (!shell.contr[0].pure)
            throw std::runtime_error("[ccpy] orbitals_at_points: Only pure shells are supported");
        for (std::size_t p = 0; p < points.size(); ++p) {
            evaluate_shell(shell, points[p], buffer);
            for (std::size_t i = 0; i < shell.size(); ++i)
                for (std::size_t j = 0; j < norb; ++j)
                    v(p, j) += buffer[i] * c(first + i, j);
        }
        first += shell.size();
    }
    return values;
}

// Orbitals on the regular grid min + i*axis[0] + j*axis[1] + k*axis[2] (k fastest).
np_matrix orbitals_on_grid(const Basis& basis, np_matrix C, const std::array<double, 3> min,
                           const std::array<std::size_t, 3> npoints,
                           const std::vector<std::array<double, 3>> axis) {
    std::vector<std::array<double, 3>> points;
    points.reserve(npoints[0] * npoints[1] * npoints[2]);
    for (std::size_t i = 0; i < npoints[0]; ++i)
        for (std::size_t j = 0; j < npoints[1]; ++j)
            for (std::size_t k = 0; k < npoints[2]; ++k) {
                std::array<double, 3> r;
                for (int x = 0; x < 3; ++x)
                    r[x] = min[x] + i * axis[0][x] + j * axis[1][x] + k * axis[2][x];
                points.push_back(r);
            }
    return orbitals_at_points(basis, points, C);
}

// =====================================================================================
//  7. Python bindings
// =====================================================================================
// Argument names, defaults, docstrings and the order of overloads below are part of the
// Python API used by ccpy.integrals; change them only together with the Python code.

NB_MODULE(_libint, m) {
    m.doc() = "libint2-based atomic-orbital integrals for ccpy (ported from forte2)";

    // ---- Shell: a wrapper around libint2::Shell with a single contraction --------------
    // chi(r) = S_lm(r - O) * sum_i c_i exp(-alpha_i |r - O|^2), S_lm a real solid harmonic.
    nb::class_<Shell>(m, "Shell")
        .def(nb::init<>())
        .def(
            "__init__",
            [](Shell* t, int l, const std::vector<double>& exponents,
               const std::vector<double>& coeffs, const std::array<double, 3>& origin,
               bool is_pure, bool embed_normalization_into_coefficients) {
                libint2::svector<double> alpha(exponents.begin(), exponents.end());
                libint2::svector<double> c(coeffs.begin(), coeffs.end());
                new (t) Shell{alpha, {{l, is_pure, c}}, origin,
                              embed_normalization_into_coefficients};
            },
            "l"_a, "exponents"_a, "coeffs"_a, "center"_a, "is_pure"_a = true,
            "embed_normalization_into_coefficients"_a = true,
            "Construct a shell from the angular momentum (l) and a list of exponents and "
            "coefficients.")
        .def("__repr__",
             [](const Shell& s) {
                 std::string str = "l = " + std::to_string(s.contr[0].l) +
                                   " nprim = " + std::to_string(s.nprim()) + " center = (" +
                                   std::to_string(s.O[0]) + ", " + std::to_string(s.O[1]) + ", " +
                                   std::to_string(s.O[2]) + ")";
                 str += std::format("\n  {:<10} {:<10}", "alpha", "coeff");
                 for (std::size_t i = 0; i < s.nprim(); ++i)
                     str += std::format("\n  {:<10.6f} {:<10.6f}", s.alpha[i], s.contr[0].coeff[i]);
                 return str;
             })
        .def_prop_ro(
            "size", [](Shell& s) { return s.size(); },
            "The number of basis functions in the shell (e.g., for l = 2, size = 5).")
        .def_prop_ro(
            "ncontr", [](Shell& s) { return s.ncontr(); },
            "The number of contractions in the shell.")
        .def_prop_ro(
            "nprim", [](Shell& s) { return s.nprim(); },
            "The number of primitive Gaussians in the shell.")
        .def_prop_ro(
            "l", [](Shell& s) { return s.contr[0].l; }, "The angular momentum of the shell.")
        .def_prop_ro(
            "coeff",
            [](Shell& s) {
                return std::vector<double>(s.contr[0].coeff.begin(), s.contr[0].coeff.end());
            },
            "The coefficients of the primitives in the shell.")
        .def_prop_ro(
            "exponents",
            [](Shell& s) { return std::vector<double>(s.alpha.begin(), s.alpha.end()); },
            "The exponents of the primitives in the shell.")
        .def_prop_ro(
            "is_pure", [](Shell& s) { return s.contr[0].pure; },
            "Is the shell pure? (i.e., we have 5d and 7f functions)")
        .def_prop_ro(
            "center", [](Shell& s) { return s.O; }, "The center of the shell (x, y, z) in bohr.");

    // ---- Basis --------------------------------------------------------------------------
    nb::class_<Basis>(m, "Basis")
        .def(nb::init<>())
        .def("add", &Basis::add, "shell"_a)
        .def("set_name", &Basis::set_name, "name"_a)
        .def("__getitem__", &Basis::operator[], "i"_a)
        .def("__len__", &Basis::size)
        .def(
            "serialize",
            [](const Basis& basis) {
                nb::dict res;
                res["schema_version"] = 1;
                res["nshells"] = basis.nshells();
                nb::list shells;
                for (std::size_t i = 0; i < basis.nshells(); ++i) {
                    const auto& shell = basis[i];
                    nb::dict d;
                    d["nprim"] = shell.nprim();
                    d["l"] = shell.contr[0].l;
                    nb::list exponents, coeffs, center;
                    for (double x : shell.alpha)
                        exponents.append(x);
                    for (double x : shell.contr[0].coeff)
                        coeffs.append(x);
                    for (double x : shell.O)
                        center.append(x);
                    d["exponents"] = exponents;
                    d["coefficients"] = coeffs;
                    d["center"] = center;
                    d["is_pure"] = shell.contr[0].pure;
                    shells.append(d);
                }
                res["shells"] = shells;
                return res;
            },
            "Serialize the basis set to a dictionary.")
        .def_prop_ro("shell_first_and_size", &Basis::shell_first_and_size,
                     "Returns a vector of pairs of the first index and size of each shell in the "
                     "basis set. The first index is the index of the first basis function in the "
                     "shell, and the size is the number of basis functions in the shell.")
        .def_prop_ro("shell_offsets", &Basis::shell_offsets,
                     "Returns a vector of the indices of the first basis function in each shell. "
                     "The last element is the total number of basis functions in the basis set.")
        .def_prop_ro(
            "center_first_and_last", [](const Basis& b) { return b.center_first_and_last(false); },
            "Returns a vector of pairs of the first and last index of the basis functions on a "
            "given center in the basis set.")
        .def_prop_ro(
            "center_first_and_last_shell",
            [](const Basis& b) { return b.center_first_and_last(true); },
            "Returns a vector of pairs of the first and last index of the shells on a given center "
            "in the basis set.")
        .def_prop_ro("size", &Basis::size, "Returns the number of basis functions in the basis set")
        .def_prop_ro("max_l", &Basis::max_l,
                     "Returns the maximum angular momentum of the shells in the basis set")
        .def_prop_ro("name", &Basis::name, "Returns the name of the basis set")
        .def_prop_ro("max_nprim", &Basis::max_nprim,
                     "Returns the maximum number of primitive Gaussians in shells of the basis set")
        .def_prop_ro("nprim", &Basis::max_nprim,
                     "Returns the maximum number of primitive Gaussians in any shell of the "
                     "basis set (alias of max_nprim)")
        .def_prop_ro("max_nbasis", &Basis::max_nbasis,
                     "Returns the maximum number of basis functions in shells of the basis set")
        .def_prop_ro("nshells", &Basis::nshells, "Returns the number of shells in the basis set")
        .def("__repr__", [](const Basis& b) {
            return "<Basis '" + b.name() + "' with " + std::to_string(b.size()) +
                   " basis functions>";
        });

    m.def("shell_label", &shell_label, "l"_a, "idx"_a,
          "Returns a label for a given angular momentum (l) and index (idx).");

    m.def(
        "evaluate_shell",
        [](const Shell& shell, const std::array<double, 3>& point) {
            std::vector<double> values(shell.size());
            evaluate_shell(shell, point, values.data());
            return values;
        },
        "shell"_a, "point"_a,
        "Evaluate the shell at a given point. Returns a list of values for each basis function.");

    m.def("nuclear_repulsion", &nuclear_repulsion, "charges"_a);

    // ---- One-electron integrals ---------------------------------------------------------
    // Most have a (basis1, basis2) form and a single-basis (basis) form meaning <b|op|b>.
    m.def("overlap", &overlap, "basis1"_a, "basis2"_a,
          R"pbdoc(
Compute the overlap integral matrix.

Parameters
----------
b1 : ccpy.integrals.Basis
    First basis set.
b2 : ccpy.integrals.Basis
    Second basis set.

Returns
-------
ndarray, shape = (nb1, nb2)
    Overlap integrals matrix.
)pbdoc");
    m.def("overlap", [](const Basis& b) { return overlap(b, b); }, "basis"_a);

    m.def("kinetic", &kinetic, "basis1"_a, "basis2"_a);
    m.def("kinetic", [](const Basis& b) { return kinetic(b, b); }, "basis"_a);

    m.def("nuclear", &nuclear, "basis1"_a, "basis2"_a, "charges"_a);
    m.def(
        "nuclear", [](const Basis& b, const Charges& q) { return nuclear(b, b, q); }, "basis"_a,
        "charges"_a);

    const std::array<double, 3> zero{0.0, 0.0, 0.0};
    m.def("emultipole1", &multipole<Operator::emultipole1, 4>, "basis1"_a, "basis2"_a,
          "origin"_a = zero);
    m.def(
        "emultipole1",
        [](const Basis& b, const std::array<double, 3>& o) {
            return multipole<Operator::emultipole1, 4>(b, b, o);
        },
        "basis"_a, "origin"_a = zero);

    m.def("emultipole2", &multipole<Operator::emultipole2, 10>, "basis1"_a, "basis2"_a,
          "origin"_a = zero);
    m.def(
        "emultipole2",
        [](const Basis& b, const std::array<double, 3>& o) {
            return multipole<Operator::emultipole2, 10>(b, b, o);
        },
        "basis"_a, "origin"_a = zero);

    m.def("emultipole3", &multipole<Operator::emultipole3, 20>, "basis1"_a, "basis2"_a,
          "origin"_a = zero);
    m.def(
        "emultipole3",
        [](const Basis& b, const std::array<double, 3>& o) {
            return multipole<Operator::emultipole3, 20>(b, b, o);
        },
        "basis"_a, "origin"_a = zero);

    m.def("opVop", &opVop, "basis1"_a, "basis2"_a, "charges"_a);
    m.def(
        "opVop", [](const Basis& b, const Charges& q) { return opVop(b, b, q); }, "basis"_a,
        "charges"_a);

    m.def("erf_nuclear", &erf_nuclear, "basis1"_a, "basis2"_a, "omega_charges"_a);
    m.def("erfc_nuclear", &erfc_nuclear, "basis1"_a, "basis2"_a, "omega_charges"_a);

    // ---- One-electron derivatives -------------------------------------------------------
    m.def("overlap_deriv", &overlap_deriv, "basis1"_a, "basis2"_a, "dm"_a, "charges"_a);
    m.def("kinetic_deriv", &kinetic_deriv, "basis1"_a, "basis2"_a, "dm"_a, "charges"_a);
    m.def("nuclear_deriv", &nuclear_deriv, "basis1"_a, "basis2"_a, "dm"_a, "charges"_a);

    // ---- Two-electron integrals and their derivatives -----------------------------------
    m.def("coulomb_4c", &coulomb_4c, "basis1"_a, "basis2"_a, "basis3"_a, "basis4"_a);
    m.def("coulomb_4c", [](const Basis& b) { return coulomb_4c(b, b, b, b); }, "basis"_a);

    m.def("coulomb_3c", &coulomb_3c, "basis1"_a, "basis2"_a, "basis3"_a);

    m.def("coulomb_3c_by_shell",
          nb::overload_cast<const Basis&, const Basis&, const Basis&, const ShellSlices&>(
              &coulomb_3c_by_shell),
          "basis1"_a, "basis2"_a, "basis3"_a, "shell_slices"_a);
    m.def("coulomb_3c_by_shell",
          nb::overload_cast<const Basis&, const Basis&, const Basis&, const ShellSlices&,
                            np_tensor3_c&>(&coulomb_3c_by_shell),
          "basis1"_a, "basis2"_a, "basis3"_a, "shell_slices"_a, "buffer"_a);

    m.def("coulomb_2c", &coulomb_2c, "basis1"_a, "basis2"_a);
    m.def("coulomb_2c", [](const Basis& b) { return coulomb_2c(b, b); }, "basis"_a);

    m.def("coulomb_3c_deriv", &coulomb_3c_deriv<np_tensor3_c>, "basis1"_a, "basis2"_a,
          "basis3"_a, "W3"_a, "charges"_a);
    m.def("coulomb_3c_opVop", &coulomb_3c_opVop, "basis1"_a, "basis2"_a, "basis3"_a);
    m.def("coulomb_3c_deriv", &coulomb_3c_deriv<np_tensor3_complex_c>, "basis1"_a, "basis2"_a,
          "basis3"_a, "W3"_a, "charges"_a);

    m.def("coulomb_4c_deriv", &coulomb_4c_deriv, "basis"_a, "W4"_a, "charges"_a,
          "Contract an 8-fold symmetric weight tensor with the nuclear derivatives of (mn|rs).");

    m.def("coulomb_4c_diagonal", &coulomb_4c_diagonal, "basis"_a,
          "(mn|mn) for all function pairs, shape (nbf, nbf).");
    m.def("coulomb_4c_shell_column", &coulomb_4c_shell_column, "basis"_a, "shell3"_a, "shell4"_a,
          "(mn|rs) for all m, n and the functions of two shells, shape (nbf, nbf, n3, n4).");
    m.def("cholesky_3c_deriv", &cholesky_3c_deriv, "basis"_a, "W3"_a, "pivots"_a,
          "charges"_a, "sum_P sum_mn W3[P,m,n] d(J_P|mn)/dR for Cholesky pivot pairs J_P.");
    m.def("cholesky_2c_deriv", &cholesky_2c_deriv, "basis"_a, "W2"_a, "pivots"_a,
          "charges"_a, "sum_PQ W2[P,Q] d(J_P|J_Q)/dR for Cholesky pivot pairs.");

    m.def("coulomb_2c_deriv", &coulomb_2c_deriv<np_matrix_c>, "basis1"_a, "basis2"_a, "W2"_a,
          "charges"_a);
    m.def("coulomb_2c_deriv", &coulomb_2c_deriv<np_matrix_complex_c>, "basis1"_a, "basis2"_a,
          "W2"_a, "charges"_a);

    m.def("erf_coulomb_3c", &erf_coulomb_3c, "basis1"_a, "basis2"_a, "basis3"_a, "omega"_a);
    m.def("erf_coulomb_2c", &erf_coulomb_2c, "basis1"_a, "basis2"_a, "omega"_a);
    m.def("erfc_coulomb_3c", &erfc_coulomb_3c, "basis1"_a, "basis2"_a, "basis3"_a, "omega"_a);
    m.def("erfc_coulomb_2c", &erfc_coulomb_2c, "basis1"_a, "basis2"_a, "omega"_a);

    // ---- Values on points and grids -----------------------------------------------------
    m.def("basis_at_points", &basis_at_points, "basis"_a, "points"_a);
    m.def("orbitals_at_points", &orbitals_at_points, "basis"_a, "points"_a, "C"_a,
          "Evaluate the orbitals on a set of points. Returns a 2D array of shape (npoints, "
          "norb).");
    m.def("orbitals_on_grid", &orbitals_on_grid, "basis"_a, "C"_a, "min"_a, "npoints"_a,
          "axis"_a);

    // ---- Module-level settings ----------------------------------------------------------
    m.attr("libint2_max_am") = nb::int_(LIBINT2_MAX_AM);

    m.def(
        "set_log_level", [](int level) { log_level = level; }, "level"_a,
        "Set the C++ logging verbosity (0=NONE ... 5=DEBUG)");
    m.def("get_num_threads", &get_num_threads,
          "Number of threads used by the integral kernels (CCPY_NUM_THREADS_OVERRIDE, "
          "OMP_NUM_THREADS, ...)");
}

} // namespace ccpy
