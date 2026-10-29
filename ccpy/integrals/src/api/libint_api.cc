#include <nanobind/nanobind.h>

#include "helpers/logger.h"
#include "helpers/parallel.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace ccpy {

void export_ints_api(nb::module_& m);

NB_MODULE(_libint, m) {
    m.doc() = "libint2-based atomic-orbital integrals for ccpy (ported from forte2)";

    export_ints_api(m);

    m.def(
        "set_log_level",
        [](int level) { Logger::getInstance().setLevel(static_cast<Logger::Level>(level)); },
        "level"_a, "Set the C++ logging verbosity (0=NONE ... 5=DEBUG)");
    m.def("get_num_threads", &get_num_threads,
          "Number of threads used by the integral kernels (CCPY_NUM_THREADS_OVERRIDE, "
          "OMP_NUM_THREADS, ...)");
}
} // namespace ccpy
