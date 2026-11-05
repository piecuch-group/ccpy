"""The DF vvvv ladder kernels call BLAS inside OpenMP parallel regions. That is only correct
when the extension and its BLAS share one OpenMP runtime (see ccpy/lib/meson.build); a build
that mixes runtimes gives wrong results with more than one thread. Run in subprocesses, since
the OpenMP runtime reads OMP_NUM_THREADS when it loads."""
import os
import subprocess
import sys

import pytest

CHECK = """
import numpy as np
from ccpy.lib.core import vvvv_contraction as vc
rng = np.random.default_rng(0)
naux, v, o = 30, 40, 7
R = rng.standard_normal((naux, v, v))
t = rng.standard_normal((v, v, o, o))
t = t - t.transpose(1, 0, 2, 3)
t = t - t.transpose(0, 1, 3, 2)
ref = np.einsum("xae,xbf,efij->abij", R, R, t, optimize=True)
g = vc.vvvv_t2(R.transpose(0, 2, 1), R.transpose(0, 2, 1), t.transpose(3, 2, 1, 0)).transpose(3, 2, 1, 0)
s = vc.vvvv_t2_sym(R.transpose(0, 2, 1), t.transpose(3, 2, 1, 0)).transpose(3, 2, 1, 0)
s = s - s.transpose(1, 0, 2, 3)
print(max(np.abs(g - ref).max(), np.abs(s - ref).max()) / np.abs(ref).max())
"""


@pytest.mark.parametrize("threads", [1, 4])
def test_ladder_kernels_with_threads(threads):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    out = subprocess.run([sys.executable, "-c", CHECK], env=env, capture_output=True, text=True, check=True)
    assert float(out.stdout.split()[-1]) < 1e-12
