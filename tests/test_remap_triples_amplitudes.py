import numpy as np
from types import SimpleNamespace
from ccpy.utilities.utilities import remap_triples_amplitudes


def test_remap_triples_amplitudes():
    rng = np.random.default_rng(0)
    spins = ["aaa", "aab", "abb", "bbb"]
    old = {s: rng.permutation(np.arange(1, 61).reshape(10, 6)) for s in spins}
    op = SimpleNamespace(spin_cases=["a", "b"] + spins, dimensions=[(4, 4), (4, 4)] + [(10,)] * 4, ndim=32 + 40)
    for s in spins:
        setattr(op, s, rng.random(10))
    amps = {s: getattr(op, s).copy() for s in spins}
    # appended P space (3 new rows), then sorted in some other order
    perm = rng.permutation(13)
    new = {s: np.vstack((old[s], 100 + np.arange(18).reshape(3, 6)))[perm] for s in spins}

    remap_triples_amplitudes(op, old, new)

    assert op.ndim == 32 + 52
    for k, s in enumerate(spins):
        assert op.dimensions[2 + k] == (13,)
        expected = np.concatenate((amps[s], np.zeros(3)))[perm]
        assert np.array_equal(getattr(op, s), expected)
