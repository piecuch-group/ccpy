"""Moller-Plesset perturbation theory (MP2 with DF integrals; MP3 and MP4 legacy)."""
from importlib import import_module

from ccpy.methods.base import MethodBase


class MBPTBase(MethodBase):
    """Base class of MPn energies (scheme: MBPT); no amplitudes are iterated."""

    function = None

    def __init__(self, upstream, nfrozen=None, ndelete=None, **options):
        super().__init__(upstream, nfrozen, ndelete, **options)
        self.correlation_energy = 0.0

    @property
    def hamiltonian(self):
        if self.requires_df_port:
            return self.reference.legacy_hamiltonian(copy_=False)
        return self.reference.H

    def solve(self):
        self._warn_df_port()
        function = getattr(import_module("ccpy.mbpt.mbpt"), self.function)
        self.correlation_energy = function(self.system, self.hamiltonian)
        self._finish_solve(True)
        return self

    def energy(self):
        self.ensure_solved()
        return self.reference_energy + self.correlation_energy


class MP2(MBPTBase):
    """MP2 with DF/Cholesky integrals."""
    function = "calc_mp2"


class MP3(MBPTBase):
    function = "calc_mp3"
    requires_df_port = True


class MP4(MBPTBase):
    function = "calc_mp4"
    requires_df_port = True
