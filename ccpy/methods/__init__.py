"""Object-oriented CCpy API built on DF/Cholesky integrals.

Each scheme is a base class (CCBase, LeftCCBase, EOMCCBase, CorrectionBase, ...) and each
level a subclass (CCSD, CCSDT, CCSDT_P, ...). All methods expose solve(), energy(),
gradient(), and energy_and_gradient():

    mol = Molecule(geometry, "cc-pvdz", auxiliary_basis="cc-pvdz-rifit")
    mf = RHF(mol)
    e_ccsd = CCSD(mf, nfrozen=1).energy()
    e_crcc23 = CRCC23(mf, nfrozen=1).energy()

Methods with ``requires_df_port = True`` (listed by ``methods_requiring_df_port()``) still
solve four-index equations, with the integrals (vvvv included) materialized from the DF
vectors, and issue a DFPortWarning.
"""
from ccpy.methods.base import MethodBase
from ccpy.methods.cc import (ACCD, ACCSD, ACCSDT_P, CC2, CC3, CC4, CCD, CCSD, CCSDT, CCSDT_P, CCSDTQ, ECCC2,
                             ACCBase, CCBase, CCPBase)
from ccpy.methods.corrections import (CCP3, CCP4, CCT3, AdaptiveCCPQ, CREACC23, CREOMCC23, CRCC23, CRCC24,
                                      CRCC34, CRIPCC23,
                                      DIPEOMCCSDTA_star, EACCP3, EAEOMCCSDTA_star, EOMCCP3, EOMCCSDTA_star,
                                      EOMCCT3, IPCCP3, IPEOMCCSDTA_star, CCSDpT, CorrectionBase,
                                      ExcitedStateCorrectionBase, GroundStateCorrectionBase)
from ccpy.methods.eomcc import (DEAEOM2, DEAEOM3, DEAEOM4, DEAEOM4_P, DIPEOM3, DIPEOM4, DIPEOMCCSDT,
                                DIPEOMCCSDTA, EAEOM2, EAEOM3, EAEOM3_P, EAEOMCCSDT, EOMCC2, EOMCC3, EOMCCSD,
                                EOMCCSDT, EOMCCSDT_P, EOMCCSDTA, IPEOM2, IPEOM3, IPEOM3_P, IPEOMCCSDT,
                                IPEOMCCSDT_P, IPEOMCCSDTA, SFEOMCC23, SFEOMCCSD, EOMCCBase)
from ccpy.methods.hamiltonian import DFHamiltonian
from ccpy.methods.hbar import (HBarBase, HBarCC2, HBarCC3, HBarCCSD, HBarCCSDLegacy, HBarCCSDT, HBarCCSDT1,
                               HBarCCSDTA, HBarCCSDT_P)
from ccpy.methods.left import (LeftCC3, LeftCCBase, LeftCCSD, LeftCCSDT, LeftCCSDT_P, LeftEAEOM2, LeftEAEOM3,
                               LeftEAEOM3_P, LeftEOMCCBase, LeftEOMCCSD, LeftEOMCCSDT, LeftEOMCCSDT_P,
                               LeftEOMCCSDTA, LeftIPEOM2, LeftIPEOM3, LeftIPEOM3_P)
from ccpy.methods.lrcc import LRCCSD, LRCCBase
from ccpy.methods.mbpt import MP2, MP3, MP4, MBPTBase
from ccpy.methods.reference import DFPortWarning, Reference


__all__ = [name for name in dir() if not name.startswith("_") and name[0].isupper()] + ["methods_requiring_df_port"]


def methods_requiring_df_port():
    """Names of the methods (and HBars) whose equations still use four-index integrals."""
    def leaves(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from leaves(sub)

    names = {c.__name__ for base in (MethodBase, HBarBase) for c in leaves(base)
             if c.requires_df_port is True and not c.__name__.endswith("Base")}
    return sorted(names)
