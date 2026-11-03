"""General-reference spin-Fock partition for DF-ADC(2).

Occupied and virtual spin blocks are internally semicanonicalized within irreps;
the input Reference is never changed. The occupied-virtual Fock operator belongs
to the perturbation, with nonzero first-order singles t_i^a = -f_ia/Delta_ia.
See Stahl, Banerjee and Sokolov, JCP 157, 044106 (2022), Eqs. 14, 27, 28.
"""
import numpy as np

from ccpy.mbpt.dfmp2 import _diagonalize_blocks


def einsum(expression, *arrays):
    return np.einsum(expression, *arrays, optimize=True)


class ADCReference:
    def __init__(self, reference):
        self.reference = reference
        sys, H = reference.system, reference.H
        na, nb, va, vb = sys.noccupied_alpha, sys.noccupied_beta, sys.nunoccupied_alpha, sys.nunoccupied_beta
        self.sizes = na, nb, va, vb
        self.rotations = {}
        self.oo = np.zeros((H.naux, na + nb, na + nb))
        self.ov = np.zeros((H.naux, na + nb, va + vb))
        self.vv = np.zeros((H.naux, va + vb, va + vb))
        self.fov = np.zeros((na + nb, va + vb))
        eo, ev = [], []
        for s, no, os, vs in (("a", na, slice(0, na), slice(0, va)),
                               ("b", nb, slice(na, na + nb), slice(va, va + vb))):
            f, B = getattr(H, s), getattr(H.chol, s)
            labels = sys.orbital_symmetries_all[s] if reference.is_uhf else sys.orbital_symmetries_all
            labels = labels[reference.nfrozen:reference.nfrozen + sys.norbitals]
            eps_o, Uo = _diagonalize_blocks(f.oo, labels[:no])
            eps_v, Uv = _diagonalize_blocks(f.vv, labels[no:])
            self.rotations[s] = {"o": Uo, "v": Uv}
            self.oo[:, os, os] = einsum("Pij,ik,jl->Pkl", B.oo, Uo, Uo)
            self.ov[:, os, vs] = einsum("Pia,ij,ab->Pjb", B.ov, Uo, Uv)
            self.vv[:, vs, vs] = einsum("Pab,ac,bd->Pcd", B.vv, Uv, Uv)
            self.fov[os, vs] = Uo.T @ f.ov @ Uv
            eo.extend(eps_o)
            ev.extend(eps_v)
        self.eo, self.ev = np.asarray(eo), np.asarray(ev)
        self.delta1 = self.ev[None, :] - self.eo[:, None]
        self.delta2 = self.delta1[:, None, :, None] + self.delta1[None, :, None, :]
        if (np.min(np.abs(self.delta1), initial=np.inf) < 1.e-12 or
                np.min(np.abs(self.delta2), initial=np.inf) < 1.e-12):
            raise ValueError("Singular ADC(2) perturbation denominator.")
        self.t1 = -self.fov / self.delta1
        g = einsum("Pia,Pjb->ijab", self.ov, self.ov) - einsum("Pib,Pja->ijab", self.ov, self.ov)
        self.t = -g / self.delta2
        self.correlation_energy = float(0.25 * np.sum(g * self.t) + np.sum(self.fov * self.t1))
        occupied = einsum("imef,jmef->ij", self.t, g)
        virtual = einsum("mnae,mnbe->ab", self.t, g)
        self.occupied = -0.25 * (occupied + occupied.T)
        self.virtual = -0.25 * (virtual + virtual.T)
        self.cross_diagonal = einsum("imae,imae->ia", self.t, g)
        # First-order orbital generator: K_vo=t1.T, K_ov=-t1.
        charge = 2 * einsum("Pia,ia->P", self.ov, self.t1)
        do = einsum("Pij,P->ij", self.oo, charge)
        dv = einsum("Pab,P->ab", self.vv, charge)
        tmp = einsum("Pia,ma->Pim", self.ov, self.t1)
        do -= einsum("Pim,Pmj->ij", tmp, self.oo)
        do -= einsum("Pim,Pjm->ij", self.oo, tmp)
        tmp = einsum("Pia,ic->Pac", self.ov, self.t1)
        dv -= einsum("Pac,Pcb->ab", tmp, self.vv)
        dv -= einsum("Pac,Pbc->ab", self.vv, tmp)
        self.single_occupied = do + 0.5 * (self.fov @ self.t1.T + self.t1 @ self.fov.T)
        self.single_virtual = dv - 0.5 * (self.fov.T @ self.t1 + self.t1.T @ self.fov)
        self.doo = einsum("Pia,ja->Pij", self.ov, self.t1) + einsum("ia,Pja->Pij", self.t1, self.ov)
        self.dvv = -einsum("Pia,ib->Pab", self.ov, self.t1) - einsum("ia,Pib->Pab", self.t1, self.ov)
        self.dov = einsum("ib,Pba->Pia", self.t1, self.vv) - einsum("Pij,ja->Pia", self.oo, self.t1)

    def rotate(self, array, spins, kinds, inverse=False):
        """Tensor axes from working MOs to internal MOs, or back."""
        value = array
        for axis, (s, kind) in enumerate(zip(spins, kinds)):
            U = self.rotations[s][kind]
            value = np.moveaxis(np.tensordot(U if inverse else U.T, value, axes=(1, axis)), 0, axis)
        return value

    def density_to_working(self, D):
        va = self.sizes[2]
        result = {}
        for s, block in (("a", D[:va, :va]), ("b", D[va:, va:])):
            U = self.rotations[s]["v"]
            result[s] = U @ (0.5 * (block + block.T)) @ U.T
        return result

    def symmetry_mask(self, irrep):
        """Independent determinants of a target total-state irrep."""
        sys = self.reference.system
        if irrep is None or sys.point_group == "C1":
            return np.ones(self.space.ndim, dtype=bool)
        numbers = sys.point_group_irrep_to_number
        occupied, virtual = [], []
        for spin, no in zip(("a", "b"), self.sizes[:2]):
            labels = sys.orbital_symmetries_all[spin] if self.reference.is_uhf else sys.orbital_symmetries_all
            labels = labels[self.reference.nfrozen:self.reference.nfrozen + sys.norbitals]
            occupied.extend(numbers[x] for x in labels[:no])
            virtual.extend(numbers[x] for x in labels[no:])
        o, v = np.asarray(occupied), np.asarray(virtual)
        target = numbers[irrep] ^ numbers[sys.reference_symmetry]
        sector = getattr(self, "sector", "EE")
        if sector == "IP":
            x, y = o, v[:, None, None] ^ o[None, :, None] ^ o[None, None, :]
        elif sector == "EA":
            x, y = v, o[:, None, None] ^ v[None, :, None] ^ v[None, None, :]
        else:
            x = o[:, None] ^ v[None, :]
            y = o[:, None, None, None] ^ o[None, :, None, None] ^ v[None, None, :, None] ^ v[None, None, None, :]
        return self.space.pack(x == target, y == target).astype(bool)
