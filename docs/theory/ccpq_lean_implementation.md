# A lean total-gradient implementation for excited-state CC(P;Q)

This is an exact algebraic reduction of the current real, normalized,
isolated-root functional. It is a proposed production route, not a claim
that a timed production implementation is already complete.

Reuse the converged CC(P) amplitudes T and projected right/left EOM vectors
r and l. All state vectors below include the reference component: r has
r0 in general, even though the physical excited l0 is zero. Let

\[
A=\overline H_{\{0,P\},\{0,P\}},\qquad B_\mu=A-E_\mu I,
\qquad l^T r=1.
\]

B_mu is the shifted projected EOM operator. It is distinct from the
CC amplitude-residual Jacobian J for a general nonclosed P.

## 1. Construct Q quantities and state sources once

\[
u_K=M_K/D_K,\qquad v_K=N_K/D_K,\qquad
w_K=u_Kv_K,\quad W=\sum_Kw_K,\quad\delta=\sum_KN_KM_K/D_K.
\]

The correction derivatives at fixed T and independent E are

\[
g_r=\overline H_{Q,\{0,P\}}^T v,\qquad
g_l=\overline H_{\{0,P\},Q}u,\qquad g_E=-W.
\]

There is no separate response solve for each Q determinant or each nuclear
coordinate. Selected-list tensor contractions can accumulate these sources.

## 2. Solve two projected EOM response equations

Compute x and y from

\[
B_\mu x=-(g_l-\delta r),\qquad l^Tx=0,
\]
\[
B_\mu^T y=-(g_r-\delta l),\qquad r^Ty=0.
\]

The solvability conditions follow from l^T g_l = r^T g_r = delta. These
are singular operators restricted away from the target root: use a
projector or a bordered solve, not an ordinary inverse of B_mu. For example,

\[
\begin{pmatrix}B_\mu&r\\l^T&0\end{pmatrix}
\begin{pmatrix}x\\\xi\end{pmatrix}
=\begin{pmatrix}-(g_l-\delta r)\\0\end{pmatrix},
\]

and use the transpose of that same bordered operator for y. The extra scalar
vanishes on an exactly converged, consistent right-hand side. Both solves
can reuse the same underlying EOM action and preconditioning structure.

To relate this to the original adjoints, choose

\[
a=y-Wl,\qquad b=x,\qquad \nu_{\rm total}=-E_\mu-\delta.
\]

Then a^T r + b^T l = -W, exactly satisfying the independent-energy response
equation. The original pivot gauge b_q=0 can be recovered by moving a
multiple of r from b into the corresponding multiple of l in a. This leaves
both the density coefficient and cluster source unchanged.

For the base objective alone, a_b=b_b=0 and nu_b=-E_mu are already known.
The generic base state-response solve is unnecessary.

## 3. Merge the total Hbar coefficient, then solve one cluster adjoint

Embed x,y,r,l in the reference+P support, and u_Q,v_Q in Q. Define

\[
\mathcal A_0=
\big[(1-W)l+y+v_Q\big]r^T
+l(x+u_Q)^T+\mathcal D_\mu,
\]

where

\[
\mathcal D_\mu^{\rm full}=\sum_{K\in Q}w_K|K\rangle\langle K|,
\qquad
\mathcal D_\mu^{\rm mp}=W|0\rangle\langle0|.
\]

Solve a single combined cluster-response equation

\[
J^T z_t=-q_t,\qquad
(q_t)_I=\sum_{AB}(\mathcal A_0)_{AB}
                  [\overline H,\tau_I]_{AB}.
\]

All factors are held fixed in this contraction. The final total coefficient is

\[
\boxed{\mathcal A_{\rm total}=\mathcal A_0+z_t e_0^T.}
\]

It has **three outer products plus the denominator diagonal**. Do not expand
it into a dense full-Hilbert-space matrix. Do not multiply two effective
corrected state vectors together: that would introduce cross terms absent
from this functional. The factor (1-W) follows from the denominator energy
response; it is not a modification of the noniterative energy formula.

For MP, retain the explicit Fock derivative

\[
d_p=-\sum_Kw_K\chi_{Kp},
\]

and its 1-/2-body denominator density exactly as in the current theory.
The projected reduction does not remove that contribution.

## 4. Build and relax the total density once

Pull back the three factors through exp(-T) and exp(T), or evaluate the
corresponding connected tensor contractions. Add the full-diagonal or
explicit Fock density. Perform one total orbital Z-vector solve and one
AO derivative contraction, including the overlap term and DF metric terms.

A production SD/selected-T implementation should use connected tensor and
selected-list kernels, rather than materializing full determinant support.
The equations themselves do not establish a particular tensor scaling;
that requires a contraction-by-contraction implementation audit.

## 5. Cache what does not change

At a fixed geometry and P selection, T, Hbar, the cluster Jacobian and HF
response operator are shared. R/L are shared across denominator variants;
the base gradient can be cached if component reporting is requested.
Correction sources change with the denominator, so their solutions cannot
simply be reused unchanged. Shifted/projected EOM operators are state-specific,
although their underlying operator and preconditioner structure can be reused.

If only the total gradient is needed, avoid separate base/correction cluster
solves and density builds. If diagnostic components are needed, compute the
base response once and reuse it, with its trivial state adjoint imposed
analytically. This preserves the two-functional interpretation.

## Verification and implementation status

`devtools/check_ccpq_lean_response.py` compares the merged bare-Hamiltonian
gradient coefficient with the dense constrained complex-step oracle for
closed/nonclosed P, two excited roots, and full/MP denominators: eight cases.
The maximum discrepancy is 5.55e-15; results are in
`ccpq_lean_response_checks.json`. The unchanged explicit MP Fock density must
still be added when differentiating physical integrals.

This verifies the algebraic reduction. The current production/oracle code
has not been replaced by this route, and no molecular speedup is claimed yet.
Implementation should retain the existing path as an oracle, verify separate
density contractions and nuclear gradients, then benchmark each phase.
