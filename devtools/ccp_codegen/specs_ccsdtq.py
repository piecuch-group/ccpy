"""CCSDTQ(P) terms expressed as generator specs."""
from codegen import Piece, T4BlockTerm

# I3A_vvooov(cdmkle) and I3B_vvooov(cdm~kle~) of ccsdtq.update_t4a, block q = (c,d,k,l) = (q1,q2,q3,q4),
# after the A(kl)A(cd) antisymmetrization done in ccsdtq.py. Stored integrals (e,m first):
#   H2A_ooov_r(e,m,n,k) = h2a_ooov(n,m,k,e)    H2A_vovv_r(e,m,f,c) = h2a_vovv(c,m,f,e)
#   H2A_oovv_r(e,m,f,n) = v2a_oovv(m,n,e,f)    H2B_oovv_r(e,m,f~,n~) = v2b_oovv(m,n~,e,f~)
#   H2B_ooov_r(e~,m~,n,k) = h2b_ooov(n,m~,k,e~)  H2B_vovv_r(e~,m~,f,c) = h2b_vovv(c,m~,f,e~)
#   H2C_oovv_r(e~,m~,f~,n~) = v2c_oovv(m~,n~,e~,f~)
T4A_VVOOOV = T4BlockTerm(
    header=["!!!! diagram: A(cd/ab)A(kl/ij) [I3A(cdmkle)*t3a(abeijm) + I3B(cdm~kle~)*t3b(abe~ijm~)]"],
    pieces=[
        Piece("IA", -1.0, "H2A_ooov_r", ("n", "q3"), "t2a(q1,q2,n,q4)", ("n", "noa")),
        Piece("IA", 1.0, "H2A_ooov_r", ("n", "q4"), "t2a(q1,q2,n,q3)", ("n", "noa")),
        Piece("IA", 1.0, "H2A_vovv_r", ("f", "q1"), "t2a(f,q2,q3,q4)", ("f", "nua")),
        Piece("IA", -1.0, "H2A_vovv_r", ("f", "q2"), "t2a(f,q1,q3,q4)", ("f", "nua")),
        Piece("IA", 0.5, "H2A_oovv_r", source="a"),
        Piece("IA", 1.0, "H2B_oovv_r", source="b"),
        Piece("IB", -1.0, "H2B_ooov_r", ("n", "q3"), "t2a(q1,q2,n,q4)", ("n", "noa")),
        Piece("IB", 1.0, "H2B_ooov_r", ("n", "q4"), "t2a(q1,q2,n,q3)", ("n", "noa")),
        Piece("IB", 1.0, "H2B_vovv_r", ("f", "q1"), "t2a(f,q2,q3,q4)", ("f", "nua")),
        Piece("IB", -1.0, "H2B_vovv_r", ("f", "q2"), "t2a(f,q1,q3,q4)", ("f", "nua")),
        Piece("IB", 0.5, "H2C_oovv_r", source="b"),
    ],
    blocks={"IA": ("nua", "noa"), "IB": ("nub", "nob")},
    contract={"a": "IA", "b": "IB"},
    shared="H2A_ooov_r,H2A_vovv_r,H2A_oovv_r,H2B_ooov_r,H2B_vovv_r,H2B_oovv_r,H2C_oovv_r",
)


T4A_VVOOOV_ROUTINE = """  subroutine t4a_vvooov(resid, t4a_amps, t4a_excits, t3a_amps, t3a_excits, t3b_amps, t3b_excits, t2a, &
                        H2A_ooov_r, H2A_vovv_r, H2A_oovv_r, H2B_ooov_r, H2B_vovv_r, H2B_oovv_r, H2C_oovv_r, &
                        n4aaaa, n3aaa, n3aab, noa, nua, nob, nub)
                  integer, intent(in) :: n4aaaa, n3aaa, n3aab, noa, nua, nob, nub
                  integer, intent(inout) :: t4a_excits(n4aaaa,8)
                  !f2py intent(in,out) :: t4a_excits(0:n4aaaa-1,0:7)
                  real(kind=8), intent(inout) :: t4a_amps(n4aaaa)
                  !f2py intent(in,out) :: t4a_amps(0:n4aaaa-1)
                  integer, intent(in) :: t3a_excits(n3aaa,6), t3b_excits(n3aab,6)
                  real(kind=8), intent(in) :: t3a_amps(n3aaa), t3b_amps(n3aab), t2a(nua,nua,noa,noa)
                  real(kind=8), intent(in) :: H2A_ooov_r(nua,noa,noa,noa), H2A_vovv_r(nua,noa,nua,nua), &
                                              H2A_oovv_r(nua,noa,nua,noa), H2B_oovv_r(nua,noa,nub,nob), &
                                              H2B_ooov_r(nub,nob,noa,noa), H2B_vovv_r(nub,nob,nua,nua), &
                                              H2C_oovv_r(nub,nob,nub,nob)
                  real(kind=8), intent(out) :: resid(n4aaaa)
                  integer, allocatable :: idx_table(:,:,:,:), loc_arr(:,:), idx_table3(:,:,:,:), loc_arr3a(:,:), loc_arr3b(:,:)
                  real(kind=8), allocatable :: t3_amps_buff(:), t3b_amps_buff(:), IA(:,:), IB(:,:)
                  integer, allocatable :: t3_excits_buff(:,:), t3b_excits_buff(:,:)
                  real(kind=8) :: t_amp, val
                  integer :: a, b, c, d, i, j, k, l, f, n, idet, jdet, kdx, idx, nloc, nloc3
                  resid = 0.0d0
{body}
  end subroutine t4a_vvooov
"""


def t4a_vvooov_routine():
    """Complete subroutine t4a_vvooov (one term of the future update_t4a_p)."""
    return T4A_VVOOOV_ROUTINE.format(body=T4A_VVOOOV.emit())
