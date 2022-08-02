module ccp_opt_loops

!!!!      USE OMP_LIB
!!!!      USE MKL_SERVICE


![TODO]:
!    1. Fully devectorize the T3 updates as well as the terms in T1 and T2 updates that involve T3.
!    2. Remove tensor structure of T3 and use linear vector with indices addressed consistently using pspace arrays.
!       E.g., pspace[a, b, c, i, j, k] = idx, where t3[idx] = t3[a, b, c, i, j, k]. So keep storage in terms of pspace,
!             but remove storage from t3. This will reduce current storage requirements as well as speed up DIIS and flatten operations.
!    3. Is there a way to build the Hbar matrix directly using this same loop structure?
!    4. Remove inner if statements using pre-ordering. Define idx_dgm_x[a, b, c, i, j, k, :] = idx,
!       where idx[p] is the linearized index referring to the tuple of contracted t3. For example, if we want to compute
!       0.5 * h2a(abef) * t3a(efcijk), then we would compute this as
!
!       do a, b, c, i, j, k in pspace:
!            do idx in idx_dgm_vvvv[a, b, c, i, j, k, :]:
!                 residual = residual + h2a[a, b, e, f] * t3a[idx] (how to we get e, f ??)
!
!       Using linear indexing, we can define a 2D addressing array A_dgm[idx, idx], where A_dgm[idx, :] enumerates all
!       linear indices in t3 used in the contraction. Associated with A_dgm, we can define B_dgm[idx, idx], where
!       B_dgm[idx, :] defines a linearized index for the Hbar slice used in the contraction.
!       For instance, using the above vvvv-type contraction as an example,
!
!       do idet in pspace:
!            do jdet in A_dgm[idx, :]:
!                 residual = residual + h2a[B_dgm[idet, jdet]] * t3a[A_dgm[idet, jdet]]
!
!       This is likely the best way to do the optimized CC(P)!!!!
!
! Update 8/01/2022:
!  (1) What we can also do is a preparatory pre-run using p_space array in the following way:
    ! For each diagram and each unique permutation, we have a dry run looping over, e.g., A(c/ab) h(abef) * t3(ebcijk).
    ! We use a hashing function to map tuples (a, b, c, i, j, k) into a unique linear index in t3 and another hashing
    ! function to map tuples (p, q, r, s) into a unique Hbar. Note that this dry run has CPU cost that scales linearly
    ! with the size of the P space since the hashing function is likely a few FLOPs and it is done for all determinants
    ! in the P space (and all potentially connecting determinants).
    ! cnt = 0
    ! do idx in pspace:
    !
    !     do e = 1, nua
    !        do f = e + 1, nua
    !           if (pspace(hash_fcn(e, f, c, i, j, k)) == 1) then
    !               cnt = cnt + 1
    !           end if
    !        end do
    !     end do
    !
    !end do
    !
    ! allocate(A(ndet_p, cnt, 2))
    ! do idx in pspace:
    !
    !     do e = 1, nua
    !        do f = e + 1, nua
    !           if (pspace(hash_fcn(e, f, c, i, j, k) == 1) then
    !               A(idx, cnt) = hash_fcn(e, f, c, i, j, k) <- for T
    !               B(idx, cnt) = hash_fcn(a, b, e, f) <- for H
    !           end if
    !        end do
    !     end do
    !
    !end do
    !
    ! Afterwards, the iterative steps can be done as
    !
    !    do idet = 1, ndim_p
    !       do jdet = 1, size(A, 2)
    !          X3(idet) = X3(idet) + H(B(idet, jdet)) * t3(jdet)
    !       end do
    !    end do


      implicit none


      contains


          !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
          !!!!!!!!!!!!!!!!!!!!!!!! OPT 1 !!!!!!!!!!!!!!!!!!!
          !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!                subroutine update_t1a_opt1(t1a, resid, X1A,&
!                                           t3a, t3b, t3c,&
!                                           pspace_aaa, pspace_aab, pspace_abb,&
!                                           vA_oovv, vB_oovv, vC_oovv,&
!                                           fA_oo, fA_vv,&
!                                           shift,&
!                                           noa, nua, nob, nub)
!
!                      integer, intent(in) :: noa, nua, nob, nub
!                      integer, intent(in) :: pspace_aaa(nua, nua, nua, noa, noa, noa),&
!                                             pspace_aab(nua, nua, nub, noa, noa, nob),&
!                                             pspace_abb(nua, nub, nub, noa, nob, nob)
!                      real(kind=8), intent(in) :: X1A(1:nua,1:noa),&
!                                                  t3a(1:nua,1:nua,1:nua,1:noa,1:noa,1:noa),&
!                                                  t3b(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob),&
!                                                  t3c(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob),&
!                                                  vA_oovv(1:noa,1:noa,1:nua,1:nua),&
!                                                  vB_oovv(1:noa,1:nob,1:nua,1:nub),&
!                                                  vC_oovv(1:nob,1:nob,1:nub,1:nub),&
!                                                  fA_oo(1:noa,1:noa),&
!                                                  fA_vv(1:nua,1:nua),&
!                                                  shift
!
!                      real(kind=8), intent(inout) :: t1a(1:nua,1:noa)
!                      !f2py intent(in,out) :: t1a(0:nua-1,0:noa-1)
!
!                      real(kind=8), intent(out) :: resid(1:nua,1:noa)
!
!                      integer :: i, a, m, n, e, f
!                      real(kind=8) :: denom, val, res1, res2, res3
!
!                      do i = 1, noa
!                        do a = 1, nua
!
!                            res1 = 0.0d0
!                            res2 = 0.0d0
!                            res3 = 0.0d0
!
!                            ! diagram 1: 0.25 * vA_oovv(mnef) * t3a(aefimn)
!                            do e = 1, nua; do f = e + 1, nua; do m = 1, noa; do n = m + 1, noa;
!                                if (pspace_aaa(a, e, f, i, m, n) /= 1) cycle
!                                res1 = res1 + vA_oovv(m, n, e, f) * t3a(a, e, f, i, m, n)
!                            end do; end do; end do; end do;
!                            ! diagram 2: vB_oovv(mnef) * t3b(aefimn)
!                            do e = 1, nua; do f = 1, nub; do m = 1, noa; do n = 1, nob;
!                                if (pspace_aab(a, e, f, i, m, n) /= 1) cycle
!                                res2 = res2 + vB_oovv(m, n, e, f) * t3b(a, e, f, i, m, n)
!                            end do; end do; end do; end do;
!                            ! diagram 3: 0.25 * vC_oovv(mnef) * t3c(aefimn)
!                            do e = 1, nub; do f = e + 1, nub; do m = 1, nob; do n = m + 1, nob;
!                                if (pspace_abb(a, e, f, i, m, n) /= 1) cycle
!                                res3 = res3 + vC_oovv(m, n, e, f) * t3c(a, e, f, i, m, n)
!                            end do; end do; end do; end do;
!
!                          denom = fA_oo(i, i) - fA_vv(a, a)
!                          val = X1A(a, i) + res1 + res2 + res3
!                          val = val/(denom - shift)
!
!                          t1a(a, i) = t1a(a, i) + val
!
!                          resid(a, i) = val
!
!                        end do
!                      end do
!
!              end subroutine update_t1a_opt1
!
!              subroutine update_t1b_opt1(t1b, resid, X1B,&
!                                           t3b, t3c, t3d,&
!                                           pspace_aab, pspace_abb, pspace_bbb,&
!                                           vA_oovv, vB_oovv, vC_oovv,&
!                                           fB_oo, fB_vv,&
!                                           shift,&
!                                           noa, nua, nob, nub)
!
!                      integer, intent(in) :: noa, nua, nob, nub
!                      integer, intent(in) :: pspace_aab(nua, nua, nub, noa, noa, nob),&
!                                             pspace_abb(nua, nub, nub, noa, nob, nob),&
!                                             pspace_bbb(nub, nub, nub, nob, nob, nob)
!                      real(kind=8), intent(in) :: X1B(1:nub,1:nob),&
!                                                  t3b(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob),&
!                                                  t3c(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob),&
!                                                  t3d(1:nub,1:nub,1:nub,1:nob,1:nob,1:nob),&
!                                                  vA_oovv(1:noa,1:noa,1:nua,1:nua),&
!                                                  vB_oovv(1:noa,1:nob,1:nua,1:nub),&
!                                                  vC_oovv(1:nob,1:nob,1:nub,1:nub),&
!                                                  fB_oo(1:nob,1:nob),&
!                                                  fB_vv(1:nub,1:nub),&
!                                                  shift
!
!                      real(kind=8), intent(inout) :: t1b(1:nub,1:nob)
!                      !f2py intent(in,out) :: t1b(0:nub-1,0:nob-1)
!
!                      real(kind=8), intent(out) :: resid(1:nub,1:nob)
!
!                      integer :: i, a, m, n, e, f
!                      real(kind=8) :: denom, val, res1, res2, res3
!
!                      do i = 1, nob
!                        do a = 1, nub
!
!                            res1 = 0.0d0
!                            res2 = 0.0d0
!                            res3 = 0.0d0
!
!                            ! diagram 1: 0.25 * vC_oovv(mnef) * t3d(aefimn)
!                            do e = 1, nub; do f = e + 1, nub; do m = 1, nob; do n = m + 1, nob;
!                                if (pspace_bbb(a, e, f, i, m, n) /= 1) cycle
!                                res1 = res1 + vC_oovv(m, n, e, f) * t3d(a, e, f, i, m, n)
!                            end do; end do; end do; end do;
!                            ! diagram 2: vB_oovv(mnef) * t3c(efamni)
!                            do e = 1, nua; do f = 1, nub; do m = 1, noa; do n = 1, nob;
!                                if (pspace_abb(e, f, a, m, n, i) /= 1) cycle
!                                res2 = res2 + vB_oovv(m, n, e, f) * t3c(e, f, a, m, n, i)
!                            end do; end do; end do; end do;
!                            ! diagram 3: 0.25 * vA_oovv(mnef) * t3b(efamni)
!                            do e = 1, nua; do f = e + 1, nua; do m = 1, noa; do n = m + 1, noa;
!                                if (pspace_aab(e, f, a, m, n, i) /= 1) cycle
!                                res3 = res3 + vA_oovv(m, n, e, f) * t3b(e, f, a, m, n, i)
!                            end do; end do; end do; end do;
!
!                          denom = fB_oo(i, i) - fB_vv(a, a)
!                          val = X1B(a, i) + res1 + res2 + res3
!                          val = val/(denom - shift)
!
!                          t1b(a, i) = t1b(a, i) + val
!
!                          resid(a, i) = val
!
!                        end do
!                      end do
!
!              end subroutine update_t1b_opt1
!    dT.aa += 0.25 * np.einsum("me,abeijm->abij", H.a.ov, T.aaa, optimize=True)
!    dT.aa += 0.25 * np.einsum("me,abeijm->abij", H.b.ov, T.aab, optimize=True)
!    dT.aa -= 0.5 * np.einsum("mnif,abfmjn->abij", H0.ab.ooov + H.ab.ooov, T.aab, optimize=True)
!    dT.aa -= 0.25 * np.einsum("mnif,abfmjn->abij", H0.aa.ooov + H.aa.ooov, T.aaa, optimize=True)
!    dT.aa += 0.25 * np.einsum("anef,ebfijn->abij", H0.aa.vovv + H.aa.vovv, T.aaa, optimize=True)
!    dT.aa += 0.5 * np.einsum("anef,ebfijn->abij", H0.ab.vovv + H.ab.vovv, T.aab, optimize=True)
!              subroutine update_t2a_opt1(t2a, resid, X2A,&
!                                           t3a, t3b,&
!                                           pspace_aaa, pspace_aab,&
!                                           H1A_ov, H1B_ov,&
!                                           H2A_ooov, H2A_vovv,&
!                                           H2B_ooov, H2B_vovv,&
!                                           fA_oo, fA_vv,&
!                                           shift,&
!                                           noa, nua, nob, nub)
!
!                      integer, intent(in) :: noa, nua, nob, nub
!                      real(8), intent(in) :: fA_oo(1:noa,1:noa), fA_vv(1:nua,1:nua), &
!                                             X2A(1:nua,1:nua,1:noa,1:noa),&
!                                             shift
!                      real(8), intent(inout) :: t2a(1:nua,1:nua,1:noa,1:noa)
!                      !f2py intent(in,out) :: t2a(0:nua-1,0:nua-1,0:noa-1,0:noa-1)
!                      real(kind=8), intent(out) :: resid(1:nua,1:nua,1:noa,1:noa)
!                      integer :: i, j, a, b
!                      real(8) :: denom, val
!
!                      do i = 1,noa
!                        do j = i+1,noa
!                          do a = 1,nua
!                            do b = a+1,nua
!                              denom = fA_oo(i,i) + fA_oo(j,j) - fA_vv(a,a) - fA_vv(b,b)
!                              val = (X2A(b,a,j,i) - X2A(a,b,j,i) - X2A(b,a,i,j) + X2A(a,b,i,j))/(denom-shift)
!                              t2a(b,a,j,i) = t2a(b,a,j,i) + val
!                              t2a(a,b,j,i) = -t2a(b,a,j,i)
!                              t2a(b,a,i,j) = -t2a(b,a,j,i)
!                              t2a(a,b,i,j) = t2a(b,a,j,i)
!
!                              resid(b,a,j,i) = val
!                              resid(a,b,j,i) = -val
!                              resid(b,a,i,j) = -val
!                              resid(a,b,i,j) = val
!                            end do
!                          end do
!                        end do
!                      end do
!
!              end subroutine update_t2a_opt1
!
!              subroutine update_t2b_opt1(t2b,resid,X2B,fA_oo,fA_vv,fB_oo,fB_vv,shift,noa,nua,nob,nub)
!
!                      implicit none
!
!                      integer, intent(in) :: noa, nua, nob, nub
!                      real(8), intent(in) :: fA_oo(1:noa,1:noa), fA_vv(1:nua,1:nua), &
!                                          fB_oo(1:nob,1:nob), fB_vv(1:nub,1:nub), &
!                                          X2B(1:nua,1:nub,1:noa,1:nob), shift
!                      real(8), intent(inout) :: t2b(1:nua,1:nub,1:noa,1:nob)
!                      !f2py intent(in,out) :: t2b(0:nua-1,0:nub-1,0:noa-1,0:nob-1)
!                      real(kind=8), intent(out) :: resid(1:nua,1:nub,1:noa,1:nob)
!                      integer :: i, j, a, b
!                      real(8) :: denom, val
!
!                      do j = 1,nob
!                        do i = 1,noa
!                          do b = 1,nub
!                            do a = 1,nua
!                              denom = fA_oo(i,i) + fB_oo(j,j) - fA_vv(a,a) - fB_vv(b,b)
!                              val = X2B(a,b,i,j)/(denom-shift)
!                              t2b(a,b,i,j) = t2b(a,b,i,j) + val
!                              resid(a,b,i,j) = val
!                            end do
!                          end do
!                        end do
!                      end do
!
!              end subroutine update_t2b_opt1
!
!              subroutine update_t2c_opt1(t2c,resid,X2C,fB_oo,fB_vv,shift,nob,nub)
!
!                      implicit none
!
!                      integer, intent(in) :: nob, nub
!                      real(8), intent(in) :: fB_oo(1:nob,1:nob), fB_vv(1:nub,1:nub), &
!                                          X2C(1:nub,1:nub,1:nob,1:nob), shift
!                      real(8), intent(inout) :: t2c(1:nub,1:nub,1:nob,1:nob)
!                      !f2py intent(in,out) :: t2c(0:nub-1,0:nub-1,0:nob-1,0:nob-1)
!                      real(kind=8), intent(out) :: resid(1:nub,1:nub,1:nob,1:nob)
!                      integer :: i, j, a, b
!                      real(8) :: denom, val
!
!                      do i = 1,nob
!                        do j = i+1,nob
!                          do a = 1,nub
!                            do b = a+1,nub
!                              denom = fB_oo(i,i) + fB_oo(j,j) - fB_vv(a,a) - fB_vv(b,b)
!                              !t2c(b,a,j,i) = t2c(b,a,j,i) + X2C(b,a,j,i)/(denom-shift)
!                              val = (X2C(b,a,j,i) - X2C(a,b,j,i) - X2C(b,a,i,j) + X2C(a,b,i,j))/(denom-shift)
!                              t2c(b,a,j,i) = t2c(b,a,j,i) + val
!                              t2c(a,b,j,i) = -t2c(b,a,j,i)
!                              t2c(b,a,i,j) = -t2c(b,a,j,i)
!                              t2c(a,b,i,j) = t2c(b,a,j,i)
!
!                              resid(b,a,j,i) = val
!                              resid(a,b,j,i) = -val
!                              resid(b,a,i,j) = -val
!                              resid(a,b,i,j) = val
!                            end do
!                          end do
!                        end do
!                      end do
!
!              end subroutine update_t2c_opt1

              subroutine update_t3a_p_opt1(t3a_new, resid,&
                                           X3A,&
                                           t2a, t3a, t3b,&
                                           pspace,&
                                           H1A_vv,&
                                           H2A_oovv, H2A_vvov, H2A_voov, H2A_vvvv,&
                                           H2B_oovv, H2B_voov,&
                                           fA_oo, fA_vv,&
                                           shift,&
                                           noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  real(kind=8), intent(in) :: t2a(nua, nua, noa, noa),&
                                              t3a(nua, nua, nua, noa, noa ,noa),&
                                              t3b(nua, nua, nub, noa, noa, nob),&
                                              H1A_vv(nua, nua),&
                                              H2A_oovv(noa, noa, nua, nua),&
                                              H2A_vvov(nua, nua, noa, nua),&
                                              H2A_voov(nua, noa, noa, nua),&
                                              H2A_vvvv(nua, nua, nua, nua),&
                                              H2B_oovv(noa, nob, nua, nub),&
                                              H2B_voov(nua, nob, noa, nub),&
                                              fA_vv(nua, nua), fA_oo(noa, noa),&
                                              shift
                  integer, intent(in) :: pspace(nua, nua, nua, noa, noa, noa)

                  real(kind=8), intent(in) :: X3A(nua, nua, nua, noa, noa, noa)

                  real(kind=8), intent(out) :: t3a_new(nua, nua, nua, noa, noa, noa),&
                                               resid(nua, nua, nua, noa, noa, noa)

                  real(kind=8) :: I2A_vvov(nua, nua, noa, nua), val, denom
                  real(kind=8) :: res1, res2, res3, res4, res5
                  integer :: idx, a, b, c, i, j, k, e, f, m, n

                  t3a_new = 0.0d0
                  resid = 0.0d0

                  I2A_vvov = H2A_vvov

                  ! loop over projection determinants in P space
                  do a = 1, nua; do b = a + 1, nua; do c = b + 1, nua;
                  do i = 1, noa; do j = i + 1, noa; do k = j + 1, noa;

                      if (pspace(a, b, c, i, j, k) /= 1) cycle

                      res1 = 0.0d0
                      res2 = 0.0d0
                      res3 = 0.0d0
                      res4 = 0.0d0
                      res5 = 0.0d0

                      do e = 1, nua

                          ! A(a/bc) h1a(ae) * t3a(ebcijk)
                          res1 = res1 + H1A_vv(a, e) * t3a(e, b, c, i, j, k)
                          res1 = res1 - H1A_vv(b, e) * t3a(e, a, c, i, j, k)
                          res1 = res1 - H1A_vv(c, e) * t3a(e, b, a, i, j, k)

                          ! 1/2 A(c/ab) h2a(abef) * t3a(efcijk)
                          do f = e + 1, nua
                              res2 = res2 + H2A_vvvv(a, b, e, f) * t3a(e, f, c, i, j, k)
                              res2 = res2 - H2A_vvvv(c, b, e, f) * t3a(e, f, a, i, j, k)
                              res2 = res2 - H2A_vvvv(a, c, e, f) * t3a(e, f, b, i, j, k)
                          end do

                          ! A(i/jk)A(c/ab) h2a(amie) * t3a(ebcmjk)
                          do m = 1, noa
                              res3 = res3 + H2A_voov(a, m, i, e) * t3a(e, b, c, m, j, k)
                              res3 = res3 - H2A_voov(a, m, j, e) * t3a(e, b, c, m, i, k)
                              res3 = res3 - H2A_voov(a, m, k, e) * t3a(e, b, c, m, j, i)
                              res3 = res3 - H2A_voov(b, m, i, e) * t3a(e, a, c, m, j, k)
                              res3 = res3 + H2A_voov(b, m, j, e) * t3a(e, a, c, m, i, k)
                              res3 = res3 + H2A_voov(b, m, k, e) * t3a(e, a, c, m, j, i)
                              res3 = res3 - H2A_voov(c, m, i, e) * t3a(e, b, a, m, j, k)
                              res3 = res3 + H2A_voov(c, m, j, e) * t3a(e, b, a, m, i, k)
                              res3 = res3 + H2A_voov(c, m, k, e) * t3a(e, b, a, m, j, i)

                          end do

                          do m = 1, nob
                              ! A(i/jk)A(a/bc) h2b(amie) * t3b(bcejkm)
                              res5 = res5 + H2B_voov(a, m, i, e) * t3b(b, c, e, j, k, m)
                              res5 = res5 - H2B_voov(b, m, i, e) * t3b(a, c, e, j, k, m)
                              res5 = res5 - H2B_voov(c, m, i, e) * t3b(b, a, e, j, k, m)
                              res5 = res5 - H2B_voov(a, m, j, e) * t3b(b, c, e, i, k, m)
                              res5 = res5 + H2B_voov(b, m, j, e) * t3b(a, c, e, i, k, m)
                              res5 = res5 + H2B_voov(c, m, j, e) * t3b(b, a, e, i, k, m)
                              res5 = res5 - H2B_voov(a, m, k, e) * t3b(b, c, e, j, i, m)
                              res5 = res5 + H2B_voov(b, m, k, e) * t3b(a, c, e, j, i, m)
                              res5 = res5 + H2B_voov(c, m, k, e) * t3b(b, a, e, j, i, m)
                          end do



                          ! A(i/jk)(c/ab) h2a(abie) * t2a(ecjk)
                          res4 = res4 + I2A_vvov(a, b, i, e) * t2a(e, c, j, k)
                          res4 = res4 - I2A_vvov(c, b, i, e) * t2a(e, a, j, k)
                          res4 = res4 - I2A_vvov(a, c, i, e) * t2a(e, b, j, k)
                          res4 = res4 - I2A_vvov(a, b, j, e) * t2a(e, c, i, k)
                          res4 = res4 + I2A_vvov(c, b, j, e) * t2a(e, a, i, k)
                          res4 = res4 + I2A_vvov(a, c, j, e) * t2a(e, b, i, k)
                          res4 = res4 - I2A_vvov(a, b, k, e) * t2a(e, c, j, i)
                          res4 = res4 + I2A_vvov(c, b, k, e) * t2a(e, a, j, i)
                          res4 = res4 + I2A_vvov(a, c, k, e) * t2a(e, b, j, i)
!                          res4 = res4 + (H2A_vvov(a, b, i, e) + vt3a(a, b, i, e)) * t2a(e, c, j, k)
!                          res4 = res4 - (H2A_vvov(c, b, i, e) + vt3a(c, b, i, e)) * t2a(e, a, j, k)
!                          res4 = res4 - (H2A_vvov(a, c, i, e) + vt3a(a, c, i, e)) * t2a(e, b, j, k)
!                          res4 = res4 - (H2A_vvov(a, b, j, e) + vt3a(a, b, j, e)) * t2a(e, c, i, k)
!                          res4 = res4 + (H2A_vvov(c, b, j, e) + vt3a(c, b, j, e)) * t2a(e, a, i, k)
!                          res4 = res4 + (H2A_vvov(a, c, j, e) + vt3a(a, c, j, e)) * t2a(e, b, i, k)
!                          res4 = res4 - (H2A_vvov(a, b, k, e) + vt3a(a, b, k, e)) * t2a(e, c, j, i)
!                          res4 = res4 + (H2A_vvov(c, b, k, e) + vt3a(c, b, k, e)) * t2a(e, a, j, i)
!                          res4 = res4 + (H2A_vvov(a, c, k, e) + vt3a(a, c, k, e)) * t2a(e, b, j, i)


                      end do

                      do e = 1, nub - nua
                          do m = 1, nob
                              ! A(i/jk)A(a/bc) h2b(amie) * t3b(bcejkm)
                              res5 = res5 + H2B_voov(a, m, i, e + nua) * t3b(b, c, e + nua, j, k, m)
                              res5 = res5 - H2B_voov(b, m, i, e + nua) * t3b(a, c, e + nua, j, k, m)
                              res5 = res5 - H2B_voov(c, m, i, e + nua) * t3b(b, a, e + nua, j, k, m)
                              res5 = res5 - H2B_voov(a, m, j, e + nua) * t3b(b, c, e + nua, i, k, m)
                              res5 = res5 + H2B_voov(b, m, j, e + nua) * t3b(a, c, e + nua, i, k, m)
                              res5 = res5 + H2B_voov(c, m, j, e + nua) * t3b(b, a, e + nua, i, k, m)
                              res5 = res5 - H2B_voov(a, m, k, e + nua) * t3b(b, c, e + nua, j, i, m)
                              res5 = res5 + H2B_voov(b, m, k, e + nua) * t3b(a, c, e + nua, j, i, m)
                              res5 = res5 + H2B_voov(c, m, k, e + nua) * t3b(b, a, e + nua, j, i, m)
                          end do
                      end do

                      denom = fA_oo(I, I) + fA_oo(J, J) + fA_oo(K, K) - fA_vv(A, A) - fA_vv(B, B) - fA_vv(C, C)

                      val = X3A(a,b,c,i,j,k)&
                              -X3A(b,a,c,i,j,k)&
                              -X3A(a,c,b,i,j,k)&
                              +X3A(b,c,a,i,j,k)&
                              -X3A(c,b,a,i,j,k)&
                              +X3A(c,a,b,i,j,k)&
                              -X3A(a,b,c,j,i,k)&
                              +X3A(b,a,c,j,i,k)&
                              +X3A(a,c,b,j,i,k)&
                              -X3A(b,c,a,j,i,k)&
                              +X3A(c,b,a,j,i,k)&
                              -X3A(c,a,b,j,i,k)&
                              -X3A(a,b,c,i,k,j)&
                              +X3A(b,a,c,i,k,j)&
                              +X3A(a,c,b,i,k,j)&
                              -X3A(b,c,a,i,k,j)&
                              +X3A(c,b,a,i,k,j)&
                              -X3A(c,a,b,i,k,j)&
                              -X3A(a,b,c,k,j,i)&
                              +X3A(b,a,c,k,j,i)&
                              +X3A(a,c,b,k,j,i)&
                              -X3A(b,c,a,k,j,i)&
                              +X3A(c,b,a,k,j,i)&
                              -X3A(c,a,b,k,j,i)&
                              +X3A(a,b,c,j,k,i)&
                              -X3A(b,a,c,j,k,i)&
                              -X3A(a,c,b,j,k,i)&
                              +X3A(b,c,a,j,k,i)&
                              -X3A(c,b,a,j,k,i)&
                              +X3A(c,a,b,j,k,i)&
                              +X3A(a,b,c,k,i,j)&
                              -X3A(b,a,c,k,i,j)&
                              -X3A(a,c,b,k,i,j)&
                              +X3A(b,c,a,k,i,j)&
                              -X3A(c,b,a,k,i,j)&
                              +X3A(c,a,b,k,i,j)

                      val = val + res1 + res2 + res3 + res4 + res5

                      val = val/(denom - shift)

                      ! update
                      t3a_new(A,B,C,I,J,K) = t3a(A,B,C,I,J,K) + val

                      t3a_new(A,B,C,K,I,J) = t3a_new(A,B,C,I,J,K)
                      t3a_new(A,B,C,J,K,I) = t3a_new(A,B,C,I,J,K)
                      t3a_new(A,B,C,I,K,J) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(A,B,C,J,I,K) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(A,B,C,K,J,I) = -t3a_new(A,B,C,I,J,K)

                      t3a_new(B,A,C,I,J,K) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(B,A,C,K,I,J) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(B,A,C,J,K,I) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(B,A,C,I,K,J) = t3a_new(A,B,C,I,J,K)
                      t3a_new(B,A,C,J,I,K) = t3a_new(A,B,C,I,J,K)
                      t3a_new(B,A,C,K,J,I) = t3a_new(A,B,C,I,J,K)

                      t3a_new(A,C,B,I,J,K) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(A,C,B,K,I,J) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(A,C,B,J,K,I) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(A,C,B,I,K,J) = t3a_new(A,B,C,I,J,K)
                      t3a_new(A,C,B,J,I,K) = t3a_new(A,B,C,I,J,K)
                      t3a_new(A,C,B,K,J,I) = t3a_new(A,B,C,I,J,K)

                      t3a_new(C,B,A,I,J,K) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(C,B,A,K,I,J) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(C,B,A,J,K,I) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(C,B,A,I,K,J) = t3a_new(A,B,C,I,J,K)
                      t3a_new(C,B,A,J,I,K) = t3a_new(A,B,C,I,J,K)
                      t3a_new(C,B,A,K,J,I) = t3a_new(A,B,C,I,J,K)

                      t3a_new(B,C,A,I,J,K) = t3a_new(A,B,C,I,J,K)
                      t3a_new(B,C,A,K,I,J) = t3a_new(A,B,C,I,J,K)
                      t3a_new(B,C,A,J,K,I) = t3a_new(A,B,C,I,J,K)
                      t3a_new(B,C,A,I,K,J) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(B,C,A,J,I,K) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(B,C,A,K,J,I) = -t3a_new(A,B,C,I,J,K)

                      t3a_new(C,A,B,I,J,K) = t3a_new(A,B,C,I,J,K)
                      t3a_new(C,A,B,K,I,J) = t3a_new(A,B,C,I,J,K)
                      t3a_new(C,A,B,J,K,I) = t3a_new(A,B,C,I,J,K)
                      t3a_new(C,A,B,I,K,J) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(C,A,B,J,I,K) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(C,A,B,K,J,I) = -t3a_new(A,B,C,I,J,K)

                      resid(A,B,C,I,J,K) = val
                      resid(A,B,C,K,I,J) = val
                      resid(A,B,C,J,K,I) = val
                      resid(A,B,C,I,K,J) = -val
                      resid(A,B,C,J,I,K) = -val
                      resid(A,B,C,K,J,I) = -val
                      resid(B,C,A,I,J,K) = val
                      resid(B,C,A,K,I,J) = val
                      resid(B,C,A,J,K,I) = val
                      resid(B,C,A,I,K,J) = -val
                      resid(B,C,A,J,I,K) = -val
                      resid(B,C,A,K,J,I) = -val
                      resid(C,A,B,I,J,K) = val
                      resid(C,A,B,K,I,J) = val
                      resid(C,A,B,J,K,I) = val
                      resid(C,A,B,I,K,J) = -val
                      resid(C,A,B,J,I,K) = -val
                      resid(C,A,B,K,J,I) = -val
                      resid(A,C,B,I,J,K) = -val
                      resid(A,C,B,K,I,J) = -val
                      resid(A,C,B,J,K,I) = -val
                      resid(A,C,B,I,K,J) = val
                      resid(A,C,B,J,I,K) = val
                      resid(A,C,B,K,J,I) = val
                      resid(B,A,C,I,J,K) = -val
                      resid(B,A,C,K,I,J) = -val
                      resid(B,A,C,J,K,I) = -val
                      resid(B,A,C,I,K,J) = val
                      resid(B,A,C,J,I,K) = val
                      resid(B,A,C,K,J,I) = val
                      resid(C,B,A,I,J,K) = -val
                      resid(C,B,A,K,I,J) = -val
                      resid(C,B,A,J,K,I) = -val
                      resid(C,B,A,I,K,J) = val
                      resid(C,B,A,J,I,K) = val
                      resid(C,B,A,K,J,I) = val

                  end do; end do; end do;
                  end do; end do; end do;

              end subroutine update_t3a_p_opt1

              subroutine update_t3b_p_opt1(t3b_new, resid,&
                                           X3B,&
                                           t2a, t2b, t3a, t3b, t3c,&
                                           pspace,&
                                           H1A_vv, H1B_vv,&
                                           H2A_oovv, H2A_vvov, H2A_voov, H2A_vvvv,&
                                           H2B_oovv, H2B_vvov, H2B_vvvo, H2B_voov, H2B_vovo, H2B_ovov, H2B_ovvo, H2B_vvvv,&
                                           H2C_voov,&
                                           fA_oo, fA_vv, fB_oo, fB_vv,&
                                           shift,&
                                           noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: pspace(nua, nua, nub, noa, noa, nob)
                  real(8), intent(in) :: t2a(1:nua,1:nua,1:noa,1:noa),&
                                         t2b(1:nua,1:nub,1:noa,1:nob),&
                                         t3a(1:nua,1:nua,1:nua,1:noa,1:noa,1:noa),&
                                         t3b(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob),&
                                         t3c(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob),&
                                         H1A_vv(1:nua,1:nua),&
                                         H1B_vv(1:nub,1:nub),&
                                         H2A_oovv(1:noa,1:noa,1:nua,1:nua),&
                                         H2A_vvov(1:nua,1:nua,1:noa,1:nua),&
                                         H2A_voov(1:nua,1:noa,1:noa,1:nua),&
                                         H2A_vvvv(1:nua,1:nua,1:nua,1:nua),&
                                         H2B_oovv(1:noa,1:nob,1:nua,1:nub),&
                                         H2B_vvov(1:nua,1:nub,1:noa,1:nub),&
                                         H2B_vvvo(1:nua,1:nub,1:nua,1:nob),&
                                         H2B_voov(1:nua,1:nob,1:noa,1:nub),&
                                         H2B_vovo(1:nua,1:nob,1:nua,1:nob),&
                                         H2B_ovov(1:noa,1:nub,1:noa,1:nub),&
                                         H2B_ovvo(1:noa,1:nub,1:nua,1:nob),&
                                         H2B_vvvv(1:nub,1:nub,1:nub,1:nub),&
                                         H2C_voov(1:nub,1:nob,1:nob,1:nub),&
                                         fA_oo(1:noa,1:noa), fA_vv(1:nua,1:nua),&
                                         fB_oo(1:nob,1:nob), fB_vv(1:nub,1:nub),&
                                         X3B(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob),&
                                         shift

                  real(8), intent(out) :: t3b_new(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob),&
                                          resid(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob)
                  integer :: i, j, k, a, b, c, m, n, e, f
                  real(8) :: denom, val,&
                             res1, res2, res3, res4, res5, res6, res7, res8, res9, res10, res11, res12, res13

                  resid = 0.0d0
                  t3b_new = 0.0d0

                  do i = 1, noa; do j = i + 1, noa; do k = 1, nob;
                  do a = 1, nua; do b = a + 1, nua; do c = 1, nub;

                      if (pspace(a, b, c, i, j, k) /= 1) cycle

                      res1 = 0.0d0
                      res2 = 0.0d0
                      res3 = 0.0d0
                      res4 = 0.0d0
                      res5 = 0.0d0
                      res6 = 0.0d0
                      res7 = 0.0d0
                      res8 = 0.0d0
                      res9 = 0.0d0
                      res10 = 0.0d0
                      res11 = 0.0d0
                      res12 = 0.0d0
                      res13 = 0.0d0

                      ! nua < nub
                      do e = 1, nua

                          ! diagram 1: A(ab) h1A(ae) * t3b(ebcijk)
                          res1 = res1 + H1A_vv(a, e) * t3b(e, b, c, i, j, k)
                          res1 = res1 - H1A_vv(b, e) * t3b(e, a, c, i, j, k)

                          ! diagram 2: h1B(ce) * t3b(abeijk)
                          res2 = res2 + H1B_vv(c, e) * t3b(a, b, e, i, j, k)

                          ! diagram 3: 0.5 * h2A(abef) * t3b(efcijk)
                          ! diagram 4: A(ab) h2B(bcef) * t3b(aefijk)
                          do f = 1, nua
                              res3 = res3 + 0.5d0 * H2A_vvvv(a, b, e, f) * t3b(e, f, c, i, j, k)
                              res4 = res4 + H2B_vvvv(b, c, e, f) * t3b(a, e, f, i, j, k)
                              res4 = res4 - H2B_vvvv(a, c, e, f) * t3b(b, e, f, i, j, k)
                          end do
                          do f = 1, nub - nua
                              res4 = res4 + H2B_vvvv(b, c, e, f + nua) * t3b(a, e, f + nua, i, j, k)
                              res4 = res4 - H2B_vvvv(a, c, e, f + nua) * t3b(b, e, f + nua, i, j, k)
                          end do

                          ! diagram 5: A(ij)A(ab) h2A(amie) * t3b(ebcmjk)
                          ! diagram 7: h2B(mcek) * t3a(abeijm)
                          ! diagram 10: -A(ij) h2B(mcie) * t3b(abemjk)
                          do m = 1, noa
                              res5 = res5 + H2A_voov(a, m, i, e) * t3b(e, b, c, m, j, k)
                              res5 = res5 - H2A_voov(b, m, i, e) * t3b(e, a, c, m, j, k)
                              res5 = res5 - H2A_voov(a, m, j, e) * t3b(e, b, c, m, i, k)
                              res5 = res5 + H2A_voov(b, m, j, e) * t3b(e, a, c, m, i, k)

                              res7 = res7 + H2B_ovvo(m, c, e, k) * t3a(a, b, e, i, j, m)

                              res10 = res10 - H2B_ovov(m, c, i, e) * t3b(a, b, e, m, j, k)
                              res10 = res10 + H2B_ovov(m, c, j, e) * t3b(a, b, e, m, i, k)
                          end do

                          ! diagram 6: A(ij)A(ab) h2B(amie) * t3c(becjmk)
                          ! diagram 8: h2C(cmke) * t3b(abeijm)
                          ! diagram 9: -A(ab) h2B(amek) * t3b(ebcijm)
                          do m = 1, nob
                              res6 = res6 + H2B_voov(a, m, i, e) * t3c(b, e, c, j, m, k)
                              res6 = res6 - H2B_voov(b, m, i, e) * t3c(a, e, c, j, m, k)
                              res6 = res6 - H2B_voov(a, m, j, e) * t3c(b, e, c, i, m, k)
                              res6 = res6 + H2B_voov(b, m, j, e) * t3c(a, e, c, i, m, k)

                              res8 = res8 + H2C_voov(c, m, k, e) * t3b(a, b, e, i, j, m)

                              res9 = res9 - H2B_vovo(a, m, e, k) * t3b(e, b, c, i, j, m)
                              res9 = res9 + H2B_vovo(b, m, e, k) * t3b(e, a, c, i, j, m)
                          end do

                          ! diagram 11: A(ab) I2B(bcek) * t2a(aeij)
                          res11 = res11 + H2B_vvvo(b, c, e, k) * t2a(a, e, i, j)
                          res11 = res11 - H2B_vvvo(a, c, e, k) * t2a(b, e, i, j)
                          ! diagram 12: A(ij)A(ab) I2B(acie) * t2b(bejk)
                          res12 = res12 + H2B_vvov(a, c, i, e) * t2b(b, e, j, k)
                          res12 = res12 - H2B_vvov(a, c, j, e) * t2b(b, e, i, k)
                          res12 = res12 - H2B_vvov(b, c, i, e) * t2b(a, e, j, k)
                          res12 = res12 + H2B_vvov(b, c, j, e) * t2b(a, e, i, k)
                          ! diagram 13: A(ij) I2A(abie) * t2b(ecjk)
                          res13 = res13 + H2A_vvov(a, b, i, e) * t2b(e, c, j, k)
                          res13 = res13 - H2A_vvov(a, b, j, e) * t2b(e, c, i, k)

                      end do

                      do e = 1, nub - nua

                          ! diagram 2 : h1B(ce) * t3b(abeijk)
                          res2 = res2 + H1B_vv(c, e + nua) * t3b(a, b, e + nua, i, j, k)

                          ! diagram 10: -A(ij) h2B(mcie) * t3b(abemjk)
                          do m = 1, noa
                              res10 = res10 - H2B_ovov(m, c, i, e + nua) * t3b(a, b, e + nua, m, j, k)
                              res10 = res10 + H2B_ovov(m, c, j, e + nua) * t3b(a, b, e + nua, m, i, k)
                          end do

                          ! diagram 6: A(ij)A(ab) h2B(amie) * t3c(becjmk)
                          ! diagram 8: h2C(cmke) * t3b(abeijm)
                          do m = 1, nob
                              res6 = res6 + H2B_voov(a, m, i, e + nua) * t3c(b, e + nua, c, j, m, k)
                              res6 = res6 - H2B_voov(b, m, i, e + nua) * t3c(a, e + nua, c, j, m, k)
                              res6 = res6 - H2B_voov(a, m, j, e + nua) * t3c(b, e + nua, c, i, m, k)
                              res6 = res6 + H2B_voov(b, m, j, e + nua) * t3c(a, e + nua, c, i, m, k)

                              res8 = res8 + H2C_voov(c, m, k, e + nua) * t3b(a, b, e + nua, i, j, m)
                          end do

                          ! diagram 12: A(ij)A(ab) I2B(acie) * t2b(bejk)
                          res12 = res12 + H2B_vvov(a, c, i, e + nua) * t2b(b, e + nua, j, k)
                          res12 = res12 - H2B_vvov(a, c, j, e + nua) * t2b(b, e + nua, i, k)
                          res12 = res12 - H2B_vvov(b, c, i, e + nua) * t2b(a, e + nua, j, k)
                          res12 = res12 + H2B_vvov(b, c, j, e + nua) * t2b(a, e + nua, i, k)

                      end do


                      denom = fA_oo(i, i) + fA_oo(j, j) + fB_oo(k, k) - fA_vv(a, a) - fA_vv(b, b) - fB_vv(c, c)

                      val = X3B(a, b, c, i, j, k) - X3B(b, a, c, i, j, k) - X3B(a, b, c, j, i, k) + X3B(b, a, c, j, i, k)

                       val = val +&
                          res1 + res2 + res3 + res4 + res5 + res6 + res7 + res8 + res9 + res10 + res11 + res12 + res13

                      val = val/(denom - shift)

                      t3b_new(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                      t3b_new(b, a, c, i, j, k) = -t3b_new(a, b, c, i, j, k)
                      t3b_new(a, b, c, j, i, k) = -t3b_new(a, b, c, i, j, k)
                      t3b_new(b, a, c, j, i, k) = t3b_new(a, b, c, i, j, k)

                      resid(a, b, c, i, j, k) = val
                      resid(b, a, c, i, j, k) = -val
                      resid(a, b, c, j, i, k) = -val
                      resid(b, a, c, j, i, k) = val

                  end do; end do; end do;
                  end do; end do; end do;

              end subroutine update_t3b_p_opt1

              subroutine update_t3c_p_opt1(t3c_new, resid,&
                                           X3C,&
                                           t2b, t2c, t3b, t3c, t3d,&
                                           pspace,&
                                           H1A_vv, H1B_vv,&
                                           H2A_voov,&
                                           H2B_oovv, H2B_vvov, H2B_vvvo, H2B_voov, H2B_vovo, H2B_ovov, H2B_ovvo, H2B_vvvv,&
                                           H2C_oovv, H2C_vvov, H2C_voov, H2C_vvvv,&
                                           fA_oo, fA_vv, fB_oo, fB_vv,&
                                           shift,&
                                           noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: pspace(nua, nub, nub, noa, nob, nob)
                  real(8), intent(in) :: t2b(1:nua,1:nub,1:noa,1:nob),&
                                         t2c(1:nub,1:nub,1:nob,1:nob),&
                                         t3b(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob),&
                                         t3c(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob),&
                                         t3d(1:nub,1:nub,1:nub,1:nob,1:nob,1:nob),&
                                         H1A_vv(1:nua,1:nua),&
                                         H1B_vv(1:nub,1:nub),&
                                         H2A_voov(1:nua,1:noa,1:noa,1:nua),&
                                         H2B_oovv(1:noa,1:nob,1:nua,1:nub),&
                                         H2B_vvov(1:nua,1:nub,1:noa,1:nub),&
                                         H2B_vvvo(1:nua,1:nub,1:nua,1:nob),&
                                         H2B_voov(1:nua,1:nob,1:noa,1:nub),&
                                         H2B_vovo(1:nua,1:nob,1:nua,1:nob),&
                                         H2B_ovov(1:noa,1:nub,1:noa,1:nub),&
                                         H2B_ovvo(1:noa,1:nub,1:nua,1:nob),&
                                         H2C_oovv(1:nob,1:nob,1:nub,1:nub),&
                                         H2C_vvov(1:nub,1:nub,1:nob,1:nub),&
                                         H2B_vvvv(1:nub,1:nub,1:nub,1:nub),&
                                         H2C_voov(1:nub,1:nob,1:nob,1:nub),&
                                         H2C_vvvv(1:nub,1:nub,1:nub,1:nub),&
                                         fA_oo(1:noa,1:noa), fA_vv(1:nua,1:nua),&
                                         fB_oo(1:nob,1:nob), fB_vv(1:nub,1:nub),&
                                         X3C(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob),&
                                         shift

                  real(8), intent(out) :: t3c_new(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob),&
                                          resid(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob)
                  integer :: i, j, k, a, b, c, m, n, e, f
                  real(8) :: denom, val,&
                             res1, res2, res3, res4, res5, res6, res7, res8, res9, res10, res11, res12, res13

                  resid = 0.0d0
                  t3c_new = 0.0d0

                  do i = 1, noa; do j = 1, nob; do k = j + 1, nob;
                  do a = 1, nua; do b = 1, nub; do c = b + 1, nub;

                      if (pspace(a, b, c, i, j, k) /= 1) cycle

                      res1 = 0.0d0
                      res2 = 0.0d0
                      res3 = 0.0d0
                      res4 = 0.0d0
                      res5 = 0.0d0
                      res6 = 0.0d0
                      res7 = 0.0d0
                      res8 = 0.0d0
                      res9 = 0.0d0
                      res10 = 0.0d0
                      res11 = 0.0d0
                      res12 = 0.0d0
                      res13 = 0.0d0

                      do e = 1, nua

                          ! diagram 1: h1A(ae) * t3c(ebcijk)
                          res1 = res1 + H1A_vv(a, e) * t3c(e, b, c, i, j, k)

                          ! diagram 2: A(be) h1B(be) * t3c(aecijk)
                          res2 = res2 + H1B_vv(b, e) * t3c(a, e, c, i, j, k)
                          res2 = res2 - H1B_vv(c, e) * t3c(a, e, b, i, j, k)

                          ! diagram 3: 0.5 * h2C(bcef) * t3c(aefijk)
                          ! diagram 4: A(bc) h2C(abef) * t3c(efcijk)
                          do f = 1, nua
                              res3 = res3 + 0.5d0 * H2C_vvvv(b, c, e, f) * t3c(a, e, f, i, j, k)
                              res4 = res4 + H2B_vvvv(a, b, e, f) * t3c(e, f, c, i, j, k)
                              res4 = res4 - H2B_vvvv(a, c, e, f) * t3c(e, f, b, i, j, k)
                          end do
                          do f = 1, nub - nua
                              res4 = res4 + H2B_vvvv(a, b, e, f + nua) * t3c(e, f + nua, c, i, j, k)
                              res4 = res4 - H2B_vvvv(a, c, e, f + nua) * t3c(e, f + nua, b, i, j, k)
                          end do

                          ! diagram 5: h2A(amie) * t3c(ebcmjk)
                          ! diagram 7: A(jk)A(bc) h2B(mbej) * t3b(aecimk)
                          ! diagram 9: -A(bc) h2B(mbie) * t3c(aecmjk)
                          do m = 1, noa
                              res5 = res5 + H2A_voov(a, m, i, e) * t3c(e, b, c, m, j, k)

                              res7 = res7 + H2B_ovvo(m, b, e, j) * t3b(a, e, c, i, m, k)
                              res7 = res7 - H2B_ovvo(m, c, e, j) * t3b(a, e, b, i, m, k)
                              res7 = res7 - H2B_ovvo(m, b, e, k) * t3b(a, e, c, i, m, j)
                              res7 = res7 + H2B_ovvo(m, c, e, k) * t3b(a, e, b, i, m, j)

                              res9 = res9 - H2B_ovov(m, b, i, e) * t3c(a, e, c, m, j, k)
                              res9 = res9 + H2B_ovov(m, c, i, e) * t3c(a, e, b, m, j, k)
                          end do

                          ! diagram 6: h2B(amie) * t3d(ebcmjk)
                          ! diagram 8: A(jk)A(bc) h2C(bmje) * t3c(aecimk)
                          ! diagram 10: -A(jk) h2B(amej) * t3c(ebcimk)
                          do m = 1, nob
                              res6 = res6 + H2B_voov(a, m, i, e) * t3d(e, b, c, m, j, k)

                              res8 = res8 + H2C_voov(b, m, j, e) * t3c(a, e, c, i, m, k)
                              res8 = res8 - H2C_voov(c, m, j, e) * t3c(a, e, b, i, m, k)
                              res8 = res8 - H2C_voov(b, m, k, e) * t3c(a, e, c, i, m, j)
                              res8 = res8 + H2C_voov(c, m, k, e) * t3c(a, e, b, i, m, j)

                              res10 = res10 - H2B_vovo(a, m, e, j) * t3c(e, b, c, i, m, k)
                              res10 = res10 + H2B_vovo(a, m, e, k) * t3c(e, b, c, i, m, j)
                          end do

                          ! diagram 11: A(bc) h2B(abie) * t2c(ecjk)
                          res11 = res11 + H2B_vvov(a, b, i, e) * t2c(e, c, j, k)
                          res11 = res11 - H2B_vvov(a, c, i, e) * t2c(e, b, j, k)
                          ! diagram 12: A(jk) h2C(cbke) * t2b(aeij)
                          res12 = res12 + H2C_vvov(c, b, k, e) * t2b(a, e, i, j)
                          res12 = res12 - H2C_vvov(c, b, j, e) * t2b(a, e, i, k)
                          ! diagram 13: A(jk)A(bc) h2B(abej) * t2b(ecik)
                          res13 = res13 + H2B_vvvo(a, b, e, j) * t2b(e, c, i, k)
                          res13 = res13 - H2B_vvvo(a, b, e, k) * t2b(e, c, i, j)
                          res13 = res13 - H2B_vvvo(a, c, e, j) * t2b(e, b, i, k)
                          res13 = res13 + H2B_vvvo(a, c, e, k) * t2b(e, b, i, j)

                      end do

                      do e = 1, nub - nua

                          ! diagram 2: A(be) h1B(be) * t3c(aecijk)
                          res2 = res2 + H1B_vv(b, e + nua) * t3c(a, e + nua, c, i, j, k)
                          res2 = res2 - H1B_vv(c, e + nua) * t3c(a, e + nua, b, i, j, k)

                          ! diagram 3: 0.5 * h2C(bcef) * t3c(aefijk)
                          do f = 1, nua - nub
                              res3 = res3 + 0.5d0 * H2C_vvvv(b, c, e + nua, f + nua) * t3c(a, e + nua, f + nua, i, j, k)
                          end do

                          ! diagram 9: -A(bc) h2B(mbie) * t3c(aecmjk)
                          do m = 1, noa
                              res9 = res9 - H2B_ovov(m, b, i, e + nua) * t3c(a, e + nua, c, m, j, k)
                              res9 = res9 + H2B_ovov(m, c, i, e + nua) * t3c(a, e + nua, b, m, j, k)
                          end do

                          ! diagram 6: h2B(amie) * t3d(ebcmjk)
                          ! diagram 8: A(jk)A(bc) h2C(bmje) * t3c(aecimk)
                          do m = 1, nob
                              res6 = res6 + H2B_voov(a, m, i, e + nua) * t3d(e + nua, b, c, m, j, k)

                              res8 = res8 + H2C_voov(b, m, j, e + nua) * t3c(a, e + nua, c, i, m, k)
                              res8 = res8 - H2C_voov(c, m, j, e + nua) * t3c(a, e + nua, b, i, m, k)
                              res8 = res8 - H2C_voov(b, m, k, e + nua) * t3c(a, e + nua, c, i, m, j)
                              res8 = res8 + H2C_voov(c, m, k, e + nua) * t3c(a, e + nua, b, i, m, j)
                          end do

                          ! diagram 11: A(bc) h2B(abie) * t2c(ecjk)
                          res11 = res11 + H2B_vvov(a, b, i, e + nua) * t2c(e + nua, c, j, k)
                          res11 = res11 - H2B_vvov(a, c, i, e + nua) * t2c(e + nua, b, j, k)
                          ! diagram 12: A(jk) h2C(cbke) * t2b(aeij)
                          res12 = res12 + H2C_vvov(c, b, k, e + nua) * t2b(a, e + nua, i, j)
                          res12 = res12 - H2C_vvov(c, b, j, e + nua) * t2b(a, e + nua, i, k)

                      end do

                      denom = fA_oo(i, i) + fB_oo(j, j) + fB_oo(k, k) - fA_vv(a, a) - fB_vv(b, b) - fB_vv(c, c)

                      val = X3C(a, b, c, i, j, k) - X3C(a, c, b, i, j, k) - X3C(a, b, c, i, k, j) + X3C(a, c, b, i, k, j)
                      val = val&
                        + res1 + res2 + res3 + res4 + res5 + res6 + res7 + res8 + res9 + res10 + res11 + res12 + res13
                      val = val/(denom - shift)

                      t3c_new(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                      t3c_new(a, c, b, i, j, k) = -t3c_new(a, b, c, i, j, k)
                      t3c_new(a, b, c, i, k, j) = -t3c_new(a, b, c, i, j, k)
                      t3c_new(a, c, b, i, k, j) = t3c_new(a, b, c, i, j, k)

                      resid(a, b, c, i, j, k) = val
                      resid(a, c, b, i, j, k) = -val
                      resid(a, b, c, i, k, j) = -val
                      resid(a, c, b, i, k, j) = val

                  end do; end do; end do;
                  end do; end do; end do;

              end subroutine update_t3c_p_opt1

              subroutine update_t3d_p_opt1(t3d_new, resid,&
                                           X3D,&
                                           t2c, t3c, t3d,&
                                           pspace,&
                                           H1B_vv,&
                                           H2B_oovv, H2B_ovvo,&
                                           H2C_oovv, H2C_vvov, H2C_voov, H2C_vvvv,&
                                           fB_oo, fB_vv,&
                                           shift,&
                                           noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  real(kind=8), intent(in) :: t2c(nub, nub, nob, nob),&
                                              t3c(nua, nub, nub, noa, nob, nob),&
                                              t3d(nub, nub, nub, nob, nob, nob),&
                                              H1B_vv(nub, nub),&
                                              H2B_oovv(noa, nob, nua, nub),&
                                              H2B_ovvo(noa, nub, nua, nob),&
                                              H2C_oovv(nob, nob, nub, nub),&
                                              H2C_vvov(nub, nub, nob, nub),&
                                              H2C_voov(nub, nob, nob, nub),&
                                              H2C_vvvv(nub, nub, nub, nub),&
                                              fB_vv(nub, nub), fB_oo(nob, nob),&
                                              shift
                  integer, intent(in) :: pspace(nub, nub, nub, nob, nob, nob)

                  real(kind=8), intent(in) :: X3D(nub, nub, nub, nob, nob, nob)

                  real(kind=8), intent(out) :: t3d_new(nub, nub, nub, nob, nob, nob),&
                                               resid(nub, nub, nub, nob, nob, nob)

                  real(kind=8) :: val, denom
                  real(kind=8) :: res1, res2, res3, res4, res5
                  integer :: a, b, c, i, j, k, e, f, m, n

                  t3d_new = 0.0d0
                  resid = 0.0d0

                  ! loop over projection determinants in P space
                  do a = 1, nub; do b = a + 1, nub; do c = b + 1, nub;
                  do i = 1, nob; do j = i + 1, nob; do k = j + 1, nob;

                      if (pspace(a, b, c, i, j, k) /= 1) cycle

                          res1 = 0.0d0
                          res2 = 0.0d0
                          res3 = 0.0d0
                          res4 = 0.0d0
                          res5 = 0.0d0

                          do e = 1, nua
                              ! diagram 1: A(a/bc) h1B(ae) * t3d(ebcijk)
                              res1 = res1 + H1B_vv(a, e) * t3d(e, b, c, i, j, k)
                              res1 = res1 - H1B_vv(b, e) * t3d(e, a, c, i, j, k)
                              res1 = res1 - H1B_vv(c, e) * t3d(e, b, a, i, j, k)

                              ! diagram 2: 1/2 A(c/ab) h2C(abef) * t3d(efcijk)
                              do f = e + 1, nua
                                  res2 = res2 + H2C_vvvv(a, b, e, f) * t3d(e, f, c, i, j, k)
                                  res2 = res2 - H2C_vvvv(c, b, e, f) * t3d(e, f, a, i, j, k)
                                  res2 = res2 - H2C_vvvv(a, c, e, f) * t3d(e, f, b, i, j, k)
                              end do

                              ! diagram 3: A(i/jk)A(a/bc) h2B(maei) * t3c(ebcmjk)
                              do m = 1, noa
                                  res3 = res3 + H2B_ovvo(m, a, e, i) * t3c(e, b, c, m, j, k)
                                  res3 = res3 - H2B_ovvo(m, a, e, j) * t3c(e, b, c, m, i, k)
                                  res3 = res3 - H2B_ovvo(m, a, e, k) * t3c(e, b, c, m, j, i)
                                  res3 = res3 - H2B_ovvo(m, b, e, i) * t3c(e, a, c, m, j, k)
                                  res3 = res3 + H2B_ovvo(m, b, e, j) * t3c(e, a, c, m, i, k)
                                  res3 = res3 + H2B_ovvo(m, b, e, k) * t3c(e, a, c, m, j, i)
                                  res3 = res3 - H2B_ovvo(m, c, e, i) * t3c(e, b, a, m, j, k)
                                  res3 = res3 + H2B_ovvo(m, c, e, j) * t3c(e, b, a, m, i, k)
                                  res3 = res3 + H2B_ovvo(m, c, e, k) * t3c(e, b, a, m, j, i)
                              end do

                              ! diagram 4: A(i/jk)A(a/bc) h2C(amie) * t3d(ebcmjk)
                              do m = 1, nob
                                  res4 = res4 + H2C_voov(a, m, i, e) * t3d(e, b, c, m, j, k)
                                  res4 = res4 - H2C_voov(a, m, j, e) * t3d(e, b, c, m, i, k)
                                  res4 = res4 - H2C_voov(a, m, k, e) * t3d(e, b, c, m, j, i)
                                  res4 = res4 - H2C_voov(b, m, i, e) * t3d(e, a, c, m, j, k)
                                  res4 = res4 + H2C_voov(b, m, j, e) * t3d(e, a, c, m, i, k)
                                  res4 = res4 + H2C_voov(b, m, k, e) * t3d(e, a, c, m, j, i)
                                  res4 = res4 - H2C_voov(c, m, i, e) * t3d(e, b, a, m, j, k)
                                  res4 = res4 + H2C_voov(c, m, j, e) * t3d(e, b, a, m, i, k)
                                  res4 = res4 + H2C_voov(c, m, k, e) * t3d(e, b, a, m, j, i)
                              end do

                              ! diagram 5: A(i/jk)A(c/ab) h2C(abie) * t2c(ecjk)
                              res5 = res5 + H2C_vvov(a, b, i, e) * t2c(e, c, j, k)
                              res5 = res5 - H2C_vvov(c, b, i, e) * t2c(e, a, j, k)
                              res5 = res5 - H2C_vvov(a, c, i, e) * t2c(e, b, j, k)
                              res5 = res5 - H2C_vvov(a, b, j, e) * t2c(e, c, i, k)
                              res5 = res5 + H2C_vvov(c, b, j, e) * t2c(e, a, i, k)
                              res5 = res5 + H2C_vvov(a, c, j, e) * t2c(e, b, i, k)
                              res5 = res5 - H2C_vvov(a, b, k, e) * t2c(e, c, j, i)
                              res5 = res5 + H2C_vvov(c, b, k, e) * t2c(e, a, j, i)
                              res5 = res5 + H2C_vvov(a, c, k, e) * t2c(e, b, j, i)

                          end do

                          do e = 1, nub - nua
                              ! diagram 1: A(a/bc) h1B(ae) * t3d(ebcijk)
                              res1 = res1 + H1B_vv(a, e + nua) * t3d(e + nua, b, c, i, j, k)
                              res1 = res1 - H1B_vv(b, e + nua) * t3d(e + nua, a, c, i, j, k)
                              res1 = res1 - H1B_vv(c, e + nua) * t3d(e + nua, b, a, i, j, k)

                              ! diagram 2: 1/2 A(c/ab) h2C(abef) * t3d(efcijk)
                              do f = e + 1, nub - nua
                                  res2 = res2 + H2C_vvvv(a, b, e + nua, f + nua) * t3d(e + nua, f + nua, c, i, j, k)
                                  res2 = res2 - H2C_vvvv(c, b, e + nua, f + nua) * t3d(e + nua, f + nua, a, i, j, k)
                                  res2 = res2 - H2C_vvvv(a, c, e + nua, f + nua) * t3d(e + nua, f + nua, b, i, j, k)
                              end do

                              ! diagram 4: A(i/jk)A(a/bc) h2C(amie) * t3d(ebcmjk)
                              do m = 1, nob
                                  res4 = res4 + H2C_voov(a, m, i, e + nua) * t3d(e + nua, b, c, m, j, k)
                                  res4 = res4 - H2C_voov(a, m, j, e + nua) * t3d(e + nua, b, c, m, i, k)
                                  res4 = res4 - H2C_voov(a, m, k, e + nua) * t3d(e + nua, b, c, m, j, i)
                                  res4 = res4 - H2C_voov(b, m, i, e + nua) * t3d(e + nua, a, c, m, j, k)
                                  res4 = res4 + H2C_voov(b, m, j, e + nua) * t3d(e + nua, a, c, m, i, k)
                                  res4 = res4 + H2C_voov(b, m, k, e + nua) * t3d(e + nua, a, c, m, j, i)
                                  res4 = res4 - H2C_voov(c, m, i, e + nua) * t3d(e + nua, b, a, m, j, k)
                                  res4 = res4 + H2C_voov(c, m, j, e + nua) * t3d(e + nua, b, a, m, i, k)
                                  res4 = res4 + H2C_voov(c, m, k, e + nua) * t3d(e + nua, b, a, m, j, i)
                              end do

                              ! diagram 5: A(i/jk)A(c/ab) h2C(abie) * t2c(ecjk)
                              res5 = res5 + H2C_vvov(a, b, i, e + nua) * t2c(e + nua, c, j, k)
                              res5 = res5 - H2C_vvov(c, b, i, e + nua) * t2c(e + nua, a, j, k)
                              res5 = res5 - H2C_vvov(a, c, i, e + nua) * t2c(e + nua, b, j, k)
                              res5 = res5 - H2C_vvov(a, b, j, e + nua) * t2c(e + nua, c, i, k)
                              res5 = res5 + H2C_vvov(c, b, j, e + nua) * t2c(e + nua, a, i, k)
                              res5 = res5 + H2C_vvov(a, c, j, e + nua) * t2c(e + nua, b, i, k)
                              res5 = res5 - H2C_vvov(a, b, k, e + nua) * t2c(e + nua, c, j, i)
                              res5 = res5 + H2C_vvov(c, b, k, e + nua) * t2c(e + nua, a, j, i)
                              res5 = res5 + H2C_vvov(a, c, k, e + nua) * t2c(e + nua, b, j, i)

                          end do

                          denom = fB_oo(I, I) + fB_oo(J, J) + fB_oo(K, K) - fB_vv(A, A) - fB_vv(B, B) - fB_vv(C, C)

                          val = X3D(a,b,c,i,j,k)&
                                  -X3D(b,a,c,i,j,k)&
                                  -X3D(a,c,b,i,j,k)&
                                  +X3D(b,c,a,i,j,k)&
                                  -X3D(c,b,a,i,j,k)&
                                  +X3D(c,a,b,i,j,k)&
                                  -X3D(a,b,c,j,i,k)&
                                  +X3D(b,a,c,j,i,k)&
                                  +X3D(a,c,b,j,i,k)&
                                  -X3D(b,c,a,j,i,k)&
                                  +X3D(c,b,a,j,i,k)&
                                  -X3D(c,a,b,j,i,k)&
                                  -X3D(a,b,c,i,k,j)&
                                  +X3D(b,a,c,i,k,j)&
                                  +X3D(a,c,b,i,k,j)&
                                  -X3D(b,c,a,i,k,j)&
                                  +X3D(c,b,a,i,k,j)&
                                  -X3D(c,a,b,i,k,j)&
                                  -X3D(a,b,c,k,j,i)&
                                  +X3D(b,a,c,k,j,i)&
                                  +X3D(a,c,b,k,j,i)&
                                  -X3D(b,c,a,k,j,i)&
                                  +X3D(c,b,a,k,j,i)&
                                  -X3D(c,a,b,k,j,i)&
                                  +X3D(a,b,c,j,k,i)&
                                  -X3D(b,a,c,j,k,i)&
                                  -X3D(a,c,b,j,k,i)&
                                  +X3D(b,c,a,j,k,i)&
                                  -X3D(c,b,a,j,k,i)&
                                  +X3D(c,a,b,j,k,i)&
                                  +X3D(a,b,c,k,i,j)&
                                  -X3D(b,a,c,k,i,j)&
                                  -X3D(a,c,b,k,i,j)&
                                  +X3D(b,c,a,k,i,j)&
                                  -X3D(c,b,a,k,i,j)&
                                  +X3D(c,a,b,k,i,j)

                          val = val + res1 + res2 + res3 + res4 + res5
                          val = val/(denom - shift)

                          t3d_new(A,B,C,I,J,K) = t3d(A,B,C,I,J,K) + val
                          t3d_new(A,B,C,K,I,J) = t3d_new(A,B,C,I,J,K)
                          t3d_new(A,B,C,J,K,I) = t3d_new(A,B,C,I,J,K)
                          t3d_new(A,B,C,I,K,J) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(A,B,C,J,I,K) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(A,B,C,K,J,I) = -t3d_new(A,B,C,I,J,K)

                          t3d_new(B,A,C,I,J,K) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(B,A,C,K,I,J) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(B,A,C,J,K,I) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(B,A,C,I,K,J) = t3d_new(A,B,C,I,J,K)
                          t3d_new(B,A,C,J,I,K) = t3d_new(A,B,C,I,J,K)
                          t3d_new(B,A,C,K,J,I) = t3d_new(A,B,C,I,J,K)

                          t3d_new(A,C,B,I,J,K) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(A,C,B,K,I,J) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(A,C,B,J,K,I) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(A,C,B,I,K,J) = t3d_new(A,B,C,I,J,K)
                          t3d_new(A,C,B,J,I,K) = t3d_new(A,B,C,I,J,K)
                          t3d_new(A,C,B,K,J,I) = t3d_new(A,B,C,I,J,K)

                          t3d_new(C,B,A,I,J,K) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(C,B,A,K,I,J) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(C,B,A,J,K,I) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(C,B,A,I,K,J) = t3d_new(A,B,C,I,J,K)
                          t3d_new(C,B,A,J,I,K) = t3d_new(A,B,C,I,J,K)
                          t3d_new(C,B,A,K,J,I) = t3d_new(A,B,C,I,J,K)

                          t3d_new(B,C,A,I,J,K) = t3d_new(A,B,C,I,J,K)
                          t3d_new(B,C,A,K,I,J) = t3d_new(A,B,C,I,J,K)
                          t3d_new(B,C,A,J,K,I) = t3d_new(A,B,C,I,J,K)
                          t3d_new(B,C,A,I,K,J) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(B,C,A,J,I,K) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(B,C,A,K,J,I) = -t3d_new(A,B,C,I,J,K)

                          t3d_new(C,A,B,I,J,K) = t3d_new(A,B,C,I,J,K)
                          t3d_new(C,A,B,K,I,J) = t3d_new(A,B,C,I,J,K)
                          t3d_new(C,A,B,J,K,I) = t3d_new(A,B,C,I,J,K)
                          t3d_new(C,A,B,I,K,J) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(C,A,B,J,I,K) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(C,A,B,K,J,I) = -t3d_new(A,B,C,I,J,K)

                          resid(A,B,C,I,J,K) = val
                          resid(A,B,C,K,I,J) = val
                          resid(A,B,C,J,K,I) = val
                          resid(A,B,C,I,K,J) = -val
                          resid(A,B,C,J,I,K) = -val
                          resid(A,B,C,K,J,I) = -val
                          resid(B,C,A,I,J,K) = val
                          resid(B,C,A,K,I,J) = val
                          resid(B,C,A,J,K,I) = val
                          resid(B,C,A,I,K,J) = -val
                          resid(B,C,A,J,I,K) = -val
                          resid(B,C,A,K,J,I) = -val
                          resid(C,A,B,I,J,K) = val
                          resid(C,A,B,K,I,J) = val
                          resid(C,A,B,J,K,I) = val
                          resid(C,A,B,I,K,J) = -val
                          resid(C,A,B,J,I,K) = -val
                          resid(C,A,B,K,J,I) = -val
                          resid(A,C,B,I,J,K) = -val
                          resid(A,C,B,K,I,J) = -val
                          resid(A,C,B,J,K,I) = -val
                          resid(A,C,B,I,K,J) = val
                          resid(A,C,B,J,I,K) = val
                          resid(A,C,B,K,J,I) = val
                          resid(B,A,C,I,J,K) = -val
                          resid(B,A,C,K,I,J) = -val
                          resid(B,A,C,J,K,I) = -val
                          resid(B,A,C,I,K,J) = val
                          resid(B,A,C,J,I,K) = val
                          resid(B,A,C,K,J,I) = val
                          resid(C,B,A,I,J,K) = -val
                          resid(C,B,A,K,I,J) = -val
                          resid(C,B,A,J,K,I) = -val
                          resid(C,B,A,I,K,J) = val
                          resid(C,B,A,J,I,K) = val
                          resid(C,B,A,K,J,I) = val

                  end do; end do; end do;
                  end do; end do; end do;

              end subroutine update_t3d_p_opt1


          !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
          !!!!!!!!!!!!!!!!!!!!!!!! OPT 2 !!!!!!!!!!!!!!!!!!!
          !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

              subroutine update_t3a_p_opt2(t3a_new, resid,&
                                           X3A,&
                                           t2a, t3a, t3b,&
                                           pspace_aaa, pspace_aab,&
                                           H1A_vv,&
                                           H2A_oovv, H2A_vvov, H2A_voov, H2A_vvvv,&
                                           H2B_oovv, H2B_voov,&
                                           fA_oo, fA_vv,&
                                           shift,&
                                           noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  real(kind=8), intent(in) :: t2a(nua, nua, noa, noa),&
                                              t3a(nua, nua, nua, noa, noa ,noa),&
                                              t3b(nua, nua, nub, noa, noa, nob),&
                                              H1A_vv(nua, nua),&
                                              H2A_oovv(noa, noa, nua, nua),&
                                              H2A_vvov(nua, nua, noa, nua),&
                                              H2A_voov(nua, noa, noa, nua),&
                                              H2A_vvvv(nua, nua, nua, nua),&
                                              H2B_oovv(noa, nob, nua, nub),&
                                              H2B_voov(nua, nob, noa, nub),&
                                              fA_vv(nua, nua), fA_oo(noa, noa),&
                                              shift
                  integer, intent(in) :: pspace_aaa(nua, nua, nua, noa, noa, noa),&
                                         pspace_aab(nua, nua, nub, noa, noa, nob)

                  real(kind=8), intent(in) :: X3A(nua, nua, nua, noa, noa, noa)

                  real(kind=8), intent(out) :: t3a_new(nua, nua, nua, noa, noa, noa),&
                                               resid(nua, nua, nua, noa, noa, noa)

                  real(kind=8) :: I2A_vvov(nua, nua, noa, nua), val, denom
                  real(kind=8) :: res1, res2, res3, res4, res5
                  integer :: idx, a, b, c, i, j, k, e, f, m, n

                  t3a_new = 0.0d0
                  resid = 0.0d0

                  I2A_vvov = H2A_vvov

                  ! loop over projection determinants in P space
                  do a = 1, nua; do b = a + 1, nua; do c = b + 1, nua;
                  do i = 1, noa; do j = i + 1, noa; do k = j + 1, noa;

                      if (pspace_aaa(a, b, c, i, j, k) /= 1) cycle

                      res1 = 0.0d0
                      res2 = 0.0d0
                      res3 = 0.0d0
                      res4 = 0.0d0
                      res5 = 0.0d0

                      do e = 1, nua

                          ! A(a/bc) h1a(ae) * t3a(ebcijk)
                          if (pspace_aaa(e, b, c, i, j, k) == 1) then
                            res1 = res1 + H1A_vv(a, e) * t3a(e, b, c, i, j, k)
                          end if
                          if (pspace_aaa(e, a, c, i, j, k) == 1) then
                            res1 = res1 - H1A_vv(b, e) * t3a(e, a, c, i, j, k)
                          end if
                          if (pspace_aaa(e, b, a, i, j, k) == 1) then
                            res1 = res1 - H1A_vv(c, e) * t3a(e, b, a, i, j, k)
                          end if

                          ! 1/2 A(c/ab) h2a(abef) * t3a(efcijk)
                          do f = e + 1, nua
                              if (pspace_aaa(e, f, c, i, j, k) == 1) then
                                res2 = res2 + H2A_vvvv(a, b, e, f) * t3a(e, f, c, i, j, k)
                              end if
                              if (pspace_aaa(e, f, a, i, j, k) == 1) then
                                res2 = res2 - H2A_vvvv(c, b, e, f) * t3a(e, f, a, i, j, k)
                              end if
                              if (pspace_aaa(e, f, b, i, j, k) == 1) then
                                res2 = res2 - H2A_vvvv(a, c, e, f) * t3a(e, f, b, i, j, k)
                              end if
                          end do

                          ! A(i/jk)A(c/ab) h2a(amie) * t3a(ebcmjk)
                          do m = 1, noa
                              if (pspace_aaa(e, b, c, m, j, k) == 1) then
                                res3 = res3 + H2A_voov(a, m, i, e) * t3a(e, b, c, m, j, k)
                              end if
                              if (pspace_aaa(e, b, c, m, i, k) == 1) then
                                res3 = res3 - H2A_voov(a, m, j, e) * t3a(e, b, c, m, i, k)
                              end if
                              if (pspace_aaa(e, b, c, m, j, i) == 1) then
                                res3 = res3 - H2A_voov(a, m, k, e) * t3a(e, b, c, m, j, i)
                              end if
                              if (pspace_aaa(e, a, c, m, j, k) == 1) then
                                res3 = res3 - H2A_voov(b, m, i, e) * t3a(e, a, c, m, j, k)
                              end if
                              if (pspace_aaa(e, a, c, m, i, k) == 1) then
                                res3 = res3 + H2A_voov(b, m, j, e) * t3a(e, a, c, m, i, k)
                              end if
                              if (pspace_aaa(e, a, c, m, j, i) == 1) then
                                res3 = res3 + H2A_voov(b, m, k, e) * t3a(e, a, c, m, j, i)
                              end if
                              if (pspace_aaa(e, b, a, m, j, k) == 1) then
                                res3 = res3 - H2A_voov(c, m, i, e) * t3a(e, b, a, m, j, k)
                              end if
                              if (pspace_aaa(e, b, a, m, i, k) == 1) then
                                res3 = res3 + H2A_voov(c, m, j, e) * t3a(e, b, a, m, i, k)
                              end if
                              if (pspace_aaa(e, b, a, m, j, i) == 1) then
                                res3 = res3 + H2A_voov(c, m, k, e) * t3a(e, b, a, m, j, i)
                              end if
                          end do

                          do m = 1, nob
                              ! A(i/jk)A(a/bc) h2b(amie) * t3b(bcejkm)
                              if (pspace_aab(b, c, e, j, k, m) == 1) then
                                res5 = res5 + H2B_voov(a, m, i, e) * t3b(b, c, e, j, k, m)
                              end if
                              if (pspace_aab(a, c, e, j, k, m) == 1) then
                                res5 = res5 - H2B_voov(b, m, i, e) * t3b(a, c, e, j, k, m)
                              end if
                              if (pspace_aab(b, a, e, j, k, m) == 1) then
                                res5 = res5 - H2B_voov(c, m, i, e) * t3b(b, a, e, j, k, m)
                              end if
                              if (pspace_aab(b, c, e, i, k, m) == 1) then
                                res5 = res5 - H2B_voov(a, m, j, e) * t3b(b, c, e, i, k, m)
                              end if
                              if (pspace_aab(a, c, e, i, k, m) == 1) then
                                res5 = res5 + H2B_voov(b, m, j, e) * t3b(a, c, e, i, k, m)
                              end if
                              if (pspace_aab(b, a, e, i, k, m) == 1) then
                                res5 = res5 + H2B_voov(c, m, j, e) * t3b(b, a, e, i, k, m)
                              end if
                              if (pspace_aab(b, c, e, j, i, m) == 1) then
                                res5 = res5 - H2B_voov(a, m, k, e) * t3b(b, c, e, j, i, m)
                              end if
                              if (pspace_aab(a, c, e, j, i, m) == 1) then
                                res5 = res5 + H2B_voov(b, m, k, e) * t3b(a, c, e, j, i, m)
                              end if
                              if (pspace_aab(b, a, e, j, i, m) == 1) then
                                res5 = res5 + H2B_voov(c, m, k, e) * t3b(b, a, e, j, i, m)
                              end if
                          end do


                          ! A(i/jk)(c/ab) h2a(abie) * t2a(ecjk)
                          res4 = res4 + I2A_vvov(a, b, i, e) * t2a(e, c, j, k)
                          res4 = res4 - I2A_vvov(c, b, i, e) * t2a(e, a, j, k)
                          res4 = res4 - I2A_vvov(a, c, i, e) * t2a(e, b, j, k)
                          res4 = res4 - I2A_vvov(a, b, j, e) * t2a(e, c, i, k)
                          res4 = res4 + I2A_vvov(c, b, j, e) * t2a(e, a, i, k)
                          res4 = res4 + I2A_vvov(a, c, j, e) * t2a(e, b, i, k)
                          res4 = res4 - I2A_vvov(a, b, k, e) * t2a(e, c, j, i)
                          res4 = res4 + I2A_vvov(c, b, k, e) * t2a(e, a, j, i)
                          res4 = res4 + I2A_vvov(a, c, k, e) * t2a(e, b, j, i)


                      end do

                      do e = 1, nub - nua
                          do m = 1, nob
                              ! A(i/jk)A(a/bc) h2b(amie) * t3b(bcejkm)
                              if (pspace_aab(b, c, e + nua, j, k, m) == 1) then
                                res5 = res5 + H2B_voov(a, m, i, e + nua) * t3b(b, c, e + nua, j, k, m)
                              end if
                              if (pspace_aab(a, c, e + nua, j, k, m) == 1) then
                                res5 = res5 - H2B_voov(b, m, i, e + nua) * t3b(a, c, e + nua, j, k, m)
                              end if
                              if (pspace_aab(b, a, e + nua, j, k, m) == 1) then
                                res5 = res5 - H2B_voov(c, m, i, e + nua) * t3b(b, a, e + nua, j, k, m)
                              end if
                              if (pspace_aab(b, c, e + nua, i, k, m) == 1) then
                                res5 = res5 - H2B_voov(a, m, j, e + nua) * t3b(b, c, e + nua, i, k, m)
                              end if
                              if (pspace_aab(a, c, e + nua, i, k, m) == 1) then
                                res5 = res5 + H2B_voov(b, m, j, e + nua) * t3b(a, c, e + nua, i, k, m)
                              end if
                              if (pspace_aab(b, a, e + nua, i, k, m) == 1) then
                                res5 = res5 + H2B_voov(c, m, j, e + nua) * t3b(b, a, e + nua, i, k, m)
                              end if
                              if (pspace_aab(b, c, e + nua, j, i, m) == 1) then
                                res5 = res5 - H2B_voov(a, m, k, e + nua) * t3b(b, c, e + nua, j, i, m)
                              end if
                              if (pspace_aab(a, c, e + nua, j, i, m) == 1) then
                                res5 = res5 + H2B_voov(b, m, k, e + nua) * t3b(a, c, e + nua, j, i, m)
                              end if
                              if (pspace_aab(b, a, e + nua, j, i, m) == 1) then
                                res5 = res5 + H2B_voov(c, m, k, e + nua) * t3b(b, a, e + nua, j, i, m)
                              end if
                          end do
                      end do

                      denom = fA_oo(I, I) + fA_oo(J, J) + fA_oo(K, K) - fA_vv(A, A) - fA_vv(B, B) - fA_vv(C, C)

                      val = X3A(a,b,c,i,j,k)&
                              -X3A(b,a,c,i,j,k)&
                              -X3A(a,c,b,i,j,k)&
                              +X3A(b,c,a,i,j,k)&
                              -X3A(c,b,a,i,j,k)&
                              +X3A(c,a,b,i,j,k)&
                              -X3A(a,b,c,j,i,k)&
                              +X3A(b,a,c,j,i,k)&
                              +X3A(a,c,b,j,i,k)&
                              -X3A(b,c,a,j,i,k)&
                              +X3A(c,b,a,j,i,k)&
                              -X3A(c,a,b,j,i,k)&
                              -X3A(a,b,c,i,k,j)&
                              +X3A(b,a,c,i,k,j)&
                              +X3A(a,c,b,i,k,j)&
                              -X3A(b,c,a,i,k,j)&
                              +X3A(c,b,a,i,k,j)&
                              -X3A(c,a,b,i,k,j)&
                              -X3A(a,b,c,k,j,i)&
                              +X3A(b,a,c,k,j,i)&
                              +X3A(a,c,b,k,j,i)&
                              -X3A(b,c,a,k,j,i)&
                              +X3A(c,b,a,k,j,i)&
                              -X3A(c,a,b,k,j,i)&
                              +X3A(a,b,c,j,k,i)&
                              -X3A(b,a,c,j,k,i)&
                              -X3A(a,c,b,j,k,i)&
                              +X3A(b,c,a,j,k,i)&
                              -X3A(c,b,a,j,k,i)&
                              +X3A(c,a,b,j,k,i)&
                              +X3A(a,b,c,k,i,j)&
                              -X3A(b,a,c,k,i,j)&
                              -X3A(a,c,b,k,i,j)&
                              +X3A(b,c,a,k,i,j)&
                              -X3A(c,b,a,k,i,j)&
                              +X3A(c,a,b,k,i,j)

                      val = val + res1 + res2 + res3 + res4 + res5

                      val = val/(denom - shift)

                      ! update
                      t3a_new(A,B,C,I,J,K) = t3a(A,B,C,I,J,K) + val

                      t3a_new(A,B,C,K,I,J) = t3a_new(A,B,C,I,J,K)
                      t3a_new(A,B,C,J,K,I) = t3a_new(A,B,C,I,J,K)
                      t3a_new(A,B,C,I,K,J) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(A,B,C,J,I,K) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(A,B,C,K,J,I) = -t3a_new(A,B,C,I,J,K)

                      t3a_new(B,A,C,I,J,K) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(B,A,C,K,I,J) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(B,A,C,J,K,I) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(B,A,C,I,K,J) = t3a_new(A,B,C,I,J,K)
                      t3a_new(B,A,C,J,I,K) = t3a_new(A,B,C,I,J,K)
                      t3a_new(B,A,C,K,J,I) = t3a_new(A,B,C,I,J,K)

                      t3a_new(A,C,B,I,J,K) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(A,C,B,K,I,J) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(A,C,B,J,K,I) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(A,C,B,I,K,J) = t3a_new(A,B,C,I,J,K)
                      t3a_new(A,C,B,J,I,K) = t3a_new(A,B,C,I,J,K)
                      t3a_new(A,C,B,K,J,I) = t3a_new(A,B,C,I,J,K)

                      t3a_new(C,B,A,I,J,K) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(C,B,A,K,I,J) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(C,B,A,J,K,I) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(C,B,A,I,K,J) = t3a_new(A,B,C,I,J,K)
                      t3a_new(C,B,A,J,I,K) = t3a_new(A,B,C,I,J,K)
                      t3a_new(C,B,A,K,J,I) = t3a_new(A,B,C,I,J,K)

                      t3a_new(B,C,A,I,J,K) = t3a_new(A,B,C,I,J,K)
                      t3a_new(B,C,A,K,I,J) = t3a_new(A,B,C,I,J,K)
                      t3a_new(B,C,A,J,K,I) = t3a_new(A,B,C,I,J,K)
                      t3a_new(B,C,A,I,K,J) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(B,C,A,J,I,K) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(B,C,A,K,J,I) = -t3a_new(A,B,C,I,J,K)

                      t3a_new(C,A,B,I,J,K) = t3a_new(A,B,C,I,J,K)
                      t3a_new(C,A,B,K,I,J) = t3a_new(A,B,C,I,J,K)
                      t3a_new(C,A,B,J,K,I) = t3a_new(A,B,C,I,J,K)
                      t3a_new(C,A,B,I,K,J) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(C,A,B,J,I,K) = -t3a_new(A,B,C,I,J,K)
                      t3a_new(C,A,B,K,J,I) = -t3a_new(A,B,C,I,J,K)

                      resid(A,B,C,I,J,K) = val
                      resid(A,B,C,K,I,J) = val
                      resid(A,B,C,J,K,I) = val
                      resid(A,B,C,I,K,J) = -val
                      resid(A,B,C,J,I,K) = -val
                      resid(A,B,C,K,J,I) = -val
                      resid(B,C,A,I,J,K) = val
                      resid(B,C,A,K,I,J) = val
                      resid(B,C,A,J,K,I) = val
                      resid(B,C,A,I,K,J) = -val
                      resid(B,C,A,J,I,K) = -val
                      resid(B,C,A,K,J,I) = -val
                      resid(C,A,B,I,J,K) = val
                      resid(C,A,B,K,I,J) = val
                      resid(C,A,B,J,K,I) = val
                      resid(C,A,B,I,K,J) = -val
                      resid(C,A,B,J,I,K) = -val
                      resid(C,A,B,K,J,I) = -val
                      resid(A,C,B,I,J,K) = -val
                      resid(A,C,B,K,I,J) = -val
                      resid(A,C,B,J,K,I) = -val
                      resid(A,C,B,I,K,J) = val
                      resid(A,C,B,J,I,K) = val
                      resid(A,C,B,K,J,I) = val
                      resid(B,A,C,I,J,K) = -val
                      resid(B,A,C,K,I,J) = -val
                      resid(B,A,C,J,K,I) = -val
                      resid(B,A,C,I,K,J) = val
                      resid(B,A,C,J,I,K) = val
                      resid(B,A,C,K,J,I) = val
                      resid(C,B,A,I,J,K) = -val
                      resid(C,B,A,K,I,J) = -val
                      resid(C,B,A,J,K,I) = -val
                      resid(C,B,A,I,K,J) = val
                      resid(C,B,A,J,I,K) = val
                      resid(C,B,A,K,J,I) = val

                  end do; end do; end do;
                  end do; end do; end do;

              end subroutine update_t3a_p_opt2

              subroutine update_t3b_p_opt2(t3b_new, resid,&
                                           X3B,&
                                           t2a, t2b, t3a, t3b, t3c,&
                                           pspace_aaa, pspace_aab, pspace_abb,&
                                           H1A_vv, H1B_vv,&
                                           H2A_oovv, H2A_vvov, H2A_voov, H2A_vvvv,&
                                           H2B_oovv, H2B_vvov, H2B_vvvo, H2B_voov, H2B_vovo, H2B_ovov, H2B_ovvo, H2B_vvvv,&
                                           H2C_voov,&
                                           fA_oo, fA_vv, fB_oo, fB_vv,&
                                           shift,&
                                           noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: pspace_aaa(nua, nua, nua, noa, noa, noa),&
                                         pspace_aab(nua, nua, nub, noa, noa, nob),&
                                         pspace_abb(nua, nub, nub, noa, nob, nob)
                  real(8), intent(in) :: t2a(1:nua,1:nua,1:noa,1:noa),&
                                         t2b(1:nua,1:nub,1:noa,1:nob),&
                                         t3a(1:nua,1:nua,1:nua,1:noa,1:noa,1:noa),&
                                         t3b(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob),&
                                         t3c(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob),&
                                         H1A_vv(1:nua,1:nua),&
                                         H1B_vv(1:nub,1:nub),&
                                         H2A_oovv(1:noa,1:noa,1:nua,1:nua),&
                                         H2A_vvov(1:nua,1:nua,1:noa,1:nua),&
                                         H2A_voov(1:nua,1:noa,1:noa,1:nua),&
                                         H2A_vvvv(1:nua,1:nua,1:nua,1:nua),&
                                         H2B_oovv(1:noa,1:nob,1:nua,1:nub),&
                                         H2B_vvov(1:nua,1:nub,1:noa,1:nub),&
                                         H2B_vvvo(1:nua,1:nub,1:nua,1:nob),&
                                         H2B_voov(1:nua,1:nob,1:noa,1:nub),&
                                         H2B_vovo(1:nua,1:nob,1:nua,1:nob),&
                                         H2B_ovov(1:noa,1:nub,1:noa,1:nub),&
                                         H2B_ovvo(1:noa,1:nub,1:nua,1:nob),&
                                         H2B_vvvv(1:nub,1:nub,1:nub,1:nub),&
                                         H2C_voov(1:nub,1:nob,1:nob,1:nub),&
                                         fA_oo(1:noa,1:noa), fA_vv(1:nua,1:nua),&
                                         fB_oo(1:nob,1:nob), fB_vv(1:nub,1:nub),&
                                         X3B(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob),&
                                         shift

                  real(8), intent(out) :: t3b_new(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob),&
                                          resid(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob)
                  integer :: i, j, k, a, b, c, m, n, e, f
                  real(8) :: denom, val,&
                             res1, res2, res3, res4, res5, res6, res7, res8, res9, res10, res11, res12, res13

                  resid = 0.0d0
                  t3b_new = 0.0d0

                  do i = 1, noa; do j = i + 1, noa; do k = 1, nob;
                  do a = 1, nua; do b = a + 1, nua; do c = 1, nub;

                      if (pspace_aab(a, b, c, i, j, k) /= 1) cycle

                      res1 = 0.0d0
                      res2 = 0.0d0
                      res3 = 0.0d0
                      res4 = 0.0d0
                      res5 = 0.0d0
                      res6 = 0.0d0
                      res7 = 0.0d0
                      res8 = 0.0d0
                      res9 = 0.0d0
                      res10 = 0.0d0
                      res11 = 0.0d0
                      res12 = 0.0d0
                      res13 = 0.0d0

                      ! nua < nub
                      do e = 1, nua

                          ! diagram 1: A(ab) h1A(ae) * t3b(ebcijk)
                          if (pspace_aab(e, b, c, i, j, k) == 1) then
                            res1 = res1 + H1A_vv(a, e) * t3b(e, b, c, i, j, k)
                          end if
                          if (pspace_aab(e, a, c, i, j, k) == 1) then
                            res1 = res1 - H1A_vv(b, e) * t3b(e, a, c, i, j, k)
                          end if

                          ! diagram 2: h1B(ce) * t3b(abeijk)
                          if (pspace_aab(a, b, e, i, j, k) == 1) then
                            res2 = res2 + H1B_vv(c, e) * t3b(a, b, e, i, j, k)
                          end if

                          ! diagram 3: 0.5 * h2A(abef) * t3b(efcijk)
                          ! diagram 4: A(ab) h2B(bcef) * t3b(aefijk)
                          do f = 1, nua
                              if (pspace_aab(e, f, c, i, j, k) == 1) then
                                res3 = res3 + 0.5d0 * H2A_vvvv(a, b, e, f) * t3b(e, f, c, i, j, k)
                              end if
                              if (pspace_aab(a, e, f, i, j, k) == 1) then
                                res4 = res4 + H2B_vvvv(b, c, e, f) * t3b(a, e, f, i, j, k)
                              end if
                              if (pspace_aab(b, e, f, i, j, k) == 1) then
                                res4 = res4 - H2B_vvvv(a, c, e, f) * t3b(b, e, f, i, j, k)
                              end if
                          end do
                          do f = 1, nub - nua
                              if (pspace_aab(a, e, f + nua, i, j, k) == 1) then
                                res4 = res4 + H2B_vvvv(b, c, e, f + nua) * t3b(a, e, f + nua, i, j, k)
                              end if
                              if (pspace_aab(b, e, f + nua, i, j, k) == 1) then
                                res4 = res4 - H2B_vvvv(a, c, e, f + nua) * t3b(b, e, f + nua, i, j, k)
                              end if
                          end do

                          ! diagram 5: A(ij)A(ab) h2A(amie) * t3b(ebcmjk)
                          ! diagram 7: h2B(mcek) * t3a(abeijm)
                          ! diagram 10: -A(ij) h2B(mcie) * t3b(abemjk)
                          do m = 1, noa
                              if (pspace_aab(e, b, c, m, j, k) == 1) then
                                res5 = res5 + H2A_voov(a, m, i, e) * t3b(e, b, c, m, j, k)
                              end if
                              if (pspace_aab(e, a, c, m, j, k) == 1) then
                                res5 = res5 - H2A_voov(b, m, i, e) * t3b(e, a, c, m, j, k)
                              end if
                              if (pspace_aab(e, b, c, m, i, k) == 1) then
                                res5 = res5 - H2A_voov(a, m, j, e) * t3b(e, b, c, m, i, k)
                              end if
                              if (pspace_aab(e, a, c, m, i, k) == 1) then
                                res5 = res5 + H2A_voov(b, m, j, e) * t3b(e, a, c, m, i, k)
                              end if

                              if (pspace_aaa(a, b, e, i, j, m) == 1) then
                                res7 = res7 + H2B_ovvo(m, c, e, k) * t3a(a, b, e, i, j, m)
                              end if

                              if (pspace_aab(a, b, e, m, j, k) == 1) then
                                res10 = res10 - H2B_ovov(m, c, i, e) * t3b(a, b, e, m, j, k)
                              end if
                              if (pspace_aab(a, b, e, m, i, k) == 1) then
                                res10 = res10 + H2B_ovov(m, c, j, e) * t3b(a, b, e, m, i, k)
                              end if
                          end do

                          ! diagram 6: A(ij)A(ab) h2B(amie) * t3c(becjmk)
                          ! diagram 8: h2C(cmke) * t3b(abeijm)
                          ! diagram 9: -A(ab) h2B(amek) * t3b(ebcijm)
                          do m = 1, nob
                              if (pspace_abb(b, e, c, j, m, k) == 1) then
                                res6 = res6 + H2B_voov(a, m, i, e) * t3c(b, e, c, j, m, k)
                              end if
                              if (pspace_abb(a, e, c, j, m, k) == 1) then
                                res6 = res6 - H2B_voov(b, m, i, e) * t3c(a, e, c, j, m, k)
                              end if
                              if (pspace_abb(b, e, c, i, m, k) == 1) then
                                res6 = res6 - H2B_voov(a, m, j, e) * t3c(b, e, c, i, m, k)
                              end if
                              if (pspace_abb(a, e, c, i, m, k) == 1) then
                                res6 = res6 + H2B_voov(b, m, j, e) * t3c(a, e, c, i, m, k)
                              end if

                              if (pspace_aab(a, b, e, i, j, m) == 1) then
                                res8 = res8 + H2C_voov(c, m, k, e) * t3b(a, b, e, i, j, m)
                              end if

                              if (pspace_aab(e, b, c, i, j, m) == 1) then
                                res9 = res9 - H2B_vovo(a, m, e, k) * t3b(e, b, c, i, j, m)
                              end if
                              if (pspace_aab(e, a, c, i, j, m) == 1) then
                                res9 = res9 + H2B_vovo(b, m, e, k) * t3b(e, a, c, i, j, m)
                              end if
                          end do

                          ! diagram 11: A(ab) I2B(bcek) * t2a(aeij)
                          res11 = res11 + H2B_vvvo(b, c, e, k) * t2a(a, e, i, j)
                          res11 = res11 - H2B_vvvo(a, c, e, k) * t2a(b, e, i, j)
                          ! diagram 12: A(ij)A(ab) I2B(acie) * t2b(bejk)
                          res12 = res12 + H2B_vvov(a, c, i, e) * t2b(b, e, j, k)
                          res12 = res12 - H2B_vvov(a, c, j, e) * t2b(b, e, i, k)
                          res12 = res12 - H2B_vvov(b, c, i, e) * t2b(a, e, j, k)
                          res12 = res12 + H2B_vvov(b, c, j, e) * t2b(a, e, i, k)
                          ! diagram 13: A(ij) I2A(abie) * t2b(ecjk)
                          res13 = res13 + H2A_vvov(a, b, i, e) * t2b(e, c, j, k)
                          res13 = res13 - H2A_vvov(a, b, j, e) * t2b(e, c, i, k)

                      end do

                      do e = 1, nub - nua

                          ! diagram 2 : h1B(ce) * t3b(abeijk)
                          if (pspace_aab(a, b, e + nua, i, j, k) == 1) then
                            res2 = res2 + H1B_vv(c, e + nua) * t3b(a, b, e + nua, i, j, k)
                          end if

                          ! diagram 10: -A(ij) h2B(mcie) * t3b(abemjk)
                          do m = 1, noa
                              if (pspace_aab(a, b, e + nua, m, j, k) == 1) then
                                res10 = res10 - H2B_ovov(m, c, i, e + nua) * t3b(a, b, e + nua, m, j, k)
                              end if
                              if (pspace_aab(a, b, e + nua, m, i, k) == 1) then
                                res10 = res10 + H2B_ovov(m, c, j, e + nua) * t3b(a, b, e + nua, m, i, k)
                              end if
                          end do

                          ! diagram 6: A(ij)A(ab) h2B(amie) * t3c(becjmk)
                          ! diagram 8: h2C(cmke) * t3b(abeijm)
                          do m = 1, nob
                              if (pspace_abb(b, e + nua, c, j, m, k) == 1) then
                                res6 = res6 + H2B_voov(a, m, i, e + nua) * t3c(b, e + nua, c, j, m, k)
                              end if
                              if (pspace_abb(a, e + nua, c, j, m, k) == 1) then
                                res6 = res6 - H2B_voov(b, m, i, e + nua) * t3c(a, e + nua, c, j, m, k)
                              end if
                              if (pspace_abb(b, e + nua, c, i, m, k) == 1) then
                              res6 = res6 - H2B_voov(a, m, j, e + nua) * t3c(b, e + nua, c, i, m, k)
                              end if
                              if (pspace_abb(a, e + nua, c, i, m, k) == 1) then
                                res6 = res6 + H2B_voov(b, m, j, e + nua) * t3c(a, e + nua, c, i, m, k)
                              end if

                              if (pspace_aab(a, b, e + nua, i, j, m) == 1) then
                                res8 = res8 + H2C_voov(c, m, k, e + nua) * t3b(a, b, e + nua, i, j, m)
                              end if
                          end do

                          ! diagram 12: A(ij)A(ab) I2B(acie) * t2b(bejk)
                          res12 = res12 + H2B_vvov(a, c, i, e + nua) * t2b(b, e + nua, j, k)
                          res12 = res12 - H2B_vvov(a, c, j, e + nua) * t2b(b, e + nua, i, k)
                          res12 = res12 - H2B_vvov(b, c, i, e + nua) * t2b(a, e + nua, j, k)
                          res12 = res12 + H2B_vvov(b, c, j, e + nua) * t2b(a, e + nua, i, k)

                      end do


                      denom = fA_oo(i, i) + fA_oo(j, j) + fB_oo(k, k) - fA_vv(a, a) - fA_vv(b, b) - fB_vv(c, c)

                      val = X3B(a, b, c, i, j, k) - X3B(b, a, c, i, j, k) - X3B(a, b, c, j, i, k) + X3B(b, a, c, j, i, k)

                       val = val +&
                          res1 + res2 + res3 + res4 + res5 + res6 + res7 + res8 + res9 + res10 + res11 + res12 + res13

                      val = val/(denom - shift)

                      t3b_new(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                      t3b_new(b, a, c, i, j, k) = -t3b_new(a, b, c, i, j, k)
                      t3b_new(a, b, c, j, i, k) = -t3b_new(a, b, c, i, j, k)
                      t3b_new(b, a, c, j, i, k) = t3b_new(a, b, c, i, j, k)

                      resid(a, b, c, i, j, k) = val
                      resid(b, a, c, i, j, k) = -val
                      resid(a, b, c, j, i, k) = -val
                      resid(b, a, c, j, i, k) = val

                  end do; end do; end do;
                  end do; end do; end do;

              end subroutine update_t3b_p_opt2

              subroutine update_t3c_p_opt2(t3c_new, resid,&
                                           X3C,&
                                           t2b, t2c, t3b, t3c, t3d,&
                                           pspace_aab, pspace_abb, pspace_bbb,&
                                           H1A_vv, H1B_vv,&
                                           H2A_voov,&
                                           H2B_oovv, H2B_vvov, H2B_vvvo, H2B_voov, H2B_vovo, H2B_ovov, H2B_ovvo, H2B_vvvv,&
                                           H2C_oovv, H2C_vvov, H2C_voov, H2C_vvvv,&
                                           fA_oo, fA_vv, fB_oo, fB_vv,&
                                           shift,&
                                           noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: pspace_aab(nua, nua, nub, noa, noa, nob),&
                                         pspace_abb(nua, nub, nub, noa, nob, nob),&
                                         pspace_bbb(nub, nub, nub, nob, nob, nob)
                  real(8), intent(in) :: t2b(1:nua,1:nub,1:noa,1:nob),&
                                         t2c(1:nub,1:nub,1:nob,1:nob),&
                                         t3b(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob),&
                                         t3c(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob),&
                                         t3d(1:nub,1:nub,1:nub,1:nob,1:nob,1:nob),&
                                         H1A_vv(1:nua,1:nua),&
                                         H1B_vv(1:nub,1:nub),&
                                         H2A_voov(1:nua,1:noa,1:noa,1:nua),&
                                         H2B_oovv(1:noa,1:nob,1:nua,1:nub),&
                                         H2B_vvov(1:nua,1:nub,1:noa,1:nub),&
                                         H2B_vvvo(1:nua,1:nub,1:nua,1:nob),&
                                         H2B_voov(1:nua,1:nob,1:noa,1:nub),&
                                         H2B_vovo(1:nua,1:nob,1:nua,1:nob),&
                                         H2B_ovov(1:noa,1:nub,1:noa,1:nub),&
                                         H2B_ovvo(1:noa,1:nub,1:nua,1:nob),&
                                         H2C_oovv(1:nob,1:nob,1:nub,1:nub),&
                                         H2C_vvov(1:nub,1:nub,1:nob,1:nub),&
                                         H2B_vvvv(1:nub,1:nub,1:nub,1:nub),&
                                         H2C_voov(1:nub,1:nob,1:nob,1:nub),&
                                         H2C_vvvv(1:nub,1:nub,1:nub,1:nub),&
                                         fA_oo(1:noa,1:noa), fA_vv(1:nua,1:nua),&
                                         fB_oo(1:nob,1:nob), fB_vv(1:nub,1:nub),&
                                         X3C(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob),&
                                         shift

                  real(8), intent(out) :: t3c_new(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob),&
                                          resid(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob)
                  integer :: i, j, k, a, b, c, m, n, e, f
                  real(8) :: denom, val,&
                             res1, res2, res3, res4, res5, res6, res7, res8, res9, res10, res11, res12, res13

                  resid = 0.0d0
                  t3c_new = 0.0d0

                  do i = 1, noa; do j = 1, nob; do k = j + 1, nob;
                  do a = 1, nua; do b = 1, nub; do c = b + 1, nub;

                      if (pspace_abb(a, b, c, i, j, k) /= 1) cycle

                      res1 = 0.0d0
                      res2 = 0.0d0
                      res3 = 0.0d0
                      res4 = 0.0d0
                      res5 = 0.0d0
                      res6 = 0.0d0
                      res7 = 0.0d0
                      res8 = 0.0d0
                      res9 = 0.0d0
                      res10 = 0.0d0
                      res11 = 0.0d0
                      res12 = 0.0d0
                      res13 = 0.0d0

                      do e = 1, nua

                          ! diagram 1: h1A(ae) * t3c(ebcijk)
                          if (pspace_abb(e, b, c, i, j, k) == 1) then
                            res1 = res1 + H1A_vv(a, e) * t3c(e, b, c, i, j, k)
                          end if

                          ! diagram 2: A(be) h1B(be) * t3c(aecijk)
                          if (pspace_abb(a, e, c, i, j, k) == 1) then
                            res2 = res2 + H1B_vv(b, e) * t3c(a, e, c, i, j, k)
                          end if
                          if (pspace_abb(a, e, b, i, j, k) == 1) then
                            res2 = res2 - H1B_vv(c, e) * t3c(a, e, b, i, j, k)
                          end if

                          ! diagram 3: 0.5 * h2C(bcef) * t3c(aefijk)
                          ! diagram 4: A(bc) h2C(abef) * t3c(efcijk)
                          do f = 1, nua
                              if (pspace_abb(a, e, f, i, j, k) == 1) then
                                res3 = res3 + 0.5d0 * H2C_vvvv(b, c, e, f) * t3c(a, e, f, i, j, k)
                              end if

                              if (pspace_abb(e, f, c, i, j, k) == 1) then
                                res4 = res4 + H2B_vvvv(a, b, e, f) * t3c(e, f, c, i, j, k)
                              end if
                              if (pspace_abb(e, f, b, i, j, k) == 1) then
                                res4 = res4 - H2B_vvvv(a, c, e, f) * t3c(e, f, b, i, j, k)
                              end if
                          end do
                          do f = 1, nub - nua
                              if (pspace_abb(e, f + nua, c, i, j, k) == 1) then
                                res4 = res4 + H2B_vvvv(a, b, e, f + nua) * t3c(e, f + nua, c, i, j, k)
                              end if
                              if (pspace_abb(e, f + nua, b, i, j, k) == 1) then
                                res4 = res4 - H2B_vvvv(a, c, e, f + nua) * t3c(e, f + nua, b, i, j, k)
                              end if
                          end do

                          ! diagram 5: h2A(amie) * t3c(ebcmjk)
                          ! diagram 7: A(jk)A(bc) h2B(mbej) * t3b(aecimk)
                          ! diagram 9: -A(bc) h2B(mbie) * t3c(aecmjk)
                          do m = 1, noa
                              if (pspace_abb(e, b, c, m, j, k) == 1) then
                                res5 = res5 + H2A_voov(a, m, i, e) * t3c(e, b, c, m, j, k)
                              end if

                              if (pspace_aab(a, e, c, i, m, k) == 1) then
                                res7 = res7 + H2B_ovvo(m, b, e, j) * t3b(a, e, c, i, m, k)
                              end if
                              if (pspace_aab(a, e, b, i, m, k) == 1) then
                                res7 = res7 - H2B_ovvo(m, c, e, j) * t3b(a, e, b, i, m, k)
                              end if
                              if (pspace_aab(a, e, c, i, m, j) == 1) then
                                res7 = res7 - H2B_ovvo(m, b, e, k) * t3b(a, e, c, i, m, j)
                              end if
                              if (pspace_aab(a, e, b, i, m, j) == 1) then
                                res7 = res7 + H2B_ovvo(m, c, e, k) * t3b(a, e, b, i, m, j)
                              end if

                              if (pspace_abb(a, e, c, m, j, k) == 1) then
                                res9 = res9 - H2B_ovov(m, b, i, e) * t3c(a, e, c, m, j, k)
                              end if
                              if (pspace_abb(a, e, b, m, j, k) == 1) then
                                res9 = res9 + H2B_ovov(m, c, i, e) * t3c(a, e, b, m, j, k)
                              end if
                          end do

                          ! diagram 6: h2B(amie) * t3d(ebcmjk)
                          ! diagram 8: A(jk)A(bc) h2C(bmje) * t3c(aecimk)
                          ! diagram 10: -A(jk) h2B(amej) * t3c(ebcimk)
                          do m = 1, nob
                              if (pspace_bbb(e, b, c, m, j, k) == 1) then
                                res6 = res6 + H2B_voov(a, m, i, e) * t3d(e, b, c, m, j, k)
                              end if

                              if (pspace_abb(a, e, c, i, m, k) == 1) then
                                res8 = res8 + H2C_voov(b, m, j, e) * t3c(a, e, c, i, m, k)
                              end if
                              if (pspace_abb(a, e, b, i, m, k) == 1) then
                                res8 = res8 - H2C_voov(c, m, j, e) * t3c(a, e, b, i, m, k)
                              end if
                              if (pspace_abb(a, e, c, i, m, j) == 1) then
                                res8 = res8 - H2C_voov(b, m, k, e) * t3c(a, e, c, i, m, j)
                              end if
                              if (pspace_abb(a, e, b, i, m, j) == 1) then
                                res8 = res8 + H2C_voov(c, m, k, e) * t3c(a, e, b, i, m, j)
                              end if

                              if (pspace_abb(e, b, c, i, m, k) == 1) then
                                res10 = res10 - H2B_vovo(a, m, e, j) * t3c(e, b, c, i, m, k)
                              end if
                              if (pspace_abb(e, b, c, i, m, j) == 1) then
                                res10 = res10 + H2B_vovo(a, m, e, k) * t3c(e, b, c, i, m, j)
                              end if
                          end do

                          ! diagram 11: A(bc) h2B(abie) * t2c(ecjk)
                          res11 = res11 + H2B_vvov(a, b, i, e) * t2c(e, c, j, k)
                          res11 = res11 - H2B_vvov(a, c, i, e) * t2c(e, b, j, k)
                          ! diagram 12: A(jk) h2C(cbke) * t2b(aeij)
                          res12 = res12 + H2C_vvov(c, b, k, e) * t2b(a, e, i, j)
                          res12 = res12 - H2C_vvov(c, b, j, e) * t2b(a, e, i, k)
                          ! diagram 13: A(jk)A(bc) h2B(abej) * t2b(ecik)
                          res13 = res13 + H2B_vvvo(a, b, e, j) * t2b(e, c, i, k)
                          res13 = res13 - H2B_vvvo(a, b, e, k) * t2b(e, c, i, j)
                          res13 = res13 - H2B_vvvo(a, c, e, j) * t2b(e, b, i, k)
                          res13 = res13 + H2B_vvvo(a, c, e, k) * t2b(e, b, i, j)

                      end do

                      do e = 1, nub - nua

                          ! diagram 2: A(be) h1B(be) * t3c(aecijk)
                          if (pspace_abb(a, e + nua, c, i, j, k) == 1) then
                            res2 = res2 + H1B_vv(b, e + nua) * t3c(a, e + nua, c, i, j, k)
                          end if
                          if (pspace_abb(a, e + nua, b, i, j, k) == 1) then
                            res2 = res2 - H1B_vv(c, e + nua) * t3c(a, e + nua, b, i, j, k)
                          end if

                          ! diagram 3: 0.5 * h2C(bcef) * t3c(aefijk)
                          do f = 1, nua - nub
                              if (pspace_abb(a, e + nua, f + nua, i, j, k) == 1) then
                                res3 = res3 + 0.5d0 * H2C_vvvv(b, c, e + nua, f + nua) * t3c(a, e + nua, f + nua, i, j, k)
                              end if
                          end do

                          ! diagram 9: -A(bc) h2B(mbie) * t3c(aecmjk)
                          do m = 1, noa
                              if (pspace_abb(a, e + nua, c, m, j, k) == 1) then
                                res9 = res9 - H2B_ovov(m, b, i, e + nua) * t3c(a, e + nua, c, m, j, k)
                              end if
                              if (pspace_abb(a, e + nua, b, m, j, k) == 1) then
                                res9 = res9 + H2B_ovov(m, c, i, e + nua) * t3c(a, e + nua, b, m, j, k)
                              end if
                          end do

                          ! diagram 6: h2B(amie) * t3d(ebcmjk)
                          ! diagram 8: A(jk)A(bc) h2C(bmje) * t3c(aecimk)
                          do m = 1, nob
                              if (pspace_bbb(e + nua, b, c, m, j, k) == 1) then
                                res6 = res6 + H2B_voov(a, m, i, e + nua) * t3d(e + nua, b, c, m, j, k)
                              end if

                              if (pspace_abb(a, e + nua, c, i, m, k) == 1) then
                                res8 = res8 + H2C_voov(b, m, j, e + nua) * t3c(a, e + nua, c, i, m, k)
                              end if
                              if (pspace_abb(a, e + nua, b, i, m, k) == 1) then
                                res8 = res8 - H2C_voov(c, m, j, e + nua) * t3c(a, e + nua, b, i, m, k)
                              end if
                              if (pspace_abb(a, e + nua, c, i, m, j) == 1) then
                                res8 = res8 - H2C_voov(b, m, k, e + nua) * t3c(a, e + nua, c, i, m, j)
                              end if
                              if (pspace_abb(a, e + nua, b, i, m, j) == 1) then
                                res8 = res8 + H2C_voov(c, m, k, e + nua) * t3c(a, e + nua, b, i, m, j)
                              end if
                          end do

                          ! diagram 11: A(bc) h2B(abie) * t2c(ecjk)
                          res11 = res11 + H2B_vvov(a, b, i, e + nua) * t2c(e + nua, c, j, k)
                          res11 = res11 - H2B_vvov(a, c, i, e + nua) * t2c(e + nua, b, j, k)
                          ! diagram 12: A(jk) h2C(cbke) * t2b(aeij)
                          res12 = res12 + H2C_vvov(c, b, k, e + nua) * t2b(a, e + nua, i, j)
                          res12 = res12 - H2C_vvov(c, b, j, e + nua) * t2b(a, e + nua, i, k)

                      end do

                      denom = fA_oo(i, i) + fB_oo(j, j) + fB_oo(k, k) - fA_vv(a, a) - fB_vv(b, b) - fB_vv(c, c)

                      val = X3C(a, b, c, i, j, k) - X3C(a, c, b, i, j, k) - X3C(a, b, c, i, k, j) + X3C(a, c, b, i, k, j)
                      val = val&
                        + res1 + res2 + res3 + res4 + res5 + res6 + res7 + res8 + res9 + res10 + res11 + res12 + res13
                      val = val/(denom - shift)

                      t3c_new(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                      t3c_new(a, c, b, i, j, k) = -t3c_new(a, b, c, i, j, k)
                      t3c_new(a, b, c, i, k, j) = -t3c_new(a, b, c, i, j, k)
                      t3c_new(a, c, b, i, k, j) = t3c_new(a, b, c, i, j, k)

                      resid(a, b, c, i, j, k) = val
                      resid(a, c, b, i, j, k) = -val
                      resid(a, b, c, i, k, j) = -val
                      resid(a, c, b, i, k, j) = val

                  end do; end do; end do;
                  end do; end do; end do;

              end subroutine update_t3c_p_opt2

              subroutine update_t3d_p_opt2(t3d_new, resid,&
                                           X3D,&
                                           t2c, t3c, t3d,&
                                           pspace_abb, pspace_bbb,&
                                           H1B_vv,&
                                           H2B_oovv, H2B_ovvo,&
                                           H2C_oovv, H2C_vvov, H2C_voov, H2C_vvvv,&
                                           fB_oo, fB_vv,&
                                           shift,&
                                           noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  real(kind=8), intent(in) :: t2c(nub, nub, nob, nob),&
                                              t3c(nua, nub, nub, noa, nob, nob),&
                                              t3d(nub, nub, nub, nob, nob, nob),&
                                              H1B_vv(nub, nub),&
                                              H2B_oovv(noa, nob, nua, nub),&
                                              H2B_ovvo(noa, nub, nua, nob),&
                                              H2C_oovv(nob, nob, nub, nub),&
                                              H2C_vvov(nub, nub, nob, nub),&
                                              H2C_voov(nub, nob, nob, nub),&
                                              H2C_vvvv(nub, nub, nub, nub),&
                                              fB_vv(nub, nub), fB_oo(nob, nob),&
                                              shift
                  integer, intent(in) :: pspace_abb(nua, nub, nub, noa, nob, nob),&
                                         pspace_bbb(nub, nub, nub, nob, nob, nob)

                  real(kind=8), intent(in) :: X3D(nub, nub, nub, nob, nob, nob)

                  real(kind=8), intent(out) :: t3d_new(nub, nub, nub, nob, nob, nob),&
                                               resid(nub, nub, nub, nob, nob, nob)

                  real(kind=8) :: val, denom
                  real(kind=8) :: res1, res2, res3, res4, res5
                  integer :: a, b, c, i, j, k, e, f, m, n

                  t3d_new = 0.0d0
                  resid = 0.0d0

                  ! loop over projection determinants in P space
                  do a = 1, nub; do b = a + 1, nub; do c = b + 1, nub;
                  do i = 1, nob; do j = i + 1, nob; do k = j + 1, nob;

                      if (pspace_bbb(a, b, c, i, j, k) /= 1) cycle

                          res1 = 0.0d0
                          res2 = 0.0d0
                          res3 = 0.0d0
                          res4 = 0.0d0
                          res5 = 0.0d0

                          do e = 1, nua
                              ! diagram 1: A(a/bc) h1B(ae) * t3d(ebcijk)
                              if (pspace_bbb(e, b, c, i, j, k) == 1) then
                                res1 = res1 + H1B_vv(a, e) * t3d(e, b, c, i, j, k)
                              end if
                              if (pspace_bbb(e, a, c, i, j, k) == 1) then
                                res1 = res1 - H1B_vv(b, e) * t3d(e, a, c, i, j, k)
                              end if
                              if (pspace_bbb(e, b, a, i, j, k) == 1) then
                                res1 = res1 - H1B_vv(c, e) * t3d(e, b, a, i, j, k)
                              end if

                              ! diagram 2: 1/2 A(c/ab) h2C(abef) * t3d(efcijk)
                              do f = e + 1, nua
                                  if (pspace_bbb(e, f, c, i, j, k) == 1) then
                                    res2 = res2 + H2C_vvvv(a, b, e, f) * t3d(e, f, c, i, j, k)
                                  end if
                                  if (pspace_bbb(e, f, a, i, j, k) == 1) then
                                    res2 = res2 - H2C_vvvv(c, b, e, f) * t3d(e, f, a, i, j, k)
                                  end if
                                  if (pspace_bbb(e, f, b, i, j, k) == 1) then
                                    res2 = res2 - H2C_vvvv(a, c, e, f) * t3d(e, f, b, i, j, k)
                                  end if
                              end do

                              ! diagram 3: A(i/jk)A(a/bc) h2B(maei) * t3c(ebcmjk)
                              do m = 1, noa
                                  if (pspace_abb(e, b, c, m, j, k) == 1) then
                                    res3 = res3 + H2B_ovvo(m, a, e, i) * t3c(e, b, c, m, j, k)
                                  end if
                                  if (pspace_abb(e, b, c, m, i, k) == 1) then
                                    res3 = res3 - H2B_ovvo(m, a, e, j) * t3c(e, b, c, m, i, k)
                                  end if
                                  if (pspace_abb(e, b, c, m, j, i) == 1) then
                                    res3 = res3 - H2B_ovvo(m, a, e, k) * t3c(e, b, c, m, j, i)
                                  end if
                                  if (pspace_abb(e, a, c, m, j, k) == 1) then
                                    res3 = res3 - H2B_ovvo(m, b, e, i) * t3c(e, a, c, m, j, k)
                                  end if
                                  if (pspace_abb(e, a, c, m, i, k) == 1) then
                                    res3 = res3 + H2B_ovvo(m, b, e, j) * t3c(e, a, c, m, i, k)
                                  end if
                                  if (pspace_abb(e, a, c, m, j, i) == 1) then
                                    res3 = res3 + H2B_ovvo(m, b, e, k) * t3c(e, a, c, m, j, i)
                                  end if
                                  if (pspace_abb(e, b, a, m, j, k) == 1) then
                                    res3 = res3 - H2B_ovvo(m, c, e, i) * t3c(e, b, a, m, j, k)
                                  end if
                                  if (pspace_abb(e, b, a, m, i, k) == 1) then
                                    res3 = res3 + H2B_ovvo(m, c, e, j) * t3c(e, b, a, m, i, k)
                                  end if
                                  if (pspace_abb(e, b, a, m, j, i) == 1) then
                                    res3 = res3 + H2B_ovvo(m, c, e, k) * t3c(e, b, a, m, j, i)
                                  end if
                              end do

                              ! diagram 4: A(i/jk)A(a/bc) h2C(amie) * t3d(ebcmjk)
                              do m = 1, nob
                                  if (pspace_bbb(e, b, c, m, j, k) == 1) then
                                    res4 = res4 + H2C_voov(a, m, i, e) * t3d(e, b, c, m, j, k)
                                  end if
                                  if (pspace_bbb(e, b, c, m, i, k) == 1) then
                                    res4 = res4 - H2C_voov(a, m, j, e) * t3d(e, b, c, m, i, k)
                                  end if
                                  if (pspace_bbb(e, b, c, m, j, i) == 1) then
                                    res4 = res4 - H2C_voov(a, m, k, e) * t3d(e, b, c, m, j, i)
                                  end if
                                  if (pspace_bbb(e, a, c, m, j, k) == 1) then
                                    res4 = res4 - H2C_voov(b, m, i, e) * t3d(e, a, c, m, j, k)
                                  end if
                                  if (pspace_bbb(e, a, c, m, i, k) == 1) then
                                    res4 = res4 + H2C_voov(b, m, j, e) * t3d(e, a, c, m, i, k)
                                  end if
                                  if (pspace_bbb(e, a, c, m, j, i) == 1) then
                                    res4 = res4 + H2C_voov(b, m, k, e) * t3d(e, a, c, m, j, i)
                                  end if
                                  if (pspace_bbb(e, b, a, m, j, k) == 1) then
                                    res4 = res4 - H2C_voov(c, m, i, e) * t3d(e, b, a, m, j, k)
                                  end if
                                  if (pspace_bbb(e, b, a, m, i, k) == 1) then
                                    res4 = res4 + H2C_voov(c, m, j, e) * t3d(e, b, a, m, i, k)
                                  end if
                                  if (pspace_bbb(e, b, a, m, j, i) == 1) then
                                    res4 = res4 + H2C_voov(c, m, k, e) * t3d(e, b, a, m, j, i)
                                  end if
                              end do

                              ! diagram 5: A(i/jk)A(c/ab) h2C(abie) * t2c(ecjk)
                              res5 = res5 + H2C_vvov(a, b, i, e) * t2c(e, c, j, k)
                              res5 = res5 - H2C_vvov(c, b, i, e) * t2c(e, a, j, k)
                              res5 = res5 - H2C_vvov(a, c, i, e) * t2c(e, b, j, k)
                              res5 = res5 - H2C_vvov(a, b, j, e) * t2c(e, c, i, k)
                              res5 = res5 + H2C_vvov(c, b, j, e) * t2c(e, a, i, k)
                              res5 = res5 + H2C_vvov(a, c, j, e) * t2c(e, b, i, k)
                              res5 = res5 - H2C_vvov(a, b, k, e) * t2c(e, c, j, i)
                              res5 = res5 + H2C_vvov(c, b, k, e) * t2c(e, a, j, i)
                              res5 = res5 + H2C_vvov(a, c, k, e) * t2c(e, b, j, i)

                          end do

                          do e = 1, nub - nua
                              ! diagram 1: A(a/bc) h1B(ae) * t3d(ebcijk)
                              if (pspace_bbb(e + nua, b, c, i, j, k) == 1) then
                                res1 = res1 + H1B_vv(a, e + nua) * t3d(e + nua, b, c, i, j, k)
                              end if
                              if (pspace_bbb(e + nua, a, c, i, j, k) == 1) then
                                res1 = res1 - H1B_vv(b, e + nua) * t3d(e + nua, a, c, i, j, k)
                              end if
                              if (pspace_bbb(e + nua, b, a, i, j, k) == 1) then
                                res1 = res1 - H1B_vv(c, e + nua) * t3d(e + nua, b, a, i, j, k)
                              end if

                              ! diagram 2: 1/2 A(c/ab) h2C(abef) * t3d(efcijk)
                              do f = e + 1, nub - nua
                                  if (pspace_bbb(e + nua, f + nua, c, i, j, k) == 1) then
                                    res2 = res2 + H2C_vvvv(a, b, e + nua, f + nua) * t3d(e + nua, f + nua, c, i, j, k)
                                  end if
                                  if (pspace_bbb(e + nua, f + nua, a, i, j, k) == 1) then
                                    res2 = res2 - H2C_vvvv(c, b, e + nua, f + nua) * t3d(e + nua, f + nua, a, i, j, k)
                                  end if
                                  if (pspace_bbb(e + nua, f + nua, b, i, j, k) == 1) then
                                    res2 = res2 - H2C_vvvv(a, c, e + nua, f + nua) * t3d(e + nua, f + nua, b, i, j, k)
                                  end if
                              end do

                              ! diagram 4: A(i/jk)A(a/bc) h2C(amie) * t3d(ebcmjk)
                              do m = 1, nob
                                  if (pspace_bbb(e + nua, b, c, m, j, k) == 1) then
                                    res4 = res4 + H2C_voov(a, m, i, e + nua) * t3d(e + nua, b, c, m, j, k)
                                  end if
                                  if (pspace_bbb(e + nua, b, c, m, i, k) == 1) then
                                    res4 = res4 - H2C_voov(a, m, j, e + nua) * t3d(e + nua, b, c, m, i, k)
                                  end if
                                  if (pspace_bbb(e + nua, b, c, m, j, i) == 1) then
                                    res4 = res4 - H2C_voov(a, m, k, e + nua) * t3d(e + nua, b, c, m, j, i)
                                  end if
                                  if (pspace_bbb(e + nua, a, c, m, j, k) == 1) then
                                    res4 = res4 - H2C_voov(b, m, i, e + nua) * t3d(e + nua, a, c, m, j, k)
                                  end if
                                  if (pspace_bbb(e + nua, a, c, m, i, k) == 1) then
                                    res4 = res4 + H2C_voov(b, m, j, e + nua) * t3d(e + nua, a, c, m, i, k)
                                  end if
                                  if (pspace_bbb(e + nua, a, c, m, j, i) == 1) then
                                    res4 = res4 + H2C_voov(b, m, k, e + nua) * t3d(e + nua, a, c, m, j, i)
                                  end if
                                  if (pspace_bbb(e + nua, b, a, m, j, k) == 1) then
                                    res4 = res4 - H2C_voov(c, m, i, e + nua) * t3d(e + nua, b, a, m, j, k)
                                  end if
                                  if (pspace_bbb(e + nua, b, a, m, i, k) == 1) then
                                    res4 = res4 + H2C_voov(c, m, j, e + nua) * t3d(e + nua, b, a, m, i, k)
                                  end if
                                  if (pspace_bbb(e + nua, b, a, m, j, i) == 1) then
                                    res4 = res4 + H2C_voov(c, m, k, e + nua) * t3d(e + nua, b, a, m, j, i)
                                  end if
                              end do

                              ! diagram 5: A(i/jk)A(c/ab) h2C(abie) * t2c(ecjk)
                              res5 = res5 + H2C_vvov(a, b, i, e + nua) * t2c(e + nua, c, j, k)
                              res5 = res5 - H2C_vvov(c, b, i, e + nua) * t2c(e + nua, a, j, k)
                              res5 = res5 - H2C_vvov(a, c, i, e + nua) * t2c(e + nua, b, j, k)
                              res5 = res5 - H2C_vvov(a, b, j, e + nua) * t2c(e + nua, c, i, k)
                              res5 = res5 + H2C_vvov(c, b, j, e + nua) * t2c(e + nua, a, i, k)
                              res5 = res5 + H2C_vvov(a, c, j, e + nua) * t2c(e + nua, b, i, k)
                              res5 = res5 - H2C_vvov(a, b, k, e + nua) * t2c(e + nua, c, j, i)
                              res5 = res5 + H2C_vvov(c, b, k, e + nua) * t2c(e + nua, a, j, i)
                              res5 = res5 + H2C_vvov(a, c, k, e + nua) * t2c(e + nua, b, j, i)

                          end do

                          denom = fB_oo(I, I) + fB_oo(J, J) + fB_oo(K, K) - fB_vv(A, A) - fB_vv(B, B) - fB_vv(C, C)

                          val = X3D(a,b,c,i,j,k)&
                                  -X3D(b,a,c,i,j,k)&
                                  -X3D(a,c,b,i,j,k)&
                                  +X3D(b,c,a,i,j,k)&
                                  -X3D(c,b,a,i,j,k)&
                                  +X3D(c,a,b,i,j,k)&
                                  -X3D(a,b,c,j,i,k)&
                                  +X3D(b,a,c,j,i,k)&
                                  +X3D(a,c,b,j,i,k)&
                                  -X3D(b,c,a,j,i,k)&
                                  +X3D(c,b,a,j,i,k)&
                                  -X3D(c,a,b,j,i,k)&
                                  -X3D(a,b,c,i,k,j)&
                                  +X3D(b,a,c,i,k,j)&
                                  +X3D(a,c,b,i,k,j)&
                                  -X3D(b,c,a,i,k,j)&
                                  +X3D(c,b,a,i,k,j)&
                                  -X3D(c,a,b,i,k,j)&
                                  -X3D(a,b,c,k,j,i)&
                                  +X3D(b,a,c,k,j,i)&
                                  +X3D(a,c,b,k,j,i)&
                                  -X3D(b,c,a,k,j,i)&
                                  +X3D(c,b,a,k,j,i)&
                                  -X3D(c,a,b,k,j,i)&
                                  +X3D(a,b,c,j,k,i)&
                                  -X3D(b,a,c,j,k,i)&
                                  -X3D(a,c,b,j,k,i)&
                                  +X3D(b,c,a,j,k,i)&
                                  -X3D(c,b,a,j,k,i)&
                                  +X3D(c,a,b,j,k,i)&
                                  +X3D(a,b,c,k,i,j)&
                                  -X3D(b,a,c,k,i,j)&
                                  -X3D(a,c,b,k,i,j)&
                                  +X3D(b,c,a,k,i,j)&
                                  -X3D(c,b,a,k,i,j)&
                                  +X3D(c,a,b,k,i,j)

                          val = val + res1 + res2 + res3 + res4 + res5
                          val = val/(denom - shift)

                          t3d_new(A,B,C,I,J,K) = t3d(A,B,C,I,J,K) + val
                          t3d_new(A,B,C,K,I,J) = t3d_new(A,B,C,I,J,K)
                          t3d_new(A,B,C,J,K,I) = t3d_new(A,B,C,I,J,K)
                          t3d_new(A,B,C,I,K,J) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(A,B,C,J,I,K) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(A,B,C,K,J,I) = -t3d_new(A,B,C,I,J,K)

                          t3d_new(B,A,C,I,J,K) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(B,A,C,K,I,J) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(B,A,C,J,K,I) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(B,A,C,I,K,J) = t3d_new(A,B,C,I,J,K)
                          t3d_new(B,A,C,J,I,K) = t3d_new(A,B,C,I,J,K)
                          t3d_new(B,A,C,K,J,I) = t3d_new(A,B,C,I,J,K)

                          t3d_new(A,C,B,I,J,K) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(A,C,B,K,I,J) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(A,C,B,J,K,I) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(A,C,B,I,K,J) = t3d_new(A,B,C,I,J,K)
                          t3d_new(A,C,B,J,I,K) = t3d_new(A,B,C,I,J,K)
                          t3d_new(A,C,B,K,J,I) = t3d_new(A,B,C,I,J,K)

                          t3d_new(C,B,A,I,J,K) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(C,B,A,K,I,J) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(C,B,A,J,K,I) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(C,B,A,I,K,J) = t3d_new(A,B,C,I,J,K)
                          t3d_new(C,B,A,J,I,K) = t3d_new(A,B,C,I,J,K)
                          t3d_new(C,B,A,K,J,I) = t3d_new(A,B,C,I,J,K)

                          t3d_new(B,C,A,I,J,K) = t3d_new(A,B,C,I,J,K)
                          t3d_new(B,C,A,K,I,J) = t3d_new(A,B,C,I,J,K)
                          t3d_new(B,C,A,J,K,I) = t3d_new(A,B,C,I,J,K)
                          t3d_new(B,C,A,I,K,J) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(B,C,A,J,I,K) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(B,C,A,K,J,I) = -t3d_new(A,B,C,I,J,K)

                          t3d_new(C,A,B,I,J,K) = t3d_new(A,B,C,I,J,K)
                          t3d_new(C,A,B,K,I,J) = t3d_new(A,B,C,I,J,K)
                          t3d_new(C,A,B,J,K,I) = t3d_new(A,B,C,I,J,K)
                          t3d_new(C,A,B,I,K,J) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(C,A,B,J,I,K) = -t3d_new(A,B,C,I,J,K)
                          t3d_new(C,A,B,K,J,I) = -t3d_new(A,B,C,I,J,K)

                          resid(A,B,C,I,J,K) = val
                          resid(A,B,C,K,I,J) = val
                          resid(A,B,C,J,K,I) = val
                          resid(A,B,C,I,K,J) = -val
                          resid(A,B,C,J,I,K) = -val
                          resid(A,B,C,K,J,I) = -val
                          resid(B,C,A,I,J,K) = val
                          resid(B,C,A,K,I,J) = val
                          resid(B,C,A,J,K,I) = val
                          resid(B,C,A,I,K,J) = -val
                          resid(B,C,A,J,I,K) = -val
                          resid(B,C,A,K,J,I) = -val
                          resid(C,A,B,I,J,K) = val
                          resid(C,A,B,K,I,J) = val
                          resid(C,A,B,J,K,I) = val
                          resid(C,A,B,I,K,J) = -val
                          resid(C,A,B,J,I,K) = -val
                          resid(C,A,B,K,J,I) = -val
                          resid(A,C,B,I,J,K) = -val
                          resid(A,C,B,K,I,J) = -val
                          resid(A,C,B,J,K,I) = -val
                          resid(A,C,B,I,K,J) = val
                          resid(A,C,B,J,I,K) = val
                          resid(A,C,B,K,J,I) = val
                          resid(B,A,C,I,J,K) = -val
                          resid(B,A,C,K,I,J) = -val
                          resid(B,A,C,J,K,I) = -val
                          resid(B,A,C,I,K,J) = val
                          resid(B,A,C,J,I,K) = val
                          resid(B,A,C,K,J,I) = val
                          resid(C,B,A,I,J,K) = -val
                          resid(C,B,A,K,I,J) = -val
                          resid(C,B,A,J,K,I) = -val
                          resid(C,B,A,I,K,J) = val
                          resid(C,B,A,J,I,K) = val
                          resid(C,B,A,K,J,I) = val

                  end do; end do; end do;
                  end do; end do; end do;

              end subroutine update_t3d_p_opt2

end module ccp_opt_loops