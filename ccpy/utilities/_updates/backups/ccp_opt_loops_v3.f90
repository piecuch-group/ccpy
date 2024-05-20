module ccp_opt_loops_v2

!!!!      USE OMP_LIB
!!!!      USE MKL_SERVICE
      implicit none

      ! We have 2 options:
      ! (1) For every lookup of pspace(a, b, c, i, j, k), we perform a binary search on the 1D array
      !     index(..) (this is sorted) of indices that uniquely map (a, b, c, i, j, k) to a number.
      !
      ! (2) For every lookup of pspace(a, b, c, i, j, k), we perform an equivalent hash table lookup.

      ! Update: 10/08/22
      ! Intermediates and T3 parts of T1 and T2 updates are computed very inefficiently. Since were are making
      ! "full" (e.g., not P-space limited) objects, we can take advantage of this and impose a cheap linear speedup.
      ! The idea here is that since the output is not sparse, *every* T3 amplitude we have contributes to the expression,
      ! thus, we should simply loop over all of them and put the resulting expression where it belongs in the intermediate.
      !
      ! E.g., I2A(amij) = H2A(amij) + 1/2 * vA(mnef) * t3a(aefijn) + vB(mnef) * t3b(aefijn)
      !       I2A(abie) = H2A(abie) - 1/2 * vA(mnef) * t3a(abfimn) - vB(mnef) * t3b(abfimn)
      !
      ! do idet = 1, n3a_p
      !    a1 = p_coo(idet, 1); b1 = p_coo(idet, 2); c1 = p_coo(idet, 3);
      !    i1 = p_coo(idet, 4); j1 = p_coo(idet, 5); k1 = p_coo(idet, 6);
      !
      !    I2A_vooo(a1, :, i1, j1) = I2A_vooo(a1, :, i1, j1) + 0.5d0 * h2a_oovv(:, k1, b1, c1) * t3a(idet)
      !    I2A_vvov(a1, b1, i1, :) = I2A_vvov(a1, b1, i1, :) - 0.5d0 * h2a_oovv(j1, k1, :, c1) * t3a(idet)
      ! end do
      ! do idet = 1, n3b_p
      !    a1 = p_coo(idet, 1); b1 = p_coo(idet, 2); c1 = p_coo(idet, 3);
      !    i1 = p_coo(idet, 4); j1 = p_coo(idet, 5); k1 = p_coo(idet, 6);
      !
      !    I2A_vooo(a1, :, i1, j1) = I2A_vooo(a1, :, i1, j1) + h2b_oovv(:, k1, b1, c1) * t3b(idet)
      !    I2A_vvov(a1, b1, i1, :) = I2A_vvov(a1, b1, i1, :) - h2b_oovv(j1, k1, :, c1) * t3b(idet)
      ! end do


      contains

               subroutine update_t1a_opt(t1a, resid,&
                                         X1A,&
                                         t3a, t3b, t3c,&
                                         pspace_aaa, pspace_aab, pspace_abb,&
                                         H2A_oovv, H2B_oovv, H2C_oovv,&
                                         fA_oo, fA_vv,&
                                         shift,&
                                         n3a_p, n3b_p, n3c_p,&
                                         noa, nua, nob, nub)

                      integer, intent(in) :: noa, nua, nob, nub, n3a_p, n3b_p, n3c_p
                      integer, intent(in) :: pspace_aaa(1:nua,1:nua,1:nua,1:noa,1:noa,1:noa),&
                                             pspace_aab(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob),&
                                             pspace_abb(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob)
                      real(kind=8), intent(in) :: X1A(1:nua,1:noa),&
                                                  t3a(n3a_p),&
                                                  t3b(n3b_p),&
                                                  t3c(n3c_p),&
                                                  H2A_oovv(1:noa,1:noa,1:nua,1:nua),&
                                                  H2B_oovv(1:noa,1:nob,1:nua,1:nub),&
                                                  H2C_oovv(1:nob,1:nob,1:nub,1:nub),&
                                                  fA_oo(1:noa,1:noa), fA_vv(1:nua,1:nua),&
                                                  shift

                      real(kind=8), intent(inout) :: t1a(1:nua,1:noa)
                      !f2py intent(in,out) :: t1a(0:nua-1,0:noa-1)

                      real(kind=8), intent(out) :: resid(1:nua,1:noa)

                      integer :: i, a, m, n, e, f
                      real(kind=8) :: denom, val, res1, res2, res3
                      real(kind=8) :: HT3(nua, noa)

                      HT3 = 0.0d0

                      ! diagram 1: 1/4 A(a/ef)A(i/mn) h2a(mnef) * t3a(aefimn)
                      do idet = 1, n3a_p
                          a = pspace_aaa(idet, 1)
                          e = pspace_aaa(idet, 2)
                          f = pspace_aaa(idet, 3)
                          i = pspace_aaa(idet, 4)
                          m = pspace_aaa(idet, 5)
                          n = pspace_aaa(idet, 6)
                          HT3(a, i) = HT3(a, i) + 0.25d0 * H2A_oovv(m, n, e, f) * t3a(idet)
                      end do

                      do idet = 1, n3b_p
                          a = pspace_aaa(idet, 1)
                          e = pspace_aaa(idet, 2)
                          f = pspace_aaa(idet, 3)
                          i = pspace_aaa(idet, 4)
                          m = pspace_aaa(idet, 5)
                          n = pspace_aaa(idet, 6)
                          HT3(a, i) = HT3(a, i) + H2B_oovv(m, n, e, f) * t3b(idet)
                      end do

                      do idet = 1, n3c_p
                          a = pspace_aaa(idet, 1)
                          e = pspace_aaa(idet, 2)
                          f = pspace_aaa(idet, 3)
                          i = pspace_aaa(idet, 4)
                          m = pspace_aaa(idet, 5)
                          n = pspace_aaa(idet, 6)
                          HT3(a, i) = HT3(a, i) + 0.25d0 * H2A_oovv(m, n, e, f) * t3c(idet)
                      end do

                      do i = 1, noa
                          do a = 1, nua

!                              res1 = 0.0d0
!                              res2 = 0.0d0
!                              res3 = 0.0d0
!
!                              do e = 1, nua
!                                  do m = 1, noa
!                                      ! diagram 1: 1/4 h2a(mnef) * t3a(aefimn)
!                                      do f = e + 1, nua
!                                          do n = m + 1, noa
!                                              if (pspace_aaa(a, e, f, i, m, n) /= 0) then
!                                                  res1 = res1 + H2A_oovv(m, n, e, f) * t3a(pspace_aaa(a, e, f, i, m, n))
!                                              end if
!                                          end do
!                                      end do
!                                      ! diagram 2: h2b(mnef) * t3b(aefimn)
!                                      do f = 1, nub
!                                          do n = 1, nob
!                                              if (pspace_aab(a, e, f, i, m, n) /= 0) then
!                                                  res2 = res2 + H2B_oovv(m, n, e, f) * t3b(pspace_aab(a, e, f, i, m, n))
!                                              end if
!                                          end do
!                                      end do
!                                  end do
!                              end do
!
!                              ! diagram 3: 1/4 h2c(mnef) * t3c(aefimn)
!                              do e = 1, nub
!                                  do f = e + 1, nub
!                                      do m = 1, nob
!                                          do n = m + 1, nob
!                                              if (pspace_abb(a, e, f, i, m, n) /= 0) then
!                                                  res3 = res3 + H2C_oovv(m, n, e, f) * t3c(pspace_abb(a, e, f, i, m, n))
!                                              end if
!                                          end do
!                                      end do
!                                  end do
!                              end do

                              denom = fA_oo(i, i) - fA_vv(a, a)
!                              val = X1A(a, i) + res1 + res2 + res3
                              val = X1A(a, i) + HT3(a, i)

                              t1a(a, i) = t1a(a, i) + val/(denom - shift)

                              resid(a, i) = val

                          end do
                      end do

              end subroutine update_t1a_opt

              subroutine update_t1b_opt(t1b, resid,&
                                         X1B,&
                                         t3b, t3c, t3d,&
                                         pspace_aab, pspace_abb, pspace_bbb,&
                                         H2A_oovv, H2B_oovv, H2C_oovv,&
                                         fB_oo, fB_vv,&
                                         shift,&
                                         n3b_p, n3c_p, n3d_p,&
                                         noa, nua, nob, nub)

                      integer, intent(in) :: noa, nua, nob, nub, n3b_p, n3c_p, n3d_p
                      integer, intent(in) :: pspace_aab(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob),&
                                             pspace_abb(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob),&
                                             pspace_bbb(1:nub,1:nub,1:nub,1:nob,1:nob,1:nob)
                      real(kind=8), intent(in) :: X1B(1:nub,1:nob),&
                                                  t3b(n3b_p),&
                                                  t3c(n3c_p),&
                                                  t3d(n3d_p),&
                                                  H2A_oovv(1:noa,1:noa,1:nua,1:nua),&
                                                  H2B_oovv(1:noa,1:nob,1:nua,1:nub),&
                                                  H2C_oovv(1:nob,1:nob,1:nub,1:nub),&
                                                  fB_oo(1:nob,1:nob), fB_vv(1:nub,1:nub),&
                                                  shift

                      real(kind=8), intent(inout) :: t1b(1:nub,1:nob)
                      !f2py intent(in,out) :: t1b(0:nub-1,0:nob-1)

                      real(kind=8), intent(out) :: resid(1:nub,1:nob)

                      integer :: i, a, m, n, e, f
                      real(kind=8) :: denom, val, res1, res2, res3

                      do i = 1, nob
                          do a = 1, nub

                              res1 = 0.0d0
                              res2 = 0.0d0
                              res3 = 0.0d0

                              do e = 1, nub
                                  do m = 1, nob
                                      ! diagram 1: 1/4 h2c(mnef) * t3d(aefimn)
                                      do f = e + 1, nub
                                          do n = m + 1, nob
                                              if (pspace_bbb(a, e, f, i, m, n) /= 0) then
                                                  res1 = res1 + H2C_oovv(m, n, e, f) * t3d(pspace_bbb(a, e, f, i, m, n))
                                              end if
                                          end do
                                      end do
                                      ! diagram 2: h2b(nmfe) * t3c(feanmi)
                                      do f = 1, nua
                                          do n = 1, noa
                                              if (pspace_abb(f, e, a, n, m, i) /= 0) then
                                                  res2 = res2 + H2B_oovv(n, m, f, e) * t3c(pspace_abb(f, e, a, n, m, i))
                                              end if
                                          end do
                                      end do
                                  end do
                              end do

                              ! diagram 3: 1/4 h2a(mnef) * t3b(feanmi)
                              do e = 1, nua
                                  do f = e + 1, nua
                                      do m = 1, noa
                                          do n = m + 1, noa
                                              if (pspace_aab(f, e, a, n, m, i) /= 0) then
                                                  res3 = res3 + H2A_oovv(m, n, e, f) * t3b(pspace_aab(f, e, a, n, m, i))
                                              end if
                                          end do
                                      end do
                                  end do
                              end do

                              denom = fB_oo(i, i) - fB_vv(a, a)
                              val = X1B(a, i) + res1 + res2 + res3

                              t1b(a, i) = t1b(a, i) + val/(denom - shift)

                              resid(a, i) = val

                          end do
                      end do

              end subroutine update_t1b_opt


              subroutine update_t2a_opt(t2a, resid,&
                                        X2A,&
                                        t3a, t3b,&
                                        pspace_aaa, pspace_aab,&
                                        H1A_ov, H1B_ov,&
                                        H2A_ooov, H2A_vovv,&
                                        H2B_ooov, H2B_vovv,&
                                        fA_oo, fA_vv,&
                                        shift,&
                                        n3a_p, n3b_p,&
                                        noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub, n3a_p, n3b_p
                  integer, intent(in) :: pspace_aaa(1:nua,1:nua,1:nua,1:noa,1:noa,1:noa),&
                                         pspace_aab(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob)
                  real(kind=8), intent(in) :: X2A(1:nua,1:nua,1:noa,1:noa),&
                                              t3a(n3a_p),&
                                              t3b(n3b_p),&
                                              H1A_ov(1:noa,1:nua), H1B_ov(1:nob,1:nub),&
                                              H2A_ooov(1:noa,1:noa,1:noa,1:nua),&
                                              H2A_vovv(1:nua,1:noa,1:nua,1:nua),&
                                              H2B_ooov(1:noa,1:nob,1:noa,1:nub),&
                                              H2B_vovv(1:nua,1:nob,1:nua,1:nub),&
                                              fA_oo(1:noa,1:noa), fA_vv(1:nua,1:nua),&
                                              shift

                  real(kind=8), intent(inout) :: t2a(1:nua,1:nua,1:noa,1:noa)
                  !f2py intent(in,out) :: t2a(0:nua-1,0:nua-1,0:noa-1,0:noa-1)

                  real(kind=8), intent(out) :: resid(1:nua,1:nua,1:noa,1:noa)

                  integer :: i, j, a, b, m, n, e, f
                  real(kind=8) :: denom, val, res1, res2, res3, res4, res5, res6

                  do i = 1, noa
                      do j = i + 1, noa
                          do a = 1, nua
                              do b = a + 1, nua

                                  res1 = 0.0d0
                                  res2 = 0.0d0
                                  res3 = 0.0d0
                                  res4 = 0.0d0
                                  res5 = 0.0d0
                                  res6 = 0.0d0

                                  do e = 1, nua
                                      ! diagram 1: h1a(me) * t3a(abeijm)
                                      do m = 1, noa
                                          if (pspace_aaa(a, b, e, i, j, m) /= 0) then
                                              res1 = res1 + H1A_ov(m, e) * t3a(pspace_aaa(a, b, e, i, j, m))
                                          end if
                                          ! diagram 3: -1/2 A(ij) h2a(mnie) * t3a(abemjn)
                                          do n = m + 1, noa
                                              if (pspace_aaa(a, b, e, m, j, n) /= 0) then
                                                  res3 = res3 - H2A_ooov(m, n, i, e) * t3a(pspace_aaa(a, b, e, m, j, n))
                                              end if
                                              if (pspace_aaa(a, b, e, m, i, n) /= 0) then
                                                  res3 = res3 + H2A_ooov(m, n, j, e) * t3a(pspace_aaa(a, b, e, m, i, n))
                                              end if
                                          end do
                                          ! diagram 4: -A(ij) h2b(mnie) * t3b(abemjn)
                                          do n = 1, nob
                                              if (pspace_aab(a, b, e, m, j, n) /= 0) then
                                                  res4 = res4 - H2B_ooov(m, n, i, e) * t3b(pspace_aab(a, b, e, m, j, n))
                                              end if
                                              if (pspace_aab(a, b, e, m, i, n) /= 0) then
                                                  res4 = res4 + H2B_ooov(m, n, j, e) * t3b(pspace_aab(a, b, e, m, i, n))
                                              end if
                                          end do
                                      end do

                                      ! diagram 2: h1b(me) * t3b(abeijm)
                                      do m = 1, nob
                                          if (pspace_aab(a, b, e, i, j, m) /= 0) then
                                              res2 = res2 + H1B_ov(m, e) * t3b(pspace_aab(a, b, e, i, j, m))
                                          end if
                                      end do

                                      ! diagram 5: 1/2 A(ab) h2a(anef) * t3a(ebfijn)
                                      do f = e + 1, nua
                                          do n = 1, noa
                                              if (pspace_aaa(e, b, f, i, j, n) /= 0) then
                                                  res5 = res5 + H2A_vovv(a, n, e, f) * t3a(pspace_aaa(e, b, f, i, j, n))
                                              end if
                                              if (pspace_aaa(e, a, f, i, j, n) /= 0) then
                                                  res5 = res5 - H2A_vovv(b, n, e, f) * t3a(pspace_aaa(e, a, f, i, j, n))
                                              end if
                                          end do
                                      end do

                                      ! diagram 6: A(ab) h2b(anef) * t3b(ebfijn)
                                      do f = 1, nub
                                          do n = 1, nob
                                              if (pspace_aab(e, b, f, i, j, n) /= 0) then
                                                  res6 = res6 + H2B_vovv(a, n, e, f) * t3b(pspace_aab(e, b, f, i, j, n))
                                              end if
                                              if (pspace_aab(e, a, f, i, j, n) /= 0) then
                                                  res6 = res6 - H2B_vovv(b, n, e, f) * t3b(pspace_aab(e, a, f, i, j, n))
                                              end if
                                          end do
                                      end do

                                  end do

                                  do e = 1, nub - nua
                                      ! diagram 2: h1b(me) * t3b(abeijm)
                                      do m = 1, nob
                                          if (pspace_aab(a, b, e + nua, i, j, m) /= 0) then
                                              res2 = res2 + H1B_ov(m, e + nua) * t3b(pspace_aab(a, b, e + nua, i, j, m))
                                          end if
                                      end do
                                      ! diagram 4: -A(ij) h2b(mnie) * t3b(abemjn)
                                      do m = 1, noa
                                          do n = 1, nob
                                              if (pspace_aab(a, b, e + nua, m, j, n) /= 0) then
                                                  res4 = res4 - H2B_ooov(m, n, i, e + nua) * t3b(pspace_aab(a, b, e + nua, m, j, n))
                                              end if
                                              if (pspace_aab(a, b, e + nua, m, i, n) /= 0) then
                                                  res4 = res4 + H2B_ooov(m, n, j, e + nua) * t3b(pspace_aab(a, b, e + nua, m, i, n))
                                              end if
                                          end do
                                      end do

                                  end do

                                  denom = fA_oo(i, i) + fA_oo(j, j) - fA_vv(a, a) - fA_vv(b, b)
                                  val = X2A(b, a, j, i) - X2A(a, b, j, i) - X2A(b, a, i, j) + X2A(a, b, i, j)
                                  val = val + res1 + res2 + res3 + res4 + res5 + res6

                                  t2a(b, a, j, i) =  t2a(b, a, j, i) + val/(denom - shift)
                                  t2a(a, b, j, i) = -t2a(b, a, j, i)
                                  t2a(b, a, i, j) = -t2a(b, a, j, i)
                                  t2a(a, b, i, j) =  t2a(b, a, j, i)

                                  resid(b, a, j, i) =  val
                                  resid(a, b, j, i) = -val
                                  resid(b, a, i, j) = -val
                                  resid(a, b, i, j) =  val

                              end do
                          end do
                      end do
                  end do

              end subroutine update_t2a_opt

              subroutine update_t2b_opt(t2b, resid,&
                                        X2B,&
                                        t3b, t3c,&
                                        pspace_aab, pspace_abb,&
                                        H1A_ov, H1B_ov,&
                                        H2A_ooov, H2A_vovv,&
                                        H2B_ooov, H2B_oovo, H2B_vovv, H2B_ovvv,&
                                        H2C_ooov, H2C_vovv,&
                                        fA_oo, fA_vv, fB_oo, fB_vv,&
                                        shift,&
                                        n3b_p, n3c_p,&
                                        noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub, n3b_p, n3c_p
                  integer, intent(in) :: pspace_aab(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob),&
                                         pspace_abb(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob)
                  real(kind=8), intent(in) :: X2B(1:nua,1:nub,1:noa,1:nob),&
                                              t3b(n3b_p),&
                                              t3c(n3c_p),&
                                              H1A_ov(1:noa,1:nua), H1B_ov(1:nob,1:nub),&
                                              H2A_ooov(1:noa,1:noa,1:noa,1:nua),&
                                              H2A_vovv(1:nua,1:noa,1:nua,1:nua),&
                                              H2B_ooov(1:noa,1:nob,1:noa,1:nub),&
                                              H2B_oovo(1:noa,1:nob,1:nua,1:nob),&
                                              H2B_vovv(1:nua,1:nob,1:nua,1:nub),&
                                              H2B_ovvv(1:noa,1:nub,1:nua,1:nub),&
                                              H2C_ooov(1:nob,1:nob,1:nob,1:nub),&
                                              H2C_vovv(1:nub,1:nob,1:nub,1:nub),&
                                              fA_oo(1:noa,1:noa), fA_vv(1:nua,1:nua),&
                                              fB_oo(1:nob,1:nob), fB_vv(1:nub,1:nub),&
                                              shift

                  real(kind=8), intent(inout) :: t2b(1:nua,1:nub,1:noa,1:nob)
                  !f2py intent(in,out) :: t2b(0:nua-1,0:nub-1,0:noa-1,0:nob-1)

                  real(kind=8), intent(out) :: resid(1:nua,1:nub,1:noa,1:nob)

                  integer :: i, j, a, b, m, n, e, f
                  real(kind=8) :: denom, val, res1, res2, res3, res4, res5, res6, res7, res8, res9, res10

                  do i = 1, noa
                      do j = 1, nob
                          do a = 1, nua
                              do b = 1, nub

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

                                  do e = 1, nua
                                      ! diagram 1: h1a(me) * t3b(aebimj)
                                      do m = 1, noa
                                          if (pspace_aab(a, e, b, i, m, j) /= 0) then
                                              res1 = res1 + H1A_ov(m, e) * t3b(pspace_aab(a, e, b, i, m, j))
                                          end if
                                          ! diagram 3: -1/2 h2a(mnie) * t3b(aebmnj)
                                          do n = m + 1, noa
                                              if (pspace_aab(a, e, b, m, n, j) /= 0) then
                                                  res3 = res3 - H2A_ooov(m, n, i, e) * t3b(pspace_aab(a, e, b, m, n, j))
                                              end if
                                          end do
                                          ! diagram 4: -h2b(mnie) * t3c(abemjn)
                                          ! diagram 9: -h2b(mnej) * t3b(aebimn)
                                          do n = 1, nob
                                              if (pspace_abb(a, b, e, m, j, n) /= 0) then
                                                  res4 = res4 - H2B_ooov(m, n, i, e) * t3c(pspace_abb(a, b, e, m, j, n))
                                              end if
                                              if (pspace_aab(a, e, b, i, m, n) /= 0) then
                                                  res9 = res9 - H2B_oovo(m, n, e, j) * t3b(pspace_aab(a, e, b, i, m, n))
                                              end if
                                          end do


                                      end do

                                      ! diagram 2: h1b(me) * t3c(abeijm)
                                      do m = 1, nob
                                          if (pspace_abb(a, b, e, i, j, m) /= 0) then
                                              res2 = res2 + H1B_ov(m, e) * t3c(pspace_abb(a, b, e, i, j, m))
                                          end if
                                          ! diagram 7: -1/2 h2c(mnje) * t3c(aebinm)
                                          do n = m + 1, nob
                                              if (pspace_abb(a, e, b, i, n, m) /= 0) then
                                                  res7 = res7 - H2C_ooov(m, n, j, e) * t3c(pspace_abb(a, e, b, i, n, m))
                                              end if
                                          end do
                                      end do

                                      ! diagram 5: 1/2 h2a(anef) * t3b(efbinj)
                                      do f = e + 1, nua
                                          do n = 1, noa
                                              if (pspace_aab(e, f, b, i, n, j) /= 0) then
                                                  res5 = res5 + H2A_vovv(a, n, e, f) * t3b(pspace_aab(e, f, b, i, n, j))
                                              end if
                                          end do
                                      end do

                                      ! diagram 6: h2b(anef) * t3c(ebfijn)
                                      ! diagram 10: h2b(nbef) * t3b(aefinj)
                                      do f = 1, nub
                                          do n = 1, nob
                                              if (pspace_abb(e, b, f, i, j, n) /= 0) then
                                                  res6 = res6 + H2B_vovv(a, n, e, f) * t3c(pspace_abb(e, b, f, i, j, n))
                                              end if
                                          end do
                                          do n = 1, noa
                                              if (pspace_aab(a, e, f, i, n, j) /= 0) then
                                                  res10 = res10 + H2B_ovvv(n, b, e, f) * t3b(pspace_aab(a, e, f, i, n, j))
                                              end if
                                          end do
                                      end do

                                  end do

                                  do e = 1, nub - nua
                                      ! diagram 2: h1b(me) * t3c(abeijm)
                                      do m = 1, nob
                                          if (pspace_abb(a, b, e + nua, i, j, m) /= 0) then
                                              res2 = res2 + H1B_ov(m, e + nua) * t3c(pspace_abb(a, b, e + nua, i, j, m))
                                          end if
                                          ! diagram 7: -1/2 h2c(mnje) * t3c(aebinm)
                                          do n = m + 1, nob
                                              if (pspace_abb(a, e + nua, b, i, n, m) /= 0) then
                                                  res7 = res7 - H2C_ooov(m, n, j, e + nua) * t3c(pspace_abb(a, e + nua, b, i, n, m))
                                              end if
                                          end do
                                      end do
                                      ! diagram 4: -h2b(mnie) * t3c(abemjn)
                                      do m = 1, noa
                                          do n = 1, nob
                                              if (pspace_abb(a, b, e + nua, m, j, n) /= 0) then
                                                  res4 = res4 - H2B_ooov(m, n, i, e + nua) * t3c(pspace_abb(a, b, e + nua, m, j, n))
                                              end if
                                          end do
                                      end do
                                  end do

                                  ! diagram 8: 1/2 h2c(bnef) * t3c(afeinj)
                                  do e = 1, nub
                                      do f = e + 1, nub
                                          do n = 1, nob
                                              if (pspace_abb(a, f, e, i, n, j) /= 0) then
                                                  res8 = res8 + H2C_vovv(b, n, e, f) * t3c(pspace_abb(a, f, e, i, n, j))
                                              end if
                                          end do
                                      end do
                                  end do

                                  denom = fA_oo(i, i) + fB_oo(j, j) - fA_vv(a, a) - fB_vv(b, b)
                                  val = X2B(a, b, i, j) + res1 + res2 + res3 + res4 + res5 + res6 + res7 + res8 + res9 + res10

                                  t2b(a, b, i, j) =  t2b(a, b, i, j) + val/(denom - shift)

                                  resid(a, b, i, j) =  val

                              end do
                          end do
                      end do
                  end do

              end subroutine update_t2b_opt

              subroutine update_t2c_opt(t2c, resid,&
                                        X2C,&
                                        t3c, t3d,&
                                        pspace_abb, pspace_bbb,&
                                        H1A_ov, H1B_ov,&
                                        H2B_oovo, H2B_ovvv,&
                                        H2C_ooov, H2C_vovv,&
                                        fB_oo, fB_vv,&
                                        shift,&
                                        n3c_p, n3d_p,&
                                        noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub, n3c_p, n3d_p
                  integer, intent(in) :: pspace_abb(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob),&
                                         pspace_bbb(1:nub,1:nub,1:nub,1:nob,1:nob,1:nob)
                  real(kind=8), intent(in) :: X2C(1:nub,1:nub,1:nob,1:nob),&
                                              t3c(n3c_p),&
                                              t3d(n3d_p),&
                                              H1A_ov(1:noa,1:nua), H1B_ov(1:nob,1:nub),&
                                              H2B_oovo(1:noa,1:nob,1:nua,1:nob),&
                                              H2B_ovvv(1:noa,1:nub,1:nua,1:nub),&
                                              H2C_ooov(1:nob,1:nob,1:nob,1:nub),&
                                              H2C_vovv(1:nub,1:nob,1:nub,1:nub),&
                                              fB_oo(1:nob,1:nob), fB_vv(1:nub,1:nub),&
                                              shift

                  real(kind=8), intent(inout) :: t2c(1:nub,1:nub,1:nob,1:nob)
                  !f2py intent(in,out) :: t2c(0:nub-1,0:nub-1,0:nob-1,0:nob-1)

                  real(kind=8), intent(out) :: resid(1:nub,1:nub,1:nob,1:nob)

                  integer :: i, j, a, b, m, n, e, f
                  real(kind=8) :: denom, val, res1, res2, res3, res4, res5, res6

                  do i = 1, nob
                      do j = i + 1, nob
                          do a = 1, nub
                              do b = a + 1, nub

                                  res1 = 0.0d0
                                  res2 = 0.0d0
                                  res3 = 0.0d0
                                  res4 = 0.0d0
                                  res5 = 0.0d0
                                  res6 = 0.0d0

                                  do e = 1, nub
                                      ! diagram 1: h1b(me) * t3d(abeijm)
                                      do m = 1, nob
                                          if (pspace_bbb(a, b, e, i, j, m) /= 0) then
                                              res1 = res1 + H1B_ov(m, e) * t3d(pspace_bbb(a, b, e, i, j, m))
                                          end if
                                          ! diagram 3: -1/2 A(ij) h2c(mnie) * t3d(abemjn)
                                          do n = m + 1, nob
                                              if (pspace_bbb(a, b, e, m, j, n) /= 0) then
                                                  res3 = res3 - H2C_ooov(m, n, i, e) * t3d(pspace_bbb(a, b, e, m, j, n))
                                              end if
                                              if (pspace_bbb(a, b, e, m, i, n) /= 0) then
                                                  res3 = res3 + H2C_ooov(m, n, j, e) * t3d(pspace_bbb(a, b, e, m, i, n))
                                              end if
                                          end do
                                      end do

                                      ! diagram 5: 1/2 A(ab) h2c(anef) * t3d(ebfijn)
                                      do f = e + 1, nub
                                          do n = 1, nob
                                              if (pspace_bbb(e, b, f, i, j, n) /= 0) then
                                                  res5 = res5 + H2C_vovv(a, n, e, f) * t3d(pspace_bbb(e, b, f, i, j, n))
                                              end if
                                              if (pspace_bbb(e, a, f, i, j, n) /= 0) then
                                                  res5 = res5 - H2C_vovv(b, n, e, f) * t3d(pspace_bbb(e, a, f, i, j, n))
                                              end if
                                          end do
                                      end do

                                      ! diagram 6: A(ab) h2b(nafe) * t3c(fbenji)
                                      do f = 1, nua
                                          do n = 1, noa
                                              if (pspace_abb(f, b, e, n, j, i) /= 0) then
                                                  res6 = res6 + H2B_ovvv(n, a, f, e) * t3c(pspace_abb(f, b, e, n, j, i))
                                              end if
                                              if (pspace_abb(f, a, e, n, j, i) /= 0) then
                                                  res6 = res6 - H2B_ovvv(n, b, f, e) * t3c(pspace_abb(f, a, e, n, j, i))
                                              end if
                                          end do
                                      end do

                                  end do

                                  do e = 1, nua
                                      do n = 1, noa
                                          ! diagram 4: -A(ij) h2b(nmei) * t3c(ebanjm)
                                          do m = 1, nob
                                              if (pspace_abb(e, b, a, n, j, m) /= 0) then
                                                  res4 = res4 - H2B_oovo(n, m, e, i) * t3c(pspace_abb(e, b, a, n, j, m))
                                              end if
                                              if (pspace_abb(e, b, a, n, i, m) /= 0) then
                                                  res4 = res4 + H2B_oovo(n, m, e, j) * t3c(pspace_abb(e, b, a, n, i, m))
                                              end if
                                          end do
                                          ! diagram 2: h1a(ne) * t3c(ebanji)
                                          if (pspace_abb(e, b, a, n, j, i) /= 0) then
                                              res2 = res2 + H1A_ov(n, e) * t3c(pspace_abb(e, b, a, n, j, i))
                                          end if
                                      end do

                                  end do

                                  denom = fB_oo(i, i) + fB_oo(j, j) - fB_vv(a, a) - fB_vv(b, b)
                                  val = X2C(b, a, j, i) - X2C(a, b, j, i) - X2C(b, a, i, j) + X2C(a, b, i, j)
                                  val = val + res1 + res2 + res3 + res4 + res5 + res6

                                  t2c(b, a, j, i) =  t2c(b, a, j, i) + val/(denom - shift)
                                  t2c(a, b, j, i) = -t2c(b, a, j, i)
                                  t2c(b, a, i, j) = -t2c(b, a, j, i)
                                  t2c(a, b, i, j) =  t2c(b, a, j, i)

                                  resid(b, a, j, i) =  val
                                  resid(a, b, j, i) = -val
                                  resid(b, a, i, j) = -val
                                  resid(a, b, i, j) =  val

                              end do
                          end do
                      end do
                  end do

              end subroutine update_t2c_opt

              subroutine update_t3a_p_opt2(t3a_new, resid,&
                                           t2a, t3a, t3b,&
                                           pspace_aaa, pspace_aab,&
                                           H1A_oo, H1A_vv,&
                                           H2A_oovv, H2A_vvov, H2A_vooo,&
                                           H2A_oooo, H2A_voov, H2A_vvvv,&
                                           H2B_oovv, H2B_voov,&
                                           fA_oo, fA_vv,&
                                           shift,&
                                           n3a_p, n3b_p,&
                                           noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub, n3a_p, n3b_p
                  real(kind=8), intent(in) :: t2a(nua, nua, noa, noa),&
                                              t3a(n3a_p),&
                                              t3b(n3b_p),&
                                              H1A_oo(noa, noa), H1A_vv(nua, nua),&
                                              H2A_oovv(noa, noa, nua, nua),&
                                              H2B_oovv(noa, nob, nua, nub),&
                                              H2A_vvov(nua, nua, noa, nua),&
                                              H2A_vooo(nua, noa, noa, noa),&
                                              H2A_oooo(noa, noa, noa, noa),&
                                              H2A_voov(nua, noa, noa, nua),&
                                              H2A_vvvv(nua, nua, nua, nua),&
                                              H2B_voov(nua, nob, noa, nub),&
                                              fA_vv(nua, nua), fA_oo(noa, noa),&
                                              shift
                  integer, intent(in) :: pspace_aaa(nua, nua, nua, noa, noa, noa),&
                                         pspace_aab(nua, nua, nub, noa, noa, nob)

                  real(kind=8), intent(out) :: t3a_new(n3a_p), resid(n3a_p)

                  real(kind=8) :: I2A_vvov(nua, nua, noa, nua), I2A_vooo(nua, noa, noa, noa)
                  real(kind=8) :: val, denom
                  real(kind=8) :: res1, res2, res3, res4, res5, res6, res7, res8
                  integer :: idx, a, b, c, i, j, k, e, f, m, n

                  t3a_new = 0.0d0
                  resid = 0.0d0

                  ! compute VT3 intermediates
                  I2A_vooo = H2A_vooo
                  I2A_vvov = H2A_vvov
                  do a = 1, nua
                      do i = 1, noa
                          do m = 1, noa
                              do e = 1, nua

                                  ! I2A(amij) = 1/2 h2a(mnef) * t3a(aefijn) + h2b(mnef) * t3b(aefijn) - h1A(me) * t2a(aeij)
                                  do j = i + 1, noa
                                      do n = 1, noa
                                          do f = e + 1, nua
                                              if (pspace_aaa(a, e, f, i, j, n) /= 0) then
                                                I2A_vooo(a, m, i, j) = I2A_vooo(a, m, i, j) + H2A_oovv(m, n, e, f) * t3a(pspace_aaa(a, e, f, i, j, n))
                                              end if
                                          end do
                                      end do
                                      do n = 1, nob
                                          do f = 1, nub
                                              if (pspace_aab(a, e, f, i, j, n) /= 0) then
                                                I2A_vooo(a, m, i, j) = I2A_vooo(a, m, i, j) + H2B_oovv(m, n, e, f) * t3b(pspace_aab(a, e, f, i, j, n))
                                              end if
                                          end do
                                      end do
                                      I2A_vooo(a, m, j, i) = -1.0 * I2A_vooo(a, m, i, j)
                                  end do
                                  ! I2A(abie) = -1/2 h2a(mnef) * t3a(abfimn) - h2b(mnef) * t3b(abfimn)
                                  do b = a + 1, nua
                                      do n = m + 1, noa
                                          do f = 1, nua
                                              if (pspace_aaa(a, b, f, i, m, n) /= 0) then
                                                I2A_vvov(a, b, i, e) = I2A_vvov(a, b, i, e) - H2A_oovv(m, n, e, f) * t3a(pspace_aaa(a, b, f, i, m, n))
                                              end if
                                          end do
                                      end do
                                      do n = 1, nob
                                          do f = 1, nub
                                              if (pspace_aab(a, b, f, i, m, n) /= 0) then
                                                I2A_vvov(a, b, i, e) = I2A_vvov(a, b, i, e) - H2B_oovv(m, n, e, f) * t3b(pspace_aab(a, b, f, i, m, n))
                                              end if
                                          end do
                                      end do
                                      I2A_vvov(b, a, i, e) = -1.0 * I2A_vvov(a, b, i, e)
                                  end do

                              end do
                          end do
                      end do
                  end do

                  ! loop over projection determinants in P space
                  do a = 1, nua; do b = a + 1, nua; do c = b + 1, nua;
                  do i = 1, noa; do j = i + 1, noa; do k = j + 1, noa;

                      if (pspace_aaa(a, b, c, i, j, k) /= 1) cycle

                      res1 = 0.0d0
                      res2 = 0.0d0
                      res3 = 0.0d0
                      res4 = 0.0d0
                      res5 = 0.0d0
                      res6 = 0.0d0
                      res7 = 0.0d0
                      res8 = 0.0d0

                      do e = 1, nua

                          ! A(c/ab) h1a(ce) * t3a(abeijk)
                          if (pspace_aaa(a, b, e, i, j, k) /= 0) then
                            res1 = res1 + H1A_vv(c, e) * t3a(pspace_aaa(a, b, e, i, j, k))
                          end if
                          if (pspace_aaa(c, b, e, i, j, k) /= 0) then
                            res1 = res1 - H1A_vv(a, e) * t3a(pspace_aaa(c, b, e, i, j, k))
                          end if
                          if (pspace_aaa(a, c, e, i, j, k) /= 0) then
                            res1 = res1 - H1A_vv(b, e) * t3a(pspace_aaa(a, c, e, i, j, k))
                          end if

                          ! 1/2 A(c/ab) h2a(abef) * t3a(efcijk)
                          do f = e + 1, nua
                              if (pspace_aaa(e, f, c, i, j, k) /= 0) then
                                res2 = res2 + H2A_vvvv(a, b, e, f) * t3a(pspace_aaa(e, f, c, i, j, k))
                              end if
                              if (pspace_aaa(e, f, a, i, j, k) /= 0) then
                                res2 = res2 - H2A_vvvv(c, b, e, f) * t3a(pspace_aaa(e, f, a, i, j, k))
                              end if
                              if (pspace_aaa(e, f, b, i, j, k) /= 0) then
                                res2 = res2 - H2A_vvvv(a, c, e, f) * t3a(pspace_aaa(e, f, b, i, j, k))
                              end if
                          end do

                          ! A(i/jk)A(c/ab) h2a(amie) * t3a(ebcmjk)
                          do m = 1, noa
                              if (pspace_aaa(e, b, c, m, j, k) /= 0) then
                                res3 = res3 + H2A_voov(a, m, i, e) * t3a(pspace_aaa(e, b, c, m, j, k))
                              end if
                              if (pspace_aaa(e, b, c, m, i, k) /= 0) then
                                res3 = res3 - H2A_voov(a, m, j, e) * t3a(pspace_aaa(e, b, c, m, i, k))
                              end if
                              if (pspace_aaa(e, b, c, m, j, i) /= 0) then
                                res3 = res3 - H2A_voov(a, m, k, e) * t3a(pspace_aaa(e, b, c, m, j, i))
                              end if
                              if (pspace_aaa(e, a, c, m, j, k) /= 0) then
                                res3 = res3 - H2A_voov(b, m, i, e) * t3a(pspace_aaa(e, a, c, m, j, k))
                              end if
                              if (pspace_aaa(e, a, c, m, i, k) /= 0) then
                                res3 = res3 + H2A_voov(b, m, j, e) * t3a(pspace_aaa(e, a, c, m, i, k))
                              end if
                              if (pspace_aaa(e, a, c, m, j, i) /= 0) then
                                res3 = res3 + H2A_voov(b, m, k, e) * t3a(pspace_aaa(e, a, c, m, j, i))
                              end if
                              if (pspace_aaa(e, b, a, m, j, k) /= 0) then
                                res3 = res3 - H2A_voov(c, m, i, e) * t3a(pspace_aaa(e, b, a, m, j, k))
                              end if
                              if (pspace_aaa(e, b, a, m, i, k) /= 0) then
                                res3 = res3 + H2A_voov(c, m, j, e) * t3a(pspace_aaa(e, b, a, m, i, k))
                              end if
                              if (pspace_aaa(e, b, a, m, j, i) /= 0) then
                                res3 = res3 + H2A_voov(c, m, k, e) * t3a(pspace_aaa(e, b, a, m, j, i))
                              end if
                          end do

                          do m = 1, nob
                              ! A(i/jk)A(a/bc) h2b(amie) * t3b(bcejkm)
                              if (pspace_aab(b, c, e, j, k, m) /= 0) then
                                res5 = res5 + H2B_voov(a, m, i, e) * t3b(pspace_aab(b, c, e, j, k, m))
                              end if
                              if (pspace_aab(a, c, e, j, k, m) /= 0) then
                                res5 = res5 - H2B_voov(b, m, i, e) * t3b(pspace_aab(a, c, e, j, k, m))
                              end if
                              if (pspace_aab(b, a, e, j, k, m) /= 0) then
                                res5 = res5 - H2B_voov(c, m, i, e) * t3b(pspace_aab(b, a, e, j, k, m))
                              end if
                              if (pspace_aab(b, c, e, i, k, m) /= 0) then
                                res5 = res5 - H2B_voov(a, m, j, e) * t3b(pspace_aab(b, c, e, i, k, m))
                              end if
                              if (pspace_aab(a, c, e, i, k, m) /= 0) then
                                res5 = res5 + H2B_voov(b, m, j, e) * t3b(pspace_aab(a, c, e, i, k, m))
                              end if
                              if (pspace_aab(b, a, e, i, k, m) /= 0) then
                                res5 = res5 + H2B_voov(c, m, j, e) * t3b(pspace_aab(b, a, e, i, k, m))
                              end if
                              if (pspace_aab(b, c, e, j, i, m) /= 0) then
                                res5 = res5 - H2B_voov(a, m, k, e) * t3b(pspace_aab(b, c, e, j, i, m))
                              end if
                              if (pspace_aab(a, c, e, j, i, m) /= 0) then
                                res5 = res5 + H2B_voov(b, m, k, e) * t3b(pspace_aab(a, c, e, j, i, m))
                              end if
                              if (pspace_aab(b, a, e, j, i, m) /= 0) then
                                res5 = res5 + H2B_voov(c, m, k, e) * t3b(pspace_aab(b, a, e, j, i, m))
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
                              if (pspace_aab(b, c, e + nua, j, k, m) /= 0) then
                                res5 = res5 + H2B_voov(a, m, i, e + nua) * t3b(pspace_aab(b, c, e + nua, j, k, m))
                              end if
                              if (pspace_aab(a, c, e + nua, j, k, m) /= 0) then
                                res5 = res5 - H2B_voov(b, m, i, e + nua) * t3b(pspace_aab(a, c, e + nua, j, k, m))
                              end if
                              if (pspace_aab(b, a, e + nua, j, k, m) /= 0) then
                                res5 = res5 - H2B_voov(c, m, i, e + nua) * t3b(pspace_aab(b, a, e + nua, j, k, m))
                              end if
                              if (pspace_aab(b, c, e + nua, i, k, m) /= 0) then
                                res5 = res5 - H2B_voov(a, m, j, e + nua) * t3b(pspace_aab(b, c, e + nua, i, k, m))
                              end if
                              if (pspace_aab(a, c, e + nua, i, k, m) /= 0) then
                                res5 = res5 + H2B_voov(b, m, j, e + nua) * t3b(pspace_aab(a, c, e + nua, i, k, m))
                              end if
                              if (pspace_aab(b, a, e + nua, i, k, m) /= 0) then
                                res5 = res5 + H2B_voov(c, m, j, e + nua) * t3b(pspace_aab(b, a, e + nua, i, k, m))
                              end if
                              if (pspace_aab(b, c, e + nua, j, i, m) /= 0) then
                                res5 = res5 - H2B_voov(a, m, k, e + nua) * t3b(pspace_aab(b, c, e + nua, j, i, m))
                              end if
                              if (pspace_aab(a, c, e + nua, j, i, m) /= 0) then
                                res5 = res5 + H2B_voov(b, m, k, e + nua) * t3b(pspace_aab(a, c, e + nua, j, i, m))
                              end if
                              if (pspace_aab(b, a, e + nua, j, i, m) /= 0) then
                                res5 = res5 + H2B_voov(c, m, k, e + nua) * t3b(pspace_aab(b, a, e + nua, j, i, m))
                              end if
                          end do
                      end do

                      do m = 1, noa

                          ! -A(i/jk) h1a(mi) * t3a(abcmjk)
                          if (pspace_aaa(a, b, c, m, j, k) /= 0) then
                              res6 = res6 - H1A_oo(m, i) * t3a(pspace_aaa(a, b, c, m, j, k))
                          end if
                          if (pspace_aaa(a, b, c, m, i, k) /= 0) then
                              res6 = res6 + H1A_oo(m, j) * t3a(pspace_aaa(a, b, c, m, i, k))
                          end if
                          if (pspace_aaa(a, b, c, m, j, i) /= 0) then
                              res6 = res6 + H1A_oo(m, k) * t3a(pspace_aaa(a, b, c, m, j, i))
                          end if

                          ! 1/2 A(k/ij) h2a(mnij) * t3a(abcmnk)
                          do n = m + 1, noa
                              if (pspace_aaa(a, b, c, m, n, k) /= 0) then
                                  res7 = res7 + H2A_oooo(m, n, i, j) * t3a(pspace_aaa(a, b, c, m, n, k))
                              end if
                              if (pspace_aaa(a, b, c, m, n, i) /= 0) then
                                  res7 = res7 - H2A_oooo(m, n, k, j) * t3a(pspace_aaa(a, b, c, m, n, i))
                              end if
                              if (pspace_aaa(a, b, c, m, n, j) /= 0) then
                                  res7 = res7 - H2A_oooo(m ,n, i, k) * t3a(pspace_aaa(a, b, c, m, n, j))
                              end if
                          end do

                          ! -A(k/ij)A(a/bc) h2a(amij) * t2a(bcmk)
                          res8 = res8 - I2A_vooo(a, m, i, j) * t2a(b, c, m, k)
                          res8 = res8 + I2A_vooo(b, m, i, j) * t2a(a, c, m, k)
                          res8 = res8 + I2A_vooo(c, m, i, j) * t2a(b, a, m, k)
                          res8 = res8 + I2A_vooo(a, m, k, j) * t2a(b, c, m, i)
                          res8 = res8 - I2A_vooo(b, m, k, j) * t2a(a, c, m, i)
                          res8 = res8 - I2A_vooo(c, m, k, j) * t2a(b, a, m, i)
                          res8 = res8 + I2A_vooo(a, m, i, k) * t2a(b, c, m, j)
                          res8 = res8 - I2A_vooo(b, m, i, k) * t2a(a, c, m, j)
                          res8 = res8 - I2A_vooo(c, m, i, k) * t2a(b, a, m, j)

                      end do

                      denom = fA_oo(I, I) + fA_oo(J, J) + fA_oo(K, K) - fA_vv(A, A) - fA_vv(B, B) - fA_vv(C, C)

                      val = res1 + res2 + res3 + res4 + res5 + res6 + res7 + res8

                      val = val/(denom - shift)

                      ! update
                      t3a_new(pspace_aaa(A,B,C,I,J,K)) = t3a(pspace_aaa(A,B,C,I,J,K)) + val

                      t3a_new(pspace_aaa(A,B,C,K,I,J)) = t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(A,B,C,J,K,I)) = t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(A,B,C,I,K,J)) = -t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(A,B,C,J,I,K)) = -t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(A,B,C,K,J,I)) = -t3a_new(pspace_aaa(A,B,C,I,J,K))

                      t3a_new(pspace_aaa(B,A,C,I,J,K)) = -t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(B,A,C,K,I,J)) = -t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(B,A,C,J,K,I)) = -t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(B,A,C,I,K,J)) = t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(B,A,C,J,I,K)) = t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(B,A,C,K,J,I)) = t3a_new(pspace_aaa(A,B,C,I,J,K))

                      t3a_new(pspace_aaa(A,C,B,I,J,K)) = -t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(A,C,B,K,I,J)) = -t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(A,C,B,J,K,I)) = -t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(A,C,B,I,K,J)) = t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(A,C,B,J,I,K)) = t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(A,C,B,K,J,I)) = t3a_new(pspace_aaa(A,B,C,I,J,K))

                      t3a_new(pspace_aaa(C,B,A,I,J,K)) = -t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(C,B,A,K,I,J)) = -t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(C,B,A,J,K,I)) = -t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(C,B,A,I,K,J)) = t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(C,B,A,J,I,K)) = t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(C,B,A,K,J,I)) = t3a_new(pspace_aaa(A,B,C,I,J,K))

                      t3a_new(pspace_aaa(B,C,A,I,J,K)) = t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(B,C,A,K,I,J)) = t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(B,C,A,J,K,I)) = t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(B,C,A,I,K,J)) = -t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(B,C,A,J,I,K)) = -t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(B,C,A,K,J,I)) = -t3a_new(pspace_aaa(A,B,C,I,J,K))

                      t3a_new(pspace_aaa(C,A,B,I,J,K)) = t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(C,A,B,K,I,J)) = t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(C,A,B,J,K,I)) = t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(C,A,B,I,K,J)) = -t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(C,A,B,J,I,K)) = -t3a_new(pspace_aaa(A,B,C,I,J,K))
                      t3a_new(pspace_aaa(C,A,B,K,J,I)) = -t3a_new(pspace_aaa(A,B,C,I,J,K))

                      resid(pspace_aaa(A,B,C,I,J,K)) = val
                      resid(pspace_aaa(A,B,C,K,I,J)) = val
                      resid(pspace_aaa(A,B,C,J,K,I)) = val
                      resid(pspace_aaa(A,B,C,I,K,J)) = -val
                      resid(pspace_aaa(A,B,C,J,I,K)) = -val
                      resid(pspace_aaa(A,B,C,K,J,I)) = -val
                      resid(pspace_aaa(B,C,A,I,J,K)) = val
                      resid(pspace_aaa(B,C,A,K,I,J)) = val
                      resid(pspace_aaa(B,C,A,J,K,I)) = val
                      resid(pspace_aaa(B,C,A,I,K,J)) = -val
                      resid(pspace_aaa(B,C,A,J,I,K)) = -val
                      resid(pspace_aaa(B,C,A,K,J,I)) = -val
                      resid(pspace_aaa(C,A,B,I,J,K)) = val
                      resid(pspace_aaa(C,A,B,K,I,J)) = val
                      resid(pspace_aaa(C,A,B,J,K,I)) = val
                      resid(pspace_aaa(C,A,B,I,K,J)) = -val
                      resid(pspace_aaa(C,A,B,J,I,K)) = -val
                      resid(pspace_aaa(C,A,B,K,J,I)) = -val
                      resid(pspace_aaa(A,C,B,I,J,K)) = -val
                      resid(pspace_aaa(A,C,B,K,I,J)) = -val
                      resid(pspace_aaa(A,C,B,J,K,I)) = -val
                      resid(pspace_aaa(A,C,B,I,K,J)) = val
                      resid(pspace_aaa(A,C,B,J,I,K)) = val
                      resid(pspace_aaa(A,C,B,K,J,I)) = val
                      resid(pspace_aaa(B,A,C,I,J,K)) = -val
                      resid(pspace_aaa(B,A,C,K,I,J)) = -val
                      resid(pspace_aaa(B,A,C,J,K,I)) = -val
                      resid(pspace_aaa(B,A,C,I,K,J)) = val
                      resid(pspace_aaa(B,A,C,J,I,K)) = val
                      resid(pspace_aaa(B,A,C,K,J,I)) = val
                      resid(pspace_aaa(C,B,A,I,J,K)) = -val
                      resid(pspace_aaa(C,B,A,K,I,J)) = -val
                      resid(pspace_aaa(C,B,A,J,K,I)) = -val
                      resid(pspace_aaa(C,B,A,I,K,J)) = val
                      resid(pspace_aaa(C,B,A,J,I,K)) = val
                      resid(pspace_aaa(C,B,A,K,J,I)) = val

                  end do; end do; end do;
                  end do; end do; end do;

              end subroutine update_t3a_p_opt2

              subroutine update_t3b_p_opt2(t3b_new, resid,&
                                           t2a, t2b, t3a, t3b, t3c,&
                                           pspace_aaa, pspace_aab, pspace_abb,&
                                           H1A_oo, H1A_vv, H1B_oo, H1B_vv,&
                                           H2A_oovv, H2A_vvov, H2A_vooo, H2A_oooo, H2A_voov, H2A_vvvv,&
                                           H2B_oovv, H2B_vvov, H2B_vvvo, H2B_vooo, H2B_ovoo,&
                                           H2B_oooo, H2B_voov, H2B_vovo, H2B_ovov, H2B_ovvo, H2B_vvvv,&
                                           H2C_oovv, H2C_voov,&
                                           fA_oo, fA_vv, fB_oo, fB_vv,&
                                           shift,&
                                           n3a_p, n3b_p, n3c_p,&
                                           noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub, n3a_p, n3b_p, n3c_p
                  integer, intent(in) :: pspace_aaa(nua, nua, nua, noa, noa, noa),&
                                         pspace_aab(nua, nua, nub, noa, noa, nob),&
                                         pspace_abb(nua, nub, nub, noa, nob, nob)
                  real(kind=8), intent(in) :: t2a(1:nua,1:nua,1:noa,1:noa),&
                                              t2b(1:nua,1:nub,1:noa,1:nob),&
                                              t3a(n3a_p),&
                                              t3b(n3b_p),&
                                              t3c(n3c_p),&
                                              H1A_oo(1:noa,1:noa),&
                                              H1A_vv(1:nua,1:nua),&
                                              H1B_oo(1:nob,1:nob),&
                                              H1B_vv(1:nub,1:nub),&
                                              H2A_oovv(1:noa,1:noa,1:nua,1:nua),&
                                              H2A_vvov(1:nua,1:nua,1:noa,1:nua),&
                                              H2A_vooo(1:nua,1:noa,1:noa,1:noa),&
                                              H2A_oooo(1:noa,1:noa,1:noa,1:noa),&
                                              H2A_voov(1:nua,1:noa,1:noa,1:nua),&
                                              H2A_vvvv(1:nua,1:nua,1:nua,1:nua),&
                                              H2B_oovv(1:noa,1:nob,1:nua,1:nub),&
                                              H2B_vooo(1:nua,1:nob,1:noa,1:nob),&
                                              H2B_ovoo(1:noa,1:nub,1:noa,1:nob),&
                                              H2B_vvov(1:nua,1:nub,1:noa,1:nub),&
                                              H2B_vvvo(1:nua,1:nub,1:nua,1:nob),&
                                              H2B_oooo(1:noa,1:nob,1:noa,1:nob),&
                                              H2B_voov(1:nua,1:nob,1:noa,1:nub),&
                                              H2B_vovo(1:nua,1:nob,1:nua,1:nob),&
                                              H2B_ovov(1:noa,1:nub,1:noa,1:nub),&
                                              H2B_ovvo(1:noa,1:nub,1:nua,1:nob),&
                                              H2B_vvvv(1:nua,1:nub,1:nua,1:nub),&
                                              H2C_oovv(1:nob,1:nob,1:nub,1:nub),&
                                              H2C_voov(1:nub,1:nob,1:nob,1:nub),&
                                              fA_oo(1:noa,1:noa), fA_vv(1:nua,1:nua),&
                                              fB_oo(1:nob,1:nob), fB_vv(1:nub,1:nub),&
                                              shift

                  real(kind=8), intent(out) :: t3b_new(n3b_p), resid(n3b_p)
                  integer :: i, j, k, a, b, c, m, n, e, f

                  real(kind=8) :: I2A_vooo(nua, noa, noa, noa),&
                                  I2A_vvov(nua, nua, noa, nua),&
                                  I2B_vooo(nua, nob, noa, nob),&
                                  I2B_ovoo(noa, nub, noa, nob),&
                                  I2B_vvov(nua, nub, noa, nub),&
                                  I2B_vvvo(nua, nub, nua, nob)

                  real(kind=8) :: denom, val,&
                                  res1, res2, res3, res4, res5, res6, res7, res8, res9, res10, res11, res12, res13,&
                                  res14, res15, res16, res17, res18, res19, res20

                  resid = 0.0d0
                  t3b_new = 0.0d0

                  ! Compute VT3 intermediates
                  I2A_vooo = H2A_vooo
                  I2A_vvov = H2A_vvov
                  I2B_vooo = H2B_vooo
                  I2B_ovoo = H2B_ovoo
                  I2B_vvov = H2B_vvov
                  I2B_vvvo = H2B_vvvo
                  
                  do a = 1, nua
                      do i = 1, noa
                          do m = 1, noa
                              do e = 1, nua

                                  ! I2A(amij) = 1/2 h2a(mnef) * t3a(aefijn) + h2b(mnef) * t3b(aefijn) - h1A(me) * t2a(aeij)
                                  do j = i + 1, noa
                                      do n = 1, noa
                                          do f = e + 1, nua
                                              if (pspace_aaa(a, e, f, i, j, n) /= 0) then
                                                I2A_vooo(a, m, i, j) = I2A_vooo(a, m, i, j) + H2A_oovv(m, n, e, f) * t3a(pspace_aaa(a, e, f, i, j, n))
                                              end if
                                          end do
                                      end do
                                      do n = 1, nob
                                          do f = 1, nub
                                              if (pspace_aab(a, e, f, i, j, n) /= 0) then
                                                I2A_vooo(a, m, i, j) = I2A_vooo(a, m, i, j) + H2B_oovv(m, n, e, f) * t3b(pspace_aab(a, e, f, i, j, n))
                                              end if
                                          end do
                                      end do
                                      I2A_vooo(a, m, j, i) = -1.0d0 * I2A_vooo(a, m, i, j)
                                  end do
                                  ! I2A(abie) = -1/2 h2a(mnef) * t3a(abfimn) - h2b(mnef) * t3b(abfimn)
                                  do b = a + 1, nua
                                      do n = m + 1, noa
                                          do f = 1, nua
                                              if (pspace_aaa(a, b, f, i, m, n) /= 0) then
                                                I2A_vvov(a, b, i, e) = I2A_vvov(a, b, i, e) - H2A_oovv(m, n, e, f) * t3a(pspace_aaa(a, b, f, i, m, n))
                                              end if
                                          end do
                                      end do
                                      do n = 1, nob
                                          do f = 1, nub
                                              if (pspace_aab(a, b, f, i, m, n) /= 0) then
                                                I2A_vvov(a, b, i, e) = I2A_vvov(a, b, i, e) - H2B_oovv(m, n, e, f) * t3b(pspace_aab(a, b, f, i, m, n))
                                              end if
                                          end do
                                      end do
                                      I2A_vvov(b, a, i, e) = -1.0d0 * I2A_vvov(a, b, i, e)
                                  end do

                              end do
                          end do
                      end do
                  end do

                  do a = 1, nua
                      do i = 1, noa
                          do m = 1, nob
                              do e = 1, nub

                                  ! I2B(amij) = h2b(nmfe) * t3b(afeinj) + 1/2 h2c(nmfe) * t3c(afeinj)
                                  do j = 1, nob
                                      do n = 1, noa
                                          do f = 1, nua
                                              if (pspace_aab(a, f, e, i, n, j) /= 0) then
                                                I2B_vooo(a, m, i, j) = I2B_vooo(a, m, i, j) + H2B_oovv(n, m, f, e) * t3b(pspace_aab(a, f, e, i, n, j))
                                              end if
                                          end do
                                      end do
                                      do n = 1, nob
                                          do f = e + 1, nub
                                              if (pspace_abb(a, f, e, i, n, j) /= 0) then
                                                I2B_vooo(a, m, i, j) = I2B_vooo(a, m, i, j) + H2C_oovv(n, m, f, e) * t3c(pspace_abb(a, f, e, i, n, j))
                                              end if
                                          end do
                                      end do
                                  end do
                                  ! I2B(abie) = -h2b(nmfe) * t3b(afbinm) - 1/2 h2c(mnef) * t3c(afbinm)
                                  do b = 1, nub
                                      do n = 1, noa
                                          do f = 1, nua
                                              if (pspace_aab(a, f, b, i, n, m) /= 0) then
                                                I2B_vvov(a, b, i, e) = I2B_vvov(a, b, i, e) - H2B_oovv(n, m, f, e) * t3b(pspace_aab(a, f, b, i, n, m))
                                              end if
                                          end do
                                      end do
                                      do n = m + 1, nob
                                          do f = 1, nub
                                              if (pspace_abb(a, f, b, i, n, m) /= 0) then
                                                I2B_vvov(a, b, i, e) = I2B_vvov(a, b, i, e) - H2C_oovv(n, m, f, e) * t3c(pspace_abb(a, f, b, i, n, m))
                                              end if
                                          end do
                                      end do
                                  end do

                            end do
                          end do
                      end do
                  end do

                  do a = 1, nub
                      do i = 1, nob
                          do m = 1, noa
                              do e = 1, nua

                                  ! I2B(maji) = 1/2 h2a(mnef) * t3b(efajni) + h2b(mnef) * t3c(efajni)
                                  do j = 1, noa
                                      do n = 1, noa
                                          do f = e + 1, nua
                                              if (pspace_aab(e, f, a, j, n, i) /= 0) then
                                                I2B_ovoo(m, a, j, i) = I2B_ovoo(m, a, j, i) + H2A_oovv(m, n, e, f) * t3b(pspace_aab(e, f, a, j, n, i))
                                              end if
                                          end do
                                      end do
                                      do n = 1, nob
                                          do f = 1, nub
                                              if (pspace_abb(e, f, a, j, n, i) /= 0) then
                                                I2B_ovoo(m, a, j, i) = I2B_ovoo(m, a, j, i) + H2B_oovv(m, n, e, f) * t3c(pspace_abb(e, f, a, j, n, i))
                                              end if
                                          end do
                                      end do
                                  end do
                                  ! I2B(baei) = -1/2 h2a(mnef) * t3b(bfamni) - h2b(mnef) * t3c(bfamni)
                                  do b = 1, nua
                                      do n = m + 1, noa
                                          do f = 1, nua
                                              if (pspace_aab(b, f, a, m, n, i) /= 0) then
                                                I2B_vvvo(b, a, e, i) = I2B_vvvo(b, a, e, i) - H2A_oovv(m, n, e, f) * t3b(pspace_aab(b, f, a, m, n, i))
                                              end if
                                          end do
                                      end do
                                      do n = 1, nob
                                          do f = 1, nub
                                              if (pspace_abb(b, f, a, m, n, i) /= 0) then
                                                I2B_vvvo(b, a, e, i) = I2B_vvvo(b, a, e, i) - H2B_oovv(m, n, e, f) * t3c(pspace_abb(b, f, a, m, n, i))
                                              end if
                                          end do
                                      end do
                                  end do

                              end do
                          end do
                      end do
                  end do

                  ! Loop over projection determinants in P space
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
                      res14 = 0.0d0
                      res15 = 0.0d0
                      res16 = 0.0d0
                      res17 = 0.0d0
                      res18 = 0.0d0
                      res19 = 0.0d0
                      res20 = 0.0d0

                      ! nua < nub
                      do e = 1, nua

                          ! diagram 1: A(ab) h1A(ae) * t3b(ebcijk)
                          if (pspace_aab(e, b, c, i, j, k) /= 0) then
                            res1 = res1 + H1A_vv(a, e) * t3b(pspace_aab(e, b, c, i, j, k))
                          end if
                          if (pspace_aab(e, a, c, i, j, k) /= 0) then
                            res1 = res1 - H1A_vv(b, e) * t3b(pspace_aab(e, a, c, i, j, k))
                          end if

                          ! diagram 2: h1B(ce) * t3b(abeijk)
                          if (pspace_aab(a, b, e, i, j, k) /= 0) then
                            res2 = res2 + H1B_vv(c, e) * t3b(pspace_aab(a, b, e, i, j, k))
                          end if

                          ! diagram 3: 0.5 * h2A(abef) * t3b(efcijk)
                          ! diagram 4: A(ab) h2B(bcef) * t3b(aefijk)
                          do f = 1, nua
                              if (pspace_aab(e, f, c, i, j, k) /= 0) then
                                res3 = res3 + 0.5d0 * H2A_vvvv(a, b, e, f) * t3b(pspace_aab(e, f, c, i, j, k))
                              end if
                              if (pspace_aab(a, e, f, i, j, k) /= 0) then
                                res4 = res4 + H2B_vvvv(b, c, e, f) * t3b(pspace_aab(a, e, f, i, j, k))
                              end if
                              if (pspace_aab(b, e, f, i, j, k) /= 0) then
                                res4 = res4 - H2B_vvvv(a, c, e, f) * t3b(pspace_aab(b, e, f, i, j, k))
                              end if
                          end do
                          do f = 1, nub - nua
                              if (pspace_aab(a, e, f + nua, i, j, k) /= 0) then
                                res4 = res4 + H2B_vvvv(b, c, e, f + nua) * t3b(pspace_aab(a, e, f + nua, i, j, k))
                              end if
                              if (pspace_aab(b, e, f + nua, i, j, k) /= 0) then
                                res4 = res4 - H2B_vvvv(a, c, e, f + nua) * t3b(pspace_aab(b, e, f + nua, i, j, k))
                              end if
                          end do

                          ! diagram 5: A(ij)A(ab) h2A(amie) * t3b(ebcmjk)
                          ! diagram 7: h2B(mcek) * t3a(abeijm)
                          ! diagram 10: -A(ij) h2B(mcie) * t3b(abemjk)
                          do m = 1, noa
                              if (pspace_aab(e, b, c, m, j, k) /= 0) then
                                res5 = res5 + H2A_voov(a, m, i, e) * t3b(pspace_aab(e, b, c, m, j, k))
                              end if
                              if (pspace_aab(e, a, c, m, j, k) /= 0) then
                                res5 = res5 - H2A_voov(b, m, i, e) * t3b(pspace_aab(e, a, c, m, j, k))
                              end if
                              if (pspace_aab(e, b, c, m, i, k) /= 0) then
                                res5 = res5 - H2A_voov(a, m, j, e) * t3b(pspace_aab(e, b, c, m, i, k))
                              end if
                              if (pspace_aab(e, a, c, m, i, k) /= 0) then
                                res5 = res5 + H2A_voov(b, m, j, e) * t3b(pspace_aab(e, a, c, m, i, k))
                              end if

                              if (pspace_aaa(a, b, e, i, j, m) /= 0) then
                                res7 = res7 + H2B_ovvo(m, c, e, k) * t3a(pspace_aaa(a, b, e, i, j, m))
                              end if

                              if (pspace_aab(a, b, e, m, j, k) /= 0) then
                                res10 = res10 - H2B_ovov(m, c, i, e) * t3b(pspace_aab(a, b, e, m, j, k))
                              end if
                              if (pspace_aab(a, b, e, m, i, k) /= 0) then
                                res10 = res10 + H2B_ovov(m, c, j, e) * t3b(pspace_aab(a, b, e, m, i, k))
                              end if
                          end do

                          ! diagram 6: A(ij)A(ab) h2B(amie) * t3c(becjmk)
                          ! diagram 8: h2C(cmke) * t3b(abeijm)
                          ! diagram 9: -A(ab) h2B(amek) * t3b(ebcijm)
                          do m = 1, nob
                              if (pspace_abb(b, e, c, j, m, k) /= 0) then
                                res6 = res6 + H2B_voov(a, m, i, e) * t3c(pspace_abb(b, e, c, j, m, k))
                              end if
                              if (pspace_abb(a, e, c, j, m, k) /= 0) then
                                res6 = res6 - H2B_voov(b, m, i, e) * t3c(pspace_abb(a, e, c, j, m, k))
                              end if
                              if (pspace_abb(b, e, c, i, m, k) /= 0) then
                                res6 = res6 - H2B_voov(a, m, j, e) * t3c(pspace_abb(b, e, c, i, m, k))
                              end if
                              if (pspace_abb(a, e, c, i, m, k) /= 0) then
                                res6 = res6 + H2B_voov(b, m, j, e) * t3c(pspace_abb(a, e, c, i, m, k))
                              end if

                              if (pspace_aab(a, b, e, i, j, m) /= 0) then
                                res8 = res8 + H2C_voov(c, m, k, e) * t3b(pspace_aab(a, b, e, i, j, m))
                              end if

                              if (pspace_aab(e, b, c, i, j, m) /= 0) then
                                res9 = res9 - H2B_vovo(a, m, e, k) * t3b(pspace_aab(e, b, c, i, j, m))
                              end if
                              if (pspace_aab(e, a, c, i, j, m) /= 0) then
                                res9 = res9 + H2B_vovo(b, m, e, k) * t3b(pspace_aab(e, a, c, i, j, m))
                              end if
                          end do

                          ! diagram 11: A(ab) I2B(bcek) * t2a(aeij)
                          res11 = res11 + I2B_vvvo(b, c, e, k) * t2a(a, e, i, j)
                          res11 = res11 - I2B_vvvo(a, c, e, k) * t2a(b, e, i, j)
                          ! diagram 12: A(ij)A(ab) I2B(acie) * t2b(bejk)
                          res12 = res12 + I2B_vvov(a, c, i, e) * t2b(b, e, j, k)
                          res12 = res12 - I2B_vvov(a, c, j, e) * t2b(b, e, i, k)
                          res12 = res12 - I2B_vvov(b, c, i, e) * t2b(a, e, j, k)
                          res12 = res12 + I2B_vvov(b, c, j, e) * t2b(a, e, i, k)
                          ! diagram 13: A(ij) I2A(abie) * t2b(ecjk)
                          res13 = res13 + I2A_vvov(a, b, i, e) * t2b(e, c, j, k)
                          res13 = res13 - I2A_vvov(a, b, j, e) * t2b(e, c, i, k)

                      end do

                      do e = 1, nub - nua

                          ! diagram 2 : h1B(ce) * t3b(abeijk)
                          if (pspace_aab(a, b, e + nua, i, j, k) /= 0) then
                            res2 = res2 + H1B_vv(c, e + nua) * t3b(pspace_aab(a, b, e + nua, i, j, k))
                          end if

                          ! diagram 10: -A(ij) h2B(mcie) * t3b(abemjk)
                          do m = 1, noa
                              if (pspace_aab(a, b, e + nua, m, j, k) /= 0) then
                                res10 = res10 - H2B_ovov(m, c, i, e + nua) * t3b(pspace_aab(a, b, e + nua, m, j, k))
                              end if
                              if (pspace_aab(a, b, e + nua, m, i, k) /= 0) then
                                res10 = res10 + H2B_ovov(m, c, j, e + nua) * t3b(pspace_aab(a, b, e + nua, m, i, k))
                              end if
                          end do

                          ! diagram 6: A(ij)A(ab) h2B(amie) * t3c(becjmk)
                          ! diagram 8: h2C(cmke) * t3b(abeijm)
                          do m = 1, nob
                              if (pspace_abb(b, e + nua, c, j, m, k) /= 0) then
                                res6 = res6 + H2B_voov(a, m, i, e + nua) * t3c(pspace_abb(b, e + nua, c, j, m, k))
                              end if
                              if (pspace_abb(a, e + nua, c, j, m, k) /= 0) then
                                res6 = res6 - H2B_voov(b, m, i, e + nua) * t3c(pspace_abb(a, e + nua, c, j, m, k))
                              end if
                              if (pspace_abb(b, e + nua, c, i, m, k) /= 0) then
                              res6 = res6 - H2B_voov(a, m, j, e + nua) * t3c(pspace_abb(b, e + nua, c, i, m, k))
                              end if
                              if (pspace_abb(a, e + nua, c, i, m, k) /= 0) then
                                res6 = res6 + H2B_voov(b, m, j, e + nua) * t3c(pspace_abb(a, e + nua, c, i, m, k))
                              end if

                              if (pspace_aab(a, b, e + nua, i, j, m) /= 0) then
                                res8 = res8 + H2C_voov(c, m, k, e + nua) * t3b(pspace_aab(a, b, e + nua, i, j, m))
                              end if
                          end do

                          ! diagram 12: A(ij)A(ab) I2B(acie) * t2b(bejk)
                          res12 = res12 + I2B_vvov(a, c, i, e + nua) * t2b(b, e + nua, j, k)
                          res12 = res12 - I2B_vvov(a, c, j, e + nua) * t2b(b, e + nua, i, k)
                          res12 = res12 - I2B_vvov(b, c, i, e + nua) * t2b(a, e + nua, j, k)
                          res12 = res12 + I2B_vvov(b, c, j, e + nua) * t2b(a, e + nua, i, k)

                      end do

                      do m = 1, noa

                          ! diagram 14: -A(ij) h1a(mi) * t3b(abcmjk)
                          if (pspace_aab(a, b, c, m, j, k) /= 0) then
                              res14 = res14 - H1A_oo(m, i) * t3b(pspace_aab(a, b, c, m, j, k))
                          end if
                          if (pspace_aab(a, b, c, m, i, k) /= 0) then
                              res14 = res14 + H1A_oo(m, j) * t3b(pspace_aab(a, b, c, m, i, k))
                          end if

                          ! diagram 16: 1/2 h2a(mnij) * t3b(abcmnk)
                          do n = m + 1, noa
                              if (pspace_aab(a, b, c, m, n, k) /= 0) then
                                res16 = res16 + H2A_oooo(m, n, i, j) * t3b(pspace_aab(a, b, c, m, n, k))
                              end if
                          end do

                          ! diagram 18: -A(ij) h2b(mcjk) * t2a(abim) 
                          res18 = res18 - I2B_ovoo(m, c, j, k) * t2a(a, b, i, m)
                          res18 = res18 + I2B_ovoo(m, c, i, k) * t2a(a, b, j, m)

                          ! diagram 20: -A(ab) h2a(amij) * t2b(bcmk)
                          res20 = res20 - I2A_vooo(a, m, i, j) * t2b(b, c, m, k)
                          res20 = res20 + I2A_vooo(b, m, i, j) * t2b(a, c, m, k)

                      end do

                      do m = 1, nob

                          ! diagram 15: -h1b(mk) * t3b(abcijm)
                          if (pspace_aab(a, b, c, i, j, m) /= 0) then
                              res15 = res15 - H1B_oo(m, k) * t3b(pspace_aab(a, b, c, i, j, m))
                          end if

                          ! diagram 17: A(ij) h2b(mnjk) * t3b(abcimn)
                          do n = 1, noa
                              if (pspace_aab(a, b, c, i, n, m) /= 0) then
                                res17 = res17 + H2B_oooo(n, m, j, k) * t3b(pspace_aab(a, b, c, i, n, m))
                              end if
                              if (pspace_aab(a, b, c, j, n, m) /= 0) then
                                res17 = res17 - H2B_oooo(n, m, i, k) * t3b(pspace_aab(a, b, c, j, n, m))
                              end if
                          end do

                          ! diagram 19: -A(ij)A(ab) h2b(amik) * t2b(bcjm)
                          res19 = res19 - I2B_vooo(a, m, i, k) * t2b(b, c, j, m)
                          res19 = res19 + I2B_vooo(b, m, i, k) * t2b(a, c, j, m)
                          res19 = res19 + I2B_vooo(a, m, j, k) * t2b(b, c, i, m)
                          res19 = res19 - I2B_vooo(b, m, j, k) * t2b(a, c, i, m)

                      end do


                      denom = fA_oo(i, i) + fA_oo(j, j) + fB_oo(k, k) - fA_vv(a, a) - fA_vv(b, b) - fB_vv(c, c)

                       val = res1 + res2 + res3 + res4 + res5 + res6&
                            + res7 + res8 + res9 + res10 + res11 + res12 + res13&
                            + res14 + res15 + res16 + res17 + res18 + res19 + res20

                      val = val/(denom - shift)

                      t3b_new(pspace_aab(a, b, c, i, j, k)) = t3b(pspace_aab(a, b, c, i, j, k)) + val
                      t3b_new(pspace_aab(b, a, c, i, j, k)) = -t3b_new(pspace_aab(a, b, c, i, j, k))
                      t3b_new(pspace_aab(a, b, c, j, i, k)) = -t3b_new(pspace_aab(a, b, c, i, j, k))
                      t3b_new(pspace_aab(b, a, c, j, i, k)) = t3b_new(pspace_aab(a, b, c, i, j, k))

                      resid(pspace_aab(a, b, c, i, j, k)) = val
                      resid(pspace_aab(b, a, c, i, j, k)) = -val
                      resid(pspace_aab(a, b, c, j, i, k)) = -val
                      resid(pspace_aab(b, a, c, j, i, k)) = val

                  end do; end do; end do;
                  end do; end do; end do;

              end subroutine update_t3b_p_opt2

              subroutine update_t3c_p_opt2(t3c_new, resid,&
                                           t2b, t2c, t3b, t3c, t3d,&
                                           pspace_aab, pspace_abb, pspace_bbb,&
                                           H1A_oo, H1A_vv, H1B_oo, H1B_vv,&
                                           H2A_oovv, H2A_voov,&
                                           H2B_oovv, H2B_vooo, H2B_ovoo, H2B_vvov, H2B_vvvo, H2B_oooo,&
                                           H2B_voov, H2B_vovo, H2B_ovov, H2B_ovvo, H2B_vvvv,&
                                           H2C_oovv, H2C_vooo, H2C_vvov, H2C_oooo, H2C_voov, H2C_vvvv,&
                                           fA_oo, fA_vv, fB_oo, fB_vv,&
                                           shift,&
                                           n3b_p, n3c_p, n3d_p,&
                                           noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub, n3b_p, n3c_p, n3d_p
                  integer, intent(in) :: pspace_aab(nua, nua, nub, noa, noa, nob),&
                                         pspace_abb(nua, nub, nub, noa, nob, nob),&
                                         pspace_bbb(nub, nub, nub, nob, nob, nob)
                  real(kind=8), intent(in) :: t2b(1:nua,1:nub,1:noa,1:nob),&
                                              t2c(1:nub,1:nub,1:nob,1:nob),&
                                              t3b(n3b_p),&
                                              t3c(n3c_p),&
                                              t3d(n3d_p),&
                                              H1A_oo(1:noa,1:noa),&
                                              H1A_vv(1:nua,1:nua),&
                                              H1B_oo(1:nob,1:nob),&
                                              H1B_vv(1:nub,1:nub),&
                                              H2A_oovv(1:noa,1:noa,1:nua,1:nua),&
                                              H2A_voov(1:nua,1:noa,1:noa,1:nua),&
                                              H2B_oovv(1:noa,1:nob,1:nua,1:nub),&
                                              H2B_vooo(1:nua,1:nob,1:noa,1:nob),&
                                              H2B_ovoo(1:noa,1:nub,1:noa,1:nob),&
                                              H2B_vvov(1:nua,1:nub,1:noa,1:nub),&
                                              H2B_vvvo(1:nua,1:nub,1:nua,1:nob),&
                                              H2B_oooo(1:noa,1:nob,1:noa,1:nob),&
                                              H2B_voov(1:nua,1:nob,1:noa,1:nub),&
                                              H2B_vovo(1:nua,1:nob,1:nua,1:nob),&
                                              H2B_ovov(1:noa,1:nub,1:noa,1:nub),&
                                              H2B_ovvo(1:noa,1:nub,1:nua,1:nob),&
                                              H2B_vvvv(1:nua,1:nub,1:nua,1:nub),&
                                              H2C_oovv(1:nob,1:nob,1:nub,1:nub),&
                                              H2C_vooo(1:nub,1:nob,1:nob,1:nob),&
                                              H2C_vvov(1:nub,1:nub,1:nob,1:nub),&
                                              H2C_oooo(1:nob,1:nob,1:nob,1:nob),&
                                              H2C_voov(1:nub,1:nob,1:nob,1:nub),&
                                              H2C_vvvv(1:nub,1:nub,1:nub,1:nub),&
                                              fA_oo(1:noa,1:noa), fA_vv(1:nua,1:nua),&
                                              fB_oo(1:nob,1:nob), fB_vv(1:nub,1:nub),&
                                              shift

                  real(kind=8), intent(out) :: t3c_new(n3c_p), resid(n3c_p)

                  integer :: i, j, k, a, b, c, m, n, e, f
                  real(kind=8) :: I2C_vooo(nub, nob, nob, nob),&
                                  I2C_vvov(nub, nub, nob, nub),&
                                  I2B_vooo(nua, nob, noa, nob),&
                                  I2B_ovoo(noa, nub, noa, nob),&
                                  I2B_vvov(nua, nub, noa, nub),&
                                  I2B_vvvo(nua, nub, nua, nob)
                  real(kind=8) :: denom, val,&
                                  res1, res2, res3, res4, res5, res6, res7, res8, res9, res10, res11, res12, res13,&
                                  res14, res15, res16, res17, res18, res19, res20

                  resid = 0.0d0
                  t3c_new = 0.0d0

                  ! VT3 intermediates
                  I2C_vooo = H2C_vooo
                  I2C_vvov = H2C_vvov
                  I2B_vooo = H2B_vooo
                  I2B_ovoo = H2B_ovoo
                  I2B_vvov = H2B_vvov
                  I2B_vvvo = H2B_vvvo

                  do a = 1, nub
                      do i = 1, nob
                          do m = 1, nob
                              do e = 1, nub

                                  ! I2C(amij) = 1/2 h2c(mnef) * t3d(aefijn) + h2b(nmfe) * t3c(feanji)
                                  do j = i + 1, nob
                                      do n = 1, nob
                                          do f = e + 1, nub
                                              if (pspace_bbb(a, e, f, i, j, n) /= 0) then
                                                I2C_vooo(a, m, i, j) = I2C_vooo(a, m, i, j) + H2C_oovv(m, n, e, f) * t3d(pspace_bbb(a, e, f, i, j, n))
                                              end if
                                          end do
                                      end do
                                      do n = 1, noa
                                          do f = 1, nua
                                              if (pspace_abb(f, e, a, n, j, i) /= 0) then
                                                I2C_vooo(a, m, i, j) = I2C_vooo(a, m, i, j) + H2B_oovv(n, m, f, e) * t3c(pspace_abb(f, e, a, n, j, i))
                                              end if
                                          end do
                                      end do
                                      I2C_vooo(a, m, j, i) = -1.0d0 * I2C_vooo(a, m, i, j)
                                  end do
                                  ! I2C(abie) = -1/2 h2c(mnef) * t3d(abfimn) - h2b(nmfe) * t3c(fbanmi)
                                  do b = a + 1, nub
                                      do n = m + 1, nob
                                          do f = 1, nub
                                              if (pspace_bbb(a, b, f, i, m, n) /= 0) then
                                                I2C_vvov(a, b, i, e) = I2C_vvov(a, b, i, e) - H2C_oovv(m, n, e, f) * t3d(pspace_bbb(a, b, f, i, m, n))
                                              end if
                                          end do
                                      end do
                                      do n = 1, noa
                                          do f = 1, nua
                                              if (pspace_abb(f, b, a, n, m, i) /= 0) then
                                                I2C_vvov(a, b, i, e) = I2C_vvov(a, b, i, e) - H2B_oovv(n, m, f, e) * t3c(pspace_abb(f, b, a, n, m, i))
                                              end if
                                          end do
                                      end do
                                      I2C_vvov(b, a, i, e) = -1.0d0 * I2C_vvov(a, b, i, e)
                                  end do

                              end do
                          end do
                      end do
                  end do

                  do a = 1, nua
                      do i = 1, noa
                          do m = 1, nob
                              do e = 1, nub

                                  ! I2B(amij) = h2b(nmfe) * t3b(afeinj) + 1/2 h2c(nmfe) * t3c(afeinj)
                                  do j = 1, nob
                                      do n = 1, noa
                                          do f = 1, nua
                                              if (pspace_aab(a, f, e, i, n, j) /= 0) then
                                                I2B_vooo(a, m, i, j) = I2B_vooo(a, m, i, j) + H2B_oovv(n, m, f, e) * t3b(pspace_aab(a, f, e, i, n, j))
                                              end if
                                          end do
                                      end do
                                      do n = 1, nob
                                          do f = e + 1, nub
                                              if (pspace_abb(a, f, e, i, n, j) /= 0) then
                                                I2B_vooo(a, m, i, j) = I2B_vooo(a, m, i, j) + H2C_oovv(n, m, f, e) * t3c(pspace_abb(a, f, e, i, n, j))
                                              end if
                                          end do
                                      end do
                                  end do
                                  ! I2B(abie) = -h2b(nmfe) * t3b(afbinm) - 1/2 h2c(mnef) * t3c(afbinm)
                                  do b = 1, nub
                                      do n = 1, noa
                                          do f = 1, nua
                                              if (pspace_aab(a, f, b, i, n, m) /= 0) then
                                                I2B_vvov(a, b, i, e) = I2B_vvov(a, b, i, e) - H2B_oovv(n, m, f, e) * t3b(pspace_aab(a, f, b, i, n, m))
                                              end if
                                          end do
                                      end do
                                      do n = m + 1, nob
                                          do f = 1, nub
                                              if (pspace_abb(a, f, b, i, n, m) /= 0) then
                                                I2B_vvov(a, b, i, e) = I2B_vvov(a, b, i, e) - H2C_oovv(n, m, f, e) * t3c(pspace_abb(a, f, b, i, n, m))
                                              end if
                                          end do
                                      end do
                                  end do

                            end do
                          end do
                      end do
                  end do


                  do a = 1, nub
                      do i = 1, nob
                          do m = 1, noa
                              do e = 1, nua

                                  ! I2B(maji) = 1/2 h2a(mnef) * t3b(efajni) + h2b(mnef) * t3c(efajni)
                                  do j = 1, noa
                                      do n = 1, noa
                                          do f = e + 1, nua
                                              if (pspace_aab(e, f, a, j, n, i) /= 0) then
                                                I2B_ovoo(m, a, j, i) = I2B_ovoo(m, a, j, i) + H2A_oovv(m, n, e, f) * t3b(pspace_aab(e, f, a, j, n, i))
                                              end if
                                          end do
                                      end do
                                      do n = 1, nob
                                          do f = 1, nub
                                              if (pspace_abb(e, f, a, j, n, i) /= 0) then
                                                I2B_ovoo(m, a, j, i) = I2B_ovoo(m, a, j, i) + H2B_oovv(m, n, e, f) * t3c(pspace_abb(e, f, a, j, n, i))
                                              end if
                                          end do
                                      end do
                                  end do
                                  ! I2B(baei) = -1/2 h2a(mnef) * t3b(bfamni) - h2b(mnef) * t3c(bfamni)
                                  do b = 1, nua
                                      do n = m + 1, noa
                                          do f = 1, nua
                                              if (pspace_aab(b, f, a, m, n, i) /= 0) then
                                                I2B_vvvo(b, a, e, i) = I2B_vvvo(b, a, e, i) - H2A_oovv(m, n, e, f) * t3b(pspace_aab(b, f, a, m, n, i))
                                              end if
                                          end do
                                      end do
                                      do n = 1, nob
                                          do f = 1, nub
                                              if (pspace_abb(b, f, a, m, n, i) /= 0) then
                                                I2B_vvvo(b, a, e, i) = I2B_vvvo(b, a, e, i) - H2B_oovv(m, n, e, f) * t3c(pspace_abb(b, f, a, m, n, i))
                                              end if
                                          end do
                                      end do
                                  end do

                              end do
                          end do
                      end do
                  end do

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
                      res14 = 0.0d0
                      res15 = 0.0d0
                      res16 = 0.0d0
                      res17 = 0.0d0
                      res18 = 0.0d0
                      res19 = 0.0d0
                      res20 = 0.0d0

                      do e = 1, nua

                          ! diagram 1: h1A(ae) * t3c(ebcijk)
                          if (pspace_abb(e, b, c, i, j, k) /= 0) then
                            res1 = res1 + H1A_vv(a, e) * t3c(pspace_abb(e, b, c, i, j, k))
                          end if

                          ! diagram 2: A(be) h1B(be) * t3c(aecijk)
                          if (pspace_abb(a, e, c, i, j, k) /= 0) then
                            res2 = res2 + H1B_vv(b, e) * t3c(pspace_abb(a, e, c, i, j, k))
                          end if
                          if (pspace_abb(a, e, b, i, j, k) /= 0) then
                            res2 = res2 - H1B_vv(c, e) * t3c(pspace_abb(a, e, b, i, j, k))
                          end if

                          ! diagram 3: 0.5 * h2C(bcef) * t3c(aefijk)
                          ! diagram 4: A(bc) h2C(abef) * t3c(efcijk)
                          do f = e + 1, nua
                              if (pspace_abb(a, e, f, i, j, k) /= 0) then
                                res3 = res3 + H2C_vvvv(b, c, e, f) * t3c(pspace_abb(a, e, f, i, j, k))
                              end if
                          end do

                          do f = 1, nua
                              if (pspace_abb(e, f, c, i, j, k) /= 0) then
                                res4 = res4 + H2B_vvvv(a, b, e, f) * t3c(pspace_abb(e, f, c, i, j, k))
                              end if
                              if (pspace_abb(e, f, b, i, j, k) /= 0) then
                                res4 = res4 - H2B_vvvv(a, c, e, f) * t3c(pspace_abb(e, f, b, i, j, k))
                              end if
                          end do
                          do f = 1, nub - nua
                              if (pspace_abb(e, f + nua, c, i, j, k) /= 0) then
                                res4 = res4 + H2B_vvvv(a, b, e, f + nua) * t3c(pspace_abb(e, f + nua, c, i, j, k))
                              end if
                              if (pspace_abb(e, f + nua, b, i, j, k) /= 0) then
                                res4 = res4 - H2B_vvvv(a, c, e, f + nua) * t3c(pspace_abb(e, f + nua, b, i, j, k))
                              end if
                          end do

                          ! diagram 5: h2A(amie) * t3c(ebcmjk)
                          ! diagram 7: A(jk)A(bc) h2B(mbej) * t3b(aecimk)
                          ! diagram 9: -A(bc) h2B(mbie) * t3c(aecmjk)
                          do m = 1, noa
                              if (pspace_abb(e, b, c, m, j, k) /= 0) then
                                res5 = res5 + H2A_voov(a, m, i, e) * t3c(pspace_abb(e, b, c, m, j, k))
                              end if

                              if (pspace_aab(a, e, c, i, m, k) /= 0) then
                                res7 = res7 + H2B_ovvo(m, b, e, j) * t3b(pspace_aab(a, e, c, i, m, k))
                              end if
                              if (pspace_aab(a, e, b, i, m, k) /= 0) then
                                res7 = res7 - H2B_ovvo(m, c, e, j) * t3b(pspace_aab(a, e, b, i, m, k))
                              end if
                              if (pspace_aab(a, e, c, i, m, j) /= 0) then
                                res7 = res7 - H2B_ovvo(m, b, e, k) * t3b(pspace_aab(a, e, c, i, m, j))
                              end if
                              if (pspace_aab(a, e, b, i, m, j) /= 0) then
                                res7 = res7 + H2B_ovvo(m, c, e, k) * t3b(pspace_aab(a, e, b, i, m, j))
                              end if

                              if (pspace_abb(a, e, c, m, j, k) /= 0) then
                                res9 = res9 - H2B_ovov(m, b, i, e) * t3c(pspace_abb(a, e, c, m, j, k))
                              end if
                              if (pspace_abb(a, e, b, m, j, k) /= 0) then
                                res9 = res9 + H2B_ovov(m, c, i, e) * t3c(pspace_abb(a, e, b, m, j, k))
                              end if
                          end do

                          ! diagram 6: h2B(amie) * t3d(ebcmjk)
                          ! diagram 8: A(jk)A(bc) h2C(bmje) * t3c(aecimk)
                          ! diagram 10: -A(jk) h2B(amej) * t3c(ebcimk)
                          do m = 1, nob
                              if (pspace_bbb(e, b, c, m, j, k) /= 0) then
                                res6 = res6 + H2B_voov(a, m, i, e) * t3d(pspace_bbb(e, b, c, m, j, k))
                              end if

                              if (pspace_abb(a, e, c, i, m, k) /= 0) then
                                res8 = res8 + H2C_voov(b, m, j, e) * t3c(pspace_abb(a, e, c, i, m, k))
                              end if
                              if (pspace_abb(a, e, b, i, m, k) /= 0) then
                                res8 = res8 - H2C_voov(c, m, j, e) * t3c(pspace_abb(a, e, b, i, m, k))
                              end if
                              if (pspace_abb(a, e, c, i, m, j) /= 0) then
                                res8 = res8 - H2C_voov(b, m, k, e) * t3c(pspace_abb(a, e, c, i, m, j))
                              end if
                              if (pspace_abb(a, e, b, i, m, j) /= 0) then
                                res8 = res8 + H2C_voov(c, m, k, e) * t3c(pspace_abb(a, e, b, i, m, j))
                              end if

                              if (pspace_abb(e, b, c, i, m, k) /= 0) then
                                res10 = res10 - H2B_vovo(a, m, e, j) * t3c(pspace_abb(e, b, c, i, m, k))
                              end if
                              if (pspace_abb(e, b, c, i, m, j) /= 0) then
                                res10 = res10 + H2B_vovo(a, m, e, k) * t3c(pspace_abb(e, b, c, i, m, j))
                              end if
                          end do

                          ! diagram 11: A(bc) h2B(abie) * t2c(ecjk)
                          res11 = res11 + I2B_vvov(a, b, i, e) * t2c(e, c, j, k)
                          res11 = res11 - I2B_vvov(a, c, i, e) * t2c(e, b, j, k)
                          ! diagram 12: A(jk) h2C(cbke) * t2b(aeij)
                          res12 = res12 + I2C_vvov(c, b, k, e) * t2b(a, e, i, j)
                          res12 = res12 - I2C_vvov(c, b, j, e) * t2b(a, e, i, k)
                          ! diagram 13: A(jk)A(bc) h2B(abej) * t2b(ecik)
                          res13 = res13 + I2B_vvvo(a, b, e, j) * t2b(e, c, i, k)
                          res13 = res13 - I2B_vvvo(a, b, e, k) * t2b(e, c, i, j)
                          res13 = res13 - I2B_vvvo(a, c, e, j) * t2b(e, b, i, k)
                          res13 = res13 + I2B_vvvo(a, c, e, k) * t2b(e, b, i, j)

                      end do

                      do e = 1, nub - nua

                          ! diagram 2: A(be) h1B(be) * t3c(aecijk)
                          if (pspace_abb(a, e + nua, c, i, j, k) /= 0) then
                            res2 = res2 + H1B_vv(b, e + nua) * t3c(pspace_abb(a, e + nua, c, i, j, k))
                          end if
                          if (pspace_abb(a, e + nua, b, i, j, k) /= 0) then
                            res2 = res2 - H1B_vv(c, e + nua) * t3c(pspace_abb(a, e + nua, b, i, j, k))
                          end if

                          ! diagram 3: 0.5 * h2C(bcef) * t3c(aefijk)
                          do f = 1, nua
                              if (pspace_abb(a, e + nua, f, i, j, k) /= 0) then
                                res3 = res3 + H2C_vvvv(b, c, e + nua, f) * t3c(pspace_abb(a, e + nua, f, i, j, k))
                              end if
                          end do
                          do f = 1, nub - nua
                              if (pspace_abb(a, e + nua, f + nua, i, j, k) /= 0) then
                                res3 = res3 + 0.5d0 * H2C_vvvv(b, c, e + nua, f + nua) * t3c(pspace_abb(a, e + nua, f + nua, i, j, k))
                              end if
                          end do

                          ! diagram 9: -A(bc) h2B(mbie) * t3c(aecmjk)
                          do m = 1, noa
                              if (pspace_abb(a, e + nua, c, m, j, k) /= 0) then
                                res9 = res9 - H2B_ovov(m, b, i, e + nua) * t3c(pspace_abb(a, e + nua, c, m, j, k))
                              end if
                              if (pspace_abb(a, e + nua, b, m, j, k) /= 0) then
                                res9 = res9 + H2B_ovov(m, c, i, e + nua) * t3c(pspace_abb(a, e + nua, b, m, j, k))
                              end if
                          end do

                          ! diagram 6: h2B(amie) * t3d(ebcmjk)
                          ! diagram 8: A(jk)A(bc) h2C(bmje) * t3c(aecimk)
                          do m = 1, nob
                              if (pspace_bbb(e + nua, b, c, m, j, k) /= 0) then
                                res6 = res6 + H2B_voov(a, m, i, e + nua) * t3d(pspace_bbb(e + nua, b, c, m, j, k))
                              end if

                              if (pspace_abb(a, e + nua, c, i, m, k) /= 0) then
                                res8 = res8 + H2C_voov(b, m, j, e + nua) * t3c(pspace_abb(a, e + nua, c, i, m, k))
                              end if
                              if (pspace_abb(a, e + nua, b, i, m, k) /= 0) then
                                res8 = res8 - H2C_voov(c, m, j, e + nua) * t3c(pspace_abb(a, e + nua, b, i, m, k))
                              end if
                              if (pspace_abb(a, e + nua, c, i, m, j) /= 0) then
                                res8 = res8 - H2C_voov(b, m, k, e + nua) * t3c(pspace_abb(a, e + nua, c, i, m, j))
                              end if
                              if (pspace_abb(a, e + nua, b, i, m, j) /= 0) then
                                res8 = res8 + H2C_voov(c, m, k, e + nua) * t3c(pspace_abb(a, e + nua, b, i, m, j))
                              end if
                          end do

                          ! diagram 11: A(bc) h2B(abie) * t2c(ecjk)
                          res11 = res11 + I2B_vvov(a, b, i, e + nua) * t2c(e + nua, c, j, k)
                          res11 = res11 - I2B_vvov(a, c, i, e + nua) * t2c(e + nua, b, j, k)
                          ! diagram 12: A(jk) h2C(cbke) * t2b(aeij)
                          res12 = res12 + I2C_vvov(c, b, k, e + nua) * t2b(a, e + nua, i, j)
                          res12 = res12 - I2C_vvov(c, b, j, e + nua) * t2b(a, e + nua, i, k)

                      end do

                      do m = 1, noa

                          ! diagram 14: -h1a(mi) * t3c(abcmjk)
                          if (pspace_abb(a, b, c, m, j, k) /= 0) then
                              res14 = res14 - H1A_oo(m, i) * t3c(pspace_abb(a, b, c, m, j, k))
                          end if

                          ! diagram 16: A(jk) h2b(mnij) * t3c(abcmnk)
                          do n = 1, nob
                              if (pspace_abb(a, b, c, m, n, k) /= 0) then
                                  res16 = res16 + H2B_oooo(m, n, i, j) * t3c(pspace_abb(a, b, c, m, n, k))
                              end if
                              if (pspace_abb(a, b, c, m, n, j) /= 0) then
                                  res16 = res16 - H2B_oooo(m, n, i, k) * t3c(pspace_abb(a, b, c, m, n, j))
                              end if
                          end do

                          ! diagram 18: -A(kj)A(bc) h2b(mbij) * t2b(acmk)
                          res18 = res18 - I2B_ovoo(m, b, i, j) * t2b(a, c, m, k)
                          res18 = res18 + I2B_ovoo(m, c, i, j) * t2b(a, b, m, k)
                          res18 = res18 + I2B_ovoo(m, b, i, k) * t2b(a, c, m, j)
                          res18 = res18 - I2B_ovoo(m, c, i, k) * t2b(a, b, m, j)

                      end do

                      do m = 1, nob

                          ! diagram 15: -A(jk) h1b(mj) * t3c(abcimk)
                          if (pspace_abb(a, b, c, i, m, k) /= 0) then
                              res15 = res15 - H1B_oo(m, j) * t3c(pspace_abb(a, b, c, i, m, k))
                          end if
                          if (pspace_abb(a, b, c, i, m, j) /= 0) then
                              res15 = res15 + H1B_oo(m, k) * t3c(pspace_abb(a, b, c, i, m, j))
                          end if

                          ! diagram 17: 1/2 h2c(mnjk) * t3c(abcimn)
                          do n = m + 1, nob
                              if (pspace_abb(a, b, c, i, m, n) /= 0) then
                                  res17 = res17 + H2C_oooo(m, n, j, k) * t3c(pspace_abb(a, b, c, i, m, n))
                              end if
                          end do

                          ! diagram 19: -A(jk) h2b(amij) * t2c(bcmk)
                          res19 = res19 - I2B_vooo(a, m, i, j) * t2c(b, c, m, k)
                          res19 = res19 + I2B_vooo(a, m, i, k) * t2c(b, c, m, j)

                          ! diagram 20: -A(bc) h2c(cmkj) * t2b(abim)
                          res20 = res20 - I2C_vooo(c, m, k, j) * t2b(a, b, i, m)
                          res20 = res20 + I2C_vooo(b, m, k, j) * t2b(a, c, i, m)

                      end do

                      denom = fA_oo(i, i) + fB_oo(j, j) + fB_oo(k, k) - fA_vv(a, a) - fB_vv(b, b) - fB_vv(c, c)

                      val = res1 + res2 + res3 + res4 + res5 + res6 + res7 + res8 + res9 + res10 + res11 + res12 + res13&
                              + res14 + res15 + res16 + res17 + res18 + res19 + res20

                      val = val/(denom - shift)

                      t3c_new(pspace_abb(a, b, c, i, j, k)) = t3c(pspace_abb(a, b, c, i, j, k)) + val
                      t3c_new(pspace_abb(a, c, b, i, j, k)) = -t3c_new(pspace_abb(a, b, c, i, j, k))
                      t3c_new(pspace_abb(a, b, c, i, k, j)) = -t3c_new(pspace_abb(a, b, c, i, j, k))
                      t3c_new(pspace_abb(a, c, b, i, k, j)) = t3c_new(pspace_abb(a, b, c, i, j, k))

                      resid(pspace_abb(a, b, c, i, j, k)) = val
                      resid(pspace_abb(a, c, b, i, j, k)) = -val
                      resid(pspace_abb(a, b, c, i, k, j)) = -val
                      resid(pspace_abb(a, c, b, i, k, j)) = val

                  end do; end do; end do;
                  end do; end do; end do;

              end subroutine update_t3c_p_opt2

              subroutine update_t3d_p_opt2(t3d_new, resid,&
                                           t2c, t3c, t3d,&
                                           pspace_abb, pspace_bbb,&
                                           H1B_oo, H1B_vv,&
                                           H2B_oovv, H2B_ovvo,&
                                           H2C_oovv, H2C_vooo, H2C_vvov, H2C_oooo, H2C_voov, H2C_vvvv,&
                                           fB_oo, fB_vv,&
                                           shift,&
                                           n3c_p, n3d_p,&
                                           noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub, n3c_p, n3d_p
                  real(kind=8), intent(in) :: t2c(nub, nub, nob, nob),&
                                              t3c(n3c_p),&
                                              t3d(n3d_p),&
                                              H1B_oo(nob, nob), H1B_vv(nub, nub),&
                                              H2B_oovv(noa, nob, nua, nub),&
                                              H2B_ovvo(noa, nub, nua, nob),&
                                              H2C_oovv(nob, nob, nub, nub),&
                                              H2C_vooo(nub, nob, nob, nob),&
                                              H2C_vvov(nub, nub, nob, nub),&
                                              H2C_oooo(nob, nob, nob, nob),&
                                              H2C_voov(nub, nob, nob, nub),&
                                              H2C_vvvv(nub, nub, nub, nub),&
                                              fB_vv(nub, nub), fB_oo(nob, nob),&
                                              shift
                  integer, intent(in) :: pspace_abb(nua, nub, nub, noa, nob, nob),&
                                         pspace_bbb(nub, nub, nub, nob, nob, nob)

                  real(kind=8), intent(out) :: t3d_new(n3d_p), resid(n3d_p)

                  real(kind=8) :: val, denom
                  real(kind=8) :: I2C_vooo(nub, nob, nob, nob),&
                                  I2C_vvov(nub, nub, nob, nub)
                  real(kind=8) :: res1, res2, res3, res4, res5, res6, res7, res8
                  integer :: a, b, c, i, j, k, e, f, m, n

                  t3d_new = 0.0d0
                  resid = 0.0d0

                  ! compute VT3 intermediates
                  I2C_vooo = H2C_vooo
                  I2C_vvov = H2C_vvov
                  do a = 1, nub
                      do i = 1, nob
                          do m = 1, nob
                              do e = 1, nub

                                  ! I2C(amij) = 1/2 h2c(mnef) * t3d(aefijn) + h2b(nmfe) * t3c(feanji)
                                  do j = i + 1, nob
                                      do n = 1, nob
                                          do f = e + 1, nub
                                              if (pspace_bbb(a, e, f, i, j, n) /= 0) then
                                                I2C_vooo(a, m, i, j) = I2C_vooo(a, m, i, j) + H2C_oovv(m, n, e, f) * t3d(pspace_bbb(a, e, f, i, j, n))
                                              end if
                                          end do
                                      end do
                                      do n = 1, noa
                                          do f = 1, nua
                                              if (pspace_abb(f, e, a, n, j, i) /= 0) then
                                                I2C_vooo(a, m, i, j) = I2C_vooo(a, m, i, j) + H2B_oovv(n, m, f, e) * t3c(pspace_abb(f, e, a, n, j, i))
                                              end if
                                          end do
                                      end do
                                      I2C_vooo(a, m, j, i) = -1.0d0 * I2C_vooo(a, m, i, j)
                                  end do
                                  ! I2C(abie) = -1/2 h2c(mnef) * t3d(abfimn) - h2b(nmfe) * t3c(fbanmi)
                                  do b = a + 1, nub
                                      do n = m + 1, nob
                                          do f = 1, nub
                                              if (pspace_bbb(a, b, f, i, m, n) /= 0) then
                                                I2C_vvov(a, b, i, e) = I2C_vvov(a, b, i, e) - H2C_oovv(m, n, e, f) * t3d(pspace_bbb(a, b, f, i, m, n))
                                              end if
                                          end do
                                      end do
                                      do n = 1, noa
                                          do f = 1, nua
                                              if (pspace_abb(f, b, a, n, m, i) /= 0) then
                                                I2C_vvov(a, b, i, e) = I2C_vvov(a, b, i, e) - H2B_oovv(n, m, f, e) * t3c(pspace_abb(f, b, a, n, m, i))
                                              end if
                                          end do
                                      end do
                                      I2C_vvov(b, a, i, e) = -1.0d0 * I2C_vvov(a, b, i, e)
                                  end do

                              end do
                          end do
                      end do
                  end do

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
                              if (pspace_bbb(e, b, c, i, j, k) /= 0) then
                                res1 = res1 + H1B_vv(a, e) * t3d(pspace_bbb(e, b, c, i, j, k))
                              end if
                              if (pspace_bbb(e, a, c, i, j, k) /= 0) then
                                res1 = res1 - H1B_vv(b, e) * t3d(pspace_bbb(e, a, c, i, j, k))
                              end if
                              if (pspace_bbb(e, b, a, i, j, k) /= 0) then
                                res1 = res1 - H1B_vv(c, e) * t3d(pspace_bbb(e, b, a, i, j, k))
                              end if

                              ! diagram 2: 1/2 A(c/ab) h2C(abef) * t3d(efcijk)
                              do f = e + 1, nua
                                  if (pspace_bbb(e, f, c, i, j, k) /= 0) then
                                    res2 = res2 + H2C_vvvv(a, b, e, f) * t3d(pspace_bbb(e, f, c, i, j, k))
                                  end if
                                  if (pspace_bbb(e, f, a, i, j, k) /= 0) then
                                    res2 = res2 - H2C_vvvv(c, b, e, f) * t3d(pspace_bbb(e, f, a, i, j, k))
                                  end if
                                  if (pspace_bbb(e, f, b, i, j, k) /= 0) then
                                    res2 = res2 - H2C_vvvv(a, c, e, f) * t3d(pspace_bbb(e, f, b, i, j, k))
                                  end if
                              end do

                              ! diagram 3: A(i/jk)A(a/bc) h2B(maei) * t3c(ebcmjk)
                              do m = 1, noa
                                  if (pspace_abb(e, b, c, m, j, k) /= 0) then
                                    res3 = res3 + H2B_ovvo(m, a, e, i) * t3c(pspace_abb(e, b, c, m, j, k))
                                  end if
                                  if (pspace_abb(e, b, c, m, i, k) /= 0) then
                                    res3 = res3 - H2B_ovvo(m, a, e, j) * t3c(pspace_abb(e, b, c, m, i, k))
                                  end if
                                  if (pspace_abb(e, b, c, m, j, i) /= 0) then
                                    res3 = res3 - H2B_ovvo(m, a, e, k) * t3c(pspace_abb(e, b, c, m, j, i))
                                  end if
                                  if (pspace_abb(e, a, c, m, j, k) /= 0) then
                                    res3 = res3 - H2B_ovvo(m, b, e, i) * t3c(pspace_abb(e, a, c, m, j, k))
                                  end if
                                  if (pspace_abb(e, a, c, m, i, k) /= 0) then
                                    res3 = res3 + H2B_ovvo(m, b, e, j) * t3c(pspace_abb(e, a, c, m, i, k))
                                  end if
                                  if (pspace_abb(e, a, c, m, j, i) /= 0) then
                                    res3 = res3 + H2B_ovvo(m, b, e, k) * t3c(pspace_abb(e, a, c, m, j, i))
                                  end if
                                  if (pspace_abb(e, b, a, m, j, k) /= 0) then
                                    res3 = res3 - H2B_ovvo(m, c, e, i) * t3c(pspace_abb(e, b, a, m, j, k))
                                  end if
                                  if (pspace_abb(e, b, a, m, i, k) /= 0) then
                                    res3 = res3 + H2B_ovvo(m, c, e, j) * t3c(pspace_abb(e, b, a, m, i, k))
                                  end if
                                  if (pspace_abb(e, b, a, m, j, i) /= 0) then
                                    res3 = res3 + H2B_ovvo(m, c, e, k) * t3c(pspace_abb(e, b, a, m, j, i))
                                  end if
                              end do

                              ! diagram 4: A(i/jk)A(a/bc) h2C(amie) * t3d(ebcmjk)
                              do m = 1, nob
                                  if (pspace_bbb(e, b, c, m, j, k) /= 0) then
                                    res4 = res4 + H2C_voov(a, m, i, e) * t3d(pspace_bbb(e, b, c, m, j, k))
                                  end if
                                  if (pspace_bbb(e, b, c, m, i, k) /= 0) then
                                    res4 = res4 - H2C_voov(a, m, j, e) * t3d(pspace_bbb(e, b, c, m, i, k))
                                  end if
                                  if (pspace_bbb(e, b, c, m, j, i) /= 0) then
                                    res4 = res4 - H2C_voov(a, m, k, e) * t3d(pspace_bbb(e, b, c, m, j, i))
                                  end if
                                  if (pspace_bbb(e, a, c, m, j, k) /= 0) then
                                    res4 = res4 - H2C_voov(b, m, i, e) * t3d(pspace_bbb(e, a, c, m, j, k))
                                  end if
                                  if (pspace_bbb(e, a, c, m, i, k) /= 0) then
                                    res4 = res4 + H2C_voov(b, m, j, e) * t3d(pspace_bbb(e, a, c, m, i, k))
                                  end if
                                  if (pspace_bbb(e, a, c, m, j, i) /= 0) then
                                    res4 = res4 + H2C_voov(b, m, k, e) * t3d(pspace_bbb(e, a, c, m, j, i))
                                  end if
                                  if (pspace_bbb(e, b, a, m, j, k) /= 0) then
                                    res4 = res4 - H2C_voov(c, m, i, e) * t3d(pspace_bbb(e, b, a, m, j, k))
                                  end if
                                  if (pspace_bbb(e, b, a, m, i, k) /= 0) then
                                    res4 = res4 + H2C_voov(c, m, j, e) * t3d(pspace_bbb(e, b, a, m, i, k))
                                  end if
                                  if (pspace_bbb(e, b, a, m, j, i) /= 0) then
                                    res4 = res4 + H2C_voov(c, m, k, e) * t3d(pspace_bbb(e, b, a, m, j, i))
                                  end if
                              end do

                              ! diagram 5: A(i/jk)A(c/ab) h2C(abie) * t2c(ecjk)
                              res5 = res5 + I2C_vvov(a, b, i, e) * t2c(e, c, j, k)
                              res5 = res5 - I2C_vvov(c, b, i, e) * t2c(e, a, j, k)
                              res5 = res5 - I2C_vvov(a, c, i, e) * t2c(e, b, j, k)
                              res5 = res5 - I2C_vvov(a, b, j, e) * t2c(e, c, i, k)
                              res5 = res5 + I2C_vvov(c, b, j, e) * t2c(e, a, i, k)
                              res5 = res5 + I2C_vvov(a, c, j, e) * t2c(e, b, i, k)
                              res5 = res5 - I2C_vvov(a, b, k, e) * t2c(e, c, j, i)
                              res5 = res5 + I2C_vvov(c, b, k, e) * t2c(e, a, j, i)
                              res5 = res5 + I2C_vvov(a, c, k, e) * t2c(e, b, j, i)

                          end do

                          do e = 1, nub - nua
                              ! diagram 1: A(a/bc) h1B(ae) * t3d(ebcijk)
                              if (pspace_bbb(e + nua, b, c, i, j, k) /= 0) then
                                res1 = res1 + H1B_vv(a, e + nua) * t3d(pspace_bbb(e + nua, b, c, i, j, k))
                              end if
                              if (pspace_bbb(e + nua, a, c, i, j, k) /= 0) then
                                res1 = res1 - H1B_vv(b, e + nua) * t3d(pspace_bbb(e + nua, a, c, i, j, k))
                              end if
                              if (pspace_bbb(e + nua, b, a, i, j, k) /= 0) then
                                res1 = res1 - H1B_vv(c, e + nua) * t3d(pspace_bbb(e + nua, b, a, i, j, k))
                              end if

                              ! diagram 2: 1/2 A(c/ab) h2C(abef) * t3d(efcijk)
                              do f = e + 1, nub - nua
                                  if (pspace_bbb(e + nua, f + nua, c, i, j, k) /= 0) then
                                    res2 = res2 + H2C_vvvv(a, b, e + nua, f + nua) * t3d(pspace_bbb(e + nua, f + nua, c, i, j, k))
                                  end if
                                  if (pspace_bbb(e + nua, f + nua, a, i, j, k) /= 0) then
                                    res2 = res2 - H2C_vvvv(c, b, e + nua, f + nua) * t3d(pspace_bbb(e + nua, f + nua, a, i, j, k))
                                  end if
                                  if (pspace_bbb(e + nua, f + nua, b, i, j, k) /= 0) then
                                    res2 = res2 - H2C_vvvv(a, c, e + nua, f + nua) * t3d(pspace_bbb(e + nua, f + nua, b, i, j, k))
                                  end if
                              end do

                              ! diagram 4: A(i/jk)A(a/bc) h2C(amie) * t3d(ebcmjk)
                              do m = 1, nob
                                  if (pspace_bbb(e + nua, b, c, m, j, k) /= 0) then
                                    res4 = res4 + H2C_voov(a, m, i, e + nua) * t3d(pspace_bbb(e + nua, b, c, m, j, k))
                                  end if
                                  if (pspace_bbb(e + nua, b, c, m, i, k) /= 0) then
                                    res4 = res4 - H2C_voov(a, m, j, e + nua) * t3d(pspace_bbb(e + nua, b, c, m, i, k))
                                  end if
                                  if (pspace_bbb(e + nua, b, c, m, j, i) /= 0) then
                                    res4 = res4 - H2C_voov(a, m, k, e + nua) * t3d(pspace_bbb(e + nua, b, c, m, j, i))
                                  end if
                                  if (pspace_bbb(e + nua, a, c, m, j, k) /= 0) then
                                    res4 = res4 - H2C_voov(b, m, i, e + nua) * t3d(pspace_bbb(e + nua, a, c, m, j, k))
                                  end if
                                  if (pspace_bbb(e + nua, a, c, m, i, k) /= 0) then
                                    res4 = res4 + H2C_voov(b, m, j, e + nua) * t3d(pspace_bbb(e + nua, a, c, m, i, k))
                                  end if
                                  if (pspace_bbb(e + nua, a, c, m, j, i) /= 0) then
                                    res4 = res4 + H2C_voov(b, m, k, e + nua) * t3d(pspace_bbb(e + nua, a, c, m, j, i))
                                  end if
                                  if (pspace_bbb(e + nua, b, a, m, j, k) /= 0) then
                                    res4 = res4 - H2C_voov(c, m, i, e + nua) * t3d(pspace_bbb(e + nua, b, a, m, j, k))
                                  end if
                                  if (pspace_bbb(e + nua, b, a, m, i, k) /= 0) then
                                    res4 = res4 + H2C_voov(c, m, j, e + nua) * t3d(pspace_bbb(e + nua, b, a, m, i, k))
                                  end if
                                  if (pspace_bbb(e + nua, b, a, m, j, i) /= 0) then
                                    res4 = res4 + H2C_voov(c, m, k, e + nua) * t3d(pspace_bbb(e + nua, b, a, m, j, i))
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

                          do m = 1, nob

                              ! diagram 6: -A(i/jk) h1b(mi) * t3d(abcmjk)
                              if (pspace_bbb(a, b, c, m, j, k) /= 0) then
                                  res6 = res6 - H1B_oo(m, i) * t3d(pspace_bbb(a, b, c, m, j, k))
                              end if
                              if (pspace_bbb(a, b, c, m, i, k) /= 0) then
                                  res6 = res6 + H1B_oo(m, j) * t3d(pspace_bbb(a, b, c, m, i, k))
                              end if
                              if (pspace_bbb(a, b, c, m, j, i) /= 0) then
                                  res6 = res6 + H1B_oo(m, k) * t3d(pspace_bbb(a, b, c, m, j, i))
                              end if

                              ! diagram 7: 1/2 A(k/ij) h2c(mnij) * t3d(abcmnk)
                              do n = m + 1, nob
                                  if (pspace_bbb(a, b, c, m, n, k) /= 0) then
                                      res7 = res7 + H2C_oooo(m, n, i, j) * t3d(pspace_bbb(a, b, c, m, n, k))
                                  end if
                                  if (pspace_bbb(a, b, c, m, n, i) /= 0) then
                                      res7 = res7 - H2C_oooo(m, n, k, j) * t3d(pspace_bbb(a, b, c, m, n, i))
                                  end if
                                  if (pspace_bbb(a, b, c, m, n, j) /= 0) then
                                      res7 = res7 - H2C_oooo(m, n, i, k) * t3d(pspace_bbb(a, b, c, m, n, j))
                                  end if
                              end do

                              ! diagram 8: - A(k/ij)A(a/bc) h2c(amij) * t2c(bcmk)
                              res8 = res8 - I2C_vooo(a, m, i, j) * t2c(b, c, m, k)
                              res8 = res8 + I2C_vooo(a, m, i, k) * t2c(b, c, m, j)
                              res8 = res8 + I2C_vooo(a, m, k, j) * t2c(b, c, m, i)
                              res8 = res8 + I2C_vooo(b, m, i, j) * t2c(a, c, m, k)
                              res8 = res8 - I2C_vooo(b, m, i, k) * t2c(a, c, m, j)
                              res8 = res8 - I2C_vooo(b, m, k, j) * t2c(a, c, m, i)
                              res8 = res8 + I2C_vooo(c, m, i, j) * t2c(b, a, m, k)
                              res8 = res8 - I2C_vooo(c, m, i, k) * t2c(b, a, m, j)
                              res8 = res8 - I2C_vooo(c, m, k, j) * t2c(b, a, m, i)

                          end do

                          denom = fB_oo(I, I) + fB_oo(J, J) + fB_oo(K, K) - fB_vv(A, A) - fB_vv(B, B) - fB_vv(C, C)

                          val = val + res1 + res2 + res3 + res4 + res5 + res6 + res7 + res8

                          val = val/(denom - shift)

                          t3d_new(pspace_bbb(A,B,C,I,J,K)) = t3d(pspace_bbb(A,B,C,I,J,K)) + val
                          t3d_new(pspace_bbb(A,B,C,K,I,J)) = t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(A,B,C,J,K,I)) = t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(A,B,C,I,K,J)) = -t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(A,B,C,J,I,K)) = -t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(A,B,C,K,J,I)) = -t3d_new(pspace_bbb(A,B,C,I,J,K))

                          t3d_new(pspace_bbb(B,A,C,I,J,K)) = -t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(B,A,C,K,I,J)) = -t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(B,A,C,J,K,I)) = -t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(B,A,C,I,K,J)) = t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(B,A,C,J,I,K)) = t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(B,A,C,K,J,I)) = t3d_new(pspace_bbb(A,B,C,I,J,K))

                          t3d_new(pspace_bbb(A,C,B,I,J,K)) = -t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(A,C,B,K,I,J)) = -t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(A,C,B,J,K,I)) = -t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(A,C,B,I,K,J)) = t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(A,C,B,J,I,K)) = t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(A,C,B,K,J,I)) = t3d_new(pspace_bbb(A,B,C,I,J,K))

                          t3d_new(pspace_bbb(C,B,A,I,J,K)) = -t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(C,B,A,K,I,J)) = -t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(C,B,A,J,K,I)) = -t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(C,B,A,I,K,J)) = t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(C,B,A,J,I,K)) = t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(C,B,A,K,J,I)) = t3d_new(pspace_bbb(A,B,C,I,J,K))

                          t3d_new(pspace_bbb(B,C,A,I,J,K)) = t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(B,C,A,K,I,J)) = t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(B,C,A,J,K,I)) = t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(B,C,A,I,K,J)) = -t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(B,C,A,J,I,K)) = -t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(B,C,A,K,J,I)) = -t3d_new(pspace_bbb(A,B,C,I,J,K))

                          t3d_new(pspace_bbb(C,A,B,I,J,K)) = t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(C,A,B,K,I,J)) = t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(C,A,B,J,K,I)) = t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(C,A,B,I,K,J)) = -t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(C,A,B,J,I,K)) = -t3d_new(pspace_bbb(A,B,C,I,J,K))
                          t3d_new(pspace_bbb(C,A,B,K,J,I)) = -t3d_new(pspace_bbb(A,B,C,I,J,K))

                          resid(pspace_bbb(A,B,C,I,J,K)) = val
                          resid(pspace_bbb(A,B,C,K,I,J)) = val
                          resid(pspace_bbb(A,B,C,J,K,I)) = val
                          resid(pspace_bbb(A,B,C,I,K,J)) = -val
                          resid(pspace_bbb(A,B,C,J,I,K)) = -val
                          resid(pspace_bbb(A,B,C,K,J,I)) = -val
                          resid(pspace_bbb(B,C,A,I,J,K)) = val
                          resid(pspace_bbb(B,C,A,K,I,J)) = val
                          resid(pspace_bbb(B,C,A,J,K,I)) = val
                          resid(pspace_bbb(B,C,A,I,K,J)) = -val
                          resid(pspace_bbb(B,C,A,J,I,K)) = -val
                          resid(pspace_bbb(B,C,A,K,J,I)) = -val
                          resid(pspace_bbb(C,A,B,I,J,K)) = val
                          resid(pspace_bbb(C,A,B,K,I,J)) = val
                          resid(pspace_bbb(C,A,B,J,K,I)) = val
                          resid(pspace_bbb(C,A,B,I,K,J)) = -val
                          resid(pspace_bbb(C,A,B,J,I,K)) = -val
                          resid(pspace_bbb(C,A,B,K,J,I)) = -val
                          resid(pspace_bbb(A,C,B,I,J,K)) = -val
                          resid(pspace_bbb(A,C,B,K,I,J)) = -val
                          resid(pspace_bbb(A,C,B,J,K,I)) = -val
                          resid(pspace_bbb(A,C,B,I,K,J)) = val
                          resid(pspace_bbb(A,C,B,J,I,K)) = val
                          resid(pspace_bbb(A,C,B,K,J,I)) = val
                          resid(pspace_bbb(B,A,C,I,J,K)) = -val
                          resid(pspace_bbb(B,A,C,K,I,J)) = -val
                          resid(pspace_bbb(B,A,C,J,K,I)) = -val
                          resid(pspace_bbb(B,A,C,I,K,J)) = val
                          resid(pspace_bbb(B,A,C,J,I,K)) = val
                          resid(pspace_bbb(B,A,C,K,J,I)) = val
                          resid(pspace_bbb(C,B,A,I,J,K)) = -val
                          resid(pspace_bbb(C,B,A,K,I,J)) = -val
                          resid(pspace_bbb(C,B,A,J,K,I)) = -val
                          resid(pspace_bbb(C,B,A,I,K,J)) = val
                          resid(pspace_bbb(C,B,A,J,I,K)) = val
                          resid(pspace_bbb(C,B,A,K,J,I)) = val

                  end do; end do; end do;
                  end do; end do; end do;

              end subroutine update_t3d_p_opt2




end module ccp_opt_loops_v2
