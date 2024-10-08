module ccp_intermediates

    implicit none

    contains

              subroutine build_intermediates(H2A_vvov, H2A_vooo, H2B_vvov, H2B_vvvo, H2B_vooo, H2B_ovoo, H2C_vvov, H2C_vooo,&
                                             t3a, t3b, t3c, t3d,&
                                             pspace_aaa, pspace_aab, pspace_abb, pspace_bbb,&
                                             H2A_oovv, H2B_oovv, H2C_oovv,&
                                             H1A_ov, H1B_ov,&
                                             noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: pspace_aaa(nua, nua, nua, noa, noa, noa), pspace_aab(nua, nua, nub, noa, noa, nob),&
                                         pspace_abb(nua, nub, nub, noa, nob, nob), pspace_bbb(nub, nub, nub, nob, nob, nob)
                  real(kind=8), intent(in) :: t3a(nua, nua, nua, noa, noa, noa), t3b(nua, nua, nub, noa, noa, nob),&
                                              t3c(nua, nub, nub, noa, nob, nob), t3d(nub, nub, nub, nob, nob, nob)
                  real(kind=8), intent(in) :: H2A_oovv(noa, noa, nua, nua), H2B_oovv(noa, nob, nua, nub), H2C_oovv(nob, nob, nub, nub),&
                                              H1A_ov(noa, nua), H1B_ov(nob, nub)

                  real(kind=8), intent(inout) :: H2A_vvov(nua, nua, noa, nua)
                  !f2py intent(in,out) H2A_vvov(nua, nua, noa, nua)
                  real(kind=8), intent(inout) :: H2A_vooo(nua, noa, noa, noa)
                  !f2py intent(in,out) H2A_vooo(nua, noa, noa, noa)
                  real(kind=8), intent(inout) :: H2B_vvov(nua, nub, noa, nub)
                  !f2py intent(in,out) H2B_vvov(nua, nub, noa, nub)
                  real(kind=8), intent(inout) :: H2B_vvvo(nua, nub, nua, nob)
                  !f2py intent(in,out) H2B_vvvo(nua, nub, nua, nob)
                  real(kind=8), intent(inout) :: H2B_vooo(nua, nob, noa ,nob)
                  !f2py intent(in,out) H2B_vooo(nua, nob, noa, nob)
                  real(kind=8), intent(inout) :: H2B_ovoo(noa, nub, noa, nob)
                  !f2py intent(in,out) H2B_ovoo(noa, nub, noa, nob)
                  real(kind=8), intent(inout) :: H2C_vvov(nub, nub, nob, nub)
                  !f2py intent(in,out) H2C_vvov(nub, nub, nob, nub)
                  real(kind=8), intent(inout) :: H2C_vooo(nub, nob, nob, nob)
                  !f2py intent(in,out) H2C_vooo(nub, nob, nob, nob)

                   do a = 1, nua
                       do b = a + 1, nua
                           do i = 1, noa
                               do e = 1, nua

                                   do f = 1, nua
                                       do m = 1, noa
                                           do n = m + 1, noa
                                           end do
                                       end do
                                   end do


                               end do
                           end do
                       end do
                  end do

!                  ! build intermediates I2A_vvov and I2A_vooo
!                  do a = 1, nua
!                      do i = 1, noa
!                          do e = 1, nua
!                              do m = 1, noa
!
!                                  do f = 1, nua
!
!                                      do n = 1, noa
!                                          do b = a + 1, nua
!                                              if (pspace_aaa(a, b, f, i, m, n) == 1) then
!                                                  I2A_vvov(a, b, i, e) = I2A_vvov(a, b, i, e) - H2A_oovv(m, n, e, f) * t3a(a, b, f, i, m, n)
!                                              end if
!                                              I2A_vvov(b, a, i, e) = -1.0 * I2A_vvov(a, b, i, e)
!                                          end do
!
!                                          do j = i + 1, nua
!                                              if (pspace_aaa(a, e, f, i, j, n) == 1) then
!                                                  I2A_vooo(a, m, i, j) = I2A_vooo(a, m, i, j) + H2A_oovv(m, n, e,f ) * t3a(a, e, f, i, j, n)
!                                              end if
!                                              I2A_vooo(a, m, j, i) = -1.0 * I2A_vooo(a, m, i, j)
!                                          end do
!                                      end do
!
!                                      do n = 1, nob
!                                          do b = a + 1, nua
!                                              if (pspace_aab(a, b, f, i, m, n) == 1) then
!                                                  I2A_vvov(a, b, i, e) = I2A_vvov(a, b, i, e) - H2B_oovv(m, n, e, f) * t3b(a, b, f, i, m, n)
!                                              end if
!                                              I2A_vvov(b, a, i, e) = -1.0 * I2A_vvov(a, b, i, e)
!                                          end do
!
!                                          do j = i + 1, nua
!                                              if (pspace_aab(a, e, f, i, j, n) == 1) then
!                                                  I2A_vooo(a, m, i, j) = I2A_vooo(a, m, i, j) + H2B_oovv(m, n, e,f ) * t3b(a, e, f, i, j, n)
!                                              end if
!                                              I2A_vooo(a, m, j, i) = -1.0 * I2A_vooo(a, m, i, j)
!                                          end do
!                                      end do
!
!                                  end do
!
!                                  do f = 1, nub - nua
!                                      do n = 1, nob
!                                          do b = a + 1, nua
!                                              if (pspace_aab(a, b, f + nua, i, m, n) == 1) then
!                                                  I2A_vvov(a, b, i, e) = I2A_vvov(a, b, i, e) - H2B_oovv(m, n, e, f + nua) * t3b(a, b, f + nua, i, m, n)
!                                              end if
!                                              I2A_vvov(b, a, i, e) = -1.0 * I2A_vvov(a, b, i, e)
!                                          end do
!
!                                          do j = i + 1, nua
!                                              if (pspace_aab(a, e, f + nua, i, j, n) == 1) then
!                                                  I2A_vooo(a, m, i, j) = I2A_vooo(a, m, i, j) + H2B_oovv(m, n, e, f + nua) * t3b(a, e, f + nua, i, j, n)
!                                              end if
!                                              I2A_vooo(a, m, j, i) = -1.0 * I2A_vooo(a, m, i, j)
!                                          end do
!                                      end do
!                                  end do
!
!                              end do
!                          end do
!                      end do
!                  end do

              end subroutine build_intermediates

end module ccp_intermediates