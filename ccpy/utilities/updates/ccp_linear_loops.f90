module ccp_linear_loops

!!!!      USE OMP_LIB
!!!!      USE MKL_SERVICE
      implicit none

      contains

               subroutine update_t1a(t1a, resid,&
                                     X1A,&
                                     t3a_excits, t3b_excits, t3c_excits,&
                                     t3a_amps, t3b_amps, t3c_amps,&
                                     H2A_oovv, H2B_oovv, H2C_oovv,&
                                     fA_oo, fA_vv,&
                                     shift,&
                                     n3aaa, n3aab, n3abb,&
                                     noa, nua, nob, nub)

                      integer, intent(in) :: noa, nua, nob, nub, n3aaa, n3aab, n3abb
                      integer, intent(in) :: t3a_excits(n3aaa, 6), t3b_excits(n3aab, 6), t3c_excits(n3abb, 6)
                      real(kind=8), intent(in) :: t3a_amps(n3aaa), t3b_amps(n3aab), t3c_amps(n3abb)
                      real(kind=8), intent(in) :: X1A(1:nua,1:noa),&
                                                  H2A_oovv(1:noa,1:noa,1:nua,1:nua),&
                                                  H2B_oovv(1:noa,1:nob,1:nua,1:nub),&
                                                  H2C_oovv(1:nob,1:nob,1:nub,1:nub),&
                                                  fA_oo(1:noa,1:noa), fA_vv(1:nua,1:nua),&
                                                  shift

                      real(kind=8), intent(inout) :: t1a(1:nua,1:noa)
                      !f2py intent(in,out) :: t1a(0:nua-1,0:noa-1)

                      real(kind=8), intent(out) :: resid(1:nua,1:noa)

                      integer :: i, a, m, n, e, f, idet
                      real(kind=8) :: denom, val, t_amp

                      ! store x1a in resid container
                      resid(:,:) = X1A(:,:)

                      do idet = 1, n3aaa
                          t_amp = t3a_amps(idet)
                          ! A(a/ef)A(i/mn) h2a(mnef) * t3a(aefimn)
                          a = t3a_excits(idet,1); e = t3a_excits(idet,2); f = t3a_excits(idet,3);
                          i = t3a_excits(idet,4); m = t3a_excits(idet,5); n = t3a_excits(idet,6);
                          resid(a,i) = resid(a,i) + H2A_oovv(m,n,e,f) * t_amp ! (1)
                          resid(e,i) = resid(e,i) - H2A_oovv(m,n,a,f) * t_amp ! (ae)
                          resid(f,i) = resid(f,i) - H2A_oovv(m,n,e,a) * t_amp ! (af)
                          resid(a,m) = resid(a,m) - H2A_oovv(i,n,e,f) * t_amp ! (im)
                          resid(e,m) = resid(e,m) + H2A_oovv(i,n,a,f) * t_amp ! (ae)(im)
                          resid(f,m) = resid(f,m) + H2A_oovv(i,n,e,a) * t_amp ! (af)(im)
                          resid(a,n) = resid(a,n) - H2A_oovv(m,i,e,f) * t_amp ! (in)
                          resid(e,n) = resid(e,n) + H2A_oovv(m,i,a,f) * t_amp ! (ae)(in)
                          resid(f,n) = resid(f,n) + H2A_oovv(m,i,e,a) * t_amp ! (af)(in)
                      end do

                      do idet = 1, n3aab
                          t_amp = t3b_amps(idet)
                          ! A(ae)A(im) h2b(mnef) * t3b(aefimn)
                          a = t3b_excits(idet,1); e = t3b_excits(idet,2); f = t3b_excits(idet,3);
                          i = t3b_excits(idet,4); m = t3b_excits(idet,5); n = t3b_excits(idet,6);
                          resid(a,i) = resid(a,i) + H2B_oovv(m,n,e,f) * t_amp ! (1)
                          resid(e,i) = resid(e,i) - H2B_oovv(m,n,a,f) * t_amp ! (ae)
                          resid(a,m) = resid(a,m) - H2B_oovv(i,n,e,f) * t_amp ! (im)
                          resid(e,m) = resid(e,m) + H2B_oovv(i,n,a,f) * t_amp ! (ae)(im)
                      end do

                      do idet = 1, n3abb
                          t_amp = t3c_amps(idet)
                          ! h2c(mnef) * t3c(aefimn)
                          a = t3c_excits(idet,1); e = t3c_excits(idet,2); f = t3c_excits(idet,3);
                          i = t3c_excits(idet,4); m = t3c_excits(idet,5); n = t3c_excits(idet,6);
                          resid(a,i) = resid(a,i) + H2A_oovv(m,n,e,f) * t_amp ! (1)
                      end do

                      do i = 1, noa
                          do a = 1, nua
                              denom = fA_oo(i,i) - fA_vv(a,a)

                              val = resid(a,i)/(denom - shift)

                              t1a(a,i) = t1a(a,i) + val

                              resid(a,i) = val
                          end do
                      end do

              end subroutine update_t1a

              subroutine update_t1b(t1b, resid,&
                                    X1B,&
                                    t3b_excits, t3c_excits, t3d_excits,&
                                    t3b_amps, t3c_amps, t3d_amps,&
                                    H2A_oovv, H2B_oovv, H2C_oovv,&
                                    fB_oo, fB_vv,&
                                    shift,&
                                    n3aab, n3abb, n3bbb,&
                                    noa, nua, nob, nub)

                      integer, intent(in) :: noa, nua, nob, nub, n3aab, n3abb, n3bbb
                      integer, intent(in) :: t3b_excits(n3aab, 6), t3c_excits(n3abb, 6), t3d_excits(n3bbb, 6)
                      real(kind=8), intent(in) :: t3b_amps(n3aab), t3c_amps(n3abb), t3d_amps(n3bbb)
                      real(kind=8), intent(in) :: X1B(1:nub,1:nob),&
                                                  H2A_oovv(1:noa,1:noa,1:nua,1:nua),&
                                                  H2B_oovv(1:noa,1:nob,1:nua,1:nub),&
                                                  H2C_oovv(1:nob,1:nob,1:nub,1:nub),&
                                                  fB_oo(1:nob,1:nob), fB_vv(1:nub,1:nub),&
                                                  shift

                      real(kind=8), intent(inout) :: t1b(1:nub,1:nob)
                      !f2py intent(in,out) :: t1b(0:nub-1,0:nob-1)

                      real(kind=8), intent(out) :: resid(1:nub,1:nob)

                      integer :: i, a, m, n, e, f, idet
                      real(kind=8) :: denom, val, t_amp

                      ! Store x1b in residual container
                      resid(:,:) = X1B(:,:)

                      do idet = 1, n3aab
                          t_amp = t3b_amps(idet)
                          ! h2a(mnef) * t3b(efamni)
                          e = t3b_excits(idet,1); f = t3b_excits(idet,2); a = t3b_excits(idet,3);
                          m = t3b_excits(idet,4); n = t3b_excits(idet,5); i = t3b_excits(idet,6);
                          resid(a,i) = resid(a,i) + H2A_oovv(m,n,e,f) * t_amp ! (1)
                      end do

                      do idet = 1, n3abb
                          t_amp = t3c_amps(idet)
                          ! A(af)A(in) h2b(mnef) * t3c(efamni)
                          e = t3c_excits(idet,1); f = t3c_excits(idet,2); a = t3c_excits(idet,3);
                          m = t3c_excits(idet,4); n = t3c_excits(idet,5); i = t3c_excits(idet,6);
                          resid(a,i) = resid(a,i) + H2B_oovv(m,n,e,f) * t_amp ! (1)
                          resid(f,i) = resid(f,i) - H2B_oovv(m,n,e,a) * t_amp ! (af)
                          resid(a,n) = resid(a,n) - H2B_oovv(m,i,e,f) * t_amp ! (in)
                          resid(f,n) = resid(f,n) + H2B_oovv(m,i,e,a) * t_amp ! (af)(in)
                      end do

                      do idet = 1, n3bbb
                          t_amp = t3d_amps(idet)
                          ! A(a/ef)A(i/mn) h2c(mnef) * t3d(aefimn)
                          a = t3d_excits(idet,1); e = t3d_excits(idet,2); f = t3d_excits(idet,3);
                          i = t3d_excits(idet,4); m = t3d_excits(idet,5); n = t3d_excits(idet,6);
                          resid(a,i) = resid(a,i) + H2C_oovv(m,n,e,f) * t_amp ! (1)
                          resid(e,i) = resid(e,i) - H2C_oovv(m,n,a,f) * t_amp ! (ae)
                          resid(f,i) = resid(f,i) - H2C_oovv(m,n,e,a) * t_amp ! (af)
                          resid(a,m) = resid(a,m) - H2C_oovv(i,n,e,f) * t_amp ! (im)
                          resid(e,m) = resid(e,m) + H2C_oovv(i,n,a,f) * t_amp ! (ae)(im)
                          resid(f,m) = resid(f,m) + H2C_oovv(i,n,e,a) * t_amp ! (af)(im)
                          resid(a,n) = resid(a,n) - H2C_oovv(m,i,e,f) * t_amp ! (in)
                          resid(e,n) = resid(e,n) + H2C_oovv(m,i,a,f) * t_amp ! (ae)(in)
                          resid(f,n) = resid(f,n) + H2C_oovv(m,i,e,a) * t_amp ! (af)(in)
                      end do

                      do i = 1, nob
                          do a = 1, nub
                              denom = fB_oo(i,i) - fB_vv(a,a)

                              val = resid(a,i)/(denom - shift)

                              t1b(a,i) = t1b(a,i) + val

                              resid(a,i) = val
                          end do
                      end do

              end subroutine update_t1b


              subroutine update_t2a(t2a, resid,&
                                    X2A,&
                                    t3a_excits, t3b_excits,&
                                    t3a_amps, t3b_amps,&
                                    H1A_ov, H1B_ov,&
                                    H2A_ooov, H2A_vovv,&
                                    H2B_ooov, H2B_vovv,&
                                    fA_oo, fA_vv,&
                                    shift,&
                                    n3aaa, n3aab,&
                                    noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub, n3aaa, n3aab
                  integer, intent(in) :: t3a_excits(n3aaa, 6), t3b_excits(n3aab, 6)
                  real(kind=8), intent(in) :: t3a_amps(n3aaa), t3b_amps(n3aab)
                  real(kind=8), intent(in) :: X2A(1:nua,1:nua,1:noa,1:noa),&
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

                  integer :: i, j, a, b, m, n, e, f, idet
                  real(kind=8) :: denom, val, t_amp

                  ! Store x2a in residual container
                  resid(:,:,:,:) = x2a(:,:,:,:)

                  do idet = 1, n3aaa
                      t_amp = t3a_amps(idet)

                      ! A(ij)A(ab) [A(m/ij)A(e/ab) h1a(me) * t3a(abeijm)]
                      a = t3a_excits(idet,1); b = t3a_excits(idet,2); e = t3a_excits(idet,3);
                      i = t3a_excits(idet,4); j = t3a_excits(idet,5); m = t3a_excits(idet,6);
                      resid(a,b,i,j) = resid(a,b,i,j) + H1A_ov(m,e) * t_amp ! (1)
                      resid(a,b,m,j) = resid(a,b,m,j) - H1A_ov(i,e) * t_amp ! (im)
                      resid(a,b,i,m) = resid(a,b,i,m) - H1A_ov(j,e) * t_amp ! (jm)
                      resid(e,b,i,j) = resid(e,b,i,j) - H1A_ov(m,a) * t_amp ! (ae)
                      resid(e,b,m,j) = resid(e,b,m,j) + H1A_ov(i,a) * t_amp ! (im)(ae)
                      resid(e,b,i,m) = resid(e,b,i,m) + H1A_ov(j,a) * t_amp ! (jm)(ae)
                      resid(a,e,i,j) = resid(a,e,i,j) - H1A_ov(m,b) * t_amp ! (be)
                      resid(a,e,m,j) = resid(a,e,m,j) + H1A_ov(i,b) * t_amp ! (im)(be)
                      resid(a,e,i,m) = resid(a,e,i,m) + H1A_ov(j,b) * t_amp ! (jm)(be)

                      ! A(ij)A(ab) [A(j/mn)A(f/ab) -h2a(mnif) * t3a(abfmjn)]
                      a = t3a_excits(idet,1); b = t3a_excits(idet,2); f = t3a_excits(idet,3);
                      m = t3a_excits(idet,4); j = t3a_excits(idet,5); n = t3a_excits(idet,6);
                      resid(a,b,:,j) = resid(a,b,:,j) - H2A_ooov(m,n,:,f) * t_amp ! (1)
                      resid(a,b,:,m) = resid(a,b,:,m) + H2A_ooov(j,n,:,f) * t_amp ! (jm)
                      resid(a,b,:,n) = resid(a,b,:,n) + H2A_ooov(m,j,:,f) * t_amp ! (jn)
                      resid(f,b,:,j) = resid(f,b,:,j) + H2A_ooov(m,n,:,a) * t_amp ! (af)
                      resid(f,b,:,m) = resid(f,b,:,m) - H2A_ooov(j,n,:,a) * t_amp ! (jm)(af)
                      resid(f,b,:,n) = resid(f,b,:,n) - H2A_ooov(m,j,:,a) * t_amp ! (jn)(af)
                      resid(a,f,:,j) = resid(a,f,:,j) + H2A_ooov(m,n,:,b) * t_amp ! (bf)
                      resid(a,f,:,m) = resid(a,f,:,m) - H2A_ooov(j,n,:,b) * t_amp ! (jm)(bf)
                      resid(a,f,:,n) = resid(a,f,:,n) - H2A_ooov(m,j,:,b) * t_amp ! (jn)(bf)

                      ! A(ij)A(ab) [A(n/ij)A(b/ef) h2a(anef) * t3a(ebfijn)]
                      e = t3a_excits(idet,1); b = t3a_excits(idet,2); f = t3a_excits(idet,3);
                      i = t3a_excits(idet,4); j = t3a_excits(idet,5); n = t3a_excits(idet,6);
                      resid(:,b,i,j) = resid(:,b,i,j) + H2A_vovv(:,n,e,f) * t_amp ! (1)
                      resid(:,b,n,j) = resid(:,b,n,j) - H2A_vovv(:,i,e,f) * t_amp ! (in)
                      resid(:,b,i,n) = resid(:,b,i,n) - H2A_vovv(:,j,e,f) * t_amp ! (jn)
                      resid(:,e,i,j) = resid(:,e,i,j) - H2A_vovv(:,n,b,f) * t_amp ! (be)
                      resid(:,e,n,j) = resid(:,e,n,j) + H2A_vovv(:,i,b,f) * t_amp ! (in)(be)
                      resid(:,e,i,n) = resid(:,e,i,n) + H2A_vovv(:,j,b,f) * t_amp ! (jn)(be)
                      resid(:,f,i,j) = resid(:,f,i,j) - H2A_vovv(:,n,e,b) * t_amp ! (bf)
                      resid(:,f,n,j) = resid(:,f,n,j) + H2A_vovv(:,i,e,b) * t_amp ! (in)(bf)
                      resid(:,f,i,n) = resid(:,f,i,n) + H2A_vovv(:,j,e,b) * t_amp ! (jn)(bf)
                  end do

                  do idet = 1, n3aab
                      t_amp = t3b_amps(idet)

                      ! A(ij)A(ab) [h1b(me) * t3b(abeijm)]
                      a = t3b_excits(idet,1); b = t3b_excits(idet,2); e = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); j = t3b_excits(idet,5); m = t3b_excits(idet,6);
                      resid(a,b,i,j) = resid(a,b,i,j) + H1B_ov(m,e) * t_amp ! (1)

                      ! A(ij)A(ab) [A(jm) -h2b(mnif) * t3b(abfmjn)]
                      a = t3b_excits(idet,1); b = t3b_excits(idet,2); f = t3b_excits(idet,3);
                      m = t3b_excits(idet,4); j = t3b_excits(idet,5); n = t3b_excits(idet,6);
                      resid(a,b,:,j) = resid(a,b,:,j) - H2B_ooov(m,n,:,f) * t_amp ! (1)
                      resid(a,b,:,m) = resid(a,b,:,m) + H2B_ooov(j,n,:,f) * t_amp ! (jm)

                      ! A(ij)A(ab) [A(be) h2b(anef) * t3b(ebfijn)]
                      e = t3b_excits(idet,1); b = t3b_excits(idet,2); f = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); j = t3b_excits(idet,5); n = t3b_excits(idet,6);
                      resid(:,b,i,j) = resid(:,b,i,j) + H2B_vovv(:,n,e,f) * t_amp ! (1)
                      resid(:,e,i,j) = resid(:,e,i,j) - H2B_vovv(:,n,b,f) * t_amp ! (be)
                  end do

                  do i = 1, noa
                      do j = i + 1, noa
                          do a = 1, nua
                              do b = a + 1, nua
                                  denom = fA_oo(i,i) + fA_oo(j,j) - fA_vv(a,a) - fA_vv(b,b)

                                  val = resid(b,a,j,i) - resid(a,b,j,i) - resid(b,a,i,j) + resid(a,b,i,j)
                                  val = val/(denom - shift)

                                  t2a(b,a,j,i) =  t2a(b,a,j,i) + val
                                  t2a(a,b,j,i) = -t2a(b,a,j,i)
                                  t2a(b,a,i,j) = -t2a(b,a,j,i)
                                  t2a(a,b,i,j) =  t2a(b,a,j,i)

                                  resid(b,a,j,i) =  val
                                  resid(a,b,j,i) = -val
                                  resid(b,a,i,j) = -val
                                  resid(a,b,i,j) =  val
                              end do
                          end do
                      end do
                  end do

                  do a = 1, nua
                     resid(a,a,:,:) = 0.0d0
                  end do
                  do i = 1, noa
                     resid(:,:,i,i) = 0.0d0
                  end do

              end subroutine update_t2a

              subroutine update_t2b(t2b, resid,&
                                    X2B,&
                                    t3b_excits, t3c_excits,&
                                    t3b_amps, t3c_amps,&
                                    H1A_ov, H1B_ov,&
                                    H2A_ooov, H2A_vovv,&
                                    H2B_ooov, H2B_oovo, H2B_vovv, H2B_ovvv,&
                                    H2C_ooov, H2C_vovv,&
                                    fA_oo, fA_vv, fB_oo, fB_vv,&
                                    shift,&
                                    n3aab, n3abb,&
                                    noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub, n3aab, n3abb
                  integer, intent(in) :: t3b_excits(n3aab, 6), t3c_excits(n3abb, 6)
                  real(kind=8), intent(in) :: t3b_amps(n3aab), t3c_amps(n3abb)
                  real(kind=8), intent(in) :: X2B(1:nua,1:nub,1:noa,1:nob),&
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

                  integer :: i, j, a, b, m, n, e, f, idet
                  real(kind=8) :: denom, val, t_amp

                  ! Store x2b in residual container
                  resid(:,:,:,:) = x2b(:,:,:,:)

                  do idet = 1, n3aab
                      t_amp = t3b_amps(idet)

                      ! A(af) -h2a(mnif) * t3b(afbmnj)
                      a = t3b_excits(idet,1); f = t3b_excits(idet,2); b = t3b_excits(idet,3);
                      m = t3b_excits(idet,4); n = t3b_excits(idet,5); j = t3b_excits(idet,6);
                      resid(a,b,:,j) = resid(a,b,:,j) - H2A_ooov(m,n,:,f) * t_amp ! (1)
                      resid(f,b,:,j) = resid(f,b,:,j) + H2A_ooov(m,n,:,a) * t_amp ! (af)

                      ! A(af)A(in) -h2b(nmfj) * t3b(afbinm)
                      a = t3b_excits(idet,1); f = t3b_excits(idet,2); b = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); n = t3b_excits(idet,5); m = t3b_excits(idet,6);
                      resid(a,b,i,:) = resid(a,b,i,:) - H2B_oovo(n,m,f,:) * t_amp ! (1)
                      resid(f,b,i,:) = resid(f,b,i,:) + H2B_oovo(n,m,a,:) * t_amp ! (af)
                      resid(a,b,n,:) = resid(a,b,n,:) + H2B_oovo(i,m,f,:) * t_amp ! (in)
                      resid(f,b,n,:) = resid(f,b,n,:) - H2B_oovo(i,m,a,:) * t_amp ! (af)(in)

                      ! A(in) h2a(anef) * t3b(efbinj)
                      e = t3b_excits(idet,1); f = t3b_excits(idet,2); b = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); n = t3b_excits(idet,5); j = t3b_excits(idet,6);
                      resid(:,b,i,j) = resid(:,b,i,j) + H2A_vovv(:,n,e,f) * t_amp ! (1)
                      resid(:,b,n,j) = resid(:,b,n,j) - H2A_vovv(:,i,e,f) * t_amp ! (in)

                      ! A(af)A(in) h2b(nbfe) * t3b(afeinj)
                      a = t3b_excits(idet,1); f = t3b_excits(idet,2); e = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); n = t3b_excits(idet,5); j = t3b_excits(idet,6);
                      resid(a,:,i,j) = resid(a,:,i,j) + H2B_ovvv(n,:,f,e) * t_amp ! (1)
                      resid(f,:,i,j) = resid(f,:,i,j) - H2B_ovvv(n,:,a,e) * t_amp ! (af)
                      resid(a,:,n,j) = resid(a,:,n,j) - H2B_ovvv(i,:,f,e) * t_amp ! (in)
                      resid(f,:,n,j) = resid(f,:,n,j) + H2B_ovvv(i,:,a,e) * t_amp ! (af)(in)

                      ! A(ae)A(im) h1a(me) * t3b(aebimj)
                      a = t3b_excits(idet,1); e = t3b_excits(idet,2); b = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); m = t3b_excits(idet,5); j = t3b_excits(idet,6);
                      resid(a,b,i,j) = resid(a,b,i,j) + H1A_ov(m,e) * t_amp ! (1)
                      resid(a,b,m,j) = resid(a,b,m,j) - H1A_ov(i,e) * t_amp ! (im)
                      resid(e,b,i,j) = resid(e,b,i,j) - H1A_ov(m,a) * t_amp ! (ae)
                      resid(e,b,m,j) = resid(e,b,m,j) + H1A_ov(i,a) * t_amp ! (im)(ae)

                  end do

                  do idet = 1, n3abb
                      t_amp = t3c_amps(idet)

                      ! A(bf) -h2c(mnjf) * t3c(afbinm)
                      a = t3c_excits(idet,1); f = t3c_excits(idet,2); b = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); n = t3c_excits(idet,5); m = t3c_excits(idet,6);
                      resid(a,b,i,:) = resid(a,b,i,:) - H2C_ooov(m,n,:,f) * t_amp ! (1)
                      resid(a,f,i,:) = resid(a,f,i,:) + H2C_ooov(m,n,:,b) * t_amp ! (bf)

                      ! A(bf)A(jn) -h2b(mnif) * t3c(afbmnj)
                      a = t3c_excits(idet,1); f = t3c_excits(idet,2); b = t3c_excits(idet,3);
                      m = t3c_excits(idet,4); n = t3c_excits(idet,5); j = t3c_excits(idet,6);
                      resid(a,b,:,j) = resid(a,b,:,j) - H2B_ooov(m,n,:,f) * t_amp ! (1)
                      resid(a,f,:,j) = resid(a,f,:,j) + H2B_ooov(m,n,:,b) * t_amp ! (bf)
                      resid(a,b,:,n) = resid(a,b,:,n) + H2B_ooov(m,j,:,f) * t_amp ! (jn)
                      resid(a,f,:,n) = resid(a,f,:,n) - H2B_ooov(m,j,:,b) * t_amp ! (bf)(jn)

                      ! A(jn) h2c(bnef) * t3c(afeinj)
                      a = t3c_excits(idet,1); f = t3c_excits(idet,2); e = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); n = t3c_excits(idet,5); j = t3c_excits(idet,6);
                      resid(a,:,i,j) = resid(a,:,i,j) + H2C_vovv(:,n,e,f) * t_amp ! (1)
                      resid(a,:,i,n) = resid(a,:,i,n) - H2C_vovv(:,j,e,f) * t_amp ! (jn)

                      ! A(bf)A(jn) h2b(anef) * t3c(efbinj)
                      e = t3c_excits(idet,1); f = t3c_excits(idet,2); b = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); n = t3c_excits(idet,5); j = t3c_excits(idet,6);
                      resid(:,b,i,j) = resid(:,b,i,j) + H2B_vovv(:,n,e,f) * t_amp ! (1)
                      resid(:,f,i,j) = resid(:,f,i,j) - H2B_vovv(:,n,e,b) * t_amp ! (bf)
                      resid(:,b,i,n) = resid(:,b,i,n) - H2B_vovv(:,j,e,f) * t_amp ! (jn)
                      resid(:,f,i,n) = resid(:,f,i,n) + H2B_vovv(:,j,e,b) * t_amp ! (bf)(jn)

                      ! [A(be)A(mj) h1b(me) * t3c(aebimj)]
                      a = t3c_excits(idet,1); e = t3c_excits(idet,2); b = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); m = t3c_excits(idet,5); j = t3c_excits(idet,6);
                      resid(a,b,i,j) = resid(a,b,i,j) + H1B_ov(m,e) * t_amp ! (1)
                      resid(a,b,i,m) = resid(a,b,i,m) - H1B_ov(j,e) * t_amp ! (jm)
                      resid(a,e,i,j) = resid(a,e,i,j) - H1B_ov(m,b) * t_amp ! (be)
                      resid(a,e,i,m) = resid(a,e,i,m) + H1B_ov(j,b) * t_amp ! (jm)(be)
                  end do

                  do j = 1, nob
                      do i = 1, noa
                          do b = 1, nub
                              do a = 1, nua
                                  denom = fA_oo(i,i) + fB_oo(j,j) - fA_vv(a,a) - fB_vv(b,b)
                                  val = resid(a,b,i,j)/(denom - shift)

                                  t2b(a,b,i,j) = t2b(a,b,i,j) + val

                                  resid(a,b,i,j) = val
                              end do
                          end do
                      end do
                  end do

              end subroutine update_t2b

              subroutine update_t2c(t2c, resid,&
                                    X2C,&
                                    t3c_excits, t3d_excits,&
                                    t3c_amps, t3d_amps,&
                                    H1A_ov, H1B_ov,&
                                    H2B_oovo, H2B_ovvv,&
                                    H2C_ooov, H2C_vovv,&
                                    fB_oo, fB_vv,&
                                    shift,&
                                    n3abb, n3bbb,&
                                    noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub, n3abb, n3bbb
                  integer, intent(in) :: t3c_excits(n3abb, 6), t3d_excits(n3bbb, 6)
                  real(kind=8), intent(in) :: t3c_amps(n3abb), t3d_amps(n3bbb)
                  real(kind=8), intent(in) :: X2C(1:nub,1:nub,1:nob,1:nob),&
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

                  integer :: i, j, a, b, m, n, e, f, idet
                  real(kind=8) :: denom, val, t_amp

                  ! store x2c in residual container
                  resid(:,:,:,:) = x2c(:,:,:,:)

                  do idet = 1, n3abb
                      t_amp = t3c_amps(idet)

                      ! A(ij)A(ab) [h1a(me) * t3c(eabmij)]
                      e = t3c_excits(idet,1); a = t3c_excits(idet,2); b = t3c_excits(idet,3);
                      m = t3c_excits(idet,4); i = t3c_excits(idet,5); j = t3c_excits(idet,6);
                      resid(a,b,i,j) = resid(a,b,i,j) + H1A_ov(m,e) * t_amp ! (1)
            
                      ! A(ij)A(ab) [A(be) h2b(nafe) * t3c(febnij)]
                      f = t3c_excits(idet,1); e = t3c_excits(idet,2); b = t3c_excits(idet,3);
                      n = t3c_excits(idet,4); i = t3c_excits(idet,5); j = t3c_excits(idet,6);
                      resid(:,b,i,j) = resid(:,b,i,j) + H2B_ovvv(n,:,f,e) * t_amp ! (1)
                      resid(:,e,i,j) = resid(:,e,i,j) - H2B_ovvv(n,:,f,b) * t_amp ! (be)

                      ! A(ij)A(ab) [A(jm) -h2b(nmfi) * t3c(fabnmj)]
                      f = t3c_excits(idet,1); a = t3c_excits(idet,2); b = t3c_excits(idet,3);
                      n = t3c_excits(idet,4); m = t3c_excits(idet,5); j = t3c_excits(idet,6);
                      resid(a,b,:,j) = resid(a,b,:,j) - H2B_oovo(n,m,f,:) * t_amp ! (1)
                      resid(a,b,:,m) = resid(a,b,:,m) + H2B_oovo(n,j,f,:) * t_amp ! (jm)
                  end do

                  do idet = 1, n3bbb
                      t_amp = t3d_amps(idet)

                      ! A(ij)A(ab) [A(m/ij)A(e/ab) h1b(me) * t3d(abeijm)]
                      a = t3d_excits(idet,1); b = t3d_excits(idet,2); e = t3d_excits(idet,3);
                      i = t3d_excits(idet,4); j = t3d_excits(idet,5); m = t3d_excits(idet,6);
                      resid(a,b,i,j) = resid(a,b,i,j) + H1B_ov(m,e) * t_amp ! (1)
                      resid(a,b,m,j) = resid(a,b,m,j) - H1B_ov(i,e) * t_amp ! (im)
                      resid(a,b,i,m) = resid(a,b,i,m) - H1B_ov(j,e) * t_amp ! (jm)
                      resid(e,b,i,j) = resid(e,b,i,j) - H1B_ov(m,a) * t_amp ! (ae)
                      resid(e,b,m,j) = resid(e,b,m,j) + H1B_ov(i,a) * t_amp ! (im)(ae)
                      resid(e,b,i,m) = resid(e,b,i,m) + H1B_ov(j,a) * t_amp ! (jm)(ae)
                      resid(a,e,i,j) = resid(a,e,i,j) - H1B_ov(m,b) * t_amp ! (be)
                      resid(a,e,m,j) = resid(a,e,m,j) + H1B_ov(i,b) * t_amp ! (im)(be)
                      resid(a,e,i,m) = resid(a,e,i,m) + H1B_ov(j,b) * t_amp ! (jm)(be)

                      ! A(ij)A(ab) [A(j/mn)A(f/ab) -h2c(mnif) * t3d(abfmjn)]
                      a = t3d_excits(idet,1); b = t3d_excits(idet,2); f = t3d_excits(idet,3);
                      m = t3d_excits(idet,4); j = t3d_excits(idet,5); n = t3d_excits(idet,6);
                      resid(a,b,:,j) = resid(a,b,:,j) - H2C_ooov(m,n,:,f) * t_amp ! (1)
                      resid(a,b,:,m) = resid(a,b,:,m) + H2C_ooov(j,n,:,f) * t_amp ! (jm)
                      resid(a,b,:,n) = resid(a,b,:,n) + H2C_ooov(m,j,:,f) * t_amp ! (jn)
                      resid(f,b,:,j) = resid(f,b,:,j) + H2C_ooov(m,n,:,a) * t_amp ! (af)
                      resid(f,b,:,m) = resid(f,b,:,m) - H2C_ooov(j,n,:,a) * t_amp ! (jm)(af)
                      resid(f,b,:,n) = resid(f,b,:,n) - H2C_ooov(m,j,:,a) * t_amp ! (jn)(af)
                      resid(a,f,:,j) = resid(a,f,:,j) + H2C_ooov(m,n,:,b) * t_amp ! (bf)
                      resid(a,f,:,m) = resid(a,f,:,m) - H2C_ooov(j,n,:,b) * t_amp ! (jm)(bf)
                      resid(a,f,:,n) = resid(a,f,:,n) - H2C_ooov(m,j,:,b) * t_amp ! (jn)(bf)

                      ! A(ij)A(ab) [A(n/ij)A(b/ef) h2c(anef) * t3d(ebfijn)]
                      e = t3d_excits(idet,1); b = t3d_excits(idet,2); f = t3d_excits(idet,3);
                      i = t3d_excits(idet,4); j = t3d_excits(idet,5); n = t3d_excits(idet,6);
                      resid(:,b,i,j) = resid(:,b,i,j) + H2C_vovv(:,n,e,f) * t_amp ! (1)
                      resid(:,b,n,j) = resid(:,b,n,j) - H2C_vovv(:,i,e,f) * t_amp ! (in)
                      resid(:,b,i,n) = resid(:,b,i,n) - H2C_vovv(:,j,e,f) * t_amp ! (jn)
                      resid(:,e,i,j) = resid(:,e,i,j) - H2C_vovv(:,n,b,f) * t_amp ! (be)
                      resid(:,e,n,j) = resid(:,e,n,j) + H2C_vovv(:,i,b,f) * t_amp ! (in)(be)
                      resid(:,e,i,n) = resid(:,e,i,n) + H2C_vovv(:,j,b,f) * t_amp ! (jn)(be)
                      resid(:,f,i,j) = resid(:,f,i,j) - H2C_vovv(:,n,e,b) * t_amp ! (bf)
                      resid(:,f,n,j) = resid(:,f,n,j) + H2C_vovv(:,i,e,b) * t_amp ! (in)(bf)
                      resid(:,f,i,n) = resid(:,f,i,n) + H2C_vovv(:,j,e,b) * t_amp ! (jn)(bf)
                  end do

                  do i = 1, nob
                      do j = i + 1, nob
                          do a = 1, nub
                              do b = a + 1, nub
                                  denom = fB_oo(i,i) + fB_oo(j,j) - fB_vv(a,a) - fB_vv(b,b)

                                  val = resid(b,a,j,i) - resid(a,b,j,i) - resid(b,a,i,j) + resid(a,b,i,j)
                                  val = val/(denom - shift)

                                  t2c(b,a,j,i) =  t2c(b,a,j,i) + val
                                  t2c(a,b,j,i) = -t2c(b,a,j,i)
                                  t2c(b,a,i,j) = -t2c(b,a,j,i)
                                  t2c(a,b,i,j) =  t2c(b,a,j,i)

                                  resid(b,a,j,i) =  val
                                  resid(a,b,j,i) = -val
                                  resid(b,a,i,j) = -val
                                  resid(a,b,i,j) =  val
                              end do
                          end do
                      end do
                  end do

                  do a = 1, nub
                     resid(a,a,:,:) = 0.0d0
                  end do
                  do i = 1, nob
                     resid(:,:,i,i) = 0.0d0
                  end do

              end subroutine update_t2c

              subroutine update_t3a_p(t3a_amps, resid,&
                                      t3a_excits, t3b_excits,&
                                      t2a,&
                                      t3b_amps,&
                                      H1A_oo, H1A_vv,&
                                      H2A_oovv, H2A_vvov, H2A_vooo,&
                                      H2A_oooo, H2A_voov, H2A_vvvv,&
                                      H2B_oovv, H2B_voov,&
                                      fA_oo, fA_vv,&
                                      shift,&
                                      n3aaa, n3aab,&
                                      noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub, n3aaa, n3aab
                  integer, intent(in) :: t3a_excits(n3aaa, 6), t3b_excits(n3aab, 6)
                  real(kind=8), intent(in) :: t2a(nua, nua, noa, noa),&
                                              t3b_amps(n3aab),&
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

                  real(kind=8), intent(inout) :: t3a_amps(n3aaa)
                  !f2py intent(in,out) :: t3a_amps(0:n3aaa-1)

                  real(kind=8), intent(out) :: resid(n3aaa)

                  real(kind=8) :: I2A_vvov(nua, nua, noa, nua), I2A_vooo(nua, noa, noa, noa)
                  real(kind=8) :: val, denom, t_amp, res_mm23
                  integer :: a, b, c, i, j, k, e, f, m, n, idet

                  real(kind=8) :: x3a(nua, nua, nua, noa, noa, noa)

                  ! Zero the projection container
                  x3a = 0.0d0
                  ! Start the VT3 intermediates at Hbar (factor of 1/2 to compensate for antisymmetrization)
                  I2A_vooo(:,:,:,:) = 0.5d0 * H2A_vooo(:,:,:,:)
                  I2A_vvov(:,:,:,:) = 0.5d0 * H2A_vvov(:,:,:,:)
                  ! Loop over aaa determinants
                  do idet = 1, n3aaa
                      t_amp = t3a_amps(idet)

                      ! x3a(abcijk) <- -A(abc)A(i/jk)A(jk)A(m/jk) h1a(mi) * t3a(abcmjk)
                      !               = -A(abc)A(ijk)[ A(m/jk) h1a(mi) * t3a(abcmjk) ]
                      a = t3a_excits(idet,1); b = t3a_excits(idet,2); c = t3a_excits(idet,3);
                      m = t3a_excits(idet,4); j = t3a_excits(idet,5); k = t3a_excits(idet,6);
                      x3a(a,b,c,:,j,k) = x3a(a,b,c,:,j,k) - H1A_oo(m,:) * t_amp ! (1)
                      x3a(a,b,c,:,m,k) = x3a(a,b,c,:,m,k) + H1A_oo(j,:) * t_amp ! (mj)
                      x3a(a,b,c,:,j,m) = x3a(a,b,c,:,j,m) + H1A_oo(k,:) * t_amp ! (mk)

                      ! x3a(abcijk) <- A(abc)A(a/bc)A(bc)A(e/bc) h1a(ae) * t3a(ebcijk)
                      !              = A(abc)A(ijk)[ A(e/bc) h1a(ae) * t3a(ebcijk) ]
                      e = t3a_excits(idet,1); b = t3a_excits(idet,2); c = t3a_excits(idet,3);
                      i = t3a_excits(idet,4); j = t3a_excits(idet,5); k = t3a_excits(idet,6);
                      x3a(:,b,c,i,j,k) = x3a(:,b,c,i,j,k) + H1A_vv(:, e) * t_amp ! (1)
                      x3a(:,e,c,i,j,k) = x3a(:,e,c,i,j,k) - H1A_vv(:, b) * t_amp ! (be)
                      x3a(:,b,e,i,j,k) = x3a(:,b,e,i,j,k) - H1A_vv(:, c) * t_amp ! (ce)

                      ! x3a(abcijk) <- A(abc)A(k/ij)[ A(k/mn) h2a(mnij) * t3a(abcmnk) ]
                      !             = A(abc)A(ijk)[ 1/2 A(k/mn) h2a(mnij) * t3a(abcmnk) ]
                      a = t3a_excits(idet,1); b = t3a_excits(idet,2); c = t3a_excits(idet,3);
                      m = t3a_excits(idet,4); n = t3a_excits(idet,5); k = t3a_excits(idet,6);
                      x3a(a,b,c,:,:,k) = x3a(a,b,c,:,:,k) + 0.5d0 * H2A_oooo(m,n,:,:) * t_amp ! (1)
                      x3a(a,b,c,:,:,m) = x3a(a,b,c,:,:,m) - 0.5d0 * H2A_oooo(k,n,:,:) * t_amp ! (mk)
                      x3a(a,b,c,:,:,n) = x3a(a,b,c,:,:,n) - 0.5d0 * H2A_oooo(m,k,:,:) * t_amp ! (nk)

                      ! x3a(abcijk) <- A(c/ab)A(ijk)[ A(c/ef) h2a(abef) * t3a(efcijk) ]
                      !              = A(abc)A(ijk)[ 1/2 A(c/ef) h2a(abef) * t3a(efcijk) ]
                      e = t3a_excits(idet,1); f = t3a_excits(idet,2); c = t3a_excits(idet,3);
                      i = t3a_excits(idet,4); j = t3a_excits(idet,5); k = t3a_excits(idet,6);
                      x3a(:,:,c,i,j,k) = x3a(:,:,c,i,j,k) + 0.5d0 * H2A_vvvv(:,:,e,f) * t_amp ! (1)
                      x3a(:,:,e,i,j,k) = x3a(:,:,e,i,j,k) - 0.5d0 * H2A_vvvv(:,:,c,f) * t_amp ! (ec)
                      x3a(:,:,f,i,j,k) = x3a(:,:,f,i,j,k) - 0.5d0 * H2A_vvvv(:,:,e,c) * t_amp ! (fc)

                      ! x3a(abcijk) <- A(a/bc)A(bc)A(jk)A(i/jk)[ A(e/bc)A(m/jk) h2a(amie) * t3a(ebcmjk) ]
                      !              = A(abc)A(ijk)[ A(e/bc)A(m/jk) h2a(amie) * t3a(ebcmjk) ]
                      e = t3a_excits(idet,1); b = t3a_excits(idet,2); c = t3a_excits(idet,3);
                      m = t3a_excits(idet,4); j = t3a_excits(idet,5); k = t3a_excits(idet,6);
                      x3a(:,b,c,:,j,k) = x3a(:,b,c,:,j,k) + H2A_voov(:,m,:,e) * t_amp ! (1)
                      x3a(:,b,c,:,m,k) = x3a(:,b,c,:,m,k) - H2A_voov(:,j,:,e) * t_amp ! (mj)
                      x3a(:,b,c,:,j,m) = x3a(:,b,c,:,j,m) - H2A_voov(:,k,:,e) * t_amp ! (mk)
                      x3a(:,e,c,:,j,k) = x3a(:,e,c,:,j,k) - H2A_voov(:,m,:,b) * t_amp ! (eb)
                      x3a(:,e,c,:,m,k) = x3a(:,e,c,:,m,k) + H2A_voov(:,j,:,b) * t_amp ! (eb)(mj)
                      x3a(:,e,c,:,j,m) = x3a(:,e,c,:,j,m) + H2A_voov(:,k,:,b) * t_amp ! (eb)(mk)
                      x3a(:,b,e,:,j,k) = x3a(:,b,e,:,j,k) - H2A_voov(:,m,:,c) * t_amp ! (ec)
                      x3a(:,b,e,:,m,k) = x3a(:,b,e,:,m,k) + H2A_voov(:,j,:,c) * t_amp ! (ec)(mj)
                      x3a(:,b,e,:,j,m) = x3a(:,b,e,:,j,m) + H2A_voov(:,k,:,c) * t_amp ! (ec)(mk)

                      ! I2A(amij) <- A(ij) [A(n/ij)A(a/ef) h2a(mnef) * t3a(aefijn)]
                      a = t3a_excits(idet,1); e = t3a_excits(idet,2); f = t3a_excits(idet,3);
                      i = t3a_excits(idet,4); j = t3a_excits(idet,5); n = t3a_excits(idet,6);
                      I2A_vooo(a,:,i,j) = I2A_vooo(a,:,i,j) + H2A_oovv(:,n,e,f) * t_amp ! (1)
                      I2A_vooo(a,:,n,j) = I2A_vooo(a,:,n,j) - H2A_oovv(:,i,e,f) * t_amp ! (in)
                      I2A_vooo(a,:,i,n) = I2A_vooo(a,:,i,n) - H2A_oovv(:,j,e,f) * t_amp ! (jn)
                      I2A_vooo(e,:,i,j) = I2A_vooo(e,:,i,j) - H2A_oovv(:,n,a,f) * t_amp ! (ae)
                      I2A_vooo(e,:,n,j) = I2A_vooo(e,:,n,j) + H2A_oovv(:,i,a,f) * t_amp ! (in)(ae)
                      I2A_vooo(e,:,i,n) = I2A_vooo(e,:,i,n) + H2A_oovv(:,j,a,f) * t_amp ! (jn)(ae)
                      I2A_vooo(f,:,i,j) = I2A_vooo(f,:,i,j) - H2A_oovv(:,n,e,a) * t_amp ! (af)
                      I2A_vooo(f,:,n,j) = I2A_vooo(f,:,n,j) + H2A_oovv(:,i,e,a) * t_amp ! (in)(af)
                      I2A_vooo(f,:,i,n) = I2A_vooo(f,:,i,n) + H2A_oovv(:,j,e,a) * t_amp ! (jn)(af)

                      ! I2A(abie) <- A(ab) [A(i/mn)A(f/ab) -h2a(mnef) * t3a(abfimn)]
                      a = t3a_excits(idet,1); b = t3a_excits(idet,2); f = t3a_excits(idet,3);
                      i = t3a_excits(idet,4); m = t3a_excits(idet,5); n = t3a_excits(idet,6);
                      I2A_vvov(a,b,i,:) = I2A_vvov(a,b,i,:) - H2A_oovv(m,n,:,f) * t_amp ! (1)
                      I2A_vvov(a,b,m,:) = I2A_vvov(a,b,m,:) + H2A_oovv(i,n,:,f) * t_amp ! (im)
                      I2A_vvov(a,b,n,:) = I2A_vvov(a,b,n,:) + H2A_oovv(m,i,:,f) * t_amp ! (in)
                      I2A_vvov(f,b,i,:) = I2A_vvov(f,b,i,:) + H2A_oovv(m,n,:,a) * t_amp ! (af)
                      I2A_vvov(f,b,m,:) = I2A_vvov(f,b,m,:) - H2A_oovv(i,n,:,a) * t_amp ! (im)(af)
                      I2A_vvov(f,b,n,:) = I2A_vvov(f,b,n,:) - H2A_oovv(m,i,:,a) * t_amp ! (in)(af)
                      I2A_vvov(a,f,i,:) = I2A_vvov(a,f,i,:) + H2A_oovv(m,n,:,b) * t_amp ! (bf)
                      I2A_vvov(a,f,m,:) = I2A_vvov(a,f,m,:) - H2A_oovv(i,n,:,b) * t_amp ! (im)(bf)
                      I2A_vvov(a,f,n,:) = I2A_vvov(a,f,n,:) - H2A_oovv(m,i,:,b) * t_amp ! (in)(bf)
                  end do
                  ! Loop over aab determinants
                  do idet = 1, n3aab
                      t_amp = t3b_amps(idet)

                      ! x3a(abcijk) <- A(c/ab)A(ab)A(ij)A(k/ij)[ h2b(cmke) * t3b(abeijm) ]
                      !              = A(abc)A(ijk)[ h2b(cmke) * t3b(abeijm) ]
                      a = t3b_excits(idet,1); b = t3b_excits(idet,2); e = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); j = t3b_excits(idet,5); m = t3b_excits(idet,6);
                      x3a(a,b,:,i,j,:) = x3a(a,b,:,i,j,:) + H2B_voov(:,m,:,e) * t_amp ! (1)

                      ! I2A(amij) <- A(ij) [A(ae) h2b(mnef) * t3b(aefijn)]
                      a = t3b_excits(idet,1); e = t3b_excits(idet,2); f = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); j = t3b_excits(idet,5); n = t3b_excits(idet,6);
                      I2A_vooo(a,:,i,j) = I2A_vooo(a,:,i,j) + H2B_oovv(:,n,e,f) * t_amp ! (1)
                      I2A_vooo(e,:,i,j) = I2A_vooo(e,:,i,j) - H2B_oovv(:,n,a,f) * t_amp ! (ae)

                      ! I2A(abie) <- A(ab) [A(im) -h2b(mnef) * t3b(abfimn)]
                      a = t3b_excits(idet,1); b = t3b_excits(idet,2); f = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); m = t3b_excits(idet,5); n = t3b_excits(idet,6);
                      I2A_vvov(a,b,i,:) = I2A_vvov(a,b,i,:) - H2B_oovv(m,n,:,f) * t_amp ! (1)
                      I2A_vvov(a,b,m,:) = I2A_vvov(a,b,m,:) + H2B_oovv(i,n,:,f) * t_amp ! (im)
                  end do

                  ! Update loop
                  do idet = 1, n3aaa
                      a = t3a_excits(idet, 1); b = t3a_excits(idet, 2); c = t3a_excits(idet, 3);
                      i = t3a_excits(idet, 4); j = t3a_excits(idet, 5); k = t3a_excits(idet, 6);

                      denom = fA_oo(i,i) + fA_oo(j,j) + fA_oo(k,k) - fA_vv(a,a) - fA_vv(b,b) - fA_vv(c,c)
                        
                      res_mm23 = 0.0d0
                      do e = 1, nua
                           ! A(i/jk)(c/ab) h2a(abie) * t2a(ecjk)
                          res_mm23 = res_mm23 + (I2A_vvov(a,b,i,e) - I2A_vvov(b,a,i,e)) * t2a(e,c,j,k)
                          res_mm23 = res_mm23 - (I2A_vvov(c,b,i,e) - I2A_vvov(b,c,i,e)) * t2a(e,a,j,k)
                          res_mm23 = res_mm23 - (I2A_vvov(a,c,i,e) - I2A_vvov(c,a,i,e)) * t2a(e,b,j,k)
                          res_mm23 = res_mm23 - (I2A_vvov(a,b,j,e) - I2A_vvov(b,a,j,e)) * t2a(e,c,i,k)
                          res_mm23 = res_mm23 + (I2A_vvov(c,b,j,e) - I2A_vvov(b,c,j,e)) * t2a(e,a,i,k)
                          res_mm23 = res_mm23 + (I2A_vvov(a,c,j,e) - I2A_vvov(c,a,j,e)) * t2a(e,b,i,k)
                          res_mm23 = res_mm23 - (I2A_vvov(a,b,k,e) - I2A_vvov(b,a,k,e)) * t2a(e,c,j,i)
                          res_mm23 = res_mm23 + (I2A_vvov(c,b,k,e) - I2A_vvov(b,c,k,e)) * t2a(e,a,j,i)
                          res_mm23 = res_mm23 + (I2A_vvov(a,c,k,e) - I2A_vvov(c,a,k,e)) * t2a(e,b,j,i)
                      end do
                      do m = 1, noa
                          ! -A(k/ij)A(a/bc) h2a(amij) * t2a(bcmk)
                          res_mm23 = res_mm23 - (I2A_vooo(a,m,i,j) - I2A_vooo(a,m,j,i)) * t2a(b,c,m,k)
                          res_mm23 = res_mm23 + (I2A_vooo(b,m,i,j) - I2A_vooo(b,m,j,i)) * t2a(a,c,m,k)
                          res_mm23 = res_mm23 + (I2A_vooo(c,m,i,j) - I2A_vooo(c,m,j,i)) * t2a(b,a,m,k)
                          res_mm23 = res_mm23 + (I2A_vooo(a,m,k,j) - I2A_vooo(a,m,j,k)) * t2a(b,c,m,i)
                          res_mm23 = res_mm23 - (I2A_vooo(b,m,k,j) - I2A_vooo(b,m,j,k)) * t2a(a,c,m,i)
                          res_mm23 = res_mm23 - (I2A_vooo(c,m,k,j) - I2A_vooo(c,m,j,k)) * t2a(b,a,m,i)
                          res_mm23 = res_mm23 + (I2A_vooo(a,m,i,k) - I2A_vooo(a,m,k,i)) * t2a(b,c,m,j)
                          res_mm23 = res_mm23 - (I2A_vooo(b,m,i,k) - I2A_vooo(b,m,k,i)) * t2a(a,c,m,j)
                          res_mm23 = res_mm23 - (I2A_vooo(c,m,i,k) - I2A_vooo(c,m,k,i)) * t2a(b,a,m,j)
                      end do

                      ! fully antisymmetrize x3a(abcijk)
                      val = &
                       x3a(a,b,c,i,j,k) - x3a(a,c,b,i,j,k) + x3a(b,c,a,i,j,k) - x3a(b,a,c,i,j,k) + x3a(c,a,b,i,j,k) - x3a(c,b,a,i,j,k)&
                      -x3a(a,b,c,i,k,j) + x3a(a,c,b,i,k,j) - x3a(b,c,a,i,k,j) + x3a(b,a,c,i,k,j) - x3a(c,a,b,i,k,j) + x3a(c,b,a,i,k,j)&
                      +x3a(a,b,c,j,k,i) - x3a(a,c,b,j,k,i) + x3a(b,c,a,j,k,i) - x3a(b,a,c,j,k,i) + x3a(c,a,b,j,k,i) - x3a(c,b,a,j,k,i)&
                      -x3a(a,b,c,j,i,k) + x3a(a,c,b,j,i,k) - x3a(b,c,a,j,i,k) + x3a(b,a,c,j,i,k) - x3a(c,a,b,j,i,k) + x3a(c,b,a,j,i,k)&
                      +x3a(a,b,c,k,i,j) - x3a(a,c,b,k,i,j) + x3a(b,c,a,k,i,j) - x3a(b,a,c,k,i,j) + x3a(c,a,b,k,i,j) - x3a(c,b,a,k,i,j)&
                      -x3a(a,b,c,k,j,i) + x3a(a,c,b,k,j,i) - x3a(b,c,a,k,j,i) + x3a(b,a,c,k,j,i) - x3a(c,a,b,k,j,i) + x3a(c,b,a,k,j,i)
                      val = (val + res_mm23)/(denom - shift)

                      t3a_amps(idet) = t3a_amps(idet) + val

                      resid(idet) = val
                  end do

              end subroutine update_t3a_p

              subroutine update_t3b_p(t3b_amps, resid,&
                                      t3a_excits, t3b_excits, t3c_excits,&
                                      t2a, t2b,&
                                      t3a_amps, t3c_amps,&
                                      H1A_oo, H1A_vv, H1B_oo, H1B_vv,&
                                      H2A_oovv, H2A_vvov, H2A_vooo, H2A_oooo, H2A_voov, H2A_vvvv,&
                                      H2B_oovv, H2B_vvov, H2B_vvvo, H2B_vooo, H2B_ovoo,&
                                      H2B_oooo, H2B_voov, H2B_vovo, H2B_ovov, H2B_ovvo, H2B_vvvv,&
                                      H2C_oovv, H2C_voov,&
                                      fA_oo, fA_vv, fB_oo, fB_vv,&
                                      shift,&
                                      n3aaa, n3aab, n3abb,&
                                      noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub, n3aaa, n3aab, n3abb
                  integer, intent(in) :: t3a_excits(n3aaa, 6), t3b_excits(n3aab, 6), t3c_excits(n3abb, 6)
                  real(kind=8), intent(in) :: t2a(1:nua,1:nua,1:noa,1:noa),&
                                              t2b(1:nua,1:nub,1:noa,1:nob),&
                                              t3a_amps(n3aaa), t3c_amps(n3abb),&
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

                  real(kind=8), intent(inout) :: t3b_amps(n3aab)
                  !f2py intent(in,out) :: t3b_amps(0:n3aab-1)

                  real(kind=8), intent(out) :: resid(n3aab)

                  real(kind=8) :: I2A_vooo(nua, noa, noa, noa),&
                                  I2A_vvov(nua, nua, noa, nua),&
                                  I2B_vooo(nua, nob, noa, nob),&
                                  I2B_ovoo(noa, nub, noa, nob),&
                                  I2B_vvov(nua, nub, noa, nub),&
                                  I2B_vvvo(nua, nub, nua, nob)
                  real(kind=8) :: denom, val, t_amp, res_mm23
                  integer :: i, j, k, a, b, c, m, n, e, f, idet

                  real(kind=8) :: x3b(nua,nua,nub,noa,noa,nob)

                  ! compute VT3 intermediates
                  I2A_vooo(:,:,:,:) = 0.5d0 * H2A_vooo(:,:,:,:)
                  I2A_vvov(:,:,:,:) = 0.5d0 * H2A_vvov(:,:,:,:)
                  I2B_vooo(:,:,:,:) = H2B_vooo(:,:,:,:)
                  I2B_ovoo(:,:,:,:) = H2B_ovoo(:,:,:,:)
                  I2B_vvov(:,:,:,:) = H2B_vvov(:,:,:,:)
                  I2B_vvvo(:,:,:,:) = H2B_vvvo(:,:,:,:)

                  ! Zero the projection container
                  x3b = 0.0d0
                  do idet = 1, n3aaa
                      t_amp = t3a_amps(idet)

                      ! x3b(abcijk) <- A(ij)A(ab) [A(m/ij)A(e/ab) h2b(mcek) * t3a(abeijm)]
                      a = t3a_excits(idet,1); b = t3a_excits(idet,2); e = t3a_excits(idet,3);
                      i = t3a_excits(idet,4); j = t3a_excits(idet,5); m = t3a_excits(idet,6);
                      x3b(a,b,:,i,j,:) = x3b(a,b,:,i,j,:) + H2B_ovvo(m,:,e,:) * t_amp ! (1)
                      x3b(a,b,:,m,j,:) = x3b(a,b,:,m,j,:) - H2B_ovvo(i,:,e,:) * t_amp ! (im)
                      x3b(a,b,:,i,m,:) = x3b(a,b,:,i,m,:) - H2B_ovvo(j,:,e,:) * t_amp ! (jm)
                      x3b(e,b,:,i,j,:) = x3b(e,b,:,i,j,:) - H2B_ovvo(m,:,a,:) * t_amp ! (ae)
                      x3b(e,b,:,m,j,:) = x3b(e,b,:,m,j,:) + H2B_ovvo(i,:,a,:) * t_amp ! (im)(ae)
                      x3b(e,b,:,i,m,:) = x3b(e,b,:,i,m,:) + H2B_ovvo(j,:,a,:) * t_amp ! (jm)(ae)
                      x3b(a,e,:,i,j,:) = x3b(a,e,:,i,j,:) - H2B_ovvo(m,:,b,:) * t_amp ! (be)
                      x3b(a,e,:,m,j,:) = x3b(a,e,:,m,j,:) + H2B_ovvo(i,:,b,:) * t_amp ! (im)(be)
                      x3b(a,e,:,i,m,:) = x3b(a,e,:,i,m,:) + H2B_ovvo(j,:,b,:) * t_amp ! (jm)(be)

                      ! I2A(amij) <- A(ij) [A(n/ij)A(a/ef) h2a(mnef) * t3a(aefijn)]
                      a = t3a_excits(idet,1); e = t3a_excits(idet,2); f = t3a_excits(idet,3);
                      i = t3a_excits(idet,4); j = t3a_excits(idet,5); n = t3a_excits(idet,6);
                      I2A_vooo(a,:,i,j) = I2A_vooo(a,:,i,j) + H2A_oovv(:,n,e,f) * t_amp ! (1)
                      I2A_vooo(a,:,n,j) = I2A_vooo(a,:,n,j) - H2A_oovv(:,i,e,f) * t_amp ! (in)
                      I2A_vooo(a,:,i,n) = I2A_vooo(a,:,i,n) - H2A_oovv(:,j,e,f) * t_amp ! (jn)
                      I2A_vooo(e,:,i,j) = I2A_vooo(e,:,i,j) - H2A_oovv(:,n,a,f) * t_amp ! (ae)
                      I2A_vooo(e,:,n,j) = I2A_vooo(e,:,n,j) + H2A_oovv(:,i,a,f) * t_amp ! (in)(ae)
                      I2A_vooo(e,:,i,n) = I2A_vooo(e,:,i,n) + H2A_oovv(:,j,a,f) * t_amp ! (jn)(ae)
                      I2A_vooo(f,:,i,j) = I2A_vooo(f,:,i,j) - H2A_oovv(:,n,e,a) * t_amp ! (af)
                      I2A_vooo(f,:,n,j) = I2A_vooo(f,:,n,j) + H2A_oovv(:,i,e,a) * t_amp ! (in)(af)
                      I2A_vooo(f,:,i,n) = I2A_vooo(f,:,i,n) + H2A_oovv(:,j,e,a) * t_amp ! (jn)(af)

                      ! I2A(abie) <- A(ab) [A(i/mn)A(f/ab) -h2a(mnef) * t3a(abfimn)]
                      a = t3a_excits(idet,1); b = t3a_excits(idet,2); f = t3a_excits(idet,3);
                      i = t3a_excits(idet,4); m = t3a_excits(idet,5); n = t3a_excits(idet,6);
                      I2A_vvov(a,b,i,:) = I2A_vvov(a,b,i,:) - H2A_oovv(m,n,:,f) * t_amp ! (1)
                      I2A_vvov(a,b,m,:) = I2A_vvov(a,b,m,:) + H2A_oovv(i,n,:,f) * t_amp ! (im)
                      I2A_vvov(a,b,n,:) = I2A_vvov(a,b,n,:) + H2A_oovv(m,i,:,f) * t_amp ! (in)
                      I2A_vvov(f,b,i,:) = I2A_vvov(f,b,i,:) + H2A_oovv(m,n,:,a) * t_amp ! (af)
                      I2A_vvov(f,b,m,:) = I2A_vvov(f,b,m,:) - H2A_oovv(i,n,:,a) * t_amp ! (im)(af)
                      I2A_vvov(f,b,n,:) = I2A_vvov(f,b,n,:) - H2A_oovv(m,i,:,a) * t_amp ! (in)(af)
                      I2A_vvov(a,f,i,:) = I2A_vvov(a,f,i,:) + H2A_oovv(m,n,:,b) * t_amp ! (bf)
                      I2A_vvov(a,f,m,:) = I2A_vvov(a,f,m,:) - H2A_oovv(i,n,:,b) * t_amp ! (im)(bf)
                      I2A_vvov(a,f,n,:) = I2A_vvov(a,f,n,:) - H2A_oovv(m,i,:,b) * t_amp ! (in)(bf)
                  end do

                  do idet = 1, n3aab
                      t_amp = t3b_amps(idet)

                      ! x3b(abcijk) <- A(ij)A(ab) [A(mj) -h1a(mi) * t3b(abcmjk)]
                      a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                      m = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                      x3b(a,b,c,:,j,k) = x3b(a,b,c,:,j,k) - H1A_oo(m,:) * t_amp ! (1)
                      x3b(a,b,c,:,m,k) = x3b(a,b,c,:,m,k) + H1A_oo(j,:) * t_amp ! (mj)

                      ! x3b(abcijk) <- A(ij)A(ab) [-h1b(mk) * t3b(abcijm)]
                      a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); j = t3b_excits(idet,5); m = t3b_excits(idet,6);
                      x3b(a,b,c,i,j,:) = x3b(a,b,c,i,j,:) - H1B_oo(m,:) * t_amp ! (1)

                      ! x3b(abcijk) <- A(ij)A(ab) [A(be) h1a(ae) * t3b(ebcijk)]
                      e = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                      x3b(:,b,c,i,j,k) = x3b(:,b,c,i,j,k) + H1A_vv(:,e) * t_amp ! (1)
                      x3b(:,e,c,i,j,k) = x3b(:,e,c,i,j,k) - H1A_vv(:,b) * t_amp ! (eb)

                      ! x3b(abcijk) <- A(ij)A(ab) [h1b(ce) * t3b(abeijk)]
                      a = t3b_excits(idet,1); b = t3b_excits(idet,2); e = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                      x3b(a,b,:,i,j,k) = x3b(a,b,:,i,j,k) + H1B_vv(:,e) * t_amp ! (1)

                      ! x3b(abcijk) <- A(ij)A(ab) [1/2 h2a(mnij) * t3b(abcmnk)]
                      a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                      m = t3b_excits(idet,4); n = t3b_excits(idet,5); k = t3b_excits(idet,6);
                      x3b(a,b,c,:,:,k) = x3b(a,b,c,:,:,k) + 0.5d0 * H2A_oooo(m,n,:,:) * t_amp ! (1)

                      ! x3b(abcijk) <- A(ij)A(ab) [A(im) h2b(mnjk) * t3b(abcimn)]
                      a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); m = t3b_excits(idet,5); n = t3b_excits(idet,6);
                      x3b(a,b,c,i,:,:) = x3b(a,b,c,i,:,:) + H2B_oooo(m,n,:,:) * t_amp ! (1)
                      x3b(a,b,c,m,:,:) = x3b(a,b,c,m,:,:) - H2B_oooo(i,n,:,:) * t_amp ! (im)

                      ! x3b(abcijk) <- A(ij)A(ab) [1/2 h2a(abef) * t3b(efcijk)]
                      e = t3b_excits(idet,1); f = t3b_excits(idet,2); c = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                      x3b(:,:,c,i,j,k) = x3b(:,:,c,i,j,k) + 0.5d0 * H2A_vvvv(:,:,e,f) * t_amp ! (1)

                      ! x3b(abcijk) <- A(ij)A(ab) [A(ae) h2b(bcef) * t3b(aefijk)]
                      a = t3b_excits(idet,1); e = t3b_excits(idet,2); f = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                      x3b(a,:,:,i,j,k) = x3b(a,:,:,i,j,k) + H2B_vvvv(:,:,e,f) * t_amp ! (1)
                      x3b(e,:,:,i,j,k) = x3b(e,:,:,i,j,k) - H2B_vvvv(:,:,a,f) * t_amp ! (ae)

                      ! x3b(abcijk) <- A(ij)A(ab) [A(be)A(jm) h2a(amie) * t3b(ebcmjk)]
                      e = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                      m = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                      x3b(:,b,c,:,j,k) = x3b(:,b,c,:,j,k) + H2A_voov(:,m,:,e) * t_amp ! (1)
                      x3b(:,e,c,:,j,k) = x3b(:,e,c,:,j,k) - H2A_voov(:,m,:,b) * t_amp ! (be)
                      x3b(:,b,c,:,m,k) = x3b(:,b,c,:,m,k) - H2A_voov(:,j,:,e) * t_amp ! (jm)
                      x3b(:,e,c,:,m,k) = x3b(:,e,c,:,m,k) + H2A_voov(:,j,:,b) * t_amp ! (be)(jm)

                      ! x3b(abcijk) <- A(ij)A(ab) [h2c(cmke) * t3b(abeijm)]
                      a = t3b_excits(idet,1); b = t3b_excits(idet,2); e = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); j = t3b_excits(idet,5); m = t3b_excits(idet,6);
                      x3b(a,b,:,i,j,:) = x3b(a,b,:,i,j,:) + H2C_voov(:,m,:,e) * t_amp ! (1)

                      ! x3b(abcijk) <- A(ij)A(ab) [A(be) -h2b(amek) * t3b(ebcijm)]
                      e = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); j = t3b_excits(idet,5); m = t3b_excits(idet,6);
                      x3b(:,b,c,i,j,:) = x3b(:,b,c,i,j,:) - H2B_vovo(:,m,e,:) * t_amp ! (1)
                      x3b(:,e,c,i,j,:) = x3b(:,e,c,i,j,:) + H2B_vovo(:,m,b,:) * t_amp ! (be)

                      ! x3b(abcijk) <- A(ij)A(ab) [A(jm) -h2b(mcie) * t3b(abemjk)]
                      a = t3b_excits(idet,1); b = t3b_excits(idet,2); e = t3b_excits(idet,3);
                      m = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                      x3b(a,b,:,:,j,k) = x3b(a,b,:,:,j,k) - H2B_ovov(m,:,:,e) * t_amp ! (1)
                      x3b(a,b,:,:,m,k) = x3b(a,b,:,:,m,k) + H2B_ovov(j,:,:,e) * t_amp ! (jm)

                      ! I2A(amij) <- A(ij) [A(ae) h2b(mnef) * t3b(aefijn)]
                      a = t3b_excits(idet,1); e = t3b_excits(idet,2); f = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); j = t3b_excits(idet,5); n = t3b_excits(idet,6);
                      I2A_vooo(a,:,i,j) = I2A_vooo(a,:,i,j) + H2B_oovv(:,n,e,f) * t_amp ! (1)
                      I2A_vooo(e,:,i,j) = I2A_vooo(e,:,i,j) - H2B_oovv(:,n,a,f) * t_amp ! (ae)

                      ! I2A(abie) <- A(ab) [A(im) -h2b(mnef) * t3b(abfimn)]
                      a = t3b_excits(idet,1); b = t3b_excits(idet,2); f = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); m = t3b_excits(idet,5); n = t3b_excits(idet,6);
                      I2A_vvov(a,b,i,:) = I2A_vvov(a,b,i,:) - H2B_oovv(m,n,:,f) * t_amp ! (1)
                      I2A_vvov(a,b,m,:) = I2A_vvov(a,b,m,:) + H2B_oovv(i,n,:,f) * t_amp ! (im)

                      ! I2B(abej) <- A(af) -h2a(mnef) * t3b(afbmnj)
                      a = t3b_excits(idet,1); f = t3b_excits(idet,2); b = t3b_excits(idet,3);
                      m = t3b_excits(idet,4); n = t3b_excits(idet,5); j = t3b_excits(idet,6);
                      I2B_vvvo(a,b,:,j) = I2B_vvvo(a,b,:,j) - H2A_oovv(m,n,:,f) * t_amp ! (1)
                      I2B_vvvo(f,b,:,j) = I2B_vvvo(f,b,:,j) + H2A_oovv(m,n,:,a) * t_amp ! (af)

                      ! I2B(mbij) <- A(in) h2a(mnef) * t3b(efbinj)
                      e = t3b_excits(idet,1); f = t3b_excits(idet,2); b = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); n = t3b_excits(idet,5); j = t3b_excits(idet,6);
                      I2B_ovoo(:,b,i,j) = I2B_ovoo(:,b,i,j) + H2A_oovv(:,n,e,f) * t_amp ! (1)
                      I2B_ovoo(:,b,n,j) = I2B_ovoo(:,b,n,j) - H2A_oovv(:,i,e,f) * t_amp ! (in)

                      ! I2B(abie) <- A(af)A(in) -h2b(nmfe) * t3b(afbinm)
                      a = t3b_excits(idet,1); f = t3b_excits(idet,2); b = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); n = t3b_excits(idet,5); m = t3b_excits(idet,6);
                      I2B_vvov(a,b,i,:) = I2B_vvov(a,b,i,:) - H2B_oovv(n,m,f,:) * t_amp ! (1)
                      I2B_vvov(f,b,i,:) = I2B_vvov(f,b,i,:) + H2B_oovv(n,m,a,:) * t_amp ! (af)
                      I2B_vvov(a,b,n,:) = I2B_vvov(a,b,n,:) + H2B_oovv(i,m,f,:) * t_amp ! (in)
                      I2B_vvov(f,b,n,:) = I2B_vvov(f,b,n,:) - H2B_oovv(i,m,a,:) * t_amp ! (af)(in)

                      ! I2B(amij) <- A(af)A(in) h2b(nmfe) * t3b(afeinj)
                      a = t3b_excits(idet,1); f = t3b_excits(idet,2); e = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); n = t3b_excits(idet,5); j = t3b_excits(idet,6);
                      I2B_vooo(a,:,i,j) = I2B_vooo(a,:,i,j) + H2B_oovv(n,:,f,e) * t_amp ! (1)
                      I2B_vooo(f,:,i,j) = I2B_vooo(f,:,i,j) - H2B_oovv(n,:,a,e) * t_amp ! (af)
                      I2B_vooo(a,:,n,j) = I2B_vooo(a,:,n,j) - H2B_oovv(i,:,f,e) * t_amp ! (in)
                      I2B_vooo(f,:,n,j) = I2B_vooo(f,:,n,j) + H2B_oovv(i,:,a,e) * t_amp ! (af)(in)
                  end do

                  do idet = 1, n3abb
                      t_amp = t3c_amps(idet)

                      ! x3b(abcijk) <- A(ij)A(ab) [A(ec)A(mk) h2b(amie) * t3c(becjmk)]
                      b = t3c_excits(idet,1); e = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      j = t3c_excits(idet,4); m = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      x3b(:,b,c,:,j,k) = x3b(:,b,c,:,j,k) + H2B_voov(:,m,:,e) * t_amp ! (1)
                      x3b(:,b,e,:,j,k) = x3b(:,b,e,:,j,k) - H2B_voov(:,m,:,c) * t_amp ! (ec)
                      x3b(:,b,c,:,j,m) = x3b(:,b,c,:,j,m) - H2B_voov(:,k,:,e) * t_amp ! (mk)
                      x3b(:,b,e,:,j,m) = x3b(:,b,e,:,j,m) + H2B_voov(:,k,:,c) * t_amp ! (ec)(mk)

                      ! I2B(abej) <- A(bf)A(jn) -h2b(mnef) * t3c(afbmnj)
                      a = t3c_excits(idet,1); f = t3c_excits(idet,2); b = t3c_excits(idet,3);
                      m = t3c_excits(idet,4); n = t3c_excits(idet,5); j = t3c_excits(idet,6);
                      I2B_vvvo(a,b,:,j) = I2B_vvvo(a,b,:,j) - H2B_oovv(m,n,:,f) * t_amp ! (1)
                      I2B_vvvo(a,f,:,j) = I2B_vvvo(a,f,:,j) + H2B_oovv(m,n,:,b) * t_amp ! (bf)
                      I2B_vvvo(a,b,:,n) = I2B_vvvo(a,b,:,n) + H2B_oovv(m,j,:,f) * t_amp ! (jn)
                      I2B_vvvo(a,f,:,n) = I2B_vvvo(a,f,:,n) - H2B_oovv(m,j,:,b) * t_amp ! (bf)(jn)

                      ! I2B(mbij) <- A(bf)A(jn) h2B(mnef) * t3c(efbinj)
                      e = t3c_excits(idet,1); f = t3c_excits(idet,2); b = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); n = t3c_excits(idet,5); j = t3c_excits(idet,6);
                      I2B_ovoo(:,b,i,j) = I2B_ovoo(:,b,i,j) + H2B_oovv(:,n,e,f) * t_amp ! (1)
                      I2B_ovoo(:,f,i,j) = I2B_ovoo(:,f,i,j) - H2B_oovv(:,n,e,b) * t_amp ! (bf)
                      I2B_ovoo(:,b,i,n) = I2B_ovoo(:,b,i,n) - H2B_oovv(:,j,e,f) * t_amp ! (jn)
                      I2B_ovoo(:,f,i,n) = I2B_ovoo(:,f,i,n) + H2B_oovv(:,j,e,b) * t_amp ! (bf)(jn)

                      ! I2B(abie) <- A(bf) -h2c(nmfe) * t3c(afbinm)
                      a = t3c_excits(idet,1); f = t3c_excits(idet,2); b = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); n = t3c_excits(idet,5); m = t3c_excits(idet,6);
                      I2B_vvov(a,b,i,:) = I2B_vvov(a,b,i,:) - H2C_oovv(n,m,f,:) * t_amp ! (1)
                      I2B_vvov(a,f,i,:) = I2B_vvov(a,f,i,:) + H2C_oovv(n,m,b,:) * t_amp ! (bf)

                      ! I2B(amij) <- A(jn) h2c(nmfe) * t3c(afeinj)
                      a = t3c_excits(idet,1); f = t3c_excits(idet,2); e = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); n = t3c_excits(idet,5); j = t3c_excits(idet,6);
                      I2B_vooo(a,:,i,j) = I2B_vooo(a,:,i,j) + H2C_oovv(n,:,f,e) * t_amp ! (1)
                      I2B_vooo(a,:,i,n) = I2B_vooo(a,:,i,n) - H2C_oovv(j,:,f,e) * t_amp ! (jn)
                  end do

                  ! Update loop
                  do idet = 1, n3aab
                      a = t3b_excits(idet, 1); b = t3b_excits(idet, 2); c = t3b_excits(idet, 3);
                      i = t3b_excits(idet, 4); j = t3b_excits(idet, 5); k = t3b_excits(idet, 6);

                      denom = fA_oo(i,i) + fA_oo(j,j) + fB_oo(k,k) - fA_vv(a,a) - fA_vv(b,b) - fB_vv(c,c)

                      res_mm23 = 0.0d0
                      do e = 1, nua
                          ! A(ab) I2B(bcek) * t2a(aeij)
                          res_mm23 = res_mm23 + I2B_vvvo(b,c,e,k) * t2a(a,e,i,j)
                          res_mm23 = res_mm23 - I2B_vvvo(a,c,e,k) * t2a(b,e,i,j)
                          ! A(ij) I2A(abie) * t2b(ecjk)
                          res_mm23 = res_mm23 + (I2A_vvov(a,b,i,e) - I2A_vvov(b,a,i,e)) * t2b(e,c,j,k)
                          res_mm23 = res_mm23 - (I2A_vvov(a,b,j,e) - I2A_vvov(b,a,j,e)) * t2b(e,c,i,k)
                      end do
                      do e = 1, nub
                          ! A(ij)A(ab) I2B(acie) * t2b(bejk)
                          res_mm23 = res_mm23 + I2B_vvov(a,c,i,e) * t2b(b,e,j,k)
                          res_mm23 = res_mm23 - I2B_vvov(a,c,j,e) * t2b(b,e,i,k)
                          res_mm23 = res_mm23 - I2B_vvov(b,c,i,e) * t2b(a,e,j,k)
                          res_mm23 = res_mm23 + I2B_vvov(b,c,j,e) * t2b(a,e,i,k)
                      end do
                      do m = 1, noa
                          ! -A(ij) h2b(mcjk) * t2a(abim) 
                          res_mm23 = res_mm23 - I2B_ovoo(m,c,j,k) * t2a(a,b,i,m)
                          res_mm23 = res_mm23 + I2B_ovoo(m,c,i,k) * t2a(a,b,j,m)
                          ! -A(ab) h2a(amij) * t2b(bcmk)
                          res_mm23 = res_mm23 - (I2A_vooo(a,m,i,j) - I2A_vooo(a,m,j,i)) * t2b(b,c,m,k)
                          res_mm23 = res_mm23 + (I2A_vooo(b,m,i,j) - I2A_vooo(b,m,j,i)) * t2b(a,c,m,k)
                      end do
                      do m = 1, nob
                          ! -A(ij)A(ab) h2b(amik) * t2b(bcjm)
                          res_mm23 = res_mm23 - I2B_vooo(a,m,i,k) * t2b(b,c,j,m)
                          res_mm23 = res_mm23 + I2B_vooo(b,m,i,k) * t2b(a,c,j,m)
                          res_mm23 = res_mm23 + I2B_vooo(a,m,j,k) * t2b(b,c,i,m)
                          res_mm23 = res_mm23 - I2B_vooo(b,m,j,k) * t2b(a,c,i,m)
                      end do

                      ! fully antisymmetrize x3a(abcijk)
                      val = x3b(a,b,c,i,j,k) - x3b(b,a,c,i,j,k) - x3b(a,b,c,j,i,k) + x3b(b,a,c,j,i,k)
                      val = (val + res_mm23)/(denom - shift)

                      t3b_amps(idet) = t3b_amps(idet) + val

                      resid(idet) = val
                  end do

              end subroutine update_t3b_p

              subroutine update_t3c_p(t3c_amps, resid,&
                                      t3b_excits, t3c_excits, t3d_excits,&
                                      t2b, t2c,&
                                      t3b_amps, t3d_amps,&
                                      H1A_oo, H1A_vv, H1B_oo, H1B_vv,&
                                      H2A_oovv, H2A_voov,&
                                      H2B_oovv, H2B_vooo, H2B_ovoo, H2B_vvov, H2B_vvvo, H2B_oooo,&
                                      H2B_voov, H2B_vovo, H2B_ovov, H2B_ovvo, H2B_vvvv,&
                                      H2C_oovv, H2C_vooo, H2C_vvov, H2C_oooo, H2C_voov, H2C_vvvv,&
                                      fA_oo, fA_vv, fB_oo, fB_vv,&
                                      shift,&
                                      n3aab, n3abb, n3bbb,&
                                      noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub, n3aab, n3abb, n3bbb
                  integer, intent(in) :: t3b_excits(n3aab, 6), t3c_excits(n3abb, 6), t3d_excits(n3bbb, 6)
                  real(kind=8), intent(in) :: t2b(1:nua,1:nub,1:noa,1:nob),&
                                              t2c(1:nub,1:nub,1:nob,1:nob),&
                                              t3b_amps(n3aab),&
                                              t3d_amps(n3bbb),&
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

                  real(kind=8), intent(inout) :: t3c_amps(n3abb)
                  !f2py intent(in,out) :: t3c_amps(0:n3abb-1)

                  real(kind=8), intent(out) :: resid(n3abb)

                  real(kind=8) :: I2C_vooo(nub, nob, nob, nob),&
                                  I2C_vvov(nub, nub, nob, nub),&
                                  I2B_vooo(nua, nob, noa, nob),&
                                  I2B_ovoo(noa, nub, noa, nob),&
                                  I2B_vvov(nua, nub, noa, nub),&
                                  I2B_vvvo(nua, nub, nua, nob)
                  real(kind=8) :: denom, val, t_amp, res_mm23
                  integer :: i, j, k, a, b, c, m, n, e, f, idet

                  real(kind=8) :: x3c(nua,nub,nub,noa,nob,nob)

                  ! VT3 intermediates
                  I2C_vooo(:,:,:,:) = 0.5d0 * H2C_vooo(:,:,:,:)
                  I2C_vvov(:,:,:,:) = 0.5d0 * H2C_vvov(:,:,:,:)
                  I2B_vooo(:,:,:,:) = H2B_vooo(:,:,:,:)
                  I2B_ovoo(:,:,:,:) = H2B_ovoo(:,:,:,:)
                  I2B_vvov(:,:,:,:) = H2B_vvov(:,:,:,:)
                  I2B_vvvo(:,:,:,:) = H2B_vvvo(:,:,:,:)

                  ! Zero the projection container
                  x3c = 0.0d0
                  do idet = 1, n3aab
                      t_amp = t3b_amps(idet)

                      ! x3c(abcijk) <- A(jk)A(bc) [A(im)A(ae) h2b(mbej) * t3b(aecimk)]
                      a = t3b_excits(idet,1); e = t3b_excits(idet,2); c = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); m = t3b_excits(idet,5); k = t3b_excits(idet,6);
                      x3c(a,:,c,i,:,k) = x3c(a,:,c,i,:,k) + H2B_ovvo(m,:,e,:) * t_amp ! (1)
                      x3c(a,:,c,m,:,k) = x3c(a,:,c,m,:,k) - H2B_ovvo(i,:,e,:) * t_amp ! (im)
                      x3c(e,:,c,i,:,k) = x3c(e,:,c,i,:,k) - H2B_ovvo(m,:,a,:) * t_amp ! (ae)
                      x3c(e,:,c,m,:,k) = x3c(e,:,c,m,:,k) + H2B_ovvo(i,:,a,:) * t_amp ! (im)(ae)

                      ! I2B(abej) <- A(af) -h2a(mnef) * t3b(afbmnj)
                      a = t3b_excits(idet,1); f = t3b_excits(idet,2); b = t3b_excits(idet,3);
                      m = t3b_excits(idet,4); n = t3b_excits(idet,5); j = t3b_excits(idet,6);
                      I2B_vvvo(a,b,:,j) = I2B_vvvo(a,b,:,j) - H2A_oovv(m,n,:,f) * t_amp ! (1)
                      I2B_vvvo(f,b,:,j) = I2B_vvvo(f,b,:,j) + H2A_oovv(m,n,:,a) * t_amp ! (af)

                      ! I2B(mbij) <- A(in) h2a(mnef) * t3b(efbinj)
                      e = t3b_excits(idet,1); f = t3b_excits(idet,2); b = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); n = t3b_excits(idet,5); j = t3b_excits(idet,6);
                      I2B_ovoo(:,b,i,j) = I2B_ovoo(:,b,i,j) + H2A_oovv(:,n,e,f) * t_amp ! (1)
                      I2B_ovoo(:,b,n,j) = I2B_ovoo(:,b,n,j) - H2A_oovv(:,i,e,f) * t_amp ! (in)

                      ! I2B(abie) <- A(af)A(in) -h2b(nmfe) * t3b(afbinm)
                      a = t3b_excits(idet,1); f = t3b_excits(idet,2); b = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); n = t3b_excits(idet,5); m = t3b_excits(idet,6);
                      I2B_vvov(a,b,i,:) = I2B_vvov(a,b,i,:) - H2B_oovv(n,m,f,:) * t_amp ! (1)
                      I2B_vvov(f,b,i,:) = I2B_vvov(f,b,i,:) + H2B_oovv(n,m,a,:) * t_amp ! (af)
                      I2B_vvov(a,b,n,:) = I2B_vvov(a,b,n,:) + H2B_oovv(i,m,f,:) * t_amp ! (in)
                      I2B_vvov(f,b,n,:) = I2B_vvov(f,b,n,:) - H2B_oovv(i,m,a,:) * t_amp ! (af)(in)

                      ! I2B(amij) <- A(af)A(in) h2b(nmfe) * t3b(afeinj)
                      a = t3b_excits(idet,1); f = t3b_excits(idet,2); e = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); n = t3b_excits(idet,5); j = t3b_excits(idet,6);
                      I2B_vooo(a,:,i,j) = I2B_vooo(a,:,i,j) + H2B_oovv(n,:,f,e) * t_amp ! (1)
                      I2B_vooo(f,:,i,j) = I2B_vooo(f,:,i,j) - H2B_oovv(n,:,a,e) * t_amp ! (af)
                      I2B_vooo(a,:,n,j) = I2B_vooo(a,:,n,j) - H2B_oovv(i,:,f,e) * t_amp ! (in)
                      I2B_vooo(f,:,n,j) = I2B_vooo(f,:,n,j) + H2B_oovv(i,:,a,e) * t_amp ! (af)(in)
                  end do

                  do idet = 1, n3abb
                      t_amp = t3c_amps(idet)

                      ! x3c(abcijk) <- A(jk)A(bc) [-h1a(mi) * t3c(abcmjk)]
                      a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      m = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      x3c(a,b,c,:,j,k) = x3c(a,b,c,:,j,k) - H1A_oo(m,:) * t_amp ! (1)

                      ! x3c(abcijk) <- A(jk)A(bc) [A(mk) -h1b(mj) * t3c(abcimk)]
                      a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); m = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      x3c(a,b,c,i,:,k) = x3c(a,b,c,i,:,k) - H1B_oo(m,:) * t_amp ! (1)
                      x3c(a,b,c,i,:,m) = x3c(a,b,c,i,:,m) + H1B_oo(k,:) * t_amp ! (mk)

                      ! x3c(abcijk) <- A(jk)A(bc) [h1a(ae) * t3c(ebcijk)]
                      e = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      x3c(:,b,c,i,j,k) = x3c(:,b,c,i,j,k) + H1A_vv(:,e) * t_amp ! (1)

                      ! x3c(abcijk) <- A(jk)A(bc) [A(ec) h1b(be) * t3c(aecijk)]
                      a = t3c_excits(idet,1); e = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      x3c(a,:,c,i,j,k) = x3c(a,:,c,i,j,k) + H1B_vv(:,e) * t_amp ! (1)
                      x3c(a,:,e,i,j,k) = x3c(a,:,e,i,j,k) - H1B_vv(:,c) * t_amp ! (ec)

                      ! x3c(abcijk) <- A(jk)A(bc) [1/2 h2c(mnjk) * t3c(abcimn)]
                      a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); m = t3c_excits(idet,5); n = t3c_excits(idet,6);
                      x3c(a,b,c,i,:,:) = x3c(a,b,c,i,:,:) + 0.5d0 * H2C_oooo(m,n,:,:) * t_amp ! (1)

                      ! x3c(abcijk) <- A(jk)A(bc) [A(kn) h2b(mnij) * t3c(abcmnk)]
                      a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      m = t3c_excits(idet,4); n = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      x3c(a,b,c,:,:,k) = x3c(a,b,c,:,:,k) + H2B_oooo(m,n,:,:) * t_amp ! (1)
                      x3c(a,b,c,:,:,n) = x3c(a,b,c,:,:,n) - H2B_oooo(m,k,:,:) * t_amp ! (kn)

                      ! x3c(abcijk) <- A(jk)A(bc) [1/2 h2c(bcef) * t3c(aefijk)]
                      a = t3c_excits(idet,1); e = t3c_excits(idet,2); f = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      x3c(a,:,:,i,j,k) = x3c(a,:,:,i,j,k) + 0.5d0 * H2C_vvvv(:,:,e,f) * t_amp ! (1)

                      ! x3c(abcijk) <- A(jk)A(bc) [A(fc) h2b(abef) * t3c(efcijk)]
                      e = t3c_excits(idet,1); f = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      x3c(:,:,c,i,j,k) = x3c(:,:,c,i,j,k) + H2B_vvvv(:,:,e,f) * t_amp ! (1)
                      x3c(:,:,f,i,j,k) = x3c(:,:,f,i,j,k) - H2B_vvvv(:,:,e,c) * t_amp ! (fc)

                      ! x3c(abcijk) <- A(jk)A(bc) [h2a(amie) * t3c(ebcmjk)]
                      e = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      m = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      x3c(:,b,c,:,j,k) = x3c(:,b,c,:,j,k) + H2A_voov(:,m,:,e) * t_amp ! (1)

                      ! x3c(abcijk) <- A(jk)A(bc) [A(ec)(mk) h2c(bmje) * t3c(aecimk)]
                      a = t3c_excits(idet,1); e = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); m = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      x3c(a,:,c,i,:,k) = x3c(a,:,c,i,:,k) + H2C_voov(:,m,:,e) * t_amp ! (1)
                      x3c(a,:,e,i,:,k) = x3c(a,:,e,i,:,k) - H2C_voov(:,m,:,c) * t_amp ! (ec)
                      x3c(a,:,c,i,:,m) = x3c(a,:,c,i,:,m) - H2C_voov(:,k,:,e) * t_amp ! (mk)
                      x3c(a,:,e,i,:,m) = x3c(a,:,e,i,:,m) + H2C_voov(:,k,:,c) * t_amp ! (ec)(mk)

                      ! x3c(abcijk) <- A(jk)A(bc) [A(ec) -h2b(mbie) * t3c(aecmjk)]
                      a = t3c_excits(idet,1); e = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      m = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      x3c(a,:,c,:,j,k) = x3c(a,:,c,:,j,k) - H2B_ovov(m,:,:,e) * t_amp ! (1)
                      x3c(a,:,e,:,j,k) = x3c(a,:,e,:,j,k) + H2B_ovov(m,:,:,c) * t_amp ! (ec)

                      ! x3c(abcijk) <- A(jk)A(bc) [A(km) -h2b(amej) * t3c(ebcimk)]
                      e = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); m = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      x3c(:,b,c,i,:,k) = x3c(:,b,c,i,:,k) - H2B_vovo(:,m,e,:) * t_amp ! (1)
                      x3c(:,b,c,i,:,m) = x3c(:,b,c,i,:,m) + H2B_vovo(:,k,e,:) * t_amp ! (km)

                      ! I2C(abie) <- A(ab) [A(im) -h2b(nmfe) * t3c(fabnim)]
                      f = t3c_excits(idet,1); a = t3c_excits(idet,2); b = t3c_excits(idet,3);
                      n = t3c_excits(idet,4); i = t3c_excits(idet,5); m = t3c_excits(idet,6);
                      I2C_vvov(a,b,i,:) = I2C_vvov(a,b,i,:) - H2B_oovv(n,m,f,:) * t_amp ! (1)
                      I2C_vvov(a,b,m,:) = I2C_vvov(a,b,m,:) + H2B_oovv(n,i,f,:) * t_amp ! (im)

                      ! I2C(amij) <- A(ij) [A(ae) h2b(nmfe) * t3c(faenij)]
                      f = t3c_excits(idet,1); a = t3c_excits(idet,2); e = t3c_excits(idet,3);
                      n = t3c_excits(idet,4); i = t3c_excits(idet,5); j = t3c_excits(idet,6);
                      I2C_vooo(a,:,i,j) = I2C_vooo(a,:,i,j) + H2B_oovv(n,:,f,e) * t_amp ! (1)
                      I2C_vooo(e,:,i,j) = I2C_vooo(e,:,i,j) - H2B_oovv(n,:,f,a) * t_amp ! (ae)

                      ! I2B(abej) <- A(bf)A(jn) -h2b(mnef) * t3c(afbmnj)
                      a = t3c_excits(idet,1); f = t3c_excits(idet,2); b = t3c_excits(idet,3);
                      m = t3c_excits(idet,4); n = t3c_excits(idet,5); j = t3c_excits(idet,6);
                      I2B_vvvo(a,b,:,j) = I2B_vvvo(a,b,:,j) - H2B_oovv(m,n,:,f) * t_amp ! (1)
                      I2B_vvvo(a,f,:,j) = I2B_vvvo(a,f,:,j) + H2B_oovv(m,n,:,b) * t_amp ! (bf)
                      I2B_vvvo(a,b,:,n) = I2B_vvvo(a,b,:,n) + H2B_oovv(m,j,:,f) * t_amp ! (jn)
                      I2B_vvvo(a,f,:,n) = I2B_vvvo(a,f,:,n) - H2B_oovv(m,j,:,b) * t_amp ! (bf)(jn)

                      ! I2B(mbij) <- A(bf)A(jn) h2B(mnef) * t3c(efbinj)
                      e = t3c_excits(idet,1); f = t3c_excits(idet,2); b = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); n = t3c_excits(idet,5); j = t3c_excits(idet,6);
                      I2B_ovoo(:,b,i,j) = I2B_ovoo(:,b,i,j) + H2B_oovv(:,n,e,f) * t_amp ! (1)
                      I2B_ovoo(:,f,i,j) = I2B_ovoo(:,f,i,j) - H2B_oovv(:,n,e,b) * t_amp ! (bf)
                      I2B_ovoo(:,b,i,n) = I2B_ovoo(:,b,i,n) - H2B_oovv(:,j,e,f) * t_amp ! (jn)
                      I2B_ovoo(:,f,i,n) = I2B_ovoo(:,f,i,n) + H2B_oovv(:,j,e,b) * t_amp ! (bf)(jn)

                      ! I2B(abie) <- A(bf) -h2c(nmfe) * t3c(afbinm)
                      a = t3c_excits(idet,1); f = t3c_excits(idet,2); b = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); n = t3c_excits(idet,5); m = t3c_excits(idet,6);
                      I2B_vvov(a,b,i,:) = I2B_vvov(a,b,i,:) - H2C_oovv(n,m,f,:) * t_amp ! (1)
                      I2B_vvov(a,f,i,:) = I2B_vvov(a,f,i,:) + H2C_oovv(n,m,b,:) * t_amp ! (bf)

                      ! I2B(amij) <- A(jn) h2c(nmfe) * t3c(afeinj)
                      a = t3c_excits(idet,1); f = t3c_excits(idet,2); e = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); n = t3c_excits(idet,5); j = t3c_excits(idet,6);
                      I2B_vooo(a,:,i,j) = I2B_vooo(a,:,i,j) + H2C_oovv(n,:,f,e) * t_amp ! (1)
                      I2B_vooo(a,:,i,n) = I2B_vooo(a,:,i,n) - H2C_oovv(j,:,f,e) * t_amp ! (jn)
                  end do

                  do idet = 1, n3bbb
                      t_amp = t3d_amps(idet)

                      ! x3c(abcijk) <- A(jk)A(bc) [h2b(amie) * t3d(ebcmjk)]
                      e = t3d_excits(idet,1); b = t3d_excits(idet,2); c = t3d_excits(idet,3);
                      m = t3d_excits(idet,4); j = t3d_excits(idet,5); k = t3d_excits(idet,6);
                      x3c(:,b,c,:,j,k) = x3c(:,b,c,:,j,k) + H2B_voov(:,m,:,e) * t_amp ! (1)
                      x3c(:,b,c,:,m,k) = x3c(:,b,c,:,m,k) - H2B_voov(:,j,:,e) * t_amp ! (jm)
                      x3c(:,b,c,:,j,m) = x3c(:,b,c,:,j,m) - H2B_voov(:,k,:,e) * t_amp ! (km)
                      x3c(:,e,c,:,j,k) = x3c(:,e,c,:,j,k) - H2B_voov(:,m,:,b) * t_amp ! (eb)
                      x3c(:,e,c,:,m,k) = x3c(:,e,c,:,m,k) + H2B_voov(:,j,:,b) * t_amp ! (jm)(eb)
                      x3c(:,e,c,:,j,m) = x3c(:,e,c,:,j,m) + H2B_voov(:,k,:,b) * t_amp ! (km)(eb)
                      x3c(:,b,e,:,j,k) = x3c(:,b,e,:,j,k) - H2B_voov(:,m,:,c) * t_amp ! (ec)
                      x3c(:,b,e,:,m,k) = x3c(:,b,e,:,m,k) + H2B_voov(:,j,:,c) * t_amp ! (jm)(ec)
                      x3c(:,b,e,:,j,m) = x3c(:,b,e,:,j,m) + H2B_voov(:,k,:,c) * t_amp ! (km)(ec)

                      ! I2C(amij) <- A(ij) [A(n/ij)A(a/ef) h2c(mnef) * t3d(aefijn)]
                      a = t3d_excits(idet,1); e = t3d_excits(idet,2); f = t3d_excits(idet,3);
                      i = t3d_excits(idet,4); j = t3d_excits(idet,5); n = t3d_excits(idet,6);
                      I2C_vooo(a,:,i,j) = I2C_vooo(a,:,i,j) + H2C_oovv(:,n,e,f) * t_amp ! (1)
                      I2C_vooo(a,:,n,j) = I2C_vooo(a,:,n,j) - H2C_oovv(:,i,e,f) * t_amp ! (in)
                      I2C_vooo(a,:,i,n) = I2C_vooo(a,:,i,n) - H2C_oovv(:,j,e,f) * t_amp ! (jn)
                      I2C_vooo(e,:,i,j) = I2C_vooo(e,:,i,j) - H2C_oovv(:,n,a,f) * t_amp ! (ae)
                      I2C_vooo(e,:,n,j) = I2C_vooo(e,:,n,j) + H2C_oovv(:,i,a,f) * t_amp ! (in)(ae)
                      I2C_vooo(e,:,i,n) = I2C_vooo(e,:,i,n) + H2C_oovv(:,j,a,f) * t_amp ! (jn)(ae)
                      I2C_vooo(f,:,i,j) = I2C_vooo(f,:,i,j) - H2C_oovv(:,n,e,a) * t_amp ! (af)
                      I2C_vooo(f,:,n,j) = I2C_vooo(f,:,n,j) + H2C_oovv(:,i,e,a) * t_amp ! (in)(af)
                      I2C_vooo(f,:,i,n) = I2C_vooo(f,:,i,n) + H2C_oovv(:,j,e,a) * t_amp ! (jn)(af)

                      ! I2C(abie) <- A(ab) [A(i/mn)A(f/ab) -h2c(mnef) * t3d(abfimn)]
                      a = t3d_excits(idet,1); b = t3d_excits(idet,2); f = t3d_excits(idet,3);
                      i = t3d_excits(idet,4); m = t3d_excits(idet,5); n = t3d_excits(idet,6);
                      I2C_vvov(a,b,i,:) = I2C_vvov(a,b,i,:) - H2C_oovv(m,n,:,f) * t_amp ! (1)
                      I2C_vvov(a,b,m,:) = I2C_vvov(a,b,m,:) + H2C_oovv(i,n,:,f) * t_amp ! (im)
                      I2C_vvov(a,b,n,:) = I2C_vvov(a,b,n,:) + H2C_oovv(m,i,:,f) * t_amp ! (in)
                      I2C_vvov(f,b,i,:) = I2C_vvov(f,b,i,:) + H2C_oovv(m,n,:,a) * t_amp ! (af)
                      I2C_vvov(f,b,m,:) = I2C_vvov(f,b,m,:) - H2C_oovv(i,n,:,a) * t_amp ! (im)(af)
                      I2C_vvov(f,b,n,:) = I2C_vvov(f,b,n,:) - H2C_oovv(m,i,:,a) * t_amp ! (in)(af)
                      I2C_vvov(a,f,i,:) = I2C_vvov(a,f,i,:) + H2C_oovv(m,n,:,b) * t_amp ! (bf)
                      I2C_vvov(a,f,m,:) = I2C_vvov(a,f,m,:) - H2C_oovv(i,n,:,b) * t_amp ! (im)(bf)
                      I2C_vvov(a,f,n,:) = I2C_vvov(a,f,n,:) - H2C_oovv(m,i,:,b) * t_amp ! (in)(bf)
                  end do

                  ! Update loop
                  do idet = 1, n3abb
                      a = t3c_excits(idet, 1); b = t3c_excits(idet, 2); c = t3c_excits(idet, 3);
                      i = t3c_excits(idet, 4); j = t3c_excits(idet, 5); k = t3c_excits(idet, 6);

                      res_mm23 = 0.0
                      do e = 1, nua
                          ! A(jk)A(bc) h2B(abej) * t2b(ecik)
                          res_mm23 = res_mm23 + I2B_vvvo(a,b,e,j) * t2b(e,c,i,k)
                          res_mm23 = res_mm23 - I2B_vvvo(a,b,e,k) * t2b(e,c,i,j)
                          res_mm23 = res_mm23 - I2B_vvvo(a,c,e,j) * t2b(e,b,i,k)
                          res_mm23 = res_mm23 + I2B_vvvo(a,c,e,k) * t2b(e,b,i,j)
                      end do
                      do e = 1, nub
                          ! A(bc) h2B(abie) * t2c(ecjk)
                          res_mm23 = res_mm23 + I2B_vvov(a,b,i,e) * t2c(e,c,j,k)
                          res_mm23 = res_mm23 - I2B_vvov(a,c,i,e) * t2c(e,b,j,k)
                          ! A(jk) h2C(cbke) * t2b(aeij)
                          res_mm23 = res_mm23 + (I2C_vvov(c,b,k,e) - I2C_vvov(b,c,k,e)) * t2b(a,e,i,j)
                          res_mm23 = res_mm23 - (I2C_vvov(c,b,j,e) - I2C_vvov(b,c,j,e)) * t2b(a,e,i,k)
                      end do
                      do m = 1, noa
                          ! -A(kj)A(bc) h2b(mbij) * t2b(acmk)
                          res_mm23 = res_mm23 - I2B_ovoo(m,b,i,j) * t2b(a,c,m,k)
                          res_mm23 = res_mm23 + I2B_ovoo(m,c,i,j) * t2b(a,b,m,k)
                          res_mm23 = res_mm23 + I2B_ovoo(m,b,i,k) * t2b(a,c,m,j)
                          res_mm23 = res_mm23 - I2B_ovoo(m,c,i,k) * t2b(a,b,m,j)
                      end do
                      do m = 1, nob
                          ! -A(jk) h2b(amij) * t2c(bcmk)
                          res_mm23 = res_mm23 - I2B_vooo(a,m,i,j) * t2c(b,c,m,k)
                          res_mm23 = res_mm23 + I2B_vooo(a,m,i,k) * t2c(b,c,m,j)
                          ! -A(bc) h2c(cmkj) * t2b(abim)
                          res_mm23 = res_mm23 - (I2C_vooo(c,m,k,j) - I2C_vooo(c,m,j,k)) * t2b(a,b,i,m)
                          res_mm23 = res_mm23 + (I2C_vooo(b,m,k,j) - I2C_vooo(b,m,j,k)) * t2b(a,c,i,m)
                      end do

                      denom = fA_oo(i,i) + fB_oo(j,j) + fB_oo(k,k) - fA_vv(a,a) - fB_vv(b,b) - fB_vv(c,c)

                      ! fully antisymmetrize x3a(abcijk)
                      val = x3c(a,b,c,i,j,k) - x3c(a,c,b,i,j,k) - x3c(a,b,c,i,k,j) + x3c(a,c,b,i,k,j)
                      val = (val + res_mm23)/(denom - shift)

                      t3c_amps(idet) = t3c_amps(idet) + val

                      resid(idet) = val
                  end do

              end subroutine update_t3c_p

              subroutine update_t3d_p(t3d_amps, resid,&
                                      t3c_excits, t3d_excits,&
                                      t2c,&
                                      t3c_amps,&
                                      H1B_oo, H1B_vv,&
                                      H2B_oovv, H2B_ovvo,&
                                      H2C_oovv, H2C_vooo, H2C_vvov, H2C_oooo, H2C_voov, H2C_vvvv,&
                                      fB_oo, fB_vv,&
                                      shift,&
                                      n3abb, n3bbb,&
                                      noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub, n3abb, n3bbb
                  integer, intent(in) :: t3c_excits(n3abb, 6), t3d_excits(n3bbb, 6)
                  real(kind=8), intent(in) :: t2c(nub, nub, nob, nob),&
                                              t3c_amps(n3abb),&
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

                  real(kind=8), intent(inout) :: t3d_amps(n3bbb)
                  !f2py intent(in,out) :: t3d_amps(0:n3bbb-1)

                  real(kind=8), intent(out) :: resid(n3bbb)

                  real(kind=8) :: val, denom, t_amp, res_mm23
                  real(kind=8) :: I2C_vooo(nub, nob, nob, nob),&
                                  I2C_vvov(nub, nub, nob, nub)
                  integer :: a, b, c, i, j, k, e, f, m, n, idet

                  real(kind=8) :: x3d(nub,nub,nub,nob,nob,nob)

                  ! compute VT3 intermediates
                  I2C_vooo(:,:,:,:) = 0.5d0 * H2C_vooo(:,:,:,:)
                  I2C_vvov(:,:,:,:) = 0.5d0 * H2C_vvov(:,:,:,:)

                  ! Zero the projection container
                  x3d = 0.0d0
                  do idet = 1, n3abb
                      t_amp = t3c_amps(idet)

                      ! x3d(abcijk) <- A(ijk)A(abc) [h2b(maei) * t3c(ebcmjk)]
                      e = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      m = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      x3d(:,b,c,:,j,k) = x3d(:,b,c,:,j,k) + H2B_ovvo(m,:,e,:) * t_amp ! (1)

                      ! I2C(abie) <- A(ab) [A(im) -h2b(nmfe) * t3c(fabnim)]
                      f = t3c_excits(idet,1); a = t3c_excits(idet,2); b = t3c_excits(idet,3);
                      n = t3c_excits(idet,4); i = t3c_excits(idet,5); m = t3c_excits(idet,6);
                      I2C_vvov(a,b,i,:) = I2C_vvov(a,b,i,:) - H2B_oovv(n,m,f,:) * t_amp ! (1)
                      I2C_vvov(a,b,m,:) = I2C_vvov(a,b,m,:) + H2B_oovv(n,i,f,:) * t_amp ! (im)

                      ! I2C(amij) <- A(ij) [A(ae) h2b(nmfe) * t3c(faenij)]
                      f = t3c_excits(idet,1); a = t3c_excits(idet,2); e = t3c_excits(idet,3);
                      n = t3c_excits(idet,4); i = t3c_excits(idet,5); j = t3c_excits(idet,6);
                      I2C_vooo(a,:,i,j) = I2C_vooo(a,:,i,j) + H2B_oovv(n,:,f,e) * t_amp ! (1)
                      I2C_vooo(e,:,i,j) = I2C_vooo(e,:,i,j) - H2B_oovv(n,:,f,a) * t_amp ! (ae)
                  end do

                  do idet = 1, n3bbb
                      t_amp = t3d_amps(idet)

                      ! x3d(abcijk) <- -A(abc)A(i/jk)A(jk)A(m/jk) h1b(mi) * t3d(abcmjk)
                      !               = -A(abc)A(ijk)[ A(m/jk) h1b(mi) * t3d(abcmjk) ]
                      a = t3d_excits(idet,1); b = t3d_excits(idet,2); c = t3d_excits(idet,3);
                      m = t3d_excits(idet,4); j = t3d_excits(idet,5); k = t3d_excits(idet,6);
                      x3d(a,b,c,:,j,k) = x3d(a,b,c,:,j,k) - H1B_oo(m,:) * t_amp ! (1)
                      x3d(a,b,c,:,m,k) = x3d(a,b,c,:,m,k) + H1B_oo(j,:) * t_amp ! (mj)
                      x3d(a,b,c,:,j,m) = x3d(a,b,c,:,j,m) + H1B_oo(k,:) * t_amp ! (mk)

                      ! x3d(abcijk) <- A(abc)A(a/bc)A(bc)A(e/bc) h1b(ae) * t3d(ebcijk)
                      !              = A(abc)A(ijk)[ A(e/bc) h1b(ae) * t3d(ebcijk) ]
                      e = t3d_excits(idet,1); b = t3d_excits(idet,2); c = t3d_excits(idet,3);
                      i = t3d_excits(idet,4); j = t3d_excits(idet,5); k = t3d_excits(idet,6);
                      x3d(:,b,c,i,j,k) = x3d(:,b,c,i,j,k) + H1B_vv(:, e) * t_amp ! (1)
                      x3d(:,e,c,i,j,k) = x3d(:,e,c,i,j,k) - H1B_vv(:, b) * t_amp ! (be)
                      x3d(:,b,e,i,j,k) = x3d(:,b,e,i,j,k) - H1B_vv(:, c) * t_amp ! (ce)

                      ! x3d(abcijk) <- A(abc)A(k/ij)[ A(k/mn) h2c(mnij) * t3d(abcmnk) ]
                      !             = A(abc)A(ijk)[ 1/2 A(k/mn) h2c(mnij) * t3d(abcmnk) ]
                      a = t3d_excits(idet,1); b = t3d_excits(idet,2); c = t3d_excits(idet,3);
                      m = t3d_excits(idet,4); n = t3d_excits(idet,5); k = t3d_excits(idet,6);
                      x3d(a,b,c,:,:,k) = x3d(a,b,c,:,:,k) + 0.5d0 * H2C_oooo(m,n,:,:) * t_amp ! (1)
                      x3d(a,b,c,:,:,m) = x3d(a,b,c,:,:,m) - 0.5d0 * H2C_oooo(k,n,:,:) * t_amp ! (mk)
                      x3d(a,b,c,:,:,n) = x3d(a,b,c,:,:,n) - 0.5d0 * H2C_oooo(m,k,:,:) * t_amp ! (nk)

                      ! x3d(abcijk) <- A(c/ab)A(ijk)[ A(c/ef) h2c(abef) * t3d(efcijk) ]
                      !              = A(abc)A(ijk)[ 1/2 A(c/ef) h2c(abef) * t3d(efcijk) ]
                      e = t3d_excits(idet,1); f = t3d_excits(idet,2); c = t3d_excits(idet,3);
                      i = t3d_excits(idet,4); j = t3d_excits(idet,5); k = t3d_excits(idet,6);
                      x3d(:,:,c,i,j,k) = x3d(:,:,c,i,j,k) + 0.5d0 * H2C_vvvv(:,:,e,f) * t_amp ! (1)
                      x3d(:,:,e,i,j,k) = x3d(:,:,e,i,j,k) - 0.5d0 * H2C_vvvv(:,:,c,f) * t_amp ! (ec)
                      x3d(:,:,f,i,j,k) = x3d(:,:,f,i,j,k) - 0.5d0 * H2C_vvvv(:,:,e,c) * t_amp ! (fc)

                      ! x3d(abcijk) <- A(a/bc)A(bc)A(jk)A(i/jk)[ A(e/bc)A(m/jk) h2c(amie) * t3d(ebcmjk) ]
                      !              = A(abc)A(ijk)[ A(e/bc)A(m/jk) h2c(amie) * t3d(ebcmjk) ]
                      e = t3d_excits(idet,1); b = t3d_excits(idet,2); c = t3d_excits(idet,3);
                      m = t3d_excits(idet,4); j = t3d_excits(idet,5); k = t3d_excits(idet,6);
                      x3d(:,b,c,:,j,k) = x3d(:,b,c,:,j,k) + H2C_voov(:,m,:,e) * t_amp ! (1)
                      x3d(:,b,c,:,m,k) = x3d(:,b,c,:,m,k) - H2C_voov(:,j,:,e) * t_amp ! (mj)
                      x3d(:,b,c,:,j,m) = x3d(:,b,c,:,j,m) - H2C_voov(:,k,:,e) * t_amp ! (mk)
                      x3d(:,e,c,:,j,k) = x3d(:,e,c,:,j,k) - H2C_voov(:,m,:,b) * t_amp ! (eb)
                      x3d(:,e,c,:,m,k) = x3d(:,e,c,:,m,k) + H2C_voov(:,j,:,b) * t_amp ! (eb)(mj)
                      x3d(:,e,c,:,j,m) = x3d(:,e,c,:,j,m) + H2C_voov(:,k,:,b) * t_amp ! (eb)(mk)
                      x3d(:,b,e,:,j,k) = x3d(:,b,e,:,j,k) - H2C_voov(:,m,:,c) * t_amp ! (ec)
                      x3d(:,b,e,:,m,k) = x3d(:,b,e,:,m,k) + H2C_voov(:,j,:,c) * t_amp ! (ec)(mj)
                      x3d(:,b,e,:,j,m) = x3d(:,b,e,:,j,m) + H2C_voov(:,k,:,c) * t_amp ! (ec)(mk)

                      ! I2C(amij) <- A(ij) [A(n/ij)A(a/ef) h2c(mnef) * t3d(aefijn)]
                      a = t3d_excits(idet,1); e = t3d_excits(idet,2); f = t3d_excits(idet,3);
                      i = t3d_excits(idet,4); j = t3d_excits(idet,5); n = t3d_excits(idet,6);
                      I2C_vooo(a,:,i,j) = I2C_vooo(a,:,i,j) + H2C_oovv(:,n,e,f) * t_amp ! (1)
                      I2C_vooo(a,:,n,j) = I2C_vooo(a,:,n,j) - H2C_oovv(:,i,e,f) * t_amp ! (in)
                      I2C_vooo(a,:,i,n) = I2C_vooo(a,:,i,n) - H2C_oovv(:,j,e,f) * t_amp ! (jn)
                      I2C_vooo(e,:,i,j) = I2C_vooo(e,:,i,j) - H2C_oovv(:,n,a,f) * t_amp ! (ae)
                      I2C_vooo(e,:,n,j) = I2C_vooo(e,:,n,j) + H2C_oovv(:,i,a,f) * t_amp ! (in)(ae)
                      I2C_vooo(e,:,i,n) = I2C_vooo(e,:,i,n) + H2C_oovv(:,j,a,f) * t_amp ! (jn)(ae)
                      I2C_vooo(f,:,i,j) = I2C_vooo(f,:,i,j) - H2C_oovv(:,n,e,a) * t_amp ! (af)
                      I2C_vooo(f,:,n,j) = I2C_vooo(f,:,n,j) + H2C_oovv(:,i,e,a) * t_amp ! (in)(af)
                      I2C_vooo(f,:,i,n) = I2C_vooo(f,:,i,n) + H2C_oovv(:,j,e,a) * t_amp ! (jn)(af)

                      ! I2C(abie) <- A(ab) [A(i/mn)A(f/ab) -h2c(mnef) * t3d(abfimn)]
                      a = t3d_excits(idet,1); b = t3d_excits(idet,2); f = t3d_excits(idet,3);
                      i = t3d_excits(idet,4); m = t3d_excits(idet,5); n = t3d_excits(idet,6);
                      I2C_vvov(a,b,i,:) = I2C_vvov(a,b,i,:) - H2C_oovv(m,n,:,f) * t_amp ! (1)
                      I2C_vvov(a,b,m,:) = I2C_vvov(a,b,m,:) + H2C_oovv(i,n,:,f) * t_amp ! (im)
                      I2C_vvov(a,b,n,:) = I2C_vvov(a,b,n,:) + H2C_oovv(m,i,:,f) * t_amp ! (in)
                      I2C_vvov(f,b,i,:) = I2C_vvov(f,b,i,:) + H2C_oovv(m,n,:,a) * t_amp ! (af)
                      I2C_vvov(f,b,m,:) = I2C_vvov(f,b,m,:) - H2C_oovv(i,n,:,a) * t_amp ! (im)(af)
                      I2C_vvov(f,b,n,:) = I2C_vvov(f,b,n,:) - H2C_oovv(m,i,:,a) * t_amp ! (in)(af)
                      I2C_vvov(a,f,i,:) = I2C_vvov(a,f,i,:) + H2C_oovv(m,n,:,b) * t_amp ! (bf)
                      I2C_vvov(a,f,m,:) = I2C_vvov(a,f,m,:) - H2C_oovv(i,n,:,b) * t_amp ! (im)(bf)
                      I2C_vvov(a,f,n,:) = I2C_vvov(a,f,n,:) - H2C_oovv(m,i,:,b) * t_amp ! (in)(bf)
                  end do

                  ! Update loop
                  do idet = 1, n3bbb
                      a = t3d_excits(idet, 1); b = t3d_excits(idet, 2); c = t3d_excits(idet, 3);
                      i = t3d_excits(idet, 4); j = t3d_excits(idet, 5); k = t3d_excits(idet, 6);

                      res_mm23 = 0.0d0
                      do e = 1, nub
                           ! A(i/jk)(c/ab) h2c(abie) * t2c(ecjk)
                          res_mm23 = res_mm23 + (I2C_vvov(a,b,i,e) - I2C_vvov(b,a,i,e)) * t2c(e,c,j,k)
                          res_mm23 = res_mm23 - (I2C_vvov(c,b,i,e) - I2C_vvov(b,c,i,e)) * t2c(e,a,j,k)
                          res_mm23 = res_mm23 - (I2C_vvov(a,c,i,e) - I2C_vvov(c,a,i,e)) * t2c(e,b,j,k)
                          res_mm23 = res_mm23 - (I2C_vvov(a,b,j,e) - I2C_vvov(b,a,j,e)) * t2c(e,c,i,k)
                          res_mm23 = res_mm23 + (I2C_vvov(c,b,j,e) - I2C_vvov(b,c,j,e)) * t2c(e,a,i,k)
                          res_mm23 = res_mm23 + (I2C_vvov(a,c,j,e) - I2C_vvov(c,a,j,e)) * t2c(e,b,i,k)
                          res_mm23 = res_mm23 - (I2C_vvov(a,b,k,e) - I2C_vvov(b,a,k,e)) * t2c(e,c,j,i)
                          res_mm23 = res_mm23 + (I2C_vvov(c,b,k,e) - I2C_vvov(b,c,k,e)) * t2c(e,a,j,i)
                          res_mm23 = res_mm23 + (I2C_vvov(a,c,k,e) - I2C_vvov(c,a,k,e)) * t2c(e,b,j,i)
                      end do
                      do m = 1, nob
                          ! -A(k/ij)A(a/bc) h2c(amij) * t2c(bcmk)
                          res_mm23 = res_mm23 - (I2C_vooo(a,m,i,j) - I2C_vooo(a,m,j,i)) * t2c(b,c,m,k)
                          res_mm23 = res_mm23 + (I2C_vooo(b,m,i,j) - I2C_vooo(b,m,j,i)) * t2c(a,c,m,k)
                          res_mm23 = res_mm23 + (I2C_vooo(c,m,i,j) - I2C_vooo(c,m,j,i)) * t2c(b,a,m,k)
                          res_mm23 = res_mm23 + (I2C_vooo(a,m,k,j) - I2C_vooo(a,m,j,k)) * t2c(b,c,m,i)
                          res_mm23 = res_mm23 - (I2C_vooo(b,m,k,j) - I2C_vooo(b,m,j,k)) * t2c(a,c,m,i)
                          res_mm23 = res_mm23 - (I2C_vooo(c,m,k,j) - I2C_vooo(c,m,j,k)) * t2c(b,a,m,i)
                          res_mm23 = res_mm23 + (I2C_vooo(a,m,i,k) - I2C_vooo(a,m,k,i)) * t2c(b,c,m,j)
                          res_mm23 = res_mm23 - (I2C_vooo(b,m,i,k) - I2C_vooo(b,m,k,i)) * t2c(a,c,m,j)
                          res_mm23 = res_mm23 - (I2C_vooo(c,m,i,k) - I2C_vooo(c,m,k,i)) * t2c(b,a,m,j)
                      end do

                      denom = fB_oo(i,i) + fB_oo(j,j) + fB_oo(k,k) - fB_vv(a,a) - fB_vv(b,b) - fB_vv(c,c)

                      ! fully antisymmetrize x3a(abcijk)
                      val = &
                       x3d(a,b,c,i,j,k) - x3d(a,c,b,i,j,k) + x3d(b,c,a,i,j,k) - x3d(b,a,c,i,j,k) + x3d(c,a,b,i,j,k) - x3d(c,b,a,i,j,k)&
                      -x3d(a,b,c,i,k,j) + x3d(a,c,b,i,k,j) - x3d(b,c,a,i,k,j) + x3d(b,a,c,i,k,j) - x3d(c,a,b,i,k,j) + x3d(c,b,a,i,k,j)&
                      +x3d(a,b,c,j,k,i) - x3d(a,c,b,j,k,i) + x3d(b,c,a,j,k,i) - x3d(b,a,c,j,k,i) + x3d(c,a,b,j,k,i) - x3d(c,b,a,j,k,i)&
                      -x3d(a,b,c,j,i,k) + x3d(a,c,b,j,i,k) - x3d(b,c,a,j,i,k) + x3d(b,a,c,j,i,k) - x3d(c,a,b,j,i,k) + x3d(c,b,a,j,i,k)&
                      +x3d(a,b,c,k,i,j) - x3d(a,c,b,k,i,j) + x3d(b,c,a,k,i,j) - x3d(b,a,c,k,i,j) + x3d(c,a,b,k,i,j) - x3d(c,b,a,k,i,j)&
                      -x3d(a,b,c,k,j,i) + x3d(a,c,b,k,j,i) - x3d(b,c,a,k,j,i) + x3d(b,a,c,k,j,i) - x3d(c,a,b,k,j,i) + x3d(c,b,a,k,j,i)
                      val = (val + res_mm23)/(denom - shift)

                      t3d_amps(idet) = t3d_amps(idet) + val

                      resid(idet) = val
                  end do

              end subroutine update_t3d_p

end module ccp_linear_loops
