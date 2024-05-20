module ccp_quadratic_loops_direct_h

      use omp_lib

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
                      integer, intent(in) :: t3a_excits(6, n3aaa), t3b_excits(6, n3aab), t3c_excits(6, n3abb)
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
                          a = t3a_excits(1,idet); e = t3a_excits(2,idet); f = t3a_excits(3,idet);
                          i = t3a_excits(4,idet); m = t3a_excits(5,idet); n = t3a_excits(6,idet);
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
                          a = t3b_excits(1,idet); e = t3b_excits(2,idet); f = t3b_excits(3,idet);
                          i = t3b_excits(4,idet); m = t3b_excits(5,idet); n = t3b_excits(6,idet);
                          resid(a,i) = resid(a,i) + H2B_oovv(m,n,e,f) * t_amp ! (1)
                          resid(e,i) = resid(e,i) - H2B_oovv(m,n,a,f) * t_amp ! (ae)
                          resid(a,m) = resid(a,m) - H2B_oovv(i,n,e,f) * t_amp ! (im)
                          resid(e,m) = resid(e,m) + H2B_oovv(i,n,a,f) * t_amp ! (ae)(im)
                      end do

                      do idet = 1, n3abb
                          t_amp = t3c_amps(idet)
                          ! h2c(mnef) * t3c(aefimn)
                          a = t3c_excits(1,idet); e = t3c_excits(2,idet); f = t3c_excits(3,idet);
                          i = t3c_excits(4,idet); m = t3c_excits(5,idet); n = t3c_excits(6,idet);
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
                      integer, intent(in) :: t3b_excits(6, n3aab), t3c_excits(6, n3abb), t3d_excits(6, n3bbb)
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
                          e = t3b_excits(1,idet); f = t3b_excits(2,idet); a = t3b_excits(3,idet);
                          m = t3b_excits(4,idet); n = t3b_excits(5,idet); i = t3b_excits(6,idet);
                          resid(a,i) = resid(a,i) + H2A_oovv(m,n,e,f) * t_amp ! (1)
                      end do

                      do idet = 1, n3abb
                          t_amp = t3c_amps(idet)
                          ! A(af)A(in) h2b(mnef) * t3c(efamni)
                          e = t3c_excits(1,idet); f = t3c_excits(2,idet); a = t3c_excits(3,idet);
                          m = t3c_excits(4,idet); n = t3c_excits(5,idet); i = t3c_excits(6,idet);
                          resid(a,i) = resid(a,i) + H2B_oovv(m,n,e,f) * t_amp ! (1)
                          resid(f,i) = resid(f,i) - H2B_oovv(m,n,e,a) * t_amp ! (af)
                          resid(a,n) = resid(a,n) - H2B_oovv(m,i,e,f) * t_amp ! (in)
                          resid(f,n) = resid(f,n) + H2B_oovv(m,i,e,a) * t_amp ! (af)(in)
                      end do

                      do idet = 1, n3bbb
                          t_amp = t3d_amps(idet)
                          ! A(a/ef)A(i/mn) h2c(mnef) * t3d(aefimn)
                          a = t3d_excits(1,idet); e = t3d_excits(2,idet); f = t3d_excits(3,idet);
                          i = t3d_excits(4,idet); m = t3d_excits(5,idet); n = t3d_excits(6,idet);
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
                  integer, intent(in) :: t3a_excits(6, n3aaa), t3b_excits(6, n3aab)
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
                      a = t3a_excits(1,idet); b = t3a_excits(2,idet); e = t3a_excits(3,idet);
                      i = t3a_excits(4,idet); j = t3a_excits(5,idet); m = t3a_excits(6,idet);
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
                      a = t3a_excits(1,idet); b = t3a_excits(2,idet); f = t3a_excits(3,idet);
                      m = t3a_excits(4,idet); j = t3a_excits(5,idet); n = t3a_excits(6,idet);
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
                      e = t3a_excits(1,idet); b = t3a_excits(2,idet); f = t3a_excits(3,idet);
                      i = t3a_excits(4,idet); j = t3a_excits(5,idet); n = t3a_excits(6,idet);
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
                      a = t3b_excits(1,idet); b = t3b_excits(2,idet); e = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); j = t3b_excits(5,idet); m = t3b_excits(6,idet);
                      resid(a,b,i,j) = resid(a,b,i,j) + H1B_ov(m,e) * t_amp ! (1)

                      ! A(ij)A(ab) [A(jm) -h2b(mnif) * t3b(abfmjn)]
                      a = t3b_excits(1,idet); b = t3b_excits(2,idet); f = t3b_excits(3,idet);
                      m = t3b_excits(4,idet); j = t3b_excits(5,idet); n = t3b_excits(6,idet);
                      resid(a,b,:,j) = resid(a,b,:,j) - H2B_ooov(m,n,:,f) * t_amp ! (1)
                      resid(a,b,:,m) = resid(a,b,:,m) + H2B_ooov(j,n,:,f) * t_amp ! (jm)

                      ! A(ij)A(ab) [A(be) h2b(anef) * t3b(ebfijn)]
                      e = t3b_excits(1,idet); b = t3b_excits(2,idet); f = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); j = t3b_excits(5,idet); n = t3b_excits(6,idet);
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
                  integer, intent(in) :: t3b_excits(6, n3aab), t3c_excits(6, n3abb)
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
                      a = t3b_excits(1,idet); f = t3b_excits(2,idet); b = t3b_excits(3,idet);
                      m = t3b_excits(4,idet); n = t3b_excits(5,idet); j = t3b_excits(6,idet);
                      resid(a,b,:,j) = resid(a,b,:,j) - H2A_ooov(m,n,:,f) * t_amp ! (1)
                      resid(f,b,:,j) = resid(f,b,:,j) + H2A_ooov(m,n,:,a) * t_amp ! (af)

                      ! A(af)A(in) -h2b(nmfj) * t3b(afbinm)
                      a = t3b_excits(1,idet); f = t3b_excits(2,idet); b = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); n = t3b_excits(5,idet); m = t3b_excits(6,idet);
                      resid(a,b,i,:) = resid(a,b,i,:) - H2B_oovo(n,m,f,:) * t_amp ! (1)
                      resid(f,b,i,:) = resid(f,b,i,:) + H2B_oovo(n,m,a,:) * t_amp ! (af)
                      resid(a,b,n,:) = resid(a,b,n,:) + H2B_oovo(i,m,f,:) * t_amp ! (in)
                      resid(f,b,n,:) = resid(f,b,n,:) - H2B_oovo(i,m,a,:) * t_amp ! (af)(in)

                      ! A(in) h2a(anef) * t3b(efbinj)
                      e = t3b_excits(1,idet); f = t3b_excits(2,idet); b = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); n = t3b_excits(5,idet); j = t3b_excits(6,idet);
                      resid(:,b,i,j) = resid(:,b,i,j) + H2A_vovv(:,n,e,f) * t_amp ! (1)
                      resid(:,b,n,j) = resid(:,b,n,j) - H2A_vovv(:,i,e,f) * t_amp ! (in)

                      ! A(af)A(in) h2b(nbfe) * t3b(afeinj)
                      a = t3b_excits(1,idet); f = t3b_excits(2,idet); e = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); n = t3b_excits(5,idet); j = t3b_excits(6,idet);
                      resid(a,:,i,j) = resid(a,:,i,j) + H2B_ovvv(n,:,f,e) * t_amp ! (1)
                      resid(f,:,i,j) = resid(f,:,i,j) - H2B_ovvv(n,:,a,e) * t_amp ! (af)
                      resid(a,:,n,j) = resid(a,:,n,j) - H2B_ovvv(i,:,f,e) * t_amp ! (in)
                      resid(f,:,n,j) = resid(f,:,n,j) + H2B_ovvv(i,:,a,e) * t_amp ! (af)(in)

                      ! A(ae)A(im) h1a(me) * t3b(aebimj)
                      a = t3b_excits(1,idet); e = t3b_excits(2,idet); b = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); m = t3b_excits(5,idet); j = t3b_excits(6,idet);
                      resid(a,b,i,j) = resid(a,b,i,j) + H1A_ov(m,e) * t_amp ! (1)
                      resid(a,b,m,j) = resid(a,b,m,j) - H1A_ov(i,e) * t_amp ! (im)
                      resid(e,b,i,j) = resid(e,b,i,j) - H1A_ov(m,a) * t_amp ! (ae)
                      resid(e,b,m,j) = resid(e,b,m,j) + H1A_ov(i,a) * t_amp ! (im)(ae)

                  end do

                  do idet = 1, n3abb
                      t_amp = t3c_amps(idet)

                      ! A(bf) -h2c(mnjf) * t3c(afbinm)
                      a = t3c_excits(1,idet); f = t3c_excits(2,idet); b = t3c_excits(3,idet);
                      i = t3c_excits(4,idet); n = t3c_excits(5,idet); m = t3c_excits(6,idet);
                      resid(a,b,i,:) = resid(a,b,i,:) - H2C_ooov(m,n,:,f) * t_amp ! (1)
                      resid(a,f,i,:) = resid(a,f,i,:) + H2C_ooov(m,n,:,b) * t_amp ! (bf)

                      ! A(bf)A(jn) -h2b(mnif) * t3c(afbmnj)
                      a = t3c_excits(1,idet); f = t3c_excits(2,idet); b = t3c_excits(3,idet);
                      m = t3c_excits(4,idet); n = t3c_excits(5,idet); j = t3c_excits(6,idet);
                      resid(a,b,:,j) = resid(a,b,:,j) - H2B_ooov(m,n,:,f) * t_amp ! (1)
                      resid(a,f,:,j) = resid(a,f,:,j) + H2B_ooov(m,n,:,b) * t_amp ! (bf)
                      resid(a,b,:,n) = resid(a,b,:,n) + H2B_ooov(m,j,:,f) * t_amp ! (jn)
                      resid(a,f,:,n) = resid(a,f,:,n) - H2B_ooov(m,j,:,b) * t_amp ! (bf)(jn)

                      ! A(jn) h2c(bnef) * t3c(afeinj)
                      a = t3c_excits(1,idet); f = t3c_excits(2,idet); e = t3c_excits(3,idet);
                      i = t3c_excits(4,idet); n = t3c_excits(5,idet); j = t3c_excits(6,idet);
                      resid(a,:,i,j) = resid(a,:,i,j) + H2C_vovv(:,n,e,f) * t_amp ! (1)
                      resid(a,:,i,n) = resid(a,:,i,n) - H2C_vovv(:,j,e,f) * t_amp ! (jn)

                      ! A(bf)A(jn) h2b(anef) * t3c(efbinj)
                      e = t3c_excits(1,idet); f = t3c_excits(2,idet); b = t3c_excits(3,idet);
                      i = t3c_excits(4,idet); n = t3c_excits(5,idet); j = t3c_excits(6,idet);
                      resid(:,b,i,j) = resid(:,b,i,j) + H2B_vovv(:,n,e,f) * t_amp ! (1)
                      resid(:,f,i,j) = resid(:,f,i,j) - H2B_vovv(:,n,e,b) * t_amp ! (bf)
                      resid(:,b,i,n) = resid(:,b,i,n) - H2B_vovv(:,j,e,f) * t_amp ! (jn)
                      resid(:,f,i,n) = resid(:,f,i,n) + H2B_vovv(:,j,e,b) * t_amp ! (bf)(jn)

                      ! [A(be)A(mj) h1b(me) * t3c(aebimj)]
                      a = t3c_excits(1,idet); e = t3c_excits(2,idet); b = t3c_excits(3,idet);
                      i = t3c_excits(4,idet); m = t3c_excits(5,idet); j = t3c_excits(6,idet);
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
                  integer, intent(in) :: t3c_excits(6, n3abb), t3d_excits(6, n3bbb)
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
                      e = t3c_excits(1,idet); a = t3c_excits(2,idet); b = t3c_excits(3,idet);
                      m = t3c_excits(4,idet); i = t3c_excits(5,idet); j = t3c_excits(6,idet);
                      resid(a,b,i,j) = resid(a,b,i,j) + H1A_ov(m,e) * t_amp ! (1)
            
                      ! A(ij)A(ab) [A(be) h2b(nafe) * t3c(febnij)]
                      f = t3c_excits(1,idet); e = t3c_excits(2,idet); b = t3c_excits(3,idet);
                      n = t3c_excits(4,idet); i = t3c_excits(5,idet); j = t3c_excits(6,idet);
                      resid(:,b,i,j) = resid(:,b,i,j) + H2B_ovvv(n,:,f,e) * t_amp ! (1)
                      resid(:,e,i,j) = resid(:,e,i,j) - H2B_ovvv(n,:,f,b) * t_amp ! (be)

                      ! A(ij)A(ab) [A(jm) -h2b(nmfi) * t3c(fabnmj)]
                      f = t3c_excits(1,idet); a = t3c_excits(2,idet); b = t3c_excits(3,idet);
                      n = t3c_excits(4,idet); m = t3c_excits(5,idet); j = t3c_excits(6,idet);
                      resid(a,b,:,j) = resid(a,b,:,j) - H2B_oovo(n,m,f,:) * t_amp ! (1)
                      resid(a,b,:,m) = resid(a,b,:,m) + H2B_oovo(n,j,f,:) * t_amp ! (jm)
                  end do

                  do idet = 1, n3bbb
                      t_amp = t3d_amps(idet)

                      ! A(ij)A(ab) [A(m/ij)A(e/ab) h1b(me) * t3d(abeijm)]
                      a = t3d_excits(1,idet); b = t3d_excits(2,idet); e = t3d_excits(3,idet);
                      i = t3d_excits(4,idet); j = t3d_excits(5,idet); m = t3d_excits(6,idet);
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
                      a = t3d_excits(1,idet); b = t3d_excits(2,idet); f = t3d_excits(3,idet);
                      m = t3d_excits(4,idet); j = t3d_excits(5,idet); n = t3d_excits(6,idet);
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
                      e = t3d_excits(1,idet); b = t3d_excits(2,idet); f = t3d_excits(3,idet);
                      i = t3d_excits(4,idet); j = t3d_excits(5,idet); n = t3d_excits(6,idet);
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
                                      id3a_h, xixjxk_table,&
                                      id3b_h, eck_table, xixj_table,&
                                      H1A_oo, H1A_vv,&
                                      H2A_oovv, H2A_vvov, H2A_vooo,&
                                      H2A_oooo, H2A_voov, H2A_vvvv,&
                                      H2B_oovv, H2B_voov,&
                                      fA_oo, fA_vv,&
                                      shift,&
                                      n3aaa, n3aab,&
                                      noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub, n3aaa, n3aab
                  integer, intent(in) :: t3a_excits(6, n3aaa), t3b_excits(6, n3aab)
                  integer, intent(in) :: id3a_h(noa*(noa-1)*(noa-2)/6,2), id3b_h(nub*nob,noa*(noa-1)/2,2)
                  integer, intent(in) :: xixjxk_table(noa,noa,noa), xixj_table(noa,noa), eck_table(nub,nob)
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
                  real(kind=8) :: val, denom, t_amp, res_mm23, hmatel
                  integer :: a, b, c, d, i, j, k, l, e, f, m, n, idet, jdet
                  integer :: ijk, ij, ik, jk, jb
                  integer :: lmi, lmj, lmk, lij, lik, ljk
                  real(kind=8) :: phase

                  ! Zero the residual container
                  resid = 0.0d0
                  ! Start the VT3 intermediates at Hbar (factor of 1/2 to compensate for antisymmetrization)
                  I2A_vooo(:,:,:,:) = 0.5d0 * H2A_vooo(:,:,:,:)
                  I2A_vvov(:,:,:,:) = 0.5d0 * H2A_vvov(:,:,:,:)

                  do idet = 1, n3aaa
                      t_amp = t3a_amps(idet)

                      ! I2A(amij) <- A(ij) [A(n/ij)A(a/ef) h2a(mnef) * t3a(aefijn)]
                      a = t3a_excits(1,idet); e = t3a_excits(2,idet); f = t3a_excits(3,idet);
                      i = t3a_excits(4,idet); j = t3a_excits(5,idet); n = t3a_excits(6,idet);
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
                      a = t3a_excits(1,idet); b = t3a_excits(2,idet); f = t3a_excits(3,idet);
                      i = t3a_excits(4,idet); m = t3a_excits(5,idet); n = t3a_excits(6,idet);
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

                      ! I2A(amij) <- A(ij) [A(ae) h2b(mnef) * t3b(aefijn)]
                      a = t3b_excits(1,idet); e = t3b_excits(2,idet); f = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); j = t3b_excits(5,idet); n = t3b_excits(6,idet);
                      I2A_vooo(a,:,i,j) = I2A_vooo(a,:,i,j) + H2B_oovv(:,n,e,f) * t_amp ! (1)
                      I2A_vooo(e,:,i,j) = I2A_vooo(e,:,i,j) - H2B_oovv(:,n,a,f) * t_amp ! (ae)

                      ! I2A(abie) <- A(ab) [A(im) -h2b(mnef) * t3b(abfimn)]
                      a = t3b_excits(1,idet); b = t3b_excits(2,idet); f = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); m = t3b_excits(5,idet); n = t3b_excits(6,idet);
                      I2A_vvov(a,b,i,:) = I2A_vvov(a,b,i,:) - H2B_oovv(m,n,:,f) * t_amp ! (1)
                      I2A_vvov(a,b,m,:) = I2A_vvov(a,b,m,:) + H2B_oovv(i,n,:,f) * t_amp ! (im)
                  end do

                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3a_excits,t3b_excits,t3a_amps,t3b_amps,t2a,&
                  !$omp id3a_h,xixjxk_table,&
                  !$omp id3b_h,eck_table,xixj_table,&
                  !$omp H1A_oo,H1A_vv,H2A_oooo,&
                  !$omp H2A_vvvv,H2A_voov,H2B_voov,I2A_vooo,I2A_vvov,&
                  !$omp fA_oo,fA_vv,shift,noa,nua,nob,nub,n3aaa,n3aab),&
                  !$omp private(hmatel,t_amp,denom,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp ij,ik,jk,jb,lmi,lmj,lmk,lij,lik,ljk,phase)

                  !$omp do schedule(static)
                  do idet = 1, n3aaa
                      a = t3a_excits(1,idet); b = t3a_excits(2,idet); c = t3a_excits(3,idet);
                      i = t3a_excits(4,idet); j = t3a_excits(5,idet); k = t3a_excits(6,idet);

                      ijk = xixjxk_table(i,j,k)
                      !do jdet = 1, n3aaa
                      !    d = t3a_excits(1,jdet); e = t3a_excits(2,jdet); f = t3a_excits(3,jdet);
                      !    l = t3a_excits(4,jdet); m = t3a_excits(5,jdet); n = t3a_excits(6,jdet);

                      !    hmatel = 0.0d0
                      !    t_amp = t3a_amps(jdet)
                      !    hmatel = hmatel + aaa_oo_aaa(i,j,k,a,b,c,l,m,n,d,e,f,h1a_oo,noa)
                      !    hmatel = hmatel + aaa_vv_aaa(i,j,k,a,b,c,l,m,n,d,e,f,h1a_vv,nua)
                      !    hmatel = hmatel + aaa_oooo_aaa(i,j,k,a,b,c,l,m,n,d,e,f,h2a_oooo,noa)
                      !    hmatel = hmatel + aaa_vvvv_aaa(i,j,k,a,b,c,l,m,n,d,e,f,h2a_vvvv,nua)
                      !    hmatel = hmatel + aaa_voov_aaa(i,j,k,a,b,c,l,m,n,d,e,f,h2a_voov,noa,nua)
                      !    if (hmatel /= 0.0d0) resid(idet) = resid(idet) + hmatel * t_amp
                      !end do
                      ! diagrams 1/3
                      do l = 1, noa
                         lij = xixjxk_table(l,i,j)
                         lik = xixjxk_table(l,i,k)
                         ljk = xixjxk_table(l,j,k)

                         if (lij/=0) then
                            phase = 1.0d0 * lij/abs(lij)
                            do jdet = id3a_h(abs(lij),1), id3a_h(abs(lij),2)
                               d = t3a_excits(1,jdet); e = t3a_excits(2,jdet); f = t3a_excits(3,jdet);
                               if (a/=d .or. b/=e .or. c/=f) cycle ! skip any p(a) difference
                               ! compute < ijkabc | h1a(oo) | lijabc >
                               hmatel = -phase * h1a_oo(l,k)
                               resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                            end do
                         end if
                         if (lik/=0) then
                            phase = 1.0d0 * lik/abs(lik)
                            do jdet = id3a_h(abs(lik),1), id3a_h(abs(lik),2)
                               d = t3a_excits(1,jdet); e = t3a_excits(2,jdet); f = t3a_excits(3,jdet);
                               if (a/=d .or. b/=e .or. c/=f) cycle ! skip any p(a) difference
                               ! compute < ijkabc | h1a(oo) | likabc >
                               hmatel = phase * h1a_oo(l,j)
                               resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                            end do
                         end if
                         if (ljk/=0) then
                            phase = 1.0d0 * ljk/abs(ljk)
                            do jdet = id3a_h(abs(ljk),1), id3a_h(abs(ljk),2)
                               d = t3a_excits(1,jdet); e = t3a_excits(2,jdet); f = t3a_excits(3,jdet);
                               if (a/=d .or. b/=e .or. c/=f) cycle ! skip any p(a) difference
                               ! compute < ijkabc | h1a(oo) | ljkabc >
                               hmatel = -phase * h1a_oo(l,i)
                               resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                            end do
                         end if

                         do m = l+1, noa
                            lmi = xixjxk_table(l,m,i)
                            lmj = xixjxk_table(l,m,j)
                            lmk = xixjxk_table(l,m,k)

                            if (lmi/=0) then
                               phase = 1.0d0 * lmi/abs(lmi)
                               do jdet = id3a_h(abs(lmi),1), id3a_h(abs(lmi),2)
                                  d = t3a_excits(1,jdet); e = t3a_excits(2,jdet); f = t3a_excits(3,jdet);
                                  if (a/=d .or. b/=e .or. c/=f) cycle ! skip any p(a) difference
                                  ! compute < ijkabc | h2a(oooo) | lmiabc >
                                  hmatel = phase * h2a_oooo(l,m,j,k)
                                  resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                               end do
                            end if
                            if (lmj/=0) then
                               phase = 1.0d0 * lmj/abs(lmj)
                               do jdet = id3a_h(abs(lmj),1), id3a_h(abs(lmj),2)
                                  d = t3a_excits(1,jdet); e = t3a_excits(2,jdet); f = t3a_excits(3,jdet);
                                  if (a/=d .or. b/=e .or. c/=f) cycle ! skip any p(a) difference
                                  ! compute < ijkabc | h2a(oooo) | lmjabc >
                                  hmatel = -phase * h2a_oooo(l,m,i,k)
                                  resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                               end do
                            end if
                            if (lmk/=0) then
                               phase = 1.0d0 * lmk/abs(lmk)
                               do jdet = id3a_h(abs(lmk),1), id3a_h(abs(lmk),2)
                                  d = t3a_excits(1,jdet); e = t3a_excits(2,jdet); f = t3a_excits(3,jdet);
                                  if (a/=d .or. b/=e .or. c/=f) cycle ! skip any p(a) difference
                                  ! compute < ijkabc | h2a(oooo) | lmkabc >
                                  hmatel = phase * h2a_oooo(l,m,i,j)
                                  resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                               end do
                            end if
                         end do
                      end do

                      ! diagram 2/4
                      do jdet = id3a_h(ijk,1), id3a_h(ijk,2)
                         d = t3a_excits(1,jdet); e = t3a_excits(2,jdet); f = t3a_excits(3,jdet);
                         if (nexc3(a,b,c,d,e,f)>2) cycle ! skip if p(a) difference is more than 2
                         hmatel = 0.0d0
                         if (d==a) then      ! case 1: d = a
                            ! compute < ijkabc | h2a(vvvv) | ijkaef >
                            hmatel = hmatel + h2a_vvvv(b,c,e,f)
                         elseif (d==b) then  ! case 2: d = b
                            ! compute < ijkabc | h2a(vvvv) | ijkbef >
                            hmatel = hmatel - h2a_vvvv(a,c,e,f)
                         elseif (d==c) then  ! case 3: d = c
                            ! compute < ijkabc | h2a(vvvv) | ijkcef >
                            hmatel = hmatel + h2a_vvvv(a,b,e,f)
                         end if
                         if (e==a) then      ! case 4: e = a
                            ! compute < ijkabc | h2a(vvvv) | ijkdaf >
                            hmatel = hmatel - h2a_vvvv(b,c,d,f)
                         elseif (e==b) then  ! case 5: e = b
                            ! compute < ijkabc | h2a(vvvv) | ijkdbf >
                            hmatel = hmatel + h2a_vvvv(a,c,d,f)
                         elseif (e==c) then  ! case 6: e = c
                            ! compute < ijkabc | h2a(vvvv) | ijkdcf >
                            hmatel = hmatel - h2a_vvvv(a,b,d,f)
                         end if
                         if (f==a) then      ! case 1: f = a
                            ! compute < ijkabc | h2a(vvvv) | ijkdea >
                            hmatel = hmatel + h2a_vvvv(b,c,d,e)
                         elseif (f==b) then  ! case 2: f = b
                            ! compute < ijkabc | h2a(vvvv) | ijkdeb >
                            hmatel = hmatel - h2a_vvvv(a,c,d,e)
                         elseif (f==c) then  ! case 3: f = c
                            ! compute < ijkabc | h2a(vvvv) | ijkdec >
                            hmatel = hmatel + h2a_vvvv(a,b,d,e)
                         end if

                         if (nexc3(a,b,c,d,e,f)<2) then ! include h1a(vv) terms in this case
                            if (d==a .and. e==b) then     ! case 1: (d,e) -> (a,b)
                               ! compute < ijkabc | h1a(vv) | ijkabf >
                               hmatel = hmatel + h1a_vv(c,f)
                            elseif (d==a .and. e==c) then ! case 2: (d,e) -> (a,c)
                               ! compute < ijkabc | h1a(vv) | ijkacf >
                               hmatel = hmatel - h1a_vv(b,f)
                            elseif (d==b .and. e==c) then ! case 3: (d,e) -> (b,c)
                               ! compute < ijkabc | h1a(vv) | ijkbcf >
                               hmatel = hmatel + h1a_vv(a,f)
                            end if
                            if (d==a .and. f==b) then     ! case 4: (d,f) -> (a,b)
                               ! compute < ijkabc | h1a(vv) | ijkaeb >
                               hmatel = hmatel - h1a_vv(c,e)
                            elseif (d==a .and. f==c) then ! case 5: (d,f) -> (a,c)
                               ! compute < ijkabc | h1a(vv) | ijkaec >
                               hmatel = hmatel + h1a_vv(b,e)
                            elseif (d==b .and. f==c) then ! case 6: (d,f) -> (b,c)
                               ! compute < ijkabc | h1a(vv) | ijkbec >
                               hmatel = hmatel - h1a_vv(a,e)
                            end if
                            if (e==a .and. f==b) then     ! case 7: (e,f) -> (a,b)
                               ! compute < ijkabc | h1a(vv) | ijkdab >
                               hmatel = hmatel + h1a_vv(c,d)
                            elseif (e==a .and. f==c) then ! case 8: (e,f) -> (a,c)
                               ! compute < ijkabc | h1a(vv) | ijkdac >
                               hmatel = hmatel - h1a_vv(b,d)
                            elseif (e==b .and. f==c) then ! case 9: (e,f) -> (b,c)
                               ! compute < ijkabc | h1a(vv) | ijkdbc >
                               hmatel = hmatel + h1a_vv(a,d)
                            end if
                         end if
                         resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                      end do

                      ! diagram 5
                      do l = 1, noa
                         lij = xixjxk_table(l,i,j)
                         lik = xixjxk_table(l,i,k)
                         ljk = xixjxk_table(l,j,k)
                         if (lij/=0) then
                            phase = 1.0d0 * lij/abs(lij)
                            do jdet = id3a_h(abs(lij),1), id3a_h(abs(lij),2)
                               d = t3a_excits(1,jdet); e = t3a_excits(2,jdet); f = t3a_excits(3,jdet);
                               if (nexc3(a,b,c,d,e,f)>1) cycle ! skip if p(a) difference is more than 1
                               hmatel = 0.0d0
                               if (d==a .and. e==b) then     ! case 1: (d,e) -> (a,b)
                                  ! compute < ijkabc | h2a(voov) | lijabf >
                                  hmatel = hmatel + phase * h2a_voov(c,l,k,f)
                               elseif (d==a .and. e==c) then ! case 2: (d,e) -> (a,c)
                                  ! compute < ijkabc | h2a(voov) | lijacf >
                                  hmatel = hmatel - phase * h2a_voov(b,l,k,f)
                               elseif (d==b .and. e==c) then ! case 3: (d,e) -> (b,c)
                                  ! compute < ijkabc | h2a(voov) | lijbcf >
                                  hmatel = hmatel + phase * h2a_voov(a,l,k,f)
                               end if
                               if (d==a .and. f==b) then     ! case 4: (d,f) -> (a,b)
                                  ! compute < ijkabc | h2a(voov) | lijaeb >
                                  hmatel = hmatel - phase * h2a_voov(c,l,k,e)
                               elseif (d==a .and. f==c) then ! case 5: (d,f) -> (a,c)
                                  ! compute < ijkabc | h2a(voov) | lijaec >
                                  hmatel = hmatel + phase * h2a_voov(b,l,k,e)
                               elseif (d==b .and. f==c) then ! case 6: (d,f) -> (b,c)
                                  ! compute < ijkabc | h2a(voov) | lijbec >
                                  hmatel = hmatel - phase * h2a_voov(a,l,k,e)
                               end if
                               if (e==a .and. f==b) then     ! case 7: (e,f) -> (a,b)
                                  ! compute < ijkabc | h2a(voov) | lijdab >
                                  hmatel = hmatel + phase * h2a_voov(c,l,k,d)
                               elseif (e==a .and. f==c) then ! case 8: (e,f) -> (a,c)
                                  ! compute < ijkabc | h2a(voov) | lijdac >
                                  hmatel = hmatel - phase * h2a_voov(b,l,k,d)
                               elseif (e==b .and. f==c) then ! case 9: (e,f) -> (b,c)
                                  ! compute < ijkabc | h2a(voov) | lijdbc >
                                  hmatel = hmatel + phase * h2a_voov(a,l,k,d)
                               end if
                               resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                            end do
                         end if
                         if (lik/=0) then
                            ! WARNING: Phase reversal here, presumably to account for (l,i,k) = -(i,l,k)
                            phase = -1.0d0 * lik/abs(lik) 
                            do jdet = id3a_h(abs(lik),1), id3a_h(abs(lik),2)
                               d = t3a_excits(1,jdet); e = t3a_excits(2,jdet); f = t3a_excits(3,jdet);
                               if (nexc3(a,b,c,d,e,f)>1) cycle ! skip if p(a) difference is more than 1
                               hmatel = 0.0d0
                               if (d==a .and. e==b) then     ! case 1: (d,e) -> (a,b)
                                  ! compute < ijkabc | h2a(voov) | likabf >
                                  hmatel = hmatel + phase * h2a_voov(c,l,j,f)
                               elseif (d==a .and. e==c) then ! case 2: (d,e) -> (a,c)
                                  ! compute < ijkabc | h2a(voov) | likacf >
                                  hmatel = hmatel - phase * h2a_voov(b,l,j,f)
                               elseif (d==b .and. e==c) then ! case 3: (d,e) -> (b,c)
                                  ! compute < ijkabc | h2a(voov) | likbcf >
                                  hmatel = hmatel + phase * h2a_voov(a,l,j,f)
                               end if
                               if (d==a .and. f==b) then     ! case 4: (d,f) -> (a,b)
                                  ! compute < ijkabc | h2a(voov) | likaeb >
                                  hmatel = hmatel - phase * h2a_voov(c,l,j,e)
                               elseif (d==a .and. f==c) then ! case 5: (d,f) -> (a,c)
                                  ! compute < ijkabc | h2a(voov) | likaec >
                                  hmatel = hmatel + phase * h2a_voov(b,l,j,e)
                               elseif (d==b .and. f==c) then ! case 6: (d,f) -> (b,c)
                                  ! compute < ijkabc | h2a(voov) | likbec >
                                  hmatel = hmatel - phase * h2a_voov(a,l,j,e)
                               end if
                               if (e==a .and. f==b) then     ! case 7: (e,f) -> (a,b)
                                  ! compute < ijkabc | h2a(voov) | likdab >
                                  hmatel = hmatel + phase * h2a_voov(c,l,j,d)
                               elseif (e==a .and. f==c) then ! case 8: (e,f) -> (a,c)
                                  ! compute < ijkabc | h2a(voov) | likdac >
                                  hmatel = hmatel - phase * h2a_voov(b,l,j,d)
                               elseif (e==b .and. f==c) then ! case 9: (e,f) -> (b,c)
                                  ! compute < ijkabc | h2a(voov) | likdbc >
                                  hmatel = hmatel + phase * h2a_voov(a,l,j,d)
                               end if
                               resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                            end do
                         end if
                         if (ljk/=0) then
                            phase = 1.0d0 * ljk/abs(ljk)
                            do jdet = id3a_h(abs(ljk),1), id3a_h(abs(ljk),2)
                               d = t3a_excits(1,jdet); e = t3a_excits(2,jdet); f = t3a_excits(3,jdet);
                               if (nexc3(a,b,c,d,e,f)>1) cycle ! skip if p(a) difference is more than 1
                               hmatel = 0.0d0
                               if (d==a .and. e==b) then     ! case 1: (d,e) -> (a,b)
                                  ! compute < ijkabc | h2a(voov) | ljkabf >
                                  hmatel = hmatel + phase * h2a_voov(c,l,i,f)
                               elseif (d==a .and. e==c) then ! case 2: (d,e) -> (a,c)
                                  ! compute < ijkabc | h2a(voov) | ljkacf >
                                  hmatel = hmatel - phase * h2a_voov(b,l,i,f)
                               elseif (d==b .and. e==c) then ! case 3: (d,e) -> (b,c)
                                  ! compute < ijkabc | h2a(voov) | ljkbcf >
                                  hmatel = hmatel + phase * h2a_voov(a,l,i,f)
                               end if
                               if (d==a .and. f==b) then     ! case 4: (d,f) -> (a,b)
                                  ! compute < ijkabc | h2a(voov) | ljkaeb >
                                  hmatel = hmatel - phase * h2a_voov(c,l,i,e)
                               elseif (d==a .and. f==c) then ! case 5: (d,f) -> (a,c)
                                  ! compute < ijkabc | h2a(voov) | ljkaec >
                                  hmatel = hmatel + phase * h2a_voov(b,l,i,e)
                               elseif (d==b .and. f==c) then ! case 6: (d,f) -> (b,c)
                                  ! compute < ijkabc | h2a(voov) | ljkbec >
                                  hmatel = hmatel - phase * h2a_voov(a,l,i,e)
                               end if
                               if (e==a .and. f==b) then     ! case 7: (e,f) -> (a,b)
                                  ! compute < ijkabc | h2a(voov) | ljkdab >
                                  hmatel = hmatel + phase * h2a_voov(c,l,i,d)
                               elseif (e==a .and. f==c) then ! case 8: (e,f) -> (a,c)
                                  ! compute < ijkabc | h2a(voov) | ljkdac >
                                  hmatel = hmatel - phase * h2a_voov(b,l,i,d)
                               elseif (e==b .and. f==c) then ! case 9: (e,f) -> (b,c)
                                  ! compute < ijkabc | h2a(voov) | ljkdbc >
                                  hmatel = hmatel + phase * h2a_voov(a,l,i,d)
                               end if
                               resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                            end do
                         end if
                      end do
                      !!!! diagram 6: A(i/jk)A(a/bc) h2b(amie) * t3b(abeijm)
                      !do jdet = 1, n3aab
                      !    d = t3b_excits(1,jdet); e = t3b_excits(2,jdet); f = t3b_excits(3,jdet);
                      !    l = t3b_excits(4,jdet); m = t3b_excits(5,jdet); n = t3b_excits(6,jdet);

                      !    hmatel = 0.0d0
                      !    t_amp = t3b_amps(jdet)
                      !    hmatel = hmatel + aaa_voov_aab(i,j,k,a,b,c,l,m,n,d,e,f,h2b_voov,noa,nua,nob,nub)
                      !    if (hmatel /= 0.0d0) resid(idet) = resid(idet) + hmatel * t_amp
                      !end do
                      do n = 1, nob
                         do f = 1, nub
                            jb = eck_table(f,n)
                            ij = xixj_table(i,j); ik = xixj_table(i,k); jk = xixj_table(j,k);
                            do jdet = id3b_h(jb,ij,1), id3b_h(jb,ij,2)
                               d = t3b_excits(1,jdet); e = t3b_excits(2,jdet);
                               ! (d,e) must be an ordered subset of (a,b,c)
                               hmatel = 0.0d0
                               if (d==a .and. e==b) then
                                  ! compute < ijkabc | h2b(voov) | ijn~abf~ >
                                  hmatel = hmatel + h2b_voov(c,n,k,f)
                               else if (d==a .and. e==c) then
                                  ! compute < ijkabc | h2b(voov) | ijn~acf~ >
                                  hmatel = hmatel - h2b_voov(b,n,k,f)
                               else if (d==b .and. e==c) then
                                  ! compute < ijkabc | h2b(voov) | ijn~bcf~ >
                                  hmatel = hmatel + h2b_voov(a,n,k,f)
                               end if
                               resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                            end do
                            do jdet = id3b_h(jb,ik,1), id3b_h(jb,ik,2)
                               d = t3b_excits(1,jdet); e = t3b_excits(2,jdet);
                               ! (d,e) must be an ordered subset of (a,b,c)
                               hmatel = 0.0d0
                               if (d==a .and. e==b) then
                                  ! compute < ijkabc | h2b(voov) | ikn~abf~ >
                                  hmatel = hmatel - h2b_voov(c,n,j,f)
                               else if (d==a .and. e==c) then
                                  ! compute < ijkabc | h2b(voov) | ikn~acf~ >
                                  hmatel = hmatel + h2b_voov(b,n,j,f)
                               else if (d==b .and. e==c) then
                                  ! compute < ijkabc | h2b(voov) | ikn~bcf~ >
                                  hmatel = hmatel - h2b_voov(a,n,j,f)
                               end if
                               resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                            end do
                            do jdet = id3b_h(jb,jk,1), id3b_h(jb,jk,2)
                               d = t3b_excits(1,jdet); e = t3b_excits(2,jdet);
                               ! (d,e) must be an ordered subset of (a,b,c)
                               hmatel = 0.0d0
                               if (d==a .and. e==b) then
                                  ! compute < ijkabc | h2b(voov) | jkn~abf~ >
                                  hmatel = hmatel + h2b_voov(c,n,i,f)
                               else if (d==a .and. e==c) then
                                  ! compute < ijkabc | h2b(voov) | jkn~acf~ >
                                  hmatel = hmatel - h2b_voov(b,n,i,f)
                               else if (d==b .and. e==c) then
                                  ! compute < ijkabc | h2b(voov) | jkn~bcf~ >
                                  hmatel = hmatel + h2b_voov(a,n,i,f)
                               end if
                               resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                            end do
                         end do
                      end do

                  end do ! end loop over idet
                  !$omp end do

                  !$omp do
                  do idet = 1, n3aaa
                      a = t3a_excits(1,idet); b = t3a_excits(2,idet); c = t3a_excits(3,idet);
                      i = t3a_excits(4,idet); j = t3a_excits(5,idet); k = t3a_excits(6,idet);

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

                      resid(idet) = (resid(idet) + res_mm23)/(denom - shift)
                      t3a_amps(idet) = t3a_amps(idet) + resid(idet)

                  end do
                  !$omp end do

                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!

              end subroutine update_t3a_p

              subroutine update_t3b_p(t3b_amps, resid,&
                                      t3a_excits, t3b_excits, t3c_excits,&
                                      t2a, t2b,&
                                      t3a_amps, t3c_amps,&
                                      id3a_h, xixjxk_table,&
                                      id3b_h, eck_table, xixj_table,&
                                      id3c_h, eai_table, xjxk_table,&
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
                  integer, intent(in) :: t3a_excits(6, n3aaa), t3b_excits(6, n3aab), t3c_excits(6, n3abb)
                  integer, intent(in) :: id3a_h(noa*(noa-1)*(noa-2)/6,2), id3b_h(nub*nob,noa*(noa-1)/2,2), id3c_h(nua*noa,nob*(nob-1)/2,2)
                  integer, intent(in) :: eck_table(nub,nob), eai_table(nua,noa)
                  integer, intent(in) :: xixjxk_table(noa,noa,noa), xixj_table(noa,noa), xjxk_table(nob,nob)
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
                  real(kind=8) :: denom, val, t_amp, res_mm23, hmatel
                  integer :: i, j, k, l, a, b, c, d, m, n, e, f, idet, jdet
                  integer :: ib, ij, jb, lm
                  integer :: ai, bj, aj, bi, mn, m1, mk
                  integer :: li, lj, lmn
                  real(kind=8) :: phase

                  integer :: d1, e1, f1, l1, n1

                  ! Zero the residual container
                  resid = 0.0d0
                  ! compute VT3 intermediates
                  I2A_vooo(:,:,:,:) = 0.5d0 * H2A_vooo(:,:,:,:)
                  I2A_vvov(:,:,:,:) = 0.5d0 * H2A_vvov(:,:,:,:)
                  I2B_vooo(:,:,:,:) = H2B_vooo(:,:,:,:)
                  I2B_ovoo(:,:,:,:) = H2B_ovoo(:,:,:,:)
                  I2B_vvov(:,:,:,:) = H2B_vvov(:,:,:,:)
                  I2B_vvvo(:,:,:,:) = H2B_vvvo(:,:,:,:)

                  do idet = 1, n3aaa
                      t_amp = t3a_amps(idet)

                      ! I2A(amij) <- A(ij) [A(n/ij)A(a/ef) h2a(mnef) * t3a(aefijn)]
                      a = t3a_excits(1,idet); e = t3a_excits(2,idet); f = t3a_excits(3,idet);
                      i = t3a_excits(4,idet); j = t3a_excits(5,idet); n = t3a_excits(6,idet);
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
                      a = t3a_excits(1,idet); b = t3a_excits(2,idet); f = t3a_excits(3,idet);
                      i = t3a_excits(4,idet); m = t3a_excits(5,idet); n = t3a_excits(6,idet);
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

                      ! I2A(amij) <- A(ij) [A(ae) h2b(mnef) * t3b(aefijn)]
                      a = t3b_excits(1,idet); e = t3b_excits(2,idet); f = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); j = t3b_excits(5,idet); n = t3b_excits(6,idet);
                      I2A_vooo(a,:,i,j) = I2A_vooo(a,:,i,j) + H2B_oovv(:,n,e,f) * t_amp ! (1)
                      I2A_vooo(e,:,i,j) = I2A_vooo(e,:,i,j) - H2B_oovv(:,n,a,f) * t_amp ! (ae)

                      ! I2A(abie) <- A(ab) [A(im) -h2b(mnef) * t3b(abfimn)]
                      a = t3b_excits(1,idet); b = t3b_excits(2,idet); f = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); m = t3b_excits(5,idet); n = t3b_excits(6,idet);
                      I2A_vvov(a,b,i,:) = I2A_vvov(a,b,i,:) - H2B_oovv(m,n,:,f) * t_amp ! (1)
                      I2A_vvov(a,b,m,:) = I2A_vvov(a,b,m,:) + H2B_oovv(i,n,:,f) * t_amp ! (im)

                      ! I2B(abej) <- A(af) -h2a(mnef) * t3b(afbmnj)
                      a = t3b_excits(1,idet); f = t3b_excits(2,idet); b = t3b_excits(3,idet);
                      m = t3b_excits(4,idet); n = t3b_excits(5,idet); j = t3b_excits(6,idet);
                      I2B_vvvo(a,b,:,j) = I2B_vvvo(a,b,:,j) - H2A_oovv(m,n,:,f) * t_amp ! (1)
                      I2B_vvvo(f,b,:,j) = I2B_vvvo(f,b,:,j) + H2A_oovv(m,n,:,a) * t_amp ! (af)

                      ! I2B(mbij) <- A(in) h2a(mnef) * t3b(efbinj)
                      e = t3b_excits(1,idet); f = t3b_excits(2,idet); b = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); n = t3b_excits(5,idet); j = t3b_excits(6,idet);
                      I2B_ovoo(:,b,i,j) = I2B_ovoo(:,b,i,j) + H2A_oovv(:,n,e,f) * t_amp ! (1)
                      I2B_ovoo(:,b,n,j) = I2B_ovoo(:,b,n,j) - H2A_oovv(:,i,e,f) * t_amp ! (in)

                      ! I2B(abie) <- A(af)A(in) -h2b(nmfe) * t3b(afbinm)
                      a = t3b_excits(1,idet); f = t3b_excits(2,idet); b = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); n = t3b_excits(5,idet); m = t3b_excits(6,idet);
                      I2B_vvov(a,b,i,:) = I2B_vvov(a,b,i,:) - H2B_oovv(n,m,f,:) * t_amp ! (1)
                      I2B_vvov(f,b,i,:) = I2B_vvov(f,b,i,:) + H2B_oovv(n,m,a,:) * t_amp ! (af)
                      I2B_vvov(a,b,n,:) = I2B_vvov(a,b,n,:) + H2B_oovv(i,m,f,:) * t_amp ! (in)
                      I2B_vvov(f,b,n,:) = I2B_vvov(f,b,n,:) - H2B_oovv(i,m,a,:) * t_amp ! (af)(in)

                      ! I2B(amij) <- A(af)A(in) h2b(nmfe) * t3b(afeinj)
                      a = t3b_excits(1,idet); f = t3b_excits(2,idet); e = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); n = t3b_excits(5,idet); j = t3b_excits(6,idet);
                      I2B_vooo(a,:,i,j) = I2B_vooo(a,:,i,j) + H2B_oovv(n,:,f,e) * t_amp ! (1)
                      I2B_vooo(f,:,i,j) = I2B_vooo(f,:,i,j) - H2B_oovv(n,:,a,e) * t_amp ! (af)
                      I2B_vooo(a,:,n,j) = I2B_vooo(a,:,n,j) - H2B_oovv(i,:,f,e) * t_amp ! (in)
                      I2B_vooo(f,:,n,j) = I2B_vooo(f,:,n,j) + H2B_oovv(i,:,a,e) * t_amp ! (af)(in)
                  end do

                  do idet = 1, n3abb
                      t_amp = t3c_amps(idet)

                      ! I2B(abej) <- A(bf)A(jn) -h2b(mnef) * t3c(afbmnj)
                      a = t3c_excits(1,idet); f = t3c_excits(2,idet); b = t3c_excits(3,idet);
                      m = t3c_excits(4,idet); n = t3c_excits(5,idet); j = t3c_excits(6,idet);
                      I2B_vvvo(a,b,:,j) = I2B_vvvo(a,b,:,j) - H2B_oovv(m,n,:,f) * t_amp ! (1)
                      I2B_vvvo(a,f,:,j) = I2B_vvvo(a,f,:,j) + H2B_oovv(m,n,:,b) * t_amp ! (bf)
                      I2B_vvvo(a,b,:,n) = I2B_vvvo(a,b,:,n) + H2B_oovv(m,j,:,f) * t_amp ! (jn)
                      I2B_vvvo(a,f,:,n) = I2B_vvvo(a,f,:,n) - H2B_oovv(m,j,:,b) * t_amp ! (bf)(jn)

                      ! I2B(mbij) <- A(bf)A(jn) h2B(mnef) * t3c(efbinj)
                      e = t3c_excits(1,idet); f = t3c_excits(2,idet); b = t3c_excits(3,idet);
                      i = t3c_excits(4,idet); n = t3c_excits(5,idet); j = t3c_excits(6,idet);
                      I2B_ovoo(:,b,i,j) = I2B_ovoo(:,b,i,j) + H2B_oovv(:,n,e,f) * t_amp ! (1)
                      I2B_ovoo(:,f,i,j) = I2B_ovoo(:,f,i,j) - H2B_oovv(:,n,e,b) * t_amp ! (bf)
                      I2B_ovoo(:,b,i,n) = I2B_ovoo(:,b,i,n) - H2B_oovv(:,j,e,f) * t_amp ! (jn)
                      I2B_ovoo(:,f,i,n) = I2B_ovoo(:,f,i,n) + H2B_oovv(:,j,e,b) * t_amp ! (bf)(jn)

                      ! I2B(abie) <- A(bf) -h2c(nmfe) * t3c(afbinm)
                      a = t3c_excits(1,idet); f = t3c_excits(2,idet); b = t3c_excits(3,idet);
                      i = t3c_excits(4,idet); n = t3c_excits(5,idet); m = t3c_excits(6,idet);
                      I2B_vvov(a,b,i,:) = I2B_vvov(a,b,i,:) - H2C_oovv(n,m,f,:) * t_amp ! (1)
                      I2B_vvov(a,f,i,:) = I2B_vvov(a,f,i,:) + H2C_oovv(n,m,b,:) * t_amp ! (bf)

                      ! I2B(amij) <- A(jn) h2c(nmfe) * t3c(afeinj)
                      a = t3c_excits(1,idet); f = t3c_excits(2,idet); e = t3c_excits(3,idet);
                      i = t3c_excits(4,idet); n = t3c_excits(5,idet); j = t3c_excits(6,idet);
                      I2B_vooo(a,:,i,j) = I2B_vooo(a,:,i,j) + H2C_oovv(n,:,f,e) * t_amp ! (1)
                      I2B_vooo(a,:,i,n) = I2B_vooo(a,:,i,n) - H2C_oovv(j,:,f,e) * t_amp ! (jn)
                  end do

                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3a_excits,t3b_excits,t3c_excits,&
                  !$omp t3a_amps,t3b_amps,t3c_amps,t2a,t2b,&
                  !$omp id3a_h,xixjxk_table,&
                  !$omp id3b_h,eck_table,xixj_table,&
                  !$omp id3c_h,eai_table,xjxk_table,&
                  !$omp H1A_oo,H1A_vv,H1B_oo,H1B_vv,H2A_oooo,H2B_oooo,&
                  !$omp H2B_ovvo,H2A_vvvv,H2B_vvvv,H2A_voov,H2C_voov,&
                  !$omp H2B_vovo,H2B_ovov,H2B_voov,&
                  !$omp I2A_vooo,I2A_vvov,I2B_vooo,I2B_ovoo,I2B_vvov,I2B_vvvo,&
                  !$omp fA_oo,fB_oo,fA_vv,fB_vv,noa,nua,nob,nub,shift,&
                  !$omp n3aaa,n3aab,n3abb),&
                  !$omp private(hmatel,phase,t_amp,denom,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp ib,ij,jb,lm,ai,bj,aj,bi,mn,m1,mk,li,lj,lmn)
                  
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                      a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                      ib = eck_table(c,k)
                      ij = xixj_table(i,j)

                      !!!! diagram 1: -A(i/jk) h1a(mi)*t3b(abcmjk)    
                      !!!! diagram 5: A(i/jk) 1/2 h2a(mnij)*t3b(abcmnk) 
                      ! cpu cost = noa**2 * nua**2/nua**2 = noa**2 
                      ! loop extent = noa**2 * nua**2 
                      ! Optimal plan: use P ordering to compute efficiently
                      do l = 1, noa
                         do m = l+1, noa
                            lm = xixj_table(l,m)
                            do jdet = id3b_h(ib,lm,1), id3b_h(ib,lm,2)
                               d = t3b_excits(1,jdet); e = t3b_excits(2,jdet);
                               if (a/=d .or. b/=e) cycle ! skip if any p(a) difference
                               ! compute h2a(oooo)
                               hmatel = h2a_oooo(l,m,i,j) 
                               if (nexc2(i,j,l,m)<2) then ! compute h1a(oo)
                                       if (j==m) then 
                                               hmatel = hmatel - h1a_oo(l,i)
                                       elseif (j==l) then
                                               hmatel = hmatel + h1a_oo(m,i)
                                       end if
                                       if (i==m) then
                                               hmatel = hmatel + h1a_oo(l,j)
                                       elseif (i==l) then
                                               hmatel = hmatel - h1a_oo(m,j)
                                       end if
                               end if
                               resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                            end do
                         end do
                      end do
                      !!!! diagram 2: A(a/bc) h1a(ae)*t3b(ebcmjk)
                      !!!! diagram 6: A(a/bc) 1/2 h2a(abef)*t3b(ebcmjk)
                      ! cpu cost = nua**2
                      ! loop extent = nua**2
                      do jdet = id3b_h(ib,ij,1), id3b_h(ib,ij,2)
                         d = t3b_excits(1,jdet); e = t3b_excits(2,jdet);
                         ! compute h2a(vvvv)
                         hmatel = h2a_vvvv(a,b,d,e)
                         if (nexc2(a,b,d,e)<2) then ! compute h1a(vv)
                             if (a==e) then 
                                     hmatel = hmatel - h1a_vv(b,d) 
                             elseif (a==d) then
                                     hmatel = hmatel + h1a_vv(b,e) 
                             end if
                             if (b==e) then 
                                     hmatel = hmatel + h1a_vv(a,d)
                             elseif (b==d) then
                                     hmatel = hmatel - h1a_vv(a,e)
                             end if
                         end if
                         resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                      end do
                      !!!! diagram 3: -h1b(mk)*t3b(abcijm)
                      !!!! diagram 7: A(ij) h2b(mnjk)*t3b(abcimn)
                      ! cpu cost = nob * noa**2/noa * nua**2/nua**2 = nob*noa
                      ! loop extent = nob * noa**2/noa * nua**2 = nob*noa*nua**2
                      !do l = 1, noa
                      !   do m = l+1, noa
                      !      if (nexc2(i,j,l,m)>1) cycle ! skip if h(a) > 1
                      !      lm = xixj_table(l,m)
                      !      do n = 1, nob
                      !         jb = eck_table(c,n)
                      !         do jdet = id3b_h(jb,lm,1), id3b_h(jb,lm,2)
                      !            d = t3b_excits(1,jdet); e = t3b_excits(2,jdet);
                      !            if (a/=d .or. b/=e) cycle ! skip any p(a) difference
                      !            hmatel = 0.0d0
                      !            ! compute h2b(oooo)
                      !            if (i==l) then 
                      !                    hmatel = hmatel + h2b_oooo(m,n,j,k) ! (1)
                      !            elseif (i==m) then
                      !                    hmatel = hmatel - h2b_oooo(l,n,j,k) ! (lm)
                      !            end if
                      !            if (j==l) then 
                      !                    hmatel = hmatel - h2b_oooo(m,n,i,k) ! (ij)
                      !            elseif (j==m) then 
                      !                    hmatel = hmatel + h2b_oooo(l,n,i,k) ! (ij)(lm)
                      !            end if
                      !            if (i==l .and. j==m) hmatel = hmatel - h1b_oo(n,k) ! compute h1b(oo)
                      !            resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                      !         end do
                      !      end do
                      !   end do
                      !end do
                      do n = 1, nob
                         jb = eck_table(c,n)
                         do jdet = id3b_h(jb,ij,1), id3b_h(jb,ij,2)
                            d = t3b_excits(1,jdet); e = t3b_excits(2,jdet);
                            if (a/=d .or. b/=e) cycle ! skip any p(a) difference
                            ! compute < ijkabc | h1b(oo) | ijnabc >
                            resid(idet) = resid(idet) - h1b_oo(n,k) * t3b_amps(jdet)
                         end do
                         do l = 1, noa
                            ! l <-> i
                            lj = xixj_table(l,j)
                            if (lj/=0) then
                               phase = 1.0d0 * lj/abs(lj)
                               do jdet = id3b_h(jb,abs(lj),1), id3b_h(jb,abs(lj),2)
                                   d = t3b_excits(1,jdet); e = t3b_excits(2,jdet);
                                   if (a/=d .or. b/=e) cycle ! skip any p(a) difference
                                   ! compute sign(lj) * < ijkabc | h2b(oooo) | ljnabc >
                                   hmatel =  phase * h2b_oooo(l,n,i,k)
                                   !hmatel = 0.0d0
                                   !if (l > j) hmatel = hmatel - h2b_oooo(l,n,i,k)
                                   !if (l < j) hmatel = hmatel + h2b_oooo(l,n,i,k)
                                   resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                               end do
                            end if
                            ! l <-> j
                            li = xixj_table(l,i)
                            if (li/=0) then
                               phase = 1.0d0 * li/abs(li)
                               do jdet = id3b_h(jb,abs(li),1), id3b_h(jb,abs(li),2)
                                   d = t3b_excits(1,jdet); e = t3b_excits(2,jdet);
                                   if (a/=d .or. b/=e) cycle ! skip any p(a) difference
                                   ! compute sign(li) * < ijkabc | h2b(oooo) | ilnabc >
                                   hmatel = -phase * h2b_oooo(l,n,j,k)
                                   !hmatel = 0.0d0
                                   !if (l > i) hmatel = hmatel + h2b_oooo(l,n,j,k)
                                   !if (l < i) hmatel = hmatel - h2b_oooo(l,n,j,k)
                                   resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                               end do
                            end if
                         end do
                      end do
                      !!!! diagram 5: h1b(ce)*t3b(abeijm)
                      !!!! diagram 8: A(ab) h2b(bcef)*t3b(aefijk)
                      ! cpu cost = nub * nua**2/nua = nua*nub
                      ! loop extent = nub * nua**2
                      do f = 1, nub
                         jb = Eck_table(f,k)
                         do jdet = id3b_h(jb,ij,1), id3b_h(jb,ij,2)
                            d = t3b_excits(1,jdet); e = t3b_excits(2,jdet);
                            if (nexc2(a,b,d,e)>1) cycle ! skip if p(a) > 1
                            hmatel = 0.0d0
                            ! compute h2b(vvvv)
                            if (a==d) then
                                    hmatel = hmatel + h2b_vvvv(b,c,e,f) ! (1)
                            elseif (a==e) then
                                    hmatel = hmatel - h2b_vvvv(b,c,d,f) ! (ed)
                            end if 
                            if (b==d) then 
                                    hmatel = hmatel - h2b_vvvv(a,c,e,f) ! (ab)
                            elseif (b==e) then 
                                    hmatel = hmatel + h2b_vvvv(a,c,d,f) ! (ab)(ed)
                            end if
                            if (a==d .and. b==e) hmatel = hmatel + h1b_vv(c,f) ! compute h1b(vv)
                            resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                         end do
                      end do
                      !!!! diagram 9: A(ij)A(ab) h2a(amie)*t3b(ebcmjk)
                      ! cpu cost = noa**2/noa * nua**2/nua = noa*nua
                      ! loop extent = noa**2/noa * nua**2 = noa*nua**2
                      !do l = 1, noa
                      !   do m = l+1, noa
                      !      if (nexc2(i,j,l,m)>1) cycle ! skip if h(a) > 1
                      !      lm = xixj_table(l,m)
                      !      do jdet = id3b_h(ib,lm,1), id3b_h(ib,lm,2)
                      !         d = t3b_excits(1,jdet); e = t3b_excits(2,jdet); 
                      !         if (nexc2(a,b,d,e)>1) cycle ! skip if p(a) > 1
                      !         ! compute h2a(voov)
                      !         hmatel = 0.0d0
                      !         ! (1)
                      !         if (j==m) then ! (1)
                      !            if (b==e) then ! (1)
                      !                    hmatel = hmatel + h2a_voov(a,l,i,d)
                      !            elseif (b==d) then ! (de)
                      !                    hmatel = hmatel - h2a_voov(a,l,i,e)
                      !            end if
                      !         elseif (j==l) then ! (lm)
                      !            if (b==e) then ! (1)
                      !                    hmatel = hmatel - h2a_voov(a,m,i,d)
                      !            elseif (b==d) then ! (de)
                      !                    hmatel = hmatel + h2a_voov(a,m,i,e)
                      !            end if
                      !         end if
                      !         ! (ij)
                      !         if (i==m) then ! (1)
                      !            if (b==e) then ! (1)
                      !                    hmatel = hmatel - h2a_voov(a,l,j,d)
                      !            elseif (b==d) then ! (de)
                      !                    hmatel = hmatel + h2a_voov(a,l,j,e)
                      !            end if
                      !         elseif (i==l) then ! (lm)
                      !            if (b==e) then ! (1)
                      !                    hmatel = hmatel + h2a_voov(a,m,j,d)
                      !            elseif (b==d) then ! (de)
                      !                    hmatel = hmatel - h2a_voov(a,m,j,e)
                      !            end if
                      !         end if
                      !         ! (ab)
                      !         if (j==m) then ! (1)
                      !            if (a==e) then ! (1)
                      !                    hmatel = hmatel - h2a_voov(b,l,i,d)
                      !            elseif (a==d) then ! (de)
                      !                    hmatel = hmatel + h2a_voov(b,l,i,e)
                      !            end if
                      !         elseif (j==l) then ! (lm)
                      !            if (a==e) then ! (1)
                      !                    hmatel = hmatel + h2a_voov(b,m,i,d)
                      !            elseif (a==d) then ! (de)
                      !                    hmatel = hmatel - h2a_voov(b,m,i,e)
                      !            end if
                      !         end if
                      !         ! (ij)(ab)
                      !         if (i==m) then ! (1)
                      !            if (a==e) then ! (1)
                      !                    hmatel = hmatel + h2a_voov(b,l,j,d)
                      !            elseif (a==d) then ! (de)
                      !                    hmatel = hmatel - h2a_voov(b,l,j,e)
                      !            end if
                      !         elseif (i==l) then ! (lm)
                      !            if (a==e) then ! (1)
                      !                    hmatel = hmatel - h2a_voov(b,m,j,d)
                      !            elseif (a==d) then ! (de)
                      !                    hmatel = hmatel + h2a_voov(b,m,j,e)
                      !            end if
                      !         end if
                      !         resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                      !      end do
                      !   end do
                      !end do
                      do l = 1, noa
                         ! l <-> j
                         li = xixj_table(l,i)
                         if (li/=0) then
                            phase = 1.0d0 * li/abs(li)
                            do jdet = id3b_h(ib,abs(li),1), id3b_h(ib,abs(li),2)
                               d = t3b_excits(1,jdet); e = t3b_excits(2,jdet); 
                               if (nexc2(a,b,d,e)>1) cycle ! skip if p(a) > 1
                               ! compute sign(li) * < ijkabc | h2a(voov) | ilkdec >
                               hmatel = 0.0
                               if (b==e) then ! (1)
                                       hmatel = hmatel + h2a_voov(a,l,j,d)
                               elseif (b==d) then ! (de)
                                       hmatel = hmatel - h2a_voov(a,l,j,e)
                               end if
                               if (a==e) then ! (1)
                                       hmatel = hmatel - h2a_voov(b,l,j,d)
                                elseif (a==d) then ! (de)
                                       hmatel = hmatel + h2a_voov(b,l,j,e)
                               end if
                               hmatel = -phase * hmatel
                               !hmatel = 0.0d0
                               !if (l > i) then ! take j==m cases
                               !   if (b==e) then ! (1)
                               !           hmatel = hmatel + h2a_voov(a,l,j,d)
                               !   elseif (b==d) then ! (de)
                               !           hmatel = hmatel - h2a_voov(a,l,j,e)
                               !   end if
                               !   if (a==e) then ! (1)
                               !           hmatel = hmatel - h2a_voov(b,l,j,d)
                               !   elseif (a==d) then ! (de)
                               !           hmatel = hmatel + h2a_voov(b,l,j,e)
                               !   end if
                               !end if
                               !if (l < i) then ! take j==l cases
                               !   if (b==e) then ! (1)
                               !           hmatel = hmatel - h2a_voov(a,l,j,d)
                               !   elseif (b==d) then ! (de)
                               !           hmatel = hmatel + h2a_voov(a,l,j,e)
                               !   end if
                               !   if (a==e) then ! (1)
                               !           hmatel = hmatel + h2a_voov(b,l,j,d)
                               !   elseif (a==d) then ! (de)
                               !           hmatel = hmatel - h2a_voov(b,l,j,e)
                               !   end if
                               !end if
                               resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                            end do
                         end if
                         ! l <-> i
                         lj = xixj_table(l,j)
                         if (lj/=0) then
                            phase = 1.0d0 * lj/abs(lj)
                            do jdet = id3b_h(ib,abs(lj),1), id3b_h(ib,abs(lj),2)
                               d = t3b_excits(1,jdet); e = t3b_excits(2,jdet); 
                               if (nexc2(a,b,d,e)>1) cycle ! skip if p(a) > 1
                               ! compute sign(lj) * < ijkabc | h2a(voov) | ljkdec >
                               hmatel = 0.0d0
                               if (b==e) then ! (1)
                                       hmatel = hmatel - h2a_voov(a,l,i,d)
                               elseif (b==d) then ! (de)
                                       hmatel = hmatel + h2a_voov(a,l,i,e)
                               end if
                               if (a==e) then ! (1)
                                       hmatel = hmatel + h2a_voov(b,l,i,d)
                               elseif (a==d) then ! (de)
                                       hmatel = hmatel - h2a_voov(b,l,i,e)
                               end if
                               hmatel = -phase * hmatel
                               !hmatel = 0.0d0
                               !if (l > j) then ! take i==m cases
                               !   if (b==e) then ! (1)
                               !           hmatel = hmatel - h2a_voov(a,l,i,d)
                               !   elseif (b==d) then ! (de)
                               !           hmatel = hmatel + h2a_voov(a,l,i,e)
                               !   end if
                               !   if (a==e) then ! (1)
                               !           hmatel = hmatel + h2a_voov(b,l,i,d)
                               !   elseif (a==d) then ! (de)
                               !           hmatel = hmatel - h2a_voov(b,l,i,e)
                               !   end if
                               !end if
                               !if (l < j) then ! take i==l cases
                               !   if (b==e) then ! (1)
                               !           hmatel = hmatel + h2a_voov(a,l,i,d)
                               !   elseif (b==d) then ! (de)
                               !           hmatel = hmatel - h2a_voov(a,l,i,e)
                               !   end if
                               !   if (a==e) then ! (1)
                               !           hmatel = hmatel - h2a_voov(b,l,i,d)
                               !   elseif (a==d) then ! (de)
                               !           hmatel = hmatel + h2a_voov(b,l,i,e)
                               !   end if
                               !end if
                               resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                            end do
                         end if
                      end do
                      !!!! diagram 10: h2c(cmke)*t3b(abeijm)
                      ! cpu cost = nob*nub * nua**2/nua**2 = nob*nub
                      ! loop extent = nob*nub * nua**2
                      do n = 1, nob
                         do f = 1, nub
                            jb = Eck_table(f,n)
                            do jdet = id3b_h(jb,ij,1), id3b_h(jb,ij,2)
                               d = t3b_excits(1,jdet); e = t3b_excits(2,jdet);
                               if (a/=d .or. b/=e) cycle ! skip if any p(a) difference
                               ! compute h2c(voov)
                               hmatel = h2c_voov(c,n,k,f) ! (1)
                               resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                            end do
                         end do
                      end do
                      !!!! diagram 11: -A(ij) h2b(mcie)*t3b(abemjk)
                      ! cpu cost = nub * noa**2/noa * nua**2/nua**2 = nub*noa
                      ! loop extent = nub * noa**2/noa * nua**2 = nub*noa*nua**2
                      !do l = 1, noa
                      !   do m = l+1, noa
                      !      if (nexc2(i,j,l,m)>1) cycle ! skip if h(a) > 1
                      !      lm = XiXj_table(l,m)
                      !      do f = 1, nub
                      !         jb = Eck_table(f,k)
                      !         do jdet = id3b_h(jb,lm,1), id3b_h(jb,lm,2)
                      !            d = t3b_excits(1,jdet); e = t3b_excits(2,jdet);
                      !            if (a/=d .or. b/=e) cycle ! skip any p(a) difference
                      !            ! compute h2b(ovov)
                      !            hmatel = 0.0d0
                      !            if(j==m) then 
                      !                    hmatel = hmatel - h2b_ovov(l,c,i,f) ! (1)
                      !            elseif (j==l) then 
                      !                    hmatel = hmatel + h2b_ovov(m,c,i,f) ! (lm)
                      !            end if
                      !            if (i==m) then 
                      !                    hmatel = hmatel + h2b_ovov(l,c,j,f) ! (ij)
                      !            elseif (i==l) then 
                      !                    hmatel = hmatel - h2b_ovov(m,c,j,f) ! (ij)(lm)
                      !            end if
                      !            resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                      !         end do
                      !      end do
                      !   end do
                      !end do
                      do f = 1, nub
                         jb = eck_table(f,k)
                         do l = 1, noa
                            ! l <-> j
                            li = xixj_table(l,i)
                            if (li/=0) then
                               phase = 1.0d0 * li/abs(li)
                               do jdet = id3b_h(jb,abs(li),1), id3b_h(jb,abs(li),2)
                                  d = t3b_excits(1,jdet); e = t3b_excits(2,jdet);
                                  if (a/=d .or. b/=e) cycle
                                  ! compute sign(li) * < ijkabc | h2b(ovov) | ilkabf >
                                  hmatel = phase * h2b_ovov(l,c,j,f)
                                  !hmatel = 0.0d0
                                  !if (l > i) hmatel = hmatel - h2b_ovov(l,c,j,f)
                                  !if (l < i) hmatel = hmatel + h2b_ovov(l,c,j,f)
                                  resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                               end do
                            end if
                            ! l <-> i
                            lj = xixj_table(l,j)
                            if (lj/=0) then
                               phase = 1.0d0 * lj/abs(lj)
                               do jdet = id3b_h(jb,abs(lj),1), id3b_h(jb,abs(lj),2)
                                  d = t3b_excits(1,jdet); e = t3b_excits(2,jdet);
                                  if (a/=d .or. b/=e) cycle
                                  ! compute sign(lj) * < ijkabc | h2b(ovov) | ljkabf >
                                  hmatel = -phase * h2b_ovov(l,c,i,f)
                                  !hmatel = 0.0d0
                                  !if (l > j) hmatel = hmatel + h2b_ovov(l,c,i,f)
                                  !if (l < j) hmatel = hmatel - h2b_ovov(l,c,i,f)
                                  resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                               end do
                            end if
                         end do
                      end do
                      !!!! diagram 12: -A(ab) h2b(amek)*t3b(ebcijm)
                      ! cpu cost = nob * nua**2/nua = nob*nua
                      ! loop extent = nob * nua**2
                      do n = 1, nob
                         jb = Eck_table(c,n)
                         do jdet = id3b_h(jb,ij,1), id3b_h(jb,ij,2)
                            d = t3b_excits(1,jdet); e = t3b_excits(2,jdet);
                            if (nexc2(a,b,d,e)>1) cycle ! skip if p(a) > 1
                            ! compute h2b(vovo)
                            hmatel = 0.0d0
                            if (b==e) then 
                                    hmatel = hmatel - h2b_vovo(a,n,d,k) ! (1)
                            elseif (b==d) then 
                                    hmatel = hmatel + h2b_vovo(a,n,e,k) ! (de)
                            end if
                            if (a==e) then 
                                    hmatel = hmatel + h2b_vovo(b,n,d,k) ! (ab)
                            elseif (a==d) then 
                                    hmatel = hmatel - h2b_vovo(b,n,e,k) ! (ab)(de)
                            end if
                            resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                         end do
                      end do
                      !!!! diagram 13: h2b(mcek)*t3a(abeijm)
                      ! cpu cost = noa * nua**3/nua = noa*nua**2
                      ! loop extent = noa * nua**3
                      !do l = 1, noa
                      !   do m = l+1, noa
                      !      do n = m+1, noa
                      !         lmn = xixjxk_table(l,m,n)
                      !         ! (i,j) must be an ordered subset of (l,m,n)
                      !         if (i==l .and. j==m) then
                      !                 do jdet = id3a_h(lmn,1), id3a_h(lmn,2)
                      !                    d = t3a_excits(1,jdet); e = t3a_excits(2,jdet); f = t3a_excits(3,jdet);
                      !                    ! (a,b) must be an ordered subset of (d,e,f)
                      !                    hmatel = 0.0d0
                      !                    ! case 1: a = d, b = e
                      !                    if (a==d .and. b==e) then
                      !                       ! compute sign(lmn) * < ijk~abc~ | h2b(ovvo) | ijnabf >
                      !                       hmatel = hmatel + h2b_ovvo(n,c,f,k)
                      !                    end if
                      !                    ! case 2: a = e, b = f
                      !                    if (a==e .and. b==f) then
                      !                       ! compute sign(lmn) * < ijk~abc~ | h2b(ovvo) | ijndab >
                      !                       hmatel = hmatel + h2b_ovvo(n,c,d,k)
                      !                    end if
                      !                    ! case 3: a = d, b = f
                      !                    if (a==d .and. b==f) then
                      !                       ! compute sign(lmn) * < ijk~abc~ | h2b(ovvo) | ijnaeb >
                      !                       hmatel = hmatel - h2b_ovvo(n,c,e,k)
                      !                    end if
                      !                    resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                      !                 end do
                      !         elseif (i==l .and. j==n) then
                      !                 do jdet = id3a_h(lmn,1), id3a_h(lmn,2)
                      !                    d = t3a_excits(1,jdet); e = t3a_excits(2,jdet); f = t3a_excits(3,jdet);
                      !                    ! (a,b) must be an ordered subset of (d,e,f)
                      !                    hmatel = 0.0d0
                      !                    ! case 1: a = d, b = e
                      !                    if (a==d .and. b==e) then
                      !                       ! compute sign(lmn) * < ijk~abc~ | h2b(ovvo) | imjabf >
                      !                       hmatel = hmatel - h2b_ovvo(m,c,f,k)
                      !                    end if
                      !                    ! case 2: a = e, b = f
                      !                    if (a==e .and. b==f) then
                      !                       ! compute sign(lmn) * < ijk~abc~ | h2b(ovvo) | imjdab >
                      !                       hmatel = hmatel - h2b_ovvo(m,c,d,k)
                      !                    end if
                      !                    ! case 3: a = d, b = f
                      !                    if (a==d .and. b==f) then
                      !                       ! compute sign(lmn) * < ijk~abc~ | h2b(ovvo) | imjaeb >
                      !                       hmatel = hmatel + h2b_ovvo(m,c,e,k)
                      !                    end if
                      !                    resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                      !                 end do
                      !         elseif (i==m .and. j==n) then
                      !                 do jdet = id3a_h(lmn,1), id3a_h(lmn,2)
                      !                    d = t3a_excits(1,jdet); e = t3a_excits(2,jdet); f = t3a_excits(3,jdet);
                      !                    ! (a,b) must be an ordered subset of (d,e,f)
                      !                    hmatel = 0.0d0
                      !                    ! case 1: a = d, b = e
                      !                    if (a==d .and. b==e) then
                      !                       ! compute sign(lmn) * < ijk~abc~ | h2b(ovvo) | lijabf >
                      !                       hmatel = hmatel + h2b_ovvo(l,c,f,k)
                      !                    end if
                      !                    ! case 2: a = e, b = f
                      !                    if (a==e .and. b==f) then
                      !                       ! compute sign(lmn) * < ijk~abc~ | h2b(ovvo) | lijdab >
                      !                       hmatel = hmatel + h2b_ovvo(l,c,d,k)
                      !                    end if
                      !                    ! case 3: a = d, b = f
                      !                    if (a==d .and. b==f) then
                      !                       ! compute sign(lmn) * < ijk~abc~ | h2b(ovvo) | lijaeb >
                      !                       hmatel = hmatel - h2b_ovvo(l,c,e,k)
                      !                    end if
                      !                    resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                      !                 end do
                      !         end if
                      !      end do
                      !   end do
                      !end do
                      do n = 1, noa
                         lmn = xixjxk_table(i,j,n)
                         if (lmn==0) cycle
                         phase = 1.0d0 * lmn/abs(lmn)
                         do jdet = id3a_h(abs(lmn),1), id3a_h(abs(lmn),2)
                            d = t3a_excits(1,jdet); e = t3a_excits(2,jdet); f = t3a_excits(3,jdet);
                            ! (a,b) must be an ordered subset of (d,e,f)
                            hmatel = 0.0d0
                            ! case 1: a = d, b = e
                            if (a==d .and. b==e) then
                               ! compute sign(lmn) * < ijk~abc~ | h2b(ovvo) | ijnabf >
                               hmatel = hmatel + phase * h2b_ovvo(n,c,f,k)
                            end if
                            ! case 2: a = e, b = f
                            if (a==e .and. b==f) then
                               ! compute sign(lmn) * < ijk~abc~ | h2b(ovvo) | ijndab >
                               hmatel = hmatel + phase * h2b_ovvo(n,c,d,k)
                            end if
                            ! case 3: a = d, b = f
                            if (a==d .and. b==f) then
                               ! compute sign(lmn) * < ijk~abc~ | h2b(ovvo) | ijnaeb >
                               hmatel = hmatel - phase * h2b_ovvo(n,c,e,k)
                            end if
                            resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                         end do
                      end do
                      !!!! diagram 14: A(ab)A(ij) h2b(bmje)*t3c(aecimk)
                      ! cpu cost = nob * 4*nub**2/nub = 4*nob*nub
                      ! loop extent = nob * 4*nub**2 = 4*nob*nub**2
                      !do m1 = 1, nob
                      !   if (m1==k) cycle
                      !   mn = xjxk_table(m1,k)
                      !   ai = eai_table(a,i)
                      !   bj = eai_table(b,j)
                      !   aj = eai_table(a,j)
                      !   bi = eai_table(b,i)

                      !   do jdet = id3c_h(ai,mn,1), id3c_h(ai,mn,2)
                      !      ! < ijkabc | h2b(voov) | imkaef >
                      !      d = t3c_excits(1,jdet); e = t3c_excits(2,jdet); f = t3c_excits(3,jdet);
                      !      l = t3c_excits(4,jdet); m = t3c_excits(5,jdet); n = t3c_excits(6,jdet);
                      !      if (a/=d .or. i/=l) cycle
                      !      ! c must be equal to one of (e,f)
                      !      if (c/=e .and. c/=f) cycle
                      !      hmatel = 0.0d0
                      !      if (c==f) then
                      !         if (k==n) hmatel = hmatel + h2b_voov(b,m,j,e) 
                      !         if (k==m) hmatel = hmatel - h2b_voov(b,n,j,e)
                      !      elseif (c==e) then
                      !         if (k==n) hmatel = hmatel - h2b_voov(b,m,j,f) 
                      !         if (k==m) hmatel = hmatel + h2b_voov(b,n,j,f)
                      !      end if 
                      !      resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                      !   end do

                      !   do jdet = id3c_h(bj,mn,1), id3c_h(bj,mn,2)
                      !      ! < ijkabc | h2b(voov) | jmkbef >
                      !      d = t3c_excits(1,jdet); e = t3c_excits(2,jdet); f = t3c_excits(3,jdet);
                      !      l = t3c_excits(4,jdet); m = t3c_excits(5,jdet); n = t3c_excits(6,jdet);
                      !      if (b/=d .or. j/=l) cycle
                      !      ! c must be equal to one of (e,f)
                      !      if (c/=e .and. c/=f) cycle
                      !      hmatel = 0.0d0
                      !      if (c==f) then
                      !         if (k==n) hmatel = hmatel + h2b_voov(a,m,i,e) 
                      !         if (k==m) hmatel = hmatel - h2b_voov(a,n,i,e)
                      !      elseif (c==e) then
                      !         if (k==n) hmatel = hmatel - h2b_voov(a,m,i,f) 
                      !         if (k==m) hmatel = hmatel + h2b_voov(a,n,i,f)
                      !      end if
                      !      resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                      !   end do

                      !   do jdet = id3c_h(aj,mn,1), id3c_h(aj,mn,2)
                      !      ! < ijkabc | h2b(voov) | jmkaef >
                      !      d = t3c_excits(1,jdet); e = t3c_excits(2,jdet); f = t3c_excits(3,jdet);
                      !      l = t3c_excits(4,jdet); m = t3c_excits(5,jdet); n = t3c_excits(6,jdet);
                      !      if (a/=d .or. j/=l) cycle
                      !      ! c must be equal to one of (e,f)
                      !      if (c/=e .and. c/=f) cycle
                      !      hmatel = 0.0d0
                      !      if (c==f) then
                      !         if (k==n) hmatel = hmatel - h2b_voov(b,m,i,e)
                      !         if (k==m) hmatel = hmatel + h2b_voov(b,n,i,e) 
                      !      elseif (c==e) then
                      !         if (k==n) hmatel = hmatel + h2b_voov(b,m,i,f) 
                      !         if (k==m) hmatel = hmatel - h2b_voov(b,n,i,f) 
                      !      end if
                      !      resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                      !   end do

                      !   do jdet = id3c_h(bi,mn,1), id3c_h(bi,mn,2)
                      !      ! < ijkabc | h2b(voov) | jmkbef >
                      !      d = t3c_excits(1,jdet); e = t3c_excits(2,jdet); f = t3c_excits(3,jdet);
                      !      l = t3c_excits(4,jdet); m = t3c_excits(5,jdet); n = t3c_excits(6,jdet);
                      !      if (b/=d .or. i/=l) cycle
                      !      ! c must be equal to one of (e,f)
                      !      if (c/=e .and. c/=f) cycle
                      !      hmatel = 0.0d0
                      !      if (c==f) then
                      !         if (k==n) hmatel = hmatel - h2b_voov(a,m,j,e) 
                      !         if (k==m) hmatel = hmatel + h2b_voov(a,n,j,e) 
                      !      elseif (c==e) then
                      !         if (k==n) hmatel = hmatel + h2b_voov(a,m,j,f) 
                      !         if (k==m) hmatel = hmatel - h2b_voov(a,n,j,f) 
                      !      end if
                      !      resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                      !   end do
                      !end do
                      do m = 1, nob 
                         if (m==k) cycle

                         mk = xjxk_table(m,k)
                         phase = 1.0d0 * mk/abs(mk)
                         ai = eai_table(a,i); bj = eai_table(b,j); aj = eai_table(a,j); bi = eai_table(b,i);

                         do jdet = id3c_h(ai,abs(mk),1), id3c_h(ai,abs(mk),2)
                            e = t3c_excits(2,jdet); f = t3c_excits(3,jdet);
                            ! c must equal one of (e,f)
                            hmatel = 0.0d0
                            ! compute sign(mk) * < ijk~abc~ | h2b(voov) | im~k~ae~f~ >
                            if (c==e) then
                               hmatel = hmatel - phase * h2b_voov(b,m,j,f)
                            elseif (c==f) then
                               hmatel = hmatel + phase * h2b_voov(b,m,j,e)
                            end if
                            resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                         end do

                         do jdet = id3c_h(bj,abs(mk),1), id3c_h(bj,abs(mk),2)
                            e = t3c_excits(2,jdet); f = t3c_excits(3,jdet);
                            ! c must equal one of (e,f)
                            hmatel = 0.0d0
                            ! compute sign(mk) * < ijk~abc~ | h2b(voov) | jm~k~be~f~ >
                            if (c==e) then
                                hmatel = hmatel - phase * h2b_voov(a,m,i,f)
                            elseif (c==f) then
                                hmatel = hmatel + phase * h2b_voov(a,m,i,e)
                            end if
                            resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                         end do

                         do jdet = id3c_h(aj,abs(mk),1), id3c_h(aj,abs(mk),2)
                            e = t3c_excits(2,jdet); f = t3c_excits(3,jdet);
                            ! c must equal one of (e,f)
                            hmatel = 0.0d0
                            ! compute sign(mk) * < ijk~abc~ | h2b(voov) | jm~k~ae~f~ >
                            if (c==e) then
                                hmatel = hmatel + phase * h2b_voov(b,m,i,f)
                            elseif (c==f) then
                                hmatel = hmatel - phase * h2b_voov(b,m,i,e)
                            end if
                            resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                         end do

                         do jdet = id3c_h(bi,abs(mk),1), id3c_h(bi,abs(mk),2)
                            e = t3c_excits(2,jdet); f = t3c_excits(3,jdet);
                            ! c must equal one of (e,f)
                            hmatel = 0.0d0
                            ! compute sign(mk) * < ijk~abc~ | h2b(voov) | im~k~be~f~ >
                            if (c==e) then
                                hmatel = hmatel + phase * h2b_voov(a,m,j,f)
                            elseif (c==f) then
                                hmatel = hmatel - phase * h2b_voov(a,m,j,e)
                            end if
                            resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                         end do
                      end do                      

                  end do ! end loop over idet
                  !$omp end do

                  !$omp do
                  do idet = 1, n3aab
                      a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);

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

                      resid(idet) = (resid(idet) + res_mm23)/(denom - shift)

                      t3b_amps(idet) = t3b_amps(idet) + resid(idet)
                  end do
                  !$omp end do

                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
              end subroutine update_t3b_p

              subroutine update_t3c_p(t3c_amps, resid,&
                                      t3b_excits, t3c_excits, t3d_excits,&
                                      t2b, t2c,&
                                      t3b_amps, t3d_amps,&
                                      id3b_h, eck_table, xixj_table,&
                                      id3c_h, eai_table, xjxk_table,&
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
                  integer, intent(in) :: t3b_excits(6,n3aab), t3c_excits(6,n3abb), t3d_excits(6,n3bbb)
                  integer, intent(in) :: id3b_h(nub*nob,noa*(noa-1)/2,2), id3c_h(nua*noa,nob*(nob-1)/2,2)
                  integer, intent(in) :: eck_table(nub,nob), eai_table(nua,noa)
                  integer, intent(in) :: xixj_table(noa,noa), xjxk_table(nob,nob)
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
                  real(kind=8) :: denom, val, t_amp, res_mm23, hmatel
                  integer :: i, j, k, l, a, b, c, d, m, n, e, f, idet, jdet
                  integer :: ia, jk, ja, mn
                  integer :: bj, ck, bk, cj, lm, l1

                  ! Zero the residual
                  resid = 0.0d0
                  ! VT3 intermediates
                  I2C_vooo(:,:,:,:) = 0.5d0 * H2C_vooo(:,:,:,:)
                  I2C_vvov(:,:,:,:) = 0.5d0 * H2C_vvov(:,:,:,:)
                  I2B_vooo(:,:,:,:) = H2B_vooo(:,:,:,:)
                  I2B_ovoo(:,:,:,:) = H2B_ovoo(:,:,:,:)
                  I2B_vvov(:,:,:,:) = H2B_vvov(:,:,:,:)
                  I2B_vvvo(:,:,:,:) = H2B_vvvo(:,:,:,:)

                  do idet = 1, n3aab
                      t_amp = t3b_amps(idet)

                      ! I2B(abej) <- A(af) -h2a(mnef) * t3b(afbmnj)
                      a = t3b_excits(1,idet); f = t3b_excits(2,idet); b = t3b_excits(3,idet);
                      m = t3b_excits(4,idet); n = t3b_excits(5,idet); j = t3b_excits(6,idet);
                      I2B_vvvo(a,b,:,j) = I2B_vvvo(a,b,:,j) - H2A_oovv(m,n,:,f) * t_amp ! (1)
                      I2B_vvvo(f,b,:,j) = I2B_vvvo(f,b,:,j) + H2A_oovv(m,n,:,a) * t_amp ! (af)

                      ! I2B(mbij) <- A(in) h2a(mnef) * t3b(efbinj)
                      e = t3b_excits(1,idet); f = t3b_excits(2,idet); b = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); n = t3b_excits(5,idet); j = t3b_excits(6,idet);
                      I2B_ovoo(:,b,i,j) = I2B_ovoo(:,b,i,j) + H2A_oovv(:,n,e,f) * t_amp ! (1)
                      I2B_ovoo(:,b,n,j) = I2B_ovoo(:,b,n,j) - H2A_oovv(:,i,e,f) * t_amp ! (in)

                      ! I2B(abie) <- A(af)A(in) -h2b(nmfe) * t3b(afbinm)
                      a = t3b_excits(1,idet); f = t3b_excits(2,idet); b = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); n = t3b_excits(5,idet); m = t3b_excits(6,idet);
                      I2B_vvov(a,b,i,:) = I2B_vvov(a,b,i,:) - H2B_oovv(n,m,f,:) * t_amp ! (1)
                      I2B_vvov(f,b,i,:) = I2B_vvov(f,b,i,:) + H2B_oovv(n,m,a,:) * t_amp ! (af)
                      I2B_vvov(a,b,n,:) = I2B_vvov(a,b,n,:) + H2B_oovv(i,m,f,:) * t_amp ! (in)
                      I2B_vvov(f,b,n,:) = I2B_vvov(f,b,n,:) - H2B_oovv(i,m,a,:) * t_amp ! (af)(in)

                      ! I2B(amij) <- A(af)A(in) h2b(nmfe) * t3b(afeinj)
                      a = t3b_excits(1,idet); f = t3b_excits(2,idet); e = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); n = t3b_excits(5,idet); j = t3b_excits(6,idet);
                      I2B_vooo(a,:,i,j) = I2B_vooo(a,:,i,j) + H2B_oovv(n,:,f,e) * t_amp ! (1)
                      I2B_vooo(f,:,i,j) = I2B_vooo(f,:,i,j) - H2B_oovv(n,:,a,e) * t_amp ! (af)
                      I2B_vooo(a,:,n,j) = I2B_vooo(a,:,n,j) - H2B_oovv(i,:,f,e) * t_amp ! (in)
                      I2B_vooo(f,:,n,j) = I2B_vooo(f,:,n,j) + H2B_oovv(i,:,a,e) * t_amp ! (af)(in)
                  end do

                  do idet = 1, n3abb
                      t_amp = t3c_amps(idet)

                      ! I2C(abie) <- A(ab) [A(im) -h2b(nmfe) * t3c(fabnim)]
                      f = t3c_excits(1,idet); a = t3c_excits(2,idet); b = t3c_excits(3,idet);
                      n = t3c_excits(4,idet); i = t3c_excits(5,idet); m = t3c_excits(6,idet);
                      I2C_vvov(a,b,i,:) = I2C_vvov(a,b,i,:) - H2B_oovv(n,m,f,:) * t_amp ! (1)
                      I2C_vvov(a,b,m,:) = I2C_vvov(a,b,m,:) + H2B_oovv(n,i,f,:) * t_amp ! (im)

                      ! I2C(amij) <- A(ij) [A(ae) h2b(nmfe) * t3c(faenij)]
                      f = t3c_excits(1,idet); a = t3c_excits(2,idet); e = t3c_excits(3,idet);
                      n = t3c_excits(4,idet); i = t3c_excits(5,idet); j = t3c_excits(6,idet);
                      I2C_vooo(a,:,i,j) = I2C_vooo(a,:,i,j) + H2B_oovv(n,:,f,e) * t_amp ! (1)
                      I2C_vooo(e,:,i,j) = I2C_vooo(e,:,i,j) - H2B_oovv(n,:,f,a) * t_amp ! (ae)

                      ! I2B(abej) <- A(bf)A(jn) -h2b(mnef) * t3c(afbmnj)
                      a = t3c_excits(1,idet); f = t3c_excits(2,idet); b = t3c_excits(3,idet);
                      m = t3c_excits(4,idet); n = t3c_excits(5,idet); j = t3c_excits(6,idet);
                      I2B_vvvo(a,b,:,j) = I2B_vvvo(a,b,:,j) - H2B_oovv(m,n,:,f) * t_amp ! (1)
                      I2B_vvvo(a,f,:,j) = I2B_vvvo(a,f,:,j) + H2B_oovv(m,n,:,b) * t_amp ! (bf)
                      I2B_vvvo(a,b,:,n) = I2B_vvvo(a,b,:,n) + H2B_oovv(m,j,:,f) * t_amp ! (jn)
                      I2B_vvvo(a,f,:,n) = I2B_vvvo(a,f,:,n) - H2B_oovv(m,j,:,b) * t_amp ! (bf)(jn)

                      ! I2B(mbij) <- A(bf)A(jn) h2B(mnef) * t3c(efbinj)
                      e = t3c_excits(1,idet); f = t3c_excits(2,idet); b = t3c_excits(3,idet);
                      i = t3c_excits(4,idet); n = t3c_excits(5,idet); j = t3c_excits(6,idet);
                      I2B_ovoo(:,b,i,j) = I2B_ovoo(:,b,i,j) + H2B_oovv(:,n,e,f) * t_amp ! (1)
                      I2B_ovoo(:,f,i,j) = I2B_ovoo(:,f,i,j) - H2B_oovv(:,n,e,b) * t_amp ! (bf)
                      I2B_ovoo(:,b,i,n) = I2B_ovoo(:,b,i,n) - H2B_oovv(:,j,e,f) * t_amp ! (jn)
                      I2B_ovoo(:,f,i,n) = I2B_ovoo(:,f,i,n) + H2B_oovv(:,j,e,b) * t_amp ! (bf)(jn)

                      ! I2B(abie) <- A(bf) -h2c(nmfe) * t3c(afbinm)
                      a = t3c_excits(1,idet); f = t3c_excits(2,idet); b = t3c_excits(3,idet);
                      i = t3c_excits(4,idet); n = t3c_excits(5,idet); m = t3c_excits(6,idet);
                      I2B_vvov(a,b,i,:) = I2B_vvov(a,b,i,:) - H2C_oovv(n,m,f,:) * t_amp ! (1)
                      I2B_vvov(a,f,i,:) = I2B_vvov(a,f,i,:) + H2C_oovv(n,m,b,:) * t_amp ! (bf)

                      ! I2B(amij) <- A(jn) h2c(nmfe) * t3c(afeinj)
                      a = t3c_excits(1,idet); f = t3c_excits(2,idet); e = t3c_excits(3,idet);
                      i = t3c_excits(4,idet); n = t3c_excits(5,idet); j = t3c_excits(6,idet);
                      I2B_vooo(a,:,i,j) = I2B_vooo(a,:,i,j) + H2C_oovv(n,:,f,e) * t_amp ! (1)
                      I2B_vooo(a,:,i,n) = I2B_vooo(a,:,i,n) - H2C_oovv(j,:,f,e) * t_amp ! (jn)
                  end do

                  do idet = 1, n3bbb
                      t_amp = t3d_amps(idet)

                      ! I2C(amij) <- A(ij) [A(n/ij)A(a/ef) h2c(mnef) * t3d(aefijn)]
                      a = t3d_excits(1,idet); e = t3d_excits(2,idet); f = t3d_excits(3,idet);
                      i = t3d_excits(4,idet); j = t3d_excits(5,idet); n = t3d_excits(6,idet);
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
                      a = t3d_excits(1,idet); b = t3d_excits(2,idet); f = t3d_excits(3,idet);
                      i = t3d_excits(4,idet); m = t3d_excits(5,idet); n = t3d_excits(6,idet);
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

                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,t3c_excits,t3d_excits,&
                  !$omp t3b_amps,t3c_amps,t3d_amps,t2b,t2c,&
                  !$omp id3b_h,eck_table,xixj_table,&
                  !$omp id3c_h,eai_table,xjxk_table,&
                  !$omp H1A_oo,H1B_oo,H1A_vv,H1B_vv,H2B_oooo,H2C_oooo,&
                  !$omp H2B_ovvo,H2B_voov,H2C_vvvv,H2B_vvvv,&
                  !$omp H2A_voov,H2C_voov,H2B_ovov,H2B_vovo,&
                  !$omp I2C_vooo,I2C_vvov,I2B_vooo,I2B_ovoo,&
                  !$omp I2B_vvov,I2B_vvvo,&
                  !$omp fA_oo,fB_oo,fA_vv,fB_vv,nua,nub,noa,nob,&
                  !$omp shift,n3aab,n3abb,n3bbb),&
                  !$omp private(a,b,c,d,i,j,k,l,m,n,e,f,denom,t_amp,hmatel,idet,jdet,&
                  !$omp ia,jk,ja,mn,bj,ck,bk,cj,lm,l1)

                  !$omp do
                  do idet = 1, n3abb
                      a = t3c_excits(1,idet); b = t3c_excits(2,idet); c = t3c_excits(3,idet);
                      i = t3c_excits(4,idet); j = t3c_excits(5,idet); k = t3c_excits(6,idet);
                      ia = eai_table(a,i)
                      jk = xjxk_table(j,k)

                      ! diagram 1: h1b(oo)
                      ! diagram 5: h2c(oooo)
                      do m = 1, nob
                         do n = m+1, nob
                            mn = xjxk_table(m,n)
                            do jdet = id3c_h(ia,mn,1), id3c_h(ia,mn,2)
                               e = t3c_excits(2,jdet); f = t3c_excits(3,jdet);
                               if (b/=e .or. c/=f) cycle ! skip if any p(b) difference
                               ! compute h2c(oooo)
                               hmatel = h2c_oooo(m,n,j,k)
                               if (nexc2(j,k,m,n)<2) then ! compute h1b(oo)
                                       if (k==n) then
                                               hmatel = hmatel - h1b_oo(m,j)
                                       elseif (k==m) then
                                               hmatel = hmatel + h1b_oo(n,j)
                                       end if
                                       if (j==n) then
                                               hmatel = hmatel + h1b_oo(m,k)
                                       elseif (j==m) then
                                               hmatel = hmatel - h1b_oo(n,k)
                                       end if
                               end if
                               resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                            end do
                         end do
                      end do
                      ! diagram 2: h1b(vv)
                      ! diagram 6: h2c(vvvv)
                      do jdet = id3c_h(ia,jk,1), id3c_h(ia,jk,2)
                         e = t3c_excits(2,jdet); f = t3c_excits(3,jdet);
                         ! compute h2c(vvvv)
                         hmatel = h2c_vvvv(b,c,e,f)
                         if (nexc2(b,c,e,f)<2) then ! compute h1b(vv)
                             if (b==f) then 
                                     hmatel = hmatel - h1b_vv(c,e) 
                             elseif (b==e) then
                                     hmatel = hmatel + h1b_vv(c,f) 
                             end if
                             if (c==f) then 
                                     hmatel = hmatel + h1b_vv(b,e)
                             elseif (c==e) then
                                     hmatel = hmatel - h1b_vv(b,f)
                             end if
                         end if
                         resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                      end do
                      ! diagram 3: h1a(oo)
                      ! diagram 7: h2b(oooo)
                      do m = 1, nob
                         do n = m+1, nob
                            if (nexc2(j,k,m,n)>1) cycle ! skip if h(b) > 1
                            mn = xjxk_table(m,n)
                            do l = 1, noa
                               ja = eai_table(a,l)
                               do jdet = id3c_h(ja,mn,1), id3c_h(ja,mn,2)
                                  e = t3c_excits(2,jdet); f = t3c_excits(3,jdet);
                                  if (b/=e .or. c/=f) cycle ! skip any p(b) difference
                                  hmatel = 0.0d0
                                  ! compute h2b(oooo)
                                  if (j==m) then 
                                          hmatel = hmatel + h2b_oooo(l,n,i,k) ! (1)
                                  elseif (j==n) then
                                          hmatel = hmatel - h2b_oooo(l,m,i,k) ! (lm)
                                  end if
                                  if (k==m) then 
                                          hmatel = hmatel - h2b_oooo(l,n,i,j) ! (ij)
                                  elseif (k==n) then 
                                          hmatel = hmatel + h2b_oooo(l,m,i,j) ! (ij)(lm)
                                  end if
                                  if (j==m .and. k==n) hmatel = hmatel - h1a_oo(l,i) ! compute h1a(oo)
                                  resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                               end do
                            end do
                         end do
                      end do
                      ! diagram 5: h1a(vv)
                      ! diagram 8: h2b(vvvv)
                      do d = 1, nua
                         ja = eai_table(d,i)
                         do jdet = id3c_h(ja,jk,1), id3c_h(ja,jk,2)
                            e = t3c_excits(2,jdet); f = t3c_excits(3,jdet);
                            if (nexc2(b,c,e,f)>1) cycle ! skip if p(b) > 1
                            hmatel = 0.0d0
                            ! compute h2b(vvvv)
                            if (b==e) then
                                    hmatel = hmatel + h2b_vvvv(a,c,d,f) ! (1)
                            elseif (b==f) then
                                    hmatel = hmatel - h2b_vvvv(a,c,d,e) ! (ed)
                            end if 
                            if (c==e) then 
                                    hmatel = hmatel - h2b_vvvv(a,b,d,f) ! (ab)
                            elseif (c==f) then 
                                    hmatel = hmatel + h2b_vvvv(a,b,d,e) ! (ab)(ed)
                            end if
                            if (b==e .and. c==f) hmatel = hmatel + h1a_vv(a,d) ! compute h1b(vv)
                            resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                         end do
                      end do


                     !do jdet = 1, n3aab
                     !   d = t3b_excits(1,jdet); e = t3b_excits(2,jdet); f = t3b_excits(3,jdet);
                     !   l = t3b_excits(4,jdet); m = t3b_excits(5,jdet); n = t3b_excits(6,jdet);

                     !   ! Check for 3 differences and early exit
                     !   if (f/=b .and. f/=c) cycle
                     !   if (n/=j .and. n/=k) cycle
                     !   if (a/=d .and. a/=e) cycle
                     !   if (i/=l .and. i/=m) cycle

                     !   hmatel = 0.0d0
                     !   t_amp = t3b_amps(jdet)
                     !   hmatel = hmatel + abb_ovvo_aab(i,j,k,a,b,c,l,m,n,d,e,f,h2b_ovvo,noa,nua,nob,nub)
                     !   if (hmatel /= 0.0d0) resid(idet) = resid(idet) + hmatel * t_amp
                     !end do
                     !do jdet = 1, n3abb
                     !   d = t3c_excits(1,jdet); e = t3c_excits(2,jdet); f = t3c_excits(3,jdet);
                     !   l = t3c_excits(4,jdet); m = t3c_excits(5,jdet); n = t3c_excits(6,jdet);

                     !   hmatel = 0.0d0
                     !   t_amp = t3c_amps(jdet)
                     !   hmatel = hmatel + abb_oo_abb(i,j,k,a,b,c,l,m,n,d,e,f,h1a_oo,h1b_oo,noa,nob)
                     !   hmatel = hmatel + abb_vv_abb(i,j,k,a,b,c,l,m,n,d,e,f,h1a_vv,h1b_vv,nua,nub)
                     !   hmatel = hmatel + abb_oooo_abb(i,j,k,a,b,c,l,m,n,d,e,f,h2c_oooo,h2b_oooo,noa,nob)
                     !   hmatel = hmatel + abb_vvvv_abb(i,j,k,a,b,c,l,m,n,d,e,f,h2c_vvvv,h2b_vvvv,nua,nub)
                     !   hmatel = hmatel + abb_voov_abb(i,j,k,a,b,c,l,m,n,d,e,f,h2a_voov,h2c_voov,noa,nua,nob,nub)
                     !   hmatel = hmatel + abb_ovov_abb(i,j,k,a,b,c,l,m,n,d,e,f,h2b_ovov,noa,nub)
                     !   hmatel = hmatel + abb_vovo_abb(i,j,k,a,b,c,l,m,n,d,e,f,h2b_vovo,nua,nob)
                     !   if (hmatel /= 0.0d0) resid(idet) = resid(idet) + hmatel * t_amp
                     !end do


                     do jdet = 1, n3bbb
                        d = t3d_excits(1,jdet); e = t3d_excits(2,jdet); f = t3d_excits(3,jdet);
                        l = t3d_excits(4,jdet); m = t3d_excits(5,jdet); n = t3d_excits(6,jdet);

                        ! Check for 3 differences and early exit
                        if ((f/=c .and. e/=b) .and. (f/=c .and. f/=b) .and. (e/=c .and. f/=b)) cycle
                        if ((n/=k .and. m/=j) .and. (n/=k .and. n/=j) .and. (m/=k .and. n/=j)) cycle

                        hmatel = 0.0d0
                        t_amp = t3d_amps(jdet)
                        hmatel = hmatel + abb_voov_bbb(i,j,k,a,b,c,l,m,n,d,e,f,h2b_voov,noa,nua,nob,nub)
                        if (hmatel /= 0.0d0) resid(idet) = resid(idet) + hmatel * t_amp
                     end do
                  end do
                  !$omp end do

                  !$omp do
                  do idet = 1, n3abb
                      a = t3c_excits(1,idet); b = t3c_excits(2,idet); c = t3c_excits(3,idet);
                      i = t3c_excits(4,idet); j = t3c_excits(5,idet); k = t3c_excits(6,idet);

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

                      resid(idet) = (resid(idet) + res_mm23)/(denom - shift)

                      t3c_amps(idet) = t3c_amps(idet) + resid(idet)
                  end do
                  !$omp end do

                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!

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
                  integer, intent(in) :: t3c_excits(6, n3abb), t3d_excits(6, n3bbb)
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

                  real(kind=8) :: val, denom, t_amp, res_mm23, hmatel
                  real(kind=8) :: I2C_vooo(nub, nob, nob, nob),&
                                  I2C_vvov(nub, nub, nob, nub)
                  integer :: a, b, c, d, i, j, k, l, e, f, m, n, idet, jdet

                  ! Zero the residual
                  resid = 0.0d0
                  ! compute VT3 intermediates
                  I2C_vooo(:,:,:,:) = 0.5d0 * H2C_vooo(:,:,:,:)
                  I2C_vvov(:,:,:,:) = 0.5d0 * H2C_vvov(:,:,:,:)

                  do idet = 1, n3abb
                      t_amp = t3c_amps(idet)

                      ! I2C(abie) <- A(ab) [A(im) -h2b(nmfe) * t3c(fabnim)]
                      f = t3c_excits(1,idet); a = t3c_excits(2,idet); b = t3c_excits(3,idet);
                      n = t3c_excits(4,idet); i = t3c_excits(5,idet); m = t3c_excits(6,idet);
                      I2C_vvov(a,b,i,:) = I2C_vvov(a,b,i,:) - H2B_oovv(n,m,f,:) * t_amp ! (1)
                      I2C_vvov(a,b,m,:) = I2C_vvov(a,b,m,:) + H2B_oovv(n,i,f,:) * t_amp ! (im)

                      ! I2C(amij) <- A(ij) [A(ae) h2b(nmfe) * t3c(faenij)]
                      f = t3c_excits(1,idet); a = t3c_excits(2,idet); e = t3c_excits(3,idet);
                      n = t3c_excits(4,idet); i = t3c_excits(5,idet); j = t3c_excits(6,idet);
                      I2C_vooo(a,:,i,j) = I2C_vooo(a,:,i,j) + H2B_oovv(n,:,f,e) * t_amp ! (1)
                      I2C_vooo(e,:,i,j) = I2C_vooo(e,:,i,j) - H2B_oovv(n,:,f,a) * t_amp ! (ae)
                  end do

                  do idet = 1, n3bbb
                      t_amp = t3d_amps(idet)

                      ! I2C(amij) <- A(ij) [A(n/ij)A(a/ef) h2c(mnef) * t3d(aefijn)]
                      a = t3d_excits(1,idet); e = t3d_excits(2,idet); f = t3d_excits(3,idet);
                      i = t3d_excits(4,idet); j = t3d_excits(5,idet); n = t3d_excits(6,idet);
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
                      a = t3d_excits(1,idet); b = t3d_excits(2,idet); f = t3d_excits(3,idet);
                      i = t3d_excits(4,idet); m = t3d_excits(5,idet); n = t3d_excits(6,idet);
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

                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3c_excits,t3d_excits,t3c_amps,t3d_amps,t2c,&
                  !$omp H1B_oo,H1B_vv,H2C_oooo,&
                  !$omp H2C_vvvv,H2C_voov,H2B_ovvo,I2C_vooo,I2C_vvov,&
                  !$omp fB_oo,fB_vv,shift,noa,nua,nob,nub,n3abb,n3bbb),&
                  !$omp private(hmatel,t_amp,denom,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet)

                  !$omp do schedule(static)
                  do idet = 1, n3bbb
                      a = t3d_excits(1,idet); b = t3d_excits(2,idet); c = t3d_excits(3,idet);
                      i = t3d_excits(4,idet); j = t3d_excits(5,idet); k = t3d_excits(6,idet);
                      do jdet = 1, n3bbb
                          d = t3d_excits(1,jdet); e = t3d_excits(2,jdet); f = t3d_excits(3,jdet);
                          l = t3d_excits(4,jdet); m = t3d_excits(5,jdet); n = t3d_excits(6,jdet);

                          hmatel = 0.0d0
                          t_amp = t3d_amps(jdet)
                          hmatel = hmatel + bbb_oo_bbb(i,j,k,a,b,c,l,m,n,d,e,f,h1b_oo,nob)
                          hmatel = hmatel + bbb_vv_bbb(i,j,k,a,b,c,l,m,n,d,e,f,h1b_vv,nub)
                          hmatel = hmatel + bbb_oooo_bbb(i,j,k,a,b,c,l,m,n,d,e,f,h2c_oooo,nob)
                          hmatel = hmatel + bbb_vvvv_bbb(i,j,k,a,b,c,l,m,n,d,e,f,h2c_vvvv,nub)
                          hmatel = hmatel + bbb_voov_bbb(i,j,k,a,b,c,l,m,n,d,e,f,h2c_voov,nob,nub)
                          if (hmatel /= 0.0d0) resid(idet) = resid(idet) + hmatel * t_amp
                      end do
                      do jdet = 1, n3abb
                          d = t3c_excits(1,jdet); e = t3c_excits(2,jdet); f = t3c_excits(3,jdet);
                          l = t3c_excits(4,jdet); m = t3c_excits(5,jdet); n = t3c_excits(6,jdet);

                          hmatel = 0.0d0
                          t_amp = t3c_amps(jdet)
                          hmatel = hmatel + bbb_ovvo_abb(i,j,k,a,b,c,l,m,n,d,e,f,h2b_ovvo,noa,nua,nob,nub)
                          if (hmatel /= 0.0d0) resid(idet) = resid(idet) + hmatel * t_amp
                      end do
                  end do
                  !$omp end do
                  
                  !$omp do
                  do idet = 1, n3bbb
                      a = t3d_excits(1,idet); b = t3d_excits(2,idet); c = t3d_excits(3,idet);
                      i = t3d_excits(4,idet); j = t3d_excits(5,idet); k = t3d_excits(6,idet);

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

                      resid(idet) = (resid(idet) + res_mm23)/(denom - shift)
                      t3d_amps(idet) = t3d_amps(idet) + resid(idet)
                  end do
                  !$omp end do

                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!

              end subroutine update_t3d_p

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!! HBAR MATRIX ELEMENTS !!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

          pure function aaa_oo_aaa(i, j, k, a, b, c, l, m, n, d, e, f, h, noa) result(hmatel)
                  ! Expression:
                  ! -A(abc)A(jk)A(l/mn)A(i/jk) d(a,d)d(b,e)d(c,f)d(j,m)d(k,n) h(l,i)

                  integer, intent(in) :: noa
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: h(1:noa,1:noa)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  if (a==d .and. b==e .and. c==f) then

                          if (j==m .and. k==n) hmatel = hmatel - h(l,i) ! (1)
                          if (j==l .and. k==n) hmatel = hmatel + h(m,i) ! (lm)
                          if (i==m .and. k==n) hmatel = hmatel + h(l,j) ! (ij)
                          if (i==l .and. k==n) hmatel = hmatel - h(m,j) ! (lm)(ij)

                          if (j==m .and. i==l) hmatel = hmatel - h(n,k) ! (ln)(ik)
                          if (k==m .and. j==l) hmatel = hmatel - h(n,i) ! (ln)
                          if (i==m .and. j==n) hmatel = hmatel - h(l,k) ! (ij)
                          if (i==l .and. j==n) hmatel = hmatel + h(m,k) ! (lm)(ij)
                          if (k==m .and. i==l) hmatel = hmatel + h(n,j) ! (ln)(ik)

                  end if
          end function aaa_oo_aaa

          pure function aaa_vv_aaa(i, j, k, a, b, c, l, m, n, d, e, f, h, nua) result(hmatel)
                  ! Expression:
                  ! A(ijk)A(bc)A(d/ef)A(a/bc) d(i,l)d(j,m)d(k,n)d(b,e)d(c,f) h(a,d)

                  integer, intent(in) :: nua
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: h(1:nua,1:nua)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  if (i==l .and. j==m .and. k==n) then

                        if (b==e .and. c==f) hmatel = hmatel + h(a,d) ! (1)
                        if (a==e .and. c==f) hmatel = hmatel - h(b,d) ! (ab)
                        if (b==d .and. c==f) hmatel = hmatel - h(a,e) ! (de)
                        if (a==d .and. c==f) hmatel = hmatel + h(b,e) ! (de)(ab)
                        if (b==e .and. a==d) hmatel = hmatel + h(c,f) ! (df)(ac)

                        if (a==e .and. b==f) hmatel = hmatel + h(c,d) ! (ab)
                        if (a==d .and. b==f) hmatel = hmatel - h(c,e) ! (de)(ab)
                        if (c==e .and. b==d) hmatel = hmatel + h(a,f) ! (df)
                        if (c==e .and. a==d) hmatel = hmatel - h(b,f) ! (df)(ac)
                  end if
          end function aaa_vv_aaa

          pure function aaa_oooo_aaa(i, j, k, a, b, c, l, m, n, d, e, f, h, noa) result(hmatel)
                  ! Expression:
                  ! A(abc)A(k/ij)A(n/lm) d(a,d)d(b,e)d(c,f)d(k,n) h(l,m,i,j)

                  integer, intent(in) :: noa
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: h(1:noa,1:noa,1:noa,1:noa)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  ! (1)
                  if (a==d .and. b==e .and. c==f) then
                      ! (1)
                      if (k==n) then 
                              hmatel = hmatel + h(l,m,i,j) ! (1)
                      elseif (k==l) then 
                              hmatel = hmatel - h(n,m,i,j) ! (ln)
                      elseif (k==m) then 
                              hmatel = hmatel - h(l,n,i,j) ! (mn)
                      end if
                      ! (ik)
                      if (i==n) then 
                              hmatel = hmatel - h(l,m,k,j) ! (1)
                      elseif (i==l) then 
                              hmatel = hmatel + h(n,m,k,j) ! (ln)
                      elseif (i==m) then 
                              hmatel = hmatel + h(l,n,k,j) ! (mn)
                      end if
                      ! (jk)
                      if (j==n) then 
                              hmatel = hmatel - h(l,m,i,k) ! (1)
                      elseif (j==l) then 
                              hmatel = hmatel + h(n,m,i,k) ! (ln)
                      elseif (j==m) then 
                              hmatel = hmatel + h(l,n,i,k) ! (mn)
                      end if
                  end if
          end function aaa_oooo_aaa

          pure function aaa_vvvv_aaa(i, j, k, a, b, c, l, m, n, d, e, f, h, nua) result(hmatel)
                  ! Expression:
                  ! A(ijk)A(c/ab)A(f/de) d(i,l)d(j,m)d(k,n)d(c,f) h(a,b,d,e)

                  integer, intent(in) :: nua
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: h(1:nua,1:nua,1:nua,1:nua)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  ! (1)
                  if (i==l .and. j==m .and. k==n) then
                      if (a==f) then 
                              hmatel = hmatel - h(c,b,d,e) ! (ac)
                      elseif (a==d) then 
                              hmatel = hmatel + h(c,b,f,e) ! (ac)
                      elseif (a==e) then 
                              hmatel = hmatel + h(c,b,d,f) ! (ac)
                      end if

                      if (b==f) then
                              hmatel = hmatel - h(a,c,d,e) ! (bc)
                      elseif (b==d) then 
                              hmatel = hmatel + h(a,c,f,e) ! (bc)
                      elseif (b==e) then 
                              hmatel = hmatel + h(a,c,d,f) ! (bc)
                      end if

                      if (c==f) then
                              hmatel = hmatel + h(a,b,d,e) ! (1)
                      elseif (c==d) then
                              hmatel = hmatel - h(a,b,f,e) ! (1)
                      elseif (c==e) then 
                              hmatel = hmatel - h(a,b,d,f) ! (1)
                      end if
                  end if
          end function aaa_vvvv_aaa

          pure function aaa_voov_aaa(i, j, k, a, b, c, l, m, n, d, e, f, h, noa, nua) result(hmatel)
              ! Expression:
              ! A(jk)A(bc)A(i/jk)A(a/bc)A(l/mn)A(d/ef) d(j,m)d(k,n)d(b,e)d(c,f) h(a,l,i,d)

              integer, intent(in) :: noa, nua
              integer, intent(in) :: i, j, k, a, b, c
              integer, intent(in) :: l, m, n, d, e, f
              real(kind=8), intent(in) :: h(1:nua,1:noa,1:noa,1:nua)

              real(kind=8) :: hmatel

              hmatel = 0.0d0

                  ! (1)
                  if (j==m .and. k==n .and. b==e .and. c==f) hmatel = hmatel + h(a,l,i,d) ! (1)
                  if (j==m .and. k==n .and. b==d .and. c==f) hmatel = hmatel - h(a,l,i,e) ! (de)
                  if (j==l .and. k==n .and. b==e .and. c==f) hmatel = hmatel - h(a,m,i,d) ! (lm)
                  if (j==l .and. k==n .and. b==d .and. c==f) hmatel = hmatel + h(a,m,i,e) ! (de)(lm)
                  if (j==m .and. k==n .and. a==e .and. c==f) hmatel = hmatel - h(b,l,i,d) ! (ab)
                  if (j==m .and. k==n .and. a==d .and. c==f) hmatel = hmatel + h(b,l,i,e) ! (de)(ab)
                  if (j==l .and. k==n .and. a==e .and. c==f) hmatel = hmatel + h(b,m,i,d) ! (lm)(ab)
                  if (j==l .and. k==n .and. a==d .and. c==f) hmatel = hmatel - h(b,m,i,e) ! (de)(lm)(ab)
                  if (j==m .and. k==n .and. b==e .and. a==d) hmatel = hmatel + h(c,l,i,f) ! (df)(ac)
                  if (j==l .and. k==n .and. b==e .and. a==d) hmatel = hmatel - h(c,m,i,f) ! (df)(lm)(ac)
                  if (i==m .and. k==n .and. b==e .and. c==f) hmatel = hmatel - h(a,l,j,d) ! (ij)
                  if (i==m .and. k==n .and. b==d .and. c==f) hmatel = hmatel + h(a,l,j,e) ! (de)(ij)
                  if (i==l .and. k==n .and. b==e .and. c==f) hmatel = hmatel + h(a,m,j,d) ! (lm)(ij)
                  if (i==l .and. k==n .and. b==d .and. c==f) hmatel = hmatel - h(a,m,j,e) ! (de)(lm)(ij)
                  if (i==m .and. k==n .and. a==e .and. c==f) hmatel = hmatel + h(b,l,j,d) ! (ab)(ij)
                  if (i==m .and. k==n .and. a==d .and. c==f) hmatel = hmatel - h(b,l,j,e) ! (de)(ab)(ij)
                  if (i==l .and. k==n .and. a==e .and. c==f) hmatel = hmatel - h(b,m,j,d) ! (lm)(ab)(ij)
                  if (i==l .and. k==n .and. a==d .and. c==f) hmatel = hmatel + h(b,m,j,e) ! (de)(lm)(ab)(ij)
                  if (i==m .and. k==n .and. b==e .and. a==d) hmatel = hmatel - h(c,l,j,f) ! (df)(ac)(ij)
                  if (i==l .and. k==n .and. b==e .and. a==d) hmatel = hmatel + h(c,m,j,f) ! (df)(lm)(ac)(ij)
                  if (j==m .and. i==l .and. b==e .and. c==f) hmatel = hmatel + h(a,n,k,d) ! (ln)(ik)
                  if (j==m .and. i==l .and. b==d .and. c==f) hmatel = hmatel - h(a,n,k,e) ! (de)(ln)(ik)
                  if (j==m .and. i==l .and. a==e .and. c==f) hmatel = hmatel - h(b,n,k,d) ! (ln)(ab)(ik)
                  if (j==m .and. i==l .and. a==d .and. c==f) hmatel = hmatel + h(b,n,k,e) ! (de)(ln)(ab)(ik)
                  if (j==m .and. i==l .and. b==e .and. a==d) hmatel = hmatel + h(c,n,k,f) ! (df)(ln)(ac)(ik)
                  ! (bc)
                  if (j==m .and. k==n .and. c==e .and. b==d) hmatel = hmatel + h(a,l,i,f) ! (df)
                  if (j==l .and. k==n .and. c==e .and. b==d) hmatel = hmatel - h(a,m,i,f) ! (df)(lm)
                  if (j==m .and. k==n .and. a==e .and. b==f) hmatel = hmatel + h(c,l,i,d) ! (ab)
                  if (j==m .and. k==n .and. a==d .and. b==f) hmatel = hmatel - h(c,l,i,e) ! (de)(ab)
                  if (j==l .and. k==n .and. a==e .and. b==f) hmatel = hmatel - h(c,m,i,d) ! (lm)(ab)
                  if (j==l .and. k==n .and. a==d .and. b==f) hmatel = hmatel + h(c,m,i,e) ! (de)(lm)(ab)
                  if (j==m .and. k==n .and. c==e .and. a==d) hmatel = hmatel - h(b,l,i,f) ! (df)(ac)
                  if (j==l .and. k==n .and. c==e .and. a==d) hmatel = hmatel + h(b,m,i,f) ! (df)(lm)(ac)
                  if (i==m .and. k==n .and. c==e .and. b==d) hmatel = hmatel - h(a,l,j,f) ! (df)(ij)
                  if (i==l .and. k==n .and. c==e .and. b==d) hmatel = hmatel + h(a,m,j,f) ! (df)(lm)(ij)
                  if (i==m .and. k==n .and. a==e .and. b==f) hmatel = hmatel - h(c,l,j,d) ! (ab)(ij)
                  if (i==m .and. k==n .and. a==d .and. b==f) hmatel = hmatel + h(c,l,j,e) ! (de)(ab)(ij)
                  if (i==l .and. k==n .and. a==e .and. b==f) hmatel = hmatel + h(c,m,j,d) ! (lm)(ab)(ij)
                  if (i==l .and. k==n .and. a==d .and. b==f) hmatel = hmatel - h(c,m,j,e) ! (de)(lm)(ab)(ij)
                  if (i==m .and. k==n .and. c==e .and. a==d) hmatel = hmatel + h(b,l,j,f) ! (df)(ac)(ij)
                  if (i==l .and. k==n .and. c==e .and. a==d) hmatel = hmatel - h(b,m,j,f) ! (df)(lm)(ac)(ij)
                  if (j==m .and. i==l .and. c==e .and. b==d) hmatel = hmatel + h(a,n,k,f) ! (df)(ln)(ik)
                  if (j==m .and. i==l .and. a==e .and. b==f) hmatel = hmatel + h(c,n,k,d) ! (ln)(ab)(ik)
                  if (j==m .and. i==l .and. a==d .and. b==f) hmatel = hmatel - h(c,n,k,e) ! (de)(ln)(ab)(ik)
                  if (j==m .and. i==l .and. c==e .and. a==d) hmatel = hmatel - h(b,n,k,f) ! (df)(ln)(ac)(ik)
                  ! (jk)
                  if (k==m .and. j==l .and. b==e .and. c==f) hmatel = hmatel + h(a,n,i,d) ! (ln)
                  if (k==m .and. j==l .and. b==d .and. c==f) hmatel = hmatel - h(a,n,i,e) ! (de)(ln)
                  if (k==m .and. j==l .and. a==e .and. c==f) hmatel = hmatel - h(b,n,i,d) ! (ln)(ab)
                  if (k==m .and. j==l .and. a==d .and. c==f) hmatel = hmatel + h(b,n,i,e) ! (de)(ln)(ab)
                  if (k==m .and. j==l .and. b==e .and. a==d) hmatel = hmatel + h(c,n,i,f) ! (df)(ln)(ac)
                  if (i==m .and. j==n .and. b==e .and. c==f) hmatel = hmatel + h(a,l,k,d) ! (ij)
                  if (i==m .and. j==n .and. b==d .and. c==f) hmatel = hmatel - h(a,l,k,e) ! (de)(ij)
                  if (i==l .and. j==n .and. b==e .and. c==f) hmatel = hmatel - h(a,m,k,d) ! (lm)(ij)
                  if (i==l .and. j==n .and. b==d .and. c==f) hmatel = hmatel + h(a,m,k,e) ! (de)(lm)(ij)
                  if (i==m .and. j==n .and. a==e .and. c==f) hmatel = hmatel - h(b,l,k,d) ! (ab)(ij)
                  if (i==m .and. j==n .and. a==d .and. c==f) hmatel = hmatel + h(b,l,k,e) ! (de)(ab)(ij)
                  if (i==l .and. j==n .and. a==e .and. c==f) hmatel = hmatel + h(b,m,k,d) ! (lm)(ab)(ij)
                  if (i==l .and. j==n .and. a==d .and. c==f) hmatel = hmatel - h(b,m,k,e) ! (de)(lm)(ab)(ij)
                  if (i==m .and. j==n .and. b==e .and. a==d) hmatel = hmatel + h(c,l,k,f) ! (df)(ac)(ij)
                  if (i==l .and. j==n .and. b==e .and. a==d) hmatel = hmatel - h(c,m,k,f) ! (df)(lm)(ac)(ij)
                  if (k==m .and. i==l .and. b==e .and. c==f) hmatel = hmatel - h(a,n,j,d) ! (ln)(ik)
                  if (k==m .and. i==l .and. b==d .and. c==f) hmatel = hmatel + h(a,n,j,e) ! (de)(ln)(ik)
                  if (k==m .and. i==l .and. a==e .and. c==f) hmatel = hmatel + h(b,n,j,d) ! (ln)(ab)(ik)
                  if (k==m .and. i==l .and. a==d .and. c==f) hmatel = hmatel - h(b,n,j,e) ! (de)(ln)(ab)(ik)
                  if (k==m .and. i==l .and. b==e .and. a==d) hmatel = hmatel - h(c,n,j,f) ! (df)(ln)(ac)(ik)
                  ! (jk)(bc), apply(jk)
                  if (k==m .and. j==l .and. c==e .and. b==d) hmatel = hmatel + h(a,n,i,f) ! (df)(ln)
                  if (k==m .and. j==l .and. a==e .and. b==f) hmatel = hmatel + h(c,n,i,d) ! (ln)(ab)
                  if (k==m .and. j==l .and. a==d .and. b==f) hmatel = hmatel - h(c,n,i,e) ! (de)(ln)(ab)
                  if (k==m .and. j==l .and. c==e .and. a==d) hmatel = hmatel - h(b,n,i,f) ! (df)(ln)(ac)
                  if (i==m .and. j==n .and. c==e .and. b==d) hmatel = hmatel + h(a,l,k,f) ! (df)(ij)
                  if (i==l .and. j==n .and. c==e .and. b==d) hmatel = hmatel - h(a,m,k,f) ! (df)(lm)(ij)
                  if (i==m .and. j==n .and. a==e .and. b==f) hmatel = hmatel + h(c,l,k,d) ! (ab)(ij)
                  if (i==m .and. j==n .and. a==d .and. b==f) hmatel = hmatel - h(c,l,k,e) ! (de)(ab)(ij)
                  if (i==l .and. j==n .and. a==e .and. b==f) hmatel = hmatel - h(c,m,k,d) ! (lm)(ab)(ij)
                  if (i==l .and. j==n .and. a==d .and. b==f) hmatel = hmatel + h(c,m,k,e) ! (de)(lm)(ab)(ij)
                  if (i==m .and. j==n .and. c==e .and. a==d) hmatel = hmatel - h(b,l,k,f) ! (df)(ac)(ij)
                  if (i==l .and. j==n .and. c==e .and. a==d) hmatel = hmatel + h(b,m,k,f) ! (df)(lm)(ac)(ij)
                  if (k==m .and. i==l .and. c==e .and. b==d) hmatel = hmatel - h(a,n,j,f) ! (df)(ln)(ik)
                  if (k==m .and. i==l .and. a==e .and. b==f) hmatel = hmatel - h(c,n,j,d) ! (ln)(ab)(ik)
                  if (k==m .and. i==l .and. a==d .and. b==f) hmatel = hmatel + h(c,n,j,e) ! (de)(ln)(ab)(ik)
                  if (k==m .and. i==l .and. c==e .and. a==d) hmatel = hmatel + h(b,n,j,f) ! (df)(ln)(ac)(ik)
          end function aaa_voov_aaa

          pure function aaa_voov_aab(i, j, k, a, b, c, l, m, n, d, e, f, h, noa, nua, nob, nub) result(hmatel)
              ! Expression:
              ! A(ij)A(ab)A(k/ij)A(c/ab) d(j,m)d(i,l)d(a,d)d(b,e) h(c,n,k,f)

              integer, intent(in) :: noa, nua, nob, nub
              integer, intent(in) :: i, j, k, a, b, c
              integer, intent(in) :: l, m, n, d, e, f
              real(kind=8), intent(in) :: h(1:nua,1:nob,1:noa,1:nub)

              real(kind=8) :: hmatel

              hmatel = 0.0d0

              if (j==m .and. i==l .and. a==d .and. b==e) hmatel = hmatel + h(c,n,k,f) ! (1)
              if (k==m .and. i==l .and. a==d .and. b==e) hmatel = hmatel - h(c,n,j,f) ! (jk)
              if (j==m .and. i==l .and. a==d .and. c==e) hmatel = hmatel - h(b,n,k,f) ! (bc)
              if (k==m .and. i==l .and. a==d .and. c==e) hmatel = hmatel + h(b,n,j,f) ! (jk)(bc)
              if (k==m .and. j==l .and. a==d .and. b==e) hmatel = hmatel + h(c,n,i,f) ! (jk)(ij)
              if (k==m .and. j==l .and. a==d .and. c==e) hmatel = hmatel - h(b,n,i,f) ! (jk)(bc)(ij)
              if (j==m .and. i==l .and. b==d .and. c==e) hmatel = hmatel + h(a,n,k,f) ! (bc)(ab)
              if (k==m .and. i==l .and. b==d .and. c==e) hmatel = hmatel - h(a,n,j,f) ! (jk)(bc)(ab)
              if (k==m .and. j==l .and. b==d .and. c==e) hmatel = hmatel + h(a,n,i,f) ! (jk)(bc)(ab)(ij)

          end function aaa_voov_aab

          pure function aab_oo_aab(i, j, k, a, b, c, l, m, n, d, e, f, ha, hb, noa, nob) result(hmatel)
                  ! Expression:
                  ! -A(ab)A(ij)A(lm) d(a,d)d(b,e)d(c,f)d(j,m)d(k,n) ha(l,i)
                  ! -A(ij)A(ab) d(a,d)d(b,e)d(c,f)d(i,l)d(j,m) hb(n,k)

                  integer, intent(in) :: noa, nob
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: ha(1:noa,1:noa), hb(1:nob,1:nob)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  ! (1)
                  if (a==d .and. b==e .and. c==f) then
                    !if (k==n) then
                    !        if (j==m) then
                    !                hmatel = hmatel - ha(l,i) ! (1)
                    !        elseif (j==l) then 
                    !                hmatel = hmatel + ha(m,i) ! (lm)
                    !        end if

                    !        if (i==m) then
                    !                hmatel = hmatel + ha(l,j) ! (ij)
                    !        elseif (i==l) then 
                    !                hmatel = hmatel - ha(m,j) ! (ij)(lm)
                    !        end if
                    !end if
                    if (i==l .and. j==m) hmatel = hmatel - hb(n,k) ! (1)
                  end if
          end function aab_oo_aab

          pure function aab_vv_aab(i, j, k, a, b, c, l, m, n, d, e, f, ha, hb, nua, nub) result(hmatel)
                  ! Expression:
                  ! A(ij)A(ab)A(ml)A(de) d(i,l)d(j,m)d(b,e)d(k,n)d(c,f) ha(a,d)
                  ! A(ij)A(ab) d(i,l)d(j,m)d(a,d)d(b,e)d(k,n) hb(c,f)

                  integer, intent(in) :: nua, nub
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: ha(1:nua,1:nua), hb(1:nub,1:nub)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  ! (1)
                  if (i==l .and. j==m .and. k==n) then
                     if (c==f) then
                             if (a==e) then 
                                     hmatel = hmatel - ha(b,d) ! (ab)
                             elseif (a==d) then
                                     hmatel = hmatel + ha(b,e) ! (ab)(de)
                             end if

                             if (b==e) then 
                                     hmatel = hmatel + ha(a,d) ! (1)
                             elseif (b==d) then
                                     hmatel = hmatel - ha(a,e) ! (de)
                             end if
                     end if
                     if (a==d .and. b==e) hmatel = hmatel + hb(c,f) ! (1)
                  end if
          end function aab_vv_aab

          pure function aab_vvvv_aab(i, j, k, a, b, c, l, m, n, d, e, f, ha, hb, nua, nub) result(hmatel)
                  ! Expression:
                  ! A(ij) d(i,l)d(j,m)d(k,n)d(c,f) ha(a,b,d,e) 
                  ! A(ij)A(ab)A(ed) d(i,l)d(j,m)d(k,n)d(a,d) hb(b,c,e,f)

                  integer, intent(in) :: nua, nub
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: ha(1:nua,1:nua,1:nua,1:nua)
                  real(kind=8), intent(in) :: hb(1:nua,1:nub,1:nua,1:nub)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  ! (1)
                  if (i==l .and. j==m .and. k==n) then
                     if (c==f) hmatel = hmatel + ha(a,b,d,e) ! (1)

                     if (a==d) then
                             hmatel = hmatel + hb(b,c,e,f) ! (1)
                     elseif (a==e) then
                             hmatel = hmatel - hb(b,c,d,f) ! (ed)
                     endif 

                     if (b==d) then 
                             hmatel = hmatel - hb(a,c,e,f) ! (ab)
                     elseif (b==e) then 
                             hmatel = hmatel + hb(a,c,d,f) ! (ab)(ed)
                     end if
                  end if

          end function aab_vvvv_aab

          pure function aab_oooo_aab(i, j, k, a, b, c, l, m, n, d, e, f, ha, hb, noa, nob) result(hmatel)
                  ! Expression:
                  ! A(ab) d(a,d)d(b,e)d(c,f)d(k,n) ha(l,m,i,j)
                  ! A(ab)A(ij)A(ml) d(i,l)d(a,d)d(b,e)d(c,f) hb(m,n,j,k)

                  integer, intent(in) :: noa, nob
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: ha(1:noa,1:noa,1:noa,1:noa)
                  real(kind=8), intent(in) :: hb(1:noa,1:nob,1:noa,1:nob)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  ! (1)
                  if (a==d .and. b==e .and. c==f) then
                     if (k==n) hmatel = hmatel + ha(l,m,i,j) ! (1)

                     if (i==l) then 
                             hmatel = hmatel + hb(m,n,j,k) ! (1)
                     elseif (i==m) then
                             hmatel = hmatel - hb(l,n,j,k) ! (lm)
                     end if
                     
                     if (j==l) then 
                             hmatel = hmatel - hb(m,n,i,k) ! (ij)
                     elseif (j==m) then 
                             hmatel = hmatel + hb(l,n,i,k) ! (ij)(lm)
                     end if
                  end if

          end function aab_oooo_aab

          pure function aab_ovov_aab(i, j, k, a, b, c, l, m, n, d, e, f, h, noa, nub) result(hmatel)
                  ! Expression:
                  ! -A(ab)A(ij)A(lm) d(a,d)d(b,e)d(k,n)d(j,m) h(l,c,i,f)

                  integer, intent(in) :: noa, nub
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: h(1:noa,1:nub,1:noa,1:nub)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  ! (1)
                  if (a==d .and. b==e .and. k==n) then
                     if(j==m) then 
                             hmatel = hmatel - h(l,c,i,f) ! (1)
                     elseif (j==l) then 
                             hmatel = hmatel + h(m,c,i,f) ! (lm)
                     end if

                     if (i==m) then 
                             hmatel = hmatel + h(l,c,j,f) ! (ij)
                     elseif (i==l) then 
                             hmatel = hmatel - h(m,c,j,f) ! (ij)(lm)
                     end if
                  end if
          end function aab_ovov_aab

          pure function aab_vovo_aab(i, j, k, a, b, c, l, m, n, d, e, f, h, nua, nob) result(hmatel)
                  ! Expression:
                  ! -A(ij)A(ab)A(de) d(i,l)d(j,m)d(c,f)d(b,e) h(a,n,d,k)

                  integer, intent(in) :: nua, nob
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: h(1:nua,1:nob,1:nua,1:nob)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  ! (1)
                  if (i==l .and. j==m .and. c==f) then
                     if (b==e) then 
                             hmatel = hmatel - h(a,n,d,k) ! (1)
                     elseif (b==d) then 
                             hmatel = hmatel + h(a,n,e,k) ! (de)
                     end if

                     if (a==e) then 
                             hmatel = hmatel + h(b,n,d,k) ! (ab)
                     elseif (a==d) then 
                             hmatel = hmatel - h(b,n,e,k) ! (ab)(de)
                     end if
                  end if
          end function aab_vovo_aab

          pure function aab_voov_aab(i, j, k, a, b, c, l, m, n, d, e, f, ha, hc, noa, nua, nob, nub) result(hmatel)
                  ! Expression:
                  ! A(ij)A(ab)A(lm)A(de) d(j,m)d(b,e)d(k,n)d(c,f) ha(a,l,i,d)
                  ! A(ij)A(ab) d(i,l)d(j,m)d(b,e)d(a,d) hc(c,n,k,f)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: ha(1:nua,1:noa,1:noa,1:nua)
                  real(kind=8), intent(in) :: hc(1:nub,1:nob,1:nob,1:nub)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  if (k==n .and. c==f) then
                          ! (1)
                          if (j==m) then ! (1)
                                  if (b==e) then ! (1)
                                          hmatel = hmatel + ha(a,l,i,d)
                                  elseif (b==d) then ! (de)
                                          hmatel = hmatel - ha(a,l,i,e)
                                  end if
                          elseif (j==l) then ! (lm)
                                  if (b==e) then ! (1)
                                          hmatel = hmatel - ha(a,m,i,d)
                                  elseif (b==d) then ! (de)
                                          hmatel = hmatel + ha(a,m,i,e)
                                  end if
                          end if
                          ! (ij)
                          if (i==m) then ! (1)
                                  if (b==e) then ! (1)
                                          hmatel = hmatel - ha(a,l,j,d)
                                  elseif (b==d) then ! (de)
                                          hmatel = hmatel + ha(a,l,j,e)
                                  end if
                          elseif (i==l) then ! (lm)
                                  if (b==e) then ! (1)
                                          hmatel = hmatel + ha(a,m,j,d)
                                  elseif (b==d) then ! (de)
                                          hmatel = hmatel - ha(a,m,j,e)
                                  end if
                          end if
                          ! (ab)
                          if (j==m) then ! (1)
                                  if (a==e) then ! (1)
                                          hmatel = hmatel - ha(b,l,i,d)
                                  elseif (a==d) then ! (de)
                                          hmatel = hmatel + ha(b,l,i,e)
                                  end if
                          elseif (j==l) then ! (lm)
                                  if (a==e) then ! (1)
                                          hmatel = hmatel + ha(b,m,i,d)
                                  elseif (a==d) then ! (de)
                                          hmatel = hmatel - ha(b,m,i,e)
                                  end if
                          end if
                          ! (ij)(ab)
                          if (i==m) then ! (1)
                                  if (a==e) then ! (1)
                                          hmatel = hmatel + ha(b,l,j,d)
                                  elseif (a==d) then ! (de)
                                          hmatel = hmatel - ha(b,l,j,e)
                                  end if
                          elseif (i==l) then ! (lm)
                                  if (a==e) then ! (1)
                                          hmatel = hmatel - ha(b,m,j,d)
                                  elseif (a==d) then ! (de)
                                          hmatel = hmatel + ha(b,m,j,e)
                                  end if
                          end if
                  end if
                  ! (1)
                  if (i==l .and. j==m .and. b==e .and. a==d) hmatel = hmatel + hc(c,n,k,f) ! (1)
          end function aab_voov_aab

          pure function aab_voov_abb(i, j, k, a, b, c, l, m, n, d, e, f, h, noa, nua, nob, nub) result(hmatel)
                  ! Expression:
                  ! A(ij)A(ab)A(mn)A(ef) d(i,l)d(k,n)d(a,d)d(c,f) h(b,m,j,e)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: h(1:nua,1:nob,1:noa,1:nub)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  if (a==d) then
                     if (c==f) then
                          if (i==l) then
                             if (k==n) hmatel = hmatel + h(b,m,j,e) 
                             if (k==m) hmatel = hmatel - h(b,n,j,e)
                          elseif (j==l) then
                             if (k==n) hmatel = hmatel - h(b,m,i,e)
                             if (k==m) hmatel = hmatel + h(b,n,i,e) 
                          end if
                     elseif (c==e) then
                          if (i==l) then
                             if (k==n) hmatel = hmatel - h(b,m,j,f) 
                             if (k==m) hmatel = hmatel + h(b,n,j,f) 
                          elseif (j==l) then
                             if (k==n) hmatel = hmatel + h(b,m,i,f) 
                             if (k==m) hmatel = hmatel - h(b,n,i,f) 
                          end if
                     end if
                  elseif (b==d) then
                     if (c==f) then
                          if (i==l) then
                             if (k==n) hmatel = hmatel - h(a,m,j,e) 
                             if (k==m) hmatel = hmatel + h(a,n,j,e) 
                          elseif (j==l) then
                             if (k==n) hmatel = hmatel + h(a,m,i,e) 
                             if (k==m) hmatel = hmatel - h(a,n,i,e)
                          end if 
                     elseif (c==e) then
                          if (i==l) then
                             if (k==n) hmatel = hmatel + h(a,m,j,f) 
                             if (k==m) hmatel = hmatel - h(a,n,j,f) 
                          elseif (j==l) then
                             if (k==n) hmatel = hmatel - h(a,m,i,f) 
                             if (k==m) hmatel = hmatel + h(a,n,i,f)
                          end if
                     end if        
                  end if
          end function aab_voov_abb

          pure function aab_ovvo_aaa(i, j, k, a, b, c, l, m, n, d, e, f, h, noa, nua, nob, nub) result(hmatel)
                  ! Expression:
                  ! A(ij)A(ab)A(n/lm)A(f/ed) d(i,l)d(j,m)d(a,d)d(b,e) h(n,c,f,k)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: h(1:noa,1:nub,1:nua,1:nob)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  ! (1)
                  if (i==l .and. j==m .and. a==d .and. b==e) hmatel = hmatel + h(n,c,f,k) ! (1)
                  if (i==l .and. j==n .and. a==d .and. b==e) hmatel = hmatel - h(m,c,f,k) ! (nm)
                  if (i==l .and. j==m .and. a==d .and. b==f) hmatel = hmatel - h(n,c,e,k) ! (fe)
                  if (i==l .and. j==n .and. a==d .and. b==f) hmatel = hmatel + h(m,c,e,k) ! (nm)(fe)
                  if (j==n .and. i==m .and. a==d .and. b==e) hmatel = hmatel + h(l,c,f,k) ! (nl)
                  if (j==n .and. i==m .and. a==d .and. b==f) hmatel = hmatel - h(l,c,e,k) ! (nl)(fe)
                  if (i==l .and. j==m .and. b==f .and. a==e) hmatel = hmatel + h(n,c,d,k) ! (fd)
                  if (i==l .and. j==n .and. b==f .and. a==e) hmatel = hmatel - h(m,c,d,k) ! (nm)(fd)
                  if (j==n .and. i==m .and. b==f .and. a==e) hmatel = hmatel + h(l,c,d,k) ! (nl)(fd)
          end function aab_ovvo_aaa

          pure function abb_ovvo_aab(i, j, k, a, b, c, l, m, n, d, e, f, h, noa, nua, nob, nub) result(hmatel)
                  ! Expression:
                  ! A(jk)A(bc)A(lm)A(ed) d(i,l)d(a,d)d(k,n)d(c,f) h(m,b,e,j)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: h(1:noa,1:nub,1:nua,1:nob)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  if (i==l) then ! (1)
                     if (a==d) then ! (1)
                        if (k==n) then ! (1)
                           if (c==f) then ! (1)
                                   hmatel = hmatel + h(m,b,e,j) 
                           elseif (b==f) then ! (bc)
                                   hmatel = hmatel - h(m,c,e,j) 
                           end if
                        elseif (j==n) then ! (jk)
                           if (c==f) then ! (1)
                                   hmatel = hmatel - h(m,b,e,k) 
                           elseif (b==f) then ! (bc)
                                   hmatel = hmatel + h(m,c,e,k) 
                           end if
                        end if 
                     elseif (a==e) then ! (de)
                        if (k==n) then ! (1)
                           if (c==f) then ! (1)
                                   hmatel = hmatel - h(m,b,d,j) 
                           elseif (b==f) then ! (bc)
                                   hmatel = hmatel + h(m,c,d,j) 
                           end if
                        elseif (j==n) then ! (jk)
                           if (c==f) then ! (1)
                                   hmatel = hmatel + h(m,b,d,k) 
                           elseif (b==f) then ! (bc)
                                   hmatel = hmatel - h(m,c,d,k) 
                           end if
                        end if
                     end if
                  elseif (i==m) then ! (lm)
                     if (a==d) then ! (1)
                        if (k==n) then ! (1)
                           if (c==f) then ! (1)
                                   hmatel = hmatel - h(l,b,e,j) 
                           elseif (b==f) then ! (bc)
                                   hmatel = hmatel + h(l,c,e,j) 
                           end if
                        elseif (j==n) then ! (jk)
                           if (c==f) then ! (1)
                                   hmatel = hmatel + h(l,b,e,k) 
                           elseif (b==f) then ! (bc)
                                   hmatel = hmatel - h(l,c,e,k) 
                           end if
                        end if 
                     elseif (a==e) then ! (de)
                        if (k==n) then ! (1)
                           if (c==f) then ! (1)
                                   hmatel = hmatel + h(l,b,d,j) 
                           elseif (b==f) then ! (bc)
                                   hmatel = hmatel - h(l,c,d,j) 
                           end if
                        elseif (j==n) then ! (jk)
                           if (c==f) then ! (1)
                                   hmatel = hmatel - h(l,b,d,k) 
                           elseif (b==f) then ! (bc)
                                   hmatel = hmatel + h(l,c,d,k) 
                           end if
                        end if
                     end if
                  end if
          end function abb_ovvo_aab

          pure function abb_oo_abb(i, j, k, a, b, c, l, m, n, d, e, f, ha, hb, noa, nob) result(hmatel)
                  ! Expression:
                  ! Found by spin-inverting the code for aab_oo_aab

                  integer, intent(in) :: noa, nob
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: ha(1:noa,1:noa), hb(1:nob,1:nob)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  ! (1)
                  if (c==f .and. b==e .and. a==d) then
                    if (i==l) then
                            if (j==m) then
                                    hmatel = hmatel - hb(n,k) ! (1)
                            elseif (j==n) then 
                                    hmatel = hmatel + hb(m,k) ! (lm)
                            end if

                            if (k==m) then
                                    hmatel = hmatel + hb(n,j) ! (ij)
                            elseif (k==n) then 
                                    hmatel = hmatel - hb(m,j) ! (ij)(lm)
                            end if
                    end if
                    if (k==n .and. j==m) hmatel = hmatel - ha(l,i) ! (1)
                  end if
          end function abb_oo_abb

          pure function abb_vv_abb(i, j, k, a, b, c, l, m, n, d, e, f, ha, hb, nua, nub) result(hmatel)
                  ! Expression:
                  ! Found by spin-inverting the code for aab_vv_aab

                  integer, intent(in) :: nua, nub
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: ha(1:nua,1:nua), hb(1:nub,1:nub)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  ! (1)
                  if (i==l .and. j==m .and. k==n) then
                     if (a==d) then
                             if (c==e) then 
                                     hmatel = hmatel - hb(b,f) ! (ab)
                             elseif (c==f) then
                                     hmatel = hmatel + hb(b,e) ! (ab)(de)
                             end if

                             if (b==e) then 
                                     hmatel = hmatel + hb(c,f) ! (1)
                             elseif (b==f) then
                                     hmatel = hmatel - hb(c,e) ! (de)
                             end if
                     end if
                     if (c==f .and. b==e) hmatel = hmatel + ha(a,d) ! (1)
                  end if
          end function abb_vv_abb

          pure function abb_vvvv_abb(i, j, k, a, b, c, l, m, n, d, e, f, hc, hb, nua, nub) result(hmatel)
                  ! Expression:
                  ! Found by spin-inverting the code for aab_vvvv_aab

                  integer, intent(in) :: nua, nub
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: hc(1:nub,1:nub,1:nub,1:nub)
                  real(kind=8), intent(in) :: hb(1:nua,1:nub,1:nua,1:nub)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  ! (1)
                  if (i==l .and. j==m .and. k==n) then
                     if (a==d) hmatel = hmatel + hc(c,b,f,e) ! (1)

                     if (c==f) then
                             hmatel = hmatel + hb(a,b,d,e) ! (1)
                     elseif (c==e) then
                             hmatel = hmatel - hb(a,b,d,f) ! (ed)
                     endif 

                     if (b==f) then 
                             hmatel = hmatel - hb(a,c,d,e) ! (ab)
                     elseif (b==e) then 
                             hmatel = hmatel + hb(a,c,d,f) ! (ab)(ed)
                     end if
                  end if

          end function abb_vvvv_abb

          pure function abb_oooo_abb(i, j, k, a, b, c, l, m, n, d, e, f, hc, hb, noa, nob) result(hmatel)
                  ! Expression:
                  ! Found by spin-inverting the code for aab_oooo_aab

                  integer, intent(in) :: noa, nob
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: hc(1:nob,1:nob,1:nob,1:nob)
                  real(kind=8), intent(in) :: hb(1:noa,1:nob,1:noa,1:nob)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  ! (1)
                  if (a==d .and. b==e .and. c==f) then
                     if (i==l) hmatel = hmatel + hc(n,m,k,j) ! (1)

                     if (k==n) then 
                             hmatel = hmatel + hb(l,m,i,j) ! (1)
                     elseif (k==m) then
                             hmatel = hmatel - hb(l,n,i,j) ! (lm)
                     end if
                     
                     if (j==n) then 
                             hmatel = hmatel - hb(l,m,i,k) ! (ij)
                     elseif (j==m) then 
                             hmatel = hmatel + hb(l,n,i,k) ! (ij)(lm)
                     end if
                  end if

          end function abb_oooo_abb

          pure function abb_vovo_abb(i, j, k, a, b, c, l, m, n, d, e, f, h, nua, nob) result(hmatel)
                  ! Expression:
                  ! Found by spin-inverting the code for aab_ovov_aab (this
                  ! turns h2b_ovov to h2b_vovo)

                  integer, intent(in) :: nua, nob
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: h(1:nua,1:nob,1:nua,1:nob)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  ! (1)
                  if (c==f .and. b==e .and. i==l) then
                     if(j==m) then 
                             hmatel = hmatel - h(a,n,d,k) ! (1)
                     elseif (j==n) then 
                             hmatel = hmatel + h(a,m,d,k) ! (lm)
                     end if

                     if (k==m) then 
                             hmatel = hmatel + h(a,n,d,j) ! (ij)
                     elseif (k==n) then 
                             hmatel = hmatel - h(a,m,d,j) ! (ij)(lm)
                     end if
                  end if
          end function abb_vovo_abb

          pure function abb_ovov_abb(i, j, k, a, b, c, l, m, n, d, e, f, h, noa, nub) result(hmatel)
                  ! Expression:
                  ! Found by spin-inverting the code for aab_vovo_aab (this
                  ! turns h2b_vovo to h2b_ovov)

                  integer, intent(in) :: noa, nub
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: h(1:noa,1:nub,1:noa,1:nub)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  ! (1)
                  if (k==n .and. j==m .and. a==d) then
                     if (b==e) then 
                             hmatel = hmatel - h(l,c,i,f) ! (1)
                     elseif (b==f) then 
                             hmatel = hmatel + h(l,c,i,e) ! (de)
                     end if

                     if (c==e) then 
                             hmatel = hmatel + h(l,b,i,f) ! (ab)
                     elseif (c==f) then 
                             hmatel = hmatel - h(l,b,i,e) ! (ab)(de)
                     end if
                  end if
          end function abb_ovov_abb

          pure function abb_voov_abb(i, j, k, a, b, c, l, m, n, d, e, f, ha, hc, noa, nua, nob, nub) result(hmatel)
                  ! Expression:
                  ! Found by spin-inverting the code for aab_voov_aab

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: ha(1:nua,1:noa,1:noa,1:nua)
                  real(kind=8), intent(in) :: hc(1:nub,1:nob,1:nob,1:nub)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  if (i==l .and. a==d) then
                          ! (1)
                          if (j==m) then ! (1)
                                  if (b==e) then ! (1)
                                          hmatel = hmatel + hc(c,n,k,f)
                                  elseif (b==f) then ! (de)
                                          hmatel = hmatel - hc(c,n,k,e)
                                  end if
                          elseif (j==n) then ! (lm)
                                  if (b==e) then ! (1)
                                          hmatel = hmatel - hc(c,m,k,f)
                                  elseif (b==f) then ! (de)
                                          hmatel = hmatel + hc(c,m,k,e)
                                  end if
                          end if
                          ! (ij)
                          if (k==m) then ! (1)
                                  if (b==e) then ! (1)
                                          hmatel = hmatel - hc(c,n,j,f)
                                  elseif (b==f) then ! (de)
                                          hmatel = hmatel + hc(c,n,j,e)
                                  end if
                          elseif (k==n) then ! (lm)
                                  if (b==e) then ! (1)
                                          hmatel = hmatel + hc(c,m,j,f)
                                  elseif (b==f) then ! (de)
                                          hmatel = hmatel - hc(c,m,j,e)
                                  end if
                          end if
                          ! (ab)
                          if (j==m) then ! (1)
                                  if (c==e) then ! (1)
                                          hmatel = hmatel - hc(b,n,k,f)
                                  elseif (c==f) then ! (de)
                                          hmatel = hmatel + hc(b,n,k,e)
                                  end if
                          elseif (j==n) then ! (lm)
                                  if (c==e) then ! (1)
                                          hmatel = hmatel + hc(b,m,k,f)
                                  elseif (c==f) then ! (de)
                                          hmatel = hmatel - hc(b,m,k,e)
                                  end if
                          end if
                          ! (ij)(ab)
                          if (k==m) then ! (1)
                                  if (c==e) then ! (1)
                                          hmatel = hmatel + hc(b,n,j,f)
                                  elseif (c==f) then ! (de)
                                          hmatel = hmatel - hc(b,n,j,e)
                                  end if
                          elseif (k==n) then ! (lm)
                                  if (c==e) then ! (1)
                                          hmatel = hmatel - hc(b,m,j,f)
                                  elseif (c==f) then ! (de)
                                          hmatel = hmatel + hc(b,m,j,e)
                                  end if
                          end if
                  end if
                  ! (1)
                  if (k==n .and. j==m .and. b==e .and. c==f) hmatel = hmatel + ha(a,l,i,d) ! (1)
          end function abb_voov_abb

          pure function abb_voov_bbb(i, j, k, a, b, c, l, m, n, d, e, f, h, noa, nua, nob, nub) result(hmatel)
                  ! Expression:
                  ! A(jk)A(bc)A(l/mn)A(d/ef) d(j,m)d(k,n)d(b,e)d(c,f) h(a,l,i,d)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: h(1:nua,1:nob,1:noa,1:nub)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  if (j==m .and. k==n .and. b==e .and. c==f) hmatel = hmatel + h(a,l,i,d) ! (1)
                  if (j==l .and. k==n .and. b==e .and. c==f) hmatel = hmatel - h(a,m,i,d) ! (lm)
                  if (j==m .and. k==n .and. b==d .and. c==f) hmatel = hmatel - h(a,l,i,e) ! (de)
                  if (j==l .and. k==n .and. b==d .and. c==f) hmatel = hmatel + h(a,m,i,e) ! (lm)(de)
                  if (k==m .and. j==l .and. b==e .and. c==f) hmatel = hmatel + h(a,n,i,d) ! (ln)
                  if (k==m .and. j==l .and. b==d .and. c==f) hmatel = hmatel - h(a,n,i,e) ! (ln)(de)
                  if (j==m .and. k==n .and. c==e .and. b==d) hmatel = hmatel + h(a,l,i,f) ! (df)
                  if (j==l .and. k==n .and. c==e .and. b==d) hmatel = hmatel - h(a,m,i,f) ! (lm)(df)
                  if (k==m .and. j==l .and. c==e .and. b==d) hmatel = hmatel + h(a,n,i,f) ! (ln)(df)
          end function abb_voov_bbb

          pure function bbb_oo_bbb(i, j, k, a, b, c, l, m, n, d, e, f, h, nob) result(hmatel)
                  ! Expression:
                  ! -A(abc)A(jk)A(l/mn)A(i/jk) d(a,d)d(b,e)d(c,f)d(j,m)d(k,n) h(l,i)

                  integer, intent(in) :: nob
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: h(1:nob,1:nob)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  if (a==d .and. b==e .and. c==f) then

                          if (j==m .and. k==n) hmatel = hmatel - h(l,i) ! (1)
                          if (j==l .and. k==n) hmatel = hmatel + h(m,i) ! (lm)
                          if (i==m .and. k==n) hmatel = hmatel + h(l,j) ! (ij)
                          if (i==l .and. k==n) hmatel = hmatel - h(m,j) ! (lm)(ij)

                          if (j==m .and. i==l) hmatel = hmatel - h(n,k) ! (ln)(ik)
                          if (k==m .and. j==l) hmatel = hmatel - h(n,i) ! (ln)
                          if (i==m .and. j==n) hmatel = hmatel - h(l,k) ! (ij)
                          if (i==l .and. j==n) hmatel = hmatel + h(m,k) ! (lm)(ij)
                          if (k==m .and. i==l) hmatel = hmatel + h(n,j) ! (ln)(ik)

                  end if
          end function bbb_oo_bbb

          pure function bbb_vv_bbb(i, j, k, a, b, c, l, m, n, d, e, f, h, nub) result(hmatel)
                  ! Expression:
                  ! A(ijk)A(bc)A(d/ef)A(a/bc) d(i,l)d(j,m)d(k,n)d(b,e)d(c,f) h(a,d)

                  integer, intent(in) :: nub
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: h(1:nub,1:nub)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  if (i==l .and. j==m .and. k==n) then

                        if (b==e .and. c==f) hmatel = hmatel + h(a,d) ! (1)
                        if (a==e .and. c==f) hmatel = hmatel - h(b,d) ! (ab)
                        if (b==d .and. c==f) hmatel = hmatel - h(a,e) ! (de)
                        if (a==d .and. c==f) hmatel = hmatel + h(b,e) ! (de)(ab)
                        if (b==e .and. a==d) hmatel = hmatel + h(c,f) ! (df)(ac)

                        if (a==e .and. b==f) hmatel = hmatel + h(c,d) ! (ab)
                        if (a==d .and. b==f) hmatel = hmatel - h(c,e) ! (de)(ab)
                        if (c==e .and. b==d) hmatel = hmatel + h(a,f) ! (df)
                        if (c==e .and. a==d) hmatel = hmatel - h(b,f) ! (df)(ac)
                  end if
          end function bbb_vv_bbb

          pure function bbb_oooo_bbb(i, j, k, a, b, c, l, m, n, d, e, f, h, nob) result(hmatel)
                  ! Expression:
                  ! A(abc)A(k/ij)A(n/lm) d(a,d)d(b,e)d(c,f)d(k,n) h(l,m,i,j)

                  integer, intent(in) :: nob
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: h(1:nob,1:nob,1:nob,1:nob)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  ! (1)
                  if (a==d .and. b==e .and. c==f) then
                      ! (1)
                      if (k==n) then 
                              hmatel = hmatel + h(l,m,i,j) ! (1)
                      elseif (k==l) then 
                              hmatel = hmatel - h(n,m,i,j) ! (ln)
                      elseif (k==m) then 
                              hmatel = hmatel - h(l,n,i,j) ! (mn)
                      end if
                      ! (ik)
                      if (i==n) then 
                              hmatel = hmatel - h(l,m,k,j) ! (1)
                      elseif (i==l) then 
                              hmatel = hmatel + h(n,m,k,j) ! (ln)
                      elseif (i==m) then 
                              hmatel = hmatel + h(l,n,k,j) ! (mn)
                      end if
                      ! (jk)
                      if (j==n) then 
                              hmatel = hmatel - h(l,m,i,k) ! (1)
                      elseif (j==l) then 
                              hmatel = hmatel + h(n,m,i,k) ! (ln)
                      elseif (j==m) then 
                              hmatel = hmatel + h(l,n,i,k) ! (mn)
                      end if
                  end if
          end function bbb_oooo_bbb

          pure function bbb_vvvv_bbb(i, j, k, a, b, c, l, m, n, d, e, f, h, nub) result(hmatel)
                  ! Expression:
                  ! A(ijk)A(c/ab)A(f/de) d(i,l)d(j,m)d(k,n)d(c,f) h(a,b,d,e)

                  integer, intent(in) :: nub
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: h(1:nub,1:nub,1:nub,1:nub)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  ! (1)
                  if (i==l .and. j==m .and. k==n) then
                      if (a==f) then 
                              hmatel = hmatel - h(c,b,d,e) ! (ac)
                      elseif (a==d) then 
                              hmatel = hmatel + h(c,b,f,e) ! (ac)
                      elseif (a==e) then 
                              hmatel = hmatel + h(c,b,d,f) ! (ac)
                      end if

                      if (b==f) then
                              hmatel = hmatel - h(a,c,d,e) ! (bc)
                      elseif (b==d) then 
                              hmatel = hmatel + h(a,c,f,e) ! (bc)
                      elseif (b==e) then 
                              hmatel = hmatel + h(a,c,d,f) ! (bc)
                      end if

                      if (c==f) then
                              hmatel = hmatel + h(a,b,d,e) ! (1)
                      elseif (c==d) then
                              hmatel = hmatel - h(a,b,f,e) ! (1)
                      elseif (c==e) then 
                              hmatel = hmatel - h(a,b,d,f) ! (1)
                      end if
                  end if
          end function bbb_vvvv_bbb

          pure function bbb_voov_bbb(i, j, k, a, b, c, l, m, n, d, e, f, h, nob, nub) result(hmatel)
              ! Expression:
              ! A(jk)A(bc)A(i/jk)A(a/bc)A(l/mn)A(d/ef) d(j,m)d(k,n)d(b,e)d(c,f) h(a,l,i,d)

              integer, intent(in) :: nob, nub
              integer, intent(in) :: i, j, k, a, b, c
              integer, intent(in) :: l, m, n, d, e, f
              real(kind=8), intent(in) :: h(1:nub,1:nob,1:nob,1:nub)

              real(kind=8) :: hmatel

              hmatel = 0.0d0

                  ! (1)
                  if (j==m .and. k==n .and. b==e .and. c==f) hmatel = hmatel + h(a,l,i,d) ! (1)
                  if (j==m .and. k==n .and. b==d .and. c==f) hmatel = hmatel - h(a,l,i,e) ! (de)
                  if (j==l .and. k==n .and. b==e .and. c==f) hmatel = hmatel - h(a,m,i,d) ! (lm)
                  if (j==l .and. k==n .and. b==d .and. c==f) hmatel = hmatel + h(a,m,i,e) ! (de)(lm)
                  if (j==m .and. k==n .and. a==e .and. c==f) hmatel = hmatel - h(b,l,i,d) ! (ab)
                  if (j==m .and. k==n .and. a==d .and. c==f) hmatel = hmatel + h(b,l,i,e) ! (de)(ab)
                  if (j==l .and. k==n .and. a==e .and. c==f) hmatel = hmatel + h(b,m,i,d) ! (lm)(ab)
                  if (j==l .and. k==n .and. a==d .and. c==f) hmatel = hmatel - h(b,m,i,e) ! (de)(lm)(ab)
                  if (j==m .and. k==n .and. b==e .and. a==d) hmatel = hmatel + h(c,l,i,f) ! (df)(ac)
                  if (j==l .and. k==n .and. b==e .and. a==d) hmatel = hmatel - h(c,m,i,f) ! (df)(lm)(ac)
                  if (i==m .and. k==n .and. b==e .and. c==f) hmatel = hmatel - h(a,l,j,d) ! (ij)
                  if (i==m .and. k==n .and. b==d .and. c==f) hmatel = hmatel + h(a,l,j,e) ! (de)(ij)
                  if (i==l .and. k==n .and. b==e .and. c==f) hmatel = hmatel + h(a,m,j,d) ! (lm)(ij)
                  if (i==l .and. k==n .and. b==d .and. c==f) hmatel = hmatel - h(a,m,j,e) ! (de)(lm)(ij)
                  if (i==m .and. k==n .and. a==e .and. c==f) hmatel = hmatel + h(b,l,j,d) ! (ab)(ij)
                  if (i==m .and. k==n .and. a==d .and. c==f) hmatel = hmatel - h(b,l,j,e) ! (de)(ab)(ij)
                  if (i==l .and. k==n .and. a==e .and. c==f) hmatel = hmatel - h(b,m,j,d) ! (lm)(ab)(ij)
                  if (i==l .and. k==n .and. a==d .and. c==f) hmatel = hmatel + h(b,m,j,e) ! (de)(lm)(ab)(ij)
                  if (i==m .and. k==n .and. b==e .and. a==d) hmatel = hmatel - h(c,l,j,f) ! (df)(ac)(ij)
                  if (i==l .and. k==n .and. b==e .and. a==d) hmatel = hmatel + h(c,m,j,f) ! (df)(lm)(ac)(ij)
                  if (j==m .and. i==l .and. b==e .and. c==f) hmatel = hmatel + h(a,n,k,d) ! (ln)(ik)
                  if (j==m .and. i==l .and. b==d .and. c==f) hmatel = hmatel - h(a,n,k,e) ! (de)(ln)(ik)
                  if (j==m .and. i==l .and. a==e .and. c==f) hmatel = hmatel - h(b,n,k,d) ! (ln)(ab)(ik)
                  if (j==m .and. i==l .and. a==d .and. c==f) hmatel = hmatel + h(b,n,k,e) ! (de)(ln)(ab)(ik)
                  if (j==m .and. i==l .and. b==e .and. a==d) hmatel = hmatel + h(c,n,k,f) ! (df)(ln)(ac)(ik)
                  ! (bc)
                  if (j==m .and. k==n .and. c==e .and. b==d) hmatel = hmatel + h(a,l,i,f) ! (df)
                  if (j==l .and. k==n .and. c==e .and. b==d) hmatel = hmatel - h(a,m,i,f) ! (df)(lm)
                  if (j==m .and. k==n .and. a==e .and. b==f) hmatel = hmatel + h(c,l,i,d) ! (ab)
                  if (j==m .and. k==n .and. a==d .and. b==f) hmatel = hmatel - h(c,l,i,e) ! (de)(ab)
                  if (j==l .and. k==n .and. a==e .and. b==f) hmatel = hmatel - h(c,m,i,d) ! (lm)(ab)
                  if (j==l .and. k==n .and. a==d .and. b==f) hmatel = hmatel + h(c,m,i,e) ! (de)(lm)(ab)
                  if (j==m .and. k==n .and. c==e .and. a==d) hmatel = hmatel - h(b,l,i,f) ! (df)(ac)
                  if (j==l .and. k==n .and. c==e .and. a==d) hmatel = hmatel + h(b,m,i,f) ! (df)(lm)(ac)
                  if (i==m .and. k==n .and. c==e .and. b==d) hmatel = hmatel - h(a,l,j,f) ! (df)(ij)
                  if (i==l .and. k==n .and. c==e .and. b==d) hmatel = hmatel + h(a,m,j,f) ! (df)(lm)(ij)
                  if (i==m .and. k==n .and. a==e .and. b==f) hmatel = hmatel - h(c,l,j,d) ! (ab)(ij)
                  if (i==m .and. k==n .and. a==d .and. b==f) hmatel = hmatel + h(c,l,j,e) ! (de)(ab)(ij)
                  if (i==l .and. k==n .and. a==e .and. b==f) hmatel = hmatel + h(c,m,j,d) ! (lm)(ab)(ij)
                  if (i==l .and. k==n .and. a==d .and. b==f) hmatel = hmatel - h(c,m,j,e) ! (de)(lm)(ab)(ij)
                  if (i==m .and. k==n .and. c==e .and. a==d) hmatel = hmatel + h(b,l,j,f) ! (df)(ac)(ij)
                  if (i==l .and. k==n .and. c==e .and. a==d) hmatel = hmatel - h(b,m,j,f) ! (df)(lm)(ac)(ij)
                  if (j==m .and. i==l .and. c==e .and. b==d) hmatel = hmatel + h(a,n,k,f) ! (df)(ln)(ik)
                  if (j==m .and. i==l .and. a==e .and. b==f) hmatel = hmatel + h(c,n,k,d) ! (ln)(ab)(ik)
                  if (j==m .and. i==l .and. a==d .and. b==f) hmatel = hmatel - h(c,n,k,e) ! (de)(ln)(ab)(ik)
                  if (j==m .and. i==l .and. c==e .and. a==d) hmatel = hmatel - h(b,n,k,f) ! (df)(ln)(ac)(ik)
                  ! (jk)
                  if (k==m .and. j==l .and. b==e .and. c==f) hmatel = hmatel + h(a,n,i,d) ! (ln)
                  if (k==m .and. j==l .and. b==d .and. c==f) hmatel = hmatel - h(a,n,i,e) ! (de)(ln)
                  if (k==m .and. j==l .and. a==e .and. c==f) hmatel = hmatel - h(b,n,i,d) ! (ln)(ab)
                  if (k==m .and. j==l .and. a==d .and. c==f) hmatel = hmatel + h(b,n,i,e) ! (de)(ln)(ab)
                  if (k==m .and. j==l .and. b==e .and. a==d) hmatel = hmatel + h(c,n,i,f) ! (df)(ln)(ac)
                  if (i==m .and. j==n .and. b==e .and. c==f) hmatel = hmatel + h(a,l,k,d) ! (ij)
                  if (i==m .and. j==n .and. b==d .and. c==f) hmatel = hmatel - h(a,l,k,e) ! (de)(ij)
                  if (i==l .and. j==n .and. b==e .and. c==f) hmatel = hmatel - h(a,m,k,d) ! (lm)(ij)
                  if (i==l .and. j==n .and. b==d .and. c==f) hmatel = hmatel + h(a,m,k,e) ! (de)(lm)(ij)
                  if (i==m .and. j==n .and. a==e .and. c==f) hmatel = hmatel - h(b,l,k,d) ! (ab)(ij)
                  if (i==m .and. j==n .and. a==d .and. c==f) hmatel = hmatel + h(b,l,k,e) ! (de)(ab)(ij)
                  if (i==l .and. j==n .and. a==e .and. c==f) hmatel = hmatel + h(b,m,k,d) ! (lm)(ab)(ij)
                  if (i==l .and. j==n .and. a==d .and. c==f) hmatel = hmatel - h(b,m,k,e) ! (de)(lm)(ab)(ij)
                  if (i==m .and. j==n .and. b==e .and. a==d) hmatel = hmatel + h(c,l,k,f) ! (df)(ac)(ij)
                  if (i==l .and. j==n .and. b==e .and. a==d) hmatel = hmatel - h(c,m,k,f) ! (df)(lm)(ac)(ij)
                  if (k==m .and. i==l .and. b==e .and. c==f) hmatel = hmatel - h(a,n,j,d) ! (ln)(ik)
                  if (k==m .and. i==l .and. b==d .and. c==f) hmatel = hmatel + h(a,n,j,e) ! (de)(ln)(ik)
                  if (k==m .and. i==l .and. a==e .and. c==f) hmatel = hmatel + h(b,n,j,d) ! (ln)(ab)(ik)
                  if (k==m .and. i==l .and. a==d .and. c==f) hmatel = hmatel - h(b,n,j,e) ! (de)(ln)(ab)(ik)
                  if (k==m .and. i==l .and. b==e .and. a==d) hmatel = hmatel - h(c,n,j,f) ! (df)(ln)(ac)(ik)
                  ! (jk)(bc), apply(jk)
                  if (k==m .and. j==l .and. c==e .and. b==d) hmatel = hmatel + h(a,n,i,f) ! (df)(ln)
                  if (k==m .and. j==l .and. a==e .and. b==f) hmatel = hmatel + h(c,n,i,d) ! (ln)(ab)
                  if (k==m .and. j==l .and. a==d .and. b==f) hmatel = hmatel - h(c,n,i,e) ! (de)(ln)(ab)
                  if (k==m .and. j==l .and. c==e .and. a==d) hmatel = hmatel - h(b,n,i,f) ! (df)(ln)(ac)
                  if (i==m .and. j==n .and. c==e .and. b==d) hmatel = hmatel + h(a,l,k,f) ! (df)(ij)
                  if (i==l .and. j==n .and. c==e .and. b==d) hmatel = hmatel - h(a,m,k,f) ! (df)(lm)(ij)
                  if (i==m .and. j==n .and. a==e .and. b==f) hmatel = hmatel + h(c,l,k,d) ! (ab)(ij)
                  if (i==m .and. j==n .and. a==d .and. b==f) hmatel = hmatel - h(c,l,k,e) ! (de)(ab)(ij)
                  if (i==l .and. j==n .and. a==e .and. b==f) hmatel = hmatel - h(c,m,k,d) ! (lm)(ab)(ij)
                  if (i==l .and. j==n .and. a==d .and. b==f) hmatel = hmatel + h(c,m,k,e) ! (de)(lm)(ab)(ij)
                  if (i==m .and. j==n .and. c==e .and. a==d) hmatel = hmatel - h(b,l,k,f) ! (df)(ac)(ij)
                  if (i==l .and. j==n .and. c==e .and. a==d) hmatel = hmatel + h(b,m,k,f) ! (df)(lm)(ac)(ij)
                  if (k==m .and. i==l .and. c==e .and. b==d) hmatel = hmatel - h(a,n,j,f) ! (df)(ln)(ik)
                  if (k==m .and. i==l .and. a==e .and. b==f) hmatel = hmatel - h(c,n,j,d) ! (ln)(ab)(ik)
                  if (k==m .and. i==l .and. a==d .and. b==f) hmatel = hmatel + h(c,n,j,e) ! (de)(ln)(ab)(ik)
                  if (k==m .and. i==l .and. c==e .and. a==d) hmatel = hmatel + h(b,n,j,f) ! (df)(ln)(ac)(ik)
          end function bbb_voov_bbb

          pure function bbb_ovvo_abb(i, j, k, a, b, c, l, m, n, d, e, f, h, noa, nua, nob, nub) result(hmatel)
              ! Expression:
              ! Found by spin-inverting the code in aaa_voov_aab (h2b_voov -> h2b_ovvo)

              integer, intent(in) :: noa, nua, nob, nub
              integer, intent(in) :: i, j, k, a, b, c
              integer, intent(in) :: l, m, n, d, e, f
              real(kind=8), intent(in) :: h(1:noa,1:nub,1:nua,1:nob)

              real(kind=8) :: hmatel

              hmatel = 0.0d0

              if (j==m .and. k==n .and. c==f .and. b==e) hmatel = hmatel + h(l,a,d,i) ! (1)
              if (i==m .and. k==n .and. c==f .and. b==e) hmatel = hmatel - h(l,a,d,j) 
              if (j==m .and. k==n .and. c==f .and. a==e) hmatel = hmatel - h(l,b,d,i) 
              if (i==m .and. k==n .and. c==f .and. a==e) hmatel = hmatel + h(l,b,d,j) 
              if (i==m .and. j==n .and. c==f .and. b==e) hmatel = hmatel + h(l,a,d,k)
              if (i==m .and. j==n .and. c==f .and. a==e) hmatel = hmatel - h(l,b,d,k) 
              if (j==m .and. k==n .and. b==f .and. a==e) hmatel = hmatel + h(l,c,d,i) 
              if (i==m .and. k==n .and. b==f .and. a==e) hmatel = hmatel - h(l,c,d,j) 
              if (i==m .and. j==n .and. b==f .and. a==e) hmatel = hmatel + h(l,c,d,k) 

          end function bbb_ovvo_abb

          pure function nexc2(i, j, k, l)
              ! Counts the number of differences that occur between
              ! the set of integers (i,j) and (k,l).
              integer, intent(in) :: i, j, k, l

              integer :: nexc2

              nexc2 = count((/i,j/)==(/k,l/))&
                     +count((/j,i/)==(/k,l/))
              nexc2 = 2 - nexc2
          end function nexc2

          pure function nexc3(i, j, k, l, m, n)
              ! Counts the number of differences that occur between
              ! the set of integers (i,j,k) and (l,m,n).
              integer, intent(in) :: i, j, k, l, m, n

              integer :: nexc3

              nexc3 = count((/i,j,k/)==(/l,m,n/))&
                     +count((/i,k,j/)==(/l,m,n/))&
                     +count((/j,k,i/)==(/l,m,n/))&
                     +count((/j,i,k/)==(/l,m,n/))&
                     +count((/k,i,j/)==(/l,m,n/))&
                     +count((/k,j,i/)==(/l,m,n/))
              nexc3 = shiftr(6 - nexc3, 1)
          end function nexc3

end module ccp_quadratic_loops_direct_h
