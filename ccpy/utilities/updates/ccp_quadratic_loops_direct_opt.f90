module ccp_quadratic_loops_direct_opt

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

              subroutine update_t3a_p(resid,&
                                      t3a_amps, t3a_excits,&
                                      t3b_amps, t3b_excits,&
                                      t2a,&
                                      H1A_oo, H1A_vv,&
                                      H2A_oovv, H2A_vvov, H2A_vooo,&
                                      H2A_oooo, H2A_voov, H2A_vvvv,&
                                      H2B_oovv, H2B_voov,&
                                      fA_oo, fA_vv,&
                                      shift,&
                                      n3aaa, n3aab,&
                                      noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub, n3aaa, n3aab
                  integer, intent(in) :: t3b_excits(6,n3aab)
                  real(kind=8), intent(in) :: t2a(nua, nua, noa, noa),&
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
                  real(kind=8), intent(in) :: t3b_amps(n3aab)

                  integer, intent(inout) :: t3a_excits(6,n3aaa)
                  !f2py intent(in,out) :: t3a_excits(6,0:n3aaa-1)
                  real(kind=8), intent(inout) :: t3a_amps(n3aaa)
                  !f2py intent(in,out) :: t3a_amps(0:n3aaa-1)

                  real(kind=8), intent(out) :: resid(n3aaa)

                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)

                  integer, allocatable :: id3a_h(:,:)
                  integer, allocatable :: xixjxk_table(:,:,:)
                  integer, allocatable :: id3b_h(:,:,:)
                  integer, allocatable :: eck_table(:,:)
                  integer, allocatable :: xixj_table(:,:)

                  real(kind=8), allocatable :: t3_amps_buff(:)
                  integer, allocatable :: t3_excits_buff(:,:)

                  real(kind=8) :: I2A_vvov(nua, nua, noa, nua), I2A_vooo(nua, noa, noa, noa)
                  real(kind=8) :: val, denom, t_amp, res_mm23, hmatel
                  integer :: a, b, c, d, i, j, k, l, e, f, m, n, idet, jdet
                  integer :: idx

                  integer :: ijk, ij, ik, jk, jb
                  integer :: lmi, lmj, lmk, lij, lik, ljk
                  real(kind=8) :: phase

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

                  ! Zero the residual container
                  resid = 0.0d0

                  !!!! diagram 1: -A(i/jk) h1a(mi) * t3a(abcmjk)
                  !!!! diagram 2: A(a/bc) h1a(ae) * t3a(ebcijk)
                  ! allocate sorting arrays
                  allocate(id3a_h(noa*(noa-1)*(noa-2)/6,2))
                  allocate(xixjxk_table(noa,noa,noa))
                  id3a_h = 0; xixjxk_table = 0;
                  call sort_t3a_h(t3a_excits, t3a_amps, id3a_h, xixjxk_table, noa, nua, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3a_excits,t3a_amps,&
                  !$omp id3a_h,xixjxk_table,&
                  !$omp H1A_oo,H1A_vv,&
                  !$omp noa,nua,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp ijk,lij,lik,ljk,phase)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa
                     a = t3a_excits(1,idet); b = t3a_excits(2,idet); c = t3a_excits(3,idet);
                     i = t3a_excits(4,idet); j = t3a_excits(5,idet); k = t3a_excits(6,idet);
                     ijk = xixjxk_table(i,j,k)
                     ! diagram 1: h1a(oo)
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
                     end do
                     ! diagram 2: h1a(vv)
                     do jdet = id3a_h(ijk,1), id3a_h(ijk,2)
                        d = t3a_excits(1,jdet); e = t3a_excits(2,jdet); f = t3a_excits(3,jdet);
                        if (nexc3(a,b,c,d,e,f)>=2) cycle
                        hmatel = 0.0d0
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
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(id3a_h,xixjxk_table)

                  !!!! diagram 3: 1/2 A(i/jk) h2a(mnij) * t3a(abcmnk) 
                  ! allocate new sorting arrays
                  allocate(loc_arr(nua*(nua-1)*(nua-2)/6*noa,2))
                  allocate(idx_table(nua,nua,nua,noa))
                  !!! ABCK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/3,noa/), nua, nua, nua, noa)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/1,2,3,6/), nua, nua, nua, noa, nua*(nua-1)*(nua-2)/6*noa, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3a_excits,&
                  !$omp t3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_oo,H2A_oooo,&
                  !$omp noa,nua,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa
                     a = t3a_excits(1,idet); b = t3a_excits(2,idet); c = t3a_excits(3,idet);
                     i = t3a_excits(4,idet); j = t3a_excits(5,idet); k = t3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        l = t3a_excits(4,jdet); m = t3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(oooo) | lmkabc >
                        hmatel = h2a_oooo(l,m,i,j)
                        ! compute < ijkabc | h1a(oo) | lmkabc > = -A(ij)A(lm) h1a_oo(l,i) * delta(m,j)
                        !if (m==j) hmatel = hmatel - h1a_oo(l,i) ! (1)
                        !if (m==i) hmatel = hmatel + h1a_oo(l,j) ! (ij)
                        !if (l==j) hmatel = hmatel + h1a_oo(m,i) ! (lm)
                        !if (l==i) hmatel = hmatel - h1a_oo(m,j) ! (ij)(lm)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     ! (ik)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l = t3a_excits(4,jdet); m = t3a_excits(5,jdet);
                           ! compute < ijkabc | h2a(oooo) | lmiabc >
                           hmatel = -h2a_oooo(l,m,k,j)
                           ! compute < ijkabc | h1a(oo) | lmiabc > = A(jk)A(lm) h1a_oo(l,k) * delta(m,j)
                           !if (m==j) hmatel = hmatel + h1a_oo(l,k) ! (1)
                           !if (m==k) hmatel = hmatel - h1a_oo(l,j) ! (jk)
                           !if (l==j) hmatel = hmatel - h1a_oo(m,k) ! (lm)
                           !if (l==k) hmatel = hmatel + h1a_oo(m,j) ! (jk)(lm)
                           resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l = t3a_excits(4,jdet); m = t3a_excits(5,jdet);
                           ! compute < ijkabc | h2a(oooo) | lmjabc >
                           hmatel = -h2a_oooo(l,m,i,k)
                           ! compute < ijkabc | h1a(oo) | lmjabc > = A(ik)A(lm) h1a_oo(l,i) * delta(m,k)
                           !if (m==k) hmatel = hmatel + h1a_oo(l,i) ! (1)
                           !if (m==i) hmatel = hmatel - h1a_oo(l,k) ! (ik)
                           !if (l==k) hmatel = hmatel - h1a_oo(m,i) ! (lm)
                           !if (l==i) hmatel = hmatel + h1a_oo(m,k) ! (ik)(lm)
                           resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABCI LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/1,noa-2/), nua, nua, nua, noa)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/1,2,3,4/), nua, nua, nua, noa, nua*(nua-1)*(nua-2)/6*noa, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3a_excits,&
                  !$omp t3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_oo,H2A_oooo,&
                  !$omp noa,nua,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa
                     a = t3a_excits(1,idet); b = t3a_excits(2,idet); c = t3a_excits(3,idet);
                     i = t3a_excits(4,idet); j = t3a_excits(5,idet); k = t3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,i)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        m = t3a_excits(5,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(oooo) | imnabc >
                        hmatel = h2a_oooo(m,n,j,k)
                        ! compute < ijkabc | h1a(oo) | imnabc > = -A(jk)A(mn) h1a_oo(m,j) * delta(n,k)
                        !if (n==k) hmatel = hmatel - h1a_oo(m,j)
                        !if (n==j) hmatel = hmatel + h1a_oo(m,k)
                        !if (m==k) hmatel = hmatel + h1a_oo(n,j)
                        !if (m==j) hmatel = hmatel - h1a_oo(n,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           m = t3a_excits(5,jdet); n = t3a_excits(6,jdet);
                           ! compute < ijkabc | h2a(oooo) | jmnabc >
                           hmatel = -h2a_oooo(m,n,i,k)
                           ! compute < ijkabc | h1a(oo) | jmnabc > = A(ik)A(mn) h1a_oo(m,i) * delta(n,k)
                           !if (n==k) hmatel = hmatel + h1a_oo(m,i)
                           !if (n==i) hmatel = hmatel - h1a_oo(m,k)
                           !if (m==k) hmatel = hmatel - h1a_oo(n,i)
                           !if (m==i) hmatel = hmatel + h1a_oo(n,k)
                           resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                        end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           m = t3a_excits(5,jdet); n = t3a_excits(6,jdet);
                           ! compute < ijkabc | h2a(oooo) | kmnabc >
                           hmatel = -h2a_oooo(m,n,j,i)
                           ! compute < ijkabc | h1a(oo) | kmnabc > = A(ij)A(mn) h1a_oo(m,j) * delta(n,i)
                           !if (n==i) hmatel = hmatel - h1a_oo(m,j)
                           !if (n==j) hmatel = hmatel + h1a_oo(m,i)
                           !if (m==i) hmatel = hmatel + h1a_oo(n,j)
                           !if (m==j) hmatel = hmatel - h1a_oo(n,i)
                           resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABCJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/2,noa-1/), nua, nua, nua, noa)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/1,2,3,5/), nua, nua, nua, noa, nua*(nua-1)*(nua-2)/6*noa, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3a_excits,&
                  !$omp t3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_oo,H2A_oooo,&
                  !$omp noa,nua,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa
                     a = t3a_excits(1,idet); b = t3a_excits(2,idet); c = t3a_excits(3,idet);
                     i = t3a_excits(4,idet); j = t3a_excits(5,idet); k = t3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,j)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        l = t3a_excits(4,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(oooo) | ljnabc >
                        hmatel = h2a_oooo(l,n,i,k)
                        ! compute < ijkabc | h1a(oo) | ljnabc > = -A(ik)A(ln) h1a_oo(l,i) * delta(n,k)
                        !if (n==k) hmatel = hmatel - h1a_oo(l,i)
                        !if (n==i) hmatel = hmatel + h1a_oo(l,k)
                        !if (l==k) hmatel = hmatel + h1a_oo(n,i)
                        !if (l==i) hmatel = hmatel - h1a_oo(n,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l = t3a_excits(4,jdet); n = t3a_excits(6,jdet);
                           ! compute < ijkabc | h2a(oooo) | linabc >
                           hmatel = -h2a_oooo(l,n,j,k)
                           ! compute < ijkabc | h1a(oo) | linabc > = A(jk)A(ln) h1a_oo(l,j) * delta(n,k)
                           !if (n==k) hmatel = hmatel + h1a_oo(l,j)
                           !if (n==j) hmatel = hmatel - h1a_oo(l,k)
                           !if (l==k) hmatel = hmatel - h1a_oo(n,j)
                           !if (l==j) hmatel = hmatel + h1a_oo(n,k)
                           resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l = t3a_excits(4,jdet); n = t3a_excits(6,jdet);
                           ! compute < ijkabc | h2a(oooo) | lknabc >
                           hmatel = -h2a_oooo(l,n,i,j)
                           ! compute < ijkabc | h1a(oo) | lknabc > = A(ij)A(ln) h1a_oo(l,i) * delta(n,j)
                           !if (n==j) hmatel = hmatel + h1a_oo(l,i)
                           !if (n==i) hmatel = hmatel - h1a_oo(l,j)
                           !if (l==j) hmatel = hmatel - h1a_oo(n,i)
                           !if (l==i) hmatel = hmatel + h1a_oo(n,j)
                           resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 4: 1/2 A(c/ab) h2a(abef) * t3a(ebcijk) 
                  ! allocate new sorting arrays
                  allocate(loc_arr(noa*(noa-1)*(noa-2)/6*nua,2))
                  allocate(idx_table(noa,noa,noa,nua))
                  !!! IJKA LOOP !!!
                  call get_index_table(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/1,nua-2/), noa, noa, noa, nua)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/4,5,6,1/), noa, noa, noa, nua, noa*(noa-1)*(noa-2)/6*nua, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3a_excits,&
                  !$omp t3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_vv,H2A_vvvv,&
                  !$omp noa,nua,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa
                     a = t3a_excits(1,idet); b = t3a_excits(2,idet); c = t3a_excits(3,idet);
                     i = t3a_excits(4,idet); j = t3a_excits(5,idet); k = t3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,a)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); f = t3a_excits(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkaef >
                        hmatel = h2a_vvvv(b,c,e,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); f = t3a_excits(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkbef >
                        hmatel = -h2a_vvvv(a,c,e,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); f = t3a_excits(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkcef >
                        hmatel = -h2a_vvvv(b,a,e,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! IJKB LOOP !!!
                  call get_index_table(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/2,nua-1/), noa, noa, noa, nua)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/4,5,6,2/), noa, noa, noa, nua, noa*(noa-1)*(noa-2)/6*nua, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3a_excits,&
                  !$omp t3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_vv,H2A_vvvv,&
                  !$omp noa,nua,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa
                     a = t3a_excits(1,idet); b = t3a_excits(2,idet); c = t3a_excits(3,idet);
                     i = t3a_excits(4,idet); j = t3a_excits(5,idet); k = t3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,b)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); f = t3a_excits(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdbf >
                        hmatel = h2a_vvvv(a,c,d,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); f = t3a_excits(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdaf >
                        hmatel = -h2a_vvvv(b,c,d,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); f = t3a_excits(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdcf >
                        hmatel = -h2a_vvvv(a,b,d,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if 
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! IJKC LOOP !!!
                  call get_index_table(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/3,nua/), noa, noa, noa, nua)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/4,5,6,3/), noa, noa, noa, nua, noa*(noa-1)*(noa-2)/6*nua, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3a_excits,&
                  !$omp t3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_vv,H2A_vvvv,&
                  !$omp noa,nua,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa
                     a = t3a_excits(1,idet); b = t3a_excits(2,idet); c = t3a_excits(3,idet);
                     i = t3a_excits(4,idet); j = t3a_excits(5,idet); k = t3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,c)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); e = t3a_excits(2,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdec >
                        hmatel = h2a_vvvv(a,b,d,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); e = t3a_excits(2,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdea >
                        hmatel = -h2a_vvvv(c,b,d,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); e = t3a_excits(2,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdeb >
                        hmatel = -h2a_vvvv(a,c,d,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 5: A(i/jk)A(a/bc) h2a(amie) * t3a(ebcmjk)
                  ! allocate sorting arrays (can be reused for each permutation)
                  allocate(loc_arr((nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2,2))
                  allocate(idx_table(nua,nua,noa,noa))
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/1,2,4,5/), nua, nua, noa, noa, (nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = t3a_excits(1,idet); b = t3a_excits(2,idet); c = t3a_excits(3,idet);
                     i = t3a_excits(4,idet); j = t3a_excits(5,idet); k = t3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3a_excits(3,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijnabf >
                        hmatel = h2a_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3a_excits(3,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijnbcf >
                        hmatel = h2a_voov(a,n,k,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3a_excits(3,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijnacf >
                        hmatel = -h2a_voov(b,n,k,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3a_excits(3,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jknabf >
                        hmatel = h2a_voov(c,n,i,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3a_excits(3,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jknbcf >
                        hmatel = h2a_voov(a,n,i,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3a_excits(3,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jknacf >
                        hmatel = -h2a_voov(b,n,i,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3a_excits(3,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | iknabf >
                        hmatel = -h2a_voov(c,n,j,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3a_excits(3,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | iknbcf >
                        hmatel = -h2a_voov(a,n,j,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3a_excits(3,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | iknacf >
                        hmatel = h2a_voov(b,n,j,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                  end do
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/1,3,4,5/), nua, nua, noa, noa, (nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = t3a_excits(1,idet); b = t3a_excits(2,idet); c = t3a_excits(3,idet);
                     i = t3a_excits(4,idet); j = t3a_excits(5,idet); k = t3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijnaec >
                        hmatel = h2a_voov(b,n,k,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijnbec >
                        hmatel = -h2a_voov(a,n,k,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijnaeb >
                        hmatel = -h2a_voov(c,n,k,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jknaec >
                        hmatel = h2a_voov(b,n,i,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jknbec >
                        hmatel = -h2a_voov(a,n,i,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jknaeb >
                        hmatel = -h2a_voov(c,n,i,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | iknaec >
                        hmatel = -h2a_voov(b,n,j,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | iknbec >
                        hmatel = h2a_voov(a,n,j,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | iknaeb >
                        hmatel = h2a_voov(c,n,j,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                  end do
                  !!! BCIJ LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/2,3,4,5/), nua, nua, noa, noa, (nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = t3a_excits(1,idet); b = t3a_excits(2,idet); c = t3a_excits(3,idet);
                     i = t3a_excits(4,idet); j = t3a_excits(5,idet); k = t3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijndbc >
                        hmatel = h2a_voov(a,n,k,d)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijndac >
                        hmatel = -h2a_voov(b,n,k,d)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijndab >
                        hmatel = h2a_voov(c,n,k,d)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jkndbc >
                        hmatel = h2a_voov(a,n,i,d)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jkndac >
                        hmatel = -h2a_voov(b,n,i,d)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jkndab >
                        hmatel = h2a_voov(c,n,i,d)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ikndbc >
                        hmatel = -h2a_voov(a,n,j,d)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ikndac >
                        hmatel = h2a_voov(b,n,j,d)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); n = t3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ikndab >
                        hmatel = -h2a_voov(c,n,j,d)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                  end do
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/1,2,4,6/), nua, nua, noa, noa, (nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = t3a_excits(1,idet); b = t3a_excits(2,idet); c = t3a_excits(3,idet);
                     i = t3a_excits(4,idet); j = t3a_excits(5,idet); k = t3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3a_excits(3,jdet); m = t3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkabf >
                        hmatel = h2a_voov(c,m,j,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3a_excits(3,jdet); m = t3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkbcf >
                        hmatel = h2a_voov(a,m,j,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3a_excits(3,jdet); m = t3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkacf >
                        hmatel = -h2a_voov(b,m,j,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3a_excits(3,jdet); m = t3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkabf >
                        hmatel = -h2a_voov(c,m,i,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3a_excits(3,jdet); m = t3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkbcf >
                        hmatel = -h2a_voov(a,m,i,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3a_excits(3,jdet); m = t3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkacf >
                        hmatel = h2a_voov(b,m,i,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3a_excits(3,jdet); m = t3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjabf >
                        hmatel = -h2a_voov(c,m,k,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3a_excits(3,jdet); m = t3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjbcf >
                        hmatel = -h2a_voov(a,m,k,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3a_excits(3,jdet); m = t3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjacf >
                        hmatel = h2a_voov(b,m,k,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                  end do
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/1,3,4,6/), nua, nua, noa, noa, (nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = t3a_excits(1,idet); b = t3a_excits(2,idet); c = t3a_excits(3,idet);
                     i = t3a_excits(4,idet); j = t3a_excits(5,idet); k = t3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); m = t3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkaec >
                        hmatel = h2a_voov(b,m,j,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); m = t3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkbec >
                        hmatel = -h2a_voov(a,m,j,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); m = t3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkaeb >
                        hmatel = -h2a_voov(c,m,j,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); m = t3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkaec >
                        hmatel = -h2a_voov(b,m,i,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); m = t3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkbec >
                        hmatel = h2a_voov(a,m,i,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); m = t3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkaeb >
                        hmatel = h2a_voov(c,m,i,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); m = t3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjaec >
                        hmatel = -h2a_voov(b,m,k,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); m = t3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjbec >
                        hmatel = h2a_voov(a,m,k,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); m = t3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjaeb >
                        hmatel = h2a_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                  end do
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/2,3,4,6/), nua, nua, noa, noa, (nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = t3a_excits(1,idet); b = t3a_excits(2,idet); c = t3a_excits(3,idet);
                     i = t3a_excits(4,idet); j = t3a_excits(5,idet); k = t3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); m = t3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkdbc >
                        hmatel = h2a_voov(a,m,j,d)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); m = t3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkdac >
                        hmatel = -h2a_voov(b,m,j,d)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); m = t3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkdab >
                        hmatel = h2a_voov(c,m,j,d)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); m = t3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkdbc >
                        hmatel = -h2a_voov(a,m,i,d)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); m = t3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkdac >
                        hmatel = h2a_voov(b,m,i,d)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); m = t3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkdab >
                        hmatel = -h2a_voov(c,m,i,d)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); m = t3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjdbc >
                        hmatel = -h2a_voov(a,m,k,d)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); m = t3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjdac >
                        hmatel = h2a_voov(b,m,k,d)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); m = t3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjdab >
                        hmatel = -h2a_voov(c,m,k,d)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                  end do
                  !!! ABJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/1,2,5,6/), nua, nua, noa, noa, (nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = t3a_excits(1,idet); b = t3a_excits(2,idet); c = t3a_excits(3,idet);
                     i = t3a_excits(4,idet); j = t3a_excits(5,idet); k = t3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3a_excits(3,jdet); l = t3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkabf >
                        hmatel = h2a_voov(c,l,i,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3a_excits(3,jdet); l = t3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkbcf >
                        hmatel = h2a_voov(a,l,i,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3a_excits(3,jdet); l = t3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkacf >
                        hmatel = -h2a_voov(b,l,i,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3a_excits(3,jdet); l = t3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likabf >
                        hmatel = -h2a_voov(c,l,j,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3a_excits(3,jdet); l = t3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likbcf >
                        hmatel = -h2a_voov(a,l,j,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3a_excits(3,jdet); l = t3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likacf >
                        hmatel = h2a_voov(b,l,j,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3a_excits(3,jdet); l = t3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijabf >
                        hmatel = h2a_voov(c,l,k,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3a_excits(3,jdet); l = t3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijbcf >
                        hmatel = h2a_voov(a,l,k,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3a_excits(3,jdet); l = t3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijacf >
                        hmatel = -h2a_voov(b,l,k,f)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                  end do
                  !!! ACJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/1,3,5,6/), nua, nua, noa, noa, (nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = t3a_excits(1,idet); b = t3a_excits(2,idet); c = t3a_excits(3,idet);
                     i = t3a_excits(4,idet); j = t3a_excits(5,idet); k = t3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); l = t3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkaec >
                        hmatel = h2a_voov(b,l,i,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); l = t3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkbec >
                        hmatel = -h2a_voov(a,l,i,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); l = t3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkaeb >
                        hmatel = -h2a_voov(c,l,i,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); l = t3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likaec >
                        hmatel = -h2a_voov(b,l,j,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); l = t3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likbec >
                        hmatel = h2a_voov(a,l,j,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); l = t3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likaeb >
                        hmatel = h2a_voov(c,l,j,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); l = t3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijaec >
                        hmatel = h2a_voov(b,l,k,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); l = t3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijbec >
                        hmatel = -h2a_voov(a,l,k,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3a_excits(2,jdet); l = t3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijaeb >
                        hmatel = -h2a_voov(c,l,k,e)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                  end do
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/2,3,5,6/), nua, nua, noa, noa, (nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = t3a_excits(1,idet); b = t3a_excits(2,idet); c = t3a_excits(3,idet);
                     i = t3a_excits(4,idet); j = t3a_excits(5,idet); k = t3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); l = t3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkdbc >
                        hmatel = h2a_voov(a,l,i,d)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); l = t3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkdac >
                        hmatel = -h2a_voov(b,l,i,d)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); l = t3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkdab >
                        hmatel = h2a_voov(c,l,i,d)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); l = t3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likdbc >
                        hmatel = -h2a_voov(a,l,j,d)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); l = t3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likdac >
                        hmatel = h2a_voov(b,l,j,d)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); l = t3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likdab >
                        hmatel = -h2a_voov(c,l,j,d)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); l = t3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijdbc >
                        hmatel = h2a_voov(a,l,k,d)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); l = t3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijdac >
                        hmatel = -h2a_voov(b,l,k,d)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3a_excits(1,jdet); l = t3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijdab >
                        hmatel = h2a_voov(c,l,k,d)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 6: A(i/jk)A(a/bc) h2b(amie) * t3b(abeijm)
                  ! allocate and copy over t3b arrays
                  allocate(t3_amps_buff(n3aab),t3_excits_buff(6,n3aab))
                  t3_amps_buff(:) = t3b_amps(:)
                  t3_excits_buff(:,:) = t3b_excits(:,:)
                  ! allocate sorting arrays (can be reused for each permutation)
                  allocate(loc_arr(nua*(nua-1)/2*noa*(noa-1)/2,2))
                  allocate(idx_table(nua,nua,noa,noa))
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,2,4,5/), nua, nua, noa, noa, nua*(nua-1)/2*noa*(noa-1)/2, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3a_excits,t3_excits_buff,&
                  !$omp t3a_amps,t3_amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa
                     a = t3a_excits(1,idet); b = t3a_excits(2,idet); c = t3a_excits(3,idet);
                     i = t3a_excits(4,idet); j = t3a_excits(5,idet); k = t3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3_excits_buff(3,jdet); n = t3_excits_buff(6,jdet);
                        ! compute < ijkabc | h2b(voov) | ijn~abf~ >
                        hmatel = h2b_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3_excits_buff(3,jdet); n = t3_excits_buff(6,jdet);
                        ! compute < ijkabc | h2b(voov) | jkn~abf~ >
                        hmatel = h2b_voov(c,n,i,f)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3_excits_buff(3,jdet); n = t3_excits_buff(6,jdet);
                        ! compute < ijkabc | h2b(voov) | ikn~abf~ >
                        hmatel = -h2b_voov(c,n,j,f)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3_excits_buff(3,jdet); n = t3_excits_buff(6,jdet);
                        ! compute < ijkabc | h2b(voov) | ijn~bcf~ >
                        hmatel = h2b_voov(a,n,k,f)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                     end if
                     ! (ik)(ac)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3_excits_buff(3,jdet); n = t3_excits_buff(6,jdet);
                        ! compute < ijkabc | h2b(voov) | jkn~bcf~ >
                        hmatel = h2b_voov(a,n,i,f)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                     end if
                     ! (jk)(ac)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3_excits_buff(3,jdet); n = t3_excits_buff(6,jdet);
                        ! compute < ijkabc | h2b(voov) | ikn~bcf~ >
                        hmatel = -h2b_voov(a,n,j,f)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3_excits_buff(3,jdet); n = t3_excits_buff(6,jdet);
                        ! compute < ijkabc | h2b(voov) | ijn~acf~ >
                        hmatel = -h2b_voov(b,n,k,f)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                     end if
                     ! (ik)(bc)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3_excits_buff(3,jdet); n = t3_excits_buff(6,jdet);
                        ! compute < ijkabc | h2b(voov) | jkn~acf~ >
                        hmatel = -h2b_voov(b,n,i,f)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                     end if
                     ! (jk)(bc)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3_excits_buff(3,jdet); n = t3_excits_buff(6,jdet);
                        ! compute < ijkabc | h2b(voov) | ikn~acf~ >
                        hmatel = h2b_voov(b,n,j,f)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                     end if
                  end do   
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3a_excits,&
                  !$omp t2a,&
                  !$omp I2A_vvov,I2A_vooo,&
                  !$omp fA_oo,fA_vv,&
                  !$omp noa,nua,n3aaa),&
                  !$omp private(idet,a,b,c,d,i,j,k,l,m,n,e,f,&
                  !$omp res_mm23,denom,shift)
                  !$omp do schedule(static)
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
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!

                  ! Update t3a in SIMD; make sure resid and t3a_amps are aligned!
                  t3a_amps = t3a_amps + resid

              end subroutine update_t3a_p

              subroutine update_t3b_p(resid,&
                                      t3a_amps, t3a_excits,&
                                      t3b_amps, t3b_excits,& 
                                      t3c_amps, t3c_excits,&
                                      t2a, t2b,&
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
                  integer, intent(in) :: t3a_excits(6,n3aaa), t3c_excits(6,n3abb)
                  real(kind=8), intent(in) :: t2a(1:nua,1:nua,1:noa,1:noa),&
                                              t2b(1:nua,1:nub,1:noa,1:nob),&
                                              t3a_amps(n3aaa),&
                                              t3c_amps(n3abb),&
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

                  integer, intent(inout) :: t3b_excits(6,n3aab)
                  !f2py intent(in,out) :: t3b_excits(0:5,0:n3aab-1)
                  real(kind=8), intent(inout) :: t3b_amps(n3aab)
                  !f2py intent(in,out) :: t3b_amps(0:n3aab-1)

                  real(kind=8), intent(out) :: resid(n3aab)

                  real(kind=8), allocatable :: t3_amps_buff(:)
                  integer, allocatable :: t3_excits_buff(:,:)

                  integer, allocatable :: loc_arr(:,:)
                  integer, allocatable :: idx_table(:,:,:,:)

                  real(kind=8) :: I2A_vooo(nua, noa, noa, noa),&
                                  I2A_vvov(nua, nua, noa, nua),&
                                  I2B_vooo(nua, nob, noa, nob),&
                                  I2B_ovoo(noa, nub, noa, nob),&
                                  I2B_vvov(nua, nub, noa, nub),&
                                  I2B_vvvo(nua, nub, nua, nob)
                  real(kind=8) :: denom, val, t_amp, res_mm23, hmatel
                  integer :: i, j, k, l, a, b, c, d, m, n, e, f, idet, jdet
                  integer :: idx

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

                  ! Zero the residual container
                  resid = 0.0d0

                  !!!! diagram 1: -A(i/jk) h1a(mi)*t3b(abcmjk)    
                  !!!! diagram 5: A(i/jk) 1/2 h2a(mnij)*t3b(abcmnk)
                  !!! ABCK LOOP !!! 
                  ! allocate new sorting arrays
                  allocate(loc_arr(nua*(nua-1)/2*nub*nob,2))
                  allocate(idx_table(nua,nua,nub,noa))
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/1,nob/), nua, nua, nub, noa)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/1,2,3,6/), nua, nua, nub, noa, nua*(nua-1)/2*nub*nob, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,&
                  !$omp t3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_oo,H2A_oooo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     idx = idx_table(a,b,c,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        l = t3b_excits(4,jdet); m = t3b_excits(5,jdet);
                        ! compute < ijk~abc~ | h2a(oooo) | lmk~abc~ >
                        hmatel = h2a_oooo(l,m,i,j)
                        ! compute < ijk~abc~ | h1a(vv) | lmk~abc~ > = -A(ij)A(lm) h1a_oo(l,i) * delta(m,j)
                        if (m==j) hmatel = hmatel - h1a_oo(l,i)
                        if (m==i) hmatel = hmatel + h1a_oo(l,j)
                        if (l==j) hmatel = hmatel + h1a_oo(m,i)
                        if (l==i) hmatel = hmatel - h1a_oo(m,j)
                        resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 2: A(a/bc) h1a(ae)*t3b(ebcmjk)
                  !!!! diagram 6: A(a/bc) 1/2 h2a(abef)*t3b(ebcmjk)
                  !!! CIJK LOOP !!!
                  ! allocate new sorting arrays
                  allocate(loc_arr(nub*noa*(noa-1)/2*nob,2))
                  allocate(idx_table(noa,noa,nob,nub))
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/1,nub/), noa, noa, nob, nub)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/4,5,6,3/), noa, noa, nob, nub, nub*noa*(noa-1)/2*nob, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,&
                  !$omp t3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_vv,H2A_vvvv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     !idx = idx_table(c,i,j,k)
                     idx = idx_table(i,j,k,c)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3b_excits(1,jdet); e = t3b_excits(2,jdet);
                        ! compute < ijk~abc~ | h2a(vvvv) | ijk~dec~ >
                        hmatel = h2a_vvvv(a,b,d,e)
                        ! compute < ijk~abc~ | h1a(vv) | ijk~dec > = A(ab)A(de) h1a_vv(a,d)*delta(b,e)
                        if (b==e) hmatel = hmatel + h1a_vv(a,d)
                        if (a==e) hmatel = hmatel - h1a_vv(b,d)
                        if (b==d) hmatel = hmatel - h1a_vv(a,e)
                        if (a==d) hmatel = hmatel + h1a_vv(b,e)
                        resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 3: -h1b(mk)*t3b(abcijm)
                  !!!! diagram 7: A(ij) h2b(mnjk)*t3b(abcimn)
                  !!! ABCI LOOP !!!
                  ! allocate new sorting arrays
                  allocate(loc_arr(nua*(nua-1)/2*nub*noa,2))
                  allocate(idx_table(nua,nua,nub,noa))
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/1,noa-1/), nua, nua, nub, noa)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/1,2,3,4/), nua, nua, nub, noa, nua*(nua-1)/2*nub*noa, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,&
                  !$omp t3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1B_oo,H2B_oooo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,i)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        m = t3b_excits(5,jdet); n = t3b_excits(6,jdet);
                        ! compute < ijk~abc~ | h2b(oooo) | imn~abc~ >
                        hmatel = h2b_oooo(m,n,j,k)
                        ! compute < ijk~abc~ | h1b(oo) | imn~abc~ >
                        if (m==j) hmatel = hmatel - h1b_oo(n,k)
                        resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           m = t3b_excits(5,jdet); n = t3b_excits(6,jdet);
                           ! compute < ijk~abc~ | h2b(oooo) | jmn~abc~ >
                           hmatel = -h2b_oooo(m,n,i,k)
                           ! compute < ijk~abc~ | h1b(oo) | jmn~abc~ >
                           if (m==i) hmatel = hmatel + h1b_oo(n,k)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if    
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABCJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/2,noa/), nua, nua, nub, noa)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/1,2,3,5/), nua, nua, nub, noa, nua*(nua-1)/2*nub*noa, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,&
                  !$omp t3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_oooo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,j)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        l = t3b_excits(4,jdet); n = t3b_excits(6,jdet);
                        ! compute < ijk~abc~ | h2b(oooo) | ljn~abc~ >
                        hmatel = h2b_oooo(l,n,i,k)
                        resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l = t3b_excits(4,jdet); n = t3b_excits(6,jdet);
                           ! compute < ijk~abc~ | h2b(oooo) | lin~abc~ >
                           hmatel = -h2b_oooo(l,n,j,k)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECITON !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table) 

                  !!!! diagram 9: A(ij)A(ab) h2a(amie)*t3b(ebcmjk)
                  ! allocate new sorting arrays
                  allocate(loc_arr(nua*nub*noa*nob,2))
                  allocate(idx_table(nua,nub,noa,nob))
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/2,nua/), (/1,nub/), (/2,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/2,3,5,6/), nua, nub, noa, nob, nua*nub*noa*nob, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,&
                  !$omp t3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,j,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3b_excits(1,jdet); l = t3b_excits(4,jdet);
                        ! compute < ijk~abc~ | h2a(voov) | ljk~dbc~ >
                        hmatel = h2a_voov(a,l,i,d)
                        resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then ! protect against case where a = 1 because b = 2, nua
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           d = t3b_excits(1,jdet); l = t3b_excits(4,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | ljk~dac~ >
                           hmatel = -h2a_voov(b,l,i,d)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if 
                     ! (ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then ! protect against case where i = 1 because j = 2, noa
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           d = t3b_excits(1,jdet); l = t3b_excits(4,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | lik~dbc~ >
                           hmatel = -h2a_voov(a,l,j,d)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if 
                     ! (ij)(ab)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then ! protect against case where a = 1 because b = 2, nua and i = 1 because j = 2, noa
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           d = t3b_excits(1,jdet); l = t3b_excits(4,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | lik~dac~ >
                           hmatel = h2a_voov(b,l,j,d)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if 
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/2,nua/), (/1,nub/), (/1,noa-1/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/2,3,4,6/), nua, nub, noa, nob, nua*nub*noa*nob, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,&
                  !$omp t3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,i,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3b_excits(1,jdet); l = t3b_excits(5,jdet);
                        ! compute < ijk~abc~ | h2a(voov) | ilk~dbc~ >
                        hmatel = h2a_voov(a,l,j,d)
                        resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then ! protect against where j = noa because i = 1, noa-1 
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           d = t3b_excits(1,jdet); l = t3b_excits(5,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | jlk~dbc~ >
                           hmatel = -h2a_voov(a,l,i,d)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then ! protect against case where a = 1 because b = 2, nua
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           d = t3b_excits(1,jdet); l = t3b_excits(5,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | ilk~dac~ >
                           hmatel = -h2a_voov(b,l,j,d)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then ! protect against case where j = noa because i = 1, noa-1 and where a = 1 because b = 2, nua
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           d = t3b_excits(1,jdet); l = t3b_excits(5,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | jlk~dac~ >
                           hmatel = h2a_voov(b,l,i,d)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/1,nub/), (/1,noa-1/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/1,3,4,6/), nua, nub, noa, nob, nua*nub*noa*nob, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,&
                  !$omp t3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3b_excits(2,jdet); l = t3b_excits(5,jdet);
                        ! compute < ijk~abc~ | h2a(voov) | ilk~adc~  >
                        hmatel = h2a_voov(b,l,j,d)
                        resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           d = t3b_excits(2,jdet); l = t3b_excits(5,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | jlk~adc~  >
                           hmatel = -h2a_voov(b,l,i,d)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           d = t3b_excits(2,jdet); l = t3b_excits(5,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | ilk~bdc~  >
                           hmatel = -h2a_voov(a,l,j,d)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           d = t3b_excits(2,jdet); l = t3b_excits(5,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | jlk~bdc~  >
                           hmatel = h2a_voov(a,l,i,d)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/1,nub/), (/2,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/1,3,5,6/), nua, nub, noa, nob, nua*nub*noa*nob, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,&
                  !$omp t3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,j,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3b_excits(2,jdet); l = t3b_excits(4,jdet);
                        ! compute < ijk~abc~ | h2a(voov) | ljk~adc~  >
                        hmatel = h2a_voov(b,l,i,d)
                        resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           d = t3b_excits(2,jdet); l = t3b_excits(4,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | lik~adc~  >
                           hmatel = -h2a_voov(b,l,j,d)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           d = t3b_excits(2,jdet); l = t3b_excits(4,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | ljk~bdc~  >
                           hmatel = -h2a_voov(a,l,i,d)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           d = t3b_excits(2,jdet); l = t3b_excits(4,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | lik~abc~  >
                           hmatel = h2a_voov(a,l,j,d)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 5: h1b(ce)*t3b(abeijm)
                  !!!! diagram 8: A(ab) h2b(bcef)*t3b(aefijk)
                  ! allocate new sorting arrays
                  allocate(loc_arr(nua*noa*(noa-1)/2*nob,2))
                  allocate(idx_table(noa,noa,nob,nua))
                  !!! AIJK LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/1,nua-1/), noa, noa, nob, nua)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/4,5,6,1/), noa, noa, nob, nua, nua*noa*(noa-1)/2*nob, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,&
                  !$omp t3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1B_vv,H2B_vvvv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                      a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                      ! (1)
                      idx = idx_table(i,j,k,a)
                      do jdet = loc_arr(idx,1), loc_arr(idx,2)
                         d = t3b_excits(1,jdet); e = t3b_excits(2,jdet); f = t3b_excits(3,jdet);
                         l = t3b_excits(4,jdet); m = t3b_excits(5,jdet); n = t3b_excits(6,jdet);
                         ! compute < ijk~abc~ | h2b(vvvv) | ijk~aef~ >
                         hmatel = h2b_vvvv(b,c,e,f)
                         if (b==e) hmatel = hmatel + h1b_vv(c,f)
                         resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                      end do
                      ! (ab)
                      idx = idx_table(i,j,k,b)
                      if (idx/=0) then ! protect against case where b = nua because a = 1, nua-1
                         do jdet = loc_arr(idx,1), loc_arr(idx,2)
                            d = t3b_excits(1,jdet); e = t3b_excits(2,jdet); f = t3b_excits(3,jdet);
                            l = t3b_excits(4,jdet); m = t3b_excits(5,jdet); n = t3b_excits(6,jdet);
                            ! compute < ijk~abc~ | h2b(vvvv) | ijk~bef~ >
                            hmatel = -h2b_vvvv(a,c,e,f)
                            if (a==e) hmatel = hmatel - h1b_vv(c,f)
                            resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                         end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BIJK LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/2,nua/), noa, noa, nob, nua)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/4,5,6,2/), noa, noa, nob, nua, nua*noa*(noa-1)/2*nob, n3aab, resid)
                  !!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,&
                  !$omp t3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_vvvv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                      a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                      idx = idx_table(i,j,k,b)
                      do jdet = loc_arr(idx,1), loc_arr(idx,2)
                         d = t3b_excits(1,jdet); e = t3b_excits(2,jdet); f = t3b_excits(3,jdet);
                         l = t3b_excits(4,jdet); m = t3b_excits(5,jdet); n = t3b_excits(6,jdet);
                         ! compute < ijk~abc~ | h2b(vvvv) | ijk~dbf~ >
                         hmatel = h2b_vvvv(a,c,d,f)
                         resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                      end do
                      idx = idx_table(i,j,k,a)
                      if (idx/=0) then ! protect against case where a = 1 because b = 2, nua
                         do jdet = loc_arr(idx,1), loc_arr(idx,2)
                            d = t3b_excits(1,jdet); e = t3b_excits(2,jdet); f = t3b_excits(3,jdet);
                            l = t3b_excits(4,jdet); m = t3b_excits(5,jdet); n = t3b_excits(6,jdet);
                            ! compute < ijk~abc~ | h2b(vvvv) | ijk~daf~ >
                            hmatel = -h2b_vvvv(b,c,d,f)
                            resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                         end do
                      end if
                  end do ! end loop over idet
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 10: h2c(cmke)*t3b(abeijm)
                  ! allocate sorting arrays
                  allocate(loc_arr(nua*(nua-1)/2*noa*(noa-1)/2,2))
                  allocate(idx_table(nua,nua,noa,noa))
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/1,2,4,5/), nua, nua, noa, noa, nua*(nua-1)/2*noa*(noa-1)/2, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,&
                  !$omp t3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,&
                  !$omp n3aab),&
                  !$omp private(hmatel,a,b,c,i,j,k,f,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                      a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                      idx = idx_table(a,b,i,j)
                      do jdet = loc_arr(idx,1), loc_arr(idx,2)
                         f = t3b_excits(3,jdet); n = t3b_excits(6,jdet);
                         ! compute < ijk~abc~ | h2c(voov) | ijn~abf~ > = h2c_voov(c,n,k,f)
                         hmatel = h2c_voov(c,n,k,f)
                         resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                      end do
                  end do ! end loop over idet
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 11: -A(ij) h2b(mcie)*t3b(abemjk)
                  ! allocate sorting arrays
                  allocate(loc_arr(nua*(nua-1)/2*noa*nob,2)) 
                  allocate(idx_table(nua,nua,noa,nob))
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,noa-1/), (/1,nob/), nua, nua, noa, nob)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/1,2,4,6/), nua, nua, noa, nob, nua*(nua-1)/2*noa*nob, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,&
                  !$omp t3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,i,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3b_excits(3,jdet); m = t3b_excits(5,jdet);
                        ! compute < ijk~abc~ | h2b(ovov) | imk~abf~ >
                        hmatel = -h2b_ovov(m,c,j,f)
                        resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           f = t3b_excits(3,jdet); m = t3b_excits(5,jdet);
                           ! compute < ijk~abc~ | h2b(ovov) | jmk~abf~ >
                           hmatel = h2b_ovov(m,c,i,f)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/2,noa/), (/1,nob/), nua, nua, noa, nob)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/1,2,5,6/), nua, nua, noa, nob, nua*(nua-1)/2*noa*nob, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,&
                  !$omp t3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,j,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3b_excits(3,jdet); l = t3b_excits(4,jdet);
                        ! compute < ijk~abc~ | h2b(ovov) | ljk~abf~ >
                        hmatel = -h2b_ovov(l,c,i,f)
                        resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           f = t3b_excits(3,jdet); l = t3b_excits(4,jdet);
                           ! compute < ijk~abc~ | h2b(ovov) | lik~abf~ >
                           hmatel = h2b_ovov(l,c,j,f)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  
                  !!!! diagram 12: -A(ab) h2b(amek)*t3b(ebcijm)
                  ! allocate sorting arrays
                  allocate(loc_arr(nua*nub*noa*(noa-1)/2,2)) 
                  allocate(idx_table(noa,noa,nua,nub))
                  !!! BCIJ LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/2,nua/), (/1,nub/), noa, noa, nua, nub)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/4,5,2,3/), noa, noa, nua, nub, nua*nub*noa*(noa-1)/2, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,&
                  !$omp t3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_vovo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,b,c)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3b_excits(1,jdet); n = t3b_excits(6,jdet);
                        ! compute < ijk~abc~ | h2b(vovo) | ijn~dbc~ >
                        hmatel = -h2b_vovo(a,n,d,k)
                        resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           d = t3b_excits(1,jdet); n = t3b_excits(6,jdet);
                           ! compute < ijk~abc~ | h2b(vovo) | ijn~dac~ >
                           hmatel = h2b_vovo(b,n,d,k)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nua-1/), (/1,nub/), noa, noa, nua, nub)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/4,5,1,3/), noa, noa, nua, nub, nua*nub*noa*(noa-1)/2, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,&
                  !$omp t3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_vovo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,a,c)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3b_excits(2,jdet); n = t3b_excits(6,jdet);
                        ! compute < ijk~abc~ | h2b(vovo) | ijn~aec~ >
                        hmatel = -h2b_vovo(b,n,e,k)
                        resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           e = t3b_excits(2,jdet); n = t3b_excits(6,jdet);
                           ! compute < ijk~abc~ | h2b(vovo) | ijn~bec~ >
                           hmatel = h2b_vovo(a,n,e,k)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 13: h2b(mcek)*t3a(abeijm) !!!!
                  ! allocate and initialize the copy of t3a
                  allocate(t3_amps_buff(n3aaa))
                  allocate(t3_excits_buff(6,n3aaa))
                  t3_amps_buff(:) = t3a_amps(:)
                  t3_excits_buff(:,:) = t3a_excits(:,:)
                  ! allocate sorting arrays (can be reused for each permutation)
                  allocate(loc_arr((nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2,2))
                  allocate(idx_table(nua,nua,noa,noa))
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,2,4,5/), nua, nua, noa, noa, (nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,t3_excits_buff,&
                  !$omp t3b_amps,t3_amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     idx = idx_table(a,b,i,j)
                     if (idx==0) cycle 
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3_excits_buff(3,jdet); n = t3_excits_buff(6,jdet);
                        ! compute < ijk~abc~ | h2b(ovvo) | ijnabf >
                        hmatel = h2b_ovvo(n,c,f,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,3,4,5/), nua, nua, noa, noa, (nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,t3_excits_buff,&
                  !$omp t3b_amps,t3_amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3_excits_buff(2,jdet); n = t3_excits_buff(6,jdet);
                        ! compute < ijk~abc~ | h2b(ovvo) | ijnaeb >
                        hmatel = -h2b_ovvo(n,c,e,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCIJ LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/2,3,4,5/), nua, nua, noa, noa, (nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,t3_excits_buff,&
                  !$omp t3b_amps,t3_amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3_excits_buff(1,jdet); n = t3_excits_buff(6,jdet);
                        ! compute < ijk~abc~ | h2b(ovvo) | ijndab >
                        hmatel = h2b_ovvo(n,c,d,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,2,4,6/), nua, nua, noa, noa, (nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,t3_excits_buff,&
                  !$omp t3b_amps,t3_amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3_excits_buff(3,jdet); m = t3_excits_buff(5,jdet);
                        ! compute < ijk~abc~ | h2b(ovvo) | imjabf >
                        hmatel = -h2b_ovvo(m,c,f,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,3,4,6/), nua, nua, noa, noa, (nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,t3_excits_buff,&
                  !$omp t3b_amps,t3_amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3_excits_buff(2,jdet); m = t3_excits_buff(5,jdet);
                        ! compute < ijk~abc~ | h2b(ovvo) | imjaeb >
                        hmatel = h2b_ovvo(m,c,e,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/2,3,4,6/), nua, nua, noa, noa, (nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,t3_excits_buff,&
                  !$omp t3b_amps,t3_amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3_excits_buff(1,jdet); m = t3_excits_buff(5,jdet);
                        ! compute < ijk~abc~ | h2b(ovvo) | imjdab >
                        hmatel = -h2b_ovvo(m,c,d,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,2,5,6/), nua, nua, noa, noa, (nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,t3_excits_buff,&
                  !$omp t3b_amps,t3_amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = t3_excits_buff(3,jdet); l = t3_excits_buff(4,jdet);
                        ! compute < ijk~abc~ | h2b(ovvo) | lijabf >
                        hmatel = h2b_ovvo(l,c,f,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,3,5,6/), nua, nua, noa, noa, (nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,t3_excits_buff,&
                  !$omp t3b_amps,t3_amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = t3_excits_buff(2,jdet); l = t3_excits_buff(4,jdet);
                        ! compute < ijk~abc~ | h2b(ovvo) | lijaeb >
                        hmatel = -h2b_ovvo(l,c,e,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/2,3,5,6/), nua, nua, noa, noa, (nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,t3_excits_buff,&
                  !$omp t3b_amps,t3_amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = t3_excits_buff(1,jdet); l = t3_excits_buff(4,jdet);
                        ! compute < ijk~abc~ | h2b(ovvo) | lijdab >
                        hmatel = h2b_ovvo(l,c,d,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate t3 buffer arrays
                  deallocate(t3_amps_buff,t3_excits_buff) 

                  !!!! diagram 14: A(ab)A(ij) h2b(bmje)*t3c(aecimk)
                  ! allocate and initialize the copy of t3a
                  allocate(t3_amps_buff(n3abb))
                  allocate(t3_excits_buff(6,n3abb))
                  t3_amps_buff(:) = t3c_amps(:)
                  t3_excits_buff(:,:) = t3c_excits(:,:)
                  ! allocate sorting arrays (can be reused for each permutation)
                  allocate(loc_arr(nua*nub*noa*nob,2))
                  allocate(idx_table(nua,nub,noa,nob))
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/2,nub/), (/1,noa/), (/2,nob/), nua, nub, noa, nob)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,3,4,6/), nua, nub, noa, nob, nua*nub*noa*nob, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,t3_excits_buff,&
                  !$omp t3b_amps,t3_amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           e = t3_excits_buff(2,jdet); m = t3_excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | im~k~ae~c~ >
                           hmatel = h2b_voov(b,m,j,e)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           e = t3_excits_buff(2,jdet); m = t3_excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | im~k~be~c~ >
                           hmatel = -h2b_voov(a,m,j,e)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           e = t3_excits_buff(2,jdet); m = t3_excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | jm~k~ae~c~ >
                           hmatel = -h2b_voov(b,m,i,e)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           e = t3_excits_buff(2,jdet); m = t3_excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | jm~k~be~c~ >
                           hmatel = h2b_voov(a,m,i,e)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/1,nub-1/), (/1,noa/), (/2,nob/), nua, nub, noa, nob)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,2,4,6/), nua, nub, noa, nob, nua*nub*noa*nob, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,t3_excits_buff,&
                  !$omp t3b_amps,t3_amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           f = t3_excits_buff(3,jdet); m = t3_excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | im~k~ac~f~ >
                           hmatel = -h2b_voov(b,m,j,f)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           f = t3_excits_buff(3,jdet); m = t3_excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | im~k~bc~f~ >
                           hmatel = h2b_voov(a,m,j,f)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           f = t3_excits_buff(3,jdet); m = t3_excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | jm~k~ac~f~ >
                           hmatel = h2b_voov(b,m,i,f)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           f = t3_excits_buff(3,jdet); m = t3_excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | jm~k~bc~f~ >
                           hmatel = -h2b_voov(a,m,i,f)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/2,nub/), (/1,noa/), (/1,nob-1/), nua, nub, noa, nob)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,3,4,5/), nua, nub, noa, nob, nua*nub*noa*nob, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,t3_excits_buff,&
                  !$omp t3b_amps,t3_amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           e = t3_excits_buff(2,jdet); n = t3_excits_buff(6,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | ik~n~ae~c~ >
                           hmatel = -h2b_voov(b,n,j,e)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           e = t3_excits_buff(2,jdet); n = t3_excits_buff(6,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | ik~n~be~c~ >
                           hmatel = h2b_voov(a,n,j,e)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           e = t3_excits_buff(2,jdet); n = t3_excits_buff(6,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | jk~n~ae~c~ >
                           hmatel = h2b_voov(b,n,i,e)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           e = t3_excits_buff(2,jdet); n = t3_excits_buff(6,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | jk~n~be~c~ >
                           hmatel = -h2b_voov(a,n,i,e)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/1,nub-1/), (/1,noa/), (/1,nob-1/), nua, nub, noa, nob)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,2,4,5/), nua, nub, noa, nob, nua*nub*noa*nob, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,t3_excits_buff,&
                  !$omp t3b_amps,t3_amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           f = t3_excits_buff(3,jdet); n = t3_excits_buff(6,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | ik~n~ac~f~ >
                           hmatel = h2b_voov(b,n,j,f)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           f = t3_excits_buff(3,jdet); n = t3_excits_buff(6,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | ik~n~bc~f~ >
                           hmatel = -h2b_voov(a,n,j,f)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           f = t3_excits_buff(3,jdet); n = t3_excits_buff(6,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | jk~n~ac~f~ >
                           hmatel = -h2b_voov(b,n,i,f)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           f = t3_excits_buff(3,jdet); n = t3_excits_buff(6,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | jk~n~bc~f~ >
                           hmatel = h2b_voov(a,n,i,f)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate t3 buffer arrays
                  deallocate(t3_amps_buff,t3_excits_buff) 

                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3b_excits,&
                  !$omp t2a,t2b,&
                  !$omp I2A_vvov,I2A_vooo,I2B_vvvo,I2B_vvov,I2B_vooo,I2B_ovoo,&
                  !$omp fA_oo,fB_oo,fA_vv,fB_vv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(idet,a,b,c,d,i,j,k,l,m,n,e,f,&
                  !$omp res_mm23,denom,shift)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                      a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);

                      ! Add MM(2,3) contribution and get final residual
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
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!

                  ! Update t3b in SIMD; make sure resid and t3b_amps are aligned!
                  t3b_amps = t3b_amps + resid

              end subroutine update_t3b_p

              subroutine update_t3c_p(resid,&
                                      t3b_amps, t3b_excits,&
                                      t3c_amps, t3c_excits,&
                                      t3d_amps, t3d_excits,&
                                      t2b, t2c,&
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
                  integer, intent(in) :: t3b_excits(6,n3aab), t3d_excits(6,n3bbb)
                  real(kind=8), intent(in) :: t2b(1:nua,1:nub,1:noa,1:nob),&
                                              t2c(1:nub,1:nub,1:nob,1:nob),&
                                              t3b_amps(n3aab),t3d_amps(n3bbb),&
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

                  integer, intent(inout) :: t3c_excits(6,n3abb)
                  !f2py intent(in,out) :: t3c_excits(0:5,0:n3abb-1)
                  real(kind=8), intent(inout) :: t3c_amps(n3abb)
                  !f2py intent(in,out) :: t3c_amps(0:n3abb-1)

                  real(kind=8), intent(out) :: resid(n3abb)

                  real(kind=8), allocatable :: t3_amps_buff(:)
                  integer, allocatable :: t3_excits_buff(:,:)

                  integer, allocatable :: id3b_h(:,:,:)
                  integer, allocatable :: eck_table(:,:)
                  integer, allocatable :: xixj_table(:,:)

                  integer, allocatable :: id3c_h(:,:,:)
                  integer, allocatable :: eai_table(:,:)
                  integer, allocatable :: xjxk_table(:,:)

                  integer, allocatable :: id3d_h(:,:)
                  integer, allocatable :: xixjxk_table(:,:,:)

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

                  ! Perform the master sort of t3c
                  allocate(id3c_h(nua*noa,nob*(nob-1)/2,2))
                  allocate(xjxk_table(nob,nob))
                  allocate(eai_table(nua,noa))
                  call sort_t3c_h(t3c_excits, t3c_amps, id3c_h, eai_table, xjxk_table, noa, nua, nob, nub, n3abb, resid)

                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3c_excits,&
                  !$omp t3c_amps,t2b,t2c,&
                  !$omp id3c_h,eai_table,xjxk_table,&
                  !$omp H1A_oo,H1B_oo,H1A_vv,H1B_vv,H2B_oooo,H2C_oooo,&
                  !$omp H2B_ovvo,H2B_voov,H2C_vvvv,H2B_vvvv,&
                  !$omp H2A_voov,H2C_voov,H2B_ovov,H2B_vovo,&
                  !$omp I2C_vooo,I2C_vvov,I2B_vooo,I2B_ovoo,&
                  !$omp I2B_vvov,I2B_vvvo,&
                  !$omp fA_oo,fB_oo,fA_vv,fB_vv,nua,nub,noa,nob,&
                  !$omp shift,n3aab,n3abb,n3bbb),&
                  !$omp private(a,b,c,d,i,j,k,l,m,n,e,f,denom,t_amp,hmatel,res_mm23,idet,jdet,&
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
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(id3c_h,eai_table,xjxk_table)

                  ! Update t3b in SIMD; make sure resid and t3b_amps are aligned!
                  t3c_amps = t3c_amps + resid

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
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!! SORTING FUNCTIONS !!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine get_index_table(idx_table, rng1, rng2, rng3, rng4, n1, n2, n3, n4)

              integer, intent(in) :: n1, n2, n3, n4
              integer, intent(in) :: rng1(2), rng2(2), rng3(2), rng4(2)

              integer, intent(inout) :: idx_table(n1,n2,n3,n4)

              integer :: kout
              integer :: p, q, r, s

              idx_table = 0
              ! 16 possible cases
              if (rng1(1) < 0 .and. rng2(1) < 0 .and. rng3(1) < 0 .and. rng4(1) < 0) then ! p < q < r < s
                 kout = 1 
                 do p = rng1(1), rng1(2)
                    do q = p-rng2(1), rng2(2)
                       do r = q-rng3(1), rng3(2)
                          do s = r-rng4(1), rng4(2)
                             idx_table(p,q,r,s) = kout
                             kout = kout + 1
                          end do
                       end do
                    end do
                 end do
              elseif (rng1(1) > 0 .and. rng2(1) < 0 .and. rng3(1) < 0 .and. rng4(1) > 0) then ! p < q < r, s
                 kout = 1 
                 do p = rng1(1), rng1(2)
                    do q = p-rng2(1), rng2(2)
                       do r = q-rng3(1), rng3(2)
                          do s = rng4(1), rng4(2)
                             idx_table(p,q,r,s) = kout
                             kout = kout + 1
                          end do
                       end do
                    end do
                 end do
              elseif (rng1(1) > 0 .and. rng2(1) < 0 .and. rng3(1) > 0 .and. rng4(1) < 0) then ! p < q, r < s
                 kout = 1 
                 do p = rng1(1), rng1(2)
                    do q = p-rng2(1), rng2(2)
                       do r = rng3(1), rng3(2)
                          do s = r-rng4(1), rng4(2)
                             idx_table(p,q,r,s) = kout
                             kout = kout + 1
                          end do
                       end do
                    end do
                 end do
              elseif (rng1(1) > 0 .and. rng2(1) < 0 .and. rng3(1) > 0 .and. rng4(1) > 0) then ! p < q, r, s
                 kout = 1 
                 do p = rng1(1), rng1(2)
                    do q = p-rng2(1), rng2(2)
                       do r = rng3(1), rng3(2)
                          do s = rng4(1), rng4(2)
                             idx_table(p,q,r,s) = kout
                             kout = kout + 1
                          end do
                       end do
                    end do
                 end do
              else ! p, q, r, s
                 kout = 1 
                 do p = rng1(1), rng1(2)
                    do q = rng2(1), rng2(2)
                       do r = rng3(1), rng3(2)
                          do s = rng4(1), rng4(2)
                             idx_table(p,q,r,s) = kout
                             kout = kout + 1
                          end do
                       end do
                    end do
                 end do
              end if

      end subroutine get_index_table

      subroutine sort4(excits, amps, loc_arr, idx_table, idims, n1, n2, n3, n4, nloc, n3p, resid)

              integer, intent(in) :: n1, n2, n3, n4, nloc, n3p
              integer, intent(in) :: idims(4)
              integer, intent(in) :: idx_table(n1,n2,n3,n4)

              integer, intent(inout) :: loc_arr(nloc,2)
              integer, intent(inout) :: excits(6,n3p)
              real(kind=8), intent(inout) :: amps(n3p)
              real(kind=8), intent(inout), optional :: resid(n3p)

              integer :: idet
              integer :: p, q, r, s
              integer :: p1, q1, r1, s1, p2, q2, r2, s2
              integer :: pqrs1, pqrs2
              integer, allocatable :: temp(:), idx(:)

              allocate(temp(n3p),idx(n3p))
              do idet = 1, n3p
                 p = excits(idims(1),idet); q = excits(idims(2),idet); r = excits(idims(3),idet); s = excits(idims(4),idet)
                 temp(idet) = idx_table(p,q,r,s)
              end do
              call argsort(temp, idx)
              excits = excits(:,idx)
              amps = amps(idx)
              if (present(resid)) resid = resid(idx)
              deallocate(temp,idx)

              loc_arr(:,1) = 1; loc_arr(:,2) = 0;
              do idet = 1, n3p-1
                 p1 = excits(idims(1),idet);   q1 = excits(idims(2),idet);   r1 = excits(idims(3),idet);   s1 = excits(idims(4),idet)
                 p2 = excits(idims(1),idet+1); q2 = excits(idims(2),idet+1); r2 = excits(idims(3),idet+1); s2 = excits(idims(4),idet+1)
                 pqrs1 = idx_table(p1,q1,r1,s1)
                 pqrs2 = idx_table(p2,q2,r2,s2)
                 if (pqrs1 /= pqrs2) then
                    loc_arr(pqrs1,2) = idet
                    loc_arr(pqrs2,1) = idet+1
                 end if
              end do
              loc_arr(pqrs2,2) = n3p

      end subroutine sort4
               
              


              subroutine sort_t3a_h(t3a_excits, t3a_amps, ID, XiXjXk_table, noa, nua, n3aaa, resid)

                      integer, intent(in) :: n3aaa, noa, nua

                      integer, intent(inout) :: t3a_excits(6,n3aaa)
                      real(kind=8), intent(inout) :: t3a_amps(n3aaa)
                      real(kind=8), optional, intent(inout) :: resid(n3aaa)
                      integer, intent(inout) :: XiXjXk_table(noa,noa,noa)
                      integer, intent(inout) :: ID(noa*(noa-1)*(noa-2)/6,2)

                      integer :: i, j, k, a, b, c
                      integer :: i1, j1, k1, a1, b1, c1
                      integer :: i2, j2, k2, a2, b2, c2
                      integer :: kout, ijk, ijk1, ijk2, idet
                      integer, allocatable :: temp(:), idx(:)

                      XiXjXk_table = 0
                      kout = 1
                      do i = 1, noa
                         do j = i+1, noa
                            do k = j+1, noa
                               XiXjXk_table(i,j,k) = kout
                               XiXjXk_table(j,k,i) = kout
                               XiXjXk_table(k,i,j) = kout
                               XiXjXk_table(i,k,j) = -kout
                               XiXjXk_table(j,i,k) = -kout
                               XiXjXk_table(k,j,i) = -kout
                               kout = kout + 1
                            end do
                         end do
                      end do

                      allocate(temp(n3aaa),idx(n3aaa))
                      do idet = 1, n3aaa
                         i = t3a_excits(4,idet); j = t3a_excits(5,idet); k = t3a_excits(6,idet);
                         ijk = XiXjXk_table(i,j,k)
                         temp(idet) = ijk
                      end do
                      call argsort(temp, idx)
                      t3a_excits = t3a_excits(:,idx)
                      t3a_amps = t3a_amps(idx)
                      if (present(resid)) resid = resid(idx)
                      deallocate(temp,idx)

                      ID = 1
                      do idet = 2, n3aaa
                         i1 = t3a_excits(4,idet-1); j1 = t3a_excits(5,idet-1); k1 = t3a_excits(6,idet-1);
                         i2 = t3a_excits(4,idet);   j2 = t3a_excits(5,idet);   k2 = t3a_excits(6,idet);

                         ijk1 = XiXjXk_table(i1,j1,k1)
                         ijk2 = XiXjXk_table(i2,j2,k2)
                         if (ijk1 /= ijk2) then
                                 ID(ijk1,2) = idet - 1
                                 ID(ijk2,1) = idet
                         end if
                      end do
                      ID(ijk2,2) = n3aaa

              end subroutine sort_t3a_h

              subroutine sort_t3b_h(t3b_excits, t3b_amps, ID, Eck_table, XiXj_table, noa, nua, nob, nub, n3aab, resid)

                      integer, intent(in) :: n3aab, noa, nua, nob, nub

                      integer, intent(inout) :: t3b_excits(6,n3aab)
                      real(kind=8), intent(inout) :: t3b_amps(n3aab)
                      real(kind=8), intent(inout), optional :: resid(n3aab)
                      integer, intent(inout) :: Eck_table(nub,nob)
                      integer, intent(inout) :: XiXj_table(noa,noa)
                      integer, intent(inout) :: ID(nub*nob,noa*(noa-1)/2,2)

                      integer :: i, j, k, a, b, c
                      integer :: i1, i2, j1, j2, c1, c2, k1, k2
                      integer:: ij, ib, ib1, ib2, ij1, ij2, kout, idet, num_ij_ib
                      integer :: beta_locs(nub*nob,2)
                      integer, allocatable :: temp(:), idx(:)

                      Eck_table=0
                      kout = 1
                      do c = 1, nub
                         do k = 1, nob
                            Eck_table(c,k) = kout
                            kout = kout + 1
                         end do
                      end do
                      XiXj_table=0
                      kout = 1
                      do i = 1, noa
                         do j = i+1, noa
                            XiXj_table(i,j) = kout
                            XiXj_table(j,i) = -kout
                            kout = kout + 1
                         end do
                      end do

                      allocate(temp(n3aab),idx(n3aab))
                      do idet = 1, n3aab
                         c = t3b_excits(3,idet); k = t3b_excits(6,idet);
                         ib = Eck_table(c,k)
                         temp(idet) = ib
                      end do
                      call argsort(temp, idx)
                      t3b_excits = t3b_excits(:,idx)
                      t3b_amps = t3b_amps(idx)
                      if (present(resid)) resid = resid(idx)
                      deallocate(temp,idx)

                      beta_locs = 1
                      do idet = 2, n3aab
                         c1 = t3b_excits(3,idet-1); k1 = t3b_excits(6,idet-1);
                         c2 = t3b_excits(3,idet);   k2 = t3b_excits(6,idet);
                         ib1 = Eck_table(c1,k1)
                         ib2 = Eck_table(c2,k2)
                         if (ib1/=ib2) then
                                 beta_locs(ib1,2) = idet - 1
                                 beta_locs(ib2,1) = idet
                         end if
                      end do
                      beta_locs(ib2,2) = n3aab

                      ID = 0
                      do c = 1,nub
                         do k = 1,nob
                            ib = Eck_table(c,k)
                            if (beta_locs(ib,1) > beta_locs(ib,2)) cycle ! skip if beta block is empty

                            num_ij_ib = beta_locs(ib,2) - beta_locs(ib,1) + 1

                            allocate(temp(num_ij_ib), idx(num_ij_ib))
                            kout = 1
                            do idet = beta_locs(ib,1), beta_locs(ib,2)
                               i = t3b_excits(4,idet); j = t3b_excits(5,idet);
                               ij = XiXj_table(i,j)
                               temp(kout) = ij
                               kout = kout + 1
                            end do
                            call argsort(temp,idx)
                            idx = idx + beta_locs(ib,1) - 1
                            t3b_excits(:,beta_locs(ib,1):beta_locs(ib,2)) = t3b_excits(:,idx)
                            t3b_amps(beta_locs(ib,1):beta_locs(ib,2)) = t3b_amps(idx)
                            if (present(resid)) resid(beta_locs(ib,1):beta_locs(ib,2)) = resid(idx)
                            deallocate(temp,idx)

                            ID(ib,:,1) = beta_locs(ib,1)
                            do ij = 1, num_ij_ib-1
                               idet = ij + beta_locs(ib,1)
                               i1 = t3b_excits(4,idet-1); j1 = t3b_excits(5,idet-1);
                               i2 = t3b_excits(4,idet);   j2 = t3b_excits(5,idet);
                               ij1 = XiXj_table(i1,j1)
                               ij2 = XiXj_table(i2,j2)
                               
                               if (ij1/=ij2) then
                                       ID(ib,ij1,2) = idet-1
                                       ID(ib,ij2,1) = idet
                               end if
                            end do
                            ID(ib,ij2,2) = beta_locs(ib,1) + num_ij_ib - 1
                         end do
                      end do

              end subroutine sort_t3b_h

              subroutine sort_t3b_p(t3b_excits, t3b_amps, ID, Eck_table, XaXb_table, noa, nua, nob, nub, n3aab, resid)

                      integer, intent(in) :: n3aab, noa, nua, nob, nub

                      integer, intent(inout) :: t3b_excits(6,n3aab)
                      real(kind=8), intent(inout) :: t3b_amps(n3aab)
                      real(kind=8), intent(inout), optional :: resid(n3aab)
                      integer, intent(inout) :: Eck_table(nub,nob)
                      integer, intent(inout) :: XaXb_table(nua,nua)
                      integer, intent(inout) :: ID(nub*nob,nua*(nua-1)/2,2)

                      integer :: i, j, k, a, b, c
                      integer :: a1, a2, b1, b2, c1, c2, k1, k2
                      integer:: ab, ib, ib1, ib2, ab1, ab2, kout, idet, num_ab_ib
                      integer :: beta_locs(nub*nob,2)
                      integer, allocatable :: temp(:), idx(:)

                      Eck_table=0
                      kout = 1
                      do c = 1, nub
                         do k = 1, nob
                            Eck_table(c,k) = kout
                            kout = kout + 1
                         end do
                      end do
                      XaXb_table=0
                      kout = 1
                      do a = 1, nua
                         do b = a+1, nua
                            XaXb_table(a,b) = kout
                            XaXb_table(b,a) = -kout
                            kout = kout + 1
                         end do
                      end do

                      allocate(temp(n3aab),idx(n3aab))
                      do idet = 1, n3aab
                         c = t3b_excits(3,idet); k = t3b_excits(6,idet);
                         ib = Eck_table(c,k)
                         temp(idet) = ib
                      end do
                      call argsort(temp, idx)
                      t3b_excits = t3b_excits(:,idx)
                      t3b_amps = t3b_amps(idx)
                      if (present(resid)) resid = resid(idx)
                      deallocate(temp,idx)

                      beta_locs = 1
                      do idet = 2, n3aab
                         c1 = t3b_excits(3,idet-1); k1 = t3b_excits(6,idet-1);
                         c2 = t3b_excits(3,idet);   k2 = t3b_excits(6,idet);
                         ib1 = Eck_table(c1,k1)
                         ib2 = Eck_table(c2,k2)
                         if (ib1/=ib2) then
                                 beta_locs(ib1,2) = idet - 1
                                 beta_locs(ib2,1) = idet
                         end if
                      end do
                      beta_locs(ib2,2) = n3aab

                      ID = 0
                      do c = 1,nub
                         do k = 1,nob
                            ib = Eck_table(c,k)
                            if (beta_locs(ib,1) > beta_locs(ib,2)) cycle ! skip if beta block is empty

                            num_ab_ib = beta_locs(ib,2) - beta_locs(ib,1) + 1

                            allocate(temp(num_ab_ib), idx(num_ab_ib))
                            kout = 1
                            do idet = beta_locs(ib,1), beta_locs(ib,2)
                               a = t3b_excits(1,idet); b = t3b_excits(2,idet);
                               ab = XaXb_table(a,b)
                               temp(kout) = ab
                               kout = kout + 1
                            end do
                            call argsort(temp,idx)
                            idx = idx + beta_locs(ib,1) - 1
                            t3b_excits(:,beta_locs(ib,1):beta_locs(ib,2)) = t3b_excits(:,idx)
                            t3b_amps(beta_locs(ib,1):beta_locs(ib,2)) = t3b_amps(idx)
                            if (present(resid)) resid(beta_locs(ib,1):beta_locs(ib,2)) = resid(idx)
                            deallocate(temp,idx)

                            ID(ib,:,1) = beta_locs(ib,1)
                            do ab = 1, num_ab_ib-1
                               idet = ab + beta_locs(ib,1)
                               a1 = t3b_excits(1,idet-1); b1 = t3b_excits(2,idet-1);
                               a2 = t3b_excits(1,idet);   b2 = t3b_excits(2,idet);
                               ab1 = XaXb_table(a1,b1)
                               ab2 = XaXb_table(a2,b2)
                               
                               if (ab1/=ab2) then
                                       ID(ib,ab1,2) = idet-1
                                       ID(ib,ab2,1) = idet
                               end if
                            end do
                            ID(ib,ab2,2) = beta_locs(ib,1) + num_ab_ib - 1
                         end do
                      end do

              end subroutine sort_t3b_p

              subroutine sort_t3c_h(t3c_excits, t3c_amps, ID, Eai_table, XjXk_table, noa, nua, nob, nub, n3abb, resid)

                      integer, intent(in) :: n3abb, noa, nua, nob, nub

                      integer, intent(inout) :: t3c_excits(6,n3abb)
                      real(kind=8), intent(inout) :: t3c_amps(n3abb)
                      real(kind=8), intent(inout), optional :: resid(n3abb)
                      integer, intent(inout) :: Eai_table(nua,noa)
                      integer, intent(inout) :: XjXk_table(nob,nob)
                      integer, intent(inout) :: ID(nua*noa,nob*(nob-1)/2,2)

                      integer :: i, j, k, a, b, c
                      integer :: j1, j2, k1, k2, a1, a2, i1, i2
                      integer:: jk, ia, ia1, ia2, jk1, jk2, kout, idet, num_jk_ia
                      integer :: alpha_locs(nua*noa,2)
                      integer, allocatable :: temp(:), idx(:)

                      Eai_table=0
                      kout = 1
                      do a = 1, nua
                         do i = 1, noa
                            Eai_table(a,i) = kout
                            kout = kout + 1
                         end do
                      end do
                      XjXk_table=0
                      kout = 1
                      do j = 1, nob
                         do k = j+1, nob
                            XjXk_table(j,k) = kout
                            XjXk_table(k,j) = -kout
                            kout = kout + 1
                         end do
                      end do

                      allocate(temp(n3abb),idx(n3abb))
                      do idet = 1, n3abb
                         a = t3c_excits(1,idet); i = t3c_excits(4,idet);
                         ia = Eai_table(a,i)
                         temp(idet) = ia
                      end do
                      call argsort(temp, idx)
                      t3c_excits = t3c_excits(:,idx)
                      t3c_amps = t3c_amps(idx)
                      if (present(resid)) resid = resid(idx)
                      deallocate(temp,idx)

                      alpha_locs = 1
                      do idet = 2, n3abb
                         a1 = t3c_excits(1,idet-1); i1 = t3c_excits(4,idet-1);
                         a2 = t3c_excits(1,idet);   i2 = t3c_excits(4,idet);
                         ia1 = Eai_table(a1,i1)
                         ia2 = Eai_table(a2,i2)
                         if (ia1/=ia2) then
                                 alpha_locs(ia1,2) = idet - 1
                                 alpha_locs(ia2,1) = idet
                         end if
                      end do
                      alpha_locs(ia2,2) = n3abb

                      ID = 0
                      do a = 1,nua
                         do i = 1,noa
                            ia = Eai_table(a,i)
                            if (alpha_locs(ia,1) > alpha_locs(ia,2)) cycle ! skip if alpha block is empty

                            num_jk_ia = alpha_locs(ia,2) - alpha_locs(ia,1) + 1

                            allocate(temp(num_jk_ia), idx(num_jk_ia))
                            kout = 1
                            do idet = alpha_locs(ia,1), alpha_locs(ia,2)
                               j = t3c_excits(5,idet); k = t3c_excits(6,idet);
                               jk = XjXk_table(j,k)
                               temp(kout) = jk
                               kout = kout + 1
                            end do
                            call argsort(temp,idx)
                            idx = idx + alpha_locs(ia,1) - 1
                            t3c_excits(:,alpha_locs(ia,1):alpha_locs(ia,2)) = t3c_excits(:,idx)
                            t3c_amps(alpha_locs(ia,1):alpha_locs(ia,2)) = t3c_amps(idx)
                            if (present(resid)) resid(alpha_locs(ia,1):alpha_locs(ia,2)) = resid(idx)
                            deallocate(temp,idx)

                            ID(ia,:,1) = alpha_locs(ia,1)
                            do jk = 1, num_jk_ia-1
                               idet = jk + alpha_locs(ia,1)
                               j1 = t3c_excits(5,idet-1); k1 = t3c_excits(6,idet-1);
                               j2 = t3c_excits(5,idet);   k2 = t3c_excits(6,idet);
                               jk1 = XjXk_table(j1,k1)
                               jk2 = XjXk_table(j2,k2)
                               
                               if (jk1/=jk2) then
                                       ID(ia,jk1,2) = idet-1
                                       ID(ia,jk2,1) = idet
                               end if
                            end do
                            ID(ia,jk2,2) = alpha_locs(ia,1) + num_jk_ia - 1
                         end do
                      end do

              end subroutine sort_t3c_h

              subroutine argsort(r,d)

                      integer, intent(in), dimension(:) :: r
                      integer, intent(out), dimension(size(r)) :: d
            
                      integer, dimension(size(r)) :: il

                      integer :: stepsize
                      integer :: i, j, n, left, k, ksize
            
                      n = size(r)
            
                      do i=1,n
                          d(i)=i
                      end do
            
                      if ( n==1 ) return
                    
                      stepsize = 1
                      do while (stepsize<n)
                          do left=1,n-stepsize,stepsize*2
                              i = left
                              j = left+stepsize
                              ksize = min(stepsize*2,n-left+1)
                              k=1
                        
                              do while ( i<left+stepsize .and. j<left+ksize )
                                  if ( r(d(i))<r(d(j)) ) then
                                      il(k)=d(i)
                                      i=i+1
                                      k=k+1
                                  else
                                      il(k)=d(j)
                                      j=j+1
                                      k=k+1
                                  endif
                              enddo
                        
                              if ( i<left+stepsize ) then
                                  ! fill up remaining from left
                                  il(k:ksize) = d(i:left+stepsize-1)
                              else
                                  ! fill up remaining from right
                                  il(k:ksize) = d(j:left+ksize-1)
                              endif
                              d(left:left+ksize-1) = il(1:ksize)
                          end do
                          stepsize=stepsize*2
                      end do

          end subroutine argsort


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


end module ccp_quadratic_loops_direct_opt
