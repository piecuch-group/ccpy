module ccp_quadratic_loops_direct

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
                  !$omp H1A_oo,H1A_vv,H2A_oooo,&
                  !$omp H2A_vvvv,H2A_voov,H2B_voov,I2A_vooo,I2A_vvov,&
                  !$omp fA_oo,fA_vv,shift,noa,nua,nob,nub,n3aaa,n3aab),&
                  !$omp private(hmatel,t_amp,denom,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet)


                  !$omp do
                  do idet = 1, n3aaa
                      do jdet = 1, n3aaa
                          hmatel = 0.0d0
                          t_amp = t3a_amps(jdet)

                          a = t3a_excits(1,idet); b = t3a_excits(2,idet); c = t3a_excits(3,idet);
                          i = t3a_excits(4,idet); j = t3a_excits(5,idet); k = t3a_excits(6,idet);
                          d = t3a_excits(1,jdet); e = t3a_excits(2,jdet); f = t3a_excits(3,jdet);
                          l = t3a_excits(4,jdet); m = t3a_excits(5,jdet); n = t3a_excits(6,jdet);

                          hmatel = hmatel + aaa_oo_aaa(i,j,k,a,b,c,l,m,n,d,e,f,h1a_oo,noa)
                          hmatel = hmatel + aaa_vv_aaa(i,j,k,a,b,c,l,m,n,d,e,f,h1a_vv,nua)
                          hmatel = hmatel + aaa_oooo_aaa(i,j,k,a,b,c,l,m,n,d,e,f,h2a_oooo,noa)
                          hmatel = hmatel + aaa_vvvv_aaa(i,j,k,a,b,c,l,m,n,d,e,f,h2a_vvvv,nua)
                          hmatel = hmatel + aaa_voov_aaa(i,j,k,a,b,c,l,m,n,d,e,f,h2a_voov,noa,nua)

                          resid(idet) = resid(idet) + hmatel * t_amp
                      end do
                      do jdet = 1, n3aab
                          hmatel = 0.0d0
                          t_amp = t3b_amps(jdet)

                          a = t3a_excits(1,idet); b = t3a_excits(2,idet); c = t3a_excits(3,idet);
                          i = t3a_excits(4,idet); j = t3a_excits(5,idet); k = t3a_excits(6,idet);
                          d = t3b_excits(1,jdet); e = t3b_excits(2,jdet); f = t3b_excits(3,jdet);
                          l = t3b_excits(4,jdet); m = t3b_excits(5,jdet); n = t3b_excits(6,jdet);

                          hmatel = hmatel + aaa_voov_aab(i,j,k,a,b,c,l,m,n,d,e,f,h2b_voov,noa,nua,nob,nub)

                          resid(idet) = resid(idet) + hmatel * t_amp
                      end do
                  end do
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

                      resid(idet) = resid(idet) + res_mm23
                      t3a_amps(idet) = t3a_amps(idet) + resid(idet)/(denom - shift)

                  end do
                  !$omp end do

                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!

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
                  integer, intent(in) :: t3a_excits(6, n3aaa), t3b_excits(6, n3aab), t3c_excits(6, n3abb)
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
                  !$omp H1A_oo,H1A_vv,H1B_oo,H1B_vv,H2A_oooo,H2B_oooo,&
                  !$omp H2B_ovvo,H2A_vvvv,H2B_vvvv,H2A_voov,H2C_voov,&
                  !$omp H2B_vovo,H2B_ovov,H2B_voov,&
                  !$omp I2A_vooo,I2A_vvov,I2B_vooo,I2B_ovoo,I2B_vvov,I2B_vvvo,&
                  !$omp fA_oo,fB_oo,fA_vv,fB_vv,noa,nua,nob,nub,shift,&
                  !$omp n3aaa,n3aab,n3abb),&
                  !$omp private(hmatel,t_amp,denom,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet)
                  
                  !$omp do
                  do idet = 1, n3aab
                      do jdet = 1, n3aab
                          hmatel = 0.0d0
                          t_amp = t3b_amps(jdet)

                          a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                          i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                          d = t3b_excits(1,jdet); e = t3b_excits(2,jdet); f = t3b_excits(3,jdet);
                          l = t3b_excits(4,jdet); m = t3b_excits(5,jdet); n = t3b_excits(6,jdet);

                          hmatel = hmatel + aab_oo_aab(i,j,k,a,b,c,l,m,n,d,e,f,h1a_oo,h1b_oo,noa,nob)

                          resid(idet) = resid(idet) + hmatel * t_amp
                      end do
                  end do
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

                      resid(idet) = resid(idet) + res_mm23

                      t3b_amps(idet) = t3b_amps(idet) + resid(idet)/(denom - shift)
                  end do
                  !$omp end do

                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
              end subroutine update_t3b_p

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

                  ! (1)
                  if (a==d .and. b==e .and. c==f) then
                     ! (1)
                     if (j==m .and. k==n) hmatel = hmatel - h(l,i) ! (1)
                     if (j==l .and. k==n) hmatel = hmatel + h(m,i) ! (lm)
                     if (j==m .and. k==l) hmatel = hmatel + h(n,i) ! (ln)
                     if (i==m .and. k==n) hmatel = hmatel + h(l,j) ! (ij)
                     if (i==l .and. k==n) hmatel = hmatel - h(m,j) ! (lm)(ij)
                     if (i==m .and. k==l) hmatel = hmatel - h(n,j) ! (ln)(ij)
                     if (j==m .and. i==n) hmatel = hmatel + h(l,k) ! (ik)
                     if (j==l .and. i==n) hmatel = hmatel - h(m,k) ! (lm)(ik)
                     if (j==m .and. i==l) hmatel = hmatel - h(n,k) ! (ln)(ik)
                     ! (jk)
                     if (k==m .and. j==n) hmatel = hmatel + h(l,i) ! (1)
                     if (k==l .and. j==n) hmatel = hmatel - h(m,i) ! (lm)
                     if (k==m .and. j==l) hmatel = hmatel - h(n,i) ! (ln)
                     if (i==m .and. j==n) hmatel = hmatel - h(l,k) ! (ij)
                     if (i==l .and. j==n) hmatel = hmatel + h(m,k) ! (lm)(ij)
                     if (i==m .and. j==l) hmatel = hmatel + h(n,k) ! (ln)(ij)
                     if (k==m .and. i==n) hmatel = hmatel - h(l,j) ! (ik)
                     if (k==l .and. i==n) hmatel = hmatel + h(m,j) ! (lm)(ik)
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

                  ! (1)
                  if (i==l .and. j==m .and. k==n) then
                      ! (1)
                      if (b==e .and. c==f) hmatel = hmatel + h(a,d) ! (1)
                      if (b==d .and. c==f) hmatel = hmatel - h(a,e) ! (de)
                      if (b==e .and. c==d) hmatel = hmatel - h(a,f) ! (df)
                      if (a==e .and. c==f) hmatel = hmatel - h(b,d) ! (ab)
                      if (a==d .and. c==f) hmatel = hmatel + h(b,e) ! (de)(ab)
                      if (a==e .and. c==d) hmatel = hmatel + h(b,f) ! (df)(ab)
                      if (b==e .and. a==f) hmatel = hmatel - h(c,d) ! (ac)
                      if (b==d .and. a==f) hmatel = hmatel + h(c,e) ! (de)(ac)
                      if (b==e .and. a==d) hmatel = hmatel + h(c,f) ! (df)(ac)
                      ! (bc)
                      if (c==e .and. b==f) hmatel = hmatel - h(a,d) ! (1)
                      if (c==d .and. b==f) hmatel = hmatel + h(a,e) ! (de)
                      if (c==e .and. b==d) hmatel = hmatel + h(a,f) ! (df)
                      if (a==e .and. b==f) hmatel = hmatel + h(c,d) ! (ab)
                      if (a==d .and. b==f) hmatel = hmatel - h(c,e) ! (de)(ab)
                      if (a==e .and. b==d) hmatel = hmatel - h(c,f) ! (df)(ab)
                      if (c==e .and. a==f) hmatel = hmatel + h(b,d) ! (ac)
                      if (c==d .and. a==f) hmatel = hmatel - h(b,e) ! (de)(ac)
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
                      if (k==n) hmatel = hmatel + h(l,m,i,j) ! (1)
                      if (k==l) hmatel = hmatel - h(n,m,i,j) ! (ln)
                      if (k==m) hmatel = hmatel - h(l,n,i,j) ! (mn)
                      ! (ik)
                      if (i==n) hmatel = hmatel - h(l,m,k,j) ! (1)
                      if (i==l) hmatel = hmatel + h(n,m,k,j) ! (ln)
                      if (i==m) hmatel = hmatel + h(l,n,k,j) ! (mn)
                      ! (jk)
                      if (j==n) hmatel = hmatel - h(l,m,i,k) ! (1)
                      if (j==l) hmatel = hmatel + h(n,m,i,k) ! (ln)
                      if (j==m) hmatel = hmatel + h(l,n,i,k) ! (mn)
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
                      ! (1)
                      if (c==f) hmatel = hmatel + h(a,b,d,e) ! (1)
                      if (a==f) hmatel = hmatel - h(c,b,d,e) ! (ac)
                      if (b==f) hmatel = hmatel - h(a,c,d,e) ! (bc)
                      ! (fd)
                      if (c==d) hmatel = hmatel - h(a,b,f,e) ! (1)
                      if (a==d) hmatel = hmatel + h(c,b,f,e) ! (ac)
                      if (b==d) hmatel = hmatel + h(a,c,f,e) ! (bc)
                      ! (fe)
                      if (c==e) hmatel = hmatel - h(a,b,d,f) ! (1)
                      if (a==e) hmatel = hmatel + h(c,b,d,f) ! (ac)
                      if (b==e) hmatel = hmatel + h(a,c,d,f) ! (bc)
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
                  if (j==m .and. k==n .and. b==e .and. c==d) hmatel = hmatel - h(a,l,i,f) ! (df)
                  if (j==l .and. k==n .and. b==e .and. c==f) hmatel = hmatel - h(a,m,i,d) ! (lm)
                  if (j==l .and. k==n .and. b==d .and. c==f) hmatel = hmatel + h(a,m,i,e) ! (de)(lm)
                  if (j==l .and. k==n .and. b==e .and. c==d) hmatel = hmatel + h(a,m,i,f) ! (df)(lm)
                  if (j==m .and. k==l .and. b==e .and. c==f) hmatel = hmatel - h(a,n,i,d) ! (ln)
                  if (j==m .and. k==l .and. b==d .and. c==f) hmatel = hmatel + h(a,n,i,e) ! (de)(ln)
                  if (j==m .and. k==l .and. b==e .and. c==d) hmatel = hmatel + h(a,n,i,f) ! (df)(ln)
                  if (j==m .and. k==n .and. a==e .and. c==f) hmatel = hmatel - h(b,l,i,d) ! (ab)
                  if (j==m .and. k==n .and. a==d .and. c==f) hmatel = hmatel + h(b,l,i,e) ! (de)(ab)
                  if (j==m .and. k==n .and. a==e .and. c==d) hmatel = hmatel + h(b,l,i,f) ! (df)(ab)
                  if (j==l .and. k==n .and. a==e .and. c==f) hmatel = hmatel + h(b,m,i,d) ! (lm)(ab)
                  if (j==l .and. k==n .and. a==d .and. c==f) hmatel = hmatel - h(b,m,i,e) ! (de)(lm)(ab)
                  if (j==l .and. k==n .and. a==e .and. c==d) hmatel = hmatel - h(b,m,i,f) ! (df)(lm)(ab)
                  if (j==m .and. k==l .and. a==e .and. c==f) hmatel = hmatel + h(b,n,i,d) ! (ln)(ab)
                  if (j==m .and. k==l .and. a==d .and. c==f) hmatel = hmatel - h(b,n,i,e) ! (de)(ln)(ab)
                  if (j==m .and. k==l .and. a==e .and. c==d) hmatel = hmatel - h(b,n,i,f) ! (df)(ln)(ab)
                  if (j==m .and. k==n .and. b==e .and. a==f) hmatel = hmatel - h(c,l,i,d) ! (ac)
                  if (j==m .and. k==n .and. b==d .and. a==f) hmatel = hmatel + h(c,l,i,e) ! (de)(ac)
                  if (j==m .and. k==n .and. b==e .and. a==d) hmatel = hmatel + h(c,l,i,f) ! (df)(ac)
                  if (j==l .and. k==n .and. b==e .and. a==f) hmatel = hmatel + h(c,m,i,d) ! (lm)(ac)
                  if (j==l .and. k==n .and. b==d .and. a==f) hmatel = hmatel - h(c,m,i,e) ! (de)(lm)(ac)
                  if (j==l .and. k==n .and. b==e .and. a==d) hmatel = hmatel - h(c,m,i,f) ! (df)(lm)(ac)
                  if (j==m .and. k==l .and. b==e .and. a==f) hmatel = hmatel + h(c,n,i,d) ! (ln)(ac)
                  if (j==m .and. k==l .and. b==d .and. a==f) hmatel = hmatel - h(c,n,i,e) ! (de)(ln)(ac)
                  if (j==m .and. k==l .and. b==e .and. a==d) hmatel = hmatel - h(c,n,i,f) ! (df)(ln)(ac)
                  if (i==m .and. k==n .and. b==e .and. c==f) hmatel = hmatel - h(a,l,j,d) ! (ij)
                  if (i==m .and. k==n .and. b==d .and. c==f) hmatel = hmatel + h(a,l,j,e) ! (de)(ij)
                  if (i==m .and. k==n .and. b==e .and. c==d) hmatel = hmatel + h(a,l,j,f) ! (df)(ij)
                  if (i==l .and. k==n .and. b==e .and. c==f) hmatel = hmatel + h(a,m,j,d) ! (lm)(ij)
                  if (i==l .and. k==n .and. b==d .and. c==f) hmatel = hmatel - h(a,m,j,e) ! (de)(lm)(ij)
                  if (i==l .and. k==n .and. b==e .and. c==d) hmatel = hmatel - h(a,m,j,f) ! (df)(lm)(ij)
                  if (i==m .and. k==l .and. b==e .and. c==f) hmatel = hmatel + h(a,n,j,d) ! (ln)(ij)
                  if (i==m .and. k==l .and. b==d .and. c==f) hmatel = hmatel - h(a,n,j,e) ! (de)(ln)(ij)
                  if (i==m .and. k==l .and. b==e .and. c==d) hmatel = hmatel - h(a,n,j,f) ! (df)(ln)(ij)
                  if (i==m .and. k==n .and. a==e .and. c==f) hmatel = hmatel + h(b,l,j,d) ! (ab)(ij)
                  if (i==m .and. k==n .and. a==d .and. c==f) hmatel = hmatel - h(b,l,j,e) ! (de)(ab)(ij)
                  if (i==m .and. k==n .and. a==e .and. c==d) hmatel = hmatel - h(b,l,j,f) ! (df)(ab)(ij)
                  if (i==l .and. k==n .and. a==e .and. c==f) hmatel = hmatel - h(b,m,j,d) ! (lm)(ab)(ij)
                  if (i==l .and. k==n .and. a==d .and. c==f) hmatel = hmatel + h(b,m,j,e) ! (de)(lm)(ab)(ij)
                  if (i==l .and. k==n .and. a==e .and. c==d) hmatel = hmatel + h(b,m,j,f) ! (df)(lm)(ab)(ij)
                  if (i==m .and. k==l .and. a==e .and. c==f) hmatel = hmatel - h(b,n,j,d) ! (ln)(ab)(ij)
                  if (i==m .and. k==l .and. a==d .and. c==f) hmatel = hmatel + h(b,n,j,e) ! (de)(ln)(ab)(ij)
                  if (i==m .and. k==l .and. a==e .and. c==d) hmatel = hmatel + h(b,n,j,f) ! (df)(ln)(ab)(ij)
                  if (i==m .and. k==n .and. b==e .and. a==f) hmatel = hmatel + h(c,l,j,d) ! (ac)(ij)
                  if (i==m .and. k==n .and. b==d .and. a==f) hmatel = hmatel - h(c,l,j,e) ! (de)(ac)(ij)
                  if (i==m .and. k==n .and. b==e .and. a==d) hmatel = hmatel - h(c,l,j,f) ! (df)(ac)(ij)
                  if (i==l .and. k==n .and. b==e .and. a==f) hmatel = hmatel - h(c,m,j,d) ! (lm)(ac)(ij)
                  if (i==l .and. k==n .and. b==d .and. a==f) hmatel = hmatel + h(c,m,j,e) ! (de)(lm)(ac)(ij)
                  if (i==l .and. k==n .and. b==e .and. a==d) hmatel = hmatel + h(c,m,j,f) ! (df)(lm)(ac)(ij)
                  if (i==m .and. k==l .and. b==e .and. a==f) hmatel = hmatel - h(c,n,j,d) ! (ln)(ac)(ij)
                  if (i==m .and. k==l .and. b==d .and. a==f) hmatel = hmatel + h(c,n,j,e) ! (de)(ln)(ac)(ij)
                  if (i==m .and. k==l .and. b==e .and. a==d) hmatel = hmatel + h(c,n,j,f) ! (df)(ln)(ac)(ij)
                  if (j==m .and. i==n .and. b==e .and. c==f) hmatel = hmatel - h(a,l,k,d) ! (ik)
                  if (j==m .and. i==n .and. b==d .and. c==f) hmatel = hmatel + h(a,l,k,e) ! (de)(ik)
                  if (j==m .and. i==n .and. b==e .and. c==d) hmatel = hmatel + h(a,l,k,f) ! (df)(ik)
                  if (j==l .and. i==n .and. b==e .and. c==f) hmatel = hmatel + h(a,m,k,d) ! (lm)(ik)
                  if (j==l .and. i==n .and. b==d .and. c==f) hmatel = hmatel - h(a,m,k,e) ! (de)(lm)(ik)
                  if (j==l .and. i==n .and. b==e .and. c==d) hmatel = hmatel - h(a,m,k,f) ! (df)(lm)(ik)
                  if (j==m .and. i==l .and. b==e .and. c==f) hmatel = hmatel + h(a,n,k,d) ! (ln)(ik)
                  if (j==m .and. i==l .and. b==d .and. c==f) hmatel = hmatel - h(a,n,k,e) ! (de)(ln)(ik)
                  if (j==m .and. i==l .and. b==e .and. c==d) hmatel = hmatel - h(a,n,k,f) ! (df)(ln)(ik)
                  if (j==m .and. i==n .and. a==e .and. c==f) hmatel = hmatel + h(b,l,k,d) ! (ab)(ik)
                  if (j==m .and. i==n .and. a==d .and. c==f) hmatel = hmatel - h(b,l,k,e) ! (de)(ab)(ik)
                  if (j==m .and. i==n .and. a==e .and. c==d) hmatel = hmatel - h(b,l,k,f) ! (df)(ab)(ik)
                  if (j==l .and. i==n .and. a==e .and. c==f) hmatel = hmatel - h(b,m,k,d) ! (lm)(ab)(ik)
                  if (j==l .and. i==n .and. a==d .and. c==f) hmatel = hmatel + h(b,m,k,e) ! (de)(lm)(ab)(ik)
                  if (j==l .and. i==n .and. a==e .and. c==d) hmatel = hmatel + h(b,m,k,f) ! (df)(lm)(ab)(ik)
                  if (j==m .and. i==l .and. a==e .and. c==f) hmatel = hmatel - h(b,n,k,d) ! (ln)(ab)(ik)
                  if (j==m .and. i==l .and. a==d .and. c==f) hmatel = hmatel + h(b,n,k,e) ! (de)(ln)(ab)(ik)
                  if (j==m .and. i==l .and. a==e .and. c==d) hmatel = hmatel + h(b,n,k,f) ! (df)(ln)(ab)(ik)
                  if (j==m .and. i==n .and. b==e .and. a==f) hmatel = hmatel + h(c,l,k,d) ! (ac)(ik)
                  if (j==m .and. i==n .and. b==d .and. a==f) hmatel = hmatel - h(c,l,k,e) ! (de)(ac)(ik)
                  if (j==m .and. i==n .and. b==e .and. a==d) hmatel = hmatel - h(c,l,k,f) ! (df)(ac)(ik)
                  if (j==l .and. i==n .and. b==e .and. a==f) hmatel = hmatel - h(c,m,k,d) ! (lm)(ac)(ik)
                  if (j==l .and. i==n .and. b==d .and. a==f) hmatel = hmatel + h(c,m,k,e) ! (de)(lm)(ac)(ik)
                  if (j==l .and. i==n .and. b==e .and. a==d) hmatel = hmatel + h(c,m,k,f) ! (df)(lm)(ac)(ik)
                  if (j==m .and. i==l .and. b==e .and. a==f) hmatel = hmatel - h(c,n,k,d) ! (ln)(ac)(ik)
                  if (j==m .and. i==l .and. b==d .and. a==f) hmatel = hmatel + h(c,n,k,e) ! (de)(ln)(ac)(ik)
                  if (j==m .and. i==l .and. b==e .and. a==d) hmatel = hmatel + h(c,n,k,f) ! (df)(ln)(ac)(ik)
                  ! (bc)
                  if (j==m .and. k==n .and. c==e .and. b==f) hmatel = hmatel - h(a,l,i,d) ! (1)
                  if (j==m .and. k==n .and. c==d .and. b==f) hmatel = hmatel + h(a,l,i,e) ! (de)
                  if (j==m .and. k==n .and. c==e .and. b==d) hmatel = hmatel + h(a,l,i,f) ! (df)
                  if (j==l .and. k==n .and. c==e .and. b==f) hmatel = hmatel + h(a,m,i,d) ! (lm)
                  if (j==l .and. k==n .and. c==d .and. b==f) hmatel = hmatel - h(a,m,i,e) ! (de)(lm)
                  if (j==l .and. k==n .and. c==e .and. b==d) hmatel = hmatel - h(a,m,i,f) ! (df)(lm)
                  if (j==m .and. k==l .and. c==e .and. b==f) hmatel = hmatel + h(a,n,i,d) ! (ln)
                  if (j==m .and. k==l .and. c==d .and. b==f) hmatel = hmatel - h(a,n,i,e) ! (de)(ln)
                  if (j==m .and. k==l .and. c==e .and. b==d) hmatel = hmatel - h(a,n,i,f) ! (df)(ln)
                  if (j==m .and. k==n .and. a==e .and. b==f) hmatel = hmatel + h(c,l,i,d) ! (ab)
                  if (j==m .and. k==n .and. a==d .and. b==f) hmatel = hmatel - h(c,l,i,e) ! (de)(ab)
                  if (j==m .and. k==n .and. a==e .and. b==d) hmatel = hmatel - h(c,l,i,f) ! (df)(ab)
                  if (j==l .and. k==n .and. a==e .and. b==f) hmatel = hmatel - h(c,m,i,d) ! (lm)(ab)
                  if (j==l .and. k==n .and. a==d .and. b==f) hmatel = hmatel + h(c,m,i,e) ! (de)(lm)(ab)
                  if (j==l .and. k==n .and. a==e .and. b==d) hmatel = hmatel + h(c,m,i,f) ! (df)(lm)(ab)
                  if (j==m .and. k==l .and. a==e .and. b==f) hmatel = hmatel - h(c,n,i,d) ! (ln)(ab)
                  if (j==m .and. k==l .and. a==d .and. b==f) hmatel = hmatel + h(c,n,i,e) ! (de)(ln)(ab)
                  if (j==m .and. k==l .and. a==e .and. b==d) hmatel = hmatel + h(c,n,i,f) ! (df)(ln)(ab)
                  if (j==m .and. k==n .and. c==e .and. a==f) hmatel = hmatel + h(b,l,i,d) ! (ac)
                  if (j==m .and. k==n .and. c==d .and. a==f) hmatel = hmatel - h(b,l,i,e) ! (de)(ac)
                  if (j==m .and. k==n .and. c==e .and. a==d) hmatel = hmatel - h(b,l,i,f) ! (df)(ac)
                  if (j==l .and. k==n .and. c==e .and. a==f) hmatel = hmatel - h(b,m,i,d) ! (lm)(ac)
                  if (j==l .and. k==n .and. c==d .and. a==f) hmatel = hmatel + h(b,m,i,e) ! (de)(lm)(ac)
                  if (j==l .and. k==n .and. c==e .and. a==d) hmatel = hmatel + h(b,m,i,f) ! (df)(lm)(ac)
                  if (j==m .and. k==l .and. c==e .and. a==f) hmatel = hmatel - h(b,n,i,d) ! (ln)(ac)
                  if (j==m .and. k==l .and. c==d .and. a==f) hmatel = hmatel + h(b,n,i,e) ! (de)(ln)(ac)
                  if (j==m .and. k==l .and. c==e .and. a==d) hmatel = hmatel + h(b,n,i,f) ! (df)(ln)(ac)
                  if (i==m .and. k==n .and. c==e .and. b==f) hmatel = hmatel + h(a,l,j,d) ! (ij)
                  if (i==m .and. k==n .and. c==d .and. b==f) hmatel = hmatel - h(a,l,j,e) ! (de)(ij)
                  if (i==m .and. k==n .and. c==e .and. b==d) hmatel = hmatel - h(a,l,j,f) ! (df)(ij)
                  if (i==l .and. k==n .and. c==e .and. b==f) hmatel = hmatel - h(a,m,j,d) ! (lm)(ij)
                  if (i==l .and. k==n .and. c==d .and. b==f) hmatel = hmatel + h(a,m,j,e) ! (de)(lm)(ij)
                  if (i==l .and. k==n .and. c==e .and. b==d) hmatel = hmatel + h(a,m,j,f) ! (df)(lm)(ij)
                  if (i==m .and. k==l .and. c==e .and. b==f) hmatel = hmatel - h(a,n,j,d) ! (ln)(ij)
                  if (i==m .and. k==l .and. c==d .and. b==f) hmatel = hmatel + h(a,n,j,e) ! (de)(ln)(ij)
                  if (i==m .and. k==l .and. c==e .and. b==d) hmatel = hmatel + h(a,n,j,f) ! (df)(ln)(ij)
                  if (i==m .and. k==n .and. a==e .and. b==f) hmatel = hmatel - h(c,l,j,d) ! (ab)(ij)
                  if (i==m .and. k==n .and. a==d .and. b==f) hmatel = hmatel + h(c,l,j,e) ! (de)(ab)(ij)
                  if (i==m .and. k==n .and. a==e .and. b==d) hmatel = hmatel + h(c,l,j,f) ! (df)(ab)(ij)
                  if (i==l .and. k==n .and. a==e .and. b==f) hmatel = hmatel + h(c,m,j,d) ! (lm)(ab)(ij)
                  if (i==l .and. k==n .and. a==d .and. b==f) hmatel = hmatel - h(c,m,j,e) ! (de)(lm)(ab)(ij)
                  if (i==l .and. k==n .and. a==e .and. b==d) hmatel = hmatel - h(c,m,j,f) ! (df)(lm)(ab)(ij)
                  if (i==m .and. k==l .and. a==e .and. b==f) hmatel = hmatel + h(c,n,j,d) ! (ln)(ab)(ij)
                  if (i==m .and. k==l .and. a==d .and. b==f) hmatel = hmatel - h(c,n,j,e) ! (de)(ln)(ab)(ij)
                  if (i==m .and. k==l .and. a==e .and. b==d) hmatel = hmatel - h(c,n,j,f) ! (df)(ln)(ab)(ij)
                  if (i==m .and. k==n .and. c==e .and. a==f) hmatel = hmatel - h(b,l,j,d) ! (ac)(ij)
                  if (i==m .and. k==n .and. c==d .and. a==f) hmatel = hmatel + h(b,l,j,e) ! (de)(ac)(ij)
                  if (i==m .and. k==n .and. c==e .and. a==d) hmatel = hmatel + h(b,l,j,f) ! (df)(ac)(ij)
                  if (i==l .and. k==n .and. c==e .and. a==f) hmatel = hmatel + h(b,m,j,d) ! (lm)(ac)(ij)
                  if (i==l .and. k==n .and. c==d .and. a==f) hmatel = hmatel - h(b,m,j,e) ! (de)(lm)(ac)(ij)
                  if (i==l .and. k==n .and. c==e .and. a==d) hmatel = hmatel - h(b,m,j,f) ! (df)(lm)(ac)(ij)
                  if (i==m .and. k==l .and. c==e .and. a==f) hmatel = hmatel + h(b,n,j,d) ! (ln)(ac)(ij)
                  if (i==m .and. k==l .and. c==d .and. a==f) hmatel = hmatel - h(b,n,j,e) ! (de)(ln)(ac)(ij)
                  if (i==m .and. k==l .and. c==e .and. a==d) hmatel = hmatel - h(b,n,j,f) ! (df)(ln)(ac)(ij)
                  if (j==m .and. i==n .and. c==e .and. b==f) hmatel = hmatel + h(a,l,k,d) ! (ik)
                  if (j==m .and. i==n .and. c==d .and. b==f) hmatel = hmatel - h(a,l,k,e) ! (de)(ik)
                  if (j==m .and. i==n .and. c==e .and. b==d) hmatel = hmatel - h(a,l,k,f) ! (df)(ik)
                  if (j==l .and. i==n .and. c==e .and. b==f) hmatel = hmatel - h(a,m,k,d) ! (lm)(ik)
                  if (j==l .and. i==n .and. c==d .and. b==f) hmatel = hmatel + h(a,m,k,e) ! (de)(lm)(ik)
                  if (j==l .and. i==n .and. c==e .and. b==d) hmatel = hmatel + h(a,m,k,f) ! (df)(lm)(ik)
                  if (j==m .and. i==l .and. c==e .and. b==f) hmatel = hmatel - h(a,n,k,d) ! (ln)(ik)
                  if (j==m .and. i==l .and. c==d .and. b==f) hmatel = hmatel + h(a,n,k,e) ! (de)(ln)(ik)
                  if (j==m .and. i==l .and. c==e .and. b==d) hmatel = hmatel + h(a,n,k,f) ! (df)(ln)(ik)
                  if (j==m .and. i==n .and. a==e .and. b==f) hmatel = hmatel - h(c,l,k,d) ! (ab)(ik)
                  if (j==m .and. i==n .and. a==d .and. b==f) hmatel = hmatel + h(c,l,k,e) ! (de)(ab)(ik)
                  if (j==m .and. i==n .and. a==e .and. b==d) hmatel = hmatel + h(c,l,k,f) ! (df)(ab)(ik)
                  if (j==l .and. i==n .and. a==e .and. b==f) hmatel = hmatel + h(c,m,k,d) ! (lm)(ab)(ik)
                  if (j==l .and. i==n .and. a==d .and. b==f) hmatel = hmatel - h(c,m,k,e) ! (de)(lm)(ab)(ik)
                  if (j==l .and. i==n .and. a==e .and. b==d) hmatel = hmatel - h(c,m,k,f) ! (df)(lm)(ab)(ik)
                  if (j==m .and. i==l .and. a==e .and. b==f) hmatel = hmatel + h(c,n,k,d) ! (ln)(ab)(ik)
                  if (j==m .and. i==l .and. a==d .and. b==f) hmatel = hmatel - h(c,n,k,e) ! (de)(ln)(ab)(ik)
                  if (j==m .and. i==l .and. a==e .and. b==d) hmatel = hmatel - h(c,n,k,f) ! (df)(ln)(ab)(ik)
                  if (j==m .and. i==n .and. c==e .and. a==f) hmatel = hmatel - h(b,l,k,d) ! (ac)(ik)
                  if (j==m .and. i==n .and. c==d .and. a==f) hmatel = hmatel + h(b,l,k,e) ! (de)(ac)(ik)
                  if (j==m .and. i==n .and. c==e .and. a==d) hmatel = hmatel + h(b,l,k,f) ! (df)(ac)(ik)
                  if (j==l .and. i==n .and. c==e .and. a==f) hmatel = hmatel + h(b,m,k,d) ! (lm)(ac)(ik)
                  if (j==l .and. i==n .and. c==d .and. a==f) hmatel = hmatel - h(b,m,k,e) ! (de)(lm)(ac)(ik)
                  if (j==l .and. i==n .and. c==e .and. a==d) hmatel = hmatel - h(b,m,k,f) ! (df)(lm)(ac)(ik)
                  if (j==m .and. i==l .and. c==e .and. a==f) hmatel = hmatel + h(b,n,k,d) ! (ln)(ac)(ik)
                  if (j==m .and. i==l .and. c==d .and. a==f) hmatel = hmatel - h(b,n,k,e) ! (de)(ln)(ac)(ik)
                  if (j==m .and. i==l .and. c==e .and. a==d) hmatel = hmatel - h(b,n,k,f) ! (df)(ln)(ac)(ik)
                  ! (jk)
                  if (k==m .and. j==n .and. b==e .and. c==f) hmatel = hmatel - h(a,l,i,d) ! (1)
                  if (k==m .and. j==n .and. b==d .and. c==f) hmatel = hmatel + h(a,l,i,e) ! (de)
                  if (k==m .and. j==n .and. b==e .and. c==d) hmatel = hmatel + h(a,l,i,f) ! (df)
                  if (k==l .and. j==n .and. b==e .and. c==f) hmatel = hmatel + h(a,m,i,d) ! (lm)
                  if (k==l .and. j==n .and. b==d .and. c==f) hmatel = hmatel - h(a,m,i,e) ! (de)(lm)
                  if (k==l .and. j==n .and. b==e .and. c==d) hmatel = hmatel - h(a,m,i,f) ! (df)(lm)
                  if (k==m .and. j==l .and. b==e .and. c==f) hmatel = hmatel + h(a,n,i,d) ! (ln)
                  if (k==m .and. j==l .and. b==d .and. c==f) hmatel = hmatel - h(a,n,i,e) ! (de)(ln)
                  if (k==m .and. j==l .and. b==e .and. c==d) hmatel = hmatel - h(a,n,i,f) ! (df)(ln)
                  if (k==m .and. j==n .and. a==e .and. c==f) hmatel = hmatel + h(b,l,i,d) ! (ab)
                  if (k==m .and. j==n .and. a==d .and. c==f) hmatel = hmatel - h(b,l,i,e) ! (de)(ab)
                  if (k==m .and. j==n .and. a==e .and. c==d) hmatel = hmatel - h(b,l,i,f) ! (df)(ab)
                  if (k==l .and. j==n .and. a==e .and. c==f) hmatel = hmatel - h(b,m,i,d) ! (lm)(ab)
                  if (k==l .and. j==n .and. a==d .and. c==f) hmatel = hmatel + h(b,m,i,e) ! (de)(lm)(ab)
                  if (k==l .and. j==n .and. a==e .and. c==d) hmatel = hmatel + h(b,m,i,f) ! (df)(lm)(ab)
                  if (k==m .and. j==l .and. a==e .and. c==f) hmatel = hmatel - h(b,n,i,d) ! (ln)(ab)
                  if (k==m .and. j==l .and. a==d .and. c==f) hmatel = hmatel + h(b,n,i,e) ! (de)(ln)(ab)
                  if (k==m .and. j==l .and. a==e .and. c==d) hmatel = hmatel + h(b,n,i,f) ! (df)(ln)(ab)
                  if (k==m .and. j==n .and. b==e .and. a==f) hmatel = hmatel + h(c,l,i,d) ! (ac)
                  if (k==m .and. j==n .and. b==d .and. a==f) hmatel = hmatel - h(c,l,i,e) ! (de)(ac)
                  if (k==m .and. j==n .and. b==e .and. a==d) hmatel = hmatel - h(c,l,i,f) ! (df)(ac)
                  if (k==l .and. j==n .and. b==e .and. a==f) hmatel = hmatel - h(c,m,i,d) ! (lm)(ac)
                  if (k==l .and. j==n .and. b==d .and. a==f) hmatel = hmatel + h(c,m,i,e) ! (de)(lm)(ac)
                  if (k==l .and. j==n .and. b==e .and. a==d) hmatel = hmatel + h(c,m,i,f) ! (df)(lm)(ac)
                  if (k==m .and. j==l .and. b==e .and. a==f) hmatel = hmatel - h(c,n,i,d) ! (ln)(ac)
                  if (k==m .and. j==l .and. b==d .and. a==f) hmatel = hmatel + h(c,n,i,e) ! (de)(ln)(ac)
                  if (k==m .and. j==l .and. b==e .and. a==d) hmatel = hmatel + h(c,n,i,f) ! (df)(ln)(ac)
                  if (i==m .and. j==n .and. b==e .and. c==f) hmatel = hmatel + h(a,l,k,d) ! (ij)
                  if (i==m .and. j==n .and. b==d .and. c==f) hmatel = hmatel - h(a,l,k,e) ! (de)(ij)
                  if (i==m .and. j==n .and. b==e .and. c==d) hmatel = hmatel - h(a,l,k,f) ! (df)(ij)
                  if (i==l .and. j==n .and. b==e .and. c==f) hmatel = hmatel - h(a,m,k,d) ! (lm)(ij)
                  if (i==l .and. j==n .and. b==d .and. c==f) hmatel = hmatel + h(a,m,k,e) ! (de)(lm)(ij)
                  if (i==l .and. j==n .and. b==e .and. c==d) hmatel = hmatel + h(a,m,k,f) ! (df)(lm)(ij)
                  if (i==m .and. j==l .and. b==e .and. c==f) hmatel = hmatel - h(a,n,k,d) ! (ln)(ij)
                  if (i==m .and. j==l .and. b==d .and. c==f) hmatel = hmatel + h(a,n,k,e) ! (de)(ln)(ij)
                  if (i==m .and. j==l .and. b==e .and. c==d) hmatel = hmatel + h(a,n,k,f) ! (df)(ln)(ij)
                  if (i==m .and. j==n .and. a==e .and. c==f) hmatel = hmatel - h(b,l,k,d) ! (ab)(ij)
                  if (i==m .and. j==n .and. a==d .and. c==f) hmatel = hmatel + h(b,l,k,e) ! (de)(ab)(ij)
                  if (i==m .and. j==n .and. a==e .and. c==d) hmatel = hmatel + h(b,l,k,f) ! (df)(ab)(ij)
                  if (i==l .and. j==n .and. a==e .and. c==f) hmatel = hmatel + h(b,m,k,d) ! (lm)(ab)(ij)
                  if (i==l .and. j==n .and. a==d .and. c==f) hmatel = hmatel - h(b,m,k,e) ! (de)(lm)(ab)(ij)
                  if (i==l .and. j==n .and. a==e .and. c==d) hmatel = hmatel - h(b,m,k,f) ! (df)(lm)(ab)(ij)
                  if (i==m .and. j==l .and. a==e .and. c==f) hmatel = hmatel + h(b,n,k,d) ! (ln)(ab)(ij)
                  if (i==m .and. j==l .and. a==d .and. c==f) hmatel = hmatel - h(b,n,k,e) ! (de)(ln)(ab)(ij)
                  if (i==m .and. j==l .and. a==e .and. c==d) hmatel = hmatel - h(b,n,k,f) ! (df)(ln)(ab)(ij)
                  if (i==m .and. j==n .and. b==e .and. a==f) hmatel = hmatel - h(c,l,k,d) ! (ac)(ij)
                  if (i==m .and. j==n .and. b==d .and. a==f) hmatel = hmatel + h(c,l,k,e) ! (de)(ac)(ij)
                  if (i==m .and. j==n .and. b==e .and. a==d) hmatel = hmatel + h(c,l,k,f) ! (df)(ac)(ij)
                  if (i==l .and. j==n .and. b==e .and. a==f) hmatel = hmatel + h(c,m,k,d) ! (lm)(ac)(ij)
                  if (i==l .and. j==n .and. b==d .and. a==f) hmatel = hmatel - h(c,m,k,e) ! (de)(lm)(ac)(ij)
                  if (i==l .and. j==n .and. b==e .and. a==d) hmatel = hmatel - h(c,m,k,f) ! (df)(lm)(ac)(ij)
                  if (i==m .and. j==l .and. b==e .and. a==f) hmatel = hmatel + h(c,n,k,d) ! (ln)(ac)(ij)
                  if (i==m .and. j==l .and. b==d .and. a==f) hmatel = hmatel - h(c,n,k,e) ! (de)(ln)(ac)(ij)
                  if (i==m .and. j==l .and. b==e .and. a==d) hmatel = hmatel - h(c,n,k,f) ! (df)(ln)(ac)(ij)
                  if (k==m .and. i==n .and. b==e .and. c==f) hmatel = hmatel + h(a,l,j,d) ! (ik)
                  if (k==m .and. i==n .and. b==d .and. c==f) hmatel = hmatel - h(a,l,j,e) ! (de)(ik)
                  if (k==m .and. i==n .and. b==e .and. c==d) hmatel = hmatel - h(a,l,j,f) ! (df)(ik)
                  if (k==l .and. i==n .and. b==e .and. c==f) hmatel = hmatel - h(a,m,j,d) ! (lm)(ik)
                  if (k==l .and. i==n .and. b==d .and. c==f) hmatel = hmatel + h(a,m,j,e) ! (de)(lm)(ik)
                  if (k==l .and. i==n .and. b==e .and. c==d) hmatel = hmatel + h(a,m,j,f) ! (df)(lm)(ik)
                  if (k==m .and. i==l .and. b==e .and. c==f) hmatel = hmatel - h(a,n,j,d) ! (ln)(ik)
                  if (k==m .and. i==l .and. b==d .and. c==f) hmatel = hmatel + h(a,n,j,e) ! (de)(ln)(ik)
                  if (k==m .and. i==l .and. b==e .and. c==d) hmatel = hmatel + h(a,n,j,f) ! (df)(ln)(ik)
                  if (k==m .and. i==n .and. a==e .and. c==f) hmatel = hmatel - h(b,l,j,d) ! (ab)(ik)
                  if (k==m .and. i==n .and. a==d .and. c==f) hmatel = hmatel + h(b,l,j,e) ! (de)(ab)(ik)
                  if (k==m .and. i==n .and. a==e .and. c==d) hmatel = hmatel + h(b,l,j,f) ! (df)(ab)(ik)
                  if (k==l .and. i==n .and. a==e .and. c==f) hmatel = hmatel + h(b,m,j,d) ! (lm)(ab)(ik)
                  if (k==l .and. i==n .and. a==d .and. c==f) hmatel = hmatel - h(b,m,j,e) ! (de)(lm)(ab)(ik)
                  if (k==l .and. i==n .and. a==e .and. c==d) hmatel = hmatel - h(b,m,j,f) ! (df)(lm)(ab)(ik)
                  if (k==m .and. i==l .and. a==e .and. c==f) hmatel = hmatel + h(b,n,j,d) ! (ln)(ab)(ik)
                  if (k==m .and. i==l .and. a==d .and. c==f) hmatel = hmatel - h(b,n,j,e) ! (de)(ln)(ab)(ik)
                  if (k==m .and. i==l .and. a==e .and. c==d) hmatel = hmatel - h(b,n,j,f) ! (df)(ln)(ab)(ik)
                  if (k==m .and. i==n .and. b==e .and. a==f) hmatel = hmatel - h(c,l,j,d) ! (ac)(ik)
                  if (k==m .and. i==n .and. b==d .and. a==f) hmatel = hmatel + h(c,l,j,e) ! (de)(ac)(ik)
                  if (k==m .and. i==n .and. b==e .and. a==d) hmatel = hmatel + h(c,l,j,f) ! (df)(ac)(ik)
                  if (k==l .and. i==n .and. b==e .and. a==f) hmatel = hmatel + h(c,m,j,d) ! (lm)(ac)(ik)
                  if (k==l .and. i==n .and. b==d .and. a==f) hmatel = hmatel - h(c,m,j,e) ! (de)(lm)(ac)(ik)
                  if (k==l .and. i==n .and. b==e .and. a==d) hmatel = hmatel - h(c,m,j,f) ! (df)(lm)(ac)(ik)
                  if (k==m .and. i==l .and. b==e .and. a==f) hmatel = hmatel + h(c,n,j,d) ! (ln)(ac)(ik)
                  if (k==m .and. i==l .and. b==d .and. a==f) hmatel = hmatel - h(c,n,j,e) ! (de)(ln)(ac)(ik)
                  if (k==m .and. i==l .and. b==e .and. a==d) hmatel = hmatel - h(c,n,j,f) ! (df)(ln)(ac)(ik)
                  ! (jk)(bc), apply(jk)
                  if (k==m .and. j==n .and. c==e .and. b==f) hmatel = hmatel + h(a,l,i,d) ! (1)
                  if (k==m .and. j==n .and. c==d .and. b==f) hmatel = hmatel - h(a,l,i,e) ! (de)
                  if (k==m .and. j==n .and. c==e .and. b==d) hmatel = hmatel - h(a,l,i,f) ! (df)
                  if (k==l .and. j==n .and. c==e .and. b==f) hmatel = hmatel - h(a,m,i,d) ! (lm)
                  if (k==l .and. j==n .and. c==d .and. b==f) hmatel = hmatel + h(a,m,i,e) ! (de)(lm)
                  if (k==l .and. j==n .and. c==e .and. b==d) hmatel = hmatel + h(a,m,i,f) ! (df)(lm)
                  if (k==m .and. j==l .and. c==e .and. b==f) hmatel = hmatel - h(a,n,i,d) ! (ln)
                  if (k==m .and. j==l .and. c==d .and. b==f) hmatel = hmatel + h(a,n,i,e) ! (de)(ln)
                  if (k==m .and. j==l .and. c==e .and. b==d) hmatel = hmatel + h(a,n,i,f) ! (df)(ln)
                  if (k==m .and. j==n .and. a==e .and. b==f) hmatel = hmatel - h(c,l,i,d) ! (ab)
                  if (k==m .and. j==n .and. a==d .and. b==f) hmatel = hmatel + h(c,l,i,e) ! (de)(ab)
                  if (k==m .and. j==n .and. a==e .and. b==d) hmatel = hmatel + h(c,l,i,f) ! (df)(ab)
                  if (k==l .and. j==n .and. a==e .and. b==f) hmatel = hmatel + h(c,m,i,d) ! (lm)(ab)
                  if (k==l .and. j==n .and. a==d .and. b==f) hmatel = hmatel - h(c,m,i,e) ! (de)(lm)(ab)
                  if (k==l .and. j==n .and. a==e .and. b==d) hmatel = hmatel - h(c,m,i,f) ! (df)(lm)(ab)
                  if (k==m .and. j==l .and. a==e .and. b==f) hmatel = hmatel + h(c,n,i,d) ! (ln)(ab)
                  if (k==m .and. j==l .and. a==d .and. b==f) hmatel = hmatel - h(c,n,i,e) ! (de)(ln)(ab)
                  if (k==m .and. j==l .and. a==e .and. b==d) hmatel = hmatel - h(c,n,i,f) ! (df)(ln)(ab)
                  if (k==m .and. j==n .and. c==e .and. a==f) hmatel = hmatel - h(b,l,i,d) ! (ac)
                  if (k==m .and. j==n .and. c==d .and. a==f) hmatel = hmatel + h(b,l,i,e) ! (de)(ac)
                  if (k==m .and. j==n .and. c==e .and. a==d) hmatel = hmatel + h(b,l,i,f) ! (df)(ac)
                  if (k==l .and. j==n .and. c==e .and. a==f) hmatel = hmatel + h(b,m,i,d) ! (lm)(ac)
                  if (k==l .and. j==n .and. c==d .and. a==f) hmatel = hmatel - h(b,m,i,e) ! (de)(lm)(ac)
                  if (k==l .and. j==n .and. c==e .and. a==d) hmatel = hmatel - h(b,m,i,f) ! (df)(lm)(ac)
                  if (k==m .and. j==l .and. c==e .and. a==f) hmatel = hmatel + h(b,n,i,d) ! (ln)(ac)
                  if (k==m .and. j==l .and. c==d .and. a==f) hmatel = hmatel - h(b,n,i,e) ! (de)(ln)(ac)
                  if (k==m .and. j==l .and. c==e .and. a==d) hmatel = hmatel - h(b,n,i,f) ! (df)(ln)(ac)
                  if (i==m .and. j==n .and. c==e .and. b==f) hmatel = hmatel - h(a,l,k,d) ! (ij)
                  if (i==m .and. j==n .and. c==d .and. b==f) hmatel = hmatel + h(a,l,k,e) ! (de)(ij)
                  if (i==m .and. j==n .and. c==e .and. b==d) hmatel = hmatel + h(a,l,k,f) ! (df)(ij)
                  if (i==l .and. j==n .and. c==e .and. b==f) hmatel = hmatel + h(a,m,k,d) ! (lm)(ij)
                  if (i==l .and. j==n .and. c==d .and. b==f) hmatel = hmatel - h(a,m,k,e) ! (de)(lm)(ij)
                  if (i==l .and. j==n .and. c==e .and. b==d) hmatel = hmatel - h(a,m,k,f) ! (df)(lm)(ij)
                  if (i==m .and. j==l .and. c==e .and. b==f) hmatel = hmatel + h(a,n,k,d) ! (ln)(ij)
                  if (i==m .and. j==l .and. c==d .and. b==f) hmatel = hmatel - h(a,n,k,e) ! (de)(ln)(ij)
                  if (i==m .and. j==l .and. c==e .and. b==d) hmatel = hmatel - h(a,n,k,f) ! (df)(ln)(ij)
                  if (i==m .and. j==n .and. a==e .and. b==f) hmatel = hmatel + h(c,l,k,d) ! (ab)(ij)
                  if (i==m .and. j==n .and. a==d .and. b==f) hmatel = hmatel - h(c,l,k,e) ! (de)(ab)(ij)
                  if (i==m .and. j==n .and. a==e .and. b==d) hmatel = hmatel - h(c,l,k,f) ! (df)(ab)(ij)
                  if (i==l .and. j==n .and. a==e .and. b==f) hmatel = hmatel - h(c,m,k,d) ! (lm)(ab)(ij)
                  if (i==l .and. j==n .and. a==d .and. b==f) hmatel = hmatel + h(c,m,k,e) ! (de)(lm)(ab)(ij)
                  if (i==l .and. j==n .and. a==e .and. b==d) hmatel = hmatel + h(c,m,k,f) ! (df)(lm)(ab)(ij)
                  if (i==m .and. j==l .and. a==e .and. b==f) hmatel = hmatel - h(c,n,k,d) ! (ln)(ab)(ij)
                  if (i==m .and. j==l .and. a==d .and. b==f) hmatel = hmatel + h(c,n,k,e) ! (de)(ln)(ab)(ij)
                  if (i==m .and. j==l .and. a==e .and. b==d) hmatel = hmatel + h(c,n,k,f) ! (df)(ln)(ab)(ij)
                  if (i==m .and. j==n .and. c==e .and. a==f) hmatel = hmatel + h(b,l,k,d) ! (ac)(ij)
                  if (i==m .and. j==n .and. c==d .and. a==f) hmatel = hmatel - h(b,l,k,e) ! (de)(ac)(ij)
                  if (i==m .and. j==n .and. c==e .and. a==d) hmatel = hmatel - h(b,l,k,f) ! (df)(ac)(ij)
                  if (i==l .and. j==n .and. c==e .and. a==f) hmatel = hmatel - h(b,m,k,d) ! (lm)(ac)(ij)
                  if (i==l .and. j==n .and. c==d .and. a==f) hmatel = hmatel + h(b,m,k,e) ! (de)(lm)(ac)(ij)
                  if (i==l .and. j==n .and. c==e .and. a==d) hmatel = hmatel + h(b,m,k,f) ! (df)(lm)(ac)(ij)
                  if (i==m .and. j==l .and. c==e .and. a==f) hmatel = hmatel - h(b,n,k,d) ! (ln)(ac)(ij)
                  if (i==m .and. j==l .and. c==d .and. a==f) hmatel = hmatel + h(b,n,k,e) ! (de)(ln)(ac)(ij)
                  if (i==m .and. j==l .and. c==e .and. a==d) hmatel = hmatel + h(b,n,k,f) ! (df)(ln)(ac)(ij)
                  if (k==m .and. i==n .and. c==e .and. b==f) hmatel = hmatel - h(a,l,j,d) ! (ik)
                  if (k==m .and. i==n .and. c==d .and. b==f) hmatel = hmatel + h(a,l,j,e) ! (de)(ik)
                  if (k==m .and. i==n .and. c==e .and. b==d) hmatel = hmatel + h(a,l,j,f) ! (df)(ik)
                  if (k==l .and. i==n .and. c==e .and. b==f) hmatel = hmatel + h(a,m,j,d) ! (lm)(ik)
                  if (k==l .and. i==n .and. c==d .and. b==f) hmatel = hmatel - h(a,m,j,e) ! (de)(lm)(ik)
                  if (k==l .and. i==n .and. c==e .and. b==d) hmatel = hmatel - h(a,m,j,f) ! (df)(lm)(ik)
                  if (k==m .and. i==l .and. c==e .and. b==f) hmatel = hmatel + h(a,n,j,d) ! (ln)(ik)
                  if (k==m .and. i==l .and. c==d .and. b==f) hmatel = hmatel - h(a,n,j,e) ! (de)(ln)(ik)
                  if (k==m .and. i==l .and. c==e .and. b==d) hmatel = hmatel - h(a,n,j,f) ! (df)(ln)(ik)
                  if (k==m .and. i==n .and. a==e .and. b==f) hmatel = hmatel + h(c,l,j,d) ! (ab)(ik)
                  if (k==m .and. i==n .and. a==d .and. b==f) hmatel = hmatel - h(c,l,j,e) ! (de)(ab)(ik)
                  if (k==m .and. i==n .and. a==e .and. b==d) hmatel = hmatel - h(c,l,j,f) ! (df)(ab)(ik)
                  if (k==l .and. i==n .and. a==e .and. b==f) hmatel = hmatel - h(c,m,j,d) ! (lm)(ab)(ik)
                  if (k==l .and. i==n .and. a==d .and. b==f) hmatel = hmatel + h(c,m,j,e) ! (de)(lm)(ab)(ik)
                  if (k==l .and. i==n .and. a==e .and. b==d) hmatel = hmatel + h(c,m,j,f) ! (df)(lm)(ab)(ik)
                  if (k==m .and. i==l .and. a==e .and. b==f) hmatel = hmatel - h(c,n,j,d) ! (ln)(ab)(ik)
                  if (k==m .and. i==l .and. a==d .and. b==f) hmatel = hmatel + h(c,n,j,e) ! (de)(ln)(ab)(ik)
                  if (k==m .and. i==l .and. a==e .and. b==d) hmatel = hmatel + h(c,n,j,f) ! (df)(ln)(ab)(ik)
                  if (k==m .and. i==n .and. c==e .and. a==f) hmatel = hmatel + h(b,l,j,d) ! (ac)(ik)
                  if (k==m .and. i==n .and. c==d .and. a==f) hmatel = hmatel - h(b,l,j,e) ! (de)(ac)(ik)
                  if (k==m .and. i==n .and. c==e .and. a==d) hmatel = hmatel - h(b,l,j,f) ! (df)(ac)(ik)
                  if (k==l .and. i==n .and. c==e .and. a==f) hmatel = hmatel - h(b,m,j,d) ! (lm)(ac)(ik)
                  if (k==l .and. i==n .and. c==d .and. a==f) hmatel = hmatel + h(b,m,j,e) ! (de)(lm)(ac)(ik)
                  if (k==l .and. i==n .and. c==e .and. a==d) hmatel = hmatel + h(b,m,j,f) ! (df)(lm)(ac)(ik)
                  if (k==m .and. i==l .and. c==e .and. a==f) hmatel = hmatel - h(b,n,j,d) ! (ln)(ac)(ik)
                  if (k==m .and. i==l .and. c==d .and. a==f) hmatel = hmatel + h(b,n,j,e) ! (de)(ln)(ac)(ik)
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

              ! (1)
              if (j==m .and. i==l .and. a==d .and. b==e) hmatel = hmatel + h(c,n,k,f) ! (1)
              if (j==m .and. k==l .and. a==d .and. b==e) hmatel = hmatel - h(c,n,i,f) ! (ik)
              if (k==m .and. i==l .and. a==d .and. b==e) hmatel = hmatel - h(c,n,j,f) ! (jk)
              if (j==m .and. i==l .and. c==d .and. b==e) hmatel = hmatel - h(a,n,k,f) ! (ac)
              if (j==m .and. k==l .and. c==d .and. b==e) hmatel = hmatel + h(a,n,i,f) ! (ik)(ac)
              if (k==m .and. i==l .and. c==d .and. b==e) hmatel = hmatel + h(a,n,j,f) ! (jk)(ac)
              if (j==m .and. i==l .and. a==d .and. c==e) hmatel = hmatel - h(b,n,k,f) ! (bc)
              if (j==m .and. k==l .and. a==d .and. c==e) hmatel = hmatel + h(b,n,i,f) ! (ik)(bc)
              if (k==m .and. i==l .and. a==d .and. c==e) hmatel = hmatel + h(b,n,j,f) ! (jk)(bc)
              ! (ij)
              if (i==m .and. j==l .and. a==d .and. b==e) hmatel = hmatel - h(c,n,k,f) ! (ij)
              if (i==m .and. k==l .and. a==d .and. b==e) hmatel = hmatel + h(c,n,j,f) ! (ik)(ij)
              if (k==m .and. j==l .and. a==d .and. b==e) hmatel = hmatel + h(c,n,i,f) ! (jk)(ij)
              if (i==m .and. j==l .and. c==d .and. b==e) hmatel = hmatel + h(a,n,k,f) ! (ac)(ij)
              if (i==m .and. k==l .and. c==d .and. b==e) hmatel = hmatel - h(a,n,j,f) ! (ik)(ac)(ij)
              if (k==m .and. j==l .and. c==d .and. b==e) hmatel = hmatel - h(a,n,i,f) ! (jk)(ac)(ij)
              if (i==m .and. j==l .and. a==d .and. c==e) hmatel = hmatel + h(b,n,k,f) ! (bc)(ij)
              if (i==m .and. k==l .and. a==d .and. c==e) hmatel = hmatel - h(b,n,j,f) ! (ik)(bc)(ij)
              if (k==m .and. j==l .and. a==d .and. c==e) hmatel = hmatel - h(b,n,i,f) ! (jk)(bc)(ij)
              ! (ab)
              if (j==m .and. i==l .and. b==d .and. a==e) hmatel = hmatel - h(c,n,k,f) ! (ab)
              if (j==m .and. k==l .and. b==d .and. a==e) hmatel = hmatel + h(c,n,i,f) ! (ik)(ab)
              if (k==m .and. i==l .and. b==d .and. a==e) hmatel = hmatel + h(c,n,j,f) ! (jk)(ab)
              if (j==m .and. i==l .and. c==d .and. a==e) hmatel = hmatel + h(b,n,k,f) ! (ac)(ab)
              if (j==m .and. k==l .and. c==d .and. a==e) hmatel = hmatel - h(b,n,i,f) ! (ik)(ac)(ab)
              if (k==m .and. i==l .and. c==d .and. a==e) hmatel = hmatel - h(b,n,j,f) ! (jk)(ac)(ab)
              if (j==m .and. i==l .and. b==d .and. c==e) hmatel = hmatel + h(a,n,k,f) ! (bc)(ab)
              if (j==m .and. k==l .and. b==d .and. c==e) hmatel = hmatel - h(a,n,i,f) ! (ik)(bc)(ab)
              if (k==m .and. i==l .and. b==d .and. c==e) hmatel = hmatel - h(a,n,j,f) ! (jk)(bc)(ab)
              ! (ij)(ab)
              if (i==m .and. j==l .and. b==d .and. a==e) hmatel = hmatel + h(c,n,k,f) ! (ab)(ij)
              if (i==m .and. k==l .and. b==d .and. a==e) hmatel = hmatel - h(c,n,j,f) ! (ik)(ab)(ij)
              if (k==m .and. j==l .and. b==d .and. a==e) hmatel = hmatel - h(c,n,i,f) ! (jk)(ab)(ij)
              if (i==m .and. j==l .and. c==d .and. a==e) hmatel = hmatel - h(b,n,k,f) ! (ac)(ab)(ij)
              if (i==m .and. k==l .and. c==d .and. a==e) hmatel = hmatel + h(b,n,j,f) ! (ik)(ac)(ab)(ij)
              if (k==m .and. j==l .and. c==d .and. a==e) hmatel = hmatel + h(b,n,i,f) ! (jk)(ac)(ab)(ij)
              if (i==m .and. j==l .and. b==d .and. c==e) hmatel = hmatel - h(a,n,k,f) ! (bc)(ab)(ij)
              if (i==m .and. k==l .and. b==d .and. c==e) hmatel = hmatel + h(a,n,j,f) ! (ik)(bc)(ab)(ij)
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
                    if (j==m .and. k==n) hmatel = hmatel - ha(l,i) ! (1)
                    if (i==m .and. k==n) hmatel = hmatel + ha(l,j) ! (ij)
                    if (j==l .and. k==n) hmatel = hmatel + ha(m,i) ! (lm)
                    if (i==l .and. k==n) hmatel = hmatel - ha(m,j) ! (ij)(lm)

                    if (i==l .and. j==m) hmatel = hmatel - hb(n,k) ! (1)
                  end if
          end function aab_oo_aab

end module ccp_quadratic_loops_direct
