module leftccsdt_p_loops

        implicit none

        contains
           
              subroutine build_LH_2A(resid,&
                                     X2A,&
                                     l3a_amps, l3a_excits,&
                                     l3b_amps, l3b_excits,&
                                     H2A_vooo, H2A_vvov,&
                                     H2B_ovoo, H2B_vvvo,&
                                     n3aaa, n3aab,&
                                     noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub, n3aaa, n3aab
                  integer, intent(in) :: l3a_excits(6, n3aaa), l3b_excits(6, n3aab)
                  real(kind=8), intent(in) :: l3a_amps(n3aaa), l3b_amps(n3aab)
                  real(kind=8), intent(in) :: X2A(1:nua,1:nua,1:noa,1:noa),&
                                              H2A_vooo(1:nua,1:noa,1:noa,1:noa),&
                                              H2A_vvov(1:nua,1:nua,1:noa,1:nua),&
                                              H2B_ovoo(1:noa,1:nub,1:noa,1:nob),&
                                              H2B_vvvo(1:nua,1:nub,1:nua,1:nob)

                  real(kind=8), intent(out) :: resid(1:nua,1:nua,1:noa,1:noa)

                  integer :: i, j, a, b, m, n, e, f, idet
                  real(kind=8) :: val, l_amp

                  ! Store x2a in residual container
                  resid(:,:,:,:) = x2a(:,:,:,:)
                  ! compute < 0 | (L3 * H(2))_C | ijab >
                  do idet = 1, n3aaa
                      l_amp = l3a_amps(idet)

                      ! A(ij)A(ab) [A(j/mn)A(f/ab) -h2a(finm) * l3a(abfmjn)]
                      a = l3a_excits(1,idet); b = l3a_excits(2,idet); f = l3a_excits(3,idet);
                      m = l3a_excits(4,idet); j = l3a_excits(5,idet); n = l3a_excits(6,idet);
                      resid(a,b,:,j) = resid(a,b,:,j) - H2A_vooo(f,:,n,m) * l_amp ! (1)
                      resid(a,b,:,m) = resid(a,b,:,m) + H2A_vooo(f,:,n,j) * l_amp ! (jm)
                      resid(a,b,:,n) = resid(a,b,:,n) + H2A_vooo(f,:,j,m) * l_amp ! (jn)
                      resid(f,b,:,j) = resid(f,b,:,j) + H2A_vooo(a,:,n,m) * l_amp ! (af)
                      resid(f,b,:,m) = resid(f,b,:,m) - H2A_vooo(a,:,n,j) * l_amp ! (jm)(af)
                      resid(f,b,:,n) = resid(f,b,:,n) - H2A_vooo(a,:,j,m) * l_amp ! (jn)(af)
                      resid(a,f,:,j) = resid(a,f,:,j) + H2A_vooo(b,:,n,m) * l_amp ! (bf)
                      resid(a,f,:,m) = resid(a,f,:,m) - H2A_vooo(b,:,n,j) * l_amp ! (jm)(bf)
                      resid(a,f,:,n) = resid(a,f,:,n) - H2A_vooo(b,:,j,m) * l_amp ! (jn)(bf)

                      ! A(ij)A(ab) [A(n/ij)A(b/ef) h2a(fena) * l3a(ebfijn)]
                      e = l3a_excits(1,idet); b = l3a_excits(2,idet); f = l3a_excits(3,idet);
                      i = l3a_excits(4,idet); j = l3a_excits(5,idet); n = l3a_excits(6,idet);
                      resid(:,b,i,j) = resid(:,b,i,j) + H2A_vvov(f,e,n,:) * l_amp ! (1)
                      resid(:,b,n,j) = resid(:,b,n,j) - H2A_vvov(f,e,i,:) * l_amp ! (in)
                      resid(:,b,i,n) = resid(:,b,i,n) - H2A_vvov(f,e,j,:) * l_amp ! (jn)
                      resid(:,e,i,j) = resid(:,e,i,j) - H2A_vvov(f,b,n,:) * l_amp ! (be)
                      resid(:,e,n,j) = resid(:,e,n,j) + H2A_vvov(f,b,i,:) * l_amp ! (in)(be)
                      resid(:,e,i,n) = resid(:,e,i,n) + H2A_vvov(f,b,j,:) * l_amp ! (jn)(be)
                      resid(:,f,i,j) = resid(:,f,i,j) - H2A_vvov(b,e,n,:) * l_amp ! (bf)
                      resid(:,f,n,j) = resid(:,f,n,j) + H2A_vvov(b,e,i,:) * l_amp ! (in)(bf)
                      resid(:,f,i,n) = resid(:,f,i,n) + H2A_vvov(b,e,j,:) * l_amp ! (jn)(bf)
                  end do
                  do idet = 1, n3aab
                      l_amp = l3b_amps(idet)

                      ! A(ij)A(ab) [A(jm) -h2b(ifmn) * l3b(abfmjn)]
                      a = l3b_excits(1,idet); b = l3b_excits(2,idet); f = l3b_excits(3,idet);
                      m = l3b_excits(4,idet); j = l3b_excits(5,idet); n = l3b_excits(6,idet);
                      resid(a,b,:,j) = resid(a,b,:,j) - H2B_ovoo(:,f,m,n) * l_amp ! (1)
                      resid(a,b,:,m) = resid(a,b,:,m) + H2B_ovoo(:,f,j,n) * l_amp ! (jm)

                      ! A(ij)A(ab) [A(be) h2b(efan) * l3b(ebfijn)]
                      e = l3b_excits(1,idet); b = l3b_excits(2,idet); f = l3b_excits(3,idet);
                      i = l3b_excits(4,idet); j = l3b_excits(5,idet); n = l3b_excits(6,idet);
                      resid(:,b,i,j) = resid(:,b,i,j) + H2B_vvvo(e,f,:,n) * l_amp ! (1)
                      resid(:,e,i,j) = resid(:,e,i,j) - H2B_vvvo(b,f,:,n) * l_amp ! (be)
                  end do
                  ! antisymmetrize (this replaces the x2a -= np.transpose(x2a, (...)) stuff in vector update
                  do i = 1, noa
                      do j = i+1, noa
                          do a = 1, nua
                              do b = a+1, nua
                                  val = resid(b,a,j,i) - resid(a,b,j,i) - resid(b,a,i,j) + resid(a,b,i,j)
                                  resid(b,a,j,i) =  val
                                  resid(a,b,j,i) = -val
                                  resid(b,a,i,j) = -val
                                  resid(a,b,i,j) =  val
                              end do
                          end do
                      end do
                  end do
                  ! (L3 * H(2))_C terms are vectorized and generally broadcast to diagonal elements, which should
                  ! be 0. Set them to 0 manually (you need to do this).
                  do a = 1, nua
                     resid(a,a,:,:) = 0.0d0
                  end do
                  do i = 1, noa
                     resid(:,:,i,i) = 0.0d0
                  end do

              end subroutine build_LH_2A
           
              subroutine build_LH_2B(resid,&
                                     X2B,&
                                     l3b_amps, l3b_excits,&
                                     l3c_amps, l3c_excits,&
                                     H2A_vooo, H2A_vvov,&
                                     H2B_vooo, H2B_ovoo, H2B_vvov, H2B_vvvo,&
                                     H2C_vooo, H2C_vvov,&
                                     n3aab, n3abb,&
                                     noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub, n3aab, n3abb
                  integer, intent(in) :: l3b_excits(6, n3aab), l3c_excits(6, n3abb)
                  real(kind=8), intent(in) :: l3b_amps(n3aab), l3c_amps(n3abb)
                  real(kind=8), intent(in) :: X2B(1:nua,1:nub,1:noa,1:nob),&
                                              H2A_vooo(1:nua,1:noa,1:noa,1:noa),&
                                              H2A_vvov(1:nua,1:nua,1:noa,1:nua),&
                                              H2B_vooo(1:nua,1:nob,1:noa,1:nob),&
                                              H2B_ovoo(1:noa,1:nub,1:noa,1:nob),&
                                              H2B_vvov(1:nua,1:nub,1:noa,1:nub),&
                                              H2B_vvvo(1:nua,1:nub,1:nua,1:nob),&
                                              H2C_vooo(1:nub,1:nob,1:nob,1:nob),&
                                              H2C_vvov(1:nub,1:nub,1:nob,1:nub)

                  real(kind=8), intent(out) :: resid(1:nua,1:nub,1:noa,1:nob)

                  integer :: i, j, a, b, m, n, e, f, idet
                  real(kind=8) :: l_amp

                  ! Store x2b in residual container
                  resid(:,:,:,:) = x2b(:,:,:,:)
                  ! compute < 0 | (L3 * H(2))_C | ij~ab~ >
                  do idet = 1, n3aab
                      l_amp = l3b_amps(idet)

                      ! A(af) -h2a(finm) * l3b(afbmnj)
                      a = l3b_excits(1,idet); f = l3b_excits(2,idet); b = l3b_excits(3,idet);
                      m = l3b_excits(4,idet); n = l3b_excits(5,idet); j = l3b_excits(6,idet);
                      resid(a,b,:,j) = resid(a,b,:,j) - H2A_vooo(f,:,n,m) * l_amp ! (1)
                      resid(f,b,:,j) = resid(f,b,:,j) + H2A_vooo(a,:,n,m) * l_amp ! (af)

                      ! A(af)A(in) -h2b(fjnm) * l3b(afbinm)
                      a = l3b_excits(1,idet); f = l3b_excits(2,idet); b = l3b_excits(3,idet);
                      i = l3b_excits(4,idet); n = l3b_excits(5,idet); m = l3b_excits(6,idet);
                      resid(a,b,i,:) = resid(a,b,i,:) - H2B_vooo(f,:,n,m) * l_amp ! (1)
                      resid(f,b,i,:) = resid(f,b,i,:) + H2B_vooo(a,:,n,m) * l_amp ! (af)
                      resid(a,b,n,:) = resid(a,b,n,:) + H2B_vooo(f,:,i,m) * l_amp ! (in)
                      resid(f,b,n,:) = resid(f,b,n,:) - H2B_vooo(a,:,i,m) * l_amp ! (af)(in)

                      ! A(in) h2a(fena) * l3b(efbinj)
                      e = l3b_excits(1,idet); f = l3b_excits(2,idet); b = l3b_excits(3,idet);
                      i = l3b_excits(4,idet); n = l3b_excits(5,idet); j = l3b_excits(6,idet);
                      resid(:,b,i,j) = resid(:,b,i,j) + H2A_vvov(f,e,n,:) * l_amp ! (1)
                      resid(:,b,n,j) = resid(:,b,n,j) - H2A_vvov(f,e,i,:) * l_amp ! (in)

                      ! A(af)A(in) h2b(fenb) * l3b(afeinj)
                      a = l3b_excits(1,idet); f = l3b_excits(2,idet); e = l3b_excits(3,idet);
                      i = l3b_excits(4,idet); n = l3b_excits(5,idet); j = l3b_excits(6,idet);
                      resid(a,:,i,j) = resid(a,:,i,j) + H2B_vvov(f,e,n,:) * l_amp ! (1)
                      resid(f,:,i,j) = resid(f,:,i,j) - H2B_vvov(a,e,n,:) * l_amp ! (af)
                      resid(a,:,n,j) = resid(a,:,n,j) - H2B_vvov(f,e,i,:) * l_amp ! (in)
                      resid(f,:,n,j) = resid(f,:,n,j) + H2B_vvov(a,e,i,:) * l_amp ! (af)(in)
                  end do
                  do idet = 1, n3abb
                      l_amp = l3c_amps(idet)

                      ! A(bf) -h2c(fjnm) * l3c(afbinm)
                      a = l3c_excits(1,idet); f = l3c_excits(2,idet); b = l3c_excits(3,idet);
                      i = l3c_excits(4,idet); n = l3c_excits(5,idet); m = l3c_excits(6,idet);
                      resid(a,b,i,:) = resid(a,b,i,:) - H2C_vooo(f,:,n,m) * l_amp ! (1)
                      resid(a,f,i,:) = resid(a,f,i,:) + H2C_vooo(b,:,n,m) * l_amp ! (bf)

                      ! A(bf)A(jn) -h2b(ifmn) * l3c(afbmnj)
                      a = l3c_excits(1,idet); f = l3c_excits(2,idet); b = l3c_excits(3,idet);
                      m = l3c_excits(4,idet); n = l3c_excits(5,idet); j = l3c_excits(6,idet);
                      resid(a,b,:,j) = resid(a,b,:,j) - H2B_ovoo(:,f,m,n) * l_amp ! (1)
                      resid(a,f,:,j) = resid(a,f,:,j) + H2B_ovoo(:,b,m,n) * l_amp ! (bf)
                      resid(a,b,:,n) = resid(a,b,:,n) + H2B_ovoo(:,f,m,j) * l_amp ! (jn)
                      resid(a,f,:,n) = resid(a,f,:,n) - H2B_ovoo(:,b,m,j) * l_amp ! (bf)(jn)

                      ! A(jn) h2c(fenb) * l3c(afeinj)
                      a = l3c_excits(1,idet); f = l3c_excits(2,idet); e = l3c_excits(3,idet);
                      i = l3c_excits(4,idet); n = l3c_excits(5,idet); j = l3c_excits(6,idet);
                      resid(a,:,i,j) = resid(a,:,i,j) + H2C_vvov(f,e,n,:) * l_amp ! (1)
                      resid(a,:,i,n) = resid(a,:,i,n) - H2C_vvov(f,e,j,:) * l_amp ! (jn)

                      ! A(bf)A(jn) h2b(efan) * l3c(efbinj)
                      e = l3c_excits(1,idet); f = l3c_excits(2,idet); b = l3c_excits(3,idet);
                      i = l3c_excits(4,idet); n = l3c_excits(5,idet); j = l3c_excits(6,idet);
                      resid(:,b,i,j) = resid(:,b,i,j) + H2B_vvvo(e,f,:,n) * l_amp ! (1)
                      resid(:,f,i,j) = resid(:,f,i,j) - H2B_vvvo(e,b,:,n) * l_amp ! (bf)
                      resid(:,b,i,n) = resid(:,b,i,n) - H2B_vvvo(e,f,:,j) * l_amp ! (jn)
                      resid(:,f,i,n) = resid(:,f,i,n) + H2B_vvvo(e,b,:,j) * l_amp ! (bf)(jn)
                  end do

              end subroutine build_LH_2B
           
              subroutine build_LH_2C(resid,&
                                     X2C,&
                                     l3c_amps, l3c_excits,&
                                     l3d_amps, l3d_excits,&
                                     H2C_vooo, H2C_vvov,&
                                     H2B_vooo, H2B_vvov,&
                                     n3abb, n3bbb,&
                                     noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub, n3abb, n3bbb
                  integer, intent(in) :: l3c_excits(6, n3abb), l3d_excits(6, n3bbb)
                  real(kind=8), intent(in) :: l3c_amps(n3abb), l3d_amps(n3bbb)
                  real(kind=8), intent(in) :: X2C(1:nub,1:nub,1:nob,1:nob),&
                                              H2C_vooo(1:nub,1:nob,1:nob,1:nob),&
                                              H2C_vvov(1:nub,1:nub,1:nob,1:nub),&
                                              H2B_vooo(1:nua,1:nob,1:noa,1:nob),&
                                              H2B_vvov(1:nua,1:nub,1:noa,1:nub)

                  real(kind=8), intent(out) :: resid(1:nub,1:nub,1:nob,1:nob)

                  integer :: i, j, a, b, m, n, e, f, idet
                  real(kind=8) :: val, l_amp

                  ! Store x2c in residual container
                  resid(:,:,:,:) = x2c(:,:,:,:)
                  ! compute < 0 | (L3 * H(2))_C | i~j~a~b~ >
                  do idet = 1, n3bbb
                      l_amp = l3d_amps(idet)

                      ! A(ij)A(ab) [A(j/mn)A(f/ab) -h2c(finm) * l3d(abfmjn)]
                      a = l3d_excits(1,idet); b = l3d_excits(2,idet); f = l3d_excits(3,idet);
                      m = l3d_excits(4,idet); j = l3d_excits(5,idet); n = l3d_excits(6,idet);
                      resid(a,b,:,j) = resid(a,b,:,j) - H2C_vooo(f,:,n,m) * l_amp ! (1)
                      resid(a,b,:,m) = resid(a,b,:,m) + H2C_vooo(f,:,n,j) * l_amp ! (jm)
                      resid(a,b,:,n) = resid(a,b,:,n) + H2C_vooo(f,:,j,m) * l_amp ! (jn)
                      resid(f,b,:,j) = resid(f,b,:,j) + H2C_vooo(a,:,n,m) * l_amp ! (af)
                      resid(f,b,:,m) = resid(f,b,:,m) - H2C_vooo(a,:,n,j) * l_amp ! (jm)(af)
                      resid(f,b,:,n) = resid(f,b,:,n) - H2C_vooo(a,:,j,m) * l_amp ! (jn)(af)
                      resid(a,f,:,j) = resid(a,f,:,j) + H2C_vooo(b,:,n,m) * l_amp ! (bf)
                      resid(a,f,:,m) = resid(a,f,:,m) - H2C_vooo(b,:,n,j) * l_amp ! (jm)(bf)
                      resid(a,f,:,n) = resid(a,f,:,n) - H2C_vooo(b,:,j,m) * l_amp ! (jn)(bf)

                      ! A(ij)A(ab) [A(n/ij)A(b/ef) h2c(fena) * l3d(ebfijn)]
                      e = l3d_excits(1,idet); b = l3d_excits(2,idet); f = l3d_excits(3,idet);
                      i = l3d_excits(4,idet); j = l3d_excits(5,idet); n = l3d_excits(6,idet);
                      resid(:,b,i,j) = resid(:,b,i,j) + H2C_vvov(f,e,n,:) * l_amp ! (1)
                      resid(:,b,n,j) = resid(:,b,n,j) - H2C_vvov(f,e,i,:) * l_amp ! (in)
                      resid(:,b,i,n) = resid(:,b,i,n) - H2C_vvov(f,e,j,:) * l_amp ! (jn)
                      resid(:,e,i,j) = resid(:,e,i,j) - H2C_vvov(f,b,n,:) * l_amp ! (be)
                      resid(:,e,n,j) = resid(:,e,n,j) + H2C_vvov(f,b,i,:) * l_amp ! (in)(be)
                      resid(:,e,i,n) = resid(:,e,i,n) + H2C_vvov(f,b,j,:) * l_amp ! (jn)(be)
                      resid(:,f,i,j) = resid(:,f,i,j) - H2C_vvov(b,e,n,:) * l_amp ! (bf)
                      resid(:,f,n,j) = resid(:,f,n,j) + H2C_vvov(b,e,i,:) * l_amp ! (in)(bf)
                      resid(:,f,i,n) = resid(:,f,i,n) + H2C_vvov(b,e,j,:) * l_amp ! (jn)(bf)
                  end do
                  do idet = 1, n3abb
                      l_amp = l3c_amps(idet)

                      ! A(ij)A(ab) [A(jm) -h2b(finm) * l3c(fbanjm)]
                      f = l3c_excits(1,idet); b = l3c_excits(2,idet); a = l3c_excits(3,idet);
                      n = l3c_excits(4,idet); j = l3c_excits(5,idet); m = l3c_excits(6,idet);
                      resid(a,b,:,j) = resid(a,b,:,j) - H2B_vooo(f,:,n,m) * l_amp ! (1)
                      resid(a,b,:,m) = resid(a,b,:,m) + H2B_vooo(f,:,n,j) * l_amp ! (jm)

                      ! A(ij)A(ab) [A(be) h2b(fena) * l3b(fbenji)]
                      f = l3c_excits(1,idet); b = l3c_excits(2,idet); e = l3c_excits(3,idet);
                      n = l3c_excits(4,idet); j = l3c_excits(5,idet); i = l3c_excits(6,idet);
                      resid(:,b,i,j) = resid(:,b,i,j) + H2B_vvov(f,e,n,:) * l_amp ! (1)
                      resid(:,e,i,j) = resid(:,e,i,j) - H2B_vvov(f,b,n,:) * l_amp ! (be)
                  end do
                  ! antisymmetrize (this replaces the x2c -= np.transpose(x2c, (...)) stuff in vector update
                  do i = 1, nob
                      do j = i+1, nob
                          do a = 1, nub
                              do b = a+1, nub
                                  val = resid(b,a,j,i) - resid(a,b,j,i) - resid(b,a,i,j) + resid(a,b,i,j)
                                  resid(b,a,j,i) =  val
                                  resid(a,b,j,i) = -val
                                  resid(b,a,i,j) = -val
                                  resid(a,b,i,j) =  val
                              end do
                          end do
                      end do
                  end do
                  ! (L3 * H(2))_C terms are vectorized and generally broadcast to diagonal elements, which should
                  ! be 0. Set them to 0 manually (you need to do this).
                  do a = 1, nub
                     resid(a,a,:,:) = 0.0d0
                  end do
                  do i = 1, nob
                     resid(:,:,i,i) = 0.0d0
                  end do

              end subroutine build_LH_2C
           
              subroutine build_LH_3A(resid,&
                                    l1a, l2a,&
                                    l3a_amps, l3a_excits,&
                                    l3b_amps, l3b_excits,&
                                    h1a_ov, h1a_oo, h1a_vv,&
                                    h2a_oooo, h2a_ooov, h2a_oovv,&
                                    h2a_voov, h2a_vovv, h2a_vvvv,&
                                    h2b_ovvo,&
                                    x2a_ooov, x2a_vovv,&
                                    n3aaa, n3aab,&
                                    noa, nua, nob, nub)
                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa, n3aab
                  ! Input L arrays
                  real(kind=8), intent(in) :: l1a(nua,noa)
                  real(kind=8), intent(in) :: l2a(nua,nua,noa,noa)
                  integer, intent(in) :: l3b_excits(6,n3aab)
                  real(kind=8), intent(in) :: l3b_amps(n3aab)
                  ! Input H and X arrays
                  real(kind=8), intent(in) :: h1a_ov(noa,nua)
                  real(kind=8), intent(in) :: h1a_oo(noa,noa)
                  real(kind=8), intent(in) :: h1a_vv(nua,nua)
                  real(kind=8), intent(in) :: h2a_oooo(noa,noa,noa,noa)
                  real(kind=8), intent(in) :: h2a_ooov(noa,noa,noa,nua)
                  real(kind=8), intent(in) :: h2a_oovv(noa,noa,nua,nua)
                  real(kind=8), intent(in) :: h2a_voov(nua,noa,noa,nua)
                  real(kind=8), intent(in) :: h2a_vovv(nua,noa,nua,nua)
                  real(kind=8), intent(in) :: h2a_vvvv(nua,nua,nua,nua)
                  real(kind=8), intent(in) :: h2b_ovvo(noa,nub,nua,nob)
                  real(kind=8), intent(in) :: x2a_ooov(noa,noa,noa,nua)
                  real(kind=8), intent(in) :: x2a_vovv(nua,noa,nua,nua)
                  ! Output and Inout variables
                  real(kind=8), intent(out) :: resid(n3aaa)
                  integer, intent(inout) :: l3a_excits(6,n3aaa)
                  !f2py intent(in,out) :: l3a_excits(6,0:n3aaa-1)
                  real(kind=8), intent(inout) :: l3a_amps(n3aaa)
                  !f2py intent(in,out) :: l3a_amps(0:n3aaa-1)
                  ! Local variables
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:)
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)
                  real(kind=8) :: l_amp, hmatel, hmatel1, res
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc

                  resid = 0.0d0
                  !!!! diagram 1: -A(i/jk) h1a(im) * l3a(abcmjk)
                  !!!! diagram 3: 1/2 A(k/ij) h2a(ijmn) * l3a(abcmnk)
                  ! NOTE: WITHIN THESE LOOPS, H1A(OO) TERMS ARE DOUBLE-COUNTED SO COMPENSATE BY FACTOR OF 1/2
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)*(nua-2)/6*noa
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nua,nua,nua,noa))
                  !!! SB: (1,2,3,6) !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/3,noa/), nua, nua, nua, noa)
                  call sort4(l3a_excits, l3a_amps, loc_arr, idx_table, (/1,2,3,6/), nua, nua, nua, noa, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits,&
                  !$omp l3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h1a_oo,h2a_oooo,&
                  !$omp noa,nua,n3aaa),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa
                     a = l3a_excits(1,idet); b = l3a_excits(2,idet); c = l3a_excits(3,idet);
                     i = l3a_excits(4,idet); j = l3a_excits(5,idet); k = l3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        l = l3a_excits(4,jdet); m = l3a_excits(5,jdet);
                        ! compute < lmkabc | h2a(oooo) | ijkabc >
                        hmatel = h2a_oooo(i,j,l,m)
                        ! compute < lmkabc | h1a(oo) | ijkabc > = -A(ij)A(lm) h1a_oo(i,l) * delta(j,m)
                        hmatel1 = 0.0d0
                        if (m==j) hmatel1 = hmatel1 - h1a_oo(i,l) ! (1)
                        if (m==i) hmatel1 = hmatel1 + h1a_oo(j,l) ! (ij)
                        if (l==j) hmatel1 = hmatel1 + h1a_oo(i,m) ! (lm)
                        if (l==i) hmatel1 = hmatel1 - h1a_oo(j,m) ! (ij)(lm)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ik)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l = l3a_excits(4,jdet); m = l3a_excits(5,jdet);
                           ! compute < lmiabc | h2a(oooo) | ijkabc >
                           hmatel = -h2a_oooo(k,j,l,m)
                           ! compute < lmiabc | h1a(oo) | ijkabc > = A(jk)A(lm) h1a_oo(k,l) * delta(j,m)
                           hmatel1 = 0.0d0
                           if (m==j) hmatel1 = hmatel1 + h1a_oo(k,l) ! (1)
                           if (m==k) hmatel1 = hmatel1 - h1a_oo(j,l) ! (jk)
                           if (l==j) hmatel1 = hmatel1 - h1a_oo(k,m) ! (lm)
                           if (l==k) hmatel1 = hmatel1 + h1a_oo(j,m) ! (jk)(lm)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l = l3a_excits(4,jdet); m = l3a_excits(5,jdet);
                           ! compute < lmjabc | h2a(oooo) | ijkabc >
                           hmatel = -h2a_oooo(i,k,l,m)
                           ! compute < lmjabc | h1a(oo) | ijkabc > = A(ik)A(lm) h1a_oo(i,l) * delta(k,m)
                           hmatel1 = 0.0d0
                           if (m==k) hmatel1 = hmatel1 + h1a_oo(i,l) ! (1)
                           if (m==i) hmatel1 = hmatel1 - h1a_oo(k,l) ! (ik)
                           if (l==k) hmatel1 = hmatel1 - h1a_oo(i,m) ! (lm)
                           if (l==i) hmatel1 = hmatel1 + h1a_oo(k,m) ! (ik)(lm)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,3,4) !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/1,noa-2/), nua, nua, nua, noa)
                  call sort4(l3a_excits, l3a_amps, loc_arr, idx_table, (/1,2,3,4/), nua, nua, nua, noa, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits,&
                  !$omp l3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_oo,H2A_oooo,&
                  !$omp noa,nua,n3aaa),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa
                     a = l3a_excits(1,idet); b = l3a_excits(2,idet); c = l3a_excits(3,idet);
                     i = l3a_excits(4,idet); j = l3a_excits(5,idet); k = l3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,i)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        m = l3a_excits(5,jdet); n = l3a_excits(6,jdet);
                        ! compute < imnabc | h2a(oooo) | ijkabc >
                        hmatel = h2a_oooo(j,k,m,n)
                        ! compute < imnabc | h1a(oo) | ijkabc > = -A(jk)A(mn) h1a_oo(j,m) * delta(k,n)
                        hmatel1 = 0.0d0
                        if (n==k) hmatel1 = hmatel1 - h1a_oo(j,m) ! (1)
                        if (n==j) hmatel1 = hmatel1 + h1a_oo(k,m) ! (jk)
                        if (m==k) hmatel1 = hmatel1 + h1a_oo(j,n) ! (mn)
                        if (m==j) hmatel1 = hmatel1 - h1a_oo(k,n) ! (jk)(mn)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           m = l3a_excits(5,jdet); n = l3a_excits(6,jdet);
                           ! compute < jmnabc | h2a(oooo) | ijkabc >
                           hmatel = -h2a_oooo(i,k,m,n)
                           ! compute < jmnabc | h1a(oo) | ijkabc > = A(ik)A(mn) h1a_oo(i,m) * delta(k,n)
                           hmatel1 = 0.0d0
                           if (n==k) hmatel1 = hmatel1 + h1a_oo(i,m) ! (1)
                           if (n==i) hmatel1 = hmatel1 - h1a_oo(k,m) ! (ik)
                           if (m==k) hmatel1 = hmatel1 - h1a_oo(i,n) ! (mn)
                           if (m==i) hmatel1 = hmatel1 + h1a_oo(k,n) ! (ik)(mn)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           m = l3a_excits(5,jdet); n = l3a_excits(6,jdet);
                           ! compute < kmnabc | h2a(oooo) | ijkabc >
                           hmatel = -h2a_oooo(j,i,m,n)
                           ! compute < kmnabc | h1a(oo) | ijkabc > = A(ij)A(mn) h1a_oo(j,m) * delta(i,n)
                           hmatel1 = 0.0d0
                           if (n==i) hmatel1 = hmatel1 - h1a_oo(j,m) ! (1)
                           if (n==j) hmatel1 = hmatel1 + h1a_oo(i,m) ! (ij)
                           if (m==i) hmatel1 = hmatel1 + h1a_oo(j,n) ! (mn)
                           if (m==j) hmatel1 = hmatel1 - h1a_oo(i,n) ! (ij)(mn)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,3,5) !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/2,noa-1/), nua, nua, nua, noa)
                  call sort4(l3a_excits, l3a_amps, loc_arr, idx_table, (/1,2,3,5/), nua, nua, nua, noa, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits,&
                  !$omp l3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_oo,H2A_oooo,&
                  !$omp noa,nua,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa
                     a = l3a_excits(1,idet); b = l3a_excits(2,idet); c = l3a_excits(3,idet);
                     i = l3a_excits(4,idet); j = l3a_excits(5,idet); k = l3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,j)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        l = l3a_excits(4,jdet); n = l3a_excits(6,jdet);
                        ! compute < ljnabc | h2a(oooo) | ijkabc >
                        hmatel = h2a_oooo(i,k,l,n)
                        ! compute < ljnabc | h1a(oo) | ijkabc > = -A(ik)A(ln) h1a_oo(i,l) * delta(k,n)
                        hmatel1 = 0.0d0
                        if (n==k) hmatel1 = hmatel1 - h1a_oo(i,l) ! (1)
                        if (n==i) hmatel1 = hmatel1 + h1a_oo(k,l) ! (ik)
                        if (l==k) hmatel1 = hmatel1 + h1a_oo(i,n) ! (ln)
                        if (l==i) hmatel1 = hmatel1 - h1a_oo(k,n) ! (ik)(ln)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l = l3a_excits(4,jdet); n = l3a_excits(6,jdet);
                           ! compute < linabc | h2a(oooo) | ijkabc >
                           hmatel = -h2a_oooo(j,k,l,n)
                           ! compute < linabc | h1a(oo) | ijkabc > = A(jk)A(ln) h1a_oo(j,l) * delta(k,n)
                           hmatel1 = 0.0d0
                           if (n==k) hmatel1 = hmatel1 + h1a_oo(j,l) ! (1)
                           if (n==j) hmatel1 = hmatel1 - h1a_oo(k,l) ! (jk)
                           if (l==k) hmatel1 = hmatel1 - h1a_oo(j,n) ! (ln)
                           if (l==j) hmatel1 = hmatel1 + h1a_oo(k,n) ! (jk)(ln)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l = l3a_excits(4,jdet); n = l3a_excits(6,jdet);
                           ! compute < lknabc | h2a(oooo) | ijkabc >
                           hmatel = -h2a_oooo(i,j,l,n)
                           ! compute < lknabc | h1a(oo) | ijkabc > = A(ij)A(ln) h1a_oo(i,l) * delta(j,n)
                           hmatel1 = 0.0d0
                           if (n==j) hmatel1 = hmatel1 + h1a_oo(i,l) ! (1)
                           if (n==i) hmatel1 = hmatel1 - h1a_oo(j,l) ! (ij)
                           if (l==j) hmatel1 = hmatel1 - h1a_oo(i,n) ! (ln)
                           if (l==i) hmatel1 = hmatel1 + h1a_oo(j,n) ! (ij)(ln)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 2: A(a/bc) h1a(ea) * l3a(ebcijk)
                  !!!! diagram 4: 1/2 A(c/ab) h2a(efab) * l3a(ebcijk)
                  ! NOTE: WITHIN THESE LOOPS, H1A(VV) TERMS ARE DOUBLE-COUNTED SO COMPENSATE BY FACTOR OF 1/2
                  ! allocate new sorting arrays
                  nloc = noa*(noa-1)*(noa-2)/6*nua
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(noa,noa,noa,nua))
                  !!! SB: (4,5,6,1) LOOP !!!
                  call get_index_table(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/1,nua-2/), noa, noa, noa, nua)
                  call sort4(l3a_excits, l3a_amps, loc_arr, idx_table, (/4,5,6,1/), noa, noa, noa, nua, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits,&
                  !$omp l3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_vv,H2A_vvvv,&
                  !$omp noa,nua,n3aaa),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa
                     a = l3a_excits(1,idet); b = l3a_excits(2,idet); c = l3a_excits(3,idet);
                     i = l3a_excits(4,idet); j = l3a_excits(5,idet); k = l3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,a)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); f = l3a_excits(3,jdet);
                        ! compute < ijkaef | h2a(vvvv) | ijkabc >
                        hmatel = h2a_vvvv(e,f,b,c)
                        ! compute < ijkaef | h1a(vv) | ijkabc > = A(bc)A(ef) h1a_vv(e,b) * delta(f,c)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 + h1a_vv(e,b) ! (1)
                        if (b==f) hmatel1 = hmatel1 - h1a_vv(e,c) ! (bc)
                        if (c==e) hmatel1 = hmatel1 - h1a_vv(f,b) ! (ef)
                        if (b==e) hmatel1 = hmatel1 + h1a_vv(f,c) ! (bc)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); f = l3a_excits(3,jdet);
                        ! compute < ijkbef | h2a(vvvv) | ijkabc >
                        hmatel = -h2a_vvvv(e,f,a,c)
                        ! compute < ijkbef | h1a(vv) | ijkabc > = -A(ac)A(ef) h1a_vv(e,a) * delta(f,c)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 - h1a_vv(e,a) ! (1)
                        if (a==f) hmatel1 = hmatel1 + h1a_vv(e,c) ! (ac)
                        if (c==e) hmatel1 = hmatel1 + h1a_vv(f,a) ! (ef)
                        if (a==e) hmatel1 = hmatel1 - h1a_vv(f,c) ! (ac)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); f = l3a_excits(3,jdet);
                        ! compute < ijkcef | h2a(vvvv) | ijkabc >
                        hmatel = -h2a_vvvv(e,f,b,a)
                        ! compute < ijkcef | h1a(vv) | ijkabc > = -A(ab)A(ef) h1a_vv(e,b) * delta(f,a)
                        hmatel1 = 0.0d0
                        if (a==f) hmatel1 = hmatel1 - h1a_vv(e,b) ! (1)
                        if (b==f) hmatel1 = hmatel1 + h1a_vv(e,a) ! (ab)
                        if (a==e) hmatel1 = hmatel1 + h1a_vv(f,b) ! (ef)
                        if (b==e) hmatel1 = hmatel1 - h1a_vv(f,a) ! (ab)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,6,2) LOOP !!!
                  call get_index_table(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/2,nua-1/), noa, noa, noa, nua)
                  call sort4(l3a_excits, l3a_amps, loc_arr, idx_table, (/4,5,6,2/), noa, noa, noa, nua, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits,&
                  !$omp l3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_vv,H2A_vvvv,&
                  !$omp noa,nua,n3aaa),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa
                     a = l3a_excits(1,idet); b = l3a_excits(2,idet); c = l3a_excits(3,idet);
                     i = l3a_excits(4,idet); j = l3a_excits(5,idet); k = l3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,b)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); f = l3a_excits(3,jdet);
                        ! compute < ijkdbf | h2a(vvvv) | ijkabc >
                        hmatel = h2a_vvvv(d,f,a,c)
                        ! compute < ijkdbf | h1a(vv) | ijkabc > = A(ac)A(df) h1a_vv(d,a) * delta(f,c)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 + h1a_vv(d,a) ! (1)
                        if (a==f) hmatel1 = hmatel1 - h1a_vv(d,c) ! (ac)
                        if (c==d) hmatel1 = hmatel1 - h1a_vv(f,a) ! (df)
                        if (a==d) hmatel1 = hmatel1 + h1a_vv(f,c) ! (ac)(df)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); f = l3a_excits(3,jdet);
                        ! compute < ijkdaf | h2a(vvvv) | ijkabc >
                        hmatel = -h2a_vvvv(d,f,b,c)
                        ! compute < ijkdaf | h1a(vv) | ijkabc > = -A(bc)A(df) h1a_vv(d,b) * delta(f,c)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 - h1a_vv(d,b) ! (1)
                        if (b==f) hmatel1 = hmatel1 + h1a_vv(d,c) ! (bc)
                        if (c==d) hmatel1 = hmatel1 + h1a_vv(f,b) ! (df)
                        if (b==d) hmatel1 = hmatel1 - h1a_vv(f,c) ! (bc)(df)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); f = l3a_excits(3,jdet);
                        ! compute < ijkdcf | h2a(vvvv) | ijkabc >
                        hmatel = -h2a_vvvv(d,f,a,b)
                        ! compute < ijkdcf | h1a(vv) | ijkabc > = -A(ab)A(df) h1a_vv(d,a) * delta(f,b)
                        hmatel1 = 0.0d0
                        if (b==f) hmatel1 = hmatel1 - h1a_vv(d,a) ! (1)
                        if (a==f) hmatel1 = hmatel1 + h1a_vv(d,b) ! (ab)
                        if (b==d) hmatel1 = hmatel1 + h1a_vv(f,a) ! (df)
                        if (a==d) hmatel1 = hmatel1 - h1a_vv(f,b) ! (ab)(df)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,6,3) LOOP !!!
                  call get_index_table(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/3,nua/), noa, noa, noa, nua)
                  call sort4(l3a_excits, l3a_amps, loc_arr, idx_table, (/4,5,6,3/), noa, noa, noa, nua, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits,&
                  !$omp l3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_vv,H2A_vvvv,&
                  !$omp noa,nua,n3aaa),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa
                     a = l3a_excits(1,idet); b = l3a_excits(2,idet); c = l3a_excits(3,idet);
                     i = l3a_excits(4,idet); j = l3a_excits(5,idet); k = l3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,c)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); e = l3a_excits(2,jdet);
                        ! compute < ijkdec | h2a(vvvv) | ijkabc >
                        hmatel = h2a_vvvv(d,e,a,b)
                        ! compute < ijkdec | h1a(vv) | ijkabc > = A(ab)A(de) h1a_vv(d,a) * delta(e,b)
                        hmatel1 = 0.0d0
                        if (b==e) hmatel1 = hmatel1 + h1a_vv(d,a) ! (1)
                        if (a==e) hmatel1 = hmatel1 - h1a_vv(d,b) ! (ab)
                        if (b==d) hmatel1 = hmatel1 - h1a_vv(e,a) ! (de)
                        if (a==d) hmatel1 = hmatel1 + h1a_vv(e,b) ! (ab)(de)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); e = l3a_excits(2,jdet);
                        ! compute < ijkdea | h2a(vvvv) | ijkabc >
                        hmatel = -h2a_vvvv(d,e,c,b)
                        ! compute < ijkdea | h1a(vv) | ijkabc > = -A(bc)A(de) h1a_vv(d,c) * delta(e,b)
                        hmatel1 = 0.0d0
                        if (b==e) hmatel1 = hmatel1 - h1a_vv(d,c) ! (1)
                        if (c==e) hmatel1 = hmatel1 + h1a_vv(d,b) ! (bc)
                        if (b==d) hmatel1 = hmatel1 + h1a_vv(e,c) ! (de)
                        if (c==d) hmatel1 = hmatel1 - h1a_vv(e,b) ! (bc)(de)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); e = l3a_excits(2,jdet);
                        ! compute < ijkdeb | h2a(vvvv) | ijkabc >
                        hmatel = -h2a_vvvv(d,e,a,c)
                        ! compute < ijkdeb | h1a(vv) | ijkabc > = -A(ac)A(de) h1a_vv(d,a) * delta(e,c)
                        hmatel1 = 0.0d0
                        if (c==e) hmatel1 = hmatel1 - h1a_vv(d,a) ! (1)
                        if (a==e) hmatel1 = hmatel1 + h1a_vv(d,c) ! (ac)
                        if (c==d) hmatel1 = hmatel1 + h1a_vv(e,a) ! (de)
                        if (a==d) hmatel1 = hmatel1 - h1a_vv(e,c) ! (ac)(de)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  
                  !!!! diagram 5: A(i/jk)A(a/bc) h2a(eima) * l3a(ebcmjk)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = (nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nua,nua,noa,noa))
                  !!! SB: (1,2,4,5) LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(l3a_excits, l3a_amps, loc_arr, idx_table, (/1,2,4,5/), nua, nua, noa, noa, nloc, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = l3a_excits(1,idet); b = l3a_excits(2,idet); c = l3a_excits(3,idet);
                     i = l3a_excits(4,idet); j = l3a_excits(5,idet); k = l3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3a_excits(3,jdet); n = l3a_excits(6,jdet);
                        ! compute < ijnabf | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(f,k,n,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3a_excits(3,jdet); n = l3a_excits(6,jdet);
                        ! compute < ijnbcf | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(f,k,n,a)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3a_excits(3,jdet); n = l3a_excits(6,jdet);
                        ! compute < ijnacf | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(f,k,n,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3a_excits(3,jdet); n = l3a_excits(6,jdet);
                        ! compute < jknabf | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(f,i,n,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3a_excits(3,jdet); n = l3a_excits(6,jdet);
                        ! compute < jknbcf | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(f,i,n,a)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3a_excits(3,jdet); n = l3a_excits(6,jdet);
                        ! compute < jknacf | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(f,i,n,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3a_excits(3,jdet); n = l3a_excits(6,jdet);
                        ! compute < iknabf | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(f,j,n,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3a_excits(3,jdet); n = l3a_excits(6,jdet);
                        ! compute < iknbcf | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(f,j,n,a)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3a_excits(3,jdet); n = l3a_excits(6,jdet);
                        ! compute < iknacf | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(f,j,n,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,3,4,5) LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(l3a_excits, l3a_amps, loc_arr, idx_table, (/1,3,4,5/), nua, nua, noa, noa, nloc, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = l3a_excits(1,idet); b = l3a_excits(2,idet); c = l3a_excits(3,idet);
                     i = l3a_excits(4,idet); j = l3a_excits(5,idet); k = l3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); n = l3a_excits(6,jdet);
                        ! compute < ijnaec | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(e,k,n,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); n = l3a_excits(6,jdet);
                        ! compute < ijnbec | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(e,k,n,a)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); n = l3a_excits(6,jdet);
                        ! compute < ijnaeb | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(e,k,n,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); n = l3a_excits(6,jdet);
                        ! compute < jknaec | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(e,i,n,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); n = l3a_excits(6,jdet);
                        ! compute < jknbec | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(e,i,n,a)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); n = l3a_excits(6,jdet);
                        ! compute < jknaeb | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(e,i,n,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); n = l3a_excits(6,jdet);
                        ! compute < iknaec | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(e,j,n,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); n = l3a_excits(6,jdet);
                        ! compute < iknbec | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(e,j,n,a)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); n = l3a_excits(6,jdet);
                        ! compute < iknaeb | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(e,j,n,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (2,3,4,5) LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(l3a_excits, l3a_amps, loc_arr, idx_table, (/2,3,4,5/), nua, nua, noa, noa, nloc, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = l3a_excits(1,idet); b = l3a_excits(2,idet); c = l3a_excits(3,idet);
                     i = l3a_excits(4,idet); j = l3a_excits(5,idet); k = l3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); n = l3a_excits(6,jdet);
                        ! compute < ijndbc | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(d,k,n,a)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); n = l3a_excits(6,jdet);
                        ! compute < ijndac | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(d,k,n,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); n = l3a_excits(6,jdet);
                        ! compute < ijndab | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(d,k,n,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); n = l3a_excits(6,jdet);
                        ! compute < jkndbc | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(d,i,n,a)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); n = l3a_excits(6,jdet);
                        ! compute < jkndac | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(d,i,n,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); n = l3a_excits(6,jdet);
                        ! compute < jkndab | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(d,i,n,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); n = l3a_excits(6,jdet);
                        ! compute < ikndbc | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(d,j,n,a)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); n = l3a_excits(6,jdet);
                        ! compute < ikndac | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(d,j,n,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); n = l3a_excits(6,jdet);
                        ! compute < ikndab | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(d,j,n,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,2,4,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(l3a_excits, l3a_amps, loc_arr, idx_table, (/1,2,4,6/), nua, nua, noa, noa, nloc, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = l3a_excits(1,idet); b = l3a_excits(2,idet); c = l3a_excits(3,idet);
                     i = l3a_excits(4,idet); j = l3a_excits(5,idet); k = l3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3a_excits(3,jdet); m = l3a_excits(5,jdet);
                        ! compute < imkabf | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(f,j,m,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3a_excits(3,jdet); m = l3a_excits(5,jdet);
                        ! compute < imkbcf | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(f,j,m,a)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3a_excits(3,jdet); m = l3a_excits(5,jdet);
                        ! compute < imkacf | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(f,j,m,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3a_excits(3,jdet); m = l3a_excits(5,jdet);
                        ! compute < jmkabf | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(f,i,m,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3a_excits(3,jdet); m = l3a_excits(5,jdet);
                        ! compute < jmkbcf | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(f,i,m,a)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3a_excits(3,jdet); m = l3a_excits(5,jdet);
                        ! compute < jmkacf | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(f,i,m,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3a_excits(3,jdet); m = l3a_excits(5,jdet);
                        ! compute < imjabf | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(f,k,m,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3a_excits(3,jdet); m = l3a_excits(5,jdet);
                        ! compute < imjbcf | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(f,k,m,a)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3a_excits(3,jdet); m = l3a_excits(5,jdet);
                        ! compute < imjacf | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(f,k,m,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,3,4,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(l3a_excits, l3a_amps, loc_arr, idx_table, (/1,3,4,6/), nua, nua, noa, noa, nloc, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = l3a_excits(1,idet); b = l3a_excits(2,idet); c = l3a_excits(3,idet);
                     i = l3a_excits(4,idet); j = l3a_excits(5,idet); k = l3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); m = l3a_excits(5,jdet);
                        ! compute < imkaec | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(e,j,m,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); m = l3a_excits(5,jdet);
                        ! compute < imkbec | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(e,j,m,a)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); m = l3a_excits(5,jdet);
                        ! compute < imkaeb | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(e,j,m,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); m = l3a_excits(5,jdet);
                        ! compute < jmkaec | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(e,i,m,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); m = l3a_excits(5,jdet);
                        ! compute < jmkbec | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(e,i,m,a)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); m = l3a_excits(5,jdet);
                        ! compute < jmkaeb | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(e,i,m,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); m = l3a_excits(5,jdet);
                        ! compute < imjaec | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(e,k,m,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); m = l3a_excits(5,jdet);
                        ! compute < imjbec | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(e,k,m,a)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); m = l3a_excits(5,jdet);
                        ! compute < imjaeb | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(e,k,m,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (2,3,4,6) LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(l3a_excits, l3a_amps, loc_arr, idx_table, (/2,3,4,6/), nua, nua, noa, noa, nloc, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = l3a_excits(1,idet); b = l3a_excits(2,idet); c = l3a_excits(3,idet);
                     i = l3a_excits(4,idet); j = l3a_excits(5,idet); k = l3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); m = l3a_excits(5,jdet);
                        ! compute < imkdbc | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(d,j,m,a)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); m = l3a_excits(5,jdet);
                        ! compute < imkdac | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(d,j,m,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); m = l3a_excits(5,jdet);
                        ! compute < imkdab | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(d,j,m,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); m = l3a_excits(5,jdet);
                        ! compute < jmkdbc | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(d,i,m,a)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); m = l3a_excits(5,jdet);
                        ! compute < jmkdac | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(d,i,m,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); m = l3a_excits(5,jdet);
                        ! compute < jmkdab | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(d,i,m,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); m = l3a_excits(5,jdet);
                        ! compute < imjdbc | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(d,k,m,a)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); m = l3a_excits(5,jdet);
                        ! compute < imjdac | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(d,k,m,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); m = l3a_excits(5,jdet);
                        ! compute < imjdab | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(d,k,m,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,2,5,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(l3a_excits, l3a_amps, loc_arr, idx_table, (/1,2,5,6/), nua, nua, noa, noa, nloc, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = l3a_excits(1,idet); b = l3a_excits(2,idet); c = l3a_excits(3,idet);
                     i = l3a_excits(4,idet); j = l3a_excits(5,idet); k = l3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3a_excits(3,jdet); l = l3a_excits(4,jdet);
                        ! compute < ljkabf | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(f,i,l,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3a_excits(3,jdet); l = l3a_excits(4,jdet);
                        ! compute < ljkbcf | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(f,i,l,a)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3a_excits(3,jdet); l = l3a_excits(4,jdet);
                        ! compute < ljkacf | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(f,i,l,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3a_excits(3,jdet); l = l3a_excits(4,jdet);
                        ! compute < likabf | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(f,j,l,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3a_excits(3,jdet); l = l3a_excits(4,jdet);
                        ! compute < likbcf | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(f,j,l,a)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3a_excits(3,jdet); l = l3a_excits(4,jdet);
                        ! compute < likacf | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(f,j,l,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3a_excits(3,jdet); l = l3a_excits(4,jdet);
                        ! compute < lijabf | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(f,k,l,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3a_excits(3,jdet); l = l3a_excits(4,jdet);
                        ! compute < lijbcf | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(f,k,l,a)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3a_excits(3,jdet); l = l3a_excits(4,jdet);
                        ! compute < lijacf | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(f,k,l,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,3,5,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(l3a_excits, l3a_amps, loc_arr, idx_table, (/1,3,5,6/), nua, nua, noa, noa, nloc, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = l3a_excits(1,idet); b = l3a_excits(2,idet); c = l3a_excits(3,idet);
                     i = l3a_excits(4,idet); j = l3a_excits(5,idet); k = l3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); l = l3a_excits(4,jdet);
                        ! compute < ljkaec | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(e,i,l,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); l = l3a_excits(4,jdet);
                        ! compute < ljkbec | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(e,i,l,a)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); l = l3a_excits(4,jdet);
                        ! compute < ljkaeb | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(e,i,l,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); l = l3a_excits(4,jdet);
                        ! compute < likaec | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(e,j,l,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); l = l3a_excits(4,jdet);
                        ! compute < likbec | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(e,j,l,a)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); l = l3a_excits(4,jdet);
                        ! compute < likaeb | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(e,j,l,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); l = l3a_excits(4,jdet);
                        ! compute < lijaec | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(e,k,l,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); l = l3a_excits(4,jdet);
                        ! compute < lijbec | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(e,k,l,a)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3a_excits(2,jdet); l = l3a_excits(4,jdet);
                        ! compute < lijaeb | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(e,k,l,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (2,3,5,6) LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(l3a_excits, l3a_amps, loc_arr, idx_table, (/2,3,5,6/), nua, nua, noa, noa, nloc, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = l3a_excits(1,idet); b = l3a_excits(2,idet); c = l3a_excits(3,idet);
                     i = l3a_excits(4,idet); j = l3a_excits(5,idet); k = l3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); l = l3a_excits(4,jdet);
                        ! compute < ljkdbc | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(d,i,l,a)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); l = l3a_excits(4,jdet);
                        ! compute < ljkdac | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(d,i,l,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); l = l3a_excits(4,jdet);
                        ! compute < ljkdab | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(d,i,l,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); l = l3a_excits(4,jdet);
                        ! compute < likdbc | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(d,j,l,a)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); l = l3a_excits(4,jdet);
                        ! compute < likdac | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(d,j,l,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); l = l3a_excits(4,jdet);
                        ! compute < likdab | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(d,j,l,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); l = l3a_excits(4,jdet);
                        ! compute < lijdbc | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(d,k,l,a)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); l = l3a_excits(4,jdet);
                        ! compute < lijdac | h2a(voov) | ijkabc >
                        hmatel = -h2a_voov(d,k,l,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3a_excits(1,jdet); l = l3a_excits(4,jdet);
                        ! compute < lijdab | h2a(voov) | ijkabc >
                        hmatel = h2a_voov(d,k,l,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)                  
                 
                  !!!! diagram 6: A(i/jk)A(a/bc) h2b(ieam) * l3b(bcejkm)
                  ! allocate and copy over l3b arrays
                  allocate(amps_buff(n3aab),excits_buff(6,n3aab))
                  amps_buff(:) = l3b_amps(:)
                  excits_buff(:,:) = l3b_excits(:,:)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = nua*(nua-1)/2*noa*(noa-1)/2
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nua,nua,noa,noa))
                  !!! SB: (1,2,4,5) LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,5/), nua, nua, noa, noa, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa
                     a = l3a_excits(1,idet); b = l3a_excits(2,idet); c = l3a_excits(3,idet);
                     i = l3a_excits(4,idet); j = l3a_excits(5,idet); k = l3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijn~abf~ | h2b(ovvo) | ijkabc >
                        hmatel = h2b_ovvo(k,f,c,n)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < jkn~abf~ | h2b(ovvo) | ijkabc >
                        hmatel = h2b_ovvo(i,f,c,n)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ikn~abf~ | h2b(ovvo) | ijkabc >
                        hmatel = -h2b_ovvo(j,f,c,n)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijn~bcf~ | h2b(ovvo) | ijkabc >
                        hmatel = h2b_ovvo(k,f,a,n)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik)(ac)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < jkn~bcf~ | h2b(ovvo) | ijkabc >
                        hmatel = h2b_ovvo(i,f,a,n)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)(ac)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ikn~bcf~ | h2b(ovvo) | ijkabc >
                        hmatel = -h2b_ovvo(j,f,a,n)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijn~acf~ | h2b(ovvo) | ijkabc >
                        hmatel = -h2b_ovvo(k,f,b,n)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik)(bc)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < jkn~acf~ | h2b(ovvo) | ijkabc >
                        hmatel = -h2b_ovvo(i,f,b,n)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)(bc)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ikn~acf~ | h2b(ovvo) | ijkabc >
                        hmatel = h2b_ovvo(j,f,b,n)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate l3 buffer arrays
                  deallocate(amps_buff,excits_buff)
                  
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits,&
                  !$omp l1a,l2a,&
                  !$omp H1A_ov,H2A_oovv,H2A_vovv,H2A_ooov,&
                  !$omp X2A_vovv,X2A_ooov,&
                  !$omp noa,nua,n3aaa),&
                  !$omp private(idet,a,b,c,d,i,j,k,l,m,n,e,f,&
                  !$omp res)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa
                      a = l3a_excits(1,idet); b = l3a_excits(2,idet); c = l3a_excits(3,idet);
                      i = l3a_excits(4,idet); j = l3a_excits(5,idet); k = l3a_excits(6,idet);
                      ! A(i/jk)A(a/bc) [l1a(ai) * h2a(jkbc) + h1a(ia) * l2a(bcjk)]
                      res =  l1a(a,i)*h2a_oovv(j,k,b,c) + h1a_ov(i,a)*l2a(b,c,j,k)& ! (1)
                            -l1a(a,j)*h2a_oovv(i,k,b,c) - h1a_ov(j,a)*l2a(b,c,i,k)& ! (ij)
                            -l1a(a,k)*h2a_oovv(j,i,b,c) - h1a_ov(k,a)*l2a(b,c,j,i)& ! (ik)
                            -l1a(b,i)*h2a_oovv(j,k,a,c) - h1a_ov(i,b)*l2a(a,c,j,k)& ! (ab)
                            +l1a(b,j)*h2a_oovv(i,k,a,c) + h1a_ov(j,b)*l2a(a,c,i,k)& ! (ij)(ab)
                            +l1a(b,k)*h2a_oovv(j,i,a,c) + h1a_ov(k,b)*l2a(a,c,j,i)& ! (ik)(ab)
                            -l1a(c,i)*h2a_oovv(j,k,b,a) - h1a_ov(i,c)*l2a(b,a,j,k)& ! (ac)
                            +l1a(c,j)*h2a_oovv(i,k,b,a) + h1a_ov(j,c)*l2a(b,a,i,k)& ! (ij)(ac)
                            +l1a(c,k)*h2a_oovv(j,i,b,a) + h1a_ov(k,c)*l2a(b,a,j,i)  ! (ik)(ac)
                      ! A(c/ab)A(j/ik) [-h2a(ikmc) * l2a(abmj) - h2a(mjab) * x2a(ikmc)]
                      do m = 1, noa
                         res = res&
                               - h2a_oovv(m,j,a,b)*x2a_ooov(i,k,m,c)& ! (1)
                               + h2a_oovv(m,i,a,b)*x2a_ooov(j,k,m,c)& ! (ij)
                               + h2a_oovv(m,k,a,b)*x2a_ooov(i,j,m,c)& ! (jk)
                               + h2a_oovv(m,j,c,b)*x2a_ooov(i,k,m,a)& ! (ac)
                               - h2a_oovv(m,i,c,b)*x2a_ooov(j,k,m,a)& ! (ij)(ac)
                               - h2a_oovv(m,k,c,b)*x2a_ooov(i,j,m,a)& ! (jk)(ac)
                               + h2a_oovv(m,j,a,c)*x2a_ooov(i,k,m,b)& ! (bc)
                               - h2a_oovv(m,i,a,c)*x2a_ooov(j,k,m,b)& ! (ij)(bc)
                               - h2a_oovv(m,k,a,c)*x2a_ooov(i,j,m,b)  ! (jk)(bc)
                         res = res&
                               - l2a(a,b,m,j)*h2a_ooov(i,k,m,c)& ! (1)
                               + l2a(a,b,m,i)*h2a_ooov(j,k,m,c)& ! (ij)
                               + l2a(a,b,m,k)*h2a_ooov(i,j,m,c)& ! (jk)
                               + l2a(c,b,m,j)*h2a_ooov(i,k,m,a)& ! (ac)
                               - l2a(c,b,m,i)*h2a_ooov(j,k,m,a)& ! (ij)(ac)
                               - l2a(c,b,m,k)*h2a_ooov(i,j,m,a)& ! (jk)(ac)
                               + l2a(a,c,m,j)*h2a_ooov(i,k,m,b)& ! (bc)
                               - l2a(a,c,m,i)*h2a_ooov(j,k,m,b)& ! (ij)(bc)
                               - l2a(a,c,m,k)*h2a_ooov(i,j,m,b)  ! (jk)(bc)
                      end do
                      ! A(b/ac)A(k/ij) [h2a_vovv(ekac)*l2a(ebij) + h2a(ijeb)*x2a(ekac)]
                      do e = 1, nua
                         res = res&
                               + h2a_oovv(i,j,e,b)*x2a_vovv(e,k,a,c)& ! (1)
                               - h2a_oovv(k,j,e,b)*x2a_vovv(e,i,a,c)& ! (ik)
                               - h2a_oovv(i,k,e,b)*x2a_vovv(e,j,a,c)& ! (jk)
                               - h2a_oovv(i,j,e,a)*x2a_vovv(e,k,b,c)& ! (ab)
                               + h2a_oovv(k,j,e,a)*x2a_vovv(e,i,b,c)& ! (ik)(ab)
                               + h2a_oovv(i,k,e,a)*x2a_vovv(e,j,b,c)& ! (jk)(ab)
                               - h2a_oovv(i,j,e,c)*x2a_vovv(e,k,a,b)& ! (bc)
                               + h2a_oovv(k,j,e,c)*x2a_vovv(e,i,a,b)& ! (ik)(bc)
                               + h2a_oovv(i,k,e,c)*x2a_vovv(e,j,a,b)  ! (jk)(bc)
                         res = res&
                               + l2a(e,b,i,j)*h2a_vovv(e,k,a,c)& ! (1)
                               - l2a(e,b,k,j)*h2a_vovv(e,i,a,c)& ! (ik)
                               - l2a(e,b,i,k)*h2a_vovv(e,j,a,c)& ! (jk)
                               - l2a(e,a,i,j)*h2a_vovv(e,k,b,c)& ! (ab)
                               + l2a(e,a,k,j)*h2a_vovv(e,i,b,c)& ! (ik)(ab)
                               + l2a(e,a,i,k)*h2a_vovv(e,j,b,c)& ! (jk)(ab)
                               - l2a(e,c,i,j)*h2a_vovv(e,k,a,b)& ! (bc)
                               + l2a(e,c,k,j)*h2a_vovv(e,i,a,b)& ! (ik)(bc)
                               + l2a(e,c,i,k)*h2a_vovv(e,j,a,b)  ! (jk)(bc)
                      end do
                      resid(idet) = resid(idet) + res
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!

              end subroutine build_LH_3A

              subroutine build_LH_3B(resid,&
                                    l1a, l1b, l2a, l2b,&
                                    l3a_amps, l3a_excits,&
                                    l3b_amps, l3b_excits,&
                                    l3c_amps, l3c_excits,&
                                    h1a_ov, h1a_oo, h1a_vv,&
                                    h1b_ov, h1b_oo, h1b_vv,&
                                    h2a_oooo, h2a_ooov, h2a_oovv,&
                                    h2a_voov, h2a_vovv, h2a_vvvv,&
                                    h2b_oooo, h2b_ooov, h2b_oovo, h2b_oovv,&
                                    h2b_voov, h2b_vovo, h2b_ovov, h2b_ovvo,&
                                    h2b_vovv, h2b_ovvv, h2b_vvvv,&
                                    h2c_voov,&
                                    x2a_ooov, x2a_vovv,&
                                    x2b_ooov, x2b_oovo, x2b_vovv, x2b_ovvv,&
                                    n3aaa, n3aab, n3abb,&
                                    noa, nua, nob, nub)
                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa, n3aab, n3abb
                  ! Input L arrays
                  real(kind=8), intent(in) :: l1a(nua,noa), l1b(nub,nob)
                  real(kind=8), intent(in) :: l2a(nua,nua,noa,noa), l2b(nua,nub,noa,nob)
                  integer, intent(in) :: l3a_excits(6,n3aaa), l3c_excits(6,n3abb)
                  real(kind=8), intent(in) :: l3a_amps(n3aaa), l3c_amps(n3abb)
                  ! Input H and X arrays
                  real(kind=8), intent(in) :: h1a_ov(noa,nua), h1b_ov(nob,nub)
                  real(kind=8), intent(in) :: h1a_oo(noa,noa), h1b_oo(nob,nob)
                  real(kind=8), intent(in) :: h1a_vv(nua,nua), h1b_vv(nub,nub)
                  real(kind=8), intent(in) :: h2a_oooo(noa,noa,noa,noa), h2b_oooo(noa,nob,noa,nob)
                  real(kind=8), intent(in) :: h2a_ooov(noa,noa,noa,nua), h2b_ooov(noa,nob,noa,nub), h2b_oovo(noa,nob,nua,nob)
                  real(kind=8), intent(in) :: h2a_oovv(noa,noa,nua,nua), h2b_oovv(noa,nob,nua,nub)
                  real(kind=8), intent(in) :: h2a_voov(nua,noa,noa,nua)
                  real(kind=8), intent(in) :: h2a_vovv(nua,noa,nua,nua), h2b_vovv(nua,nob,nua,nub), h2b_ovvv(noa,nub,nua,nub)
                  real(kind=8), intent(in) :: h2a_vvvv(nua,nua,nua,nua), h2b_vvvv(nua,nub,nua,nub)
                  real(kind=8), intent(in) :: h2b_voov(nua,nob,noa,nub), h2b_vovo(nua,nob,nua,nob), h2b_ovov(noa,nub,noa,nub), h2b_ovvo(noa,nub,nua,nob)
                  real(kind=8), intent(in) :: x2a_ooov(noa,noa,noa,nua), x2b_ooov(noa,nob,noa,nub), x2b_oovo(noa,nob,nua,nob)
                  real(kind=8), intent(in) :: x2a_vovv(nua,noa,nua,nua), x2b_vovv(nua,nob,nua,nub), x2b_ovvv(noa,nub,nua,nub)
                  real(kind=8), intent(in) :: h2c_voov(nub,nob,nob,nub)
                  ! Output and Inout variables
                  real(kind=8), intent(out) :: resid(n3aab)
                  integer, intent(inout) :: l3b_excits(6,n3aab)
                  !f2py intent(in,out) :: l3b_excits(6,0:n3aab-1)
                  real(kind=8), intent(inout) :: l3b_amps(n3aab)
                  !f2py intent(in,out) :: l3b_amps(0:n3aab-1)
                  ! Local variables
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:)
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)
                  real(kind=8) :: l_amp, hmatel, hmatel1, res
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc

                  resid = 0.0d0
                  
                  !!!! diagram 1: -A(ij) h1a(im)*l3b(abcmjk)
                  !!!! diagram 5: A(ij) 1/2 h2a(ijmn)*l3b(abcmnk)
                  !!! SB: (1,2,3,6) LOOP !!!
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2*nub*nob
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nua,nua,nub,noa))
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/1,nob/), nua, nua, nub, noa)
                  call sort4(l3b_excits, l3b_amps, loc_arr, idx_table, (/1,2,3,6/), nua, nua, nub, noa, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_oo,H2A_oooo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     idx = idx_table(a,b,c,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        l = l3b_excits(4,jdet); m = l3b_excits(5,jdet);
                        ! compute < lmk~abc~ | h2a(oooo) | ijk~abc~ >
                        hmatel = h2a_oooo(i,j,l,m)
                        ! compute < lmk~abc~ | h1a(oo) | ijk~abc~ > = -A(ij)A(lm) h1a_oo(i,l) * delta(j,m)
                        if (m==j) hmatel = hmatel - h1a_oo(i,l)
                        if (m==i) hmatel = hmatel + h1a_oo(j,l)
                        if (l==j) hmatel = hmatel + h1a_oo(i,m)
                        if (l==i) hmatel = hmatel - h1a_oo(j,m)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                 
                  !!!! diagram 2: A(ab) h1a(ea)*l3b(ebcmjk)
                  !!!! diagram 6: A(ab) 1/2 h2a(efab)*l3b(ebcmjk)
                  !!! SB: (4,5,6,3) LOOP !!!
                  ! allocate new sorting arrays
                  nloc = nub*noa*(noa-1)/2*nob
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(noa,noa,nob,nub))
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/1,nub/), noa, noa, nob, nub)
                  call sort4(l3b_excits, l3b_amps, loc_arr, idx_table, (/4,5,6,3/), noa, noa, nob, nub, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_vv,H2A_vvvv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     !idx = idx_table(c,i,j,k)
                     idx = idx_table(i,j,k,c)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3b_excits(1,jdet); e = l3b_excits(2,jdet);
                        ! compute < ijk~dec~ | h2a(vvvv) | ijk~abc~ >
                        hmatel = h2a_vvvv(d,e,a,b)
                        ! compute < ijk~dec~ | h1a(vv) | ijk~abc~ > = A(ab)A(de) h1a_vv(d,a)*delta(e,b)
                        if (b==e) hmatel = hmatel + h1a_vv(d,a)
                        if (a==e) hmatel = hmatel - h1a_vv(d,b)
                        if (b==d) hmatel = hmatel - h1a_vv(e,a)
                        if (a==d) hmatel = hmatel + h1a_vv(e,b)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                 
                  !!!! diagram 3: -h1b(km)*l3b(abcijm)
                  !!!! diagram 7: A(ij) h2b(jkmn)*l3b(abcimn)
                  !!! SB: (1,2,3,4) LOOP !!!
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2*nub*noa
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nua,nua,nub,noa))
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/1,noa-1/), nua, nua, nub, noa)
                  call sort4(l3b_excits, l3b_amps, loc_arr, idx_table, (/1,2,3,4/), nua, nua, nub, noa, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1B_oo,H2B_oooo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,i)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        m = l3b_excits(5,jdet); n = l3b_excits(6,jdet);
                        ! compute < imn~abc~ | h2b(oooo) | ijk~abc~ >
                        hmatel = h2b_oooo(j,k,m,n)
                        ! compute < imn~abc~ | h1b(oo) | ijk~abc~ >
                        if (m==j) hmatel = hmatel - h1b_oo(k,n)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           m = l3b_excits(5,jdet); n = l3b_excits(6,jdet);
                           ! compute < jmn~abc~ | h2b(oooo) | ijk~abc~ >
                           hmatel = -h2b_oooo(i,k,m,n)
                           ! compute < jmn~abc~ | h1b(oo) | ijk~abc~ >
                           if (m==i) hmatel = hmatel + h1b_oo(k,n)
                           resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,3,5) LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/2,noa/), nua, nua, nub, noa)
                  call sort4(l3b_excits, l3b_amps, loc_arr, idx_table, (/1,2,3,5/), nua, nua, nub, noa, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_oooo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,j)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        l = l3b_excits(4,jdet); n = l3b_excits(6,jdet);
                        ! compute < ljn~abc~ | h2b(oooo) | ijk~abc~ >
                        hmatel = h2b_oooo(i,k,l,n)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l = l3b_excits(4,jdet); n = l3b_excits(6,jdet);
                           ! compute < lin~abc~ | h2b(oooo) | ijk~abc~ >
                           hmatel = -h2b_oooo(j,k,l,n)
                           resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECITON !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                 
                  !!!! diagram 5: h1b(ec)*l3b(abeijm)
                  !!!! diagram 8: A(ab) h2b(efbc)*l3b(aefijk)
                  ! allocate new sorting arrays
                  nloc = nua*noa*(noa-1)/2*nob
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(noa,noa,nob,nua))
                  !!! SB: (4,5,6,1) LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/1,nua-1/), noa, noa, nob, nua)
                  call sort4(l3b_excits, l3b_amps, loc_arr, idx_table, (/4,5,6,1/), noa, noa, nob, nua, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1B_vv,H2B_vvvv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                      a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                      i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                      ! (1)
                      idx = idx_table(i,j,k,a)
                      do jdet = loc_arr(idx,1), loc_arr(idx,2)
                         e = l3b_excits(2,jdet); f = l3b_excits(3,jdet);
                         ! compute < ijk~aef~ | h2b(vvvv) | ijk~abc~ >
                         hmatel = h2b_vvvv(e,f,b,c)
                         if (b==e) hmatel = hmatel + h1b_vv(f,c)
                         resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                      end do
                      ! (ab)
                      idx = idx_table(i,j,k,b)
                      if (idx/=0) then ! protect against case where b = nua because a = 1, nua-1
                         do jdet = loc_arr(idx,1), loc_arr(idx,2)
                            e = l3b_excits(2,jdet); f = l3b_excits(3,jdet);
                            ! compute < ijk~bef~ | h2b(vvvv) | ijk~abc~ >
                            hmatel = -h2b_vvvv(e,f,a,c)
                            if (a==e) hmatel = hmatel - h1b_vv(f,c)
                            resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                         end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,6,2) LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/2,nua/), noa, noa, nob, nua)
                  call sort4(l3b_excits, l3b_amps, loc_arr, idx_table, (/4,5,6,2/), noa, noa, nob, nua, nloc, n3aab, resid)
                  !!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_vvvv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                      a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                      i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                      idx = idx_table(i,j,k,b)
                      do jdet = loc_arr(idx,1), loc_arr(idx,2)
                         d = l3b_excits(1,jdet); f = l3b_excits(3,jdet);
                         ! compute < ijk~dbf~ | h2b(vvvv) | ijk~abc~ >
                         hmatel = h2b_vvvv(d,f,a,c)
                         resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                      end do
                      idx = idx_table(i,j,k,a)
                      if (idx/=0) then ! protect against case where a = 1 because b = 2, nua
                         do jdet = loc_arr(idx,1), loc_arr(idx,2)
                            d = l3b_excits(1,jdet); f = l3b_excits(3,jdet);
                            ! compute < ijk~daf~ | h2b(vvvv) | ijk~abc~ >
                            hmatel = -h2b_vvvv(d,f,b,c)
                            resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                         end do
                      end if
                  end do ! end loop over idet
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                 
                  !!!! diagram 9: A(ij)A(ab) h2a(eima)*l3b(ebcmjk)
                  ! allocate new sorting arrays
                  nloc = nua*nub*noa*nob
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nua,nub,noa,nob))
                  !!! SB: (2,3,5,6) LOOP !!!
                  call get_index_table(idx_table, (/2,nua/), (/1,nub/), (/2,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(l3b_excits, l3b_amps, loc_arr, idx_table, (/2,3,5,6/), nua, nub, noa, nob, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,j,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3b_excits(1,jdet); l = l3b_excits(4,jdet);
                        ! compute < ijk~abc~ | h2a(voov) | ljk~dbc~ >
                        hmatel = h2a_voov(d,i,l,a)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then ! protect against case where a = 1 because b = 2, nua
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           d = l3b_excits(1,jdet); l = l3b_excits(4,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | ljk~dac~ >
                           hmatel = -h2a_voov(d,i,l,b)
                           resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then ! protect against case where i = 1 because j = 2, noa
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           d = l3b_excits(1,jdet); l = l3b_excits(4,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | lik~dbc~ >
                           hmatel = -h2a_voov(d,j,l,a)
                           resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then ! protect against case where a = 1 because b = 2, nua and i = 1 because j = 2, noa
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           d = l3b_excits(1,jdet); l = l3b_excits(4,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | lik~dac~ >
                           hmatel = h2a_voov(d,j,l,b)
                           resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4,6) LOOP !!!
                  call get_index_table(idx_table, (/2,nua/), (/1,nub/), (/1,noa-1/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(l3b_excits, l3b_amps, loc_arr, idx_table, (/2,3,4,6/), nua, nub, noa, nob, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,i,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3b_excits(1,jdet); l = l3b_excits(5,jdet);
                        ! compute < ijk~abc~ | h2a(voov) | ilk~dbc~ >
                        hmatel = h2a_voov(d,j,l,a)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then ! protect against where j = noa because i = 1, noa-1
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           d = l3b_excits(1,jdet); l = l3b_excits(5,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | jlk~dbc~ >
                           hmatel = -h2a_voov(d,i,l,a)
                           resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then ! protect against case where a = 1 because b = 2, nua
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           d = l3b_excits(1,jdet); l = l3b_excits(5,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | ilk~dac~ >
                           hmatel = -h2a_voov(d,j,l,b)
                           resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then ! protect against case where j = noa because i = 1, noa-1 and where a = 1 because b = 2, nua
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           d = l3b_excits(1,jdet); l = l3b_excits(5,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | jlk~dac~ >
                           hmatel = h2a_voov(d,i,l,b)
                           resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,4,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/1,nub/), (/1,noa-1/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(l3b_excits, l3b_amps, loc_arr, idx_table, (/1,3,4,6/), nua, nub, noa, nob, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3b_excits(2,jdet); l = l3b_excits(5,jdet);
                        ! compute < ijk~abc~ | h2a(voov) | ilk~adc~  >
                        hmatel = h2a_voov(d,j,l,b)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           d = l3b_excits(2,jdet); l = l3b_excits(5,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | jlk~adc~  >
                           hmatel = -h2a_voov(d,i,l,b)
                           resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           d = l3b_excits(2,jdet); l = l3b_excits(5,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | ilk~bdc~  >
                           hmatel = -h2a_voov(d,j,l,a)
                           resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           d = l3b_excits(2,jdet); l = l3b_excits(5,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | jlk~bdc~  >
                           hmatel = h2a_voov(d,i,l,a)
                           resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,5,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/1,nub/), (/2,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(l3b_excits, l3b_amps, loc_arr, idx_table, (/1,3,5,6/), nua, nub, noa, nob, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,j,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3b_excits(2,jdet); l = l3b_excits(4,jdet);
                        ! compute < ijk~abc~ | h2a(voov) | ljk~adc~  >
                        hmatel = h2a_voov(d,i,l,b)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           d = l3b_excits(2,jdet); l = l3b_excits(4,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | lik~adc~  >
                           hmatel = -h2a_voov(d,j,l,b)
                           resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           d = l3b_excits(2,jdet); l = l3b_excits(4,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | ljk~bdc~  >
                           hmatel = -h2a_voov(d,i,l,a)
                           resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           d = l3b_excits(2,jdet); l = l3b_excits(4,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | lik~abc~  >
                           hmatel = h2a_voov(d,j,l,a)
                           resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                 
                  !!!! diagram 10: h2c(ekmc)*l3b(abeijm)
                  ! allocate sorting arrays
                  nloc = nua*(nua-1)/2*noa*(noa-1)/2
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nua,nua,noa,noa))
                  !!! SB: (1,2,4,5) LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(l3b_excits, l3b_amps, loc_arr, idx_table, (/1,2,4,5/), nua, nua, noa, noa, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,&
                  !$omp n3aab),&
                  !$omp private(hmatel,a,b,c,i,j,k,f,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                      a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                      i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                      idx = idx_table(a,b,i,j)
                      do jdet = loc_arr(idx,1), loc_arr(idx,2)
                         f = l3b_excits(3,jdet); n = l3b_excits(6,jdet);
                         ! compute < ijn~abf~ | h2c(voov) | ijk~abc~ >
                         hmatel = h2c_voov(f,k,n,c)
                         resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                      end do
                  end do ! end loop over idet
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 11: -A(ij) h2b(iemc)*l3b(abemjk)
                  ! allocate sorting arrays
                  nloc = nua*(nua-1)/2*noa*nob
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nua,nua,noa,nob))
                  !!! SB: (1,2,4,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,noa-1/), (/1,nob/), nua, nua, noa, nob)
                  call sort4(l3b_excits, l3b_amps, loc_arr, idx_table, (/1,2,4,6/), nua, nua, noa, nob, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,i,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3b_excits(3,jdet); m = l3b_excits(5,jdet);
                        ! compute < imk~abf~ | h2b(ovov) | ijk~abc~ >
                        hmatel = -h2b_ovov(j,f,m,c)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           f = l3b_excits(3,jdet); m = l3b_excits(5,jdet);
                           ! compute < jmk~abf~ | h2b(ovov) | ijk~abc~ >
                           hmatel = h2b_ovov(i,f,m,c)
                           resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,5,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/2,noa/), (/1,nob/), nua, nua, noa, nob)
                  call sort4(l3b_excits, l3b_amps, loc_arr, idx_table, (/1,2,5,6/), nua, nua, noa, nob, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,j,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3b_excits(3,jdet); l = l3b_excits(4,jdet);
                        ! compute < ljk~abf~ | h2b(ovov) | ijk~abc~ >
                        hmatel = -h2b_ovov(i,f,l,c)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           f = l3b_excits(3,jdet); l = l3b_excits(4,jdet);
                           ! compute < lik~abf~ | h2b(ovov) | ijk~abc~ >
                           hmatel = h2b_ovov(j,f,l,c)
                           resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 12: -A(ab) h2b(ekam)*l3b(ebcijm)
                  ! allocate sorting arrays
                  nloc = nua*nub*noa*(noa-1)/2
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(noa,noa,nua,nub))
                  !!! SB: (4,5,2,3) LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/2,nua/), (/1,nub/), noa, noa, nua, nub)
                  call sort4(l3b_excits, l3b_amps, loc_arr, idx_table, (/4,5,2,3/), noa, noa, nua, nub, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_vovo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,b,c)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3b_excits(1,jdet); n = l3b_excits(6,jdet);
                        ! compute < ijn~dbc~ | h2b(vovo) | ijk~abc~ >
                        hmatel = -h2b_vovo(d,k,a,n)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           d = l3b_excits(1,jdet); n = l3b_excits(6,jdet);
                           ! compute < ijn~dac~ | h2b(vovo) | ijk~abc~ >
                           hmatel = h2b_vovo(d,k,b,n)
                           resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,1,3) LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nua-1/), (/1,nub/), noa, noa, nua, nub)
                  call sort4(l3b_excits, l3b_amps, loc_arr, idx_table, (/4,5,1,3/), noa, noa, nua, nub, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_vovo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,a,c)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3b_excits(2,jdet); n = l3b_excits(6,jdet);
                        ! compute < ijn~aec~ | h2b(vovo) | ijk~abc~ >
                        hmatel = -h2b_vovo(e,k,b,n)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           e = l3b_excits(2,jdet); n = l3b_excits(6,jdet);
                           ! compute < ijn~bec~ | h2b(vovo) | ijk~abc~ >
                           hmatel = h2b_vovo(e,k,a,n)
                           resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                 
                  !!!! diagram 13: h2b(ekmc)*l3a(abeijm) !!!!
                  ! allocate and initialize the copy of l3a
                  allocate(amps_buff(n3aaa))
                  allocate(excits_buff(6,n3aaa))
                  amps_buff(:) = l3a_amps(:)
                  excits_buff(:,:) = l3a_excits(:,:)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = (nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nua,nua,noa,noa))
                  !!! SB: (1,2,4,5) LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,5/), nua, nua, noa, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     idx = idx_table(a,b,i,j)
                     if (idx==0) cycle 
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijnabf | h2b(voov) | ijk~abc~ >
                        hmatel = h2b_voov(f,k,n,c)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,4,5) LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,5/), nua, nua, noa, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                        ! compute < ijnaeb | h2b(voov) | ijk~abc~ >
                        hmatel = -h2b_voov(e,k,n,c)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4,5) LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,5/), nua, nua, noa, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = excits_buff(1,jdet); n = excits_buff(6,jdet);
                        ! compute < ijndab | h2b(voov) | ijk~abc~ >
                        hmatel = h2b_voov(d,k,n,c)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,6/), nua, nua, noa, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                        ! compute < imjabf | h2b(voov) | ijk~abc~ >
                        hmatel = -h2b_voov(f,k,m,c)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,4,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,6/), nua, nua, noa, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                        ! compute < imjaeb | h2b(voov) | ijk~abc~ >
                        hmatel = h2b_voov(e,k,m,c)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4,6) LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,6/), nua, nua, noa, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                        ! compute < imjdab | h2b(voov) | ijk~abc~ >
                        hmatel = -h2b_voov(d,k,m,c)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,5,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,5,6/), nua, nua, noa, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = excits_buff(3,jdet); l = excits_buff(4,jdet);
                        ! compute < lijabf | h2b(voov) | ijk~abc~ >
                        hmatel = h2b_voov(f,k,l,c)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,5,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,5,6/), nua, nua, noa, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                        ! compute < lijaeb | h2b(voov) | ijk~abc~ >
                        hmatel = -h2b_voov(e,k,l,c)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,5,6) LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,5,6/), nua, nua, noa, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < lijdab | h2b(voov) | ijk~abc~ >
                        hmatel = h2b_voov(d,k,l,c)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate l3 buffer arrays
                  deallocate(amps_buff,excits_buff) 
                 
                  !!!! diagram 14: A(ab)A(ij) h2b(jebm)*l3c(aecimk)
                  ! allocate and initialize the copy of l3c
                  allocate(amps_buff(n3abb))
                  allocate(excits_buff(6,n3abb))
                  amps_buff(:) = l3c_amps(:)
                  excits_buff(:,:) = l3c_excits(:,:)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = nua*nub*noa*nob
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nua,nub,noa,nob))
                  !!! SB: (1,3,4,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/2,nub/), (/1,noa/), (/2,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,6/), nua, nub, noa, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                           ! compute < im~k~ae~c~ | h2b(ovvo) | ijk~abc~ >
                           hmatel = h2b_ovvo(j,e,b,m)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                           ! compute < im~k~be~c~ | h2b(ovvo) | ijk~abc~ >
                           hmatel = -h2b_ovvo(j,e,a,m)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                           ! compute < jm~k~ae~c~ | h2b(ovvo) | ijk~abc~ >
                           hmatel = -h2b_ovvo(i,e,b,m)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                           ! compute < jm~k~be~c~ | h2b(ovvo) | ijk~abc~ >
                           hmatel = h2b_ovvo(i,e,a,m)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/1,nub-1/), (/1,noa/), (/2,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,6/), nua, nub, noa, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                           ! compute < im~k~ac~f~ | h2b(ovvo) | ijk~abc~ >
                           hmatel = -h2b_ovvo(j,f,b,m)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                           ! compute < im~k~bc~f~ | h2b(ovvo) | ijk~abc~ >
                           hmatel = h2b_ovvo(j,f,a,m)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                           ! compute < jm~k~ac~f~ | h2b(ovvo) | ijk~abc~ >
                           hmatel = h2b_ovvo(i,f,b,m)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                           ! compute < jm~k~bc~f~ | h2b(ovvo) | ijk~abc~ >
                           hmatel = -h2b_ovvo(i,f,a,m)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,4,5) LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/2,nub/), (/1,noa/), (/1,nob-1/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,5/), nua, nub, noa, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                           ! compute < ik~n~ae~c~ | h2b(ovvo) | ijk~abc~ >
                           hmatel = -h2b_ovvo(j,e,b,n)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                           ! compute < ik~n~be~c~ | h2b(ovvo) | ijk~abc~ >
                           hmatel = h2b_ovvo(j,e,a,n)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                           ! compute < jk~n~ae~c~ | h2b(ovvo) | ijk~abc~ >
                           hmatel = h2b_ovvo(i,e,b,n)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                           ! compute < jk~n~be~c~ | h2b(ovvo) | ijk~abc~ >
                           hmatel = -h2b_ovvo(i,e,a,n)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4,5) LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/1,nub-1/), (/1,noa/), (/1,nob-1/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,5/), nua, nub, noa, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                           ! compute < ik~n~ac~f~ | h2b(ovvo) | ijk~abc~ >
                           hmatel = h2b_ovvo(j,f,b,n)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                           ! compute < ik~n~bc~f~ | h2b(ovvo) | ijk~abc~ >
                           hmatel = -h2b_ovvo(j,f,a,n)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                           ! compute < jk~n~ac~f~ | h2b(ovvo) | ijk~abc~ >
                           hmatel = -h2b_ovvo(i,f,b,n)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                           ! compute < jk~n~bc~f~ | h2b(ovvo) | ijk~abc~ >
                           hmatel = h2b_ovvo(i,f,a,n)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate l3 buffer arrays
                  deallocate(amps_buff,excits_buff)
                  
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l1a,l1b,l2a,l2b,&
                  !$omp H1A_ov,H1B_ov,&
                  !$omp H2A_oovv,H2B_oovv,&
                  !$omp H2A_ooov,H2A_vovv,
                  !$omp H2B_ooov,H2B_oovo,H2B_vovv,H2B_ovvv,&
                  !$omp X2A_ooov,X2A_vovv,
                  !$omp X2B_ooov,X2B_oovo,X2B_vovv,X2B_ovvv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(idet,a,b,c,d,i,j,k,l,m,n,e,f,&
                  !$omp res)
                  !$omp do schedule(static)
                  do idet = 1, n3aab
                      a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                      i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);

                      ! A(ab)A(ij) [l1a(ai)*h2b(jkbc) + h1a(ia)*l2a(bcjk)
                      res =  l1a(a,i)*h2b_oovv(j,k,b,c) + h1a_ov(i,a)*l2b(b,c,j,k)& ! (1)
                            -l1a(a,j)*h2b_oovv(i,k,b,c) - h1a_ov(j,a)*l2b(b,c,i,k)& ! (ij)
                            -l1a(b,i)*h2b_oovv(j,k,a,c) - h1a_ov(i,b)*l2b(a,c,j,k)& ! (ab)
                            +l1a(b,j)*h2b_oovv(i,k,a,c) + h1a_ov(j,b)*l2b(a,c,i,k)  ! (ij)(ab)
                      ! l1b(ck)*h2a(ijab) + h1b(kc)*l2a(abij)
                      res = res + l1b(c,k)*h2a_oovv(i,j,a,b) + h1b_ov(k,c)*l2a(a,b,i,j)
                      ! -A(ij) h2b(jkmc)*l2a(abim)
                      ! -A(ab) h2a(jima)*l2b(bcmk)
                      ! -A(ij) x2b(jkmc)*h2a(imab)
                      ! -A(ab) x2a(jima)*h2b(mkbc)
                      do m = 1, noa
                         res = res&
                               -h2b_ooov(j,k,m,c)*l2a(a,b,i,m) + h2b_ooov(i,k,m,c)*l2a(a,b,j,m)&
                               -h2a_ooov(j,i,m,a)*l2b(b,c,m,k) + h2a_ooov(j,i,m,b)*l2b(a,c,m,k)&
                               -x2b_ooov(j,k,m,c)*h2a_oovv(i,m,a,b) + x2b_ooov(i,k,m,c)*h2a_oovv(j,m,a,b)&
                               -x2a_ooov(j,i,m,a)*h2b_oovv(m,k,b,c) + x2a_ooov(j,i,m,b)*h2b_oovv(m,k,a,c)
                      end do
                      ! -A(ij)A(ab) h2b(ikam)*l2b(bcjm)
                      ! -A(ij)A(ab) x2b(ikam)*h2b(jmbc)
                      do m = 1, nob
                         res = res&
                               -h2b_oovo(i,k,a,m)*l2b(b,c,j,m) - x2b_oovo(i,k,a,m)*h2b_oovv(j,m,b,c)& ! (1)
                               +h2b_oovo(j,k,a,m)*l2b(b,c,i,m) + x2b_oovo(j,k,a,m)*h2b_oovv(i,m,b,c)& ! (ij)
                               +h2b_oovo(i,k,b,m)*l2b(a,c,j,m) + x2b_oovo(i,k,b,m)*h2b_oovv(j,m,a,c)& ! (ab)
                               -h2b_oovo(j,k,b,m)*l2b(a,c,i,m) - x2b_oovo(j,k,b,m)*h2b_oovv(i,m,a,c)  ! (ij)(ab)
                      end do
                      ! A(ab) h2b(ekbc)*l2a(aeij)
                      ! A(ij) h2a(eiba)*l2b(ecjk)
                      ! A(ab) x2b(ekbc)*h2a(ijae)
                      ! A(ij) x2a(eiba)*h2b(jkec)
                      do e = 1, nua
                         res = res&
                               +h2b_vovv(e,k,b,c)*l2a(a,e,i,j) - h2b_vovv(e,k,a,c)*l2a(b,e,i,j)&
                               +h2a_vovv(e,i,b,a)*l2b(e,c,j,k) - h2a_vovv(e,j,b,a)*l2b(e,c,i,k)&
                               +x2b_vovv(e,k,b,c)*h2a_oovv(i,j,a,e) - x2b_vovv(e,k,a,c)*h2a_oovv(i,j,b,e)&
                               +x2a_vovv(e,i,b,a)*h2b_oovv(j,k,e,c) - x2a_vovv(e,j,b,a)*h2b_oovv(i,k,e,c)
                      end do
                      ! A(ij)A(ab) h2b(ieac)*l2b(bejk)
                      ! A(ij)A(ab) x2b(ieac)*h2b(jkbe)
                      do e = 1, nub
                         res = res&
                               +h2b_ovvv(i,e,a,c)*l2b(b,e,j,k) + x2b_ovvv(i,e,a,c)*h2b_oovv(j,k,b,e)& ! (1)
                               -h2b_ovvv(j,e,a,c)*l2b(b,e,i,k) - x2b_ovvv(j,e,a,c)*h2b_oovv(i,k,b,e)& ! (ij)
                               -h2b_ovvv(i,e,b,c)*l2b(a,e,j,k) - x2b_ovvv(i,e,b,c)*h2b_oovv(j,k,a,e)& ! (ab)
                               +h2b_ovvv(j,e,b,c)*l2b(a,e,i,k) + x2b_ovvv(j,e,b,c)*h2b_oovv(i,k,a,e)  ! (ij)(ab)
                      end do
                      resid(idet) = resid(idet) + res
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!                 
                 
        end subroutine build_LH_3B

        subroutine build_LH_3C(resid,&
                              l1a, l1b, l2b, l2c,&
                              l3b_amps, l3b_excits,&
                              l3c_amps, l3c_excits,&
                              l3d_amps, l3d_excits,&
                              h1a_ov, h1a_oo, h1a_vv,&
                              h1b_ov, h1b_oo, h1b_vv,&
                              h2a_voov,&
                              h2b_oooo, h2b_ooov, h2b_oovo, h2b_oovv,&
                              h2b_voov, h2b_vovo, h2b_ovov, h2b_ovvo,&
                              h2b_vovv, h2b_ovvv, h2b_vvvv,&
                              h2c_oooo, h2c_ooov, h2c_oovv,&
                              h2c_voov, h2c_vovv, h2c_vvvv,&
                              x2b_ooov, x2b_oovo, x2b_vovv, x2b_ovvv,&
                              x2c_ooov, x2c_vovv,&
                              n3aab, n3abb, n3bbb,&
                              noa, nua, nob, nub)
                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab, n3abb, n3bbb
                  ! Input L arrays
                  real(kind=8), intent(in) :: l1a(nua,noa), l1b(nub,nob)
                  real(kind=8), intent(in) :: l2c(nub,nub,nob,nob), l2b(nua,nub,noa,nob)
                  integer, intent(in) :: l3d_excits(6,n3bbb), l3b_excits(6,n3aab)
                  real(kind=8), intent(in) :: l3d_amps(n3bbb), l3b_amps(n3aab)
                  ! Input H and X arrays
                  real(kind=8), intent(in) :: h1a_ov(noa,nua), h1b_ov(nob,nub)
                  real(kind=8), intent(in) :: h1a_oo(noa,noa), h1b_oo(nob,nob)
                  real(kind=8), intent(in) :: h1a_vv(nua,nua), h1b_vv(nub,nub)
                  real(kind=8), intent(in) :: h2c_oooo(nob,nob,nob,nob), h2b_oooo(noa,nob,noa,nob)
                  real(kind=8), intent(in) :: h2c_ooov(nob,nob,nob,nub), h2b_ooov(noa,nob,noa,nub), h2b_oovo(noa,nob,nua,nob)
                  real(kind=8), intent(in) :: h2c_oovv(nob,nob,nub,nub), h2b_oovv(noa,nob,nua,nub)
                  real(kind=8), intent(in) :: h2c_voov(nub,nob,nob,nub)
                  real(kind=8), intent(in) :: h2c_vovv(nub,nob,nub,nub), h2b_vovv(nua,nob,nua,nub), h2b_ovvv(noa,nub,nua,nub)
                  real(kind=8), intent(in) :: h2c_vvvv(nub,nub,nub,nub), h2b_vvvv(nua,nub,nua,nub)
                  real(kind=8), intent(in) :: h2b_voov(nua,nob,noa,nub), h2b_vovo(nua,nob,nua,nob), h2b_ovov(noa,nub,noa,nub), h2b_ovvo(noa,nub,nua,nob)
                  real(kind=8), intent(in) :: x2c_ooov(nob,nob,nob,nub), x2b_ooov(noa,nob,noa,nub), x2b_oovo(noa,nob,nua,nob)
                  real(kind=8), intent(in) :: x2c_vovv(nub,nob,nub,nub), x2b_vovv(nua,nob,nua,nub), x2b_ovvv(noa,nub,nua,nub)
                  real(kind=8), intent(in) :: h2a_voov(nua,noa,noa,nua)
                  ! Output and Inout variables
                  real(kind=8), intent(out) :: resid(n3abb)
                  integer, intent(inout) :: l3c_excits(6,n3abb)
                  !f2py intent(in,out) :: l3c_excits(6,0:n3abb-1)
                  real(kind=8), intent(inout) :: l3c_amps(n3abb)
                  !f2py intent(in,out) :: l3c_amps(0:n3abb-1)
                  ! Local variables
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:)
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)
                  real(kind=8) :: l_amp, hmatel, hmatel1, res
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc

                  resid = 0.0d0
                  
                  !!!! diagram 1: -A(jk) h1b(km)*l3c(abcijm)
                  !!!! diagram 5: A(jk) 1/2 h2c(jkmn)*l3c(abcimn)
                  !!! SB: (2,3,1,4) LOOP !!!
                  ! allocate new sorting arrays
                  nloc = nub*(nub-1)/2*nua*noa
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nub,nub,nua,noa))
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nua/), (/1,noa/), nub, nub, nua, nob)
                  call sort4(l3c_excits, l3c_amps, loc_arr, idx_table, (/2,3,1,4/), nub, nub, nua, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1B_oo,H2C_oooo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                     a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                     i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                     idx = idx_table(b,c,a,i)
                     ! (1)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        m = l3c_excits(5,jdet); n = l3c_excits(6,jdet);
                        ! compute < im~n~ab~c~ | h2c(oooo) | ij~k~ab~c~ >
                        hmatel = h2c_oooo(j,k,m,n)
                        ! compute < im~n~ab~c~ | h1b(oo) | ij~k~ab~c~ > = -A(jk)A(mn) h1b_oo(j,m) * delta(k,n)
                        if (n==k) hmatel = hmatel - h1b_oo(j,m) ! (1)
                        if (n==j) hmatel = hmatel + h1b_oo(k,m) ! (jk)
                        if (m==k) hmatel = hmatel + h1b_oo(j,n) ! (mn)
                        if (m==j) hmatel = hmatel - h1b_oo(k,n) ! (jk)(mn)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
           
                  !!!! diagram 2: A(bc) h1b(ec)*l3c(abeijk)
                  !!!! diagram 6: A(bc) 1/2 h2c(efbc)*l3c(aefijk)
                  !!! SB: (5,6,4,1) LOOP !!!
                  ! allocate new sorting arrays
                  nloc = nua*nob*(nob-1)/2*noa
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nob,nob,noa,nua))
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,noa/), (/1,nua/), nob, nob, noa, nua)
                  call sort4(l3c_excits, l3c_amps, loc_arr, idx_table, (/5,6,4,1/), nob, nob, noa, nua, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1B_vv,H2C_vvvv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                     a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                     i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                     idx = idx_table(j,k,i,a)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3c_excits(2,jdet); f = l3c_excits(3,jdet);
                        ! compute < ij~k~ab~c~ | h2c(vvvv) | ij~k~ae~f~ >
                        hmatel = h2c_vvvv(e,f,b,c)
                        ! compute < ij~k~ab~c~ | h2c(vvvv) | ij~k~ae~f~ > = A(bc)A(ef) h1b_vv(b,e) * delta(c,f)
                        if (c==f) hmatel = hmatel + h1b_vv(e,b) ! (1)
                        if (b==f) hmatel = hmatel - h1b_vv(e,c) ! (bc)
                        if (c==e) hmatel = hmatel - h1b_vv(f,b) ! (ef)
                        if (b==e) hmatel = hmatel + h1b_vv(f,c) ! (bc)(ef)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)    
                  
                  !!!! diagram 3: -h1a(im)*l3c(abcmjk)
                  !!!! diagram 7: A(jk) h2b(ijmn)*l3c(abcmnk)
                  !!! SB: (2,3,1,6) LOOP !!!
                  ! allocate new sorting arrays
                  nloc = nub*(nub-1)/2*nua*nob
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nub,nub,nua,nob))
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nua/), (/2,nob/), nub, nub, nua, nob)
                  call sort4(l3c_excits, l3c_amps, loc_arr, idx_table, (/2,3,1,6/), nub, nub, nua, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_oo,H2B_oooo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                     a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                     i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,a,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        l = l3c_excits(4,jdet); m = l3c_excits(5,jdet);
                        ! compute < ij~k~ab~c~ | h2b(oooo) | lm~k~ab~c~ >
                        hmatel = h2b_oooo(i,j,l,m)
                        ! compute < ij~k~ab~c~ | h1a(oo) | lm~k~ab~c~ >
                        if (m==j) hmatel = hmatel - h1a_oo(i,l)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table(b,c,a,j)
                     if (idx/=0) then
                         do jdet = loc_arr(idx,1), loc_arr(idx,2)
                            l = l3c_excits(4,jdet); m = l3c_excits(5,jdet);
                            ! compute < ij~k~ab~c~ | h2b(oooo) | lm~j~ab~c~ >
                            hmatel = -h2b_oooo(i,k,l,m)
                            ! compute < ij~k~ab~c~ | h1a(oo) | lm~j~ab~c~ >
                            if (m==k) hmatel = hmatel + h1a_oo(i,l)
                            resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                         end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,1,5) LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nua/), (/1,nob-1/), nub, nub, nua, nob)
                  call sort4(l3c_excits, l3c_amps, loc_arr, idx_table, (/2,3,1,5/), nub, nub, nua, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_oooo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                     a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                     i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,a,j)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        l = l3c_excits(4,jdet); n = l3c_excits(6,jdet);
                        ! compute < ij~k~ab~c~ | h2b(oooo) | lj~n~ab~c~ >
                        hmatel = h2b_oooo(i,k,l,n)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table(b,c,a,k)
                     if (idx/=0) then
                         do jdet = loc_arr(idx,1), loc_arr(idx,2)
                            l = l3c_excits(4,jdet); n = l3c_excits(6,jdet);
                            ! compute < ij~k~ab~c~ | h2b(oooo) | lk~n~ab~c~ >
                            hmatel = -h2b_oooo(i,j,l,n)
                            resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                         end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECITON !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  
                  !!!! diagram 5: h1a(ea)*l3c(ebcijk)
                  !!!! diagram 8: A(bc) h2b(efab)*l3c(efcijk)
                  ! allocate new sorting arrays
                  nloc = nub*nob*(nob-1)/2*noa
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nob,nob,noa,nub))
                  !!! SB: (5,6,4,2) LOOP !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,noa/), (/1,nub-1/), nob, nob, noa, nub)
                  call sort4(l3c_excits, l3c_amps, loc_arr, idx_table, (/5,6,4,2/), nob, nob, noa, nub, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_vv,H2B_vvvv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                      a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                      i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(j,k,i,b)
                      do jdet = loc_arr(idx,1), loc_arr(idx,2)
                         d = l3c_excits(1,jdet); f = l3c_excits(3,jdet);
                         ! compute < ij~k~ab~c~ | h2b(vvvv) | ij~k~db~f~ >
                         hmatel = h2b_vvvv(d,f,a,c)
                         if (c==f) hmatel = hmatel + h1a_vv(d,a)
                         resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                      end do
                      ! (bc)
                      idx = idx_table(j,k,i,c)
                      if (idx/=0) then ! protect against case where b = nua because a = 1, nua-1
                         do jdet = loc_arr(idx,1), loc_arr(idx,2)
                            d = l3c_excits(1,jdet); f = l3c_excits(3,jdet);
                            ! compute < ij~k~ab~c~ | h2b(vvvv) | ij~k~dc~f~ >
                            hmatel = -h2b_vvvv(d,f,a,b)
                            if (b==f) hmatel = hmatel - h1a_vv(d,a)
                            resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                         end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (5,6,4,3) LOOP !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,noa/), (/2,nub/), nob, nob, noa, nub)
                  call sort4(l3c_excits, l3c_amps, loc_arr, idx_table, (/5,6,4,3/), nob, nob, noa, nub, nloc, n3abb, resid)
                  !!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_vvvv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                      a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                      i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                      idx = idx_table(j,k,i,c)
                      do jdet = loc_arr(idx,1), loc_arr(idx,2)
                         d = l3c_excits(1,jdet); e = l3c_excits(2,jdet);
                         ! compute < ij~k~ab~c~ | h2b(vvvv) | ij~k~de~c~ >
                         hmatel = h2b_vvvv(d,e,a,b)
                         resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                      end do
                      ! (bc)
                      idx = idx_table(j,k,i,b)
                      if (idx/=0) then ! protect against case where a = 1 because b = 2, nua
                         do jdet = loc_arr(idx,1), loc_arr(idx,2)
                            d = l3c_excits(1,jdet); e = l3c_excits(2,jdet);
                            ! compute < ij~k~ab~c~ | h2b(vvvv) | ij~k~de~b~ >
                            hmatel = -h2b_vvvv(d,e,a,c)
                            resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                         end do
                      end if
                  end do ! end loop over idet
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  
                  !!!! diagram 9: A(jk)A(bc) h2c(ekmc)*l3c(abeijm)
                  ! allocate new sorting arrays
                  nloc = nub*nua*nob*noa
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nua,nub,noa,nob))
                  !!! SB: (1,2,4,5) LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/1,nub-1/), (/1,noa/), (/1,nob-1/), nua, nub, noa, nob)
                  call sort4(l3c_excits, l3c_amps, loc_arr, idx_table, (/1,2,4,5/), nua, nub, noa, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                     a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                     i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,i,j)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3c_excits(3,jdet); n = l3c_excits(6,jdet);
                        ! compute < ij~k~ab~c~ | h2a(voov) | ij~n~ab~f~ >
                        hmatel = h2c_voov(f,k,n,c)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                         do jdet = loc_arr(idx,1), loc_arr(idx,2)
                            f = l3c_excits(3,jdet); n = l3c_excits(6,jdet);
                            ! compute < ij~k~ab~c~ | h2a(voov) | ik~n~ab~f~ >
                            hmatel = -h2c_voov(f,j,n,c)
                            resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                         end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                         do jdet = loc_arr(idx,1), loc_arr(idx,2)
                            f = l3c_excits(3,jdet); n = l3c_excits(6,jdet);
                            ! compute < ij~k~ab~c~ | h2a(voov) | ij~n~ac~f~ >
                            hmatel = -h2c_voov(f,k,n,b)
                            resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                         end do
                     end if
                     ! (jk)(bc)
                      idx = idx_table(a,c,i,k)
                      if (idx/=0) then
                          do jdet = loc_arr(idx,1), loc_arr(idx,2)
                             f = l3c_excits(3,jdet); n = l3c_excits(6,jdet);
                             ! compute < ij~k~ab~c~ | h2a(voov) | ik~n~ac~f~ >
                             hmatel = h2c_voov(f,j,n,b)
                             resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,4,5) LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/2,nub/), (/1,noa/), (/1,nob-1/), nua, nub, noa, nob)
                  call sort4(l3c_excits, l3c_amps, loc_arr, idx_table, (/1,3,4,5/), nua, nub, noa, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                      a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                      i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(a,c,i,j)
                      do jdet = loc_arr(idx,1), loc_arr(idx,2)
                          e = l3c_excits(2,jdet); n = l3c_excits(6,jdet);
                          ! compute < ij~k~ab~c~ | h2c(voov) | ij~n~ae~c~ >
                          hmatel = h2c_voov(e,k,n,b)
                          resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                      end do
                      ! (jk)
                      idx = idx_table(a,c,i,k)
                      if (idx/=0) then
                          do jdet = loc_arr(idx,1), loc_arr(idx,2)
                              e = l3c_excits(2,jdet); n = l3c_excits(6,jdet);
                              ! compute < ij~k~ab~c~ | h2c(voov) | ik~n~ae~c~ >
                              hmatel = -h2c_voov(e,j,n,b)
                              resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                          end do
                      end if
                      ! (bc)
                      idx = idx_table(a,b,i,j)
                      if (idx/=0) then
                          do jdet = loc_arr(idx,1), loc_arr(idx,2)
                              e = l3c_excits(2,jdet); n = l3c_excits(6,jdet);
                              ! compute < ij~k~ab~c~ | h2c(voov) | ij~n~ae~b~ >
                              hmatel = -h2c_voov(e,k,n,c)
                              resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                          end do
                      end if
                      ! (jk)(bc)
                      idx = idx_table(a,b,i,k)
                      if (idx/=0) then
                          do jdet = loc_arr(idx,1), loc_arr(idx,2)
                              e = l3c_excits(2,jdet); n = l3c_excits(6,jdet);
                              ! compute < ij~k~ab~c~ | h2c(voov) | ik~n~ae~b~ >
                              hmatel = h2c_voov(e,j,n,c)
                              resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/1,nub-1/), (/1,noa/), (/2,nob/), nua, nub, noa, nob)
                  call sort4(l3c_excits, l3c_amps, loc_arr, idx_table, (/1,2,4,6/), nua, nub, noa, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                      a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                      i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(a,b,i,k)
                      do jdet = loc_arr(idx,1), loc_arr(idx,2)
                          f = l3c_excits(3,jdet); m = l3c_excits(5,jdet);
                          ! compute < ij~k~ab~c~ | h2c(voov) | im~k~ab~f~ >
                          hmatel = h2c_voov(f,j,m,c)
                          resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                      end do
                      ! (jk)
                      idx = idx_table(a,b,i,j)
                      if (idx/=0) then
                          do jdet = loc_arr(idx,1), loc_arr(idx,2)
                              f = l3c_excits(3,jdet); m = l3c_excits(5,jdet);
                              ! compute < ij~k~ab~c~ | h2c(voov) | im~j~ab~f~ >
                              hmatel = -h2c_voov(f,k,m,c)
                              resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                          end do
                      end if
                      ! (bc)
                      idx = idx_table(a,c,i,k)
                      if (idx/=0) then
                          do jdet = loc_arr(idx,1), loc_arr(idx,2)
                              f = l3c_excits(3,jdet); m = l3c_excits(5,jdet);
                              ! compute < ij~k~ab~c~ | h2c(voov) | im~k~ac~f~ >
                              hmatel = -h2c_voov(f,j,m,b)
                              resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                          end do
                      end if
                      ! (jk)(bc)
                      idx = idx_table(a,c,i,j)
                      if (idx/=0) then
                          do jdet = loc_arr(idx,1), loc_arr(idx,2)
                              f = l3c_excits(3,jdet); m = l3c_excits(5,jdet);
                              ! compute < ij~k~ab~c~ | h2c(voov) | im~j~ac~f~ >
                              hmatel = h2c_voov(f,k,m,b)
                              resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,4,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/2,nub/), (/1,noa/), (/2,nob/), nua, nub, noa, nob)
                  call sort4(l3c_excits, l3c_amps, loc_arr, idx_table, (/1,3,4,6/), nua, nub, noa, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                      a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                      i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(a,c,i,k)
                      do jdet = loc_arr(idx,1), loc_arr(idx,2)
                          e = l3c_excits(2,jdet); m = l3c_excits(5,jdet);
                          ! compute < ij~k~ab~c~ | h2c(voov) | im~k~ae~c~ >
                          hmatel = h2c_voov(e,j,m,b)
                          resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                      end do
                      ! (jk)
                      idx = idx_table(a,c,i,j)
                      if (idx/=0) then
                          do jdet = loc_arr(idx,1), loc_arr(idx,2)
                              e = l3c_excits(2,jdet); m = l3c_excits(5,jdet);
                              ! compute < ij~k~ab~c~ | h2c(voov) | im~j~ae~c~ >
                              hmatel = -h2c_voov(e,k,m,b)
                              resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                          end do
                      end if
                      ! (bc)
                      idx = idx_table(a,b,i,k)
                      if (idx/=0) then
                          do jdet = loc_arr(idx,1), loc_arr(idx,2)
                              e = l3c_excits(2,jdet); m = l3c_excits(5,jdet);
                              ! compute < ij~k~ab~c~ | h2c(voov) | im~k~ae~b~ >
                              hmatel = -h2c_voov(e,j,m,c)
                              resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                          end do
                      end if
                      ! (jk)(bc)
                      idx = idx_table(a,b,i,j)
                      if (idx/=0) then
                          do jdet = loc_arr(idx,1), loc_arr(idx,2)
                              e = l3c_excits(2,jdet); m = l3c_excits(5,jdet);
                              ! compute < ij~k~ab~c~ | h2c(voov) | im~j~ae~b~ >
                              hmatel = h2c_voov(e,k,m,c)
                              resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
           
                  !!!! diagram 10: h2a(amie)*t3c(ebcmjk)
                  ! allocate sorting arrays
                  nloc = nub*(nub-1)/2*nob*(nob-1)/2
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nub,nub,nob,nob))
                  !!! SB: (2,3,5,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(l3c_excits, l3c_amps, loc_arr, idx_table, (/2,3,5,6/), nub, nub, nob, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,&
                  !$omp n3abb),&
                  !$omp private(hmatel,a,b,c,i,j,k,f,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                      a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                      i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                      idx = idx_table(b,c,j,k)
                      do jdet = loc_arr(idx,1), loc_arr(idx,2)
                         d = l3c_excits(1,jdet); l = l3c_excits(4,jdet);
                         ! compute < ij~k~ab~c~ | h2a(voov) | lj~k~db~c~ >
                         hmatel = h2a_voov(d,i,l,a)
                         resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                      end do
                  end do ! end loop over idet
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
           
                  !!!! diagram 11: -A(bc) h2b(iemb)*l3c(aecmjk)
                  ! allocate sorting arrays
                  nloc = nob*(nob-1)/2*nub*nua
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nob,nob,nua,nub))
                  !!! SB: (5,6,1,3) LOOP !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,nua/), (/2,nub/), nob, nob, nua, nub)
                  call sort4(l3c_excits, l3c_amps, loc_arr, idx_table, (/5,6,1,3/), nob, nob, nua, nub, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                      a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                      i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(j,k,a,c)
                      do jdet = loc_arr(idx,1), loc_arr(idx,2)
                          e = l3c_excits(2,jdet); l = l3c_excits(4,jdet);
                          ! compute < ij~k~ab~c~ | h2b(ovov) | lj~k~ae~c~ >
                          hmatel = -h2b_ovov(i,e,l,b)
                          resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                      end do
                      ! (bc)
                      idx = idx_table(j,k,a,b)
                      if (idx/=0) then
                          do jdet = loc_arr(idx,1), loc_arr(idx,2)
                              e = l3c_excits(2,jdet); l = l3c_excits(4,jdet);
                              ! compute < ij~k~ab~c~ | h2b(ovov) | lj~k~ae~b~ >
                              hmatel = h2b_ovov(i,e,l,c)
                              resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (5,6,1,2) LOOP !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,nua/), (/1,nub-1/), nob, nob, nua, nub)
                  call sort4(l3c_excits, l3c_amps, loc_arr, idx_table, (/5,6,1,2/), nob, nob, nua, nub, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                      a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                      i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(j,k,a,b)
                      do jdet = loc_arr(idx,1), loc_arr(idx,2)
                          f = l3c_excits(3,jdet); l = l3c_excits(4,jdet);
                          ! compute < ij~k~ab~c~ | h2b(ovov) | lj~k~ab~f~ >
                          hmatel = -h2b_ovov(i,f,l,c)
                          resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                      end do
                      ! (bc)
                      idx = idx_table(j,k,a,c)
                      if (idx/=0) then
                          do jdet = loc_arr(idx,1), loc_arr(idx,2)
                              f = l3c_excits(3,jdet); l = l3c_excits(4,jdet);
                              ! compute < ij~k~ab~c~ | h2b(ovov) | lj~k~ac~f~ >
                              hmatel = h2b_ovov(i,f,l,b)
                              resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  
                  !!!! diagram 12: -A(bc) h2b(ejam)*l3c(ebcimk)
                  ! allocate sorting arrays
                  nloc = nub*(nub-1)/2*noa*nob
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nub,nub,noa,nob))
                  !!! SB: (2,3,4,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,noa/), (/2,nob/), nub, nub, noa, nob)
                  call sort4(l3c_excits, l3c_amps, loc_arr, idx_table, (/2,3,4,6/), nub, nub, noa, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_vovo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                      a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                      i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,i,k)
                      do jdet = loc_arr(idx,1), loc_arr(idx,2)
                          d = l3c_excits(1,jdet); m = l3c_excits(5,jdet);
                          ! compute < ij~k~ab~c~ | h2b(vovo) | im~k~db~c~ >
                          hmatel = -h2b_vovo(d,j,a,m)
                          resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                      end do
                      ! (jk)
                      idx = idx_table(b,c,i,j)
                      if (idx/=0) then
                          do jdet = loc_arr(idx,1), loc_arr(idx,2)
                              d = l3c_excits(1,jdet); m = l3c_excits(5,jdet);
                              ! compute < ij~k~ab~c~ | h2b(vovo) | im~j~db~c~ >
                              hmatel = h2b_vovo(d,k,a,m)
                              resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4,5) LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,noa/), (/1,nob-1/), nub, nub, noa, nob)
                  call sort4(l3c_excits, l3c_amps, loc_arr, idx_table, (/2,3,4,5/), nub, nub, noa, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_vovo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                      a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                      i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,i,j)
                      do jdet = loc_arr(idx,1), loc_arr(idx,2)
                          d = l3c_excits(1,jdet); n = l3c_excits(6,jdet);
                          ! compute < ij~k~ab~c~ | h2b(vovo) | ij~n~db~c~ >
                          hmatel = -h2b_vovo(d,k,a,n)
                          resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                      end do
                      ! (jk)
                      idx = idx_table(b,c,i,k)
                      if (idx/=0) then
                          do jdet = loc_arr(idx,1), loc_arr(idx,2)
                              d = l3c_excits(1,jdet); n = l3c_excits(6,jdet);
                              ! compute < ij~k~ab~c~ | h2b(vovo) | ik~n~db~c~ >
                              hmatel = h2b_vovo(d,j,a,n)
                              resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)           

                  !!!! diagram 13: h2b(ieam)*l3d(ebcmjk)
                  ! allocate and initialize the copy of l3d
                  allocate(amps_buff(n3bbb))
                  allocate(excits_buff(6,n3bbb))
                  amps_buff(:) = l3d_amps(:)
                  excits_buff(:,:) = l3d_excits(:,:)
                  ! allocate sorting arrays
                  nloc = (nub-1)*(nub-2)/2*(nob-1)*(nob-2)/2
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nub,nub,nob,nob))
                  !!! SB: (2,3,5,6) LOOP !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,5,6/), nub, nub, nob, nob, nloc, n3bbb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                      a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                      i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(idx,1), loc_arr(idx,2)
                              d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                              ! compute < ij~k~ab~c~ | h2b(ovvo) | l~j~k~d~b~c~ >
                              hmatel = h2b_ovvo(i,d,a,l)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4,6) LOOP !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,6/), nub, nub, nob, nob, nloc, n3bbb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                      a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                      i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(idx,1), loc_arr(idx,2)
                              d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                              ! compute < ij~k~ab~c~ | h2b(ovvo) | j~m~k~d~b~c~ >
                              hmatel = -h2b_ovvo(i,d,a,m)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4,5) LOOP !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,5/), nub, nub, nob, nob, nloc, n3bbb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                      a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                      i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(idx,1), loc_arr(idx,2)
                              d = excits_buff(1,jdet); n = excits_buff(6,jdet);
                              ! compute < ij~k~ab~c~ | h2b(ovvo) | j~k~n~d~b~c~ >
                              hmatel = h2b_ovvo(i,d,a,n)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,5,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,5,6/), nub, nub, nob, nob, nloc, n3bbb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                      a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                      i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(idx,1), loc_arr(idx,2)
                              e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                              ! compute < ij~k~ab~c~ | h2b(ovvo) | l~j~k~b~e~c~ >
                              hmatel = -h2b_ovvo(i,e,a,l)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,4,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,6/), nub, nub, nob, nob, nloc, n3bbb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                      a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                      i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(idx,1), loc_arr(idx,2)
                              e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                              ! compute < ij~k~ab~c~ | h2b(ovvo) | j~m~k~b~e~c~ >
                              hmatel = h2b_ovvo(i,e,a,m)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,4,5) LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,5/), nub, nub, nob, nob, nloc, n3bbb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                      a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                      i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(idx,1), loc_arr(idx,2)
                              e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                              ! compute < ij~k~ab~c~ | h2b(ovvo) | j~k~n~b~e~c~ >
                              hmatel = -h2b_ovvo(i,e,a,n)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,5,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,5,6/), nub, nub, nob, nob, nloc, n3bbb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                      a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                      i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(idx,1), loc_arr(idx,2)
                              f = excits_buff(3,jdet); l = excits_buff(4,jdet);
                              ! compute < ij~k~ab~c~ | h2b(ovvo) | l~j~k~b~c~f~ >
                              hmatel = h2b_ovvo(i,f,a,l)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,6/), nub, nub, nob, nob, nloc, n3bbb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                      a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                      i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(idx,1), loc_arr(idx,2)
                              f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                              ! compute < ij~k~ab~c~ | h2b(ovvo) | j~m~k~b~c~f~ >
                              hmatel = -h2b_ovvo(i,f,a,m)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4,5) LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,5/), nub, nub, nob, nob, nloc, n3bbb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                      a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                      i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(idx,1), loc_arr(idx,2)
                              f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                              ! compute < ij~k~ab~c~ | h2b(ovvo) | j~k~n~b~c~f~ >
                              hmatel = h2b_ovvo(i,f,a,n)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate l3 buffer arrays
                  deallocate(amps_buff,excits_buff)

                  !!!! diagram 14: A(bc)A(jk) h2b(ejmb)*l3b(aecimk)
                  ! allocate and initialize the copy of l3a
                  allocate(amps_buff(n3aab))
                  allocate(excits_buff(6,n3aab))
                  amps_buff(:) = l3b_amps(:)
                  excits_buff(:,:) = l3b_excits(:,:)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = nua*nub*noa*nob
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nua,nub,noa,nob))
                  !!! SB: (1,3,4,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/1,nub/), (/1,noa-1/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,6/), nua, nub, noa, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                     a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                     i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                            e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                            ! compute < ij~k~ab~c~ | h2b(voov) | imk~aec~ >
                            hmatel = h2b_voov(e,j,m,b)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                            e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                            ! compute < ij~k~ab~c~ | h2b(voov) | imk~aeb~ >
                            hmatel = -h2b_voov(e,j,m,c)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                            e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                            ! compute < ij~k~ab~c~ | h2b(voov) | imj~aec~ >
                            hmatel = -h2b_voov(e,k,m,b)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                            e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                            ! compute < ij~k~ab~c~ | h2b(voov) | imj~aeb~ >
                            hmatel = h2b_voov(e,k,m,c)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4,6) LOOP !!!
                  call get_index_table(idx_table, (/2,nua/), (/1,nub/), (/1,noa-1/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,6/), nua, nub, noa, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                     a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                     i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                            d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                            ! compute < ij~k~ab~c~ | h2b(voov) | imk~dac~ >
                            hmatel = -h2b_voov(d,j,m,b)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                            d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                            ! compute < ij~k~ab~c~ | h2b(voov) | imk~dab~ >
                            hmatel = h2b_voov(d,j,m,c)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                            d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                            ! compute < ij~k~ab~c~ | h2b(voov) | imj~dac~ >
                            hmatel = h2b_voov(d,k,m,b)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                            d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                            ! compute < ij~k~ab~c~ | h2b(voov) | imj~dab~ >
                            hmatel = -h2b_voov(d,k,m,c)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,5,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/1,nub/), (/2,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,5,6/), nua, nub, noa, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                     a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                     i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                            e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                            ! compute < ij~k~ab~c~ | h2b(voov) | lik~aec~ >
                            hmatel = -h2b_voov(e,j,l,b)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                            e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                            ! compute < ij~k~ab~c~ | h2b(voov) | lik~aeb~ >
                            hmatel = h2b_voov(e,j,l,c)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                            e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                            ! compute < ij~k~ab~c~ | h2b(voov) | lij~aec~ >
                            hmatel = h2b_voov(e,k,l,b)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                            e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                            ! compute < ij~k~ab~c~ | h2b(voov) | lij~aeb~ >
                            hmatel = -h2b_voov(e,k,l,c)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,5,6) LOOP !!!
                  call get_index_table(idx_table, (/2,nua/), (/1,nub/), (/2,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,5,6/), nua, nub, noa, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                     a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                     i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                            d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                            ! compute < ij~k~ab~c~ | h2b(voov) | lik~dac~ >
                            hmatel = h2b_voov(d,j,l,b)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                            d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                            ! compute < ij~k~ab~c~ | h2b(voov) | lik~dab~ >
                            hmatel = -h2b_voov(d,j,l,c)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                            d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                            ! compute < ij~k~ab~c~ | h2b(voov) | lij~dac~ >
                            hmatel = -h2b_voov(d,k,l,b)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                            d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                            ! compute < ij~k~ab~c~ | h2b(voov) | lij~dab~ >
                            hmatel = h2b_voov(d,k,l,c)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate l3 buffer arrays
                  deallocate(amps_buff,excits_buff)

                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l1a,l1b,l2b,l2c,&
                  !$omp H1A_ov,H1B_ov,H2B_oovv,H2C_oovv,&
                  !$omp H2C_vovv,H2A_ooov,&
                  !$omp H2B_vovv,H2B_ovvv,H2B_ooov,H2B_oovo,&
                  !$omp X2C_vovv,X2C_ooov,&
                  !$omp X2B_vovv,X2B_ovvv,X2B_ooov,X2B_oovo,&
                  !$omp noa,nob,nua,nub,n3bbb),&
                  !$omp private(idet,a,b,c,d,i,j,k,l,m,n,e,f,&
                  !$omp res)
                  !$omp do schedule(static)
                  do idet = 1, n3abb
                      c = l3c_excits(1,idet); b = l3c_excits(2,idet); a = l3c_excits(3,idet);
                      k = l3c_excits(4,idet); j = l3c_excits(5,idet); i = l3c_excits(6,idet);
                      ! A(ab)A(ij) l1b(ai)*h2b(kjcb)
                      ! l1a(ck)*h2c(ijab)
                      ! A(ab)A(ij) l2b(cbkj)*h1b(ia)
                      ! l2c(abij)*h1a(kc)
                      res =  l1b(a,i)*h2b_oovv(k,j,c,b) + l2b(c,b,k,j)*h1b_ov(i,a)& ! (1)
                            -l1b(a,j)*h2b_oovv(k,i,c,b) - l2b(c,b,k,i)*h1b_ov(j,a)& ! (ij)
                            -l1b(b,i)*h2b_oovv(k,j,c,a) - l2b(c,a,k,j)*h1b_ov(i,b)& ! (ab)
                            +l1b(b,j)*h2b_oovv(k,i,c,a) + l2b(c,a,k,i)*h1b_ov(j,b)& ! (ab)(ij)
                            +l1a(c,k)*h2c_oovv(i,j,a,b) + l2c(a,b,i,j)*h1a_ov(k,c)
                      ! A(ab)A(ij) h2b(eica)*l2b(ebkj)
                      ! A(ab)A(ij) x2b(eica)*h2b(kjeb)
                      do e = 1, nua
                         res = res&
                               +h2b_vovv(e,i,c,a)*l2b(e,b,k,j) + x2b_vovv(e,i,c,a)*h2b_oovv(k,j,e,b)& ! (1)
                               -h2b_vovv(e,j,c,a)*l2b(e,b,k,i) - x2b_vovv(e,j,c,a)*h2b_oovv(k,i,e,b)& ! (ij)
                               -h2b_vovv(e,i,c,b)*l2b(e,a,k,j) - x2b_vovv(e,i,c,b)*h2b_oovv(k,j,e,a)& ! (ab)
                               +h2b_vovv(e,j,c,b)*l2b(e,a,k,i) + x2b_vovv(e,j,c,b)*h2b_oovv(k,i,e,a)  ! (ij)(ab)
                      end do
                      ! A(ab) h2b(kecb)*l2c(aeij)
                      ! A(ij) h2c(eiba)*l2b(cekj)
                      ! A(ab) x2b(kecb)*h2c(ijae)
                      ! A(ij) x2c(eiba)*h2b(kjce)
                      do e = 1, nub
                         res = res&
                               +h2b_ovvv(k,e,c,b)*l2c(a,e,i,j) - h2b_ovvv(k,e,c,a)*l2c(b,e,i,j)&
                               +h2c_vovv(e,i,b,a)*l2b(c,e,k,j) - h2c_vovv(e,j,b,a)*l2b(c,e,k,i)&
                               +x2b_ovvv(k,e,c,b)*h2c_oovv(i,j,a,e) - x2b_ovvv(k,e,c,a)*h2c_oovv(i,j,b,e)&
                               +x2c_vovv(e,i,b,a)*h2b_oovv(k,j,c,e) - x2c_vovv(e,j,b,a)*h2b_oovv(k,i,c,e)
                      end do
                      ! A(ij)A(ab) -h2b(kima)*l2b(cbmj)
                      ! A(ij)A(ab) -x2b(kima)*h2b(mjcb)
                      do m = 1, noa
                         res = res&
                              -h2b_ooov(k,i,m,a)*l2b(c,b,m,j) - x2b_ooov(k,i,m,a)*h2b_oovv(m,j,c,b)& ! (1)
                              +h2b_ooov(k,j,m,a)*l2b(c,b,m,i) + x2b_ooov(k,j,m,a)*h2b_oovv(m,i,c,b)& ! (ij)
                              +h2b_ooov(k,i,m,b)*l2b(c,a,m,j) + x2b_ooov(k,i,m,b)*h2b_oovv(m,j,c,a)& ! (ab)
                              -h2b_ooov(k,j,m,b)*l2b(c,a,m,i) - x2b_ooov(k,j,m,b)*h2b_oovv(m,i,c,a)  ! (ij)(ab)
                      end do
                      ! A(ij) -h2b(kjcm)*l2c(abim)
                      ! A(ab) -h2c(jima)*l2b(cbkm)
                      ! A(ij) -x2b(kjcm)*h2c(imab)
                      ! A(ab) -x2c(jima)*h2b(kmcb)
                      do m = 1, nob
                         res = res&
                              -h2b_oovo(k,j,c,m)*l2c(a,b,i,m) + h2b_oovo(k,i,c,m)*l2c(a,b,j,m)&
                              -h2c_ooov(j,i,m,a)*l2b(c,b,k,m) + h2c_ooov(j,i,m,b)*l2b(c,a,k,m)&
                              -x2b_oovo(k,j,c,m)*h2c_oovv(i,m,a,b) + x2b_oovo(k,i,c,m)*h2c_oovv(j,m,a,b)&
                              -x2c_ooov(j,i,m,a)*h2b_oovv(k,m,c,b) + x2c_ooov(j,i,m,b)*h2b_oovv(k,m,c,a)
                      end do
                      resid(idet) = resid(idet) + res
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!

        end subroutine build_LH_3C

        subroutine build_LH_3D(resid,&
                              l1b, l2c,&
                              l3c_amps, l3c_excits,&
                              l3d_amps, l3d_excits,&
                              h1b_ov, h1b_oo, h1b_vv,&
                              h2b_voov,&
                              h2c_oooo, h2c_ooov, h2c_oovv,&
                              h2c_voov, h2c_vovv, h2c_vvvv,&
                              x2c_ooov, x2c_vovv,&
                              n3abb, n3bbb,&
                              noa, nua, nob, nub)
                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3abb, n3bbb
                  ! Input L arrays
                  real(kind=8), intent(in) :: l1b(nub,nob)
                  real(kind=8), intent(in) :: l2c(nub,nub,nob,nob)
                  integer, intent(in) :: l3c_excits(6,n3abb)
                  real(kind=8), intent(in) :: l3c_amps(n3abb)
                  ! Input H and X arrays
                  real(kind=8), intent(in) :: h1b_ov(nob,nub)
                  real(kind=8), intent(in) :: h1b_oo(nob,nob)
                  real(kind=8), intent(in) :: h1b_vv(nub,nub)
                  real(kind=8), intent(in) :: h2c_oooo(nob,nob,nob,nob)
                  real(kind=8), intent(in) :: h2c_ooov(nob,nob,nob,nub)
                  real(kind=8), intent(in) :: h2c_oovv(nob,nob,nub,nub)
                  real(kind=8), intent(in) :: h2c_voov(nub,nob,nob,nub)
                  real(kind=8), intent(in) :: h2c_vovv(nub,nob,nub,nub)
                  real(kind=8), intent(in) :: h2c_vvvv(nub,nub,nub,nub)
                  real(kind=8), intent(in) :: h2b_voov(nua,nob,noa,nub)
                  real(kind=8), intent(in) :: x2c_ooov(nob,nob,nob,nub)
                  real(kind=8), intent(in) :: x2c_vovv(nub,nob,nub,nub)
                  ! Output and Inout variables
                  real(kind=8), intent(out) :: resid(n3bbb)
                  integer, intent(inout) :: l3d_excits(6,n3bbb)
                  !f2py intent(in,out) :: l3d_excits(6,0:n3bbb-1)
                  real(kind=8), intent(inout) :: l3d_amps(n3bbb)
                  !f2py intent(in,out) :: l3d_amps(0:n3bbb-1)
                  ! Local variables
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:)
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)
                  real(kind=8) :: l_amp, hmatel, hmatel1, res
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc

                  resid = 0.0d0
                  !!!! diagram 1: -A(i/jk) h1b(im) * l3d(abcmjk)
                  !!!! diagram 3: 1/2 A(k/ij) h2c(ijmn) * l3d(abcmnk)
                  ! NOTE: WITHIN THESE LOOPS, H1B(OO) TERMS ARE DOUBLE-COUNTED SO COMPENSATE BY FACTOR OF 1/2
                  ! allocate new sorting arrays
                  nloc = nub*(nub-1)*(nub-2)/6*nob
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nub,nub,nub,nob))
                  !!! SB: (1,2,3,6) !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/-1,nub/), (/3,nob/), nub, nub, nub, nob)
                  call sort4(l3d_excits, l3d_amps, loc_arr, idx_table, (/1,2,3,6/), nub, nub, nub, nob, nloc, n3bbb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3d_excits,&
                  !$omp l3d_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h1b_oo,h2c_oooo,&
                  !$omp noa,nua,n3bbb),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb
                     a = l3d_excits(1,idet); b = l3d_excits(2,idet); c = l3d_excits(3,idet);
                     i = l3d_excits(4,idet); j = l3d_excits(5,idet); k = l3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        l = l3d_excits(4,jdet); m = l3d_excits(5,jdet);
                        ! compute < lmkabc | h2c(oooo) | ijkabc >
                        hmatel = h2c_oooo(i,j,l,m)
                        ! compute < lmkabc | h1b(oo) | ijkabc > = -A(ij)A(lm) h1b_oo(i,l) * delta(j,m)
                        hmatel1 = 0.0d0
                        if (m==j) hmatel1 = hmatel1 - h1b_oo(i,l) ! (1)
                        if (m==i) hmatel1 = hmatel1 + h1b_oo(j,l) ! (ij)
                        if (l==j) hmatel1 = hmatel1 + h1b_oo(i,m) ! (lm)
                        if (l==i) hmatel1 = hmatel1 - h1b_oo(j,m) ! (ij)(lm)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     ! (ik)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l = l3d_excits(4,jdet); m = l3d_excits(5,jdet);
                           ! compute < lmiabc | h2c(oooo) | ijkabc >
                           hmatel = -h2c_oooo(k,j,l,m)
                           ! compute < lmiabc | h1b(oo) | ijkabc > = A(jk)A(lm) h1b_oo(k,l) * delta(j,m)
                           hmatel1 = 0.0d0
                           if (m==j) hmatel1 = hmatel1 + h1b_oo(k,l) ! (1)
                           if (m==k) hmatel1 = hmatel1 - h1b_oo(j,l) ! (jk)
                           if (l==j) hmatel1 = hmatel1 - h1b_oo(k,m) ! (lm)
                           if (l==k) hmatel1 = hmatel1 + h1b_oo(j,m) ! (jk)(lm)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l = l3d_excits(4,jdet); m = l3d_excits(5,jdet);
                           ! compute < lmjabc | h2c(oooo) | ijkabc >
                           hmatel = -h2c_oooo(i,k,l,m)
                           ! compute < lmjabc | h1b(oo) | ijkabc > = A(ik)A(lm) h1b_oo(i,l) * delta(k,m)
                           hmatel1 = 0.0d0
                           if (m==k) hmatel1 = hmatel1 + h1b_oo(i,l) ! (1)
                           if (m==i) hmatel1 = hmatel1 - h1b_oo(k,l) ! (ik)
                           if (l==k) hmatel1 = hmatel1 - h1b_oo(i,m) ! (lm)
                           if (l==i) hmatel1 = hmatel1 + h1b_oo(k,m) ! (ik)(lm)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,3,4) !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/-1,nub/), (/1,nob-2/), nub, nub, nub, nob)
                  call sort4(l3d_excits, l3d_amps, loc_arr, idx_table, (/1,2,3,4/), nub, nub, nub, nob, nloc, n3bbb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3d_excits,&
                  !$omp l3d_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1B_oo,H2C_oooo,&
                  !$omp noa,nua,n3bbb),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb
                     a = l3d_excits(1,idet); b = l3d_excits(2,idet); c = l3d_excits(3,idet);
                     i = l3d_excits(4,idet); j = l3d_excits(5,idet); k = l3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,i)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        m = l3d_excits(5,jdet); n = l3d_excits(6,jdet);
                        ! compute < imnabc | h2c(oooo) | ijkabc >
                        hmatel = h2c_oooo(j,k,m,n)
                        ! compute < imnabc | h1b(oo) | ijkabc > = -A(jk)A(mn) h1b_oo(j,m) * delta(k,n)
                        hmatel1 = 0.0d0
                        if (n==k) hmatel1 = hmatel1 - h1b_oo(j,m) ! (1)
                        if (n==j) hmatel1 = hmatel1 + h1b_oo(k,m) ! (jk)
                        if (m==k) hmatel1 = hmatel1 + h1b_oo(j,n) ! (mn)
                        if (m==j) hmatel1 = hmatel1 - h1b_oo(k,n) ! (jk)(mn)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           m = l3d_excits(5,jdet); n = l3d_excits(6,jdet);
                           ! compute < jmnabc | h2c(oooo) | ijkabc >
                           hmatel = -h2c_oooo(i,k,m,n)
                           ! compute < jmnabc | h1b(oo) | ijkabc > = A(ik)A(mn) h1b_oo(i,m) * delta(k,n)
                           hmatel1 = 0.0d0
                           if (n==k) hmatel1 = hmatel1 + h1b_oo(i,m) ! (1)
                           if (n==i) hmatel1 = hmatel1 - h1b_oo(k,m) ! (ik)
                           if (m==k) hmatel1 = hmatel1 - h1b_oo(i,n) ! (mn)
                           if (m==i) hmatel1 = hmatel1 + h1b_oo(k,n) ! (ik)(mn)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                        end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           m = l3d_excits(5,jdet); n = l3d_excits(6,jdet);
                           ! compute < kmnabc | h2c(oooo) | ijkabc >
                           hmatel = -h2c_oooo(j,i,m,n)
                           ! compute < kmnabc | h1b(oo) | ijkabc > = A(ij)A(mn) h1b_oo(j,m) * delta(i,n)
                           hmatel1 = 0.0d0
                           if (n==i) hmatel1 = hmatel1 - h1b_oo(j,m) ! (1)
                           if (n==j) hmatel1 = hmatel1 + h1b_oo(i,m) ! (ij)
                           if (m==i) hmatel1 = hmatel1 + h1b_oo(j,n) ! (mn)
                           if (m==j) hmatel1 = hmatel1 - h1b_oo(i,n) ! (ij)(mn)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,3,5) !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/-1,nub/), (/2,nob-1/), nub, nub, nub, nob)
                  call sort4(l3d_excits, l3d_amps, loc_arr, idx_table, (/1,2,3,5/), nub, nub, nub, nob, nloc, n3bbb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3d_excits,&
                  !$omp l3d_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1B_oo,H2C_oooo,&
                  !$omp noa,nua,n3bbb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb
                     a = l3d_excits(1,idet); b = l3d_excits(2,idet); c = l3d_excits(3,idet);
                     i = l3d_excits(4,idet); j = l3d_excits(5,idet); k = l3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,j)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        l = l3d_excits(4,jdet); n = l3d_excits(6,jdet);
                        ! compute < ljnabc | h2c(oooo) | ijkabc >
                        hmatel = h2c_oooo(i,k,l,n)
                        ! compute < ljnabc | h1b(oo) | ijkabc > = -A(ik)A(ln) h1b_oo(i,l) * delta(k,n)
                        hmatel1 = 0.0d0
                        if (n==k) hmatel1 = hmatel1 - h1b_oo(i,l) ! (1)
                        if (n==i) hmatel1 = hmatel1 + h1b_oo(k,l) ! (ik)
                        if (l==k) hmatel1 = hmatel1 + h1b_oo(i,n) ! (ln)
                        if (l==i) hmatel1 = hmatel1 - h1b_oo(k,n) ! (ik)(ln)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l = l3d_excits(4,jdet); n = l3d_excits(6,jdet);
                           ! compute < linabc | h2c(oooo) | ijkabc >
                           hmatel = -h2c_oooo(j,k,l,n)
                           ! compute < linabc | h1b(oo) | ijkabc > = A(jk)A(ln) h1b_oo(j,l) * delta(k,n)
                           hmatel1 = 0.0d0
                           if (n==k) hmatel1 = hmatel1 + h1b_oo(j,l) ! (1)
                           if (n==j) hmatel1 = hmatel1 - h1b_oo(k,l) ! (jk)
                           if (l==k) hmatel1 = hmatel1 - h1b_oo(j,n) ! (ln)
                           if (l==j) hmatel1 = hmatel1 + h1b_oo(k,n) ! (jk)(ln)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l = l3d_excits(4,jdet); n = l3d_excits(6,jdet);
                           ! compute < lknabc | h2c(oooo) | ijkabc >
                           hmatel = -h2c_oooo(i,j,l,n)
                           ! compute < lknabc | h1b(oo) | ijkabc > = A(ij)A(ln) h1b_oo(i,l) * delta(j,n)
                           hmatel1 = 0.0d0
                           if (n==j) hmatel1 = hmatel1 + h1b_oo(i,l) ! (1)
                           if (n==i) hmatel1 = hmatel1 - h1b_oo(j,l) ! (ij)
                           if (l==j) hmatel1 = hmatel1 - h1b_oo(i,n) ! (ln)
                           if (l==i) hmatel1 = hmatel1 + h1b_oo(j,n) ! (ij)(ln)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 2: A(a/bc) h1b(ea) * l3d(ebcijk)
                  !!!! diagram 4: 1/2 A(c/ab) h2c(efab) * l3d(ebcijk) 
                  ! NOTE: WITHIN THESE LOOPS, H1B(VV) TERMS ARE DOUBLE-COUNTED SO COMPENSATE BY FACTOR OF 1/2  
                  ! allocate new sorting arrays
                  nloc = nob*(nob-1)*(nob-2)/6*nub
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nob,nob,nob,nub))
                  !!! SB: (4,5,6,1) LOOP !!!
                  call get_index_table(idx_table, (/1,nob-2/), (/-1,nob-1/), (/-1,nob/), (/1,nub-2/), nob, nob, nob, nub)
                  call sort4(l3d_excits, l3d_amps, loc_arr, idx_table, (/4,5,6,1/), nob, nob, nob, nub, nloc, n3bbb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3d_excits,&
                  !$omp l3d_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1B_vv,H2C_vvvv,&
                  !$omp nob,nub,n3bbb),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb
                     a = l3d_excits(1,idet); b = l3d_excits(2,idet); c = l3d_excits(3,idet);
                     i = l3d_excits(4,idet); j = l3d_excits(5,idet); k = l3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,a)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); f = l3d_excits(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkaef >
                        hmatel = h2c_vvvv(e,f,b,c)
                        ! compute < ijkabc | h1a(vv) | ijkaef > = A(bc)A(ef) h1b_vv(b,e) * delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 + h1b_vv(e,b) ! (1)
                        if (b==f) hmatel1 = hmatel1 - h1b_vv(e,c) ! (bc)
                        if (c==e) hmatel1 = hmatel1 - h1b_vv(f,b) ! (ef)
                        if (b==e) hmatel1 = hmatel1 + h1b_vv(f,c) ! (bc)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); f = l3d_excits(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkbef >
                        hmatel = -h2c_vvvv(e,f,a,c)
                        ! compute < ijkabc | h1a(vv) | ijkbef > = -A(ac)A(ef) h1b_vv(a,e) * delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 - h1b_vv(e,a) ! (1)
                        if (a==f) hmatel1 = hmatel1 + h1b_vv(e,c) ! (ac)
                        if (c==e) hmatel1 = hmatel1 + h1b_vv(f,a) ! (ef)
                        if (a==e) hmatel1 = hmatel1 - h1b_vv(f,c) ! (ac)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); f = l3d_excits(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkcef >
                        hmatel = -h2c_vvvv(e,f,b,a)
                        ! compute < ijkabc | h1a(vv) | ijkcef > = -A(ab)A(ef) h1b_vv(b,e) * delta(a,f)
                        hmatel1 = 0.0d0
                        if (a==f) hmatel1 = hmatel1 - h1b_vv(e,b) ! (1)
                        if (b==f) hmatel1 = hmatel1 + h1b_vv(e,a) ! (ab)
                        if (a==e) hmatel1 = hmatel1 + h1b_vv(f,b) ! (ef)
                        if (b==e) hmatel1 = hmatel1 - h1b_vv(f,a) ! (ab)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,6,2) LOOP !!!
                  call get_index_table(idx_table, (/1,nob-2/), (/-1,nob-1/), (/-1,nob/), (/2,nub-1/), nob, nob, nob, nub)
                  call sort4(l3d_excits, l3d_amps, loc_arr, idx_table, (/4,5,6,2/), nob, nob, nob, nub, nloc, n3bbb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3d_excits,&
                  !$omp l3d_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1B_vv,H2C_vvvv,&
                  !$omp nob,nub,n3bbb),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb
                     a = l3d_excits(1,idet); b = l3d_excits(2,idet); c = l3d_excits(3,idet);
                     i = l3d_excits(4,idet); j = l3d_excits(5,idet); k = l3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,b)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); f = l3d_excits(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdbf >
                        hmatel = h2c_vvvv(d,f,a,c)
                        ! compute < ijkabc | h1a(vv) | ijkdbf > = A(ac)A(df) h1b_vv(a,d) * delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 + h1b_vv(d,a) ! (1)
                        if (a==f) hmatel1 = hmatel1 - h1b_vv(d,c) ! (ac)
                        if (c==d) hmatel1 = hmatel1 - h1b_vv(f,a) ! (df)
                        if (a==d) hmatel1 = hmatel1 + h1b_vv(f,c) ! (ac)(df)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); f = l3d_excits(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdaf >
                        hmatel = -h2c_vvvv(d,f,b,c)
                        ! compute < ijkabc | h1a(vv) | ijkdaf > = -A(bc)A(df) h1b_vv(b,d) * delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 - h1b_vv(d,b) ! (1)
                        if (b==f) hmatel1 = hmatel1 + h1b_vv(d,c) ! (bc)
                        if (c==d) hmatel1 = hmatel1 + h1b_vv(f,b) ! (df)
                        if (b==d) hmatel1 = hmatel1 - h1b_vv(f,c) ! (bc)(df)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); f = l3d_excits(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdcf >
                        hmatel = -h2c_vvvv(d,f,a,b)
                        ! compute < ijkabc | h1a(vv) | ijkdcf > = -A(ab)A(df) h1b_vv(a,d) * delta(b,f)
                        hmatel1 = 0.0d0
                        if (b==f) hmatel1 = hmatel1 - h1b_vv(d,a) ! (1)
                        if (a==f) hmatel1 = hmatel1 + h1b_vv(d,b) ! (ab)
                        if (b==d) hmatel1 = hmatel1 + h1b_vv(f,a) ! (df)
                        if (a==d) hmatel1 = hmatel1 - h1b_vv(f,b) ! (ab)(df)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if 
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,6,3) LOOP !!!
                  call get_index_table(idx_table, (/1,nob-2/), (/-1,nob-1/), (/-1,nob/), (/3,nub/), nob, nob, nob, nub)
                  call sort4(l3d_excits, l3d_amps, loc_arr, idx_table, (/4,5,6,3/), nob, nob, nob, nub, nloc, n3bbb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3d_excits,&
                  !$omp l3d_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1B_vv,H2C_vvvv,&
                  !$omp nob,nub,n3bbb),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb
                     a = l3d_excits(1,idet); b = l3d_excits(2,idet); c = l3d_excits(3,idet);
                     i = l3d_excits(4,idet); j = l3d_excits(5,idet); k = l3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,c)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); e = l3d_excits(2,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdec >
                        hmatel = h2c_vvvv(d,e,a,b)
                        ! compute < ijkabc | h1a(vv) | ijkdec > = A(ab)A(de) h1b_vv(a,d) * delta(b,e)
                        hmatel1 = 0.0d0
                        if (b==e) hmatel1 = hmatel1 + h1b_vv(d,a) ! (1)
                        if (a==e) hmatel1 = hmatel1 - h1b_vv(d,b) ! (ab)
                        if (b==d) hmatel1 = hmatel1 - h1b_vv(e,a) ! (de)
                        if (a==d) hmatel1 = hmatel1 + h1b_vv(e,b) ! (ab)(de)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); e = l3d_excits(2,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdea >
                        hmatel = -h2c_vvvv(d,e,c,b)
                        ! compute < ijkabc | h1a(vv) | ijkdea > = -A(bc)A(de) h1b_vv(c,d) * delta(b,e)
                        hmatel1 = 0.0d0
                        if (b==e) hmatel1 = hmatel1 - h1b_vv(d,c) ! (1)
                        if (c==e) hmatel1 = hmatel1 + h1b_vv(d,b) ! (bc)
                        if (b==d) hmatel1 = hmatel1 + h1b_vv(e,c) ! (de)
                        if (c==d) hmatel1 = hmatel1 - h1b_vv(e,b) ! (bc)(de)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); e = l3d_excits(2,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdeb >
                        hmatel = -h2c_vvvv(d,e,a,c)
                        ! compute < ijkabc | h1a(vv) | ijkdeb > = -A(ac)A(de) h1b_vv(a,d) * delta(c,e)
                        hmatel1 = 0.0d0
                        if (c==e) hmatel1 = hmatel1 - h1b_vv(d,a) ! (1)
                        if (a==e) hmatel1 = hmatel1 + h1b_vv(d,c) ! (ac)
                        if (c==d) hmatel1 = hmatel1 + h1b_vv(e,a) ! (de)
                        if (a==d) hmatel1 = hmatel1 - h1b_vv(e,c) ! (ac)(de)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  
                  !!!! diagram 5: A(i/jk)A(a/bc) h2c(eima) * l3c(ebcmjk)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = (nub-1)*(nub-2)/2*(nob-1)*(nob-2)/2
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nub,nub,nob,nob))
                  !!! SB: (1,2,4,5) LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4(l3d_excits, l3d_amps, loc_arr, idx_table, (/1,2,4,5/), nub, nub, nob, nob, nloc, n3bbb, resid)
                  do idet = 1, n3bbb
                     a = l3d_excits(1,idet); b = l3d_excits(2,idet); c = l3d_excits(3,idet);
                     i = l3d_excits(4,idet); j = l3d_excits(5,idet); k = l3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3d_excits(3,jdet); n = l3d_excits(6,jdet);
                        ! compute < ijnabf | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(f,k,n,c)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3d_excits(3,jdet); n = l3d_excits(6,jdet);
                        ! compute < ijnbcf | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(f,k,n,a)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3d_excits(3,jdet); n = l3d_excits(6,jdet);
                        ! compute < ijnacf | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(f,k,n,b)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3d_excits(3,jdet); n = l3d_excits(6,jdet);
                        ! compute < jknabf | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(f,i,n,c)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3d_excits(3,jdet); n = l3d_excits(6,jdet);
                        ! compute < jknbcf | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(f,i,n,a)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3d_excits(3,jdet); n = l3d_excits(6,jdet);
                        ! compute < jknacf | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(f,i,n,b)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3d_excits(3,jdet); n = l3d_excits(6,jdet);
                        ! compute < iknabf | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(f,j,n,c)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3d_excits(3,jdet); n = l3d_excits(6,jdet);
                        ! compute < iknbcf | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(f,j,n,a)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3d_excits(3,jdet); n = l3d_excits(6,jdet);
                        ! compute < iknacf | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(f,j,n,b)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,3,4,5) LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4(l3d_excits, l3d_amps, loc_arr, idx_table, (/1,3,4,5/), nub, nub, nob, nob, nloc, n3bbb, resid)
                  do idet = 1, n3bbb
                     a = l3d_excits(1,idet); b = l3d_excits(2,idet); c = l3d_excits(3,idet);
                     i = l3d_excits(4,idet); j = l3d_excits(5,idet); k = l3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); n = l3d_excits(6,jdet);
                        ! compute < ijnaec | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(e,k,n,b)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); n = l3d_excits(6,jdet);
                        ! compute < ijnbec | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(e,k,n,a)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); n = l3d_excits(6,jdet);
                        ! compute < ijnaeb | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(e,k,n,c)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); n = l3d_excits(6,jdet);
                        ! compute < jknaec | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(e,i,n,b)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); n = l3d_excits(6,jdet);
                        ! compute < jknbec | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(e,i,n,a)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); n = l3d_excits(6,jdet);
                        ! compute < jknaeb | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(e,i,n,c)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); n = l3d_excits(6,jdet);
                        ! compute < iknaec | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(e,j,n,b)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); n = l3d_excits(6,jdet);
                        ! compute < iknbec | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(e,j,n,a)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); n = l3d_excits(6,jdet);
                        ! compute < iknaeb | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(e,j,n,c)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (2,3,4,5) LOOP !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4(l3d_excits, l3d_amps, loc_arr, idx_table, (/2,3,4,5/), nub, nub, nob, nob, nloc, n3bbb, resid)
                  do idet = 1, n3bbb
                     a = l3d_excits(1,idet); b = l3d_excits(2,idet); c = l3d_excits(3,idet);
                     i = l3d_excits(4,idet); j = l3d_excits(5,idet); k = l3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); n = l3d_excits(6,jdet);
                        ! compute < ijndbc | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(d,k,n,a)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); n = l3d_excits(6,jdet);
                        ! compute < ijndac | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(d,k,n,b)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); n = l3d_excits(6,jdet);
                        ! compute < ijndab | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(d,k,n,c)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); n = l3d_excits(6,jdet);
                        ! compute < jkndbc | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(d,i,n,a)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); n = l3d_excits(6,jdet);
                        ! compute < jkndac | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(d,i,n,b)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); n = l3d_excits(6,jdet);
                        ! compute < jkndab | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(d,i,n,c)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); n = l3d_excits(6,jdet);
                        ! compute < ikndbc | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(d,j,n,a)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); n = l3d_excits(6,jdet);
                        ! compute < ikndac | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(d,j,n,b)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); n = l3d_excits(6,jdet);
                        ! compute < ikndab | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(d,j,n,c)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,2,4,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4(l3d_excits, l3d_amps, loc_arr, idx_table, (/1,2,4,6/), nub, nub, nob, nob, nloc, n3bbb, resid)
                  do idet = 1, n3bbb
                     a = l3d_excits(1,idet); b = l3d_excits(2,idet); c = l3d_excits(3,idet);
                     i = l3d_excits(4,idet); j = l3d_excits(5,idet); k = l3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3d_excits(3,jdet); m = l3d_excits(5,jdet);
                        ! compute < imkabf | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(f,j,m,c)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3d_excits(3,jdet); m = l3d_excits(5,jdet);
                        ! compute < imkbcf | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(f,j,m,a)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3d_excits(3,jdet); m = l3d_excits(5,jdet);
                        ! compute < imkacf | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(f,j,m,b)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3d_excits(3,jdet); m = l3d_excits(5,jdet);
                        ! compute < jmkabf | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(f,i,m,c)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3d_excits(3,jdet); m = l3d_excits(5,jdet);
                        ! compute < jmkbcf | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(f,i,m,a)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3d_excits(3,jdet); m = l3d_excits(5,jdet);
                        ! compute < jmkacf | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(f,i,m,b)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3d_excits(3,jdet); m = l3d_excits(5,jdet);
                        ! compute < imjabf | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(f,k,m,c)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3d_excits(3,jdet); m = l3d_excits(5,jdet);
                        ! compute < imjbcf | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(f,k,m,a)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3d_excits(3,jdet); m = l3d_excits(5,jdet);
                        ! compute < imjacf | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(f,k,m,b)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,3,4,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4(l3d_excits, l3d_amps, loc_arr, idx_table, (/1,3,4,6/), nub, nub, nob, nob, nloc, n3bbb, resid)
                  do idet = 1, n3bbb
                     a = l3d_excits(1,idet); b = l3d_excits(2,idet); c = l3d_excits(3,idet);
                     i = l3d_excits(4,idet); j = l3d_excits(5,idet); k = l3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); m = l3d_excits(5,jdet);
                        ! compute < imkaec | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(e,j,m,b)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); m = l3d_excits(5,jdet);
                        ! compute < imkbec | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(e,j,m,a)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); m = l3d_excits(5,jdet);
                        ! compute < imkaeb | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(e,j,m,c)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); m = l3d_excits(5,jdet);
                        ! compute < jmkaec | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(e,i,m,b)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); m = l3d_excits(5,jdet);
                        ! compute < jmkbec | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(e,i,m,a)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); m = l3d_excits(5,jdet);
                        ! compute < jmkaeb | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(e,i,m,c)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); m = l3d_excits(5,jdet);
                        ! compute < imjaec | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(e,k,m,b)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); m = l3d_excits(5,jdet);
                        ! compute < imjbec | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(e,k,m,a)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); m = l3d_excits(5,jdet);
                        ! compute < imjaeb | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(e,k,m,c)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (2,3,4,6) LOOP !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4(l3d_excits, l3d_amps, loc_arr, idx_table, (/2,3,4,6/), nub, nub, nob, nob, nloc, n3bbb, resid)
                  do idet = 1, n3bbb
                     a = l3d_excits(1,idet); b = l3d_excits(2,idet); c = l3d_excits(3,idet);
                     i = l3d_excits(4,idet); j = l3d_excits(5,idet); k = l3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); m = l3d_excits(5,jdet);
                        ! compute < imkdbc | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(d,j,m,a)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); m = l3d_excits(5,jdet);
                        ! compute < imkdac | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(d,j,m,b)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); m = l3d_excits(5,jdet);
                        ! compute < imkdab | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(d,j,m,c)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); m = l3d_excits(5,jdet);
                        ! compute < jmkdbc | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(d,i,m,a)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); m = l3d_excits(5,jdet);
                        ! compute < jmkdac | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(d,i,m,b)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); m = l3d_excits(5,jdet);
                        ! compute < jmkdab | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(d,i,m,c)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); m = l3d_excits(5,jdet);
                        ! compute < imjdbc | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(d,k,m,a)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); m = l3d_excits(5,jdet);
                        ! compute < imjdac | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(d,k,m,b)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); m = l3d_excits(5,jdet);
                        ! compute < imjdab | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(d,k,m,c)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,2,5,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(l3d_excits, l3d_amps, loc_arr, idx_table, (/1,2,5,6/), nub, nub, nob, nob, nloc, n3bbb, resid)
                  do idet = 1, n3bbb
                     a = l3d_excits(1,idet); b = l3d_excits(2,idet); c = l3d_excits(3,idet);
                     i = l3d_excits(4,idet); j = l3d_excits(5,idet); k = l3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3d_excits(3,jdet); l = l3d_excits(4,jdet);
                        ! compute < ljkabf | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(f,i,l,c)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3d_excits(3,jdet); l = l3d_excits(4,jdet);
                        ! compute < ljkbcf | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(f,i,l,a)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3d_excits(3,jdet); l = l3d_excits(4,jdet);
                        ! compute < ljkacf | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(f,i,l,b)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3d_excits(3,jdet); l = l3d_excits(4,jdet);
                        ! compute < likabf | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(f,j,l,c)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3d_excits(3,jdet); l = l3d_excits(4,jdet);
                        ! compute < likbcf | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(f,j,l,a)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3d_excits(3,jdet); l = l3d_excits(4,jdet);
                        ! compute < likacf | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(f,j,l,b)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3d_excits(3,jdet); l = l3d_excits(4,jdet);
                        ! compute < lijabf | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(f,k,l,c)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3d_excits(3,jdet); l = l3d_excits(4,jdet);
                        ! compute < lijbcf | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(f,k,l,a)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        f = l3d_excits(3,jdet); l = l3d_excits(4,jdet);
                        ! compute < lijacf | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(f,k,l,b)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,3,5,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(l3d_excits, l3d_amps, loc_arr, idx_table, (/1,3,5,6/), nub, nub, nob, nob, nloc, n3bbb, resid)
                  do idet = 1, n3bbb
                     a = l3d_excits(1,idet); b = l3d_excits(2,idet); c = l3d_excits(3,idet);
                     i = l3d_excits(4,idet); j = l3d_excits(5,idet); k = l3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); l = l3d_excits(4,jdet);
                        ! compute < ljkaec | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(e,i,l,b)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); l = l3d_excits(4,jdet);
                        ! compute < ljkbec | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(e,i,l,a)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); l = l3d_excits(4,jdet);
                        ! compute < ljkaeb | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(e,i,l,c)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); l = l3d_excits(4,jdet);
                        ! compute < likaec | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(e,j,l,b)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); l = l3d_excits(4,jdet);
                        ! compute < likbec | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(e,j,l,a)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); l = l3d_excits(4,jdet);
                        ! compute < likaeb | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(e,j,l,c)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); l = l3d_excits(4,jdet);
                        ! compute < lijaec | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(e,k,l,b)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); l = l3d_excits(4,jdet);
                        ! compute < lijbec | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(e,k,l,a)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = l3d_excits(2,jdet); l = l3d_excits(4,jdet);
                        ! compute < lijaeb | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(e,k,l,c)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (2,3,5,6) LOOP !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(l3d_excits, l3d_amps, loc_arr, idx_table, (/2,3,5,6/), nub, nub, nob, nob, nloc, n3bbb, resid)
                  do idet = 1, n3bbb
                     a = l3d_excits(1,idet); b = l3d_excits(2,idet); c = l3d_excits(3,idet);
                     i = l3d_excits(4,idet); j = l3d_excits(5,idet); k = l3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); l = l3d_excits(4,jdet);
                        ! compute < ljkdbc | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(d,i,l,a)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); l = l3d_excits(4,jdet);
                        ! compute < ljkdac | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(d,i,l,b)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); l = l3d_excits(4,jdet);
                        ! compute < ljkdab | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(d,i,l,c)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); l = l3d_excits(4,jdet);
                        ! compute < likdbc | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(d,j,l,a)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); l = l3d_excits(4,jdet);
                        ! compute < likdac | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(d,j,l,b)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); l = l3d_excits(4,jdet);
                        ! compute < likdab | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(d,j,l,c)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); l = l3d_excits(4,jdet);
                        ! compute < lijdbc | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(d,k,l,a)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); l = l3d_excits(4,jdet);
                        ! compute < lijdac | h2c(voov) | ijkabc >
                        hmatel = -h2c_voov(d,k,l,b)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = l3d_excits(1,jdet); l = l3d_excits(4,jdet);
                        ! compute < lijdab | h2c(voov) | ijkabc >
                        hmatel = h2c_voov(d,k,l,c)
                        resid(idet) = resid(idet) + hmatel * l3d_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  
                  !!!! diagram 6: A(i/jk)A(a/bc) h2b(eima) * l3c(ebcmjk)
                  ! allocate and copy over t3c arrays
                  allocate(amps_buff(n3abb),excits_buff(6,n3abb))
                  amps_buff(:) = l3c_amps(:)
                  excits_buff(:,:) = l3c_excits(:,:)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = nub*(nub-1)/2*nob*(nob-1)/2
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nub,nub,nob,nob))
                  !!! SB: (2,3,5,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,5,6/), nub, nub, nob, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3d_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3bbb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb
                     a = l3d_excits(1,idet); b = l3d_excits(2,idet); c = l3d_excits(3,idet);
                     i = l3d_excits(4,idet); j = l3d_excits(5,idet); k = l3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < i~j~k~a~b~c~ | h2b(voov) | lj~k~db~c~ >
                        hmatel = h2b_voov(d,i,l,a)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < i~j~k~a~b~c~ | h2b(voov) | lj~k~da~c~ >
                        hmatel = -h2b_voov(d,i,l,b)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < i~j~k~a~b~c~ | h2b(voov) | lj~k~da~b~ >
                        hmatel = h2b_voov(d,i,l,c)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < i~j~k~a~b~c~ | h2b(voov) | li~k~db~c~ >
                        hmatel = -h2b_voov(d,j,l,a)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < i~j~k~a~b~c~ | h2b(voov) | li~k~da~c~ >
                        hmatel = h2b_voov(d,j,l,b)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < i~j~k~a~b~c~ | h2b(voov) | li~k~da~b~ >
                        hmatel = -h2b_voov(d,j,l,c)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < i~j~k~a~b~c~ | h2b(voov) | li~j~db~c~ >
                        hmatel = h2b_voov(d,k,l,a)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < i~j~k~a~b~c~ | h2b(voov) | li~j~da~c~ >
                        hmatel = -h2b_voov(d,k,l,b)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < i~j~k~a~b~c~ | h2b(voov) | li~j~da~b~ >
                        hmatel = h2b_voov(d,k,l,c)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                 end do
                 !$omp end do
                 !$omp end parallel
                 !!!! END OMP PARALLEL SECTION !!!!
                 ! deallocate sorting arrays
                 deallocate(loc_arr,idx_table)
                 ! deallocate l3 buffer arrays
                 deallocate(amps_buff,excits_buff)
           
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3d_excits,&
                  !$omp l1b,l2c,&
                  !$omp H1B_ov,H2C_oovv,H2C_vovv,H2C_ooov,&
                  !$omp X2C_vovv,X2C_ooov,&
                  !$omp nob,nub,n3bbb),&
                  !$omp private(idet,a,b,c,d,i,j,k,l,m,n,e,f,&
                  !$omp res)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb
                      a = l3d_excits(1,idet); b = l3d_excits(2,idet); c = l3d_excits(3,idet);
                      i = l3d_excits(4,idet); j = l3d_excits(5,idet); k = l3d_excits(6,idet);
                      ! A(i/jk)A(a/bc) [l1b(ai) * h2c(jkbc) + h1b(ia) * l2c(bcjk)]
                      res =  l1b(a,i)*h2c_oovv(j,k,b,c) + h1b_ov(i,a)*l2c(b,c,j,k)& ! (1)
                            -l1b(a,j)*h2c_oovv(i,k,b,c) - h1b_ov(j,a)*l2c(b,c,i,k)& ! (ij)
                            -l1b(a,k)*h2c_oovv(j,i,b,c) - h1b_ov(k,a)*l2c(b,c,j,i)& ! (ik)
                            -l1b(b,i)*h2c_oovv(j,k,a,c) - h1b_ov(i,b)*l2c(a,c,j,k)& ! (ab)
                            +l1b(b,j)*h2c_oovv(i,k,a,c) + h1b_ov(j,b)*l2c(a,c,i,k)& ! (ij)(ab)
                            +l1b(b,k)*h2c_oovv(j,i,a,c) + h1b_ov(k,b)*l2c(a,c,j,i)& ! (ik)(ab)
                            -l1b(c,i)*h2c_oovv(j,k,b,a) - h1b_ov(i,c)*l2c(b,a,j,k)& ! (ac)
                            +l1b(c,j)*h2c_oovv(i,k,b,a) + h1b_ov(j,c)*l2c(b,a,i,k)& ! (ij)(ac)
                            +l1b(c,k)*h2c_oovv(j,i,b,a) + h1b_ov(k,c)*l2c(b,a,j,i)  ! (ik)(ac)
                      ! A(c/ab)A(j/ik) [-h2c(ikmc) * l2c(abmj) - h2c(mjab) * x2c(ikmc)]
                      do m = 1, nob
                         res = res&
                               - h2c_oovv(m,j,a,b)*x2c_ooov(i,k,m,c)& ! (1)
                               + h2c_oovv(m,i,a,b)*x2c_ooov(j,k,m,c)& ! (ij)
                               + h2c_oovv(m,k,a,b)*x2c_ooov(i,j,m,c)& ! (jk)
                               + h2c_oovv(m,j,c,b)*x2c_ooov(i,k,m,a)& ! (ac)
                               - h2c_oovv(m,i,c,b)*x2c_ooov(j,k,m,a)& ! (ij)(ac)
                               - h2c_oovv(m,k,c,b)*x2c_ooov(i,j,m,a)& ! (jk)(ac)
                               + h2c_oovv(m,j,a,c)*x2c_ooov(i,k,m,b)& ! (bc)
                               - h2c_oovv(m,i,a,c)*x2c_ooov(j,k,m,b)& ! (ij)(bc)
                               - h2c_oovv(m,k,a,c)*x2c_ooov(i,j,m,b)  ! (jk)(bc)
                         res = res&
                               - l2c(a,b,m,j)*h2c_ooov(i,k,m,c)& ! (1)
                               + l2c(a,b,m,i)*h2c_ooov(j,k,m,c)& ! (ij)
                               + l2c(a,b,m,k)*h2c_ooov(i,j,m,c)& ! (jk)
                               + l2c(c,b,m,j)*h2c_ooov(i,k,m,a)& ! (ac)
                               - l2c(c,b,m,i)*h2c_ooov(j,k,m,a)& ! (ij)(ac)
                               - l2c(c,b,m,k)*h2c_ooov(i,j,m,a)& ! (jk)(ac)
                               + l2c(a,c,m,j)*h2c_ooov(i,k,m,b)& ! (bc)
                               - l2c(a,c,m,i)*h2c_ooov(j,k,m,b)& ! (ij)(bc)
                               - l2c(a,c,m,k)*h2c_ooov(i,j,m,b)  ! (jk)(bc)
                      end do
                      ! A(b/ac)A(k/ij) [h2c_vovv(ekac)*l2c(ebij) + h2c(ijeb)*x2c(ekac)]
                      do e = 1, nub
                         res = res&
                               + h2c_oovv(i,j,e,b)*x2c_vovv(e,k,a,c)& ! (1)
                               - h2c_oovv(k,j,e,b)*x2c_vovv(e,i,a,c)& ! (ik)
                               - h2c_oovv(i,k,e,b)*x2c_vovv(e,j,a,c)& ! (jk)
                               - h2c_oovv(i,j,e,a)*x2c_vovv(e,k,b,c)& ! (ab)
                               + h2c_oovv(k,j,e,a)*x2c_vovv(e,i,b,c)& ! (ik)(ab)
                               + h2c_oovv(i,k,e,a)*x2c_vovv(e,j,b,c)& ! (jk)(ab)
                               - h2c_oovv(i,j,e,c)*x2c_vovv(e,k,a,b)& ! (bc)
                               + h2c_oovv(k,j,e,c)*x2c_vovv(e,i,a,b)& ! (ik)(bc)
                               + h2c_oovv(i,k,e,c)*x2c_vovv(e,j,a,b)  ! (jk)(bc)
                         res = res&
                               + l2c(e,b,i,j)*h2c_vovv(e,k,a,c)& ! (1)
                               - l2c(e,b,k,j)*h2c_vovv(e,i,a,c)& ! (ik)
                               - l2c(e,b,i,k)*h2c_vovv(e,j,a,c)& ! (jk)
                               - l2c(e,a,i,j)*h2c_vovv(e,k,b,c)& ! (ab)
                               + l2c(e,a,k,j)*h2c_vovv(e,i,b,c)& ! (ik)(ab)
                               + l2c(e,a,i,k)*h2c_vovv(e,j,b,c)& ! (jk)(ab)
                               - l2c(e,c,i,j)*h2c_vovv(e,k,a,b)& ! (bc)
                               + l2c(e,c,k,j)*h2c_vovv(e,i,a,b)& ! (ik)(bc)
                               + l2c(e,c,i,k)*h2c_vovv(e,j,a,b)  ! (jk)(bc)
                      end do
                      resid(idet) = resid(idet) + res
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
           
        end subroutine build_LH_3D
        
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! L UPDATE LOOPS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        subroutine update_L1(l1a, l1b, X1A, X1B,&
                             omega,&
                             H1A_oo, H1A_vv, H1B_oo, H1B_vv,&
                             shift,&
                             noa, nua, nob, nub)
           
              integer, intent(in) :: noa, nua, nob, nub
              real(kind=8), intent(in) :: H1A_oo(1:noa,1:noa), H1A_vv(1:nua,1:nua),&
                                          H1B_oo(1:nob,1:nob), H1B_vv(1:nub,1:nub),&
                                          shift, omega

              real(kind=8), intent(inout) :: l1a(1:nua,1:noa)
              !f2py intent(in,out) :: l1a(0:nua-1,0:noa-1)
              real(kind=8), intent(inout) :: l1b(1:nub,1:nob)
              !f2py intent(in,out) :: l1b(0:nub-1,0:nob-1)

              real(kind=8), intent(inout) :: X1A(1:nua,1:noa)
              !f2py intent(in,out) :: X1A(0:nua-1,0:noa-1)
              real(kind=8), intent(inout) :: X1B(1:nub,1:nob)
              !f2py intent(in,out) :: X1B(0:nub-1,0:nob-1)

              integer :: i, a
              real(kind=8) :: denom, val

              do i = 1,noa
                do a = 1,nua
                  denom = H1A_vv(a,a) - H1A_oo(i,i)
                  val = omega*l1a(a,i) - X1A(a,i)
                  l1a(a,i) = l1a(a,i) + val/(denom - omega + shift)
                  X1A(a,i) = val/(denom - omega + shift)
                end do
              end do

              do i = 1,nob
                do a = 1,nub
                  denom = H1B_vv(a,a) - H1B_oo(i,i)
                  val = omega*l1b(a,i) - X1B(a,i)
                  l1b(a,i) = l1b(a,i) + val/(denom - omega + shift)
                  X1B(a,i) = val/(denom - omega + shift)
                end do
              end do

        end subroutine update_L1

        subroutine update_L2(l2a, l2b, l2c, X2A, X2B, X2C,&
                             omega,&
                             H1A_oo, H1A_vv, H1B_oo, H1B_vv,&
                             shift,&
                             noa, nua, nob, nub)
           
              integer, intent(in) :: noa, nua, nob, nub
              real(kind=8), intent(in) :: H1A_oo(1:noa,1:noa), H1A_vv(1:nua,1:nua), &
                                          H1B_oo(1:nob,1:nob), H1B_vv(1:nub,1:nub), shift, omega

              real(kind=8), intent(inout) :: l2a(1:nua,1:nua,1:noa,1:noa)
              !f2py intent(in,out) :: l2a(0:nua-1,0:nua-1,0:noa-1,0:noa-1)
              real(kind=8), intent(inout) :: l2b(1:nua,1:nub,1:noa,1:nob)
              !f2py intent(in,out) :: l2b(0:nua-1,0:nub-1,0:noa-1,0:nob-1)
              real(kind=8), intent(inout) :: l2c(1:nub,1:nub,1:nob,1:nob)
              !f2py intent(in,out) :: l2c(0:nub-1,0:nub-1,0:nob-1,0:nob-1)

              real(kind=8), intent(inout) :: X2A(1:nua,1:nua,1:noa,1:noa)
              !f2py intent(in,out) :: X2A(0:nua-1,0:nua-1,0:noa-1,0:noa-1)
              real(kind=8), intent(inout) :: X2B(1:nua,1:nub,1:noa,1:nob)
              !f2py intent(in,out) :: X2B(0:nua-1,0:nub-1,0:noa-1,0:nob-1)
              real(kind=8), intent(inout) :: X2C(1:nub,1:nub,1:nob,1:nob)
              !f2py intent(in,out) :: X2C(0:nub-1,0:nub-1,0:nob-1,0:nob-1)

              integer :: i, j, a, b
              real(kind=8) :: denom, val

              do i = 1, noa
                do j = i+1, noa
                  do a = 1, nua
                    do b = a+1, nua
                      denom = H1A_vv(a,a) + H1A_vv(b,b) - H1A_oo(i,i) - H1A_oo(j,j)

                      val = omega*l2a(a,b,i,j) - X2A(a,b,i,j)

                      l2a(a,b,i,j) = l2a(a,b,i,j) + val/(denom - omega + shift)
                      l2a(b,a,i,j) = -l2a(a,b,i,j)
                      l2a(a,b,j,i) = -l2a(a,b,i,j)
                      l2a(b,a,j,i) = l2a(a,b,i,j)

                      X2A(a,b,i,j) = val/(denom - omega + shift)
                      X2A(b,a,i,j) = -X2A(a,b,i,j)
                      X2A(a,b,j,i) = -X2A(a,b,i,j)
                      X2A(b,a,j,i) = X2A(a,b,i,j)
                    end do
                  end do
                end do
              end do

              do j = 1, nob
                do i = 1, noa
                  do b = 1, nub
                    do a = 1, nua
                      denom = H1A_vv(a,a) + H1B_vv(b,b) - H1A_oo(i,i) - H1B_oo(j,j)

                      val = omega*l2b(a,b,i,j) - X2B(a,b,i,j)

                      l2b(a,b,i,j) = l2b(a,b,i,j) + val/(denom - omega + shift)
                      X2B(a,b,i,j) = val/(denom - omega + shift)
                    end do
                  end do
                end do
              end do

              do i = 1, nob
                do j = i+1, nob
                  do a = 1, nub
                    do b = a+1, nub
                      denom = H1B_vv(a,a) + H1B_vv(b,b) - H1B_oo(i,i) - H1B_oo(j,j)

                      val = omega*l2c(a,b,i,j) - X2C(a,b,i,j)

                      l2c(a,b,i,j) = l2c(a,b,i,j) + val/(denom - omega + shift)
                      l2c(b,a,i,j) = -l2c(a,b,i,j)
                      l2c(a,b,j,i) = -l2c(a,b,i,j)
                      l2c(b,a,j,i) = l2c(a,b,i,j)

                      X2C(a,b,i,j) = val/(denom - omega + shift)
                      X2C(b,a,i,j) = -X2C(a,b,i,j)
                      X2C(a,b,j,i) = -X2C(a,b,i,j)
                      X2C(b,a,j,i) = X2C(a,b,i,j)
                    end do
                  end do
                end do
              end do

        end subroutine update_L2

        subroutine update_L3(l3a_amps,l3a_excits,&
                             l3b_amps,l3b_excits,&
                             l3c_amps,l3c_excits,&
                             l3d_amps,l3d_excits,&
                             X3A,X3B,X3C,X3D,&
                             omega,&
                             H1A_oo,H1A_vv,H1B_oo,H1B_vv,&
                             shift,&
                             n3aaa,n3aab,n3abb,n3bbb,&
                             noa,nua,nob,nub)
           
              integer, intent(in) :: noa, nua, nob, nub, n3aaa, n3aab, n3abb, n3bbb
              real(kind=8), intent(in) :: H1A_oo(1:noa,1:noa), H1A_vv(1:nua,1:nua), &
                                          H1B_oo(1:nob,1:nob), H1B_vv(1:nub,1:nub), shift, omega
              integer, intent(in) :: l3a_excits(6,n3aaa), l3b_excits(6,n3aab), l3c_excits(6,n3abb), l3d_excits(6,n3bbb)

              real(kind=8), intent(inout) :: l3a_amps(n3aaa)
              !f2py intent(in,out) :: l3a_amps(0:n3aaa-1)
              real(kind=8), intent(inout) :: l3b_amps(n3aab)
              !f2py intent(in,out) :: l3b_amps(0:n3aab-1)
              real(kind=8), intent(inout) :: l3c_amps(n3abb)
              !f2py intent(in,out) :: l3c_amps(0:n3abb-1)
              real(kind=8), intent(inout) :: l3d_amps(n3bbb)
              !f2py intent(in,out) :: l3d_amps(0:n3bbb-1)

              real(kind=8), intent(inout) :: X3A(n3aaa)
              !f2py intent(in,out) :: X3A(0:n3aaa-1)
              real(kind=8), intent(inout) :: X3B(n3aab)
              !f2py intent(in,out) :: X3B(0:n3aab-1)
              real(kind=8), intent(inout) :: X3C(n3abb)
              !f2py intent(in,out) :: X3C(0:n3abb-1)
              real(kind=8), intent(inout) :: X3D(n3bbb)
              !f2py intent(in,out) :: X3D(0:n3bbb-1)

              integer :: i, j, k, a, b, c, idet
              real(kind=8) :: denom, val
              
              do idet = 1, n3aaa
                 a = l3a_excits(1,idet); b = l3a_excits(2,idet); c = l3a_excits(3,idet);
                 i = l3a_excits(4,idet); j = l3a_excits(5,idet); k = l3a_excits(6,idet);
                 
                 denom = -H1A_oo(I,I)-H1A_oo(J,J)-H1A_oo(K,K)+H1A_vv(A,A)+H1A_vv(B,B)+H1A_vv(C,C)
                 val = omega*l3a_amps(idet) - X3A(idet)
                 
                 l3a_amps(idet) = l3a_amps(idet) + val/(denom - omega + shift)
                 val = val/(denom - omega + shift)
                 X3A(idet) = val
              end do
              do idet = 1, n3aab
                 a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                 i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                 
                 denom = -H1A_oo(I,I)-H1A_oo(J,J)-H1B_oo(K,K)+H1A_vv(A,A)+H1A_vv(B,B)+H1B_vv(C,C)
                 val = omega*l3b_amps(idet) - X3B(idet)
                 
                 l3b_amps(idet) = l3b_amps(idet) + val/(denom - omega + shift)
                 val = val/(denom - omega + shift)
                 X3B(idet) = val
              end do
              do idet = 1, n3abb
                 a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                 i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                 
                 denom = -H1A_oo(I,I)-H1B_oo(J,J)-H1B_oo(K,K)+H1A_vv(A,A)+H1B_vv(B,B)+H1B_vv(C,C)
                 val = omega*l3c_amps(idet) - X3C(idet)
                 
                 l3c_amps(idet) = l3c_amps(idet) + val/(denom - omega + shift)
                 val = val/(denom - omega + shift)
                 X3C(idet) = val
              end do
              do idet = 1, n3bbb
                 a = l3d_excits(1,idet); b = l3d_excits(2,idet); c = l3d_excits(3,idet);
                 i = l3d_excits(4,idet); j = l3d_excits(5,idet); k = l3d_excits(6,idet);
                 
                 denom = -H1B_oo(I,I)-H1B_oo(J,J)-H1B_oo(K,K)+H1B_vv(A,A)+H1B_vv(B,B)+H1B_vv(C,C)
                 val = omega*l3d_amps(idet) - X3D(idet)
                 
                 l3d_amps(idet) = l3d_amps(idet) + val/(denom - omega + shift)
                 val = val/(denom - omega + shift)
                 X3D(idet) = val
              end do
           
        end subroutine update_L3

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
              ! 5 possible cases. Always organize so that ordered indices appear first.
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
      ! Sort the 1D array of T3 amplitudes, the 2D array of T3 excitations, and, optionally, the
      ! associated 1D residual array such that triple excitations with the same spatial orbital
      ! indices in the positions indicated by idims are next to one another.
      ! In:
      !   idims: array of 4 integer dimensions along which T3 will be sorted
      !   n1, n2, n3, and n4: no/nu sizes of each dimension in idims
      !   nloc: permutationally unique number of possible (p,q,r,s) tuples
      !   n3p: Number of P-space triples of interest
      ! In,Out:
      !   excits: T3 excitation array (can be aaa, aab, abb, or bbb)
      !   amps: T3 amplitude vector (can be aaa, aab, abb, or bbb)
      !   resid (optional): T3 residual vector (can be aaa, aab, abb, or bbb)
      !   loc_arr: array providing the start- and end-point indices for each sorted block in t3 excitations

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

              ! obtain the lexcial index for each triple excitation in the P space along the sorting dimensions idims
              allocate(temp(n3p),idx(n3p))
              do idet = 1, n3p
                 p = excits(idims(1),idet); q = excits(idims(2),idet); r = excits(idims(3),idet); s = excits(idims(4),idet)
                 temp(idet) = idx_table(p,q,r,s)
              end do
              ! get the sorting array
              call argsort(temp, idx)
              ! apply sorting array to t3 excitations, amplitudes, and, optionally, residual arrays
              excits = excits(:,idx)
              amps = amps(idx)
              if (present(resid)) resid = resid(idx)
              deallocate(temp,idx)
              ! obtain the start- and end-point indices for each lexical index in the sorted t3 excitation and amplitude arrays
              loc_arr(:,1) = 1; loc_arr(:,2) = 0; ! set default start > end so that empty sets do not trigger loops
              do idet = 1, n3p-1
                 ! get consecutive lexcial indices
                 p1 = excits(idims(1),idet);   q1 = excits(idims(2),idet);   r1 = excits(idims(3),idet);   s1 = excits(idims(4),idet)
                 p2 = excits(idims(1),idet+1); q2 = excits(idims(2),idet+1); r2 = excits(idims(3),idet+1); s2 = excits(idims(4),idet+1)
                 pqrs1 = idx_table(p1,q1,r1,s1)
                 pqrs2 = idx_table(p2,q2,r2,s2)
                 ! if change occurs between consecutive indices, record these locations in loc_arr as new start/end points
                 if (pqrs1 /= pqrs2) then
                    loc_arr(pqrs1,2) = idet
                    loc_arr(pqrs2,1) = idet+1
                 end if
              end do
              loc_arr(pqrs2,2) = n3p

      end subroutine sort4

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

              if (n==1) return

              stepsize = 1
              do while (stepsize < n)
                 do left = 1, n-stepsize,stepsize*2
                    i = left
                    j = left+stepsize
                    ksize = min(stepsize*2,n-left+1)
                    k=1

                    do while (i < left+stepsize .and. j < left+ksize)
                       if (r(d(i)) < r(d(j))) then
                          il(k) = d(i)
                          i = i+1
                          k = k+1
                       else
                          il(k) = d(j)
                          j = j+1
                          k = k+1
                       endif
                    enddo

                    if (i < left+stepsize) then
                       ! fill up remaining from left
                       il(k:ksize) = d(i:left+stepsize-1)
                    else
                       ! fill up remaining from right
                       il(k:ksize) = d(j:left+ksize-1)
                    endif
                    d(left:left+ksize-1) = il(1:ksize)
                 end do
                 stepsize = stepsize*2
              end do

      end subroutine argsort

end module leftccsdt_p_loops
