module eomccsdt_p_loops

      use omp_lib

      implicit none
   
      ! Checklist for converting (H*R3)_C update into (X*T3)_C
      ! [ ] - idet loop is over R3 quantities; idet = 1, n3_r
      ! [ ] - jdet loop is over T3 quantities; jdet = loc_arr(1,idx), loc_arr(2,idx)
      ! [ ] - replace n3_r parameter in sort4 to n3_t
      ! [ ] - remove resid from sort4 function when sorting T3 excitations

      contains

               subroutine build_hr_1a(x1a,&
                                      r3a_amps, r3a_excits,&
                                      r3b_amps, r3b_excits,&
                                      r3c_amps, r3c_excits,&
                                      h2a_oovv, h2b_oovv, h2c_oovv,&
                                      n3aaa, n3aab, n3abb,&
                                      noa, nua, nob, nub)

                      integer, intent(in) :: noa, nua, nob, nub, n3aaa, n3aab, n3abb
                      integer, intent(in) :: r3a_excits(6,n3aaa), r3b_excits(6,n3aab), r3c_excits(6,n3abb)
                      real(kind=8), intent(in) :: r3a_amps(n3aaa), r3b_amps(n3aab), r3c_amps(n3abb)
                      real(kind=8), intent(in) :: h2a_oovv(1:noa,1:noa,1:nua,1:nua),&
                                                  h2b_oovv(1:noa,1:nob,1:nua,1:nub),&
                                                  h2c_oovv(1:nob,1:nob,1:nub,1:nub)
                      
                      real(kind=8), intent(inout) :: x1a(1:nua,1:noa)
                      !f2py intent(in,out) :: x1a(0:nua-1,0:noa-1)
                      
                      integer :: i, a, m, n, e, f, idet
                      real(kind=8) :: denom, val, r_amp
                      
                      ! compute < ia | (H(2) * R3)_C | 0 >
                      do idet = 1, n3aaa
                          r_amp = r3a_amps(idet)
                          ! A(a/ef)A(i/mn) h2a(mnef) * r3a(aefimn)
                          a = r3a_excits(1,idet); e = r3a_excits(2,idet); f = r3a_excits(3,idet);
                          i = r3a_excits(4,idet); m = r3a_excits(5,idet); n = r3a_excits(6,idet);
                          x1a(a,i) = x1a(a,i) + h2a_oovv(m,n,e,f) * r_amp ! (1)
                          x1a(e,i) = x1a(e,i) - h2a_oovv(m,n,a,f) * r_amp ! (ae)
                          x1a(f,i) = x1a(f,i) - h2a_oovv(m,n,e,a) * r_amp ! (af)
                          x1a(a,m) = x1a(a,m) - h2a_oovv(i,n,e,f) * r_amp ! (im)
                          x1a(e,m) = x1a(e,m) + h2a_oovv(i,n,a,f) * r_amp ! (ae)(im)
                          x1a(f,m) = x1a(f,m) + h2a_oovv(i,n,e,a) * r_amp ! (af)(im)
                          x1a(a,n) = x1a(a,n) - h2a_oovv(m,i,e,f) * r_amp ! (in)
                          x1a(e,n) = x1a(e,n) + h2a_oovv(m,i,a,f) * r_amp ! (ae)(in)
                          x1a(f,n) = x1a(f,n) + h2a_oovv(m,i,e,a) * r_amp ! (af)(in)
                      end do
                      do idet = 1, n3aab
                          r_amp = r3b_amps(idet)
                          ! A(ae)A(im) h2b(mnef) * r3b(aefimn)
                          a = r3b_excits(1,idet); e = r3b_excits(2,idet); f = r3b_excits(3,idet);
                          i = r3b_excits(4,idet); m = r3b_excits(5,idet); n = r3b_excits(6,idet);
                          x1a(a,i) = x1a(a,i) + h2b_oovv(m,n,e,f) * r_amp ! (1)
                          x1a(e,i) = x1a(e,i) - h2b_oovv(m,n,a,f) * r_amp ! (ae)
                          x1a(a,m) = x1a(a,m) - h2b_oovv(i,n,e,f) * r_amp ! (im)
                          x1a(e,m) = x1a(e,m) + h2b_oovv(i,n,a,f) * r_amp ! (ae)(im)
                      end do
                      do idet = 1, n3abb
                          r_amp = r3c_amps(idet)
                          ! h2c(mnef) * r3c(aefimn)
                          a = r3c_excits(1,idet); e = r3c_excits(2,idet); f = r3c_excits(3,idet);
                          i = r3c_excits(4,idet); m = r3c_excits(5,idet); n = r3c_excits(6,idet);
                          x1a(a,i) = x1a(a,i) + h2c_oovv(m,n,e,f) * r_amp ! (1)
                      end do
              end subroutine build_hr_1a
         
              subroutine build_hr_1b(x1b,&
                                     r3b_amps, r3b_excits,&
                                     r3c_amps, r3c_excits,&
                                     r3d_amps, r3d_excits,&
                                     h2a_oovv, h2b_oovv, h2c_oovv,&
                                     n3aab, n3abb, n3bbb,&
                                     noa, nua, nob, nub)

                      integer, intent(in) :: noa, nua, nob, nub, n3aab, n3abb, n3bbb
                      integer, intent(in) :: r3b_excits(6,n3aab), r3c_excits(6,n3abb), r3d_excits(6,n3bbb)
                      real(kind=8), intent(in) :: r3b_amps(n3aab), r3c_amps(n3abb), r3d_amps(n3bbb)
                      real(kind=8), intent(in) :: h2a_oovv(1:noa,1:noa,1:nua,1:nua),&
                                                  h2b_oovv(1:noa,1:nob,1:nua,1:nub),&
                                                  h2c_oovv(1:nob,1:nob,1:nub,1:nub)

                      real(kind=8), intent(inout) :: x1b(1:nub,1:nob)
                      !f2py intent(in,out) :: x1b(0:nub-1,0:nob-1)
                      
                      integer :: i, a, m, n, e, f, idet
                      real(kind=8) :: denom, val, r_amp
                      
                      ! compute < i~a~ | (H(2) * R3)_C | 0 >
                      do idet = 1, n3aab
                          r_amp = r3b_amps(idet)
                          ! h2a(mnef) * r3b(efamni)
                          e = r3b_excits(1,idet); f = r3b_excits(2,idet); a = r3b_excits(3,idet);
                          m = r3b_excits(4,idet); n = r3b_excits(5,idet); i = r3b_excits(6,idet);
                          x1b(a,i) = x1b(a,i) + h2a_oovv(m,n,e,f) * r_amp ! (1)
                      end do
                      do idet = 1, n3abb
                          r_amp = r3c_amps(idet)
                          ! A(af)A(in) h2b(mnef) * r3c(efamni)
                          e = r3c_excits(1,idet); f = r3c_excits(2,idet); a = r3c_excits(3,idet);
                          m = r3c_excits(4,idet); n = r3c_excits(5,idet); i = r3c_excits(6,idet);
                          x1b(a,i) = x1b(a,i) + h2b_oovv(m,n,e,f) * r_amp ! (1)
                          x1b(f,i) = x1b(f,i) - h2b_oovv(m,n,e,a) * r_amp ! (af)
                          x1b(a,n) = x1b(a,n) - h2b_oovv(m,i,e,f) * r_amp ! (in)
                          x1b(f,n) = x1b(f,n) + h2b_oovv(m,i,e,a) * r_amp ! (af)(in)
                      end do
                      do idet = 1, n3bbb
                          r_amp = r3d_amps(idet)
                          ! A(a/ef)A(i/mn) h2c(mnef) * r3d(aefimn)
                          a = r3d_excits(1,idet); e = r3d_excits(2,idet); f = r3d_excits(3,idet);
                          i = r3d_excits(4,idet); m = r3d_excits(5,idet); n = r3d_excits(6,idet);
                          x1b(a,i) = x1b(a,i) + h2c_oovv(m,n,e,f) * r_amp ! (1)
                          x1b(e,i) = x1b(e,i) - h2c_oovv(m,n,a,f) * r_amp ! (ae)
                          x1b(f,i) = x1b(f,i) - h2c_oovv(m,n,e,a) * r_amp ! (af)
                          x1b(a,m) = x1b(a,m) - h2c_oovv(i,n,e,f) * r_amp ! (im)
                          x1b(e,m) = x1b(e,m) + h2c_oovv(i,n,a,f) * r_amp ! (ae)(im)
                          x1b(f,m) = x1b(f,m) + h2c_oovv(i,n,e,a) * r_amp ! (af)(im)
                          x1b(a,n) = x1b(a,n) - h2c_oovv(m,i,e,f) * r_amp ! (in)
                          x1b(e,n) = x1b(e,n) + h2c_oovv(m,i,a,f) * r_amp ! (ae)(in)
                          x1b(f,n) = x1b(f,n) + h2c_oovv(m,i,e,a) * r_amp ! (af)(in)
                      end do
              end subroutine build_hr_1b

              subroutine build_hr_2a(sigma_2a,&
                                     r3a_amps, r3a_excits,&
                                     r3b_amps, r3b_excits,&
                                     t3a_amps, t3a_excits,&
                                     t3b_amps, t3b_excits,&
                                     h1a_ov, h1b_ov,&
                                     h2a_ooov, h2a_vovv,&
                                     h2b_ooov, h2b_vovv,&
                                     x1a_ov, x1b_ov,&
                                     n3aaa_r, n3aab_r,&
                                     n3aaa_t, n3aab_t,&
                                     noa, nua, nob, nub)
                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa_r, n3aaa_t, n3aab_r, n3aab_t
                  ! Input R and T arrays
                  integer, intent(in) :: r3a_excits(6,n3aaa_r), r3b_excits(6,n3aab_r)
                  integer, intent(in) :: t3a_excits(6,n3aaa_t), t3b_excits(6,n3aab_t) 
                  real(kind=8), intent(in) :: r3a_amps(n3aaa_r), r3b_amps(n3aab_r)
                  real(kind=8), intent(in) :: t3a_amps(n3aaa_t), t3b_amps(n3aab_t)
                  ! Input H and X arrays
                  real(kind=8), intent(in) :: h1a_ov(noa,nua), h1b_ov(nob,nub)
                  real(kind=8), intent(in) :: h2a_ooov(noa,noa,noa,nua)
                  real(kind=8), intent(in) :: h2a_vovv(nua,noa,nua,nua)
                  real(kind=8), intent(in) :: h2b_ooov(noa,nob,noa,nub)
                  real(kind=8), intent(in) :: h2b_vovv(nua,nob,nua,nub)
                  real(kind=8), intent(in) :: x1a_ov(noa,nua), x1b_ov(nob,nub)
                  ! Output and Inout variables
                  real(kind=8), intent(inout) :: sigma_2a(nua,nua,noa,noa)
                  !f2py intent(in,out) :: sigma_2a(0:nua-1,0:nua-1,0:noa-1,0:noa-1)
                  ! Local variables
                  real(kind=8) :: t_amp, r_amp, val
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  ! compute < ijab | (H(2) * T3)_C | 0 >
                  do idet = 1, n3aaa_r
                      r_amp = r3a_amps(idet)

                      ! A(ij)A(ab) [A(m/ij)A(e/ab) h1a(me) * t3a(abeijm)]
                      a = r3a_excits(1,idet); b = r3a_excits(2,idet); e = r3a_excits(3,idet);
                      i = r3a_excits(4,idet); j = r3a_excits(5,idet); m = r3a_excits(6,idet);
                      sigma_2a(a,b,i,j) = sigma_2a(a,b,i,j) + h1a_ov(m,e) * r_amp ! (1)
                      sigma_2a(a,b,m,j) = sigma_2a(a,b,m,j) - h1a_ov(i,e) * r_amp ! (im)
                      sigma_2a(a,b,i,m) = sigma_2a(a,b,i,m) - h1a_ov(j,e) * r_amp ! (jm)
                      sigma_2a(e,b,i,j) = sigma_2a(e,b,i,j) - h1a_ov(m,a) * r_amp ! (ae)
                      sigma_2a(e,b,m,j) = sigma_2a(e,b,m,j) + h1a_ov(i,a) * r_amp ! (im)(ae)
                      sigma_2a(e,b,i,m) = sigma_2a(e,b,i,m) + h1a_ov(j,a) * r_amp ! (jm)(ae)
                      sigma_2a(a,e,i,j) = sigma_2a(a,e,i,j) - h1a_ov(m,b) * r_amp ! (be)
                      sigma_2a(a,e,m,j) = sigma_2a(a,e,m,j) + h1a_ov(i,b) * r_amp ! (im)(be)
                      sigma_2a(a,e,i,m) = sigma_2a(a,e,i,m) + h1a_ov(j,b) * r_amp ! (jm)(be)

                      ! A(ij)A(ab) [A(j/mn)A(f/ab) -h2a(mnif) * t3a(abfmjn)]
                      a = r3a_excits(1,idet); b = r3a_excits(2,idet); f = r3a_excits(3,idet);
                      m = r3a_excits(4,idet); j = r3a_excits(5,idet); n = r3a_excits(6,idet);
                      sigma_2a(a,b,:,j) = sigma_2a(a,b,:,j) - h2a_ooov(m,n,:,f) * r_amp ! (1)
                      sigma_2a(a,b,:,m) = sigma_2a(a,b,:,m) + h2a_ooov(j,n,:,f) * r_amp ! (jm)
                      sigma_2a(a,b,:,n) = sigma_2a(a,b,:,n) + h2a_ooov(m,j,:,f) * r_amp ! (jn)
                      sigma_2a(f,b,:,j) = sigma_2a(f,b,:,j) + h2a_ooov(m,n,:,a) * r_amp ! (af)
                      sigma_2a(f,b,:,m) = sigma_2a(f,b,:,m) - h2a_ooov(j,n,:,a) * r_amp ! (jm)(af)
                      sigma_2a(f,b,:,n) = sigma_2a(f,b,:,n) - h2a_ooov(m,j,:,a) * r_amp ! (jn)(af)
                      sigma_2a(a,f,:,j) = sigma_2a(a,f,:,j) + h2a_ooov(m,n,:,b) * r_amp ! (bf)
                      sigma_2a(a,f,:,m) = sigma_2a(a,f,:,m) - h2a_ooov(j,n,:,b) * r_amp ! (jm)(bf)
                      sigma_2a(a,f,:,n) = sigma_2a(a,f,:,n) - h2a_ooov(m,j,:,b) * r_amp ! (jn)(bf)

                      ! A(ij)A(ab) [A(n/ij)A(b/ef) h2a(anef) * t3a(ebfijn)]
                      e = r3a_excits(1,idet); b = r3a_excits(2,idet); f = r3a_excits(3,idet);
                      i = r3a_excits(4,idet); j = r3a_excits(5,idet); n = r3a_excits(6,idet);
                      sigma_2a(:,b,i,j) = sigma_2a(:,b,i,j) + h2a_vovv(:,n,e,f) * r_amp ! (1)
                      sigma_2a(:,b,n,j) = sigma_2a(:,b,n,j) - h2a_vovv(:,i,e,f) * r_amp ! (in)
                      sigma_2a(:,b,i,n) = sigma_2a(:,b,i,n) - h2a_vovv(:,j,e,f) * r_amp ! (jn)
                      sigma_2a(:,e,i,j) = sigma_2a(:,e,i,j) - h2a_vovv(:,n,b,f) * r_amp ! (be)
                      sigma_2a(:,e,n,j) = sigma_2a(:,e,n,j) + h2a_vovv(:,i,b,f) * r_amp ! (in)(be)
                      sigma_2a(:,e,i,n) = sigma_2a(:,e,i,n) + h2a_vovv(:,j,b,f) * r_amp ! (jn)(be)
                      sigma_2a(:,f,i,j) = sigma_2a(:,f,i,j) - h2a_vovv(:,n,e,b) * r_amp ! (bf)
                      sigma_2a(:,f,n,j) = sigma_2a(:,f,n,j) + h2a_vovv(:,i,e,b) * r_amp ! (in)(bf)
                      sigma_2a(:,f,i,n) = sigma_2a(:,f,i,n) + H2A_vovv(:,j,e,b) * r_amp ! (jn)(bf)
                  end do
                  do idet = 1, n3aab_r
                      r_amp = r3b_amps(idet)

                      ! A(ij)A(ab) [h1b(me) * t3b(abeijm)]
                      a = r3b_excits(1,idet); b = r3b_excits(2,idet); e = r3b_excits(3,idet);
                      i = r3b_excits(4,idet); j = r3b_excits(5,idet); m = r3b_excits(6,idet);
                      sigma_2a(a,b,i,j) = sigma_2a(a,b,i,j) + h1b_ov(m,e) * r_amp ! (1)

                      ! A(ij)A(ab) [A(jm) -h2b(mnif) * t3b(abfmjn)]
                      a = r3b_excits(1,idet); b = r3b_excits(2,idet); f = r3b_excits(3,idet);
                      m = r3b_excits(4,idet); j = r3b_excits(5,idet); n = r3b_excits(6,idet);
                      sigma_2a(a,b,:,j) = sigma_2a(a,b,:,j) - h2b_ooov(m,n,:,f) * r_amp ! (1)
                      sigma_2a(a,b,:,m) = sigma_2a(a,b,:,m) + h2b_ooov(j,n,:,f) * r_amp ! (jm)

                      ! A(ij)A(ab) [A(be) h2b(anef) * t3b(ebfijn)]
                      e = r3b_excits(1,idet); b = r3b_excits(2,idet); f = r3b_excits(3,idet);
                      i = r3b_excits(4,idet); j = r3b_excits(5,idet); n = r3b_excits(6,idet);
                      sigma_2a(:,b,i,j) = sigma_2a(:,b,i,j) + h2b_vovv(:,n,e,f) * r_amp ! (1)
                      sigma_2a(:,e,i,j) = sigma_2a(:,e,i,j) - h2b_vovv(:,n,b,f) * r_amp ! (be)
                  end do
                  do idet = 1, n3aaa_t
                      t_amp = t3a_amps(idet)

                      ! A(ij)A(ab) [A(m/ij)A(e/ab) x1a(me) * t3a(abeijm)]
                      a = t3a_excits(1,idet); b = t3a_excits(2,idet); e = t3a_excits(3,idet);
                      i = t3a_excits(4,idet); j = t3a_excits(5,idet); m = t3a_excits(6,idet);
                      sigma_2a(a,b,i,j) = sigma_2a(a,b,i,j) + x1a_ov(m,e) * t_amp ! (1)
                      sigma_2a(a,b,m,j) = sigma_2a(a,b,m,j) - x1a_ov(i,e) * t_amp ! (im)
                      sigma_2a(a,b,i,m) = sigma_2a(a,b,i,m) - x1a_ov(j,e) * t_amp ! (jm)
                      sigma_2a(e,b,i,j) = sigma_2a(e,b,i,j) - x1a_ov(m,a) * t_amp ! (ae)
                      sigma_2a(e,b,m,j) = sigma_2a(e,b,m,j) + x1a_ov(i,a) * t_amp ! (im)(ae)
                      sigma_2a(e,b,i,m) = sigma_2a(e,b,i,m) + x1a_ov(j,a) * t_amp ! (jm)(ae)
                      sigma_2a(a,e,i,j) = sigma_2a(a,e,i,j) - x1a_ov(m,b) * t_amp ! (be)
                      sigma_2a(a,e,m,j) = sigma_2a(a,e,m,j) + x1a_ov(i,b) * t_amp ! (im)(be)
                      sigma_2a(a,e,i,m) = sigma_2a(a,e,i,m) + x1a_ov(j,b) * t_amp ! (jm)(be)
                  end do
                  do idet = 1, n3aab_t
                      t_amp = t3b_amps(idet)
                      ! A(ij)A(ab) [x1b(me) * t3b(abeijm)]
                      a = t3b_excits(1,idet); b = t3b_excits(2,idet); e = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); j = t3b_excits(5,idet); m = t3b_excits(6,idet);
                      sigma_2a(a,b,i,j) = sigma_2a(a,b,i,j) + x1b_ov(m,e) * t_amp ! (1)
                  end do
                  ! antisymmetrize (this replaces the x2a -= np.transpose(x2a, (...)) stuff in vector update
                  do i = 1, noa
                      do j = i+1, noa
                          do a = 1, nua
                              do b = a+1, nua
                                  val = sigma_2a(b,a,j,i) - sigma_2a(a,b,j,i) - sigma_2a(b,a,i,j) + sigma_2a(a,b,i,j)
                                  sigma_2a(b,a,j,i) =  val
                                  sigma_2a(a,b,j,i) = -val
                                  sigma_2a(b,a,i,j) = -val
                                  sigma_2a(a,b,i,j) =  val
                              end do
                          end do
                      end do
                  end do
                  ! (H(2) * T3)_C terms are vectorized and generally broadcast to diagonal elements, which should
                  ! be 0. Set them to 0 manually (you need to do this).
                  do a = 1, nua
                     sigma_2a(a,a,:,:) = 0.0d0
                  end do
                  do i = 1, noa
                     sigma_2a(:,:,i,i) = 0.0d0
                  end do
 
              end subroutine build_hr_2a
         
              subroutine build_hr_2b(sigma_2b,&
                                     r3b_amps, r3b_excits,&
                                     r3c_amps, r3c_excits,&
                                     t3b_amps, t3b_excits,&
                                     t3c_amps, t3c_excits,&
                                     h1a_ov, h1b_ov,&
                                     h2a_ooov, h2a_vovv,&
                                     h2b_ooov, h2b_vovv, h2b_oovo, h2b_ovvv,&
                                     h2c_ooov, h2c_vovv,&
                                     x1a_ov, x1b_ov,&
                                     n3aab_r, n3abb_r,&
                                     n3aab_t, n3abb_t,&
                                     noa, nua, nob, nub)
                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab_r, n3aab_t, n3abb_r, n3abb_t
                  ! Input R and T arrays
                  integer, intent(in) :: r3b_excits(6,n3aab_r), r3c_excits(6,n3abb_r)
                  integer, intent(in) :: t3b_excits(6,n3aab_t), t3c_excits(6,n3abb_t)
                  real(kind=8), intent(in) :: r3b_amps(n3aab_r), r3c_amps(n3abb_r)
                  real(kind=8), intent(in) :: t3b_amps(n3aab_t), t3c_amps(n3abb_t)
                  ! Input H and X arrays
                  real(kind=8), intent(in) :: h1a_ov(noa,nua), h1b_ov(nob,nub)
                  real(kind=8), intent(in) :: h2a_ooov(noa,noa,noa,nua)
                  real(kind=8), intent(in) :: h2a_vovv(nua,noa,nua,nua)
                  real(kind=8), intent(in) :: h2b_ooov(noa,nob,noa,nub)
                  real(kind=8), intent(in) :: h2b_vovv(nua,nob,nua,nub)
                  real(kind=8), intent(in) :: h2b_oovo(noa,nob,nua,nob)
                  real(kind=8), intent(in) :: h2b_ovvv(noa,nub,nua,nub)
                  real(kind=8), intent(in) :: h2c_ooov(nob,nob,nob,nub)
                  real(kind=8), intent(in) :: h2c_vovv(nub,nob,nub,nub)
                  real(kind=8), intent(in) :: x1a_ov(noa,nua), x1b_ov(nob,nub)
                  ! Output and Inout variables
                  real(kind=8), intent(inout) :: sigma_2b(nua,nub,noa,nob)
                  !f2py intent(in,out) :: sigma_2b(0:nua-1,0:nub-1,0:noa-1,0:nob-1)
                  ! Local variables
                  real(kind=8) :: t_amp, r_amp, val
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet
                  
                  do idet = 1, n3aab_r
                      r_amp = r3b_amps(idet)
                      
                      ! A(af) -h2a(mnif) * r3b(afbmnj)
                      a = r3b_excits(1,idet); f = r3b_excits(2,idet); b = r3b_excits(3,idet);
                      m = r3b_excits(4,idet); n = r3b_excits(5,idet); j = r3b_excits(6,idet);
                      sigma_2b(a,b,:,j) = sigma_2b(a,b,:,j) - h2a_ooov(m,n,:,f) * r_amp ! (1)
                      sigma_2b(f,b,:,j) = sigma_2b(f,b,:,j) + h2a_ooov(m,n,:,a) * r_amp ! (af)

                      ! A(af)A(in) -h2b(nmfj) * r3b(afbinm)
                      a = r3b_excits(1,idet); f = r3b_excits(2,idet); b = r3b_excits(3,idet);
                      i = r3b_excits(4,idet); n = r3b_excits(5,idet); m = r3b_excits(6,idet);
                      sigma_2b(a,b,i,:) = sigma_2b(a,b,i,:) - h2b_oovo(n,m,f,:) * r_amp ! (1)
                      sigma_2b(f,b,i,:) = sigma_2b(f,b,i,:) + h2b_oovo(n,m,a,:) * r_amp ! (af)
                      sigma_2b(a,b,n,:) = sigma_2b(a,b,n,:) + h2b_oovo(i,m,f,:) * r_amp ! (in)
                      sigma_2b(f,b,n,:) = sigma_2b(f,b,n,:) - h2b_oovo(i,m,a,:) * r_amp ! (af)(in)

                      ! A(in) h2a(anef) * r3b(efbinj)
                      e = r3b_excits(1,idet); f = r3b_excits(2,idet); b = r3b_excits(3,idet);
                      i = r3b_excits(4,idet); n = r3b_excits(5,idet); j = r3b_excits(6,idet);
                      sigma_2b(:,b,i,j) = sigma_2b(:,b,i,j) + h2a_vovv(:,n,e,f) * r_amp ! (1)
                      sigma_2b(:,b,n,j) = sigma_2b(:,b,n,j) - h2a_vovv(:,i,e,f) * r_amp ! (in)

                      ! A(af)A(in) h2b(nbfe) * r3b(afeinj)
                      a = r3b_excits(1,idet); f = r3b_excits(2,idet); e = r3b_excits(3,idet);
                      i = r3b_excits(4,idet); n = r3b_excits(5,idet); j = r3b_excits(6,idet);
                      sigma_2b(a,:,i,j) = sigma_2b(a,:,i,j) + h2b_ovvv(n,:,f,e) * r_amp ! (1)
                      sigma_2b(f,:,i,j) = sigma_2b(f,:,i,j) - h2b_ovvv(n,:,a,e) * r_amp ! (af)
                      sigma_2b(a,:,n,j) = sigma_2b(a,:,n,j) - h2b_ovvv(i,:,f,e) * r_amp ! (in)
                      sigma_2b(f,:,n,j) = sigma_2b(f,:,n,j) + h2b_ovvv(i,:,a,e) * r_amp ! (af)(in)

                      ! A(ae)A(im) h1a(me) * r3b(aebimj)
                      a = r3b_excits(1,idet); e = r3b_excits(2,idet); b = r3b_excits(3,idet);
                      i = r3b_excits(4,idet); m = r3b_excits(5,idet); j = r3b_excits(6,idet);
                      sigma_2b(a,b,i,j) = sigma_2b(a,b,i,j) + h1a_ov(m,e) * r_amp ! (1)
                      sigma_2b(a,b,m,j) = sigma_2b(a,b,m,j) - h1a_ov(i,e) * r_amp ! (im)
                      sigma_2b(e,b,i,j) = sigma_2b(e,b,i,j) - h1a_ov(m,a) * r_amp ! (ae)
                      sigma_2b(e,b,m,j) = sigma_2b(e,b,m,j) + h1a_ov(i,a) * r_amp ! (im)(ae)
                  end do
                  do idet = 1, n3aab_t
                      t_amp = t3b_amps(idet)
                      ! A(ae)A(im) x1(me) * t3b(aebimj)
                      a = t3b_excits(1,idet); e = t3b_excits(2,idet); b = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); m = t3b_excits(5,idet); j = t3b_excits(6,idet);
                      sigma_2b(a,b,i,j) = sigma_2b(a,b,i,j) + x1a_ov(m,e) * t_amp ! (1)
                      sigma_2b(a,b,m,j) = sigma_2b(a,b,m,j) - x1a_ov(i,e) * t_amp ! (im)
                      sigma_2b(e,b,i,j) = sigma_2b(e,b,i,j) - x1a_ov(m,a) * t_amp ! (ae)
                      sigma_2b(e,b,m,j) = sigma_2b(e,b,m,j) + x1a_ov(i,a) * t_amp ! (im)(ae)
                  end do
                  do idet = 1, n3abb_r
                      r_amp = r3c_amps(idet)

                      ! A(bf) -h2c(mnjf) * r3c(afbinm)
                      a = r3c_excits(1,idet); f = r3c_excits(2,idet); b = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); n = r3c_excits(5,idet); m = r3c_excits(6,idet);
                      sigma_2b(a,b,i,:) = sigma_2b(a,b,i,:) - h2c_ooov(m,n,:,f) * r_amp ! (1)
                      sigma_2b(a,f,i,:) = sigma_2b(a,f,i,:) + h2c_ooov(m,n,:,b) * r_amp ! (bf)

                      ! A(bf)A(jn) -h2b(mnif) * r3c(afbmnj)
                      a = r3c_excits(1,idet); f = r3c_excits(2,idet); b = r3c_excits(3,idet);
                      m = r3c_excits(4,idet); n = r3c_excits(5,idet); j = r3c_excits(6,idet);
                      sigma_2b(a,b,:,j) = sigma_2b(a,b,:,j) - h2b_ooov(m,n,:,f) * r_amp ! (1)
                      sigma_2b(a,f,:,j) = sigma_2b(a,f,:,j) + h2b_ooov(m,n,:,b) * r_amp ! (bf)
                      sigma_2b(a,b,:,n) = sigma_2b(a,b,:,n) + h2b_ooov(m,j,:,f) * r_amp ! (jn)
                      sigma_2b(a,f,:,n) = sigma_2b(a,f,:,n) - h2b_ooov(m,j,:,b) * r_amp ! (bf)(jn)

                      ! A(jn) h2c(bnef) * r3c(afeinj)
                      a = r3c_excits(1,idet); f = r3c_excits(2,idet); e = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); n = r3c_excits(5,idet); j = r3c_excits(6,idet);
                      sigma_2b(a,:,i,j) = sigma_2b(a,:,i,j) + h2c_vovv(:,n,e,f) * r_amp ! (1)
                      sigma_2b(a,:,i,n) = sigma_2b(a,:,i,n) - h2c_vovv(:,j,e,f) * r_amp ! (jn)

                      ! A(bf)A(jn) h2b(anef) * r3c(efbinj)
                      e = r3c_excits(1,idet); f = r3c_excits(2,idet); b = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); n = r3c_excits(5,idet); j = r3c_excits(6,idet);
                      sigma_2b(:,b,i,j) = sigma_2b(:,b,i,j) + h2b_vovv(:,n,e,f) * r_amp ! (1)
                      sigma_2b(:,f,i,j) = sigma_2b(:,f,i,j) - h2b_vovv(:,n,e,b) * r_amp ! (bf)
                      sigma_2b(:,b,i,n) = sigma_2b(:,b,i,n) - h2b_vovv(:,j,e,f) * r_amp ! (jn)
                      sigma_2b(:,f,i,n) = sigma_2b(:,f,i,n) + h2b_vovv(:,j,e,b) * r_amp ! (bf)(jn)

                      ! [A(be)A(mj) h1b(me) * r3c(aebimj)]
                      a = r3c_excits(1,idet); e = r3c_excits(2,idet); b = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); m = r3c_excits(5,idet); j = r3c_excits(6,idet);
                      sigma_2b(a,b,i,j) = sigma_2b(a,b,i,j) + h1b_ov(m,e) * r_amp ! (1)
                      sigma_2b(a,b,i,m) = sigma_2b(a,b,i,m) - h1b_ov(j,e) * r_amp ! (jm)
                      sigma_2b(a,e,i,j) = sigma_2b(a,e,i,j) - h1b_ov(m,b) * r_amp ! (be)
                      sigma_2b(a,e,i,m) = sigma_2b(a,e,i,m) + h1b_ov(j,b) * r_amp ! (jm)(be)
                  end do
                  do idet = 1, n3abb_t
                      t_amp = t3c_amps(idet)
                      ! [A(be)A(mj) h1b(me) * t3c(aebimj)]
                      a = t3c_excits(1,idet); e = t3c_excits(2,idet); b = t3c_excits(3,idet);
                      i = t3c_excits(4,idet); m = t3c_excits(5,idet); j = t3c_excits(6,idet);
                      sigma_2b(a,b,i,j) = sigma_2b(a,b,i,j) + x1b_ov(m,e) * t_amp ! (1)
                      sigma_2b(a,b,i,m) = sigma_2b(a,b,i,m) - x1b_ov(j,e) * t_amp ! (jm)
                      sigma_2b(a,e,i,j) = sigma_2b(a,e,i,j) - x1b_ov(m,b) * t_amp ! (be)
                      sigma_2b(a,e,i,m) = sigma_2b(a,e,i,m) + x1b_ov(j,b) * t_amp ! (jm)(be)
                  end do
                 
              end subroutine build_hr_2b

              subroutine build_hr_2c(sigma_2c,&
                                     r3c_amps, r3c_excits,&
                                     r3d_amps, r3d_excits,&
                                     t3c_amps, t3c_excits,&
                                     t3d_amps, t3d_excits,&
                                     h1a_ov, h1b_ov,&
                                     h2b_oovo, h2b_ovvv,&
                                     h2c_ooov, h2c_vovv,&
                                     x1a_ov, x1b_ov,&
                                     n3abb_r, n3bbb_r,&
                                     n3abb_t, n3bbb_t,&
                                     noa, nua, nob, nub)
                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3abb_r, n3abb_t, n3bbb_r, n3bbb_t
                  ! Input R and T arrays
                  integer, intent(in) :: r3c_excits(6,n3abb_r), r3d_excits(6,n3bbb_r)
                  integer, intent(in) :: t3c_excits(6,n3abb_t), t3d_excits(6,n3bbb_t) 
                  real(kind=8), intent(in) :: r3c_amps(n3abb_r), r3d_amps(n3bbb_r)
                  real(kind=8), intent(in) :: t3c_amps(n3abb_t), t3d_amps(n3bbb_t)
                  ! Input H and X arrays
                  real(kind=8), intent(in) :: h1a_ov(noa,nua), h1b_ov(nob,nub)
                  real(kind=8), intent(in) :: h2b_oovo(noa,nob,nua,nob)
                  real(kind=8), intent(in) :: h2b_ovvv(noa,nub,nua,nub)
                  real(kind=8), intent(in) :: h2c_ooov(nob,nob,nob,nub)
                  real(kind=8), intent(in) :: h2c_vovv(nub,nob,nub,nub)
                  real(kind=8), intent(in) :: x1a_ov(noa,nua), x1b_ov(nob,nub)
                  ! Output and Inout variables
                  real(kind=8), intent(inout) :: sigma_2c(nub,nub,nob,nob)
                  !f2py intent(in,out) :: sigma_2c(0:nub-1,0:nub-1,0:nob-1,0:nob-1)
                  ! Local variables
                  real(kind=8) :: t_amp, r_amp, val
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  ! compute < ijab | (H(2) * T3)_C | 0 >
                  do idet = 1, n3abb_r
                      r_amp = r3c_amps(idet)

                      ! A(ij)A(ab) [h1a(me) * r3c(eabmij)]
                      e = r3c_excits(1,idet); a = r3c_excits(2,idet); b = r3c_excits(3,idet);
                      m = r3c_excits(4,idet); i = r3c_excits(5,idet); j = r3c_excits(6,idet);
                      sigma_2c(a,b,i,j) = sigma_2c(a,b,i,j) + h1a_ov(m,e) * r_amp ! (1)
            
                      ! A(ij)A(ab) [A(be) h2b(nafe) * r3c(febnij)]
                      f = r3c_excits(1,idet); e = r3c_excits(2,idet); b = r3c_excits(3,idet);
                      n = r3c_excits(4,idet); i = r3c_excits(5,idet); j = r3c_excits(6,idet);
                      sigma_2c(:,b,i,j) = sigma_2c(:,b,i,j) + h2b_ovvv(n,:,f,e) * r_amp ! (1)
                      sigma_2c(:,e,i,j) = sigma_2c(:,e,i,j) - h2b_ovvv(n,:,f,b) * r_amp ! (be)

                      ! A(ij)A(ab) [A(jm) -h2b(nmfi) * r3c(fabnmj)]
                      f = r3c_excits(1,idet); a = r3c_excits(2,idet); b = r3c_excits(3,idet);
                      n = r3c_excits(4,idet); m = r3c_excits(5,idet); j = r3c_excits(6,idet);
                      sigma_2c(a,b,:,j) = sigma_2c(a,b,:,j) - h2b_oovo(n,m,f,:) * r_amp ! (1)
                      sigma_2c(a,b,:,m) = sigma_2c(a,b,:,m) + h2b_oovo(n,j,f,:) * r_amp ! (jm)
                  end do
                  do idet = 1, n3bbb_r
                      r_amp = r3d_amps(idet)

                      ! A(ij)A(ab) [A(m/ij)A(e/ab) h1b(me) * r3d(abeijm)]
                      a = r3d_excits(1,idet); b = r3d_excits(2,idet); e = r3d_excits(3,idet);
                      i = r3d_excits(4,idet); j = r3d_excits(5,idet); m = r3d_excits(6,idet);
                      sigma_2c(a,b,i,j) = sigma_2c(a,b,i,j) + h1b_ov(m,e) * r_amp ! (1)
                      sigma_2c(a,b,m,j) = sigma_2c(a,b,m,j) - h1b_ov(i,e) * r_amp ! (im)
                      sigma_2c(a,b,i,m) = sigma_2c(a,b,i,m) - h1b_ov(j,e) * r_amp ! (jm)
                      sigma_2c(e,b,i,j) = sigma_2c(e,b,i,j) - h1b_ov(m,a) * r_amp ! (ae)
                      sigma_2c(e,b,m,j) = sigma_2c(e,b,m,j) + h1b_ov(i,a) * r_amp ! (im)(ae)
                      sigma_2c(e,b,i,m) = sigma_2c(e,b,i,m) + h1b_ov(j,a) * r_amp ! (jm)(ae)
                      sigma_2c(a,e,i,j) = sigma_2c(a,e,i,j) - h1b_ov(m,b) * r_amp ! (be)
                      sigma_2c(a,e,m,j) = sigma_2c(a,e,m,j) + h1b_ov(i,b) * r_amp ! (im)(be)
                      sigma_2c(a,e,i,m) = sigma_2c(a,e,i,m) + h1b_ov(j,b) * r_amp ! (jm)(be)

                      ! A(ij)A(ab) [A(j/mn)A(f/ab) -h2c(mnif) * r3d(abfmjn)]
                      a = r3d_excits(1,idet); b = r3d_excits(2,idet); f = r3d_excits(3,idet);
                      m = r3d_excits(4,idet); j = r3d_excits(5,idet); n = r3d_excits(6,idet);
                      sigma_2c(a,b,:,j) = sigma_2c(a,b,:,j) - h2c_ooov(m,n,:,f) * r_amp ! (1)
                      sigma_2c(a,b,:,m) = sigma_2c(a,b,:,m) + h2c_ooov(j,n,:,f) * r_amp ! (jm)
                      sigma_2c(a,b,:,n) = sigma_2c(a,b,:,n) + h2c_ooov(m,j,:,f) * r_amp ! (jn)
                      sigma_2c(f,b,:,j) = sigma_2c(f,b,:,j) + h2c_ooov(m,n,:,a) * r_amp ! (af)
                      sigma_2c(f,b,:,m) = sigma_2c(f,b,:,m) - h2c_ooov(j,n,:,a) * r_amp ! (jm)(af)
                      sigma_2c(f,b,:,n) = sigma_2c(f,b,:,n) - h2c_ooov(m,j,:,a) * r_amp ! (jn)(af)
                      sigma_2c(a,f,:,j) = sigma_2c(a,f,:,j) + h2c_ooov(m,n,:,b) * r_amp ! (bf)
                      sigma_2c(a,f,:,m) = sigma_2c(a,f,:,m) - h2c_ooov(j,n,:,b) * r_amp ! (jm)(bf)
                      sigma_2c(a,f,:,n) = sigma_2c(a,f,:,n) - h2c_ooov(m,j,:,b) * r_amp ! (jn)(bf)

                      ! A(ij)A(ab) [A(n/ij)A(b/ef) h2c(anef) * r3d(ebfijn)]
                      e = r3d_excits(1,idet); b = r3d_excits(2,idet); f = r3d_excits(3,idet);
                      i = r3d_excits(4,idet); j = r3d_excits(5,idet); n = r3d_excits(6,idet);
                      sigma_2c(:,b,i,j) = sigma_2c(:,b,i,j) + h2c_vovv(:,n,e,f) * r_amp ! (1)
                      sigma_2c(:,b,n,j) = sigma_2c(:,b,n,j) - h2c_vovv(:,i,e,f) * r_amp ! (in)
                      sigma_2c(:,b,i,n) = sigma_2c(:,b,i,n) - h2c_vovv(:,j,e,f) * r_amp ! (jn)
                      sigma_2c(:,e,i,j) = sigma_2c(:,e,i,j) - h2c_vovv(:,n,b,f) * r_amp ! (be)
                      sigma_2c(:,e,n,j) = sigma_2c(:,e,n,j) + h2c_vovv(:,i,b,f) * r_amp ! (in)(be)
                      sigma_2c(:,e,i,n) = sigma_2c(:,e,i,n) + h2c_vovv(:,j,b,f) * r_amp ! (jn)(be)
                      sigma_2c(:,f,i,j) = sigma_2c(:,f,i,j) - h2c_vovv(:,n,e,b) * r_amp ! (bf)
                      sigma_2c(:,f,n,j) = sigma_2c(:,f,n,j) + h2c_vovv(:,i,e,b) * r_amp ! (in)(bf)
                      sigma_2c(:,f,i,n) = sigma_2c(:,f,i,n) + h2c_vovv(:,j,e,b) * r_amp ! (jn)(bf)
                  end do
                  do idet = 1, n3abb_t
                      t_amp = t3c_amps(idet)

                      ! A(ij)A(ab) [h1a(me) * t3c(eabmij)]
                      e = t3c_excits(1,idet); a = t3c_excits(2,idet); b = t3c_excits(3,idet);
                      m = t3c_excits(4,idet); i = t3c_excits(5,idet); j = t3c_excits(6,idet);
                      sigma_2c(a,b,i,j) = sigma_2c(a,b,i,j) + x1a_ov(m,e) * t_amp ! (1)
                  end do
                  do idet = 1, n3bbb_t
                      t_amp = t3d_amps(idet)

                      ! A(ij)A(ab) [A(m/ij)A(e/ab) x1b(me) * t3d(abeijm)]
                      a = t3d_excits(1,idet); b = t3d_excits(2,idet); e = t3d_excits(3,idet);
                      i = t3d_excits(4,idet); j = t3d_excits(5,idet); m = t3d_excits(6,idet);
                      sigma_2c(a,b,i,j) = sigma_2c(a,b,i,j) + x1b_ov(m,e) * t_amp ! (1)
                      sigma_2c(a,b,m,j) = sigma_2c(a,b,m,j) - x1b_ov(i,e) * t_amp ! (im)
                      sigma_2c(a,b,i,m) = sigma_2c(a,b,i,m) - x1b_ov(j,e) * t_amp ! (jm)
                      sigma_2c(e,b,i,j) = sigma_2c(e,b,i,j) - x1b_ov(m,a) * t_amp ! (ae)
                      sigma_2c(e,b,m,j) = sigma_2c(e,b,m,j) + x1b_ov(i,a) * t_amp ! (im)(ae)
                      sigma_2c(e,b,i,m) = sigma_2c(e,b,i,m) + x1b_ov(j,a) * t_amp ! (jm)(ae)
                      sigma_2c(a,e,i,j) = sigma_2c(a,e,i,j) - x1b_ov(m,b) * t_amp ! (be)
                      sigma_2c(a,e,m,j) = sigma_2c(a,e,m,j) + x1b_ov(i,b) * t_amp ! (im)(be)
                      sigma_2c(a,e,i,m) = sigma_2c(a,e,i,m) + x1b_ov(j,b) * t_amp ! (jm)(be)
                  end do
                  ! antisymmetrize (this replaces the x2c -= np.transpose(x2c, (...)) stuff in vector update
                  do i = 1, nob
                      do j = i+1, nob
                          do a = 1, nub
                              do b = a+1, nub
                                  val = sigma_2c(b,a,j,i) - sigma_2c(a,b,j,i) - sigma_2c(b,a,i,j) + sigma_2c(a,b,i,j)
                                  sigma_2c(b,a,j,i) =  val
                                  sigma_2c(a,b,j,i) = -val
                                  sigma_2c(b,a,i,j) = -val
                                  sigma_2c(a,b,i,j) =  val
                              end do
                          end do
                      end do
                  end do
                  ! (H(2) * T3)_C terms are vectorized and generally broadcast to diagonal elements, which should
                  ! be 0. Set them to 0 manually (you need to do this).
                  do a = 1, nub
                     sigma_2c(a,a,:,:) = 0.0d0
                  end do
                  do i = 1, nob
                     sigma_2c(:,:,i,i) = 0.0d0
                  end do
 
              end subroutine build_hr_2c
         
              subroutine build_hr_3a(resid,&
                                     r2a,&
                                     r3a_amps, r3a_excits,&
                                     r3b_amps, r3b_excits,&
                                     t2a,&
                                     t3a_amps, t3a_excits,&
                                     t3b_amps, t3b_excits,&
                                     h1a_oo, h1a_vv,&
                                     h2a_oooo, h2a_vooo, h2a_oovv,&
                                     h2a_voov, h2a_vvov, h2a_vvvv,&
                                     h2b_voov,&
                                     x1a_oo, x1a_vv,&
                                     x2a_oooo, x2a_vooo, x2a_oovv,&
                                     x2a_voov, x2a_vvov, x2a_vvvv,&
                                     x2b_voov,&
                                     n3aaa_r, n3aab_r,&
                                     n3aaa_t, n3aab_t,&
                                     noa, nua, nob, nub)
                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa_r, n3aaa_t, n3aab_r, n3aab_t
                  ! Input R and T arrays
                  real(kind=8), intent(in) :: r2a(nua,nua,noa,noa), t2a(nua,nua,noa,noa)
                  integer, intent(in) :: r3b_excits(6,n3aab_r), t3b_excits(6,n3aab_t)
                  integer, intent(in) :: t3a_excits(6,n3aaa_t) 
                  real(kind=8), intent(in) :: r3b_amps(n3aab_r), t3b_amps(n3aab_t)
                  real(kind=8), intent(in) :: t3a_amps(n3aaa_t)
                  ! Input H and X arrays
                  real(kind=8), intent(in) :: h1a_oo(noa,noa)
                  real(kind=8), intent(in) :: h1a_vv(nua,nua)
                  real(kind=8), intent(in) :: h2a_oooo(noa,noa,noa,noa)
                  real(kind=8), intent(in) :: h2a_vooo(nua,noa,noa,noa)
                  real(kind=8), intent(in) :: h2a_oovv(noa,noa,nua,nua)
                  !real(kind=8), intent(in) :: h2a_voov(nua,noa,noa,nua)
                  real(kind=8), intent(in) :: h2a_voov(noa,nua,nua,noa)
                  real(kind=8), intent(in) :: h2a_vvov(nua,nua,noa,nua)
                  real(kind=8), intent(in) :: h2a_vvvv(nua,nua,nua,nua)
                  real(kind=8), intent(in) :: h2b_voov(nua,nob,noa,nub)
                  real(kind=8), intent(in) :: x1a_oo(noa,noa)
                  real(kind=8), intent(in) :: x1a_vv(nua,nua)
                  real(kind=8), intent(in) :: x2a_oooo(noa,noa,noa,noa)
                  real(kind=8), intent(in) :: x2a_vooo(nua,noa,noa,noa)
                  real(kind=8), intent(in) :: x2a_oovv(noa,noa,nua,nua)
                  real(kind=8), intent(in) :: x2a_voov(nua,noa,noa,nua)
                  real(kind=8), intent(in) :: x2a_vvov(nua,nua,noa,nua)
                  real(kind=8), intent(in) :: x2a_vvvv(nua,nua,nua,nua)
                  real(kind=8), intent(in) :: x2b_voov(nua,nob,noa,nub)
                  ! Output and Inout variables
                  real(kind=8), intent(out) :: resid(n3aaa_r)
                  integer, intent(inout) :: r3a_excits(6,n3aaa_r)
                  !f2py intent(in,out) :: r3a_excits(6,0:n3aaa_r-1)
                  real(kind=8), intent(inout) :: r3a_amps(n3aaa_r)
                  !f2py intent(in,out) :: r3a_amps(0:n3aaa_r-1)
                  ! Local variables
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:)
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)
                  real(kind=8) :: l_amp, hmatel, hmatel1, res_mm23
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc
                  
                  ! Zero the container that holds H*R
                  resid = 0.0d0
                  !!!! diagram 1a: -A(i/jk) h1a(mi) * r3a(abcmjk)
                  !!!! diagram 3a: 1/2 A(i/jk) h2a(mnij) * r3a(abcmnk)
                  ! NOTE: WITHIN THESE LOOPS, H1A(OO) TERMS ARE DOUBLE-COUNTED SO COMPENSATE BY FACTOR OF 1/2
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)*(nua-2)/6*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,nua,noa))
                  !!! ABCK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/3,noa/), nua, nua, nua, noa)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table, (/1,2,3,6/), nua, nua, nua, noa, nloc, n3aaa_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_oo,H2A_oooo,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        l = r3a_excits(4,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(oooo) | lmkabc >
                        hmatel = h2a_oooo(l,m,i,j)
                        ! compute < ijkabc | h1a(oo) | lmkabc > = -A(ij)A(lm) h1a_oo(l,i) * delta(m,j)
                        hmatel1 = 0.0d0
                        if (m==j) hmatel1 = hmatel1 - h1a_oo(l,i) ! (1)      < ijkabc | h1a(oo) | ljkabc >
                        if (m==i) hmatel1 = hmatel1 + h1a_oo(l,j) ! (ij)     < ijkabc | h1a(oo) | likabc >
                        if (l==j) hmatel1 = hmatel1 + h1a_oo(m,i) ! (lm)     < ijkabc | h1a(oo) | jmkabc >
                        if (l==i) hmatel1 = hmatel1 - h1a_oo(m,j) ! (ij)(lm) < ijkabc | h1a(oo) | imkabc >
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ik)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = r3a_excits(4,jdet); m = r3a_excits(5,jdet);
                           ! compute < ijkabc | h2a(oooo) | lmiabc >
                           hmatel = -h2a_oooo(l,m,k,j)
                           ! compute < ijkabc | h1a(oo) | lmiabc > = A(jk)A(lm) h1a_oo(l,k) * delta(m,j)
                           hmatel1 = 0.0d0
                           if (m==j) hmatel1 = hmatel1 + h1a_oo(l,k) ! (1)      < ijkabc | h1a(oo) | ljiabc >
                           if (m==k) hmatel1 = hmatel1 - h1a_oo(l,j) ! (jk)     < ijkabc | h1a(oo) | lkiabc >
                           if (l==j) hmatel1 = hmatel1 - h1a_oo(m,k) ! (lm)
                           if (l==k) hmatel1 = hmatel1 + h1a_oo(m,j) ! (jk)(lm)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = r3a_excits(4,jdet); m = r3a_excits(5,jdet);
                           ! compute < ijkabc | h2a(oooo) | lmjabc >
                           hmatel = -h2a_oooo(l,m,i,k)
                           ! compute < ijkabc | h1a(oo) | lmjabc > = A(ik)A(lm) h1a_oo(l,i) * delta(m,k)
                           hmatel1 = 0.0d0
                           if (m==k) hmatel1 = hmatel1 + h1a_oo(l,i) ! (1)      < ijkabc | h1a(oo) | lkjabc >
                           if (m==i) hmatel1 = hmatel1 - h1a_oo(l,k) ! (ik)
                           if (l==k) hmatel1 = hmatel1 - h1a_oo(m,i) ! (lm)
                           if (l==i) hmatel1 = hmatel1 + h1a_oo(m,k) ! (ik)(lm)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABCI LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/1,noa-2/), nua, nua, nua, noa)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table, (/1,2,3,4/), nua, nua, nua, noa, nloc, n3aaa_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_oo,H2A_oooo,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,i)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r3a_excits(5,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(oooo) | imnabc >
                        hmatel = h2a_oooo(m,n,j,k)
                        ! compute < ijkabc | h1a(oo) | imnabc > = -A(jk)A(mn) h1a_oo(m,j) * delta(n,k)
                        hmatel1 = 0.0d0
                        if (n==k) hmatel1 = hmatel1 - h1a_oo(m,j)  ! < ijkabc | h1a(oo) | imkabc >
                        if (n==j) hmatel1 = hmatel1 + h1a_oo(m,k)
                        if (m==k) hmatel1 = hmatel1 + h1a_oo(n,j)
                        if (m==j) hmatel1 = hmatel1 - h1a_oo(n,k)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           m = r3a_excits(5,jdet); n = r3a_excits(6,jdet);
                           ! compute < ijkabc | h2a(oooo) | jmnabc >
                           hmatel = -h2a_oooo(m,n,i,k)
                           ! compute < ijkabc | h1a(oo) | jmnabc > = A(ik)A(mn) h1a_oo(m,i) * delta(n,k)
                           hmatel1 = 0.0d0
                           if (n==k) hmatel1 = hmatel1 + h1a_oo(m,i)
                           if (n==i) hmatel1 = hmatel1 - h1a_oo(m,k)
                           if (m==k) hmatel1 = hmatel1 - h1a_oo(n,i)
                           if (m==i) hmatel1 = hmatel1 + h1a_oo(n,k)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           m = r3a_excits(5,jdet); n = r3a_excits(6,jdet);
                           ! compute < ijkabc | h2a(oooo) | kmnabc >
                           hmatel = -h2a_oooo(m,n,j,i)
                           ! compute < ijkabc | h1a(oo) | kmnabc > = A(ij)A(mn) h1a_oo(m,j) * delta(n,i)
                           hmatel1 = 0.0d0
                           if (n==i) hmatel1 = hmatel1 - h1a_oo(m,j)
                           if (n==j) hmatel1 = hmatel1 + h1a_oo(m,i)
                           if (m==i) hmatel1 = hmatel1 + h1a_oo(n,j)
                           if (m==j) hmatel1 = hmatel1 - h1a_oo(n,i)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABCJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/2,noa-1/), nua, nua, nua, noa)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table, (/1,2,3,5/), nua, nua, nua, noa, nloc, n3aaa_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_oo,H2A_oooo,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,j)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        l = r3a_excits(4,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(oooo) | ljnabc >
                        hmatel = h2a_oooo(l,n,i,k)
                        ! compute < ijkabc | h1a(oo) | ljnabc > = -A(ik)A(ln) h1a_oo(l,i) * delta(n,k)
                        hmatel1 = 0.0d0
                        if (n==k) hmatel1 = hmatel1 - h1a_oo(l,i)
                        if (n==i) hmatel1 = hmatel1 + h1a_oo(l,k)
                        if (l==k) hmatel1 = hmatel1 + h1a_oo(n,i)
                        if (l==i) hmatel1 = hmatel1 - h1a_oo(n,k)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = r3a_excits(4,jdet); n = r3a_excits(6,jdet);
                           ! compute < ijkabc | h2a(oooo) | linabc >
                           hmatel = -h2a_oooo(l,n,j,k)
                           ! compute < ijkabc | h1a(oo) | linabc > = A(jk)A(ln) h1a_oo(l,j) * delta(n,k)
                           hmatel1 = 0.0d0
                           if (n==k) hmatel1 = hmatel1 + h1a_oo(l,j)
                           if (n==j) hmatel1 = hmatel1 - h1a_oo(l,k)
                           if (l==k) hmatel1 = hmatel1 - h1a_oo(n,j)
                           if (l==j) hmatel1 = hmatel1 + h1a_oo(n,k)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = r3a_excits(4,jdet); n = r3a_excits(6,jdet);
                           ! compute < ijkabc | h2a(oooo) | lknabc >
                           hmatel = -h2a_oooo(l,n,i,j)
                           ! compute < ijkabc | h1a(oo) | lknabc > = A(ij)A(ln) h1a_oo(l,i) * delta(n,j)
                           hmatel1 = 0.0d0
                           if (n==j) hmatel1 = hmatel1 + h1a_oo(l,i)
                           if (n==i) hmatel1 = hmatel1 - h1a_oo(l,j)
                           if (l==j) hmatel1 = hmatel1 - h1a_oo(n,i)
                           if (l==i) hmatel1 = hmatel1 + h1a_oo(n,j)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  
                  !!!! diagram 2a: A(a/bc) h1a(ae) * r3a(ebcijk)
                  !!!! diagram 4a: 1/2 A(c/ab) h2a(abef) * r3a(ebcijk)
                  ! NOTE: WITHIN THESE LOOPS, H1A(VV) TERMS ARE DOUBLE-COUNTED SO COMPENSATE BY FACTOR OF 1/2
                  ! allocate new sorting arrays
                  nloc = noa*(noa-1)*(noa-2)/6*nua
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(noa,noa,noa,nua))
                  !!! IJKA LOOP !!!
                  call get_index_table(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/1,nua-2/), noa, noa, noa, nua)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table, (/4,5,6,1/), noa, noa, noa, nua, nloc, n3aaa_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_vv,H2A_vvvv,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,a)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); f = r3a_excits(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkaef >
                        !hmatel = h2a_vvvv(b,c,e,f)
                        hmatel = h2a_vvvv(e,f,b,c)
                        ! compute < ijkabc | h1a(vv) | ijkaef > = A(bc)A(ef) h1a_vv(b,e) * delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 + h1a_vv(b,e) ! (1)
                        if (b==f) hmatel1 = hmatel1 - h1a_vv(c,e) ! (bc)
                        if (c==e) hmatel1 = hmatel1 - h1a_vv(b,f) ! (ef)
                        if (b==e) hmatel1 = hmatel1 + h1a_vv(c,f) ! (bc)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); f = r3a_excits(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkbef >
                        !hmatel = -h2a_vvvv(a,c,e,f)
                        hmatel = -h2a_vvvv(e,f,a,c)
                        ! compute < ijkabc | h1a(vv) | ijkbef > = -A(ac)A(ef) h1a_vv(a,e) * delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 - h1a_vv(a,e) ! (1)
                        if (a==f) hmatel1 = hmatel1 + h1a_vv(c,e) ! (ac)
                        if (c==e) hmatel1 = hmatel1 + h1a_vv(a,f) ! (ef)
                        if (a==e) hmatel1 = hmatel1 - h1a_vv(c,f) ! (ac)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); f = r3a_excits(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkcef >
                        !hmatel = -h2a_vvvv(b,a,e,f)
                        hmatel = -h2a_vvvv(e,f,b,a)
                        ! compute < ijkabc | h1a(vv) | ijkcef > = -A(ab)A(ef) h1a_vv(b,e) * delta(a,f)
                        hmatel1 = 0.0d0
                        if (a==f) hmatel1 = hmatel1 - h1a_vv(b,e) ! (1)
                        if (b==f) hmatel1 = hmatel1 + h1a_vv(a,e) ! (ab)
                        if (a==e) hmatel1 = hmatel1 + h1a_vv(b,f) ! (ef)
                        if (b==e) hmatel1 = hmatel1 - h1a_vv(a,f) ! (ab)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! IJKB LOOP !!!
                  call get_index_table(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/2,nua-1/), noa, noa, noa, nua)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table, (/4,5,6,2/), noa, noa, noa, nua, nloc, n3aaa_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_vv,H2A_vvvv,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); f = r3a_excits(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdbf >
                        !hmatel = h2a_vvvv(a,c,d,f)
                        hmatel = h2a_vvvv(d,f,a,c)
                        ! compute < ijkabc | h1a(vv) | ijkdbf > = A(ac)A(df) h1a_vv(a,d) * delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 + h1a_vv(a,d) ! (1)
                        if (a==f) hmatel1 = hmatel1 - h1a_vv(c,d) ! (ac)
                        if (c==d) hmatel1 = hmatel1 - h1a_vv(a,f) ! (df)
                        if (a==d) hmatel1 = hmatel1 + h1a_vv(c,f) ! (ac)(df)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); f = r3a_excits(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdaf >
                        !hmatel = -h2a_vvvv(b,c,d,f)
                        hmatel = -h2a_vvvv(d,f,b,c)
                        ! compute < ijkabc | h1a(vv) | ijkdaf > = -A(bc)A(df) h1a_vv(b,d) * delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 - h1a_vv(b,d) ! (1)
                        if (b==f) hmatel1 = hmatel1 + h1a_vv(c,d) ! (bc)
                        if (c==d) hmatel1 = hmatel1 + h1a_vv(b,f) ! (df)
                        if (b==d) hmatel1 = hmatel1 - h1a_vv(c,f) ! (bc)(df)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); f = r3a_excits(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdcf >
                        !hmatel = -h2a_vvvv(a,b,d,f)
                        hmatel = -h2a_vvvv(d,f,a,b)
                        ! compute < ijkabc | h1a(vv) | ijkdcf > = -A(ab)A(df) h1a_vv(a,d) * delta(b,f)
                        hmatel1 = 0.0d0
                        if (b==f) hmatel1 = hmatel1 - h1a_vv(a,d) ! (1)
                        if (a==f) hmatel1 = hmatel1 + h1a_vv(b,d) ! (ab)
                        if (b==d) hmatel1 = hmatel1 + h1a_vv(a,f) ! (df)
                        if (a==d) hmatel1 = hmatel1 - h1a_vv(b,f) ! (ab)(df)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! IJKC LOOP !!!
                  call get_index_table(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/3,nua/), noa, noa, noa, nua)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table, (/4,5,6,3/), noa, noa, noa, nua, nloc, n3aaa_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_vv,H2A_vvvv,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,c)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); e = r3a_excits(2,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdec >
                        !hmatel = h2a_vvvv(a,b,d,e)
                        hmatel = h2a_vvvv(d,e,a,b)
                        ! compute < ijkabc | h1a(vv) | ijkdec > = A(ab)A(de) h1a_vv(a,d) * delta(b,e)
                        hmatel1 = 0.0d0
                        if (b==e) hmatel1 = hmatel1 + h1a_vv(a,d) ! (1)
                        if (a==e) hmatel1 = hmatel1 - h1a_vv(b,d) ! (ab)
                        if (b==d) hmatel1 = hmatel1 - h1a_vv(a,e) ! (de)
                        if (a==d) hmatel1 = hmatel1 + h1a_vv(b,e) ! (ab)(de)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); e = r3a_excits(2,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdea >
                        !hmatel = -h2a_vvvv(c,b,d,e)
                        hmatel = -h2a_vvvv(d,e,c,b)
                        ! compute < ijkabc | h1a(vv) | ijkdea > = -A(bc)A(de) h1a_vv(c,d) * delta(b,e)
                        hmatel1 = 0.0d0
                        if (b==e) hmatel1 = hmatel1 - h1a_vv(c,d) ! (1)
                        if (c==e) hmatel1 = hmatel1 + h1a_vv(b,d) ! (bc)
                        if (b==d) hmatel1 = hmatel1 + h1a_vv(c,e) ! (de)
                        if (c==d) hmatel1 = hmatel1 - h1a_vv(b,e) ! (bc)(de)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); e = r3a_excits(2,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdeb >
                        !hmatel = -h2a_vvvv(a,c,d,e)
                        hmatel = -h2a_vvvv(d,e,a,c)
                        ! compute < ijkabc | h1a(vv) | ijkdeb > = -A(ac)A(de) h1a_vv(a,d) * delta(c,e)
                        hmatel1 = 0.0d0
                        if (c==e) hmatel1 = hmatel1 - h1a_vv(a,d) ! (1)
                        if (a==e) hmatel1 = hmatel1 + h1a_vv(c,d) ! (ac)
                        if (c==d) hmatel1 = hmatel1 + h1a_vv(a,e) ! (de)
                        if (a==d) hmatel1 = hmatel1 - h1a_vv(c,e) ! (ac)(de)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  
                  !!!! diagram 5a: A(i/jk)A(a/bc) h2a(amie) * r3a(ebcmjk)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = (nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,noa,noa))
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table, (/1,2,4,5/), nua, nua, noa, noa, nloc, n3aaa_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3a_excits(3,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijnabf >
                        !hmatel = h2a_voov(c,n,k,f)
                        hmatel = h2a_voov(n,f,c,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3a_excits(3,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijnbcf >
                        !hmatel = h2a_voov(a,n,k,f)
                        hmatel = h2a_voov(n,f,a,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3a_excits(3,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijnacf >
                        !hmatel = -h2a_voov(b,n,k,f)
                        hmatel = -h2a_voov(n,f,b,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3a_excits(3,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jknabf >
                        !hmatel = h2a_voov(c,n,i,f)
                        hmatel = h2a_voov(n,f,c,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3a_excits(3,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jknbcf >
                        !hmatel = h2a_voov(a,n,i,f)
                        hmatel = h2a_voov(n,f,a,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3a_excits(3,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jknacf >
                        !hmatel = -h2a_voov(b,n,i,f)
                        hmatel = -h2a_voov(n,f,b,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3a_excits(3,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | iknabf >
                        !hmatel = -h2a_voov(c,n,j,f)
                        hmatel = -h2a_voov(n,f,c,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3a_excits(3,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | iknbcf >
                        !hmatel = -h2a_voov(a,n,j,f)
                        hmatel = -h2a_voov(n,f,a,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3a_excits(3,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | iknacf >
                        !hmatel = h2a_voov(b,n,j,f)
                        hmatel = h2a_voov(n,f,b,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table, (/1,3,4,5/), nua, nua, noa, noa, nloc, n3aaa_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijnaec >
                        !hmatel = h2a_voov(b,n,k,e)
                        hmatel = h2a_voov(n,e,b,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijnbec >
                        !hmatel = -h2a_voov(a,n,k,e)
                        hmatel = -h2a_voov(n,e,a,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijnaeb >
                        !hmatel = -h2a_voov(c,n,k,e)
                        hmatel = -h2a_voov(n,e,c,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jknaec >
                        !hmatel = h2a_voov(b,n,i,e)
                        hmatel = h2a_voov(n,e,b,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jknbec >
                        !hmatel = -h2a_voov(a,n,i,e)
                        hmatel = -h2a_voov(n,e,a,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jknaeb >
                        !hmatel = -h2a_voov(c,n,i,e)
                        hmatel = -h2a_voov(n,e,c,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | iknaec >
                        !hmatel = -h2a_voov(b,n,j,e)
                        hmatel = -h2a_voov(n,e,b,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | iknbec >
                        !hmatel = h2a_voov(a,n,j,e)
                        hmatel = h2a_voov(n,e,a,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | iknaeb >
                        !hmatel = h2a_voov(c,n,j,e)
                        hmatel = h2a_voov(n,e,c,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCIJ LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table, (/2,3,4,5/), nua, nua, noa, noa, nloc, n3aaa_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijndbc >
                        !hmatel = h2a_voov(a,n,k,d)
                        hmatel = h2a_voov(n,d,a,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijndac >
                        !hmatel = -h2a_voov(b,n,k,d)
                        hmatel = -h2a_voov(n,d,b,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijndab >
                        !hmatel = h2a_voov(c,n,k,d)
                        hmatel = h2a_voov(n,d,c,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jkndbc >
                        !hmatel = h2a_voov(a,n,i,d)
                        hmatel = h2a_voov(n,d,a,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jkndac >
                        !hmatel = -h2a_voov(b,n,i,d)
                        hmatel = -h2a_voov(n,d,b,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jkndab >
                        !hmatel = h2a_voov(c,n,i,d)
                        hmatel = h2a_voov(n,d,c,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ikndbc >
                        !hmatel = -h2a_voov(a,n,j,d)
                        hmatel = -h2a_voov(n,d,a,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ikndac >
                        !hmatel = h2a_voov(b,n,j,d)
                        hmatel = h2a_voov(n,d,b,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ikndab >
                        !hmatel = -h2a_voov(c,n,j,d)
                        hmatel = -h2a_voov(n,d,c,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table, (/1,2,4,6/), nua, nua, noa, noa, nloc, n3aaa_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3a_excits(3,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkabf >
                        !hmatel = h2a_voov(c,m,j,f)
                        hmatel = h2a_voov(m,f,c,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3a_excits(3,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkbcf >
                        !hmatel = h2a_voov(a,m,j,f)
                        hmatel = h2a_voov(m,f,a,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3a_excits(3,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkacf >
                        !hmatel = -h2a_voov(b,m,j,f)
                        hmatel = -h2a_voov(m,f,b,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3a_excits(3,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkabf >
                        !hmatel = -h2a_voov(c,m,i,f)
                        hmatel = -h2a_voov(m,f,c,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3a_excits(3,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkbcf >
                        !hmatel = -h2a_voov(a,m,i,f)
                        hmatel = -h2a_voov(m,f,a,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3a_excits(3,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkacf >
                        !hmatel = h2a_voov(b,m,i,f)
                        hmatel = h2a_voov(m,f,b,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3a_excits(3,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjabf >
                        !hmatel = -h2a_voov(c,m,k,f)
                        hmatel = -h2a_voov(m,f,c,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3a_excits(3,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjbcf >
                        !hmatel = -h2a_voov(a,m,k,f)
                        hmatel = -h2a_voov(m,f,a,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3a_excits(3,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjacf >
                        !hmatel = h2a_voov(b,m,k,f)
                        hmatel = h2a_voov(m,f,b,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table, (/1,3,4,6/), nua, nua, noa, noa, nloc, n3aaa_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkaec >
                        !hmatel = h2a_voov(b,m,j,e)
                        hmatel = h2a_voov(m,e,b,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkbec >
                        !hmatel = -h2a_voov(a,m,j,e)
                        hmatel = -h2a_voov(m,e,a,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkaeb >
                        !hmatel = -h2a_voov(c,m,j,e)
                        hmatel = -h2a_voov(m,e,c,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkaec >
                        !hmatel = -h2a_voov(b,m,i,e)
                        hmatel = -h2a_voov(m,e,b,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkbec >
                        !hmatel = h2a_voov(a,m,i,e)
                        hmatel = h2a_voov(m,e,a,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkaeb >
                        !hmatel = h2a_voov(c,m,i,e)
                        hmatel = h2a_voov(m,e,c,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjaec >
                        !hmatel = -h2a_voov(b,m,k,e)
                        hmatel = -h2a_voov(m,e,b,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjbec >
                        !hmatel = h2a_voov(a,m,k,e)
                        hmatel = h2a_voov(m,e,a,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjaeb >
                        !hmatel = h2a_voov(c,m,k,e)
                        hmatel = h2a_voov(m,e,c,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table, (/2,3,4,6/), nua, nua, noa, noa, nloc, n3aaa_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkdbc >
                        !hmatel = h2a_voov(a,m,j,d)
                        hmatel = h2a_voov(m,d,a,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkdac >
                        !hmatel = -h2a_voov(b,m,j,d)
                        hmatel = -h2a_voov(m,d,b,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkdab >
                        !hmatel = h2a_voov(c,m,j,d)
                        hmatel = h2a_voov(m,d,c,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkdbc >
                        !hmatel = -h2a_voov(a,m,i,d)
                        hmatel = -h2a_voov(m,d,a,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkdac >
                        !hmatel = h2a_voov(b,m,i,d)
                        hmatel = h2a_voov(m,d,b,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkdab >
                        !hmatel = -h2a_voov(c,m,i,d)
                        hmatel = -h2a_voov(m,d,c,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjdbc >
                        !hmatel = -h2a_voov(a,m,k,d)
                        hmatel = -h2a_voov(m,d,a,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjdac >
                        !hmatel = h2a_voov(b,m,k,d)
                        hmatel = h2a_voov(m,d,b,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjdab >
                        !hmatel = -h2a_voov(c,m,k,d)
                        hmatel = -h2a_voov(m,d,c,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table, (/1,2,5,6/), nua, nua, noa, noa, nloc, n3aaa_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3a_excits(3,jdet); l = r3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkabf >
                        !hmatel = h2a_voov(c,l,i,f)
                        hmatel = h2a_voov(l,f,c,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3a_excits(3,jdet); l = r3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkbcf >
                        !hmatel = h2a_voov(a,l,i,f)
                        hmatel = h2a_voov(l,f,a,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3a_excits(3,jdet); l = r3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkacf >
                        !hmatel = -h2a_voov(b,l,i,f)
                        hmatel = -h2a_voov(l,f,b,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3a_excits(3,jdet); l = r3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likabf >
                        !hmatel = -h2a_voov(c,l,j,f)
                        hmatel = -h2a_voov(l,f,c,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3a_excits(3,jdet); l = r3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likbcf >
                        !hmatel = -h2a_voov(a,l,j,f)
                        hmatel = -h2a_voov(l,f,a,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3a_excits(3,jdet); l = r3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likacf >
                        !hmatel = h2a_voov(b,l,j,f)
                        hmatel = h2a_voov(l,f,b,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3a_excits(3,jdet); l = r3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijabf >
                        !hmatel = h2a_voov(c,l,k,f)
                        hmatel = h2a_voov(l,f,c,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3a_excits(3,jdet); l = r3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijbcf >
                        !hmatel = h2a_voov(a,l,k,f)
                        hmatel = h2a_voov(l,f,a,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3a_excits(3,jdet); l = r3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijacf >
                        !hmatel = -h2a_voov(b,l,k,f)
                        hmatel = -h2a_voov(l,f,b,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table, (/1,3,5,6/), nua, nua, noa, noa, nloc, n3aaa_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); l = r3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkaec >
                        !hmatel = h2a_voov(b,l,i,e)
                        hmatel = h2a_voov(l,e,b,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); l = r3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkbec >
                        !hmatel = -h2a_voov(a,l,i,e)
                        hmatel = -h2a_voov(l,e,a,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); l = r3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkaeb >
                        !hmatel = -h2a_voov(c,l,i,e)
                        hmatel = -h2a_voov(l,e,c,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); l = r3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likaec >
                        !hmatel = -h2a_voov(b,l,j,e)
                        hmatel = -h2a_voov(l,e,b,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); l = r3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likbec >
                        !hmatel = h2a_voov(a,l,j,e)
                        hmatel = h2a_voov(l,e,a,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); l = r3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likaeb >
                        !hmatel = h2a_voov(c,l,j,e)
                        hmatel = h2a_voov(l,e,c,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); l = r3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijaec >
                        !hmatel = h2a_voov(b,l,k,e)
                        hmatel = h2a_voov(l,e,b,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); l = r3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijbec >
                        !hmatel = -h2a_voov(a,l,k,e)
                        hmatel = -h2a_voov(l,e,a,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3a_excits(2,jdet); l = r3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijaeb >
                        !hmatel = -h2a_voov(c,l,k,e)
                        hmatel = -h2a_voov(l,e,c,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table, (/2,3,5,6/), nua, nua, noa, noa, nloc, n3aaa_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); l = r3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkdbc >
                        !hmatel = h2a_voov(a,l,i,d)
                        hmatel = h2a_voov(l,d,a,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); l = r3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkdac >
                        !hmatel = -h2a_voov(b,l,i,d)
                        hmatel = -h2a_voov(l,d,b,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); l = r3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkdab >
                        !hmatel = h2a_voov(c,l,i,d)
                        hmatel = h2a_voov(l,d,c,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); l = r3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likdbc >
                        !hmatel = -h2a_voov(a,l,j,d)
                        hmatel = -h2a_voov(l,d,a,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); l = r3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likdac >
                        !hmatel = h2a_voov(b,l,j,d)
                        hmatel = h2a_voov(l,d,b,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); l = r3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likdab >
                        !hmatel = -h2a_voov(c,l,j,d)
                        hmatel = -h2a_voov(l,d,c,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); l = r3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijdbc >
                        !hmatel = h2a_voov(a,l,k,d)
                        hmatel = h2a_voov(l,d,a,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); l = r3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijdac >
                        !hmatel = -h2a_voov(b,l,k,d)
                        hmatel = -h2a_voov(l,d,b,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3a_excits(1,jdet); l = r3a_excits(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijdab >
                        !hmatel = h2a_voov(c,l,k,d)
                        hmatel = h2a_voov(l,d,c,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 6a: A(i/jk)A(a/bc) h2b(amie) * r3b(abeijm)
                  ! allocate and copy over t3b arrays
                  allocate(amps_buff(n3aab_r),excits_buff(6,n3aab_r))
                  amps_buff(:) = r3b_amps(:)
                  excits_buff(:,:) = r3b_excits(:,:)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = nua*(nua-1)/2*noa*(noa-1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,noa,noa))
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,5/), nua, nua, noa, noa, nloc, n3aab_r)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aaa_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2b(voov) | ijn~abf~ >
                        hmatel = h2b_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2b(voov) | jkn~abf~ >
                        hmatel = h2b_voov(c,n,i,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2b(voov) | ikn~abf~ >
                        hmatel = -h2b_voov(c,n,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2b(voov) | ijn~bcf~ >
                        hmatel = h2b_voov(a,n,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik)(ac)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2b(voov) | jkn~bcf~ >
                        hmatel = h2b_voov(a,n,i,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)(ac)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2b(voov) | ikn~bcf~ >
                        hmatel = -h2b_voov(a,n,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2b(voov) | ijn~acf~ >
                        hmatel = -h2b_voov(b,n,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik)(bc)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2b(voov) | jkn~acf~ >
                        hmatel = -h2b_voov(b,n,i,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)(bc)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2b(voov) | ikn~acf~ >
                        hmatel = h2b_voov(b,n,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do   
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate buffer arrays
                  deallocate(amps_buff,excits_buff) 
                  
                  !!!! diagram 1b: -A(i/jk) x1a(mi) * t3a(abcmjk)
                  !!!! diagram 3b: 1/2 A(i/jk) x2a(mnij) * t3a(abcmnk)
                  ! NOTE: WITHIN THESE LOOPS, X1A(OO) TERMS ARE DOUBLE-COUNTED SO COMPENSATE BY FACTOR OF 1/2
                  ! allocate and initialize the copy of t3a
                  allocate(amps_buff(n3aaa_t))
                  allocate(excits_buff(6,n3aaa_t))
                  amps_buff(:) = t3a_amps(:)
                  excits_buff(:,:) = t3a_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)*(nua-2)/6*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,nua,noa))
                  !!! ABCK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/3,noa/), nua, nua, nua, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,3,6/), nua, nua, nua, noa, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp excits_buff,r3a_excits,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X1A_oo,X2A_oooo,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        l = excits_buff(4,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(oooo) | lmkabc >
                        hmatel = x2a_oooo(l,m,i,j)
                        ! compute < ijkabc | h1a(oo) | lmkabc > = -A(ij)A(lm) x1a_oo(l,i) * delta(m,j)
                        hmatel1 = 0.0d0
                        if (m==j) hmatel1 = hmatel1 - x1a_oo(l,i) ! (1)      < ijkabc | h1a(oo) | ljkabc >
                        if (m==i) hmatel1 = hmatel1 + x1a_oo(l,j) ! (ij)     < ijkabc | h1a(oo) | likabc >
                        if (l==j) hmatel1 = hmatel1 + x1a_oo(m,i) ! (lm)     < ijkabc | h1a(oo) | jmkabc >
                        if (l==i) hmatel1 = hmatel1 - x1a_oo(m,j) ! (ij)(lm) < ijkabc | h1a(oo) | imkabc >
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ik)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = excits_buff(4,jdet); m = excits_buff(5,jdet);
                           ! compute < ijkabc | h2a(oooo) | lmiabc >
                           hmatel = -x2a_oooo(l,m,k,j)
                           ! compute < ijkabc | h1a(oo) | lmiabc > = A(jk)A(lm) x1a_oo(l,k) * delta(m,j)
                           hmatel1 = 0.0d0
                           if (m==j) hmatel1 = hmatel1 + x1a_oo(l,k) ! (1)      < ijkabc | h1a(oo) | ljiabc >
                           if (m==k) hmatel1 = hmatel1 - x1a_oo(l,j) ! (jk)     < ijkabc | h1a(oo) | lkiabc >
                           if (l==j) hmatel1 = hmatel1 - x1a_oo(m,k) ! (lm)
                           if (l==k) hmatel1 = hmatel1 + x1a_oo(m,j) ! (jk)(lm)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = excits_buff(4,jdet); m = excits_buff(5,jdet);
                           ! compute < ijkabc | h2a(oooo) | lmjabc >
                           hmatel = -x2a_oooo(l,m,i,k)
                           ! compute < ijkabc | h1a(oo) | lmjabc > = A(ik)A(lm) x1a_oo(l,i) * delta(m,k)
                           hmatel1 = 0.0d0
                           if (m==k) hmatel1 = hmatel1 + x1a_oo(l,i) ! (1)      < ijkabc | h1a(oo) | lkjabc >
                           if (m==i) hmatel1 = hmatel1 - x1a_oo(l,k) ! (ik)
                           if (l==k) hmatel1 = hmatel1 - x1a_oo(m,i) ! (lm)
                           if (l==i) hmatel1 = hmatel1 + x1a_oo(m,k) ! (ik)(lm)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABCI LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/1,noa-2/), nua, nua, nua, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,3,4/), nua, nua, nua, noa, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp excits_buff,r3a_excits,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X1A_oo,X2A_oooo,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,i)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = excits_buff(5,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(oooo) | imnabc >
                        hmatel = x2a_oooo(m,n,j,k)
                        ! compute < ijkabc | h1a(oo) | imnabc > = -A(jk)A(mn) x1a_oo(m,j) * delta(n,k)
                        hmatel1 = 0.0d0
                        if (n==k) hmatel1 = hmatel1 - x1a_oo(m,j)  ! < ijkabc | h1a(oo) | imkabc >
                        if (n==j) hmatel1 = hmatel1 + x1a_oo(m,k)
                        if (m==k) hmatel1 = hmatel1 + x1a_oo(n,j)
                        if (m==j) hmatel1 = hmatel1 - x1a_oo(n,k)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           m = excits_buff(5,jdet); n = excits_buff(6,jdet);
                           ! compute < ijkabc | h2a(oooo) | jmnabc >
                           hmatel = -x2a_oooo(m,n,i,k)
                           ! compute < ijkabc | h1a(oo) | jmnabc > = A(ik)A(mn) x1a_oo(m,i) * delta(n,k)
                           hmatel1 = 0.0d0
                           if (n==k) hmatel1 = hmatel1 + x1a_oo(m,i)
                           if (n==i) hmatel1 = hmatel1 - x1a_oo(m,k)
                           if (m==k) hmatel1 = hmatel1 - x1a_oo(n,i)
                           if (m==i) hmatel1 = hmatel1 + x1a_oo(n,k)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           m = excits_buff(5,jdet); n = excits_buff(6,jdet);
                           ! compute < ijkabc | h2a(oooo) | kmnabc >
                           hmatel = -x2a_oooo(m,n,j,i)
                           ! compute < ijkabc | h1a(oo) | kmnabc > = A(ij)A(mn) x1a_oo(m,j) * delta(n,i)
                           hmatel1 = 0.0d0
                           if (n==i) hmatel1 = hmatel1 - x1a_oo(m,j)
                           if (n==j) hmatel1 = hmatel1 + x1a_oo(m,i)
                           if (m==i) hmatel1 = hmatel1 + x1a_oo(n,j)
                           if (m==j) hmatel1 = hmatel1 - x1a_oo(n,i)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABCJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/2,noa-1/), nua, nua, nua, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,3,5/), nua, nua, nua, noa, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp excits_buff,r3a_excits,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X1A_oo,X2A_oooo,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,j)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        l = excits_buff(4,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(oooo) | ljnabc >
                        hmatel = x2a_oooo(l,n,i,k)
                        ! compute < ijkabc | h1a(oo) | ljnabc > = -A(ik)A(ln) x1a_oo(l,i) * delta(n,k)
                        hmatel1 = 0.0d0
                        if (n==k) hmatel1 = hmatel1 - x1a_oo(l,i)
                        if (n==i) hmatel1 = hmatel1 + x1a_oo(l,k)
                        if (l==k) hmatel1 = hmatel1 + x1a_oo(n,i)
                        if (l==i) hmatel1 = hmatel1 - x1a_oo(n,k)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = excits_buff(4,jdet); n = excits_buff(6,jdet);
                           ! compute < ijkabc | h2a(oooo) | linabc >
                           hmatel = -x2a_oooo(l,n,j,k)
                           ! compute < ijkabc | h1a(oo) | linabc > = A(jk)A(ln) x1a_oo(l,j) * delta(n,k)
                           hmatel1 = 0.0d0
                           if (n==k) hmatel1 = hmatel1 + x1a_oo(l,j)
                           if (n==j) hmatel1 = hmatel1 - x1a_oo(l,k)
                           if (l==k) hmatel1 = hmatel1 - x1a_oo(n,j)
                           if (l==j) hmatel1 = hmatel1 + x1a_oo(n,k)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = excits_buff(4,jdet); n = excits_buff(6,jdet);
                           ! compute < ijkabc | h2a(oooo) | lknabc >
                           hmatel = -x2a_oooo(l,n,i,j)
                           ! compute < ijkabc | h1a(oo) | lknabc > = A(ij)A(ln) x1a_oo(l,i) * delta(n,j)
                           hmatel1 = 0.0d0
                           if (n==j) hmatel1 = hmatel1 + x1a_oo(l,i)
                           if (n==i) hmatel1 = hmatel1 - x1a_oo(l,j)
                           if (l==j) hmatel1 = hmatel1 - x1a_oo(n,i)
                           if (l==i) hmatel1 = hmatel1 + x1a_oo(n,j)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate temporary amplitude arrays
                  !deallocate(excits_buff,amps_buff)
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 2b: A(a/bc) x1a(ae) * t3a(ebcijk)
                  !!!! diagram 4b: 1/2 A(c/ab) x2a(abef) * t3a(ebcijk)
                  ! NOTE: WITHIN THESE LOOPS, X1A(VV) TERMS ARE DOUBLE-COUNTED SO COMPENSATE BY FACTOR OF 1/2
                  ! allocate and initialize the copy of t3a
                  !allocate(amps_buff(n3aaa_t))
                  !allocate(excits_buff(6,n3aaa_t))
                  !amps_buff(:) = t3a_amps(:)
                  !excits_buff(:,:) = t3a_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = noa*(noa-1)*(noa-2)/6*nua
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(noa,noa,noa,nua))
                  !!! IJKA LOOP !!!
                  call get_index_table(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/1,nua-2/), noa, noa, noa, nua)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,1/), noa, noa, noa, nua, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp excits_buff,r3a_excits,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X1A_vv,X2A_vvvv,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,a)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); f = excits_buff(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkaef >
                        !hmatel = x2a_vvvv(b,c,e,f)
                        hmatel = x2a_vvvv(e,f,b,c)
                        ! compute < ijkabc | h1a(vv) | ijkaef > = A(bc)A(ef) x1a_vv(b,e) * delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 + x1a_vv(b,e) ! (1)
                        if (b==f) hmatel1 = hmatel1 - x1a_vv(c,e) ! (bc)
                        if (c==e) hmatel1 = hmatel1 - x1a_vv(b,f) ! (ef)
                        if (b==e) hmatel1 = hmatel1 + x1a_vv(c,f) ! (bc)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); f = excits_buff(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkbef >
                        !hmatel = -x2a_vvvv(a,c,e,f)
                        hmatel = -x2a_vvvv(e,f,a,c)
                        ! compute < ijkabc | h1a(vv) | ijkbef > = -A(ac)A(ef) x1a_vv(a,e) * delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 - x1a_vv(a,e) ! (1)
                        if (a==f) hmatel1 = hmatel1 + x1a_vv(c,e) ! (ac)
                        if (c==e) hmatel1 = hmatel1 + x1a_vv(a,f) ! (ef)
                        if (a==e) hmatel1 = hmatel1 - x1a_vv(c,f) ! (ac)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); f = excits_buff(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkcef >
                        !hmatel = -x2a_vvvv(b,a,e,f)
                        hmatel = -x2a_vvvv(e,f,b,a)
                        ! compute < ijkabc | h1a(vv) | ijkcef > = -A(ab)A(ef) x1a_vv(b,e) * delta(a,f)
                        hmatel1 = 0.0d0
                        if (a==f) hmatel1 = hmatel1 - x1a_vv(b,e) ! (1)
                        if (b==f) hmatel1 = hmatel1 + x1a_vv(a,e) ! (ab)
                        if (a==e) hmatel1 = hmatel1 + x1a_vv(b,f) ! (ef)
                        if (b==e) hmatel1 = hmatel1 - x1a_vv(a,f) ! (ab)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! IJKB LOOP !!!
                  call get_index_table(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/2,nua-1/), noa, noa, noa, nua)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,2/), noa, noa, noa, nua, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp excits_buff,r3a_excits,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X1A_vv,X2A_vvvv,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); f = excits_buff(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdbf >
                        !hmatel = x2a_vvvv(a,c,d,f)
                        hmatel = x2a_vvvv(d,f,a,c)
                        ! compute < ijkabc | h1a(vv) | ijkdbf > = A(ac)A(df) x1a_vv(a,d) * delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 + x1a_vv(a,d) ! (1)
                        if (a==f) hmatel1 = hmatel1 - x1a_vv(c,d) ! (ac)
                        if (c==d) hmatel1 = hmatel1 - x1a_vv(a,f) ! (df)
                        if (a==d) hmatel1 = hmatel1 + x1a_vv(c,f) ! (ac)(df)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); f = excits_buff(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdaf >
                        !hmatel = -x2a_vvvv(b,c,d,f)
                        hmatel = -x2a_vvvv(d,f,b,c)
                        ! compute < ijkabc | h1a(vv) | ijkdaf > = -A(bc)A(df) x1a_vv(b,d) * delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 - x1a_vv(b,d) ! (1)
                        if (b==f) hmatel1 = hmatel1 + x1a_vv(c,d) ! (bc)
                        if (c==d) hmatel1 = hmatel1 + x1a_vv(b,f) ! (df)
                        if (b==d) hmatel1 = hmatel1 - x1a_vv(c,f) ! (bc)(df)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); f = excits_buff(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdcf >
                        !hmatel = -x2a_vvvv(a,b,d,f)
                        hmatel = -x2a_vvvv(d,f,a,b)
                        ! compute < ijkabc | h1a(vv) | ijkdcf > = -A(ab)A(df) x1a_vv(a,d) * delta(b,f)
                        hmatel1 = 0.0d0
                        if (b==f) hmatel1 = hmatel1 - x1a_vv(a,d) ! (1)
                        if (a==f) hmatel1 = hmatel1 + x1a_vv(b,d) ! (ab)
                        if (b==d) hmatel1 = hmatel1 + x1a_vv(a,f) ! (df)
                        if (a==d) hmatel1 = hmatel1 - x1a_vv(b,f) ! (ab)(df)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! IJKC LOOP !!!
                  call get_index_table(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/3,nua/), noa, noa, noa, nua)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,3/), noa, noa, noa, nua, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp excits_buff,r3a_excits,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X1A_vv,X2A_vvvv,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,c)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); e = excits_buff(2,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdec >
                        !hmatel = x2a_vvvv(a,b,d,e)
                        hmatel = x2a_vvvv(d,e,a,b)
                        ! compute < ijkabc | h1a(vv) | ijkdec > = A(ab)A(de) x1a_vv(a,d) * delta(b,e)
                        hmatel1 = 0.0d0
                        if (b==e) hmatel1 = hmatel1 + x1a_vv(a,d) ! (1)
                        if (a==e) hmatel1 = hmatel1 - x1a_vv(b,d) ! (ab)
                        if (b==d) hmatel1 = hmatel1 - x1a_vv(a,e) ! (de)
                        if (a==d) hmatel1 = hmatel1 + x1a_vv(b,e) ! (ab)(de)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); e = excits_buff(2,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdea >
                        !hmatel = -x2a_vvvv(c,b,d,e)
                        hmatel = -x2a_vvvv(d,e,c,b)
                        ! compute < ijkabc | h1a(vv) | ijkdea > = -A(bc)A(de) x1a_vv(c,d) * delta(b,e)
                        hmatel1 = 0.0d0
                        if (b==e) hmatel1 = hmatel1 - x1a_vv(c,d) ! (1)
                        if (c==e) hmatel1 = hmatel1 + x1a_vv(b,d) ! (bc)
                        if (b==d) hmatel1 = hmatel1 + x1a_vv(c,e) ! (de)
                        if (c==d) hmatel1 = hmatel1 - x1a_vv(b,e) ! (bc)(de)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); e = excits_buff(2,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdeb >
                        !hmatel = -x2a_vvvv(a,c,d,e)
                        hmatel = -x2a_vvvv(d,e,a,c)
                        ! compute < ijkabc | h1a(vv) | ijkdeb > = -A(ac)A(de) x1a_vv(a,d) * delta(c,e)
                        hmatel1 = 0.0d0
                        if (c==e) hmatel1 = hmatel1 - x1a_vv(a,d) ! (1)
                        if (a==e) hmatel1 = hmatel1 + x1a_vv(c,d) ! (ac)
                        if (c==d) hmatel1 = hmatel1 + x1a_vv(a,e) ! (de)
                        if (a==d) hmatel1 = hmatel1 - x1a_vv(c,e) ! (ac)(de)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate temporary amplitude arrays
                  !deallocate(excits_buff,amps_buff)
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 5b: A(i/jk)A(a/bc) x2a(amie) * t3a(ebcmjk)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = (nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,noa,noa))
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,5/), nua, nua, noa, noa, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2A_voov,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijnabf >
                        hmatel = x2a_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijnbcf >
                        hmatel = x2a_voov(a,n,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijnacf >
                        hmatel = -x2a_voov(b,n,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jknabf >
                        hmatel = x2a_voov(c,n,i,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jknbcf >
                        hmatel = x2a_voov(a,n,i,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jknacf >
                        hmatel = -x2a_voov(b,n,i,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | iknabf >
                        hmatel = -x2a_voov(c,n,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | iknbcf >
                        hmatel = -x2a_voov(a,n,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | iknacf >
                        hmatel = x2a_voov(b,n,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,5/), nua, nua, noa, noa, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2A_voov,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijnaec >
                        hmatel = x2a_voov(b,n,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijnbec >
                        hmatel = -x2a_voov(a,n,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijnaeb >
                        hmatel = -x2a_voov(c,n,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jknaec >
                        hmatel = x2a_voov(b,n,i,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jknbec >
                        hmatel = -x2a_voov(a,n,i,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jknaeb >
                        hmatel = -x2a_voov(c,n,i,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | iknaec >
                        hmatel = -x2a_voov(b,n,j,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | iknbec >
                        hmatel = x2a_voov(a,n,j,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | iknaeb >
                        hmatel = x2a_voov(c,n,j,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCIJ LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,5/), nua, nua, noa, noa, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2A_voov,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijndbc >
                        hmatel = x2a_voov(a,n,k,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijndac >
                        hmatel = -x2a_voov(b,n,k,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijndab >
                        hmatel = x2a_voov(c,n,k,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jkndbc >
                        hmatel = x2a_voov(a,n,i,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jkndac >
                        hmatel = -x2a_voov(b,n,i,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jkndab >
                        hmatel = x2a_voov(c,n,i,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ikndbc >
                        hmatel = -x2a_voov(a,n,j,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ikndac >
                        hmatel = x2a_voov(b,n,j,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ikndab >
                        hmatel = -x2a_voov(c,n,j,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,6/), nua, nua, noa, noa, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2A_voov,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkabf >
                        hmatel = x2a_voov(c,m,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkbcf >
                        hmatel = x2a_voov(a,m,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkacf >
                        hmatel = -x2a_voov(b,m,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkabf >
                        hmatel = -x2a_voov(c,m,i,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkbcf >
                        hmatel = -x2a_voov(a,m,i,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkacf >
                        hmatel = x2a_voov(b,m,i,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjabf >
                        hmatel = -x2a_voov(c,m,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjbcf >
                        hmatel = -x2a_voov(a,m,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjacf >
                        hmatel = x2a_voov(b,m,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,6/), nua, nua, noa, noa, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2A_voov,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkaec >
                        hmatel = x2a_voov(b,m,j,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkbec >
                        hmatel = -x2a_voov(a,m,j,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkaeb >
                        hmatel = -x2a_voov(c,m,j,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkaec >
                        hmatel = -x2a_voov(b,m,i,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkbec >
                        hmatel = x2a_voov(a,m,i,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkaeb >
                        hmatel = x2a_voov(c,m,i,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjaec >
                        hmatel = -x2a_voov(b,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjbec >
                        hmatel = x2a_voov(a,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjaeb >
                        hmatel = x2a_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,6/), nua, nua, noa, noa, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2A_voov,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkdbc >
                        hmatel = x2a_voov(a,m,j,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkdac >
                        hmatel = -x2a_voov(b,m,j,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkdab >
                        hmatel = x2a_voov(c,m,j,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkdbc >
                        hmatel = -x2a_voov(a,m,i,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkdac >
                        hmatel = x2a_voov(b,m,i,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkdab >
                        hmatel = -x2a_voov(c,m,i,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjdbc >
                        hmatel = -x2a_voov(a,m,k,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjdac >
                        hmatel = x2a_voov(b,m,k,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjdab >
                        hmatel = -x2a_voov(c,m,k,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,5,6/), nua, nua, noa, noa, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2A_voov,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkabf >
                        hmatel = x2a_voov(c,l,i,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkbcf >
                        hmatel = x2a_voov(a,l,i,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkacf >
                        hmatel = -x2a_voov(b,l,i,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likabf >
                        hmatel = -x2a_voov(c,l,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likbcf >
                        hmatel = -x2a_voov(a,l,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likacf >
                        hmatel = x2a_voov(b,l,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijabf >
                        hmatel = x2a_voov(c,l,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijbcf >
                        hmatel = x2a_voov(a,l,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijacf >
                        hmatel = -x2a_voov(b,l,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,5,6/), nua, nua, noa, noa, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2A_voov,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkaec >
                        hmatel = x2a_voov(b,l,i,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkbec >
                        hmatel = -x2a_voov(a,l,i,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkaeb >
                        hmatel = -x2a_voov(c,l,i,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likaec >
                        hmatel = -x2a_voov(b,l,j,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likbec >
                        hmatel = x2a_voov(a,l,j,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likaeb >
                        hmatel = x2a_voov(c,l,j,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijaec >
                        hmatel = x2a_voov(b,l,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijbec >
                        hmatel = -x2a_voov(a,l,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijaeb >
                        hmatel = -x2a_voov(c,l,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,5,6/), nua, nua, noa, noa, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2A_voov,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkdbc >
                        hmatel = x2a_voov(a,l,i,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkdac >
                        hmatel = -x2a_voov(b,l,i,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkdab >
                        hmatel = x2a_voov(c,l,i,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likdbc >
                        hmatel = -x2a_voov(a,l,j,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likdac >
                        hmatel = x2a_voov(b,l,j,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likdab >
                        hmatel = -x2a_voov(c,l,j,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijdbc >
                        hmatel = x2a_voov(a,l,k,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijdac >
                        hmatel = -x2a_voov(b,l,k,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijdab >
                        hmatel = x2a_voov(c,l,k,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate temporary amplitude arrays
                  deallocate(excits_buff,amps_buff)
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 6b: A(i/jk)A(a/bc) x2b(amie) * t3b(abeijm)
                  ! allocate and copy over t3b arrays
                  allocate(amps_buff(n3aab_t),excits_buff(6,n3aab_t))
                  amps_buff(:) = t3b_amps(:)
                  excits_buff(:,:) = t3b_excits(:,:)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = nua*(nua-1)/2*noa*(noa-1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,noa,noa))
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,5/), nua, nua, noa, noa, nloc, n3aab_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2B_voov,&
                  !$omp noa,nua,nob,nub,n3aaa_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2b(voov) | ijn~abf~ >
                        hmatel = x2b_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2b(voov) | jkn~abf~ >
                        hmatel = x2b_voov(c,n,i,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2b(voov) | ikn~abf~ >
                        hmatel = -x2b_voov(c,n,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2b(voov) | ijn~bcf~ >
                        hmatel = x2b_voov(a,n,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik)(ac)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2b(voov) | jkn~bcf~ >
                        hmatel = x2b_voov(a,n,i,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)(ac)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2b(voov) | ikn~bcf~ >
                        hmatel = -x2b_voov(a,n,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2b(voov) | ijn~acf~ >
                        hmatel = -x2b_voov(b,n,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik)(bc)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2b(voov) | jkn~acf~ >
                        hmatel = -x2b_voov(b,n,i,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)(bc)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2b(voov) | ikn~acf~ >
                        hmatel = x2b_voov(b,n,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do   
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate buffer arrays
                  deallocate(amps_buff,excits_buff) 

                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r2a,t2a,&
                  !$omp H2A_vvov,H2A_vooo,&
                  !$omp X2A_vvov,X2A_vooo,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(idet,a,b,c,d,i,j,k,l,m,n,e,f,&
                  !$omp res_mm23)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                      a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                      i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);

                      res_mm23 = 0.0d0
                      do e = 1, nua
                           ! A(i/jk)(c/ab) h2a(abie) * r2a(ecjk)
                          res_mm23 = res_mm23 + h2a_vvov(a,b,i,e) * r2a(e,c,j,k)
                          res_mm23 = res_mm23 - h2a_vvov(c,b,i,e) * r2a(e,a,j,k)
                          res_mm23 = res_mm23 - h2a_vvov(a,c,i,e) * r2a(e,b,j,k)
                          res_mm23 = res_mm23 - h2a_vvov(a,b,j,e) * r2a(e,c,i,k)
                          res_mm23 = res_mm23 + h2a_vvov(c,b,j,e) * r2a(e,a,i,k)
                          res_mm23 = res_mm23 + h2a_vvov(a,c,j,e) * r2a(e,b,i,k)
                          res_mm23 = res_mm23 - h2a_vvov(a,b,k,e) * r2a(e,c,j,i)
                          res_mm23 = res_mm23 + h2a_vvov(c,b,k,e) * r2a(e,a,j,i)
                          res_mm23 = res_mm23 + h2a_vvov(a,c,k,e) * r2a(e,b,j,i)
                           ! A(i/jk)(c/ab) x2a(abie) * t2a(ecjk)
                          res_mm23 = res_mm23 + x2a_vvov(a,b,i,e) * t2a(e,c,j,k)
                          res_mm23 = res_mm23 - x2a_vvov(c,b,i,e) * t2a(e,a,j,k)
                          res_mm23 = res_mm23 - x2a_vvov(a,c,i,e) * t2a(e,b,j,k)
                          res_mm23 = res_mm23 - x2a_vvov(a,b,j,e) * t2a(e,c,i,k)
                          res_mm23 = res_mm23 + x2a_vvov(c,b,j,e) * t2a(e,a,i,k)
                          res_mm23 = res_mm23 + x2a_vvov(a,c,j,e) * t2a(e,b,i,k)
                          res_mm23 = res_mm23 - x2a_vvov(a,b,k,e) * t2a(e,c,j,i)
                          res_mm23 = res_mm23 + x2a_vvov(c,b,k,e) * t2a(e,a,j,i)
                          res_mm23 = res_mm23 + x2a_vvov(a,c,k,e) * t2a(e,b,j,i)
                      end do
                      do m = 1, noa
                          ! -A(k/ij)A(a/bc) h2a(amij) * r2a(bcmk)
                          res_mm23 = res_mm23 - h2a_vooo(a,m,i,j) * r2a(b,c,m,k)
                          res_mm23 = res_mm23 + h2a_vooo(b,m,i,j) * r2a(a,c,m,k)
                          res_mm23 = res_mm23 + h2a_vooo(c,m,i,j) * r2a(b,a,m,k)
                          res_mm23 = res_mm23 + h2a_vooo(a,m,k,j) * r2a(b,c,m,i)
                          res_mm23 = res_mm23 - h2a_vooo(b,m,k,j) * r2a(a,c,m,i)
                          res_mm23 = res_mm23 - h2a_vooo(c,m,k,j) * r2a(b,a,m,i)
                          res_mm23 = res_mm23 + h2a_vooo(a,m,i,k) * r2a(b,c,m,j)
                          res_mm23 = res_mm23 - h2a_vooo(b,m,i,k) * r2a(a,c,m,j)
                          res_mm23 = res_mm23 - h2a_vooo(c,m,i,k) * r2a(b,a,m,j)
                          ! -A(k/ij)A(a/bc) x2a(amij) * t2a(bcmk)
                          res_mm23 = res_mm23 - x2a_vooo(a,m,i,j) * t2a(b,c,m,k)
                          res_mm23 = res_mm23 + x2a_vooo(b,m,i,j) * t2a(a,c,m,k)
                          res_mm23 = res_mm23 + x2a_vooo(c,m,i,j) * t2a(b,a,m,k)
                          res_mm23 = res_mm23 + x2a_vooo(a,m,k,j) * t2a(b,c,m,i)
                          res_mm23 = res_mm23 - x2a_vooo(b,m,k,j) * t2a(a,c,m,i)
                          res_mm23 = res_mm23 - x2a_vooo(c,m,k,j) * t2a(b,a,m,i)
                          res_mm23 = res_mm23 + x2a_vooo(a,m,i,k) * t2a(b,c,m,j)
                          res_mm23 = res_mm23 - x2a_vooo(b,m,i,k) * t2a(a,c,m,j)
                          res_mm23 = res_mm23 - x2a_vooo(c,m,i,k) * t2a(b,a,m,j)
                      end do
                      resid(idet) = resid(idet) + res_mm23
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
              end subroutine build_hr_3a

              subroutine build_hr_3b(resid,&
                                     r2a,r2b,&
                                     r3a_amps, r3a_excits,&
                                     r3b_amps, r3b_excits,&
                                     r3c_amps, r3c_excits,&
                                     t2a,t2b,&
                                     t3a_amps, t3a_excits,&
                                     t3b_amps, t3b_excits,&
                                     t3c_amps, t3c_excits,&
                                     h1a_oo, h1a_vv, h1b_oo, h1b_vv,&
                                     h2a_oooo, h2a_vooo, h2a_oovv,&
                                     h2a_voov, h2a_vvov, h2a_vvvv,&
                                     h2b_oooo, h2b_vooo, h2b_ovoo,&
                                     h2b_oovv, h2b_voov, h2b_vovo,&
                                     h2b_ovov, h2b_ovvo, h2b_vvov,&
                                     h2b_vvvo, h2b_vvvv,&
                                     h2c_oovv, h2c_voov,&
                                     x1a_oo, x1a_vv, x1b_oo, x1b_vv,&
                                     x2a_oooo, x2a_vooo, x2a_oovv,&
                                     x2a_voov, x2a_vvov, x2a_vvvv,&
                                     x2b_oooo, x2b_vooo, x2b_ovoo,&
                                     x2b_oovv, x2b_voov, x2b_vovo,&
                                     x2b_ovov, x2b_ovvo, x2b_vvov,&
                                     x2b_vvvo, x2b_vvvv,&
                                     x2c_oovv, x2c_voov,&
                                     n3aaa_r, n3aab_r, n3abb_r,&
                                     n3aaa_t, n3aab_t, n3abb_t,&
                                     noa, nua, nob, nub)
                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa_r, n3aaa_t 
                  integer, intent(in) :: n3aab_r, n3aab_t
                  integer, intent(in) :: n3abb_r, n3abb_t
                  ! Input R and T arrays
                  real(kind=8), intent(in) :: r2a(nua,nua,noa,noa), t2a(nua,nua,noa,noa)
                  real(kind=8), intent(in) :: r2b(nua,nub,noa,nob), t2b(nua,nub,noa,nob)
                  integer, intent(in) :: r3a_excits(6,n3aaa_r), t3a_excits(6,n3aaa_t)
                  integer, intent(in) :: r3c_excits(6,n3abb_r), t3c_excits(6,n3abb_t)
                  integer, intent(in) :: t3b_excits(6,n3aab_t)
                  real(kind=8), intent(in) :: r3a_amps(n3aaa_r), t3a_amps(n3aaa_t)
                  real(kind=8), intent(in) :: r3c_amps(n3abb_r), t3c_amps(n3abb_t)
                  real(kind=8), intent(in) :: t3b_amps(n3aab_t)
                  ! Input H arrays
                  real(kind=8), intent(in) :: h1a_oo(noa,noa), h1b_oo(nob,nob)
                  real(kind=8), intent(in) :: h1a_vv(nua,nua), h1b_vv(nub,nub)
                  real(kind=8), intent(in) :: h2a_oooo(noa,noa,noa,noa)
                  real(kind=8), intent(in) :: h2a_vooo(nua,noa,noa,noa)
                  real(kind=8), intent(in) :: h2a_oovv(noa,noa,nua,nua)
                  !real(kind=8), intent(in) :: h2a_voov(nua,noa,noa,nua)
                  real(kind=8), intent(in) :: h2a_voov(noa,nua,nua,noa) ! reordered
                  !real(kind=8), intent(in) :: h2a_vvov(nua,nua,noa,nua)
                  real(kind=8), intent(in) :: h2a_vvov(nua,nua,nua,noa) ! reordered
                  real(kind=8), intent(in) :: h2a_vvvv(nua,nua,nua,nua)
                  real(kind=8), intent(in) :: h2b_oooo(noa,nob,noa,nob)
                  real(kind=8), intent(in) :: h2b_vooo(nua,nob,noa,nob)
                  real(kind=8), intent(in) :: h2b_ovoo(noa,nub,noa,nob)
                  real(kind=8), intent(in) :: h2b_oovv(noa,nob,nua,nub)
                  !real(kind=8), intent(in) :: h2b_voov(nua,nob,noa,nub)
                  real(kind=8), intent(in) :: h2b_voov(nob,nub,nua,nob) ! reordered
                  !real(kind=8), intent(in) :: h2b_vovo(nua,nob,nua,nob)
                  real(kind=8), intent(in) :: h2b_vovo(nob,nua,nua,nob) ! reordered
                  !real(kind=8), intent(in) :: h2b_ovov(noa,nub,noa,nub)
                  real(kind=8), intent(in) :: h2b_ovov(noa,nub,nub,noa) ! reordered
                  !real(kind=8), intent(in) :: h2b_ovvo(noa,nub,nua,nob)
                  real(kind=8), intent(in) :: h2b_ovvo(noa,nua,nub,nob) ! reordered
                  !real(kind=8), intent(in) :: h2b_vvov(nua,nub,noa,nub)
                  real(kind=8), intent(in) :: h2b_vvov(nub,nua,nub,noa) ! reordered
                  !real(kind=8), intent(in) :: h2b_vvvo(nua,nub,nua,nob)
                  real(kind=8), intent(in) :: h2b_vvvo(nua,nua,nub,nob) ! reordered
                  real(kind=8), intent(in) :: h2b_vvvv(nua,nub,nua,nub)
                  real(kind=8), intent(in) :: h2c_oovv(nob,nob,nub,nub)
                  !real(kind=8), intent(in) :: h2c_voov(nub,nob,nob,nub)
                  real(kind=8), intent(in) :: h2c_voov(nob,nub,nub,nob) ! reordered
                  ! Input X arrays
                  real(kind=8), intent(in) :: x1a_oo(noa,noa), x1b_oo(nob,nob)
                  real(kind=8), intent(in) :: x1a_vv(nua,nua), x1b_vv(nub,nub)
                  real(kind=8), intent(in) :: x2a_oooo(noa,noa,noa,noa)
                  real(kind=8), intent(in) :: x2a_vooo(nua,noa,noa,noa)
                  real(kind=8), intent(in) :: x2a_oovv(noa,noa,nua,nua)
                  !real(kind=8), intent(in) :: x2a_voov(nua,noa,noa,nua)
                  real(kind=8), intent(in) :: x2a_voov(noa,nua,nua,noa) ! reordered
                  !real(kind=8), intent(in) :: x2a_vvov(nua,nua,noa,nua)
                  real(kind=8), intent(in) :: x2a_vvov(nua,nua,nua,noa) ! reordered
                  real(kind=8), intent(in) :: x2a_vvvv(nua,nua,nua,nua)
                  real(kind=8), intent(in) :: x2b_oooo(noa,nob,noa,nob)
                  real(kind=8), intent(in) :: x2b_vooo(nua,nob,noa,nob)
                  real(kind=8), intent(in) :: x2b_ovoo(noa,nub,noa,nob)
                  real(kind=8), intent(in) :: x2b_oovv(noa,nob,nua,nub)
                  !real(kind=8), intent(in) :: x2b_voov(nua,nob,noa,nub)
                  real(kind=8), intent(in) :: x2b_voov(nob,nub,nua,nob) ! reordered
                  !real(kind=8), intent(in) :: x2b_vovo(nua,nob,nua,nob)
                  real(kind=8), intent(in) :: x2b_vovo(nob,nua,nua,nob) ! reordered
                  !real(kind=8), intent(in) :: x2b_ovov(noa,nub,noa,nub)
                  real(kind=8), intent(in) :: x2b_ovov(noa,nub,nub,noa) ! reordered
                  !real(kind=8), intent(in) :: x2b_ovvo(noa,nub,nua,nob)
                  real(kind=8), intent(in) :: x2b_ovvo(noa,nua,nub,nob) ! reordered
                  !real(kind=8), intent(in) :: x2b_vvov(nua,nub,noa,nub)
                  real(kind=8), intent(in) :: x2b_vvov(nub,nua,nub,noa) ! reordered
                  !real(kind=8), intent(in) :: x2b_vvvo(nua,nub,nua,nob)
                  real(kind=8), intent(in) :: x2b_vvvo(nua,nua,nub,nob) ! reordered
                  real(kind=8), intent(in) :: x2b_vvvv(nua,nub,nua,nub)
                  real(kind=8), intent(in) :: x2c_oovv(nob,nob,nub,nub)
                  !real(kind=8), intent(in) :: x2c_voov(nub,nob,nob,nub)
                  real(kind=8), intent(in) :: x2c_voov(nob,nub,nub,nob) ! reordered
                  ! Output and Inout variables
                  real(kind=8), intent(out) :: resid(n3aab_r)
                  integer, intent(inout) :: r3b_excits(6,n3aab_r)
                  !f2py intent(in,out) :: r3b_excits(6,0:n3aab_r-1)
                  real(kind=8), intent(inout) :: r3b_amps(n3aab_r)
                  !f2py intent(in,out) :: r3b_amps(0:n3aab_r-1)
                  ! Local variables
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:)
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)
                  real(kind=8) :: t_amp, r_amp, hmatel, hmatel1, res_mm23
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc
                  ! new strategy
                  !real(kind=8), external :: ddot
                  ! timings variables
                  !real(kind=8) :: tic, toc, ttic, ttoc, stic, stoc, dtic, dtoc, dtime, stime, ttime
                  
                  ! zero timings variables
                  !dtime = 0.0
                  !stime = 0.0
                  !ttime = 0.0
                  ! Zero the container that holds H*R
                  resid = 0.0d0

                  !!!! diagram 1a: -A(ij) h1a(mi)*r3b(abcmjk)
                  !!!! diagram 5a: A(ij) 1/2 h2a(mnij)*r3b(abcmnk)
                  !!! ABCK LOOP !!! 
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2*nub*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,nub,noa))
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/1,nob/), nua, nua, nub, noa)
                  call sort4(r3b_excits, r3b_amps, loc_arr, idx_table, (/1,2,3,6/), nua, nua, nub, noa, nloc, n3aab_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h1a_oo,h2a_oooo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     idx = idx_table(a,b,c,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        l = r3b_excits(4,jdet); m = r3b_excits(5,jdet);
                        ! compute < ijk~abc~ | h2a(oooo) | lmk~abc~ >
                        hmatel = h2a_oooo(l,m,i,j)
                        ! compute < ijk~abc~ | h1a(oo) | lmk~abc~ > = -A(ij)A(lm) h1a_oo(l,i) * delta(m,j)
                        if (m==j) hmatel = hmatel - h1a_oo(l,i)
                        if (m==i) hmatel = hmatel + h1a_oo(l,j)
                        if (l==j) hmatel = hmatel + h1a_oo(m,i)
                        if (l==i) hmatel = hmatel - h1a_oo(m,j)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  !!!! diagram 1b: -A(ij) x1a(mi)*t3b(abcmjk)
                  !!!! diagram 5b: A(ij) 1/2 x2a(mnij)*t3b(abcmnk)
                  !!! ABCK LOOP !!!
                  ! allocate temporary arrays
                  allocate(excits_buff(6,n3aab_t),amps_buff(n3aab_t))
                  excits_buff(:,:) = t3b_excits(:,:)
                  amps_buff(:) = t3b_amps(:) 
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2*nub*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,nub,noa))
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/1,nob/), nua, nua, nub, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,3,6/), nua, nua, nub, noa, nloc, n3aab_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x1a_oo,x2a_oooo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     idx = idx_table(a,b,c,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        l = excits_buff(4,jdet); m = excits_buff(5,jdet);
                        ! compute < ijk~abc~ | h2a(oooo) | lmk~abc~ >
                        hmatel = x2a_oooo(l,m,i,j)
                        ! compute < ijk~abc~ | h1a(oo) | lmk~abc~ > = -A(ij)A(lm) h1a_oo(l,i) * delta(m,j)
                        if (m==j) hmatel = hmatel - x1a_oo(l,i)
                        if (m==i) hmatel = hmatel + x1a_oo(l,j)
                        if (l==j) hmatel = hmatel + x1a_oo(m,i)
                        if (l==i) hmatel = hmatel - x1a_oo(m,j)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate temporary arrays
                  deallocate(excits_buff,amps_buff)

                  !call cpu_time(tic)
                  !!!! diagram 2a: A(ab) h1a(ae)*r3b(ebcmjk)
                  !!!! diagram 6a: A(ab) 1/2 h2a(abef)*r3b(ebcmjk)
                  !!! CIJK LOOP !!!
                  ! allocate new sorting arrays
                  nloc = nub*noa*(noa-1)/2*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(noa,noa,nob,nub))
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/1,nub/), noa, noa, nob, nub)
                  call sort4(r3b_excits, r3b_amps, loc_arr, idx_table, (/4,5,6,3/), noa, noa, nob, nub, nloc, n3aab_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_vv,H2A_vvvv,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     idx = idx_table(i,j,k,c)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3b_excits(1,jdet); e = r3b_excits(2,jdet);
                        ! compute < ijk~abc~ | h2a(vvvv) | ijk~dec~ >
                        !hmatel = h2a_vvvv(a,b,d,e)
                        hmatel = h2a_vvvv(d,e,a,b)
                        ! compute < ijk~abc~ | h1a(vv) | ijk~dec > = A(ab)A(de) h1a_vv(a,d)*delta(b,e)
                        if (b==e) hmatel = hmatel + h1a_vv(a,d)
                        if (a==e) hmatel = hmatel - h1a_vv(b,d)
                        if (b==d) hmatel = hmatel - h1a_vv(a,e)
                        if (a==d) hmatel = hmatel + h1a_vv(b,e)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  !!!! diagram 2b: A(ab) x1a(ae)*t3b(ebcmjk)
                  !!!! diagram 6b: A(ab) 1/2 x2a(abef)*t3b(ebcmjk)
                  !!! CIJK LOOP !!!
                  ! allocate temporary arrays
                  allocate(excits_buff(6,n3aab_t),amps_buff(n3aab_t))
                  excits_buff(:,:) = t3b_excits(:,:)
                  amps_buff(:) = t3b_amps(:) 
                  ! allocate new sorting arrays
                  nloc = nub*noa*(noa-1)/2*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(noa,noa,nob,nub))
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/1,nub/), noa, noa, nob, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,3/), noa, noa, nob, nub, nloc, n3aab_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_amps,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x1a_vv,x2a_vvvv,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     idx = idx_table(i,j,k,c)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); e = excits_buff(2,jdet);
                        ! compute < ijk~abc~ | x2a(vvvv) | ijk~dec~ >
                        !hmatel = x2a_vvvv(a,b,d,e)
                        hmatel = x2a_vvvv(d,e,a,b)
                        ! compute < ijk~abc~ | x1a(vv) | ijk~dec > = A(ab)A(de) x1a_vv(a,d)*delta(b,e)
                        if (b==e) hmatel = hmatel + x1a_vv(a,d)
                        if (a==e) hmatel = hmatel - x1a_vv(b,d)
                        if (b==d) hmatel = hmatel - x1a_vv(a,e)
                        if (a==d) hmatel = hmatel + x1a_vv(b,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate temporary arrays
                  deallocate(excits_buff,amps_buff)
                  !call cpu_time(toc)
                  !print*, "R3B - diagrams 2/6 = ", toc - tic

                  !!!! diagram 3a: -h1b(mk)*r3b(abcijm)
                  !!!! diagram 7a: A(ij) h2b(mnjk)*r3b(abcimn)
                  !!! ABCI LOOP !!!
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2*nub*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,nub,noa))
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/1,noa-1/), nua, nua, nub, noa)
                  call sort4(r3b_excits, r3b_amps, loc_arr, idx_table, (/1,2,3,4/), nua, nua, nub, noa, nloc, n3aab_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1B_oo,H2B_oooo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,i)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r3b_excits(5,jdet); n = r3b_excits(6,jdet);
                        ! compute < ijk~abc~ | h2b(oooo) | imn~abc~ >
                        hmatel = h2b_oooo(m,n,j,k)
                        ! compute < ijk~abc~ | h1b(oo) | imn~abc~ >
                        if (m==j) hmatel = hmatel - h1b_oo(n,k)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           m = r3b_excits(5,jdet); n = r3b_excits(6,jdet);
                           ! compute < ijk~abc~ | h2b(oooo) | jmn~abc~ >
                           hmatel = -h2b_oooo(m,n,i,k)
                           ! compute < ijk~abc~ | h1b(oo) | jmn~abc~ >
                           if (m==i) hmatel = hmatel + h1b_oo(n,k)
                           resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                        end do
                     end if    
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABCJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/2,noa/), nua, nua, nub, noa)
                  call sort4(r3b_excits, r3b_amps, loc_arr, idx_table, (/1,2,3,5/), nua, nua, nub, noa, nloc, n3aab_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_oooo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,j)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        l = r3b_excits(4,jdet); n = r3b_excits(6,jdet);
                        ! compute < ijk~abc~ | h2b(oooo) | ljn~abc~ >
                        hmatel = h2b_oooo(l,n,i,k)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = r3b_excits(4,jdet); n = r3b_excits(6,jdet);
                           ! compute < ijk~abc~ | h2b(oooo) | lin~abc~ >
                           hmatel = -h2b_oooo(l,n,j,k)
                           resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECITON !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table) 
                  !!!! diagram 3b: -h1b(mk)*t3b(abcijm)
                  !!!! diagram 7b: A(ij) h2b(mnjk)*t3b(abcimn)
                  !!! ABCI LOOP !!!
                  ! allocate temporary arrays
                  allocate(excits_buff(6,n3aab_t),amps_buff(n3aab_t))
                  excits_buff(:,:) = t3b_excits(:,:)
                  amps_buff(:) = t3b_amps(:) 
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2*nub*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,nub,noa))
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/1,noa-1/), nua, nua, nub, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,3,4/), nua, nua, nub, noa, nloc, n3aab_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x1b_oo,x2b_oooo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,i)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = excits_buff(5,jdet); n = excits_buff(6,jdet);
                        ! compute < ijk~abc~ | x2b(oooo) | imn~abc~ >
                        hmatel = x2b_oooo(m,n,j,k)
                        ! compute < ijk~abc~ | x1b(oo) | imn~abc~ >
                        if (m==j) hmatel = hmatel - x1b_oo(n,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           m = excits_buff(5,jdet); n = excits_buff(6,jdet);
                           ! compute < ijk~abc~ | x2b(oooo) | jmn~abc~ >
                           hmatel = -x2b_oooo(m,n,i,k)
                           ! compute < ijk~abc~ | x1b(oo) | jmn~abc~ >
                           if (m==i) hmatel = hmatel + x1b_oo(n,k)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if    
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABCJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/2,noa/), nua, nua, nub, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,3,5/), nua, nua, nub, noa, nloc, n3aab_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x2b_oooo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,j)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        l = excits_buff(4,jdet); n = excits_buff(6,jdet);
                        ! compute < ijk~abc~ | x2b(oooo) | ljn~abc~ >
                        hmatel = x2b_oooo(l,n,i,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = excits_buff(4,jdet); n = excits_buff(6,jdet);
                           ! compute < ijk~abc~ | x2b(oooo) | lin~abc~ >
                           hmatel = -x2b_oooo(l,n,j,k)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECITON !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table) 
                  ! deallocate temporary arrays
                  deallocate(excits_buff,amps_buff)

                  !call cpu_time(tic)
                  !! >> HOT SPOT << !
                  !!!! diagram 5a: h1b(ce)*r3b(abeijm)
                  !!!! diagram 8a: A(ab) h2b(bcef)*r3b(aefijk)
                  ! allocate new sorting arrays
                  nloc = nua*noa*(noa-1)/2*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(noa,noa,nob,nua))
                  !!! AIJK LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/1,nua-1/), noa, noa, nob, nua)
                  call sort4(r3b_excits, r3b_amps, loc_arr, idx_table, (/4,5,6,1/), noa, noa, nob, nua, nloc, n3aab_r, resid)
                  !!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1B_vv,H2B_vvvv,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                      a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                      i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                      ! (1)
                      idx = idx_table(i,j,k,a)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                         e = r3b_excits(2,jdet); f = r3b_excits(3,jdet);
                         ! compute < ijk~abc~ | h2b(vvvv) | ijk~aef~ >
                         !hmatel = h2b_vvvv(b,c,e,f)
                         hmatel = h2b_vvvv(e,f,b,c)
                         if (b==e) hmatel = hmatel + h1b_vv(c,f)
                         resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                      end do
                      ! (ab)
                      idx = idx_table(i,j,k,b)
                      if (idx/=0) then ! protect against case where b = nua because a = 1, nua-1
                         do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            e = r3b_excits(2,jdet); f = r3b_excits(3,jdet);
                            ! compute < ijk~abc~ | h2b(vvvv) | ijk~bef~ >
                            !hmatel = -h2b_vvvv(a,c,e,f)
                            hmatel = -h2b_vvvv(e,f,a,c)
                            if (a==e) hmatel = hmatel - h1b_vv(c,f)
                            resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                         end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!! END OMP PARALLEL SECTION !!!!
                  !!! BIJK LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/2,nua/), noa, noa, nob, nua)
                  call sort4(r3b_excits, r3b_amps, loc_arr, idx_table, (/4,5,6,2/), noa, noa, nob, nua, nloc, n3aab_r, resid)
                  !!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_vvvv,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                      a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                      i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                      idx = idx_table(i,j,k,b)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                         d = r3b_excits(1,jdet); f = r3b_excits(3,jdet);
                         ! compute < ijk~abc~ | h2b(vvvv) | ijk~dbf~ >
                         !hmatel = h2b_vvvv(a,c,d,f)
                         hmatel = h2b_vvvv(d,f,a,c)
                         resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                      end do
                      idx = idx_table(i,j,k,a)
                      if (idx/=0) then ! protect against case where a = 1 because b = 2, nua
                         do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = r3b_excits(1,jdet); f = r3b_excits(3,jdet);
                            ! compute < ijk~abc~ | h2b(vvvv) | ijk~daf~ >
                            !hmatel = -h2b_vvvv(b,c,d,f)
                            hmatel = -h2b_vvvv(d,f,b,c)
                            resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                         end do
                      end if
                  end do ! end loop over idet
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  !!!! diagram 5b: x1b(ce)*t3b(abeijm)
                  !!!! diagram 8b: A(ab) x2b(bcef)*t3b(aefijk)
                  ! allocate temporary arrays
                  allocate(excits_buff(6,n3aab_t),amps_buff(n3aab_t))
                  excits_buff(:,:) = t3b_excits(:,:)
                  amps_buff(:) = t3b_amps(:)
                  ! allocate new sorting arrays
                  nloc = nua*noa*(noa-1)/2*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(noa,noa,nob,nua))
                  !!! AIJK LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/1,nua-1/), noa, noa, nob, nua)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,1/), noa, noa, nob, nua, nloc, n3aab_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x1b_vv,x2b_vvvv,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                      a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                      i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                      ! (1)
                      idx = idx_table(i,j,k,a)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                         e = excits_buff(2,jdet); f = excits_buff(3,jdet);
                         ! compute < ijk~abc~ | x2b(vvvv) | ijk~aef~ >
                         !hmatel = x2b_vvvv(b,c,e,f)
                         hmatel = x2b_vvvv(e,f,b,c)
                         if (b==e) hmatel = hmatel + x1b_vv(c,f)
                         resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                      end do
                      ! (ab)
                      idx = idx_table(i,j,k,b)
                      if (idx/=0) then ! protect against case where b = nua because a = 1, nua-1
                         do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            e = excits_buff(2,jdet); f = excits_buff(3,jdet);
                            ! compute < ijk~abc~ | x2b(vvvv) | ijk~bef~ >
                            !hmatel = -x2b_vvvv(a,c,e,f)
                            hmatel = -x2b_vvvv(e,f,a,c)
                            if (a==e) hmatel = hmatel - x1b_vv(c,f)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                         end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BIJK LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/2,nua/), noa, noa, nob, nua)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,2/), noa, noa, nob, nua, nloc, n3aab_t)
                  !!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x2b_vvvv,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                      a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                      i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                      idx = idx_table(i,j,k,b)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                         d = excits_buff(1,jdet); f = excits_buff(3,jdet);
                         ! compute < ijk~abc~ | x2b(vvvv) | ijk~dbf~ >
                         !hmatel = x2b_vvvv(a,c,d,f)
                         hmatel = x2b_vvvv(d,f,a,c)
                         resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                      end do
                      idx = idx_table(i,j,k,a)
                      if (idx/=0) then ! protect against case where a = 1 because b = 2, nua
                         do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = excits_buff(1,jdet); f = excits_buff(3,jdet);
                            ! compute < ijk~abc~ | x2b(vvvv) | ijk~daf~ >
                            !hmatel = -x2b_vvvv(b,c,d,f)
                            hmatel = -x2b_vvvv(d,f,b,c)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                         end do
                      end if
                  end do ! end loop over idet
                  !$omp end do
                  !$omp end parallel
                  !!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate temporary arrays
                  deallocate(excits_buff,amps_buff)
                  !call cpu_time(toc)
                  !print*, "R3B - diagrams 5/8 = ", toc - tic

!                  call cpu_time(ttic)
!                  ! >> HOT SPOT << !
!                  !!!! diagram 5a: h1b(ce)*r3b(abeijm)
!                  !!!! diagram 8a: A(ab) h2b(bcef)*r3b(aefijk)
!                  allocate(h_buff(nua*nub),x_buff(nua*nub))
!                  ! allocate new sorting arrays
!                  nloc = nua*noa*(noa-1)/2*nob
!                  allocate(loc_arr(2,nloc))
!                  allocate(idx_table(noa,noa,nob,nua))
!                  !!! AIJK LOOP !!!
!                  call cpu_time(stic)
!                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/1,nua-1/), noa, noa, nob, nua)
!                  call sort4(r3b_excits, r3b_amps, loc_arr, idx_table, (/4,5,6,1/), noa, noa, nob, nua, nloc, n3aab_r, resid)
!                  call cpu_time(stoc)
!                  stime = stime + (stoc - stic)
!                  call cpu_time(dtic)
!                  ! Can refactor this
!                  ! do idx = 1, N_aijk
!                  !    idet = loc_arr(1,idx)
!                  !    a, b, c, i, j, k = r3b_excits(:,idet)
!                  !    do jdet = loc_arr(1,idx), loc_arr(2,idx)
!                  !       a, e, f, i, j, k = r3b_excits(:,jdet)
!                  !       hmatel = <ijkabc|H|ijkaef>
!                  !       sigma(idet) += hmatel * r3b_amps(jdet)
!                  !    end do
!                  ! end do
!                  do idet = 1, n3aab_r ! loop over r3b ordered in (a,i,j,k) blocks -> (b,c) are changing fast
!                      a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
!                      i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
!                      ! (1)
!                      idx = idx_table(i,j,k,a) ! -> repeat fetching; diagonal-blocks,
!                      ij = 1
!                      do jdet = loc_arr(1,idx), loc_arr(2,idx) ! -> repeat fetching
!                         e = r3b_excits(2,jdet); f = r3b_excits(3,jdet);
!                         ! compute < ijk~abc~ | h2b(vvvv) | ijk~aef~ >
!                         h_buff(ij) = h2b_vvvv(b,c,e,f)
!                         if (b==e) h_buff(ij) = h_buff(ij) + h1b_vv(c,f)
!                         x_buff(ij) = r3b_amps(jdet)
!                         ij = ij + 1
!                      end do
!                      print*, ij
!                      !h_buff(1:njdet) = h2b_vvvv(b,c,r3b_excits(2,loc_arr(1,idx):loc_arr(2,idx)),r3b_excits(3,loc_arr(1,idx):loc_arr(2,idx)))
!                      !hx = ddot(ij-1, h_buff(1:ij-1), 1, x_buff(1:ij-1), 1)
!                      !hx = 0.0d0
!                      !resid(idet) = resid(idet) + hx
!                      ! (ab)
!                      idx = idx_table(i,j,k,b) ! off-diagonal blocks, e.g., Loop a2 <- a1 + 1, N_aijk
!                      if (idx/=0) then ! protect against case where b = nua because a = 1, nua-1
!                         ij = 1
!                         do jdet = loc_arr(1,idx), loc_arr(2,idx)
!                            e = r3b_excits(2,jdet); f = r3b_excits(3,jdet);
!                            ! compute < ijk~abc~ | h2b(vvvv) | ijk~bef~ >
!                            h_buff(ij) = -h2b_vvvv(a,c,e,f)
!                            if (a==e) h_buff(ij) = h_buff(ij) - h1b_vv(c,f)
!                            x_buff(ij) = r3b_amps(jdet)
!                            ij = ij + 1
!                         end do
!                         !print*, ij
!                         !hx = ddot(ij-1, h_buff(1:ij-1), 1, x_buff(1:ij-1), 1)
!                         !hx = 0.0d0
!                         !resid(idet) = resid(idet) + hx
!                      end if
!                  end do
!                  call cpu_time(dtoc)
!                  dtime = dtime + (dtoc - dtic)
!                  !!! BIJK LOOP !!!
!                  call cpu_time(stic)
!                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/2,nua/), noa, noa, nob, nua)
!                  call sort4(r3b_excits, r3b_amps, loc_arr, idx_table, (/4,5,6,2/), noa, noa, nob, nua, nloc, n3aab_r, resid)
!                  call cpu_time(stoc)
!                  stime = stime + (stoc - stic)
!                  call cpu_time(dtic)
!                  do idet = 1, n3aab_r
!                      a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
!                      i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
!                      !idx = idx_table(i,j,k,b)
!                      ij = 1
!                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
!                         d = r3b_excits(1,jdet); f = r3b_excits(3,jdet);
!                         ! compute < ijk~abc~ | h2b(vvvv) | ijk~dbf~ >
!                         !h_buff(ij) = h2b_vvvv(a,c,d,f)
!                         x_buff(ij) = r3b_amps(jdet)
!                         ij = ij + 1
!                      end do
!                      !hx = ddot(ij-1, h_buff(1:ij-1), 1, x_buff(1:ij-1), 1)
!                      !hx = 0.0d0
!                      !resid(idet) = resid(idet) + hx
!                      !idx = idx_table(i,j,k,a)
!                      if (idx/=0) then ! protect against case where a = 1 because b = 2, nua
!                         ij = 1
!                         do jdet = loc_arr(1,idx), loc_arr(2,idx)
!                            d = r3b_excits(1,jdet); f = r3b_excits(3,jdet);
!                            ! compute < ijk~abc~ | h2b(vvvv) | ijk~daf~ >
!                            !h_buff(ij) = -h2b_vvvv(b,c,d,f)
!                            x_buff(ij) = r3b_amps(jdet)
!                            ij = ij + 1
!                         end do
!                         !hx = ddot(ij-1, h_buff(1:ij-1), 1, x_buff(1:ij-1), 1)
!                         !hx = 0.0d0
!                         !resid(idet) = resid(idet) + hx
!                      end if
!                  end do
!                  call cpu_time(dtoc)
!                  dtime = dtime + (dtoc - dtic)
!                  ! deallocate sorting arrays
!                  deallocate(loc_arr,idx_table)
!                  !!!! diagram 5b: x1b(ce)*t3b(abeijm)
!                  !!!! diagram 8b: A(ab) x2b(bcef)*t3b(aefijk)
!                  ! allocate temporary arrays
!                  allocate(excits_buff(6,n3aab_t),amps_buff(n3aab_t))
!                  excits_buff(:,:) = t3b_excits(:,:)
!                  amps_buff(:) = t3b_amps(:)
!                  ! allocate new sorting arrays
!                  nloc = nua*noa*(noa-1)/2*nob
!                  allocate(loc_arr(2,nloc))
!                  allocate(idx_table(noa,noa,nob,nua))
!                  !!! AIJK LOOP !!!
!                  call cpu_time(stic)
!                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/1,nua-1/), noa, noa, nob, nua)
!                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,1/), noa, noa, nob, nua, nloc, n3aab_t)
!                  call cpu_time(stoc)
!                  stime = stime + (stoc - stic)
!                  call cpu_time(dtic)
!                  do idet = 1, n3aab_r
!                      a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
!                      i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
!                      ! (1)
!                      !idx = idx_table(i,j,k,a)
!                      ij = 1
!                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
!                         e = excits_buff(2,jdet); f = excits_buff(3,jdet);
!                         ! compute < ijk~abc~ | x2b(vvvv) | ijk~aef~ >
!                         !h_buff(ij) = x2b_vvvv(b,c,e,f)
!                         !if (b==e) h_buff(ij) = h_buff(ij) + x1b_vv(c,f)
!                         x_buff(ij) = amps_buff(jdet)
!                         ij = ij + 1
!                      end do
!                      !hx = ddot(ij-1, h_buff(1:ij-1), 1, x_buff(1:ij-1), 1)
!                      !hx = 0.0d0
!                      !resid(idet) = resid(idet) + hx
!                      ! (ab)
!                      !idx = idx_table(i,j,k,b)
!                      if (idx/=0) then ! protect against case where b = nua because a = 1, nua-1
!                         ij = 1
!                         do jdet = loc_arr(1,idx), loc_arr(2,idx)
!                            e = excits_buff(2,jdet); f = excits_buff(3,jdet);
!                            ! compute < ijk~abc~ | x2b(vvvv) | ijk~bef~ >
!                            !h_buff(ij) = -x2b_vvvv(a,c,e,f)
!                            !if (a==e) h_buff(ij) = h_buff(ij) - x1b_vv(c,f)
!                            x_buff(ij) = amps_buff(jdet)
!                            ij = ij + 1
!                         end do
!                         !hx = ddot(ij-1, h_buff(1:ij-1), 1, x_buff(1:ij-1), 1)
!                         !hx = 0.0d0
!                         !resid(idet) = resid(idet) + hx
!                      end if
!                  end do
!                  call cpu_time(dtoc)
!                  dtime = dtime + (dtoc - dtic)
!                  !!! BIJK LOOP !!!
!                  call cpu_time(stic)
!                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/2,nua/), noa, noa, nob, nua)
!                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,2/), noa, noa, nob, nua, nloc, n3aab_t)
!                  call cpu_time(stoc)
!                  stime = stime + (stoc - stic)
!                  call cpu_time(dtic)
!                  do idet = 1, n3aab_r
!                      a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
!                      i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
!                      idx = idx_table(i,j,k,b)
!                      ij = 1
!                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
!                         d = excits_buff(1,jdet); f = excits_buff(3,jdet);
!                         ! compute < ijk~abc~ | x2b(vvvv) | ijk~dbf~ >
!                         h_buff(ij) = x2b_vvvv(a,c,d,f)
!                         x_buff(ij) = amps_buff(jdet)
!                         ij = ij + 1
!                      end do
!                      !hx = ddot(ij-1, h_buff(1:ij-1), 1, x_buff(1:ij-1), 1)
!                      !hx = 0.0d0
!                      !resid(idet) = resid(idet) + hx
!                      idx = idx_table(i,j,k,a)
!                      if (idx/=0) then ! protect against case where a = 1 because b = 2, nua
!                         ij = 1
!                         do jdet = loc_arr(1,idx), loc_arr(2,idx)
!                            d = excits_buff(1,jdet); f = excits_buff(3,jdet);
!                            ! compute < ijk~abc~ | x2b(vvvv) | ijk~daf~ >
!                            h_buff(ij) = -x2b_vvvv(b,c,d,f)
!                            x_buff(ij) = amps_buff(jdet)
!                            ij = ij + 1
!                         end do
!                         !hx = ddot(ij-1, h_buff(1:ij-1), 1, x_buff(1:ij-1), 1)
!                         !hx = 0.0d0
!                         !resid(idet) = resid(idet) + hx
!                      end if
!                  end do
!                  call cpu_time(dtoc)
!                  dtime = dtime + (dtoc - dtic)
!                  ! deallocate sorting arrays
!                  deallocate(loc_arr,idx_table)
!                  ! deallocate temporary arrays
!                  deallocate(excits_buff,amps_buff)
!                  deallocate(h_buff,x_buff)
!                  call cpu_time(ttoc)
!                  print*, "R3B - diagrams 5/8 = ", ttoc - ttic, "Diagram time =", dtime, "Sort time = ", stime

                  !call cpu_time(tic)
                  !!!! diagram 9a: A(ij)A(ab) h2a(amie)*r3b(ebcmjk)
                  ! allocate new sorting arrays
                  nloc = nua*nub*noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nub,noa,nob))
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/2,nua/), (/1,nub/), (/2,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(r3b_excits, r3b_amps, loc_arr, idx_table, (/2,3,5,6/), nua, nub, noa, nob, nloc, n3aab_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,j,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3b_excits(1,jdet); l = r3b_excits(4,jdet);
                        ! compute < ijk~abc~ | h2a(voov) | ljk~dbc~ >
                        !hmatel = h2a_voov(a,l,i,d)
                        hmatel = h2a_voov(l,d,a,i)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then ! protect against case where a = 1 because b = 2, nua
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = r3b_excits(1,jdet); l = r3b_excits(4,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | ljk~dac~ >
                           !hmatel = -h2a_voov(b,l,i,d)
                           hmatel = -h2a_voov(l,d,b,i)
                           resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                        end do
                     end if 
                     ! (ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then ! protect against case where i = 1 because j = 2, noa
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = r3b_excits(1,jdet); l = r3b_excits(4,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | lik~dbc~ >
                           !hmatel = -h2a_voov(a,l,j,d)
                           hmatel = -h2a_voov(l,d,a,j)
                           resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                        end do
                     end if 
                     ! (ij)(ab)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then ! protect against case where a = 1 because b = 2, nua and i = 1 because j = 2, noa
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = r3b_excits(1,jdet); l = r3b_excits(4,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | lik~dac~ >
                           !hmatel = h2a_voov(b,l,j,d)
                           hmatel = h2a_voov(l,d,b,j)
                           resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                        end do
                     end if 
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/2,nua/), (/1,nub/), (/1,noa-1/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(r3b_excits, r3b_amps, loc_arr, idx_table, (/2,3,4,6/), nua, nub, noa, nob, nloc, n3aab_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,i,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3b_excits(1,jdet); l = r3b_excits(5,jdet);
                        ! compute < ijk~abc~ | h2a(voov) | ilk~dbc~ >
                        !hmatel = h2a_voov(a,l,j,d)
                        hmatel = h2a_voov(l,d,a,j)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then ! protect against where j = noa because i = 1, noa-1 
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = r3b_excits(1,jdet); l = r3b_excits(5,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | jlk~dbc~ >
                           !hmatel = -h2a_voov(a,l,i,d)
                           hmatel = -h2a_voov(l,d,a,i)
                           resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then ! protect against case where a = 1 because b = 2, nua
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = r3b_excits(1,jdet); l = r3b_excits(5,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | ilk~dac~ >
                           !hmatel = -h2a_voov(b,l,j,d)
                           hmatel = -h2a_voov(l,d,b,j)
                           resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then ! protect against case where j = noa because i = 1, noa-1 and where a = 1 because b = 2, nua
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = r3b_excits(1,jdet); l = r3b_excits(5,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | jlk~dac~ >
                           !hmatel = h2a_voov(b,l,i,d)
                           hmatel = h2a_voov(l,d,b,i)
                           resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/1,nub/), (/1,noa-1/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(r3b_excits, r3b_amps, loc_arr, idx_table, (/1,3,4,6/), nua, nub, noa, nob, nloc, n3aab_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3b_excits(2,jdet); l = r3b_excits(5,jdet);
                        ! compute < ijk~abc~ | h2a(voov) | ilk~adc~  >
                        !hmatel = h2a_voov(b,l,j,d)
                        hmatel = h2a_voov(l,d,b,j)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = r3b_excits(2,jdet); l = r3b_excits(5,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | jlk~adc~  >
                           !hmatel = -h2a_voov(b,l,i,d)
                           hmatel = -h2a_voov(l,d,b,i)
                           resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = r3b_excits(2,jdet); l = r3b_excits(5,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | ilk~bdc~  >
                           !hmatel = -h2a_voov(a,l,j,d)
                           hmatel = -h2a_voov(l,d,a,j)
                           resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = r3b_excits(2,jdet); l = r3b_excits(5,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | jlk~bdc~  >
                           !hmatel = h2a_voov(a,l,i,d)
                           hmatel = h2a_voov(l,d,a,i)
                           resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/1,nub/), (/2,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(r3b_excits, r3b_amps, loc_arr, idx_table, (/1,3,5,6/), nua, nub, noa, nob, nloc, n3aab_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,j,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3b_excits(2,jdet); l = r3b_excits(4,jdet);
                        ! compute < ijk~abc~ | h2a(voov) | ljk~adc~  >
                        !hmatel = h2a_voov(b,l,i,d)
                        hmatel = h2a_voov(l,d,b,i)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = r3b_excits(2,jdet); l = r3b_excits(4,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | lik~adc~  >
                           !hmatel = -h2a_voov(b,l,j,d)
                           hmatel = -h2a_voov(l,d,b,j)
                           resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = r3b_excits(2,jdet); l = r3b_excits(4,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | ljk~bdc~  >
                           !hmatel = -h2a_voov(a,l,i,d)
                           hmatel = -h2a_voov(l,d,a,i)
                           resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = r3b_excits(2,jdet); l = r3b_excits(4,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | lik~abc~  >
                           !hmatel = h2a_voov(a,l,j,d)
                           hmatel = h2a_voov(l,d,a,j)
                           resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  !!!! diagram 9b: A(ij)A(ab) x2a(amie)*t3b(ebcmjk)
                  ! allocate temporary arrays
                  allocate(excits_buff(6,n3aab_t),amps_buff(n3aab_t))
                  excits_buff(:,:) = t3b_excits(:,:)
                  amps_buff(:) = t3b_amps(:) 
                  ! allocate new sorting arrays
                  nloc = nua*nub*noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nub,noa,nob))
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/2,nua/), (/1,nub/), (/2,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,5,6/), nua, nub, noa, nob, nloc, n3aab_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x2a_voov,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,j,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < ijk~abc~ | h2a(voov) | ljk~dbc~ >
                        !hmatel = x2a_voov(a,l,i,d)
                        hmatel = x2a_voov(l,d,a,i)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then ! protect against case where a = 1 because b = 2, nua
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | ljk~dac~ >
                           !hmatel = -x2a_voov(b,l,i,d)
                           hmatel = -x2a_voov(l,d,b,i)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if 
                     ! (ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then ! protect against case where i = 1 because j = 2, noa
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | lik~dbc~ >
                           !hmatel = -x2a_voov(a,l,j,d)
                           hmatel = -x2a_voov(l,d,a,j)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if 
                     ! (ij)(ab)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then ! protect against case where a = 1 because b = 2, nua and i = 1 because j = 2, noa
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | lik~dac~ >
                           !hmatel = x2a_voov(b,l,j,d)
                           hmatel = x2a_voov(l,d,b,j)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if 
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/2,nua/), (/1,nub/), (/1,noa-1/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,6/), nua, nub, noa, nob, nloc, n3aab_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x2a_voov,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,i,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(5,jdet);
                        ! compute < ijk~abc~ | h2a(voov) | ilk~dbc~ >
                        !hmatel = x2a_voov(a,l,j,d)
                        hmatel = x2a_voov(l,d,a,j)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then ! protect against where j = noa because i = 1, noa-1 
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = excits_buff(1,jdet); l = excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | jlk~dbc~ >
                           !hmatel = -x2a_voov(a,l,i,d)
                           hmatel = -x2a_voov(l,d,a,i)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then ! protect against case where a = 1 because b = 2, nua
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = excits_buff(1,jdet); l = excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | ilk~dac~ >
                           !hmatel = -x2a_voov(b,l,j,d)
                           hmatel = -x2a_voov(l,d,b,j)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then ! protect against case where j = noa because i = 1, noa-1 and where a = 1 because b = 2, nua
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = excits_buff(1,jdet); l = excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | jlk~dac~ >
                           !hmatel = x2a_voov(b,l,i,d)
                           hmatel = x2a_voov(l,d,b,i)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/1,nub/), (/1,noa-1/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,6/), nua, nub, noa, nob, nloc, n3aab_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x2a_voov,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(2,jdet); l = excits_buff(5,jdet);
                        ! compute < ijk~abc~ | h2a(voov) | ilk~adc~  >
                        !hmatel = x2a_voov(b,l,j,d)
                        hmatel = x2a_voov(l,d,b,j)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = excits_buff(2,jdet); l = excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | jlk~adc~  >
                           !hmatel = -x2a_voov(b,l,i,d)
                           hmatel = -x2a_voov(l,d,b,i)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = excits_buff(2,jdet); l = excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | ilk~bdc~  >
                           !hmatel = -x2a_voov(a,l,j,d)
                           hmatel = -x2a_voov(l,d,a,j)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = excits_buff(2,jdet); l = excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | jlk~bdc~  >
                           !hmatel = x2a_voov(a,l,i,d)
                           hmatel = x2a_voov(l,d,a,i)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/1,nub/), (/2,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,5,6/), nua, nub, noa, nob, nloc, n3aab_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x2a_voov,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,j,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(2,jdet); l = excits_buff(4,jdet);
                        ! compute < ijk~abc~ | h2a(voov) | ljk~adc~  >
                        !hmatel = x2a_voov(b,l,i,d)
                        hmatel = x2a_voov(l,d,b,i)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = excits_buff(2,jdet); l = excits_buff(4,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | lik~adc~  >
                           !hmatel = -x2a_voov(b,l,j,d)
                           hmatel = -x2a_voov(l,d,b,j)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = excits_buff(2,jdet); l = excits_buff(4,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | ljk~bdc~  >
                           !hmatel = -x2a_voov(a,l,i,d)
                           hmatel = -x2a_voov(l,d,a,i)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = excits_buff(2,jdet); l = excits_buff(4,jdet);
                           ! compute < ijk~abc~ | h2a(voov) | lik~abc~  >
                           !hmatel = x2a_voov(a,l,j,d)
                           hmatel = x2a_voov(l,d,a,j)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate temporary arrays
                  deallocate(excits_buff,amps_buff)
                  !call cpu_time(toc)
                  !print*, "R3B - diagram 9 = ", toc - tic

                  !call cpu_time(tic)
                  !!!! diagram 10a: h2c(cmke)*r3b(abeijm)
                  ! allocate sorting arrays
                  nloc = nua*(nua-1)/2*noa*(noa-1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,noa,noa))
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(r3b_excits, r3b_amps, loc_arr, idx_table, (/1,2,4,5/), nua, nua, noa, noa, nloc, n3aab_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,&
                  !$omp n3aab_r),&
                  !$omp private(hmatel,a,b,c,i,j,k,f,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                      a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                      i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                      idx = idx_table(a,b,i,j)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                         f = r3b_excits(3,jdet); n = r3b_excits(6,jdet);
                         ! compute < ijk~abc~ | h2c(voov) | ijn~abf~ > = h2c_voov(c,n,k,f)
                         !hmatel = h2c_voov(c,n,k,f)
                         hmatel = h2c_voov(n,f,c,k)
                         resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                      end do
                  end do ! end loop over idet
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  !!!! diagram 10b: x2c(cmke)*t3b(abeijm)
                  ! allocate temporary arrays
                  allocate(excits_buff(6,n3aab_t),amps_buff(n3aab_t))
                  excits_buff(:,:) = t3b_excits(:,:)
                  amps_buff(:) = t3b_amps(:) 
                  ! allocate sorting arrays
                  nloc = nua*(nua-1)/2*noa*(noa-1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,noa,noa))
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,5/), nua, nua, noa, noa, nloc, n3aab_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x2c_voov,&
                  !$omp noa,nua,nob,nub,&
                  !$omp n3aab_r),&
                  !$omp private(hmatel,a,b,c,i,j,k,f,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                      a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                      i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                      idx = idx_table(a,b,i,j)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                         f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                         ! compute < ijk~abc~ | h2c(voov) | ijn~abf~ > = h2c_voov(c,n,k,f)
                         !hmatel = x2c_voov(c,n,k,f)
                         hmatel = x2c_voov(n,f,c,k)
                         resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                      end do
                  end do ! end loop over idet
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate temporary arrays
                  deallocate(excits_buff,amps_buff)
                  !call cpu_time(toc)
                  !print*, "R3B - diagram 10 = ", toc - tic

                  !call cpu_time(tic)
                  !!!! diagram 11a: -A(ij) h2b(mcie)*r3b(abemjk)
                  ! allocate sorting arrays
                  nloc = nua*(nua-1)/2*noa*nob
                  allocate(loc_arr(2,nloc)) 
                  allocate(idx_table(nua,nua,noa,nob))
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,noa-1/), (/1,nob/), nua, nua, noa, nob)
                  call sort4(r3b_excits, r3b_amps, loc_arr, idx_table, (/1,2,4,6/), nua, nua, noa, nob, nloc, n3aab_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovov,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,i,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3b_excits(3,jdet); m = r3b_excits(5,jdet);
                        ! compute < ijk~abc~ | h2b(ovov) | imk~abf~ >
                        !hmatel = -h2b_ovov(m,c,j,f)
                        hmatel = -h2b_ovov(m,f,c,j)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = r3b_excits(3,jdet); m = r3b_excits(5,jdet);
                           ! compute < ijk~abc~ | h2b(ovov) | jmk~abf~ >
                           !hmatel = h2b_ovov(m,c,i,f)
                           hmatel = h2b_ovov(m,f,c,i)
                           resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                        end do
                     end if
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/2,noa/), (/1,nob/), nua, nua, noa, nob)
                  call sort4(r3b_excits, r3b_amps, loc_arr, idx_table, (/1,2,5,6/), nua, nua, noa, nob, nloc, n3aab_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovov,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,j,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3b_excits(3,jdet); l = r3b_excits(4,jdet);
                        ! compute < ijk~abc~ | h2b(ovov) | ljk~abf~ >
                        !hmatel = -h2b_ovov(l,c,i,f)
                        hmatel = -h2b_ovov(l,f,c,i)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = r3b_excits(3,jdet); l = r3b_excits(4,jdet);
                           ! compute < ijk~abc~ | h2b(ovov) | lik~abf~ >
                           !hmatel = h2b_ovov(l,c,j,f)
                           hmatel = h2b_ovov(l,f,c,j)
                           resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                        end do
                     end if
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  !!!! diagram 11b: -A(ij) x2b(mcie)*t3b(abemjk)
                  ! allocate temporary arrays
                  allocate(excits_buff(6,n3aab_t),amps_buff(n3aab_t))
                  excits_buff(:,:) = t3b_excits(:,:)
                  amps_buff(:) = t3b_amps(:) 
                  ! allocate sorting arrays
                  nloc = nua*(nua-1)/2*noa*nob
                  allocate(loc_arr(2,nloc)) 
                  allocate(idx_table(nua,nua,noa,nob))
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,noa-1/), (/1,nob/), nua, nua, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,6/), nua, nua, noa, nob, nloc, n3aab_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x2b_ovov,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,i,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                        ! compute < ijk~abc~ | h2b(ovov) | imk~abf~ >
                        !hmatel = -x2b_ovov(m,c,j,f)
                        hmatel = -x2b_ovov(m,f,c,j)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2b(ovov) | jmk~abf~ >
                           !hmatel = x2b_ovov(m,c,i,f)
                           hmatel = x2b_ovov(m,f,c,i)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/2,noa/), (/1,nob/), nua, nua, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,5,6/), nua, nua, noa, nob, nloc, n3aab_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x2b_ovov,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,j,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); l = excits_buff(4,jdet);
                        ! compute < ijk~abc~ | h2b(ovov) | ljk~abf~ >
                        !hmatel = -x2b_ovov(l,c,i,f)
                        hmatel = -x2b_ovov(l,f,c,i)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = excits_buff(3,jdet); l = excits_buff(4,jdet);
                           ! compute < ijk~abc~ | h2b(ovov) | lik~abf~ >
                           !hmatel = x2b_ovov(l,c,j,f)
                           hmatel = x2b_ovov(l,f,c,j)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate temporary arrays
                  deallocate(excits_buff,amps_buff)
                  !call cpu_time(toc)
                  !print*, "R3B - diagram 11 = ", toc - tic
                  
                  !call cpu_time(tic)
                  !!!! diagram 12a: -A(ab) h2b(amek)*r3b(ebcijm)
                  ! allocate sorting arrays
                  nloc = nua*nub*noa*(noa-1)/2
                  allocate(loc_arr(2,nloc)) 
                  allocate(idx_table(noa,noa,nua,nub))
                  !!! BCIJ LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/2,nua/), (/1,nub/), noa, noa, nua, nub)
                  call sort4(r3b_excits, r3b_amps, loc_arr, idx_table, (/4,5,2,3/), noa, noa, nua, nub, nloc, n3aab_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_vovo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,b,c)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3b_excits(1,jdet); n = r3b_excits(6,jdet);
                        ! compute < ijk~abc~ | h2b(vovo) | ijn~dbc~ >
                        !hmatel = -h2b_vovo(a,n,d,k)
                        hmatel = -h2b_vovo(n,d,a,k)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = r3b_excits(1,jdet); n = r3b_excits(6,jdet);
                           ! compute < ijk~abc~ | h2b(vovo) | ijn~dac~ >
                           !hmatel = h2b_vovo(b,n,d,k)
                           hmatel = h2b_vovo(n,d,b,k)
                           resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                        end do
                     end if
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nua-1/), (/1,nub/), noa, noa, nua, nub)
                  call sort4(r3b_excits, r3b_amps, loc_arr, idx_table, (/4,5,1,3/), noa, noa, nua, nub, nloc, n3aab_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_vovo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,a,c)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3b_excits(2,jdet); n = r3b_excits(6,jdet);
                        ! compute < ijk~abc~ | h2b(vovo) | ijn~aec~ >
                        !hmatel = -h2b_vovo(b,n,e,k)
                        hmatel = -h2b_vovo(n,e,b,k)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           e = r3b_excits(2,jdet); n = r3b_excits(6,jdet);
                           ! compute < ijk~abc~ | h2b(vovo) | ijn~bec~ >
                           !hmatel = h2b_vovo(a,n,e,k)
                           hmatel = h2b_vovo(n,e,a,k)
                           resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                        end do
                     end if
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  !!!! diagram 12b: -A(ab) x2b(amek)*t3b(ebcijm)
                  ! allocate temporary arrays
                  allocate(excits_buff(6,n3aab_t),amps_buff(n3aab_t))
                  excits_buff(:,:) = t3b_excits(:,:)
                  amps_buff(:) = t3b_amps(:) 
                  ! allocate sorting arrays
                  nloc = nua*nub*noa*(noa-1)/2
                  allocate(loc_arr(2,nloc)) 
                  allocate(idx_table(noa,noa,nua,nub))
                  !!! BCIJ LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/2,nua/), (/1,nub/), noa, noa, nua, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,2,3/), noa, noa, nua, nub, nloc, n3aab_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x2b_vovo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,b,c)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); n = excits_buff(6,jdet);
                        ! compute < ijk~abc~ | h2b(vovo) | ijn~dbc~ >
                        !hmatel = -x2b_vovo(a,n,d,k)
                        hmatel = -x2b_vovo(n,d,a,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = excits_buff(1,jdet); n = excits_buff(6,jdet);
                           ! compute < ijk~abc~ | h2b(vovo) | ijn~dac~ >
                           !hmatel = x2b_vovo(b,n,d,k)
                           hmatel = x2b_vovo(n,d,b,k)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nua-1/), (/1,nub/), noa, noa, nua, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,1,3/), noa, noa, nua, nub, nloc, n3aab_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x2b_vovo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,a,c)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                        ! compute < ijk~abc~ | h2b(vovo) | ijn~aec~ >
                        !hmatel = -x2b_vovo(b,n,e,k)
                        hmatel = -x2b_vovo(n,e,b,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                           ! compute < ijk~abc~ | h2b(vovo) | ijn~bec~ >
                           !hmatel = x2b_vovo(a,n,e,k)
                           hmatel = x2b_vovo(n,e,a,k)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate temporary arrays
                  deallocate(excits_buff,amps_buff)
                  !call cpu_time(toc)
                  !print*, "R3B - diagram 12 = ", toc - tic

                  !call cpu_time(tic)
                  !!!! diagram 13a: h2b(mcek)*r3a(abeijm) !!!!
                  ! allocate and initialize the copy of r3a
                  allocate(amps_buff(n3aaa_r))
                  allocate(excits_buff(6,n3aaa_r))
                  amps_buff(:) = r3a_amps(:)
                  excits_buff(:,:) = r3a_excits(:,:)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = (nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,noa,noa))
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,5/), nua, nua, noa, noa, nloc, n3aaa_r)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     idx = idx_table(a,b,i,j)
                     if (idx==0) cycle 
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijk~abc~ | h2b(ovvo) | ijnabf >
                        !hmatel = h2b_ovvo(n,c,f,k)
                        hmatel = h2b_ovvo(n,f,c,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,5/), nua, nua, noa, noa, nloc, n3aaa_r)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                        ! compute < ijk~abc~ | h2b(ovvo) | ijnaeb >
                        !hmatel = -h2b_ovvo(n,c,e,k)
                        hmatel = -h2b_ovvo(n,e,c,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCIJ LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,5/), nua, nua, noa, noa, nloc, n3aaa_r)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); n = excits_buff(6,jdet);
                        ! compute < ijk~abc~ | h2b(ovvo) | ijndab >
                        !hmatel = h2b_ovvo(n,c,d,k)
                        hmatel = h2b_ovvo(n,d,c,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,6/), nua, nua, noa, noa, nloc, n3aaa_r)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                        ! compute < ijk~abc~ | h2b(ovvo) | imjabf >
                        !hmatel = -h2b_ovvo(m,c,f,k)
                        hmatel = -h2b_ovvo(m,f,c,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,6/), nua, nua, noa, noa, nloc, n3aaa_r)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                        ! compute < ijk~abc~ | h2b(ovvo) | imjaeb >
                        !hmatel = h2b_ovvo(m,c,e,k)
                        hmatel = h2b_ovvo(m,e,c,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,6/), nua, nua, noa, noa, nloc, n3aaa_r)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                        ! compute < ijk~abc~ | h2b(ovvo) | imjdab >
                        !hmatel = -h2b_ovvo(m,c,d,k)
                        hmatel = -h2b_ovvo(m,d,c,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,5,6/), nua, nua, noa, noa, nloc, n3aaa_r)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); l = excits_buff(4,jdet);
                        ! compute < ijk~abc~ | h2b(ovvo) | lijabf >
                        !hmatel = h2b_ovvo(l,c,f,k)
                        hmatel = h2b_ovvo(l,f,c,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,5,6/), nua, nua, noa, noa, nloc, n3aaa_r)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                        ! compute < ijk~abc~ | h2b(ovvo) | lijaeb >
                        !hmatel = -h2b_ovvo(l,c,e,k)
                        hmatel = -h2b_ovvo(l,e,c,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,5,6/), nua, nua, noa, noa, nloc, n3aaa_r)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < ijk~abc~ | h2b(ovvo) | lijdab >
                        !hmatel = h2b_ovvo(l,c,d,k)
                        hmatel = h2b_ovvo(l,d,c,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate temporary arrays
                  deallocate(amps_buff,excits_buff) 
                  !!!! diagram 13b: x2b(mcek)*t3a(abeijm) !!!!
                  ! allocate and initialize the copy of t3a
                  allocate(amps_buff(n3aaa_t))
                  allocate(excits_buff(6,n3aaa_t))
                  amps_buff(:) = t3a_amps(:)
                  excits_buff(:,:) = t3a_excits(:,:)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = (nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,noa,noa))
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,5/), nua, nua, noa, noa, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x2b_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     idx = idx_table(a,b,i,j)
                     if (idx==0) cycle 
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijk~abc~ | h2b(ovvo) | ijnabf >
                        !hmatel = x2b_ovvo(n,c,f,k)
                        hmatel = x2b_ovvo(n,f,c,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,5/), nua, nua, noa, noa, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x2b_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                        ! compute < ijk~abc~ | h2b(ovvo) | ijnaeb >
                        !hmatel = -x2b_ovvo(n,c,e,k)
                        hmatel = -x2b_ovvo(n,e,c,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCIJ LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,5/), nua, nua, noa, noa, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x2b_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); n = excits_buff(6,jdet);
                        ! compute < ijk~abc~ | h2b(ovvo) | ijndab >
                        !hmatel = x2b_ovvo(n,c,d,k)
                        hmatel = x2b_ovvo(n,d,c,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,6/), nua, nua, noa, noa, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x2b_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                        ! compute < ijk~abc~ | h2b(ovvo) | imjabf >
                        !hmatel = -x2b_ovvo(m,c,f,k)
                        hmatel = -x2b_ovvo(m,f,c,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,6/), nua, nua, noa, noa, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x2b_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                        ! compute < ijk~abc~ | h2b(ovvo) | imjaeb >
                        !hmatel = x2b_ovvo(m,c,e,k)
                        hmatel = x2b_ovvo(m,e,c,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,6/), nua, nua, noa, noa, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x2b_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                        ! compute < ijk~abc~ | h2b(ovvo) | imjdab >
                        !hmatel = -x2b_ovvo(m,c,d,k)
                        hmatel = -x2b_ovvo(m,d,c,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,5,6/), nua, nua, noa, noa, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x2b_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); l = excits_buff(4,jdet);
                        ! compute < ijk~abc~ | h2b(ovvo) | lijabf >
                        !hmatel = x2b_ovvo(l,c,f,k)
                        hmatel = x2b_ovvo(l,f,c,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,5,6/), nua, nua, noa, noa, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x2b_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                        ! compute < ijk~abc~ | h2b(ovvo) | lijaeb >
                        !hmatel = -x2b_ovvo(l,c,e,k)
                        hmatel = -x2b_ovvo(l,e,c,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do 
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,5,6/), nua, nua, noa, noa, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x2b_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     idx = idx_table(a,b,i,j) 
                     if (idx==0) cycle
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < ijk~abc~ | h2b(ovvo) | lijdab >
                        !hmatel = x2b_ovvo(l,c,d,k)
                        hmatel = x2b_ovvo(l,d,c,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate temporary arrays
                  deallocate(amps_buff,excits_buff)
                  !call cpu_time(toc)
                  !print*, "R3B - diagram 13 = ", toc - tic

                  !call cpu_time(tic)
                  !!!! diagram 14a: A(ab)A(ij) h2b(bmje)*r3c(aecimk)
                  ! allocate and initialize the copy of r3c
                  allocate(amps_buff(n3abb_r))
                  allocate(excits_buff(6,n3abb_r))
                  amps_buff(:) = r3c_amps(:)
                  excits_buff(:,:) = r3c_excits(:,:)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = nua*nub*noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nub,noa,nob))
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/2,nub/), (/1,noa/), (/2,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,6/), nua, nub, noa, nob, nloc, n3abb_r)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | im~k~ae~c~ >
                           !hmatel = h2b_voov(b,m,j,e)
                           hmatel = h2b_voov(m,e,b,j)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | im~k~be~c~ >
                           !hmatel = -h2b_voov(a,m,j,e)
                           hmatel = -h2b_voov(m,e,a,j)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | jm~k~ae~c~ >
                           !hmatel = -h2b_voov(b,m,i,e)
                           hmatel = -h2b_voov(m,e,b,i)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | jm~k~be~c~ >
                           !hmatel = h2b_voov(a,m,i,e)
                           hmatel = h2b_voov(m,e,a,i)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/1,nub-1/), (/1,noa/), (/2,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,6/), nua, nub, noa, nob, nloc, n3abb_r)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | im~k~ac~f~ >
                           !hmatel = -h2b_voov(b,m,j,f)
                           hmatel = -h2b_voov(m,f,b,j)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | im~k~bc~f~ >
                           !hmatel = h2b_voov(a,m,j,f)
                           hmatel = h2b_voov(m,f,a,j)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | jm~k~ac~f~ >
                           !hmatel = h2b_voov(b,m,i,f)
                           hmatel = h2b_voov(m,f,b,i)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | jm~k~bc~f~ >
                           !hmatel = -h2b_voov(a,m,i,f)
                           hmatel = -h2b_voov(m,f,a,i)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/2,nub/), (/1,noa/), (/1,nob-1/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,5/), nua, nub, noa, nob, nloc, n3abb_r)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | ik~n~ae~c~ >
                           !hmatel = -h2b_voov(b,n,j,e)
                           hmatel = -h2b_voov(n,e,b,j)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | ik~n~be~c~ >
                           !hmatel = h2b_voov(a,n,j,e)
                           hmatel = h2b_voov(n,e,a,j)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | jk~n~ae~c~ >
                           !hmatel = h2b_voov(b,n,i,e)
                           hmatel = h2b_voov(n,e,b,i)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | jk~n~be~c~ >
                           !hmatel = -h2b_voov(a,n,i,e)
                           hmatel = -h2b_voov(n,e,a,i)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/1,nub-1/), (/1,noa/), (/1,nob-1/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,5/), nua, nub, noa, nob, nloc, n3abb_r)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | ik~n~ac~f~ >
                           !hmatel = h2b_voov(b,n,j,f)
                           hmatel = h2b_voov(n,f,b,j)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | ik~n~bc~f~ >
                           !hmatel = -h2b_voov(a,n,j,f)
                           hmatel = -h2b_voov(n,f,a,j)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | jk~n~ac~f~ >
                           !hmatel = -h2b_voov(b,n,i,f)
                           hmatel = -h2b_voov(n,f,b,i)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | jk~n~bc~f~ >
                           !hmatel = h2b_voov(a,n,i,f)
                           hmatel = h2b_voov(n,f,a,i)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate temporary arrays
                  deallocate(amps_buff,excits_buff) 
                  !!!! diagram 14b: A(ab)A(ij) x2b(bmje)*t3c(aecimk)
                  ! allocate and initialize the copy of t3c
                  allocate(amps_buff(n3abb_t))
                  allocate(excits_buff(6,n3abb_t))
                  amps_buff(:) = t3c_amps(:)
                  excits_buff(:,:) = t3c_excits(:,:)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = nua*nub*noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nub,noa,nob))
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/2,nub/), (/1,noa/), (/2,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,6/), nua, nub, noa, nob, nloc, n3abb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | im~k~ae~c~ >
                           !hmatel = x2b_voov(b,m,j,e)
                           hmatel = x2b_voov(m,e,b,j)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | im~k~be~c~ >
                           !hmatel = -x2b_voov(a,m,j,e)
                           hmatel = -x2b_voov(m,e,a,j)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | jm~k~ae~c~ >
                           !hmatel = -x2b_voov(b,m,i,e)
                           hmatel = -x2b_voov(m,e,b,i)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | jm~k~be~c~ >
                           !hmatel = x2b_voov(a,m,i,e)
                           hmatel = x2b_voov(m,e,a,i)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/1,nub-1/), (/1,noa/), (/2,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,6/), nua, nub, noa, nob, nloc, n3abb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x2b_voov,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | im~k~ac~f~ >
                           !hmatel = -x2b_voov(b,m,j,f)
                           hmatel = -x2b_voov(m,f,b,j)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | im~k~bc~f~ >
                           !hmatel = x2b_voov(a,m,j,f)
                           hmatel = x2b_voov(m,f,a,j)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | jm~k~ac~f~ >
                           !hmatel = x2b_voov(b,m,i,f)
                           hmatel = x2b_voov(m,f,b,i)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | jm~k~bc~f~ >
                           !hmatel = -x2b_voov(a,m,i,f)
                           hmatel = -x2b_voov(m,f,a,i)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/2,nub/), (/1,noa/), (/1,nob-1/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,5/), nua, nub, noa, nob, nloc, n3abb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x2b_voov,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | ik~n~ae~c~ >
                           !hmatel = -x2b_voov(b,n,j,e)
                           hmatel = -x2b_voov(n,e,b,j)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | ik~n~be~c~ >
                           !hmatel = x2b_voov(a,n,j,e)
                           hmatel = x2b_voov(n,e,a,j)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | jk~n~ae~c~ >
                           !hmatel = x2b_voov(b,n,i,e)
                           hmatel = x2b_voov(n,e,b,i)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | jk~n~be~c~ >
                           !hmatel = -x2b_voov(a,n,i,e)
                           hmatel = -x2b_voov(n,e,a,i)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/1,nub-1/), (/1,noa/), (/1,nob-1/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,5/), nua, nub, noa, nob, nloc, n3abb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x2b_voov,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                     a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                     i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | ik~n~ac~f~ >
                           !hmatel = x2b_voov(b,n,j,f)
                           hmatel = x2b_voov(n,f,b,j)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | ik~n~bc~f~ >
                           !hmatel = -x2b_voov(a,n,j,f)
                           hmatel = -x2b_voov(n,f,a,j)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | jk~n~ac~f~ >
                           !hmatel = -x2b_voov(b,n,i,f)
                           hmatel = -x2b_voov(n,f,b,i)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k) 
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                           ! compute < ijk~abc~ | h2b(voov) | jk~n~bc~f~ >
                           !hmatel = x2b_voov(a,n,i,f)
                           hmatel = x2b_voov(n,f,a,i)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate temporary arrays
                  deallocate(amps_buff,excits_buff) 
                  !call cpu_time(toc)
                  !print*, "R3B - diagram 14 = ", toc - tic

                  !call cpu_time(tic)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp t2a,t2b,r2a,r2b,&
                  !$omp h2a_vvov,h2a_vooo,h2b_vvvo,h2b_vvov,h2b_vooo,h2b_ovoo,&
                  !$omp x2a_vvov,x2a_vooo,x2b_vvvo,x2b_vvov,x2b_vooo,x2b_ovoo,&
                  !$omp noa,nua,nob,nub,n3aab_r),&
                  !$omp private(idet,a,b,c,d,i,j,k,l,m,n,e,f,res_mm23)
                  !$omp do schedule(static)
                  do idet = 1, n3aab_r
                      a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                      i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);

                      res_mm23 = 0.0d0
                      do e = 1, nua
                          ! A(ab) h2b(bcek) * r2a(aeij)
                          !res_mm23 = res_mm23 + h2b_vvvo(b,c,e,k) * r2a(a,e,i,j)
                          res_mm23 = res_mm23 + h2b_vvvo(e,b,c,k) * r2a(a,e,i,j)
                          !res_mm23 = res_mm23 - h2b_vvvo(a,c,e,k) * r2a(b,e,i,j)
                          res_mm23 = res_mm23 - h2b_vvvo(e,a,c,k) * r2a(b,e,i,j)
                          ! A(ij) h2a(abie) * r2b(ecjk)
                          !res_mm23 = res_mm23 + h2a_vvov(a,b,i,e) * r2b(e,c,j,k)
                          res_mm23 = res_mm23 + h2a_vvov(e,a,b,i) * r2b(e,c,j,k)
                          !res_mm23 = res_mm23 - h2a_vvov(a,b,j,e) * r2b(e,c,i,k)
                          res_mm23 = res_mm23 - h2a_vvov(e,a,b,j) * r2b(e,c,i,k)
                          ! A(ab) x2b(bcek) * t2a(aeij)
                          !res_mm23 = res_mm23 + x2b_vvvo(b,c,e,k) * t2a(a,e,i,j)
                          res_mm23 = res_mm23 + x2b_vvvo(e,b,c,k) * t2a(a,e,i,j)
                          !res_mm23 = res_mm23 - x2b_vvvo(a,c,e,k) * t2a(b,e,i,j)
                          res_mm23 = res_mm23 - x2b_vvvo(e,a,c,k) * t2a(b,e,i,j)
                          ! A(ij) x2a(abie) * t2b(ecjk)
                          !res_mm23 = res_mm23 + x2a_vvov(a,b,i,e) * t2b(e,c,j,k)
                          res_mm23 = res_mm23 + x2a_vvov(e,a,b,i) * t2b(e,c,j,k)
                          !res_mm23 = res_mm23 - x2a_vvov(a,b,j,e) * t2b(e,c,i,k)
                          res_mm23 = res_mm23 - x2a_vvov(e,a,b,j) * t2b(e,c,i,k)
                      end do
                      do e = 1, nub
                          ! A(ij)A(ab) h2b(acie) * r2b(bejk)
                          !res_mm23 = res_mm23 + h2b_vvov(a,c,i,e) * r2b(b,e,j,k)
                          res_mm23 = res_mm23 + h2b_vvov(e,a,c,i) * r2b(b,e,j,k)
                          !res_mm23 = res_mm23 - h2b_vvov(a,c,j,e) * r2b(b,e,i,k)
                          res_mm23 = res_mm23 - h2b_vvov(e,a,c,j) * r2b(b,e,i,k)
                          !res_mm23 = res_mm23 - h2b_vvov(b,c,i,e) * r2b(a,e,j,k)
                          res_mm23 = res_mm23 - h2b_vvov(e,b,c,i) * r2b(a,e,j,k)
                          !res_mm23 = res_mm23 + h2b_vvov(b,c,j,e) * r2b(a,e,i,k)
                          res_mm23 = res_mm23 + h2b_vvov(e,b,c,j) * r2b(a,e,i,k)
                          ! A(ij)A(ab) x2b(acie) * t2b(bejk)
                          !res_mm23 = res_mm23 + x2b_vvov(a,c,i,e) * t2b(b,e,j,k)
                          res_mm23 = res_mm23 + x2b_vvov(e,a,c,i) * t2b(b,e,j,k)
                          !res_mm23 = res_mm23 - x2b_vvov(a,c,j,e) * t2b(b,e,i,k)
                          res_mm23 = res_mm23 - x2b_vvov(e,a,c,j) * t2b(b,e,i,k)
                          !res_mm23 = res_mm23 - x2b_vvov(b,c,i,e) * t2b(a,e,j,k)
                          res_mm23 = res_mm23 - x2b_vvov(e,b,c,i) * t2b(a,e,j,k)
                          !res_mm23 = res_mm23 + x2b_vvov(b,c,j,e) * t2b(a,e,i,k)
                          res_mm23 = res_mm23 + x2b_vvov(e,b,c,j) * t2b(a,e,i,k)
                      end do
                      do m = 1, noa
                          ! -A(ij) h2b(mcjk) * r2a(abim) 
                          res_mm23 = res_mm23 - h2b_ovoo(m,c,j,k) * r2a(a,b,i,m)
                          res_mm23 = res_mm23 + h2b_ovoo(m,c,i,k) * r2a(a,b,j,m)
                          ! -A(ab) h2a(amij) * r2b(bcmk)
                          res_mm23 = res_mm23 - h2a_vooo(a,m,i,j) * r2b(b,c,m,k)
                          res_mm23 = res_mm23 + h2a_vooo(b,m,i,j) * r2b(a,c,m,k)
                          ! -A(ij) x2b(mcjk) * t2a(abim) 
                          res_mm23 = res_mm23 - x2b_ovoo(m,c,j,k) * t2a(a,b,i,m)
                          res_mm23 = res_mm23 + x2b_ovoo(m,c,i,k) * t2a(a,b,j,m)
                          ! -A(ab) x2a(amij) * t2b(bcmk)
                          res_mm23 = res_mm23 - x2a_vooo(a,m,i,j) * t2b(b,c,m,k)
                          res_mm23 = res_mm23 + x2a_vooo(b,m,i,j) * t2b(a,c,m,k)
                      end do
                      do m = 1, nob
                          ! -A(ij)A(ab) h2b(amik) * r2b(bcjm)
                          res_mm23 = res_mm23 - h2b_vooo(a,m,i,k) * r2b(b,c,j,m)
                          res_mm23 = res_mm23 + h2b_vooo(b,m,i,k) * r2b(a,c,j,m)
                          res_mm23 = res_mm23 + h2b_vooo(a,m,j,k) * r2b(b,c,i,m)
                          res_mm23 = res_mm23 - h2b_vooo(b,m,j,k) * r2b(a,c,i,m)
                          ! -A(ij)A(ab) x2b(amik) * t2b(bcjm)
                          res_mm23 = res_mm23 - x2b_vooo(a,m,i,k) * t2b(b,c,j,m)
                          res_mm23 = res_mm23 + x2b_vooo(b,m,i,k) * t2b(a,c,j,m)
                          res_mm23 = res_mm23 + x2b_vooo(a,m,j,k) * t2b(b,c,i,m)
                          res_mm23 = res_mm23 - x2b_vooo(b,m,j,k) * t2b(a,c,i,m)
                      end do
                      resid(idet) = resid(idet) + res_mm23
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !call cpu_time(toc)
                  !print*, "R3B - moments = ", toc - tic

              end subroutine build_hr_3b

              subroutine build_hr_3c(resid,&
                                     r2b,r2c,&
                                     r3b_amps, r3b_excits,&
                                     r3c_amps, r3c_excits,&
                                     r3d_amps, r3d_excits,&
                                     t2b,t2c,&
                                     t3b_amps, t3b_excits,&
                                     t3c_amps, t3c_excits,&
                                     t3d_amps, t3d_excits,&
                                     h1a_oo, h1a_vv, h1b_oo, h1b_vv,&
                                     h2a_oovv, h2a_voov,&
                                     h2b_oooo, h2b_vooo, h2b_ovoo,&
                                     h2b_oovv, h2b_voov, h2b_vovo,&
                                     h2b_ovov, h2b_ovvo, h2b_vvov,&
                                     h2b_vvvo, h2b_vvvv,&
                                     h2c_oooo, h2c_vooo, h2c_oovv,&
                                     h2c_voov, h2c_vvov, h2c_vvvv,&
                                     x1a_oo, x1a_vv, x1b_oo, x1b_vv,&
                                     x2a_oovv, x2a_voov,&
                                     x2b_oooo, x2b_vooo, x2b_ovoo,&
                                     x2b_oovv, x2b_voov, x2b_vovo,&
                                     x2b_ovov, x2b_ovvo, x2b_vvov,&
                                     x2b_vvvo, x2b_vvvv,&
                                     x2c_oooo, x2c_vooo, x2c_oovv,&
                                     x2c_voov, x2c_vvov, x2c_vvvv,&
                                     n3aab_r, n3abb_r, n3bbb_r,&
                                     n3aab_t, n3abb_t, n3bbb_t,&
                                     noa, nua, nob, nub)
                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab_r, n3aab_t 
                  integer, intent(in) :: n3abb_r, n3abb_t
                  integer, intent(in) :: n3bbb_r, n3bbb_t
                  ! Input R and T arrays
                  real(kind=8), intent(in) :: r2b(nua,nub,noa,nob), t2b(nua,nub,noa,nob)
                  real(kind=8), intent(in) :: r2c(nub,nub,nob,nob), t2c(nub,nub,nob,nob)
                  integer, intent(in) :: r3b_excits(6,n3aab_r), t3b_excits(6,n3aab_t)
                  integer, intent(in) :: r3d_excits(6,n3bbb_r), t3d_excits(6,n3bbb_t)
                  integer, intent(in) :: t3c_excits(6,n3abb_t)
                  real(kind=8), intent(in) :: r3b_amps(n3aab_r), t3b_amps(n3aab_t)
                  real(kind=8), intent(in) :: r3d_amps(n3bbb_r), t3d_amps(n3bbb_t)
                  real(kind=8), intent(in) :: t3c_amps(n3abb_t)
                  ! Input H arrays
                  real(kind=8), intent(in) :: h1a_oo(noa,noa), h1b_oo(nob,nob)
                  real(kind=8), intent(in) :: h1a_vv(nua,nua), h1b_vv(nub,nub)
                  real(kind=8), intent(in) :: h2a_oovv(noa,noa,nua,nua)
                  real(kind=8), intent(in) :: h2a_voov(nua,noa,noa,nua)
                  real(kind=8), intent(in) :: h2b_oooo(noa,nob,noa,nob)
                  real(kind=8), intent(in) :: h2b_vooo(nua,nob,noa,nob)
                  real(kind=8), intent(in) :: h2b_ovoo(noa,nub,noa,nob)
                  real(kind=8), intent(in) :: h2b_oovv(noa,nob,nua,nub)
                  real(kind=8), intent(in) :: h2b_voov(nua,nob,noa,nub)
                  real(kind=8), intent(in) :: h2b_vovo(nua,nob,nua,nob)
                  real(kind=8), intent(in) :: h2b_ovov(noa,nub,noa,nub)
                  real(kind=8), intent(in) :: h2b_ovvo(noa,nub,nua,nob)
                  real(kind=8), intent(in) :: h2b_vvov(nua,nub,noa,nub)
                  real(kind=8), intent(in) :: h2b_vvvo(nua,nub,nua,nob)
                  real(kind=8), intent(in) :: h2b_vvvv(nua,nub,nua,nub)
                  real(kind=8), intent(in) :: h2c_oooo(nob,nob,nob,nob)
                  real(kind=8), intent(in) :: h2c_vooo(nub,nob,nob,nob)
                  real(kind=8), intent(in) :: h2c_oovv(nob,nob,nub,nub)
                  real(kind=8), intent(in) :: h2c_voov(nub,nob,nob,nub)
                  real(kind=8), intent(in) :: h2c_vvov(nub,nub,nob,nub)
                  real(kind=8), intent(in) :: h2c_vvvv(nub,nub,nub,nub)
                  ! Input X arrays
                  real(kind=8), intent(in) :: x1a_oo(noa,noa), x1b_oo(nob,nob)
                  real(kind=8), intent(in) :: x1a_vv(nua,nua), x1b_vv(nub,nub)
                  real(kind=8), intent(in) :: x2a_oovv(noa,noa,nua,nua)
                  real(kind=8), intent(in) :: x2a_voov(nua,noa,noa,nua)
                  real(kind=8), intent(in) :: x2b_oooo(noa,nob,noa,nob)
                  real(kind=8), intent(in) :: x2b_vooo(nua,nob,noa,nob)
                  real(kind=8), intent(in) :: x2b_ovoo(noa,nub,noa,nob)
                  real(kind=8), intent(in) :: x2b_oovv(noa,nob,nua,nub)
                  real(kind=8), intent(in) :: x2b_voov(nua,nob,noa,nub)
                  real(kind=8), intent(in) :: x2b_vovo(nua,nob,nua,nob)
                  real(kind=8), intent(in) :: x2b_ovov(noa,nub,noa,nub)
                  real(kind=8), intent(in) :: x2b_ovvo(noa,nub,nua,nob)
                  real(kind=8), intent(in) :: x2b_vvov(nua,nub,noa,nub)
                  real(kind=8), intent(in) :: x2b_vvvo(nua,nub,nua,nob)
                  real(kind=8), intent(in) :: x2b_vvvv(nua,nub,nua,nub)
                  real(kind=8), intent(in) :: x2c_oooo(nob,nob,nob,nob)
                  real(kind=8), intent(in) :: x2c_vooo(nub,nob,nob,nob)
                  real(kind=8), intent(in) :: x2c_oovv(nob,nob,nub,nub)
                  real(kind=8), intent(in) :: x2c_voov(nub,nob,nob,nub)
                  real(kind=8), intent(in) :: x2c_vvov(nub,nub,nob,nub)
                  real(kind=8), intent(in) :: x2c_vvvv(nub,nub,nub,nub)
                  ! Output and Inout variables
                  real(kind=8), intent(out) :: resid(n3abb_r)
                  integer, intent(inout) :: r3c_excits(6,n3abb_r)
                  !f2py intent(in,out) :: r3c_excits(6,0:n3abb_r-1)
                  real(kind=8), intent(inout) :: r3c_amps(n3abb_r)
                  !f2py intent(in,out) :: r3c_amps(0:n3abb_r-1)
                  ! Local variables
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:)
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)
                  real(kind=8) :: t_amp, r_amp, hmatel, hmatel1, res_mm23
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc
                  
                  ! Zero the container that holds H*R
                  resid = 0.0d0

                  !!!! diagram 1a: -A(jk) h1b(mk)*r3c(abcijm)
                  !!!! diagram 5a: A(jk) 1/2 h2c(mnjk)*r3c(abcimn)
                  !!! BCAI LOOP !!!
                  ! allocate new sorting arrays
                  nloc = nub*(nub-1)/2*nua*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nua,noa))
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nua/), (/1,noa/), nub, nub, nua, nob)
                  call sort4(r3c_excits, r3c_amps, loc_arr, idx_table, (/2,3,1,4/), nub, nub, nua, nob, nloc, n3abb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1B_oo,H2C_oooo,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                     a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                     i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                     idx = idx_table(b,c,a,i)
                     ! (1)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r3c_excits(5,jdet); n = r3c_excits(6,jdet);
                        ! compute < ij~k~ab~c~ | h2c(oooo) | im~n~ab~c~ >
                        hmatel = h2c_oooo(m,n,j,k)
                        ! compute < ij~k~ab~c~ | h1b(oo) | im~n~ab~c~ > = -A(jk)A(mn) h1b_oo(m,j) * delta(n,k)
                        if (n==k) hmatel = hmatel - h1b_oo(m,j) ! (1)
                        if (n==j) hmatel = hmatel + h1b_oo(m,k) ! (jk)
                        if (m==k) hmatel = hmatel + h1b_oo(n,j) ! (mn)
                        if (m==j) hmatel = hmatel - h1b_oo(n,k) ! (jk)(mn)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  !!!! diagram 1b: -A(jk) x1b(mk)*t3c(abcijm)
                  !!!! diagram 5b: A(jk) 1/2 x2c(mnjk)*t3c(abcimn)
                  !!! BCAI LOOP !!!
                  ! allocate temporary arrays
                  allocate(excits_buff(6,n3abb_t),amps_buff(n3abb_t))
                  excits_buff(:,:) = t3c_excits(:,:)
                  amps_buff(:) = t3c_amps(:)
                  ! allocate new sorting arrays
                  nloc = nub*(nub-1)/2*nua*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nua,noa))
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nua/), (/1,noa/), nub, nub, nua, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,1,4/), nub, nub, nua, nob, nloc, n3abb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x1b_oo,x2c_oooo,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                     a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                     i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                     idx = idx_table(b,c,a,i)
                     ! (1)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = excits_buff(5,jdet); n = excits_buff(6,jdet);
                        ! compute < ij~k~ab~c~ | h2c(oooo) | im~n~ab~c~ >
                        hmatel = x2c_oooo(m,n,j,k)
                        ! compute < ij~k~ab~c~ | h1b(oo) | im~n~ab~c~ > = -A(jk)A(mn) h1b_oo(m,j) * delta(n,k)
                        if (n==k) hmatel = hmatel - x1b_oo(m,j) ! (1)
                        if (n==j) hmatel = hmatel + x1b_oo(m,k) ! (jk)
                        if (m==k) hmatel = hmatel + x1b_oo(n,j) ! (mn)
                        if (m==j) hmatel = hmatel - x1b_oo(n,k) ! (jk)(mn)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate temporary arrays
                  deallocate(excits_buff,amps_buff)

                  !!!! diagram 2a: A(bc) h1b(ce)*r3c(abeijk)
                  !!!! diagram 6a: A(bc) 1/2 h2c(bcef)*r3c(aefijk)
                  !!! JKIA LOOP !!!
                  ! allocate new sorting arrays
                  nloc = nua*nob*(nob-1)/2*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nob,nob,noa,nua))
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,noa/), (/1,nua/), nob, nob, noa, nua)
                  call sort4(r3c_excits, r3c_amps, loc_arr, idx_table, (/5,6,4,1/), nob, nob, noa, nua, nloc, n3abb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1B_vv,H2C_vvvv,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                     a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                     i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                     idx = idx_table(j,k,i,a)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3c_excits(2,jdet); f = r3c_excits(3,jdet);
                        ! compute < ij~k~ab~c~ | h2c(vvvv) | ij~k~ae~f~ >
                        !hmatel = h2c_vvvv(b,c,e,f)
                        hmatel = h2c_vvvv(e,f,b,c)
                        ! compute < ij~k~ab~c~ | h2c(vvvv) | ij~k~ae~f~ > = A(bc)A(ef) h1b_vv(b,e) * delta(c,f)
                        if (c==f) hmatel = hmatel + h1b_vv(b,e) ! (1)
                        if (b==f) hmatel = hmatel - h1b_vv(c,e) ! (bc)
                        if (c==e) hmatel = hmatel - h1b_vv(b,f) ! (ef)
                        if (b==e) hmatel = hmatel + h1b_vv(c,f) ! (bc)(ef)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  !!!! diagram 2b: A(bc) x1b(ce)*t3c(abeijk)
                  !!!! diagram 6b: A(bc) 1/2 x2c(bcef)*t3c(aefijk)
                  !!! JKIA LOOP !!!
                  ! allocate temporary arrays
                  allocate(excits_buff(6,n3abb_t),amps_buff(n3abb_t))
                  excits_buff(:,:) = t3c_excits(:,:)
                  amps_buff(:) = t3c_amps(:)
                  ! allocate new sorting arrays
                  nloc = nua*nob*(nob-1)/2*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nob,nob,noa,nua))
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,noa/), (/1,nua/), nob, nob, noa, nua)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/5,6,4,1/), nob, nob, noa, nua, nloc, n3abb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp x1B_vv,x2C_vvvv,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                     a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                     i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                     idx = idx_table(j,k,i,a)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); f = excits_buff(3,jdet);
                        ! compute < ij~k~ab~c~ | x2c(vvvv) | ij~k~ae~f~ >
                        !hmatel = x2c_vvvv(b,c,e,f)
                        hmatel = x2c_vvvv(e,f,b,c)
                        ! compute < ij~k~ab~c~ | x2c(vvvv) | ij~k~ae~f~ > = A(bc)A(ef) x1b_vv(b,e) * delta(c,f)
                        if (c==f) hmatel = hmatel + x1b_vv(b,e) ! (1)
                        if (b==f) hmatel = hmatel - x1b_vv(c,e) ! (bc)
                        if (c==e) hmatel = hmatel - x1b_vv(b,f) ! (ef)
                        if (b==e) hmatel = hmatel + x1b_vv(c,f) ! (bc)(ef)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate temporary arrays
                  deallocate(excits_buff,amps_buff)

                  !!!! diagram 3a: -h1a(mi)*r3c(abcmjk)
                  !!!! diagram 7a: A(jk) h2b(mnij)*r3c(abcmnk)
                  !!! BCAK LOOP !!!
                  ! allocate new sorting arrays
                  nloc = nub*(nub-1)/2*nua*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nua,nob))
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nua/), (/2,nob/), nub, nub, nua, nob)
                  call sort4(r3c_excits, r3c_amps, loc_arr, idx_table, (/2,3,1,6/), nub, nub, nua, nob, nloc, n3abb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_oo,H2B_oooo,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                     a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                     i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,a,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        l = r3c_excits(4,jdet); m = r3c_excits(5,jdet);
                        ! compute < ij~k~ab~c~ | h2b(oooo) | lm~k~ab~c~ >
                        hmatel = h2b_oooo(l,m,i,j)
                        ! compute < ij~k~ab~c~ | h1a(oo) | lm~k~ab~c~ >
                        if (m==j) hmatel = hmatel - h1a_oo(l,i)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table(b,c,a,j)
                     if (idx/=0) then
                         do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            l = r3c_excits(4,jdet); m = r3c_excits(5,jdet);
                            ! compute < ij~k~ab~c~ | h2b(oooo) | lm~j~ab~c~ >
                            hmatel = -h2b_oooo(l,m,i,k)
                            ! compute < ij~k~ab~c~ | h1a(oo) | lm~j~ab~c~ >
                            if (m==k) hmatel = hmatel + h1a_oo(l,i)
                            resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                         end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCAJ LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nua/), (/1,nob-1/), nub, nub, nua, nob)
                  call sort4(r3c_excits, r3c_amps, loc_arr, idx_table, (/2,3,1,5/), nub, nub, nua, nob, nloc, n3abb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_oooo,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                     a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                     i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,a,j)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        l = r3c_excits(4,jdet); n = r3c_excits(6,jdet);
                        ! compute < ij~k~ab~c~ | h2b(oooo) | lj~n~ab~c~ >
                        hmatel = h2b_oooo(l,n,i,k)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table(b,c,a,k)
                     if (idx/=0) then
                         do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            l = r3c_excits(4,jdet); n = r3c_excits(6,jdet);
                            ! compute < ij~k~ab~c~ | h2b(oooo) | lk~n~ab~c~ >
                            hmatel = -h2b_oooo(l,n,i,j)
                            resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                         end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECITON !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  !!!! diagram 3b: -x1a(mi)*t3c(abcmjk)
                  !!!! diagram 7b: A(jk) x2b(mnij)*t3c(abcmnk)
                  !!! BCAK LOOP !!!
                  ! allocate temporary arrays
                  allocate(excits_buff(6,n3abb_t),amps_buff(n3abb_t))
                  excits_buff(:,:) = t3c_excits(:,:)
                  amps_buff(:) = t3c_amps(:)
                  ! allocate new sorting arrays
                  nloc = nub*(nub-1)/2*nua*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nua,nob))
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nua/), (/2,nob/), nub, nub, nua, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,1,6/), nub, nub, nua, nob, nloc, n3abb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X1A_oo,X2B_oooo,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                     a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                     i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,a,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        l = excits_buff(4,jdet); m = excits_buff(5,jdet);
                        ! compute < ij~k~ab~c~ | x2b(oooo) | lm~k~ab~c~ >
                        hmatel = x2b_oooo(l,m,i,j)
                        ! compute < ij~k~ab~c~ | x1a(oo) | lm~k~ab~c~ >
                        if (m==j) hmatel = hmatel - x1a_oo(l,i)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (jk)
                     idx = idx_table(b,c,a,j)
                     if (idx/=0) then
                         do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            l = excits_buff(4,jdet); m = excits_buff(5,jdet);
                            ! compute < ij~k~ab~c~ | x2b(oooo) | lm~j~ab~c~ >
                            hmatel = -x2b_oooo(l,m,i,k)
                            ! compute < ij~k~ab~c~ | x1a(oo) | lm~j~ab~c~ >
                            if (m==k) hmatel = hmatel + x1a_oo(l,i)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                         end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCAJ LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nua/), (/1,nob-1/), nub, nub, nua, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,1,5/), nub, nub, nua, nob, nloc, n3abb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2B_oooo,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                     a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                     i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,a,j)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        l = excits_buff(4,jdet); n = excits_buff(6,jdet);
                        ! compute < ij~k~ab~c~ | x2b(oooo) | lj~n~ab~c~ >
                        hmatel = x2b_oooo(l,n,i,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (jk)
                     idx = idx_table(b,c,a,k)
                     if (idx/=0) then
                         do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            l = excits_buff(4,jdet); n = excits_buff(6,jdet);
                            ! compute < ij~k~ab~c~ | x2b(oooo) | lk~n~ab~c~ >
                            hmatel = -x2b_oooo(l,n,i,j)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                         end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECITON !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate temporary arrays
                  deallocate(excits_buff,amps_buff)

                  !!!! diagram 5a: h1a(ae)*r3c(ebcijk)
                  !!!! diagram 8a: A(bc) h2b(abef)*r3c(efcijk)
                  ! allocate new sorting arrays
                  nloc = nub*nob*(nob-1)/2*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nob,nob,noa,nub))
                  !!! JKIB LOOP !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,noa/), (/1,nub-1/), nob, nob, noa, nub)
                  call sort4(r3c_excits, r3c_amps, loc_arr, idx_table, (/5,6,4,2/), nob, nob, noa, nub, nloc, n3abb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_vv,H2B_vvvv,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(j,k,i,b)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                         d = r3c_excits(1,jdet); f = r3c_excits(3,jdet);
                         ! compute < ij~k~ab~c~ | h2b(vvvv) | ij~k~db~f~ >
                         !hmatel = h2b_vvvv(a,c,d,f)
                         hmatel = h2b_vvvv(d,f,a,c)
                         if (c==f) hmatel = hmatel + h1a_vv(a,d)
                         resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                      end do
                      ! (bc)
                      idx = idx_table(j,k,i,c)
                      if (idx/=0) then ! protect against case where b = nua because a = 1, nua-1
                         do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = r3c_excits(1,jdet); f = r3c_excits(3,jdet);
                            ! compute < ij~k~ab~c~ | h2b(vvvv) | ij~k~dc~f~ >
                            !hmatel = -h2b_vvvv(a,b,d,f)
                            hmatel = -h2b_vvvv(d,f,a,b)
                            if (b==f) hmatel = hmatel - h1a_vv(a,d)
                            resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                         end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! JKIC LOOP !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,noa/), (/2,nub/), nob, nob, noa, nub)
                  call sort4(r3c_excits, r3c_amps, loc_arr, idx_table, (/5,6,4,3/), nob, nob, noa, nub, nloc, n3abb_r, resid)
                  !!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_vvvv,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      idx = idx_table(j,k,i,c)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                         d = r3c_excits(1,jdet); e = r3c_excits(2,jdet);
                         ! compute < ij~k~ab~c~ | h2b(vvvv) | ij~k~de~c~ >
                         !hmatel = h2b_vvvv(a,b,d,e)
                         hmatel = h2b_vvvv(d,e,a,b)
                         resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                      end do
                      ! (bc)
                      idx = idx_table(j,k,i,b)
                      if (idx/=0) then ! protect against case where a = 1 because b = 2, nua
                         do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = r3c_excits(1,jdet); e = r3c_excits(2,jdet);
                            ! compute < ij~k~ab~c~ | h2b(vvvv) | ij~k~de~b~ >
                            !hmatel = -h2b_vvvv(a,c,d,e)
                            hmatel = -h2b_vvvv(d,e,a,c)
                            resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                         end do
                      end if
                  end do ! end loop over idet
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  !!!! diagram 5b: x1a(ae)*t3c(ebcijk)
                  !!!! diagram 8b: A(bc) x2b(abef)*t3c(efcijk)
                  ! allocate temporary arrays
                  allocate(excits_buff(6,n3abb_t),amps_buff(n3abb_t))
                  excits_buff(:,:) = t3c_excits(:,:)
                  amps_buff(:) = t3c_amps(:)
                  ! allocate new sorting arrays
                  nloc = nub*nob*(nob-1)/2*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nob,nob,noa,nub))
                  !!! JKIB LOOP !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,noa/), (/1,nub-1/), nob, nob, noa, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/5,6,4,2/), nob, nob, noa, nub, nloc, n3abb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X1A_vv,X2B_vvvv,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(j,k,i,b)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                         d = excits_buff(1,jdet); f = excits_buff(3,jdet);
                         ! compute < ij~k~ab~c~ | x2b(vvvv) | ij~k~db~f~ >
                         !hmatel = x2b_vvvv(a,c,d,f)
                         hmatel = x2b_vvvv(d,f,a,c)
                         if (c==f) hmatel = hmatel + x1a_vv(a,d)
                         resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                      end do
                      ! (bc)
                      idx = idx_table(j,k,i,c)
                      if (idx/=0) then ! protect against case where b = nua because a = 1, nua-1
                         do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = excits_buff(1,jdet); f = excits_buff(3,jdet);
                            ! compute < ij~k~ab~c~ | x2b(vvvv) | ij~k~dc~f~ >
                            !hmatel = -x2b_vvvv(a,b,d,f)
                            hmatel = -x2b_vvvv(d,f,a,b)
                            if (b==f) hmatel = hmatel - x1a_vv(a,d)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                         end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! JKIC LOOP !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,noa/), (/2,nub/), nob, nob, noa, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/5,6,4,3/), nob, nob, noa, nub, nloc, n3abb_t)
                  !!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2B_vvvv,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      idx = idx_table(j,k,i,c)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                         d = excits_buff(1,jdet); e = excits_buff(2,jdet);
                         ! compute < ij~k~ab~c~ | x2b(vvvv) | ij~k~de~c~ >
                         !hmatel = x2b_vvvv(a,b,d,e)
                         hmatel = x2b_vvvv(d,e,a,b)
                         resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                      end do
                      ! (bc)
                      idx = idx_table(j,k,i,b)
                      if (idx/=0) then ! protect against case where a = 1 because b = 2, nua
                         do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = excits_buff(1,jdet); e = excits_buff(2,jdet);
                            ! compute < ij~k~ab~c~ | x2b(vvvv) | ij~k~de~b~ >
                            !hmatel = -x2b_vvvv(a,c,d,e)
                            hmatel = -x2b_vvvv(d,e,a,c)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                         end do
                      end if
                  end do ! end loop over idet
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate temporary arrays
                  deallocate(excits_buff,amps_buff)

                  !!!! diagram 9a: A(jk)A(bc) h2c(cmke)*r3c(abeijm)
                  ! allocate new sorting arrays
                  nloc = nub*nua*nob*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nub,noa,nob))
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/1,nub-1/), (/1,noa/), (/1,nob-1/), nua, nub, noa, nob)
                  call sort4(r3c_excits, r3c_amps, loc_arr, idx_table, (/1,2,4,5/), nua, nub, noa, nob, nloc, n3abb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                     a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                     i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,i,j)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3c_excits(3,jdet); n = r3c_excits(6,jdet);
                        ! compute < ij~k~ab~c~ | h2a(voov) | ij~n~ab~f~ >
                        hmatel = h2c_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                         do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            f = r3c_excits(3,jdet); n = r3c_excits(6,jdet);
                            ! compute < ij~k~ab~c~ | h2a(voov) | ik~n~ab~f~ >
                            hmatel = -h2c_voov(c,n,j,f)
                            resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                         end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                         do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            f = r3c_excits(3,jdet); n = r3c_excits(6,jdet);
                            ! compute < ij~k~ab~c~ | h2a(voov) | ij~n~ac~f~ >
                            hmatel = -h2c_voov(b,n,k,f)
                            resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                         end do
                     end if
                     ! (jk)(bc)
                      idx = idx_table(a,c,i,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                             f = r3c_excits(3,jdet); n = r3c_excits(6,jdet);
                             ! compute < ij~k~ab~c~ | h2a(voov) | ik~n~ac~f~ >
                             hmatel = h2c_voov(b,n,j,f)
                             resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/2,nub/), (/1,noa/), (/1,nob-1/), nua, nub, noa, nob)
                  call sort4(r3c_excits, r3c_amps, loc_arr, idx_table, (/1,3,4,5/), nua, nub, noa, nob, nloc, n3abb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(a,c,i,j)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                          e = r3c_excits(2,jdet); n = r3c_excits(6,jdet);
                          ! compute < ij~k~ab~c~ | h2c(voov) | ij~n~ae~c~ >
                          hmatel = h2c_voov(b,n,k,e)
                          resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                      end do
                      ! (jk)
                      idx = idx_table(a,c,i,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = r3c_excits(2,jdet); n = r3c_excits(6,jdet);
                              ! compute < ij~k~ab~c~ | h2c(voov) | ik~n~ae~c~ >
                              hmatel = -h2c_voov(b,n,j,e)
                              resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                          end do
                      end if
                      ! (bc)
                      idx = idx_table(a,b,i,j)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = r3c_excits(2,jdet); n = r3c_excits(6,jdet);
                              ! compute < ij~k~ab~c~ | h2c(voov) | ij~n~ae~b~ >
                              hmatel = -h2c_voov(c,n,k,e)
                              resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                          end do
                      end if
                      ! (jk)(bc)
                      idx = idx_table(a,b,i,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = r3c_excits(2,jdet); n = r3c_excits(6,jdet);
                              ! compute < ij~k~ab~c~ | h2c(voov) | ik~n~ae~b~ >
                              hmatel = h2c_voov(c,n,j,e)
                              resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/1,nub-1/), (/1,noa/), (/2,nob/), nua, nub, noa, nob)
                  call sort4(r3c_excits, r3c_amps, loc_arr, idx_table, (/1,2,4,6/), nua, nub, noa, nob, nloc, n3abb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(a,b,i,k)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                          f = r3c_excits(3,jdet); m = r3c_excits(5,jdet);
                          ! compute < ij~k~ab~c~ | h2c(voov) | im~k~ab~f~ >
                          hmatel = h2c_voov(c,m,j,f)
                          resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                      end do
                      ! (jk)
                      idx = idx_table(a,b,i,j)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              f = r3c_excits(3,jdet); m = r3c_excits(5,jdet);
                              ! compute < ij~k~ab~c~ | h2c(voov) | im~j~ab~f~ >
                              hmatel = -h2c_voov(c,m,k,f)
                              resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                          end do
                      end if
                      ! (bc)
                      idx = idx_table(a,c,i,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              f = r3c_excits(3,jdet); m = r3c_excits(5,jdet);
                              ! compute < ij~k~ab~c~ | h2c(voov) | im~k~ac~f~ >
                              hmatel = -h2c_voov(b,m,j,f)
                              resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                          end do
                      end if
                      ! (jk)(bc)
                      idx = idx_table(a,c,i,j)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              f = r3c_excits(3,jdet); m = r3c_excits(5,jdet);
                              ! compute < ij~k~ab~c~ | h2c(voov) | im~j~ac~f~ >
                              hmatel = h2c_voov(b,m,k,f)
                              resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/2,nub/), (/1,noa/), (/2,nob/), nua, nub, noa, nob)
                  call sort4(r3c_excits, r3c_amps, loc_arr, idx_table, (/1,3,4,6/), nua, nub, noa, nob, nloc, n3abb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(a,c,i,k)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                          e = r3c_excits(2,jdet); m = r3c_excits(5,jdet);
                          ! compute < ij~k~ab~c~ | h2c(voov) | im~k~ae~c~ >
                          hmatel = h2c_voov(b,m,j,e)
                          resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                      end do
                      ! (jk)
                      idx = idx_table(a,c,i,j)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = r3c_excits(2,jdet); m = r3c_excits(5,jdet);
                              ! compute < ij~k~ab~c~ | h2c(voov) | im~j~ae~c~ >
                              hmatel = -h2c_voov(b,m,k,e)
                              resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                          end do
                      end if
                      ! (bc)
                      idx = idx_table(a,b,i,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = r3c_excits(2,jdet); m = r3c_excits(5,jdet);
                              ! compute < ij~k~ab~c~ | h2c(voov) | im~k~ae~b~ >
                              hmatel = -h2c_voov(c,m,j,e)
                              resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                          end do
                      end if
                      ! (jk)(bc)
                      idx = idx_table(a,b,i,j)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = r3c_excits(2,jdet); m = r3c_excits(5,jdet);
                              ! compute < ij~k~ab~c~ | h2c(voov) | im~j~ae~b~ >
                              hmatel = h2c_voov(c,m,k,e)
                              resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  !!!! diagram 9b: A(jk)A(bc) x2c(cmke)*t3c(abeijm)
                  ! allocate temporary arrays
                  allocate(excits_buff(6,n3abb_t),amps_buff(n3abb_t))
                  excits_buff(:,:) = t3c_excits(:,:)
                  amps_buff(:) = t3c_amps(:)
                  ! allocate new sorting arrays
                  nloc = nub*nua*nob*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nub,noa,nob))
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/1,nub-1/), (/1,noa/), (/1,nob-1/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,5/), nua, nub, noa, nob, nloc, n3abb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2C_voov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                     a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                     i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,i,j)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ij~k~ab~c~ | x2a(voov) | ij~n~ab~f~ >
                        hmatel = x2c_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                         do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                            ! compute < ij~k~ab~c~ | x2a(voov) | ik~n~ab~f~ >
                            hmatel = -x2c_voov(c,n,j,f)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                         end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                         do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                            ! compute < ij~k~ab~c~ | x2a(voov) | ij~n~ac~f~ >
                            hmatel = -x2c_voov(b,n,k,f)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                         end do
                     end if
                     ! (jk)(bc)
                      idx = idx_table(a,c,i,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                             f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                             ! compute < ij~k~ab~c~ | x2a(voov) | ik~n~ac~f~ >
                             hmatel = x2c_voov(b,n,j,f)
                             resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/2,nub/), (/1,noa/), (/1,nob-1/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,5/), nua, nub, noa, nob, nloc, n3abb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2C_voov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(a,c,i,j)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                          e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                          ! compute < ij~k~ab~c~ | x2c(voov) | ij~n~ae~c~ >
                          hmatel = x2c_voov(b,n,k,e)
                          resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                      end do
                      ! (jk)
                      idx = idx_table(a,c,i,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                              ! compute < ij~k~ab~c~ | x2c(voov) | ik~n~ae~c~ >
                              hmatel = -x2c_voov(b,n,j,e)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                      ! (bc)
                      idx = idx_table(a,b,i,j)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                              ! compute < ij~k~ab~c~ | x2c(voov) | ij~n~ae~b~ >
                              hmatel = -x2c_voov(c,n,k,e)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                      ! (jk)(bc)
                      idx = idx_table(a,b,i,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                              ! compute < ij~k~ab~c~ | x2c(voov) | ik~n~ae~b~ >
                              hmatel = x2c_voov(c,n,j,e)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/1,nub-1/), (/1,noa/), (/2,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,6/), nua, nub, noa, nob, nloc, n3abb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2C_voov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(a,b,i,k)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                          f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                          ! compute < ij~k~ab~c~ | x2c(voov) | im~k~ab~f~ >
                          hmatel = x2c_voov(c,m,j,f)
                          resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                      end do
                      ! (jk)
                      idx = idx_table(a,b,i,j)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                              ! compute < ij~k~ab~c~ | x2c(voov) | im~j~ab~f~ >
                              hmatel = -x2c_voov(c,m,k,f)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                      ! (bc)
                      idx = idx_table(a,c,i,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                              ! compute < ij~k~ab~c~ | x2c(voov) | im~k~ac~f~ >
                              hmatel = -x2c_voov(b,m,j,f)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                      ! (jk)(bc)
                      idx = idx_table(a,c,i,j)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                              ! compute < ij~k~ab~c~ | x2c(voov) | im~j~ac~f~ >
                              hmatel = x2c_voov(b,m,k,f)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/2,nub/), (/1,noa/), (/2,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,6/), nua, nub, noa, nob, nloc, n3abb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2C_voov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(a,c,i,k)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                          e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                          ! compute < ij~k~ab~c~ | x2c(voov) | im~k~ae~c~ >
                          hmatel = x2c_voov(b,m,j,e)
                          resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                      end do
                      ! (jk)
                      idx = idx_table(a,c,i,j)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                              ! compute < ij~k~ab~c~ | x2c(voov) | im~j~ae~c~ >
                              hmatel = -x2c_voov(b,m,k,e)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                      ! (bc)
                      idx = idx_table(a,b,i,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                              ! compute < ij~k~ab~c~ | x2c(voov) | im~k~ae~b~ >
                              hmatel = -x2c_voov(c,m,j,e)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                      ! (jk)(bc)
                      idx = idx_table(a,b,i,j)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                              ! compute < ij~k~ab~c~ | x2c(voov) | im~j~ae~b~ >
                              hmatel = x2c_voov(c,m,k,e)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate sorting arrays
                  deallocate(excits_buff,amps_buff)

                  !!!! diagram 10a: h2a(amie)*r3c(ebcmjk)
                  ! allocate sorting arrays
                  nloc = nub*(nub-1)/2*nob*(nob-1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nob,nob))
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(r3c_excits, r3c_amps, loc_arr, idx_table, (/2,3,5,6/), nub, nub, nob, nob, nloc, n3abb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,&
                  !$omp n3abb_r),&
                  !$omp private(hmatel,a,b,c,i,j,k,f,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      idx = idx_table(b,c,j,k)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                         d = r3c_excits(1,jdet); l = r3c_excits(4,jdet);
                         ! compute < ij~k~ab~c~ | h2a(voov) | lj~k~db~c~ >
                         hmatel = h2a_voov(a,l,i,d)
                         resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                      end do
                  end do ! end loop over idet
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  !!!! diagram 10b: x2a(amie)*t3c(ebcmjk)
                  ! allocate temporary arrays
                  allocate(excits_buff(6,n3abb_t),amps_buff(n3abb_t))
                  excits_buff(:,:) = t3c_excits(:,:)
                  amps_buff(:) = t3c_amps(:)
                  ! allocate sorting arrays
                  nloc = nub*(nub-1)/2*nob*(nob-1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nob,nob))
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,5,6/), nub, nub, nob, nob, nloc, n3abb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2A_voov,&
                  !$omp noa,nua,nob,nub,&
                  !$omp n3abb_r),&
                  !$omp private(hmatel,a,b,c,i,j,k,f,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      idx = idx_table(b,c,j,k)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                         d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                         ! compute < ij~k~ab~c~ | h2a(voov) | lj~k~db~c~ >
                         hmatel = x2a_voov(a,l,i,d)
                         resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                      end do
                  end do ! end loop over idet
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate temporary arrays
                  deallocate(excits_buff,amps_buff)

                  !!!! diagram 11a: -A(bc) h2b(mbie)*r3c(aecmjk)
                  ! allocate sorting arrays
                  nloc = nob*(nob-1)/2*nub*nua
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nob,nob,nua,nub))
                  !!! JKAC LOOP !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,nua/), (/2,nub/), nob, nob, nua, nub)
                  call sort4(r3c_excits, r3c_amps, loc_arr, idx_table, (/5,6,1,3/), nob, nob, nua, nub, nloc, n3abb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(j,k,a,c)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                          e = r3c_excits(2,jdet); l = r3c_excits(4,jdet);
                          ! compute < ij~k~ab~c~ | h2b(ovov) | lj~k~ae~c~ >
                          hmatel = -h2b_ovov(l,b,i,e)
                          resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                      end do
                      ! (bc)
                      idx = idx_table(j,k,a,b)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = r3c_excits(2,jdet); l = r3c_excits(4,jdet);
                              ! compute < ij~k~ab~c~ | h2b(ovov) | lj~k~ae~b~ >
                              hmatel = h2b_ovov(l,c,i,e)
                              resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! JKAB LOOP !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,nua/), (/1,nub-1/), nob, nob, nua, nub)
                  call sort4(r3c_excits, r3c_amps, loc_arr, idx_table, (/5,6,1,2/), nob, nob, nua, nub, nloc, n3abb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(j,k,a,b)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                          f = r3c_excits(3,jdet); l = r3c_excits(4,jdet);
                          ! compute < ij~k~ab~c~ | h2b(ovov) | lj~k~ab~f~ >
                          hmatel = -h2b_ovov(l,c,i,f)
                          resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                      end do
                      ! (bc)
                      idx = idx_table(j,k,a,c)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              f = r3c_excits(3,jdet); l = r3c_excits(4,jdet);
                              ! compute < ij~k~ab~c~ | h2b(ovov) | lj~k~ac~f~ >
                              hmatel = h2b_ovov(l,b,i,f)
                              resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  !!!! diagram 11b: -A(bc) x2b(mbie)*t3c(aecmjk)
                  ! allocate temporary arrays
                  allocate(excits_buff(6,n3abb_t),amps_buff(n3abb_t))
                  excits_buff(:,:) = t3c_excits(:,:)
                  amps_buff(:) = t3c_amps(:)
                  ! allocate sorting arrays
                  nloc = nob*(nob-1)/2*nub*nua
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nob,nob,nua,nub))
                  !!! JKAC LOOP !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,nua/), (/2,nub/), nob, nob, nua, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/5,6,1,3/), nob, nob, nua, nub, nloc, n3abb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2B_ovov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(j,k,a,c)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                          e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                          ! compute < ij~k~ab~c~ | x2b(ovov) | lj~k~ae~c~ >
                          hmatel = -x2b_ovov(l,b,i,e)
                          resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                      end do
                      ! (bc)
                      idx = idx_table(j,k,a,b)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                              ! compute < ij~k~ab~c~ | x2b(ovov) | lj~k~ae~b~ >
                              hmatel = x2b_ovov(l,c,i,e)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! JKAB LOOP !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,nua/), (/1,nub-1/), nob, nob, nua, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/5,6,1,2/), nob, nob, nua, nub, nloc, n3abb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2B_ovov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(j,k,a,b)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                          f = excits_buff(3,jdet); l = excits_buff(4,jdet);
                          ! compute < ij~k~ab~c~ | x2b(ovov) | lj~k~ab~f~ >
                          hmatel = -x2b_ovov(l,c,i,f)
                          resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                      end do
                      ! (bc)
                      idx = idx_table(j,k,a,c)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              f = excits_buff(3,jdet); l = excits_buff(4,jdet);
                              ! compute < ij~k~ab~c~ | x2b(ovov) | lj~k~ac~f~ >
                              hmatel = x2b_ovov(l,b,i,f)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate temporary arrays
                  deallocate(excits_buff,amps_buff)
                  
                  !!!! diagram 12a: -A(bc) h2b(amej)*r3c(ebcimk)
                  ! allocate sorting arrays
                  nloc = nub*(nub-1)/2*noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,noa,nob))
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,noa/), (/2,nob/), nub, nub, noa, nob)
                  call sort4(r3c_excits, r3c_amps, loc_arr, idx_table, (/2,3,4,6/), nub, nub, noa, nob, nloc, n3abb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_vovo,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,i,k)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                          d = r3c_excits(1,jdet); m = r3c_excits(5,jdet);
                          ! compute < ij~k~ab~c~ | h2b(vovo) | im~k~db~c~ >
                          hmatel = -h2b_vovo(a,m,d,j)
                          resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                      end do
                      ! (jk)
                      idx = idx_table(b,c,i,j)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              d = r3c_excits(1,jdet); m = r3c_excits(5,jdet);
                              ! compute < ij~k~ab~c~ | h2b(vovo) | im~j~db~c~ >
                              hmatel = h2b_vovo(a,m,d,k)
                              resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,noa/), (/1,nob-1/), nub, nub, noa, nob)
                  call sort4(r3c_excits, r3c_amps, loc_arr, idx_table, (/2,3,4,5/), nub, nub, noa, nob, nloc, n3abb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_vovo,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,i,j)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                          d = r3c_excits(1,jdet); n = r3c_excits(6,jdet);
                          ! compute < ij~k~ab~c~ | h2b(vovo) | ij~n~db~c~ >
                          hmatel = -h2b_vovo(a,n,d,k)
                          resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                      end do
                      ! (jk)
                      idx = idx_table(b,c,i,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              d = r3c_excits(1,jdet); n = r3c_excits(6,jdet);
                              ! compute < ij~k~ab~c~ | h2b(vovo) | ik~n~db~c~ >
                              hmatel = h2b_vovo(a,n,d,j)
                              resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  !!!! diagram 12b: -A(bc) x2b(amej)*t3c(ebcimk)
                  ! allocate temporary arrays
                  allocate(excits_buff(6,n3abb_t),amps_buff(n3abb_t))
                  excits_buff(:,:) = t3c_excits(:,:)
                  amps_buff(:) = t3c_amps(:)
                  ! allocate sorting arrays
                  nloc = nub*(nub-1)/2*noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,noa,nob))
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,noa/), (/2,nob/), nub, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,6/), nub, nub, noa, nob, nloc, n3abb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2B_vovo,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,i,k)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                          d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                          ! compute < ij~k~ab~c~ | x2b(vovo) | im~k~db~c~ >
                          hmatel = -x2b_vovo(a,m,d,j)
                          resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                      end do
                      ! (jk)
                      idx = idx_table(b,c,i,j)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                              ! compute < ij~k~ab~c~ | x2b(vovo) | im~j~db~c~ >
                              hmatel = x2b_vovo(a,m,d,k)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,noa/), (/1,nob-1/), nub, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,5/), nub, nub, noa, nob, nloc, n3abb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2B_vovo,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,i,j)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                          d = excits_buff(1,jdet); n = excits_buff(6,jdet);
                          ! compute < ij~k~ab~c~ | x2b(vovo) | ij~n~db~c~ >
                          hmatel = -x2b_vovo(a,n,d,k)
                          resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                      end do
                      ! (jk)
                      idx = idx_table(b,c,i,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              d = excits_buff(1,jdet); n = excits_buff(6,jdet);
                              ! compute < ij~k~ab~c~ | x2b(vovo) | ik~n~db~c~ >
                              hmatel = x2b_vovo(a,n,d,j)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate temporary arrays
                  deallocate(excits_buff,amps_buff)

                  !!!! diagram 13a: h2b(amie)*r3d(ebcmjk)
                  ! allocate and initialize the copy of r3d
                  allocate(amps_buff(n3bbb_r))
                  allocate(excits_buff(6,n3bbb_r))
                  amps_buff(:) = r3d_amps(:)
                  excits_buff(:,:) = r3d_excits(:,:)
                  ! allocate sorting arrays
                  nloc = (nub-1)*(nub-2)/2*(nob-1)*(nob-2)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nob,nob))
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,5,6/), nub, nub, nob, nob, nloc, n3bbb_r)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                              ! compute < ij~k~ab~c~ | h2b(voov) | l~j~k~d~b~c~ >
                              hmatel = h2b_voov(a,l,i,d)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,6/), nub, nub, nob, nob, nloc, n3bbb_r)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                              ! compute < ij~k~ab~c~ | h2b(voov) | j~m~k~d~b~c~ >
                              hmatel = -h2b_voov(a,m,i,d)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCIJ LOOP !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,5/), nub, nub, nob, nob, nloc, n3bbb_r)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              d = excits_buff(1,jdet); n = excits_buff(6,jdet);
                              ! compute < ij~k~ab~c~ | h2b(voov) | j~k~n~d~b~c~ >
                              hmatel = h2b_voov(a,n,i,d)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACJK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,5,6/), nub, nub, nob, nob, nloc, n3bbb_r)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                              ! compute < ij~k~ab~c~ | h2b(voov) | l~j~k~b~e~c~ >
                              hmatel = -h2b_voov(a,l,i,e)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,6/), nub, nub, nob, nob, nloc, n3bbb_r)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                              ! compute < ij~k~ab~c~ | h2b(voov) | j~m~k~b~e~c~ >
                              hmatel = h2b_voov(a,m,i,e)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,5/), nub, nub, nob, nob, nloc, n3bbb_r)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                              ! compute < ij~k~ab~c~ | h2b(voov) | j~k~n~b~e~c~ >
                              hmatel = -h2b_voov(a,n,i,e)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABJK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,5,6/), nub, nub, nob, nob, nloc, n3bbb_r)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              f = excits_buff(3,jdet); l = excits_buff(4,jdet);
                              ! compute < ij~k~ab~c~ | h2b(voov) | l~j~k~b~c~f~ >
                              hmatel = h2b_voov(a,l,i,f)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,6/), nub, nub, nob, nob, nloc, n3bbb_r)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                              ! compute < ij~k~ab~c~ | h2b(voov) | j~m~k~b~c~f~ >
                              hmatel = -h2b_voov(a,m,i,f)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,5/), nub, nub, nob, nob, nloc, n3bbb_r)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                              ! compute < ij~k~ab~c~ | h2b(voov) | j~k~n~b~c~f~ >
                              hmatel = h2b_voov(a,n,i,f)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate r3 buffer arrays
                  deallocate(amps_buff,excits_buff)
                  !!!! diagram 13b: x2b(amie)*t3d(ebcmjk)
                  ! allocate and initialize the copy of t3d
                  allocate(amps_buff(n3bbb_t))
                  allocate(excits_buff(6,n3bbb_t))
                  amps_buff(:) = t3d_amps(:)
                  excits_buff(:,:) = t3d_excits(:,:)
                  ! allocate sorting arrays
                  nloc = (nub-1)*(nub-2)/2*(nob-1)*(nob-2)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nob,nob))
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,5,6/), nub, nub, nob, nob, nloc, n3bbb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2B_voov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                              ! compute < ij~k~ab~c~ | x2b(voov) | l~j~k~d~b~c~ >
                              hmatel = x2b_voov(a,l,i,d)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,6/), nub, nub, nob, nob, nloc, n3bbb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2B_voov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                              ! compute < ij~k~ab~c~ | x2b(voov) | j~m~k~d~b~c~ >
                              hmatel = -x2b_voov(a,m,i,d)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCIJ LOOP !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,5/), nub, nub, nob, nob, nloc, n3bbb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2B_voov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              d = excits_buff(1,jdet); n = excits_buff(6,jdet);
                              ! compute < ij~k~ab~c~ | x2b(voov) | j~k~n~d~b~c~ >
                              hmatel = x2b_voov(a,n,i,d)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACJK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,5,6/), nub, nub, nob, nob, nloc, n3bbb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2B_voov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                              ! compute < ij~k~ab~c~ | x2b(voov) | l~j~k~b~e~c~ >
                              hmatel = -x2b_voov(a,l,i,e)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,6/), nub, nub, nob, nob, nloc, n3bbb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2B_voov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                              ! compute < ij~k~ab~c~ | x2b(voov) | j~m~k~b~e~c~ >
                              hmatel = x2b_voov(a,m,i,e)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,5/), nub, nub, nob, nob, nloc, n3bbb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2B_voov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                              ! compute < ij~k~ab~c~ | x2b(voov) | j~k~n~b~e~c~ >
                              hmatel = -x2b_voov(a,n,i,e)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABJK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,5,6/), nub, nub, nob, nob, nloc, n3bbb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2B_voov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              f = excits_buff(3,jdet); l = excits_buff(4,jdet);
                              ! compute < ij~k~ab~c~ | x2b(voov) | l~j~k~b~c~f~ >
                              hmatel = x2b_voov(a,l,i,f)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,6/), nub, nub, nob, nob, nloc, n3bbb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2B_voov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                              ! compute < ij~k~ab~c~ | x2b(voov) | j~m~k~b~c~f~ >
                              hmatel = -x2b_voov(a,m,i,f)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,5/), nub, nub, nob, nob, nloc, n3bbb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2B_voov,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                              ! compute < ij~k~ab~c~ | x2b(voov) | j~k~n~b~c~f~ >
                              hmatel = x2b_voov(a,n,i,f)
                              resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                          end do
                      end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate t3 buffer arrays
                  deallocate(amps_buff,excits_buff)

                  !!!! diagram 14a: A(bc)A(jk) h2b(mbej)*r3b(aecimk)
                  ! allocate and initialize the copy of r3b
                  allocate(amps_buff(n3aab_r))
                  allocate(excits_buff(6,n3aab_r))
                  amps_buff(:) = r3b_amps(:)
                  excits_buff(:,:) = r3b_excits(:,:)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = nua*nub*noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nub,noa,nob))
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/1,nub/), (/1,noa-1/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,6/), nua, nub, noa, nob, nloc, n3aab_r)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                     a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                     i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | imk~aec~ >
                            hmatel = h2b_ovvo(m,b,e,j)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | imk~aeb~ >
                            hmatel = -h2b_ovvo(m,c,e,j)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | imj~aec~ >
                            hmatel = -h2b_ovvo(m,b,e,k)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | imj~aeb~ >
                            hmatel = h2b_ovvo(m,c,e,k)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/2,nua/), (/1,nub/), (/1,noa-1/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,6/), nua, nub, noa, nob, nloc, n3aab_r)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                     a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                     i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | imk~dac~ >
                            hmatel = -h2b_ovvo(m,b,d,j)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | imk~dab~ >
                            hmatel = h2b_ovvo(m,c,d,j)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | imj~dac~ >
                            hmatel = h2b_ovvo(m,b,d,k)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | imj~dab~ >
                            hmatel = -h2b_ovvo(m,c,d,k)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/1,nub/), (/2,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,5,6/), nua, nub, noa, nob, nloc, n3aab_r)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                     a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                     i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | lik~aec~ >
                            hmatel = -h2b_ovvo(l,b,e,j)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | lik~aeb~ >
                            hmatel = h2b_ovvo(l,c,e,j)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | lij~aec~ >
                            hmatel = h2b_ovvo(l,b,e,k)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | lij~aeb~ >
                            hmatel = -h2b_ovvo(l,c,e,k)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/2,nua/), (/1,nub/), (/2,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,5,6/), nua, nub, noa, nob, nloc, n3aab_r)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                     a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                     i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | lik~dac~ >
                            hmatel = h2b_ovvo(l,b,d,j)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | lik~dab~ >
                            hmatel = -h2b_ovvo(l,c,d,j)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | lij~dac~ >
                            hmatel = -h2b_ovvo(l,b,d,k)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | lij~dab~ >
                            hmatel = h2b_ovvo(l,c,d,k)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate r3 buffer arrays
                  deallocate(amps_buff,excits_buff)
                  !!!! diagram 14b: A(bc)A(jk) x2b(mbej)*t3b(aecimk)
                  ! allocate and initialize the copy of t3b
                  allocate(amps_buff(n3aab_t))
                  allocate(excits_buff(6,n3aab_t))
                  amps_buff(:) = t3b_amps(:)
                  excits_buff(:,:) = t3b_excits(:,:)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = nua*nub*noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nub,noa,nob))
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/1,nub/), (/1,noa-1/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,6/), nua, nub, noa, nob, nloc, n3aab_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                     a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                     i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                            ! compute < ij~k~ab~c~ | x2b(ovvo) | imk~aec~ >
                            hmatel = x2b_ovvo(m,b,e,j)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                            ! compute < ij~k~ab~c~ | x2b(ovvo) | imk~aeb~ >
                            hmatel = -x2b_ovvo(m,c,e,j)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                            ! compute < ij~k~ab~c~ | x2b(ovvo) | imj~aec~ >
                            hmatel = -x2b_ovvo(m,b,e,k)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                            ! compute < ij~k~ab~c~ | x2b(ovvo) | imj~aeb~ >
                            hmatel = x2b_ovvo(m,c,e,k)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/2,nua/), (/1,nub/), (/1,noa-1/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,6/), nua, nub, noa, nob, nloc, n3aab_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                     a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                     i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                            ! compute < ij~k~ab~c~ | x2b(ovvo) | imk~dac~ >
                            hmatel = -x2b_ovvo(m,b,d,j)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                            ! compute < ij~k~ab~c~ | x2b(ovvo) | imk~dab~ >
                            hmatel = x2b_ovvo(m,c,d,j)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                            ! compute < ij~k~ab~c~ | x2b(ovvo) | imj~dac~ >
                            hmatel = x2b_ovvo(m,b,d,k)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                            ! compute < ij~k~ab~c~ | x2b(ovvo) | imj~dab~ >
                            hmatel = -x2b_ovvo(m,c,d,k)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/1,nub/), (/2,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,5,6/), nua, nub, noa, nob, nloc, n3aab_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                     a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                     i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                            ! compute < ij~k~ab~c~ | x2b(ovvo) | lik~aec~ >
                            hmatel = -x2b_ovvo(l,b,e,j)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                            ! compute < ij~k~ab~c~ | x2b(ovvo) | lik~aeb~ >
                            hmatel = x2b_ovvo(l,c,e,j)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                            ! compute < ij~k~ab~c~ | x2b(ovvo) | lij~aec~ >
                            hmatel = x2b_ovvo(l,b,e,k)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                            ! compute < ij~k~ab~c~ | x2b(ovvo) | lij~aeb~ >
                            hmatel = -x2b_ovvo(l,c,e,k)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/2,nua/), (/1,nub/), (/2,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,5,6/), nua, nub, noa, nob, nloc, n3aab_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                     a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                     i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                            ! compute < ij~k~ab~c~ | x2b(ovvo) | lik~dac~ >
                            hmatel = x2b_ovvo(l,b,d,j)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                            ! compute < ij~k~ab~c~ | x2b(ovvo) | lik~dab~ >
                            hmatel = -x2b_ovvo(l,c,d,j)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                            ! compute < ij~k~ab~c~ | x2b(ovvo) | lij~dac~ >
                            hmatel = -x2b_ovvo(l,b,d,k)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                            ! compute < ij~k~ab~c~ | x2b(ovvo) | lij~dab~ >
                            hmatel = x2b_ovvo(l,c,d,k)
                            resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate t3 buffer arrays
                  deallocate(amps_buff,excits_buff)

                  
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp t2b,t2c,r2b,r2c,&
                  !$omp h2c_vvov,h2c_vooo,h2b_vvvo,h2b_vvov,h2b_vooo,h2b_ovoo,&
                  !$omp x2c_vvov,x2c_vooo,x2b_vvvo,x2b_vvov,x2b_vooo,x2b_ovoo,&
                  !$omp noa,nua,nob,nub,n3abb_r),&
                  !$omp private(idet,a,b,c,d,i,j,k,l,m,n,e,f,res_mm23)
                  !$omp do schedule(static)
                  do idet = 1, n3abb_r
                      a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                      i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);
                      res_mm23 = 0.0
                      do e = 1, nua
                          ! A(jk)A(bc) h2b(abej) * r2b(ecik)
                          res_mm23 = res_mm23 + h2b_vvvo(a,b,e,j) * r2b(e,c,i,k)
                          res_mm23 = res_mm23 - h2b_vvvo(a,b,e,k) * r2b(e,c,i,j)
                          res_mm23 = res_mm23 - h2b_vvvo(a,c,e,j) * r2b(e,b,i,k)
                          res_mm23 = res_mm23 + h2b_vvvo(a,c,e,k) * r2b(e,b,i,j)
                          ! A(jk)A(bc) x2b(abej) * t2b(ecik)
                          res_mm23 = res_mm23 + x2b_vvvo(a,b,e,j) * t2b(e,c,i,k)
                          res_mm23 = res_mm23 - x2b_vvvo(a,b,e,k) * t2b(e,c,i,j)
                          res_mm23 = res_mm23 - x2b_vvvo(a,c,e,j) * t2b(e,b,i,k)
                          res_mm23 = res_mm23 + x2b_vvvo(a,c,e,k) * t2b(e,b,i,j)
                      end do
                      do e = 1, nub
                          ! A(bc) h2B(abie) * r2c(ecjk)
                          res_mm23 = res_mm23 + h2B_vvov(a,b,i,e) * r2c(e,c,j,k)
                          res_mm23 = res_mm23 - h2B_vvov(a,c,i,e) * r2c(e,b,j,k)
                          ! A(jk) h2C(cbke) * r2b(aeij)
                          res_mm23 = res_mm23 + h2C_vvov(c,b,k,e) * r2b(a,e,i,j)
                          res_mm23 = res_mm23 - h2C_vvov(c,b,j,e) * r2b(a,e,i,k)
                          ! A(bc) x2B(abie) * t2c(ecjk)
                          res_mm23 = res_mm23 + x2B_vvov(a,b,i,e) * t2c(e,c,j,k)
                          res_mm23 = res_mm23 - x2B_vvov(a,c,i,e) * t2c(e,b,j,k)
                          ! A(jk) x2C(cbke) * t2b(aeij)
                          res_mm23 = res_mm23 + x2C_vvov(c,b,k,e) * t2b(a,e,i,j)
                          res_mm23 = res_mm23 - x2C_vvov(c,b,j,e) * t2b(a,e,i,k)
                      end do
                      do m = 1, noa
                          ! -A(kj)A(bc) h2b(mbij) * r2b(acmk)
                          res_mm23 = res_mm23 - h2B_ovoo(m,b,i,j) * r2b(a,c,m,k)
                          res_mm23 = res_mm23 + h2B_ovoo(m,c,i,j) * r2b(a,b,m,k)
                          res_mm23 = res_mm23 + h2B_ovoo(m,b,i,k) * r2b(a,c,m,j)
                          res_mm23 = res_mm23 - h2B_ovoo(m,c,i,k) * r2b(a,b,m,j)
                          ! -A(kj)A(bc) x2b(mbij) * t2b(acmk)
                          res_mm23 = res_mm23 - x2B_ovoo(m,b,i,j) * t2b(a,c,m,k)
                          res_mm23 = res_mm23 + x2B_ovoo(m,c,i,j) * t2b(a,b,m,k)
                          res_mm23 = res_mm23 + x2B_ovoo(m,b,i,k) * t2b(a,c,m,j)
                          res_mm23 = res_mm23 - x2B_ovoo(m,c,i,k) * t2b(a,b,m,j)
                      end do
                      do m = 1, nob
                          ! -A(jk) h2b(amij) * r2c(bcmk)
                          res_mm23 = res_mm23 - h2B_vooo(a,m,i,j) * r2c(b,c,m,k)
                          res_mm23 = res_mm23 + h2B_vooo(a,m,i,k) * r2c(b,c,m,j)
                          ! -A(bc) h2c(cmkj) * r2b(abim)
                          res_mm23 = res_mm23 - h2C_vooo(c,m,k,j) * r2b(a,b,i,m)
                          res_mm23 = res_mm23 + h2C_vooo(b,m,k,j) * r2b(a,c,i,m)
                          ! -A(jk) x2b(amij) * t2c(bcmk)
                          res_mm23 = res_mm23 - x2B_vooo(a,m,i,j) * t2c(b,c,m,k)
                          res_mm23 = res_mm23 + x2B_vooo(a,m,i,k) * t2c(b,c,m,j)
                          ! -A(bc) x2c(cmkj) * t2b(abim)
                          res_mm23 = res_mm23 - x2C_vooo(c,m,k,j) * t2b(a,b,i,m)
                          res_mm23 = res_mm23 + x2C_vooo(b,m,k,j) * t2b(a,c,i,m)
                      end do
                      resid(idet) = resid(idet) + res_mm23
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
              end subroutine build_hr_3c

              subroutine build_hr_3d(resid,&
                                     r2c,&
                                     r3c_amps, r3c_excits,&
                                     r3d_amps, r3d_excits,&
                                     t2c,&
                                     t3c_amps, t3c_excits,&
                                     t3d_amps, t3d_excits,&
                                     h1b_oo, h1b_vv,&
                                     h2c_oooo, h2c_vooo, h2c_oovv,&
                                     h2c_voov, h2c_vvov, h2c_vvvv,&
                                     h2b_ovvo,&
                                     x1b_oo, x1b_vv,&
                                     x2c_oooo, x2c_vooo, x2c_oovv,&
                                     x2c_voov, x2c_vvov, x2c_vvvv,&
                                     x2b_ovvo,&
                                     n3abb_r, n3bbb_r,&
                                     n3abb_t, n3bbb_t,&
                                     noa, nua, nob, nub)
                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3abb_r, n3abb_t, n3bbb_r, n3bbb_t
                  ! Input R and T arrays
                  real(kind=8), intent(in) :: r2c(nub,nub,nob,nob), t2c(nub,nub,nob,nob)
                  integer, intent(in) :: r3c_excits(6,n3abb_r), t3c_excits(6,n3abb_t)
                  integer, intent(in) :: t3d_excits(6,n3bbb_t) 
                  real(kind=8), intent(in) :: r3c_amps(n3abb_r), t3c_amps(n3abb_t)
                  real(kind=8), intent(in) :: t3d_amps(n3bbb_t)
                  ! Input H and X arrays
                  real(kind=8), intent(in) :: h1b_oo(nob,nob)
                  real(kind=8), intent(in) :: h1b_vv(nub,nub)
                  real(kind=8), intent(in) :: h2c_oooo(nob,nob,nob,nob)
                  real(kind=8), intent(in) :: h2c_vooo(nub,nob,nob,nob)
                  real(kind=8), intent(in) :: h2c_oovv(nob,nob,nub,nub)
                  real(kind=8), intent(in) :: h2c_voov(nub,nob,nob,nub)
                  real(kind=8), intent(in) :: h2c_vvov(nub,nub,nob,nub)
                  real(kind=8), intent(in) :: h2c_vvvv(nub,nub,nub,nub)
                  real(kind=8), intent(in) :: h2b_ovvo(noa,nub,nua,nob)
                  real(kind=8), intent(in) :: x1b_oo(nob,nob)
                  real(kind=8), intent(in) :: x1b_vv(nub,nub)
                  real(kind=8), intent(in) :: x2c_oooo(nob,nob,nob,nob)
                  real(kind=8), intent(in) :: x2c_vooo(nub,nob,nob,nob)
                  real(kind=8), intent(in) :: x2c_oovv(nob,nob,nub,nub)
                  real(kind=8), intent(in) :: x2c_voov(nub,nob,nob,nub)
                  real(kind=8), intent(in) :: x2c_vvov(nub,nub,nob,nub)
                  real(kind=8), intent(in) :: x2c_vvvv(nub,nub,nub,nub)
                  real(kind=8), intent(in) :: x2b_ovvo(noa,nub,nua,nob)
                  ! Output and Inout variables
                  real(kind=8), intent(out) :: resid(n3bbb_r)
                  integer, intent(inout) :: r3d_excits(6,n3bbb_r)
                  !f2py intent(in,out) :: r3d_excits(6,0:n3bbb_r-1)
                  real(kind=8), intent(inout) :: r3d_amps(n3bbb_r)
                  !f2py intent(in,out) :: r3d_amps(0:n3bbb_r-1)
                  ! Local variables
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:)
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)
                  real(kind=8) :: l_amp, hmatel, hmatel1, res_mm23
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc
                  
                  ! Zero the container that holds H*R
                  resid = 0.0d0

                  !!!! diagram 1a: -A(i/jk) h1b(mi) * r3d(abcmjk)
                  !!!! diagram 3a: 1/2 A(i/jk) h2c(mnij) * r3d(abcmnk)
                  ! NOTE: WITHIN THESE LOOPS, H1B(OO) TERMS ARE DOUBLE-COUNTED SO COMPENSATE BY FACTOR OF 1/2
                  ! allocate new sorting arrays
                  nloc = nub*(nub-1)*(nub-2)/6*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nub,nob))
                  !!! ABCK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/-1,nub/), (/3,nob/), nub, nub, nub, nob)
                  call sort4(r3d_excits, r3d_amps, loc_arr, idx_table, (/1,2,3,6/), nub, nub, nub, nob, nloc, n3bbb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,&
                  !$omp r3d_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1B_oo,H2C_oooo,&
                  !$omp nob,nub,n3bbb_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        l = r3d_excits(4,jdet); m = r3d_excits(5,jdet);
                        ! compute < ijkabc | h2c(oooo) | lmkabc >
                        hmatel = h2c_oooo(l,m,i,j)
                        ! compute < ijkabc | h1b(oo) | lmkabc > = -A(ij)A(lm) h1b_oo(l,i) * delta(m,j)
                        hmatel1 = 0.0d0
                        if (m==j) hmatel1 = hmatel1 - h1b_oo(l,i) ! (1)      < ijkabc | h1b(oo) | ljkabc >
                        if (m==i) hmatel1 = hmatel1 + h1b_oo(l,j) ! (ij)     < ijkabc | h1b(oo) | likabc >
                        if (l==j) hmatel1 = hmatel1 + h1b_oo(m,i) ! (lm)     < ijkabc | h1b(oo) | jmkabc >
                        if (l==i) hmatel1 = hmatel1 - h1b_oo(m,j) ! (ij)(lm) < ijkabc | h1b(oo) | imkabc >
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     ! (ik)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = r3d_excits(4,jdet); m = r3d_excits(5,jdet);
                           ! compute < ijkabc | h2c(oooo) | lmiabc >
                           hmatel = -h2c_oooo(l,m,k,j)
                           ! compute < ijkabc | h1b(oo) | lmiabc > = A(jk)A(lm) h1b_oo(l,k) * delta(m,j)
                           hmatel1 = 0.0d0
                           if (m==j) hmatel1 = hmatel1 + h1b_oo(l,k) ! (1)      < ijkabc | h1b(oo) | ljiabc >
                           if (m==k) hmatel1 = hmatel1 - h1b_oo(l,j) ! (jk)     < ijkabc | h1b(oo) | lkiabc >
                           if (l==j) hmatel1 = hmatel1 - h1b_oo(m,k) ! (lm)
                           if (l==k) hmatel1 = hmatel1 + h1b_oo(m,j) ! (jk)(lm)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = r3d_excits(4,jdet); m = r3d_excits(5,jdet);
                           ! compute < ijkabc | h2c(oooo) | lmjabc >
                           hmatel = -h2c_oooo(l,m,i,k)
                           ! compute < ijkabc | h1b(oo) | lmjabc > = A(ik)A(lm) h1b_oo(l,i) * delta(m,k)
                           hmatel1 = 0.0d0
                           if (m==k) hmatel1 = hmatel1 + h1b_oo(l,i) ! (1)      < ijkabc | h1b(oo) | lkjabc >
                           if (m==i) hmatel1 = hmatel1 - h1b_oo(l,k) ! (ik)
                           if (l==k) hmatel1 = hmatel1 - h1b_oo(m,i) ! (lm)
                           if (l==i) hmatel1 = hmatel1 + h1b_oo(m,k) ! (ik)(lm)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABCI LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/-1,nub/), (/1,nob-2/), nub, nub, nub, nob)
                  call sort4(r3d_excits, r3d_amps, loc_arr, idx_table, (/1,2,3,4/), nub, nub, nub, nob, nloc, n3bbb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,&
                  !$omp r3d_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1B_oo,H2C_oooo,&
                  !$omp nob,nub,n3bbb_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,i)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r3d_excits(5,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(oooo) | imnabc >
                        hmatel = h2c_oooo(m,n,j,k)
                        ! compute < ijkabc | h1b(oo) | imnabc > = -A(jk)A(mn) h1b_oo(m,j) * delta(n,k)
                        hmatel1 = 0.0d0
                        if (n==k) hmatel1 = hmatel1 - h1b_oo(m,j)  ! < ijkabc | h1b(oo) | imkabc >
                        if (n==j) hmatel1 = hmatel1 + h1b_oo(m,k)
                        if (m==k) hmatel1 = hmatel1 + h1b_oo(n,j)
                        if (m==j) hmatel1 = hmatel1 - h1b_oo(n,k)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           m = r3d_excits(5,jdet); n = r3d_excits(6,jdet);
                           ! compute < ijkabc | h2c(oooo) | jmnabc >
                           hmatel = -h2c_oooo(m,n,i,k)
                           ! compute < ijkabc | h1b(oo) | jmnabc > = A(ik)A(mn) h1b_oo(m,i) * delta(n,k)
                           hmatel1 = 0.0d0
                           if (n==k) hmatel1 = hmatel1 + h1b_oo(m,i)
                           if (n==i) hmatel1 = hmatel1 - h1b_oo(m,k)
                           if (m==k) hmatel1 = hmatel1 - h1b_oo(n,i)
                           if (m==i) hmatel1 = hmatel1 + h1b_oo(n,k)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                        end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           m = r3d_excits(5,jdet); n = r3d_excits(6,jdet);
                           ! compute < ijkabc | h2c(oooo) | kmnabc >
                           hmatel = -h2c_oooo(m,n,j,i)
                           ! compute < ijkabc | h1b(oo) | kmnabc > = A(ij)A(mn) h1b_oo(m,j) * delta(n,i)
                           hmatel1 = 0.0d0
                           if (n==i) hmatel1 = hmatel1 - h1b_oo(m,j)
                           if (n==j) hmatel1 = hmatel1 + h1b_oo(m,i)
                           if (m==i) hmatel1 = hmatel1 + h1b_oo(n,j)
                           if (m==j) hmatel1 = hmatel1 - h1b_oo(n,i)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABCJ LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/-1,nub/), (/2,nob-1/), nub, nub, nub, nob)
                  call sort4(r3d_excits, r3d_amps, loc_arr, idx_table, (/1,2,3,5/), nub, nub, nub, nob, nloc, n3bbb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,&
                  !$omp r3d_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1B_oo,H2C_oooo,&
                  !$omp nob,nub,n3bbb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,j)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        l = r3d_excits(4,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(oooo) | ljnabc >
                        hmatel = h2c_oooo(l,n,i,k)
                        ! compute < ijkabc | h1b(oo) | ljnabc > = -A(ik)A(ln) h1b_oo(l,i) * delta(n,k)
                        hmatel1 = 0.0d0
                        if (n==k) hmatel1 = hmatel1 - h1b_oo(l,i)
                        if (n==i) hmatel1 = hmatel1 + h1b_oo(l,k)
                        if (l==k) hmatel1 = hmatel1 + h1b_oo(n,i)
                        if (l==i) hmatel1 = hmatel1 - h1b_oo(n,k)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = r3d_excits(4,jdet); n = r3d_excits(6,jdet);
                           ! compute < ijkabc | h2c(oooo) | linabc >
                           hmatel = -h2c_oooo(l,n,j,k)
                           ! compute < ijkabc | h1b(oo) | linabc > = A(jk)A(ln) h1b_oo(l,j) * delta(n,k)
                           hmatel1 = 0.0d0
                           if (n==k) hmatel1 = hmatel1 + h1b_oo(l,j)
                           if (n==j) hmatel1 = hmatel1 - h1b_oo(l,k)
                           if (l==k) hmatel1 = hmatel1 - h1b_oo(n,j)
                           if (l==j) hmatel1 = hmatel1 + h1b_oo(n,k)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = r3d_excits(4,jdet); n = r3d_excits(6,jdet);
                           ! compute < ijkabc | h2c(oooo) | lknabc >
                           hmatel = -h2c_oooo(l,n,i,j)
                           ! compute < ijkabc | h1b(oo) | lknabc > = A(ij)A(ln) h1b_oo(l,i) * delta(n,j)
                           hmatel1 = 0.0d0
                           if (n==j) hmatel1 = hmatel1 + h1b_oo(l,i)
                           if (n==i) hmatel1 = hmatel1 - h1b_oo(l,j)
                           if (l==j) hmatel1 = hmatel1 - h1b_oo(n,i)
                           if (l==i) hmatel1 = hmatel1 + h1b_oo(n,j)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  !!!! diagram 1b: -A(i/jk) x1b(mi) * t3d(abcmjk)
                  !!!! diagram 3b: 1/2 A(i/jk) x2c(mnij) * t3d(abcmnk)
                  ! NOTE: WITHIN THESE LOOPS, X1B(OO) TERMS ARE DOUBLE-COUNTED SO COMPENSATE BY FACTOR OF 1/2
                  ! allocate temporary arrays
                  allocate(excits_buff(6,n3bbb_t),amps_buff(n3bbb_t))
                  excits_buff(:,:) = t3d_excits(:,:)
                  amps_buff(:) = t3d_amps(:)
                  ! allocate new sorting arrays
                  nloc = nub*(nub-1)*(nub-2)/6*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nub,nob))
                  !!! ABCK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/-1,nub/), (/3,nob/), nub, nub, nub, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,3,6/), nub, nub, nub, nob, nloc, n3bbb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X1B_oo,X2C_oooo,&
                  !$omp nob,nub,n3bbb_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        l = excits_buff(4,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | x2c(oooo) | lmkabc >
                        hmatel = x2c_oooo(l,m,i,j)
                        ! compute < ijkabc | x1b(oo) | lmkabc > = -A(ij)A(lm) x1b_oo(l,i) * delta(m,j)
                        hmatel1 = 0.0d0
                        if (m==j) hmatel1 = hmatel1 - x1b_oo(l,i) ! (1)      < ijkabc | x1b(oo) | ljkabc >
                        if (m==i) hmatel1 = hmatel1 + x1b_oo(l,j) ! (ij)     < ijkabc | x1b(oo) | likabc >
                        if (l==j) hmatel1 = hmatel1 + x1b_oo(m,i) ! (lm)     < ijkabc | x1b(oo) | jmkabc >
                        if (l==i) hmatel1 = hmatel1 - x1b_oo(m,j) ! (ij)(lm) < ijkabc | x1b(oo) | imkabc >
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ik)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = excits_buff(4,jdet); m = excits_buff(5,jdet);
                           ! compute < ijkabc | x2c(oooo) | lmiabc >
                           hmatel = -x2c_oooo(l,m,k,j)
                           ! compute < ijkabc | x1b(oo) | lmiabc > = A(jk)A(lm) x1b_oo(l,k) * delta(m,j)
                           hmatel1 = 0.0d0
                           if (m==j) hmatel1 = hmatel1 + x1b_oo(l,k) ! (1)      < ijkabc | x1b(oo) | ljiabc >
                           if (m==k) hmatel1 = hmatel1 - x1b_oo(l,j) ! (jk)     < ijkabc | x1b(oo) | lkiabc >
                           if (l==j) hmatel1 = hmatel1 - x1b_oo(m,k) ! (lm)
                           if (l==k) hmatel1 = hmatel1 + x1b_oo(m,j) ! (jk)(lm)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = excits_buff(4,jdet); m = excits_buff(5,jdet);
                           ! compute < ijkabc | x2c(oooo) | lmjabc >
                           hmatel = -x2c_oooo(l,m,i,k)
                           ! compute < ijkabc | x1b(oo) | lmjabc > = A(ik)A(lm) x1b_oo(l,i) * delta(m,k)
                           hmatel1 = 0.0d0
                           if (m==k) hmatel1 = hmatel1 + x1b_oo(l,i) ! (1)      < ijkabc | x1b(oo) | lkjabc >
                           if (m==i) hmatel1 = hmatel1 - x1b_oo(l,k) ! (ik)
                           if (l==k) hmatel1 = hmatel1 - x1b_oo(m,i) ! (lm)
                           if (l==i) hmatel1 = hmatel1 + x1b_oo(m,k) ! (ik)(lm)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABCI LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/-1,nub/), (/1,nob-2/), nub, nub, nub, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,3,4/), nub, nub, nub, nob, nloc, n3bbb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X1B_oo,X2C_oooo,&
                  !$omp nob,nub,n3bbb_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,i)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = excits_buff(5,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | x2c(oooo) | imnabc >
                        hmatel = x2c_oooo(m,n,j,k)
                        ! compute < ijkabc | x1b(oo) | imnabc > = -A(jk)A(mn) x1b_oo(m,j) * delta(n,k)
                        hmatel1 = 0.0d0
                        if (n==k) hmatel1 = hmatel1 - x1b_oo(m,j)  ! < ijkabc | x1b(oo) | imkabc >
                        if (n==j) hmatel1 = hmatel1 + x1b_oo(m,k)
                        if (m==k) hmatel1 = hmatel1 + x1b_oo(n,j)
                        if (m==j) hmatel1 = hmatel1 - x1b_oo(n,k)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           m = excits_buff(5,jdet); n = excits_buff(6,jdet);
                           ! compute < ijkabc | x2c(oooo) | jmnabc >
                           hmatel = -x2c_oooo(m,n,i,k)
                           ! compute < ijkabc | x1b(oo) | jmnabc > = A(ik)A(mn) x1b_oo(m,i) * delta(n,k)
                           hmatel1 = 0.0d0
                           if (n==k) hmatel1 = hmatel1 + x1b_oo(m,i)
                           if (n==i) hmatel1 = hmatel1 - x1b_oo(m,k)
                           if (m==k) hmatel1 = hmatel1 - x1b_oo(n,i)
                           if (m==i) hmatel1 = hmatel1 + x1b_oo(n,k)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           m = excits_buff(5,jdet); n = excits_buff(6,jdet);
                           ! compute < ijkabc | x2c(oooo) | kmnabc >
                           hmatel = -x2c_oooo(m,n,j,i)
                           ! compute < ijkabc | x1b(oo) | kmnabc > = A(ij)A(mn) x1b_oo(m,j) * delta(n,i)
                           hmatel1 = 0.0d0
                           if (n==i) hmatel1 = hmatel1 - x1b_oo(m,j)
                           if (n==j) hmatel1 = hmatel1 + x1b_oo(m,i)
                           if (m==i) hmatel1 = hmatel1 + x1b_oo(n,j)
                           if (m==j) hmatel1 = hmatel1 - x1b_oo(n,i)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABCJ LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/-1,nub/), (/2,nob-1/), nub, nub, nub, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,3,5/), nub, nub, nub, nob, nloc, n3bbb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X1B_oo,X2C_oooo,&
                  !$omp nob,nub,n3bbb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,j)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        l = excits_buff(4,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | x2c(oooo) | ljnabc >
                        hmatel = x2c_oooo(l,n,i,k)
                        ! compute < ijkabc | x1b(oo) | ljnabc > = -A(ik)A(ln) x1b_oo(l,i) * delta(n,k)
                        hmatel1 = 0.0d0
                        if (n==k) hmatel1 = hmatel1 - x1b_oo(l,i)
                        if (n==i) hmatel1 = hmatel1 + x1b_oo(l,k)
                        if (l==k) hmatel1 = hmatel1 + x1b_oo(n,i)
                        if (l==i) hmatel1 = hmatel1 - x1b_oo(n,k)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = excits_buff(4,jdet); n = excits_buff(6,jdet);
                           ! compute < ijkabc | x2c(oooo) | linabc >
                           hmatel = -x2c_oooo(l,n,j,k)
                           ! compute < ijkabc | x1b(oo) | linabc > = A(jk)A(ln) x1b_oo(l,j) * delta(n,k)
                           hmatel1 = 0.0d0
                           if (n==k) hmatel1 = hmatel1 + x1b_oo(l,j)
                           if (n==j) hmatel1 = hmatel1 - x1b_oo(l,k)
                           if (l==k) hmatel1 = hmatel1 - x1b_oo(n,j)
                           if (l==j) hmatel1 = hmatel1 + x1b_oo(n,k)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = excits_buff(4,jdet); n = excits_buff(6,jdet);
                           ! compute < ijkabc | x2c(oooo) | lknabc >
                           hmatel = -x2c_oooo(l,n,i,j)
                           ! compute < ijkabc | x1b(oo) | lknabc > = A(ij)A(ln) x1b_oo(l,i) * delta(n,j)
                           hmatel1 = 0.0d0
                           if (n==j) hmatel1 = hmatel1 + x1b_oo(l,i)
                           if (n==i) hmatel1 = hmatel1 - x1b_oo(l,j)
                           if (l==j) hmatel1 = hmatel1 - x1b_oo(n,i)
                           if (l==i) hmatel1 = hmatel1 + x1b_oo(n,j)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate sorting arrays
                  deallocate(excits_buff,amps_buff)

                  !!!! diagram 2a: A(a/bc) h1b(ae) * r3d(ebcijk)
                  !!!! diagram 4a: 1/2 A(c/ab) h2c(abef) * r3d(ebcijk)
                  ! NOTE: WITHIN THESE LOOPS, H1B(VV) TERMS ARE DOUBLE-COUNTED SO COMPENSATE BY FACTOR OF 1/2
                  ! allocate new sorting arrays
                  nloc = nob*(nob-1)*(nob-2)/6*nub
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nob,nob,nob,nub))
                  !!! IJKA LOOP !!!
                  call get_index_table(idx_table, (/1,nob-2/), (/-1,nob-1/), (/-1,nob/), (/1,nub-2/), nob, nob, nob, nub)
                  call sort4(r3d_excits, r3d_amps, loc_arr, idx_table, (/4,5,6,1/), nob, nob, nob, nub, nloc, n3bbb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,&
                  !$omp r3d_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1B_vv,H2C_vvvv,&
                  !$omp nob,nub,n3bbb_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,a)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); f = r3d_excits(3,jdet);
                        ! compute < ijkabc | h2c(vvvv) | ijkaef >
                        !hmatel = h2c_vvvv(b,c,e,f)
                        hmatel = h2c_vvvv(e,f,b,c)
                        ! compute < ijkabc | h1b(vv) | ijkaef > = A(bc)A(ef) h1b_vv(b,e) * delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 + h1b_vv(b,e) ! (1)
                        if (b==f) hmatel1 = hmatel1 - h1b_vv(c,e) ! (bc)
                        if (c==e) hmatel1 = hmatel1 - h1b_vv(b,f) ! (ef)
                        if (b==e) hmatel1 = hmatel1 + h1b_vv(c,f) ! (bc)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); f = r3d_excits(3,jdet);
                        ! compute < ijkabc | h2c(vvvv) | ijkbef >
                        !hmatel = -h2c_vvvv(a,c,e,f)
                        hmatel = -h2c_vvvv(e,f,a,c)
                        ! compute < ijkabc | h1b(vv) | ijkbef > = -A(ac)A(ef) h1b_vv(a,e) * delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 - h1b_vv(a,e) ! (1)
                        if (a==f) hmatel1 = hmatel1 + h1b_vv(c,e) ! (ac)
                        if (c==e) hmatel1 = hmatel1 + h1b_vv(a,f) ! (ef)
                        if (a==e) hmatel1 = hmatel1 - h1b_vv(c,f) ! (ac)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); f = r3d_excits(3,jdet);
                        ! compute < ijkabc | h2c(vvvv) | ijkcef >
                        !hmatel = -h2c_vvvv(b,a,e,f)
                        hmatel = -h2c_vvvv(e,f,b,a)
                        ! compute < ijkabc | h1b(vv) | ijkcef > = -A(ab)A(ef) h1b_vv(b,e) * delta(a,f)
                        hmatel1 = 0.0d0
                        if (a==f) hmatel1 = hmatel1 - h1b_vv(b,e) ! (1)
                        if (b==f) hmatel1 = hmatel1 + h1b_vv(a,e) ! (ab)
                        if (a==e) hmatel1 = hmatel1 + h1b_vv(b,f) ! (ef)
                        if (b==e) hmatel1 = hmatel1 - h1b_vv(a,f) ! (ab)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! IJKB LOOP !!!
                  call get_index_table(idx_table, (/1,nob-2/), (/-1,nob-1/), (/-1,nob/), (/2,nub-1/), nob, nob, nob, nub)
                  call sort4(r3d_excits, r3d_amps, loc_arr, idx_table, (/4,5,6,2/), nob, nob, nob, nub, nloc, n3bbb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,&
                  !$omp r3d_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1B_vv,H2C_vvvv,&
                  !$omp nob,nub,n3bbb_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); f = r3d_excits(3,jdet);
                        ! compute < ijkabc | h2c(vvvv) | ijkdbf >
                        !hmatel = h2c_vvvv(a,c,d,f)
                        hmatel = h2c_vvvv(d,f,a,c)
                        ! compute < ijkabc | h1b(vv) | ijkdbf > = A(ac)A(df) h1b_vv(a,d) * delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 + h1b_vv(a,d) ! (1)
                        if (a==f) hmatel1 = hmatel1 - h1b_vv(c,d) ! (ac)
                        if (c==d) hmatel1 = hmatel1 - h1b_vv(a,f) ! (df)
                        if (a==d) hmatel1 = hmatel1 + h1b_vv(c,f) ! (ac)(df)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); f = r3d_excits(3,jdet);
                        ! compute < ijkabc | h2c(vvvv) | ijkdaf >
                        !hmatel = -h2c_vvvv(b,c,d,f)
                        hmatel = -h2c_vvvv(d,f,b,c)
                        ! compute < ijkabc | h1b(vv) | ijkdaf > = -A(bc)A(df) h1b_vv(b,d) * delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 - h1b_vv(b,d) ! (1)
                        if (b==f) hmatel1 = hmatel1 + h1b_vv(c,d) ! (bc)
                        if (c==d) hmatel1 = hmatel1 + h1b_vv(b,f) ! (df)
                        if (b==d) hmatel1 = hmatel1 - h1b_vv(c,f) ! (bc)(df)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); f = r3d_excits(3,jdet);
                        ! compute < ijkabc | h2c(vvvv) | ijkdcf >
                        !hmatel = -h2c_vvvv(a,b,d,f)
                        hmatel = -h2c_vvvv(d,f,a,b)
                        ! compute < ijkabc | h1b(vv) | ijkdcf > = -A(ab)A(df) h1b_vv(a,d) * delta(b,f)
                        hmatel1 = 0.0d0
                        if (b==f) hmatel1 = hmatel1 - h1b_vv(a,d) ! (1)
                        if (a==f) hmatel1 = hmatel1 + h1b_vv(b,d) ! (ab)
                        if (b==d) hmatel1 = hmatel1 + h1b_vv(a,f) ! (df)
                        if (a==d) hmatel1 = hmatel1 - h1b_vv(b,f) ! (ab)(df)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! IJKC LOOP !!!
                  call get_index_table(idx_table, (/1,nob-2/), (/-1,nob-1/), (/-1,nob/), (/3,nub/), nob, nob, nob, nub)
                  call sort4(r3d_excits, r3d_amps, loc_arr, idx_table, (/4,5,6,3/), nob, nob, nob, nub, nloc, n3bbb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,&
                  !$omp r3d_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1B_vv,H2C_vvvv,&
                  !$omp nob,nub,n3bbb_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,c)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); e = r3d_excits(2,jdet);
                        ! compute < ijkabc | h2c(vvvv) | ijkdec >
                        !hmatel = h2c_vvvv(a,b,d,e)
                        hmatel = h2c_vvvv(d,e,a,b)
                        ! compute < ijkabc | h1b(vv) | ijkdec > = A(ab)A(de) h1b_vv(a,d) * delta(b,e)
                        hmatel1 = 0.0d0
                        if (b==e) hmatel1 = hmatel1 + h1b_vv(a,d) ! (1)
                        if (a==e) hmatel1 = hmatel1 - h1b_vv(b,d) ! (ab)
                        if (b==d) hmatel1 = hmatel1 - h1b_vv(a,e) ! (de)
                        if (a==d) hmatel1 = hmatel1 + h1b_vv(b,e) ! (ab)(de)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); e = r3d_excits(2,jdet);
                        ! compute < ijkabc | h2c(vvvv) | ijkdea >
                        !hmatel = -h2c_vvvv(c,b,d,e)
                        hmatel = -h2c_vvvv(d,e,c,b)
                        ! compute < ijkabc | h1b(vv) | ijkdea > = -A(bc)A(de) h1b_vv(c,d) * delta(b,e)
                        hmatel1 = 0.0d0
                        if (b==e) hmatel1 = hmatel1 - h1b_vv(c,d) ! (1)
                        if (c==e) hmatel1 = hmatel1 + h1b_vv(b,d) ! (bc)
                        if (b==d) hmatel1 = hmatel1 + h1b_vv(c,e) ! (de)
                        if (c==d) hmatel1 = hmatel1 - h1b_vv(b,e) ! (bc)(de)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); e = r3d_excits(2,jdet);
                        ! compute < ijkabc | h2c(vvvv) | ijkdeb >
                        !hmatel = -h2c_vvvv(a,c,d,e)
                        hmatel = -h2c_vvvv(d,e,a,c)
                        ! compute < ijkabc | h1b(vv) | ijkdeb > = -A(ac)A(de) h1b_vv(a,d) * delta(c,e)
                        hmatel1 = 0.0d0
                        if (c==e) hmatel1 = hmatel1 - h1b_vv(a,d) ! (1)
                        if (a==e) hmatel1 = hmatel1 + h1b_vv(c,d) ! (ac)
                        if (c==d) hmatel1 = hmatel1 + h1b_vv(a,e) ! (de)
                        if (a==d) hmatel1 = hmatel1 - h1b_vv(c,e) ! (ac)(de)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  !!!! diagram 2b: A(a/bc) x1b(ae) * t3d(ebcijk)
                  !!!! diagram 4b: 1/2 A(c/ab) x2c(abef) * t3d(ebcijk)
                  ! NOTE: WITHIN THESE LOOPS, X1B(VV) TERMS ARE DOUBLE-COUNTED SO COMPENSATE BY FACTOR OF 1/2
                  ! allocate temporary arrays
                  allocate(excits_buff(6,n3bbb_t),amps_buff(n3bbb_t))
                  excits_buff(:,:) = t3d_excits(:,:)
                  amps_buff(:) = t3d_amps(:)
                  ! allocate new sorting arrays
                  nloc = nob*(nob-1)*(nob-2)/6*nub
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nob,nob,nob,nub))
                  !!! IJKA LOOP !!!
                  call get_index_table(idx_table, (/1,nob-2/), (/-1,nob-1/), (/-1,nob/), (/1,nub-2/), nob, nob, nob, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,1/), nob, nob, nob, nub, nloc, n3bbb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X1B_vv,X2C_vvvv,&
                  !$omp nob,nub,n3bbb_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,a)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); f = excits_buff(3,jdet);
                        ! compute < ijkabc | x2c(vvvv) | ijkaef >
                        !hmatel = x2c_vvvv(b,c,e,f)
                        hmatel = x2c_vvvv(e,f,b,c)
                        ! compute < ijkabc | x1b(vv) | ijkaef > = A(bc)A(ef) x1b_vv(b,e) * delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 + x1b_vv(b,e) ! (1)
                        if (b==f) hmatel1 = hmatel1 - x1b_vv(c,e) ! (bc)
                        if (c==e) hmatel1 = hmatel1 - x1b_vv(b,f) ! (ef)
                        if (b==e) hmatel1 = hmatel1 + x1b_vv(c,f) ! (bc)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); f = excits_buff(3,jdet);
                        ! compute < ijkabc | x2c(vvvv) | ijkbef >
                        !hmatel = -x2c_vvvv(a,c,e,f)
                        hmatel = -x2c_vvvv(e,f,a,c)
                        ! compute < ijkabc | x1b(vv) | ijkbef > = -A(ac)A(ef) x1b_vv(a,e) * delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 - x1b_vv(a,e) ! (1)
                        if (a==f) hmatel1 = hmatel1 + x1b_vv(c,e) ! (ac)
                        if (c==e) hmatel1 = hmatel1 + x1b_vv(a,f) ! (ef)
                        if (a==e) hmatel1 = hmatel1 - x1b_vv(c,f) ! (ac)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); f = excits_buff(3,jdet);
                        ! compute < ijkabc | x2c(vvvv) | ijkcef >
                        !hmatel = -x2c_vvvv(b,a,e,f)
                        hmatel = -x2c_vvvv(e,f,b,a)
                        ! compute < ijkabc | x1b(vv) | ijkcef > = -A(ab)A(ef) x1b_vv(b,e) * delta(a,f)
                        hmatel1 = 0.0d0
                        if (a==f) hmatel1 = hmatel1 - x1b_vv(b,e) ! (1)
                        if (b==f) hmatel1 = hmatel1 + x1b_vv(a,e) ! (ab)
                        if (a==e) hmatel1 = hmatel1 + x1b_vv(b,f) ! (ef)
                        if (b==e) hmatel1 = hmatel1 - x1b_vv(a,f) ! (ab)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! IJKB LOOP !!!
                  call get_index_table(idx_table, (/1,nob-2/), (/-1,nob-1/), (/-1,nob/), (/2,nub-1/), nob, nob, nob, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,2/), nob, nob, nob, nub, nloc, n3bbb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X1B_vv,X2C_vvvv,&
                  !$omp nob,nub,n3bbb_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); f = excits_buff(3,jdet);
                        ! compute < ijkabc | x2c(vvvv) | ijkdbf >
                        !hmatel = x2c_vvvv(a,c,d,f)
                        hmatel = x2c_vvvv(d,f,a,c)
                        ! compute < ijkabc | x1b(vv) | ijkdbf > = A(ac)A(df) x1b_vv(a,d) * delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 + x1b_vv(a,d) ! (1)
                        if (a==f) hmatel1 = hmatel1 - x1b_vv(c,d) ! (ac)
                        if (c==d) hmatel1 = hmatel1 - x1b_vv(a,f) ! (df)
                        if (a==d) hmatel1 = hmatel1 + x1b_vv(c,f) ! (ac)(df)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); f = excits_buff(3,jdet);
                        ! compute < ijkabc | x2c(vvvv) | ijkdaf >
                        !hmatel = -x2c_vvvv(b,c,d,f)
                        hmatel = -x2c_vvvv(d,f,b,c)
                        ! compute < ijkabc | x1b(vv) | ijkdaf > = -A(bc)A(df) x1b_vv(b,d) * delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 - x1b_vv(b,d) ! (1)
                        if (b==f) hmatel1 = hmatel1 + x1b_vv(c,d) ! (bc)
                        if (c==d) hmatel1 = hmatel1 + x1b_vv(b,f) ! (df)
                        if (b==d) hmatel1 = hmatel1 - x1b_vv(c,f) ! (bc)(df)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); f = excits_buff(3,jdet);
                        ! compute < ijkabc | x2c(vvvv) | ijkdcf >
                        !hmatel = -x2c_vvvv(a,b,d,f)
                        hmatel = -x2c_vvvv(d,f,a,b)
                        ! compute < ijkabc | x1b(vv) | ijkdcf > = -A(ab)A(df) x1b_vv(a,d) * delta(b,f)
                        hmatel1 = 0.0d0
                        if (b==f) hmatel1 = hmatel1 - x1b_vv(a,d) ! (1)
                        if (a==f) hmatel1 = hmatel1 + x1b_vv(b,d) ! (ab)
                        if (b==d) hmatel1 = hmatel1 + x1b_vv(a,f) ! (df)
                        if (a==d) hmatel1 = hmatel1 - x1b_vv(b,f) ! (ab)(df)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! IJKC LOOP !!!
                  call get_index_table(idx_table, (/1,nob-2/), (/-1,nob-1/), (/-1,nob/), (/3,nub/), nob, nob, nob, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,3/), nob, nob, nob, nub, nloc, n3bbb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X1B_vv,X2C_vvvv,&
                  !$omp nob,nub,n3bbb_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,c)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); e = excits_buff(2,jdet);
                        ! compute < ijkabc | x2c(vvvv) | ijkdec >
                        !hmatel = x2c_vvvv(a,b,d,e)
                        hmatel = x2c_vvvv(d,e,a,b)
                        ! compute < ijkabc | x1b(vv) | ijkdec > = A(ab)A(de) x1b_vv(a,d) * delta(b,e)
                        hmatel1 = 0.0d0
                        if (b==e) hmatel1 = hmatel1 + x1b_vv(a,d) ! (1)
                        if (a==e) hmatel1 = hmatel1 - x1b_vv(b,d) ! (ab)
                        if (b==d) hmatel1 = hmatel1 - x1b_vv(a,e) ! (de)
                        if (a==d) hmatel1 = hmatel1 + x1b_vv(b,e) ! (ab)(de)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); e = excits_buff(2,jdet);
                        ! compute < ijkabc | x2c(vvvv) | ijkdea >
                        !hmatel = -x2c_vvvv(c,b,d,e)
                        hmatel = -x2c_vvvv(d,e,c,b)
                        ! compute < ijkabc | x1b(vv) | ijkdea > = -A(bc)A(de) x1b_vv(c,d) * delta(b,e)
                        hmatel1 = 0.0d0
                        if (b==e) hmatel1 = hmatel1 - x1b_vv(c,d) ! (1)
                        if (c==e) hmatel1 = hmatel1 + x1b_vv(b,d) ! (bc)
                        if (b==d) hmatel1 = hmatel1 + x1b_vv(c,e) ! (de)
                        if (c==d) hmatel1 = hmatel1 - x1b_vv(b,e) ! (bc)(de)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); e = excits_buff(2,jdet);
                        ! compute < ijkabc | x2c(vvvv) | ijkdeb >
                        !hmatel = -x2c_vvvv(a,c,d,e)
                        hmatel = -x2c_vvvv(d,e,a,c)
                        ! compute < ijkabc | x1b(vv) | ijkdeb > = -A(ac)A(de) x1b_vv(a,d) * delta(c,e)
                        hmatel1 = 0.0d0
                        if (c==e) hmatel1 = hmatel1 - x1b_vv(a,d) ! (1)
                        if (a==e) hmatel1 = hmatel1 + x1b_vv(c,d) ! (ac)
                        if (c==d) hmatel1 = hmatel1 + x1b_vv(a,e) ! (de)
                        if (a==d) hmatel1 = hmatel1 - x1b_vv(c,e) ! (ac)(de)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate temporary arrays
                  deallocate(excits_buff,amps_buff)

                  !!!! diagram 5a: A(i/jk)A(a/bc) h2c(amie) * r3d(ebcmjk)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = (nub-1)*(nub-2)/2*(nob-1)*(nob-2)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nob,nob))
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4(r3d_excits, r3d_amps, loc_arr, idx_table, (/1,2,4,5/), nub, nub, nob, nob, nloc, n3bbb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,&
                  !$omp r3d_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,n3bbb_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3d_excits(3,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(voov) | ijnabf >
                        hmatel = h2c_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3d_excits(3,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(voov) | ijnbcf >
                        hmatel = h2c_voov(a,n,k,f)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3d_excits(3,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(voov) | ijnacf >
                        hmatel = -h2c_voov(b,n,k,f)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3d_excits(3,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(voov) | jknabf >
                        hmatel = h2c_voov(c,n,i,f)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3d_excits(3,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(voov) | jknbcf >
                        hmatel = h2c_voov(a,n,i,f)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3d_excits(3,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(voov) | jknacf >
                        hmatel = -h2c_voov(b,n,i,f)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3d_excits(3,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(voov) | iknabf >
                        hmatel = -h2c_voov(c,n,j,f)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3d_excits(3,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(voov) | iknbcf >
                        hmatel = -h2c_voov(a,n,j,f)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3d_excits(3,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(voov) | iknacf >
                        hmatel = h2c_voov(b,n,j,f)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4(r3d_excits, r3d_amps, loc_arr, idx_table, (/1,3,4,5/), nub, nub, nob, nob, nloc, n3bbb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,&
                  !$omp r3d_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,n3bbb_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(voov) | ijnaec >
                        hmatel = h2c_voov(b,n,k,e)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(voov) | ijnbec >
                        hmatel = -h2c_voov(a,n,k,e)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(voov) | ijnaeb >
                        hmatel = -h2c_voov(c,n,k,e)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(voov) | jknaec >
                        hmatel = h2c_voov(b,n,i,e)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(voov) | jknbec >
                        hmatel = -h2c_voov(a,n,i,e)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(voov) | jknaeb >
                        hmatel = -h2c_voov(c,n,i,e)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(voov) | iknaec >
                        hmatel = -h2c_voov(b,n,j,e)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(voov) | iknbec >
                        hmatel = h2c_voov(a,n,j,e)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(voov) | iknaeb >
                        hmatel = h2c_voov(c,n,j,e)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCIJ LOOP !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4(r3d_excits, r3d_amps, loc_arr, idx_table, (/2,3,4,5/), nub, nub, nob, nob, nloc, n3bbb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,&
                  !$omp r3d_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,n3bbb_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(voov) | ijndbc >
                        hmatel = h2c_voov(a,n,k,d)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(voov) | ijndac >
                        hmatel = -h2c_voov(b,n,k,d)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(voov) | ijndab >
                        hmatel = h2c_voov(c,n,k,d)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(voov) | jkndbc >
                        hmatel = h2c_voov(a,n,i,d)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(voov) | jkndac >
                        hmatel = -h2c_voov(b,n,i,d)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(voov) | jkndab >
                        hmatel = h2c_voov(c,n,i,d)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(voov) | ikndbc >
                        hmatel = -h2c_voov(a,n,j,d)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(voov) | ikndac >
                        hmatel = h2c_voov(b,n,j,d)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); n = r3d_excits(6,jdet);
                        ! compute < ijkabc | h2c(voov) | ikndab >
                        hmatel = -h2c_voov(c,n,j,d)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4(r3d_excits, r3d_amps, loc_arr, idx_table, (/1,2,4,6/), nub, nub, nob, nob, nloc, n3bbb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,&
                  !$omp r3d_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,n3bbb_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3d_excits(3,jdet); m = r3d_excits(5,jdet);
                        ! compute < ijkabc | h2c(voov) | imkabf >
                        hmatel = h2c_voov(c,m,j,f)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3d_excits(3,jdet); m = r3d_excits(5,jdet);
                        ! compute < ijkabc | h2c(voov) | imkbcf >
                        hmatel = h2c_voov(a,m,j,f)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3d_excits(3,jdet); m = r3d_excits(5,jdet);
                        ! compute < ijkabc | h2c(voov) | imkacf >
                        hmatel = -h2c_voov(b,m,j,f)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3d_excits(3,jdet); m = r3d_excits(5,jdet);
                        ! compute < ijkabc | h2c(voov) | jmkabf >
                        hmatel = -h2c_voov(c,m,i,f)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3d_excits(3,jdet); m = r3d_excits(5,jdet);
                        ! compute < ijkabc | h2c(voov) | jmkbcf >
                        hmatel = -h2c_voov(a,m,i,f)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3d_excits(3,jdet); m = r3d_excits(5,jdet);
                        ! compute < ijkabc | h2c(voov) | jmkacf >
                        hmatel = h2c_voov(b,m,i,f)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3d_excits(3,jdet); m = r3d_excits(5,jdet);
                        ! compute < ijkabc | h2c(voov) | imjabf >
                        hmatel = -h2c_voov(c,m,k,f)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3d_excits(3,jdet); m = r3d_excits(5,jdet);
                        ! compute < ijkabc | h2c(voov) | imjbcf >
                        hmatel = -h2c_voov(a,m,k,f)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3d_excits(3,jdet); m = r3d_excits(5,jdet);
                        ! compute < ijkabc | h2c(voov) | imjacf >
                        hmatel = h2c_voov(b,m,k,f)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4(r3d_excits, r3d_amps, loc_arr, idx_table, (/1,3,4,6/), nub, nub, nob, nob, nloc, n3bbb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,&
                  !$omp r3d_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,n3bbb_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); m = r3d_excits(5,jdet);
                        ! compute < ijkabc | h2c(voov) | imkaec >
                        hmatel = h2c_voov(b,m,j,e)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); m = r3d_excits(5,jdet);
                        ! compute < ijkabc | h2c(voov) | imkbec >
                        hmatel = -h2c_voov(a,m,j,e)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); m = r3d_excits(5,jdet);
                        ! compute < ijkabc | h2c(voov) | imkaeb >
                        hmatel = -h2c_voov(c,m,j,e)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); m = r3d_excits(5,jdet);
                        ! compute < ijkabc | h2c(voov) | jmkaec >
                        hmatel = -h2c_voov(b,m,i,e)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); m = r3d_excits(5,jdet);
                        ! compute < ijkabc | h2c(voov) | jmkbec >
                        hmatel = h2c_voov(a,m,i,e)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); m = r3d_excits(5,jdet);
                        ! compute < ijkabc | h2c(voov) | jmkaeb >
                        hmatel = h2c_voov(c,m,i,e)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); m = r3d_excits(5,jdet);
                        ! compute < ijkabc | h2c(voov) | imjaec >
                        hmatel = -h2c_voov(b,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); m = r3d_excits(5,jdet);
                        ! compute < ijkabc | h2c(voov) | imjbec >
                        hmatel = h2c_voov(a,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); m = r3d_excits(5,jdet);
                        ! compute < ijkabc | h2c(voov) | imjaeb >
                        hmatel = h2c_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4(r3d_excits, r3d_amps, loc_arr, idx_table, (/2,3,4,6/), nub, nub, nob, nob, nloc, n3bbb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,&
                  !$omp r3d_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,n3bbb_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); m = r3d_excits(5,jdet);
                        ! compute < ijkabc | h2c(voov) | imkdbc >
                        hmatel = h2c_voov(a,m,j,d)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); m = r3d_excits(5,jdet);
                        ! compute < ijkabc | h2c(voov) | imkdac >
                        hmatel = -h2c_voov(b,m,j,d)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); m = r3d_excits(5,jdet);
                        ! compute < ijkabc | h2c(voov) | imkdab >
                        hmatel = h2c_voov(c,m,j,d)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); m = r3d_excits(5,jdet);
                        ! compute < ijkabc | h2c(voov) | jmkdbc >
                        hmatel = -h2c_voov(a,m,i,d)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); m = r3d_excits(5,jdet);
                        ! compute < ijkabc | h2c(voov) | jmkdac >
                        hmatel = h2c_voov(b,m,i,d)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); m = r3d_excits(5,jdet);
                        ! compute < ijkabc | h2c(voov) | jmkdab >
                        hmatel = -h2c_voov(c,m,i,d)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); m = r3d_excits(5,jdet);
                        ! compute < ijkabc | h2c(voov) | imjdbc >
                        hmatel = -h2c_voov(a,m,k,d)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); m = r3d_excits(5,jdet);
                        ! compute < ijkabc | h2c(voov) | imjdac >
                        hmatel = h2c_voov(b,m,k,d)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); m = r3d_excits(5,jdet);
                        ! compute < ijkabc | h2c(voov) | imjdab >
                        hmatel = -h2c_voov(c,m,k,d)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABJK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(r3d_excits, r3d_amps, loc_arr, idx_table, (/1,2,5,6/), nub, nub, nob, nob, nloc, n3bbb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,&
                  !$omp r3d_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,n3bbb_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3d_excits(3,jdet); l = r3d_excits(4,jdet);
                        ! compute < ijkabc | h2c(voov) | ljkabf >
                        hmatel = h2c_voov(c,l,i,f)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3d_excits(3,jdet); l = r3d_excits(4,jdet);
                        ! compute < ijkabc | h2c(voov) | ljkbcf >
                        hmatel = h2c_voov(a,l,i,f)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3d_excits(3,jdet); l = r3d_excits(4,jdet);
                        ! compute < ijkabc | h2c(voov) | ljkacf >
                        hmatel = -h2c_voov(b,l,i,f)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3d_excits(3,jdet); l = r3d_excits(4,jdet);
                        ! compute < ijkabc | h2c(voov) | likabf >
                        hmatel = -h2c_voov(c,l,j,f)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3d_excits(3,jdet); l = r3d_excits(4,jdet);
                        ! compute < ijkabc | h2c(voov) | likbcf >
                        hmatel = -h2c_voov(a,l,j,f)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3d_excits(3,jdet); l = r3d_excits(4,jdet);
                        ! compute < ijkabc | h2c(voov) | likacf >
                        hmatel = h2c_voov(b,l,j,f)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3d_excits(3,jdet); l = r3d_excits(4,jdet);
                        ! compute < ijkabc | h2c(voov) | lijabf >
                        hmatel = h2c_voov(c,l,k,f)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3d_excits(3,jdet); l = r3d_excits(4,jdet);
                        ! compute < ijkabc | h2c(voov) | lijbcf >
                        hmatel = h2c_voov(a,l,k,f)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r3d_excits(3,jdet); l = r3d_excits(4,jdet);
                        ! compute < ijkabc | h2c(voov) | lijacf >
                        hmatel = -h2c_voov(b,l,k,f)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACJK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(r3d_excits, r3d_amps, loc_arr, idx_table, (/1,3,5,6/), nub, nub, nob, nob, nloc, n3bbb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,&
                  !$omp r3d_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,n3bbb_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); l = r3d_excits(4,jdet);
                        ! compute < ijkabc | h2c(voov) | ljkaec >
                        hmatel = h2c_voov(b,l,i,e)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); l = r3d_excits(4,jdet);
                        ! compute < ijkabc | h2c(voov) | ljkbec >
                        hmatel = -h2c_voov(a,l,i,e)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); l = r3d_excits(4,jdet);
                        ! compute < ijkabc | h2c(voov) | ljkaeb >
                        hmatel = -h2c_voov(c,l,i,e)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); l = r3d_excits(4,jdet);
                        ! compute < ijkabc | h2c(voov) | likaec >
                        hmatel = -h2c_voov(b,l,j,e)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); l = r3d_excits(4,jdet);
                        ! compute < ijkabc | h2c(voov) | likbec >
                        hmatel = h2c_voov(a,l,j,e)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); l = r3d_excits(4,jdet);
                        ! compute < ijkabc | h2c(voov) | likaeb >
                        hmatel = h2c_voov(c,l,j,e)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); l = r3d_excits(4,jdet);
                        ! compute < ijkabc | h2c(voov) | lijaec >
                        hmatel = h2c_voov(b,l,k,e)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); l = r3d_excits(4,jdet);
                        ! compute < ijkabc | h2c(voov) | lijbec >
                        hmatel = -h2c_voov(a,l,k,e)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r3d_excits(2,jdet); l = r3d_excits(4,jdet);
                        ! compute < ijkabc | h2c(voov) | lijaeb >
                        hmatel = -h2c_voov(c,l,k,e)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(r3d_excits, r3d_amps, loc_arr, idx_table, (/2,3,5,6/), nub, nub, nob, nob, nloc, n3bbb_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,&
                  !$omp r3d_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,n3bbb_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); l = r3d_excits(4,jdet);
                        ! compute < ijkabc | h2c(voov) | ljkdbc >
                        hmatel = h2c_voov(a,l,i,d)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); l = r3d_excits(4,jdet);
                        ! compute < ijkabc | h2c(voov) | ljkdac >
                        hmatel = -h2c_voov(b,l,i,d)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); l = r3d_excits(4,jdet);
                        ! compute < ijkabc | h2c(voov) | ljkdab >
                        hmatel = h2c_voov(c,l,i,d)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); l = r3d_excits(4,jdet);
                        ! compute < ijkabc | h2c(voov) | likdbc >
                        hmatel = -h2c_voov(a,l,j,d)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); l = r3d_excits(4,jdet);
                        ! compute < ijkabc | h2c(voov) | likdac >
                        hmatel = h2c_voov(b,l,j,d)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); l = r3d_excits(4,jdet);
                        ! compute < ijkabc | h2c(voov) | likdab >
                        hmatel = -h2c_voov(c,l,j,d)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); l = r3d_excits(4,jdet);
                        ! compute < ijkabc | h2c(voov) | lijdbc >
                        hmatel = h2c_voov(a,l,k,d)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); l = r3d_excits(4,jdet);
                        ! compute < ijkabc | h2c(voov) | lijdac >
                        hmatel = -h2c_voov(b,l,k,d)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r3d_excits(1,jdet); l = r3d_excits(4,jdet);
                        ! compute < ijkabc | h2c(voov) | lijdab >
                        hmatel = h2c_voov(c,l,k,d)
                        resid(idet) = resid(idet) + hmatel * r3d_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  !!!! diagram 5b: A(i/jk)A(a/bc) x2c(amie) * t3a(ebcmjk)
                  ! allocate temporary arrays
                  allocate(excits_buff(6,n3bbb_t),amps_buff(n3bbb_t))
                  excits_buff(:,:) = t3d_excits(:,:)
                  amps_buff(:) = t3d_amps(:)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = (nub-1)*(nub-2)/2*(nob-1)*(nob-2)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nob,nob))
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,5/), nub, nub, nob, nob, nloc, n3bbb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2C_voov,&
                  !$omp noa,nua,n3bbb_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijnabf >
                        hmatel = x2c_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijnbcf >
                        hmatel = x2c_voov(a,n,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijnacf >
                        hmatel = -x2c_voov(b,n,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jknabf >
                        hmatel = x2c_voov(c,n,i,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jknbcf >
                        hmatel = x2c_voov(a,n,i,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jknacf >
                        hmatel = -x2c_voov(b,n,i,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | iknabf >
                        hmatel = -x2c_voov(c,n,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | iknbcf >
                        hmatel = -x2c_voov(a,n,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | iknacf >
                        hmatel = x2c_voov(b,n,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,5/), nub, nub, nob, nob, nloc, n3bbb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2C_voov,&
                  !$omp noa,nua,n3bbb_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijnaec >
                        hmatel = x2c_voov(b,n,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijnbec >
                        hmatel = -x2c_voov(a,n,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijnaeb >
                        hmatel = -x2c_voov(c,n,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jknaec >
                        hmatel = x2c_voov(b,n,i,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jknbec >
                        hmatel = -x2c_voov(a,n,i,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jknaeb >
                        hmatel = -x2c_voov(c,n,i,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | iknaec >
                        hmatel = -x2c_voov(b,n,j,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | iknbec >
                        hmatel = x2c_voov(a,n,j,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | iknaeb >
                        hmatel = x2c_voov(c,n,j,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCIJ LOOP !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,5/), nub, nub, nob, nob, nloc, n3bbb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2C_voov,&
                  !$omp noa,nua,n3bbb_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijndbc >
                        hmatel = x2c_voov(a,n,k,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijndac >
                        hmatel = -x2c_voov(b,n,k,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ijndab >
                        hmatel = x2c_voov(c,n,k,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jkndbc >
                        hmatel = x2c_voov(a,n,i,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jkndac >
                        hmatel = -x2c_voov(b,n,i,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | jkndab >
                        hmatel = x2c_voov(c,n,i,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ikndbc >
                        hmatel = -x2c_voov(a,n,j,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ikndac >
                        hmatel = x2c_voov(b,n,j,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(voov) | ikndab >
                        hmatel = -x2c_voov(c,n,j,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,6/), nub, nub, nob, nob, nloc, n3bbb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2C_voov,&
                  !$omp noa,nua,n3bbb_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkabf >
                        hmatel = x2c_voov(c,m,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkbcf >
                        hmatel = x2c_voov(a,m,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkacf >
                        hmatel = -x2c_voov(b,m,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkabf >
                        hmatel = -x2c_voov(c,m,i,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkbcf >
                        hmatel = -x2c_voov(a,m,i,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkacf >
                        hmatel = x2c_voov(b,m,i,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjabf >
                        hmatel = -x2c_voov(c,m,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjbcf >
                        hmatel = -x2c_voov(a,m,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjacf >
                        hmatel = x2c_voov(b,m,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,6/), nub, nub, nob, nob, nloc, n3bbb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2C_voov,&
                  !$omp noa,nua,n3bbb_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkaec >
                        hmatel = x2c_voov(b,m,j,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkbec >
                        hmatel = -x2c_voov(a,m,j,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkaeb >
                        hmatel = -x2c_voov(c,m,j,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkaec >
                        hmatel = -x2c_voov(b,m,i,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkbec >
                        hmatel = x2c_voov(a,m,i,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkaeb >
                        hmatel = x2c_voov(c,m,i,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjaec >
                        hmatel = -x2c_voov(b,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjbec >
                        hmatel = x2c_voov(a,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjaeb >
                        hmatel = x2c_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,6/), nub, nub, nob, nob, nloc, n3bbb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2C_voov,&
                  !$omp noa,nua,n3bbb_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkdbc >
                        hmatel = x2c_voov(a,m,j,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkdac >
                        hmatel = -x2c_voov(b,m,j,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imkdab >
                        hmatel = x2c_voov(c,m,j,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkdbc >
                        hmatel = -x2c_voov(a,m,i,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkdac >
                        hmatel = x2c_voov(b,m,i,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | jmkdab >
                        hmatel = -x2c_voov(c,m,i,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjdbc >
                        hmatel = -x2c_voov(a,m,k,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjdac >
                        hmatel = x2c_voov(b,m,k,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(voov) | imjdab >
                        hmatel = -x2c_voov(c,m,k,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABJK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,5,6/), nub, nub, nob, nob, nloc, n3bbb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2C_voov,&
                  !$omp noa,nua,n3bbb_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkabf >
                        hmatel = x2c_voov(c,l,i,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkbcf >
                        hmatel = x2c_voov(a,l,i,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkacf >
                        hmatel = -x2c_voov(b,l,i,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likabf >
                        hmatel = -x2c_voov(c,l,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likbcf >
                        hmatel = -x2c_voov(a,l,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likacf >
                        hmatel = x2c_voov(b,l,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijabf >
                        hmatel = x2c_voov(c,l,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijbcf >
                        hmatel = x2c_voov(a,l,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(3,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijacf >
                        hmatel = -x2c_voov(b,l,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACJK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,5,6/), nub, nub, nob, nob, nloc, n3bbb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2C_voov,&
                  !$omp noa,nua,n3bbb_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkaec >
                        hmatel = x2c_voov(b,l,i,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkbec >
                        hmatel = -x2c_voov(a,l,i,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkaeb >
                        hmatel = -x2c_voov(c,l,i,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likaec >
                        hmatel = -x2c_voov(b,l,j,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likbec >
                        hmatel = x2c_voov(a,l,j,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likaeb >
                        hmatel = x2c_voov(c,l,j,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijaec >
                        hmatel = x2c_voov(b,l,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijbec >
                        hmatel = -x2c_voov(a,l,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(2,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijaeb >
                        hmatel = -x2c_voov(c,l,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,5,6/), nub, nub, nob, nob, nloc, n3bbb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2C_voov,&
                  !$omp noa,nua,n3bbb_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkdbc >
                        hmatel = x2c_voov(a,l,i,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkdac >
                        hmatel = -x2c_voov(b,l,i,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | ljkdab >
                        hmatel = x2c_voov(c,l,i,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likdbc >
                        hmatel = -x2c_voov(a,l,j,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likdac >
                        hmatel = x2c_voov(b,l,j,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | likdab >
                        hmatel = -x2c_voov(c,l,j,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijdbc >
                        hmatel = x2c_voov(a,l,k,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijdac >
                        hmatel = -x2c_voov(b,l,k,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < ijkabc | h2a(voov) | lijdab >
                        hmatel = x2c_voov(c,l,k,d)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate temporary arrays
                  deallocate(excits_buff,amps_buff)

                  !!!! diagram 6a: A(i/jk)A(a/bc) h2b(maei) * r3c(ebcmjk)
                  ! allocate and copy over r3c arrays
                  allocate(amps_buff(n3abb_r),excits_buff(6,n3abb_r))
                  amps_buff(:) = r3c_amps(:)
                  excits_buff(:,:) = r3c_excits(:,:)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = nub*(nub-1)/2*nob*(nob-1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nob,nob))
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,5,6/), nub, nub, nob, nob, nloc, n3abb_r)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3bbb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < i~j~k~a~b~c~ | h2b(ovvo) | lj~k~db~c~ >
                        hmatel = h2b_ovvo(l,a,d,i)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < i~j~k~a~b~c~ | h2b(ovvo) | lj~k~da~c~ >
                        hmatel = -h2b_ovvo(l,b,d,i)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < i~j~k~a~b~c~ | h2b(ovvo) | lj~k~da~b~ >
                        hmatel = h2b_ovvo(l,c,d,i)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < i~j~k~a~b~c~ | h2b(ovvo) | li~k~db~c~ >
                        hmatel = -h2b_ovvo(l,a,d,j)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < i~j~k~a~b~c~ | h2b(ovvo) | li~k~da~c~ >
                        hmatel = h2b_ovvo(l,b,d,j)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < i~j~k~a~b~c~ | h2b(ovvo) | li~k~da~b~ >
                        hmatel = -h2b_ovvo(l,c,d,j)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < i~j~k~a~b~c~ | h2b(ovvo) | li~j~db~c~ >
                        hmatel = h2b_ovvo(l,a,d,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < i~j~k~a~b~c~ | h2b(ovvo) | li~j~da~c~ >
                        hmatel = -h2b_ovvo(l,b,d,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < i~j~k~a~b~c~ | h2b(ovvo) | li~j~da~b~ >
                        hmatel = h2b_ovvo(l,c,d,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                 end do
                 !$omp end do
                 !$omp end parallel
                 !!!! END OMP PARALLEL SECTION !!!!
                 ! deallocate sorting arrays
                 deallocate(loc_arr,idx_table)
                 ! deallocate temporary arrays
                 deallocate(amps_buff,excits_buff) 
                  !!!! diagram 6b: A(i/jk)A(a/bc) x2b(maei) * t3c(ebcmjk)
                  ! allocate and copy over t3c arrays
                  allocate(amps_buff(n3abb_t),excits_buff(6,n3abb_t))
                  amps_buff(:) = t3c_amps(:)
                  excits_buff(:,:) = t3c_excits(:,:)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = nub*(nub-1)/2*nob*(nob-1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nob,nob))
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,5,6/), nub, nub, nob, nob, nloc, n3abb_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3bbb_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                     a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                     i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < i~j~k~a~b~c~ | x2b(ovvo) | lj~k~db~c~ >
                        hmatel = x2b_ovvo(l,a,d,i)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < i~j~k~a~b~c~ | x2b(ovvo) | lj~k~da~c~ >
                        hmatel = -x2b_ovvo(l,b,d,i)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < i~j~k~a~b~c~ | x2b(ovvo) | lj~k~da~b~ >
                        hmatel = x2b_ovvo(l,c,d,i)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < i~j~k~a~b~c~ | x2b(ovvo) | li~k~db~c~ >
                        hmatel = -x2b_ovvo(l,a,d,j)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < i~j~k~a~b~c~ | x2b(ovvo) | li~k~da~c~ >
                        hmatel = x2b_ovvo(l,b,d,j)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < i~j~k~a~b~c~ | x2b(ovvo) | li~k~da~b~ >
                        hmatel = -x2b_ovvo(l,c,d,j)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < i~j~k~a~b~c~ | x2b(ovvo) | li~j~db~c~ >
                        hmatel = x2b_ovvo(l,a,d,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < i~j~k~a~b~c~ | x2b(ovvo) | li~j~da~c~ >
                        hmatel = -x2b_ovvo(l,b,d,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = excits_buff(1,jdet); l = excits_buff(4,jdet);
                        ! compute < i~j~k~a~b~c~ | x2b(ovvo) | li~j~da~b~ >
                        hmatel = x2b_ovvo(l,c,d,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                 end do
                 !$omp end do
                 !$omp end parallel
                 !!!! END OMP PARALLEL SECTION !!!!
                 ! deallocate sorting arrays
                 deallocate(loc_arr,idx_table)
                 ! deallocate temporary arrays
                 deallocate(amps_buff,excits_buff) 

                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3d_excits,&
                  !$omp r2c,t2c,&
                  !$omp H2C_vvov,H2C_vooo,&
                  !$omp X2C_vvov,X2C_vooo,&
                  !$omp nob,nub,n3bbb_r),&
                  !$omp private(idet,a,b,c,d,i,j,k,l,m,n,e,f,&
                  !$omp res_mm23)
                  !$omp do schedule(static)
                  do idet = 1, n3bbb_r
                      a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                      i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);

                      res_mm23 = 0.0d0
                      do e = 1, nub
                           ! A(i/jk)(c/ab) h2c(abie) * r2c(ecjk)
                          res_mm23 = res_mm23 + h2c_vvov(a,b,i,e) * r2c(e,c,j,k)
                          res_mm23 = res_mm23 - h2c_vvov(c,b,i,e) * r2c(e,a,j,k)
                          res_mm23 = res_mm23 - h2c_vvov(a,c,i,e) * r2c(e,b,j,k)
                          res_mm23 = res_mm23 - h2c_vvov(a,b,j,e) * r2c(e,c,i,k)
                          res_mm23 = res_mm23 + h2c_vvov(c,b,j,e) * r2c(e,a,i,k)
                          res_mm23 = res_mm23 + h2c_vvov(a,c,j,e) * r2c(e,b,i,k)
                          res_mm23 = res_mm23 - h2c_vvov(a,b,k,e) * r2c(e,c,j,i)
                          res_mm23 = res_mm23 + h2c_vvov(c,b,k,e) * r2c(e,a,j,i)
                          res_mm23 = res_mm23 + h2c_vvov(a,c,k,e) * r2c(e,b,j,i)
                           ! A(i/jk)(c/ab) x2c(abie) * t2c(ecjk)
                          res_mm23 = res_mm23 + x2c_vvov(a,b,i,e) * t2c(e,c,j,k)
                          res_mm23 = res_mm23 - x2c_vvov(c,b,i,e) * t2c(e,a,j,k)
                          res_mm23 = res_mm23 - x2c_vvov(a,c,i,e) * t2c(e,b,j,k)
                          res_mm23 = res_mm23 - x2c_vvov(a,b,j,e) * t2c(e,c,i,k)
                          res_mm23 = res_mm23 + x2c_vvov(c,b,j,e) * t2c(e,a,i,k)
                          res_mm23 = res_mm23 + x2c_vvov(a,c,j,e) * t2c(e,b,i,k)
                          res_mm23 = res_mm23 - x2c_vvov(a,b,k,e) * t2c(e,c,j,i)
                          res_mm23 = res_mm23 + x2c_vvov(c,b,k,e) * t2c(e,a,j,i)
                          res_mm23 = res_mm23 + x2c_vvov(a,c,k,e) * t2c(e,b,j,i)
                      end do
                      do m = 1, nob
                          ! -A(k/ij)A(a/bc) h2c(amij) * r2c(bcmk)
                          res_mm23 = res_mm23 - h2c_vooo(a,m,i,j) * r2c(b,c,m,k)
                          res_mm23 = res_mm23 + h2c_vooo(b,m,i,j) * r2c(a,c,m,k)
                          res_mm23 = res_mm23 + h2c_vooo(c,m,i,j) * r2c(b,a,m,k)
                          res_mm23 = res_mm23 + h2c_vooo(a,m,k,j) * r2c(b,c,m,i)
                          res_mm23 = res_mm23 - h2c_vooo(b,m,k,j) * r2c(a,c,m,i)
                          res_mm23 = res_mm23 - h2c_vooo(c,m,k,j) * r2c(b,a,m,i)
                          res_mm23 = res_mm23 + h2c_vooo(a,m,i,k) * r2c(b,c,m,j)
                          res_mm23 = res_mm23 - h2c_vooo(b,m,i,k) * r2c(a,c,m,j)
                          res_mm23 = res_mm23 - h2c_vooo(c,m,i,k) * r2c(b,a,m,j)
                          ! -A(k/ij)A(a/bc) x2c(amij) * t2c(bcmk)
                          res_mm23 = res_mm23 - x2c_vooo(a,m,i,j) * t2c(b,c,m,k)
                          res_mm23 = res_mm23 + x2c_vooo(b,m,i,j) * t2c(a,c,m,k)
                          res_mm23 = res_mm23 + x2c_vooo(c,m,i,j) * t2c(b,a,m,k)
                          res_mm23 = res_mm23 + x2c_vooo(a,m,k,j) * t2c(b,c,m,i)
                          res_mm23 = res_mm23 - x2c_vooo(b,m,k,j) * t2c(a,c,m,i)
                          res_mm23 = res_mm23 - x2c_vooo(c,m,k,j) * t2c(b,a,m,i)
                          res_mm23 = res_mm23 + x2c_vooo(a,m,i,k) * t2c(b,c,m,j)
                          res_mm23 = res_mm23 - x2c_vooo(b,m,i,k) * t2c(a,c,m,j)
                          res_mm23 = res_mm23 - x2c_vooo(c,m,i,k) * t2c(b,a,m,j)
                      end do
                      resid(idet) = resid(idet) + res_mm23
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!

              end subroutine build_hr_3d

              subroutine update_r(r1a,r1b,r2a,r2b,r2c,&
                                  r3a_amps, r3a_excits,&
                                  r3b_amps, r3b_excits,&
                                  r3c_amps, r3c_excits,&
                                  r3d_amps, r3d_excits,&
                                  omega,&
                                  h1a_oo, h1a_vv, h1b_oo, h1b_vv,&
                                  n3aaa, n3aab, n3abb, n3bbb,&
                                  noa, nua, nob, nub)

                      integer, intent(in) :: noa, nua, nob, nub, n3aaa, n3aab, n3abb, n3bbb
                      integer, intent(in) :: r3a_excits(6,n3aaa), r3b_excits(6,n3aab), r3c_excits(6,n3abb), r3d_excits(6,n3bbb)
                      real(kind=8), intent(in) :: h1a_oo(noa,noa), h1a_vv(nua,nua), h1b_oo(nob,nob), h1b_vv(nub,nub)
                      real(kind=8), intent(in) :: omega
                      
                      real(kind=8), intent(inout) :: r1a(1:nua,1:noa)
                      !f2py intent(in,out) :: r1a(0:nua-1,0:noa-1)
                      real(kind=8), intent(inout) :: r1b(1:nub,1:nob)
                      !f2py intent(in,out) :: r1b(0:nub-1,0:nob-1)
                      real(kind=8), intent(inout) :: r2a(1:nua,1:nua,1:noa,1:noa)
                      !f2py intent(in,out) :: r2a(0:nua-1,0:nua-1,0:noa-1,0:noa-1)
                      real(kind=8), intent(inout) :: r2b(1:nua,1:nub,1:noa,1:nob)
                      !f2py intent(in,out) :: r2b(0:nua-1,0:nub-1,0:noa-1,0:nob-1)
                      real(kind=8), intent(inout) :: r2c(1:nub,1:nub,1:nob,1:nob)
                      !f2py intent(in,out) :: r2c(0:nub-1,0:nub-1,0:nob-1,0:nob-1)
                      real(kind=8), intent(inout) :: r3a_amps(n3aaa)
                      !f2py intent(in,out) :: r3a_amps(0:n3aaa-1) 
                      real(kind=8), intent(inout) :: r3b_amps(n3aab)
                      !f2py intent(in,out) :: r3b_amps(0:n3aab-1) 
                      real(kind=8), intent(inout) :: r3c_amps(n3abb)
                      !f2py intent(in,out) :: r3c_amps(0:n3abb-1) 
                      real(kind=8), intent(inout) :: r3d_amps(n3bbb)
                      !f2py intent(in,out) :: r3d_amps(0:n3bbb-1) 
                      
                      integer :: i, j, k, a, b, c, idet
                      real(kind=8) :: denom

                      do i = 1,noa
                        do a = 1,nua
                          denom = H1A_vv(a,a) - H1A_oo(i,i)
                          r1a(a,i) = r1a(a,i)/(omega - denom)
                        end do
                      end do

                      do i = 1,nob
                        do a = 1,nub
                          denom = H1B_vv(a,a) - H1B_oo(i,i)
                          r1b(a,i) = r1b(a,i)/(omega - denom)
                        end do
                      end do

                      do i = 1,noa
                        do j = 1,noa
                          do a = 1,nua
                            do b = 1,nua
                              denom = H1A_vv(a,a) + H1A_vv(b,b) - H1A_oo(i,i) - H1A_oo(j,j)
                              r2a(b,a,j,i) = r2a(b,a,j,i)/(omega - denom)
                            end do
                          end do
                        end do
                      end do

                      do j = 1,nob
                        do i = 1,noa
                          do b = 1,nub
                            do a = 1,nua
                              denom = H1A_vv(a,a) + H1B_vv(b,b) - H1A_oo(i,i) - H1B_oo(j,j)
                              r2b(a,b,i,j) = r2b(a,b,i,j)/(omega - denom)
                            end do
                          end do
                        end do
                      end do

                      do i = 1,nob
                        do j = 1,nob
                          do a = 1,nub
                            do b = 1,nub
                              denom = H1B_vv(a,a) + H1B_vv(b,b) - H1B_oo(i,i) - H1B_oo(j,j)
                              r2c(b,a,j,i) = r2c(b,a,j,i)/(omega - denom)
                            end do
                          end do
                        end do
                      end do

                      do idet = 1, n3aaa
                         a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                         i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);

                         denom = H1A_vv(a,a) + H1A_vv(b,b) + H1A_vv(c,c)&
                                -H1A_oo(i,i) - H1A_oo(j,j) - H1A_oo(k,k)
                         
                         r3a_amps(idet) = r3a_amps(idet)/(omega - denom)
                      end do

                      do idet = 1, n3aab
                         a = r3b_excits(1,idet); b = r3b_excits(2,idet); c = r3b_excits(3,idet);
                         i = r3b_excits(4,idet); j = r3b_excits(5,idet); k = r3b_excits(6,idet);

                         denom = H1A_vv(a,a) + H1A_vv(b,b) + H1B_vv(c,c)&
                                -H1A_oo(i,i) - H1A_oo(j,j) - H1B_oo(k,k)
                         
                         r3b_amps(idet) = r3b_amps(idet)/(omega - denom)
                      end do

                      do idet = 1, n3abb
                         a = r3c_excits(1,idet); b = r3c_excits(2,idet); c = r3c_excits(3,idet);
                         i = r3c_excits(4,idet); j = r3c_excits(5,idet); k = r3c_excits(6,idet);

                         denom = H1A_vv(a,a) + H1B_vv(b,b) + H1B_vv(c,c)&
                                -H1A_oo(i,i) - H1B_oo(j,j) - H1B_oo(k,k)
                         
                         r3c_amps(idet) = r3c_amps(idet)/(omega - denom)
                      end do

                      do idet = 1, n3bbb
                         a = r3d_excits(1,idet); b = r3d_excits(2,idet); c = r3d_excits(3,idet);
                         i = r3d_excits(4,idet); j = r3d_excits(5,idet); k = r3d_excits(6,idet);

                         denom = H1B_vv(a,a) + H1B_vv(b,b) + H1B_vv(c,c)&
                                -H1B_oo(i,i) - H1B_oo(j,j) - H1B_oo(k,k)
                         
                         r3d_amps(idet) = r3d_amps(idet)/(omega - denom)
                      end do

              end subroutine update_r

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

              subroutine sort4(excits, amps, loc_arr, idx_table, idims, n1, n2, n3, n4, nloc, n3p, x1a)

                    integer, intent(in) :: n1, n2, n3, n4, nloc, n3p
                    integer, intent(in) :: idims(4)
                    integer, intent(in) :: idx_table(n1,n2,n3,n4)
      
                    integer, intent(inout) :: loc_arr(2,nloc)
                    integer, intent(inout) :: excits(6,n3p)
                    real(kind=8), intent(inout) :: amps(n3p)
                    real(kind=8), intent(inout), optional :: x1a(n3p)
      
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
                    if (present(x1a)) x1a = x1a(idx)
                    deallocate(temp,idx)
      
                    loc_arr(1,:) = 1; loc_arr(2,:) = 0;
                    do idet = 1, n3p-1
                       p1 = excits(idims(1),idet);   q1 = excits(idims(2),idet);   r1 = excits(idims(3),idet);   s1 = excits(idims(4),idet)
                       p2 = excits(idims(1),idet+1); q2 = excits(idims(2),idet+1); r2 = excits(idims(3),idet+1); s2 = excits(idims(4),idet+1)
                       pqrs1 = idx_table(p1,q1,r1,s1)
                       pqrs2 = idx_table(p2,q2,r2,s2)
                       if (pqrs1 /= pqrs2) then
                          loc_arr(2,pqrs1) = idet
                          loc_arr(1,pqrs2) = idet+1
                       end if
                    end do
                    loc_arr(2,pqrs2) = n3p

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

end module eomccsdt_p_loops
 
