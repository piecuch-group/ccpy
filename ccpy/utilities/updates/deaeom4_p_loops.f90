module deaeom4_p_loops

      use omp_lib

      implicit none
	
      contains

!              subroutine build_hr_1b(x1b,&
!                                     r3b_amps, r3b_excits,&
!                                     r3c_amps, r3c_excits,&
!                                     r3d_amps, r3d_excits,&
!                                     h2a_oovv, h2b_oovv, h2c_oovv,&
!                                     n3aab, n3abb, n3bbb,&
!                                     noa, nua, nob, nub)
!
!                      integer, intent(in) :: noa, nua, nob, nub, n3aab, n3abb, n3bbb
!                      integer, intent(in) :: r3b_excits(n3aab,6), r3c_excits(n3abb,6), r3d_excits(n3bbb,6)
!                      real(kind=8), intent(in) :: r3b_amps(n3aab), r3c_amps(n3abb), r3d_amps(n3bbb)
!                      real(kind=8), intent(in) :: h2a_oovv(1:noa,1:noa,1:nua,1:nua),&
!                                                  h2b_oovv(1:noa,1:nob,1:nua,1:nub),&
!                                                  h2c_oovv(1:nob,1:nob,1:nub,1:nub)
!
!                      real(kind=8), intent(inout) :: x1b(1:nub,1:nob)
!                      !f2py intent(in,out) :: x1b(0:nub-1,0:nob-1)
!
!                      integer :: i, a, m, n, e, f, idet
!                      real(kind=8) :: denom, val, r_amp
!
!                      ! compute < i~a~ | (H(2) * R3)_C | 0 >
!                      do idet = 1, n3aab
!                          r_amp = r3b_amps(idet)
!                          ! h2a(mnef) * r3b(efamni)
!                          e = r3b_excits(idet,1); f = r3b_excits(idet,2); a = r3b_excits(idet,3);
!                          m = r3b_excits(idet,4); n = r3b_excits(idet,5); i = r3b_excits(idet,6);
!                          x1b(a,i) = x1b(a,i) + h2a_oovv(m,n,e,f) * r_amp ! (1)
!                      end do
!                      do idet = 1, n3abb
!                          r_amp = r3c_amps(idet)
!                          ! A(af)A(in) h2b(mnef) * r3c(efamni)
!                          e = r3c_excits(idet,1); f = r3c_excits(idet,2); a = r3c_excits(idet,3);
!                          m = r3c_excits(idet,4); n = r3c_excits(idet,5); i = r3c_excits(idet,6);
!                          x1b(a,i) = x1b(a,i) + h2b_oovv(m,n,e,f) * r_amp ! (1)
!                          x1b(f,i) = x1b(f,i) - h2b_oovv(m,n,e,a) * r_amp ! (af)
!                          x1b(a,n) = x1b(a,n) - h2b_oovv(m,i,e,f) * r_amp ! (in)
!                          x1b(f,n) = x1b(f,n) + h2b_oovv(m,i,e,a) * r_amp ! (af)(in)
!                      end do
!                      do idet = 1, n3bbb
!                          r_amp = r3d_amps(idet)
!                          ! A(a/ef)A(i/mn) h2c(mnef) * r3d(aefimn)
!                          a = r3d_excits(idet,1); e = r3d_excits(idet,2); f = r3d_excits(idet,3);
!                          i = r3d_excits(idet,4); m = r3d_excits(idet,5); n = r3d_excits(idet,6);
!                          x1b(a,i) = x1b(a,i) + h2c_oovv(m,n,e,f) * r_amp ! (1)
!                          x1b(e,i) = x1b(e,i) - h2c_oovv(m,n,a,f) * r_amp ! (ae)
!                          x1b(f,i) = x1b(f,i) - h2c_oovv(m,n,e,a) * r_amp ! (af)
!                          x1b(a,m) = x1b(a,m) - h2c_oovv(i,n,e,f) * r_amp ! (im)
!                          x1b(e,m) = x1b(e,m) + h2c_oovv(i,n,a,f) * r_amp ! (ae)(im)
!                          x1b(f,m) = x1b(f,m) + h2c_oovv(i,n,e,a) * r_amp ! (af)(im)
!                          x1b(a,n) = x1b(a,n) - h2c_oovv(m,i,e,f) * r_amp ! (in)
!                          x1b(e,n) = x1b(e,n) + h2c_oovv(m,i,a,f) * r_amp ! (ae)(in)
!                          x1b(f,n) = x1b(f,n) + h2c_oovv(m,i,e,a) * r_amp ! (af)(in)
!                      end do
!              end subroutine build_hr_1b
!
!              subroutine build_hr_2b(sigma_2b,&
!                                     r3b_amps, r3b_excits,&
!                                     r3c_amps, r3c_excits,&
!                                     t3b_amps, t3b_excits,&
!                                     t3c_amps, t3c_excits,&
!                                     h1a_ov, h1b_ov,&
!                                     h2a_ooov, h2a_vovv,&
!                                     h2b_ooov, h2b_vovv, h2b_oovo, h2b_ovvv,&
!                                     h2c_ooov, h2c_vovv,&
!                                     x1a_ov, x1b_ov,&
!                                     n3aab_r, n3abb_r,&
!                                     n3aab_t, n3abb_t,&
!                                     noa, nua, nob, nub)
!                  ! Input dimension variables
!                  integer, intent(in) :: noa, nua, nob, nub
!                  integer, intent(in) :: n3aab_r, n3aab_t, n3abb_r, n3abb_t
!                  ! Input R and T arrays
!                  integer, intent(in) :: r3b_excits(n3aab_r,6), r3c_excits(n3abb_r,6)
!                  integer, intent(in) :: t3b_excits(n3aab_t,6), t3c_excits(n3abb_t,6)
!                  real(kind=8), intent(in) :: r3b_amps(n3aab_r), r3c_amps(n3abb_r)
!                  real(kind=8), intent(in) :: t3b_amps(n3aab_t), t3c_amps(n3abb_t)
!                  ! Input H and X arrays
!                  real(kind=8), intent(in) :: h1a_ov(noa,nua), h1b_ov(nob,nub)
!                  real(kind=8), intent(in) :: h2a_ooov(noa,noa,noa,nua)
!                  real(kind=8), intent(in) :: h2a_vovv(nua,noa,nua,nua)
!                  real(kind=8), intent(in) :: h2b_ooov(noa,nob,noa,nub)
!                  real(kind=8), intent(in) :: h2b_vovv(nua,nob,nua,nub)
!                  real(kind=8), intent(in) :: h2b_oovo(noa,nob,nua,nob)
!                  real(kind=8), intent(in) :: h2b_ovvv(noa,nub,nua,nub)
!                  real(kind=8), intent(in) :: h2c_ooov(nob,nob,nob,nub)
!                  real(kind=8), intent(in) :: h2c_vovv(nub,nob,nub,nub)
!                  real(kind=8), intent(in) :: x1a_ov(noa,nua), x1b_ov(nob,nub)
!                  ! Output and Inout variables
!                  real(kind=8), intent(inout) :: sigma_2b(nua,nub,noa,nob)
!                  !f2py intent(in,out) :: sigma_2b(0:nua-1,0:nub-1,0:noa-1,0:nob-1)
!                  ! Local variables
!                  real(kind=8) :: t_amp, r_amp, val
!                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet
!
!                  do idet = 1, n3aab_r
!                      r_amp = r3b_amps(idet)
!
!                      ! A(af) -h2a(mnif) * r3b(afbmnj)
!                      a = r3b_excits(idet,1); f = r3b_excits(idet,2); b = r3b_excits(idet,3);
!                      m = r3b_excits(idet,4); n = r3b_excits(idet,5); j = r3b_excits(idet,6);
!                      sigma_2b(a,b,:,j) = sigma_2b(a,b,:,j) - h2a_ooov(m,n,:,f) * r_amp ! (1)
!                      sigma_2b(f,b,:,j) = sigma_2b(f,b,:,j) + h2a_ooov(m,n,:,a) * r_amp ! (af)
!
!                      ! A(af)A(in) -h2b(nmfj) * r3b(afbinm)
!                      a = r3b_excits(idet,1); f = r3b_excits(idet,2); b = r3b_excits(idet,3);
!                      i = r3b_excits(idet,4); n = r3b_excits(idet,5); m = r3b_excits(idet,6);
!                      sigma_2b(a,b,i,:) = sigma_2b(a,b,i,:) - h2b_oovo(n,m,f,:) * r_amp ! (1)
!                      sigma_2b(f,b,i,:) = sigma_2b(f,b,i,:) + h2b_oovo(n,m,a,:) * r_amp ! (af)
!                      sigma_2b(a,b,n,:) = sigma_2b(a,b,n,:) + h2b_oovo(i,m,f,:) * r_amp ! (in)
!                      sigma_2b(f,b,n,:) = sigma_2b(f,b,n,:) - h2b_oovo(i,m,a,:) * r_amp ! (af)(in)
!
!                      ! A(in) h2a(anef) * r3b(efbinj)
!                      e = r3b_excits(idet,1); f = r3b_excits(idet,2); b = r3b_excits(idet,3);
!                      i = r3b_excits(idet,4); n = r3b_excits(idet,5); j = r3b_excits(idet,6);
!                      sigma_2b(:,b,i,j) = sigma_2b(:,b,i,j) + h2a_vovv(:,n,e,f) * r_amp ! (1)
!                      sigma_2b(:,b,n,j) = sigma_2b(:,b,n,j) - h2a_vovv(:,i,e,f) * r_amp ! (in)
!
!                      ! A(af)A(in) h2b(nbfe) * r3b(afeinj)
!                      a = r3b_excits(idet,1); f = r3b_excits(idet,2); e = r3b_excits(idet,3);
!                      i = r3b_excits(idet,4); n = r3b_excits(idet,5); j = r3b_excits(idet,6);
!                      sigma_2b(a,:,i,j) = sigma_2b(a,:,i,j) + h2b_ovvv(n,:,f,e) * r_amp ! (1)
!                      sigma_2b(f,:,i,j) = sigma_2b(f,:,i,j) - h2b_ovvv(n,:,a,e) * r_amp ! (af)
!                      sigma_2b(a,:,n,j) = sigma_2b(a,:,n,j) - h2b_ovvv(i,:,f,e) * r_amp ! (in)
!                      sigma_2b(f,:,n,j) = sigma_2b(f,:,n,j) + h2b_ovvv(i,:,a,e) * r_amp ! (af)(in)
!
!                      ! A(ae)A(im) h1a(me) * r3b(aebimj)
!                      a = r3b_excits(idet,1); e = r3b_excits(idet,2); b = r3b_excits(idet,3);
!                      i = r3b_excits(idet,4); m = r3b_excits(idet,5); j = r3b_excits(idet,6);
!                      sigma_2b(a,b,i,j) = sigma_2b(a,b,i,j) + h1a_ov(m,e) * r_amp ! (1)
!                      sigma_2b(a,b,m,j) = sigma_2b(a,b,m,j) - h1a_ov(i,e) * r_amp ! (im)
!                      sigma_2b(e,b,i,j) = sigma_2b(e,b,i,j) - h1a_ov(m,a) * r_amp ! (ae)
!                      sigma_2b(e,b,m,j) = sigma_2b(e,b,m,j) + h1a_ov(i,a) * r_amp ! (im)(ae)
!                  end do
!                  do idet = 1, n3aab_t
!                      t_amp = t3b_amps(idet)
!                      ! A(ae)A(im) x1(me) * t3b(aebimj)
!                      a = t3b_excits(idet,1); e = t3b_excits(idet,2); b = t3b_excits(idet,3);
!                      i = t3b_excits(idet,4); m = t3b_excits(idet,5); j = t3b_excits(idet,6);
!                      sigma_2b(a,b,i,j) = sigma_2b(a,b,i,j) + x1a_ov(m,e) * t_amp ! (1)
!                      sigma_2b(a,b,m,j) = sigma_2b(a,b,m,j) - x1a_ov(i,e) * t_amp ! (im)
!                      sigma_2b(e,b,i,j) = sigma_2b(e,b,i,j) - x1a_ov(m,a) * t_amp ! (ae)
!                      sigma_2b(e,b,m,j) = sigma_2b(e,b,m,j) + x1a_ov(i,a) * t_amp ! (im)(ae)
!                  end do
!                  do idet = 1, n3abb_r
!                      r_amp = r3c_amps(idet)
!
!                      ! A(bf) -h2c(mnjf) * r3c(afbinm)
!                      a = r3c_excits(idet,1); f = r3c_excits(idet,2); b = r3c_excits(idet,3);
!                      i = r3c_excits(idet,4); n = r3c_excits(idet,5); m = r3c_excits(idet,6);
!                      sigma_2b(a,b,i,:) = sigma_2b(a,b,i,:) - h2c_ooov(m,n,:,f) * r_amp ! (1)
!                      sigma_2b(a,f,i,:) = sigma_2b(a,f,i,:) + h2c_ooov(m,n,:,b) * r_amp ! (bf)
!
!                      ! A(bf)A(jn) -h2b(mnif) * r3c(afbmnj)
!                      a = r3c_excits(idet,1); f = r3c_excits(idet,2); b = r3c_excits(idet,3);
!                      m = r3c_excits(idet,4); n = r3c_excits(idet,5); j = r3c_excits(idet,6);
!                      sigma_2b(a,b,:,j) = sigma_2b(a,b,:,j) - h2b_ooov(m,n,:,f) * r_amp ! (1)
!                      sigma_2b(a,f,:,j) = sigma_2b(a,f,:,j) + h2b_ooov(m,n,:,b) * r_amp ! (bf)
!                      sigma_2b(a,b,:,n) = sigma_2b(a,b,:,n) + h2b_ooov(m,j,:,f) * r_amp ! (jn)
!                      sigma_2b(a,f,:,n) = sigma_2b(a,f,:,n) - h2b_ooov(m,j,:,b) * r_amp ! (bf)(jn)
!
!                      ! A(jn) h2c(bnef) * r3c(afeinj)
!                      a = r3c_excits(idet,1); f = r3c_excits(idet,2); e = r3c_excits(idet,3);
!                      i = r3c_excits(idet,4); n = r3c_excits(idet,5); j = r3c_excits(idet,6);
!                      sigma_2b(a,:,i,j) = sigma_2b(a,:,i,j) + h2c_vovv(:,n,e,f) * r_amp ! (1)
!                      sigma_2b(a,:,i,n) = sigma_2b(a,:,i,n) - h2c_vovv(:,j,e,f) * r_amp ! (jn)
!
!                      ! A(bf)A(jn) h2b(anef) * r3c(efbinj)
!                      e = r3c_excits(idet,1); f = r3c_excits(idet,2); b = r3c_excits(idet,3);
!                      i = r3c_excits(idet,4); n = r3c_excits(idet,5); j = r3c_excits(idet,6);
!                      sigma_2b(:,b,i,j) = sigma_2b(:,b,i,j) + h2b_vovv(:,n,e,f) * r_amp ! (1)
!                      sigma_2b(:,f,i,j) = sigma_2b(:,f,i,j) - h2b_vovv(:,n,e,b) * r_amp ! (bf)
!                      sigma_2b(:,b,i,n) = sigma_2b(:,b,i,n) - h2b_vovv(:,j,e,f) * r_amp ! (jn)
!                      sigma_2b(:,f,i,n) = sigma_2b(:,f,i,n) + h2b_vovv(:,j,e,b) * r_amp ! (bf)(jn)
!
!                      ! [A(be)A(mj) h1b(me) * r3c(aebimj)]
!                      a = r3c_excits(idet,1); e = r3c_excits(idet,2); b = r3c_excits(idet,3);
!                      i = r3c_excits(idet,4); m = r3c_excits(idet,5); j = r3c_excits(idet,6);
!                      sigma_2b(a,b,i,j) = sigma_2b(a,b,i,j) + h1b_ov(m,e) * r_amp ! (1)
!                      sigma_2b(a,b,i,m) = sigma_2b(a,b,i,m) - h1b_ov(j,e) * r_amp ! (jm)
!                      sigma_2b(a,e,i,j) = sigma_2b(a,e,i,j) - h1b_ov(m,b) * r_amp ! (be)
!                      sigma_2b(a,e,i,m) = sigma_2b(a,e,i,m) + h1b_ov(j,b) * r_amp ! (jm)(be)
!                  end do
!                  do idet = 1, n3abb_t
!                      t_amp = t3c_amps(idet)
!                      ! [A(be)A(mj) h1b(me) * t3c(aebimj)]
!                      a = t3c_excits(idet,1); e = t3c_excits(idet,2); b = t3c_excits(idet,3);
!                      i = t3c_excits(idet,4); m = t3c_excits(idet,5); j = t3c_excits(idet,6);
!                      sigma_2b(a,b,i,j) = sigma_2b(a,b,i,j) + x1b_ov(m,e) * t_amp ! (1)
!                      sigma_2b(a,b,i,m) = sigma_2b(a,b,i,m) - x1b_ov(j,e) * t_amp ! (jm)
!                      sigma_2b(a,e,i,j) = sigma_2b(a,e,i,j) - x1b_ov(m,b) * t_amp ! (be)
!                      sigma_2b(a,e,i,m) = sigma_2b(a,e,i,m) + x1b_ov(j,b) * t_amp ! (jm)(be)
!                  end do
!
!              end subroutine build_hr_2b
!
!              subroutine build_hr_2c(sigma_2c,&
!                                     r3c_amps, r3c_excits,&
!                                     r3d_amps, r3d_excits,&
!                                     t3c_amps, t3c_excits,&
!                                     t3d_amps, t3d_excits,&
!                                     h1a_ov, h1b_ov,&
!                                     h2b_oovo, h2b_ovvv,&
!                                     h2c_ooov, h2c_vovv,&
!                                     x1a_ov, x1b_ov,&
!                                     n3abb_r, n3bbb_r,&
!                                     n3abb_t, n3bbb_t,&
!                                     noa, nua, nob, nub)
!                  ! Input dimension variables
!                  integer, intent(in) :: noa, nua, nob, nub
!                  integer, intent(in) :: n3abb_r, n3abb_t, n3bbb_r, n3bbb_t
!                  ! Input R and T arrays
!                  integer, intent(in) :: r3c_excits(n3abb_r,6), r3d_excits(n3bbb_r,6)
!                  integer, intent(in) :: t3c_excits(n3abb_t,6), t3d_excits(n3bbb_t,6)
!                  real(kind=8), intent(in) :: r3c_amps(n3abb_r), r3d_amps(n3bbb_r)
!                  real(kind=8), intent(in) :: t3c_amps(n3abb_t), t3d_amps(n3bbb_t)
!                  ! Input H and X arrays
!                  real(kind=8), intent(in) :: h1a_ov(noa,nua), h1b_ov(nob,nub)
!                  real(kind=8), intent(in) :: h2b_oovo(noa,nob,nua,nob)
!                  real(kind=8), intent(in) :: h2b_ovvv(noa,nub,nua,nub)
!                  real(kind=8), intent(in) :: h2c_ooov(nob,nob,nob,nub)
!                  real(kind=8), intent(in) :: h2c_vovv(nub,nob,nub,nub)
!                  real(kind=8), intent(in) :: x1a_ov(noa,nua), x1b_ov(nob,nub)
!                  ! Output and Inout variables
!                  real(kind=8), intent(inout) :: sigma_2c(nub,nub,nob,nob)
!                  !f2py intent(in,out) :: sigma_2c(0:nub-1,0:nub-1,0:nob-1,0:nob-1)
!                  ! Local variables
!                  real(kind=8) :: t_amp, r_amp, val
!                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet
!
!                  ! compute < ijab | (H(2) * T3)_C | 0 >
!                  do idet = 1, n3abb_r
!                      r_amp = r3c_amps(idet)
!
!                      ! A(ij)A(ab) [h1a(me) * r3c(eabmij)]
!                      e = r3c_excits(idet,1); a = r3c_excits(idet,2); b = r3c_excits(idet,3);
!                      m = r3c_excits(idet,4); i = r3c_excits(idet,5); j = r3c_excits(idet,6);
!                      sigma_2c(a,b,i,j) = sigma_2c(a,b,i,j) + h1a_ov(m,e) * r_amp ! (1)
!
!                      ! A(ij)A(ab) [A(be) h2b(nafe) * r3c(febnij)]
!                      f = r3c_excits(idet,1); e = r3c_excits(idet,2); b = r3c_excits(idet,3);
!                      n = r3c_excits(idet,4); i = r3c_excits(idet,5); j = r3c_excits(idet,6);
!                      sigma_2c(:,b,i,j) = sigma_2c(:,b,i,j) + h2b_ovvv(n,:,f,e) * r_amp ! (1)
!                      sigma_2c(:,e,i,j) = sigma_2c(:,e,i,j) - h2b_ovvv(n,:,f,b) * r_amp ! (be)
!
!                      ! A(ij)A(ab) [A(jm) -h2b(nmfi) * r3c(fabnmj)]
!                      f = r3c_excits(idet,1); a = r3c_excits(idet,2); b = r3c_excits(idet,3);
!                      n = r3c_excits(idet,4); m = r3c_excits(idet,5); j = r3c_excits(idet,6);
!                      sigma_2c(a,b,:,j) = sigma_2c(a,b,:,j) - h2b_oovo(n,m,f,:) * r_amp ! (1)
!                      sigma_2c(a,b,:,m) = sigma_2c(a,b,:,m) + h2b_oovo(n,j,f,:) * r_amp ! (jm)
!                  end do
!                  do idet = 1, n3bbb_r
!                      r_amp = r3d_amps(idet)
!
!                      ! A(ij)A(ab) [A(m/ij)A(e/ab) h1b(me) * r3d(abeijm)]
!                      a = r3d_excits(idet,1); b = r3d_excits(idet,2); e = r3d_excits(idet,3);
!                      i = r3d_excits(idet,4); j = r3d_excits(idet,5); m = r3d_excits(idet,6);
!                      sigma_2c(a,b,i,j) = sigma_2c(a,b,i,j) + h1b_ov(m,e) * r_amp ! (1)
!                      sigma_2c(a,b,m,j) = sigma_2c(a,b,m,j) - h1b_ov(i,e) * r_amp ! (im)
!                      sigma_2c(a,b,i,m) = sigma_2c(a,b,i,m) - h1b_ov(j,e) * r_amp ! (jm)
!                      sigma_2c(e,b,i,j) = sigma_2c(e,b,i,j) - h1b_ov(m,a) * r_amp ! (ae)
!                      sigma_2c(e,b,m,j) = sigma_2c(e,b,m,j) + h1b_ov(i,a) * r_amp ! (im)(ae)
!                      sigma_2c(e,b,i,m) = sigma_2c(e,b,i,m) + h1b_ov(j,a) * r_amp ! (jm)(ae)
!                      sigma_2c(a,e,i,j) = sigma_2c(a,e,i,j) - h1b_ov(m,b) * r_amp ! (be)
!                      sigma_2c(a,e,m,j) = sigma_2c(a,e,m,j) + h1b_ov(i,b) * r_amp ! (im)(be)
!                      sigma_2c(a,e,i,m) = sigma_2c(a,e,i,m) + h1b_ov(j,b) * r_amp ! (jm)(be)
!
!                      ! A(ij)A(ab) [A(j/mn)A(f/ab) -h2c(mnif) * r3d(abfmjn)]
!                      a = r3d_excits(idet,1); b = r3d_excits(idet,2); f = r3d_excits(idet,3);
!                      m = r3d_excits(idet,4); j = r3d_excits(idet,5); n = r3d_excits(idet,6);
!                      sigma_2c(a,b,:,j) = sigma_2c(a,b,:,j) - h2c_ooov(m,n,:,f) * r_amp ! (1)
!                      sigma_2c(a,b,:,m) = sigma_2c(a,b,:,m) + h2c_ooov(j,n,:,f) * r_amp ! (jm)
!                      sigma_2c(a,b,:,n) = sigma_2c(a,b,:,n) + h2c_ooov(m,j,:,f) * r_amp ! (jn)
!                      sigma_2c(f,b,:,j) = sigma_2c(f,b,:,j) + h2c_ooov(m,n,:,a) * r_amp ! (af)
!                      sigma_2c(f,b,:,m) = sigma_2c(f,b,:,m) - h2c_ooov(j,n,:,a) * r_amp ! (jm)(af)
!                      sigma_2c(f,b,:,n) = sigma_2c(f,b,:,n) - h2c_ooov(m,j,:,a) * r_amp ! (jn)(af)
!                      sigma_2c(a,f,:,j) = sigma_2c(a,f,:,j) + h2c_ooov(m,n,:,b) * r_amp ! (bf)
!                      sigma_2c(a,f,:,m) = sigma_2c(a,f,:,m) - h2c_ooov(j,n,:,b) * r_amp ! (jm)(bf)
!                      sigma_2c(a,f,:,n) = sigma_2c(a,f,:,n) - h2c_ooov(m,j,:,b) * r_amp ! (jn)(bf)
!
!                      ! A(ij)A(ab) [A(n/ij)A(b/ef) h2c(anef) * r3d(ebfijn)]
!                      e = r3d_excits(idet,1); b = r3d_excits(idet,2); f = r3d_excits(idet,3);
!                      i = r3d_excits(idet,4); j = r3d_excits(idet,5); n = r3d_excits(idet,6);
!                      sigma_2c(:,b,i,j) = sigma_2c(:,b,i,j) + h2c_vovv(:,n,e,f) * r_amp ! (1)
!                      sigma_2c(:,b,n,j) = sigma_2c(:,b,n,j) - h2c_vovv(:,i,e,f) * r_amp ! (in)
!                      sigma_2c(:,b,i,n) = sigma_2c(:,b,i,n) - h2c_vovv(:,j,e,f) * r_amp ! (jn)
!                      sigma_2c(:,e,i,j) = sigma_2c(:,e,i,j) - h2c_vovv(:,n,b,f) * r_amp ! (be)
!                      sigma_2c(:,e,n,j) = sigma_2c(:,e,n,j) + h2c_vovv(:,i,b,f) * r_amp ! (in)(be)
!                      sigma_2c(:,e,i,n) = sigma_2c(:,e,i,n) + h2c_vovv(:,j,b,f) * r_amp ! (jn)(be)
!                      sigma_2c(:,f,i,j) = sigma_2c(:,f,i,j) - h2c_vovv(:,n,e,b) * r_amp ! (bf)
!                      sigma_2c(:,f,n,j) = sigma_2c(:,f,n,j) + h2c_vovv(:,i,e,b) * r_amp ! (in)(bf)
!                      sigma_2c(:,f,i,n) = sigma_2c(:,f,i,n) + h2c_vovv(:,j,e,b) * r_amp ! (jn)(bf)
!                  end do
!                  do idet = 1, n3abb_t
!                      t_amp = t3c_amps(idet)
!
!                      ! A(ij)A(ab) [h1a(me) * t3c(eabmij)]
!                      e = t3c_excits(idet,1); a = t3c_excits(idet,2); b = t3c_excits(idet,3);
!                      m = t3c_excits(idet,4); i = t3c_excits(idet,5); j = t3c_excits(idet,6);
!                      sigma_2c(a,b,i,j) = sigma_2c(a,b,i,j) + x1a_ov(m,e) * t_amp ! (1)
!                  end do
!                  do idet = 1, n3bbb_t
!                      t_amp = t3d_amps(idet)
!
!                      ! A(ij)A(ab) [A(m/ij)A(e/ab) x1b(me) * t3d(abeijm)]
!                      a = t3d_excits(idet,1); b = t3d_excits(idet,2); e = t3d_excits(idet,3);
!                      i = t3d_excits(idet,4); j = t3d_excits(idet,5); m = t3d_excits(idet,6);
!                      sigma_2c(a,b,i,j) = sigma_2c(a,b,i,j) + x1b_ov(m,e) * t_amp ! (1)
!                      sigma_2c(a,b,m,j) = sigma_2c(a,b,m,j) - x1b_ov(i,e) * t_amp ! (im)
!                      sigma_2c(a,b,i,m) = sigma_2c(a,b,i,m) - x1b_ov(j,e) * t_amp ! (jm)
!                      sigma_2c(e,b,i,j) = sigma_2c(e,b,i,j) - x1b_ov(m,a) * t_amp ! (ae)
!                      sigma_2c(e,b,m,j) = sigma_2c(e,b,m,j) + x1b_ov(i,a) * t_amp ! (im)(ae)
!                      sigma_2c(e,b,i,m) = sigma_2c(e,b,i,m) + x1b_ov(j,a) * t_amp ! (jm)(ae)
!                      sigma_2c(a,e,i,j) = sigma_2c(a,e,i,j) - x1b_ov(m,b) * t_amp ! (be)
!                      sigma_2c(a,e,m,j) = sigma_2c(a,e,m,j) + x1b_ov(i,b) * t_amp ! (im)(be)
!                      sigma_2c(a,e,i,m) = sigma_2c(a,e,i,m) + x1b_ov(j,b) * t_amp ! (jm)(be)
!                  end do
!                  ! antisymmetrize (this replaces the x2c -= np.transpose(x2c, (...)) stuff in vector update
!                  do i = 1, nob
!                      do j = i+1, nob
!                          do a = 1, nub
!                              do b = a+1, nub
!                                  val = sigma_2c(b,a,j,i) - sigma_2c(a,b,j,i) - sigma_2c(b,a,i,j) + sigma_2c(a,b,i,j)
!                                  sigma_2c(b,a,j,i) =  val
!                                  sigma_2c(a,b,j,i) = -val
!                                  sigma_2c(b,a,i,j) = -val
!                                  sigma_2c(a,b,i,j) =  val
!                              end do
!                          end do
!                      end do
!                  end do
!                  ! (H(2) * T3)_C terms are vectorized and generally broadcast to diagonal elements, which should
!                  ! be 0. Set them to 0 manually (you need to do this).
!                  do a = 1, nub
!                     sigma_2c(a,a,:,:) = 0.0d0
!                  end do
!                  do i = 1, nob
!                     sigma_2c(:,:,i,i) = 0.0d0
!                  end do
!
!              end subroutine build_hr_2c
         
              subroutine build_hr_4b(resid,&
                                     r4b_amps, r4b_excits,&
                                     h1a_vv,&
                                     h2a_vvvv,&
                                     n4abaa,&
                                     noa, nua, nob, nub)
                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n4abaa
                  ! Input H and X arrays
                  real(kind=8), intent(in) :: h1a_vv(nua,nua)
                  real(kind=8), intent(in) :: h2a_vvvv(nua,nua,nua,nua)
                  ! Output and Inout variables
                  real(kind=8), intent(out) :: resid(n4abaa)
                  integer, intent(inout) :: r4b_excits(n4abaa,6)
                  !f2py intent(in,out) :: r4b_excits(0:n4abaa-1,0:5)
                  real(kind=8), intent(inout) :: r4b_amps(n4abaa)
                  !f2py intent(in,out) :: r4b_amps(0:n4abaa-1)
                  ! Local variables
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:)
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)
                  real(kind=8) :: r_amp, hmatel, hmatel1, res_mm23
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc
                  
                  ! Zero the container that holds H*R
                  resid = 0.0d0

!                   # diagram 5: h1b(b~e~) r_abaa(ae~cdkl)
!                   x4b += (1.0 / 12.0) * np.einsum("be,aecdkl->abcdkl", H.b.vv, R.abaa, optimize=True)
!                   # diagram 6: -A(kl) h1a(ml) r_abaa(ab~cdkm)
!                   x4b -= (2.0 / 12.0) * np.einsum("ml,abcdkm->abcdkl", H.a.oo, R.abaa, optimize=True)
!                   # diagram 7: 1/2 h2a(mnkl) r_abaa(ab~cdmn)
!                   x4b += (1.0 / 24.0) * np.einsum("mnkl,abcdmn->abcdkl", H.aa.oooo, R.abaa, optimize=True)
!                   # diagram 9: A(a/cd) h2b(ab~ef~) r_abaa(ef~cdkl)
!                   x4b += (3.0 / 12.0) * np.einsum("abef,efcdkl->abcdkl", H.ab.vvvv, R.abaa, optimize=True)
!                   # diagram 10: A(d/ac)A(kl) h2a(dmle) r_abaa(ab~cekm)
!                   x4b += (6.0 / 12.0) * np.einsum("dmle,abcekm->abcdkl", H.aa.voov, R.abaa, optimize=True)
!                   # diagram 11: A(d/ac)A(kl) h2b(dm~le~) r_abab(ab~ce~km~)
!                   x4b += (6.0 / 12.0) * np.einsum("dmle,abcekm->abcdkl", H.ab.voov, R.abab, optimize=True)
!                   # diagram 12: -A(kl) h2b(mb~le~) r_abaa(ae~cdkm)
!                   x4b -= (2.0 / 12.0) * np.einsum("mble,aecdkm->abcdkl", H.ab.ovov, R.abaa, optimize=True)
                  
                  
                  !!!! diagram 4: A(d/ac) h1a(de) r4b(ab~cekl)
                  !!!! diagram 8: 1/2 A(a/cd) h2a(cdef) r4b(ab~efkl)
                  ! NOTE: WITHIN THESE LOOPS, H1A(VV) TERMS ARE DOUBLE-COUNTED SO COMPENSATE BY FACTOR OF 1/2
                  ! allocate new sorting arrays
                  nloc = noa*(noa-1)/2*nua*nub
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(noa,noa,nua,nub))
                  !!! SB: (5,6,1,2) -> KLAB~ LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nua-2/), (/1,nub/), noa, noa, nua, nub)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/5,6,1,2/), noa, noa, nua, nub, nloc, n4abaa, resid)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(k,l,a,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); f = r4b_excits(jdet,4);
                        ! compute < klab~cd | h2a(vvvv) | klab~ef >
                        hmatel = h2a_vvvv(c,d,e,f)
                        ! compute < klab~cd | h1a(vv) | klab~ef > = A(cd)A(ef) h1a(ce) delta(d,f)
                        hmatel1 = 0.0d0
                        if (d==f) hmatel1 = hmatel1 + h1a_vv(c,e) ! (1)
                        if (c==f) hmatel1 = hmatel1 - h1a_vv(d,e) ! (cd)
                        if (d==e) hmatel1 = hmatel1 - h1a_vv(c,f) ! (ef)
                        if (c==e) hmatel1 = hmatel1 + h1a_vv(d,f) ! (cd)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(k,l,c,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); f = r4b_excits(jdet,4);
                        ! compute < klab~cd | h2a(vvvv) | klcb~ef >
                        hmatel = -h2a_vvvv(a,d,e,f)
                        ! compute < klab~cd | h1a(vv) | klcb~ef > = -A(ad)A(ef) h1a(ae) delta(d,f)
                        hmatel1 = 0.0d0
                        if (d==f) hmatel1 = hmatel1 - h1a_vv(a,e) ! (1)
                        if (a==f) hmatel1 = hmatel1 + h1a_vv(d,e) ! (ad)
                        if (d==e) hmatel1 = hmatel1 + h1a_vv(a,f) ! (ef)
                        if (a==e) hmatel1 = hmatel1 - h1a_vv(d,f) ! (ad)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (ad)
                     idx = idx_table(k,l,d,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); f = r4b_excits(jdet,4);
                        ! compute < klab~cd | h2a(vvvv) | kldb~ef >
                        hmatel = h2a_vvvv(c,a,e,f)
                        ! compute < klab~cd | h1a(vv) | kldb~ef > = A(ac)A(ef) h1a(ce) delta(a,f)
                        hmatel1 = 0.0d0
                        if (a==f) hmatel1 = hmatel1 + h1a_vv(c,e) ! (1)
                        if (c==f) hmatel1 = hmatel1 - h1a_vv(a,e) ! (ac)
                        if (a==e) hmatel1 = hmatel1 - h1a_vv(c,f) ! (ef)
                        if (c==e) hmatel1 = hmatel1 + h1a_vv(a,f) ! (ac)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (5,6,3,2) -> KLCB~ LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/2,nua-1/), (/1,nub/), noa, noa, nua, nub)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/5,6,3,2/), noa, noa, nua, nub, nloc, n4abaa, resid)
                  do idet = 1, n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(k,l,c,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); f = r4b_excits(jdet,4);
                        ! compute < klab~cd | h2a(vvvv) | kleb~cf >
                        hmatel = h2a_vvvv(a,d,e,f)
                        ! compute < klab~cd | h1a(vv) | kleb~cf > = A(ad)A(ef) h1a(ae) delta(d,f)
                        hmatel1 = 0.0d0
                        if (d==f) hmatel1 = hmatel1 + h1a_vv(a,e) ! (1)
                        if (a==f) hmatel1 = hmatel1 - h1a_vv(d,e) ! (ad)
                        if (d==e) hmatel1 = hmatel1 - h1a_vv(a,f) ! (ef)
                        if (a==e) hmatel1 = hmatel1 + h1a_vv(d,f) ! (ad)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(k,l,a,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); f = r4b_excits(jdet,4);
                        ! compute < klab~cd | h2a(vvvv) | kleb~af >
                        hmatel = -h2a_vvvv(c,d,e,f)
                        ! compute < klab~cd | h1a(vv) | kleb~af > = -A(cd)A(ef) h1a(ce) delta(d,f)
                        hmatel1 = 0.0d0
                        if (d==f) hmatel1 = hmatel1 - h1a_vv(c,e) ! (1)
                        if (c==f) hmatel1 = hmatel1 + h1a_vv(d,e) ! (cd)
                        if (d==e) hmatel1 = hmatel1 + h1a_vv(c,f) ! (ef)
                        if (c==e) hmatel1 = hmatel1 - h1a_vv(d,f) ! (cd)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(k,l,d,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); f = r4b_excits(jdet,4);
                        ! compute < klab~cd | h2a(vvvv) | kleb~df >
                        hmatel = -h2a_vvvv(a,c,e,f)
                        ! compute < klab~cd | h1a(vv) | kleb~df > = -A(ac)A(ef) h1a(ae) delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 - h1a_vv(a,e) ! (1)
                        if (a==f) hmatel1 = hmatel1 + h1a_vv(c,e) ! (ac)
                        if (c==e) hmatel1 = hmatel1 + h1a_vv(a,f) ! (ef)
                        if (a==e) hmatel1 = hmatel1 - h1a_vv(c,f) ! (ac)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (5,6,4,2) -> KLDB~ LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/3,nua/), (/1,nub/), noa, noa, nua, nub)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/5,6,4,2/), noa, noa, nua, nub, nloc, n4abaa, resid)
  
                  
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  
              end subroutine build_hr_4b
         
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
                    integer, intent(inout) :: excits(n3p,6)
                    real(kind=8), intent(inout) :: amps(n3p)
                    real(kind=8), intent(inout), optional :: x1a(n3p)
      
                    integer :: idet
                    integer :: p, q, r, s
                    integer :: p1, q1, r1, s1, p2, q2, r2, s2
                    integer :: pqrs1, pqrs2
                    integer, allocatable :: temp(:), idx(:)
      
                    allocate(temp(n3p),idx(n3p))
                    do idet = 1, n3p
                       p = excits(idet,idims(1)); q = excits(idet,idims(2)); r = excits(idet,idims(3)); s = excits(idet,idims(4))
                       temp(idet) = idx_table(p,q,r,s)
                    end do
                    call argsort(temp, idx)
                    excits = excits(idx,:)
                    amps = amps(idx)
                    if (present(x1a)) x1a = x1a(idx)
                    deallocate(temp,idx)
      
                    loc_arr(1,:) = 1; loc_arr(2,:) = 0;
                    !!! WARNING: THERE IS A MEMORY LEAK HERE! pqrs2 is used below but is not set if n3p <= 1
                    !if (n3p <= 1) print*, "eomccsdt_p_loops >> WARNING: potential memory leakage in sort4 function. pqrs2 set to -1"
                    if (n3p == 1) then
                       if (excits(1,1)==1 .and. excits(1,2)==1 .and. excits(1,3)==1 .and. excits(1,4)==1 .and. excits(1,5)==1 .and. excits(1,6)==1) return
                       p2 = excits(n3p,idims(1)); q2 = excits(n3p,idims(2)); r2 = excits(n3p,idims(3)); s2 = excits(n3p,idims(4))
                       pqrs2 = idx_table(p2,q2,r2,s2)
                    else
                       pqrs2 = -1
                    end if
                    do idet = 1, n3p-1
                       p1 = excits(idet,idims(1));   q1 = excits(idet,idims(2));   r1 = excits(idet,idims(3));   s1 = excits(idet,idims(4))
                       p2 = excits(idet+1,idims(1)); q2 = excits(idet+1,idims(2)); r2 = excits(idet+1,idims(3)); s2 = excits(idet+1,idims(4))
                       pqrs1 = idx_table(p1,q1,r1,s1)
                       pqrs2 = idx_table(p2,q2,r2,s2)
                       if (pqrs1 /= pqrs2) then
                          loc_arr(2,pqrs1) = idet
                          loc_arr(1,pqrs2) = idet+1
                       end if
                    end do
                    !if (n3p > 1) then
                    loc_arr(2,pqrs2) = n3p
                    !end if
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

end module deaeom4_p_loops