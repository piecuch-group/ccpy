module eaeom3_p_loops

      use omp_lib

      implicit none

      contains

              subroutine build_hr_1a(sigma_1a,&
                                     r3a_amps,r3a_excits,&
                                     r3b_amps,r3b_excits,&
                                     r3c_amps,r3c_excits,&
                                     h2a_oovv,h2b_oovv,h2c_oovv,&
                                     n3aaa,n3aab,n3abb,&
                                     noa,nua,nob,nub)

                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa, n3aab, n3abb
                  ! Input R and T arrays
                  integer, intent(in) :: r3a_excits(n3aaa,5)
                  integer, intent(in) :: r3b_excits(n3aab,5)
                  integer, intent(in) :: r3c_excits(n3abb,5)
                  real(kind=8), intent(in) :: r3a_amps(n3aaa)
                  real(kind=8), intent(in) :: r3b_amps(n3aab)
                  real(kind=8), intent(in) :: r3c_amps(n3abb)
                  ! Input H  and X arrays
                  real(kind=8), intent(in) :: h2a_oovv(noa,noa,nua,nua)
                  real(kind=8), intent(in) :: h2b_oovv(noa,nob,nua,nub)
                  real(kind=8), intent(in) :: h2c_oovv(nob,nob,nub,nub)
                  ! Output and Inout variables
                  real(kind=8), intent(inout) :: sigma_1a(nua)
                  !f2py intent(in,out) :: sigma_1a(0:nua-1)
                  ! Local variables
                  real(kind=8) :: r_amp, val
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  do idet = 1,n3aaa
                     r_amp = r3a_amps(idet)
                     ! x1(a) <- A(a/ef) h2a(mnef)*r3a(aefmn)
                     a = r3a_excits(idet,1); e = r3a_excits(idet,2); f = r3a_excits(idet,3);
                     m = r3a_excits(idet,4); n = r3a_excits(idet,5);
                     sigma_1a(a) = sigma_1a(a) + h2a_oovv(m,n,e,f) * r_amp ! (1)
                     sigma_1a(e) = sigma_1a(e) - h2a_oovv(m,n,a,f) * r_amp ! (ae)
                     sigma_1a(f) = sigma_1a(f) - h2a_oovv(m,n,e,a) * r_amp ! (af)
                  end do

                  do idet = 1,n3aab
                     r_amp = r3b_amps(idet)
                     ! x1(a) <- A(ae) h2b(mnef)*r3b(aefmn)
                     a = r3b_excits(idet,1); e = r3b_excits(idet,2); f = r3b_excits(idet,3);
                     m = r3b_excits(idet,4); n = r3b_excits(idet,5);
                     sigma_1a(a) = sigma_1a(a) + h2b_oovv(m,n,e,f) * r_amp ! (1)
                     sigma_1a(e) = sigma_1a(e) - h2b_oovv(m,n,a,f) * r_amp ! (ae)
                  end do

                  do idet = 1,n3abb
                     r_amp = r3c_amps(idet)
                     ! x1(a) <- h2c(mnef)*r3c(aefmn)
                     a = r3c_excits(idet,1); e = r3c_excits(idet,2); f = r3c_excits(idet,3);
                     m = r3c_excits(idet,4); n = r3c_excits(idet,5);
                     sigma_1a(a) = sigma_1a(a) + h2c_oovv(m,n,e,f) * r_amp ! (1)
                  end do
              end subroutine build_hr_1a

              subroutine build_hr_2a(sigma_2a,&
                                     r3a_amps,r3a_excits,&
                                     r3b_amps,r3b_excits,&
                                     h1a_ov,h1b_ov,&
                                     h2a_ooov,h2a_vovv,h2b_ooov,h2b_vovv,&
                                     n3aaa,n3aab,&
                                     noa,nua,nob,nub)

                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa, n3aab
                  ! Input R and T arrays
                  integer, intent(in) :: r3a_excits(n3aaa,5)
                  integer, intent(in) :: r3b_excits(n3aab,5)
                  real(kind=8), intent(in) :: r3a_amps(n3aaa)
                  real(kind=8), intent(in) :: r3b_amps(n3aab)
                  ! Input H  and X arrays
                  real(kind=8), intent(in) :: h1a_ov(noa,nua)
                  real(kind=8), intent(in) :: h1b_ov(nob,nub)
                  real(kind=8), intent(in) :: h2a_ooov(noa,noa,noa,nua)
                  real(kind=8), intent(in) :: h2b_ooov(noa,nob,noa,nub)
                  real(kind=8), intent(in) :: h2a_vovv(nua,noa,nua,nua)
                  real(kind=8), intent(in) :: h2b_vovv(nua,nob,nua,nub)
                  ! Output and Inout variables
                  real(kind=8), intent(inout) :: sigma_2a(nua,nua,noa)
                  !f2py intent(in,out) :: sigma_2a(0:nua-1,0:nua-1,0:noa-1)
                  ! Local variables
                  real(kind=8) :: r_amp, val
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  do idet = 1,n3aaa
                     r_amp = r3a_amps(idet)
                     ! A(ab) [A(e/ab)A(jm) h1a(me)*r3a(abejm)]
                     a = r3a_excits(idet,1); b = r3a_excits(idet,2); e = r3a_excits(idet,3);
                     j = r3a_excits(idet,4); m = r3a_excits(idet,5);
                     sigma_2a(a,b,j) = sigma_2a(a,b,j) + h1a_ov(m,e) * r_amp ! (1)
                     sigma_2a(e,b,j) = sigma_2a(e,b,j) - h1a_ov(m,a) * r_amp ! (ae)
                     sigma_2a(a,e,j) = sigma_2a(a,e,j) - h1a_ov(m,b) * r_amp ! (be)
                     sigma_2a(a,b,m) = sigma_2a(a,b,m) - h1a_ov(j,e) * r_amp ! (jm)
                     sigma_2a(e,b,m) = sigma_2a(e,b,m) + h1a_ov(j,a) * r_amp ! (ae)(jm)
                     sigma_2a(a,e,m) = sigma_2a(a,e,m) + h1a_ov(j,b) * r_amp ! (be)(jm)
                     ! A(ab) [A(f/ab) -h2a(mnjf)*r3a(abfmn)]
                     a = r3a_excits(idet,1); b = r3a_excits(idet,2); f = r3a_excits(idet,3);
                     m = r3a_excits(idet,4); n = r3a_excits(idet,5);
                     sigma_2a(a,b,:) = sigma_2a(a,b,:) - h2a_ooov(m,n,:,f) * r_amp ! (1)
                     sigma_2a(b,f,:) = sigma_2a(b,f,:) - h2a_ooov(m,n,:,a) * r_amp ! (af), I flipped f <-> b here to update unique element
                     sigma_2a(a,f,:) = sigma_2a(a,f,:) + h2a_ooov(m,n,:,b) * r_amp ! (bf)
                     ! A(ab) [A(a/ef)A(jn) h2a(bnef)*r3a(aefjn)]
                     a = r3a_excits(idet,1); e = r3a_excits(idet,2); f = r3a_excits(idet,3);
                     j = r3a_excits(idet,4); n = r3a_excits(idet,5);
                     sigma_2a(a,:,j) = sigma_2a(a,:,j) + h2a_vovv(:,n,e,f) * r_amp ! (1)
                     sigma_2a(e,:,j) = sigma_2a(e,:,j) - h2a_vovv(:,n,a,f) * r_amp ! (ae)
                     sigma_2a(f,:,j) = sigma_2a(f,:,j) - h2a_vovv(:,n,e,a) * r_amp ! (af)
                     sigma_2a(a,:,n) = sigma_2a(a,:,n) - h2a_vovv(:,j,e,f) * r_amp ! (jn)
                     sigma_2a(e,:,n) = sigma_2a(e,:,n) + h2a_vovv(:,j,a,f) * r_amp ! (ae)(jn)
                     sigma_2a(f,:,n) = sigma_2a(f,:,n) + h2a_vovv(:,j,e,a) * r_amp ! (af)(jn)
                  end do                 

                  do idet = 1,n3aab
                     r_amp = r3b_amps(idet)
                     ! A(ab) [h1b(me)*r3b(abejm)]
                     a = r3b_excits(idet,1); b = r3b_excits(idet,2); e = r3b_excits(idet,3);
                     j = r3b_excits(idet,4); m = r3b_excits(idet,5);
                     sigma_2a(a,b,j) = sigma_2a(a,b,j) + h1b_ov(m,e) * r_amp ! (1)
                     ! A(ab) [-h2b(mnjf)*r3b(abfmn)]
                     a = r3b_excits(idet,1); b = r3b_excits(idet,2); f = r3b_excits(idet,3);
                     m = r3b_excits(idet,4); n = r3b_excits(idet,5);
                     sigma_2a(a,b,:) = sigma_2a(a,b,:) - h2b_ooov(m,n,:,f) * r_amp ! (1)
                     ! A(ab) [A(ae) h2b(bnef)*r3b(aefjn)] 
                     a = r3b_excits(idet,1); e = r3b_excits(idet,2); f = r3b_excits(idet,3);
                     j = r3b_excits(idet,4); n = r3b_excits(idet,5);
                     sigma_2a(a,:,j) = sigma_2a(a,:,j) + h2b_vovv(:,n,e,f) * r_amp ! (1)
                     sigma_2a(e,:,j) = sigma_2a(e,:,j) - h2b_vovv(:,n,a,f) * r_amp ! (ae)
                  end do                  
                  
                  ! antisymmetrize (this replaces the x2a -= np.transpose(x2a, (...)) stuff in vector update
                  do j = 1, noa
                     do a = 1, nua
                        do b = a+1, nua
                           val = sigma_2a(a,b,j) - sigma_2a(b,a,j)
                           sigma_2a(a,b,j) =  val
                           sigma_2a(b,a,j) = -val
                        end do
                     end do
                  end do
                  ! (H(2) * T3)_C terms are vectorized and generally broadcast to diagonal elements, which should
                  ! be 0. Set them to 0 manually (you need to do this).
                  do a = 1,nua
                     sigma_2a(a,a,:) = 0.0d0
                  end do
              end subroutine build_hr_2a

              subroutine build_hr_2b(sigma_2b,&
                                     r3b_amps,r3b_excits,&
                                     r3c_amps,r3c_excits,&
                                     h1a_ov,h1b_ov,&
                                     h2a_vovv,&
                                     h2b_oovo,h2b_vovv,h2b_ovvv,&
                                     h2c_ooov,h2c_vovv,&
                                     n3aab,n3abb,&
                                     noa,nua,nob,nub)

                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab, n3abb
                  ! Input R and T arrays
                  integer, intent(in) :: r3b_excits(n3aab,5)
                  integer, intent(in) :: r3c_excits(n3abb,5)
                  real(kind=8), intent(in) :: r3b_amps(n3aab)
                  real(kind=8), intent(in) :: r3c_amps(n3abb)
                  ! Input H  and X arrays
                  real(kind=8), intent(in) :: h1a_ov(noa,nua)
                  real(kind=8), intent(in) :: h1b_ov(nob,nub)
                  real(kind=8), intent(in) :: h2b_oovo(noa,nob,nua,nob)
                  real(kind=8), intent(in) :: h2a_vovv(nua,noa,nua,nua)
                  real(kind=8), intent(in) :: h2b_vovv(nua,nob,nua,nub)
                  real(kind=8), intent(in) :: h2b_ovvv(noa,nub,nua,nub)
                  real(kind=8), intent(in) :: h2c_ooov(nob,nob,nob,nub)
                  real(kind=8), intent(in) :: h2c_vovv(nub,nob,nub,nub)
                  ! Output and Inout variables
                  real(kind=8), intent(inout) :: sigma_2b(nua,nub,nob)
                  !f2py intent(in,out) :: sigma_2b(0:nua-1,0:nub-1,0:nob-1)
                  ! Local variables
                  real(kind=8) :: r_amp, val
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  do idet = 1,n3aab
                     r_amp = r3b_amps(idet)
                     ! A(ae) h1a(me)*r3b(aebmj)
                     a = r3b_excits(idet,1); e = r3b_excits(idet,2); b = r3b_excits(idet,3);
                     m = r3b_excits(idet,4); j = r3b_excits(idet,5);
                     sigma_2b(a,b,j) = sigma_2b(a,b,j) + h1a_ov(m,e) * r_amp ! (1)
                     sigma_2b(e,b,j) = sigma_2b(e,b,j) - h1a_ov(m,a) * r_amp ! (ae)
                     ! A(af) -h2b(nmfj)*r3b(afbnm)
                     a = r3b_excits(idet,1); f = r3b_excits(idet,2); b = r3b_excits(idet,3);
                     n = r3b_excits(idet,4); m = r3b_excits(idet,5);
                     sigma_2b(a,b,:) = sigma_2b(a,b,:) - h2b_oovo(n,m,f,:) * r_amp ! (1)
                     sigma_2b(f,b,:) = sigma_2b(f,b,:) + h2b_oovo(n,m,a,:) * r_amp ! (af)
                     ! A(af) h2b(nbfe)*r3b(afenj)
                     a = r3b_excits(idet,1); f = r3b_excits(idet,2); e = r3b_excits(idet,3);
                     n = r3b_excits(idet,4); j = r3b_excits(idet,5);
                     sigma_2b(a,:,j) = sigma_2b(a,:,j) + h2b_ovvv(n,:,f,e) * r_amp ! (1)
                     sigma_2b(f,:,j) = sigma_2b(f,:,j) - h2b_ovvv(n,:,a,e) * r_amp ! (af)
                     ! h2a(anef)*r3b(efbnj)
                     e = r3b_excits(idet,1); f = r3b_excits(idet,2); b = r3b_excits(idet,3);
                     n = r3b_excits(idet,4); j = r3b_excits(idet,5);
                     sigma_2b(:,b,j) = sigma_2b(:,b,j) + h2a_vovv(:,n,e,f) * r_amp ! (1)
                  end do                 

                  do idet = 1,n3abb
                     r_amp = r3c_amps(idet)
                     ! A(be)A(jm) h1b(me)*r3c(aebmj)
                     a = r3c_excits(idet,1); e = r3c_excits(idet,2); b = r3c_excits(idet,3);
                     m = r3c_excits(idet,4); j = r3c_excits(idet,5);
                     sigma_2b(a,b,j) = sigma_2b(a,b,j) + h1b_ov(m,e) * r_amp ! (1)
                     sigma_2b(a,e,j) = sigma_2b(a,e,j) - h1b_ov(m,b) * r_amp ! (be)
                     sigma_2b(a,b,m) = sigma_2b(a,b,m) - h1b_ov(j,e) * r_amp ! (jm)
                     sigma_2b(a,e,m) = sigma_2b(a,e,m) + h1b_ov(j,b) * r_amp ! (be)(jm)
                     ! A(bf) -h2c(mnjf)*r3c(abfmn)
                     a = r3c_excits(idet,1); b = r3c_excits(idet,2); f = r3c_excits(idet,3);
                     m = r3c_excits(idet,4); n = r3c_excits(idet,5);
                     sigma_2b(a,b,:) = sigma_2b(a,b,:) - h2c_ooov(m,n,:,f) * r_amp ! (1)
                     sigma_2b(a,f,:) = sigma_2b(a,f,:) + h2c_ooov(m,n,:,b) * r_amp ! (bf)
                     ! A(jn) h2c(bnef)*r3c(aefjn)
                     a = r3c_excits(idet,1); e = r3c_excits(idet,2); f = r3c_excits(idet,3);
                     j = r3c_excits(idet,4); n = r3c_excits(idet,5);
                     sigma_2b(a,:,j) = sigma_2b(a,:,j) + h2c_vovv(:,n,e,f) * r_amp ! (1)
                     sigma_2b(a,:,n) = sigma_2b(a,:,n) - h2c_vovv(:,j,e,f) * r_amp ! (jn)
                     ! A(bf)A(jn) h2b(anef)*r3c(ebfjn)
                     e = r3c_excits(idet,1); b = r3c_excits(idet,2); f = r3c_excits(idet,3);
                     j = r3c_excits(idet,4); n = r3c_excits(idet,5);
                     sigma_2b(:,b,j) = sigma_2b(:,b,j) + h2b_vovv(:,n,e,f) * r_amp ! (1)
                     sigma_2b(:,f,j) = sigma_2b(:,f,j) - h2b_vovv(:,n,e,b) * r_amp ! (bf)
                     sigma_2b(:,b,n) = sigma_2b(:,b,n) - h2b_vovv(:,j,e,f) * r_amp ! (jn)
                     sigma_2b(:,f,n) = sigma_2b(:,f,n) + h2b_vovv(:,j,e,b) * r_amp ! (bf)(jn)
                  end do                  
              end subroutine build_hr_2b

              subroutine build_hr_3a(resid,&
                                     r2a,&
                                     r3a_amps,r3a_excits,&
                                     r3b_amps,r3b_excits,&
                                     t2a,&
                                     h1a_oo,h1a_vv,&
                                     h2a_vvvv,h2a_oooo,h2a_voov,h2a_vooo,h2a_vvov,&
                                     h2b_voov,&
                                     x2a_voo,x2a_vvv,&
                                     n3aaa,n3aab,&
                                     noa,nua,nob,nub)

                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa, n3aab
                  ! Input R and T arrays
                  real(kind=8), intent(in) :: r2a(nua,nua,noa), t2a(nua,nua,noa,noa)
                  integer, intent(in) :: r3b_excits(n3aab,5)
                  real(kind=8), intent(in) :: r3b_amps(n3aab)
                  ! Input H  and X arrays
                  real(kind=8), intent(in) :: h1a_oo(noa,noa)
                  real(kind=8), intent(in) :: h1a_vv(nua,nua)
                  real(kind=8), intent(in) :: h2a_vvvv(nua,nua,nua,nua)
                  real(kind=8), intent(in) :: h2a_oooo(noa,noa,noa,noa)
                  real(kind=8), intent(in) :: h2a_voov(nua,noa,noa,nua)
                  real(kind=8), intent(in) :: h2a_vooo(nua,noa,noa,noa)
                  real(kind=8), intent(in) :: h2a_vvov(nua,nua,noa,nua)
                  real(kind=8), intent(in) :: h2b_voov(nua,nob,noa,nub)
                  real(kind=8), intent(in) :: x2a_voo(nua,noa,noa)
                  real(kind=8), intent(in) :: x2a_vvv(nua,nua,nua)
                  ! Output and Inout variables
                  real(kind=8), intent(out) :: resid(n3aaa)
                  integer, intent(inout) :: r3a_excits(n3aaa,5)
                  !f2py intent(in,out) :: r3a_excits(0:n3aaa-1,0:4)
                  real(kind=8), intent(inout) :: r3a_amps(n3aaa)
                  !f2py intent(in,out) :: r3a_amps(0:n3aaa-1)
                  ! Local variables
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:)
                  integer, allocatable :: idx_table4(:,:,:,:)
                  integer, allocatable :: idx_table3(:,:,:)
                  integer, allocatable :: loc_arr(:,:)
                  real(kind=8) :: r_amp, hmatel, res_mm23
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc
                  
                  ! Zero the container that holds H*R
                  resid = 0.0d0

                  !!!! diagram 1: -A(jk) h1a(mj)*r3a(abcmk)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)*(nua-2)/6 * noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nua,nua,noa))
                  !!! SB: (1,2,3,4) !!!
                  call get_index_table4(idx_table4, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/1,noa-1/), nua, nua, nua, noa)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table4, (/1,2,3,4/), nua, nua, nua, noa, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_oo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     a = r3a_excits(idet,1); b = r3a_excits(idet,2); c = r3a_excits(idet,3);
                     j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(a,b,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3a_excits(jdet,5);
                        ! compute < abcjk | h1a(oo) | abcjm >
                        hmatel = -h1a_oo(m,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table4(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           m = r3a_excits(jdet,5);
                           ! compute < abcjk | h1a(oo) | abckm >
                           hmatel = h1a_oo(m,j)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,3,5) !!!
                  call get_index_table4(idx_table4, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/2,noa/), nua, nua, nua, noa)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table4, (/1,2,3,5/), nua, nua, nua, noa, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_oo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     a = r3a_excits(idet,1); b = r3a_excits(idet,2); c = r3a_excits(idet,3);
                     j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(a,b,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3a_excits(jdet,4);
                        ! compute < abcjk | h1a(oo) | abcmk >
                        hmatel = -h1a_oo(m,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table4(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           m = r3a_excits(jdet,4);
                           ! compute < abcjk | h1a(oo) | abcmj >
                           hmatel = h1a_oo(m,k)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 2: A(b/ac) h1a(be)*r3a(aecjk)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*(nua - 2)/2*noa*(noa - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nua,noa,noa))
                  !!! SB: (1,3,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nua-2/), (/-2,nua/), (/1,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table4, (/1,3,4,5/), nua, nua, noa, noa, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_vv,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     a = r3a_excits(idet,1); b = r3a_excits(idet,2); c = r3a_excits(idet,3);
                     j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(a,c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3a_excits(jdet,2);
                        ! compute < abcjk | h1a(vv) | aecjk >
                        hmatel = h1a_vv(b,e)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits(jdet,2);
                           ! compute < abcjk | h1a(vv) | becjk >
                           hmatel = -h1a_vv(a,e)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table4(a,b,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits(jdet,2);
                           ! compute < abcjk | h1a(vv) | aebjk >
                           hmatel = -h1a_vv(c,e)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nua-2/), (/-1,nua-1/), (/1,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table4, (/1,2,4,5/), nua, nua, noa, noa, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_vv,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     a = r3a_excits(idet,1); b = r3a_excits(idet,2); c = r3a_excits(idet,3);
                     j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(a,b,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3a_excits(jdet,3);
                        ! compute < abcjk | h1a(vv) | abfjk >
                        hmatel = h1a_vv(c,f)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = r3a_excits(jdet,3);
                           ! compute < abcjk | h1a(vv) | bcfjk >
                           hmatel = h1a_vv(a,f)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table4(a,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = r3a_excits(jdet,3);
                           ! compute < abcjk | h1a(vv) | acfjk >
                           hmatel = -h1a_vv(b,f)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4,5) !!!
                  call get_index_table4(idx_table4, (/2,nua-1/), (/-1,nua/), (/1,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table4, (/2,3,4,5/), nua, nua, noa, noa, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_vv,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     a = r3a_excits(idet,1); b = r3a_excits(idet,2); c = r3a_excits(idet,3);
                     j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3a_excits(jdet,1);
                        ! compute < abcjk | h1a(vv) | dbcjk >
                        hmatel = h1a_vv(a,d)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table4(a,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits(jdet,1);
                           ! compute < abcjk | h1a(vv) | dacjk >
                           hmatel = -h1a_vv(b,d)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (ac)
                     idx = idx_table4(a,b,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits(jdet,1);
                           ! compute < abcjk | h1a(vv) | dabjk >
                           hmatel = h1a_vv(c,d)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 3: 1/2 A(c/ab) h2a(abef)*r3a(efcjk)
                  ! allocate new sorting arrays
                  nloc = (nua - 2)*noa*(noa - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,noa,noa))
                  !!! SB: (3,4,5) !!!
                  call get_index_table3(idx_table3, (/3,nua/), (/1,noa-1/), (/-1,noa/), nua, noa, noa)
                  call sort3(r3a_excits, r3a_amps, loc_arr, idx_table3, (/3,4,5/), nua, noa, noa, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_vvvv,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     a = r3a_excits(idet,1); b = r3a_excits(idet,2); c = r3a_excits(idet,3);
                     j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3a_excits(jdet,1); e = r3a_excits(jdet,2);
                        ! compute < abcjk | h2a(vvvv) | decjk >
                        hmatel = h2a_vvvv(a,b,d,e)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table3(a,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits(jdet,1); e = r3a_excits(jdet,2);
                           ! compute < abcjk | h2a(vvvv) | deajk >
                           hmatel = h2a_vvvv(b,c,d,e)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table3(b,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits(jdet,1); e = r3a_excits(jdet,2);
                           ! compute < abcjk | h2a(vvvv) | debjk >
                           hmatel = -h2a_vvvv(a,c,d,e)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,4,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/1,noa-1/), (/-1,noa/), nua, noa, noa)
                  call sort3(r3a_excits, r3a_amps, loc_arr, idx_table3, (/1,4,5/), nua, noa, noa, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_vvvv,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     a = r3a_excits(idet,1); b = r3a_excits(idet,2); c = r3a_excits(idet,3);
                     j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3a_excits(jdet,2); f = r3a_excits(jdet,3);
                        ! compute < abcjk | h2a(vvvv) | aefjk >
                        hmatel = h2a_vvvv(b,c,e,f)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(b,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits(jdet,2); f = r3a_excits(jdet,3);
                           ! compute < abcjk | h2a(vvvv) | befjk >
                           hmatel = -h2a_vvvv(a,c,e,f)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (ac)
                     idx = idx_table3(c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits(jdet,2); f = r3a_excits(jdet,3);
                           ! compute < abcjk | h2a(vvvv) | cefjk >
                           hmatel = h2a_vvvv(a,b,e,f)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,4,5)
                  call get_index_table3(idx_table3, (/2,nua-1/), (/1,noa-1/), (/-1,noa/), nua, noa, noa)
                  call sort3(r3a_excits, r3a_amps, loc_arr, idx_table3, (/2,4,5/), nua, noa, noa, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_vvvv,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     a = r3a_excits(idet,1); b = r3a_excits(idet,2); c = r3a_excits(idet,3);
                     j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3a_excits(jdet,1); f = r3a_excits(jdet,3);
                        ! compute < abcjk | h2a(vvvv) | dbfjk >
                        hmatel = h2a_vvvv(a,c,d,f)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(a,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits(jdet,1); f = r3a_excits(jdet,3);
                           ! compute < abcjk | h2a(vvvv) | dafjk >
                           hmatel = -h2a_vvvv(b,c,d,f)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table3(c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits(jdet,1); f = r3a_excits(jdet,3);
                           ! compute < abcjk | h2a(vvvv) | dcfjk >
                           hmatel = -h2a_vvvv(a,b,d,f)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 4: 1/4 h2a(mnjk)*r3a(abcmn)
                  ! allocate new sorting arrays
                  nloc = (nua - 2)*(nua - 1)*nua/6
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,nua))
                  !!! SB: (1,2,3) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), nua, nua, nua)
                  call sort3(r3a_excits, r3a_amps, loc_arr, idx_table3, (/1,2,3/), nua, nua, nua, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_oooo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     a = r3a_excits(idet,1); b = r3a_excits(idet,2); c = r3a_excits(idet,3);
                     j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,b,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3a_excits(jdet,4); n = r3a_excits(jdet,5);
                        ! compute < abcjk | h2a(oooo) | abcmn >
                        hmatel = h2a_oooo(m,n,j,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 5: A(c/ab)A(jk) h2a(cmke)*r3a(abejm)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*(nua - 2)/2*(noa - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,noa))
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-1,nua-1/), (/1,noa-1/), nua, nua, noa)
                  call sort3(r3a_excits, r3a_amps, loc_arr, idx_table3, (/1,2,4/), nua, nua, noa, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     a = r3a_excits(idet,1); b = r3a_excits(idet,2); c = r3a_excits(idet,3);
                     j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,b,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3a_excits(jdet,3); n = r3a_excits(jdet,5);
                        ! compute < abcjk | h2a(voov) | abfjn >
                        hmatel = h2a_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ac), (-1)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = r3a_excits(jdet,3); n = r3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | bcfjn >
                           hmatel = h2a_voov(a,n,k,f)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = r3a_excits(jdet,3); n = r3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | acfjn >
                           hmatel = -h2a_voov(b,n,k,f)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = r3a_excits(jdet,3); n = r3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | abfkn >
                           hmatel = -h2a_voov(c,n,j,f)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (ac)(jk), (-1)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = r3a_excits(jdet,3); n = r3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | bcfkn >
                           hmatel = -h2a_voov(a,n,j,f)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = r3a_excits(jdet,3); n = r3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | acfkn >
                           hmatel = h2a_voov(b,n,j,f)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-2,nua/), (/1,noa-1/), nua, nua, noa)
                  call sort3(r3a_excits, r3a_amps, loc_arr, idx_table3, (/1,3,4/), nua, nua, noa, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     a = r3a_excits(idet,1); b = r3a_excits(idet,2); c = r3a_excits(idet,3);
                     j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3a_excits(jdet,2); n = r3a_excits(jdet,5);
                        ! compute < abcjk | h2a(voov) | aecjn >
                        hmatel = h2a_voov(b,n,k,e)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits(jdet,2); n = r3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | becjn >
                           hmatel = -h2a_voov(a,n,k,e)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits(jdet,2); n = r3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | aebjn >
                           hmatel = -h2a_voov(c,n,k,e)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits(jdet,2); n = r3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | aeckn >
                           hmatel = -h2a_voov(b,n,j,e)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits(jdet,2); n = r3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | beckn >
                           hmatel = h2a_voov(a,n,j,e)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits(jdet,2); n = r3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | aebkn >
                           hmatel = h2a_voov(c,n,j,e)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4) !!!
                  call get_index_table3(idx_table3, (/2,nua-1/), (/-1,nua/), (/1,noa-1/), nua, nua, noa)
                  call sort3(r3a_excits, r3a_amps, loc_arr, idx_table3, (/2,3,4/), nua, nua, noa, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     a = r3a_excits(idet,1); b = r3a_excits(idet,2); c = r3a_excits(idet,3);
                     j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3a_excits(jdet,1); n = r3a_excits(jdet,5);
                        ! compute < abcjk | h2a(voov) | dbcjn >
                        hmatel = h2a_voov(a,n,k,d)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits(jdet,1); n = r3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | dacjn >
                           hmatel = -h2a_voov(b,n,k,d)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if 
                     ! (ac), (-1)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits(jdet,1); n = r3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | dabjn >
                           hmatel = h2a_voov(c,n,k,d)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if 
                     ! (jk)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits(jdet,1); n = r3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | dbckn >
                           hmatel = -h2a_voov(a,n,j,d)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if 
                     ! (ab)(jk)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits(jdet,1); n = r3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | dackn >
                           hmatel = h2a_voov(b,n,j,d)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if 
                     ! (ac)(jk), (-1)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits(jdet,1); n = r3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | dabkn >
                           hmatel = -h2a_voov(c,n,j,d)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if 
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-1,nua-1/), (/2,noa/), nua, nua, noa)
                  call sort3(r3a_excits, r3a_amps, loc_arr, idx_table3, (/1,2,5/), nua, nua, noa, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     a = r3a_excits(idet,1); b = r3a_excits(idet,2); c = r3a_excits(idet,3);
                     j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,b,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3a_excits(jdet,3); m = r3a_excits(jdet,4);
                        ! compute < abcjk | h2a(voov) | abfmk >
                        hmatel = h2a_voov(c,m,j,f)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ac), (-1)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = r3a_excits(jdet,3); m = r3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | bcfmk >
                           hmatel = h2a_voov(a,m,j,f)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = r3a_excits(jdet,3); m = r3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | acfmk >
                           hmatel = -h2a_voov(b,m,j,f)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = r3a_excits(jdet,3); m = r3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | abfmj >
                           hmatel = -h2a_voov(c,m,k,f)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (ac)(jk), (-1)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = r3a_excits(jdet,3); m = r3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | bcfmj >
                           hmatel = -h2a_voov(a,m,k,f)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = r3a_excits(jdet,3); m = r3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | acfmj >
                           hmatel = h2a_voov(b,m,k,f)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-2,nua/), (/2,noa/), nua, nua, noa)
                  call sort3(r3a_excits, r3a_amps, loc_arr, idx_table3, (/1,3,5/), nua, nua, noa, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     a = r3a_excits(idet,1); b = r3a_excits(idet,2); c = r3a_excits(idet,3);
                     j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3a_excits(jdet,2); m = r3a_excits(jdet,4);
                        ! compute < abcjk | h2a(voov) | aecmk >
                        hmatel = h2a_voov(b,m,j,e)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits(jdet,2); m = r3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | becmk >
                           hmatel = -h2a_voov(a,m,j,e)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits(jdet,2); m = r3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | aebmk >
                           hmatel = -h2a_voov(c,m,j,e)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits(jdet,2); m = r3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | aecmj >
                           hmatel = -h2a_voov(b,m,k,e)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits(jdet,2); m = r3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | becmj >
                           hmatel = h2a_voov(a,m,k,e)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits(jdet,2); m = r3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | aebmj >
                           hmatel = h2a_voov(c,m,k,e)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,5) !!!
                  call get_index_table3(idx_table3, (/2,nua-1/), (/-1,nua/), (/2,noa/), nua, nua, noa)
                  call sort3(r3a_excits, r3a_amps, loc_arr, idx_table3, (/2,3,5/), nua, nua, noa, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     a = r3a_excits(idet,1); b = r3a_excits(idet,2); c = r3a_excits(idet,3);
                     j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3a_excits(jdet,1); m = r3a_excits(jdet,4);
                        ! compute < abcjk | h2a(voov) | dbcmk >
                        hmatel = h2a_voov(a,m,j,d)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits(jdet,1); m = r3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | dacmk >
                           hmatel = -h2a_voov(b,m,j,d)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (ac), (-1)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits(jdet,1); m = r3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | dabmk >
                           hmatel = h2a_voov(c,m,j,d)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits(jdet,1); m = r3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | dbcmj >
                           hmatel = -h2a_voov(a,m,k,d)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits(jdet,1); m = r3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | dacmj >
                           hmatel = h2a_voov(b,m,k,d)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (ac)(jk), (-1)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits(jdet,1); m = r3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | dabmj >
                           hmatel = -h2a_voov(c,m,k,d)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 6: A(c/ab)A(jk) h2b(cmke)*r3b(abejm)
                  ! allocate and copy over r3b arrays
                  allocate(amps_buff(n3aab),excits_buff(n3aab,5))
                  amps_buff(:) = r3b_amps(:)
                  excits_buff(:,:) = r3b_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = nua*(nua - 1)/2*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,noa))
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/-1,nua/), (/1,noa/), nua, nua, noa)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,2,4/), nua, nua, noa, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     a = r3a_excits(idet,1); b = r3a_excits(idet,2); c = r3a_excits(idet,3);
                     j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,b,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                        ! compute < abcjk | h2a(voov) | abfjn >
                        hmatel = h2b_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ac), (-1)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                           ! compute < abcjk | h2a(voov) | bcfjn >
                           hmatel = h2b_voov(a,n,k,f)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                           ! compute < abcjk | h2a(voov) | acfjn >
                           hmatel = -h2b_voov(b,n,k,f)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                           ! compute < abcjk | h2a(voov) | abfkn >
                           hmatel = -h2b_voov(c,n,j,f)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ac)(jk), (-1)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                           ! compute < abcjk | h2a(voov) | bcfkn >
                           hmatel = -h2b_voov(a,n,j,f)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                           ! compute < abcjk | h2a(voov) | acfkn >
                           hmatel = h2b_voov(b,n,j,f)
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3,excits_buff,amps_buff)

                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp t2a,r2a,&
                  !$omp h2a_vvov,h2a_vooo,&
                  !$omp x2a_voo,x2a_vvv,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(idet,a,b,c,d,i,j,k,l,m,n,e,f,res_mm23)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                      a = r3a_excits(idet,1); b = r3a_excits(idet,2); c = r3a_excits(idet,3);
                      j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                      ! zero out value
                      res_mm23 = 0.0d0
                      do m = 1,noa
                         ! -A(c/ab) h2a(cmkj)*r2a(abm)
                         res_mm23 = res_mm23 - h2a_vooo(c,m,k,j) * r2a(a,b,m) ! (1)
                         res_mm23 = res_mm23 + h2a_vooo(a,m,k,j) * r2a(c,b,m) ! (ac)
                         res_mm23 = res_mm23 + h2a_vooo(b,m,k,j) * r2a(a,c,m) ! (bc)
                         ! -A(a/bc)A(jk) x2a_voo(amj)*t2a(bcmk)
                         res_mm23 = res_mm23 - x2a_voo(a,m,j) * t2a(b,c,m,k) ! (1)
                         res_mm23 = res_mm23 + x2a_voo(b,m,j) * t2a(a,c,m,k) ! (ab)
                         res_mm23 = res_mm23 + x2a_voo(c,m,j) * t2a(b,a,m,k) ! (ac)
                         res_mm23 = res_mm23 + x2a_voo(a,m,k) * t2a(b,c,m,j) ! (jk)
                         res_mm23 = res_mm23 - x2a_voo(b,m,k) * t2a(a,c,m,j) ! (ab)(jk)
                         res_mm23 = res_mm23 - x2a_voo(c,m,k) * t2a(b,a,m,j) ! (ac)(jk)
                      end do
                      do e = 1,nua
                         ! A(a/bc)A(jk) h2a(cbke)*r2a(aej)
                         res_mm23 = res_mm23 + h2a_vvov(c,b,k,e) * r2a(a,e,j) ! (1)
                         res_mm23 = res_mm23 - h2a_vvov(c,a,k,e) * r2a(b,e,j) ! (ab)
                         res_mm23 = res_mm23 - h2a_vvov(a,b,k,e) * r2a(c,e,j) ! (ac)
                         res_mm23 = res_mm23 - h2a_vvov(c,b,j,e) * r2a(a,e,k) ! (jk)
                         res_mm23 = res_mm23 + h2a_vvov(c,a,j,e) * r2a(b,e,k) ! (ab)(jk)
                         res_mm23 = res_mm23 + h2a_vvov(a,b,j,e) * r2a(c,e,k) ! (ac)(jk)
                         ! A(c/ab) x2a_vvv(abe)*t2a(ecjk)
                         res_mm23 = res_mm23 + x2a_vvv(a,b,e) * t2a(e,c,j,k) ! (1)
                         res_mm23 = res_mm23 - x2a_vvv(c,b,e) * t2a(e,a,j,k) ! (ac)
                         res_mm23 = res_mm23 - x2a_vvv(a,c,e) * t2a(e,b,j,k) ! (bc)
                      end do

                      resid(idet) = resid(idet) + res_mm23
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
              end subroutine build_hr_3a

              subroutine build_hr_3b(resid,&
                                     r2a,r2b,&
                                     r3a_amps,r3a_excits,&
                                     r3b_amps,r3b_excits,&
                                     r3c_amps,r3c_excits,&
                                     t2a,t2b,&
                                     h1a_oo,h1a_vv,h1b_oo,h1b_vv,&
                                     h2a_vvvv,h2a_voov,h2a_vvov,&
                                     h2b_vvvv,h2b_oooo,h2b_voov,h2b_vovo,h2b_ovov,h2b_ovvo,&
                                     h2b_vooo,h2b_ovoo,h2b_vvov,h2b_vvvo,&
                                     h2c_voov,&
                                     x2a_voo,x2a_vvv,&
                                     x2b_voo,x2b_ovo,x2b_vvv,&
                                     n3aaa,n3aab,n3abb,&
                                     noa,nua,nob,nub)

                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa, n3aab, n3abb
                  ! Input R and T arrays
                  real(kind=8), intent(in) :: r2a(nua,nua,noa), t2a(nua,nua,noa,noa)
                  real(kind=8), intent(in) :: r2b(nua,nub,nob), t2b(nua,nub,noa,nob)
                  integer, intent(in) :: r3a_excits(n3aaa,5), r3c_excits(n3abb,5)
                  real(kind=8), intent(in) :: r3a_amps(n3aaa), r3c_amps(n3abb)
                  ! Input H  and X arrays
                  real(kind=8), intent(in) :: h1a_oo(noa,noa)
                  real(kind=8), intent(in) :: h1a_vv(nua,nua)
                  real(kind=8), intent(in) :: h1b_oo(nob,nob)
                  real(kind=8), intent(in) :: h1b_vv(nub,nub)
                  real(kind=8), intent(in) :: h2a_vvvv(nua,nua,nua,nua)
                  real(kind=8), intent(in) :: h2a_voov(nua,noa,noa,nua)
                  real(kind=8), intent(in) :: h2a_vvov(nua,nua,noa,nua)
                  real(kind=8), intent(in) :: h2b_vvvv(nua,nub,nua,nub)
                  real(kind=8), intent(in) :: h2b_oooo(noa,nob,noa,nob)
                  real(kind=8), intent(in) :: h2b_voov(nua,nob,noa,nub)
                  real(kind=8), intent(in) :: h2b_vovo(nua,nob,nua,nob)
                  real(kind=8), intent(in) :: h2b_ovov(noa,nub,noa,nub)
                  real(kind=8), intent(in) :: h2b_ovvo(noa,nub,nua,nob)
                  real(kind=8), intent(in) :: h2b_vooo(nua,nob,noa,nob)
                  real(kind=8), intent(in) :: h2b_ovoo(noa,nub,noa,nob)
                  real(kind=8), intent(in) :: h2b_vvov(nua,nub,noa,nub)
                  real(kind=8), intent(in) :: h2b_vvvo(nua,nub,nua,nob)
                  real(kind=8), intent(in) :: h2c_voov(nub,nob,nob,nub)
                  real(kind=8), intent(in) :: x2a_voo(nua,noa,noa)
                  real(kind=8), intent(in) :: x2a_vvv(nua,nua,nua)
                  real(kind=8), intent(in) :: x2b_voo(nua,nob,nob)
                  real(kind=8), intent(in) :: x2b_ovo(noa,nub,nob)
                  real(kind=8), intent(in) :: x2b_vvv(nua,nub,nub)
                  ! Output and Inout variables
                  real(kind=8), intent(out) :: resid(n3aab)
                  integer, intent(inout) :: r3b_excits(n3aab,5)
                  !f2py intent(in,out) :: r3b_excits(0:n3aab-1,0:4)
                  real(kind=8), intent(inout) :: r3b_amps(n3aab)
                  !f2py intent(in,out) :: r3b_amps(0:n3aab-1)
                  ! Local variables
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:)
                  integer, allocatable :: idx_table4(:,:,:,:)
                  integer, allocatable :: idx_table3(:,:,:)
                  integer, allocatable :: loc_arr(:,:)
                  real(kind=8) :: r_amp, hmatel, res_mm23
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc
                  
                  ! Zero the container that holds H*R
                  resid = 0.0d0

                  !!! diagram 1: -h1a(mj)*r3b(abcmk)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2*nub * nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nua,nub,nob))
                  !!! SB: (1,2,3,5) !!!
                  call get_index_table4(idx_table4, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/1,nob/), nua, nua, nub, nob)
                  call sort4(r3b_excits, r3b_amps, loc_arr, idx_table4, (/1,2,3,5/), nua, nua, nub, nob, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_oo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = r3b_excits(idet,1); b = r3b_excits(idet,2); c = r3b_excits(idet,3);
                     j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(a,b,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3b_excits(jdet,4);
                        ! compute < abcjk | h1a(oo) | abcmk >
                        hmatel = -h1a_oo(m,j)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 2: -h1b(mk)*r3b(abcjm)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2*nub * noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nua,nub,noa))
                  !!! SB: (1,2,3,4) !!!
                  call get_index_table4(idx_table4, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/1,noa/), nua, nua, nub, noa)
                  call sort4(r3b_excits, r3b_amps, loc_arr, idx_table4, (/1,2,3,4/), nua, nua, nub, noa, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1B_oo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = r3b_excits(idet,1); b = r3b_excits(idet,2); c = r3b_excits(idet,3);
                     j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(a,b,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        n = r3b_excits(jdet,5);
                        ! compute < abcjk | h1b(oo) | abcjn >
                        hmatel = -h1b_oo(n,k)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 3: h1b(ce)*r3b(abejk)
                  ! allocate new sorting arrays
                  nloc = nua*(nua - 1)/2 * noa * nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nua,noa,nob))
                  !!! SB: (1,2,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nua-1/), (/-1,nua/), (/1,noa/), (/1,nob/), nua, nua, noa, nob)
                  call sort4(r3b_excits, r3b_amps, loc_arr, idx_table4, (/1,2,4,5/), nua, nua, noa, nob, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1B_vv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = r3b_excits(idet,1); b = r3b_excits(idet,2); c = r3b_excits(idet,3);
                     j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(a,b,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3b_excits(jdet,3);
                        ! compute < abcjk | h1b(vv) | abfjk >
                        hmatel = h1b_vv(c,f)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 4: h1a(be)*r3b(aecjk)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*nub*noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nub,noa,nob))
                  !!! SB: (1,3,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nua-1/), (/1,nub/), (/1,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(r3b_excits, r3b_amps, loc_arr, idx_table4, (/1,3,4,5/), nua, nub, noa, nob, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_vv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = r3b_excits(idet,1); b = r3b_excits(idet,2); c = r3b_excits(idet,3);
                     j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(a,c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3b_excits(jdet,2);
                        ! compute < abcjk | h1a(vv) | aecjk >
                        hmatel = h1a_vv(b,e)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3b_excits(jdet,2);
                           ! compute < abcjk | h1a(vv) | becjk >
                           hmatel = -h1a_vv(a,e)
                           resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4,5) !!!
                  call get_index_table4(idx_table4, (/2,nua/), (/1,nub/), (/1,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(r3b_excits, r3b_amps, loc_arr, idx_table4, (/2,3,4,5/), nua, nub, noa, nob, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_vv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = r3b_excits(idet,1); b = r3b_excits(idet,2); c = r3b_excits(idet,3);
                     j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3b_excits(jdet,1);
                        ! compute < abcjk | h1a(vv) | dbcjk >
                        hmatel = h1a_vv(a,d)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table4(a,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3b_excits(jdet,1);
                           ! compute < abcjk | h1a(vv) | dacjk >
                           hmatel = -h1a_vv(b,d)
                           resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 5: h2b(mnjk)*r3b(abcmn)
                  ! allocate new sorting arrays
                  nloc = nua*(nua - 1)/2*nub
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,nub))
                  !!! SB: (1,2,3) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/-1,nua/), (/1,nub/), nua, nua, nub)
                  call sort3(r3b_excits, r3b_amps, loc_arr, idx_table3, (/1,2,3/), nua, nua, nub, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_oooo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = r3b_excits(idet,1); b = r3b_excits(idet,2); c = r3b_excits(idet,3);
                     j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,b,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3b_excits(jdet,4); n = r3b_excits(jdet,5);
                        ! compute < abcjk | h2b(oooo) | abcmn >
                        hmatel = h2b_oooo(m,n,j,k)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 6: h2b(mcek)*r3a(abejm)
                  ! allocate and copy over r3a arrays
                  allocate(amps_buff(n3aaa),excits_buff(n3aaa,5))
                  amps_buff(:) = r3a_amps(:)
                  excits_buff(:,:) = r3a_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*(nua - 2)/2*(noa - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,noa))
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-1,nua-1/), (/1,noa-1/), nua, nua, noa)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,2,4/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = r3b_excits(idet,1); b = r3b_excits(idet,2); c = r3b_excits(idet,3);
                     j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                        ! compute < abc~jk~ | h2b(ovvo) | abfjn >
                        hmatel = h2b_ovvo(n,c,f,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-1,nua-1/), (/2,noa/), nua, nua, noa)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,2,5/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = r3b_excits(idet,1); b = r3b_excits(idet,2); c = r3b_excits(idet,3);
                     j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,3); m = excits_buff(jdet,4);
                        ! compute < abc~jk~ | h2b(ovvo) | abfmj >
                        hmatel = -h2b_ovvo(m,c,f,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-2,nua/), (/1,noa-1/), nua, nua, noa)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,3,4/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = r3b_excits(idet,1); b = r3b_excits(idet,2); c = r3b_excits(idet,3);
                     j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); n = excits_buff(jdet,5);
                        ! compute < abc~jk~ | h2b(ovvo) | aebjn >
                        hmatel = -h2b_ovvo(n,c,e,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-2,nua/), (/2,noa/), nua, nua, noa)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,3,5/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = r3b_excits(idet,1); b = r3b_excits(idet,2); c = r3b_excits(idet,3);
                     j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < abc~jk~ | h2b(ovvo) | aebmj >
                        hmatel = h2b_ovvo(m,c,e,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4) !!!
                  call get_index_table3(idx_table3, (/2,nua-1/), (/-1,nua/), (/1,noa-1/), nua, nua, noa)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/2,3,4/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = r3b_excits(idet,1); b = r3b_excits(idet,2); c = r3b_excits(idet,3);
                     j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = excits_buff(jdet,1); n = excits_buff(jdet,5);
                        ! compute < abc~jk~ | h2b(ovvo) | dabjn >
                        hmatel = h2b_ovvo(n,c,d,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,5) !!!
                  call get_index_table3(idx_table3, (/2,nua-1/), (/-1,nua/), (/2,noa/), nua, nua, noa)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/2,3,5/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = r3b_excits(idet,1); b = r3b_excits(idet,2); c = r3b_excits(idet,3);
                     j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < abc~jk~ | h2b(ovvo) | dabmj >
                        hmatel = -h2b_ovvo(m,c,d,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3,excits_buff,amps_buff)

                  !!! diagram 7: h2c(cmke)*r3b(abejm)
                  ! allocate new sorting arrays
                  nloc = nua*(nua - 1)/2*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,noa))
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/-1,nua/), (/1,noa/), nua, nua, noa)
                  call sort3(r3b_excits, r3b_amps, loc_arr, idx_table3, (/1,2,4/), nua, nua, noa, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = r3b_excits(idet,1); b = r3b_excits(idet,2); c = r3b_excits(idet,3);
                     j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,b,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3b_excits(jdet,3); n = r3b_excits(jdet,5);
                        ! compute < abcjk | h2c(voov) | abfjn >
                        hmatel = h2c_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 8: h2a(bmje)*r3b(aecmk)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*nub*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nub,nob))
                  !!! SB: (1,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/1,nub/), (/1,nob/), nua, nub, nob)
                  call sort3(r3b_excits, r3b_amps, loc_arr, idx_table3, (/1,3,5/), nua, nub, nob, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = r3b_excits(idet,1); b = r3b_excits(idet,2); c = r3b_excits(idet,3);
                     j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3b_excits(jdet,2); m = r3b_excits(jdet,4);
                        ! compute < abc~jk~ | h2a(voov) | aec~mk~ >
                        hmatel = h2a_voov(b,m,j,e)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3b_excits(jdet,2); m = r3b_excits(jdet,4);
                        ! compute < abc~jk~ | h2a(voov) | bec~mk~ >
                        hmatel = -h2a_voov(a,m,j,e)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,5) !!!
                  call get_index_table3(idx_table3, (/2,nua/), (/1,nub/), (/1,nob/), nua, nub, nob)
                  call sort3(r3b_excits, r3b_amps, loc_arr, idx_table3, (/2,3,5/), nua, nub, nob, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = r3b_excits(idet,1); b = r3b_excits(idet,2); c = r3b_excits(idet,3);
                     j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3b_excits(jdet,1); m = r3b_excits(jdet,4);
                        ! compute < abc~jk~ | h2a(voov) | dbc~mk~ >
                        hmatel = h2a_voov(a,m,j,d)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3b_excits(jdet,1); m = r3b_excits(jdet,4);
                        ! compute < abc~jk~ | h2a(voov) | dac~mk~ >
                        hmatel = -h2a_voov(b,m,j,d)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 9: -A(ab) h2b(bmek)*r3b(aecjm)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*nub*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nub,noa))
                  !!! SB: (1,3,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/1,nub/), (/1,noa/), nua, nub, noa)
                  call sort3(r3b_excits, r3b_amps, loc_arr, idx_table3, (/1,3,4/), nua, nub, noa, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vovo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = r3b_excits(idet,1); b = r3b_excits(idet,2); c = r3b_excits(idet,3);
                     j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3b_excits(jdet,2); n = r3b_excits(jdet,5);
                        ! compute < abc~jk~ | h2b(vovo) | aec~jn~ >
                        hmatel = -h2b_vovo(b,n,e,k)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3b_excits(jdet,2); n = r3b_excits(jdet,5);
                        ! compute < abc~jk~ | h2b(vovo) | bec~jn~ >
                        hmatel = h2b_vovo(a,n,e,k)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4) !!!
                  call get_index_table3(idx_table3, (/2,nua/), (/1,nub/), (/1,noa/), nua, nub, noa)
                  call sort3(r3b_excits, r3b_amps, loc_arr, idx_table3, (/2,3,4/), nua, nub, noa, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vovo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = r3b_excits(idet,1); b = r3b_excits(idet,2); c = r3b_excits(idet,3);
                     j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3b_excits(jdet,1); n = r3b_excits(jdet,5);
                        ! compute < abc~jk~ | h2b(vovo) | bec~jn~ >
                        hmatel = -h2b_vovo(a,n,d,k)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3b_excits(jdet,1); n = r3b_excits(jdet,5);
                        ! compute < abc~jk~ | h2b(vovo) | aec~jn~ >
                        hmatel = h2b_vovo(b,n,d,k)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 10: -h2b(mcje)*r3b(abemk)
                  ! allocate new sorting arrays
                  nloc = nua*(nua - 1)/2 * nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,nob))
                  !!! SB: (1,2,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/-1,nua/), (/1,nob/), nua, nua, nob)
                  call sort3(r3b_excits, r3b_amps, loc_arr, idx_table3, (/1,2,5/), nua, nua, nob, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = r3b_excits(idet,1); b = r3b_excits(idet,2); c = r3b_excits(idet,3);
                     j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,b,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3b_excits(jdet,3); m = r3b_excits(jdet,4);
                        ! compute < abc~jk~ | h2b(vovo) | abf~mk~ >
                        hmatel = -h2b_ovov(m,c,j,f)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 11: A(ab) h2b(bmje)*r3c(aecmk)
                  ! allocate and copy over r3c arrays
                  allocate(amps_buff(n3abb),excits_buff(n3abb,5))
                  amps_buff(:) = r3c_amps(:)
                  excits_buff(:,:) = r3c_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = nua*(nub - 1)*(nob - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nub,nob))
                  !!! SB: (1,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/2,nub/), (/2,nob/), nua, nub, nob)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,3,5/), nua, nub, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = r3b_excits(idet,1); b = r3b_excits(idet,2); c = r3b_excits(idet,3);
                     j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < abc~jk~ | h2b(voov) | ae~c~m~k~ >
                        hmatel = h2b_voov(b,m,j,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < abc~jk~ | h2b(voov) | be~c~m~k~ >
                        hmatel = -h2b_voov(a,m,j,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,4) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/2,nub/), (/1,nob-1/), nua, nub, nob)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,3,4/), nua, nub, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = r3b_excits(idet,1); b = r3b_excits(idet,2); c = r3b_excits(idet,3);
                     j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); n = excits_buff(jdet,5);
                        ! compute < abc~jk~ | h2b(voov) | ae~c~k~n~ >
                        hmatel = -h2b_voov(b,n,j,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); n = excits_buff(jdet,5);
                        ! compute < abc~jk~ | h2b(voov) | be~c~k~n~ >
                        hmatel = h2b_voov(a,n,j,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,5) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/1,nub-1/), (/2,nob/), nua, nub, nob)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,2,5/), nua, nub, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = r3b_excits(idet,1); b = r3b_excits(idet,2); c = r3b_excits(idet,3);
                     j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,3); m = excits_buff(jdet,4);
                        ! compute < abc~jk~ | h2b(voov) | ac~f~m~k~ >
                        hmatel = -h2b_voov(b,m,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,3); m = excits_buff(jdet,4);
                        ! compute < abc~jk~ | h2b(voov) | bc~f~m~k~ >
                        hmatel = h2b_voov(a,m,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/1,nub-1/), (/1,nob-1/), nua, nub, nob)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,2,4/), nua, nub, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = r3b_excits(idet,1); b = r3b_excits(idet,2); c = r3b_excits(idet,3);
                     j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                        ! compute < abc~jk~ | h2b(voov) | ac~f~k~n~ >
                        hmatel = h2b_voov(b,n,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                        ! compute < abc~jk~ | h2b(voov) | bc~f~k~n~ >
                        hmatel = -h2b_voov(a,n,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3,excits_buff,amps_buff)

                  !!! diagram 12: 1/2 h2a(abef)*r3b(efcjk)
                  ! allocate new sorting arrays
                  nloc = nub*noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nub,noa,nob))
                  !!! SB: (3,4,5) !!!
                  call get_index_table3(idx_table3, (/1,nub/), (/1,noa/), (/1,nob/), nub, noa, nob)
                  call sort3(r3b_excits, r3b_amps, loc_arr, idx_table3, (/3,4,5/), nub, noa, nob, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_vvvv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = r3b_excits(idet,1); b = r3b_excits(idet,2); c = r3b_excits(idet,3);
                     j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3b_excits(jdet,1); e = r3b_excits(jdet,2);
                        ! compute < abc~jk~ | h2a(vvvv) | dfc~jk~ >
                        hmatel = h2a_vvvv(a,b,d,e)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 13: A(ab) h2b(bcef)*r3b(aefjk)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,noa,nob))
                  !!! SB: (1,4,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/1,noa/), (/1,nob/), nua, noa, nob)
                  call sort3(r3b_excits, r3b_amps, loc_arr, idx_table3, (/1,4,5/), nua, noa, nob, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vvvv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = r3b_excits(idet,1); b = r3b_excits(idet,2); c = r3b_excits(idet,3);
                     j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3b_excits(jdet,2); f = r3b_excits(jdet,3);
                        ! compute < abc~jk~ | h2b(vvvv) | aef~jk~ >
                        hmatel = h2b_vvvv(b,c,e,f)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3b_excits(jdet,2); f = r3b_excits(jdet,3);
                        ! compute < abc~jk~ | h2b(vvvv) | bef~jk~ >
                        hmatel = -h2b_vvvv(a,c,e,f)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,4,5) !!!
                  call get_index_table3(idx_table3, (/2,nua/), (/1,noa/), (/1,nob/), nua, noa, nob)
                  call sort3(r3b_excits, r3b_amps, loc_arr, idx_table3, (/2,4,5/), nua, noa, nob, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vvvv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = r3b_excits(idet,1); b = r3b_excits(idet,2); c = r3b_excits(idet,3);
                     j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3b_excits(jdet,1); f = r3b_excits(jdet,3);
                        ! compute < abc~jk~ | h2b(vvvv) | dbf~jk~ >
                        hmatel = h2b_vvvv(a,c,d,f)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(a,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3b_excits(jdet,1); f = r3b_excits(jdet,3);
                        ! compute < abc~jk~ | h2b(vvvv) | daf~jk~ >
                        hmatel = -h2b_vvvv(b,c,d,f)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp t2a,t2b,r2a,r2b,&
                  !$omp h2a_vvov,h2b_vooo,h2b_ovoo,h2b_vvov,h2b_vvvo,&
                  !$omp x2a_voo,x2a_vvv,x2b_voo,x2b_ovo,x2b_vvv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(idet,a,b,c,d,i,j,k,l,m,n,e,f,res_mm23)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                      a = r3b_excits(idet,1); b = r3b_excits(idet,2); c = r3b_excits(idet,3);
                      j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                      ! zero out value
                      res_mm23 = 0.0d0
                      do m = 1,noa
                         ! -h2b(mcjk)*r2a(abm)
                         res_mm23 = res_mm23 - h2b_ovoo(m,c,j,k) * r2a(a,b,m) ! (1)
                         ! -x2b_ovo(mck)*t2a(abmj)
                         res_mm23 = res_mm23 - x2b_ovo(m,c,k) * t2a(a,b,m,j) ! (1)
                         ! -A(ab) x2a_voo(amj)*t2b(bcmk)
                         res_mm23 = res_mm23 - x2a_voo(a,m,j) * t2b(b,c,m,k) ! (1)
                         res_mm23 = res_mm23 + x2a_voo(b,m,j) * t2b(a,c,m,k) ! (ab)
                      end do
                      do m = 1,nob
                         ! -A(ab) h2b(bmjk)*r2b(acm)
                         res_mm23 = res_mm23 - h2b_vooo(b,m,j,k) * r2b(a,c,m) ! (1)
                         res_mm23 = res_mm23 + h2b_vooo(a,m,j,k) * r2b(b,c,m) ! (ab)
                         ! -A(ab) x2b_voo(amk)*t2b(bcjm)
                         res_mm23 = res_mm23 - x2b_voo(a,m,k) * t2b(b,c,j,m) ! (1)
                         res_mm23 = res_mm23 + x2b_voo(b,m,k) * t2b(a,c,j,m) ! (ab)
                      end do
                      do e = 1,nua
                         ! A(ab) h2b(bcek)*r2a(aej)
                         res_mm23 = res_mm23 + h2b_vvvo(b,c,e,k) * r2a(a,e,j) ! (1)
                         res_mm23 = res_mm23 - h2b_vvvo(a,c,e,k) * r2a(b,e,j) ! (ab)
                         ! h2a(baje)*r2b(eck)
                         res_mm23 = res_mm23 + h2a_vvov(b,a,j,e) * r2b(e,c,k) ! (1)
                         ! x2a_vvv(abe)*t2b(ecjk)
                         res_mm23 = res_mm23 + x2a_vvv(a,b,e) * t2b(e,c,j,k) ! (1)
                      end do
                      do e = 1,nub
                         ! A(ab) h2b(bcje)*r2b(aek)
                         res_mm23 = res_mm23 + h2b_vvov(b,c,j,e) * r2b(a,e,k) ! (1)
                         res_mm23 = res_mm23 - h2b_vvov(a,c,j,e) * r2b(b,e,k) ! (ab)
                         ! A(ab) x2b_vvv(ace)*t2b(bejk)
                         res_mm23 = res_mm23 + x2b_vvv(a,c,e) * t2b(b,e,j,k) ! (1)
                         res_mm23 = res_mm23 - x2b_vvv(b,c,e) * t2b(a,e,j,k) ! (ab)
                      end do
                      resid(idet) = resid(idet) + res_mm23
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
              end subroutine build_hr_3b

              subroutine build_hr_3c(resid,&
                                     r2b,&
                                     r3b_amps,r3b_excits,&
                                     r3c_amps,r3c_excits,&
                                     t2b,t2c,&
                                     h1a_vv,h1b_oo,h1b_vv,&
                                     h2b_vvvv,h2b_vovo,h2b_ovvo,h2b_vvvo,&
                                     h2c_vvvv,h2c_oooo,h2c_voov,h2c_vooo,h2c_vvov,&
                                     x2b_voo,x2b_ovo,x2b_vvv,&
                                     n3aab,n3abb,&
                                     noa,nua,nob,nub)

                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab, n3abb
                  ! Input R and T arrays
                  real(kind=8), intent(in) :: t2b(nua,nub,noa,nob), t2c(nub,nub,nob,nob)
                  real(kind=8), intent(in) :: r2b(nua,nub,nob) 
                  integer, intent(in) :: r3b_excits(n3aab,5)
                  real(kind=8), intent(in) :: r3b_amps(n3aab)
                  ! Input H  and X arrays
                  real(kind=8), intent(in) :: h1a_vv(nua,nua)
                  real(kind=8), intent(in) :: h1b_oo(nob,nob)
                  real(kind=8), intent(in) :: h1b_vv(nub,nub)
                  real(kind=8), intent(in) :: h2b_vvvv(nua,nub,nua,nub)
                  real(kind=8), intent(in) :: h2b_vovo(nua,nob,nua,nob)
                  real(kind=8), intent(in) :: h2b_ovvo(noa,nub,nua,nob)
                  real(kind=8), intent(in) :: h2b_vvvo(nua,nub,nua,nob)
                  real(kind=8), intent(in) :: h2c_vvvv(nub,nub,nub,nub)
                  real(kind=8), intent(in) :: h2c_oooo(nob,nob,nob,nob)
                  real(kind=8), intent(in) :: h2c_voov(nub,nob,nob,nub)
                  real(kind=8), intent(in) :: h2c_vooo(nub,nob,nob,nob)
                  real(kind=8), intent(in) :: h2c_vvov(nub,nub,nob,nub)
                  real(kind=8), intent(in) :: x2b_voo(nua,nob,nob)
                  real(kind=8), intent(in) :: x2b_ovo(noa,nub,nob)
                  real(kind=8), intent(in) :: x2b_vvv(nua,nub,nub)
                  ! Output and Inout variables
                  real(kind=8), intent(out) :: resid(n3abb)
                  integer, intent(inout) :: r3c_excits(n3abb,5)
                  !f2py intent(in,out) :: r3c_excits(0:n3abb-1,0:4)
                  real(kind=8), intent(inout) :: r3c_amps(n3abb)
                  !f2py intent(in,out) :: r3c_amps(0:n3abb-1)
                  ! Local variables
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:)
                  integer, allocatable :: idx_table4(:,:,:,:)
                  integer, allocatable :: idx_table3(:,:,:)
                  integer, allocatable :: loc_arr(:,:)
                  real(kind=8) :: r_amp, hmatel, res_mm23
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc
                  
                  ! Zero the container that holds H*R
                  resid = 0.0d0

                  !!! diagram 1: -A(jk) h1b(mj)*r3c(abcmk)
                  ! allocate new sorting arrays
                  nloc = nua*nub*(nub - 1)/2*(nob - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nub,nub,nua,nob))
                  !!! SB: (2,3,1,5) !!!
                  call get_index_table4(idx_table4, (/1,nub-1/), (/-1,nub/), (/1,nua/), (/2,nob/), nub, nub, nua, nob)
                  call sort4(r3c_excits, r3c_amps, loc_arr, idx_table4, (/2,3,1,5/), nub, nub, nua, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1B_oo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     a = r3c_excits(idet,1); b = r3c_excits(idet,2); c = r3c_excits(idet,3);
                     j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,a,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3c_excits(jdet,4);
                        ! compute < ab~c~j~k~ | h1b(oo) | ab~c~m~k~ >
                        hmatel = -h1b_oo(m,j)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table4(b,c,a,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3c_excits(jdet,4);
                        ! compute < ab~c~j~k~ | h1b(oo) | ab~c~m~j~ >
                        hmatel = h1b_oo(m,k)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,1,4) !!!
                  call get_index_table4(idx_table4, (/1,nub-1/), (/-1,nub/), (/1,nua/), (/1,nob-1/), nub, nub, nua, nob)
                  call sort4(r3c_excits, r3c_amps, loc_arr, idx_table4, (/2,3,1,4/), nub, nub, nua, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1B_oo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     a = r3c_excits(idet,1); b = r3c_excits(idet,2); c = r3c_excits(idet,3);
                     j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,a,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        n = r3c_excits(jdet,5);
                        ! compute < ab~c~j~k~ | h1b(oo) | ab~c~j~n~ >
                        hmatel = -h1b_oo(n,k)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table4(b,c,a,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        n = r3c_excits(jdet,5);
                        ! compute < ab~c~j~k~ | h1b(oo) | ab~c~k~n~ >
                        hmatel = h1b_oo(n,j)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 2: A(bc) h1b(be)*r3c(aecjk)
                  ! allocate new sorting arrays
                  nloc = nua*(nub - 1)*nob*(nob - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nob,nob,nua,nub))
                  !!! SB: (4,5,1,3) !!!
                  call get_index_table4(idx_table4, (/1,nob-1/), (/-1,nob/), (/1,nua/), (/2,nub/), nob, nob, nua, nub)
                  call sort4(r3c_excits, r3c_amps, loc_arr, idx_table4, (/4,5,1,3/), nob, nob, nua, nub, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1B_vv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     a = r3c_excits(idet,1); b = r3c_excits(idet,2); c = r3c_excits(idet,3);
                     j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(j,k,a,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits(jdet,2);
                        ! compute < ab~c~j~k~ | h1b(vv) | ae~c~j~k~ >
                        hmatel = h1b_vv(b,e)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table4(j,k,a,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits(jdet,2);
                        ! compute < ab~c~j~k~ | h1b(vv) | ae~b~j~k~ >
                        hmatel = -h1b_vv(c,e)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,1,2) !!!
                  call get_index_table4(idx_table4, (/1,nob-1/), (/-1,nob/), (/1,nua/), (/1,nub-1/), nob, nob, nua, nub)
                  call sort4(r3c_excits, r3c_amps, loc_arr, idx_table4, (/4,5,1,2/), nob, nob, nua, nub, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1B_vv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     a = r3c_excits(idet,1); b = r3c_excits(idet,2); c = r3c_excits(idet,3);
                     j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(j,k,a,b)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits(jdet,3);
                        ! compute < ab~c~j~k~ | h1b(vv) | ab~f~j~k~ >
                        hmatel = h1b_vv(c,f)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table4(j,k,a,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits(jdet,3);
                        ! compute < ab~c~j~k~ | h1b(vv) | ac~f~j~k~ >
                        hmatel = -h1b_vv(b,f)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)
                 
                  !!! diagram 3: h1a(ae)*r3c(ebcjk)
                  ! allocate new sorting arrays
                  nloc = nub*(nub - 1)/2*nob*(nob - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nub,nub,nob,nob))
                  !!! SB: (2,3,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nub-1/), (/-1,nub/), (/1,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(r3c_excits, r3c_amps, loc_arr, idx_table4, (/2,3,4,5/), nub, nub, nob, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_vv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     a = r3c_excits(idet,1); b = r3c_excits(idet,2); c = r3c_excits(idet,3);
                     j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3c_excits(jdet,1);
                        ! compute < ab~c~j~k~ | h1a(vv) | db~c~j~k~ >
                        hmatel = h1a_vv(a,d)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 4: h2c(mnjk)*r3c(abcmn)
                  ! allocate new sorting arrays
                  nloc = nua*nub*(nub - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nub,nub,nua))
                  !!! SB: (2,3,1) !!!
                  call get_index_table3(idx_table3, (/1,nub-1/), (/-1,nub/), (/1,nua/), nub, nub, nua)
                  call sort3(r3c_excits, r3c_amps, loc_arr, idx_table3, (/2,3,1/), nub, nub, nua, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2C_oooo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     a = r3c_excits(idet,1); b = r3c_excits(idet,2); c = r3c_excits(idet,3);
                     j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,a)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3c_excits(jdet,4); n = r3c_excits(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(oooo) | ab~c~m~n~ >
                        hmatel = h2c_oooo(m,n,j,k)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 5: A(bc)A(jk) h2b(mbej)*r3b(aecmk)
                  ! allocate and copy over r3b arrays
                  allocate(amps_buff(n3aab),excits_buff(n3aab,5))
                  amps_buff(:) = r3b_amps(:)
                  excits_buff(:,:) = r3b_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*nub*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nub,nob))
                  !!! SB: (1,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/1,nub/), (/1,nob/), nua, nub, nob)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,3,5/), nua, nub, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     a = r3c_excits(idet,1); b = r3c_excits(idet,2); c = r3c_excits(idet,3);
                     j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(ovvo) | aec~mk~ >
                        hmatel = h2b_ovvo(m,b,e,j)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(ovvo) | aeb~mk~ >
                        hmatel = -h2b_ovvo(m,c,e,j)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(ovvo) | aec~mj~ >
                        hmatel = -h2b_ovvo(m,b,e,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(ovvo) | aeb~mj~ >
                        hmatel = h2b_ovvo(m,c,e,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,5) !!!
                  call get_index_table3(idx_table3, (/2,nua/), (/1,nub/), (/1,nob/), nua, nub, nob)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/2,3,5/), nua, nub, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     a = r3c_excits(idet,1); b = r3c_excits(idet,2); c = r3c_excits(idet,3);
                     j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(ovvo) | dac~mk~ >
                        hmatel = -h2b_ovvo(m,b,d,j)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(ovvo) | dab~mk~ >
                        hmatel = h2b_ovvo(m,c,d,j)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(ovvo) | dac~mj~ >
                        hmatel = h2b_ovvo(m,b,d,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(ovvo) | dab~mj~ >
                        hmatel = -h2b_ovvo(m,c,d,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3,excits_buff,amps_buff)

                  !!! diagram 6: A(bc)A(jk) h2c(bmje) * r3c(aecmk) 
                  ! allocate new sorting arrays
                  nloc = nua*(nub - 1)*(nob - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nub,nob))
                  !!! SB: (1,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/2,nub/), (/2,nob/), nua, nub, nob)
                  call sort3(r3c_excits, r3c_amps, loc_arr, idx_table3, (/1,3,5/), nua, nub, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     a = r3c_excits(idet,1); b = r3c_excits(idet,2); c = r3c_excits(idet,3);
                     j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits(jdet,2); m = r3c_excits(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~c~m~k~ >
                        hmatel = h2c_voov(b,m,j,e)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits(jdet,2); m = r3c_excits(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~b~m~k~ >
                        hmatel = -h2c_voov(c,m,j,e)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits(jdet,2); m = r3c_excits(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~c~m~j~ >
                        hmatel = -h2c_voov(b,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits(jdet,2); m = r3c_excits(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~b~m~j~ >
                        hmatel = h2c_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,5) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/1,nub-1/), (/2,nob/), nua, nub, nob)
                  call sort3(r3c_excits, r3c_amps, loc_arr, idx_table3, (/1,2,5/), nua, nub, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     a = r3c_excits(idet,1); b = r3c_excits(idet,2); c = r3c_excits(idet,3);
                     j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,b,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits(jdet,3); m = r3c_excits(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ab~f~m~k~ >
                        hmatel = h2c_voov(c,m,j,f)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits(jdet,3); m = r3c_excits(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ac~f~m~k~ >
                        hmatel = -h2c_voov(b,m,j,f)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits(jdet,3); m = r3c_excits(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ab~f~m~j~ >
                        hmatel = -h2c_voov(c,m,k,f)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits(jdet,3); m = r3c_excits(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ac~f~m~j~ >
                        hmatel = h2c_voov(b,m,k,f)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,4) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/2,nub/), (/1,nob-1/), nua, nub, nob)
                  call sort3(r3c_excits, r3c_amps, loc_arr, idx_table3, (/1,3,4/), nua, nub, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     a = r3c_excits(idet,1); b = r3c_excits(idet,2); c = r3c_excits(idet,3);
                     j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits(jdet,2); n = r3c_excits(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~c~j~n~ >
                        hmatel = h2c_voov(b,n,k,e)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits(jdet,2); n = r3c_excits(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~b~j~n~ >
                        hmatel = -h2c_voov(c,n,k,e)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits(jdet,2); n = r3c_excits(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~c~k~n~ >
                        hmatel = -h2c_voov(b,n,j,e)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits(jdet,2); n = r3c_excits(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~b~k~n~ >
                        hmatel = h2c_voov(c,n,j,e)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/1,nub-1/), (/1,nob-1/), nua, nub, nob)
                  call sort3(r3c_excits, r3c_amps, loc_arr, idx_table3, (/1,2,4/), nua, nub, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     a = r3c_excits(idet,1); b = r3c_excits(idet,2); c = r3c_excits(idet,3);
                     j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,b,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits(jdet,3); n = r3c_excits(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ab~f~j~n~ >
                        hmatel = h2c_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits(jdet,3); n = r3c_excits(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ac~f~j~n~ >
                        hmatel = -h2c_voov(b,n,k,f)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits(jdet,3); n = r3c_excits(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ab~f~k~n~ >
                        hmatel = -h2c_voov(c,n,j,f)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits(jdet,3); n = r3c_excits(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ac~f~k~n~ >
                        hmatel = h2c_voov(b,n,j,f)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 7: -A(jk) h2b(amej)*r3c(ebcmk)
                  ! allocate new sorting arrays
                  nloc = nub*(nub - 1)/2*(nob - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nub,nub,nob))
                  !!! SB: (2,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nub-1/), (/-1,nub/), (/2,nob/), nub, nub, nob)
                  call sort3(r3c_excits, r3c_amps, loc_arr, idx_table3, (/2,3,5/), nub, nub, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vovo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     a = r3c_excits(idet,1); b = r3c_excits(idet,2); c = r3c_excits(idet,3);
                     j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3c_excits(jdet,1); m = r3c_excits(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(vovo) | db~c~m~k~ >
                        hmatel = -h2b_vovo(a,m,d,j)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3c_excits(jdet,1); m = r3c_excits(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(vovo) | db~c~m~j~ >
                        hmatel = h2b_vovo(a,m,d,k)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4) !!!
                  call get_index_table3(idx_table3, (/1,nub-1/), (/-1,nub/), (/1,nob-1/), nub, nub, nob)
                  call sort3(r3c_excits, r3c_amps, loc_arr, idx_table3, (/2,3,4/), nub, nub, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vovo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     a = r3c_excits(idet,1); b = r3c_excits(idet,2); c = r3c_excits(idet,3);
                     j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3c_excits(jdet,1); n = r3c_excits(jdet,5);
                        ! compute < ab~c~j~k~ | h2b(vovo) | db~c~j~n~ >
                        hmatel = -h2b_vovo(a,n,d,k)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3c_excits(jdet,1); n = r3c_excits(jdet,5);
                        ! compute < ab~c~j~k~ | h2b(vovo) | db~c~k~n~ >
                        hmatel = h2b_vovo(a,n,d,j)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 8: A(bc) h2b(abef)*r3c(efcjk)
                  ! allocate new sorting arrays
                  nloc = nob*(nob - 1)/2*(nub - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nob,nob,nub))
                  !!! SB: (4,5,3) !!!
                  call get_index_table3(idx_table3, (/1,nob-1/), (/-1,nob/), (/2,nub/), nob, nob, nub)
                  call sort3(r3c_excits, r3c_amps, loc_arr, idx_table3, (/4,5,3/), nob, nob, nub, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vvvv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     a = r3c_excits(idet,1); b = r3c_excits(idet,2); c = r3c_excits(idet,3);
                     j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(j,k,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3c_excits(jdet,1); e = r3c_excits(jdet,2);
                        ! compute < ab~c~j~k~ | h2b(vvvv) | de~c~j~k~ >
                        hmatel = h2b_vvvv(a,b,d,e)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table3(j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3c_excits(jdet,1); e = r3c_excits(jdet,2);
                        ! compute < ab~c~j~k~ | h2b(vvvv) | de~b~j~k~ >
                        hmatel = -h2b_vvvv(a,c,d,e)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,2) !!!
                  call get_index_table3(idx_table3, (/1,nob-1/), (/-1,nob/), (/1,nub-1/), nob, nob, nub)
                  call sort3(r3c_excits, r3c_amps, loc_arr, idx_table3, (/4,5,2/), nob, nob, nub, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vvvv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     a = r3c_excits(idet,1); b = r3c_excits(idet,2); c = r3c_excits(idet,3);
                     j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(j,k,b)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3c_excits(jdet,1); f = r3c_excits(jdet,3);
                        ! compute < ab~c~j~k~ | h2b(vvvv) | db~f~j~k~ >
                        hmatel = h2b_vvvv(a,c,d,f)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table3(j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3c_excits(jdet,1); f = r3c_excits(jdet,3);
                        ! compute < ab~c~j~k~ | h2b(vvvv) | dc~f~j~k~ >
                        hmatel = -h2b_vvvv(a,b,d,f)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 9: 1/2 h2c(bcef)*r3c(aefjk)
                  ! allocate new sorting arrays
                  nloc = nua*nob*(nob - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nob,nob,nua))
                  !!! SB: (4,5,1) !!!
                  call get_index_table3(idx_table3, (/1,nob-1/), (/-1,nob/), (/1,nua/), nob, nob, nua)
                  call sort3(r3c_excits, r3c_amps, loc_arr, idx_table3, (/4,5,1/), nob, nob, nua, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2C_vvvv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     a = r3c_excits(idet,1); b = r3c_excits(idet,2); c = r3c_excits(idet,3);
                     j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(j,k,a)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits(jdet,2); f = r3c_excits(jdet,3);
                        ! compute < ab~c~j~k~ | h2b(vvvv) | ae~f~j~k~ >
                        hmatel = h2c_vvvv(b,c,e,f)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp t2b,t2c,r2b,&
                  !$omp h2b_vvvo,&
                  !$omp h2c_vooo,h2c_vvov,&
                  !$omp x2b_voo,x2b_ovo,x2b_vvv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(idet,a,b,c,d,i,j,k,l,m,n,e,f,res_mm23)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                      a = r3c_excits(idet,1); b = r3c_excits(idet,2); c = r3c_excits(idet,3);
                      j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                      ! zero out value
                      res_mm23 = 0.0d0
                      do m = 1,noa
                         ! -A(jk)A(bc) x2b_ovo(mck) * t2b(abmj)
                         res_mm23 = res_mm23 - x2b_ovo(m,c,k) * t2b(a,b,m,j) ! (1)
                         res_mm23 = res_mm23 + x2b_ovo(m,b,k) * t2b(a,c,m,j) ! (bc)
                         res_mm23 = res_mm23 + x2b_ovo(m,c,j) * t2b(a,b,m,k) ! (jk)
                         res_mm23 = res_mm23 - x2b_ovo(m,b,j) * t2b(a,c,m,k) ! (bc)(jk)
                      end do
                      do m = 1,nob
                         ! -A(bc) h2c(cmkj) * r2b(abm)
                         res_mm23 = res_mm23 - h2c_vooo(c,m,k,j) * r2b(a,b,m) ! (1)
                         res_mm23 = res_mm23 + h2c_vooo(b,m,k,j) * r2b(a,c,m) ! (bc)
                         ! -A(jk) x2b_voo(amj) * t2c(bcmk)
                         res_mm23 = res_mm23 - x2b_voo(a,m,j) * t2c(b,c,m,k) ! (1)
                         res_mm23 = res_mm23 + x2b_voo(a,m,k) * t2c(b,c,m,j) ! (jk)
                      end do
                      do e = 1,nua
                         ! A(bc)A(jk) h2b(acek) * r2b(ebj)
                         res_mm23 = res_mm23 + h2b_vvvo(a,c,e,k) * r2b(e,b,j) ! (1)
                         res_mm23 = res_mm23 - h2b_vvvo(a,b,e,k) * r2b(e,c,j) ! (bc)
                         res_mm23 = res_mm23 - h2b_vvvo(a,c,e,j) * r2b(e,b,k) ! (jk)
                         res_mm23 = res_mm23 + h2b_vvvo(a,b,e,j) * r2b(e,c,k) ! (bc)(jk)
                      end do
                      do e = 1,nub
                         ! A(jk) h2c(cbke) * r2b(aej)
                         res_mm23 = res_mm23 + h2c_vvov(c,b,k,e) * r2b(a,e,j) ! (1)
                         res_mm23 = res_mm23 - h2c_vvov(c,b,j,e) * r2b(a,e,k) ! (jk)
                         ! A(bc) x2b_vvv(abe) * t2c(ecjk)
                         res_mm23 = res_mm23 + x2b_vvv(a,b,e) * t2c(e,c,j,k) ! (1)
                         res_mm23 = res_mm23 - x2b_vvv(a,c,e) * t2c(e,b,j,k) ! (bc)
                      end do
                      resid(idet) = resid(idet) + res_mm23
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
              end subroutine build_hr_3c

              subroutine update_r(r1a,r2a,r2b,&
                                  r3a_amps, r3a_excits,&
                                  r3b_amps, r3b_excits,&
                                  r3c_amps, r3c_excits,&
                                  omega,&
                                  h1a_oo, h1a_vv, h1b_oo, h1b_vv,&
                                  n3aaa, n3aab, n3abb,&
                                  noa, nua, nob, nub)

                      integer, intent(in) :: noa, nua, nob, nub, n3aaa, n3aab, n3abb
                      integer, intent(in) :: r3a_excits(n3aaa,5), r3b_excits(n3aab,5), r3c_excits(n3abb,5)
                      real(kind=8), intent(in) :: h1a_oo(noa,noa), h1a_vv(nua,nua), h1b_oo(nob,nob), h1b_vv(nub,nub)
                      real(kind=8), intent(in) :: omega
                      
                      real(kind=8), intent(inout) :: r1a(1:nua)
                      !f2py intent(in,out) :: r1a(0:nua-1)
                      real(kind=8), intent(inout) :: r2a(1:nua,1:nua,1:noa)
                      !f2py intent(in,out) :: r2a(0:nua-1,0:nua-1,0:noa-1)
                      real(kind=8), intent(inout) :: r2b(1:nua,1:nub,1:nob)
                      !f2py intent(in,out) :: r2b(0:nua-1,0:nub-1,0:nob-1)
                      real(kind=8), intent(inout) :: r3a_amps(n3aaa)
                      !f2py intent(in,out) :: r3a_amps(0:n3aaa-1) 
                      real(kind=8), intent(inout) :: r3b_amps(n3aab)
                      !f2py intent(in,out) :: r3b_amps(0:n3aab-1) 
                      real(kind=8), intent(inout) :: r3c_amps(n3abb)
                      !f2py intent(in,out) :: r3c_amps(0:n3abb-1) 
                      
                      integer :: j, k, a, b, c, idet
                      real(kind=8) :: denom

                      do a = 1,nua
                          denom = omega - H1A_vv(a,a)
                          if (denom==0.0d0) cycle
                          r1a(a) = r1a(a)/denom
                      end do

                      do j = 1,noa
                        do a = 1,nua
                           do b = 1,nua
                              if (a==b) cycle
                              denom = omega - H1A_vv(a,a) - H1A_vv(b,b) + H1A_oo(j,j)
                              r2a(a,b,j) = r2a(a,b,j)/denom
                          end do
                        end do
                      end do

                      do j = 1,nob
                        do a = 1,nua
                           do b = 1,nub
                              denom = omega - H1A_vv(a,a) - H1B_vv(b,b) + H1B_oo(j,j)
                              r2b(a,b,j) = r2b(a,b,j)/denom
                          end do
                        end do
                      end do

                      do idet = 1, n3aaa
                         a = r3a_excits(idet,1); b = r3a_excits(idet,2); c = r3a_excits(idet,3);
                         j = r3a_excits(idet,4); k = r3a_excits(idet,5);

                         denom = H1A_vv(a,a) + H1A_vv(b,b) + H1A_vv(c,c)&
                                -H1A_oo(j,j) - H1A_oo(k,k)
                         
                         r3a_amps(idet) = r3a_amps(idet)/(omega - denom)
                      end do

                      do idet = 1, n3aab
                         a = r3b_excits(idet,1); b = r3b_excits(idet,2); c = r3b_excits(idet,3);
                         j = r3b_excits(idet,4); k = r3b_excits(idet,5);

                         denom = H1A_vv(a,a) + H1A_vv(b,b) + H1B_vv(c,c)&
                                -H1A_oo(j,j) - H1B_oo(k,k)
                         
                         r3b_amps(idet) = r3b_amps(idet)/(omega - denom)
                      end do

                      do idet = 1, n3abb
                         a = r3c_excits(idet,1); b = r3c_excits(idet,2); c = r3c_excits(idet,3);
                         j = r3c_excits(idet,4); k = r3c_excits(idet,5);

                         denom = H1A_vv(a,a) + H1B_vv(b,b) + H1B_vv(c,c)&
                                -H1B_oo(j,j) - H1B_oo(k,k)
                         
                         r3c_amps(idet) = r3c_amps(idet)/(omega - denom)
                      end do

              end subroutine update_r

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!! SORTING FUNCTIONS !!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

               subroutine get_index_table3(idx_table, rng1, rng2, rng3, n1, n2, n3)

                    integer, intent(in) :: n1, n2, n3
                    integer, intent(in) :: rng1(2), rng2(2), rng3(2)
      
                    integer, intent(inout) :: idx_table(n1,n2,n3)
      
                    integer :: kout
                    integer :: p, q, r
      
                    idx_table = 0
                    if (rng1(1) > 0 .and. rng2(1) < 0 .and. rng3(1) < 0) then ! p < q < r
                       kout = 1
                       do p = rng1(1), rng1(2)
                          do q = p-rng2(1), rng2(2)
                             do r = q-rng3(1), rng3(2)
                                idx_table(p,q,r) = kout
                                kout = kout + 1
                             end do
                          end do
                       end do
                    elseif (rng1(1) > 0 .and. rng2(1) > 0 .and. rng3(1) < 0) then ! p, q < r
                       kout = 1
                       do p = rng1(1), rng1(2)
                          do q = rng2(1), rng2(2)
                             do r = q-rng3(1), rng3(2)
                                idx_table(p,q,r) = kout
                                kout = kout + 1
                             end do
                          end do
                       end do
                    elseif (rng1(1) > 0 .and. rng2(1) < 0 .and. rng3(1) > 0) then ! p < q, r
                       kout = 1
                       do p = rng1(1), rng1(2)
                          do q = p-rng2(1), rng2(2)
                             do r = rng3(1), rng3(2)
                                idx_table(p,q,r) = kout
                                kout = kout + 1
                             end do
                          end do
                       end do
                    else ! p, q, r
                       kout = 1
                       do p = rng1(1), rng1(2)
                          do q = rng2(1), rng2(2)
                             do r = rng3(1), rng3(2)
                                idx_table(p,q,r) = kout
                                kout = kout + 1
                             end do
                          end do
                       end do
                    end if

              end subroutine get_index_table3

              subroutine sort3(excits, amps, loc_arr, idx_table, idims, n1, n2, n3, nloc, n3p, x1a)

                    integer, intent(in) :: n1, n2, n3, nloc, n3p
                    integer, intent(in) :: idims(3)
                    integer, intent(in) :: idx_table(n1,n2,n3)
      
                    integer, intent(inout) :: loc_arr(2,nloc)
                    integer, intent(inout) :: excits(n3p,5)
                    real(kind=8), intent(inout) :: amps(n3p)
                    real(kind=8), intent(inout), optional :: x1a(n3p)
      
                    integer :: idet
                    integer :: p, q, r
                    integer :: p1, q1, r1, p2, q2, r2
                    integer :: pqr1, pqr2
                    integer, allocatable :: temp(:), idx(:)
      
                    allocate(temp(n3p),idx(n3p))
                    do idet = 1, n3p
                       p = excits(idet,idims(1)); q = excits(idet,idims(2)); r = excits(idet,idims(3));
                       temp(idet) = idx_table(p,q,r)
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
                       if (excits(1,1)==1 .and. excits(1,2)==1 .and. excits(1,3)==1 .and. excits(1,4)==1 .and. excits(1,5)==1) return
                       p2 = excits(n3p,idims(1)); q2 = excits(n3p,idims(2)); r2 = excits(n3p,idims(3));
                       pqr2 = idx_table(p2,q2,r2)
                    else               
                       pqr2 = -1
                    end if
                    do idet = 1, n3p-1
                       p1 = excits(idet,idims(1));   q1 = excits(idet,idims(2));   r1 = excits(idet,idims(3));
                       p2 = excits(idet+1,idims(1)); q2 = excits(idet+1,idims(2)); r2 = excits(idet+1,idims(3));
                       pqr1 = idx_table(p1,q1,r1)
                       pqr2 = idx_table(p2,q2,r2)
                       if (pqr1 /= pqr2) then
                          loc_arr(2,pqr1) = idet
                          loc_arr(1,pqr2) = idet+1
                       end if
                    end do
                    if (n3p > 1) then
                       loc_arr(2,pqr2) = n3p
                    end if
              end subroutine sort3

              subroutine get_index_table4(idx_table, rng1, rng2, rng3, rng4, n1, n2, n3, n4)

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

              end subroutine get_index_table4

              subroutine sort4(excits, amps, loc_arr, idx_table, idims, n1, n2, n3, n4, nloc, n3p, x1a)

                    integer, intent(in) :: n1, n2, n3, n4, nloc, n3p
                    integer, intent(in) :: idims(4)
                    integer, intent(in) :: idx_table(n1,n2,n3,n4)
      
                    integer, intent(inout) :: loc_arr(2,nloc)
                    integer, intent(inout) :: excits(n3p,5)
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
                       if (excits(1,1)==1 .and. excits(1,2)==1 .and. excits(1,3)==1 .and. excits(1,4)==1 .and. excits(1,5)==1) return
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

end module eaeom3_p_loops
