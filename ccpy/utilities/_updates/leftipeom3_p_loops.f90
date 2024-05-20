module leftipeom3_p_loops

      use omp_lib

      implicit none

      contains

              subroutine build_lh_2a(sigma_2a,&
                                     l3a_amps,l3a_excits,&
                                     l3b_amps,l3b_excits,&
                                     h2a_vooo,h2a_vvov,h2b_ovoo,h2b_vvvo,&
                                     n3aaa,n3aab,&
                                     noa,nua,nob,nub)

                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa, n3aab
                  ! Input R and T arrays
                  integer, intent(in) :: l3a_excits(n3aaa,5)
                  integer, intent(in) :: l3b_excits(n3aab,5)
                  real(kind=8), intent(in) :: l3a_amps(n3aaa)
                  real(kind=8), intent(in) :: l3b_amps(n3aab)
                  ! Input H  and X arrays
                  real(kind=8), intent(in) :: h2a_vooo(nua,noa,noa,noa)
                  real(kind=8), intent(in) :: h2a_vvov(nua,nua,noa,nua)
                  real(kind=8), intent(in) :: h2b_vvvo(nua,nub,nua,nob)
                  real(kind=8), intent(in) :: h2b_ovoo(noa,nub,noa,nob)
                  ! Output and Inout variables
                  real(kind=8), intent(inout) :: sigma_2a(noa,nua,noa)
                  !f2py intent(in,out) :: sigma_2a(0:noa-1,0:nua-1,0:noa-1)
                  ! Local variables
                  real(kind=8) :: l_amp, val
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  do idet = 1,n3aaa
                     l_amp = l3a_amps(idet)
                     ! x2a(ibj) <- A(j/mn)A(bf) -h2a_vooo(finm)*l3a(bfmjn)
                     b = l3a_excits(idet,1); f = l3a_excits(idet,2);
                     m = l3a_excits(idet,3); j = l3a_excits(idet,4); n = l3a_excits(idet,5);
                     sigma_2a(:,b,j) = sigma_2a(:,b,j) - h2a_vooo(f,:,n,m)*l_amp ! (1)
                     sigma_2a(:,b,m) = sigma_2a(:,b,m) + h2a_vooo(f,:,n,j)*l_amp ! (jm)
                     sigma_2a(:,b,n) = sigma_2a(:,b,n) + h2a_vooo(f,:,j,m)*l_amp ! (jn)
                     sigma_2a(:,f,j) = sigma_2a(:,f,j) + h2a_vooo(b,:,n,m)*l_amp ! (bf)
                     sigma_2a(:,f,m) = sigma_2a(:,f,m) - h2a_vooo(b,:,n,j)*l_amp ! (jm)(bf)
                     sigma_2a(:,f,n) = sigma_2a(:,f,n) - h2a_vooo(b,:,j,m)*l_amp ! (jn)(bf)
                     ! x2a(ibj) <- A(n/ij) h2a_vvov(fenb)*l3a(efijn)
                     e = l3a_excits(idet,1); f = l3a_excits(idet,2);
                     i = l3a_excits(idet,3); j = l3a_excits(idet,4); n = l3a_excits(idet,5);
                     sigma_2a(i,:,j) = sigma_2a(i,:,j) + h2a_vvov(f,e,n,:)*l_amp ! (1)
                     sigma_2a(j,:,n) = sigma_2a(j,:,n) + h2a_vvov(f,e,i,:)*l_amp ! (1)
                     sigma_2a(i,:,n) = sigma_2a(i,:,n) - h2a_vvov(f,e,j,:)*l_amp ! (1)
                  end do                 

                  do idet = 1,n3aab
                     l_amp = l3b_amps(idet)
                     ! x2a(ibj) <- h2b_vvvo(efbn)*l3b(efijn)
                     e = l3b_excits(idet,1); f = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); j = l3b_excits(idet,4); n = l3b_excits(idet,5);
                     sigma_2a(i,:,j) = sigma_2a(i,:,j) + h2b_vvvo(e,f,:,n)*l_amp ! (1)
                     ! x2a(ibj) <- A(jm) -h2b_ovoo(ifmn)*l3b(bfmjn)
                     b = l3b_excits(idet,1); f = l3b_excits(idet,2);
                     m = l3b_excits(idet,3); j = l3b_excits(idet,4); n = l3b_excits(idet,5);
                     sigma_2a(:,b,j) = sigma_2a(:,b,j) - h2b_ovoo(:,f,m,n)*l_amp ! (1)
                     sigma_2a(:,b,m) = sigma_2a(:,b,m) + h2b_ovoo(:,f,j,n)*l_amp ! (jm)
                  end do                  
                  
                  ! antisymmetrize (this replaces the x2a -= np.transpose(x2a, (...)) stuff in vector update
                  do i = 1, noa
                     do b = 1, nua
                        do j = i+1, noa
                           val = sigma_2a(i,b,j) - sigma_2a(j,b,i)
                           sigma_2a(i,b,j) =  val
                           sigma_2a(j,b,i) = -val
                        end do
                     end do
                  end do
                  ! (H(2) * T3)_C terms are vectorized and generally broadcast to diagonal elements, which should
                  ! be 0. Set them to 0 manually (you need to do this).
                  do i = 1,noa
                     sigma_2a(i,:,i) = 0.0d0
                  end do
              end subroutine build_lh_2a

              subroutine build_lh_2b(sigma_2b,&
                                     l3b_amps,l3b_excits,&
                                     l3c_amps,l3c_excits,&
                                     h2a_vooo,&
                                     h2b_vooo,h2b_ovoo,h2b_vvov,&
                                     h2c_vooo,h2c_vvov,&
                                     n3aab,n3abb,&
                                     noa,nua,nob,nub)

                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab, n3abb
                  ! Input R and T arrays
                  integer, intent(in) :: l3b_excits(n3aab,5)
                  integer, intent(in) :: l3c_excits(n3abb,5)
                  real(kind=8), intent(in) :: l3b_amps(n3aab)
                  real(kind=8), intent(in) :: l3c_amps(n3abb)
                  ! Input H  and X arrays
                  real(kind=8), intent(in) :: h2a_vooo(nua,noa,noa,noa)
                  real(kind=8), intent(in) :: h2b_vooo(nua,nob,noa,nob)
                  real(kind=8), intent(in) :: h2b_ovoo(noa,nub,noa,nob)
                  real(kind=8), intent(in) :: h2b_vvov(nua,nub,noa,nub)
                  real(kind=8), intent(in) :: h2c_vooo(nub,nob,nob,nob)
                  real(kind=8), intent(in) :: h2c_vvov(nub,nub,nob,nub)
                  ! Output and Inout variables
                  real(kind=8), intent(inout) :: sigma_2b(noa,nub,nob)
                  !f2py intent(in,out) :: sigma_2b(0:noa-1,0:nub-1,0:nob-1)
                  ! Local variables
                  real(kind=8) :: l_amp, val
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  do idet = 1,n3aab
                     l_amp = l3b_amps(idet)
                     ! x2b(ibj) <- -l3b(fbmnj)*h2a_vooo(finm)
                     f = l3b_excits(idet,1); b = l3b_excits(idet,2);
                     m = l3b_excits(idet,3); n = l3b_excits(idet,4); j = l3b_excits(idet,5);
                     sigma_2b(:,b,j) = sigma_2b(:,b,j) - h2a_vooo(f,:,n,m)*l_amp ! (1)
                     ! x2b(ibj) <- A(in) l3b(feinj)*h2b_vvov(fenb)
                     f = l3b_excits(idet,1); e = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); n = l3b_excits(idet,4); j = l3b_excits(idet,5);
                     sigma_2b(i,:,j) = sigma_2b(i,:,j) + h2b_vvov(f,e,n,:)*l_amp ! (1)
                     sigma_2b(n,:,j) = sigma_2b(n,:,j) - h2b_vvov(f,e,i,:)*l_amp ! (in)
                     ! x2b(ibj) <- A(in) -l3b(fbinm)*h2b_vooo(fjnm)
                     f = l3b_excits(idet,1); b = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); n = l3b_excits(idet,4); m = l3b_excits(idet,5);
                     sigma_2b(i,b,:) = sigma_2b(i,b,:) - h2b_vooo(f,:,n,m)*l_amp ! (1)
                     sigma_2b(n,b,:) = sigma_2b(n,b,:) + h2b_vooo(f,:,i,m)*l_amp ! (in)
                  end do                 

                  do idet = 1,n3abb
                     l_amp = l3c_amps(idet)
                     ! x2b(ibj) <- A(fb)A(jn) -l3c(fbmnj)*h2b_ovoo(ifmn)
                     f = l3c_excits(idet,1); b = l3c_excits(idet,2);
                     m = l3c_excits(idet,3); n = l3c_excits(idet,4); j = l3c_excits(idet,5);
                     sigma_2b(:,b,j) = sigma_2b(:,b,j) - h2b_ovoo(:,f,m,n)*l_amp ! (1)
                     sigma_2b(:,b,n) = sigma_2b(:,b,n) + h2b_ovoo(:,f,m,j)*l_amp ! (jn)
                     sigma_2b(:,f,j) = sigma_2b(:,f,j) + h2b_ovoo(:,b,m,n)*l_amp ! (fb)
                     sigma_2b(:,f,n) = sigma_2b(:,f,n) - h2b_ovoo(:,b,m,j)*l_amp ! (jn)(fb)
                     ! x2b(ibj) <- A(jn) l3c(feinj)*h2c_vvov(fenb)
                     f = l3c_excits(idet,1); e = l3c_excits(idet,2);
                     i = l3c_excits(idet,3); n = l3c_excits(idet,4); j = l3c_excits(idet,5);
                     sigma_2b(i,:,j) = sigma_2b(i,:,j) + h2c_vvov(f,e,n,:)*l_amp ! (1)
                     sigma_2b(i,:,n) = sigma_2b(i,:,n) - h2c_vvov(f,e,j,:)*l_amp ! (jn)
                     ! x2b(ibj) <- A(fb) -l3c(fbinm)*h2c_vooo(fjnm)
                     f = l3c_excits(idet,1); b = l3c_excits(idet,2);
                     i = l3c_excits(idet,3); n = l3c_excits(idet,4); m = l3c_excits(idet,5);
                     sigma_2b(i,b,:) = sigma_2b(i,b,:) - h2c_vooo(f,:,n,m)*l_amp ! (1)
                     sigma_2b(i,f,:) = sigma_2b(i,f,:) + h2c_vooo(b,:,n,m)*l_amp ! (fb)
                  end do                  
              end subroutine build_lh_2b

              subroutine build_lh_3a(resid,&
                                     l1a,l2a,&
                                     l3a_amps,l3a_excits,&
                                     l3b_amps,l3b_excits,&
                                     h1a_ov,h1a_oo,h1a_vv,&
                                     h2a_vvvv,h2a_oooo,h2a_voov,h2a_ooov,h2a_vovv,h2a_oovv,&
                                     h2b_voov,&
                                     x2a_vvo,x2a_ooo,&
                                     n3aaa,n3aab,&
                                     noa,nua,nob,nub)

                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa, n3aab
                  ! Input L arrays
                  real(kind=8), intent(in) :: l1a(noa)
                  real(kind=8), intent(in) :: l2a(noa,nua,noa)
                  integer, intent(in) :: l3b_excits(n3aab,5)
                  real(kind=8), intent(in) :: l3b_amps(n3aab)
                  ! Input H  and X arrays
                  real(kind=8), intent(in) :: h1a_ov(noa,nua)
                  real(kind=8), intent(in) :: h1a_oo(noa,noa)
                  real(kind=8), intent(in) :: h1a_vv(nua,nua)
                  real(kind=8), intent(in) :: h2a_vvvv(nua,nua,nua,nua)
                  real(kind=8), intent(in) :: h2a_oooo(noa,noa,noa,noa)
                  real(kind=8), intent(in) :: h2a_voov(nua,noa,noa,nua)
                  real(kind=8), intent(in) :: h2a_ooov(noa,noa,noa,nua)
                  real(kind=8), intent(in) :: h2a_vovv(nua,noa,nua,nua)
                  real(kind=8), intent(in) :: h2a_oovv(noa,noa,nua,nua)
                  real(kind=8), intent(in) :: h2b_voov(nua,nob,noa,nub)
                  real(kind=8), intent(in) :: x2a_vvo(nua,nua,noa)
                  real(kind=8), intent(in) :: x2a_ooo(noa,noa,noa)
                  ! Output and Inout variables
                  real(kind=8), intent(out) :: resid(n3aaa)
                  integer, intent(inout) :: l3a_excits(n3aaa,5)
                  !f2py intent(in,out) :: l3a_excits(0:n3aaa-1,0:4)
                  real(kind=8), intent(inout) :: l3a_amps(n3aaa)
                  !f2py intent(in,out) :: l3a_amps(0:n3aaa-1)
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

                  !!!! diagram 1: -A(j/ik) h1a(mj)*l3a(bcimk)
                  ! allocate new sorting arrays
                  nloc = (noa - 1)*(noa - 2)/2*nua*(nua - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(noa,noa,nua,nua))
                  !!! SB: (3,5,1,2) !!!
                  call get_index_table4(idx_table4, (/1,noa-2/), (/-2,noa/), (/1,nua-1/), (/-1,nua/), noa, noa, nua, nua)
                  call sort4(l3a_excits, l3a_amps, loc_arr, idx_table4, (/3,5,1,2/), noa, noa, nua, nua, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits,&
                  !$omp l3a_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_oo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = l3a_excits(idet,1); c = l3a_excits(idet,2);
                     i = l3a_excits(idet,3); j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(i,k,b,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3a_excits(jdet,4);
                        ! compute < bcijk | h1a(oo) | bcimk >
                        hmatel = -h1a_oo(m,j)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table4(j,k,b,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3a_excits(jdet,4);
                        ! compute < bcijk | h1a(oo) | bcjmk >
                        hmatel = h1a_oo(m,i)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table4(i,j,b,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3a_excits(jdet,4);
                        ! compute < bcijk | h1a(oo) | bcimj >
                        hmatel = h1a_oo(m,k)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (3,4,1,2) !!!
                  call get_index_table4(idx_table4, (/1,noa-2/), (/-1,noa-1/), (/1,nua-1/), (/-1,nua/), noa, noa, nua, nua)
                  call sort4(l3a_excits, l3a_amps, loc_arr, idx_table4, (/3,4,1,2/), noa, noa, nua, nua, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits,&
                  !$omp l3a_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_oo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = l3a_excits(idet,1); c = l3a_excits(idet,2);
                     i = l3a_excits(idet,3); j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(i,j,b,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        n = l3a_excits(jdet,5);
                        ! compute < bcijk | h1a(oo) | bcijn >
                        hmatel = -h1a_oo(n,k)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ik), (-1)
                     idx = idx_table4(j,k,b,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        n = l3a_excits(jdet,5);
                        ! compute < bcijk | h1a(oo) | bcjkn >
                        hmatel = -h1a_oo(n,i)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table4(i,k,b,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        n = l3a_excits(jdet,5);
                        ! compute < bcijk | h1a(oo) | bcikn >
                        hmatel = h1a_oo(n,j)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,1,2) !!!
                  call get_index_table4(idx_table4, (/2,noa-1/), (/-1,noa/), (/1,nua-1/), (/-1,nua/), noa, noa, nua, nua)
                  call sort4(l3a_excits, l3a_amps, loc_arr, idx_table4, (/4,5,1,2/), noa, noa, nua, nua, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits,&
                  !$omp l3a_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_oo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = l3a_excits(idet,1); c = l3a_excits(idet,2);
                     i = l3a_excits(idet,3); j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(j,k,b,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = l3a_excits(jdet,3);
                        ! compute < bcijk | h1a(oo) | bcljk >
                        hmatel = -h1a_oo(l,i)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table4(i,k,b,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = l3a_excits(jdet,3);
                        ! compute < bcijk | h1a(oo) | bclik >
                        hmatel = h1a_oo(l,j)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ik), (-1)
                     idx = idx_table4(i,j,b,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = l3a_excits(jdet,3);
                        ! compute < bcijk | h1a(oo) | bclij >
                        hmatel = -h1a_oo(l,k)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!!! diagram 2: A(bc) h1a(be)*l3a(ecijk)
                  ! allocate new sorting arrays
                  nloc = noa*(noa-1)*(noa-2)/6 * nua
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(noa,noa,noa,nua))
                  !!! SB: (3,4,5,1) !!!
                  call get_index_table4(idx_table4, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/1,nua-1/), noa, noa, noa, nua)
                  call sort4(l3a_excits, l3a_amps, loc_arr, idx_table4, (/3,4,5,1/), noa, noa, noa, nua, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits,&
                  !$omp l3a_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_vv,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = l3a_excits(idet,1); c = l3a_excits(idet,2);
                     i = l3a_excits(idet,3); j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(i,j,k,b)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3a_excits(jdet,2);
                        ! compute < bcijk | h1a(vv) | bfijk >
                        hmatel = h1a_vv(f,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table4(i,j,k,c)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = l3a_excits(jdet,2);
                           ! compute < bcijk | h1a(vv) | cfijk >
                           hmatel = -h1a_vv(f,b)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (3,4,5,2) !!!
                  call get_index_table4(idx_table4, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/2,nua/), noa, noa, noa, nua)
                  call sort4(l3a_excits, l3a_amps, loc_arr, idx_table4, (/3,4,5,2/), noa, noa, noa, nua, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits,&
                  !$omp l3a_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_vv,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = l3a_excits(idet,1); c = l3a_excits(idet,2);
                     i = l3a_excits(idet,3); j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(i,j,k,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3a_excits(jdet,1);
                        ! compute < bcijk | h1a(vv) | ecijk >
                        hmatel = h1a_vv(e,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table4(i,j,k,b)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits(jdet,1);
                           ! compute < bcijk | h1a(vv) | ebijk >
                           hmatel = -h1a_vv(e,c)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!!! diagram 3: A(i/jk) 1/2 h2a(mnjk)*l3a(bcimn)
                  ! allocate new sorting arrays
                  nloc = nua*(nua - 1)/2*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,noa))
                  !!! SB: (1,2,3) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/-1,nua/), (/1,noa-2/), nua, nua, noa)
                  call sort3(l3a_excits, l3a_amps, loc_arr, idx_table3, (/1,2,3/), nua, nua, noa, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits,&
                  !$omp l3a_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_oooo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = l3a_excits(idet,1); c = l3a_excits(idet,2);
                     i = l3a_excits(idet,3); j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,i)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3a_excits(jdet,4); n = l3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | bcimn >
                        hmatel = h2a_oooo(m,n,j,k)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3a_excits(jdet,4); n = l3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | bcjmn >
                        hmatel = -h2a_oooo(m,n,i,k)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3a_excits(jdet,4); n = l3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | bckmn >
                        hmatel = h2a_oooo(m,n,i,j)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/-1,nua/), (/2,noa-1/), nua, nua, noa)
                  call sort3(l3a_excits, l3a_amps, loc_arr, idx_table3, (/1,2,4/), nua, nua, noa, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits,&
                  !$omp l3a_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_oooo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = l3a_excits(idet,1); c = l3a_excits(idet,2);
                     i = l3a_excits(idet,3); j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = l3a_excits(jdet,3); n = l3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | bcljn >
                        hmatel = h2a_oooo(l,n,i,k)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table3(b,c,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = l3a_excits(jdet,3); n = l3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | bclin >
                        hmatel = -h2a_oooo(l,n,j,k)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = l3a_excits(jdet,3); n = l3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | bclkn >
                        hmatel = -h2a_oooo(l,n,i,j)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/-1,nua/), (/3,noa/), nua, nua, noa)
                  call sort3(l3a_excits, l3a_amps, loc_arr, idx_table3, (/1,2,5/), nua, nua, noa, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits,&
                  !$omp l3a_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_oooo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = l3a_excits(idet,1); c = l3a_excits(idet,2);
                     i = l3a_excits(idet,3); j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = l3a_excits(jdet,3); m = l3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | bclmk >
                        hmatel = h2a_oooo(l,m,i,j)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = l3a_excits(jdet,3); m = l3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | bclmj >
                        hmatel = -h2a_oooo(l,m,i,k)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table3(b,c,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = l3a_excits(jdet,3); m = l3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | bclmi >
                        hmatel = h2a_oooo(l,m,j,k)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!!! diagram 4: 1/2 h2a(bcef)*l3a(efijk)
                  ! allocate new sorting arrays
                  nloc = noa*(noa - 1)*(noa - 2)/6
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(noa,noa,noa))
                  !!! SB: (3,4,5) !!!
                  call get_index_table3(idx_table3, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), noa, noa, noa)
                  call sort3(l3a_excits, l3a_amps, loc_arr, idx_table3, (/3,4,5/), noa, noa, noa, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits,&
                  !$omp l3a_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_vvvv,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = l3a_excits(idet,1); c = l3a_excits(idet,2);
                     i = l3a_excits(idet,3); j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3a_excits(jdet,1); f = l3a_excits(jdet,2);
                        ! compute < bcijk | h2a(oooo) | efijk >
                        hmatel = h2a_vvvv(e,f,b,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!!! diagram 5: A(i/jk)A(bc) h2a(bmje)*l3a(ecimk)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*(noa - 1)*(noa - 2)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(noa,noa,nua))
                  !!! [1] SB: (3,5,1) !!!
                  call get_index_table3(idx_table3, (/1,noa-2/), (/-2,noa/), (/1,nua-1/), noa, noa, nua)
                  call sort3(l3a_excits, l3a_amps, loc_arr, idx_table3, (/3,5,1/), noa, noa, nua, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits,&
                  !$omp l3a_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = l3a_excits(idet,1); c = l3a_excits(idet,2);
                     i = l3a_excits(idet,3); j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,k,b)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3a_excits(jdet,2); m = l3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | bfimk >
                        hmatel = h2a_voov(c,m,j,f)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table3(j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3a_excits(jdet,2); m = l3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | bfjmk >
                        hmatel = -h2a_voov(c,m,i,f)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(i,j,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3a_excits(jdet,2); m = l3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | bfimj >
                        hmatel = -h2a_voov(c,m,k,f)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(i,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3a_excits(jdet,2); m = l3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | cfimk >
                        hmatel = -h2a_voov(b,m,j,f)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ij)(bc)
                     idx = idx_table3(j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3a_excits(jdet,2); m = l3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | cfjmk >
                        hmatel = h2a_voov(b,m,i,f)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (jk)(bc)
                     idx = idx_table3(i,j,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3a_excits(jdet,2); m = l3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | cfimj >
                        hmatel = h2a_voov(b,m,k,f)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! [2] SB: (3,4,1) !!!
                  call get_index_table3(idx_table3, (/1,noa-2/), (/-1,noa-1/), (/1,nua-1/), noa, noa, nua)
                  call sort3(l3a_excits, l3a_amps, loc_arr, idx_table3, (/3,4,1/), noa, noa, nua, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits,&
                  !$omp l3a_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = l3a_excits(idet,1); c = l3a_excits(idet,2);
                     i = l3a_excits(idet,3); j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,j,b)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3a_excits(jdet,2); n = l3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | bfijn >
                        hmatel = h2a_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ik), (-1)
                     idx = idx_table3(j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3a_excits(jdet,2); n = l3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | bfjkn >
                        hmatel = h2a_voov(c,n,i,f)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(i,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3a_excits(jdet,2); n = l3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | bfikn >
                        hmatel = -h2a_voov(c,n,j,f)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(i,j,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3a_excits(jdet,2); n = l3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | cfijn >
                        hmatel = -h2a_voov(b,n,k,f)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ik)(bc), (-1)
                     idx = idx_table3(j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3a_excits(jdet,2); n = l3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | cfjkn >
                        hmatel = -h2a_voov(b,n,i,f)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (jk)(bc)
                     idx = idx_table3(i,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3a_excits(jdet,2); n = l3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | cfikn >
                        hmatel = h2a_voov(b,n,j,f)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! [3] SB: (4,5,1) !!!
                  call get_index_table3(idx_table3, (/2,noa-1/), (/-1,noa/), (/1,nua-1/), noa, noa, nua)
                  call sort3(l3a_excits, l3a_amps, loc_arr, idx_table3, (/4,5,1/), noa, noa, nua, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits,&
                  !$omp l3a_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = l3a_excits(idet,1); c = l3a_excits(idet,2);
                     i = l3a_excits(idet,3); j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(j,k,b)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3a_excits(jdet,2); l = l3a_excits(jdet,3);
                        ! compute < bcijk | h2a(oooo) | bfljk >
                        hmatel = h2a_voov(c,l,i,f)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table3(i,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3a_excits(jdet,2); l = l3a_excits(jdet,3);
                        ! compute < bcijk | h2a(oooo) | bflik >
                        hmatel = -h2a_voov(c,l,j,f)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ik), (-1)
                     idx = idx_table3(i,j,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3a_excits(jdet,2); l = l3a_excits(jdet,3);
                        ! compute < bcijk | h2a(oooo) | bflij >
                        hmatel = h2a_voov(c,l,k,f)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3a_excits(jdet,2); l = l3a_excits(jdet,3);
                        ! compute < bcijk | h2a(oooo) | cfljk >
                        hmatel = -h2a_voov(b,l,i,f)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ij)(bc)
                     idx = idx_table3(i,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3a_excits(jdet,2); l = l3a_excits(jdet,3);
                        ! compute < bcijk | h2a(oooo) | cflik >
                        hmatel = h2a_voov(b,l,j,f)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ik)(bc), (-1)
                     idx = idx_table3(i,j,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3a_excits(jdet,2); l = l3a_excits(jdet,3);
                        ! compute < bcijk | h2a(oooo) | cflij >
                        hmatel = -h2a_voov(b,l,k,f)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! [4] SB: (3,5,2) !!!
                  call get_index_table3(idx_table3, (/1,noa-2/), (/-2,noa/), (/2,nua/), noa, noa, nua)
                  call sort3(l3a_excits, l3a_amps, loc_arr, idx_table3, (/3,5,2/), noa, noa, nua, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits,&
                  !$omp l3a_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = l3a_excits(idet,1); c = l3a_excits(idet,2);
                     i = l3a_excits(idet,3); j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,k,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3a_excits(jdet,1); m = l3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | bfimk >
                        hmatel = h2a_voov(b,m,j,e)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table3(j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3a_excits(jdet,1); m = l3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | bfjmk >
                        hmatel = -h2a_voov(b,m,i,e)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(i,j,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3a_excits(jdet,1); m = l3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | bfimj >
                        hmatel = -h2a_voov(b,m,k,e)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(i,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3a_excits(jdet,1); m = l3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | cfimk >
                        hmatel = -h2a_voov(c,m,j,e)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ij)(bc)
                     idx = idx_table3(j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3a_excits(jdet,1); m = l3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | cfjmk >
                        hmatel = h2a_voov(c,m,i,e)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (jk)(bc)
                     idx = idx_table3(i,j,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3a_excits(jdet,1); m = l3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | cfimj >
                        hmatel = h2a_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! [5] SB: (3,4,2) !!!
                  call get_index_table3(idx_table3, (/1,noa-2/), (/-1,noa-1/), (/2,nua/), noa, noa, nua)
                  call sort3(l3a_excits, l3a_amps, loc_arr, idx_table3, (/3,4,2/), noa, noa, nua, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits,&
                  !$omp l3a_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = l3a_excits(idet,1); c = l3a_excits(idet,2);
                     i = l3a_excits(idet,3); j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,j,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3a_excits(jdet,1); n = l3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | bfijn >
                        hmatel = h2a_voov(b,n,k,e)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ik), (-1)
                     idx = idx_table3(j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3a_excits(jdet,1); n = l3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | bfjkn >
                        hmatel = h2a_voov(b,n,i,e)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(i,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3a_excits(jdet,1); n = l3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | bfikn >
                        hmatel = -h2a_voov(b,n,j,e)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(i,j,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3a_excits(jdet,1); n = l3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | cfijn >
                        hmatel = -h2a_voov(c,n,k,e)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ik)(bc), (-1)
                     idx = idx_table3(j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3a_excits(jdet,1); n = l3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | cfjkn >
                        hmatel = -h2a_voov(c,n,i,e)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (jk)(bc)
                     idx = idx_table3(i,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3a_excits(jdet,1); n = l3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | cfikn >
                        hmatel = h2a_voov(c,n,j,e)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! [6] SB: (4,5,2) !!!
                  call get_index_table3(idx_table3, (/2,noa-1/), (/-1,noa/), (/2,nua/), noa, noa, nua)
                  call sort3(l3a_excits, l3a_amps, loc_arr, idx_table3, (/4,5,2/), noa, noa, nua, nloc, n3aaa, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits,&
                  !$omp l3a_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = l3a_excits(idet,1); c = l3a_excits(idet,2);
                     i = l3a_excits(idet,3); j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(j,k,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3a_excits(jdet,1); l = l3a_excits(jdet,3);
                        ! compute < bcijk | h2a(oooo) | bfljk >
                        hmatel = h2a_voov(b,l,i,e)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table3(i,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3a_excits(jdet,1); l = l3a_excits(jdet,3);
                        ! compute < bcijk | h2a(oooo) | bflik >
                        hmatel = -h2a_voov(b,l,j,e)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ik), (-1)
                     idx = idx_table3(i,j,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3a_excits(jdet,1); l = l3a_excits(jdet,3);
                        ! compute < bcijk | h2a(oooo) | bflij >
                        hmatel = h2a_voov(b,l,k,e)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3a_excits(jdet,1); l = l3a_excits(jdet,3);
                        ! compute < bcijk | h2a(oooo) | cfljk >
                        hmatel = -h2a_voov(c,l,i,e)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ij)(bc)
                     idx = idx_table3(i,k,b)
                     if (idx/=0) then   
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3a_excits(jdet,1); l = l3a_excits(jdet,3);
                        ! compute < bcijk | h2a(oooo) | cflik >
                        hmatel = h2a_voov(c,l,j,e)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                     ! (ik)(bc), (-1)
                     idx = idx_table3(i,j,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3a_excits(jdet,1); l = l3a_excits(jdet,3);
                        ! compute < bcijk | h2a(oooo) | cflij >
                        hmatel = -h2a_voov(c,l,k,e)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!!! diagram 6: A(i/jk)A(bc) h2b(bmje)*l3b(ceikm)
                  ! copy over l3b arrays
                  allocate(excits_buff(n3aab,5),amps_buff(n3aab))
                  excits_buff(:,:) = l3b_excits(:,:)
                  amps_buff(:) = l3b_amps(:)
                  ! allocate new sorting arrays
                  nloc = noa*(noa - 1)/2*nua 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(noa,noa,nua))
                  !!! SB: (3,4,1) !!!
                  call get_index_table3(idx_table3, (/1,noa-1/), (/-1,noa/), (/1,nua/), noa, noa, nua)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/3,4,1/), noa, noa, nua, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = l3a_excits(idet,1); c = l3a_excits(idet,2);
                     i = l3a_excits(idet,3); j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,j,b)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,2); n = excits_buff(jdet,5);
                        ! compute < bcijk | h2b(voov) | bf~ijn~ >
                        hmatel = h2b_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ik), (-1)
                     idx = idx_table3(j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,2); n = excits_buff(jdet,5);
                        ! compute < bcijk | h2b(voov) | bf~jkn~ >
                        hmatel = h2b_voov(c,n,i,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(i,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,2); n = excits_buff(jdet,5);
                        ! compute < bcijk | h2b(voov) | bf~ikn~ >
                        hmatel = -h2b_voov(c,n,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(i,j,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,2); n = excits_buff(jdet,5);
                        ! compute < bcijk | h2b(voov) | cf~ijn~ >
                        hmatel = -h2b_voov(b,n,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik)(bc), (-1)
                     idx = idx_table3(j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,2); n = excits_buff(jdet,5);
                        ! compute < bcijk | h2b(voov) | cf~jkn~ >
                        hmatel = -h2b_voov(b,n,i,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)(bc)
                     idx = idx_table3(i,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,2); n = excits_buff(jdet,5);
                        ! compute < bcijk | h2b(voov) | cf~ikn~ >
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
                  !$omp l3a_excits,&
                  !$omp l1a,l2a,&
                  !$omp h1a_ov,h2a_oovv,h2a_vovv,h2a_ooov,&
                  !$omp x2a_vvo,x2a_ooo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(idet,a,b,c,d,i,j,k,l,m,n,e,f,res_mm23)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                      b = l3a_excits(idet,1); c = l3a_excits(idet,2);
                      i = l3a_excits(idet,3); j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                      ! zero out value
                      res_mm23 = 0.0d0
                      ! A(i/jk) l1a(i)*h2a_oovv(j,k,b,c)
                      res_mm23 = res_mm23 + l1a(i)*h2a_oovv(j,k,b,c) ! (1)
                      res_mm23 = res_mm23 - l1a(j)*h2a_oovv(i,k,b,c) ! (ij)
                      res_mm23 = res_mm23 - l1a(k)*h2a_oovv(j,i,b,c) ! (ik)
                      ! A(k/ij)A(bc) l2a(i,b,j)*h1a_ov(k,c)
                      res_mm23 = res_mm23 + l2a(i,b,j)*h1a_ov(k,c) ! (1)
                      res_mm23 = res_mm23 - l2a(k,b,j)*h1a_ov(i,c) ! (ik)
                      res_mm23 = res_mm23 - l2a(i,b,k)*h1a_ov(j,c) ! (jk)
                      res_mm23 = res_mm23 - l2a(i,c,j)*h1a_ov(k,b) ! (bc)
                      res_mm23 = res_mm23 + l2a(k,c,j)*h1a_ov(i,b) ! (ik)(bc)
                      res_mm23 = res_mm23 + l2a(i,c,k)*h1a_ov(j,b) ! (jk)(bc)

                      do m = 1,noa
                         ! A(k/ij)A(bc) -l2a(mck)*h2a_ooov(ijmb)
                         res_mm23 = res_mm23 - l2a(m,c,k)*h2a_ooov(i,j,m,b) ! (1)
                         res_mm23 = res_mm23 + l2a(m,c,i)*h2a_ooov(k,j,m,b) ! (ik)
                         res_mm23 = res_mm23 + l2a(m,c,j)*h2a_ooov(i,k,m,b) ! (jk)
                         res_mm23 = res_mm23 + l2a(m,b,k)*h2a_ooov(i,j,m,c) ! (bc)
                         res_mm23 = res_mm23 - l2a(m,b,i)*h2a_ooov(k,j,m,c) ! (ik)(bc)
                         res_mm23 = res_mm23 - l2a(m,b,j)*h2a_ooov(i,k,m,c) ! (jk)(bc)
                         ! A(j/ik) -x2a_ooo(ikm)*h2a_oovv(mjcb)
                         res_mm23 = res_mm23 - x2a_ooo(i,k,m)*h2a_oovv(m,j,c,b) ! (1)
                         res_mm23 = res_mm23 + x2a_ooo(j,k,m)*h2a_oovv(m,i,c,b) ! (ij)
                         res_mm23 = res_mm23 + x2a_ooo(i,j,m)*h2a_oovv(m,k,c,b) ! (ik)
                      end do

                      do e = 1,nua
                         ! A(k/ij) l2a(iej)*h2a_vovv(ekbc)
                         res_mm23 = res_mm23 + l2a(i,e,j)*h2a_vovv(e,k,b,c) ! (1)
                         res_mm23 = res_mm23 - l2a(k,e,j)*h2a_vovv(e,i,b,c) ! (ik)
                         res_mm23 = res_mm23 - l2a(i,e,k)*h2a_vovv(e,j,b,c) ! (jk)
                         ! A(k/ij)A(bc) x2a_vvo(eck)*h2a_oovv(ijeb)
                         res_mm23 = res_mm23 + x2a_vvo(e,c,k)*h2a_oovv(i,j,e,b) ! (1)
                         res_mm23 = res_mm23 - x2a_vvo(e,c,i)*h2a_oovv(k,j,e,b) ! (ik)
                         res_mm23 = res_mm23 - x2a_vvo(e,c,j)*h2a_oovv(i,k,e,b) ! (jk)
                         res_mm23 = res_mm23 - x2a_vvo(e,b,k)*h2a_oovv(i,j,e,c) ! (bc)
                         res_mm23 = res_mm23 + x2a_vvo(e,b,i)*h2a_oovv(k,j,e,c) ! (ik)(bc)
                         res_mm23 = res_mm23 + x2a_vvo(e,b,j)*h2a_oovv(i,k,e,c) ! (jk)(bc)
                      end do

                      resid(idet) = resid(idet) + res_mm23
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!

              end subroutine build_lh_3a

              subroutine build_lh_3b(resid,&
                                     l1a,l2a,l2b,&
                                     l3a_amps,l3a_excits,&
                                     l3b_amps,l3b_excits,&
                                     l3c_amps,l3c_excits,&
                                     h1a_ov,h1b_ov,h1a_oo,h1a_vv,h1b_oo,h1b_vv,&
                                     h2a_oooo,h2a_voov,h2a_ooov,h2a_oovv,&
                                     h2b_vvvv,h2b_oooo,h2b_voov,h2b_vovo,h2b_ovov,h2b_ovvo,h2b_oovv,&
                                     h2b_ooov,h2b_oovo,h2b_vovv,h2b_ovvv,&
                                     h2c_voov,&
                                     x2a_vvo,x2a_ooo,&
                                     x2b_vvo,x2b_ovv,x2b_ooo,&
                                     n3aaa,n3aab,n3abb,&
                                     noa,nua,nob,nub)

                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa, n3aab, n3abb
                  ! Input R and T arrays
                  real(kind=8), intent(in) :: l1a(noa)
                  real(kind=8), intent(in) :: l2a(noa,nua,noa)
                  real(kind=8), intent(in) :: l2b(noa,nub,nob)
                  integer, intent(in) :: l3a_excits(n3aaa,5), l3c_excits(n3abb,5)
                  real(kind=8), intent(in) :: l3a_amps(n3aaa), l3c_amps(n3abb)
                  ! Input H  and X arrays
                  real(kind=8), intent(in) :: h1a_ov(noa,nua)
                  real(kind=8), intent(in) :: h1a_oo(noa,noa)
                  real(kind=8), intent(in) :: h1a_vv(nua,nua)
                  real(kind=8), intent(in) :: h1b_ov(nob,nub)
                  real(kind=8), intent(in) :: h1b_oo(nob,nob)
                  real(kind=8), intent(in) :: h1b_vv(nub,nub)
                  real(kind=8), intent(in) :: h2a_oooo(noa,noa,noa,noa)
                  real(kind=8), intent(in) :: h2a_voov(nua,noa,noa,nua)
                  real(kind=8), intent(in) :: h2a_ooov(noa,noa,noa,nua)
                  real(kind=8), intent(in) :: h2a_oovv(noa,noa,nua,nua)
                  real(kind=8), intent(in) :: h2b_vvvv(nua,nub,nua,nub)
                  real(kind=8), intent(in) :: h2b_oooo(noa,nob,noa,nob)
                  real(kind=8), intent(in) :: h2b_voov(nua,nob,noa,nub)
                  real(kind=8), intent(in) :: h2b_vovo(nua,nob,nua,nob)
                  real(kind=8), intent(in) :: h2b_ovov(noa,nub,noa,nub)
                  real(kind=8), intent(in) :: h2b_ovvo(noa,nub,nua,nob)
                  real(kind=8), intent(in) :: h2b_oovv(noa,nob,nua,nub)
                  real(kind=8), intent(in) :: h2b_ooov(noa,nob,noa,nub)
                  real(kind=8), intent(in) :: h2b_oovo(noa,nob,nua,nob)
                  real(kind=8), intent(in) :: h2b_vovv(nua,nob,nua,nub)
                  real(kind=8), intent(in) :: h2b_ovvv(noa,nub,nua,nub)
                  real(kind=8), intent(in) :: h2c_voov(nub,nob,nob,nub)
                  real(kind=8), intent(in) :: x2a_vvo(nua,nua,noa)
                  real(kind=8), intent(in) :: x2a_ooo(noa,noa,noa)
                  real(kind=8), intent(in) :: x2b_vvo(nua,nub,nob)
                  real(kind=8), intent(in) :: x2b_ovv(noa,nub,nub)
                  real(kind=8), intent(in) :: x2b_ooo(noa,nob,nob)
                  ! Output and Inout variables
                  real(kind=8), intent(out) :: resid(n3aab)
                  integer, intent(inout) :: l3b_excits(n3aab,5)
                  !f2py intent(in,out) :: l3b_excits(0:n3aab-1,0:4)
                  real(kind=8), intent(inout) :: l3b_amps(n3aab)
                  !f2py intent(in,out) :: l3b_amps(0:n3aab-1)
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

                  ! diagram 1: -A(ij) h1a(mj)*l3b(bcimk) 
                  ! allocate new sorting arrays
                  nloc = nua*nub*(noa - 1)*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nub,noa,nob))
                  !!! SB: (1,2,3,5) !!!
                  call get_index_table4(idx_table4, (/1,nua/), (/1,nub/), (/1,noa-1/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(l3b_excits, l3b_amps, loc_arr, idx_table4, (/1,2,3,5/), nua, nub, noa, nob, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_oo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = l3b_excits(idet,1); c = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,i,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3b_excits(jdet,4);
                        ! compute < bc~ijk~ | h1a(oo) | bc~imk~ >
                        hmatel = -h1a_oo(m,j)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3b_excits(jdet,4);
                        ! compute < bc~ijk~ | h1a(oo) | bc~jmk~ >
                        hmatel = h1a_oo(m,i)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nua/), (/1,nub/), (/2,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(l3b_excits, l3b_amps, loc_arr, idx_table4, (/1,2,4,5/), nua, nub, noa, nob, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_oo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = l3b_excits(idet,1); c = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = l3b_excits(jdet,3);
                        ! compute < bc~ijk~ | h1a(oo) | bc~ljk~ >
                        hmatel = -h1a_oo(l,i)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table4(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = l3b_excits(jdet,3);
                        ! compute < bc~ijk~ | h1a(oo) | bc~lik~ >
                        hmatel = h1a_oo(l,j)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  ! diagram 2: -h1b(mk)*l3b(bcijm)
                  ! allocate new sorting arrays
                  nloc = nua*nub*noa*(noa - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(noa,noa,nua,nub))
                  !!! SB: (3,4,1,2) !!!
                  call get_index_table4(idx_table4, (/1,noa-1/), (/-1,noa/), (/1,nua/), (/1,nub/), noa, noa, nua, nub)
                  call sort4(l3b_excits, l3b_amps, loc_arr, idx_table4, (/3,4,1,2/), noa, noa, nua, nub, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1B_oo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = l3b_excits(idet,1); c = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(i,j,b,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        n = l3b_excits(jdet,5);
                        ! compute < bc~ijk~ | h1b(oo) | bc~ijn~ >
                        hmatel = -h1b_oo(n,k)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  ! diagram 3: h1a(be)*l3b(ecijk)
                  ! allocate new sorting arrays
                  nloc = nub*noa*(noa - 1)/2*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(noa,noa,nub,nob))
                  !!! SB: (3,4,2,5) !!!
                  call get_index_table4(idx_table4, (/1,noa-1/), (/-1,noa/), (/1,nub/), (/1,nob/), noa, noa, nub, nob)
                  call sort4(l3b_excits, l3b_amps, loc_arr, idx_table4, (/3,4,2,5/), noa, noa, nub, nob, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_vv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = l3b_excits(idet,1); c = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(i,j,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3b_excits(jdet,1);
                        ! compute < bc~ijk~ | h1a(vv) | ec~ijk~ >
                        hmatel = h1a_vv(e,b)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  ! diagram 4: h1b(ce)*l3b(beijk)
                  ! allocate new sorting arrays
                  nloc = noa*(noa - 1)/2*nob*nua
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(noa,noa,nob,nua))
                  !!! SB: (3,4,5,1) !!!
                  call get_index_table4(idx_table4, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/1,nua/), noa, noa, nob, nua)
                  call sort4(l3b_excits, l3b_amps, loc_arr, idx_table4, (/3,4,5,1/), noa, noa, nob, nua, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1B_vv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = l3b_excits(idet,1); c = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(i,j,k,b)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3b_excits(jdet,2);
                        ! compute < bc~ijk~ | h1b(vv) | bf~ijk~ >
                        hmatel = h1b_vv(f,c)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  ! diagram 5: A(ij) h2b(mnjk)*l3b(bcimn)
                  ! allocate new sorting arrays
                  nloc = nua*nub*(noa - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nub,noa))
                  !!! SB: (1,2,3) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/1,nub/), (/1,noa-1/), nua, nub, noa)
                  call sort3(l3b_excits, l3b_amps, loc_arr, idx_table3, (/1,2,3/), nua, nub, noa, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_oooo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = l3b_excits(idet,1); c = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,i)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3b_excits(jdet,4); n = l3b_excits(jdet,5);
                        ! compute < bc~ijk~ | h2b(oooo) | bc~imn~ >
                        hmatel = h2b_oooo(m,n,j,k)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3b_excits(jdet,4); n = l3b_excits(jdet,5);
                        ! compute < bc~ijk~ | h2b(oooo) | bc~jmn~ >
                        hmatel = -h2b_oooo(m,n,i,k)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/1,nub/), (/2,noa/), nua, nub, noa)
                  call sort3(l3b_excits, l3b_amps, loc_arr, idx_table3, (/1,2,4/), nua, nub, noa, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_oooo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = l3b_excits(idet,1); c = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = l3b_excits(jdet,3); n = l3b_excits(jdet,5);
                        ! compute < bc~ijk~ | h2b(oooo) | bc~ljn~ >
                        hmatel = h2b_oooo(l,n,i,k)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table3(b,c,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = l3b_excits(jdet,3); n = l3b_excits(jdet,5);
                        ! compute < bc~ijk~ | h2b(oooo) | bc~lin~ >
                        hmatel = -h2b_oooo(l,n,j,k)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  ! diagram 6: 1/2 h2a(mnij)*l3b(bcmnk)
                  ! allocate new sorting arrays
                  nloc = nua*nub*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nub,nob))
                  !!! SB: (1,2,5) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/1,nub/), (/1,nob/), nua, nub, nob)
                  call sort3(l3b_excits, l3b_amps, loc_arr, idx_table3, (/1,2,5/), nua, nub, nob, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_oooo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = l3b_excits(idet,1); c = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = l3b_excits(jdet,3); m = l3b_excits(jdet,4);
                        ! compute < bc~ijk~ | h2a(oooo) | bc~lmk~ >
                        hmatel = h2a_oooo(l,m,i,j)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  ! diagram 7: h2b(bcef)*l3b(efijk)
                  ! allocate new sorting arrays
                  nloc = noa*(noa - 1)/2*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(noa,noa,nob))
                  !!! SB: (3,4,5) !!!
                  call get_index_table3(idx_table3, (/1,noa-1/), (/-1,noa/), (/1,nob/), noa, noa, nob)
                  call sort3(l3b_excits, l3b_amps, loc_arr, idx_table3, (/3,4,5/), noa, noa, nob, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vvvv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = l3b_excits(idet,1); c = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3b_excits(jdet,1); f = l3b_excits(jdet,2);
                        ! compute < bc~ijk~ | h2a(oooo) | ef~ijk~ >
                        hmatel = h2b_vvvv(e,f,b,c)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  ! diagram 8: h2b(mcek)*l3a(beijm)
                  ! copy over l3a arrays
                  allocate(excits_buff(n3aaa,5),amps_buff(n3aaa))
                  excits_buff(:,:) = l3a_excits(:,:)
                  amps_buff(:) = l3a_amps(:)
                  ! allocate new sorting arrays
                  nloc = (noa - 1)*(noa - 2)/2*(nua - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(noa,noa,nua))
                  !!! SB: (3,4,1) !!!
                  call get_index_table3(idx_table3, (/1,noa-2/), (/-1,noa-1/), (/1,nua-1/), noa, noa, nua)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/3,4,1/), noa, noa, nua, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = l3b_excits(idet,1); c = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,j,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,2); n = excits_buff(jdet,5);
                        ! compute < bc~ijk~ | h2a(ovvo) | bfijn >
                        hmatel = h2b_ovvo(n,c,f,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (3,4,2) !!!
                  call get_index_table3(idx_table3, (/1,noa-2/), (/-1,noa-1/), (/2,nua/), noa, noa, nua)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/3,4,2/), noa, noa, nua, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = l3b_excits(idet,1); c = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,j,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); n = excits_buff(jdet,5);
                        ! compute < bc~ijk~ | h2a(ovvo) | ebijn >
                        hmatel = -h2b_ovvo(n,c,e,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (3,5,1) !!!
                  call get_index_table3(idx_table3, (/1,noa-2/), (/-2,noa/), (/1,nua-1/), noa, noa, nua)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/3,5,1/), noa, noa, nua, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = l3b_excits(idet,1); c = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,j,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < bc~ijk~ | h2a(ovvo) | bfimj >
                        hmatel = -h2b_ovvo(m,c,f,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (3,5,2) !!!
                  call get_index_table3(idx_table3, (/1,noa-2/), (/-2,noa/), (/2,nua/), noa, noa, nua)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/3,5,2/), noa, noa, nua, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = l3b_excits(idet,1); c = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,j,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < bc~ijk~ | h2a(ovvo) | ebimj >
                        hmatel = h2b_ovvo(m,c,e,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,1) !!!
                  call get_index_table3(idx_table3, (/2,noa-1/), (/-1,noa/), (/1,nua-1/), noa, noa, nua)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/4,5,1/), noa, noa, nua, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = l3b_excits(idet,1); c = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,j,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,2); l = excits_buff(jdet,3);
                        ! compute < bc~ijk~ | h2a(ovvo) | bflij >
                        hmatel = h2b_ovvo(l,c,f,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,2) !!!
                  call get_index_table3(idx_table3, (/2,noa-1/), (/-1,noa/), (/2,nua/), noa, noa, nua)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/4,5,2/), noa, noa, nua, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = l3b_excits(idet,1); c = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,j,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); l = excits_buff(jdet,3);
                        ! compute < bc~ijk~ | h2a(ovvo) | eblij >
                        hmatel = -h2b_ovvo(l,c,e,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3,excits_buff,amps_buff)

                  ! diagram 9: h2c(cmke)*l3b(beijm)
                  ! allocate new sorting arrays
                  nloc = noa*(noa - 1)/2*nua
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(noa,noa,nua))
                  !!! SB: (3,4,1) !!!
                  call get_index_table3(idx_table3, (/1,noa-1/), (/-1,noa/), (/1,nua/), noa, noa, nua)
                  call sort3(l3b_excits, l3b_amps, loc_arr, idx_table3, (/3,4,1/), noa, noa, nua, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = l3b_excits(idet,1); c = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,j,b)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3b_excits(jdet,2); n = l3b_excits(jdet,5);
                        ! compute < bc~ijk~ | h2c(voov) | bf~ijn~ >
                        hmatel = h2c_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  ! diagram 10: h2a(bmje)*l3b(ecimk)
                  ! allocate new sorting arrays
                  nloc = (noa - 1)*nob*nua
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(noa,nob,nua))
                  !!! SB: (3,5,2) !!!
                  call get_index_table3(idx_table3, (/1,noa-1/), (/1,nob/), (/1,nua/), noa, nob, nua)
                  call sort3(l3b_excits, l3b_amps, loc_arr, idx_table3, (/3,5,2/), noa, nob, nua, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = l3b_excits(idet,1); c = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,k,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3b_excits(jdet,1); m = l3b_excits(jdet,4);
                        ! compute < bc~ijk~ | h2c(voov) | ec~imk~ >
                        hmatel = h2a_voov(b,m,j,e)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table3(j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3b_excits(jdet,1); m = l3b_excits(jdet,4);
                        ! compute < bc~ijk~ | h2c(voov) | ec~jmk~ >
                        hmatel = -h2a_voov(b,m,i,e)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,2) !!!
                  call get_index_table3(idx_table3, (/2,noa/), (/1,nob/), (/1,nua/), noa, nob, nua)
                  call sort3(l3b_excits, l3b_amps, loc_arr, idx_table3, (/4,5,2/), noa, nob, nua, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = l3b_excits(idet,1); c = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(j,k,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3b_excits(jdet,1); l = l3b_excits(jdet,3);
                        ! compute < bc~ijk~ | h2c(voov) | ec~ljk~ >
                        hmatel = h2a_voov(b,l,i,e)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table3(i,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3b_excits(jdet,1); l = l3b_excits(jdet,3);
                        ! compute < bc~ijk~ | h2c(voov) | ec~lik~ >
                        hmatel = -h2a_voov(b,l,j,e)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  ! diagram 11: A(ij) h2b(bmje)*l3c(ecimk)
                  ! copy over l3c arrays
                  allocate(excits_buff(n3abb,5),amps_buff(n3abb))
                  excits_buff(:,:) = l3c_excits(:,:)
                  amps_buff(:) = l3c_amps(:)
                  ! allocate new sorting arrays
                  nloc = (nub - 1)*noa*(nob - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(noa,nob,nub))
                  !!! SB: (3,5,2) !!!
                  call get_index_table3(idx_table3, (/1,noa/), (/2,nob/), (/2,nub/), noa, nob, nub)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/3,5,2/), noa, nob, nub, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = l3b_excits(idet,1); c = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < bc~ijk~ | h2b(voov) | e~c~im~k~ >
                        hmatel = h2b_voov(b,m,j,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table3(j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < bc~ijk~ | h2b(voov) | e~c~jm~k~ >
                        hmatel = -h2b_voov(b,m,i,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (3,5,1) !!!
                  call get_index_table3(idx_table3, (/1,noa/), (/2,nob/), (/1,nub-1/), noa, nob, nub)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/3,5,1/), noa, nob, nub, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = l3b_excits(idet,1); c = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < bc~ijk~ | h2b(voov) | c~f~im~k~ >
                        hmatel = -h2b_voov(b,m,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table3(j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < bc~ijk~ | h2b(voov) | c~f~jm~k~ >
                        hmatel = h2b_voov(b,m,i,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (3,4,2) !!!
                  call get_index_table3(idx_table3, (/1,noa/), (/1,nob-1/), (/2,nub/), noa, nob, nub)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/3,4,2/), noa, nob, nub, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = l3b_excits(idet,1); c = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); n = excits_buff(jdet,5);
                        ! compute < bc~ijk~ | h2b(voov) | e~c~ik~n~ >
                        hmatel = -h2b_voov(b,n,j,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table3(j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); n = excits_buff(jdet,5);
                        ! compute < bc~ijk~ | h2b(voov) | e~c~jk~n~ >
                        hmatel = h2b_voov(b,n,i,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (3,4,1) !!!
                  call get_index_table3(idx_table3, (/1,noa/), (/1,nob-1/), (/1,nub-1/), noa, nob, nub)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/3,4,1/), noa, nob, nub, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = l3b_excits(idet,1); c = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,2); n = excits_buff(jdet,5);
                        ! compute < bc~ijk~ | h2b(voov) | c~f~ik~n~ >
                        hmatel = h2b_voov(b,n,j,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table3(j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,2); n = excits_buff(jdet,5);
                        ! compute < bc~ijk~ | h2b(voov) | c~f~jk~n~ >
                        hmatel = -h2b_voov(b,n,i,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3,excits_buff,amps_buff)

                  ! diagram 12: -A(ij) h2b(mcje)*l3b(beimk)
                  ! allocate new sorting arrays
                  nloc = nua*(noa - 1)*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(noa,nua,nob))
                  !!! SB: (3,1,5) !!!
                  call get_index_table3(idx_table3, (/1,noa-1/), (/1,nua/), (/1,nob/), noa, nua, nob)
                  call sort3(l3b_excits, l3b_amps, loc_arr, idx_table3, (/3,1,5/), noa, nua, nob, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = l3b_excits(idet,1); c = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,b,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3b_excits(jdet,2); m = l3b_excits(jdet,4);
                        ! compute < bc~ijk~ | h2b(ovov) | bf~imk~ >
                        hmatel = -h2b_ovov(m,c,j,f)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table3(j,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3b_excits(jdet,2); m = l3b_excits(jdet,4);
                        ! compute < bc~ijk~ | h2b(ovov) | bf~jmk~ >
                        hmatel = h2b_ovov(m,c,i,f)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!! SB: (4,1,5) !!!
                  call get_index_table3(idx_table3, (/2,noa/), (/1,nua/), (/1,nob/), noa, nua, nob)
                  call sort3(l3b_excits, l3b_amps, loc_arr, idx_table3, (/4,1,5/), noa, nua, nob, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = l3b_excits(idet,1); c = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(j,b,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3b_excits(jdet,2); l = l3b_excits(jdet,3);
                        ! compute < bc~ijk~ | h2b(ovov) | bf~ljk~ >
                        hmatel = -h2b_ovov(l,c,i,f)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table3(i,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3b_excits(jdet,2); l = l3b_excits(jdet,3);
                        ! compute < bc~ijk~ | h2b(ovov) | bf~lik~ >
                        hmatel = h2b_ovov(l,c,j,f)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  ! diagram 13: -h2b(bmek)*l3b(ecijm)
                  ! allocate new sorting arrays
                  nloc = noa*(noa - 1)/2*nub
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(noa,noa,nub))
                  !!! SB: (3,4,2) !!!
                  call get_index_table3(idx_table3, (/1,noa-1/), (/-1,noa/), (/1,nub/), noa, noa, nub)
                  call sort3(l3b_excits, l3b_amps, loc_arr, idx_table3, (/3,4,2/), noa, noa, nub, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vovo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = l3b_excits(idet,1); c = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,j,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3b_excits(jdet,1); n = l3b_excits(jdet,5);
                        ! compute < bc~ijk~ | h2b(vovo) | ec~ijn~ >
                        hmatel = -h2b_vovo(b,n,e,k)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l1a,l2a,l2b,&
                  !$omp h2a_ooov,h2b_ooov,h2b_oovo,h2b_vovv,h2b_ovvv,&
                  !$omp h1a_ov,h1b_ov,h2a_oovv,h2b_oovv,&
                  !$omp x2a_vvo,x2a_ooo,x2b_vvo,x2b_ovv,x2b_ooo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(idet,a,b,c,d,i,j,k,l,m,n,e,f,res_mm23)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                      b = l3b_excits(idet,1); c = l3b_excits(idet,2);
                      i = l3b_excits(idet,3); j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                      ! zero out value
                      res_mm23 = 0.0d0
                      ! A(ij) l1a(i)*h2b_oovv(jkbc)
                      res_mm23 = res_mm23 + l1a(i)*h2b_oovv(j,k,b,c) ! (1)
                      res_mm23 = res_mm23 - l1a(j)*h2b_oovv(i,k,b,c) ! (ij)
                      ! l2a(ibj)*h1b_ov(kc)
                      res_mm23 = res_mm23 + l2a(i,b,j)*h1b_ov(k,c)
                      ! A(ij) l2b(ick)*h1a_ov(jb)
                      res_mm23 = res_mm23 + l2b(i,c,k)*h1a_ov(j,b) ! (1)
                      res_mm23 = res_mm23 - l2b(j,c,k)*h1a_ov(i,b) ! (ij)
                      do m = 1,noa
                         ! A(ij) -l2a(mbj)*h2b_ooov(ikmc)
                         res_mm23 = res_mm23 - l2a(m,b,j)*h2b_ooov(i,k,m,c) ! (1)
                         res_mm23 = res_mm23 + l2a(m,b,i)*h2b_ooov(j,k,m,c) ! (ij)
                         ! -l2b(mck)*h2a_ooov(ijmb)
                         res_mm23 = res_mm23 - l2b(m,c,k)*h2a_ooov(i,j,m,b) ! (1)
                         ! -x2a_ooo(ijm)*h2b_oovv(mkbc)
                         res_mm23 = res_mm23 - x2a_ooo(i,j,m)*h2b_oovv(m,k,b,c) ! (1)
                      end do
                      do m = 1,nob
                         ! A(ij) -l2b(icm)*h2b_oovo(jkbm)
                         res_mm23 = res_mm23 - l2b(i,c,m)*h2b_oovo(j,k,b,m) ! (1)
                         res_mm23 = res_mm23 + l2b(j,c,m)*h2b_oovo(i,k,b,m) ! (ij)
                         ! A(ij) -x2b_ooo(ikm)*h2b_oovv(jmbc)
                         res_mm23 = res_mm23 - x2b_ooo(i,k,m)*h2b_oovv(j,m,b,c) ! (1)
                         res_mm23 = res_mm23 + x2b_ooo(j,k,m)*h2b_oovv(i,m,b,c) ! (ij)
                      end do
                      do e = 1,nua
                         ! l2a(iej)*h2b_vovv(ekbc)
                         res_mm23 = res_mm23 + l2a(i,e,j)*h2b_vovv(e,k,b,c) ! (1)
                         ! A(ij) x2a_vvo(ebj)*h2b_oovv(ikec)
                         res_mm23 = res_mm23 + x2a_vvo(e,b,j)*h2b_oovv(i,k,e,c) ! (1)
                         res_mm23 = res_mm23 - x2a_vvo(e,b,i)*h2b_oovv(j,k,e,c) ! (ij)
                         ! x2b_vvo(eck)*h2a_oovv(ijeb)
                         res_mm23 = res_mm23 + x2b_vvo(e,c,k)*h2a_oovv(i,j,e,b)
                      end do
                      do e = 1,nub
                         ! A(ij) l2b(iek)*h2b_ovvv(jebc)
                         res_mm23 = res_mm23 + l2b(i,e,k)*h2b_ovvv(j,e,b,c) ! (1)
                         res_mm23 = res_mm23 - l2b(j,e,k)*h2b_ovvv(i,e,b,c) ! (ij)
                         ! A(ij) x2b_ovv(iec)*h2b_oovv(jkbe)
                         res_mm23 = res_mm23 + x2b_ovv(i,e,c)*h2b_oovv(j,k,b,e) ! (1)
                         res_mm23 = res_mm23 - x2b_ovv(j,e,c)*h2b_oovv(i,k,b,e) ! (ij)
                      end do
                      resid(idet) = resid(idet) + res_mm23
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
              end subroutine build_lh_3b

              subroutine build_lh_3c(resid,&
                                     l1a,l2b,&
                                     l3b_amps,l3b_excits,&
                                     l3c_amps,l3c_excits,&
                                     h1b_ov,h1a_oo,h1b_oo,h1b_vv,&
                                     h2b_oooo,h2b_ovov,h2b_ovvo,h2b_ooov,h2b_oovv,&
                                     h2c_vvvv,h2c_oooo,h2c_voov,h2c_ooov,h2c_vovv,h2c_oovv,&
                                     x2b_vvo,x2b_ovv,x2b_ooo,&
                                     n3aab,n3abb,&
                                     noa,nua,nob,nub)

                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab, n3abb
                  ! Input R and T arrays
                  real(kind=8), intent(in) :: l1a(noa)
                  real(kind=8), intent(in) :: l2b(noa,nub,nob) 
                  integer, intent(in) :: l3b_excits(n3aab,5)
                  real(kind=8), intent(in) :: l3b_amps(n3aab)
                  ! Input H  and X arrays
                  real(kind=8), intent(in) :: h1b_ov(nob,nub)
                  real(kind=8), intent(in) :: h1a_oo(noa,noa)
                  real(kind=8), intent(in) :: h1b_oo(nob,nob)
                  real(kind=8), intent(in) :: h1b_vv(nub,nub)
                  real(kind=8), intent(in) :: h2b_oooo(noa,nob,noa,nob)
                  real(kind=8), intent(in) :: h2b_ovov(noa,nub,noa,nub)
                  real(kind=8), intent(in) :: h2b_ovvo(noa,nub,nua,nob)
                  real(kind=8), intent(in) :: h2b_ooov(noa,nob,noa,nub)
                  real(kind=8), intent(in) :: h2b_oovv(noa,nob,nua,nub)
                  real(kind=8), intent(in) :: h2c_vvvv(nub,nub,nub,nub)
                  real(kind=8), intent(in) :: h2c_oooo(nob,nob,nob,nob)
                  real(kind=8), intent(in) :: h2c_voov(nub,nob,nob,nub)
                  real(kind=8), intent(in) :: h2c_ooov(nob,nob,nob,nub)
                  real(kind=8), intent(in) :: h2c_vovv(nub,nob,nub,nub)
                  real(kind=8), intent(in) :: h2c_oovv(nob,nob,nub,nub)
                  real(kind=8), intent(in) :: x2b_vvo(nua,nub,nob)
                  real(kind=8), intent(in) :: x2b_ovv(noa,nub,nub)
                  real(kind=8), intent(in) :: x2b_ooo(noa,nob,nob)
                  ! Output and Inout variables
                  real(kind=8), intent(out) :: resid(n3abb)
                  integer, intent(inout) :: l3c_excits(n3abb,5)
                  !f2py intent(in,out) :: l3c_excits(0:n3abb-1,0:4)
                  real(kind=8), intent(inout) :: l3c_amps(n3abb)
                  !f2py intent(in,out) :: l3c_amps(0:n3abb-1)
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

                  ! diagram 1: -A(jk) h1b_oo(mj)*l3c(bcimk) 
                  ! allocate new sorting arrays
                  nloc = nub*(nub - 1)/2*noa*(nob - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nub,nub,noa,nob))
                  !!! SB: (1,2,3,5) !!!
                  call get_index_table4(idx_table4, (/1,nub-1/), (/-1,nub/), (/1,noa/), (/2,nob/), nub, nub, noa, nob)
                  call sort4(l3c_excits, l3c_amps, loc_arr, idx_table4, (/1,2,3,5/), nub, nub, noa, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1B_oo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = l3c_excits(idet,1); c = l3c_excits(idet,2);
                     i = l3c_excits(idet,3); j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,i,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3c_excits(jdet,4);
                        ! compute < b~c~ij~k~ | h1b(oo) | b~c~im~k~ >
                        hmatel = -h1b_oo(m,j)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table4(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3c_excits(jdet,4);
                        ! compute < b~c~ij~k~ | h1b(oo) | b~c~im~j~ >
                        hmatel = h1b_oo(m,k)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,3,4) !!!
                  call get_index_table4(idx_table4, (/1,nub-1/), (/-1,nub/), (/1,noa/), (/1,nob-1/), nub, nub, noa, nob)
                  call sort4(l3c_excits, l3c_amps, loc_arr, idx_table4, (/1,2,3,4/), nub, nub, noa, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1B_oo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = l3c_excits(idet,1); c = l3c_excits(idet,2);
                     i = l3c_excits(idet,3); j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,i,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        n = l3c_excits(jdet,5);
                        ! compute < b~c~ij~k~ | h1b(oo) | b~c~ij~n~ >
                        hmatel = -h1b_oo(n,k)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table4(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        n = l3c_excits(jdet,5);
                        ! compute < b~c~ij~k~ | h1b(oo) | b~c~ik~n~>
                        hmatel = h1b_oo(n,j)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  ! diagram 2: -h1a_oo(mi)*l3c(bcmjk) 
                  ! allocate new sorting arrays
                  nloc = nub*(nub - 1)/2*nob*(nob - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nub,nub,nob,nob))
                  !!! SB: (1,2,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nub-1/), (/-1,nub/), (/1,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(l3c_excits, l3c_amps, loc_arr, idx_table4, (/1,2,4,5/), nub, nub, nob, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_oo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = l3c_excits(idet,1); c = l3c_excits(idet,2);
                     i = l3c_excits(idet,3); j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = l3c_excits(jdet,3);
                        ! compute < b~c~ij~k~ | h1a(oo) | b~c~lj~k~ >
                        hmatel = -h1a_oo(l,i)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  ! diagram 3: A(bc) h1b(be)*l3c(ecijk)
                  ! allocate new sorting arrays
                  nloc = noa*nob*(nob - 1)/2*(nub - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nob,nob,noa,nub))
                  !!! SB: (4,5,3,2) !!!
                  call get_index_table4(idx_table4, (/1,nob-1/), (/-1,nob/), (/1,noa/), (/2,nub/), nob, nob, noa, nub)
                  call sort4(l3c_excits, l3c_amps, loc_arr, idx_table4, (/4,5,3,2/), nob, nob, noa, nub, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1B_vv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = l3c_excits(idet,1); c = l3c_excits(idet,2);
                     i = l3c_excits(idet,3); j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(j,k,i,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits(jdet,1);
                        ! compute < b~c~ij~k~ | h1b(vv) | e~c~ij~k~ >
                        hmatel = h1b_vv(e,b)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table4(j,k,i,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits(jdet,1);
                        ! compute < b~c~ij~k~ | h1b(vv) | e~b~ij~k~ >
                        hmatel = -h1b_vv(e,c)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,3,1) !!!
                  call get_index_table4(idx_table4, (/1,nob-1/), (/-1,nob/), (/1,noa/), (/1,nub-1/), nob, nob, noa, nub)
                  call sort4(l3c_excits, l3c_amps, loc_arr, idx_table4, (/4,5,3,1/), nob, nob, noa, nub, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1B_vv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = l3c_excits(idet,1); c = l3c_excits(idet,2);
                     i = l3c_excits(idet,3); j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(j,k,i,b)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits(jdet,2);
                        ! compute < b~c~ij~k~ | h1b(vv) | b~f~ij~k~ >
                        hmatel = h1b_vv(f,c)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table4(j,k,i,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits(jdet,2);
                        ! compute < b~c~ij~k~ | h1b(vv) | c~f~ij~k~ >
                        hmatel = -h1b_vv(f,b)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)
                  
                  ! diagram 4: 1/2 h2c(mnjk)*l3c(bcimn)
                  ! allocate new sorting arrays
                  nloc = nub*(nub - 1)/2*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nub,nub,noa))
                  !!! SB: (1,2,3) !!!
                  call get_index_table3(idx_table3, (/1,nub-1/), (/-1,nub/), (/1,noa/), nub, nub, noa)
                  call sort3(l3c_excits, l3c_amps, loc_arr, idx_table3, (/1,2,3/), nub, nub, noa, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2C_oooo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = l3c_excits(idet,1); c = l3c_excits(idet,2);
                     i = l3c_excits(idet,3); j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,i)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3c_excits(jdet,4); n = l3c_excits(jdet,5);
                        ! compute < b~c~ij~k~ | h2c(oooo) | b~c~im~n~ >
                        hmatel = h2c_oooo(m,n,j,k)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  ! diagram 5: A(jk) h2b(mnij)*l3c(bcmnk)
                  ! allocate new sorting arrays
                  nloc = nub*(nub - 1)/2*(nob - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nub,nub,nob))
                  !!! SB: (1,2,5) !!!
                  call get_index_table3(idx_table3, (/1,nub-1/), (/-1,nub/), (/2,nob/), nub, nub, nob)
                  call sort3(l3c_excits, l3c_amps, loc_arr, idx_table3, (/1,2,5/), nub, nub, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_oooo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = l3c_excits(idet,1); c = l3c_excits(idet,2);
                     i = l3c_excits(idet,3); j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = l3c_excits(jdet,3); m = l3c_excits(jdet,4);
                        ! compute < b~c~ij~k~ | h2b(oooo) | b~c~lm~k~ >
                        hmatel = h2b_oooo(l,m,i,j)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = l3c_excits(jdet,3); m = l3c_excits(jdet,4);
                        ! compute < b~c~ij~k~ | h2b(oooo) | b~c~lm~j~ >
                        hmatel = -h2b_oooo(l,m,i,k)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nub-1/), (/-1,nub/), (/1,nob-1/), nub, nub, nob)
                  call sort3(l3c_excits, l3c_amps, loc_arr, idx_table3, (/1,2,4/), nub, nub, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_oooo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = l3c_excits(idet,1); c = l3c_excits(idet,2);
                     i = l3c_excits(idet,3); j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = l3c_excits(jdet,3); n = l3c_excits(jdet,5);
                        ! compute < b~c~ij~k~ | h2b(oooo) | b~c~lj~n~ >
                        hmatel = h2b_oooo(l,n,i,k)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = l3c_excits(jdet,3); n = l3c_excits(jdet,5);
                        ! compute < b~c~ij~k~ | h2b(oooo) | b~c~lk~n~ >
                        hmatel = -h2b_oooo(l,n,i,j)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  ! diagram 6: 1/2 h2c(bcef)*l3c(efijk)
                  ! allocate new sorting arrays
                  nloc = nob*(nob - 1)/2*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nob,nob,noa))
                  !!! SB: (4,5,3) !!!
                  call get_index_table3(idx_table3, (/1,nob-1/), (/-1,nob/), (/1,noa/), nob, nob, noa)
                  call sort3(l3c_excits, l3c_amps, loc_arr, idx_table3, (/4,5,3/), nob, nob, noa, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2C_vvvv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = l3c_excits(idet,1); c = l3c_excits(idet,2);
                     i = l3c_excits(idet,3); j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(j,k,i)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits(jdet,1); f = l3c_excits(jdet,2);
                        ! compute < b~c~ij~k~ | h2c(vvvv) | e~f~ij~k~ >
                        !hmatel = h2c_vvvv(b,c,e,f)
                        hmatel = h2c_vvvv(e,f,b,c)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  ! diagram 7: A(jk)A(bc) h2b(mbej)*l3b(ecimk)
                  ! copy over l3b arrays
                  allocate(excits_buff(n3aab,5),amps_buff(n3aab))
                  excits_buff(:,:) = l3b_excits(:,:)
                  amps_buff(:) = l3b_amps(:)
                  ! allocate new sorting arrays
                  nloc = nub*(noa - 1)*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nub,noa,nob))
                  !!! SB: (2,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nub/), (/1,noa-1/), (/1,nob/), nub, noa, nob)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/2,3,5/), nub, noa, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = l3c_excits(idet,1); c = l3c_excits(idet,2);
                     i = l3c_excits(idet,3); j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < b~c~ij~k~ | h2b(ovvo) | ec~imk~ >
                        hmatel = h2b_ovvo(m,b,e,j)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < b~c~ij~k~ | h2b(ovvo) | eb~imk~ >
                        hmatel = -h2b_ovvo(m,c,e,j)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < b~c~ij~k~ | h2b(ovvo) | ec~imj~ >
                        hmatel = -h2b_ovvo(m,b,e,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < b~c~ij~k~ | h2b(ovvo) | eb~imj~ >
                        hmatel = h2b_ovvo(m,c,e,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,4,5) !!!
                  call get_index_table3(idx_table3, (/1,nub/), (/2,noa/), (/1,nob/), nub, noa, nob)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/2,4,5/), nub, noa, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = l3c_excits(idet,1); c = l3c_excits(idet,2);
                     i = l3c_excits(idet,3); j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); l = excits_buff(jdet,3);
                        ! compute < b~c~ij~k~ | h2b(ovvo) | ec~lik~ >
                        hmatel = -h2b_ovvo(l,b,e,j)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); l = excits_buff(jdet,3);
                        ! compute < b~c~ij~k~ | h2b(ovvo) | eb~lik~ >
                        hmatel = h2b_ovvo(l,c,e,j)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); l = excits_buff(jdet,3);
                        ! compute < b~c~ij~k~ | h2b(ovvo) | ec~lij~ >
                        hmatel = h2b_ovvo(l,b,e,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); l = excits_buff(jdet,3);
                        ! compute < b~c~ij~k~ | h2b(ovvo) | eb~lij~ >
                        hmatel = -h2b_ovvo(l,c,e,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3,excits_buff,amps_buff)

                  ! diagram 8: A(jk)A(bc) h2c(bmje)*l3c(ecimk)
                  ! allocate new sorting arrays
                  nloc = (nob - 1)*noa*(nub - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nub,nob,noa))
                  !!! SB: (2,5,3) !!!
                  call get_index_table3(idx_table3, (/2,nub/), (/2,nob/), (/1,noa/), nub, nob, noa)
                  call sort3(l3c_excits, l3c_amps, loc_arr, idx_table3, (/2,5,3/), nub, nob, noa, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = l3c_excits(idet,1); c = l3c_excits(idet,2);
                     i = l3c_excits(idet,3); j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(c,k,i)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits(jdet,1); m = l3c_excits(jdet,4);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~c~im~k~ >
                        hmatel = h2c_voov(b,m,j,e)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table3(c,j,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits(jdet,1); m = l3c_excits(jdet,4);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~c~im~j~ >
                        hmatel = -h2c_voov(b,m,k,e)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(b,k,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits(jdet,1); m = l3c_excits(jdet,4);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~b~im~k~ >
                        hmatel = -h2c_voov(c,m,j,e)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                     ! (jk)(bc)
                     idx = idx_table3(b,j,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits(jdet,1); m = l3c_excits(jdet,4);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~b~im~j~ >
                        hmatel = h2c_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,5,3) !!!
                  call get_index_table3(idx_table3, (/1,nub-1/), (/2,nob/), (/1,noa/), nub, nob, noa)
                  call sort3(l3c_excits, l3c_amps, loc_arr, idx_table3, (/1,5,3/), nub, nob, noa, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = l3c_excits(idet,1); c = l3c_excits(idet,2);
                     i = l3c_excits(idet,3); j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,k,i)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits(jdet,2); m = l3c_excits(jdet,4);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~c~im~k~ >
                        hmatel = h2c_voov(c,m,j,f)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table3(b,j,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits(jdet,2); m = l3c_excits(jdet,4);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~c~im~j~ >
                        hmatel = -h2c_voov(c,m,k,f)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(c,k,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits(jdet,2); m = l3c_excits(jdet,4);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~b~im~k~ >
                        hmatel = -h2c_voov(b,m,j,f)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                     ! (jk)(bc)
                     idx = idx_table3(c,j,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits(jdet,2); m = l3c_excits(jdet,4);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~b~im~j~ >
                        hmatel = h2c_voov(b,m,k,f)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,4,3) !!!
                  call get_index_table3(idx_table3, (/2,nub/), (/1,nob-1/), (/1,noa/), nub, nob, noa)
                  call sort3(l3c_excits, l3c_amps, loc_arr, idx_table3, (/2,4,3/), nub, nob, noa, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = l3c_excits(idet,1); c = l3c_excits(idet,2);
                     i = l3c_excits(idet,3); j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(c,j,i)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits(jdet,1); n = l3c_excits(jdet,5);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~c~im~k~ >
                        hmatel = h2c_voov(b,n,k,e)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table3(c,k,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits(jdet,1); n = l3c_excits(jdet,5);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~c~im~j~ >
                        hmatel = -h2c_voov(b,n,j,e)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(b,j,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits(jdet,1); n = l3c_excits(jdet,5);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~b~im~k~ >
                        hmatel = -h2c_voov(c,n,k,e)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                     ! (jk)(bc)
                     idx = idx_table3(b,k,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits(jdet,1); n = l3c_excits(jdet,5);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~b~im~j~ >
                        hmatel = h2c_voov(c,n,j,e)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,4,3) !!!
                  call get_index_table3(idx_table3, (/1,nub-1/), (/1,nob-1/), (/1,noa/), nub, nob, noa)
                  call sort3(l3c_excits, l3c_amps, loc_arr, idx_table3, (/1,4,3/), nub, nob, noa, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = l3c_excits(idet,1); c = l3c_excits(idet,2);
                     i = l3c_excits(idet,3); j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,j,i)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits(jdet,2); n = l3c_excits(jdet,5);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~c~im~k~ >
                        hmatel = h2c_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table3(b,k,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits(jdet,2); n = l3c_excits(jdet,5);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~c~im~j~ >
                        hmatel = -h2c_voov(c,n,j,f)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(c,j,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits(jdet,2); n = l3c_excits(jdet,5);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~b~im~k~ >
                        hmatel = -h2c_voov(b,n,k,f)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                     ! (jk)(bc)
                     idx = idx_table3(c,k,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits(jdet,2); n = l3c_excits(jdet,5);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~b~im~j~ >
                        hmatel = h2c_voov(b,n,j,f)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  ! diagram 9: -A(bc) h2b(mbie)*l3c(ecmjk)
                  ! allocate new sorting arrays
                  nloc = nob*(nob - 1)/2*(nub - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nob,nob,nub))
                  !!! SB: (4,5,2) !!!
                  call get_index_table3(idx_table3, (/1,nob/), (/-1,nob/), (/2,nub/), nob, nob, nub)
                  call sort3(l3c_excits, l3c_amps, loc_arr, idx_table3, (/4,5,2/), nob, nob, nub, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = l3c_excits(idet,1); c = l3c_excits(idet,2);
                     i = l3c_excits(idet,3); j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(j,k,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits(jdet,1); l = l3c_excits(jdet,3);
                        ! compute < b~c~ij~k~ | h2b(ovov) | e~c~lj~k~ >
                        hmatel = -h2b_ovov(l,b,i,e)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table3(j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits(jdet,1); l = l3c_excits(jdet,3);
                        ! compute < b~c~ij~k~ | h2b(ovov) | e~b~lj~k~ >
                        hmatel = h2b_ovov(l,c,i,e)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,1) !!!
                  call get_index_table3(idx_table3, (/1,nob/), (/-1,nob/), (/1,nub-1/), nob, nob, nub)
                  call sort3(l3c_excits, l3c_amps, loc_arr, idx_table3, (/4,5,1/), nob, nob, nub, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = l3c_excits(idet,1); c = l3c_excits(idet,2);
                     i = l3c_excits(idet,3); j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(j,k,b)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits(jdet,2); l = l3c_excits(jdet,3);
                        ! compute < b~c~ij~k~ | h2b(ovov) | b~f~lj~k~ >
                        hmatel = -h2b_ovov(l,c,i,f)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table3(j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits(jdet,2); l = l3c_excits(jdet,3);
                        ! compute < b~c~ij~k~ | h2b(ovov) | c~f~lj~k~ >
                        hmatel = h2b_ovov(l,b,i,f)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
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
                  !$omp l3c_excits,&
                  !$omp l1a,l2b,&
                  !$omp h2b_ooov,h2b_oovv,&
                  !$omp h2c_ooov,h2c_vovv,h2c_oovv,&
                  !$omp x2b_vvo,x2b_ovv,x2b_ooo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(idet,a,b,c,d,i,j,k,l,m,n,e,f,res_mm23)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                      b = l3c_excits(idet,1); c = l3c_excits(idet,2);
                      i = l3c_excits(idet,3); j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                      ! zero out value
                      res_mm23 = 0.0d0
                      ! l1a(i)*h2c_oovv(jkbc)
                      res_mm23 = res_mm23 + l1a(i)*h2c_oovv(j,k,b,c) ! (1)
                      ! A(bc)A(jk) l2b(abj)*h1b_ov(kc)
                      res_mm23 = res_mm23 + l2b(i,b,j)*h1b_ov(k,c) ! (1)
                      res_mm23 = res_mm23 - l2b(i,b,k)*h1b_ov(j,c) ! (jk)
                      res_mm23 = res_mm23 - l2b(i,c,j)*h1b_ov(k,b) ! (bc)
                      res_mm23 = res_mm23 + l2b(i,c,k)*h1b_ov(j,b) ! (jk)(bc)
                      do m = 1,noa
                         ! A(jk)A(bc) -l2b(mck)*h2b_ooov(ijmb)
                         res_mm23 = res_mm23 - l2b(m,c,k) * h2b_ooov(i,j,m,b) ! (1)
                         res_mm23 = res_mm23 + l2b(m,c,j) * h2b_ooov(i,k,m,b) ! (jk)
                         res_mm23 = res_mm23 + l2b(m,b,k) * h2b_ooov(i,j,m,c) ! (bc)
                         res_mm23 = res_mm23 - l2b(m,b,j) * h2b_ooov(i,k,m,c) ! (jk)(bc)
                      end do
                      do m = 1,nob
                         ! A(bc) -l2b(ibm)*h2c_ooov(jkmc)
                         res_mm23 = res_mm23 - l2b(i,b,m)*h2c_ooov(j,k,m,c) ! (1)
                         res_mm23 = res_mm23 + l2b(i,c,m)*h2c_ooov(j,k,m,b) ! (bc)
                         ! A(jk) -x2b_ooo(ijm)*h2b_oovv(mkbc)
                         res_mm23 = res_mm23 - x2b_ooo(i,j,m)*h2c_oovv(m,k,b,c) ! (1)
                         res_mm23 = res_mm23 + x2b_ooo(i,k,m)*h2c_oovv(m,j,b,c) ! (jk)
                      end do
                      do e = 1,nua
                         ! A(jk)A(bc) x2b_vvo(eck)*h2b_oovv(ijeb)
                         res_mm23 = res_mm23 + x2b_vvo(e,c,k)*h2b_oovv(i,j,e,b) ! (1)
                         res_mm23 = res_mm23 - x2b_vvo(e,b,k)*h2b_oovv(i,j,e,c) ! (bc)
                         res_mm23 = res_mm23 - x2b_vvo(e,c,j)*h2b_oovv(i,k,e,b) ! (jk)
                         res_mm23 = res_mm23 + x2b_vvo(e,b,j)*h2b_oovv(i,k,e,c) ! (bc)(jk)
                      end do
                      do e = 1,nub
                         ! A(jk) l2b(iej)*h2c_vovv(ekbc)
                         res_mm23 = res_mm23 + l2b(i,e,j)*h2c_vovv(e,k,b,c) ! (1)
                         res_mm23 = res_mm23 - l2b(i,e,k)*h2c_vovv(e,j,b,c) ! (jk)
                         ! A(bc) x2b_ovv(iec)*h2c_oovv(jkbe)
                         res_mm23 = res_mm23 + x2b_ovv(i,e,c)*h2c_oovv(j,k,b,e) ! (1)
                         res_mm23 = res_mm23 - x2b_ovv(i,e,b)*h2c_oovv(j,k,c,e) ! (bc)
                      end do
                      resid(idet) = resid(idet) + res_mm23
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
              end subroutine build_lh_3c

              subroutine update_l(l1a,l2a,l2b,&
                                  l3a_amps, l3a_excits,&
                                  l3b_amps, l3b_excits,&
                                  l3c_amps, l3c_excits,&
                                  omega,&
                                  h1a_oo, h1a_vv, h1b_oo, h1b_vv,&
                                  n3aaa, n3aab, n3abb,&
                                  noa, nua, nob, nub)

                      integer, intent(in) :: noa, nua, nob, nub, n3aaa, n3aab, n3abb
                      integer, intent(in) :: l3a_excits(n3aaa,5), l3b_excits(n3aab,5), l3c_excits(n3abb,5)
                      real(kind=8), intent(in) :: h1a_oo(noa,noa), h1a_vv(nua,nua), h1b_oo(nob,nob), h1b_vv(nub,nub)
                      real(kind=8), intent(in) :: omega
                      
                      real(kind=8), intent(inout) :: l1a(1:noa)
                      !f2py intent(in,out) :: l1a(0:noa-1)
                      real(kind=8), intent(inout) :: l2a(1:noa,1:nua,1:noa)
                      !f2py intent(in,out) :: l2a(0:noa-1,0:nua-1,0:noa-1)
                      real(kind=8), intent(inout) :: l2b(1:noa,1:nub,1:nob)
                      !f2py intent(in,out) :: l2b(0:noa-1,0:nub-1,0:nob-1)
                      real(kind=8), intent(inout) :: l3a_amps(n3aaa)
                      !f2py intent(in,out) :: l3a_amps(0:n3aaa-1) 
                      real(kind=8), intent(inout) :: l3b_amps(n3aab)
                      !f2py intent(in,out) :: l3b_amps(0:n3aab-1) 
                      real(kind=8), intent(inout) :: l3c_amps(n3abb)
                      !f2py intent(in,out) :: l3c_amps(0:n3abb-1) 
                      
                      integer :: j, k, i, b, c, idet
                      real(kind=8) :: denom

                      do i = 1,noa
                          denom = omega + H1A_oo(i,i)
                          ! If you use CIS-type guess, then omega == -H1A_oo(i,i)
                          ! exactly, leading to NaNs
                          if (denom==0.0d0) cycle
                          l1a(i) = l1a(i)/denom
                      end do

                      do j = 1,noa
                        do b = 1,nua
                           do i = 1,noa
                              if (i==j) cycle
                              denom = omega - H1A_vv(b,b) + H1A_oo(i,i) + H1A_oo(j,j)
                              l2a(i,b,j) = l2a(i,b,j)/denom
                          end do
                        end do
                      end do

                      do j = 1,nob
                        do b = 1,nub
                           do i = 1,noa
                              denom = omega - H1B_vv(b,b) + H1A_oo(i,i) + H1B_oo(j,j)
                              l2b(i,b,j) = l2b(i,b,j)/denom
                          end do
                        end do
                      end do

                      do idet = 1, n3aaa
                         b = l3a_excits(idet,1); c = l3a_excits(idet,2);
                         i = l3a_excits(idet,3); j = l3a_excits(idet,4); k = l3a_excits(idet,5);

                         denom = omega - H1A_vv(b,b) - H1A_vv(c,c) + H1A_oo(i,i) + H1A_oo(j,j) + H1A_oo(k,k)
                         
                         l3a_amps(idet) = l3a_amps(idet)/denom
                      end do

                      do idet = 1, n3aab
                         b = l3b_excits(idet,1); c = l3b_excits(idet,2);
                         i = l3b_excits(idet,3); j = l3b_excits(idet,4); k = l3b_excits(idet,5);

                         denom = omega - H1A_vv(b,b) - H1B_vv(c,c) + H1A_oo(i,i) + H1A_oo(j,j) + H1B_oo(k,k)
                         
                         l3b_amps(idet) = l3b_amps(idet)/denom
                      end do

                      do idet = 1, n3abb
                         b = l3c_excits(idet,1); c = l3c_excits(idet,2);
                         i = l3c_excits(idet,3); j = l3c_excits(idet,4); k = l3c_excits(idet,5);

                         denom = omega - H1B_vv(b,b) - H1B_vv(c,c) + H1A_oo(i,i) + H1B_oo(j,j) + H1B_oo(k,k)
                         
                         l3c_amps(idet) = l3c_amps(idet)/denom
                      end do

              end subroutine update_l

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

end module leftipeom3_p_loops
