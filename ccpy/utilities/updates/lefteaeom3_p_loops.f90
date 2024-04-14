module lefteaeom3_p_loops

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
                  real(kind=8), intent(inout) :: sigma_2a(nua,nua,noa)
                  !f2py intent(in,out) :: sigma_2a(0:nua-1,0:nua-1,0:noa-1)
                  ! Local variables
                  real(kind=8) :: l_amp, val
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  do idet = 1,n3aaa
                     l_amp = l3a_amps(idet)
                     ! A(b/ef)A(jn) h2a(fena)*l3a(ebfjn)
                     e = l3a_excits(idet,1); b = l3a_excits(idet,2); f = l3a_excits(idet,3);
                     j = l3a_excits(idet,4); n = l3a_excits(idet,5);
                     sigma_2a(:,b,j) = sigma_2a(:,b,j) + h2a_vvov(f,e,n,:)*l_amp ! (1)
                     sigma_2a(:,e,j) = sigma_2a(:,e,j) - h2a_vvov(f,b,n,:)*l_amp ! (be)
                     sigma_2a(:,f,j) = sigma_2a(:,f,j) - h2a_vvov(b,e,n,:)*l_amp ! (bf)
                     sigma_2a(:,b,n) = sigma_2a(:,b,n) - h2a_vvov(f,e,j,:)*l_amp ! (jn)
                     sigma_2a(:,e,n) = sigma_2a(:,e,n) + h2a_vvov(f,b,j,:)*l_amp ! (be)(jn)
                     sigma_2a(:,f,n) = sigma_2a(:,f,n) + h2a_vvov(b,e,j,:)*l_amp ! (bf)(jn)
                     ! A(f/ab) -h2a_vooo(fjnm)*l3a(abfmn)
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); f = l3a_excits(idet,3);
                     m = l3a_excits(idet,4); n = l3a_excits(idet,5);
                     sigma_2a(a,b,:) = sigma_2a(a,b,:) - h2a_vooo(f,:,n,m)*l_amp ! (1)
                     sigma_2a(f,b,:) = sigma_2a(f,b,:) + h2a_vooo(a,:,n,m)*l_amp ! (af)
                     sigma_2a(a,f,:) = sigma_2a(a,f,:) + h2a_vooo(b,:,n,m)*l_amp ! (bf)
                  end do                 

                  do idet = 1,n3aab
                     l_amp = l3b_amps(idet)
                     ! A(be) h2b_vvvo(efan)*l3b(ebfjn)
                     e = l3b_excits(idet,1); b = l3b_excits(idet,2); f = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); n = l3b_excits(idet,5);
                     sigma_2a(:,b,j) = sigma_2a(:,b,j) + h2b_vvvo(e,f,:,n)*l_amp ! (1)
                     sigma_2a(:,e,j) = sigma_2a(:,e,j) - h2b_vvvo(b,f,:,n)*l_amp ! (be)
                     ! -h2b_ovoo(jfmn)*l3b(abfmn)
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); f = l3b_excits(idet,3);
                     m = l3b_excits(idet,4); n = l3b_excits(idet,5);
                     sigma_2a(a,b,:) = sigma_2a(a,b,:) - h2b_ovoo(:,f,m,n)*l_amp ! (1)
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
              end subroutine build_lh_2a

              subroutine build_lh_2b(sigma_2b,&
                                     l3b_amps,l3b_excits,&
                                     l3c_amps,l3c_excits,&
                                     h2a_vvov,&
                                     h2b_vooo,h2b_vvov,h2b_vvvo,&
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
                  real(kind=8), intent(in) :: h2a_vvov(nua,nua,noa,nua)
                  real(kind=8), intent(in) :: h2b_vooo(nua,nob,noa,nob)
                  real(kind=8), intent(in) :: h2b_vvov(nua,nub,noa,nub)
                  real(kind=8), intent(in) :: h2b_vvvo(nua,nub,nua,nob)
                  real(kind=8), intent(in) :: h2c_vooo(nub,nob,nob,nob)
                  real(kind=8), intent(in) :: h2c_vvov(nub,nub,nob,nub)
                  ! Output and Inout variables
                  real(kind=8), intent(inout) :: sigma_2b(nua,nub,nob)
                  !f2py intent(in,out) :: sigma_2b(0:nua-1,0:nub-1,0:nob-1)
                  ! Local variables
                  real(kind=8) :: l_amp, val
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  do idet = 1,n3aab
                     l_amp = l3b_amps(idet)
                     ! h2a_vvov(fena)*l3b(efbnj)
                     e = l3b_excits(idet,1); f = l3b_excits(idet,2); b = l3b_excits(idet,3);
                     n = l3b_excits(idet,4); j = l3b_excits(idet,5);
                     sigma_2b(:,b,j) = sigma_2b(:,b,j) + h2a_vvov(f,e,n,:)*l_amp ! (1)
                     ! A(af) h2b_vvov(fenb)*l3b(afenj)
                     a = l3b_excits(idet,1); f = l3b_excits(idet,2); e = l3b_excits(idet,3);
                     n = l3b_excits(idet,4); j = l3b_excits(idet,5);
                     sigma_2b(a,:,j) = sigma_2b(a,:,j) + h2b_vvov(f,e,n,:)*l_amp ! (1)
                     sigma_2b(f,:,j) = sigma_2b(f,:,j) - h2b_vvov(a,e,n,:)*l_amp ! (af)
                     ! A(af) -h2b_vooo(fjnm)*l3b(afbnm)
                     a = l3b_excits(idet,1); f = l3b_excits(idet,2); b = l3b_excits(idet,3);
                     n = l3b_excits(idet,4); m = l3b_excits(idet,5);
                     sigma_2b(a,b,:) = sigma_2b(a,b,:) - h2b_vooo(f,:,n,m)*l_amp ! (1)
                     sigma_2b(f,b,:) = sigma_2b(f,b,:) + h2b_vooo(a,:,n,m)*l_amp ! (af)
                  end do                 

                  do idet = 1,n3abb
                     l_amp = l3c_amps(idet)
                     ! A(bf)A(jn) h2b_vvvo(efan)*l3c(efbnj) 
                     e = l3c_excits(idet,1); f = l3c_excits(idet,2); b = l3c_excits(idet,3);
                     n = l3c_excits(idet,4); j = l3c_excits(idet,5);
                     sigma_2b(:,b,j) = sigma_2b(:,b,j) + h2b_vvvo(e,f,:,n)*l_amp ! (1)
                     sigma_2b(:,f,j) = sigma_2b(:,f,j) - h2b_vvvo(e,b,:,n)*l_amp ! (bf)
                     sigma_2b(:,b,n) = sigma_2b(:,b,n) - h2b_vvvo(e,f,:,j)*l_amp ! (jn)
                     sigma_2b(:,f,n) = sigma_2b(:,f,n) + h2b_vvvo(e,b,:,j)*l_amp ! (bf)(jn)
                     ! A(jn) h2c_vvov(fenb)*l3c(afenj)
                     a = l3c_excits(idet,1); f = l3c_excits(idet,2); e = l3c_excits(idet,3);
                     n = l3c_excits(idet,4); j = l3c_excits(idet,5);
                     sigma_2b(a,:,j) = sigma_2b(a,:,j) + h2c_vvov(f,e,n,:)*l_amp ! (1)
                     sigma_2b(a,:,n) = sigma_2b(a,:,n) - h2c_vvov(f,e,j,:)*l_amp ! (jn)
                     ! A(bf) -h2c_vooo(fjnm)*l3c(abfmn)
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); f = l3c_excits(idet,3);
                     m = l3c_excits(idet,4); n = l3c_excits(idet,5);
                     sigma_2b(a,b,:) = sigma_2b(a,b,:) - h2c_vooo(f,:,n,m)*l_amp ! (1)
                     sigma_2b(a,f,:) = sigma_2b(a,f,:) + h2c_vooo(b,:,n,m)*l_amp ! (bf)
                  end do                  
              end subroutine build_lh_2b

              subroutine build_lh_3a(resid,&
                                     l1a,l2a,&
                                     l3a_amps,l3a_excits,&
                                     l3b_amps,l3b_excits,&
                                     h1a_ov,h1a_oo,h1a_vv,&
                                     h2a_vvvv,h2a_oooo,h2a_voov,h2a_ooov,h2a_vovv,h2a_oovv,&
                                     h2b_voov,&
                                     x2a_ovo,x2a_vvv,&
                                     n3aaa,n3aab,&
                                     noa,nua,nob,nub)

                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa, n3aab
                  ! Input L arrays
                  real(kind=8), intent(in) :: l1a(nua)
                  real(kind=8), intent(in) :: l2a(nua,nua,noa)
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
                  real(kind=8), intent(in) :: x2a_ovo(noa,nua,noa)
                  real(kind=8), intent(in) :: x2a_vvv(nua,nua,nua)
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

                  !!!! diagram 1: -A(jk) h1a(mj)*l3a(abcmk)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)*(nua-2)/6 * noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nua,nua,noa))
                  !!! SB: (1,2,3,4) !!!
                  call get_index_table4(idx_table4, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/1,noa-1/), nua, nua, nua, noa)
                  call sort4(l3a_excits, l3a_amps, loc_arr, idx_table4, (/1,2,3,4/), nua, nua, nua, noa, nloc, n3aaa, resid)
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
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(a,b,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3a_excits(jdet,5);
                        ! compute < abcjk | h1a(oo) | abcjm >
                        hmatel = -h1a_oo(k,m)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table4(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           m = l3a_excits(jdet,5);
                           ! compute < abcjk | h1a(oo) | abckm >
                           hmatel = h1a_oo(j,m)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,3,5) !!!
                  call get_index_table4(idx_table4, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/2,noa/), nua, nua, nua, noa)
                  call sort4(l3a_excits, l3a_amps, loc_arr, idx_table4, (/1,2,3,5/), nua, nua, nua, noa, nloc, n3aaa, resid)
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
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(a,b,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3a_excits(jdet,4);
                        ! compute < abcjk | h1a(oo) | abcmk >
                        hmatel = -h1a_oo(j,m)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table4(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           m = l3a_excits(jdet,4);
                           ! compute < abcjk | h1a(oo) | abcmj >
                           hmatel = h1a_oo(k,m)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 2: A(b/ac) h1a(be)*l3a(aecjk)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*(nua - 2)/2*noa*(noa - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nua,noa,noa))
                  !!! SB: (1,3,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nua-2/), (/-2,nua/), (/1,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(l3a_excits, l3a_amps, loc_arr, idx_table4, (/1,3,4,5/), nua, nua, noa, noa, nloc, n3aaa, resid)
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
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(a,c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3a_excits(jdet,2);
                        ! compute < abcjk | h1a(vv) | aecjk >
                        hmatel = h1a_vv(e,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits(jdet,2);
                           ! compute < abcjk | h1a(vv) | becjk >
                           hmatel = -h1a_vv(e,a)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table4(a,b,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits(jdet,2);
                           ! compute < abcjk | h1a(vv) | aebjk >
                           hmatel = -h1a_vv(e,c)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nua-2/), (/-1,nua-1/), (/1,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(l3a_excits, l3a_amps, loc_arr, idx_table4, (/1,2,4,5/), nua, nua, noa, noa, nloc, n3aaa, resid)
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
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(a,b,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3a_excits(jdet,3);
                        ! compute < abcjk | h1a(vv) | abfjk >
                        hmatel = h1a_vv(f,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = l3a_excits(jdet,3);
                           ! compute < abcjk | h1a(vv) | bcfjk >
                           hmatel = h1a_vv(f,a)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table4(a,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = l3a_excits(jdet,3);
                           ! compute < abcjk | h1a(vv) | acfjk >
                           hmatel = -h1a_vv(f,b)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4,5) !!!
                  call get_index_table4(idx_table4, (/2,nua-1/), (/-1,nua/), (/1,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(l3a_excits, l3a_amps, loc_arr, idx_table4, (/2,3,4,5/), nua, nua, noa, noa, nloc, n3aaa, resid)
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
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3a_excits(jdet,1);
                        ! compute < abcjk | h1a(vv) | dbcjk >
                        hmatel = h1a_vv(d,a)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table4(a,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits(jdet,1);
                           ! compute < abcjk | h1a(vv) | dacjk >
                           hmatel = -h1a_vv(d,b)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (ac)
                     idx = idx_table4(a,b,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits(jdet,1);
                           ! compute < abcjk | h1a(vv) | dabjk >
                           hmatel = h1a_vv(d,c)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 3: 1/2 A(c/ab) h2a(abef)*l3a(efcjk)
                  ! allocate new sorting arrays
                  nloc = (nua - 2)*noa*(noa - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,noa,noa))
                  !!! SB: (3,4,5) !!!
                  call get_index_table3(idx_table3, (/3,nua/), (/1,noa-1/), (/-1,noa/), nua, noa, noa)
                  call sort3(l3a_excits, l3a_amps, loc_arr, idx_table3, (/3,4,5/), nua, noa, noa, nloc, n3aaa, resid)
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
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3a_excits(jdet,1); e = l3a_excits(jdet,2);
                        ! compute < abcjk | h2a(vvvv) | decjk >
                        hmatel = h2a_vvvv(d,e,a,b)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table3(a,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits(jdet,1); e = l3a_excits(jdet,2);
                           ! compute < abcjk | h2a(vvvv) | deajk >
                           hmatel = h2a_vvvv(d,e,b,c)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table3(b,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits(jdet,1); e = l3a_excits(jdet,2);
                           ! compute < abcjk | h2a(vvvv) | debjk >
                           hmatel = -h2a_vvvv(d,e,a,c)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,4,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/1,noa-1/), (/-1,noa/), nua, noa, noa)
                  call sort3(l3a_excits, l3a_amps, loc_arr, idx_table3, (/1,4,5/), nua, noa, noa, nloc, n3aaa, resid)
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
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3a_excits(jdet,2); f = l3a_excits(jdet,3);
                        ! compute < abcjk | h2a(vvvv) | aefjk >
                        hmatel = h2a_vvvv(e,f,b,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(b,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits(jdet,2); f = l3a_excits(jdet,3);
                           ! compute < abcjk | h2a(vvvv) | befjk >
                           hmatel = -h2a_vvvv(e,f,a,c)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (ac)
                     idx = idx_table3(c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits(jdet,2); f = l3a_excits(jdet,3);
                           ! compute < abcjk | h2a(vvvv) | cefjk >
                           hmatel = h2a_vvvv(e,f,a,b)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,4,5)
                  call get_index_table3(idx_table3, (/2,nua-1/), (/1,noa-1/), (/-1,noa/), nua, noa, noa)
                  call sort3(l3a_excits, l3a_amps, loc_arr, idx_table3, (/2,4,5/), nua, noa, noa, nloc, n3aaa, resid)
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
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3a_excits(jdet,1); f = l3a_excits(jdet,3);
                        ! compute < abcjk | h2a(vvvv) | dbfjk >
                        hmatel = h2a_vvvv(d,f,a,c)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(a,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits(jdet,1); f = l3a_excits(jdet,3);
                           ! compute < abcjk | h2a(vvvv) | dafjk >
                           hmatel = -h2a_vvvv(d,f,b,c)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table3(c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits(jdet,1); f = l3a_excits(jdet,3);
                           ! compute < abcjk | h2a(vvvv) | dcfjk >
                           hmatel = -h2a_vvvv(d,f,a,b)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 4: 1/4 h2a(mnjk)*l3a(abcmn)
                  ! allocate new sorting arrays
                  nloc = (nua - 2)*(nua - 1)*nua/6
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,nua))
                  !!! SB: (1,2,3) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), nua, nua, nua)
                  call sort3(l3a_excits, l3a_amps, loc_arr, idx_table3, (/1,2,3/), nua, nua, nua, nloc, n3aaa, resid)
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
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,b,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3a_excits(jdet,4); n = l3a_excits(jdet,5);
                        ! compute < abcjk | h2a(oooo) | abcmn >
                        hmatel = h2a_oooo(j,k,m,n)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 5: A(c/ab)A(jk) h2a(cmke)*l3a(abejm)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*(nua - 2)/2*(noa - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,noa))
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-1,nua-1/), (/1,noa-1/), nua, nua, noa)
                  call sort3(l3a_excits, l3a_amps, loc_arr, idx_table3, (/1,2,4/), nua, nua, noa, nloc, n3aaa, resid)
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
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,b,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3a_excits(jdet,3); n = l3a_excits(jdet,5);
                        ! compute < abcjk | h2a(voov) | abfjn >
                        hmatel = h2a_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ac), (-1)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = l3a_excits(jdet,3); n = l3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | bcfjn >
                           hmatel = h2a_voov(a,n,k,f)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = l3a_excits(jdet,3); n = l3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | acfjn >
                           hmatel = -h2a_voov(b,n,k,f)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = l3a_excits(jdet,3); n = l3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | abfkn >
                           hmatel = -h2a_voov(c,n,j,f)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (ac)(jk), (-1)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = l3a_excits(jdet,3); n = l3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | bcfkn >
                           hmatel = -h2a_voov(a,n,j,f)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = l3a_excits(jdet,3); n = l3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | acfkn >
                           hmatel = h2a_voov(b,n,j,f)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-2,nua/), (/1,noa-1/), nua, nua, noa)
                  call sort3(l3a_excits, l3a_amps, loc_arr, idx_table3, (/1,3,4/), nua, nua, noa, nloc, n3aaa, resid)
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
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3a_excits(jdet,2); n = l3a_excits(jdet,5);
                        ! compute < abcjk | h2a(voov) | aecjn >
                        hmatel = h2a_voov(b,n,k,e)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits(jdet,2); n = l3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | becjn >
                           hmatel = -h2a_voov(a,n,k,e)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits(jdet,2); n = l3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | aebjn >
                           hmatel = -h2a_voov(c,n,k,e)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits(jdet,2); n = l3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | aeckn >
                           hmatel = -h2a_voov(b,n,j,e)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits(jdet,2); n = l3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | beckn >
                           hmatel = h2a_voov(a,n,j,e)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits(jdet,2); n = l3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | aebkn >
                           hmatel = h2a_voov(c,n,j,e)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4) !!!
                  call get_index_table3(idx_table3, (/2,nua-1/), (/-1,nua/), (/1,noa-1/), nua, nua, noa)
                  call sort3(l3a_excits, l3a_amps, loc_arr, idx_table3, (/2,3,4/), nua, nua, noa, nloc, n3aaa, resid)
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
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3a_excits(jdet,1); n = l3a_excits(jdet,5);
                        ! compute < abcjk | h2a(voov) | dbcjn >
                        hmatel = h2a_voov(a,n,k,d)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits(jdet,1); n = l3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | dacjn >
                           hmatel = -h2a_voov(b,n,k,d)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if 
                     ! (ac), (-1)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits(jdet,1); n = l3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | dabjn >
                           hmatel = h2a_voov(c,n,k,d)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if 
                     ! (jk)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits(jdet,1); n = l3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | dbckn >
                           hmatel = -h2a_voov(a,n,j,d)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if 
                     ! (ab)(jk)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits(jdet,1); n = l3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | dackn >
                           hmatel = h2a_voov(b,n,j,d)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if 
                     ! (ac)(jk), (-1)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits(jdet,1); n = l3a_excits(jdet,5);
                           ! compute < abcjk | h2a(voov) | dabkn >
                           hmatel = -h2a_voov(c,n,j,d)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if 
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-1,nua-1/), (/2,noa/), nua, nua, noa)
                  call sort3(l3a_excits, l3a_amps, loc_arr, idx_table3, (/1,2,5/), nua, nua, noa, nloc, n3aaa, resid)
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
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,b,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3a_excits(jdet,3); m = l3a_excits(jdet,4);
                        ! compute < abcjk | h2a(voov) | abfmk >
                        hmatel = h2a_voov(c,m,j,f)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ac), (-1)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = l3a_excits(jdet,3); m = l3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | bcfmk >
                           hmatel = h2a_voov(a,m,j,f)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = l3a_excits(jdet,3); m = l3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | acfmk >
                           hmatel = -h2a_voov(b,m,j,f)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = l3a_excits(jdet,3); m = l3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | abfmj >
                           hmatel = -h2a_voov(c,m,k,f)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (ac)(jk), (-1)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = l3a_excits(jdet,3); m = l3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | bcfmj >
                           hmatel = -h2a_voov(a,m,k,f)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = l3a_excits(jdet,3); m = l3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | acfmj >
                           hmatel = h2a_voov(b,m,k,f)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-2,nua/), (/2,noa/), nua, nua, noa)
                  call sort3(l3a_excits, l3a_amps, loc_arr, idx_table3, (/1,3,5/), nua, nua, noa, nloc, n3aaa, resid)
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
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3a_excits(jdet,2); m = l3a_excits(jdet,4);
                        ! compute < abcjk | h2a(voov) | aecmk >
                        hmatel = h2a_voov(b,m,j,e)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits(jdet,2); m = l3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | becmk >
                           hmatel = -h2a_voov(a,m,j,e)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits(jdet,2); m = l3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | aebmk >
                           hmatel = -h2a_voov(c,m,j,e)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits(jdet,2); m = l3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | aecmj >
                           hmatel = -h2a_voov(b,m,k,e)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits(jdet,2); m = l3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | becmj >
                           hmatel = h2a_voov(a,m,k,e)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits(jdet,2); m = l3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | aebmj >
                           hmatel = h2a_voov(c,m,k,e)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,5) !!!
                  call get_index_table3(idx_table3, (/2,nua-1/), (/-1,nua/), (/2,noa/), nua, nua, noa)
                  call sort3(l3a_excits, l3a_amps, loc_arr, idx_table3, (/2,3,5/), nua, nua, noa, nloc, n3aaa, resid)
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
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3a_excits(jdet,1); m = l3a_excits(jdet,4);
                        ! compute < abcjk | h2a(voov) | dbcmk >
                        hmatel = h2a_voov(a,m,j,d)
                        resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits(jdet,1); m = l3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | dacmk >
                           hmatel = -h2a_voov(b,m,j,d)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (ac), (-1)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits(jdet,1); m = l3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | dabmk >
                           hmatel = h2a_voov(c,m,j,d)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits(jdet,1); m = l3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | dbcmj >
                           hmatel = -h2a_voov(a,m,k,d)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits(jdet,1); m = l3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | dacmj >
                           hmatel = h2a_voov(b,m,k,d)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                     ! (ac)(jk), (-1)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits(jdet,1); m = l3a_excits(jdet,4);
                           ! compute < abcjk | h2a(voov) | dabmj >
                           hmatel = -h2a_voov(c,m,k,d)
                           resid(idet) = resid(idet) + hmatel * l3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 6: A(c/ab)A(jk) h2b(cmke)*l3b(abejm)
                  ! allocate and copy over l3b arrays
                  allocate(amps_buff(n3aab),excits_buff(n3aab,5))
                  amps_buff(:) = l3b_amps(:)
                  excits_buff(:,:) = l3b_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = nua*(nua - 1)/2*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,noa))
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/-1,nua/), (/1,noa/), nua, nua, noa)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,2,4/), nua, nua, noa, nloc, n3aab)
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
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     j = l3a_excits(idet,4); k = l3a_excits(idet,5);
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
                  !$omp l3a_excits,&
                  !$omp l1a,l2a,&
                  !$omp h1a_ov,h2a_oovv,h2a_vovv,h2a_ooov,&
                  !$omp x2a_ovo,x2a_vvv,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(idet,a,b,c,d,i,j,k,l,m,n,e,f,res_mm23)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                      a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                      j = l3a_excits(idet,4); k = l3a_excits(idet,5);
                      ! zero out value
                      res_mm23 = 0.0d0
                      ! A(a/bc) l1a(a)*h2a_oovv(j,k,b,c)
                      res_mm23 = res_mm23 + l1a(a)*h2a_oovv(j,k,b,c) ! (1)
                      res_mm23 = res_mm23 - l1a(b)*h2a_oovv(j,k,a,c) ! (ab)
                      res_mm23 = res_mm23 - l1a(c)*h2a_oovv(j,k,b,a) ! (ac)
                      ! A(c/ab)A(jk) l2a(a,b,j)*h1a_ov(k,c)
                      res_mm23 = res_mm23 + l2a(a,b,j)*h1a_ov(k,c) ! (1)
                      res_mm23 = res_mm23 - l2a(a,c,j)*h1a_ov(k,b) ! (bc)
                      res_mm23 = res_mm23 - l2a(c,b,j)*h1a_ov(k,a) ! (ac)
                      res_mm23 = res_mm23 - l2a(a,b,k)*h1a_ov(j,c) ! (jk)
                      res_mm23 = res_mm23 + l2a(a,c,k)*h1a_ov(j,b) ! (bc)(jk)
                      res_mm23 = res_mm23 + l2a(c,b,k)*h1a_ov(j,a) ! (ac)(jk)

                      do m = 1,noa
                         ! -A(c/ab) l2a(abm)*h2a_ooov(jkmc)
                         res_mm23 = res_mm23 - l2a(a,b,m)*h2a_ooov(j,k,m,c) ! (1)
                         res_mm23 = res_mm23 + l2a(c,b,m)*h2a_ooov(j,k,m,a) ! (ac)
                         res_mm23 = res_mm23 + l2a(a,c,m)*h2a_ooov(j,k,m,b) ! (bc)
                         ! -A(jk)A(c/ab) x2a_ovo(mck)*h2a_oovv(mjab)
                         res_mm23 = res_mm23 - x2a_ovo(m,c,k)*h2a_oovv(m,j,a,b) ! (1)
                         res_mm23 = res_mm23 + x2a_ovo(m,a,k)*h2a_oovv(m,j,c,b) ! (ac)
                         res_mm23 = res_mm23 + x2a_ovo(m,b,k)*h2a_oovv(m,j,a,c) ! (ab)
                         res_mm23 = res_mm23 + x2a_ovo(m,c,j)*h2a_oovv(m,k,a,b) ! (jk)
                         res_mm23 = res_mm23 - x2a_ovo(m,a,j)*h2a_oovv(m,k,c,b) ! (ac)(jk)
                         res_mm23 = res_mm23 - x2a_ovo(m,b,j)*h2a_oovv(m,k,a,c) ! (ab)(jk)
                      end do
                      do e = 1,nua
                         ! A(jk)A(c/ab) l2a(eck)*h2a_vovv(ejab)
                         res_mm23 = res_mm23 + l2a(e,c,k)*h2a_vovv(e,j,a,b) ! (1)
                         res_mm23 = res_mm23 - l2a(e,a,k)*h2a_vovv(e,j,c,b) ! (ac)
                         res_mm23 = res_mm23 - l2a(e,b,k)*h2a_vovv(e,j,a,c) ! (bc)
                         res_mm23 = res_mm23 - l2a(e,c,j)*h2a_vovv(e,k,a,b) ! (jk)
                         res_mm23 = res_mm23 + l2a(e,a,j)*h2a_vovv(e,k,c,b) ! (ac)(jk)
                         res_mm23 = res_mm23 + l2a(e,b,j)*h2a_vovv(e,k,a,c) ! (bc)(jk)
                         ! A(c/ab) x2a_vvv(aeb)*h2a_oovv(jkec)
                         res_mm23 = res_mm23 + x2a_vvv(a,e,b)*h2a_oovv(j,k,e,c) ! (1)
                         res_mm23 = res_mm23 - x2a_vvv(c,e,b)*h2a_oovv(j,k,e,a) ! (ac)
                         res_mm23 = res_mm23 - x2a_vvv(a,e,c)*h2a_oovv(j,k,e,b) ! (bc)
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
                                     h2a_vvvv,h2a_voov,h2a_vovv,h2a_oovv,&
                                     h2b_vvvv,h2b_oooo,h2b_voov,h2b_vovo,h2b_ovov,h2b_ovvo,h2b_oovv,&
                                     h2b_ooov,h2b_oovo,h2b_vovv,h2b_ovvv,&
                                     h2c_voov,&
                                     x2a_ovo,x2a_vvv,&
                                     x2b_voo,x2b_ovo,x2b_vvv,&
                                     n3aaa,n3aab,n3abb,&
                                     noa,nua,nob,nub)

                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa, n3aab, n3abb
                  ! Input R and T arrays
                  real(kind=8), intent(in) :: l1a(nua)
                  real(kind=8), intent(in) :: l2a(nua,nua,noa)
                  real(kind=8), intent(in) :: l2b(nua,nub,nob)
                  integer, intent(in) :: l3a_excits(n3aaa,5), l3c_excits(n3abb,5)
                  real(kind=8), intent(in) :: l3a_amps(n3aaa), l3c_amps(n3abb)
                  ! Input H  and X arrays
                  real(kind=8), intent(in) :: h1a_ov(noa,nua)
                  real(kind=8), intent(in) :: h1a_oo(noa,noa)
                  real(kind=8), intent(in) :: h1a_vv(nua,nua)
                  real(kind=8), intent(in) :: h1b_ov(nob,nub)
                  real(kind=8), intent(in) :: h1b_oo(nob,nob)
                  real(kind=8), intent(in) :: h1b_vv(nub,nub)
                  real(kind=8), intent(in) :: h2a_vvvv(nua,nua,nua,nua)
                  real(kind=8), intent(in) :: h2a_voov(nua,noa,noa,nua)
                  real(kind=8), intent(in) :: h2a_vovv(nua,noa,nua,nua)
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
                  real(kind=8), intent(in) :: x2a_ovo(noa,nua,noa)
                  real(kind=8), intent(in) :: x2a_vvv(nua,nua,nua)
                  real(kind=8), intent(in) :: x2b_voo(nua,nob,nob)
                  real(kind=8), intent(in) :: x2b_ovo(noa,nub,nob)
                  real(kind=8), intent(in) :: x2b_vvv(nua,nub,nub)
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

                  !!! diagram 1: -h1a(mj)*l3b(abcmk)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2*nub * nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nua,nub,nob))
                  !!! SB: (1,2,3,5) !!!
                  call get_index_table4(idx_table4, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/1,nob/), nua, nua, nub, nob)
                  call sort4(l3b_excits, l3b_amps, loc_arr, idx_table4, (/1,2,3,5/), nua, nua, nub, nob, nloc, n3aab, resid)
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
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(a,b,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3b_excits(jdet,4);
                        ! compute < abcjk | h1a(oo) | abcmk >
                        hmatel = -h1a_oo(j,m)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 2: -h1b(mk)*l3b(abcjm)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2*nub * noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nua,nub,noa))
                  !!! SB: (1,2,3,4) !!!
                  call get_index_table4(idx_table4, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/1,noa/), nua, nua, nub, noa)
                  call sort4(l3b_excits, l3b_amps, loc_arr, idx_table4, (/1,2,3,4/), nua, nua, nub, noa, nloc, n3aab, resid)
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
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(a,b,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        n = l3b_excits(jdet,5);
                        ! compute < abcjk | h1b(oo) | abcjn >
                        hmatel = -h1b_oo(k,n)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 3: h1b(ce)*l3b(abejk)
                  ! allocate new sorting arrays
                  nloc = nua*(nua - 1)/2 * noa * nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nua,noa,nob))
                  !!! SB: (1,2,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nua-1/), (/-1,nua/), (/1,noa/), (/1,nob/), nua, nua, noa, nob)
                  call sort4(l3b_excits, l3b_amps, loc_arr, idx_table4, (/1,2,4,5/), nua, nua, noa, nob, nloc, n3aab, resid)
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
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(a,b,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3b_excits(jdet,3);
                        ! compute < abcjk | h1b(vv) | abfjk >
                        hmatel = h1b_vv(f,c)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 4: h1a(be)*l3b(aecjk)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*nub*noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nub,noa,nob))
                  !!! SB: (1,3,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nua-1/), (/1,nub/), (/1,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(l3b_excits, l3b_amps, loc_arr, idx_table4, (/1,3,4,5/), nua, nub, noa, nob, nloc, n3aab, resid)
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
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(a,c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3b_excits(jdet,2);
                        ! compute < abcjk | h1a(vv) | aecjk >
                        hmatel = h1a_vv(e,b)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3b_excits(jdet,2);
                           ! compute < abcjk | h1a(vv) | becjk >
                           hmatel = -h1a_vv(e,a)
                           resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4,5) !!!
                  call get_index_table4(idx_table4, (/2,nua/), (/1,nub/), (/1,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(l3b_excits, l3b_amps, loc_arr, idx_table4, (/2,3,4,5/), nua, nub, noa, nob, nloc, n3aab, resid)
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
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3b_excits(jdet,1);
                        ! compute < abcjk | h1a(vv) | dbcjk >
                        hmatel = h1a_vv(d,a)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table4(a,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3b_excits(jdet,1);
                           ! compute < abcjk | h1a(vv) | dacjk >
                           hmatel = -h1a_vv(d,b)
                           resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 5: h2b(mnjk)*l3b(abcmn)
                  ! allocate new sorting arrays
                  nloc = nua*(nua - 1)/2*nub
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,nub))
                  !!! SB: (1,2,3) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/-1,nua/), (/1,nub/), nua, nua, nub)
                  call sort3(l3b_excits, l3b_amps, loc_arr, idx_table3, (/1,2,3/), nua, nua, nub, nloc, n3aab, resid)
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
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,b,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3b_excits(jdet,4); n = l3b_excits(jdet,5);
                        ! compute < abcjk | h2b(oooo) | abcmn >
                        hmatel = h2b_oooo(j,k,m,n)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
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
                  amps_buff(:) = l3a_amps(:)
                  excits_buff(:,:) = l3a_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*(nua - 2)/2*(noa - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,noa))
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-1,nua-1/), (/1,noa-1/), nua, nua, noa)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,2,4/), nua, nua, noa, nloc, n3aaa)
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
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); k = l3b_excits(idet,5);
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
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); k = l3b_excits(idet,5);
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
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); k = l3b_excits(idet,5);
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
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); k = l3b_excits(idet,5);
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
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); k = l3b_excits(idet,5);
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
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); k = l3b_excits(idet,5);
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

                  !!! diagram 7: h2c(cmke)*l3b(abejm)
                  ! allocate new sorting arrays
                  nloc = nua*(nua - 1)/2*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,noa))
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/-1,nua/), (/1,noa/), nua, nua, noa)
                  call sort3(l3b_excits, l3b_amps, loc_arr, idx_table3, (/1,2,4/), nua, nua, noa, nloc, n3aab, resid)
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
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,b,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3b_excits(jdet,3); n = l3b_excits(jdet,5);
                        ! compute < abcjk | h2c(voov) | abfjn >
                        hmatel = h2c_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 8: h2a(bmje)*l3b(aecmk)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*nub*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nub,nob))
                  !!! SB: (1,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/1,nub/), (/1,nob/), nua, nub, nob)
                  call sort3(l3b_excits, l3b_amps, loc_arr, idx_table3, (/1,3,5/), nua, nub, nob, nloc, n3aab, resid)
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
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3b_excits(jdet,2); m = l3b_excits(jdet,4);
                        ! compute < abc~jk~ | h2a(voov) | aec~mk~ >
                        hmatel = h2a_voov(b,m,j,e)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3b_excits(jdet,2); m = l3b_excits(jdet,4);
                        ! compute < abc~jk~ | h2a(voov) | bec~mk~ >
                        hmatel = -h2a_voov(a,m,j,e)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,5) !!!
                  call get_index_table3(idx_table3, (/2,nua/), (/1,nub/), (/1,nob/), nua, nub, nob)
                  call sort3(l3b_excits, l3b_amps, loc_arr, idx_table3, (/2,3,5/), nua, nub, nob, nloc, n3aab, resid)
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
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3b_excits(jdet,1); m = l3b_excits(jdet,4);
                        ! compute < abc~jk~ | h2a(voov) | dbc~mk~ >
                        hmatel = h2a_voov(a,m,j,d)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3b_excits(jdet,1); m = l3b_excits(jdet,4);
                        ! compute < abc~jk~ | h2a(voov) | dac~mk~ >
                        hmatel = -h2a_voov(b,m,j,d)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)
                
                  !!! diagram 9: -A(ab) h2b(bmek)*l3b(aecjm)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*nub*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nub,noa))
                  !!! SB: (1,3,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/1,nub/), (/1,noa/), nua, nub, noa)
                  call sort3(l3b_excits, l3b_amps, loc_arr, idx_table3, (/1,3,4/), nua, nub, noa, nloc, n3aab, resid)
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
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3b_excits(jdet,2); n = l3b_excits(jdet,5);
                        ! compute < abc~jk~ | h2b(vovo) | aec~jn~ >
                        hmatel = -h2b_vovo(b,n,e,k)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3b_excits(jdet,2); n = l3b_excits(jdet,5);
                        ! compute < abc~jk~ | h2b(vovo) | bec~jn~ >
                        hmatel = h2b_vovo(a,n,e,k)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4) !!!
                  call get_index_table3(idx_table3, (/2,nua/), (/1,nub/), (/1,noa/), nua, nub, noa)
                  call sort3(l3b_excits, l3b_amps, loc_arr, idx_table3, (/2,3,4/), nua, nub, noa, nloc, n3aab, resid)
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
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3b_excits(jdet,1); n = l3b_excits(jdet,5);
                        ! compute < abc~jk~ | h2b(vovo) | bec~jn~ >
                        hmatel = -h2b_vovo(a,n,d,k)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3b_excits(jdet,1); n = l3b_excits(jdet,5);
                        ! compute < abc~jk~ | h2b(vovo) | aec~jn~ >
                        hmatel = h2b_vovo(b,n,d,k)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 10: -h2b(mcje)*l3b(abemk)
                  ! allocate new sorting arrays
                  nloc = nua*(nua - 1)/2 * nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,nob))
                  !!! SB: (1,2,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/-1,nua/), (/1,nob/), nua, nua, nob)
                  call sort3(l3b_excits, l3b_amps, loc_arr, idx_table3, (/1,2,5/), nua, nua, nob, nloc, n3aab, resid)
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
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,b,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3b_excits(jdet,3); m = l3b_excits(jdet,4);
                        ! compute < abc~jk~ | h2b(vovo) | abf~mk~ >
                        hmatel = -h2b_ovov(m,c,j,f)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 11: A(ab) h2b(bmje)*l3c(aecmk)
                  ! allocate and copy over l3c arrays
                  allocate(amps_buff(n3abb),excits_buff(n3abb,5))
                  amps_buff(:) = l3c_amps(:)
                  excits_buff(:,:) = l3c_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = nua*(nub - 1)*(nob - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nub,nob))
                  !!! SB: (1,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/2,nub/), (/2,nob/), nua, nub, nob)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,3,5/), nua, nub, nob, nloc, n3abb)
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
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); k = l3b_excits(idet,5);
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
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); k = l3b_excits(idet,5);
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
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); k = l3b_excits(idet,5);
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
                  !$omp l3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); k = l3b_excits(idet,5);
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

                  !!! diagram 12: 1/2 h2a(abef)*l3b(efcjk)
                  ! allocate new sorting arrays
                  nloc = nub*noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nub,noa,nob))
                  !!! SB: (3,4,5) !!!
                  call get_index_table3(idx_table3, (/1,nub/), (/1,noa/), (/1,nob/), nub, noa, nob)
                  call sort3(l3b_excits, l3b_amps, loc_arr, idx_table3, (/3,4,5/), nub, noa, nob, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits,&
                  !$omp l3b_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_vvvv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3b_excits(jdet,1); e = l3b_excits(jdet,2);
                        ! compute < abc~jk~ | h2a(vvvv) | dfc~jk~ >
                        hmatel = h2a_vvvv(d,e,a,b)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 13: A(ab) h2b(bcef)*l3b(aefjk)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,noa,nob))
                  !!! SB: (1,4,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/1,noa/), (/1,nob/), nua, noa, nob)
                  call sort3(l3b_excits, l3b_amps, loc_arr, idx_table3, (/1,4,5/), nua, noa, nob, nloc, n3aab, resid)
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
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3b_excits(jdet,2); f = l3b_excits(jdet,3);
                        ! compute < abc~jk~ | h2b(vvvv) | aef~jk~ >
                        hmatel = h2b_vvvv(e,f,b,c)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3b_excits(jdet,2); f = l3b_excits(jdet,3);
                        ! compute < abc~jk~ | h2b(vvvv) | bef~jk~ >
                        hmatel = -h2b_vvvv(e,f,a,c)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,4,5) !!!
                  call get_index_table3(idx_table3, (/2,nua/), (/1,noa/), (/1,nob/), nua, noa, nob)
                  call sort3(l3b_excits, l3b_amps, loc_arr, idx_table3, (/2,4,5/), nua, noa, nob, nloc, n3aab, resid)
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
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3b_excits(jdet,1); f = l3b_excits(jdet,3);
                        ! compute < abc~jk~ | h2b(vvvv) | dbf~jk~ >
                        hmatel = h2b_vvvv(d,f,a,c)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(a,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3b_excits(jdet,1); f = l3b_excits(jdet,3);
                        ! compute < abc~jk~ | h2b(vvvv) | daf~jk~ >
                        hmatel = -h2b_vvvv(d,f,b,c)
                        resid(idet) = resid(idet) + hmatel * l3b_amps(jdet)
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
                  !$omp l3b_excits,&
                  !$omp l1a,l2a,l2b,&
                  !$omp h2a_vovv,h2b_ooov,h2b_oovo,h2b_vovv,h2b_ovvv,&
                  !$omp h1a_ov,h1b_ov,h2a_oovv,h2b_oovv,&
                  !$omp x2a_ovo,x2a_vvv,x2b_voo,x2b_ovo,x2b_vvv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(idet,a,b,c,d,i,j,k,l,m,n,e,f,res_mm23)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                      a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                      j = l3b_excits(idet,4); k = l3b_excits(idet,5);
                      ! zero out value
                      res_mm23 = 0.0d0
                      ! A(ab) l1a(a)*h2b_oovv(jkbc)
                      res_mm23 = res_mm23 + l1a(a)*h2b_oovv(j,k,b,c) ! (1)
                      res_mm23 = res_mm23 - l1a(b)*h2b_oovv(j,k,a,c) ! (ab)
                      ! l2a(abj)*h1b_ov(kc)
                      res_mm23 = res_mm23 + l2a(a,b,j)*h1b_ov(k,c) ! (1)
                      ! A(ab) l2b(ack)*h1a_ov(jb)
                      res_mm23 = res_mm23 + l2b(a,c,k)*h1a_ov(j,b) ! (1)
                      res_mm23 = res_mm23 - l2b(b,c,k)*h1a_ov(j,a) ! (ab)
                      do m = 1,noa
                         ! -l2a(abm)*h2b_ooov(jkmc)
                         res_mm23 = res_mm23 - l2a(a,b,m)*h2b_ooov(j,k,m,c)
                         ! -x2b_ovo(mck)*h2a_oovv(mjab)
                         res_mm23 = res_mm23 - x2b_ovo(m,c,k)*h2a_oovv(m,j,a,b)
                         ! -A(ab) x2a_ovo(mbj)*h2b_oovv(mkac)
                         res_mm23 = res_mm23 - x2a_ovo(m,b,j)*h2b_oovv(m,k,a,c) ! (1)
                         res_mm23 = res_mm23 + x2a_ovo(m,a,j)*h2b_oovv(m,k,b,c) ! (ab)
                      end do
                      do m = 1,nob
                         ! A(ab) -l2b(acm)*h2b_oovo(jkbm)
                         res_mm23 = res_mm23 - l2b(a,c,m)*h2b_oovo(j,k,b,m) ! (1)
                         res_mm23 = res_mm23 + l2b(b,c,m)*h2b_oovo(j,k,a,m) ! (ab)
                         ! A(ab) -x2b_voo(akm)*h2b_oovv(jmbc)
                         res_mm23 = res_mm23 - x2b_voo(a,k,m)*h2b_oovv(j,m,b,c) ! (1)
                         res_mm23 = res_mm23 + x2b_voo(b,k,m)*h2b_oovv(j,m,a,c) ! (ab)
                      end do
                      do e = 1,nua
                         ! A(ab) l2a(aej)*h2b_vovv(ekbc)
                         res_mm23 = res_mm23 + l2a(a,e,j)*h2b_vovv(e,k,b,c) ! (1)
                         res_mm23 = res_mm23 - l2a(b,e,j)*h2b_vovv(e,k,a,c) ! (ab)
                         ! l2b(eck)*h2a_vovv(ejab)
                         res_mm23 = res_mm23 + l2b(e,c,k)*h2a_vovv(e,j,a,b) ! (1)
                         ! x2a_vvv(aeb)*h2b_oovv(jkec)
                         res_mm23 = res_mm23 + x2a_vvv(a,e,b)*h2b_oovv(j,k,e,c) ! (1)
                      end do
                      do e = 1,nub
                         ! A(ab) l2b(aek)*h2b_ovvv(jebc)
                         res_mm23 = res_mm23 + l2b(a,e,k)*h2b_ovvv(j,e,b,c) ! (1)
                         res_mm23 = res_mm23 - l2b(b,e,k)*h2b_ovvv(j,e,a,c) ! (ab)
                         ! A(ab) x2b_vvv(aec)*h2b_oovv(jkbe)
                         res_mm23 = res_mm23 + x2b_vvv(a,e,c)*h2b_oovv(j,k,b,e) ! (1)
                         res_mm23 = res_mm23 - x2b_vvv(b,e,c)*h2b_oovv(j,k,a,e) ! (ab)
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
                                     h1b_ov,h1a_vv,h1b_oo,h1b_vv,&
                                     h2b_vvvv,h2b_vovo,h2b_ovvo,h2b_vovv,h2b_oovv,&
                                     h2c_vvvv,h2c_oooo,h2c_voov,h2c_ooov,h2c_vovv,h2c_oovv,&
                                     x2b_voo,x2b_ovo,x2b_vvv,&
                                     n3aab,n3abb,&
                                     noa,nua,nob,nub)

                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab, n3abb
                  ! Input R and T arrays
                  real(kind=8), intent(in) :: l1a(nua)
                  real(kind=8), intent(in) :: l2b(nua,nub,nob) 
                  integer, intent(in) :: l3b_excits(n3aab,5)
                  real(kind=8), intent(in) :: l3b_amps(n3aab)
                  ! Input H  and X arrays
                  real(kind=8), intent(in) :: h1b_ov(nob,nub)
                  real(kind=8), intent(in) :: h1a_vv(nua,nua)
                  real(kind=8), intent(in) :: h1b_oo(nob,nob)
                  real(kind=8), intent(in) :: h1b_vv(nub,nub)
                  real(kind=8), intent(in) :: h2b_vvvv(nua,nub,nua,nub)
                  real(kind=8), intent(in) :: h2b_vovo(nua,nob,nua,nob)
                  real(kind=8), intent(in) :: h2b_ovvo(noa,nub,nua,nob)
                  real(kind=8), intent(in) :: h2b_vovv(nua,nob,nua,nub)
                  real(kind=8), intent(in) :: h2b_oovv(noa,nob,nua,nub)
                  real(kind=8), intent(in) :: h2c_vvvv(nub,nub,nub,nub)
                  real(kind=8), intent(in) :: h2c_oooo(nob,nob,nob,nob)
                  real(kind=8), intent(in) :: h2c_voov(nub,nob,nob,nub)
                  real(kind=8), intent(in) :: h2c_ooov(nob,nob,nob,nub)
                  real(kind=8), intent(in) :: h2c_vovv(nub,nob,nub,nub)
                  real(kind=8), intent(in) :: h2c_oovv(nob,nob,nub,nub)
                  real(kind=8), intent(in) :: x2b_voo(nua,nob,nob)
                  real(kind=8), intent(in) :: x2b_ovo(noa,nub,nob)
                  real(kind=8), intent(in) :: x2b_vvv(nua,nub,nub)
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

                  !!! diagram 1: -A(jk) h1b(mj)*l3c(abcmk)
                  ! allocate new sorting arrays
                  nloc = nua*nub*(nub - 1)/2*(nob - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nub,nub,nua,nob))
                  !!! SB: (2,3,1,5) !!!
                  call get_index_table4(idx_table4, (/1,nub-1/), (/-1,nub/), (/1,nua/), (/2,nob/), nub, nub, nua, nob)
                  call sort4(l3c_excits, l3c_amps, loc_arr, idx_table4, (/2,3,1,5/), nub, nub, nua, nob, nloc, n3abb, resid)
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
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,a,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3c_excits(jdet,4);
                        ! compute < ab~c~j~k~ | h1b(oo) | ab~c~m~k~ >
                        hmatel = -h1b_oo(j,m)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table4(b,c,a,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3c_excits(jdet,4);
                        ! compute < ab~c~j~k~ | h1b(oo) | ab~c~m~j~ >
                        hmatel = h1b_oo(k,m)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,1,4) !!!
                  call get_index_table4(idx_table4, (/1,nub-1/), (/-1,nub/), (/1,nua/), (/1,nob-1/), nub, nub, nua, nob)
                  call sort4(l3c_excits, l3c_amps, loc_arr, idx_table4, (/2,3,1,4/), nub, nub, nua, nob, nloc, n3abb, resid)
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
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,a,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        n = l3c_excits(jdet,5);
                        ! compute < ab~c~j~k~ | h1b(oo) | ab~c~j~n~ >
                        hmatel = -h1b_oo(k,n)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table4(b,c,a,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        n = l3c_excits(jdet,5);
                        ! compute < ab~c~j~k~ | h1b(oo) | ab~c~k~n~ >
                        hmatel = h1b_oo(j,n)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 2: A(bc) h1b(be)*l3c(aecjk)
                  ! allocate new sorting arrays
                  nloc = nua*(nub - 1)*nob*(nob - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nob,nob,nua,nub))
                  !!! SB: (4,5,1,3) !!!
                  call get_index_table4(idx_table4, (/1,nob-1/), (/-1,nob/), (/1,nua/), (/2,nub/), nob, nob, nua, nub)
                  call sort4(l3c_excits, l3c_amps, loc_arr, idx_table4, (/4,5,1,3/), nob, nob, nua, nub, nloc, n3abb, resid)
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
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(j,k,a,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits(jdet,2);
                        ! compute < ab~c~j~k~ | h1b(vv) | ae~c~j~k~ >
                        hmatel = h1b_vv(e,b)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table4(j,k,a,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits(jdet,2);
                        ! compute < ab~c~j~k~ | h1b(vv) | ae~b~j~k~ >
                        hmatel = -h1b_vv(e,c)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,1,2) !!!
                  call get_index_table4(idx_table4, (/1,nob-1/), (/-1,nob/), (/1,nua/), (/1,nub-1/), nob, nob, nua, nub)
                  call sort4(l3c_excits, l3c_amps, loc_arr, idx_table4, (/4,5,1,2/), nob, nob, nua, nub, nloc, n3abb, resid)
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
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(j,k,a,b)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits(jdet,3);
                        ! compute < ab~c~j~k~ | h1b(vv) | ab~f~j~k~ >
                        hmatel = h1b_vv(f,c)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table4(j,k,a,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits(jdet,3);
                        ! compute < ab~c~j~k~ | h1b(vv) | ac~f~j~k~ >
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
                 
                  !!! diagram 3: h1a(ae)*l3c(ebcjk)
                  ! allocate new sorting arrays
                  nloc = nub*(nub - 1)/2*nob*(nob - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nub,nub,nob,nob))
                  !!! SB: (2,3,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nub-1/), (/-1,nub/), (/1,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(l3c_excits, l3c_amps, loc_arr, idx_table4, (/2,3,4,5/), nub, nub, nob, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_vv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3c_excits(jdet,1);
                        ! compute < ab~c~j~k~ | h1a(vv) | db~c~j~k~ >
                        hmatel = h1a_vv(d,a)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 4: h2c(mnjk)*l3c(abcmn)
                  ! allocate new sorting arrays
                  nloc = nua*nub*(nub - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nub,nub,nua))
                  !!! SB: (2,3,1) !!!
                  call get_index_table3(idx_table3, (/1,nub-1/), (/-1,nub/), (/1,nua/), nub, nub, nua)
                  call sort3(l3c_excits, l3c_amps, loc_arr, idx_table3, (/2,3,1/), nub, nub, nua, nloc, n3abb, resid)
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
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,a)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3c_excits(jdet,4); n = l3c_excits(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(oooo) | ab~c~m~n~ >
                        hmatel = h2c_oooo(j,k,m,n)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 5: A(bc)A(jk) h2b(mbej)*l3b(aecmk)
                  ! allocate and copy over l3b arrays
                  allocate(amps_buff(n3aab),excits_buff(n3aab,5))
                  amps_buff(:) = l3b_amps(:)
                  excits_buff(:,:) = l3b_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*nub*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nub,nob))
                  !!! SB: (1,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/1,nub/), (/1,nob/), nua, nub, nob)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,3,5/), nua, nub, nob, nloc, n3aab)
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
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     j = l3c_excits(idet,4); k = l3c_excits(idet,5);
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
                  !$omp l3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     j = l3c_excits(idet,4); k = l3c_excits(idet,5);
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

                  !!! diagram 6: A(bc)A(jk) h2c(bmje) * l3c(aecmk) 
                  ! allocate new sorting arrays
                  nloc = nua*(nub - 1)*(nob - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nub,nob))
                  !!! SB: (1,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/2,nub/), (/2,nob/), nua, nub, nob)
                  call sort3(l3c_excits, l3c_amps, loc_arr, idx_table3, (/1,3,5/), nua, nub, nob, nloc, n3abb, resid)
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
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits(jdet,2); m = l3c_excits(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~c~m~k~ >
                        hmatel = h2c_voov(b,m,j,e)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits(jdet,2); m = l3c_excits(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~b~m~k~ >
                        hmatel = -h2c_voov(c,m,j,e)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits(jdet,2); m = l3c_excits(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~c~m~j~ >
                        hmatel = -h2c_voov(b,m,k,e)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits(jdet,2); m = l3c_excits(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~b~m~j~ >
                        hmatel = h2c_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,5) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/1,nub-1/), (/2,nob/), nua, nub, nob)
                  call sort3(l3c_excits, l3c_amps, loc_arr, idx_table3, (/1,2,5/), nua, nub, nob, nloc, n3abb, resid)
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
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,b,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits(jdet,3); m = l3c_excits(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ab~f~m~k~ >
                        hmatel = h2c_voov(c,m,j,f)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits(jdet,3); m = l3c_excits(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ac~f~m~k~ >
                        hmatel = -h2c_voov(b,m,j,f)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits(jdet,3); m = l3c_excits(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ab~f~m~j~ >
                        hmatel = -h2c_voov(c,m,k,f)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits(jdet,3); m = l3c_excits(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ac~f~m~j~ >
                        hmatel = h2c_voov(b,m,k,f)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,4) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/2,nub/), (/1,nob-1/), nua, nub, nob)
                  call sort3(l3c_excits, l3c_amps, loc_arr, idx_table3, (/1,3,4/), nua, nub, nob, nloc, n3abb, resid)
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
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits(jdet,2); n = l3c_excits(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~c~j~n~ >
                        hmatel = h2c_voov(b,n,k,e)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits(jdet,2); n = l3c_excits(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~b~j~n~ >
                        hmatel = -h2c_voov(c,n,k,e)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits(jdet,2); n = l3c_excits(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~c~k~n~ >
                        hmatel = -h2c_voov(b,n,j,e)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits(jdet,2); n = l3c_excits(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~b~k~n~ >
                        hmatel = h2c_voov(c,n,j,e)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/1,nub-1/), (/1,nob-1/), nua, nub, nob)
                  call sort3(l3c_excits, l3c_amps, loc_arr, idx_table3, (/1,2,4/), nua, nub, nob, nloc, n3abb, resid)
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
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(a,b,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits(jdet,3); n = l3c_excits(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ab~f~j~n~ >
                        hmatel = h2c_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits(jdet,3); n = l3c_excits(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ac~f~j~n~ >
                        hmatel = -h2c_voov(b,n,k,f)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits(jdet,3); n = l3c_excits(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ab~f~k~n~ >
                        hmatel = -h2c_voov(c,n,j,f)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits(jdet,3); n = l3c_excits(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ac~f~k~n~ >
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

                  !!! diagram 7: -A(jk) h2b(amej)*l3c(ebcmk)
                  ! allocate new sorting arrays
                  nloc = nub*(nub - 1)/2*(nob - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nub,nub,nob))
                  !!! SB: (2,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nub-1/), (/-1,nub/), (/2,nob/), nub, nub, nob)
                  call sort3(l3c_excits, l3c_amps, loc_arr, idx_table3, (/2,3,5/), nub, nub, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vovo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3c_excits(jdet,1); m = l3c_excits(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(vovo) | db~c~m~k~ >
                        hmatel = -h2b_vovo(a,m,d,j)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3c_excits(jdet,1); m = l3c_excits(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(vovo) | db~c~m~j~ >
                        hmatel = h2b_vovo(a,m,d,k)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4) !!!
                  call get_index_table3(idx_table3, (/1,nub-1/), (/-1,nub/), (/1,nob-1/), nub, nub, nob)
                  call sort3(l3c_excits, l3c_amps, loc_arr, idx_table3, (/2,3,4/), nub, nub, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vovo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3c_excits(jdet,1); n = l3c_excits(jdet,5);
                        ! compute < ab~c~j~k~ | h2b(vovo) | db~c~j~n~ >
                        hmatel = -h2b_vovo(a,n,d,k)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3c_excits(jdet,1); n = l3c_excits(jdet,5);
                        ! compute < ab~c~j~k~ | h2b(vovo) | db~c~k~n~ >
                        hmatel = h2b_vovo(a,n,d,j)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 8: A(bc) h2b(abef)*l3c(efcjk)
                  ! allocate new sorting arrays
                  nloc = nob*(nob - 1)/2*(nub - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nob,nob,nub))
                  !!! SB: (4,5,3) !!!
                  call get_index_table3(idx_table3, (/1,nob-1/), (/-1,nob/), (/2,nub/), nob, nob, nub)
                  call sort3(l3c_excits, l3c_amps, loc_arr, idx_table3, (/4,5,3/), nob, nob, nub, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vvvv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(j,k,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3c_excits(jdet,1); e = l3c_excits(jdet,2);
                        ! compute < ab~c~j~k~ | h2b(vvvv) | de~c~j~k~ >
                        hmatel = h2b_vvvv(d,e,a,b)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table3(j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3c_excits(jdet,1); e = l3c_excits(jdet,2);
                        ! compute < ab~c~j~k~ | h2b(vvvv) | de~b~j~k~ >
                        hmatel = -h2b_vvvv(d,e,a,c)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,2) !!!
                  call get_index_table3(idx_table3, (/1,nob-1/), (/-1,nob/), (/1,nub-1/), nob, nob, nub)
                  call sort3(l3c_excits, l3c_amps, loc_arr, idx_table3, (/4,5,2/), nob, nob, nub, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits,&
                  !$omp l3c_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vvvv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(j,k,b)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3c_excits(jdet,1); f = l3c_excits(jdet,3);
                        ! compute < ab~c~j~k~ | h2b(vvvv) | db~f~j~k~ >
                        hmatel = h2b_vvvv(d,f,a,c)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table3(j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3c_excits(jdet,1); f = l3c_excits(jdet,3);
                        ! compute < ab~c~j~k~ | h2b(vvvv) | dc~f~j~k~ >
                        hmatel = -h2b_vvvv(d,f,a,b)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 9: 1/2 h2c(bcef)*l3c(aefjk)
                  ! allocate new sorting arrays
                  nloc = nua*nob*(nob - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nob,nob,nua))
                  !!! SB: (4,5,1) !!!
                  call get_index_table3(idx_table3, (/1,nob-1/), (/-1,nob/), (/1,nua/), nob, nob, nua)
                  call sort3(l3c_excits, l3c_amps, loc_arr, idx_table3, (/4,5,1/), nob, nob, nua, nloc, n3abb, resid)
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
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(j,k,a)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits(jdet,2); f = l3c_excits(jdet,3);
                        ! compute < ab~c~j~k~ | h2b(vvvv) | ae~f~j~k~ >
                        hmatel = h2c_vvvv(e,f,b,c)
                        resid(idet) = resid(idet) + hmatel * l3c_amps(jdet)
                     end do
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
                  !$omp h2b_vovv,h2b_oovv,&
                  !$omp h2c_ooov,h2c_vovv,h2c_oovv,&
                  !$omp x2b_voo,x2b_ovo,x2b_vvv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(idet,a,b,c,d,i,j,k,l,m,n,e,f,res_mm23)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                      a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                      j = l3c_excits(idet,4); k = l3c_excits(idet,5);
                      ! zero out value
                      res_mm23 = 0.0d0
                      ! l1a(a)*h2c_oovv(jkbc)
                      res_mm23 = res_mm23 + l1a(a)*h2c_oovv(j,k,b,c) ! (1)
                      ! A(bc)A(jk) l2b(abj)*h1b_ov(kc)
                      res_mm23 = res_mm23 + l2b(a,b,j)*h1b_ov(k,c) ! (1)
                      res_mm23 = res_mm23 - l2b(a,c,j)*h1b_ov(k,b) ! (bc)
                      res_mm23 = res_mm23 - l2b(a,b,k)*h1b_ov(j,c) ! (jk)
                      res_mm23 = res_mm23 + l2b(a,c,k)*h1b_ov(j,b) ! (bc)(jk)
                      do m = 1,noa
                         ! -A(bc)A(jk) x2b_ovo(mck)*h2b_oovv(mjab) 
                         res_mm23 = res_mm23 - x2b_ovo(m,c,k)*h2b_oovv(m,j,a,b) ! (1)
                         res_mm23 = res_mm23 + x2b_ovo(m,b,k)*h2b_oovv(m,j,a,c) ! (bc)
                         res_mm23 = res_mm23 + x2b_ovo(m,c,j)*h2b_oovv(m,k,a,b) ! (jk)
                         res_mm23 = res_mm23 - x2b_ovo(m,b,j)*h2b_oovv(m,k,a,c) ! (bc)(jk)
                      end do
                      do m = 1,nob
                         ! A(bc) -l2b(abm)*h2c_ooov(jkmc)
                         res_mm23 = res_mm23 - l2b(a,b,m)*h2c_ooov(j,k,m,c) ! (1)
                         res_mm23 = res_mm23 + l2b(a,c,m)*h2c_ooov(j,k,m,b) ! (bc)
                         ! A(jk) -x2b_voo(ajm)*h2c_oovv(mkbc)
                         res_mm23 = res_mm23 - x2b_voo(a,j,m)*h2c_oovv(m,k,b,c) ! (1)
                         res_mm23 = res_mm23 + x2b_voo(a,k,m)*h2c_oovv(m,j,b,c) ! (jk)
                      end do
                      do e = 1,nua
                         ! A(jk)A(bc) l2b(eck)*h2b_vovv(ejab)
                         res_mm23 = res_mm23 + l2b(e,c,k)*h2b_vovv(e,j,a,b) ! (1)
                         res_mm23 = res_mm23 - l2b(e,b,k)*h2b_vovv(e,j,a,c) ! (bc)
                         res_mm23 = res_mm23 - l2b(e,c,j)*h2b_vovv(e,k,a,b) ! (jk)
                         res_mm23 = res_mm23 + l2b(e,b,j)*h2b_vovv(e,k,a,c) ! (bc)(jk)
                      end do
                      do e = 1,nub
                         ! A(jk) l2b(aej)*h2c_vovv(ekbc)
                         res_mm23 = res_mm23 + l2b(a,e,j)*h2c_vovv(e,k,b,c) ! (1)
                         res_mm23 = res_mm23 - l2b(a,e,k)*h2c_vovv(e,j,b,c) ! (jk)
                         ! A(bc) x2b_vvv(aeb)*h2c_oovv(jkec)
                         res_mm23 = res_mm23 + x2b_vvv(a,e,b)*h2c_oovv(j,k,e,c) ! (1)
                         res_mm23 = res_mm23 - x2b_vvv(a,e,c)*h2c_oovv(j,k,e,b) ! (bc)
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
                      
                      real(kind=8), intent(inout) :: l1a(1:nua)
                      !f2py intent(in,out) :: l1a(0:nua-1)
                      real(kind=8), intent(inout) :: l2a(1:nua,1:nua,1:noa)
                      !f2py intent(in,out) :: l2a(0:nua-1,0:nua-1,0:noa-1)
                      real(kind=8), intent(inout) :: l2b(1:nua,1:nub,1:nob)
                      !f2py intent(in,out) :: l2b(0:nua-1,0:nub-1,0:nob-1)
                      real(kind=8), intent(inout) :: l3a_amps(n3aaa)
                      !f2py intent(in,out) :: l3a_amps(0:n3aaa-1) 
                      real(kind=8), intent(inout) :: l3b_amps(n3aab)
                      !f2py intent(in,out) :: l3b_amps(0:n3aab-1) 
                      real(kind=8), intent(inout) :: l3c_amps(n3abb)
                      !f2py intent(in,out) :: l3c_amps(0:n3abb-1) 
                      
                      integer :: j, k, a, b, c, idet
                      real(kind=8) :: denom

                      do a = 1,nua
                          denom = omega - H1A_vv(a,a)
                          if (denom==0.0d0) cycle
                          l1a(a) = l1a(a)/denom
                      end do

                      do j = 1,noa
                        do a = 1,nua
                           do b = 1,nua
                              if (a==b) cycle
                              denom = omega - H1A_vv(a,a) - H1A_vv(b,b) + H1A_oo(j,j)
                              l2a(a,b,j) = l2a(a,b,j)/denom
                          end do
                        end do
                      end do

                      do j = 1,nob
                        do a = 1,nua
                           do b = 1,nub
                              denom = omega - H1A_vv(a,a) - H1B_vv(b,b) + H1B_oo(j,j)
                              l2b(a,b,j) = l2b(a,b,j)/denom
                          end do
                        end do
                      end do

                      do idet = 1, n3aaa
                         a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                         j = l3a_excits(idet,4); k = l3a_excits(idet,5);

                         denom = H1A_vv(a,a) + H1A_vv(b,b) + H1A_vv(c,c)&
                                -H1A_oo(j,j) - H1A_oo(k,k)
                         
                         l3a_amps(idet) = l3a_amps(idet)/(omega - denom)
                      end do

                      do idet = 1, n3aab
                         a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                         j = l3b_excits(idet,4); k = l3b_excits(idet,5);

                         denom = H1A_vv(a,a) + H1A_vv(b,b) + H1B_vv(c,c)&
                                -H1A_oo(j,j) - H1B_oo(k,k)
                         
                         l3b_amps(idet) = l3b_amps(idet)/(omega - denom)
                      end do

                      do idet = 1, n3abb
                         a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                         j = l3c_excits(idet,4); k = l3c_excits(idet,5);

                         denom = H1A_vv(a,a) + H1B_vv(b,b) + H1B_vv(c,c)&
                                -H1B_oo(j,j) - H1B_oo(k,k)
                         
                         l3c_amps(idet) = l3c_amps(idet)/(omega - denom)
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

end module lefteaeom3_p_loops
