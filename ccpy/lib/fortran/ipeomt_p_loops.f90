module ipeomt_p_loops

      use omp_lib

      implicit none

      contains

              subroutine build_hr_3a(resid,&
                                     r2a,&
                                     r3a_amps,r3a_excits,&
                                     r3b_amps,r3b_excits,&
                                     t2a,&
                                     t3a_amps,t3a_excits,&
                                     t3b_amps,t3b_excits,&
                                     h1a_oo,h1a_vv,&
                                     h2a_vvvv,h2a_oooo,h2a_voov,h2a_vooo,h2a_vvov,&
                                     h2b_voov,&
                                     x1a_v,&
                                     x2a_ovv,x2a_ooo,&
                                     x2a_ovo,x2a_vvv,&
                                     x2b_ovo,&
                                     n3aaa,n3aab,&
                                     m3aaa,m3aab,&
                                     noa,nua,nob,nub)

                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa, n3aab, m3aaa, m3aab
                  ! Input R and T arrays
                  real(kind=8), intent(in) :: r2a(noa,nua,noa), t2a(nua,nua,noa,noa)
                  integer, intent(in) :: r3b_excits(n3aab,5)
                  real(kind=8), intent(in) :: r3b_amps(n3aab)
                  real(kind=8), intent(in) :: t3a_amps(m3aaa), t3b_amps(m3aab)
                  integer, intent(in) :: t3a_excits(m3aaa,6), t3b_excits(m3aab,6)
                  ! Input H  and X arrays
                  real(kind=8), intent(in) :: h1a_oo(noa,noa)
                  real(kind=8), intent(in) :: h1a_vv(nua,nua)
                  real(kind=8), intent(in) :: h2a_vvvv(nua,nua,nua,nua)
                  real(kind=8), intent(in) :: h2a_oooo(noa,noa,noa,noa)
                  real(kind=8), intent(in) :: h2a_voov(nua,noa,noa,nua)
                  real(kind=8), intent(in) :: h2a_vooo(nua,noa,noa,noa)
                  real(kind=8), intent(in) :: h2a_vvov(nua,nua,noa,nua)
                  real(kind=8), intent(in) :: h2b_voov(nua,nob,noa,nub)
                  real(kind=8), intent(in) :: x1a_v(nua)
                  real(kind=8), intent(in) :: x2a_ovv(noa,nua,nua)
                  real(kind=8), intent(in) :: x2a_ooo(noa,noa,noa)
                  real(kind=8), intent(in) :: x2a_ovo(noa,nua,noa)
                  real(kind=8), intent(in) :: x2a_vvv(nua,nua,nua)
                  real(kind=8), intent(in) :: x2b_ovo(noa,nub,nob)
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
                  real(kind=8) :: r_amp, hmatel, res_mm23, hmatel1
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc
                  
                  ! Zero the container that holds H*R
                  resid = 0.0d0

                  !!!! diagram 1: -A(j/ik) h1a(mj)*r3a(bcimk)
                  ! allocate new sorting arrays
                  nloc = (noa - 1)*(noa - 2)/2*nua*(nua - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(noa,noa,nua,nua))
                  !!! SB: (3,5,1,2) !!!
                  call get_index_table4(idx_table4, (/1,noa-2/), (/-2,noa/), (/1,nua-1/), (/-1,nua/), noa, noa, nua, nua)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table4, (/3,5,1,2/), noa, noa, nua, nua, nloc, n3aaa, resid)
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
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(i,k,b,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3a_excits(jdet,4);
                        ! compute < bcijk | h1a(oo) | bcimk >
                        hmatel = -h1a_oo(m,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table4(j,k,b,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3a_excits(jdet,4);
                        ! compute < bcijk | h1a(oo) | bcjmk >
                        hmatel = h1a_oo(m,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table4(i,j,b,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3a_excits(jdet,4);
                        ! compute < bcijk | h1a(oo) | bcimj >
                        hmatel = h1a_oo(m,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (3,4,1,2) !!!
                  call get_index_table4(idx_table4, (/1,noa-2/), (/-1,noa-1/), (/1,nua-1/), (/-1,nua/), noa, noa, nua, nua)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table4, (/3,4,1,2/), noa, noa, nua, nua, nloc, n3aaa, resid)
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
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(i,j,b,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        n = r3a_excits(jdet,5);
                        ! compute < bcijk | h1a(oo) | bcijn >
                        hmatel = -h1a_oo(n,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ik), (-1)
                     idx = idx_table4(j,k,b,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        n = r3a_excits(jdet,5);
                        ! compute < bcijk | h1a(oo) | bcjkn >
                        hmatel = -h1a_oo(n,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table4(i,k,b,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        n = r3a_excits(jdet,5);
                        ! compute < bcijk | h1a(oo) | bcikn >
                        hmatel = h1a_oo(n,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,1,2) !!!
                  call get_index_table4(idx_table4, (/2,noa-1/), (/-1,noa/), (/1,nua-1/), (/-1,nua/), noa, noa, nua, nua)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table4, (/4,5,1,2/), noa, noa, nua, nua, nloc, n3aaa, resid)
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
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(j,k,b,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = r3a_excits(jdet,3);
                        ! compute < bcijk | h1a(oo) | bcljk >
                        hmatel = -h1a_oo(l,i)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table4(i,k,b,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = r3a_excits(jdet,3);
                        ! compute < bcijk | h1a(oo) | bclik >
                        hmatel = h1a_oo(l,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ik), (-1)
                     idx = idx_table4(i,j,b,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = r3a_excits(jdet,3);
                        ! compute < bcijk | h1a(oo) | bclij >
                        hmatel = -h1a_oo(l,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!!! diagram 2: A(bc) h1a(be)*r3a(ecijk)
                  ! allocate new sorting arrays
                  nloc = noa*(noa-1)*(noa-2)/6 * nua
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(noa,noa,noa,nua))
                  !!! SB: (3,4,5,1) !!!
                  call get_index_table4(idx_table4, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/1,nua-1/), noa, noa, noa, nua)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table4, (/3,4,5,1/), noa, noa, noa, nua, nloc, n3aaa, resid)
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
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(i,j,k,b)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3a_excits(jdet,2);
                        ! compute < bcijk | h1a(vv) | bfijk >
                        hmatel = h1a_vv(c,f)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table4(i,j,k,c)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = r3a_excits(jdet,2);
                           ! compute < bcijk | h1a(vv) | cfijk >
                           hmatel = -h1a_vv(b,f)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (3,4,5,2) !!!
                  call get_index_table4(idx_table4, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/2,nua/), noa, noa, noa, nua)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table4, (/3,4,5,2/), noa, noa, noa, nua, nloc, n3aaa, resid)
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
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(i,j,k,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3a_excits(jdet,1);
                        ! compute < bcijk | h1a(vv) | ecijk >
                        hmatel = h1a_vv(b,e)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table4(i,j,k,b)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits(jdet,1);
                           ! compute < bcijk | h1a(vv) | ebijk >
                           hmatel = -h1a_vv(c,e)
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!!! diagram 3: A(i/jk) 1/2 h2a(mnjk)*r3a(bcimn)
                  ! allocate new sorting arrays
                  nloc = nua*(nua - 1)/2*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,noa))
                  !!! SB: (1,2,3) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/-1,nua/), (/1,noa-2/), nua, nua, noa)
                  call sort3(r3a_excits, r3a_amps, loc_arr, idx_table3, (/1,2,3/), nua, nua, noa, nloc, n3aaa, resid)
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
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,i)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3a_excits(jdet,4); n = r3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | bcimn >
                        hmatel = h2a_oooo(m,n,j,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3a_excits(jdet,4); n = r3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | bcjmn >
                        hmatel = -h2a_oooo(m,n,i,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3a_excits(jdet,4); n = r3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | bckmn >
                        hmatel = h2a_oooo(m,n,i,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/-1,nua/), (/2,noa-1/), nua, nua, noa)
                  call sort3(r3a_excits, r3a_amps, loc_arr, idx_table3, (/1,2,4/), nua, nua, noa, nloc, n3aaa, resid)
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
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = r3a_excits(jdet,3); n = r3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | bcljn >
                        hmatel = h2a_oooo(l,n,i,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table3(b,c,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = r3a_excits(jdet,3); n = r3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | bclin >
                        hmatel = -h2a_oooo(l,n,j,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = r3a_excits(jdet,3); n = r3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | bclkn >
                        hmatel = -h2a_oooo(l,n,i,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/-1,nua/), (/3,noa/), nua, nua, noa)
                  call sort3(r3a_excits, r3a_amps, loc_arr, idx_table3, (/1,2,5/), nua, nua, noa, nloc, n3aaa, resid)
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
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = r3a_excits(jdet,3); m = r3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | bclmk >
                        hmatel = h2a_oooo(l,m,i,j)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = r3a_excits(jdet,3); m = r3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | bclmj >
                        hmatel = -h2a_oooo(l,m,i,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table3(b,c,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = r3a_excits(jdet,3); m = r3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | bclmi >
                        hmatel = h2a_oooo(l,m,j,k)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!!! diagram 4: 1/2 h2a(bcef)*r3a(efijk)
                  ! allocate new sorting arrays
                  nloc = noa*(noa - 1)*(noa - 2)/6
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(noa,noa,noa))
                  !!! SB: (3,4,5) !!!
                  call get_index_table3(idx_table3, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), noa, noa, noa)
                  call sort3(r3a_excits, r3a_amps, loc_arr, idx_table3, (/3,4,5/), noa, noa, noa, nloc, n3aaa, resid)
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
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3a_excits(jdet,1); f = r3a_excits(jdet,2);
                        ! compute < bcijk | h2a(oooo) | efijk >
                        !hmatel = h2a_vvvv(b,c,e,f)
                        hmatel = h2a_vvvv(e,f,b,c)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!!! diagram 5: A(i/jk)A(bc) h2a(bmje)*r3a(ecimk)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*(noa - 1)*(noa - 2)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(noa,noa,nua))
                  !!! [1] SB: (3,5,1) !!!
                  call get_index_table3(idx_table3, (/1,noa-2/), (/-2,noa/), (/1,nua-1/), noa, noa, nua)
                  call sort3(r3a_excits, r3a_amps, loc_arr, idx_table3, (/3,5,1/), noa, noa, nua, nloc, n3aaa, resid)
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
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,k,b)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3a_excits(jdet,2); m = r3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | bfimk >
                        hmatel = h2a_voov(c,m,j,f)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table3(j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3a_excits(jdet,2); m = r3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | bfjmk >
                        hmatel = -h2a_voov(c,m,i,f)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(i,j,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3a_excits(jdet,2); m = r3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | bfimj >
                        hmatel = -h2a_voov(c,m,k,f)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(i,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3a_excits(jdet,2); m = r3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | cfimk >
                        hmatel = -h2a_voov(b,m,j,f)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ij)(bc)
                     idx = idx_table3(j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3a_excits(jdet,2); m = r3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | cfjmk >
                        hmatel = h2a_voov(b,m,i,f)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (jk)(bc)
                     idx = idx_table3(i,j,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3a_excits(jdet,2); m = r3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | cfimj >
                        hmatel = h2a_voov(b,m,k,f)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! [2] SB: (3,4,1) !!!
                  call get_index_table3(idx_table3, (/1,noa-2/), (/-1,noa-1/), (/1,nua-1/), noa, noa, nua)
                  call sort3(r3a_excits, r3a_amps, loc_arr, idx_table3, (/3,4,1/), noa, noa, nua, nloc, n3aaa, resid)
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
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,j,b)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3a_excits(jdet,2); n = r3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | bfijn >
                        hmatel = h2a_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ik), (-1)
                     idx = idx_table3(j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3a_excits(jdet,2); n = r3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | bfjkn >
                        hmatel = h2a_voov(c,n,i,f)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(i,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3a_excits(jdet,2); n = r3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | bfikn >
                        hmatel = -h2a_voov(c,n,j,f)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(i,j,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3a_excits(jdet,2); n = r3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | cfijn >
                        hmatel = -h2a_voov(b,n,k,f)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ik)(bc), (-1)
                     idx = idx_table3(j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3a_excits(jdet,2); n = r3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | cfjkn >
                        hmatel = -h2a_voov(b,n,i,f)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (jk)(bc)
                     idx = idx_table3(i,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3a_excits(jdet,2); n = r3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | cfikn >
                        hmatel = h2a_voov(b,n,j,f)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! [3] SB: (4,5,1) !!!
                  call get_index_table3(idx_table3, (/2,noa-1/), (/-1,noa/), (/1,nua-1/), noa, noa, nua)
                  call sort3(r3a_excits, r3a_amps, loc_arr, idx_table3, (/4,5,1/), noa, noa, nua, nloc, n3aaa, resid)
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
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(j,k,b)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3a_excits(jdet,2); l = r3a_excits(jdet,3);
                        ! compute < bcijk | h2a(oooo) | bfljk >
                        hmatel = h2a_voov(c,l,i,f)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table3(i,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3a_excits(jdet,2); l = r3a_excits(jdet,3);
                        ! compute < bcijk | h2a(oooo) | bflik >
                        hmatel = -h2a_voov(c,l,j,f)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ik), (-1)
                     idx = idx_table3(i,j,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3a_excits(jdet,2); l = r3a_excits(jdet,3);
                        ! compute < bcijk | h2a(oooo) | bflij >
                        hmatel = h2a_voov(c,l,k,f)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3a_excits(jdet,2); l = r3a_excits(jdet,3);
                        ! compute < bcijk | h2a(oooo) | cfljk >
                        hmatel = -h2a_voov(b,l,i,f)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ij)(bc)
                     idx = idx_table3(i,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3a_excits(jdet,2); l = r3a_excits(jdet,3);
                        ! compute < bcijk | h2a(oooo) | cflik >
                        hmatel = h2a_voov(b,l,j,f)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ik)(bc), (-1)
                     idx = idx_table3(i,j,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3a_excits(jdet,2); l = r3a_excits(jdet,3);
                        ! compute < bcijk | h2a(oooo) | cflij >
                        hmatel = -h2a_voov(b,l,k,f)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! [4] SB: (3,5,2) !!!
                  call get_index_table3(idx_table3, (/1,noa-2/), (/-2,noa/), (/2,nua/), noa, noa, nua)
                  call sort3(r3a_excits, r3a_amps, loc_arr, idx_table3, (/3,5,2/), noa, noa, nua, nloc, n3aaa, resid)
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
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,k,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3a_excits(jdet,1); m = r3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | bfimk >
                        hmatel = h2a_voov(b,m,j,e)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table3(j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3a_excits(jdet,1); m = r3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | bfjmk >
                        hmatel = -h2a_voov(b,m,i,e)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(i,j,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3a_excits(jdet,1); m = r3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | bfimj >
                        hmatel = -h2a_voov(b,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(i,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3a_excits(jdet,1); m = r3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | cfimk >
                        hmatel = -h2a_voov(c,m,j,e)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ij)(bc)
                     idx = idx_table3(j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3a_excits(jdet,1); m = r3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | cfjmk >
                        hmatel = h2a_voov(c,m,i,e)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (jk)(bc)
                     idx = idx_table3(i,j,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3a_excits(jdet,1); m = r3a_excits(jdet,4);
                        ! compute < bcijk | h2a(oooo) | cfimj >
                        hmatel = h2a_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! [5] SB: (3,4,2) !!!
                  call get_index_table3(idx_table3, (/1,noa-2/), (/-1,noa-1/), (/2,nua/), noa, noa, nua)
                  call sort3(r3a_excits, r3a_amps, loc_arr, idx_table3, (/3,4,2/), noa, noa, nua, nloc, n3aaa, resid)
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
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,j,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3a_excits(jdet,1); n = r3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | bfijn >
                        hmatel = h2a_voov(b,n,k,e)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ik), (-1)
                     idx = idx_table3(j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3a_excits(jdet,1); n = r3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | bfjkn >
                        hmatel = h2a_voov(b,n,i,e)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(i,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3a_excits(jdet,1); n = r3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | bfikn >
                        hmatel = -h2a_voov(b,n,j,e)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(i,j,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3a_excits(jdet,1); n = r3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | cfijn >
                        hmatel = -h2a_voov(c,n,k,e)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ik)(bc), (-1)
                     idx = idx_table3(j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3a_excits(jdet,1); n = r3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | cfjkn >
                        hmatel = -h2a_voov(c,n,i,e)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (jk)(bc)
                     idx = idx_table3(i,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3a_excits(jdet,1); n = r3a_excits(jdet,5);
                        ! compute < bcijk | h2a(oooo) | cfikn >
                        hmatel = h2a_voov(c,n,j,e)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! [6] SB: (4,5,2) !!!
                  call get_index_table3(idx_table3, (/2,noa-1/), (/-1,noa/), (/2,nua/), noa, noa, nua)
                  call sort3(r3a_excits, r3a_amps, loc_arr, idx_table3, (/4,5,2/), noa, noa, nua, nloc, n3aaa, resid)
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
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table3(j,k,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3a_excits(jdet,1); l = r3a_excits(jdet,3);
                        ! compute < bcijk | h2a(oooo) | bfljk >
                        hmatel = h2a_voov(b,l,i,e)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table3(i,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3a_excits(jdet,1); l = r3a_excits(jdet,3);
                        ! compute < bcijk | h2a(oooo) | bflik >
                        hmatel = -h2a_voov(b,l,j,e)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ik), (-1)
                     idx = idx_table3(i,j,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3a_excits(jdet,1); l = r3a_excits(jdet,3);
                        ! compute < bcijk | h2a(oooo) | bflij >
                        hmatel = h2a_voov(b,l,k,e)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3a_excits(jdet,1); l = r3a_excits(jdet,3);
                        ! compute < bcijk | h2a(oooo) | cfljk >
                        hmatel = -h2a_voov(c,l,i,e)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ij)(bc)
                     idx = idx_table3(i,k,b)
                     if (idx/=0) then   
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3a_excits(jdet,1); l = r3a_excits(jdet,3);
                        ! compute < bcijk | h2a(oooo) | cflik >
                        hmatel = h2a_voov(c,l,j,e)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ik)(bc), (-1)
                     idx = idx_table3(i,j,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3a_excits(jdet,1); l = r3a_excits(jdet,3);
                        ! compute < bcijk | h2a(oooo) | cflij >
                        hmatel = -h2a_voov(c,l,k,e)
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!!! diagram 6: A(i/jk)A(bc) h2b(bmje)*r3b(ceikm)
                  ! copy over r3b arrays
                  allocate(excits_buff(n3aab,5),amps_buff(n3aab))
                  excits_buff(:,:) = r3b_excits(:,:)
                  amps_buff(:) = r3b_amps(:)
                  ! allocate new sorting arrays
                  nloc = noa*(noa - 1)/2*nua 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(noa,noa,nua))
                  !!! SB: (3,4,1) !!!
                  call get_index_table3(idx_table3, (/1,noa-1/), (/-1,noa/), (/1,nua/), noa, noa, nua)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/3,4,1/), noa, noa, nua, nloc, n3aab)
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
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
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

                  !!!! diagram 7: A(i/jk) x2a_ovo(iem) t3a(ebcmjk) 
                  ! copy over t3a arrays
                  allocate(excits_buff(m3aaa,6),amps_buff(m3aaa))
                  excits_buff(:,:) = t3a_excits(:,:)
                  amps_buff(:) = t3a_amps(:)
                  ! allocate new sorting arrays
                  nloc = (nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nua,noa,noa))
                  !!! SB: (2,3,5,6) !!!
                  call get_index_table4(idx_table4, (/2,nua-1/), (/-1,nua/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/2,3,5,6/), nua, nua, noa, noa, nloc, m3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2a_ovo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < bcijk | x2a(ovo) | ebcmjk >
                        hmatel = x2a_ovo(i,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table4(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < bcijk | x2a(ovo) | ebcmik >
                        hmatel = -x2a_ovo(j,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik), -
                     idx = idx_table4(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < bcijk | x2a(ovo) | ebcmij >
                        hmatel = x2a_ovo(k,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4,6) !!!
                  call get_index_table4(idx_table4, (/2,nua-1/), (/-1,nua/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/2,3,4,6/), nua, nua, noa, noa, nloc, m3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2a_ovo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        ! compute < bcijk | x2a(ovo) | ebcimk >
                        hmatel = x2a_ovo(j,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        ! compute < bcijk | x2a(ovo) | ebcjmk >
                        hmatel = -x2a_ovo(i,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table4(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        ! compute < bcijk | x2a(ovo) | ebcimj >
                        hmatel = -x2a_ovo(k,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4,5) !!!
                  call get_index_table4(idx_table4, (/2,nua-1/), (/-1,nua/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/2,3,4,5/), nua, nua, noa, noa, nloc, m3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2a_ovo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,6);
                        ! compute < bcijk | x2a(ovo) | ebcijm >
                        hmatel = x2a_ovo(k,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table4(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,6);
                        ! compute < bcijk | x2a(ovo) | ebcikm >
                        hmatel = -x2a_ovo(j,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik), -
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,6);
                        ! compute < bcijk | x2a(ovo) | ebcjkm >
                        hmatel = x2a_ovo(i,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,5,6) !!!
                  call get_index_table4(idx_table4, (/1,nua-2/), (/-2,nua/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/1,3,5,6/), nua, nua, noa, noa, nloc, m3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2a_ovo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < bcijk | x2a(ovo) | becmjk >
                        hmatel = -x2a_ovo(i,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table4(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < bcijk | x2a(ovo) | becmik >
                        hmatel = x2a_ovo(j,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik), -
                     idx = idx_table4(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < bcijk | x2a(ovo) | becmij >
                        hmatel = -x2a_ovo(k,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,4,6) !!!
                  call get_index_table4(idx_table4, (/1,nua-2/), (/-2,nua/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/1,3,4,6/), nua, nua, noa, noa, nloc, m3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2a_ovo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        ! compute < bcijk | x2a(ovo) | becimk >
                        hmatel = -x2a_ovo(j,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        ! compute < bcijk | x2a(ovo) | becjmk >
                        hmatel = x2a_ovo(i,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table4(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        ! compute < bcijk | x2a(ovo) | becimj >
                        hmatel = x2a_ovo(k,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nua-2/), (/-2,nua/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/1,3,4,5/), nua, nua, noa, noa, nloc, m3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2a_ovo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,6);
                        ! compute < bcijk | x2a(ovo) | becijm >
                        hmatel = -x2a_ovo(k,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table4(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,6);
                        ! compute < bcijk | x2a(ovo) | becikm >
                        hmatel = x2a_ovo(j,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik), -
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,6);
                        ! compute < bcijk | x2a(ovo) | becjkm >
                        hmatel = -x2a_ovo(i,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,5,6) !!!
                  call get_index_table4(idx_table4, (/1,nua-2/), (/-1,nua-1/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/1,2,5,6/), nua, nua, noa, noa, nloc, m3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2a_ovo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,4);
                        ! compute < bcijk | x2a(ovo) | bcemjk >
                        hmatel = x2a_ovo(i,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table4(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,4);
                        ! compute < bcijk | x2a(ovo) | bcemik >
                        hmatel = -x2a_ovo(j,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik), -
                     idx = idx_table4(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,4);
                        ! compute < bcijk | x2a(ovo) | bcemij >
                        hmatel = x2a_ovo(k,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4,6) !!!
                  call get_index_table4(idx_table4, (/1,nua-2/), (/-1,nua-1/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/1,2,4,6/), nua, nua, noa, noa, nloc, m3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2a_ovo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        ! compute < bcijk | x2a(ovo) | bceimk >
                        hmatel = x2a_ovo(j,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        ! compute < bcijk | x2a(ovo) | bcejmk >
                        hmatel = -x2a_ovo(i,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table4(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        ! compute < bcijk | x2a(ovo) | bceimj >
                        hmatel = -x2a_ovo(k,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nua-2/), (/-1,nua-1/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/1,2,4,5/), nua, nua, noa, noa, nloc, m3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2a_ovo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                        ! compute < bcijk | x2a(ovo) | bceijm >
                        hmatel = x2a_ovo(k,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table4(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                        ! compute < bcijk | x2a(ovo) | bceikm >
                        hmatel = -x2a_ovo(j,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik), -
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                        ! compute < bcijk | x2a(ovo) | bcejkm >
                        hmatel = x2a_ovo(i,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4,excits_buff,amps_buff)

                  !!!! diagram 8: A(i/jk) x2b_ovo(ie~m~) t3b(bce~jkm~) 
                  ! copy over t3b arrays
                  allocate(excits_buff(m3aab,6),amps_buff(m3aab))
                  excits_buff(:,:) = t3b_excits(:,:)
                  amps_buff(:) = t3b_amps(:)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2*noa*(noa-1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nub,noa,nob))
                  !!! SB: (1,2,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nua-1/), (/-1,nua/), (/1,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/1,2,4,5/), nua, nua, noa, noa, nloc, m3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2b_ovo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                        ! compute < bcijk | x2b(ovo) | bce~jkm~ >
                        hmatel = x2b_ovo(i,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table4(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                        ! compute < bcijk | x2b(ovo) | bce~ikm~ >
                        hmatel = -x2b_ovo(j,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ik), -
                     idx = idx_table4(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                        ! compute < bcijk | x2b(ovo) | bce~ijm~ >
                        hmatel = x2b_ovo(k,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4,excits_buff,amps_buff)

                  !!!! diagram 9: A(bc) 1/2 x2a_vvv(bef) t3a(fecijk) 
                  !!!! diagram 10: x1a_v(e) t3a(ebcijk)
                  ! copy over t3a arrays
                  allocate(excits_buff(m3aaa,6),amps_buff(m3aaa))
                  excits_buff(:,:) = t3a_excits(:,:)
                  amps_buff(:) = t3a_amps(:)
                  ! allocate new sorting arrays
                  nloc = (nua-2)*noa*(noa-1)*(noa-2)/6 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(noa,noa,noa,nua))
                  !!! SB: (4,5,6,1) !!!
                  call get_index_table4(idx_table4, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/1,nua-2/), noa, noa, noa, nua)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/4,5,6,1/), noa, noa, noa, nua, nloc, m3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2a_vvv,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(i,j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); f = excits_buff(jdet,3);
                        ! compute < bcijk | x2a(vvv) | cefijk >
                        hmatel = -x2a_vvv(b,e,f)
                        ! compute < bcijk | x1a(v) | cefijk >
                        hmatel1 = 0.0d0
                        if (b==e) hmatel1 = hmatel1 - x1a_v(f)
                        if (b==f) hmatel1 = hmatel1 + x1a_v(e)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table4(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); f = excits_buff(jdet,3);
                        ! compute < bcijk | x2a(vvv) | befijk >
                        hmatel = x2a_vvv(c,e,f)
                        ! compute < bcijk | x1a(v) | befijk >
                        hmatel1 = 0.0d0
                        if (c==e) hmatel1 = hmatel1 + x1a_v(f)
                        if (c==f) hmatel1 = hmatel1 - x1a_v(e)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,6,2) !!!
                  call get_index_table4(idx_table4, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/2,nua-1/), noa, noa, noa, nua)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/4,5,6,2/), noa, noa, noa, nua, nloc, m3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2a_vvv,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); f = excits_buff(jdet,3);
                        ! compute < bcijk | x2a(vvv) | ebfijk >
                        hmatel = -x2a_vvv(c,e,f)
                        ! compute < bcijk | x1a(v) | ebfijk >
                        hmatel1 = 0.0d0
                        if (c==e) hmatel1 = hmatel1 - x1a_v(f)
                        if (c==f) hmatel1 = hmatel1 + x1a_v(e)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table4(i,j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); f = excits_buff(jdet,3);
                        ! compute < bcijk | x2a(vvv) | ecfijk >
                        hmatel = x2a_vvv(b,e,f)
                        ! compute < bcijk | x1a(v) | ecfijk >
                        hmatel1 = 0.0d0
                        if (b==e) hmatel1 = hmatel1 + x1a_v(f)
                        if (b==f) hmatel1 = hmatel1 - x1a_v(e)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,6,3) !!!
                  call get_index_table4(idx_table4, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/3,nua/), noa, noa, noa, nua)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/4,5,6,3/), noa, noa, noa, nua, nloc, m3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2a_vvv,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                     b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                     ! (1)
                     idx = idx_table4(i,j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); f = excits_buff(jdet,2);
                        ! compute < bcijk | x2a(vvv) | efcijk >
                        hmatel = -x2a_vvv(b,e,f)
                        ! compute < bcijk | x1a(v) | efcijk >
                        hmatel1 = 0.0d0
                        if (b==e) hmatel1 = hmatel1 - x1a_v(f)
                        if (b==f) hmatel1 = hmatel1 + x1a_v(e)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table4(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); f = excits_buff(jdet,2);
                        ! compute < bcijk | x2a(vvv) | efbijk >
                        hmatel = x2a_vvv(c,e,f)
                        ! compute < bcijk | x1a(v) | efbijk >
                        hmatel1 = 0.0d0
                        if (c==e) hmatel1 = hmatel1 + x1a_v(f)
                        if (c==f) hmatel1 = hmatel1 - x1a_v(e)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4,excits_buff,amps_buff)

                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp t2a,r2a,&
                  !$omp h2a_vvov,h2a_vooo,&
                  !$omp x2a_ovv,x2a_ooo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(idet,a,b,c,d,i,j,k,l,m,n,e,f,res_mm23)
                  !$omp do schedule(static)
                  do idet = 1,n3aaa
                      b = r3a_excits(idet,1); c = r3a_excits(idet,2);
                      i = r3a_excits(idet,3); j = r3a_excits(idet,4); k = r3a_excits(idet,5);
                      ! zero out value
                      res_mm23 = 0.0d0
                      do m = 1,noa
                         ! -A(i/jk)A(bc) h2a_vooo(cmkj)*r2a(ibm)
                         res_mm23 = res_mm23 - h2a_vooo(c,m,k,j)*r2a(i,b,m) ! (1)
                         res_mm23 = res_mm23 + h2a_vooo(c,m,k,i)*r2a(j,b,m) ! (ij)
                         res_mm23 = res_mm23 + h2a_vooo(c,m,i,j)*r2a(k,b,m) ! (ik)
                         res_mm23 = res_mm23 + h2a_vooo(b,m,k,j)*r2a(i,c,m) ! (bc)
                         res_mm23 = res_mm23 - h2a_vooo(b,m,k,i)*r2a(j,c,m) ! (ij)(bc)
                         res_mm23 = res_mm23 - h2a_vooo(b,m,i,j)*r2a(k,c,m) ! (ik)(bc)
                         ! -A(k/ij) x2a_ooo(imj)*t2a(bcmk)
                         res_mm23 = res_mm23 - x2a_ooo(i,m,j)*t2a(b,c,m,k) ! (1)
                         res_mm23 = res_mm23 + x2a_ooo(k,m,j)*t2a(b,c,m,i) ! (ik)
                         res_mm23 = res_mm23 + x2a_ooo(i,m,k)*t2a(b,c,m,j) ! (jk)
                      end do
                      do e = 1,nua
                         ! A(k/ij) h2a_vvov(cbke)*r2a(iej)
                         res_mm23 = res_mm23 + h2a_vvov(c,b,k,e)*r2a(i,e,j) ! (1)
                         res_mm23 = res_mm23 - h2a_vvov(c,b,i,e)*r2a(k,e,j) ! (ik)
                         res_mm23 = res_mm23 - h2a_vvov(c,b,j,e)*r2a(i,e,k) ! (jk)
                         ! A(i/jk)A(bc) x2a_ovv(ibe)*t2a(ecjk)
                         res_mm23 = res_mm23 + x2a_ovv(i,b,e)*t2a(e,c,j,k) ! (1)
                         res_mm23 = res_mm23 - x2a_ovv(j,b,e)*t2a(e,c,i,k) ! (ij)
                         res_mm23 = res_mm23 - x2a_ovv(k,b,e)*t2a(e,c,j,i) ! (ik)
                         res_mm23 = res_mm23 - x2a_ovv(i,c,e)*t2a(e,b,j,k) ! (bc)
                         res_mm23 = res_mm23 + x2a_ovv(j,c,e)*t2a(e,b,i,k) ! (ij)(bc)
                         res_mm23 = res_mm23 + x2a_ovv(k,c,e)*t2a(e,b,j,i) ! (ik)(bc)
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
                                     t3b_amps,t3b_excits,&
                                     t3c_amps,t3c_excits,&
                                     h1a_oo,h1a_vv,h1b_oo,h1b_vv,&
                                     h2a_oooo,h2a_voov,h2a_vooo,&
                                     h2b_vvvv,h2b_oooo,h2b_voov,h2b_vovo,h2b_ovov,h2b_ovvo,&
                                     h2b_vooo,h2b_ovoo,h2b_vvov,h2b_vvvo,&
                                     h2c_voov,&
                                     x1a_v,&
                                     x2a_ovv,x2a_ooo,&
                                     x2a_ovo,x2a_vvv,&
                                     x2b_ovv,x2b_vvo,x2b_ooo,&
                                     x2b_ovo,x2b_voo,x2b_vvv,&
                                     n3aaa,n3aab,n3abb,&
                                     m3aab,m3abb,&
                                     noa,nua,nob,nub)

                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa, n3aab, n3abb, m3aab, m3abb
                  ! Input R and T arrays
                  real(kind=8), intent(in) :: r2a(noa,nua,noa), t2a(nua,nua,noa,noa)
                  real(kind=8), intent(in) :: r2b(noa,nub,nob), t2b(nua,nub,noa,nob)
                  integer, intent(in) :: r3a_excits(n3aaa,5), r3c_excits(n3abb,5)
                  real(kind=8), intent(in) :: r3a_amps(n3aaa), r3c_amps(n3abb)
                  real(kind=8), intent(in) :: t3b_amps(m3aab), t3c_amps(m3abb)
                  integer, intent(in) :: t3b_excits(m3aab,6), t3c_excits(m3abb,6)
                  ! Input H  and X arrays
                  real(kind=8), intent(in) :: h1a_oo(noa,noa)
                  real(kind=8), intent(in) :: h1a_vv(nua,nua)
                  real(kind=8), intent(in) :: h1b_oo(nob,nob)
                  real(kind=8), intent(in) :: h1b_vv(nub,nub)
                  real(kind=8), intent(in) :: h2a_oooo(noa,noa,noa,noa)
                  real(kind=8), intent(in) :: h2a_voov(nua,noa,noa,nua)
                  real(kind=8), intent(in) :: h2a_vooo(nua,noa,noa,noa)
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
                  real(kind=8), intent(in) :: x1a_v(nua)
                  real(kind=8), intent(in) :: x2a_ovv(noa,nua,nua)
                  real(kind=8), intent(in) :: x2a_ooo(noa,noa,noa)
                  real(kind=8), intent(in) :: x2b_ovv(noa,nub,nub)
                  real(kind=8), intent(in) :: x2b_vvo(nua,nub,nob)
                  real(kind=8), intent(in) :: x2b_ooo(noa,nob,nob)
                  real(kind=8), intent(in) :: x2a_ovo(noa,nua,noa)
                  real(kind=8), intent(in) :: x2a_vvv(nua,nua,nua)
                  real(kind=8), intent(in) :: x2b_ovo(noa,nub,nob)
                  real(kind=8), intent(in) :: x2b_voo(nua,nob,nob)
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
                  real(kind=8) :: r_amp, hmatel, res_mm23, hmatel1
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc
                  
                  ! Zero the container that holds H*R
                  resid = 0.0d0

                  ! diagram 1: -A(ij) h1a(mj)*r3b(bcimk) 
                  ! allocate new sorting arrays
                  nloc = nua*nub*(noa - 1)*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nub,noa,nob))
                  !!! SB: (1,2,3,5) !!!
                  call get_index_table4(idx_table4, (/1,nua/), (/1,nub/), (/1,noa-1/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(r3b_excits, r3b_amps, loc_arr, idx_table4, (/1,2,3,5/), nua, nub, noa, nob, nloc, n3aab, resid)
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
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,i,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3b_excits(jdet,4);
                        ! compute < bc~ijk~ | h1a(oo) | bc~imk~ >
                        hmatel = -h1a_oo(m,j)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3b_excits(jdet,4);
                        ! compute < bc~ijk~ | h1a(oo) | bc~jmk~ >
                        hmatel = h1a_oo(m,i)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nua/), (/1,nub/), (/2,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(r3b_excits, r3b_amps, loc_arr, idx_table4, (/1,2,4,5/), nua, nub, noa, nob, nloc, n3aab, resid)
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
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = r3b_excits(jdet,3);
                        ! compute < bc~ijk~ | h1a(oo) | bc~ljk~ >
                        hmatel = -h1a_oo(l,i)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table4(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = r3b_excits(jdet,3);
                        ! compute < bc~ijk~ | h1a(oo) | bc~lik~ >
                        hmatel = h1a_oo(l,j)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  ! diagram 2: -h1b(mk)*r3b(bcijm)
                  ! allocate new sorting arrays
                  nloc = nua*nub*noa*(noa - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(noa,noa,nua,nub))
                  !!! SB: (3,4,1,2) !!!
                  call get_index_table4(idx_table4, (/1,noa-1/), (/-1,noa/), (/1,nua/), (/1,nub/), noa, noa, nua, nub)
                  call sort4(r3b_excits, r3b_amps, loc_arr, idx_table4, (/3,4,1,2/), noa, noa, nua, nub, nloc, n3aab, resid)
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
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(i,j,b,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        n = r3b_excits(jdet,5);
                        ! compute < bc~ijk~ | h1b(oo) | bc~ijn~ >
                        hmatel = -h1b_oo(n,k)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  ! diagram 3: h1a(be)*r3b(ecijk)
                  ! allocate new sorting arrays
                  nloc = nub*noa*(noa - 1)/2*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(noa,noa,nub,nob))
                  !!! SB: (3,4,2,5) !!!
                  call get_index_table4(idx_table4, (/1,noa-1/), (/-1,noa/), (/1,nub/), (/1,nob/), noa, noa, nub, nob)
                  call sort4(r3b_excits, r3b_amps, loc_arr, idx_table4, (/3,4,2,5/), noa, noa, nub, nob, nloc, n3aab, resid)
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
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(i,j,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3b_excits(jdet,1);
                        ! compute < bc~ijk~ | h1a(vv) | ec~ijk~ >
                        hmatel = h1a_vv(b,e)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  ! diagram 4: h1b(ce)*r3b(beijk)
                  ! allocate new sorting arrays
                  nloc = noa*(noa - 1)/2*nob*nua
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(noa,noa,nob,nua))
                  !!! SB: (3,4,5,1) !!!
                  call get_index_table4(idx_table4, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/1,nua/), noa, noa, nob, nua)
                  call sort4(r3b_excits, r3b_amps, loc_arr, idx_table4, (/3,4,5,1/), noa, noa, nob, nua, nloc, n3aab, resid)
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
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(i,j,k,b)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3b_excits(jdet,2);
                        ! compute < bc~ijk~ | h1b(vv) | bf~ijk~ >
                        hmatel = h1b_vv(c,f)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  ! diagram 5: A(ij) h2b(mnjk)*r3b(bcimn)
                  ! allocate new sorting arrays
                  nloc = nua*nub*(noa - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nub,noa))
                  !!! SB: (1,2,3) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/1,nub/), (/1,noa-1/), nua, nub, noa)
                  call sort3(r3b_excits, r3b_amps, loc_arr, idx_table3, (/1,2,3/), nua, nub, noa, nloc, n3aab, resid)
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
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,i)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3b_excits(jdet,4); n = r3b_excits(jdet,5);
                        ! compute < bc~ijk~ | h2b(oooo) | bc~imn~ >
                        hmatel = h2b_oooo(m,n,j,k)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3b_excits(jdet,4); n = r3b_excits(jdet,5);
                        ! compute < bc~ijk~ | h2b(oooo) | bc~jmn~ >
                        hmatel = -h2b_oooo(m,n,i,k)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/1,nub/), (/2,noa/), nua, nub, noa)
                  call sort3(r3b_excits, r3b_amps, loc_arr, idx_table3, (/1,2,4/), nua, nub, noa, nloc, n3aab, resid)
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
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = r3b_excits(jdet,3); n = r3b_excits(jdet,5);
                        ! compute < bc~ijk~ | h2b(oooo) | bc~ljn~ >
                        hmatel = h2b_oooo(l,n,i,k)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table3(b,c,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = r3b_excits(jdet,3); n = r3b_excits(jdet,5);
                        ! compute < bc~ijk~ | h2b(oooo) | bc~lin~ >
                        hmatel = -h2b_oooo(l,n,j,k)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  ! diagram 6: 1/2 h2a(mnij)*r3b(bcmnk)
                  ! allocate new sorting arrays
                  nloc = nua*nub*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nub,nob))
                  !!! SB: (1,2,5) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/1,nub/), (/1,nob/), nua, nub, nob)
                  call sort3(r3b_excits, r3b_amps, loc_arr, idx_table3, (/1,2,5/), nua, nub, nob, nloc, n3aab, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp r3b_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_oooo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = r3b_excits(jdet,3); m = r3b_excits(jdet,4);
                        ! compute < bc~ijk~ | h2a(oooo) | bc~lmk~ >
                        hmatel = h2a_oooo(l,m,i,j)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  ! diagram 7: h2b(bcef)*r3b(efijk)
                  ! allocate new sorting arrays
                  nloc = noa*(noa - 1)/2*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(noa,noa,nob))
                  !!! SB: (3,4,5) !!!
                  call get_index_table3(idx_table3, (/1,noa-1/), (/-1,noa/), (/1,nob/), noa, noa, nob)
                  call sort3(r3b_excits, r3b_amps, loc_arr, idx_table3, (/3,4,5/), noa, noa, nob, nloc, n3aab, resid)
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
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3b_excits(jdet,1); f = r3b_excits(jdet,2);
                        ! compute < bc~ijk~ | h2a(oooo) | ef~ijk~ >
                        !hmatel = h2b_vvvv(b,c,e,f)
                        hmatel = h2b_vvvv(e,f,b,c)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  ! diagram 8: h2b(mcek)*r3a(beijm)
                  ! copy over r3a arrays
                  allocate(excits_buff(n3aaa,5),amps_buff(n3aaa))
                  excits_buff(:,:) = r3a_excits(:,:)
                  amps_buff(:) = r3a_amps(:)
                  ! allocate new sorting arrays
                  nloc = (noa - 1)*(noa - 2)/2*(nua - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(noa,noa,nua))
                  !!! SB: (3,4,1) !!!
                  call get_index_table3(idx_table3, (/1,noa-2/), (/-1,noa-1/), (/1,nua-1/), noa, noa, nua)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/3,4,1/), noa, noa, nua, nloc, n3aaa)
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
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
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
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
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
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
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
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
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
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
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
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
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

                  ! diagram 9: h2c(cmke)*r3b(beijm)
                  ! allocate new sorting arrays
                  nloc = noa*(noa - 1)/2*nua
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(noa,noa,nua))
                  !!! SB: (3,4,1) !!!
                  call get_index_table3(idx_table3, (/1,noa-1/), (/-1,noa/), (/1,nua/), noa, noa, nua)
                  call sort3(r3b_excits, r3b_amps, loc_arr, idx_table3, (/3,4,1/), noa, noa, nua, nloc, n3aab, resid)
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
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,j,b)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3b_excits(jdet,2); n = r3b_excits(jdet,5);
                        ! compute < bc~ijk~ | h2c(voov) | bf~ijn~ >
                        hmatel = h2c_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  ! diagram 10: h2a(bmje)*r3b(ecimk)
                  ! allocate new sorting arrays
                  nloc = (noa - 1)*nob*nua
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(noa,nob,nua))
                  !!! SB: (3,5,2) !!!
                  call get_index_table3(idx_table3, (/1,noa-1/), (/1,nob/), (/1,nua/), noa, nob, nua)
                  call sort3(r3b_excits, r3b_amps, loc_arr, idx_table3, (/3,5,2/), noa, nob, nua, nloc, n3aab, resid)
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
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,k,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3b_excits(jdet,1); m = r3b_excits(jdet,4);
                        ! compute < bc~ijk~ | h2c(voov) | ec~imk~ >
                        hmatel = h2a_voov(b,m,j,e)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table3(j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3b_excits(jdet,1); m = r3b_excits(jdet,4);
                        ! compute < bc~ijk~ | h2c(voov) | ec~jmk~ >
                        hmatel = -h2a_voov(b,m,i,e)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,2) !!!
                  call get_index_table3(idx_table3, (/2,noa/), (/1,nob/), (/1,nua/), noa, nob, nua)
                  call sort3(r3b_excits, r3b_amps, loc_arr, idx_table3, (/4,5,2/), noa, nob, nua, nloc, n3aab, resid)
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
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(j,k,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3b_excits(jdet,1); l = r3b_excits(jdet,3);
                        ! compute < bc~ijk~ | h2c(voov) | ec~ljk~ >
                        hmatel = h2a_voov(b,l,i,e)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table3(i,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3b_excits(jdet,1); l = r3b_excits(jdet,3);
                        ! compute < bc~ijk~ | h2c(voov) | ec~lik~ >
                        hmatel = -h2a_voov(b,l,j,e)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  ! diagram 11: A(ij) h2b(bmje)*r3c(ecimk)
                  ! copy over r3c arrays
                  allocate(excits_buff(n3abb,5),amps_buff(n3abb))
                  excits_buff(:,:) = r3c_excits(:,:)
                  amps_buff(:) = r3c_amps(:)
                  ! allocate new sorting arrays
                  nloc = (nub - 1)*noa*(nob - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(noa,nob,nub))
                  !!! SB: (3,5,2) !!!
                  call get_index_table3(idx_table3, (/1,noa/), (/2,nob/), (/2,nub/), noa, nob, nub)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/3,5,2/), noa, nob, nub, nloc, n3abb)
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
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
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
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
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
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
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
                  !$omp r3b_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
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

                  ! diagram 12: -A(ij) h2b(mcje)*r3b(beimk)
                  ! allocate new sorting arrays
                  nloc = nua*(noa - 1)*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(noa,nua,nob))
                  !!! SB: (3,1,5) !!!
                  call get_index_table3(idx_table3, (/1,noa-1/), (/1,nua/), (/1,nob/), noa, nua, nob)
                  call sort3(r3b_excits, r3b_amps, loc_arr, idx_table3, (/3,1,5/), noa, nua, nob, nloc, n3aab, resid)
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
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,b,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3b_excits(jdet,2); m = r3b_excits(jdet,4);
                        ! compute < bc~ijk~ | h2b(ovov) | bf~imk~ >
                        hmatel = -h2b_ovov(m,c,j,f)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table3(j,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3b_excits(jdet,2); m = r3b_excits(jdet,4);
                        ! compute < bc~ijk~ | h2b(ovov) | bf~jmk~ >
                        hmatel = h2b_ovov(m,c,i,f)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!! SB: (4,1,5) !!!
                  call get_index_table3(idx_table3, (/2,noa/), (/1,nua/), (/1,nob/), noa, nua, nob)
                  call sort3(r3b_excits, r3b_amps, loc_arr, idx_table3, (/4,1,5/), noa, nua, nob, nloc, n3aab, resid)
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
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(j,b,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3b_excits(jdet,2); l = r3b_excits(jdet,3);
                        ! compute < bc~ijk~ | h2b(ovov) | bf~ljk~ >
                        hmatel = -h2b_ovov(l,c,i,f)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table3(i,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3b_excits(jdet,2); l = r3b_excits(jdet,3);
                        ! compute < bc~ijk~ | h2b(ovov) | bf~lik~ >
                        hmatel = h2b_ovov(l,c,j,f)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  ! diagram 13: -h2b(bmek)*r3b(ecijm)
                  ! allocate new sorting arrays
                  nloc = noa*(noa - 1)/2*nub
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(noa,noa,nub))
                  !!! SB: (3,4,2) !!!
                  call get_index_table3(idx_table3, (/1,noa-1/), (/-1,noa/), (/1,nub/), noa, noa, nub)
                  call sort3(r3b_excits, r3b_amps, loc_arr, idx_table3, (/3,4,2/), noa, noa, nub, nloc, n3aab, resid)
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
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table3(i,j,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3b_excits(jdet,1); n = r3b_excits(jdet,5);
                        ! compute < bc~ijk~ | h2b(vovo) | ec~ijn~ >
                        hmatel = -h2b_vovo(b,n,e,k)
                        resid(idet) = resid(idet) + hmatel * r3b_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!!! diagram 14: A(ij) x2a_ovo(iem) t3b(ebc~mjk~)
                  ! copy over t3b arrays
                  allocate(excits_buff(m3aab,6),amps_buff(m3aab))
                  excits_buff(:,:) = t3b_excits(:,:)
                  amps_buff(:) = t3b_amps(:)
                  ! allocate new sorting arrays
                  nloc = (nua-1)*nub*(noa-1)*nob 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nub,noa,nob))
                  !!! SB: (2,3,5,6) !!!
                  call get_index_table4(idx_table4, (/2,nua/), (/1,nub/), (/2,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/2,3,5,6/), nua, nub, noa, nob, nloc, m3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2a_ovo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < bc~ijk~ | x2a(ovo) | ebc~mjk~ >
                        hmatel = x2a_ovo(i,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table4(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < bcijk | x2a(ovo) | ebcmik >
                        hmatel = -x2a_ovo(j,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,5,6) !!!
                  call get_index_table4(idx_table4, (/1,nua-1/), (/1,nub/), (/2,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/1,3,5,6/), nua, nub, noa, nob, nloc, m3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2a_ovo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < bc~ijk~ | x2a(ovo) | bec~mjk~ >
                        hmatel = -x2a_ovo(i,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table4(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < bc~ijk~ | x2a(ovo) | bec~mik~ >
                        hmatel = x2a_ovo(j,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4,6) !!!
                  call get_index_table4(idx_table4, (/2,nua/), (/1,nub/), (/1,noa-1/), (/1,nob/), nua, nub, noa, nob)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/2,3,4,6/), nua, nub, noa, nob, nloc, m3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2a_ovo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        ! compute < bc~ijk~ | x2a(ovo) | bec~mjk~ >
                        hmatel = -x2a_ovo(i,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table4(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        ! compute < bcijk | x2a(ovo) | ebcmik >
                        hmatel = x2a_ovo(j,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,4,6) !!!
                  call get_index_table4(idx_table4, (/1,nua-1/), (/1,nub/), (/1,noa-1/), (/1,nob/), nua, nub, noa, nob)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/1,3,4,6/), nua, nub, noa, nob, nloc, m3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2a_ovo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        ! compute < bc~ijk~ | x2a(ovo) | bec~mjk~ >
                        hmatel = x2a_ovo(i,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table4(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        ! compute < bcijk | x2a(ovo) | ebcmik >
                        hmatel = -x2a_ovo(j,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4,excits_buff,amps_buff)

                  !!!! diagram 15: A(ij) x2b_ovo(ie~m~) t3c(be~c~jm~k~)
                  ! copy over t3c arrays
                  allocate(excits_buff(m3abb,6),amps_buff(m3abb))
                  excits_buff(:,:) = t3c_excits(:,:)
                  amps_buff(:) = t3c_amps(:)
                  ! allocate new sorting arrays
                  nloc = nua*(nub-1)*noa*(nob-1) 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nub,noa,nob))
                  !!! SB: (1,3,4,6) !!!
                  call get_index_table4(idx_table4, (/1,nua/), (/2,nub/), (/1,noa/), (/2,nob/), nua, nub, noa, nob)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/1,3,4,6/), nua, nub, noa, nob, nloc, m3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2b_ovo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        ! compute < bc~ijk~ | x2b(ovo) | be~c~jm~k~ >
                        hmatel = x2b_ovo(i,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table4(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        ! compute < bc~ijk~ | x2b(ovo) | be~c~im~k~ >
                        hmatel = -x2b_ovo(j,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4,6) !!!
                  call get_index_table4(idx_table4, (/1,nua/), (/1,nub-1/), (/1,noa/), (/2,nob/), nua, nub, noa, nob)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/1,2,4,6/), nua, nub, noa, nob, nloc, m3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2b_ovo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        ! compute < bc~ijk~ | x2b(ovo) | bc~e~jm~k~ >
                        hmatel = -x2b_ovo(i,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table4(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        ! compute < bc~ijk~ | x2b(ovo) | bc~e~im~k~ >
                        hmatel = x2b_ovo(j,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nua/), (/2,nub/), (/1,noa/), (/1,nob-1/), nua, nub, noa, nob)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/1,3,4,5/), nua, nub, noa, nob, nloc, m3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2b_ovo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,6);
                        ! compute < bc~ijk~ | x2b(ovo) | be~c~jk~m~ >
                        hmatel = -x2b_ovo(i,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table4(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,6);
                        ! compute < bc~ijk~ | x2b(ovo) | be~c~ik~m~ >
                        hmatel = x2b_ovo(j,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nua/), (/1,nub-1/), (/1,noa/), (/1,nob-1/), nua, nub, noa, nob)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/1,2,4,5/), nua, nub, noa, nob, nloc, m3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2b_ovo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                        ! compute < bc~ijk~ | x2b(ovo) | bc~e~jk~m~ >
                        hmatel = x2b_ovo(i,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table4(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                        ! compute < bc~ijk~ | x2b(ovo) | bc~e~ik~m~ >
                        hmatel = -x2b_ovo(j,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4,excits_buff,amps_buff)

                  !!!! diagram 16: -x2b_voo(em~k~) t3b(ebc~ij~m~)
                  ! copy over t3b arrays
                  allocate(excits_buff(m3aab,6),amps_buff(m3aab))
                  excits_buff(:,:) = t3b_excits(:,:)
                  amps_buff(:) = t3b_amps(:)
                  ! allocate new sorting arrays
                  nloc = (nua-1)*nub*noa*(noa-1)/2 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(noa,noa,nua,nub))
                  !!! SB: (4,5,2,3) !!!
                  call get_index_table4(idx_table4, (/1,noa-1/), (/-1,noa/), (/2,nua/), (/1,nub/), noa, noa, nua, nub)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/4,5,2,3/), noa, noa, nua, nub, nloc, m3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2b_voo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(i,j,b,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,6);
                        ! compute < bc~ijk~ | x2b(voo) | ebc~ijm~ >
                        hmatel = -x2b_voo(e,m,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,1,3) !!!
                  call get_index_table4(idx_table4, (/1,noa-1/), (/-1,noa/), (/1,nua-1/), (/1,nub/), noa, noa, nua, nub)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/4,5,1,3/), noa, noa, nua, nub, nloc, m3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2b_voo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(i,j,b,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,6);
                        ! compute < bc~ijk~ | x2b(voo) | bec~ijm~ >
                        hmatel = x2b_voo(e,m,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4,excits_buff,amps_buff)

                  !!!! diagram 17: x2a_vvv(bfe) t3b(efc~ijk~)
                  !!!! diagram 19: x1a_v(e) t3b(ebc~ijk~) 
                  ! copy over t3b arrays
                  allocate(excits_buff(m3aab,6),amps_buff(m3aab))
                  excits_buff(:,:) = t3b_excits(:,:)
                  amps_buff(:) = t3b_amps(:)
                  ! allocate new sorting arrays
                  nloc = noa*(noa-1)/2*nob*nub
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(noa,noa,nob,nub))
                  !!! SB: (4,5,6,3) !!!
                  call get_index_table4(idx_table4, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/1,nub/), noa, noa, nob, nub)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/4,5,6,3/), noa, noa, nob, nub, nloc, m3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2a_vvv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(i,j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); f = excits_buff(jdet,2);
                        ! compute < bc~ijk~ | x2a(vvv) | efc~ijk~ >
                        hmatel = -x2a_vvv(b,e,f)
                        ! compute < bc~ijk~ | x1a(v) | efc~ijk~ >
                        hmatel1 = 0.0
                        if (b==e) hmatel1 = hmatel1 - x1a_v(f)
                        if (b==f) hmatel1 = hmatel1 + x1a_v(e)
                        hmatel = hmatel + hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4,excits_buff,amps_buff)

                  !!!! diagram 18: x2b_vvv(ec~f~) t3b(ebf~ijk~) 
                  ! copy over t3b arrays
                  allocate(excits_buff(m3aab,6),amps_buff(m3aab))
                  excits_buff(:,:) = t3b_excits(:,:)
                  amps_buff(:) = t3b_amps(:)
                  ! allocate new sorting arrays
                  nloc = noa*(noa-1)/2*nob*(nua-1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(noa,noa,nob,nua))
                  !!! SB: (4,5,6,2) !!!
                  call get_index_table4(idx_table4, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/2,nua/), noa, noa, nob, nua)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/4,5,6,2/), noa, noa, nob, nua, nloc, m3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2b_vvv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); f = excits_buff(jdet,3);
                        ! compute < bc~ijk~ | x2b(vvv) | ebf~ijk~ >
                        hmatel = x2b_vvv(e,c,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,6,1) !!!
                  call get_index_table4(idx_table4, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/1,nua-1/), noa, noa, nob, nua)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/4,5,6,1/), noa, noa, nob, nua, nloc, m3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2b_vvv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                     b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                     ! (1)
                     idx = idx_table4(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); f = excits_buff(jdet,3);
                        ! compute < bc~ijk~ | x2b(vvv) | bef~ijk~ >
                        hmatel = -x2b_vvv(e,c,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4,excits_buff,amps_buff)

                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits,&
                  !$omp t2a,t2b,r2a,r2b,&
                  !$omp h2a_vooo,h2b_vooo,h2b_ovoo,h2b_vvov,h2b_vvvo,&
                  !$omp x2a_ovv,x2a_ooo,x2b_ovv,x2b_vvo,x2b_ooo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp private(idet,a,b,c,d,i,j,k,l,m,n,e,f,res_mm23)
                  !$omp do schedule(static)
                  do idet = 1,n3aab
                      b = r3b_excits(idet,1); c = r3b_excits(idet,2);
                      i = r3b_excits(idet,3); j = r3b_excits(idet,4); k = r3b_excits(idet,5);
                      ! zero out value
                      res_mm23 = 0.0d0
                      do m = 1,noa
                         ! -A(ij) h2b(mcjk)*r2a(ibm)
                         res_mm23 = res_mm23 - h2b_ovoo(m,c,j,k) * r2a(i,b,m) ! (1)
                         res_mm23 = res_mm23 + h2b_ovoo(m,c,i,k) * r2a(j,b,m) ! (1)
                         ! -h2a(bmji)*r2b(mck)
                         res_mm23 = res_mm23 - h2a_vooo(b,m,j,i) * r2b(m,c,k) ! (1)
                         ! -x2a_ooo(imj)*t2b(bcmk)
                         res_mm23 = res_mm23 - x2a_ooo(i,m,j) * t2b(b,c,m,k) ! (1)
                      end do
                      do m = 1,nob
                         ! -A(ij) h2b(bmjk)*r2b(icm)
                         res_mm23 = res_mm23 - h2b_vooo(b,m,j,k) * r2b(i,c,m) ! (1)
                         res_mm23 = res_mm23 + h2b_vooo(b,m,i,k) * r2b(j,c,m) ! (1)
                         ! -A(ij) x2b_ooo(imk)*t2b(bcjm)
                         res_mm23 = res_mm23 - x2b_ooo(i,m,k) * t2b(b,c,j,m) ! (1)
                         res_mm23 = res_mm23 + x2b_ooo(j,m,k) * t2b(b,c,i,m) ! (ij)
                      end do
                      do e = 1,nua
                         ! h2b(bcek)*r2a(iej)
                         res_mm23 = res_mm23 + h2b_vvvo(b,c,e,k) * r2a(i,e,j) ! (1)
                         ! x2b_vvo(eck)*t2a(ebij)
                         res_mm23 = res_mm23 + x2b_vvo(e,c,k) * t2a(e,b,i,j) ! (1)
                         ! A(ij) x2a_ovv(ibe)*t2b(ecjk)
                         res_mm23 = res_mm23 + x2a_ovv(i,b,e) * t2b(e,c,j,k) ! (1)
                         res_mm23 = res_mm23 - x2a_ovv(j,b,e) * t2b(e,c,i,k) ! (1)
                      end do
                      do e = 1,nub
                         ! A(ij) h2b(bcje)*r2b(iek)
                         res_mm23 = res_mm23 + h2b_vvov(b,c,j,e) * r2b(i,e,k) ! (1)
                         res_mm23 = res_mm23 - h2b_vvov(b,c,i,e) * r2b(j,e,k) ! (ij)
                         ! A(ij) x2b_ovv(ice)*t2b(bejk)
                         res_mm23 = res_mm23 + x2b_ovv(i,c,e) * t2b(b,e,j,k) ! (1)
                         res_mm23 = res_mm23 - x2b_ovv(j,c,e) * t2b(b,e,i,k) ! (ij)
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
                                     t3c_amps,t3c_excits,&
                                     t3d_amps,t3d_excits,&
                                     h1a_oo,h1b_oo,h1b_vv,&
                                     h2b_oooo,h2b_ovov,h2b_ovvo,h2b_ovoo,&
                                     h2c_vvvv,h2c_oooo,h2c_voov,h2c_vooo,h2c_vvov,&
                                     x1a_v,&
                                     x2a_ovo,&
                                     x2b_ovv,x2b_vvo,x2b_ooo,&
                                     x2b_ovo,x2b_voo,x2b_vvv,&
                                     n3aab,n3abb,&
                                     m3abb,m3bbb,&
                                     noa,nua,nob,nub)

                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab, n3abb, m3abb, m3bbb
                  ! Input R and T arrays
                  real(kind=8), intent(in) :: t2b(nua,nub,noa,nob), t2c(nub,nub,nob,nob)
                  real(kind=8), intent(in) :: r2b(noa,nub,nob) 
                  integer, intent(in) :: r3b_excits(n3aab,5)
                  real(kind=8), intent(in) :: r3b_amps(n3aab)
                  real(kind=8), intent(in) :: t3c_amps(m3abb), t3d_amps(m3bbb)
                  integer, intent(in) :: t3c_excits(m3abb,6), t3d_excits(m3bbb,6)
                  ! Input H  and X arrays
                  real(kind=8), intent(in) :: h1a_oo(noa,noa)
                  real(kind=8), intent(in) :: h1b_oo(nob,nob)
                  real(kind=8), intent(in) :: h1b_vv(nub,nub)
                  real(kind=8), intent(in) :: h2b_oooo(noa,nob,noa,nob)
                  real(kind=8), intent(in) :: h2b_ovvo(noa,nub,nua,nob)
                  real(kind=8), intent(in) :: h2b_ovov(noa,nub,noa,nub)
                  real(kind=8), intent(in) :: h2b_ovoo(noa,nub,noa,nob)
                  real(kind=8), intent(in) :: h2c_vvvv(nub,nub,nub,nub)
                  real(kind=8), intent(in) :: h2c_oooo(nob,nob,nob,nob)
                  real(kind=8), intent(in) :: h2c_voov(nub,nob,nob,nub)
                  real(kind=8), intent(in) :: h2c_vooo(nub,nob,nob,nob)
                  real(kind=8), intent(in) :: h2c_vvov(nub,nub,nob,nub)
                  real(kind=8), intent(in) :: x1a_v(nua)
                  real(kind=8), intent(in) :: x2b_ovv(noa,nub,nub)
                  real(kind=8), intent(in) :: x2b_vvo(nua,nub,nob)
                  real(kind=8), intent(in) :: x2b_ooo(noa,nob,nob)
                  real(kind=8), intent(in) :: x2a_ovo(noa,nua,noa)
                  real(kind=8), intent(in) :: x2b_ovo(noa,nub,nob)
                  real(kind=8), intent(in) :: x2b_voo(nua,nob,nob)
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
                  real(kind=8) :: r_amp, hmatel, res_mm23, hmatel1
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc
                  

                  ! Zero the container that holds H*R
                  resid = 0.0d0

                  ! diagram 1: -A(jk) h1b_oo(mj)*r3c(bcimk) 
                  ! allocate new sorting arrays
                  nloc = nub*(nub - 1)/2*noa*(nob - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nub,nub,noa,nob))
                  !!! SB: (1,2,3,5) !!!
                  call get_index_table4(idx_table4, (/1,nub-1/), (/-1,nub/), (/1,noa/), (/2,nob/), nub, nub, noa, nob)
                  call sort4(r3c_excits, r3c_amps, loc_arr, idx_table4, (/1,2,3,5/), nub, nub, noa, nob, nloc, n3abb, resid)
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
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,i,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3c_excits(jdet,4);
                        ! compute < b~c~ij~k~ | h1b(oo) | b~c~im~k~ >
                        hmatel = -h1b_oo(m,j)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table4(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3c_excits(jdet,4);
                        ! compute < b~c~ij~k~ | h1b(oo) | b~c~im~j~ >
                        hmatel = h1b_oo(m,k)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,3,4) !!!
                  call get_index_table4(idx_table4, (/1,nub-1/), (/-1,nub/), (/1,noa/), (/1,nob-1/), nub, nub, noa, nob)
                  call sort4(r3c_excits, r3c_amps, loc_arr, idx_table4, (/1,2,3,4/), nub, nub, noa, nob, nloc, n3abb, resid)
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
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,i,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        n = r3c_excits(jdet,5);
                        ! compute < b~c~ij~k~ | h1b(oo) | b~c~ij~n~ >
                        hmatel = -h1b_oo(n,k)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table4(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        n = r3c_excits(jdet,5);
                        ! compute < b~c~ij~k~ | h1b(oo) | b~c~ik~n~>
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

                  ! diagram 2: -h1a_oo(mi)*r3c(bcmjk) 
                  ! allocate new sorting arrays
                  nloc = nub*(nub - 1)/2*nob*(nob - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nub,nub,nob,nob))
                  !!! SB: (1,2,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nub-1/), (/-1,nub/), (/1,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(r3c_excits, r3c_amps, loc_arr, idx_table4, (/1,2,4,5/), nub, nub, nob, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_oo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = r3c_excits(jdet,3);
                        ! compute < b~c~ij~k~ | h1a(oo) | b~c~lj~k~ >
                        hmatel = -h1a_oo(l,i)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  ! diagram 3: A(bc) h1b(be)*r3c(ecijk)
                  ! allocate new sorting arrays
                  nloc = noa*nob*(nob - 1)/2*(nub - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nob,nob,noa,nub))
                  !!! SB: (4,5,3,2) !!!
                  call get_index_table4(idx_table4, (/1,nob-1/), (/-1,nob/), (/1,noa/), (/2,nub/), nob, nob, noa, nub)
                  call sort4(r3c_excits, r3c_amps, loc_arr, idx_table4, (/4,5,3,2/), nob, nob, noa, nub, nloc, n3abb, resid)
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
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(j,k,i,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits(jdet,1);
                        ! compute < b~c~ij~k~ | h1b(vv) | e~c~ij~k~ >
                        hmatel = h1b_vv(b,e)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table4(j,k,i,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits(jdet,1);
                        ! compute < b~c~ij~k~ | h1b(vv) | e~b~ij~k~ >
                        hmatel = -h1b_vv(c,e)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,3,1) !!!
                  call get_index_table4(idx_table4, (/1,nob-1/), (/-1,nob/), (/1,noa/), (/1,nub-1/), nob, nob, noa, nub)
                  call sort4(r3c_excits, r3c_amps, loc_arr, idx_table4, (/4,5,3,1/), nob, nob, noa, nub, nloc, n3abb, resid)
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
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(j,k,i,b)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits(jdet,2);
                        ! compute < b~c~ij~k~ | h1b(vv) | b~f~ij~k~ >
                        hmatel = h1b_vv(c,f)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table4(j,k,i,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits(jdet,2);
                        ! compute < b~c~ij~k~ | h1b(vv) | c~f~ij~k~ >
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
                  
                  ! diagram 4: 1/2 h2c(mnjk)*r3c(bcimn)
                  ! allocate new sorting arrays
                  nloc = nub*(nub - 1)/2*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nub,nub,noa))
                  !!! SB: (1,2,3) !!!
                  call get_index_table3(idx_table3, (/1,nub-1/), (/-1,nub/), (/1,noa/), nub, nub, noa)
                  call sort3(r3c_excits, r3c_amps, loc_arr, idx_table3, (/1,2,3/), nub, nub, noa, nloc, n3abb, resid)
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
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,i)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3c_excits(jdet,4); n = r3c_excits(jdet,5);
                        ! compute < b~c~ij~k~ | h2c(oooo) | b~c~im~n~ >
                        hmatel = h2c_oooo(m,n,j,k)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  ! diagram 5: A(jk) h2b(mnij)*r3c(bcmnk)
                  ! allocate new sorting arrays
                  nloc = nub*(nub - 1)/2*(nob - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nub,nub,nob))
                  !!! SB: (1,2,5) !!!
                  call get_index_table3(idx_table3, (/1,nub-1/), (/-1,nub/), (/2,nob/), nub, nub, nob)
                  call sort3(r3c_excits, r3c_amps, loc_arr, idx_table3, (/1,2,5/), nub, nub, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_oooo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = r3c_excits(jdet,3); m = r3c_excits(jdet,4);
                        ! compute < b~c~ij~k~ | h2b(oooo) | b~c~lm~k~ >
                        hmatel = h2b_oooo(l,m,i,j)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = r3c_excits(jdet,3); m = r3c_excits(jdet,4);
                        ! compute < b~c~ij~k~ | h2b(oooo) | b~c~lm~j~ >
                        hmatel = -h2b_oooo(l,m,i,k)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nub-1/), (/-1,nub/), (/1,nob-1/), nub, nub, nob)
                  call sort3(r3c_excits, r3c_amps, loc_arr, idx_table3, (/1,2,4/), nub, nub, nob, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_oooo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = r3c_excits(jdet,3); n = r3c_excits(jdet,5);
                        ! compute < b~c~ij~k~ | h2b(oooo) | b~c~lj~n~ >
                        hmatel = h2b_oooo(l,n,i,k)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        l = r3c_excits(jdet,3); n = r3c_excits(jdet,5);
                        ! compute < b~c~ij~k~ | h2b(oooo) | b~c~lk~n~ >
                        hmatel = -h2b_oooo(l,n,i,j)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  ! diagram 6: 1/2 h2c(bcef)*r3c(efijk)
                  ! allocate new sorting arrays
                  nloc = nob*(nob - 1)/2*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nob,nob,noa))
                  !!! SB: (4,5,3) !!!
                  call get_index_table3(idx_table3, (/1,nob-1/), (/-1,nob/), (/1,noa/), nob, nob, noa)
                  call sort3(r3c_excits, r3c_amps, loc_arr, idx_table3, (/4,5,3/), nob, nob, noa, nloc, n3abb, resid)
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
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(j,k,i)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits(jdet,1); f = r3c_excits(jdet,2);
                        ! compute < b~c~ij~k~ | h2c(vvvv) | e~f~ij~k~ >
                        !hmatel = h2c_vvvv(b,c,e,f)
                        hmatel = h2c_vvvv(e,f,b,c)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  ! diagram 7: A(jk)A(bc) h2b(mbej)*r3b(ecimk)
                  ! copy over r3b arrays
                  allocate(excits_buff(n3aab,5),amps_buff(n3aab))
                  excits_buff(:,:) = r3b_excits(:,:)
                  amps_buff(:) = r3b_amps(:)
                  ! allocate new sorting arrays
                  nloc = nub*(noa - 1)*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nub,noa,nob))
                  !!! SB: (2,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nub/), (/1,noa-1/), (/1,nob/), nub, noa, nob)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/2,3,5/), nub, noa, nob, nloc, n3aab)
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
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
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
                  !$omp r3c_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
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

                  ! diagram 8: A(jk)A(bc) h2c(bmje)*r3c(ecimk)
                  ! allocate new sorting arrays
                  nloc = (nob - 1)*noa*(nub - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nub,nob,noa))
                  !!! SB: (2,5,3) !!!
                  call get_index_table3(idx_table3, (/2,nub/), (/2,nob/), (/1,noa/), nub, nob, noa)
                  call sort3(r3c_excits, r3c_amps, loc_arr, idx_table3, (/2,5,3/), nub, nob, noa, nloc, n3abb, resid)
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
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(c,k,i)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits(jdet,1); m = r3c_excits(jdet,4);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~c~im~k~ >
                        hmatel = h2c_voov(b,m,j,e)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table3(c,j,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits(jdet,1); m = r3c_excits(jdet,4);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~c~im~j~ >
                        hmatel = -h2c_voov(b,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(b,k,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits(jdet,1); m = r3c_excits(jdet,4);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~b~im~k~ >
                        hmatel = -h2c_voov(c,m,j,e)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                     ! (jk)(bc)
                     idx = idx_table3(b,j,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits(jdet,1); m = r3c_excits(jdet,4);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~b~im~j~ >
                        hmatel = h2c_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,5,3) !!!
                  call get_index_table3(idx_table3, (/1,nub-1/), (/2,nob/), (/1,noa/), nub, nob, noa)
                  call sort3(r3c_excits, r3c_amps, loc_arr, idx_table3, (/1,5,3/), nub, nob, noa, nloc, n3abb, resid)
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
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,k,i)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits(jdet,2); m = r3c_excits(jdet,4);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~c~im~k~ >
                        hmatel = h2c_voov(c,m,j,f)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table3(b,j,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits(jdet,2); m = r3c_excits(jdet,4);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~c~im~j~ >
                        hmatel = -h2c_voov(c,m,k,f)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(c,k,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits(jdet,2); m = r3c_excits(jdet,4);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~b~im~k~ >
                        hmatel = -h2c_voov(b,m,j,f)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                     ! (jk)(bc)
                     idx = idx_table3(c,j,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits(jdet,2); m = r3c_excits(jdet,4);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~b~im~j~ >
                        hmatel = h2c_voov(b,m,k,f)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,4,3) !!!
                  call get_index_table3(idx_table3, (/2,nub/), (/1,nob-1/), (/1,noa/), nub, nob, noa)
                  call sort3(r3c_excits, r3c_amps, loc_arr, idx_table3, (/2,4,3/), nub, nob, noa, nloc, n3abb, resid)
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
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(c,j,i)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits(jdet,1); n = r3c_excits(jdet,5);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~c~im~k~ >
                        hmatel = h2c_voov(b,n,k,e)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table3(c,k,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits(jdet,1); n = r3c_excits(jdet,5);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~c~im~j~ >
                        hmatel = -h2c_voov(b,n,j,e)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(b,j,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits(jdet,1); n = r3c_excits(jdet,5);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~b~im~k~ >
                        hmatel = -h2c_voov(c,n,k,e)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                     ! (jk)(bc)
                     idx = idx_table3(b,k,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits(jdet,1); n = r3c_excits(jdet,5);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~b~im~j~ >
                        hmatel = h2c_voov(c,n,j,e)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,4,3) !!!
                  call get_index_table3(idx_table3, (/1,nub-1/), (/1,nob-1/), (/1,noa/), nub, nob, noa)
                  call sort3(r3c_excits, r3c_amps, loc_arr, idx_table3, (/1,4,3/), nub, nob, noa, nloc, n3abb, resid)
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
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(b,j,i)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits(jdet,2); n = r3c_excits(jdet,5);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~c~im~k~ >
                        hmatel = h2c_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table3(b,k,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits(jdet,2); n = r3c_excits(jdet,5);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~c~im~j~ >
                        hmatel = -h2c_voov(c,n,j,f)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(c,j,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits(jdet,2); n = r3c_excits(jdet,5);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~b~im~k~ >
                        hmatel = -h2c_voov(b,n,k,f)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                     ! (jk)(bc)
                     idx = idx_table3(c,k,i)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits(jdet,2); n = r3c_excits(jdet,5);
                        ! compute < b~c~ij~k~ | h2c(voov) | e~b~im~j~ >
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

                  ! diagram 9: -A(bc) h2b(mbie)*r3c(ecmjk)
                  ! allocate new sorting arrays
                  nloc = nob*(nob - 1)/2*(nub - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nob,nob,nub))
                  !!! SB: (4,5,2) !!!
                  call get_index_table3(idx_table3, (/1,nob/), (/-1,nob/), (/2,nub/), nob, nob, nub)
                  call sort3(r3c_excits, r3c_amps, loc_arr, idx_table3, (/4,5,2/), nob, nob, nub, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(j,k,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits(jdet,1); l = r3c_excits(jdet,3);
                        ! compute < b~c~ij~k~ | h2b(ovov) | e~c~lj~k~ >
                        hmatel = -h2b_ovov(l,b,i,e)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table3(j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits(jdet,1); l = r3c_excits(jdet,3);
                        ! compute < b~c~ij~k~ | h2b(ovov) | e~b~lj~k~ >
                        hmatel = h2b_ovov(l,c,i,e)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,1) !!!
                  call get_index_table3(idx_table3, (/1,nob/), (/-1,nob/), (/1,nub-1/), nob, nob, nub)
                  call sort3(r3c_excits, r3c_amps, loc_arr, idx_table3, (/4,5,1/), nob, nob, nub, nloc, n3abb, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp r3c_amps,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table3(j,k,b)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits(jdet,2); l = r3c_excits(jdet,3);
                        ! compute < b~c~ij~k~ | h2b(ovov) | b~f~lj~k~ >
                        hmatel = -h2b_ovov(l,c,i,f)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table3(j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits(jdet,2); l = r3c_excits(jdet,3);
                        ! compute < b~c~ij~k~ | h2b(ovov) | c~f~lj~k~ >
                        hmatel = h2b_ovov(l,b,i,f)
                        resid(idet) = resid(idet) + hmatel * r3c_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!!! diagram 10: x2a_ovo(iem) t3c(eb~c~mj~k~)
                  ! copy over t3c arrays
                  allocate(excits_buff(m3abb,6),amps_buff(m3abb))
                  excits_buff(:,:) = t3c_excits(:,:)
                  amps_buff(:) = t3c_amps(:)
                  ! allocate new sorting arrays
                  nloc = nub*(nub-1)/2*nob*(nob-1)/2 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nub,nub,nob,nob))
                  !!! SB: (2,3,5,6) !!!
                  call get_index_table4(idx_table4, (/1,nub-1/), (/-1,nub/), (/1,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/2,3,5,6/), nub, nub, nob, nob, nloc, m3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2a_ovo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < b~c~ij~k~ | x2a(ovo) | eb~c~mj~k~ >
                        hmatel = x2a_ovo(i,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4,excits_buff,amps_buff)

                  !!!! diagram 11: x2b_ovo(ie~m~) t3d(e~b~c~m~j~k~)
                  ! copy over t3d arrays
                  allocate(excits_buff(m3bbb,6),amps_buff(m3bbb))
                  excits_buff(:,:) = t3d_excits(:,:)
                  amps_buff(:) = t3d_amps(:)
                  ! allocate new sorting arrays
                  nloc = (nub-1)*(nub-2)/2*(nob-1)*(nob-2)/2 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nub,nub,nob,nob))
                  !!! SB: (2,3,5,6) !!!
                  call get_index_table4(idx_table4, (/2,nub-1/), (/-1,nub/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/2,3,5,6/), nub, nub, nob, nob, nloc, m3bbb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2b_ovo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < b~c~ij~k~ | x2b(ovo) | e~b~c~m~j~k~ >
                        hmatel = x2b_ovo(i,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4,5) !!!
                  call get_index_table4(idx_table4, (/2,nub-1/), (/-1,nub/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/2,3,4,5/), nub, nub, nob, nob, nloc, m3bbb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2b_ovo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,6);
                        ! compute < b~c~ij~k~ | x2b(ovo) | e~b~c~j~k~m~ >
                        hmatel = x2b_ovo(i,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4,6) !!!
                  call get_index_table4(idx_table4, (/2,nub-1/), (/-1,nub/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/2,3,4,6/), nub, nub, nob, nob, nloc, m3bbb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2b_ovo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        ! compute < b~c~ij~k~ | x2b(ovo) | e~b~c~j~m~k~ >
                        hmatel = -x2b_ovo(i,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,5,6) !!!
                  call get_index_table4(idx_table4, (/1,nub-2/), (/-1,nub-1/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/1,2,5,6/), nub, nub, nob, nob, nloc, m3bbb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2b_ovo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,4);
                        ! compute < b~c~ij~k~ | x2b(ovo) | b~c~e~m~j~k~ >
                        hmatel = x2b_ovo(i,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nub-2/), (/-1,nub-1/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/1,2,4,5/), nub, nub, nob, nob, nloc, m3bbb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2b_ovo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                        ! compute < b~c~ij~k~ | x2b(ovo) | b~c~e~j~k~m~ >
                        hmatel = x2b_ovo(i,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4,6) !!!
                  call get_index_table4(idx_table4, (/1,nub-2/), (/-1,nub-1/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/1,2,4,6/), nub, nub, nob, nob, nloc, m3bbb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2b_ovo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        ! compute < b~c~ij~k~ | x2b(ovo) | b~c~e~j~m~k~ >
                        hmatel = -x2b_ovo(i,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,5,6) !!!
                  call get_index_table4(idx_table4, (/1,nub-2/), (/-2,nub/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/1,3,5,6/), nub, nub, nob, nob, nloc, m3bbb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2b_ovo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < b~c~ij~k~ | x2b(ovo) | b~e~c~m~j~k~ >
                        hmatel = -x2b_ovo(i,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nub-2/), (/-2,nub/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/1,3,4,5/), nub, nub, nob, nob, nloc, m3bbb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2b_ovo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,6);
                        ! compute < b~c~ij~k~ | x2b(ovo) | b~e~c~j~k~m~ >
                        hmatel = -x2b_ovo(i,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,4,6) !!!
                  call get_index_table4(idx_table4, (/1,nub-2/), (/-2,nub/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/1,3,4,6/), nub, nub, nob, nob, nloc, m3bbb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2b_ovo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        ! compute < b~c~ij~k~ | x2b(ovo) | b~e~c~j~m~k~ >
                        hmatel = x2b_ovo(i,e,m)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4,excits_buff,amps_buff)

                  !!!! diagram 12: A(j~k~) -x2b_voo(em~j~) t3c(eb~c~im~k~)
                  ! copy over t3c arrays
                  allocate(excits_buff(m3abb,6),amps_buff(m3abb))
                  excits_buff(:,:) = t3c_excits(:,:)
                  amps_buff(:) = t3c_amps(:)
                  ! allocate new sorting arrays
                  nloc = nub*(nub-1)/2*noa*(nob-1) 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nub,nub,noa,nob))
                  !!! SB: (2,3,4,6) !!!
                  call get_index_table4(idx_table4, (/1,nub-1/), (/-1,nub/), (/1,noa/), (/2,nob/), nub, nub, noa, nob)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/2,3,4,6/), nub, nub, noa, nob, nloc, m3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2b_voo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        ! compute < b~c~ij~k~ | x2b(voo) | eb~c~im~k~ >
                        hmatel = -x2b_voo(e,m,j)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table4(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        ! compute < b~c~ij~k~ | x2b(voo) | eb~c~im~j~ >
                        hmatel = x2b_voo(e,m,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nub-1/), (/-1,nub/), (/1,noa/), (/1,nob-1/), nub, nub, noa, nob)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/2,3,4,5/), nub, nub, noa, nob, nloc, m3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2b_voo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,6);
                        ! compute < b~c~ij~k~ | x2b(voo) | eb~c~ij~m~ >
                        hmatel = -x2b_voo(e,m,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table4(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,6);
                        ! compute < b~c~ij~k~ | x2b(voo) | eb~c~im~j~ >
                        hmatel = x2b_voo(e,m,j)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4,excits_buff,amps_buff)

                  !!!! diagram 13: A(b~c~) x2b_vvv(eb~f~) t3c(ef~c~ij~k~)
                  !!!! diagram 14: x1a_v(e) t3c(eb~c~ij~k~)
                  ! copy over t3c arrays
                  allocate(excits_buff(m3abb,6),amps_buff(m3abb))
                  excits_buff(:,:) = t3c_excits(:,:)
                  amps_buff(:) = t3c_amps(:)
                  ! allocate new sorting arrays
                  nloc = noa*nob*(nob-1)/2*(nub-1) 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nob,nob,noa,nub))
                  !!! SB: (5,6,4,3) !!!
                  call get_index_table4(idx_table4, (/1,nob-1/), (/-1,nob/), (/1,noa/), (/2,nub/), nob, nob, noa, nub)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/5,6,4,3/), nob, nob, noa, nub, nloc, m3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2b_vvv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(j,k,i,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); f = excits_buff(jdet,2);
                        ! compute < b~c~ij~k~ | x2b(vvv) | ef~c~ij~k~ >
                        hmatel = x2b_vvv(e,b,f)
                        ! compute < b~c~ij~k~ | x1a(v) | ef~c~ij~k~ >
                        hmatel1 = 0.0d0
                        if (b==f) hmatel1 = hmatel1 + x1a_v(e)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table4(j,k,i,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); f = excits_buff(jdet,2);
                        ! compute < b~c~ij~k~ | x2b(vvv) | ef~b~ij~k~ >
                        hmatel = -x2b_vvv(e,c,f)
                        ! compute < b~c~ij~k~ | x1a(v) | ef~b~ij~k~ >
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 - x1a_v(e)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (5,6,4,2) !!!
                  call get_index_table4(idx_table4, (/1,nob-1/), (/-1,nob/), (/1,noa/), (/1,nub-1/), nob, nob, noa, nub)
                  call sort4t(excits_buff, amps_buff, loc_arr, idx_table4, (/5,6,4,2/), nob, nob, noa, nub, nloc, m3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table4,&
                  !$omp x2b_vvv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                     b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                     ! (1)
                     idx = idx_table4(j,k,i,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); f = excits_buff(jdet,3);
                        ! compute < b~c~ij~k~ | x2b(vvv) | eb~f~ij~k~ >
                        hmatel = x2b_vvv(e,c,f)
                        ! compute < b~c~ij~k~ | x1a(v) | eb~f~ij~k~ >
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 + x1a_v(e)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table4(j,k,i,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,1); f = excits_buff(jdet,3);
                        ! compute < b~c~ij~k~ | x2b(vvv) | ec~f~ij~k~ >
                        hmatel = -x2b_vvv(e,b,f)
                        ! compute < b~c~ij~k~ | x1a(v) | ec~f~ij~k~ >
                        hmatel1 = 0.0d0
                        if (b==f) hmatel1 = hmatel1 - x1a_v(e)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4,excits_buff,amps_buff)

                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits,&
                  !$omp t2b,t2c,r2b,&
                  !$omp h2b_ovoo,&
                  !$omp h2c_vooo,h2c_vvov,&
                  !$omp x2b_ovv,x2b_vvo,x2b_ooo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp private(idet,a,b,c,d,i,j,k,l,m,n,e,f,res_mm23)
                  !$omp do schedule(static)
                  do idet = 1,n3abb
                      b = r3c_excits(idet,1); c = r3c_excits(idet,2);
                      i = r3c_excits(idet,3); j = r3c_excits(idet,4); k = r3c_excits(idet,5);
                      ! zero out value
                      res_mm23 = 0.0d0
                      do m = 1,noa
                         ! -A(jk)A(bc) h2b(mcik)*r2b(mbj)
                         res_mm23 = res_mm23 - h2b_ovoo(m,c,i,k) * r2b(m,b,j) ! (1)
                         res_mm23 = res_mm23 + h2b_ovoo(m,c,i,j) * r2b(m,b,k) ! (jk)
                         res_mm23 = res_mm23 + h2b_ovoo(m,b,i,k) * r2b(m,c,j) ! (bc)
                         res_mm23 = res_mm23 - h2b_ovoo(m,b,i,j) * r2b(m,c,k) ! (jk)(bc)
                      end do
                      do m = 1,nob
                         ! -A(bc) h2c(cmkj)*r2b(ibm)
                         res_mm23 = res_mm23 - h2c_vooo(c,m,k,j) * r2b(i,b,m) ! (1)
                         res_mm23 = res_mm23 + h2c_vooo(b,m,k,j) * r2b(i,c,m) ! (bc)
                         ! -A(jk) x2b_ooo(imj)*t2c(bcmk)
                         res_mm23 = res_mm23 - x2b_ooo(i,m,j) * t2c(b,c,m,k) ! (1)
                         res_mm23 = res_mm23 + x2b_ooo(i,m,k) * t2c(b,c,m,j) ! (jk)
                      end do
                      do e = 1,nua
                         ! A(jk)A(bc) x2b_vvo(ebj)*t2b(ecik)
                         res_mm23 = res_mm23 + x2b_vvo(e,b,j) * t2b(e,c,i,k) ! (1)
                         res_mm23 = res_mm23 - x2b_vvo(e,b,k) * t2b(e,c,i,j) ! (jk)
                         res_mm23 = res_mm23 - x2b_vvo(e,c,j) * t2b(e,b,i,k) ! (bc)
                         res_mm23 = res_mm23 + x2b_vvo(e,c,k) * t2b(e,b,i,j) ! (jk)(bc)
                      end do
                      do e = 1,nub
                         ! A(jk) h2c(cbke)*r2b(iej)
                         res_mm23 = res_mm23 + h2c_vvov(c,b,k,e) * r2b(i,e,j) ! (1)
                         res_mm23 = res_mm23 - h2c_vvov(c,b,j,e) * r2b(i,e,k) ! (jk)
                         ! A(bc) x2b_ovv(ibe)*t2c(ecjk)
                         res_mm23 = res_mm23 + x2b_ovv(i,b,e) * t2c(e,c,j,k) ! (1)
                         res_mm23 = res_mm23 - x2b_ovv(i,c,e) * t2c(e,b,j,k) ! (bc)
                      end do
                      resid(idet) = resid(idet) + res_mm23
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
              end subroutine build_hr_3c

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

              subroutine sort4t(excits, amps, loc_arr, idx_table, idims, n1, n2, n3, n4, nloc, n3p, x1a)

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
              end subroutine sort4t

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

end module ipeomt_p_loops
