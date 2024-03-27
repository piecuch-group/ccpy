module ipeom3_p_loops

      use omp_lib

      implicit none

      contains

              subroutine build_hr_3a(resid,&
                                     r2a,&
                                     r3a_amps,r3a_excits,&
                                     r3b_amps,r3b_excits,&
                                     t2a,&
                                     h1a_oo,h1a_vv,&
                                     h2a_vvvv,h2a_oooo,h2a_voov,h2a_vooo,h2a_vvov,&
                                     h2b_voov,&
                                     x2a_ovv,x2a_ooo,&
                                     n3aaa,n3aab,&
                                     noa,nua,nob,nub)

                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa, n3aab
                  ! Input R and T arrays
                  real(kind=8), intent(in) :: r2a(noa,nua,noa), t2a(nua,nua,noa,noa)
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
                  real(kind=8), intent(in) :: x2a_ovv(noa,nua,nua)
                  real(kind=8), intent(in) :: x2a_ooo(noa,noa,noa)
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
                        hmatel = h2a_vvvv(b,c,e,f)
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

end module ipeom3_p_loops
