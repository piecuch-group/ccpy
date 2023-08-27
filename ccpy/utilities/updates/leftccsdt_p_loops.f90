module leftccsdt_p_loops

        implicit none

        contains

!    LH.aaa += (9.0 / 36.0) * np.einsum("eima,ebcmjk->abcijk", H.aa.voov, L.aaa, optimize=True)

              subroutine update_l3a(resid,&
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

              end subroutine update_l3a
           
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
