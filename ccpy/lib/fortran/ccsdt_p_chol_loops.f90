module ccsdt_p_chol_loops

      use omp_lib
      implicit none

      contains
              subroutine update_t3a_p(resid,&
                                      t3a_amps, t3a_excits,&
                                      t3b_amps, t3b_excits,&
                                      t2a,&
                                      H1A_oo, H1A_vv,&
                                      H2A_oovv, H2A_vvov, H2A_vooo,&
                                      H2A_oooo, H2A_voov, chol_a_vv,&
                                      H2B_oovv, H2B_voov,&
                                      fA_oo, fA_vv,&
                                      shift,&
                                      n3aaa, n3aab,&
                                      noa, nua, nob, nub, nchol)

                  integer, intent(in) :: noa, nua, nob, nub, n3aaa, n3aab, nchol
                  integer, intent(in) :: t3b_excits(n3aab,6)
                  real(kind=8), intent(in) :: t2a(nua,nua,noa,noa),&
                                              H1A_oo(noa,noa), H1A_vv(nua,nua),&
                                              H2A_oovv(noa,noa,nua,nua),&
                                              H2B_oovv(noa,nob,nua,nub),&
                                              !H2A_vvov(nua,nua,noa,nua),&
                                              H2A_vvov(nua,nua,nua,noa),& ! reordered
                                              !H2A_vooo(nua,noa,noa,noa),&
                                              H2A_vooo(noa,nua,noa,noa),& ! reordered
                                              H2A_oooo(noa,noa,noa,noa),&
                                              !H2A_voov(nua,noa,noa,nua),&
                                              H2A_voov(noa,nua,nua,noa),& ! reordered
                                              !H2A_vvvv_buff(nua,nua,nua,nua),&
                                              chol_a_vv(nchol,nua,nua),&
                                              !H2B_voov(nua,nob,noa,nub),&
                                              H2B_voov(nob,nub,nua,noa),& ! reordered
                                              fA_vv(nua,nua), fA_oo(noa,noa),&
                                              shift
                  real(kind=8), intent(in) :: t3b_amps(n3aab)

                  integer, intent(inout) :: t3a_excits(n3aaa,6)
                  !f2py intent(in,out) :: t3a_excits(0:n3aaa-1,0:5)
                  real(kind=8), intent(inout) :: t3a_amps(n3aaa)
                  !f2py intent(in,out) :: t3a_amps(0:n3aaa-1)

                  real(kind=8), intent(out) :: resid(n3aaa)

                  integer, allocatable :: idx_table(:,:,:,:), idx_table_copy1(:,:,:,:), idx_table_copy2(:,:,:,:), idx_table_copy3(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:), loc_arr_copy1(:,:), loc_arr_copy2(:,:), loc_arr_copy3(:,:)

                  real(kind=8), allocatable :: t3a_amps_copy1(:), t3a_amps_copy2(:), t3a_amps_copy3(:)
                  integer, allocatable :: t3a_excits_copy1(:,:), t3a_excits_copy2(:,:), t3a_excits_copy3(:,:)
                  real(kind=8), allocatable :: t3_amps_buff(:), xbuf(:,:,:,:)
                  integer, allocatable :: t3_excits_buff(:,:)

                  !real(kind=8) :: I2A_vvov(nua,nua,noa,nua)
                  real(kind=8) :: I2A_vvov(nua,nua,nua,noa) ! reordered
                  !real(kind=8) :: I2A_vooo(nua, noa, noa, noa)
                  real(kind=8) :: I2A_vooo(noa,nua,noa,noa) ! reordered
                  real(kind=8) :: H2A_vvvv(nua,nua)
                  real(kind=8) :: val, denom, t_amp, res_mm23, hmatel
                  real(kind=8) :: hmatel1, hmatel2, hmatel3, hmatel4
                  integer :: a, b, c, d, i, j, k, l, e, f, m, n, k1, idet, jdet
                  integer :: a_chol, b_chol
                  integer :: idx, nloc
                  integer :: ab_table(nua*(nua-1)/2,2)

                  ! Start the VT3 intermediates at Hbar (factor of 1/2 to compensate for antisymmetrization)
                  I2A_vooo = 0.5d0 * H2A_vooo
                  I2A_vvov = 0.5d0 * H2A_vvov
                  call calc_I2A_vooo(I2A_vooo,&
                               H2A_oovv,H2B_oovv,&
                               t3a_excits,t3a_amps,t3b_excits,t3b_amps,&
                               n3aaa,n3aab,noa,nua,nob,nub)
                  call calc_I2A_vvov(I2A_vvov,&
                               H2A_oovv,H2B_oovv,&
                               t3a_excits,t3a_amps,t3b_excits,t3b_amps,&
                               n3aaa,n3aab,noa,nua,nob,nub)

                  ! Zero the residual container
                  resid = 0.0d0

                  !!!! diagram 1: -A(i/jk) h1a(mi) * t3a(abcmjk)
                  !!!! diagram 3: 1/2 A(i/jk) h2a(mnij) * t3a(abcmnk)
                  ! NOTE: WITHIN THESE LOOPS, H1A(OO) TERMS ARE DOUBLE-COUNTED SO COMPENSATE BY FACTOR OF 1/2
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)*(nua-2)/6*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,nua,noa))
                  !!! ABCK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/3,noa/), nua, nua, nua, noa)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/1,2,3,6/), nua, nua, nua, noa, nloc, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = t3a_excits(idet,1); b = t3a_excits(idet,2); c = t3a_excits(idet,3);
                     i = t3a_excits(idet,4); j = t3a_excits(idet,5); k = t3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        l = t3a_excits(jdet,4); m = t3a_excits(jdet,5);
                        ! compute < ijkabc | h2a(oooo) | lmkabc >
                        !hmatel = h2a_oooo(l,m,i,j)
                        hmatel = h2a_oooo(m,l,j,i)
                        ! compute < ijkabc | h1a(oo) | lmkabc > = -A(ij)A(lm) h1a_oo(l,i) * delta(m,j)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (m==j) hmatel1 = -h1a_oo(l,i) ! (1)      < ijkabc | h1a(oo) | ljkabc >
                        if (m==i) hmatel2 = h1a_oo(l,j) ! (ij)     < ijkabc | h1a(oo) | likabc >
                        if (l==j) hmatel3 = h1a_oo(m,i) ! (lm)     < ijkabc | h1a(oo) | jmkabc >
                        if (l==i) hmatel4 = -h1a_oo(m,j) ! (ij)(lm) < ijkabc | h1a(oo) | imkabc >
                        hmatel = hmatel + 0.5d0*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*t3a_amps(jdet)
                     end do
                     ! (ik)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = t3a_excits(jdet,4); m = t3a_excits(jdet,5);
                           ! compute < ijkabc | h2a(oooo) | lmiabc >
                           !hmatel = -h2a_oooo(l,m,k,j)
                           hmatel = h2a_oooo(m,l,k,j)
                           ! compute < ijkabc | h1a(oo) | lmiabc > = A(jk)A(lm) h1a_oo(l,k) * delta(m,j)
                           hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                           if (m==j) hmatel1 = h1a_oo(l,k) ! (1)      < ijkabc | h1a(oo) | ljiabc >
                           if (m==k) hmatel2 = -h1a_oo(l,j) ! (jk)     < ijkabc | h1a(oo) | lkiabc >
                           if (l==j) hmatel3 = -h1a_oo(m,k) ! (lm)
                           if (l==k) hmatel4 = h1a_oo(m,j) ! (jk)(lm)
                           hmatel = hmatel + 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                           resid(idet) = resid(idet) + hmatel*t3a_amps(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = t3a_excits(jdet,4); m = t3a_excits(jdet,5);
                           ! compute < ijkabc | h2a(oooo) | lmjabc >
                           !hmatel = -h2a_oooo(l,m,i,k)
                           hmatel = -h2a_oooo(m,l,k,i)
                           ! compute < ijkabc | h1a(oo) | lmjabc > = A(ik)A(lm) h1a_oo(l,i) * delta(m,k)
                           hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                           if (m==k) hmatel1 = h1a_oo(l,i) ! (1)      < ijkabc | h1a(oo) | lkjabc >
                           if (m==i) hmatel2 = -h1a_oo(l,k) ! (ik)
                           if (l==k) hmatel3 = -h1a_oo(m,i) ! (lm)
                           if (l==i) hmatel4 = h1a_oo(m,k) ! (ik)(lm)
                           hmatel = hmatel + 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                           resid(idet) = resid(idet) + hmatel*t3a_amps(jdet)
                        end do
                     end if
                  end do
                  !!! ABCI LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/1,noa-2/), nua, nua, nua, noa)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/1,2,3,4/), nua, nua, nua, noa, nloc, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = t3a_excits(idet,1); b = t3a_excits(idet,2); c = t3a_excits(idet,3);
                     i = t3a_excits(idet,4); j = t3a_excits(idet,5); k = t3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,i)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = t3a_excits(jdet,5); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(oooo) | imnabc >
                        !hmatel = h2a_oooo(m,n,j,k)
                        hmatel = h2a_oooo(n,m,k,j)
                        ! compute < ijkabc | h1a(oo) | imnabc > = -A(jk)A(mn) h1a_oo(m,j) * delta(n,k)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (n==k) hmatel1 = -h1a_oo(m,j)  ! < ijkabc | h1a(oo) | imkabc >
                        if (n==j) hmatel2 = h1a_oo(m,k)
                        if (m==k) hmatel3 = h1a_oo(n,j)
                        if (m==j) hmatel4 = -h1a_oo(n,k)
                        hmatel = hmatel + 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel*t3a_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           m = t3a_excits(jdet,5); n = t3a_excits(jdet,6);
                           ! compute < ijkabc | h2a(oooo) | jmnabc >
                           !hmatel = -h2a_oooo(m,n,i,k)
                           hmatel = -h2a_oooo(n,m,k,i)
                           ! compute < ijkabc | h1a(oo) | jmnabc > = A(ik)A(mn) h1a_oo(m,i) * delta(n,k)
                           hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                           if (n==k) hmatel1 = h1a_oo(m,i)
                           if (n==i) hmatel2 = -h1a_oo(m,k)
                           if (m==k) hmatel3 = -h1a_oo(n,i)
                           if (m==i) hmatel4 = h1a_oo(n,k)
                           hmatel = hmatel + 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                           resid(idet) = resid(idet) + hmatel*t3a_amps(jdet)
                        end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           m = t3a_excits(jdet,5); n = t3a_excits(jdet,6);
                           ! compute < ijkabc | h2a(oooo) | kmnabc >
                           !hmatel = -h2a_oooo(m,n,j,i)
                           hmatel = h2a_oooo(n,m,j,i)
                           ! compute < ijkabc | h1a(oo) | kmnabc > = A(ij)A(mn) h1a_oo(m,j) * delta(n,i)
                           hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                           if (n==i) hmatel1 = -h1a_oo(m,j)
                           if (n==j) hmatel2 = h1a_oo(m,i)
                           if (m==i) hmatel3 = h1a_oo(n,j)
                           if (m==j) hmatel4 = -h1a_oo(n,i)
                           hmatel = hmatel + 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                           resid(idet) = resid(idet) + hmatel*t3a_amps(jdet)
                        end do
                     end if
                  end do
                  !!! ABCJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/2,noa-1/), nua, nua, nua, noa)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/1,2,3,5/), nua, nua, nua, noa, nloc, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = t3a_excits(idet,1); b = t3a_excits(idet,2); c = t3a_excits(idet,3);
                     i = t3a_excits(idet,4); j = t3a_excits(idet,5); k = t3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,j)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        l = t3a_excits(jdet,4); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(oooo) | ljnabc >
                        !hmatel = h2a_oooo(l,n,i,k)
                        hmatel = h2a_oooo(n,l,k,i)
                        ! compute < ijkabc | h1a(oo) | ljnabc > = -A(ik)A(ln) h1a_oo(l,i) * delta(n,k)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (n==k) hmatel1 = -h1a_oo(l,i)
                        if (n==i) hmatel2 = h1a_oo(l,k)
                        if (l==k) hmatel3 = h1a_oo(n,i)
                        if (l==i) hmatel4 = -h1a_oo(n,k)
                        hmatel = hmatel + 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel*t3a_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = t3a_excits(jdet,4); n = t3a_excits(jdet,6);
                           ! compute < ijkabc | h2a(oooo) | linabc >
                           !hmatel = -h2a_oooo(l,n,j,k)
                           hmatel = -h2a_oooo(n,l,k,j)
                           ! compute < ijkabc | h1a(oo) | linabc > = A(jk)A(ln) h1a_oo(l,j) * delta(n,k)
                           hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                           if (n==k) hmatel1 = h1a_oo(l,j)
                           if (n==j) hmatel2 = -h1a_oo(l,k)
                           if (l==k) hmatel3 = -h1a_oo(n,j)
                           if (l==j) hmatel4 = h1a_oo(n,k)
                           hmatel = hmatel + 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                           resid(idet) = resid(idet) + hmatel*t3a_amps(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = t3a_excits(jdet,4); n = t3a_excits(jdet,6);
                           ! compute < ijkabc | h2a(oooo) | lknabc >
                           !hmatel = -h2a_oooo(l,n,i,j)
                           hmatel = -h2a_oooo(n,l,j,i)
                           ! compute < ijkabc | h1a(oo) | lknabc > = A(ij)A(ln) h1a_oo(l,i) * delta(n,j)
                           hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                           if (n==j) hmatel1 = h1a_oo(l,i)
                           if (n==i) hmatel2 = -h1a_oo(l,j)
                           if (l==j) hmatel3 = -h1a_oo(n,i)
                           if (l==i) hmatel4 = h1a_oo(n,j)
                           hmatel = hmatel + 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                           resid(idet) = resid(idet) + hmatel*t3a_amps(jdet)
                        end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 2: A(a/bc) h1a(ae) * t3a(ebcijk)
                  ! NOTE: WITHIN THESE LOOPS, H1A(VV) TERMS ARE DOUBLE-COUNTED SO COMPENSATE BY FACTOR OF 1/2
                  ! allocate new sorting arrays
                  nloc = noa*(noa-1)*(noa-2)/6*nua
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(noa,noa,noa,nua))
                  !!! IJKA LOOP !!!
                  call get_index_table(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/1,nua-2/), noa, noa, noa, nua)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/4,5,6,1/), noa, noa, noa, nua, nloc, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = t3a_excits(idet,1); b = t3a_excits(idet,2); c = t3a_excits(idet,3);
                     i = t3a_excits(idet,4); j = t3a_excits(idet,5); k = t3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,a)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); f = t3a_excits(jdet,3);
                        ! compute < ijkabc | h1a(vv) | ijkaef > = A(bc)A(ef) h1a_vv(b,e) * delta(c,f)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (c==f) hmatel1 = h1a_vv(e,b)  !h1a_vv(b,e) ! (1)
                        if (b==f) hmatel2 = -h1a_vv(e,c) !-h1a_vv(c,e) ! (bc)
                        if (c==e) hmatel3 = -h1a_vv(f,b) !-h1a_vv(b,f) ! (ef)
                        if (b==e) hmatel4 = h1a_vv(f,c)  ! h1a_vv(c,f) ! (bc)(ef)
                        hmatel = 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel*t3a_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); f = t3a_excits(jdet,3);
                        ! compute < ijkabc | h1a(vv) | ijkbef > = -A(ac)A(ef) h1a_vv(a,e) * delta(c,f)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (c==f) hmatel1 = -h1a_vv(e,a) !-h1a_vv(a,e) ! (1)
                        if (a==f) hmatel2 = h1a_vv(e,c)  !h1a_vv(c,e) ! (ac)
                        if (c==e) hmatel3 = h1a_vv(f,a)  !h1a_vv(a,f) ! (ef)
                        if (a==e) hmatel4 = -h1a_vv(f,c) !-h1a_vv(c,f) ! (ac)(ef)
                        hmatel = 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel*t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); f = t3a_excits(jdet,3);
                        ! compute < ijkabc | h1a(vv) | ijkcef > = -A(ab)A(ef) h1a_vv(b,e) * delta(a,f)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (a==f) hmatel1 = -h1a_vv(e,b) !-h1a_vv(b,e) ! (1)
                        if (b==f) hmatel2 = h1a_vv(e,a)  !h1a_vv(a,e) ! (ab)
                        if (a==e) hmatel3 = h1a_vv(f,b)  !h1a_vv(b,f) ! (ef)
                        if (b==e) hmatel4 = -h1a_vv(f,a) !-h1a_vv(a,f) ! (ab)(ef)
                        hmatel = 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel*t3a_amps(jdet)
                     end do
                     end if
                  end do
                  !!! IJKB LOOP !!!
                  call get_index_table(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/2,nua-1/), noa, noa, noa, nua)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/4,5,6,2/), noa, noa, noa, nua, nloc, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = t3a_excits(idet,1); b = t3a_excits(idet,2); c = t3a_excits(idet,3);
                     i = t3a_excits(idet,4); j = t3a_excits(idet,5); k = t3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); f = t3a_excits(jdet,3);
                        ! compute < ijkabc | h1a(vv) | ijkdbf > = A(ac)A(df) h1a_vv(a,d) * delta(c,f)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (c==f) hmatel1 = h1a_vv(d,a)  !h1a_vv(a,d) ! (1)
                        if (a==f) hmatel2 = -h1a_vv(d,c) !-h1a_vv(c,d) ! (ac)
                        if (c==d) hmatel3 = -h1a_vv(f,a) !-h1a_vv(a,f) ! (df)
                        if (a==d) hmatel4 = h1a_vv(f,c)  !h1a_vv(c,f) ! (ac)(df)
                        hmatel = 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel*t3a_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); f = t3a_excits(jdet,3);
                        ! compute < ijkabc | h1a(vv) | ijkdaf > = -A(bc)A(df) h1a_vv(b,d) * delta(c,f)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (c==f) hmatel1 = -h1a_vv(d,b) !-h1a_vv(b,d) ! (1)
                        if (b==f) hmatel2 = h1a_vv(d,c)  !h1a_vv(c,d) ! (bc)
                        if (c==d) hmatel3 = h1a_vv(f,b)  !h1a_vv(b,f) ! (df)
                        if (b==d) hmatel4 = -h1a_vv(f,c) !-h1a_vv(c,f) ! (bc)(df)
                        hmatel = 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel*t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); f = t3a_excits(jdet,3);
                        ! compute < ijkabc | h1a(vv) | ijkdcf > = -A(ab)A(df) h1a_vv(a,d) * delta(b,f)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (b==f) hmatel1 = -h1a_vv(d,a) !-h1a_vv(a,d) ! (1)
                        if (a==f) hmatel2 = h1a_vv(d,b)  !h1a_vv(b,d) ! (ab)
                        if (b==d) hmatel3 = h1a_vv(f,a)  !h1a_vv(a,f) ! (df)
                        if (a==d) hmatel4 = -h1a_vv(f,b) !-h1a_vv(b,f) ! (ab)(df)
                        hmatel = 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel*t3a_amps(jdet)
                     end do
                     end if
                  end do
                  !!! IJKC LOOP !!!
                  call get_index_table(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/3,nua/), noa, noa, noa, nua)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/4,5,6,3/), noa, noa, noa, nua, nloc, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = t3a_excits(idet,1); b = t3a_excits(idet,2); c = t3a_excits(idet,3);
                     i = t3a_excits(idet,4); j = t3a_excits(idet,5); k = t3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,c)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); e = t3a_excits(jdet,2);
                        ! compute < ijkabc | h1a(vv) | ijkdec > = A(ab)A(de) h1a_vv(a,d) * delta(b,e)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (b==e) hmatel1 = h1a_vv(d,a)  !h1a_vv(a,d) ! (1)
                        if (a==e) hmatel2 = -h1a_vv(d,b) !-h1a_vv(b,d) ! (ab)
                        if (b==d) hmatel3 = -h1a_vv(e,a) !-h1a_vv(a,e) ! (de)
                        if (a==d) hmatel4 = h1a_vv(e,b)  !h1a_vv(b,e) ! (ab)(de)
                        hmatel = 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel*t3a_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); e = t3a_excits(jdet,2);
                        ! compute < ijkabc | h1a(vv) | ijkdea > = -A(bc)A(de) h1a_vv(c,d) * delta(b,e)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (b==e) hmatel1 = -h1a_vv(d,c) !-h1a_vv(c,d) ! (1)
                        if (c==e) hmatel2 = h1a_vv(d,b)  !h1a_vv(b,d) ! (bc)
                        if (b==d) hmatel3 = h1a_vv(e,c)  !h1a_vv(c,e) ! (de)
                        if (c==d) hmatel4 = -h1a_vv(e,b) !-h1a_vv(b,e) ! (bc)(de)
                        hmatel = 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel*t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); e = t3a_excits(jdet,2);
                        ! compute < ijkabc | h1a(vv) | ijkdeb > = -A(ac)A(de) h1a_vv(a,d) * delta(c,e)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (c==e) hmatel1 = -h1a_vv(d,a) !-h1a_vv(a,d) ! (1)
                        if (a==e) hmatel2 = h1a_vv(d,c)  !h1a_vv(c,d) ! (ac)
                        if (c==d) hmatel3 = h1a_vv(e,a)  !h1a_vv(a,e) ! (de)
                        if (a==d) hmatel4 = -h1a_vv(e,c) !-h1a_vv(c,e) ! (ac)(de)
                        hmatel = 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel*t3a_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 5: A(i/jk)A(a/bc) h2a(amie) * t3a(ebcmjk)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = (nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,noa,noa))
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/1,2,4,5/), nua, nua, noa, noa, nloc, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = t3a_excits(idet,1); b = t3a_excits(idet,2); c = t3a_excits(idet,3);
                     i = t3a_excits(idet,4); j = t3a_excits(idet,5); k = t3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3a_excits(jdet,3); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | ijnabf >
                        !hmatel = h2a_voov(c,n,k,f)
                        hmatel = h2a_voov(n,f,c,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3a_excits(jdet,3); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | ijnbcf >
                        !hmatel = h2a_voov(a,n,k,f)
                        hmatel = h2a_voov(n,f,a,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3a_excits(jdet,3); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | ijnacf >
                        !hmatel = -h2a_voov(b,n,k,f)
                        hmatel = -h2a_voov(n,f,b,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3a_excits(jdet,3); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | jknabf >
                        !hmatel = h2a_voov(c,n,i,f)
                        hmatel = h2a_voov(n,f,c,i)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3a_excits(jdet,3); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | jknbcf >
                        !hmatel = h2a_voov(a,n,i,f)
                        hmatel = h2a_voov(n,f,a,i)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3a_excits(jdet,3); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | jknacf >
                        !hmatel = -h2a_voov(b,n,i,f)
                        hmatel = -h2a_voov(n,f,b,i)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3a_excits(jdet,3); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | iknabf >
                        !hmatel = -h2a_voov(c,n,j,f)
                        hmatel = -h2a_voov(n,f,c,j)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3a_excits(jdet,3); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | iknbcf >
                        !hmatel = -h2a_voov(a,n,j,f)
                        hmatel = -h2a_voov(n,f,a,j)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3a_excits(jdet,3); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | iknacf >
                        !hmatel = h2a_voov(b,n,j,f)
                        hmatel = h2a_voov(n,f,b,j)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                  end do
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/1,3,4,5/), nua, nua, noa, noa, nloc, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = t3a_excits(idet,1); b = t3a_excits(idet,2); c = t3a_excits(idet,3);
                     i = t3a_excits(idet,4); j = t3a_excits(idet,5); k = t3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | ijnaec >
                        !hmatel = h2a_voov(b,n,k,e)
                        hmatel = h2a_voov(n,e,b,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | ijnbec >
                        !hmatel = -h2a_voov(a,n,k,e)
                        hmatel = -h2a_voov(n,e,a,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | ijnaeb >
                        !hmatel = -h2a_voov(c,n,k,e)
                        hmatel = -h2a_voov(n,e,c,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | jknaec >
                        !hmatel = h2a_voov(b,n,i,e)
                        hmatel = h2a_voov(n,e,b,i)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | jknbec >
                        !hmatel = -h2a_voov(a,n,i,e)
                        hmatel = -h2a_voov(n,e,a,i)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | jknaeb >
                        !hmatel = -h2a_voov(c,n,i,e)
                        hmatel = -h2a_voov(n,e,c,i)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | iknaec >
                        !hmatel = -h2a_voov(b,n,j,e)
                        hmatel = -h2a_voov(n,e,b,j)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | iknbec >
                        !hmatel = h2a_voov(a,n,j,e)
                        hmatel = h2a_voov(n,e,a,j)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | iknaeb >
                        !hmatel = h2a_voov(c,n,j,e)
                        hmatel = h2a_voov(n,e,c,j)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                  end do
                  !!! BCIJ LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/2,3,4,5/), nua, nua, noa, noa, nloc, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = t3a_excits(idet,1); b = t3a_excits(idet,2); c = t3a_excits(idet,3);
                     i = t3a_excits(idet,4); j = t3a_excits(idet,5); k = t3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | ijndbc >
                        !hmatel = h2a_voov(a,n,k,d)
                        hmatel = h2a_voov(n,d,a,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | ijndac >
                        !hmatel = -h2a_voov(b,n,k,d)
                        hmatel = -h2a_voov(n,d,b,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | ijndab >
                        !hmatel = h2a_voov(c,n,k,d)
                        hmatel = h2a_voov(n,d,c,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | jkndbc >
                        !hmatel = h2a_voov(a,n,i,d)
                        hmatel = h2a_voov(n,d,a,i)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | jkndac >
                        !hmatel = -h2a_voov(b,n,i,d)
                        hmatel = -h2a_voov(n,d,b,i)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | jkndab >
                        !hmatel = h2a_voov(c,n,i,d)
                        hmatel = h2a_voov(n,d,c,i)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | ikndbc >
                        !hmatel = -h2a_voov(a,n,j,d)
                        hmatel = -h2a_voov(n,d,a,j)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | ikndac >
                        !hmatel = h2a_voov(b,n,j,d)
                        hmatel = h2a_voov(n,d,b,j)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); n = t3a_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | ikndab >
                        !hmatel = -h2a_voov(c,n,j,d)
                        hmatel = -h2a_voov(n,d,c,j)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                  end do
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/1,2,4,6/), nua, nua, noa, noa, nloc, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = t3a_excits(idet,1); b = t3a_excits(idet,2); c = t3a_excits(idet,3);
                     i = t3a_excits(idet,4); j = t3a_excits(idet,5); k = t3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3a_excits(jdet,3); m = t3a_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imkabf >
                        !hmatel = h2a_voov(c,m,j,f)
                        hmatel = h2a_voov(m,f,c,j)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3a_excits(jdet,3); m = t3a_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imkbcf >
                        !hmatel = h2a_voov(a,m,j,f)
                        hmatel = h2a_voov(m,f,a,j)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3a_excits(jdet,3); m = t3a_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imkacf >
                        !hmatel = -h2a_voov(b,m,j,f)
                        hmatel = -h2a_voov(m,f,b,j)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3a_excits(jdet,3); m = t3a_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | jmkabf >
                        !hmatel = -h2a_voov(c,m,i,f)
                        hmatel = -h2a_voov(m,f,c,i)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3a_excits(jdet,3); m = t3a_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | jmkbcf >
                        !hmatel = -h2a_voov(a,m,i,f)
                        hmatel = -h2a_voov(m,f,a,i)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3a_excits(jdet,3); m = t3a_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | jmkacf >
                        !hmatel = h2a_voov(b,m,i,f)
                        hmatel = h2a_voov(m,f,b,i)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3a_excits(jdet,3); m = t3a_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imjabf >
                        !hmatel = -h2a_voov(c,m,k,f)
                        hmatel = -h2a_voov(m,f,c,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3a_excits(jdet,3); m = t3a_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imjbcf >
                        !hmatel = -h2a_voov(a,m,k,f)
                        hmatel = -h2a_voov(m,f,a,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3a_excits(jdet,3); m = t3a_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imjacf >
                        !hmatel = h2a_voov(b,m,k,f)
                        hmatel = h2a_voov(m,f,b,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                  end do
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/1,3,4,6/), nua, nua, noa, noa, nloc, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = t3a_excits(idet,1); b = t3a_excits(idet,2); c = t3a_excits(idet,3);
                     i = t3a_excits(idet,4); j = t3a_excits(idet,5); k = t3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); m = t3a_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imkaec >
                        !hmatel = h2a_voov(b,m,j,e)
                        hmatel = h2a_voov(m,e,b,j)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); m = t3a_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imkbec >
                        !hmatel = -h2a_voov(a,m,j,e)
                        hmatel = -h2a_voov(m,e,a,j)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); m = t3a_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imkaeb >
                        !hmatel = -h2a_voov(c,m,j,e)
                        hmatel = -h2a_voov(m,e,c,j)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); m = t3a_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | jmkaec >
                        !hmatel = -h2a_voov(b,m,i,e)
                        hmatel = -h2a_voov(m,e,b,i)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); m = t3a_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | jmkbec >
                        !hmatel = h2a_voov(a,m,i,e)
                        hmatel = h2a_voov(m,e,a,i)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); m = t3a_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | jmkaeb >
                        !hmatel = h2a_voov(c,m,i,e)
                        hmatel = h2a_voov(m,e,c,i)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); m = t3a_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imjaec >
                        !hmatel = -h2a_voov(b,m,k,e)
                        hmatel = -h2a_voov(m,e,b,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); m = t3a_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imjbec >
                        !hmatel = h2a_voov(a,m,k,e)
                        hmatel = h2a_voov(m,e,a,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); m = t3a_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imjaeb >
                        !hmatel = h2a_voov(c,m,k,e)
                        hmatel = h2a_voov(m,e,c,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                  end do
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/2,3,4,6/), nua, nua, noa, noa, nloc, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = t3a_excits(idet,1); b = t3a_excits(idet,2); c = t3a_excits(idet,3);
                     i = t3a_excits(idet,4); j = t3a_excits(idet,5); k = t3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); m = t3a_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imkdbc >
                        !hmatel = h2a_voov(a,m,j,d)
                        hmatel = h2a_voov(m,d,a,j)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); m = t3a_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imkdac >
                        !hmatel = -h2a_voov(b,m,j,d)
                        hmatel = -h2a_voov(m,d,b,j)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); m = t3a_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imkdab >
                        !hmatel = h2a_voov(c,m,j,d)
                        hmatel = h2a_voov(m,d,c,j)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); m = t3a_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | jmkdbc >
                        !hmatel = -h2a_voov(a,m,i,d)
                        hmatel = -h2a_voov(m,d,a,i)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); m = t3a_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | jmkdac >
                        !hmatel = h2a_voov(b,m,i,d)
                        hmatel = h2a_voov(m,d,b,i)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); m = t3a_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | jmkdab >
                        !hmatel = -h2a_voov(c,m,i,d)
                        hmatel = -h2a_voov(m,d,c,i)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); m = t3a_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imjdbc >
                        !hmatel = -h2a_voov(a,m,k,d)
                        hmatel = -h2a_voov(m,d,a,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); m = t3a_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imjdac >
                        !hmatel = h2a_voov(b,m,k,d)
                        hmatel = h2a_voov(m,d,b,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); m = t3a_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imjdab >
                        !hmatel = -h2a_voov(c,m,k,d)
                        hmatel = -h2a_voov(m,d,c,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                  end do
                  !!! ABJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/1,2,5,6/), nua, nua, noa, noa, nloc, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = t3a_excits(idet,1); b = t3a_excits(idet,2); c = t3a_excits(idet,3);
                     i = t3a_excits(idet,4); j = t3a_excits(idet,5); k = t3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3a_excits(jdet,3); l = t3a_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | ljkabf >
                        !hmatel = h2a_voov(c,l,i,f)
                        hmatel = h2a_voov(l,f,c,i)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3a_excits(jdet,3); l = t3a_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | ljkbcf >
                        !hmatel = h2a_voov(a,l,i,f)
                        hmatel = h2a_voov(l,f,a,i)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3a_excits(jdet,3); l = t3a_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | ljkacf >
                        !hmatel = -h2a_voov(b,l,i,f)
                        hmatel = -h2a_voov(l,f,b,i)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3a_excits(jdet,3); l = t3a_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | likabf >
                        !hmatel = -h2a_voov(c,l,j,f)
                        hmatel = -h2a_voov(l,f,c,j)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3a_excits(jdet,3); l = t3a_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | likbcf >
                        !hmatel = -h2a_voov(a,l,j,f)
                        hmatel = -h2a_voov(l,f,a,j)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3a_excits(jdet,3); l = t3a_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | likacf >
                        !hmatel = h2a_voov(b,l,j,f)
                        hmatel = h2a_voov(l,f,b,j)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3a_excits(jdet,3); l = t3a_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | lijabf >
                        !hmatel = h2a_voov(c,l,k,f)
                        hmatel = h2a_voov(l,f,c,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3a_excits(jdet,3); l = t3a_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | lijbcf >
                        !hmatel = h2a_voov(a,l,k,f)
                        hmatel = h2a_voov(l,f,a,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3a_excits(jdet,3); l = t3a_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | lijacf >
                        !hmatel = -h2a_voov(b,l,k,f)
                        hmatel = -h2a_voov(l,f,b,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                  end do
                  !!! ACJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/1,3,5,6/), nua, nua, noa, noa, nloc, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = t3a_excits(idet,1); b = t3a_excits(idet,2); c = t3a_excits(idet,3);
                     i = t3a_excits(idet,4); j = t3a_excits(idet,5); k = t3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); l = t3a_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | ljkaec >
                        !hmatel = h2a_voov(b,l,i,e)
                        hmatel = h2a_voov(l,e,b,i)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); l = t3a_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | ljkbec >
                        !hmatel = -h2a_voov(a,l,i,e)
                        hmatel = -h2a_voov(l,e,a,i)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); l = t3a_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | ljkaeb >
                        !hmatel = -h2a_voov(c,l,i,e)
                        hmatel = -h2a_voov(l,e,c,i)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); l = t3a_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | likaec >
                        !hmatel = -h2a_voov(b,l,j,e)
                        hmatel = -h2a_voov(l,e,b,j)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); l = t3a_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | likbec >
                        !hmatel = h2a_voov(a,l,j,e)
                        hmatel = h2a_voov(l,e,a,j)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); l = t3a_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | likaeb >
                        !hmatel = h2a_voov(c,l,j,e)
                        hmatel = h2a_voov(l,e,c,j)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); l = t3a_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | lijaec >
                        !hmatel = h2a_voov(b,l,k,e)
                        hmatel = h2a_voov(l,e,b,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); l = t3a_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | lijbec >
                        !hmatel = -h2a_voov(a,l,k,e)
                        hmatel = -h2a_voov(l,e,a,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3a_excits(jdet,2); l = t3a_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | lijaeb >
                        !hmatel = -h2a_voov(c,l,k,e)
                        hmatel = -h2a_voov(l,e,c,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                  end do
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/2,3,5,6/), nua, nua, noa, noa, nloc, n3aaa, resid)
                  do idet = 1, n3aaa
                     a = t3a_excits(idet,1); b = t3a_excits(idet,2); c = t3a_excits(idet,3);
                     i = t3a_excits(idet,4); j = t3a_excits(idet,5); k = t3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); l = t3a_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | ljkdbc >
                        !hmatel = h2a_voov(a,l,i,d)
                        hmatel = h2a_voov(l,d,a,i)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); l = t3a_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | ljkdac >
                        !hmatel = -h2a_voov(b,l,i,d)
                        hmatel = -h2a_voov(l,d,b,i)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); l = t3a_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | ljkdab >
                        !hmatel = h2a_voov(c,l,i,d)
                        hmatel = h2a_voov(l,d,c,i)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); l = t3a_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | likdbc >
                        !hmatel = -h2a_voov(a,l,j,d)
                        hmatel = -h2a_voov(l,d,a,j)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); l = t3a_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | likdac >
                        !hmatel = h2a_voov(b,l,j,d)
                        hmatel = h2a_voov(l,d,b,j)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); l = t3a_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | likdab >
                        !hmatel = -h2a_voov(c,l,j,d)
                        hmatel = -h2a_voov(l,d,c,j)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); l = t3a_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | lijdbc >
                        !hmatel = h2a_voov(a,l,k,d)
                        hmatel = h2a_voov(l,d,a,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); l = t3a_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | lijdac >
                        !hmatel = -h2a_voov(b,l,k,d)
                        hmatel = -h2a_voov(l,d,b,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3a_excits(jdet,1); l = t3a_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | lijdab >
                        !hmatel = h2a_voov(c,l,k,d)
                        hmatel = h2a_voov(l,d,c,k)
                        resid(idet) = resid(idet) + hmatel * t3a_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 6: A(i/jk)A(a/bc) h2b(amie) * t3b(abeijm)
                  ! allocate and copy over t3b arrays
                  allocate(t3_amps_buff(n3aab),t3_excits_buff(n3aab,6))
                  t3_amps_buff(:) = t3b_amps(:)
                  t3_excits_buff(:,:) = t3b_excits(:,:)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = nua*(nua-1)/2*noa*(noa-1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,noa,noa))
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,2,4,5/), nua, nua, noa, noa, nloc, n3aab)
                  do idet = 1, n3aaa
                     a = t3a_excits(idet,1); b = t3a_excits(idet,2); c = t3a_excits(idet,3);
                     i = t3a_excits(idet,4); j = t3a_excits(idet,5); k = t3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits_buff(jdet,3); n = t3_excits_buff(jdet,6);
                        ! compute < ijkabc | h2b(voov) | ijn~abf~ >
                        !hmatel = h2b_voov(c,n,k,f)
                        hmatel = h2b_voov(n,f,c,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits_buff(jdet,3); n = t3_excits_buff(jdet,6);
                        ! compute < ijkabc | h2b(voov) | jkn~abf~ >
                        !hmatel = h2b_voov(c,n,i,f)
                        hmatel = h2b_voov(n,f,c,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits_buff(jdet,3); n = t3_excits_buff(jdet,6);
                        ! compute < ijkabc | h2b(voov) | ikn~abf~ >
                        !hmatel = -h2b_voov(c,n,j,f)
                        hmatel = -h2b_voov(n,f,c,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits_buff(jdet,3); n = t3_excits_buff(jdet,6);
                        ! compute < ijkabc | h2b(voov) | ijn~bcf~ >
                        !hmatel = h2b_voov(a,n,k,f)
                        hmatel = h2b_voov(n,f,a,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                     end if
                     ! (ik)(ac)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits_buff(jdet,3); n = t3_excits_buff(jdet,6);
                        ! compute < ijkabc | h2b(voov) | jkn~bcf~ >
                        !hmatel = h2b_voov(a,n,i,f)
                        hmatel = h2b_voov(n,f,a,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                     end if
                     ! (jk)(ac)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits_buff(jdet,3); n = t3_excits_buff(jdet,6);
                        ! compute < ijkabc | h2b(voov) | ikn~bcf~ >
                        !hmatel = -h2b_voov(a,n,j,f)
                        hmatel = -h2b_voov(n,f,a,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits_buff(jdet,3); n = t3_excits_buff(jdet,6);
                        ! compute < ijkabc | h2b(voov) | ijn~acf~ >
                        !hmatel = -h2b_voov(b,n,k,f)
                        hmatel = -h2b_voov(n,f,b,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                     end if
                     ! (ik)(bc)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits_buff(jdet,3); n = t3_excits_buff(jdet,6);
                        ! compute < ijkabc | h2b(voov) | jkn~acf~ >
                        !hmatel = -h2b_voov(b,n,i,f)
                        hmatel = -h2b_voov(n,f,b,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                     end if
                     ! (jk)(bc)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits_buff(jdet,3); n = t3_excits_buff(jdet,6);
                        ! compute < ijkabc | h2b(voov) | ikn~acf~ >
                        !hmatel = h2b_voov(b,n,j,f)
                        hmatel = h2b_voov(n,f,b,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate t3 buffer arrays
                  deallocate(t3_amps_buff,t3_excits_buff)

                  !
                  ! Moment contributions
                  !
                  allocate(xbuf(noa,noa,nua,nua))
                  do a = 1,nua
                     do b = 1,nua
                        do i = 1,noa
                           do j = 1,noa
                              xbuf(j,i,b,a) = t2a(b,a,j,i)
                           end do
                        end do
                     end do
                  end do
                  do idet = 1, n3aaa
                      a = t3a_excits(idet,1); b = t3a_excits(idet,2); c = t3a_excits(idet,3);
                      i = t3a_excits(idet,4); j = t3a_excits(idet,5); k = t3a_excits(idet,6);
                      do m = 1, noa
                          ! -A(k/ij)A(a/bc) h2a(amij) * t2a(bcmk)
                          resid(idet) = resid(idet) - I2A_vooo(m,a,i,j) * xbuf(m,k,b,c)
                          resid(idet) = resid(idet) + I2A_vooo(m,b,i,j) * xbuf(m,k,a,c)
                          resid(idet) = resid(idet) + I2A_vooo(m,c,i,j) * xbuf(m,k,b,a)
                          resid(idet) = resid(idet) + I2A_vooo(m,a,k,j) * xbuf(m,i,b,c)
                          resid(idet) = resid(idet) - I2A_vooo(m,b,k,j) * xbuf(m,i,a,c)
                          resid(idet) = resid(idet) - I2A_vooo(m,c,k,j) * xbuf(m,i,b,a)
                          resid(idet) = resid(idet) + I2A_vooo(m,a,i,k) * xbuf(m,j,b,c)
                          resid(idet) = resid(idet) - I2A_vooo(m,b,i,k) * xbuf(m,j,a,c)
                          resid(idet) = resid(idet) - I2A_vooo(m,c,i,k) * xbuf(m,j,b,a)
                      end do
                  end do
                  deallocate(xbuf)

                  do idet = 1, n3aaa
                      a = t3a_excits(idet,1); b = t3a_excits(idet,2); c = t3a_excits(idet,3);
                      i = t3a_excits(idet,4); j = t3a_excits(idet,5); k = t3a_excits(idet,6);
                      do e = 1, nua
                           ! A(i/jk)(c/ab) h2a(abie) * t2a(ecjk)
                          resid(idet) = resid(idet) + I2A_vvov(e,a,b,i) * t2a(e,c,j,k)
                          resid(idet) = resid(idet) - I2A_vvov(e,c,b,i) * t2a(e,a,j,k)
                          resid(idet) = resid(idet) - I2A_vvov(e,a,c,i) * t2a(e,b,j,k)
                          resid(idet) = resid(idet) - I2A_vvov(e,a,b,j) * t2a(e,c,i,k)
                          resid(idet) = resid(idet) + I2A_vvov(e,c,b,j) * t2a(e,a,i,k)
                          resid(idet) = resid(idet) + I2A_vvov(e,a,c,j) * t2a(e,b,i,k)
                          resid(idet) = resid(idet) - I2A_vvov(e,a,b,k) * t2a(e,c,j,i)
                          resid(idet) = resid(idet) + I2A_vvov(e,c,b,k) * t2a(e,a,j,i)
                          resid(idet) = resid(idet) + I2A_vvov(e,a,c,k) * t2a(e,b,j,i)
                      end do
                  end do

                  !!!! diagram 4: 1/2 A(c/ab) h2a(abef) * t3a(ebcijk)
                  k1 = 1
                  do a_chol=1,nua
                     do b_chol=a_chol+1,nua
                        ab_table(k1,1) = a_chol
                        ab_table(k1,2) = b_chol
                        k1 = k1 + 1
                     end do
                  end do
                  ! allocate new sorting arrays
                  nloc = noa*(noa-1)*(noa-2)/6*nua
                  allocate(loc_arr_copy1(2,nloc),loc_arr_copy2(2,nloc),loc_arr_copy3(2,nloc))
                  allocate(idx_table_copy1(noa,noa,noa,nua),idx_table_copy2(noa,noa,noa,nua),idx_table_copy3(noa,noa,noa,nua))
                  allocate(t3a_excits_copy1(n3aaa,6),t3a_excits_copy2(n3aaa,6),t3a_excits_copy3(n3aaa,6))
                  allocate(t3a_amps_copy1(n3aaa),t3a_amps_copy2(n3aaa),t3a_amps_copy3(n3aaa))
                  t3a_excits_copy1(:,:) = t3a_excits(:,:)
                  t3a_excits_copy2(:,:) = t3a_excits(:,:)
                  t3a_excits_copy3(:,:) = t3a_excits(:,:)
                  t3a_amps_copy1(:) = t3a_amps(:)
                  t3a_amps_copy2(:) = t3a_amps(:)
                  t3a_amps_copy3(:) = t3a_amps(:)
                  !!! IJKA LOOP !!!
                  call get_index_table(idx_table_copy1, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/1,nua-2/), noa, noa, noa, nua)
                  call sort4(t3a_excits_copy1, t3a_amps_copy1, loc_arr_copy1, idx_table_copy1, (/4,5,6,1/), noa, noa, noa, nua, nloc, n3aaa)
                  !!! IJKB LOOP !!!
                  call get_index_table(idx_table_copy2, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/2,nua-1/), noa, noa, noa, nua)
                  call sort4(t3a_excits_copy2, t3a_amps_copy2, loc_arr_copy2, idx_table_copy2, (/4,5,6,2/), noa, noa, noa, nua, nloc, n3aaa)
                  !!! IJKC LOOP !!!
                  call get_index_table(idx_table_copy3, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/3,nua/), noa, noa, noa, nua)
                  call sort4(t3a_excits_copy3, t3a_amps_copy3, loc_arr_copy3, idx_table_copy3, (/4,5,6,3/), noa, noa, noa, nua, nloc, n3aaa)
                  !$omp parallel do reduction(+:resid) private(h2a_vvvv,a,b,c,i,j,k,l,m,n,d,e,f,a_chol,b_chol,idx,hmatel,idet,jdet)
                  do k1=1,nua*(nua-1)/2
                     a_chol = ab_table(k1,1); b_chol = ab_table(k1,2);
                        !
                        ! get a batch of h2a_vvvv(ef)[a_chol,b_chol] integrals, where a_chol < b_chol
                        !
                        call dgemm('t','n',nua,nua,nchol,1.0d0,chol_a_vv(:,:,a_chol),nchol,chol_a_vv(:,:,b_chol),nchol,0.0d0,h2a_vvvv,nua)
                        h2a_vvvv = h2a_vvvv - transpose(h2a_vvvv)
                        do e=1,nua
                           do f=e+1,nua
                              !h2a_vvvv(e,f) = 0.0d0
                              !do d=1,nchol
                              !   h2a_vvvv(e,f) = h2a_vvvv(e,f)&
                              !                   +chol_a_vv(d,e,a_chol)*chol_a_vv(d,f,b_chol)&
                              !                   -chol_a_vv(d,f,a_chol)*chol_a_vv(d,e,b_chol)
                              !end do
                              do m=1,noa
                                 do n=m+1,noa
                                    h2a_vvvv(e,f) = h2a_vvvv(e,f) + h2a_oovv(m,n,e,f)*t2a(a_chol,b_chol,m,n)
                                 end do
                              end do
                              h2a_vvvv(f,e) = -h2a_vvvv(e,f)
                           end do
                        end do
                        do idet = 1, n3aaa ! master copy loop
                           a = t3a_excits(idet,1); b = t3a_excits(idet,2); c = t3a_excits(idet,3);
                           i = t3a_excits(idet,4); j = t3a_excits(idet,5); k = t3a_excits(idet,6);

                           !
                           ! build the following blocks:
                           !
                           ! >> h2a(b,c,:,:)
                           ! >> h2a(a,c,:,:)
                           ! >> h2a(a,b,:,:)

                           !
                           !
                           !!!! SB 1 !!!!
                           ! (1)
                           if (a_chol==b .and. b_chol==c) then
                              idx = idx_table_copy1(i,j,k,a)
                              do jdet = loc_arr_copy1(1,idx), loc_arr_copy1(2,idx)
                                 e = t3a_excits_copy1(jdet,2); f = t3a_excits_copy1(jdet,3);
                                 ! compute < ijkabc | h2a(vvvv) | ijkaef >
                                 hmatel = h2a_vvvv(e,f)
                                 resid(idet) = resid(idet) + hmatel*t3a_amps_copy1(jdet)
                              end do
                           end if
                           ! (ab)
                           if (a_chol==a .and. b_chol==c) then
                              idx = idx_table_copy1(i,j,k,b)
                              if (idx/=0) then
                                 do jdet = loc_arr_copy1(1,idx), loc_arr_copy1(2,idx)
                                    e = t3a_excits_copy1(jdet,2); f = t3a_excits_copy1(jdet,3);
                                    ! compute < ijkabc | h2a(vvvv) | ijkbef >
                                    hmatel = -h2a_vvvv(e,f)
                                    resid(idet) = resid(idet) + hmatel*t3a_amps_copy1(jdet)
                                 end do
                              end if
                           end if
                           ! (ac)
                           if (a_chol==a .and. b_chol==b) then
                              idx = idx_table_copy1(i,j,k,c)
                              if (idx/=0) then
                                 do jdet = loc_arr_copy1(1,idx), loc_arr_copy1(2,idx)
                                    e = t3a_excits_copy1(jdet,2); f = t3a_excits_copy1(jdet,3);
                                    ! compute < ijkabc | h2a(vvvv) | ijkcef >
                                    hmatel = h2a_vvvv(e,f)
                                    resid(idet) = resid(idet) + hmatel*t3a_amps_copy1(jdet)
                                 end do
                              end if
                           end if
                               !!!! END SB 1 !!!!
                               !
                               !!!! SB 2 !!!!
                               ! (1)
                               if (a_chol==a .and. b_chol==c) then
                                  idx = idx_table_copy2(i,j,k,b)
                                  do jdet = loc_arr_copy2(1,idx), loc_arr_copy2(2,idx)
                                     d = t3a_excits_copy2(jdet,1); f = t3a_excits_copy2(jdet,3);
                                     ! compute < ijkabc | h2a(vvvv) | ijkdbf >
                                     hmatel = h2a_vvvv(d,f)
                                     resid(idet) = resid(idet) + hmatel*t3a_amps_copy2(jdet)
                                  end do
                               end if
                               ! (ab)
                               if (a_chol==b .and. b_chol==c) then
                                  idx = idx_table_copy2(i,j,k,a)
                                  if (idx/=0) then
                                  do jdet = loc_arr_copy2(1,idx), loc_arr_copy2(2,idx)
                                     d = t3a_excits_copy2(jdet,1); f = t3a_excits_copy2(jdet,3);
                                     ! compute < ijkabc | h2a(vvvv) | ijkdaf >
                                     hmatel = -h2a_vvvv(d,f)
                                     resid(idet) = resid(idet) + hmatel*t3a_amps_copy2(jdet)
                                  end do
                                  end if
                               end if
                               ! (bc)
                               if (a_chol==a .and. b_chol==b) then
                                  idx = idx_table_copy2(i,j,k,c)
                                  if (idx/=0) then
                                  do jdet = loc_arr_copy2(1,idx), loc_arr_copy2(2,idx)
                                     d = t3a_excits_copy2(jdet,1); f = t3a_excits_copy2(jdet,3);
                                     ! compute < ijkabc | h2a(vvvv) | ijkdcf >
                                     hmatel = -h2a_vvvv(d,f)
                                     resid(idet) = resid(idet) + hmatel*t3a_amps_copy2(jdet)
                                  end do
                                  end if
                               end if
                               !!!! END SB 2 !!!!
                               !
                               !!!! SB 3 !!!!
                               ! (1)
                               if (a_chol==a .and. b_chol==b) then
                                  idx = idx_table_copy3(i,j,k,c)
                                  do jdet = loc_arr_copy3(1,idx), loc_arr_copy3(2,idx)
                                     d = t3a_excits_copy3(jdet,1); e = t3a_excits_copy3(jdet,2);
                                     ! compute < ijkabc | h2a(vvvv) | ijkdec >
                                     hmatel = h2a_vvvv(d,e)
                                     resid(idet) = resid(idet) + hmatel*t3a_amps_copy3(jdet)
                                  end do
                               end if
                               ! (ac)
                               if (a_chol==b .and. b_chol==c) then ! this was reversed and I put a minus sign since a_chol < b_chol
                                  idx = idx_table_copy3(i,j,k,a)
                                  if (idx/=0) then
                                  do jdet = loc_arr_copy3(1,idx), loc_arr_copy3(2,idx)
                                     d = t3a_excits_copy3(jdet,1); e = t3a_excits_copy3(jdet,2);
                                     ! compute < ijkabc | h2a(vvvv) | ijkdea >
                                     hmatel = h2a_vvvv(d,e)
                                     resid(idet) = resid(idet) + hmatel*t3a_amps_copy3(jdet)
                                  end do
                                  end if
                               end if
                               ! (bc)
                               if (a_chol==a .and. b_chol==c) then
                                  idx = idx_table_copy3(i,j,k,b)
                                  if (idx/=0) then
                                  do jdet = loc_arr_copy3(1,idx), loc_arr_copy3(2,idx)
                                     d = t3a_excits_copy3(jdet,1); e = t3a_excits_copy3(jdet,2);
                                     ! compute < ijkabc | h2a(vvvv) | ijkdeb >
                                     hmatel = -h2a_vvvv(d,e)
                                     resid(idet) = resid(idet) + hmatel*t3a_amps_copy3(jdet)
                                  end do
                                  end if
                               end if
                               !!!! END SB 3 !!!!
                               !
                            end do ! end master copy loop
                            !
                  end do
                  !$omp end parallel do
                  deallocate(t3a_excits_copy1,t3a_excits_copy2,t3a_excits_copy3)
                  deallocate(t3a_amps_copy1,t3a_amps_copy2,t3a_amps_copy3)
                  deallocate(idx_table_copy1,idx_table_copy2,idx_table_copy3)
                  deallocate(loc_arr_copy1,loc_arr_copy2,loc_arr_copy3)

                  ! Update t3 vector
                  do idet = 1,n3aaa
                      a = t3a_excits(idet,1); b = t3a_excits(idet,2); c = t3a_excits(idet,3);
                      i = t3a_excits(idet,4); j = t3a_excits(idet,5); k = t3a_excits(idet,6);
                      denom = fA_oo(i,i) + fA_oo(j,j) + fA_oo(k,k) - fA_vv(a,a) - fA_vv(b,b) - fA_vv(c,c)
                      resid(idet) = resid(idet)/(denom - shift)
                      t3a_amps(idet) = t3a_amps(idet) + resid(idet)
                  end do

              end subroutine update_t3a_p

              subroutine update_t3b_p(resid,&
                                      t3a_amps, t3a_excits,&
                                      t3b_amps, t3b_excits,&
                                      t3c_amps, t3c_excits,&
                                      t2a, t2b,&
                                      H1A_oo, H1A_vv, H1B_oo, H1B_vv,&
                                      H2A_oovv, H2A_vvov, H2A_vooo, H2A_oooo, H2A_voov, chol_a_vv,&
                                      H2B_oovv, H2B_vvov, H2B_vvvo, H2B_vooo, H2B_ovoo,&
                                      H2B_oooo, H2B_voov, H2B_vovo, H2B_ovov, H2B_ovvo, chol_b_vv,&
                                      H2C_oovv, H2C_voov,&
                                      fA_oo, fA_vv, fB_oo, fB_vv,&
                                      shift,&
                                      n3aaa, n3aab, n3abb,&
                                      noa, nua, nob, nub, nchol)

                  integer, intent(in) :: noa, nua, nob, nub, n3aaa, n3aab, n3abb, nchol
                  integer, intent(in) :: t3a_excits(n3aaa,6), t3c_excits(n3abb,6)
                  real(kind=8), intent(in) :: t2a(1:nua,1:nua,1:noa,1:noa),&
                                              t2b(1:nua,1:nub,1:noa,1:nob),&
                                              t3a_amps(n3aaa),&
                                              t3c_amps(n3abb),&
                                              H1A_oo(1:noa,1:noa),&
                                              H1A_vv(1:nua,1:nua),&
                                              H1B_oo(1:nob,1:nob),&
                                              H1B_vv(1:nub,1:nub),&
                                              H2A_oovv(1:noa,1:noa,1:nua,1:nua),&
                                              !H2A_vvov(1:nua,1:nua,1:noa,1:nua),&
                                              H2A_vvov(nua,nua,nua,noa),& ! reordered
                                              !H2A_vooo(1:nua,1:noa,1:noa,1:noa),&
                                              H2A_vooo(noa,nua,noa,noa),& ! reordered
                                              H2A_oooo(1:noa,1:noa,1:noa,1:noa),&
                                              !H2A_voov(1:nua,1:noa,1:noa,1:nua),&
                                              H2A_voov(noa,nua,nua,noa),& ! reordered
                                              !H2A_vvvv(1:nua,1:nua,1:nua,1:nua),&
                                              chol_a_vv(nchol,nua,nua),&
                                              H2B_oovv(1:noa,1:nob,1:nua,1:nub),&
                                              !H2B_vooo(1:nua,1:nob,1:noa,1:nob),&
                                              H2B_vooo(nob,nua,noa,nob),& ! reordered
                                              H2B_ovoo(1:noa,1:nub,1:noa,1:nob),&
                                              !H2B_vvov(1:nua,1:nub,1:noa,1:nub),&
                                              H2B_vvov(nub,nua,nub,noa),& ! reordered
                                              !H2B_vvvo(1:nua,1:nub,1:nua,1:nob),&
                                              H2B_vvvo(nua,nua,nub,nob),& ! reordered
                                              H2B_oooo(1:noa,1:nob,1:noa,1:nob),&
                                              !H2B_voov(1:nua,1:nob,1:noa,1:nub),&
                                              H2B_voov(nob,nub,nua,noa),& ! reordered
                                              !H2B_vovo(1:nua,1:nob,1:nua,1:nob),&
                                              H2B_vovo(nob,nua,nua,nob),& ! reordered
                                              !H2B_ovov(1:noa,1:nub,1:noa,1:nub),&
                                              H2B_ovov(noa,nub,nub,noa),& ! reordered
                                              !H2B_ovvo(1:noa,1:nub,1:nua,1:nob),&
                                              H2B_ovvo(noa,nua,nub,nob),& ! reordered
                                              !H2B_vvvv(1:nua,1:nub,1:nua,1:nub),&
                                              chol_b_vv(nchol,nub,nub),&
                                              H2C_oovv(1:nob,1:nob,1:nub,1:nub),&
                                              !H2C_voov(1:nub,1:nob,1:nob,1:nub),&
                                              H2C_voov(nob,nub,nub,nob),& ! reordered
                                              fA_oo(1:noa,1:noa), fA_vv(1:nua,1:nua),&
                                              fB_oo(1:nob,1:nob), fB_vv(1:nub,1:nub),&
                                              shift

                  integer, intent(inout) :: t3b_excits(n3aab,6)
                  !f2py intent(in,out) :: t3b_excits(0:n3aab-1,0:5)
                  real(kind=8), intent(inout) :: t3b_amps(n3aab)
                  !f2py intent(in,out) :: t3b_amps(0:n3aab-1)

                  real(kind=8), intent(out) :: resid(n3aab)

                  real(kind=8), allocatable :: t3_amps_buff(:), xbuf(:,:,:,:)
                  integer, allocatable :: t3_excits_buff(:,:)
                  real(kind=8), allocatable :: t3b_amps1(:), t3b_amps2(:)
                  integer, allocatable :: t3b_excits1(:,:), t3b_excits2(:,:)

                  integer, allocatable :: idx_table(:,:,:,:), idx_table1(:,:,:,:), idx_table2(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:), loc_arr1(:,:), loc_arr2(:,:)
                  real(kind=8), allocatable :: h_vv(:,:)

                  !real(kind=8) :: I2A_vooo(nua,noa,noa,noa)
                  real(kind=8) :: I2A_vooo(noa,nua,noa,noa) ! reordered
                  !real(kind=8) :: I2A_vvov(nua,nua,noa,nua)
                  real(kind=8) :: I2A_vvov(nua,nua,nua,noa) ! reordered
                  !real(kind=8) :: I2B_vooo(nua,nob,noa,nob)
                  real(kind=8) :: I2B_vooo(nob,nua,noa,nob) ! reordered
                  real(kind=8) :: I2B_ovoo(noa,nub,noa,nob)
                  !real(kind=8) :: I2B_vvov(nua,nub,noa,nub)
                  real(kind=8) :: I2B_vvov(nub,nua,nub,noa) ! reordered
                  !real(kind=8) :: I2B_vvvo(nua,nub,nua,nob)
                  real(kind=8) :: I2B_vvvo(nua,nua,nub,nob) ! reordered
                  real(kind=8) :: denom, val, t_amp, res_mm23, hmatel
                  real(kind=8) :: hmatel1, hmatel2, hmatel3, hmatel4
                  integer :: i, j, k, l, a, b, c, d, m, n, e, f, idet, jdet
                  integer :: idx, nloc
                  integer :: a_chol, b_chol, k1
                  integer :: ab_table(nua*(nua-1)/2,2)

                  ! compute VT3 intermediates
                  I2A_vooo = 0.5d0 * H2A_vooo
                  call calc_I2A_vooo(I2A_vooo,&
                               H2A_oovv,H2B_oovv,&
                               t3a_excits,t3a_amps,t3b_excits,t3b_amps,&
                               n3aaa,n3aab,noa,nua,nob,nub)
                  I2A_vvov = 0.5d0 * H2A_vvov
                  call calc_I2A_vvov(I2A_vvov,&
                               H2A_oovv,H2B_oovv,&
                               t3a_excits,t3a_amps,t3b_excits,t3b_amps,&
                               n3aaa,n3aab,noa,nua,nob,nub)
                  I2B_vooo(:,:,:,:) = H2B_vooo(:,:,:,:)
                  call calc_I2B_vooo(I2B_vooo,&
                               H2B_oovv,H2C_oovv,&
                               t3b_excits,t3b_amps,t3c_excits,t3c_amps,&
                               n3aab,n3abb,noa,nua,nob,nub)
                  I2B_ovoo(:,:,:,:) = H2B_ovoo(:,:,:,:)
                  call calc_I2B_ovoo(I2B_ovoo,&
                               H2A_oovv,H2B_oovv,&
                               t3b_excits,t3b_amps,t3c_excits,t3c_amps,&
                               n3aab,n3abb,noa,nua,nob,nub)
                  I2B_vvov(:,:,:,:) = H2B_vvov(:,:,:,:)
                  call calc_I2B_vvov(I2B_vvov,&
                               H2B_oovv,H2C_oovv,&
                               t3b_excits,t3b_amps,t3c_excits,t3c_amps,&
                               n3aab,n3abb,noa,nua,nob,nub)
                  I2B_vvvo(:,:,:,:) = H2B_vvvo(:,:,:,:)
                  call calc_I2B_vvvo(I2B_vvvo,&
                               H2A_oovv,H2B_oovv,&
                               t3b_excits,t3b_amps,t3c_excits,t3c_amps,&
                               n3aab,n3abb,noa,nua,nob,nub)

                  ! Zero the residual container
                  resid = 0.0d0

                  !!!! diagram 1: -A(ij) h1a(mi)*t3b(abcmjk)
                  !!!! diagram 5: A(ij) 1/2 h2a(mnij)*t3b(abcmnk)
                  !!! ABCK LOOP !!!
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2*nub*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,nub,noa))
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/1,nob/), nua, nua, nub, noa)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/1,2,3,6/), nua, nua, nub, noa, nloc, n3aab, resid)
                  do idet = 1, n3aab
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     idx = idx_table(a,b,c,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        l = t3b_excits(jdet,4); m = t3b_excits(jdet,5);
                        ! compute < ijk~abc~ | h2a(oooo) | lmk~abc~ >
                        !hmatel = h2a_oooo(l,m,i,j)
                        hmatel = h2a_oooo(m,l,j,i)
                        ! compute < ijk~abc~ | h1a(oo) | lmk~abc~ > = -A(ij)A(lm) h1a_oo(l,i) * delta(m,j)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (m==j) hmatel1 = -h1a_oo(l,i)
                        if (m==i) hmatel2 = h1a_oo(l,j)
                        if (l==j) hmatel3 = h1a_oo(m,i)
                        if (l==i) hmatel4 = -h1a_oo(m,j)
                        resid(idet) = resid(idet) + (hmatel + hmatel1 + hmatel2 + hmatel3 + hmatel4)*t3b_amps(jdet)
                     end do
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 2: A(ab) h1a(ae)*t3b(ebcmjk)
                  !!! CIJK LOOP !!!
                  ! allocate new sorting arrays
                  nloc = nub*noa*(noa-1)/2*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(noa,noa,nob,nub))
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/1,nub/), noa, noa, nob, nub)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/4,5,6,3/), noa, noa, nob, nub, nloc, n3aab, resid)
                  do idet = 1, n3aab
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     !idx = idx_table(c,i,j,k)
                     idx = idx_table(i,j,k,c)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3b_excits(jdet,1); e = t3b_excits(jdet,2); ! swap t3b_excits everywher
                        ! compute < ijk~abc~ | h1a(vv) | ijk~dec > = A(ab)A(de) h1a_vv(a,d)*delta(b,e)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (b==e) hmatel1 = h1a_vv(d,a)  !h1a_vv(a,d)
                        if (a==e) hmatel2 = -h1a_vv(d,b) !-h1a_vv(b,d)
                        if (b==d) hmatel3 = -h1a_vv(e,a) !-h1a_vv(a,e)
                        if (a==d) hmatel4 = h1a_vv(e,b)  !h1a_vv(b,e)
                        resid(idet) = resid(idet) + (hmatel1 + hmatel2 + hmatel3 + hmatel4)*t3b_amps(jdet)
                     end do
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 3: -h1b(mk)*t3b(abcijm)
                  !!!! diagram 7: A(ij) h2b(mnjk)*t3b(abcimn)
                  !!! ABCI LOOP !!!
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2*nub*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,nub,noa))
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/1,noa-1/), nua, nua, nub, noa)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/1,2,3,4/), nua, nua, nub, noa, nloc, n3aab, resid)
                  do idet = 1, n3aab
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,i)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = t3b_excits(jdet,5); n = t3b_excits(jdet,6);
                        ! compute < ijk~abc~ | h2b(oooo) | imn~abc~ >
                        hmatel = h2b_oooo(m,n,j,k)
                        ! compute < ijk~abc~ | h1b(oo) | imn~abc~ >
                        hmatel1 = 0.0d0
                        if (m==j) hmatel1 = -h1b_oo(n,k)
                        resid(idet) = resid(idet) + (hmatel + hmatel1)*t3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           m = t3b_excits(jdet,5); n = t3b_excits(jdet,6);
                           ! compute < ijk~abc~ | h2b(oooo) | jmn~abc~ >
                           hmatel = -h2b_oooo(m,n,i,k)
                           ! compute < ijk~abc~ | h1b(oo) | jmn~abc~ >
                           hmatel1 = 0.0d0
                           if (m==i) hmatel1 = h1b_oo(n,k)
                           resid(idet) = resid(idet) + (hmatel + hmatel1)*t3b_amps(jdet)
                        end do
                     end if
                  end do
                  !!! ABCJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/2,noa/), nua, nua, nub, noa)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/1,2,3,5/), nua, nua, nub, noa, nloc, n3aab, resid)
                  do idet = 1, n3aab
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,j)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        l = t3b_excits(jdet,4); n = t3b_excits(jdet,6);
                        ! compute < ijk~abc~ | h2b(oooo) | ljn~abc~ >
                        hmatel = h2b_oooo(l,n,i,k)
                        resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = t3b_excits(jdet,4); n = t3b_excits(jdet,6);
                           ! compute < ijk~abc~ | h2b(oooo) | lin~abc~ >
                           hmatel = -h2b_oooo(l,n,j,k)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 5: h1b(ce)*t3b(abeijm)
                  ! allocate new sorting arrays
                  nloc = nua*noa*(noa-1)/2*nob ! no3nu
                  allocate(loc_arr(2,nloc)) ! 2*no3nu
                  allocate(idx_table(noa,noa,nob,nua)) ! no3nu
                  !!! AIJK LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/1,nua-1/), noa, noa, nob, nua)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/4,5,6,1/), noa, noa, nob, nua, nloc, n3aab, resid)
                  do idet = 1, n3aab
                      a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                      ! (1)
                      idx = idx_table(i,j,k,a) ! make sizes powers of two to speed up address lookup
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                         e = t3b_excits(jdet,2); f = t3b_excits(jdet,3);
                         hmatel1 = 0.0d0
                         if (b==e) hmatel1 = h1b_vv(f,c) !h1b_vv(c,f)
                         resid(idet) = resid(idet) + hmatel1*t3b_amps(jdet)
                      end do
                      ! (ab)
                      idx = idx_table(i,j,k,b)
                      if (idx/=0) then ! protect against case where b = nua because a = 1, nua-1
                         do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            e = t3b_excits(jdet,2); f = t3b_excits(jdet,3);
                            hmatel1 = 0.0d0
                            if (a==e) hmatel1 = -h1b_vv(f,c) !-h1b_vv(c,f)
                            resid(idet) = resid(idet) + hmatel1*t3b_amps(jdet)
                         end do
                      end if
                  end do
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 9: A(ij)A(ab) h2a(amie)*t3b(ebcmjk)
                  ! allocate new sorting arrays
                  nloc = nua*nub*noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nub,noa,nob))
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/2,nua/), (/1,nub/), (/2,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/2,3,5,6/), nua, nub, noa, nob, nloc, n3aab, resid)
                  do idet = 1, n3aab
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,j,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3b_excits(jdet,1); l = t3b_excits(jdet,4);
                        ! compute < ijk~abc~ | h2a(voov) | ljk~dbc~ >
                        !hmatel = h2a_voov(a,l,i,d)
                        hmatel = h2a_voov(l,d,a,i)
                        resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then ! protect against case where a = 1 because b = 2, nua
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = t3b_excits(jdet,1); l = t3b_excits(jdet,4);
                           ! compute < ijk~abc~ | h2a(voov) | ljk~dac~ >
                           !hmatel = -h2a_voov(b,l,i,d)
                           hmatel = -h2a_voov(l,d,b,i)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then ! protect against case where i = 1 because j = 2, noa
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = t3b_excits(jdet,1); l = t3b_excits(jdet,4);
                           ! compute < ijk~abc~ | h2a(voov) | lik~dbc~ >
                           !hmatel = -h2a_voov(a,l,j,d)
                           hmatel = -h2a_voov(l,d,a,j)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then ! protect against case where a = 1 because b = 2, nua and i = 1 because j = 2, noa
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = t3b_excits(jdet,1); l = t3b_excits(jdet,4);
                           ! compute < ijk~abc~ | h2a(voov) | lik~dac~ >
                           !hmatel = h2a_voov(b,l,j,d)
                           hmatel = h2a_voov(l,d,b,j)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                  end do
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/2,nua/), (/1,nub/), (/1,noa-1/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/2,3,4,6/), nua, nub, noa, nob, nloc, n3aab, resid)
                  do idet = 1, n3aab
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,i,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3b_excits(jdet,1); l = t3b_excits(jdet,5);
                        ! compute < ijk~abc~ | h2a(voov) | ilk~dbc~ >
                        !hmatel = h2a_voov(a,l,j,d)
                        hmatel = h2a_voov(l,d,a,j)
                        resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then ! protect against where j = noa because i = 1, noa-1
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = t3b_excits(jdet,1); l = t3b_excits(jdet,5);
                           ! compute < ijk~abc~ | h2a(voov) | jlk~dbc~ >
                           !hmatel = -h2a_voov(a,l,i,d)
                           hmatel = -h2a_voov(l,d,a,i)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then ! protect against case where a = 1 because b = 2, nua
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = t3b_excits(jdet,1); l = t3b_excits(jdet,5);
                           ! compute < ijk~abc~ | h2a(voov) | ilk~dac~ >
                           !hmatel = -h2a_voov(b,l,j,d)
                           hmatel = -h2a_voov(l,d,b,j)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then ! protect against case where j = noa because i = 1, noa-1 and where a = 1 because b = 2, nua
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = t3b_excits(jdet,1); l = t3b_excits(jdet,5);
                           ! compute < ijk~abc~ | h2a(voov) | jlk~dac~ >
                           !hmatel = h2a_voov(b,l,i,d)
                           hmatel = h2a_voov(l,d,b,i)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                  end do
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/1,nub/), (/1,noa-1/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/1,3,4,6/), nua, nub, noa, nob, nloc, n3aab, resid)
                  do idet = 1, n3aab
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3b_excits(jdet,2); l = t3b_excits(jdet,5);
                        ! compute < ijk~abc~ | h2a(voov) | ilk~adc~  >
                        !hmatel = h2a_voov(b,l,j,d)
                        hmatel = h2a_voov(l,d,b,j)
                        resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = t3b_excits(jdet,2); l = t3b_excits(jdet,5);
                           ! compute < ijk~abc~ | h2a(voov) | jlk~adc~  >
                           !hmatel = -h2a_voov(b,l,i,d)
                           hmatel = -h2a_voov(l,d,b,i)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = t3b_excits(jdet,2); l = t3b_excits(jdet,5);
                           ! compute < ijk~abc~ | h2a(voov) | ilk~bdc~  >
                           !hmatel = -h2a_voov(a,l,j,d)
                           hmatel = -h2a_voov(l,d,a,j)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = t3b_excits(jdet,2); l = t3b_excits(jdet,5);
                           ! compute < ijk~abc~ | h2a(voov) | jlk~bdc~  >
                           !hmatel = h2a_voov(a,l,i,d)
                           hmatel = h2a_voov(l,d,a,i)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                  end do
                  !!! ACJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/1,nub/), (/2,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/1,3,5,6/), nua, nub, noa, nob, nloc, n3aab, resid)
                  do idet = 1, n3aab
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,j,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3b_excits(jdet,2); l = t3b_excits(jdet,4);
                        ! compute < ijk~abc~ | h2a(voov) | ljk~adc~  >
                        !hmatel = h2a_voov(b,l,i,d)
                        hmatel = h2a_voov(l,d,b,i)
                        resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = t3b_excits(jdet,2); l = t3b_excits(jdet,4);
                           ! compute < ijk~abc~ | h2a(voov) | lik~adc~  >
                           !hmatel = -h2a_voov(b,l,j,d)
                           hmatel = -h2a_voov(l,d,b,j)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = t3b_excits(jdet,2); l = t3b_excits(jdet,4);
                           ! compute < ijk~abc~ | h2a(voov) | ljk~bdc~  >
                           !hmatel = -h2a_voov(a,l,i,d)
                           hmatel = -h2a_voov(l,d,a,i)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = t3b_excits(jdet,2); l = t3b_excits(jdet,4);
                           ! compute < ijk~abc~ | h2a(voov) | lik~abc~  >
                           !hmatel = h2a_voov(a,l,j,d)
                           hmatel = h2a_voov(l,d,a,j)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 10: h2c(cmke)*t3b(abeijm)
                  ! allocate sorting arrays
                  nloc = nua*(nua-1)/2*noa*(noa-1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,noa,noa))
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/1,2,4,5/), nua, nua, noa, noa, nloc, n3aab, resid)
                  do idet = 1, n3aab
                      a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                      idx = idx_table(a,b,i,j)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                         f = t3b_excits(jdet,3); n = t3b_excits(jdet,6);
                         ! compute < ijk~abc~ | h2c(voov) | ijn~abf~ > = h2c_voov(c,n,k,f)
                         !hmatel = h2c_voov(c,n,k,f)
                         hmatel = h2c_voov(n,f,c,k)
                         resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                      end do
                  end do ! end loop over idet
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 11: -A(ij) h2b(mcie)*t3b(abemjk)
                  ! allocate sorting arrays
                  nloc = nua*(nua-1)/2*noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,noa,nob))
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,noa-1/), (/1,nob/), nua, nua, noa, nob)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/1,2,4,6/), nua, nua, noa, nob, nloc, n3aab, resid)
                  do idet = 1, n3aab
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,i,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3b_excits(jdet,3); m = t3b_excits(jdet,5);
                        ! compute < ijk~abc~ | h2b(ovov) | imk~abf~ >
                        !hmatel = -h2b_ovov(m,c,j,f)
                        hmatel = -h2b_ovov(m,f,c,j)
                        resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = t3b_excits(jdet,3); m = t3b_excits(jdet,5);
                           ! compute < ijk~abc~ | h2b(ovov) | jmk~abf~ >
                           !hmatel = h2b_ovov(m,c,i,f)
                           hmatel = h2b_ovov(m,f,c,i)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                  end do
                  !!! ABJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/2,noa/), (/1,nob/), nua, nua, noa, nob)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/1,2,5,6/), nua, nua, noa, nob, nloc, n3aab, resid)
                  do idet = 1, n3aab
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,j,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3b_excits(jdet,3); l = t3b_excits(jdet,4);
                        ! compute < ijk~abc~ | h2b(ovov) | ljk~abf~ >
                        !hmatel = -h2b_ovov(l,c,i,f)
                        hmatel = -h2b_ovov(l,f,c,i)
                        resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = t3b_excits(jdet,3); l = t3b_excits(jdet,4);
                           ! compute < ijk~abc~ | h2b(ovov) | lik~abf~ >
                           !hmatel = h2b_ovov(l,c,j,f)
                           hmatel = h2b_ovov(l,f,c,j)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 12: -A(ab) h2b(amek)*t3b(ebcijm)
                  ! allocate sorting arrays
                  nloc = nua*nub*noa*(noa-1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(noa,noa,nua,nub))
                  !!! BCIJ LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/2,nua/), (/1,nub/), noa, noa, nua, nub)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/4,5,2,3/), noa, noa, nua, nub, nloc, n3aab, resid)
                  do idet = 1, n3aab
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,b,c)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3b_excits(jdet,1); n = t3b_excits(jdet,6);
                        ! compute < ijk~abc~ | h2b(vovo) | ijn~dbc~ >
                        !hmatel = -h2b_vovo(a,n,d,k)
                        hmatel = -h2b_vovo(n,d,a,k)
                        resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = t3b_excits(jdet,1); n = t3b_excits(jdet,6);
                           ! compute < ijk~abc~ | h2b(vovo) | ijn~dac~ >
                           !hmatel = h2b_vovo(b,n,d,k)
                           hmatel = h2b_vovo(n,d,b,k)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                  end do
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nua-1/), (/1,nub/), noa, noa, nua, nub)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/4,5,1,3/), noa, noa, nua, nub, nloc, n3aab, resid)
                  do idet = 1, n3aab
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,a,c)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3b_excits(jdet,2); n = t3b_excits(jdet,6);
                        ! compute < ijk~abc~ | h2b(vovo) | ijn~aec~ >
                        !hmatel = -h2b_vovo(b,n,e,k)
                        hmatel = -h2b_vovo(n,e,b,k)
                        resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           e = t3b_excits(jdet,2); n = t3b_excits(jdet,6);
                           ! compute < ijk~abc~ | h2b(vovo) | ijn~bec~ >
                           !hmatel = h2b_vovo(a,n,e,k)
                           hmatel = h2b_vovo(n,e,a,k)
                           resid(idet) = resid(idet) + hmatel * t3b_amps(jdet)
                        end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 13: h2b(mcek)*t3a(abeijm) !!!!
                  ! allocate and initialize the copy of t3a
                  allocate(t3_amps_buff(n3aaa))
                  allocate(t3_excits_buff(n3aaa,6))
                  t3_amps_buff(:) = t3a_amps(:)
                  t3_excits_buff(:,:) = t3a_excits(:,:)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = (nua-1)*(nua-2)/2*(noa-1)*(noa-2)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,noa,noa))
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,2,4,5/), nua, nua, noa, noa, nloc, n3aaa)
                  do idet = 1, n3aab
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     idx = idx_table(a,b,i,j)
                     if (idx==0) cycle
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits_buff(jdet,3); n = t3_excits_buff(jdet,6);
                        ! compute < ijk~abc~ | h2b(ovvo) | ijnabf >
                        !hmatel = h2b_ovvo(n,c,f,k)
                        hmatel = h2b_ovvo(n,f,c,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                  end do
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,3,4,5/), nua, nua, noa, noa, nloc, n3aaa)
                  do idet = 1, n3aab
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     idx = idx_table(a,b,i,j)
                     if (idx==0) cycle
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits_buff(jdet,2); n = t3_excits_buff(jdet,6);
                        ! compute < ijk~abc~ | h2b(ovvo) | ijnaeb >
                        !hmatel = -h2b_ovvo(n,c,e,k)
                        hmatel = -h2b_ovvo(n,e,c,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                  end do
                  !!! BCIJ LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/2,3,4,5/), nua, nua, noa, noa, nloc, n3aaa)
                  do idet = 1, n3aab
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     idx = idx_table(a,b,i,j)
                     if (idx==0) cycle
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits_buff(jdet,1); n = t3_excits_buff(jdet,6);
                        ! compute < ijk~abc~ | h2b(ovvo) | ijndab >
                        !hmatel = h2b_ovvo(n,c,d,k)
                        hmatel = h2b_ovvo(n,d,c,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                  end do
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,2,4,6/), nua, nua, noa, noa, nloc, n3aaa)
                  do idet = 1, n3aab
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     idx = idx_table(a,b,i,j)
                     if (idx==0) cycle
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits_buff(jdet,3); m = t3_excits_buff(jdet,5);
                        ! compute < ijk~abc~ | h2b(ovvo) | imjabf >
                        !hmatel = -h2b_ovvo(m,c,f,k)
                        hmatel = -h2b_ovvo(m,f,c,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                  end do
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,3,4,6/), nua, nua, noa, noa, nloc, n3aaa)
                  do idet = 1, n3aab
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     idx = idx_table(a,b,i,j)
                     if (idx==0) cycle
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits_buff(jdet,2); m = t3_excits_buff(jdet,5);
                        ! compute < ijk~abc~ | h2b(ovvo) | imjaeb >
                        !hmatel = h2b_ovvo(m,c,e,k)
                        hmatel = h2b_ovvo(m,e,c,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                  end do
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/2,3,4,6/), nua, nua, noa, noa, nloc, n3aaa)
                  do idet = 1, n3aab
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     idx = idx_table(a,b,i,j)
                     if (idx==0) cycle
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits_buff(jdet,1); m = t3_excits_buff(jdet,5);
                        ! compute < ijk~abc~ | h2b(ovvo) | imjdab >
                        !hmatel = -h2b_ovvo(m,c,d,k)
                        hmatel = -h2b_ovvo(m,d,c,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                  end do
                  !!! ABJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,2,5,6/), nua, nua, noa, noa, nloc, n3aaa)
                  do idet = 1, n3aab
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     idx = idx_table(a,b,i,j)
                     if (idx==0) cycle
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits_buff(jdet,3); l = t3_excits_buff(jdet,4);
                        ! compute < ijk~abc~ | h2b(ovvo) | lijabf >
                        !hmatel = h2b_ovvo(l,c,f,k)
                        hmatel = h2b_ovvo(l,f,c,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                  end do
                  !!! ACJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,3,5,6/), nua, nua, noa, noa, nloc, n3aaa)
                  do idet = 1, n3aab
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     idx = idx_table(a,b,i,j)
                     if (idx==0) cycle
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits_buff(jdet,2); l = t3_excits_buff(jdet,4);
                        ! compute < ijk~abc~ | h2b(ovvo) | lijaeb >
                        !hmatel = -h2b_ovvo(l,c,e,k)
                        hmatel = -h2b_ovvo(l,e,c,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                  end do
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/2,3,5,6/), nua, nua, noa, noa, nloc, n3aaa)
                  do idet = 1, n3aab
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     idx = idx_table(a,b,i,j)
                     if (idx==0) cycle
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits_buff(jdet,1); l = t3_excits_buff(jdet,4);
                        ! compute < ijk~abc~ | h2b(ovvo) | lijdab >
                        !hmatel = h2b_ovvo(l,c,d,k)
                        hmatel = h2b_ovvo(l,d,c,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate t3 buffer arrays
                  deallocate(t3_amps_buff,t3_excits_buff)

                  !!!! diagram 14: A(ab)A(ij) h2b(bmje)*t3c(aecimk)
                  ! allocate and initialize the copy of t3c
                  allocate(t3_amps_buff(n3abb))
                  allocate(t3_excits_buff(n3abb,6))
                  t3_amps_buff(:) = t3c_amps(:)
                  t3_excits_buff(:,:) = t3c_excits(:,:)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = nua*nub*noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nub,noa,nob))
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/2,nub/), (/1,noa/), (/2,nob/), nua, nub, noa, nob)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,3,4,6/), nua, nub, noa, nob, nloc, n3abb)
                  do idet = 1, n3aab
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           e = t3_excits_buff(jdet,2); m = t3_excits_buff(jdet,5);
                           ! compute < ijk~abc~ | h2b(voov) | im~k~ae~c~ >
                           !hmatel = h2b_voov(b,m,j,e)
                           hmatel = h2b_voov(m,e,b,j)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           e = t3_excits_buff(jdet,2); m = t3_excits_buff(jdet,5);
                           ! compute < ijk~abc~ | h2b(voov) | im~k~be~c~ >
                           !hmatel = -h2b_voov(a,m,j,e)
                           hmatel = -h2b_voov(m,e,a,j)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           e = t3_excits_buff(jdet,2); m = t3_excits_buff(jdet,5);
                           ! compute < ijk~abc~ | h2b(voov) | jm~k~ae~c~ >
                           !hmatel = -h2b_voov(b,m,i,e)
                           hmatel = -h2b_voov(m,e,b,i)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           e = t3_excits_buff(jdet,2); m = t3_excits_buff(jdet,5);
                           ! compute < ijk~abc~ | h2b(voov) | jm~k~be~c~ >
                           !hmatel = h2b_voov(a,m,i,e)
                           hmatel = h2b_voov(m,e,a,i)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                  end do
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/1,nub-1/), (/1,noa/), (/2,nob/), nua, nub, noa, nob)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,2,4,6/), nua, nub, noa, nob, nloc, n3abb)
                  do idet = 1, n3aab
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = t3_excits_buff(jdet,3); m = t3_excits_buff(jdet,5);
                           ! compute < ijk~abc~ | h2b(voov) | im~k~ac~f~ >
                           !hmatel = -h2b_voov(b,m,j,f)
                           hmatel = -h2b_voov(m,f,b,j)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = t3_excits_buff(jdet,3); m = t3_excits_buff(jdet,5);
                           ! compute < ijk~abc~ | h2b(voov) | im~k~bc~f~ >
                           !hmatel = h2b_voov(a,m,j,f)
                           hmatel = h2b_voov(m,f,a,j)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = t3_excits_buff(jdet,3); m = t3_excits_buff(jdet,5);
                           ! compute < ijk~abc~ | h2b(voov) | jm~k~ac~f~ >
                           !hmatel = h2b_voov(b,m,i,f)
                           hmatel = h2b_voov(m,f,b,i)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = t3_excits_buff(jdet,3); m = t3_excits_buff(jdet,5);
                           ! compute < ijk~abc~ | h2b(voov) | jm~k~bc~f~ >
                           !hmatel = -h2b_voov(a,m,i,f)
                           hmatel = -h2b_voov(m,f,a,i)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                  end do
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/2,nub/), (/1,noa/), (/1,nob-1/), nua, nub, noa, nob)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,3,4,5/), nua, nub, noa, nob, nloc, n3abb)
                  do idet = 1, n3aab
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           e = t3_excits_buff(jdet,2); n = t3_excits_buff(jdet,6);
                           ! compute < ijk~abc~ | h2b(voov) | ik~n~ae~c~ >
                           !hmatel = -h2b_voov(b,n,j,e)
                           hmatel = -h2b_voov(n,e,b,j)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           e = t3_excits_buff(jdet,2); n = t3_excits_buff(jdet,6);
                           ! compute < ijk~abc~ | h2b(voov) | ik~n~be~c~ >
                           !hmatel = h2b_voov(a,n,j,e)
                           hmatel = h2b_voov(n,e,a,j)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           e = t3_excits_buff(jdet,2); n = t3_excits_buff(jdet,6);
                           ! compute < ijk~abc~ | h2b(voov) | jk~n~ae~c~ >
                           !hmatel = h2b_voov(b,n,i,e)
                           hmatel = h2b_voov(n,e,b,i)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           e = t3_excits_buff(jdet,2); n = t3_excits_buff(jdet,6);
                           ! compute < ijk~abc~ | h2b(voov) | jk~n~be~c~ >
                           !hmatel = -h2b_voov(a,n,i,e)
                           hmatel = -h2b_voov(n,e,a,i)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                  end do
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/1,nub-1/), (/1,noa/), (/1,nob-1/), nua, nub, noa, nob)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,2,4,5/), nua, nub, noa, nob, nloc, n3abb)
                  do idet = 1, n3aab
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = t3_excits_buff(jdet,3); n = t3_excits_buff(jdet,6);
                           ! compute < ijk~abc~ | h2b(voov) | ik~n~ac~f~ >
                           !hmatel = h2b_voov(b,n,j,f)
                           hmatel = h2b_voov(n,f,b,j)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = t3_excits_buff(jdet,3); n = t3_excits_buff(jdet,6);
                           ! compute < ijk~abc~ | h2b(voov) | ik~n~bc~f~ >
                           !hmatel = -h2b_voov(a,n,j,f)
                           hmatel = -h2b_voov(n,f,a,j)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = t3_excits_buff(jdet,3); n = t3_excits_buff(jdet,6);
                           ! compute < ijk~abc~ | h2b(voov) | jk~n~ac~f~ >
                           !hmatel = -h2b_voov(b,n,i,f)
                           hmatel = -h2b_voov(n,f,b,i)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           f = t3_excits_buff(jdet,3); n = t3_excits_buff(jdet,6);
                           ! compute < ijk~abc~ | h2b(voov) | jk~n~bc~f~ >
                           !hmatel = h2b_voov(a,n,i,f)
                           hmatel = h2b_voov(n,f,a,i)
                           resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate t3 buffer arrays
                  deallocate(t3_amps_buff,t3_excits_buff)

                  !!!! diagram 6: A(ab) 1/2 h2a(abef)*t3b(ebcmjk)
                  k1 = 1
                  do a_chol=1,nua
                     do b_chol=a_chol+1,nua
                        ab_table(k1,1) = a_chol
                        ab_table(k1,2) = b_chol
                        k1 = k1 + 1
                     end do
                  end do
                  !!! CIJK LOOP !!!
                  ! allocate new sorting arrays
                  nloc = nub*noa*(noa-1)/2*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(noa,noa,nob,nub))
                  allocate(h_vv(nua,nua))
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/1,nub/), noa, noa, nob, nub)
                  call sort4(t3b_excits, t3b_amps, loc_arr, idx_table, (/4,5,6,3/), noa, noa, nob, nub, nloc, n3aab, resid)
                  !do a_chol=1,nua; do b_chol=a_chol+1,nua;
                  !$omp parallel do reduction(+:resid) private(h_vv,a_chol,b_chol,a,b,c,i,j,k,l,m,n,d,e,f,idx,idet,jdet,hmatel)
                  do k1=1,nua*(nua-1)/2
                  a_chol = ab_table(k1,1); b_chol = ab_table(k1,2);
                  !
                  ! get a batch of h2a_vvvv(ef)[a_chol,b_chol] integrals, where a_chol < b_chol
                  !
                  call dgemm('t','n',nua,nua,nchol,1.0d0,chol_a_vv(:,:,a_chol),nchol,chol_a_vv(:,:,b_chol),nchol,0.0d0,h_vv,nua)
                  h_vv = h_vv - transpose(h_vv)
                  do e=1,nua
                     do f=e+1,nua
                        !h_vv(e,f) = 0.0d0
                        !do d=1,nchol
                        !   h_vv(e,f) = h_vv(e,f)&
                        !               +chol_a_vv(d,e,a_chol)*chol_a_vv(d,f,b_chol)&
                        !               -chol_a_vv(d,f,a_chol)*chol_a_vv(d,e,b_chol)
                        !end do
                        do m=1,noa
                           do n=m+1,noa
                              h_vv(e,f) = h_vv(e,f) + h2a_oovv(m,n,e,f)*t2a(a_chol,b_chol,m,n)
                           end do
                        end do
                        h_vv(f,e) = -h_vv(e,f)
                     end do
                  end do
                  do idet = 1, n3aab
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     if (a_chol==a .and. b_chol==b) then
                        idx = idx_table(i,j,k,c)
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           d = t3b_excits(jdet,1); e = t3b_excits(jdet,2); ! swap t3b_excits everywher
                           ! compute < ijk~abc~ | h2a(vvvv) | ijk~dec~ >
                           hmatel = h_vv(d,e)
                           resid(idet) = resid(idet) + hmatel*t3b_amps(jdet)
                        end do
                     end if
                  end do
                  end do ! end loop over a_chol < b_chol
                  !$omp end parallel do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table,h_vv)

                  !!!! diagram 8: A(ab) h2b(bcef)*t3b(aefijk)
                  ! allocate new sorting arrays
                  nloc = nua*noa*(noa-1)/2*nob ! no3nu
                  allocate(h_vv(nub,nua))
                  allocate(loc_arr1(2,nloc)) ! 2*no3nu
                  allocate(loc_arr2(2,nloc)) ! 2*no3nu
                  allocate(idx_table1(noa,noa,nob,nua)) ! no3nu
                  allocate(idx_table2(noa,noa,nob,nua)) ! no3nu
                  allocate(t3b_excits1(n3aab,6),t3b_excits2(n3aab,6))
                  allocate(t3b_amps1(n3aab),t3b_amps2(n3aab))
                  t3b_excits1(:,:) = t3b_excits(:,:); t3b_amps1(:) = t3b_amps(:);
                  t3b_excits2(:,:) = t3b_excits(:,:); t3b_amps2(:) = t3b_amps(:);
                  !!! AIJK LOOP !!!
                  call get_index_table(idx_table1, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/1,nua-1/), noa, noa, nob, nua)
                  call sort4(t3b_excits1, t3b_amps1, loc_arr1, idx_table1, (/4,5,6,1/), noa, noa, nob, nua, nloc, n3aab)
                  !!! BIJK LOOP !!!
                  call get_index_table(idx_table2, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/2,nua/), noa, noa, nob, nua)
                  call sort4(t3b_excits2, t3b_amps2, loc_arr2, idx_table2, (/4,5,6,2/), noa, noa, nob, nua, nloc, n3aab)
                  !$omp parallel do reduction(+:resid) private(a_chol,b_chol,h_vv,a,b,c,i,j,k,l,m,n,d,e,f,idx,idet,jdet,hmatel)
                  do a_chol=1,nua; do b_chol=1,nub;
                    !
                    ! get a batch of h2b_vvvv(ef)[a_chol,b_chol] integrals
                    !
                    call dgemm('t','n',nub,nua,nchol,1.0d0,chol_b_vv(:,:,b_chol),nchol,chol_a_vv(:,:,a_chol),nchol,0.0d0,h_vv,nub)
                    do e=1,nua
                       do f=1,nub
                          !h_vv(f,e) = 0.0d0
                          !do d=1,nchol
                          !   h_vv(f,e) = h_vv(f,e) + chol_b_vv(d,f,b_chol)*chol_a_vv(d,e,a_chol)
                          !end do
                          do m=1,noa
                             do n=1,nob
                                h_vv(f,e) = h_vv(f,e) + h2b_oovv(m,n,e,f)*t2b(a_chol,b_chol,m,n)
                             end do
                          end do
                       end do
                    end do
                    do idet = 1, n3aab
                      a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                      !!!! SB 1 !!!!
                      ! (1)
                      if (a_chol==b .and. b_chol==c) then
                      idx = idx_table1(i,j,k,a) ! make sizes powers of two to speed up address lookup
                      do jdet = loc_arr1(1,idx), loc_arr1(2,idx)
                         e = t3b_excits1(jdet,2); f = t3b_excits1(jdet,3);
                         ! compute < ijk~abc~ | h2b(vvvv) | ijk~aef~ >
                         hmatel = h_vv(f,e)
                         resid(idet) = resid(idet) + hmatel*t3b_amps1(jdet)
                      end do
                      end if
                      ! (ab)
                      if (a_chol==a .and. b_chol==c) then
                      idx = idx_table1(i,j,k,b)
                      if (idx/=0) then ! protect against case where b = nua because a = 1, nua-1
                         do jdet = loc_arr1(1,idx), loc_arr1(2,idx)
                            e = t3b_excits1(jdet,2); f = t3b_excits1(jdet,3);
                            ! compute < ijk~abc~ | h2b(vvvv) | ijk~bef~ >
                            hmatel = -h_vv(f,e)
                            resid(idet) = resid(idet) + hmatel*t3b_amps1(jdet)
                         end do
                      end if
                      end if
                      !!!! SB 2 !!!!
                      ! (1)
                      if (a_chol==a .and. b_chol==c) then
                      idx = idx_table2(i,j,k,b)
                      do jdet = loc_arr2(1,idx), loc_arr2(2,idx)
                         d = t3b_excits2(jdet,1); f = t3b_excits2(jdet,3);
                         ! compute < ijk~abc~ | h2b(vvvv) | ijk~dbf~ >
                         hmatel = h_vv(f,d)
                         resid(idet) = resid(idet) + hmatel*t3b_amps2(jdet)
                      end do
                      end if
                      if (a_chol==b .and. b_chol==c) then
                      idx = idx_table2(i,j,k,a)
                      if (idx/=0) then ! protect against case where a = 1 because b = 2, nua
                         do jdet = loc_arr2(1,idx), loc_arr2(2,idx)
                            d = t3b_excits2(jdet,1); f = t3b_excits2(jdet,3);
                            ! compute < ijk~abc~ | h2b(vvvv) | ijk~daf~ >
                            hmatel = -h_vv(f,d)
                            resid(idet) = resid(idet) + hmatel*t3b_amps2(jdet)
                         end do
                      end if
                      end if
                    end do ! end loop over idet
                  end do; end do; ! end loop over a_chol, b_chol
                  !$omp end parallel do
                  ! deallocate sorting arrays
                  deallocate(loc_arr1,loc_arr2,idx_table1,idx_table2,h_vv)
                  deallocate(t3b_excits1,t3b_excits2,t3b_amps1,t3b_amps2)

                  !
                  ! Moment contributions
                  !
                  do idet = 1, n3aab
                      a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                      do e = 1, nua
                          ! A(ab) I2B(bcek) * t2a(aeij)
                          resid(idet) = resid(idet) + I2B_vvvo(e,b,c,k) * t2a(e,a,j,i)
                          resid(idet) = resid(idet) - I2B_vvvo(e,a,c,k) * t2a(e,b,j,i)
                      end do
                  end do

                  do idet = 1, n3aab
                      a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                      do e = 1, nua
                          ! A(ij) I2A(abie) * t2b(ecjk)
                          resid(idet) = resid(idet) + I2A_vvov(e,a,b,i) * t2b(e,c,j,k)
                          resid(idet) = resid(idet) - I2A_vvov(e,a,b,j) * t2b(e,c,i,k)
                      end do
                  end do

                  allocate(xbuf(nub,nua,nob,noa))
                  do i = 1,noa
                     do j = 1,nob
                        do a = 1,nua
                           do b = 1,nub
                              xbuf(b,a,j,i) = t2b(a,b,i,j)
                           end do
                        end do
                     end do
                  end do
                  do idet = 1, n3aab
                      a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                      do e = 1, nub
                          ! A(ij)A(ab) I2b(acie) * t2b(bejk)
                          resid(idet) = resid(idet) + I2B_vvov(e,a,c,i) * xbuf(e,b,k,j)
                          resid(idet) = resid(idet) - I2B_vvov(e,a,c,j) * xbuf(e,b,k,i)
                          resid(idet) = resid(idet) - I2B_vvov(e,b,c,i) * xbuf(e,a,k,j)
                          resid(idet) = resid(idet) + I2B_vvov(e,b,c,j) * xbuf(e,a,k,i)
                      end do
                  end do
                  deallocate(xbuf)

                  allocate(xbuf(noa,noa,nua,nua))
                  do a = 1,nua
                     do b = 1,nua
                        do i = 1,noa
                           do j = 1,noa
                              xbuf(j,i,b,a) = t2a(b,a,j,i)
                           end do
                        end do
                     end do
                  end do
                  do idet = 1, n3aab
                      a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                      do m = 1, noa
                          ! -A(ij) h2b(mcjk) * t2a(abim)
                          resid(idet) = resid(idet) - I2B_ovoo(m,c,j,k) * xbuf(m,i,b,a)
                          resid(idet) = resid(idet) + I2B_ovoo(m,c,i,k) * xbuf(m,j,b,a)
                      end do
                  end do
                  deallocate(xbuf)

                  allocate(xbuf(noa,nob,nua,nub))
                  do b = 1,nub
                     do a = 1,nua
                        do j = 1,nob
                           do i = 1,noa
                              xbuf(i,j,a,b) = t2b(a,b,i,j)
                           end do
                        end do
                     end do
                  end do
                  do idet = 1, n3aab
                      a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                      do m = 1, noa
                          ! -A(ab) h2a(amij) * t2b(bcmk)
                          resid(idet) = resid(idet) - I2A_vooo(m,a,i,j) * xbuf(m,k,b,c)
                          resid(idet) = resid(idet) + I2A_vooo(m,b,i,j) * xbuf(m,k,a,c)
                      end do
                  end do
                  deallocate(xbuf)

                  allocate(xbuf(nob,noa,nub,nua))
                  do a = 1,nua
                     do b = 1,nub
                        do i = 1,noa
                           do j = 1,nob
                              xbuf(j,i,b,a) = t2b(a,b,i,j)
                           end do
                        end do
                     end do
                  end do
                  do idet = 1, n3aab
                      a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                      do m = 1, nob
                          ! -A(ij)A(ab) h2b(amik) * t2b(bcjm)
                          resid(idet) = resid(idet) - I2B_vooo(m,a,i,k) * xbuf(m,j,c,b)
                          resid(idet) = resid(idet) + I2B_vooo(m,b,i,k) * xbuf(m,j,c,a)
                          resid(idet) = resid(idet) + I2B_vooo(m,a,j,k) * xbuf(m,i,c,b)
                          resid(idet) = resid(idet) - I2B_vooo(m,b,j,k) * xbuf(m,i,c,a)
                      end do
                  end do
                  deallocate(xbuf)

                  ! Update t3 vector
                  do idet = 1, n3aab
                      a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                      denom = fa_oo(i,i) + fa_oo(j,j) + fb_oo(k,k) - fa_vv(a,a) - fa_vv(b,b) - fb_vv(c,c)
                      resid(idet) = resid(idet)/(denom - shift)
                      t3b_amps(idet) = t3b_amps(idet) + resid(idet)
                  end do

              end subroutine update_t3b_p

              subroutine update_t3c_p(resid,&
                                      t3b_amps, t3b_excits,&
                                      t3c_amps, t3c_excits,&
                                      t3d_amps, t3d_excits,&
                                      t2b, t2c,&
                                      H1A_oo, H1A_vv, H1B_oo, H1B_vv,&
                                      H2A_oovv, H2A_voov,&
                                      H2B_oovv, H2B_vooo, H2B_ovoo, H2B_vvov, H2B_vvvo, H2B_oooo,&
                                      H2B_voov, H2B_vovo, H2B_ovov, H2B_ovvo, chol_a_vv,&
                                      H2C_oovv, H2C_vooo, H2C_vvov, H2C_oooo, H2C_voov, chol_b_vv,&
                                      fA_oo, fA_vv, fB_oo, fB_vv,&
                                      shift,&
                                      n3aab, n3abb, n3bbb,&
                                      noa, nua, nob, nub, nchol)

                  integer, intent(in) :: noa, nua, nob, nub, n3aab, n3abb, n3bbb, nchol
                  integer, intent(in) :: t3b_excits(n3aab,6), t3d_excits(n3bbb,6)
                  real(kind=8), intent(in) :: t2b(1:nua,1:nub,1:noa,1:nob),&
                                              t2c(1:nub,1:nub,1:nob,1:nob),&
                                              t3b_amps(n3aab),t3d_amps(n3bbb),&
                                              H1A_oo(1:noa,1:noa),&
                                              H1A_vv(1:nua,1:nua),&
                                              H1B_oo(1:nob,1:nob),&
                                              H1B_vv(1:nub,1:nub),&
                                              H2A_oovv(1:noa,1:noa,1:nua,1:nua),&
                                              !H2A_voov(1:nua,1:noa,1:noa,1:nua),&
                                              H2A_voov(noa,nua,nua,noa),& ! reordered
                                              H2B_oovv(1:noa,1:nob,1:nua,1:nub),&
                                              !H2B_vooo(1:nua,1:nob,1:noa,1:nob),&
                                              H2B_vooo(nob,nua,noa,nob),& ! reordered
                                              H2B_ovoo(1:noa,1:nub,1:noa,1:nob),&
                                              !H2B_vvov(1:nua,1:nub,1:noa,1:nub),&
                                              H2B_vvov(nub,nua,nub,noa),& ! reordered
                                              !H2B_vvvo(1:nua,1:nub,1:nua,1:nob),&
                                              H2B_vvvo(nua,nua,nub,nob),& ! reordered
                                              H2B_oooo(1:noa,1:nob,1:noa,1:nob),&
                                              !H2B_voov(1:nua,1:nob,1:noa,1:nub),&
                                              H2B_voov(nob,nub,nua,noa),& ! reordered
                                              !H2B_vovo(1:nua,1:nob,1:nua,1:nob),&
                                              H2B_vovo(nob,nua,nua,nob),& ! reordered
                                              !H2B_ovov(1:noa,1:nub,1:noa,1:nub),&
                                              H2B_ovov(noa,nub,nub,noa),& ! reordered
                                              !H2B_ovvo(1:noa,1:nub,1:nua,1:nob),&
                                              H2B_ovvo(noa,nua,nub,nob),& ! reordered
                                              !H2B_vvvv(1:nua,1:nub,1:nua,1:nub),&
                                              chol_a_vv(nchol,nua,nua),&
                                              H2C_oovv(1:nob,1:nob,1:nub,1:nub),&
                                              !H2C_vooo(1:nub,1:nob,1:nob,1:nob),&
                                              H2C_vooo(nob,nub,nob,nob),& ! reordered
                                              !H2C_vvov(1:nub,1:nub,1:nob,1:nub),&
                                              H2C_vvov(nub,nub,nub,nob),& ! reordered
                                              H2C_oooo(1:nob,1:nob,1:nob,1:nob),&
                                              !H2C_voov(1:nub,1:nob,1:nob,1:nub),&
                                              H2C_voov(nob,nub,nub,nob),& ! reordered
                                              !H2C_vvvv(1:nub,1:nub,1:nub,1:nub),&
                                              chol_b_vv(nchol,nub,nub),&
                                              fA_oo(1:noa,1:noa), fA_vv(1:nua,1:nua),&
                                              fB_oo(1:nob,1:nob), fB_vv(1:nub,1:nub),&
                                              shift

                  integer, intent(inout) :: t3c_excits(n3abb,6)
                  !f2py intent(in,out) :: t3c_excits(0:n3abb-1,0:5)
                  real(kind=8), intent(inout) :: t3c_amps(n3abb)
                  !f2py intent(in,out) :: t3c_amps(0:n3abb-1)

                  real(kind=8), intent(out) :: resid(n3abb)

                  real(kind=8), allocatable :: t3_amps_buff(:)
                  integer, allocatable :: t3_excits_buff(:,:)
                  real(kind=8), allocatable :: t3c_amps1(:), t3c_amps2(:)
                  integer, allocatable :: t3c_excits1(:,:), t3c_excits2(:,:)

                  integer, allocatable :: idx_table(:,:,:,:), idx_table1(:,:,:,:), idx_table2(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:), loc_arr1(:,:), loc_arr2(:,:)

                                  !I2C_vooo(nub,nob,nob,nob),&
                  real(kind=8) :: I2C_vooo(nob,nub,nob,nob),& ! reordered
                                  !I2C_vvov(nub,nub,nob,nub),&
                                  I2C_vvov(nub,nub,nub,nob),& ! reordered
                                  !I2B_vooo(nua,nob,noa,nob),&
                                  I2B_vooo(nob,nua,noa,nob),& ! reordered
                                  I2B_ovoo(noa,nub,noa,nob),&
                                  !I2B_vvov(nua,nub,noa,nub),&
                                  I2B_vvov(nub,nua,nub,noa),&
                                  !I2B_vvvo(nua,nub,nua,nob)
                                  I2B_vvvo(nua,nua,nub,nob) ! reordered
                  real(kind=8) :: denom, val, t_amp, res_mm23, hmatel
                  real(kind=8) :: hmatel1, hmatel2, hmatel3, hmatel4
                  integer :: i, j, k, l, a, b, c, d, m, n, e, f, idet, jdet
                  integer :: idx, nloc, a_chol, b_chol, k1
                  integer :: ab_table(nub*(nub-1)/2,2)
                  real(kind=8), allocatable :: xbuf(:,:,:,:), h_vv(:,:)

                  ! VT3 intermediates
                  I2C_vooo(:,:,:,:) = 0.5d0 * H2C_vooo(:,:,:,:)
                  call calc_I2C_vooo(I2C_vooo,&
                               H2B_oovv,H2C_oovv,&
                               t3c_excits,t3c_amps,t3d_excits,t3d_amps,&
                               n3abb,n3bbb,noa,nua,nob,nub)
                  I2C_vvov(:,:,:,:) = 0.5d0 * H2C_vvov(:,:,:,:)
                  call calc_I2C_vvov(I2C_vvov,&
                               H2B_oovv,H2C_oovv,&
                               t3c_excits,t3c_amps,t3d_excits,t3d_amps,&
                               n3abb,n3bbb,noa,nua,nob,nub)
                  I2B_vooo(:,:,:,:) = H2B_vooo(:,:,:,:)
                  call calc_I2B_vooo(I2B_vooo,&
                               H2B_oovv,H2C_oovv,&
                               t3b_excits,t3b_amps,t3c_excits,t3c_amps,&
                               n3aab,n3abb,noa,nua,nob,nub)
                  I2B_ovoo(:,:,:,:) = H2B_ovoo(:,:,:,:)
                  call calc_I2B_ovoo(I2B_ovoo,&
                               H2A_oovv,H2B_oovv,&
                               t3b_excits,t3b_amps,t3c_excits,t3c_amps,&
                               n3aab,n3abb,noa,nua,nob,nub)
                  I2B_vvov(:,:,:,:) = H2B_vvov(:,:,:,:)
                  call calc_I2B_vvov(I2B_vvov,&
                               H2B_oovv,H2C_oovv,&
                               t3b_excits,t3b_amps,t3c_excits,t3c_amps,&
                               n3aab,n3abb,noa,nua,nob,nub)
                  I2B_vvvo(:,:,:,:) = H2B_vvvo(:,:,:,:)
                  call calc_I2B_vvvo(I2B_vvvo,&
                               H2A_oovv,H2B_oovv,&
                               t3b_excits,t3b_amps,t3c_excits,t3c_amps,&
                               n3aab,n3abb,noa,nua,nob,nub)

                  ! Zero the residual container
                  resid = 0.0d0

                  !!!! diagram 1: -A(jk) h1b(mk)*t3c(abcijm)
                  !!!! diagram 5: A(jk) 1/2 h2c(mnjk)*t3c(abcimn)
                  !!! BCAI LOOP !!!
                  ! allocate new sorting arrays
                  nloc = nub*(nub-1)/2*nua*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nua,noa))
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nua/), (/1,noa/), nub, nub, nua, noa)
                  call sort4(t3c_excits, t3c_amps, loc_arr, idx_table, (/2,3,1,4/), nub, nub, nua, noa, nloc, n3abb, resid)
                  do idet = 1, n3abb
                     a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                     i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                     idx = idx_table(b,c,a,i)
                     ! (1)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = t3c_excits(jdet,5); n = t3c_excits(jdet,6);
                        ! compute < ij~k~ab~c~ | h2c(oooo) | im~n~ab~c~ >
                        !hmatel = h2c_oooo(m,n,j,k)
                        hmatel = h2c_oooo(n,m,k,j)
                        ! compute < ij~k~ab~c~ | h1b(oo) | im~n~ab~c~ > = -A(jk)A(mn) h1b_oo(m,j) * delta(n,k)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.d0;
                        if (n==k) hmatel1 = -h1b_oo(m,j) ! (1)
                        if (n==j) hmatel2 = h1b_oo(m,k) ! (jk)
                        if (m==k) hmatel3 = h1b_oo(n,j) ! (mn)
                        if (m==j) hmatel4 = -h1b_oo(n,k) ! (jk)(mn)
                        resid(idet) = resid(idet) + (hmatel + hmatel1 + hmatel2 + hmatel3 + hmatel4)*t3c_amps(jdet)
                     end do
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 2: A(bc) h1b(ce)*t3c(abeijk)
                  !!! JKIA LOOP !!!
                  ! allocate new sorting arrays
                  nloc = nua*nob*(nob-1)/2*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nob,nob,noa,nua))
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,noa/), (/1,nua/), nob, nob, noa, nua)
                  call sort4(t3c_excits, t3c_amps, loc_arr, idx_table, (/5,6,4,1/), nob, nob, noa, nua, nloc, n3abb, resid)
                  do idet = 1, n3abb
                     a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                     i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                     idx = idx_table(j,k,i,a)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3c_excits(jdet,2); f = t3c_excits(jdet,3);
                        ! compute < ij~k~ab~c~ | h2c(vvvv) | ij~k~ae~f~ > = A(bc)A(ef) h1b_vv(b,e) * delta(c,f)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.d0;
                        if (c==f) hmatel1 = h1b_vv(e,b)  !h1b_vv(b,e) ! (1)
                        if (b==f) hmatel2 = -h1b_vv(e,c) !-h1b_vv(c,e) ! (bc)
                        if (c==e) hmatel3 = -h1b_vv(f,b) !-h1b_vv(b,f) ! (ef)
                        if (b==e) hmatel4 = h1b_vv(f,c)  !h1b_vv(c,f) ! (bc)(ef)
                        resid(idet) = resid(idet) + (hmatel1 + hmatel2 + hmatel3 + hmatel4)*t3c_amps(jdet)
                     end do
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 3: -h1a(mi)*t3c(abcmjk)
                  !!!! diagram 7: A(jk) h2b(mnij)*t3c(abcmnk)
                  !!! BCAK LOOP !!!
                  ! allocate new sorting arrays
                  nloc = nub*(nub-1)/2*nua*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nua,nob))
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nua/), (/2,nob/), nub, nub, nua, nob)
                  call sort4(t3c_excits, t3c_amps, loc_arr, idx_table, (/2,3,1,6/), nub, nub, nua, nob, nloc, n3abb, resid)
                  do idet = 1, n3abb
                     a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                     i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,a,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        l = t3c_excits(jdet,4); m = t3c_excits(jdet,5);
                        ! compute < ij~k~ab~c~ | h2b(oooo) | lm~k~ab~c~ >
                        hmatel = h2b_oooo(l,m,i,j)
                        ! compute < ij~k~ab~c~ | h1a(oo) | lm~k~ab~c~ >
                        hmatel1 = 0.0d0
                        if (m==j) hmatel1 = -h1a_oo(l,i)
                        resid(idet) = resid(idet) + (hmatel + hmatel1)*t3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table(b,c,a,j)
                     if (idx/=0) then
                         do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            l = t3c_excits(jdet,4); m = t3c_excits(jdet,5);
                            ! compute < ij~k~ab~c~ | h2b(oooo) | lm~j~ab~c~ >
                            hmatel = -h2b_oooo(l,m,i,k)
                            ! compute < ij~k~ab~c~ | h1a(oo) | lm~j~ab~c~ >
                            hmatel1 = 0.0d0
                            if (m==k) hmatel1 = h1a_oo(l,i)
                            resid(idet) = resid(idet) + (hmatel + hmatel1)*t3c_amps(jdet)
                         end do
                     end if
                  end do
                  !!! BCAJ LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nua/), (/1,nob-1/), nub, nub, nua, nob)
                  call sort4(t3c_excits, t3c_amps, loc_arr, idx_table, (/2,3,1,5/), nub, nub, nua, nob, nloc, n3abb, resid)
                  do idet = 1, n3abb
                     a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                     i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,a,j)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        l = t3c_excits(jdet,4); n = t3c_excits(jdet,6);
                        ! compute < ij~k~ab~c~ | h2b(oooo) | lj~n~ab~c~ >
                        hmatel = h2b_oooo(l,n,i,k)
                        resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table(b,c,a,k)
                     if (idx/=0) then
                         do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            l = t3c_excits(jdet,4); n = t3c_excits(jdet,6);
                            ! compute < ij~k~ab~c~ | h2b(oooo) | lk~n~ab~c~ >
                            hmatel = -h2b_oooo(l,n,i,j)
                            resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                         end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 5: h1a(ae)*t3c(ebcijk)
                  ! allocate new sorting arrays
                  nloc = nub*nob*(nob-1)/2*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nob,nob,noa,nub))
                  !!! JKIB LOOP !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,noa/), (/1,nub-1/), nob, nob, noa, nub)
                  call sort4(t3c_excits, t3c_amps, loc_arr, idx_table, (/5,6,4,2/), nob, nob, noa, nub, nloc, n3abb, resid)
                  do idet = 1, n3abb
                      a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      ! (1)
                      idx = idx_table(j,k,i,b) ! a changes faster than c
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                         d = t3c_excits(jdet,1); f = t3c_excits(jdet,3); ! unlike in t3b, d changes faster than f
                         hmatel1 = 0.0d0
                         if (c==f) hmatel1 = h1a_vv(d,a) !h1a_vv(a,d)
                         resid(idet) = resid(idet) + hmatel1*t3c_amps(jdet)
                      end do
                      ! (bc)
                      idx = idx_table(j,k,i,c)
                      if (idx/=0) then ! protect against case where b = nua because a = 1, nua-1
                         do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = t3c_excits(jdet,1); f = t3c_excits(jdet,3);
                            hmatel1 = 0.0d0
                            if (b==f) hmatel1 = -h1a_vv(d,a) !-h1a_vv(a,d)
                            resid(idet) = resid(idet) + hmatel1*t3c_amps(jdet)
                         end do
                      end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 9: A(jk)A(bc) h2c(cmke)*t3c(abeijm)
                  ! allocate new sorting arrays
                  nloc = nub*nua*nob*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nub,noa,nob))
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/1,nub-1/), (/1,noa/), (/1,nob-1/), nua, nub, noa, nob)
                  call sort4(t3c_excits, t3c_amps, loc_arr, idx_table, (/1,2,4,5/), nua, nub, noa, nob, nloc, n3abb, resid)
                  do idet = 1, n3abb
                     a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                     i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,i,j)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3c_excits(jdet,3); n = t3c_excits(jdet,6);
                        ! compute < ij~k~ab~c~ | h2a(voov) | ij~n~ab~f~ >
                        !hmatel = h2c_voov(c,n,k,f)
                        hmatel = h2c_voov(n,f,c,k)
                        resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                     end do
                     ! (jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                         do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            f = t3c_excits(jdet,3); n = t3c_excits(jdet,6);
                            ! compute < ij~k~ab~c~ | h2a(voov) | ik~n~ab~f~ >
                            !hmatel = -h2c_voov(c,n,j,f)
                            hmatel = -h2c_voov(n,f,c,j)
                            resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                         end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                         do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            f = t3c_excits(jdet,3); n = t3c_excits(jdet,6);
                            ! compute < ij~k~ab~c~ | h2a(voov) | ij~n~ac~f~ >
                            !hmatel = -h2c_voov(b,n,k,f)
                            hmatel = -h2c_voov(n,f,b,k)
                            resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                         end do
                     end if
                     ! (jk)(bc)
                      idx = idx_table(a,c,i,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                             f = t3c_excits(jdet,3); n = t3c_excits(jdet,6);
                             ! compute < ij~k~ab~c~ | h2a(voov) | ik~n~ac~f~ >
                             !hmatel = h2c_voov(b,n,j,f)
                             hmatel = h2c_voov(n,f,b,j)
                             resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                          end do
                      end if
                  end do
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/2,nub/), (/1,noa/), (/1,nob-1/), nua, nub, noa, nob)
                  call sort4(t3c_excits, t3c_amps, loc_arr, idx_table, (/1,3,4,5/), nua, nub, noa, nob, nloc, n3abb, resid)
                  do idet = 1, n3abb
                      a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      ! (1)
                      idx = idx_table(a,c,i,j)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                          e = t3c_excits(jdet,2); n = t3c_excits(jdet,6);
                          ! compute < ij~k~ab~c~ | h2c(voov) | ij~n~ae~c~ >
                          !hmatel = h2c_voov(b,n,k,e)
                          hmatel = h2c_voov(n,e,b,k)
                          resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                      end do
                      ! (jk)
                      idx = idx_table(a,c,i,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = t3c_excits(jdet,2); n = t3c_excits(jdet,6);
                              ! compute < ij~k~ab~c~ | h2c(voov) | ik~n~ae~c~ >
                              !hmatel = -h2c_voov(b,n,j,e)
                              hmatel = -h2c_voov(n,e,b,j)
                              resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                          end do
                      end if
                      ! (bc)
                      idx = idx_table(a,b,i,j)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = t3c_excits(jdet,2); n = t3c_excits(jdet,6);
                              ! compute < ij~k~ab~c~ | h2c(voov) | ij~n~ae~b~ >
                              !hmatel = -h2c_voov(c,n,k,e)
                              hmatel = -h2c_voov(n,e,c,k)
                              resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                          end do
                      end if
                      ! (jk)(bc)
                      idx = idx_table(a,b,i,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = t3c_excits(jdet,2); n = t3c_excits(jdet,6);
                              ! compute < ij~k~ab~c~ | h2c(voov) | ik~n~ae~b~ >
                              !hmatel = h2c_voov(c,n,j,e)
                              hmatel = h2c_voov(n,e,c,j)
                              resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                          end do
                      end if
                  end do
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/1,nub-1/), (/1,noa/), (/2,nob/), nua, nub, noa, nob)
                  call sort4(t3c_excits, t3c_amps, loc_arr, idx_table, (/1,2,4,6/), nua, nub, noa, nob, nloc, n3abb, resid)
                  do idet = 1, n3abb
                      a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      ! (1)
                      idx = idx_table(a,b,i,k)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                          f = t3c_excits(jdet,3); m = t3c_excits(jdet,5);
                          ! compute < ij~k~ab~c~ | h2c(voov) | im~k~ab~f~ >
                          !hmatel = h2c_voov(c,m,j,f)
                          hmatel = h2c_voov(m,f,c,j)
                          resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                      end do
                      ! (jk)
                      idx = idx_table(a,b,i,j)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              f = t3c_excits(jdet,3); m = t3c_excits(jdet,5);
                              ! compute < ij~k~ab~c~ | h2c(voov) | im~j~ab~f~ >
                              !hmatel = -h2c_voov(c,m,k,f)
                              hmatel = -h2c_voov(m,f,c,k)
                              resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                          end do
                      end if
                      ! (bc)
                      idx = idx_table(a,c,i,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              f = t3c_excits(jdet,3); m = t3c_excits(jdet,5);
                              ! compute < ij~k~ab~c~ | h2c(voov) | im~k~ac~f~ >
                              !hmatel = -h2c_voov(b,m,j,f)
                              hmatel = -h2c_voov(m,f,b,j)
                              resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                          end do
                      end if
                      ! (jk)(bc)
                      idx = idx_table(a,c,i,j)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              f = t3c_excits(jdet,3); m = t3c_excits(jdet,5);
                              ! compute < ij~k~ab~c~ | h2c(voov) | im~j~ac~f~ >
                              !hmatel = h2c_voov(b,m,k,f)
                              hmatel = h2c_voov(m,f,b,k)
                              resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                          end do
                      end if
                  end do
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/2,nub/), (/1,noa/), (/2,nob/), nua, nub, noa, nob)
                  call sort4(t3c_excits, t3c_amps, loc_arr, idx_table, (/1,3,4,6/), nua, nub, noa, nob, nloc, n3abb, resid)
                  do idet = 1, n3abb
                      a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      ! (1)
                      idx = idx_table(a,c,i,k)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                          e = t3c_excits(jdet,2); m = t3c_excits(jdet,5);
                          ! compute < ij~k~ab~c~ | h2c(voov) | im~k~ae~c~ >
                          !hmatel = h2c_voov(b,m,j,e)
                          hmatel = h2c_voov(m,e,b,j)
                          resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                      end do
                      ! (jk)
                      idx = idx_table(a,c,i,j)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = t3c_excits(jdet,2); m = t3c_excits(jdet,5);
                              ! compute < ij~k~ab~c~ | h2c(voov) | im~j~ae~c~ >
                              !hmatel = -h2c_voov(b,m,k,e)
                              hmatel = -h2c_voov(m,e,b,k)
                              resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                          end do
                      end if
                      ! (bc)
                      idx = idx_table(a,b,i,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = t3c_excits(jdet,2); m = t3c_excits(jdet,5);
                              ! compute < ij~k~ab~c~ | h2c(voov) | im~k~ae~b~ >
                              !hmatel = -h2c_voov(c,m,j,e)
                              hmatel = -h2c_voov(m,e,c,j)
                              resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                          end do
                      end if
                      ! (jk)(bc)
                      idx = idx_table(a,b,i,j)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = t3c_excits(jdet,2); m = t3c_excits(jdet,5);
                              ! compute < ij~k~ab~c~ | h2c(voov) | im~j~ae~b~ >
                              !hmatel = h2c_voov(c,m,k,e)
                              hmatel = h2c_voov(m,e,c,k)
                              resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                          end do
                      end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 10: h2a(amie)*t3c(ebcmjk)
                  ! allocate sorting arrays
                  nloc = nub*(nub-1)/2*nob*(nob-1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nob,nob))
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(t3c_excits, t3c_amps, loc_arr, idx_table, (/2,3,5,6/), nub, nub, nob, nob, nloc, n3abb, resid)
                  do idet = 1, n3abb
                      a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      idx = idx_table(b,c,j,k)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                         d = t3c_excits(jdet,1); l = t3c_excits(jdet,4);
                         ! compute < ij~k~ab~c~ | h2a(voov) | lj~k~db~c~ >
                         !hmatel = h2a_voov(a,l,i,d)
                         hmatel = h2a_voov(l,d,a,i)
                         resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                      end do
                  end do ! end loop over idet
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 11: -A(bc) h2b(mbie)*t3c(aecmjk)
                  ! allocate sorting arrays
                  nloc = nob*(nob-1)/2*nub*nua
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nob,nob,nua,nub))
                  !!! JKAC LOOP !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,nua/), (/2,nub/), nob, nob, nua, nub)
                  call sort4(t3c_excits, t3c_amps, loc_arr, idx_table, (/5,6,1,3/), nob, nob, nua, nub, nloc, n3abb, resid)
                  do idet = 1, n3abb
                      a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      ! (1)
                      idx = idx_table(j,k,a,c)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                          e = t3c_excits(jdet,2); l = t3c_excits(jdet,4);
                          ! compute < ij~k~ab~c~ | h2b(ovov) | lj~k~ae~c~ >
                          !hmatel = -h2b_ovov(l,b,i,e)
                          hmatel = -h2b_ovov(l,e,b,i)
                          resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                      end do
                      ! (bc)
                      idx = idx_table(j,k,a,b)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = t3c_excits(jdet,2); l = t3c_excits(jdet,4);
                              ! compute < ij~k~ab~c~ | h2b(ovov) | lj~k~ae~b~ >
                              !hmatel = h2b_ovov(l,c,i,e)
                              hmatel = h2b_ovov(l,e,c,i)
                              resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                          end do
                      end if
                  end do
                  !!! JKAB LOOP !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,nua/), (/1,nub-1/), nob, nob, nua, nub)
                  call sort4(t3c_excits, t3c_amps, loc_arr, idx_table, (/5,6,1,2/), nob, nob, nua, nub, nloc, n3abb, resid)
                  do idet = 1, n3abb
                      a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      ! (1)
                      idx = idx_table(j,k,a,b)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                          f = t3c_excits(jdet,3); l = t3c_excits(jdet,4);
                          ! compute < ij~k~ab~c~ | h2b(ovov) | lj~k~ab~f~ >
                          !hmatel = -h2b_ovov(l,c,i,f)
                          hmatel = -h2b_ovov(l,f,c,i)
                          resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                      end do
                      ! (bc)
                      idx = idx_table(j,k,a,c)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              f = t3c_excits(jdet,3); l = t3c_excits(jdet,4);
                              ! compute < ij~k~ab~c~ | h2b(ovov) | lj~k~ac~f~ >
                              !hmatel = h2b_ovov(l,b,i,f)
                              hmatel = h2b_ovov(l,f,b,i)
                              resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                          end do
                      end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 12: -A(bc) h2b(amej)*t3c(ebcimk)
                  ! allocate sorting arrays
                  nloc = nub*(nub-1)/2*noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,noa,nob))
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,noa/), (/2,nob/), nub, nub, noa, nob)
                  call sort4(t3c_excits, t3c_amps, loc_arr, idx_table, (/2,3,4,6/), nub, nub, noa, nob, nloc, n3abb, resid)
                  do idet = 1, n3abb
                      a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      ! (1)
                      idx = idx_table(b,c,i,k)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                          d = t3c_excits(jdet,1); m = t3c_excits(jdet,5);
                          ! compute < ij~k~ab~c~ | h2b(vovo) | im~k~db~c~ >
                          !hmatel = -h2b_vovo(a,m,d,j)
                          hmatel = -h2b_vovo(m,d,a,j)
                          resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                      end do
                      ! (jk)
                      idx = idx_table(b,c,i,j)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              d = t3c_excits(jdet,1); m = t3c_excits(jdet,5);
                              ! compute < ij~k~ab~c~ | h2b(vovo) | im~j~db~c~ >
                              !hmatel = h2b_vovo(a,m,d,k)
                              hmatel = h2b_vovo(m,d,a,k)
                              resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                          end do
                      end if
                  end do
                  !!! BCIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,noa/), (/1,nob-1/), nub, nub, noa, nob)
                  call sort4(t3c_excits, t3c_amps, loc_arr, idx_table, (/2,3,4,5/), nub, nub, noa, nob, nloc, n3abb, resid)
                  do idet = 1, n3abb
                      a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      ! (1)
                      idx = idx_table(b,c,i,j)
                      do jdet = loc_arr(1,idx), loc_arr(2,idx)
                          d = t3c_excits(jdet,1); n = t3c_excits(jdet,6);
                          ! compute < ij~k~ab~c~ | h2b(vovo) | ij~n~db~c~ >
                          !hmatel = -h2b_vovo(a,n,d,k)
                          hmatel = -h2b_vovo(n,d,a,k)
                          resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                      end do
                      ! (jk)
                      idx = idx_table(b,c,i,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              d = t3c_excits(jdet,1); n = t3c_excits(jdet,6);
                              ! compute < ij~k~ab~c~ | h2b(vovo) | ik~n~db~c~ >
                              !hmatel = h2b_vovo(a,n,d,j)
                              hmatel = h2b_vovo(n,d,a,j)
                              resid(idet) = resid(idet) + hmatel * t3c_amps(jdet)
                          end do
                      end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 13: h2b(amie)*t3d(ebcmjk)
                  ! allocate and initialize the copy of t3d
                  allocate(t3_amps_buff(n3bbb))
                  allocate(t3_excits_buff(n3bbb,6))
                  t3_amps_buff(:) = t3d_amps(:)
                  t3_excits_buff(:,:) = t3d_excits(:,:)
                  ! allocate sorting arrays
                  nloc = (nub-1)*(nub-2)/2*(nob-1)*(nob-2)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nob,nob))
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/2,3,5,6/), nub, nub, nob, nob, nloc, n3bbb)
                  do idet = 1, n3abb
                      a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              d = t3_excits_buff(jdet,1); l = t3_excits_buff(jdet,4);
                              ! compute < ij~k~ab~c~ | h2b(voov) | l~j~k~d~b~c~ >
                              !hmatel = h2b_voov(a,l,i,d)
                              hmatel = h2b_voov(l,d,a,i)
                              resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                          end do
                      end if
                  end do
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/2,3,4,6/), nub, nub, nob, nob, nloc, n3bbb)
                  do idet = 1, n3abb
                      a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              d = t3_excits_buff(jdet,1); m = t3_excits_buff(jdet,5);
                              ! compute < ij~k~ab~c~ | h2b(voov) | j~m~k~d~b~c~ >
                              !hmatel = -h2b_voov(a,m,i,d)
                              hmatel = -h2b_voov(m,d,a,i)
                              resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                          end do
                      end if
                  end do
                  !!! BCIJ LOOP !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/2,3,4,5/), nub, nub, nob, nob, nloc, n3bbb)
                  do idet = 1, n3abb
                      a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              d = t3_excits_buff(jdet,1); n = t3_excits_buff(jdet,6);
                              ! compute < ij~k~ab~c~ | h2b(voov) | j~k~n~d~b~c~ >
                              !hmatel = h2b_voov(a,n,i,d)
                              hmatel = h2b_voov(n,d,a,i)
                              resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                          end do
                      end if
                  end do
                  !!! ACJK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,3,5,6/), nub, nub, nob, nob, nloc, n3bbb)
                  do idet = 1, n3abb
                      a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = t3_excits_buff(jdet,2); l = t3_excits_buff(jdet,4);
                              ! compute < ij~k~ab~c~ | h2b(voov) | l~j~k~b~e~c~ >
                              !hmatel = -h2b_voov(a,l,i,e)
                              hmatel = -h2b_voov(l,e,a,i)
                              resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                          end do
                      end if
                  end do
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,3,4,6/), nub, nub, nob, nob, nloc, n3bbb)
                  do idet = 1, n3abb
                      a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = t3_excits_buff(jdet,2); m = t3_excits_buff(jdet,5);
                              ! compute < ij~k~ab~c~ | h2b(voov) | j~m~k~b~e~c~ >
                              !hmatel = h2b_voov(a,m,i,e)
                              hmatel = h2b_voov(m,e,a,i)
                              resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                          end do
                      end if
                  end do
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,3,4,5/), nub, nub, nob, nob, nloc, n3bbb)
                  do idet = 1, n3abb
                      a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              e = t3_excits_buff(jdet,2); n = t3_excits_buff(jdet,6);
                              ! compute < ij~k~ab~c~ | h2b(voov) | j~k~n~b~e~c~ >
                              !hmatel = -h2b_voov(a,n,i,e)
                              hmatel = -h2b_voov(n,e,a,i)
                              resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                          end do
                      end if
                  end do
                  !!! ABJK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,2,5,6/), nub, nub, nob, nob, nloc, n3bbb)
                  do idet = 1, n3abb
                      a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              f = t3_excits_buff(jdet,3); l = t3_excits_buff(jdet,4);
                              ! compute < ij~k~ab~c~ | h2b(voov) | l~j~k~b~c~f~ >
                              !hmatel = h2b_voov(a,l,i,f)
                              hmatel = h2b_voov(l,f,a,i)
                              resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                          end do
                      end if
                  end do
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,2,4,6/), nub, nub, nob, nob, nloc, n3bbb)
                  do idet = 1, n3abb
                      a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              f = t3_excits_buff(jdet,3); m = t3_excits_buff(jdet,5);
                              ! compute < ij~k~ab~c~ | h2b(voov) | j~m~k~b~c~f~ >
                              !hmatel = -h2b_voov(a,m,i,f)
                              hmatel = -h2b_voov(m,f,a,i)
                              resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                          end do
                      end if
                  end do
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,2,4,5/), nub, nub, nob, nob, nloc, n3bbb)
                  do idet = 1, n3abb
                      a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      ! (1)
                      idx = idx_table(b,c,j,k)
                      if (idx/=0) then
                          do jdet = loc_arr(1,idx), loc_arr(2,idx)
                              f = t3_excits_buff(jdet,3); n = t3_excits_buff(jdet,6);
                              ! compute < ij~k~ab~c~ | h2b(voov) | j~k~n~b~c~f~ >
                              !hmatel = h2b_voov(a,n,i,f)
                              hmatel = h2b_voov(n,f,a,i)
                              resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                          end do
                      end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate t3 buffer arrays
                  deallocate(t3_amps_buff,t3_excits_buff)

                  !!!! diagram 14: A(bc)A(jk) h2b(mbej)*t3b(aecimk)
                  ! allocate and initialize the copy of t3b
                  allocate(t3_amps_buff(n3aab))
                  allocate(t3_excits_buff(n3aab,6))
                  t3_amps_buff(:) = t3b_amps(:)
                  t3_excits_buff(:,:) = t3b_excits(:,:)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = nua*nub*noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nub,noa,nob))
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/1,nub/), (/1,noa-1/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,3,4,6/), nua, nub, noa, nob, nloc, n3aab)
                  do idet = 1, n3abb
                     a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                     i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            e = t3_excits_buff(jdet,2); m = t3_excits_buff(jdet,5);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | imk~aec~ >
                            !hmatel = h2b_ovvo(m,b,e,j)
                            hmatel = h2b_ovvo(m,e,b,j)
                            resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            e = t3_excits_buff(jdet,2); m = t3_excits_buff(jdet,5);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | imk~aeb~ >
                            !hmatel = -h2b_ovvo(m,c,e,j)
                            hmatel = -h2b_ovvo(m,e,c,j)
                            resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            e = t3_excits_buff(jdet,2); m = t3_excits_buff(jdet,5);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | imj~aec~ >
                            !hmatel = -h2b_ovvo(m,b,e,k)
                            hmatel = -h2b_ovvo(m,e,b,k)
                            resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            e = t3_excits_buff(jdet,2); m = t3_excits_buff(jdet,5);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | imj~aeb~ >
                            !hmatel = h2b_ovvo(m,c,e,k)
                            hmatel = h2b_ovvo(m,e,c,k)
                            resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                  end do
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/2,nua/), (/1,nub/), (/1,noa-1/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/2,3,4,6/), nua, nub, noa, nob, nloc, n3aab)
                  do idet = 1, n3abb
                     a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                     i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = t3_excits_buff(jdet,1); m = t3_excits_buff(jdet,5);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | imk~dac~ >
                            !hmatel = -h2b_ovvo(m,b,d,j)
                            hmatel = -h2b_ovvo(m,d,b,j)
                            resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = t3_excits_buff(jdet,1); m = t3_excits_buff(jdet,5);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | imk~dab~ >
                            !hmatel = h2b_ovvo(m,c,d,j)
                            hmatel = h2b_ovvo(m,d,c,j)
                            resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = t3_excits_buff(jdet,1); m = t3_excits_buff(jdet,5);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | imj~dac~ >
                            !hmatel = h2b_ovvo(m,b,d,k)
                            hmatel = h2b_ovvo(m,d,b,k)
                            resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = t3_excits_buff(jdet,1); m = t3_excits_buff(jdet,5);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | imj~dab~ >
                            !hmatel = -h2b_ovvo(m,c,d,k)
                            hmatel = -h2b_ovvo(m,d,c,k)
                            resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                  end do
                  !!! ACJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/1,nub/), (/2,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,3,5,6/), nua, nub, noa, nob, nloc, n3aab)
                  do idet = 1, n3abb
                     a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                     i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            e = t3_excits_buff(jdet,2); l = t3_excits_buff(jdet,4);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | lik~aec~ >
                            !hmatel = -h2b_ovvo(l,b,e,j)
                            hmatel = -h2b_ovvo(l,e,b,j)
                            resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            e = t3_excits_buff(jdet,2); l = t3_excits_buff(jdet,4);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | lik~aeb~ >
                            !hmatel = h2b_ovvo(l,c,e,j)
                            hmatel = h2b_ovvo(l,e,c,j)
                            resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            e = t3_excits_buff(jdet,2); l = t3_excits_buff(jdet,4);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | lij~aec~ >
                            !hmatel = h2b_ovvo(l,b,e,k)
                            hmatel = h2b_ovvo(l,e,b,k)
                            resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            e = t3_excits_buff(jdet,2); l = t3_excits_buff(jdet,4);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | lij~aeb~ >
                            !hmatel = -h2b_ovvo(l,c,e,k)
                            hmatel = -h2b_ovvo(l,e,c,k)
                            resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                  end do
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/2,nua/), (/1,nub/), (/2,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/2,3,5,6/), nua, nub, noa, nob, nloc, n3aab)
                  do idet = 1, n3abb
                     a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                     i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = t3_excits_buff(jdet,1); l = t3_excits_buff(jdet,4);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | lik~dac~ >
                            !hmatel = h2b_ovvo(l,b,d,j)
                            hmatel = h2b_ovvo(l,d,b,j)
                            resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = t3_excits_buff(jdet,1); l = t3_excits_buff(jdet,4);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | lik~dab~ >
                            !hmatel = -h2b_ovvo(l,c,d,j)
                            hmatel = -h2b_ovvo(l,d,c,j)
                            resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = t3_excits_buff(jdet,1); l = t3_excits_buff(jdet,4);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | lij~dac~ >
                            !hmatel = -h2b_ovvo(l,b,d,k)
                            hmatel = -h2b_ovvo(l,d,b,k)
                            resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                            d = t3_excits_buff(jdet,1); l = t3_excits_buff(jdet,4);
                            ! compute < ij~k~ab~c~ | h2b(ovvo) | lij~dab~ >
                            !hmatel = h2b_ovvo(l,c,d,k)
                            hmatel = h2b_ovvo(l,d,c,k)
                            resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                        end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate t3 buffer arrays
                  deallocate(t3_amps_buff,t3_excits_buff)

                  !!!! diagram 6: A(bc) 1/2 h2c(bcef)*t3c(aefijk)
                  k1 = 1
                  do a_chol=1,nub
                     do b_chol=a_chol+1,nub
                        ab_table(k1,1) = a_chol
                        ab_table(k1,2) = b_chol
                        k1 = k1 + 1
                     end do
                  end do
                  !!! JKIA LOOP !!!
                  ! allocate new sorting arrays
                  nloc = nua*nob*(nob-1)/2*noa
                  allocate(h_vv(nub,nub))
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nob,nob,noa,nua))
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,noa/), (/1,nua/), nob, nob, noa, nua)
                  call sort4(t3c_excits, t3c_amps, loc_arr, idx_table, (/5,6,4,1/), nob, nob, noa, nua, nloc, n3abb, resid)
                  !$omp parallel do reduction(+:resid) private(h_vv,a_chol,b_chol,a,b,c,d,i,j,k,l,m,n,e,f,idx,hmatel,idet,jdet)
                  !do a_chol=1,nub; do b_chol=a_chol+1,nub;
                  do k1=1,nub*(nub-1)/2
                  a_chol = ab_table(k1,1); b_chol = ab_table(k1,2);
                  !
                  ! get a batch of h2c_vvvv(ef)[a_chol,b_chol] integrals, where a_chol < b_chol
                  !
                  call dgemm('t','n',nub,nub,nchol,1.0d0,chol_b_vv(:,:,a_chol),nchol,chol_b_vv(:,:,b_chol),nchol,0.0d0,h_vv,nub)
                  h_vv = h_vv - transpose(h_vv)
                  do e=1,nub
                     do f=e+1,nub
                        do m=1,nob
                           do n=m+1,nob
                              h_vv(e,f) = h_vv(e,f) + h2c_oovv(m,n,e,f)*t2c(a_chol,b_chol,m,n)
                           end do
                        end do
                        h_vv(f,e) = -h_vv(e,f)
                     end do
                  end do
                  do idet = 1, n3abb
                     a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                     i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                     if (a_chol==b .and. b_chol==c) then
                     idx = idx_table(j,k,i,a)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3c_excits(jdet,2); f = t3c_excits(jdet,3);
                        ! compute < ij~k~ab~c~ | h2c(vvvv) | ij~k~ae~f~ >
                        hmatel = h_vv(e,f)
                        resid(idet) = resid(idet) + hmatel*t3c_amps(jdet)
                     end do
                     end if
                  end do
                  end do ! end loop over a_chol < b_chol
                  !$omp end parallel do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table,h_vv)

                  !!!! diagram 8: A(bc) h2b(abef)*t3c(efcijk)
                  ! allocate new sorting arrays
                  nloc = nub*nob*(nob-1)/2*noa
                  allocate(h_vv(nub,nua))
                  allocate(loc_arr1(2,nloc))
                  allocate(loc_arr2(2,nloc))
                  allocate(idx_table1(nob,nob,noa,nub))
                  allocate(idx_table2(nob,nob,noa,nub))
                  allocate(t3c_excits1(n3abb,6),t3c_excits2(n3abb,6))
                  allocate(t3c_amps1(n3abb),t3c_amps2(n3abb))
                  t3c_excits1(:,:) = t3c_excits(:,:); t3c_excits2(:,:) = t3c_excits(:,:);
                  t3c_amps1(:) = t3c_amps(:); t3c_amps2(:) = t3c_amps(:);
                  !!! JKIB LOOP !!!
                  call get_index_table(idx_table1, (/1,nob-1/), (/-1,nob/), (/1,noa/), (/1,nub-1/), nob, nob, noa, nub)
                  call sort4(t3c_excits1, t3c_amps1, loc_arr1, idx_table1, (/5,6,4,2/), nob, nob, noa, nub, nloc, n3abb)
                  !!! JKIC LOOP !!!
                  call get_index_table(idx_table2, (/1,nob-1/), (/-1,nob/), (/1,noa/), (/2,nub/), nob, nob, noa, nub)
                  call sort4(t3c_excits2, t3c_amps2, loc_arr2, idx_table2, (/5,6,4,3/), nob, nob, noa, nub, nloc, n3abb)
                  !$omp parallel do reduction(+:resid) private(b_chol,h_vv,a,b,c,d,i,j,k,l,m,n,e,f,idx,idet,jdet,hmatel)
                  do a_chol=1,nua; do b_chol=1,nub;
                  !
                  ! get a batch of h2b_vvvv(ef)[a_chol,b_chol] integrals
                  !
                  call dgemm('t','n',nub,nua,nchol,1.0d0,chol_b_vv(:,:,b_chol),nchol,chol_a_vv(:,:,a_chol),nchol,0.0d0,h_vv,nub)
                  do e=1,nua
                     do f=1,nub
                        do m=1,noa
                           do n=1,nob
                              h_vv(f,e) = h_vv(f,e) + h2b_oovv(m,n,e,f)*t2b(a_chol,b_chol,m,n)
                           end do
                        end do
                     end do
                  end do
                  do idet = 1, n3abb
                      a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      !!!! SB 1 !!!!
                      ! (1)
                      if (a_chol==a .and. b_chol==c) then
                      idx = idx_table1(j,k,i,b) ! a changes faster than c
                      do jdet = loc_arr1(1,idx), loc_arr1(2,idx)
                         d = t3c_excits1(jdet,1); f = t3c_excits1(jdet,3); ! unlike in t3b, d changes faster than f
                         ! compute < ij~k~ab~c~ | h2b(vvvv) | ij~k~db~f~ >
                         hmatel = h_vv(f,d)
                         resid(idet) = resid(idet) + hmatel*t3c_amps1(jdet)
                      end do
                      end if
                      ! (bc)
                      if (a_chol==a .and. b_chol==b) then
                      idx = idx_table1(j,k,i,c)
                      if (idx/=0) then ! protect against case where b = nua because a = 1, nua-1
                         do jdet = loc_arr1(1,idx), loc_arr1(2,idx)
                            d = t3c_excits1(jdet,1); f = t3c_excits1(jdet,3);
                            ! compute < ij~k~ab~c~ | h2b(vvvv) | ij~k~dc~f~ >
                            hmatel = -h_vv(f,d)
                            resid(idet) = resid(idet) + hmatel*t3c_amps1(jdet)
                         end do
                      end if
                      end if
                      !!!! SB 2 !!!!
                      if (a_chol==a .and. b_chol==b) then
                      idx = idx_table2(j,k,i,c)
                      do jdet = loc_arr2(1,idx), loc_arr2(2,idx)
                         d = t3c_excits2(jdet,1); e = t3c_excits2(jdet,2);
                         ! compute < ij~k~ab~c~ | h2b(vvvv) | ij~k~de~c~ >
                         hmatel = h_vv(e,d)
                         resid(idet) = resid(idet) + hmatel * t3c_amps2(jdet)
                      end do
                      end if
                      ! (bc)
                      if (a_chol==a .and. b_chol==c) then
                      idx = idx_table2(j,k,i,b)
                      if (idx/=0) then ! protect against case where a = 1 because b = 2, nua
                         do jdet = loc_arr2(1,idx), loc_arr2(2,idx)
                            d = t3c_excits2(jdet,1); e = t3c_excits2(jdet,2);
                            ! compute < ij~k~ab~c~ | h2b(vvvv) | ij~k~de~b~ >
                            hmatel = -h_vv(e,d)
                            resid(idet) = resid(idet) + hmatel * t3c_amps2(jdet)
                         end do
                      end if
                      end if
                  end do ! end loop over idet
                  end do; end do; ! end loop over a_chol, b_chol
                  !$omp end parallel do
                  ! deallocate sorting arrays
                  deallocate(h_vv)
                  deallocate(loc_arr1)
                  deallocate(loc_arr2)
                  deallocate(idx_table1)
                  deallocate(idx_table2)
                  deallocate(t3c_excits1,t3c_excits2)
                  deallocate(t3c_amps1,t3c_amps2)

                  !
                  ! Moment contributions
                  !
                  do idet = 1, n3abb
                     a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                     i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                     do e = 1, nua
                        ! A(jk)A(bc) h2B(abej) * t2b(ecik)
                        resid(idet) = resid(idet) + I2B_vvvo(e,a,b,j) * t2b(e,c,i,k)
                        resid(idet) = resid(idet) - I2B_vvvo(e,a,b,k) * t2b(e,c,i,j)
                        resid(idet) = resid(idet) - I2B_vvvo(e,a,c,j) * t2b(e,b,i,k)
                        resid(idet) = resid(idet) + I2B_vvvo(e,a,c,k) * t2b(e,b,i,j)
                     end do
                  end do

                   do idet = 1,n3abb
                      a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      do e = 1, nub
                         ! A(bc) h2B(abie) * t2c(ecjk)
                         resid(idet) = resid(idet) + I2B_vvov(e,a,b,i) * t2c(e,c,j,k)
                         resid(idet) = resid(idet) - I2B_vvov(e,a,c,i) * t2c(e,b,j,k)
                      end do
                   end do

                   allocate(xbuf(nub,nua,nob,noa))
                   do i = 1,noa
                      do j = 1,nob
                         do a = 1,nua
                            do b = 1,nub
                               xbuf(b,a,j,i) = t2b(a,b,i,j)
                            end do
                         end do
                      end do
                   end do
                   do idet = 1,n3abb
                      a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      do e = 1, nub
                         ! A(jk) h2C(cbke) * t2b(aeij)
                         resid(idet) = resid(idet) + I2C_vvov(e,c,b,k) * xbuf(e,a,j,i)
                         resid(idet) = resid(idet) - I2C_vvov(e,c,b,j) * xbuf(e,a,k,i)
                      end do
                   end do
                   deallocate(xbuf)

                   allocate(xbuf(noa,nob,nua,nub))
                   do b = 1,nub
                      do a = 1,nua
                         do j = 1,nob
                            do i = 1,noa
                               xbuf(i,j,a,b) = t2b(a,b,i,j)
                            end do
                         end do
                      end do
                   end do
                   do idet = 1,n3abb
                      a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      do m = 1, noa
                         ! -A(kj)A(bc) h2b(mbij) * t2b(acmk)
                         resid(idet) = resid(idet) - I2B_ovoo(m,b,i,j) * xbuf(m,k,a,c)
                         resid(idet) = resid(idet) + I2B_ovoo(m,c,i,j) * xbuf(m,k,a,b)
                         resid(idet) = resid(idet) + I2B_ovoo(m,b,i,k) * xbuf(m,j,a,c)
                         resid(idet) = resid(idet) - I2B_ovoo(m,c,i,k) * xbuf(m,j,a,b)
                      end do
                   end do
                   deallocate(xbuf)

                   allocate(xbuf(nob,nob,nub,nub))
                   do b = 1,nub
                      do a = 1,nub
                         do j = 1,nob
                            do i = 1,nob
                               xbuf(i,j,a,b) = t2c(a,b,i,j)
                            end do
                         end do
                      end do
                   end do
                   do idet = 1,n3abb
                      a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      do m = 1, nob
                         ! -A(jk) h2b(amij) * t2c(bcmk)
                         resid(idet) = resid(idet) - I2B_vooo(m,a,i,j) * xbuf(m,k,b,c)
                         resid(idet) = resid(idet) + I2B_vooo(m,a,i,k) * xbuf(m,j,b,c)
                      end do
                   end do
                   deallocate(xbuf)

                   allocate(xbuf(nob,noa,nub,nua))
                   do a = 1,nua
                      do b = 1,nub
                         do i = 1,noa
                            do j = 1,nob
                               xbuf(j,i,b,a) = t2b(a,b,i,j)
                            end do
                         end do
                      end do
                   end do
                   do idet = 1,n3abb
                      a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                      do m = 1, nob
                         ! -A(bc) h2c(cmkj) * t2b(abim)
                         resid(idet) = resid(idet) - I2C_vooo(m,c,k,j) * xbuf(m,i,b,a)
                         resid(idet) = resid(idet) + I2C_vooo(m,b,k,j) * xbuf(m,i,c,a)
                      end do
                   end do
                   deallocate(xbuf)

                   ! Update t3
                   do idet = 1, n3abb
                       a = t3c_excits(idet,1); b = t3c_excits(idet,2); c = t3c_excits(idet,3);
                       i = t3c_excits(idet,4); j = t3c_excits(idet,5); k = t3c_excits(idet,6);
                       denom = fA_oo(i,i) + fB_oo(j,j) + fB_oo(k,k) - fA_vv(a,a) - fB_vv(b,b) - fB_vv(c,c)
                       resid(idet) = resid(idet)/(denom - shift)
                       t3c_amps(idet) = t3c_amps(idet) + resid(idet)
                   end do

              end subroutine update_t3c_p

              subroutine update_t3d_p(resid,&
                                      t3c_amps, t3c_excits,&
                                      t3d_amps, t3d_excits,&
                                      t2c,&
                                      H1B_oo, H1B_vv,&
                                      H2C_oovv, H2C_vvov, H2C_vooo,&
                                      H2C_oooo, H2C_voov, chol_b_vv,&
                                      H2B_oovv, H2B_ovvo,&
                                      fB_oo, fB_vv,&
                                      shift,&
                                      n3abb, n3bbb,&
                                      noa, nua, nob, nub, nchol)

                  integer, intent(in) :: noa, nua, nob, nub, n3abb, n3bbb, nchol
                  integer, intent(in) :: t3c_excits(n3abb,6)
                  real(kind=8), intent(in) :: t2c(nub,nub,nob,nob),&
                                              t3c_amps(n3abb),&
                                              H1B_oo(nob,nob), H1B_vv(nub,nub),&
                                              H2B_oovv(noa,nob,nua,nub),&
                                              !H2B_ovvo(noa,nub,nua,nob),&
                                              H2B_ovvo(noa,nua,nub,nob),& ! reordered
                                              H2C_oovv(nob,nob,nub,nub),&
                                              !H2C_vooo(1:nub,1:nob,1:nob,1:nob),&
                                              H2C_vooo(nob,nub,nob,nob),& ! reordered
                                              !H2C_vvov(1:nub,1:nub,1:nob,1:nub),&
                                              H2C_vvov(nub,nub,nub,nob),& ! reordered
                                              H2C_oooo(nob,nob,nob,nob),&
                                              !H2C_voov(nub,nob,nob,nub),&
                                              H2C_voov(nob,nub,nub,nob),& ! reordered
                                              !H2C_vvvv(nub,nub,nub,nub),&
                                              chol_b_vv(nchol,nub,nub),&
                                              fB_vv(nub,nub), fB_oo(nob,nob),&
                                              shift

                  integer, intent(inout) :: t3d_excits(n3bbb,6)
                  !f2py intent(in,out) :: t3d_excits(0:n3bbb-1,0:5)
                  real(kind=8), intent(inout) :: t3d_amps(n3bbb)
                  !f2py intent(in,out) :: t3d_amps(0:n3bbb-1)

                  real(kind=8), intent(out) :: resid(n3bbb)

                  integer, allocatable :: idx_table(:,:,:,:), idx_table_copy1(:,:,:,:), idx_table_copy2(:,:,:,:), idx_table_copy3(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:), loc_arr_copy1(:,:), loc_arr_copy2(:,:), loc_arr_copy3(:,:)

                  real(kind=8), allocatable :: t3d_amps_copy1(:), t3d_amps_copy2(:), t3d_amps_copy3(:)
                  integer, allocatable :: t3d_excits_copy1(:,:), t3d_excits_copy2(:,:), t3d_excits_copy3(:,:)
                  real(kind=8), allocatable :: t3_amps_buff(:), xbuf(:,:,:,:)
                  integer, allocatable :: t3_excits_buff(:,:)

                  real(kind=8) :: val, denom, t_amp, res_mm23, hmatel
                  real(kind=8) :: hmatel1, hmatel2, hmatel3, hmatel4
                  !real(kind=8) :: I2C_vooo(nub,nob,nob,nob)
                  real(kind=8) :: I2C_vooo(nob,nub,nob,nob) ! reordered
                  !real(kind=8) :: I2C_vvov(nub,nub,nob,nub)
                  real(kind=8) :: I2C_vvov(nub,nub,nub,nob) ! reordered
                  real(kind=8) :: H2C_vvvv(nub,nub)
                  integer :: a, b, c, d, i, j, k, l, e, f, m, n, idet, jdet
                  integer :: idx, nloc, a_chol, b_chol, k1
                  integer :: ab_table(nub*(nub-1)/2,2)

                  ! compute VT3 intermediates
                  I2C_vooo(:,:,:,:) = 0.5d0 * H2C_vooo(:,:,:,:)
                  call calc_I2C_vooo(I2C_vooo,&
                               H2B_oovv,H2C_oovv,&
                               t3c_excits,t3c_amps,t3d_excits,t3d_amps,&
                               n3abb,n3bbb,noa,nua,nob,nub)
                  I2C_vvov(:,:,:,:) = 0.5d0 * H2C_vvov(:,:,:,:)
                  call calc_I2C_vvov(I2C_vvov,&
                               H2B_oovv,H2C_oovv,&
                               t3c_excits,t3c_amps,t3d_excits,t3d_amps,&
                               n3abb,n3bbb,noa,nua,nob,nub)

                  ! Zero the residual
                  resid = 0.0d0

                  !!!! diagram 1: -A(i/jk) h1b(mi) * t3d(abcmjk)
                  !!!! diagram 3: 1/2 A(i/jk) h2c(mnij) * t3d(abcmnk)
                  ! NOTE: WITHIN THESE LOOPS, H1B(OO) TERMS ARE DOUBLE-COUNTED SO COMPENSATE BY FACTOR OF 1/2
                  ! allocate new sorting arrays
                  nloc = nub*(nub-1)*(nub-2)/6*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nub,nob))
                  !!! ABCK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/-1,nub/), (/3,nob/), nub, nub, nub, nob)
                  call sort4(t3d_excits, t3d_amps, loc_arr, idx_table, (/1,2,3,6/), nub, nub, nub, nob, nloc, n3bbb, resid)
                  do idet = 1, n3bbb
                     a = t3d_excits(idet,1); b = t3d_excits(idet,2); c = t3d_excits(idet,3);
                     i = t3d_excits(idet,4); j = t3d_excits(idet,5); k = t3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        l = t3d_excits(jdet,4); m = t3d_excits(jdet,5);
                        ! compute < ijkabc | h2a(oooo) | lmkabc >
                        !hmatel = h2c_oooo(l,m,i,j)
                        hmatel = h2c_oooo(m,l,j,i)
                        ! compute < ijkabc | h1a(oo) | lmkabc > = -A(ij)A(lm) h1b_oo(l,i) * delta(m,j)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (m==j) hmatel1 = -h1b_oo(l,i) ! (1)      < ijkabc | h1a(oo) | ljkabc >
                        if (m==i) hmatel2 = h1b_oo(l,j) ! (ij)     < ijkabc | h1a(oo) | likabc >
                        if (l==j) hmatel3 = h1b_oo(m,i) ! (lm)     < ijkabc | h1a(oo) | jmkabc >
                        if (l==i) hmatel4 = -h1b_oo(m,j) ! (ij)(lm) < ijkabc | h1a(oo) | imkabc >
                        hmatel = hmatel + 0.5d0 * (hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     ! (ik)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = t3d_excits(jdet,4); m = t3d_excits(jdet,5);
                           ! compute < ijkabc | h2a(oooo) | lmiabc >
                           !hmatel = -h2c_oooo(l,m,k,j)
                           hmatel = h2c_oooo(m,l,k,j)
                           ! compute < ijkabc | h1a(oo) | lmiabc > = A(jk)A(lm) h1b_oo(l,k) * delta(m,j)
                           hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                           if (m==j) hmatel1 = h1b_oo(l,k) ! (1)      < ijkabc | h1a(oo) | ljiabc >
                           if (m==k) hmatel2 = -h1b_oo(l,j) ! (jk)     < ijkabc | h1a(oo) | lkiabc >
                           if (l==j) hmatel3 = -h1b_oo(m,k) ! (lm)
                           if (l==k) hmatel4 = h1b_oo(m,j) ! (jk)(lm)
                           hmatel = hmatel + 0.5d0 * (hmatel1 + hmatel2 + hmatel3 + hmatel4)
                           resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = t3d_excits(jdet,4); m = t3d_excits(jdet,5);
                           ! compute < ijkabc | h2a(oooo) | lmjabc >
                           !hmatel = -h2c_oooo(l,m,i,k)
                           hmatel = -h2c_oooo(m,l,k,i)
                           ! compute < ijkabc | h1a(oo) | lmjabc > = A(ik)A(lm) h1b_oo(l,i) * delta(m,k)
                           hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                           if (m==k) hmatel1 = h1b_oo(l,i) ! (1)      < ijkabc | h1a(oo) | lkjabc >
                           if (m==i) hmatel2 = -h1b_oo(l,k) ! (ik)
                           if (l==k) hmatel3 = -h1b_oo(m,i) ! (lm)
                           if (l==i) hmatel4 = h1b_oo(m,k) ! (ik)(lm)
                           hmatel = hmatel + 0.5d0 * (hmatel1 + hmatel2 + hmatel3 + hmatel4)
                           resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                        end do
                     end if
                  end do
                  !!! ABCI LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/-1,nub/), (/1,nob-2/), nub, nub, nub, nob)
                  call sort4(t3d_excits, t3d_amps, loc_arr, idx_table, (/1,2,3,4/), nub, nub, nub, nob, nloc, n3bbb, resid)
                  do idet = 1, n3bbb
                     a = t3d_excits(idet,1); b = t3d_excits(idet,2); c = t3d_excits(idet,3);
                     i = t3d_excits(idet,4); j = t3d_excits(idet,5); k = t3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,i)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = t3d_excits(jdet,5); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(oooo) | imnabc >
                        !hmatel = h2c_oooo(m,n,j,k)
                        hmatel = h2c_oooo(n,m,k,j)
                        ! compute < ijkabc | h1a(oo) | imnabc > = -A(jk)A(mn) h1b_oo(m,j) * delta(n,k)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (n==k) hmatel1 = -h1b_oo(m,j)  ! < ijkabc | h1a(oo) | imkabc >
                        if (n==j) hmatel2 = h1b_oo(m,k)
                        if (m==k) hmatel3 = h1b_oo(n,j)
                        if (m==j) hmatel4 = -h1b_oo(n,k)
                        hmatel = hmatel + 0.5d0 * (hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           m = t3d_excits(jdet,5); n = t3d_excits(jdet,6);
                           ! compute < ijkabc | h2a(oooo) | jmnabc >
                           !hmatel = -h2c_oooo(m,n,i,k)
                           hmatel = -h2c_oooo(n,m,k,i)
                           ! compute < ijkabc | h1a(oo) | jmnabc > = A(ik)A(mn) h1b_oo(m,i) * delta(n,k)
                           hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                           if (n==k) hmatel1 = h1b_oo(m,i)
                           if (n==i) hmatel2 = -h1b_oo(m,k)
                           if (m==k) hmatel3 = -h1b_oo(n,i)
                           if (m==i) hmatel4 = h1b_oo(n,k)
                           hmatel = hmatel + 0.5d0 * (hmatel1 + hmatel2 + hmatel3 + hmatel4)
                           resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                        end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           m = t3d_excits(jdet,5); n = t3d_excits(jdet,6);
                           ! compute < ijkabc | h2a(oooo) | kmnabc >
                           !hmatel = -h2c_oooo(m,n,j,i)
                           hmatel = h2c_oooo(n,m,j,i)
                           ! compute < ijkabc | h1a(oo) | kmnabc > = A(ij)A(mn) h1b_oo(m,j) * delta(n,i)
                           hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                           if (n==i) hmatel1 = -h1b_oo(m,j)
                           if (n==j) hmatel2 = h1b_oo(m,i)
                           if (m==i) hmatel3 = h1b_oo(n,j)
                           if (m==j) hmatel4 = -h1b_oo(n,i)
                           hmatel = hmatel + 0.5d0 * (hmatel1 + hmatel2 + hmatel3 + hmatel4)
                           resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                        end do
                     end if
                  end do
                  !!! ABCJ LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/-1,nub/), (/2,nob-1/), nub, nub, nub, nob)
                  call sort4(t3d_excits, t3d_amps, loc_arr, idx_table, (/1,2,3,5/), nub, nub, nub, nob, nloc, n3bbb, resid)
                  do idet = 1, n3bbb
                     a = t3d_excits(idet,1); b = t3d_excits(idet,2); c = t3d_excits(idet,3);
                     i = t3d_excits(idet,4); j = t3d_excits(idet,5); k = t3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,j)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        l = t3d_excits(jdet,4); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(oooo) | ljnabc >
                        !hmatel = h2c_oooo(l,n,i,k)
                        hmatel = h2c_oooo(n,l,k,i)
                        ! compute < ijkabc | h1a(oo) | ljnabc > = -A(ik)A(ln) h1b_oo(l,i) * delta(n,k)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (n==k) hmatel1 = -h1b_oo(l,i)
                        if (n==i) hmatel2 = h1b_oo(l,k)
                        if (l==k) hmatel3 = h1b_oo(n,i)
                        if (l==i) hmatel4 = -h1b_oo(n,k)
                        hmatel = hmatel + 0.5d0 * (hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = t3d_excits(jdet,4); n = t3d_excits(jdet,6);
                           ! compute < ijkabc | h2a(oooo) | linabc >
                           !hmatel = -h2c_oooo(l,n,j,k)
                           hmatel = -h2c_oooo(n,l,k,j)
                           ! compute < ijkabc | h1a(oo) | linabc > = A(jk)A(ln) h1b_oo(l,j) * delta(n,k)
                           hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                           if (n==k) hmatel1 = h1b_oo(l,j)
                           if (n==j) hmatel2 = -h1b_oo(l,k)
                           if (l==k) hmatel3 = -h1b_oo(n,j)
                           if (l==j) hmatel4 = h1b_oo(n,k)
                           hmatel = hmatel + 0.5d0 * (hmatel1 + hmatel2 + hmatel3 + hmatel4)
                           resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = t3d_excits(jdet,4); n = t3d_excits(jdet,6);
                           ! compute < ijkabc | h2a(oooo) | lknabc >
                           !hmatel = -h2c_oooo(l,n,i,j)
                           hmatel = -h2c_oooo(n,l,j,i)
                           ! compute < ijkabc | h1a(oo) | lknabc > = A(ij)A(ln) h1b_oo(l,i) * delta(n,j)
                           hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                           if (n==j) hmatel1 = h1b_oo(l,i)
                           if (n==i) hmatel2 = -h1b_oo(l,j)
                           if (l==j) hmatel3 = -h1b_oo(n,i)
                           if (l==i) hmatel4 = h1b_oo(n,j)
                           hmatel = hmatel + 0.5d0 * (hmatel1 + hmatel2 + hmatel3 + hmatel4)
                           resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                        end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 2: A(a/bc) h1b(ae) * t3d(ebcijk)
                  ! NOTE: WITHIN THESE LOOPS, H1B(VV) TERMS ARE DOUBLE-COUNTED SO COMPENSATE BY FACTOR OF 1/2
                  ! allocate new sorting arrays
                  nloc = nob*(nob-1)*(nob-2)/6*nub
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nob,nob,nob,nub))
                  !!! IJKA LOOP !!!
                  call get_index_table(idx_table, (/1,nob-2/), (/-1,nob-1/), (/-1,nob/), (/1,nub-2/), nob, nob, nob, nub)
                  call sort4(t3d_excits, t3d_amps, loc_arr, idx_table, (/4,5,6,1/), nob, nob, nob, nub, nloc, n3bbb, resid)
                  do idet = 1, n3bbb
                     a = t3d_excits(idet,1); b = t3d_excits(idet,2); c = t3d_excits(idet,3);
                     i = t3d_excits(idet,4); j = t3d_excits(idet,5); k = t3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,a)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); f = t3d_excits(jdet,3);
                        ! compute < ijkabc | h1a(vv) | ijkaef > = A(bc)A(ef) h1b_vv(b,e) * delta(c,f)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (c==f) hmatel1 = h1b_vv(e,b)  !h1b_vv(b,e) ! (1)
                        if (b==f) hmatel2 = -h1b_vv(e,c) !-h1b_vv(c,e) ! (bc)
                        if (c==e) hmatel3 = -h1b_vv(f,b) !-h1b_vv(b,f) ! (ef)
                        if (b==e) hmatel4 = h1b_vv(f,c)  !h1b_vv(c,f) ! (bc)(ef)
                        hmatel = 0.5d0 * (hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); f = t3d_excits(jdet,3);
                        ! compute < ijkabc | h1a(vv) | ijkbef > = -A(ac)A(ef) h1b_vv(a,e) * delta(c,f)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (c==f) hmatel1 = -h1b_vv(e,a) !-h1b_vv(a,e) ! (1)
                        if (a==f) hmatel2 = h1b_vv(e,c)  !h1b_vv(c,e) ! (ac)
                        if (c==e) hmatel3 = h1b_vv(f,a)  !h1b_vv(a,f) ! (ef)
                        if (a==e) hmatel4 = -h1b_vv(f,c) !-h1b_vv(c,f) ! (ac)(ef)
                        hmatel = 0.5d0 * (hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); f = t3d_excits(jdet,3);
                        ! compute < ijkabc | h1a(vv) | ijkcef > = -A(ab)A(ef) h1b_vv(b,e) * delta(a,f)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (a==f) hmatel1 = -h1b_vv(e,b) !-h1b_vv(b,e) ! (1)
                        if (b==f) hmatel2 = h1b_vv(e,a)  !h1b_vv(a,e) ! (ab)
                        if (a==e) hmatel3 = h1b_vv(f,b)  !h1b_vv(b,f) ! (ef)
                        if (b==e) hmatel4 = -h1b_vv(f,a) !-h1b_vv(a,f) ! (ab)(ef)
                        hmatel = 0.5d0 * (hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                  end do
                  !!! IJKB LOOP !!!
                  call get_index_table(idx_table, (/1,nob-2/), (/-1,nob-1/), (/-1,nob/), (/2,nub-1/), nob, nob, nob, nub)
                  call sort4(t3d_excits, t3d_amps, loc_arr, idx_table, (/4,5,6,2/), nob, nob, nob, nub, nloc, n3bbb, resid)
                  do idet = 1, n3bbb
                     a = t3d_excits(idet,1); b = t3d_excits(idet,2); c = t3d_excits(idet,3);
                     i = t3d_excits(idet,4); j = t3d_excits(idet,5); k = t3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); f = t3d_excits(jdet,3);
                        ! compute < ijkabc | h1a(vv) | ijkdbf > = A(ac)A(df) h1b_vv(a,d) * delta(c,f)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (c==f) hmatel1 = h1b_vv(d,a)  !h1b_vv(a,d) ! (1)
                        if (a==f) hmatel2 = -h1b_vv(d,c) !-h1b_vv(c,d) ! (ac)
                        if (c==d) hmatel3 = -h1b_vv(f,a) !-h1b_vv(a,f) ! (df)
                        if (a==d) hmatel4 = h1b_vv(f,c)  !h1b_vv(c,f) ! (ac)(df)
                        hmatel = 0.5d0 * (hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); f = t3d_excits(jdet,3);
                        ! compute < ijkabc | h1a(vv) | ijkdaf > = -A(bc)A(df) h1b_vv(b,d) * delta(c,f)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (c==f) hmatel1 = -h1b_vv(d,b) !-h1b_vv(b,d) ! (1)
                        if (b==f) hmatel2 = h1b_vv(d,c)  !h1b_vv(c,d) ! (bc)
                        if (c==d) hmatel3 = h1b_vv(f,b)  !h1b_vv(b,f) ! (df)
                        if (b==d) hmatel4 = -h1b_vv(f,c) !-h1b_vv(c,f) ! (bc)(df)
                        hmatel = 0.5d0 * (hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); f = t3d_excits(jdet,3);
                        ! compute < ijkabc | h1a(vv) | ijkdcf > = -A(ab)A(df) h1b_vv(a,d) * delta(b,f)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (b==f) hmatel1 = -h1b_vv(d,a) !-h1b_vv(a,d) ! (1)
                        if (a==f) hmatel2 = h1b_vv(d,b)  !h1b_vv(b,d) ! (ab)
                        if (b==d) hmatel3 = h1b_vv(f,a)  !h1b_vv(a,f) ! (df)
                        if (a==d) hmatel4 = -h1b_vv(f,b) !-h1b_vv(b,f) ! (ab)(df)
                        hmatel = 0.5d0 * (hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                  end do
                  !!! IJKC LOOP !!!
                  call get_index_table(idx_table, (/1,nob-2/), (/-1,nob-1/), (/-1,nob/), (/3,nub/), nob, nob, nob, nub)
                  call sort4(t3d_excits, t3d_amps, loc_arr, idx_table, (/4,5,6,3/), nob, nob, nob, nub, nloc, n3bbb, resid)
                  do idet = 1, n3bbb
                     a = t3d_excits(idet,1); b = t3d_excits(idet,2); c = t3d_excits(idet,3);
                     i = t3d_excits(idet,4); j = t3d_excits(idet,5); k = t3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,c)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); e = t3d_excits(jdet,2);
                        ! compute < ijkabc | h1a(vv) | ijkdec > = A(ab)A(de) h1b_vv(a,d) * delta(b,e)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (b==e) hmatel1 = h1b_vv(d,a)  !h1b_vv(a,d) ! (1)
                        if (a==e) hmatel2 = -h1b_vv(d,b) !-h1b_vv(b,d) ! (ab)
                        if (b==d) hmatel3 = -h1b_vv(e,a) !-h1b_vv(a,e) ! (de)
                        if (a==d) hmatel4 = h1b_vv(e,b)  !h1b_vv(b,e) ! (ab)(de)
                        hmatel = 0.5d0 * (hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); e = t3d_excits(jdet,2);
                        ! compute < ijkabc | h1a(vv) | ijkdea > = -A(bc)A(de) h1b_vv(c,d) * delta(b,e)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (b==e) hmatel1 = -h1b_vv(d,c) !-h1b_vv(c,d) ! (1)
                        if (c==e) hmatel2 = h1b_vv(d,b)  !h1b_vv(b,d) ! (bc)
                        if (b==d) hmatel3 = h1b_vv(e,c)  !h1b_vv(c,e) ! (de)
                        if (c==d) hmatel4 = -h1b_vv(e,b) !-h1b_vv(b,e) ! (bc)(de)
                        hmatel = 0.5d0 * (hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); e = t3d_excits(jdet,2);
                        ! compute < ijkabc | h1a(vv) | ijkdeb > = -A(ac)A(de) h1b_vv(a,d) * delta(c,e)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (c==e) hmatel1 = -h1b_vv(d,a) !-h1b_vv(a,d) ! (1)
                        if (a==e) hmatel2 = h1b_vv(d,c)  !h1b_vv(c,d) ! (ac)
                        if (c==d) hmatel3 = h1b_vv(e,a)  !h1b_vv(a,e) ! (de)
                        if (a==d) hmatel4 = -h1b_vv(e,c) !-h1b_vv(c,e) ! (ac)(de)
                        hmatel = 0.5d0 * (hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 5: A(i/jk)A(a/bc) h2c(amie) * t3d(ebcmjk)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = (nub-1)*(nub-2)/2*(nob-1)*(nob-2)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nob,nob))
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4(t3d_excits, t3d_amps, loc_arr, idx_table, (/1,2,4,5/), nub, nub, nob, nob, nloc, n3bbb, resid)
                  do idet = 1, n3bbb
                     a = t3d_excits(idet,1); b = t3d_excits(idet,2); c = t3d_excits(idet,3);
                     i = t3d_excits(idet,4); j = t3d_excits(idet,5); k = t3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3d_excits(jdet,3); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | ijnabf >
                        !hmatel = h2c_voov(c,n,k,f)
                        hmatel = h2c_voov(n,f,c,k)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3d_excits(jdet,3); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | ijnbcf >
                        !hmatel = h2c_voov(a,n,k,f)
                        hmatel = h2c_voov(n,f,a,k)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3d_excits(jdet,3); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | ijnacf >
                        !hmatel = -h2c_voov(b,n,k,f)
                        hmatel = -h2c_voov(n,f,b,k)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3d_excits(jdet,3); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | jknabf >
                        !hmatel = h2c_voov(c,n,i,f)
                        hmatel = h2c_voov(n,f,c,i)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3d_excits(jdet,3); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | jknbcf >
                        !hmatel = h2c_voov(a,n,i,f)
                        hmatel = h2c_voov(n,f,a,i)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3d_excits(jdet,3); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | jknacf >
                        !hmatel = -h2c_voov(b,n,i,f)
                        hmatel = -h2c_voov(n,f,b,i)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3d_excits(jdet,3); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | iknabf >
                        !hmatel = -h2c_voov(c,n,j,f)
                        hmatel = -h2c_voov(n,f,c,j)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3d_excits(jdet,3); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | iknbcf >
                        !hmatel = -h2c_voov(a,n,j,f)
                        hmatel = -h2c_voov(n,f,a,j)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3d_excits(jdet,3); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | iknacf >
                        !hmatel = h2c_voov(b,n,j,f)
                        hmatel = h2c_voov(n,f,b,j)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                  end do
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4(t3d_excits, t3d_amps, loc_arr, idx_table, (/1,3,4,5/), nub, nub, nob, nob, nloc, n3bbb, resid)
                  do idet = 1, n3bbb
                     a = t3d_excits(idet,1); b = t3d_excits(idet,2); c = t3d_excits(idet,3);
                     i = t3d_excits(idet,4); j = t3d_excits(idet,5); k = t3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | ijnaec >
                        !hmatel = h2c_voov(b,n,k,e)
                        hmatel = h2c_voov(n,e,b,k)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | ijnbec >
                        !hmatel = -h2c_voov(a,n,k,e)
                        hmatel = -h2c_voov(n,e,a,k)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | ijnaeb >
                        !hmatel = -h2c_voov(c,n,k,e)
                        hmatel = -h2c_voov(n,e,c,k)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | jknaec >
                        !hmatel = h2c_voov(b,n,i,e)
                        hmatel = h2c_voov(n,e,b,i)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | jknbec >
                        !hmatel = -h2c_voov(a,n,i,e)
                        hmatel = -h2c_voov(n,e,a,i)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | jknaeb >
                        !hmatel = -h2c_voov(c,n,i,e)
                        hmatel = -h2c_voov(n,e,c,i)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | iknaec >
                        !hmatel = -h2c_voov(b,n,j,e)
                        hmatel = -h2c_voov(n,e,b,j)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | iknbec >
                        !hmatel = h2c_voov(a,n,j,e)
                        hmatel = h2c_voov(n,e,a,j)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | iknaeb >
                        !hmatel = h2c_voov(c,n,j,e)
                        hmatel = h2c_voov(n,e,c,j)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                  end do
                  !!! BCIJ LOOP !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4(t3d_excits, t3d_amps, loc_arr, idx_table, (/2,3,4,5/), nub, nub, nob, nob, nloc, n3bbb, resid)
                  do idet = 1, n3bbb
                     a = t3d_excits(idet,1); b = t3d_excits(idet,2); c = t3d_excits(idet,3);
                     i = t3d_excits(idet,4); j = t3d_excits(idet,5); k = t3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | ijndbc >
                        !hmatel = h2c_voov(a,n,k,d)
                        hmatel = h2c_voov(n,d,a,k)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | ijndac >
                        !hmatel = -h2c_voov(b,n,k,d)
                        hmatel = -h2c_voov(n,d,b,k)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | ijndab >
                        !hmatel = h2c_voov(c,n,k,d)
                        hmatel = h2c_voov(n,d,c,k)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | jkndbc >
                        !hmatel = h2c_voov(a,n,i,d)
                        hmatel = h2c_voov(n,d,a,i)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | jkndac >
                        !hmatel = -h2c_voov(b,n,i,d)
                        hmatel = -h2c_voov(n,d,b,i)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | jkndab >
                        !hmatel = h2c_voov(c,n,i,d)
                        hmatel = h2c_voov(n,d,c,i)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | ikndbc >
                        !hmatel = -h2c_voov(a,n,j,d)
                        hmatel = -h2c_voov(n,d,a,j)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | ikndac >
                        !hmatel = h2c_voov(b,n,j,d)
                        hmatel = h2c_voov(n,d,b,j)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); n = t3d_excits(jdet,6);
                        ! compute < ijkabc | h2a(voov) | ikndab >
                        !hmatel = -h2c_voov(c,n,j,d)
                        hmatel = -h2c_voov(n,d,c,j)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                  end do
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4(t3d_excits, t3d_amps, loc_arr, idx_table, (/1,2,4,6/), nub, nub, nob, nob, nloc, n3bbb, resid)
                  do idet = 1, n3bbb
                     a = t3d_excits(idet,1); b = t3d_excits(idet,2); c = t3d_excits(idet,3);
                     i = t3d_excits(idet,4); j = t3d_excits(idet,5); k = t3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3d_excits(jdet,3); m = t3d_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imkabf >
                        !hmatel = h2c_voov(c,m,j,f)
                        hmatel = h2c_voov(m,f,c,j)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3d_excits(jdet,3); m = t3d_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imkbcf >
                        !hmatel = h2c_voov(a,m,j,f)
                        hmatel = h2c_voov(m,f,a,j)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3d_excits(jdet,3); m = t3d_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imkacf >
                        !hmatel = -h2c_voov(b,m,j,f)
                        hmatel = -h2c_voov(m,f,b,j)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3d_excits(jdet,3); m = t3d_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | jmkabf >
                        !hmatel = -h2c_voov(c,m,i,f)
                        hmatel = -h2c_voov(m,f,c,i)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3d_excits(jdet,3); m = t3d_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | jmkbcf >
                        !hmatel = -h2c_voov(a,m,i,f)
                        hmatel = -h2c_voov(m,f,a,i)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3d_excits(jdet,3); m = t3d_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | jmkacf >
                        !hmatel = h2c_voov(b,m,i,f)
                        hmatel = h2c_voov(m,f,b,i)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3d_excits(jdet,3); m = t3d_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imjabf >
                        !hmatel = -h2c_voov(c,m,k,f)
                        hmatel = -h2c_voov(m,f,c,k)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3d_excits(jdet,3); m = t3d_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imjbcf >
                        !hmatel = -h2c_voov(a,m,k,f)
                        hmatel = -h2c_voov(m,f,a,k)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3d_excits(jdet,3); m = t3d_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imjacf >
                        !hmatel = h2c_voov(b,m,k,f)
                        hmatel = h2c_voov(m,f,b,k)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                  end do
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4(t3d_excits, t3d_amps, loc_arr, idx_table, (/1,3,4,6/), nub, nub, nob, nob, nloc, n3bbb, resid)
                  do idet = 1, n3bbb
                     a = t3d_excits(idet,1); b = t3d_excits(idet,2); c = t3d_excits(idet,3);
                     i = t3d_excits(idet,4); j = t3d_excits(idet,5); k = t3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); m = t3d_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imkaec >
                        !hmatel = h2c_voov(b,m,j,e)
                        hmatel = h2c_voov(m,e,b,j)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); m = t3d_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imkbec >
                        !hmatel = -h2c_voov(a,m,j,e)
                        hmatel = -h2c_voov(m,e,a,j)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); m = t3d_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imkaeb >
                        !hmatel = -h2c_voov(c,m,j,e)
                        hmatel = -h2c_voov(m,e,c,j)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); m = t3d_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | jmkaec >
                        !hmatel = -h2c_voov(b,m,i,e)
                        hmatel = -h2c_voov(m,e,b,i)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); m = t3d_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | jmkbec >
                        !hmatel = h2c_voov(a,m,i,e)
                        hmatel = h2c_voov(m,e,a,i)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); m = t3d_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | jmkaeb >
                        !hmatel = h2c_voov(c,m,i,e)
                        hmatel = h2c_voov(m,e,c,i)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); m = t3d_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imjaec >
                        !hmatel = -h2c_voov(b,m,k,e)
                        hmatel = -h2c_voov(m,e,b,k)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); m = t3d_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imjbec >
                        !hmatel = h2c_voov(a,m,k,e)
                        hmatel = h2c_voov(m,e,a,k)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); m = t3d_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imjaeb >
                        !hmatel = h2c_voov(c,m,k,e)
                        hmatel = h2c_voov(m,e,c,k)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                  end do
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4(t3d_excits, t3d_amps, loc_arr, idx_table, (/2,3,4,6/), nub, nub, nob, nob, nloc, n3bbb, resid)
                  do idet = 1, n3bbb
                     a = t3d_excits(idet,1); b = t3d_excits(idet,2); c = t3d_excits(idet,3);
                     i = t3d_excits(idet,4); j = t3d_excits(idet,5); k = t3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); m = t3d_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imkdbc >
                        !hmatel = h2c_voov(a,m,j,d)
                        hmatel = h2c_voov(m,d,a,j)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); m = t3d_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imkdac >
                        !hmatel = -h2c_voov(b,m,j,d)
                        hmatel = -h2c_voov(m,d,b,j)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); m = t3d_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imkdab >
                        !hmatel = h2c_voov(c,m,j,d)
                        hmatel = h2c_voov(m,d,c,j)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); m = t3d_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | jmkdbc >
                        !hmatel = -h2c_voov(a,m,i,d)
                        hmatel = -h2c_voov(m,d,a,i)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); m = t3d_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | jmkdac >
                        !hmatel = h2c_voov(b,m,i,d)
                        hmatel = h2c_voov(m,d,b,i)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); m = t3d_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | jmkdab >
                        !hmatel = -h2c_voov(c,m,i,d)
                        hmatel = -h2c_voov(m,d,c,i)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); m = t3d_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imjdbc >
                        !hmatel = -h2c_voov(a,m,k,d)
                        hmatel = -h2c_voov(m,d,a,k)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); m = t3d_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imjdac >
                        !hmatel = h2c_voov(b,m,k,d)
                        hmatel = h2c_voov(m,d,b,k)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); m = t3d_excits(jdet,5);
                        ! compute < ijkabc | h2a(voov) | imjdab >
                        !hmatel = -h2c_voov(c,m,k,d)
                        hmatel = -h2c_voov(m,d,c,k)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                  end do
                  !!! ABJK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(t3d_excits, t3d_amps, loc_arr, idx_table, (/1,2,5,6/), nub, nub, nob, nob, nloc, n3bbb, resid)
                  do idet = 1, n3bbb
                     a = t3d_excits(idet,1); b = t3d_excits(idet,2); c = t3d_excits(idet,3);
                     i = t3d_excits(idet,4); j = t3d_excits(idet,5); k = t3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3d_excits(jdet,3); l = t3d_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | ljkabf >
                        !hmatel = h2c_voov(c,l,i,f)
                        hmatel = h2c_voov(l,f,c,i)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3d_excits(jdet,3); l = t3d_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | ljkbcf >
                        !hmatel = h2c_voov(a,l,i,f)
                        hmatel = h2c_voov(l,f,a,i)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3d_excits(jdet,3); l = t3d_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | ljkacf >
                        !hmatel = -h2c_voov(b,l,i,f)
                        hmatel = -h2c_voov(l,f,b,i)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3d_excits(jdet,3); l = t3d_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | likabf >
                        !hmatel = -h2c_voov(c,l,j,f)
                        hmatel = -h2c_voov(l,f,c,j)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3d_excits(jdet,3); l = t3d_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | likbcf >
                        !hmatel = -h2c_voov(a,l,j,f)
                        hmatel = -h2c_voov(l,f,a,j)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3d_excits(jdet,3); l = t3d_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | likacf >
                        !hmatel = h2c_voov(b,l,j,f)
                        hmatel = h2c_voov(l,f,b,j)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3d_excits(jdet,3); l = t3d_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | lijabf >
                        !hmatel = h2c_voov(c,l,k,f)
                        hmatel = h2c_voov(l,f,c,k)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3d_excits(jdet,3); l = t3d_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | lijbcf >
                        !hmatel = h2c_voov(a,l,k,f)
                        hmatel = h2c_voov(l,f,a,k)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3d_excits(jdet,3); l = t3d_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | lijacf >
                        !hmatel = -h2c_voov(b,l,k,f)
                        hmatel = -h2c_voov(l,f,b,k)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                  end do
                  !!! ACJK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(t3d_excits, t3d_amps, loc_arr, idx_table, (/1,3,5,6/), nub, nub, nob, nob, nloc, n3bbb, resid)
                  do idet = 1, n3bbb
                     a = t3d_excits(idet,1); b = t3d_excits(idet,2); c = t3d_excits(idet,3);
                     i = t3d_excits(idet,4); j = t3d_excits(idet,5); k = t3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); l = t3d_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | ljkaec >
                        !hmatel = h2c_voov(b,l,i,e)
                        hmatel = h2c_voov(l,e,b,i)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); l = t3d_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | ljkbec >
                        !hmatel = -h2c_voov(a,l,i,e)
                        hmatel = -h2c_voov(l,e,a,i)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); l = t3d_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | ljkaeb >
                        !hmatel = -h2c_voov(c,l,i,e)
                        hmatel = -h2c_voov(l,e,c,i)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); l = t3d_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | likaec >
                        !hmatel = -h2c_voov(b,l,j,e)
                        hmatel = -h2c_voov(l,e,b,j)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); l = t3d_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | likbec >
                        !hmatel = h2c_voov(a,l,j,e)
                        hmatel = h2c_voov(l,e,a,j)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); l = t3d_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | likaeb >
                        !hmatel = h2c_voov(c,l,j,e)
                        hmatel = h2c_voov(l,e,c,j)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); l = t3d_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | lijaec >
                        !hmatel = h2c_voov(b,l,k,e)
                        hmatel = h2c_voov(l,e,b,k)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); l = t3d_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | lijbec >
                        !hmatel = -h2c_voov(a,l,k,e)
                        hmatel = -h2c_voov(l,e,a,k)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3d_excits(jdet,2); l = t3d_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | lijaeb >
                        !hmatel = -h2c_voov(c,l,k,e)
                        hmatel = -h2c_voov(l,e,c,k)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                  end do
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(t3d_excits, t3d_amps, loc_arr, idx_table, (/2,3,5,6/), nub, nub, nob, nob, nloc, n3bbb, resid)
                  do idet = 1, n3bbb
                     a = t3d_excits(idet,1); b = t3d_excits(idet,2); c = t3d_excits(idet,3);
                     i = t3d_excits(idet,4); j = t3d_excits(idet,5); k = t3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); l = t3d_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | ljkdbc >
                        !hmatel = h2c_voov(a,l,i,d)
                        hmatel = h2c_voov(l,d,a,i)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); l = t3d_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | ljkdac >
                        !hmatel = -h2c_voov(b,l,i,d)
                        hmatel = -h2c_voov(l,d,b,i)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); l = t3d_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | ljkdab >
                        !hmatel = h2c_voov(c,l,i,d)
                        hmatel = h2c_voov(l,d,c,i)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); l = t3d_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | likdbc >
                        !hmatel = -h2c_voov(a,l,j,d)
                        hmatel = -h2c_voov(l,d,a,j)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); l = t3d_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | likdac >
                        !hmatel = h2c_voov(b,l,j,d)
                        hmatel = h2c_voov(l,d,b,j)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); l = t3d_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | likdab >
                        !hmatel = -h2c_voov(c,l,j,d)
                        hmatel = -h2c_voov(l,d,c,j)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); l = t3d_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | lijdbc >
                        !hmatel = h2c_voov(a,l,k,d)
                        hmatel = h2c_voov(l,d,a,k)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); l = t3d_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | lijdac >
                        !hmatel = -h2c_voov(b,l,k,d)
                        hmatel = -h2c_voov(l,d,b,k)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3d_excits(jdet,1); l = t3d_excits(jdet,4);
                        ! compute < ijkabc | h2a(voov) | lijdab >
                        !hmatel = h2c_voov(c,l,k,d)
                        hmatel = h2c_voov(l,d,c,k)
                        resid(idet) = resid(idet) + hmatel * t3d_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 6: A(i/jk)A(a/bc) h2b(maei) * t3c(ebcmjk)
                  ! allocate and copy over t3c arrays
                  allocate(t3_amps_buff(n3abb),t3_excits_buff(n3abb,6))
                  t3_amps_buff(:) = t3c_amps(:)
                  t3_excits_buff(:,:) = t3c_excits(:,:)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = nub*(nub-1)/2*nob*(nob-1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nob,nob))
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/2,3,5,6/), nub, nub, nob, nob, nloc, n3abb)
                  do idet = 1, n3bbb
                     a = t3d_excits(idet,1); b = t3d_excits(idet,2); c = t3d_excits(idet,3);
                     i = t3d_excits(idet,4); j = t3d_excits(idet,5); k = t3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits_buff(jdet,1); l = t3_excits_buff(jdet,4);
                        ! compute < i~j~k~a~b~c~ | h2b(ovvo) | lj~k~db~c~ >
                        !hmatel = h2b_ovvo(l,a,d,i)
                        hmatel = h2b_ovvo(l,d,a,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits_buff(jdet,1); l = t3_excits_buff(jdet,4);
                        ! compute < i~j~k~a~b~c~ | h2b(ovvo) | lj~k~da~c~ >
                        !hmatel = -h2b_ovvo(l,b,d,i)
                        hmatel = -h2b_ovvo(l,d,b,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits_buff(jdet,1); l = t3_excits_buff(jdet,4);
                        ! compute < i~j~k~a~b~c~ | h2b(ovvo) | lj~k~da~b~ >
                        !hmatel = h2b_ovvo(l,c,d,i)
                        hmatel = h2b_ovvo(l,d,c,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits_buff(jdet,1); l = t3_excits_buff(jdet,4);
                        ! compute < i~j~k~a~b~c~ | h2b(ovvo) | li~k~db~c~ >
                        !hmatel = -h2b_ovvo(l,a,d,j)
                        hmatel = -h2b_ovvo(l,d,a,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits_buff(jdet,1); l = t3_excits_buff(jdet,4);
                        ! compute < i~j~k~a~b~c~ | h2b(ovvo) | li~k~da~c~ >
                        !hmatel = h2b_ovvo(l,b,d,j)
                        hmatel = h2b_ovvo(l,d,b,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits_buff(jdet,1); l = t3_excits_buff(jdet,4);
                        ! compute < i~j~k~a~b~c~ | h2b(ovvo) | li~k~da~b~ >
                        !hmatel = -h2b_ovvo(l,c,d,j)
                        hmatel = -h2b_ovvo(l,d,c,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits_buff(jdet,1); l = t3_excits_buff(jdet,4);
                        ! compute < i~j~k~a~b~c~ | h2b(ovvo) | li~j~db~c~ >
                        !hmatel = h2b_ovvo(l,a,d,k)
                        hmatel = h2b_ovvo(l,d,a,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits_buff(jdet,1); l = t3_excits_buff(jdet,4);
                        ! compute < i~j~k~a~b~c~ | h2b(ovvo) | li~j~da~c~ >
                        !hmatel = -h2b_ovvo(l,b,d,k)
                        hmatel = -h2b_ovvo(l,d,b,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits_buff(jdet,1); l = t3_excits_buff(jdet,4);
                        ! compute < i~j~k~a~b~c~ | h2b(ovvo) | li~j~da~b~ >
                        !hmatel = h2b_ovvo(l,c,d,k)
                        hmatel = h2b_ovvo(l,d,c,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps_buff(jdet)
                     end do
                     end if
                 end do
                 ! deallocate sorting arrays
                 deallocate(loc_arr,idx_table)
                 ! deallocate t3 buffer arrays
                 deallocate(t3_amps_buff,t3_excits_buff)

                  !!!! diagram 4: 1/2 A(c/ab) h2c(abef) * t3d(ebcijk)
                  k1 = 1
                  do a_chol=1,nub
                     do b_chol=a_chol+1,nub
                        ab_table(k1,1) = a_chol
                        ab_table(k1,2) = b_chol
                        k1 = k1 + 1
                     end do
                  end do
                  ! allocate new sorting arrays
                  nloc = nob*(nob-1)*(nob-2)/6*nub
                  allocate(loc_arr_copy1(2,nloc),loc_arr_copy2(2,nloc),loc_arr_copy3(2,nloc))
                  allocate(idx_table_copy1(nob,nob,nob,nub),idx_table_copy2(nob,nob,nob,nub),idx_table_copy3(nob,nob,nob,nub))
                  allocate(t3d_excits_copy1(n3bbb,6),t3d_excits_copy2(n3bbb,6),t3d_excits_copy3(n3bbb,6))
                  allocate(t3d_amps_copy1(n3bbb),t3d_amps_copy2(n3bbb),t3d_amps_copy3(n3bbb))
                  t3d_excits_copy1(:,:) = t3d_excits(:,:)
                  t3d_excits_copy2(:,:) = t3d_excits(:,:)
                  t3d_excits_copy3(:,:) = t3d_excits(:,:)
                  t3d_amps_copy1(:) = t3d_amps(:)
                  t3d_amps_copy2(:) = t3d_amps(:)
                  t3d_amps_copy3(:) = t3d_amps(:)
                  !!! IJKA LOOP !!!
                  call get_index_table(idx_table_copy1, (/1,nob-2/), (/-1,nob-1/), (/-1,nob/), (/1,nub-2/), nob, nob, nob, nub)
                  call sort4(t3d_excits_copy1, t3d_amps_copy1, loc_arr_copy1, idx_table_copy1, (/4,5,6,1/), nob, nob, nob, nub, nloc, n3bbb)
                  !!! IJKB LOOP !!!
                  call get_index_table(idx_table_copy2, (/1,nob-2/), (/-1,nob-1/), (/-1,nob/), (/2,nub-1/), nob, nob, nob, nub)
                  call sort4(t3d_excits_copy2, t3d_amps_copy2, loc_arr_copy2, idx_table_copy2, (/4,5,6,2/), nob, nob, nob, nub, nloc, n3bbb)
                  !!! IJKC LOOP !!!
                  call get_index_table(idx_table_copy3, (/1,nob-2/), (/-1,nob-1/), (/-1,nob/), (/3,nub/), nob, nob, nob, nub)
                  call sort4(t3d_excits_copy3, t3d_amps_copy3, loc_arr_copy3, idx_table_copy3, (/4,5,6,3/), nob, nob, nob, nub, nloc, n3bbb)
                  !do a_chol = 1,nub
                  !   do b_chol = a_chol+1,nub
                  !$omp parallel do reduction(+:resid) private(a_chol,b_chol,h2c_vvvv,a,b,c,d,i,j,k,l,m,n,e,f,idx,idet,jdet,hmatel)
                  do k1=1,nub*(nub-1)/2
                     a_chol = ab_table(k1,1); b_chol = ab_table(k1,2);
                        !
                        ! get a batch of h2c_vvvv(ef)[a_chol,b_chol] integrals, where a_chol < b_chol
                        !
                        call dgemm('t','n',nub,nub,nchol,1.0d0,chol_b_vv(:,:,a_chol),nchol,chol_b_vv(:,:,b_chol),nchol,0.0d0,h2c_vvvv,nub)
                        h2c_vvvv = h2c_vvvv - transpose(h2c_vvvv)
                        do e=1,nub
                           do f=e+1,nub
                              do m=1,nob
                                 do n=m+1,nob
                                    h2c_vvvv(e,f) = h2c_vvvv(e,f) + h2c_oovv(m,n,e,f)*t2c(a_chol,b_chol,m,n)
                                 end do
                              end do
                              h2c_vvvv(f,e) = -h2c_vvvv(e,f)
                           end do
                        end do
                        do idet = 1, n3bbb ! master copy loop
                           a = t3d_excits(idet,1); b = t3d_excits(idet,2); c = t3d_excits(idet,3);
                           i = t3d_excits(idet,4); j = t3d_excits(idet,5); k = t3d_excits(idet,6);
                           !
                           !
                           !!!! SB 1 !!!!
                           ! (1)
                           if (a_chol==b .and. b_chol==c) then
                              idx = idx_table_copy1(i,j,k,a)
                              do jdet = loc_arr_copy1(1,idx), loc_arr_copy1(2,idx)
                                 e = t3d_excits_copy1(jdet,2); f = t3d_excits_copy1(jdet,3);
                                 ! compute < ijkabc | h2a(vvvv) | ijkaef >
                                 hmatel = h2c_vvvv(e,f)
                                 resid(idet) = resid(idet) + hmatel*t3d_amps_copy1(jdet)
                              end do
                           end if
                           ! (ab)
                           if (a_chol==a .and. b_chol==c) then
                              idx = idx_table_copy1(i,j,k,b)
                              if (idx/=0) then
                                 do jdet = loc_arr_copy1(1,idx), loc_arr_copy1(2,idx)
                                    e = t3d_excits_copy1(jdet,2); f = t3d_excits_copy1(jdet,3);
                                    ! compute < ijkabc | h2a(vvvv) | ijkbef >
                                    hmatel = -h2c_vvvv(e,f)
                                    resid(idet) = resid(idet) + hmatel*t3d_amps_copy1(jdet)
                                 end do
                              end if
                           end if
                           ! (ac)
                           if (a_chol==a .and. b_chol==b) then
                              idx = idx_table_copy1(i,j,k,c)
                              if (idx/=0) then
                                 do jdet = loc_arr_copy1(1,idx), loc_arr_copy1(2,idx)
                                    e = t3d_excits_copy1(jdet,2); f = t3d_excits_copy1(jdet,3);
                                    ! compute < ijkabc | h2a(vvvv) | ijkcef >
                                    hmatel = h2c_vvvv(e,f)
                                    resid(idet) = resid(idet) + hmatel*t3d_amps_copy1(jdet)
                                 end do
                              end if
                           end if
                               !!!! END SB 1 !!!!
                               !
                               !!!! SB 2 !!!!
                               ! (1)
                               if (a_chol==a .and. b_chol==c) then
                                  idx = idx_table_copy2(i,j,k,b)
                                  do jdet = loc_arr_copy2(1,idx), loc_arr_copy2(2,idx)
                                     d = t3d_excits_copy2(jdet,1); f = t3d_excits_copy2(jdet,3);
                                     ! compute < ijkabc | h2a(vvvv) | ijkdbf >
                                     hmatel = h2c_vvvv(d,f)
                                     resid(idet) = resid(idet) + hmatel*t3d_amps_copy2(jdet)
                                  end do
                               end if
                               ! (ab)
                               if (a_chol==b .and. b_chol==c) then
                                  idx = idx_table_copy2(i,j,k,a)
                                  if (idx/=0) then
                                  do jdet = loc_arr_copy2(1,idx), loc_arr_copy2(2,idx)
                                     d = t3d_excits_copy2(jdet,1); f = t3d_excits_copy2(jdet,3);
                                     ! compute < ijkabc | h2a(vvvv) | ijkdaf >
                                     hmatel = -h2c_vvvv(d,f)
                                     resid(idet) = resid(idet) + hmatel*t3d_amps_copy2(jdet)
                                  end do
                                  end if
                               end if
                               ! (bc)
                               if (a_chol==a .and. b_chol==b) then
                                  idx = idx_table_copy2(i,j,k,c)
                                  if (idx/=0) then
                                  do jdet = loc_arr_copy2(1,idx), loc_arr_copy2(2,idx)
                                     d = t3d_excits_copy2(jdet,1); f = t3d_excits_copy2(jdet,3);
                                     ! compute < ijkabc | h2a(vvvv) | ijkdcf >
                                     hmatel = -h2c_vvvv(d,f)
                                     resid(idet) = resid(idet) + hmatel*t3d_amps_copy2(jdet)
                                  end do
                                  end if
                               end if
                               !!!! END SB 2 !!!!
                               !
                               !!!! SB 3 !!!!
                               ! (1)
                               if (a_chol==a .and. b_chol==b) then
                                  idx = idx_table_copy3(i,j,k,c)
                                  do jdet = loc_arr_copy3(1,idx), loc_arr_copy3(2,idx)
                                     d = t3d_excits_copy3(jdet,1); e = t3d_excits_copy3(jdet,2);
                                     ! compute < ijkabc | h2a(vvvv) | ijkdec >
                                     hmatel = h2c_vvvv(d,e)
                                     resid(idet) = resid(idet) + hmatel*t3d_amps_copy3(jdet)
                                  end do
                               end if
                               ! (ac)
                               if (a_chol==b .and. b_chol==c) then ! this was reversed and I put a minus sign since a_chol < b_chol
                                  idx = idx_table_copy3(i,j,k,a)
                                  if (idx/=0) then
                                  do jdet = loc_arr_copy3(1,idx), loc_arr_copy3(2,idx)
                                     d = t3d_excits_copy3(jdet,1); e = t3d_excits_copy3(jdet,2);
                                     ! compute < ijkabc | h2a(vvvv) | ijkdea >
                                     hmatel = h2c_vvvv(d,e)
                                     resid(idet) = resid(idet) + hmatel*t3d_amps_copy3(jdet)
                                  end do
                                  end if
                               end if
                               ! (bc)
                               if (a_chol==a .and. b_chol==c) then
                                  idx = idx_table_copy3(i,j,k,b)
                                  if (idx/=0) then
                                  do jdet = loc_arr_copy3(1,idx), loc_arr_copy3(2,idx)
                                     d = t3d_excits_copy3(jdet,1); e = t3d_excits_copy3(jdet,2);
                                     ! compute < ijkabc | h2a(vvvv) | ijkdeb >
                                     hmatel = -h2c_vvvv(d,e)
                                     resid(idet) = resid(idet) + hmatel*t3d_amps_copy3(jdet)
                                  end do
                                  end if
                               end if
                               !!!! END SB 3 !!!!
                               !
                            end do ! end master copy loop
                            !
                  !     end do
                  end do
                  !$omp end parallel do
                  deallocate(t3d_excits_copy1,t3d_excits_copy2,t3d_excits_copy3)
                  deallocate(t3d_amps_copy1,t3d_amps_copy2,t3d_amps_copy3)
                  deallocate(idx_table_copy1,idx_table_copy2,idx_table_copy3)
                  deallocate(loc_arr_copy1,loc_arr_copy2,loc_arr_copy3)

                 !
                 ! Moment contributions
                 !
                 allocate(xbuf(nob,nob,nub,nub))
                 do a = 1,nub
                    do b = 1,nub
                       do i = 1,nob
                          do j = 1,nob
                             xbuf(j,i,b,a) = t2c(b,a,j,i)
                          end do
                       end do
                    end do
                 end do
                 do idet = 1, n3bbb
                    a = t3d_excits(idet,1); b = t3d_excits(idet,2); c = t3d_excits(idet,3);
                    i = t3d_excits(idet,4); j = t3d_excits(idet,5); k = t3d_excits(idet,6);
                    do m = 1, nob
                       ! -A(k/ij)A(a/bc) h2a(amij) * t2c(bcmk)
                       resid(idet) = resid(idet) - I2C_vooo(m,a,i,j) * xbuf(m,k,b,c)
                       resid(idet) = resid(idet) + I2C_vooo(m,b,i,j) * xbuf(m,k,a,c)
                       resid(idet) = resid(idet) + I2C_vooo(m,c,i,j) * xbuf(m,k,b,a)
                       resid(idet) = resid(idet) + I2C_vooo(m,a,k,j) * xbuf(m,i,b,c)
                       resid(idet) = resid(idet) - I2C_vooo(m,b,k,j) * xbuf(m,i,a,c)
                       resid(idet) = resid(idet) - I2C_vooo(m,c,k,j) * xbuf(m,i,b,a)
                       resid(idet) = resid(idet) + I2C_vooo(m,a,i,k) * xbuf(m,j,b,c)
                       resid(idet) = resid(idet) - I2C_vooo(m,b,i,k) * xbuf(m,j,a,c)
                       resid(idet) = resid(idet) - I2C_vooo(m,c,i,k) * xbuf(m,j,b,a)
                    end do
                 end do
                 deallocate(xbuf)

                 do idet = 1, n3bbb
                    a = t3d_excits(idet,1); b = t3d_excits(idet,2); c = t3d_excits(idet,3);
                    i = t3d_excits(idet,4); j = t3d_excits(idet,5); k = t3d_excits(idet,6);
                    do e = 1, nub
                       ! A(i/jk)(c/ab) h2a(abie) * t2c(ecjk)
                       resid(idet) = resid(idet) + I2C_vvov(e,a,b,i) * t2c(e,c,j,k)
                       resid(idet) = resid(idet) - I2C_vvov(e,c,b,i) * t2c(e,a,j,k)
                       resid(idet) = resid(idet) - I2C_vvov(e,a,c,i) * t2c(e,b,j,k)
                       resid(idet) = resid(idet) - I2C_vvov(e,a,b,j) * t2c(e,c,i,k)
                       resid(idet) = resid(idet) + I2C_vvov(e,c,b,j) * t2c(e,a,i,k)
                       resid(idet) = resid(idet) + I2C_vvov(e,a,c,j) * t2c(e,b,i,k)
                       resid(idet) = resid(idet) - I2C_vvov(e,a,b,k) * t2c(e,c,j,i)
                       resid(idet) = resid(idet) + I2C_vvov(e,c,b,k) * t2c(e,a,j,i)
                       resid(idet) = resid(idet) + I2C_vvov(e,a,c,k) * t2c(e,b,j,i)
                    end do
                 end do

                 ! Update t3 vector
                 do idet = 1,n3bbb
                    a = t3d_excits(idet,1); b = t3d_excits(idet,2); c = t3d_excits(idet,3);
                    i = t3d_excits(idet,4); j = t3d_excits(idet,5); k = t3d_excits(idet,6);
                    denom = fB_oo(i,i) + fB_oo(j,j) + fB_oo(k,k) - fB_vv(a,a) - fB_vv(b,b) - fB_vv(c,c)
                    resid(idet) = resid(idet)/(denom - shift)
                    t3d_amps(idet) = t3d_amps(idet) + resid(idet)
                 end do

              end subroutine update_t3d_p

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!! INTERMEDIATES FUNCTIONS !!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine calc_I2A_vooo(I2A_vooo,&
                               H2A_oovv,H2B_oovv,&
                               t3a_excits,t3a_amps,t3b_excits,t3b_amps,&
                               n3aaa,n3aab,noa,nua,nob,nub)

                  integer, intent(in) :: noa, nua, nob, nub, n3aaa, n3aab
                  integer, intent(in) :: t3a_excits(n3aaa,6), t3b_excits(n3aab,6)
                  real(kind=8), intent(in) :: t3a_amps(n3aaa), t3b_amps(n3aab)
                  real(kind=8), intent(in) :: H2A_oovv(noa,noa,nua,nua),H2B_oovv(noa,nob,nua,nub)
                  real(kind=8), intent(inout) :: I2A_vooo(noa,nua,noa,noa)

                  integer :: idet, a, b, c, i, j, k, m, n, e, f
                  real(kind=8) :: t_amp

                  do idet = 1, n3aaa
                      t_amp = t3a_amps(idet)
                      ! I2A(amij) <- A(ij) [A(n/ij)A(a/ef) h2a(mnef) * t3a(aefijn)]
                      a = t3a_excits(idet,1); e = t3a_excits(idet,2); f = t3a_excits(idet,3);
                      i = t3a_excits(idet,4); j = t3a_excits(idet,5); n = t3a_excits(idet,6);
                      I2A_vooo(:,a,i,j) = I2A_vooo(:,a,i,j) + H2A_oovv(:,n,e,f) * t_amp ! (1)
                      I2A_vooo(:,a,n,j) = I2A_vooo(:,a,n,j) - H2A_oovv(:,i,e,f) * t_amp ! (in)
                      I2A_vooo(:,a,i,n) = I2A_vooo(:,a,i,n) - H2A_oovv(:,j,e,f) * t_amp ! (jn)
                      I2A_vooo(:,e,i,j) = I2A_vooo(:,e,i,j) - H2A_oovv(:,n,a,f) * t_amp ! (ae)
                      I2A_vooo(:,e,n,j) = I2A_vooo(:,e,n,j) + H2A_oovv(:,i,a,f) * t_amp ! (in)(ae)
                      I2A_vooo(:,e,i,n) = I2A_vooo(:,e,i,n) + H2A_oovv(:,j,a,f) * t_amp ! (jn)(ae)
                      I2A_vooo(:,f,i,j) = I2A_vooo(:,f,i,j) - H2A_oovv(:,n,e,a) * t_amp ! (af)
                      I2A_vooo(:,f,n,j) = I2A_vooo(:,f,n,j) + H2A_oovv(:,i,e,a) * t_amp ! (in)(af)
                      I2A_vooo(:,f,i,n) = I2A_vooo(:,f,i,n) + H2A_oovv(:,j,e,a) * t_amp ! (jn)(af)
                  end do
                  do idet = 1,n3aab
                      t_amp = t3b_amps(idet)
                      ! I2A(amij) <- A(ij) [A(ae) h2b(mnef) * t3b(aefijn)]
                      a = t3b_excits(idet,1); e = t3b_excits(idet,2); f = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); j = t3b_excits(idet,5); n = t3b_excits(idet,6);
                      I2A_vooo(:,a,i,j) = I2A_vooo(:,a,i,j) + H2B_oovv(:,n,e,f) * t_amp ! (1)
                      I2A_vooo(:,e,i,j) = I2A_vooo(:,e,i,j) - H2B_oovv(:,n,a,f) * t_amp ! (ae)
                  end do
                  ! antisymmetrize
                  do i = 1,noa
                     do j = i+1,noa
                        do a = 1,nua
                           do m = 1,noa
                              I2A_vooo(m,a,i,j) = I2A_vooo(m,a,i,j) - I2A_vooo(m,a,j,i)
                              I2A_vooo(m,a,j,i) = -I2A_vooo(m,a,i,j)
                           end do
                        end do
                     end do
                  end do
      end subroutine calc_I2A_vooo

      subroutine calc_I2B_ovoo(I2B_ovoo,&
                               H2A_oovv,H2B_oovv,&
                               t3b_excits,t3b_amps,t3c_excits,t3c_amps,&
                               n3aab,n3abb,noa,nua,nob,nub)

                  integer, intent(in) :: noa, nua, nob, nub, n3aab, n3abb
                  integer, intent(in) :: t3b_excits(n3aab,6), t3c_excits(n3abb,6)
                  real(kind=8), intent(in) :: t3b_amps(n3aab), t3c_amps(n3abb)
                  real(kind=8), intent(in) :: H2A_oovv(noa,noa,nua,nua),H2B_oovv(noa,nob,nua,nub)
                  real(kind=8), intent(inout) :: I2B_ovoo(noa,nub,noa,nob)

                  integer :: idet, a, b, c, i, j, k, m, n, e, f
                  real(kind=8) :: t_amp

                  do idet = 1, n3aab
                      t_amp = t3b_amps(idet)
                      ! I2B(mbij) <- A(in) h2a(mnef) * t3b(efbinj)
                      e = t3b_excits(idet,1); f = t3b_excits(idet,2); b = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); n = t3b_excits(idet,5); j = t3b_excits(idet,6);
                      I2B_ovoo(:,b,i,j) = I2B_ovoo(:,b,i,j) + H2A_oovv(:,n,e,f) * t_amp ! (1)
                      I2B_ovoo(:,b,n,j) = I2B_ovoo(:,b,n,j) - H2A_oovv(:,i,e,f) * t_amp ! (in)
                  end do

                  do idet = 1, n3abb
                      t_amp = t3c_amps(idet)
                      ! I2B(mbij) <- A(bf)A(jn) h2B(mnef) * t3c(efbinj)
                      e = t3c_excits(idet,1); f = t3c_excits(idet,2); b = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); n = t3c_excits(idet,5); j = t3c_excits(idet,6);
                      I2B_ovoo(:,b,i,j) = I2B_ovoo(:,b,i,j) + H2B_oovv(:,n,e,f) * t_amp ! (1)
                      I2B_ovoo(:,f,i,j) = I2B_ovoo(:,f,i,j) - H2B_oovv(:,n,e,b) * t_amp ! (bf)
                      I2B_ovoo(:,b,i,n) = I2B_ovoo(:,b,i,n) - H2B_oovv(:,j,e,f) * t_amp ! (jn)
                      I2B_ovoo(:,f,i,n) = I2B_ovoo(:,f,i,n) + H2B_oovv(:,j,e,b) * t_amp ! (bf)(jn)
                  end do

      end subroutine calc_I2B_ovoo

      subroutine calc_I2B_vooo(I2B_vooo,&
                               H2B_oovv,H2C_oovv,&
                               t3b_excits,t3b_amps,t3c_excits,t3c_amps,&
                               n3aab,n3abb,noa,nua,nob,nub)

                  integer, intent(in) :: noa, nua, nob, nub, n3aab, n3abb
                  integer, intent(in) :: t3b_excits(n3aab,6), t3c_excits(n3abb,6)
                  real(kind=8), intent(in) :: t3b_amps(n3aab), t3c_amps(n3abb)
                  real(kind=8), intent(in) :: H2B_oovv(noa,nob,nua,nub),H2C_oovv(nob,nob,nub,nub)
                  !real(kind=8), intent(inout) :: I2B_vooo(nua,nob,noa,nob)
                  real(kind=8), intent(inout) :: I2B_vooo(nob,nua,noa,nob) ! reordered

                  integer :: idet, a, b, c, i, j, k, m, n, e, f
                  real(kind=8) :: t_amp
                  real(kind=8), allocatable :: intbuf(:,:,:,:)

                  allocate(intbuf(nob,noa,nub,nua))
                  do a = 1,nua
                     do b = 1,nub
                        do i = 1,noa
                           do j = 1,nob
                              intbuf(j,i,b,a) = h2b_oovv(i,j,a,b)
                           end do
                        end do
                     end do
                  end do
                  do idet = 1, n3aab
                      t_amp = t3b_amps(idet)
                      ! I2B(amij) <- A(af)A(in) h2b(nmfe) * t3b(afeinj)
                      a = t3b_excits(idet,1); f = t3b_excits(idet,2); e = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); n = t3b_excits(idet,5); j = t3b_excits(idet,6);
                      I2B_vooo(:,a,i,j) = I2B_vooo(:,a,i,j) + intbuf(:,n,e,f) * t_amp ! (1)
                      I2B_vooo(:,f,i,j) = I2B_vooo(:,f,i,j) - intbuf(:,n,e,a) * t_amp ! (af)
                      I2B_vooo(:,a,n,j) = I2B_vooo(:,a,n,j) - intbuf(:,i,e,f) * t_amp ! (in)
                      I2B_vooo(:,f,n,j) = I2B_vooo(:,f,n,j) + intbuf(:,i,e,a) * t_amp ! (af)(in)
                  end do
                  deallocate(intbuf)

                  allocate(intbuf(nob,nob,nub,nub))
                  do a = 1,nub
                     do b = 1,nub
                        do i = 1,nob
                           do j = 1,nob
                              intbuf(j,i,b,a) = h2c_oovv(i,j,a,b)
                           end do
                        end do
                     end do
                  end do
                  do idet = 1, n3abb
                      t_amp = t3c_amps(idet)
                      ! I2B(amij) <- A(jn) h2c(nmfe) * t3c(afeinj)
                      a = t3c_excits(idet,1); f = t3c_excits(idet,2); e = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); n = t3c_excits(idet,5); j = t3c_excits(idet,6);
                      I2B_vooo(:,a,i,j) = I2B_vooo(:,a,i,j) + intbuf(:,n,e,f) * t_amp ! (1)
                      I2B_vooo(:,a,i,n) = I2B_vooo(:,a,i,n) - intbuf(:,j,e,f) * t_amp ! (jn)
                  end do
                  deallocate(intbuf)

      end subroutine calc_I2B_vooo

      subroutine calc_I2C_vooo(I2C_vooo,&
                               H2B_oovv,H2C_oovv,&
                               t3c_excits,t3c_amps,t3d_excits,t3d_amps,&
                               n3abb,n3bbb,noa,nua,nob,nub)

                  integer, intent(in) :: noa, nua, nob, nub, n3abb, n3bbb
                  integer, intent(in) :: t3c_excits(n3abb,6), t3d_excits(n3bbb,6)
                  real(kind=8), intent(in) :: t3c_amps(n3abb), t3d_amps(n3bbb)
                  real(kind=8), intent(in) :: H2B_oovv(noa,nob,nua,nub),H2C_oovv(nob,nob,nub,nub)
                  real(kind=8), intent(inout) :: I2C_vooo(nob,nub,nob,nob)

                  integer :: idet, a, b, c, i, j, k, m, n, e, f
                  real(kind=8) :: t_amp
                  real(kind=8), allocatable :: intbuf(:,:,:,:)

                  do idet = 1, n3bbb
                      t_amp = t3d_amps(idet)
                      ! I2C(amij) <- A(ij) [A(n/ij)A(a/ef) h2c(mnef) * t3d(aefijn)]
                      a = t3d_excits(idet,1); e = t3d_excits(idet,2); f = t3d_excits(idet,3);
                      i = t3d_excits(idet,4); j = t3d_excits(idet,5); n = t3d_excits(idet,6);
                      I2C_vooo(:,a,i,j) = I2C_vooo(:,a,i,j) + H2C_oovv(:,n,e,f) * t_amp ! (1)
                      I2C_vooo(:,a,n,j) = I2C_vooo(:,a,n,j) - H2C_oovv(:,i,e,f) * t_amp ! (in)
                      I2C_vooo(:,a,i,n) = I2C_vooo(:,a,i,n) - H2C_oovv(:,j,e,f) * t_amp ! (jn)
                      I2C_vooo(:,e,i,j) = I2C_vooo(:,e,i,j) - H2C_oovv(:,n,a,f) * t_amp ! (ae)
                      I2C_vooo(:,e,n,j) = I2C_vooo(:,e,n,j) + H2C_oovv(:,i,a,f) * t_amp ! (in)(ae)
                      I2C_vooo(:,e,i,n) = I2C_vooo(:,e,i,n) + H2C_oovv(:,j,a,f) * t_amp ! (jn)(ae)
                      I2C_vooo(:,f,i,j) = I2C_vooo(:,f,i,j) - H2C_oovv(:,n,e,a) * t_amp ! (af)
                      I2C_vooo(:,f,n,j) = I2C_vooo(:,f,n,j) + H2C_oovv(:,i,e,a) * t_amp ! (in)(af)
                      I2C_vooo(:,f,i,n) = I2C_vooo(:,f,i,n) + H2C_oovv(:,j,e,a) * t_amp ! (jn)(af)
                  end do
                  allocate(intbuf(nob,noa,nub,nua))
                  do a = 1,nua
                     do b = 1,nub
                        do i = 1,noa
                           do j = 1,nob
                              intbuf(j,i,b,a) = h2b_oovv(i,j,a,b)
                           end do
                        end do
                     end do
                  end do
                  do idet = 1, n3abb
                      t_amp = t3c_amps(idet)
                      ! I2C(amij) <- A(ij) [A(ae) h2b(nmfe) * t3c(faenij)]
                      f = t3c_excits(idet,1); a = t3c_excits(idet,2); e = t3c_excits(idet,3);
                      n = t3c_excits(idet,4); i = t3c_excits(idet,5); j = t3c_excits(idet,6);
                      I2C_vooo(:,a,i,j) = I2C_vooo(:,a,i,j) + intbuf(:,n,e,f) * t_amp ! (1)
                      I2C_vooo(:,e,i,j) = I2C_vooo(:,e,i,j) - intbuf(:,n,a,f) * t_amp ! (ae)
                  end do
                  deallocate(intbuf)
                  ! antisymmetrize
                  do i = 1,nob
                     do j = i+1,nob
                        do a = 1,nub
                           do m = 1,nob
                              I2C_vooo(m,a,i,j) = I2C_vooo(m,a,i,j) - I2C_vooo(m,a,j,i)
                              I2C_vooo(m,a,j,i) = -I2C_vooo(m,a,i,j)
                           end do
                        end do
                     end do
                  end do
      end subroutine calc_I2C_vooo

      subroutine calc_I2A_vvov(I2A_vvov,&
                               H2A_oovv,H2B_oovv,&
                               t3a_excits,t3a_amps,t3b_excits,t3b_amps,&
                               n3aaa,n3aab,noa,nua,nob,nub)

                  integer, intent(in) :: noa, nua, nob, nub, n3aaa, n3aab
                  integer, intent(in) :: t3a_excits(n3aaa,6), t3b_excits(n3aab,6)
                  real(kind=8), intent(in) :: t3a_amps(n3aaa), t3b_amps(n3aab)
                  real(kind=8), intent(in) :: H2A_oovv(noa,noa,nua,nua),H2B_oovv(noa,nob,nua,nub)

                  real(kind=8), intent(inout) :: I2A_vvov(nua,nua,nua,noa) ! reordered

                  integer :: idet, a, b, c, i, j, k, m, n, e, f
                  real(kind=8) :: t_amp
                  real(kind=8), allocatable :: intbuf(:,:,:,:)

                  allocate(intbuf(nua,nua,noa,noa))
                  do i = 1,noa
                     do j = 1,noa
                        do a = 1,nua
                           do b = 1,nua
                              intbuf(b,a,j,i) = h2a_oovv(j,i,b,a)
                           end do
                        end do
                     end do
                  end do
                  do idet = 1, n3aaa
                      t_amp = t3a_amps(idet)
                      ! I2A(abie) <- A(ab) [A(i/mn)A(f/ab) -h2a(mnef) * t3a(abfimn)]
                      a = t3a_excits(idet,1); b = t3a_excits(idet,2); f = t3a_excits(idet,3);
                      i = t3a_excits(idet,4); m = t3a_excits(idet,5); n = t3a_excits(idet,6);
                      I2A_vvov(:,a,b,i) = I2A_vvov(:,a,b,i) - intbuf(:,f,m,n) * t_amp ! (1)
                      I2A_vvov(:,a,b,m) = I2A_vvov(:,a,b,m) + intbuf(:,f,i,n) * t_amp ! (im)
                      I2A_vvov(:,a,b,n) = I2A_vvov(:,a,b,n) + intbuf(:,f,m,i) * t_amp ! (in)
                      I2A_vvov(:,f,b,i) = I2A_vvov(:,f,b,i) + intbuf(:,a,m,n) * t_amp ! (af)
                      I2A_vvov(:,f,b,m) = I2A_vvov(:,f,b,m) - intbuf(:,a,i,n) * t_amp ! (im)(af)
                      I2A_vvov(:,f,b,n) = I2A_vvov(:,f,b,n) - intbuf(:,a,m,i) * t_amp ! (in)(af)
                      I2A_vvov(:,a,f,i) = I2A_vvov(:,a,f,i) + intbuf(:,b,m,n) * t_amp ! (bf)
                      I2A_vvov(:,a,f,m) = I2A_vvov(:,a,f,m) - intbuf(:,b,i,n) * t_amp ! (im)(bf)
                      I2A_vvov(:,a,f,n) = I2A_vvov(:,a,f,n) - intbuf(:,b,m,i) * t_amp ! (in)(bf)
                  end do
                  deallocate(intbuf)
                  allocate(intbuf(nua,nub,noa,nob))
                  do j = 1,nob
                     do i = 1,noa
                        do b = 1,nub
                           do a = 1,nua
                              intbuf(a,b,i,j) = h2b_oovv(i,j,a,b)
                           end do
                        end do
                     end do
                  end do
                  do idet = 1,n3aab
                      t_amp = t3b_amps(idet)
                      ! I2A(abie) <- A(ab) [A(im) -h2b(mnef) * t3b(abfimn)]
                      a = t3b_excits(idet,1); b = t3b_excits(idet,2); f = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); m = t3b_excits(idet,5); n = t3b_excits(idet,6);
                      I2A_vvov(:,a,b,i) = I2A_vvov(:,a,b,i) - intbuf(:,f,m,n) * t_amp ! (1)
                      I2A_vvov(:,a,b,m) = I2A_vvov(:,a,b,m) + intbuf(:,f,i,n) * t_amp ! (im)
                  end do
                  deallocate(intbuf)
                  ! antisymmetrize
                  do i = 1,noa
                     do a = 1,nua
                        do b = a+1,nua
                           do e = 1,nua
                              I2A_vvov(e,a,b,i) = I2A_vvov(e,a,b,i) - I2A_vvov(e,b,a,i)
                              I2A_vvov(e,b,a,i) = -I2A_vvov(e,a,b,i)
                           end do
                        end do
                     end do
                  end do
      end subroutine calc_I2A_vvov

      subroutine calc_I2B_vvov(I2B_vvov,&
                               H2B_oovv,H2C_oovv,&
                               t3b_excits,t3b_amps,t3c_excits,t3c_amps,&
                               n3aab,n3abb,noa,nua,nob,nub)

                  integer, intent(in) :: noa, nua, nob, nub, n3aab, n3abb
                  integer, intent(in) :: t3b_excits(n3aab,6), t3c_excits(n3abb,6)
                  real(kind=8), intent(in) :: t3b_amps(n3aab), t3c_amps(n3abb)
                  real(kind=8), intent(in) :: H2B_oovv(noa,nob,nua,nub),H2C_oovv(nob,nob,nub,nub)
                  !real(kind=8), intent(inout) :: I2B_vvov(nua,nub,noa,nub)
                  real(kind=8), intent(inout) :: I2B_vvov(nub,nua,nub,noa) ! reordered

                  integer :: idet, a, b, c, i, j, k, m, n, e, f
                  real(kind=8) :: t_amp
                  real(kind=8), allocatable :: intbuf(:,:,:,:)

                  allocate(intbuf(nub,nua,nob,noa))
                  do i = 1,noa
                     do j = 1,nob
                        do a = 1,nua
                           do b = 1,nub
                              intbuf(b,a,j,i) = H2B_oovv(i,j,a,b)
                           end do
                        end do
                     end do
                  end do
                  do idet = 1, n3aab
                      t_amp = t3b_amps(idet)
                      ! I2B(abie) <- A(af)A(in) -h2b(nmfe) * t3b(afbinm)
                      a = t3b_excits(idet,1); f = t3b_excits(idet,2); b = t3b_excits(idet,3);
                      i = t3b_excits(idet,4); n = t3b_excits(idet,5); m = t3b_excits(idet,6);
                      !I2B_vvov(:,a,b,i) = I2B_vvov(:,a,b,i) - H2B_oovv(n,m,f,:) * t_amp ! (1)
                      !I2B_vvov(:,f,b,i) = I2B_vvov(:,f,b,i) + H2B_oovv(n,m,a,:) * t_amp ! (af)
                      !I2B_vvov(:,a,b,n) = I2B_vvov(:,a,b,n) + H2B_oovv(i,m,f,:) * t_amp ! (in)
                      !I2B_vvov(:,f,b,n) = I2B_vvov(:,f,b,n) - H2B_oovv(i,m,a,:) * t_amp ! (af)(in)
                      I2B_vvov(:,a,b,i) = I2B_vvov(:,a,b,i) - intbuf(:,f,m,n) * t_amp ! (1)
                      I2B_vvov(:,f,b,i) = I2B_vvov(:,f,b,i) + intbuf(:,a,m,n) * t_amp ! (af)
                      I2B_vvov(:,a,b,n) = I2B_vvov(:,a,b,n) + intbuf(:,f,m,i) * t_amp ! (in)
                      I2B_vvov(:,f,b,n) = I2B_vvov(:,f,b,n) - intbuf(:,a,m,i) * t_amp ! (af)(in)
                  end do
                  deallocate(intbuf)

                  allocate(intbuf(nub,nub,nob,nob))
                  do i = 1,nob
                     do j = 1,nob
                        do a = 1,nub
                           do b = 1,nub
                              intbuf(b,a,j,i) = H2C_oovv(i,j,a,b)
                           end do
                        end do
                     end do
                  end do
                  do idet = 1, n3abb
                      t_amp = t3c_amps(idet)
                      ! I2B(abie) <- A(bf) -h2c(nmfe) * t3c(afbinm)
                      a = t3c_excits(idet,1); f = t3c_excits(idet,2); b = t3c_excits(idet,3);
                      i = t3c_excits(idet,4); n = t3c_excits(idet,5); m = t3c_excits(idet,6);
                      !I2B_vvov(:,a,b,i) = I2B_vvov(:,a,b,i) - H2C_oovv(n,m,f,:) * t_amp ! (1)
                      !I2B_vvov(:,a,f,i) = I2B_vvov(:,a,f,i) + H2C_oovv(n,m,b,:) * t_amp ! (bf)
                      I2B_vvov(:,a,b,i) = I2B_vvov(:,a,b,i) - intbuf(:,f,m,n) * t_amp ! (1)
                      I2B_vvov(:,a,f,i) = I2B_vvov(:,a,f,i) + intbuf(:,b,m,n) * t_amp ! (bf)
                  end do
                  deallocate(intbuf)

      end subroutine calc_I2B_vvov

      subroutine calc_I2B_vvvo(I2B_vvvo,&
                               H2A_oovv,H2B_oovv,&
                               t3b_excits,t3b_amps,t3c_excits,t3c_amps,&
                               n3aab,n3abb,noa,nua,nob,nub)

                  integer, intent(in) :: noa, nua, nob, nub, n3aab, n3abb
                  integer, intent(in) :: t3b_excits(n3aab,6), t3c_excits(n3abb,6)
                  real(kind=8), intent(in) :: t3b_amps(n3aab), t3c_amps(n3abb)
                  real(kind=8), intent(in) :: H2A_oovv(noa,noa,nua,nua),H2B_oovv(noa,nob,nua,nub)
                  !real(kind=8), intent(inout) :: I2B_vvvo(nua,nub,nua,nob)
                  real(kind=8), intent(inout) :: I2B_vvvo(nua,nua,nub,nob)

                  integer :: idet, a, b, c, i, j, k, m, n, e, f
                  real(kind=8) :: t_amp
                  real(kind=8), allocatable :: intbuf(:,:,:,:)

                  allocate(intbuf(nua,nua,noa,noa))
                  do j = 1,noa
                     do i = 1,noa
                        do b = 1,nua
                           do a = 1,nua
                              intbuf(a,b,i,j) = H2A_oovv(i,j,a,b)
                           end do
                        end do
                     end do
                  end do
                  do idet = 1, n3aab
                      t_amp = t3b_amps(idet)
                      ! I2B(abej) <- A(af) -h2a(mnef) * t3b(afbmnj)
                      a = t3b_excits(idet,1); f = t3b_excits(idet,2); b = t3b_excits(idet,3);
                      m = t3b_excits(idet,4); n = t3b_excits(idet,5); j = t3b_excits(idet,6);
                      I2B_vvvo(:,a,b,j) = I2B_vvvo(:,a,b,j) - intbuf(:,f,m,n) * t_amp ! (1)
                      I2B_vvvo(:,f,b,j) = I2B_vvvo(:,f,b,j) + intbuf(:,a,m,n) * t_amp ! (af)
                  end do
                  deallocate(intbuf)

                  allocate(intbuf(nua,nub,noa,nob))
                  do j = 1,nob
                     do i = 1,noa
                        do b = 1,nub
                           do a = 1,nua
                              intbuf(a,b,i,j) = H2B_oovv(i,j,a,b)
                           end do
                        end do
                     end do
                  end do
                  do idet = 1, n3abb
                      t_amp = t3c_amps(idet)
                      ! I2B(abej) <- A(bf)A(jn) -h2b(mnef) * t3c(afbmnj)
                      a = t3c_excits(idet,1); f = t3c_excits(idet,2); b = t3c_excits(idet,3);
                      m = t3c_excits(idet,4); n = t3c_excits(idet,5); j = t3c_excits(idet,6);
                      I2B_vvvo(:,a,b,j) = I2B_vvvo(:,a,b,j) - intbuf(:,f,m,n) * t_amp ! (1)
                      I2B_vvvo(:,a,f,j) = I2B_vvvo(:,a,f,j) + intbuf(:,b,m,n) * t_amp ! (bf)
                      I2B_vvvo(:,a,b,n) = I2B_vvvo(:,a,b,n) + intbuf(:,f,m,j) * t_amp ! (jn)
                      I2B_vvvo(:,a,f,n) = I2B_vvvo(:,a,f,n) - intbuf(:,b,m,j) * t_amp ! (bf)(jn)
                  end do
                  deallocate(intbuf)

      end subroutine calc_I2B_vvvo

      subroutine calc_I2C_vvov(I2C_vvov,&
                               H2B_oovv,H2C_oovv,&
                               t3c_excits,t3c_amps,t3d_excits,t3d_amps,&
                               n3abb,n3bbb,noa,nua,nob,nub)

                  integer, intent(in) :: noa, nua, nob, nub, n3abb, n3bbb
                  integer, intent(in) :: t3c_excits(n3abb,6), t3d_excits(n3bbb,6)
                  real(kind=8), intent(in) :: t3c_amps(n3abb), t3d_amps(n3bbb)
                  real(kind=8), intent(in) :: H2B_oovv(noa,nob,nua,nub),H2C_oovv(nob,nob,nub,nub)

                  real(kind=8), intent(inout) :: I2C_vvov(nub,nub,nub,nob) ! reordered

                  integer :: idet, a, b, c, i, j, k, m, n, e, f
                  real(kind=8) :: t_amp
                  real(kind=8), allocatable :: intbuf(:,:,:,:)

                  allocate(intbuf(nub,nub,nob,nob))
                  do i = 1,nob
                     do j = 1,nob
                        do a = 1,nub
                           do b = 1,nub
                              intbuf(b,a,j,i) = h2c_oovv(j,i,b,a)
                           end do
                        end do
                     end do
                  end do
                  do idet = 1, n3bbb
                      t_amp = t3d_amps(idet)
                      ! I2C(abie) <- A(ab) [A(i/mn)A(f/ab) -h2c(mnef) * t3d(abfimn)]
                      a = t3d_excits(idet,1); b = t3d_excits(idet,2); f = t3d_excits(idet,3);
                      i = t3d_excits(idet,4); m = t3d_excits(idet,5); n = t3d_excits(idet,6);
                      I2C_vvov(:,a,b,i) = I2C_vvov(:,a,b,i) - intbuf(:,f,m,n) * t_amp ! (1)
                      I2C_vvov(:,a,b,m) = I2C_vvov(:,a,b,m) + intbuf(:,f,i,n) * t_amp ! (im)
                      I2C_vvov(:,a,b,n) = I2C_vvov(:,a,b,n) + intbuf(:,f,m,i) * t_amp ! (in)
                      I2C_vvov(:,f,b,i) = I2C_vvov(:,f,b,i) + intbuf(:,a,m,n) * t_amp ! (af)
                      I2C_vvov(:,f,b,m) = I2C_vvov(:,f,b,m) - intbuf(:,a,i,n) * t_amp ! (im)(af)
                      I2C_vvov(:,f,b,n) = I2C_vvov(:,f,b,n) - intbuf(:,a,m,i) * t_amp ! (in)(af)
                      I2C_vvov(:,a,f,i) = I2C_vvov(:,a,f,i) + intbuf(:,b,m,n) * t_amp ! (bf)
                      I2C_vvov(:,a,f,m) = I2C_vvov(:,a,f,m) - intbuf(:,b,i,n) * t_amp ! (im)(bf)
                      I2C_vvov(:,a,f,n) = I2C_vvov(:,a,f,n) - intbuf(:,b,m,i) * t_amp ! (in)(bf)
                  end do
                  deallocate(intbuf)
                  allocate(intbuf(nub,nua,nob,noa))
                  do i = 1,noa
                     do j = 1,nob
                        do a = 1,nua
                           do b = 1,nub
                              intbuf(b,a,j,i) = h2b_oovv(i,j,a,b)
                           end do
                        end do
                     end do
                  end do
                  do idet = 1, n3abb
                      t_amp = t3c_amps(idet)
                      ! I2C(abie) <- A(ab) [A(im) -h2b(nmfe) * t3c(fabnim)]
                      f = t3c_excits(idet,1); a = t3c_excits(idet,2); b = t3c_excits(idet,3);
                      n = t3c_excits(idet,4); i = t3c_excits(idet,5); m = t3c_excits(idet,6);
                      I2C_vvov(:,a,b,i) = I2C_vvov(:,a,b,i) - intbuf(:,f,m,n) * t_amp ! (1)
                      I2C_vvov(:,a,b,m) = I2C_vvov(:,a,b,m) + intbuf(:,f,i,n) * t_amp ! (im)
                  end do
                  deallocate(intbuf)
                  ! antisymmetrize
                  do i = 1,nob
                     do a = 1,nub
                        do b = a+1,nub
                           do e = 1,nub
                              I2C_vvov(e,a,b,i) = I2C_vvov(e,a,b,i) - I2C_vvov(e,b,a,i)
                              I2C_vvov(e,b,a,i) = -I2C_vvov(e,a,b,i)
                           end do
                        end do
                     end do
                  end do

      end subroutine calc_I2C_vvov

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

              integer, intent(inout) :: loc_arr(2,nloc)
              integer, intent(inout) :: excits(n3p,6)
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
                 p = excits(idet,idims(1)); q = excits(idet,idims(2)); r = excits(idet,idims(3)); s = excits(idet,idims(4))
                 temp(idet) = idx_table(p,q,r,s)
              end do
              ! get the sorting array
              call argsort(temp, idx)
              ! apply sorting array to t3 excitations, amplitudes, and, optionally, residual arrays
              excits = excits(idx,:)
              amps = amps(idx)
              if (present(resid)) resid = resid(idx)
              deallocate(temp,idx)
              ! obtain the start- and end-point indices for each lexical index in the sorted t3 excitation and amplitude arrays
              loc_arr(1,:) = 1; loc_arr(2,:) = 0; ! set default start > end so that empty sets do not trigger loops
              !!! WARNING: THERE IS A MEMORY LEAK HERE! pqrs2 is used below but is not set if n3p <= 1
              !if (n3p <= 1) print*, "(ccsdt_p_loops) >> WARNING: potential memory leakage in sort4 function. pqrs2 set to -1"
              if (n3p == 1) then
                 if (excits(1,1)==1 .and. excits(1,2)==1 .and. excits(1,3)==1 .and. excits(1,4)==1 .and. excits(1,5)==1 .and. excits(1,6)==1) return
                 p2 = excits(n3p,idims(1)); q2 = excits(n3p,idims(2)); r2 = excits(n3p,idims(3)); s2 = excits(n3p,idims(4))
                 pqrs2 = idx_table(p2,q2,r2,s2)
              else
                 pqrs2 = -1
              end if
              do idet = 1, n3p-1
                 ! get consecutive lexcial indices
                 p1 = excits(idet,idims(1));   q1 = excits(idet,idims(2));   r1 = excits(idet,idims(3));   s1 = excits(idet,idims(4))
                 p2 = excits(idet+1,idims(1)); q2 = excits(idet+1,idims(2)); r2 = excits(idet+1,idims(3)); s2 = excits(idet+1,idims(4))
                 pqrs1 = idx_table(p1,q1,r1,s1)
                 pqrs2 = idx_table(p2,q2,r2,s2)
                 ! if change occurs between consecutive indices, record these locations in loc_arr as new start/end points
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

      subroutine sum4(x, y, iorder)

          integer, intent(in) :: iorder(4)
          real(kind=8), intent(in) :: y(:,:,:,:)

          real(kind=8), intent(inout) :: x(:,:,:,:)

          integer :: i, j, k, l
          integer :: vec(4)

          do i = 1, size(x,1)
             do j = 1, size(x,2)
                do k = 1, size(x,3)
                   do l = 1, size(x,4)
                      vec = (/i,j,k,l/)
                      x(i,j,k,l) = x(i,j,k,l) + y(vec(iorder(1)),vec(iorder(2)),vec(iorder(3)),vec(iorder(4)))
                   end do
                end do
             end do
          end do

      end subroutine sum4

end module ccsdt_p_chol_loops
