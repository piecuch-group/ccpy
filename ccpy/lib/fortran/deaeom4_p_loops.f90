module deaeom4_p_loops

      use omp_lib

      implicit none

      contains

              subroutine build_hr_4b(resid,&
                                     r3b,&
                                     r4b_amps, r4b_excits,&
                                     r4c_amps, r4c_excits,&
                                     t2a, t2b,&
                                     h1a_oo, h1a_vv, h1b_vv,&
                                     h2a_vvvv, h2a_oooo, h2a_voov, h2a_vooo, h2a_vvov,&
                                     h2b_vvvv, h2b_voov, h2b_ovov,&
                                     x3b_vvoo, x3b_vvvv, x3b_vovo, x2b_oo,&
                                     n4abaa, n4abab,&
                                     noa, nua, nob, nub)
                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n4abaa, n4abab
                  !
                  real(kind=8), intent(in) :: r3b(nua,nub,nua,noa)
                  real(kind=8), intent(in) :: t2a(nua,nua,noa,noa)
                  real(kind=8), intent(in) :: t2b(nua,nub,noa,nob)
                  real(kind=8), intent(in) :: r4c_amps(n4abab)
                  integer, intent(in) :: r4c_excits(n4abab,6)
                  ! Input H and X arrays
                  real(kind=8), intent(in) :: h1a_oo(noa,noa)
                  real(kind=8), intent(in) :: h1b_vv(nub,nub)
                  real(kind=8), intent(in) :: h1a_vv(nua,nua)
                  real(kind=8), intent(in) :: h2a_vvvv(nua,nua,nua,nua)
                  real(kind=8), intent(in) :: h2a_oooo(noa,noa,noa,noa)
                  real(kind=8), intent(in) :: h2a_voov(nua,noa,noa,nua)
                  real(kind=8), intent(in) :: h2a_vooo(nua,noa,noa,noa)
                  real(kind=8), intent(in) :: h2a_vvov(nua,nua,noa,nua)
                  real(kind=8), intent(in) :: h2b_vvvv(nua,nub,nua,nub)
                  real(kind=8), intent(in) :: h2b_voov(nua,nob,noa,nub)
                  real(kind=8), intent(in) :: h2b_ovov(noa,nub,noa,nub)
                  real(kind=8), intent(in) :: x3b_vvoo(nua,nub,noa,noa)
                  real(kind=8), intent(in) :: x3b_vvvv(nua,nub,nua,nua)
                  real(kind=8), intent(in) :: x3b_vovo(nua,nob,nua,nob)
                  real(kind=8), intent(in) :: x2b_oo(noa,nob)
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
                  real(kind=8) :: ff
                  
                  ! Zero the container that holds H*R
                  resid = 0.0d0
                  
                  !!!! diagram 1: A(d/ac) h1a(de) r4b(ab~cekl)
                  !!!! diagram 2: 1/2 A(a/cd) h2a(cdef) r4b(ab~efkl)
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
                        hmatel = -h2a_vvvv(c,a,e,f)
                        ! compute < klab~cd | h1a(vv) | kldb~ef > = A(ac)A(ef) h1a(ce) delta(a,f)
                        hmatel1 = 0.0d0
                        if (a==f) hmatel1 = hmatel1 - h1a_vv(c,e) ! (1)
                        if (c==f) hmatel1 = hmatel1 + h1a_vv(a,e) ! (ac)
                        if (a==e) hmatel1 = hmatel1 + h1a_vv(c,f) ! (ef)
                        if (c==e) hmatel1 = hmatel1 - h1a_vv(a,f) ! (ac)(ef)
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
                  do idet = 1, n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(k,l,d,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); f = r4b_excits(jdet,3);
                        ! compute < klab~cd | h2a(vvvv) | kleb~fd >
                        hmatel = h2a_vvvv(a,c,e,f)
                        ! compute < klab~cd | h1a(vv) | kleb~fd > = A(ac)A(ef) h1a(ae) delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 + h1a_vv(a,e) ! (1)
                        if (a==f) hmatel1 = hmatel1 - h1a_vv(c,e) ! (ac)
                        if (c==e) hmatel1 = hmatel1 - h1a_vv(a,f) ! (ef)
                        if (a==e) hmatel1 = hmatel1 + h1a_vv(c,f) ! (ac)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     ! (ad)
                     idx = idx_table(k,l,a,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); f = r4b_excits(jdet,3);
                        ! compute < klab~cd | h2a(vvvv) | kleb~fa >
                        hmatel = -h2a_vvvv(d,c,e,f)
                        ! compute < klab~cd | h1a(vv) | kleb~fa > = -A(cd)A(ef) h1a(de) delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 - h1a_vv(d,e) ! (1)
                        if (d==f) hmatel1 = hmatel1 + h1a_vv(c,e) ! (ac)
                        if (c==e) hmatel1 = hmatel1 + h1a_vv(d,f) ! (ef)
                        if (d==e) hmatel1 = hmatel1 - h1a_vv(c,f) ! (ac)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(k,l,c,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); f = r4b_excits(jdet,3);
                        ! compute < klab~cd | h2a(vvvv) | kleb~fc >
                        hmatel = -h2a_vvvv(a,d,e,f)
                        ! compute < klab~cd | h1a(vv) | kleb~fc > = -A(ad)A(ef) h1a(ae) delta(d,f)
                        hmatel1 = 0.0d0
                        if (d==f) hmatel1 = hmatel1 - h1a_vv(a,e) ! (1)
                        if (a==f) hmatel1 = hmatel1 + h1a_vv(d,e) ! (ac)
                        if (d==e) hmatel1 = hmatel1 + h1a_vv(a,f) ! (ef)
                        if (a==e) hmatel1 = hmatel1 - h1a_vv(d,f) ! (ac)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 3: h1b(b~e~) * r4b(ae~cdkl)
                  !!!! diagram 4: A(a/cd) h2b(ab~ef~) * r4b(ef~cdkl)
                  ff = 1.0d0 / 3.0d0
                  ! allocate new sorting arrays
                  nloc = noa*(noa-1)/2*(nua-1)*(nua-2)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(noa,noa,nua,nua))
                  !!! SB: (5,6,3,4) !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/2,nua-1/), (/-1,nua/), noa, noa, nua, nua)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/5,6,3,4/), noa, noa, nua, nua, nloc, n4abaa, resid)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(k,l,c,d)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); f = r4b_excits(jdet,2);
                        ! compute < klab~cd | h2b(vvvv) | klef~cd >
                        hmatel = h2b_vvvv(a,b,e,f)
                        ! compute < klab~cd | h1b(vv) | klef~cd > = delta(a,e) * h1b(b~f~)
                        hmatel1 = 0.0d0
                        if (a==e) hmatel1 = hmatel1 + h1b_vv(b,f) 
                        hmatel = hmatel + ff * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(k,l,a,d)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); f = r4b_excits(jdet,2);
                        ! compute < klab~cd | h2b(vvvv) | klef~ad >
                        hmatel = -h2b_vvvv(c,b,e,f)
                        ! compute < klab~cd | h1b(vv) | klef~cd > = delta(c,e) * h1b(b~f~)
                        hmatel1 = 0.0d0
                        if (c==e) hmatel1 = hmatel1 - h1b_vv(b,f) 
                        hmatel = hmatel + ff * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (ad), -
                     idx = idx_table(k,l,a,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); f = r4b_excits(jdet,2);
                        ! compute < klab~cd | h2b(vvvv) | klef~ac >
                        hmatel = h2b_vvvv(d,b,e,f)
                        ! compute < klab~cd | h1b(vv) | klef~cd > = delta(d,e) * h1b(b~f~)
                        hmatel1 = 0.0d0
                        if (d==e) hmatel1 = hmatel1 + h1b_vv(b,f) 
                        hmatel = hmatel + ff * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (5,6,1,4) !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nua-2/), (/-2,nua/), noa, noa, nua, nua)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/5,6,1,4/), noa, noa, nua, nua, nloc, n4abaa, resid)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(k,l,a,d)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); f = r4b_excits(jdet,2);
                        ! compute < klab~cd | h2b(vvvv) | klaf~ed >
                        hmatel = h2b_vvvv(c,b,e,f)
                        ! compute < klab~cd | h1b(vv) | klaf~ed > = delta(c,e) * h1b(b~f~)
                        hmatel1 = 0.0d0
                        if (c==e) hmatel1 = hmatel1 + h1b_vv(b,f) 
                        hmatel = hmatel + ff * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(k,l,c,d)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); f = r4b_excits(jdet,2);
                        ! compute < klab~cd | h2b(vvvv) | klcf~ed >
                        hmatel = -h2b_vvvv(a,b,e,f)
                        ! compute < klab~cd | h1b(vv) | klaf~ed > = delta(a,e) * h1b(b~f~)
                        hmatel1 = 0.0d0
                        if (a==e) hmatel1 = hmatel1 - h1b_vv(b,f) 
                        hmatel = hmatel + ff * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(k,l,a,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); f = r4b_excits(jdet,2);
                        ! compute < klab~cd | h2b(vvvv) | klaf~ec >
                        hmatel = -h2b_vvvv(d,b,e,f)
                        ! compute < klab~cd | h1b(vv) | klaf~ed > = delta(d,e) * h1b(b~f~)
                        hmatel1 = 0.0d0
                        if (d==e) hmatel1 = hmatel1 - h1b_vv(b,f) 
                        hmatel = hmatel + ff * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (5,6,1,3) !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nua-2/), (/-1,nua-1/), noa, noa, nua, nua)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/5,6,1,3/), noa, noa, nua, nua, nloc, n4abaa, resid)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(k,l,a,c)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,4); f = r4b_excits(jdet,2);
                        ! compute < klab~cd | h2b(vvvv) | klaf~ce >
                        hmatel = h2b_vvvv(d,b,e,f)
                        ! compute < klab~cd | h1b(vv) | klaf~ce > = delta(d,e) * h1b(b~f~)
                        hmatel1 = 0.0d0
                        if (d==e) hmatel1 = hmatel1 + h1b_vv(b,f) 
                        hmatel = hmatel + ff * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     ! (ad), -
                     idx = idx_table(k,l,c,d)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,4); f = r4b_excits(jdet,2);
                        ! compute < klab~cd | h2b(vvvv) | klaf~ce >
                        hmatel = h2b_vvvv(a,b,e,f)
                        ! compute < klab~cd | h1b(vv) | klaf~ce > = delta(a,e) * h1b(b~f~)
                        hmatel1 = 0.0d0
                        if (a==e) hmatel1 = hmatel1 + h1b_vv(b,f) 
                        hmatel = hmatel + ff * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(k,l,a,d)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,4); f = r4b_excits(jdet,2);
                        ! compute < klab~cd | h2b(vvvv) | klaf~ce >
                        hmatel = -h2b_vvvv(c,b,e,f)
                        ! compute < klab~cd | h1b(vv) | klaf~ce > = delta(c,e) * h1b(b~f~)
                        hmatel1 = 0.0d0
                        if (c==e) hmatel1 = hmatel1 - h1b_vv(b,f) 
                        hmatel = hmatel + ff * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!! diagram 5: h2a(mnkl) * r4b(ab~cdmn)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)*(nua-2)/6 * nub 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,nua,nub))
                  !!! SB: (1,3,4,2) !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/1,nub/), nua, nua, nua, nub)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/1,3,4,2/), nua, nua, nua, nub, nloc, n4abaa, resid)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,d,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4b_excits(jdet,5); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(oooo) | mnab~cd >
                        hmatel = h2a_oooo(m,n,k,l)
                        ! compute < klab~cd | h1a(oo) | mnab~cd > = A(kl)A(mn) -delta(m,k) * h1a_oo(n,l)
                        hmatel1 = 0.0d0
                        if (m==k) hmatel1 = hmatel1  - h1a_oo(n,l) ! (1) 
                        if (m==l) hmatel1 = hmatel1  + h1a_oo(n,k) ! (kl) 
                        if (n==k) hmatel1 = hmatel1  + h1a_oo(m,l) ! (mn) 
                        if (n==l) hmatel1 = hmatel1  - h1a_oo(m,k) ! (kl)(mn) 
                        hmatel = hmatel + hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!! diagram 6: A(d/ac)A(kl) h2a(dmle) * r4b(abcekm)
                  ! allocate new sorting arrays
                  nloc = (nua-1)*(nua-2)/2 * (noa-1) * nub 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,noa,nub))
                  !!! SB: (1,3,5,2) !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/1,noa-1/), (/1,nub/), nua, nua, noa, nub)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/1,3,5,2/), nua, nua, noa, nub, nloc, n4abaa, resid)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,k,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,4); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | knab~cf >
                        hmatel = h2a_voov(d,n,l,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     ! (ad), -
                     idx = idx_table(c,d,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,4); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | kndb~cf >
                        hmatel = h2a_voov(a,n,l,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(a,d,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,4); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | knab~df >
                        hmatel = -h2a_voov(c,n,l,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(a,c,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,4); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | lnab~cf >
                        hmatel = -h2a_voov(d,n,k,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (ad)(kl), -
                     idx = idx_table(c,d,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,4); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | lndb~cf >
                        hmatel = -h2a_voov(a,n,k,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (cd)(kl)
                     idx = idx_table(a,d,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,4); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | lnab~df >
                        hmatel = h2a_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,4,5,2) !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/1,noa-1/), (/1,nub/), nua, nua, noa, nub)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/1,4,5,2/), nua, nua, noa, nub, nloc, n4abaa, resid)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,d,k,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | knab~ed >
                        hmatel = h2a_voov(c,n,l,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(c,d,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | kncb~ed >
                        hmatel = -h2a_voov(a,n,l,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(a,c,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | knab~ec >
                        hmatel = -h2a_voov(d,n,l,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(a,d,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | lnab~ed >
                        hmatel = -h2a_voov(c,n,k,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (ac)(kl)
                     idx = idx_table(c,d,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | lncb~ed >
                        hmatel = h2a_voov(a,n,k,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (cd)(kl)
                     idx = idx_table(a,c,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | lnab~ec >
                        hmatel = h2a_voov(d,n,k,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (3,4,5,2) !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/1,noa-1/), (/1,nub/), nua, nua, noa, nub)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/3,4,5,2/), nua, nua, noa, nub, nloc, n4abaa, resid)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(c,d,k,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | kneb~cd >
                        hmatel = h2a_voov(a,n,l,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(a,d,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | kneb~ad >
                        hmatel = -h2a_voov(c,n,l,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (ad), -
                     idx = idx_table(a,c,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | kneb~ac >
                        hmatel = h2a_voov(d,n,l,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(c,d,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | lneb~cd >
                        hmatel = -h2a_voov(a,n,k,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (ac)(kl)
                     idx = idx_table(a,d,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | lneb~ad >
                        hmatel = h2a_voov(c,n,k,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (ad)(kl), -
                     idx = idx_table(a,c,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | lneb~ac >
                        hmatel = -h2a_voov(d,n,k,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,3,6,2) !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/2,noa/), (/1,nub/), nua, nua, noa, nub)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/1,3,6,2/), nua, nua, noa, nub, nloc, n4abaa, resid)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,l,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,4); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mlab~cf >
                        hmatel = h2a_voov(d,m,k,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     ! (ad), -
                     idx = idx_table(c,d,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,4); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mldb~cf >
                        hmatel = h2a_voov(a,m,k,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(a,d,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,4); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mlab~df >
                        hmatel = -h2a_voov(c,m,k,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(a,c,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,4); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mkab~cf >
                        hmatel = -h2a_voov(d,m,l,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (ad)(kl), -
                     idx = idx_table(c,d,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,4); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mkdb~cf >
                        hmatel = -h2a_voov(a,m,l,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (cd)(kl)
                     idx = idx_table(a,d,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,4); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mkab~df >
                        hmatel = h2a_voov(c,m,l,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,4,6,2) !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/2,noa/), (/1,nub/), nua, nua, noa, nub)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/1,4,6,2/), nua, nua, noa, nub, nloc, n4abaa, resid)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,d,l,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mlab~ed >
                        hmatel = h2a_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(c,d,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mlcb~ed >
                        hmatel = -h2a_voov(a,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(a,c,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mlab~ec >
                        hmatel = -h2a_voov(d,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(a,d,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mkab~ed >
                        hmatel = -h2a_voov(c,m,l,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (ac)(kl)
                     idx = idx_table(c,d,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mkcb~ed >
                        hmatel = h2a_voov(a,m,l,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (cd)(kl)
                     idx = idx_table(a,c,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mkab~ec >
                        hmatel = h2a_voov(d,m,l,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (3,4,6,2) !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/2,noa/), (/1,nub/), nua, nua, noa, nub)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/3,4,6,2/), nua, nua, noa, nub, nloc, n4abaa, resid)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(c,d,l,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mleb~cd >
                        hmatel = h2a_voov(a,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(a,d,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mleb~ad >
                        hmatel = -h2a_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (ad), -
                     idx = idx_table(a,c,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mleb~ac >
                        hmatel = h2a_voov(d,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(c,d,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mkeb~cd >
                        hmatel = -h2a_voov(a,m,l,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (ac)(kl)
                     idx = idx_table(a,d,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mkeb~ad >
                        hmatel = h2a_voov(c,m,l,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (ad)(kl), -
                     idx = idx_table(a,c,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mkeb~ac >
                        hmatel = -h2a_voov(d,m,l,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!! diagram 7: A(d/ac)A(kl) h2b(dm~le~) * r4c(ab~ce~km~)
                  ! copy over excitations
                  allocate(excits_buff(n4abab,6),amps_buff(n4abab))
                  excits_buff(:,:) = r4c_excits(:,:)
                  amps_buff(:) = r4c_amps(:)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2 * (nub-1) * noa 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,nub,noa))
                  !!! SB: (1,3,2,5) !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub-1/), (/1,noa/), nua, nua, nub, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,2,5/), nua, nua, nub, noa, nloc, n4abab)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,4); n = excits_buff(jdet,6);
                        ! compute < klab~cd | h2b(voov) | kn~ab~cf~ >
                        hmatel = h2b_voov(d,n,l,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ad), -
                     idx = idx_table(c,d,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,4); n = excits_buff(jdet,6);
                        ! compute < klab~cd | h2b(voov) | kn~cb~df~ >
                        hmatel = h2b_voov(a,n,l,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(a,d,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,4); n = excits_buff(jdet,6);
                        ! compute < klab~cd | h2b(voov) | kn~ab~df~ >
                        hmatel = -h2b_voov(c,n,l,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(a,c,b,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,4); n = excits_buff(jdet,6);
                        ! compute < klab~cd | h2b(voov) | ln~ab~cf~ >
                        hmatel = -h2b_voov(d,n,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ad)(kl), -
                     idx = idx_table(c,d,b,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,4); n = excits_buff(jdet,6);
                        ! compute < klab~cd | h2b(voov) | ln~cb~df~ >
                        hmatel = -h2b_voov(a,n,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (cd)(kl)
                     idx = idx_table(a,d,b,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,4); n = excits_buff(jdet,6);
                        ! compute < klab~cd | h2b(voov) | ln~ab~df~ >
                        hmatel = h2b_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,3,4,5) !!!
                  !!! THIS SB IS IDENTICAL TO THE ABOVE ON EXCEPT ALL SIGNS ARE REVERSED !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/2,nub/), (/1,noa/), nua, nua, nub, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,5/), nua, nua, nub, noa, nloc, n4abab)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        ! compute < klab~cd | h2b(voov) | kn~af~cb~ >
                        hmatel = -h2b_voov(d,n,l,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ad), -
                     idx = idx_table(c,d,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        ! compute < klab~cd | h2b(voov) | kn~cf~db~ >
                        hmatel = -h2b_voov(a,n,l,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(a,d,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        ! compute < klab~cd | h2b(voov) | kn~af~db~ >
                        hmatel = h2b_voov(c,n,l,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(a,c,b,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        ! compute < klab~cd | h2b(voov) | ln~af~cb~ >
                        hmatel = h2b_voov(d,n,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ad)(kl), -
                     idx = idx_table(c,d,b,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        ! compute < klab~cd | h2b(voov) | ln~cf~db~ >
                        hmatel = h2b_voov(a,n,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (cd)(kl)
                     idx = idx_table(a,d,b,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        ! compute < klab~cd | h2b(voov) | ln~af~db~ >
                        hmatel = -h2b_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate buffer arrays
                  deallocate(excits_buff,amps_buff)

                  !!! diagram 8: A(kl) -h2b(mb~le~) * r4b(ae~cdkm)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)*(nua-2)/6 * (noa-1) 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,nua,noa))
                  !!! SB: (1,3,4,5) !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/1,noa-1/), nua, nua, nua, noa)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/1,3,4,5/), nua, nua, nua, noa, nloc, n4abaa, resid)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,d,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,2); n = r4b_excits(jdet,6);
                        ! compute < ab~cdkl | h2b(ovov) | af~cdkn >
                        hmatel = -h2b_ovov(n,b,l,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     ! (kl)
                     idx = idx_table(a,c,d,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,2); n = r4b_excits(jdet,6);
                        ! compute < ab~cdkl | h2b(ovov) | af~cdln >
                        hmatel = h2b_ovov(n,b,k,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,3,4,6) !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/2,noa/), nua, nua, nua, noa)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/1,3,4,6/), nua, nua, nua, noa, nloc, n4abaa, resid)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,d,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,2); m = r4b_excits(jdet,5);
                        ! compute < ab~cdkl | h2b(ovov) | af~cdml >
                        hmatel = -h2b_ovov(m,b,k,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     ! (kl)
                     idx = idx_table(a,c,d,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,2); m = r4b_excits(jdet,5);
                        ! compute < ab~cdkl | h2b(ovov) | af~cdmk >
                        hmatel = h2b_ovov(m,b,l,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !do idet=1,n4abaa
                  !end do
                  
              end subroutine build_hr_4b

              subroutine build_hr_4c(resid,&
                                     r3b, r3c,&
                                     r4b_amps, r4b_excits,&
                                     r4c_amps, r4c_excits,&
                                     r4d_amps, r4d_excits,&
                                     t2a, t2b, t2c,&
                                     h1a_oo, h1b_oo, h1a_vv, h1b_vv,&
                                     h2a_vvvv, h2a_voov, h2a_vvov,&
                                     h2b_vvvv, h2b_oooo, h2b_voov, h2b_ovvo, h2b_ovov, h2b_vovo,&
                                     h2b_vvov, h2b_vvvo, h2b_vooo, h2b_ovoo,&
                                     h2c_vvvv, h2c_voov, h2c_vvov,&
                                     x3b_vvvv, x3b_vovo, x3c_vvvv, x3c_ovvo,&
                                     x2b_oo,&
                                     n4abaa, n4abab, n4abbb,&
                                     noa, nua, nob, nub)
                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n4abaa, n4abab, n4abbb
                  !
                  real(kind=8), intent(in) :: r3b(nua,nub,nua,noa)
                  real(kind=8), intent(in) :: r3c(nua,nub,nub,nob)
                  real(kind=8), intent(in) :: t2a(nua,nua,noa,noa)
                  real(kind=8), intent(in) :: t2b(nua,nub,noa,nob)
                  real(kind=8), intent(in) :: t2c(nub,nub,nob,nob)
                  real(kind=8), intent(in) :: r4b_amps(n4abaa), r4d_amps(n4abbb)
                  integer, intent(in) :: r4b_excits(n4abaa,6), r4d_excits(n4abbb,6)
                  ! Input H and X arrays
                  real(kind=8), intent(in) :: h1a_oo(noa,noa)
                  real(kind=8), intent(in) :: h1b_oo(nob,nob)
                  real(kind=8), intent(in) :: h1b_vv(nub,nub)
                  real(kind=8), intent(in) :: h1a_vv(nua,nua)
                  real(kind=8), intent(in) :: h2a_vvvv(nua,nua,nua,nua)
                  real(kind=8), intent(in) :: h2a_voov(nua,noa,noa,nua)
                  real(kind=8), intent(in) :: h2a_vvov(nua,nua,noa,nua)
                  real(kind=8), intent(in) :: h2b_vvvv(nua,nub,nua,nub)
                  real(kind=8), intent(in) :: h2b_oooo(noa,nob,noa,nob)
                  real(kind=8), intent(in) :: h2b_voov(nua,nob,noa,nub)
                  real(kind=8), intent(in) :: h2b_ovvo(noa,nub,nua,nob)
                  real(kind=8), intent(in) :: h2b_ovov(noa,nub,noa,nub)
                  real(kind=8), intent(in) :: h2b_vovo(nua,nob,nua,nob)
                  real(kind=8), intent(in) :: h2b_vvov(nua,nub,noa,nub)
                  real(kind=8), intent(in) :: h2b_vvvo(nua,nub,nua,nob)
                  real(kind=8), intent(in) :: h2b_vooo(nua,nob,noa,nob)
                  real(kind=8), intent(in) :: h2b_ovoo(noa,nub,noa,nob)
                  real(kind=8), intent(in) :: h2c_vvvv(nub,nub,nub,nub)
                  real(kind=8), intent(in) :: h2c_voov(nub,nob,nob,nub)
                  real(kind=8), intent(in) :: h2c_vvov(nub,nub,nob,nub)
                  !
                  real(kind=8), intent(in) :: x3b_vvvv(nua,nub,nua,nua)
                  real(kind=8), intent(in) :: x3b_vovo(nua,nob,nua,nob)
                  real(kind=8), intent(in) :: x3c_vvvv(nua,nub,nub,nub)
                  real(kind=8), intent(in) :: x3c_ovvo(noa,nub,nub,nob)
                  real(kind=8), intent(in) :: x2b_oo(noa,nob)
                  ! Output and Inout variables
                  real(kind=8), intent(out) :: resid(n4abab)
                  integer, intent(inout) :: r4c_excits(n4abab,6)
                  !f2py intent(in,out) :: r4c_excits(0:n4abab-1,0:5)
                  real(kind=8), intent(inout) :: r4c_amps(n4abab)
                  !f2py intent(in,out) :: r4c_amps(0:n4abab-1)
                  ! Local variables
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:)
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)
                  real(kind=8) :: r_amp, hmatel, hmatel1, hmatel2, res_mm23
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc
                  real(kind=8) :: ff
                  
                  ! Zero the container that holds H*R
                  resid = 0.0d0
                  
                  !!! diagram 1: A(ac) h1a(ae) * r4c(eb~cd~kl~)
                  !!! diagram 7: h2a(acef) * r4c(eb~fd~kl~) 
                  ! allocate new sorting arrays
                  nloc = nub*(nub-1)/2 * noa * nob 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,noa,nob))
                  !!! SB: (2,4,5,6) !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,noa/), (/1,nob/), nub, nub, noa, nob)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/2,4,5,6/), nub, nub, noa, nob, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,d,k,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,1); f = r4c_excits(jdet,3);
                        ! compute < ab~cd~kl~ | h2a(vvvv) | eb~fd~kl~ >
                        hmatel = h2a_vvvv(a,c,e,f)
                        ! compute < ab~cd~kl~ | h1a(vv) | eb~fd~kl~ > = A(ac)(ef) h1a(ae) delta(f,c)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 + h1a_vv(a,e) ! (1)
                        if (a==f) hmatel1 = hmatel1 - h1a_vv(c,e) ! (ac)
                        if (c==e) hmatel1 = hmatel1 - h1a_vv(a,f) ! (ef)
                        if (a==e) hmatel1 = hmatel1 + h1a_vv(c,f) ! (ac)(ef)
                        hmatel = hmatel + hmatel1
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!! diagram 2: A(bd) h1b(bf) * r4c(af~cd~kl~)
                  !!! diagram 8: A(bd) h2c(bdef) * r4c(ae~cf~kl~)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2 * noa * nob 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,noa,nob))
                  !!! SB: (1,3,5,6) !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,noa/), (/1,nob/), nua, nua, noa, nob)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/1,3,5,6/), nua, nua, noa, nob, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,k,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,2); f = r4c_excits(jdet,4);
                        ! compute < ab~cd~kl~ | h2c(vvvv) | ae~cf~kl~ >
                        hmatel = h2c_vvvv(b,d,e,f)
                        ! compute < ab~cd~kl~ | h1a(vv) | ae~cf~kl~ > = A(bd)(ef) h1b(be) delta(f,d)
                        hmatel1 = 0.0d0
                        if (d==f) hmatel1 = hmatel1 + h1b_vv(b,e) ! (1)
                        if (b==f) hmatel1 = hmatel1 - h1b_vv(d,e) ! (bd)
                        if (d==e) hmatel1 = hmatel1 - h1b_vv(b,f) ! (ef)
                        if (b==e) hmatel1 = hmatel1 + h1b_vv(d,f) ! (bd)(ef)
                        hmatel = hmatel + hmatel1
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!! diagram 6: A(ac)A(bd) h2b(cd~ef~) * r4c(ab~ef~kl~)
                  ! allocate new sorting arrays
                  nloc = (nua-1)*(nub-1)*noa*nob 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nub,noa,nob))
                  !!! SB: (1,2,5,6) !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/1,nub-1/), (/1,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/1,2,5,6/), nua, nub, noa, nob, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,k,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,3); f = r4c_excits(jdet,4);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | ab~ef~kl~ >
                        hmatel = h2b_vvvv(c,d,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(c,b,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,3); f = r4c_excits(jdet,4);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | cb~ef~kl~ >
                        hmatel = -h2b_vvvv(a,d,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                     ! (bd)
                     idx = idx_table(a,d,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,3); f = r4c_excits(jdet,4);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | ad~ef~kl~ >
                        hmatel = -h2b_vvvv(c,b,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                     ! (ac)(bd)
                     idx = idx_table(c,d,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,3); f = r4c_excits(jdet,4);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | cd~ef~kl~ >
                        hmatel = h2b_vvvv(a,b,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,4,5,6) !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/2,nub/), (/1,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/1,4,5,6/), nua, nub, noa, nob, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,d,k,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,3); f = r4c_excits(jdet,2);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | af~ed~kl~ >
                        hmatel = h2b_vvvv(c,b,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(c,d,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,3); f = r4c_excits(jdet,2);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | cf~ed~kl~ >
                        hmatel = -h2b_vvvv(a,b,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                     ! (bd)
                     idx = idx_table(a,b,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,3); f = r4c_excits(jdet,2);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | af~eb~kl~ >
                        hmatel = -h2b_vvvv(c,d,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                     ! (ac)(bd)
                     idx = idx_table(c,b,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,3); f = r4c_excits(jdet,2);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | cf~eb~kl~ >
                        hmatel = h2b_vvvv(a,d,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (3,2,5,6) !!!
                  call get_index_table(idx_table, (/2,nua/), (/1,nub-1/), (/1,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/3,2,5,6/), nua, nub, noa, nob, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(c,b,k,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,1); f = r4c_excits(jdet,4);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | eb~cf~kl~ >
                        hmatel = h2b_vvvv(a,d,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(a,b,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,1); f = r4c_excits(jdet,4);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | eb~af~kl~ >
                        hmatel = -h2b_vvvv(c,d,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                     ! (bd)
                     idx = idx_table(c,d,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,1); f = r4c_excits(jdet,4);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | ed~cf~kl~ >
                        hmatel = -h2b_vvvv(a,b,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                     ! (ac)(bd)
                     idx = idx_table(a,d,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,1); f = r4c_excits(jdet,4);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | ed~af~kl~ >
                        hmatel = h2b_vvvv(c,b,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (3,4,5,6) !!!
                  call get_index_table(idx_table, (/2,nua/), (/2,nub/), (/1,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/3,4,5,6/), nua, nub, noa, nob, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(c,d,k,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,1); f = r4c_excits(jdet,2);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | ef~cd~kl~ >
                        hmatel = h2b_vvvv(a,b,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(a,d,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,1); f = r4c_excits(jdet,2);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | ef~ad~kl~ >
                        hmatel = -h2b_vvvv(c,b,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                     ! (bd)
                     idx = idx_table(c,b,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,1); f = r4c_excits(jdet,2);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | ef~cb~kl~ >
                        hmatel = -h2b_vvvv(a,d,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                     ! (ac)(bd)
                     idx = idx_table(a,b,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,1); f = r4c_excits(jdet,2);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | ef~ab~kl~ >
                        hmatel = h2b_vvvv(c,d,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!! diagram 3: -h1a(mk) * r4c(ab~cd~ml~) 
                  !!! diagram 4: -h1b(nl) * r4c(ab~cd~kn~)
                  !!! diagram 5: h2b(mn~kl~) * r4c(ab~cd~mn~)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2 * nub*(nub-1)/2 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,nub,nub))
                  !!! SB: (1,3,2,4) !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub-1/), (/-1,nub/), nua, nua, nub, nub)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/1,3,2,4/), nua, nua, nub, nub, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,b,d)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4c_excits(jdet,5); n = r4c_excits(jdet,6);
                        ! compute < ab~cd~kl~ | h2b(oooo) | ab~cd~mn~ >
                        hmatel = h2b_oooo(m,n,k,l)
                        ! compute < ab~cd~kl~ | h1a(oo) | ab~cd~mn~ > = -h1b(m,k) * delta(l,n)
                        hmatel1 = 0.0d0
                        if (l==n) hmatel1 = hmatel1 - h1a_oo(m,k) ! (1)
                        ! compute < ab~cd~kl~ | h1b(oo) | ab~cd~mn~ > = -h1b(n,l) * delta(k,m)
                        hmatel2 = 0.0d0
                        if (k==m) hmatel2 = hmatel2 - h1b_oo(n,l) ! (1)
                        hmatel = hmatel + hmatel1 + hmatel2
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!! diagram 9: A(ac) h2a(cmke) * r4c(ab~ed~ml~) 
                  ! allocate new sorting arrays
                  nloc = nub*(nub-1)/2 * (nua-1) * nob 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nua,nob))
                  !!! SB: (2,4,1,6) !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nua-1/), (/1,nob/), nub, nub, nua, nob)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/2,4,1,6/), nub, nub, nua, nob, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,d,a,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,3); m = r4c_excits(jdet,5);
                        ! compute < ab~cd~kl~ | h2a(voov) | ab~ed~ml~ >
                        hmatel = h2a_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(b,d,c,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,3); m = r4c_excits(jdet,5);
                        ! compute < ab~cd~kl~ | h2a(voov) | cb~ed~ml~ >
                        hmatel = -h2a_voov(a,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (2,4,3,6) !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/2,nua/), (/1,nob/), nub, nub, nua, nob)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/2,4,3,6/), nub, nub, nua, nob, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,d,c,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,1); m = r4c_excits(jdet,5);
                        ! compute < ab~cd~kl~ | h2a(voov) | eb~cd~ml~ >
                        hmatel = h2a_voov(a,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(b,d,a,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,1); m = r4c_excits(jdet,5);
                        ! compute < ab~cd~kl~ | h2a(voov) | eb~ad~ml~ >
                        hmatel = -h2a_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!! diagram 10: h2b(cm~ke~) * r4d(ab~e~d~m~l~) 
                  ! copy over excitations
                  allocate(excits_buff(n4abbb,6),amps_buff(n4abbb))
                  excits_buff(:,:) = r4d_excits(:,:)
                  amps_buff(:) = r4d_amps(:)
                  ! allocate new sorting arrays
                  nloc = nua * (nub-1)*(nub-2)/2 * (nob-1) 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nua,nob))
                  !!! SB: (2,4,1,6) !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/1,nua/), (/2,nob/), nub, nub, nua, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,4,1,6/), nub, nub, nua, nob, nloc, n4abbb)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,d,a,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        ! compute < ab~cd~kl~ | h2a(voov) | ab~e~d~m~l~ >
                        hmatel = h2b_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,d,c,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        ! compute < ab~cd~kl~ | h2a(voov) | cb~e~d~m~l~ >
                        hmatel = -h2b_voov(a,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (2,4,1,5) !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/1,nua/), (/1,nob-1/), nub, nub, nua, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,4,1,5/), nub, nub, nua, nob, nloc, n4abbb)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,d,a,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                        ! compute < ab~cd~kl~ | h2a(voov) | ab~e~d~l~m~ >
                        hmatel = -h2b_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,d,c,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                        ! compute < ab~cd~kl~ | h2a(voov) | cb~e~d~l~m~ >
                        hmatel = h2b_voov(a,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (2,3,1,6) !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/1,nua/), (/2,nob/), nub, nub, nua, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,1,6/), nub, nub, nua, nob, nloc, n4abbb)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,d,a,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,4); m = excits_buff(jdet,5);
                        ! compute < ab~cd~kl~ | h2a(voov) | ab~d~e~m~l~ >
                        hmatel = -h2b_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,d,c,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,4); m = excits_buff(jdet,5);
                        ! compute < ab~cd~kl~ | h2a(voov) | cb~d~e~m~l~ >
                        hmatel = h2b_voov(a,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (2,3,1,5) !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/1,nua/), (/1,nob-1/), nub, nub, nua, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,1,5/), nub, nub, nua, nob, nloc, n4abbb)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,d,a,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,4); m = excits_buff(jdet,6);
                        ! compute < ab~cd~kl~ | h2a(voov) | ab~d~e~m~l~ >
                        hmatel = h2b_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,d,c,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,4); m = excits_buff(jdet,6);
                        ! compute < ab~cd~kl~ | h2a(voov) | cb~d~e~m~l~ >
                        hmatel = -h2b_voov(a,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (3,4,1,6) !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/1,nua/), (/2,nob/), nub, nub, nua, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/3,4,1,6/), nub, nub, nua, nob, nloc, n4abbb)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,d,a,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        ! compute < ab~cd~kl~ | h2a(voov) | ae~b~d~m~l~ >
                        hmatel = -h2b_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,d,c,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        ! compute < ab~cd~kl~ | h2a(voov) | ce~b~d~m~l~ >
                        hmatel = h2b_voov(a,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (3,4,1,5) !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/1,nua/), (/1,nob-1/), nub, nub, nua, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/3,4,1,5/), nub, nub, nua, nob, nloc, n4abbb)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,d,a,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,6);
                        ! compute < ab~cd~kl~ | h2a(voov) | ae~b~d~m~l~ >
                        hmatel = h2b_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,d,c,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,6);
                        ! compute < ab~cd~kl~ | h2a(voov) | ce~b~d~m~l~ >
                        hmatel = -h2b_voov(a,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate buffer arrays
                  deallocate(excits_buff,amps_buff)

                  !!! diagram 11: A(bd) h2b(md~el~) * r4b(ab~cekm) 
                  ! copy over excitations
                  allocate(excits_buff(n4abaa,6),amps_buff(n4abaa))
                  excits_buff(:,:) = r4b_excits(:,:)
                  amps_buff(:) = r4b_amps(:)
                  ! allocate new sorting arrays
                  nloc = (nua-1)*(nua-2)/2 * nub * (noa-1) 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,nub,noa))
                  !!! SB: (1,3,2,5) !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/1,nub/), (/1,noa-1/), nua, nua, nub, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,2,5/), nua, nua, nub, noa, nloc, n4abaa)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,4); m = excits_buff(jdet,6);
                        ! compute < ab~cd~kl~ | h2b(ovvo) | ab~cekm >
                        hmatel = h2b_ovvo(m,d,e,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bd)
                     idx = idx_table(a,c,d,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,4); m = excits_buff(jdet,6);
                        ! compute < ab~cd~kl~ | h2b(ovvo) | ad~cekm >
                        hmatel = -h2b_ovvo(m,b,e,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,3,2,6) !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/1,nub/), (/2,noa/), nua, nua, nub, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,2,6/), nua, nua, nub, noa, nloc, n4abaa)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,4); m = excits_buff(jdet,5);
                        ! compute < ab~cd~kl~ | h2b(ovvo) | ab~cekm >
                        hmatel = -h2b_ovvo(m,d,e,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bd)
                     idx = idx_table(a,c,d,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,4); m = excits_buff(jdet,5);
                        ! compute < ab~cd~kl~ | h2b(ovvo) | ab~cekm >
                        hmatel = h2b_ovvo(m,b,e,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,4,2,5) !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/1,nub/), (/1,noa-1/), nua, nua, nub, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,4,2,5/), nua, nua, nub, noa, nloc, n4abaa)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                        ! compute < ab~cd~kl~ | h2b(ovvo) | ab~eckm >
                        hmatel = -h2b_ovvo(m,d,e,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bd)
                     idx = idx_table(a,c,d,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                        ! compute < ab~cd~kl~ | h2b(ovvo) | ad~eckm >
                        hmatel = h2b_ovvo(m,b,e,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,4,2,6) !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/1,nub/), (/2,noa/), nua, nua, nub, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,4,2,6/), nua, nua, nub, noa, nloc, n4abaa)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        ! compute < ab~cd~kl~ | h2b(ovvo) | ab~cekm >
                        hmatel = h2b_ovvo(m,d,e,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bd)
                     idx = idx_table(a,c,d,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        ! compute < ab~cd~kl~ | h2b(ovvo) | ab~cekm >
                        hmatel = -h2b_ovvo(m,b,e,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (3,4,2,5) !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/1,nub/), (/1,noa-1/), nua, nua, nub, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/3,4,2,5/), nua, nua, nub, noa, nloc, n4abaa)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,6);
                        ! compute < ab~cd~kl~ | h2b(ovvo) | eb~ackm >
                        hmatel = h2b_ovvo(m,d,e,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bd)
                     idx = idx_table(a,c,d,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,6);
                        ! compute < ab~cd~kl~ | h2b(ovvo) | ab~cekm >
                        hmatel = -h2b_ovvo(m,b,e,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (3,4,2,6) !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/1,nub/), (/2,noa/), nua, nua, nub, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/3,4,2,6/), nua, nua, nub, noa, nloc, n4abaa)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        ! compute < ab~cd~kl~ | h2b(ovvo) | ab~cekm >
                        hmatel = -h2b_ovvo(m,d,e,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bd)
                     idx = idx_table(a,c,d,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        ! compute < ab~cd~kl~ | h2b(ovvo) | ab~cekm >
                        hmatel = h2b_ovvo(m,b,e,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate buffer arrays
                  deallocate(excits_buff,amps_buff)
                  
                  !!! diagram 12: A(bd) h2c(d~m~l~e~) * r4c(ab~ce~km~) 
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2 * (nub-1) * noa 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,nub,noa))
                  !!! SB: (1,3,2,5) !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub-1/), (/1,noa/), nua, nua, nub, noa)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/1,3,2,5/), nua, nua, nub, noa, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,b,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,4); m = r4c_excits(jdet,6);
                        ! compute < ab~cd~kl~ | h2c(voov) | ab~ce~km~ >
                        hmatel = h2c_voov(d,m,l,e)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     ! (bd)
                     idx = idx_table(a,c,d,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,4); m = r4c_excits(jdet,6);
                        ! compute < ab~cd~kl~ | h2c(voov) | ad~ce~km~ >
                        hmatel = -h2c_voov(b,m,l,e)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,3,4,5) !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/2,nub/), (/1,noa/), nua, nua, nub, noa)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/1,3,4,5/), nua, nua, nub, noa, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,d,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,2); m = r4c_excits(jdet,6);
                        ! compute < ab~cd~kl~ | h2c(voov) | ae~cd~km~ >
                        hmatel = h2c_voov(b,m,l,e)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     ! (bd)
                     idx = idx_table(a,c,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,2); m = r4c_excits(jdet,6);
                        ! compute < ab~cd~kl~ | h2c(voov) | ae~cb~km~ >
                        hmatel = -h2c_voov(d,m,l,e)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!! diagram 13: A(bd) -h2b(md~ke~) * r4c(ab~ce~ml~) 
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2 * (nub-1) * nob 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,nub,nob))
                  !!! SB: (1,3,2,6) !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub-1/), (/1,nob/), nua, nua, nub, nob)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/1,3,2,6/), nua, nua, nub, nob, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,b,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,4); m = r4c_excits(jdet,5);
                        ! compute < ab~cd~kl~ | h2b(ovov) | ab~ce~ml~ >
                        hmatel = -h2b_ovov(m,d,k,e)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     ! (bd)
                     idx = idx_table(a,c,d,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,4); m = r4c_excits(jdet,5);
                        ! compute < ab~cd~kl~ | h2b(ovov) | ad~ce~ml~ >
                        hmatel = h2b_ovov(m,b,k,e)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,3,4,6) !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/2,nub/), (/1,nob/), nua, nua, nub, nob)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/1,3,4,6/), nua, nua, nub, nob, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,d,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,2); m = r4c_excits(jdet,5);
                        ! compute < ab~cd~kl~ | h2b(ovov) | ae~cd~ml~ >
                        hmatel = -h2b_ovov(m,b,k,e)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     ! (bd)
                     idx = idx_table(a,c,b,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,2); m = r4c_excits(jdet,5);
                        ! compute < ab~cd~kl~ | h2b(ovov) | ae~cb~ml~ >
                        hmatel = h2b_ovov(m,d,k,e)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!! diagram 14: A(ac) -h2b(cm~el~) * r4c(ab~ed~km~) 
                  ! allocate new sorting arrays
                  nloc = nub*(nub-1)/2 * (nua-1) * noa 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nua,noa))
                  !!! SB: (2,4,1,5) !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nua-1/), (/1,noa/), nub, nub, nua, noa)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/2,4,1,5/), nub, nub, nua, noa, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,d,a,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,3); m = r4c_excits(jdet,6);
                        ! compute < ab~cd~kl~ | h2b(vovo) | ab~ed~km~ >
                        hmatel = -h2b_vovo(c,m,e,l)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(b,d,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,3); m = r4c_excits(jdet,6);
                        ! compute < ab~cd~kl~ | h2b(vovo) | cb~ed~km~ >
                        hmatel = h2b_vovo(a,m,e,l)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (2,4,3,5) !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/2,nua/), (/1,noa/), nub, nub, nua, noa)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/2,4,3,5/), nub, nub, nua, noa, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,d,c,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,1); m = r4c_excits(jdet,6);
                        ! compute < ab~cd~kl~ | h2b(vovo) | eb~cd~km~ >
                        hmatel = -h2b_vovo(a,m,e,l)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(b,d,a,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,1); m = r4c_excits(jdet,6);
                        ! compute < ab~cd~kl~ | h2b(vovo) | eb~ad~km~ >
                        hmatel = h2b_vovo(c,m,e,l)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

              end subroutine build_hr_4c

              subroutine build_hr_4d(resid,&
                                     r3c,&
                                     r4c_amps, r4c_excits,&
                                     r4d_amps, r4d_excits,&
                                     t2b, t2c,&
                                     h1b_oo, h1a_vv, h1b_vv,&
                                     h2b_vvvv, h2b_ovvo, h2b_vovo, h2b_vvvo,&
                                     h2c_vvvv, h2c_oooo, h2c_voov, h2c_vvov, h2c_vooo,&
                                     x3c_vvvv, x3c_ovvo, x3c_vvoo, x2b_oo,&
                                     n4abab, n4abbb,&
                                     noa, nua, nob, nub)
                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n4abab, n4abbb
                  !
                  real(kind=8), intent(in) :: r3c(nua,nub,nub,nob)
                  real(kind=8), intent(in) :: t2b(nua,nub,noa,nob)
                  real(kind=8), intent(in) :: t2c(nub,nub,nob,nob)
                  real(kind=8), intent(in) :: r4c_amps(n4abab)
                  integer, intent(in) :: r4c_excits(n4abab,6)
                  ! Input H and X arrays
                  real(kind=8), intent(in) :: h1b_oo(nob,nob)
                  real(kind=8), intent(in) :: h1b_vv(nub,nub)
                  real(kind=8), intent(in) :: h1a_vv(nua,nua)
                  real(kind=8), intent(in) :: h2b_vvvv(nua,nub,nua,nub)
                  real(kind=8), intent(in) :: h2b_ovvo(noa,nub,nua,nob)
                  real(kind=8), intent(in) :: h2b_vovo(nua,nob,nua,nob)
                  real(kind=8), intent(in) :: h2b_vvvo(nua,nub,nua,nob)
                  real(kind=8), intent(in) :: h2c_vvvv(nub,nub,nub,nub)
                  real(kind=8), intent(in) :: h2c_oooo(nob,nob,nob,nob)
                  real(kind=8), intent(in) :: h2c_voov(nub,nob,nob,nub)
                  real(kind=8), intent(in) :: h2c_vvov(nub,nub,nob,nub)
                  real(kind=8), intent(in) :: h2c_vooo(nub,nob,nob,nob)
                  real(kind=8), intent(in) :: x3c_vvvv(nua,nub,nub,nub)
                  real(kind=8), intent(in) :: x3c_ovvo(noa,nub,nub,nob)
                  real(kind=8), intent(in) :: x3c_vvoo(nua,nub,nob,nob)
                  real(kind=8), intent(in) :: x2b_oo(noa,nob)
                  ! Output and Inout variables
                  real(kind=8), intent(out) :: resid(n4abbb)
                  integer, intent(inout) :: r4d_excits(n4abbb,6)
                  !f2py intent(in,out) :: r4d_excits(0:n4abbb-1,0:5)
                  real(kind=8), intent(inout) :: r4d_amps(n4abbb)
                  !f2py intent(in,out) :: r4d_amps(0:n4abbb-1)
                  ! Local variables
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:)
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)
                  real(kind=8) :: r_amp, hmatel, hmatel1, res_mm23
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc
                  real(kind=8) :: ff
                  
                  ! Zero the container that holds H*R
                  resid = 0.0d0
         
              end subroutine build_hr_4d

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
