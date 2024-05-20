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
                  real(kind=8), intent(in) :: r3b(nua,nub,nua,nob)
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
                  
                  ! Zero the container that holds H*R
                  resid = 0.0d0
                  
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
