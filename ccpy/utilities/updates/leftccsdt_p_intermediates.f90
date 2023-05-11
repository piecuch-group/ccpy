module leftccsdt_p_intermediates

        implicit none
        
        contains
        
              subroutine compute_x1a_oo(x1a_oo,&
                                        t3a_amps, t3a_excits,&
                                        t3b_amps, t3b_excits,&
                                        t3c_amps, t3c_excits,&
                                        l3a_amps, l3a_excits,&
                                        l3b_amps, l3b_excits,&
                                        l3c_amps, l3c_excits,&
                                        n3aaa_t, n3aab_t, n3abb_t,&
                                        n3aaa_l, n3aab_l, n3abb_l,&
                                        noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa_t, n3aab_t, n3abb_t
                  integer, intent(in) :: n3aaa_l, n3aab_l, n3abb_l

                  integer, intent(in) :: t3a_excits(6,n3aaa_t)
                  real(kind=8), intent(in) :: t3a_amps(n3aaa_t)
                  integer, intent(in) :: t3b_excits(6,n3aab_t)
                  real(kind=8), intent(in) :: t3b_amps(n3aab_t)
                  integer, intent(in) :: t3c_excits(6,n3abb_t)
                  real(kind=8), intent(in) :: t3c_amps(n3abb_t)
                  integer, intent(in) :: l3a_excits(6,n3aaa_l)
                  real(kind=8), intent(in) :: l3a_amps(n3aaa_l)
                  integer, intent(in) :: l3b_excits(6,n3aab_l)
                  real(kind=8), intent(in) :: l3b_amps(n3aab_l)
                  integer, intent(in) :: l3c_excits(6,n3abb_l)
                  real(kind=8), intent(in) :: l3c_amps(n3abb_l)

                  real(kind=8), intent(out) :: x1a_oo(noa,noa)
                 
                  integer, allocatable :: t3_excits_buff(:,:)
                  real(kind=8), allocatable :: t3_amps_buff(:) 
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)

                  real(kind=8) :: t_amp, l_amp, lt_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx

                  x1a_oo = 0.0d0
                  !!!! X1A(mi) = 1/6 l3a(efgmno) t3a(efgino) -> X1A(im) = 1/6 l3a(abcijk) * t3a(abcmjk)
                  ! copy t3a into buffer
                  allocate(t3_amps_buff(n3aaa_t), t3_excits_buff(6,n3aaa_t))
                  t3_amps_buff(:) = t3a_amps(:)
                  t3_excits_buff(:,:) = t3a_excits(:,:)
                  ! allocate new sorting arrays
                  allocate(loc_arr(nua*(nua-1)*(nua-2)/6*noa,2))
                  allocate(idx_table(nua,nua,nua,noa))
                  !!! ABCK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/3,noa/), nua, nua, nua, noa)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,2,3,6/), nua, nua, nua, noa, nua*(nua-1)*(nua-2)/6*noa, n3aaa_t)
                  do idet = 1, n3aaa_l
                     l_amp = l3a_amps(idet)
                     a = l3a_excits(1,idet); b = l3a_excits(2,idet); c = l3a_excits(3,idet);
                     i = l3a_excits(4,idet); j = l3a_excits(5,idet); k = l3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = t3_amps_buff(jdet)
                        m = t3_excits_buff(4,jdet); n = t3_excits_buff(5,jdet);
                        ! compute < ijkabc | mnkabc > ->
                        ! N[i+j+k+cba a+b+c+knm] = delta(j,n)N[i+m] + delta(i,m) N[j+n] - delta(i,n)N[j+m] - delta(j,m) N[i+n]
                        !                        = A(ij)A(nm) delta(i,m) N[j+n]
                        lt_amp = 0.5d0 * l_amp * t_amp
                        if (i==m) x1a_oo(j,n) = x1a_oo(j,n) + lt_amp ! (1)
                        if (j==m) x1a_oo(i,n) = x1a_oo(i,n) - lt_amp ! (ij)
                        if (i==n) x1a_oo(j,m) = x1a_oo(j,m) - lt_amp ! (nm)
                        if (j==n) x1a_oo(i,m) = x1a_oo(i,m) + lt_amp ! (ij)(nm)
                     end do
                     ! (ik)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                          t_amp = t3_amps_buff(jdet)
                          m = t3_excits_buff(4,jdet); n = t3_excits_buff(5,jdet);
                          ! compute < ijkabc | mniabc > ->
                          ! N[i+j+k+cba a+b+c+inm] = -A(jk)A(nm) delta(k,m) N[j+n]
                          lt_amp = 0.5d0 * l_amp * t_amp
                          if (k==m) x1a_oo(j,n) = x1a_oo(j,n) - lt_amp ! (1)
                          if (j==m) x1a_oo(k,n) = x1a_oo(k,n) + lt_amp ! (jk)
                          if (k==n) x1a_oo(j,m) = x1a_oo(j,m) + lt_amp ! (nm)
                          if (j==n) x1a_oo(k,m) = x1a_oo(k,m) - lt_amp ! (jk)(nm)
                       end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                          t_amp = t3_amps_buff(jdet)
                          m = t3_excits_buff(4,jdet); n = t3_excits_buff(5,jdet);
                          ! compute < ijkabc | mnjabc > ->
                          ! N[i+j+k+cba a+b+c+jnm] = -A(ik)A(nm) delta(i,m) N[k+n]
                          lt_amp = 0.5d0 * l_amp * t_amp
                          if (i==m) x1a_oo(k,n) = x1a_oo(k,n) - lt_amp ! (1)
                          if (k==m) x1a_oo(i,n) = x1a_oo(i,n) + lt_amp ! (ik)
                          if (i==n) x1a_oo(k,m) = x1a_oo(k,m) + lt_amp ! (nm)  
                          if (k==n) x1a_oo(i,m) = x1a_oo(i,m) - lt_amp ! (ik)(nm)
                       end do
                     end if
                  end do
                  !!! ABCI LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/1,noa-2/), nua, nua, nua, noa)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,2,3,4/), nua, nua, nua, noa, nua*(nua-1)*(nua-2)/6*noa, n3aaa_t)
                  do idet = 1, n3aaa_l
                     l_amp = l3a_amps(idet)
                     a = l3a_excits(1,idet); b = l3a_excits(2,idet); c = l3a_excits(3,idet);
                     i = l3a_excits(4,idet); j = l3a_excits(5,idet); k = l3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,i)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = t3_amps_buff(jdet)
                        m = t3_excits_buff(5,jdet); n = t3_excits_buff(6,jdet);
                        ! compute < ijkabc | imnabc > -> A(jk)A(mn) delta(j,m) N[k+n]
                        lt_amp = 0.5d0 * l_amp * t_amp
                        if (j==m) x1a_oo(k,n) = x1a_oo(k,n) + lt_amp ! (1)
                        if (k==m) x1a_oo(j,n) = x1a_oo(j,n) - lt_amp ! (jk)
                        if (j==n) x1a_oo(k,m) = x1a_oo(k,m) - lt_amp ! (mn)
                        if (k==n) x1a_oo(j,m) = x1a_oo(j,m) + lt_amp ! (jk)(mn)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                          t_amp = t3_amps_buff(jdet)
                          m = t3_excits_buff(5,jdet); n = t3_excits_buff(6,jdet);
                          ! compute < ijkabc | jmnabc > -> -A(ik)A(mn) delta(i,m) N[k+n]
                          lt_amp = 0.5d0 * l_amp * t_amp
                          if (i==m) x1a_oo(k,n) = x1a_oo(k,n) - lt_amp ! (1)
                          if (k==m) x1a_oo(i,n) = x1a_oo(i,n) + lt_amp ! (ik)
                          if (i==n) x1a_oo(k,m) = x1a_oo(k,m) + lt_amp ! (mn)
                          if (k==n) x1a_oo(i,m) = x1a_oo(i,m) - lt_amp ! (ik)(mn)
                       end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                          t_amp = t3_amps_buff(jdet)
                          m = t3_excits_buff(5,jdet); n = t3_excits_buff(6,jdet);
                          ! compute < ijkabc | kmnabc > -> -A(ij)A(mn) delta(j,m) N[i+n]
                          lt_amp = 0.5d0 * l_amp * t_amp
                          if (j==m) x1a_oo(i,n) = x1a_oo(i,n) - lt_amp ! (1)
                          if (i==m) x1a_oo(j,n) = x1a_oo(j,n) + lt_amp ! (ij)
                          if (j==n) x1a_oo(i,m) = x1a_oo(i,m) + lt_amp ! (mn)
                          if (i==n) x1a_oo(j,m) = x1a_oo(j,m) - lt_amp ! (ij)(mn)
                        end do
                     end if
                  end do
                  !!! ABCJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/2,noa-1/), nua, nua, nua, noa)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,2,3,5/), nua, nua, nua, noa, nua*(nua-1)*(nua-2)/6*noa, n3aaa_t)
                  do idet = 1, n3aaa_l
                     l_amp = l3a_amps(idet)
                     a = l3a_excits(1,idet); b = l3a_excits(2,idet); c = l3a_excits(3,idet);
                     i = l3a_excits(4,idet); j = l3a_excits(5,idet); k = l3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,j)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                       t_amp = t3_amps_buff(jdet)
                       m = t3_excits_buff(4,jdet); n = t3_excits_buff(6,jdet);
                       ! compute < ijkabc | mjnabc > -> A(ik)A(mn) delta(k,n) N[i+ m]
                       lt_amp = 0.5d0 * l_amp * t_amp
                       if (k==n) x1a_oo(i,m) = x1a_oo(i,m) + lt_amp ! (1)
                       if (i==n) x1a_oo(k,m) = x1a_oo(k,m) - lt_amp ! (ik)
                       if (k==m) x1a_oo(i,n) = x1a_oo(i,n) - lt_amp ! (mn)
                       if (i==m) x1a_oo(k,n) = x1a_oo(k,n) + lt_amp ! (ik)(mn)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                          t_amp = t3_amps_buff(jdet)
                          m = t3_excits_buff(4,jdet); n = t3_excits_buff(6,jdet);
                          ! compute < ijkabc | minabc > -> -A(jk)A(mn) delta(k,n) N[j+ m]
                          lt_amp = 0.5d0 * l_amp * t_amp
                          if (k==n) x1a_oo(j,m) = x1a_oo(j,m) - lt_amp ! (1)
                          if (j==n) x1a_oo(k,m) = x1a_oo(k,m) + lt_amp ! (jk)
                          if (k==m) x1a_oo(j,n) = x1a_oo(j,n) + lt_amp ! (mn)
                          if (j==m) x1a_oo(k,n) = x1a_oo(k,n) - lt_amp ! (jk)(mn)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                          t_amp = t3_amps_buff(jdet)
                          m = t3_excits_buff(4,jdet); n = t3_excits_buff(6,jdet);
                          ! compute < ijkabc | mknabc > -> -A(ij)A(mn) delta(j,n) N[i+ m]
                          lt_amp = 0.50 * l_amp * t_amp
                          if (j==n) x1a_oo(i,m) = x1a_oo(i,m) - lt_amp ! (1)
                          if (i==n) x1a_oo(j,m) = x1a_oo(j,m) + lt_amp ! (ij)
                          if (j==m) x1a_oo(i,n) = x1a_oo(i,n) + lt_amp ! (mn)
                          if (i==m) x1a_oo(j,n) = x1a_oo(j,n) - lt_amp ! (ij)(mn)
                        end do 
                     end if
                  end do
                  deallocate(loc_arr,idx_table,t3_amps_buff,t3_excits_buff)
                  !!!! X1A(mi) = 1/6 l3b(efgmno) t3b(efgino) -> X1A(im) = 1/6 l3b(abcijk) * t3b(abcmjk)
                  ! copy t3b into buffer
                  allocate(t3_amps_buff(n3aab_t), t3_excits_buff(6,n3aab_t))
                  t3_amps_buff(:) = t3b_amps(:)
                  t3_excits_buff(:,:) = t3b_excits(:,:)
                  ! allocate new sorting arrays
                  allocate(loc_arr(nua*(nua-1)/2*nub*nob,2))
                  allocate(idx_table(nua,nua,nub,nob))
                  !!! ABCK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/1,nob/), nua, nua, nub, nob)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,2,3,6/), nua, nua, nub, nob, nua*(nua-1)/2*nub*nob, n3aab_t)
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = t3_amps_buff(jdet)
                        m = t3_excits_buff(4,jdet); n = t3_excits_buff(5,jdet);
                        ! compute < ijk~abc~ | mnk~abc~ > -> A(ij)A(mn) delta(j,n) N[i+ m]
                        lt_amp = l_amp * t_amp
                        ! NOTE: no factor of 1/2 here
                        if (i==m) x1a_oo(j,n) = x1a_oo(j,n) + lt_amp ! (1)
                        if (j==m) x1a_oo(i,n) = x1a_oo(i,n) - lt_amp ! (ij)
                        if (i==n) x1a_oo(j,m) = x1a_oo(j,m) - lt_amp ! (nm)
                        if (j==n) x1a_oo(i,m) = x1a_oo(i,m) + lt_amp ! (ij)(nm)
                     end do
                  end do
                  deallocate(loc_arr,idx_table,t3_amps_buff,t3_excits_buff)
                  !!!! X1A(mi) = 1/6 l3c(efgmno) t3c(efgino) -> X1A(im) = 1/6 l3c(abcijk) * t3c(abcmjk)
                  ! copy t3c into buffer
                  allocate(t3_amps_buff(n3abb_t), t3_excits_buff(6,n3abb_t))
                  t3_amps_buff(:) = t3c_amps(:)
                  t3_excits_buff(:,:) = t3c_excits(:,:)
                  ! allocate new sorting arrays
                  allocate(loc_arr(nub*(nub-1)/2*nua*nob,2))
                  allocate(idx_table(nub,nub,nua,nob))
                  !!! BCAJ LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nua/), (/1,nob-1/), nub, nub, nua, nob)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/2,3,1,5/), nub, nub, nua, nob, nub*(nub-1)/2*nua*nob, n3abb_t)
                  do idet = 1, n3abb_l
                     l_amp = l3c_amps(idet)
                     a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                     i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,a,j)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = t3_amps_buff(jdet)
                        m = t3_excits_buff(4,jdet); n = t3_excits_buff(6,jdet);
                        ! compute < ij~k~ab~c~ | mj~n~ab~c~ > -> delta(k,n) N[i+ m]
                        lt_amp = 0.5d0 * l_amp * t_amp
                        if (k==n) x1a_oo(i,m) = x1a_oo(i,m) + lt_amp ! (1)
                     end do
                     ! (jk)
                     idx = idx_table(b,c,a,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = t3_amps_buff(jdet)
                           m = t3_excits_buff(4,jdet); n = t3_excits_buff(6,jdet);
                           ! compute < ij~k~ab~c~ | mk~n~ab~c~ > -> -delta(j,n) N[i+ m]
                           lt_amp = 0.5d0 * l_amp * t_amp
                           if (j==n) x1a_oo(i,m) = x1a_oo(i,m) - lt_amp ! (1)
                        end do
                     end if
                  end do
                  !!! BCAK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nua/), (/2,nob/), nub, nub, nua, nob)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/2,3,1,6/), nub, nub, nua, nob, nub*(nub-1)/2*nua*nob, n3abb_t)
                  do idet = 1, n3abb_l
                     l_amp = l3c_amps(idet)
                     a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                     i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                     ! (1)
                     idx = idx_table(b,c,a,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = t3_amps_buff(jdet)
                        m = t3_excits_buff(4,jdet); n = t3_excits_buff(5,jdet);
                        ! compute < ij~k~ab~c~ | mn~k~ab~c~ > -> delta(j,n) N[i+ m]
                        lt_amp = 0.5d0 * l_amp * t_amp
                        if (j==n) x1a_oo(i,m) = x1a_oo(i,m) + lt_amp ! (1)
                     end do
                     ! (jk)
                     idx = idx_table(b,c,a,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = t3_amps_buff(jdet)
                           m = t3_excits_buff(4,jdet); n = t3_excits_buff(5,jdet);
                           ! compute < ij~k~ab~c~ | mn~j~ab~c~ > -> -delta(k,n) N[i+ m]
                           lt_amp = 0.5d0 * l_amp * t_amp
                           if (k==n) x1a_oo(i,m) = x1a_oo(i,m) - lt_amp ! (1)
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,t3_amps_buff,t3_excits_buff)

              end subroutine compute_x1a_oo

              subroutine compute_x2a_oooo(x2a_oooo,&
                                          t3a_amps, t3a_excits,&
                                          t3b_amps, t3b_excits,&
                                          l3a_amps, l3a_excits,&
                                          l3b_amps, l3b_excits,&
                                          n3aaa_t, n3aab_t,&
                                          n3aaa_l, n3aab_l,&
                                          noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa_t, n3aab_t
                  integer, intent(in) :: n3aaa_l, n3aab_l

                  integer, intent(in) :: t3a_excits(6,n3aaa_t)
                  real(kind=8), intent(in) :: t3a_amps(n3aaa_t)
                  integer, intent(in) :: t3b_excits(6,n3aab_t)
                  real(kind=8), intent(in) :: t3b_amps(n3aab_t)
                  
                  integer, intent(in) :: l3a_excits(6,n3aaa_l)
                  real(kind=8), intent(in) :: l3a_amps(n3aaa_l)
                  integer, intent(in) :: l3b_excits(6,n3aab_l)
                  real(kind=8), intent(in) :: l3b_amps(n3aab_l)

                  real(kind=8), intent(out) :: x2a_oooo(noa,noa,noa,noa)
                  
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:) 
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)

                  real(kind=8) :: t_amp, l_amp, lt_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc

                  x2a_oooo = 0.0d0
                  !!!! x2a(ijmn) = 1/6 l3a(abcijk) t3a(abcmnk)
                  ! copy t3a into buffer
                  allocate(amps_buff(n3aaa_t),excits_buff(6,n3aaa_t))
                  amps_buff(:) = t3a_amps(:)
                  excits_buff(:,:) = t3a_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)*(nua-2)/6*noa
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nua,nua,nua,noa))
                  !!! ABCK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/3,noa/), nua, nua, nua, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,3,6/), nua, nua, nua, noa, nloc, n3aaa_t)
                  do idet = 1, n3aaa_l
                     l_amp = l3a_amps(idet)
                     a = l3a_excits(1,idet); b = l3a_excits(2,idet); c = l3a_excits(3,idet);
                     i = l3a_excits(4,idet); j = l3a_excits(5,idet); k = l3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        l = excits_buff(4,jdet); m = excits_buff(5,jdet);
                        ! < ijkabc | N[i1+ i2+ j2 j1] | lmkabc >
                        x2a_oooo(i,j,l,m) = x2a_oooo(i,j,l,m) + l_amp * t_amp
                     end do
                     ! (ik)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           l = excits_buff(4,jdet); m = excits_buff(5,jdet);
                           x2a_oooo(j,k,l,m) = x2a_oooo(j,k,l,m) + l_amp * t_amp ! flip sign to compute permutationally unique term
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           l = excits_buff(4,jdet); m = excits_buff(5,jdet);
                           x2a_oooo(i,k,l,m) = x2a_oooo(i,k,l,m) - l_amp * t_amp
                        end do
                     end if
                  end do
                  !!! ABCI LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/1,noa-2/), nua, nua, nua, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,3,4/), nua, nua, nua, noa, nloc, n3aaa_t)
                  do idet = 1, n3aaa_l
                     l_amp = l3a_amps(idet)
                     a = l3a_excits(1,idet); b = l3a_excits(2,idet); c = l3a_excits(3,idet);
                     i = l3a_excits(4,idet); j = l3a_excits(5,idet); k = l3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,i)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        m = excits_buff(5,jdet); n = excits_buff(6,jdet);
                        x2a_oooo(j,k,m,n) = x2a_oooo(j,k,m,n) + l_amp * t_amp
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           m = excits_buff(5,jdet); n = excits_buff(6,jdet);
                           x2a_oooo(i,k,m,n) = x2a_oooo(i,k,m,n) - l_amp * t_amp
                        end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           m = excits_buff(5,jdet); n = excits_buff(6,jdet);
                           x2a_oooo(i,j,m,n) = x2a_oooo(i,j,m,n) + l_amp * t_amp ! flip sign to compute permutationally unique term
                        end do
                     end if
                  end do
                  !!! ABCJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/2,noa-1/), nua, nua, nua, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,3,5/), nua, nua, nua, noa, nloc, n3aaa_t)
                  do idet = 1, n3aaa_l
                     l_amp = l3a_amps(idet)
                     a = l3a_excits(1,idet); b = l3a_excits(2,idet); c = l3a_excits(3,idet);
                     i = l3a_excits(4,idet); j = l3a_excits(5,idet); k = l3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,j)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        l = excits_buff(4,jdet); n = excits_buff(6,jdet);
                        x2a_oooo(i,k,l,n) = x2a_oooo(i,k,l,n) + l_amp * t_amp
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           l = excits_buff(4,jdet); n = excits_buff(6,jdet);
                           x2a_oooo(j,k,l,n) = x2a_oooo(j,k,l,n) - l_amp * t_amp
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           l = excits_buff(4,jdet); n = excits_buff(6,jdet);
                           x2a_oooo(i,j,l,n) = x2a_oooo(i,j,l,n) - l_amp * t_amp
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,excits_buff,amps_buff)
                  !!!! x2a(ijmn) = 1/6 l3b(abcijk) t3b(abcmnk)
                  ! copy t3b into buffer
                  allocate(amps_buff(n3aab_t),excits_buff(6,n3aab_t))
                  amps_buff(:) = t3b_amps(:)
                  excits_buff(:,:) = t3b_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2*nub*nob
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nua,nua,nub,nob))
                  !!! ABCK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/1,nob/), nua, nua, nub, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,3,6/), nua, nua, nub, nob, nloc, n3aab_t)
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        l = excits_buff(4,jdet); m = excits_buff(5,jdet);
                        ! < ijk~abc~ | lmk~abc~ >
                        x2a_oooo(i,j,l,m) = x2a_oooo(i,j,l,m) + l_amp * t_amp
                     end do
                  end do
                  deallocate(loc_arr,idx_table,excits_buff,amps_buff)
                  ! explicitly enforce antisymmetry 
                  ! To ensure this works, all computations to x2a_oooo(i,j,k,l) should be to
                  ! permutationally unique elements, meaning only for i<j and k<l
                  do i = 1, noa
                     do j = i+1, noa
                        do k = 1, noa
                           do l = k+1, noa
                              x2a_oooo(j,i,k,l) = -x2a_oooo(i,j,k,l)
                              x2a_oooo(i,j,l,k) = -x2a_oooo(i,j,k,l)
                              x2a_oooo(j,i,l,k) = x2a_oooo(i,j,k,l)
                           end do
                        end do     
                     end do        
                  end do

              end subroutine compute_x2a_oooo        

              subroutine compute_x2b_vvvv(x2b_vvvv,&
                                          t3b_amps, t3b_excits,&
                                          t3c_amps, t3c_excits,&
                                          l3b_amps, l3b_excits,&
                                          l3c_amps, l3c_excits,&
                                          n3aab_t, n3abb_t,&
                                          n3aab_l, n3abb_l,&
                                          noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab_t, n3abb_t
                  integer, intent(in) :: n3aab_l, n3abb_l

                  integer, intent(in) :: t3b_excits(6,n3aab_t)
                  real(kind=8), intent(in) :: t3b_amps(n3aab_t)
                  integer, intent(in) :: t3c_excits(6,n3abb_t)
                  real(kind=8), intent(in) :: t3c_amps(n3abb_t)
                  
                  integer, intent(in) :: l3b_excits(6,n3aab_l)
                  real(kind=8), intent(in) :: l3b_amps(n3aab_l)
                  integer, intent(in) :: l3c_excits(6,n3abb_l)
                  real(kind=8), intent(in) :: l3c_amps(n3abb_l)

                  real(kind=8), intent(out) :: x2b_vvvv(nua,nub,nua,nub)
                  
                  integer, allocatable :: t3_excits_buff(:,:)
                  real(kind=8), allocatable :: t3_amps_buff(:) 
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)

                  real(kind=8) :: t_amp, l_amp, lt_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc

                  x2b_vvvv = 0.0d0
                  !!!! x2b(abef) = 1/2 l3b(egfmon) t3b(agbmon) -> x2b(efbc) = 1/2 l3b(abcijk) t3b(aefijk)
                  ! copy t3b into buffer
                  allocate(t3_amps_buff(n3aab_t),t3_excits_buff(6,n3aab_t))
                  t3_amps_buff(:) = t3b_amps(:)
                  t3_excits_buff(:,:) = t3b_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = noa*(noa-1)/2*nob*nua
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(noa,noa,nob,nua))
                  !!! IJKA LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/1,nua-1/), noa, noa, nob, nua)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/4,5,6,1/), noa, noa, nob, nua, nloc, n3aab_t)
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,a)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = t3_amps_buff(jdet)
                        e = t3_excits_buff(2,jdet); f = t3_excits_buff(3,jdet);
                        ! compute < ijk~abc~ | ijk~aef~ >
                        x2b_vvvv(e,f,b,c) = x2b_vvvv(e,f,b,c) + l_amp * t_amp
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = t3_amps_buff(jdet)
                           e = t3_excits_buff(2,jdet); f = t3_excits_buff(3,jdet);
                           ! compute < ijk~abc~ | ijk~bef~ >
                           x2b_vvvv(e,f,a,c) = x2b_vvvv(e,f,a,c) - l_amp * t_amp
                        end do
                     end if
                  end do
                  !!! IJKB LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/2,nua/), noa, noa, nob, nua)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/4,5,6,2/), noa, noa, nob, nua, nloc, n3aab_t)
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     a = l3b_excits(1,idet); b = l3b_excits(2,idet); c = l3b_excits(3,idet);
                     i = l3b_excits(4,idet); j = l3b_excits(5,idet); k = l3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,b)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = t3_amps_buff(jdet)
                        e = t3_excits_buff(1,jdet); f = t3_excits_buff(3,jdet);
                        ! compute < ijk~abc~ | ijk~ebf~ >
                        x2b_vvvv(e,f,a,c) = x2b_vvvv(e,f,a,c) + l_amp * t_amp
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = t3_amps_buff(jdet)
                           e = t3_excits_buff(1,jdet); f = t3_excits_buff(3,jdet);
                           ! compute < ijk~abc~ | ijk~eaf~ >
                           x2b_vvvv(e,f,b,c) = x2b_vvvv(e,f,b,c) - l_amp * t_amp
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,t3_amps_buff,t3_excits_buff)
                  !!!! x2b(efab) <- 1/2 l3c(abcijk) t3c(efcijk)
                  ! copy t3c into buffer
                  allocate(t3_amps_buff(n3abb_t),t3_excits_buff(6,n3abb_t))
                  t3_amps_buff(:) = t3c_amps(:)
                  t3_excits_buff(:,:) = t3c_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = nob*(nob-1)/2*noa*nub
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nob,nob,noa,nub))
                  !!! JKIC LOOP !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,noa/), (/2,nub/), nob, nob, noa, nub)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/5,6,4,3/), nob, nob, noa, nub, nloc, n3abb_t)
                  do idet = 1, n3abb_l
                     l_amp = l3c_amps(idet)
                     a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                     i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                     ! (1)
                     idx = idx_table(j,k,i,c)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = t3_amps_buff(jdet)
                        e = t3_excits_buff(1,jdet); f = t3_excits_buff(2,jdet);
                        ! compute < ij~k~ab~c~ | ij~k~ef~c~ >
                        x2b_vvvv(e,f,a,b) = x2b_vvvv(e,f,a,b) + l_amp * t_amp
                     end do
                     ! (bc)
                     idx = idx_table(j,k,i,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = t3_amps_buff(jdet)
                           e = t3_excits_buff(1,jdet); f = t3_excits_buff(2,jdet);
                           ! compute < ij~k~ab~c~ | ij~k~ef~b~ >
                           x2b_vvvv(e,f,a,c) = x2b_vvvv(e,f,a,c) - l_amp * t_amp
                        end do
                     end if
                  end do
                  !!! JKIB LOOP !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,noa/), (/1,nub-1/), nob, nob, noa, nub)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/5,6,4,2/), nob, nob, noa, nub, nloc, n3abb_t)
                  do idet = 1, n3abb_l
                     l_amp = l3c_amps(idet)
                     a = l3c_excits(1,idet); b = l3c_excits(2,idet); c = l3c_excits(3,idet);
                     i = l3c_excits(4,idet); j = l3c_excits(5,idet); k = l3c_excits(6,idet);
                     ! (1)
                     idx = idx_table(j,k,i,b)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = t3_amps_buff(jdet)
                        e = t3_excits_buff(1,jdet); f = t3_excits_buff(3,jdet);
                        ! compute < ij~k~ab~c~ | ij~k~eb~f~ >
                        x2b_vvvv(e,f,a,c) = x2b_vvvv(e,f,a,c) + l_amp * t_amp
                     end do
                     ! (bc)
                     idx = idx_table(j,k,i,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = t3_amps_buff(jdet)
                           e = t3_excits_buff(1,jdet); f = t3_excits_buff(3,jdet);
                           ! compute < ij~k~ab~c~ | ij~k~ec~f~ >
                           x2b_vvvv(e,f,a,b) = x2b_vvvv(e,f,a,b) - l_amp * t_amp
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,t3_amps_buff,t3_excits_buff)

              end subroutine compute_x2b_vvvv

              subroutine compute_x2b_voov(x2b_voov,&
                                          t3a_amps, t3a_excits,&
                                          t3b_amps, t3b_excits,&
                                          t3c_amps, t3c_excits,&
                                          l3b_amps, l3b_excits,&
                                          l3c_amps, l3c_excits,&
                                          l3d_amps, l3d_excits,&
                                          n3aaa_t, n3aab_t, n3abb_t,&
                                          n3aab_l, n3abb_l, n3bbb_l,&
                                          noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa_t, n3aab_t, n3abb_t
                  integer, intent(in) :: n3aab_l, n3abb_l, n3bbb_l

                  integer, intent(in) :: t3a_excits(6,n3aaa_t)
                  real(kind=8), intent(in) :: t3a_amps(n3aaa_t)
                  integer, intent(in) :: t3b_excits(6,n3aab_t)
                  real(kind=8), intent(in) :: t3b_amps(n3aab_t)
                  integer, intent(in) :: t3c_excits(6,n3abb_t)
                  real(kind=8), intent(in) :: t3c_amps(n3abb_t)
                  
                  integer, intent(in) :: l3b_excits(6,n3aab_l)
                  real(kind=8), intent(in) :: l3b_amps(n3aab_l)
                  integer, intent(in) :: l3c_excits(6,n3abb_l)
                  real(kind=8), intent(in) :: l3c_amps(n3abb_l)
                  integer, intent(in) :: l3d_excits(6,n3bbb_l)
                  real(kind=8), intent(in) :: l3d_amps(n3bbb_l)

                  real(kind=8), intent(out) :: x2b_voov(nua,nob,noa,nub)
                  
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:) 
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)

                  real(kind=8) :: t_amp, l_amp, lt_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc

                  x2b_voov = 0.0d0
                  !!!! x2b(cmke) <- 1/4 l3b(abeijm) t3a(abcijk) 
                  ! copy l3b into buffer
                  allocate(amps_buff(n3aab_l),excits_buff(6,n3aab_l))
                  amps_buff(:) = l3b_amps(:)
                  excits_buff(:,:) = l3b_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = noa*(noa-1)/2*nua*(nua-1)/2
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(noa,noa,nua,nua))
                  !!! IJAB LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nua-1/), (/-1,nua/), noa, noa, nua, nua)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,1,2/), noa, noa, nua, nua, nloc, n3aab_l)
                  do idet = 1, n3aaa_t
                     t_amp = t3a_amps(idet)
                     a = t3a_excits(1,idet); b = t3a_excits(2,idet); c = t3a_excits(3,idet);
                     i = t3a_excits(4,idet); j = t3a_excits(5,idet); k = t3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,a,b)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        l_amp = amps_buff(jdet)
                        e = excits_buff(3,jdet); m = excits_buff(6,jdet);
                        ! compute < ijm~abe~ | ijkabc >
                        x2b_voov(c,m,k,e) = x2b_voov(c,m,k,e) + l_amp * t_amp
                     end do
                     ! (ac)
                     idx = idx_table(i,j,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(3,jdet); m = excits_buff(6,jdet);
                           ! compute < ijm~bce~ | ijkabc >
                           x2b_voov(a,m,k,e) = x2b_voov(a,m,k,e) + l_amp * t_amp
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(3,jdet); m = excits_buff(6,jdet);
                           ! compute < ijm~ace~ | ijkabc >
                           x2b_voov(b,m,k,e) = x2b_voov(b,m,k,e) - l_amp * t_amp
                        end do
                     end if
                     ! (ik)
                     idx = idx_table(j,k,a,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(3,jdet); m = excits_buff(6,jdet);
                           ! compute < jkm~abe~ | ijkabc >
                           x2b_voov(c,m,i,e) = x2b_voov(c,m,i,e) + l_amp * t_amp
                        end do
                     end if
                     ! (ik)(ac)
                     idx = idx_table(j,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(3,jdet); m = excits_buff(6,jdet);
                           ! compute < jkm~bce~ | ijkabc >
                           x2b_voov(a,m,i,e) = x2b_voov(a,m,i,e) + l_amp * t_amp
                        end do
                     end if
                     ! (ik)(bc)
                     idx = idx_table(j,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(3,jdet); m = excits_buff(6,jdet);
                           ! compute < jkm~ace~ | ijkabc >
                           x2b_voov(b,m,i,e) = x2b_voov(b,m,i,e) - l_amp * t_amp
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(i,k,a,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(3,jdet); m = excits_buff(6,jdet);
                           ! compute < ikm~abe~ | ijkabc >
                           x2b_voov(c,m,j,e) = x2b_voov(c,m,j,e) - l_amp * t_amp
                        end do
                     end if
                     ! (jk)(ac)
                     idx = idx_table(i,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(3,jdet); m = excits_buff(6,jdet);
                           ! compute < ikm~bce~ | ijkabc >
                           x2b_voov(a,m,j,e) = x2b_voov(a,m,j,e) - l_amp * t_amp
                        end do
                     end if
                     ! (jk)(bc)
                     idx = idx_table(i,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(3,jdet); m = excits_buff(6,jdet);
                           ! compute < ikm~ace~ | ijkabc >
                           x2b_voov(b,m,j,e) = x2b_voov(b,m,j,e) + l_amp * t_amp
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,amps_buff,excits_buff)
                  !!!! x2b(amie) <- l3c(bcejkm) t3b(abcijk) : This one's tricky. See diagram 14 in update_t3b_p in ccsdt_p for help. 
                  ! copy l3c into buffer
                  allocate(amps_buff(n3abb_l),excits_buff(6,n3abb_l))
                  amps_buff(:) = l3c_amps(:)
                  excits_buff(:,:) = l3c_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = noa*nob*nua*nub
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(noa,nob,nua,nub))
                  !!! IJAB LOOP !!!
                  call get_index_table(idx_table, (/1,noa/), (/1,nob-1/), (/1,nua/), (/1,nub-1/), noa, nob, nua, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,1,2/), noa, nob, nua, nub, nloc, n3abb_l)
                  do idet = 1, n3aab_t
                     t_amp = t3b_amps(idet)
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(j,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(3,jdet); m = excits_buff(6,jdet);
                           x2b_voov(a,m,i,e) = x2b_voov(a,m,i,e) + l_amp * t_amp
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(i,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(3,jdet); m = excits_buff(6,jdet);
                           x2b_voov(a,m,j,e) = x2b_voov(a,m,j,e) - l_amp * t_amp
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(j,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(3,jdet); m = excits_buff(6,jdet);
                           x2b_voov(b,m,i,e) = x2b_voov(b,m,i,e) - l_amp * t_amp
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(i,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(3,jdet); m = excits_buff(6,jdet);
                           x2b_voov(b,m,j,e) = x2b_voov(b,m,j,e) + l_amp * t_amp
                        end do
                     end if
                  end do
                  !!! IKAB LOOP !!!
                  call get_index_table(idx_table, (/1,noa/), (/2,nob/), (/1,nua/), (/1,nub-1/), noa, nob, nua, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,6,1,2/), noa, nob, nua, nub, nloc, n3abb_l)
                  do idet = 1, n3aab_t
                     t_amp = t3b_amps(idet)
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(j,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(3,jdet); m = excits_buff(5,jdet);
                           x2b_voov(a,m,i,e) = x2b_voov(a,m,i,e) - l_amp * t_amp
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(i,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(3,jdet); m = excits_buff(5,jdet);
                           x2b_voov(a,m,j,e) = x2b_voov(a,m,j,e) + l_amp * t_amp
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(j,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(3,jdet); m = excits_buff(5,jdet);
                           x2b_voov(b,m,i,e) = x2b_voov(b,m,i,e) + l_amp * t_amp
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(i,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(3,jdet); m = excits_buff(5,jdet);
                           x2b_voov(b,m,j,e) = x2b_voov(b,m,j,e) - l_amp * t_amp
                        end do
                     end if
                  end do
                  !!! IJAC LOOP !!!
                  call get_index_table(idx_table, (/1,noa/), (/1,nob-1/), (/1,nua/), (/2,nub/), noa, nob, nua, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,1,3/), noa, nob, nua, nub, nloc, n3abb_l)
                  do idet = 1, n3aab_t
                     t_amp = t3b_amps(idet)
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(j,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(2,jdet); m = excits_buff(6,jdet);
                           x2b_voov(a,m,i,e) = x2b_voov(a,m,i,e) - l_amp * t_amp
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(i,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(2,jdet); m = excits_buff(6,jdet);
                           x2b_voov(a,m,j,e) = x2b_voov(a,m,j,e) + l_amp * t_amp
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(j,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(2,jdet); m = excits_buff(6,jdet);
                           x2b_voov(b,m,i,e) = x2b_voov(b,m,i,e) + l_amp * t_amp
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(i,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(2,jdet); m = excits_buff(6,jdet);
                           x2b_voov(b,m,j,e) = x2b_voov(b,m,j,e) - l_amp * t_amp
                        end do
                     end if
                  end do
                  !!! IKAC LOOP !!!
                  call get_index_table(idx_table, (/1,noa/), (/2,nob/), (/1,nua/), (/2,nub/), noa, nob, nua, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,6,1,3/), noa, nob, nua, nub, nloc, n3abb_l)
                  do idet = 1, n3aab_t
                     t_amp = t3b_amps(idet)
                     a = t3b_excits(1,idet); b = t3b_excits(2,idet); c = t3b_excits(3,idet);
                     i = t3b_excits(4,idet); j = t3b_excits(5,idet); k = t3b_excits(6,idet);
                     ! (1)
                     idx = idx_table(j,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                           x2b_voov(a,m,i,e) = x2b_voov(a,m,i,e) + l_amp * t_amp
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(i,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                           x2b_voov(a,m,j,e) = x2b_voov(a,m,j,e) - l_amp * t_amp
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(j,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                           x2b_voov(b,m,i,e) = x2b_voov(b,m,i,e) - l_amp * t_amp
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(i,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(2,jdet); m = excits_buff(5,jdet);
                           x2b_voov(b,m,j,e) = x2b_voov(b,m,j,e) + l_amp * t_amp
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,excits_buff,amps_buff)
                  !!!! x2b(ekmc) <- 1/4 l3d(abcijk) t3c(eabmij) : a little tricky, see diagram 6 of update_t3d_p for help.
                  ! copy t3c into buffer
                  allocate(amps_buff(n3abb_t),excits_buff(6,n3abb_t))
                  amps_buff(:) = t3c_amps(:)
                  excits_buff(:,:) = t3c_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = nob*(nob-1)/2*nub*(nub-1)/2
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nob,nob,nub,nub))
                  !!! JKBC LOOP !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,nub-1/), (/-1,nub/), nob, nob, nub, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/5,6,2,3/), nob, nob, nub, nub, nloc, n3abb_t)
                  do idet = 1, n3bbb_l
                     l_amp = l3d_amps(idet)
                     a = l3d_excits(1,idet); b = l3d_excits(2,idet); c = l3d_excits(3,idet);
                     i = l3d_excits(4,idet); j = l3d_excits(5,idet); k = l3d_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,a,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(1,jdet); m = excits_buff(4,jdet);
                           x2b_voov(e,k,m,c) = x2b_voov(e,k,m,c) + l_amp * t_amp
                        end do
                     end if
                     ! (ik)
                     idx = idx_table(j,k,a,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(1,jdet); m = excits_buff(4,jdet);
                           x2b_voov(e,i,m,c) = x2b_voov(e,i,m,c) + l_amp * t_amp
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(i,k,a,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(1,jdet); m = excits_buff(4,jdet);
                           x2b_voov(e,j,m,c) = x2b_voov(e,j,m,c) - l_amp * t_amp
                        end do
                     end if
                     ! (ac)
                     idx = idx_table(i,j,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(1,jdet); m = excits_buff(4,jdet);
                           x2b_voov(e,k,m,a) = x2b_voov(e,k,m,a) + l_amp * t_amp
                        end do
                     end if
                     ! (ik)(ac)
                     idx = idx_table(j,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(1,jdet); m = excits_buff(4,jdet);
                           x2b_voov(e,i,m,a) = x2b_voov(e,i,m,a) + l_amp * t_amp
                        end do
                     end if
                     ! (jk)(ac)
                     idx = idx_table(i,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(1,jdet); m = excits_buff(4,jdet);
                           x2b_voov(e,j,m,a) = x2b_voov(e,j,m,a) - l_amp * t_amp
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(1,jdet); m = excits_buff(4,jdet);
                           x2b_voov(e,k,m,b) = x2b_voov(e,k,m,b) - l_amp * t_amp
                        end do
                     end if
                     ! (ik)(bc)
                     idx = idx_table(j,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(1,jdet); m = excits_buff(4,jdet);
                           x2b_voov(e,i,m,b) = x2b_voov(e,i,m,b) - l_amp * t_amp
                        end do
                     end if
                     ! (jk)(bc)
                     idx = idx_table(i,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(1,jdet); m = excits_buff(4,jdet);
                           x2b_voov(e,j,m,b) = x2b_voov(e,j,m,b) + l_amp * t_amp
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,amps_buff,excits_buff)

              end subroutine compute_x2b_voov

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

end module leftccsdt_p_intermediates
