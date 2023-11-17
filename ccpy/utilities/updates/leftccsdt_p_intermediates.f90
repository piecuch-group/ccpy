module leftccsdt_p_intermediates

        implicit none
        
        contains

              subroutine compute_x1a_vo(x1a_vo,&
                                        t3a_amps, t3a_excits,&
                                        t3b_amps, t3b_excits,&
                                        t3c_amps, t3c_excits,&
                                        l2a, l2b, l2c,&
                                        n3aaa_t, n3aab_t, n3abb_t,&
                                        noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa_t, n3aab_t, n3abb_t

                  integer, intent(in) :: t3a_excits(n3aaa_t,6)
                  real(kind=8), intent(in) :: t3a_amps(n3aaa_t)
                  integer, intent(in) :: t3b_excits(n3aab_t,6)
                  real(kind=8), intent(in) :: t3b_amps(n3aab_t)
                  integer, intent(in) :: t3c_excits(n3abb_t,6)
                  real(kind=8), intent(in) :: t3c_amps(n3abb_t)

                  real(kind=8), intent(in) :: l2a(nua,nua,noa,noa),&
                                              l2b(nua,nub,noa,nob),&
                                              l2c(nub,nub,nob,nob)

                  real(kind=8), intent(out) :: x1a_vo(nua,noa)

                  real(kind=8) :: t_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  x1a_vo = 0.0d0
                  do idet = 1, n3aaa_t
                     t_amp = t3a_amps(idet)
                     ! x1a(ai) <- A(a/ef)A(i/mn) l2a(efmn) * t3a(aefimn)
                     a = t3a_excits(idet,1); e = t3a_excits(idet,2); f = t3a_excits(idet,3);
                     i = t3a_excits(idet,4); m = t3a_excits(idet,5); n = t3a_excits(idet,6);
                     x1a_vo(a,i) = x1a_vo(a,i) + l2a(e,f,m,n) * t_amp ! (1)
                     x1a_vo(e,i) = x1a_vo(e,i) - l2a(a,f,m,n) * t_amp ! (ae)
                     x1a_vo(f,i) = x1a_vo(f,i) - l2a(e,a,m,n) * t_amp ! (af)
                     x1a_vo(a,m) = x1a_vo(a,m) - l2a(e,f,i,n) * t_amp ! (im)
                     x1a_vo(e,m) = x1a_vo(e,m) + l2a(a,f,i,n) * t_amp ! (ae)(im)
                     x1a_vo(f,m) = x1a_vo(f,m) + l2a(e,a,i,n) * t_amp ! (af)(im)
                     x1a_vo(a,n) = x1a_vo(a,n) - l2a(e,f,m,i) * t_amp ! (in)
                     x1a_vo(e,n) = x1a_vo(e,n) + l2a(a,f,m,i) * t_amp ! (ae)(in)
                     x1a_vo(f,n) = x1a_vo(f,n) + l2a(e,a,m,i) * t_amp ! (af)(in)
                  end do
                  do idet = 1, n3aab_t
                     t_amp = t3b_amps(idet)
                     ! x1a(ai) <- A(ae)A(im) l2b(efmn) * t3b(aefimn)
                     a = t3b_excits(idet,1); e = t3b_excits(idet,2); f = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); m = t3b_excits(idet,5); n = t3b_excits(idet,6);
                     x1a_vo(a,i) = x1a_vo(a,i) + l2b(e,f,m,n) * t_amp ! (1)
                     x1a_vo(e,i) = x1a_vo(e,i) - l2b(a,f,m,n) * t_amp ! (ae)
                     x1a_vo(a,m) = x1a_vo(a,m) - l2b(e,f,i,n) * t_amp ! (im)
                     x1a_vo(e,m) = x1a_vo(e,m) + l2b(a,f,i,n) * t_amp ! (ae)(im)
                  end do
                  do idet = 1, n3abb_t
                     t_amp = t3c_amps(idet)
                     ! x1a(ai) <- l2c(efmn) * t3c(aefimn)
                     a = t3c_excits(idet,1); e = t3c_excits(idet,2); f = t3c_excits(idet,3);
                     i = t3c_excits(idet,4); m = t3c_excits(idet,5); n = t3c_excits(idet,6);
                     x1a_vo(a,i) = x1a_vo(a,i) + l2c(e,f,m,n) * t_amp ! (1)
                  end do

              end subroutine compute_x1a_vo

              subroutine compute_x1a_oo(x1a_oo,&
                                        t3a_amps, t3a_excits,&
                                        t3b_amps, t3b_excits,&
                                        t3c_amps, t3c_excits,&
                                        l3a_amps, l3a_excits,&
                                        l3b_amps, l3b_excits,&
                                        l3c_amps, l3c_excits,&
                                        do_aaa_t, do_aab_t, do_abb_t,&
                                        n3aaa_t, n3aab_t, n3abb_t,&
                                        n3aaa_l, n3aab_l, n3abb_l,&
                                        noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa_t, n3aab_t, n3abb_t
                  integer, intent(in) :: n3aaa_l, n3aab_l, n3abb_l
                  logical, intent(in) :: do_aaa_t, do_aab_t, do_abb_t

                  integer, intent(in) :: t3a_excits(n3aaa_t,6)
                  real(kind=8), intent(in) :: t3a_amps(n3aaa_t)
                  integer, intent(in) :: t3b_excits(n3aab_t,6)
                  real(kind=8), intent(in) :: t3b_amps(n3aab_t)
                  integer, intent(in) :: t3c_excits(n3abb_t,6)
                  real(kind=8), intent(in) :: t3c_amps(n3abb_t)
                  integer, intent(in) :: l3a_excits(n3aaa_l,6)
                  real(kind=8), intent(in) :: l3a_amps(n3aaa_l)
                  integer, intent(in) :: l3b_excits(n3aab_l,6)
                  real(kind=8), intent(in) :: l3b_amps(n3aab_l)
                  integer, intent(in) :: l3c_excits(n3abb_l,6)
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
                  if (do_aaa_t) then
                  ! copy t3a into buffer
                  allocate(t3_amps_buff(n3aaa_t), t3_excits_buff(n3aaa_t,6))
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
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     i = l3a_excits(idet,4); j = l3a_excits(idet,5); k = l3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = t3_amps_buff(jdet)
                        m = t3_excits_buff(jdet,4); n = t3_excits_buff(jdet,5);
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
                          m = t3_excits_buff(jdet,4); n = t3_excits_buff(jdet,5);
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
                          m = t3_excits_buff(jdet,4); n = t3_excits_buff(jdet,5);
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
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     i = l3a_excits(idet,4); j = l3a_excits(idet,5); k = l3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,i)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = t3_amps_buff(jdet)
                        m = t3_excits_buff(jdet,5); n = t3_excits_buff(jdet,6);
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
                          m = t3_excits_buff(jdet,5); n = t3_excits_buff(jdet,6);
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
                          m = t3_excits_buff(jdet,5); n = t3_excits_buff(jdet,6);
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
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     i = l3a_excits(idet,4); j = l3a_excits(idet,5); k = l3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,j)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                       t_amp = t3_amps_buff(jdet)
                       m = t3_excits_buff(jdet,4); n = t3_excits_buff(jdet,6);
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
                          m = t3_excits_buff(jdet,4); n = t3_excits_buff(jdet,6);
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
                          m = t3_excits_buff(jdet,4); n = t3_excits_buff(jdet,6);
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
                  end if
                  !!!! X1A(mi) = 1/6 l3b(efgmno) t3b(efgino) -> X1A(im) = 1/6 l3b(abcijk) * t3b(abcmjk)
                  if (do_aab_t) then
                  ! copy t3b into buffer
                  allocate(t3_amps_buff(n3aab_t), t3_excits_buff(n3aab_t,6))
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
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     i = l3b_excits(idet,4); j = l3b_excits(idet,5); k = l3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = t3_amps_buff(jdet)
                        m = t3_excits_buff(jdet,4); n = t3_excits_buff(jdet,5);
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
                  end if
                  !!!! X1A(mi) = 1/6 l3c(efgmno) t3c(efgino) -> X1A(im) = 1/6 l3c(abcijk) * t3c(abcmjk)
                  if (do_abb_t) then
                  ! copy t3c into buffer
                  allocate(t3_amps_buff(n3abb_t), t3_excits_buff(n3abb_t,6))
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
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     i = l3c_excits(idet,4); j = l3c_excits(idet,5); k = l3c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,a,j)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = t3_amps_buff(jdet)
                        m = t3_excits_buff(jdet,4); n = t3_excits_buff(jdet,6);
                        ! compute < ij~k~ab~c~ | mj~n~ab~c~ > -> delta(k,n) N[i+ m]
                        lt_amp = 0.5d0 * l_amp * t_amp
                        if (k==n) x1a_oo(i,m) = x1a_oo(i,m) + lt_amp ! (1)
                     end do
                     ! (jk)
                     idx = idx_table(b,c,a,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = t3_amps_buff(jdet)
                           m = t3_excits_buff(jdet,4); n = t3_excits_buff(jdet,6);
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
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     i = l3c_excits(idet,4); j = l3c_excits(idet,5); k = l3c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,a,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = t3_amps_buff(jdet)
                        m = t3_excits_buff(jdet,4); n = t3_excits_buff(jdet,5);
                        ! compute < ij~k~ab~c~ | mn~k~ab~c~ > -> delta(j,n) N[i+ m]
                        lt_amp = 0.5d0 * l_amp * t_amp
                        if (j==n) x1a_oo(i,m) = x1a_oo(i,m) + lt_amp ! (1)
                     end do
                     ! (jk)
                     idx = idx_table(b,c,a,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = t3_amps_buff(jdet)
                           m = t3_excits_buff(jdet,4); n = t3_excits_buff(jdet,5);
                           ! compute < ij~k~ab~c~ | mn~j~ab~c~ > -> -delta(k,n) N[i+ m]
                           lt_amp = 0.5d0 * l_amp * t_amp
                           if (k==n) x1a_oo(i,m) = x1a_oo(i,m) - lt_amp ! (1)
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,t3_amps_buff,t3_excits_buff)
                  end if

              end subroutine compute_x1a_oo

              subroutine compute_x1a_vv(x1a_vv,&
                                        t3a_amps, t3a_excits,&
                                        t3b_amps, t3b_excits,&
                                        t3c_amps, t3c_excits,&
                                        l3a_amps, l3a_excits,&
                                        l3b_amps, l3b_excits,&
                                        l3c_amps, l3c_excits,&
                                        do_aaa_t, do_aab_t, do_abb_t,&
                                        n3aaa_t, n3aab_t, n3abb_t,&
                                        n3aaa_l, n3aab_l, n3abb_l,&
                                        noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa_t, n3aab_t, n3abb_t
                  integer, intent(in) :: n3aaa_l, n3aab_l, n3abb_l
                  logical, intent(in) :: do_aaa_t, do_aab_t, do_abb_t

                  integer, intent(in) :: t3a_excits(n3aaa_t,6)
                  real(kind=8), intent(in) :: t3a_amps(n3aaa_t)
                  integer, intent(in) :: t3b_excits(n3aab_t,6)
                  real(kind=8), intent(in) :: t3b_amps(n3aab_t)
                  integer, intent(in) :: t3c_excits(n3abb_t,6)
                  real(kind=8), intent(in) :: t3c_amps(n3abb_t)
                  integer, intent(in) :: l3a_excits(n3aaa_l,6)
                  real(kind=8), intent(in) :: l3a_amps(n3aaa_l)
                  integer, intent(in) :: l3b_excits(n3aab_l,6)
                  real(kind=8), intent(in) :: l3b_amps(n3aab_l)
                  integer, intent(in) :: l3c_excits(n3abb_l,6)
                  real(kind=8), intent(in) :: l3c_amps(n3abb_l)

                  real(kind=8), intent(out) :: x1a_vv(nua,nua)
                 
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:) 
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)

                  real(kind=8) :: t_amp, l_amp, lt_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc

                  x1a_vv = 0.0d0
                  !!!! x1a(ea) <- -1/6 l3a(abcijk) t3a(ebcijk)
                  if (do_aaa_t) then
                  ! copy t3a into buffer
                  allocate(amps_buff(n3aaa_t), excits_buff(n3aaa_t,6))
                  amps_buff(:) = t3a_amps(:)
                  excits_buff(:,:) = t3a_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = noa*(noa-1)*(noa-2)/6*nua
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(noa,noa,noa,nua))
                  !!! IJKC LOOP !!!
                  call get_index_table(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/3,nua/), noa, noa, noa, nua)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,3/), noa, noa, noa, nua, nloc, n3aaa_t)
                  do idet = 1, n3aaa_l
                     l_amp = l3a_amps(idet)
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     i = l3a_excits(idet,4); j = l3a_excits(idet,5); k = l3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,c)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); e = excits_buff(jdet,2);
                        ! < ijkabc | ijkdec >
                        lt_amp = 0.5d0 * l_amp * t_amp
                        if (a==d) x1a_vv(e,b) = x1a_vv(e,b) - lt_amp ! (1)
                        if (b==d) x1a_vv(e,a) = x1a_vv(e,a) + lt_amp ! (ab)
                        if (a==e) x1a_vv(d,b) = x1a_vv(d,b) + lt_amp ! (de)
                        if (b==e) x1a_vv(d,a) = x1a_vv(d,a) - lt_amp ! (ab)(de)
                     end do
                     ! (ac)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); e = excits_buff(jdet,2);
                           lt_amp = 0.5d0 * l_amp * t_amp
                           if (c==d) x1a_vv(e,b) = x1a_vv(e,b) + lt_amp ! (1)
                           if (b==d) x1a_vv(e,c) = x1a_vv(e,c) - lt_amp ! (ab)
                           if (c==e) x1a_vv(d,b) = x1a_vv(d,b) - lt_amp ! (de)
                           if (b==e) x1a_vv(d,c) = x1a_vv(d,c) + lt_amp ! (ab)(de)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); e = excits_buff(jdet,2);
                           lt_amp = 0.5d0 * l_amp * t_amp
                           if (a==d) x1a_vv(e,c) = x1a_vv(e,c) + lt_amp ! (1)
                           if (c==d) x1a_vv(e,a) = x1a_vv(e,a) - lt_amp ! (ab)
                           if (a==e) x1a_vv(d,c) = x1a_vv(d,c) - lt_amp ! (de)
                           if (c==e) x1a_vv(d,a) = x1a_vv(d,a) + lt_amp ! (ab)(de)
                        end do
                     end if
                  end do
                  !!! IJKB LOOP !!!
                  call get_index_table(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/2,nua-1/), noa, noa, noa, nua)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,2/), noa, noa, noa, nua, nloc, n3aaa_t)
                  do idet = 1, n3aaa_l
                     l_amp = l3a_amps(idet)
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     i = l3a_excits(idet,4); j = l3a_excits(idet,5); k = l3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,b)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); f = excits_buff(jdet,3);
                        lt_amp = 0.5d0 * l_amp * t_amp
                        if (a==d) x1a_vv(f,c) = x1a_vv(f,c) - lt_amp ! (1)
                        if (c==d) x1a_vv(f,a) = x1a_vv(f,a) + lt_amp ! (ac)
                        if (a==f) x1a_vv(d,c) = x1a_vv(d,c) + lt_amp ! (df)
                        if (c==f) x1a_vv(d,a) = x1a_vv(d,a) - lt_amp ! (ac)(df)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); f = excits_buff(jdet,3);
                           lt_amp = 0.5d0 * l_amp * t_amp
                           if (b==d) x1a_vv(f,c) = x1a_vv(f,c) + lt_amp ! (1)
                           if (c==d) x1a_vv(f,b) = x1a_vv(f,b) - lt_amp ! (ac)
                           if (b==f) x1a_vv(d,c) = x1a_vv(d,c) - lt_amp ! (df)
                           if (c==f) x1a_vv(d,b) = x1a_vv(d,b) + lt_amp ! (ac)(df)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); f = excits_buff(jdet,3);
                           lt_amp = 0.5d0 * l_amp * t_amp
                           if (a==d) x1a_vv(f,b) = x1a_vv(f,b) + lt_amp ! (1)
                           if (b==d) x1a_vv(f,a) = x1a_vv(f,a) - lt_amp ! (ac)
                           if (a==f) x1a_vv(d,b) = x1a_vv(d,b) - lt_amp ! (df)
                           if (b==f) x1a_vv(d,a) = x1a_vv(d,a) + lt_amp ! (ac)(df)
                        end do
                     end if
                  end do
                  !!! IJKA LOOP !!!
                  call get_index_table(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/1,nua-2/), noa, noa, noa, nua)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,1/), noa, noa, noa, nua, nloc, n3aaa_t)
                  do idet = 1, n3aaa_l
                     l_amp = l3a_amps(idet)
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     i = l3a_excits(idet,4); j = l3a_excits(idet,5); k = l3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,a)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); f = excits_buff(jdet,3);
                        lt_amp = 0.5d0 * l_amp * t_amp
                        if (b==e) x1a_vv(f,c) = x1a_vv(f,c) - lt_amp ! (1)
                        if (c==e) x1a_vv(f,b) = x1a_vv(f,b) + lt_amp ! (bc)
                        if (b==f) x1a_vv(e,c) = x1a_vv(e,c) + lt_amp ! (ef)
                        if (c==f) x1a_vv(e,b) = x1a_vv(e,b) - lt_amp ! (bc)(ef)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,2); f = excits_buff(jdet,3);
                           lt_amp = 0.5d0 * l_amp * t_amp
                           if (a==e) x1a_vv(f,c) = x1a_vv(f,c) + lt_amp ! (1)
                           if (c==e) x1a_vv(f,a) = x1a_vv(f,a) - lt_amp ! (bc)
                           if (a==f) x1a_vv(e,c) = x1a_vv(e,c) - lt_amp ! (ef)
                           if (c==f) x1a_vv(e,a) = x1a_vv(e,a) + lt_amp ! (bc)(ef)
                        end do
                     end if
                     ! (ac)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,2); f = excits_buff(jdet,3);
                           lt_amp = 0.5d0 * l_amp * t_amp
                           if (b==e) x1a_vv(f,a) = x1a_vv(f,a) + lt_amp ! (1)
                           if (a==e) x1a_vv(f,b) = x1a_vv(f,b) - lt_amp ! (bc)
                           if (b==f) x1a_vv(e,a) = x1a_vv(e,a) - lt_amp ! (ef)
                           if (a==f) x1a_vv(e,b) = x1a_vv(e,b) + lt_amp ! (bc)(ef)
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,amps_buff,excits_buff)
                  end if
                  !!!! x1a(ea) <- -l3b(abcijk) t3b(ebcijk)
                  if (do_aab_t) then
                  ! copy t3b into buffer
                  allocate(amps_buff(n3aab_t), excits_buff(n3aab_t,6))
                  amps_buff(:) = t3b_amps(:)
                  excits_buff(:,:) = t3b_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = noa*(noa-1)/2*nob*nub
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(noa,noa,nob,nub))
                  !!! IJKC LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/1,nub/), noa, noa, nob, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,3/), noa, noa, nob, nub, nloc, n3aab_t)
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     i = l3b_excits(idet,4); j = l3b_excits(idet,5); k = l3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,c)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); e = excits_buff(jdet,2);
                        ! < ijk~abc~ | ijk~dec~ >
                        lt_amp = l_amp * t_amp
                        if (a==d) x1a_vv(e,b) = x1a_vv(e,b) - lt_amp ! (1)
                        if (b==d) x1a_vv(e,a) = x1a_vv(e,a) + lt_amp ! (ab)
                        if (a==e) x1a_vv(d,b) = x1a_vv(d,b) + lt_amp ! (de)
                        if (b==e) x1a_vv(d,a) = x1a_vv(d,a) - lt_amp ! (ab)(de)
                     end do
                  end do
                  deallocate(loc_arr,idx_table,amps_buff,excits_buff)
                  end if
                  !!!! x1a(ea) <- -1/4 l3c(abcijk) t3c(ebcijk)
                  if (do_abb_t) then
                  ! copy t3c into buffer
                  allocate(amps_buff(n3abb_t), excits_buff(n3abb_t,6))
                  amps_buff(:) = t3c_amps(:)
                  excits_buff(:,:) = t3c_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = noa*nob*(nob-1)/2*nub
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nob,nob,noa,nub))
                  !!! JKIB LOOP !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,noa/), (/1,nub-1/), nob, nob, noa, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/5,6,4,2/), nob, nob, noa, nub, nloc, n3abb_t)
                  do idet = 1, n3abb_l
                     l_amp = l3c_amps(idet)
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     i = l3c_excits(idet,4); j = l3c_excits(idet,5); k = l3c_excits(idet,6);
                     ! (1)
                     idx = idx_table(j,k,i,b)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); f = excits_buff(jdet,3);
                        ! < ij~k~ab~c~ | ij~k~db~f~ >
                        lt_amp = 0.5d0 * l_amp * t_amp
                        if (c==f) x1a_vv(d,a) = x1a_vv(d,a) - lt_amp 
                     end do
                     ! (bc)
                     idx = idx_table(j,k,i,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); f = excits_buff(jdet,3);
                           lt_amp = 0.5d0 * l_amp * t_amp
                           if (b==f) x1a_vv(d,a) = x1a_vv(d,a) + lt_amp 
                        end do
                     end if
                  end do
                  !!! JKIC LOOP !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,noa/), (/2,nub/), nob, nob, noa, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/5,6,4,3/), nob, nob, noa, nub, nloc, n3abb_t)
                  do idet = 1, n3abb_l
                     l_amp = l3c_amps(idet)
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     i = l3c_excits(idet,4); j = l3c_excits(idet,5); k = l3c_excits(idet,6);
                     ! (1)
                     idx = idx_table(j,k,i,c)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); e = excits_buff(jdet,2);
                        lt_amp = 0.5d0 * l_amp * t_amp
                        if (b==e) x1a_vv(d,a) = x1a_vv(d,a) - lt_amp
                     end do
                     ! (bc)
                     idx = idx_table(j,k,i,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); e = excits_buff(jdet,2);
                           lt_amp = 0.5d0 * l_amp * t_amp
                           if (c==e) x1a_vv(d,a) = x1a_vv(d,a) + lt_amp 
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,amps_buff,excits_buff)
                  end if

              end subroutine compute_x1a_vv

              subroutine compute_x1b_vo(x1b_vo,&
                                        t3b_amps, t3b_excits,&
                                        t3c_amps, t3c_excits,&
                                        t3d_amps, t3d_excits,&
                                        l2a, l2b, l2c,&
                                        n3aab_t, n3abb_t, n3bbb_t,&
                                        noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab_t, n3abb_t, n3bbb_t

                  integer, intent(in) :: t3b_excits(n3aab_t,6)
                  real(kind=8), intent(in) :: t3b_amps(n3aab_t)
                  integer, intent(in) :: t3c_excits(n3abb_t,6)
                  real(kind=8), intent(in) :: t3c_amps(n3abb_t)
                  integer, intent(in) :: t3d_excits(n3bbb_t,6)
                  real(kind=8), intent(in) :: t3d_amps(n3bbb_t)

                  real(kind=8), intent(in) :: l2a(nua,nua,noa,noa),&
                                              l2b(nua,nub,noa,nob),&
                                              l2c(nub,nub,nob,nob)

                  real(kind=8), intent(out) :: x1b_vo(nub,nob)

                  real(kind=8) :: t_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  x1b_vo = 0.0d0
                  do idet = 1, n3bbb_t
                     t_amp = t3d_amps(idet)
                     ! x1b(ai) <- A(a/ef)A(i/mn) l2c(efmn) * t3d(aefimn)
                     a = t3d_excits(idet,1); e = t3d_excits(idet,2); f = t3d_excits(idet,3);
                     i = t3d_excits(idet,4); m = t3d_excits(idet,5); n = t3d_excits(idet,6);
                     x1b_vo(a,i) = x1b_vo(a,i) + l2c(e,f,m,n) * t_amp ! (1)
                     x1b_vo(e,i) = x1b_vo(e,i) - l2c(a,f,m,n) * t_amp ! (ae)
                     x1b_vo(f,i) = x1b_vo(f,i) - l2c(e,a,m,n) * t_amp ! (af)
                     x1b_vo(a,m) = x1b_vo(a,m) - l2c(e,f,i,n) * t_amp ! (im)
                     x1b_vo(e,m) = x1b_vo(e,m) + l2c(a,f,i,n) * t_amp ! (ae)(im)
                     x1b_vo(f,m) = x1b_vo(f,m) + l2c(e,a,i,n) * t_amp ! (af)(im)
                     x1b_vo(a,n) = x1b_vo(a,n) - l2c(e,f,m,i) * t_amp ! (in)
                     x1b_vo(e,n) = x1b_vo(e,n) + l2c(a,f,m,i) * t_amp ! (ae)(in)
                     x1b_vo(f,n) = x1b_vo(f,n) + l2c(e,a,m,i) * t_amp ! (af)(in)
                  end do
                  do idet = 1, n3aab_t
                     t_amp = t3b_amps(idet)
                     ! x1b(ai) <- l2a(efmn) * t3b(efamni)
                     e = t3b_excits(idet,1); f = t3b_excits(idet,2); a = t3b_excits(idet,3);
                     m = t3b_excits(idet,4); n = t3b_excits(idet,5); i = t3b_excits(idet,6);
                     x1b_vo(a,i) = x1b_vo(a,i) + l2a(e,f,m,n) * t_amp ! (1)
                  end do
                  do idet = 1, n3abb_t
                     t_amp = t3c_amps(idet)
                     ! x1b(ai) <- A(af)A(in) l2b(efmn) * t3c(efamni)
                     e = t3c_excits(idet,1); f = t3c_excits(idet,2); a = t3c_excits(idet,3);
                     m = t3c_excits(idet,4); n = t3c_excits(idet,5); i = t3c_excits(idet,6);
                     x1b_vo(a,i) = x1b_vo(a,i) + l2b(e,f,m,n) * t_amp ! (1)
                     x1b_vo(f,i) = x1b_vo(f,i) - l2b(e,a,m,n) * t_amp ! (af)
                     x1b_vo(a,n) = x1b_vo(a,n) - l2b(e,f,m,i) * t_amp ! (in)
                     x1b_vo(f,n) = x1b_vo(f,n) + l2b(e,a,m,i) * t_amp ! (af)(in)
                  end do

              end subroutine compute_x1b_vo

              subroutine compute_x1b_oo(x1b_oo,&
                                        t3b_amps, t3b_excits,&
                                        t3c_amps, t3c_excits,&
                                        t3d_amps, t3d_excits,&
                                        l3b_amps, l3b_excits,&
                                        l3c_amps, l3c_excits,&
                                        l3d_amps, l3d_excits,&
                                        do_aab_t, do_abb_t, do_bbb_t,&
                                        n3aab_t, n3abb_t, n3bbb_t,&
                                        n3aab_l, n3abb_l, n3bbb_l,&
                                        noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab_t, n3abb_t, n3bbb_t
                  integer, intent(in) :: n3aab_l, n3abb_l, n3bbb_l
                  logical, intent(in) :: do_aab_t, do_abb_t, do_bbb_t

                  integer, intent(in) :: t3b_excits(n3aab_t,6)
                  real(kind=8), intent(in) :: t3b_amps(n3aab_t)
                  integer, intent(in) :: t3c_excits(n3abb_t,6)
                  real(kind=8), intent(in) :: t3c_amps(n3abb_t)
                  integer, intent(in) :: t3d_excits(n3bbb_t,6)
                  real(kind=8), intent(in) :: t3d_amps(n3bbb_t)
                  integer, intent(in) :: l3b_excits(n3aab_l,6)
                  real(kind=8), intent(in) :: l3b_amps(n3aab_l)
                  integer, intent(in) :: l3c_excits(n3abb_l,6)
                  real(kind=8), intent(in) :: l3c_amps(n3abb_l)
                  integer, intent(in) :: l3d_excits(n3bbb_l,6)
                  real(kind=8), intent(in) :: l3d_amps(n3bbb_l)

                  real(kind=8), intent(out) :: x1b_oo(nob,nob)
                 
                  integer, allocatable :: t3_excits_buff(:,:)
                  real(kind=8), allocatable :: t3_amps_buff(:) 
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)

                  real(kind=8) :: t_amp, l_amp, lt_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc

                  x1b_oo = 0.0d0
                  !!!! X1A(mi) = 1/6 l3a(efgmno) t3a(efgino) -> X1A(im) = 1/6 l3a(abcijk) * t3a(abcmjk)
                  if (do_bbb_t) then
                  ! copy t3d into buffer
                  allocate(t3_amps_buff(n3bbb_t), t3_excits_buff(n3bbb_t,6))
                  t3_amps_buff(:) = t3d_amps(:)
                  t3_excits_buff(:,:) = t3d_excits(:,:)
                  ! allocate new sorting arrays
                  allocate(loc_arr(nub*(nub-1)*(nub-2)/6*nob,2))
                  allocate(idx_table(nub,nub,nub,nob))
                  !!! ABCK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/-1,nub/), (/3,nob/), nub, nub, nub, nob)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,2,3,6/), nub, nub, nub, nob, nub*(nub-1)*(nub-2)/6*nob, n3bbb_t)
                  do idet = 1, n3bbb_l
                     l_amp = l3d_amps(idet)
                     a = l3d_excits(idet,1); b = l3d_excits(idet,2); c = l3d_excits(idet,3);
                     i = l3d_excits(idet,4); j = l3d_excits(idet,5); k = l3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = t3_amps_buff(jdet)
                        m = t3_excits_buff(jdet,4); n = t3_excits_buff(jdet,5);
                        ! compute < ijkabc | mnkabc > ->
                        ! N[i+j+k+cba a+b+c+knm] = delta(j,n)N[i+m] + delta(i,m) N[j+n] - delta(i,n)N[j+m] - delta(j,m) N[i+n]
                        !                        = A(ij)A(nm) delta(i,m) N[j+n]
                        lt_amp = 0.5d0 * l_amp * t_amp
                        if (i==m) x1b_oo(j,n) = x1b_oo(j,n) + lt_amp ! (1)
                        if (j==m) x1b_oo(i,n) = x1b_oo(i,n) - lt_amp ! (ij)
                        if (i==n) x1b_oo(j,m) = x1b_oo(j,m) - lt_amp ! (nm)
                        if (j==n) x1b_oo(i,m) = x1b_oo(i,m) + lt_amp ! (ij)(nm)
                     end do
                     ! (ik)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                          t_amp = t3_amps_buff(jdet)
                          m = t3_excits_buff(jdet,4); n = t3_excits_buff(jdet,5);
                          ! compute < ijkabc | mniabc > ->
                          ! N[i+j+k+cba a+b+c+inm] = -A(jk)A(nm) delta(k,m) N[j+n]
                          lt_amp = 0.5d0 * l_amp * t_amp
                          if (k==m) x1b_oo(j,n) = x1b_oo(j,n) - lt_amp ! (1)
                          if (j==m) x1b_oo(k,n) = x1b_oo(k,n) + lt_amp ! (jk)
                          if (k==n) x1b_oo(j,m) = x1b_oo(j,m) + lt_amp ! (nm)
                          if (j==n) x1b_oo(k,m) = x1b_oo(k,m) - lt_amp ! (jk)(nm)
                       end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                          t_amp = t3_amps_buff(jdet)
                          m = t3_excits_buff(jdet,4); n = t3_excits_buff(jdet,5);
                          ! compute < ijkabc | mnjabc > ->
                          ! N[i+j+k+cba a+b+c+jnm] = -A(ik)A(nm) delta(i,m) N[k+n]
                          lt_amp = 0.5d0 * l_amp * t_amp
                          if (i==m) x1b_oo(k,n) = x1b_oo(k,n) - lt_amp ! (1)
                          if (k==m) x1b_oo(i,n) = x1b_oo(i,n) + lt_amp ! (ik)
                          if (i==n) x1b_oo(k,m) = x1b_oo(k,m) + lt_amp ! (nm)  
                          if (k==n) x1b_oo(i,m) = x1b_oo(i,m) - lt_amp ! (ik)(nm)
                       end do
                     end if
                  end do
                  !!! ABCI LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/-1,nub/), (/1,nob-2/), nub, nub, nub, nob)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,2,3,4/), nub, nub, nub, nob, nub*(nub-1)*(nub-2)/6*nob, n3bbb_t)
                  do idet = 1, n3bbb_l
                     l_amp = l3d_amps(idet)
                     a = l3d_excits(idet,1); b = l3d_excits(idet,2); c = l3d_excits(idet,3);
                     i = l3d_excits(idet,4); j = l3d_excits(idet,5); k = l3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,i)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = t3_amps_buff(jdet)
                        m = t3_excits_buff(jdet,5); n = t3_excits_buff(jdet,6);
                        ! compute < ijkabc | imnabc > -> A(jk)A(mn) delta(j,m) N[k+n]
                        lt_amp = 0.5d0 * l_amp * t_amp
                        if (j==m) x1b_oo(k,n) = x1b_oo(k,n) + lt_amp ! (1)
                        if (k==m) x1b_oo(j,n) = x1b_oo(j,n) - lt_amp ! (jk)
                        if (j==n) x1b_oo(k,m) = x1b_oo(k,m) - lt_amp ! (mn)
                        if (k==n) x1b_oo(j,m) = x1b_oo(j,m) + lt_amp ! (jk)(mn)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                          t_amp = t3_amps_buff(jdet)
                          m = t3_excits_buff(jdet,5); n = t3_excits_buff(jdet,6);
                          ! compute < ijkabc | jmnabc > -> -A(ik)A(mn) delta(i,m) N[k+n]
                          lt_amp = 0.5d0 * l_amp * t_amp
                          if (i==m) x1b_oo(k,n) = x1b_oo(k,n) - lt_amp ! (1)
                          if (k==m) x1b_oo(i,n) = x1b_oo(i,n) + lt_amp ! (ik)
                          if (i==n) x1b_oo(k,m) = x1b_oo(k,m) + lt_amp ! (mn)
                          if (k==n) x1b_oo(i,m) = x1b_oo(i,m) - lt_amp ! (ik)(mn)
                       end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                          t_amp = t3_amps_buff(jdet)
                          m = t3_excits_buff(jdet,5); n = t3_excits_buff(jdet,6);
                          ! compute < ijkabc | kmnabc > -> -A(ij)A(mn) delta(j,m) N[i+n]
                          lt_amp = 0.5d0 * l_amp * t_amp
                          if (j==m) x1b_oo(i,n) = x1b_oo(i,n) - lt_amp ! (1)
                          if (i==m) x1b_oo(j,n) = x1b_oo(j,n) + lt_amp ! (ij)
                          if (j==n) x1b_oo(i,m) = x1b_oo(i,m) + lt_amp ! (mn)
                          if (i==n) x1b_oo(j,m) = x1b_oo(j,m) - lt_amp ! (ij)(mn)
                        end do
                     end if
                  end do
                  !!! ABCJ LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/-1,nub/), (/2,nob-1/), nub, nub, nub, nob)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,2,3,5/), nub, nub, nub, nob, nub*(nub-1)*(nub-2)/6*nob, n3bbb_t)
                  do idet = 1, n3bbb_l
                     l_amp = l3d_amps(idet)
                     a = l3d_excits(idet,1); b = l3d_excits(idet,2); c = l3d_excits(idet,3);
                     i = l3d_excits(idet,4); j = l3d_excits(idet,5); k = l3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,j)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                       t_amp = t3_amps_buff(jdet)
                       m = t3_excits_buff(jdet,4); n = t3_excits_buff(jdet,6);
                       ! compute < ijkabc | mjnabc > -> A(ik)A(mn) delta(k,n) N[i+ m]
                       lt_amp = 0.5d0 * l_amp * t_amp
                       if (k==n) x1b_oo(i,m) = x1b_oo(i,m) + lt_amp ! (1)
                       if (i==n) x1b_oo(k,m) = x1b_oo(k,m) - lt_amp ! (ik)
                       if (k==m) x1b_oo(i,n) = x1b_oo(i,n) - lt_amp ! (mn)
                       if (i==m) x1b_oo(k,n) = x1b_oo(k,n) + lt_amp ! (ik)(mn)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                          t_amp = t3_amps_buff(jdet)
                          m = t3_excits_buff(jdet,4); n = t3_excits_buff(jdet,6);
                          ! compute < ijkabc | minabc > -> -A(jk)A(mn) delta(k,n) N[j+ m]
                          lt_amp = 0.5d0 * l_amp * t_amp
                          if (k==n) x1b_oo(j,m) = x1b_oo(j,m) - lt_amp ! (1)
                          if (j==n) x1b_oo(k,m) = x1b_oo(k,m) + lt_amp ! (jk)
                          if (k==m) x1b_oo(j,n) = x1b_oo(j,n) + lt_amp ! (mn)
                          if (j==m) x1b_oo(k,n) = x1b_oo(k,n) - lt_amp ! (jk)(mn)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                          t_amp = t3_amps_buff(jdet)
                          m = t3_excits_buff(jdet,4); n = t3_excits_buff(jdet,6);
                          ! compute < ijkabc | mknabc > -> -A(ij)A(mn) delta(j,n) N[i+ m]
                          lt_amp = 0.50 * l_amp * t_amp
                          if (j==n) x1b_oo(i,m) = x1b_oo(i,m) - lt_amp ! (1)
                          if (i==n) x1b_oo(j,m) = x1b_oo(j,m) + lt_amp ! (ij)
                          if (j==m) x1b_oo(i,n) = x1b_oo(i,n) + lt_amp ! (mn)
                          if (i==m) x1b_oo(j,n) = x1b_oo(j,n) - lt_amp ! (ij)(mn)
                        end do 
                     end if
                  end do
                  deallocate(loc_arr,idx_table,t3_amps_buff,t3_excits_buff)
                  end if
                  !!!! x1b(mi) = l3c(abcijk) * t3c(abcijm)
                  if (do_abb_t) then
                  ! copy t3c into buffer
                  allocate(t3_amps_buff(n3abb_t), t3_excits_buff(n3abb_t,6))
                  t3_amps_buff(:) = t3c_amps(:)
                  t3_excits_buff(:,:) = t3c_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = nua*nub*(nub-1)/2*noa
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nub,nub,nua,noa))
                  !!! BCAI LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nua/), (/1,noa/), nub, nub, nua, noa)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/2,3,1,4/), nub, nub, nua, noa, nloc, n3abb_t)
                  do idet = 1, n3abb_l
                     l_amp = l3c_amps(idet)
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     i = l3c_excits(idet,4); j = l3c_excits(idet,5); k = l3c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,a,i)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = t3_amps_buff(jdet)
                        m = t3_excits_buff(jdet,5); n = t3_excits_buff(jdet,6);
                        lt_amp = l_amp * t_amp
                        if (j==m) x1b_oo(k,n) = x1b_oo(k,n) + lt_amp ! (1)
                        if (j==n) x1b_oo(k,m) = x1b_oo(k,m) - lt_amp ! (mn)
                        if (k==m) x1b_oo(j,n) = x1b_oo(j,n) - lt_amp ! (jk)
                        if (k==n) x1b_oo(j,m) = x1b_oo(j,m) + lt_amp ! (mn)(jk)
                     end do
                  end do
                  deallocate(loc_arr,idx_table,t3_amps_buff,t3_excits_buff)
                  end if
                  !!!! x1b(mi) = l3b(abcijk) * t3b(abcijm)
                  if (do_aab_t) then
                  ! copy t3b into buffer
                  allocate(t3_amps_buff(n3aab_t), t3_excits_buff(n3aab_t,6))
                  t3_amps_buff(:) = t3b_amps(:)
                  t3_excits_buff(:,:) = t3b_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2*nub*noa
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nua,nua,nub,noa))
                  !!! ABCI LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/1,noa-1/), nua, nua, nub, noa)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,2,3,4/), nua, nua, nub, noa, nloc, n3aab_t)
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     i = l3b_excits(idet,4); j = l3b_excits(idet,5); k = l3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,i)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = t3_amps_buff(jdet)
                        m = t3_excits_buff(jdet,5); n = t3_excits_buff(jdet,6);
                        lt_amp = 0.5d0 * l_amp * t_amp
                        if (j==m) x1b_oo(k,n) = x1b_oo(k,n) + lt_amp ! (1)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = t3_amps_buff(jdet)
                           m = t3_excits_buff(jdet,5); n = t3_excits_buff(jdet,6);
                           lt_amp = 0.5d0 * l_amp * t_amp
                           if (i==m) x1b_oo(k,n) = x1b_oo(k,n) - lt_amp ! (1)
                        end do
                     end if
                  end do
                  !!! ABCJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/2,noa/), nua, nua, nub, noa)
                  call sort4(t3_excits_buff, t3_amps_buff, loc_arr, idx_table, (/1,2,3,5/), nua, nua, nub, noa, nloc, n3aab_t)
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     i = l3b_excits(idet,4); j = l3b_excits(idet,5); k = l3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,j)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = t3_amps_buff(jdet)
                        l = t3_excits_buff(jdet,4); n = t3_excits_buff(jdet,6);
                        lt_amp = 0.5d0 * l_amp * t_amp
                        if (i==l) x1b_oo(k,n) = x1b_oo(k,n) + lt_amp ! (1)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = t3_amps_buff(jdet)
                           l = t3_excits_buff(jdet,4); n = t3_excits_buff(jdet,6);
                           lt_amp = 0.5d0 * l_amp * t_amp
                           if (j==l) x1b_oo(k,n) = x1b_oo(k,n) - lt_amp ! (1)
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,t3_amps_buff,t3_excits_buff)
                  end if

              end subroutine compute_x1b_oo

              subroutine compute_x1b_vv(x1b_vv,&
                                        t3b_amps, t3b_excits,&
                                        t3c_amps, t3c_excits,&
                                        t3d_amps, t3d_excits,&
                                        l3b_amps, l3b_excits,&
                                        l3c_amps, l3c_excits,&
                                        l3d_amps, l3d_excits,&
                                        do_aab_t, do_abb_t, do_bbb_t,&
                                        n3aab_t, n3abb_t, n3bbb_t,&
                                        n3aab_l, n3abb_l, n3bbb_l,&
                                        noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab_t, n3abb_t, n3bbb_t
                  integer, intent(in) :: n3aab_l, n3abb_l, n3bbb_l
                  logical, intent(in) :: do_aab_t, do_abb_t, do_bbb_t

                  integer, intent(in) :: t3b_excits(n3aab_t,6)
                  real(kind=8), intent(in) :: t3b_amps(n3aab_t)
                  integer, intent(in) :: t3c_excits(n3abb_t,6)
                  real(kind=8), intent(in) :: t3c_amps(n3abb_t)
                  integer, intent(in) :: t3d_excits(n3bbb_t,6)
                  real(kind=8), intent(in) :: t3d_amps(n3bbb_t)
                  integer, intent(in) :: l3b_excits(n3aab_l,6)
                  real(kind=8), intent(in) :: l3b_amps(n3aab_l)
                  integer, intent(in) :: l3c_excits(n3abb_l,6)
                  real(kind=8), intent(in) :: l3c_amps(n3abb_l)
                  integer, intent(in) :: l3d_excits(n3bbb_l,6)
                  real(kind=8), intent(in) :: l3d_amps(n3bbb_l)

                  real(kind=8), intent(out) :: x1b_vv(nub,nub)
                 
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:) 
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)

                  real(kind=8) :: t_amp, l_amp, lt_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc

                  x1b_vv = 0.0d0
                  !!!! x1a(ea) <- -1/6 l3a(abcijk) t3a(ebcijk)
                  if (do_bbb_t) then
                  ! copy t3d into buffer
                  allocate(amps_buff(n3bbb_t), excits_buff(n3bbb_t,6))
                  amps_buff(:) = t3d_amps(:)
                  excits_buff(:,:) = t3d_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = nob*(nob-1)*(nob-2)/6*nub
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nob,nob,nob,nub))
                  !!! IJKC LOOP !!!
                  call get_index_table(idx_table, (/1,nob-2/), (/-1,nob-1/), (/-1,nob/), (/3,nub/), nob, nob, nob, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,3/), nob, nob, nob, nub, nloc, n3bbb_t)
                  do idet = 1, n3bbb_l
                     l_amp = l3d_amps(idet)
                     a = l3d_excits(idet,1); b = l3d_excits(idet,2); c = l3d_excits(idet,3);
                     i = l3d_excits(idet,4); j = l3d_excits(idet,5); k = l3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,c)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); e = excits_buff(jdet,2);
                        ! < ijkabc | ijkdec >
                        lt_amp = 0.5d0 * l_amp * t_amp
                        if (a==d) x1b_vv(e,b) = x1b_vv(e,b) - lt_amp ! (1)
                        if (b==d) x1b_vv(e,a) = x1b_vv(e,a) + lt_amp ! (ab)
                        if (a==e) x1b_vv(d,b) = x1b_vv(d,b) + lt_amp ! (de)
                        if (b==e) x1b_vv(d,a) = x1b_vv(d,a) - lt_amp ! (ab)(de)
                     end do
                     ! (ac)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); e = excits_buff(jdet,2);
                           lt_amp = 0.5d0 * l_amp * t_amp
                           if (c==d) x1b_vv(e,b) = x1b_vv(e,b) + lt_amp ! (1)
                           if (b==d) x1b_vv(e,c) = x1b_vv(e,c) - lt_amp ! (ab)
                           if (c==e) x1b_vv(d,b) = x1b_vv(d,b) - lt_amp ! (de)
                           if (b==e) x1b_vv(d,c) = x1b_vv(d,c) + lt_amp ! (ab)(de)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); e = excits_buff(jdet,2);
                           lt_amp = 0.5d0 * l_amp * t_amp
                           if (a==d) x1b_vv(e,c) = x1b_vv(e,c) + lt_amp ! (1)
                           if (c==d) x1b_vv(e,a) = x1b_vv(e,a) - lt_amp ! (ab)
                           if (a==e) x1b_vv(d,c) = x1b_vv(d,c) - lt_amp ! (de)
                           if (c==e) x1b_vv(d,a) = x1b_vv(d,a) + lt_amp ! (ab)(de)
                        end do
                     end if
                  end do
                  !!! IJKB LOOP !!!
                  call get_index_table(idx_table, (/1,nob-2/), (/-1,nob-1/), (/-1,nob/), (/2,nub-1/), nob, nob, nob, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,2/), nob, nob, nob, nub, nloc, n3bbb_t)
                  do idet = 1, n3bbb_l
                     l_amp = l3d_amps(idet)
                     a = l3d_excits(idet,1); b = l3d_excits(idet,2); c = l3d_excits(idet,3);
                     i = l3d_excits(idet,4); j = l3d_excits(idet,5); k = l3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,b)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); f = excits_buff(jdet,3);
                        lt_amp = 0.5d0 * l_amp * t_amp
                        if (a==d) x1b_vv(f,c) = x1b_vv(f,c) - lt_amp ! (1)
                        if (c==d) x1b_vv(f,a) = x1b_vv(f,a) + lt_amp ! (ac)
                        if (a==f) x1b_vv(d,c) = x1b_vv(d,c) + lt_amp ! (df)
                        if (c==f) x1b_vv(d,a) = x1b_vv(d,a) - lt_amp ! (ac)(df)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); f = excits_buff(jdet,3);
                           lt_amp = 0.5d0 * l_amp * t_amp
                           if (b==d) x1b_vv(f,c) = x1b_vv(f,c) + lt_amp ! (1)
                           if (c==d) x1b_vv(f,b) = x1b_vv(f,b) - lt_amp ! (ac)
                           if (b==f) x1b_vv(d,c) = x1b_vv(d,c) - lt_amp ! (df)
                           if (c==f) x1b_vv(d,b) = x1b_vv(d,b) + lt_amp ! (ac)(df)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); f = excits_buff(jdet,3);
                           lt_amp = 0.5d0 * l_amp * t_amp
                           if (a==d) x1b_vv(f,b) = x1b_vv(f,b) + lt_amp ! (1)
                           if (b==d) x1b_vv(f,a) = x1b_vv(f,a) - lt_amp ! (ac)
                           if (a==f) x1b_vv(d,b) = x1b_vv(d,b) - lt_amp ! (df)
                           if (b==f) x1b_vv(d,a) = x1b_vv(d,a) + lt_amp ! (ac)(df)
                        end do
                     end if
                  end do
                  !!! IJKA LOOP !!!
                  call get_index_table(idx_table, (/1,nob-2/), (/-1,nob-1/), (/-1,nob/), (/1,nub-2/), nob, nob, nob, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,1/), nob, nob, nob, nub, nloc, n3bbb_t)
                  do idet = 1, n3bbb_l
                     l_amp = l3d_amps(idet)
                     a = l3d_excits(idet,1); b = l3d_excits(idet,2); c = l3d_excits(idet,3);
                     i = l3d_excits(idet,4); j = l3d_excits(idet,5); k = l3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,a)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); f = excits_buff(jdet,3);
                        lt_amp = 0.5d0 * l_amp * t_amp
                        if (b==e) x1b_vv(f,c) = x1b_vv(f,c) - lt_amp ! (1)
                        if (c==e) x1b_vv(f,b) = x1b_vv(f,b) + lt_amp ! (bc)
                        if (b==f) x1b_vv(e,c) = x1b_vv(e,c) + lt_amp ! (ef)
                        if (c==f) x1b_vv(e,b) = x1b_vv(e,b) - lt_amp ! (bc)(ef)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,2); f = excits_buff(jdet,3);
                           lt_amp = 0.5d0 * l_amp * t_amp
                           if (a==e) x1b_vv(f,c) = x1b_vv(f,c) + lt_amp ! (1)
                           if (c==e) x1b_vv(f,a) = x1b_vv(f,a) - lt_amp ! (bc)
                           if (a==f) x1b_vv(e,c) = x1b_vv(e,c) - lt_amp ! (ef)
                           if (c==f) x1b_vv(e,a) = x1b_vv(e,a) + lt_amp ! (bc)(ef)
                        end do
                     end if
                     ! (ac)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,2); f = excits_buff(jdet,3);
                           lt_amp = 0.5d0 * l_amp * t_amp
                           if (b==e) x1b_vv(f,a) = x1b_vv(f,a) + lt_amp ! (1)
                           if (a==e) x1b_vv(f,b) = x1b_vv(f,b) - lt_amp ! (bc)
                           if (b==f) x1b_vv(e,a) = x1b_vv(e,a) - lt_amp ! (ef)
                           if (a==f) x1b_vv(e,b) = x1b_vv(e,b) + lt_amp ! (bc)(ef)
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,amps_buff,excits_buff)
                  end if
                  !!!! x1b(ea) <- -1/6 l3b(abcijk) t3b(abeijk)
                  if (do_aab_t) then
                  ! copy t3b into buffer
                  allocate(amps_buff(n3aab_t), excits_buff(n3aab_t,6))
                  amps_buff(:) = t3b_amps(:)
                  excits_buff(:,:) = t3b_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = noa*(noa-1)/2*nob*nua
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(noa,noa,nob,nua))
                  !!! IJKA LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/1,nua-1/), noa, noa, nob, nua)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,1/), noa, noa, nob, nua, nloc, n3aab_t)
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     i = l3b_excits(idet,4); j = l3b_excits(idet,5); k = l3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,a)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); f = excits_buff(jdet,3);
                        lt_amp = 0.5d0 * l_amp * t_amp
                        if (e==b) x1b_vv(f,c) = x1b_vv(f,c) - lt_amp ! (1)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,2); f = excits_buff(jdet,3);
                           lt_amp = 0.5d0 * l_amp * t_amp
                           if (e==a) x1b_vv(f,c) = x1b_vv(f,c) + lt_amp ! (1)
                        end do
                     end if
                  end do
                  !!! IJKB LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/2,nua/), noa, noa, nob, nua)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,2/), noa, noa, nob, nua, nloc, n3aab_t)
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     i = l3b_excits(idet,4); j = l3b_excits(idet,5); k = l3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,b)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); f = excits_buff(jdet,3);
                        lt_amp = 0.5d0 * l_amp * t_amp
                        if (d==a) x1b_vv(f,c) = x1b_vv(f,c) - lt_amp ! (1)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); f = excits_buff(jdet,3);
                           lt_amp = 0.5d0 * l_amp * t_amp
                           if (d==b) x1b_vv(f,c) = x1b_vv(f,c) + lt_amp ! (1)
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,amps_buff,excits_buff)
                  end if
                  !!!! x1b(ea) <- -l3c(abcijk) t3c(abeijk)
                  if (do_abb_t) then
                  ! copy t3c into buffer
                  allocate(amps_buff(n3abb_t), excits_buff(n3abb_t,6))
                  amps_buff(:) = t3c_amps(:)
                  excits_buff(:,:) = t3c_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = nob*(nob-1)/2*noa*nua
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nob,nob,noa,nua))
                  !!! JKIA LOOP !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,noa/), (/1,nua/), nob, nob, noa, nua)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/5,6,4,1/), nob, nob, noa, nua, nloc, n3abb_t)
                  do idet = 1, n3abb_l
                     l_amp = l3c_amps(idet)
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     i = l3c_excits(idet,4); j = l3c_excits(idet,5); k = l3c_excits(idet,6);
                     ! (1)
                     idx = idx_table(j,k,i,a)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); f = excits_buff(jdet,3);
                        lt_amp = l_amp * t_amp
                        if (e==b) x1b_vv(f,c) = x1b_vv(f,c) - lt_amp ! (1)
                        if (f==b) x1b_vv(e,c) = x1b_vv(e,c) + lt_amp ! (ef)
                        if (e==c) x1b_vv(f,b) = x1b_vv(f,b) + lt_amp ! (bc)
                        if (f==c) x1b_vv(e,b) = x1b_vv(e,b) - lt_amp ! (ef)(bc)
                     end do
                  end do
                  deallocate(loc_arr,idx_table,amps_buff,excits_buff)
                  end if

              end subroutine compute_x1b_vv

              subroutine compute_x2a_ooov(x2a_ooov,&
                                          t2a, t2b,&
                                          l3a_amps, l3a_excits,&
                                          l3b_amps, l3b_excits,&
                                          n3aaa_l, n3aab_l,&
                                          noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa_l, n3aab_l

                  real(kind=8), intent(in) :: t2a(nua,nua,noa,noa), t2b(nua,nub,noa,nob)
                  integer, intent(in) :: l3a_excits(n3aaa_l,6)
                  real(kind=8), intent(in) :: l3a_amps(n3aaa_l)
                  integer, intent(in) :: l3b_excits(n3aab_l,6)
                  real(kind=8), intent(in) :: l3b_amps(n3aab_l)

                  real(kind=8), intent(out) :: x2a_ooov(noa,noa,noa,nua)
                 
                  real(kind=8) :: l_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  x2a_ooov = 0.0d0
                  do idet = 1, n3aaa_l
                     l_amp = l3a_amps(idet)
                     ! x2a(jima) <- A(ij) [A(n/ij)A(a/ef) l3a(aefijn) * t2a(efmn)]
                     a = l3a_excits(idet,1); e = l3a_excits(idet,2); f = l3a_excits(idet,3);
                     i = l3a_excits(idet,4); j = l3a_excits(idet,5); n = l3a_excits(idet,6);
                     ! only fill permutationally unique elements!
                     x2a_ooov(i,j,:,a) = x2a_ooov(i,j,:,a) - l_amp * t2a(e,f,:,n) ! (1)
                     x2a_ooov(i,j,:,e) = x2a_ooov(i,j,:,e) + l_amp * t2a(a,f,:,n) ! (ae)
                     x2a_ooov(i,j,:,f) = x2a_ooov(i,j,:,f) + l_amp * t2a(e,a,:,n) ! (af)
                     x2a_ooov(j,n,:,a) = x2a_ooov(j,n,:,a) - l_amp * t2a(e,f,:,i) ! (in)
                     x2a_ooov(j,n,:,e) = x2a_ooov(j,n,:,e) + l_amp * t2a(a,f,:,i) ! (ae)(in)
                     x2a_ooov(j,n,:,f) = x2a_ooov(j,n,:,f) + l_amp * t2a(e,a,:,i) ! (af)(in)
                     x2a_ooov(i,n,:,a) = x2a_ooov(i,n,:,a) + l_amp * t2a(e,f,:,j) ! (jn)
                     x2a_ooov(i,n,:,e) = x2a_ooov(i,n,:,e) - l_amp * t2a(a,f,:,j) ! (ae)(jn)
                     x2a_ooov(i,n,:,f) = x2a_ooov(i,n,:,f) - l_amp * t2a(e,a,:,j) ! (af)(jn)
                  end do
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     ! x2a(jima) <- A(ij) [A(ae) l3b(aefijn) * t2b(efmn)]
                     a = l3b_excits(idet,1); e = l3b_excits(idet,2); f = l3b_excits(idet,3);
                     i = l3b_excits(idet,4); j = l3b_excits(idet,5); n = l3b_excits(idet,6);
                     ! only fill permutationally unique elements!
                     x2a_ooov(i,j,:,a) = x2a_ooov(i,j,:,a) - l_amp * t2b(e,f,:,n) ! (1)
                     x2a_ooov(i,j,:,e) = x2a_ooov(i,j,:,e) + l_amp * t2b(a,f,:,n) ! (ae)
                  end do

                  ! apply the common A(ij) antisymmetrizer
                  do i = 1, noa
                     do j = i+1, noa
                        do m = 1, noa
                           do a = 1, nua
                              x2a_ooov(i,j,m,a) = x2a_ooov(i,j,m,a) - x2a_ooov(j,i,m,a)
                           end do
                        end do
                     end do
                  end do
                  ! explicitly antisymmetrize
                  do i = 1, noa
                     do j = i+1, noa
                        x2a_ooov(j,i,:,:) = -x2a_ooov(i,j,:,:)
                     end do
                  end do

              end subroutine compute_x2a_ooov   

              subroutine compute_x2a_vovv(x2a_vovv,&
                                          t2a, t2b,&
                                          l3a_amps, l3a_excits,&
                                          l3b_amps, l3b_excits,&
                                          n3aaa_l, n3aab_l,&
                                          noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa_l, n3aab_l

                  real(kind=8), intent(in) :: t2a(nua,nua,noa,noa), t2b(nua,nub,noa,nob)
                  integer, intent(in) :: l3a_excits(n3aaa_l,6)
                  real(kind=8), intent(in) :: l3a_amps(n3aaa_l)
                  integer, intent(in) :: l3b_excits(n3aab_l,6)
                  real(kind=8), intent(in) :: l3b_amps(n3aab_l)

                  real(kind=8), intent(out) :: x2a_vovv(nua,noa,nua,nua)
                 
                  real(kind=8) :: l_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  x2a_vovv = 0.0d0
                  do idet = 1, n3aaa_l
                     l_amp = l3a_amps(idet)
                     ! x2a(eiba) <- A(ab) [A(i/mn)A(f/ab) l3a(abfimn) * t2a(efmn)]
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); f = l3a_excits(idet,3);
                     i = l3a_excits(idet,4); m = l3a_excits(idet,5); n = l3a_excits(idet,6);
                     ! only fill permutationally unique elements!
                     x2a_vovv(:,i,a,b) = x2a_vovv(:,i,a,b) + l_amp * t2a(:,f,m,n) ! (1)
                     x2a_vovv(:,m,a,b) = x2a_vovv(:,m,a,b) - l_amp * t2a(:,f,i,n) ! (im)
                     x2a_vovv(:,n,a,b) = x2a_vovv(:,n,a,b) - l_amp * t2a(:,f,m,i) ! (in)
                     x2a_vovv(:,i,b,f) = x2a_vovv(:,i,b,f) + l_amp * t2a(:,a,m,n) ! (af)
                     x2a_vovv(:,m,b,f) = x2a_vovv(:,m,b,f) - l_amp * t2a(:,a,i,n) ! (im)(af)
                     x2a_vovv(:,n,b,f) = x2a_vovv(:,n,b,f) - l_amp * t2a(:,a,m,i) ! (in)(af)
                     x2a_vovv(:,i,a,f) = x2a_vovv(:,i,a,f) - l_amp * t2a(:,b,m,n) ! (bf)
                     x2a_vovv(:,m,a,f) = x2a_vovv(:,m,a,f) + l_amp * t2a(:,b,i,n) ! (im)(bf)
                     x2a_vovv(:,n,a,f) = x2a_vovv(:,n,a,f) + l_amp * t2a(:,b,m,i) ! (in)(bf)
                  end do
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     ! x2a(eiba) <- A(ab) [A(im) l3b(abfimn) * t2b(efmn)]
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); f = l3b_excits(idet,3);
                     i = l3b_excits(idet,4); m = l3b_excits(idet,5); n = l3b_excits(idet,6);
                     ! only fill permutationally unique elements!
                     x2a_vovv(:,i,a,b) = x2a_vovv(:,i,a,b) + l_amp * t2b(:,f,m,n) ! (1)
                     x2a_vovv(:,m,a,b) = x2a_vovv(:,m,a,b) - l_amp * t2b(:,f,i,n) ! (im)
                  end do

                  ! apply the common A(ij) antisymmetrizer
                  do a = 1, nua
                     do b = a+1, nua
                        do i = 1, noa
                           do e = 1, nua
                              x2a_vovv(e,i,a,b) = x2a_vovv(e,i,a,b) - x2a_vovv(e,i,b,a)
                           end do
                        end do
                     end do
                  end do
                  ! explicitly antisymmetrize
                  do a = 1, nua
                     do b = a+1, nua
                        x2a_vovv(:,:,b,a) = -x2a_vovv(:,:,a,b)
                     end do
                  end do

              end subroutine compute_x2a_vovv   

              subroutine compute_x2b_oovo(x2b_oovo,&
                                          t2b, t2c,&
                                          l3b_amps, l3b_excits,&
                                          l3c_amps, l3c_excits,&
                                          n3aab_l, n3abb_l,&
                                          noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab_l, n3abb_l

                  real(kind=8), intent(in) :: t2b(nua,nub,noa,nob), t2c(nub,nub,nob,nob)
                  integer, intent(in) :: l3b_excits(n3aab_l,6)
                  real(kind=8), intent(in) :: l3b_amps(n3aab_l)
                  integer, intent(in) :: l3c_excits(n3abb_l,6)
                  real(kind=8), intent(in) :: l3c_amps(n3abb_l)

                  real(kind=8), intent(out) :: x2b_oovo(noa,nob,nua,nob)
                 
                  real(kind=8) :: l_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  x2b_oovo = 0.0d0
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     ! x2b(jkbm) <- A(bf)A(jn) l3b(bfejnk) * t2b(fenm)
                     b = l3b_excits(idet,1); f = l3b_excits(idet,2); e = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); n = l3b_excits(idet,5); k = l3b_excits(idet,6);
                     x2b_oovo(j,k,b,:) = x2b_oovo(j,k,b,:) + l_amp * t2b(f,e,n,:) ! (1)
                     x2b_oovo(j,k,f,:) = x2b_oovo(j,k,f,:) - l_amp * t2b(b,e,n,:) ! (bf)
                     x2b_oovo(n,k,b,:) = x2b_oovo(n,k,b,:) - l_amp * t2b(f,e,j,:) ! (jn)
                     x2b_oovo(n,k,f,:) = x2b_oovo(n,k,f,:) + l_amp * t2b(b,e,j,:) ! (bf)(jn)
                  end do
                  do idet = 1, n3abb_l
                     l_amp = l3c_amps(idet)
                     ! x2b(jkbm) <- A(nk) l3c(bfejnk) * t2c(fenm)
                     b = l3c_excits(idet,1); f = l3c_excits(idet,2); e = l3c_excits(idet,3);
                     j = l3c_excits(idet,4); n = l3c_excits(idet,5); k = l3c_excits(idet,6);
                     x2b_oovo(j,k,b,:) = x2b_oovo(j,k,b,:) + l_amp * t2c(f,e,n,:) ! (1)
                     x2b_oovo(j,n,b,:) = x2b_oovo(j,n,b,:) - l_amp * t2c(f,e,k,:) ! (kn)
                  end do

              end subroutine compute_x2b_oovo

              subroutine compute_x2b_ooov(x2b_ooov,&
                                          t2a, t2b,&
                                          l3b_amps, l3b_excits,&
                                          l3c_amps, l3c_excits,&
                                          n3aab_l, n3abb_l,&
                                          noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab_l, n3abb_l

                  real(kind=8), intent(in) :: t2a(nua,nua,noa,noa), t2b(nua,nub,noa,nob)
                  integer, intent(in) :: l3b_excits(n3aab_l,6)
                  real(kind=8), intent(in) :: l3b_amps(n3aab_l)
                  integer, intent(in) :: l3c_excits(n3abb_l,6)
                  real(kind=8), intent(in) :: l3c_amps(n3abb_l)

                  real(kind=8), intent(out) :: x2b_ooov(noa,nob,noa,nub)
                 
                  real(kind=8) :: l_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  x2b_ooov = 0.0d0
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     ! x2b(jkmc) <- A(jn) l3b(efcjnk) * t2a(efmn)
                     e = l3b_excits(idet,1); f = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); n = l3b_excits(idet,5); k = l3b_excits(idet,6);
                     x2b_ooov(j,k,:,c) = x2b_ooov(j,k,:,c) + l_amp * t2a(e,f,:,n) ! (1)
                     x2b_ooov(n,k,:,c) = x2b_ooov(n,k,:,c) - l_amp * t2a(e,f,:,j) ! (jn)
                  end do
                  do idet = 1, n3abb_l
                     l_amp = l3c_amps(idet)
                     ! x2b(jkmc) <-  A(fc)A(kn) l3c(efcjnk) * t2b(efmn)
                     e = l3c_excits(idet,1); f = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     j = l3c_excits(idet,4); n = l3c_excits(idet,5); k = l3c_excits(idet,6);
                     x2b_ooov(j,k,:,c) = x2b_ooov(j,k,:,c) + l_amp * t2b(e,f,:,n) ! (1)
                     x2b_ooov(j,n,:,c) = x2b_ooov(j,n,:,c) - l_amp * t2b(e,f,:,k) ! (kn)
                     x2b_ooov(j,k,:,f) = x2b_ooov(j,k,:,f) - l_amp * t2b(e,c,:,n) ! (fc)
                     x2b_ooov(j,n,:,f) = x2b_ooov(j,n,:,f) + l_amp * t2b(e,c,:,k) ! (kn)(fc)
                  end do

              end subroutine compute_x2b_ooov

              subroutine compute_x2b_ovvv(x2b_ovvv,&
                                          t2b, t2c,&
                                          l3b_amps, l3b_excits,&
                                          l3c_amps, l3c_excits,&
                                          n3aab_l, n3abb_l,&
                                          noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab_l, n3abb_l

                  real(kind=8), intent(in) :: t2b(nua,nub,noa,nob), t2c(nub,nub,nob,nob)
                  integer, intent(in) :: l3b_excits(n3aab_l,6)
                  real(kind=8), intent(in) :: l3b_amps(n3aab_l)
                  integer, intent(in) :: l3c_excits(n3abb_l,6)
                  real(kind=8), intent(in) :: l3c_amps(n3abb_l)

                  real(kind=8), intent(out) :: x2b_ovvv(noa,nub,nua,nub)
                 
                  real(kind=8) :: l_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  x2b_ovvv = 0.0d0
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     ! x2b(ieab) <- A(af)A(in) -l3b(afbinm) * t2b(fenm)
                     a = l3b_excits(idet,1); f = l3b_excits(idet,2); b = l3b_excits(idet,3);
                     i = l3b_excits(idet,4); n = l3b_excits(idet,5); m = l3b_excits(idet,6);
                     x2b_ovvv(i,:,a,b) = x2b_ovvv(i,:,a,b) - l_amp * t2b(f,:,n,m) ! (1)
                     x2b_ovvv(i,:,f,b) = x2b_ovvv(i,:,f,b) + l_amp * t2b(a,:,n,m) ! (af)
                     x2b_ovvv(n,:,a,b) = x2b_ovvv(n,:,a,b) + l_amp * t2b(f,:,i,m) ! (in)
                     x2b_ovvv(n,:,f,b) = x2b_ovvv(n,:,f,b) - l_amp * t2b(a,:,i,m) ! (af)(in)
                  end do
                  do idet = 1, n3abb_l
                     l_amp = l3c_amps(idet)
                     ! x2b(ieab) <- A(fb) -l3c(afbinm) * t2c(fenm)
                     a = l3c_excits(idet,1); f = l3c_excits(idet,2); b = l3c_excits(idet,3);
                     i = l3c_excits(idet,4); n = l3c_excits(idet,5); m = l3c_excits(idet,6);
                     x2b_ovvv(i,:,a,b) = x2b_ovvv(i,:,a,b) - l_amp * t2c(f,:,n,m) ! (1)
                     x2b_ovvv(i,:,a,f) = x2b_ovvv(i,:,a,f) + l_amp * t2c(b,:,n,m) ! (fb)
                  end do

              end subroutine compute_x2b_ovvv

              subroutine compute_x2b_vovv(x2b_vovv,&
                                          t2a, t2b,&
                                          l3b_amps, l3b_excits,&
                                          l3c_amps, l3c_excits,&
                                          n3aab_l, n3abb_l,&
                                          noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab_l, n3abb_l

                  real(kind=8), intent(in) :: t2a(nua,nua,noa,noa), t2b(nua,nub,noa,nob)
                  integer, intent(in) :: l3b_excits(n3aab_l,6)
                  real(kind=8), intent(in) :: l3b_amps(n3aab_l)
                  integer, intent(in) :: l3c_excits(n3abb_l,6)
                  real(kind=8), intent(in) :: l3c_amps(n3abb_l)

                  real(kind=8), intent(out) :: x2b_vovv(nua,nob,nua,nub)
                 
                  real(kind=8) :: l_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  x2b_vovv = 0.0d0
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     ! x2b(ejab) <- A(af) -l3b(afbmnj) * t2a(efmn)
                     a = l3b_excits(idet,1); f = l3b_excits(idet,2); b = l3b_excits(idet,3);
                     m = l3b_excits(idet,4); n = l3b_excits(idet,5); j = l3b_excits(idet,6);
                     x2b_vovv(:,j,a,b) = x2b_vovv(:,j,a,b) - l_amp * t2a(:,f,m,n) ! (1)
                     x2b_vovv(:,j,f,b) = x2b_vovv(:,j,f,b) + l_amp * t2a(:,a,m,n) ! (af)
                  end do
                  do idet = 1, n3abb_l
                     l_amp = l3c_amps(idet)
                     ! x2b(ejab) <-  A(bf)A(nj) -l3c(afbmnj) * t2b(efmn)
                     a = l3c_excits(idet,1); f = l3c_excits(idet,2); b = l3c_excits(idet,3);
                     m = l3c_excits(idet,4); n = l3c_excits(idet,5); j = l3c_excits(idet,6);
                     x2b_vovv(:,j,a,b) = x2b_vovv(:,j,a,b) - l_amp * t2b(:,f,m,n) ! (1)
                     x2b_vovv(:,j,a,f) = x2b_vovv(:,j,a,f) + l_amp * t2b(:,b,m,n) ! (bf)
                     x2b_vovv(:,n,a,b) = x2b_vovv(:,n,a,b) + l_amp * t2b(:,f,m,j) ! (nj)
                     x2b_vovv(:,n,a,f) = x2b_vovv(:,n,a,f) - l_amp * t2b(:,b,m,j) ! (bf)(nj)
                  end do

              end subroutine compute_x2b_vovv

              subroutine compute_x2c_ooov(x2c_ooov,&
                                          t2b, t2c,&
                                          l3c_amps, l3c_excits,&
                                          l3d_amps, l3d_excits,&
                                          n3abb_l, n3bbb_l,&
                                          noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3abb_l, n3bbb_l

                  real(kind=8), intent(in) :: t2b(nua,nub,noa,nob), t2c(nub,nub,nob,nob)
                  integer, intent(in) :: l3c_excits(n3abb_l,6)
                  real(kind=8), intent(in) :: l3c_amps(n3abb_l)
                  integer, intent(in) :: l3d_excits(n3bbb_l,6)
                  real(kind=8), intent(in) :: l3d_amps(n3bbb_l)

                  real(kind=8), intent(out) :: x2c_ooov(nob,nob,nob,nub)
                 
                  real(kind=8) :: l_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  x2c_ooov = 0.0d0
                  do idet = 1, n3bbb_l
                     l_amp = l3d_amps(idet)
                     ! x2c(jima) <- A(ij) [A(n/ij)A(a/ef) l3d(aefijn) * t2c(efmn)]
                     a = l3d_excits(idet,1); e = l3d_excits(idet,2); f = l3d_excits(idet,3);
                     i = l3d_excits(idet,4); j = l3d_excits(idet,5); n = l3d_excits(idet,6);
                     ! only fill permutationally unique elements!
                     x2c_ooov(i,j,:,a) = x2c_ooov(i,j,:,a) - l_amp * t2c(e,f,:,n) ! (1)
                     x2c_ooov(i,j,:,e) = x2c_ooov(i,j,:,e) + l_amp * t2c(a,f,:,n) ! (ae)
                     x2c_ooov(i,j,:,f) = x2c_ooov(i,j,:,f) + l_amp * t2c(e,a,:,n) ! (af)
                     x2c_ooov(j,n,:,a) = x2c_ooov(j,n,:,a) - l_amp * t2c(e,f,:,i) ! (in)
                     x2c_ooov(j,n,:,e) = x2c_ooov(j,n,:,e) + l_amp * t2c(a,f,:,i) ! (ae)(in)
                     x2c_ooov(j,n,:,f) = x2c_ooov(j,n,:,f) + l_amp * t2c(e,a,:,i) ! (af)(in)
                     x2c_ooov(i,n,:,a) = x2c_ooov(i,n,:,a) + l_amp * t2c(e,f,:,j) ! (jn)
                     x2c_ooov(i,n,:,e) = x2c_ooov(i,n,:,e) - l_amp * t2c(a,f,:,j) ! (ae)(jn)
                     x2c_ooov(i,n,:,f) = x2c_ooov(i,n,:,f) - l_amp * t2c(e,a,:,j) ! (af)(jn)
                  end do
                  do idet = 1, n3abb_l
                     l_amp = l3c_amps(idet)
                     ! x2c(jima) <- A(ij) [A(ae) l3c(feanji) * t2b(fenm)]
                     f = l3c_excits(idet,1); e = l3c_excits(idet,2); a = l3c_excits(idet,3);
                     n = l3c_excits(idet,4); j = l3c_excits(idet,5); i = l3c_excits(idet,6);
                     ! only fill permutationally unique elements!
                     x2c_ooov(i,j,:,a) = x2c_ooov(i,j,:,a) - l_amp * t2b(f,e,n,:) ! (1)
                     x2c_ooov(i,j,:,e) = x2c_ooov(i,j,:,e) + l_amp * t2b(f,a,n,:) ! (ae)
                  end do

                  ! apply the common A(ij) antisymmetrizer
                  do i = 1, nob
                     do j = i+1, nob
                        do m = 1, nob
                           do a = 1, nub
                              x2c_ooov(i,j,m,a) = x2c_ooov(i,j,m,a) - x2c_ooov(j,i,m,a)
                           end do
                        end do
                     end do
                  end do
                  ! explicitly antisymmetrize
                  do i = 1, nob
                     do j = i+1, nob
                        x2c_ooov(j,i,:,:) = -x2c_ooov(i,j,:,:)
                     end do
                  end do

              end subroutine compute_x2c_ooov   

              subroutine compute_x2c_vovv(x2c_vovv,&
                                          t2b, t2c,&
                                          l3c_amps, l3c_excits,&
                                          l3d_amps, l3d_excits,&
                                          n3abb_l, n3bbb_l,&
                                          noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3abb_l, n3bbb_l

                  real(kind=8), intent(in) :: t2b(nua,nub,noa,nob), t2c(nub,nub,nob,nob)
                  integer, intent(in) :: l3c_excits(n3abb_l,6)
                  real(kind=8), intent(in) :: l3c_amps(n3abb_l)
                  integer, intent(in) :: l3d_excits(n3bbb_l,6)
                  real(kind=8), intent(in) :: l3d_amps(n3bbb_l)

                  real(kind=8), intent(out) :: x2c_vovv(nub,nob,nub,nub)
                 
                  real(kind=8) :: l_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  x2c_vovv = 0.0d0
                  do idet = 1, n3bbb_l
                     l_amp = l3d_amps(idet)
                     ! x2c(eiba) <- A(ab) [A(i/mn)A(f/ab) l3d(abfimn) * t2c(efmn)]
                     a = l3d_excits(idet,1); b = l3d_excits(idet,2); f = l3d_excits(idet,3);
                     i = l3d_excits(idet,4); m = l3d_excits(idet,5); n = l3d_excits(idet,6);
                     ! only fill permutationally unique elements!
                     x2c_vovv(:,i,a,b) = x2c_vovv(:,i,a,b) + l_amp * t2c(:,f,m,n) ! (1)
                     x2c_vovv(:,m,a,b) = x2c_vovv(:,m,a,b) - l_amp * t2c(:,f,i,n) ! (im)
                     x2c_vovv(:,n,a,b) = x2c_vovv(:,n,a,b) - l_amp * t2c(:,f,m,i) ! (in)
                     x2c_vovv(:,i,b,f) = x2c_vovv(:,i,b,f) + l_amp * t2c(:,a,m,n) ! (af)
                     x2c_vovv(:,m,b,f) = x2c_vovv(:,m,b,f) - l_amp * t2c(:,a,i,n) ! (im)(af)
                     x2c_vovv(:,n,b,f) = x2c_vovv(:,n,b,f) - l_amp * t2c(:,a,m,i) ! (in)(af)
                     x2c_vovv(:,i,a,f) = x2c_vovv(:,i,a,f) - l_amp * t2c(:,b,m,n) ! (bf)
                     x2c_vovv(:,m,a,f) = x2c_vovv(:,m,a,f) + l_amp * t2c(:,b,i,n) ! (im)(bf)
                     x2c_vovv(:,n,a,f) = x2c_vovv(:,n,a,f) + l_amp * t2c(:,b,m,i) ! (in)(bf)
                  end do
                  do idet = 1, n3abb_l
                     l_amp = l3c_amps(idet)
                     ! x2c(eiba) <- A(ab) [A(im) l3c(fbanmi) * t2b(fenm)]
                     f = l3c_excits(idet,1); b = l3c_excits(idet,2); a = l3c_excits(idet,3);
                     n = l3c_excits(idet,4); m = l3c_excits(idet,5); i = l3c_excits(idet,6);
                     ! only fill permutationally unique elements!
                     x2c_vovv(:,i,a,b) = x2c_vovv(:,i,a,b) + l_amp * t2b(f,:,n,m) ! (1)
                     x2c_vovv(:,m,a,b) = x2c_vovv(:,m,a,b) - l_amp * t2b(f,:,n,i) ! (im)
                  end do

                  ! apply the common A(ij) antisymmetrizer
                  do a = 1, nub
                     do b = a+1, nub
                        do i = 1, nob
                           do e = 1, nub
                              x2c_vovv(e,i,a,b) = x2c_vovv(e,i,a,b) - x2c_vovv(e,i,b,a)
                           end do
                        end do
                     end do
                  end do
                  ! explicitly antisymmetrize
                  do a = 1, nub
                     do b = a+1, nub
                        x2c_vovv(:,:,b,a) = -x2c_vovv(:,:,a,b)
                     end do
                  end do

              end subroutine compute_x2c_vovv   
        
              subroutine compute_x2a_oooo(x2a_oooo,&
                                          t3a_amps, t3a_excits,&
                                          t3b_amps, t3b_excits,&
                                          l3a_amps, l3a_excits,&
                                          l3b_amps, l3b_excits,&
                                          do_aaa_t, do_aab_t,&
                                          n3aaa_t, n3aab_t,&
                                          n3aaa_l, n3aab_l,&
                                          noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa_t, n3aab_t
                  integer, intent(in) :: n3aaa_l, n3aab_l
                  logical, intent(in) :: do_aaa_t, do_aab_t

                  integer, intent(in) :: t3a_excits(n3aaa_t,6)
                  real(kind=8), intent(in) :: t3a_amps(n3aaa_t)
                  integer, intent(in) :: t3b_excits(n3aab_t,6)
                  real(kind=8), intent(in) :: t3b_amps(n3aab_t)
                  
                  integer, intent(in) :: l3a_excits(n3aaa_l,6)
                  real(kind=8), intent(in) :: l3a_amps(n3aaa_l)
                  integer, intent(in) :: l3b_excits(n3aab_l,6)
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
                  if (do_aaa_t) then
                  ! copy t3a into buffer
                  allocate(amps_buff(n3aaa_t),excits_buff(n3aaa_t,6))
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
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     i = l3a_excits(idet,4); j = l3a_excits(idet,5); k = l3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        l = excits_buff(jdet,4); m = excits_buff(jdet,5);
                        ! < ijkabc | N[i1+ i2+ j2 j1] | lmkabc >
                        x2a_oooo(i,j,l,m) = x2a_oooo(i,j,l,m) + l_amp * t_amp
                     end do
                     ! (ik)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           l = excits_buff(jdet,4); m = excits_buff(jdet,5);
                           x2a_oooo(j,k,l,m) = x2a_oooo(j,k,l,m) + l_amp * t_amp ! flip sign to compute permutationally unique term
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           l = excits_buff(jdet,4); m = excits_buff(jdet,5);
                           x2a_oooo(i,k,l,m) = x2a_oooo(i,k,l,m) - l_amp * t_amp
                        end do
                     end if
                  end do
                  !!! ABCI LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/1,noa-2/), nua, nua, nua, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,3,4/), nua, nua, nua, noa, nloc, n3aaa_t)
                  do idet = 1, n3aaa_l
                     l_amp = l3a_amps(idet)
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     i = l3a_excits(idet,4); j = l3a_excits(idet,5); k = l3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,i)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        m = excits_buff(jdet,5); n = excits_buff(jdet,6);
                        x2a_oooo(j,k,m,n) = x2a_oooo(j,k,m,n) + l_amp * t_amp
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           m = excits_buff(jdet,5); n = excits_buff(jdet,6);
                           x2a_oooo(i,k,m,n) = x2a_oooo(i,k,m,n) - l_amp * t_amp
                        end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           m = excits_buff(jdet,5); n = excits_buff(jdet,6);
                           x2a_oooo(i,j,m,n) = x2a_oooo(i,j,m,n) + l_amp * t_amp ! flip sign to compute permutationally unique term
                        end do
                     end if
                  end do
                  !!! ABCJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/2,noa-1/), nua, nua, nua, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,3,5/), nua, nua, nua, noa, nloc, n3aaa_t)
                  do idet = 1, n3aaa_l
                     l_amp = l3a_amps(idet)
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     i = l3a_excits(idet,4); j = l3a_excits(idet,5); k = l3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,j)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        l = excits_buff(jdet,4); n = excits_buff(jdet,6);
                        x2a_oooo(i,k,l,n) = x2a_oooo(i,k,l,n) + l_amp * t_amp
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           l = excits_buff(jdet,4); n = excits_buff(jdet,6);
                           x2a_oooo(j,k,l,n) = x2a_oooo(j,k,l,n) - l_amp * t_amp
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           l = excits_buff(jdet,4); n = excits_buff(jdet,6);
                           x2a_oooo(i,j,l,n) = x2a_oooo(i,j,l,n) - l_amp * t_amp
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,excits_buff,amps_buff)
                  end if
                  !!!! x2a(ijmn) = 1/6 l3b(abcijk) t3b(abcmnk)
                  if (do_aab_t) then
                  ! copy t3b into buffer
                  allocate(amps_buff(n3aab_t),excits_buff(n3aab_t,6))
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
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     i = l3b_excits(idet,4); j = l3b_excits(idet,5); k = l3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        l = excits_buff(jdet,4); m = excits_buff(jdet,5);
                        ! < ijk~abc~ | lmk~abc~ >
                        x2a_oooo(i,j,l,m) = x2a_oooo(i,j,l,m) + l_amp * t_amp
                     end do
                  end do
                  deallocate(loc_arr,idx_table,excits_buff,amps_buff)
                  end if
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

              subroutine compute_x2a_vvvv(x2a_vvvv,&
                                          t3a_amps, t3a_excits,&
                                          t3b_amps, t3b_excits,&
                                          l3a_amps, l3a_excits,&
                                          l3b_amps, l3b_excits,&
                                          do_aaa_t, do_aab_t,&
                                          n3aaa_t, n3aab_t,&
                                          n3aaa_l, n3aab_l,&
                                          noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa_t, n3aab_t
                  integer, intent(in) :: n3aaa_l, n3aab_l
                  logical, intent(in) :: do_aaa_t, do_aab_t

                  integer, intent(in) :: t3a_excits(n3aaa_t,6)
                  real(kind=8), intent(in) :: t3a_amps(n3aaa_t)
                  integer, intent(in) :: t3b_excits(n3aab_t,6)
                  real(kind=8), intent(in) :: t3b_amps(n3aab_t)
                  
                  integer, intent(in) :: l3a_excits(n3aaa_l,6)
                  real(kind=8), intent(in) :: l3a_amps(n3aaa_l)
                  integer, intent(in) :: l3b_excits(n3aab_l,6)
                  real(kind=8), intent(in) :: l3b_amps(n3aab_l)

                  real(kind=8), intent(out) :: x2a_vvvv(nua,nua,nua,nua)
                  
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:) 
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)

                  real(kind=8) :: t_amp, l_amp, lt_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc

                  x2a_vvvv = 0.0d0
                  !!!! x2a(deab) = 1/6 l3a(abcijk) t3a(decijk)
                  if (do_aaa_t) then
                  ! copy t3a into buffer
                  allocate(amps_buff(n3aaa_t),excits_buff(n3aaa_t,6))
                  amps_buff(:) = t3a_amps(:)
                  excits_buff(:,:) = t3a_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = noa*(noa-1)*(noa-2)/6*nua
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(noa,noa,noa,nua))
                  !!! IJKC LOOP !!!
                  call get_index_table(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/3,nua/), noa, noa, noa, nua)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,3/), noa, noa, noa, nua, nloc, n3aaa_t)
                  do idet = 1, n3aaa_l
                     l_amp = l3a_amps(idet)
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     i = l3a_excits(idet,4); j = l3a_excits(idet,5); k = l3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,c)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); e = excits_buff(jdet,2);
                        x2a_vvvv(d,e,a,b) = x2a_vvvv(d,e,a,b) + l_amp * t_amp
                     end do
                     ! (ac)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); e = excits_buff(jdet,2);
                           x2a_vvvv(d,e,b,c) = x2a_vvvv(d,e,b,c) + l_amp * t_amp
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); e = excits_buff(jdet,2);
                           x2a_vvvv(d,e,a,c) = x2a_vvvv(d,e,a,c) - l_amp * t_amp
                        end do
                     end if
                  end do
                  !!! IJKA LOOP !!!
                  call get_index_table(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/1,nua-2/), noa, noa, noa, nua)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,1/), noa, noa, noa, nua, nloc, n3aaa_t)
                  do idet = 1, n3aaa_l
                     l_amp = l3a_amps(idet)
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     i = l3a_excits(idet,4); j = l3a_excits(idet,5); k = l3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,a)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); f = excits_buff(jdet,3);
                        x2a_vvvv(e,f,b,c) = x2a_vvvv(e,f,b,c) + l_amp * t_amp
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,2); f = excits_buff(jdet,3);
                           x2a_vvvv(e,f,a,c) = x2a_vvvv(e,f,a,c) - l_amp * t_amp
                        end do
                     end if
                     ! (ac)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,2); f = excits_buff(jdet,3);
                           x2a_vvvv(e,f,a,b) = x2a_vvvv(e,f,a,b) + l_amp * t_amp
                        end do
                     end if
                  end do
                  !!! IJKB LOOP !!!
                  call get_index_table(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/2,nua-1/), noa, noa, noa, nua)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,2/), noa, noa, noa, nua, nloc, n3aaa_t)
                  do idet = 1, n3aaa_l
                     l_amp = l3a_amps(idet)
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     i = l3a_excits(idet,4); j = l3a_excits(idet,5); k = l3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,b)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); f = excits_buff(jdet,3);
                        x2a_vvvv(d,f,a,c) = x2a_vvvv(d,f,a,c) + l_amp * t_amp
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); f = excits_buff(jdet,3);
                           x2a_vvvv(d,f,b,c) = x2a_vvvv(d,f,b,c) - l_amp * t_amp
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); f = excits_buff(jdet,3);
                           x2a_vvvv(d,f,a,b) = x2a_vvvv(d,f,a,b) - l_amp * t_amp
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,excits_buff,amps_buff)
                  end if
                  !!!! x2a(deab) = 1/6 l3b(abcijk) t3b(decijk)
                  if (do_aab_t) then
                  ! copy t3b into buffer
                  allocate(amps_buff(n3aab_t),excits_buff(n3aab_t,6))
                  amps_buff(:) = t3b_amps(:)
                  excits_buff(:,:) = t3b_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = noa*(noa-1)/2*nob*nub
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(noa,noa,nob,nub))
                  !!! IJKC LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nob/), (/1,nub/), noa, noa, nob, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,3/), noa, noa, nob, nub, nloc, n3aab_t)
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     i = l3b_excits(idet,4); j = l3b_excits(idet,5); k = l3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,c)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); e = excits_buff(jdet,2);
                        x2a_vvvv(d,e,a,b) = x2a_vvvv(d,e,a,b) + l_amp * t_amp
                     end do
                  end do
                  deallocate(loc_arr,idx_table,excits_buff,amps_buff)
                  end if
                  ! explicitly enforce antisymmetry 
                  ! To ensure this works, all computations to x2a_vvvv(a,b,c,d) should be to
                  ! permutationally unique elements, meaning only for a<b and c<d
                  do a = 1, nua
                     do b = a+1, nua
                        do c = 1, nua
                           do d = c+1, nua
                              x2a_vvvv(b,a,c,d) = -x2a_vvvv(a,b,c,d)
                              x2a_vvvv(a,b,d,c) = -x2a_vvvv(a,b,c,d)
                              x2a_vvvv(b,a,d,c) = x2a_vvvv(a,b,c,d)
                           end do
                        end do     
                     end do        
                  end do

              end subroutine compute_x2a_vvvv 

              subroutine compute_x2a_voov(x2a_voov,& 
                                          t3a_amps, t3a_excits,&
                                          t3b_amps, t3b_excits,&
                                          t3c_amps, t3c_excits,&
                                          l3a_amps, l3a_excits,&
                                          l3b_amps, l3b_excits,&
                                          l3c_amps, l3c_excits,&
                                          do_aaa_t, do_aab_t, do_abb_t,&
                                          n3aaa_t, n3aab_t, n3abb_t,&
                                          n3aaa_l, n3aab_l, n3abb_l,&
                                          noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa_t, n3aab_t, n3abb_t
                  integer, intent(in) :: n3aaa_l, n3aab_l, n3abb_l
                  logical, intent(in) :: do_aaa_t, do_aab_t, do_abb_t

                  integer, intent(in) :: t3a_excits(n3aaa_t,6)
                  real(kind=8), intent(in) :: t3a_amps(n3aaa_t)
                  integer, intent(in) :: t3b_excits(n3aab_t,6)
                  real(kind=8), intent(in) :: t3b_amps(n3aab_t)
                  integer, intent(in) :: t3c_excits(n3abb_t,6)
                  real(kind=8), intent(in) :: t3c_amps(n3abb_t)
                  
                  integer, intent(in) :: l3a_excits(n3aaa_l,6)
                  real(kind=8), intent(in) :: l3a_amps(n3aaa_l)
                  integer, intent(in) :: l3b_excits(n3aab_l,6)
                  real(kind=8), intent(in) :: l3b_amps(n3aab_l)
                  integer, intent(in) :: l3c_excits(n3abb_l,6)
                  real(kind=8), intent(in) :: l3c_amps(n3abb_l)

                  real(kind=8), intent(out) :: x2a_voov(nua,noa,noa,nua)
                  
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:) 
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)

                  real(kind=8) :: t_amp, l_amp, lt_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc

                  x2a_voov = 0.0d0
                  !!!! x2a(eima) <- 1/4 l3a(abcijk) t3a(ebcmjk)
                  if (do_aaa_t) then
                  ! copy t3a into buffer
                  allocate(amps_buff(n3aaa_t),excits_buff(n3aaa_t,6))
                  amps_buff(:) = t3a_amps(:)
                  excits_buff(:,:) = t3a_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = (noa-1)*(noa-2)/2*(nua-1)*(nua-2)/2
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nua,nua,noa,noa))
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,5/), nua, nua, noa, noa, nloc, n3aaa_t)
                  do idet = 1, n3aaa_l
                     l_amp = l3a_amps(idet)
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     i = l3a_excits(idet,4); j = l3a_excits(idet,5); k = l3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                        x2a_voov(f,k,n,c) = x2a_voov(f,k,n,c) + l_amp * t_amp
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                        x2a_voov(f,k,n,a) = x2a_voov(f,k,n,a) + l_amp * t_amp
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                        x2a_voov(f,k,n,b) = x2a_voov(f,k,n,b) - l_amp * t_amp
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                        x2a_voov(f,i,n,c) = x2a_voov(f,i,n,c) + l_amp * t_amp
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                        x2a_voov(f,i,n,a) = x2a_voov(f,i,n,a) + l_amp * t_amp
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                        x2a_voov(f,i,n,b) = x2a_voov(f,i,n,b) - l_amp * t_amp
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                        x2a_voov(f,j,n,c) = x2a_voov(f,j,n,c) - l_amp * t_amp
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                        x2a_voov(f,j,n,a) = x2a_voov(f,j,n,a) - l_amp * t_amp
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                        x2a_voov(f,j,n,b) = x2a_voov(f,j,n,b) + l_amp * t_amp
                     end do
                     end if
                  end do
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,5/), nua, nua, noa, noa, nloc, n3aaa_t)
                  do idet = 1, n3aaa_l
                     l_amp = l3a_amps(idet)
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     i = l3a_excits(idet,4); j = l3a_excits(idet,5); k = l3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        x2a_voov(e,k,n,b) = x2a_voov(e,k,n,b) + l_amp * t_amp
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        x2a_voov(e,k,n,a) = x2a_voov(e,k,n,a) - l_amp * t_amp
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        x2a_voov(e,k,n,c) = x2a_voov(e,k,n,c) - l_amp * t_amp
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        x2a_voov(e,i,n,b) = x2a_voov(e,i,n,b) + l_amp * t_amp
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        x2a_voov(e,i,n,a) = x2a_voov(e,i,n,a) - l_amp * t_amp
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        x2a_voov(e,i,n,c) = x2a_voov(e,i,n,c) - l_amp * t_amp
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        x2a_voov(e,j,n,b) = x2a_voov(e,j,n,b) - l_amp * t_amp
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        x2a_voov(e,j,n,a) = x2a_voov(e,j,n,a) + l_amp * t_amp
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        x2a_voov(e,j,n,c) = x2a_voov(e,j,n,c) + l_amp * t_amp
                     end do
                     end if
                  end do
                  !!! BCIJ LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/1,noa-2/), (/-1,noa-1/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,5/), nua, nua, noa, noa, nloc, n3aaa_t)
                  do idet = 1, n3aaa_l
                     l_amp = l3a_amps(idet)
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     i = l3a_excits(idet,4); j = l3a_excits(idet,5); k = l3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); n = excits_buff(jdet,6);
                        x2a_voov(d,k,n,a) = x2a_voov(d,k,n,a) + l_amp * t_amp
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); n = excits_buff(jdet,6);
                        x2a_voov(d,k,n,b) = x2a_voov(d,k,n,b) - l_amp * t_amp
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); n = excits_buff(jdet,6);
                        x2a_voov(d,k,n,c) = x2a_voov(d,k,n,c) + l_amp * t_amp
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); n = excits_buff(jdet,6);
                        x2a_voov(d,i,n,a) = x2a_voov(d,i,n,a) + l_amp * t_amp
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); n = excits_buff(jdet,6);
                        x2a_voov(d,i,n,b) = x2a_voov(d,i,n,b) - l_amp * t_amp
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); n = excits_buff(jdet,6);
                        x2a_voov(d,i,n,c) = x2a_voov(d,i,n,c) + l_amp * t_amp
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); n = excits_buff(jdet,6);
                        x2a_voov(d,j,n,a) = x2a_voov(d,j,n,a) - l_amp * t_amp
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); n = excits_buff(jdet,6);
                        x2a_voov(d,j,n,b) = x2a_voov(d,j,n,b) + l_amp * t_amp
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); n = excits_buff(jdet,6);
                        x2a_voov(d,j,n,c) = x2a_voov(d,j,n,c) - l_amp * t_amp
                     end do
                     end if
                  end do
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,6/), nua, nua, noa, noa, nloc, n3aaa_t)
                  do idet = 1, n3aaa_l
                     l_amp = l3a_amps(idet)
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     i = l3a_excits(idet,4); j = l3a_excits(idet,5); k = l3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        x2a_voov(f,j,m,c) = x2a_voov(f,j,m,c) + l_amp * t_amp
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        x2a_voov(f,j,m,a) = x2a_voov(f,j,m,a) + l_amp * t_amp
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        x2a_voov(f,j,m,b) = x2a_voov(f,j,m,b) - l_amp * t_amp
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        x2a_voov(f,i,m,c) = x2a_voov(f,i,m,c) - l_amp * t_amp
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        x2a_voov(f,i,m,a) = x2a_voov(f,i,m,a) - l_amp * t_amp
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        x2a_voov(f,i,m,b) = x2a_voov(f,i,m,b) + l_amp * t_amp
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        x2a_voov(f,k,m,c) = x2a_voov(f,k,m,c) - l_amp * t_amp
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        x2a_voov(f,k,m,a) = x2a_voov(f,k,m,a) - l_amp * t_amp
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        x2a_voov(f,k,m,b) = x2a_voov(f,k,m,b) + l_amp * t_amp
                     end do
                     end if
                  end do
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,6/), nua, nua, noa, noa, nloc, n3aaa_t)
                  do idet = 1, n3aaa_l
                     l_amp = l3a_amps(idet)
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     i = l3a_excits(idet,4); j = l3a_excits(idet,5); k = l3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        x2a_voov(e,j,m,b) = x2a_voov(e,j,m,b) + l_amp * t_amp
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        x2a_voov(e,j,m,a) = x2a_voov(e,j,m,a) - l_amp * t_amp
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        x2a_voov(e,j,m,c) = x2a_voov(e,j,m,c) - l_amp * t_amp
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        x2a_voov(e,i,m,b) = x2a_voov(e,i,m,b) - l_amp * t_amp
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        x2a_voov(e,i,m,a) = x2a_voov(e,i,m,a) + l_amp * t_amp
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        x2a_voov(e,i,m,c) = x2a_voov(e,i,m,c) + l_amp * t_amp
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        x2a_voov(e,k,m,b) = x2a_voov(e,k,m,b) - l_amp * t_amp
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        x2a_voov(e,k,m,a) = x2a_voov(e,k,m,a) + l_amp * t_amp
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        x2a_voov(e,k,m,c) = x2a_voov(e,k,m,c) + l_amp * t_amp
                     end do
                     end if
                  end do
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/1,noa-2/), (/-2,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,6/), nua, nua, noa, noa, nloc, n3aaa_t)
                  do idet = 1, n3aaa_l
                     l_amp = l3a_amps(idet)
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     i = l3a_excits(idet,4); j = l3a_excits(idet,5); k = l3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        x2a_voov(d,j,m,a) = x2a_voov(d,j,m,a) + l_amp * t_amp
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        x2a_voov(d,j,m,b) = x2a_voov(d,j,m,b) - l_amp * t_amp
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        x2a_voov(d,j,m,c) = x2a_voov(d,j,m,c) + l_amp * t_amp
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        x2a_voov(d,i,m,a) = x2a_voov(d,i,m,a) - l_amp * t_amp
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        x2a_voov(d,i,m,b) = x2a_voov(d,i,m,b) + l_amp * t_amp
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        x2a_voov(d,i,m,c) = x2a_voov(d,i,m,c) - l_amp * t_amp
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        x2a_voov(d,k,m,a) = x2a_voov(d,k,m,a) - l_amp * t_amp
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        x2a_voov(d,k,m,b) = x2a_voov(d,k,m,b) + l_amp * t_amp
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        x2a_voov(d,k,m,c) = x2a_voov(d,k,m,c) - l_amp * t_amp
                     end do
                     end if
                  end do
                  !!! ABJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,5,6/), nua, nua, noa, noa, nloc, n3aaa_t)
                  do idet = 1, n3aaa_l
                     l_amp = l3a_amps(idet)
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     i = l3a_excits(idet,4); j = l3a_excits(idet,5); k = l3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); l = excits_buff(jdet,4);
                        x2a_voov(f,i,l,c) = x2a_voov(f,i,l,c) + l_amp * t_amp
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); l = excits_buff(jdet,4);
                        x2a_voov(f,i,l,a) = x2a_voov(f,i,l,a) + l_amp * t_amp
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); l = excits_buff(jdet,4);
                        x2a_voov(f,i,l,b) = x2a_voov(f,i,l,b) - l_amp * t_amp
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); l = excits_buff(jdet,4);
                        x2a_voov(f,j,l,c) = x2a_voov(f,j,l,c) - l_amp * t_amp
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); l = excits_buff(jdet,4);
                        x2a_voov(f,j,l,a) = x2a_voov(f,j,l,a) - l_amp * t_amp
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); l = excits_buff(jdet,4);
                        x2a_voov(f,j,l,b) = x2a_voov(f,j,l,b) + l_amp * t_amp
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); l = excits_buff(jdet,4);
                        x2a_voov(f,k,l,c) = x2a_voov(f,k,l,c) + l_amp * t_amp
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); l = excits_buff(jdet,4);
                        x2a_voov(f,k,l,a) = x2a_voov(f,k,l,a) + l_amp * t_amp
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); l = excits_buff(jdet,4);
                        x2a_voov(f,k,l,b) = x2a_voov(f,k,l,b) - l_amp * t_amp
                     end do
                     end if
                  end do
                  !!! ACJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,5,6/), nua, nua, noa, noa, nloc, n3aaa_t)
                  do idet = 1, n3aaa_l
                     l_amp = l3a_amps(idet)
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     i = l3a_excits(idet,4); j = l3a_excits(idet,5); k = l3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); l = excits_buff(jdet,4);
                        x2a_voov(e,i,l,b) = x2a_voov(e,i,l,b) + l_amp * t_amp
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); l = excits_buff(jdet,4);
                        x2a_voov(e,i,l,a) = x2a_voov(e,i,l,a) - l_amp * t_amp
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); l = excits_buff(jdet,4);
                        x2a_voov(e,i,l,c) = x2a_voov(e,i,l,c) - l_amp * t_amp
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); l = excits_buff(jdet,4);
                        x2a_voov(e,j,l,b) = x2a_voov(e,j,l,b) - l_amp * t_amp
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); l = excits_buff(jdet,4);
                        x2a_voov(e,j,l,a) = x2a_voov(e,j,l,a) + l_amp * t_amp
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); l = excits_buff(jdet,4);
                        x2a_voov(e,j,l,c) = x2a_voov(e,j,l,c) + l_amp * t_amp
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); l = excits_buff(jdet,4);
                        x2a_voov(e,k,l,b) = x2a_voov(e,k,l,b) + l_amp * t_amp
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); l = excits_buff(jdet,4);
                        x2a_voov(e,k,l,a) = x2a_voov(e,k,l,a) - l_amp * t_amp
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); l = excits_buff(jdet,4);
                        x2a_voov(e,k,l,c) = x2a_voov(e,k,l,c) - l_amp * t_amp
                     end do
                     end if
                  end do
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/2,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,5,6/), nua, nua, noa, noa, nloc, n3aaa_t)
                  do idet = 1, n3aaa_l
                     l_amp = l3a_amps(idet)
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     i = l3a_excits(idet,4); j = l3a_excits(idet,5); k = l3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                        ! < ijkabc | ljkdbc >
                        x2a_voov(d,i,l,a) = x2a_voov(d,i,l,a) + l_amp * t_amp
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                        ! < ijkabc | ljkdac >
                        x2a_voov(d,i,l,b) = x2a_voov(d,i,l,b) - l_amp * t_amp
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                        x2a_voov(d,i,l,c) = x2a_voov(d,i,l,c) + l_amp * t_amp
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                        x2a_voov(d,j,l,a) = x2a_voov(d,j,l,a) - l_amp * t_amp
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                        x2a_voov(d,j,l,b) = x2a_voov(d,j,l,b) + l_amp * t_amp
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                        x2a_voov(d,j,l,c) = x2a_voov(d,j,l,c) - l_amp * t_amp
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                        x2a_voov(d,k,l,a) = x2a_voov(d,k,l,a) + l_amp * t_amp
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                        x2a_voov(d,k,l,b) = x2a_voov(d,k,l,b) - l_amp * t_amp
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                        x2a_voov(d,k,l,c) = x2a_voov(d,k,l,c) + l_amp * t_amp
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table,excits_buff,amps_buff)
                  end if
                  !!!! x2a(eima) <- 1/4 l3b(abcijk) t3b(ebcmjk)
                  if (do_aab_t) then
                  ! copy t3b into buffer
                  allocate(amps_buff(n3aab_t),excits_buff(n3aab_t,6))
                  amps_buff(:) = t3b_amps(:)
                  excits_buff(:,:) = t3b_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = noa*nob*nua*nub
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nua,nub,noa,nob))
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/2,nua/), (/1,nub/), (/2,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,5,6/), nua, nub, noa, nob, nloc, n3aab_t)
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     i = l3b_excits(idet,4); j = l3b_excits(idet,5); k = l3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,j,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                        x2a_voov(d,i,l,a) = x2a_voov(d,i,l,a) + l_amp * t_amp
                     end do
                     ! (ab)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                        x2a_voov(d,i,l,b) = x2a_voov(d,i,l,b) - l_amp * t_amp
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                        x2a_voov(d,j,l,a) = x2a_voov(d,j,l,a) - l_amp * t_amp
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                        x2a_voov(d,j,l,b) = x2a_voov(d,j,l,b) + l_amp * t_amp
                     end do
                     end if
                  end do 
                  !!! ACJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/1,nub/), (/2,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,5,6/), nua, nub, noa, nob, nloc, n3aab_t)
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     i = l3b_excits(idet,4); j = l3b_excits(idet,5); k = l3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,j,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); l = excits_buff(jdet,4);
                        x2a_voov(e,i,l,b) = x2a_voov(e,i,l,b) + l_amp * t_amp
                     end do
                     ! (ab)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); l = excits_buff(jdet,4);
                        x2a_voov(e,i,l,a) = x2a_voov(e,i,l,a) - l_amp * t_amp
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); l = excits_buff(jdet,4);
                        x2a_voov(e,j,l,b) = x2a_voov(e,j,l,b) - l_amp * t_amp
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); l = excits_buff(jdet,4);
                        x2a_voov(e,j,l,a) = x2a_voov(e,j,l,a) + l_amp * t_amp
                     end do
                     end if
                  end do 
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/2,nua/), (/1,nub/), (/1,noa-1/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,6/), nua, nub, noa, nob, nloc, n3aab_t)
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     i = l3b_excits(idet,4); j = l3b_excits(idet,5); k = l3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,i,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        x2a_voov(d,j,m,a) = x2a_voov(d,j,m,a) + l_amp * t_amp
                     end do
                     ! (ab)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        x2a_voov(d,j,m,b) = x2a_voov(d,j,m,b) - l_amp * t_amp
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        x2a_voov(d,i,m,a) = x2a_voov(d,i,m,a) - l_amp * t_amp
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        x2a_voov(d,i,m,b) = x2a_voov(d,i,m,b) + l_amp * t_amp
                     end do
                     end if
                  end do 
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/1,nub/), (/1,noa-1/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,6/), nua, nub, noa, nob, nloc, n3aab_t)
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     i = l3b_excits(idet,4); j = l3b_excits(idet,5); k = l3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        x2a_voov(e,j,m,b) = x2a_voov(e,j,m,b) + l_amp * t_amp
                     end do
                     ! (ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        x2a_voov(e,j,m,a) = x2a_voov(e,j,m,a) - l_amp * t_amp
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        x2a_voov(e,i,m,b) = x2a_voov(e,i,m,b) - l_amp * t_amp
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        x2a_voov(e,i,m,a) = x2a_voov(e,i,m,a) + l_amp * t_amp
                     end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,excits_buff,amps_buff)
                  end if
                  !!!! x2a(dila) <- 1/4 l3c(abcijk) t3c(dbcljk)
                  if (do_abb_t) then
                  ! copy t3c into buffer
                  allocate(amps_buff(n3abb_t),excits_buff(n3abb_t,6))
                  amps_buff(:) = t3c_amps(:)
                  excits_buff(:,:) = t3c_excits(:,:)
                  ! allocate sorting arrays
                  nloc = nob*(nob-1)/2*nub*(nub-1)/2
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nub,nub,nob,nob))
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,5,6/), nub, nub, nob, nob, nloc, n3abb_t)
                  do idet = 1, n3abb_l
                      l_amp = l3c_amps(idet)
                      a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                      i = l3c_excits(idet,4); j = l3c_excits(idet,5); k = l3c_excits(idet,6);
                      idx = idx_table(b,c,j,k)
                      do jdet = loc_arr(idx,1), loc_arr(idx,2)
                         t_amp = amps_buff(jdet)
                         d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                         x2a_voov(d,i,l,a) = x2a_voov(d,i,l,a) + l_amp * t_amp
                      end do
                  end do 
                  deallocate(loc_arr,idx_table,excits_buff,amps_buff)
                  end if

              end subroutine compute_x2a_voov                    

              subroutine compute_x2b_oooo(x2b_oooo,&
                                          t3b_amps, t3b_excits,&
                                          t3c_amps, t3c_excits,&
                                          l3b_amps, l3b_excits,&
                                          l3c_amps, l3c_excits,&
                                          do_aab_t, do_abb_t,&
                                          n3aab_t, n3abb_t,&
                                          n3aab_l, n3abb_l,&
                                          noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab_t, n3abb_t
                  integer, intent(in) :: n3aab_l, n3abb_l
                  logical, intent(in) :: do_aab_t, do_abb_t

                  integer, intent(in) :: t3b_excits(n3aab_t,6)
                  real(kind=8), intent(in) :: t3b_amps(n3aab_t)
                  integer, intent(in) :: t3c_excits(n3abb_t,6)
                  real(kind=8), intent(in) :: t3c_amps(n3abb_t)
                  
                  integer, intent(in) :: l3b_excits(n3aab_l,6)
                  real(kind=8), intent(in) :: l3b_amps(n3aab_l)
                  integer, intent(in) :: l3c_excits(n3abb_l,6)
                  real(kind=8), intent(in) :: l3c_amps(n3abb_l)

                  real(kind=8), intent(out) :: x2b_oooo(noa,nob,noa,nob)
                  
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:) 
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)

                  real(kind=8) :: t_amp, l_amp, lt_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc

                  x2b_oooo = 0.0d0
                  !!!! x2b(jkmn) = 1/2 l3b(abcijk) t3b(abcimn)
                  if (do_aab_t) then
                  ! copy t3b into buffer
                  allocate(amps_buff(n3aab_t),excits_buff(n3aab_t,6))
                  amps_buff(:) = t3b_amps(:)
                  excits_buff(:,:) = t3b_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2*nub*noa
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nua,nua,nub,noa))
                  !!! ABCI LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/1,noa-1/), nua, nua, nub, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,3,4/), nua, nua, nub, noa, nloc, n3aab_t)
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     i = l3b_excits(idet,4); j = l3b_excits(idet,5); k = l3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,i)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        m = excits_buff(jdet,5); n = excits_buff(jdet,6);
                        x2b_oooo(j,k,m,n) = x2b_oooo(j,k,m,n) + l_amp * t_amp
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           m = excits_buff(jdet,5); n = excits_buff(jdet,6);
                           x2b_oooo(i,k,m,n) = x2b_oooo(i,k,m,n) - l_amp * t_amp
                        end do
                     end if
                  end do
                  !!! ABCJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/2,noa/), nua, nua, nub, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,3,5/), nua, nua, nub, noa, nloc, n3aab_t)
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     i = l3b_excits(idet,4); j = l3b_excits(idet,5); k = l3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,j)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        l = excits_buff(jdet,4); n = excits_buff(jdet,6);
                        x2b_oooo(i,k,l,n) = x2b_oooo(i,k,l,n) + l_amp * t_amp
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           l = excits_buff(jdet,4); n = excits_buff(jdet,6);
                           x2b_oooo(j,k,l,n) = x2b_oooo(j,k,l,n) - l_amp * t_amp
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,excits_buff,amps_buff)
                  end if
                  !!!! x2b(jkmn) = 1/2 l3c(abcijk) t3c(abclmk)
                  if (do_abb_t) then
                  ! copy t3c into buffer
                  allocate(amps_buff(n3abb_t),excits_buff(n3abb_t,6))
                  amps_buff(:) = t3c_amps(:)
                  excits_buff(:,:) = t3c_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = nub*(nub-1)/2*nua*nob
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nub,nub,nua,nob))
                  !!! BCAK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nua/), (/2,nob/), nub, nub, nua, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,1,6/), nub, nub, nua, nob, nloc, n3abb_t)
                  do idet = 1, n3abb_l
                     l_amp = l3c_amps(idet)
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     i = l3c_excits(idet,4); j = l3c_excits(idet,5); k = l3c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,a,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        l = excits_buff(jdet,4); m = excits_buff(jdet,5);
                        x2b_oooo(i,j,l,m) = x2b_oooo(i,j,l,m) + l_amp * t_amp
                     end do
                     ! (jk)
                     idx = idx_table(b,c,a,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           l = excits_buff(jdet,4); m = excits_buff(jdet,5);
                           x2b_oooo(i,k,l,m) = x2b_oooo(i,k,l,m) - l_amp * t_amp
                        end do
                     end if
                  end do
                  !!! BCAJ LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nua/), (/1,nob-1/), nub, nub, nua, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,1,5/), nub, nub, nua, nob, nloc, n3abb_t)
                  do idet = 1, n3abb_l
                     l_amp = l3c_amps(idet)
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     i = l3c_excits(idet,4); j = l3c_excits(idet,5); k = l3c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,a,j)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        l = excits_buff(jdet,4); n = excits_buff(jdet,6);
                        x2b_oooo(i,k,l,n) = x2b_oooo(i,k,l,n) + l_amp * t_amp
                     end do
                     ! (jk)
                     idx = idx_table(b,c,a,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           l = excits_buff(jdet,4); n = excits_buff(jdet,6);
                           x2b_oooo(i,j,l,n) = x2b_oooo(i,j,l,n) - l_amp * t_amp
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,excits_buff,amps_buff)
                  end if

              end subroutine compute_x2b_oooo

              subroutine compute_x2b_vvvv(x2b_vvvv,&
                                          t3b_amps, t3b_excits,&
                                          t3c_amps, t3c_excits,&
                                          l3b_amps, l3b_excits,&
                                          l3c_amps, l3c_excits,&
                                          do_aab_t, do_abb_t,&
                                          n3aab_t, n3abb_t,&
                                          n3aab_l, n3abb_l,&
                                          noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab_t, n3abb_t
                  integer, intent(in) :: n3aab_l, n3abb_l
                  logical, intent(in) :: do_aab_t, do_abb_t

                  integer, intent(in) :: t3b_excits(n3aab_t,6)
                  real(kind=8), intent(in) :: t3b_amps(n3aab_t)
                  integer, intent(in) :: t3c_excits(n3abb_t,6)
                  real(kind=8), intent(in) :: t3c_amps(n3abb_t)
                  
                  integer, intent(in) :: l3b_excits(n3aab_l,6)
                  real(kind=8), intent(in) :: l3b_amps(n3aab_l)
                  integer, intent(in) :: l3c_excits(n3abb_l,6)
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
                  if (do_aab_t) then
                  ! copy t3b into buffer
                  allocate(t3_amps_buff(n3aab_t),t3_excits_buff(n3aab_t,6))
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
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     i = l3b_excits(idet,4); j = l3b_excits(idet,5); k = l3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,a)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = t3_amps_buff(jdet)
                        e = t3_excits_buff(jdet,2); f = t3_excits_buff(jdet,3);
                        ! compute < ijk~abc~ | ijk~aef~ >
                        x2b_vvvv(e,f,b,c) = x2b_vvvv(e,f,b,c) + l_amp * t_amp
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = t3_amps_buff(jdet)
                           e = t3_excits_buff(jdet,2); f = t3_excits_buff(jdet,3);
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
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     i = l3b_excits(idet,4); j = l3b_excits(idet,5); k = l3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,b)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = t3_amps_buff(jdet)
                        e = t3_excits_buff(jdet,1); f = t3_excits_buff(jdet,3);
                        ! compute < ijk~abc~ | ijk~ebf~ >
                        x2b_vvvv(e,f,a,c) = x2b_vvvv(e,f,a,c) + l_amp * t_amp
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = t3_amps_buff(jdet)
                           e = t3_excits_buff(jdet,1); f = t3_excits_buff(jdet,3);
                           ! compute < ijk~abc~ | ijk~eaf~ >
                           x2b_vvvv(e,f,b,c) = x2b_vvvv(e,f,b,c) - l_amp * t_amp
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,t3_amps_buff,t3_excits_buff)
                  end if
                  !!!! x2b(efab) <- 1/2 l3c(abcijk) t3c(efcijk)
                  if (do_abb_t) then
                  ! copy t3c into buffer
                  allocate(t3_amps_buff(n3abb_t),t3_excits_buff(n3abb_t,6))
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
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     i = l3c_excits(idet,4); j = l3c_excits(idet,5); k = l3c_excits(idet,6);
                     ! (1)
                     idx = idx_table(j,k,i,c)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = t3_amps_buff(jdet)
                        e = t3_excits_buff(jdet,1); f = t3_excits_buff(jdet,2);
                        ! compute < ij~k~ab~c~ | ij~k~ef~c~ >
                        x2b_vvvv(e,f,a,b) = x2b_vvvv(e,f,a,b) + l_amp * t_amp
                     end do
                     ! (bc)
                     idx = idx_table(j,k,i,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = t3_amps_buff(jdet)
                           e = t3_excits_buff(jdet,1); f = t3_excits_buff(jdet,2);
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
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     i = l3c_excits(idet,4); j = l3c_excits(idet,5); k = l3c_excits(idet,6);
                     ! (1)
                     idx = idx_table(j,k,i,b)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = t3_amps_buff(jdet)
                        e = t3_excits_buff(jdet,1); f = t3_excits_buff(jdet,3);
                        ! compute < ij~k~ab~c~ | ij~k~eb~f~ >
                        x2b_vvvv(e,f,a,c) = x2b_vvvv(e,f,a,c) + l_amp * t_amp
                     end do
                     ! (bc)
                     idx = idx_table(j,k,i,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = t3_amps_buff(jdet)
                           e = t3_excits_buff(jdet,1); f = t3_excits_buff(jdet,3);
                           ! compute < ij~k~ab~c~ | ij~k~ec~f~ >
                           x2b_vvvv(e,f,a,b) = x2b_vvvv(e,f,a,b) - l_amp * t_amp
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,t3_amps_buff,t3_excits_buff)
                  end if

              end subroutine compute_x2b_vvvv

              subroutine compute_x2b_voov(x2b_voov,&
                                          t3a_amps, t3a_excits,&
                                          t3b_amps, t3b_excits,&
                                          t3c_amps, t3c_excits,&
                                          l3b_amps, l3b_excits,&
                                          l3c_amps, l3c_excits,&
                                          l3d_amps, l3d_excits,&
                                          do_abb_t, do_aab_l, do_abb_l,&
                                          n3aaa_t, n3aab_t, n3abb_t,&
                                          n3aab_l, n3abb_l, n3bbb_l,&
                                          noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa_t, n3aab_t, n3abb_t
                  integer, intent(in) :: n3aab_l, n3abb_l, n3bbb_l
                  logical, intent(in) :: do_abb_t, do_aab_l, do_abb_l

                  integer, intent(in) :: t3a_excits(n3aaa_t,6)
                  real(kind=8), intent(in) :: t3a_amps(n3aaa_t)
                  integer, intent(in) :: t3b_excits(n3aab_t,6)
                  real(kind=8), intent(in) :: t3b_amps(n3aab_t)
                  integer, intent(in) :: t3c_excits(n3abb_t,6)
                  real(kind=8), intent(in) :: t3c_amps(n3abb_t)
                  
                  integer, intent(in) :: l3b_excits(n3aab_l,6)
                  real(kind=8), intent(in) :: l3b_amps(n3aab_l)
                  integer, intent(in) :: l3c_excits(n3abb_l,6)
                  real(kind=8), intent(in) :: l3c_amps(n3abb_l)
                  integer, intent(in) :: l3d_excits(n3bbb_l,6)
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
                  if (do_aab_l) then
                  ! copy l3b into buffer
                  allocate(amps_buff(n3aab_l),excits_buff(n3aab_l,6))
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
                     a = t3a_excits(idet,1); b = t3a_excits(idet,2); c = t3a_excits(idet,3);
                     i = t3a_excits(idet,4); j = t3a_excits(idet,5); k = t3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,a,b)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        l_amp = amps_buff(jdet)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                        ! compute < ijm~abe~ | ijkabc >
                        x2b_voov(c,m,k,e) = x2b_voov(c,m,k,e) + l_amp * t_amp
                     end do
                     ! (ac)
                     idx = idx_table(i,j,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                           ! compute < ijm~bce~ | ijkabc >
                           x2b_voov(a,m,k,e) = x2b_voov(a,m,k,e) + l_amp * t_amp
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                           ! compute < ijm~ace~ | ijkabc >
                           x2b_voov(b,m,k,e) = x2b_voov(b,m,k,e) - l_amp * t_amp
                        end do
                     end if
                     ! (ik)
                     idx = idx_table(j,k,a,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                           ! compute < jkm~abe~ | ijkabc >
                           x2b_voov(c,m,i,e) = x2b_voov(c,m,i,e) + l_amp * t_amp
                        end do
                     end if
                     ! (ik)(ac)
                     idx = idx_table(j,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                           ! compute < jkm~bce~ | ijkabc >
                           x2b_voov(a,m,i,e) = x2b_voov(a,m,i,e) + l_amp * t_amp
                        end do
                     end if
                     ! (ik)(bc)
                     idx = idx_table(j,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                           ! compute < jkm~ace~ | ijkabc >
                           x2b_voov(b,m,i,e) = x2b_voov(b,m,i,e) - l_amp * t_amp
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(i,k,a,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                           ! compute < ikm~abe~ | ijkabc >
                           x2b_voov(c,m,j,e) = x2b_voov(c,m,j,e) - l_amp * t_amp
                        end do
                     end if
                     ! (jk)(ac)
                     idx = idx_table(i,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                           ! compute < ikm~bce~ | ijkabc >
                           x2b_voov(a,m,j,e) = x2b_voov(a,m,j,e) - l_amp * t_amp
                        end do
                     end if
                     ! (jk)(bc)
                     idx = idx_table(i,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                           ! compute < ikm~ace~ | ijkabc >
                           x2b_voov(b,m,j,e) = x2b_voov(b,m,j,e) + l_amp * t_amp
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,amps_buff,excits_buff)
                  end if
                  !!!! x2b(amie) <- l3c(bcejkm) t3b(abcijk) : This one's tricky. See diagram 14 in update_t3b_p in ccsdt_p for help.
                  if (do_abb_l) then
                  ! copy l3c into buffer
                  allocate(amps_buff(n3abb_l),excits_buff(n3abb_l,6))
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
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(j,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                           x2b_voov(a,m,i,e) = x2b_voov(a,m,i,e) + l_amp * t_amp
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(i,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                           x2b_voov(a,m,j,e) = x2b_voov(a,m,j,e) - l_amp * t_amp
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(j,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                           x2b_voov(b,m,i,e) = x2b_voov(b,m,i,e) - l_amp * t_amp
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(i,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                           x2b_voov(b,m,j,e) = x2b_voov(b,m,j,e) + l_amp * t_amp
                        end do
                     end if
                  end do
                  !!! IKAB LOOP !!!
                  call get_index_table(idx_table, (/1,noa/), (/2,nob/), (/1,nua/), (/1,nub-1/), noa, nob, nua, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,6,1,2/), noa, nob, nua, nub, nloc, n3abb_l)
                  do idet = 1, n3aab_t
                     t_amp = t3b_amps(idet)
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(j,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(jdet,3); m = excits_buff(jdet,5);
                           x2b_voov(a,m,i,e) = x2b_voov(a,m,i,e) - l_amp * t_amp
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(i,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(jdet,3); m = excits_buff(jdet,5);
                           x2b_voov(a,m,j,e) = x2b_voov(a,m,j,e) + l_amp * t_amp
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(j,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(jdet,3); m = excits_buff(jdet,5);
                           x2b_voov(b,m,i,e) = x2b_voov(b,m,i,e) + l_amp * t_amp
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(i,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(jdet,3); m = excits_buff(jdet,5);
                           x2b_voov(b,m,j,e) = x2b_voov(b,m,j,e) - l_amp * t_amp
                        end do
                     end if
                  end do
                  !!! IJAC LOOP !!!
                  call get_index_table(idx_table, (/1,noa/), (/1,nob-1/), (/1,nua/), (/2,nub/), noa, nob, nua, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,1,3/), noa, nob, nua, nub, nloc, n3abb_l)
                  do idet = 1, n3aab_t
                     t_amp = t3b_amps(idet)
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(j,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(jdet,2); m = excits_buff(jdet,6);
                           x2b_voov(a,m,i,e) = x2b_voov(a,m,i,e) - l_amp * t_amp
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(i,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(jdet,2); m = excits_buff(jdet,6);
                           x2b_voov(a,m,j,e) = x2b_voov(a,m,j,e) + l_amp * t_amp
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(j,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(jdet,2); m = excits_buff(jdet,6);
                           x2b_voov(b,m,i,e) = x2b_voov(b,m,i,e) + l_amp * t_amp
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(i,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(jdet,2); m = excits_buff(jdet,6);
                           x2b_voov(b,m,j,e) = x2b_voov(b,m,j,e) - l_amp * t_amp
                        end do
                     end if
                  end do
                  !!! IKAC LOOP !!!
                  call get_index_table(idx_table, (/1,noa/), (/2,nob/), (/1,nua/), (/2,nub/), noa, nob, nua, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,6,1,3/), noa, nob, nua, nub, nloc, n3abb_l)
                  do idet = 1, n3aab_t
                     t_amp = t3b_amps(idet)
                     a = t3b_excits(idet,1); b = t3b_excits(idet,2); c = t3b_excits(idet,3);
                     i = t3b_excits(idet,4); j = t3b_excits(idet,5); k = t3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(j,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                           x2b_voov(a,m,i,e) = x2b_voov(a,m,i,e) + l_amp * t_amp
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(i,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                           x2b_voov(a,m,j,e) = x2b_voov(a,m,j,e) - l_amp * t_amp
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(j,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                           x2b_voov(b,m,i,e) = x2b_voov(b,m,i,e) - l_amp * t_amp
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(i,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                           x2b_voov(b,m,j,e) = x2b_voov(b,m,j,e) + l_amp * t_amp
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,excits_buff,amps_buff)
                  end if
                  !!!! x2b(ekmc) <- 1/4 l3d(abcijk) t3c(eabmij) : a little tricky, see diagram 6 of update_t3d_p for help.
                  if (do_abb_t) then
                  ! copy t3c into buffer
                  allocate(amps_buff(n3abb_t),excits_buff(n3abb_t,6))
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
                     a = l3d_excits(idet,1); b = l3d_excits(idet,2); c = l3d_excits(idet,3);
                     i = l3d_excits(idet,4); j = l3d_excits(idet,5); k = l3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,a,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,1); m = excits_buff(jdet,4);
                           x2b_voov(e,k,m,c) = x2b_voov(e,k,m,c) + l_amp * t_amp
                        end do
                     end if
                     ! (ik)
                     idx = idx_table(j,k,a,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,1); m = excits_buff(jdet,4);
                           x2b_voov(e,i,m,c) = x2b_voov(e,i,m,c) + l_amp * t_amp
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(i,k,a,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,1); m = excits_buff(jdet,4);
                           x2b_voov(e,j,m,c) = x2b_voov(e,j,m,c) - l_amp * t_amp
                        end do
                     end if
                     ! (ac)
                     idx = idx_table(i,j,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,1); m = excits_buff(jdet,4);
                           x2b_voov(e,k,m,a) = x2b_voov(e,k,m,a) + l_amp * t_amp
                        end do
                     end if
                     ! (ik)(ac)
                     idx = idx_table(j,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,1); m = excits_buff(jdet,4);
                           x2b_voov(e,i,m,a) = x2b_voov(e,i,m,a) + l_amp * t_amp
                        end do
                     end if
                     ! (jk)(ac)
                     idx = idx_table(i,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,1); m = excits_buff(jdet,4);
                           x2b_voov(e,j,m,a) = x2b_voov(e,j,m,a) - l_amp * t_amp
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,1); m = excits_buff(jdet,4);
                           x2b_voov(e,k,m,b) = x2b_voov(e,k,m,b) - l_amp * t_amp
                        end do
                     end if
                     ! (ik)(bc)
                     idx = idx_table(j,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,1); m = excits_buff(jdet,4);
                           x2b_voov(e,i,m,b) = x2b_voov(e,i,m,b) - l_amp * t_amp
                        end do
                     end if
                     ! (jk)(bc)
                     idx = idx_table(i,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,1); m = excits_buff(jdet,4);
                           x2b_voov(e,j,m,b) = x2b_voov(e,j,m,b) + l_amp * t_amp
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,amps_buff,excits_buff)
                  end if

              end subroutine compute_x2b_voov

              subroutine compute_x2b_ovvo(x2b_ovvo,&
                                          t3b_amps, t3b_excits,&
                                          t3c_amps, t3c_excits,&
                                          t3d_amps, t3d_excits,&
                                          l3a_amps, l3a_excits,&
                                          l3b_amps, l3b_excits,&
                                          l3c_amps, l3c_excits,&
                                          do_aab_t, do_abb_t, do_abb_l,&
                                          n3aab_t, n3abb_t, n3bbb_t,&
                                          n3aaa_l, n3aab_l, n3abb_l,&
                                          noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab_t, n3abb_t, n3bbb_t
                  integer, intent(in) :: n3aaa_l, n3aab_l, n3abb_l
                  logical, intent(in) :: do_aab_t, do_abb_t, do_abb_l

                  integer, intent(in) :: t3b_excits(n3aab_t,6)
                  real(kind=8), intent(in) :: t3b_amps(n3aab_t)
                  integer, intent(in) :: t3c_excits(n3abb_t,6)
                  real(kind=8), intent(in) :: t3c_amps(n3abb_t)
                  integer, intent(in) :: t3d_excits(n3bbb_t,6)
                  real(kind=8), intent(in) :: t3d_amps(n3bbb_t)
                  
                  integer, intent(in) :: l3a_excits(n3aaa_l,6)
                  real(kind=8), intent(in) :: l3a_amps(n3aaa_l)
                  integer, intent(in) :: l3b_excits(n3aab_l,6)
                  real(kind=8), intent(in) :: l3b_amps(n3aab_l)
                  integer, intent(in) :: l3c_excits(n3abb_l,6)
                  real(kind=8), intent(in) :: l3c_amps(n3abb_l)

                  real(kind=8), intent(out) :: x2b_ovvo(noa,nub,nua,nob)
                  
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:) 
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)

                  real(kind=8) :: t_amp, l_amp, lt_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc

                  x2b_ovvo = 0.0d0
                  !!!! x2b(kecm) <- 1/4 l3a(abcijk) t3b(abeijm)
                  if (do_aab_t) then
                  ! copy t3b into buffer
                  allocate(amps_buff(n3aab_t),excits_buff(n3aab_t,6))
                  amps_buff(:) = t3b_amps(:)
                  excits_buff(:,:) = t3b_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = noa*(noa-1)/2*nua*(nua-1)/2
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(noa,noa,nua,nua))
                  !!! IJAB LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nua-1/), (/-1,nua/), noa, noa, nua, nua)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,1,2/), noa, noa, nua, nua, nloc, n3aab_t)
                  do idet = 1, n3aaa_l
                     l_amp = l3a_amps(idet)
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); c = l3a_excits(idet,3);
                     i = l3a_excits(idet,4); j = l3a_excits(idet,5); k = l3a_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,a,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                           x2b_ovvo(k,f,c,n) = x2b_ovvo(k,f,c,n) + l_amp * t_amp
                        end do
                     end if
                     ! (ik)
                     idx = idx_table(j,k,a,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                           x2b_ovvo(i,f,c,n) = x2b_ovvo(i,f,c,n) + l_amp * t_amp
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(i,k,a,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                           x2b_ovvo(j,f,c,n) = x2b_ovvo(j,f,c,n) - l_amp * t_amp
                        end do
                     end if
                     ! (ac)
                     idx = idx_table(i,j,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                           x2b_ovvo(k,f,a,n) = x2b_ovvo(k,f,a,n) + l_amp * t_amp
                        end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(j,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                           x2b_ovvo(i,f,a,n) = x2b_ovvo(i,f,a,n) + l_amp * t_amp
                        end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(i,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                           x2b_ovvo(j,f,a,n) = x2b_ovvo(j,f,a,n) - l_amp * t_amp
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                           x2b_ovvo(k,f,b,n) = x2b_ovvo(k,f,b,n) - l_amp * t_amp
                        end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(j,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                           x2b_ovvo(i,f,b,n) = x2b_ovvo(i,f,b,n) - l_amp * t_amp
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(i,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                           x2b_ovvo(j,f,b,n) = x2b_ovvo(j,f,b,n) + l_amp * t_amp
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,excits_buff,amps_buff)
                  end if
                  !!!! x2b(ieam) <- l3b(abcijk) t3c(becjmk)
                  if (do_abb_t) then
                  ! copy t3c into buffer
                  allocate(amps_buff(n3abb_t),excits_buff(n3abb_t,6))
                  amps_buff(:) = t3c_amps(:)
                  excits_buff(:,:) = t3c_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = noa*nob*nua*nub
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(noa,nob,nua,nub))
                  !!! IKAC LOOP !!!
                  call get_index_table(idx_table, (/1,noa/), (/2,nob/), (/1,nua/), (/2,nub/), noa, nob, nua, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,6,1,3/), noa, nob, nua, nub, nloc, n3abb_t)
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     i = l3b_excits(idet,4); j = l3b_excits(idet,5); k = l3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(j,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           ! < abc~ijk~ | be~c~jm~k~ > = < bac~jik~ | be~c~jm~k~ >
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                           x2b_ovvo(i,e,a,m) = x2b_ovvo(i,e,a,m) + l_amp * t_amp
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(i,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           ! < abc~ijk~ | be~c~im~k~ > = -< bac~ijk~ | be~c~im~k~ >
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                           x2b_ovvo(j,e,a,m) = x2b_ovvo(j,e,a,m) - l_amp * t_amp
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(j,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           ! < abc~ijk~ | ae~c~jm~k~ > = -< abc~jik~ | ae~c~jm~k~ >
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                           x2b_ovvo(i,e,b,m) = x2b_ovvo(i,e,b,m) - l_amp * t_amp
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(i,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           ! < abc~ijk~ | ae~c~im~k~ >
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                           x2b_ovvo(j,e,b,m) = x2b_ovvo(j,e,b,m) + l_amp * t_amp
                        end do
                     end if
                  end do
                  !!! IJAC LOOP !!!
                  call get_index_table(idx_table, (/1,noa/), (/1,nob-1/), (/1,nua/), (/2,nub/), noa, nob, nua, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,1,3/), noa, nob, nua, nub, nloc, n3abb_t)
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     i = l3b_excits(idet,4); j = l3b_excits(idet,5); k = l3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(j,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           ! < abc~ijk~ |  be~c~jk~m~ > = -< bac~jik~ | be~c~jm~k~ >
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,2); m = excits_buff(jdet,6);
                           x2b_ovvo(i,e,a,m) = x2b_ovvo(i,e,a,m) - l_amp * t_amp
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(i,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,2); m = excits_buff(jdet,6);
                           x2b_ovvo(j,e,a,m) = x2b_ovvo(j,e,a,m) + l_amp * t_amp
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(j,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,2); m = excits_buff(jdet,6);
                           x2b_ovvo(i,e,b,m) = x2b_ovvo(i,e,b,m) + l_amp * t_amp
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(i,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,2); m = excits_buff(jdet,6);
                           x2b_ovvo(j,e,b,m) = x2b_ovvo(j,e,b,m) - l_amp * t_amp
                        end do
                     end if
                  end do
                  !!! IKAB LOOP !!!
                  call get_index_table(idx_table, (/1,noa/), (/2,nob/), (/1,nua/), (/1,nub-1/), noa, nob, nua, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,6,1,2/), noa, nob, nua, nub, nloc, n3abb_t)
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     i = l3b_excits(idet,4); j = l3b_excits(idet,5); k = l3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(j,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           ! < abc~ijk~ |  bc~e~jm~k~ > = -< bac~jik~ | be~c~jm~k~ >
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,3); m = excits_buff(jdet,5);
                           x2b_ovvo(i,e,a,m) = x2b_ovvo(i,e,a,m) - l_amp * t_amp
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(i,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,3); m = excits_buff(jdet,5);
                           x2b_ovvo(j,e,a,m) = x2b_ovvo(j,e,a,m) + l_amp * t_amp
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(j,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,3); m = excits_buff(jdet,5);
                           x2b_ovvo(i,e,b,m) = x2b_ovvo(i,e,b,m) + l_amp * t_amp
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(i,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,3); m = excits_buff(jdet,5);
                           x2b_ovvo(j,e,b,m) = x2b_ovvo(j,e,b,m) - l_amp * t_amp
                        end do
                     end if
                  end do
                  !!! IJAB LOOP !!!
                  call get_index_table(idx_table, (/1,noa/), (/1,nob-1/), (/1,nua/), (/1,nub-1/), noa, nob, nua, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,1,2/), noa, nob, nua, nub, nloc, n3abb_t)
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     i = l3b_excits(idet,4); j = l3b_excits(idet,5); k = l3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(j,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           ! < abc~ijk~ |  bc~e~jk~m~ > = < bac~jik~ | be~c~jm~k~ >
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                           x2b_ovvo(i,e,a,m) = x2b_ovvo(i,e,a,m) + l_amp * t_amp
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(i,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                           x2b_ovvo(j,e,a,m) = x2b_ovvo(j,e,a,m) - l_amp * t_amp
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(j,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                           x2b_ovvo(i,e,b,m) = x2b_ovvo(i,e,b,m) - l_amp * t_amp
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(i,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                           x2b_ovvo(j,e,b,m) = x2b_ovvo(j,e,b,m) + l_amp * t_amp
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,excits_buff,amps_buff)
                  end if
                  !!!! x2b(ladi) <- 1/4 l3c(dbcljk) t3d(abcijk)
                  if (do_abb_l) then
                  ! copy l3c into buffer
                  allocate(amps_buff(n3abb_l),excits_buff(n3abb_l,6))
                  amps_buff(:) = l3c_amps(:)
                  excits_buff(:,:) = l3c_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = nob*(nob-1)/2*nub*(nub-1)/2
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nob,nob,nub,nub))
                  !!! JKBC LOOP !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,nub-1/), (/-1,nub/), nob, nob, nub, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/5,6,2,3/), nob, nob, nub, nub, nloc, n3abb_l)
                  do idet = 1, n3bbb_t
                     t_amp = t3d_amps(idet)
                     a = t3d_excits(idet,1); b = t3d_excits(idet,2); c = t3d_excits(idet,3);
                     i = t3d_excits(idet,4); j = t3d_excits(idet,5); k = t3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(j,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           ! < db~c~j~lj~k~ | a~b~c~i~j~k~ >
                           d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                           x2b_ovvo(l,a,d,i) = x2b_ovvo(l,a,d,i) + l_amp * t_amp
                        end do
                     end if
                     ! (ij)
                     idx = idx_table(i,k,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                           x2b_ovvo(l,a,d,j) = x2b_ovvo(l,a,d,j) - l_amp * t_amp
                        end do
                     end if
                     ! (ik) [x]
                     idx = idx_table(i,j,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                           x2b_ovvo(l,a,d,k) = x2b_ovvo(l,a,d,k) + l_amp * t_amp
                        end do
                     end if
                     ! (ab)
                     idx = idx_table(j,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                           x2b_ovvo(l,b,d,i) = x2b_ovvo(l,b,d,i) - l_amp * t_amp
                        end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(i,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                           x2b_ovvo(l,b,d,j) = x2b_ovvo(l,b,d,j) + l_amp * t_amp
                        end do
                     end if
                     ! (ik)(ab) [x]
                     idx = idx_table(i,j,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                           x2b_ovvo(l,b,d,k) = x2b_ovvo(l,b,d,k) - l_amp * t_amp
                        end do
                     end if
                     ! (ac) [x]
                     idx = idx_table(j,k,a,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                           x2b_ovvo(l,c,d,i) = x2b_ovvo(l,c,d,i) + l_amp * t_amp
                        end do
                     end if
                     ! (ij)(ac) [x]
                     idx = idx_table(i,k,a,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                           x2b_ovvo(l,c,d,j) = x2b_ovvo(l,c,d,j) - l_amp * t_amp
                        end do
                     end if
                     ! (ik)(ac) [x]
                     idx = idx_table(i,j,a,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                           x2b_ovvo(l,c,d,k) = x2b_ovvo(l,c,d,k) + l_amp * t_amp
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,excits_buff,amps_buff)
                  end if

              end subroutine compute_x2b_ovvo

              subroutine compute_x2b_vovo(x2b_vovo,&
                                          t3b_amps, t3b_excits,&
                                          t3c_amps, t3c_excits,&
                                          l3b_amps, l3b_excits,&
                                          l3c_amps, l3c_excits,&
                                          do_aab_t, do_abb_t,&
                                          n3aab_t, n3abb_t,&
                                          n3aab_l, n3abb_l,&
                                          noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab_t, n3abb_t
                  integer, intent(in) :: n3aab_l, n3abb_l
                  logical, intent(in) :: do_aab_t, do_abb_t

                  integer, intent(in) :: t3b_excits(n3aab_t,6)
                  real(kind=8), intent(in) :: t3b_amps(n3aab_t)
                  integer, intent(in) :: t3c_excits(n3abb_t,6)
                  real(kind=8), intent(in) :: t3c_amps(n3abb_t)
                  
                  integer, intent(in) :: l3b_excits(n3aab_l,6)
                  real(kind=8), intent(in) :: l3b_amps(n3aab_l)
                  integer, intent(in) :: l3c_excits(n3abb_l,6)
                  real(kind=8), intent(in) :: l3c_amps(n3abb_l)

                  real(kind=8), intent(out) :: x2b_vovo(nua,nob,nua,nob)
                  
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:) 
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)

                  real(kind=8) :: t_amp, l_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc

                  x2b_vovo = 0.0d0
                  !!!! x2b(ekma) <- -1/2 l3b(abcijk) t3b(ebcijm)
                  if (do_aab_t) then
                  ! copy t3b into buffer
                  allocate(amps_buff(n3aab_t),excits_buff(n3aab_t,6))
                  amps_buff(:) = t3b_amps(:)
                  excits_buff(:,:) = t3b_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = noa*(noa-1)/2*nua*nub
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(noa,noa,nua,nub))
                  !!! IJBC LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/2,nua/), (/1,nub/), noa, noa, nua, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,2,3/), noa, noa, nua, nub, nloc, n3aab_t)
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     i = l3b_excits(idet,4); j = l3b_excits(idet,5); k = l3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,b,c)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); n = excits_buff(jdet,6);
                        x2b_vovo(d,k,a,n) = x2b_vovo(d,k,a,n) - l_amp * t_amp
                     end do
                     ! (ab)
                     idx = idx_table(i,j,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); n = excits_buff(jdet,6);
                           x2b_vovo(d,k,b,n) = x2b_vovo(d,k,b,n) + l_amp * t_amp
                        end do
                     end if
                  end do
                  !!! IJAC LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nua-1/), (/1,nub/), noa, noa, nua, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,1,3/), noa, noa, nua, nub, nloc, n3aab_t)
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     i = l3b_excits(idet,4); j = l3b_excits(idet,5); k = l3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,a,c)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        x2b_vovo(e,k,b,n) = x2b_vovo(e,k,b,n) - l_amp * t_amp
                     end do
                     ! (ab)
                     idx = idx_table(i,j,b,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,2); n = excits_buff(jdet,6);
                           x2b_vovo(e,k,a,n) = x2b_vovo(e,k,a,n) + l_amp * t_amp
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,excits_buff,amps_buff)
                  end if
                  !!!! x2b(ekma) <- -1/2 l3c(abcijk) t3c(ebcijm)
                  if (do_abb_t) then
                  ! copy t3c into buffer
                  allocate(amps_buff(n3abb_t),excits_buff(n3abb_t,6))
                  amps_buff(:) = t3c_amps(:)
                  excits_buff(:,:) = t3c_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = nub*(nub-1)/2*noa*nob
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nub,nub,noa,nob))
                  !!! BCIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,noa/), (/1,nob-1/), nub, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,5/), nub, nub, noa, nob, nloc, n3abb_t)
                  do idet = 1, n3abb_l
                     l_amp = l3c_amps(idet)
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     i = l3c_excits(idet,4); j = l3c_excits(idet,5); k = l3c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,i,j)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); n = excits_buff(jdet,6);
                        x2b_vovo(d,k,a,n) = x2b_vovo(d,k,a,n) - l_amp * t_amp
                     end do
                     ! (jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); n = excits_buff(jdet,6);
                           x2b_vovo(d,j,a,n) = x2b_vovo(d,j,a,n) + l_amp * t_amp
                        end do
                     end if
                  end do
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,noa/), (/2,nob/), nub, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,6/), nub, nub, noa, nob, nloc, n3abb_t)
                  do idet = 1, n3abb_l
                     l_amp = l3c_amps(idet)
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     i = l3c_excits(idet,4); j = l3c_excits(idet,5); k = l3c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,i,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        x2b_vovo(d,j,a,m) = x2b_vovo(d,j,a,m) - l_amp * t_amp
                     end do
                     ! (jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); m = excits_buff(jdet,5);
                           x2b_vovo(d,k,a,m) = x2b_vovo(d,k,a,m) + l_amp * t_amp
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,excits_buff,amps_buff)
                  end if

              end subroutine compute_x2b_vovo

              subroutine compute_x2b_ovov(x2b_ovov,&
                                          t3b_amps, t3b_excits,&
                                          t3c_amps, t3c_excits,&
                                          l3b_amps, l3b_excits,&
                                          l3c_amps, l3c_excits,&
                                          do_aab_t, do_abb_t,&
                                          n3aab_t, n3abb_t,&
                                          n3aab_l, n3abb_l,&
                                          noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab_t, n3abb_t
                  integer, intent(in) :: n3aab_l, n3abb_l
                  logical, intent(in) :: do_aab_t, do_abb_t

                  integer, intent(in) :: t3b_excits(n3aab_t,6)
                  real(kind=8), intent(in) :: t3b_amps(n3aab_t)
                  integer, intent(in) :: t3c_excits(n3abb_t,6)
                  real(kind=8), intent(in) :: t3c_amps(n3abb_t)
                  
                  integer, intent(in) :: l3b_excits(n3aab_l,6)
                  real(kind=8), intent(in) :: l3b_amps(n3aab_l)
                  integer, intent(in) :: l3c_excits(n3abb_l,6)
                  real(kind=8), intent(in) :: l3c_amps(n3abb_l)

                  real(kind=8), intent(out) :: x2b_ovov(noa,nub,noa,nub)
                  
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:) 
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)

                  real(kind=8) :: t_amp, l_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc

                  x2b_ovov = 0.0d0
                  !!!! x2b(iemc) <- -1/2 l3b(abcijk) t3b(abemjk)
                  if (do_aab_t) then
                  ! copy t3b into buffer
                  allocate(amps_buff(n3aab_t),excits_buff(n3aab_t,6))
                  amps_buff(:) = t3b_amps(:)
                  excits_buff(:,:) = t3b_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2*noa*nob
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nua,nua,noa,nob))
                  !!! ABJK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/2,noa/), (/1,nob/), nua, nua, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,5,6/), nua, nua, noa, nob, nloc, n3aab_t)
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     i = l3b_excits(idet,4); j = l3b_excits(idet,5); k = l3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,j,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); l = excits_buff(jdet,4);
                        x2b_ovov(i,f,l,c) = x2b_ovov(i,f,l,c) - l_amp * t_amp
                     end do
                     ! (ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           f = excits_buff(jdet,3); l = excits_buff(jdet,4);
                           x2b_ovov(j,f,l,c) = x2b_ovov(j,f,l,c) + l_amp * t_amp
                        end do
                     end if
                  end do
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,noa-1/), (/1,nob/), nua, nua, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,6/), nua, nua, noa, nob, nloc, n3aab_t)
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     i = l3b_excits(idet,4); j = l3b_excits(idet,5); k = l3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,i,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        x2b_ovov(j,f,m,c) = x2b_ovov(j,f,m,c) - l_amp * t_amp
                     end do
                     ! (ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           f = excits_buff(jdet,3); m = excits_buff(jdet,5);
                           x2b_ovov(i,f,m,c) = x2b_ovov(i,f,m,c) + l_amp * t_amp
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,excits_buff,amps_buff)
                  end if
                  !!!! x2b(iemc) <- -1/2 l3c(abcijk) t3c(abemjk)
                  if (do_abb_t) then
                  ! copy t3c into buffer
                  allocate(amps_buff(n3abb_t),excits_buff(n3abb_t,6))
                  amps_buff(:) = t3c_amps(:)
                  excits_buff(:,:) = t3c_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = nob*(nob-1)/2*nua*nub
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nob,nob,nua,nub))
                  !!! JKAB LOOP !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,nua/), (/1,nub-1/), nob, nob, nua, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/5,6,1,2/), nob, nob, nua, nub, nloc, n3abb_t)
                  do idet = 1, n3abb_l
                     l_amp = l3c_amps(idet)
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     i = l3c_excits(idet,4); j = l3c_excits(idet,5); k = l3c_excits(idet,6);
                     ! (1)
                     idx = idx_table(j,k,a,b)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); l = excits_buff(jdet,4);  
                        x2b_ovov(i,f,l,c) = x2b_ovov(i,f,l,c) - l_amp * t_amp
                     end do
                     ! (bc)
                     idx = idx_table(j,k,a,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           f = excits_buff(jdet,3); l = excits_buff(jdet,4);  
                           x2b_ovov(i,f,l,b) = x2b_ovov(i,f,l,b) + l_amp * t_amp
                        end do
                     end if
                  end do
                  !!! JKAC LOOP !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,nua/), (/2,nub/), nob, nob, nua, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/5,6,1,3/), nob, nob, nua, nub, nloc, n3abb_t)
                  do idet = 1, n3abb_l
                     l_amp = l3c_amps(idet)
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     i = l3c_excits(idet,4); j = l3c_excits(idet,5); k = l3c_excits(idet,6);
                     ! (1)
                     idx = idx_table(j,k,a,c)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); l = excits_buff(jdet,4);  
                        x2b_ovov(i,e,l,b) = x2b_ovov(i,e,l,b) - l_amp * t_amp
                     end do
                     ! (bc)
                     idx = idx_table(j,k,a,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,2); l = excits_buff(jdet,4);  
                           x2b_ovov(i,e,l,c) = x2b_ovov(i,e,l,c) + l_amp * t_amp
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,excits_buff,amps_buff)
                  end if

              end subroutine compute_x2b_ovov

              subroutine compute_x2c_oooo(x2c_oooo,&
                                          t3c_amps, t3c_excits,&
                                          t3d_amps, t3d_excits,&
                                          l3c_amps, l3c_excits,&
                                          l3d_amps, l3d_excits,&
                                          do_abb_t, do_bbb_t,&
                                          n3abb_t, n3bbb_t,&
                                          n3abb_l, n3bbb_l,&
                                          noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3abb_t, n3bbb_t
                  integer, intent(in) :: n3abb_l, n3bbb_l
                  logical, intent(in) :: do_abb_t, do_bbb_t

                  integer, intent(in) :: t3c_excits(n3abb_t,6)
                  real(kind=8), intent(in) :: t3c_amps(n3abb_t)
                  integer, intent(in) :: t3d_excits(n3bbb_t,6)
                  real(kind=8), intent(in) :: t3d_amps(n3bbb_t)
                  
                  integer, intent(in) :: l3c_excits(n3abb_l,6)
                  real(kind=8), intent(in) :: l3c_amps(n3abb_l)
                  integer, intent(in) :: l3d_excits(n3bbb_l,6)
                  real(kind=8), intent(in) :: l3d_amps(n3bbb_l)

                  real(kind=8), intent(out) :: x2c_oooo(nob,nob,nob,nob)
                  
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:) 
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)

                  real(kind=8) :: t_amp, l_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc

                  x2c_oooo = 0.0d0
                  !!!! x2a(ijmn) = 1/6 l3a(abcijk) t3a(abcmnk)
                  if (do_bbb_t) then
                  ! copy t3d into buffer
                  allocate(amps_buff(n3bbb_t),excits_buff(n3bbb_t,6))
                  amps_buff(:) = t3d_amps(:)
                  excits_buff(:,:) = t3d_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = nub*(nub-1)*(nub-2)/6*nob
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nub,nub,nub,nob))
                  !!! ABCK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/-1,nub/), (/3,nob/), nub, nub, nub, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,3,6/), nub, nub, nub, nob, nloc, n3bbb_t)
                  do idet = 1, n3bbb_l
                     l_amp = l3d_amps(idet)
                     a = l3d_excits(idet,1); b = l3d_excits(idet,2); c = l3d_excits(idet,3);
                     i = l3d_excits(idet,4); j = l3d_excits(idet,5); k = l3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        l = excits_buff(jdet,4); m = excits_buff(jdet,5);
                        ! < ijkabc | N[i1+ i2+ j2 j1] | lmkabc >
                        x2c_oooo(i,j,l,m) = x2c_oooo(i,j,l,m) + l_amp * t_amp
                     end do
                     ! (ik)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           l = excits_buff(jdet,4); m = excits_buff(jdet,5);
                           x2c_oooo(j,k,l,m) = x2c_oooo(j,k,l,m) + l_amp * t_amp ! flip sign to compute permutationally unique term
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           l = excits_buff(jdet,4); m = excits_buff(jdet,5);
                           x2c_oooo(i,k,l,m) = x2c_oooo(i,k,l,m) - l_amp * t_amp
                        end do
                     end if
                  end do
                  !!! ABCI LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/-1,nub/), (/1,nob-2/), nub, nub, nub, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,3,4/), nub, nub, nub, nob, nloc, n3bbb_t)
                  do idet = 1, n3bbb_l
                     l_amp = l3d_amps(idet)
                     a = l3d_excits(idet,1); b = l3d_excits(idet,2); c = l3d_excits(idet,3);
                     i = l3d_excits(idet,4); j = l3d_excits(idet,5); k = l3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,i)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        m = excits_buff(jdet,5); n = excits_buff(jdet,6);
                        x2c_oooo(j,k,m,n) = x2c_oooo(j,k,m,n) + l_amp * t_amp
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           m = excits_buff(jdet,5); n = excits_buff(jdet,6);
                           x2c_oooo(i,k,m,n) = x2c_oooo(i,k,m,n) - l_amp * t_amp
                        end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           m = excits_buff(jdet,5); n = excits_buff(jdet,6);
                           x2c_oooo(i,j,m,n) = x2c_oooo(i,j,m,n) + l_amp * t_amp ! flip sign to compute permutationally unique term
                        end do
                     end if
                  end do
                  !!! ABCJ LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/-1,nub/), (/2,nob-1/), nub, nub, nub, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,3,5/), nub, nub, nub, nob, nloc, n3bbb_t)
                  do idet = 1, n3bbb_l
                     l_amp = l3d_amps(idet)
                     a = l3d_excits(idet,1); b = l3d_excits(idet,2); c = l3d_excits(idet,3);
                     i = l3d_excits(idet,4); j = l3d_excits(idet,5); k = l3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,j)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        l = excits_buff(jdet,4); n = excits_buff(jdet,6);
                        x2c_oooo(i,k,l,n) = x2c_oooo(i,k,l,n) + l_amp * t_amp
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           l = excits_buff(jdet,4); n = excits_buff(jdet,6);
                           x2c_oooo(j,k,l,n) = x2c_oooo(j,k,l,n) - l_amp * t_amp
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           l = excits_buff(jdet,4); n = excits_buff(jdet,6);
                           x2c_oooo(i,j,l,n) = x2c_oooo(i,j,l,n) - l_amp * t_amp
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,excits_buff,amps_buff)
                  end if
                  !!!! x2c(jkmn) = 1/6 l3c(abcijk) t3c(abcimn)
                  if (do_abb_t) then
                  ! copy t3c into buffer
                  allocate(amps_buff(n3abb_t),excits_buff(n3abb_t,6))
                  amps_buff(:) = t3c_amps(:)
                  excits_buff(:,:) = t3c_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = nub*(nub-1)/2*nua*noa
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nub,nub,nua,noa))
                  !!! BCAI LOOP !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nua/), (/1,noa/), nub, nub, nua, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,1,4/), nub, nub, nua, noa, nloc, n3abb_t)
                  do idet = 1, n3abb_l
                     l_amp = l3c_amps(idet)
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     i = l3c_excits(idet,4); j = l3c_excits(idet,5); k = l3c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,a,i)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        m = excits_buff(jdet,5); n = excits_buff(jdet,6);
                        ! < ij~k~ab~c~ | im~n~ab~c~ >
                        x2c_oooo(j,k,m,n) = x2c_oooo(j,k,m,n) + l_amp * t_amp
                     end do
                  end do
                  deallocate(loc_arr,idx_table,excits_buff,amps_buff)
                  end if
                  ! explicitly enforce antisymmetry 
                  ! To ensure this works, all computations to x2a_oooo(i,j,k,l) should be to
                  ! permutationally unique elements, meaning only for i<j and k<l
                  do i = 1, nob
                     do j = i+1, nob
                        do k = 1, nob
                           do l = k+1, nob
                              x2c_oooo(j,i,k,l) = -x2c_oooo(i,j,k,l)
                              x2c_oooo(i,j,l,k) = -x2c_oooo(i,j,k,l)
                              x2c_oooo(j,i,l,k) = x2c_oooo(i,j,k,l)
                           end do
                        end do     
                     end do        
                  end do

              end subroutine compute_x2c_oooo 

              subroutine compute_x2c_vvvv(x2c_vvvv,&
                                          t3c_amps, t3c_excits,&
                                          t3d_amps, t3d_excits,&
                                          l3c_amps, l3c_excits,&
                                          l3d_amps, l3d_excits,&
                                          do_abb_t, do_bbb_t,&
                                          n3abb_t, n3bbb_t,&
                                          n3abb_l, n3bbb_l,&
                                          noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3abb_t, n3bbb_t
                  integer, intent(in) :: n3abb_l, n3bbb_l
                  logical, intent(in) :: do_abb_t, do_bbb_t

                  integer, intent(in) :: t3c_excits(n3abb_t,6)
                  real(kind=8), intent(in) :: t3c_amps(n3abb_t)
                  integer, intent(in) :: t3d_excits(n3bbb_t,6)
                  real(kind=8), intent(in) :: t3d_amps(n3bbb_t)
                  
                  integer, intent(in) :: l3c_excits(n3abb_l,6)
                  real(kind=8), intent(in) :: l3c_amps(n3abb_l)
                  integer, intent(in) :: l3d_excits(n3bbb_l,6)
                  real(kind=8), intent(in) :: l3d_amps(n3bbb_l)

                  real(kind=8), intent(out) :: x2c_vvvv(nub,nub,nub,nub)
                  
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:) 
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)

                  real(kind=8) :: t_amp, l_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc

                  x2c_vvvv = 0.0d0
                  !!!! x2a(deab) = 1/6 l3a(abcijk) t3a(decijk)
                  if (do_bbb_t) then
                  ! copy t3a into buffer
                  allocate(amps_buff(n3bbb_t),excits_buff(n3bbb_t,6))
                  amps_buff(:) = t3d_amps(:)
                  excits_buff(:,:) = t3d_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = nob*(nob-1)*(nob-2)/6*nub
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nob,nob,nob,nub))
                  !!! IJKC LOOP !!!
                  call get_index_table(idx_table, (/1,nob-2/), (/-1,nob-1/), (/-1,nob/), (/3,nub/), nob, nob, nob, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,3/), nob, nob, nob, nub, nloc, n3bbb_t)
                  do idet = 1, n3bbb_l
                     l_amp = l3d_amps(idet)
                     a = l3d_excits(idet,1); b = l3d_excits(idet,2); c = l3d_excits(idet,3);
                     i = l3d_excits(idet,4); j = l3d_excits(idet,5); k = l3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,c)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); e = excits_buff(jdet,2);
                        x2c_vvvv(d,e,a,b) = x2c_vvvv(d,e,a,b) + l_amp * t_amp
                     end do
                     ! (ac)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); e = excits_buff(jdet,2);
                           x2c_vvvv(d,e,b,c) = x2c_vvvv(d,e,b,c) + l_amp * t_amp
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); e = excits_buff(jdet,2);
                           x2c_vvvv(d,e,a,c) = x2c_vvvv(d,e,a,c) - l_amp * t_amp
                        end do
                     end if
                  end do
                  !!! IJKA LOOP !!!
                  call get_index_table(idx_table, (/1,nob-2/), (/-1,nob-1/), (/-1,nob/), (/1,nub-2/), nob, nob, nob, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,1/), nob, nob, nob, nub, nloc, n3bbb_t)
                  do idet = 1, n3bbb_l
                     l_amp = l3d_amps(idet)
                     a = l3d_excits(idet,1); b = l3d_excits(idet,2); c = l3d_excits(idet,3);
                     i = l3d_excits(idet,4); j = l3d_excits(idet,5); k = l3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,a)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); f = excits_buff(jdet,3);
                        x2c_vvvv(e,f,b,c) = x2c_vvvv(e,f,b,c) + l_amp * t_amp
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,2); f = excits_buff(jdet,3);
                           x2c_vvvv(e,f,a,c) = x2c_vvvv(e,f,a,c) - l_amp * t_amp
                        end do
                     end if
                     ! (ac)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           e = excits_buff(jdet,2); f = excits_buff(jdet,3);
                           x2c_vvvv(e,f,a,b) = x2c_vvvv(e,f,a,b) + l_amp * t_amp
                        end do
                     end if
                  end do
                  !!! IJKB LOOP !!!
                  call get_index_table(idx_table, (/1,nob-2/), (/-1,nob-1/), (/-1,nob/), (/2,nub-1/), nob, nob, nob, nub)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,2/), nob, nob, nob, nub, nloc, n3bbb_t)
                  do idet = 1, n3bbb_l
                     l_amp = l3d_amps(idet)
                     a = l3d_excits(idet,1); b = l3d_excits(idet,2); c = l3d_excits(idet,3);
                     i = l3d_excits(idet,4); j = l3d_excits(idet,5); k = l3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,b)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); f = excits_buff(jdet,3);
                        x2c_vvvv(d,f,a,c) = x2c_vvvv(d,f,a,c) + l_amp * t_amp
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); f = excits_buff(jdet,3);
                           x2c_vvvv(d,f,b,c) = x2c_vvvv(d,f,b,c) - l_amp * t_amp
                        end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           d = excits_buff(jdet,1); f = excits_buff(jdet,3);
                           x2c_vvvv(d,f,a,b) = x2c_vvvv(d,f,a,b) - l_amp * t_amp
                        end do
                     end if
                  end do
                  deallocate(loc_arr,idx_table,excits_buff,amps_buff)
                  end if
                  !!!! x2c(efbc) = 1/6 l3c(abcijk) t3c(aefijk)
                  if (do_abb_t) then
                  ! copy t3c into buffer
                  allocate(amps_buff(n3abb_t),excits_buff(n3abb_t,6))
                  amps_buff(:) = t3c_amps(:)
                  excits_buff(:,:) = t3c_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = nob*(nob-1)/2*noa*nua
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nob,nob,noa,nua))
                  !!! JKIA LOOP !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,noa/), (/1,nua/), nob, nob, noa, nua)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/5,6,4,1/), nob, nob, noa, nua, nloc, n3abb_t)
                  do idet = 1, n3abb_l
                     l_amp = l3c_amps(idet)
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     i = l3c_excits(idet,4); j = l3c_excits(idet,5); k = l3c_excits(idet,6);
                     ! (1)
                     idx = idx_table(j,k,i,a)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); f = excits_buff(jdet,3);
                        x2c_vvvv(e,f,b,c) = x2c_vvvv(e,f,b,c) + l_amp * t_amp
                     end do
                  end do
                  deallocate(loc_arr,idx_table,excits_buff,amps_buff)
                  end if
                  ! explicitly enforce antisymmetry 
                  ! To ensure this works, all computations to x2c_vvvv(a,b,c,d) should be to
                  ! permutationally unique elements, meaning only for a<b and c<d
                  do a = 1, nub
                     do b = a+1, nub
                        do c = 1, nub
                           do d = c+1, nub
                              x2c_vvvv(b,a,c,d) = -x2c_vvvv(a,b,c,d)
                              x2c_vvvv(a,b,d,c) = -x2c_vvvv(a,b,c,d)
                              x2c_vvvv(b,a,d,c) = x2c_vvvv(a,b,c,d)
                           end do
                        end do     
                     end do        
                  end do

              end subroutine compute_x2c_vvvv

              subroutine compute_x2c_voov(x2c_voov,& 
                                          t3b_amps, t3b_excits,&
                                          t3c_amps, t3c_excits,&
                                          t3d_amps, t3d_excits,&
                                          l3b_amps, l3b_excits,&
                                          l3c_amps, l3c_excits,&
                                          l3d_amps, l3d_excits,&
                                          do_aab_t, do_abb_t, do_bbb_t,&
                                          n3aab_t, n3abb_t, n3bbb_t,&
                                          n3aab_l, n3abb_l, n3bbb_l,&
                                          noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab_t, n3abb_t, n3bbb_t
                  integer, intent(in) :: n3aab_l, n3abb_l, n3bbb_l
                  logical, intent(in) :: do_aab_t, do_abb_t, do_bbb_t

                  integer, intent(in) :: t3b_excits(n3aab_t,6)
                  real(kind=8), intent(in) :: t3b_amps(n3aab_t)
                  integer, intent(in) :: t3c_excits(n3abb_t,6)
                  real(kind=8), intent(in) :: t3c_amps(n3abb_t)
                  integer, intent(in) :: t3d_excits(n3bbb_t,6)
                  real(kind=8), intent(in) :: t3d_amps(n3bbb_t)
                  
                  integer, intent(in) :: l3b_excits(n3aab_l,6)
                  real(kind=8), intent(in) :: l3b_amps(n3aab_l)
                  integer, intent(in) :: l3c_excits(n3abb_l,6)
                  real(kind=8), intent(in) :: l3c_amps(n3abb_l)
                  integer, intent(in) :: l3d_excits(n3bbb_l,6)
                  real(kind=8), intent(in) :: l3d_amps(n3bbb_l)

                  real(kind=8), intent(out) :: x2c_voov(nub,nob,nob,nub)
                  
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:) 
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)

                  real(kind=8) :: t_amp, l_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc

                  x2c_voov = 0.0d0
                  !!!! x2c(eima) <- 1/4 l3d(abcijk) t3d(ebcmjk)
                  if (do_bbb_t) then
                  ! copy t3d into buffer
                  allocate(amps_buff(n3bbb_t),excits_buff(n3bbb_t,6))
                  amps_buff(:) = t3d_amps(:)
                  excits_buff(:,:) = t3d_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = (nob-1)*(nob-2)/2*(nub-1)*(nub-2)/2
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nub,nub,nob,nob))
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,5/), nub, nub, nob, nob, nloc, n3bbb_t)
                  do idet = 1, n3bbb_l
                     l_amp = l3d_amps(idet)
                     a = l3d_excits(idet,1); b = l3d_excits(idet,2); c = l3d_excits(idet,3);
                     i = l3d_excits(idet,4); j = l3d_excits(idet,5); k = l3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                        x2c_voov(f,k,n,c) = x2c_voov(f,k,n,c) + l_amp * t_amp
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                        x2c_voov(f,k,n,a) = x2c_voov(f,k,n,a) + l_amp * t_amp
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                        x2c_voov(f,k,n,b) = x2c_voov(f,k,n,b) - l_amp * t_amp
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                        x2c_voov(f,i,n,c) = x2c_voov(f,i,n,c) + l_amp * t_amp
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                        x2c_voov(f,i,n,a) = x2c_voov(f,i,n,a) + l_amp * t_amp
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                        x2c_voov(f,i,n,b) = x2c_voov(f,i,n,b) - l_amp * t_amp
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                        x2c_voov(f,j,n,c) = x2c_voov(f,j,n,c) - l_amp * t_amp
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                        x2c_voov(f,j,n,a) = x2c_voov(f,j,n,a) - l_amp * t_amp
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                        x2c_voov(f,j,n,b) = x2c_voov(f,j,n,b) + l_amp * t_amp
                     end do
                     end if
                  end do
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,5/), nub, nub, nob, nob, nloc, n3bbb_t)
                  do idet = 1, n3bbb_l
                     l_amp = l3d_amps(idet)
                     a = l3d_excits(idet,1); b = l3d_excits(idet,2); c = l3d_excits(idet,3);
                     i = l3d_excits(idet,4); j = l3d_excits(idet,5); k = l3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        x2c_voov(e,k,n,b) = x2c_voov(e,k,n,b) + l_amp * t_amp
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        x2c_voov(e,k,n,a) = x2c_voov(e,k,n,a) - l_amp * t_amp
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        x2c_voov(e,k,n,c) = x2c_voov(e,k,n,c) - l_amp * t_amp
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        x2c_voov(e,i,n,b) = x2c_voov(e,i,n,b) + l_amp * t_amp
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        x2c_voov(e,i,n,a) = x2c_voov(e,i,n,a) - l_amp * t_amp
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        x2c_voov(e,i,n,c) = x2c_voov(e,i,n,c) - l_amp * t_amp
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        x2c_voov(e,j,n,b) = x2c_voov(e,j,n,b) - l_amp * t_amp
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        x2c_voov(e,j,n,a) = x2c_voov(e,j,n,a) + l_amp * t_amp
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        x2c_voov(e,j,n,c) = x2c_voov(e,j,n,c) + l_amp * t_amp
                     end do
                     end if
                  end do
                  !!! BCIJ LOOP !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/1,nob-2/), (/-1,nob-1/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,5/), nub, nub, nob, nob, nloc, n3bbb_t)
                  do idet = 1, n3bbb_l
                     l_amp = l3d_amps(idet)
                     a = l3d_excits(idet,1); b = l3d_excits(idet,2); c = l3d_excits(idet,3);
                     i = l3d_excits(idet,4); j = l3d_excits(idet,5); k = l3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); n = excits_buff(jdet,6);
                        x2c_voov(d,k,n,a) = x2c_voov(d,k,n,a) + l_amp * t_amp
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); n = excits_buff(jdet,6);
                        x2c_voov(d,k,n,b) = x2c_voov(d,k,n,b) - l_amp * t_amp
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); n = excits_buff(jdet,6);
                        x2c_voov(d,k,n,c) = x2c_voov(d,k,n,c) + l_amp * t_amp
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); n = excits_buff(jdet,6);
                        x2c_voov(d,i,n,a) = x2c_voov(d,i,n,a) + l_amp * t_amp
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); n = excits_buff(jdet,6);
                        x2c_voov(d,i,n,b) = x2c_voov(d,i,n,b) - l_amp * t_amp
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); n = excits_buff(jdet,6);
                        x2c_voov(d,i,n,c) = x2c_voov(d,i,n,c) + l_amp * t_amp
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); n = excits_buff(jdet,6);
                        x2c_voov(d,j,n,a) = x2c_voov(d,j,n,a) - l_amp * t_amp
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); n = excits_buff(jdet,6);
                        x2c_voov(d,j,n,b) = x2c_voov(d,j,n,b) + l_amp * t_amp
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); n = excits_buff(jdet,6);
                        x2c_voov(d,j,n,c) = x2c_voov(d,j,n,c) - l_amp * t_amp
                     end do
                     end if
                  end do
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,6/), nub, nub, nob, nob, nloc, n3bbb_t)
                  do idet = 1, n3bbb_l
                     l_amp = l3d_amps(idet)
                     a = l3d_excits(idet,1); b = l3d_excits(idet,2); c = l3d_excits(idet,3);
                     i = l3d_excits(idet,4); j = l3d_excits(idet,5); k = l3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        x2c_voov(f,j,m,c) = x2c_voov(f,j,m,c) + l_amp * t_amp
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        x2c_voov(f,j,m,a) = x2c_voov(f,j,m,a) + l_amp * t_amp
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        x2c_voov(f,j,m,b) = x2c_voov(f,j,m,b) - l_amp * t_amp
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        x2c_voov(f,i,m,c) = x2c_voov(f,i,m,c) - l_amp * t_amp
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        x2c_voov(f,i,m,a) = x2c_voov(f,i,m,a) - l_amp * t_amp
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        x2c_voov(f,i,m,b) = x2c_voov(f,i,m,b) + l_amp * t_amp
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        x2c_voov(f,k,m,c) = x2c_voov(f,k,m,c) - l_amp * t_amp
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        x2c_voov(f,k,m,a) = x2c_voov(f,k,m,a) - l_amp * t_amp
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        x2c_voov(f,k,m,b) = x2c_voov(f,k,m,b) + l_amp * t_amp
                     end do
                     end if
                  end do
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,6/), nub, nub, nob, nob, nloc, n3bbb_t)
                  do idet = 1, n3bbb_l
                     l_amp = l3d_amps(idet)
                     a = l3d_excits(idet,1); b = l3d_excits(idet,2); c = l3d_excits(idet,3);
                     i = l3d_excits(idet,4); j = l3d_excits(idet,5); k = l3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        x2c_voov(e,j,m,b) = x2c_voov(e,j,m,b) + l_amp * t_amp
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        x2c_voov(e,j,m,a) = x2c_voov(e,j,m,a) - l_amp * t_amp
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        x2c_voov(e,j,m,c) = x2c_voov(e,j,m,c) - l_amp * t_amp
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        x2c_voov(e,i,m,b) = x2c_voov(e,i,m,b) - l_amp * t_amp
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        x2c_voov(e,i,m,a) = x2c_voov(e,i,m,a) + l_amp * t_amp
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        x2c_voov(e,i,m,c) = x2c_voov(e,i,m,c) + l_amp * t_amp
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        x2c_voov(e,k,m,b) = x2c_voov(e,k,m,b) - l_amp * t_amp
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        x2c_voov(e,k,m,a) = x2c_voov(e,k,m,a) + l_amp * t_amp
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        x2c_voov(e,k,m,c) = x2c_voov(e,k,m,c) + l_amp * t_amp
                     end do
                     end if
                  end do
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/1,nob-2/), (/-2,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,4,6/), nub, nub, nob, nob, nloc, n3bbb_t)
                  do idet = 1, n3bbb_l
                     l_amp = l3d_amps(idet)
                     a = l3d_excits(idet,1); b = l3d_excits(idet,2); c = l3d_excits(idet,3);
                     i = l3d_excits(idet,4); j = l3d_excits(idet,5); k = l3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        x2c_voov(d,j,m,a) = x2c_voov(d,j,m,a) + l_amp * t_amp
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        x2c_voov(d,j,m,b) = x2c_voov(d,j,m,b) - l_amp * t_amp
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        x2c_voov(d,j,m,c) = x2c_voov(d,j,m,c) + l_amp * t_amp
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        x2c_voov(d,i,m,a) = x2c_voov(d,i,m,a) - l_amp * t_amp
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        x2c_voov(d,i,m,b) = x2c_voov(d,i,m,b) + l_amp * t_amp
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        x2c_voov(d,i,m,c) = x2c_voov(d,i,m,c) - l_amp * t_amp
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        x2c_voov(d,k,m,a) = x2c_voov(d,k,m,a) - l_amp * t_amp
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        x2c_voov(d,k,m,b) = x2c_voov(d,k,m,b) + l_amp * t_amp
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        x2c_voov(d,k,m,c) = x2c_voov(d,k,m,c) - l_amp * t_amp
                     end do
                     end if
                  end do
                  !!! ABJK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,5,6/), nub, nub, nob, nob, nloc, n3bbb_t)
                  do idet = 1, n3bbb_l
                     l_amp = l3d_amps(idet)
                     a = l3d_excits(idet,1); b = l3d_excits(idet,2); c = l3d_excits(idet,3);
                     i = l3d_excits(idet,4); j = l3d_excits(idet,5); k = l3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); l = excits_buff(jdet,4);
                        x2c_voov(f,i,l,c) = x2c_voov(f,i,l,c) + l_amp * t_amp
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); l = excits_buff(jdet,4);
                        x2c_voov(f,i,l,a) = x2c_voov(f,i,l,a) + l_amp * t_amp
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); l = excits_buff(jdet,4);
                        x2c_voov(f,i,l,b) = x2c_voov(f,i,l,b) - l_amp * t_amp
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); l = excits_buff(jdet,4);
                        x2c_voov(f,j,l,c) = x2c_voov(f,j,l,c) - l_amp * t_amp
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); l = excits_buff(jdet,4);
                        x2c_voov(f,j,l,a) = x2c_voov(f,j,l,a) - l_amp * t_amp
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); l = excits_buff(jdet,4);
                        x2c_voov(f,j,l,b) = x2c_voov(f,j,l,b) + l_amp * t_amp
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); l = excits_buff(jdet,4);
                        x2c_voov(f,k,l,c) = x2c_voov(f,k,l,c) + l_amp * t_amp
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); l = excits_buff(jdet,4);
                        x2c_voov(f,k,l,a) = x2c_voov(f,k,l,a) + l_amp * t_amp
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); l = excits_buff(jdet,4);
                        x2c_voov(f,k,l,b) = x2c_voov(f,k,l,b) - l_amp * t_amp
                     end do
                     end if
                  end do
                  !!! ACJK LOOP !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,5,6/), nub, nub, nob, nob, nloc, n3bbb_t)
                  do idet = 1, n3bbb_l
                     l_amp = l3d_amps(idet)
                     a = l3d_excits(idet,1); b = l3d_excits(idet,2); c = l3d_excits(idet,3);
                     i = l3d_excits(idet,4); j = l3d_excits(idet,5); k = l3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); l = excits_buff(jdet,4);
                        x2c_voov(e,i,l,b) = x2c_voov(e,i,l,b) + l_amp * t_amp
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); l = excits_buff(jdet,4);
                        x2c_voov(e,i,l,a) = x2c_voov(e,i,l,a) - l_amp * t_amp
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); l = excits_buff(jdet,4);
                        x2c_voov(e,i,l,c) = x2c_voov(e,i,l,c) - l_amp * t_amp
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); l = excits_buff(jdet,4);
                        x2c_voov(e,j,l,b) = x2c_voov(e,j,l,b) - l_amp * t_amp
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); l = excits_buff(jdet,4);
                        x2c_voov(e,j,l,a) = x2c_voov(e,j,l,a) + l_amp * t_amp
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); l = excits_buff(jdet,4);
                        x2c_voov(e,j,l,c) = x2c_voov(e,j,l,c) + l_amp * t_amp
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); l = excits_buff(jdet,4);
                        x2c_voov(e,k,l,b) = x2c_voov(e,k,l,b) + l_amp * t_amp
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); l = excits_buff(jdet,4);
                        x2c_voov(e,k,l,a) = x2c_voov(e,k,l,a) - l_amp * t_amp
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        e = excits_buff(jdet,2); l = excits_buff(jdet,4);
                        x2c_voov(e,k,l,c) = x2c_voov(e,k,l,c) - l_amp * t_amp
                     end do
                     end if
                  end do
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/2,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,5,6/), nub, nub, nob, nob, nloc, n3bbb_t)
                  do idet = 1, n3bbb_l
                     l_amp = l3d_amps(idet)
                     a = l3d_excits(idet,1); b = l3d_excits(idet,2); c = l3d_excits(idet,3);
                     i = l3d_excits(idet,4); j = l3d_excits(idet,5); k = l3d_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                        ! < ijkabc | ljkdbc >
                        x2c_voov(d,i,l,a) = x2c_voov(d,i,l,a) + l_amp * t_amp
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                        ! < ijkabc | ljkdac >
                        x2c_voov(d,i,l,b) = x2c_voov(d,i,l,b) - l_amp * t_amp
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                        x2c_voov(d,i,l,c) = x2c_voov(d,i,l,c) + l_amp * t_amp
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                        x2c_voov(d,j,l,a) = x2c_voov(d,j,l,a) - l_amp * t_amp
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                        x2c_voov(d,j,l,b) = x2c_voov(d,j,l,b) + l_amp * t_amp
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                        x2c_voov(d,j,l,c) = x2c_voov(d,j,l,c) - l_amp * t_amp
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                        x2c_voov(d,k,l,a) = x2c_voov(d,k,l,a) + l_amp * t_amp
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                        x2c_voov(d,k,l,b) = x2c_voov(d,k,l,b) - l_amp * t_amp
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        d = excits_buff(jdet,1); l = excits_buff(jdet,4);
                        x2c_voov(d,k,l,c) = x2c_voov(d,k,l,c) + l_amp * t_amp
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table,excits_buff,amps_buff)
                  end if
                  !!!! x2c(eima) <- 1/4 l3b(abcijk) t3b(abeijm)
                  if (do_aab_t) then
                  ! copy t3b into buffer
                  allocate(amps_buff(n3aab_t),excits_buff(n3aab_t,6))
                  amps_buff(:) = t3b_amps(:)
                  excits_buff(:,:) = t3b_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = noa*(noa-1)/2*nua*(nua-1)/2
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nua,nua,noa,noa))
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,5/), nua, nua, noa, noa, nloc, n3aab_t)
                  do idet = 1, n3aab_l
                     l_amp = l3b_amps(idet)
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); c = l3b_excits(idet,3);
                     i = l3b_excits(idet,4); j = l3b_excits(idet,5); k = l3b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,i,j)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                        x2c_voov(f,k,n,c) = x2c_voov(f,k,n,c) + l_amp * t_amp
                     end do
                  end do 
                  deallocate(loc_arr,idx_table,excits_buff,amps_buff)
                  end if
                  !!!! x2c(eima) <- l3c(abcijk) t3c(abeijm)
                  if (do_abb_t) then
                  ! copy t3c into buffer
                  allocate(amps_buff(n3aab_t),excits_buff(n3abb_t,6))
                  amps_buff(:) = t3c_amps(:)
                  excits_buff(:,:) = t3c_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = noa*nob*nua*nub
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nua,nub,noa,nob))
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/1,nub-1/), (/1,noa/), (/1,nob-1/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,5/), nua, nub, noa, nob, nloc, n3abb_t)
                  do idet = 1, n3abb_l
                     l_amp = l3c_amps(idet)
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     i = l3c_excits(idet,4); j = l3c_excits(idet,5); k = l3c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,i,j)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                        x2c_voov(f,k,n,c) = x2c_voov(f,k,n,c) + l_amp * t_amp
                     end do
                     ! (bc)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                           x2c_voov(f,k,n,b) = x2c_voov(f,k,n,b) - l_amp * t_amp
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                           x2c_voov(f,j,n,c) = x2c_voov(f,j,n,c) - l_amp * t_amp
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,6);
                           x2c_voov(f,j,n,b) = x2c_voov(f,j,n,b) + l_amp * t_amp
                        end do
                     end if
                  end do 
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/2,nub/), (/1,noa/), (/1,nob-1/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,5/), nua, nub, noa, nob, nloc, n3abb_t)
                  do idet = 1, n3abb_l
                     l_amp = l3c_amps(idet)
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     i = l3c_excits(idet,4); j = l3c_excits(idet,5); k = l3c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,i,j)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        x2c_voov(f,k,n,b) = x2c_voov(f,k,n,b) + l_amp * t_amp
                     end do
                     ! (bc)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           f = excits_buff(jdet,2); n = excits_buff(jdet,6);
                           x2c_voov(f,k,n,c) = x2c_voov(f,k,n,c) - l_amp * t_amp
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           f = excits_buff(jdet,2); n = excits_buff(jdet,6);
                           x2c_voov(f,j,n,b) = x2c_voov(f,j,n,b) - l_amp * t_amp
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           f = excits_buff(jdet,2); n = excits_buff(jdet,6);
                           x2c_voov(f,j,n,c) = x2c_voov(f,j,n,c) + l_amp * t_amp
                        end do
                     end if
                  end do 
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/1,nub-1/), (/1,noa/), (/2,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,4,6/), nua, nub, noa, nob, nloc, n3abb_t)
                  do idet = 1, n3abb_l
                     l_amp = l3c_amps(idet)
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     i = l3c_excits(idet,4); j = l3c_excits(idet,5); k = l3c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,i,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                        x2c_voov(f,j,n,c) = x2c_voov(f,j,n,c) + l_amp * t_amp
                     end do
                     ! (bc)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                           x2c_voov(f,j,n,b) = x2c_voov(f,j,n,b) - l_amp * t_amp
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                           x2c_voov(f,k,n,c) = x2c_voov(f,k,n,c) - l_amp * t_amp
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                           x2c_voov(f,k,n,b) = x2c_voov(f,k,n,b) + l_amp * t_amp
                        end do
                     end if
                  end do 
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nua/), (/2,nub/), (/1,noa/), (/2,nob/), nua, nub, noa, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,6/), nua, nub, noa, nob, nloc, n3abb_t)
                  do idet = 1, n3abb_l
                     l_amp = l3c_amps(idet)
                     a = l3c_excits(idet,1); b = l3c_excits(idet,2); c = l3c_excits(idet,3);
                     i = l3c_excits(idet,4); j = l3c_excits(idet,5); k = l3c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        t_amp = amps_buff(jdet)
                        f = excits_buff(jdet,2); n = excits_buff(jdet,5);
                        x2c_voov(f,j,n,b) = x2c_voov(f,j,n,b) + l_amp * t_amp
                     end do
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           f = excits_buff(jdet,2); n = excits_buff(jdet,5);
                           x2c_voov(f,j,n,c) = x2c_voov(f,j,n,c) - l_amp * t_amp
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           f = excits_buff(jdet,2); n = excits_buff(jdet,5);
                           x2c_voov(f,k,n,b) = x2c_voov(f,k,n,b) - l_amp * t_amp
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           t_amp = amps_buff(jdet)
                           f = excits_buff(jdet,2); n = excits_buff(jdet,5);
                           x2c_voov(f,k,n,c) = x2c_voov(f,k,n,c) + l_amp * t_amp
                        end do
                     end if
                  end do 
                  deallocate(loc_arr,idx_table,excits_buff,amps_buff)
                  end if

              end subroutine compute_x2c_voov


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
              loc_arr(:,1) = 1; loc_arr(:,2) = 0; ! set default start > end so that empty sets do not trigger loops
              do idet = 1, n3p-1
                 ! get consecutive lexcial indices
                 p1 = excits(idet,idims(1));   q1 = excits(idet,idims(2));   r1 = excits(idet,idims(3));   s1 = excits(idet,idims(4))
                 p2 = excits(idet+1,idims(1)); q2 = excits(idet+1,idims(2)); r2 = excits(idet+1,idims(3)); s2 = excits(idet+1,idims(4))
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
