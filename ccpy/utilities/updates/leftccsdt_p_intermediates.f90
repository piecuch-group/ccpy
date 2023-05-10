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

                  integer, intent(inout) :: t3a_excits(6,n3aaa_t)
                  !f2py intent(in,out) :: t3a_excits(6,0:n3aaa_t-1)
                  real(kind=8), intent(inout) :: t3a_amps(n3aaa_t)
                  !f2py intent(in,out) :: t3a_amps(0:n3aaa_t-1)
                  integer, intent(inout) :: t3b_excits(6,n3aab_t)
                  !f2py intent(in,out) :: t3b_excits(6,0:n3aab_t-1)
                  real(kind=8), intent(inout) :: t3b_amps(n3aab_t)
                  !f2py intent(in,out) :: t3b_amps(0:n3aab_t-1)
                  integer, intent(inout) :: t3c_excits(6,n3abb_t)
                  !f2py intent(in,out) :: t3c_excits(6,0:n3abb_t-1)
                  real(kind=8), intent(inout) :: t3c_amps(n3abb_t)
                  !f2py intent(in,out) :: t3c_amps(0:n3abb_t-1)
                  
                  integer, intent(inout) :: l3a_excits(6,n3aaa_l)
                  !f2py intent(in,out) :: l3a_excits(6,0:n3aaa_l-1)
                  real(kind=8), intent(inout) :: l3a_amps(n3aaa_l)
                  !f2py intent(in,out) :: l3a_amps(0:n3aaa_l-1)
                  integer, intent(inout) :: l3b_excits(6,n3aab_l)
                  !f2py intent(in,out) :: l3b_excits(6,0:n3aab_l-1)
                  real(kind=8), intent(inout) :: l3b_amps(n3aab_l)
                  !f2py intent(in,out) :: l3b_amps(0:n3aab_l-1)
                  integer, intent(inout) :: l3c_excits(6,n3abb_l)
                  !f2py intent(in,out) :: l3c_excits(6,0:n3abb_l-1)
                  real(kind=8), intent(inout) :: l3c_amps(n3abb_l)
                  !f2py intent(in,out) :: l3c_amps(0:n3abb_l-1)

                  real(kind=8), intent(out) :: x1a_oo(noa,noa)
                  
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)

                  real(kind=8) :: t_amp, l_amp, lt_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx
			
				      !!!! X1A(mi) = 1/6 l3a(efgmno) t3a(efgino) -> X1A(im) = 1/6 l3a(abcijk) * t3a(abcmjk)
                  x1a_oo = 0.0d0
                  ! allocate new sorting arrays
                  allocate(loc_arr(nua*(nua-1)*(nua-2)/6*noa,2))
                  allocate(idx_table(nua,nua,nua,noa))
                  !!! ABCK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/3,noa/), nua, nua, nua, noa)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/1,2,3,6/), nua, nua, nua, noa, nua*(nua-1)*(nua-2)/6*noa, n3aaa_t)
                  do idet = 1, n3aaa_l
	                  l_amp = l3a_amps(idet)
                     a = l3a_excits(1,idet); b = l3a_excits(2,idet); c = l3a_excits(3,idet);
                     i = l3a_excits(4,idet); j = l3a_excits(5,idet); k = l3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
	                     t_amp = t3a_amps(jdet)
                        m = t3a_excits(4,jdet); n = t3a_excits(5,jdet);
	                     ! compute < ijkabc | mnkabc > ->
	                     ! N[i+j+k+cba a+b+c+knm] = delta(j,n)N[i+m] + delta(i,m) N[j+n] - delta(i,n)N[j+m] - delta(j,m) N[i+n]
	                     !                        = A(ij)A(nm) delta(i,m) N[j+n]
	                     lt_amp = l_amp * t_amp
                        if (i==m) x1a_oo(j,n) = x1a_oo(j,n) + 0.5d0 * lt_amp ! (1)
                        if (j==m) x1a_oo(i,n) = x1a_oo(i,n) - 0.5d0 * lt_amp ! (ij)
                        if (i==n) x1a_oo(j,m) = x1a_oo(j,m) - 0.5d0 * lt_amp ! (nm)
                        if (j==n) x1a_oo(i,m) = x1a_oo(i,m) + 0.5d0 * lt_amp ! (ij)(nm)
                     end do
                     ! (ik)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
	                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
		                     t_amp = t3a_amps(jdet)
	                        m = t3a_excits(4,jdet); n = t3a_excits(5,jdet);
		                     ! compute < ijkabc | mniabc > ->
		                     ! N[i+j+k+cba a+b+c+inm] = -A(jk)A(nm) delta(k,m) N[j+n]
		                     lt_amp = l_amp * t_amp
	                        if (k==m) x1a_oo(j,n) = x1a_oo(j,n) - 0.5d0 * lt_amp ! (1)
	                        if (j==m) x1a_oo(k,n) = x1a_oo(k,n) + 0.5d0 * lt_amp ! (jk)
	                        if (k==n) x1a_oo(j,m) = x1a_oo(j,m) + 0.5d0 * lt_amp ! (nm)
	                        if (j==n) x1a_oo(k,m) = x1a_oo(k,m) - 0.5d0 * lt_amp ! (jk)(nm)
	                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
	                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
		                     t_amp = t3a_amps(jdet)
	                        m = t3a_excits(4,jdet); n = t3a_excits(5,jdet);
		                     ! compute < ijkabc | mnjabc > ->
		                     ! N[i+j+k+cba a+b+c+jnm] = -A(ik)A(nm) delta(i,m) N[k+n]
		                     lt_amp = l_amp * t_amp
	                        if (i==m) x1a_oo(k,n) = x1a_oo(k,n) - 0.5d0 * lt_amp ! (1)
	                        if (k==m) x1a_oo(i,n) = x1a_oo(i,n) + 0.5d0 * lt_amp ! (ik)
	                        if (i==n) x1a_oo(k,m) = x1a_oo(k,m) + 0.5d0 * lt_amp ! (nm)
	                        if (k==n) x1a_oo(i,m) = x1a_oo(i,m) - 0.5d0 * lt_amp ! (ik)(nm)
	                     end do
                     end if
                  end do
                  !!! ABCI LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/1,noa-2/), nua, nua, nua, noa)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/1,2,3,4/), nua, nua, nua, noa, nua*(nua-1)*(nua-2)/6*noa, n3aaa_t)
                  do idet = 1, n3aaa_l
	                  l_amp = l3a_amps(idet)
                     a = l3a_excits(1,idet); b = l3a_excits(2,idet); c = l3a_excits(3,idet);
                     i = l3a_excits(4,idet); j = l3a_excits(5,idet); k = l3a_excits(6,idet);
                     ! (1)
	                  idx = idx_table(a,b,c,i)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
	                     t_amp = t3a_amps(jdet)
                        m = t3a_excits(5,jdet); n = t3a_excits(6,jdet);
	                     ! compute < ijkabc | imnabc > -> A(jk)A(mn) delta(j,m) N[k+n]
	                     lt_amp = l_amp * t_amp
                        if (j==m) x1a_oo(k,n) = x1a_oo(k,n) + 0.5d0 * lt_amp ! (1)
	                     if (k==m) x1a_oo(j,n) = x1a_oo(j,n) - 0.5d0 * lt_amp ! (jk)
                        if (j==n) x1a_oo(k,m) = x1a_oo(k,m) - 0.5d0 * lt_amp ! (mn)
                        if (k==n) x1a_oo(j,m) = x1a_oo(j,m) + 0.5d0 * lt_amp ! (jk)(mn)
                     end do
	                  ! (ij)
	                  idx = idx_table(a,b,c,j)
	                  if (idx/=0) then
	                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
		                     t_amp = t3a_amps(jdet)
	                        m = t3a_excits(5,jdet); n = t3a_excits(6,jdet);
		                     ! compute < ijkabc | jmnabc > -> -A(ik)A(mn) delta(i,m) N[k+n]
		                     lt_amp = l_amp * t_amp
	                        if (i==m) x1a_oo(k,n) = x1a_oo(k,n) - 0.5d0 * lt_amp ! (1)
		                     if (k==m) x1a_oo(i,n) = x1a_oo(i,n) + 0.5d0 * lt_amp ! (ik)
	                        if (i==n) x1a_oo(k,m) = x1a_oo(k,m) + 0.5d0 * lt_amp ! (mn)
	                        if (k==n) x1a_oo(i,m) = x1a_oo(i,m) - 0.5d0 * lt_amp ! (ik)(mn)
	                     end do
		               end if
		              ! (ik)
		              idx = idx_table(a,b,c,k)
		              if (idx/=0) then
				           do jdet = loc_arr(idx,1), loc_arr(idx,2)
		                     t_amp = t3a_amps(jdet)
	                        m = t3a_excits(5,jdet); n = t3a_excits(6,jdet);
		                     ! compute < ijkabc | kmnabc > -> -A(ij)A(mn) delta(j,m) N[i+n]
		                     lt_amp = l_amp * t_amp
	                        if (j==m) x1a_oo(i,n) = x1a_oo(i,n) - 0.5d0 * lt_amp ! (1)
		                     if (i==m) x1a_oo(j,n) = x1a_oo(j,n) + 0.5d0 * lt_amp ! (ij)
	                        if (j==n) x1a_oo(i,m) = x1a_oo(i,m) + 0.5d0 * lt_amp ! (mn)
	                        if (i==n) x1a_oo(j,m) = x1a_oo(j,m) - 0.5d0 * lt_amp ! (ij)(mn)
				           end do
	                 end if
	               end do
			         !!! ABCJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/2,noa-1/), nua, nua, nua, noa)
                  call sort4(t3a_excits, t3a_amps, loc_arr, idx_table, (/1,2,3,5/), nua, nua, nua, noa, nua*(nua-1)*(nua-2)/6*noa, n3aaa_t)
	               do idet = 1, n3aaa_l
	                  l_amp = l3a_amps(idet)
                     a = l3a_excits(1,idet); b = l3a_excits(2,idet); c = l3a_excits(3,idet);
                     i = l3a_excits(4,idet); j = l3a_excits(5,idet); k = l3a_excits(6,idet);
                     ! (1)
		               idx = idx_table(a,b,c,j)
		               do jdet = loc_arr(idx,1), loc_arr(idx,2)
			               t_amp = t3a_amps(jdet)
			               m = t3a_excits(4,jdet); n = t3a_excits(6,jdet);
			               ! compute < ijkabc | mjnabc > -> A(ik)A(mn) delta(k,n) N[i+m]
			               lt_amp = l_amp * t_amp
			               if (k==n) x1a_oo(i,m) = x1a_oo(i,m) + 0.5d0 * lt_amp ! (1)
			               if (i==n) x1a_oo(k,m) = x1a_oo(k,m) - 0.5d0 * lt_amp ! (ik)
			               if (k==m) x1a_oo(i,n) = x1a_oo(i,n) - 0.5d0 * lt_amp ! (mn)
			               if (i==m) x1a_oo(k,n) = x1a_oo(k,n) + 0.5d0 * lt_amp ! (ik)(mn)
	                  end do
		               ! (ij)
		               idx = idx_table(a,b,c,i)
	                  if (idx/=0) then
			               do jdet = loc_arr(idx,1), loc_arr(idx,2)
				               t_amp = t3a_amps(jdet)
				               m = t3a_excits(4,jdet); n = t3a_excits(6,jdet);
				               ! compute < ijkabc | minabc > -> -A(jk)A(mn) delta(k,n) N[j+m]
				               lt_amp = l_amp * t_amp
				               if (k==n) x1a_oo(j,m) = x1a_oo(j,m) - 0.5d0 * lt_amp ! (1)
				               if (j==n) x1a_oo(k,m) = x1a_oo(k,m) + 0.5d0 * lt_amp ! (jk)
				               if (k==m) x1a_oo(j,n) = x1a_oo(j,n) + 0.5d0 * lt_amp ! (mn)
				               if (j==m) x1a_oo(k,n) = x1a_oo(k,n) - 0.5d0 * lt_amp ! (jk)(mn)
		                  end do
		               end if
		               ! (jk)
		               idx = idx_table(a,b,c,k)
		               if (idx/=0) then
				            do jdet = loc_arr(idx,1), loc_arr(idx,2)
				               t_amp = t3a_amps(jdet)
				               m = t3a_excits(4,jdet); n = t3a_excits(6,jdet);
				               ! compute < ijkabc | mknabc > -> -A(ij)A(mn) delta(j,n) N[i+m]
				               lt_amp = l_amp * t_amp
				               if (j==n) x1a_oo(i,m) = x1a_oo(i,m) - 0.5d0 * lt_amp ! (1)
				               if (i==n) x1a_oo(j,m) = x1a_oo(j,m) + 0.5d0 * lt_amp ! (ij)
				               if (j==m) x1a_oo(i,n) = x1a_oo(i,n) + 0.5d0 * lt_amp ! (mn)
				               if (i==m) x1a_oo(j,n) = x1a_oo(j,n) - 0.5d0 * lt_amp ! (ij)(mn)
		                  end do
	                  end if
	               end do
		        end subroutine compute_x1a_oo
	
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