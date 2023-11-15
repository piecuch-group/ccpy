module reorder

        implicit none

        contains
           
              subroutine reorder_amplitudes(l3_amps, l3_excits, t3_excits, n3)
                 
                 integer, intent(in) :: n3
                 integer, intent(in) :: t3_excits(6,n3)

                 integer, intent(inout) :: l3_excits(6,n3)
                 !f2py intent(in,out) :: l3_excits(6,0:n3-1)
                 real(kind=8), intent(inout) :: l3_amps(n3)
                 !f2py intent(in,out) :: l3_amps(0:n3-1)
      
                 integer :: i, j, k, a, b, c, l, m, n, d, e, f, idet, jdet, tmp(6)
                 real(kind=8) :: l_amp
                 
                 do idet = 1, n3
                    a = t3_excits(1,idet); b = t3_excits(2,idet); c = t3_excits(3,idet);
                    i = t3_excits(4,idet); j = t3_excits(5,idet); k = t3_excits(6,idet);
                    do jdet = 1, n3
                       d = l3_excits(1,jdet); e = l3_excits(2,jdet); f = l3_excits(3,jdet);
                       l = l3_excits(4,jdet); m = l3_excits(5,jdet); n = l3_excits(6,jdet);
                       if (a==d .and. b==e .and. c==f .and. i==l .and. j==m .and. k==n) then
                          ! swap the values at l3(deflmn) with l3(abcijk)
                          l_amp = l3_amps(jdet)
                          l3_amps(jdet) = l3_amps(idet)
                          l3_amps(idet) = l_amp
                          ! also swap the corresponding entries in the l3_excits array
                          tmp(1) = l3_excits(1,jdet); tmp(2) = l3_excits(2,jdet); tmp(3) = l3_excits(3,jdet);
                          tmp(4) = l3_excits(4,jdet); tmp(5) = l3_excits(5,jdet); tmp(6) = l3_excits(6,jdet);
                          
                          l3_excits(1,jdet) = l3_excits(1,idet); l3_excits(2,jdet) = l3_excits(2,idet); l3_excits(3,jdet) = l3_excits(3,idet);
                          l3_excits(4,jdet) = l3_excits(4,idet); l3_excits(5,jdet) = l3_excits(5,idet); l3_excits(6,jdet) = l3_excits(6,idet);
                          
                          l3_excits(1,idet) = tmp(1); l3_excits(2,idet) = tmp(2); l3_excits(3,idet) = tmp(3);
                          l3_excits(4,idet) = tmp(4); l3_excits(5,idet) = tmp(5); l3_excits(6,idet) = tmp(6);
                       end if
                    end do
                 end do
                 
              end subroutine reorder_amplitudes

              subroutine reorder4(y, x, iorder)

                  integer, intent(in) :: iorder(4)
                  real(kind=8), intent(in) :: x(:,:,:,:)

                  real(kind=8), intent(out) :: y(:,:,:,:)

                  integer :: i, j, k, l
                  integer :: vec(4)

                  y = 0.0d0
                  do i = 1, size(x,1)
                     do j = 1, size(x,2)
                        do k = 1, size(x,3)
                           do l = 1, size(x,4)
                              vec = (/i,j,k,l/)
                              y(vec(iorder(1)),vec(iorder(2)),vec(iorder(3)),vec(iorder(4))) = x(i,j,k,l)
                           end do
                        end do
                     end do
                  end do

              end subroutine reorder4
            
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

!              subroutine get_index_table3(idx_table, rng1, rng2, rng3, n1, n2, n3)
!
!                    integer, intent(in) :: n1, n2, n3
!                    integer, intent(in) :: rng1(2), rng2(2), rng3(2)
!
!                    integer, intent(inout) :: idx_table(n1,n2,n3)
!                    !f2py intent(in,out) :: idx_table(0:n1-1,0:n2-1,0:n3-1)
!
!                    integer :: kout
!                    integer :: p, q, r
!
!                    idx_table = 0
!                    ! 5 possible cases. Always organize so that ordered indices appear first.
!                    if (rng1(1) < 0 .and. rng2(1) < 0 .and. rng3(1) < 0) then ! p < q < r
!                       kout = 1
!                       do p = rng1(1), rng1(2)
!                          do q = p-rng2(1), rng2(2)
!                             do r = q-rng3(1), rng3(2)
!                                idx_table(p,q,r) = kout
!                                kout = kout + 1
!                             end do
!                          end do
!                       end do
!                    elseif (rng1(1) > 0 .and. rng2(1) > 0 .and. rng3(1) < 0) then ! p, q < r
!                       kout = 1
!                       do p = rng1(1), rng1(2)
!                          do q = rng2(1), rng2(2)
!                             do r = q-rng3(1), rng3(2)
!                                idx_table(p,q,r) = kout
!                                kout = kout + 1
!                             end do
!                          end do
!                       end do
!                    elseif (rng1(1) > 0 .and. rng2(1) < 0 .and. rng3(1) > 0) then ! p < q, r
!                       kout = 1
!                       do p = rng1(1), rng1(2)
!                          do q = p-rng2(1), rng2(2)
!                             do r = rng3(1), rng3(2)
!                                idx_table(p,q,r) = kout
!                                kout = kout + 1
!                             end do
!                          end do
!                       end do
!                    else ! p, q, r
!                       kout = 1
!                       do p = rng1(1), rng1(2)
!                          do q = rng2(1), rng2(2)
!                             do r = rng3(1), rng3(2)
!                                idx_table(p,q,r) = kout
!                                kout = kout + 1
!                             end do
!                          end do
!                       end do
!                    end if
!
!              end subroutine get_index_table3
!
!              subroutine excitation_sort3(idx, loc_arr, excits, idx_table, idims, n1, n2, n3, nloc, n3p)
!
!                    integer, intent(in) :: n1, n2, n3, nloc, n3p
!                    integer, intent(in) :: idims(3)
!                    integer, intent(in) :: idx_table(n1,n2,n3)
!
!                    integer, intent(out) :: idx(n3p)
!                    integer, intent(out) :: loc_arr(2,nloc)
!                    integer, intent(inout) :: excits(6,n3p)
!                    !f2py intent(in,out) :: excits(0:5,0:n3p-1)
!
!                    integer :: idet
!                    integer :: p, q, r
!                    integer :: p1, q1, r1, p2, q2, r2
!                    integer :: pqr1, pqr2
!                    integer, allocatable :: temp(:)
!
!                    allocate(temp(n3p),idx(n3p))
!                    do idet = 1, n3p
!                       p = excits(idims(1),idet); q = excits(idims(2),idet); r = excits(idims(3),idet);
!                       temp(idet) = idx_table(p,q,r)
!                    end do
!                    call argsort(temp, idx)
!                    excits = excits(:,idx)
!                    deallocate(temp)
!
!                    loc_arr(1,:) = 1; loc_arr(2,:) = 0;
!                    do idet = 1, n3p-1
!                       p1 = excits(idims(1),idet);   q1 = excits(idims(2),idet);   r1 = excits(idims(3),idet);
!                       p2 = excits(idims(1),idet+1); q2 = excits(idims(2),idet+1); r2 = excits(idims(3),idet+1);
!                       pqrs1 = idx_table(p1,q1,r1)
!                       pqrs2 = idx_table(p2,q2,r2)
!                       if (pqr1 /= pqr2) then
!                          loc_arr(2,pqr1) = idet
!                          loc_arr(1,pqr2) = idet+1
!                       end if
!                    end do
!                    loc_arr(2,pqr2) = n3p
!
!              end subroutine excitation_sort3
!
!              subroutine argsort(r,d)
!
!                    integer, intent(in), dimension(:) :: r
!                    integer, intent(out), dimension(size(r)) :: d
!
!                    integer, dimension(size(r)) :: il
!
!                    integer :: stepsize
!                    integer :: i, j, n, left, k, ksize
!
!                    n = size(r)
!
!                    do i=1,n
!                       d(i)=i
!                    end do
!
!                    if (n==1) return
!
!                    stepsize = 1
!                    do while (stepsize < n)
!                       do left = 1, n-stepsize,stepsize*2
!                          i = left
!                          j = left+stepsize
!                          ksize = min(stepsize*2,n-left+1)
!                          k=1
!
!                          do while (i < left+stepsize .and. j < left+ksize)
!                             if (r(d(i)) < r(d(j))) then
!                                il(k) = d(i)
!                                i = i+1
!                                k = k+1
!                             else
!                                il(k) = d(j)
!                                j = j+1
!                                k = k+1
!                             endif
!                          enddo
!
!                          if (i < left+stepsize) then
!                             ! fill up remaining from left
!                             il(k:ksize) = d(i:left+stepsize-1)
!                          else
!                             ! fill up remaining from right
!                             il(k:ksize) = d(j:left+ksize-1)
!                          endif
!                          d(left:left+ksize-1) = il(1:ksize)
!                       end do
!                       stepsize = stepsize*2
!                    end do
!
!              end subroutine argsort
   
end module reorder
