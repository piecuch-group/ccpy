module reorder

  implicit none

  contains

  subroutine reorder_stripe(mat_rank, mat_shape, mat_size, permutation_string, A, B)

    use constants, only: p
    use combinatorics, only: gen_perm_array

    implicit none

    integer, intent(in) :: mat_rank
    integer, intent(in) :: mat_shape(mat_rank)
    integer, intent(in) :: mat_size
    character(len=*), intent(in) :: permutation_string

    real(p), intent(in) :: A(mat_size)
    real(p), intent(in out) :: B(mat_size)

    integer :: permutation(mat_rank)

    integer :: identity(mat_rank)

    integer :: stride_a(mat_rank)
    integer :: stride_a_inner
    integer :: size_outer, size_inner
    integer :: offset_a, offset_b

    integer :: i, j, j_tmp
    integer :: current_index

    call gen_perm_array(permutation_string, permutation)

    do i=1, mat_rank
      identity(i) = i
    enddo

    if (all(identity == permutation)) then
      B = A
      return
    endif

    stride_a(1) = 1
    do i=2, mat_rank
      stride_a(i) = stride_a(i-1) * mat_shape(i-1)
    enddo
    !print *, 'size a',  size(a)
    !print *, 'stide_a', stride_a
    !print *, 'mat_shape', mat_shape

    size_outer = 1
    do i=1, mat_rank
      if (i /= permutation(1)) size_outer = size_outer * mat_shape(i)
    enddo
    !print *, 'size_outer', size_outer

    ! Size of the fastest changing index
    size_inner = mat_shape(permutation(1))
    ! Size of the stride corresponding to this index in the A matrix
    stride_a_inner = stride_a(permutation(1))

    !$omp parallel default(none), &
    !$omp shared(size_outer, mat_rank, mat_shape, permutation, stride_a, &
    !$omp size_inner, stride_a_inner, a, b), &
    !$omp private(i, j, j_tmp, offset_a, offset_b, current_index)

    !$omp do
    do j=0, size_outer-1
      offset_a = 0

      j_tmp = j

      do i=2, mat_rank
        current_index = modulo(j_tmp, mat_shape(permutation(i)))
        j_tmp = j_tmp / mat_shape(permutation(i))
        offset_a = offset_a + (current_index * stride_a(permutation(i)))
      enddo

      offset_b = j * size_inner

      do i=0, size_inner-1
        b(offset_b + i + 1) = a(offset_a + (i * stride_a_inner) + 1)
      enddo


    enddo
    !$omp end do
    !$omp end parallel


  end subroutine reorder_stripe

  subroutine reorder_shift(mat_rank, a_shape, a_size, b_shape, b_size, b_shifts, permutation_string, A, B)

    use constants, only: p
    use combinatorics, only: gen_perm_array

    integer, intent(in) :: mat_rank
    integer, intent(in) :: a_shape(mat_rank)
    integer, intent(in) :: a_size
    integer, intent(in) :: b_shape(mat_rank)
    integer, intent(in) :: b_size
    integer, intent(in) :: b_shifts(mat_rank)
    character(len=*), intent(in) :: permutation_string

    real(p), intent(in) :: A(a_size)
    real(p), intent(in out) :: B(b_size)


    !integer :: identity(rank)
    integer :: permutation(mat_rank)

    integer :: new_shape(mat_rank)
    integer :: shifts(mat_rank)

    integer :: stride_a(mat_rank)

    integer :: stride_a_inner
    integer :: size_outer, size_inner
    integer :: shift_inner
    integer :: offset_a, offset_b

    integer :: i, j, j_tmp
    integer :: rank_indx

    call gen_perm_array(permutation_string, permutation)

    !identity = [ (i, i=1, rank) ]

    !if (all(identity == permutation)) then
    !    B = A
    !    return
    !endif

    call permute_array(mat_rank, permutation, b_shape, new_shape)
    call permute_array(mat_rank, permutation, b_shifts, shifts)

    ! Stride sizes per dimension
    stride_a(1) = 1
    do i=2, mat_rank
      stride_a(i) = stride_a(i-1) * a_shape(i-1)
    enddo

    !print *, '---------------------------------'
    !print *, "a_shape:  ", a_shape
    !print *, "b_shape:  ", b_shape
    !print *, "b_shifts: ", b_shifts
    !print *, "stride_a: ", stride_a
    !print *, "shifts:   ", shifts
    !print *, '---------------------------------'

    ! The idea here is to reshape the N-rank array into a 2-rank array
    ! where the inner_size corresponds to the left index (fastest changing)
    ! and outer_size, to the right index (slowest chaging)

    ! Size of the fastest changing index
    size_inner = new_shape(permutation(1))
    ! Size of the remaining dimension
    size_outer = product(new_shape) / size_inner
    ! Size of the stride corresponding to this index in the A matrix
    stride_a_inner = stride_a(permutation(1))
    shift_inner = shifts(permutation(1))

    !$omp parallel default(none), &
    !$omp shared(mat_rank, new_shape, shifts, permutation, &
    !$omp size_outer, size_inner, shift_inner, &
    !$omp stride_a, stride_a_inner, &
    !$omp a, b), &
    !$omp private(i, j, j_tmp, offset_a, offset_b, rank_indx)

    !$omp do
    do j=0, size_outer-1
      offset_a = 0

      j_tmp = j

      do i=2, mat_rank
        rank_indx = modulo(j_tmp, new_shape(permutation(i)))
        j_tmp = j_tmp / new_shape(permutation(i))
        offset_a = offset_a &
        + (rank_indx * stride_a(permutation(i))) &
        + (shifts(permutation(i)) * stride_a(permutation(i)))
      enddo
      offset_a = offset_a + (shift_inner * stride_a_inner)

      offset_b = j * size_inner
      do i=0, size_inner-1
        b(offset_b + i + 1) = a(offset_a + (i * stride_a_inner) + 1)
      enddo


    enddo
    !$omp end do

    !$omp end parallel


  end subroutine reorder_shift

  subroutine sum_stripe(mat_rank, mat_shape, mat_size, perm_str, beta, A, B)

    use constants, only: p
    use combinatorics, only: gen_perm_array

    integer, intent(in) :: mat_rank
    integer, intent(in) :: mat_shape(mat_rank)
    integer, intent(in) :: mat_size
    character(len=*), intent(in) :: perm_str
    real(p), intent(in) :: beta
    real(p), intent(in out) :: A(mat_size)
    real(p), intent(in) :: B(mat_size)

    integer :: perm(mat_rank)

    integer :: ident(mat_rank)

    integer :: stride_a(mat_rank)
    integer :: stride_a_inner
    integer :: size_outer, size_inner
    integer :: offset_a, offset_b

    integer :: i, j, j_tmp
    integer :: current_index

    call gen_perm_array(perm_str, perm)

    do i=1, mat_rank
      ident(i) = i
    enddo

    if (all(ident == perm)) then
      A = A + beta * B
      return
    endif

    stride_a(1) = 1
    do i=2, mat_rank
      stride_a(i) = stride_a(i-1) * mat_shape(i-1)
    enddo
    !print *, 'size a',  size(a)
    !print *, 'stide_a', stride_a
    !print *, 'mat_shape', mat_shape

    size_outer = 1
    do i=1, mat_rank
      if (i /= perm(1)) size_outer = size_outer * mat_shape(i)
    enddo
    !print *, 'size_outer', size_outer


    size_inner = mat_shape(perm(1))

    !$omp parallel default(none), &
    !$omp shared(size_outer, mat_rank, mat_shape, perm, stride_a, size_inner, a, b, beta), &
    !$omp private(i, j, j_tmp, offset_a, offset_b, current_index, stride_a_inner)

    !$omp do
    do j=0, size_outer-1
      offset_a = 0

      j_tmp = j

      do i=2, mat_rank

        current_index = modulo(j_tmp, mat_shape(perm(i)))
        j_tmp = j_tmp / mat_shape(perm(i))
        offset_a = offset_a + (current_index * stride_a(perm(i)))

      enddo

      offset_b = j * size_inner
      stride_a_inner = stride_a(perm(1))

      do i=0, size_inner-1
        a(offset_a + (i * stride_a_inner) + 1) = a(offset_a + (i * stride_a_inner) + 1) &
        + beta * b(offset_b + i + 1)
      enddo


    enddo
    !$omp end do

    !$omp end parallel


  end subroutine sum_stripe

  subroutine sum_shift(mat_rank, a_shape, a_size, b_shape, b_size, b_shifts, permutation_string, beta, A, B)

    use constants, only: P
    use combinatorics, only: gen_perm_array

    integer, intent(in) :: mat_rank
    integer, intent(in) :: a_shape(mat_rank)
    integer, intent(in) :: a_size
    integer, intent(in) :: b_shape(mat_rank)
    integer, intent(in) :: b_size
    integer, intent(in) :: b_shifts(mat_rank)
    character(len=*), intent(in) :: permutation_string
    real(p), intent(in) :: beta

    real(p), intent(in) :: A(a_size)
    real(p), intent(in out) :: B(b_size)


    !integer :: identity(rank)
    integer :: permutation(mat_rank)
    integer :: tmp_permutation(mat_rank)

    integer :: identity(mat_rank)

    integer :: new_shape(mat_rank)
    integer :: shifts(mat_rank)

    integer :: stride_a(mat_rank)

    integer :: stride_a_inner
    integer :: size_outer, size_inner
    integer :: shift_inner
    integer :: offset_a, offset_b

    integer :: i, j, j_tmp
    integer :: rank_indx

    call gen_perm_array(permutation_string, permutation)

    tmp_permutation = permutation
    identity = [ (i, i=1, mat_rank) ]
    call permute_array(mat_rank, tmp_permutation, identity, permutation)

    call permute_array(mat_rank, permutation, b_shape, new_shape)
    call permute_array(mat_rank, permutation, b_shifts, shifts)


    !permutation = tmp_permutation


    ! Stride sizes per dimension
    stride_a(1) = 1
    do i=2, mat_rank
      stride_a(i) = stride_a(i-1) * a_shape(i-1)
    enddo

    ! The idea here is to reshape the N-rank array into a 2-rank array
    ! where the inner_size corresponds to the left index (fastest changing)
    ! and outer_size, to the right index (slowest chaging)

    ! Size of the fastest changing index
    size_inner = new_shape(permutation(1))
    ! Size of the remaining dimension
    size_outer = product(new_shape) / size_inner
    ! Size of the stride corresponding to this index in the A matrix
    stride_a_inner = stride_a(permutation(1))
    shift_inner = shifts(permutation(1))

    !$omp parallel default(none), &
    !$omp shared(mat_rank, new_shape, shifts, permutation, &
    !$omp size_outer, size_inner, shift_inner, &
    !$omp stride_a, stride_a_inner, &
    !$omp beta, A, B), &
    !$omp private(i, j, j_tmp, offset_a, offset_b, rank_indx)

    !$omp do
    do j=0, size_outer-1
      offset_a = 0

      j_tmp = j

      do i=2, mat_rank
        rank_indx = modulo(j_tmp, new_shape(permutation(i)))
        j_tmp = j_tmp / new_shape(permutation(i))
        offset_a = offset_a &
        + (rank_indx * stride_a(permutation(i))) &
        + (shifts(permutation(i)) * stride_a(permutation(i)))
      enddo
      offset_a = offset_a + (shift_inner * stride_a_inner)

      offset_b = j * size_inner
      do i=0, size_inner-1
        B(offset_b + i + 1) = B(offset_b + i + 1) + beta * A(offset_a + (i * stride_a_inner) + 1)
      enddo


    enddo
    !$omp end do

    !$omp end parallel


  end subroutine sum_shift

  subroutine permute_array(mat_rank, permutation, A, B)
    integer, intent(in) :: mat_rank
    integer, intent(in) :: permutation(mat_rank)
    integer, intent(in) :: A(mat_rank)
    integer, intent(in out) :: B(mat_rank)

    integer :: i

    do i=1, mat_rank
      B(permutation(i)) = A(i)
    end do

  end subroutine permute_array

  subroutine inverse_permute_array(mat_rank, permutation, A, B)
    integer, intent(in) :: mat_rank
    integer, intent(in) :: permutation(mat_rank)
    integer, intent(in) :: A(mat_rank)
    integer, intent(in out) :: B(mat_rank)

    integer :: i

    do i=1, mat_rank
      B(i) = A(permutation(i))
    end do
  end subroutine inverse_permute_array

  subroutine reorder_amplitudes(l3_amps, l3_excits, t3_excits, n3)

    use constants, only: p

    integer, intent(in) :: n3
    integer, intent(in) :: t3_excits(6,n3)

    integer, intent(inout) :: l3_excits(6,n3)
    !f2py intent(in,out) :: l3_excits(6,0:n3-1)
    real(p), intent(inout) :: l3_amps(n3)
    !f2py intent(in,out) :: l3_amps(0:n3-1)

    integer :: i, j, k, a, b, c, l, m, n, d, e, f, idet, jdet, tmp(6)
    real(p) :: l_amp

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

    use constants, only: p

    integer, intent(in) :: iorder(4)
    real(p), intent(in) :: x(:,:,:,:)

    real(p), intent(out) :: y(:,:,:,:)

    integer :: i, j, k, l
    integer :: vec(4)

    y = 0.0_p
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

    use constants, only: p

    integer, intent(in) :: iorder(4)
    real(p), intent(in) :: y(:,:,:,:)

    real(p), intent(inout) :: x(:,:,:,:)

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
