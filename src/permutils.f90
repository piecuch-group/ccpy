module permutils

    implicit none

    contains


    subroutine reorder_stripe(rank, mat_shape, mat_size, perm_str, A, B)

        integer, intent(in) :: rank
        integer, intent(in) :: mat_shape(:)
        integer, intent(in) :: mat_size
        character(len=*), intent(in) :: perm_str

        real, intent(in) :: A(mat_size)
        real, intent(inout) :: B(mat_size)

        integer :: perm(rank)

        integer :: ident(rank)

        integer :: stride_a(rank)
        integer :: stride_a_inner
        integer :: size_outer, size_inner
        integer :: offset_a, offset_b

        integer :: i, j, j_tmp
        integer :: current_index

        call gen_perm_array(perm_str, perm)

        do i=1, rank
            ident(i) = i
        enddo

        if (all(ident == perm)) then
            B = A
            return
        endif

        stride_a(1) = 1
        do i=2, rank
            stride_a(i) = stride_a(i-1) * mat_shape(i-1)
        enddo

        size_outer = 1
        do i=1, rank
            if (i /= perm(1)) size_outer = size_outer * mat_shape(i)
        enddo

        size_inner = mat_shape(perm(1))
        do j=0, size_outer-1
            offset_a = 0

            j_tmp = j

            do i=2, rank

                current_index = modulo(j_tmp, mat_shape(perm(i)))
                j_tmp = j_tmp / mat_shape(perm(i))
                offset_a = offset_a + (current_index * stride_a(perm(i)))

            enddo

            offset_b = j * size_inner
            stride_a_inner = stride_a(perm(1))

            do i=0, size_inner-1
                b(offset_b + i + 1) = a(offset_a + (i * stride_a_inner) + 1)
            enddo


        enddo

    end subroutine reorder_stripe

    subroutine gen_perm_array(perm_str, perm_array)

        character(len=*), intent(in) :: perm_str
        integer, intent(in out) :: perm_array(:)

        integer :: perm_rank
        integer :: i, idx

        perm_rank = len_trim(perm_str)

        do i=1, perm_rank
            idx = iachar(perm_str(i:i)) - 48
            perm_array(i) = idx
        enddo

    end subroutine gen_perm_array

    subroutine permsign(sgn,p)

        integer, intent(in) :: p(:)
        real, intent(out) :: sgn
        integer :: k, ct, L, n
        integer, allocatable :: visited(:)

        n = size(p)
        allocate(visited(1:n))

        do k = 1,n
           visited(k) = 0
        end do

        sgn = 1.0
        do k = 1,n
          if (visited(k) == 0) then
             ct = k
             L = 0
             do while (visited(ct) == 0) 
                L = L + 1
                visited(ct) = 1
                ct = p(ct)
             end do 
             if (mod(L,2) == 0) then
                sgn = -1.0 * sgn
             end if
          end if
        end do

        deallocate(visited)

    end subroutine permsign


end module permutils
