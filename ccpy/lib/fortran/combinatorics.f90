module combinatorics

  use constants, only: p
  ! Various utilities and tools...
  implicit none

  contains

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

  elemental function binom_i(m, n) result(binom)

    ! ACM Algorithm 160 translated to Fortran.
    ! Returns the binomial coefficient ^mC_n: the number of
    ! combinations of m things taken n at a time.

    integer :: binom
    integer, intent(in) :: m, n
    integer :: pool, i, n1

    n1 = n
    pool = m - n1
    if (n1 < pool) then
      pool = n1
      n1 = m - pool
    end if
    binom = n1 + 1
    if (pool == 0) binom = 1
    do i = 2, pool
      binom = (binom*(n1+i))/i
    end do

  end function binom_i

  elemental function binom_r(m, n) result(binom)

    ! ACM Algorithm 160 translated to Fortran.
    ! Returns the binomial coefficient ^mC_n: the number of
    ! combinations of m things taken n at a time.

    !use constants, only: p

    real(p) :: binom
    integer, intent(in) :: m, n
    integer :: pool, i, n1

    n1 = n
    pool = m - n1
    if (n1 < pool) then
      pool = n1
      n1 = m - pool
    end if
    binom = n1 + 1
    if (pool == 0) binom = 1
    do i = 2, pool
      binom = (binom*(n1+i))/i
    end do

  end function binom_r

  elemental function factorial(n) result(fac)

    ! In:
    !   n: integer
    ! Returns:
    !   factorial of n, n!

    ! WARNING:
    ! This does *not* safeguard against integer overflow.  As such, it is
    ! only suitably for 0 <= n <= 12.  Investigating using log(n!) or the
    ! Gamma function if required for larger values.

    integer :: fac
    integer, intent(in) :: n

    integer :: i

    fac = 1
    do i = 2, n
      fac = fac*i
    end do

  end function factorial

  elemental function factorial_combination_1(m,n) result (combination)

    ! Given m and n input, this function returns
    ! combination = n!m!/(n+m+1)!
    ! Required to calculate amplitudes of the Neel singlet
    ! trial function for the Heisenberg model

    use constants, only: p

    real(p) :: combination
    integer, intent(in) :: m, n
    integer :: m1, n1, i

    ! Choose m1 to be the larger of m and n
    if (m >= n) then
      m1 = m
      n1 = n
    else
      m1 = n
      n1 = m
    end if
    combination = 1
    do i = 1, n1
      combination = combination * i/(i+m1)
    end do
    combination = combination/(m1+n1+1)

  end function factorial_combination_1

  pure subroutine next_combination(nitems, inds, r, done)

    ! Generate the next combination of a selection of r items
    ! in the inds array. This subroutine was translated from
    ! python's itertools example.
    ! https://docs.python.org/3.7/library/itertools.html#itertools.combinations

    ! In:
    !    nitems: total number of items
    !    r: subset of items to choose
    ! In/Out:
    !    inds: index combination
    !    done: whether the last combination has been reached

    integer, intent(in) :: nitems
    integer, intent(inout) :: inds(:)
    integer, intent(in) :: r
    logical, intent(out) :: done

    integer :: i, j

    done = .false.

    do i=r, 1, -1
      if (inds(i) /= i + nitems - r) exit

      if (i == 1) then
        done = .true.
        return
      endif
    enddo

    inds(i) = inds(i) + 1
    do j=i+1, r
      inds(j) = inds(j-1) + 1
    enddo

  end subroutine next_combination

  !subroutine combinations(items, r, comb)

  !  integer, intent(in) :: items(:)
  !  integer, intent(in) :: r
  !  integer, allocatable, intent(inout) :: comb(:,:)

  !  integer, allocatable :: inds(:)
  !  integer :: n
  !  integer :: i, j, k, indx
  !  integer :: n_combs

  !  n = size(items)

  !  n_combs = binom_i(n, r)

  !  allocate(comb(r, n_combs))

  !  allocate(inds(n))

  !  do i=1, r
  !    inds(i) = i
  !  enddo

  !  indx = 1
  !  do k=1, r
  !    comb(k,indx) = items(inds(k))
  !  enddo

  !  outer: do
  !    do i=r, 1, -1
  !      if (inds(i) /= i + n - r) exit
  !      if (i == 1) exit outer
  !    enddo
  !    inds(i) = inds(i) + 1
  !    do j=i+1, r
  !      inds(j) = inds(j-1) + 1
  !    enddo

  !    indx = indx + 1
  !    do k=1, r
  !      comb(k,indx) = items(inds(k))
  !    enddo
  !  enddo outer

  !  deallocate(inds)

  !end subroutine combinations


end module combinatorics
