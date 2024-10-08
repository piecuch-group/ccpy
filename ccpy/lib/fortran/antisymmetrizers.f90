module antisymmetrizers

  use const, only: p 

  implicit none

contains

  subroutine t1(t1_operator, residual, x1, f_oo, f_vv, shift, no, nu)
    integer, intent(in) :: no, nu  ! array sizes (occupied, unnocupied)
    real(p), intent(in) :: shift  ! shift energy
    real(p), intent(in) :: f_oo(1:no, 1:no), f_vv(1:nu, 1:nu)  ! fock operator
    real(p), intent(in) :: x1(1:nu, 1:no)  ! cluster intermediate
    real(p), intent(out) :: residual(1:nu, 1:no)  ! residuals
    real(p), intent(in out) :: t1_operator(1:nu, 1:no)  ! cluster operator

    integer :: i, a
    real(p) :: denominator, residual_value

    do i=1, no
    do a=1, nu
        denominator = f_oo(i, i) - f_vv(a, a)
        residual_value = x1(a, i) / (denominator - shift)
        t1_operator(a, i) = t1_operator(a, i) + residual_value
        residual(a, i) = residual_value
    end do
    end do

  end subroutine t1

  subroutine t2_triplet(t2_operator, residual, x2, f_oo, f_vv, shift, no, nu)
    integer, intent(in) :: no, nu  ! array sizes (occupied, unnocupied)
    real(p), intent(in) :: shift  ! shift energy
    real(p), intent(in) :: f_oo(1:no, 1:no), f_vv(1:nu, 1:nu)  ! fock operator
    real(p), intent(in) :: x2(1:nu, 1:nu, 1:no, 1:no)  ! cluster intermediate
    real(p), intent(out) :: residual(1:nu, 1:nu, 1:no, 1:no)  ! residuals
    real(p), intent(in out) :: t2_operator(1:nu, 1:nu, 1:no, 1:no)  ! cluster operator

    integer :: i, j, a, b
    real(p) :: denominator, residual_value

    do i=1, no
    do j=i+1, no
    do a=1, nu
    do b=a+1, nu
        denominator = f_oo(i, i) + f_oo(j, j) - f_vv(a, a) - f_vv(b, b)
        residual_value = x2(b, a, j, i) / (denominator - shift)
        t2_operator(b, a, j, i) = t2_operator(b, a, j, i) + residual_value
        t2_operator(a, b, j, i) = -t2_operator(b, a, j, i)
        t2_operator(b, a, i, j) = -t2_operator(b, a, j, i)
        t2_operator(a, b, i, j) = t2_operator(b, a, j, i)

        residual(b, a, j, i) = residual_value
        residual(a, b, j, i) = -residual_value
        residual(b, a, i, j) = -residual_value
        residual(a, b, i, j) = residual_value
    end do
    end do
    end do
  end do

  end subroutine t2_triplet

  subroutine t2_singlet(t2_operator, residual, x2, fA_oo, fA_vv, fB_oo, fB_vv, shift, noa, nua, nob, nub)
    integer, intent(in) :: noa, nua, nob, nub  ! array sizes (occupied, unnocupied)
    real(p), intent(in) :: shift  ! shift energy
    real(p), intent(in) :: fA_oo(1:noa, 1:noa), fA_vv(1:nua, 1:nua)  ! fock operator
    real(p), intent(in) :: fB_oo(1:nob, 1:nob), fB_vv(1:nub, 1:nub)  ! fock operator
    real(p), intent(in) :: x2(1:nua, 1:nub, 1:noa, 1:nob)  ! cluster intermediate
    real(p), intent(out) :: residual(1:nu, 1:nu, 1:no, 1:no)  ! residuals
    real(p), intent(in out) :: t2_operator(1:nu, 1:nu, 1:no, 1:no)  ! cluster operator

    integer :: i, j, a, b
    real(p) :: denominator, residual_value

    do j=1, nob
    do i=1, noa
    do b=1, nub
    do a=1, nua
        denominator = fA_oo(i, i) + fB_oo(j, j) - fA_vv(a, a) - fB_vv(b, b)
        residual_value = x2(a, b, i, j) / (denominator - shift)
        t2_operator(a, b, i, j) = t2_operator(a, b, i, j) + residual_value

        residual(a, b, i, j) = residual_value
    end do
    end do
    end do
  end do

  end subroutine t2_singlet

end module antisymmetrizers
