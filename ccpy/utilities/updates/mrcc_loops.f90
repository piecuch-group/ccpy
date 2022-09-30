module mrcc_loops

  implicit none

  contains

      subroutine compute_coupling_t1a_pq(X1A, t1a_p, t1a_q, det_p, det_q, heff, cpq, noa, nua, nelec)

          integer, intent(in) :: noa, nua, nelec
          integer, intent(in) :: det_p(1:nelec), det_q(1:nelec)
          real(kind=8), intent(in) :: t1a_p(nua, noa), t1a_q(nua, noa)
          real(kind=8), intent(in) :: heff, cpq

          real(kind=8), intent(inout) :: X1A(1:nua, 1:noa)
          !f2py intent(in,out) :: X1A(0:nua-1,0:noa-1)

          integer :: i, a
          integer :: i_sp, a_sp
          real(kind=8) :: heffc

          heffc = heff * cpq

          do i = 1, noa
              do a = 1, nua
                  X1A(a, i) = X1A(a, i) - heffc * t1a_p(a, i)

                  i_sp = 2 * i - 1
                  a_sp = 2 * a - 1

                  if ( any(det_q == i_sp) .and. all(det_q /= a_sp) ) then
                      X1A(a, i) = X1A(a, i) + heffc * t1a_q(a, i)
                  end if

              end do
          end do

      end subroutine compute_coupling_t1a_pq

      subroutine compute_coupling_t1b_pq(X1B, t1b_p, t1b_q, det_p, det_q, heff, cpq, nob, nub, nelec)

          integer, intent(in) :: nob, nub, nelec
          integer, intent(in) :: det_p(1:nelec), det_q(1:nelec)
          real(kind=8), intent(in) :: t1b_p(nub, nob), t1b_q(nub, nob)
          real(kind=8), intent(in) :: heff, cpq

          real(kind=8), intent(inout) :: X1B(1:nub, 1:nob)
          !f2py intent(in,out) :: X1B(0:nub-1,0:nob-1)

          integer :: i, a
          integer :: i_sp, a_sp
          real(kind=8) :: heffc

          heffc = heff * cpq

          do i = 1, nob
              do a = 1, nub
                  X1B(a, i) = X1B(a, i) - heffc * t1b_p(a, i)

                  i_sp = 2 * i
                  a_sp = 2 * a

                  if ( any(det_q == i_sp) .and. all(det_q /= a_sp) ) then
                      X1B(a, i) = X1B(a, i) + heffc * t1b_q(a, i)
                  end if

              end do
          end do

      end subroutine compute_coupling_t1b_pq

      subroutine compute_coupling_t2a_pq(X2A, t1a_p, t2a_p, t1a_q, t2a_q, det_p, det_q, heff, cpq, noa, nua, nelec)

          integer, intent(in) :: noa, nua, nelec
          integer, intent(in) :: det_p(1:nelec), det_q(1:nelec)
          real(kind=8), intent(in) :: t1a_p(nua, noa), t2a_p(nua, nua, noa, noa), t1a_q(nua, noa), t2a_q(nua, nua, noa, noa)
          real(kind=8), intent(in) :: heff, cpq

          real(kind=8), intent(inout) :: X2A(1:nua, 1:nua, 1:noa, 1:noa)
          !f2py intent(in,out) :: X2A(0:nua-1,0:nua-1,0:noa-1,0:noa-1)

          integer :: i, j, a, b
          integer :: i_sp, j_sp, a_sp, b_sp
          real(kind=8) :: heffc

          heffc = heff * cpq

          do i = 1, noa
              do j = 1, noa
                  do a = 1, nua
                      do b = 1, nua

                          X2A(a, b, i, j) = X2A(a, b, i, j) - heffc * 0.25 * t2a_p(a, b, i, j)
                          X2A(a, b, i, j) = X2A(a, b, i, j) - heffc * 0.5 * t1a_p(a, i) * t1a_p(b, j)

                          i_sp = 2 * i - 1
                          j_sp = 2 * j - 1
                          a_sp = 2 * a - 1
                          b_sp = 2 * b - 1

                          if ( any(det_q == i_sp) .and. all(det_q /= a_sp)&
                              .and. any(det_q == j_sp) .and. all(det_q /= b_sp) ) then
                              X2A(a, b, i, j) = X2A(a, b, i, j) + heffc * 0.25 * t2a_q(a, b, i, j)
                              X2A(a, b, i, j) = X2A(a, b, i, j) + heffc * 0.5 * t1a_q(a, i) * t1a_q(b, j)
                              X2A(a, b, i, j) = X2A(a, b, i, j) - heffc * t1a_p(a, i) * t1a_q(b, j)
                          end if

                      end do
                  end do
              end do
          end do

      end subroutine compute_coupling_t2a_pq

      subroutine compute_coupling_t2b_pq(X2B, t1a_p, t1b_p, t2b_p, t1a_q, t1b_q, t2b_q, det_p, det_q, heff, cpq, noa, nua, nob, nub, nelec)

          integer, intent(in) :: noa, nua, nob, nub, nelec
          integer, intent(in) :: det_p(1:nelec), det_q(1:nelec)
          real(kind=8), intent(in) :: t1a_p(nua, noa), t1b_p(nub, nob), t2b_p(nua, nub, noa, nob),&
                                      t1a_q(nua, noa), t1b_q(nub, nob), t2b_q(nua, nub, noa, nob)
          real(kind=8), intent(in) :: heff, cpq

          real(kind=8), intent(inout) :: X2B(1:nua, 1:nub, 1:noa, 1:nob)
          !f2py intent(in,out) :: X2B(0:nua-1,0:nub-1,0:noa-1,0:nob-1)

          integer :: i, j, a, b
          integer :: i_sp, j_sp, a_sp, b_sp
          real(kind=8) :: heffc

          heffc = heff * cpq

          do i = 1, noa
              do j = 1, nob
                  do a = 1, nua
                      do b = 1, nub

                          X2B(a, b, i, j) = X2B(a, b, i, j) - heffc * t2b_p(a, b, i, j)
                          X2B(a, b, i, j) = X2B(a, b, i, j) - heffc * t1a_p(a, i) * t1a_p(b, j)

                          i_sp = 2 * i - 1
                          j_sp = 2 * j
                          a_sp = 2 * a - 1
                          b_sp = 2 * b

                          if ( any(det_q == i_sp) .and. all(det_q /= a_sp)&
                              .and. any(det_q == j_sp) .and. all(det_q /= b_sp) ) then
                              X2B(a, b, i, j) = X2B(a, b, i, j) + heffc * t2b_q(a, b, i, j)
                              X2B(a, b, i, j) = X2B(a, b, i, j) + heffc * t1a_q(a, i) * t1b_q(b, j)
                              X2B(a, b, i, j) = X2B(a, b, i, j) - heffc * t1a_p(a, i) * t1b_q(b, j)
                              X2B(a, b, i, j) = X2B(a, b, i, j) - heffc * t1b_p(b, j) * t1a_q(a, i)
                          end if

                      end do
                  end do
              end do
          end do

      end subroutine compute_coupling_t2b_pq

      subroutine compute_coupling_t2c_pq(X2C, t1b_p, t2c_p, t1b_q, t2c_q, det_p, det_q, heff, cpq, nob, nub, nelec)

          integer, intent(in) :: nob, nub, nelec
          integer, intent(in) :: det_p(1:nelec), det_q(1:nelec)
          real(kind=8), intent(in) :: t1b_p(nub, nob), t2c_p(nub, nub, nob, nob), t1b_q(nub, nob), t2c_q(nub, nub, nob, nob)
          real(kind=8), intent(in) :: heff, cpq

          real(kind=8), intent(inout) :: X2C(1:nub, 1:nub, 1:nob, 1:nob)
          !f2py intent(in,out) :: X2C(0:nub-1,0:nub-1,0:nob-1,0:nob-1)

          integer :: i, j, a, b
          integer :: i_sp, j_sp, a_sp, b_sp
          real(kind=8) :: heffc

          heffc = heff * cpq

          do i = 1, nob
              do j = 1, nob
                  do a = 1, nub
                      do b = 1, nub

                          X2C(a, b, i, j) = X2C(a, b, i, j) - heffc * 0.25 * t2c_p(a, b, i, j)
                          X2C(a, b, i, j) = X2C(a, b, i, j) - heffc * 0.5 * t1b_p(a, i) * t1b_p(b, j)

                          i_sp = 2 * i
                          j_sp = 2 * j
                          a_sp = 2 * a
                          b_sp = 2 * b

                          if ( any(det_q == i_sp) .and. all(det_q /= a_sp)&
                              .and. any(det_q == j_sp) .and. all(det_q /= b_sp) ) then
                              X2C(a, b, i, j) = X2C(a, b, i, j) + heffc * 0.25 * t2c_q(a, b, i, j)
                              X2C(a, b, i, j) = X2C(a, b, i, j) + heffc * 0.5 * t1b_q(a, i) * t1b_q(b, j)
                              X2C(a, b, i, j) = X2C(a, b, i, j) - heffc * t1b_p(a, i) * t1b_q(b, j)
                          end if

                      end do
                  end do
              end do
          end do

      end subroutine compute_coupling_t2c_pq

      subroutine update_t1a(t1a,resid,X1A,fA_oo,fA_vv,shift,noa,nua)

              implicit none

              integer, intent(in) :: noa, nua
              real(kind=8), intent(in) :: fA_oo(1:noa,1:noa), fA_vv(1:nua,1:nua), &
                                  X1A(1:nua,1:noa), shift
              real(kind=8), intent(inout) :: t1a(1:nua,1:noa)
              !f2py intent(in,out) :: t1a(0:nua-1,0:noa-1)
              real(kind=8), intent(out) :: resid(1:nua,1:noa)
              integer :: i, a
              real(kind=8) :: denom, val

              do i = 1,noa
                do a = 1,nua
                  denom = fA_oo(i,i) - fA_vv(a,a)
                  val = X1A(a,i)/(denom-shift)
                  t1a(a,i) = t1a(a,i) + val
                  resid(a,i) = val
                end do
              end do

      end subroutine update_t1a

      subroutine update_t1b(t1b,resid,X1B,fB_oo,fB_vv,shift,nob,nub)

              implicit none

              integer, intent(in) :: nob, nub
              real(8), intent(in) :: fB_oo(1:nob,1:nob), fB_vv(1:nub,1:nub), &
                                  X1B(1:nub,1:nob), shift
              real(8), intent(inout) :: t1b(1:nub,1:nob)
              !f2py intent(in,out) :: t1b(0:nub-1,0:nob-1)
              real(kind=8), intent(out) :: resid(1:nub,1:nob)
              integer :: i, a
              real(8) :: denom, val

              do i = 1,nob
                do a = 1,nub
                  denom = fB_oo(i,i) - fB_vv(a,a)
                  val = X1B(a,i)/(denom-shift)
                  t1b(a,i) = t1b(a,i) + val
                  resid(a,i) = val
                end do
              end do

      end subroutine update_t1b

      subroutine update_t2a(t2a,resid,X2A,fA_oo,fA_vv,shift,noa,nua)

              implicit none

              integer, intent(in) :: noa, nua
              real(8), intent(in) :: fA_oo(1:noa,1:noa), fA_vv(1:nua,1:nua), &
                                  X2A(1:nua,1:nua,1:noa,1:noa), shift
              real(8), intent(inout) :: t2a(1:nua,1:nua,1:noa,1:noa)
              !f2py intent(in,out) :: t2a(0:nua-1,0:nua-1,0:noa-1,0:noa-1)
              real(kind=8), intent(out) :: resid(1:nua,1:nua,1:noa,1:noa)
              integer :: i, j, a, b
              real(8) :: denom, val

              do i = 1,noa
                do j = i+1,noa
                  do a = 1,nua
                    do b = a+1,nua
                      denom = fA_oo(i,i) + fA_oo(j,j) - fA_vv(a,a) - fA_vv(b,b)
                      val = (X2A(b,a,j,i) - X2A(a,b,j,i) - X2A(b,a,i,j) + X2A(a,b,i,j))/(denom-shift)
                      t2a(b,a,j,i) = t2a(b,a,j,i) + val
                      t2a(a,b,j,i) = -t2a(b,a,j,i)
                      t2a(b,a,i,j) = -t2a(b,a,j,i)
                      t2a(a,b,i,j) = t2a(b,a,j,i)

                      resid(b,a,j,i) = val
                      resid(a,b,j,i) = -val
                      resid(b,a,i,j) = -val
                      resid(a,b,i,j) = val
                    end do
                  end do
                end do
              end do

      end subroutine update_t2a

      subroutine update_t2b(t2b,resid,X2B,fA_oo,fA_vv,fB_oo,fB_vv,shift,noa,nua,nob,nub)

              implicit none

              integer, intent(in) :: noa, nua, nob, nub
              real(8), intent(in) :: fA_oo(1:noa,1:noa), fA_vv(1:nua,1:nua), &
                                  fB_oo(1:nob,1:nob), fB_vv(1:nub,1:nub), &
                                  X2B(1:nua,1:nub,1:noa,1:nob), shift
              real(8), intent(inout) :: t2b(1:nua,1:nub,1:noa,1:nob)
              !f2py intent(in,out) :: t2b(0:nua-1,0:nub-1,0:noa-1,0:nob-1)
              real(kind=8), intent(out) :: resid(1:nua,1:nub,1:noa,1:nob)
              integer :: i, j, a, b
              real(8) :: denom, val

              do j = 1,nob
                do i = 1,noa
                  do b = 1,nub
                    do a = 1,nua
                      denom = fA_oo(i,i) + fB_oo(j,j) - fA_vv(a,a) - fB_vv(b,b)
                      val = X2B(a,b,i,j)/(denom-shift)
                      t2b(a,b,i,j) = t2b(a,b,i,j) + val
                      resid(a,b,i,j) = val
                    end do
                  end do
                end do
              end do

      end subroutine update_t2b

      subroutine update_t2c(t2c,resid,X2C,fB_oo,fB_vv,shift,nob,nub)

              implicit none

              integer, intent(in) :: nob, nub
              real(8), intent(in) :: fB_oo(1:nob,1:nob), fB_vv(1:nub,1:nub), &
                                  X2C(1:nub,1:nub,1:nob,1:nob), shift
              real(8), intent(inout) :: t2c(1:nub,1:nub,1:nob,1:nob)
              !f2py intent(in,out) :: t2c(0:nub-1,0:nub-1,0:nob-1,0:nob-1)
              real(kind=8), intent(out) :: resid(1:nub,1:nub,1:nob,1:nob)
              integer :: i, j, a, b
              real(8) :: denom, val

              do i = 1,nob
                do j = i+1,nob
                  do a = 1,nub
                    do b = a+1,nub
                      denom = fB_oo(i,i) + fB_oo(j,j) - fB_vv(a,a) - fB_vv(b,b)
                      !t2c(b,a,j,i) = t2c(b,a,j,i) + X2C(b,a,j,i)/(denom-shift)
                      val = (X2C(b,a,j,i) - X2C(a,b,j,i) - X2C(b,a,i,j) + X2C(a,b,i,j))/(denom-shift)
                      t2c(b,a,j,i) = t2c(b,a,j,i) + val
                      t2c(a,b,j,i) = -t2c(b,a,j,i)
                      t2c(b,a,i,j) = -t2c(b,a,j,i)
                      t2c(a,b,i,j) = t2c(b,a,j,i)

                      resid(b,a,j,i) = val
                      resid(a,b,j,i) = -val
                      resid(b,a,i,j) = -val
                      resid(a,b,i,j) = val
                    end do
                  end do
                end do
              end do

      end subroutine update_t2c


end module mrcc_loops