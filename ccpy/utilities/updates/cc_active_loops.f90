module cc_active_loops

  implicit none

  contains

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
                      val = X2A(b,a,j,i)/(denom-shift)
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
                      val = X2C(b,a,j,i)/(denom-shift)
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

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  T3A UPDATES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine update_t3a_111111(t3a, resid, X3A, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      real(kind=8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact)
      real(kind=8), intent(in)  :: X3A(1:nua_act, 1:nua_act, 1:nua_act, 1:noa_act, 1:noa_act, 1:noa_act)
      real(kind=8), intent(in)  :: shift

      real(kind=8), intent(inout) :: t3a(1:nua_act, 1:nua_act, 1:nua_act, 1:noa_act, 1:noa_act, 1:noa_act)
      !f2py intent(in, out)  :: t3a(0:nua_act-1, 0:nua_act-1, 0:nua_act-1, 0:noa_act-1, 0:noa_act-1, 0:noa_act-1)
      real(kind=8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nua_act, 1:noa_act, 1:noa_act, 1:noa_act)

      integer :: i, j, k, a, b, c
      real(kind=8) :: denom, val

      do i = 1 , noa_act
         do j = i+1 , noa_act
            do k = j+1 , noa_act
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = b+1 , nua_act

                        denom = fA_oo_act(i,i) + fA_oo_act(j,j) + fA_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fA_vv_act(c,c)

                        val = X3A(a, b, c, i, j, k)/(denom - shift)

                        t3a(a, b, c, i, j, k) = t3a(a, b, c, i, j, k) + val
                        t3a(a, b, c, i, k, j) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, b, c, j, i, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, b, c, j, k, i) = t3a(a, b, c, i, j, k)
                        t3a(a, b, c, k, i, j) = t3a(a, b, c, i, j, k)
                        t3a(a, b, c, k, j, i) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, c, b, i, j, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, c, b, i, k, j) = t3a(a, b, c, i, j, k)
                        t3a(a, c, b, j, i, k) = t3a(a, b, c, i, j, k)
                        t3a(a, c, b, j, k, i) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, c, b, k, i, j) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, c, b, k, j, i) = t3a(a, b, c, i, j, k)
                        t3a(b, a, c, i, j, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(b, a, c, i, k, j) = t3a(a, b, c, i, j, k)
                        t3a(b, a, c, j, i, k) = t3a(a, b, c, i, j, k)
                        t3a(b, a, c, j, k, i) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(b, a, c, k, i, j) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(b, a, c, k, j, i) = t3a(a, b, c, i, j, k)
                        t3a(b, c, a, i, j, k) = t3a(a, b, c, i, j, k)
                        t3a(b, c, a, i, k, j) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(b, c, a, j, i, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(b, c, a, j, k, i) = t3a(a, b, c, i, j, k)
                        t3a(b, c, a, k, i, j) = t3a(a, b, c, i, j, k)
                        t3a(b, c, a, k, j, i) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(c, a, b, i, j, k) = t3a(a, b, c, i, j, k)
                        t3a(c, a, b, i, k, j) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(c, a, b, j, i, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(c, a, b, j, k, i) = t3a(a, b, c, i, j, k)
                        t3a(c, a, b, k, i, j) = t3a(a, b, c, i, j, k)
                        t3a(c, a, b, k, j, i) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(c, b, a, i, j, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(c, b, a, i, k, j) = t3a(a, b, c, i, j, k)
                        t3a(c, b, a, j, i, k) = t3a(a, b, c, i, j, k)
                        t3a(c, b, a, j, k, i) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(c, b, a, k, i, j) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(c, b, a, k, j, i) = t3a(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(a, b, c, j, k, i) = val
                        resid(a, b, c, k, i, j) = val
                        resid(a, b, c, k, j, i) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, i, k, j) = val
                        resid(a, c, b, j, i, k) = val
                        resid(a, c, b, j, k, i) = -1.0 * val
                        resid(a, c, b, k, i, j) = -1.0 * val
                        resid(a, c, b, k, j, i) = val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, i, k, j) = val
                        resid(b, a, c, j, i, k) = val
                        resid(b, a, c, j, k, i) = -1.0 * val
                        resid(b, a, c, k, i, j) = -1.0 * val
                        resid(b, a, c, k, j, i) = val
                        resid(b, c, a, i, j, k) = val
                        resid(b, c, a, i, k, j) = -1.0 * val
                        resid(b, c, a, j, i, k) = -1.0 * val
                        resid(b, c, a, j, k, i) = val
                        resid(b, c, a, k, i, j) = val
                        resid(b, c, a, k, j, i) = -1.0 * val
                        resid(c, a, b, i, j, k) = val
                        resid(c, a, b, i, k, j) = -1.0 * val
                        resid(c, a, b, j, i, k) = -1.0 * val
                        resid(c, a, b, j, k, i) = val
                        resid(c, a, b, k, i, j) = val
                        resid(c, a, b, k, j, i) = -1.0 * val
                        resid(c, b, a, i, j, k) = -1.0 * val
                        resid(c, b, a, i, k, j) = val
                        resid(c, b, a, j, i, k) = val
                        resid(c, b, a, j, k, i) = -1.0 * val
                        resid(c, b, a, k, i, j) = -1.0 * val
                        resid(c, b, a, k, j, i) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3a_111111

subroutine update_t3a_110111(t3a, resid, X3A, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      real(kind=8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact)
      real(kind=8), intent(in)  :: X3A(1:nua_act, 1:nua_act, 1:nua_inact, 1:noa_act, 1:noa_act, 1:noa_act)
      real(kind=8), intent(in)  :: shift

      real(kind=8), intent(inout) :: t3a(1:nua_act, 1:nua_act, 1:nua_inact, 1:noa_act, 1:noa_act, 1:noa_act)
      !f2py intent(in, out)  :: t3a(0:nua_act-1, 0:nua_act-1, 0:nua_inact-1, 0:noa_act-1, 0:noa_act-1, 0:noa_act-1)
      real(kind=8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nua_inact, 1:noa_act, 1:noa_act, 1:noa_act)

      integer :: i, j, k, a, b, c
      real(kind=8) :: denom, val

      do i = 1 , noa_act
         do j = i+1 , noa_act
            do k = j+1 , noa_act
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = 1 , nua_inact

                        denom = fA_oo_act(i,i) + fA_oo_act(j,j) + fA_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fA_vv_inact(c,c)

                        val = X3A(a, b, c, i, j, k)/(denom - shift)

                        t3a(a, b, c, i, j, k) = t3a(a, b, c, i, j, k) + val
                        t3a(a, b, c, i, k, j) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, b, c, j, i, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, b, c, j, k, i) = t3a(a, b, c, i, j, k)
                        t3a(a, b, c, k, i, j) = t3a(a, b, c, i, j, k)
                        t3a(a, b, c, k, j, i) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(b, a, c, i, j, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(b, a, c, i, k, j) = t3a(a, b, c, i, j, k)
                        t3a(b, a, c, j, i, k) = t3a(a, b, c, i, j, k)
                        t3a(b, a, c, j, k, i) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(b, a, c, k, i, j) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(b, a, c, k, j, i) = t3a(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(a, b, c, j, k, i) = val
                        resid(a, b, c, k, i, j) = val
                        resid(a, b, c, k, j, i) = -1.0 * val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, i, k, j) = val
                        resid(b, a, c, j, i, k) = val
                        resid(b, a, c, j, k, i) = -1.0 * val
                        resid(b, a, c, k, i, j) = -1.0 * val
                        resid(b, a, c, k, j, i) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3a_110111

subroutine update_t3a_111011(t3a, resid, X3A, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      real(kind=8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact)
      real(kind=8), intent(in)  :: X3A(1:nua_act, 1:nua_act, 1:nua_act, 1:noa_inact, 1:noa_act, 1:noa_act)
      real(kind=8), intent(in)  :: shift

      real(kind=8), intent(inout) :: t3a(1:nua_act, 1:nua_act, 1:nua_act, 1:noa_inact, 1:noa_act, 1:noa_act)
      !f2py intent(in, out)  :: t3a(0:nua_act-1, 0:nua_act-1, 0:nua_act-1, 0:noa_inact-1, 0:noa_act-1, 0:noa_act-1)
      real(kind=8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nua_act, 1:noa_inact, 1:noa_act, 1:noa_act)

      integer :: i, j, k, a, b, c
      real(kind=8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = j+1 , noa_act
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = b+1 , nua_act

                        denom = fA_oo_inact(i,i) + fA_oo_act(j,j) + fA_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fA_vv_act(c,c)

                        val = X3A(a, b, c, i, j, k)/(denom - shift)

                        t3a(a, b, c, i, j, k) = t3a(a, b, c, i, j, k) + val
                        t3a(a, b, c, i, k, j) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, c, b, i, j, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, c, b, i, k, j) = t3a(a, b, c, i, j, k)
                        t3a(b, a, c, i, j, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(b, a, c, i, k, j) = t3a(a, b, c, i, j, k)
                        t3a(b, c, a, i, j, k) = t3a(a, b, c, i, j, k)
                        t3a(b, c, a, i, k, j) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(c, a, b, i, j, k) = t3a(a, b, c, i, j, k)
                        t3a(c, a, b, i, k, j) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(c, b, a, i, j, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(c, b, a, i, k, j) = t3a(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, i, k, j) = val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, i, k, j) = val
                        resid(b, c, a, i, j, k) = val
                        resid(b, c, a, i, k, j) = -1.0 * val
                        resid(c, a, b, i, j, k) = val
                        resid(c, a, b, i, k, j) = -1.0 * val
                        resid(c, b, a, i, j, k) = -1.0 * val
                        resid(c, b, a, i, k, j) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3a_111011


    subroutine update_t3a_100111(t3a, resid, X3A, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      real(kind=8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact)
      real(kind=8), intent(in)  :: X3A(1:nua_act, 1:nua_inact, 1:nua_inact, 1:noa_act, 1:noa_act, 1:noa_act)
      real(kind=8), intent(in)  :: shift

      real(8), intent(inout) :: t3a(1:nua_act, 1:nua_inact, 1:nua_inact, 1:noa_act, 1:noa_act, 1:noa_act)
      !f2py intent(in, out)  :: t3a(0:nua_act-1, 0:nua_inact-1, 0:nua_inact-1, 0:noa_act-1, 0:noa_act-1, 0:noa_act-1)
      real(kind=8), intent(out)   :: resid(1:nua_act, 1:nua_inact, 1:nua_inact, 1:noa_act, 1:noa_act, 1:noa_act)

      integer :: i, j, k, a, b, c
      real(kind=8) :: denom, val

      do i = 1 , noa_act
         do j = i+1 , noa_act
            do k = j+1 , noa_act
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = b+1 , nua_inact

                        denom = fA_oo_act(i,i) + fA_oo_act(j,j) + fA_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_inact(b,b) - fA_vv_inact(c,c)

                        val = X3A(a, b, c, i, j, k)/(denom - shift)

                        t3a(a, b, c, i, j, k) = t3a(a, b, c, i, j, k) + val
                        t3a(a, b, c, i, k, j) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, b, c, j, i, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, b, c, j, k, i) = t3a(a, b, c, i, j, k)
                        t3a(a, b, c, k, i, j) = t3a(a, b, c, i, j, k)
                        t3a(a, b, c, k, j, i) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, c, b, i, j, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, c, b, i, k, j) = t3a(a, b, c, i, j, k)
                        t3a(a, c, b, j, i, k) = t3a(a, b, c, i, j, k)
                        t3a(a, c, b, j, k, i) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, c, b, k, i, j) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, c, b, k, j, i) = t3a(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(a, b, c, j, k, i) = val
                        resid(a, b, c, k, i, j) = val
                        resid(a, b, c, k, j, i) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, i, k, j) = val
                        resid(a, c, b, j, i, k) = val
                        resid(a, c, b, j, k, i) = -1.0 * val
                        resid(a, c, b, k, i, j) = -1.0 * val
                        resid(a, c, b, k, j, i) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

    end subroutine update_t3a_100111

    subroutine update_t3a_111001(t3a, resid, X3A, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      real(kind=8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact)
      real(kind=8), intent(in)  :: X3A(1:nua_act, 1:nua_act, 1:nua_act, 1:noa_inact, 1:noa_inact, 1:noa_act)
      real(kind=8), intent(in)  :: shift

      real(kind=8), intent(inout) :: t3a(1:nua_act, 1:nua_act, 1:nua_act, 1:noa_inact, 1:noa_inact, 1:noa_act)
      !f2py intent(in, out)  :: t3a(0:nua_act-1, 0:nua_act-1, 0:nua_act-1, 0:noa_inact-1, 0:noa_inact-1, 0:noa_act-1)
      real(kind=8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nua_act, 1:noa_inact, 1:noa_inact, 1:noa_act)

      integer :: i, j, k, a, b, c
      real(kind=8) :: denom, val

      do i = 1 , noa_inact
         do j = i+1 , noa_inact
            do k = 1 , noa_act
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = b+1 , nua_act

                        denom = fA_oo_inact(i,i) + fA_oo_inact(j,j) + fA_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fA_vv_act(c,c)

                        val = X3A(a, b, c, i, j, k)/(denom - shift)

                        t3a(a, b, c, i, j, k) = t3a(a, b, c, i, j, k) + val
                        t3a(a, b, c, j, i, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, c, b, i, j, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, c, b, j, i, k) = t3a(a, b, c, i, j, k)
                        t3a(b, a, c, i, j, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(b, a, c, j, i, k) = t3a(a, b, c, i, j, k)
                        t3a(b, c, a, i, j, k) = t3a(a, b, c, i, j, k)
                        t3a(b, c, a, j, i, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(c, a, b, i, j, k) = t3a(a, b, c, i, j, k)
                        t3a(c, a, b, j, i, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(c, b, a, i, j, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(c, b, a, j, i, k) = t3a(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, j, i, k) = val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, j, i, k) = val
                        resid(b, c, a, i, j, k) = val
                        resid(b, c, a, j, i, k) = -1.0 * val
                        resid(c, a, b, i, j, k) = val
                        resid(c, a, b, j, i, k) = -1.0 * val
                        resid(c, b, a, i, j, k) = -1.0 * val
                        resid(c, b, a, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

    end subroutine update_t3a_111001

    subroutine update_t3a_110011(t3a, resid, X3A, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      real(kind=8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact)
      real(kind=8), intent(in)  :: X3A(1:nua_act, 1:nua_act, 1:nua_inact, 1:noa_inact, 1:noa_act, 1:noa_act)
      real(kind=8), intent(in)  :: shift

      real(kind=8), intent(inout) :: t3a(1:nua_act, 1:nua_act, 1:nua_inact, 1:noa_inact, 1:noa_act, 1:noa_act)
      !f2py intent(in, out)  :: t3a(0:nua_act-1, 0:nua_act-1, 0:nua_inact-1, 0:noa_inact-1, 0:noa_act-1, 0:noa_act-1)
      real(kind=8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nua_inact, 1:noa_inact, 1:noa_act, 1:noa_act)

      integer :: i, j, k, a, b, c
      real(kind=8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = j+1 , noa_act
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = 1 , nua_inact

                        denom = fA_oo_inact(i,i) + fA_oo_act(j,j) + fA_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fA_vv_inact(c,c)

                        val = X3A(a, b, c, i, j, k)/(denom - shift)

                        t3a(a, b, c, i, j, k) = t3a(a, b, c, i, j, k) + val
                        t3a(a, b, c, i, k, j) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(b, a, c, i, j, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(b, a, c, i, k, j) = t3a(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, i, k, j) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

    end subroutine update_t3a_110011


    subroutine update_t3a_100011(t3a, resid, X3A, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      real(kind=8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact)
      real(kind=8), intent(in)  :: X3A(1:nua_act, 1:nua_inact, 1:nua_inact, 1:noa_inact, 1:noa_act, 1:noa_act)
      real(kind=8), intent(in)  :: shift

      real(8), intent(inout) :: t3a(1:nua_act, 1:nua_inact, 1:nua_inact, 1:noa_inact, 1:noa_act, 1:noa_act)
      !f2py intent(in, out)  :: t3a(0:nua_act-1, 0:nua_inact-1, 0:nua_inact-1, 0:noa_inact-1, 0:noa_act-1, 0:noa_act-1)
      real(kind=8), intent(out)   :: resid(1:nua_act, 1:nua_inact, 1:nua_inact, 1:noa_inact, 1:noa_act, 1:noa_act)

      integer :: i, j, k, a, b, c
      real(kind=8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = j+1 , noa_act
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = b+1 , nua_inact

                        denom = fA_oo_inact(i,i) + fA_oo_act(j,j) + fA_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_inact(b,b) - fA_vv_inact(c,c)

                        val = X3A(a, b, c, i, j, k)/(denom - shift)

                        t3a(a, b, c, i, j, k) = t3a(a, b, c, i, j, k) + val
                        t3a(a, b, c, i, k, j) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, c, b, i, j, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, c, b, i, k, j) = t3a(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, i, k, j) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

    end subroutine update_t3a_100011

    subroutine update_t3a_110001(t3a, resid, X3A, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      real(kind=8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact)
      real(kind=8), intent(in)  :: X3A(1:nua_act, 1:nua_act, 1:nua_inact, 1:noa_inact, 1:noa_inact, 1:noa_act)
      real(kind=8), intent(in)  :: shift

      real(kind=8), intent(inout) :: t3a(1:nua_act, 1:nua_act, 1:nua_inact, 1:noa_inact, 1:noa_inact, 1:noa_act)
      !f2py intent(in, out)  :: t3a(0:nua_act-1, 0:nua_act-1, 0:nua_inact-1, 0:noa_inact-1, 0:noa_inact-1, 0:noa_act-1)
      real(kind=8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nua_inact, 1:noa_inact, 1:noa_inact, 1:noa_act)

      integer :: i, j, k, a, b, c
      real(kind=8) :: denom, val

      do i = 1 , noa_inact
         do j = i+1 , noa_inact
            do k = 1 , noa_act
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = 1 , nua_inact

                        denom = fA_oo_inact(i,i) + fA_oo_inact(j,j) + fA_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fA_vv_inact(c,c)

                        val = X3A(a, b, c, i, j, k)/(denom - shift)

                        t3a(a, b, c, i, j, k) = t3a(a, b, c, i, j, k) + val
                        t3a(a, b, c, j, i, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(b, a, c, i, j, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(b, a, c, j, i, k) = t3a(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

      end subroutine update_t3a_110001


      subroutine update_t3a_100001(t3a, resid, X3A, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      real(kind=8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact)
      real(kind=8), intent(in)  :: X3A(1:nua_act, 1:nua_inact, 1:nua_inact, 1:noa_inact, 1:noa_inact, 1:noa_act)
      real(kind=8), intent(in)  :: shift

      real(kind=8), intent(inout) :: t3a(1:nua_act, 1:nua_inact, 1:nua_inact, 1:noa_inact, 1:noa_inact, 1:noa_act)
      !f2py intent(in, out)  :: t3a(0:nua_act-1, 0:nua_inact-1, 0:nua_inact-1, 0:noa_inact-1, 0:noa_inact-1, 0:noa_act-1)
      real(kind=8), intent(out)   :: resid(1:nua_act, 1:nua_inact, 1:nua_inact, 1:noa_inact, 1:noa_inact, 1:noa_act)

      integer :: i, j, k, a, b, c
      real(kind=8) :: denom, val

      do i = 1 , noa_inact
         do j = i+1 , noa_inact
            do k = 1 , noa_act
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = b+1 , nua_inact

                        denom = fA_oo_inact(i,i) + fA_oo_inact(j,j) + fA_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_inact(b,b) - fA_vv_inact(c,c)

                        val = X3A(a, b, c, i, j, k)/(denom - shift)

                        t3a(a, b, c, i, j, k) = t3a(a, b, c, i, j, k) + val
                        t3a(a, b, c, j, i, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, c, b, i, j, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, c, b, j, i, k) = t3a(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

    end subroutine update_t3a_100001

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  T3B UPDATES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine update_t3b_111111(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_act-1, 0:nub_act-1, 0:noa_act-1, 0:noa_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = i+1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = 1 , nub_act

                        denom = fA_oo_act(i,i) + fA_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, i, j, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, j, i, k) = t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_111111

subroutine update_t3b_110111(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_act, 1:noa_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_act, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_act-1, 0:nub_inact-1, 0:noa_act-1, 0:noa_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_act, 1:noa_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = i+1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = 1 , nub_inact

                        denom = fA_oo_act(i,i) + fA_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fB_vv_inact(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, i, j, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, j, i, k) = t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_110111

subroutine update_t3b_101111(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_act-1, 0:noa_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = i+1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_act

                        denom = fA_oo_act(i,i) + fA_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_inact(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_101111

subroutine update_t3b_111011(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_act-1, 0:nub_act-1, 0:noa_inact-1, 0:noa_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = 1 , nub_act

                        denom = fA_oo_inact(i,i) + fA_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(b, a, c, i, j, k) = -1.0 * t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(b, a, c, i, j, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_111011

subroutine update_t3b_111110(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_inact)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_act-1, 0:nub_act-1, 0:noa_act-1, 0:noa_act-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = i+1 , noa_act
            do k = 1 , nob_inact
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = 1 , nub_act

                        denom = fA_oo_act(i,i) + fA_oo_act(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, i, j, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, j, i, k) = t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_111110

subroutine update_t3b_110011(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_inact, 1:noa_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_inact, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_act-1, 0:nub_inact-1, 0:noa_inact-1, 0:noa_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_inact, 1:noa_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = 1 , nub_inact

                        denom = fA_oo_inact(i,i) + fA_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fB_vv_inact(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(b, a, c, i, j, k) = -1.0 * t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(b, a, c, i, j, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_110011

subroutine update_t3b_110110(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_act, 1:noa_act, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_act, 1:noa_act, 1:nob_inact)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_act-1, 0:nub_inact-1, 0:noa_act-1, 0:noa_act-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_act, 1:noa_act, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = i+1 , noa_act
            do k = 1 , nob_inact
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = 1 , nub_inact

                        denom = fA_oo_act(i,i) + fA_oo_act(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fB_vv_inact(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, i, j, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, j, i, k) = t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_110110

subroutine update_t3b_101011(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_inact-1, 0:noa_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_act

                        denom = fA_oo_inact(i,i) + fA_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_inact(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val

                        resid(a, b, c, i, j, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_101011

subroutine update_t3b_101110(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_inact)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_act-1, 0:noa_act-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = i+1 , noa_act
            do k = 1 , nob_inact
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_act

                        denom = fA_oo_act(i,i) + fA_oo_act(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_act(a,a) - fA_vv_inact(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_101110

      subroutine update_t3b_101010(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_inact)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_inact-1, 0:noa_act-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = 1 , nob_inact
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_act

                        denom = fA_oo_inact(i,i) + fA_oo_act(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_act(a,a) - fA_vv_inact(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val

                        resid(a, b, c, i, j, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_101010


subroutine update_t3b_111010(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_inact)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_act-1, 0:nub_act-1, 0:noa_inact-1, 0:noa_act-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = 1 , nob_inact
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = 1 , nub_act

                        denom = fA_oo_inact(i,i) + fA_oo_act(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(b, a, c, i, j, k) = -1.0 * t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(b, a, c, i, j, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_111010


subroutine update_t3b_100111(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_act, 1:noa_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_act, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_inact-1, 0:nub_inact-1, 0:noa_act-1, 0:noa_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_act, 1:noa_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = i+1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_inact

                        denom = fA_oo_act(i,i) + fA_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_inact(b,b) - fB_vv_inact(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_100111

subroutine update_t3b_001111(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_inact-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_act-1, 0:noa_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = i+1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_inact
                  do b = a+1 , nua_inact
                     do c = 1 , nub_act

                        denom = fA_oo_act(i,i) + fA_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_inact(a,a) - fA_vv_inact(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, i, j, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, j, i, k) = t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_001111

subroutine update_t3b_111001(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_inact, 1:noa_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_inact, 1:noa_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_act-1, 0:nub_act-1, 0:noa_inact-1, 0:noa_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_inact, 1:noa_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = i+1 , noa_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = 1 , nub_act

                        denom = fA_oo_inact(i,i) + fA_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, i, j, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, j, i, k) = t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_111001



subroutine update_t3b_001011(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_inact-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_inact-1, 0:noa_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_inact
                  do b = a+1 , nua_inact
                     do c = 1 , nub_act

                        denom = fA_oo_inact(i,i) + fA_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_inact(a,a) - fA_vv_inact(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(b, a, c, i, j, k) = -1.0 * t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(b, a, c, i, j, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_001011

subroutine update_t3b_001110(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_inact)
      !f2py intent(in, out)  :: t3b(0:nua_inact-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_act-1, 0:noa_act-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = i+1 , noa_act
            do k = 1 , nob_inact
               do a = 1 , nua_inact
                  do b = a+1 , nua_inact
                     do c = 1 , nub_act

                        denom = fA_oo_act(i,i) + fA_oo_act(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_inact(a,a) - fA_vv_inact(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, i, j, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, j, i, k) = t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_001110

subroutine update_t3b_100110(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_act, 1:noa_act, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_act, 1:noa_act, 1:nob_inact)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_inact-1, 0:nub_inact-1, 0:noa_act-1, 0:noa_act-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_act, 1:noa_act, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = i+1 , noa_act
            do k = 1 , nob_inact
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_inact

                        denom = fA_oo_act(i,i) + fA_oo_act(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_act(a,a) - fA_vv_inact(b,b) - fB_vv_inact(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_100110

subroutine update_t3b_100011(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_inact, 1:noa_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_inact, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_inact-1, 0:nub_inact-1, 0:noa_inact-1, 0:noa_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_inact, 1:noa_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_inact

                        denom = fA_oo_inact(i,i) + fA_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_inact(b,b) - fB_vv_inact(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val

                        resid(a, b, c, i, j, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_100011

subroutine update_t3b_110001(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_inact, 1:noa_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_inact, 1:noa_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_act-1, 0:nub_inact-1, 0:noa_inact-1, 0:noa_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_inact, 1:noa_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = i+1 , noa_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = 1 , nub_inact

                        denom = fA_oo_inact(i,i) + fA_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fB_vv_inact(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, i, j, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, j, i, k) = t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_110001

subroutine update_t3b_101001(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_inact-1, 0:noa_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = i+1 , noa_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_act

                        denom = fA_oo_inact(i,i) + fA_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_inact(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_101001





subroutine update_t3b_001001(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_inact-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_inact-1, 0:noa_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = i+1 , noa_inact
            do k = 1 , nob_act
               do a = 1 , nua_inact
                  do b = a+1 , nua_inact
                     do c = 1 , nub_act

                        denom = fA_oo_inact(i,i) + fA_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_inact(a,a) - fA_vv_inact(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, i, j, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, j, i, k) = t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_001001


subroutine update_t3b_100001(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_inact, 1:noa_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_inact, 1:noa_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_inact-1, 0:nub_inact-1, 0:noa_inact-1, 0:noa_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_inact, 1:noa_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = i+1 , noa_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_inact

                        denom = fA_oo_inact(i,i) + fA_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_inact(b,b) - fB_vv_inact(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_100001

      subroutine update_t3b_110010(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_inact, 1:noa_act, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_inact, 1:noa_act, 1:nob_inact)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_act-1, 0:nub_inact-1, 0:noa_inact-1, 0:noa_act-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_inact, 1:noa_act, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = 1 , nob_inact
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = 1 , nub_inact

                        denom = fA_oo_inact(i,i) + fA_oo_act(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fB_vv_inact(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(b, a, c, i, j, k) = -1.0 * t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(b, a, c, i, j, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_110010

      subroutine update_t3b_001010(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_inact)
      !f2py intent(in, out)  :: t3b(0:nua_inact-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_inact-1, 0:noa_act-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = 1 , nob_inact
               do a = 1 , nua_inact
                  do b = a+1 , nua_inact
                     do c = 1 , nub_act

                        denom = fA_oo_inact(i,i) + fA_oo_act(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_inact(a,a) - fA_vv_inact(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(b, a, c, i, j, k) = -1.0 * t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(b, a, c, i, j, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_001010

subroutine update_t3b_100010(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_inact, 1:noa_act, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_inact, 1:noa_act, 1:nob_inact)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_inact-1, 0:nub_inact-1, 0:noa_inact-1, 0:noa_act-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_inact, 1:noa_act, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = 1 , nob_inact
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_inact

                        denom = fA_oo_inact(i,i) + fA_oo_act(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_act(a,a) - fA_vv_inact(b,b) - fB_vv_inact(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val

                        resid(a, b, c, i, j, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_100010



      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  T3C UPDATES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine update_t3c_111111(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_act-1, 0:nub_act-1, 0:noa_act-1, 0:nob_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_act
            do k = j+1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = b+1 , nub_act

                        denom = fA_oo_act(i,i) + fB_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fB_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, k, j) = t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, i, k, j) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_111111

subroutine update_t3c_011111(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_inact-1, 0:nub_act-1, 0:nub_act-1, 0:noa_act-1, 0:nob_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_act
            do k = j+1 , nob_act
               do a = 1 , nua_inact
                  do b = 1 , nub_act
                     do c = b+1 , nub_act

                        denom = fA_oo_act(i,i) + fB_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_inact(a,a) - fB_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, k, j) = t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, i, k, j) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_011111

subroutine update_t3c_110111(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_act-1, 0:nub_inact-1, 0:noa_act-1, 0:nob_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_act
            do k = j+1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = 1 , nub_inact

                        denom = fA_oo_act(i,i) + fB_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fB_vv_act(b,b) - fB_vv_inact(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_110111

subroutine update_t3c_111011(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_inact, 1:nob_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_inact, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_act-1, 0:nub_act-1, 0:noa_inact-1, 0:nob_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_inact, 1:nob_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , nob_act
            do k = j+1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = b+1 , nub_act

                        denom = fA_oo_inact(i,i) + fB_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fB_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, k, j) = t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, i, k, j) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_111011

subroutine update_t3c_111101(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_act-1, 0:nub_act-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = b+1 , nub_act

                        denom = fA_oo_act(i,i) + fB_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fB_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, c, b, i, j, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_111101



subroutine update_t3c_100111(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_act, 1:nob_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_act, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_inact-1, 0:nub_inact-1, 0:noa_act-1, 0:nob_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_act, 1:nob_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_act
            do k = j+1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_inact
                     do c = b+1 , nub_inact

                        denom = fA_oo_act(i,i) + fB_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fB_vv_inact(b,b) - fB_vv_inact(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, k, j) = t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, i, k, j) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_100111

subroutine update_t3c_100011(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_inact, 1:nob_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_inact, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_inact-1, 0:nub_inact-1, 0:noa_inact-1, 0:nob_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_inact, 1:nob_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , nob_act
            do k = j+1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_inact
                     do c = b+1 , nub_inact

                        denom = fA_oo_inact(i,i) + fB_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fB_vv_inact(b,b) - fB_vv_inact(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, k, j) = t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, i, k, j) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_100011

subroutine update_t3c_100101(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_inact-1, 0:nub_inact-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_inact
                     do c = b+1 , nub_inact

                        denom = fA_oo_act(i,i) + fB_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fB_vv_inact(b,b) - fB_vv_inact(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, c, b, i, j, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_100101

subroutine update_t3c_111001(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_inact, 1:nob_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_inact, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_act-1, 0:nub_act-1, 0:noa_inact-1, 0:nob_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_inact, 1:nob_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = b+1 , nub_act

                        denom = fA_oo_inact(i,i) + fB_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fB_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, c, b, i, j, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_111001

subroutine update_t3c_011001(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_inact, 1:nob_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_inact, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_inact-1, 0:nub_act-1, 0:nub_act-1, 0:noa_inact-1, 0:nob_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_inact, 1:nob_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_inact
                  do b = 1 , nub_act
                     do c = b+1 , nub_act

                        denom = fA_oo_inact(i,i) + fB_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_inact(a,a) - fB_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, c, b, i, j, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_011001

subroutine update_t3c_110001(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_inact, 1:nob_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_inact, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_act-1, 0:nub_inact-1, 0:noa_inact-1, 0:nob_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_inact, 1:nob_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = 1 , nub_inact

                        denom = fA_oo_inact(i,i) + fB_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fB_vv_act(b,b) - fB_vv_inact(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val

                        resid(a, b, c, i, j, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_110001

subroutine update_t3c_111100(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_inact)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_act-1, 0:nub_act-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = j+1 , nob_inact
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = b+1 , nub_act

                        denom = fA_oo_act(i,i) + fB_oo_inact(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_act(a,a) - fB_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, k, j) = t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, i, k, j) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_111100

subroutine update_t3c_011100(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_inact)
      !f2py intent(in, out)  :: t3c(0:nua_inact-1, 0:nub_act-1, 0:nub_act-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = j+1 , nob_inact
               do a = 1 , nua_inact
                  do b = 1 , nub_act
                     do c = b+1 , nub_act

                        denom = fA_oo_act(i,i) + fB_oo_inact(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_inact(a,a) - fB_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, k, j) = t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, i, k, j) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_011100

subroutine update_t3c_110100(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_inact)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_act-1, 0:nub_inact-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = j+1 , nob_inact
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = 1 , nub_inact

                        denom = fA_oo_act(i,i) + fB_oo_inact(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_act(a,a) - fB_vv_act(b,b) - fB_vv_inact(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_110100

subroutine update_t3c_011011(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_inact, 1:nob_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_inact, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_inact-1, 0:nub_act-1, 0:nub_act-1, 0:noa_inact-1, 0:nob_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_inact, 1:nob_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , nob_act
            do k = j+1 , nob_act
               do a = 1 , nua_inact
                  do b = 1 , nub_act
                     do c = b+1 , nub_act

                        denom = fA_oo_inact(i,i) + fB_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_inact(a,a) - fB_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, k, j) = t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, i, k, j) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_011011

subroutine update_t3c_011101(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_inact-1, 0:nub_act-1, 0:nub_act-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_inact
                  do b = 1 , nub_act
                     do c = b+1 , nub_act

                        denom = fA_oo_act(i,i) + fB_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_inact(a,a) - fB_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, c, b, i, j, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_011101

subroutine update_t3c_110011(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_inact, 1:nob_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_inact, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_act-1, 0:nub_inact-1, 0:noa_inact-1, 0:nob_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_inact, 1:nob_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , nob_act
            do k = j+1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = 1 , nub_inact

                        denom = fA_oo_inact(i,i) + fB_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fB_vv_act(b,b) - fB_vv_inact(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_110011

subroutine update_t3c_110101(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_act-1, 0:nub_inact-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = 1 , nub_inact

                        denom = fA_oo_act(i,i) + fB_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fB_vv_act(b,b) - fB_vv_inact(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val

                        resid(a, b, c, i, j, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_110101

      subroutine update_t3c_010111(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_inact, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_inact, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_inact-1, 0:nub_act-1, 0:nub_inact-1, 0:noa_act-1, 0:nob_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_act
            do k = j+1 , nob_act
               do a = 1 , nua_inact
                  do b = 1 , nub_act
                     do c = 1 , nub_inact

                        denom = fA_oo_act(i,i) + fB_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_inact(a,a) - fB_vv_act(b,b) - fB_vv_inact(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_010111

subroutine update_t3c_010011(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_inact, 1:nub_act, 1:nub_inact, 1:noa_inact, 1:nob_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_inact, 1:nub_act, 1:nub_inact, 1:noa_inact, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_inact-1, 0:nub_act-1, 0:nub_inact-1, 0:noa_inact-1, 0:nob_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nub_act, 1:nub_inact, 1:noa_inact, 1:nob_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , nob_act
            do k = j+1 , nob_act
               do a = 1 , nua_inact
                  do b = 1 , nub_act
                     do c = 1 , nub_inact

                        denom = fA_oo_inact(i,i) + fB_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_inact(a,a) - fB_vv_act(b,b) - fB_vv_inact(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_010011

subroutine update_t3c_010101(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_inact, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_inact, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_inact-1, 0:nub_act-1, 0:nub_inact-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_inact
                  do b = 1 , nub_act
                     do c = 1 , nub_inact

                        denom = fA_oo_act(i,i) + fB_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_inact(a,a) - fB_vv_act(b,b) - fB_vv_inact(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val

                        resid(a, b, c, i, j, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_010101

subroutine update_t3c_010100(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_inact, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_inact, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_inact)
      !f2py intent(in, out)  :: t3c(0:nua_inact-1, 0:nub_act-1, 0:nub_inact-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = j+1 , nob_inact
               do a = 1 , nua_inact
                  do b = 1 , nub_act
                     do c = 1 , nub_inact

                        denom = fA_oo_act(i,i) + fB_oo_inact(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_inact(a,a) - fB_vv_act(b,b) - fB_vv_inact(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_010100

subroutine update_t3c_010001(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_inact, 1:nub_act, 1:nub_inact, 1:noa_inact, 1:nob_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_inact, 1:nub_act, 1:nub_inact, 1:noa_inact, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_inact-1, 0:nub_act-1, 0:nub_inact-1, 0:noa_inact-1, 0:nob_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nub_act, 1:nub_inact, 1:noa_inact, 1:nob_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_inact
                  do b = 1 , nub_act
                     do c = 1 , nub_inact

                        denom = fA_oo_inact(i,i) + fB_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_inact(a,a) - fB_vv_act(b,b) - fB_vv_inact(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val

                        resid(a, b, c, i, j, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_010001



subroutine update_t3c_100100(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_inact)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_inact-1, 0:nub_inact-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = j+1 , nob_inact
               do a = 1 , nua_act
                  do b = 1 , nub_inact
                     do c = b+1 , nub_inact

                        denom = fA_oo_act(i,i) + fB_oo_inact(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_act(a,a) - fB_vv_inact(b,b) - fB_vv_inact(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, k, j) = t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, i, k, j) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_100100

subroutine update_t3c_100001(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_inact, 1:nob_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_inact, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_inact-1, 0:nub_inact-1, 0:noa_inact-1, 0:nob_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_inact, 1:nob_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_inact
                     do c = b+1 , nub_inact

                        denom = fA_oo_inact(i,i) + fB_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fB_vv_inact(b,b) - fB_vv_inact(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, c, b, i, j, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_100001

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  T3D UPDATES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine update_t3d_111111(t3d, resid, X3D, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3D(1:nub_act, 1:nub_act, 1:nub_act, 1:nob_act, 1:nob_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3d(1:nub_act, 1:nub_act, 1:nub_act, 1:nob_act, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: t3d(0:nub_act-1, 0:nub_act-1, 0:nub_act-1, 0:nob_act-1, 0:nob_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nub_act, 1:nub_act, 1:nub_act, 1:nob_act, 1:nob_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , nob_act
         do j = i+1 , nob_act
            do k = j+1 , nob_act
               do a = 1 , nub_act
                  do b = a+1 , nub_act
                     do c = b+1 , nub_act

                        denom = fB_oo_act(i,i) + fB_oo_act(j,j) + fB_oo_act(k,k)&
                               -fB_vv_act(a,a) - fB_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3D(a, b, c, i, j, k)/(denom - shift)

                        t3d(a, b, c, i, j, k) = t3d(a, b, c, i, j, k) + val
                        t3d(a, b, c, i, k, j) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(a, b, c, j, i, k) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(a, b, c, j, k, i) = t3d(a, b, c, i, j, k)
                        t3d(a, b, c, k, i, j) = t3d(a, b, c, i, j, k)
                        t3d(a, b, c, k, j, i) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(a, c, b, i, j, k) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(a, c, b, i, k, j) = t3d(a, b, c, i, j, k)
                        t3d(a, c, b, j, i, k) = t3d(a, b, c, i, j, k)
                        t3d(a, c, b, j, k, i) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(a, c, b, k, i, j) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(a, c, b, k, j, i) = t3d(a, b, c, i, j, k)
                        t3d(b, a, c, i, j, k) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(b, a, c, i, k, j) = t3d(a, b, c, i, j, k)
                        t3d(b, a, c, j, i, k) = t3d(a, b, c, i, j, k)
                        t3d(b, a, c, j, k, i) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(b, a, c, k, i, j) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(b, a, c, k, j, i) = t3d(a, b, c, i, j, k)
                        t3d(b, c, a, i, j, k) = t3d(a, b, c, i, j, k)
                        t3d(b, c, a, i, k, j) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(b, c, a, j, i, k) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(b, c, a, j, k, i) = t3d(a, b, c, i, j, k)
                        t3d(b, c, a, k, i, j) = t3d(a, b, c, i, j, k)
                        t3d(b, c, a, k, j, i) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(c, a, b, i, j, k) = t3d(a, b, c, i, j, k)
                        t3d(c, a, b, i, k, j) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(c, a, b, j, i, k) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(c, a, b, j, k, i) = t3d(a, b, c, i, j, k)
                        t3d(c, a, b, k, i, j) = t3d(a, b, c, i, j, k)
                        t3d(c, a, b, k, j, i) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(c, b, a, i, j, k) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(c, b, a, i, k, j) = t3d(a, b, c, i, j, k)
                        t3d(c, b, a, j, i, k) = t3d(a, b, c, i, j, k)
                        t3d(c, b, a, j, k, i) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(c, b, a, k, i, j) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(c, b, a, k, j, i) = t3d(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(a, b, c, j, k, i) = val
                        resid(a, b, c, k, i, j) = val
                        resid(a, b, c, k, j, i) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, i, k, j) = val
                        resid(a, c, b, j, i, k) = val
                        resid(a, c, b, j, k, i) = -1.0 * val
                        resid(a, c, b, k, i, j) = -1.0 * val
                        resid(a, c, b, k, j, i) = val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, i, k, j) = val
                        resid(b, a, c, j, i, k) = val
                        resid(b, a, c, j, k, i) = -1.0 * val
                        resid(b, a, c, k, i, j) = -1.0 * val
                        resid(b, a, c, k, j, i) = val
                        resid(b, c, a, i, j, k) = val
                        resid(b, c, a, i, k, j) = -1.0 * val
                        resid(b, c, a, j, i, k) = -1.0 * val
                        resid(b, c, a, j, k, i) = val
                        resid(b, c, a, k, i, j) = val
                        resid(b, c, a, k, j, i) = -1.0 * val
                        resid(c, a, b, i, j, k) = val
                        resid(c, a, b, i, k, j) = -1.0 * val
                        resid(c, a, b, j, i, k) = -1.0 * val
                        resid(c, a, b, j, k, i) = val
                        resid(c, a, b, k, i, j) = val
                        resid(c, a, b, k, j, i) = -1.0 * val
                        resid(c, b, a, i, j, k) = -1.0 * val
                        resid(c, b, a, i, k, j) = val
                        resid(c, b, a, j, i, k) = val
                        resid(c, b, a, j, k, i) = -1.0 * val
                        resid(c, b, a, k, i, j) = -1.0 * val
                        resid(c, b, a, k, j, i) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3d_111111

subroutine update_t3d_110111(t3d, resid, X3D, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3D(1:nub_act, 1:nub_act, 1:nub_inact, 1:nob_act, 1:nob_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3d(1:nub_act, 1:nub_act, 1:nub_inact, 1:nob_act, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: t3d(0:nub_act-1, 0:nub_act-1, 0:nub_inact-1, 0:nob_act-1, 0:nob_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nub_act, 1:nub_act, 1:nub_inact, 1:nob_act, 1:nob_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , nob_act
         do j = i+1 , nob_act
            do k = j+1 , nob_act
               do a = 1 , nub_act
                  do b = a+1 , nub_act
                     do c = 1 , nub_inact

                        denom = fB_oo_act(i,i) + fB_oo_act(j,j) + fB_oo_act(k,k)&
                               -fB_vv_act(a,a) - fB_vv_act(b,b) - fB_vv_inact(c,c)

                        val = X3D(a, b, c, i, j, k)/(denom - shift)

                        t3d(a, b, c, i, j, k) = t3d(a, b, c, i, j, k) + val
                        t3d(a, b, c, i, k, j) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(a, b, c, j, i, k) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(a, b, c, j, k, i) = t3d(a, b, c, i, j, k)
                        t3d(a, b, c, k, i, j) = t3d(a, b, c, i, j, k)
                        t3d(a, b, c, k, j, i) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(b, a, c, i, j, k) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(b, a, c, i, k, j) = t3d(a, b, c, i, j, k)
                        t3d(b, a, c, j, i, k) = t3d(a, b, c, i, j, k)
                        t3d(b, a, c, j, k, i) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(b, a, c, k, i, j) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(b, a, c, k, j, i) = t3d(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(a, b, c, j, k, i) = val
                        resid(a, b, c, k, i, j) = val
                        resid(a, b, c, k, j, i) = -1.0 * val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, i, k, j) = val
                        resid(b, a, c, j, i, k) = val
                        resid(b, a, c, j, k, i) = -1.0 * val
                        resid(b, a, c, k, i, j) = -1.0 * val
                        resid(b, a, c, k, j, i) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3d_110111

subroutine update_t3d_111011(t3d, resid, X3D, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3D(1:nub_act, 1:nub_act, 1:nub_act, 1:nob_inact, 1:nob_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3d(1:nub_act, 1:nub_act, 1:nub_act, 1:nob_inact, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: t3d(0:nub_act-1, 0:nub_act-1, 0:nub_act-1, 0:nob_inact-1, 0:nob_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nub_act, 1:nub_act, 1:nub_act, 1:nob_inact, 1:nob_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , nob_inact
         do j = 1 , nob_act
            do k = j+1 , nob_act
               do a = 1 , nub_act
                  do b = a+1 , nub_act
                     do c = b+1 , nub_act

                        denom = fB_oo_inact(i,i) + fB_oo_act(j,j) + fB_oo_act(k,k)&
                               -fB_vv_act(a,a) - fB_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3D(a, b, c, i, j, k)/(denom - shift)

                        t3d(a, b, c, i, j, k) = t3d(a, b, c, i, j, k) + val
                        t3d(a, b, c, i, k, j) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(a, c, b, i, j, k) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(a, c, b, i, k, j) = t3d(a, b, c, i, j, k)
                        t3d(b, a, c, i, j, k) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(b, a, c, i, k, j) = t3d(a, b, c, i, j, k)
                        t3d(b, c, a, i, j, k) = t3d(a, b, c, i, j, k)
                        t3d(b, c, a, i, k, j) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(c, a, b, i, j, k) = t3d(a, b, c, i, j, k)
                        t3d(c, a, b, i, k, j) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(c, b, a, i, j, k) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(c, b, a, i, k, j) = t3d(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, i, k, j) = val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, i, k, j) = val
                        resid(b, c, a, i, j, k) = val
                        resid(b, c, a, i, k, j) = -1.0 * val
                        resid(c, a, b, i, j, k) = val
                        resid(c, a, b, i, k, j) = -1.0 * val
                        resid(c, b, a, i, j, k) = -1.0 * val
                        resid(c, b, a, i, k, j) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3d_111011

subroutine update_t3d_110011(t3d, resid, X3D, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3D(1:nub_act, 1:nub_act, 1:nub_inact, 1:nob_inact, 1:nob_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3d(1:nub_act, 1:nub_act, 1:nub_inact, 1:nob_inact, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: t3d(0:nub_act-1, 0:nub_act-1, 0:nub_inact-1, 0:nob_inact-1, 0:nob_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nub_act, 1:nub_act, 1:nub_inact, 1:nob_inact, 1:nob_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , nob_inact
         do j = 1 , nob_act
            do k = j+1 , nob_act
               do a = 1 , nub_act
                  do b = a+1 , nub_act
                     do c = 1 , nub_inact

                        denom = fB_oo_inact(i,i) + fB_oo_act(j,j) + fB_oo_act(k,k)&
                               -fB_vv_act(a,a) - fB_vv_act(b,b) - fB_vv_inact(c,c)

                        val = X3D(a, b, c, i, j, k)/(denom - shift)

                        t3d(a, b, c, i, j, k) = t3d(a, b, c, i, j, k) + val
                        t3d(a, b, c, i, k, j) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(b, a, c, i, j, k) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(b, a, c, i, k, j) = t3d(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, i, k, j) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3d_110011

subroutine update_t3d_100111(t3d, resid, X3D, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3D(1:nub_act, 1:nub_inact, 1:nub_inact, 1:nob_act, 1:nob_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3d(1:nub_act, 1:nub_inact, 1:nub_inact, 1:nob_act, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: t3d(0:nub_act-1, 0:nub_inact-1, 0:nub_inact-1, 0:nob_act-1, 0:nob_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nub_act, 1:nub_inact, 1:nub_inact, 1:nob_act, 1:nob_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , nob_act
         do j = i+1 , nob_act
            do k = j+1 , nob_act
               do a = 1 , nub_act
                  do b = 1 , nub_inact
                     do c = b+1 , nub_inact

                        denom = fB_oo_act(i,i) + fB_oo_act(j,j) + fB_oo_act(k,k)&
                               -fB_vv_act(a,a) - fB_vv_inact(b,b) - fB_vv_inact(c,c)

                        val = X3D(a, b, c, i, j, k)/(denom - shift)

                        t3d(a, b, c, i, j, k) = t3d(a, b, c, i, j, k) + val
                        t3d(a, b, c, i, k, j) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(a, b, c, j, i, k) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(a, b, c, j, k, i) = t3d(a, b, c, i, j, k)
                        t3d(a, b, c, k, i, j) = t3d(a, b, c, i, j, k)
                        t3d(a, b, c, k, j, i) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(a, c, b, i, j, k) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(a, c, b, i, k, j) = t3d(a, b, c, i, j, k)
                        t3d(a, c, b, j, i, k) = t3d(a, b, c, i, j, k)
                        t3d(a, c, b, j, k, i) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(a, c, b, k, i, j) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(a, c, b, k, j, i) = t3d(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(a, b, c, j, k, i) = val
                        resid(a, b, c, k, i, j) = val
                        resid(a, b, c, k, j, i) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, i, k, j) = val
                        resid(a, c, b, j, i, k) = val
                        resid(a, c, b, j, k, i) = -1.0 * val
                        resid(a, c, b, k, i, j) = -1.0 * val
                        resid(a, c, b, k, j, i) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3d_100111

subroutine update_t3d_111001(t3d, resid, X3D, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3D(1:nub_act, 1:nub_act, 1:nub_act, 1:nob_inact, 1:nob_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3d(1:nub_act, 1:nub_act, 1:nub_act, 1:nob_inact, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3d(0:nub_act-1, 0:nub_act-1, 0:nub_act-1, 0:nob_inact-1, 0:nob_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nub_act, 1:nub_act, 1:nub_act, 1:nob_inact, 1:nob_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , nob_inact
         do j = i+1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nub_act
                  do b = a+1 , nub_act
                     do c = b+1 , nub_act

                        denom = fB_oo_inact(i,i) + fB_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fB_vv_act(a,a) - fB_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3D(a, b, c, i, j, k)/(denom - shift)

                        t3d(a, b, c, i, j, k) = t3d(a, b, c, i, j, k) + val
                        t3d(a, b, c, j, i, k) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(a, c, b, i, j, k) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(a, c, b, j, i, k) = t3d(a, b, c, i, j, k)
                        t3d(b, a, c, i, j, k) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(b, a, c, j, i, k) = t3d(a, b, c, i, j, k)
                        t3d(b, c, a, i, j, k) = t3d(a, b, c, i, j, k)
                        t3d(b, c, a, j, i, k) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(c, a, b, i, j, k) = t3d(a, b, c, i, j, k)
                        t3d(c, a, b, j, i, k) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(c, b, a, i, j, k) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(c, b, a, j, i, k) = t3d(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, j, i, k) = val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, j, i, k) = val
                        resid(b, c, a, i, j, k) = val
                        resid(b, c, a, j, i, k) = -1.0 * val
                        resid(c, a, b, i, j, k) = val
                        resid(c, a, b, j, i, k) = -1.0 * val
                        resid(c, b, a, i, j, k) = -1.0 * val
                        resid(c, b, a, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3d_111001

subroutine update_t3d_110001(t3d, resid, X3D, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3D(1:nub_act, 1:nub_act, 1:nub_inact, 1:nob_inact, 1:nob_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3d(1:nub_act, 1:nub_act, 1:nub_inact, 1:nob_inact, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3d(0:nub_act-1, 0:nub_act-1, 0:nub_inact-1, 0:nob_inact-1, 0:nob_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nub_act, 1:nub_act, 1:nub_inact, 1:nob_inact, 1:nob_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , nob_inact
         do j = i+1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nub_act
                  do b = a+1 , nub_act
                     do c = 1 , nub_inact

                        denom = fB_oo_inact(i,i) + fB_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fB_vv_act(a,a) - fB_vv_act(b,b) - fB_vv_inact(c,c)

                        val = X3D(a, b, c, i, j, k)/(denom - shift)

                        t3d(a, b, c, i, j, k) = t3d(a, b, c, i, j, k) + val
                        t3d(a, b, c, j, i, k) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(b, a, c, i, j, k) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(b, a, c, j, i, k) = t3d(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3d_110001

subroutine update_t3d_100011(t3d, resid, X3D, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3D(1:nub_act, 1:nub_inact, 1:nub_inact, 1:nob_inact, 1:nob_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3d(1:nub_act, 1:nub_inact, 1:nub_inact, 1:nob_inact, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: t3d(0:nub_act-1, 0:nub_inact-1, 0:nub_inact-1, 0:nob_inact-1, 0:nob_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nub_act, 1:nub_inact, 1:nub_inact, 1:nob_inact, 1:nob_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , nob_inact
         do j = 1 , nob_act
            do k = j+1 , nob_act
               do a = 1 , nub_act
                  do b = 1 , nub_inact
                     do c = b+1 , nub_inact

                        denom = fB_oo_inact(i,i) + fB_oo_act(j,j) + fB_oo_act(k,k)&
                               -fB_vv_act(a,a) - fB_vv_inact(b,b) - fB_vv_inact(c,c)

                        val = X3D(a, b, c, i, j, k)/(denom - shift)

                        t3d(a, b, c, i, j, k) = t3d(a, b, c, i, j, k) + val
                        t3d(a, b, c, i, k, j) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(a, c, b, i, j, k) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(a, c, b, i, k, j) = t3d(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, i, k, j) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3d_100011

subroutine update_t3d_100001(t3d, resid, X3D, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3D(1:nub_act, 1:nub_inact, 1:nub_inact, 1:nob_inact, 1:nob_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3d(1:nub_act, 1:nub_inact, 1:nub_inact, 1:nob_inact, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3d(0:nub_act-1, 0:nub_inact-1, 0:nub_inact-1, 0:nob_inact-1, 0:nob_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nub_act, 1:nub_inact, 1:nub_inact, 1:nob_inact, 1:nob_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , nob_inact
         do j = i+1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nub_act
                  do b = 1 , nub_inact
                     do c = b+1 , nub_inact

                        denom = fB_oo_inact(i,i) + fB_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fB_vv_act(a,a) - fB_vv_inact(b,b) - fB_vv_inact(c,c)

                        val = X3D(a, b, c, i, j, k)/(denom - shift)

                        t3d(a, b, c, i, j, k) = t3d(a, b, c, i, j, k) + val
                        t3d(a, b, c, j, i, k) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(a, c, b, i, j, k) = -1.0 * t3d(a, b, c, i, j, k)
                        t3d(a, c, b, j, i, k) = t3d(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3d_100001

end module cc_active_loops
