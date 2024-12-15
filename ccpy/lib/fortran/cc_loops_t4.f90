module cc_loops_t4

    implicit none

    contains

subroutine update_t4a(t4a, resid, X4A, &
                             fA_oo, fA_vv, &
                             shift, &
                             noa, nua)

      integer, intent(in)  :: noa, nua
      real(8), intent(in)  :: fA_oo(1:noa, 1:noa), &
                              fA_vv(1:nua, 1:nua)
      real(8), intent(in)  :: X4A(1:nua, 1:nua, 1:nua, 1:nua, 1:noa, 1:noa, 1:noa, 1:noa)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t4a(1:nua, 1:nua, 1:nua, 1:nua, 1:noa, 1:noa, 1:noa, 1:noa)
      !f2py intent(in, out)  :: t4a(0:nua-1, 0:nua-1, 0:nua-1, 0:nua-1, 0:noa-1, 0:noa-1, 0:noa-1, 0:noa-1)
      real(8), intent(out)   :: resid(1:nua, 1:nua, 1:nua, 1:nua, 1:noa, 1:noa, 1:noa, 1:noa)

      integer :: i, j, k, l, a, b, c, d
      real(8) :: denom, val

   do i = 1 , noa
      do j = i + 1 , noa
         do k = j + 1 , noa
            do l = k + 1 , noa
               do a = 1 , nua
                  do b = a + 1 , nua
                     do c = b + 1 , nua
                        do d = c + 1 , nua

                        denom = fA_oo(i,i) + fA_oo(j,j) + fA_oo(k,k) + fA_oo(l,l)&
                               -fA_vv(a,a) - fA_vv(b,b) - fA_vv(c,c) - fA_vv(d,d)

                        val = &
                        +X4A(a, b, c, d, i, j, k, l)&
                        -1.0 * X4A(a, b, c, d, i, j, l, k)&
                        -1.0 * X4A(a, b, c, d, i, k, j, l)&
                        +X4A(a, b, c, d, i, k, l, j)&
                        +X4A(a, b, c, d, i, l, j, k)&
                        -1.0 * X4A(a, b, c, d, i, l, k, j)&
                        -1.0 * X4A(a, b, c, d, j, i, k, l)&
                        +X4A(a, b, c, d, j, i, l, k)&
                        +X4A(a, b, c, d, j, k, i, l)&
                        -1.0 * X4A(a, b, c, d, j, k, l, i)&
                        -1.0 * X4A(a, b, c, d, j, l, i, k)&
                        +X4A(a, b, c, d, j, l, k, i)&
                        +X4A(a, b, c, d, k, i, j, l)&
                        -1.0 * X4A(a, b, c, d, k, i, l, j)&
                        -1.0 * X4A(a, b, c, d, k, j, i, l)&
                        +X4A(a, b, c, d, k, j, l, i)&
                        +X4A(a, b, c, d, k, l, i, j)&
                        -1.0 * X4A(a, b, c, d, k, l, j, i)&
                        -1.0 * X4A(a, b, c, d, l, i, j, k)&
                        +X4A(a, b, c, d, l, i, k, j)&
                        +X4A(a, b, c, d, l, j, i, k)&
                        -1.0 * X4A(a, b, c, d, l, j, k, i)&
                        -1.0 * X4A(a, b, c, d, l, k, i, j)&
                        +X4A(a, b, c, d, l, k, j, i)&
                        -1.0 * X4A(a, b, d, c, i, j, k, l)&
                        +X4A(a, b, d, c, i, j, l, k)&
                        +X4A(a, b, d, c, i, k, j, l)&
                        -1.0 * X4A(a, b, d, c, i, k, l, j)&
                        -1.0 * X4A(a, b, d, c, i, l, j, k)&
                        +X4A(a, b, d, c, i, l, k, j)&
                        +X4A(a, b, d, c, j, i, k, l)&
                        -1.0 * X4A(a, b, d, c, j, i, l, k)&
                        -1.0 * X4A(a, b, d, c, j, k, i, l)&
                        +X4A(a, b, d, c, j, k, l, i)&
                        +X4A(a, b, d, c, j, l, i, k)&
                        -1.0 * X4A(a, b, d, c, j, l, k, i)&
                        -1.0 * X4A(a, b, d, c, k, i, j, l)&
                        +X4A(a, b, d, c, k, i, l, j)&
                        +X4A(a, b, d, c, k, j, i, l)&
                        -1.0 * X4A(a, b, d, c, k, j, l, i)&
                        -1.0 * X4A(a, b, d, c, k, l, i, j)&
                        +X4A(a, b, d, c, k, l, j, i)&
                        +X4A(a, b, d, c, l, i, j, k)&
                        -1.0 * X4A(a, b, d, c, l, i, k, j)&
                        -1.0 * X4A(a, b, d, c, l, j, i, k)&
                        +X4A(a, b, d, c, l, j, k, i)&
                        +X4A(a, b, d, c, l, k, i, j)&
                        -1.0 * X4A(a, b, d, c, l, k, j, i)&
                        -1.0 * X4A(a, c, b, d, i, j, k, l)&
                        +X4A(a, c, b, d, i, j, l, k)&
                        +X4A(a, c, b, d, i, k, j, l)&
                        -1.0 * X4A(a, c, b, d, i, k, l, j)&
                        -1.0 * X4A(a, c, b, d, i, l, j, k)&
                        +X4A(a, c, b, d, i, l, k, j)&
                        +X4A(a, c, b, d, j, i, k, l)&
                        -1.0 * X4A(a, c, b, d, j, i, l, k)&
                        -1.0 * X4A(a, c, b, d, j, k, i, l)&
                        +X4A(a, c, b, d, j, k, l, i)&
                        +X4A(a, c, b, d, j, l, i, k)&
                        -1.0 * X4A(a, c, b, d, j, l, k, i)&
                        -1.0 * X4A(a, c, b, d, k, i, j, l)&
                        +X4A(a, c, b, d, k, i, l, j)&
                        +X4A(a, c, b, d, k, j, i, l)&
                        -1.0 * X4A(a, c, b, d, k, j, l, i)&
                        -1.0 * X4A(a, c, b, d, k, l, i, j)&
                        +X4A(a, c, b, d, k, l, j, i)&
                        +X4A(a, c, b, d, l, i, j, k)&
                        -1.0 * X4A(a, c, b, d, l, i, k, j)&
                        -1.0 * X4A(a, c, b, d, l, j, i, k)&
                        +X4A(a, c, b, d, l, j, k, i)&
                        +X4A(a, c, b, d, l, k, i, j)&
                        -1.0 * X4A(a, c, b, d, l, k, j, i)&
                        +X4A(a, c, d, b, i, j, k, l)&
                        -1.0 * X4A(a, c, d, b, i, j, l, k)&
                        -1.0 * X4A(a, c, d, b, i, k, j, l)&
                        +X4A(a, c, d, b, i, k, l, j)&
                        +X4A(a, c, d, b, i, l, j, k)&
                        -1.0 * X4A(a, c, d, b, i, l, k, j)&
                        -1.0 * X4A(a, c, d, b, j, i, k, l)&
                        +X4A(a, c, d, b, j, i, l, k)&
                        +X4A(a, c, d, b, j, k, i, l)&
                        -1.0 * X4A(a, c, d, b, j, k, l, i)&
                        -1.0 * X4A(a, c, d, b, j, l, i, k)&
                        +X4A(a, c, d, b, j, l, k, i)&
                        +X4A(a, c, d, b, k, i, j, l)&
                        -1.0 * X4A(a, c, d, b, k, i, l, j)&
                        -1.0 * X4A(a, c, d, b, k, j, i, l)&
                        +X4A(a, c, d, b, k, j, l, i)&
                        +X4A(a, c, d, b, k, l, i, j)&
                        -1.0 * X4A(a, c, d, b, k, l, j, i)&
                        -1.0 * X4A(a, c, d, b, l, i, j, k)&
                        +X4A(a, c, d, b, l, i, k, j)&
                        +X4A(a, c, d, b, l, j, i, k)&
                        -1.0 * X4A(a, c, d, b, l, j, k, i)&
                        -1.0 * X4A(a, c, d, b, l, k, i, j)&
                        +X4A(a, c, d, b, l, k, j, i)&
                        +X4A(a, d, b, c, i, j, k, l)&
                        -1.0 * X4A(a, d, b, c, i, j, l, k)&
                        -1.0 * X4A(a, d, b, c, i, k, j, l)&
                        +X4A(a, d, b, c, i, k, l, j)&
                        +X4A(a, d, b, c, i, l, j, k)&
                        -1.0 * X4A(a, d, b, c, i, l, k, j)&
                        -1.0 * X4A(a, d, b, c, j, i, k, l)&
                        +X4A(a, d, b, c, j, i, l, k)&
                        +X4A(a, d, b, c, j, k, i, l)&
                        -1.0 * X4A(a, d, b, c, j, k, l, i)&
                        -1.0 * X4A(a, d, b, c, j, l, i, k)&
                        +X4A(a, d, b, c, j, l, k, i)&
                        +X4A(a, d, b, c, k, i, j, l)&
                        -1.0 * X4A(a, d, b, c, k, i, l, j)&
                        -1.0 * X4A(a, d, b, c, k, j, i, l)&
                        +X4A(a, d, b, c, k, j, l, i)&
                        +X4A(a, d, b, c, k, l, i, j)&
                        -1.0 * X4A(a, d, b, c, k, l, j, i)&
                        -1.0 * X4A(a, d, b, c, l, i, j, k)&
                        +X4A(a, d, b, c, l, i, k, j)&
                        +X4A(a, d, b, c, l, j, i, k)&
                        -1.0 * X4A(a, d, b, c, l, j, k, i)&
                        -1.0 * X4A(a, d, b, c, l, k, i, j)&
                        +X4A(a, d, b, c, l, k, j, i)&
                        -1.0 * X4A(a, d, c, b, i, j, k, l)&
                        +X4A(a, d, c, b, i, j, l, k)&
                        +X4A(a, d, c, b, i, k, j, l)&
                        -1.0 * X4A(a, d, c, b, i, k, l, j)&
                        -1.0 * X4A(a, d, c, b, i, l, j, k)&
                        +X4A(a, d, c, b, i, l, k, j)&
                        +X4A(a, d, c, b, j, i, k, l)&
                        -1.0 * X4A(a, d, c, b, j, i, l, k)&
                        -1.0 * X4A(a, d, c, b, j, k, i, l)&
                        +X4A(a, d, c, b, j, k, l, i)&
                        +X4A(a, d, c, b, j, l, i, k)&
                        -1.0 * X4A(a, d, c, b, j, l, k, i)&
                        -1.0 * X4A(a, d, c, b, k, i, j, l)&
                        +X4A(a, d, c, b, k, i, l, j)&
                        +X4A(a, d, c, b, k, j, i, l)&
                        -1.0 * X4A(a, d, c, b, k, j, l, i)&
                        -1.0 * X4A(a, d, c, b, k, l, i, j)&
                        +X4A(a, d, c, b, k, l, j, i)&
                        +X4A(a, d, c, b, l, i, j, k)&
                        -1.0 * X4A(a, d, c, b, l, i, k, j)&
                        -1.0 * X4A(a, d, c, b, l, j, i, k)&
                        +X4A(a, d, c, b, l, j, k, i)&
                        +X4A(a, d, c, b, l, k, i, j)&
                        -1.0 * X4A(a, d, c, b, l, k, j, i)&
                        -1.0 * X4A(b, a, c, d, i, j, k, l)&
                        +X4A(b, a, c, d, i, j, l, k)&
                        +X4A(b, a, c, d, i, k, j, l)&
                        -1.0 * X4A(b, a, c, d, i, k, l, j)&
                        -1.0 * X4A(b, a, c, d, i, l, j, k)&
                        +X4A(b, a, c, d, i, l, k, j)&
                        +X4A(b, a, c, d, j, i, k, l)&
                        -1.0 * X4A(b, a, c, d, j, i, l, k)&
                        -1.0 * X4A(b, a, c, d, j, k, i, l)&
                        +X4A(b, a, c, d, j, k, l, i)&
                        +X4A(b, a, c, d, j, l, i, k)&
                        -1.0 * X4A(b, a, c, d, j, l, k, i)&
                        -1.0 * X4A(b, a, c, d, k, i, j, l)&
                        +X4A(b, a, c, d, k, i, l, j)&
                        +X4A(b, a, c, d, k, j, i, l)&
                        -1.0 * X4A(b, a, c, d, k, j, l, i)&
                        -1.0 * X4A(b, a, c, d, k, l, i, j)&
                        +X4A(b, a, c, d, k, l, j, i)&
                        +X4A(b, a, c, d, l, i, j, k)&
                        -1.0 * X4A(b, a, c, d, l, i, k, j)&
                        -1.0 * X4A(b, a, c, d, l, j, i, k)&
                        +X4A(b, a, c, d, l, j, k, i)&
                        +X4A(b, a, c, d, l, k, i, j)&
                        -1.0 * X4A(b, a, c, d, l, k, j, i)&
                        +X4A(b, a, d, c, i, j, k, l)&
                        -1.0 * X4A(b, a, d, c, i, j, l, k)&
                        -1.0 * X4A(b, a, d, c, i, k, j, l)&
                        +X4A(b, a, d, c, i, k, l, j)&
                        +X4A(b, a, d, c, i, l, j, k)&
                        -1.0 * X4A(b, a, d, c, i, l, k, j)&
                        -1.0 * X4A(b, a, d, c, j, i, k, l)&
                        +X4A(b, a, d, c, j, i, l, k)&
                        +X4A(b, a, d, c, j, k, i, l)&
                        -1.0 * X4A(b, a, d, c, j, k, l, i)&
                        -1.0 * X4A(b, a, d, c, j, l, i, k)&
                        +X4A(b, a, d, c, j, l, k, i)&
                        +X4A(b, a, d, c, k, i, j, l)&
                        -1.0 * X4A(b, a, d, c, k, i, l, j)&
                        -1.0 * X4A(b, a, d, c, k, j, i, l)&
                        +X4A(b, a, d, c, k, j, l, i)&
                        +X4A(b, a, d, c, k, l, i, j)&
                        -1.0 * X4A(b, a, d, c, k, l, j, i)&
                        -1.0 * X4A(b, a, d, c, l, i, j, k)&
                        +X4A(b, a, d, c, l, i, k, j)&
                        +X4A(b, a, d, c, l, j, i, k)&
                        -1.0 * X4A(b, a, d, c, l, j, k, i)&
                        -1.0 * X4A(b, a, d, c, l, k, i, j)&
                        +X4A(b, a, d, c, l, k, j, i)&
                        +X4A(b, c, a, d, i, j, k, l)&
                        -1.0 * X4A(b, c, a, d, i, j, l, k)&
                        -1.0 * X4A(b, c, a, d, i, k, j, l)&
                        +X4A(b, c, a, d, i, k, l, j)&
                        +X4A(b, c, a, d, i, l, j, k)&
                        -1.0 * X4A(b, c, a, d, i, l, k, j)&
                        -1.0 * X4A(b, c, a, d, j, i, k, l)&
                        +X4A(b, c, a, d, j, i, l, k)&
                        +X4A(b, c, a, d, j, k, i, l)&
                        -1.0 * X4A(b, c, a, d, j, k, l, i)&
                        -1.0 * X4A(b, c, a, d, j, l, i, k)&
                        +X4A(b, c, a, d, j, l, k, i)&
                        +X4A(b, c, a, d, k, i, j, l)&
                        -1.0 * X4A(b, c, a, d, k, i, l, j)&
                        -1.0 * X4A(b, c, a, d, k, j, i, l)&
                        +X4A(b, c, a, d, k, j, l, i)&
                        +X4A(b, c, a, d, k, l, i, j)&
                        -1.0 * X4A(b, c, a, d, k, l, j, i)&
                        -1.0 * X4A(b, c, a, d, l, i, j, k)&
                        +X4A(b, c, a, d, l, i, k, j)&
                        +X4A(b, c, a, d, l, j, i, k)&
                        -1.0 * X4A(b, c, a, d, l, j, k, i)&
                        -1.0 * X4A(b, c, a, d, l, k, i, j)&
                        +X4A(b, c, a, d, l, k, j, i)&
                        -1.0 * X4A(b, c, d, a, i, j, k, l)&
                        +X4A(b, c, d, a, i, j, l, k)&
                        +X4A(b, c, d, a, i, k, j, l)&
                        -1.0 * X4A(b, c, d, a, i, k, l, j)&
                        -1.0 * X4A(b, c, d, a, i, l, j, k)&
                        +X4A(b, c, d, a, i, l, k, j)&
                        +X4A(b, c, d, a, j, i, k, l)&
                        -1.0 * X4A(b, c, d, a, j, i, l, k)&
                        -1.0 * X4A(b, c, d, a, j, k, i, l)&
                        +X4A(b, c, d, a, j, k, l, i)&
                        +X4A(b, c, d, a, j, l, i, k)&
                        -1.0 * X4A(b, c, d, a, j, l, k, i)&
                        -1.0 * X4A(b, c, d, a, k, i, j, l)&
                        +X4A(b, c, d, a, k, i, l, j)&
                        +X4A(b, c, d, a, k, j, i, l)&
                        -1.0 * X4A(b, c, d, a, k, j, l, i)&
                        -1.0 * X4A(b, c, d, a, k, l, i, j)&
                        +X4A(b, c, d, a, k, l, j, i)&
                        +X4A(b, c, d, a, l, i, j, k)&
                        -1.0 * X4A(b, c, d, a, l, i, k, j)&
                        -1.0 * X4A(b, c, d, a, l, j, i, k)&
                        +X4A(b, c, d, a, l, j, k, i)&
                        +X4A(b, c, d, a, l, k, i, j)&
                        -1.0 * X4A(b, c, d, a, l, k, j, i)&
                        -1.0 * X4A(b, d, a, c, i, j, k, l)&
                        +X4A(b, d, a, c, i, j, l, k)&
                        +X4A(b, d, a, c, i, k, j, l)&
                        -1.0 * X4A(b, d, a, c, i, k, l, j)&
                        -1.0 * X4A(b, d, a, c, i, l, j, k)&
                        +X4A(b, d, a, c, i, l, k, j)&
                        +X4A(b, d, a, c, j, i, k, l)&
                        -1.0 * X4A(b, d, a, c, j, i, l, k)&
                        -1.0 * X4A(b, d, a, c, j, k, i, l)&
                        +X4A(b, d, a, c, j, k, l, i)&
                        +X4A(b, d, a, c, j, l, i, k)&
                        -1.0 * X4A(b, d, a, c, j, l, k, i)&
                        -1.0 * X4A(b, d, a, c, k, i, j, l)&
                        +X4A(b, d, a, c, k, i, l, j)&
                        +X4A(b, d, a, c, k, j, i, l)

                        val = val &
                        -1.0 * X4A(b, d, a, c, k, j, l, i)&
                        -1.0 * X4A(b, d, a, c, k, l, i, j)&
                        +X4A(b, d, a, c, k, l, j, i)&
                        +X4A(b, d, a, c, l, i, j, k)&
                        -1.0 * X4A(b, d, a, c, l, i, k, j)&
                        -1.0 * X4A(b, d, a, c, l, j, i, k)&
                        +X4A(b, d, a, c, l, j, k, i)&
                        +X4A(b, d, a, c, l, k, i, j)&
                        -1.0 * X4A(b, d, a, c, l, k, j, i)&
                        +X4A(b, d, c, a, i, j, k, l)&
                        -1.0 * X4A(b, d, c, a, i, j, l, k)&
                        -1.0 * X4A(b, d, c, a, i, k, j, l)&
                        +X4A(b, d, c, a, i, k, l, j)&
                        +X4A(b, d, c, a, i, l, j, k)&
                        -1.0 * X4A(b, d, c, a, i, l, k, j)&
                        -1.0 * X4A(b, d, c, a, j, i, k, l)&
                        +X4A(b, d, c, a, j, i, l, k)&
                        +X4A(b, d, c, a, j, k, i, l)&
                        -1.0 * X4A(b, d, c, a, j, k, l, i)&
                        -1.0 * X4A(b, d, c, a, j, l, i, k)&
                        +X4A(b, d, c, a, j, l, k, i)&
                        +X4A(b, d, c, a, k, i, j, l)&
                        -1.0 * X4A(b, d, c, a, k, i, l, j)&
                        -1.0 * X4A(b, d, c, a, k, j, i, l)&
                        +X4A(b, d, c, a, k, j, l, i)&
                        +X4A(b, d, c, a, k, l, i, j)&
                        -1.0 * X4A(b, d, c, a, k, l, j, i)&
                        -1.0 * X4A(b, d, c, a, l, i, j, k)&
                        +X4A(b, d, c, a, l, i, k, j)&
                        +X4A(b, d, c, a, l, j, i, k)&
                        -1.0 * X4A(b, d, c, a, l, j, k, i)&
                        -1.0 * X4A(b, d, c, a, l, k, i, j)&
                        +X4A(b, d, c, a, l, k, j, i)&
                        +X4A(c, a, b, d, i, j, k, l)&
                        -1.0 * X4A(c, a, b, d, i, j, l, k)&
                        -1.0 * X4A(c, a, b, d, i, k, j, l)&
                        +X4A(c, a, b, d, i, k, l, j)&
                        +X4A(c, a, b, d, i, l, j, k)&
                        -1.0 * X4A(c, a, b, d, i, l, k, j)&
                        -1.0 * X4A(c, a, b, d, j, i, k, l)&
                        +X4A(c, a, b, d, j, i, l, k)&
                        +X4A(c, a, b, d, j, k, i, l)&
                        -1.0 * X4A(c, a, b, d, j, k, l, i)&
                        -1.0 * X4A(c, a, b, d, j, l, i, k)&
                        +X4A(c, a, b, d, j, l, k, i)&
                        +X4A(c, a, b, d, k, i, j, l)&
                        -1.0 * X4A(c, a, b, d, k, i, l, j)&
                        -1.0 * X4A(c, a, b, d, k, j, i, l)&
                        +X4A(c, a, b, d, k, j, l, i)&
                        +X4A(c, a, b, d, k, l, i, j)&
                        -1.0 * X4A(c, a, b, d, k, l, j, i)&
                        -1.0 * X4A(c, a, b, d, l, i, j, k)&
                        +X4A(c, a, b, d, l, i, k, j)&
                        +X4A(c, a, b, d, l, j, i, k)&
                        -1.0 * X4A(c, a, b, d, l, j, k, i)&
                        -1.0 * X4A(c, a, b, d, l, k, i, j)&
                        +X4A(c, a, b, d, l, k, j, i)&
                        -1.0 * X4A(c, a, d, b, i, j, k, l)&
                        +X4A(c, a, d, b, i, j, l, k)&
                        +X4A(c, a, d, b, i, k, j, l)&
                        -1.0 * X4A(c, a, d, b, i, k, l, j)&
                        -1.0 * X4A(c, a, d, b, i, l, j, k)&
                        +X4A(c, a, d, b, i, l, k, j)&
                        +X4A(c, a, d, b, j, i, k, l)&
                        -1.0 * X4A(c, a, d, b, j, i, l, k)&
                        -1.0 * X4A(c, a, d, b, j, k, i, l)&
                        +X4A(c, a, d, b, j, k, l, i)&
                        +X4A(c, a, d, b, j, l, i, k)&
                        -1.0 * X4A(c, a, d, b, j, l, k, i)&
                        -1.0 * X4A(c, a, d, b, k, i, j, l)&
                        +X4A(c, a, d, b, k, i, l, j)&
                        +X4A(c, a, d, b, k, j, i, l)&
                        -1.0 * X4A(c, a, d, b, k, j, l, i)&
                        -1.0 * X4A(c, a, d, b, k, l, i, j)&
                        +X4A(c, a, d, b, k, l, j, i)&
                        +X4A(c, a, d, b, l, i, j, k)&
                        -1.0 * X4A(c, a, d, b, l, i, k, j)&
                        -1.0 * X4A(c, a, d, b, l, j, i, k)&
                        +X4A(c, a, d, b, l, j, k, i)&
                        +X4A(c, a, d, b, l, k, i, j)&
                        -1.0 * X4A(c, a, d, b, l, k, j, i)&
                        -1.0 * X4A(c, b, a, d, i, j, k, l)&
                        +X4A(c, b, a, d, i, j, l, k)&
                        +X4A(c, b, a, d, i, k, j, l)&
                        -1.0 * X4A(c, b, a, d, i, k, l, j)&
                        -1.0 * X4A(c, b, a, d, i, l, j, k)&
                        +X4A(c, b, a, d, i, l, k, j)&
                        +X4A(c, b, a, d, j, i, k, l)&
                        -1.0 * X4A(c, b, a, d, j, i, l, k)&
                        -1.0 * X4A(c, b, a, d, j, k, i, l)&
                        +X4A(c, b, a, d, j, k, l, i)&
                        +X4A(c, b, a, d, j, l, i, k)&
                        -1.0 * X4A(c, b, a, d, j, l, k, i)&
                        -1.0 * X4A(c, b, a, d, k, i, j, l)&
                        +X4A(c, b, a, d, k, i, l, j)&
                        +X4A(c, b, a, d, k, j, i, l)&
                        -1.0 * X4A(c, b, a, d, k, j, l, i)&
                        -1.0 * X4A(c, b, a, d, k, l, i, j)&
                        +X4A(c, b, a, d, k, l, j, i)&
                        +X4A(c, b, a, d, l, i, j, k)&
                        -1.0 * X4A(c, b, a, d, l, i, k, j)&
                        -1.0 * X4A(c, b, a, d, l, j, i, k)&
                        +X4A(c, b, a, d, l, j, k, i)&
                        +X4A(c, b, a, d, l, k, i, j)&
                        -1.0 * X4A(c, b, a, d, l, k, j, i)&
                        +X4A(c, b, d, a, i, j, k, l)&
                        -1.0 * X4A(c, b, d, a, i, j, l, k)&
                        -1.0 * X4A(c, b, d, a, i, k, j, l)&
                        +X4A(c, b, d, a, i, k, l, j)&
                        +X4A(c, b, d, a, i, l, j, k)&
                        -1.0 * X4A(c, b, d, a, i, l, k, j)&
                        -1.0 * X4A(c, b, d, a, j, i, k, l)&
                        +X4A(c, b, d, a, j, i, l, k)&
                        +X4A(c, b, d, a, j, k, i, l)&
                        -1.0 * X4A(c, b, d, a, j, k, l, i)&
                        -1.0 * X4A(c, b, d, a, j, l, i, k)&
                        +X4A(c, b, d, a, j, l, k, i)&
                        +X4A(c, b, d, a, k, i, j, l)&
                        -1.0 * X4A(c, b, d, a, k, i, l, j)&
                        -1.0 * X4A(c, b, d, a, k, j, i, l)&
                        +X4A(c, b, d, a, k, j, l, i)&
                        +X4A(c, b, d, a, k, l, i, j)&
                        -1.0 * X4A(c, b, d, a, k, l, j, i)&
                        -1.0 * X4A(c, b, d, a, l, i, j, k)&
                        +X4A(c, b, d, a, l, i, k, j)&
                        +X4A(c, b, d, a, l, j, i, k)&
                        -1.0 * X4A(c, b, d, a, l, j, k, i)&
                        -1.0 * X4A(c, b, d, a, l, k, i, j)&
                        +X4A(c, b, d, a, l, k, j, i)&
                        +X4A(c, d, a, b, i, j, k, l)&
                        -1.0 * X4A(c, d, a, b, i, j, l, k)&
                        -1.0 * X4A(c, d, a, b, i, k, j, l)&
                        +X4A(c, d, a, b, i, k, l, j)&
                        +X4A(c, d, a, b, i, l, j, k)&
                        -1.0 * X4A(c, d, a, b, i, l, k, j)&
                        -1.0 * X4A(c, d, a, b, j, i, k, l)&
                        +X4A(c, d, a, b, j, i, l, k)&
                        +X4A(c, d, a, b, j, k, i, l)&
                        -1.0 * X4A(c, d, a, b, j, k, l, i)&
                        -1.0 * X4A(c, d, a, b, j, l, i, k)&
                        +X4A(c, d, a, b, j, l, k, i)&
                        +X4A(c, d, a, b, k, i, j, l)&
                        -1.0 * X4A(c, d, a, b, k, i, l, j)&
                        -1.0 * X4A(c, d, a, b, k, j, i, l)&
                        +X4A(c, d, a, b, k, j, l, i)&
                        +X4A(c, d, a, b, k, l, i, j)&
                        -1.0 * X4A(c, d, a, b, k, l, j, i)&
                        -1.0 * X4A(c, d, a, b, l, i, j, k)&
                        +X4A(c, d, a, b, l, i, k, j)&
                        +X4A(c, d, a, b, l, j, i, k)&
                        -1.0 * X4A(c, d, a, b, l, j, k, i)&
                        -1.0 * X4A(c, d, a, b, l, k, i, j)&
                        +X4A(c, d, a, b, l, k, j, i)&
                        -1.0 * X4A(c, d, b, a, i, j, k, l)&
                        +X4A(c, d, b, a, i, j, l, k)&
                        +X4A(c, d, b, a, i, k, j, l)&
                        -1.0 * X4A(c, d, b, a, i, k, l, j)&
                        -1.0 * X4A(c, d, b, a, i, l, j, k)&
                        +X4A(c, d, b, a, i, l, k, j)&
                        +X4A(c, d, b, a, j, i, k, l)&
                        -1.0 * X4A(c, d, b, a, j, i, l, k)&
                        -1.0 * X4A(c, d, b, a, j, k, i, l)&
                        +X4A(c, d, b, a, j, k, l, i)&
                        +X4A(c, d, b, a, j, l, i, k)&
                        -1.0 * X4A(c, d, b, a, j, l, k, i)&
                        -1.0 * X4A(c, d, b, a, k, i, j, l)&
                        +X4A(c, d, b, a, k, i, l, j)&
                        +X4A(c, d, b, a, k, j, i, l)&
                        -1.0 * X4A(c, d, b, a, k, j, l, i)&
                        -1.0 * X4A(c, d, b, a, k, l, i, j)&
                        +X4A(c, d, b, a, k, l, j, i)&
                        +X4A(c, d, b, a, l, i, j, k)&
                        -1.0 * X4A(c, d, b, a, l, i, k, j)&
                        -1.0 * X4A(c, d, b, a, l, j, i, k)&
                        +X4A(c, d, b, a, l, j, k, i)&
                        +X4A(c, d, b, a, l, k, i, j)&
                        -1.0 * X4A(c, d, b, a, l, k, j, i)&
                        -1.0 * X4A(d, a, b, c, i, j, k, l)&
                        +X4A(d, a, b, c, i, j, l, k)&
                        +X4A(d, a, b, c, i, k, j, l)&
                        -1.0 * X4A(d, a, b, c, i, k, l, j)&
                        -1.0 * X4A(d, a, b, c, i, l, j, k)&
                        +X4A(d, a, b, c, i, l, k, j)&
                        +X4A(d, a, b, c, j, i, k, l)&
                        -1.0 * X4A(d, a, b, c, j, i, l, k)&
                        -1.0 * X4A(d, a, b, c, j, k, i, l)&
                        +X4A(d, a, b, c, j, k, l, i)&
                        +X4A(d, a, b, c, j, l, i, k)&
                        -1.0 * X4A(d, a, b, c, j, l, k, i)&
                        -1.0 * X4A(d, a, b, c, k, i, j, l)&
                        +X4A(d, a, b, c, k, i, l, j)&
                        +X4A(d, a, b, c, k, j, i, l)&
                        -1.0 * X4A(d, a, b, c, k, j, l, i)&
                        -1.0 * X4A(d, a, b, c, k, l, i, j)&
                        +X4A(d, a, b, c, k, l, j, i)&
                        +X4A(d, a, b, c, l, i, j, k)&
                        -1.0 * X4A(d, a, b, c, l, i, k, j)&
                        -1.0 * X4A(d, a, b, c, l, j, i, k)&
                        +X4A(d, a, b, c, l, j, k, i)&
                        +X4A(d, a, b, c, l, k, i, j)&
                        -1.0 * X4A(d, a, b, c, l, k, j, i)&
                        +X4A(d, a, c, b, i, j, k, l)&
                        -1.0 * X4A(d, a, c, b, i, j, l, k)&
                        -1.0 * X4A(d, a, c, b, i, k, j, l)&
                        +X4A(d, a, c, b, i, k, l, j)&
                        +X4A(d, a, c, b, i, l, j, k)&
                        -1.0 * X4A(d, a, c, b, i, l, k, j)&
                        -1.0 * X4A(d, a, c, b, j, i, k, l)&
                        +X4A(d, a, c, b, j, i, l, k)&
                        +X4A(d, a, c, b, j, k, i, l)&
                        -1.0 * X4A(d, a, c, b, j, k, l, i)&
                        -1.0 * X4A(d, a, c, b, j, l, i, k)&
                        +X4A(d, a, c, b, j, l, k, i)&
                        +X4A(d, a, c, b, k, i, j, l)&
                        -1.0 * X4A(d, a, c, b, k, i, l, j)&
                        -1.0 * X4A(d, a, c, b, k, j, i, l)&
                        +X4A(d, a, c, b, k, j, l, i)&
                        +X4A(d, a, c, b, k, l, i, j)&
                        -1.0 * X4A(d, a, c, b, k, l, j, i)&
                        -1.0 * X4A(d, a, c, b, l, i, j, k)&
                        +X4A(d, a, c, b, l, i, k, j)&
                        +X4A(d, a, c, b, l, j, i, k)&
                        -1.0 * X4A(d, a, c, b, l, j, k, i)&
                        -1.0 * X4A(d, a, c, b, l, k, i, j)&
                        +X4A(d, a, c, b, l, k, j, i)&
                        +X4A(d, b, a, c, i, j, k, l)&
                        -1.0 * X4A(d, b, a, c, i, j, l, k)&
                        -1.0 * X4A(d, b, a, c, i, k, j, l)&
                        +X4A(d, b, a, c, i, k, l, j)&
                        +X4A(d, b, a, c, i, l, j, k)&
                        -1.0 * X4A(d, b, a, c, i, l, k, j)&
                        -1.0 * X4A(d, b, a, c, j, i, k, l)&
                        +X4A(d, b, a, c, j, i, l, k)&
                        +X4A(d, b, a, c, j, k, i, l)&
                        -1.0 * X4A(d, b, a, c, j, k, l, i)&
                        -1.0 * X4A(d, b, a, c, j, l, i, k)&
                        +X4A(d, b, a, c, j, l, k, i)&
                        +X4A(d, b, a, c, k, i, j, l)&
                        -1.0 * X4A(d, b, a, c, k, i, l, j)&
                        -1.0 * X4A(d, b, a, c, k, j, i, l)&
                        +X4A(d, b, a, c, k, j, l, i)&
                        +X4A(d, b, a, c, k, l, i, j)&
                        -1.0 * X4A(d, b, a, c, k, l, j, i)&
                        -1.0 * X4A(d, b, a, c, l, i, j, k)&
                        +X4A(d, b, a, c, l, i, k, j)&
                        +X4A(d, b, a, c, l, j, i, k)&
                        -1.0 * X4A(d, b, a, c, l, j, k, i)&
                        -1.0 * X4A(d, b, a, c, l, k, i, j)&
                        +X4A(d, b, a, c, l, k, j, i)&
                        -1.0 * X4A(d, b, c, a, i, j, k, l)&
                        +X4A(d, b, c, a, i, j, l, k)&
                        +X4A(d, b, c, a, i, k, j, l)&
                        -1.0 * X4A(d, b, c, a, i, k, l, j)&
                        -1.0 * X4A(d, b, c, a, i, l, j, k)&
                        +X4A(d, b, c, a, i, l, k, j)

                        val = val &
                        +X4A(d, b, c, a, j, i, k, l)&
                        -1.0 * X4A(d, b, c, a, j, i, l, k)&
                        -1.0 * X4A(d, b, c, a, j, k, i, l)&
                        +X4A(d, b, c, a, j, k, l, i)&
                        +X4A(d, b, c, a, j, l, i, k)&
                        -1.0 * X4A(d, b, c, a, j, l, k, i)&
                        -1.0 * X4A(d, b, c, a, k, i, j, l)&
                        +X4A(d, b, c, a, k, i, l, j)&
                        +X4A(d, b, c, a, k, j, i, l)&
                        -1.0 * X4A(d, b, c, a, k, j, l, i)&
                        -1.0 * X4A(d, b, c, a, k, l, i, j)&
                        +X4A(d, b, c, a, k, l, j, i)&
                        +X4A(d, b, c, a, l, i, j, k)&
                        -1.0 * X4A(d, b, c, a, l, i, k, j)&
                        -1.0 * X4A(d, b, c, a, l, j, i, k)&
                        +X4A(d, b, c, a, l, j, k, i)&
                        +X4A(d, b, c, a, l, k, i, j)&
                        -1.0 * X4A(d, b, c, a, l, k, j, i)&
                        -1.0 * X4A(d, c, a, b, i, j, k, l)&
                        +X4A(d, c, a, b, i, j, l, k)&
                        +X4A(d, c, a, b, i, k, j, l)&
                        -1.0 * X4A(d, c, a, b, i, k, l, j)&
                        -1.0 * X4A(d, c, a, b, i, l, j, k)&
                        +X4A(d, c, a, b, i, l, k, j)&
                        +X4A(d, c, a, b, j, i, k, l)&
                        -1.0 * X4A(d, c, a, b, j, i, l, k)&
                        -1.0 * X4A(d, c, a, b, j, k, i, l)&
                        +X4A(d, c, a, b, j, k, l, i)&
                        +X4A(d, c, a, b, j, l, i, k)&
                        -1.0 * X4A(d, c, a, b, j, l, k, i)&
                        -1.0 * X4A(d, c, a, b, k, i, j, l)&
                        +X4A(d, c, a, b, k, i, l, j)&
                        +X4A(d, c, a, b, k, j, i, l)&
                        -1.0 * X4A(d, c, a, b, k, j, l, i)&
                        -1.0 * X4A(d, c, a, b, k, l, i, j)&
                        +X4A(d, c, a, b, k, l, j, i)&
                        +X4A(d, c, a, b, l, i, j, k)&
                        -1.0 * X4A(d, c, a, b, l, i, k, j)&
                        -1.0 * X4A(d, c, a, b, l, j, i, k)&
                        +X4A(d, c, a, b, l, j, k, i)&
                        +X4A(d, c, a, b, l, k, i, j)&
                        -1.0 * X4A(d, c, a, b, l, k, j, i)&
                        +X4A(d, c, b, a, i, j, k, l)&
                        -1.0 * X4A(d, c, b, a, i, j, l, k)&
                        -1.0 * X4A(d, c, b, a, i, k, j, l)&
                        +X4A(d, c, b, a, i, k, l, j)&
                        +X4A(d, c, b, a, i, l, j, k)&
                        -1.0 * X4A(d, c, b, a, i, l, k, j)&
                        -1.0 * X4A(d, c, b, a, j, i, k, l)&
                        +X4A(d, c, b, a, j, i, l, k)&
                        +X4A(d, c, b, a, j, k, i, l)&
                        -1.0 * X4A(d, c, b, a, j, k, l, i)&
                        -1.0 * X4A(d, c, b, a, j, l, i, k)&
                        +X4A(d, c, b, a, j, l, k, i)&
                        +X4A(d, c, b, a, k, i, j, l)&
                        -1.0 * X4A(d, c, b, a, k, i, l, j)&
                        -1.0 * X4A(d, c, b, a, k, j, i, l)&
                        +X4A(d, c, b, a, k, j, l, i)&
                        +X4A(d, c, b, a, k, l, i, j)&
                        -1.0 * X4A(d, c, b, a, k, l, j, i)&
                        -1.0 * X4A(d, c, b, a, l, i, j, k)&
                        +X4A(d, c, b, a, l, i, k, j)&
                        +X4A(d, c, b, a, l, j, i, k)&
                        -1.0 * X4A(d, c, b, a, l, j, k, i)&
                        -1.0 * X4A(d, c, b, a, l, k, i, j)&
                        +X4A(d, c, b, a, l, k, j, i)

                        t4a(a, b, c, d, i, j, k, l) = t4a(a, b, c, d, i, j, k, l) + val/(denom-shift)
                        t4a(a, b, c, d, i, j, l, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, c, d, i, k, j, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, c, d, i, k, l, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, c, d, i, l, j, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, c, d, i, l, k, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, c, d, j, i, k, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, c, d, j, i, l, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, c, d, j, k, i, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, c, d, j, k, l, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, c, d, j, l, i, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, c, d, j, l, k, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, c, d, k, i, j, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, c, d, k, i, l, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, c, d, k, j, i, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, c, d, k, j, l, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, c, d, k, l, i, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, c, d, k, l, j, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, c, d, l, i, j, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, c, d, l, i, k, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, c, d, l, j, i, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, c, d, l, j, k, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, c, d, l, k, i, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, c, d, l, k, j, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, d, c, i, j, k, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, d, c, i, j, l, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, d, c, i, k, j, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, d, c, i, k, l, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, d, c, i, l, j, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, d, c, i, l, k, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, d, c, j, i, k, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, d, c, j, i, l, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, d, c, j, k, i, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, d, c, j, k, l, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, d, c, j, l, i, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, d, c, j, l, k, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, d, c, k, i, j, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, d, c, k, i, l, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, d, c, k, j, i, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, d, c, k, j, l, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, d, c, k, l, i, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, d, c, k, l, j, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, d, c, l, i, j, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, d, c, l, i, k, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, d, c, l, j, i, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, d, c, l, j, k, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, d, c, l, k, i, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, b, d, c, l, k, j, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, b, d, i, j, k, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, b, d, i, j, l, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, b, d, i, k, j, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, b, d, i, k, l, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, b, d, i, l, j, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, b, d, i, l, k, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, b, d, j, i, k, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, b, d, j, i, l, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, b, d, j, k, i, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, b, d, j, k, l, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, b, d, j, l, i, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, b, d, j, l, k, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, b, d, k, i, j, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, b, d, k, i, l, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, b, d, k, j, i, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, b, d, k, j, l, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, b, d, k, l, i, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, b, d, k, l, j, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, b, d, l, i, j, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, b, d, l, i, k, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, b, d, l, j, i, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, b, d, l, j, k, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, b, d, l, k, i, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, b, d, l, k, j, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, d, b, i, j, k, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, d, b, i, j, l, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, d, b, i, k, j, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, d, b, i, k, l, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, d, b, i, l, j, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, d, b, i, l, k, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, d, b, j, i, k, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, d, b, j, i, l, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, d, b, j, k, i, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, d, b, j, k, l, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, d, b, j, l, i, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, d, b, j, l, k, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, d, b, k, i, j, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, d, b, k, i, l, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, d, b, k, j, i, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, d, b, k, j, l, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, d, b, k, l, i, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, d, b, k, l, j, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, d, b, l, i, j, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, d, b, l, i, k, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, d, b, l, j, i, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, d, b, l, j, k, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, d, b, l, k, i, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, c, d, b, l, k, j, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, b, c, i, j, k, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, b, c, i, j, l, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, b, c, i, k, j, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, b, c, i, k, l, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, b, c, i, l, j, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, b, c, i, l, k, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, b, c, j, i, k, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, b, c, j, i, l, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, b, c, j, k, i, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, b, c, j, k, l, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, b, c, j, l, i, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, b, c, j, l, k, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, b, c, k, i, j, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, b, c, k, i, l, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, b, c, k, j, i, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, b, c, k, j, l, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, b, c, k, l, i, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, b, c, k, l, j, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, b, c, l, i, j, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, b, c, l, i, k, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, b, c, l, j, i, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, b, c, l, j, k, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, b, c, l, k, i, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, b, c, l, k, j, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, c, b, i, j, k, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, c, b, i, j, l, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, c, b, i, k, j, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, c, b, i, k, l, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, c, b, i, l, j, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, c, b, i, l, k, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, c, b, j, i, k, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, c, b, j, i, l, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, c, b, j, k, i, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, c, b, j, k, l, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, c, b, j, l, i, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, c, b, j, l, k, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, c, b, k, i, j, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, c, b, k, i, l, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, c, b, k, j, i, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, c, b, k, j, l, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, c, b, k, l, i, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, c, b, k, l, j, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, c, b, l, i, j, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, c, b, l, i, k, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, c, b, l, j, i, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, c, b, l, j, k, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, c, b, l, k, i, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(a, d, c, b, l, k, j, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, c, d, i, j, k, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, c, d, i, j, l, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, c, d, i, k, j, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, c, d, i, k, l, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, c, d, i, l, j, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, c, d, i, l, k, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, c, d, j, i, k, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, c, d, j, i, l, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, c, d, j, k, i, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, c, d, j, k, l, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, c, d, j, l, i, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, c, d, j, l, k, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, c, d, k, i, j, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, c, d, k, i, l, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, c, d, k, j, i, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, c, d, k, j, l, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, c, d, k, l, i, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, c, d, k, l, j, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, c, d, l, i, j, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, c, d, l, i, k, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, c, d, l, j, i, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, c, d, l, j, k, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, c, d, l, k, i, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, c, d, l, k, j, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, d, c, i, j, k, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, d, c, i, j, l, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, d, c, i, k, j, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, d, c, i, k, l, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, d, c, i, l, j, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, d, c, i, l, k, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, d, c, j, i, k, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, d, c, j, i, l, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, d, c, j, k, i, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, d, c, j, k, l, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, d, c, j, l, i, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, d, c, j, l, k, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, d, c, k, i, j, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, d, c, k, i, l, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, d, c, k, j, i, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, d, c, k, j, l, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, d, c, k, l, i, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, d, c, k, l, j, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, d, c, l, i, j, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, d, c, l, i, k, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, d, c, l, j, i, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, d, c, l, j, k, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, d, c, l, k, i, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, a, d, c, l, k, j, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, a, d, i, j, k, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, a, d, i, j, l, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, a, d, i, k, j, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, a, d, i, k, l, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, a, d, i, l, j, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, a, d, i, l, k, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, a, d, j, i, k, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, a, d, j, i, l, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, a, d, j, k, i, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, a, d, j, k, l, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, a, d, j, l, i, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, a, d, j, l, k, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, a, d, k, i, j, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, a, d, k, i, l, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, a, d, k, j, i, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, a, d, k, j, l, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, a, d, k, l, i, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, a, d, k, l, j, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, a, d, l, i, j, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, a, d, l, i, k, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, a, d, l, j, i, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, a, d, l, j, k, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, a, d, l, k, i, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, a, d, l, k, j, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, d, a, i, j, k, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, d, a, i, j, l, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, d, a, i, k, j, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, d, a, i, k, l, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, d, a, i, l, j, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, d, a, i, l, k, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, d, a, j, i, k, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, d, a, j, i, l, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, d, a, j, k, i, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, d, a, j, k, l, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, d, a, j, l, i, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, d, a, j, l, k, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, d, a, k, i, j, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, d, a, k, i, l, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, d, a, k, j, i, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, d, a, k, j, l, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, d, a, k, l, i, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, d, a, k, l, j, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, d, a, l, i, j, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, d, a, l, i, k, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, d, a, l, j, i, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, d, a, l, j, k, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, d, a, l, k, i, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, c, d, a, l, k, j, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, a, c, i, j, k, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, a, c, i, j, l, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, a, c, i, k, j, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, a, c, i, k, l, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, a, c, i, l, j, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, a, c, i, l, k, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, a, c, j, i, k, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, a, c, j, i, l, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, a, c, j, k, i, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, a, c, j, k, l, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, a, c, j, l, i, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, a, c, j, l, k, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, a, c, k, i, j, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, a, c, k, i, l, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, a, c, k, j, i, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, a, c, k, j, l, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, a, c, k, l, i, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, a, c, k, l, j, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, a, c, l, i, j, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, a, c, l, i, k, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, a, c, l, j, i, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, a, c, l, j, k, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, a, c, l, k, i, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, a, c, l, k, j, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, c, a, i, j, k, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, c, a, i, j, l, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, c, a, i, k, j, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, c, a, i, k, l, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, c, a, i, l, j, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, c, a, i, l, k, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, c, a, j, i, k, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, c, a, j, i, l, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, c, a, j, k, i, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, c, a, j, k, l, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, c, a, j, l, i, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, c, a, j, l, k, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, c, a, k, i, j, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, c, a, k, i, l, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, c, a, k, j, i, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, c, a, k, j, l, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, c, a, k, l, i, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, c, a, k, l, j, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, c, a, l, i, j, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, c, a, l, i, k, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, c, a, l, j, i, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, c, a, l, j, k, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, c, a, l, k, i, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(b, d, c, a, l, k, j, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, b, d, i, j, k, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, b, d, i, j, l, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, b, d, i, k, j, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, b, d, i, k, l, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, b, d, i, l, j, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, b, d, i, l, k, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, b, d, j, i, k, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, b, d, j, i, l, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, b, d, j, k, i, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, b, d, j, k, l, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, b, d, j, l, i, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, b, d, j, l, k, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, b, d, k, i, j, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, b, d, k, i, l, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, b, d, k, j, i, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, b, d, k, j, l, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, b, d, k, l, i, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, b, d, k, l, j, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, b, d, l, i, j, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, b, d, l, i, k, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, b, d, l, j, i, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, b, d, l, j, k, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, b, d, l, k, i, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, b, d, l, k, j, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, d, b, i, j, k, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, d, b, i, j, l, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, d, b, i, k, j, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, d, b, i, k, l, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, d, b, i, l, j, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, d, b, i, l, k, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, d, b, j, i, k, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, d, b, j, i, l, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, d, b, j, k, i, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, d, b, j, k, l, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, d, b, j, l, i, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, d, b, j, l, k, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, d, b, k, i, j, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, d, b, k, i, l, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, d, b, k, j, i, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, d, b, k, j, l, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, d, b, k, l, i, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, d, b, k, l, j, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, d, b, l, i, j, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, d, b, l, i, k, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, d, b, l, j, i, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, d, b, l, j, k, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, d, b, l, k, i, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, a, d, b, l, k, j, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, a, d, i, j, k, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, a, d, i, j, l, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, a, d, i, k, j, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, a, d, i, k, l, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, a, d, i, l, j, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, a, d, i, l, k, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, a, d, j, i, k, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, a, d, j, i, l, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, a, d, j, k, i, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, a, d, j, k, l, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, a, d, j, l, i, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, a, d, j, l, k, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, a, d, k, i, j, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, a, d, k, i, l, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, a, d, k, j, i, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, a, d, k, j, l, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, a, d, k, l, i, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, a, d, k, l, j, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, a, d, l, i, j, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, a, d, l, i, k, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, a, d, l, j, i, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, a, d, l, j, k, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, a, d, l, k, i, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, a, d, l, k, j, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, d, a, i, j, k, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, d, a, i, j, l, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, d, a, i, k, j, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, d, a, i, k, l, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, d, a, i, l, j, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, d, a, i, l, k, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, d, a, j, i, k, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, d, a, j, i, l, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, d, a, j, k, i, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, d, a, j, k, l, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, d, a, j, l, i, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, d, a, j, l, k, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, d, a, k, i, j, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, d, a, k, i, l, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, d, a, k, j, i, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, d, a, k, j, l, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, d, a, k, l, i, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, d, a, k, l, j, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, d, a, l, i, j, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, d, a, l, i, k, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, d, a, l, j, i, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, d, a, l, j, k, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, d, a, l, k, i, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, b, d, a, l, k, j, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, a, b, i, j, k, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, a, b, i, j, l, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, a, b, i, k, j, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, a, b, i, k, l, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, a, b, i, l, j, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, a, b, i, l, k, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, a, b, j, i, k, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, a, b, j, i, l, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, a, b, j, k, i, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, a, b, j, k, l, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, a, b, j, l, i, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, a, b, j, l, k, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, a, b, k, i, j, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, a, b, k, i, l, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, a, b, k, j, i, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, a, b, k, j, l, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, a, b, k, l, i, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, a, b, k, l, j, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, a, b, l, i, j, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, a, b, l, i, k, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, a, b, l, j, i, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, a, b, l, j, k, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, a, b, l, k, i, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, a, b, l, k, j, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, b, a, i, j, k, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, b, a, i, j, l, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, b, a, i, k, j, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, b, a, i, k, l, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, b, a, i, l, j, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, b, a, i, l, k, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, b, a, j, i, k, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, b, a, j, i, l, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, b, a, j, k, i, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, b, a, j, k, l, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, b, a, j, l, i, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, b, a, j, l, k, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, b, a, k, i, j, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, b, a, k, i, l, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, b, a, k, j, i, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, b, a, k, j, l, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, b, a, k, l, i, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, b, a, k, l, j, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, b, a, l, i, j, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, b, a, l, i, k, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, b, a, l, j, i, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, b, a, l, j, k, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, b, a, l, k, i, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(c, d, b, a, l, k, j, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, b, c, i, j, k, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, b, c, i, j, l, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, b, c, i, k, j, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, b, c, i, k, l, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, b, c, i, l, j, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, b, c, i, l, k, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, b, c, j, i, k, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, b, c, j, i, l, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, b, c, j, k, i, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, b, c, j, k, l, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, b, c, j, l, i, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, b, c, j, l, k, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, b, c, k, i, j, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, b, c, k, i, l, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, b, c, k, j, i, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, b, c, k, j, l, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, b, c, k, l, i, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, b, c, k, l, j, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, b, c, l, i, j, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, b, c, l, i, k, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, b, c, l, j, i, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, b, c, l, j, k, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, b, c, l, k, i, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, b, c, l, k, j, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, c, b, i, j, k, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, c, b, i, j, l, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, c, b, i, k, j, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, c, b, i, k, l, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, c, b, i, l, j, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, c, b, i, l, k, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, c, b, j, i, k, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, c, b, j, i, l, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, c, b, j, k, i, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, c, b, j, k, l, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, c, b, j, l, i, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, c, b, j, l, k, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, c, b, k, i, j, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, c, b, k, i, l, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, c, b, k, j, i, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, c, b, k, j, l, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, c, b, k, l, i, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, c, b, k, l, j, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, c, b, l, i, j, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, c, b, l, i, k, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, c, b, l, j, i, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, c, b, l, j, k, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, c, b, l, k, i, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, a, c, b, l, k, j, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, a, c, i, j, k, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, a, c, i, j, l, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, a, c, i, k, j, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, a, c, i, k, l, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, a, c, i, l, j, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, a, c, i, l, k, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, a, c, j, i, k, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, a, c, j, i, l, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, a, c, j, k, i, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, a, c, j, k, l, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, a, c, j, l, i, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, a, c, j, l, k, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, a, c, k, i, j, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, a, c, k, i, l, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, a, c, k, j, i, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, a, c, k, j, l, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, a, c, k, l, i, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, a, c, k, l, j, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, a, c, l, i, j, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, a, c, l, i, k, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, a, c, l, j, i, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, a, c, l, j, k, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, a, c, l, k, i, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, a, c, l, k, j, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, c, a, i, j, k, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, c, a, i, j, l, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, c, a, i, k, j, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, c, a, i, k, l, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, c, a, i, l, j, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, c, a, i, l, k, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, c, a, j, i, k, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, c, a, j, i, l, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, c, a, j, k, i, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, c, a, j, k, l, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, c, a, j, l, i, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, c, a, j, l, k, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, c, a, k, i, j, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, c, a, k, i, l, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, c, a, k, j, i, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, c, a, k, j, l, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, c, a, k, l, i, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, c, a, k, l, j, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, c, a, l, i, j, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, c, a, l, i, k, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, c, a, l, j, i, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, c, a, l, j, k, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, c, a, l, k, i, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, b, c, a, l, k, j, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, a, b, i, j, k, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, a, b, i, j, l, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, a, b, i, k, j, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, a, b, i, k, l, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, a, b, i, l, j, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, a, b, i, l, k, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, a, b, j, i, k, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, a, b, j, i, l, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, a, b, j, k, i, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, a, b, j, k, l, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, a, b, j, l, i, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, a, b, j, l, k, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, a, b, k, i, j, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, a, b, k, i, l, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, a, b, k, j, i, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, a, b, k, j, l, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, a, b, k, l, i, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, a, b, k, l, j, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, a, b, l, i, j, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, a, b, l, i, k, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, a, b, l, j, i, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, a, b, l, j, k, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, a, b, l, k, i, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, a, b, l, k, j, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, b, a, i, j, k, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, b, a, i, j, l, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, b, a, i, k, j, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, b, a, i, k, l, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, b, a, i, l, j, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, b, a, i, l, k, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, b, a, j, i, k, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, b, a, j, i, l, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, b, a, j, k, i, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, b, a, j, k, l, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, b, a, j, l, i, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, b, a, j, l, k, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, b, a, k, i, j, l) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, b, a, k, i, l, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, b, a, k, j, i, l) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, b, a, k, j, l, i) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, b, a, k, l, i, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, b, a, k, l, j, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, b, a, l, i, j, k) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, b, a, l, i, k, j) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, b, a, l, j, i, k) = t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, b, a, l, j, k, i) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, b, a, l, k, i, j) = -1.0 * t4a(a, b, c, d, i, j, k, l)
                        t4a(d, c, b, a, l, k, j, i) = t4a(a, b, c, d, i, j, k, l)

                        resid(a, b, c, d, i, j, k, l) = val
                        resid(a, b, c, d, i, j, l, k) = -1.0 * val
                        resid(a, b, c, d, i, k, j, l) = -1.0 * val
                        resid(a, b, c, d, i, k, l, j) = val
                        resid(a, b, c, d, i, l, j, k) = val
                        resid(a, b, c, d, i, l, k, j) = -1.0 * val
                        resid(a, b, c, d, j, i, k, l) = -1.0 * val
                        resid(a, b, c, d, j, i, l, k) = val
                        resid(a, b, c, d, j, k, i, l) = val
                        resid(a, b, c, d, j, k, l, i) = -1.0 * val
                        resid(a, b, c, d, j, l, i, k) = -1.0 * val
                        resid(a, b, c, d, j, l, k, i) = val
                        resid(a, b, c, d, k, i, j, l) = val
                        resid(a, b, c, d, k, i, l, j) = -1.0 * val
                        resid(a, b, c, d, k, j, i, l) = -1.0 * val
                        resid(a, b, c, d, k, j, l, i) = val
                        resid(a, b, c, d, k, l, i, j) = val
                        resid(a, b, c, d, k, l, j, i) = -1.0 * val
                        resid(a, b, c, d, l, i, j, k) = -1.0 * val
                        resid(a, b, c, d, l, i, k, j) = val
                        resid(a, b, c, d, l, j, i, k) = val
                        resid(a, b, c, d, l, j, k, i) = -1.0 * val
                        resid(a, b, c, d, l, k, i, j) = -1.0 * val
                        resid(a, b, c, d, l, k, j, i) = val
                        resid(a, b, d, c, i, j, k, l) = -1.0 * val
                        resid(a, b, d, c, i, j, l, k) = val
                        resid(a, b, d, c, i, k, j, l) = val
                        resid(a, b, d, c, i, k, l, j) = -1.0 * val
                        resid(a, b, d, c, i, l, j, k) = -1.0 * val
                        resid(a, b, d, c, i, l, k, j) = val
                        resid(a, b, d, c, j, i, k, l) = val
                        resid(a, b, d, c, j, i, l, k) = -1.0 * val
                        resid(a, b, d, c, j, k, i, l) = -1.0 * val
                        resid(a, b, d, c, j, k, l, i) = val
                        resid(a, b, d, c, j, l, i, k) = val
                        resid(a, b, d, c, j, l, k, i) = -1.0 * val
                        resid(a, b, d, c, k, i, j, l) = -1.0 * val
                        resid(a, b, d, c, k, i, l, j) = val
                        resid(a, b, d, c, k, j, i, l) = val
                        resid(a, b, d, c, k, j, l, i) = -1.0 * val
                        resid(a, b, d, c, k, l, i, j) = -1.0 * val
                        resid(a, b, d, c, k, l, j, i) = val
                        resid(a, b, d, c, l, i, j, k) = val
                        resid(a, b, d, c, l, i, k, j) = -1.0 * val
                        resid(a, b, d, c, l, j, i, k) = -1.0 * val
                        resid(a, b, d, c, l, j, k, i) = val
                        resid(a, b, d, c, l, k, i, j) = val
                        resid(a, b, d, c, l, k, j, i) = -1.0 * val
                        resid(a, c, b, d, i, j, k, l) = -1.0 * val
                        resid(a, c, b, d, i, j, l, k) = val
                        resid(a, c, b, d, i, k, j, l) = val
                        resid(a, c, b, d, i, k, l, j) = -1.0 * val
                        resid(a, c, b, d, i, l, j, k) = -1.0 * val
                        resid(a, c, b, d, i, l, k, j) = val
                        resid(a, c, b, d, j, i, k, l) = val
                        resid(a, c, b, d, j, i, l, k) = -1.0 * val
                        resid(a, c, b, d, j, k, i, l) = -1.0 * val
                        resid(a, c, b, d, j, k, l, i) = val
                        resid(a, c, b, d, j, l, i, k) = val
                        resid(a, c, b, d, j, l, k, i) = -1.0 * val
                        resid(a, c, b, d, k, i, j, l) = -1.0 * val
                        resid(a, c, b, d, k, i, l, j) = val
                        resid(a, c, b, d, k, j, i, l) = val
                        resid(a, c, b, d, k, j, l, i) = -1.0 * val
                        resid(a, c, b, d, k, l, i, j) = -1.0 * val
                        resid(a, c, b, d, k, l, j, i) = val
                        resid(a, c, b, d, l, i, j, k) = val
                        resid(a, c, b, d, l, i, k, j) = -1.0 * val
                        resid(a, c, b, d, l, j, i, k) = -1.0 * val
                        resid(a, c, b, d, l, j, k, i) = val
                        resid(a, c, b, d, l, k, i, j) = val
                        resid(a, c, b, d, l, k, j, i) = -1.0 * val
                        resid(a, c, d, b, i, j, k, l) = val
                        resid(a, c, d, b, i, j, l, k) = -1.0 * val
                        resid(a, c, d, b, i, k, j, l) = -1.0 * val
                        resid(a, c, d, b, i, k, l, j) = val
                        resid(a, c, d, b, i, l, j, k) = val
                        resid(a, c, d, b, i, l, k, j) = -1.0 * val
                        resid(a, c, d, b, j, i, k, l) = -1.0 * val
                        resid(a, c, d, b, j, i, l, k) = val
                        resid(a, c, d, b, j, k, i, l) = val
                        resid(a, c, d, b, j, k, l, i) = -1.0 * val
                        resid(a, c, d, b, j, l, i, k) = -1.0 * val
                        resid(a, c, d, b, j, l, k, i) = val
                        resid(a, c, d, b, k, i, j, l) = val
                        resid(a, c, d, b, k, i, l, j) = -1.0 * val
                        resid(a, c, d, b, k, j, i, l) = -1.0 * val
                        resid(a, c, d, b, k, j, l, i) = val
                        resid(a, c, d, b, k, l, i, j) = val
                        resid(a, c, d, b, k, l, j, i) = -1.0 * val
                        resid(a, c, d, b, l, i, j, k) = -1.0 * val
                        resid(a, c, d, b, l, i, k, j) = val
                        resid(a, c, d, b, l, j, i, k) = val
                        resid(a, c, d, b, l, j, k, i) = -1.0 * val
                        resid(a, c, d, b, l, k, i, j) = -1.0 * val
                        resid(a, c, d, b, l, k, j, i) = val
                        resid(a, d, b, c, i, j, k, l) = val
                        resid(a, d, b, c, i, j, l, k) = -1.0 * val
                        resid(a, d, b, c, i, k, j, l) = -1.0 * val
                        resid(a, d, b, c, i, k, l, j) = val
                        resid(a, d, b, c, i, l, j, k) = val
                        resid(a, d, b, c, i, l, k, j) = -1.0 * val
                        resid(a, d, b, c, j, i, k, l) = -1.0 * val
                        resid(a, d, b, c, j, i, l, k) = val
                        resid(a, d, b, c, j, k, i, l) = val
                        resid(a, d, b, c, j, k, l, i) = -1.0 * val
                        resid(a, d, b, c, j, l, i, k) = -1.0 * val
                        resid(a, d, b, c, j, l, k, i) = val
                        resid(a, d, b, c, k, i, j, l) = val
                        resid(a, d, b, c, k, i, l, j) = -1.0 * val
                        resid(a, d, b, c, k, j, i, l) = -1.0 * val
                        resid(a, d, b, c, k, j, l, i) = val
                        resid(a, d, b, c, k, l, i, j) = val
                        resid(a, d, b, c, k, l, j, i) = -1.0 * val
                        resid(a, d, b, c, l, i, j, k) = -1.0 * val
                        resid(a, d, b, c, l, i, k, j) = val
                        resid(a, d, b, c, l, j, i, k) = val
                        resid(a, d, b, c, l, j, k, i) = -1.0 * val
                        resid(a, d, b, c, l, k, i, j) = -1.0 * val
                        resid(a, d, b, c, l, k, j, i) = val
                        resid(a, d, c, b, i, j, k, l) = -1.0 * val
                        resid(a, d, c, b, i, j, l, k) = val
                        resid(a, d, c, b, i, k, j, l) = val
                        resid(a, d, c, b, i, k, l, j) = -1.0 * val
                        resid(a, d, c, b, i, l, j, k) = -1.0 * val
                        resid(a, d, c, b, i, l, k, j) = val
                        resid(a, d, c, b, j, i, k, l) = val
                        resid(a, d, c, b, j, i, l, k) = -1.0 * val
                        resid(a, d, c, b, j, k, i, l) = -1.0 * val
                        resid(a, d, c, b, j, k, l, i) = val
                        resid(a, d, c, b, j, l, i, k) = val
                        resid(a, d, c, b, j, l, k, i) = -1.0 * val
                        resid(a, d, c, b, k, i, j, l) = -1.0 * val
                        resid(a, d, c, b, k, i, l, j) = val
                        resid(a, d, c, b, k, j, i, l) = val
                        resid(a, d, c, b, k, j, l, i) = -1.0 * val
                        resid(a, d, c, b, k, l, i, j) = -1.0 * val
                        resid(a, d, c, b, k, l, j, i) = val
                        resid(a, d, c, b, l, i, j, k) = val
                        resid(a, d, c, b, l, i, k, j) = -1.0 * val
                        resid(a, d, c, b, l, j, i, k) = -1.0 * val
                        resid(a, d, c, b, l, j, k, i) = val
                        resid(a, d, c, b, l, k, i, j) = val
                        resid(a, d, c, b, l, k, j, i) = -1.0 * val
                        resid(b, a, c, d, i, j, k, l) = -1.0 * val
                        resid(b, a, c, d, i, j, l, k) = val
                        resid(b, a, c, d, i, k, j, l) = val
                        resid(b, a, c, d, i, k, l, j) = -1.0 * val
                        resid(b, a, c, d, i, l, j, k) = -1.0 * val
                        resid(b, a, c, d, i, l, k, j) = val
                        resid(b, a, c, d, j, i, k, l) = val
                        resid(b, a, c, d, j, i, l, k) = -1.0 * val
                        resid(b, a, c, d, j, k, i, l) = -1.0 * val
                        resid(b, a, c, d, j, k, l, i) = val
                        resid(b, a, c, d, j, l, i, k) = val
                        resid(b, a, c, d, j, l, k, i) = -1.0 * val
                        resid(b, a, c, d, k, i, j, l) = -1.0 * val
                        resid(b, a, c, d, k, i, l, j) = val
                        resid(b, a, c, d, k, j, i, l) = val
                        resid(b, a, c, d, k, j, l, i) = -1.0 * val
                        resid(b, a, c, d, k, l, i, j) = -1.0 * val
                        resid(b, a, c, d, k, l, j, i) = val
                        resid(b, a, c, d, l, i, j, k) = val
                        resid(b, a, c, d, l, i, k, j) = -1.0 * val
                        resid(b, a, c, d, l, j, i, k) = -1.0 * val
                        resid(b, a, c, d, l, j, k, i) = val
                        resid(b, a, c, d, l, k, i, j) = val
                        resid(b, a, c, d, l, k, j, i) = -1.0 * val
                        resid(b, a, d, c, i, j, k, l) = val
                        resid(b, a, d, c, i, j, l, k) = -1.0 * val
                        resid(b, a, d, c, i, k, j, l) = -1.0 * val
                        resid(b, a, d, c, i, k, l, j) = val
                        resid(b, a, d, c, i, l, j, k) = val
                        resid(b, a, d, c, i, l, k, j) = -1.0 * val
                        resid(b, a, d, c, j, i, k, l) = -1.0 * val
                        resid(b, a, d, c, j, i, l, k) = val
                        resid(b, a, d, c, j, k, i, l) = val
                        resid(b, a, d, c, j, k, l, i) = -1.0 * val
                        resid(b, a, d, c, j, l, i, k) = -1.0 * val
                        resid(b, a, d, c, j, l, k, i) = val
                        resid(b, a, d, c, k, i, j, l) = val
                        resid(b, a, d, c, k, i, l, j) = -1.0 * val
                        resid(b, a, d, c, k, j, i, l) = -1.0 * val
                        resid(b, a, d, c, k, j, l, i) = val
                        resid(b, a, d, c, k, l, i, j) = val
                        resid(b, a, d, c, k, l, j, i) = -1.0 * val
                        resid(b, a, d, c, l, i, j, k) = -1.0 * val
                        resid(b, a, d, c, l, i, k, j) = val
                        resid(b, a, d, c, l, j, i, k) = val
                        resid(b, a, d, c, l, j, k, i) = -1.0 * val
                        resid(b, a, d, c, l, k, i, j) = -1.0 * val
                        resid(b, a, d, c, l, k, j, i) = val
                        resid(b, c, a, d, i, j, k, l) = val
                        resid(b, c, a, d, i, j, l, k) = -1.0 * val
                        resid(b, c, a, d, i, k, j, l) = -1.0 * val
                        resid(b, c, a, d, i, k, l, j) = val
                        resid(b, c, a, d, i, l, j, k) = val
                        resid(b, c, a, d, i, l, k, j) = -1.0 * val
                        resid(b, c, a, d, j, i, k, l) = -1.0 * val
                        resid(b, c, a, d, j, i, l, k) = val
                        resid(b, c, a, d, j, k, i, l) = val
                        resid(b, c, a, d, j, k, l, i) = -1.0 * val
                        resid(b, c, a, d, j, l, i, k) = -1.0 * val
                        resid(b, c, a, d, j, l, k, i) = val
                        resid(b, c, a, d, k, i, j, l) = val
                        resid(b, c, a, d, k, i, l, j) = -1.0 * val
                        resid(b, c, a, d, k, j, i, l) = -1.0 * val
                        resid(b, c, a, d, k, j, l, i) = val
                        resid(b, c, a, d, k, l, i, j) = val
                        resid(b, c, a, d, k, l, j, i) = -1.0 * val
                        resid(b, c, a, d, l, i, j, k) = -1.0 * val
                        resid(b, c, a, d, l, i, k, j) = val
                        resid(b, c, a, d, l, j, i, k) = val
                        resid(b, c, a, d, l, j, k, i) = -1.0 * val
                        resid(b, c, a, d, l, k, i, j) = -1.0 * val
                        resid(b, c, a, d, l, k, j, i) = val
                        resid(b, c, d, a, i, j, k, l) = -1.0 * val
                        resid(b, c, d, a, i, j, l, k) = val
                        resid(b, c, d, a, i, k, j, l) = val
                        resid(b, c, d, a, i, k, l, j) = -1.0 * val
                        resid(b, c, d, a, i, l, j, k) = -1.0 * val
                        resid(b, c, d, a, i, l, k, j) = val
                        resid(b, c, d, a, j, i, k, l) = val
                        resid(b, c, d, a, j, i, l, k) = -1.0 * val
                        resid(b, c, d, a, j, k, i, l) = -1.0 * val
                        resid(b, c, d, a, j, k, l, i) = val
                        resid(b, c, d, a, j, l, i, k) = val
                        resid(b, c, d, a, j, l, k, i) = -1.0 * val
                        resid(b, c, d, a, k, i, j, l) = -1.0 * val
                        resid(b, c, d, a, k, i, l, j) = val
                        resid(b, c, d, a, k, j, i, l) = val
                        resid(b, c, d, a, k, j, l, i) = -1.0 * val
                        resid(b, c, d, a, k, l, i, j) = -1.0 * val
                        resid(b, c, d, a, k, l, j, i) = val
                        resid(b, c, d, a, l, i, j, k) = val
                        resid(b, c, d, a, l, i, k, j) = -1.0 * val
                        resid(b, c, d, a, l, j, i, k) = -1.0 * val
                        resid(b, c, d, a, l, j, k, i) = val
                        resid(b, c, d, a, l, k, i, j) = val
                        resid(b, c, d, a, l, k, j, i) = -1.0 * val
                        resid(b, d, a, c, i, j, k, l) = -1.0 * val
                        resid(b, d, a, c, i, j, l, k) = val
                        resid(b, d, a, c, i, k, j, l) = val
                        resid(b, d, a, c, i, k, l, j) = -1.0 * val
                        resid(b, d, a, c, i, l, j, k) = -1.0 * val
                        resid(b, d, a, c, i, l, k, j) = val
                        resid(b, d, a, c, j, i, k, l) = val
                        resid(b, d, a, c, j, i, l, k) = -1.0 * val
                        resid(b, d, a, c, j, k, i, l) = -1.0 * val
                        resid(b, d, a, c, j, k, l, i) = val
                        resid(b, d, a, c, j, l, i, k) = val
                        resid(b, d, a, c, j, l, k, i) = -1.0 * val
                        resid(b, d, a, c, k, i, j, l) = -1.0 * val
                        resid(b, d, a, c, k, i, l, j) = val
                        resid(b, d, a, c, k, j, i, l) = val
                        resid(b, d, a, c, k, j, l, i) = -1.0 * val
                        resid(b, d, a, c, k, l, i, j) = -1.0 * val
                        resid(b, d, a, c, k, l, j, i) = val
                        resid(b, d, a, c, l, i, j, k) = val
                        resid(b, d, a, c, l, i, k, j) = -1.0 * val
                        resid(b, d, a, c, l, j, i, k) = -1.0 * val
                        resid(b, d, a, c, l, j, k, i) = val
                        resid(b, d, a, c, l, k, i, j) = val
                        resid(b, d, a, c, l, k, j, i) = -1.0 * val
                        resid(b, d, c, a, i, j, k, l) = val
                        resid(b, d, c, a, i, j, l, k) = -1.0 * val
                        resid(b, d, c, a, i, k, j, l) = -1.0 * val
                        resid(b, d, c, a, i, k, l, j) = val
                        resid(b, d, c, a, i, l, j, k) = val
                        resid(b, d, c, a, i, l, k, j) = -1.0 * val
                        resid(b, d, c, a, j, i, k, l) = -1.0 * val
                        resid(b, d, c, a, j, i, l, k) = val
                        resid(b, d, c, a, j, k, i, l) = val
                        resid(b, d, c, a, j, k, l, i) = -1.0 * val
                        resid(b, d, c, a, j, l, i, k) = -1.0 * val
                        resid(b, d, c, a, j, l, k, i) = val
                        resid(b, d, c, a, k, i, j, l) = val
                        resid(b, d, c, a, k, i, l, j) = -1.0 * val
                        resid(b, d, c, a, k, j, i, l) = -1.0 * val
                        resid(b, d, c, a, k, j, l, i) = val
                        resid(b, d, c, a, k, l, i, j) = val
                        resid(b, d, c, a, k, l, j, i) = -1.0 * val
                        resid(b, d, c, a, l, i, j, k) = -1.0 * val
                        resid(b, d, c, a, l, i, k, j) = val
                        resid(b, d, c, a, l, j, i, k) = val
                        resid(b, d, c, a, l, j, k, i) = -1.0 * val
                        resid(b, d, c, a, l, k, i, j) = -1.0 * val
                        resid(b, d, c, a, l, k, j, i) = val
                        resid(c, a, b, d, i, j, k, l) = val
                        resid(c, a, b, d, i, j, l, k) = -1.0 * val
                        resid(c, a, b, d, i, k, j, l) = -1.0 * val
                        resid(c, a, b, d, i, k, l, j) = val
                        resid(c, a, b, d, i, l, j, k) = val
                        resid(c, a, b, d, i, l, k, j) = -1.0 * val
                        resid(c, a, b, d, j, i, k, l) = -1.0 * val
                        resid(c, a, b, d, j, i, l, k) = val
                        resid(c, a, b, d, j, k, i, l) = val
                        resid(c, a, b, d, j, k, l, i) = -1.0 * val
                        resid(c, a, b, d, j, l, i, k) = -1.0 * val
                        resid(c, a, b, d, j, l, k, i) = val
                        resid(c, a, b, d, k, i, j, l) = val
                        resid(c, a, b, d, k, i, l, j) = -1.0 * val
                        resid(c, a, b, d, k, j, i, l) = -1.0 * val
                        resid(c, a, b, d, k, j, l, i) = val
                        resid(c, a, b, d, k, l, i, j) = val
                        resid(c, a, b, d, k, l, j, i) = -1.0 * val
                        resid(c, a, b, d, l, i, j, k) = -1.0 * val
                        resid(c, a, b, d, l, i, k, j) = val
                        resid(c, a, b, d, l, j, i, k) = val
                        resid(c, a, b, d, l, j, k, i) = -1.0 * val
                        resid(c, a, b, d, l, k, i, j) = -1.0 * val
                        resid(c, a, b, d, l, k, j, i) = val
                        resid(c, a, d, b, i, j, k, l) = -1.0 * val
                        resid(c, a, d, b, i, j, l, k) = val
                        resid(c, a, d, b, i, k, j, l) = val
                        resid(c, a, d, b, i, k, l, j) = -1.0 * val
                        resid(c, a, d, b, i, l, j, k) = -1.0 * val
                        resid(c, a, d, b, i, l, k, j) = val
                        resid(c, a, d, b, j, i, k, l) = val
                        resid(c, a, d, b, j, i, l, k) = -1.0 * val
                        resid(c, a, d, b, j, k, i, l) = -1.0 * val
                        resid(c, a, d, b, j, k, l, i) = val
                        resid(c, a, d, b, j, l, i, k) = val
                        resid(c, a, d, b, j, l, k, i) = -1.0 * val
                        resid(c, a, d, b, k, i, j, l) = -1.0 * val
                        resid(c, a, d, b, k, i, l, j) = val
                        resid(c, a, d, b, k, j, i, l) = val
                        resid(c, a, d, b, k, j, l, i) = -1.0 * val
                        resid(c, a, d, b, k, l, i, j) = -1.0 * val
                        resid(c, a, d, b, k, l, j, i) = val
                        resid(c, a, d, b, l, i, j, k) = val
                        resid(c, a, d, b, l, i, k, j) = -1.0 * val
                        resid(c, a, d, b, l, j, i, k) = -1.0 * val
                        resid(c, a, d, b, l, j, k, i) = val
                        resid(c, a, d, b, l, k, i, j) = val
                        resid(c, a, d, b, l, k, j, i) = -1.0 * val
                        resid(c, b, a, d, i, j, k, l) = -1.0 * val
                        resid(c, b, a, d, i, j, l, k) = val
                        resid(c, b, a, d, i, k, j, l) = val
                        resid(c, b, a, d, i, k, l, j) = -1.0 * val
                        resid(c, b, a, d, i, l, j, k) = -1.0 * val
                        resid(c, b, a, d, i, l, k, j) = val
                        resid(c, b, a, d, j, i, k, l) = val
                        resid(c, b, a, d, j, i, l, k) = -1.0 * val
                        resid(c, b, a, d, j, k, i, l) = -1.0 * val
                        resid(c, b, a, d, j, k, l, i) = val
                        resid(c, b, a, d, j, l, i, k) = val
                        resid(c, b, a, d, j, l, k, i) = -1.0 * val
                        resid(c, b, a, d, k, i, j, l) = -1.0 * val
                        resid(c, b, a, d, k, i, l, j) = val
                        resid(c, b, a, d, k, j, i, l) = val
                        resid(c, b, a, d, k, j, l, i) = -1.0 * val
                        resid(c, b, a, d, k, l, i, j) = -1.0 * val
                        resid(c, b, a, d, k, l, j, i) = val
                        resid(c, b, a, d, l, i, j, k) = val
                        resid(c, b, a, d, l, i, k, j) = -1.0 * val
                        resid(c, b, a, d, l, j, i, k) = -1.0 * val
                        resid(c, b, a, d, l, j, k, i) = val
                        resid(c, b, a, d, l, k, i, j) = val
                        resid(c, b, a, d, l, k, j, i) = -1.0 * val
                        resid(c, b, d, a, i, j, k, l) = val
                        resid(c, b, d, a, i, j, l, k) = -1.0 * val
                        resid(c, b, d, a, i, k, j, l) = -1.0 * val
                        resid(c, b, d, a, i, k, l, j) = val
                        resid(c, b, d, a, i, l, j, k) = val
                        resid(c, b, d, a, i, l, k, j) = -1.0 * val
                        resid(c, b, d, a, j, i, k, l) = -1.0 * val
                        resid(c, b, d, a, j, i, l, k) = val
                        resid(c, b, d, a, j, k, i, l) = val
                        resid(c, b, d, a, j, k, l, i) = -1.0 * val
                        resid(c, b, d, a, j, l, i, k) = -1.0 * val
                        resid(c, b, d, a, j, l, k, i) = val
                        resid(c, b, d, a, k, i, j, l) = val
                        resid(c, b, d, a, k, i, l, j) = -1.0 * val
                        resid(c, b, d, a, k, j, i, l) = -1.0 * val
                        resid(c, b, d, a, k, j, l, i) = val
                        resid(c, b, d, a, k, l, i, j) = val
                        resid(c, b, d, a, k, l, j, i) = -1.0 * val
                        resid(c, b, d, a, l, i, j, k) = -1.0 * val
                        resid(c, b, d, a, l, i, k, j) = val
                        resid(c, b, d, a, l, j, i, k) = val
                        resid(c, b, d, a, l, j, k, i) = -1.0 * val
                        resid(c, b, d, a, l, k, i, j) = -1.0 * val
                        resid(c, b, d, a, l, k, j, i) = val
                        resid(c, d, a, b, i, j, k, l) = val
                        resid(c, d, a, b, i, j, l, k) = -1.0 * val
                        resid(c, d, a, b, i, k, j, l) = -1.0 * val
                        resid(c, d, a, b, i, k, l, j) = val
                        resid(c, d, a, b, i, l, j, k) = val
                        resid(c, d, a, b, i, l, k, j) = -1.0 * val
                        resid(c, d, a, b, j, i, k, l) = -1.0 * val
                        resid(c, d, a, b, j, i, l, k) = val
                        resid(c, d, a, b, j, k, i, l) = val
                        resid(c, d, a, b, j, k, l, i) = -1.0 * val
                        resid(c, d, a, b, j, l, i, k) = -1.0 * val
                        resid(c, d, a, b, j, l, k, i) = val
                        resid(c, d, a, b, k, i, j, l) = val
                        resid(c, d, a, b, k, i, l, j) = -1.0 * val
                        resid(c, d, a, b, k, j, i, l) = -1.0 * val
                        resid(c, d, a, b, k, j, l, i) = val
                        resid(c, d, a, b, k, l, i, j) = val
                        resid(c, d, a, b, k, l, j, i) = -1.0 * val
                        resid(c, d, a, b, l, i, j, k) = -1.0 * val
                        resid(c, d, a, b, l, i, k, j) = val
                        resid(c, d, a, b, l, j, i, k) = val
                        resid(c, d, a, b, l, j, k, i) = -1.0 * val
                        resid(c, d, a, b, l, k, i, j) = -1.0 * val
                        resid(c, d, a, b, l, k, j, i) = val
                        resid(c, d, b, a, i, j, k, l) = -1.0 * val
                        resid(c, d, b, a, i, j, l, k) = val
                        resid(c, d, b, a, i, k, j, l) = val
                        resid(c, d, b, a, i, k, l, j) = -1.0 * val
                        resid(c, d, b, a, i, l, j, k) = -1.0 * val
                        resid(c, d, b, a, i, l, k, j) = val
                        resid(c, d, b, a, j, i, k, l) = val
                        resid(c, d, b, a, j, i, l, k) = -1.0 * val
                        resid(c, d, b, a, j, k, i, l) = -1.0 * val
                        resid(c, d, b, a, j, k, l, i) = val
                        resid(c, d, b, a, j, l, i, k) = val
                        resid(c, d, b, a, j, l, k, i) = -1.0 * val
                        resid(c, d, b, a, k, i, j, l) = -1.0 * val
                        resid(c, d, b, a, k, i, l, j) = val
                        resid(c, d, b, a, k, j, i, l) = val
                        resid(c, d, b, a, k, j, l, i) = -1.0 * val
                        resid(c, d, b, a, k, l, i, j) = -1.0 * val
                        resid(c, d, b, a, k, l, j, i) = val
                        resid(c, d, b, a, l, i, j, k) = val
                        resid(c, d, b, a, l, i, k, j) = -1.0 * val
                        resid(c, d, b, a, l, j, i, k) = -1.0 * val
                        resid(c, d, b, a, l, j, k, i) = val
                        resid(c, d, b, a, l, k, i, j) = val
                        resid(c, d, b, a, l, k, j, i) = -1.0 * val
                        resid(d, a, b, c, i, j, k, l) = -1.0 * val
                        resid(d, a, b, c, i, j, l, k) = val
                        resid(d, a, b, c, i, k, j, l) = val
                        resid(d, a, b, c, i, k, l, j) = -1.0 * val
                        resid(d, a, b, c, i, l, j, k) = -1.0 * val
                        resid(d, a, b, c, i, l, k, j) = val
                        resid(d, a, b, c, j, i, k, l) = val
                        resid(d, a, b, c, j, i, l, k) = -1.0 * val
                        resid(d, a, b, c, j, k, i, l) = -1.0 * val
                        resid(d, a, b, c, j, k, l, i) = val
                        resid(d, a, b, c, j, l, i, k) = val
                        resid(d, a, b, c, j, l, k, i) = -1.0 * val
                        resid(d, a, b, c, k, i, j, l) = -1.0 * val
                        resid(d, a, b, c, k, i, l, j) = val
                        resid(d, a, b, c, k, j, i, l) = val
                        resid(d, a, b, c, k, j, l, i) = -1.0 * val
                        resid(d, a, b, c, k, l, i, j) = -1.0 * val
                        resid(d, a, b, c, k, l, j, i) = val
                        resid(d, a, b, c, l, i, j, k) = val
                        resid(d, a, b, c, l, i, k, j) = -1.0 * val
                        resid(d, a, b, c, l, j, i, k) = -1.0 * val
                        resid(d, a, b, c, l, j, k, i) = val
                        resid(d, a, b, c, l, k, i, j) = val
                        resid(d, a, b, c, l, k, j, i) = -1.0 * val
                        resid(d, a, c, b, i, j, k, l) = val
                        resid(d, a, c, b, i, j, l, k) = -1.0 * val
                        resid(d, a, c, b, i, k, j, l) = -1.0 * val
                        resid(d, a, c, b, i, k, l, j) = val
                        resid(d, a, c, b, i, l, j, k) = val
                        resid(d, a, c, b, i, l, k, j) = -1.0 * val
                        resid(d, a, c, b, j, i, k, l) = -1.0 * val
                        resid(d, a, c, b, j, i, l, k) = val
                        resid(d, a, c, b, j, k, i, l) = val
                        resid(d, a, c, b, j, k, l, i) = -1.0 * val
                        resid(d, a, c, b, j, l, i, k) = -1.0 * val
                        resid(d, a, c, b, j, l, k, i) = val
                        resid(d, a, c, b, k, i, j, l) = val
                        resid(d, a, c, b, k, i, l, j) = -1.0 * val
                        resid(d, a, c, b, k, j, i, l) = -1.0 * val
                        resid(d, a, c, b, k, j, l, i) = val
                        resid(d, a, c, b, k, l, i, j) = val
                        resid(d, a, c, b, k, l, j, i) = -1.0 * val
                        resid(d, a, c, b, l, i, j, k) = -1.0 * val
                        resid(d, a, c, b, l, i, k, j) = val
                        resid(d, a, c, b, l, j, i, k) = val
                        resid(d, a, c, b, l, j, k, i) = -1.0 * val
                        resid(d, a, c, b, l, k, i, j) = -1.0 * val
                        resid(d, a, c, b, l, k, j, i) = val
                        resid(d, b, a, c, i, j, k, l) = val
                        resid(d, b, a, c, i, j, l, k) = -1.0 * val
                        resid(d, b, a, c, i, k, j, l) = -1.0 * val
                        resid(d, b, a, c, i, k, l, j) = val
                        resid(d, b, a, c, i, l, j, k) = val
                        resid(d, b, a, c, i, l, k, j) = -1.0 * val
                        resid(d, b, a, c, j, i, k, l) = -1.0 * val
                        resid(d, b, a, c, j, i, l, k) = val
                        resid(d, b, a, c, j, k, i, l) = val
                        resid(d, b, a, c, j, k, l, i) = -1.0 * val
                        resid(d, b, a, c, j, l, i, k) = -1.0 * val
                        resid(d, b, a, c, j, l, k, i) = val
                        resid(d, b, a, c, k, i, j, l) = val
                        resid(d, b, a, c, k, i, l, j) = -1.0 * val
                        resid(d, b, a, c, k, j, i, l) = -1.0 * val
                        resid(d, b, a, c, k, j, l, i) = val
                        resid(d, b, a, c, k, l, i, j) = val
                        resid(d, b, a, c, k, l, j, i) = -1.0 * val
                        resid(d, b, a, c, l, i, j, k) = -1.0 * val
                        resid(d, b, a, c, l, i, k, j) = val
                        resid(d, b, a, c, l, j, i, k) = val
                        resid(d, b, a, c, l, j, k, i) = -1.0 * val
                        resid(d, b, a, c, l, k, i, j) = -1.0 * val
                        resid(d, b, a, c, l, k, j, i) = val
                        resid(d, b, c, a, i, j, k, l) = -1.0 * val
                        resid(d, b, c, a, i, j, l, k) = val
                        resid(d, b, c, a, i, k, j, l) = val
                        resid(d, b, c, a, i, k, l, j) = -1.0 * val
                        resid(d, b, c, a, i, l, j, k) = -1.0 * val
                        resid(d, b, c, a, i, l, k, j) = val
                        resid(d, b, c, a, j, i, k, l) = val
                        resid(d, b, c, a, j, i, l, k) = -1.0 * val
                        resid(d, b, c, a, j, k, i, l) = -1.0 * val
                        resid(d, b, c, a, j, k, l, i) = val
                        resid(d, b, c, a, j, l, i, k) = val
                        resid(d, b, c, a, j, l, k, i) = -1.0 * val
                        resid(d, b, c, a, k, i, j, l) = -1.0 * val
                        resid(d, b, c, a, k, i, l, j) = val
                        resid(d, b, c, a, k, j, i, l) = val
                        resid(d, b, c, a, k, j, l, i) = -1.0 * val
                        resid(d, b, c, a, k, l, i, j) = -1.0 * val
                        resid(d, b, c, a, k, l, j, i) = val
                        resid(d, b, c, a, l, i, j, k) = val
                        resid(d, b, c, a, l, i, k, j) = -1.0 * val
                        resid(d, b, c, a, l, j, i, k) = -1.0 * val
                        resid(d, b, c, a, l, j, k, i) = val
                        resid(d, b, c, a, l, k, i, j) = val
                        resid(d, b, c, a, l, k, j, i) = -1.0 * val
                        resid(d, c, a, b, i, j, k, l) = -1.0 * val
                        resid(d, c, a, b, i, j, l, k) = val
                        resid(d, c, a, b, i, k, j, l) = val
                        resid(d, c, a, b, i, k, l, j) = -1.0 * val
                        resid(d, c, a, b, i, l, j, k) = -1.0 * val
                        resid(d, c, a, b, i, l, k, j) = val
                        resid(d, c, a, b, j, i, k, l) = val
                        resid(d, c, a, b, j, i, l, k) = -1.0 * val
                        resid(d, c, a, b, j, k, i, l) = -1.0 * val
                        resid(d, c, a, b, j, k, l, i) = val
                        resid(d, c, a, b, j, l, i, k) = val
                        resid(d, c, a, b, j, l, k, i) = -1.0 * val
                        resid(d, c, a, b, k, i, j, l) = -1.0 * val
                        resid(d, c, a, b, k, i, l, j) = val
                        resid(d, c, a, b, k, j, i, l) = val
                        resid(d, c, a, b, k, j, l, i) = -1.0 * val
                        resid(d, c, a, b, k, l, i, j) = -1.0 * val
                        resid(d, c, a, b, k, l, j, i) = val
                        resid(d, c, a, b, l, i, j, k) = val
                        resid(d, c, a, b, l, i, k, j) = -1.0 * val
                        resid(d, c, a, b, l, j, i, k) = -1.0 * val
                        resid(d, c, a, b, l, j, k, i) = val
                        resid(d, c, a, b, l, k, i, j) = val
                        resid(d, c, a, b, l, k, j, i) = -1.0 * val
                        resid(d, c, b, a, i, j, k, l) = val
                        resid(d, c, b, a, i, j, l, k) = -1.0 * val
                        resid(d, c, b, a, i, k, j, l) = -1.0 * val
                        resid(d, c, b, a, i, k, l, j) = val
                        resid(d, c, b, a, i, l, j, k) = val
                        resid(d, c, b, a, i, l, k, j) = -1.0 * val
                        resid(d, c, b, a, j, i, k, l) = -1.0 * val
                        resid(d, c, b, a, j, i, l, k) = val
                        resid(d, c, b, a, j, k, i, l) = val
                        resid(d, c, b, a, j, k, l, i) = -1.0 * val
                        resid(d, c, b, a, j, l, i, k) = -1.0 * val
                        resid(d, c, b, a, j, l, k, i) = val
                        resid(d, c, b, a, k, i, j, l) = val
                        resid(d, c, b, a, k, i, l, j) = -1.0 * val
                        resid(d, c, b, a, k, j, i, l) = -1.0 * val
                        resid(d, c, b, a, k, j, l, i) = val
                        resid(d, c, b, a, k, l, i, j) = val
                        resid(d, c, b, a, k, l, j, i) = -1.0 * val
                        resid(d, c, b, a, l, i, j, k) = -1.0 * val
                        resid(d, c, b, a, l, i, k, j) = val
                        resid(d, c, b, a, l, j, i, k) = val
                        resid(d, c, b, a, l, j, k, i) = -1.0 * val
                        resid(d, c, b, a, l, k, i, j) = -1.0 * val
                        resid(d, c, b, a, l, k, j, i) = val

                        end do
                     end do
                  end do
               end do
            end do
         end do
      end do
   end do

end subroutine update_t4a

subroutine update_t4b(t4b, resid, X4B, &
                             fA_oo, fA_vv, fB_oo, fB_vv, &
                             shift, &
                             noa, nua, nob, nub)

      integer, intent(in)  :: noa, nob, nua, nub
      real(8), intent(in)  :: fA_oo(1:noa, 1:noa), &
                              fB_oo(1:nob, 1:nob), &
                              fA_vv(1:nua, 1:nua), &
                              fB_vv(1:nub, 1:nub)
      real(8), intent(in)  :: X4B(1:nua, 1:nua, 1:nua, 1:nub, 1:noa, 1:noa, 1:noa, 1:nob)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t4b(1:nua, 1:nua, 1:nua, 1:nub, 1:noa, 1:noa, 1:noa, 1:nob)
      !f2py intent(in, out)  :: t4b(0:nua-1, 0:nua-1, 0:nua-1, 0:nub-1, 0:noa-1, 0:noa-1, 0:noa-1, 0:nob-1)
      real(8), intent(out)   :: resid(1:nua, 1:nua, 1:nua, 1:nub, 1:noa, 1:noa, 1:noa, 1:nob)

      integer :: i, j, k, l, a, b, c, d
      real(8) :: denom, val

   do i = 1 , noa
      do j = i + 1 , noa
         do k = j + 1 , noa
            do l = 1 , nob
               do a = 1 , nua
                  do b = a + 1 , nua
                     do c = b + 1 , nua
                        do d = 1 , nub

                        denom = fA_oo(i,i) + fA_oo(j,j) + fA_oo(k,k) + fB_oo(l,l)&
                               -fA_vv(a,a) - fA_vv(b,b) - fA_vv(c,c) - fB_vv(d,d)

                        val = &
                        +X4B(a, b, c, d, i, j, k, l)&
                        -1.0 * X4B(a, b, c, d, i, k, j, l)&
                        -1.0 * X4B(a, b, c, d, j, i, k, l)&
                        +X4B(a, b, c, d, j, k, i, l)&
                        +X4B(a, b, c, d, k, i, j, l)&
                        -1.0 * X4B(a, b, c, d, k, j, i, l)&
                        -1.0 * X4B(a, c, b, d, i, j, k, l)&
                        +X4B(a, c, b, d, i, k, j, l)&
                        +X4B(a, c, b, d, j, i, k, l)&
                        -1.0 * X4B(a, c, b, d, j, k, i, l)&
                        -1.0 * X4B(a, c, b, d, k, i, j, l)&
                        +X4B(a, c, b, d, k, j, i, l)&
                        -1.0 * X4B(b, a, c, d, i, j, k, l)&
                        +X4B(b, a, c, d, i, k, j, l)&
                        +X4B(b, a, c, d, j, i, k, l)&
                        -1.0 * X4B(b, a, c, d, j, k, i, l)&
                        -1.0 * X4B(b, a, c, d, k, i, j, l)&
                        +X4B(b, a, c, d, k, j, i, l)&
                        +X4B(b, c, a, d, i, j, k, l)&
                        -1.0 * X4B(b, c, a, d, i, k, j, l)&
                        -1.0 * X4B(b, c, a, d, j, i, k, l)&
                        +X4B(b, c, a, d, j, k, i, l)&
                        +X4B(b, c, a, d, k, i, j, l)&
                        -1.0 * X4B(b, c, a, d, k, j, i, l)&
                        +X4B(c, a, b, d, i, j, k, l)&
                        -1.0 * X4B(c, a, b, d, i, k, j, l)&
                        -1.0 * X4B(c, a, b, d, j, i, k, l)&
                        +X4B(c, a, b, d, j, k, i, l)&
                        +X4B(c, a, b, d, k, i, j, l)&
                        -1.0 * X4B(c, a, b, d, k, j, i, l)&
                        -1.0 * X4B(c, b, a, d, i, j, k, l)&
                        +X4B(c, b, a, d, i, k, j, l)&
                        +X4B(c, b, a, d, j, i, k, l)&
                        -1.0 * X4B(c, b, a, d, j, k, i, l)&
                        -1.0 * X4B(c, b, a, d, k, i, j, l)&
                        +X4B(c, b, a, d, k, j, i, l)

                        t4b(a, b, c, d, i, j, k, l) = t4b(a, b, c, d, i, j, k, l) + val/(denom-shift)
                        t4b(a, b, c, d, i, k, j, l) = -1.0 * t4b(a, b, c, d, i, j, k, l)
                        t4b(a, b, c, d, j, i, k, l) = -1.0 * t4b(a, b, c, d, i, j, k, l)
                        t4b(a, b, c, d, j, k, i, l) = t4b(a, b, c, d, i, j, k, l)
                        t4b(a, b, c, d, k, i, j, l) = t4b(a, b, c, d, i, j, k, l)
                        t4b(a, b, c, d, k, j, i, l) = -1.0 * t4b(a, b, c, d, i, j, k, l)
                        t4b(a, c, b, d, i, j, k, l) = -1.0 * t4b(a, b, c, d, i, j, k, l)
                        t4b(a, c, b, d, i, k, j, l) = t4b(a, b, c, d, i, j, k, l)
                        t4b(a, c, b, d, j, i, k, l) = t4b(a, b, c, d, i, j, k, l)
                        t4b(a, c, b, d, j, k, i, l) = -1.0 * t4b(a, b, c, d, i, j, k, l)
                        t4b(a, c, b, d, k, i, j, l) = -1.0 * t4b(a, b, c, d, i, j, k, l)
                        t4b(a, c, b, d, k, j, i, l) = t4b(a, b, c, d, i, j, k, l)
                        t4b(b, a, c, d, i, j, k, l) = -1.0 * t4b(a, b, c, d, i, j, k, l)
                        t4b(b, a, c, d, i, k, j, l) = t4b(a, b, c, d, i, j, k, l)
                        t4b(b, a, c, d, j, i, k, l) = t4b(a, b, c, d, i, j, k, l)
                        t4b(b, a, c, d, j, k, i, l) = -1.0 * t4b(a, b, c, d, i, j, k, l)
                        t4b(b, a, c, d, k, i, j, l) = -1.0 * t4b(a, b, c, d, i, j, k, l)
                        t4b(b, a, c, d, k, j, i, l) = t4b(a, b, c, d, i, j, k, l)
                        t4b(b, c, a, d, i, j, k, l) = t4b(a, b, c, d, i, j, k, l)
                        t4b(b, c, a, d, i, k, j, l) = -1.0 * t4b(a, b, c, d, i, j, k, l)
                        t4b(b, c, a, d, j, i, k, l) = -1.0 * t4b(a, b, c, d, i, j, k, l)
                        t4b(b, c, a, d, j, k, i, l) = t4b(a, b, c, d, i, j, k, l)
                        t4b(b, c, a, d, k, i, j, l) = t4b(a, b, c, d, i, j, k, l)
                        t4b(b, c, a, d, k, j, i, l) = -1.0 * t4b(a, b, c, d, i, j, k, l)
                        t4b(c, a, b, d, i, j, k, l) = t4b(a, b, c, d, i, j, k, l)
                        t4b(c, a, b, d, i, k, j, l) = -1.0 * t4b(a, b, c, d, i, j, k, l)
                        t4b(c, a, b, d, j, i, k, l) = -1.0 * t4b(a, b, c, d, i, j, k, l)
                        t4b(c, a, b, d, j, k, i, l) = t4b(a, b, c, d, i, j, k, l)
                        t4b(c, a, b, d, k, i, j, l) = t4b(a, b, c, d, i, j, k, l)
                        t4b(c, a, b, d, k, j, i, l) = -1.0 * t4b(a, b, c, d, i, j, k, l)
                        t4b(c, b, a, d, i, j, k, l) = -1.0 * t4b(a, b, c, d, i, j, k, l)
                        t4b(c, b, a, d, i, k, j, l) = t4b(a, b, c, d, i, j, k, l)
                        t4b(c, b, a, d, j, i, k, l) = t4b(a, b, c, d, i, j, k, l)
                        t4b(c, b, a, d, j, k, i, l) = -1.0 * t4b(a, b, c, d, i, j, k, l)
                        t4b(c, b, a, d, k, i, j, l) = -1.0 * t4b(a, b, c, d, i, j, k, l)
                        t4b(c, b, a, d, k, j, i, l) = t4b(a, b, c, d, i, j, k, l)

                        resid(a, b, c, d, i, j, k, l) = val
                        resid(a, b, c, d, i, k, j, l) = -1.0 * val
                        resid(a, b, c, d, j, i, k, l) = -1.0 * val
                        resid(a, b, c, d, j, k, i, l) = val
                        resid(a, b, c, d, k, i, j, l) = val
                        resid(a, b, c, d, k, j, i, l) = -1.0 * val
                        resid(a, c, b, d, i, j, k, l) = -1.0 * val
                        resid(a, c, b, d, i, k, j, l) = val
                        resid(a, c, b, d, j, i, k, l) = val
                        resid(a, c, b, d, j, k, i, l) = -1.0 * val
                        resid(a, c, b, d, k, i, j, l) = -1.0 * val
                        resid(a, c, b, d, k, j, i, l) = val
                        resid(b, a, c, d, i, j, k, l) = -1.0 * val
                        resid(b, a, c, d, i, k, j, l) = val
                        resid(b, a, c, d, j, i, k, l) = val
                        resid(b, a, c, d, j, k, i, l) = -1.0 * val
                        resid(b, a, c, d, k, i, j, l) = -1.0 * val
                        resid(b, a, c, d, k, j, i, l) = val
                        resid(b, c, a, d, i, j, k, l) = val
                        resid(b, c, a, d, i, k, j, l) = -1.0 * val
                        resid(b, c, a, d, j, i, k, l) = -1.0 * val
                        resid(b, c, a, d, j, k, i, l) = val
                        resid(b, c, a, d, k, i, j, l) = val
                        resid(b, c, a, d, k, j, i, l) = -1.0 * val
                        resid(c, a, b, d, i, j, k, l) = val
                        resid(c, a, b, d, i, k, j, l) = -1.0 * val
                        resid(c, a, b, d, j, i, k, l) = -1.0 * val
                        resid(c, a, b, d, j, k, i, l) = val
                        resid(c, a, b, d, k, i, j, l) = val
                        resid(c, a, b, d, k, j, i, l) = -1.0 * val
                        resid(c, b, a, d, i, j, k, l) = -1.0 * val
                        resid(c, b, a, d, i, k, j, l) = val
                        resid(c, b, a, d, j, i, k, l) = val
                        resid(c, b, a, d, j, k, i, l) = -1.0 * val
                        resid(c, b, a, d, k, i, j, l) = -1.0 * val
                        resid(c, b, a, d, k, j, i, l) = val

                        end do
                     end do
                  end do
               end do
            end do
         end do
      end do
   end do

end subroutine update_t4b


subroutine update_t4c(t4c, resid, X4C, &
                             fA_oo, fA_vv, fB_oo, fB_vv, &
                             shift, &
                             noa, nua, nob, nub)

      integer, intent(in)  :: noa, nua, nob, nub
      real(8), intent(in)  :: fA_oo(1:noa, 1:noa), &
                              fB_oo(1:nob, 1:nob), &
                              fA_vv(1:nua, 1:nua), &
                              fB_vv(1:nub, 1:nub)
      real(8), intent(in)  :: X4C(1:nua, 1:nua, 1:nub, 1:nub, 1:noa, 1:noa, 1:nob, 1:nob)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t4c(1:nua, 1:nua, 1:nub, 1:nub, 1:noa, 1:noa, 1:nob, 1:nob)
      !f2py intent(in, out)  :: t4c(0:nua-1, 0:nua-1, 0:nub-1, 0:nub-1, 0:noa-1, 0:noa-1, 0:nob-1, 0:nob-1)
      real(8), intent(out)   :: resid(1:nua, 1:nua, 1:nub, 1:nub, 1:noa, 1:noa, 1:nob, 1:nob)

      integer :: i, j, k, l, a, b, c, d
      real(8) :: denom, val

   do i = 1 , noa
      do j = i + 1 , noa
         do k = 1 , nob
            do l = k + 1 , nob
               do a = 1 , nua
                  do b = a + 1 , nua
                     do c = 1 , nub
                        do d = c + 1 , nub

                        denom = fA_oo(i,i) + fA_oo(j,j) + fB_oo(k,k) + fB_oo(l,l)&
                               -fA_vv(a,a) - fA_vv(b,b) - fB_vv(c,c) - fB_vv(d,d)

                        val = &
                        +X4C(a, b, c, d, i, j, k, l)&
                        -1.0 * X4C(a, b, c, d, i, j, l, k)&
                        -1.0 * X4C(a, b, c, d, j, i, k, l)&
                        +X4C(a, b, c, d, j, i, l, k)&
                        -1.0 * X4C(a, b, d, c, i, j, k, l)&
                        +X4C(a, b, d, c, i, j, l, k)&
                        +X4C(a, b, d, c, j, i, k, l)&
                        -1.0 * X4C(a, b, d, c, j, i, l, k)&
                        -1.0 * X4C(b, a, c, d, i, j, k, l)&
                        +X4C(b, a, c, d, i, j, l, k)&
                        +X4C(b, a, c, d, j, i, k, l)&
                        -1.0 * X4C(b, a, c, d, j, i, l, k)&
                        +X4C(b, a, d, c, i, j, k, l)&
                        -1.0 * X4C(b, a, d, c, i, j, l, k)&
                        -1.0 * X4C(b, a, d, c, j, i, k, l)&
                        +X4C(b, a, d, c, j, i, l, k)

                        t4c(a, b, c, d, i, j, k, l) = t4c(a, b, c, d, i, j, k, l) + val/(denom-shift)
                        t4c(a, b, c, d, i, j, l, k) = -1.0 * t4c(a, b, c, d, i, j, k, l)
                        t4c(a, b, c, d, j, i, k, l) = -1.0 * t4c(a, b, c, d, i, j, k, l)
                        t4c(a, b, c, d, j, i, l, k) = +t4c(a, b, c, d, i, j, k, l)
                        t4c(a, b, d, c, i, j, k, l) = -1.0 * t4c(a, b, c, d, i, j, k, l)
                        t4c(a, b, d, c, i, j, l, k) = +t4c(a, b, c, d, i, j, k, l)
                        t4c(a, b, d, c, j, i, k, l) = +t4c(a, b, c, d, i, j, k, l)
                        t4c(a, b, d, c, j, i, l, k) = -1.0 * t4c(a, b, c, d, i, j, k, l)
                        t4c(b, a, c, d, i, j, k, l) = -1.0 * t4c(a, b, c, d, i, j, k, l)
                        t4c(b, a, c, d, i, j, l, k) = +t4c(a, b, c, d, i, j, k, l)
                        t4c(b, a, c, d, j, i, k, l) = +t4c(a, b, c, d, i, j, k, l)
                        t4c(b, a, c, d, j, i, l, k) = -1.0 * t4c(a, b, c, d, i, j, k, l)
                        t4c(b, a, d, c, i, j, k, l) = +t4c(a, b, c, d, i, j, k, l)
                        t4c(b, a, d, c, i, j, l, k) = -1.0 * t4c(a, b, c, d, i, j, k, l)
                        t4c(b, a, d, c, j, i, k, l) = -1.0 * t4c(a, b, c, d, i, j, k, l)
                        t4c(b, a, d, c, j, i, l, k) = +t4c(a, b, c, d, i, j, k, l)

                        resid(a, b, c, d, i, j, k, l) = val
                        resid(a, b, c, d, i, j, l, k) = -1.0 * val
                        resid(a, b, c, d, j, i, k, l) = -1.0 * val
                        resid(a, b, c, d, j, i, l, k) = val
                        resid(a, b, d, c, i, j, k, l) = -1.0 * val
                        resid(a, b, d, c, i, j, l, k) = val
                        resid(a, b, d, c, j, i, k, l) = val
                        resid(a, b, d, c, j, i, l, k) = -1.0 * val
                        resid(b, a, c, d, i, j, k, l) = -1.0 * val
                        resid(b, a, c, d, i, j, l, k) = val
                        resid(b, a, c, d, j, i, k, l) = val
                        resid(b, a, c, d, j, i, l, k) = -1.0 * val
                        resid(b, a, d, c, i, j, k, l) = val
                        resid(b, a, d, c, i, j, l, k) = -1.0 * val
                        resid(b, a, d, c, j, i, k, l) = -1.0 * val
                        resid(b, a, d, c, j, i, l, k) = val

                        end do
                     end do
                  end do
               end do
            end do
         end do
      end do
   end do

end subroutine update_t4c

subroutine update_t4d(t4d, resid, X4D, &
                             fA_oo, fA_vv, fB_oo, fB_vv, &
                             shift, &
                             noa, nua, nob, nub)

      integer, intent(in)  :: noa, nua, nob, nub
      real(8), intent(in)  :: fA_oo(1:noa, 1:noa), &
                              fB_oo(1:nob, 1:nob), &
                              fA_vv(1:nua, 1:nua), &
                              fB_vv(1:nub, 1:nub)
      real(8), intent(in)  :: X4D(1:nua, 1:nub, 1:nub, 1:nub, 1:noa, 1:nob, 1:nob, 1:nob)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t4d(1:nua, 1:nub, 1:nub, 1:nub, 1:noa, 1:nob, 1:nob, 1:nob)
      !f2py intent(in, out)  :: t4d(0:nua-1, 0:nub-1, 0:nub-1, 0:nub-1, 0:noa-1, 0:nob-1, 0:nob-1, 0:nob-1)
      real(8), intent(out)   :: resid(1:nua, 1:nub, 1:nub, 1:nub, 1:noa, 1:nob, 1:nob, 1:nob)

      integer :: i, j, k, l, a, b, c, d
      real(8) :: denom, val

   do i = 1 , noa
      do j = 1 , nob
         do k = j + 1 , nob
            do l = k + 1 , nob
               do a = 1 , nua
                  do b = 1 , nub
                     do c = b + 1 , nub
                        do d = c + 1 , nub

                        denom = fA_oo(i,i) + fB_oo(j,j) + fB_oo(k,k) + fB_oo(l,l)&
                               -fA_vv(a,a) - fB_vv(b,b) - fB_vv(c,c) - fB_vv(d,d)

                        val = &
                        +X4D(a, b, c, d, i, j, k, l)&
                        -1.0 * X4D(a, b, c, d, i, j, l, k)&
                        -1.0 * X4D(a, b, c, d, i, k, j, l)&
                        +X4D(a, b, c, d, i, k, l, j)&
                        +X4D(a, b, c, d, i, l, j, k)&
                        -1.0 * X4D(a, b, c, d, i, l, k, j)&
                        -1.0 * X4D(a, b, d, c, i, j, k, l)&
                        +X4D(a, b, d, c, i, j, l, k)&
                        +X4D(a, b, d, c, i, k, j, l)&
                        -1.0 * X4D(a, b, d, c, i, k, l, j)&
                        -1.0 * X4D(a, b, d, c, i, l, j, k)&
                        +X4D(a, b, d, c, i, l, k, j)&
                        -1.0 * X4D(a, c, b, d, i, j, k, l)&
                        +X4D(a, c, b, d, i, j, l, k)&
                        +X4D(a, c, b, d, i, k, j, l)&
                        -1.0 * X4D(a, c, b, d, i, k, l, j)&
                        -1.0 * X4D(a, c, b, d, i, l, j, k)&
                        +X4D(a, c, b, d, i, l, k, j)&
                        +X4D(a, c, d, b, i, j, k, l)&
                        -1.0 * X4D(a, c, d, b, i, j, l, k)&
                        -1.0 * X4D(a, c, d, b, i, k, j, l)&
                        +X4D(a, c, d, b, i, k, l, j)&
                        +X4D(a, c, d, b, i, l, j, k)&
                        -1.0 * X4D(a, c, d, b, i, l, k, j)&
                        +X4D(a, d, b, c, i, j, k, l)&
                        -1.0 * X4D(a, d, b, c, i, j, l, k)&
                        -1.0 * X4D(a, d, b, c, i, k, j, l)&
                        +X4D(a, d, b, c, i, k, l, j)&
                        +X4D(a, d, b, c, i, l, j, k)&
                        -1.0 * X4D(a, d, b, c, i, l, k, j)&
                        -1.0 * X4D(a, d, c, b, i, j, k, l)&
                        +X4D(a, d, c, b, i, j, l, k)&
                        +X4D(a, d, c, b, i, k, j, l)&
                        -1.0 * X4D(a, d, c, b, i, k, l, j)&
                        -1.0 * X4D(a, d, c, b, i, l, j, k)&
                        +X4D(a, d, c, b, i, l, k, j)

                        t4d(a, b, c, d, i, j, k, l) = t4d(a, b, c, d, i, j, k, l) + val/(denom-shift)
                        t4d(a, b, c, d, i, j, l, k) = -1.0 * t4d(a, b, c, d, i, j, k, l)
                        t4d(a, b, c, d, i, k, j, l) = -1.0 * t4d(a, b, c, d, i, j, k, l)
                        t4d(a, b, c, d, i, k, l, j) = t4d(a, b, c, d, i, j, k, l)
                        t4d(a, b, c, d, i, l, j, k) = t4d(a, b, c, d, i, j, k, l)
                        t4d(a, b, c, d, i, l, k, j) = -1.0 * t4d(a, b, c, d, i, j, k, l)
                        t4d(a, b, d, c, i, j, k, l) = -1.0 * t4d(a, b, c, d, i, j, k, l)
                        t4d(a, b, d, c, i, j, l, k) = t4d(a, b, c, d, i, j, k, l)
                        t4d(a, b, d, c, i, k, j, l) = t4d(a, b, c, d, i, j, k, l)
                        t4d(a, b, d, c, i, k, l, j) = -1.0 * t4d(a, b, c, d, i, j, k, l)
                        t4d(a, b, d, c, i, l, j, k) = -1.0 * t4d(a, b, c, d, i, j, k, l)
                        t4d(a, b, d, c, i, l, k, j) = t4d(a, b, c, d, i, j, k, l)
                        t4d(a, c, b, d, i, j, k, l) = -1.0 * t4d(a, b, c, d, i, j, k, l)
                        t4d(a, c, b, d, i, j, l, k) = t4d(a, b, c, d, i, j, k, l)
                        t4d(a, c, b, d, i, k, j, l) = t4d(a, b, c, d, i, j, k, l)
                        t4d(a, c, b, d, i, k, l, j) = -1.0 * t4d(a, b, c, d, i, j, k, l)
                        t4d(a, c, b, d, i, l, j, k) = -1.0 * t4d(a, b, c, d, i, j, k, l)
                        t4d(a, c, b, d, i, l, k, j) = t4d(a, b, c, d, i, j, k, l)
                        t4d(a, c, d, b, i, j, k, l) = t4d(a, b, c, d, i, j, k, l)
                        t4d(a, c, d, b, i, j, l, k) = -1.0 * t4d(a, b, c, d, i, j, k, l)
                        t4d(a, c, d, b, i, k, j, l) = -1.0 * t4d(a, b, c, d, i, j, k, l)
                        t4d(a, c, d, b, i, k, l, j) = t4d(a, b, c, d, i, j, k, l)
                        t4d(a, c, d, b, i, l, j, k) = t4d(a, b, c, d, i, j, k, l)
                        t4d(a, c, d, b, i, l, k, j) = -1.0 * t4d(a, b, c, d, i, j, k, l)
                        t4d(a, d, b, c, i, j, k, l) = t4d(a, b, c, d, i, j, k, l)
                        t4d(a, d, b, c, i, j, l, k) = -1.0 * t4d(a, b, c, d, i, j, k, l)
                        t4d(a, d, b, c, i, k, j, l) = -1.0 * t4d(a, b, c, d, i, j, k, l)
                        t4d(a, d, b, c, i, k, l, j) = t4d(a, b, c, d, i, j, k, l)
                        t4d(a, d, b, c, i, l, j, k) = t4d(a, b, c, d, i, j, k, l)
                        t4d(a, d, b, c, i, l, k, j) = -1.0 * t4d(a, b, c, d, i, j, k, l)
                        t4d(a, d, c, b, i, j, k, l) = -1.0 * t4d(a, b, c, d, i, j, k, l)
                        t4d(a, d, c, b, i, j, l, k) = t4d(a, b, c, d, i, j, k, l)
                        t4d(a, d, c, b, i, k, j, l) = t4d(a, b, c, d, i, j, k, l)
                        t4d(a, d, c, b, i, k, l, j) = -1.0 * t4d(a, b, c, d, i, j, k, l)
                        t4d(a, d, c, b, i, l, j, k) = -1.0 * t4d(a, b, c, d, i, j, k, l)
                        t4d(a, d, c, b, i, l, k, j) = t4d(a, b, c, d, i, j, k, l)

                        resid(a, b, c, d, i, j, k, l) = val
                        resid(a, b, c, d, i, j, l, k) = -1.0 * val
                        resid(a, b, c, d, i, k, j, l) = -1.0 * val
                        resid(a, b, c, d, i, k, l, j) = val
                        resid(a, b, c, d, i, l, j, k) = val
                        resid(a, b, c, d, i, l, k, j) = -1.0 * val
                        resid(a, b, d, c, i, j, k, l) = -1.0 * val
                        resid(a, b, d, c, i, j, l, k) = val
                        resid(a, b, d, c, i, k, j, l) = val
                        resid(a, b, d, c, i, k, l, j) = -1.0 * val
                        resid(a, b, d, c, i, l, j, k) = -1.0 * val
                        resid(a, b, d, c, i, l, k, j) = val
                        resid(a, c, b, d, i, j, k, l) = -1.0 * val
                        resid(a, c, b, d, i, j, l, k) = val
                        resid(a, c, b, d, i, k, j, l) = val
                        resid(a, c, b, d, i, k, l, j) = -1.0 * val
                        resid(a, c, b, d, i, l, j, k) = -1.0 * val
                        resid(a, c, b, d, i, l, k, j) = val
                        resid(a, c, d, b, i, j, k, l) = val
                        resid(a, c, d, b, i, j, l, k) = -1.0 * val
                        resid(a, c, d, b, i, k, j, l) = -1.0 * val
                        resid(a, c, d, b, i, k, l, j) = val
                        resid(a, c, d, b, i, l, j, k) = val
                        resid(a, c, d, b, i, l, k, j) = -1.0 * val
                        resid(a, d, b, c, i, j, k, l) = val
                        resid(a, d, b, c, i, j, l, k) = -1.0 * val
                        resid(a, d, b, c, i, k, j, l) = -1.0 * val
                        resid(a, d, b, c, i, k, l, j) = val
                        resid(a, d, b, c, i, l, j, k) = val
                        resid(a, d, b, c, i, l, k, j) = -1.0 * val
                        resid(a, d, c, b, i, j, k, l) = -1.0 * val
                        resid(a, d, c, b, i, j, l, k) = val
                        resid(a, d, c, b, i, k, j, l) = val
                        resid(a, d, c, b, i, k, l, j) = -1.0 * val
                        resid(a, d, c, b, i, l, j, k) = -1.0 * val
                        resid(a, d, c, b, i, l, k, j) = val

                        end do
                     end do
                  end do
               end do
            end do
         end do
      end do
   end do

end subroutine update_t4d

subroutine update_t4e(t4e, resid, X4E, &
                             fB_oo, fB_vv, &
                             shift, &
                             nob, nub)

      integer, intent(in)  :: nob, nub
      real(8), intent(in)  :: fB_oo(1:nob, 1:nob), &
                              fB_vv(1:nub, 1:nub)
      real(8), intent(in)  :: X4E(1:nub, 1:nub, 1:nub, 1:nub, 1:nob, 1:nob, 1:nob, 1:nob)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t4e(1:nub, 1:nub, 1:nub, 1:nub, 1:nob, 1:nob, 1:nob, 1:nob)
      !f2py intent(in, out)  :: t4e(0:nub-1, 0:nub-1, 0:nub-1, 0:nub-1, 0:nob-1, 0:nob-1, 0:nob-1, 0:nob-1)
      real(8), intent(out)   :: resid(1:nub, 1:nub, 1:nub, 1:nub, 1:nob, 1:nob, 1:nob, 1:nob)

      integer :: i, j, k, l, a, b, c, d
      real(8) :: denom, val

   do i = 1 , nob
      do j = i + 1 , nob
         do k = j + 1 , nob
            do l = k + 1 , nob
               do a = 1 , nub
                  do b = a + 1 , nub
                     do c = b + 1 , nub
                        do d = c + 1 , nub

                        denom = fB_oo(i,i) + fB_oo(j,j) + fB_oo(k,k) + fB_oo(l,l)&
                               -fB_vv(a,a) - fB_vv(b,b) - fB_vv(c,c) - fB_vv(d,d)

                        val = &
                        +X4E(a, b, c, d, i, j, k, l)&
                        -1.0 * X4E(a, b, c, d, i, j, l, k)&
                        -1.0 * X4E(a, b, c, d, i, k, j, l)&
                        +X4E(a, b, c, d, i, k, l, j)&
                        +X4E(a, b, c, d, i, l, j, k)&
                        -1.0 * X4E(a, b, c, d, i, l, k, j)&
                        -1.0 * X4E(a, b, c, d, j, i, k, l)&
                        +X4E(a, b, c, d, j, i, l, k)&
                        +X4E(a, b, c, d, j, k, i, l)&
                        -1.0 * X4E(a, b, c, d, j, k, l, i)&
                        -1.0 * X4E(a, b, c, d, j, l, i, k)&
                        +X4E(a, b, c, d, j, l, k, i)&
                        +X4E(a, b, c, d, k, i, j, l)&
                        -1.0 * X4E(a, b, c, d, k, i, l, j)&
                        -1.0 * X4E(a, b, c, d, k, j, i, l)&
                        +X4E(a, b, c, d, k, j, l, i)&
                        +X4E(a, b, c, d, k, l, i, j)&
                        -1.0 * X4E(a, b, c, d, k, l, j, i)&
                        -1.0 * X4E(a, b, c, d, l, i, j, k)&
                        +X4E(a, b, c, d, l, i, k, j)&
                        +X4E(a, b, c, d, l, j, i, k)&
                        -1.0 * X4E(a, b, c, d, l, j, k, i)&
                        -1.0 * X4E(a, b, c, d, l, k, i, j)&
                        +X4E(a, b, c, d, l, k, j, i)&
                        -1.0 * X4E(a, b, d, c, i, j, k, l)&
                        +X4E(a, b, d, c, i, j, l, k)&
                        +X4E(a, b, d, c, i, k, j, l)&
                        -1.0 * X4E(a, b, d, c, i, k, l, j)&
                        -1.0 * X4E(a, b, d, c, i, l, j, k)&
                        +X4E(a, b, d, c, i, l, k, j)&
                        +X4E(a, b, d, c, j, i, k, l)&
                        -1.0 * X4E(a, b, d, c, j, i, l, k)&
                        -1.0 * X4E(a, b, d, c, j, k, i, l)&
                        +X4E(a, b, d, c, j, k, l, i)&
                        +X4E(a, b, d, c, j, l, i, k)&
                        -1.0 * X4E(a, b, d, c, j, l, k, i)&
                        -1.0 * X4E(a, b, d, c, k, i, j, l)&
                        +X4E(a, b, d, c, k, i, l, j)&
                        +X4E(a, b, d, c, k, j, i, l)&
                        -1.0 * X4E(a, b, d, c, k, j, l, i)&
                        -1.0 * X4E(a, b, d, c, k, l, i, j)&
                        +X4E(a, b, d, c, k, l, j, i)&
                        +X4E(a, b, d, c, l, i, j, k)&
                        -1.0 * X4E(a, b, d, c, l, i, k, j)&
                        -1.0 * X4E(a, b, d, c, l, j, i, k)&
                        +X4E(a, b, d, c, l, j, k, i)&
                        +X4E(a, b, d, c, l, k, i, j)&
                        -1.0 * X4E(a, b, d, c, l, k, j, i)&
                        -1.0 * X4E(a, c, b, d, i, j, k, l)&
                        +X4E(a, c, b, d, i, j, l, k)&
                        +X4E(a, c, b, d, i, k, j, l)&
                        -1.0 * X4E(a, c, b, d, i, k, l, j)&
                        -1.0 * X4E(a, c, b, d, i, l, j, k)&
                        +X4E(a, c, b, d, i, l, k, j)&
                        +X4E(a, c, b, d, j, i, k, l)&
                        -1.0 * X4E(a, c, b, d, j, i, l, k)&
                        -1.0 * X4E(a, c, b, d, j, k, i, l)&
                        +X4E(a, c, b, d, j, k, l, i)&
                        +X4E(a, c, b, d, j, l, i, k)&
                        -1.0 * X4E(a, c, b, d, j, l, k, i)&
                        -1.0 * X4E(a, c, b, d, k, i, j, l)&
                        +X4E(a, c, b, d, k, i, l, j)&
                        +X4E(a, c, b, d, k, j, i, l)&
                        -1.0 * X4E(a, c, b, d, k, j, l, i)&
                        -1.0 * X4E(a, c, b, d, k, l, i, j)&
                        +X4E(a, c, b, d, k, l, j, i)&
                        +X4E(a, c, b, d, l, i, j, k)&
                        -1.0 * X4E(a, c, b, d, l, i, k, j)&
                        -1.0 * X4E(a, c, b, d, l, j, i, k)&
                        +X4E(a, c, b, d, l, j, k, i)&
                        +X4E(a, c, b, d, l, k, i, j)&
                        -1.0 * X4E(a, c, b, d, l, k, j, i)&
                        +X4E(a, c, d, b, i, j, k, l)&
                        -1.0 * X4E(a, c, d, b, i, j, l, k)&
                        -1.0 * X4E(a, c, d, b, i, k, j, l)&
                        +X4E(a, c, d, b, i, k, l, j)&
                        +X4E(a, c, d, b, i, l, j, k)&
                        -1.0 * X4E(a, c, d, b, i, l, k, j)&
                        -1.0 * X4E(a, c, d, b, j, i, k, l)&
                        +X4E(a, c, d, b, j, i, l, k)&
                        +X4E(a, c, d, b, j, k, i, l)&
                        -1.0 * X4E(a, c, d, b, j, k, l, i)&
                        -1.0 * X4E(a, c, d, b, j, l, i, k)&
                        +X4E(a, c, d, b, j, l, k, i)&
                        +X4E(a, c, d, b, k, i, j, l)&
                        -1.0 * X4E(a, c, d, b, k, i, l, j)&
                        -1.0 * X4E(a, c, d, b, k, j, i, l)&
                        +X4E(a, c, d, b, k, j, l, i)&
                        +X4E(a, c, d, b, k, l, i, j)&
                        -1.0 * X4E(a, c, d, b, k, l, j, i)&
                        -1.0 * X4E(a, c, d, b, l, i, j, k)&
                        +X4E(a, c, d, b, l, i, k, j)&
                        +X4E(a, c, d, b, l, j, i, k)&
                        -1.0 * X4E(a, c, d, b, l, j, k, i)&
                        -1.0 * X4E(a, c, d, b, l, k, i, j)&
                        +X4E(a, c, d, b, l, k, j, i)&
                        +X4E(a, d, b, c, i, j, k, l)&
                        -1.0 * X4E(a, d, b, c, i, j, l, k)&
                        -1.0 * X4E(a, d, b, c, i, k, j, l)&
                        +X4E(a, d, b, c, i, k, l, j)&
                        +X4E(a, d, b, c, i, l, j, k)&
                        -1.0 * X4E(a, d, b, c, i, l, k, j)&
                        -1.0 * X4E(a, d, b, c, j, i, k, l)&
                        +X4E(a, d, b, c, j, i, l, k)&
                        +X4E(a, d, b, c, j, k, i, l)&
                        -1.0 * X4E(a, d, b, c, j, k, l, i)&
                        -1.0 * X4E(a, d, b, c, j, l, i, k)&
                        +X4E(a, d, b, c, j, l, k, i)&
                        +X4E(a, d, b, c, k, i, j, l)&
                        -1.0 * X4E(a, d, b, c, k, i, l, j)&
                        -1.0 * X4E(a, d, b, c, k, j, i, l)&
                        +X4E(a, d, b, c, k, j, l, i)&
                        +X4E(a, d, b, c, k, l, i, j)&
                        -1.0 * X4E(a, d, b, c, k, l, j, i)&
                        -1.0 * X4E(a, d, b, c, l, i, j, k)&
                        +X4E(a, d, b, c, l, i, k, j)&
                        +X4E(a, d, b, c, l, j, i, k)&
                        -1.0 * X4E(a, d, b, c, l, j, k, i)&
                        -1.0 * X4E(a, d, b, c, l, k, i, j)&
                        +X4E(a, d, b, c, l, k, j, i)&
                        -1.0 * X4E(a, d, c, b, i, j, k, l)&
                        +X4E(a, d, c, b, i, j, l, k)&
                        +X4E(a, d, c, b, i, k, j, l)&
                        -1.0 * X4E(a, d, c, b, i, k, l, j)&
                        -1.0 * X4E(a, d, c, b, i, l, j, k)&
                        +X4E(a, d, c, b, i, l, k, j)&
                        +X4E(a, d, c, b, j, i, k, l)&
                        -1.0 * X4E(a, d, c, b, j, i, l, k)&
                        -1.0 * X4E(a, d, c, b, j, k, i, l)&
                        +X4E(a, d, c, b, j, k, l, i)&
                        +X4E(a, d, c, b, j, l, i, k)&
                        -1.0 * X4E(a, d, c, b, j, l, k, i)&
                        -1.0 * X4E(a, d, c, b, k, i, j, l)&
                        +X4E(a, d, c, b, k, i, l, j)&
                        +X4E(a, d, c, b, k, j, i, l)&
                        -1.0 * X4E(a, d, c, b, k, j, l, i)&
                        -1.0 * X4E(a, d, c, b, k, l, i, j)&
                        +X4E(a, d, c, b, k, l, j, i)&
                        +X4E(a, d, c, b, l, i, j, k)&
                        -1.0 * X4E(a, d, c, b, l, i, k, j)&
                        -1.0 * X4E(a, d, c, b, l, j, i, k)&
                        +X4E(a, d, c, b, l, j, k, i)&
                        +X4E(a, d, c, b, l, k, i, j)&
                        -1.0 * X4E(a, d, c, b, l, k, j, i)&
                        -1.0 * X4E(b, a, c, d, i, j, k, l)&
                        +X4E(b, a, c, d, i, j, l, k)&
                        +X4E(b, a, c, d, i, k, j, l)&
                        -1.0 * X4E(b, a, c, d, i, k, l, j)&
                        -1.0 * X4E(b, a, c, d, i, l, j, k)&
                        +X4E(b, a, c, d, i, l, k, j)&
                        +X4E(b, a, c, d, j, i, k, l)&
                        -1.0 * X4E(b, a, c, d, j, i, l, k)&
                        -1.0 * X4E(b, a, c, d, j, k, i, l)&
                        +X4E(b, a, c, d, j, k, l, i)&
                        +X4E(b, a, c, d, j, l, i, k)&
                        -1.0 * X4E(b, a, c, d, j, l, k, i)&
                        -1.0 * X4E(b, a, c, d, k, i, j, l)&
                        +X4E(b, a, c, d, k, i, l, j)&
                        +X4E(b, a, c, d, k, j, i, l)&
                        -1.0 * X4E(b, a, c, d, k, j, l, i)&
                        -1.0 * X4E(b, a, c, d, k, l, i, j)&
                        +X4E(b, a, c, d, k, l, j, i)&
                        +X4E(b, a, c, d, l, i, j, k)&
                        -1.0 * X4E(b, a, c, d, l, i, k, j)&
                        -1.0 * X4E(b, a, c, d, l, j, i, k)&
                        +X4E(b, a, c, d, l, j, k, i)&
                        +X4E(b, a, c, d, l, k, i, j)&
                        -1.0 * X4E(b, a, c, d, l, k, j, i)&
                        +X4E(b, a, d, c, i, j, k, l)&
                        -1.0 * X4E(b, a, d, c, i, j, l, k)&
                        -1.0 * X4E(b, a, d, c, i, k, j, l)&
                        +X4E(b, a, d, c, i, k, l, j)&
                        +X4E(b, a, d, c, i, l, j, k)&
                        -1.0 * X4E(b, a, d, c, i, l, k, j)&
                        -1.0 * X4E(b, a, d, c, j, i, k, l)&
                        +X4E(b, a, d, c, j, i, l, k)&
                        +X4E(b, a, d, c, j, k, i, l)&
                        -1.0 * X4E(b, a, d, c, j, k, l, i)&
                        -1.0 * X4E(b, a, d, c, j, l, i, k)&
                        +X4E(b, a, d, c, j, l, k, i)&
                        +X4E(b, a, d, c, k, i, j, l)&
                        -1.0 * X4E(b, a, d, c, k, i, l, j)&
                        -1.0 * X4E(b, a, d, c, k, j, i, l)&
                        +X4E(b, a, d, c, k, j, l, i)&
                        +X4E(b, a, d, c, k, l, i, j)&
                        -1.0 * X4E(b, a, d, c, k, l, j, i)&
                        -1.0 * X4E(b, a, d, c, l, i, j, k)&
                        +X4E(b, a, d, c, l, i, k, j)&
                        +X4E(b, a, d, c, l, j, i, k)&
                        -1.0 * X4E(b, a, d, c, l, j, k, i)&
                        -1.0 * X4E(b, a, d, c, l, k, i, j)&
                        +X4E(b, a, d, c, l, k, j, i)&
                        +X4E(b, c, a, d, i, j, k, l)&
                        -1.0 * X4E(b, c, a, d, i, j, l, k)&
                        -1.0 * X4E(b, c, a, d, i, k, j, l)&
                        +X4E(b, c, a, d, i, k, l, j)&
                        +X4E(b, c, a, d, i, l, j, k)&
                        -1.0 * X4E(b, c, a, d, i, l, k, j)&
                        -1.0 * X4E(b, c, a, d, j, i, k, l)&
                        +X4E(b, c, a, d, j, i, l, k)&
                        +X4E(b, c, a, d, j, k, i, l)&
                        -1.0 * X4E(b, c, a, d, j, k, l, i)&
                        -1.0 * X4E(b, c, a, d, j, l, i, k)&
                        +X4E(b, c, a, d, j, l, k, i)&
                        +X4E(b, c, a, d, k, i, j, l)&
                        -1.0 * X4E(b, c, a, d, k, i, l, j)&
                        -1.0 * X4E(b, c, a, d, k, j, i, l)&
                        +X4E(b, c, a, d, k, j, l, i)&
                        +X4E(b, c, a, d, k, l, i, j)&
                        -1.0 * X4E(b, c, a, d, k, l, j, i)&
                        -1.0 * X4E(b, c, a, d, l, i, j, k)&
                        +X4E(b, c, a, d, l, i, k, j)&
                        +X4E(b, c, a, d, l, j, i, k)&
                        -1.0 * X4E(b, c, a, d, l, j, k, i)&
                        -1.0 * X4E(b, c, a, d, l, k, i, j)&
                        +X4E(b, c, a, d, l, k, j, i)&
                        -1.0 * X4E(b, c, d, a, i, j, k, l)&
                        +X4E(b, c, d, a, i, j, l, k)&
                        +X4E(b, c, d, a, i, k, j, l)&
                        -1.0 * X4E(b, c, d, a, i, k, l, j)&
                        -1.0 * X4E(b, c, d, a, i, l, j, k)&
                        +X4E(b, c, d, a, i, l, k, j)&
                        +X4E(b, c, d, a, j, i, k, l)&
                        -1.0 * X4E(b, c, d, a, j, i, l, k)&
                        -1.0 * X4E(b, c, d, a, j, k, i, l)&
                        +X4E(b, c, d, a, j, k, l, i)&
                        +X4E(b, c, d, a, j, l, i, k)&
                        -1.0 * X4E(b, c, d, a, j, l, k, i)&
                        -1.0 * X4E(b, c, d, a, k, i, j, l)&
                        +X4E(b, c, d, a, k, i, l, j)&
                        +X4E(b, c, d, a, k, j, i, l)&
                        -1.0 * X4E(b, c, d, a, k, j, l, i)&
                        -1.0 * X4E(b, c, d, a, k, l, i, j)&
                        +X4E(b, c, d, a, k, l, j, i)&
                        +X4E(b, c, d, a, l, i, j, k)&
                        -1.0 * X4E(b, c, d, a, l, i, k, j)&
                        -1.0 * X4E(b, c, d, a, l, j, i, k)&
                        +X4E(b, c, d, a, l, j, k, i)&
                        +X4E(b, c, d, a, l, k, i, j)&
                        -1.0 * X4E(b, c, d, a, l, k, j, i)&
                        -1.0 * X4E(b, d, a, c, i, j, k, l)&
                        +X4E(b, d, a, c, i, j, l, k)&
                        +X4E(b, d, a, c, i, k, j, l)&
                        -1.0 * X4E(b, d, a, c, i, k, l, j)&
                        -1.0 * X4E(b, d, a, c, i, l, j, k)&
                        +X4E(b, d, a, c, i, l, k, j)&
                        +X4E(b, d, a, c, j, i, k, l)&
                        -1.0 * X4E(b, d, a, c, j, i, l, k)&
                        -1.0 * X4E(b, d, a, c, j, k, i, l)&
                        +X4E(b, d, a, c, j, k, l, i)&
                        +X4E(b, d, a, c, j, l, i, k)&
                        -1.0 * X4E(b, d, a, c, j, l, k, i)&
                        -1.0 * X4E(b, d, a, c, k, i, j, l)&
                        +X4E(b, d, a, c, k, i, l, j)&
                        +X4E(b, d, a, c, k, j, i, l)&
                        -1.0 * X4E(b, d, a, c, k, j, l, i)&
                        -1.0 * X4E(b, d, a, c, k, l, i, j)&
                        +X4E(b, d, a, c, k, l, j, i)&
                        +X4E(b, d, a, c, l, i, j, k)&
                        -1.0 * X4E(b, d, a, c, l, i, k, j)&
                        -1.0 * X4E(b, d, a, c, l, j, i, k)&
                        +X4E(b, d, a, c, l, j, k, i)&
                        +X4E(b, d, a, c, l, k, i, j)&
                        -1.0 * X4E(b, d, a, c, l, k, j, i)&
                        +X4E(b, d, c, a, i, j, k, l)&
                        -1.0 * X4E(b, d, c, a, i, j, l, k)&
                        -1.0 * X4E(b, d, c, a, i, k, j, l)&
                        +X4E(b, d, c, a, i, k, l, j)&
                        +X4E(b, d, c, a, i, l, j, k)&
                        -1.0 * X4E(b, d, c, a, i, l, k, j)&
                        -1.0 * X4E(b, d, c, a, j, i, k, l)&
                        +X4E(b, d, c, a, j, i, l, k)&
                        +X4E(b, d, c, a, j, k, i, l)&
                        -1.0 * X4E(b, d, c, a, j, k, l, i)&
                        -1.0 * X4E(b, d, c, a, j, l, i, k)&
                        +X4E(b, d, c, a, j, l, k, i)&
                        +X4E(b, d, c, a, k, i, j, l)&
                        -1.0 * X4E(b, d, c, a, k, i, l, j)&
                        -1.0 * X4E(b, d, c, a, k, j, i, l)&
                        +X4E(b, d, c, a, k, j, l, i)&
                        +X4E(b, d, c, a, k, l, i, j)&
                        -1.0 * X4E(b, d, c, a, k, l, j, i)&
                        -1.0 * X4E(b, d, c, a, l, i, j, k)&
                        +X4E(b, d, c, a, l, i, k, j)&
                        +X4E(b, d, c, a, l, j, i, k)&
                        -1.0 * X4E(b, d, c, a, l, j, k, i)&
                        -1.0 * X4E(b, d, c, a, l, k, i, j)&
                        +X4E(b, d, c, a, l, k, j, i)&
                        +X4E(c, a, b, d, i, j, k, l)&
                        -1.0 * X4E(c, a, b, d, i, j, l, k)&
                        -1.0 * X4E(c, a, b, d, i, k, j, l)&
                        +X4E(c, a, b, d, i, k, l, j)&
                        +X4E(c, a, b, d, i, l, j, k)&
                        -1.0 * X4E(c, a, b, d, i, l, k, j)&
                        -1.0 * X4E(c, a, b, d, j, i, k, l)&
                        +X4E(c, a, b, d, j, i, l, k)&
                        +X4E(c, a, b, d, j, k, i, l)&
                        -1.0 * X4E(c, a, b, d, j, k, l, i)&
                        -1.0 * X4E(c, a, b, d, j, l, i, k)&
                        +X4E(c, a, b, d, j, l, k, i)&
                        +X4E(c, a, b, d, k, i, j, l)&
                        -1.0 * X4E(c, a, b, d, k, i, l, j)&
                        -1.0 * X4E(c, a, b, d, k, j, i, l)&
                        +X4E(c, a, b, d, k, j, l, i)&
                        +X4E(c, a, b, d, k, l, i, j)&
                        -1.0 * X4E(c, a, b, d, k, l, j, i)&
                        -1.0 * X4E(c, a, b, d, l, i, j, k)&
                        +X4E(c, a, b, d, l, i, k, j)&
                        +X4E(c, a, b, d, l, j, i, k)&
                        -1.0 * X4E(c, a, b, d, l, j, k, i)&
                        -1.0 * X4E(c, a, b, d, l, k, i, j)&
                        +X4E(c, a, b, d, l, k, j, i)&
                        -1.0 * X4E(c, a, d, b, i, j, k, l)&
                        +X4E(c, a, d, b, i, j, l, k)&
                        +X4E(c, a, d, b, i, k, j, l)&
                        -1.0 * X4E(c, a, d, b, i, k, l, j)&
                        -1.0 * X4E(c, a, d, b, i, l, j, k)&
                        +X4E(c, a, d, b, i, l, k, j)&
                        +X4E(c, a, d, b, j, i, k, l)&
                        -1.0 * X4E(c, a, d, b, j, i, l, k)&
                        -1.0 * X4E(c, a, d, b, j, k, i, l)&
                        +X4E(c, a, d, b, j, k, l, i)&
                        +X4E(c, a, d, b, j, l, i, k)&
                        -1.0 * X4E(c, a, d, b, j, l, k, i)&
                        -1.0 * X4E(c, a, d, b, k, i, j, l)&
                        +X4E(c, a, d, b, k, i, l, j)&
                        +X4E(c, a, d, b, k, j, i, l)&
                        -1.0 * X4E(c, a, d, b, k, j, l, i)&
                        -1.0 * X4E(c, a, d, b, k, l, i, j)&
                        +X4E(c, a, d, b, k, l, j, i)&
                        +X4E(c, a, d, b, l, i, j, k)&
                        -1.0 * X4E(c, a, d, b, l, i, k, j)&
                        -1.0 * X4E(c, a, d, b, l, j, i, k)&
                        +X4E(c, a, d, b, l, j, k, i)&
                        +X4E(c, a, d, b, l, k, i, j)&
                        -1.0 * X4E(c, a, d, b, l, k, j, i)&
                        -1.0 * X4E(c, b, a, d, i, j, k, l)&
                        +X4E(c, b, a, d, i, j, l, k)&
                        +X4E(c, b, a, d, i, k, j, l)&
                        -1.0 * X4E(c, b, a, d, i, k, l, j)&
                        -1.0 * X4E(c, b, a, d, i, l, j, k)&
                        +X4E(c, b, a, d, i, l, k, j)&
                        +X4E(c, b, a, d, j, i, k, l)&
                        -1.0 * X4E(c, b, a, d, j, i, l, k)&
                        -1.0 * X4E(c, b, a, d, j, k, i, l)&
                        +X4E(c, b, a, d, j, k, l, i)&
                        +X4E(c, b, a, d, j, l, i, k)&
                        -1.0 * X4E(c, b, a, d, j, l, k, i)&
                        -1.0 * X4E(c, b, a, d, k, i, j, l)&
                        +X4E(c, b, a, d, k, i, l, j)&
                        +X4E(c, b, a, d, k, j, i, l)&
                        -1.0 * X4E(c, b, a, d, k, j, l, i)&
                        -1.0 * X4E(c, b, a, d, k, l, i, j)&
                        +X4E(c, b, a, d, k, l, j, i)&
                        +X4E(c, b, a, d, l, i, j, k)&
                        -1.0 * X4E(c, b, a, d, l, i, k, j)&
                        -1.0 * X4E(c, b, a, d, l, j, i, k)&
                        +X4E(c, b, a, d, l, j, k, i)&
                        +X4E(c, b, a, d, l, k, i, j)&
                        -1.0 * X4E(c, b, a, d, l, k, j, i)&
                        +X4E(c, b, d, a, i, j, k, l)&
                        -1.0 * X4E(c, b, d, a, i, j, l, k)&
                        -1.0 * X4E(c, b, d, a, i, k, j, l)&
                        +X4E(c, b, d, a, i, k, l, j)&
                        +X4E(c, b, d, a, i, l, j, k)&
                        -1.0 * X4E(c, b, d, a, i, l, k, j)&
                        -1.0 * X4E(c, b, d, a, j, i, k, l)&
                        +X4E(c, b, d, a, j, i, l, k)&
                        +X4E(c, b, d, a, j, k, i, l)&
                        -1.0 * X4E(c, b, d, a, j, k, l, i)&
                        -1.0 * X4E(c, b, d, a, j, l, i, k)&
                        +X4E(c, b, d, a, j, l, k, i)&
                        +X4E(c, b, d, a, k, i, j, l)&
                        -1.0 * X4E(c, b, d, a, k, i, l, j)&
                        -1.0 * X4E(c, b, d, a, k, j, i, l)&
                        +X4E(c, b, d, a, k, j, l, i)&
                        +X4E(c, b, d, a, k, l, i, j)&
                        -1.0 * X4E(c, b, d, a, k, l, j, i)&
                        -1.0 * X4E(c, b, d, a, l, i, j, k)&
                        +X4E(c, b, d, a, l, i, k, j)&
                        +X4E(c, b, d, a, l, j, i, k)&
                        -1.0 * X4E(c, b, d, a, l, j, k, i)&
                        -1.0 * X4E(c, b, d, a, l, k, i, j)&
                        +X4E(c, b, d, a, l, k, j, i)&
                        +X4E(c, d, a, b, i, j, k, l)&
                        -1.0 * X4E(c, d, a, b, i, j, l, k)&
                        -1.0 * X4E(c, d, a, b, i, k, j, l)&
                        +X4E(c, d, a, b, i, k, l, j)&
                        +X4E(c, d, a, b, i, l, j, k)&
                        -1.0 * X4E(c, d, a, b, i, l, k, j)&
                        -1.0 * X4E(c, d, a, b, j, i, k, l)&
                        +X4E(c, d, a, b, j, i, l, k)&
                        +X4E(c, d, a, b, j, k, i, l)&
                        -1.0 * X4E(c, d, a, b, j, k, l, i)&
                        -1.0 * X4E(c, d, a, b, j, l, i, k)&
                        +X4E(c, d, a, b, j, l, k, i)&
                        +X4E(c, d, a, b, k, i, j, l)&
                        -1.0 * X4E(c, d, a, b, k, i, l, j)&
                        -1.0 * X4E(c, d, a, b, k, j, i, l)&
                        +X4E(c, d, a, b, k, j, l, i)&
                        +X4E(c, d, a, b, k, l, i, j)&
                        -1.0 * X4E(c, d, a, b, k, l, j, i)&
                        -1.0 * X4E(c, d, a, b, l, i, j, k)&
                        +X4E(c, d, a, b, l, i, k, j)&
                        +X4E(c, d, a, b, l, j, i, k)&
                        -1.0 * X4E(c, d, a, b, l, j, k, i)&
                        -1.0 * X4E(c, d, a, b, l, k, i, j)&
                        +X4E(c, d, a, b, l, k, j, i)&
                        -1.0 * X4E(c, d, b, a, i, j, k, l)&
                        +X4E(c, d, b, a, i, j, l, k)&
                        +X4E(c, d, b, a, i, k, j, l)&
                        -1.0 * X4E(c, d, b, a, i, k, l, j)&
                        -1.0 * X4E(c, d, b, a, i, l, j, k)&
                        +X4E(c, d, b, a, i, l, k, j)&
                        +X4E(c, d, b, a, j, i, k, l)&
                        -1.0 * X4E(c, d, b, a, j, i, l, k)&
                        -1.0 * X4E(c, d, b, a, j, k, i, l)&
                        +X4E(c, d, b, a, j, k, l, i)&
                        +X4E(c, d, b, a, j, l, i, k)&
                        -1.0 * X4E(c, d, b, a, j, l, k, i)&
                        -1.0 * X4E(c, d, b, a, k, i, j, l)&
                        +X4E(c, d, b, a, k, i, l, j)&
                        +X4E(c, d, b, a, k, j, i, l)&
                        -1.0 * X4E(c, d, b, a, k, j, l, i)&
                        -1.0 * X4E(c, d, b, a, k, l, i, j)&
                        +X4E(c, d, b, a, k, l, j, i)&
                        +X4E(c, d, b, a, l, i, j, k)&
                        -1.0 * X4E(c, d, b, a, l, i, k, j)&
                        -1.0 * X4E(c, d, b, a, l, j, i, k)&
                        +X4E(c, d, b, a, l, j, k, i)&
                        +X4E(c, d, b, a, l, k, i, j)&
                        -1.0 * X4E(c, d, b, a, l, k, j, i)&
                        -1.0 * X4E(d, a, b, c, i, j, k, l)&
                        +X4E(d, a, b, c, i, j, l, k)&
                        +X4E(d, a, b, c, i, k, j, l)&
                        -1.0 * X4E(d, a, b, c, i, k, l, j)&
                        -1.0 * X4E(d, a, b, c, i, l, j, k)&
                        +X4E(d, a, b, c, i, l, k, j)&
                        +X4E(d, a, b, c, j, i, k, l)&
                        -1.0 * X4E(d, a, b, c, j, i, l, k)&
                        -1.0 * X4E(d, a, b, c, j, k, i, l)&
                        +X4E(d, a, b, c, j, k, l, i)&
                        +X4E(d, a, b, c, j, l, i, k)&
                        -1.0 * X4E(d, a, b, c, j, l, k, i)&
                        -1.0 * X4E(d, a, b, c, k, i, j, l)&
                        +X4E(d, a, b, c, k, i, l, j)&
                        +X4E(d, a, b, c, k, j, i, l)&
                        -1.0 * X4E(d, a, b, c, k, j, l, i)&
                        -1.0 * X4E(d, a, b, c, k, l, i, j)&
                        +X4E(d, a, b, c, k, l, j, i)&
                        +X4E(d, a, b, c, l, i, j, k)&
                        -1.0 * X4E(d, a, b, c, l, i, k, j)&
                        -1.0 * X4E(d, a, b, c, l, j, i, k)&
                        +X4E(d, a, b, c, l, j, k, i)&
                        +X4E(d, a, b, c, l, k, i, j)&
                        -1.0 * X4E(d, a, b, c, l, k, j, i)&
                        +X4E(d, a, c, b, i, j, k, l)&
                        -1.0 * X4E(d, a, c, b, i, j, l, k)&
                        -1.0 * X4E(d, a, c, b, i, k, j, l)&
                        +X4E(d, a, c, b, i, k, l, j)&
                        +X4E(d, a, c, b, i, l, j, k)&
                        -1.0 * X4E(d, a, c, b, i, l, k, j)&
                        -1.0 * X4E(d, a, c, b, j, i, k, l)&
                        +X4E(d, a, c, b, j, i, l, k)&
                        +X4E(d, a, c, b, j, k, i, l)&
                        -1.0 * X4E(d, a, c, b, j, k, l, i)&
                        -1.0 * X4E(d, a, c, b, j, l, i, k)&
                        +X4E(d, a, c, b, j, l, k, i)&
                        +X4E(d, a, c, b, k, i, j, l)&
                        -1.0 * X4E(d, a, c, b, k, i, l, j)&
                        -1.0 * X4E(d, a, c, b, k, j, i, l)&
                        +X4E(d, a, c, b, k, j, l, i)&
                        +X4E(d, a, c, b, k, l, i, j)&
                        -1.0 * X4E(d, a, c, b, k, l, j, i)&
                        -1.0 * X4E(d, a, c, b, l, i, j, k)&
                        +X4E(d, a, c, b, l, i, k, j)&
                        +X4E(d, a, c, b, l, j, i, k)&
                        -1.0 * X4E(d, a, c, b, l, j, k, i)&
                        -1.0 * X4E(d, a, c, b, l, k, i, j)&
                        +X4E(d, a, c, b, l, k, j, i)&
                        +X4E(d, b, a, c, i, j, k, l)&
                        -1.0 * X4E(d, b, a, c, i, j, l, k)&
                        -1.0 * X4E(d, b, a, c, i, k, j, l)&
                        +X4E(d, b, a, c, i, k, l, j)&
                        +X4E(d, b, a, c, i, l, j, k)&
                        -1.0 * X4E(d, b, a, c, i, l, k, j)&
                        -1.0 * X4E(d, b, a, c, j, i, k, l)&
                        +X4E(d, b, a, c, j, i, l, k)&
                        +X4E(d, b, a, c, j, k, i, l)&
                        -1.0 * X4E(d, b, a, c, j, k, l, i)&
                        -1.0 * X4E(d, b, a, c, j, l, i, k)&
                        +X4E(d, b, a, c, j, l, k, i)&
                        +X4E(d, b, a, c, k, i, j, l)&
                        -1.0 * X4E(d, b, a, c, k, i, l, j)&
                        -1.0 * X4E(d, b, a, c, k, j, i, l)&
                        +X4E(d, b, a, c, k, j, l, i)&
                        +X4E(d, b, a, c, k, l, i, j)&
                        -1.0 * X4E(d, b, a, c, k, l, j, i)&
                        -1.0 * X4E(d, b, a, c, l, i, j, k)&
                        +X4E(d, b, a, c, l, i, k, j)&
                        +X4E(d, b, a, c, l, j, i, k)&
                        -1.0 * X4E(d, b, a, c, l, j, k, i)&
                        -1.0 * X4E(d, b, a, c, l, k, i, j)&
                        +X4E(d, b, a, c, l, k, j, i)&
                        -1.0 * X4E(d, b, c, a, i, j, k, l)&
                        +X4E(d, b, c, a, i, j, l, k)&
                        +X4E(d, b, c, a, i, k, j, l)&
                        -1.0 * X4E(d, b, c, a, i, k, l, j)&
                        -1.0 * X4E(d, b, c, a, i, l, j, k)&
                        +X4E(d, b, c, a, i, l, k, j)&
                        +X4E(d, b, c, a, j, i, k, l)&
                        -1.0 * X4E(d, b, c, a, j, i, l, k)&
                        -1.0 * X4E(d, b, c, a, j, k, i, l)&
                        +X4E(d, b, c, a, j, k, l, i)&
                        +X4E(d, b, c, a, j, l, i, k)&
                        -1.0 * X4E(d, b, c, a, j, l, k, i)&
                        -1.0 * X4E(d, b, c, a, k, i, j, l)&
                        +X4E(d, b, c, a, k, i, l, j)&
                        +X4E(d, b, c, a, k, j, i, l)&
                        -1.0 * X4E(d, b, c, a, k, j, l, i)&
                        -1.0 * X4E(d, b, c, a, k, l, i, j)&
                        +X4E(d, b, c, a, k, l, j, i)&
                        +X4E(d, b, c, a, l, i, j, k)&
                        -1.0 * X4E(d, b, c, a, l, i, k, j)&
                        -1.0 * X4E(d, b, c, a, l, j, i, k)&
                        +X4E(d, b, c, a, l, j, k, i)&
                        +X4E(d, b, c, a, l, k, i, j)&
                        -1.0 * X4E(d, b, c, a, l, k, j, i)&
                        -1.0 * X4E(d, c, a, b, i, j, k, l)&
                        +X4E(d, c, a, b, i, j, l, k)&
                        +X4E(d, c, a, b, i, k, j, l)&
                        -1.0 * X4E(d, c, a, b, i, k, l, j)&
                        -1.0 * X4E(d, c, a, b, i, l, j, k)&
                        +X4E(d, c, a, b, i, l, k, j)&
                        +X4E(d, c, a, b, j, i, k, l)&
                        -1.0 * X4E(d, c, a, b, j, i, l, k)&
                        -1.0 * X4E(d, c, a, b, j, k, i, l)&
                        +X4E(d, c, a, b, j, k, l, i)&
                        +X4E(d, c, a, b, j, l, i, k)&
                        -1.0 * X4E(d, c, a, b, j, l, k, i)&
                        -1.0 * X4E(d, c, a, b, k, i, j, l)&
                        +X4E(d, c, a, b, k, i, l, j)&
                        +X4E(d, c, a, b, k, j, i, l)&
                        -1.0 * X4E(d, c, a, b, k, j, l, i)&
                        -1.0 * X4E(d, c, a, b, k, l, i, j)&
                        +X4E(d, c, a, b, k, l, j, i)&
                        +X4E(d, c, a, b, l, i, j, k)&
                        -1.0 * X4E(d, c, a, b, l, i, k, j)&
                        -1.0 * X4E(d, c, a, b, l, j, i, k)&
                        +X4E(d, c, a, b, l, j, k, i)&
                        +X4E(d, c, a, b, l, k, i, j)&
                        -1.0 * X4E(d, c, a, b, l, k, j, i)&
                        +X4E(d, c, b, a, i, j, k, l)&
                        -1.0 * X4E(d, c, b, a, i, j, l, k)&
                        -1.0 * X4E(d, c, b, a, i, k, j, l)&
                        +X4E(d, c, b, a, i, k, l, j)&
                        +X4E(d, c, b, a, i, l, j, k)&
                        -1.0 * X4E(d, c, b, a, i, l, k, j)&
                        -1.0 * X4E(d, c, b, a, j, i, k, l)&
                        +X4E(d, c, b, a, j, i, l, k)&
                        +X4E(d, c, b, a, j, k, i, l)&
                        -1.0 * X4E(d, c, b, a, j, k, l, i)&
                        -1.0 * X4E(d, c, b, a, j, l, i, k)&
                        +X4E(d, c, b, a, j, l, k, i)&
                        +X4E(d, c, b, a, k, i, j, l)&
                        -1.0 * X4E(d, c, b, a, k, i, l, j)&
                        -1.0 * X4E(d, c, b, a, k, j, i, l)&
                        +X4E(d, c, b, a, k, j, l, i)&
                        +X4E(d, c, b, a, k, l, i, j)&
                        -1.0 * X4E(d, c, b, a, k, l, j, i)&
                        -1.0 * X4E(d, c, b, a, l, i, j, k)&
                        +X4E(d, c, b, a, l, i, k, j)&
                        +X4E(d, c, b, a, l, j, i, k)&
                        -1.0 * X4E(d, c, b, a, l, j, k, i)&
                        -1.0 * X4E(d, c, b, a, l, k, i, j)&
                        +X4E(d, c, b, a, l, k, j, i)

                        t4e(a, b, c, d, i, j, k, l) = t4e(a, b, c, d, i, j, k, l) + val/(denom-shift)
                        t4e(a, b, c, d, i, j, l, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, c, d, i, k, j, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, c, d, i, k, l, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, c, d, i, l, j, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, c, d, i, l, k, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, c, d, j, i, k, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, c, d, j, i, l, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, c, d, j, k, i, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, c, d, j, k, l, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, c, d, j, l, i, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, c, d, j, l, k, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, c, d, k, i, j, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, c, d, k, i, l, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, c, d, k, j, i, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, c, d, k, j, l, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, c, d, k, l, i, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, c, d, k, l, j, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, c, d, l, i, j, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, c, d, l, i, k, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, c, d, l, j, i, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, c, d, l, j, k, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, c, d, l, k, i, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, c, d, l, k, j, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, d, c, i, j, k, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, d, c, i, j, l, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, d, c, i, k, j, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, d, c, i, k, l, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, d, c, i, l, j, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, d, c, i, l, k, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, d, c, j, i, k, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, d, c, j, i, l, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, d, c, j, k, i, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, d, c, j, k, l, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, d, c, j, l, i, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, d, c, j, l, k, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, d, c, k, i, j, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, d, c, k, i, l, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, d, c, k, j, i, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, d, c, k, j, l, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, d, c, k, l, i, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, d, c, k, l, j, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, d, c, l, i, j, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, d, c, l, i, k, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, d, c, l, j, i, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, d, c, l, j, k, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, d, c, l, k, i, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, b, d, c, l, k, j, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, b, d, i, j, k, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, b, d, i, j, l, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, b, d, i, k, j, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, b, d, i, k, l, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, b, d, i, l, j, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, b, d, i, l, k, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, b, d, j, i, k, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, b, d, j, i, l, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, b, d, j, k, i, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, b, d, j, k, l, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, b, d, j, l, i, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, b, d, j, l, k, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, b, d, k, i, j, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, b, d, k, i, l, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, b, d, k, j, i, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, b, d, k, j, l, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, b, d, k, l, i, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, b, d, k, l, j, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, b, d, l, i, j, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, b, d, l, i, k, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, b, d, l, j, i, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, b, d, l, j, k, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, b, d, l, k, i, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, b, d, l, k, j, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, d, b, i, j, k, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, d, b, i, j, l, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, d, b, i, k, j, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, d, b, i, k, l, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, d, b, i, l, j, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, d, b, i, l, k, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, d, b, j, i, k, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, d, b, j, i, l, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, d, b, j, k, i, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, d, b, j, k, l, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, d, b, j, l, i, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, d, b, j, l, k, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, d, b, k, i, j, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, d, b, k, i, l, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, d, b, k, j, i, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, d, b, k, j, l, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, d, b, k, l, i, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, d, b, k, l, j, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, d, b, l, i, j, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, d, b, l, i, k, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, d, b, l, j, i, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, d, b, l, j, k, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, d, b, l, k, i, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, c, d, b, l, k, j, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, b, c, i, j, k, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, b, c, i, j, l, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, b, c, i, k, j, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, b, c, i, k, l, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, b, c, i, l, j, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, b, c, i, l, k, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, b, c, j, i, k, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, b, c, j, i, l, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, b, c, j, k, i, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, b, c, j, k, l, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, b, c, j, l, i, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, b, c, j, l, k, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, b, c, k, i, j, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, b, c, k, i, l, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, b, c, k, j, i, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, b, c, k, j, l, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, b, c, k, l, i, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, b, c, k, l, j, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, b, c, l, i, j, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, b, c, l, i, k, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, b, c, l, j, i, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, b, c, l, j, k, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, b, c, l, k, i, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, b, c, l, k, j, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, c, b, i, j, k, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, c, b, i, j, l, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, c, b, i, k, j, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, c, b, i, k, l, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, c, b, i, l, j, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, c, b, i, l, k, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, c, b, j, i, k, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, c, b, j, i, l, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, c, b, j, k, i, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, c, b, j, k, l, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, c, b, j, l, i, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, c, b, j, l, k, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, c, b, k, i, j, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, c, b, k, i, l, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, c, b, k, j, i, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, c, b, k, j, l, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, c, b, k, l, i, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, c, b, k, l, j, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, c, b, l, i, j, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, c, b, l, i, k, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, c, b, l, j, i, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, c, b, l, j, k, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, c, b, l, k, i, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(a, d, c, b, l, k, j, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, c, d, i, j, k, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, c, d, i, j, l, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, c, d, i, k, j, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, c, d, i, k, l, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, c, d, i, l, j, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, c, d, i, l, k, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, c, d, j, i, k, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, c, d, j, i, l, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, c, d, j, k, i, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, c, d, j, k, l, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, c, d, j, l, i, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, c, d, j, l, k, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, c, d, k, i, j, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, c, d, k, i, l, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, c, d, k, j, i, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, c, d, k, j, l, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, c, d, k, l, i, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, c, d, k, l, j, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, c, d, l, i, j, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, c, d, l, i, k, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, c, d, l, j, i, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, c, d, l, j, k, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, c, d, l, k, i, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, c, d, l, k, j, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, d, c, i, j, k, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, d, c, i, j, l, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, d, c, i, k, j, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, d, c, i, k, l, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, d, c, i, l, j, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, d, c, i, l, k, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, d, c, j, i, k, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, d, c, j, i, l, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, d, c, j, k, i, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, d, c, j, k, l, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, d, c, j, l, i, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, d, c, j, l, k, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, d, c, k, i, j, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, d, c, k, i, l, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, d, c, k, j, i, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, d, c, k, j, l, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, d, c, k, l, i, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, d, c, k, l, j, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, d, c, l, i, j, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, d, c, l, i, k, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, d, c, l, j, i, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, d, c, l, j, k, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, d, c, l, k, i, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, a, d, c, l, k, j, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, a, d, i, j, k, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, a, d, i, j, l, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, a, d, i, k, j, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, a, d, i, k, l, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, a, d, i, l, j, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, a, d, i, l, k, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, a, d, j, i, k, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, a, d, j, i, l, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, a, d, j, k, i, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, a, d, j, k, l, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, a, d, j, l, i, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, a, d, j, l, k, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, a, d, k, i, j, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, a, d, k, i, l, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, a, d, k, j, i, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, a, d, k, j, l, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, a, d, k, l, i, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, a, d, k, l, j, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, a, d, l, i, j, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, a, d, l, i, k, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, a, d, l, j, i, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, a, d, l, j, k, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, a, d, l, k, i, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, a, d, l, k, j, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, d, a, i, j, k, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, d, a, i, j, l, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, d, a, i, k, j, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, d, a, i, k, l, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, d, a, i, l, j, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, d, a, i, l, k, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, d, a, j, i, k, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, d, a, j, i, l, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, d, a, j, k, i, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, d, a, j, k, l, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, d, a, j, l, i, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, d, a, j, l, k, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, d, a, k, i, j, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, d, a, k, i, l, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, d, a, k, j, i, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, d, a, k, j, l, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, d, a, k, l, i, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, d, a, k, l, j, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, d, a, l, i, j, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, d, a, l, i, k, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, d, a, l, j, i, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, d, a, l, j, k, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, d, a, l, k, i, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, c, d, a, l, k, j, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, a, c, i, j, k, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, a, c, i, j, l, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, a, c, i, k, j, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, a, c, i, k, l, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, a, c, i, l, j, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, a, c, i, l, k, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, a, c, j, i, k, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, a, c, j, i, l, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, a, c, j, k, i, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, a, c, j, k, l, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, a, c, j, l, i, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, a, c, j, l, k, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, a, c, k, i, j, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, a, c, k, i, l, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, a, c, k, j, i, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, a, c, k, j, l, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, a, c, k, l, i, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, a, c, k, l, j, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, a, c, l, i, j, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, a, c, l, i, k, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, a, c, l, j, i, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, a, c, l, j, k, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, a, c, l, k, i, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, a, c, l, k, j, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, c, a, i, j, k, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, c, a, i, j, l, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, c, a, i, k, j, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, c, a, i, k, l, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, c, a, i, l, j, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, c, a, i, l, k, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, c, a, j, i, k, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, c, a, j, i, l, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, c, a, j, k, i, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, c, a, j, k, l, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, c, a, j, l, i, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, c, a, j, l, k, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, c, a, k, i, j, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, c, a, k, i, l, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, c, a, k, j, i, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, c, a, k, j, l, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, c, a, k, l, i, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, c, a, k, l, j, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, c, a, l, i, j, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, c, a, l, i, k, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, c, a, l, j, i, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, c, a, l, j, k, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, c, a, l, k, i, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(b, d, c, a, l, k, j, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, b, d, i, j, k, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, b, d, i, j, l, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, b, d, i, k, j, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, b, d, i, k, l, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, b, d, i, l, j, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, b, d, i, l, k, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, b, d, j, i, k, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, b, d, j, i, l, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, b, d, j, k, i, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, b, d, j, k, l, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, b, d, j, l, i, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, b, d, j, l, k, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, b, d, k, i, j, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, b, d, k, i, l, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, b, d, k, j, i, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, b, d, k, j, l, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, b, d, k, l, i, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, b, d, k, l, j, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, b, d, l, i, j, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, b, d, l, i, k, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, b, d, l, j, i, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, b, d, l, j, k, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, b, d, l, k, i, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, b, d, l, k, j, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, d, b, i, j, k, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, d, b, i, j, l, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, d, b, i, k, j, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, d, b, i, k, l, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, d, b, i, l, j, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, d, b, i, l, k, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, d, b, j, i, k, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, d, b, j, i, l, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, d, b, j, k, i, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, d, b, j, k, l, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, d, b, j, l, i, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, d, b, j, l, k, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, d, b, k, i, j, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, d, b, k, i, l, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, d, b, k, j, i, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, d, b, k, j, l, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, d, b, k, l, i, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, d, b, k, l, j, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, d, b, l, i, j, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, d, b, l, i, k, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, d, b, l, j, i, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, d, b, l, j, k, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, d, b, l, k, i, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, a, d, b, l, k, j, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, a, d, i, j, k, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, a, d, i, j, l, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, a, d, i, k, j, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, a, d, i, k, l, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, a, d, i, l, j, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, a, d, i, l, k, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, a, d, j, i, k, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, a, d, j, i, l, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, a, d, j, k, i, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, a, d, j, k, l, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, a, d, j, l, i, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, a, d, j, l, k, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, a, d, k, i, j, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, a, d, k, i, l, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, a, d, k, j, i, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, a, d, k, j, l, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, a, d, k, l, i, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, a, d, k, l, j, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, a, d, l, i, j, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, a, d, l, i, k, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, a, d, l, j, i, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, a, d, l, j, k, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, a, d, l, k, i, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, a, d, l, k, j, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, d, a, i, j, k, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, d, a, i, j, l, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, d, a, i, k, j, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, d, a, i, k, l, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, d, a, i, l, j, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, d, a, i, l, k, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, d, a, j, i, k, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, d, a, j, i, l, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, d, a, j, k, i, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, d, a, j, k, l, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, d, a, j, l, i, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, d, a, j, l, k, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, d, a, k, i, j, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, d, a, k, i, l, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, d, a, k, j, i, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, d, a, k, j, l, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, d, a, k, l, i, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, d, a, k, l, j, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, d, a, l, i, j, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, d, a, l, i, k, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, d, a, l, j, i, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, d, a, l, j, k, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, d, a, l, k, i, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, b, d, a, l, k, j, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, a, b, i, j, k, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, a, b, i, j, l, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, a, b, i, k, j, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, a, b, i, k, l, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, a, b, i, l, j, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, a, b, i, l, k, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, a, b, j, i, k, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, a, b, j, i, l, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, a, b, j, k, i, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, a, b, j, k, l, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, a, b, j, l, i, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, a, b, j, l, k, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, a, b, k, i, j, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, a, b, k, i, l, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, a, b, k, j, i, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, a, b, k, j, l, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, a, b, k, l, i, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, a, b, k, l, j, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, a, b, l, i, j, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, a, b, l, i, k, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, a, b, l, j, i, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, a, b, l, j, k, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, a, b, l, k, i, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, a, b, l, k, j, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, b, a, i, j, k, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, b, a, i, j, l, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, b, a, i, k, j, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, b, a, i, k, l, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, b, a, i, l, j, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, b, a, i, l, k, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, b, a, j, i, k, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, b, a, j, i, l, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, b, a, j, k, i, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, b, a, j, k, l, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, b, a, j, l, i, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, b, a, j, l, k, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, b, a, k, i, j, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, b, a, k, i, l, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, b, a, k, j, i, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, b, a, k, j, l, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, b, a, k, l, i, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, b, a, k, l, j, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, b, a, l, i, j, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, b, a, l, i, k, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, b, a, l, j, i, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, b, a, l, j, k, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, b, a, l, k, i, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(c, d, b, a, l, k, j, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, b, c, i, j, k, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, b, c, i, j, l, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, b, c, i, k, j, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, b, c, i, k, l, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, b, c, i, l, j, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, b, c, i, l, k, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, b, c, j, i, k, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, b, c, j, i, l, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, b, c, j, k, i, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, b, c, j, k, l, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, b, c, j, l, i, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, b, c, j, l, k, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, b, c, k, i, j, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, b, c, k, i, l, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, b, c, k, j, i, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, b, c, k, j, l, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, b, c, k, l, i, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, b, c, k, l, j, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, b, c, l, i, j, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, b, c, l, i, k, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, b, c, l, j, i, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, b, c, l, j, k, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, b, c, l, k, i, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, b, c, l, k, j, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, c, b, i, j, k, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, c, b, i, j, l, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, c, b, i, k, j, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, c, b, i, k, l, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, c, b, i, l, j, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, c, b, i, l, k, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, c, b, j, i, k, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, c, b, j, i, l, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, c, b, j, k, i, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, c, b, j, k, l, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, c, b, j, l, i, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, c, b, j, l, k, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, c, b, k, i, j, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, c, b, k, i, l, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, c, b, k, j, i, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, c, b, k, j, l, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, c, b, k, l, i, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, c, b, k, l, j, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, c, b, l, i, j, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, c, b, l, i, k, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, c, b, l, j, i, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, c, b, l, j, k, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, c, b, l, k, i, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, a, c, b, l, k, j, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, a, c, i, j, k, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, a, c, i, j, l, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, a, c, i, k, j, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, a, c, i, k, l, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, a, c, i, l, j, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, a, c, i, l, k, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, a, c, j, i, k, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, a, c, j, i, l, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, a, c, j, k, i, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, a, c, j, k, l, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, a, c, j, l, i, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, a, c, j, l, k, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, a, c, k, i, j, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, a, c, k, i, l, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, a, c, k, j, i, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, a, c, k, j, l, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, a, c, k, l, i, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, a, c, k, l, j, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, a, c, l, i, j, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, a, c, l, i, k, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, a, c, l, j, i, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, a, c, l, j, k, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, a, c, l, k, i, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, a, c, l, k, j, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, c, a, i, j, k, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, c, a, i, j, l, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, c, a, i, k, j, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, c, a, i, k, l, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, c, a, i, l, j, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, c, a, i, l, k, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, c, a, j, i, k, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, c, a, j, i, l, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, c, a, j, k, i, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, c, a, j, k, l, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, c, a, j, l, i, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, c, a, j, l, k, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, c, a, k, i, j, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, c, a, k, i, l, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, c, a, k, j, i, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, c, a, k, j, l, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, c, a, k, l, i, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, c, a, k, l, j, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, c, a, l, i, j, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, c, a, l, i, k, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, c, a, l, j, i, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, c, a, l, j, k, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, c, a, l, k, i, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, b, c, a, l, k, j, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, a, b, i, j, k, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, a, b, i, j, l, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, a, b, i, k, j, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, a, b, i, k, l, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, a, b, i, l, j, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, a, b, i, l, k, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, a, b, j, i, k, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, a, b, j, i, l, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, a, b, j, k, i, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, a, b, j, k, l, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, a, b, j, l, i, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, a, b, j, l, k, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, a, b, k, i, j, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, a, b, k, i, l, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, a, b, k, j, i, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, a, b, k, j, l, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, a, b, k, l, i, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, a, b, k, l, j, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, a, b, l, i, j, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, a, b, l, i, k, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, a, b, l, j, i, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, a, b, l, j, k, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, a, b, l, k, i, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, a, b, l, k, j, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, b, a, i, j, k, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, b, a, i, j, l, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, b, a, i, k, j, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, b, a, i, k, l, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, b, a, i, l, j, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, b, a, i, l, k, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, b, a, j, i, k, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, b, a, j, i, l, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, b, a, j, k, i, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, b, a, j, k, l, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, b, a, j, l, i, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, b, a, j, l, k, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, b, a, k, i, j, l) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, b, a, k, i, l, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, b, a, k, j, i, l) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, b, a, k, j, l, i) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, b, a, k, l, i, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, b, a, k, l, j, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, b, a, l, i, j, k) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, b, a, l, i, k, j) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, b, a, l, j, i, k) = t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, b, a, l, j, k, i) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, b, a, l, k, i, j) = -1.0 * t4e(a, b, c, d, i, j, k, l)
                        t4e(d, c, b, a, l, k, j, i) = t4e(a, b, c, d, i, j, k, l)

                        resid(a, b, c, d, i, j, k, l) = val
                        resid(a, b, c, d, i, j, l, k) = -1.0 * val
                        resid(a, b, c, d, i, k, j, l) = -1.0 * val
                        resid(a, b, c, d, i, k, l, j) = val
                        resid(a, b, c, d, i, l, j, k) = val
                        resid(a, b, c, d, i, l, k, j) = -1.0 * val
                        resid(a, b, c, d, j, i, k, l) = -1.0 * val
                        resid(a, b, c, d, j, i, l, k) = val
                        resid(a, b, c, d, j, k, i, l) = val
                        resid(a, b, c, d, j, k, l, i) = -1.0 * val
                        resid(a, b, c, d, j, l, i, k) = -1.0 * val
                        resid(a, b, c, d, j, l, k, i) = val
                        resid(a, b, c, d, k, i, j, l) = val
                        resid(a, b, c, d, k, i, l, j) = -1.0 * val
                        resid(a, b, c, d, k, j, i, l) = -1.0 * val
                        resid(a, b, c, d, k, j, l, i) = val
                        resid(a, b, c, d, k, l, i, j) = val
                        resid(a, b, c, d, k, l, j, i) = -1.0 * val
                        resid(a, b, c, d, l, i, j, k) = -1.0 * val
                        resid(a, b, c, d, l, i, k, j) = val
                        resid(a, b, c, d, l, j, i, k) = val
                        resid(a, b, c, d, l, j, k, i) = -1.0 * val
                        resid(a, b, c, d, l, k, i, j) = -1.0 * val
                        resid(a, b, c, d, l, k, j, i) = val
                        resid(a, b, d, c, i, j, k, l) = -1.0 * val
                        resid(a, b, d, c, i, j, l, k) = val
                        resid(a, b, d, c, i, k, j, l) = val
                        resid(a, b, d, c, i, k, l, j) = -1.0 * val
                        resid(a, b, d, c, i, l, j, k) = -1.0 * val
                        resid(a, b, d, c, i, l, k, j) = val
                        resid(a, b, d, c, j, i, k, l) = val
                        resid(a, b, d, c, j, i, l, k) = -1.0 * val
                        resid(a, b, d, c, j, k, i, l) = -1.0 * val
                        resid(a, b, d, c, j, k, l, i) = val
                        resid(a, b, d, c, j, l, i, k) = val
                        resid(a, b, d, c, j, l, k, i) = -1.0 * val
                        resid(a, b, d, c, k, i, j, l) = -1.0 * val
                        resid(a, b, d, c, k, i, l, j) = val
                        resid(a, b, d, c, k, j, i, l) = val
                        resid(a, b, d, c, k, j, l, i) = -1.0 * val
                        resid(a, b, d, c, k, l, i, j) = -1.0 * val
                        resid(a, b, d, c, k, l, j, i) = val
                        resid(a, b, d, c, l, i, j, k) = val
                        resid(a, b, d, c, l, i, k, j) = -1.0 * val
                        resid(a, b, d, c, l, j, i, k) = -1.0 * val
                        resid(a, b, d, c, l, j, k, i) = val
                        resid(a, b, d, c, l, k, i, j) = val
                        resid(a, b, d, c, l, k, j, i) = -1.0 * val
                        resid(a, c, b, d, i, j, k, l) = -1.0 * val
                        resid(a, c, b, d, i, j, l, k) = val
                        resid(a, c, b, d, i, k, j, l) = val
                        resid(a, c, b, d, i, k, l, j) = -1.0 * val
                        resid(a, c, b, d, i, l, j, k) = -1.0 * val
                        resid(a, c, b, d, i, l, k, j) = val
                        resid(a, c, b, d, j, i, k, l) = val
                        resid(a, c, b, d, j, i, l, k) = -1.0 * val
                        resid(a, c, b, d, j, k, i, l) = -1.0 * val
                        resid(a, c, b, d, j, k, l, i) = val
                        resid(a, c, b, d, j, l, i, k) = val
                        resid(a, c, b, d, j, l, k, i) = -1.0 * val
                        resid(a, c, b, d, k, i, j, l) = -1.0 * val
                        resid(a, c, b, d, k, i, l, j) = val
                        resid(a, c, b, d, k, j, i, l) = val
                        resid(a, c, b, d, k, j, l, i) = -1.0 * val
                        resid(a, c, b, d, k, l, i, j) = -1.0 * val
                        resid(a, c, b, d, k, l, j, i) = val
                        resid(a, c, b, d, l, i, j, k) = val
                        resid(a, c, b, d, l, i, k, j) = -1.0 * val
                        resid(a, c, b, d, l, j, i, k) = -1.0 * val
                        resid(a, c, b, d, l, j, k, i) = val
                        resid(a, c, b, d, l, k, i, j) = val
                        resid(a, c, b, d, l, k, j, i) = -1.0 * val
                        resid(a, c, d, b, i, j, k, l) = val
                        resid(a, c, d, b, i, j, l, k) = -1.0 * val
                        resid(a, c, d, b, i, k, j, l) = -1.0 * val
                        resid(a, c, d, b, i, k, l, j) = val
                        resid(a, c, d, b, i, l, j, k) = val
                        resid(a, c, d, b, i, l, k, j) = -1.0 * val
                        resid(a, c, d, b, j, i, k, l) = -1.0 * val
                        resid(a, c, d, b, j, i, l, k) = val
                        resid(a, c, d, b, j, k, i, l) = val
                        resid(a, c, d, b, j, k, l, i) = -1.0 * val
                        resid(a, c, d, b, j, l, i, k) = -1.0 * val
                        resid(a, c, d, b, j, l, k, i) = val
                        resid(a, c, d, b, k, i, j, l) = val
                        resid(a, c, d, b, k, i, l, j) = -1.0 * val
                        resid(a, c, d, b, k, j, i, l) = -1.0 * val
                        resid(a, c, d, b, k, j, l, i) = val
                        resid(a, c, d, b, k, l, i, j) = val
                        resid(a, c, d, b, k, l, j, i) = -1.0 * val
                        resid(a, c, d, b, l, i, j, k) = -1.0 * val
                        resid(a, c, d, b, l, i, k, j) = val
                        resid(a, c, d, b, l, j, i, k) = val
                        resid(a, c, d, b, l, j, k, i) = -1.0 * val
                        resid(a, c, d, b, l, k, i, j) = -1.0 * val
                        resid(a, c, d, b, l, k, j, i) = val
                        resid(a, d, b, c, i, j, k, l) = val
                        resid(a, d, b, c, i, j, l, k) = -1.0 * val
                        resid(a, d, b, c, i, k, j, l) = -1.0 * val
                        resid(a, d, b, c, i, k, l, j) = val
                        resid(a, d, b, c, i, l, j, k) = val
                        resid(a, d, b, c, i, l, k, j) = -1.0 * val
                        resid(a, d, b, c, j, i, k, l) = -1.0 * val
                        resid(a, d, b, c, j, i, l, k) = val
                        resid(a, d, b, c, j, k, i, l) = val
                        resid(a, d, b, c, j, k, l, i) = -1.0 * val
                        resid(a, d, b, c, j, l, i, k) = -1.0 * val
                        resid(a, d, b, c, j, l, k, i) = val
                        resid(a, d, b, c, k, i, j, l) = val
                        resid(a, d, b, c, k, i, l, j) = -1.0 * val
                        resid(a, d, b, c, k, j, i, l) = -1.0 * val
                        resid(a, d, b, c, k, j, l, i) = val
                        resid(a, d, b, c, k, l, i, j) = val
                        resid(a, d, b, c, k, l, j, i) = -1.0 * val
                        resid(a, d, b, c, l, i, j, k) = -1.0 * val
                        resid(a, d, b, c, l, i, k, j) = val
                        resid(a, d, b, c, l, j, i, k) = val
                        resid(a, d, b, c, l, j, k, i) = -1.0 * val
                        resid(a, d, b, c, l, k, i, j) = -1.0 * val
                        resid(a, d, b, c, l, k, j, i) = val
                        resid(a, d, c, b, i, j, k, l) = -1.0 * val
                        resid(a, d, c, b, i, j, l, k) = val
                        resid(a, d, c, b, i, k, j, l) = val
                        resid(a, d, c, b, i, k, l, j) = -1.0 * val
                        resid(a, d, c, b, i, l, j, k) = -1.0 * val
                        resid(a, d, c, b, i, l, k, j) = val
                        resid(a, d, c, b, j, i, k, l) = val
                        resid(a, d, c, b, j, i, l, k) = -1.0 * val
                        resid(a, d, c, b, j, k, i, l) = -1.0 * val
                        resid(a, d, c, b, j, k, l, i) = val
                        resid(a, d, c, b, j, l, i, k) = val
                        resid(a, d, c, b, j, l, k, i) = -1.0 * val
                        resid(a, d, c, b, k, i, j, l) = -1.0 * val
                        resid(a, d, c, b, k, i, l, j) = val
                        resid(a, d, c, b, k, j, i, l) = val
                        resid(a, d, c, b, k, j, l, i) = -1.0 * val
                        resid(a, d, c, b, k, l, i, j) = -1.0 * val
                        resid(a, d, c, b, k, l, j, i) = val
                        resid(a, d, c, b, l, i, j, k) = val
                        resid(a, d, c, b, l, i, k, j) = -1.0 * val
                        resid(a, d, c, b, l, j, i, k) = -1.0 * val
                        resid(a, d, c, b, l, j, k, i) = val
                        resid(a, d, c, b, l, k, i, j) = val
                        resid(a, d, c, b, l, k, j, i) = -1.0 * val
                        resid(b, a, c, d, i, j, k, l) = -1.0 * val
                        resid(b, a, c, d, i, j, l, k) = val
                        resid(b, a, c, d, i, k, j, l) = val
                        resid(b, a, c, d, i, k, l, j) = -1.0 * val
                        resid(b, a, c, d, i, l, j, k) = -1.0 * val
                        resid(b, a, c, d, i, l, k, j) = val
                        resid(b, a, c, d, j, i, k, l) = val
                        resid(b, a, c, d, j, i, l, k) = -1.0 * val
                        resid(b, a, c, d, j, k, i, l) = -1.0 * val
                        resid(b, a, c, d, j, k, l, i) = val
                        resid(b, a, c, d, j, l, i, k) = val
                        resid(b, a, c, d, j, l, k, i) = -1.0 * val
                        resid(b, a, c, d, k, i, j, l) = -1.0 * val
                        resid(b, a, c, d, k, i, l, j) = val
                        resid(b, a, c, d, k, j, i, l) = val
                        resid(b, a, c, d, k, j, l, i) = -1.0 * val
                        resid(b, a, c, d, k, l, i, j) = -1.0 * val
                        resid(b, a, c, d, k, l, j, i) = val
                        resid(b, a, c, d, l, i, j, k) = val
                        resid(b, a, c, d, l, i, k, j) = -1.0 * val
                        resid(b, a, c, d, l, j, i, k) = -1.0 * val
                        resid(b, a, c, d, l, j, k, i) = val
                        resid(b, a, c, d, l, k, i, j) = val
                        resid(b, a, c, d, l, k, j, i) = -1.0 * val
                        resid(b, a, d, c, i, j, k, l) = val
                        resid(b, a, d, c, i, j, l, k) = -1.0 * val
                        resid(b, a, d, c, i, k, j, l) = -1.0 * val
                        resid(b, a, d, c, i, k, l, j) = val
                        resid(b, a, d, c, i, l, j, k) = val
                        resid(b, a, d, c, i, l, k, j) = -1.0 * val
                        resid(b, a, d, c, j, i, k, l) = -1.0 * val
                        resid(b, a, d, c, j, i, l, k) = val
                        resid(b, a, d, c, j, k, i, l) = val
                        resid(b, a, d, c, j, k, l, i) = -1.0 * val
                        resid(b, a, d, c, j, l, i, k) = -1.0 * val
                        resid(b, a, d, c, j, l, k, i) = val
                        resid(b, a, d, c, k, i, j, l) = val
                        resid(b, a, d, c, k, i, l, j) = -1.0 * val
                        resid(b, a, d, c, k, j, i, l) = -1.0 * val
                        resid(b, a, d, c, k, j, l, i) = val
                        resid(b, a, d, c, k, l, i, j) = val
                        resid(b, a, d, c, k, l, j, i) = -1.0 * val
                        resid(b, a, d, c, l, i, j, k) = -1.0 * val
                        resid(b, a, d, c, l, i, k, j) = val
                        resid(b, a, d, c, l, j, i, k) = val
                        resid(b, a, d, c, l, j, k, i) = -1.0 * val
                        resid(b, a, d, c, l, k, i, j) = -1.0 * val
                        resid(b, a, d, c, l, k, j, i) = val
                        resid(b, c, a, d, i, j, k, l) = val
                        resid(b, c, a, d, i, j, l, k) = -1.0 * val
                        resid(b, c, a, d, i, k, j, l) = -1.0 * val
                        resid(b, c, a, d, i, k, l, j) = val
                        resid(b, c, a, d, i, l, j, k) = val
                        resid(b, c, a, d, i, l, k, j) = -1.0 * val
                        resid(b, c, a, d, j, i, k, l) = -1.0 * val
                        resid(b, c, a, d, j, i, l, k) = val
                        resid(b, c, a, d, j, k, i, l) = val
                        resid(b, c, a, d, j, k, l, i) = -1.0 * val
                        resid(b, c, a, d, j, l, i, k) = -1.0 * val
                        resid(b, c, a, d, j, l, k, i) = val
                        resid(b, c, a, d, k, i, j, l) = val
                        resid(b, c, a, d, k, i, l, j) = -1.0 * val
                        resid(b, c, a, d, k, j, i, l) = -1.0 * val
                        resid(b, c, a, d, k, j, l, i) = val
                        resid(b, c, a, d, k, l, i, j) = val
                        resid(b, c, a, d, k, l, j, i) = -1.0 * val
                        resid(b, c, a, d, l, i, j, k) = -1.0 * val
                        resid(b, c, a, d, l, i, k, j) = val
                        resid(b, c, a, d, l, j, i, k) = val
                        resid(b, c, a, d, l, j, k, i) = -1.0 * val
                        resid(b, c, a, d, l, k, i, j) = -1.0 * val
                        resid(b, c, a, d, l, k, j, i) = val
                        resid(b, c, d, a, i, j, k, l) = -1.0 * val
                        resid(b, c, d, a, i, j, l, k) = val
                        resid(b, c, d, a, i, k, j, l) = val
                        resid(b, c, d, a, i, k, l, j) = -1.0 * val
                        resid(b, c, d, a, i, l, j, k) = -1.0 * val
                        resid(b, c, d, a, i, l, k, j) = val
                        resid(b, c, d, a, j, i, k, l) = val
                        resid(b, c, d, a, j, i, l, k) = -1.0 * val
                        resid(b, c, d, a, j, k, i, l) = -1.0 * val
                        resid(b, c, d, a, j, k, l, i) = val
                        resid(b, c, d, a, j, l, i, k) = val
                        resid(b, c, d, a, j, l, k, i) = -1.0 * val
                        resid(b, c, d, a, k, i, j, l) = -1.0 * val
                        resid(b, c, d, a, k, i, l, j) = val
                        resid(b, c, d, a, k, j, i, l) = val
                        resid(b, c, d, a, k, j, l, i) = -1.0 * val
                        resid(b, c, d, a, k, l, i, j) = -1.0 * val
                        resid(b, c, d, a, k, l, j, i) = val
                        resid(b, c, d, a, l, i, j, k) = val
                        resid(b, c, d, a, l, i, k, j) = -1.0 * val
                        resid(b, c, d, a, l, j, i, k) = -1.0 * val
                        resid(b, c, d, a, l, j, k, i) = val
                        resid(b, c, d, a, l, k, i, j) = val
                        resid(b, c, d, a, l, k, j, i) = -1.0 * val
                        resid(b, d, a, c, i, j, k, l) = -1.0 * val
                        resid(b, d, a, c, i, j, l, k) = val
                        resid(b, d, a, c, i, k, j, l) = val
                        resid(b, d, a, c, i, k, l, j) = -1.0 * val
                        resid(b, d, a, c, i, l, j, k) = -1.0 * val
                        resid(b, d, a, c, i, l, k, j) = val
                        resid(b, d, a, c, j, i, k, l) = val
                        resid(b, d, a, c, j, i, l, k) = -1.0 * val
                        resid(b, d, a, c, j, k, i, l) = -1.0 * val
                        resid(b, d, a, c, j, k, l, i) = val
                        resid(b, d, a, c, j, l, i, k) = val
                        resid(b, d, a, c, j, l, k, i) = -1.0 * val
                        resid(b, d, a, c, k, i, j, l) = -1.0 * val
                        resid(b, d, a, c, k, i, l, j) = val
                        resid(b, d, a, c, k, j, i, l) = val
                        resid(b, d, a, c, k, j, l, i) = -1.0 * val
                        resid(b, d, a, c, k, l, i, j) = -1.0 * val
                        resid(b, d, a, c, k, l, j, i) = val
                        resid(b, d, a, c, l, i, j, k) = val
                        resid(b, d, a, c, l, i, k, j) = -1.0 * val
                        resid(b, d, a, c, l, j, i, k) = -1.0 * val
                        resid(b, d, a, c, l, j, k, i) = val
                        resid(b, d, a, c, l, k, i, j) = val
                        resid(b, d, a, c, l, k, j, i) = -1.0 * val
                        resid(b, d, c, a, i, j, k, l) = val
                        resid(b, d, c, a, i, j, l, k) = -1.0 * val
                        resid(b, d, c, a, i, k, j, l) = -1.0 * val
                        resid(b, d, c, a, i, k, l, j) = val
                        resid(b, d, c, a, i, l, j, k) = val
                        resid(b, d, c, a, i, l, k, j) = -1.0 * val
                        resid(b, d, c, a, j, i, k, l) = -1.0 * val
                        resid(b, d, c, a, j, i, l, k) = val
                        resid(b, d, c, a, j, k, i, l) = val
                        resid(b, d, c, a, j, k, l, i) = -1.0 * val
                        resid(b, d, c, a, j, l, i, k) = -1.0 * val
                        resid(b, d, c, a, j, l, k, i) = val
                        resid(b, d, c, a, k, i, j, l) = val
                        resid(b, d, c, a, k, i, l, j) = -1.0 * val
                        resid(b, d, c, a, k, j, i, l) = -1.0 * val
                        resid(b, d, c, a, k, j, l, i) = val
                        resid(b, d, c, a, k, l, i, j) = val
                        resid(b, d, c, a, k, l, j, i) = -1.0 * val
                        resid(b, d, c, a, l, i, j, k) = -1.0 * val
                        resid(b, d, c, a, l, i, k, j) = val
                        resid(b, d, c, a, l, j, i, k) = val
                        resid(b, d, c, a, l, j, k, i) = -1.0 * val
                        resid(b, d, c, a, l, k, i, j) = -1.0 * val
                        resid(b, d, c, a, l, k, j, i) = val
                        resid(c, a, b, d, i, j, k, l) = val
                        resid(c, a, b, d, i, j, l, k) = -1.0 * val
                        resid(c, a, b, d, i, k, j, l) = -1.0 * val
                        resid(c, a, b, d, i, k, l, j) = val
                        resid(c, a, b, d, i, l, j, k) = val
                        resid(c, a, b, d, i, l, k, j) = -1.0 * val
                        resid(c, a, b, d, j, i, k, l) = -1.0 * val
                        resid(c, a, b, d, j, i, l, k) = val
                        resid(c, a, b, d, j, k, i, l) = val
                        resid(c, a, b, d, j, k, l, i) = -1.0 * val
                        resid(c, a, b, d, j, l, i, k) = -1.0 * val
                        resid(c, a, b, d, j, l, k, i) = val
                        resid(c, a, b, d, k, i, j, l) = val
                        resid(c, a, b, d, k, i, l, j) = -1.0 * val
                        resid(c, a, b, d, k, j, i, l) = -1.0 * val
                        resid(c, a, b, d, k, j, l, i) = val
                        resid(c, a, b, d, k, l, i, j) = val
                        resid(c, a, b, d, k, l, j, i) = -1.0 * val
                        resid(c, a, b, d, l, i, j, k) = -1.0 * val
                        resid(c, a, b, d, l, i, k, j) = val
                        resid(c, a, b, d, l, j, i, k) = val
                        resid(c, a, b, d, l, j, k, i) = -1.0 * val
                        resid(c, a, b, d, l, k, i, j) = -1.0 * val
                        resid(c, a, b, d, l, k, j, i) = val
                        resid(c, a, d, b, i, j, k, l) = -1.0 * val
                        resid(c, a, d, b, i, j, l, k) = val
                        resid(c, a, d, b, i, k, j, l) = val
                        resid(c, a, d, b, i, k, l, j) = -1.0 * val
                        resid(c, a, d, b, i, l, j, k) = -1.0 * val
                        resid(c, a, d, b, i, l, k, j) = val
                        resid(c, a, d, b, j, i, k, l) = val
                        resid(c, a, d, b, j, i, l, k) = -1.0 * val
                        resid(c, a, d, b, j, k, i, l) = -1.0 * val
                        resid(c, a, d, b, j, k, l, i) = val
                        resid(c, a, d, b, j, l, i, k) = val
                        resid(c, a, d, b, j, l, k, i) = -1.0 * val
                        resid(c, a, d, b, k, i, j, l) = -1.0 * val
                        resid(c, a, d, b, k, i, l, j) = val
                        resid(c, a, d, b, k, j, i, l) = val
                        resid(c, a, d, b, k, j, l, i) = -1.0 * val
                        resid(c, a, d, b, k, l, i, j) = -1.0 * val
                        resid(c, a, d, b, k, l, j, i) = val
                        resid(c, a, d, b, l, i, j, k) = val
                        resid(c, a, d, b, l, i, k, j) = -1.0 * val
                        resid(c, a, d, b, l, j, i, k) = -1.0 * val
                        resid(c, a, d, b, l, j, k, i) = val
                        resid(c, a, d, b, l, k, i, j) = val
                        resid(c, a, d, b, l, k, j, i) = -1.0 * val
                        resid(c, b, a, d, i, j, k, l) = -1.0 * val
                        resid(c, b, a, d, i, j, l, k) = val
                        resid(c, b, a, d, i, k, j, l) = val
                        resid(c, b, a, d, i, k, l, j) = -1.0 * val
                        resid(c, b, a, d, i, l, j, k) = -1.0 * val
                        resid(c, b, a, d, i, l, k, j) = val
                        resid(c, b, a, d, j, i, k, l) = val
                        resid(c, b, a, d, j, i, l, k) = -1.0 * val
                        resid(c, b, a, d, j, k, i, l) = -1.0 * val
                        resid(c, b, a, d, j, k, l, i) = val
                        resid(c, b, a, d, j, l, i, k) = val
                        resid(c, b, a, d, j, l, k, i) = -1.0 * val
                        resid(c, b, a, d, k, i, j, l) = -1.0 * val
                        resid(c, b, a, d, k, i, l, j) = val
                        resid(c, b, a, d, k, j, i, l) = val
                        resid(c, b, a, d, k, j, l, i) = -1.0 * val
                        resid(c, b, a, d, k, l, i, j) = -1.0 * val
                        resid(c, b, a, d, k, l, j, i) = val
                        resid(c, b, a, d, l, i, j, k) = val
                        resid(c, b, a, d, l, i, k, j) = -1.0 * val
                        resid(c, b, a, d, l, j, i, k) = -1.0 * val
                        resid(c, b, a, d, l, j, k, i) = val
                        resid(c, b, a, d, l, k, i, j) = val
                        resid(c, b, a, d, l, k, j, i) = -1.0 * val
                        resid(c, b, d, a, i, j, k, l) = val
                        resid(c, b, d, a, i, j, l, k) = -1.0 * val
                        resid(c, b, d, a, i, k, j, l) = -1.0 * val
                        resid(c, b, d, a, i, k, l, j) = val
                        resid(c, b, d, a, i, l, j, k) = val
                        resid(c, b, d, a, i, l, k, j) = -1.0 * val
                        resid(c, b, d, a, j, i, k, l) = -1.0 * val
                        resid(c, b, d, a, j, i, l, k) = val
                        resid(c, b, d, a, j, k, i, l) = val
                        resid(c, b, d, a, j, k, l, i) = -1.0 * val
                        resid(c, b, d, a, j, l, i, k) = -1.0 * val
                        resid(c, b, d, a, j, l, k, i) = val
                        resid(c, b, d, a, k, i, j, l) = val
                        resid(c, b, d, a, k, i, l, j) = -1.0 * val
                        resid(c, b, d, a, k, j, i, l) = -1.0 * val
                        resid(c, b, d, a, k, j, l, i) = val
                        resid(c, b, d, a, k, l, i, j) = val
                        resid(c, b, d, a, k, l, j, i) = -1.0 * val
                        resid(c, b, d, a, l, i, j, k) = -1.0 * val
                        resid(c, b, d, a, l, i, k, j) = val
                        resid(c, b, d, a, l, j, i, k) = val
                        resid(c, b, d, a, l, j, k, i) = -1.0 * val
                        resid(c, b, d, a, l, k, i, j) = -1.0 * val
                        resid(c, b, d, a, l, k, j, i) = val
                        resid(c, d, a, b, i, j, k, l) = val
                        resid(c, d, a, b, i, j, l, k) = -1.0 * val
                        resid(c, d, a, b, i, k, j, l) = -1.0 * val
                        resid(c, d, a, b, i, k, l, j) = val
                        resid(c, d, a, b, i, l, j, k) = val
                        resid(c, d, a, b, i, l, k, j) = -1.0 * val
                        resid(c, d, a, b, j, i, k, l) = -1.0 * val
                        resid(c, d, a, b, j, i, l, k) = val
                        resid(c, d, a, b, j, k, i, l) = val
                        resid(c, d, a, b, j, k, l, i) = -1.0 * val
                        resid(c, d, a, b, j, l, i, k) = -1.0 * val
                        resid(c, d, a, b, j, l, k, i) = val
                        resid(c, d, a, b, k, i, j, l) = val
                        resid(c, d, a, b, k, i, l, j) = -1.0 * val
                        resid(c, d, a, b, k, j, i, l) = -1.0 * val
                        resid(c, d, a, b, k, j, l, i) = val
                        resid(c, d, a, b, k, l, i, j) = val
                        resid(c, d, a, b, k, l, j, i) = -1.0 * val
                        resid(c, d, a, b, l, i, j, k) = -1.0 * val
                        resid(c, d, a, b, l, i, k, j) = val
                        resid(c, d, a, b, l, j, i, k) = val
                        resid(c, d, a, b, l, j, k, i) = -1.0 * val
                        resid(c, d, a, b, l, k, i, j) = -1.0 * val
                        resid(c, d, a, b, l, k, j, i) = val
                        resid(c, d, b, a, i, j, k, l) = -1.0 * val
                        resid(c, d, b, a, i, j, l, k) = val
                        resid(c, d, b, a, i, k, j, l) = val
                        resid(c, d, b, a, i, k, l, j) = -1.0 * val
                        resid(c, d, b, a, i, l, j, k) = -1.0 * val
                        resid(c, d, b, a, i, l, k, j) = val
                        resid(c, d, b, a, j, i, k, l) = val
                        resid(c, d, b, a, j, i, l, k) = -1.0 * val
                        resid(c, d, b, a, j, k, i, l) = -1.0 * val
                        resid(c, d, b, a, j, k, l, i) = val
                        resid(c, d, b, a, j, l, i, k) = val
                        resid(c, d, b, a, j, l, k, i) = -1.0 * val
                        resid(c, d, b, a, k, i, j, l) = -1.0 * val
                        resid(c, d, b, a, k, i, l, j) = val
                        resid(c, d, b, a, k, j, i, l) = val
                        resid(c, d, b, a, k, j, l, i) = -1.0 * val
                        resid(c, d, b, a, k, l, i, j) = -1.0 * val
                        resid(c, d, b, a, k, l, j, i) = val
                        resid(c, d, b, a, l, i, j, k) = val
                        resid(c, d, b, a, l, i, k, j) = -1.0 * val
                        resid(c, d, b, a, l, j, i, k) = -1.0 * val
                        resid(c, d, b, a, l, j, k, i) = val
                        resid(c, d, b, a, l, k, i, j) = val
                        resid(c, d, b, a, l, k, j, i) = -1.0 * val
                        resid(d, a, b, c, i, j, k, l) = -1.0 * val
                        resid(d, a, b, c, i, j, l, k) = val
                        resid(d, a, b, c, i, k, j, l) = val
                        resid(d, a, b, c, i, k, l, j) = -1.0 * val
                        resid(d, a, b, c, i, l, j, k) = -1.0 * val
                        resid(d, a, b, c, i, l, k, j) = val
                        resid(d, a, b, c, j, i, k, l) = val
                        resid(d, a, b, c, j, i, l, k) = -1.0 * val
                        resid(d, a, b, c, j, k, i, l) = -1.0 * val
                        resid(d, a, b, c, j, k, l, i) = val
                        resid(d, a, b, c, j, l, i, k) = val
                        resid(d, a, b, c, j, l, k, i) = -1.0 * val
                        resid(d, a, b, c, k, i, j, l) = -1.0 * val
                        resid(d, a, b, c, k, i, l, j) = val
                        resid(d, a, b, c, k, j, i, l) = val
                        resid(d, a, b, c, k, j, l, i) = -1.0 * val
                        resid(d, a, b, c, k, l, i, j) = -1.0 * val
                        resid(d, a, b, c, k, l, j, i) = val
                        resid(d, a, b, c, l, i, j, k) = val
                        resid(d, a, b, c, l, i, k, j) = -1.0 * val
                        resid(d, a, b, c, l, j, i, k) = -1.0 * val
                        resid(d, a, b, c, l, j, k, i) = val
                        resid(d, a, b, c, l, k, i, j) = val
                        resid(d, a, b, c, l, k, j, i) = -1.0 * val
                        resid(d, a, c, b, i, j, k, l) = val
                        resid(d, a, c, b, i, j, l, k) = -1.0 * val
                        resid(d, a, c, b, i, k, j, l) = -1.0 * val
                        resid(d, a, c, b, i, k, l, j) = val
                        resid(d, a, c, b, i, l, j, k) = val
                        resid(d, a, c, b, i, l, k, j) = -1.0 * val
                        resid(d, a, c, b, j, i, k, l) = -1.0 * val
                        resid(d, a, c, b, j, i, l, k) = val
                        resid(d, a, c, b, j, k, i, l) = val
                        resid(d, a, c, b, j, k, l, i) = -1.0 * val
                        resid(d, a, c, b, j, l, i, k) = -1.0 * val
                        resid(d, a, c, b, j, l, k, i) = val
                        resid(d, a, c, b, k, i, j, l) = val
                        resid(d, a, c, b, k, i, l, j) = -1.0 * val
                        resid(d, a, c, b, k, j, i, l) = -1.0 * val
                        resid(d, a, c, b, k, j, l, i) = val
                        resid(d, a, c, b, k, l, i, j) = val
                        resid(d, a, c, b, k, l, j, i) = -1.0 * val
                        resid(d, a, c, b, l, i, j, k) = -1.0 * val
                        resid(d, a, c, b, l, i, k, j) = val
                        resid(d, a, c, b, l, j, i, k) = val
                        resid(d, a, c, b, l, j, k, i) = -1.0 * val
                        resid(d, a, c, b, l, k, i, j) = -1.0 * val
                        resid(d, a, c, b, l, k, j, i) = val
                        resid(d, b, a, c, i, j, k, l) = val
                        resid(d, b, a, c, i, j, l, k) = -1.0 * val
                        resid(d, b, a, c, i, k, j, l) = -1.0 * val
                        resid(d, b, a, c, i, k, l, j) = val
                        resid(d, b, a, c, i, l, j, k) = val
                        resid(d, b, a, c, i, l, k, j) = -1.0 * val
                        resid(d, b, a, c, j, i, k, l) = -1.0 * val
                        resid(d, b, a, c, j, i, l, k) = val
                        resid(d, b, a, c, j, k, i, l) = val
                        resid(d, b, a, c, j, k, l, i) = -1.0 * val
                        resid(d, b, a, c, j, l, i, k) = -1.0 * val
                        resid(d, b, a, c, j, l, k, i) = val
                        resid(d, b, a, c, k, i, j, l) = val
                        resid(d, b, a, c, k, i, l, j) = -1.0 * val
                        resid(d, b, a, c, k, j, i, l) = -1.0 * val
                        resid(d, b, a, c, k, j, l, i) = val
                        resid(d, b, a, c, k, l, i, j) = val
                        resid(d, b, a, c, k, l, j, i) = -1.0 * val
                        resid(d, b, a, c, l, i, j, k) = -1.0 * val
                        resid(d, b, a, c, l, i, k, j) = val
                        resid(d, b, a, c, l, j, i, k) = val
                        resid(d, b, a, c, l, j, k, i) = -1.0 * val
                        resid(d, b, a, c, l, k, i, j) = -1.0 * val
                        resid(d, b, a, c, l, k, j, i) = val
                        resid(d, b, c, a, i, j, k, l) = -1.0 * val
                        resid(d, b, c, a, i, j, l, k) = val
                        resid(d, b, c, a, i, k, j, l) = val
                        resid(d, b, c, a, i, k, l, j) = -1.0 * val
                        resid(d, b, c, a, i, l, j, k) = -1.0 * val
                        resid(d, b, c, a, i, l, k, j) = val
                        resid(d, b, c, a, j, i, k, l) = val
                        resid(d, b, c, a, j, i, l, k) = -1.0 * val
                        resid(d, b, c, a, j, k, i, l) = -1.0 * val
                        resid(d, b, c, a, j, k, l, i) = val
                        resid(d, b, c, a, j, l, i, k) = val
                        resid(d, b, c, a, j, l, k, i) = -1.0 * val
                        resid(d, b, c, a, k, i, j, l) = -1.0 * val
                        resid(d, b, c, a, k, i, l, j) = val
                        resid(d, b, c, a, k, j, i, l) = val
                        resid(d, b, c, a, k, j, l, i) = -1.0 * val
                        resid(d, b, c, a, k, l, i, j) = -1.0 * val
                        resid(d, b, c, a, k, l, j, i) = val
                        resid(d, b, c, a, l, i, j, k) = val
                        resid(d, b, c, a, l, i, k, j) = -1.0 * val
                        resid(d, b, c, a, l, j, i, k) = -1.0 * val
                        resid(d, b, c, a, l, j, k, i) = val
                        resid(d, b, c, a, l, k, i, j) = val
                        resid(d, b, c, a, l, k, j, i) = -1.0 * val
                        resid(d, c, a, b, i, j, k, l) = -1.0 * val
                        resid(d, c, a, b, i, j, l, k) = val
                        resid(d, c, a, b, i, k, j, l) = val
                        resid(d, c, a, b, i, k, l, j) = -1.0 * val
                        resid(d, c, a, b, i, l, j, k) = -1.0 * val
                        resid(d, c, a, b, i, l, k, j) = val
                        resid(d, c, a, b, j, i, k, l) = val
                        resid(d, c, a, b, j, i, l, k) = -1.0 * val
                        resid(d, c, a, b, j, k, i, l) = -1.0 * val
                        resid(d, c, a, b, j, k, l, i) = val
                        resid(d, c, a, b, j, l, i, k) = val
                        resid(d, c, a, b, j, l, k, i) = -1.0 * val
                        resid(d, c, a, b, k, i, j, l) = -1.0 * val
                        resid(d, c, a, b, k, i, l, j) = val
                        resid(d, c, a, b, k, j, i, l) = val
                        resid(d, c, a, b, k, j, l, i) = -1.0 * val
                        resid(d, c, a, b, k, l, i, j) = -1.0 * val
                        resid(d, c, a, b, k, l, j, i) = val
                        resid(d, c, a, b, l, i, j, k) = val
                        resid(d, c, a, b, l, i, k, j) = -1.0 * val
                        resid(d, c, a, b, l, j, i, k) = -1.0 * val
                        resid(d, c, a, b, l, j, k, i) = val
                        resid(d, c, a, b, l, k, i, j) = val
                        resid(d, c, a, b, l, k, j, i) = -1.0 * val
                        resid(d, c, b, a, i, j, k, l) = val
                        resid(d, c, b, a, i, j, l, k) = -1.0 * val
                        resid(d, c, b, a, i, k, j, l) = -1.0 * val
                        resid(d, c, b, a, i, k, l, j) = val
                        resid(d, c, b, a, i, l, j, k) = val
                        resid(d, c, b, a, i, l, k, j) = -1.0 * val
                        resid(d, c, b, a, j, i, k, l) = -1.0 * val
                        resid(d, c, b, a, j, i, l, k) = val
                        resid(d, c, b, a, j, k, i, l) = val
                        resid(d, c, b, a, j, k, l, i) = -1.0 * val
                        resid(d, c, b, a, j, l, i, k) = -1.0 * val
                        resid(d, c, b, a, j, l, k, i) = val
                        resid(d, c, b, a, k, i, j, l) = val
                        resid(d, c, b, a, k, i, l, j) = -1.0 * val
                        resid(d, c, b, a, k, j, i, l) = -1.0 * val
                        resid(d, c, b, a, k, j, l, i) = val
                        resid(d, c, b, a, k, l, i, j) = val
                        resid(d, c, b, a, k, l, j, i) = -1.0 * val
                        resid(d, c, b, a, l, i, j, k) = -1.0 * val
                        resid(d, c, b, a, l, i, k, j) = val
                        resid(d, c, b, a, l, j, i, k) = val
                        resid(d, c, b, a, l, j, k, i) = -1.0 * val
                        resid(d, c, b, a, l, k, i, j) = -1.0 * val
                        resid(d, c, b, a, l, k, j, i) = val

                        end do
                     end do
                  end do
               end do
            end do
         end do
      end do
   end do

end subroutine update_t4e


end module cc_loops_t4
