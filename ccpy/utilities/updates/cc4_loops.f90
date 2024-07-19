module cc4_loops

    implicit none

    contains

subroutine update_t4a(t4a, &
                      fA_oo, fA_vv, &
                      noa, nua)

      integer, intent(in)  :: noa, nua
      real(8), intent(in)  :: fA_oo(1:noa, 1:noa), &
                              fA_vv(1:nua, 1:nua)

      real(8), intent(inout) :: t4a(1:nua, 1:nua, 1:nua, 1:nua, 1:noa, 1:noa, 1:noa, 1:noa)
      !f2py intent(in, out)  :: t4a(0:nua-1, 0:nua-1, 0:nua-1, 0:nua-1, 0:noa-1, 0:noa-1, 0:noa-1, 0:noa-1)

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
                        +t4a(a, b, c, d, i, j, k, l)&
                        -1.0 * t4a(a, b, c, d, i, j, l, k)&
                        -1.0 * t4a(a, b, c, d, i, k, j, l)&
                        +t4a(a, b, c, d, i, k, l, j)&
                        +t4a(a, b, c, d, i, l, j, k)&
                        -1.0 * t4a(a, b, c, d, i, l, k, j)&
                        -1.0 * t4a(a, b, c, d, j, i, k, l)&
                        +t4a(a, b, c, d, j, i, l, k)&
                        +t4a(a, b, c, d, j, k, i, l)&
                        -1.0 * t4a(a, b, c, d, j, k, l, i)&
                        -1.0 * t4a(a, b, c, d, j, l, i, k)&
                        +t4a(a, b, c, d, j, l, k, i)&
                        +t4a(a, b, c, d, k, i, j, l)&
                        -1.0 * t4a(a, b, c, d, k, i, l, j)&
                        -1.0 * t4a(a, b, c, d, k, j, i, l)&
                        +t4a(a, b, c, d, k, j, l, i)&
                        +t4a(a, b, c, d, k, l, i, j)&
                        -1.0 * t4a(a, b, c, d, k, l, j, i)&
                        -1.0 * t4a(a, b, c, d, l, i, j, k)&
                        +t4a(a, b, c, d, l, i, k, j)&
                        +t4a(a, b, c, d, l, j, i, k)&
                        -1.0 * t4a(a, b, c, d, l, j, k, i)&
                        -1.0 * t4a(a, b, c, d, l, k, i, j)&
                        +t4a(a, b, c, d, l, k, j, i)&
                        -1.0 * t4a(a, b, d, c, i, j, k, l)&
                        +t4a(a, b, d, c, i, j, l, k)&
                        +t4a(a, b, d, c, i, k, j, l)&
                        -1.0 * t4a(a, b, d, c, i, k, l, j)&
                        -1.0 * t4a(a, b, d, c, i, l, j, k)&
                        +t4a(a, b, d, c, i, l, k, j)&
                        +t4a(a, b, d, c, j, i, k, l)&
                        -1.0 * t4a(a, b, d, c, j, i, l, k)&
                        -1.0 * t4a(a, b, d, c, j, k, i, l)&
                        +t4a(a, b, d, c, j, k, l, i)&
                        +t4a(a, b, d, c, j, l, i, k)&
                        -1.0 * t4a(a, b, d, c, j, l, k, i)&
                        -1.0 * t4a(a, b, d, c, k, i, j, l)&
                        +t4a(a, b, d, c, k, i, l, j)&
                        +t4a(a, b, d, c, k, j, i, l)&
                        -1.0 * t4a(a, b, d, c, k, j, l, i)&
                        -1.0 * t4a(a, b, d, c, k, l, i, j)&
                        +t4a(a, b, d, c, k, l, j, i)&
                        +t4a(a, b, d, c, l, i, j, k)&
                        -1.0 * t4a(a, b, d, c, l, i, k, j)&
                        -1.0 * t4a(a, b, d, c, l, j, i, k)&
                        +t4a(a, b, d, c, l, j, k, i)&
                        +t4a(a, b, d, c, l, k, i, j)&
                        -1.0 * t4a(a, b, d, c, l, k, j, i)&
                        -1.0 * t4a(a, c, b, d, i, j, k, l)&
                        +t4a(a, c, b, d, i, j, l, k)&
                        +t4a(a, c, b, d, i, k, j, l)&
                        -1.0 * t4a(a, c, b, d, i, k, l, j)&
                        -1.0 * t4a(a, c, b, d, i, l, j, k)&
                        +t4a(a, c, b, d, i, l, k, j)&
                        +t4a(a, c, b, d, j, i, k, l)&
                        -1.0 * t4a(a, c, b, d, j, i, l, k)&
                        -1.0 * t4a(a, c, b, d, j, k, i, l)&
                        +t4a(a, c, b, d, j, k, l, i)&
                        +t4a(a, c, b, d, j, l, i, k)&
                        -1.0 * t4a(a, c, b, d, j, l, k, i)&
                        -1.0 * t4a(a, c, b, d, k, i, j, l)&
                        +t4a(a, c, b, d, k, i, l, j)&
                        +t4a(a, c, b, d, k, j, i, l)&
                        -1.0 * t4a(a, c, b, d, k, j, l, i)&
                        -1.0 * t4a(a, c, b, d, k, l, i, j)&
                        +t4a(a, c, b, d, k, l, j, i)&
                        +t4a(a, c, b, d, l, i, j, k)&
                        -1.0 * t4a(a, c, b, d, l, i, k, j)&
                        -1.0 * t4a(a, c, b, d, l, j, i, k)&
                        +t4a(a, c, b, d, l, j, k, i)&
                        +t4a(a, c, b, d, l, k, i, j)&
                        -1.0 * t4a(a, c, b, d, l, k, j, i)&
                        +t4a(a, c, d, b, i, j, k, l)&
                        -1.0 * t4a(a, c, d, b, i, j, l, k)&
                        -1.0 * t4a(a, c, d, b, i, k, j, l)&
                        +t4a(a, c, d, b, i, k, l, j)&
                        +t4a(a, c, d, b, i, l, j, k)&
                        -1.0 * t4a(a, c, d, b, i, l, k, j)&
                        -1.0 * t4a(a, c, d, b, j, i, k, l)&
                        +t4a(a, c, d, b, j, i, l, k)&
                        +t4a(a, c, d, b, j, k, i, l)&
                        -1.0 * t4a(a, c, d, b, j, k, l, i)&
                        -1.0 * t4a(a, c, d, b, j, l, i, k)&
                        +t4a(a, c, d, b, j, l, k, i)&
                        +t4a(a, c, d, b, k, i, j, l)&
                        -1.0 * t4a(a, c, d, b, k, i, l, j)&
                        -1.0 * t4a(a, c, d, b, k, j, i, l)&
                        +t4a(a, c, d, b, k, j, l, i)&
                        +t4a(a, c, d, b, k, l, i, j)&
                        -1.0 * t4a(a, c, d, b, k, l, j, i)&
                        -1.0 * t4a(a, c, d, b, l, i, j, k)&
                        +t4a(a, c, d, b, l, i, k, j)&
                        +t4a(a, c, d, b, l, j, i, k)&
                        -1.0 * t4a(a, c, d, b, l, j, k, i)&
                        -1.0 * t4a(a, c, d, b, l, k, i, j)&
                        +t4a(a, c, d, b, l, k, j, i)&
                        +t4a(a, d, b, c, i, j, k, l)&
                        -1.0 * t4a(a, d, b, c, i, j, l, k)&
                        -1.0 * t4a(a, d, b, c, i, k, j, l)&
                        +t4a(a, d, b, c, i, k, l, j)&
                        +t4a(a, d, b, c, i, l, j, k)&
                        -1.0 * t4a(a, d, b, c, i, l, k, j)&
                        -1.0 * t4a(a, d, b, c, j, i, k, l)&
                        +t4a(a, d, b, c, j, i, l, k)&
                        +t4a(a, d, b, c, j, k, i, l)&
                        -1.0 * t4a(a, d, b, c, j, k, l, i)&
                        -1.0 * t4a(a, d, b, c, j, l, i, k)&
                        +t4a(a, d, b, c, j, l, k, i)&
                        +t4a(a, d, b, c, k, i, j, l)&
                        -1.0 * t4a(a, d, b, c, k, i, l, j)&
                        -1.0 * t4a(a, d, b, c, k, j, i, l)&
                        +t4a(a, d, b, c, k, j, l, i)&
                        +t4a(a, d, b, c, k, l, i, j)&
                        -1.0 * t4a(a, d, b, c, k, l, j, i)&
                        -1.0 * t4a(a, d, b, c, l, i, j, k)&
                        +t4a(a, d, b, c, l, i, k, j)&
                        +t4a(a, d, b, c, l, j, i, k)&
                        -1.0 * t4a(a, d, b, c, l, j, k, i)&
                        -1.0 * t4a(a, d, b, c, l, k, i, j)&
                        +t4a(a, d, b, c, l, k, j, i)&
                        -1.0 * t4a(a, d, c, b, i, j, k, l)&
                        +t4a(a, d, c, b, i, j, l, k)&
                        +t4a(a, d, c, b, i, k, j, l)&
                        -1.0 * t4a(a, d, c, b, i, k, l, j)&
                        -1.0 * t4a(a, d, c, b, i, l, j, k)&
                        +t4a(a, d, c, b, i, l, k, j)&
                        +t4a(a, d, c, b, j, i, k, l)&
                        -1.0 * t4a(a, d, c, b, j, i, l, k)&
                        -1.0 * t4a(a, d, c, b, j, k, i, l)&
                        +t4a(a, d, c, b, j, k, l, i)&
                        +t4a(a, d, c, b, j, l, i, k)&
                        -1.0 * t4a(a, d, c, b, j, l, k, i)&
                        -1.0 * t4a(a, d, c, b, k, i, j, l)&
                        +t4a(a, d, c, b, k, i, l, j)&
                        +t4a(a, d, c, b, k, j, i, l)&
                        -1.0 * t4a(a, d, c, b, k, j, l, i)&
                        -1.0 * t4a(a, d, c, b, k, l, i, j)&
                        +t4a(a, d, c, b, k, l, j, i)&
                        +t4a(a, d, c, b, l, i, j, k)&
                        -1.0 * t4a(a, d, c, b, l, i, k, j)&
                        -1.0 * t4a(a, d, c, b, l, j, i, k)&
                        +t4a(a, d, c, b, l, j, k, i)&
                        +t4a(a, d, c, b, l, k, i, j)&
                        -1.0 * t4a(a, d, c, b, l, k, j, i)&
                        -1.0 * t4a(b, a, c, d, i, j, k, l)&
                        +t4a(b, a, c, d, i, j, l, k)&
                        +t4a(b, a, c, d, i, k, j, l)&
                        -1.0 * t4a(b, a, c, d, i, k, l, j)&
                        -1.0 * t4a(b, a, c, d, i, l, j, k)&
                        +t4a(b, a, c, d, i, l, k, j)&
                        +t4a(b, a, c, d, j, i, k, l)&
                        -1.0 * t4a(b, a, c, d, j, i, l, k)&
                        -1.0 * t4a(b, a, c, d, j, k, i, l)&
                        +t4a(b, a, c, d, j, k, l, i)&
                        +t4a(b, a, c, d, j, l, i, k)&
                        -1.0 * t4a(b, a, c, d, j, l, k, i)&
                        -1.0 * t4a(b, a, c, d, k, i, j, l)&
                        +t4a(b, a, c, d, k, i, l, j)&
                        +t4a(b, a, c, d, k, j, i, l)&
                        -1.0 * t4a(b, a, c, d, k, j, l, i)&
                        -1.0 * t4a(b, a, c, d, k, l, i, j)&
                        +t4a(b, a, c, d, k, l, j, i)&
                        +t4a(b, a, c, d, l, i, j, k)&
                        -1.0 * t4a(b, a, c, d, l, i, k, j)&
                        -1.0 * t4a(b, a, c, d, l, j, i, k)&
                        +t4a(b, a, c, d, l, j, k, i)&
                        +t4a(b, a, c, d, l, k, i, j)&
                        -1.0 * t4a(b, a, c, d, l, k, j, i)&
                        +t4a(b, a, d, c, i, j, k, l)&
                        -1.0 * t4a(b, a, d, c, i, j, l, k)&
                        -1.0 * t4a(b, a, d, c, i, k, j, l)&
                        +t4a(b, a, d, c, i, k, l, j)&
                        +t4a(b, a, d, c, i, l, j, k)&
                        -1.0 * t4a(b, a, d, c, i, l, k, j)&
                        -1.0 * t4a(b, a, d, c, j, i, k, l)&
                        +t4a(b, a, d, c, j, i, l, k)&
                        +t4a(b, a, d, c, j, k, i, l)&
                        -1.0 * t4a(b, a, d, c, j, k, l, i)&
                        -1.0 * t4a(b, a, d, c, j, l, i, k)&
                        +t4a(b, a, d, c, j, l, k, i)&
                        +t4a(b, a, d, c, k, i, j, l)&
                        -1.0 * t4a(b, a, d, c, k, i, l, j)&
                        -1.0 * t4a(b, a, d, c, k, j, i, l)&
                        +t4a(b, a, d, c, k, j, l, i)&
                        +t4a(b, a, d, c, k, l, i, j)&
                        -1.0 * t4a(b, a, d, c, k, l, j, i)&
                        -1.0 * t4a(b, a, d, c, l, i, j, k)&
                        +t4a(b, a, d, c, l, i, k, j)&
                        +t4a(b, a, d, c, l, j, i, k)&
                        -1.0 * t4a(b, a, d, c, l, j, k, i)&
                        -1.0 * t4a(b, a, d, c, l, k, i, j)&
                        +t4a(b, a, d, c, l, k, j, i)&
                        +t4a(b, c, a, d, i, j, k, l)&
                        -1.0 * t4a(b, c, a, d, i, j, l, k)&
                        -1.0 * t4a(b, c, a, d, i, k, j, l)&
                        +t4a(b, c, a, d, i, k, l, j)&
                        +t4a(b, c, a, d, i, l, j, k)&
                        -1.0 * t4a(b, c, a, d, i, l, k, j)&
                        -1.0 * t4a(b, c, a, d, j, i, k, l)&
                        +t4a(b, c, a, d, j, i, l, k)&
                        +t4a(b, c, a, d, j, k, i, l)&
                        -1.0 * t4a(b, c, a, d, j, k, l, i)&
                        -1.0 * t4a(b, c, a, d, j, l, i, k)&
                        +t4a(b, c, a, d, j, l, k, i)&
                        +t4a(b, c, a, d, k, i, j, l)&
                        -1.0 * t4a(b, c, a, d, k, i, l, j)&
                        -1.0 * t4a(b, c, a, d, k, j, i, l)&
                        +t4a(b, c, a, d, k, j, l, i)&
                        +t4a(b, c, a, d, k, l, i, j)&
                        -1.0 * t4a(b, c, a, d, k, l, j, i)&
                        -1.0 * t4a(b, c, a, d, l, i, j, k)&
                        +t4a(b, c, a, d, l, i, k, j)&
                        +t4a(b, c, a, d, l, j, i, k)&
                        -1.0 * t4a(b, c, a, d, l, j, k, i)&
                        -1.0 * t4a(b, c, a, d, l, k, i, j)&
                        +t4a(b, c, a, d, l, k, j, i)&
                        -1.0 * t4a(b, c, d, a, i, j, k, l)&
                        +t4a(b, c, d, a, i, j, l, k)&
                        +t4a(b, c, d, a, i, k, j, l)&
                        -1.0 * t4a(b, c, d, a, i, k, l, j)&
                        -1.0 * t4a(b, c, d, a, i, l, j, k)&
                        +t4a(b, c, d, a, i, l, k, j)&
                        +t4a(b, c, d, a, j, i, k, l)&
                        -1.0 * t4a(b, c, d, a, j, i, l, k)&
                        -1.0 * t4a(b, c, d, a, j, k, i, l)&
                        +t4a(b, c, d, a, j, k, l, i)&
                        +t4a(b, c, d, a, j, l, i, k)&
                        -1.0 * t4a(b, c, d, a, j, l, k, i)&
                        -1.0 * t4a(b, c, d, a, k, i, j, l)&
                        +t4a(b, c, d, a, k, i, l, j)&
                        +t4a(b, c, d, a, k, j, i, l)&
                        -1.0 * t4a(b, c, d, a, k, j, l, i)&
                        -1.0 * t4a(b, c, d, a, k, l, i, j)&
                        +t4a(b, c, d, a, k, l, j, i)&
                        +t4a(b, c, d, a, l, i, j, k)&
                        -1.0 * t4a(b, c, d, a, l, i, k, j)&
                        -1.0 * t4a(b, c, d, a, l, j, i, k)&
                        +t4a(b, c, d, a, l, j, k, i)&
                        +t4a(b, c, d, a, l, k, i, j)&
                        -1.0 * t4a(b, c, d, a, l, k, j, i)&
                        -1.0 * t4a(b, d, a, c, i, j, k, l)&
                        +t4a(b, d, a, c, i, j, l, k)&
                        +t4a(b, d, a, c, i, k, j, l)&
                        -1.0 * t4a(b, d, a, c, i, k, l, j)&
                        -1.0 * t4a(b, d, a, c, i, l, j, k)&
                        +t4a(b, d, a, c, i, l, k, j)&
                        +t4a(b, d, a, c, j, i, k, l)&
                        -1.0 * t4a(b, d, a, c, j, i, l, k)&
                        -1.0 * t4a(b, d, a, c, j, k, i, l)&
                        +t4a(b, d, a, c, j, k, l, i)&
                        +t4a(b, d, a, c, j, l, i, k)&
                        -1.0 * t4a(b, d, a, c, j, l, k, i)&
                        -1.0 * t4a(b, d, a, c, k, i, j, l)&
                        +t4a(b, d, a, c, k, i, l, j)&
                        +t4a(b, d, a, c, k, j, i, l)

                        val = val &
                        -1.0 * t4a(b, d, a, c, k, j, l, i)&
                        -1.0 * t4a(b, d, a, c, k, l, i, j)&
                        +t4a(b, d, a, c, k, l, j, i)&
                        +t4a(b, d, a, c, l, i, j, k)&
                        -1.0 * t4a(b, d, a, c, l, i, k, j)&
                        -1.0 * t4a(b, d, a, c, l, j, i, k)&
                        +t4a(b, d, a, c, l, j, k, i)&
                        +t4a(b, d, a, c, l, k, i, j)&
                        -1.0 * t4a(b, d, a, c, l, k, j, i)&
                        +t4a(b, d, c, a, i, j, k, l)&
                        -1.0 * t4a(b, d, c, a, i, j, l, k)&
                        -1.0 * t4a(b, d, c, a, i, k, j, l)&
                        +t4a(b, d, c, a, i, k, l, j)&
                        +t4a(b, d, c, a, i, l, j, k)&
                        -1.0 * t4a(b, d, c, a, i, l, k, j)&
                        -1.0 * t4a(b, d, c, a, j, i, k, l)&
                        +t4a(b, d, c, a, j, i, l, k)&
                        +t4a(b, d, c, a, j, k, i, l)&
                        -1.0 * t4a(b, d, c, a, j, k, l, i)&
                        -1.0 * t4a(b, d, c, a, j, l, i, k)&
                        +t4a(b, d, c, a, j, l, k, i)&
                        +t4a(b, d, c, a, k, i, j, l)&
                        -1.0 * t4a(b, d, c, a, k, i, l, j)&
                        -1.0 * t4a(b, d, c, a, k, j, i, l)&
                        +t4a(b, d, c, a, k, j, l, i)&
                        +t4a(b, d, c, a, k, l, i, j)&
                        -1.0 * t4a(b, d, c, a, k, l, j, i)&
                        -1.0 * t4a(b, d, c, a, l, i, j, k)&
                        +t4a(b, d, c, a, l, i, k, j)&
                        +t4a(b, d, c, a, l, j, i, k)&
                        -1.0 * t4a(b, d, c, a, l, j, k, i)&
                        -1.0 * t4a(b, d, c, a, l, k, i, j)&
                        +t4a(b, d, c, a, l, k, j, i)&
                        +t4a(c, a, b, d, i, j, k, l)&
                        -1.0 * t4a(c, a, b, d, i, j, l, k)&
                        -1.0 * t4a(c, a, b, d, i, k, j, l)&
                        +t4a(c, a, b, d, i, k, l, j)&
                        +t4a(c, a, b, d, i, l, j, k)&
                        -1.0 * t4a(c, a, b, d, i, l, k, j)&
                        -1.0 * t4a(c, a, b, d, j, i, k, l)&
                        +t4a(c, a, b, d, j, i, l, k)&
                        +t4a(c, a, b, d, j, k, i, l)&
                        -1.0 * t4a(c, a, b, d, j, k, l, i)&
                        -1.0 * t4a(c, a, b, d, j, l, i, k)&
                        +t4a(c, a, b, d, j, l, k, i)&
                        +t4a(c, a, b, d, k, i, j, l)&
                        -1.0 * t4a(c, a, b, d, k, i, l, j)&
                        -1.0 * t4a(c, a, b, d, k, j, i, l)&
                        +t4a(c, a, b, d, k, j, l, i)&
                        +t4a(c, a, b, d, k, l, i, j)&
                        -1.0 * t4a(c, a, b, d, k, l, j, i)&
                        -1.0 * t4a(c, a, b, d, l, i, j, k)&
                        +t4a(c, a, b, d, l, i, k, j)&
                        +t4a(c, a, b, d, l, j, i, k)&
                        -1.0 * t4a(c, a, b, d, l, j, k, i)&
                        -1.0 * t4a(c, a, b, d, l, k, i, j)&
                        +t4a(c, a, b, d, l, k, j, i)&
                        -1.0 * t4a(c, a, d, b, i, j, k, l)&
                        +t4a(c, a, d, b, i, j, l, k)&
                        +t4a(c, a, d, b, i, k, j, l)&
                        -1.0 * t4a(c, a, d, b, i, k, l, j)&
                        -1.0 * t4a(c, a, d, b, i, l, j, k)&
                        +t4a(c, a, d, b, i, l, k, j)&
                        +t4a(c, a, d, b, j, i, k, l)&
                        -1.0 * t4a(c, a, d, b, j, i, l, k)&
                        -1.0 * t4a(c, a, d, b, j, k, i, l)&
                        +t4a(c, a, d, b, j, k, l, i)&
                        +t4a(c, a, d, b, j, l, i, k)&
                        -1.0 * t4a(c, a, d, b, j, l, k, i)&
                        -1.0 * t4a(c, a, d, b, k, i, j, l)&
                        +t4a(c, a, d, b, k, i, l, j)&
                        +t4a(c, a, d, b, k, j, i, l)&
                        -1.0 * t4a(c, a, d, b, k, j, l, i)&
                        -1.0 * t4a(c, a, d, b, k, l, i, j)&
                        +t4a(c, a, d, b, k, l, j, i)&
                        +t4a(c, a, d, b, l, i, j, k)&
                        -1.0 * t4a(c, a, d, b, l, i, k, j)&
                        -1.0 * t4a(c, a, d, b, l, j, i, k)&
                        +t4a(c, a, d, b, l, j, k, i)&
                        +t4a(c, a, d, b, l, k, i, j)&
                        -1.0 * t4a(c, a, d, b, l, k, j, i)&
                        -1.0 * t4a(c, b, a, d, i, j, k, l)&
                        +t4a(c, b, a, d, i, j, l, k)&
                        +t4a(c, b, a, d, i, k, j, l)&
                        -1.0 * t4a(c, b, a, d, i, k, l, j)&
                        -1.0 * t4a(c, b, a, d, i, l, j, k)&
                        +t4a(c, b, a, d, i, l, k, j)&
                        +t4a(c, b, a, d, j, i, k, l)&
                        -1.0 * t4a(c, b, a, d, j, i, l, k)&
                        -1.0 * t4a(c, b, a, d, j, k, i, l)&
                        +t4a(c, b, a, d, j, k, l, i)&
                        +t4a(c, b, a, d, j, l, i, k)&
                        -1.0 * t4a(c, b, a, d, j, l, k, i)&
                        -1.0 * t4a(c, b, a, d, k, i, j, l)&
                        +t4a(c, b, a, d, k, i, l, j)&
                        +t4a(c, b, a, d, k, j, i, l)&
                        -1.0 * t4a(c, b, a, d, k, j, l, i)&
                        -1.0 * t4a(c, b, a, d, k, l, i, j)&
                        +t4a(c, b, a, d, k, l, j, i)&
                        +t4a(c, b, a, d, l, i, j, k)&
                        -1.0 * t4a(c, b, a, d, l, i, k, j)&
                        -1.0 * t4a(c, b, a, d, l, j, i, k)&
                        +t4a(c, b, a, d, l, j, k, i)&
                        +t4a(c, b, a, d, l, k, i, j)&
                        -1.0 * t4a(c, b, a, d, l, k, j, i)&
                        +t4a(c, b, d, a, i, j, k, l)&
                        -1.0 * t4a(c, b, d, a, i, j, l, k)&
                        -1.0 * t4a(c, b, d, a, i, k, j, l)&
                        +t4a(c, b, d, a, i, k, l, j)&
                        +t4a(c, b, d, a, i, l, j, k)&
                        -1.0 * t4a(c, b, d, a, i, l, k, j)&
                        -1.0 * t4a(c, b, d, a, j, i, k, l)&
                        +t4a(c, b, d, a, j, i, l, k)&
                        +t4a(c, b, d, a, j, k, i, l)&
                        -1.0 * t4a(c, b, d, a, j, k, l, i)&
                        -1.0 * t4a(c, b, d, a, j, l, i, k)&
                        +t4a(c, b, d, a, j, l, k, i)&
                        +t4a(c, b, d, a, k, i, j, l)&
                        -1.0 * t4a(c, b, d, a, k, i, l, j)&
                        -1.0 * t4a(c, b, d, a, k, j, i, l)&
                        +t4a(c, b, d, a, k, j, l, i)&
                        +t4a(c, b, d, a, k, l, i, j)&
                        -1.0 * t4a(c, b, d, a, k, l, j, i)&
                        -1.0 * t4a(c, b, d, a, l, i, j, k)&
                        +t4a(c, b, d, a, l, i, k, j)&
                        +t4a(c, b, d, a, l, j, i, k)&
                        -1.0 * t4a(c, b, d, a, l, j, k, i)&
                        -1.0 * t4a(c, b, d, a, l, k, i, j)&
                        +t4a(c, b, d, a, l, k, j, i)&
                        +t4a(c, d, a, b, i, j, k, l)&
                        -1.0 * t4a(c, d, a, b, i, j, l, k)&
                        -1.0 * t4a(c, d, a, b, i, k, j, l)&
                        +t4a(c, d, a, b, i, k, l, j)&
                        +t4a(c, d, a, b, i, l, j, k)&
                        -1.0 * t4a(c, d, a, b, i, l, k, j)&
                        -1.0 * t4a(c, d, a, b, j, i, k, l)&
                        +t4a(c, d, a, b, j, i, l, k)&
                        +t4a(c, d, a, b, j, k, i, l)&
                        -1.0 * t4a(c, d, a, b, j, k, l, i)&
                        -1.0 * t4a(c, d, a, b, j, l, i, k)&
                        +t4a(c, d, a, b, j, l, k, i)&
                        +t4a(c, d, a, b, k, i, j, l)&
                        -1.0 * t4a(c, d, a, b, k, i, l, j)&
                        -1.0 * t4a(c, d, a, b, k, j, i, l)&
                        +t4a(c, d, a, b, k, j, l, i)&
                        +t4a(c, d, a, b, k, l, i, j)&
                        -1.0 * t4a(c, d, a, b, k, l, j, i)&
                        -1.0 * t4a(c, d, a, b, l, i, j, k)&
                        +t4a(c, d, a, b, l, i, k, j)&
                        +t4a(c, d, a, b, l, j, i, k)&
                        -1.0 * t4a(c, d, a, b, l, j, k, i)&
                        -1.0 * t4a(c, d, a, b, l, k, i, j)&
                        +t4a(c, d, a, b, l, k, j, i)&
                        -1.0 * t4a(c, d, b, a, i, j, k, l)&
                        +t4a(c, d, b, a, i, j, l, k)&
                        +t4a(c, d, b, a, i, k, j, l)&
                        -1.0 * t4a(c, d, b, a, i, k, l, j)&
                        -1.0 * t4a(c, d, b, a, i, l, j, k)&
                        +t4a(c, d, b, a, i, l, k, j)&
                        +t4a(c, d, b, a, j, i, k, l)&
                        -1.0 * t4a(c, d, b, a, j, i, l, k)&
                        -1.0 * t4a(c, d, b, a, j, k, i, l)&
                        +t4a(c, d, b, a, j, k, l, i)&
                        +t4a(c, d, b, a, j, l, i, k)&
                        -1.0 * t4a(c, d, b, a, j, l, k, i)&
                        -1.0 * t4a(c, d, b, a, k, i, j, l)&
                        +t4a(c, d, b, a, k, i, l, j)&
                        +t4a(c, d, b, a, k, j, i, l)&
                        -1.0 * t4a(c, d, b, a, k, j, l, i)&
                        -1.0 * t4a(c, d, b, a, k, l, i, j)&
                        +t4a(c, d, b, a, k, l, j, i)&
                        +t4a(c, d, b, a, l, i, j, k)&
                        -1.0 * t4a(c, d, b, a, l, i, k, j)&
                        -1.0 * t4a(c, d, b, a, l, j, i, k)&
                        +t4a(c, d, b, a, l, j, k, i)&
                        +t4a(c, d, b, a, l, k, i, j)&
                        -1.0 * t4a(c, d, b, a, l, k, j, i)&
                        -1.0 * t4a(d, a, b, c, i, j, k, l)&
                        +t4a(d, a, b, c, i, j, l, k)&
                        +t4a(d, a, b, c, i, k, j, l)&
                        -1.0 * t4a(d, a, b, c, i, k, l, j)&
                        -1.0 * t4a(d, a, b, c, i, l, j, k)&
                        +t4a(d, a, b, c, i, l, k, j)&
                        +t4a(d, a, b, c, j, i, k, l)&
                        -1.0 * t4a(d, a, b, c, j, i, l, k)&
                        -1.0 * t4a(d, a, b, c, j, k, i, l)&
                        +t4a(d, a, b, c, j, k, l, i)&
                        +t4a(d, a, b, c, j, l, i, k)&
                        -1.0 * t4a(d, a, b, c, j, l, k, i)&
                        -1.0 * t4a(d, a, b, c, k, i, j, l)&
                        +t4a(d, a, b, c, k, i, l, j)&
                        +t4a(d, a, b, c, k, j, i, l)&
                        -1.0 * t4a(d, a, b, c, k, j, l, i)&
                        -1.0 * t4a(d, a, b, c, k, l, i, j)&
                        +t4a(d, a, b, c, k, l, j, i)&
                        +t4a(d, a, b, c, l, i, j, k)&
                        -1.0 * t4a(d, a, b, c, l, i, k, j)&
                        -1.0 * t4a(d, a, b, c, l, j, i, k)&
                        +t4a(d, a, b, c, l, j, k, i)&
                        +t4a(d, a, b, c, l, k, i, j)&
                        -1.0 * t4a(d, a, b, c, l, k, j, i)&
                        +t4a(d, a, c, b, i, j, k, l)&
                        -1.0 * t4a(d, a, c, b, i, j, l, k)&
                        -1.0 * t4a(d, a, c, b, i, k, j, l)&
                        +t4a(d, a, c, b, i, k, l, j)&
                        +t4a(d, a, c, b, i, l, j, k)&
                        -1.0 * t4a(d, a, c, b, i, l, k, j)&
                        -1.0 * t4a(d, a, c, b, j, i, k, l)&
                        +t4a(d, a, c, b, j, i, l, k)&
                        +t4a(d, a, c, b, j, k, i, l)&
                        -1.0 * t4a(d, a, c, b, j, k, l, i)&
                        -1.0 * t4a(d, a, c, b, j, l, i, k)&
                        +t4a(d, a, c, b, j, l, k, i)&
                        +t4a(d, a, c, b, k, i, j, l)&
                        -1.0 * t4a(d, a, c, b, k, i, l, j)&
                        -1.0 * t4a(d, a, c, b, k, j, i, l)&
                        +t4a(d, a, c, b, k, j, l, i)&
                        +t4a(d, a, c, b, k, l, i, j)&
                        -1.0 * t4a(d, a, c, b, k, l, j, i)&
                        -1.0 * t4a(d, a, c, b, l, i, j, k)&
                        +t4a(d, a, c, b, l, i, k, j)&
                        +t4a(d, a, c, b, l, j, i, k)&
                        -1.0 * t4a(d, a, c, b, l, j, k, i)&
                        -1.0 * t4a(d, a, c, b, l, k, i, j)&
                        +t4a(d, a, c, b, l, k, j, i)&
                        +t4a(d, b, a, c, i, j, k, l)&
                        -1.0 * t4a(d, b, a, c, i, j, l, k)&
                        -1.0 * t4a(d, b, a, c, i, k, j, l)&
                        +t4a(d, b, a, c, i, k, l, j)&
                        +t4a(d, b, a, c, i, l, j, k)&
                        -1.0 * t4a(d, b, a, c, i, l, k, j)&
                        -1.0 * t4a(d, b, a, c, j, i, k, l)&
                        +t4a(d, b, a, c, j, i, l, k)&
                        +t4a(d, b, a, c, j, k, i, l)&
                        -1.0 * t4a(d, b, a, c, j, k, l, i)&
                        -1.0 * t4a(d, b, a, c, j, l, i, k)&
                        +t4a(d, b, a, c, j, l, k, i)&
                        +t4a(d, b, a, c, k, i, j, l)&
                        -1.0 * t4a(d, b, a, c, k, i, l, j)&
                        -1.0 * t4a(d, b, a, c, k, j, i, l)&
                        +t4a(d, b, a, c, k, j, l, i)&
                        +t4a(d, b, a, c, k, l, i, j)&
                        -1.0 * t4a(d, b, a, c, k, l, j, i)&
                        -1.0 * t4a(d, b, a, c, l, i, j, k)&
                        +t4a(d, b, a, c, l, i, k, j)&
                        +t4a(d, b, a, c, l, j, i, k)&
                        -1.0 * t4a(d, b, a, c, l, j, k, i)&
                        -1.0 * t4a(d, b, a, c, l, k, i, j)&
                        +t4a(d, b, a, c, l, k, j, i)&
                        -1.0 * t4a(d, b, c, a, i, j, k, l)&
                        +t4a(d, b, c, a, i, j, l, k)&
                        +t4a(d, b, c, a, i, k, j, l)&
                        -1.0 * t4a(d, b, c, a, i, k, l, j)&
                        -1.0 * t4a(d, b, c, a, i, l, j, k)&
                        +t4a(d, b, c, a, i, l, k, j)

                        val = val &
                        +t4a(d, b, c, a, j, i, k, l)&
                        -1.0 * t4a(d, b, c, a, j, i, l, k)&
                        -1.0 * t4a(d, b, c, a, j, k, i, l)&
                        +t4a(d, b, c, a, j, k, l, i)&
                        +t4a(d, b, c, a, j, l, i, k)&
                        -1.0 * t4a(d, b, c, a, j, l, k, i)&
                        -1.0 * t4a(d, b, c, a, k, i, j, l)&
                        +t4a(d, b, c, a, k, i, l, j)&
                        +t4a(d, b, c, a, k, j, i, l)&
                        -1.0 * t4a(d, b, c, a, k, j, l, i)&
                        -1.0 * t4a(d, b, c, a, k, l, i, j)&
                        +t4a(d, b, c, a, k, l, j, i)&
                        +t4a(d, b, c, a, l, i, j, k)&
                        -1.0 * t4a(d, b, c, a, l, i, k, j)&
                        -1.0 * t4a(d, b, c, a, l, j, i, k)&
                        +t4a(d, b, c, a, l, j, k, i)&
                        +t4a(d, b, c, a, l, k, i, j)&
                        -1.0 * t4a(d, b, c, a, l, k, j, i)&
                        -1.0 * t4a(d, c, a, b, i, j, k, l)&
                        +t4a(d, c, a, b, i, j, l, k)&
                        +t4a(d, c, a, b, i, k, j, l)&
                        -1.0 * t4a(d, c, a, b, i, k, l, j)&
                        -1.0 * t4a(d, c, a, b, i, l, j, k)&
                        +t4a(d, c, a, b, i, l, k, j)&
                        +t4a(d, c, a, b, j, i, k, l)&
                        -1.0 * t4a(d, c, a, b, j, i, l, k)&
                        -1.0 * t4a(d, c, a, b, j, k, i, l)&
                        +t4a(d, c, a, b, j, k, l, i)&
                        +t4a(d, c, a, b, j, l, i, k)&
                        -1.0 * t4a(d, c, a, b, j, l, k, i)&
                        -1.0 * t4a(d, c, a, b, k, i, j, l)&
                        +t4a(d, c, a, b, k, i, l, j)&
                        +t4a(d, c, a, b, k, j, i, l)&
                        -1.0 * t4a(d, c, a, b, k, j, l, i)&
                        -1.0 * t4a(d, c, a, b, k, l, i, j)&
                        +t4a(d, c, a, b, k, l, j, i)&
                        +t4a(d, c, a, b, l, i, j, k)&
                        -1.0 * t4a(d, c, a, b, l, i, k, j)&
                        -1.0 * t4a(d, c, a, b, l, j, i, k)&
                        +t4a(d, c, a, b, l, j, k, i)&
                        +t4a(d, c, a, b, l, k, i, j)&
                        -1.0 * t4a(d, c, a, b, l, k, j, i)&
                        +t4a(d, c, b, a, i, j, k, l)&
                        -1.0 * t4a(d, c, b, a, i, j, l, k)&
                        -1.0 * t4a(d, c, b, a, i, k, j, l)&
                        +t4a(d, c, b, a, i, k, l, j)&
                        +t4a(d, c, b, a, i, l, j, k)&
                        -1.0 * t4a(d, c, b, a, i, l, k, j)&
                        -1.0 * t4a(d, c, b, a, j, i, k, l)&
                        +t4a(d, c, b, a, j, i, l, k)&
                        +t4a(d, c, b, a, j, k, i, l)&
                        -1.0 * t4a(d, c, b, a, j, k, l, i)&
                        -1.0 * t4a(d, c, b, a, j, l, i, k)&
                        +t4a(d, c, b, a, j, l, k, i)&
                        +t4a(d, c, b, a, k, i, j, l)&
                        -1.0 * t4a(d, c, b, a, k, i, l, j)&
                        -1.0 * t4a(d, c, b, a, k, j, i, l)&
                        +t4a(d, c, b, a, k, j, l, i)&
                        +t4a(d, c, b, a, k, l, i, j)&
                        -1.0 * t4a(d, c, b, a, k, l, j, i)&
                        -1.0 * t4a(d, c, b, a, l, i, j, k)&
                        +t4a(d, c, b, a, l, i, k, j)&
                        +t4a(d, c, b, a, l, j, i, k)&
                        -1.0 * t4a(d, c, b, a, l, j, k, i)&
                        -1.0 * t4a(d, c, b, a, l, k, i, j)&
                        +t4a(d, c, b, a, l, k, j, i)

                        t4a(a, b, c, d, i, j, k, l) = val/denom
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
                        end do
                     end do
                  end do
               end do
            end do
         end do
      end do
   end do

end subroutine update_t4a

subroutine update_t4b(t4b, &
                      fA_oo, fA_vv, fB_oo, fB_vv, &
                      noa, nua, nob, nub)

      integer, intent(in)  :: noa, nob, nua, nub
      real(8), intent(in)  :: fA_oo(1:noa, 1:noa), &
                              fB_oo(1:nob, 1:nob), &
                              fA_vv(1:nua, 1:nua), &
                              fB_vv(1:nub, 1:nub)

      real(8), intent(inout) :: t4b(1:nua, 1:nua, 1:nua, 1:nub, 1:noa, 1:noa, 1:noa, 1:nob)
      !f2py intent(in, out)  :: t4b(0:nua-1, 0:nua-1, 0:nua-1, 0:nub-1, 0:noa-1, 0:noa-1, 0:noa-1, 0:nob-1)

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
                        +t4b(a, b, c, d, i, j, k, l)&
                        -1.0 * t4b(a, b, c, d, i, k, j, l)&
                        -1.0 * t4b(a, b, c, d, j, i, k, l)&
                        +t4b(a, b, c, d, j, k, i, l)&
                        +t4b(a, b, c, d, k, i, j, l)&
                        -1.0 * t4b(a, b, c, d, k, j, i, l)&
                        -1.0 * t4b(a, c, b, d, i, j, k, l)&
                        +t4b(a, c, b, d, i, k, j, l)&
                        +t4b(a, c, b, d, j, i, k, l)&
                        -1.0 * t4b(a, c, b, d, j, k, i, l)&
                        -1.0 * t4b(a, c, b, d, k, i, j, l)&
                        +t4b(a, c, b, d, k, j, i, l)&
                        -1.0 * t4b(b, a, c, d, i, j, k, l)&
                        +t4b(b, a, c, d, i, k, j, l)&
                        +t4b(b, a, c, d, j, i, k, l)&
                        -1.0 * t4b(b, a, c, d, j, k, i, l)&
                        -1.0 * t4b(b, a, c, d, k, i, j, l)&
                        +t4b(b, a, c, d, k, j, i, l)&
                        +t4b(b, c, a, d, i, j, k, l)&
                        -1.0 * t4b(b, c, a, d, i, k, j, l)&
                        -1.0 * t4b(b, c, a, d, j, i, k, l)&
                        +t4b(b, c, a, d, j, k, i, l)&
                        +t4b(b, c, a, d, k, i, j, l)&
                        -1.0 * t4b(b, c, a, d, k, j, i, l)&
                        +t4b(c, a, b, d, i, j, k, l)&
                        -1.0 * t4b(c, a, b, d, i, k, j, l)&
                        -1.0 * t4b(c, a, b, d, j, i, k, l)&
                        +t4b(c, a, b, d, j, k, i, l)&
                        +t4b(c, a, b, d, k, i, j, l)&
                        -1.0 * t4b(c, a, b, d, k, j, i, l)&
                        -1.0 * t4b(c, b, a, d, i, j, k, l)&
                        +t4b(c, b, a, d, i, k, j, l)&
                        +t4b(c, b, a, d, j, i, k, l)&
                        -1.0 * t4b(c, b, a, d, j, k, i, l)&
                        -1.0 * t4b(c, b, a, d, k, i, j, l)&
                        +t4b(c, b, a, d, k, j, i, l)

                        t4b(a, b, c, d, i, j, k, l) = val/denom
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
                        end do
                     end do
                  end do
               end do
            end do
         end do
      end do
   end do

end subroutine update_t4b


subroutine update_t4c(t4c,&
                      fA_oo, fA_vv, fB_oo, fB_vv, &
                      noa, nua, nob, nub)

      integer, intent(in)  :: noa, nua, nob, nub
      real(8), intent(in)  :: fA_oo(1:noa, 1:noa), &
                              fB_oo(1:nob, 1:nob), &
                              fA_vv(1:nua, 1:nua), &
                              fB_vv(1:nub, 1:nub)

      real(8), intent(inout) :: t4c(1:nua, 1:nua, 1:nub, 1:nub, 1:noa, 1:noa, 1:nob, 1:nob)
      !f2py intent(in, out)  :: t4c(0:nua-1, 0:nua-1, 0:nub-1, 0:nub-1, 0:noa-1, 0:noa-1, 0:nob-1, 0:nob-1)

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
                        +t4c(a, b, c, d, i, j, k, l)&
                        -1.0 * t4c(a, b, c, d, i, j, l, k)&
                        -1.0 * t4c(a, b, c, d, j, i, k, l)&
                        +t4c(a, b, c, d, j, i, l, k)&
                        -1.0 * t4c(a, b, d, c, i, j, k, l)&
                        +t4c(a, b, d, c, i, j, l, k)&
                        +t4c(a, b, d, c, j, i, k, l)&
                        -1.0 * t4c(a, b, d, c, j, i, l, k)&
                        -1.0 * t4c(b, a, c, d, i, j, k, l)&
                        +t4c(b, a, c, d, i, j, l, k)&
                        +t4c(b, a, c, d, j, i, k, l)&
                        -1.0 * t4c(b, a, c, d, j, i, l, k)&
                        +t4c(b, a, d, c, i, j, k, l)&
                        -1.0 * t4c(b, a, d, c, i, j, l, k)&
                        -1.0 * t4c(b, a, d, c, j, i, k, l)&
                        +t4c(b, a, d, c, j, i, l, k)

                        t4c(a, b, c, d, i, j, k, l) = val/denom
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

                        end do
                     end do
                  end do
               end do
            end do
         end do
      end do
   end do

end subroutine update_t4c




end module cc4_loops
