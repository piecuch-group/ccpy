module reorder

        implicit none

        contains


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


end module reorder
