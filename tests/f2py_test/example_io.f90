module example_io

      implicit none

      contains

              subroutine test()

                      integer, parameter :: io = 101

                      real(kind=8) :: a(10)

                      integer :: i

                      open(file='test.out', unit=io, status='new', form='formatted')
                      do i = 1 , 10
                         a(i) = float(i)**2 * 5.0
                         print*, a(i)
                         write(io,*) a(i)
                      end do
                      close(io)

              end subroutine test

end module example_io
