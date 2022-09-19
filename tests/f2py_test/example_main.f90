program example_io

      implicit none

      integer, parameter :: io = 101

      real(kind=8) :: a(10)

      integer :: i

      open(file='test.out', unit=io, status='replace', form='formatted')
      do i = 1 , 10
         a(i) = float(i)**2 * 5.0
         write(io,*) a(i)
      end do
      close(io)

end program example_io
