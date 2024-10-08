program main
  use qsort_module
  implicit none
  double precision :: A(10)
  integer :: iorder(10)

  integer :: i
   do i=1,10
     A(i) = 10-i
   enddo
   print *, A

   do i=1,10
     iorder(i) = i
   enddo
   call dsort(A, iorder, 10)

   print *, iorder
   print *, A

end

