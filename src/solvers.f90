module solvers
! Module containing common functions for CC solvers, including
! DIIS extrapolation for ground-state and left CC routines and
! non-Hermitian Davidson solver for EOMCC routines. Emphasis
! placed on low-memory schemes that use data I/O to load in
! only a small number of vectors at a time.

      implicit none

      contains

              subroutine diis(t_xtrap, vec_dim, ndiis)

                      integer, intent(in) :: vec_dim, ndiis
                      real(kind=8), intent(out) :: t_xtrap(vec_dim)

                      real(kind=8) :: Bmat(ndiis+1,nddis+1), rhs(ndiis+1), coeff(nddis+1)
                      ! record numbers for the T and diis_resid vectors
                      !(ndiis of them stored in each file)
                      integer, parameter :: iT=100, idT=101 
                      integer :: i, j
                      real(kind=8), allocatable :: x1(:), x2(:)

                      Bmat = -1.0d0
                      rhs = 0.0d0
                      rhs(ndiis+1) = -1.0d0

                      allocate(x1(vec_dim),x2(vec_dim))

                      do i = 1,ndiis
                         read(idT,file='dx_list.bin',rec=i) x1
                         do j = i,ndiis
                            read(idT,file='dx_list.bin',rec=j) x2
                            Bmat(i,j) = ddot(vec_dim,x1,x2)
                            Bmat(j,i) = Bmat(i,j)
                         end do
                      end do
                      
                      deallocate(x1,x2)

                      coeff = solve_gauss(Bmat,rhs)

                      allocate(x1(vec_dim))
                      t_xtrap = 0.0d0
                      do i = 1,ndiis
                         read(iT,file='x_list.bin',rec=i) x1
                         t_xtrap = t_xtrap + coeff(i)*x1
                      end do
                      deallocate(x1)

            end subroutine diis

            function solve_gauss(A,x) result(y)
            
            end function solve_gauss

end module solvers
                          

