module reorder

        implicit none

        contains
           
              subroutine reorder_amplitudes(l3_amps, l3_excits, t3_excits, n3)
                 
                 integer, intent(in) :: n3
                 integer, intent(in) :: t3_excits(6,n3)

                 integer, intent(inout) :: l3_excits(6,n3)
                 !f2py intent(in,out) :: l3_excits(6,0:n3-1)
                 real(kind=8), intent(inout) :: l3_amps(n3)
                 !f2py intent(in,out) :: l3_amps(0:n3-1)
      
                 integer :: i, j, k, a, b, c, l, m, n, d, e, f, idet, jdet, tmp(6)
                 real(kind=8) :: l_amp
                 
                 do idet = 1, n3
                    a = t3_excits(1,idet); b = t3_excits(2,idet); c = t3_excits(3,idet);
                    i = t3_excits(4,idet); j = t3_excits(5,idet); k = t3_excits(6,idet);
                    do jdet = 1, n3
                       d = l3_excits(1,jdet); e = l3_excits(2,jdet); f = l3_excits(3,jdet);
                       l = l3_excits(4,jdet); m = l3_excits(5,jdet); n = l3_excits(6,jdet);
                       if (a==d .and. b==e .and. c==f .and. i==l .and. j==m .and. k==n) then
                          ! swap the values at l3(deflmn) with l3(abcijk)
                          l_amp = l3_amps(jdet)
                          l3_amps(jdet) = l3_amps(idet)
                          l3_amps(idet) = l_amp
                          ! also swap the corresponding entries in the l3_excits array
                          tmp(1) = l3_excits(1,jdet); tmp(2) = l3_excits(2,jdet); tmp(3) = l3_excits(3,jdet);
                          tmp(4) = l3_excits(4,jdet); tmp(5) = l3_excits(5,jdet); tmp(6) = l3_excits(6,jdet);
                          
                          l3_excits(1,jdet) = l3_excits(1,idet); l3_excits(2,jdet) = l3_excits(2,idet); l3_excits(3,jdet) = l3_excits(3,idet);
                          l3_excits(4,jdet) = l3_excits(4,idet); l3_excits(5,jdet) = l3_excits(5,idet); l3_excits(6,jdet) = l3_excits(6,idet);
                          
                          l3_excits(1,idet) = tmp(1); l3_excits(2,idet) = tmp(2); l3_excits(3,idet) = tmp(3);
                          l3_excits(4,idet) = tmp(4); l3_excits(5,idet) = tmp(5); l3_excits(6,idet) = tmp(6);
                       end if
                    end do
                 end do
                 
              end subroutine reorder_amplitudes

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
