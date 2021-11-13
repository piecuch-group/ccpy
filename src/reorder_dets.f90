module reorder_dets

      implicit none

      contains

              subroutine reorder_1A_3A(det1_out,det2_out,phase,det1_in,det2_in)
        
                    integer, intent(in) :: det1_in(2), det2_in(6)
                    integer, intent(out) :: det1_out(2), det2_out(6)
                    real(kind=8), intent(out) :: phase

                    integer :: p, q, perm, tmp

                    perm = 0

                    do p = 1,1
                       do q = 1,3
                          if (det1_in(p) == det2_in(q)) then
                             perm = perm + (q-1)
                             
                          end if
                          if (det1_in(p+1) == det2_in(q+3)) then
                             perm = perm + (q-1)
                          end if
                        end do
                     end do



                       
            
