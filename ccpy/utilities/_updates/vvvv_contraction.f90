module vvvv_contraction

        implicit none

        contains
           
              subroutine vvvv_index(idx,a,b,c,d,nu)
                         
                         integer, intent(in) :: a, b, c, d, nu
                         integer, intent(out) :: idx
                         
                         integer :: ab, cd, n
                 
                         ! linear index of (a,b), a<b
                         ab = shiftr((2*nu - 2 - a) * (a - 1),1) + b - 2
                         ! linear index of (c,d), c<d
                         cd = shiftr((2*nu - 2 - d) * (c - 1),1) + d - 2
                         ! dimension of each (a,b), and (c,d) pairs = nu*(nu-1)/2
                         n = shiftr(nu*(nu - 1),1)
                         ! effective linear index h(idx) = <ab||cd>
                         idx = cd + n*ab + 1
                         
              end subroutine vvvv_index
           
              subroutine contract_vt2_pppp(resid,h2_vvvv,t2,no,nu)
                 
                         integer, intent(in) :: no, nu

                         real(kind=8), intent(in) :: h2_vvvv(nu,nu,nu,nu)
                         real(kind=8), intent(in) :: t2(nu,nu,no,no)

                         real(kind=8), intent(out) :: resid(nu,nu,no,no)
              
                         integer :: i, j, a, b, e, f
   
                         resid = 0.0d0
                         do i = 1,no
                            do j = i+1,no
                               do a = 1,nu
                                  do b = a+1,nu
                                     ! 1/2 h2(abef) * t2(efij)
                                     do e = 1,nu
                                        do f = e+1,nu
                                           ! idx = vvvv_index(f,e,b,a)
                                           resid(b,a,j,i) = resid(b,a,j,i) + h2_vvvv(f,e,b,a)*t2(f,e,j,i)
                                        end do
                                     end do
                                     resid(a,b,j,i) = -resid(b,a,j,i)
                                     resid(b,a,i,j) = -resid(b,a,j,i)
                                     resid(a,b,i,j) = resid(b,a,j,i)
                                  end do
                               end do
                            end do
                         end do
                 
              end subroutine contract_vt2_pppp


end module vvvv_contraction
