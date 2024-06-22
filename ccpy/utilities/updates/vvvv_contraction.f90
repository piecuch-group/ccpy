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
   
              subroutine contract_vt2_aa_cholesky(resid, R_chol, t2a, noa, nua, naux, norb)
                 
                        ! input variables
                        integer, intent(in) :: noa, nua, norb
                        real(kind=8), intent(in) :: R_chol(naux,norb,norb)
                        real(kind=8), intent(in) :: t2a(nua,nua,noa,noa)
                        ! output variables
                        real(kind=8), intent(inout) :: resid(nua,nua,noa,noa)
                        ! local variables
                        integer :: x, e, f, i, j, a, b, ab, nua2
                        integer, allocatable :: idx(:,:)
                        real(kind=8), allocatable :: batch_ints(:,:)
                        
                        !
                        ! get map of linear index ab -> (a,b) for a<b
                        !
                        nua2 = nua*(nua-1)/2
                        allocate(idx(nua2,2))
                        ab = 1
                        do a=1,nua
                           do b=a+1,nua
                              idx(ab,1)=a
                              idx(ab,2)=b
                              ab = ab + 1
                           end do
                        end do
                        
                        !
                        ! Perform loop over pairs a<b
                        !
                        allocate(batch_ints(nua,nua))
                        do ab=1,nua2
                           ! indices a<b
                           a = idx(ab,1); b = idx(ab,2);
                           !
                           ! build cholesky integral block
                           !
                           batch_ints = 0.0d0
                           do x=1,naux
                              do e=1,nua
                                 do f=e+1,nua
                                    batch_ints(e,f)=batch_ints(e,f)+R_chol(x,a+noa,e+noa)*R_chol(x,b+noa,f+noa)
                                 end do
                              end do
                           end do
                           !
                           ! perform PPL contraction
                           !
                           do i=1,noa
                              do j=i+1,noa
                                 do e=1,nua
                                    do f=e+1,nua
                                       resid(a,b,i,j)=resid(a,b,i,j)+batch_ints(e,f)*t2a(e,f,i,j)
                                       resid(b,a,i,j)=resid(b,a,i,j)+batch_ints(e,f)*t2a(e,f,i,j)
                                    end do
                                 end do
                              end do
                           end do
                        end do
                        deallocate(idx,batch_ints)
                 
              end subroutine contract_vt2_aa_cholesky


end module vvvv_contraction
