module vvvv_contraction

        implicit none

        contains

              subroutine vvvv_t2_sym(resid, R, t2, no, nu, naux)
                 ! Input variables
                 integer, intent(in) :: no, nu, naux
                 real(kind=8), intent(in) :: R(naux,nu,nu)
                 real(kind=8), intent(in) :: t2(no,no,nu,nu)
                 ! InOut variables
                 ! real(kind=8), intent(inout) :: resid(nu,nu,no,no)
                 !!!!!!f2py intent(in,out) :: resid(0:nu-1,0:nu-1,0:no-1,0:no-1)
                 real(kind=8), intent(out) :: resid(no,no,nu,nu)
                 ! Local variables
                 integer :: a, b, e, f, x, k1
                 real(kind=8) :: batch_ints(nu,nu)
                 integer :: ab_table(nu*(nu-1)/2,2)

                 resid = 0.0d0

                 k1 = 1
                 do a=1,nu
                    do b=a+1,nu
                       ab_table(k1,1) = a
                       ab_table(k1,2) = b
                       k1 = k1 + 1
                    end do
                 end do
                 
                 !$omp parallel default(shared),&
                 !$omp private(a,b,e,f,batch_ints)
                 !$omp do reduction(+:resid)
                 do k1=1,nu*(nu-1)/2
                       a = ab_table(k1,1); b = ab_table(k1,2);
                       ! compute batch_ints(a,b) = <x|ae>*<x|bf>
                       call dgemm('t','n',nu,nu,naux,1.0d0,R(:,:,b),naux,R(:,:,a),naux,0.0d0,batch_ints,nu)
                       ! antisymmetrize batch_ints(a,b) <- batch_ints(a,b) - v_fe(a,b)
                       batch_ints = batch_ints - transpose(batch_ints)
                       ! contract dT(a,b,:,:) <- 1/2 batch_ints(a,b) * t(efij)
                       do e=1,nu
                          do f=e+1,nu
                             !batch_ints(e,f) = 0.0d0
                             !do x=1,naux
                             !   batch_ints(e,f) = batch_ints(e,f) + R(x,a+no,e+no)*R(x,b+no,f+no)&
                             !                                     - R(x,a+no,f+no)*R(x,b+no,e+no)
                             !end do
                             resid(:,:,b,a) = resid(:,:,b,a) + batch_ints(f,e)*t2(:,:,f,e)
                          end do
                       end do
                 end do
                 !$omp end do
                 !$omp end parallel
              end subroutine vvvv_t2_sym

              subroutine vvvv_t2(resid, Ra, Rb, t2, noa, nua, nob, nub, naux)
                 ! Input variables
                 integer, intent(in) :: noa, nua, nob, nub, naux
                 real(kind=8), intent(in) :: Ra(naux,nua,nua)
                 real(kind=8), intent(in) :: Rb(naux,nub,nub)
                 real(kind=8), intent(in) :: t2(nob,noa,nub,nua)
                 ! InOut variables
                 ! real(kind=8), intent(inout) :: resid(noa,nob,nua,nub)
                 !!!!f2py intent(in,out) :: resid(0:nua-1,0:nub-1,0:noa-1,0:nob-1)
                 real(kind=8), intent(out) :: resid(nob,noa,nub,nua)
                 ! Local variables
                 integer :: a, b, e, f, x
                 real(kind=8) :: batch_ints(nub,nua)
                 
                 resid = 0.0d0
                 !$omp parallel default(shared),&
                 !$omp private(b,e,f,batch_ints)
                 !$omp do reduction(+:resid)
                 do a=1,nua
                    do b=1,nub
                       ! compute batch_ints(a) = <x|ae>*<x|bf>
                       ! call dgemm('t','n',nua,nub,naux,1.0d0,Ra(:,a,:),naux,Rb(:,b,:),naux,0.0d0,batch_ints,nua)
                       call dgemm('t','n',nub,nua,naux,1.0d0,Rb(:,:,b),naux,Ra(:,:,a),naux,0.0d0,batch_ints,nub)
                       ! contract dT(a,b,:,:) <- batch_ints(a,b) * t(efij)
                       do e=1,nua
                          do f=1,nub
                             resid(:,:,b,a) = resid(:,:,b,a) + batch_ints(f,e)*t2(:,:,f,e)
                          end do
                       end do
                    end do
                 end do
                 !$omp end do
                 !$omp end parallel
                 
              end subroutine vvvv_t2
           
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
                         integer :: idx

                         resid = 0.0d0
                         do i = 1,no
                            do j = i+1,no
                               do a = 1,nu
                                  do b = a+1,nu
                                     ! 1/2 h2(abef) * t2(efij)
                                     do e = 1,nu
                                        do f = e+1,nu
                                           call vvvv_index(idx,a,b,e,f,nu)
                                           !resid(b,a,j,i) = resid(b,a,j,i) + h2_vvvv(idx)*t2(f,e,j,i)
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

   
!              subroutine contract_vt2_aa_cholesky(resid, R_chol, t2a, noa, nua, naux, norb)
!
!                        ! input variables
!                        integer, intent(in) :: noa, nua, norb
!                        real(kind=8), intent(in) :: R_chol(naux,norb,norb)
!                        real(kind=8), intent(in) :: t2a(nua,nua,noa,noa)
!                        ! output variables
!                        real(kind=8), intent(inout) :: resid(nua,nua,noa,noa)
!                        ! local variables
!                        integer :: x, e, f, i, j, a, b, ab, nua2
!                        integer, allocatable :: idx(:,:)
!                        real(kind=8), allocatable :: batch_ints(:,:)
!
!                        !
!                        ! get map of linear index ab -> (a,b) for a<b
!                        !
!                        nua2 = nua*(nua-1)/2
!                        allocate(idx(nua2,2))
!                        ab = 1
!                        do a=1,nua
!                           do b=a+1,nua
!                              idx(ab,1)=a
!                              idx(ab,2)=b
!                              ab = ab + 1
!                           end do
!                        end do
!
!                        !
!                        ! Perform loop over pairs a<b
!                        !
!                        allocate(batch_ints(nua,nua))
!                        do ab=1,nua2
!                           ! indices a<b
!                           a = idx(ab,1); b = idx(ab,2);
!                           !
!                           ! build cholesky integral block
!                           !
!                           batch_ints = 0.0d0
!                           do x=1,naux
!                              do e=1,nua
!                                 do f=e+1,nua
!                                    batch_ints(e,f)=batch_ints(e,f)+R_chol(x,a+noa,e+noa)*R_chol(x,b+noa,f+noa)
!                                 end do
!                              end do
!                           end do
!                           !
!                           ! perform PPL contraction
!                           !
!                           do i=1,noa
!                              do j=i+1,noa
!                                 do e=1,nua
!                                    do f=e+1,nua
!                                       resid(a,b,i,j)=resid(a,b,i,j)+batch_ints(e,f)*t2a(e,f,i,j)
!                                       resid(b,a,i,j)=resid(b,a,i,j)+batch_ints(e,f)*t2a(e,f,i,j)
!                                    end do
!                                 end do
!                              end do
!                           end do
!                        end do
!                        deallocate(idx,batch_ints)
!
!              end subroutine contract_vt2_aa_cholesky


end module vvvv_contraction
