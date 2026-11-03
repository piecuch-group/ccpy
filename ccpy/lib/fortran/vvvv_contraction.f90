module vvvv_contraction

        implicit none

        contains

              subroutine vvvv_t2_sym(resid, R, t2, no, nu, naux)
                 ! resid(:,:,b,a) = sum_{e<f} [<ab|ef> - <ab|fe>](f,e) * t2(:,:,f,e) for a < b (zero otherwise),
                 ! with <ab|ef> = sum_x R(x,e,a) R(x,f,b). For each a, the integrals of a block of b are
                 ! built by one DGEMM and contracted with t2 (packed over e < f) by another.
                 ! Input variables
                 integer, intent(in) :: no, nu, naux
                 real(kind=8), intent(in) :: R(naux,nu,nu)
                 real(kind=8), intent(in) :: t2(no,no,nu,nu)
                 real(kind=8), intent(out) :: resid(no,no,nu,nu)
                 ! Local variables
                 integer, parameter :: nblk = 32
                 integer :: a, b0, nb, k, e, f, kp, npair
                 real(kind=8), allocatable :: tpack(:,:), y(:,:,:), z(:,:)

                 resid = 0.0d0
                 npair = nu*(nu-1)/2
                 if (npair == 0) return
                 ! t2 packed over the pairs e < f
                 allocate(tpack(no*no,npair))
                 kp = 0
                 do e = 1, nu
                    do f = e+1, nu
                       kp = kp + 1
                       tpack(:,kp) = reshape(t2(:,:,f,e), (/no*no/))
                    end do
                 end do

                 !$omp parallel default(shared) private(a,b0,nb,k,e,f,kp,y,z)
                 allocate(y(nu,nu,nblk), z(npair,nblk))
                 !$omp do schedule(dynamic)
                 do a = 1, nu-1
                    do b0 = a+1, nu, nblk
                       nb = min(nblk, nu-b0+1)
                       ! y(e,f,k) = sum_x R(x,e,a) R(x,f,b) for b = b0+k-1
                       call dgemm('t','n',nu,nu*nb,naux,1.0d0,R(:,:,a),naux,R(:,:,b0:b0+nb-1),naux,0.0d0,y,nu)
                       ! z(ef,k) = [<ab|ef> - <ab|fe>] evaluated at (f,e), for e < f
                       do k = 1, nb
                          kp = 0
                          do e = 1, nu
                             do f = e+1, nu
                                kp = kp + 1
                                z(kp,k) = y(e,f,k) - y(f,e,k)
                             end do
                          end do
                       end do
                       call dgemm('n','n',no*no,nb,npair,1.0d0,tpack,no*no,z,npair,0.0d0,resid(:,:,b0:b0+nb-1,a),no*no)
                    end do
                 end do
                 !$omp end do
                 deallocate(y, z)
                 !$omp end parallel
                 deallocate(tpack)
              end subroutine vvvv_t2_sym

              subroutine vvvv_t2(resid, Ra, Rb, t2, noa, nua, nob, nub, naux)
                 ! resid(:,:,b,a) = sum_{e,f} <ab|ef> * t2(:,:,f,e), with <ab|ef> = sum_x Ra(x,e,a) Rb(x,f,b)
                 ! (opposite spin). For each a, the integrals of a block of b are built by one DGEMM and
                 ! contracted with t2 by another.
                 ! Input variables
                 integer, intent(in) :: noa, nua, nob, nub, naux
                 real(kind=8), intent(in) :: Ra(naux,nua,nua)
                 real(kind=8), intent(in) :: Rb(naux,nub,nub)
                 real(kind=8), intent(in) :: t2(nob,noa,nub,nua)
                 real(kind=8), intent(out) :: resid(nob,noa,nub,nua)
                 ! Local variables
                 integer, parameter :: nblk = 32
                 integer :: a, b0, nb, k
                 real(kind=8), allocatable :: y(:,:,:), z(:,:,:)

                 resid = 0.0d0
                 !$omp parallel default(shared) private(a,b0,nb,k,y,z)
                 allocate(y(nua,nub,nblk), z(nub,nua,nblk))
                 !$omp do schedule(dynamic)
                 do a = 1, nua
                    do b0 = 1, nub, nblk
                       nb = min(nblk, nub-b0+1)
                       ! y(e,f,k) = sum_x Ra(x,e,a) Rb(x,f,b) for b = b0+k-1
                       call dgemm('t','n',nua,nub*nb,naux,1.0d0,Ra(:,:,a),naux,Rb(:,:,b0:b0+nb-1),naux,0.0d0,y,nua)
                       do k = 1, nb
                          z(:,:,k) = transpose(y(:,:,k))
                       end do
                       call dgemm('n','n',nob*noa,nb,nub*nua,1.0d0,t2,nob*noa,z,nub*nua,0.0d0,resid(:,:,b0:b0+nb-1,a),nob*noa)
                    end do
                 end do
                 !$omp end do
                 deallocate(y, z)
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
