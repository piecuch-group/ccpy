module mbgf_loops

      implicit none

      contains

              subroutine mbgf2_selfenergy(omega,fA_oo,fA_vv,fB_oo,fB_vv,&
                            vA_oovv,vA_vvoo,vA_ooov,vA_ovoo,&
                            vB_oovv,vB_vvoo,vB_ooov,vB_ovoo,vB_oovo,vB_vooo,&
                            vC_oovv,vC_vvoo,vC_ooov,vC_ovoo,&
                            noa,nua,nob,nub,sigma_a,sigma_b)

                    integer, intent(in) :: noa, nua, nob, nub
                    real(kind=8), intent(in) :: omega,fA_oo(noa,noa),fA_vv(nua,nua),fB_oo(nob,nob),fB_vv(nub,nub),&
                                                vA_oovv(noa,noa,nua,nua),vA_vvoo(nua,nua,noa,noa),&
                                                vA_ooov(noa,noa,noa,nua),vA_ovoo(noa,nua,noa,noa),&
                                                vB_oovv(noa,nob,nua,nub),vB_vvoo(nua,nub,noa,nob),&
                                                vB_ooov(noa,nob,noa,nub),vB_ovoo(noa,nub,noa,nob),&
                                                vB_oovo(noa,nob,nua,nob),vB_vooo(nua,nob,noa,nob),&
                                                vC_oovv(nob,nob,nub,nub),vC_vvoo(nub,nub,nob,nob),&
                                                vC_ooov(nob,nob,nob,nub),vC_ovoo(nob,nub,nob,nob)
                    real(kind=8), intent(out) :: sigma_a(noa+nua,noa+nua),sigma_b(nob+nub,nob+nub)

                    integer :: i, j, a, b, m, n, e, f
                    real(kind=8) :: denom ,e_nf

                    sigma_a = 0.0d0
                    sigma_b = 0.0d0

                    do i = 1 , noa
                       do j = 1 , noa
                          ! alpha loop
                          do n = 1 , noa
                             do f = 1 , nua
                                e_nf = fA_oo(n,n) - fA_vv(f,f)
                                ! hole diagram
                                do m = n+1 , noa
                                   denom = fA_oo(m,m) + e_nf - omega
                                   sigma_a(i,j) = sigma_a(i,j) -&
                                           vA_ooov(m,n,i,f)*vA_ovoo(j,f,m,n)/denom
                                end do
                                ! particle diagram
                                do e = f+1 , nua
                                   denom = e_nf - fA_vv(e,e) + omega
                                   sigma_a(i,j) = sigma_a(i,j) +&
                                           vA_oovv(j,n,e,f)*vA_vvoo(e,f,i,n)/denom
                                end do
                             end do
                          end do
                          ! beta loop
                          do n = 1 , nob
                             do f = 1 , nub
                                e_nf = fB_oo(n,n) - fB_vv(f,f)
                                ! hole diagram
                                do m = 1 , noa
                                   denom = fA_oo(m,m) + e_nf - omega
                                   sigma_a(i,j) = sigma_a(i,j) -&
                                             vB_ooov(m,n,i,f)*vB_ovoo(j,f,m,n)/denom
                                end do
                                ! particle diagram
                                do e = 1 , nua
                                   denom = e_nf - fA_vv(e,e) + omega
                                   sigma_a(i,j) = sigma_a(i,j) +&
                                             vB_oovv(j,n,e,f)*vB_vvoo(e,f,i,n)/denom
                                end do
                             end do
                          end do
                       end do
                    end do

              end subroutine mbgf2_selfenergy

end module mbgf_loops

