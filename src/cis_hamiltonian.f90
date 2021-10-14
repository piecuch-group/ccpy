module cis_hamiltonian

        implicit none

        contains 

                subroutine build_cis(H,fa_oo,fa_vv,fb_oo,fb_vv,vA_voov,vB_voov,vB_ovvo,vC_voov,noa,nua,nob,nub)

                        integer, intent(in) :: noa, nua, nob, nub
                        ! f2py integer, intent(in) :: noa, nua, nob, nub
                        real, intent(in) :: fa_oo(noa,noa), fa_vv(nua,nua), vA_voov(nua,noa,noa,nua), &
                                fb_oo(nob,nob), fb_vv(nub,nub), vB_voov(nua,nob,noa,nub), &
                                vB_ovvo(noa,nub,nua,nob), vC_voov(nub,nob,nob,nub)
                        ! f2py real, intent(in) :: fa_oo(0:noa-1,0:noa-1), fa_vv(0:nua-1,0:nua-1)
                        ! f2py real, intent(in) :: fb_oo(0:nob-1,0:nob-1), fb_vv(0:nub-1,0:nub-1)
                        ! f2py real, intent(in) :: vA_voov(0:nua-1,0:noa-1,0:noa-1,0:nua-1)
                        ! f2py real, intent(in) :: vB_voov(0:nua-1,0:nob-1,0:noa-1,0:nub-1) 
                        ! f2py real, intent(in) :: vB_ovvo(0:noa-1,0:nub-1,0:nua-1,0:nob-1)
                        ! f2py real, intent(in) :: vC_voov(0:nub-1,0:nob-1,0:nob-1,0:nub-1)
                        real, intent(out) :: H(nua*noa+nub*nob,nua*noa+nub*nob)
                        ! f2py real, intent(out) :: H(0:nua*noa+nub*nob-1,0:nua*noa+nub*nob-1)
                        real :: HAA(nua*noa,nua*noa), HAB(nua*noa,nub*nob), HBA(nub*nob,nua*noa), HBB(nub*nob,nub*nob)
                        integer :: ct1, ct2, i, j, a, b, n1a, n1b 

                        ct1 = 1
                        do i = 1,noa
                                do a = 1,nua
                                ct2 = 1
                                do j = 1,noa
                                        do b = 1,nua
                                        HAA(ct1,ct2) = vA_voov(a,j,i,b) 
                                        if (i==j) then 
                                                HAA(ct1,ct2) = HAA(ct1,ct2) + fA_vv(a,b) 
                                        end if 
                                        if (a==b) then 
                                                HAA(ct1,ct2) = HAA(ct1,ct2) - fA_oo(j,i) 
                                        end if 
                                        ct2=ct2+1
                                        end do 
                                end do 
                                ct1=ct1+1
                                end do 
                        end do 

                        ct1 = 1
                        do i = 1,noa
                                do a = 1,nua
                                ct2 = 1
                                do j = 1,nob
                                        do b = 1,nub
                                        HAB(ct1,ct2) = vB_voov(a,j,i,b)
                                        ct2=ct2+1
                                        end do 
                                end do 
                                ct1=ct1+1
                                end do 
                        end do

                        ct1 = 1
                        do i = 1,nob
                                do a = 1,nub
                                ct2 = 1
                                do j = 1,noa
                                        do b = 1,nua
                                        HBA(ct1,ct2) = vB_ovvo(j,a,b,i)
                                        ct2=ct2+1
                                        end do 
                                end do 
                                ct1=ct1+1
                                end do 
                        end do 

                        ct1 = 1
                        do i = 1,nob
                                do a = 1,nub
                                ct2 = 1
                                do j = 1,nob
                                        do b = 1,nub
                                        HBB(ct1,ct2) = vC_voov(a,j,i,b) 
                                        if (i==j) then 
                                                HBB(ct1,ct2) = HBB(ct1,ct2) + fB_vv(a,b) 
                                        end if 
                                        if (a==b) then 
                                                HBB(ct1,ct2) = HBB(ct1,ct2) - fB_oo(j,i) 
                                        end if 
                                        ct2=ct2+1
                                        end do 
                                end do 
                                ct1=ct1+1
                                end do 
                        end do 

                        n1a = noa * nua 
                        n1b = nob * nub

                        H(1:n1a,1:n1a) = HAA 
                        H(1:n1a,n1a+1:n1a+n1b) = HAB 
                        H(n1a+1:n1a+n1b,1:n1a) = HBA 
                        H(n1a+1:n1a+n1b,n1a+1:n1a+n1b) = HBB 


                end subroutine build_cis

end module cis_hamiltonian
