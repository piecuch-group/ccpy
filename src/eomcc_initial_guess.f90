module eomcc_initial_guess

        implicit none

        contains

                subroutine eomccs_d(nroot,noact,nuact,Rvec,omega,&
                                    H1A_oo,H1A_vv,H1A_ov,&
                                    H1B_oo,H1B_vv,H1B_ov,&
                                    H2A_oooo,H2A_vvvv,H2A_voov,H2A_vooo,H2A_vvov,H2A_ooov,H2A_vovv,&
                                    H2B_oooo,H2B_vvvv,H2B_voov,H2B_ovvo,H2B_vovo,H2B_ovov,H2B_vooo,&
                                    H2B_ovoo,H2B_vvov,H2B_vvvo,H2B_ooov,H2B_oovo,H2B_vovv,H2B_ovvv,&
                                    H2C_oooo,H2C_vvvv,H2C_voov,H2C_vooo,H2C_vvov,H2C_ooov,H2C_vovv,&
                                    noa,nua,nob,nub)

                        integer, intent(in) :: nroot, noa, nua, nob, nub, noact, nuact
                        real(kind=8), intent(in) :: H1A_oo(noa,noa),H1A_vv(nua,nua),H1A_ov(noa,nua),&
                                    H1B_oo(nob,nob),H1B_vv(nub,nub),H1B_ov(nob,nub),&
                                    H2A_oooo(noa,noa,noa,noa),H2A_vvvv(nua,nua,nua,nua),H2A_voov(nua,noa,noa,nua),&
                                    H2A_vooo(nua,noa,noa,noa),H2A_vvov(nua,nua,noa,nua),H2A_ooov(noa,noa,noa,nua),&
                                    H2A_vovv(nua,noa,nua,nua),&
                                    H2B_oooo(noa,nob,noa,nob),H2B_vvvv(nua,nub,nua,nub),H2B_voov(nua,nob,noa,nub),&
                                    H2B_ovvo(noa,nub,nua,nob),H2B_vovo(nua,nob,nua,nob),H2B_ovov(noa,nub,noa,nub),&
                                    H2B_vooo(nua,nob,noa,nob),H2B_ovoo(noa,nub,noa,nob),H2B_vvov(nua,nub,noa,nub),&
                                    H2B_vvvo(nua,nub,nua,nob),H2B_ooov(noa,nob,noa,nub),H2B_oovo(noa,nob,nua,nob),&
                                    H2B_vovv(nua,nob,nua,nub),H2B_ovvv(noa,nub,nua,nub),&
                                    H2C_oooo(nob,nob,nob,nob),H2C_vvvv(nub,nub,nub,nub),H2C_voov(nub,nob,nob,nub),&
                                    H2C_vooo(nub,nob,nob,nob),H2C_vvov(nub,nub,nob,nub),H2C_ooov(nob,nob,nob,nub),&
                                    H2C_vovv(nub,nob,nub,nub)

                        real(kind=8), intent(out) :: omega(nroot),Rvec(noa**2*nua**2+noa*nua*nob*nub+nob**2*nub**2,nroot)

                        real(kind=8), allocatable :: Hmat(:,:), evecs(:,:), evals(:), Htemp(:,:)
                        real(kind=8) :: onebody, twobody
                        integer :: i, j, a, b, i2, j2, a2, b2, ndim, n1a, n1b,&
                                   n2a_unique, n2b, n2c_unique, ct1, ct2

                        n1a = noa * nua
                        n1b = nob * nub
                        n2a_unique = noa*(noa - 1)/2 * nua*(nua - 1)/2
                        n2b = noa*nob*nua*nub
                        n2c_unique = nob*(nob - 1)/2 * nub*(nub - 1)/2

                        ndim = n1a + n1b + n2a_unique + n2b + n2c_unique

                        allocate(Hmat(ndim,ndim), evecs(ndim,ndim), evals(ndim))

                        ct1 = 0
                        ct2 = 0
                        allocate(Htemp(n1a,n1a))
                        ! < ia | H | jb >
                        do i = 1 , noa
                        do a = 1 , nua
                            ct1 = ct + 1
                            do j = 1 , noa
                            do b = 1 , nua
                                ct2 = ct2 + 1
                                if ( i == j .and. a == b ) ! < ia | H | ia >
                                    onebody = H1A_vv(a,a) - H1A_oo(i,i)
                                    twobody = H2A_voov(a,i,i,a)
                                elseif ( a == b ) ! < ia | H | ja >
                                    onebody = -H1A_oo(j,i)
                                    twobody = H2A_voov(a,j,i,a)
                                elseif ( i == j ) ! < ia | H | ib >
                                    onebody = H1A_vv(a,b)
                                    twobody = H2A_voov(a,i,i,b)
                                else ! < ia | H | jb >
                                    onebody = 0.0
                                    twobody = H2A_voov(a,j,i,b)
                                end if
                                Htemp(ct1,ct2) = onebody + twobody
                            end do
                            end do
                        end do
                        end do
                        Hmat(1:n1a,1:n1a) = Htemp
                        deallocate(Htemp)

                        ct1 = 0
                        ct2 = 0
                        allocate(Htemp(n1b,n1a))
                        ! < i~a~ | H | jb >
                        do i = 1 , nob
                        do a = 1 , nub
                            ct1 = ct1 + 1
                            do j = 1 , noa
                            do b = 1 , nua
                                ct2 = ct2 + 1
                                onebody = 0.0
                                twobody = H2B_ovvo(j,a,b,i)
                                Htemp(ct1,ct2) = onebody + twobody
                            end do
                            end do
                        end do
                        end do
                        Hmat(n1a+1:n1a+n1b,1:n1a) = Htemp
                        deallocate(Htemp)

                        ct1 = 0
                        ct2 = 0
                        allocate(Htemp(n1a,n1b))
                        ! < ia | H | j~b~ >
                        do i = 1 , noa
                        do a = 1 , nua
                            ct1 = ct1 + 1
                            do j = 1 , nob
                            do b = 1 , nub
                                ct2 = ct2 + 1
                                onebody = 0.0
                                twobody = H2B_voov(a,j,i,b)
                                Htemp(ct1,ct2) = onebody + twobody
                            end do
                            end do
                        end do
                        end do
                        Hmat(1:n1a,n1a+1:n1a+n1b) = Htemp
                        deallocate(Htemp)

                        ct1 = 0
                        ct2 = 0
                        allocate(Htemp(n1b,n1b))
                        ! < i~a~ | H | j~b~ >
                        do i = 1 , nob
                        do a = 1 , nub
                            ct1 = ct + 1
                            do j = 1 , nob
                            do b = 1 , nub
                                ct2 = ct2 + 1
                                if ( i == j .and. a == b ) ! < i~a~ | H | i~a~ >
                                    onebody = H1B_vv(a,a) - H1B_oo(i,i)
                                    twobody = H2C_voov(a,i,i,a)
                                elseif ( a == b ) ! < i~a~ | H | j~a~ >
                                    onebody = -H1B_oo(j,i)
                                    twobody = H2C_voov(a,j,i,a)
                                elseif ( i == j ) ! < i~a~ | H | i~b~ >
                                    onebody = H1B_vv(a,b)
                                    twobody = H2C_voov(a,i,i,b)
                                else ! < i~a~ | H | j~b~ >
                                    onebody = 0.0
                                    twobody = H2C_voov(a,j,i,b)
                                end if
                                Htemp(ct1,ct2) = onebody + twobody
                            end do
                            end do
                        end do
                        end do
                        Hmat(n1a+1:n1a+n1b,n1a+1:n1a+n1b) = Htemp
                        deallocate(Htemp)





                end subroutine eomccs_d

                subroutine reorder_dets(I1,I2,idx1,idx2,phase)
                    ! reorder bitstring determinants I1 = [I1a, I1b] and I2 =
                    ! [I2a,I2b] into the order of maximum coincidence as
                    ! excitations out of HF, idx1 and idx2, and the resulting
                    ! phase



                end subroutine reorder_dets

                !function onebody_HBar_slater(det1,det2,H1A_oo,H1A_vv,H1A_ov,H1B_oo,H1B_vv,H1B_ov,noa,nua,nob,nub) result(val)

                 !       integer, intent(in) :: noa, nua, nob, nub, i1, a1, i2, a2
                 !       real(kind=8), intent(in) :: H1A_oo(noa,noa),H1A_vv(nua,nua),H1A_ov(noa,nua),&
                 !                                   H1B_oo(nob,nob),H1B_vv(nub,nub),H1B_ov(nob,nub)

                 !       real(kind=8) :: val


                !end function onebody_HBar_slater

end module eomcc_initial_guess                         
