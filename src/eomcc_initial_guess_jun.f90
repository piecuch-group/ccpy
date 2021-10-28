module eomcc_initial_guess_jun

        implicit none

        contains

                subroutine eomccs_d(nroot,noact,nuact,nfroz,Rvec,omega,&
                                    H1A_oo,H1A_vv,H1A_ov,&
                                    H1B_oo,H1B_vv,H1B_ov,&
                                    H2A_oooo,H2A_vvvv,H2A_voov,H2A_vooo,H2A_vvov,H2A_ooov,H2A_vovv,&
                                    H2B_oooo,H2B_vvvv,H2B_voov,H2B_ovvo,H2B_vovo,H2B_ovov,H2B_vooo,&
                                    H2B_ovoo,H2B_vvov,H2B_vvvo,H2B_ooov,H2B_oovo,H2B_vovv,H2B_ovvv,&
                                    H2C_oooo,H2C_vvvv,H2C_voov,H2C_vooo,H2C_vvov,H2C_ooov,H2C_vovv,&
                                    noa,nua,nob,nub)

                        integer, intent(in) :: nroot, noa, nua, nob, nub, noact, nuact, nfroz
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
                                   n2a_unique, n2b, n2c_unique, ct1, ct2,&
                                   N0, N1, N2, N3, KKK, LLL, LLLA
                        integer, allocatable :: indAll(:), ind1A(:,:),&
                        ind1B(:,:), indA2A(:,:,:,:), indA2B(:,:,:,:),&
                        indA2C(:,:,:,:), ind2A(:,:,:,:), ind2B(:,:,:,:),&
                        ind2C(:,:,:,:)

                        n1a = noa * nua
                        n1b = nob * nub
                        n2a_unique = noa*(noa - 1)/2 * nua*(nua - 1)/2
                        n2b = noa*nob*nua*nub
                        n2c_unique = nob*(nob - 1)/2 * nub*(nub - 1)/2

                        ndim = n1a + n1b + n2a_unique + n2b + n2c_unique

                        allocate(Hmat(ndim,ndim), evecs(ndim,ndim), evals(ndim))

                        N0 = nfroz
                        N1 = noa+nfroz
                        N2 = nob+nfroz
                        N3 = nua+noa+nfroz
                        N4 = nub+nob+nfroz

       allocate(indAll(KKK))
       allocate(ind1A(N1+1:N3,N0+1:N1))
       allocate(ind1B(N2+1:N3,N0+1:N2))
       allocate(ind2A(N1+1:N3,N1+1:N3,N0+1:N1,N0+1:N1))
       allocate(ind2B(N2+1:N3,N1+1:N3,N0+1:N2,N0+1:N1))
       allocate(ind2C(N2+1:N3,N2+1:N3,N0+1:N2,N0+1:N2))
       allocate(indA2A(N1+1:N3,N1+1:N3,N0+1:N1,N0+1:N1))
       allocate(indA2B(N2+1:N3,N1+1:N3,N0+1:N2,N0+1:N1))
       allocate(indA2C(N2+1:N3,N2+1:N3,N0+1:N2,N0+1:N2))
       indAll=0
       ind1A=0;ind1B=0
       ind2A=0;ind2B=0;ind2C=0
       indA2A=0;indA2B=0;indA2C=0
C
       LLL=0;LLLA=0
!1A
       do i=N0+1,N1;do a=N1+1,N3
        LLL=LLL+1
        ind1A(a,i)=LLL
        indAll(LLL)=LLL
       enddo;enddo
!1B
       do i=N0+1,N2;do a=N2+1,N3
        LLL=LLL+1
        ind1B(a,i)=LLL
        indAll(LLL)=LLL
       enddo;enddo
C
       LLLA=LLL;ii=2;ia=2
!2A
       do i=N0+1,N1;do j=N0+1,N1;do a=N1+1,N3;do b=N1+1,N3
        LLL=LLL+1
        ind2A(b,a,j,i)=LLL
        if(i.eq.j.or.a.eq.b)cycle
        if(numA(i,j,M3).lt.ii.or.numA1(a,b,M4).lt.ia)cycle
        LLLA=LLLA+1
        indA2A(b,a,j,i)=LLLA
        indAll(LLL)=LLLA
       enddo;enddo;enddo;enddo
!2B
       do i=N0+1,N1;do j=N0+1,N2;do a=N1+1,N3;do b=N2+1,N3
        LLL=LLL+1
        ind2B(b,a,j,i)=LLL
        if(numA(i,j,M3).lt.ii.or.numA1(a,b,M4).lt.ia)cycle
        LLLA=LLLA+1
        indA2B(b,a,j,i)=LLLA
        indAll(LLL)=LLLA
       enddo;enddo;enddo;enddo
!2C
       do i=N0+1,N2;do j=N0+1,N2;do a=N2+1,N3;do b=N2+1,N3
        LLL=LLL+1
        ind2C(b,a,j,i)=LLL
        if(i.eq.j.or.a.eq.b)cycle
        if(numA(i,j,M3).lt.ii.or.numA1(a,b,M4).lt.ia)cycle
        LLLA=LLLA+1
        indA2C(b,a,j,i)=LLLA
        indAll(LLL)=LLLA
       enddo;enddo;enddo;enddo
       if(LLL.ne.KKK)stop 'LLL != KKK'
C       
       allocate(HM(LLLA,LLLA))
       HM=0.0d0

       print*,LLLA,LLL
       print*,ndim

       end subroutine eomccs_d

end module eomcc_initial_guess_jun
