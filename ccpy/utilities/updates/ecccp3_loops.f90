module ecccp3_loops

      implicit none

      contains


              subroutine ecccp3a(deltaintA,deltaintB,deltaintC,deltaintD,&
                              deltaextA,deltaextB,deltaextC,deltaextD,&
                              pspace,&
                              M3A,L3A,C3A,omega,&
                              fA_oo,fA_vv,H1A_oo,H1A_vv,&
                              H2A_voov,H2A_oooo,H2A_vvvv,&
                              D3A_O,D3A_V,noa,nua)

                        real(kind=8), intent(out) :: deltaintA, deltaintB, deltaintC, deltaintD
                        real(kind=8), intent(out) :: deltaextA, deltaextB, deltaextC, deltaextD
                        integer, intent(in) :: noa, nua
                        integer, intent(in) :: pspace(1:nua,1:nua,1:nua,1:noa,1:noa,1:noa)
                        real(kind=8), intent(in) :: M3A(1:nua,1:nua,1:nua,1:noa,1:noa,1:noa),&
                        L3A(1:nua,1:nua,1:nua,1:noa,1:noa,1:noa),&
                        C3A(1:nua,1:nua,1:nua,1:noa,1:noa,1:noa),&
                        fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),&
                        H1A_oo(1:noa,1:noa),H1A_vv(1:nua,1:nua),&
                        H2A_voov(1:nua,1:noa,1:noa,1:nua),&
                        H2A_oooo(1:noa,1:noa,1:noa,1:noa),&
                        H2A_vvvv(1:nua,1:nua,1:nua,1:nua),&
                        D3A_O(1:nua,1:noa,1:noa),&
                        D3A_V(1:nua,1:noa,1:nua),&
                        omega
                        integer :: i, j, k, a, b, c
                        real(kind=8) :: D, LM

                        deltaextA = 0.0d0
                        deltaextB = 0.0d0
                        deltaextC = 0.0d0
                        deltaextD = 0.0d0

                        deltaintA = 0.0d0
                        deltaintB = 0.0d0
                        deltaintC = 0.0d0
                        deltaintD = 0.0d0

                        do i = 1 , noa
                            do j = i+1, noa
                                do k = j+1, noa
                                    do a = 1, nua
                                        do b = a+1, nua
                                            do c = b+1, nua

                                                if (pspace(a, b, c, i, j, k) == 0) then ! external

                                                        LM = M3A(a,b,c,i,j,k) * L3A(a,b,c,i,j,k)

                                                        D = fA_oo(i,i) + fA_oo(j,j) + fA_oo(k,k)&
                                                        - fA_vv(a,a) - fA_vv(b,b) - fA_vv(c,c)

                                                        deltaextA = deltaextA + LM/D

                                                        D = H1A_oo(i,i) + H1A_oo(j,j) + H1A_oo(k,k)&
                                                        - H1A_vv(a,a) - H1A_vv(b,b) - H1A_vv(c,c)

                                                        deltaextB = deltaextB + LM/D

                                                        D = D &
                                                        -H2A_voov(a,i,i,a) - H2A_voov(b,i,i,b) - H2A_voov(c,i,i,c)&
                                                        -H2A_voov(a,j,j,a) - H2A_voov(b,j,j,b) - H2A_voov(c,j,j,c)&
                                                        -H2A_voov(a,k,k,a) - H2A_voov(b,k,k,b) - H2A_voov(c,k,k,c)&
                                                        -H2A_oooo(j,i,j,i) - H2A_oooo(k,i,k,i) - H2A_oooo(k,j,k,j)&
                                                        -H2A_vvvv(b,a,b,a) - H2A_vvvv(c,a,c,a) - H2A_vvvv(c,b,c,b)

                                                        deltaextC = deltaextC + LM/D

                                                        D = D &
                                                        +D3A_O(a,i,j)+D3A_O(a,i,k)+D3A_O(a,j,k)&
                                                        +D3A_O(b,i,j)+D3A_O(b,i,k)+D3A_O(b,j,k)&
                                                        +D3A_O(c,i,j)+D3A_O(c,i,k)+D3A_O(c,j,k)&
                                                        -D3A_V(a,i,b)-D3A_V(a,i,c)-D3A_V(b,i,c)&
                                                        -D3A_V(a,j,b)-D3A_V(a,j,c)-D3A_V(b,j,c)&
                                                        -D3A_V(a,k,b)-D3A_V(a,k,c)-D3A_V(b,k,c)

                                                        deltaextD = deltaextD + LM/D

                                                else ! internal

                                                        LM = M3A(a,b,c,i,j,k) * C3A(a,b,c,i,j,k)

                                                        D = fA_oo(i,i) + fA_oo(j,j) + fA_oo(k,k)&
                                                        - fA_vv(a,a) - fA_vv(b,b) - fA_vv(c,c)

                                                        deltaintA = deltaintA + LM/D

                                                        D = H1A_oo(i,i) + H1A_oo(j,j) + H1A_oo(k,k)&
                                                        - H1A_vv(a,a) - H1A_vv(b,b) - H1A_vv(c,c)

                                                        deltaintB = deltaintB + LM/D

                                                        D = D &
                                                        -H2A_voov(a,i,i,a) - H2A_voov(b,i,i,b) - H2A_voov(c,i,i,c)&
                                                        -H2A_voov(a,j,j,a) - H2A_voov(b,j,j,b) - H2A_voov(c,j,j,c)&
                                                        -H2A_voov(a,k,k,a) - H2A_voov(b,k,k,b) - H2A_voov(c,k,k,c)&
                                                        -H2A_oooo(j,i,j,i) - H2A_oooo(k,i,k,i) - H2A_oooo(k,j,k,j)&
                                                        -H2A_vvvv(b,a,b,a) - H2A_vvvv(c,a,c,a) - H2A_vvvv(c,b,c,b)

                                                        deltaintC = deltaintC + LM/D

                                                        D = D &
                                                        +D3A_O(a,i,j)+D3A_O(a,i,k)+D3A_O(a,j,k)&
                                                        +D3A_O(b,i,j)+D3A_O(b,i,k)+D3A_O(b,j,k)&
                                                        +D3A_O(c,i,j)+D3A_O(c,i,k)+D3A_O(c,j,k)&
                                                        -D3A_V(a,i,b)-D3A_V(a,i,c)-D3A_V(b,i,c)&
                                                        -D3A_V(a,j,b)-D3A_V(a,j,c)-D3A_V(b,j,c)&
                                                        -D3A_V(a,k,b)-D3A_V(a,k,c)-D3A_V(b,k,c)

                                                        deltaintD = deltaintD + LM/D

                                                end if

                                            end do
                                        end do
                                    end do
                                end do
                            end do
                        end do

              end subroutine ecccp3a

              subroutine ecccp3b(deltaintA,deltaintB,deltaintC,deltaintD,&
                              deltaextA,deltaextB,deltaextC,deltaextD,&
                              pspace,&
                              M3B,L3B,C3B,omega,&
                              fA_oo,fA_vv,fB_oo,fB_vv,&
                              H1A_oo,H1A_vv,H1B_oo,H1B_vv,&
                              H2A_voov,H2A_oooo,H2A_vvvv,&
                              H2B_ovov,H2B_vovo,&
                              H2B_oooo,H2B_vvvv,&
                              H2C_voov,&
                              D3A_O,D3A_V,D3B_O,D3B_V,D3C_O,D3C_V,&
                              noa,nua,nob,nub)

                        real(kind=8), intent(out) :: deltaintA, deltaintB, deltaintC, deltaintD
                        real(kind=8), intent(out) :: deltaextA, deltaextB, deltaextC, deltaextD
                        integer, intent(in) :: noa, nua, nob, nub
                        integer, intent(in) :: pspace(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob)
                        real(kind=8), intent(in) :: M3B(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob),&
                        L3B(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob),&
                        C3B(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob),&
                        fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),&
                        fB_oo(1:nob,1:nob),fB_vv(1:nub,1:nub),&
                        H1A_oo(1:noa,1:noa),H1A_vv(1:nua,1:nua),&
                        H1B_oo(1:nob,1:nob),H1B_vv(1:nub,1:nub),&
                        H2A_voov(1:nua,1:noa,1:noa,1:nua),&
                        H2A_oooo(1:noa,1:noa,1:noa,1:noa),&
                        H2A_vvvv(1:nua,1:nua,1:nua,1:nua),&
                        H2B_ovov(1:noa,1:nub,1:noa,1:nub),&
                        H2B_vovo(1:nua,1:nob,1:nua,1:nob),&
                        H2B_oooo(1:noa,1:nob,1:noa,1:nob),&
                        H2B_vvvv(1:nua,1:nub,1:nua,1:nub),&
                        H2C_voov(1:nub,1:nob,1:nob,1:nub),&
                        D3A_O(1:nua,1:noa,1:noa),&
                        D3A_V(1:nua,1:noa,1:nua),&
                        D3B_O(1:nua,1:noa,1:nob),&
                        D3B_V(1:nua,1:noa,1:nub),&
                        D3C_O(1:nub,1:noa,1:nob),&
                        D3C_V(1:nua,1:nob,1:nub),&
                        omega
                        integer :: i, j, k, a, b, c
                        real(kind=8) :: D, LM

                        deltaintA = 0.0d0
                        deltaintB = 0.0d0
                        deltaintC = 0.0d0
                        deltaintD = 0.0d0

                        deltaextA = 0.0d0
                        deltaextB = 0.0d0
                        deltaextC = 0.0d0
                        deltaextD = 0.0d0

                        do i = 1, noa
                            do j = i+1, noa
                                do k = 1, nob
                                    do a = 1, nua
                                        do b = a+1, nua
                                            do c = 1, nub

                                                if (pspace(a, b, c, i, j, k) == 0) then ! external

                                                        LM = M3B(a,b,c,i,j,k) * L3B(a,b,c,i,j,k)

                                                        D = fA_oo(i,i) + fA_oo(j,j) + fB_oo(k,k)&
                                                        - fA_vv(a,a) - fA_vv(b,b) - fB_vv(c,c)

                                                        deltaextA = deltaextA + LM/D

                                                        D = H1A_oo(i,i) + H1A_oo(j,j) + H1B_oo(k,k)&
                                                        - H1A_vv(a,a) - H1A_vv(b,b) - H1B_vv(c,c)

                                                        deltaextB = deltaextB + LM/D

                                                        D = D &
                                                        -H2A_voov(a,i,i,a)-H2A_voov(b,i,i,b)+H2B_ovov(i,c,i,c)&
                                                        -H2A_voov(a,j,j,a)-H2A_voov(b,j,j,b)+H2B_ovov(j,c,j,c)&
                                                        +H2B_vovo(a,k,a,k)+H2B_vovo(b,k,b,k)-H2C_voov(c,k,k,c)&
                                                        -H2A_oooo(j,i,j,i)-H2B_oooo(i,k,i,k)-H2B_oooo(j,k,j,k)&
                                                        -H2A_vvvv(b,a,b,a)-H2B_vvvv(a,c,a,c)-H2B_vvvv(b,c,b,c)

                                                        deltaextC = deltaextC + LM/D

                                                        D = D &
                                                        +D3A_O(a,i,j)+D3B_O(a,i,k)+D3B_O(a,j,k)&
                                                        +D3A_O(b,i,j)+D3B_O(b,i,k)+D3B_O(b,j,k)&
                                                        +D3C_O(c,i,k)+D3C_O(c,j,k)&
                                                        -D3A_V(a,i,b)-D3B_V(a,i,c)-D3B_V(b,i,c)&
                                                        -D3A_V(a,j,b)-D3B_V(a,j,c)-D3B_V(b,j,c)&
                                                        -D3C_V(a,k,c)-D3C_V(b,k,c)

                                                        deltaextD = deltaextD + LM/D

                                                else ! internal

                                                        LM = M3B(a,b,c,i,j,k) * C3B(a,b,c,i,j,k)

                                                        D = fA_oo(i,i) + fA_oo(j,j) + fB_oo(k,k)&
                                                        - fA_vv(a,a) - fA_vv(b,b) - fB_vv(c,c)

                                                        deltaintA = deltaintA + LM/D

                                                        D = H1A_oo(i,i) + H1A_oo(j,j) + H1B_oo(k,k)&
                                                        - H1A_vv(a,a) - H1A_vv(b,b) - H1B_vv(c,c)

                                                        deltaintB = deltaintB + LM/D

                                                        D = D &
                                                        -H2A_voov(a,i,i,a)-H2A_voov(b,i,i,b)+H2B_ovov(i,c,i,c)&
                                                        -H2A_voov(a,j,j,a)-H2A_voov(b,j,j,b)+H2B_ovov(j,c,j,c)&
                                                        +H2B_vovo(a,k,a,k)+H2B_vovo(b,k,b,k)-H2C_voov(c,k,k,c)&
                                                        -H2A_oooo(j,i,j,i)-H2B_oooo(i,k,i,k)-H2B_oooo(j,k,j,k)&
                                                        -H2A_vvvv(b,a,b,a)-H2B_vvvv(a,c,a,c)-H2B_vvvv(b,c,b,c)

                                                        deltaintC = deltaintC + LM/D

                                                        D = D &
                                                        +D3A_O(a,i,j)+D3B_O(a,i,k)+D3B_O(a,j,k)&
                                                        +D3A_O(b,i,j)+D3B_O(b,i,k)+D3B_O(b,j,k)&
                                                        +D3C_O(c,i,k)+D3C_O(c,j,k)&
                                                        -D3A_V(a,i,b)-D3B_V(a,i,c)-D3B_V(b,i,c)&
                                                        -D3A_V(a,j,b)-D3B_V(a,j,c)-D3B_V(b,j,c)&
                                                        -D3C_V(a,k,c)-D3C_V(b,k,c)

                                                        deltaintD = deltaintD + LM/D

                                                end if

                                            end do
                                        end do
                                    end do
                                end do
                            end do
                        end do

              end subroutine ecccp3b

              subroutine ecccp3c(deltaintA,deltaintB,deltaintC,deltaintD,&
                              deltaextA,deltaextB,deltaextC,deltaextD,&
                              pspace,&
                              M3C,L3C,C3C,omega,&
                              fA_oo,fA_vv,fB_oo,fB_vv,&
                              H1A_oo,H1A_vv,H1B_oo,H1B_vv,&
                              H2A_voov,&
                              H2B_ovov,H2B_vovo,&
                              H2B_oooo,H2B_vvvv,&
                              H2C_voov,H2C_oooo,H2C_vvvv,&
                              D3B_O,D3B_V,D3C_O,D3C_V,D3D_O,D3D_V,&
                              noa,nua,nob,nub)

                        real(kind=8), intent(out) :: deltaintA, deltaintB, deltaintC, deltaintD
                        real(kind=8), intent(out) :: deltaextA, deltaextB, deltaextC, deltaextD
                        integer, intent(in) :: noa, nua, nob, nub
                        integer, intent(in) :: pspace(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob)
                        real(kind=8), intent(in) :: M3C(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob),&
                        L3C(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob),&
                        C3C(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob),&
                        fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),&
                        fB_oo(1:nob,1:nob),fB_vv(1:nub,1:nub),&
                        H1A_oo(1:noa,1:noa),H1A_vv(1:nua,1:nua),&
                        H1B_oo(1:nob,1:nob),H1B_vv(1:nub,1:nub),&
                        H2A_voov(1:nua,1:noa,1:noa,1:nua),&
                        H2B_ovov(1:noa,1:nub,1:noa,1:nub),&
                        H2B_vovo(1:nua,1:nob,1:nua,1:nob),&
                        H2B_oooo(1:noa,1:nob,1:noa,1:nob),&
                        H2B_vvvv(1:nua,1:nub,1:nua,1:nub),&
                        H2C_voov(1:nub,1:nob,1:nob,1:nub),&
                        H2C_oooo(1:nob,1:nob,1:nob,1:nob),&
                        H2C_vvvv(1:nub,1:nub,1:nub,1:nub),&
                        D3B_O(1:nua,1:noa,1:nob),&
                        D3B_V(1:nua,1:noa,1:nub),&
                        D3C_O(1:nub,1:noa,1:nob),&
                        D3C_V(1:nua,1:nob,1:nub),&
                        D3D_O(1:nub,1:nob,1:nob),&
                        D3D_V(1:nub,1:nob,1:nub),&
                        omega
                        integer :: i, j, k, a, b, c
                        real(kind=8) :: D, temp

                        deltaintA = 0.0d0
                        deltaintB = 0.0d0
                        deltaintC = 0.0d0
                        deltaintD = 0.0d0

                        deltaextA = 0.0d0
                        deltaextB = 0.0d0
                        deltaextC = 0.0d0
                        deltaextD = 0.0d0

                        do i = 1 , noa
                            do j = 1, nob
                                do k = j+1, nob
                                    do a = 1, nua
                                        do b = 1, nub
                                            do c = b+1, nub

                                                if (pspace(a, b, c, i, j, k) == 0) then ! external

                                                        temp = M3C(a,b,c,i,j,k) * L3C(a,b,c,i,j,k)

                                                        D = fA_oo(i,i) + fB_oo(j,j) + fB_oo(k,k)&
                                                        - fA_vv(a,a) - fB_vv(b,b) - fB_vv(c,c)

                                                        deltaextA = deltaextA + temp/(omega+D)

                                                        D = H1A_oo(i,i) + H1B_oo(j,j) + H1B_oo(k,k)&
                                                        - H1A_vv(a,a) - H1B_vv(b,b) - H1B_vv(c,c)

                                                        deltaextB = deltaextB + temp/(omega+D)

                                                        D = D &
                                                        -H2A_voov(a,i,i,a)+H2B_ovov(i,b,i,b)+H2B_ovov(i,c,i,c)&
                                                        +H2B_vovo(a,j,a,j)-H2C_voov(b,j,j,b)-H2C_voov(c,j,j,c)&
                                                        +H2B_vovo(a,k,a,k)-H2C_voov(b,k,k,b)-H2C_voov(c,k,k,c)&
                                                        -H2B_oooo(i,j,i,j)-H2B_oooo(i,k,i,k)-H2C_oooo(k,j,k,j)&
                                                        -H2B_vvvv(a,b,a,b)-H2B_vvvv(a,c,a,c)-H2C_vvvv(c,b,c,b)

                                                        deltaextC = deltaextC + temp/(omega+D)

                                                        D = D &
                                                        +D3B_O(a,i,j)+D3B_O(a,i,k)&
                                                        +D3C_O(b,i,j)+D3C_O(b,i,k)+D3D_O(b,j,k)&
                                                        +D3C_O(c,i,j)+D3C_O(c,i,k)+D3D_O(c,j,k)&
                                                        -D3B_V(a,i,b)-D3B_V(a,i,c)&
                                                        -D3C_V(a,j,b)-D3C_V(a,j,c)-D3D_V(b,j,c)&
                                                        -D3C_V(a,k,b)-D3C_V(a,k,c)-D3D_V(b,k,c)

                                                        deltaextD = deltaextD + temp/(omega+D)

                                                else ! internal

                                                        temp = M3C(a,b,c,i,j,k) * C3C(a,b,c,i,j,k)

                                                        D = fA_oo(i,i) + fB_oo(j,j) + fB_oo(k,k)&
                                                        - fA_vv(a,a) - fB_vv(b,b) - fB_vv(c,c)

                                                        deltaintA = deltaintA + temp/(omega+D)

                                                        D = H1A_oo(i,i) + H1B_oo(j,j) + H1B_oo(k,k)&
                                                        - H1A_vv(a,a) - H1B_vv(b,b) - H1B_vv(c,c)

                                                        deltaintB = deltaintB + temp/(omega+D)

                                                        D = D &
                                                        -H2A_voov(a,i,i,a)+H2B_ovov(i,b,i,b)+H2B_ovov(i,c,i,c)&
                                                        +H2B_vovo(a,j,a,j)-H2C_voov(b,j,j,b)-H2C_voov(c,j,j,c)&
                                                        +H2B_vovo(a,k,a,k)-H2C_voov(b,k,k,b)-H2C_voov(c,k,k,c)&
                                                        -H2B_oooo(i,j,i,j)-H2B_oooo(i,k,i,k)-H2C_oooo(k,j,k,j)&
                                                        -H2B_vvvv(a,b,a,b)-H2B_vvvv(a,c,a,c)-H2C_vvvv(c,b,c,b)

                                                        deltaintC = deltaintC + temp/(omega+D)

                                                        D = D &
                                                        +D3B_O(a,i,j)+D3B_O(a,i,k)&
                                                        +D3C_O(b,i,j)+D3C_O(b,i,k)+D3D_O(b,j,k)&
                                                        +D3C_O(c,i,j)+D3C_O(c,i,k)+D3D_O(c,j,k)&
                                                        -D3B_V(a,i,b)-D3B_V(a,i,c)&
                                                        -D3C_V(a,j,b)-D3C_V(a,j,c)-D3D_V(b,j,c)&
                                                        -D3C_V(a,k,b)-D3C_V(a,k,c)-D3D_V(b,k,c)

                                                        deltaintD = deltaintD + temp/(omega+D)
                                                end if

                                            end do
                                        end do
                                    end do
                                end do
                            end do
                        end do

              end subroutine ecccp3c

              subroutine ecccp3d(deltaintA,deltaintB,deltaintC,deltaintD,&
                              deltaextA,deltaextB,deltaextC,deltaextD,&
                              pspace,&
                              M3D,L3D,C3D,omega,&
                              fB_oo,fB_vv,H1B_oo,H1B_vv,&
                              H2C_voov,H2C_oooo,H2C_vvvv,&
                              D3D_O,D3D_V,nob,nub)

                        real(kind=8), intent(out) :: deltaintA, deltaintB, deltaintC, deltaintD
                        real(kind=8), intent(out) :: deltaextA, deltaextB, deltaextC, deltaextD
                        integer, intent(in) :: nob, nub
                        integer, intent(in) :: pspace(1:nub,1:nub,1:nub,1:nob,1:nob,1:nob)
                        real(kind=8), intent(in) :: M3D(1:nub,1:nub,1:nub,1:nob,1:nob,1:nob),&
                        L3D(1:nub,1:nub,1:nub,1:nob,1:nob,1:nob),&
                        C3D(1:nub,1:nub,1:nub,1:nob,1:nob,1:nob),&
                        fB_oo(1:nob,1:nob),fB_vv(1:nub,1:nub),&
                        H1B_oo(1:nob,1:nob),H1B_vv(1:nub,1:nub),&
                        H2C_voov(1:nub,1:nob,1:nob,1:nub),&
                        H2C_oooo(1:nob,1:nob,1:nob,1:nob),&
                        H2C_vvvv(1:nub,1:nub,1:nub,1:nub),&
                        D3D_O(1:nub,1:nob,1:nob),&
                        D3D_V(1:nub,1:nob,1:nub),&
                        omega
                        integer :: i, j, k, a, b, c
                        real(kind=8) :: D, temp

                        deltaintA = 0.0d0
                        deltaintB = 0.0d0
                        deltaintC = 0.0d0
                        deltaintD = 0.0d0

                        deltaextA = 0.0d0
                        deltaextB = 0.0d0
                        deltaextC = 0.0d0
                        deltaextD = 0.0d0

                        do i = 1 , nob
                            do j = i+1, nob
                                do k = j+1, nob
                                    do a = 1, nub
                                        do b = a+1, nub
                                            do c = b+1, nub

                                                if (pspace(a, b, c, i, j, k) == 0) then ! external

                                                        temp = M3D(a,b,c,i,j,k) * L3D(a,b,c,i,j,k)

                                                        D = fB_oo(i,i) + fB_oo(j,j) + fB_oo(k,k)&
                                                        - fB_vv(a,a) - fB_vv(b,b) - fB_vv(c,c)

                                                        deltaextA = deltaextA + temp/(omega+D)

                                                        D = H1B_oo(i,i) + H1B_oo(j,j) + H1B_oo(k,k)&
                                                        - H1B_vv(a,a) - H1B_vv(b,b) - H1B_vv(c,c)

                                                        deltaextB = deltaextB + temp/(omega+D)

                                                        D = D &
                                                        -H2C_voov(a,i,i,a) - H2C_voov(b,i,i,b) - H2C_voov(c,i,i,c)&
                                                        -H2C_voov(a,j,j,a) - H2C_voov(b,j,j,b) - H2C_voov(c,j,j,c)&
                                                        -H2C_voov(a,k,k,a) - H2C_voov(b,k,k,b) - H2C_voov(c,k,k,c)&
                                                        -H2C_oooo(j,i,j,i) - H2C_oooo(k,i,k,i) - H2C_oooo(k,j,k,j)&
                                                        -H2C_vvvv(b,a,b,a) - H2C_vvvv(c,a,c,a) - H2C_vvvv(c,b,c,b)

                                                        deltaextC = deltaextC + temp/(omega+D)

                                                        D = D &
                                                        +D3D_O(a,i,j)+D3D_O(a,i,k)+D3D_O(a,j,k)&
                                                        +D3D_O(b,i,j)+D3D_O(b,i,k)+D3D_O(b,j,k)&
                                                        +D3D_O(c,i,j)+D3D_O(c,i,k)+D3D_O(c,j,k)&
                                                        -D3D_V(a,i,b)-D3D_V(a,i,c)-D3D_V(b,i,c)&
                                                        -D3D_V(a,j,b)-D3D_V(a,j,c)-D3D_V(b,j,c)&
                                                        -D3D_V(a,k,b)-D3D_V(a,k,c)-D3D_V(b,k,c)

                                                        deltaextD = deltaextD + temp/(omega+D)

                                                else ! internal

                                                        temp = M3D(a,b,c,i,j,k) * C3D(a,b,c,i,j,k)

                                                        D = fB_oo(i,i) + fB_oo(j,j) + fB_oo(k,k)&
                                                        - fB_vv(a,a) - fB_vv(b,b) - fB_vv(c,c)

                                                        deltaintA = deltaintA + temp/(omega+D)

                                                        D = H1B_oo(i,i) + H1B_oo(j,j) + H1B_oo(k,k)&
                                                        - H1B_vv(a,a) - H1B_vv(b,b) - H1B_vv(c,c)

                                                        deltaintB = deltaintB + temp/(omega+D)

                                                        D = D &
                                                        -H2C_voov(a,i,i,a) - H2C_voov(b,i,i,b) - H2C_voov(c,i,i,c)&
                                                        -H2C_voov(a,j,j,a) - H2C_voov(b,j,j,b) - H2C_voov(c,j,j,c)&
                                                        -H2C_voov(a,k,k,a) - H2C_voov(b,k,k,b) - H2C_voov(c,k,k,c)&
                                                        -H2C_oooo(j,i,j,i) - H2C_oooo(k,i,k,i) - H2C_oooo(k,j,k,j)&
                                                        -H2C_vvvv(b,a,b,a) - H2C_vvvv(c,a,c,a) - H2C_vvvv(c,b,c,b)

                                                        deltaintC = deltaintC + temp/(omega+D)

                                                        D = D &
                                                        +D3D_O(a,i,j)+D3D_O(a,i,k)+D3D_O(a,j,k)&
                                                        +D3D_O(b,i,j)+D3D_O(b,i,k)+D3D_O(b,j,k)&
                                                        +D3D_O(c,i,j)+D3D_O(c,i,k)+D3D_O(c,j,k)&
                                                        -D3D_V(a,i,b)-D3D_V(a,i,c)-D3D_V(b,i,c)&
                                                        -D3D_V(a,j,b)-D3D_V(a,j,c)-D3D_V(b,j,c)&
                                                        -D3D_V(a,k,b)-D3D_V(a,k,c)-D3D_V(b,k,c)

                                                        deltaintD = deltaintD + temp/(omega+D)
                                                end if

                                            end do
                                        end do
                                    end do
                                end do
                            end do
                        end do

              end subroutine ecccp3d


end module ecccp3_loops
