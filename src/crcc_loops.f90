module crcc_loops

      implicit none

      contains

              subroutine crcc23A(deltaA,deltaB,deltaC,deltaD,&
                              MM23A,L3A,omega,&
                              fA_oo,fA_vv,H1A_oo,H1A_vv,&
                              H2A_voov,H2A_oooo,H2A_vvvv,&
                              D3A_O,D3A_V,noa,nua)
                        
                        real(kind=8), intent(out) :: deltaA, deltaB, deltaC, deltaD
                        integer, intent(in) :: noa, nua
                        real(kind=8), intent(in) :: MM23A(1:nua,1:nua,1:nua,1:noa,1:noa,1:noa),&
                        L3A(1:nua,1:nua,1:nua,1:noa,1:noa,1:noa),&
                        fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),&
                        H1A_oo(1:noa,1:noa),H1A_vv(1:nua,1:nua),&
                        H2A_voov(1:nua,1:noa,1:noa,1:nua),&
                        H2A_oooo(1:noa,1:noa,1:noa,1:noa),&
                        H2A_vvvv(1:nua,1:nua,1:nua,1:nua),&
                        D3A_O(1:nua,1:noa,1:noa),&
                        D3A_V(1:nua,1:noa,1:nua),&
                        omega
                        integer :: i, j, k, a, b, c
                        real(kind=8) :: D, temp

                        deltaA = 0.0
                        deltaB = 0.0
                        deltaC = 0.0
                        deltaD = 0.0

                        do i = 1 , noa
                            do j = i+1, noa
                                do k = j+1, noa
                                    do a = 1, nua
                                        do b = a+1, nua
                                            do c = b+1, nua
                                       

                                                temp = MM23A(a,b,c,i,j,k) * L3A(a,b,c,i,j,k)

                                                D = fA_oo(i,i) + fA_oo(j,j) + fA_oo(k,k)&
                                                - fA_vv(a,a) - fA_vv(b,b) - fA_vv(c,c)

                                                deltaA = deltaA + temp/(omega+D)

                                                D = H1A_oo(i,i) + H1A_oo(j,j) + H1A_oo(k,k)&
                                                - H1A_vv(a,a) - H1A_vv(b,b) - H1A_vv(c,c)

                                                deltaB = deltaB + temp/(omega+D)

                                                D = D &
                                                -H2A_voov(a,i,i,a) - H2A_voov(b,i,i,b) - H2A_voov(c,i,i,c)&
                                                -H2A_voov(a,j,j,a) - H2A_voov(b,j,j,b) - H2A_voov(c,j,j,c)&
                                                -H2A_voov(a,k,k,a) - H2A_voov(b,k,k,b) - H2A_voov(c,k,k,c)&
                                                -H2A_oooo(j,i,j,i) - H2A_oooo(k,i,k,i) - H2A_oooo(k,j,k,j)&
                                                -H2A_vvvv(b,a,b,a) - H2A_vvvv(c,a,c,a) - H2A_vvvv(c,b,c,b)

                                                deltaC = deltaC + temp/(omega+D)

                                                D = D &
                                                +D3A_O(a,i,j)+D3A_O(a,i,k)+D3A_O(a,j,k)&
                                                +D3A_O(b,i,j)+D3A_O(b,i,k)+D3A_O(b,j,k)&
                                                +D3A_O(c,i,j)+D3A_O(c,i,k)+D3A_O(c,j,k)&
                                                -D3A_V(a,i,b)-D3A_V(a,i,c)-D3A_V(b,i,c)&
                                                -D3A_V(a,j,b)-D3A_V(a,j,c)-D3A_V(b,j,c)&
                                                -D3A_V(a,k,b)-D3A_V(a,k,c)-D3A_V(b,k,c)

                                                deltaD = deltaD + temp/(omega+D)

                                            end do
                                        end do 
                                    end do 
                                end do 
                            end do 
                        end do

              end subroutine crcc23A

              subroutine crcc23B(deltaA,deltaB,deltaC,deltaD,&
                              MM23B,L3B,omega,&
                              fA_oo,fA_vv,fB_oo,fB_vv,&
                              H1A_oo,H1A_vv,H1B_oo,H1B_vv,&
                              H2A_voov,H2A_oooo,H2A_vvvv,&
                              H2B_ovov,H2B_vovo,&
                              H2B_oooo,H2B_vvvv,&
                              H2C_voov,&
                              D3A_O,D3A_V,D3B_O,D3B_V,D3C_O,D3C_V,&
                              noa,nua,nob,nub)
                        
                        real(kind=8), intent(out) :: deltaA, deltaB, deltaC, deltaD
                        integer, intent(in) :: noa, nua, nob, nub
                        real(kind=8), intent(in) :: MM23B(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob),&
                        L3B(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob),&
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
                        real(kind=8) :: D, temp

                        deltaA = 0.0
                        deltaB = 0.0
                        deltaC = 0.0
                        deltaD = 0.0

                        do i = 1, noa
                            do j = i+1, noa
                                do k = 1, nob
                                    do a = 1, nua
                                        do b = a+1, nua
                                            do c = 1, nub

                                                temp = MM23B(a,b,c,i,j,k) * L3B(a,b,c,i,j,k)

                                                D = fA_oo(i,i) + fA_oo(j,j) + fB_oo(k,k)&
                                                - fA_vv(a,a) - fA_vv(b,b) - fB_vv(c,c)

                                                deltaA = deltaA + temp/(omega+D)

                                                D = H1A_oo(i,i) + H1A_oo(j,j) + H1B_oo(k,k)&
                                                - H1A_vv(a,a) - H1A_vv(b,b) - H1B_vv(c,c)

                                                deltaB = deltaB + temp/(omega+D)

                                                D = D &
                                                -H2A_voov(a,i,i,a)-H2A_voov(b,i,i,b)+H2B_ovov(i,c,i,c)&
                                                -H2A_voov(a,j,j,a)-H2A_voov(b,j,j,b)+H2B_ovov(j,c,j,c)&
                                                +H2B_vovo(a,k,a,k)+H2B_vovo(b,k,b,k)-H2C_voov(c,k,k,c)&
                                                -H2A_oooo(j,i,j,i)-H2B_oooo(i,k,i,k)-H2B_oooo(j,k,j,k)&
                                                -H2A_vvvv(b,a,b,a)-H2B_vvvv(a,c,a,c)-H2B_vvvv(b,c,b,c)
     
                                                deltaC = deltaC + temp/(omega+D)

                                                D = D &
                                                +D3A_O(a,i,j)+D3B_O(a,i,k)+D3B_O(a,j,k)&
                                                +D3A_O(b,i,j)+D3B_O(b,i,k)+D3B_O(b,j,k)&
                                                +D3C_O(c,i,k)+D3C_O(c,j,k)&
                                                -D3A_V(a,i,b)-D3B_V(a,i,c)-D3B_V(b,i,c)&
                                                -D3A_V(a,j,b)-D3B_V(a,j,c)-D3B_V(b,j,c)&
                                                -D3C_V(a,k,c)-D3C_V(b,k,c)

                                                deltaD = deltaD + temp/(omega+D)

                                            end do
                                        end do 
                                    end do 
                                end do 
                            end do 
                        end do

              end subroutine crcc23B

              subroutine crcc23C(deltaA,deltaB,deltaC,deltaD,&
                              MM23C,L3C,omega,&
                              fA_oo,fA_vv,fB_oo,fB_vv,&
                              H1A_oo,H1A_vv,H1B_oo,H1B_vv,&
                              H2A_voov,&
                              H2B_ovov,H2B_vovo,&
                              H2B_oooo,H2B_vvvv,&
                              H2C_voov,H2C_oooo,H2C_vvvv,&
                              D3B_O,D3B_V,D3C_O,D3C_V,D3D_O,D3D_V,&
                              noa,nua,nob,nub)
                        
                        real(kind=8), intent(out) :: deltaA, deltaB, deltaC, deltaD
                        integer, intent(in) :: noa, nua, nob, nub
                        real(kind=8), intent(in) :: MM23C(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob),&
                        L3C(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob),&
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

                        deltaA = 0.0
                        deltaB = 0.0
                        deltaC = 0.0
                        deltaD = 0.0

                        do i = 1 , noa
                            do j = 1, nob
                                do k = j+1, nob
                                    do a = 1, nua
                                        do b = 1, nub
                                            do c = b+1, nub

                                                temp = MM23C(a,b,c,i,j,k) * L3C(a,b,c,i,j,k)

                                                D = fA_oo(i,i) + fB_oo(j,j) + fB_oo(k,k)&
                                                - fA_vv(a,a) - fB_vv(b,b) - fB_vv(c,c)

                                                deltaA = deltaA + temp/(omega+D)

                                                D = H1A_oo(i,i) + H1B_oo(j,j) + H1B_oo(k,k)&
                                                - H1A_vv(a,a) - H1B_vv(b,b) - H1B_vv(c,c)

                                                deltaB = deltaB + temp/(omega+D)

                                                D = D &
                                                -H2A_voov(a,i,i,a)+H2B_ovov(i,b,i,b)+H2B_ovov(i,c,i,c)&
                                                +H2B_vovo(a,j,a,j)-H2C_voov(b,j,j,b)-H2C_voov(c,j,j,c)&
                                                +H2B_vovo(a,k,a,k)-H2C_voov(b,k,k,b)-H2C_voov(c,k,k,c)&
                                                -H2B_oooo(i,j,i,j)-H2B_oooo(i,k,i,k)-H2C_oooo(k,j,k,j)&
                                                -H2B_vvvv(a,b,a,b)-H2B_vvvv(a,c,a,c)-H2C_vvvv(c,b,c,b)
     
                                                deltaC = deltaC + temp/(omega+D)
                                                D = D &
                                                +D3B_O(a,i,j)+D3B_O(a,i,k)&
                                                +D3C_O(b,i,j)+D3C_O(b,i,k)+D3D_O(b,j,k)&
                                                +D3C_O(c,i,j)+D3C_O(c,i,k)+D3D_O(c,j,k)&
                                                -D3B_V(a,i,b)-D3B_V(a,i,c)&
                                                -D3C_V(a,j,b)-D3C_V(a,j,c)-D3D_V(b,j,c)&
                                                -D3C_V(a,k,b)-D3C_V(a,k,c)-D3D_V(b,k,c)

                                                deltaD = deltaD + temp/(omega+D)

                                            end do
                                        end do 
                                    end do 
                                end do 
                            end do 
                        end do

              end subroutine crcc23C

              subroutine crcc23D(deltaA,deltaB,deltaC,deltaD,&
                              MM23D,L3D,omega,&
                              fB_oo,fB_vv,H1B_oo,H1B_vv,&
                              H2C_voov,H2C_oooo,H2C_vvvv,&
                              D3D_O,D3D_V,nob,nub)
                        
                        real(kind=8), intent(out) :: deltaA, deltaB, deltaC, deltaD
                        integer, intent(in) :: nob, nub
                        real(kind=8), intent(in) :: MM23D(1:nub,1:nub,1:nub,1:nob,1:nob,1:nob),&
                        L3D(1:nub,1:nub,1:nub,1:nob,1:nob,1:nob),&
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

                        deltaA = 0.0
                        deltaB = 0.0
                        deltaC = 0.0
                        deltaD = 0.0

                        do i = 1 , nob
                            do j = i+1, nob
                                do k = j+1, nob
                                    do a = 1, nub
                                        do b = a+1, nub
                                            do c = b+1, nub

                                                temp = MM23D(a,b,c,i,j,k) * L3D(a,b,c,i,j,k)

                                                D = fB_oo(i,i) + fB_oo(j,j) + fB_oo(k,k)&
                                                - fB_vv(a,a) - fB_vv(b,b) - fB_vv(c,c)

                                                deltaA = deltaA + temp/(omega+D)

                                                D = H1B_oo(i,i) + H1B_oo(j,j) + H1B_oo(k,k)&
                                                - H1B_vv(a,a) - H1B_vv(b,b) - H1B_vv(c,c)

                                                deltaB = deltaB + temp/(omega+D)

                                                D = D &
                                                -H2C_voov(a,i,i,a) - H2C_voov(b,i,i,b) - H2C_voov(c,i,i,c)&
                                                -H2C_voov(a,j,j,a) - H2C_voov(b,j,j,b) - H2C_voov(c,j,j,c)&
                                                -H2C_voov(a,k,k,a) - H2C_voov(b,k,k,b) - H2C_voov(c,k,k,c)&
                                                -H2C_oooo(j,i,j,i) - H2C_oooo(k,i,k,i) - H2C_oooo(k,j,k,j)&
                                                -H2C_vvvv(b,a,b,a) - H2C_vvvv(c,a,c,a) - H2C_vvvv(c,b,c,b)

                                                deltaC = deltaC + temp/(omega+D)

                                                D = D &
                                                +D3D_O(a,i,j)+D3D_O(a,i,k)+D3D_O(a,j,k)&
                                                +D3D_O(b,i,j)+D3D_O(b,i,k)+D3D_O(b,j,k)&
                                                +D3D_O(c,i,j)+D3D_O(c,i,k)+D3D_O(c,j,k)&
                                                -D3D_V(a,i,b)-D3D_V(a,i,c)-D3D_V(b,i,c)&
                                                -D3D_V(a,j,b)-D3D_V(a,j,c)-D3D_V(b,j,c)&
                                                -D3D_V(a,k,b)-D3D_V(a,k,c)-D3D_V(b,k,c)

                                                deltaD = deltaD + temp/(omega+D)

                                            end do
                                        end do 
                                    end do 
                                end do 
                            end do 
                        end do

              end subroutine crcc23D

              subroutine crcc24A(deltaA,deltaB,deltaC,deltaD,&
                              MM24A,L4A,&
                              fA_oo,fA_vv,H1A_oo,H1A_vv,&
                              H2A_voov,H2A_oooo,H2A_vvvv,&
                              D3A_O,D3A_V,&
                              noa,nua)
                        
                        real(kind=8), intent(out) :: deltaA, deltaB, deltaC, deltaD
                        integer, intent(in) :: noa, nua
                        real(kind=8), intent(in) :: MM24A(1:nua,1:nua,1:nua,1:nua,1:noa,1:noa,1:noa,1:noa),&
                        L4A(1:nua,1:nua,1:nua,1:nua,1:noa,1:noa,1:noa,1:noa),&
                        fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),&
                        H1A_oo(1:noa,1:noa),H1A_vv(1:nua,1:nua),&
                        H2A_voov(1:nua,1:noa,1:noa,1:nua),&
                        H2A_oooo(1:noa,1:noa,1:noa,1:noa),&
                        H2A_vvvv(1:nua,1:nua,1:nua,1:nua),&
                        D3A_O(1:nua,1:noa,1:noa),&
                        D3A_V(1:nua,1:noa,1:nua)
                        integer :: i, j, k, l, a, b, c, d, ii, jj, kk, ll, aa, bb, cc, dd
                        real(kind=8) :: denom, temp

                        deltaA = 0.0
                        deltaB = 0.0
                        deltaC = 0.0
                        deltaD = 0.0

                        do ii = 1 , noa
                            do jj = ii+1, noa
                                do kk = jj+1, noa
                                    do ll = kk+1, noa
                                        do aa = 1, nua
                                            do bb = aa+1, nua
                                                do cc = bb+1, nua
                                                    do dd = cc+1, nua

                                                        a = dd; b = cc; c = bb;
                                                        d = aa;
                                                        i = ll; j = kk; k = jj;
                                                        l = ii;

                                                        temp = MM24A(a,b,c,d,i,j,k,l) * L4A(a,b,c,d,i,j,k,l)

                                                        denom = fA_oo(i,i) + fA_oo(j,j) + fA_oo(k,k) + fA_oo(l,l)&
                                                        - fA_vv(a,a) - fA_vv(b,b) - fA_vv(c,c) - fA_vv(d,d)

                                                        deltaA = deltaA + temp/denom

                                                        denom = H1A_oo(i,i) + H1A_oo(j,j) + H1A_oo(k,k) + H1A_oo(l,l)&
                                                        - H1A_vv(a,a) - H1A_vv(b,b) - H1A_vv(c,c) - H1A_vv(d,d)

                                                        deltaB = deltaB + temp/denom

                                                        denom = denom &
                                                        -H2A_oooo(j,i,j,i)-H2A_oooo(k,i,k,i)-H2A_oooo(l,i,l,i)&
                                                        -H2A_oooo(k,j,k,j)-H2A_oooo(l,j,l,j)-H2A_oooo(l,k,l,k)&
                                                        -H2A_voov(a,i,i,a)-H2A_voov(a,j,j,a)-H2A_voov(a,k,k,a)&
                                                        -H2A_voov(a,l,l,a)-H2A_voov(b,i,i,b)-H2A_voov(b,j,j,b)&
                                                        -H2A_voov(b,k,k,b)-H2A_voov(b,l,l,b)-H2A_voov(c,i,i,c)&
                                                        -H2A_voov(c,j,j,c)-H2A_voov(c,k,k,c)-H2A_voov(c,l,l,c)&
                                                        -H2A_voov(d,i,i,d)-H2A_voov(d,j,j,d)-H2A_voov(d,k,k,d)&
                                                        -H2A_voov(d,l,l,d)-H2A_vvvv(a,b,a,b)-H2A_vvvv(a,c,a,c)&
                                                        -H2A_vvvv(a,d,a,d)-H2A_vvvv(b,c,b,c)-H2A_vvvv(b,d,b,d)&
                                                        -H2A_vvvv(c,d,c,d)

                                                        deltaC = deltaC + temp/denom

                                                        denom = denom &
                                                        +D3A_O(a,i,j)+D3A_O(a,i,k)+D3A_O(a,i,l)+D3A_O(a,j,k)&
                                                        +D3A_O(a,j,l)+D3A_O(a,k,l)+D3A_O(b,i,j)+D3A_O(b,i,k)&
                                                        +D3A_O(b,i,l)+D3A_O(b,j,k)+D3A_O(b,j,l)+D3A_O(b,k,l)&
                                                        +D3A_O(c,i,j)+D3A_O(c,i,k)+D3A_O(c,i,l)+D3A_O(c,j,k)&
                                                        +D3A_O(c,j,l)+D3A_O(c,k,l)+D3A_O(d,i,j)+D3A_O(d,i,k)&
                                                        +D3A_O(d,i,l)+D3A_O(d,j,k)+D3A_O(d,j,l)+D3A_O(d,k,l)&
                                                        -D3A_V(a,i,b)-D3A_V(a,j,b)-D3A_V(a,k,b)-D3A_V(a,l,b)&
                                                        -D3A_V(a,i,c)-D3A_V(a,j,c)-D3A_V(a,k,c)-D3A_V(a,l,c)&
                                                        -D3A_V(a,i,d)-D3A_V(a,j,d)-D3A_V(a,k,d)-D3A_V(a,l,d)&
                                                        -D3A_V(b,i,c)-D3A_V(b,j,c)-D3A_V(b,k,c)-D3A_V(b,l,c)&
                                                        -D3A_V(b,i,d)-D3A_V(b,j,d)-D3A_V(b,k,d)-D3A_V(b,l,d)&
                                                        -D3A_V(c,i,d)-D3A_V(c,j,d)-D3A_V(c,k,d)-D3A_V(c,l,d)

                                                        deltaD = deltaD + temp/denom

                                                    end do 
                                                end do 
                                            end do
                                        end do 
                                    end do 
                                end do 
                            end do 
                        end do

              end subroutine crcc24A

              subroutine crcc24B(deltaA,deltaB,deltaC,deltaD,&
                              MM24B,L4B,&
                              fA_oo,fA_vv,fB_oo,fB_vv,&
                              H1A_oo,H1A_vv,H1B_oo,H1B_vv,&
                              H2A_voov, H2A_oooo, H2A_vvvv,&
                              H2B_ovov,H2B_vovo,H2B_oooo,H2B_vvvv,&
                              H2C_voov,&
                              D3A_O,D3A_V,D3B_O,D3B_V,D3C_O,D3C_V,&
                              noa,nua,nob,nub)
                        
                        real(kind=8), intent(out) :: deltaA, deltaB, deltaC, deltaD
                        integer, intent(in) :: noa, nua, nob, nub
                        real(kind=8), intent(in) :: MM24B(1:nua,1:nua,1:nua,1:nub,1:noa,1:noa,1:noa,1:nob),&
                        L4B(1:nua,1:nua,1:nua,1:nub,1:noa,1:noa,1:noa,1:nob),&
                        fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),fB_oo(1:nob,1:nob),fB_vv(1:nub,1:nub),&
                        H1A_oo(1:noa,1:noa),H1A_vv(1:nua,1:nua),H1B_oo(1:nob,1:nob),H1B_vv(1:nub,1:nub),&
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
                        D3C_V(1:nua,1:nob,1:nub)
                        integer :: i, j, k, l, a, b, c, d, ii, jj, kk, ll, aa, bb, cc, dd
                        real(kind=8) :: denom, temp

                        deltaA = 0.0
                        deltaB = 0.0
                        deltaC = 0.0
                        deltaD = 0.0

                        do ii = 1 , noa
                            do jj = ii+1, noa
                                do kk = jj+1, noa
                                    do ll = 1, nob
                                        do aa = 1, nua
                                            do bb = aa+1, nua
                                                do cc = bb+1, nua
                                                    do dd = 1, nub

                                                        a = cc; b = bb; c = aa;
                                                        d = dd;
                                                        i = kk; j = jj; k = ii;
                                                        l = ll;

                                                        temp = MM24B(a,b,c,d,i,j,k,l) * L4B(a,b,c,d,i,j,k,l)

                                                        denom = fA_oo(i,i) + fA_oo(j,j) + fA_oo(k,k) + fB_oo(l,l)&
                                                        - fA_vv(a,a) - fA_vv(b,b) - fA_vv(c,c) - fB_vv(d,d)

                                                        deltaA = deltaA + temp/denom

                                                        denom = H1A_oo(i,i) + H1A_oo(j,j) + H1A_oo(k,k) + H1B_oo(l,l)&
                                                        - H1A_vv(a,a) - H1A_vv(b,b) - H1A_vv(c,c) - H1B_vv(d,d)

                                                        deltaB = deltaB + temp/denom

                                                        denom = denom & 
                                                        -H2A_oooo(i,j,i,j)-H2A_oooo(i,k,i,k)-H2A_oooo(j,k,j,k)-H2A_voov(a,i,i,a)&
                                                        -H2A_voov(a,j,j,a)-H2A_voov(a,k,k,a)-H2A_voov(b,i,i,b)-H2A_voov(b,j,j,b)&
                                                        -H2A_voov(b,k,k,b)-H2A_voov(c,i,i,c)-H2A_voov(c,j,j,c)-H2A_voov(c,k,k,c)&
                                                        -H2A_vvvv(a,b,a,b)-H2A_vvvv(a,c,a,c)-H2A_vvvv(b,c,b,c)-H2B_oooo(i,l,i,l)&
                                                        -H2B_oooo(j,l,j,l)-H2B_oooo(k,l,k,l)+H2B_ovov(i,d,i,d)+H2B_ovov(j,d,j,d)&
                                                        +H2B_ovov(k,d,k,d)+H2B_vovo(a,l,a,l)+H2B_vovo(b,l,b,l)+H2B_vovo(c,l,c,l)&
                                                        -H2B_vvvv(a,d,a,d)-H2B_vvvv(b,d,b,d)-H2B_vvvv(c,d,c,d)-H2C_voov(d,l,l,d)

                                                        deltaC = deltaC + temp/denom

                                                        denom = denom &
                                                        +D3A_O(a,i,j)+D3A_O(a,i,k)+D3A_O(a,j,k)+D3A_O(b,i,j)&
                                                        +D3A_O(b,i,k)+D3A_O(b,j,k)+D3A_O(c,i,j)+D3A_O(c,i,k)&
                                                        +D3A_O(c,j,k)-D3A_V(a,i,b)-D3A_V(a,j,b)-D3A_V(a,k,b)&
                                                        -D3A_V(a,i,c)-D3A_V(a,j,c)-D3A_V(a,k,c)-D3A_V(b,i,c)&
                                                        -D3A_V(b,j,c)-D3A_V(b,k,c)+D3B_O(a,i,l)+D3B_O(a,j,l)&
                                                        +D3B_O(a,k,l)+D3B_O(b,i,l)+D3B_O(b,j,l)+D3B_O(b,k,l)&
                                                        +D3B_O(c,i,l)+D3B_O(c,j,l)+D3B_O(c,k,l)-D3B_V(a,i,d)&
                                                        -D3B_V(a,j,d)-D3B_V(a,k,d)-D3B_V(b,i,d)-D3B_V(b,j,d)&
                                                        -D3B_V(b,k,d)-D3B_V(c,i,d)-D3B_V(c,j,d)-D3B_V(c,k,d)&
                                                        +D3C_O(d,i,l)+D3C_O(d,j,l)+D3C_O(d,k,l)-D3C_V(a,l,d)&
                                                        -D3C_V(b,l,d)-D3C_V(c,l,d)

                                                        deltaD = deltaD + temp/denom

                                                    end do 
                                                end do 
                                            end do
                                        end do 
                                    end do 
                                end do 
                            end do 
                        end do

              end subroutine crcc24B

              subroutine crcc24C(deltaA,deltaB,deltaC,deltaD,&
                              MM24C,L4C,&
                              fA_oo,fA_vv,fB_oo,fB_vv,&
                              H1A_oo,H1A_vv,H1B_oo,H1B_vv,&
                              H2A_voov,H2A_oooo,H2A_vvvv,&
                              H2B_ovov,H2B_vovo,H2B_oooo,H2B_vvvv,&
                              H2C_voov,H2C_oooo,H2C_vvvv,&
                              D3A_O,D3A_V,D3B_O,D3B_V,D3C_O,D3C_V,D3D_O,D3D_V,&
                              noa,nua,nob,nub)
                        
                        real(kind=8), intent(out) :: deltaA, deltaB, deltaC, deltaD
                        integer, intent(in) :: noa, nua, nob, nub
                        real(kind=8), intent(in) :: MM24C(1:nua,1:nua,1:nub,1:nub,1:noa,1:noa,1:nob,1:nob),&
                        L4C(1:nua,1:nua,1:nub,1:nub,1:noa,1:noa,1:nob,1:nob),&
                        fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),fB_oo(1:nob,1:nob),fB_vv(1:nub,1:nub),&
                        H1A_oo(1:noa,1:noa),H1A_vv(1:nua,1:nua),H1B_oo(1:nob,1:nob),H1B_vv(1:nub,1:nub),&
                        H2A_voov(1:nua,1:noa,1:noa,1:nua),&
                        H2A_oooo(1:noa,1:noa,1:noa,1:noa),&
                        H2A_vvvv(1:nua,1:nua,1:nua,1:nua),&
                        H2B_ovov(1:noa,1:nub,1:noa,1:nub),&
                        H2B_vovo(1:nua,1:nob,1:nua,1:nob),&
                        H2B_oooo(1:noa,1:nob,1:noa,1:nob),&
                        H2B_vvvv(1:nua,1:nub,1:nua,1:nub),&
                        H2C_voov(1:nub,1:nob,1:nob,1:nub),&
                        H2C_oooo(1:nob,1:nob,1:nob,1:nob),&
                        H2C_vvvv(1:nub,1:nub,1:nub,1:nub),&
                        D3A_O(1:nua,1:noa,1:noa),&
                        D3A_V(1:nua,1:noa,1:nua),&
                        D3B_O(1:nua,1:noa,1:nob),&
                        D3B_V(1:nua,1:noa,1:nub),&
                        D3C_O(1:nub,1:noa,1:nob),&
                        D3C_V(1:nua,1:nob,1:nub),&
                        D3D_O(1:nub,1:nob,1:nob),&
                        D3D_V(1:nub,1:nob,1:nub)
                        integer :: i, j, k, l, a, b, c, d, ii, jj, kk, ll, aa, bb, cc, dd
                        real(kind=8) :: denom, temp

                        deltaA = 0.0
                        deltaB = 0.0
                        deltaC = 0.0
                        deltaD = 0.0

                        do ii = 1 , noa
                            do jj = ii+1, noa
                                do kk = 1, nob
                                    do ll = kk+1, nob
                                        do aa = 1, nua
                                            do bb = aa+1, nua
                                                do cc = 1, nub
                                                    do dd = cc+1, nub

                                                        a = bb; b = aa;
                                                        c = dd; d = cc;
                                                        i = jj; j = ii;
                                                        k = ll; l = kk;

                                                        temp = MM24C(a,b,c,d,i,j,k,l) * L4C(a,b,c,d,i,j,k,l)

                                                        denom = fA_oo(i,i) + fA_oo(j,j) + fB_oo(k,k) + fB_oo(l,l)&
                                                        - fA_vv(a,a) - fA_vv(b,b) - fB_vv(c,c) - fB_vv(d,d)

                                                        deltaA = deltaA + temp/denom

                                                        denom = H1A_oo(i,i) + H1A_oo(j,j) + H1B_oo(k,k) + H1B_oo(l,l)&
                                                        - H1A_vv(a,a) - H1A_vv(b,b) - H1B_vv(c,c) - H1B_vv(d,d)

                                                        deltaB = deltaB + temp/denom

                                                        denom = denom &
                                                        -H2A_oooo(i,j,i,j)-H2A_voov(a,i,i,a)-H2A_voov(a,j,j,a)-H2A_voov(b,i,i,b)&
                                                        -H2A_voov(b,j,j,b)-H2A_vvvv(a,b,a,b)-H2B_oooo(i,k,i,k)-H2B_oooo(j,k,j,k)&
                                                        -H2B_oooo(i,l,i,l)-H2B_oooo(j,l,j,l)+H2B_ovov(i,c,i,c)+H2B_ovov(j,c,j,c)&
                                                        +H2B_ovov(i,d,i,d)+H2B_ovov(j,d,j,d)+H2B_vovo(a,k,a,k)+H2B_vovo(a,l,a,l)&
                                                        +H2B_vovo(b,k,b,k)+H2B_vovo(b,l,b,l)-H2B_vvvv(a,c,a,c)-H2B_vvvv(a,d,a,d)&
                                                        -H2B_vvvv(b,c,b,c)-H2B_vvvv(b,d,b,d)-H2C_oooo(l,k,l,k)-H2C_voov(c,k,k,c)&
                                                        -H2C_voov(c,l,l,c)-H2C_voov(d,k,k,d)-H2C_voov(d,l,l,d)-H2C_vvvv(c,d,c,d)

                                                        deltaC = deltaC + temp/denom

                                                        denom = denom &
                                                        +D3A_O(a,i,j)+D3A_O(b,i,j)-D3A_V(a,i,b)-D3A_V(a,j,b)&
                                                        +D3B_O(a,i,k)+D3B_O(a,i,l)+D3B_O(a,j,k)+D3B_O(a,j,l)&
                                                        +D3B_O(b,i,k)+D3B_O(b,i,l)+D3B_O(b,j,k)+D3B_O(b,j,l)&
                                                        -D3B_V(a,i,c)-D3B_V(a,j,c)-D3B_V(b,i,c)-D3B_V(b,j,c)&
                                                        -D3B_V(a,i,d)-D3B_V(a,j,d)-D3B_V(b,i,d)-D3B_V(b,j,d)&
                                                        +D3C_O(c,i,k)+D3C_O(c,i,l)+D3C_O(c,j,k)+D3C_O(c,j,l)&
                                                        +D3C_O(d,i,k)+D3C_O(d,i,l)+D3C_O(d,j,k)+D3C_O(d,j,l)&
                                                        -D3C_V(a,k,c)-D3C_V(a,l,c)-D3C_V(b,k,c)-D3C_V(b,l,c)&
                                                        -D3C_V(a,k,d)-D3C_V(a,l,d)-D3C_V(b,k,d)-D3C_V(b,l,d)&
                                                        +D3D_O(c,k,l)+D3D_O(d,k,l)-D3D_V(c,k,d)-D3D_V(c,l,d)

                                                        deltaD = deltaD + temp/denom

                                                    end do 
                                                end do 
                                            end do
                                        end do 
                                    end do 
                                end do 
                            end do 
                        end do

              end subroutine crcc24C

end module crcc_loops
