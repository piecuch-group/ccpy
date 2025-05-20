module creacc_loops

      implicit none

      contains

              subroutine crcc23A(deltaA,deltaB,deltaC,deltaD,&
                                 M3A,L3A,omega,&
                                 fA_oo,fA_vv,H1A_oo,H1A_vv,&
                                 H2A_vvvv,H2A_oooo,H2A_voov,&
                                 d3A_o,d3A_v,&
                                 noa,nua)

                        ! input variables
                        integer, intent(in) :: noa, nua
                        real(kind=8), intent(in) :: M3A(nua,nua,nua,noa,noa)
                        real(kind=8), intent(in) :: L3A(nua,nua,nua,noa,noa)
                        real(kind=8), intent(in) :: omega
                        real(kind=8), intent(in) :: fA_oo(noa,noa), fA_vv(nua,nua)
                        real(kind=8), intent(in) :: h1A_oo(noa,noa), h1A_vv(nua,nua)
                        real(kind=8), intent(in) :: h2a_oooo(noa,noa,noa,noa)
                        real(kind=8), intent(in) :: h2a_vvvv(nua,nua,nua,nua)
                        real(kind=8), intent(in) :: h2a_voov(nua,noa,noa,nua)
                        real(kind=8), intent(in) :: d3A_o(nua,noa,noa)
                        real(kind=8), intent(in) :: d3A_v(nua,noa,nua)
                        ! output variables
                        real(kind=8), intent(out) :: deltaA, deltaB, deltaC, deltaD
                        ! Local variables
                        integer :: j, k, a, b, c
                        real(kind=8) :: D, temp

                        deltaA = 0.0
                        deltaB = 0.0
                        deltaC = 0.0
                        deltaD = 0.0

                        do a = 1,nua
                           do b = a+1,nua
                              do c = b+1,nua
                                 do j = 1,noa
                                    do k = j+1,noa
                                       temp = M3A(a,b,c,j,k)*L3A(a,b,c,j,k)
                                       ! A correction
                                       D = -fA_vv(b,b) - fA_vv(c,c) - fA_vv(a,a) + fA_oo(j,j) + fA_oo(k,k)
                                       deltaA = deltaA + temp/(omega+D)
                                       ! B correction
                                       D = -h1a_vv(b,b) - h1a_vv(c,c) - h1a_vv(a,a) + h1a_oo(j,j) + h1a_oo(k,k)
                                       deltaB = deltaB + temp/(omega+D)
                                       ! C correction
                                       D = D&
                                       -h2a_vvvv(a,b,a,b) - h2a_vvvv(a,c,a,c) - h2a_vvvv(b,c,b,c)&
                                       -h2a_oooo(j,k,j,k)&
                                       -h2a_voov(a,j,j,a) - h2a_voov(a,k,k,a)&
                                       -h2a_voov(b,j,j,b) - h2a_voov(b,k,k,b)&
                                       -h2a_voov(c,j,j,c) - h2a_voov(c,k,k,c)
                                       deltaC = deltaC + temp/(omega+D)
                                       ! D correction
                                       D = D&
                                       +d3a_o(a,j,k) + d3a_o(b,j,k) + d3a_o(c,j,k)&
                                       -d3a_v(a,j,b) - d3a_v(a,j,c) - d3a_v(b,j,c)&
                                       -d3a_v(a,k,b) - d3a_v(a,k,c) - d3a_v(b,k,c)
                                       deltaD = deltaD + temp/(omega+D)
                                    end do
                                 end do
                              end do
                           end do
                        end do

              end subroutine crcc23A

              subroutine crcc23B(deltaA,deltaB,deltaC,deltaD,&
                                 M3B,L3B,omega,&
                                 fA_oo,fA_vv,fB_oo,fB_vv,&
                                 H1A_oo,H1A_vv,H1B_oo,H1B_vv,&
                                 H2A_vvvv,H2A_voov,&
                                 H2B_vvvv,H2B_oooo,H2B_ovov,H2B_vovo,&
                                 H2C_voov,&
                                 d3A_o,d3A_v,d3B_o,d3B_v,d3C_o,d3C_v,&
                                 noa,nua,nob,nub)

                        ! input variables
                        integer, intent(in) :: noa, nua, nob, nub
                        real(kind=8), intent(in) :: M3B(nua,nua,nub,noa,nob)
                        real(kind=8), intent(in) :: L3B(nua,nua,nub,noa,nob)
                        real(kind=8), intent(in) :: omega
                        real(kind=8), intent(in) :: fA_oo(noa,noa), fA_vv(nua,nua)
                        real(kind=8), intent(in) :: fB_oo(nob,nob), fB_vv(nub,nub)
                        real(kind=8), intent(in) :: h1A_oo(noa,noa), h1A_vv(nua,nua)
                        real(kind=8), intent(in) :: h1B_oo(nob,nob), h1B_vv(nub,nub)
                        real(kind=8), intent(in) :: h2a_vvvv(nua,nua,nua,nua)
                        real(kind=8), intent(in) :: h2a_voov(nua,noa,noa,nua)
                        real(kind=8), intent(in) :: h2b_vvvv(nua,nub,nua,nub)
                        real(kind=8), intent(in) :: h2b_oooo(noa,nob,noa,nob)
                        real(kind=8), intent(in) :: h2b_ovov(noa,nub,noa,nub)
                        real(kind=8), intent(in) :: h2b_vovo(nua,nob,nua,nob)
                        real(kind=8), intent(in) :: h2c_voov(nub,nob,nob,nub)
                        real(kind=8), intent(in) :: d3A_o(nua,noa,noa)
                        real(kind=8), intent(in) :: d3A_v(nua,noa,nua)
                        real(kind=8), intent(in) :: d3B_o(nua,noa,nob)
                        real(kind=8), intent(in) :: d3B_v(nua,noa,nub)
                        real(kind=8), intent(in) :: d3C_o(nub,noa,nob)
                        real(kind=8), intent(in) :: d3C_v(nua,nob,nub)
                        ! output variables
                        real(kind=8), intent(out) :: deltaA, deltaB, deltaC, deltaD
                        ! Local variables
                        integer :: j, k, a, b, c
                        real(kind=8) :: D, temp

                        deltaA = 0.0
                        deltaB = 0.0
                        deltaC = 0.0
                        deltaD = 0.0

                        do a = 1,nua
                           do b = a+1,nua
                              do c = 1,nub
                                 do j = 1,noa
                                    do k = 1,nob
                                       temp = M3B(a,b,c,j,k)*L3B(a,b,c,j,k)
                                       ! A correction
                                       D = -fA_vv(b,b) - fB_vv(c,c) - fA_vv(a,a) + fA_oo(j,j) + fB_oo(k,k)
                                       deltaA = deltaA + temp/(omega+D)
                                       ! B correction
                                       D = -h1a_vv(b,b) - h1b_vv(c,c) - h1a_vv(a,a) + h1a_oo(j,j) + h1b_oo(k,k)
                                       deltaB = deltaB + temp/(omega+D)
                                       ! C correction
                                       D = D&
                                       -h2b_oooo(j,k,j,k)&
                                       -h2c_voov(c,k,k,c)&
                                       -h2a_voov(a,j,j,a) - h2a_voov(b,j,j,b)&
                                       +h2b_vovo(b,k,b,k) + h2b_vovo(a,k,a,k)&
                                       +h2b_ovov(j,c,j,c)&
                                       -h2a_vvvv(a,b,a,b)&
                                       -h2b_vvvv(a,c,a,c) - h2b_vvvv(b,c,b,c)
                                       deltaC = deltaC + temp/(omega+D)
                                       ! D correction
                                       D = D&
                                       +D3B_O(a,j,k)&
                                       +D3B_O(b,j,k)&
                                       +D3C_O(c,j,k)&
                                       -D3A_V(a,j,b)-D3B_V(a,j,c)-D3B_V(b,j,c)&
                                       -D3C_V(a,k,c)-D3C_V(b,k,c)
                                       deltaD = deltaD + temp/(omega+D)
                                    end do
                                 end do
                              end do
                           end do
                        end do

              end subroutine crcc23B

              subroutine crcc23C(deltaA,deltaB,deltaC,deltaD,&
                                 M3C,L3C,omega,&
                                 fA_oo,fA_vv,fB_oo,fB_vv,&
                                 H1A_oo,H1A_vv,H1B_oo,H1B_vv,&
                                 H2B_vvvv,H2B_ovov,H2B_vovo,&
                                 H2C_vvvv,H2C_oooo,H2C_voov,&
                                 d3B_o,d3B_v,d3C_o,d3C_v,d3D_o,d3D_v,&
                                 noa,nua,nob,nub)

                        ! input variables
                        integer, intent(in) :: noa, nua, nob, nub
                        real(kind=8), intent(in) :: M3C(nua,nub,nub,nob,nob)
                        real(kind=8), intent(in) :: L3C(nua,nub,nub,nob,nob)
                        real(kind=8), intent(in) :: omega
                        real(kind=8), intent(in) :: fA_oo(noa,noa), fA_vv(nua,nua)
                        real(kind=8), intent(in) :: fB_oo(nob,nob), fB_vv(nub,nub)
                        real(kind=8), intent(in) :: h1A_oo(noa,noa), h1A_vv(nua,nua)
                        real(kind=8), intent(in) :: h1B_oo(nob,nob), h1B_vv(nub,nub)
                        real(kind=8), intent(in) :: h2b_vvvv(nua,nub,nua,nub)
                        real(kind=8), intent(in) :: h2b_ovov(noa,nub,noa,nub)
                        real(kind=8), intent(in) :: h2b_vovo(nua,nob,nua,nob)
                        real(kind=8), intent(in) :: h2c_vvvv(nub,nub,nub,nub)
                        real(kind=8), intent(in) :: h2c_oooo(nob,nob,nob,nob)
                        real(kind=8), intent(in) :: h2c_voov(nub,nob,nob,nub)
                        real(kind=8), intent(in) :: d3B_o(nua,noa,nob)
                        real(kind=8), intent(in) :: d3B_v(nua,noa,nub)
                        real(kind=8), intent(in) :: d3C_o(nub,noa,nob)
                        real(kind=8), intent(in) :: d3C_v(nua,nob,nub)
                        real(kind=8), intent(in) :: d3D_o(nub,nob,nob)
                        real(kind=8), intent(in) :: d3D_v(nub,nob,nub)
                        ! output variables
                        real(kind=8), intent(out) :: deltaA, deltaB, deltaC, deltaD
                        ! Local variables
                        integer :: j, k, a, b, c
                        real(kind=8) :: D, temp

                        deltaA = 0.0
                        deltaB = 0.0
                        deltaC = 0.0
                        deltaD = 0.0

                        do a = 1,nua
                           do b = 1,nub
                              do c = b+1,nub
                                 do j = 1,nob
                                    do k = j+1,nob
                                       temp = M3C(a,b,c,j,k)*L3C(a,b,c,j,k)
                                       ! A correction
                                       D = -fB_vv(b,b) - fB_vv(c,c) - fA_vv(a,a) + fB_oo(j,j) + fB_oo(k,k)
                                       deltaA = deltaA + temp/(omega+D)
                                       ! B correction
                                       D = -h1b_vv(b,b) - h1b_vv(c,c) - h1a_vv(a,a) + h1b_oo(j,j) + h1b_oo(k,k)
                                       deltaB = deltaB + temp/(omega+D)
                                       ! C correction
                                       D = D&
                                       +H2B_vovo(a,j,a,j)-H2C_voov(b,j,j,b)-H2C_voov(c,j,j,c)&
                                       +H2B_vovo(a,k,a,k)-H2C_voov(b,k,k,b)-H2C_voov(c,k,k,c)&
                                       -H2C_oooo(k,j,k,j)&
                                       -H2B_vvvv(a,b,a,b)-H2B_vvvv(a,c,a,c)-H2C_vvvv(c,b,c,b)
                                       deltaC = deltaC + temp/(omega+D)
                                       ! D correction
                                       D = D&
                                       +D3D_O(b,j,k)&
                                       +D3D_O(c,j,k)&
                                       -D3C_V(a,j,b)-D3C_V(a,j,c)-D3D_V(b,j,c)&
                                       -D3C_V(a,k,b)-D3C_V(a,k,c)-D3D_V(b,k,c)
                                       deltaD = deltaD + temp/(omega+D)
                                    end do
                                 end do
                              end do
                           end do
                        end do

              end subroutine crcc23C

end module creacc_loops
