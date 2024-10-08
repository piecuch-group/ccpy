module cripcc_loops

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
                        real(kind=8), intent(in) :: M3A(noa,nua,nua,noa,noa)
                        real(kind=8), intent(in) :: L3A(noa,nua,nua,noa,noa)
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
                        integer :: i, j, k, b, c
                        real(kind=8) :: D, temp
                        real(kind=8) :: phase

                        deltaA = 0.0
                        deltaB = 0.0
                        deltaC = 0.0
                        deltaD = 0.0

                        phase = 1.0d0

                        do i = 1,noa
                           do j = i+1,noa
                              do k = j+1,noa
                                 do b = 1,nua
                                    do c = b+1,nua
                                       temp = M3A(i,b,c,j,k)*L3A(i,b,c,j,k)
                                       ! A correction
                                       D = -fA_vv(b,b) - fA_vv(c,c) + fA_oo(i,i) + fA_oo(j,j) + fA_oo(k,k)
                                       deltaA = deltaA + temp/(omega+D)
                                       ! B correction
                                       D = -h1a_vv(b,b) - h1a_vv(c,c) + h1a_oo(i,i) + h1a_oo(j,j) + h1a_oo(k,k)
                                       deltaB = deltaB + temp/(omega+D)
                                       ! C correction
                                       D = D&
                                       + phase*(-h2a_oooo(i,j,i,j) - h2a_oooo(i,k,i,k) - h2a_oooo(j,k,j,k)&
                                       -h2a_vvvv(b,c,b,c)&
                                       -h2a_voov(b,i,i,b) - h2a_voov(c,i,i,c)&
                                       -h2a_voov(b,j,j,b) - h2a_voov(c,j,j,c)&
                                       -h2a_voov(b,k,k,b) - h2a_voov(c,k,k,c))
                                       deltaC = deltaC + temp/(omega+D)
                                       ! D correction
                                       D = D&
                                       + phase*(D3A_O(b,i,j)+D3A_O(b,i,k)+D3A_O(b,j,k)&
                                       +D3A_O(c,i,j)+D3A_O(c,i,k)+D3A_O(c,j,k)&
                                       -D3A_V(b,i,c)&
                                       -D3A_V(b,j,c)&
                                       -D3A_V(b,k,c))
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
                                 H2A_oooo,H2A_voov,&
                                 H2B_vvvv,H2B_oooo,H2B_ovov,H2B_vovo,&
                                 H2C_voov,&
                                 d3A_o,d3A_v,d3B_o,d3B_v,d3C_o,d3C_v,&
                                 noa,nua,nob,nub)

                        ! input variables
                        integer, intent(in) :: noa, nua, nob, nub
                        real(kind=8), intent(in) :: M3B(noa,nua,nub,noa,nob)
                        real(kind=8), intent(in) :: L3B(noa,nua,nub,noa,nob)
                        real(kind=8), intent(in) :: omega
                        real(kind=8), intent(in) :: fA_oo(noa,noa), fA_vv(nua,nua)
                        real(kind=8), intent(in) :: fB_oo(nob,nob), fB_vv(nub,nub)
                        real(kind=8), intent(in) :: h1A_oo(noa,noa), h1A_vv(nua,nua)
                        real(kind=8), intent(in) :: h1B_oo(nob,nob), h1B_vv(nub,nub)
                        real(kind=8), intent(in) :: h2a_oooo(noa,noa,noa,noa)
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
                        integer :: i, j, k, b, c
                        real(kind=8) :: D, temp
                        real(kind=8) :: phase

                        deltaA = 0.0
                        deltaB = 0.0
                        deltaC = 0.0
                        deltaD = 0.0

                        phase = 1.0d0

                        do i = 1,noa
                           do j = i+1,noa
                              do k = 1,nob
                                 do b = 1,nua
                                    do c = 1,nub
                                       temp = M3B(i,b,c,j,k)*L3B(i,b,c,j,k)
                                       ! A correction
                                       D = -fA_vv(b,b) - fB_vv(c,c) + fA_oo(i,i) + fA_oo(j,j) + fB_oo(k,k)
                                       deltaA = deltaA + temp/(omega+D)
                                       ! B correction
                                       D = -h1a_vv(b,b) - h1b_vv(c,c) + h1a_oo(i,i) + h1a_oo(j,j) + h1b_oo(k,k)
                                       deltaB = deltaB + temp/(omega+D)
                                       ! C correction
                                       D = D&
                                       + phase*(-h2a_oooo(i,j,i,j) - h2b_oooo(i,k,i,k) - h2b_oooo(j,k,j,k)&
                                       -h2b_vvvv(b,c,b,c)&
                                       -h2a_voov(b,i,i,b) + h2b_ovov(i,c,i,c)&
                                       -h2a_voov(b,j,j,b) + h2b_ovov(j,c,j,c)&
                                       +h2b_vovo(b,k,b,k) - h2c_voov(c,k,k,c))
                                       deltaC = deltaC + temp/(omega+D)
                                       ! D correction
                                       D = D &
                                       + phase*(D3A_O(b,i,j)+D3B_O(b,i,k)+D3B_O(b,j,k)&
                                       +D3C_O(c,i,k)+D3C_O(c,j,k)&
                                       -D3B_V(b,i,c)&
                                       -D3B_V(b,j,c)&
                                       -D3C_V(b,k,c))
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
                                 H2B_oooo,H2B_ovov,H2B_vovo,&
                                 H2C_vvvv,H2C_oooo,H2C_voov,&
                                 d3B_o,d3B_v,d3C_o,d3C_v,d3D_o,d3D_v,&
                                 noa,nua,nob,nub)

                        ! input variables
                        integer, intent(in) :: noa, nua, nob, nub
                        real(kind=8), intent(in) :: M3C(noa,nub,nub,nob,nob)
                        real(kind=8), intent(in) :: L3C(noa,nub,nub,nob,nob)
                        real(kind=8), intent(in) :: omega
                        real(kind=8), intent(in) :: fA_oo(noa,noa), fA_vv(nua,nua)
                        real(kind=8), intent(in) :: fB_oo(nob,nob), fB_vv(nub,nub)
                        real(kind=8), intent(in) :: h1A_oo(noa,noa), h1A_vv(nua,nua)
                        real(kind=8), intent(in) :: h1B_oo(nob,nob), h1B_vv(nub,nub)
                        real(kind=8), intent(in) :: h2b_oooo(noa,nob,noa,nob)
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
                        integer :: i, j, k, b, c
                        real(kind=8) :: D, temp
                        real(kind=8) :: phase

                        deltaA = 0.0
                        deltaB = 0.0
                        deltaC = 0.0
                        deltaD = 0.0

                        phase = 1.0d0

                        do i = 1,noa
                           do j = 1,nob
                              do k = j+1,nob
                                 do b = 1,nub
                                    do c = b+1,nub
                                       temp = M3C(i,b,c,j,k)*L3C(i,b,c,j,k)
                                       ! A correction
                                       D = -fB_vv(b,b) - fB_vv(c,c) + fA_oo(i,i) + fB_oo(j,j) + fB_oo(k,k)
                                       deltaA = deltaA + temp/(omega+D)
                                       ! B correction
                                       D = -h1b_vv(b,b) - h1b_vv(c,c) + h1a_oo(i,i) + h1b_oo(j,j) + h1b_oo(k,k)
                                       deltaB = deltaB + temp/(omega+D)
                                       ! C correction
                                       D = D&
                                       + phase*(-h2b_oooo(i,j,i,j) - h2b_oooo(i,k,i,k) - h2c_oooo(j,k,j,k)&
                                       -h2c_vvvv(b,c,b,c)&
                                       +h2b_ovov(i,b,i,b) + h2b_ovov(i,c,i,c)&
                                       -h2c_voov(b,j,j,b) - h2c_voov(c,j,j,c)&
                                       -h2c_voov(b,k,k,b) - h2c_voov(c,k,k,c))
                                       deltaC = deltaC + temp/(omega+D)
                                       ! D correction
                                       D = D&
                                       + phase*(D3C_O(b,i,j)+D3C_O(b,i,k)+D3D_O(b,j,k)&
                                       +D3C_O(c,i,j)+D3C_O(c,i,k)+D3D_O(c,j,k)&
                                       -D3D_V(b,j,c)&
                                       -D3D_V(b,k,c))
                                       deltaD = deltaD + temp/(omega+D)
                                    end do
                                 end do
                              end do
                           end do
                        end do

              end subroutine crcc23C



              !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! REORDER ROUTINES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         
!               subroutine reorder4(y, x, iorder)
!
!                   integer, intent(in) :: iorder(4)
!                   real(kind=8), intent(in) :: x(:,:,:,:)
!
!                   real(kind=8), intent(out) :: y(:,:,:,:)
!
!                   integer :: i, j, k, l
!                   integer :: vec(4)
!
!                   y = 0.0d0
!                   do i = 1, size(x,1)
!                      do j = 1, size(x,2)
!                         do k = 1, size(x,3)
!                            do l = 1, size(x,4)
!                               vec = (/i,j,k,l/)
!                               y(vec(iorder(1)),vec(iorder(2)),vec(iorder(3)),vec(iorder(4))) = x(i,j,k,l)
!                            end do
!                         end do
!                      end do
!                   end do
!
!               end subroutine reorder4

              subroutine reorder3412(x_in,x_out)

                      real(kind=8), intent(in) :: x_in(:,:,:,:)
                      real(kind=8), intent(out) :: x_out(:,:,:,:)

                      integer :: i1, i2, i3, i4

                      do i1 = 1,size(x_in,1)
                         do i2 = 1,size(x_in,2)
                            do i3 = 1,size(x_in,3)
                               do i4= 1,size(x_in,4)
                                  x_out(i3,i4,i1,i2) = x_in(i1,i2,i3,i4)
                               end do
                            end do
                         end do
                      end do

             end subroutine reorder3412

             subroutine reorder1342(x_in,x_out)

                      real(kind=8), intent(in) :: x_in(:,:,:,:)
                      real(kind=8), intent(out) :: x_out(:,:,:,:)

                      integer :: i1, i2, i3, i4

                      do i1 = 1,size(x_in,1)
                         do i2 = 1,size(x_in,2)
                            do i3 = 1,size(x_in,3)
                               do i4= 1,size(x_in,4)
                                  x_out(i1,i3,i4,i2) = x_in(i1,i2,i3,i4)
                               end do
                            end do
                         end do
                      end do

             end subroutine reorder1342

            subroutine reorder3421(x_in,x_out)

                      real(kind=8), intent(in) :: x_in(:,:,:,:)
                      real(kind=8), intent(out) :: x_out(:,:,:,:)

                      integer :: i1, i2, i3, i4

                      do i1 = 1,size(x_in,1)
                         do i2 = 1,size(x_in,2)
                            do i3 = 1,size(x_in,3)
                               do i4= 1,size(x_in,4)
                                  x_out(i3,i4,i2,i1) = x_in(i1,i2,i3,i4)
                               end do
                            end do
                         end do
                      end do

             end subroutine reorder3421

             subroutine reorder2134(x_in,x_out)

                      real(kind=8), intent(in) :: x_in(:,:,:,:)
                      real(kind=8), intent(out) :: x_out(:,:,:,:)

                      integer :: i1, i2, i3, i4

                      do i1 = 1,size(x_in,1)
                         do i2 = 1,size(x_in,2)
                            do i3 = 1,size(x_in,3)
                               do i4= 1,size(x_in,4)
                                  x_out(i2,i1,i3,i4) = x_in(i1,i2,i3,i4)
                               end do
                            end do
                         end do
                      end do

             end subroutine reorder2134

            subroutine reorder1243(x_in,x_out)

                      real(kind=8), intent(in) :: x_in(:,:,:,:)
                      real(kind=8), intent(out) :: x_out(:,:,:,:)

                      integer :: i1, i2, i3, i4

                      do i1 = 1,size(x_in,1)
                         do i2 = 1,size(x_in,2)
                            do i3 = 1,size(x_in,3)
                               do i4= 1,size(x_in,4)
                                  x_out(i1,i2,i4,i3) = x_in(i1,i2,i3,i4)
                               end do
                            end do
                         end do
                      end do

             end subroutine reorder1243

             subroutine reorder4213(x_in,x_out)

                      real(kind=8), intent(in) :: x_in(:,:,:,:)
                      real(kind=8), intent(out) :: x_out(:,:,:,:)

                      integer :: i1, i2, i3, i4

                      do i1 = 1,size(x_in,1)
                         do i2 = 1,size(x_in,2)
                            do i3 = 1,size(x_in,3)
                               do i4= 1,size(x_in,4)
                                  x_out(i4,i2,i1,i3) = x_in(i1,i2,i3,i4)
                               end do
                            end do
                         end do
                      end do

             end subroutine reorder4213

             subroutine reorder4312(x_in,x_out)

                      real(kind=8), intent(in) :: x_in(:,:,:,:)
                      real(kind=8), intent(out) :: x_out(:,:,:,:)

                      integer :: i1, i2, i3, i4

                      do i1 = 1,size(x_in,1)
                         do i2 = 1,size(x_in,2)
                            do i3 = 1,size(x_in,3)
                               do i4= 1,size(x_in,4)
                                  x_out(i4,i3,i1,i2) = x_in(i1,i2,i3,i4)
                               end do
                            end do
                         end do
                      end do

             end subroutine reorder4312

             subroutine reorder2341(x_in,x_out)

                      real(kind=8), intent(in) :: x_in(:,:,:,:)
                      real(kind=8), intent(out) :: x_out(:,:,:,:)

                      integer :: i1, i2, i3, i4

                      do i1 = 1,size(x_in,1)
                         do i2 = 1,size(x_in,2)
                            do i3 = 1,size(x_in,3)
                               do i4= 1,size(x_in,4)
                                  x_out(i2,i3,i4,i1) = x_in(i1,i2,i3,i4)
                               end do
                            end do
                         end do
                      end do

             end subroutine reorder2341

             subroutine reorder2143(x_in,x_out)

                      real(kind=8), intent(in) :: x_in(:,:,:,:)
                      real(kind=8), intent(out) :: x_out(:,:,:,:)

                      integer :: i1, i2, i3, i4

                      do i1 = 1,size(x_in,1)
                         do i2 = 1,size(x_in,2)
                            do i3 = 1,size(x_in,3)
                               do i4= 1,size(x_in,4)
                                  x_out(i2,i1,i4,i3) = x_in(i1,i2,i3,i4)
                               end do
                            end do
                         end do
                      end do

             end subroutine reorder2143

             subroutine reorder4123(x_in,x_out)

                      real(kind=8), intent(in) :: x_in(:,:,:,:)
                      real(kind=8), intent(out) :: x_out(:,:,:,:)

                      integer :: i1, i2, i3, i4

                      do i1 = 1,size(x_in,1)
                         do i2 = 1,size(x_in,2)
                            do i3 = 1,size(x_in,3)
                               do i4= 1,size(x_in,4)
                                  x_out(i4,i1,i2,i3) = x_in(i1,i2,i3,i4)
                               end do
                            end do
                         end do
                      end do

             end subroutine reorder4123

             subroutine reorder3214(x_in,x_out)

                      real(kind=8), intent(in) :: x_in(:,:,:,:)
                      real(kind=8), intent(out) :: x_out(:,:,:,:)

                      integer :: i1, i2, i3, i4

                      do i1 = 1,size(x_in,1)
                         do i2 = 1,size(x_in,2)
                            do i3 = 1,size(x_in,3)
                               do i4= 1,size(x_in,4)
                                  x_out(i3,i2,i1,i4) = x_in(i1,i2,i3,i4)
                               end do
                            end do
                         end do
                      end do

             end subroutine reorder3214

end module cripcc_loops
