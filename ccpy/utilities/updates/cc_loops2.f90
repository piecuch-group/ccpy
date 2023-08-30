module cc_loops2

  implicit none

  contains

      subroutine update_t1a(t1a,resid,X1A,fA_oo,fA_vv,shift,noa,nua)


              integer, intent(in) :: noa, nua
              real(kind=8), intent(in) :: fA_oo(1:noa,1:noa), fA_vv(1:nua,1:nua), &
                                  X1A(1:nua,1:noa), shift               
              real(kind=8), intent(inout) :: t1a(1:nua,1:noa)
              !f2py intent(in,out) :: t1a(0:nua-1,0:noa-1)
              real(kind=8), intent(out) :: resid(1:nua,1:noa)
              integer :: i, a
              real(kind=8) :: denom, val

              resid = 0.0d0
              do i = 1,noa
                do a = 1,nua
                  denom = fA_oo(i,i) - fA_vv(a,a)
                  val = X1A(a,i)/(denom-shift)
                  t1a(a,i) = t1a(a,i) + val
                  resid(a,i) = val
                end do
              end do

      end subroutine update_t1a

      subroutine update_t1b(t1b,resid,X1B,fB_oo,fB_vv,shift,nob,nub)

              integer, intent(in) :: nob, nub
              real(kind=8), intent(in) :: fB_oo(1:nob,1:nob), fB_vv(1:nub,1:nub), &
                                  X1B(1:nub,1:nob), shift               
              real(kind=8), intent(inout) :: t1b(1:nub,1:nob)
              !f2py intent(in,out) :: t1b(0:nub-1,0:nob-1)
              real(kind=8), intent(out) :: resid(1:nub,1:nob)
              integer :: i, a
              real(kind=8) :: denom, val

              resid = 0.0d0
              do i = 1,nob
                do a = 1,nub
                  denom = fB_oo(i,i) - fB_vv(a,a)
                  val = X1B(a,i)/(denom-shift)
                  t1b(a,i) = t1b(a,i) + val
                  resid(a,i) = val
                end do
              end do

      end subroutine update_t1b

      subroutine update_t2a(t2a,resid,X2A,fA_oo,fA_vv,shift,noa,nua)

              integer, intent(in) :: noa, nua
              real(kind=8), intent(in) :: fA_oo(1:noa,1:noa), fA_vv(1:nua,1:nua), &
                                  X2A(1:nua,1:nua,1:noa,1:noa), shift               
              real(kind=8), intent(inout) :: t2a(1:nua,1:nua,1:noa,1:noa)
              !f2py intent(in,out) :: t2a(0:nua-1,0:nua-1,0:noa-1,0:noa-1)
              real(kind=8), intent(out) :: resid(1:nua,1:nua,1:noa,1:noa)
              integer :: i, j, a, b
              real(kind=8) :: denom, val

              resid = 0.0d0
              do i = 1,noa
                do j = i+1,noa
                  do a = 1,nua
                    do b = a+1,nua
                      denom = fA_oo(i,i) + fA_oo(j,j) - fA_vv(a,a) - fA_vv(b,b)
                      val = (X2A(b,a,j,i) - X2A(a,b,j,i) - X2A(b,a,i,j) + X2A(a,b,i,j))/(denom-shift)
                      t2a(b,a,j,i) = t2a(b,a,j,i) + val
                      t2a(a,b,j,i) = -t2a(b,a,j,i)
                      t2a(b,a,i,j) = -t2a(b,a,j,i)
                      t2a(a,b,i,j) = t2a(b,a,j,i)

                      resid(b,a,j,i) = val
                      resid(a,b,j,i) = -val
                      resid(b,a,i,j) = -val
                      resid(a,b,i,j) = val
                    end do
                  end do
                end do
              end do

      end subroutine update_t2a

      subroutine update_t2b(t2b,resid,X2B,fA_oo,fA_vv,fB_oo,fB_vv,shift,noa,nua,nob,nub)

              integer, intent(in) :: noa, nua, nob, nub
              real(kind=8), intent(in) :: fA_oo(1:noa,1:noa), fA_vv(1:nua,1:nua), &
                                  fB_oo(1:nob,1:nob), fB_vv(1:nub,1:nub), &
                                  X2B(1:nua,1:nub,1:noa,1:nob), shift               
              real(kind=8), intent(inout) :: t2b(1:nua,1:nub,1:noa,1:nob)
              !f2py intent(in,out) :: t2b(0:nua-1,0:nub-1,0:noa-1,0:nob-1)
              real(kind=8), intent(out) :: resid(1:nua,1:nub,1:noa,1:nob)
              integer :: i, j, a, b
              real(kind=8) :: denom, val

              resid = 0.0d0
              do j = 1,nob
                do i = 1,noa
                  do b = 1,nub
                    do a = 1,nua
                      denom = fA_oo(i,i) + fB_oo(j,j) - fA_vv(a,a) - fB_vv(b,b)
                      val = X2B(a,b,i,j)/(denom-shift)
                      t2b(a,b,i,j) = t2b(a,b,i,j) + val
                      resid(a,b,i,j) = val
                    end do
                  end do
                end do
              end do

      end subroutine update_t2b

      subroutine update_t2c(t2c,resid,X2C,fB_oo,fB_vv,shift,nob,nub)

              integer, intent(in) :: nob, nub
              real(kind=8), intent(in) :: fB_oo(1:nob,1:nob), fB_vv(1:nub,1:nub), &
                                  X2C(1:nub,1:nub,1:nob,1:nob), shift               
              real(kind=8), intent(inout) :: t2c(1:nub,1:nub,1:nob,1:nob)
              !f2py intent(in,out) :: t2c(0:nub-1,0:nub-1,0:nob-1,0:nob-1)
              real(kind=8), intent(out) :: resid(1:nub,1:nub,1:nob,1:nob)
              integer :: i, j, a, b
              real(kind=8) :: denom, val

              resid = 0.0d0
              do i = 1,nob
                do j = i+1,nob
                  do a = 1,nub
                    do b = a+1,nub
                      denom = fB_oo(i,i) + fB_oo(j,j) - fB_vv(a,a) - fB_vv(b,b)
                      !t2c(b,a,j,i) = t2c(b,a,j,i) + X2C(b,a,j,i)/(denom-shift)
                      val = (X2C(b,a,j,i) - X2C(a,b,j,i) - X2C(b,a,i,j) + X2C(a,b,i,j))/(denom-shift)
                      t2c(b,a,j,i) = t2c(b,a,j,i) + val
                      t2c(a,b,j,i) = -t2c(b,a,j,i)
                      t2c(b,a,i,j) = -t2c(b,a,j,i)
                      t2c(a,b,i,j) = t2c(b,a,j,i)

                      resid(b,a,j,i) = val
                      resid(a,b,j,i) = -val
                      resid(b,a,i,j) = -val
                      resid(a,b,i,j) = val
                    end do
                  end do
                end do
              end do

      end subroutine update_t2c

      subroutine update_t3a_v2(t3a,resid,X3A,fA_oo,fA_vv,shift,noa,nua)

              integer, intent(in) :: noa, nua
              real(kind=8), intent(in) :: fA_oo(1:noa,1:noa), fA_vv(1:nua,1:nua), &
                                  X3A(1:nua,1:nua,1:nua,1:noa,1:noa,1:noa), shift               
              real(kind=8), intent(inout) :: t3a(1:nua,1:nua,1:nua,1:noa,1:noa,1:noa)
              !f2py intent(in,out) :: t3a(0:nua-1,0:nua-1,0:nua-1,0:noa-1,0:noa-1,0:noa-1)
              real(kind=8), intent(out) :: resid(1:nua,1:nua,1:nua,1:noa,1:noa,1:noa)
              integer :: i, j, k, a, b, c, ii, jj, kk, aa, bb, cc
              real(kind=8) :: denom, val

              resid = 0.0d0
              do ii = 1,noa
                  do jj = ii+1,noa
                      do kk = jj+1,noa
                          do aa = 1,nua
                              do bb = aa+1,nua
                                  do cc = bb+1,nua

                                      A = cc; B = bb; C = aa;
                                      I = kk; J = jj; K = ii;
                                      
                                      denom = fA_oo(I,I)+fA_oo(J,J)+fA_oo(K,K)-fA_vv(A,A)-fA_vv(B,B)-fA_vv(C,C)

                                      val = X3A(a,b,c,i,j,k)&
                                              -X3A(b,a,c,i,j,k)&
                                              -X3A(a,c,b,i,j,k)&
                                              +X3A(b,c,a,i,j,k)&
                                              -X3A(c,b,a,i,j,k)&
                                              +X3A(c,a,b,i,j,k)&
                                              -X3A(a,b,c,j,i,k)&
                                              +X3A(b,a,c,j,i,k)&
                                              +X3A(a,c,b,j,i,k)&
                                              -X3A(b,c,a,j,i,k)&
                                              +X3A(c,b,a,j,i,k)&
                                              -X3A(c,a,b,j,i,k)&
                                              -X3A(a,b,c,i,k,j)&
                                              +X3A(b,a,c,i,k,j)&
                                              +X3A(a,c,b,i,k,j)&
                                              -X3A(b,c,a,i,k,j)&
                                              +X3A(c,b,a,i,k,j)&
                                              -X3A(c,a,b,i,k,j)&
                                              -X3A(a,b,c,k,j,i)&
                                              +X3A(b,a,c,k,j,i)&
                                              +X3A(a,c,b,k,j,i)&
                                              -X3A(b,c,a,k,j,i)&
                                              +X3A(c,b,a,k,j,i)&
                                              -X3A(c,a,b,k,j,i)&
                                              +X3A(a,b,c,j,k,i)&
                                              -X3A(b,a,c,j,k,i)&
                                              -X3A(a,c,b,j,k,i)&
                                              +X3A(b,c,a,j,k,i)&
                                              -X3A(c,b,a,j,k,i)&
                                              +X3A(c,a,b,j,k,i)&
                                              +X3A(a,b,c,k,i,j)&
                                              -X3A(b,a,c,k,i,j)&
                                              -X3A(a,c,b,k,i,j)&
                                              +X3A(b,c,a,k,i,j)&
                                              -X3A(c,b,a,k,i,j)&
                                              +X3A(c,a,b,k,i,j)

                                      val = val/(denom-shift)

                                      t3a(A,B,C,I,J,K) = t3a(A,B,C,I,J,K) + val                            
                                      t3a(A,B,C,K,I,J) = t3a(A,B,C,I,J,K)
                                      t3a(A,B,C,J,K,I) = t3a(A,B,C,I,J,K)
                                      t3a(A,B,C,I,K,J) = -t3a(A,B,C,I,J,K)
                                      t3a(A,B,C,J,I,K) = -t3a(A,B,C,I,J,K)
                                      t3a(A,B,C,K,J,I) = -t3a(A,B,C,I,J,K)
                                      
                                      t3a(B,A,C,I,J,K) = -t3a(A,B,C,I,J,K)
                                      t3a(B,A,C,K,I,J) = -t3a(A,B,C,I,J,K)
                                      t3a(B,A,C,J,K,I) = -t3a(A,B,C,I,J,K)
                                      t3a(B,A,C,I,K,J) = t3a(A,B,C,I,J,K)
                                      t3a(B,A,C,J,I,K) = t3a(A,B,C,I,J,K)
                                      t3a(B,A,C,K,J,I) = t3a(A,B,C,I,J,K)
                                      
                                      t3a(A,C,B,I,J,K) = -t3a(A,B,C,I,J,K)
                                      t3a(A,C,B,K,I,J) = -t3a(A,B,C,I,J,K)
                                      t3a(A,C,B,J,K,I) = -t3a(A,B,C,I,J,K)
                                      t3a(A,C,B,I,K,J) = t3a(A,B,C,I,J,K)
                                      t3a(A,C,B,J,I,K) = t3a(A,B,C,I,J,K)
                                      t3a(A,C,B,K,J,I) = t3a(A,B,C,I,J,K)
                                      
                                      t3a(C,B,A,I,J,K) = -t3a(A,B,C,I,J,K)
                                      t3a(C,B,A,K,I,J) = -t3a(A,B,C,I,J,K)
                                      t3a(C,B,A,J,K,I) = -t3a(A,B,C,I,J,K)
                                      t3a(C,B,A,I,K,J) = t3a(A,B,C,I,J,K)
                                      t3a(C,B,A,J,I,K) = t3a(A,B,C,I,J,K)
                                      t3a(C,B,A,K,J,I) = t3a(A,B,C,I,J,K)
                                      
                                      t3a(B,C,A,I,J,K) = t3a(A,B,C,I,J,K)
                                      t3a(B,C,A,K,I,J) = t3a(A,B,C,I,J,K)
                                      t3a(B,C,A,J,K,I) = t3a(A,B,C,I,J,K)
                                      t3a(B,C,A,I,K,J) = -t3a(A,B,C,I,J,K)
                                      t3a(B,C,A,J,I,K) = -t3a(A,B,C,I,J,K)
                                      t3a(B,C,A,K,J,I) = -t3a(A,B,C,I,J,K)
                                      
                                      t3a(C,A,B,I,J,K) = t3a(A,B,C,I,J,K)
                                      t3a(C,A,B,K,I,J) = t3a(A,B,C,I,J,K)
                                      t3a(C,A,B,J,K,I) = t3a(A,B,C,I,J,K)
                                      t3a(C,A,B,I,K,J) = -t3a(A,B,C,I,J,K)
                                      t3a(C,A,B,J,I,K) = -t3a(A,B,C,I,J,K)
                                      t3a(C,A,B,K,J,I) = -t3a(A,B,C,I,J,K)


                                      resid(A,B,C,I,J,K) = val                            
                                      resid(A,B,C,K,I,J) = val
                                      resid(A,B,C,J,K,I) = val
                                      resid(A,B,C,I,K,J) = -val
                                      resid(A,B,C,J,I,K) = -val
                                      resid(A,B,C,K,J,I) = -val
                                      resid(B,C,A,I,J,K) = val                            
                                      resid(B,C,A,K,I,J) = val
                                      resid(B,C,A,J,K,I) = val
                                      resid(B,C,A,I,K,J) = -val
                                      resid(B,C,A,J,I,K) = -val
                                      resid(B,C,A,K,J,I) = -val
                                      resid(C,A,B,I,J,K) = val                            
                                      resid(C,A,B,K,I,J) = val
                                      resid(C,A,B,J,K,I) = val
                                      resid(C,A,B,I,K,J) = -val
                                      resid(C,A,B,J,I,K) = -val
                                      resid(C,A,B,K,J,I) = -val
                                      resid(A,C,B,I,J,K) = -val                            
                                      resid(A,C,B,K,I,J) = -val
                                      resid(A,C,B,J,K,I) = -val
                                      resid(A,C,B,I,K,J) = val
                                      resid(A,C,B,J,I,K) = val
                                      resid(A,C,B,K,J,I) = val
                                      resid(B,A,C,I,J,K) = -val                            
                                      resid(B,A,C,K,I,J) = -val
                                      resid(B,A,C,J,K,I) = -val
                                      resid(B,A,C,I,K,J) = val
                                      resid(B,A,C,J,I,K) = val
                                      resid(B,A,C,K,J,I) = val
                                      resid(C,B,A,I,J,K) = -val                            
                                      resid(C,B,A,K,I,J) = -val
                                      resid(C,B,A,J,K,I) = -val
                                      resid(C,B,A,I,K,J) = val
                                      resid(C,B,A,J,I,K) = val
                                      resid(C,B,A,K,J,I) = val
                                  end do
                              end do
                          end do
                      end do
                  end do
              end do

      end subroutine update_t3a_v2


      subroutine update_t3b_v2(t3b,resid,X3B,fA_oo,fA_vv,fB_oo,fB_vv,shift,noa,nua,nob,nub)

              integer, intent(in) :: noa, nua, nob, nub
              real(kind=8), intent(in) :: fA_oo(1:noa,1:noa), fA_vv(1:nua,1:nua), &
                                  fB_oo(1:nob,1:nob), fB_vv(1:nub,1:nub), &
                                  X3B(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob), shift               
              real(kind=8), intent(inout) :: t3b(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob)
              !f2py intent(in,out) :: t3b(0:nua-1,0:nua-1,0:nub-1,0:noa-1,0:noa-1,0:nob-1)
              real(kind=8), intent(out) :: resid(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob)
              integer :: i, j, k, a, b, c, ii, jj, kk, aa, bb, cc
              real(kind=8) :: denom, val

              resid = 0.0d0
              do ii = 1,noa
                  do jj = ii+1,noa
                      do kk = 1,nob
                          do aa = 1,nua
                              do bb = aa+1,nua
                                  do cc = 1,nub
                  
                                      a = bb; b = aa; c = cc;
                                      i = jj; j = ii; k = kk;

                                      denom = fA_oo(i,i)+fA_oo(j,j)+fB_oo(k,k)-fA_vv(a,a)-fA_vv(b,b)-fB_vv(c,c)
                                      val = X3B(a,b,c,i,j,k) - X3B(b,a,c,i,j,k) - X3B(a,b,c,j,i,k) + X3B(b,a,c,j,i,k)
                                      val = val/(denom-shift)
                                      t3b(a,b,c,i,j,k) = t3b(a,b,c,i,j,k) + val
                                      t3b(b,a,c,i,j,k) = -t3b(a,b,c,i,j,k)
                                      t3b(a,b,c,j,i,k) = -t3b(a,b,c,i,j,k)
                                      t3b(b,a,c,j,i,k) = t3b(a,b,c,i,j,k)

                                      resid(a,b,c,i,j,k) = val
                                      resid(b,a,c,i,j,k) = -val
                                      resid(a,b,c,j,i,k) = -val
                                      resid(b,a,c,j,i,k) = val

                                  end do
                              end do
                          end do
                      end do
                  end do
              end do

      end subroutine update_t3b_v2

      subroutine update_t3c_v2(t3c,resid,X3C,fA_oo,fA_vv,fB_oo,fB_vv,shift,noa,nua,nob,nub)

              integer, intent(in) :: noa, nua, nob, nub
              real(kind=8), intent(in) :: fA_oo(1:noa,1:noa), fA_vv(1:nua,1:nua), &
                                  fB_oo(1:nob,1:nob), fB_vv(1:nub,1:nub), &
                                  X3C(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob), shift               
              real(kind=8), intent(inout) :: t3c(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob)
              !f2py intent(in,out) :: t3c(0:nua-1,0:nub-1,0:nub-1,0:noa-1,0:nob-1,0:nob-1)
              real(kind=8), intent(out) :: resid(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob)
              integer :: i, j, k, a, b, c, ii, jj, kk, aa, bb, cc
              real(kind=8) :: denom, val

              resid = 0.0d0
              do ii = 1,noa
                  do jj = 1,nob
                      do kk = jj+1,nob
                          do aa = 1,nua
                              do bb = 1,nub
                                  do cc = bb+1,nub
                  
                                      a = aa; b = cc; c = bb;
                                      i = ii; j = kk; k = jj;

                                      denom = fA_oo(i,i)+fB_oo(j,j)+fB_oo(k,k)-fA_vv(a,a)-fB_vv(b,b)-fB_vv(c,c)
                                      val = X3C(a,b,c,i,j,k) - X3C(a,c,b,i,j,k) - X3C(a,b,c,i,k,j) + X3C(a,c,b,i,k,j)
                                      val = val/(denom-shift)
                                      t3c(a,b,c,i,j,k) = t3c(a,b,c,i,j,k) + val
                                      t3c(a,c,b,i,j,k) = -t3c(a,b,c,i,j,k)
                                      t3c(a,b,c,i,k,j) = -t3c(a,b,c,i,j,k)
                                      t3c(a,c,b,i,k,j) = t3c(a,b,c,i,j,k)

                                      resid(a,b,c,i,j,k) = val
                                      resid(a,c,b,i,j,k) = -val
                                      resid(a,b,c,i,k,j) = -val
                                      resid(a,c,b,i,k,j) = val

                                  end do
                              end do
                          end do
                      end do
                  end do
              end do

      end subroutine update_t3c_v2

      subroutine update_t3d_v2(t3d,resid,X3D,fB_oo,fB_vv,shift,nob,nub)

              integer, intent(in) :: nob, nub
              real(kind=8), intent(in) :: fB_oo(1:nob,1:nob), fB_vv(1:nub,1:nub), &
                                  X3D(1:nub,1:nub,1:nub,1:nob,1:nob,1:nob), shift               
              real(kind=8), intent(inout) :: t3d(1:nub,1:nub,1:nub,1:nob,1:nob,1:nob)
              !f2py intent(in,out) :: t3d(0:nub-1,0:nub-1,0:nub-1,0:nob-1,0:nob-1,0:nob-1)
              real(kind=8), intent(out) :: resid(1:nub,1:nub,1:nub,1:nob,1:nob,1:nob)
              integer :: i, j, k, a, b, c, ii, jj, kk, aa, bb, cc
              real(kind=8) :: denom, val

              resid = 0.0d0
              do ii = 1,nob
                  do jj = ii+1,nob
                      do kk = jj+1,nob
                          do aa = 1,nub
                              do bb = aa+1,nub
                                  do cc = bb+1,nub

                                      A = cc; B = bb; C = aa;
                                      I = kk; J = jj; K = ii;
                                      
                                      denom = fB_oo(I,I)+fB_oo(J,J)+fB_oo(K,K)-fB_vv(A,A)-fB_vv(B,B)-fB_vv(C,C)

                                      val = X3D(a,b,c,i,j,k)&
                                              -X3D(b,a,c,i,j,k)&
                                              -X3D(a,c,b,i,j,k)&
                                              +X3D(b,c,a,i,j,k)&
                                              -X3D(c,b,a,i,j,k)&
                                              +X3D(c,a,b,i,j,k)&
                                              -X3D(a,b,c,j,i,k)&
                                              +X3D(b,a,c,j,i,k)&
                                              +X3D(a,c,b,j,i,k)&
                                              -X3D(b,c,a,j,i,k)&
                                              +X3D(c,b,a,j,i,k)&
                                              -X3D(c,a,b,j,i,k)&
                                              -X3D(a,b,c,i,k,j)&
                                              +X3D(b,a,c,i,k,j)&
                                              +X3D(a,c,b,i,k,j)&
                                              -X3D(b,c,a,i,k,j)&
                                              +X3D(c,b,a,i,k,j)&
                                              -X3D(c,a,b,i,k,j)&
                                              -X3D(a,b,c,k,j,i)&
                                              +X3D(b,a,c,k,j,i)&
                                              +X3D(a,c,b,k,j,i)&
                                              -X3D(b,c,a,k,j,i)&
                                              +X3D(c,b,a,k,j,i)&
                                              -X3D(c,a,b,k,j,i)&
                                              +X3D(a,b,c,j,k,i)&
                                              -X3D(b,a,c,j,k,i)&
                                              -X3D(a,c,b,j,k,i)&
                                              +X3D(b,c,a,j,k,i)&
                                              -X3D(c,b,a,j,k,i)&
                                              +X3D(c,a,b,j,k,i)&
                                              +X3D(a,b,c,k,i,j)&
                                              -X3D(b,a,c,k,i,j)&
                                              -X3D(a,c,b,k,i,j)&
                                              +X3D(b,c,a,k,i,j)&
                                              -X3D(c,b,a,k,i,j)&
                                              +X3D(c,a,b,k,i,j)
                                      val = val/(denom-shift)

                                      t3d(A,B,C,I,J,K) = t3d(A,B,C,I,J,K) + val                           
                                      t3d(A,B,C,K,I,J) = t3d(A,B,C,I,J,K)
                                      t3d(A,B,C,J,K,I) = t3d(A,B,C,I,J,K)
                                      t3d(A,B,C,I,K,J) = -t3d(A,B,C,I,J,K)
                                      t3d(A,B,C,J,I,K) = -t3d(A,B,C,I,J,K)
                                      t3d(A,B,C,K,J,I) = -t3d(A,B,C,I,J,K)
                                      
                                      t3d(B,A,C,I,J,K) = -t3d(A,B,C,I,J,K)
                                      t3d(B,A,C,K,I,J) = -t3d(A,B,C,I,J,K)
                                      t3d(B,A,C,J,K,I) = -t3d(A,B,C,I,J,K)
                                      t3d(B,A,C,I,K,J) = t3d(A,B,C,I,J,K)
                                      t3d(B,A,C,J,I,K) = t3d(A,B,C,I,J,K)
                                      t3d(B,A,C,K,J,I) = t3d(A,B,C,I,J,K)
                                      
                                      t3d(A,C,B,I,J,K) = -t3d(A,B,C,I,J,K)
                                      t3d(A,C,B,K,I,J) = -t3d(A,B,C,I,J,K)
                                      t3d(A,C,B,J,K,I) = -t3d(A,B,C,I,J,K)
                                      t3d(A,C,B,I,K,J) = t3d(A,B,C,I,J,K)
                                      t3d(A,C,B,J,I,K) = t3d(A,B,C,I,J,K)
                                      t3d(A,C,B,K,J,I) = t3d(A,B,C,I,J,K)
                                      
                                      t3d(C,B,A,I,J,K) = -t3d(A,B,C,I,J,K)
                                      t3d(C,B,A,K,I,J) = -t3d(A,B,C,I,J,K)
                                      t3d(C,B,A,J,K,I) = -t3d(A,B,C,I,J,K)
                                      t3d(C,B,A,I,K,J) = t3d(A,B,C,I,J,K)
                                      t3d(C,B,A,J,I,K) = t3d(A,B,C,I,J,K)
                                      t3d(C,B,A,K,J,I) = t3d(A,B,C,I,J,K)
                                      
                                      t3d(B,C,A,I,J,K) = t3d(A,B,C,I,J,K)
                                      t3d(B,C,A,K,I,J) = t3d(A,B,C,I,J,K)
                                      t3d(B,C,A,J,K,I) = t3d(A,B,C,I,J,K)
                                      t3d(B,C,A,I,K,J) = -t3d(A,B,C,I,J,K)
                                      t3d(B,C,A,J,I,K) = -t3d(A,B,C,I,J,K)
                                      t3d(B,C,A,K,J,I) = -t3d(A,B,C,I,J,K)
                                      
                                      t3d(C,A,B,I,J,K) = t3d(A,B,C,I,J,K)
                                      t3d(C,A,B,K,I,J) = t3d(A,B,C,I,J,K)
                                      t3d(C,A,B,J,K,I) = t3d(A,B,C,I,J,K)
                                      t3d(C,A,B,I,K,J) = -t3d(A,B,C,I,J,K)
                                      t3d(C,A,B,J,I,K) = -t3d(A,B,C,I,J,K)
                                      t3d(C,A,B,K,J,I) = -t3d(A,B,C,I,J,K)

                                      resid(A,B,C,I,J,K) = val                            
                                      resid(A,B,C,K,I,J) = val
                                      resid(A,B,C,J,K,I) = val
                                      resid(A,B,C,I,K,J) = -val
                                      resid(A,B,C,J,I,K) = -val
                                      resid(A,B,C,K,J,I) = -val
                                      resid(B,C,A,I,J,K) = val                            
                                      resid(B,C,A,K,I,J) = val
                                      resid(B,C,A,J,K,I) = val
                                      resid(B,C,A,I,K,J) = -val
                                      resid(B,C,A,J,I,K) = -val
                                      resid(B,C,A,K,J,I) = -val
                                      resid(C,A,B,I,J,K) = val                            
                                      resid(C,A,B,K,I,J) = val
                                      resid(C,A,B,J,K,I) = val
                                      resid(C,A,B,I,K,J) = -val
                                      resid(C,A,B,J,I,K) = -val
                                      resid(C,A,B,K,J,I) = -val
                                      resid(A,C,B,I,J,K) = -val                            
                                      resid(A,C,B,K,I,J) = -val
                                      resid(A,C,B,J,K,I) = -val
                                      resid(A,C,B,I,K,J) = val
                                      resid(A,C,B,J,I,K) = val
                                      resid(A,C,B,K,J,I) = val
                                      resid(B,A,C,I,J,K) = -val                            
                                      resid(B,A,C,K,I,J) = -val
                                      resid(B,A,C,J,K,I) = -val
                                      resid(B,A,C,I,K,J) = val
                                      resid(B,A,C,J,I,K) = val
                                      resid(B,A,C,K,J,I) = val
                                      resid(C,B,A,I,J,K) = -val                            
                                      resid(C,B,A,K,I,J) = -val
                                      resid(C,B,A,J,K,I) = -val
                                      resid(C,B,A,I,K,J) = val
                                      resid(C,B,A,J,I,K) = val
                                      resid(C,B,A,K,J,I) = val
                                  end do
                              end do
                          end do
                      end do
                  end do
              end do

      end subroutine update_t3d_v2

      subroutine update_R(r1a,r1b,r2a,r2b,r2c,omega,&
                          H1A_oo,H1A_vv,H1B_oo,H1B_vv,&
                          shift,noa,nua,nob,nub)

              implicit none

              integer, intent(in) :: noa, nua, nob, nub
              real(kind=8), intent(in) :: H1A_oo(1:noa,1:noa), H1A_vv(1:nua,1:nua), &
                                  H1B_oo(1:nob,1:nob), H1B_vv(1:nub,1:nub), shift, &
                                  omega  
              real(kind=8), intent(inout) :: r1a(1:nua,1:noa)
              !f2py intent(in,out) :: r1a(0:nua-1,0:noa-1) 
              real(kind=8), intent(inout) :: r1b(1:nub,1:nob)
              !f2py intent(in,out) :: r1b(0:nub-1,0:nob-1)   
              real(kind=8), intent(inout) :: r2a(1:nua,1:nua,1:noa,1:noa)
              !f2py intent(in,out) :: r2a(0:nua-1,0:nua-1,0:noa-1,0:noa-1)        
              real(kind=8), intent(inout) :: r2b(1:nua,1:nub,1:noa,1:nob)
              !f2py intent(in,out) :: r2b(0:nua-1,0:nub-1,0:noa-1,0:nob-1)
              real(kind=8), intent(inout) :: r2c(1:nub,1:nub,1:nob,1:nob)
              !f2py intent(in,out) :: r2c(0:nub-1,0:nub-1,0:nob-1,0:nob-1)
              integer :: i, j, a, b
              real(kind=8) :: denom

              do i = 1,noa
                do a = 1,nua
                  denom = H1A_vv(a,a) - H1A_oo(i,i)
                  r1a(a,i) = r1a(a,i)/(omega-denom)
                end do
              end do

              do i = 1,nob
                do a = 1,nub
                  denom = H1B_vv(a,a) - H1B_oo(i,i)
                  r1b(a,i) = r1b(a,i)/(omega-denom)
                end do
              end do

              do i = 1,noa
                do j = 1,noa
                  do a = 1,nua
                    do b = 1,nua
                      denom = H1A_vv(a,a) + H1A_vv(b,b) - H1A_oo(i,i) - H1A_oo(j,j)
                      r2a(b,a,j,i) = r2a(b,a,j,i)/(omega-denom)
                      !r2a(a,b,j,i) = -r2a(b,a,j,i)
                      !r2a(b,a,i,j) = -r2a(b,a,j,i)
                      !r2a(a,b,i,j) = r2a(b,a,j,i)
                    end do
                  end do
                end do
              end do

              do j = 1,nob
                do i = 1,noa
                  do b = 1,nub
                    do a = 1,nua
                      denom = H1A_vv(a,a) + H1B_vv(b,b) - H1A_oo(i,i) - H1B_oo(j,j)
                      r2b(a,b,i,j) = r2b(a,b,i,j)/(omega-denom)
                    end do
                  end do
                end do
              end do

              do i = 1,nob
                do j = 1,nob
                  do a = 1,nub
                    do b = 1,nub
                      denom = H1B_vv(a,a) + H1B_vv(b,b) - H1B_oo(i,i) - H1B_oo(j,j)
                      r2c(b,a,j,i) = r2c(b,a,j,i)/(omega-denom)
                      !r2c(a,b,j,i) = -r2c(b,a,j,i)
                      !r2c(b,a,i,j) = -r2c(b,a,j,i)
                      !r2c(a,b,i,j) = r2c(b,a,j,i)
                    end do
                  end do
                end do
              end do

      end subroutine update_R


      subroutine update_R_ccsdt(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,&
                      omega,H1A_oo,H1A_vv,H1B_oo,H1B_vv,shift,noa,nua,nob,nub)

              implicit none

              integer, intent(in) :: noa, nua, nob, nub
              real(kind=8), intent(in) :: H1A_oo(1:noa,1:noa), H1A_vv(1:nua,1:nua), &
                                  H1B_oo(1:nob,1:nob), H1B_vv(1:nub,1:nub), shift, &
                                  omega  
              real(kind=8), intent(inout) :: r1a(1:nua,1:noa)
              !f2py intent(in,out) :: r1a(0:nua-1,0:noa-1) 
              real(kind=8), intent(inout) :: r1b(1:nub,1:nob)
              !f2py intent(in,out) :: r1b(0:nub-1,0:nob-1)   
              real(kind=8), intent(inout) :: r2a(1:nua,1:nua,1:noa,1:noa)
              !f2py intent(in,out) :: r2a(0:nua-1,0:nua-1,0:noa-1,0:noa-1)        
              real(kind=8), intent(inout) :: r2b(1:nua,1:nub,1:noa,1:nob)
              !f2py intent(in,out) :: r2b(0:nua-1,0:nub-1,0:noa-1,0:nob-1)
              real(kind=8), intent(inout) :: r2c(1:nub,1:nub,1:nob,1:nob)
              !f2py intent(in,out) :: r2c(0:nub-1,0:nub-1,0:nob-1,0:nob-1)
              real(kind=8), intent(inout) :: r3a(1:nua,1:nua,1:nua,1:noa,1:noa,1:noa)
              !f2py intent(in,out) :: r3a(0:nua-1,0:nua-1,0:nua-1,0:noa-1,0:noa-1,0:noa-1)
              real(kind=8), intent(inout) :: r3b(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob)
              !f2py intent(in,out) :: r3b(0:nua-1,0:nua-1,0:nub-1,0:noa-1,0:noa-1,0:nob-1)
              real(kind=8), intent(inout) :: r3c(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob)
              !f2py intent(in,out) :: r3c(0:nua-1,0:nub-1,0:nub-1,0:noa-1,0:nob-1,0:nob-1)
              real(kind=8), intent(inout) :: r3d(1:nub,1:nub,1:nub,1:nob,1:nob,1:nob)
              !f2py intent(in,out) :: r3d(0:nub-1,0:nub-1,0:nub-1,0:nob-1,0:nob-1,0:nob-1)
              integer :: i, j, k, a, b, c
              real(kind=8) :: denom

              do i = 1,noa
                do a = 1,nua
                  denom = H1A_vv(a,a) - H1A_oo(i,i)
                  r1a(a,i) = r1a(a,i)/(omega-denom+shift)
                end do
              end do

              do i = 1,nob
                do a = 1,nub
                  denom = H1B_vv(a,a) - H1B_oo(i,i)
                  r1b(a,i) = r1b(a,i)/(omega-denom+shift)
                end do
              end do

              do i = 1,noa
                do j = 1,noa
                  do a = 1,nua
                    do b = 1,nua
                      denom = H1A_vv(a,a) + H1A_vv(b,b) - H1A_oo(i,i) - H1A_oo(j,j)
                      r2a(b,a,j,i) = r2a(b,a,j,i)/(omega-denom+shift)
                    end do
                  end do
                end do
              end do

              do j = 1,nob
                do i = 1,noa
                  do b = 1,nub
                    do a = 1,nua
                      denom = H1A_vv(a,a) + H1B_vv(b,b) - H1A_oo(i,i) - H1B_oo(j,j)
                      r2b(a,b,i,j) = r2b(a,b,i,j)/(omega-denom+shift)
                    end do
                  end do
                end do
              end do

              do i = 1,nob
                do j = 1,nob
                  do a = 1,nub
                    do b = 1,nub
                      denom = H1B_vv(a,a) + H1B_vv(b,b) - H1B_oo(i,i) - H1B_oo(j,j)
                      r2c(b,a,j,i) = r2c(b,a,j,i)/(omega-denom+shift)
                    end do
                  end do
                end do
              end do

              do i = 1,noa
                do j = 1,noa
                   do k = 1,noa
                     do a = 1,nua
                       do b = 1,nua
                          do c = 1,nua
                            denom = H1A_vv(a,a) + H1A_vv(b,b) + H1A_vv(c,c)&
                                    -H1A_oo(i,i) - H1A_oo(j,j) - H1A_oo(k,k)
                            r3a(a,b,c,i,j,k) = r3a(a,b,c,i,j,k)/(omega-denom+shift)
                          end do
                       end do
                    end do
                  end do
                end do
              end do

              do i = 1,noa
                do j = 1,noa
                   do k = 1,nob
                     do a = 1,nua
                       do b = 1,nua
                          do c = 1,nub
                            denom = H1A_vv(a,a) + H1A_vv(b,b) + H1B_vv(c,c)&
                                    -H1A_oo(i,i) - H1A_oo(j,j) - H1B_oo(k,k)
                            r3b(a,b,c,i,j,k) = r3b(a,b,c,i,j,k)/(omega-denom+shift)
                          end do
                       end do
                    end do
                  end do
                end do
              end do

              do i = 1,noa
                do j = 1,nob
                   do k = 1,nob
                     do a = 1,nua
                       do b = 1,nub
                          do c = 1,nub
                            denom = H1A_vv(a,a) + H1B_vv(b,b) + H1B_vv(c,c)&
                                    -H1A_oo(i,i) - H1B_oo(j,j) - H1B_oo(k,k)
                            r3c(a,b,c,i,j,k) = r3c(a,b,c,i,j,k)/(omega-denom+shift)
                          end do
                       end do
                    end do
                  end do
                end do
              end do

              do i = 1,nob
                do j = 1,nob
                   do k = 1,nob
                     do a = 1,nub
                       do b = 1,nub
                          do c = 1,nub
                            denom = H1B_vv(a,a) + H1B_vv(b,b) + H1B_vv(c,c)&
                                    -H1B_oo(i,i) - H1B_oo(j,j) - H1B_oo(k,k)
                            r3d(a,b,c,i,j,k) = r3d(a,b,c,i,j,k)/(omega-denom+shift)
                          end do
                       end do
                    end do
                  end do
                end do
              end do

      end subroutine update_R_ccsdt
     
      subroutine update_L1(l1a, l1b, X1A, X1B,&
                           omega,&
                           H1A_oo, H1A_vv, H1B_oo, H1B_vv,&
                           shift,&
                           noa, nua, nob, nub)

              implicit none

              integer, intent(in) :: noa, nua, nob, nub
              real(kind=8), intent(in) :: H1A_oo(1:noa,1:noa), H1A_vv(1:nua,1:nua),&
                                     H1B_oo(1:nob,1:nob), H1B_vv(1:nub,1:nub),&
                                     shift, omega

              real(kind=8), intent(inout) :: l1a(1:nua,1:noa)
              !f2py intent(in,out) :: l1a(0:nua-1,0:noa-1)
              real(kind=8), intent(inout) :: l1b(1:nub,1:nob)
              !f2py intent(in,out) :: l1b(0:nub-1,0:nob-1)

              real(kind=8), intent(inout) :: X1A(1:nua,1:noa)
              !f2py intent(in,out) :: X1A(0:nua-1,0:noa-1)
              real(kind=8), intent(inout) :: X1B(1:nub,1:nob)
              !f2py intent(in,out) :: X1B(0:nub-1,0:nob-1)

              integer :: i, a
              real(kind=8) :: denom, val

              do i = 1,noa
                do a = 1,nua
                  denom = H1A_vv(a,a) - H1A_oo(i,i)
                  val = omega*l1a(a,i) - X1A(a,i)
                  l1a(a,i) = l1a(a,i) + val/(denom - omega + shift)
                  X1A(a,i) = val/(denom - omega + shift)
                end do
              end do

              do i = 1,nob
                do a = 1,nub
                  denom = H1B_vv(a,a) - H1B_oo(i,i)
                  val = omega*l1b(a,i) - X1B(a,i)
                  l1b(a,i) = l1b(a,i) + val/(denom - omega + shift)
                  X1B(a,i) = val/(denom - omega + shift)
                end do
              end do

      end subroutine update_L1

      subroutine update_L2(l2a, l2b, l2c, X2A, X2B, X2C,&
                           omega,&
                           H1A_oo, H1A_vv, H1B_oo, H1B_vv,&
                           shift,&
                           noa, nua, nob, nub)

              implicit none

              integer, intent(in) :: noa, nua, nob, nub
              real(kind=8), intent(in) :: H1A_oo(1:noa,1:noa), H1A_vv(1:nua,1:nua), &
                                     H1B_oo(1:nob,1:nob), H1B_vv(1:nub,1:nub), shift, omega

              real(kind=8), intent(inout) :: l2a(1:nua,1:nua,1:noa,1:noa)
              !f2py intent(in,out) :: l2a(0:nua-1,0:nua-1,0:noa-1,0:noa-1)
              real(kind=8), intent(inout) :: l2b(1:nua,1:nub,1:noa,1:nob)
              !f2py intent(in,out) :: l2b(0:nua-1,0:nub-1,0:noa-1,0:nob-1)
              real(kind=8), intent(inout) :: l2c(1:nub,1:nub,1:nob,1:nob)
              !f2py intent(in,out) :: l2c(0:nub-1,0:nub-1,0:nob-1,0:nob-1)

              real(kind=8), intent(inout) :: X2A(1:nua,1:nua,1:noa,1:noa)
              !f2py intent(in,out) :: X2A(0:nua-1,0:nua-1,0:noa-1,0:noa-1)
              real(kind=8), intent(inout) :: X2B(1:nua,1:nub,1:noa,1:nob)
              !f2py intent(in,out) :: X2B(0:nua-1,0:nub-1,0:noa-1,0:nob-1)
              real(kind=8), intent(inout) :: X2C(1:nub,1:nub,1:nob,1:nob)
              !f2py intent(in,out) :: X2C(0:nub-1,0:nub-1,0:nob-1,0:nob-1)

              integer :: i, j, a, b
              real(kind=8) :: denom, val

              do i = 1, noa
                do j = i+1, noa
                  do a = 1, nua
                    do b = a+1, nua
                      denom = H1A_vv(a,a) + H1A_vv(b,b) - H1A_oo(i,i) - H1A_oo(j,j)

                      val = omega*l2a(a,b,i,j) - X2A(a,b,i,j)

                      l2a(a,b,i,j) = l2a(a,b,i,j) + val/(denom - omega + shift)
                      l2a(b,a,i,j) = -l2a(a,b,i,j)
                      l2a(a,b,j,i) = -l2a(a,b,i,j)
                      l2a(b,a,j,i) = l2a(a,b,i,j)

                      X2A(a,b,i,j) = val/(denom - omega + shift)
                      X2A(b,a,i,j) = -X2A(a,b,i,j)
                      X2A(a,b,j,i) = -X2A(a,b,i,j)
                      X2A(b,a,j,i) = X2A(a,b,i,j)
                    end do
                  end do
                end do
              end do

              do j = 1, nob
                do i = 1, noa
                  do b = 1, nub
                    do a = 1, nua
                      denom = H1A_vv(a,a) + H1B_vv(b,b) - H1A_oo(i,i) - H1B_oo(j,j)

                      val = omega*l2b(a,b,i,j) - X2B(a,b,i,j)

                      l2b(a,b,i,j) = l2b(a,b,i,j) + val/(denom - omega + shift)
                      X2B(a,b,i,j) = val/(denom - omega + shift)
                    end do
                  end do
                end do
              end do

              do i = 1, nob
                do j = i+1, nob
                  do a = 1, nub
                    do b = a+1, nub
                      denom = H1B_vv(a,a) + H1B_vv(b,b) - H1B_oo(i,i) - H1B_oo(j,j)

                      val = omega*l2c(a,b,i,j) - X2C(a,b,i,j)

                      l2c(a,b,i,j) = l2c(a,b,i,j) + val/(denom - omega + shift)
                      l2c(b,a,i,j) = -l2c(a,b,i,j)
                      l2c(a,b,j,i) = -l2c(a,b,i,j)
                      l2c(b,a,j,i) = l2c(a,b,i,j)

                      X2C(a,b,i,j) = val/(denom - omega + shift)
                      X2C(b,a,i,j) = -X2C(a,b,i,j)
                      X2C(a,b,j,i) = -X2C(a,b,i,j)
                      X2C(b,a,j,i) = X2C(a,b,i,j)
                    end do
                  end do
                end do
              end do

      end subroutine update_L2

      subroutine update_L3(l3a,l3b,l3c,l3d,X3A,X3B,X3C,X3D,&
                           omega,&
                           H1A_oo,H1A_vv,H1B_oo,H1B_vv,&
                           shift,&
                           noa,nua,nob,nub)

              implicit none

              integer, intent(in) :: noa, nua, nob, nub
              real(kind=8), intent(in) :: H1A_oo(1:noa,1:noa), H1A_vv(1:nua,1:nua), &
                                     H1B_oo(1:nob,1:nob), H1B_vv(1:nub,1:nub), shift, omega

              real(kind=8), intent(inout) :: l3a(1:nua,1:nua,1:nua,1:noa,1:noa,1:noa)
              !f2py intent(in,out) :: l3a(0:nua-1,0:nua-1,0:nua-1,0:noa-1,0:noa-1,0:noa-1)
              real(kind=8), intent(inout) :: l3b(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob)
              !f2py intent(in,out) :: l3b(0:nua-1,0:nua-1,0:nub-1,0:noa-1,0:noa-1,0:nob-1)
              real(kind=8), intent(inout) :: l3c(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob)
              !f2py intent(in,out) :: l3c(0:nua-1,0:nub-1,0:nub-1,0:noa-1,0:nob-1,0:nob-1)
              real(kind=8), intent(inout) :: l3d(1:nub,1:nub,1:nub,1:nob,1:nob,1:nob)
              !f2py intent(in,out) :: l3d(0:nub-1,0:nub-1,0:nub-1,0:nob-1,0:nob-1,0:nob-1)

              real(kind=8), intent(inout) :: X3A(1:nua,1:nua,1:nua,1:noa,1:noa,1:noa)
              !f2py intent(in,out) :: X3A(0:nua-1,0:nua-1,0:nua-1,0:noa-1,0:noa-1,0:noa-1)
              real(kind=8), intent(inout) :: X3B(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob)
              !f2py intent(in,out) :: X3B(0:nua-1,0:nua-1,0:nub-1,0:noa-1,0:noa-1,0:nob-1)
              real(kind=8), intent(inout) :: X3C(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob)
              !f2py intent(in,out) :: X3C(0:nua-1,0:nub-1,0:nub-1,0:noa-1,0:nob-1,0:nob-1)
              real(kind=8), intent(inout) :: X3D(1:nub,1:nub,1:nub,1:nob,1:nob,1:nob)
              !f2py intent(in,out) :: X3D(0:nub-1,0:nub-1,0:nub-1,0:nob-1,0:nob-1,0:nob-1)

              integer :: i, j, k, a, b, c
              real(kind=8) :: denom, val


              do i = 1,noa
                  do j = i + 1,noa
                      do k = j + 1,noa
                          do a = 1,nua
                              do b = a + 1,nua
                                  do c = b + 1,nua

                                      denom = -H1A_oo(I,I)-H1A_oo(J,J)-H1A_oo(K,K)+H1A_vv(A,A)+H1A_vv(B,B)+H1A_vv(C,C)
                                      val = omega*l3a(a,b,c,i,j,k) - X3A(a,b,c,i,j,k)

                                      l3a(a,b,c,i,j,k) = l3a(a,b,c,i,j,k) + val/(denom - omega + shift)
                                      l3a(A,B,C,K,I,J) = l3a(A,B,C,I,J,K)
                                      l3a(A,B,C,J,K,I) = l3a(A,B,C,I,J,K)
                                      l3a(A,B,C,I,K,J) = -l3a(A,B,C,I,J,K)
                                      l3a(A,B,C,J,I,K) = -l3a(A,B,C,I,J,K)
                                      l3a(A,B,C,K,J,I) = -l3a(A,B,C,I,J,K)

                                      l3a(B,A,C,I,J,K) = -l3a(A,B,C,I,J,K)
                                      l3a(B,A,C,K,I,J) = -l3a(A,B,C,I,J,K)
                                      l3a(B,A,C,J,K,I) = -l3a(A,B,C,I,J,K)
                                      l3a(B,A,C,I,K,J) = l3a(A,B,C,I,J,K)
                                      l3a(B,A,C,J,I,K) = l3a(A,B,C,I,J,K)
                                      l3a(B,A,C,K,J,I) = l3a(A,B,C,I,J,K)

                                      l3a(A,C,B,I,J,K) = -l3a(A,B,C,I,J,K)
                                      l3a(A,C,B,K,I,J) = -l3a(A,B,C,I,J,K)
                                      l3a(A,C,B,J,K,I) = -l3a(A,B,C,I,J,K)
                                      l3a(A,C,B,I,K,J) = l3a(A,B,C,I,J,K)
                                      l3a(A,C,B,J,I,K) = l3a(A,B,C,I,J,K)
                                      l3a(A,C,B,K,J,I) = l3a(A,B,C,I,J,K)

                                      l3a(C,B,A,I,J,K) = -l3a(A,B,C,I,J,K)
                                      l3a(C,B,A,K,I,J) = -l3a(A,B,C,I,J,K)
                                      l3a(C,B,A,J,K,I) = -l3a(A,B,C,I,J,K)
                                      l3a(C,B,A,I,K,J) = l3a(A,B,C,I,J,K)
                                      l3a(C,B,A,J,I,K) = l3a(A,B,C,I,J,K)
                                      l3a(C,B,A,K,J,I) = l3a(A,B,C,I,J,K)

                                      l3a(B,C,A,I,J,K) = l3a(A,B,C,I,J,K)
                                      l3a(B,C,A,K,I,J) = l3a(A,B,C,I,J,K)
                                      l3a(B,C,A,J,K,I) = l3a(A,B,C,I,J,K)
                                      l3a(B,C,A,I,K,J) = -l3a(A,B,C,I,J,K)
                                      l3a(B,C,A,J,I,K) = -l3a(A,B,C,I,J,K)
                                      l3a(B,C,A,K,J,I) = -l3a(A,B,C,I,J,K)

                                      l3a(C,A,B,I,J,K) = l3a(A,B,C,I,J,K)
                                      l3a(C,A,B,K,I,J) = l3a(A,B,C,I,J,K)
                                      l3a(C,A,B,J,K,I) = l3a(A,B,C,I,J,K)
                                      l3a(C,A,B,I,K,J) = -l3a(A,B,C,I,J,K)
                                      l3a(C,A,B,J,I,K) = -l3a(A,B,C,I,J,K)
                                      l3a(C,A,B,K,J,I) = -l3a(A,B,C,I,J,K)

                                      val = val/(denom - omega + shift)
                                      X3A(A,B,C,I,J,K) = val
                                      X3A(A,B,C,K,I,J) = val
                                      X3A(A,B,C,J,K,I) = val
                                      X3A(A,B,C,I,K,J) = -val
                                      X3A(A,B,C,J,I,K) = -val
                                      X3A(A,B,C,K,J,I) = -val

                                      X3A(B,C,A,I,J,K) = val
                                      X3A(B,C,A,K,I,J) = val
                                      X3A(B,C,A,J,K,I) = val
                                      X3A(B,C,A,I,K,J) = -val
                                      X3A(B,C,A,J,I,K) = -val
                                      X3A(B,C,A,K,J,I) = -val

                                      X3A(C,A,B,I,J,K) = val
                                      X3A(C,A,B,K,I,J) = val
                                      X3A(C,A,B,J,K,I) = val
                                      X3A(C,A,B,I,K,J) = -val
                                      X3A(C,A,B,J,I,K) = -val
                                      X3A(C,A,B,K,J,I) = -val

                                      X3A(A,C,B,I,J,K) = -val
                                      X3A(A,C,B,K,I,J) = -val
                                      X3A(A,C,B,J,K,I) = -val
                                      X3A(A,C,B,I,K,J) = val
                                      X3A(A,C,B,J,I,K) = val
                                      X3A(A,C,B,K,J,I) = val

                                      X3A(B,A,C,I,J,K) = -val
                                      X3A(B,A,C,K,I,J) = -val
                                      X3A(B,A,C,J,K,I) = -val
                                      X3A(B,A,C,I,K,J) = val
                                      X3A(B,A,C,J,I,K) = val
                                      X3A(B,A,C,K,J,I) = val

                                      X3A(C,B,A,I,J,K) = -val
                                      X3A(C,B,A,K,I,J) = -val
                                      X3A(C,B,A,J,K,I) = -val
                                      X3A(C,B,A,I,K,J) = val
                                      X3A(C,B,A,J,I,K) = val
                                      X3A(C,B,A,K,J,I) = val
                                  end do
                              end do
                          end do
                      end do
                  end do
              end do

              do i = 1,noa
                  do j = i + 1,noa
                      do k = 1,nob
                          do a = 1,nua
                              do b = a + 1,nua
                                  do c = 1,nub

                                      denom = -H1A_oo(I,I)-H1A_oo(J,J)-H1B_oo(K,K)+H1A_vv(A,A)+H1A_vv(B,B)+H1B_vv(C,C)
                                      val = omega*l3b(a,b,c,i,j,k) - X3B(a,b,c,i,j,k)

                                      l3b(a,b,c,i,j,k) = l3b(a,b,c,i,j,k) + val/(denom - omega + shift)
                                      l3b(b, a, c, i, j, k) = -1.0 * l3b(A,B,C,I,J,K)
                                      l3b(a, b, c, j, i, k) = -1.0 * l3b(A,B,C,I,J,K)
                                      l3b(b, a, c, j, i, k) = l3b(A,B,C,I,J,K)

                                      val = val/(denom - omega + shift)
                                      X3B(A,B,C,I,J,K) = val
                                      X3B(B,A,C,I,J,K) = -1.0 * val
                                      X3B(A,B,C,J,I,K) = -1.0 * val
                                      X3B(B,A,C,J,I,K) = val
                                  end do
                              end do
                          end do
                      end do
                  end do
              end do

              do i = 1,noa
                  do j = 1,nob
                      do k = j+1,nob
                          do a = 1,nua
                              do b = 1,nub
                                  do c = b+1,nub

                                      denom = -H1A_oo(I,I)-H1B_oo(J,J)-H1B_oo(K,K)+H1A_vv(A,A)+H1B_vv(B,B)+H1B_vv(C,C)
                                      val = omega*l3c(a,b,c,i,j,k) - X3C(a,b,c,i,j,k)

                                      l3c(a,b,c,i,j,k) = l3c(a,b,c,i,j,k) + val/(denom - omega + shift)
                                      l3c(a, c, b, i, j, k) = -1.0 * l3c(A,B,C,I,J,K)
                                      l3c(a, b, c, i, k, j) = -1.0 * l3c(A,B,C,I,J,K)
                                      l3c(a, c, b, i, k, j) = l3c(A,B,C,I,J,K)

                                      val = val/(denom - omega + shift)
                                      X3C(A,B,C,I,J,K) = val
                                      X3C(A,C,B,I,J,K) = -1.0 * val
                                      X3C(A,B,C,I,K,J) = -1.0 * val
                                      X3C(A,C,B,I,K,J) = val
                                  end do
                              end do
                          end do
                      end do
                  end do
              end do

              do i = 1,nob
                  do j = i+1,nob
                      do k = j+1,nob
                          do a = 1,nub
                              do b = a+1,nub
                                  do c = b+1,nub

                                      denom = -H1B_oo(I,I)-H1B_oo(J,J)-H1B_oo(K,K)+H1B_vv(A,A)+H1B_vv(B,B)+H1B_vv(C,C)
                                      val = omega*l3d(a,b,c,i,j,k) - X3D(a,b,c,i,j,k)

                                      l3d(a,b,c,i,j,k) = l3d(a,b,c,i,j,k) + val/(denom - omega + shift)
                                      l3d(A,B,C,K,I,J) = l3d(A,B,C,I,J,K)
                                      l3d(A,B,C,J,K,I) = l3d(A,B,C,I,J,K)
                                      l3d(A,B,C,I,K,J) = -l3d(A,B,C,I,J,K)
                                      l3d(A,B,C,J,I,K) = -l3d(A,B,C,I,J,K)
                                      l3d(A,B,C,K,J,I) = -l3d(A,B,C,I,J,K)

                                      l3d(B,A,C,I,J,K) = -l3d(A,B,C,I,J,K)
                                      l3d(B,A,C,K,I,J) = -l3d(A,B,C,I,J,K)
                                      l3d(B,A,C,J,K,I) = -l3d(A,B,C,I,J,K)
                                      l3d(B,A,C,I,K,J) = l3d(A,B,C,I,J,K)
                                      l3d(B,A,C,J,I,K) = l3d(A,B,C,I,J,K)
                                      l3d(B,A,C,K,J,I) = l3d(A,B,C,I,J,K)

                                      l3d(A,C,B,I,J,K) = -l3d(A,B,C,I,J,K)
                                      l3d(A,C,B,K,I,J) = -l3d(A,B,C,I,J,K)
                                      l3d(A,C,B,J,K,I) = -l3d(A,B,C,I,J,K)
                                      l3d(A,C,B,I,K,J) = l3d(A,B,C,I,J,K)
                                      l3d(A,C,B,J,I,K) = l3d(A,B,C,I,J,K)
                                      l3d(A,C,B,K,J,I) = l3d(A,B,C,I,J,K)

                                      l3d(C,B,A,I,J,K) = -l3d(A,B,C,I,J,K)
                                      l3d(C,B,A,K,I,J) = -l3d(A,B,C,I,J,K)
                                      l3d(C,B,A,J,K,I) = -l3d(A,B,C,I,J,K)
                                      l3d(C,B,A,I,K,J) = l3d(A,B,C,I,J,K)
                                      l3d(C,B,A,J,I,K) = l3d(A,B,C,I,J,K)
                                      l3d(C,B,A,K,J,I) = l3d(A,B,C,I,J,K)

                                      l3d(B,C,A,I,J,K) = l3d(A,B,C,I,J,K)
                                      l3d(B,C,A,K,I,J) = l3d(A,B,C,I,J,K)
                                      l3d(B,C,A,J,K,I) = l3d(A,B,C,I,J,K)
                                      l3d(B,C,A,I,K,J) = -l3d(A,B,C,I,J,K)
                                      l3d(B,C,A,J,I,K) = -l3d(A,B,C,I,J,K)
                                      l3d(B,C,A,K,J,I) = -l3d(A,B,C,I,J,K)

                                      l3d(C,A,B,I,J,K) = l3d(A,B,C,I,J,K)
                                      l3d(C,A,B,K,I,J) = l3d(A,B,C,I,J,K)
                                      l3d(C,A,B,J,K,I) = l3d(A,B,C,I,J,K)
                                      l3d(C,A,B,I,K,J) = -l3d(A,B,C,I,J,K)
                                      l3d(C,A,B,J,I,K) = -l3d(A,B,C,I,J,K)
                                      l3d(C,A,B,K,J,I) = -l3d(A,B,C,I,J,K)

                                      val = val/(denom - omega + shift)
                                      X3D(A,B,C,I,J,K) = val
                                      X3D(A,B,C,K,I,J) = val
                                      X3D(A,B,C,J,K,I) = val
                                      X3D(A,B,C,I,K,J) = -val
                                      X3D(A,B,C,J,I,K) = -val
                                      X3D(A,B,C,K,J,I) = -val

                                      X3D(B,C,A,I,J,K) = val
                                      X3D(B,C,A,K,I,J) = val
                                      X3D(B,C,A,J,K,I) = val
                                      X3D(B,C,A,I,K,J) = -val
                                      X3D(B,C,A,J,I,K) = -val
                                      X3D(B,C,A,K,J,I) = -val

                                      X3D(C,A,B,I,J,K) = val
                                      X3D(C,A,B,K,I,J) = val
                                      X3D(C,A,B,J,K,I) = val
                                      X3D(C,A,B,I,K,J) = -val
                                      X3D(C,A,B,J,I,K) = -val
                                      X3D(C,A,B,K,J,I) = -val

                                      X3D(A,C,B,I,J,K) = -val
                                      X3D(A,C,B,K,I,J) = -val
                                      X3D(A,C,B,J,K,I) = -val
                                      X3D(A,C,B,I,K,J) = val
                                      X3D(A,C,B,J,I,K) = val
                                      X3D(A,C,B,K,J,I) = val

                                      X3D(B,A,C,I,J,K) = -val
                                      X3D(B,A,C,K,I,J) = -val
                                      X3D(B,A,C,J,K,I) = -val
                                      X3D(B,A,C,I,K,J) = val
                                      X3D(B,A,C,J,I,K) = val
                                      X3D(B,A,C,K,J,I) = val

                                      X3D(C,B,A,I,J,K) = -val
                                      X3D(C,B,A,K,I,J) = -val
                                      X3D(C,B,A,J,K,I) = -val
                                      X3D(C,B,A,I,K,J) = val
                                      X3D(C,B,A,J,I,K) = val
                                      X3D(C,B,A,K,J,I) = val
                                  end do
                              end do
                          end do
                      end do
                  end do
              end do


      end subroutine update_L3

      subroutine update_R_2h1p(r1a,r2a,r2b,omega,H1A_oo,H1A_vv,H1B_oo,H1B_vv,shift,noa,nua,nob,nub)

              implicit none

              integer, intent(in) :: noa, nua, nob, nub
              real(8), intent(in) :: H1A_oo(1:noa,1:noa), H1A_vv(1:nua,1:nua), &
                                     H1B_oo(1:nob,1:nob), H1B_vv(1:nub,1:nub), shift, &
                                     omega
              real(8), intent(inout) :: r1a(1:noa)
              !f2py intent(in,out) :: r1a(0:noa-1)
              real(8), intent(inout) :: r2a(1:noa,1:nua,1:noa)
              !f2py intent(in,out) :: r2a(0:noa-1,0:nua-1,0:noa-1)
              real(8), intent(inout) :: r2b(1:noa,1:nub,1:nob)
              !f2py intent(in,out) :: r2b(0:noa-1,0:nub-1,0:nob-1)
              integer :: i, j, b
              real(8) :: denom

              do i = 1,noa
                  denom = H1A_oo(i,i)
                  r1a(i) = r1a(i)/(omega-denom+shift)
              end do

              do j = 1,noa
                do b = 1,nua
                   do i = 1,noa
                      denom = -H1A_vv(b,b) + H1A_oo(i,i) + H1A_oo(j,j)
                      r2a(i,b,j) = r2a(i,b,j)/(omega-denom+shift)
                  end do
                end do
              end do

              do j = 1,nob
                do b = 1,nub
                   do i = 1,noa
                      denom = -H1B_vv(b,b) + H1A_oo(i,i) + H1B_oo(j,j)
                      r2b(i,b,j) = r2b(i,b,j)/(omega-denom+shift)
                  end do
                end do
              end do

      end subroutine update_R_2h1p

      subroutine update_R_2p1h(r1a,r2a,r2b,omega,H1A_oo,H1A_vv,H1B_oo,H1B_vv,shift,noa,nua,nob,nub)

              implicit none

              integer, intent(in) :: noa, nua, nob, nub
              real(8), intent(in) :: H1A_oo(1:noa,1:noa), H1A_vv(1:nua,1:nua), &
                                     H1B_oo(1:nob,1:nob), H1B_vv(1:nub,1:nub), shift, &
                                     omega
              real(8), intent(inout) :: r1a(1:nua)
              !f2py intent(in,out) :: r1a(0:nua-1)
              real(8), intent(inout) :: r2a(1:nua,1:nua,1:noa)
              !f2py intent(in,out) :: r2a(0:nua-1,0:nua-1,0:noa-1)
              real(8), intent(inout) :: r2b(1:nua,1:nub,1:nob)
              !f2py intent(in,out) :: r2b(0:nua-1,0:nub-1,0:nob-1)
              integer :: j, a, b
              real(8) :: denom

              do a = 1,nua
                  denom = -H1A_vv(a,a)
                  r1a(a) = r1a(a)/(omega-denom+shift)
              end do

              do j = 1,noa
                do a = 1,nua
                   do b = 1,nua
                      denom = -H1A_vv(a,a) - H1A_vv(b,b) + H1A_oo(j,j)
                      r2a(a,b,j) = r2a(a,b,j)/(omega-denom+shift)
                  end do
                end do
              end do

              do j = 1,nob
                do a = 1,nua
                   do b = 1,nub
                      denom = -H1A_vv(a,a) - H1B_vv(b,b) + H1B_oo(j,j)
                      r2b(a,b,j) = r2b(a,b,j)/(omega-denom+shift)
                  end do
                end do
              end do

      end subroutine update_R_2p1h

      subroutine update_L_2h1p(l1a, l1b, l2a, l2b, l2c, l2d,&
                               X1A, X1B, X2A, X2B, X2C, X2D,&
                               omega,&
                               H1A_oo, H1A_vv, H1B_oo, H1B_vv,&
                               shift,&
                               noa, nua, nob, nub)

              implicit none

              integer, intent(in) :: noa, nua, nob, nub
              real(kind=8), intent(in) :: H1A_oo(1:noa,1:noa), H1A_vv(1:nua,1:nua), &
                                     H1B_oo(1:nob,1:nob), H1B_vv(1:nub,1:nub), shift, omega

              real(kind=8), intent(inout) :: l1a(1:noa)
              !f2py intent(in,out) :: l1a(0:noa-1)
              real(kind=8), intent(inout) :: l1b(1:nob)
              !f2py intent(in,out) :: l1b(0:nob-1)
              real(kind=8), intent(inout) :: l2a(1:nua,1:noa,1:noa)
              !f2py intent(in,out) :: l2a(0:nua-1,0:noa-1,0:noa-1)
              real(kind=8), intent(inout) :: l2b(1:nua,1:noa,1:nob)
              !f2py intent(in,out) :: l2b(0:nua-1,0:noa-1,0:nob-1)
              real(kind=8), intent(inout) :: l2c(1:nub,1:nob,1:noa)
              !f2py intent(in,out) :: l2c(0:nub-1,0:nob-1,0:noa-1)
              real(kind=8), intent(inout) :: l2d(1:nub,1:nob,1:nob)
              !f2py intent(in,out) :: l2d(0:nub-1,0:nob-1,0:nob-1)

              real(kind=8), intent(inout) :: X1A(1:noa)
              !f2py intent(in,out) :: X1A(0:noa-1)
              real(kind=8), intent(inout) :: X1B(1:nob)
              !f2py intent(in,out) :: X1B(0:nob-1)
              real(kind=8), intent(inout) :: X2A(1:nua,1:noa,1:noa)
              !f2py intent(in,out) :: X2A(0:nua-1,0:noa-1,0:noa-1)
              real(kind=8), intent(inout) :: X2B(1:nua,1:noa,1:nob)
              !f2py intent(in,out) :: X2B(0:nua-1,0:noa-1,0:nob-1)
              real(kind=8), intent(inout) :: X2C(1:nub,1:nob,1:noa)
              !f2py intent(in,out) :: X2C(0:nub-1,0:nob-1,0:noa-1)
              real(kind=8), intent(inout) :: X2D(1:nub,1:nob,1:nob)
              !f2py intent(in,out) :: X2D(0:nub-1,0:nob-1,0:nob-1)

              integer :: i, j, b
              real(kind=8) :: denom, val

              do i = 1,noa
                  val = X1A(i)
                  denom = H1A_oo(i,i)
                  l1a(i) = l1a(i) - (val - omega * l1a(i))/(denom - omega + shift)

                  X1A(i) = val - omega * l1a(i)
              end do

              do i = 1,nob
                  val = X1B(i)
                  denom = H1B_oo(i,i)
                  l1b(i) = l1b(i) - (val - omega * l1b(i))/(denom - omega + shift)

                  X1B(i) = val - omega * l1b(i)
              end do

              do i = 1,noa
                do j = 1,noa
                   do b = 1,nua
                      val = X2A(b, j, i)
                      denom = -H1A_vv(b,b) + H1A_oo(i,i) + H1A_oo(j,j)
                      l2a(b,j,i) = l2a(b,j,i) - (val - omega * l2a(b, j, i))/(denom - omega + shift)
                      !l2a(b,i,j) = -1.0 * l2a(b,j,i)

                       X2A(b, j, i) = val - omega * l2a(b, j, i)
                       !X2A(b, i, j) = -1.0 * X2A(b, j, i)
                  end do
                end do
              end do

              do i = 1,nob
                do j = 1,noa
                   do b = 1,nua
                      val = X2B(b, j, i)
                      denom = -H1A_vv(b,b) + H1B_oo(i,i) + H1A_oo(j,j)
                      l2b(b,j,i) = l2b(b,j,i) - (val - omega * l2b(b, j, i))/(denom - omega + shift)

                      X2B(b, j, i) = val - omega * l2b(b, j, i)
                  end do
                end do
              end do

              do i = 1,noa
                do j = 1,nob
                   do b = 1,nub
                      val = X2C(b, j, i)
                      denom = -H1B_vv(b,b) + H1A_oo(i,i) + H1B_oo(j,j)
                      l2c(b,j,i) = l2c(b,j,i) - (val - omega * l2c(b, j, i))/(denom - omega + shift)

                      X2C(b, j, i) = val - omega * l2c(b, j, i)
                  end do
                end do
              end do

              do i = 1,nob
                do j = 1,nob
                   do b = 1,nub
                      val = X2D(b, j, i)
                      denom = -H1B_vv(b,b) + H1B_oo(i,i) + H1B_oo(j,j)
                      l2d(b,j,i) = l2d(b,j,i) - (val - omega * l2d(b, j, i))/(denom - omega + shift)
                      !l2d(b,i,j) = -1.0 * l2d(b,j,i)

                      X2D(b, j, i) = val - omega * l2d(b, j, i)
                      !X2D(b, i, j) = -1.0 * X2D(b, j, i)
                  end do
                end do
              end do

      end subroutine update_L_2h1p

      subroutine update_R_sfccsd(r1b,r2b,r2c,omega,H1A_oo,H1A_vv,H1B_oo,H1B_vv,shift,noa,nua,nob,nub)

              integer, intent(in) :: noa, nua, nob, nub
              real(kind=8), intent(in) :: H1A_oo(1:noa,1:noa), H1A_vv(1:nua,1:nua), &
                                          H1B_oo(1:nob,1:nob), H1B_vv(1:nub,1:nub), shift, &
                                          omega
              real(kind=8), intent(inout) :: r1b(1:nub,1:noa)
              !f2py intent(in,out) :: r1b(0:nub-1,0:noa-1)
              real(kind=8), intent(inout) :: r2b(1:nua,1:nub,1:noa,1:noa)
              !f2py intent(in,out) :: r2b(0:nua-1,0:nub-1,0:noa-1,0:noa-1)
              real(kind=8), intent(inout) :: r2c(1:nub,1:nub,1:nob,1:noa)
              !f2py intent(in,out) :: r2c(0:nub-1,0:nub-1,0:nob-1,0:noa-1)
              integer :: i, j, a, b
              real(kind=8) :: denom

              do i = 1, noa
                 do a = 1, nub
                    denom = H1B_vv(a,a) - H1A_oo(i,i)
                    r1b(a,i) = r1b(a,i)/(omega-denom+shift)
                 end do
              end do
              do i = 1, noa
                 do j = 1, noa
                    do a = 1, nua
                       do b = 1, nub
                           denom = H1B_vv(b,b) + H1A_vv(a,a) - H1A_oo(i,i) - H1A_oo(j,j)
                           r2b(a,b,i,j) = r2b(a,b,i,j)/(omega - denom + shift)
                       end do
                    end do
                 end do
              end do
              do i = 1, nob
                 do j = 1, noa
                    do a = 1, nub
                       do b = 1, nub
                           denom = H1B_vv(b,b) + H1B_vv(a,a) - H1B_oo(i,i) - H1A_oo(j,j)
                           r2c(a,b,i,j) = r2c(a,b,i,j)/(omega - denom + shift)
                       end do
                    end do
                 end do
              end do
      end subroutine update_R_sfccsd

      subroutine update_R_2p(r2b,omega,H1A_vv,H1B_vv,shift,nua,nub)

              integer, intent(in) :: nua, nub
              real(kind=8), intent(in) :: H1A_vv(1:nua,1:nua), H1B_vv(1:nub,1:nub), &
                                          shift, &
                                          omega
              real(kind=8), intent(inout) :: r2b(1:nua,1:nub)
              !f2py intent(in,out) :: r2b(0:nua-1,0:nub-1)

              integer :: a, b, c, k
              real(kind=8) :: denom

              do a = 1, nua; do b = 1, nub;
                  denom = H1A_vv(a,a) + H1B_vv(b,b)
                  r2b(a,b) = r2b(a,b)/(omega - denom)
              end do; end do;

      end subroutine update_R_2p

      subroutine update_R_3p1h(r2b,r3b,r3c,omega,H1A_oo,H1A_vv,H1B_oo,H1B_vv,shift,noa,nua,nob,nub)

              integer, intent(in) :: noa, nua, nob, nub
              real(kind=8), intent(in) :: H1A_oo(1:noa,1:noa), H1A_vv(1:nua,1:nua), &
                                          H1B_oo(1:nob,1:nob), H1B_vv(1:nub,1:nub), shift, &
                                          omega
              real(kind=8), intent(inout) :: r2b(1:nua,1:nub)
              !f2py intent(in,out) :: r2b(0:nua-1,0:nub-1)
              real(kind=8), intent(inout) :: r3b(1:nua,1:nub,1:nua,1:noa)
              !f2py intent(in,out) :: r3b(0:nua-1,0:nub-1,0:nua-1,0:noa-1)
              real(kind=8), intent(inout) :: r3c(1:nua,1:nub,1:nub,1:nob)
              !f2py intent(in,out) :: r3c(0:nua-1,0:nub-1,0:nub-1,0:nob-1)
              integer :: a, b, c, k
              real(kind=8) :: denom

              do a = 1, nua; do b = 1, nub;
                  denom = H1A_vv(a,a) + H1B_vv(b,b)
                  r2b(a,b) = r2b(a,b)/(omega - denom)
              end do; end do;

              do a = 1, nua; do b = 1, nub; do c = 1, nua; do k = 1, noa;
                  if (a == c) cycle
                  denom = H1A_vv(a,a) + H1B_vv(b,b) + H1A_vv(c,c) - H1A_oo(k,k)
                  r3b(a,b,c,k) = r3b(a,b,c,k)/(omega - denom)
              end do; end do; end do; end do;

              do a = 1, nua; do b = 1, nub; do c = 1, nub; do k = 1, nob;
                  if (b == c) cycle
                  denom = H1A_vv(a,a) + H1B_vv(b,b) + H1B_vv(c,c) - H1B_oo(k,k)
                  r3c(a,b,c,k) = r3c(a,b,c,k)/(omega - denom)
              end do; end do; end do; end do;

      end subroutine update_R_3p1h

end module cc_loops2
