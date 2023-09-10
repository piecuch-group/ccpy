module cc3_loops
	
	implicit none
	
	contains
	
	   subroutine compute_t3a(t3a,X3A,fA_oo,fA_vv,noa,nua)

              integer, intent(in) :: noa, nua
              real(kind=8), intent(in) :: fA_oo(1:noa,1:noa), fA_vv(1:nua,1:nua), &
                                          X3A(1:nua,1:nua,1:nua,1:noa,1:noa,1:noa)
              real(kind=8), intent(out) :: t3a(1:nua,1:nua,1:nua,1:noa,1:noa,1:noa)
              integer :: i, j, k, a, b, c, ii, jj, kk, aa, bb, cc
              real(kind=8) :: denom, val

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

                                      val = val/denom

                                      t3a(A,B,C,I,J,K) = val
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
                                  end do
                              end do
                          end do
                      end do
                  end do
              end do

      end subroutine compute_t3a
		
      subroutine compute_t3b(t3b,X3B,fA_oo,fA_vv,fB_oo,fB_vv,noa,nua,nob,nub)

              integer, intent(in) :: noa, nua, nob, nub
              real(kind=8), intent(in) :: fA_oo(1:noa,1:noa), fA_vv(1:nua,1:nua), &
                                          fB_oo(1:nob,1:nob), fB_vv(1:nub,1:nub), &
                                          X3B(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob)
              real(kind=8), intent(out) :: t3b(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob)
              integer :: i, j, k, a, b, c, ii, jj, kk, aa, bb, cc
              real(kind=8) :: denom, val

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
                                      val = val/denom
                                      t3b(a,b,c,i,j,k) = val
                                      t3b(b,a,c,i,j,k) = -t3b(a,b,c,i,j,k)
                                      t3b(a,b,c,j,i,k) = -t3b(a,b,c,i,j,k)
                                      t3b(b,a,c,j,i,k) = t3b(a,b,c,i,j,k)
                                  end do
                              end do
                          end do
                      end do
                  end do
              end do

      end subroutine compute_t3b

      subroutine compute_t3c(t3c,X3C,fA_oo,fA_vv,fB_oo,fB_vv,noa,nua,nob,nub)

              integer, intent(in) :: noa, nua, nob, nub
              real(kind=8), intent(in) :: fA_oo(1:noa,1:noa), fA_vv(1:nua,1:nua), &
		                                    fB_oo(1:nob,1:nob), fB_vv(1:nub,1:nub), &
                                          X3C(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob)
              real(kind=8), intent(out) :: t3c(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob)
              integer :: i, j, k, a, b, c, ii, jj, kk, aa, bb, cc
              real(kind=8) :: denom, val

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
                                      val = val/denom
                                      t3c(a,b,c,i,j,k) = val
                                      t3c(a,c,b,i,j,k) = -t3c(a,b,c,i,j,k)
                                      t3c(a,b,c,i,k,j) = -t3c(a,b,c,i,j,k)
                                      t3c(a,c,b,i,k,j) = t3c(a,b,c,i,j,k)
                                  end do
                              end do
                          end do
                      end do
                  end do
              end do

      end subroutine compute_t3c

      subroutine compute_t3d(t3d,X3D,fB_oo,fB_vv,nob,nub)

              integer, intent(in) :: nob, nub
              real(kind=8), intent(in) :: fB_oo(1:nob,1:nob), fB_vv(1:nub,1:nub), &
                                          X3D(1:nub,1:nub,1:nub,1:nob,1:nob,1:nob)
              real(kind=8), intent(out) :: t3d(1:nub,1:nub,1:nub,1:nob,1:nob,1:nob)
              integer :: i, j, k, a, b, c, ii, jj, kk, aa, bb, cc
              real(kind=8) :: denom, val

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
                                      val = val/denom

                                      t3d(A,B,C,I,J,K) = val
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
                                  end do
                              end do
                          end do
                      end do
                  end do
              end do

      end subroutine compute_t3d
	
end module cc3_loops