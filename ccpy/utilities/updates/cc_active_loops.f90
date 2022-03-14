module cc_active_loops

  implicit none

  contains

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  T3A UPDATES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine update_t3a_111111(t3a,resid,X3A,fA_oo,fA_vv,shift,noa_act,nua_act)

              integer, intent(in) :: noa_act, nua_act
              real(8), intent(in) :: fA_oo(1:noa_act,1:noa_act), fA_vv(1:nua_act,1:nua_act), &
                                     X3A(1:nua_act,1:nua_act,1:nua_act,1:noa_act,1:noa_act,1:noa_act), shift               
              real(8), intent(inout) :: t3a(1:nua_act,1:nua_act,1:nua_act,1:noa_act,1:noa_act,1:noa_act)
              !f2py intent(in,out) :: t3a(0:nua_act-1,0:nua_act-1,0:nua_act-1,0:noa_act-1,0:noa_act-1,0:noa_act-1)
              real(8), intent(out) :: resid(1:nua_act,1:nua_act,1:nua_act,1:noa_act,1:noa_act,1:noa_act)
              integer :: i, j, k, a, b, c, ii, jj, kk, aa, bb, cc
              real(8) :: denom, val

              do ii = 1,noa_act
                  do jj = ii+1,noa_act
                      do kk = jj+1,noa_act
                          do aa = 1,nua_act
                              do bb = aa+1,nua_act
                                  do cc = bb+1,nua_act

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

      end subroutine update_t3a_111111


      subroutine update_t3a_110111(t3a,resid,X3A,fA_oo_act,fA_vv_act,fA_oo_inact,fA_vv_inact,shift,noa_act,nua_act,noa_inact,nua_inact)

              integer, intent(in) :: noa_act, nua_act, noa_inact, nua_inact
              real(8), intent(in) :: fA_oo_act(1:noa_act,1:noa_act), fA_vv_act(1:nua_act,1:nua_act),&
                                     fA_oo_inact(1:noa_inact,1:noa_inact), fA_vv_inact(1:nua_inact,1:nua_inact),&
                                     X3A(1:nua_act,1:nua_act,1:nua_inact,1:noa_act,1:noa_act,1:noa_act), shift               
              real(8), intent(inout) :: t3a(1:nua_act,1:nua_act,1:nua_inact,1:noa_act,1:noa_act,1:noa_act)
              !f2py intent(in,out) :: t3a(0:nua_act-1,0:nua_act-1,0:nua_inact-1,0:noa_act-1,0:noa_act-1,0:noa_act-1)
              real(8), intent(out) :: resid(1:nua_act,1:nua_act,1:nua_inact,1:noa_act,1:noa_act,1:noa_act)
              integer :: i, j, k, a, b, c
              real(8) :: denom, val

              do I = 1,noa_act
                  do J = I+1,noa_act
                      do K = J+1,noa_act
                          do A = 1,nua_act
                              do B = A+1,nua_act
                                  do c = 1,nua_inact
                                      
                                      denom = fA_oo_act(I,I)+fA_oo_act(J,J)+fA_oo_act(K,K)&
                                              -fA_vv_act(A,A)-fA_vv_act(B,B)-fA_vv_inact(c,c)
                                      val = X3A(A, B, c, I, J, K)&
                                           -X3A(B, A, c, I, J, K)&
                                           -X3A(A, B, c, J, I, K)&
                                           +X3A(B, A, c, J, I, K)&
                                           -X3A(A, B, c, I, K, J)&
                                           +X3A(B, A, c, I, K, J)&
                                           -X3A(A, B, c, K, J, I)&
                                           +X3A(A, B, c, K, J, I)&
                                           +X3A(A, B, c, J, K, I)&
                                           -X3A(B, A, c, J, K, I)&
                                           +X3A(A, B, c, K, I, J)&
                                           -X3A(B, A, c, K, I, J)
                                     val = val/(denom - shift)

                                     t3a(A, B, c, I, J, K) = t3a(A, B, c, I, J, K) + val
                                     t3a(B, A, c, I, J, K) = -1.0*t3a(A, B, c, I, J, K)
                                     t3a(A, B, c, J, I, K) = -1.0*t3a(A, B, c, I, J, K)
                                     t3a(B, A, c, J, I, K) = t3a(A, B, c, I, J, K)
                                     t3a(A, B, c, I, K, J) = -1.0*t3a(A, B, c, I, J, K)
                                     t3a(B, A, c, I, K, J) = t3a(A, B, c, I, J, K)
                                     t3a(A, B, c, K, J, I) = -1.0*t3a(A, B, c, I, J, K)
                                     t3a(A, B, c, K, J, I) = t3a(A, B, c, I, J, K)
                                     t3a(A, B, c, J, K, I) = t3a(A, B, c, I, J, K)
                                     t3a(B, A, c, J, K, I) = -1.0*t3a(A, B, c, I, J, K)
                                     t3a(A, B, c, K, I, J) = t3a(A, B, c, I, J, K)
                                     t3a(B, A, c, K, I, J) = -1.0*t3a(A, B, c, I, J, k)

                                     resid(A, B, c, I, J, K) = val
                                     resid(B, A, c, I, J, K) = -1.0*val
                                     resid(A, B, c, J, I, K) = -1.0*val
                                     resid(B, A, c, J, I, K) = val
                                     resid(A, B, c, I, K, J) = -1.0*val
                                     resid(B, A, c, I, K, J) = val
                                     resid(A, B, c, K, J, I) = -1.0*val
                                     resid(A, B, c, K, J, I) = val
                                     resid(A, B, c, J, K, I) = val
                                     resid(B, A, c, J, K, I) = -1.0*val
                                     resid(A, B, c, K, I, J) = val
                                     resid(B, A, c, K, I, J) = -1.0*val
                                end do
                             end do
                          end do
                       end do
                    end do
                 end do
          end subroutine update_t3a_110111



      subroutine update_t3a_111011(t3a,resid,X3A,fA_oo_act,fA_vv_act,fA_oo_inact,fA_vv_inact,shift,noa_act,nua_act,noa_inact,nua_inact)

              integer, intent(in) :: noa_act, nua_act, noa_inact, nua_inact
              real(8), intent(in) :: fA_oo_act(1:noa_act,1:noa_act), fA_vv_act(1:nua_act,1:nua_act),&
                                     fA_oo_inact(1:noa_inact,1:noa_inact), fA_vv_inact(1:nua_inact,1:nua_inact),&
                                     X3A(1:nua_act,1:nua_act,1:nua_act,1:noa_inact,1:noa_act,1:noa_act), shift               
              real(8), intent(inout) :: t3a(1:nua_act,1:nua_act,1:nua_act,1:noa_inact,1:noa_act,1:noa_act)
              !f2py intent(in,out) :: t3a(0:nua_act-1,0:nua_act-1,0:nua_act-1,0:noa_inact-1,0:noa_act-1,0:noa_act-1)
              real(8), intent(out) :: resid(1:nua_act,1:nua_act,1:nua_act,1:noa_inact,1:noa_act,1:noa_act)
              integer :: i, j, k, a, b, c 
              real(8) :: denom, val

              do i = 1,noa_inact
                  do J = 1,noa_act
                      do K = J+1,noa_act
                          do A = 1,nua_act
                              do B = A+1,nua_act
                                  do C = B+1,nua_act
                                      
                                      denom = fA_oo_inact(i,i)+fA_oo_act(J,J)+fA_oo_act(K,K)&
                                              -fA_vv_act(A,A)-fA_vv_act(B,B)-fA_vv_act(C,C)
                                      val = X3A(A, B, C, i, J, K)&
                                           -X3A(A, B, C, i, K, J)&
                                           -X3A(B, A, C, i, J, K)&
                                           +X3A(B, A, C, i, K, J)&
                                           -X3A(A, C, B, i, J, K)&
                                           +X3A(A, C, B, i, K, J)&
                                           -X3A(C, B, A, i, J, K)&
                                           +X3A(C, B, A, i, K, J)&
                                           +X3A(B, C, A, i, J, K)&
                                           -X3A(B, C, A, i, K, J)&
                                           +X3A(C, A, B, i, J, K)&
                                           -X3A(C, A, B, i, K, J)
                                     val = val/(denom - shift)

                                     t3a(A, B, C, i, J, K) = t3a(A, B, C, i, J, K) + val
                                     t3a(A, B, C, i, K, J) = -1.0*t3a(A, B, C, i, J, K)
                                     t3a(B, A, C, i, J, K) = -1.0*t3a(A, B, C, i, J, K)
                                     t3a(B, A, C, i, K, J) = t3a(A, B, C, i, J, K)
                                     t3a(A, C, B, i, J, K) = -1.0*t3a(A, B, C, i, J, K)
                                     t3a(A, C, B, i, K, J) = t3a(A, B, C, i, J, K)
                                     t3a(C, B, A, i, J, K) = -1.0*t3a(A, B, C, i, J, K)
                                     t3a(C, B, A, i, K, J) = t3a(A, B, C, i, J, K)
                                     t3a(B, C, A, i, J, K) = t3a(A, B, C, i, J, K)
                                     t3a(B, C, A, i, K, J) = -1.0*t3a(A, B, C, i, J, K)
                                     t3a(C, A, B, i, J, K) = t3a(A, B, C, i, J, K)
                                     t3a(C, A, B, i, K, J) = -1.0*t3a(A, B, C, i, J, K)

                                     resid(A, B, C, i, J, K) = val
                                     resid(A, B, C, i, K, J) = -1.0*val
                                     resid(B, A, C, i, J, K) = -1.0*val
                                     resid(B, A, C, i, K, J) = val
                                     resid(A, C, B, i, J, K) = -1.0*val
                                     resid(A, C, B, i, K, J) = val
                                     resid(C, B, A, i, J, K) = -1.0*val
                                     resid(C, B, A, i, K, J) = val
                                     resid(B, C, A, i, J, K) = val
                                     resid(B, C, A, i, K, J) = -1.0*val
                                     resid(C, A, B, i, J, K) = val
                                     resid(C, A, B, i, K, J) = -1.0*val

                               end do
                           end do
                        end do
                     end do
                  end do
               end do
      end subroutine update_t3a_111011

    subroutine update_t3a_100111(t3a, resid, X3A, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact)
      real(8), intent(in)  :: X3A(1:nua_act, 1:nua_inact, 1:nua_inact, 1:noa_act, 1:noa_act, 1:noa_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3a(1:nua_act, 1:nua_inact, 1:nua_inact, 1:noa_act, 1:noa_act, 1:noa_act)
      !f2py intent(in, out)  :: t3a(0:nua_act-1, 0:nua_inact-1, 0:nua_inact-1, 0:noa_act-1, 0:noa_act-1, 0:noa_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_inact, 1:nua_inact, 1:noa_act, 1:noa_act, 1:noa_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = i+1 , noa_act
            do k = j+1 , noa_act
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = b+1 , nua_inact

                        denom = fA_oo_act(i,i) + fA_oo_act(j,j) + fA_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_inact(b,b) - fA_vv_inact(c,c)

                        val = X3A(a, b, c, i, j, k)/(denom - shift)

                        t3a(a, b, c, i, j, k) = t3a(a, b, c, i, j, k) + val
                        t3a(a, b, c, i, k, j) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, b, c, j, i, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, b, c, j, k, i) = t3a(a, b, c, i, j, k)
                        t3a(a, b, c, k, i, j) = t3a(a, b, c, i, j, k)
                        t3a(a, b, c, k, j, i) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, c, b, i, j, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, c, b, i, k, j) = t3a(a, b, c, i, j, k)
                        t3a(a, c, b, j, i, k) = t3a(a, b, c, i, j, k)
                        t3a(a, c, b, j, k, i) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, c, b, k, i, j) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, c, b, k, j, i) = t3a(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(a, b, c, j, k, i) = val
                        resid(a, b, c, k, i, j) = val
                        resid(a, b, c, k, j, i) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, i, k, j) = val
                        resid(a, c, b, j, i, k) = val
                        resid(a, c, b, j, k, i) = -1.0 * val
                        resid(a, c, b, k, i, j) = -1.0 * val
                        resid(a, c, b, k, j, i) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

    end subroutine update_t3a_100111

    subroutine update_t3a_111001(t3a, resid, X3A, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact)
      real(8), intent(in)  :: X3A(1:nua_act, 1:nua_act, 1:nua_act, 1:noa_inact, 1:noa_inact, 1:noa_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3a(1:nua_act, 1:nua_act, 1:nua_act, 1:noa_inact, 1:noa_inact, 1:noa_act)
      !f2py intent(in, out)  :: t3a(0:nua_act-1, 0:nua_act-1, 0:nua_act-1, 0:noa_inact-1, 0:noa_inact-1, 0:noa_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nua_act, 1:noa_inact, 1:noa_inact, 1:noa_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = i+1 , noa_inact
            do k = 1 , noa_act
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = b+1 , nua_act

                        denom = fA_oo_inact(i,i) + fA_oo_inact(j,j) + fA_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fA_vv_act(c,c)

                        val = X3A(a, b, c, i, j, k)/(denom - shift)

                        t3a(a, b, c, i, j, k) = t3a(a, b, c, i, j, k) + val
                        t3a(a, b, c, j, i, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, c, b, i, j, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, c, b, j, i, k) = t3a(a, b, c, i, j, k)
                        t3a(b, a, c, i, j, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(b, a, c, j, i, k) = t3a(a, b, c, i, j, k)
                        t3a(b, c, a, i, j, k) = t3a(a, b, c, i, j, k)
                        t3a(b, c, a, j, i, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(c, a, b, i, j, k) = t3a(a, b, c, i, j, k)
                        t3a(c, a, b, j, i, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(c, b, a, i, j, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(c, b, a, j, i, k) = t3a(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, j, i, k) = val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, j, i, k) = val
                        resid(b, c, a, i, j, k) = val
                        resid(b, c, a, j, i, k) = -1.0 * val
                        resid(c, a, b, i, j, k) = val
                        resid(c, a, b, j, i, k) = -1.0 * val
                        resid(c, b, a, i, j, k) = -1.0 * val
                        resid(c, b, a, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

    end subroutine update_t3a_111001

    subroutine update_t3a_110011(t3a, resid, X3A, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact)
      real(8), intent(in)  :: X3A(1:nua_act, 1:nua_act, 1:nua_inact, 1:noa_inact, 1:noa_act, 1:noa_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3a(1:nua_act, 1:nua_act, 1:nua_inact, 1:noa_inact, 1:noa_act, 1:noa_act)
      !f2py intent(in, out)  :: t3a(0:nua_act-1, 0:nua_act-1, 0:nua_inact-1, 0:noa_inact-1, 0:noa_act-1, 0:noa_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nua_inact, 1:noa_inact, 1:noa_act, 1:noa_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = j+1 , noa_act
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = 1 , nua_inact

                        denom = fA_oo_inact(i,i) + fA_oo_act(j,j) + fA_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fA_vv_inact(c,c)

                        val = X3A(a, b, c, i, j, k)/(denom - shift)

                        t3a(a, b, c, i, j, k) = t3a(a, b, c, i, j, k) + val
                        t3a(a, b, c, i, k, j) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(b, a, c, i, j, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(b, a, c, i, k, j) = t3a(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, i, k, j) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

    end subroutine update_t3a_110011


    subroutine update_t3a_100011(t3a, resid, X3A, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact)
      real(8), intent(in)  :: X3A(1:nua_act, 1:nua_inact, 1:nua_inact, 1:noa_inact, 1:noa_act, 1:noa_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3a(1:nua_act, 1:nua_inact, 1:nua_inact, 1:noa_inact, 1:noa_act, 1:noa_act)
      !f2py intent(in, out)  :: t3a(0:nua_act-1, 0:nua_inact-1, 0:nua_inact-1, 0:noa_inact-1, 0:noa_act-1, 0:noa_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_inact, 1:nua_inact, 1:noa_inact, 1:noa_act, 1:noa_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = j+1 , noa_act
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = b+1 , nua_inact

                        denom = fA_oo_inact(i,i) + fA_oo_act(j,j) + fA_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_inact(b,b) - fA_vv_inact(c,c)

                        val = X3A(a, b, c, i, j, k)/(denom - shift)

                        t3a(a, b, c, i, j, k) = t3a(a, b, c, i, j, k) + val
                        t3a(a, b, c, i, k, j) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, c, b, i, j, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, c, b, i, k, j) = t3a(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, i, k, j) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

    end subroutine update_t3a_100011

    subroutine update_t3a_110001(t3a, resid, X3A, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact)
      real(8), intent(in)  :: X3A(1:nua_act, 1:nua_act, 1:nua_inact, 1:noa_inact, 1:noa_inact, 1:noa_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3a(1:nua_act, 1:nua_act, 1:nua_inact, 1:noa_inact, 1:noa_inact, 1:noa_act)
      !f2py intent(in, out)  :: t3a(0:nua_act-1, 0:nua_act-1, 0:nua_inact-1, 0:noa_inact-1, 0:noa_inact-1, 0:noa_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nua_inact, 1:noa_inact, 1:noa_inact, 1:noa_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = i+1 , noa_inact
            do k = 1 , noa_act
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = 1 , nua_inact

                        denom = fA_oo_inact(i,i) + fA_oo_inact(j,j) + fA_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fA_vv_inact(c,c)

                        val = X3A(a, b, c, i, j, k)/(denom - shift)

                        t3a(a, b, c, i, j, k) = t3a(a, b, c, i, j, k) + val
                        t3a(a, b, c, j, i, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(b, a, c, i, j, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(b, a, c, j, i, k) = t3a(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

      end subroutine update_t3a_110001


      subroutine update_t3a_100001(t3a, resid, X3A, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact)
      real(8), intent(in)  :: X3A(1:nua_act, 1:nua_inact, 1:nua_inact, 1:noa_inact, 1:noa_inact, 1:noa_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3a(1:nua_act, 1:nua_inact, 1:nua_inact, 1:noa_inact, 1:noa_inact, 1:noa_act)
      !f2py intent(in, out)  :: t3a(0:nua_act-1, 0:nua_inact-1, 0:nua_inact-1, 0:noa_inact-1, 0:noa_inact-1, 0:noa_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_inact, 1:nua_inact, 1:noa_inact, 1:noa_inact, 1:noa_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = i+1 , noa_inact
            do k = 1 , noa_act
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = b+1 , nua_inact

                        denom = fA_oo_inact(i,i) + fA_oo_inact(j,j) + fA_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_inact(b,b) - fA_vv_inact(c,c)

                        val = X3A(a, b, c, i, j, k)/(denom - shift)

                        t3a(a, b, c, i, j, k) = t3a(a, b, c, i, j, k) + val
                        t3a(a, b, c, j, i, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, c, b, i, j, k) = -1.0 * t3a(a, b, c, i, j, k)
                        t3a(a, c, b, j, i, k) = t3a(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

    end subroutine update_t3a_100001

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  T3B UPDATES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine update_t3b_111111(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_act-1, 0:nub_act-1, 0:noa_act-1, 0:noa_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = i+1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = 1 , nub_act

                        denom = fA_oo_act(i,i) + fA_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, i, j, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, j, i, k) = t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_111111

subroutine update_t3b_110111(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_act, 1:noa_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_act, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_act-1, 0:nub_inact-1, 0:noa_act-1, 0:noa_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_act, 1:noa_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = i+1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = 1 , nub_inact

                        denom = fA_oo_act(i,i) + fA_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fB_vv_inact(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, i, j, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, j, i, k) = t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_110111

subroutine update_t3b_101111(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_act-1, 0:noa_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = i+1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_act

                        denom = fA_oo_act(i,i) + fA_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_inact(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_101111

subroutine update_t3b_111011(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_act-1, 0:nub_act-1, 0:noa_inact-1, 0:noa_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = 1 , nub_act

                        denom = fA_oo_inact(i,i) + fA_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(b, a, c, i, j, k) = -1.0 * t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(b, a, c, i, j, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_111011

subroutine update_t3b_111110(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_inact)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_act-1, 0:nub_act-1, 0:noa_act-1, 0:noa_act-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = i+1 , noa_act
            do k = 1 , nob_inact
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = 1 , nub_act

                        denom = fA_oo_act(i,i) + fA_oo_act(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, i, j, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, j, i, k) = t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_111110

subroutine update_t3b_110011(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_inact, 1:noa_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_inact, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_act-1, 0:nub_inact-1, 0:noa_inact-1, 0:noa_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_inact, 1:noa_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = 1 , nub_inact

                        denom = fA_oo_inact(i,i) + fA_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fB_vv_inact(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(b, a, c, i, j, k) = -1.0 * t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(b, a, c, i, j, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_110011

subroutine update_t3b_110110(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_act, 1:noa_act, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_act, 1:noa_act, 1:nob_inact)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_act-1, 0:nub_inact-1, 0:noa_act-1, 0:noa_act-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_act, 1:noa_act, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = i+1 , noa_act
            do k = 1 , nob_inact
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = 1 , nub_inact

                        denom = fA_oo_act(i,i) + fA_oo_act(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fB_vv_inact(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, i, j, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, j, i, k) = t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_110110

subroutine update_t3b_101011(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_inact-1, 0:noa_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_act

                        denom = fA_oo_inact(i,i) + fA_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_inact(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val

                        resid(a, b, c, i, j, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_101011

subroutine update_t3b_101110(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_inact)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_act-1, 0:noa_act-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = i+1 , noa_act
            do k = 1 , nob_inact
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_act

                        denom = fA_oo_act(i,i) + fA_oo_act(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_act(a,a) - fA_vv_inact(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_101110
subroutine update_t3b_100111(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_act, 1:noa_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_act, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_inact-1, 0:nub_inact-1, 0:noa_act-1, 0:noa_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_act, 1:noa_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = i+1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_inact

                        denom = fA_oo_act(i,i) + fA_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_inact(b,b) - fB_vv_inact(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_100111

subroutine update_t3b_001111(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_inact-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_act-1, 0:noa_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = i+1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_inact
                  do b = a+1 , nua_inact
                     do c = 1 , nub_act

                        denom = fA_oo_act(i,i) + fA_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_inact(a,a) - fA_vv_inact(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, i, j, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, j, i, k) = t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_001111

subroutine update_t3b_111001(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_inact, 1:noa_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_inact, 1:noa_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_act-1, 0:nub_act-1, 0:noa_inact-1, 0:noa_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_inact, 1:noa_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = i+1 , noa_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = 1 , nub_act

                        denom = fA_oo_inact(i,i) + fA_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, i, j, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, j, i, k) = t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_111001

subroutine update_t3b_111100(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_act, 1:noa_inact, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_act, 1:noa_inact, 1:nob_inact)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_act-1, 0:nub_act-1, 0:noa_act-1, 0:noa_inact-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_act, 1:noa_inact, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , noa_inact
            do k = 1 , nob_inact
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = 1 , nub_act

                        denom = fA_oo_act(i,i) + fA_oo_inact(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(b, a, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)

                        resid(a, b, c, j, i, k) = val
                        resid(b, a, c, j, i, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_111100

subroutine update_t3b_001011(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_inact-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_inact-1, 0:noa_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_inact
                  do b = a+1 , nua_inact
                     do c = 1 , nub_act

                        denom = fA_oo_inact(i,i) + fA_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_inact(a,a) - fA_vv_inact(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(b, a, c, i, j, k) = -1.0 * t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(b, a, c, i, j, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_001011

subroutine update_t3b_001110(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_inact)
      !f2py intent(in, out)  :: t3b(0:nua_inact-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_act-1, 0:noa_act-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = i+1 , noa_act
            do k = 1 , nob_inact
               do a = 1 , nua_inact
                  do b = a+1 , nua_inact
                     do c = 1 , nub_act

                        denom = fA_oo_act(i,i) + fA_oo_act(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_inact(a,a) - fA_vv_inact(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, i, j, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, j, i, k) = t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_001110

subroutine update_t3b_100110(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_act, 1:noa_act, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_act, 1:noa_act, 1:nob_inact)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_inact-1, 0:nub_inact-1, 0:noa_act-1, 0:noa_act-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_act, 1:noa_act, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = i+1 , noa_act
            do k = 1 , nob_inact
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_inact

                        denom = fA_oo_act(i,i) + fA_oo_act(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_act(a,a) - fA_vv_inact(b,b) - fB_vv_inact(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_100110

subroutine update_t3b_100011(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_inact, 1:noa_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_inact, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_inact-1, 0:nub_inact-1, 0:noa_inact-1, 0:noa_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_inact, 1:noa_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_inact

                        denom = fA_oo_inact(i,i) + fA_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_inact(b,b) - fB_vv_inact(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val

                        resid(a, b, c, i, j, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_100011

subroutine update_t3b_110001(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_inact, 1:noa_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_inact, 1:noa_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_act-1, 0:nub_inact-1, 0:noa_inact-1, 0:noa_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_inact, 1:noa_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = i+1 , noa_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = 1 , nub_inact

                        denom = fA_oo_inact(i,i) + fA_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fB_vv_inact(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, i, j, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, j, i, k) = t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_110001

subroutine update_t3b_101001(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_inact-1, 0:noa_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = i+1 , noa_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_act

                        denom = fA_oo_inact(i,i) + fA_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_inact(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_101001

subroutine update_t3b_101100(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_inact, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_inact, 1:nob_inact)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_act-1, 0:noa_inact-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_inact, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , noa_inact
            do k = 1 , nob_inact
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_act

                        denom = fA_oo_act(i,i) + fA_oo_inact(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_act(a,a) - fA_vv_inact(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val

                        resid(a, b, c, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_101100

subroutine update_t3b_110100(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_act, 1:noa_inact, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_act, 1:noa_inact, 1:nob_inact)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_act-1, 0:nub_inact-1, 0:noa_act-1, 0:noa_inact-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_act, 1:noa_inact, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , noa_inact
            do k = 1 , nob_inact
               do a = 1 , nua_act
                  do b = a+1 , nua_act
                     do c = 1 , nub_inact

                        denom = fA_oo_act(i,i) + fA_oo_inact(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_act(a,a) - fA_vv_act(b,b) - fB_vv_inact(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(b, a, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)

                        resid(a, b, c, j, i, k) = val
                        resid(b, a, c, j, i, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_110100

subroutine update_t3b_001001(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_inact-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_inact-1, 0:noa_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = i+1 , noa_inact
            do k = 1 , nob_act
               do a = 1 , nua_inact
                  do b = a+1 , nua_inact
                     do c = 1 , nub_act

                        denom = fA_oo_inact(i,i) + fA_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_inact(a,a) - fA_vv_inact(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, i, j, k) = -1.0 * t3b(a, b, c, i, j, k)
                        t3b(b, a, c, j, i, k) = t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val
                        resid(b, a, c, i, j, k) = -1.0 * val
                        resid(b, a, c, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_001001

subroutine update_t3b_001100(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_inact, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_inact, 1:nob_inact)
      !f2py intent(in, out)  :: t3b(0:nua_inact-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_act-1, 0:noa_inact-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_inact, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , noa_inact
            do k = 1 , nob_inact
               do a = 1 , nua_inact
                  do b = a+1 , nua_inact
                     do c = 1 , nub_act

                        denom = fA_oo_act(i,i) + fA_oo_inact(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_inact(a,a) - fA_vv_inact(b,b) - fB_vv_act(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(b, a, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)

                        resid(a, b, c, j, i, k) = val
                        resid(b, a, c, j, i, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_001100

subroutine update_t3b_100100(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_act, 1:noa_inact, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_act, 1:noa_inact, 1:nob_inact)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_inact-1, 0:nub_inact-1, 0:noa_act-1, 0:noa_inact-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_act, 1:noa_inact, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , noa_inact
            do k = 1 , nob_inact
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_inact

                        denom = fA_oo_act(i,i) + fA_oo_inact(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_act(a,a) - fA_vv_inact(b,b) - fB_vv_inact(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val

                        resid(a, b, c, j, i, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_100100

subroutine update_t3b_100001(t3b, resid, X3B, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3B(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_inact, 1:noa_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3b(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_inact, 1:noa_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3b(0:nua_act-1, 0:nua_inact-1, 0:nub_inact-1, 0:noa_inact-1, 0:noa_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_inact, 1:noa_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = i+1 , noa_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_inact

                        denom = fA_oo_inact(i,i) + fA_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fA_vv_inact(b,b) - fB_vv_inact(c,c)

                        val = X3B(a, b, c, i, j, k)/(denom - shift)

                        t3b(a, b, c, i, j, k) = t3b(a, b, c, i, j, k) + val
                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, j, i, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3b_100001

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  T3C UPDATES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine update_t3c_111111(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_act-1, 0:nub_act-1, 0:noa_act-1, 0:nob_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_act
            do k = j+1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = b+1 , nub_act

                        denom = fA_oo_act(i,i) + fB_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fB_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, k, j) = t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, i, k, j) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_111111

subroutine update_t3c_011111(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_inact-1, 0:nub_act-1, 0:nub_act-1, 0:noa_act-1, 0:nob_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_act
            do k = j+1 , nob_act
               do a = 1 , nua_inact
                  do b = 1 , nub_act
                     do c = b+1 , nub_act

                        denom = fA_oo_act(i,i) + fB_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_inact(a,a) - fB_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, k, j) = t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, i, k, j) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_011111

subroutine update_t3c_110111(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_act-1, 0:nub_inact-1, 0:noa_act-1, 0:nob_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_act
            do k = j+1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = 1 , nub_inact

                        denom = fA_oo_act(i,i) + fB_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fB_vv_act(b,b) - fB_vv_inact(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_110111

subroutine update_t3c_111011(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_inact, 1:nob_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_inact, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_act-1, 0:nub_act-1, 0:noa_inact-1, 0:nob_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_inact, 1:nob_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , nob_act
            do k = j+1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = b+1 , nub_act

                        denom = fA_oo_inact(i,i) + fB_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fB_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, k, j) = t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, i, k, j) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_111011

subroutine update_t3c_111101(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_act-1, 0:nub_act-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = b+1 , nub_act

                        denom = fA_oo_act(i,i) + fB_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fB_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, c, b, i, j, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_111101

subroutine update_t3c_001111(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_inact, 1:nub_inact, 1:nub_act, 1:noa_act, 1:nob_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_inact, 1:nub_inact, 1:nub_act, 1:noa_act, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_inact-1, 0:nub_inact-1, 0:nub_act-1, 0:noa_act-1, 0:nob_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nub_inact, 1:nub_act, 1:noa_act, 1:nob_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_act
            do k = j+1 , nob_act
               do a = 1 , nua_inact
                  do b = 1 , nub_inact
                     do c = 1 , nub_act

                        denom = fA_oo_act(i,i) + fB_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_inact(a,a) - fB_vv_inact(b,b) - fB_vv_act(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, c, b, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)

                        resid(a, c, b, i, j, k) = val
                        resid(a, c, b, i, k, j) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_001111

subroutine update_t3c_001011(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_inact, 1:nub_inact, 1:nub_act, 1:noa_inact, 1:nob_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_inact, 1:nub_inact, 1:nub_act, 1:noa_inact, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_inact-1, 0:nub_inact-1, 0:nub_act-1, 0:noa_inact-1, 0:nob_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nub_inact, 1:nub_act, 1:noa_inact, 1:nob_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , nob_act
            do k = j+1 , nob_act
               do a = 1 , nua_inact
                  do b = 1 , nub_inact
                     do c = 1 , nub_act

                        denom = fA_oo_inact(i,i) + fB_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_inact(a,a) - fB_vv_inact(b,b) - fB_vv_act(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, c, b, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)

                        resid(a, c, b, i, j, k) = val
                        resid(a, c, b, i, k, j) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_001011

subroutine update_t3c_001101(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_inact, 1:nub_inact, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_inact, 1:nub_inact, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_inact-1, 0:nub_inact-1, 0:nub_act-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nub_inact, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_inact
                  do b = 1 , nub_inact
                     do c = 1 , nub_act

                        denom = fA_oo_act(i,i) + fB_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_inact(a,a) - fB_vv_inact(b,b) - fB_vv_act(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val

                        resid(a, c, b, i, j, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_001101

subroutine update_t3c_100111(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_act, 1:nob_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_act, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_inact-1, 0:nub_inact-1, 0:noa_act-1, 0:nob_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_act, 1:nob_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_act
            do k = j+1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_inact
                     do c = b+1 , nub_inact

                        denom = fA_oo_act(i,i) + fB_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fB_vv_inact(b,b) - fB_vv_inact(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, k, j) = t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, i, k, j) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_100111

subroutine update_t3c_100011(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_inact, 1:nob_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_inact, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_inact-1, 0:nub_inact-1, 0:noa_inact-1, 0:nob_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_inact, 1:nob_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , nob_act
            do k = j+1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_inact
                     do c = b+1 , nub_inact

                        denom = fA_oo_inact(i,i) + fB_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fB_vv_inact(b,b) - fB_vv_inact(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, k, j) = t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, i, k, j) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_100011

subroutine update_t3c_100101(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_inact-1, 0:nub_inact-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_inact
                     do c = b+1 , nub_inact

                        denom = fA_oo_act(i,i) + fB_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fB_vv_inact(b,b) - fB_vv_inact(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, c, b, i, j, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_100101

subroutine update_t3c_111001(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_inact, 1:nob_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_inact, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_act-1, 0:nub_act-1, 0:noa_inact-1, 0:nob_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_inact, 1:nob_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = b+1 , nub_act

                        denom = fA_oo_inact(i,i) + fB_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fB_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, c, b, i, j, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_111001

subroutine update_t3c_011001(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_inact, 1:nob_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_inact, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_inact-1, 0:nub_act-1, 0:nub_act-1, 0:noa_inact-1, 0:nob_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_inact, 1:nob_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_inact
                  do b = 1 , nub_act
                     do c = b+1 , nub_act

                        denom = fA_oo_inact(i,i) + fB_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_inact(a,a) - fB_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, c, b, i, j, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_011001

subroutine update_t3c_110001(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_inact, 1:nob_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_inact, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_act-1, 0:nub_inact-1, 0:noa_inact-1, 0:nob_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_inact, 1:nob_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = 1 , nub_inact

                        denom = fA_oo_inact(i,i) + fB_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fB_vv_act(b,b) - fB_vv_inact(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val

                        resid(a, b, c, i, j, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_110001

subroutine update_t3c_111100(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_inact)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_act-1, 0:nub_act-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = j+1 , nob_inact
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = b+1 , nub_act

                        denom = fA_oo_act(i,i) + fB_oo_inact(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_act(a,a) - fB_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, k, j) = t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, i, k, j) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_111100

subroutine update_t3c_011100(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_inact)
      !f2py intent(in, out)  :: t3c(0:nua_inact-1, 0:nub_act-1, 0:nub_act-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = j+1 , nob_inact
               do a = 1 , nua_inact
                  do b = 1 , nub_act
                     do c = b+1 , nub_act

                        denom = fA_oo_act(i,i) + fB_oo_inact(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_inact(a,a) - fB_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, k, j) = t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, i, k, j) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_011100

subroutine update_t3c_110100(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_inact)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_act-1, 0:nub_inact-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = j+1 , nob_inact
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = 1 , nub_inact

                        denom = fA_oo_act(i,i) + fB_oo_inact(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_act(a,a) - fB_vv_act(b,b) - fB_vv_inact(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_110100

subroutine update_t3c_011011(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_inact, 1:nob_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_inact, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_inact-1, 0:nub_act-1, 0:nub_act-1, 0:noa_inact-1, 0:nob_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_inact, 1:nob_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , nob_act
            do k = j+1 , nob_act
               do a = 1 , nua_inact
                  do b = 1 , nub_act
                     do c = b+1 , nub_act

                        denom = fA_oo_inact(i,i) + fB_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_inact(a,a) - fB_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, k, j) = t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, i, k, j) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_011011

subroutine update_t3c_011101(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_inact-1, 0:nub_act-1, 0:nub_act-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_inact
                  do b = 1 , nub_act
                     do c = b+1 , nub_act

                        denom = fA_oo_act(i,i) + fB_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_inact(a,a) - fB_vv_act(b,b) - fB_vv_act(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, c, b, i, j, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_011101

subroutine update_t3c_110011(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_inact, 1:nob_act, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_inact, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_act-1, 0:nub_inact-1, 0:noa_inact-1, 0:nob_act-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_inact, 1:nob_act, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , nob_act
            do k = j+1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = 1 , nub_inact

                        denom = fA_oo_inact(i,i) + fB_oo_act(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fB_vv_act(b,b) - fB_vv_inact(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_110011

subroutine update_t3c_110101(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_act-1, 0:nub_inact-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = 1 , nub_inact

                        denom = fA_oo_act(i,i) + fB_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fB_vv_act(b,b) - fB_vv_inact(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val

                        resid(a, b, c, i, j, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_110101

subroutine update_t3c_001001(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_inact, 1:nub_inact, 1:nub_act, 1:noa_inact, 1:nob_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_inact, 1:nub_inact, 1:nub_act, 1:noa_inact, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_inact-1, 0:nub_inact-1, 0:nub_act-1, 0:noa_inact-1, 0:nob_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nub_inact, 1:nub_act, 1:noa_inact, 1:nob_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_inact
                  do b = 1 , nub_inact
                     do c = 1 , nub_act

                        denom = fA_oo_inact(i,i) + fB_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_inact(a,a) - fB_vv_inact(b,b) - fB_vv_act(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val

                        resid(a, c, b, i, j, k) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_001001

subroutine update_t3c_001100(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_inact, 1:nub_inact, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_inact, 1:nub_inact, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_inact)
      !f2py intent(in, out)  :: t3c(0:nua_inact-1, 0:nub_inact-1, 0:nub_act-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_inact, 1:nub_inact, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = j+1 , nob_inact
               do a = 1 , nua_inact
                  do b = 1 , nub_inact
                     do c = 1 , nub_act

                        denom = fA_oo_act(i,i) + fB_oo_inact(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_inact(a,a) - fB_vv_inact(b,b) - fB_vv_act(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, c, b, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)

                        resid(a, c, b, i, j, k) = val
                        resid(a, c, b, i, k, j) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_001100

subroutine update_t3c_100100(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_inact)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_inact)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_inact-1, 0:nub_inact-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_inact-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_inact)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = j+1 , nob_inact
               do a = 1 , nua_act
                  do b = 1 , nub_inact
                     do c = b+1 , nub_inact

                        denom = fA_oo_act(i,i) + fB_oo_inact(j,j) + fB_oo_inact(k,k)&
                               -fA_vv_act(a,a) - fB_vv_inact(b,b) - fB_vv_inact(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)
                        t3c(a, c, b, i, k, j) = t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, b, c, i, k, j) = -1.0 * val
                        resid(a, c, b, i, j, k) = -1.0 * val
                        resid(a, c, b, i, k, j) = val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_100100

subroutine update_t3c_100001(t3c, resid, X3C, &
                             fA_oo_act, fA_vv_act, fA_oo_inact, fA_vv_inact, &
                             fB_oo_act, fB_vv_act, fB_oo_inact, fB_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: fA_oo_act(1:noa_act, 1:noa_act), &
                              fA_vv_act(1:nua_act, 1:nua_act), &
                              fA_oo_inact(1:noa_inact, 1:noa_inact), &
                              fA_vv_inact(1:nua_inact, 1:nua_inact), &
                              fB_oo_act(1:nob_act, 1:nob_act), &
                              fB_vv_act(1:nub_act, 1:nub_act), &
                              fB_oo_inact(1:nob_inact, 1:nob_inact), &
                              fB_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: X3C(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_inact, 1:nob_inact, 1:nob_act)
      real(8), intent(in)  :: shift

      real(8), intent(inout) :: t3c(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_inact, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: t3c(0:nua_act-1, 0:nub_inact-1, 0:nub_inact-1, 0:noa_inact-1, 0:nob_inact-1, 0:nob_act-1)
      real(8), intent(out)   :: resid(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_inact, 1:nob_inact, 1:nob_act)

      integer :: i, j, k, a, b, c
      real(8) :: denom, val

      do i = 1 , noa_inact
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_inact
                     do c = b+1 , nub_inact

                        denom = fA_oo_inact(i,i) + fB_oo_inact(j,j) + fB_oo_act(k,k)&
                               -fA_vv_act(a,a) - fB_vv_inact(b,b) - fB_vv_inact(c,c)

                        val = X3C(a, b, c, i, j, k)/(denom - shift)

                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k) + val
                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)

                        resid(a, b, c, i, j, k) = val
                        resid(a, c, b, i, j, k) = -1.0 * val

                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_t3c_100001

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  T3D UPDATES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


end module cc_active_loops
