module cc_active_loops

  implicit none

  contains

      subroutine update_t3a_111111(t3a,resid,X3A,&
                                   fA_oo,fA_vv,&
                                   shift,&
                                   noa_act,nua_act)

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

                                      val = X3A(A, B, C, I, J, K)/(denom-shift)

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


      subroutine update_t3a_110111(t3a,resid,X3A,&
                                   fA_oo_act,fA_vv_act,fA_oo_inact,fA_vv_inact,&
                                   shift,&
                                   noa_act,nua_act,noa_inact,nua_inact)

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
                                      val = X3A(A, B, c, I, J, K)/(denom - shift)

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



      subroutine update_t3a_111011(t3a,resid,X3A,&
                                   fA_oo_act,fA_vv_act,fA_oo_inact,fA_vv_inact,&
                                   shift,&
                                   noa_act,nua_act,noa_inact,nua_inact)

              integer, intent(in) :: noa_act, nua_act, noa_inact, nua_inact
              real(8), intent(in) :: fA_oo_act(1:noa_act,1:noa_act),&
                                     fA_vv_act(1:nua_act,1:nua_act),&
                                     fA_oo_inact(1:noa_inact,1:noa_inact),&
                                     fA_vv_inact(1:nua_inact,1:nua_inact),&
                                     X3A(1:nua_act,1:nua_act,1:nua_act,1:noa_inact,1:noa_act,1:noa_act),&
                                     shift
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

                                      val = X3A(A, B, C, i, J, K)/(denom - shift)

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


          subroutine update_t3a_110011(t3a,resid,X3A,&
                                   fA_oo_act,fA_vv_act,fA_oo_inact,fA_vv_inact,&
                                   shift,&
                                   noa_act,nua_act,noa_inact,nua_inact)

              integer, intent(in) :: noa_act, nua_act, noa_inact, nua_inact
              real(8), intent(in) :: fA_oo_act(1:noa_act,1:noa_act),&
                                     fA_vv_act(1:nua_act,1:nua_act),&
                                     fA_oo_inact(1:noa_inact,1:noa_inact),&
                                     fA_vv_inact(1:nua_inact,1:nua_inact),&
                                     X3A(1:nua_act,1:nua_act,1:nua_act,1:noa_inact,1:noa_act,1:noa_act),&
                                     shift
              real(8), intent(inout) :: t3a(1:nua_act,1:nua_act,1:nua_inact,1:noa_inact,1:noa_act,1:noa_act)
              !f2py intent(in,out) :: t3a(0:nua_act-1,0:nua_act-1,0:nua_inact-1,0:noa_inact-1,0:noa_act-1,0:noa_act-1)
              real(8), intent(out) :: resid(1:nua_act,1:nua_act,1:nua_inact,1:noa_inact,1:noa_act,1:noa_act)
              integer :: i, j, k, a, b, c
              real(8) :: denom, val

              do i = 1,noa_inact
                  do J = 1,noa_act
                      do K = J+1,noa_act
                          do A = 1,nua_act
                              do B = A+1,nua_act
                                  do c = 1,nua_inact

                                      denom = fA_oo_inact(i,i)+fA_oo_act(J,J)+fA_oo_act(K,K)&
                                              -fA_vv_act(A,A)-fA_vv_act(B,B)-fA_vv_inact(c,c)

                                      val = X3A(A, B, c, i, J, K)/(denom - shift)

                                      t3a(A, B, c, i, J, K) = t3a(A, B, c, i, J, K) + val
                                      t3a(B, A, c, i, J, K) = -1.0 * t3a(A, B, c, i, J, K)
                                      t3a(A, B, c, i, K, J) = -1.0 * t3a(A, B, c, i, J, K)
                                      t3a(B, A, c, i, K, J) = t3a(A, B, c, i, J, K)

                                      resid(A, B, c, i, J, K) = val
                                      resid(B, A, c, i, J, K) = -1.0 * val
                                      resid(A, B, c, i, K, J) = -1.0 * val
                                      resid(B, A, c, i, K, J) = val
                               end do
                           end do
                        end do
                     end do
                  end do
               end do
          end subroutine update_t3a_110011

        subroutine update_t3a_100111(t3a,resid,X3A,&
                                       fA_oo_act,fA_vv_act,fA_oo_inact,fA_vv_inact,&
                                       shift,&
                                       noa_act,nua_act,noa_inact,nua_inact)

              integer, intent(in) :: noa_act, nua_act, noa_inact, nua_inact
              real(8), intent(in) :: fA_oo_act(1:noa_act,1:noa_act),&
                                     fA_vv_act(1:nua_act,1:nua_act),&
                                     fA_oo_inact(1:noa_inact,1:noa_inact),&
                                     fA_vv_inact(1:nua_inact,1:nua_inact),&
                                     X3A(1:nua_act,1:nua_inact,1:nua_inact,1:noa_act,1:noa_act,1:noa_act),&
                                     shift
              real(8), intent(inout) :: t3a(1:nua_act,1:nua_inact,1:nua_inact,1:noa_act,1:noa_act,1:noa_act)
              !f2py intent(in,out) :: t3a(0:nua_act-1,0:nua_inact-1,0:nua_inact-1,0:noa_act-1,0:noa_act-1,0:noa_act-1)
              real(8), intent(out) :: resid(1:nua_act,1:nua_inact,1:nua_inact,1:noa_act,1:noa_act,1:noa_act)
              integer :: i, j, k, a, b, c
              real(8) :: denom, val

              do I = 1,noa_act
                  do J = I+1,noa_act
                      do K = J+1,noa_act
                          do A = 1,nua_act
                              do b = 1,nua_inact
                                  do c = B+1,nua_inact

                                      denom = fA_oo_act(I,I)+fA_oo_act(J,J)+fA_oo_act(K,K)&
                                              -fA_vv_act(A,A)-fA_vv_inact(b,b)-fA_vv_inact(c,c)

                                      val = X3A(A, b, c, I, J, K)/(denom - shift)

                                      t3a(A, b, c, I, J, K) = t3a(A, b, c, I, J, K) + val
                                      t3a(A, b, c, I, K, J) = -1.0 * t3a(A, b, c, I, J, K)
                                      t3a(A, b, c, J, K, I) = t3a(A, b, c, I, J, K)
                                      t3a(A, b, c, J, I, K) = -1.0 * t3a(A, b, c, I, J, K)
                                      t3a(A, b, c, K, I, J) = t3a(A, b, c, I, J, K)
                                      t3a(A, b, c, K, J, I) = -1.0 * t3a(A, b, c, I, J, K)
                                      t3a(A, c, b, I, J, K) = -1.0 * t3a(A, b, c, I, J, K)
                                      t3a(A, c, b, I, K, J) = t3a(A, b, c, I, J, K)
                                      t3a(A, c, b, J, K, I) = -1.0 * t3a(A, b, c, I, J, K)
                                      t3a(A, c, b, J, I, K) = t3a(A, b, c, I, J, K)
                                      t3a(A, c, b, K, I, J) = -1.0 * t3a(A, b, c, I, J, K)
                                      t3a(A, c, b, K, J, I) = t3a(A, b, c, I, J, K)

                                      resid(A, b, c, I, J, K) = val
                                      resid(A, b, c, I, K, J) = -1.0 * val
                                      resid(A, b, c, J, K, I) = val
                                      resid(A, b, c, J, I, K) = -1.0 * val
                                      resid(A, b, c, K, I, J) = val
                                      resid(A, b, c, K, J, I) = -1.0 * val
                                      resid(A, c, b, I, J, K) = -1.0 * val
                                      resid(A, c, b, I, K, J) = val
                                      resid(A, c, b, J, K, I) = -1.0 * val
                                      resid(A, c, b, J, I, K) = val
                                      resid(A, c, b, K, I, J) = -1.0 * val
                                      resid(A, c, b, K, J, I) = val
                               end do
                           end do
                        end do
                     end do
                  end do
               end do
        end subroutine update_t3a_100111

        subroutine update_t3a_111001(t3a,resid,X3A,&
                                     fA_oo_act,fA_vv_act,fA_oo_inact,fA_vv_inact,&
                                     shift,&
                                     noa_act,nua_act,noa_inact,nua_inact)

          integer, intent(in) :: noa_act, nua_act, noa_inact, nua_inact
          real(8), intent(in) :: fA_oo_act(1:noa_act,1:noa_act),&
                                 fA_vv_act(1:nua_act,1:nua_act),&
                                 fA_oo_inact(1:noa_inact,1:noa_inact),&
                                 fA_vv_inact(1:nua_inact,1:nua_inact),&
                                 X3A(1:nua_act,1:nua_act,1:nua_act,1:noa_inact,1:noa_inact,1:noa_act),&
                                 shift
          real(8), intent(inout) :: t3a(1:nua_act,1:nua_act,1:nua_act,1:noa_inact,1:noa_inact,1:noa_act)
          !f2py intent(in,out) :: t3a(0:nua_act-1,0:nua_act-1,0:nua_act-1,0:noa_inact-1,0:noa_inact-1,0:noa_act-1)
          real(8), intent(out) :: resid(1:nua_act,1:nua_act,1:nua_act,1:noa_inact,1:noa_inact,1:noa_act)
          integer :: i, j, k, a, b, c
          real(8) :: denom, val

            do I = 1 , noa_act
                do j = 1, noa_inact
                    do k = j + 1, noa_inact
                        do A = 1 , nua_act
                            do B = A + 1 , nua_act
                                do C = B + 1, nua_act
                                     denom = fA_oo_act(I,I)+fA_oo_inact(J,J)+fA_oo_inact(K,K)&
                                              -fA_vv_act(A,A)-fA_vv_act(b,b)-fA_vv_act(c,c)

                                      val = X3A(A, B, C, i, j, K)/(denom - shift)

                                    t3a(A, B, C, i, j, K) = t3a(A, B, C, i, j, K) + val
                                    t3a(A, C, B, i, j, K) = -1.0 * t3a(A, B, C, i, j, K)
                                    t3a(B, C, A, i, j, K) = t3a(A, B, C, i, j, K)
                                    t3a(B, A, C, i, j, K) = -1.0 * t3a(A, B, C, i, j, K)
                                    t3a(C, A, B, i, j, K) = t3a(A, B, C, i, j, K)
                                    t3a(C, B, A, i, j, K) = -1.0 * t3a(A, B, C, i, j, K)
                                    t3a(A, B, C, j, i, K) = -1.0 * t3a(A, B, C, i, j, K)
                                    t3a(A, C, B, j, i, K) = t3a(A, B, C, i, j, K)
                                    t3a(B, C, A, j, i, K) = -1.0 * t3a(A, B, C, i, j, K)
                                    t3a(B, A, C, j, i, K) = t3a(A, B, C, i, j, K)
                                    t3a(C, A, B, j, i, K) = -1.0 * t3a(A, B, C, i, j, K)
                                    t3a(C, B, A, j, i, K) = t3a(A, B, C, i, j, K)

                                    resid(A, B, C, i, j, K) = val
                                    resid(A, C, B, i, j, K) = -1.0 * val
                                    resid(B, C, A, i, j, K) = val
                                    resid(B, A, C, i, j, K) = -1.0 * val
                                    resid(C, A, B, i, j, K) = val
                                    resid(C, B, A, i, j, K) = -1.0 * val
                                    resid(A, B, C, j, i, K) = -1.0 * val
                                    resid(A, C, B, j, i, K) = val
                                    resid(B, C, A, j, i, K) = -1.0 * val
                                    resid(B, A, C, j, i, K) = val
                                    resid(C, A, B, j, i, K) = -1.0 * val
                                    resid(C, B, A, j, i, K) = val
                                end do
                            end do
                        end do
                    end do
                end do
          end do

      end subroutine update_t3a_111001

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
                 do j = i + 1 , noa_inact
                    do k = 1 , noa_act
                       do a = 1 , nua_act
                          do b = a + 1 , nua_act
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


end module cc_active_loops
