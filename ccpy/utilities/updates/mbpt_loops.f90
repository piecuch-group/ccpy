module mbpt_loops

      implicit none

      contains

              subroutine mp2(fA_oo,fA_vv,fB_oo,fB_vv,&
                             vA_oovv,vA_vvoo,vB_oovv,vB_vvoo,vC_oovv,vC_vvoo,&
                             noa,nob,nua,nub,Emp2)

                    integer, intent(in) :: noa, nob, nua, nub
                    real(kind=8), intent(in) :: fA_oo(noa,noa),fB_oo(nob,nob),&
                                                fA_vv(nua,nua),fB_vv(nub,nub),&
                                                vA_oovv(noa,noa,nua,nua),&
                                                vA_vvoo(nua,nua,noa,noa),&
                                                vB_oovv(noa,nob,nua,nub),&
                                                vB_vvoo(nua,nub,noa,nob),&
                                                vC_oovv(nob,nob,nub,nub),&
                                                vC_vvoo(nub,nub,nob,nob)
                    real(kind=8), intent(out) :: Emp2

                    real(kind=8) :: denom, t_amp
                    integer :: i, j, a, b

                    Emp2 = 0.0d0
                    do i = 1 , noa
                       do j = i+1 , noa
                          do a = 1 , nua
                             do b = a+1 , nua
                                denom = fA_oo(i,i)+fA_oo(j,j)-fA_vv(a,a)-fA_vv(b,b)
                                t_amp = vA_vvoo(a,b,i,j)/denom
                                Emp2 = Emp2 + vA_oovv(i,j,a,b)*t_amp
                             end do
                          end do
                       end do
                    end do
                    do i = 1 , noa
                       do j = 1 , nob
                          do a = 1 , nua
                             do b = 1 , nub
                                denom = fA_oo(i,i)+fB_oo(j,j)-fA_vv(a,a)-fB_vv(b,b)
                                t_amp = vB_vvoo(a,b,i,j)/denom
                                Emp2 = Emp2 + vB_oovv(i,j,a,b)*t_amp
                             end do
                          end do
                       end do
                    end do
                    do i = 1 , nob
                       do j = i+1 , nob
                          do a = 1 , nub
                             do b = a+1 , nub
                                denom = fB_oo(i,i)+fB_oo(j,j)-fB_vv(a,a)-fB_vv(b,b)
                                t_amp = vC_vvoo(a,b,i,j)/denom
                                Emp2 = Emp2 + vC_oovv(i,j,a,b)*t_amp
                             end do
                          end do
                       end do
                    end do
              end subroutine mp2

              subroutine mp3(fA_oo,fA_vv,fB_oo,fB_vv,&
                             vA_oovv,vA_vvoo,vA_voov,vA_oooo,vA_vvvv,&
                             vB_oovv,vB_vvoo,vB_voov,vB_ovvo,vB_vovo,vB_ovov,vB_oooo,vB_vvvv,&
                             vC_oovv,vC_vvoo,vC_voov,vC_oooo,vC_vvvv,&
                             noa,nob,nua,nub,Emp3)

                    integer, intent(in) :: noa, nob, nua, nub
                    real(kind=8), intent(in) :: fA_oo(noa,noa),fB_oo(nob,nob),&
                                                fA_vv(nua,nua),fB_vv(nub,nub),&
                                                vA_oovv(noa,noa,nua,nua),&
                                                vA_vvoo(nua,nua,noa,noa),&
                                                vA_voov(nua,noa,noa,nua),&
                                                vA_oooo(noa,noa,noa,noa),&
                                                vA_vvvv(nua,nua,nua,nua),&
                                                vB_oovv(noa,nob,nua,nub),&
                                                vB_vvoo(nua,nub,noa,nob),&
                                                vB_voov(nua,nob,noa,nub),&
                                                vB_ovvo(noa,nub,nua,nob),&
                                                vB_ovov(noa,nub,noa,nub),&
                                                vB_vovo(nua,nob,nua,nob),&
                                                vB_oooo(noa,nob,noa,nob),&
                                                vB_vvvv(nua,nub,nua,nub),&
                                                vC_oovv(nob,nob,nub,nub),&
                                                vC_vvoo(nub,nub,nob,nob),&
                                                vC_voov(nub,nob,nob,nub),&
                                                vC_oooo(nob,nob,nob,nob),&
                                                vC_vvvv(nub,nub,nub,nub)
                    real(kind=8), intent(out) :: Emp3

                    real(kind=8) :: t2a_1(nua,nua,noa,noa), t2b_1(nua,nub,noa,nob), t2c_1(nub,nub,nob,nob)

                    integer :: i, j, k, a, b, c, m, n, e, f
                    real(kind=8) :: denom, t_amp

                    ! compute t2a_1, t2b_1, and t2c_1 using R2*V|0>
                    t2a_1 = 0.0d0; t2b_1 = 0.0d0; t2c_1 = 0.0d0;
                    do a = 1, nua
                       do b = a+1, nua
                          do i = 1, noa
                             do j = i+1, noa
                                denom = fA_oo(i,i) + fA_oo(j,j) - fA_vv(a,a) - fA_vv(b,b)
                                t2a_1(a,b,i,j) = vA_vvoo(a,b,i,j)/denom
                                t2a_1(b,a,i,j) = -t2a_1(a,b,i,j)
                                t2a_1(a,b,j,i) = -t2a_1(a,b,i,j)
                                t2a_1(b,a,j,i) = t2a_1(a,b,i,j)
                             end do
                          end do
                       end do
                    end do
                    do a = 1, nua
                       do b = 1, nub
                          do i = 1, noa
                             do j = 1, nob
                                denom = fA_oo(i,i) + fB_oo(j,j) - fA_vv(a,a) - fB_vv(b,b)
                                t2b_1(a,b,i,j) = vB_vvoo(a,b,i,j)/denom
                             end do
                          end do
                       end do
                    end do
                    do a = 1, nub
                       do b = a+1, nub
                          do i = 1, nob
                             do j = i+1, nob
                                denom = fB_oo(i,i) + fB_oo(j,j) - fB_vv(a,a) - fB_vv(b,b)
                                t2c_1(a,b,i,j) = vC_vvoo(a,b,i,j)/denom
                                t2c_1(b,a,i,j) = -t2c_1(a,b,i,j)
                                t2c_1(a,b,j,i) = -t2c_1(a,b,i,j)
                                t2c_1(b,a,j,i) = t2c_1(a,b,i,j)
                             end do
                          end do
                       end do
                    end do
                    
                    ! compute E(D) = <0|V * T2[2]|0>, where T2[2] = R2*V*T2[1]|0>
                    do a = 1, nua
                       do b = a+1, nua
                          do i = 1, noa
                             do j = i+1, noa
                                denom = fA_oo(i,i) + fA_oo(j,j) - fA_vv(a,a) - fA_vv(b,b)
                                t_amp = 0.0d0
                                ! (i) hole diagram
                                do m = 1, noa
                                   do n = m+1, noa
                                      t_amp = t_amp + vA_oooo(m,n,i,j) * t2a_1(a,b,m,n)
                                   end do
                                end do
                                ! (ii) particle diagram
                                do e = 1, nua
                                   do f = e+1, nua
                                      t_amp = t_amp + vA_vvvv(a,b,e,f) * t2a_1(e,f,i,j)
                                   end do
                                end do
                                ! (iii) hole-particle diagrams
                                do e = 1, nua
                                   do m = 1, noa
                                      t_amp = t_amp + vA_voov(a,m,i,e) * t2a_1(e,b,m,j)&
                                                    - vA_voov(b,m,i,e) * t2a_1(e,a,m,j)&
                                                    - vA_voov(a,m,j,e) * t2a_1(e,b,m,i)&
                                                    + vA_voov(b,m,j,e) * t2a_1(e,a,m,i)
                                   end do
                                end do
                                do e = 1, nub
                                   do m = 1, nob
                                      t_amp = t_amp + vB_voov(a,m,i,e) * t2b_1(b,e,j,m)&
                                                    - vB_voov(b,m,i,e) * t2b_1(a,e,j,m)&
                                                    - vB_voov(a,m,j,e) * t2b_1(b,e,i,m)&
                                                    + vB_voov(b,m,j,e) * t2b_1(a,e,i,m)
                                   end do
                                end do
                                t_amp = t_amp/denom
                                Emp3 = Emp3 + vA_oovv(i,j,a,b) * t_amp
                             end do
                          end do
                       end do
                    end do
                    do a = 1, nua
                       do b = 1, nub
                          do i = 1, noa
                             do j = 1, nob
                                denom = fA_oo(i,i) + fB_oo(j,j) - fA_vv(a,a) - fB_vv(b,b)
                                t_amp = 0.0d0
                                ! (i) hole diagram
                                do m = 1, noa
                                   do n = 1, nob
                                      t_amp = t_amp + vB_oooo(m,n,i,j) * t2b_1(a,b,m,n)
                                   end do
                                end do
                                ! (ii) particle diagram
                                do e = 1, nua
                                   do f = 1, nub
                                      t_amp = t_amp + vB_vvvv(a,b,e,f) * t2b_1(e,f,i,j)
                                   end do
                                end do
                                ! (iii) hole-particle diagrams
                                do e = 1, nua
                                   do m = 1, noa
                                      t_amp = t_amp + vA_voov(a,m,i,e) * t2b_1(e,b,m,j)
                                   end do
                                end do
                                do e = 1, nub
                                   do m = 1, nob
                                      t_amp = t_amp + vC_voov(b,m,j,e) * t2b_1(a,e,i,m)
                                   end do
                                end do
                                do e = 1, nua
                                   do m = 1, noa
                                      t_amp = t_amp + vB_ovvo(m,b,e,j) * t2a_1(a,e,i,m)
                                   end do
                                end do
                                do e = 1, nub
                                   do m = 1, nob
                                      t_amp = t_amp + vB_voov(a,m,i,e) * t2c_1(b,e,j,m)
                                   end do
                                end do
                                do e = 1, nub
                                   do m = 1, noa
                                      t_amp = t_amp - vB_ovov(m,b,i,e) * t2b_1(a,e,m,j)
                                   end do
                                end do
                                do e = 1, nua
                                   do m = 1, nob
                                      t_amp = t_amp - vB_vovo(a,m,e,j) * t2b_1(e,b,i,m)
                                   end do
                                end do
                                t_amp = t_amp/denom
                                Emp3 = Emp3 + vB_oovv(i,j,a,b) * t_amp
                             end do
                          end do
                       end do
                    end do
                    do a = 1, nub
                       do b = a+1, nub
                          do i = 1, nob
                             do j = i+1, nob
                                denom = fB_oo(i,i) + fB_oo(j,j) - fB_vv(a,a) - fB_vv(b,b)
                                t_amp = 0.0d0
                                ! (i) hole diagram
                                do m = 1, nob
                                   do n = m+1, nob
                                      t_amp = t_amp + vC_oooo(m,n,i,j) * t2c_1(a,b,m,n)
                                   end do
                                end do
                                ! (ii) particle diagram
                                do e = 1, nub
                                   do f = e+1, nub
                                      t_amp = t_amp + vC_vvvv(a,b,e,f) * t2c_1(e,f,i,j)
                                   end do
                                end do
                                ! (iii) hole-particle diagrams
                                do e = 1, nub
                                   do m = 1, nob
                                      t_amp = t_amp + vC_voov(a,m,i,e) * t2c_1(e,b,m,j)&
                                                    - vC_voov(b,m,i,e) * t2c_1(e,a,m,j)&
                                                    - vC_voov(a,m,j,e) * t2c_1(e,b,m,i)&
                                                    + vC_voov(b,m,j,e) * t2c_1(e,a,m,i)
                                   end do
                                end do
                                do e = 1, nua
                                   do m = 1, noa
                                      t_amp = t_amp + vB_ovvo(m,a,e,i) * t2b_1(e,b,m,j)&
                                                    - vB_ovvo(m,b,e,i) * t2b_1(e,a,m,j)&
                                                    - vB_ovvo(m,a,e,j) * t2b_1(e,b,m,i)&
                                                    + vB_ovvo(m,b,e,j) * t2b_1(e,a,m,i)
                                   end do
                                end do
                                t_amp = t_amp/denom
                                Emp3 = Emp3 + vC_oovv(i,j,a,b) * t_amp
                             end do
                          end do
                       end do
                    end do

              end subroutine mp3

              subroutine mp4(fA_oo,fA_vv,fB_oo,fB_vv,&
                             vA_oovv,vA_vvoo,vA_voov,vA_oooo,vA_vvvv,vA_vooo,vA_vvov,vA_ooov,vA_vovv,&
                             vB_oovv,vB_vvoo,vB_voov,vB_ovvo,vB_vovo,vB_ovov,vB_oooo,vB_vvvv,&
                             vB_vooo,vB_ovoo,vB_vvov,vB_vvvo,vB_ooov,vB_oovo,vB_vovv,vB_ovvv,&
                             vC_oovv,vC_vvoo,vC_voov,vC_oooo,vC_vvvv,vC_vooo,vC_vvov,vC_ooov,vC_vovv,&
                             noa,nob,nua,nub,Emp4,eS,eD,eT,eQ)

                    integer, intent(in) :: noa, nob, nua, nub
                    real(kind=8), intent(in) :: fA_oo(noa,noa),fB_oo(nob,nob),&
                                                fA_vv(nua,nua),fB_vv(nub,nub),&
                                                vA_oovv(noa,noa,nua,nua),&
                                                vA_vvoo(nua,nua,noa,noa),&
                                                vA_voov(nua,noa,noa,nua),&
                                                vA_oooo(noa,noa,noa,noa),&
                                                vA_vvvv(nua,nua,nua,nua),&
                                                vA_vooo(nua,noa,noa,noa),&
                                                vA_vvov(nua,nua,noa,nua),&
                                                vA_ooov(noa,noa,noa,nua),&
                                                vA_vovv(nua,noa,nua,nua),&
                                                vB_oovv(noa,nob,nua,nub),&
                                                vB_vvoo(nua,nub,noa,nob),&
                                                vB_voov(nua,nob,noa,nub),&
                                                vB_ovvo(noa,nub,nua,nob),&
                                                vB_ovov(noa,nub,noa,nub),&
                                                vB_vovo(nua,nob,nua,nob),&
                                                vB_oooo(noa,nob,noa,nob),&
                                                vB_vvvv(nua,nub,nua,nub),&
                                                vB_vooo(nua,nob,noa,nob),&
                                                vB_ovoo(noa,nub,noa,nob),&
                                                vB_vvov(nua,nub,noa,nub),&
                                                vB_vvvo(nua,nub,nua,nob),&
                                                vB_ooov(noa,nob,noa,nub),&
                                                vB_oovo(noa,nob,nua,nob),&
                                                vB_vovv(nua,nob,nua,nub),&
                                                vB_ovvv(noa,nub,nua,nub),&
                                                vC_oovv(nob,nob,nub,nub),&
                                                vC_vvoo(nub,nub,nob,nob),&
                                                vC_voov(nub,nob,nob,nub),&
                                                vC_oooo(nob,nob,nob,nob),&
                                                vC_vvvv(nub,nub,nub,nub),&
                                                vC_vooo(nub,nob,nob,nob),&
                                                vC_vvov(nub,nub,nob,nub),&
                                                vC_ooov(nob,nob,nob,nub),&
                                                vC_vovv(nub,nob,nub,nub)
                    real(kind=8), intent(out) :: Emp4, eS, eD, eT, eQ

                    real(kind=8) :: t2a_1(nua,nua,noa,noa), t2b_1(nua,nub,noa,nob), t2c_1(nub,nub,nob,nob)
                    real(kind=8) :: x2a_voov(nua,noa,noa,nua),&
                                    x2a_oooo(noa,noa,noa,noa),&
                                    x1a_oo(noa,noa)

                    integer :: i, j, k, l, a, b, c, d, m, n, e, f
                    real(kind=8) :: denom, t_amp

                    ! compute t2a_1, t2b_1, and t2c_1 using R2*V|0>
                    t2a_1 = 0.0d0; t2b_1 = 0.0d0; t2c_1 = 0.0d0;
                    do a = 1, nua
                       do b = a+1, nua
                          do i = 1, noa
                             do j = i+1, noa
                                denom = fA_oo(i,i) + fA_oo(j,j) - fA_vv(a,a) - fA_vv(b,b)
                                t2a_1(a,b,i,j) = vA_vvoo(a,b,i,j)/denom
                                t2a_1(b,a,i,j) = -t2a_1(a,b,i,j)
                                t2a_1(a,b,j,i) = -t2a_1(a,b,i,j)
                                t2a_1(b,a,j,i) = t2a_1(a,b,i,j)
                             end do
                          end do
                       end do
                    end do
                    do a = 1, nua
                       do b = 1, nub
                          do i = 1, noa
                             do j = 1, nob
                                denom = fA_oo(i,i) + fB_oo(j,j) - fA_vv(a,a) - fB_vv(b,b)
                                t2b_1(a,b,i,j) = vB_vvoo(a,b,i,j)/denom
                             end do
                          end do
                       end do
                    end do
                    do a = 1, nub
                       do b = a+1, nub
                          do i = 1, nob
                             do j = i+1, nob
                                denom = fB_oo(i,i) + fB_oo(j,j) - fB_vv(a,a) - fB_vv(b,b)
                                t2c_1(a,b,i,j) = vC_vvoo(a,b,i,j)/denom
                                t2c_1(b,a,i,j) = -t2c_1(a,b,i,j)
                                t2c_1(a,b,j,i) = -t2c_1(a,b,i,j)
                                t2c_1(b,a,j,i) = t2c_1(a,b,i,j)
                             end do
                          end do
                       end do
                    end do
                    
                    ! compute E(D) = <0|(T2[2])^+ * R2 * T2[2]|0>, where T2[2] = V*T2[1|0>
                    eD = 0.0d0
                    do a = 1, nua
                       do b = a+1, nua
                          do i = 1, noa
                             do j = i+1, noa
                                denom = fA_oo(i,i) + fA_oo(j,j) - fA_vv(a,a) - fA_vv(b,b)
                                t_amp = 0.0d0
                                ! (i) hole diagram
                                do m = 1, noa
                                   do n = m+1, noa
                                      t_amp = t_amp + vA_oooo(m,n,i,j) * t2a_1(a,b,m,n)
                                   end do
                                end do
                                ! (ii) particle diagram
                                do e = 1, nua
                                   do f = e+1, nua
                                      t_amp = t_amp + vA_vvvv(a,b,e,f) * t2a_1(e,f,i,j)
                                   end do
                                end do
                                ! (iii) hole-particle diagrams
                                do e = 1, nua
                                   do m = 1, noa
                                      t_amp = t_amp + vA_voov(a,m,i,e) * t2a_1(e,b,m,j)&
                                                    - vA_voov(b,m,i,e) * t2a_1(e,a,m,j)&
                                                    - vA_voov(a,m,j,e) * t2a_1(e,b,m,i)&
                                                    + vA_voov(b,m,j,e) * t2a_1(e,a,m,i)
                                   end do
                                end do
                                do e = 1, nub
                                   do m = 1, nob
                                      t_amp = t_amp + vB_voov(a,m,i,e) * t2b_1(b,e,j,m)&
                                                    - vB_voov(b,m,i,e) * t2b_1(a,e,j,m)&
                                                    - vB_voov(a,m,j,e) * t2b_1(b,e,i,m)&
                                                    + vB_voov(b,m,j,e) * t2b_1(a,e,i,m)
                                   end do
                                end do
                                eD = eD + t_amp * t_amp / denom
                             end do
                          end do
                       end do
                    end do
                    do a = 1, nua
                       do b = 1, nub
                          do i = 1, noa
                             do j = 1, nob
                                denom = fA_oo(i,i) + fB_oo(j,j) - fA_vv(a,a) - fB_vv(b,b)
                                t_amp = 0.0d0
                                ! (i) hole diagram
                                do m = 1, noa
                                   do n = 1, nob
                                      t_amp = t_amp + vB_oooo(m,n,i,j) * t2b_1(a,b,m,n)
                                   end do
                                end do
                                ! (ii) particle diagram
                                do e = 1, nua
                                   do f = 1, nub
                                      t_amp = t_amp + vB_vvvv(a,b,e,f) * t2b_1(e,f,i,j)
                                   end do
                                end do
                                ! (iii) hole-particle diagrams
                                do e = 1, nua
                                   do m = 1, noa
                                      t_amp = t_amp + vA_voov(a,m,i,e) * t2b_1(e,b,m,j)
                                   end do
                                end do
                                do e = 1, nub
                                   do m = 1, nob
                                      t_amp = t_amp + vC_voov(b,m,j,e) * t2b_1(a,e,i,m)
                                   end do
                                end do
                                do e = 1, nua
                                   do m = 1, noa
                                      t_amp = t_amp + vB_ovvo(m,b,e,j) * t2a_1(a,e,i,m)
                                   end do
                                end do
                                do e = 1, nub
                                   do m = 1, nob
                                      t_amp = t_amp + vB_voov(a,m,i,e) * t2c_1(b,e,j,m)
                                   end do
                                end do
                                do e = 1, nub
                                   do m = 1, noa
                                      t_amp = t_amp - vB_ovov(m,b,i,e) * t2b_1(a,e,m,j)
                                   end do
                                end do
                                do e = 1, nua
                                   do m = 1, nob
                                      t_amp = t_amp - vB_vovo(a,m,e,j) * t2b_1(e,b,i,m)
                                   end do
                                end do
                                eD = eD + t_amp * t_amp / denom
                             end do
                          end do
                       end do
                    end do
                    do a = 1, nub
                       do b = a+1, nub
                          do i = 1, nob
                             do j = i+1, nob
                                denom = fB_oo(i,i) + fB_oo(j,j) - fB_vv(a,a) - fB_vv(b,b)
                                t_amp = 0.0d0
                                ! (i) hole diagram
                                do m = 1, nob
                                   do n = m+1, nob
                                      t_amp = t_amp + vC_oooo(m,n,i,j) * t2c_1(a,b,m,n)
                                   end do
                                end do
                                ! (ii) particle diagram
                                do e = 1, nub
                                   do f = e+1, nub
                                      t_amp = t_amp + vC_vvvv(a,b,e,f) * t2c_1(e,f,i,j)
                                   end do
                                end do
                                ! (iii) hole-particle diagrams
                                do e = 1, nub
                                   do m = 1, nob
                                      t_amp = t_amp + vC_voov(a,m,i,e) * t2c_1(e,b,m,j)&
                                                    - vC_voov(b,m,i,e) * t2c_1(e,a,m,j)&
                                                    - vC_voov(a,m,j,e) * t2c_1(e,b,m,i)&
                                                    + vC_voov(b,m,j,e) * t2c_1(e,a,m,i)
                                   end do
                                end do
                                do e = 1, nua
                                   do m = 1, noa
                                      t_amp = t_amp + vB_ovvo(m,a,e,i) * t2b_1(e,b,m,j)&
                                                    - vB_ovvo(m,b,e,i) * t2b_1(e,a,m,j)&
                                                    - vB_ovvo(m,a,e,j) * t2b_1(e,b,m,i)&
                                                    + vB_ovvo(m,b,e,j) * t2b_1(e,a,m,i)
                                   end do
                                end do
                                eD = eD + t_amp * t_amp / denom
                             end do
                          end do
                       end do
                    end do
                    ! compute E(S) = <0|(T1[2])^+ * R1* T1[2]|0>, where T1[2] = V*T2[1]|0>
                    eS = 0.0d0
                    do a = 1, nua
                       do i = 1, noa
                          denom = fA_oo(i,i) - fA_vv(a,a)
                          t_amp = 0.0d0
                          ! (i) hole diagrams
                          do m = 1, noa
                             do n = m+1, noa
                                do f = 1, nua
                                   t_amp = t_amp - vA_ooov(m,n,i,f) * t2a_1(a,f,m,n)
                                end do
                             end do
                             do n = 1, nob
                                do f = 1, nub
                                   t_amp = t_amp - vB_ooov(m,n,i,f) * t2b_1(a,f,m,n)
                                end do
                             end do
                          end do
                          ! (ii) particle diagrams
                          do e = 1, nua
                             do f = e+1, nua
                                do n = 1, noa
                                   t_amp = t_amp + vA_vovv(a,n,e,f) * t2a_1(e,f,i,n)
                                end do
                             end do
                             do f = 1, nub
                                do n = 1, nob
                                   t_amp = t_amp + vB_vovv(a,n,e,f) * t2b_1(e,f,i,n)
                                end do
                             end do
                          end do
                          eS = eS + t_amp * t_amp / denom
                       end do
                    end do
                    do a = 1, nub
                       do i = 1, nob
                          denom = fB_oo(i,i) - fB_vv(a,a)
                          t_amp = 0.0d0
                          ! (i) hole diagrams
                          do m = 1, nob
                             do n = m+1, nob
                                do f = 1, nub
                                   t_amp = t_amp - vC_ooov(m,n,i,f) * t2c_1(a,f,m,n)
                                end do
                             end do
                             do n = 1, noa
                                do f = 1, nua
                                   t_amp = t_amp - vB_oovo(n,m,f,i) * t2b_1(f,a,n,m)
                                end do
                             end do
                          end do
                          ! (ii) particle diagrams
                          do e = 1, nub
                             do f = e+1, nub
                                do n = 1, nob
                                   t_amp = t_amp + vC_vovv(a,n,e,f) * t2c_1(e,f,i,n)
                                end do
                             end do
                             do f = 1, nua
                                do n = 1, noa
                                   t_amp = t_amp + vB_ovvv(n,a,f,e) * t2b_1(f,e,n,i)
                                end do
                             end do
                          end do
                          eS = eS + t_amp * t_amp / denom
                       end do
                    end do

                    ! compute E(T) = <0|(T3[2])^+ * R3 * T3[2]|0>, where T3[2] = V*T2[1]|0>
                    eT = 0.0d0
                    do a = 1, nua
                       do b = a+1, nua
                          do c = b+1, nua
                             do i = 1, noa
                                do j = i+1, noa
                                   do k = j+1, noa
                                      denom = fA_oo(i,i) + fA_oo(j,j) + fA_oo(k,k) - fA_vv(a,a) - fA_vv(b,b) - fA_vv(c,c)
                                      t_amp = 0.0d0
                                      ! (i) hole diagram
                                      do m = 1, noa
                                         t_amp = t_amp - vA_vooo(a,m,i,j) * t2a_1(b,c,m,k)& ! (1)
                                                       + vA_vooo(a,m,k,j) * t2a_1(b,c,m,i)& ! (ik)
                                                       + vA_vooo(a,m,i,k) * t2a_1(b,c,m,j)& ! (jk)
                                                       + vA_vooo(c,m,i,j) * t2a_1(b,a,m,k)& ! (ac)
                                                       - vA_vooo(c,m,k,j) * t2a_1(b,a,m,i)& ! (ac)(ik)
                                                       - vA_vooo(c,m,i,k) * t2a_1(b,a,m,j)& ! (ac)(jk)
                                                       + vA_vooo(b,m,i,j) * t2a_1(a,c,m,k)& ! (ab)
                                                       - vA_vooo(b,m,k,j) * t2a_1(a,c,m,i)& ! (ab)(ik)
                                                       - vA_vooo(b,m,i,k) * t2a_1(a,c,m,j)  ! (ab)(jk)
                                      end do
                                      ! (ii) particle diagram
                                      do e = 1, nua
                                         t_amp = t_amp + vA_vvov(a,b,i,e) * t2a_1(e,c,j,k)& ! (1)
                                                       - vA_vvov(a,b,j,e) * t2a_1(e,c,i,k)& ! (ij)
                                                       - vA_vvov(a,b,k,e) * t2a_1(e,c,j,i)& ! (ik)
                                                       - vA_vvov(c,b,i,e) * t2a_1(e,a,j,k)& ! (ac)
                                                       + vA_vvov(c,b,j,e) * t2a_1(e,a,i,k)& ! (ac)(ij)
                                                       + vA_vvov(c,b,k,e) * t2a_1(e,a,j,i)& ! (ac)(ik) 
                                                       - vA_vvov(a,c,i,e) * t2a_1(e,b,j,k)& ! (bc)
                                                       + vA_vvov(a,c,j,e) * t2a_1(e,b,i,k)& ! (bc)(ij)
                                                       + vA_vvov(a,c,k,e) * t2a_1(e,b,j,i)  ! (bc)(ik) 
                                      end do
                                      eT = eT + t_amp * t_amp / denom
                                   end do
                                end do
                             end do
                          end do
                       end do
                    end do
                    do a = 1, nua
                       do b = a+1, nua
                          do c = 1, nub
                             do i = 1, noa
                                do j = i+1, noa
                                   do k = 1, nob
                                      denom = fA_oo(i,i) + fA_oo(j,j) + fB_oo(k,k) - fA_vv(a,a) - fA_vv(b,b) - fB_vv(c,c)
                                      t_amp = 0.0d0
                                      do e = 1, nua
                                         t_amp = t_amp + vB_vvvo(b,c,e,k) * t2a_1(a,e,i,j)&
                                                       - vB_vvvo(a,c,e,k) * t2a_1(b,e,i,j)&
                                                       + vA_vvov(a,b,i,e) * t2b_1(e,c,j,k)&
                                                       - vA_vvov(a,b,j,e) * t2b_1(e,c,i,k)
                                      end do
                                      do e = 1, nub
                                         t_amp = t_amp + vB_vvov(a,c,i,e) * t2b_1(b,e,j,k)&
                                                       - vB_vvov(a,c,j,e) * t2b_1(b,e,i,k)&
                                                       - vB_vvov(b,c,i,e) * t2b_1(a,e,j,k)&
                                                       + vB_vvov(b,c,j,e) * t2b_1(a,e,i,k)
                                      end do
                                      do m = 1, noa
                                         t_amp = t_amp - vB_ovoo(m,c,j,k) * t2a_1(a,b,i,m)&
                                                       + vB_ovoo(m,c,i,k) * t2a_1(a,b,j,m)&
                                                       - vA_vooo(a,m,i,j) * t2b_1(b,c,m,k)&
                                                       + vA_vooo(b,m,i,j) * t2b_1(a,c,m,k)
                                      end do
                                      do m = 1, nob
                                         t_amp = t_amp - vB_vooo(a,m,i,k) * t2b_1(b,c,j,m)&
                                                       + vB_vooo(b,m,i,k) * t2b_1(a,c,j,m)&
                                                       + vB_vooo(a,m,j,k) * t2b_1(b,c,i,m)&
                                                       - vB_vooo(b,m,j,k) * t2b_1(a,c,i,m)
                                      end do 
                                      eT = eT + t_amp * t_amp / denom
                                   end do
                                end do
                             end do
                          end do
                       end do
                    end do
                    do a = 1, nua
                       do b = 1, nub
                          do c = b+1, nub
                             do i = 1, noa
                                do j = 1, nob
                                   do k = j+1, nob
                                      denom = fA_oo(i,i) + fB_oo(j,j) + fB_oo(k,k) - fA_vv(a,a) - fB_vv(b,b) - fB_vv(c,c)
                                      t_amp = 0.0d0
                                      do e = 1, nua
                                         t_amp = t_amp + vB_vvvo(a,b,e,j) * t2b_1(e,c,i,k)&
                                                       - vB_vvvo(a,b,e,k) * t2b_1(e,c,i,j)&
                                                       - vB_vvvo(a,c,e,j) * t2b_1(e,b,i,k)&
                                                       + vB_vvvo(a,c,e,k) * t2b_1(e,b,i,j)
                                      end do
                                      do e = 1, nub
                                         t_amp = t_amp + vB_vvov(a,b,i,e) * t2c_1(e,c,j,k)&
                                                       - vB_vvov(a,c,i,e) * t2c_1(e,b,j,k)&
                                                       + vC_vvov(c,b,k,e) * t2b_1(a,e,i,j)&
                                                       - vC_vvov(c,b,j,e) * t2b_1(a,e,i,k)
                                      end do
                                      do m = 1, noa
                                         t_amp = t_amp - vB_ovoo(m,b,i,j) * t2b_1(a,c,m,k)&
                                                       + vB_ovoo(m,c,i,j) * t2b_1(a,b,m,k)&
                                                       + vB_ovoo(m,b,i,k) * t2b_1(a,c,m,j)&
                                                       - vB_ovoo(m,c,i,k) * t2b_1(a,b,m,j)
                                      end do
                                      do m = 1, nob
                                         t_amp = t_amp - vB_vooo(a,m,i,j) * t2c_1(b,c,m,k)&
                                                       + vB_vooo(a,m,i,k) * t2c_1(b,c,m,j)&
                                                       - vC_vooo(c,m,k,j) * t2b_1(a,b,i,m)&
                                                       + vC_vooo(b,m,k,j) * t2b_1(a,c,i,m)
                                      end do
                                      eT = eT + t_amp * t_amp / denom
                                   end do
                                end do
                             end do
                          end do
                       end do
                    end do
                    do a = 1, nub
                       do b = a+1, nub
                          do c = b+1, nub
                             do i = 1, nob
                                do j = i+1, nob
                                   do k = j+1, nob
                                      denom = fB_oo(i,i) + fB_oo(j,j) + fB_oo(k,k) - fB_vv(a,a) - fB_vv(b,b) - fB_vv(c,c)
                                      t_amp = 0.0d0
                                      ! (i) hole diagram
                                      do m = 1, nob
                                         t_amp = t_amp - vC_vooo(a,m,i,j) * t2c_1(b,c,m,k)& ! (1)
                                                       + vC_vooo(a,m,k,j) * t2c_1(b,c,m,i)& ! (ik)
                                                       + vC_vooo(a,m,i,k) * t2c_1(b,c,m,j)& ! (jk)
                                                       + vC_vooo(c,m,i,j) * t2c_1(b,a,m,k)& ! (ac)
                                                       - vC_vooo(c,m,k,j) * t2c_1(b,a,m,i)& ! (ac)(ik)
                                                       - vC_vooo(c,m,i,k) * t2c_1(b,a,m,j)& ! (ac)(jk)
                                                       + vC_vooo(b,m,i,j) * t2c_1(a,c,m,k)& ! (ab)
                                                       - vC_vooo(b,m,k,j) * t2c_1(a,c,m,i)& ! (ab)(ik)
                                                       - vC_vooo(b,m,i,k) * t2c_1(a,c,m,j)  ! (ab)(jk)
                                      end do
                                      ! (ii) particle diagram
                                      do e = 1, nub
                                         t_amp = t_amp + vC_vvov(a,b,i,e) * t2c_1(e,c,j,k)& ! (1)
                                                       - vC_vvov(a,b,j,e) * t2c_1(e,c,i,k)& ! (ij)
                                                       - vC_vvov(a,b,k,e) * t2c_1(e,c,j,i)& ! (ik)
                                                       - vC_vvov(c,b,i,e) * t2c_1(e,a,j,k)& ! (ac)
                                                       + vC_vvov(c,b,j,e) * t2c_1(e,a,i,k)& ! (ac)(ij)
                                                       + vC_vvov(c,b,k,e) * t2c_1(e,a,j,i)& ! (ac)(ik)
                                                       - vC_vvov(a,c,i,e) * t2c_1(e,b,j,k)& ! (bc)
                                                       + vC_vvov(a,c,j,e) * t2c_1(e,b,i,k)& ! (bc)(ij)
                                                       + vC_vvov(a,c,k,e) * t2c_1(e,b,j,i)  ! (bc)(ik)
                                      end do
                                      eT = eT + t_amp * t_amp / denom
                                   end do
                                end do
                             end do
                          end do
                       end do
                    end do

                    ! compute E(Q) = <0|(T2[1]**2)^+ * R4 * (T2[1]**2)|0>, where T2[1] = V|0>)
                    ! This is accomplished using diagram factorization (adding of different tv1's)
                    ! check Bartlett & Shavitt book
                    eQ = 0.0d0


                    Emp4 = eS + eD + eT + eQ


              end subroutine mp4
              
end module mbpt_loops
