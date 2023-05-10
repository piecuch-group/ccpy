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

                    real(kind=8) :: denom
                    integer :: i, j, a, b

                    Emp2 = 0.0d0
                    do i = 1 , noa
                       do j = i+1 , noa
                          do a = 1 , nua
                             do b = a+1 , nua
                                denom = fA_oo(i,i)+fA_oo(j,j)-fA_vv(a,a)-fA_vv(b,b)
                                Emp2 = Emp2 + vA_oovv(i,j,a,b)*vA_vvoo(a,b,i,j)/denom
                             end do
                          end do
                       end do
                    end do
                    do i = 1 , noa
                       do j = 1 , nob
                          do a = 1 , nua
                             do b = 1 , nub
                                denom = fA_oo(i,i)+fB_oo(j,j)-fA_vv(a,a)-fB_vv(b,b)
                                Emp2 = Emp2 + vB_oovv(i,j,a,b)*vB_vvoo(a,b,i,j)/denom
                             end do
                          end do
                       end do
                    end do
                    do i = 1 , nob
                       do j = i+1 , nob
                          do a = 1 , nub
                             do b = a+1 , nub
                                denom = fB_oo(i,i)+fB_oo(j,j)-fB_vv(a,a)-fB_vv(b,b)
                                Emp2 = Emp2 + vC_oovv(i,j,a,b)*vC_vvoo(a,b,i,j)/denom
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

                    real(kind=8) :: d0, d1, d2, d3, d4, val,&
                                           t2a(nua,nua,noa,noa),&
                                           t2b(nua,nub,noa,nob),&
                                           t2c(nub,nub,nob,nob)
                    integer :: i, j, a, b, m, n, e, f

                    Emp3 = 0.0d0

                    t2a = 0.0d0
                    ! 2nd order estimate of T2A
                    ! t2A(2) = A(ij)A(ab)[ vA(amie)*vA(ebmj)/D(ebmj)
                    !                     +vB(amie)*vA(bejm)/D(bejm)
                    !                     +1/8*vA(mnij)*vA(abmn)/D(abmn)
                    !                     +1/8*vA(abef)*vA(efij)/D(efij) ]
                    do i = 1,noa
                       do j = i+1,noa
                          do a = 1,nua
                             do b = a+1,nua
                                d0 = fA_oo(i,i)+fA_oo(j,j)-fA_vv(a,a)-fA_vv(b,b)

                                val = 0.0d0
                                do m = 1,noa; do e = 1,nua;
                                   d1 = fA_oo(j,j)+fA_oo(m,m)-fA_vv(b,b)-fA_vv(e,e)
                                   d2 = fA_oo(i,i)+fA_oo(m,m)-fA_vv(b,b)-fA_vv(e,e)
                                   d3 = fA_oo(j,j)+fA_oo(m,m)-fA_vv(a,a)-fA_vv(e,e)
                                   d4 = fA_oo(i,i)+fA_oo(m,m)-fA_vv(a,a)-fA_vv(e,e)
                                   val = val + vA_voov(a,m,i,e)*vA_vvoo(e,b,m,j)/d1
                                   val = val - vA_voov(a,m,j,e)*vA_vvoo(e,b,m,i)/d2
                                   val = val - vA_voov(b,m,i,e)*vA_vvoo(e,a,m,j)/d3
                                   val = val + vA_voov(b,m,j,e)*vA_vvoo(e,a,m,i)/d4
                                end do; end do;
                                t2a(b,a,j,i) = t2a(b,a,j,i) + val
                                
                                val = 0.0d0
                                do m = 1,nob; do e = 1,nub;
                                   d1 = fA_oo(j,j)+fB_oo(m,m)-fA_vv(b,b)-fB_vv(e,e)
                                   d2 = fA_oo(i,i)+fB_oo(m,m)-fA_vv(b,b)-fB_vv(e,e)
                                   d3 = fA_oo(j,j)+fB_oo(m,m)-fA_vv(a,a)-fB_vv(e,e)
                                   d4 = fA_oo(i,i)+fB_oo(m,m)-fA_vv(a,a)-fB_vv(e,e)
                                   val = val + vB_voov(a,m,i,e)*vB_vvoo(b,e,j,m)/d1
                                   val = val - vB_voov(a,m,j,e)*vB_vvoo(b,e,i,m)/d2
                                   val = val - vB_voov(b,m,i,e)*vB_vvoo(a,e,j,m)/d3
                                   val = val + vB_voov(b,m,j,e)*vB_vvoo(a,e,i,m)/d4
                                end do; end do;
                                t2a(b,a,j,i) = t2a(b,a,j,i) + val

                                val = 0.0d0
                                do m = 1,noa; do n = m+1,noa;
                                   d1 = fA_oo(m,m)+fA_oo(n,n)-fA_vv(a,a)-fA_vv(b,b)
                                   val = val + vA_oooo(m,n,i,j)*vA_vvoo(a,b,m,n)/d1
                                end do; end do;
                                t2a(b,a,j,i) = t2a(b,a,j,i) + val

                                val = 0.0d0
                                do e = 1,nua; do f = e+1,nua;
                                   d1 = fA_oo(i,i)+fA_oo(j,j)-fA_vv(e,e)-fA_vv(f,f)
                                   val = val + vA_vvvv(a,b,e,f)*vA_vvoo(e,f,i,j)/d1
                                end do; end do;
                                t2a(b,a,j,i) = t2a(b,a,j,i) + val

                                Emp3 = Emp3 + vA_oovv(i,j,a,b)*t2a(b,a,j,i)/d0
                             end do
                          end do
                       end do
                    end do

                    t2b = 0.0d0                    
                    ! 2nd order estimate of T2B
                    ! t2B(2) = vA
                    do j = 1,nob
                       do i = 1,noa
                          do b = 1,nub
                             do a = 1,nua
                                d0 = fA_oo(i,i)+fB_oo(j,j)-fA_vv(a,a)-fB_vv(b,b)

                                val = 0.0d0
                                do m = 1,noa; do e = 1,nua;
                                   d1 = fB_oo(j,j)+fA_oo(m,m)-fB_vv(b,b)-fA_vv(e,e)
                                   val = val + vA_voov(a,m,i,e)*vB_vvoo(e,b,m,j)/d1
                                end do; end do;
                                t2b(a,b,i,j) = t2b(a,b,i,j) + val
                                
                                val = 0.0d0
                                do m = 1,nob; do e = 1,nub;
                                   d1 = fB_oo(j,j)+fB_oo(m,m)-fB_vv(b,b)-fB_vv(e,e)
                                   val = val + vB_voov(a,m,i,e)*vC_vvoo(b,e,j,m)/d1
                                end do; end do;
                                t2b(a,b,i,j) = t2b(a,b,i,j) + val

                                val = 0.0d0
                                do m = 1,noa; do e = 1,nua;
                                   d1 = fA_oo(i,i)+fA_oo(m,m)-fA_vv(a,a)-fA_vv(e,e)
                                   val = val + vB_ovvo(m,b,e,j)*vA_vvoo(a,e,i,m)/d1
                                end do; end do;
                                t2b(a,b,i,j) = t2b(a,b,i,j) + val

                                val = 0.0d0
                                do m = 1,nob; do e = 1,nub;
                                   d1 = fA_oo(i,i)+fB_oo(m,m)-fA_vv(a,a)-fB_vv(e,e)
                                   val = val + vC_voov(b,m,j,e)*vB_vvoo(a,e,i,m)/d1
                                end do; end do;
                                t2b(a,b,i,j) = t2b(a,b,i,j) + val

                                val = 0.0d0
                                do m = 1,noa; do e = 1,nub;
                                   d1 = fB_oo(j,j)+fA_oo(m,m)-fA_vv(a,a)-fB_vv(e,e)
                                   val = val - vB_ovov(m,b,i,e)*vB_vvoo(a,e,m,j)/d1
                                end do; end do;
                                t2b(a,b,i,j) = t2b(a,b,i,j) + val

                                val = 0.0d0
                                do m = 1,nob; do e = 1,nua;
                                   d1 = fA_oo(i,i)+fB_oo(m,m)-fB_vv(b,b)-fA_vv(e,e)
                                   val = val - vB_vovo(a,m,e,j)*vB_vvoo(e,b,i,m)/d1
                                end do; end do;
                                t2b(a,b,i,j) = t2b(a,b,i,j) + val

                                val = 0.0d0
                                do m = 1,noa; do n = 1,nob;
                                   d1 = fA_oo(m,m)+fB_oo(n,n)-fA_vv(a,a)-fB_vv(b,b)
                                   val = val + vB_oooo(m,n,i,j)*vB_vvoo(a,b,m,n)/d1
                                end do; end do;
                                t2b(a,b,i,j) = t2b(a,b,i,j) + val

                                val = 0.0d0
                                do e = 1,nua; do f = 1,nub;
                                   d1 = fA_oo(i,i)+fB_oo(j,j)-fA_vv(e,e)-fB_vv(f,f)
                                   val = val + vB_vvvv(a,b,e,f)*vB_vvoo(e,f,i,j)/d1
                                end do; end do;
                                t2b(a,b,i,j) = t2b(a,b,i,j) + val

                                Emp3 = Emp3 + vB_oovv(i,j,a,b)*t2b(a,b,i,j)/d0
                             end do
                          end do
                       end do
                    end do

                    t2c = 0.0d0
                    ! 2nd order estimate of T2C
                    do i = 1,nob
                       do j = i+1,nob
                          do a = 1,nub
                             do b = a+1,nub
                                d0 = fB_oo(i,i)+fB_oo(j,j)-fB_vv(a,a)-fB_vv(b,b)

                                val = 0.0d0
                                do m = 1,nob; do e = 1,nub;
                                   d1 = fB_oo(j,j)+fB_oo(m,m)-fB_vv(b,b)-fB_vv(e,e)
                                   d2 = fB_oo(i,i)+fB_oo(m,m)-fB_vv(b,b)-fB_vv(e,e)
                                   d3 = fB_oo(j,j)+fB_oo(m,m)-fB_vv(a,a)-fB_vv(e,e)
                                   d4 = fB_oo(i,i)+fB_oo(m,m)-fB_vv(a,a)-fB_vv(e,e)
                                   val = val + vC_voov(a,m,i,e)*vC_vvoo(e,b,m,j)/d1
                                   val = val - vC_voov(a,m,j,e)*vC_vvoo(e,b,m,i)/d2
                                   val = val - vC_voov(b,m,i,e)*vC_vvoo(e,a,m,j)/d3
                                   val = val + vC_voov(b,m,j,e)*vC_vvoo(e,a,m,i)/d4
                                end do; end do;
                                t2c(b,a,j,i) = t2c(b,a,j,i) + val
                                
                                val = 0.0d0
                                do m = 1,noa; do e = 1,nua;
                                   d1 = fB_oo(j,j)+fA_oo(m,m)-fB_vv(b,b)-fA_vv(e,e)
                                   d2 = fB_oo(i,i)+fA_oo(m,m)-fB_vv(b,b)-fA_vv(e,e)
                                   d3 = fB_oo(j,j)+fA_oo(m,m)-fB_vv(a,a)-fA_vv(e,e)
                                   d4 = fB_oo(i,i)+fA_oo(m,m)-fB_vv(a,a)-fA_vv(e,e)
                                   val = val + vB_ovvo(m,a,e,i)*vB_vvoo(e,b,m,j)/d1
                                   val = val - vB_ovvo(m,a,e,j)*vB_vvoo(e,b,m,i)/d2
                                   val = val - vB_ovvo(m,b,e,i)*vB_vvoo(e,a,m,j)/d3
                                   val = val + vB_ovvo(m,b,e,j)*vB_vvoo(e,a,m,i)/d4
                                end do; end do;
                                t2c(b,a,j,i) = t2c(b,a,j,i) + val

                                val = 0.0d0
                                do m = 1,nob; do n = m+1,nob;
                                   d1 = fB_oo(m,m)+fB_oo(n,n)-fB_vv(a,a)-fB_vv(b,b)
                                   val = val + vC_oooo(m,n,i,j)*vC_vvoo(a,b,m,n)/d1
                                end do; end do;
                                t2c(b,a,j,i) = t2c(b,a,j,i) + val

                                val = 0.0d0
                                do e = 1,nub; do f = e+1,nub;
                                   d1 = fB_oo(i,i)+fB_oo(j,j)-fB_vv(e,e)-fB_vv(f,f)
                                   val = val + vC_vvvv(a,b,e,f)*vC_vvoo(e,f,i,j)/d1
                                end do; end do;
                                t2c(b,a,j,i) = t2c(b,a,j,i) + val

                                Emp3 = Emp3 + vC_oovv(i,j,a,b)*t2c(b,a,j,i)/d0
                             end do
                          end do
                       end do
                    end do
              end subroutine mp3

end module mbpt_loops
