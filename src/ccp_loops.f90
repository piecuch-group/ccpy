module ccp_loops

      implicit none

      contains

              subroutine update_t3a(t2a,t3a,t3b,&
                         triples_list,&
                         vA_oovv,vB_oovv,&
                         H1A_oo,H1A_vv,H2A_oooo,&
                         H2A_vvvv,H2A_voov,H2B_voov,&
                         H2A_vooo,I2A_vvov,&
                         fA_oo,fA_vv,shift,&
                         noa,nua,nob,nub,num_triples,t3a_new)

                 integer, intent(in) :: noa, nua, nob, nub, num_triples
                 real(kind=8), intent(in) :: shift
                 integer, intent(in) :: triples_list(num_triples,6)
                 real(kind=8), intent(in) :: t2a(nua,nua,noa,noa),&
                                             t3a(nua,nua,nua,noa,noa,noa),&
                                             t3b(nua,nua,nub,noa,noa,nob),&
                                             vA_oovv(noa,noa,nua,nua),&
                                             vB_oovv(noa,nob,nua,nub),&
                                             H1A_oo(noa,noa),H1A_vv(nua,nua),&
                                             H2A_oooo(noa,noa,noa,noa),&
                                             H2A_vvvv(nua,nua,nua,nua),&
                                             H2A_voov(nua,noa,noa,nua),&
                                             H2B_voov(nua,nob,noa,nub),&
                                             H2A_vooo(nua,noa,noa,noa),&
                                             I2A_vvov(nua,nua,noa,nua),&
                                             fA_oo(noa,noa),fA_vv(nua,nua)

                real(kind=8), intent(out) :: t3a_new(nua,nua,nua,noa,noa,noa)

                integer :: a, b, c, i, j, k, ct,&
                           noa2, nua2, noaua, noanua2, nuanobnub,&
                           noanobnub, noa2nua, nobub
                real(kind=8) :: m1, m2, d1, d2, d3, d4, d5, d6,&
                                residual, denom, val, mval
                real(kind=8) :: vt3_oa(noa), vt3_ua(nua),&
                                H2A_voov_r(nua,nua,noa,noa),&
                                H2B_voov_r(nua,nub,noa,nob),&
                                vA_oovv_r1(nua,nua,noa,noa),&
                                vA_oovv_r2(nua,noa,noa,nua),&
                                vB_oovv_r1(nua,nub,nob,noa),&
                                vB_oovv_r2(nub,noa,nob,nua),&
                                temp1(noa), temp2(noa),&
                                temp3(nua), temp4(nua)


                real(kind=8), parameter :: MINUSONE=-1.0d+0, HALF=0.5d+0, ZERO=0.0d+0

                !integer :: m, n, e ,f
                !real(kind=8) :: error(8), refval

                noa2 = noa**2
                nua2 = nua**2
                noaua = noa*nua
                nobub = nob*nub
                noanua2 = noa*nua2
                nuanobnub = nua*nob*nub
                noa2nua = noa2*nua
                noanobnub = noa*nob*nub

                call reorder1432(H2A_voov,H2A_voov_r)
                call reorder1432(H2B_voov,H2B_voov_r)
                call reorder3421(vA_oovv,vA_oovv_r1)
                call reorder4123(vA_oovv,vA_oovv_r2)
                call reorder3421(vB_oovv,vB_oovv_r1)
                call reorder4123(vB_oovv,vB_oovv_r2)

                !do i = 1,8
                !    error(i) = ZERO
                !end do

                do ct = 1 , num_triples
            
                   ! Shift indices up by since the triples list is coming from
                   ! Python, where indices start from 0       
                   a = triples_list(ct,1)+1
                   b = triples_list(ct,2)+1
                   c = triples_list(ct,3)+1
                   i = triples_list(ct,4)+1
                   j = triples_list(ct,5)+1
                   k = triples_list(ct,6)+1
                    
                   ! Calculate devectorized residual for triple |ijkabc>
                   call dgemv(noanua2,noa,vA_oovv_r1,t3a(a,:,:,i,j,:),temp1)
                   call dgemv(nuanobnub,noa,vB_oovv_r1,t3b(a,:,:,i,j,:),temp2)
                   vt3_oa = HALF*temp1 + temp2
                   m1 = MINUSONE*ddot(noa,H2A_vooo(a,:,i,j)+vt3_oa,t2a(b,c,:,k))
                
                   call dgemv(noanua2,noa,vA_oovv_r1,t3a(c,:,:,i,j,:),temp1)
                   call dgemv(nuanobnub,noa,vB_oovv_r1,t3b(c,:,:,i,j,:),temp2)
                   vt3_oa = HALF*temp1 + temp2
                   m1 = m1 + ddot(noa,H2A_vooo(c,:,i,j)+vt3_oa,t2a(b,a,:,k))
                    
                   call dgemv(noanua2,noa,vA_oovv_r1,t3a(b,:,:,i,j,:),temp1)
                   call dgemv(nuanobnub,noa,vB_oovv_r1,t3b(b,:,:,i,j,:),temp2)
                   vt3_oa = HALF*temp1 + temp2
                   m1 = m1 + ddot(noa,H2A_vooo(b,:,i,j)+vt3_oa,t2a(a,c,:,k))

                   call dgemv(noanua2,noa,vA_oovv_r1,t3a(a,:,:,k,j,:),temp1)
                   call dgemv(nuanobnub,noa,vB_oovv_r1,t3b(a,:,:,k,j,:),temp2)
                   vt3_oa = HALF*temp1 + temp2
                   m1 = m1 + ddot(noa,H2A_vooo(a,:,k,j)+vt3_oa,t2a(b,c,:,i))

                   call dgemv(noanua2,noa,vA_oovv_r1,t3a(c,:,:,k,j,:),temp1)
                   call dgemv(nuanobnub,noa,vB_oovv_r1,t3b(c,:,:,k,j,:),temp2)
                   vt3_oa = HALF*temp1 + temp2
                   m1 = m1 - ddot(noa,H2A_vooo(c,:,k,j)+vt3_oa,t2a(b,a,:,i))

                   call dgemv(noanua2,noa,vA_oovv_r1,t3a(b,:,:,k,j,:),temp1)
                   call dgemv(nuanobnub,noa,vB_oovv_r1,t3b(b,:,:,k,j,:),temp2)
                   vt3_oa = HALF*temp1 + temp2
                   m1 = m1 - ddot(noa,H2A_vooo(b,:,k,j)+vt3_oa,t2a(a,c,:,i))

                   call dgemv(noanua2,noa,vA_oovv_r1,t3a(a,:,:,i,k,:),temp1)
                   call dgemv(nuanobnub,noa,vB_oovv_r1,t3b(a,:,:,i,k,:),temp2)
                   vt3_oa = HALF*temp1 + temp2
                   m1 = m1 + ddot(noa,H2A_vooo(a,:,i,k)+vt3_oa,t2a(b,c,:,j))

                   call dgemv(noanua2,noa,vA_oovv_r1,t3a(c,:,:,i,k,:),temp1)
                   call dgemv(nuanobnub,noa,vB_oovv_r1,t3b(c,:,:,i,k,:),temp2)
                   vt3_oa = HALF*temp1 + temp2
                   m1 = m1 - ddot(noa,H2A_vooo(c,:,i,k)+vt3_oa,t2a(b,a,:,j))

                   call dgemv(noanua2,noa,vA_oovv_r1,t3a(b,:,:,i,k,:),temp1)
                   call dgemv(nuanobnub,noa,vB_oovv_r1,t3b(b,:,:,i,k,:),temp2)
                   vt3_oa = HALF*temp1 + temp2
                   m1 = m1 - ddot(noa,H2A_vooo(b,:,i,k)+vt3_oa,t2a(a,c,:,j))

                   call dgemv(noa2nua,nua,vA_oovv_r2,t3a(a,b,:,i,:,:),temp3)
                   call dgemv(noanobnub,nua,vB_oovv_r2,t3b(a,b,:,i,:,:),temp4)
                   vt3_ua = HALF*temp3 + temp4
                   m2 = ddot(nua,I2A_vvov(a,b,i,:)-vt3_ua,t2a(:,c,j,k))

                   call dgemv(noa2nua,nua,vA_oovv_r2,t3a(a,b,:,j,:,:),temp3)
                   call dgemv(noanobnub,nua,vB_oovv_r2,t3b(a,b,:,j,:,:),temp4)
                   vt3_ua = HALF*temp3 + temp4
                   m2 = m2 - ddot(nua,I2A_vvov(a,b,j,:)-vt3_ua,t2a(:,c,i,k))

                   call dgemv(noa2nua,nua,vA_oovv_r2,t3a(a,b,:,k,:,:),temp3)
                   call dgemv(noanobnub,nua,vB_oovv_r2,t3b(a,b,:,k,:,:),temp4)
                   vt3_ua = HALF*temp3 + temp4
                   m2 = m2 - ddot(nua,I2A_vvov(a,b,k,:)-vt3_ua,t2a(:,c,j,i))

                   call dgemv(noa2nua,nua,vA_oovv_r2,t3a(c,b,:,i,:,:),temp3)
                   call dgemv(noanobnub,nua,vB_oovv_r2,t3b(c,b,:,i,:,:),temp4)
                   vt3_ua = HALF*temp3 + temp4
                   m2 = m2 - ddot(nua,I2A_vvov(c,b,i,:)-vt3_ua,t2a(:,a,j,k))
                    
                   call dgemv(noa2nua,nua,vA_oovv_r2,t3a(c,b,:,j,:,:),temp3)
                   call dgemv(noanobnub,nua,vB_oovv_r2,t3b(c,b,:,j,:,:),temp4)
                   vt3_ua = HALF*temp3 + temp4
                   m2 = m2 + ddot(nua,I2A_vvov(c,b,j,:)-vt3_ua,t2a(:,a,i,k))

                   call dgemv(noa2nua,nua,vA_oovv_r2,t3a(c,b,:,k,:,:),temp3)
                   call dgemv(noanobnub,nua,vB_oovv_r2,t3b(c,b,:,k,:,:),temp4)
                   vt3_ua = HALF*temp3 + temp4
                   m2 = m2 + ddot(nua,I2A_vvov(c,b,k,:)-vt3_ua,t2a(:,a,j,i))

                   call dgemv(noa2nua,nua,vA_oovv_r2,t3a(a,c,:,i,:,:),temp3)
                   call dgemv(noanobnub,nua,vB_oovv_r2,t3b(a,c,:,i,:,:),temp4)
                   vt3_ua = HALF*temp3 + temp4
                   m2 = m2 - ddot(nua,I2A_vvov(a,c,i,:)-vt3_ua,t2a(:,b,j,k))

                   call dgemv(noa2nua,nua,vA_oovv_r2,t3a(a,c,:,j,:,:),temp3)
                   call dgemv(noanobnub,nua,vB_oovv_r2,t3b(a,c,:,j,:,:),temp4)
                   vt3_ua = HALF*temp3 + temp4
                   m2 = m2 + ddot(nua,I2A_vvov(a,c,j,:)-vt3_ua,t2a(:,b,i,k))

                   call dgemv(noa2nua,nua,vA_oovv_r2,t3a(a,c,:,k,:,:),temp3)
                   call dgemv(noanobnub,nua,vB_oovv_r2,t3b(a,c,:,k,:,:),temp4)
                   vt3_ua = HALF*temp3 + temp4
                   m2 = m2 + ddot(nua,I2A_vvov(a,c,k,:)-vt3_ua,t2a(:,b,j,i))
                    
                   d1 = MINUSONE*ddot(noa,H1A_oo(:,k),t3a(a,b,c,i,j,:))
                   d1 = d1 + ddot(noa,H1A_oo(:,j),t3a(a,b,c,i,k,:))
                   d1 = d1 + ddot(noa,H1A_oo(:,i),t3a(a,b,c,k,j,:))

                   d2 = ddot(nua,H1A_vv(c,:),t3a(a,b,:,i,j,k))
                   d2 = d2 - ddot(nua,H1A_vv(b,:),t3a(a,c,:,i,j,k))
                   d2 = d2 - ddot(nua,H1A_vv(a,:),t3a(c,b,:,i,j,k))

                   d3 = ddot(noa2,H2A_oooo(:,:,i,j),t3a(a,b,c,:,:,k))
                   d3 = d3 - ddot(noa2,H2A_oooo(:,:,k,j),t3a(a,b,c,:,:,i))
                   d3 = d3 - ddot(noa2,H2A_oooo(:,:,i,k),t3a(a,b,c,:,:,j))
                   d3 = HALF*d3
        
                   d4 = ddot(nua2,H2A_vvvv(a,b,:,:),t3a(:,:,c,i,j,k))
                   d4 = d4 - ddot(nua2,H2A_vvvv(c,b,:,:),t3a(:,:,a,i,j,k))
                   d4 = d4 - ddot(nua2,H2A_vvvv(a,c,:,:),t3a(:,:,b,i,j,k))
                   d4 = HALF*d4

                   d5 = ddot(noaua,h2a_voov_r(c,:,k,:),t3a(a,b,:,i,j,:))
                   d5 = d5 - ddot(noaua,h2a_voov_r(c,:,i,:),t3a(a,b,:,k,j,:))
                   d5 = d5 - ddot(noaua,h2a_voov_r(c,:,j,:),t3a(a,b,:,i,k,:))
                   d5 = d5 - ddot(noaua,h2a_voov_r(a,:,k,:),t3a(c,b,:,i,j,:))
                   d5 = d5 + ddot(noaua,h2a_voov_r(a,:,i,:),t3a(c,b,:,k,j,:))
                   d5 = d5 + ddot(noaua,h2a_voov_r(a,:,j,:),t3a(c,b,:,i,k,:))
                   d5 = d5 - ddot(noaua,h2a_voov_r(b,:,k,:),t3a(a,c,:,i,j,:))
                   d5 = d5 + ddot(noaua,h2a_voov_r(b,:,i,:),t3a(a,c,:,k,j,:))
                   d5 = d5 + ddot(noaua,h2a_voov_r(b,:,j,:),t3a(a,c,:,i,k,:))

                   d6 = ddot(nobub,h2b_voov_r(c,:,k,:),t3b(a,b,:,i,j,:))
                   d6 = d6 - ddot(nobub,h2b_voov_r(c,:,i,:),t3b(a,b,:,k,j,:))
                   d6 = d6 - ddot(nobub,h2b_voov_r(c,:,j,:),t3b(a,b,:,i,k,:))
                   d6 = d6 - ddot(nobub,h2b_voov_r(a,:,k,:),t3b(c,b,:,i,j,:))
                   d6 = d6 + ddot(nobub,h2b_voov_r(a,:,i,:),t3b(c,b,:,k,j,:))
                   d6 = d6 + ddot(nobub,h2b_voov_r(a,:,j,:),t3b(c,b,:,i,k,:))
                   d6 = d6 - ddot(nobub,h2b_voov_r(b,:,k,:),t3b(a,c,:,i,j,:))
                   d6 = d6 + ddot(nobub,h2b_voov_r(b,:,i,:),t3b(a,c,:,k,j,:))
                   d6 = d6 + ddot(nobub,h2b_voov_r(b,:,j,:),t3b(a,c,:,i,k,:))
                
                   residual = m1 + m2 + d1 + d2 + d3 + d4 + d5 + d6
                   denom = fA_oo(i,i)+fA_oo(j,j)+fA_oo(k,k)&
                           -fA_vv(a,a)-fA_vv(b,b)-fA_vv(c,c)
                   val = t3a(a,b,c,i,j,k) + residual/(denom-shift)
                   mval = MINUSONE*val

                   t3a_new(a,b,c,i,j,k) = val
                   t3a_new(A,B,C,K,I,J) = val
                   t3a_new(A,B,C,J,K,I) = val
                   t3a_new(A,B,C,I,K,J) = mval
                   t3a_new(A,B,C,J,I,K) = mval
                   t3a_new(A,B,C,K,J,I) = mval
                                      
                   t3a_new(B,A,C,I,J,K) = mval
                   t3a_new(B,A,C,K,I,J) = mval
                   t3a_new(B,A,C,J,K,I) = mval
                   t3a_new(B,A,C,I,K,J) = val
                   t3a_new(B,A,C,J,I,K) = val
                   t3a_new(B,A,C,K,J,I) = val
                                      
                   t3a_new(A,C,B,I,J,K) = mval
                   t3a_new(A,C,B,K,I,J) = mval
                   t3a_new(A,C,B,J,K,I) = mval
                   t3a_new(A,C,B,I,K,J) = val
                   t3a_new(A,C,B,J,I,K) = val
                   t3a_new(A,C,B,K,J,I) = val
                                      
                   t3a_new(C,B,A,I,J,K) = mval
                   t3a_new(C,B,A,K,I,J) = mval
                   t3a_new(C,B,A,J,K,I) = mval
                   t3a_new(C,B,A,I,K,J) = val
                   t3a_new(C,B,A,J,I,K) = val
                   t3a_new(C,B,A,K,J,I) = val
                                      
                   t3a_new(B,C,A,I,J,K) = val
                   t3a_new(B,C,A,K,I,J) = val
                   t3a_new(B,C,A,J,K,I) = val
                   t3a_new(B,C,A,I,K,J) = mval
                   t3a_new(B,C,A,J,I,K) = mval
                   t3a_new(B,C,A,K,J,I) = mval
                                      
                   t3a_new(C,A,B,I,J,K) = val
                   t3a_new(C,A,B,K,I,J) = val
                   t3a_new(C,A,B,J,K,I) = val
                   t3a_new(C,A,B,I,K,J) = mval
                   t3a_new(C,A,B,J,I,K) = mval
                   t3a_new(C,A,B,K,J,I) = mval

                end do

                !do i = 1,8
                !   print*,'Error in term',i,'=',error(i)
                !end do

            end subroutine update_t3a

            subroutine update_t3b(t2a,t2b,&
                         t3a,t3b,t3c,&
                         triples_list,&
                         vA_oovv,vB_oovv,vC_oovv,&
                         H1A_oo,H1A_vv,H1B_oo,H1B_vv,&
                         H2A_oooo,H2A_vvvv,H2A_voov,&
                         H2B_oooo,H2B_vvvv,H2B_voov,&
                         H2B_ovov,H2B_vovo,H2B_ovvo,&
                         H2C_voov,&
                         I2A_vooo,H2A_vvov,&
                         I2B_vooo,I2B_ovoo,H2B_vvov,H2B_vvvo,&
                         fA_oo,fA_vv,fB_oo,fB_vv,shift,&
                         noa,nua,nob,nub,num_triples,t3b_new)

                integer, intent(in) :: noa, nua, nob, nub, num_triples
                real(kind=8), intent(in) :: shift
                integer, intent(in) :: triples_list(num_triples,6)
                real(kind=8), intent(in) ::  t2a(nua,nua,noa,noa),&
                                             t2b(nua,nub,noa,nob),&
                                             t3a(nua,nua,nua,noa,noa,noa),&
                                             t3b(nua,nua,nub,noa,noa,nob),&
                                             t3c(nua,nub,nub,noa,nob,nob),&
                                             vA_oovv(noa,noa,nua,nua),&
                                             vB_oovv(noa,nob,nua,nub),&
                                             vC_oovv(nob,nob,nub,nub),&
                                             H1A_oo(noa,noa),H1A_vv(nua,nua),&
                                             H1B_oo(nob,nob),H1B_vv(nub,nub),&
                                             H2A_oooo(noa,noa,noa,noa),&
                                             H2A_vvvv(nua,nua,nua,nua),&
                                             H2A_voov(nua,noa,noa,nua),&
                                             H2B_oooo(noa,nob,noa,nob),&
                                             H2B_vvvv(nua,nub,nua,nub),&
                                             H2B_voov(nua,nob,noa,nub),&
                                             H2B_ovov(noa,nub,noa,nub),&
                                             H2B_vovo(nua,nob,nua,nob),&
                                             H2B_ovvo(noa,nub,nua,nob),&
                                             H2C_voov(nub,nob,nob,nub),&
                                             I2A_vooo(nua,noa,noa,noa),&
                                             H2A_vvov(nua,nua,noa,nua),&
                                             I2B_vooo(nua,nob,noa,nob),&
                                             I2B_ovoo(noa,nub,noa,nob),&
                                             H2B_vvov(nua,nub,noa,nub),&
                                             H2B_vvvo(nua,nub,nua,nob),&
                                             fA_oo(noa,noa),fA_vv(nua,nua),&
                                             fB_oo(nob,nob),fB_vv(nub,nub)

                real(kind=8), intent(out) :: t3b_new(nua,nua,nub,noa,noa,nob)

                integer :: a, b, c, i, j, k, ct,&
                           Noo_aa, Noo_ab, Nvv_aa, Nvv_ab, Nov_aa, Nov_bb,&
                           Nov_ab, Nov_ba, Noov_aaa, Noov_abb, Novv_aaa,&
                           Novv_bba, Noov_baa, Noov_bbb, Novv_aab, Novv_bbb
                real(kind=8) :: m1, m2, m3, m4, m5, m6,&
                                d1, d2, d3, d4, d5, d6, d7,&
                                d8, d9, d10, d11, d12, d13, d14,&
                                residual, denom, val, mval
                real(kind=8) :: H2A_voov_r(nua,nua,noa,noa),& ! reorder 1432
                                H2B_voov_r(nua,nub,noa,nob),& ! reorder 1432
                                H2B_ovvo_r(nua,nub,noa,nob),& ! reorder 3214
                                H2C_voov_r(nub,nub,nob,nob),& ! reorder 1432
                                H2B_vovo_r(nua,nua,nob,nob),& ! reorder 1324
                                H2B_ovov_r(nub,nub,noa,noa),& ! reorder 4231
                                vA_oovv_r1(nua,noa,noa,nua),& ! reorder 4123
                                vB_oovv_r1(nub,noa,nob,nua),& ! reorder 4123
                                vA_oovv_r2(nua,nua,noa,noa),& ! reorder 3421
                                vB_oovv_r2(nua,nub,nob,noa),& ! reorder 3421
                                vB_oovv_r3(nua,noa,nob,nub),& ! reorder 3124
                                vC_oovv_r1(nub,nob,nob,nub),& ! reorder 3124
                                vB_oovv_r4(nua,nub,noa,nob),& ! reorder 3412
                                vC_oovv_r2(nub,nub,nob,nob),& ! reorder 4321
                                temp1a(nua),temp1b(nua),&
                                temp2a(noa),temp2b(noa),&
                                temp3a(nub),temp3b(nub),&
                                temp4a(nob),temp4b(nob),&
                                vt3_oa(noa),vt3_ua(nua),vt3_ob(nob),vt3_ub(nub)


                real(kind=8), parameter :: MINUSONE=-1.0d+0, HALF=0.5d+0, ZERO=0.0d+0,&
                                           MINUSHALF=-0.5d+0

                !integer :: m, n, e, f
                !real(kind=8) :: error(20), refval

                Noo_aa = noa*noa
                Noo_ab = noa*nob
                Nvv_aa = nua*nua
                Nvv_ab = nua*nub
                Nov_aa = noa*nua
                Nov_bb = nob*nub
                Nov_ab = noa*nub
                Nov_ba = nob*nua
                Noov_aaa = noa*noa*nua
                Noov_abb = noa*nob*nub
                Novv_aaa = noa*nua*nua
                Novv_bba = nob*nub*nua 
                Noov_baa = nob*noa*nua
                Noov_bbb = nob*nob*nub
                Novv_aab = noa*nua*nub
                Novv_bbb = nob*nub*nub

                call reorder1432(H2A_voov,H2A_voov_r)
                call reorder1432(H2B_voov,H2B_voov_r)
                call reorder3214(H2B_ovvo,H2B_ovvo_r)
                call reorder1432(H2C_voov,H2C_voov_r)
                call reorder1324(H2B_vovo,H2B_vovo_r)
                call reorder4231(H2B_ovov,H2B_ovov_r)
                call reorder4123(vA_oovv,vA_oovv_r1)
                call reorder4123(vB_oovv,vB_oovv_r1)
                call reorder3421(vA_oovv,vA_oovv_r2)
                call reorder3421(vB_oovv,vB_oovv_r2)
                call reorder3124(vB_oovv,vB_oovv_r3)
                call reorder3124(vC_oovv,vC_oovv_r1)
                call reorder3412(vB_oovv,vB_oovv_r4)
                call reorder4321(vC_oovv,vC_oovv_r2)

                !do i = 1,20
                !   error(i) = ZERO
                !end do

                do ct = 1 , num_triples
            
                   ! Shift indices up by since the triples list is coming from
                   ! Python, where indices start from 0       
                   a = triples_list(ct,1)+1
                   b = triples_list(ct,2)+1
                   c = triples_list(ct,3)+1
                   i = triples_list(ct,4)+1
                   j = triples_list(ct,5)+1
                   k = triples_list(ct,6)+1
                    
                   ! Calculate devectorized residual for triple |ijk~abc~>

                   call dgemv(Noov_aaa,nua,vA_oovv_r1,t3b(b,:,c,:,:,k),temp1a)
                   call dgemv(Noov_abb,nua,vB_oovv_r1,t3c(b,:,c,:,:,k),temp1b)
                   vt3_ua = HALF*temp1a + temp1b
                   m1 = ddot(nua,H2B_vvvo(b,c,:,k)-vt3_ua,t2a(a,:,i,j))
                    
                   call dgemv(Noov_aaa,nua,vA_oovv_r1,t3b(a,:,c,:,:,k),temp1a)
                   call dgemv(Noov_abb,nua,vB_oovv_r1,t3c(a,:,c,:,:,k),temp1b)
                   vt3_ua = HALF*temp1a + temp1b
                   m1 = m1 - ddot(nua,H2B_vvvo(a,c,:,k)-vt3_ua,t2a(b,:,i,j))

                   !refval = ZERO
                   !do e = 1,nua                    
                   !   do f = 1,nua
                   !      do m = 1,noa
                   !         do n = m+1,noa
                   !            refval = refval -&
                   !            vA_oovv(m,n,e,f)*t3b(b,f,c,m,n,k)*t2a(a,e,i,j)+&
                   !            vA_oovv(m,n,e,f)*t3b(a,f,c,m,n,k)*t2a(b,e,i,j)
                   !         end do
                   !      end do
                   !   end do
                   !   do f = 1,nub
                   !      do m = 1,noa
                   !         do n = 1,nob
                   !            refval = refval -&
                   !            vB_oovv(m,n,e,f)*t3c(b,f,c,m,n,k)*t2a(a,e,i,j)+&
                   !            vB_oovv(m,n,e,f)*t3c(a,f,c,m,n,k)*t2a(b,e,i,j)
                   !         end do
                   !      end do
                   !   end do
                   !   refval = refval + H2B_vvvo(b,c,e,k)*t2a(a,e,i,j)&
                   !                   - H2B_vvvo(a,c,e,k)*t2a(b,e,i,j)
                   !end do
                   !error(1) = error(1) + (m1 - refval)

                  call dgemv(Novv_aaa,noa,vA_oovv_r2,t3b(:,:,c,i,:,k),temp2a)
                  call dgemv(Novv_bba,noa,vB_oovv_r2,t3c(:,:,c,i,:,k),temp2b)
                  vt3_oa = HALF*temp2a + temp2b
                  m2 = MINUSONE*ddot(noa,I2B_ovoo(:,c,i,k)+vt3_oa,t2a(a,b,:,j))

                  call dgemv(Novv_aaa,noa,vA_oovv_r2,t3b(:,:,c,j,:,k),temp2a)
                  call dgemv(Novv_bba,noa,vB_oovv_r2,t3c(:,:,c,j,:,k),temp2b)
                  vt3_oa = HALF*temp2a + temp2b
                  m2 = m2 + ddot(noa,I2B_ovoo(:,c,j,k)+vt3_oa,t2a(a,b,:,i))

                  !refval = ZERO
                  !do m = 1,noa
                  !   do f = 1,nua
                  !      do e = f+1,nua
                  !         do n = 1,noa
                  !            refval = refval -&
                  !            vA_oovv(m,n,e,f)*t3b(e,f,c,i,n,k)*t2a(a,b,m,j)+&
                  !            vA_oovv(m,n,e,f)*t3b(e,f,c,j,n,k)*t2a(a,b,m,i)
                  !         end do
                  !      end do
                  !  end do
                  !  do f = 1,nub
                  !     do e = 1,nua
                  !        do n = 1,nob
                  !           refval = refval -&
                  !           vB_oovv(m,n,e,f)*t3c(e,f,c,i,n,k)*t2a(a,b,m,j)+&
                  !           vB_oovv(m,n,e,f)*t3c(e,f,c,j,n,k)*t2a(a,b,m,i)
                  !        end do
                  !      end do
                  !  end do
                  !  refval = refval - I2B_ovoo(m,c,i,k)*t2a(a,b,m,j)&
                  !                  + I2B_ovoo(m,c,j,k)*t2a(a,b,m,i)
                  !end do
                  !error(2) = error(2) + (m2-refval)

                  call dgemv(Noov_baa,nub,vB_oovv_r3,t3b(a,:,c,i,:,:),temp3a)
                  call dgemv(Noov_bbb,nub,vC_oovv_r1,t3c(a,:,c,i,:,:),temp3b)
                  vt3_ub = temp3a + HALF*temp3b
                  m3 = ddot(nub,H2B_vvov(a,c,i,:)-vt3_ub,t2b(b,:,j,k))
                  
                  call dgemv(Noov_baa,nub,vB_oovv_r3,t3b(b,:,c,i,:,:),temp3a)
                  call dgemv(Noov_bbb,nub,vC_oovv_r1,t3c(b,:,c,i,:,:),temp3b)
                  vt3_ub = temp3a + HALF*temp3b
                  m3 = m3 - ddot(nub,H2B_vvov(b,c,i,:)-vt3_ub,t2b(a,:,j,k))

                  call dgemv(Noov_baa,nub,vB_oovv_r3,t3b(a,:,c,j,:,:),temp3a)
                  call dgemv(Noov_bbb,nub,vC_oovv_r1,t3c(a,:,c,j,:,:),temp3b)
                  vt3_ub = temp3a + HALF*temp3b
                  m3 = m3 - ddot(nub,H2B_vvov(a,c,j,:)-vt3_ub,t2b(b,:,i,k))

                  call dgemv(Noov_baa,nub,vB_oovv_r3,t3b(b,:,c,j,:,:),temp3a)
                  call dgemv(Noov_bbb,nub,vC_oovv_r1,t3c(b,:,c,j,:,:),temp3b)
                  vt3_ub = temp3a + HALF*temp3b
                  m3 = m3 + ddot(nub,H2B_vvov(b,c,j,:)-vt3_ub,t2b(a,:,i,k))

                  !refval = ZERO
                  !do e = 1,nub
                  !   do f = 1,nua
                  !      do m = 1,nob
                  !         do n = 1,noa
                  !            refval = refval -&
                  !            vB_oovv(n,m,f,e)*t3b(a,f,c,i,n,m)*t2b(b,e,j,k)+&
                  !            vB_oovv(n,m,f,e)*t3b(b,f,c,i,n,m)*t2b(a,e,j,k)+&
                  !            vB_oovv(n,m,f,e)*t3b(a,f,c,j,n,m)*t2b(b,e,i,k)-&
                  !            vB_oovv(n,m,f,e)*t3b(b,f,c,j,n,m)*t2b(a,e,i,k)
                  !         end do
                  !      end do
                  !   end do
                  !   do f = 1,nub
                  !      do m = 1,nob
                  !         do n = m+1,nob
                  !            refval = refval -&
                  !            vC_oovv(n,m,f,e)*t3c(a,f,c,i,n,m)*t2b(b,e,j,k)+&
                  !            vC_oovv(n,m,f,e)*t3c(b,f,c,i,n,m)*t2b(a,e,j,k)+&
                  !            vC_oovv(n,m,f,e)*t3c(a,f,c,j,n,m)*t2b(b,e,i,k)-&
                  !            vC_oovv(n,m,f,e)*t3c(b,f,c,j,n,m)*t2b(a,e,i,k)
                  !         end do
                  !      end do
                  !   end do
                  !   refval = refval + H2B_vvov(a,c,i,e)*t2b(b,e,j,k)&
                  !                   - H2B_vvov(b,c,i,e)*t2b(a,e,j,k)&
                  !                   - H2B_vvov(a,c,j,e)*t2b(b,e,i,k)&
                  !                   + H2B_vvov(b,c,j,e)*t2b(a,e,i,k)
                  !end do
                  !error(3) = error(3) + (m3-refval)

                   call dgemv(Novv_aab,nob,vB_oovv_r4,t3b(b,:,:,j,:,k),temp4a)
                   call dgemv(Novv_bbb,nob,vC_oovv_r2,t3c(b,:,:,j,:,k),temp4b)
                   vt3_ob = temp4a + HALF*temp4b
                   m4 = MINUSONE*ddot(nob,I2B_vooo(b,:,j,k)+vt3_ob,t2b(a,c,i,:))

                   call dgemv(Novv_aab,nob,vB_oovv_r4,t3b(a,:,:,j,:,k),temp4a)
                   call dgemv(Novv_bbb,nob,vC_oovv_r2,t3c(a,:,:,j,:,k),temp4b)
                   vt3_ob = temp4a + HALF*temp4b
                   m4 = m4 + ddot(nob,I2B_vooo(a,:,j,k)+vt3_ob,t2b(b,c,i,:))

                   call dgemv(Novv_aab,nob,vB_oovv_r4,t3b(b,:,:,i,:,k),temp4a)
                   call dgemv(Novv_bbb,nob,vC_oovv_r2,t3c(b,:,:,i,:,k),temp4b)
                   vt3_ob = temp4a + HALF*temp4b
                   m4 = m4 + ddot(nob,I2B_vooo(b,:,i,k)+vt3_ob,t2b(a,c,j,:))

                   call dgemv(Novv_aab,nob,vB_oovv_r4,t3b(a,:,:,i,:,k),temp4a)
                   call dgemv(Novv_bbb,nob,vC_oovv_r2,t3c(a,:,:,i,:,k),temp4b)
                   vt3_ob = temp4a + HALF*temp4b
                   m4 = m4 - ddot(nob,I2B_vooo(a,:,i,k)+vt3_ob,t2b(b,c,j,:))

                   !refval = ZERO
                   !do m = 1,nob
                   !   do f = 1,nua
                   !      do e = 1,nub
                   !         do n = 1,noa
                   !            refval = refval -&
                   !            vB_oovv(n,m,f,e)*t3b(b,f,e,j,n,k)*t2b(a,c,i,m)+&
                   !            vB_oovv(n,m,f,e)*t3b(a,f,e,j,n,k)*t2b(b,c,i,m)+&
                   !            vB_oovv(n,m,f,e)*t3b(b,f,e,i,n,k)*t2b(a,c,j,m)-&
                   !            vB_oovv(n,m,f,e)*t3b(a,f,e,i,n,k)*t2b(b,c,j,m)
                   !         end do
                   !      end do
                   !   end do
                   !   do f = 1,nub
                   !      do e = f+1,nub
                   !         do n = 1,nob
                   !            refval = refval -&
                   !            vC_oovv(m,n,e,f)*t3c(b,f,e,j,n,k)*t2b(a,c,i,m)+&
                   !            vC_oovv(m,n,e,f)*t3c(a,f,e,j,n,k)*t2b(b,c,i,m)+&
                   !            vC_oovv(m,n,e,f)*t3c(b,f,e,i,n,k)*t2b(a,c,j,m)-&
                   !            vC_oovv(m,n,e,f)*t3c(a,f,e,i,n,k)*t2b(b,c,j,m)
                   !         end do
                   !      end do
                   !   end do
                   !   refval = refval - I2B_vooo(b,m,j,k)*t2b(a,c,i,m)&
                   !                   + I2B_vooo(a,m,j,k)*t2b(b,c,i,m)&
                   !                   + I2B_vooo(b,m,i,k)*t2b(a,c,j,m)&
                   !                   - I2B_vooo(a,m,i,k)*t2b(b,c,j,m)
                   !end do
                   !error(4) = error(4) + (m4-refval)

                   call dgemv(Noov_aaa,nua,vA_oovv_r1,t3a(a,b,:,i,:,:),temp1a)
                   call dgemv(Noov_abb,nua,vB_oovv_r1,t3b(a,b,:,i,:,:),temp1b)
                   vt3_ua = HALF*temp1a + temp1b
                   m5 = ddot(nua,H2A_vvov(a,b,i,:)-vt3_ua,t2b(:,c,j,k))

                   call dgemv(Noov_aaa,nua,vA_oovv_r1,t3a(a,b,:,j,:,:),temp1a)
                   call dgemv(Noov_abb,nua,vB_oovv_r1,t3b(a,b,:,j,:,:),temp1b)
                   vt3_ua = HALF*temp1a + temp1b
                   m5 = m5 - ddot(nua,H2A_vvov(a,b,j,:)-vt3_ua,t2b(:,c,i,k))

                   !refval = ZERO
                   !do e = 1,nua
                   !   do f = 1,nua
                   !      do m = 1,noa
                   !         do n = m+1,noa
                   !            refval = refval -&
                   !            vA_oovv(m,n,e,f)*t3a(a,b,f,i,m,n)*t2b(e,c,j,k)+&
                   !            vA_oovv(m,n,e,f)*t3a(a,b,f,j,m,n)*t2b(e,c,i,k)
                   !         end do
                   !      end do
                   !   end do
                   !   do f = 1,nub
                   !      do m = 1,noa
                   !         do n = 1,nob
                   !            refval = refval -&
                   !            vB_oovv(m,n,e,f)*t3b(a,b,f,i,m,n)*t2b(e,c,j,k)+&
                   !            vB_oovv(m,n,e,f)*t3b(a,b,f,j,m,n)*t2b(e,c,i,k)
                   !         end do
                   !      end do
                   !   end do
                   !   refval = refval + H2A_vvov(a,b,i,e)*t2b(e,c,j,k)&
                   !                   - H2A_vvov(a,b,j,e)*t2b(e,c,i,k)
                   ! end do
                   ! error(5) = error(5) + (m5-refval)

                   call dgemv(Novv_aaa,noa,vA_oovv_r2,t3a(a,:,:,i,j,:),temp2a)
                   call dgemv(Novv_bba,noa,vB_oovv_r2,t3b(a,:,:,i,j,:),temp2b)
                   vt3_oa = HALF*temp2a + temp2b
                   m6 = MINUSONE*ddot(noa,I2A_vooo(a,:,i,j)+vt3_oa,t2b(b,c,:,k))

                   call dgemv(Novv_aaa,noa,vA_oovv_r2,t3a(b,:,:,i,j,:),temp2a)
                   call dgemv(Novv_bba,noa,vB_oovv_r2,t3b(b,:,:,i,j,:),temp2b)
                   vt3_oa = HALF*temp2a + temp2b
                   m6 = m6 + ddot(noa,I2A_vooo(b,:,i,j)+vt3_oa,t2b(a,c,:,k))

                   !refval = ZERO
                   !do m = 1,noa
                   !   do e = 1,nua
                   !      do f = e+1,nua
                   !         do n = 1,noa
                   !            refval = refval -&
                   !            vA_oovv(m,n,e,f)*t3a(a,e,f,i,j,n)*t2b(b,c,m,k)+&
                   !            vA_oovv(m,n,e,f)*t3a(b,e,f,i,j,n)*t2b(a,c,m,k)
                   !         end do
                   !      end do
                   !   end do
                   !   do e = 1,nua
                   !      do f = 1,nub
                   !         do n = 1,nob
                   !            refval = refval -&
                   !            vB_oovv(m,n,e,f)*t3b(a,e,f,i,j,n)*t2b(b,c,m,k)+&
                   !            vB_oovv(m,n,e,f)*t3b(b,e,f,i,j,n)*t2b(a,c,m,k)
                   !         end do
                   !      end do
                   !   end do
                   !   refval = refval - I2A_vooo(a,m,i,j)*t2b(b,c,m,k)&
                   !                   + I2A_vooo(b,m,i,j)*t2b(a,c,m,k)
                   ! end do
                   ! error(6) = error(6) + (m6-refval)
                   
                   d1 = MINUSONE*ddot(noa,H1A_oo(:,i),t3b(a,b,c,:,j,k))
                   d1 = d1 + ddot(noa,H1A_oo(:,j),t3b(a,b,c,:,i,k))
                   !refval = ZERO
                   !do m = 1,noa
                   !   refval = refval - H1A_oo(m,i)*t3b(a,b,c,m,j,k)&
                   !                   + H1A_oo(m,j)*t3b(a,b,c,m,i,k)
                   !end do
                   !error(7) = error(7) + (d1-refval)
    
                   d2 = MINUSONE*ddot(nob,H1B_oo(:,k),t3b(a,b,c,i,j,:))
                   !refval = ZERO
                   !do m = 1,nob
                   !   refval = refval - H1B_oo(m,k)*t3b(a,b,c,i,j,m)
                   !end do
                   !error(8) = error(8) + (d2-refval)

                   d3 = ddot(nua,H1A_vv(a,:),t3b(:,b,c,i,j,k))
                   d3 = d3 - ddot(nua,H1A_vv(b,:),t3b(:,a,c,i,j,k))
                   !refval = ZERO
                   !do e = 1,nua
                   !   refval = refval + H1A_vv(a,e)*t3b(e,b,c,i,j,k)&
                   !                   - H1A_vv(b,e)*t3b(e,a,c,i,j,k)
                   !end do
                   !error(9) = error(9) + (d3-refval)

                   d4 = ddot(nub,H1B_vv(c,:),t3b(a,b,:,i,j,k))
                   !refval = ZERO
                   !do e = 1,nub
                   !   refval = refval + H1B_vv(c,e)*t3b(a,b,e,i,j,k)
                   !end do
                   !error(10) = error(10) + (d4-refval)
    
                   d5 = HALF*ddot(Noo_aa,H2A_oooo(:,:,i,j),t3b(a,b,c,:,:,k))
                   !refval = ZERO
                   !do m = 1,noa
                   !   do n = m+1,noa
                   !      refval = refval + H2A_oooo(m,n,i,j)*t3b(a,b,c,m,n,k)
                   !   end do
                   !end do
                   !error(11) = error(11) + (d5-refval)

                   d6 = ddot(Noo_ab,H2B_oooo(:,:,j,k),t3b(a,b,c,i,:,:))
                   d6 = d6 - ddot(Noo_ab,H2B_oooo(:,:,i,k),t3b(a,b,c,j,:,:))
                   !refval = ZERO
                   !do m = 1,noa
                   !   do n = 1,nob 
                   !      refval = refval + H2B_oooo(m,n,j,k)*t3b(a,b,c,i,m,n)&
                   !                      - H2B_oooo(m,n,i,k)*t3b(a,b,c,j,m,n)
                   !   end do
                   !end do
                   !error(12) = error(12) + (d6-refval)

                   d7 = HALF*ddot(Nvv_aa,H2A_vvvv(a,b,:,:),t3b(:,:,c,i,j,k))
                   !refval = ZERO
                   !do e = 1,nua
                   !   do f = e+1,nua
                   !      refval = refval + H2A_vvvv(a,b,e,f)*t3b(e,f,c,i,j,k)
                   !   end do
                   !end do
                   !error(13) = error(13) + (d7-refval)

                   d8 = ddot(Nvv_ab,H2B_vvvv(b,c,:,:),t3b(a,:,:,i,j,k))
                   d8 = d8 - ddot(Nvv_ab,H2B_vvvv(a,c,:,:),t3b(b,:,:,i,j,k))
                   !refval = ZERO
                   !do e = 1,nua
                   !   do f = 1,nub
                   !      refval = refval + H2B_vvvv(b,c,e,f)*t3b(a,e,f,i,j,k)&
                   !                      - H2B_vvvv(a,c,e,f)*t3b(b,e,f,i,j,k)
                   !   end do
                   !end do
                   !error(14) = error(14) + (d8-refval)

                   d9 = ddot(Nov_aa,H2A_voov_r(a,:,i,:),t3b(:,b,c,:,j,k))
                   d9 = d9 - ddot(Nov_aa,H2A_voov_r(b,:,i,:),t3b(:,a,c,:,j,k))
                   d9 = d9 - ddot(Nov_aa,H2A_voov_r(a,:,j,:),t3b(:,b,c,:,i,k))
                   d9 = d9 + ddot(Nov_aa,H2A_voov_r(b,:,j,:),t3b(:,a,c,:,i,k))
                   !refval = ZERO
                   !do m = 1,noa
                   !   do e = 1,nua
                   !      refval = refval + H2A_voov(a,m,i,e)*t3b(e,b,c,m,j,k)&
                   !                      - H2A_voov(b,m,i,e)*t3b(e,a,c,m,j,k)&
                   !                      - H2A_voov(a,m,j,e)*t3b(e,b,c,m,i,k)&
                   !                      + H2A_voov(b,m,j,e)*t3b(e,a,c,m,i,k)
                   !   end do
                   !end do
                   !error(15) = error(15) + (d9-refval)
    
                   d10 = ddot(Nov_bb,H2B_voov_r(a,:,i,:),t3c(b,:,c,j,:,k))
                   d10 = d10 - ddot(Nov_bb,H2B_voov_r(b,:,i,:),t3c(a,:,c,j,:,k))
                   d10 = d10 - ddot(Nov_bb,H2B_voov_r(a,:,j,:),t3c(b,:,c,i,:,k))
                   d10 = d10 + ddot(Nov_bb,H2B_voov_r(b,:,j,:),t3c(a,:,c,i,:,k))
                   !refval = ZERO
                   !do m = 1,nob
                   !   do e = 1,nub
                   !      refval = refval + H2B_voov(a,m,i,e)*t3c(b,e,c,j,m,k)&
                   !                      - H2B_voov(b,m,i,e)*t3c(a,e,c,j,m,k)&
                   !                      - H2B_voov(a,m,j,e)*t3c(b,e,c,i,m,k)&
                   !                      + H2B_voov(b,m,j,e)*t3c(a,e,c,i,m,k)
                   !   end do
                   !end do
                   !error(16) = error(16) + (d10-refval)

                   d11 = ddot(Nov_aa,H2B_ovvo_r(:,c,:,k),t3a(a,b,:,i,j,:))
                   !refval = ZERO
                   !do m = 1,noa
                   !   do e = 1,nua
                   !      refval = refval + H2B_ovvo(m,c,e,k)*t3a(a,b,e,i,j,m)
                   !   end do
                   !end do
                   !error(17) = error(17) + (d11-refval)

                   d12 = ddot(Nov_bb,H2C_voov_r(c,:,k,:),t3b(a,b,:,i,j,:))
                   !refval = ZERO
                   !do m = 1,nob
                   !   do e = 1,nub
                   !      refval = refval + H2C_voov(c,m,k,e)*t3b(a,b,e,i,j,m)
                   !   end do
                   !end do
                   !error(18) = error(18) + (d12-refval)

                   d13 = MINUSONE*ddot(Nov_ba,H2B_vovo_r(a,:,:,k),t3b(:,b,c,i,j,:))
                   d13 = d13 + ddot(Nov_ba,H2B_vovo_r(b,:,:,k),t3b(:,a,c,i,j,:))
                   !refval = ZERO
                   !do m = 1,nob
                   !   do e = 1,nua
                   !      refval = refval - H2B_vovo(a,m,e,k)*t3b(e,b,c,i,j,m)&
                   !                      + H2B_vovo(b,m,e,k)*t3b(e,a,c,i,j,m)
                   !   end do
                   !end do
                   !error(19) = error(19) + (d13-refval)
                   
                   d14 = MINUSONE*ddot(Nov_ab,H2B_ovov_r(:,c,i,:),t3b(a,b,:,:,j,k))
                   d14 = d14 + ddot(Nov_ab,H2B_ovov_r(:,c,j,:),t3b(a,b,:,:,i,k))
                   !refval = ZERO
                   !do m = 1,noa
                   !   do e = 1,nub
                   !      refval = refval - H2B_ovov(m,c,i,e)*t3b(a,b,e,m,j,k)&
                   !                      + H2B_ovov(m,c,j,e)*t3b(a,b,e,m,i,k)
                   !   end do
                   !end do
                   !error(20) = error(20) + (d14-refval)

                   residual = d1+d2+d3+d4+d5+d6+d7+d8+d9+d10+&
                   d11+d12+d13+d14+m1+m2+m3+m4+m5+m6
                   denom = fA_oo(i,i) + fA_oo(j,j) + fB_oo(k,k)&
                          -fA_vv(a,a) - fA_vv(b,b) - fB_vv(c,c)
                   val = t3b(a,b,c,i,j,k) + residual/denom
                   mval = MINUSONE*val

                   t3b_new(a,b,c,i,j,k) = val
                   t3b_new(b,a,c,i,j,k) = mval
                   t3b_new(a,b,c,j,i,k) = mval
                   t3b_new(b,a,c,j,i,k) = val

               end do

               !do i = 1,20
               !   print*,'Error in term',i,'=',error(i)
               !end do

            end subroutine update_t3b

            subroutine update_t3c(t2b,t2c,&
                         t3b,t3c,t3d,&
                         triples_list,&
                         vA_oovv,vB_oovv,vC_oovv,&
                         H1A_oo,H1A_vv,H1B_oo,H1B_vv,&
                         H2A_voov,&
                         H2B_oooo,H2B_vvvv,H2B_voov,&
                         H2B_ovov,H2B_vovo,H2B_ovvo,&
                         H2C_oooo,H2C_vvvv,H2C_voov,&
                         I2C_vooo,H2C_vvov,&
                         I2B_vooo,I2B_ovoo,H2B_vvov,H2B_vvvo,&
                         fA_oo,fA_vv,fB_oo,fB_vv,shift,&
                         noa,nua,nob,nub,num_triples,t3c_new)

                integer, intent(in) :: noa, nua, nob, nub, num_triples
                real(kind=8), intent(in) :: shift
                integer, intent(in) :: triples_list(num_triples,6)
                real(kind=8), intent(in) ::  t2b(nua,nub,noa,nob),&
                                             t2c(nub,nub,nob,nob),&
                                             t3b(nua,nua,nub,noa,noa,nob),&
                                             t3c(nua,nub,nub,noa,nob,nob),&
                                             t3d(nub,nub,nub,nob,nob,nob),&
                                             vA_oovv(noa,noa,nua,nua),&
                                             vB_oovv(noa,nob,nua,nub),&
                                             vC_oovv(nob,nob,nub,nub),&
                                             H1A_oo(noa,noa),H1A_vv(nua,nua),&
                                             H1B_oo(nob,nob),H1B_vv(nub,nub),&
                                             H2A_voov(nua,noa,noa,nua),&
                                             H2B_oooo(noa,nob,noa,nob),&
                                             H2B_vvvv(nua,nub,nua,nub),&
                                             H2B_voov(nua,nob,noa,nub),&
                                             H2B_ovov(noa,nub,noa,nub),&
                                             H2B_vovo(nua,nob,nua,nob),&
                                             H2B_ovvo(noa,nub,nua,nob),&
                                             H2C_oooo(nob,nob,nob,nob),&
                                             H2C_vvvv(nub,nub,nub,nub),&
                                             H2C_voov(nub,nob,nob,nub),&
                                             I2B_ovoo(noa,nub,noa,nob),&
                                             I2B_vooo(nua,nob,noa,nob),&
                                             I2C_vooo(nub,nob,nob,nob),&
                                             H2B_vvov(nua,nub,noa,nub),&
                                             H2C_vvov(nub,nub,nob,nub),&
                                             H2B_vvvo(nua,nub,nua,nob),&
                                             fA_oo(noa,noa),fA_vv(nua,nua),&
                                             fB_oo(nob,nob),fB_vv(nub,nub)

                real(kind=8), intent(out) :: t3c_new(nua,nub,nub,noa,nob,nob)

                integer :: a, b, c, i, j, k, ct,&
                           Noo_aa, Noo_bb, Nvv_aa, Nvv_bb,&
                           Nov_aa, Nov_bb, Nov_ab, Nov_ba,&
                           Noo_ab, Nvv_ab,&
                           Noov_baa, Noov_bbb, Novv_aab, Novv_bbb,&
                           Noov_aaa, Noov_abb, Novv_aaa, Novv_bba

                real(kind=8) :: m1, m2, m3, m4, m5, m6,&
                                d1, d2, d3, d4, d5, d6, d7,&
                                d8, d9, d10, d11, d12, d13, d14,&
                                residual, denom, val, mval

                real(kind=8) :: H2A_voov_r1(nua,nua,noa,noa),& ! reorder 1432
                                H2B_voov_r1(nua,nub,noa,nob),& ! reorder 1432
                                H2B_ovvo_r1(nua,nub,noa,nob),& ! reorder 3214
                                H2C_voov_r1(nub,nub,nob,nob),& ! reorder 1432
                                H2B_ovov_r1(nub,nub,noa,noa),& ! reorder 4231
                                H2B_vovo_r1(nua,nua,nob,nob),& ! reorder 1324
                                vB_oovv_r1(nua,noa,nob,nub),& ! reorder 3124
                                vC_oovv_r1(nub,nob,nob,nub),& ! reorder 3124
                                vB_oovv_r2(nua,nub,noa,nob),& ! reorder 3412
                                vC_oovv_r2(nub,nub,nob,nob),& ! reorder 3412
                                vC_oovv_r3(nub,nob,nob,nub),& ! reorder 4123
                                vC_oovv_r4(nub,nub,nob,noa),& ! reorder 3421
                                vA_oovv_r1(nua,noa,noa,nua),& ! reorder 4123
                                vB_oovv_r3(nub,noa,nob,nua),& ! reorder 4123
                                vA_oovv_r2(nua,nua,noa,noa),& ! reorder 3421
                                vB_oovv_r4(nua,nub,nob,noa),& ! reorder 3421
                                temp1a(nua),temp1b(nua),&
                                temp2a(noa),temp2b(noa),&
                                temp3a(nub),temp3b(nub),&
                                temp4a(nob),temp4b(nob),&
                                vt3_oa(noa),vt3_ua(nua),vt3_ob(nob),vt3_ub(nub)


                real(kind=8), parameter :: MINUSONE=-1.0d+0, HALF=0.5d+0, ZERO=0.0d+0,&
                                           MINUSHALF=-0.5d+0

                !integer :: m, n, e, f
                !real(kind=8) :: error(20), refval

                Noo_aa = noa*noa
                Noo_ab = noa*nob
                Nvv_aa = nua*nua
                Nvv_ab = nua*nub
                Noo_bb = nob*nob
                Nvv_bb = nub*nub
                Nov_aa = noa*nua
                Nov_bb = nob*nub
                Nov_ab = noa*nub
                Nov_ba = nob*nua
                Noov_baa = nob*noa*nua
                Noov_bbb = nob*nob*nub
                Novv_aab = noa*nua*nub
                Novv_bbb = nob*nub*nub
                Noov_aaa = noa*noa*nua
                Noov_abb = noa*nob*nub
                Novv_aaa = noa*nua*nua
                Novv_bba = nob*nub*nua

                call reorder1432(H2A_voov,H2A_voov_r1)
                call reorder1432(H2B_voov,H2B_voov_r1)
                call reorder3214(H2B_ovvo,H2B_ovvo_r1)
                call reorder1432(H2C_voov,H2C_voov_r1)
                call reorder4231(H2B_ovov,H2B_ovov_r1)
                call reorder1324(H2B_vovo,H2B_vovo_r1)
                call reorder3124(vB_oovv,vB_oovv_r1)
                call reorder3124(vC_oovv,vC_oovv_r1)
                call reorder3412(vB_oovv,vB_oovv_r2)
                call reorder3412(vC_oovv,vC_oovv_r2)
                call reorder4123(vC_oovv,vC_oovv_r3)
                call reorder3421(vC_oovv,vC_oovv_r4)
                call reorder4123(vA_oovv,vA_oovv_r1)
                call reorder4123(vB_oovv,vB_oovv_r3)
                call reorder3421(vA_oovv,vA_oovv_r2)
                call reorder3421(vB_oovv,vB_oovv_r4)

                !do i = 1,20
                !   error(i) = ZERO
                !end do

                do ct = 1 , num_triples
                   
                   ! Shift indices up by 1 since the triples list is coming from
                   ! Python, where indices start from 0       
                   a = triples_list(ct,1)+1
                   b = triples_list(ct,2)+1
                   c = triples_list(ct,3)+1
                   i = triples_list(ct,4)+1
                   j = triples_list(ct,5)+1
                   k = triples_list(ct,6)+1

                   ! calculate devectorized residual for triple |ij~k~ab~c~>
                   
                   call dgemv(Noov_baa,nub,vB_oovv_r1,t3b(a,:,b,i,:,:),temp3a)
                   call dgemv(Noov_bbb,nub,vC_oovv_r1,t3c(a,:,b,i,:,:),temp3b)
                   vt3_ub = temp3a + HALF*temp3b
                   m1 = ddot(nub,H2B_vvov(a,b,i,:)-vt3_ub,t2c(:,c,j,k))

                   call dgemv(Noov_baa,nub,vB_oovv_r1,t3b(a,:,c,i,:,:),temp3a)
                   call dgemv(Noov_bbb,nub,vC_oovv_r1,t3c(a,:,c,i,:,:),temp3b)
                   vt3_ub = temp3a + HALF*temp3b
                   m1 = m1 - ddot(nub,H2B_vvov(a,c,i,:)-vt3_ub,t2c(:,b,j,k))

                    !refval = ZERO
                    !do e = 1,nub
                    !  do f = 1,nua
                    !     do n = 1,noa
                    !        do m = 1,nob
                    !           refval = refval -&
                    !           vB_oovv(n,m,f,e)*t3b(a,f,b,i,n,m)*t2c(e,c,j,k)+&
                    !           vB_oovv(n,m,f,e)*t3b(a,f,c,i,n,m)*t2c(e,b,j,k)
                    !        end do
                    !     end do
                    !  end do
                    !  do f = 1,nub
                    !     do n = 1,nob
                    !        do m = n+1,nob
                    !           refval = refval -&
                    !           vC_oovv(n,m,f,e)*t3c(a,f,b,i,n,m)*t2c(e,c,j,k)+&
                    !           vC_oovv(n,m,f,e)*t3c(a,f,c,i,n,m)*t2c(e,b,j,k)
                    !        end do
                    !     end do
                    !  end do
                    !  refval = refval + H2B_vvov(a,b,i,e)*t2c(e,c,j,k)&
                    !                  - H2B_vvov(a,c,i,e)*t2c(e,b,j,k)
                    !end do
                    !error(1) = error(1) + (m1-refval)

                   call dgemv(Novv_aab,nob,vB_oovv_r2,t3b(a,:,:,i,:,j),temp4a)
                   call dgemv(Novv_bbb,nob,vC_oovv_r2,t3c(a,:,:,i,:,j),temp4b)
                   vt3_ob = temp4a + HALF*temp4b
                   m2 = MINUSONE*ddot(nob,I2B_vooo(a,:,i,j)+vt3_ob,t2c(b,c,:,k))

                   call dgemv(Novv_aab,nob,vB_oovv_r2,t3b(a,:,:,i,:,k),temp4a)
                   call dgemv(Novv_bbb,nob,vC_oovv_r2,t3c(a,:,:,i,:,k),temp4b)
                   vt3_ob = temp4a + HALF*temp4b
                   m2 = m2 + ddot(nob,I2B_vooo(a,:,i,k)+vt3_ob,t2c(b,c,:,j))

                   !refval = ZERO
                   !do m = 1,nob
                   !   do f = 1,nua
                   !      do e = 1,nub
                   !         do n = 1,noa
                   !            refval = refval -&
                   !            vB_oovv(n,m,f,e)*t3b(a,f,e,i,n,j)*t2c(b,c,m,k)+&
                   !            vB_oovv(n,m,f,e)*t3b(a,f,e,i,n,k)*t2c(b,c,m,j)
                   !         end do
                   !       end do
                   !    end do
                   !    do f = 1,nub
                   !       do e = f+1,nub
                   !          do n = 1,nob
                   !             refval = refval -&
                   !             vC_oovv(n,m,f,e)*t3c(a,f,e,i,n,j)*t2c(b,c,m,k)+&
                   !             vC_oovv(n,m,f,e)*t3c(a,f,e,i,n,k)*t2c(b,c,m,j)
                   !          end do
                   !       end do
                   !    end do
                   !    refval = refval - I2B_vooo(a,m,i,j)*t2c(b,c,m,k)&
                   !                    + I2B_vooo(a,m,i,k)*t2c(b,c,m,j)
                   ! end do
                   ! error(2) = error(2) + (m2-refval)


                   call dgemv(Noov_bbb,nub,vC_oovv_r3,t3d(c,b,:,k,:,:),temp3a)
                   call dgemv(Noov_baa,nub,vB_oovv_r1,t3c(:,c,b,:,k,:),temp3b)
                   vt3_ub = HALF*temp3a + temp3b
                   m3 = ddot(nub,H2C_vvov(c,b,k,:)-vt3_ub,t2b(a,:,i,j))

                   call dgemv(Noov_bbb,nub,vC_oovv_r3,t3d(c,b,:,j,:,:),temp3a)
                   call dgemv(Noov_baa,nub,vB_oovv_r1,t3c(:,c,b,:,j,:),temp3b)
                   vt3_ub = HALF*temp3a + temp3b
                   m3 = m3 - ddot(nub,H2C_vvov(c,b,j,:)-vt3_ub,t2b(a,:,i,k))

                   !refval = ZERO
                   !do e = 1,nub
                   !   do f = 1,nub
                   !      do m = 1,nob
                   !         do n = m+1,nob
                   !            refval = refval -&
                   !            vC_oovv(m,n,e,f)*t3d(c,b,f,k,m,n)*t2b(a,e,i,j)+&
                   !            vC_oovv(m,n,e,f)*t3d(c,b,f,j,m,n)*t2b(a,e,i,k)
                   !         end do
                   !       end do
                   !    end do
                   !    do f = 1,nua
                   !       do m = 1,nob
                   !          do n = 1,noa
                   !             refval = refval -&
                   !             vB_oovv(n,m,f,e)*t3c(f,c,b,n,k,m)*t2b(a,e,i,j)+&
                   !             vB_oovv(n,m,f,e)*t3c(f,c,b,n,j,m)*t2b(a,e,i,k)
                   !          end do
                   !       end do
                   !     end do
                   !     refval = refval + H2C_vvov(c,b,k,e)*t2b(a,e,i,j)&
                   !                     - H2C_vvov(c,b,j,e)*t2b(a,e,i,k)
                   !  end do
                   !  error(3) = error(3) + (m3-refval)

                   call dgemv(Novv_aab,nob,vB_oovv_r2,t3c(:,c,:,:,k,j),temp4a)
                   call dgemv(Novv_bbb,nob,vC_oovv_r4,t3d(c,:,:,k,j,:),temp4b)
                   vt3_ob = temp4a + HALF*temp4b
                   m4 = MINUSONE*ddot(nob,I2C_vooo(c,:,k,j)+vt3_ob,t2b(a,b,i,:))

                   call dgemv(Novv_aab,nob,vB_oovv_r2,t3c(:,b,:,:,k,j),temp4a)
                   call dgemv(Novv_bbb,nob,vC_oovv_r4,t3d(b,:,:,k,j,:),temp4b)
                   vt3_ob = temp4a + HALF*temp4b
                   m4 = m4 + ddot(nob,I2C_vooo(b,:,k,j)+vt3_ob,t2b(a,c,i,:))

                   !refval = ZERO
                   !do m = 1,nob
                   !    do f = 1,nua
                   !      do e = 1,nub
                   !         do n = 1,noa
                   !            refval = refval -&
                   !            vB_oovv(n,m,f,e)*t3c(f,c,e,n,k,j)*t2b(a,b,i,m)+&
                   !            vB_oovv(n,m,f,e)*t3c(f,b,e,n,k,j)*t2b(a,c,i,m)
                   !         end do
                   !      end do
                   !    end do
                   !    do f = 1,nub
                   !       do e = f+1,nub
                   !          do n = 1,nob
                   !             refval = refval -&
                   !             vC_oovv(m,n,e,f)*t3d(c,e,f,k,j,n)*t2b(a,b,i,m)+&
                   !             vC_oovv(m,n,e,f)*t3d(b,e,f,k,j,n)*t2b(a,c,i,m)
                   !          end do
                   !       end do
                   !    end do
                   !    refval = refval - I2C_vooo(c,m,k,j)*t2b(a,b,i,m)&
                   !                    + I2C_vooo(b,m,k,j)*t2b(a,c,i,m)
                   ! end do
                   ! error(4) = error(4) + (m4-refval)

                   call dgemv(Noov_aaa,nua,vA_oovv_r1,t3b(a,:,b,:,:,j),temp1a)
                   call dgemv(Noov_abb,nua,vB_oovv_r3,t3c(a,:,b,:,:,j),temp1b)
                   vt3_ua = HALF*temp1a + temp1b
                   m5 = ddot(nua,H2B_vvvo(a,b,:,j)-vt3_ua,t2b(:,c,i,k))

                   call dgemv(Noov_aaa,nua,vA_oovv_r1,t3b(a,:,c,:,:,j),temp1a)
                   call dgemv(Noov_abb,nua,vB_oovv_r3,t3c(a,:,c,:,:,j),temp1b)
                   vt3_ua = HALF*temp1a + temp1b
                   m5 = m5 - ddot(nua,H2B_vvvo(a,c,:,j)-vt3_ua,t2b(:,b,i,k))

                   call dgemv(Noov_aaa,nua,vA_oovv_r1,t3b(a,:,b,:,:,k),temp1a)
                   call dgemv(Noov_abb,nua,vB_oovv_r3,t3c(a,:,b,:,:,k),temp1b)
                   vt3_ua = HALF*temp1a + temp1b
                   m5 = m5 - ddot(nua,H2B_vvvo(a,b,:,k)-vt3_ua,t2b(:,c,i,j))

                   call dgemv(Noov_aaa,nua,vA_oovv_r1,t3b(a,:,c,:,:,k),temp1a)
                   call dgemv(Noov_abb,nua,vB_oovv_r3,t3c(a,:,c,:,:,k),temp1b)
                   vt3_ua = HALF*temp1a + temp1b
                   m5 = m5 + ddot(nua,H2B_vvvo(a,c,:,k)-vt3_ua,t2b(:,b,i,j))

                   !refval = ZERO
                   !do e = 1,nua
                   !   do f = 1,nua
                   !      do m = 1,noa
                   !         do n = m+1,noa
                   !            refval = refval -&
                   !            vA_oovv(m,n,e,f)*t3b(a,f,b,m,n,j)*t2b(e,c,i,k)+&
                   !            vA_oovv(m,n,e,f)*t3b(a,f,c,m,n,j)*t2b(e,b,i,k)+&
                   !            vA_oovv(m,n,e,f)*t3b(a,f,b,m,n,k)*t2b(e,c,i,j)-&
                   !            vA_oovv(m,n,e,f)*t3b(a,f,c,m,n,k)*t2b(e,b,i,j)
                   !         end do
                   !      end do
                   !    end do
                   !    do f = 1,nub
                   !       do m = 1,noa
                   !          do n = 1,nob
                   !             refval = refval -&
                   !             vB_oovv(m,n,e,f)*t3c(a,f,b,m,n,j)*t2b(e,c,i,k)+&
                   !             vB_oovv(m,n,e,f)*t3c(a,f,c,m,n,j)*t2b(e,b,i,k)+&
                   !             vB_oovv(m,n,e,f)*t3c(a,f,b,m,n,k)*t2b(e,c,i,j)-&
                   !             vB_oovv(m,n,e,f)*t3c(a,f,c,m,n,k)*t2b(e,b,i,j)
                   !          end do
                   !       end do
                   !     end do
                   !     refval = refval + H2B_vvvo(a,b,e,j)*t2b(e,c,i,k)&
                   !                     - H2B_vvvo(a,c,e,j)*t2b(e,b,i,k)&
                   !                     - H2B_vvvo(a,b,e,k)*t2b(e,c,i,j)&
                   !                     + H2B_vvvo(a,c,e,k)*t2b(e,b,i,j)
                   ! end do
                   ! error(5) = error(5) + (m5-refval)

                   call dgemv(Novv_aaa,noa,vA_oovv_r2,t3b(:,:,b,i,:,j),temp2a)
                   call dgemv(Novv_bba,noa,vB_oovv_r4,t3c(:,:,b,i,:,j),temp2b)
                   vt3_oa = HALF*temp2a + temp2b
                   m6 = MINUSONE*ddot(noa,I2B_ovoo(:,b,i,j)+vt3_oa,t2b(a,c,:,k))

                   call dgemv(Novv_aaa,noa,vA_oovv_r2,t3b(:,:,c,i,:,j),temp2a)
                   call dgemv(Novv_bba,noa,vB_oovv_r4,t3c(:,:,c,i,:,j),temp2b)
                   vt3_oa = HALF*temp2a + temp2b
                   m6 = m6 + ddot(noa,I2B_ovoo(:,c,i,j)+vt3_oa,t2b(a,b,:,k))

                   call dgemv(Novv_aaa,noa,vA_oovv_r2,t3b(:,:,b,i,:,k),temp2a)
                   call dgemv(Novv_bba,noa,vB_oovv_r4,t3c(:,:,b,i,:,k),temp2b)
                   vt3_oa = HALF*temp2a + temp2b
                   m6 = m6 + ddot(noa,I2B_ovoo(:,b,i,k)+vt3_oa,t2b(a,c,:,j))

                   call dgemv(Novv_aaa,noa,vA_oovv_r2,t3b(:,:,c,i,:,k),temp2a)
                   call dgemv(Novv_bba,noa,vB_oovv_r4,t3c(:,:,c,i,:,k),temp2b)
                   vt3_oa = HALF*temp2a + temp2b
                   m6 = m6 - ddot(noa,I2B_ovoo(:,c,i,k)+vt3_oa,t2b(a,b,:,j))

                   !refval = ZERO
                   !do m = 1,noa
                   !   do e = 1,nua
                   !      do f = e+1,nua
                   !         do n = 1,noa
                   !            refval = refval -&
                   !            vA_oovv(m,n,e,f)*t3b(e,f,b,i,n,j)*t2b(a,c,m,k)+&
                   !            vA_oovv(m,n,e,f)*t3b(e,f,c,i,n,j)*t2b(a,b,m,k)+&
                   !            vA_oovv(m,n,e,f)*t3b(e,f,b,i,n,k)*t2b(a,c,m,j)-&
                   !            vA_oovv(m,n,e,f)*t3b(e,f,c,i,n,k)*t2b(a,b,m,j)
                   !         end do
                   !      end do
                   !   end do
                   !   do e = 1,nua
                   !      do f = 1,nub
                   !         do n = 1,nob
                   !            refval = refval -&
                   !            vB_oovv(m,n,e,f)*t3c(e,f,b,i,n,j)*t2b(a,c,m,k)+&
                   !            vB_oovv(m,n,e,f)*t3c(e,f,c,i,n,j)*t2b(a,b,m,k)+&
                   !            vB_oovv(m,n,e,f)*t3c(e,f,b,i,n,k)*t2b(a,c,m,j)-&
                   !            vB_oovv(m,n,e,f)*t3c(e,f,c,i,n,k)*t2b(a,b,m,j)
                   !         end do
                   !      end do
                   !   end do
                   !   refval = refval - I2B_ovoo(m,b,i,j)*t2b(a,c,m,k)&
                   !                   + I2B_ovoo(m,c,i,j)*t2b(a,b,m,k)&
                   !                   + I2B_ovoo(m,b,i,k)*t2b(a,c,m,j)&
                   !                   - I2B_ovoo(m,c,i,k)*t2b(a,b,m,j)
                   ! end do
                   ! error(6) = error(6) + (m6-refval)

                   ! (HBar T3)_C
                   d1 = MINUSONE*ddot(noa,H1A_oo(:,i),t3c(a,b,c,:,j,k))
                   !refval = ZERO
                   !do m = 1,noa
                   !   refval = refval - H1A_oo(m,i)*t3c(a,b,c,m,j,k)
                   !end do
                   !error(7) = error(7) + (d1-refval)

                   d2 = MINUSONE*ddot(nob,H1B_oo(:,j),t3c(a,b,c,i,:,k))
                   d2 = d2 + ddot(nob,H1B_oo(:,k),t3c(a,b,c,i,:,j))
                   !refval = ZERO
                   !do m = 1,nob
                   !   refval = refval - H1B_oo(m,j)*t3c(a,b,c,i,m,k)&
                   !                   + H1B_oo(m,k)*t3c(a,b,c,i,m,j)
                   !end do
                   !error(8) = error(8) + (d2-refval)

                   d3 = ddot(nua,H1A_vv(a,:),t3c(:,b,c,i,j,k))
                   !refval = ZERO
                   !do e = 1,nua
                   !   refval = refval + H1A_vv(a,e)*t3c(e,b,c,i,j,k)
                   !end do
                   !error(9) = error(9) + (d3-refval)

                   d4 = ddot(nub,H1B_vv(b,:),t3c(a,:,c,i,j,k))
                   d4 = d4 - ddot(nub,H1B_vv(c,:),t3c(a,:,b,i,j,k))
                   !refval = ZERO
                   !do e = 1,nub
                   !   refval = refval + H1B_vv(b,e)*t3c(a,e,c,i,j,k)&
                   !                   - H1B_vv(c,e)*t3c(a,e,b,i,j,k)
                   !end do
                   !error(10) = error(10) + (d4-refval)

                   d5 = HALF*ddot(Noo_bb,H2C_oooo(:,:,j,k),t3c(a,b,c,i,:,:))
                   !refval = ZERO
                   !do m = 1,nob
                   !   do n = m+1,nob
                   !      refval = refval + H2C_oooo(m,n,j,k)*t3c(a,b,c,i,m,n)
                   !   end do
                   !end do
                   !error(11) = error(11) + (d5-refval)

                   d6 = ddot(Noo_ab,H2B_oooo(:,:,i,k),t3c(a,b,c,:,j,:))
                   d6 = d6 - ddot(Noo_ab,H2B_oooo(:,:,i,j),t3c(a,b,c,:,k,:))
                   !refval = ZERO
                   !do m = 1,noa
                   !   do n = 1,nob
                   !      refval = refval + H2B_oooo(m,n,i,k)*t3c(a,b,c,m,j,n)&
                   !                      - H2B_oooo(m,n,i,j)*t3c(a,b,c,m,k,n)
                   !   end do
                   !end do
                   !error(12) = error(12) + (d6-refval)

                   d7 = HALF*ddot(Nvv_bb,H2C_vvvv(b,c,:,:),t3c(a,:,:,i,j,k))
                   !refval = ZERO
                   !do e = 1,nub
                   !   do f = e+1,nub
                   !      refval = refval + H2C_vvvv(b,c,e,f)*t3c(a,e,f,i,j,k)
                   !   end do
                   !end do
                   !error(13) = error(13) + (d7-refval)

                   d8 = ddot(Nvv_ab,H2B_vvvv(a,b,:,:),t3c(:,:,c,i,j,k))
                   d8 = d8 - ddot(Nvv_ab,H2B_vvvv(a,c,:,:),t3c(:,:,b,i,j,k))
                   !refval = ZERO
                   !do e = 1,nua
                   !   do f = 1,nub
                   !      refval = refval + H2B_vvvv(a,b,e,f)*t3c(e,f,c,i,j,k)&
                   !                      - H2B_vvvv(a,c,e,f)*t3c(e,f,b,i,j,k)
                   !   end do
                   !end do
                   !error(14) = error(14) + (d8-refval)

                   d9 = ddot(Nov_aa,H2A_voov_r1(a,:,i,:),t3c(:,b,c,:,j,k))
                   !refval = ZERO
                   !do e = 1,nua
                   !   do m = 1,noa
                   !      refval = refval + H2A_voov(a,m,i,e)*t3c(e,b,c,m,j,k)
                   !   end do
                   !end do
                   !error(15) = error(15) + (d9-refval)

                   d10 = ddot(Nov_bb,H2B_voov_r1(a,:,i,:),t3d(:,b,c,:,j,k))
                   !refval = ZERO
                   !do e = 1,nub
                   !   do m = 1,nob
                   !      refval = refval + H2B_voov(a,m,i,e)*t3d(e,b,c,m,j,k)
                   !   end do
                   !end do
                   !error(16) = error(16) + (d10-refval)

                   d11 = ddot(Nov_aa,H2B_ovvo_r1(:,b,:,j),t3b(a,:,c,i,:,k))
                   d11 = d11 - ddot(Nov_aa,H2B_ovvo_r1(:,c,:,j),t3b(a,:,b,i,:,k))
                   d11 = d11 - ddot(Nov_aa,H2B_ovvo_r1(:,b,:,k),t3b(a,:,c,i,:,j))
                   d11 = d11 + ddot(Nov_aa,H2B_ovvo_r1(:,c,:,k),t3b(a,:,b,i,:,j))
                   !refval = ZERO
                   !do m = 1,noa
                   !   do e = 1,nua
                   !      refval = refval + H2B_ovvo(m,b,e,j)*t3b(a,e,c,i,m,k)&
                   !                      - H2B_ovvo(m,c,e,j)*t3b(a,e,b,i,m,k)&
                   !                      - H2B_ovvo(m,b,e,k)*t3b(a,e,c,i,m,j)&
                   !                      + H2B_ovvo(m,c,e,k)*t3b(a,e,b,i,m,j)
                   !   end do
                   !end do
                   !error(17) = error(17) + (d11-refval)

                   d12 = ddot(Nov_bb,H2C_voov_r1(b,:,j,:),t3c(a,:,c,i,:,k))
                   d12 = d12 - ddot(Nov_bb,H2C_voov_r1(c,:,j,:),t3c(a,:,b,i,:,k))
                   d12 = d12 - ddot(Nov_bb,H2C_voov_r1(b,:,k,:),t3c(a,:,c,i,:,j))
                   d12 = d12 + ddot(Nov_bb,H2C_voov_r1(c,:,k,:),t3c(a,:,b,i,:,j))
                   !refval = ZERO
                   !do m = 1,nob
                   !   do e = 1,nub
                   !      refval = refval + H2C_voov(b,m,j,e)*t3c(a,e,c,i,m,k)&
                   !                      - H2C_voov(c,m,j,e)*t3c(a,e,b,i,m,k)&
                   !                      - H2C_voov(b,m,k,e)*t3c(a,e,c,i,m,j)&
                   !                      + H2C_voov(c,m,k,e)*t3c(a,e,b,i,m,j)
                   !   end do
                   !end do
                   !error(18) = error(18) + (d12-refval)

                   d13 = MINUSONE*ddot(Nov_ab,H2B_ovov_r1(:,b,i,:),t3c(a,:,c,:,j,k))
                   d13 = d13 + ddot(Nov_ab,H2B_ovov_r1(:,c,i,:),t3c(a,:,b,:,j,k))
                   !refval = ZERO
                   !do m = 1,noa
                   !   do e = 1,nub
                   !      refval = refval - H2B_ovov(m,b,i,e)*t3c(a,e,c,m,j,k)&
                   !                      + H2B_ovov(m,c,i,e)*t3c(a,e,b,m,j,k)
                   !   end do
                   !end do
                   !error(19) = error(19) + (d13-refval)

                   d14 = MINUSONE*ddot(Nov_ba,H2B_vovo_r1(a,:,:,j),t3c(:,b,c,i,:,k))
                   d14 = d14 + ddot(Nov_ba,H2B_vovo_r1(a,:,:,k),t3c(:,b,c,i,:,j))
                   !refval = ZERO
                   !do m = 1,nob
                   !   do e = 1,nua
                   !      refval = refval - H2B_vovo(a,m,e,j)*t3c(e,b,c,i,m,k)&
                   !                      + H2B_vovo(a,m,e,k)*t3c(e,b,c,i,m,j)
                   !   end do
                   !end do
                   !error(20) = error(20) + (d14-refval)

                   residual = d1+d2+d3+d4+d5+d6+d7+d8+d9+d10&
                              +d11+d12+d13+d14&
                              +m1+m2+m3+m4+m5+m6
                   denom = fA_oo(i,i)+fB_oo(j,j)+fB_oo(k,k)&
                           -fA_vv(a,a)-fB_vv(b,b)-fB_vv(c,c)
                   val = t3c(a,b,c,i,j,k) + residual/(denom-shift)
                   mval = MINUSONE*val

                   t3c_new(a,b,c,i,j,k) = val
                   t3c_new(a,c,b,i,j,k) = mval
                   t3c_new(a,b,c,i,k,j) = mval
                   t3c_new(a,c,b,i,k,j) = val

                end do

                !do i = 1,20
                !   print*,'Error in term',i,'=',error(i)
                !end do

            end subroutine update_t3c
                    
            subroutine update_t3d(t2c,t3c,t3d,&
                         triples_list,&
                         vB_oovv,vC_oovv,&
                         H1B_oo,H1B_vv,H2C_oooo,&
                         H2C_vvvv,H2C_voov,H2B_ovvo,&
                         H2C_vooo,I2C_vvov,&
                         fB_oo,fB_vv,shift,&
                         noa,nua,nob,nub,num_triples,t3d_new)

                 integer, intent(in) :: noa, nua, nob, nub, num_triples
                 real(kind=8), intent(in) :: shift
                 integer, intent(in) :: triples_list(num_triples,6)
                 real(kind=8), intent(in) :: t2c(nub,nub,nob,nob),&
                                             t3c(nua,nub,nub,noa,nob,nob),&
                                             t3d(nub,nub,nub,nob,nob,nob),&
                                             vB_oovv(noa,nob,nua,nub),&
                                             vC_oovv(nob,nob,nub,nub),&
                                             H1B_oo(nob,nob),H1B_vv(nub,nub),&
                                             H2C_oooo(nob,nob,nob,nob),&
                                             H2C_vvvv(nub,nub,nub,nub),&
                                             H2C_voov(nub,nob,nob,nub),&
                                             H2B_ovvo(noa,nub,nua,nob),&
                                             H2C_vooo(nub,nob,nob,nob),&
                                             I2C_vvov(nub,nub,nob,nub),&
                                             fB_oo(nob,nob),fB_vv(nub,nub)

                real(kind=8), intent(out) :: t3d_new(nub,nub,nub,nob,nob,nob)

                integer :: a, b, c, i, j, k, ct,&
                           Noo_bb, Nvv_bb, Nov_bb, Novv_bbb, Noov_bbb,&
                           Noov_baa, Novv_aab, Nov_aa
                real(kind=8) :: m1, m2, d1, d2, d3, d4, d5, d6,&
                                residual, denom, val, mval
                real(kind=8) :: vt3_ob(nob), vt3_ub(nub),&
                                H2C_voov_r(nub,nub,nob,nob),& ! reorder 1432
                                H2B_ovvo_r(nua,nub,noa,nob),& ! reorder 3214
                                vC_oovv_r1(nub,nub,nob,nob),& ! reorder 3421
                                vC_oovv_r2(nub,nob,nob,nub),& ! reorder 4123 
                                vB_oovv_r1(nua,nub,noa,nob),& ! reorder 3412
                                vB_oovv_r2(nua,noa,nob,nub),& ! reorder 3124
                                temp1a(nob), temp1b(nob),&
                                temp2a(nub), temp2b(nub)

                real(kind=8), parameter :: MINUSONE=-1.0d+0, HALF=0.5d+0, ZERO=0.0d+0

                !integer :: m, n, e, f
                !real(kind=8) :: refval ,error(8)

                Nov_aa = noa*nua
                Noo_bb = nob*nob
                Nvv_bb = nub*nub
                Nov_bb = nob*nub
                Novv_bbb = nob*nub*nub
                Noov_bbb = nob*nob*nub
                Noov_baa = nob*noa*nua
                Novv_aab = noa*nua*nub

                call reorder1432(H2C_voov,H2C_voov_r)
                call reorder3214(H2B_ovvo,H2B_ovvo_r)
                call reorder3421(vC_oovv,vC_oovv_r1)
                call reorder4123(vC_oovv,vC_oovv_r2)
                call reorder3412(vB_oovv,vB_oovv_r1)
                call reorder3124(vB_oovv,vB_oovv_r2)

                !do i = 1,8
                !   error(i) = ZERO
                !end do

                do ct = 1 , num_triples
            
                   ! Shift indices up by since the triples list is coming from
                   ! Python, where indices start from 0       
                   a = triples_list(ct,1)+1
                   b = triples_list(ct,2)+1
                   c = triples_list(ct,3)+1
                   i = triples_list(ct,4)+1
                   j = triples_list(ct,5)+1
                   k = triples_list(ct,6)+1
                    
                   ! Calculate devectorized residual for triple |ijkabc>

                   call dgemv(Novv_bbb,nob,vC_oovv_r1,t3d(a,:,:,i,j,:),temp1a)
                   call dgemv(Novv_aab,nob,vB_oovv_r1,t3c(:,a,:,:,i,j),temp1b)
                   vt3_ob = HALF*temp1a + temp1b
                   m1 = MINUSONE*ddot(nob,H2C_vooo(a,:,i,j)+vt3_ob,t2c(b,c,:,k))

                   call dgemv(Novv_bbb,nob,vC_oovv_r1,t3d(a,:,:,k,j,:),temp1a)
                   call dgemv(Novv_aab,nob,vB_oovv_r1,t3c(:,a,:,:,k,j),temp1b)
                   vt3_ob = HALF*temp1a + temp1b
                   m1 = m1 + ddot(nob,H2C_vooo(a,:,k,j)+vt3_ob,t2c(b,c,:,i))

                   call dgemv(Novv_bbb,nob,vC_oovv_r1,t3d(a,:,:,i,k,:),temp1a)
                   call dgemv(Novv_aab,nob,vB_oovv_r1,t3c(:,a,:,:,i,k),temp1b)
                   vt3_ob = HALF*temp1a + temp1b
                   m1 = m1 + ddot(nob,H2C_vooo(a,:,i,k)+vt3_ob,t2c(b,c,:,j))

                   call dgemv(Novv_bbb,nob,vC_oovv_r1,t3d(b,:,:,i,j,:),temp1a)
                   call dgemv(Novv_aab,nob,vB_oovv_r1,t3c(:,b,:,:,i,j),temp1b)
                   vt3_ob = HALF*temp1a + temp1b
                   m1 = m1 + ddot(nob,H2C_vooo(b,:,i,j)+vt3_ob,t2c(a,c,:,k))

                   call dgemv(Novv_bbb,nob,vC_oovv_r1,t3d(b,:,:,k,j,:),temp1a)
                   call dgemv(Novv_aab,nob,vB_oovv_r1,t3c(:,b,:,:,k,j),temp1b)
                   vt3_ob = HALF*temp1a + temp1b
                   m1 = m1 - ddot(nob,H2C_vooo(b,:,k,j)+vt3_ob,t2c(a,c,:,i))

                   call dgemv(Novv_bbb,nob,vC_oovv_r1,t3d(b,:,:,i,k,:),temp1a)
                   call dgemv(Novv_aab,nob,vB_oovv_r1,t3c(:,b,:,:,i,k),temp1b)
                   vt3_ob = HALF*temp1a + temp1b
                   m1 = m1 - ddot(nob,H2C_vooo(b,:,i,k)+vt3_ob,t2c(a,c,:,j))

                   call dgemv(Novv_bbb,nob,vC_oovv_r1,t3d(c,:,:,i,j,:),temp1a)
                   call dgemv(Novv_aab,nob,vB_oovv_r1,t3c(:,c,:,:,i,j),temp1b)
                   vt3_ob = HALF*temp1a + temp1b
                   m1 = m1 + ddot(nob,H2C_vooo(c,:,i,j)+vt3_ob,t2c(b,a,:,k))

                   call dgemv(Novv_bbb,nob,vC_oovv_r1,t3d(c,:,:,k,j,:),temp1a)
                   call dgemv(Novv_aab,nob,vB_oovv_r1,t3c(:,c,:,:,k,j),temp1b)
                   vt3_ob = HALF*temp1a + temp1b
                   m1 = m1 - ddot(nob,H2C_vooo(c,:,k,j)+vt3_ob,t2c(b,a,:,i))

                   call dgemv(Novv_bbb,nob,vC_oovv_r1,t3d(c,:,:,i,k,:),temp1a)
                   call dgemv(Novv_aab,nob,vB_oovv_r1,t3c(:,c,:,:,i,k),temp1b)
                   vt3_ob = HALF*temp1a + temp1b
                   m1 = m1 - ddot(nob,H2C_vooo(c,:,i,k)+vt3_ob,t2c(b,a,:,j))

                    !refval = ZERO
                    !do m = 1,nob
                    ! do f = 1,nub
                    !   do e = f+1,nub
                    !     do n = 1,nob
                    !       refval = refval&
                    !       -vC_oovv(m,n,e,f)*t3d(a,e,f,i,j,n)*t2c(b,c,m,k)&
                    !       +vC_oovv(m,n,e,f)*t3d(a,e,f,k,j,n)*t2c(b,c,m,i)&
                    !       +vC_oovv(m,n,e,f)*t3d(a,e,f,i,k,n)*t2c(b,c,m,j)&
                    !       +vC_oovv(m,n,e,f)*t3d(b,e,f,i,j,n)*t2c(a,c,m,k)&
                    !       -vC_oovv(m,n,e,f)*t3d(b,e,f,k,j,n)*t2c(a,c,m,i)&
                    !       -vC_oovv(m,n,e,f)*t3d(b,e,f,i,k,n)*t2c(a,c,m,j)&
                    !       +vC_oovv(m,n,e,f)*t3d(c,e,f,i,j,n)*t2c(b,a,m,k)&
                    !       -vC_oovv(m,n,e,f)*t3d(c,e,f,k,j,n)*t2c(b,a,m,i)&
                    !       -vC_oovv(m,n,e,f)*t3d(c,e,f,i,k,n)*t2c(b,a,m,j)
                    !     end do
                    !   end do
                    ! end do
                    ! do f = 1,nua
                    !    do e = 1,nub
                    !       do n = 1,noa
                    !          refval = refval&
                    !          -vB_oovv(n,m,f,e)*t3c(f,a,e,n,i,j)*t2c(b,c,m,k)&
                    !          +vB_oovv(n,m,f,e)*t3c(f,a,e,n,k,j)*t2c(b,c,m,i)&
                    !          +vB_oovv(n,m,f,e)*t3c(f,a,e,n,i,k)*t2c(b,c,m,j)&
                    !          +vB_oovv(n,m,f,e)*t3c(f,b,e,n,i,j)*t2c(a,c,m,k)&
                    !          -vB_oovv(n,m,f,e)*t3c(f,b,e,n,k,j)*t2c(a,c,m,i)&
                    !          -vB_oovv(n,m,f,e)*t3c(f,b,e,n,i,k)*t2c(a,c,m,j)&
                    !          +vB_oovv(n,m,f,e)*t3c(f,c,e,n,i,j)*t2c(b,a,m,k)&
                    !          -vB_oovv(n,m,f,e)*t3c(f,c,e,n,k,j)*t2c(b,a,m,i)&
                    !          -vB_oovv(n,m,f,e)*t3c(f,c,e,n,i,k)*t2c(b,a,m,j)
                    !       end do
                    !    end do
                    !  end do
                    !  refval = refval - H2C_vooo(a,m,i,j)*t2c(b,c,m,k)&
                    !                  + H2C_vooo(a,m,k,j)*t2c(b,c,m,i)&
                    !                  + H2C_vooo(a,m,i,k)*t2c(b,c,m,j)&
                    !                  + H2C_vooo(b,m,i,j)*t2c(a,c,m,k)&
                    !                  - H2C_vooo(b,m,k,j)*t2c(a,c,m,i)&
                    !                  - H2C_vooo(b,m,i,k)*t2c(a,c,m,j)&
                    !                  + H2C_vooo(c,m,i,j)*t2c(b,a,m,k)&
                    !                  - H2C_vooo(c,m,k,j)*t2c(b,a,m,i)&
                    !                  - H2C_vooo(c,m,i,k)*t2c(b,a,m,j)
                    !end do
                    !error(1) = error(1) + (m1-refval)

                   call dgemv(Noov_bbb,nub,vC_oovv_r2,t3d(a,b,:,i,:,:),temp2a)
                   call dgemv(Noov_baa,nub,vB_oovv_r2,t3c(:,a,b,:,i,:),temp2b)
                   vt3_ub = HALF*temp2a + temp2b
                   m2 = ddot(nub,I2C_vvov(a,b,i,:)-vt3_ub,t2c(:,c,j,k))

                   call dgemv(Noov_bbb,nub,vC_oovv_r2,t3d(a,c,:,i,:,:),temp2a)
                   call dgemv(Noov_baa,nub,vB_oovv_r2,t3c(:,a,c,:,i,:),temp2b)
                   vt3_ub = HALF*temp2a + temp2b
                   m2 = m2 - ddot(nub,I2C_vvov(a,c,i,:)-vt3_ub,t2c(:,b,j,k))

                   call dgemv(Noov_bbb,nub,vC_oovv_r2,t3d(c,b,:,i,:,:),temp2a)
                   call dgemv(Noov_baa,nub,vB_oovv_r2,t3c(:,c,b,:,i,:),temp2b)
                   vt3_ub = HALF*temp2a + temp2b
                   m2 = m2 - ddot(nub,I2C_vvov(c,b,i,:)-vt3_ub,t2c(:,a,j,k))

                   call dgemv(Noov_bbb,nub,vC_oovv_r2,t3d(a,b,:,j,:,:),temp2a)
                   call dgemv(Noov_baa,nub,vB_oovv_r2,t3c(:,a,b,:,j,:),temp2b)
                   vt3_ub = HALF*temp2a + temp2b
                   m2 = m2 - ddot(nub,I2C_vvov(a,b,j,:)-vt3_ub,t2c(:,c,i,k))

                   call dgemv(Noov_bbb,nub,vC_oovv_r2,t3d(a,c,:,j,:,:),temp2a)
                   call dgemv(Noov_baa,nub,vB_oovv_r2,t3c(:,a,c,:,j,:),temp2b)
                   vt3_ub = HALF*temp2a + temp2b
                   m2 = m2 + ddot(nub,I2C_vvov(a,c,j,:)-vt3_ub,t2c(:,b,i,k))

                   call dgemv(Noov_bbb,nub,vC_oovv_r2,t3d(c,b,:,j,:,:),temp2a)
                   call dgemv(Noov_baa,nub,vB_oovv_r2,t3c(:,c,b,:,j,:),temp2b)
                   vt3_ub = HALF*temp2a + temp2b
                   m2 = m2 + ddot(nub,I2C_vvov(c,b,j,:)-vt3_ub,t2c(:,a,i,k))

                   call dgemv(Noov_bbb,nub,vC_oovv_r2,t3d(a,b,:,k,:,:),temp2a)
                   call dgemv(Noov_baa,nub,vB_oovv_r2,t3c(:,a,b,:,k,:),temp2b)
                   vt3_ub = HALF*temp2a + temp2b
                   m2 = m2 - ddot(nub,I2C_vvov(a,b,k,:)-vt3_ub,t2c(:,c,j,i))
                   
                   call dgemv(Noov_bbb,nub,vC_oovv_r2,t3d(a,c,:,k,:,:),temp2a)
                   call dgemv(Noov_baa,nub,vB_oovv_r2,t3c(:,a,c,:,k,:),temp2b)
                   vt3_ub = HALF*temp2a + temp2b
                   m2 = m2 + ddot(nub,I2C_vvov(a,c,k,:)-vt3_ub,t2c(:,b,j,i))
                   
                   call dgemv(Noov_bbb,nub,vC_oovv_r2,t3d(c,b,:,k,:,:),temp2a)
                   call dgemv(Noov_baa,nub,vB_oovv_r2,t3c(:,c,b,:,k,:),temp2b)
                   vt3_ub = HALF*temp2a + temp2b
                   m2 = m2 + ddot(nub,I2C_vvov(c,b,k,:)-vt3_ub,t2c(:,a,j,i))

                   !refval = ZERO
                   !do e = 1,nub
                   !   do f = 1,nub
                   !      do m = 1,nob
                   !         do n = m+1,nob
                   !            refval = refval&
                   !            -vC_oovv(m,n,e,f)*t3d(a,b,f,i,m,n)*t2c(e,c,j,k)&
                   !            +vC_oovv(m,n,e,f)*t3d(c,b,f,i,m,n)*t2c(e,a,j,k)&
                   !            +vC_oovv(m,n,e,f)*t3d(a,c,f,i,m,n)*t2c(e,b,j,k)&
                   !            +vC_oovv(m,n,e,f)*t3d(a,b,f,j,m,n)*t2c(e,c,i,k)&
                   !            -vC_oovv(m,n,e,f)*t3d(c,b,f,j,m,n)*t2c(e,a,i,k)&
                   !            -vC_oovv(m,n,e,f)*t3d(a,c,f,j,m,n)*t2c(e,b,i,k)&
                   !            +vC_oovv(m,n,e,f)*t3d(a,b,f,k,m,n)*t2c(e,c,j,i)&
                   !            -vC_oovv(m,n,e,f)*t3d(c,b,f,k,m,n)*t2c(e,a,j,i)&
                   !            -vC_oovv(m,n,e,f)*t3d(a,c,f,k,m,n)*t2c(e,b,j,i)
                   !         end do
                   !      end do
                   !   end do
                   !   do f = 1,nua
                   !      do m = 1,nob
                   !         do n = 1,noa
                   !            refval = refval&
                   !            -vB_oovv(n,m,f,e)*t3c(f,a,b,n,i,m)*t2c(e,c,j,k)&
                   !            +vB_oovv(n,m,f,e)*t3c(f,c,b,n,i,m)*t2c(e,a,j,k)&
                   !            +vB_oovv(n,m,f,e)*t3c(f,a,c,n,i,m)*t2c(e,b,j,k)&
                   !            +vB_oovv(n,m,f,e)*t3c(f,a,b,n,j,m)*t2c(e,c,i,k)&
                   !            -vB_oovv(n,m,f,e)*t3c(f,c,b,n,j,m)*t2c(e,a,i,k)&
                   !            -vB_oovv(n,m,f,e)*t3c(f,a,c,n,j,m)*t2c(e,b,i,k)&
                   !            +vB_oovv(n,m,f,e)*t3c(f,a,b,n,k,m)*t2c(e,c,j,i)&
                   !            -vB_oovv(n,m,f,e)*t3c(f,c,b,n,k,m)*t2c(e,a,j,i)&
                   !            -vB_oovv(n,m,f,e)*t3c(f,a,c,n,k,m)*t2c(e,b,j,i)
                   !         end do
                   !      end do
                   !   end do
                   !   refval = refval + I2C_vvov(a,b,i,e)*t2c(e,c,j,k)&
                   !                   - I2C_vvov(c,b,i,e)*t2c(e,a,j,k)&
                   !                   - I2C_vvov(a,c,i,e)*t2c(e,b,j,k)&
                   !                   - I2C_vvov(a,b,j,e)*t2c(e,c,i,k)&
                   !                   + I2C_vvov(c,b,j,e)*t2c(e,a,i,k)&
                   !                   + I2C_vvov(a,c,j,e)*t2c(e,b,i,k)&
                   !                   - I2C_vvov(a,b,k,e)*t2c(e,c,j,i)&
                   !                   + I2C_vvov(c,b,k,e)*t2c(e,a,j,i)&
                   !                   + I2C_vvov(a,c,k,e)*t2c(e,b,j,i)
                   ! end do
                   ! error(2) = error(2) + (m2-refval)
                    
                   d1 = MINUSONE*ddot(nob,H1B_oo(:,k),t3d(a,b,c,i,j,:))
                   d1 = d1 + ddot(nob,H1B_oo(:,j),t3d(a,b,c,i,k,:))
                   d1 = d1 + ddot(nob,H1B_oo(:,i),t3d(a,b,c,k,j,:))
                   !refval = ZERO
                   !do m = 1,nob
                   !   refval = refval - H1B_oo(m,k)*t3d(a,b,c,i,j,m)&
                   !                   + H1B_oo(m,j)*t3d(a,b,c,i,k,m)&
                   !                   + H1B_oo(m,i)*t3d(a,b,c,k,j,m)
                   !end do
                   !error(3) = error(3) + (d1-refval)

                   d2 = ddot(nub,H1B_vv(c,:),t3d(a,b,:,i,j,k))
                   d2 = d2 - ddot(nub,H1B_vv(b,:),t3d(a,c,:,i,j,k))
                   d2 = d2 - ddot(nub,H1B_vv(a,:),t3d(c,b,:,i,j,k))
                   !refval = ZERO
                   !do e = 1,nub
                   !   refval = refval + H1B_vv(c,e)*t3d(a,b,e,i,j,k)&
                   !                   - H1B_vv(b,e)*t3d(a,c,e,i,j,k)&
                   !                   - H1B_vv(a,e)*t3d(c,b,e,i,j,k)
                   !end do
                   !error(4) = error(4) + (d2-refval)

                   d3 = ddot(Noo_bb,H2C_oooo(:,:,i,j),t3d(a,b,c,:,:,k))
                   d3 = d3 - ddot(Noo_bb,H2C_oooo(:,:,k,j),t3d(a,b,c,:,:,i))
                   d3 = d3 - ddot(Noo_bb,H2C_oooo(:,:,i,k),t3d(a,b,c,:,:,j))
                   d3 = HALF*d3
                   !refval = ZERO
                   !do m = 1,nob
                   !   do n = m+1,nob
                   !      refval = refval + H2C_oooo(m,n,i,j)*t3d(a,b,c,m,n,k)&
                   !                      - H2C_oooo(m,n,k,j)*t3d(a,b,c,m,n,i)&
                   !                      - H2C_oooo(m,n,i,k)*t3d(a,b,c,m,n,j)
                   !   end do
                   !end do
                   !error(5) = error(5) + (d3-refval)
        
                   d4 = ddot(Nvv_bb,H2C_vvvv(a,b,:,:),t3d(:,:,c,i,j,k))
                   d4 = d4 - ddot(Nvv_bb,H2C_vvvv(c,b,:,:),t3d(:,:,a,i,j,k))
                   d4 = d4 - ddot(Nvv_bb,H2C_vvvv(a,c,:,:),t3d(:,:,b,i,j,k))
                   d4 = HALF*d4
                   !refval = ZERO
                   !do e = 1,nub
                   !   do f = e+1,nub
                   !      refval = refval + H2C_vvvv(a,b,e,f)*t3d(e,f,c,i,j,k)&
                   !                      - H2C_vvvv(a,c,e,f)*t3d(e,f,b,i,j,k)&
                   !                      - H2C_vvvv(c,b,e,f)*t3d(e,f,a,i,j,k)
                   !   end do
                   !end do
                   !error(6) = error(6) + (d4-refval)

                   d5 = ddot(Nov_aa,H2B_ovvo_r(:,a,:,i),t3c(:,b,c,:,j,k))
                   d5 = d5 - ddot(Nov_aa,H2B_ovvo_r(:,a,:,j),t3c(:,b,c,:,i,k))
                   d5 = d5 - ddot(Nov_aa,H2B_ovvo_r(:,a,:,k),t3c(:,b,c,:,j,i))
                   d5 = d5 - ddot(Nov_aa,H2B_ovvo_r(:,b,:,i),t3c(:,a,c,:,j,k))
                   d5 = d5 + ddot(Nov_aa,H2B_ovvo_r(:,b,:,j),t3c(:,a,c,:,i,k))
                   d5 = d5 + ddot(Nov_aa,H2B_ovvo_r(:,b,:,k),t3c(:,a,c,:,j,i))
                   d5 = d5 - ddot(Nov_aa,H2B_ovvo_r(:,c,:,i),t3c(:,b,a,:,j,k))
                   d5 = d5 + ddot(Nov_aa,H2B_ovvo_r(:,c,:,j),t3c(:,b,a,:,i,k))
                   d5 = d5 + ddot(Nov_aa,H2B_ovvo_r(:,c,:,k),t3c(:,b,a,:,j,i))
                   !refval = ZERO
                   !do m = 1,noa
                   !   do e = 1,nua
                   !      refval = refval&
                   !      +H2B_ovvo(m,a,e,i)*t3c(e,b,c,m,j,k)&
                   !      -H2B_ovvo(m,a,e,j)*t3c(e,b,c,m,i,k)&
                   !      -H2B_ovvo(m,a,e,k)*t3c(e,b,c,m,j,i)&
                   !      -H2B_ovvo(m,b,e,i)*t3c(e,a,c,m,j,k)&
                   !      +H2B_ovvo(m,b,e,j)*t3c(e,a,c,m,i,k)&
                   !      +H2B_ovvo(m,b,e,k)*t3c(e,a,c,m,j,i)&
                   !      -H2B_ovvo(m,c,e,i)*t3c(e,b,a,m,j,k)&
                   !      +H2B_ovvo(m,c,e,j)*t3c(e,b,a,m,i,k)&
                   !      +H2B_ovvo(m,c,e,k)*t3c(e,b,a,m,j,i)
                   !   end do
                   !end do
                   !error(7) = error(7) + (d5-refval)

                   d6 = ddot(Nov_bb,H2C_voov_r(c,:,k,:),t3d(a,b,:,i,j,:))
                   d6 = d6 - ddot(Nov_bb,H2C_voov_r(c,:,i,:),t3d(a,b,:,k,j,:))
                   d6 = d6 - ddot(Nov_bb,H2C_voov_r(c,:,j,:),t3d(a,b,:,i,k,:))
                   d6 = d6 - ddot(Nov_bb,H2C_voov_r(a,:,k,:),t3d(c,b,:,i,j,:))
                   d6 = d6 + ddot(Nov_bb,H2C_voov_r(a,:,i,:),t3d(c,b,:,k,j,:))
                   d6 = d6 + ddot(Nov_bb,H2C_voov_r(a,:,j,:),t3d(c,b,:,i,k,:))
                   d6 = d6 - ddot(Nov_bb,H2C_voov_r(b,:,k,:),t3d(a,c,:,i,j,:))
                   d6 = d6 + ddot(Nov_bb,H2C_voov_r(b,:,i,:),t3d(a,c,:,k,j,:))
                   d6 = d6 + ddot(Nov_bb,H2C_voov_r(b,:,j,:),t3d(a,c,:,i,k,:))
                   !refval = ZERO
                   !do m = 1,nob
                   !   do e = 1,nub
                   !      refval = refval&
                   !      +H2C_voov(c,m,k,e)*t3d(a,b,e,i,j,m)&
                   !      -H2C_voov(c,m,i,e)*t3d(a,b,e,k,j,m)&
                   !      -H2C_voov(c,m,j,e)*t3d(a,b,e,i,k,m)&
                   !      -H2C_voov(a,m,k,e)*t3d(c,b,e,i,j,m)&
                   !      +H2C_voov(a,m,i,e)*t3d(c,b,e,k,j,m)&
                   !      +H2C_voov(a,m,j,e)*t3d(c,b,e,i,k,m)&
                   !      -H2C_voov(b,m,k,e)*t3d(a,c,e,i,j,m)&
                   !      +H2C_voov(b,m,i,e)*t3d(a,c,e,k,j,m)&
                   !      +H2C_voov(b,m,j,e)*t3d(a,c,e,i,k,m)
                   !   end do
                   !end do
                   !error(8) = error(8) + (d6-refval)

                   residual = m1 + m2 + d1 + d2 + d3 + d4 + d5 + d6
                   denom = fB_oo(i,i)+fB_oo(j,j)+fB_oo(k,k)&
                           -fB_vv(a,a)-fB_vv(b,b)-fB_vv(c,c)
                   val = t3d(a,b,c,i,j,k) + residual/(denom-shift)
                   mval = MINUSONE*val

                   t3d_new(a,b,c,i,j,k) = val
                   t3d_new(A,B,C,K,I,J) = val
                   t3d_new(A,B,C,J,K,I) = val
                   t3d_new(A,B,C,I,K,J) = mval
                   t3d_new(A,B,C,J,I,K) = mval
                   t3d_new(A,B,C,K,J,I) = mval
                                      
                   t3d_new(B,A,C,I,J,K) = mval
                   t3d_new(B,A,C,K,I,J) = mval
                   t3d_new(B,A,C,J,K,I) = mval
                   t3d_new(B,A,C,I,K,J) = val
                   t3d_new(B,A,C,J,I,K) = val
                   t3d_new(B,A,C,K,J,I) = val
                                      
                   t3d_new(A,C,B,I,J,K) = mval
                   t3d_new(A,C,B,K,I,J) = mval
                   t3d_new(A,C,B,J,K,I) = mval
                   t3d_new(A,C,B,I,K,J) = val
                   t3d_new(A,C,B,J,I,K) = val
                   t3d_new(A,C,B,K,J,I) = val
                                      
                   t3d_new(C,B,A,I,J,K) = mval
                   t3d_new(C,B,A,K,I,J) = mval
                   t3d_new(C,B,A,J,K,I) = mval
                   t3d_new(C,B,A,I,K,J) = val
                   t3d_new(C,B,A,J,I,K) = val
                   t3d_new(C,B,A,K,J,I) = val
                                      
                   t3d_new(B,C,A,I,J,K) = val
                   t3d_new(B,C,A,K,I,J) = val
                   t3d_new(B,C,A,J,K,I) = val
                   t3d_new(B,C,A,I,K,J) = mval
                   t3d_new(B,C,A,J,I,K) = mval
                   t3d_new(B,C,A,K,J,I) = mval
                                      
                   t3d_new(C,A,B,I,J,K) = val
                   t3d_new(C,A,B,K,I,J) = val
                   t3d_new(C,A,B,J,K,I) = val
                   t3d_new(C,A,B,I,K,J) = mval
                   t3d_new(C,A,B,J,I,K) = mval
                   t3d_new(C,A,B,K,J,I) = mval

                end do

                !do i = 1,8
                !   print*,'Error in term',i,'=',error(i)
                !end do

            end subroutine update_t3d



            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            !!!!!!!!!!!!!!!!!!!!!! UTILITY ROUTINES !!!!!!!!!!!!!!!!!!!!!!!!
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            subroutine reorder4321(x_in,x_out)

                    real(kind=8), intent(in) :: x_in(:,:,:,:)
                    real(kind=8), intent(out) :: x_out(:,:,:,:)

                    integer :: i1, i2, i3, i4
                    integer :: L1, L2, L3, L4

                    L1 = size(x_in,1)
                    L2 = size(x_in,2)
                    L3 = size(x_in,3)
                    L4 = size(x_in,4)

                    do i1 = 1,L1
                       do i2 = 1,L2
                          do i3 = 1,L3
                             do i4 = 1,L4
                                x_out(i4,i3,i2,i1) = x_in(i1,i2,i3,i4)
                             end do
                          end do
                       end do
                    end do

            end subroutine reorder4321

            subroutine reorder3412(x_in,x_out)

                    real(kind=8), intent(in) :: x_in(:,:,:,:)
                    real(kind=8), intent(out) :: x_out(:,:,:,:)

                    integer :: i1, i2, i3, i4
                    integer :: L1, L2, L3, L4

                    L1 = size(x_in,1)
                    L2 = size(x_in,2)
                    L3 = size(x_in,3)
                    L4 = size(x_in,4)

                    do i1 = 1,L1
                       do i2 = 1,L2
                          do i3 = 1,L3
                             do i4 = 1,L4
                                x_out(i3,i4,i1,i2) = x_in(i1,i2,i3,i4)
                             end do
                          end do
                       end do
                    end do

            end subroutine reorder3412

            subroutine reorder3124(x_in,x_out)

                    real(kind=8), intent(in) :: x_in(:,:,:,:)
                    real(kind=8), intent(out) :: x_out(:,:,:,:)

                    integer :: i1, i2, i3, i4
                    integer :: L1, L2, L3, L4

                    L1 = size(x_in,1)
                    L2 = size(x_in,2)
                    L3 = size(x_in,3)
                    L4 = size(x_in,4)

                    do i1 = 1,L1
                       do i2 = 1,L2
                          do i3 = 1,L3
                             do i4 = 1,L4
                                x_out(i3,i1,i2,i4) = x_in(i1,i2,i3,i4)
                             end do
                          end do
                       end do
                    end do

            end subroutine reorder3124

            subroutine reorder4231(x_in,x_out)

                    real(kind=8), intent(in) :: x_in(:,:,:,:)
                    real(kind=8), intent(out) :: x_out(:,:,:,:)

                    integer :: i1, i2, i3, i4
                    integer :: L1, L2, L3, L4

                    L1 = size(x_in,1)
                    L2 = size(x_in,2)
                    L3 = size(x_in,3)
                    L4 = size(x_in,4)

                    do i1 = 1,L1
                       do i2 = 1,L2
                          do i3 = 1,L3
                             do i4 = 1,L4
                                x_out(i4,i2,i3,i1) = x_in(i1,i2,i3,i4)
                             end do
                          end do
                       end do
                    end do

            end subroutine reorder4231

            subroutine reorder1324(x_in,x_out)

                    real(kind=8), intent(in) :: x_in(:,:,:,:)
                    real(kind=8), intent(out) :: x_out(:,:,:,:)

                    integer :: i1, i2, i3, i4
                    integer :: L1, L2, L3, L4

                    L1 = size(x_in,1)
                    L2 = size(x_in,2)
                    L3 = size(x_in,3)
                    L4 = size(x_in,4)

                    do i1 = 1,L1
                       do i2 = 1,L2
                          do i3 = 1,L3
                             do i4 = 1,L4
                                x_out(i1,i3,i2,i4) = x_in(i1,i2,i3,i4)
                             end do
                          end do
                       end do
                    end do

            end subroutine reorder1324

            subroutine reorder3214(x_in,x_out)

                    real(kind=8), intent(in) :: x_in(:,:,:,:)
                    real(kind=8), intent(out) :: x_out(:,:,:,:)

                    integer :: i1, i2, i3, i4
                    integer :: L1, L2, L3, L4

                    L1 = size(x_in,1)
                    L2 = size(x_in,2)
                    L3 = size(x_in,3)
                    L4 = size(x_in,4)

                    do i1 = 1,L1
                       do i2 = 1,L2
                          do i3 = 1,L3
                             do i4 = 1,L4
                                x_out(i3,i2,i1,i4) = x_in(i1,i2,i3,i4)
                             end do
                          end do
                       end do
                    end do

            end subroutine reorder3214

            subroutine reorder4123(x_in,x_out)

                    real(kind=8), intent(in) :: x_in(:,:,:,:)
                    real(kind=8), intent(out) :: x_out(:,:,:,:)

                    integer :: i1, i2, i3, i4
                    integer :: L1, L2, L3, L4

                    L1 = size(x_in,1)
                    L2 = size(x_in,2)
                    L3 = size(x_in,3)
                    L4 = size(x_in,4)

                    do i1 = 1,L1
                       do i2 = 1,L2
                          do i3 = 1,L3
                             do i4 = 1,L4
                                x_out(i4,i1,i2,i3) = x_in(i1,i2,i3,i4)
                             end do
                          end do
                       end do
                    end do

            end subroutine reorder4123

            subroutine reorder3421(x_in,x_out)

                    real(kind=8), intent(in) :: x_in(:,:,:,:)
                    real(kind=8), intent(out) :: x_out(:,:,:,:)

                    integer :: i1, i2, i3, i4
                    integer :: L1, L2, L3, L4

                    L1 = size(x_in,1)
                    L2 = size(x_in,2)
                    L3 = size(x_in,3)
                    L4 = size(x_in,4)

                    do i1 = 1,L1
                       do i2 = 1,L2
                          do i3 = 1,L3
                             do i4 = 1,L4
                                x_out(i3,i4,i2,i1) = x_in(i1,i2,i3,i4)
                             end do
                          end do
                       end do
                    end do

            end subroutine reorder3421

            subroutine reorder1432(x_in,x_out)

                    real(kind=8), intent(in) :: x_in(:,:,:,:)
                    real(kind=8), intent(out) :: x_out(:,:,:,:)

                    integer :: i1, i2, i3, i4
                    integer :: L1, L2, L3, L4

                    L1 = size(x_in,1)
                    L2 = size(x_in,2)
                    L3 = size(x_in,3)
                    L4 = size(x_in,4)

                    do i1 = 1,L1
                       do i2 = 1,L2
                          do i3 = 1,L3
                             do i4 = 1,L4
                                x_out(i1,i4,i3,i2) = x_in(i1,i2,i3,i4)
                             end do
                          end do
                       end do
                    end do

            end subroutine reorder1432


            subroutine dgemv(K,N,A,x,y)
                    ! Assuming:
                    ! trans='t'
                    ! K = M = LDA = LDB (on Netlib), contraction dim
                    ! INCX = INCY = 1
                    ! ALPHA = 1.0
                    ! BETA = 0.0

                    integer :: K, N
                    double precision :: A(K,*), X(*), Y(*)
                    double precision :: zero
                    parameter(zero=0.0d+0)
                    double precision :: temp
                    integer :: i, j, jy
                    
                    jy = 1
                    do j = 1,n
                       temp = zero
                       do i = 1,k
                          temp = temp + a(i,j)*x(i)
                       end do
                       y(jy) = temp
                       jy = jy + 1
                    end do

            end subroutine dgemv

            double precision function ddot(N,dx,dy)

                    integer :: N
                    real(8) :: dx(*), dy(*)
                    real(8) :: dtemp
                    integer :: i, ix, iy, m, mp1

                    intrinsic mod

                    ddot = 0.0d0
                    dtemp = 0.0d0

                    ! perform the dot product using batches of 5
                    m = mod(n,5)

                    !
                    if (m .ne. 0) then
                       do i = 1,m
                          dtemp = dtemp + dx(i)*dy(i)
                       end do
                       if (n .lt. 5) then
                          ddot = dtemp
                          return
                       end if
                    end if

                    ! 
                    mp1 = m + 1
                    do i = mp1,N,5
                       dtemp = dtemp + dx(i)*dy(i)&
                                     + dx(i+1)*dy(i+1)&
                                     + dx(i+2)*dy(i+2)&
                                     + dx(i+3)*dy(i+3)&
                                     + dx(i+4)*dy(i+4)
                    end do

                    ! return the final dot product
                    ddot = dtemp

            end function ddot



end module ccp_loops
                                             
