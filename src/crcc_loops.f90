module crcc_loops

      implicit none

      contains

              subroutine crcc23A_opt(deltaA,deltaB,deltaC,deltaD,&
                              t2a,l1a,l2a,&
                              H2A_vooo,I2A_vvov,vA_oovv,H1A_ov,H2A_vovv,H2A_ooov,fA_oo,fA_vv,&
                              H1A_oo,H1A_vv,&
                              H2A_voov,H2A_oooo,H2A_vvvv,&
                              D3A_O,D3A_v,noa,nua)

                        real(kind=8), intent(out) :: deltaA, deltaB, deltaC, deltaD
                        integer, intent(in) :: noa, nua
                        real(kind=8), intent(in) :: fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),&
                        H1A_oo(1:noa,1:noa),H1A_vv(1:nua,1:nua),&
                        H2A_voov(1:nua,1:noa,1:noa,1:nua),&
                        H2A_oooo(1:noa,1:noa,1:noa,1:noa),&
                        H2A_vvvv(1:nua,1:nua,1:nua,1:nua),&
                        D3A_O(1:nua,1:noa,1:noa),&
                        D3A_V(1:nua,1:noa,1:nua),&
                        H2A_vooo(nua,noa,noa,noa),I2A_vvov(nua,nua,noa,nua),t2a(nua,nua,noa,noa),&
                        l1a(nua,noa),l2a(nua,nua,noa,noa),vA_oovv(noa,noa,nua,nua),&
                        H1A_ov(noa,nua),H2A_vovv(nua,noa,nua,nua),H2A_ooov(noa,noa,noa,nua)
                        integer :: i, j, k, a, b, c, nua2, i1, i2, i3, i4
                        real(kind=8) :: D, temp1, temp2, temp3, LM, X3A(nua,nua,nua), L3A(nua,nua,nua)

                        ! reordered arrays for DGEMMs
                        real(kind=8) :: I2A_vvov_1243(nua,nua,nua,noa), H2A_vovv_4312(nua,nua,nua,noa), H2A_ooov_4312(nua,noa,noa,noa)

                        call reorder1243(I2A_vvov,I2A_vvov_1243)
                        call reorder4312(H2A_vovv,H2A_vovv_4312)
                        call reorder4312(H2A_ooov,H2A_ooov_4312)

                        deltaA = 0.0d0
                        deltaB = 0.0d0
                        deltaC = 0.0d0
                        deltaD = 0.0d0

                        nua2 = nua*nua
                        do i = 1 , noa
                            do j = i+1, noa
                                do k = j+1, noa

                                   X3A = 0.0d0
                                   L3A = 0.0d0
                                   !!!!! MM(2,3)A !!!!!
                                   ! Diagram 1: -A(k/ij)A(a/bc) I2A_vooo(a,m,i,j)*t2a(b,c,m,k)
                                   call dgemm('n','t',nua,nua2,noa,-0.5d0,H2A_vooo(:,:,i,j),nua,t2a(:,:,:,k),nua2,1.0d0,X3A,nua)
                                   call dgemm('n','t',nua,nua2,noa,0.5d0,H2A_vooo(:,:,k,j),nua,t2a(:,:,:,i),nua2,1.0d0,X3A,nua)
                                   call dgemm('n','t',nua,nua2,noa,0.5d0,H2A_vooo(:,:,i,k),nua,t2a(:,:,:,j),nua2,1.0d0,X3A,nua)
                                   ! Diagram 2: A(i/jk)A(c/ab) I2A_vvov(a,b,i,e)*t2a(e,c,j,k)
                                   call dgemm('n','n',nua2,nua,nua,0.5d0,I2A_vvov_1243(:,:,:,i),nua2,t2a(:,:,j,k),nua,1.0d0,X3A,nua2)
                                   call dgemm('n','n',nua2,nua,nua,-0.5d0,I2A_vvov_1243(:,:,:,j),nua2,t2a(:,:,i,k),nua,1.0d0,X3A,nua2)
                                   call dgemm('n','n',nua2,nua,nua,-0.5d0,I2A_vvov_1243(:,:,:,k),nua2,t2a(:,:,j,i),nua,1.0d0,X3A,nua2)
                                   !!!!! L3A !!!!!
                                   ! Diagram 1: A(i/jk)A(c/ab) H2A_vovv(e,i,b,a)*l2a(e,c,j,k)
                                   call dgemm('n','n',nua2,nua,nua,0.5d0,H2A_vovv_4312(:,:,:,i),nua2,l2a(:,:,j,k),nua,1.0d0,L3A,nua2)                         
                                   call dgemm('n','n',nua2,nua,nua,-0.5d0,H2A_vovv_4312(:,:,:,j),nua2,l2a(:,:,i,k),nua,1.0d0,L3A,nua2)                         
                                   call dgemm('n','n',nua2,nua,nua,-0.5d0,H2A_vovv_4312(:,:,:,k),nua2,l2a(:,:,j,i),nua,1.0d0,L3A,nua2)
                                   ! Diagram 2: -A(k/ij)A(a/bc) H2A_ooov(j,i,m,a)*l2a(b,c,m,k)-> a,m,j,i * (b,c,m,k)'
                                   call dgemm('n','t',nua,nua2,noa,-0.5d0,H2A_ooov_4312(:,:,j,i),nua,l2a(:,:,:,k),nua2,1.0d0,L3A,nua)
                                   call dgemm('n','t',nua,nua2,noa,0.5d0,H2A_ooov_4312(:,:,k,i),nua,l2a(:,:,:,j),nua2,1.0d0,L3A,nua)
                                   call dgemm('n','t',nua,nua2,noa,0.5d0,H2A_ooov_4312(:,:,j,k),nua,l2a(:,:,:,i),nua2,1.0d0,L3A,nua)

                                    do a = 1, nua
                                        do b = a+1, nua
                                            do c = b+1, nua

                                                temp1 = X3A(a,b,c) + X3A(b,c,a) + X3A(c,a,b)&
                                                - X3A(a,c,b) - X3A(b,a,c) - X3A(c,b,a)

                                                temp2 = L3A(a,b,c) + L3A(b,c,a) + L3A(c,a,b)&
                                                - L3A(a,c,b) - L3A(b,a,c) - L3A(c,b,a)

                                                temp3 =&
                                                l1a(c,k)*vA_oovv(i,j,a,b)&
                                                -l1a(a,k)*vA_oovv(i,j,c,b)&
                                                -l1a(b,k)*vA_oovv(i,j,a,c)&
                                                -l1a(c,i)*vA_oovv(k,j,a,b)&
                                                -l1a(c,j)*vA_oovv(i,k,a,b)&
                                                +l1a(a,i)*vA_oovv(k,j,c,b)&
                                                +l1a(b,i)*vA_oovv(k,j,a,c)&
                                                +l1a(a,j)*vA_oovv(i,k,c,b)&
                                                +l1a(b,j)*vA_oovv(i,k,a,c)&
                                                +H1A_ov(k,c)*l2a(a,b,i,j)&
                                                -H1A_ov(k,a)*l2a(c,b,i,j)&
                                                -H1A_ov(k,b)*l2a(a,c,i,j)&
                                                -H1A_ov(i,c)*l2a(a,b,k,j)&
                                                -H1A_ov(j,c)*l2a(a,b,i,k)&
                                                +H1A_ov(i,a)*l2a(c,b,k,j)&
                                                +H1A_ov(i,b)*l2a(a,c,k,j)&
                                                +H1A_ov(j,a)*l2a(c,b,i,k)&
                                                +H1A_ov(j,b)*l2a(a,c,i,k)

                                                LM = temp1*(temp2+temp3)                                        
        
                                                D = fA_oo(i,i) + fA_oo(j,j) + fA_oo(k,k)&
                                                - fA_vv(a,a) - fA_vv(b,b) - fA_vv(c,c)

                                                deltaA = deltaA + LM/D

                                                D = H1A_oo(i,i) + H1A_oo(j,j) + H1A_oo(k,k)&
                                                - H1A_vv(a,a) - H1A_vv(b,b) - H1A_vv(c,c)

                                                deltaB = deltaB + LM/D

                                                D = D &
                                                -H2A_voov(a,i,i,a) - H2A_voov(b,i,i,b) - H2A_voov(c,i,i,c)&
                                                -H2A_voov(a,j,j,a) - H2A_voov(b,j,j,b) - H2A_voov(c,j,j,c)&
                                                -H2A_voov(a,k,k,a) - H2A_voov(b,k,k,b) - H2A_voov(c,k,k,c)&
                                                -H2A_oooo(j,i,j,i) - H2A_oooo(k,i,k,i) - H2A_oooo(k,j,k,j)&
                                                -H2A_vvvv(b,a,b,a) - H2A_vvvv(c,a,c,a) - H2A_vvvv(c,b,c,b)

                                                deltaC = deltaC + LM/D

                                                D = D &
                                                +D3A_O(a,i,j)+D3A_O(a,i,k)+D3A_O(a,j,k)&
                                                +D3A_O(b,i,j)+D3A_O(b,i,k)+D3A_O(b,j,k)&
                                                +D3A_O(c,i,j)+D3A_O(c,i,k)+D3A_O(c,j,k)&
                                                -D3A_V(a,i,b)-D3A_V(a,i,c)-D3A_V(b,i,c)&
                                                -D3A_V(a,j,b)-D3A_V(a,j,c)-D3A_V(b,j,c)&
                                                -D3A_V(a,k,b)-D3A_V(a,k,c)-D3A_V(b,k,c)

                                                deltaD = deltaD + LM/D

                                            end do
                                        end do 
                                    end do 

                                end do 
                            end do 
                        end do

              end subroutine crcc23A_opt

              subroutine crcc23B_opt(deltaA,deltaB,deltaC,deltaD,&
                              t2a,t2b,l1a,l1b,l2a,l2b,&
                              I2B_ovoo,I2B_vooo,I2A_vooo,&
                              H2B_vvvo,H2B_vvov,H2A_vvov,&
                              H2B_vovv,H2B_ovvv,H2A_vovv,&
                              H2B_ooov,H2B_oovo,H2A_ooov,&
                              H1A_ov,H1B_ov,&
                              vA_oovv,vB_oovv,&
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
                        real(kind=8), intent(in) :: t2a(nua,nua,noa,noa),t2b(nua,nub,noa,nob),&
                        l1a(nua,noa),l1b(nub,nob),&
                        l2a(nua,nua,noa,noa),l2b(nua,nub,noa,nob),&
                        I2B_ovoo(noa,nub,noa,nob),I2B_vooo(nua,nob,noa,nob),&
                        I2A_vooo(nua,noa,noa,noa),H2B_vvvo(nua,nub,nua,nob),&
                        H2B_vvov(nua,nub,noa,nub),H2A_vvov(nua,nua,noa,nua),&
                        H2B_vovv(nua,nob,nua,nub),H2B_ovvv(noa,nub,nua,nub),&
                        H2A_vovv(nua,noa,nua,nua),H2B_ooov(noa,nob,noa,nub),&
                        H2B_oovo(noa,nob,nua,nob),H2A_ooov(noa,noa,noa,nua),&
                        H1A_ov(noa,nua),H1B_ov(nob,nub),&
                        vA_oovv(noa,noa,nua,nua),vB_oovv(noa,nob,nua,nub),& 
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
                        D3C_V(1:nua,1:nob,1:nub)
                        integer :: i, j, k, a, b, c, i1, i2, i3, i4, nuanub, nua2
                        real(kind=8) :: D, temp1, temp2, temp3, LM, X3B(nua,nua,nub), L3B(nua,nua,nub)

                        ! arrays for reordering 
                        real(kind=8) :: t2a_1243(nua,nua,noa,noa), H2B_vvov_1243(nua,nub,nub,noa),&
                            t2b_1243(nua,nub,nob,noa), H2A_vvov_1243(nua,nua,nua,noa),&
                            H2B_vovv_1342(nua,nua,nub,nob), H2A_vovv_4312(nua,nua,nua,noa),&
                            H2B_ovvv_2341(nub,nua,nub,noa), H2B_ooov_3412(noa,nub,noa,nob),&
                            H2A_ooov_4312(nua,noa,noa,noa), H2B_oovo_3412(nua,nob,noa,nob),&
                            l2b_1243(nua,nub,nob,noa)

                        call reorder1243(t2a,t2a_1243)
                        call reorder1243(H2B_vvov,H2B_vvov_1243)
                        call reorder1243(t2b,t2b_1243)
                        call reorder1243(H2A_vvov,H2A_vvov_1243)
                        call reorder1342(H2B_vovv,H2B_vovv_1342)
                        call reorder4312(H2A_vovv,H2A_vovv_4312)                            
                        call reorder2341(H2B_ovvv,H2B_ovvv_2341)
                        call reorder3412(H2B_ooov,H2B_ooov_3412)
                        call reorder4312(H2A_ooov,H2A_ooov_4312)
                        call reorder3412(H2B_oovo,H2B_oovo_3412)
                        call reorder1243(l2b,l2b_1243)

                        deltaA = 0.0d0
                        deltaB = 0.0d0
                        deltaC = 0.0d0
                        deltaD = 0.0d0

                        nuanub = nua*nub
                        nua2 = nua*nua
                        do i = 1, noa
                            do j = i+1, noa
                                do k = 1, nob

                                X3B = 0.0d0
                                L3B = 0.0d0
                                !!!!! MM(2,3)B !!!!!
                                ! Diagram 1: A(ab) H2B(bcek)*t2a(aeij)
                                call dgemm('n','t',nua,nuanub,nua,1.0d0,t2a(:,:,i,j),nua,H2B_vvvo(:,:,:,k),nuanub,1.0d0,X3B,nua)
                                ! Diagram 2: -A(ij) I2B(mcjk)*t2a(abim)
                                call dgemm('n','n',nua2,nub,noa,0.5d0,t2a(:,:,:,i),nua2,I2B_ovoo(:,:,j,k),noa,1.0d0,X3B,nua2)
                                call dgemm('n','n',nua2,nub,noa,-0.5d0,t2a(:,:,:,j),nua2,I2B_ovoo(:,:,i,k),noa,1.0d0,X3B,nua2)
                                ! Diagram 3: A(ab)A(ij) H2B(acie)*t2b(bejk) -> A(ab)A(ij) t2b(aeik)*H2B(bcje)
                                call dgemm('n','t',nua,nuanub,nub,1.0d0,t2b(:,:,i,k),nua,H2B_vvov_1243(:,:,:,j),nuanub,1.0d0,X3B,nua)
                                call dgemm('n','t',nua,nuanub,nub,-1.0d0,t2b(:,:,j,k),nua,H2B_vvov_1243(:,:,:,i),nuanub,1.0d0,X3B,nua)
                                ! Diagram 4: -A(ab)A(ij) I2B(amik)*t2b(bcjm)
                                call dgemm('n','t',nua,nuanub,nob,-1.0d0,I2B_vooo(:,:,i,k),nua,t2b_1243(:,:,:,j),nuanub,1.0d0,X3B,nua)
                                call dgemm('n','t',nua,nuanub,nob,1.0d0,I2B_vooo(:,:,j,k),nua,t2b_1243(:,:,:,i),nuanub,1.0d0,X3B,nua)
                                ! Diagram 5: A(ij) H2A(abie)*t2b(ecjk)
                                call dgemm('n','n',nua2,nub,nua,0.5d0,H2A_vvov_1243(:,:,:,i),nua2,t2b(:,:,j,k),nua,1.0d0,X3B,nua2)
                                call dgemm('n','n',nua2,nub,nua,-0.5d0,H2A_vvov_1243(:,:,:,j),nua2,t2b(:,:,i,k),nua,1.0d0,X3B,nua2)
                                ! Diagram 6: -A(ab) I2A(amij)*t2b(bcmk)
                                call dgemm('n','t',nua,nuanub,noa,-1.0d0,I2A_vooo(:,:,i,j),nua,t2b(:,:,:,k),nuanub,1.0d0,X3B,nua)
                                !!!!! L3B !!!!!
                                ! Diagram 1: A(ab) H2B(ekbc)*l2a(aeij)
                                call dgemm('n','n',nua,nuanub,nua,1.0d0,l2a(:,:,i,j),nua,H2B_vovv_1342(:,:,:,k),nua,1.0d0,L3B,nua)
                                ! Diagram 2: A(ij) H2A(eiba)*l2b(ecjk)
                                call dgemm('n','n',nua2,nub,nua,0.5d0,H2A_vovv_4312(:,:,:,i),nua2,l2b(:,:,j,k),nua,1.0d0,L3B,nua2)
                                call dgemm('n','n',nua2,nub,nua,-0.5d0,H2A_vovv_4312(:,:,:,j),nua2,l2b(:,:,i,k),nua,1.0d0,L3B,nua2)
                                ! Diagram 3: A(ij)A(ab) H2B(ieac)*l2b(bejk) -> l2b(aeik)*H2B(jebc)
                                call dgemm('n','n',nua,nuanub,nub,1.0d0,l2b(:,:,i,k),nua,H2B_ovvv_2341(:,:,:,j),nub,1.0d0,L3B,nua)
                                call dgemm('n','n',nua,nuanub,nub,-1.0d0,l2b(:,:,j,k),nua,H2B_ovvv_2341(:,:,:,i),nub,1.0d0,L3B,nua)
                                ! Diagram 4: -A(ij) H2B(jkmc)*l2a(abim) -> +A(ij) H2B(jkmc)*l2a(abmi)
                                call dgemm('n','n',nua2,nub,noa,0.5d0,l2a(:,:,:,i),nua2,H2B_ooov_3412(:,:,j,k),noa,1.0d0,L3B,nua2)
                                call dgemm('n','n',nua2,nub,noa,-0.5d0,l2a(:,:,:,j),nua2,H2B_ooov_3412(:,:,i,k),noa,1.0d0,L3B,nua2)
                                ! Diagram 5: -A(ab) H2A(jima)*l2b(bcmk)
                                call dgemm('n','t',nua,nuanub,noa,-1.0d0,H2A_ooov_4312(:,:,j,i),nua,l2b(:,:,:,k),nuanub,1.0d0,L3B,nua)
                                ! Diagram 6: -A(ij)A(ab) H2B(ikam)*l2b(bcjm)
                                call dgemm('n','t',nua,nuanub,nob,-1.0d0,H2B_oovo_3412(:,:,i,k),nua,l2b_1243(:,:,:,j),nuanub,1.0d0,L3B,nua)
                                call dgemm('n','t',nua,nuanub,nob,1.0d0,H2B_oovo_3412(:,:,j,k),nua,l2b_1243(:,:,:,i),nuanub,1.0d0,L3B,nua)

                                    do a = 1, nua
                                        do b = a+1, nua
                                            do c = 1, nub

                                                temp1 = X3B(a,b,c) - X3B(b,a,c)
                                                temp2 = L3B(a,b,c) - L3B(b,a,c)
                                                temp3 = l1a(a,i)*vB_oovv(j,k,b,c)&
                                                       -l1a(a,j)*vB_oovv(i,k,b,c)&
                                                       -l1a(b,i)*vB_oovv(j,k,a,c)&
                                                       +l1a(b,j)*vB_oovv(i,k,a,c)&
                                                       +l1b(c,k)*vA_oovv(i,j,a,b)&
                                                       +l2b(b,c,j,k)*H1A_ov(i,a)&
                                                       -l2b(b,c,i,k)*H1A_ov(j,a)&
                                                       -l2b(a,c,j,k)*H1A_ov(i,b)&
                                                       +l2b(a,c,i,k)*H1A_ov(j,b)&
                                                       +l2a(a,b,i,j)*H1B_ov(k,c)

                                                LM = temp1*(temp2+temp3)

                                                D = fA_oo(i,i) + fA_oo(j,j) + fB_oo(k,k)&
                                                - fA_vv(a,a) - fA_vv(b,b) - fB_vv(c,c)

                                                deltaA = deltaA + LM/D

                                                D = H1A_oo(i,i) + H1A_oo(j,j) + H1B_oo(k,k)&
                                                - H1A_vv(a,a) - H1A_vv(b,b) - H1B_vv(c,c)

                                                deltaB = deltaB + LM/D

                                                D = D &
                                                -H2A_voov(a,i,i,a)-H2A_voov(b,i,i,b)+H2B_ovov(i,c,i,c)&
                                                -H2A_voov(a,j,j,a)-H2A_voov(b,j,j,b)+H2B_ovov(j,c,j,c)&
                                                +H2B_vovo(a,k,a,k)+H2B_vovo(b,k,b,k)-H2C_voov(c,k,k,c)&
                                                -H2A_oooo(j,i,j,i)-H2B_oooo(i,k,i,k)-H2B_oooo(j,k,j,k)&
                                                -H2A_vvvv(b,a,b,a)-H2B_vvvv(a,c,a,c)-H2B_vvvv(b,c,b,c)
     
                                                deltaC = deltaC + LM/D

                                                D = D &
                                                +D3A_O(a,i,j)+D3B_O(a,i,k)+D3B_O(a,j,k)&
                                                +D3A_O(b,i,j)+D3B_O(b,i,k)+D3B_O(b,j,k)&
                                                +D3C_O(c,i,k)+D3C_O(c,j,k)&
                                                -D3A_V(a,i,b)-D3B_V(a,i,c)-D3B_V(b,i,c)&
                                                -D3A_V(a,j,b)-D3B_V(a,j,c)-D3B_V(b,j,c)&
                                                -D3C_V(a,k,c)-D3C_V(b,k,c)

                                                deltaD = deltaD + LM/D

                                            end do
                                        end do 
                                    end do 
                                end do 
                            end do 
                        end do


              end subroutine crcc23B_opt

              subroutine crcc23C_opt(deltaA,deltaB,deltaC,deltaD,&
                              t2b,t2c,l1a,l1b,l2b,l2c,&
                              I2B_vooo,I2C_vooo,I2B_ovoo,&
                              H2B_vvov,H2C_vvov,H2B_vvvo,&
                              H2B_ovvv,H2B_vovv,H2C_vovv,&
                              H2B_oovo,H2B_ooov,H2C_ooov,&
                              H1A_ov,H1B_ov,&
                              vB_oovv,vC_oovv,&
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
                        real(kind=8), intent(in) :: t2b(nua,nub,noa,nob),&
                        t2c(nub,nub,nob,nob),l1a(nua,noa),l1b(nub,nob),&
                        l2b(nua,nub,noa,nob),l2c(nub,nub,nob,nob),&
                        I2B_vooo(nua,nob,noa,nob),I2C_vooo(nub,nob,nob,nob),&
                        I2B_ovoo(noa,nub,noa,nob),H2B_vvov(nua,nub,noa,nub),&
                        H2C_vvov(nub,nub,nob,nub),H2B_vvvo(nua,nub,nua,nob),&
                        H2B_ovvv(noa,nub,nua,nub),H2B_vovv(nua,nob,nua,nub),&
                        H2C_vovv(nub,nob,nub,nub),H2B_oovo(noa,nob,nua,nob),&
                        H2B_ooov(noa,nob,noa,nub),H2C_ooov(nob,nob,nob,nub),&
                        H1A_ov(noa,nua),H1B_ov(nob,nub),&
                        vB_oovv(noa,nob,nua,nub),vC_oovv(nob,nob,nub,nub),& 
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
                        D3D_V(1:nub,1:nob,1:nub)
                        integer :: i, j, k, a, b, c, i1, i2, i3, i4, nuanub, nub2
                        real(kind=8) :: D, LM, temp1, temp2, temp3, X3C(nua,nub,nub), L3C(nua,nub,nub)

                        ! arrays for reordering
                        real(kind=8) :: H2B_vvov_1243(nua,nub,nub,noa),&
                                H2C_vvov_4213(nub,nub,nub,noa),&
                                t2b_1243(nua,nub,nob,noa),&
                                I2C_vooo_2134(nob,nub,nob,nob),&
                                H2B_ovvv_3421(nua,nub,nub,noa),&
                                H2C_vovv_1342(nub,nub,nub,nob),&
                                H2B_vovv_3412(nua,nub,nua,nob),&
                                H2B_oovo_3412(nua,nob,noa,nob),&
                                H2C_ooov_3412(nob,nub,nob,nob),&
                                l2b_1243(nua,nub,nob,noa),&
                                H2B_ooov_3412(noa,nub,noa,nob)

                        deltaA = 0.0d0
                        deltaB = 0.0d0
                        deltaC = 0.0d0
                        deltaD = 0.0d0
                        
                        nuanub = nua*nub
                        nub2 = nub*nub

                        call reorder1243(H2B_vvov,H2B_vvov_1243)
                        call reorder4213(H2C_vvov,H2C_vvov_4213)
                        call reorder1243(t2b,t2b_1243)
                        call reorder2134(I2C_vooo,I2C_vooo_2134)
                        call reorder3421(H2B_ovvv,H2B_ovvv_3421)
                        call reorder1342(H2C_vovv,H2C_vovv_1342)
                        call reorder3412(H2B_vovv,H2B_vovv_3412)
                        call reorder3412(H2B_oovo,H2B_oovo_3412)
                        call reorder3412(H2C_ooov,H2C_ooov_3412)
                        call reorder1243(l2b,l2b_1243)
                        call reorder3412(H2B_ooov,H2B_ooov_3412)

                        do i = 1 , noa
                            do j = 1, nob
                                do k = j+1, nob
                                X3C = 0.0d0
                                L3C = 0.0d0

                                !!!!! MM(2,3)C !!!!!
                                ! Diagram 1: A(bc) H2B_vvov(a,b,i,e)*t2c(e,c,j,k)
                                call dgemm('n','n',nuanub,nub,nub,1.0d0,H2B_vvov_1243(:,:,:,i),nuanub,t2c(:,:,j,k),nub,1.0d0,X3C,nuanub)
                                ! Diagram 2: -A(jk) I2B_vooo(a,m,i,j)*t2c(b,c,m,k)
                                call dgemm('n','t',nua,nub2,nob,-0.5d0,I2B_vooo(:,:,i,j),nua,t2c(:,:,:,k),nub2,1.0d0,X3C,nua)
                                call dgemm('n','t',nua,nub2,nob,0.5d0,I2B_vooo(:,:,i,k),nua,t2c(:,:,:,j),nub2,1.0d0,X3C,nua)
                                ! Diagram 3: A(jk) H2C_vvov(c,b,k,e)*t2b(a,e,i,j)
                                call dgemm('n','n',nua,nub2,nub,0.5d0,t2b(:,:,i,j),nua,H2C_vvov_4213(:,:,:,k),nub,1.0d0,X3C,nua)
                                call dgemm('n','n',nua,nub2,nub,-0.5d0,t2b(:,:,i,k),nua,H2C_vvov_4213(:,:,:,j),nub,1.0d0,X3C,nua)
                                ! Diagram 4: -A(bc) I2C_vooo(c,m,k,j)*t2b(a,b,i,m)
                                call dgemm('n','n',nuanub,nub,nob,-1.0d0,t2b_1243(:,:,:,i),nuanub,I2C_vooo_2134(:,:,k,j),nob,1.0d0,X3C,nuanub)
                                ! Diagram 5: A(jk)A(bc) H2B_vvvo(a,b,e,j)*t2b(e,c,i,k)
                                call dgemm('n','n',nuanub,nub,nua,1.0d0,H2B_vvvo(:,:,:,j),nuanub,t2b(:,:,i,k),nua,1.0d0,X3C,nuanub)
                                call dgemm('n','n',nuanub,nub,nua,-1.0d0,H2B_vvvo(:,:,:,k),nuanub,t2b(:,:,i,j),nua,1.0d0,X3C,nuanub)
                                ! Diagram 6: -A(jk)A(bc) I2B_ovoo(m,b,i,j)*t2b(a,c,m,k) -> -A(jk)A(bc) I2B_ovoo(m,c,i,k)*t2b(a,b,m,j)
                                call dgemm('n','n',nuanub,nub,noa,-1.0d0,t2b(:,:,:,j),nuanub,I2B_ovoo(:,:,i,k),noa,1.0d0,X3C,nuanub)
                                call dgemm('n','n',nuanub,nub,noa,1.0d0,t2b(:,:,:,k),nuanub,I2B_ovoo(:,:,i,j),noa,1.0d0,X3C,nuanub)

                                !!!!! L3C !!!!!
                                ! Diagram 1: A(bc) H2B_ovvv(i,e,a,b)*l2c(e,c,j,k)
                                call dgemm('n','n',nuanub,nub,nub,1.0d0,H2B_ovvv_3421(:,:,:,i),nuanub,l2c(:,:,j,k),nub,1.0d0,L3C,nuanub)
                                ! Diagram 2: A(jk) H2C_vovv(e,k,b,c)*l2b(a,e,i,j)
                                call dgemm('n','n',nua,nub2,nub,0.5d0,l2b(:,:,i,j),nua,H2C_vovv_1342(:,:,:,k),nub,1.0d0,L3C,nua)
                                call dgemm('n','n',nua,nub2,nub,-0.5d0,l2b(:,:,i,k),nua,H2C_vovv_1342(:,:,:,j),nub,1.0d0,L3C,nua)
                                ! Diagram 3: A(jk)A(bc) H2B_vovv(e,j,a,b)*l2b(e,c,i,k)
                                call dgemm('n','n',nuanub,nub,nua,1.0d0,H2B_vovv_3412(:,:,:,j),nuanub,l2b(:,:,i,k),nua,1.0d0,L3C,nuanub)
                                call dgemm('n','n',nuanub,nub,nua,-1.0d0,H2B_vovv_3412(:,:,:,k),nuanub,l2b(:,:,i,j),nua,1.0d0,L3C,nuanub)
                                ! Diagram 4: -A(jk) H2B_oovo(i,j,a,m)*l2c(b,c,m,k)
                                call dgemm('n','t',nua,nub2,nob,-0.5d0,H2B_oovo_3412(:,:,i,j),nua,l2c(:,:,:,k),nub2,1.0d0,L3C,nua)
                                call dgemm('n','t',nua,nub2,nob,0.5d0,H2B_oovo_3412(:,:,i,k),nua,l2c(:,:,:,j),nub2,1.0d0,L3C,nua)
                                ! Diagram 5: -A(bc) H2C_ooov(j,k,m,c)*l2b(a,b,i,m)
                                call dgemm('n','n',nuanub,nub,nob,-1.0d0,l2b_1243(:,:,:,i),nuanub,H2C_ooov_3412(:,:,j,k),nob,1.0d0,L3C,nuanub)
                                ! Diagram 6: -A(jk)A(bc) H2B_ooov(i,j,m,b)*l2b(a,c,m,k) -> -A(jk)A(bc) H2B_ooov(i,k,m,c)*l2b(a,b,m,j)
                                call dgemm('n','n',nuanub,nub,noa,-1.0d0,l2b(:,:,:,j),nuanub,H2B_ooov_3412(:,:,i,k),noa,1.0d0,L3C,nuanub)
                                call dgemm('n','n',nuanub,nub,noa,1.0d0,l2b(:,:,:,k),nuanub,H2B_ooov_3412(:,:,i,j),noa,1.0d0,L3C,nuanub)
        

                                    do a = 1, nua
                                        do b = 1, nub
                                            do c = b+1, nub

                                                temp1 = X3C(a,b,c) - X3C(a,c,b)
                                                temp2 = L3C(a,b,c) - L3C(a,c,b)
                                                temp3 = l1b(c,k)*vB_oovv(i,j,a,b)&
                                                -l1b(b,k)*vB_oovv(i,j,a,c)&
                                                -l1b(c,j)*vB_oovv(i,k,a,b)&
                                                +l1b(b,j)*vB_oovv(i,k,a,c)&
                                                +l1a(a,i)*vC_oovv(j,k,b,c)&
                                                +H1B_ov(k,c)*l2b(a,b,i,j)&
                                                -H1B_ov(k,b)*l2b(a,c,i,j)&
                                                -H1B_ov(j,c)*l2b(a,b,i,k)&
                                                +H1B_ov(j,b)*l2b(a,c,i,k)&
                                                +H1A_ov(i,a)*l2c(b,c,j,k)

                                                LM = temp1*(temp2+temp3)

                                                D = fA_oo(i,i) + fB_oo(j,j) + fB_oo(k,k)&
                                                - fA_vv(a,a) - fB_vv(b,b) - fB_vv(c,c)

                                                deltaA = deltaA + LM/D

                                                D = H1A_oo(i,i) + H1B_oo(j,j) + H1B_oo(k,k)&
                                                - H1A_vv(a,a) - H1B_vv(b,b) - H1B_vv(c,c)

                                                deltaB = deltaB + LM/D

                                                D = D &
                                                -H2A_voov(a,i,i,a)+H2B_ovov(i,b,i,b)+H2B_ovov(i,c,i,c)&
                                                +H2B_vovo(a,j,a,j)-H2C_voov(b,j,j,b)-H2C_voov(c,j,j,c)&
                                                +H2B_vovo(a,k,a,k)-H2C_voov(b,k,k,b)-H2C_voov(c,k,k,c)&
                                                -H2B_oooo(i,j,i,j)-H2B_oooo(i,k,i,k)-H2C_oooo(k,j,k,j)&
                                                -H2B_vvvv(a,b,a,b)-H2B_vvvv(a,c,a,c)-H2C_vvvv(c,b,c,b)
     
                                                deltaC = deltaC + LM/D
                                                D = D &
                                                +D3B_O(a,i,j)+D3B_O(a,i,k)&
                                                +D3C_O(b,i,j)+D3C_O(b,i,k)+D3D_O(b,j,k)&
                                                +D3C_O(c,i,j)+D3C_O(c,i,k)+D3D_O(c,j,k)&
                                                -D3B_V(a,i,b)-D3B_V(a,i,c)&
                                                -D3C_V(a,j,b)-D3C_V(a,j,c)-D3D_V(b,j,c)&
                                                -D3C_V(a,k,b)-D3C_V(a,k,c)-D3D_V(b,k,c)

                                                deltaD = deltaD + LM/D

                                            end do
                                        end do 
                                    end do 
                                end do 
                            end do 
                        end do

              end subroutine crcc23C_opt

              subroutine crcc23D_opt(deltaA,deltaB,deltaC,deltaD,&
                              t2c,l1b,l2c,&
                              H2C_vooo,I2C_vvov,vC_oovv,H1B_ov,H2C_vovv,H2C_ooov,fB_oo,fB_vv,&
                              H1B_oo,H1B_vv,&
                              H2C_voov,H2C_oooo,H2C_vvvv,&
                              D3D_O,D3D_V,nob,nub)

                        real(kind=8), intent(out) :: deltaA, deltaB, deltaC, deltaD
                        integer, intent(in) :: nob, nub
                        real(kind=8), intent(in) :: fB_oo(1:nob,1:nob),fB_vv(1:nub,1:nub),&
                        H1B_oo(1:nob,1:nob),H1B_vv(1:nub,1:nub),&
                        H2C_voov(1:nub,1:nob,1:nob,1:nub),&
                        H2C_oooo(1:nob,1:nob,1:nob,1:nob),&
                        H2C_vvvv(1:nub,1:nub,1:nub,1:nub),&
                        D3D_O(1:nub,1:nob,1:nob),&
                        D3D_V(1:nub,1:nob,1:nub),&
                        H2C_vooo(nub,nob,nob,nob),I2C_vvov(nub,nub,nob,nub),t2c(nub,nub,nob,nob),&
                        l1b(nub,nob),l2c(nub,nub,nob,nob),vC_oovv(nob,nob,nub,nub),&
                        H1B_ov(nob,nub),H2C_vovv(nub,nob,nub,nub),H2C_ooov(nob,nob,nob,nub)
                        integer :: i, j, k, a, b, c, nub2
                        real(kind=8) :: D, temp1, temp2, temp3, LM, X3D(nub,nub,nub), L3D(nub,nub,nub)

                        ! reordered arrays for DGEMMs
                        real(kind=8) :: I2C_vvov_1243(nub,nub,nub,nob), H2C_vovv_4312(nub,nub,nub,nob), H2C_ooov_4312(nub,nob,nob,nob)

                        call reorder1243(I2C_vvov,I2C_vvov_1243)
                        call reorder4312(H2C_vovv,H2C_vovv_4312)
                        call reorder4312(H2C_ooov,H2C_ooov_4312)

                        deltaA = 0.0d0
                        deltaB = 0.0d0
                        deltaC = 0.0d0
                        deltaD = 0.0d0

                        nub2 = nub*nub
                        do i = 1 , nob
                            do j = i+1, nob
                                do k = j+1, nob

                                   X3D = 0.0d0
                                   L3D = 0.0d0
                                   !!!!! MM(2,3)D !!!!!
                                   ! Diagram 1: -A(k/ij)A(a/bc) H2C_vooo(a,m,i,j)*t2c(b,c,m,k)
                                   call dgemm('n','t',nub,nub2,nob,-0.5d0,H2C_vooo(:,:,i,j),nub,t2c(:,:,:,k),nub2,1.0d0,X3D,nub)
                                   call dgemm('n','t',nub,nub2,nob,0.5d0,H2C_vooo(:,:,k,j),nub,t2c(:,:,:,i),nub2,1.0d0,X3D,nub)
                                   call dgemm('n','t',nub,nub2,nob,0.5d0,H2C_vooo(:,:,i,k),nub,t2c(:,:,:,j),nub2,1.0d0,X3D,nub)
                                   ! Diagram 2: A(i/jk)A(c/ab) I2C_vvov(a,b,i,e)*t2c(e,c,j,k)
                                   call dgemm('n','n',nub2,nub,nub,0.5d0,I2C_vvov_1243(:,:,:,i),nub2,t2c(:,:,j,k),nub,1.0d0,X3D,nub2)
                                   call dgemm('n','n',nub2,nub,nub,-0.5d0,I2C_vvov_1243(:,:,:,j),nub2,t2c(:,:,i,k),nub,1.0d0,X3D,nub2)
                                   call dgemm('n','n',nub2,nub,nub,-0.5d0,I2C_vvov_1243(:,:,:,k),nub2,t2c(:,:,j,i),nub,1.0d0,X3D,nub2)
                                   !!!!! L3A !!!!!
                                   ! Diagram 1: A(i/jk)A(c/ab) H2C_vovv(e,i,b,a)*l2c(e,c,j,k)
                                   call dgemm('n','n',nub2,nub,nub,0.5d0,H2C_vovv_4312(:,:,:,i),nub2,l2c(:,:,j,k),nub,1.0d0,L3D,nub2)                         
                                   call dgemm('n','n',nub2,nub,nub,-0.5d0,H2C_vovv_4312(:,:,:,j),nub2,l2c(:,:,i,k),nub,1.0d0,L3D,nub2)                         
                                   call dgemm('n','n',nub2,nub,nub,-0.5d0,H2C_vovv_4312(:,:,:,k),nub2,l2c(:,:,j,i),nub,1.0d0,L3D,nub2)
                                   ! Diagram 2: -A(k/ij)A(a/bc) H2A_ooov(j,i,m,a)*l2a(b,c,m,k)-> a,m,j,i * (b,c,m,k)'
                                   call dgemm('n','t',nub,nub2,nob,-0.5d0,H2C_ooov_4312(:,:,j,i),nub,l2c(:,:,:,k),nub2,1.0d0,L3D,nub)
                                   call dgemm('n','t',nub,nub2,nob,0.5d0,H2C_ooov_4312(:,:,k,i),nub,l2c(:,:,:,j),nub2,1.0d0,L3D,nub)
                                   call dgemm('n','t',nub,nub2,nob,0.5d0,H2C_ooov_4312(:,:,j,k),nub,l2c(:,:,:,i),nub2,1.0d0,L3D,nub)

                                    do a = 1, nub
                                        do b = a+1, nub
                                            do c = b+1, nub

                                                temp1 = X3D(a,b,c) + X3D(b,c,a) + X3D(c,a,b)&
                                                - X3D(a,c,b) - X3D(b,a,c) - X3D(c,b,a)

                                                temp2 = L3D(a,b,c) + L3D(b,c,a) + L3D(c,a,b)&
                                                - L3D(a,c,b) - L3D(b,a,c) - L3D(c,b,a)

                                                temp3 =&
                                                l1b(c,k)*vC_oovv(i,j,a,b)&
                                                -l1b(a,k)*vC_oovv(i,j,c,b)&
                                                -l1b(b,k)*vC_oovv(i,j,a,c)&
                                                -l1b(c,i)*vC_oovv(k,j,a,b)&
                                                -l1b(c,j)*vC_oovv(i,k,a,b)&
                                                +l1b(a,i)*vC_oovv(k,j,c,b)&
                                                +l1b(b,i)*vC_oovv(k,j,a,c)&
                                                +l1b(a,j)*vC_oovv(i,k,c,b)&
                                                +l1b(b,j)*vC_oovv(i,k,a,c)&
                                                +H1B_ov(k,c)*l2c(a,b,i,j)&
                                                -H1B_ov(k,a)*l2c(c,b,i,j)&
                                                -H1B_ov(k,b)*l2c(a,c,i,j)&
                                                -H1B_ov(i,c)*l2c(a,b,k,j)&
                                                -H1B_ov(j,c)*l2c(a,b,i,k)&
                                                +H1B_ov(i,a)*l2c(c,b,k,j)&
                                                +H1B_ov(i,b)*l2c(a,c,k,j)&
                                                +H1B_ov(j,a)*l2c(c,b,i,k)&
                                                +H1b_ov(j,b)*l2c(a,c,i,k)

                                                LM = temp1*(temp2+temp3)                                        
        
                                                D = fB_oo(i,i) + fB_oo(j,j) + fB_oo(k,k)&
                                                - fB_vv(a,a) - fB_vv(b,b) - fB_vv(c,c)

                                                deltaA = deltaA + LM/D

                                                D = H1B_oo(i,i) + H1B_oo(j,j) + H1B_oo(k,k)&
                                                - H1B_vv(a,a) - H1B_vv(b,b) - H1B_vv(c,c)

                                                deltaB = deltaB + LM/D

                                                D = D &
                                                -H2C_voov(a,i,i,a) - H2C_voov(b,i,i,b) - H2C_voov(c,i,i,c)&
                                                -H2C_voov(a,j,j,a) - H2C_voov(b,j,j,b) - H2C_voov(c,j,j,c)&
                                                -H2C_voov(a,k,k,a) - H2C_voov(b,k,k,b) - H2C_voov(c,k,k,c)&
                                                -H2C_oooo(j,i,j,i) - H2C_oooo(k,i,k,i) - H2C_oooo(k,j,k,j)&
                                                -H2C_vvvv(b,a,b,a) - H2C_vvvv(c,a,c,a) - H2C_vvvv(c,b,c,b)

                                                deltaC = deltaC + LM/D

                                                D = D &
                                                +D3D_O(a,i,j)+D3D_O(a,i,k)+D3D_O(a,j,k)&
                                                +D3D_O(b,i,j)+D3D_O(b,i,k)+D3D_O(b,j,k)&
                                                +D3D_O(c,i,j)+D3D_O(c,i,k)+D3D_O(c,j,k)&
                                                -D3D_V(a,i,b)-D3D_V(a,i,c)-D3D_V(b,i,c)&
                                                -D3D_V(a,j,b)-D3D_V(a,j,c)-D3D_V(b,j,c)&
                                                -D3D_V(a,k,b)-D3D_V(a,k,c)-D3D_V(b,k,c)

                                                deltaD = deltaD + LM/D

                                            end do
                                        end do 
                                    end do 

                                end do 
                            end do 
                        end do

              end subroutine crcc23D_opt

              subroutine creomcc23A_opt(deltaA,deltaB,deltaC,deltaD,&
                              omega,r0,t2a,r2a,l1a,l2a,&
                              H2A_vooo,I2A_vvov,H2A_vvov,&
                              chi2A_vvvo,chi2A_ovoo,&
                              vA_oovv,&
                              H1A_ov,&
                              H2A_vovv,H2A_ooov,&
                              fA_oo,fA_vv,&
                              H1A_oo,H1A_vv,&
                              H2A_voov,H2A_oooo,H2A_vvvv,&
                              D3A_O,D3A_v,noa,nua)

                        real(kind=8), intent(out) :: deltaA, deltaB, deltaC, deltaD
                        integer, intent(in) :: noa, nua
                        real(kind=8), intent(in) :: fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),&
                        H1A_oo(1:noa,1:noa),H1A_vv(1:nua,1:nua),&
                        H2A_voov(1:nua,1:noa,1:noa,1:nua),&
                        H2A_oooo(1:noa,1:noa,1:noa,1:noa),&
                        H2A_vvvv(1:nua,1:nua,1:nua,1:nua),&
                        D3A_O(1:nua,1:noa,1:noa),&
                        D3A_V(1:nua,1:noa,1:nua),&
                        H2A_vooo(nua,noa,noa,noa),I2A_vvov(nua,nua,noa,nua),t2a(nua,nua,noa,noa),&
                        l1a(nua,noa),l2a(nua,nua,noa,noa),vA_oovv(noa,noa,nua,nua),&
                        H1A_ov(noa,nua),H2A_vovv(nua,noa,nua,nua),H2A_ooov(noa,noa,noa,nua),&
                        H2A_vvov(nua,nua,noa,nua),r2a(nua,nua,noa,noa),&
                        chi2A_vvvo(nua,nua,nua,noa),chi2A_ovoo(noa,nua,noa,noa)
                        real(kind=8), intent(in) :: r0, omega

                        integer :: i, j, k, a, b, c, nua2
                        real(kind=8) :: D, temp1, temp2, temp3, temp4, LM,&
                                Y3A(nua,nua,nua), X3A(nua,nua,nua), L3A(nua,nua,nua)

                        ! reordered arrays for DGEMMs
                        real(kind=8) :: I2A_vvov_1243(nua,nua,nua,noa), H2A_vovv_4312(nua,nua,nua,noa),&
                                        H2A_ooov_4312(nua,noa,noa,noa), H2A_vvov_2143(nua,nua,nua,noa),&
                                        H2A_vooo_2143(noa,nua,noa,noa)

                        call reorder1243(I2A_vvov,I2A_vvov_1243)
                        call reorder4312(H2A_vovv,H2A_vovv_4312)
                        call reorder4312(H2A_ooov,H2A_ooov_4312)
                        call reorder2143(H2A_vvov,H2A_vvov_2143)
                        call reorder2143(H2A_vooo,H2A_vooo_2143)

                        deltaA = 0.0d0
                        deltaB = 0.0d0
                        deltaC = 0.0d0
                        deltaD = 0.0d0

                        nua2 = nua*nua
                        do i = 1 , noa
                            do j = i+1, noa
                                do k = j+1, noa

                                   X3A = 0.0d0
                                   Y3A = 0.0d0
                                   L3A = 0.0d0
                                   !!!!! EOMMM(2,3)A !!!!!
                                   ! Diagram 1: A(j/ik)A(c/ab) chi2A_vvvo(a,b,e,j)*t2a(e,c,i,k)
                                   call dgemm('n','n',nua2,nua,nua,0.5d0,chi2A_vvvo(:,:,:,j),nua2,t2a(:,:,i,k),nua,1.0d0,Y3A,nua2)
                                   call dgemm('n','n',nua2,nua,nua,-0.5d0,chi2A_vvvo(:,:,:,k),nua2,t2a(:,:,i,j),nua,1.0d0,Y3A,nua2)
                                   call dgemm('n','n',nua2,nua,nua,-0.5d0,chi2A_vvvo(:,:,:,i),nua2,t2a(:,:,j,k),nua,1.0d0,Y3A,nua2)
                                   ! Diagram 2: A(j/ik)A(c/ab) H2A_vvov(b,a,j,e)*r2a(e,c,i,k)
                                   call dgemm('n','n',nua2,nua,nua,0.5d0,H2A_vvov_2143(:,:,:,j),nua2,r2a(:,:,i,k),nua,1.0d0,Y3A,nua2)
                                   call dgemm('n','n',nua2,nua,nua,-0.5d0,H2A_vvov_2143(:,:,:,k),nua2,r2a(:,:,i,j),nua,1.0d0,Y3A,nua2)
                                   call dgemm('n','n',nua2,nua,nua,-0.5d0,H2A_vvov_2143(:,:,:,i),nua2,r2a(:,:,j,k),nua,1.0d0,Y3A,nua2)
                                   ! Diagram 3: -A(k/ij)A(b/ac) chi2A_ovoo(m,b,i,j)*t2a(a,c,m,k)
                                   call dgemm('n','n',nua2,nua,noa,-0.5d0,t2a(:,:,:,j),nua2,chi2A_ovoo(:,:,i,k),noa,1.0d0,Y3A,nua2)
                                   call dgemm('n','n',nua2,nua,noa,0.5d0,t2a(:,:,:,i),nua2,chi2A_ovoo(:,:,j,k),noa,1.0d0,Y3A,nua2)
                                   call dgemm('n','n',nua2,nua,noa,0.5d0,t2a(:,:,:,k),nua2,chi2A_ovoo(:,:,i,j),noa,1.0d0,Y3A,nua2)
                                   ! Diagram 4: -A(k/ij)A(b/ac) H2A_vooo(b,m,j,i)*r2a(a,c,m,k)
                                   call dgemm('n','n',nua2,nua,noa,-0.5d0,r2a(:,:,:,j),nua2,H2A_vooo_2143(:,:,i,k),noa,1.0d0,Y3A,nua2)
                                   call dgemm('n','n',nua2,nua,noa,0.5d0,r2a(:,:,:,i),nua2,H2A_vooo_2143(:,:,j,k),noa,1.0d0,Y3A,nua2)
                                   call dgemm('n','n',nua2,nua,noa,0.5d0,r2a(:,:,:,k),nua2,H2A_vooo_2143(:,:,i,j),noa,1.0d0,Y3A,nua2)
                                   !!!!! MM(2,3)A !!!!!
                                   ! Diagram 1: -A(k/ij)A(a/bc) I2A_vooo(a,m,i,j)*t2a(b,c,m,k)
                                   call dgemm('n','t',nua,nua2,noa,-0.5d0,H2A_vooo(:,:,i,j),nua,t2a(:,:,:,k),nua2,1.0d0,X3A,nua)
                                   call dgemm('n','t',nua,nua2,noa,0.5d0,H2A_vooo(:,:,k,j),nua,t2a(:,:,:,i),nua2,1.0d0,X3A,nua)
                                   call dgemm('n','t',nua,nua2,noa,0.5d0,H2A_vooo(:,:,i,k),nua,t2a(:,:,:,j),nua2,1.0d0,X3A,nua)
                                   ! Diagram 2: A(i/jk)A(c/ab) I2A_vvov(a,b,i,e)*t2a(e,c,j,k)
                                   call dgemm('n','n',nua2,nua,nua,0.5d0,I2A_vvov_1243(:,:,:,i),nua2,t2a(:,:,j,k),nua,1.0d0,X3A,nua2)
                                   call dgemm('n','n',nua2,nua,nua,-0.5d0,I2A_vvov_1243(:,:,:,j),nua2,t2a(:,:,i,k),nua,1.0d0,X3A,nua2)
                                   call dgemm('n','n',nua2,nua,nua,-0.5d0,I2A_vvov_1243(:,:,:,k),nua2,t2a(:,:,j,i),nua,1.0d0,X3A,nua2)
                                   !!!!! L3A !!!!!
                                   ! Diagram 1: A(i/jk)A(c/ab) H2A_vovv(e,i,b,a)*l2a(e,c,j,k)
                                   call dgemm('n','n',nua2,nua,nua,0.5d0,H2A_vovv_4312(:,:,:,i),nua2,l2a(:,:,j,k),nua,1.0d0,L3A,nua2)                         
                                   call dgemm('n','n',nua2,nua,nua,-0.5d0,H2A_vovv_4312(:,:,:,j),nua2,l2a(:,:,i,k),nua,1.0d0,L3A,nua2)                         
                                   call dgemm('n','n',nua2,nua,nua,-0.5d0,H2A_vovv_4312(:,:,:,k),nua2,l2a(:,:,j,i),nua,1.0d0,L3A,nua2)
                                   ! Diagram 2: -A(k/ij)A(a/bc) H2A_ooov(j,i,m,a)*l2a(b,c,m,k)-> a,m,j,i * (b,c,m,k)'
                                   call dgemm('n','t',nua,nua2,noa,-0.5d0,H2A_ooov_4312(:,:,j,i),nua,l2a(:,:,:,k),nua2,1.0d0,L3A,nua)
                                   call dgemm('n','t',nua,nua2,noa,0.5d0,H2A_ooov_4312(:,:,k,i),nua,l2a(:,:,:,j),nua2,1.0d0,L3A,nua)
                                   call dgemm('n','t',nua,nua2,noa,0.5d0,H2A_ooov_4312(:,:,j,k),nua,l2a(:,:,:,i),nua2,1.0d0,L3A,nua)

                                    do a = 1, nua
                                        do b = a+1, nua
                                            do c = b+1, nua

                                                temp1 = X3A(a,b,c) + X3A(b,c,a) + X3A(c,a,b)&
                                                - X3A(a,c,b) - X3A(b,a,c) - X3A(c,b,a)

                                                temp2 = L3A(a,b,c) + L3A(b,c,a) + L3A(c,a,b)&
                                                - L3A(a,c,b) - L3A(b,a,c) - L3A(c,b,a)

                                                temp3 =&
                                                l1a(c,k)*vA_oovv(i,j,a,b)&
                                                -l1a(a,k)*vA_oovv(i,j,c,b)&
                                                -l1a(b,k)*vA_oovv(i,j,a,c)&
                                                -l1a(c,i)*vA_oovv(k,j,a,b)&
                                                -l1a(c,j)*vA_oovv(i,k,a,b)&
                                                +l1a(a,i)*vA_oovv(k,j,c,b)&
                                                +l1a(b,i)*vA_oovv(k,j,a,c)&
                                                +l1a(a,j)*vA_oovv(i,k,c,b)&
                                                +l1a(b,j)*vA_oovv(i,k,a,c)&
                                                +H1A_ov(k,c)*l2a(a,b,i,j)&
                                                -H1A_ov(k,a)*l2a(c,b,i,j)&
                                                -H1A_ov(k,b)*l2a(a,c,i,j)&
                                                -H1A_ov(i,c)*l2a(a,b,k,j)&
                                                -H1A_ov(j,c)*l2a(a,b,i,k)&
                                                +H1A_ov(i,a)*l2a(c,b,k,j)&
                                                +H1A_ov(i,b)*l2a(a,c,k,j)&
                                                +H1A_ov(j,a)*l2a(c,b,i,k)&
                                                +H1A_ov(j,b)*l2a(a,c,i,k)

                                                temp4 = Y3A(a,b,c) + Y3A(b,c,a) + Y3A(c,a,b)&
                                                - Y3A(a,c,b) - Y3A(b,a,c) - Y3A(c,b,a)

                                                LM = (r0*temp1+temp4)*(temp2+temp3)                                        
        
                                                D = fA_oo(i,i) + fA_oo(j,j) + fA_oo(k,k)&
                                                - fA_vv(a,a) - fA_vv(b,b) - fA_vv(c,c)

                                                deltaA = deltaA + LM/(omega+D)

                                                D = H1A_oo(i,i) + H1A_oo(j,j) + H1A_oo(k,k)&
                                                - H1A_vv(a,a) - H1A_vv(b,b) - H1A_vv(c,c)

                                                deltaB = deltaB + LM/(omega+D)

                                                D = D &
                                                -H2A_voov(a,i,i,a) - H2A_voov(b,i,i,b) - H2A_voov(c,i,i,c)&
                                                -H2A_voov(a,j,j,a) - H2A_voov(b,j,j,b) - H2A_voov(c,j,j,c)&
                                                -H2A_voov(a,k,k,a) - H2A_voov(b,k,k,b) - H2A_voov(c,k,k,c)&
                                                -H2A_oooo(j,i,j,i) - H2A_oooo(k,i,k,i) - H2A_oooo(k,j,k,j)&
                                                -H2A_vvvv(b,a,b,a) - H2A_vvvv(c,a,c,a) - H2A_vvvv(c,b,c,b)

                                                deltaC = deltaC + LM/(omega+D)

                                                D = D &
                                                +D3A_O(a,i,j)+D3A_O(a,i,k)+D3A_O(a,j,k)&
                                                +D3A_O(b,i,j)+D3A_O(b,i,k)+D3A_O(b,j,k)&
                                                +D3A_O(c,i,j)+D3A_O(c,i,k)+D3A_O(c,j,k)&
                                                -D3A_V(a,i,b)-D3A_V(a,i,c)-D3A_V(b,i,c)&
                                                -D3A_V(a,j,b)-D3A_V(a,j,c)-D3A_V(b,j,c)&
                                                -D3A_V(a,k,b)-D3A_V(a,k,c)-D3A_V(b,k,c)

                                                deltaD = deltaD + LM/(omega+D)

                                            end do
                                        end do 
                                    end do 
                                end do 
                            end do 
                        end do

              end subroutine creomcc23A_opt

              subroutine creomcc23B_opt(deltaA,deltaB,deltaC,deltaD,&
                              omega,r0,&
                              t2a,t2b,r2a,r2b,l1a,l1b,l2a,l2b,&
                              I2B_ovoo,I2B_vooo,I2A_vooo,&
                              H2B_vvvo,H2B_vvov,H2A_vvov,&
                              H2B_vovv,H2B_ovvv,H2A_vovv,&
                              H2B_ooov,H2B_oovo,H2A_ooov,&
                              chi2B_vvvo,chi2B_ovoo,chi2A_vvvo,&
                              chi2A_vooo,chi2B_vvov,chi2B_vooo,&
                              H2B_ovoo,H2A_vooo,H2B_vooo,&
                              H1A_ov,H1B_ov,&
                              vA_oovv,vB_oovv,&
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
                        real(kind=8), intent(in) :: t2a(nua,nua,noa,noa),t2b(nua,nub,noa,nob),&
                        l1a(nua,noa),l1b(nub,nob),&
                        l2a(nua,nua,noa,noa),l2b(nua,nub,noa,nob),&
                        r2a(nua,nua,noa,noa),r2b(nua,nub,noa,nob),&
                        I2B_ovoo(noa,nub,noa,nob),I2B_vooo(nua,nob,noa,nob),&
                        I2A_vooo(nua,noa,noa,noa),H2B_vvvo(nua,nub,nua,nob),&
                        H2B_vvov(nua,nub,noa,nub),H2A_vvov(nua,nua,noa,nua),&
                        H2B_vovv(nua,nob,nua,nub),H2B_ovvv(noa,nub,nua,nub),&
                        H2A_vovv(nua,noa,nua,nua),H2B_ooov(noa,nob,noa,nub),&
                        H2B_oovo(noa,nob,nua,nob),H2A_ooov(noa,noa,noa,nua),&
                        chi2B_vvvo(nua,nub,nua,nob),chi2B_ovoo(noa,nub,noa,nob),&
                        chi2A_vvvo(nua,nua,nua,noa),chi2A_vooo(nua,noa,noa,noa),&
                        chi2B_vvov(nua,nub,noa,nub),chi2B_vooo(nua,nob,noa,nob),&
                        H2B_ovoo(noa,nub,noa,nob),H2A_vooo(nua,noa,noa,noa),H2B_vooo(nua,nob,noa,nob),&
                        H1A_ov(noa,nua),H1B_ov(nob,nub),&
                        vA_oovv(noa,noa,nua,nua),vB_oovv(noa,nob,nua,nub),& 
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
                        D3C_V(1:nua,1:nob,1:nub)
                        real(kind=8), intent(in) :: omega, r0

                        integer :: i, j, k, a, b, c, nuanub, nua2
                        real(kind=8) :: D, temp1, temp2, temp3, temp4, LM,&
                                X3B(nua,nua,nub), L3B(nua,nua,nub), Y3B(nua,nua,nub)

                        ! arrays for reordering 
                        real(kind=8) :: t2a_1243(nua,nua,noa,noa), H2B_vvov_1243(nua,nub,nub,noa),&
                            t2b_1243(nua,nub,nob,noa), H2A_vvov_1243(nua,nua,nua,noa),&
                            H2B_vovv_1342(nua,nua,nub,nob), H2A_vovv_4312(nua,nua,nua,noa),&
                            H2B_ovvv_2341(nub,nua,nub,noa), H2B_ooov_3412(noa,nub,noa,nob),&
                            H2A_ooov_4312(nua,noa,noa,noa), H2B_oovo_3412(nua,nob,noa,nob),&
                            l2b_1243(nua,nub,nob,noa), H2A_vvov_2143(nua,nua,nua,noa),&
                            chi2B_vvov_4123(nub,nua,nub,noa),H2B_vvov_4123(nub,nua,nub,noa),&
                            r2b_1243(nua,nub,nob,noa)

                        call reorder1243(t2a,t2a_1243)
                        call reorder1243(H2B_vvov,H2B_vvov_1243)
                        call reorder1243(t2b,t2b_1243)
                        call reorder1243(H2A_vvov,H2A_vvov_1243)
                        call reorder1342(H2B_vovv,H2B_vovv_1342)
                        call reorder4312(H2A_vovv,H2A_vovv_4312)                            
                        call reorder2341(H2B_ovvv,H2B_ovvv_2341)
                        call reorder3412(H2B_ooov,H2B_ooov_3412)
                        call reorder4312(H2A_ooov,H2A_ooov_4312)
                        call reorder3412(H2B_oovo,H2B_oovo_3412)
                        call reorder1243(l2b,l2b_1243)
                        call reorder2143(H2A_vvov,H2A_vvov_2143)
                        call reorder4123(chi2B_vvov,chi2B_vvov_4123)
                        call reorder4123(H2B_vvov,H2B_vvov_4123)
                        call reorder1243(r2b,r2b_1243)

                        deltaA = 0.0d0
                        deltaB = 0.0d0
                        deltaC = 0.0d0
                        deltaD = 0.0d0

                        nuanub = nua*nub
                        nua2 = nua*nua
                        do i = 1, noa
                            do j = i+1, noa
                                do k = 1, nob

                                X3B = 0.0d0
                                L3B = 0.0d0
                                Y3B = 0.0d0
                                !!!!! EOMMM(2,3)B !!!!!
                                ! Diagram 1: A(ab) chi2B_vvvo(b,c,e,k)*t2a(a,e,i,j)
                                call dgemm('n','t',nua,nuanub,nua,1.0d0,t2a(:,:,i,j),nua,chi2B_vvvo(:,:,:,k),nuanub,1.0d0,Y3B,nua)
                                ! Diagram 2: A(ab) H2B_vvvo(b,c,e,k)*r2a(a,e,i,j)
                                call dgemm('n','t',nua,nuanub,nua,1.0d0,r2a(:,:,i,j),nua,H2B_vvvo(:,:,:,k),nuanub,1.0d0,Y3B,nua)
                                ! Diagram 3: -A(ij) chi2B_ovoo(n,c,j,k)*t2a(a,b,i,n)
                                call dgemm('n','n',nua2,nub,noa,0.5d0,t2a(:,:,:,i),nua2,chi2B_ovoo(:,:,j,k),noa,1.0d0,Y3B,nua2)
                                call dgemm('n','n',nua2,nub,noa,-0.5d0,t2a(:,:,:,j),nua2,chi2B_ovoo(:,:,i,k),noa,1.0d0,Y3B,nua2)
                                ! Diagram 4: -A(ij) H2B_ovoo(n,c,j,k)*r2a(a,b,i,n)
                                call dgemm('n','n',nua2,nub,noa,0.5d0,r2a(:,:,:,i),nua2,H2B_ovoo(:,:,j,k),noa,1.0d0,Y3B,nua2)
                                call dgemm('n','n',nua2,nub,noa,-0.5d0,r2a(:,:,:,j),nua2,H2B_ovoo(:,:,i,k),noa,1.0d0,Y3B,nua2)
                                ! Diagram 5: A(ij) chi2A_vvvo(a,b,e,j)*t2b(e,c,i,k)
                                call dgemm('n','n',nua2,nub,nua,0.5d0,chi2A_vvvo(:,:,:,j),nua2,t2b(:,:,i,k),nua,1.0d0,Y3B,nua2)
                                call dgemm('n','n',nua2,nub,nua,-0.5d0,chi2A_vvvo(:,:,:,i),nua2,t2b(:,:,j,k),nua,1.0d0,Y3B,nua2)
                                ! Diagram 6: A(ij) H2A_vvov(b,a,j,e)*r2b(e,c,i,k)
                                call dgemm('n','n',nua2,nub,nua,0.5d0,H2A_vvov_2143(:,:,:,j),nua2,r2b(:,:,i,k),nua,1.0d0,Y3B,nua2)
                                call dgemm('n','n',nua2,nub,nua,-0.5d0,H2A_vvov_2143(:,:,:,i),nua2,r2b(:,:,j,k),nua,1.0d0,Y3B,nua2)
                                ! Diagram 7: -A(ab) chi2A_vooo(b,n,j,i)*t2b(a,c,n,k) -> -A(ab) chi2A_vooo(a,n,i,j)*t2b(b,c,n,k)
                                call dgemm('n','t',nua,nuanub,noa,-1.0d0,chi2A_vooo(:,:,i,j),nua,t2b(:,:,:,k),nuanub,1.0d0,Y3B,nua)
                                ! Diagram 8: -A(ab) H2A_vooo(b,n,j,i)*r2b(a,c,n,k)
                                call dgemm('n','t',nua,nuanub,noa,-1.0d0,H2A_vooo(:,:,i,j),nua,r2b(:,:,:,k),nuanub,1.0d0,Y3B,nua)
                                ! Diagram 9: A(ij)A(ab) chi2B_vvov(b,c,j,e)*t2b(a,e,i,k)
                                call dgemm('n','n',nua,nuanub,nub,1.0d0,t2b(:,:,i,k),nua,chi2B_vvov_4123(:,:,:,j),nua,1.0d0,Y3B,nua)
                                call dgemm('n','n',nua,nuanub,nub,-1.0d0,t2b(:,:,j,k),nua,chi2B_vvov_4123(:,:,:,i),nua,1.0d0,Y3B,nua)
                                ! Diagram 10: A(ij)A(ab) H2B_vvov(b,c,j,e)*r2b(a,e,i,k)
                                call dgemm('n','n',nua,nuanub,nub,1.0d0,r2b(:,:,i,k),nua,H2B_vvov_4123(:,:,:,j),nua,1.0d0,Y3B,nua)
                                call dgemm('n','n',nua,nuanub,nub,-1.0d0,r2b(:,:,j,k),nua,H2B_vvov_4123(:,:,:,i),nua,1.0d0,Y3B,nua)
                                ! Diagram 11: -A(ij)A(ab) chi2B_vooo(b,n,j,k)*t2b(a,c,i,n) -> -A(ij)A(ab) chi2B_vooo(a,n,i,k)*t2b(b,c,j,n)
                                call dgemm('n','t',nua,nuanub,nob,-1.0d0,chi2B_vooo(:,:,i,k),nua,t2b_1243(:,:,:,j),nuanub,1.0d0,Y3B,nua)
                                call dgemm('n','t',nua,nuanub,nob,1.0d0,chi2B_vooo(:,:,j,k),nua,t2b_1243(:,:,:,i),nuanub,1.0d0,Y3B,nua)
                                ! Diagram 12: -A(ij)A(ab) H2B_vooo(b,n,j,k)*r2b(a,c,i,n)
                                call dgemm('n','t',nua,nuanub,nob,-1.0d0,H2B_vooo(:,:,i,k),nua,r2b_1243(:,:,:,j),nuanub,1.0d0,Y3B,nua)
                                call dgemm('n','t',nua,nuanub,nob,1.0d0,H2B_vooo(:,:,j,k),nua,r2b_1243(:,:,:,i),nuanub,1.0d0,Y3B,nua)
                                !!!!! MM(2,3)B !!!!!
                                ! Diagram 1: A(ab) H2B(bcek)*t2a(aeij)
                                call dgemm('n','t',nua,nuanub,nua,1.0d0,t2a(:,:,i,j),nua,H2B_vvvo(:,:,:,k),nuanub,1.0d0,X3B,nua)
                                ! Diagram 2: -A(ij) I2B(mcjk)*t2a(abim)
                                call dgemm('n','n',nua2,nub,noa,0.5d0,t2a(:,:,:,i),nua2,I2B_ovoo(:,:,j,k),noa,1.0d0,X3B,nua2)
                                call dgemm('n','n',nua2,nub,noa,-0.5d0,t2a(:,:,:,j),nua2,I2B_ovoo(:,:,i,k),noa,1.0d0,X3B,nua2)
                                ! Diagram 3: A(ab)A(ij) H2B(acie)*t2b(bejk) -> A(ab)A(ij) t2b(aeik)*H2B(bcje)
                                call dgemm('n','t',nua,nuanub,nub,1.0d0,t2b(:,:,i,k),nua,H2B_vvov_1243(:,:,:,j),nuanub,1.0d0,X3B,nua)
                                call dgemm('n','t',nua,nuanub,nub,-1.0d0,t2b(:,:,j,k),nua,H2B_vvov_1243(:,:,:,i),nuanub,1.0d0,X3B,nua)
                                ! Diagram 4: -A(ab)A(ij) I2B(amik)*t2b(bcjm)
                                call dgemm('n','t',nua,nuanub,nob,-1.0d0,I2B_vooo(:,:,i,k),nua,t2b_1243(:,:,:,j),nuanub,1.0d0,X3B,nua)
                                call dgemm('n','t',nua,nuanub,nob,1.0d0,I2B_vooo(:,:,j,k),nua,t2b_1243(:,:,:,i),nuanub,1.0d0,X3B,nua)
                                ! Diagram 5: A(ij) H2A(abie)*t2b(ecjk)
                                call dgemm('n','n',nua2,nub,nua,0.5d0,H2A_vvov_1243(:,:,:,i),nua2,t2b(:,:,j,k),nua,1.0d0,X3B,nua2)
                                call dgemm('n','n',nua2,nub,nua,-0.5d0,H2A_vvov_1243(:,:,:,j),nua2,t2b(:,:,i,k),nua,1.0d0,X3B,nua2)
                                ! Diagram 6: -A(ab) I2A(amij)*t2b(bcmk)
                                call dgemm('n','t',nua,nuanub,noa,-1.0d0,I2A_vooo(:,:,i,j),nua,t2b(:,:,:,k),nuanub,1.0d0,X3B,nua)
                                !!!!! L3B !!!!!
                                ! Diagram 1: A(ab) H2B(ekbc)*l2a(aeij)
                                call dgemm('n','n',nua,nuanub,nua,1.0d0,l2a(:,:,i,j),nua,H2B_vovv_1342(:,:,:,k),nua,1.0d0,L3B,nua)
                                ! Diagram 2: A(ij) H2A(eiba)*l2b(ecjk)
                                call dgemm('n','n',nua2,nub,nua,0.5d0,H2A_vovv_4312(:,:,:,i),nua2,l2b(:,:,j,k),nua,1.0d0,L3B,nua2)
                                call dgemm('n','n',nua2,nub,nua,-0.5d0,H2A_vovv_4312(:,:,:,j),nua2,l2b(:,:,i,k),nua,1.0d0,L3B,nua2)
                                ! Diagram 3: A(ij)A(ab) H2B(ieac)*l2b(bejk) -> l2b(aeik)*H2B(jebc)
                                call dgemm('n','n',nua,nuanub,nub,1.0d0,l2b(:,:,i,k),nua,H2B_ovvv_2341(:,:,:,j),nub,1.0d0,L3B,nua)
                                call dgemm('n','n',nua,nuanub,nub,-1.0d0,l2b(:,:,j,k),nua,H2B_ovvv_2341(:,:,:,i),nub,1.0d0,L3B,nua)
                                ! Diagram 4: -A(ij) H2B(jkmc)*l2a(abim) -> +A(ij) H2B(jkmc)*l2a(abmi)
                                call dgemm('n','n',nua2,nub,noa,0.5d0,l2a(:,:,:,i),nua2,H2B_ooov_3412(:,:,j,k),noa,1.0d0,L3B,nua2)
                                call dgemm('n','n',nua2,nub,noa,-0.5d0,l2a(:,:,:,j),nua2,H2B_ooov_3412(:,:,i,k),noa,1.0d0,L3B,nua2)
                                ! Diagram 5: -A(ab) H2A(jima)*l2b(bcmk)
                                call dgemm('n','t',nua,nuanub,noa,-1.0d0,H2A_ooov_4312(:,:,j,i),nua,l2b(:,:,:,k),nuanub,1.0d0,L3B,nua)
                                ! Diagram 6: -A(ij)A(ab) H2B(ikam)*l2b(bcjm)
                                call dgemm('n','t',nua,nuanub,nob,-1.0d0,H2B_oovo_3412(:,:,i,k),nua,l2b_1243(:,:,:,j),nuanub,1.0d0,L3B,nua)
                                call dgemm('n','t',nua,nuanub,nob,1.0d0,H2B_oovo_3412(:,:,j,k),nua,l2b_1243(:,:,:,i),nuanub,1.0d0,L3B,nua)

                                    do a = 1, nua
                                        do b = a+1, nua
                                            do c = 1, nub

                                                temp1 = X3B(a,b,c) - X3B(b,a,c)
                                                temp2 = L3B(a,b,c) - L3B(b,a,c)
                                                temp3 = l1a(a,i)*vB_oovv(j,k,b,c)&
                                                       -l1a(a,j)*vB_oovv(i,k,b,c)&
                                                       -l1a(b,i)*vB_oovv(j,k,a,c)&
                                                       +l1a(b,j)*vB_oovv(i,k,a,c)&
                                                       +l1b(c,k)*vA_oovv(i,j,a,b)&
                                                       +l2b(b,c,j,k)*H1A_ov(i,a)&
                                                       -l2b(b,c,i,k)*H1A_ov(j,a)&
                                                       -l2b(a,c,j,k)*H1A_ov(i,b)&
                                                       +l2b(a,c,i,k)*H1A_ov(j,b)&
                                                       +l2a(a,b,i,j)*H1B_ov(k,c)
                                                temp4 = Y3B(a,b,c) - Y3B(b,a,c)

                                                LM = (r0*temp1+temp4)*(temp2+temp3)

                                                D = fA_oo(i,i) + fA_oo(j,j) + fB_oo(k,k)&
                                                - fA_vv(a,a) - fA_vv(b,b) - fB_vv(c,c)

                                                deltaA = deltaA + LM/(omega+D)

                                                D = H1A_oo(i,i) + H1A_oo(j,j) + H1B_oo(k,k)&
                                                - H1A_vv(a,a) - H1A_vv(b,b) - H1B_vv(c,c)

                                                deltaB = deltaB + LM/(omega+D)

                                                D = D &
                                                -H2A_voov(a,i,i,a)-H2A_voov(b,i,i,b)+H2B_ovov(i,c,i,c)&
                                                -H2A_voov(a,j,j,a)-H2A_voov(b,j,j,b)+H2B_ovov(j,c,j,c)&
                                                +H2B_vovo(a,k,a,k)+H2B_vovo(b,k,b,k)-H2C_voov(c,k,k,c)&
                                                -H2A_oooo(j,i,j,i)-H2B_oooo(i,k,i,k)-H2B_oooo(j,k,j,k)&
                                                -H2A_vvvv(b,a,b,a)-H2B_vvvv(a,c,a,c)-H2B_vvvv(b,c,b,c)
     
                                                deltaC = deltaC + LM/(omega+D)

                                                D = D &
                                                +D3A_O(a,i,j)+D3B_O(a,i,k)+D3B_O(a,j,k)&
                                                +D3A_O(b,i,j)+D3B_O(b,i,k)+D3B_O(b,j,k)&
                                                +D3C_O(c,i,k)+D3C_O(c,j,k)&
                                                -D3A_V(a,i,b)-D3B_V(a,i,c)-D3B_V(b,i,c)&
                                                -D3A_V(a,j,b)-D3B_V(a,j,c)-D3B_V(b,j,c)&
                                                -D3C_V(a,k,c)-D3C_V(b,k,c)

                                                deltaD = deltaD + LM/(omega+D)

                                            end do
                                        end do 
                                    end do 
                                end do 
                            end do 
                        end do


              end subroutine creomcc23B_opt

              !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! OLD ROUTINES
              !!!!!!!!!!!!!!!!!!!!!!

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


end module crcc_loops
