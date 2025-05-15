module ccsdpt_loops

  use reorder, only: reorder_stripe

      implicit none

      contains

              !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! OPTIMIZED CCSD(T) ROUTINES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

              subroutine ccsdptA_opt(deltaA,&
                                      t1a,t2a,&
                                      vA_vooo,I2A_vvov,vA_oovv,fA_ov,vA_vovv,vA_ooov,fA_oo,fA_vv,&
                                      noa,nua)

                        real(kind=8), intent(out) :: deltaA
                        integer, intent(in) :: noa, nua
                        real(kind=8), intent(in) :: fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),&
                        vA_vooo(nua,noa,noa,noa),I2A_vvov(nua,nua,noa,nua),t1a(nua,noa),t2a(nua,nua,noa,noa),&
                        vA_oovv(noa,noa,nua,nua),&
                        fA_ov(noa,nua),vA_vovv(nua,noa,nua,nua),vA_ooov(noa,noa,noa,nua)
                        integer :: i, j, k, a, b, c, nua2
                        real(kind=8) :: D, temp1, temp2, temp3, LM, X3A(nua,nua,nua), L3A(nua,nua,nua)

                        ! reordered arrays for DGEMMs
                        real(kind=8) :: I2A_vvov_1243(nua,nua,nua,noa), vA_vovv_4312(nua,nua,nua,noa), vA_ooov_4312(nua,noa,noa,noa)

                        call reorder_stripe(4, shape(I2A_vvov), size(I2A_vvov), '1243', I2A_vvov, I2A_vvov_1243)
                        call reorder_stripe(4, shape(vA_vovv), size(vA_vovv), '4312', vA_vovv, vA_vovv_4312)
                        call reorder_stripe(4, shape(vA_ooov), size(vA_ooov), '4312', vA_ooov, vA_ooov_4312)

                        deltaA = 0.0d0

                        nua2 = nua*nua
                        do i = 1 , noa
                            do j = i+1, noa
                                do k = j+1, noa

                                   X3A = 0.0d0
                                   L3A = 0.0d0
                                   !!!!! MM(2,3)A -> V*T2 !!!!!
                                   ! Diagram 1: -A(k/ij)A(a/bc) I2A_vooo(a,m,i,j)*t2a(b,c,m,k)
                                   call dgemm('n','t',nua,nua2,noa,-0.5d0,vA_vooo(:,:,i,j),nua,t2a(:,:,:,k),nua2,1.0d0,X3A,nua)
                                   call dgemm('n','t',nua,nua2,noa,0.5d0,vA_vooo(:,:,k,j),nua,t2a(:,:,:,i),nua2,1.0d0,X3A,nua)
                                   call dgemm('n','t',nua,nua2,noa,0.5d0,vA_vooo(:,:,i,k),nua,t2a(:,:,:,j),nua2,1.0d0,X3A,nua)
                                   ! Diagram 2: A(i/jk)A(c/ab) I2A_vvov(a,b,i,e)*t2a(e,c,j,k)
                                   call dgemm('n','n',nua2,nua,nua,0.5d0,I2A_vvov_1243(:,:,:,i),nua2,t2a(:,:,j,k),nua,1.0d0,X3A,nua2)
                                   call dgemm('n','n',nua2,nua,nua,-0.5d0,I2A_vvov_1243(:,:,:,j),nua2,t2a(:,:,i,k),nua,1.0d0,X3A,nua2)
                                   call dgemm('n','n',nua2,nua,nua,-0.5d0,I2A_vvov_1243(:,:,:,k),nua2,t2a(:,:,j,i),nua,1.0d0,X3A,nua2)
                                   !!!!! L3A -> (T3[2])^+ !!!!!
                                   ! Diagram 1: A(i/jk)A(c/ab) H2A_vovv(e,i,b,a)*l2a(e,c,j,k)
                                   !call dgemm('n','n',nua2,nua,nua,0.5d0,vA_vovv_4312(:,:,:,i),nua2,t2a(:,:,j,k),nua,1.0d0,L3A,nua2)
                                   !call dgemm('n','n',nua2,nua,nua,-0.5d0,vA_vovv_4312(:,:,:,j),nua2,t2a(:,:,i,k),nua,1.0d0,L3A,nua2)
                                   !call dgemm('n','n',nua2,nua,nua,-0.5d0,vA_vovv_4312(:,:,:,k),nua2,t2a(:,:,j,i),nua,1.0d0,L3A,nua2)
                                   ! Diagram 2: -A(k/ij)A(a/bc) H2A_ooov(j,i,m,a)*l2a(b,c,m,k)-> a,m,j,i * (b,c,m,k)'
                                   !call dgemm('n','t',nua,nua2,noa,-0.5d0,vA_ooov_4312(:,:,j,i),nua,t2a(:,:,:,k),nua2,1.0d0,L3A,nua)
                                   !call dgemm('n','t',nua,nua2,noa,0.5d0,vA_ooov_4312(:,:,k,i),nua,t2a(:,:,:,j),nua2,1.0d0,L3A,nua)
                                   !call dgemm('n','t',nua,nua2,noa,0.5d0,vA_ooov_4312(:,:,j,k),nua,t2a(:,:,:,i),nua2,1.0d0,L3A,nua)

                                    do a = 1, nua
                                        do b = a+1, nua
                                            do c = b+1, nua

                                                temp1 = X3A(a,b,c) + X3A(b,c,a) + X3A(c,a,b)&
                                                - X3A(a,c,b) - X3A(b,a,c) - X3A(c,b,a)

                                                !temp2 = L3A(a,b,c) + L3A(b,c,a) + L3A(c,a,b)&
                                                !- L3A(a,c,b) - L3A(b,a,c) - L3A(c,b,a)

                                                temp3 =&
                                                t1a(c,k)*vA_oovv(i,j,a,b)&
                                                -t1a(a,k)*vA_oovv(i,j,c,b)&
                                                -t1a(b,k)*vA_oovv(i,j,a,c)&
                                                -t1a(c,i)*vA_oovv(k,j,a,b)&
                                                -t1a(c,j)*vA_oovv(i,k,a,b)&
                                                +t1a(a,i)*vA_oovv(k,j,c,b)&
                                                +t1a(b,i)*vA_oovv(k,j,a,c)&
                                                +t1a(a,j)*vA_oovv(i,k,c,b)&
                                                +t1a(b,j)*vA_oovv(i,k,a,c)
                                                ! These terms correspond to < [T2]^+ F_N T_3[2] >, which is 4th order for non-HF (i.e., ROHF) orbitals
                                                temp3 = temp3&
                                                +fA_ov(k,c)*t2a(a,b,i,j)&
                                                -fA_ov(k,a)*t2a(c,b,i,j)&
                                                -fA_ov(k,b)*t2a(a,c,i,j)&
                                                -fA_ov(i,c)*t2a(a,b,k,j)&
                                                -fA_ov(j,c)*t2a(a,b,i,k)&
                                                +fA_ov(i,a)*t2a(c,b,k,j)&
                                                +fA_ov(i,b)*t2a(a,c,k,j)&
                                                +fA_ov(j,a)*t2a(c,b,i,k)&
                                                +fA_ov(j,b)*t2a(a,c,i,k)

                                                LM = temp1*(temp1+temp3)

                                                D = fA_oo(i,i) + fA_oo(j,j) + fA_oo(k,k)&
                                                - fA_vv(a,a) - fA_vv(b,b) - fA_vv(c,c)

                                                deltaA = deltaA + LM/D

                                            end do
                                        end do
                                    end do

                                end do
                            end do
                        end do
              end subroutine ccsdptA_opt

              subroutine ccsdptB_opt(deltaA,&
                                     t1a,t1b,t2a,t2b,&
                                     I2B_ovoo,I2B_vooo,I2A_vooo,&
                                     vB_vvvo,vB_vvov,vA_vvov,&
                                     vB_vovv,vB_ovvv,vA_vovv,&
                                     vB_ooov,vB_oovo,vA_ooov,&
                                     fA_ov,fB_ov,&
                                     vA_oovv,vB_oovv,&
                                     fA_oo,fA_vv,fB_oo,fB_vv,&
                                     noa,nua,nob,nub)

                        real(kind=8), intent(out) :: deltaA
                        integer, intent(in) :: noa, nua, nob, nub
                        real(kind=8), intent(in) :: t2a(nua,nua,noa,noa),t2b(nua,nub,noa,nob),&
                        t1a(nua,noa),t1b(nub,nob),&
                        I2B_ovoo(noa,nub,noa,nob),I2B_vooo(nua,nob,noa,nob),&
                        I2A_vooo(nua,noa,noa,noa),vB_vvvo(nua,nub,nua,nob),&
                        vB_vvov(nua,nub,noa,nub),vA_vvov(nua,nua,noa,nua),&
                        vB_vovv(nua,nob,nua,nub),vB_ovvv(noa,nub,nua,nub),&
                        vA_vovv(nua,noa,nua,nua),vB_ooov(noa,nob,noa,nub),&
                        vB_oovo(noa,nob,nua,nob),vA_ooov(noa,noa,noa,nua),&
                        fA_ov(noa,nua),fB_ov(nob,nub),&
                        vA_oovv(noa,noa,nua,nua),vB_oovv(noa,nob,nua,nub),&
                        fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),&
                        fB_oo(1:nob,1:nob),fB_vv(1:nub,1:nub)
                        integer :: i, j, k, a, b, c, nuanub, nua2
                        real(kind=8) :: D, temp1, temp2, temp3, LM, X3B(nua,nua,nub), L3B(nua,nua,nub)

                        ! arrays for reordering
                        real(kind=8) :: t2a_1243(nua,nua,noa,noa), vB_vvov_1243(nua,nub,nub,noa),&
                            t2b_1243(nua,nub,nob,noa), vA_vvov_1243(nua,nua,nua,noa),&
                            vB_vovv_1342(nua,nua,nub,nob), vA_vovv_4312(nua,nua,nua,noa),&
                            vB_ovvv_2341(nub,nua,nub,noa), vB_ooov_3412(noa,nub,noa,nob),&
                            vA_ooov_4312(nua,noa,noa,noa), vB_oovo_3412(nua,nob,noa,nob)

                        call reorder_stripe(4, shape(t2a), size(t2a), '1243', t2a, t2a_1243)
                        call reorder_stripe(4, shape(vB_vvov), size(vB_vvov), '1243', vB_vvov, vB_vvov_1243)
                        call reorder_stripe(4, shape(t2b), size(t2b), '1243', t2b, t2b_1243)
                        call reorder_stripe(4, shape(vA_vvov), size(vA_vvov), '1243', vA_vvov, vA_vvov_1243)
                        call reorder_stripe(4, shape(vB_vovv), size(vB_vovv), '1342', vB_vovv, vB_vovv_1342)
                        call reorder_stripe(4, shape(vA_vovv), size(vA_vovv), '4312', vA_vovv, vA_vovv_4312)
                        call reorder_stripe(4, shape(vB_ovvv), size(vB_ovvv), '2341', vB_ovvv, vB_ovvv_2341)
                        call reorder_stripe(4, shape(vB_ooov), size(vB_ooov), '3412', vB_ooov, vB_ooov_3412)
                        call reorder_stripe(4, shape(vA_ooov), size(vA_ooov), '4312', vA_ooov, vA_ooov_4312)
                        call reorder_stripe(4, shape(vB_oovo), size(vB_oovo), '3412', vB_oovo, vB_oovo_3412)

                        deltaA = 0.0d0

                        nuanub = nua*nub
                        nua2 = nua*nua
                        do i = 1, noa
                            do j = i+1, noa
                                do k = 1, nob

                                    X3B = 0.0d0
                                    L3B = 0.0d0
                                    !!!!! MM(2,3)B -> V*T2 !!!!!
                                    ! Diagram 1: A(ab) H2B(bcek)*t2a(aeij)
                                    call dgemm('n','t',nua,nuanub,nua,1.0d0,t2a(:,:,i,j),nua,vB_vvvo(:,:,:,k),nuanub,1.0d0,X3B,nua)
                                    ! Diagram 2: -A(ij) I2B(mcjk)*t2a(abim)
                                    call dgemm('n','n',nua2,nub,noa,0.5d0,t2a(:,:,:,i),nua2,I2B_ovoo(:,:,j,k),noa,1.0d0,X3B,nua2)
                                    call dgemm('n','n',nua2,nub,noa,-0.5d0,t2a(:,:,:,j),nua2,I2B_ovoo(:,:,i,k),noa,1.0d0,X3B,nua2)
                                    ! Diagram 3: A(ab)A(ij) H2B(acie)*t2b(bejk) -> A(ab)A(ij) t2b(aeik)*H2B(bcje)
                                    call dgemm('n','t',nua,nuanub,nub,1.0d0,t2b(:,:,i,k),nua,vB_vvov_1243(:,:,:,j),nuanub,1.0d0,X3B,nua)
                                    call dgemm('n','t',nua,nuanub,nub,-1.0d0,t2b(:,:,j,k),nua,vB_vvov_1243(:,:,:,i),nuanub,1.0d0,X3B,nua)
                                    ! Diagram 4: -A(ab)A(ij) I2B(amik)*t2b(bcjm)
                                    call dgemm('n','t',nua,nuanub,nob,-1.0d0,I2B_vooo(:,:,i,k),nua,t2b_1243(:,:,:,j),nuanub,1.0d0,X3B,nua)
                                    call dgemm('n','t',nua,nuanub,nob,1.0d0,I2B_vooo(:,:,j,k),nua,t2b_1243(:,:,:,i),nuanub,1.0d0,X3B,nua)
                                    ! Diagram 5: A(ij) H2A(abie)*t2b(ecjk)
                                    call dgemm('n','n',nua2,nub,nua,0.5d0,vA_vvov_1243(:,:,:,i),nua2,t2b(:,:,j,k),nua,1.0d0,X3B,nua2)
                                    call dgemm('n','n',nua2,nub,nua,-0.5d0,vA_vvov_1243(:,:,:,j),nua2,t2b(:,:,i,k),nua,1.0d0,X3B,nua2)
                                    ! Diagram 6: -A(ab) I2A(amij)*t2b(bcmk)
                                    call dgemm('n','t',nua,nuanub,noa,-1.0d0,I2A_vooo(:,:,i,j),nua,t2b(:,:,:,k),nuanub,1.0d0,X3B,nua)
                                    !!!!! L3B -> (T3[2])^+ !!!!!
                                    ! Diagram 1: A(ab) H2B(ekbc)*l2a(aeij)
                                    !call dgemm('n','n',nua,nuanub,nua,1.0d0,t2a(:,:,i,j),nua,vB_vovv_1342(:,:,:,k),nua,1.0d0,L3B,nua)
                                    ! Diagram 2: A(ij) H2A(eiba)*l2b(ecjk)
                                    !call dgemm('n','n',nua2,nub,nua,0.5d0,vA_vovv_4312(:,:,:,i),nua2,t2b(:,:,j,k),nua,1.0d0,L3B,nua2)
                                    !call dgemm('n','n',nua2,nub,nua,-0.5d0,vA_vovv_4312(:,:,:,j),nua2,t2b(:,:,i,k),nua,1.0d0,L3B,nua2)
                                    ! Diagram 3: A(ij)A(ab) H2B(ieac)*l2b(bejk) -> l2b(aeik)*H2B(jebc)
                                    !call dgemm('n','n',nua,nuanub,nub,1.0d0,t2b(:,:,i,k),nua,vB_ovvv_2341(:,:,:,j),nub,1.0d0,L3B,nua)
                                    !call dgemm('n','n',nua,nuanub,nub,-1.0d0,t2b(:,:,j,k),nua,vB_ovvv_2341(:,:,:,i),nub,1.0d0,L3B,nua)
                                    ! Diagram 4: -A(ij) H2B(jkmc)*l2a(abim) -> +A(ij) H2B(jkmc)*l2a(abmi)
                                    !call dgemm('n','n',nua2,nub,noa,0.5d0,t2a(:,:,:,i),nua2,vB_ooov_3412(:,:,j,k),noa,1.0d0,L3B,nua2)
                                    !call dgemm('n','n',nua2,nub,noa,-0.5d0,t2a(:,:,:,j),nua2,vB_ooov_3412(:,:,i,k),noa,1.0d0,L3B,nua2)
                                    ! Diagram 5: -A(ab) H2A(jima)*l2b(bcmk)
                                    !call dgemm('n','t',nua,nuanub,noa,-1.0d0,vA_ooov_4312(:,:,j,i),nua,t2b(:,:,:,k),nuanub,1.0d0,L3B,nua)
                                    ! Diagram 6: -A(ij)A(ab) H2B(ikam)*l2b(bcjm)
                                    !call dgemm('n','t',nua,nuanub,nob,-1.0d0,vB_oovo_3412(:,:,i,k),nua,t2b_1243(:,:,:,j),nuanub,1.0d0,L3B,nua)
                                    !call dgemm('n','t',nua,nuanub,nob,1.0d0,vB_oovo_3412(:,:,j,k),nua,t2b_1243(:,:,:,i),nuanub,1.0d0,L3B,nua)

                                    do a = 1, nua
                                        do b = a+1, nua
                                            do c = 1, nub

                                                temp1 = X3B(a,b,c) - X3B(b,a,c)
                                                !temp2 = L3B(a,b,c) - L3B(b,a,c)
                                                temp3 = t1a(a,i)*vB_oovv(j,k,b,c)&
                                                       -t1a(a,j)*vB_oovv(i,k,b,c)&
                                                       -t1a(b,i)*vB_oovv(j,k,a,c)&
                                                       +t1a(b,j)*vB_oovv(i,k,a,c)&
                                                       +t1b(c,k)*vA_oovv(i,j,a,b)
                                               ! These terms correspond to < [T2]^+ F_N T_3[2] >, which is 4th order for non-HF (i.e., ROHF) orbitals
                                               temp3 = temp3 &
                                                       +t2b(b,c,j,k)*fA_ov(i,a)&
                                                       -t2b(b,c,i,k)*fA_ov(j,a)&
                                                       -t2b(a,c,j,k)*fA_ov(i,b)&
                                                       +t2b(a,c,i,k)*fA_ov(j,b)&
                                                       +t2a(a,b,i,j)*fB_ov(k,c)

                                                LM = temp1*(temp1+temp3)

                                                D = fA_oo(i,i) + fA_oo(j,j) + fB_oo(k,k)&
                                                - fA_vv(a,a) - fA_vv(b,b) - fB_vv(c,c)

                                                deltaA = deltaA + LM/D

                                            end do
                                        end do
                                    end do
                                end do
                            end do
                        end do
              end subroutine ccsdptB_opt

              subroutine ccsdptC_opt(deltaA,&
                              t1a,t1b,t2b,t2c,&
                              I2B_vooo,I2C_vooo,I2B_ovoo,&
                              vB_vvov,vC_vvov,vB_vvvo,&
                              vB_ovvv,vB_vovv,vC_vovv,&
                              vB_oovo,vB_ooov,vC_ooov,&
                              fA_ov,fB_ov,&
                              vB_oovv,vC_oovv,&
                              fA_oo,fA_vv,fB_oo,fB_vv,&
                              noa,nua,nob,nub)

                        real(kind=8), intent(out) :: deltaA
                        integer, intent(in) :: noa, nua, nob, nub
                        real(kind=8), intent(in) :: t2b(nua,nub,noa,nob),&
                        t2c(nub,nub,nob,nob),t1a(nua,noa),t1b(nub,nob),&
                        I2B_vooo(nua,nob,noa,nob),I2C_vooo(nub,nob,nob,nob),&
                        I2B_ovoo(noa,nub,noa,nob),vB_vvov(nua,nub,noa,nub),&
                        vC_vvov(nub,nub,nob,nub),vB_vvvo(nua,nub,nua,nob),&
                        vB_ovvv(noa,nub,nua,nub),vB_vovv(nua,nob,nua,nub),&
                        vC_vovv(nub,nob,nub,nub),vB_oovo(noa,nob,nua,nob),&
                        vB_ooov(noa,nob,noa,nub),vC_ooov(nob,nob,nob,nub),&
                        fA_ov(noa,nua),fB_ov(nob,nub),&
                        vB_oovv(noa,nob,nua,nub),vC_oovv(nob,nob,nub,nub),&
                        fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),&
                        fB_oo(1:nob,1:nob),fB_vv(1:nub,1:nub)
                        integer :: i, j, k, a, b, c, nuanub, nub2
                        real(kind=8) :: D, LM, temp1, temp2, temp3, X3C(nua,nub,nub), L3C(nua,nub,nub)

                        ! arrays for reordering
                        real(kind=8) :: vB_vvov_1243(nua,nub,nub,noa),&
                                vC_vvov_4213(nub,nub,nub,noa),&
                                t2b_1243(nua,nub,nob,noa),&
                                I2C_vooo_2134(nob,nub,nob,nob),&
                                vB_ovvv_3421(nua,nub,nub,noa),&
                                vC_vovv_1342(nub,nub,nub,nob),&
                                vB_vovv_3412(nua,nub,nua,nob),&
                                vB_oovo_3412(nua,nob,noa,nob),&
                                vC_ooov_3412(nob,nub,nob,nob),&
                                vB_ooov_3412(noa,nub,noa,nob)

                        deltaA = 0.0d0

                        nuanub = nua*nub
                        nub2 = nub*nub

                        call reorder_stripe(4, shape(vB_vvov), size(vB_vvov), '1243', vB_vvov, vB_vvov_1243)
                        call reorder_stripe(4, shape(vC_vvov), size(vC_vvov), '4213', vC_vvov, vC_vvov_4213)
                        call reorder_stripe(4, shape(t2b), size(t2b), '1243', t2b, t2b_1243)
                        call reorder_stripe(4, shape(I2C_vooo), size(I2C_vooo), '2134', I2C_vooo, I2C_vooo_2134)
                        call reorder_stripe(4, shape(vB_ovvv), size(vB_ovvv), '3421', vB_ovvv, vB_ovvv_3421)
                        call reorder_stripe(4, shape(vC_vovv), size(vC_vovv), '1342', vC_vovv, vC_vovv_1342)
                        call reorder_stripe(4, shape(vB_vovv), size(vB_vovv), '3412', vB_vovv, vB_vovv_3412)
                        call reorder_stripe(4, shape(vB_oovo), size(vB_oovo), '3412', vB_oovo, vB_oovo_3412)
                        call reorder_stripe(4, shape(vC_ooov), size(vC_ooov), '3412', vC_ooov, vC_ooov_3412)
                        call reorder_stripe(4, shape(vB_ooov), size(vB_ooov), '3412', vB_ooov, vB_ooov_3412)

                        do i = 1 , noa
                            do j = 1, nob
                                do k = j+1, nob
                                    X3C = 0.0d0
                                    L3C = 0.0d0

                                    !!!!! MM(2,3)C -> V*T2 !!!!!
                                    ! Diagram 1: A(bc) H2B_vvov(a,b,i,e)*t2c(e,c,j,k)
                                    call dgemm('n','n',nuanub,nub,nub,1.0d0,vB_vvov_1243(:,:,:,i),nuanub,t2c(:,:,j,k),nub,1.0d0,X3C,nuanub)
                                    ! Diagram 2: -A(jk) I2B_vooo(a,m,i,j)*t2c(b,c,m,k)
                                    call dgemm('n','t',nua,nub2,nob,-0.5d0,I2B_vooo(:,:,i,j),nua,t2c(:,:,:,k),nub2,1.0d0,X3C,nua)
                                    call dgemm('n','t',nua,nub2,nob,0.5d0,I2B_vooo(:,:,i,k),nua,t2c(:,:,:,j),nub2,1.0d0,X3C,nua)
                                    ! Diagram 3: A(jk) H2C_vvov(c,b,k,e)*t2b(a,e,i,j)
                                    call dgemm('n','n',nua,nub2,nub,0.5d0,t2b(:,:,i,j),nua,vC_vvov_4213(:,:,:,k),nub,1.0d0,X3C,nua)
                                    call dgemm('n','n',nua,nub2,nub,-0.5d0,t2b(:,:,i,k),nua,vC_vvov_4213(:,:,:,j),nub,1.0d0,X3C,nua)
                                    ! Diagram 4: -A(bc) I2C_vooo(c,m,k,j)*t2b(a,b,i,m)
                                    call dgemm('n','n',nuanub,nub,nob,-1.0d0,t2b_1243(:,:,:,i),nuanub,I2C_vooo_2134(:,:,k,j),nob,1.0d0,X3C,nuanub)
                                    ! Diagram 5: A(jk)A(bc) H2B_vvvo(a,b,e,j)*t2b(e,c,i,k)
                                    call dgemm('n','n',nuanub,nub,nua,1.0d0,vB_vvvo(:,:,:,j),nuanub,t2b(:,:,i,k),nua,1.0d0,X3C,nuanub)
                                    call dgemm('n','n',nuanub,nub,nua,-1.0d0,vB_vvvo(:,:,:,k),nuanub,t2b(:,:,i,j),nua,1.0d0,X3C,nuanub)
                                    ! Diagram 6: -A(jk)A(bc) I2B_ovoo(m,b,i,j)*t2b(a,c,m,k) -> -A(jk)A(bc) I2B_ovoo(m,c,i,k)*t2b(a,b,m,j)
                                    call dgemm('n','n',nuanub,nub,noa,-1.0d0,t2b(:,:,:,j),nuanub,I2B_ovoo(:,:,i,k),noa,1.0d0,X3C,nuanub)
                                    call dgemm('n','n',nuanub,nub,noa,1.0d0,t2b(:,:,:,k),nuanub,I2B_ovoo(:,:,i,j),noa,1.0d0,X3C,nuanub)

                                    !!!!! L3C -> (T3[2])^+ !!!!!
                                    ! Diagram 1: A(bc) H2B_ovvv(i,e,a,b)*l2c(e,c,j,k)
                                    !call dgemm('n','n',nuanub,nub,nub,1.0d0,vB_ovvv_3421(:,:,:,i),nuanub,t2c(:,:,j,k),nub,1.0d0,L3C,nuanub)
                                    ! Diagram 2: A(jk) H2C_vovv(e,k,b,c)*l2b(a,e,i,j)
                                    !call dgemm('n','n',nua,nub2,nub,0.5d0,t2b(:,:,i,j),nua,vC_vovv_1342(:,:,:,k),nub,1.0d0,L3C,nua)
                                    !call dgemm('n','n',nua,nub2,nub,-0.5d0,t2b(:,:,i,k),nua,vC_vovv_1342(:,:,:,j),nub,1.0d0,L3C,nua)
                                    ! Diagram 3: A(jk)A(bc) H2B_vovv(e,j,a,b)*l2b(e,c,i,k)
                                    !call dgemm('n','n',nuanub,nub,nua,1.0d0,vB_vovv_3412(:,:,:,j),nuanub,t2b(:,:,i,k),nua,1.0d0,L3C,nuanub)
                                    !call dgemm('n','n',nuanub,nub,nua,-1.0d0,vB_vovv_3412(:,:,:,k),nuanub,t2b(:,:,i,j),nua,1.0d0,L3C,nuanub)
                                    ! Diagram 4: -A(jk) H2B_oovo(i,j,a,m)*l2c(b,c,m,k)
                                    !call dgemm('n','t',nua,nub2,nob,-0.5d0,vB_oovo_3412(:,:,i,j),nua,t2c(:,:,:,k),nub2,1.0d0,L3C,nua)
                                    !call dgemm('n','t',nua,nub2,nob,0.5d0,vB_oovo_3412(:,:,i,k),nua,t2c(:,:,:,j),nub2,1.0d0,L3C,nua)
                                    ! Diagram 5: -A(bc) H2C_ooov(j,k,m,c)*l2b(a,b,i,m)
                                    !call dgemm('n','n',nuanub,nub,nob,-1.0d0,t2b_1243(:,:,:,i),nuanub,vC_ooov_3412(:,:,j,k),nob,1.0d0,L3C,nuanub)
                                    ! Diagram 6: -A(jk)A(bc) H2B_ooov(i,j,m,b)*l2b(a,c,m,k) -> -A(jk)A(bc) H2B_ooov(i,k,m,c)*l2b(a,b,m,j)
                                    !call dgemm('n','n',nuanub,nub,noa,-1.0d0,t2b(:,:,:,j),nuanub,vB_ooov_3412(:,:,i,k),noa,1.0d0,L3C,nuanub)
                                    !call dgemm('n','n',nuanub,nub,noa,1.0d0,t2b(:,:,:,k),nuanub,vB_ooov_3412(:,:,i,j),noa,1.0d0,L3C,nuanub)

                                    do a = 1, nua
                                        do b = 1, nub
                                            do c = b+1, nub

                                                    temp1 = X3C(a,b,c) - X3C(a,c,b)
                                                    !temp2 = L3C(a,b,c) - L3C(a,c,b)
                                                    temp3 = t1b(c,k)*vB_oovv(i,j,a,b)&
                                                            -t1b(b,k)*vB_oovv(i,j,a,c)&
                                                            -t1b(c,j)*vB_oovv(i,k,a,b)&
                                                            +t1b(b,j)*vB_oovv(i,k,a,c)&
                                                            +t1a(a,i)*vC_oovv(j,k,b,c)
                                                    ! These terms correspond to < [T2]^+ F_N T_3[2] >, which is 4th order for non-HF (i.e., ROHF) orbitals
                                                    temp3 = temp3 &
                                                            +fB_ov(k,c)*t2b(a,b,i,j)&
                                                            -fB_ov(k,b)*t2b(a,c,i,j)&
                                                            -fB_ov(j,c)*t2b(a,b,i,k)&
                                                            +fB_ov(j,b)*t2b(a,c,i,k)&
                                                            +fA_ov(i,a)*t2c(b,c,j,k)

                                                    LM = temp1*(temp1+temp3)

                                                    D = fA_oo(i,i) + fB_oo(j,j) + fB_oo(k,k)&
                                                    - fA_vv(a,a) - fB_vv(b,b) - fB_vv(c,c)

                                                    deltaA = deltaA + LM/D
                                            end do
                                        end do
                                    end do
                                end do
                            end do
                        end do
              end subroutine ccsdptC_opt

              subroutine ccsdptD_opt(deltaA,&
                                     t1b,t2c,&
                                     vC_vooo,I2C_vvov,vC_oovv,fB_ov,vC_vovv,vC_ooov,fB_oo,fB_vv,&
                                     nob,nub)

                        real(kind=8), intent(out) :: deltaA
                        integer, intent(in) :: nob, nub
                        real(kind=8), intent(in) :: fB_oo(1:nob,1:nob),fB_vv(1:nub,1:nub),&
                        vC_vooo(nub,nob,nob,nob),I2C_vvov(nub,nub,nob,nub),t2c(nub,nub,nob,nob),&
                        t1b(nub,nob),vC_oovv(nob,nob,nub,nub),&
                        fB_ov(nob,nub),vC_vovv(nub,nob,nub,nub),vC_ooov(nob,nob,nob,nub)
                        integer :: i, j, k, a, b, c, nub2
                        real(kind=8) :: D, temp1, temp2, temp3, LM, X3D(nub,nub,nub), L3D(nub,nub,nub)

                        ! reordered arrays for DGEMMs
                        real(kind=8) :: I2C_vvov_1243(nub,nub,nub,nob), vC_vovv_4312(nub,nub,nub,nob), vC_ooov_4312(nub,nob,nob,nob)

                        call reorder_stripe(4, shape(I2C_vvov), size(I2C_vvov), '1243', I2C_vvov, I2C_vvov_1243)
                        call reorder_stripe(4, shape(vC_vovv), size(vC_vovv), '4312', vC_vovv, vC_vovv_4312)
                        call reorder_stripe(4, shape(vC_ooov), size(vC_ooov), '4312', vC_ooov, vC_ooov_4312)

                        deltaA = 0.0d0

                        nub2 = nub*nub
                        do i = 1 , nob
                            do j = i+1, nob
                                do k = j+1, nob

                                   X3D = 0.0d0
                                   L3D = 0.0d0
                                   !!!!! MM(2,3)D -> V*T2 !!!!!
                                   ! Diagram 1: -A(k/ij)A(a/bc) H2C_vooo(a,m,i,j)*t2c(b,c,m,k)
                                   call dgemm('n','t',nub,nub2,nob,-0.5d0,vC_vooo(:,:,i,j),nub,t2c(:,:,:,k),nub2,1.0d0,X3D,nub)
                                   call dgemm('n','t',nub,nub2,nob,0.5d0,vC_vooo(:,:,k,j),nub,t2c(:,:,:,i),nub2,1.0d0,X3D,nub)
                                   call dgemm('n','t',nub,nub2,nob,0.5d0,vC_vooo(:,:,i,k),nub,t2c(:,:,:,j),nub2,1.0d0,X3D,nub)
                                   ! Diagram 2: A(i/jk)A(c/ab) I2C_vvov(a,b,i,e)*t2c(e,c,j,k)
                                   call dgemm('n','n',nub2,nub,nub,0.5d0,I2C_vvov_1243(:,:,:,i),nub2,t2c(:,:,j,k),nub,1.0d0,X3D,nub2)
                                   call dgemm('n','n',nub2,nub,nub,-0.5d0,I2C_vvov_1243(:,:,:,j),nub2,t2c(:,:,i,k),nub,1.0d0,X3D,nub2)
                                   call dgemm('n','n',nub2,nub,nub,-0.5d0,I2C_vvov_1243(:,:,:,k),nub2,t2c(:,:,j,i),nub,1.0d0,X3D,nub2)
                                   !!!!! L3A -? (T3[2])^+ !!!!!
                                   ! Diagram 1: A(i/jk)A(c/ab) H2C_vovv(e,i,b,a)*l2c(e,c,j,k)
                                   !call dgemm('n','n',nub2,nub,nub,0.5d0,vC_vovv_4312(:,:,:,i),nub2,t2c(:,:,j,k),nub,1.0d0,L3D,nub2)
                                   !call dgemm('n','n',nub2,nub,nub,-0.5d0,vC_vovv_4312(:,:,:,j),nub2,t2c(:,:,i,k),nub,1.0d0,L3D,nub2)
                                   !call dgemm('n','n',nub2,nub,nub,-0.5d0,vC_vovv_4312(:,:,:,k),nub2,t2c(:,:,j,i),nub,1.0d0,L3D,nub2)
                                   ! Diagram 2: -A(k/ij)A(a/bc) H2A_ooov(j,i,m,a)*l2a(b,c,m,k)-> a,m,j,i * (b,c,m,k)'
                                   !call dgemm('n','t',nub,nub2,nob,-0.5d0,vC_ooov_4312(:,:,j,i),nub,t2c(:,:,:,k),nub2,1.0d0,L3D,nub)
                                   !call dgemm('n','t',nub,nub2,nob,0.5d0,vC_ooov_4312(:,:,k,i),nub,t2c(:,:,:,j),nub2,1.0d0,L3D,nub)
                                   !call dgemm('n','t',nub,nub2,nob,0.5d0,vC_ooov_4312(:,:,j,k),nub,t2c(:,:,:,i),nub2,1.0d0,L3D,nub)

                                    do a = 1, nub
                                        do b = a+1, nub
                                            do c = b+1, nub

                                                temp1 = X3D(a,b,c) + X3D(b,c,a) + X3D(c,a,b)&
                                                - X3D(a,c,b) - X3D(b,a,c) - X3D(c,b,a)

                                                !temp2 = L3D(a,b,c) + L3D(b,c,a) + L3D(c,a,b)&
                                                !- L3D(a,c,b) - L3D(b,a,c) - L3D(c,b,a)

                                                temp3 =&
                                                t1b(c,k)*vC_oovv(i,j,a,b)&
                                                -t1b(a,k)*vC_oovv(i,j,c,b)&
                                                -t1b(b,k)*vC_oovv(i,j,a,c)&
                                                -t1b(c,i)*vC_oovv(k,j,a,b)&
                                                -t1b(c,j)*vC_oovv(i,k,a,b)&
                                                +t1b(a,i)*vC_oovv(k,j,c,b)&
                                                +t1b(b,i)*vC_oovv(k,j,a,c)&
                                                +t1b(a,j)*vC_oovv(i,k,c,b)&
                                                +t1b(b,j)*vC_oovv(i,k,a,c)
                                                ! These terms correspond to < [T2]^+ F_N T_3[2] >, which is 4th order for non-HF (i.e., ROHF) orbitals
                                                temp3 = temp3 &
                                                +fB_ov(k,c)*t2c(a,b,i,j)&
                                                -fB_ov(k,a)*t2c(c,b,i,j)&
                                                -fB_ov(k,b)*t2c(a,c,i,j)&
                                                -fB_ov(i,c)*t2c(a,b,k,j)&
                                                -fB_ov(j,c)*t2c(a,b,i,k)&
                                                +fB_ov(i,a)*t2c(c,b,k,j)&
                                                +fB_ov(i,b)*t2c(a,c,k,j)&
                                                +fB_ov(j,a)*t2c(c,b,i,k)&
                                                +fB_ov(j,b)*t2c(a,c,i,k)

                                                LM = temp1*(temp1+temp3)

                                                D = fB_oo(i,i) + fB_oo(j,j) + fB_oo(k,k)&
                                                - fB_vv(a,a) - fB_vv(b,b) - fB_vv(c,c)

                                                deltaA = deltaA + LM/D
                                            end do
                                        end do
                                    end do
                                end do
                            end do
                        end do

              end subroutine ccsdptD_opt

              !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DIRECT P SPACE CCSD(T) CORRECTION ROUTINE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

              subroutine ccsdptA_p(deltaA,&
                                   pspace,&
                                   t1a,t2a,&
                                   vA_vooo,I2A_vvov,vA_oovv,fA_ov,vA_vovv,vA_ooov,fA_oo,fA_vv,&
                                   noa,nua)

                        integer, intent(in) :: noa, nua
                        logical(kind=1), intent(in) :: pspace(nua,nua,nua,noa,noa,noa)
                        real(kind=8), intent(in) :: fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),&
                        vA_vooo(nua,noa,noa,noa),I2A_vvov(nua,nua,noa,nua),t1a(nua,noa),t2a(nua,nua,noa,noa),&
                        vA_oovv(noa,noa,nua,nua),&
                        fA_ov(noa,nua),vA_vovv(nua,noa,nua,nua),vA_ooov(noa,noa,noa,nua)

                        real(kind=8), intent(out) :: deltaA

                        integer :: i, j, k, a, b, c, nua2
                        real(kind=8) :: D, temp1, temp2, temp3, LM, X3A(nua,nua,nua), L3A(nua,nua,nua)

                        ! reordered arrays for DGEMMs
                        real(kind=8) :: I2A_vvov_1243(nua,nua,nua,noa), vA_vovv_4312(nua,nua,nua,noa), vA_ooov_4312(nua,noa,noa,noa)

                        call reorder_stripe(4, shape(I2A_vvov), size(I2A_vvov), '1243', I2A_vvov, I2A_vvov_1243)
                        call reorder_stripe(4, shape(vA_vovv), size(vA_vovv), '4312', vA_vovv, vA_vovv_4312)
                        call reorder_stripe(4, shape(vA_ooov), size(vA_ooov), '4312', vA_ooov, vA_ooov_4312)

                        deltaA = 0.0d0

                        nua2 = nua*nua
                        do i = 1 , noa
                            do j = i+1, noa
                                do k = j+1, noa

                                   X3A = 0.0d0
                                   L3A = 0.0d0
                                   !!!!! MM(2,3)A -> V*T2 !!!!!
                                   ! Diagram 1: -A(k/ij)A(a/bc) I2A_vooo(a,m,i,j)*t2a(b,c,m,k)
                                   call dgemm('n','t',nua,nua2,noa,-0.5d0,vA_vooo(:,:,i,j),nua,t2a(:,:,:,k),nua2,1.0d0,X3A,nua)
                                   call dgemm('n','t',nua,nua2,noa,0.5d0,vA_vooo(:,:,k,j),nua,t2a(:,:,:,i),nua2,1.0d0,X3A,nua)
                                   call dgemm('n','t',nua,nua2,noa,0.5d0,vA_vooo(:,:,i,k),nua,t2a(:,:,:,j),nua2,1.0d0,X3A,nua)
                                   ! Diagram 2: A(i/jk)A(c/ab) I2A_vvov(a,b,i,e)*t2a(e,c,j,k)
                                   call dgemm('n','n',nua2,nua,nua,0.5d0,I2A_vvov_1243(:,:,:,i),nua2,t2a(:,:,j,k),nua,1.0d0,X3A,nua2)
                                   call dgemm('n','n',nua2,nua,nua,-0.5d0,I2A_vvov_1243(:,:,:,j),nua2,t2a(:,:,i,k),nua,1.0d0,X3A,nua2)
                                   call dgemm('n','n',nua2,nua,nua,-0.5d0,I2A_vvov_1243(:,:,:,k),nua2,t2a(:,:,j,i),nua,1.0d0,X3A,nua2)
                                   !!!!! L3A -> (T3[2])^+ !!!!!
                                   ! Diagram 1: A(i/jk)A(c/ab) H2A_vovv(e,i,b,a)*l2a(e,c,j,k)
                                   !call dgemm('n','n',nua2,nua,nua,0.5d0,vA_vovv_4312(:,:,:,i),nua2,t2a(:,:,j,k),nua,1.0d0,L3A,nua2)
                                   !call dgemm('n','n',nua2,nua,nua,-0.5d0,vA_vovv_4312(:,:,:,j),nua2,t2a(:,:,i,k),nua,1.0d0,L3A,nua2)
                                   !call dgemm('n','n',nua2,nua,nua,-0.5d0,vA_vovv_4312(:,:,:,k),nua2,t2a(:,:,j,i),nua,1.0d0,L3A,nua2)
                                   ! Diagram 2: -A(k/ij)A(a/bc) H2A_ooov(j,i,m,a)*l2a(b,c,m,k)-> a,m,j,i * (b,c,m,k)'
                                   !call dgemm('n','t',nua,nua2,noa,-0.5d0,vA_ooov_4312(:,:,j,i),nua,t2a(:,:,:,k),nua2,1.0d0,L3A,nua)
                                   !call dgemm('n','t',nua,nua2,noa,0.5d0,vA_ooov_4312(:,:,k,i),nua,t2a(:,:,:,j),nua2,1.0d0,L3A,nua)
                                   !call dgemm('n','t',nua,nua2,noa,0.5d0,vA_ooov_4312(:,:,j,k),nua,t2a(:,:,:,i),nua2,1.0d0,L3A,nua)

                                    do a = 1, nua
                                        do b = a+1, nua
                                            do c = b+1, nua

                                                if (pspace(a,b,c,i,j,k)) cycle

                                                temp1 = X3A(a,b,c) + X3A(b,c,a) + X3A(c,a,b)&
                                                - X3A(a,c,b) - X3A(b,a,c) - X3A(c,b,a)

                                                !temp2 = L3A(a,b,c) + L3A(b,c,a) + L3A(c,a,b)&
                                                !- L3A(a,c,b) - L3A(b,a,c) - L3A(c,b,a)

                                                temp3 =&
                                                t1a(c,k)*vA_oovv(i,j,a,b)&
                                                -t1a(a,k)*vA_oovv(i,j,c,b)&
                                                -t1a(b,k)*vA_oovv(i,j,a,c)&
                                                -t1a(c,i)*vA_oovv(k,j,a,b)&
                                                -t1a(c,j)*vA_oovv(i,k,a,b)&
                                                +t1a(a,i)*vA_oovv(k,j,c,b)&
                                                +t1a(b,i)*vA_oovv(k,j,a,c)&
                                                +t1a(a,j)*vA_oovv(i,k,c,b)&
                                                +t1a(b,j)*vA_oovv(i,k,a,c)
                                                temp3 = temp3&
                                                +fA_ov(k,c)*t2a(a,b,i,j)&
                                                -fA_ov(k,a)*t2a(c,b,i,j)&
                                                -fA_ov(k,b)*t2a(a,c,i,j)&
                                                -fA_ov(i,c)*t2a(a,b,k,j)&
                                                -fA_ov(j,c)*t2a(a,b,i,k)&
                                                +fA_ov(i,a)*t2a(c,b,k,j)&
                                                +fA_ov(i,b)*t2a(a,c,k,j)&
                                                +fA_ov(j,a)*t2a(c,b,i,k)&
                                                +fA_ov(j,b)*t2a(a,c,i,k)

                                                LM = temp1*(temp1+temp3)

                                                D = fA_oo(i,i) + fA_oo(j,j) + fA_oo(k,k)&
                                                - fA_vv(a,a) - fA_vv(b,b) - fA_vv(c,c)

                                                deltaA = deltaA + LM/D

                                            end do
                                        end do
                                    end do
                                end do
                            end do
                        end do

              end subroutine ccsdptA_p

              subroutine ccsdptB_p(deltaA,&
                                   pspace,&
                                   t1a,t1b,t2a,t2b,&
                                   I2B_ovoo,I2B_vooo,I2A_vooo,&
                                   vB_vvvo,vB_vvov,vA_vvov,&
                                   vB_vovv,vB_ovvv,vA_vovv,&
                                   vB_ooov,vB_oovo,vA_ooov,&
                                   fA_ov,fB_ov,&
                                   vA_oovv,vB_oovv,&
                                   fA_oo,fA_vv,fB_oo,fB_vv,&
                                   noa,nua,nob,nub)

                        real(kind=8), intent(out) :: deltaA
                        integer, intent(in) :: noa, nua, nob, nub
                        logical(kind=1), intent(in) :: pspace(nua,nua,nub,noa,noa,nob)
                        real(kind=8), intent(in) :: t2a(nua,nua,noa,noa),t2b(nua,nub,noa,nob),&
                        t1a(nua,noa),t1b(nub,nob),&
                        I2B_ovoo(noa,nub,noa,nob),I2B_vooo(nua,nob,noa,nob),&
                        I2A_vooo(nua,noa,noa,noa),vB_vvvo(nua,nub,nua,nob),&
                        vB_vvov(nua,nub,noa,nub),vA_vvov(nua,nua,noa,nua),&
                        vB_vovv(nua,nob,nua,nub),vB_ovvv(noa,nub,nua,nub),&
                        vA_vovv(nua,noa,nua,nua),vB_ooov(noa,nob,noa,nub),&
                        vB_oovo(noa,nob,nua,nob),vA_ooov(noa,noa,noa,nua),&
                        fA_ov(noa,nua),fB_ov(nob,nub),&
                        vA_oovv(noa,noa,nua,nua),vB_oovv(noa,nob,nua,nub),&
                        fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),&
                        fB_oo(1:nob,1:nob),fB_vv(1:nub,1:nub)

                        integer :: i, j, k, a, b, c, nuanub, nua2
                        real(kind=8) :: D, temp1, temp2, temp3, LM, X3B(nua,nua,nub), L3B(nua,nua,nub)

                        ! arrays for reordering
                        real(kind=8) :: t2a_1243(nua,nua,noa,noa), vB_vvov_1243(nua,nub,nub,noa),&
                            t2b_1243(nua,nub,nob,noa), vA_vvov_1243(nua,nua,nua,noa),&
                            vB_vovv_1342(nua,nua,nub,nob), vA_vovv_4312(nua,nua,nua,noa),&
                            vB_ovvv_2341(nub,nua,nub,noa), vB_ooov_3412(noa,nub,noa,nob),&
                            vA_ooov_4312(nua,noa,noa,noa), vB_oovo_3412(nua,nob,noa,nob)

                        call reorder_stripe(4, shape(t2a), size(t2a), '1243', t2a, t2a_1243)
                        call reorder_stripe(4, shape(vB_vvov), size(vB_vvov), '1243', vB_vvov, vB_vvov_1243)
                        call reorder_stripe(4, shape(t2b), size(t2b), '1243', t2b, t2b_1243)
                        call reorder_stripe(4, shape(vA_vvov), size(vA_vvov), '1243', vA_vvov, vA_vvov_1243)
                        call reorder_stripe(4, shape(vB_vovv), size(vB_vovv), '1342', vB_vovv, vB_vovv_1342)
                        call reorder_stripe(4, shape(vA_vovv), size(vA_vovv), '4312', vA_vovv, vA_vovv_4312)
                        call reorder_stripe(4, shape(vB_ovvv), size(vB_ovvv), '2341', vB_ovvv, vB_ovvv_2341)
                        call reorder_stripe(4, shape(vB_ooov), size(vB_ooov), '3412', vB_ooov, vB_ooov_3412)
                        call reorder_stripe(4, shape(vA_ooov), size(vA_ooov), '4312', vA_ooov, vA_ooov_4312)
                        call reorder_stripe(4, shape(vB_oovo), size(vB_oovo), '3412', vB_oovo, vB_oovo_3412)

                        deltaA = 0.0d0

                        nuanub = nua*nub
                        nua2 = nua*nua
                        do i = 1, noa
                            do j = i+1, noa
                                do k = 1, nob

                                    X3B = 0.0d0
                                    L3B = 0.0d0
                                    !!!!! MM(2,3)B -> V*T2 !!!!!
                                    ! Diagram 1: A(ab) H2B(bcek)*t2a(aeij)
                                    call dgemm('n','t',nua,nuanub,nua,1.0d0,t2a(:,:,i,j),nua,vB_vvvo(:,:,:,k),nuanub,1.0d0,X3B,nua)
                                    ! Diagram 2: -A(ij) I2B(mcjk)*t2a(abim)
                                    call dgemm('n','n',nua2,nub,noa,0.5d0,t2a(:,:,:,i),nua2,I2B_ovoo(:,:,j,k),noa,1.0d0,X3B,nua2)
                                    call dgemm('n','n',nua2,nub,noa,-0.5d0,t2a(:,:,:,j),nua2,I2B_ovoo(:,:,i,k),noa,1.0d0,X3B,nua2)
                                    ! Diagram 3: A(ab)A(ij) H2B(acie)*t2b(bejk) -> A(ab)A(ij) t2b(aeik)*H2B(bcje)
                                    call dgemm('n','t',nua,nuanub,nub,1.0d0,t2b(:,:,i,k),nua,vB_vvov_1243(:,:,:,j),nuanub,1.0d0,X3B,nua)
                                    call dgemm('n','t',nua,nuanub,nub,-1.0d0,t2b(:,:,j,k),nua,vB_vvov_1243(:,:,:,i),nuanub,1.0d0,X3B,nua)
                                    ! Diagram 4: -A(ab)A(ij) I2B(amik)*t2b(bcjm)
                                    call dgemm('n','t',nua,nuanub,nob,-1.0d0,I2B_vooo(:,:,i,k),nua,t2b_1243(:,:,:,j),nuanub,1.0d0,X3B,nua)
                                    call dgemm('n','t',nua,nuanub,nob,1.0d0,I2B_vooo(:,:,j,k),nua,t2b_1243(:,:,:,i),nuanub,1.0d0,X3B,nua)
                                    ! Diagram 5: A(ij) H2A(abie)*t2b(ecjk)
                                    call dgemm('n','n',nua2,nub,nua,0.5d0,vA_vvov_1243(:,:,:,i),nua2,t2b(:,:,j,k),nua,1.0d0,X3B,nua2)
                                    call dgemm('n','n',nua2,nub,nua,-0.5d0,vA_vvov_1243(:,:,:,j),nua2,t2b(:,:,i,k),nua,1.0d0,X3B,nua2)
                                    ! Diagram 6: -A(ab) I2A(amij)*t2b(bcmk)
                                    call dgemm('n','t',nua,nuanub,noa,-1.0d0,I2A_vooo(:,:,i,j),nua,t2b(:,:,:,k),nuanub,1.0d0,X3B,nua)
                                    !!!!! L3B -> (T3[2])^+ !!!!!
                                    ! Diagram 1: A(ab) H2B(ekbc)*l2a(aeij)
                                    !call dgemm('n','n',nua,nuanub,nua,1.0d0,t2a(:,:,i,j),nua,vB_vovv_1342(:,:,:,k),nua,1.0d0,L3B,nua)
                                    ! Diagram 2: A(ij) H2A(eiba)*l2b(ecjk)
                                    !call dgemm('n','n',nua2,nub,nua,0.5d0,vA_vovv_4312(:,:,:,i),nua2,t2b(:,:,j,k),nua,1.0d0,L3B,nua2)
                                    !call dgemm('n','n',nua2,nub,nua,-0.5d0,vA_vovv_4312(:,:,:,j),nua2,t2b(:,:,i,k),nua,1.0d0,L3B,nua2)
                                    ! Diagram 3: A(ij)A(ab) H2B(ieac)*l2b(bejk) -> l2b(aeik)*H2B(jebc)
                                    !call dgemm('n','n',nua,nuanub,nub,1.0d0,t2b(:,:,i,k),nua,vB_ovvv_2341(:,:,:,j),nub,1.0d0,L3B,nua)
                                    !call dgemm('n','n',nua,nuanub,nub,-1.0d0,t2b(:,:,j,k),nua,vB_ovvv_2341(:,:,:,i),nub,1.0d0,L3B,nua)
                                    ! Diagram 4: -A(ij) H2B(jkmc)*l2a(abim) -> +A(ij) H2B(jkmc)*l2a(abmi)
                                    !call dgemm('n','n',nua2,nub,noa,0.5d0,t2a(:,:,:,i),nua2,vB_ooov_3412(:,:,j,k),noa,1.0d0,L3B,nua2)
                                    !call dgemm('n','n',nua2,nub,noa,-0.5d0,t2a(:,:,:,j),nua2,vB_ooov_3412(:,:,i,k),noa,1.0d0,L3B,nua2)
                                    ! Diagram 5: -A(ab) H2A(jima)*l2b(bcmk)
                                    !call dgemm('n','t',nua,nuanub,noa,-1.0d0,vA_ooov_4312(:,:,j,i),nua,t2b(:,:,:,k),nuanub,1.0d0,L3B,nua)
                                    ! Diagram 6: -A(ij)A(ab) H2B(ikam)*l2b(bcjm)
                                    !call dgemm('n','t',nua,nuanub,nob,-1.0d0,vB_oovo_3412(:,:,i,k),nua,t2b_1243(:,:,:,j),nuanub,1.0d0,L3B,nua)
                                    !call dgemm('n','t',nua,nuanub,nob,1.0d0,vB_oovo_3412(:,:,j,k),nua,t2b_1243(:,:,:,i),nuanub,1.0d0,L3B,nua)

                                    do a = 1, nua
                                        do b = a+1, nua
                                            do c = 1, nub

                                                if (pspace(a,b,c,i,j,k)) cycle

                                                temp1 = X3B(a,b,c) - X3B(b,a,c)
                                                !temp2 = L3B(a,b,c) - L3B(b,a,c)
                                                temp3 = t1a(a,i)*vB_oovv(j,k,b,c)&
                                                       -t1a(a,j)*vB_oovv(i,k,b,c)&
                                                       -t1a(b,i)*vB_oovv(j,k,a,c)&
                                                       +t1a(b,j)*vB_oovv(i,k,a,c)&
                                                       +t1b(c,k)*vA_oovv(i,j,a,b)
                                                temp3 = temp3 &
                                                       +t2b(b,c,j,k)*fA_ov(i,a)&
                                                       -t2b(b,c,i,k)*fA_ov(j,a)&
                                                       -t2b(a,c,j,k)*fA_ov(i,b)&
                                                       +t2b(a,c,i,k)*fA_ov(j,b)&
                                                       +t2a(a,b,i,j)*fB_ov(k,c)

                                                LM = temp1*(temp1+temp3)

                                                D = fA_oo(i,i) + fA_oo(j,j) + fB_oo(k,k)&
                                                - fA_vv(a,a) - fA_vv(b,b) - fB_vv(c,c)

                                                deltaA = deltaA + LM/D

                                            end do
                                        end do
                                    end do
                                end do
                            end do
                        end do
              end subroutine ccsdptB_p

              subroutine ccsdptC_p(deltaA,&
                                   pspace,&
                                   t1a,t1b,t2b,t2c,&
                                   I2B_vooo,I2C_vooo,I2B_ovoo,&
                                   vB_vvov,vC_vvov,vB_vvvo,&
                                   vB_ovvv,vB_vovv,vC_vovv,&
                                   vB_oovo,vB_ooov,vC_ooov,&
                                   fA_ov,fB_ov,&
                                   vB_oovv,vC_oovv,&
                                   fA_oo,fA_vv,fB_oo,fB_vv,&
                                   noa,nua,nob,nub)

                        real(kind=8), intent(out) :: deltaA
                        integer, intent(in) :: noa, nua, nob, nub
                        logical(kind=1), intent(in) :: pspace(nua,nub,nub,noa,nob,nob)
                        real(kind=8), intent(in) :: t2b(nua,nub,noa,nob),&
                        t2c(nub,nub,nob,nob),t1a(nua,noa),t1b(nub,nob),&
                        I2B_vooo(nua,nob,noa,nob),I2C_vooo(nub,nob,nob,nob),&
                        I2B_ovoo(noa,nub,noa,nob),vB_vvov(nua,nub,noa,nub),&
                        vC_vvov(nub,nub,nob,nub),vB_vvvo(nua,nub,nua,nob),&
                        vB_ovvv(noa,nub,nua,nub),vB_vovv(nua,nob,nua,nub),&
                        vC_vovv(nub,nob,nub,nub),vB_oovo(noa,nob,nua,nob),&
                        vB_ooov(noa,nob,noa,nub),vC_ooov(nob,nob,nob,nub),&
                        fA_ov(noa,nua),fB_ov(nob,nub),&
                        vB_oovv(noa,nob,nua,nub),vC_oovv(nob,nob,nub,nub),&
                        fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),&
                        fB_oo(1:nob,1:nob),fB_vv(1:nub,1:nub)

                        integer :: i, j, k, a, b, c, nuanub, nub2
                        real(kind=8) :: D, LM, temp1, temp2, temp3, X3C(nua,nub,nub), L3C(nua,nub,nub)

                        ! arrays for reordering
                        real(kind=8) :: vB_vvov_1243(nua,nub,nub,noa),&
                                vC_vvov_4213(nub,nub,nub,noa),&
                                t2b_1243(nua,nub,nob,noa),&
                                I2C_vooo_2134(nob,nub,nob,nob),&
                                vB_ovvv_3421(nua,nub,nub,noa),&
                                vC_vovv_1342(nub,nub,nub,nob),&
                                vB_vovv_3412(nua,nub,nua,nob),&
                                vB_oovo_3412(nua,nob,noa,nob),&
                                vC_ooov_3412(nob,nub,nob,nob),&
                                vB_ooov_3412(noa,nub,noa,nob)

                        call reorder_stripe(4, shape(vB_vvov), size(vB_vvov), '1243', vB_vvov, vB_vvov_1243)
                        call reorder_stripe(4, shape(vC_vvov), size(vC_vvov), '4213', vC_vvov, vC_vvov_4213)
                        call reorder_stripe(4, shape(t2b), size(t2b), '1243', t2b, t2b_1243)
                        call reorder_stripe(4, shape(I2C_vooo), size(I2C_vooo), '2134', I2C_vooo, I2C_vooo_2134)
                        call reorder_stripe(4, shape(vB_ovvv), size(vB_ovvv), '3421', vB_ovvv, vB_ovvv_3421)
                        call reorder_stripe(4, shape(vC_vovv), size(vC_vovv), '1342', vC_vovv, vC_vovv_1342)
                        call reorder_stripe(4, shape(vB_vovv), size(vB_vovv), '3412', vB_vovv, vB_vovv_3412)
                        call reorder_stripe(4, shape(vB_oovo), size(vB_oovo), '3412', vB_oovo, vB_oovo_3412)
                        call reorder_stripe(4, shape(vC_ooov), size(vC_ooov), '3412', vC_ooov, vC_ooov_3412)
                        call reorder_stripe(4, shape(vB_ooov), size(vB_ooov), '3412', vB_ooov, vB_ooov_3412)

                        deltaA = 0.0d0

                        nuanub = nua*nub
                        nub2 = nub*nub
                        do i = 1 , noa
                            do j = 1, nob
                                do k = j+1, nob
                                    X3C = 0.0d0
                                    L3C = 0.0d0

                                    !!!!! MM(2,3)C -> V*T2 !!!!!
                                    ! Diagram 1: A(bc) H2B_vvov(a,b,i,e)*t2c(e,c,j,k)
                                    call dgemm('n','n',nuanub,nub,nub,1.0d0,vB_vvov_1243(:,:,:,i),nuanub,t2c(:,:,j,k),nub,1.0d0,X3C,nuanub)
                                    ! Diagram 2: -A(jk) I2B_vooo(a,m,i,j)*t2c(b,c,m,k)
                                    call dgemm('n','t',nua,nub2,nob,-0.5d0,I2B_vooo(:,:,i,j),nua,t2c(:,:,:,k),nub2,1.0d0,X3C,nua)
                                    call dgemm('n','t',nua,nub2,nob,0.5d0,I2B_vooo(:,:,i,k),nua,t2c(:,:,:,j),nub2,1.0d0,X3C,nua)
                                    ! Diagram 3: A(jk) H2C_vvov(c,b,k,e)*t2b(a,e,i,j)
                                    call dgemm('n','n',nua,nub2,nub,0.5d0,t2b(:,:,i,j),nua,vC_vvov_4213(:,:,:,k),nub,1.0d0,X3C,nua)
                                    call dgemm('n','n',nua,nub2,nub,-0.5d0,t2b(:,:,i,k),nua,vC_vvov_4213(:,:,:,j),nub,1.0d0,X3C,nua)
                                    ! Diagram 4: -A(bc) I2C_vooo(c,m,k,j)*t2b(a,b,i,m)
                                    call dgemm('n','n',nuanub,nub,nob,-1.0d0,t2b_1243(:,:,:,i),nuanub,I2C_vooo_2134(:,:,k,j),nob,1.0d0,X3C,nuanub)
                                    ! Diagram 5: A(jk)A(bc) H2B_vvvo(a,b,e,j)*t2b(e,c,i,k)
                                    call dgemm('n','n',nuanub,nub,nua,1.0d0,vB_vvvo(:,:,:,j),nuanub,t2b(:,:,i,k),nua,1.0d0,X3C,nuanub)
                                    call dgemm('n','n',nuanub,nub,nua,-1.0d0,vB_vvvo(:,:,:,k),nuanub,t2b(:,:,i,j),nua,1.0d0,X3C,nuanub)
                                    ! Diagram 6: -A(jk)A(bc) I2B_ovoo(m,b,i,j)*t2b(a,c,m,k) -> -A(jk)A(bc) I2B_ovoo(m,c,i,k)*t2b(a,b,m,j)
                                    call dgemm('n','n',nuanub,nub,noa,-1.0d0,t2b(:,:,:,j),nuanub,I2B_ovoo(:,:,i,k),noa,1.0d0,X3C,nuanub)
                                    call dgemm('n','n',nuanub,nub,noa,1.0d0,t2b(:,:,:,k),nuanub,I2B_ovoo(:,:,i,j),noa,1.0d0,X3C,nuanub)

                                    !!!!! L3C -> (T3[2])^+ !!!!!
                                    ! Diagram 1: A(bc) H2B_ovvv(i,e,a,b)*l2c(e,c,j,k)
                                    !call dgemm('n','n',nuanub,nub,nub,1.0d0,vB_ovvv_3421(:,:,:,i),nuanub,t2c(:,:,j,k),nub,1.0d0,L3C,nuanub)
                                    ! Diagram 2: A(jk) H2C_vovv(e,k,b,c)*l2b(a,e,i,j)
                                    !call dgemm('n','n',nua,nub2,nub,0.5d0,t2b(:,:,i,j),nua,vC_vovv_1342(:,:,:,k),nub,1.0d0,L3C,nua)
                                    !call dgemm('n','n',nua,nub2,nub,-0.5d0,t2b(:,:,i,k),nua,vC_vovv_1342(:,:,:,j),nub,1.0d0,L3C,nua)
                                    ! Diagram 3: A(jk)A(bc) H2B_vovv(e,j,a,b)*l2b(e,c,i,k)
                                    !call dgemm('n','n',nuanub,nub,nua,1.0d0,vB_vovv_3412(:,:,:,j),nuanub,t2b(:,:,i,k),nua,1.0d0,L3C,nuanub)
                                    !call dgemm('n','n',nuanub,nub,nua,-1.0d0,vB_vovv_3412(:,:,:,k),nuanub,t2b(:,:,i,j),nua,1.0d0,L3C,nuanub)
                                    ! Diagram 4: -A(jk) H2B_oovo(i,j,a,m)*l2c(b,c,m,k)
                                    !call dgemm('n','t',nua,nub2,nob,-0.5d0,vB_oovo_3412(:,:,i,j),nua,t2c(:,:,:,k),nub2,1.0d0,L3C,nua)
                                    !call dgemm('n','t',nua,nub2,nob,0.5d0,vB_oovo_3412(:,:,i,k),nua,t2c(:,:,:,j),nub2,1.0d0,L3C,nua)
                                    ! Diagram 5: -A(bc) H2C_ooov(j,k,m,c)*l2b(a,b,i,m)
                                    !call dgemm('n','n',nuanub,nub,nob,-1.0d0,t2b_1243(:,:,:,i),nuanub,vC_ooov_3412(:,:,j,k),nob,1.0d0,L3C,nuanub)
                                    ! Diagram 6: -A(jk)A(bc) H2B_ooov(i,j,m,b)*l2b(a,c,m,k) -> -A(jk)A(bc) H2B_ooov(i,k,m,c)*l2b(a,b,m,j)
                                    !call dgemm('n','n',nuanub,nub,noa,-1.0d0,t2b(:,:,:,j),nuanub,vB_ooov_3412(:,:,i,k),noa,1.0d0,L3C,nuanub)
                                    !call dgemm('n','n',nuanub,nub,noa,1.0d0,t2b(:,:,:,k),nuanub,vB_ooov_3412(:,:,i,j),noa,1.0d0,L3C,nuanub)

                                    do a = 1, nua
                                        do b = 1, nub
                                            do c = b+1, nub

                                                if (pspace(a,b,c,i,j,k)) cycle

                                                temp1 = X3C(a,b,c) - X3C(a,c,b)
                                                !temp2 = L3C(a,b,c) - L3C(a,c,b)
                                                temp3 = t1b(c,k)*vB_oovv(i,j,a,b)&
                                                        -t1b(b,k)*vB_oovv(i,j,a,c)&
                                                        -t1b(c,j)*vB_oovv(i,k,a,b)&
                                                        +t1b(b,j)*vB_oovv(i,k,a,c)&
                                                        +t1a(a,i)*vC_oovv(j,k,b,c)
                                                temp3 = temp3 &
                                                        +fB_ov(k,c)*t2b(a,b,i,j)&
                                                        -fB_ov(k,b)*t2b(a,c,i,j)&
                                                        -fB_ov(j,c)*t2b(a,b,i,k)&
                                                        +fB_ov(j,b)*t2b(a,c,i,k)&
                                                        +fA_ov(i,a)*t2c(b,c,j,k)

                                                LM = temp1*(temp1+temp3)

                                                D = fA_oo(i,i) + fB_oo(j,j) + fB_oo(k,k)&
                                                - fA_vv(a,a) - fB_vv(b,b) - fB_vv(c,c)

                                                deltaA = deltaA + LM/D
                                            end do
                                        end do
                                    end do
                                end do
                            end do
                        end do
              end subroutine ccsdptC_p

              subroutine ccsdptD_p(deltaA,&
                                   pspace,&
                                   t1b,t2c,&
                                   vC_vooo,I2C_vvov,vC_oovv,fB_ov,vC_vovv,vC_ooov,fB_oo,fB_vv,&
                                   nob,nub)

                        real(kind=8), intent(out) :: deltaA
                        integer, intent(in) :: nob, nub
                        logical(kind=1), intent(in) :: pspace(nub,nub,nub,nob,nob,nob)
                        real(kind=8), intent(in) :: fB_oo(1:nob,1:nob),fB_vv(1:nub,1:nub),&
                        vC_vooo(nub,nob,nob,nob),I2C_vvov(nub,nub,nob,nub),t2c(nub,nub,nob,nob),&
                        t1b(nub,nob),vC_oovv(nob,nob,nub,nub),&
                        fB_ov(nob,nub),vC_vovv(nub,nob,nub,nub),vC_ooov(nob,nob,nob,nub)

                        integer :: i, j, k, a, b, c, nub2
                        real(kind=8) :: D, temp1, temp2, temp3, LM, X3D(nub,nub,nub), L3D(nub,nub,nub)

                        ! reordered arrays for DGEMMs
                        real(kind=8) :: I2C_vvov_1243(nub,nub,nub,nob), vC_vovv_4312(nub,nub,nub,nob), vC_ooov_4312(nub,nob,nob,nob)

                        call reorder_stripe(4, shape(I2C_vvov), size(I2C_vvov), '1243', I2C_vvov, I2C_vvov_1243)
                        call reorder_stripe(4, shape(vC_vovv), size(vC_vovv), '4312', vC_vovv, vC_vovv_4312)
                        call reorder_stripe(4, shape(vC_ooov), size(vC_ooov), '4312', vC_ooov, vC_ooov_4312)

                        deltaA = 0.0d0

                        nub2 = nub*nub
                        do i = 1 , nob
                            do j = i+1, nob
                                do k = j+1, nob

                                   X3D = 0.0d0
                                   L3D = 0.0d0
                                   !!!!! MM(2,3)D -> V*T2 !!!!!
                                   ! Diagram 1: -A(k/ij)A(a/bc) H2C_vooo(a,m,i,j)*t2c(b,c,m,k)
                                   call dgemm('n','t',nub,nub2,nob,-0.5d0,vC_vooo(:,:,i,j),nub,t2c(:,:,:,k),nub2,1.0d0,X3D,nub)
                                   call dgemm('n','t',nub,nub2,nob,0.5d0,vC_vooo(:,:,k,j),nub,t2c(:,:,:,i),nub2,1.0d0,X3D,nub)
                                   call dgemm('n','t',nub,nub2,nob,0.5d0,vC_vooo(:,:,i,k),nub,t2c(:,:,:,j),nub2,1.0d0,X3D,nub)
                                   ! Diagram 2: A(i/jk)A(c/ab) I2C_vvov(a,b,i,e)*t2c(e,c,j,k)
                                   call dgemm('n','n',nub2,nub,nub,0.5d0,I2C_vvov_1243(:,:,:,i),nub2,t2c(:,:,j,k),nub,1.0d0,X3D,nub2)
                                   call dgemm('n','n',nub2,nub,nub,-0.5d0,I2C_vvov_1243(:,:,:,j),nub2,t2c(:,:,i,k),nub,1.0d0,X3D,nub2)
                                   call dgemm('n','n',nub2,nub,nub,-0.5d0,I2C_vvov_1243(:,:,:,k),nub2,t2c(:,:,j,i),nub,1.0d0,X3D,nub2)
                                   !!!!! L3A -? (T3[2])^+ !!!!!
                                   ! Diagram 1: A(i/jk)A(c/ab) H2C_vovv(e,i,b,a)*l2c(e,c,j,k)
                                   !call dgemm('n','n',nub2,nub,nub,0.5d0,vC_vovv_4312(:,:,:,i),nub2,t2c(:,:,j,k),nub,1.0d0,L3D,nub2)
                                   !call dgemm('n','n',nub2,nub,nub,-0.5d0,vC_vovv_4312(:,:,:,j),nub2,t2c(:,:,i,k),nub,1.0d0,L3D,nub2)
                                   !call dgemm('n','n',nub2,nub,nub,-0.5d0,vC_vovv_4312(:,:,:,k),nub2,t2c(:,:,j,i),nub,1.0d0,L3D,nub2)
                                   ! Diagram 2: -A(k/ij)A(a/bc) H2A_ooov(j,i,m,a)*l2a(b,c,m,k)-> a,m,j,i * (b,c,m,k)'
                                   !call dgemm('n','t',nub,nub2,nob,-0.5d0,vC_ooov_4312(:,:,j,i),nub,t2c(:,:,:,k),nub2,1.0d0,L3D,nub)
                                   !call dgemm('n','t',nub,nub2,nob,0.5d0,vC_ooov_4312(:,:,k,i),nub,t2c(:,:,:,j),nub2,1.0d0,L3D,nub)
                                   !call dgemm('n','t',nub,nub2,nob,0.5d0,vC_ooov_4312(:,:,j,k),nub,t2c(:,:,:,i),nub2,1.0d0,L3D,nub)

                                    do a = 1, nub
                                        do b = a+1, nub
                                            do c = b+1, nub

                                                if (pspace(a,b,c,i,j,k)) cycle

                                                temp1 = X3D(a,b,c) + X3D(b,c,a) + X3D(c,a,b)&
                                                - X3D(a,c,b) - X3D(b,a,c) - X3D(c,b,a)

                                                !temp2 = L3D(a,b,c) + L3D(b,c,a) + L3D(c,a,b)&
                                                !- L3D(a,c,b) - L3D(b,a,c) - L3D(c,b,a)

                                                temp3 =&
                                                t1b(c,k)*vC_oovv(i,j,a,b)&
                                                -t1b(a,k)*vC_oovv(i,j,c,b)&
                                                -t1b(b,k)*vC_oovv(i,j,a,c)&
                                                -t1b(c,i)*vC_oovv(k,j,a,b)&
                                                -t1b(c,j)*vC_oovv(i,k,a,b)&
                                                +t1b(a,i)*vC_oovv(k,j,c,b)&
                                                +t1b(b,i)*vC_oovv(k,j,a,c)&
                                                +t1b(a,j)*vC_oovv(i,k,c,b)&
                                                +t1b(b,j)*vC_oovv(i,k,a,c)
                                                temp3 = temp3 &
                                                +fB_ov(k,c)*t2c(a,b,i,j)&
                                                -fB_ov(k,a)*t2c(c,b,i,j)&
                                                -fB_ov(k,b)*t2c(a,c,i,j)&
                                                -fB_ov(i,c)*t2c(a,b,k,j)&
                                                -fB_ov(j,c)*t2c(a,b,i,k)&
                                                +fB_ov(i,a)*t2c(c,b,k,j)&
                                                +fB_ov(i,b)*t2c(a,c,k,j)&
                                                +fB_ov(j,a)*t2c(c,b,i,k)&
                                                +fB_ov(j,b)*t2c(a,c,i,k)

                                                LM = temp1*(temp1+temp3)

                                                D = fB_oo(i,i) + fB_oo(j,j) + fB_oo(k,k)&
                                                - fB_vv(a,a) - fB_vv(b,b) - fB_vv(c,c)

                                                deltaA = deltaA + LM/D

                                            end do
                                        end do
                                    end do
                                end do
                            end do
                        end do

              end subroutine ccsdptD_p

              !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ADAPTIVE ON-THE-FLY MOMENT SELECTIONS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

              subroutine ccsdptA_p_with_selection(deltaA,moments,triples_list,&
                                                  pspace,&
                                                  t1a,t2a,&
                                                  vA_vooo,I2A_vvov,vA_oovv,fA_ov,vA_vovv,vA_ooov,fA_oo,fA_vv,&
                                                  noa,nua,num_add)

                        integer, intent(in) :: noa, nua, num_add
                        logical(kind=1), intent(in) :: pspace(nua,nua,nua,noa,noa,noa)
                        real(kind=8), intent(in) :: fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),&
                        vA_vooo(nua,noa,noa,noa),I2A_vvov(nua,nua,noa,nua),t1a(nua,noa),t2a(nua,nua,noa,noa),&
                        vA_oovv(noa,noa,nua,nua),&
                        fA_ov(noa,nua),vA_vovv(nua,noa,nua,nua),vA_ooov(noa,noa,noa,nua)

                        real(kind=8), intent(out) :: deltaA

                        real(kind=8), intent(inout) :: moments(num_add)
                        !f2py intent(in,out) :: moments(0:num_add-1)
                        integer, intent(inout) :: triples_list(num_add, 6)
                        !f2py intent(in,out) :: triples_list(0:num_add-1, 0:5)

                        integer :: i, j, k, a, b, c, nua2, idx_min
                        real(kind=8) :: D, temp1, temp2, temp3, LM, X3A(nua,nua,nua), L3A(nua,nua,nua)

                        ! reordered arrays for DGEMMs
                        real(kind=8) :: I2A_vvov_1243(nua,nua,nua,noa), vA_vovv_4312(nua,nua,nua,noa), vA_ooov_4312(nua,noa,noa,noa)

                        call reorder_stripe(4, shape(I2A_vvov), size(I2A_vvov), '1243', I2A_vvov, I2A_vvov_1243)
                        call reorder_stripe(4, shape(vA_vovv), size(vA_vovv), '4312', vA_vovv, vA_vovv_4312)
                        call reorder_stripe(4, shape(vA_ooov), size(vA_ooov), '4312', vA_ooov, vA_ooov_4312)

                        deltaA = 0.0d0
                        idx_min = minloc(abs(moments), dim=1)

                        nua2 = nua*nua
                        do i = 1 , noa
                            do j = i+1, noa
                                do k = j+1, noa

                                   X3A = 0.0d0
                                   L3A = 0.0d0
                                   !!!!! MM(2,3)A -> V*T2 !!!!!
                                   ! Diagram 1: -A(k/ij)A(a/bc) I2A_vooo(a,m,i,j)*t2a(b,c,m,k)
                                   call dgemm('n','t',nua,nua2,noa,-0.5d0,vA_vooo(:,:,i,j),nua,t2a(:,:,:,k),nua2,1.0d0,X3A,nua)
                                   call dgemm('n','t',nua,nua2,noa,0.5d0,vA_vooo(:,:,k,j),nua,t2a(:,:,:,i),nua2,1.0d0,X3A,nua)
                                   call dgemm('n','t',nua,nua2,noa,0.5d0,vA_vooo(:,:,i,k),nua,t2a(:,:,:,j),nua2,1.0d0,X3A,nua)
                                   ! Diagram 2: A(i/jk)A(c/ab) I2A_vvov(a,b,i,e)*t2a(e,c,j,k)
                                   call dgemm('n','n',nua2,nua,nua,0.5d0,I2A_vvov_1243(:,:,:,i),nua2,t2a(:,:,j,k),nua,1.0d0,X3A,nua2)
                                   call dgemm('n','n',nua2,nua,nua,-0.5d0,I2A_vvov_1243(:,:,:,j),nua2,t2a(:,:,i,k),nua,1.0d0,X3A,nua2)
                                   call dgemm('n','n',nua2,nua,nua,-0.5d0,I2A_vvov_1243(:,:,:,k),nua2,t2a(:,:,j,i),nua,1.0d0,X3A,nua2)
                                   !!!!! L3A -> (T3[2])^+ !!!!!
                                   ! Diagram 1: A(i/jk)A(c/ab) H2A_vovv(e,i,b,a)*l2a(e,c,j,k)
                                   !call dgemm('n','n',nua2,nua,nua,0.5d0,vA_vovv_4312(:,:,:,i),nua2,t2a(:,:,j,k),nua,1.0d0,L3A,nua2)
                                   !call dgemm('n','n',nua2,nua,nua,-0.5d0,vA_vovv_4312(:,:,:,j),nua2,t2a(:,:,i,k),nua,1.0d0,L3A,nua2)
                                   !call dgemm('n','n',nua2,nua,nua,-0.5d0,vA_vovv_4312(:,:,:,k),nua2,t2a(:,:,j,i),nua,1.0d0,L3A,nua2)
                                   ! Diagram 2: -A(k/ij)A(a/bc) H2A_ooov(j,i,m,a)*l2a(b,c,m,k)-> a,m,j,i * (b,c,m,k)'
                                   !call dgemm('n','t',nua,nua2,noa,-0.5d0,vA_ooov_4312(:,:,j,i),nua,t2a(:,:,:,k),nua2,1.0d0,L3A,nua)
                                   !call dgemm('n','t',nua,nua2,noa,0.5d0,vA_ooov_4312(:,:,k,i),nua,t2a(:,:,:,j),nua2,1.0d0,L3A,nua)
                                   !call dgemm('n','t',nua,nua2,noa,0.5d0,vA_ooov_4312(:,:,j,k),nua,t2a(:,:,:,i),nua2,1.0d0,L3A,nua)

                                    do a = 1, nua
                                        do b = a+1, nua
                                            do c = b+1, nua

                                                if (pspace(a,b,c,i,j,k)) cycle

                                                temp1 = X3A(a,b,c) + X3A(b,c,a) + X3A(c,a,b)&
                                                - X3A(a,c,b) - X3A(b,a,c) - X3A(c,b,a)

                                                !temp2 = L3A(a,b,c) + L3A(b,c,a) + L3A(c,a,b)&
                                                !- L3A(a,c,b) - L3A(b,a,c) - L3A(c,b,a)

                                                temp3 =&
                                                t1a(c,k)*vA_oovv(i,j,a,b)&
                                                -t1a(a,k)*vA_oovv(i,j,c,b)&
                                                -t1a(b,k)*vA_oovv(i,j,a,c)&
                                                -t1a(c,i)*vA_oovv(k,j,a,b)&
                                                -t1a(c,j)*vA_oovv(i,k,a,b)&
                                                +t1a(a,i)*vA_oovv(k,j,c,b)&
                                                +t1a(b,i)*vA_oovv(k,j,a,c)&
                                                +t1a(a,j)*vA_oovv(i,k,c,b)&
                                                +t1a(b,j)*vA_oovv(i,k,a,c)
                                                temp3 = temp3 &
                                                +fA_ov(k,c)*t2a(a,b,i,j)&
                                                -fA_ov(k,a)*t2a(c,b,i,j)&
                                                -fA_ov(k,b)*t2a(a,c,i,j)&
                                                -fA_ov(i,c)*t2a(a,b,k,j)&
                                                -fA_ov(j,c)*t2a(a,b,i,k)&
                                                +fA_ov(i,a)*t2a(c,b,k,j)&
                                                +fA_ov(i,b)*t2a(a,c,k,j)&
                                                +fA_ov(j,a)*t2a(c,b,i,k)&
                                                +fA_ov(j,b)*t2a(a,c,i,k)

                                                LM = temp1*(temp1+temp3)
                                                if (abs(LM) == 0.0d0) cycle

                                                D = fA_oo(i,i) + fA_oo(j,j) + fA_oo(k,k)&
                                                - fA_vv(a,a) - fA_vv(b,b) - fA_vv(c,c)

                                                deltaA = deltaA + LM/D

                                                if ( abs(LM/D) > abs(moments(idx_min)) ) then
                                                    triples_list(idx_min, :) = (/2*a-1, 2*b-1, 2*c-1, 2*i-1, 2*j-1, 2*k-1/)
                                                    moments(idx_min) = LM/D
                                                    idx_min = minloc(abs(moments), dim=1)
                                                end if

                                            end do
                                        end do
                                    end do

                                end do
                            end do
                        end do
              end subroutine ccsdptA_p_with_selection

              subroutine ccsdptB_p_with_selection(deltaA,moments,triples_list,&
                                                  pspace,&
                                                  t1a,t1b,t2a,t2b,&
                                                  I2B_ovoo,I2B_vooo,I2A_vooo,&
                                                  vB_vvvo,vB_vvov,vA_vvov,&
                                                  vB_vovv,vB_ovvv,vA_vovv,&
                                                  vB_ooov,vB_oovo,vA_ooov,&
                                                  fA_ov,fB_ov,&
                                                  vA_oovv,vB_oovv,&
                                                  fA_oo,fA_vv,fB_oo,fB_vv,&
                                                  noa,nua,nob,nub,num_add)

                        real(kind=8), intent(out) :: deltaA
                        integer, intent(in) :: noa, nua, nob, nub, num_add
                        logical(kind=1), intent(in) :: pspace(nua,nua,nub,noa,noa,nob)
                        real(kind=8), intent(in) :: t2a(nua,nua,noa,noa),t2b(nua,nub,noa,nob),&
                        t1a(nua,noa),t1b(nub,nob),&
                        I2B_ovoo(noa,nub,noa,nob),I2B_vooo(nua,nob,noa,nob),&
                        I2A_vooo(nua,noa,noa,noa),vB_vvvo(nua,nub,nua,nob),&
                        vB_vvov(nua,nub,noa,nub),vA_vvov(nua,nua,noa,nua),&
                        vB_vovv(nua,nob,nua,nub),vB_ovvv(noa,nub,nua,nub),&
                        vA_vovv(nua,noa,nua,nua),vB_ooov(noa,nob,noa,nub),&
                        vB_oovo(noa,nob,nua,nob),vA_ooov(noa,noa,noa,nua),&
                        fA_ov(noa,nua),fB_ov(nob,nub),&
                        vA_oovv(noa,noa,nua,nua),vB_oovv(noa,nob,nua,nub),&
                        fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),&
                        fB_oo(1:nob,1:nob),fB_vv(1:nub,1:nub)

                        real(kind=8), intent(inout) :: moments(num_add)
                        !f2py intent(in,out) :: moments(0:num_add-1)
                        integer, intent(inout) :: triples_list(num_add, 6)
                        !f2py intent(in,out) :: triples_list(0:num_add-1, 0:5)

                        integer :: i, j, k, a, b, c, nuanub, nua2, idx_min
                        real(kind=8) :: D, temp1, temp2, temp3, LM, X3B(nua,nua,nub), L3B(nua,nua,nub)

                        ! arrays for reordering
                        real(kind=8) :: t2a_1243(nua,nua,noa,noa), vB_vvov_1243(nua,nub,nub,noa),&
                            t2b_1243(nua,nub,nob,noa), vA_vvov_1243(nua,nua,nua,noa),&
                            vB_vovv_1342(nua,nua,nub,nob), vA_vovv_4312(nua,nua,nua,noa),&
                            vB_ovvv_2341(nub,nua,nub,noa), vB_ooov_3412(noa,nub,noa,nob),&
                            vA_ooov_4312(nua,noa,noa,noa), vB_oovo_3412(nua,nob,noa,nob)

                        call reorder_stripe(4, shape(t2a), size(t2a), '1243', t2a, t2a_1243)
                        call reorder_stripe(4, shape(vB_vvov), size(vB_vvov), '1243', vB_vvov, vB_vvov_1243)
                        call reorder_stripe(4, shape(t2b), size(t2b), '1243', t2b, t2b_1243)
                        call reorder_stripe(4, shape(vA_vvov), size(vA_vvov), '1243', vA_vvov, vA_vvov_1243)
                        call reorder_stripe(4, shape(vB_vovv), size(vB_vovv), '1342', vB_vovv, vB_vovv_1342)
                        call reorder_stripe(4, shape(vA_vovv), size(vA_vovv), '4312', vA_vovv, vA_vovv_4312)
                        call reorder_stripe(4, shape(vB_ovvv), size(vB_ovvv), '2341', vB_ovvv, vB_ovvv_2341)
                        call reorder_stripe(4, shape(vB_ooov), size(vB_ooov), '3412', vB_ooov, vB_ooov_3412)
                        call reorder_stripe(4, shape(vA_ooov), size(vA_ooov), '4312', vA_ooov, vA_ooov_4312)
                        call reorder_stripe(4, shape(vB_oovo), size(vB_oovo), '3412', vB_oovo, vB_oovo_3412)

                        deltaA = 0.0d0
                        idx_min = minloc(abs(moments), dim=1)

                        nuanub = nua*nub
                        nua2 = nua*nua
                        do i = 1, noa
                            do j = i+1, noa
                                do k = 1, nob

                                    X3B = 0.0d0
                                    L3B = 0.0d0
                                    !!!!! MM(2,3)B -> V*T2 !!!!!
                                    ! Diagram 1: A(ab) H2B(bcek)*t2a(aeij)
                                    call dgemm('n','t',nua,nuanub,nua,1.0d0,t2a(:,:,i,j),nua,vB_vvvo(:,:,:,k),nuanub,1.0d0,X3B,nua)
                                    ! Diagram 2: -A(ij) I2B(mcjk)*t2a(abim)
                                    call dgemm('n','n',nua2,nub,noa,0.5d0,t2a(:,:,:,i),nua2,I2B_ovoo(:,:,j,k),noa,1.0d0,X3B,nua2)
                                    call dgemm('n','n',nua2,nub,noa,-0.5d0,t2a(:,:,:,j),nua2,I2B_ovoo(:,:,i,k),noa,1.0d0,X3B,nua2)
                                    ! Diagram 3: A(ab)A(ij) H2B(acie)*t2b(bejk) -> A(ab)A(ij) t2b(aeik)*H2B(bcje)
                                    call dgemm('n','t',nua,nuanub,nub,1.0d0,t2b(:,:,i,k),nua,vB_vvov_1243(:,:,:,j),nuanub,1.0d0,X3B,nua)
                                    call dgemm('n','t',nua,nuanub,nub,-1.0d0,t2b(:,:,j,k),nua,vB_vvov_1243(:,:,:,i),nuanub,1.0d0,X3B,nua)
                                    ! Diagram 4: -A(ab)A(ij) I2B(amik)*t2b(bcjm)
                                    call dgemm('n','t',nua,nuanub,nob,-1.0d0,I2B_vooo(:,:,i,k),nua,t2b_1243(:,:,:,j),nuanub,1.0d0,X3B,nua)
                                    call dgemm('n','t',nua,nuanub,nob,1.0d0,I2B_vooo(:,:,j,k),nua,t2b_1243(:,:,:,i),nuanub,1.0d0,X3B,nua)
                                    ! Diagram 5: A(ij) H2A(abie)*t2b(ecjk)
                                    call dgemm('n','n',nua2,nub,nua,0.5d0,vA_vvov_1243(:,:,:,i),nua2,t2b(:,:,j,k),nua,1.0d0,X3B,nua2)
                                    call dgemm('n','n',nua2,nub,nua,-0.5d0,vA_vvov_1243(:,:,:,j),nua2,t2b(:,:,i,k),nua,1.0d0,X3B,nua2)
                                    ! Diagram 6: -A(ab) I2A(amij)*t2b(bcmk)
                                    call dgemm('n','t',nua,nuanub,noa,-1.0d0,I2A_vooo(:,:,i,j),nua,t2b(:,:,:,k),nuanub,1.0d0,X3B,nua)
                                    !!!!! L3B -> (T3[2])^+ !!!!!
                                    ! Diagram 1: A(ab) H2B(ekbc)*l2a(aeij)
                                    !call dgemm('n','n',nua,nuanub,nua,1.0d0,t2a(:,:,i,j),nua,vB_vovv_1342(:,:,:,k),nua,1.0d0,L3B,nua)
                                    ! Diagram 2: A(ij) H2A(eiba)*l2b(ecjk)
                                    !call dgemm('n','n',nua2,nub,nua,0.5d0,vA_vovv_4312(:,:,:,i),nua2,t2b(:,:,j,k),nua,1.0d0,L3B,nua2)
                                    !call dgemm('n','n',nua2,nub,nua,-0.5d0,vA_vovv_4312(:,:,:,j),nua2,t2b(:,:,i,k),nua,1.0d0,L3B,nua2)
                                    ! Diagram 3: A(ij)A(ab) H2B(ieac)*l2b(bejk) -> l2b(aeik)*H2B(jebc)
                                    !call dgemm('n','n',nua,nuanub,nub,1.0d0,t2b(:,:,i,k),nua,vB_ovvv_2341(:,:,:,j),nub,1.0d0,L3B,nua)
                                    !call dgemm('n','n',nua,nuanub,nub,-1.0d0,t2b(:,:,j,k),nua,vB_ovvv_2341(:,:,:,i),nub,1.0d0,L3B,nua)
                                    ! Diagram 4: -A(ij) H2B(jkmc)*l2a(abim) -> +A(ij) H2B(jkmc)*l2a(abmi)
                                    !call dgemm('n','n',nua2,nub,noa,0.5d0,t2a(:,:,:,i),nua2,vB_ooov_3412(:,:,j,k),noa,1.0d0,L3B,nua2)
                                    !call dgemm('n','n',nua2,nub,noa,-0.5d0,t2a(:,:,:,j),nua2,vB_ooov_3412(:,:,i,k),noa,1.0d0,L3B,nua2)
                                    ! Diagram 5: -A(ab) H2A(jima)*l2b(bcmk)
                                    !call dgemm('n','t',nua,nuanub,noa,-1.0d0,vA_ooov_4312(:,:,j,i),nua,t2b(:,:,:,k),nuanub,1.0d0,L3B,nua)
                                    ! Diagram 6: -A(ij)A(ab) H2B(ikam)*l2b(bcjm)
                                    !call dgemm('n','t',nua,nuanub,nob,-1.0d0,vB_oovo_3412(:,:,i,k),nua,t2b_1243(:,:,:,j),nuanub,1.0d0,L3B,nua)
                                    !call dgemm('n','t',nua,nuanub,nob,1.0d0,vB_oovo_3412(:,:,j,k),nua,t2b_1243(:,:,:,i),nuanub,1.0d0,L3B,nua)

                                    do a = 1, nua
                                        do b = a+1, nua
                                            do c = 1, nub

                                                if (pspace(a,b,c,i,j,k)) cycle

                                                temp1 = X3B(a,b,c) - X3B(b,a,c)
                                                !temp2 = L3B(a,b,c) - L3B(b,a,c)
                                                temp3 = t1a(a,i)*vB_oovv(j,k,b,c)&
                                                       -t1a(a,j)*vB_oovv(i,k,b,c)&
                                                       -t1a(b,i)*vB_oovv(j,k,a,c)&
                                                       +t1a(b,j)*vB_oovv(i,k,a,c)&
                                                       +t1b(c,k)*vA_oovv(i,j,a,b)
                                               temp3 = temp3 &
                                                       +t2b(b,c,j,k)*fA_ov(i,a)&
                                                       -t2b(b,c,i,k)*fA_ov(j,a)&
                                                       -t2b(a,c,j,k)*fA_ov(i,b)&
                                                       +t2b(a,c,i,k)*fA_ov(j,b)&
                                                       +t2a(a,b,i,j)*fB_ov(k,c)

                                                LM = temp1*(temp1+temp3)
                                                if (abs(LM) == 0.0d0) cycle

                                                D = fA_oo(i,i) + fA_oo(j,j) + fB_oo(k,k)&
                                                - fA_vv(a,a) - fA_vv(b,b) - fB_vv(c,c)

                                                deltaA = deltaA + LM/D

                                                if( abs(LM/D) > abs(moments(idx_min)) ) then
                                                    triples_list(idx_min, :) = (/2*a-1, 2*b-1, 2*c, 2*i-1, 2*j-1, 2*k/)
                                                    moments(idx_min) = LM/D
                                                    idx_min = minloc(abs(moments), dim=1)
                                                end if
                                            end do
                                        end do
                                    end do
                                end do
                            end do
                        end do
              end subroutine ccsdptB_p_with_selection

              subroutine ccsdptC_p_with_selection(deltaA,moments,triples_list,&
                                                  pspace,&
                                                  t1a,t1b,t2b,t2c,&
                                                  I2B_vooo,I2C_vooo,I2B_ovoo,&
                                                  vB_vvov,vC_vvov,vB_vvvo,&
                                                  vB_ovvv,vB_vovv,vC_vovv,&
                                                  vB_oovo,vB_ooov,vC_ooov,&
                                                  fA_ov,fB_ov,&
                                                  vB_oovv,vC_oovv,&
                                                  fA_oo,fA_vv,fB_oo,fB_vv,&
                                                  noa,nua,nob,nub,num_add)

                        real(kind=8), intent(out) :: deltaA
                        integer, intent(in) :: noa, nua, nob, nub, num_add
                        logical(kind=1), intent(in) :: pspace(nua,nub,nub,noa,nob,nob)
                        real(kind=8), intent(in) :: t2b(nua,nub,noa,nob),&
                        t2c(nub,nub,nob,nob),t1a(nua,noa),t1b(nub,nob),&
                        I2B_vooo(nua,nob,noa,nob),I2C_vooo(nub,nob,nob,nob),&
                        I2B_ovoo(noa,nub,noa,nob),vB_vvov(nua,nub,noa,nub),&
                        vC_vvov(nub,nub,nob,nub),vB_vvvo(nua,nub,nua,nob),&
                        vB_ovvv(noa,nub,nua,nub),vB_vovv(nua,nob,nua,nub),&
                        vC_vovv(nub,nob,nub,nub),vB_oovo(noa,nob,nua,nob),&
                        vB_ooov(noa,nob,noa,nub),vC_ooov(nob,nob,nob,nub),&
                        fA_ov(noa,nua),fB_ov(nob,nub),&
                        vB_oovv(noa,nob,nua,nub),vC_oovv(nob,nob,nub,nub),&
                        fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),&
                        fB_oo(1:nob,1:nob),fB_vv(1:nub,1:nub)

                        real(kind=8), intent(inout) :: moments(num_add)
                        !f2py intent(in,out) moments(0:num_add-1)
                        integer, intent(inout) :: triples_list(num_add, 6)
                        !f2py intent(in,out) triples_list(0:num_add-1, 0:5)

                        integer :: i, j, k, a, b, c, nuanub, nub2, idx_min
                        real(kind=8) :: D, LM, temp1, temp2, temp3, X3C(nua,nub,nub), L3C(nua,nub,nub)

                        ! arrays for reordering
                        real(kind=8) :: vB_vvov_1243(nua,nub,nub,noa),&
                                vC_vvov_4213(nub,nub,nub,noa),&
                                t2b_1243(nua,nub,nob,noa),&
                                I2C_vooo_2134(nob,nub,nob,nob),&
                                vB_ovvv_3421(nua,nub,nub,noa),&
                                vC_vovv_1342(nub,nub,nub,nob),&
                                vB_vovv_3412(nua,nub,nua,nob),&
                                vB_oovo_3412(nua,nob,noa,nob),&
                                vC_ooov_3412(nob,nub,nob,nob),&
                                vB_ooov_3412(noa,nub,noa,nob)

                        call reorder_stripe(4, shape(vB_vvov), size(vB_vvov), '1243', vB_vvov, vB_vvov_1243)
                        call reorder_stripe(4, shape(vC_vvov), size(vC_vvov), '4213', vC_vvov, vC_vvov_4213)
                        call reorder_stripe(4, shape(t2b), size(t2b), '1243', t2b, t2b_1243)
                        call reorder_stripe(4, shape(I2C_vooo), size(I2C_vooo), '2134', I2C_vooo, I2C_vooo_2134)
                        call reorder_stripe(4, shape(vB_ovvv), size(vB_ovvv), '3421', vB_ovvv, vB_ovvv_3421)
                        call reorder_stripe(4, shape(vC_vovv), size(vC_vovv), '1342', vC_vovv, vC_vovv_1342)
                        call reorder_stripe(4, shape(vB_vovv), size(vB_vovv), '3412', vB_vovv, vB_vovv_3412)
                        call reorder_stripe(4, shape(vB_oovo), size(vB_oovo), '3412', vB_oovo, vB_oovo_3412)
                        call reorder_stripe(4, shape(vC_ooov), size(vC_ooov), '3412', vC_ooov, vC_ooov_3412)
                        call reorder_stripe(4, shape(vB_ooov), size(vB_ooov), '3412', vB_ooov, vB_ooov_3412)

                        deltaA = 0.0d0
                        idx_min = minloc(abs(moments), dim=1)

                        nuanub = nua*nub
                        nub2 = nub*nub
                        do i = 1 , noa
                            do j = 1, nob
                                do k = j+1, nob
                                    X3C = 0.0d0
                                    L3C = 0.0d0

                                    !!!!! MM(2,3)C -> V*T2 !!!!!
                                    ! Diagram 1: A(bc) H2B_vvov(a,b,i,e)*t2c(e,c,j,k)
                                    call dgemm('n','n',nuanub,nub,nub,1.0d0,vB_vvov_1243(:,:,:,i),nuanub,t2c(:,:,j,k),nub,1.0d0,X3C,nuanub)
                                    ! Diagram 2: -A(jk) I2B_vooo(a,m,i,j)*t2c(b,c,m,k)
                                    call dgemm('n','t',nua,nub2,nob,-0.5d0,I2B_vooo(:,:,i,j),nua,t2c(:,:,:,k),nub2,1.0d0,X3C,nua)
                                    call dgemm('n','t',nua,nub2,nob,0.5d0,I2B_vooo(:,:,i,k),nua,t2c(:,:,:,j),nub2,1.0d0,X3C,nua)
                                    ! Diagram 3: A(jk) H2C_vvov(c,b,k,e)*t2b(a,e,i,j)
                                    call dgemm('n','n',nua,nub2,nub,0.5d0,t2b(:,:,i,j),nua,vC_vvov_4213(:,:,:,k),nub,1.0d0,X3C,nua)
                                    call dgemm('n','n',nua,nub2,nub,-0.5d0,t2b(:,:,i,k),nua,vC_vvov_4213(:,:,:,j),nub,1.0d0,X3C,nua)
                                    ! Diagram 4: -A(bc) I2C_vooo(c,m,k,j)*t2b(a,b,i,m)
                                    call dgemm('n','n',nuanub,nub,nob,-1.0d0,t2b_1243(:,:,:,i),nuanub,I2C_vooo_2134(:,:,k,j),nob,1.0d0,X3C,nuanub)
                                    ! Diagram 5: A(jk)A(bc) H2B_vvvo(a,b,e,j)*t2b(e,c,i,k)
                                    call dgemm('n','n',nuanub,nub,nua,1.0d0,vB_vvvo(:,:,:,j),nuanub,t2b(:,:,i,k),nua,1.0d0,X3C,nuanub)
                                    call dgemm('n','n',nuanub,nub,nua,-1.0d0,vB_vvvo(:,:,:,k),nuanub,t2b(:,:,i,j),nua,1.0d0,X3C,nuanub)
                                    ! Diagram 6: -A(jk)A(bc) I2B_ovoo(m,b,i,j)*t2b(a,c,m,k) -> -A(jk)A(bc) I2B_ovoo(m,c,i,k)*t2b(a,b,m,j)
                                    call dgemm('n','n',nuanub,nub,noa,-1.0d0,t2b(:,:,:,j),nuanub,I2B_ovoo(:,:,i,k),noa,1.0d0,X3C,nuanub)
                                    call dgemm('n','n',nuanub,nub,noa,1.0d0,t2b(:,:,:,k),nuanub,I2B_ovoo(:,:,i,j),noa,1.0d0,X3C,nuanub)

                                    !!!!! L3C -> (T3[2])^+ !!!!!
                                    ! Diagram 1: A(bc) H2B_ovvv(i,e,a,b)*l2c(e,c,j,k)
                                    !call dgemm('n','n',nuanub,nub,nub,1.0d0,vB_ovvv_3421(:,:,:,i),nuanub,t2c(:,:,j,k),nub,1.0d0,L3C,nuanub)
                                    ! Diagram 2: A(jk) H2C_vovv(e,k,b,c)*l2b(a,e,i,j)
                                    !call dgemm('n','n',nua,nub2,nub,0.5d0,t2b(:,:,i,j),nua,vC_vovv_1342(:,:,:,k),nub,1.0d0,L3C,nua)
                                    !call dgemm('n','n',nua,nub2,nub,-0.5d0,t2b(:,:,i,k),nua,vC_vovv_1342(:,:,:,j),nub,1.0d0,L3C,nua)
                                    ! Diagram 3: A(jk)A(bc) H2B_vovv(e,j,a,b)*l2b(e,c,i,k)
                                    !call dgemm('n','n',nuanub,nub,nua,1.0d0,vB_vovv_3412(:,:,:,j),nuanub,t2b(:,:,i,k),nua,1.0d0,L3C,nuanub)
                                    !call dgemm('n','n',nuanub,nub,nua,-1.0d0,vB_vovv_3412(:,:,:,k),nuanub,t2b(:,:,i,j),nua,1.0d0,L3C,nuanub)
                                    ! Diagram 4: -A(jk) H2B_oovo(i,j,a,m)*l2c(b,c,m,k)
                                    !call dgemm('n','t',nua,nub2,nob,-0.5d0,vB_oovo_3412(:,:,i,j),nua,t2c(:,:,:,k),nub2,1.0d0,L3C,nua)
                                    !call dgemm('n','t',nua,nub2,nob,0.5d0,vB_oovo_3412(:,:,i,k),nua,t2c(:,:,:,j),nub2,1.0d0,L3C,nua)
                                    ! Diagram 5: -A(bc) H2C_ooov(j,k,m,c)*l2b(a,b,i,m)
                                    !call dgemm('n','n',nuanub,nub,nob,-1.0d0,t2b_1243(:,:,:,i),nuanub,vC_ooov_3412(:,:,j,k),nob,1.0d0,L3C,nuanub)
                                    ! Diagram 6: -A(jk)A(bc) H2B_ooov(i,j,m,b)*l2b(a,c,m,k) -> -A(jk)A(bc) H2B_ooov(i,k,m,c)*l2b(a,b,m,j)
                                    !call dgemm('n','n',nuanub,nub,noa,-1.0d0,t2b(:,:,:,j),nuanub,vB_ooov_3412(:,:,i,k),noa,1.0d0,L3C,nuanub)
                                    !call dgemm('n','n',nuanub,nub,noa,1.0d0,t2b(:,:,:,k),nuanub,vB_ooov_3412(:,:,i,j),noa,1.0d0,L3C,nuanub)

                                    do a = 1, nua
                                        do b = 1, nub
                                            do c = b+1, nub

                                                if (pspace(a,b,c,i,j,k)) cycle

                                                temp1 = X3C(a,b,c) - X3C(a,c,b)
                                                !temp2 = L3C(a,b,c) - L3C(a,c,b)
                                                temp3 = t1b(c,k)*vB_oovv(i,j,a,b)&
                                                        -t1b(b,k)*vB_oovv(i,j,a,c)&
                                                        -t1b(c,j)*vB_oovv(i,k,a,b)&
                                                        +t1b(b,j)*vB_oovv(i,k,a,c)&
                                                        +t1a(a,i)*vC_oovv(j,k,b,c)
                                                temp3 = temp3 &
                                                        +fB_ov(k,c)*t2b(a,b,i,j)&
                                                        -fB_ov(k,b)*t2b(a,c,i,j)&
                                                        -fB_ov(j,c)*t2b(a,b,i,k)&
                                                        +fB_ov(j,b)*t2b(a,c,i,k)&
                                                        +fA_ov(i,a)*t2c(b,c,j,k)

                                                LM = temp1*(temp1+temp3)
                                                if (abs(LM) == 0.0d0) cycle

                                                D = fA_oo(i,i) + fB_oo(j,j) + fB_oo(k,k)&
                                                - fA_vv(a,a) - fB_vv(b,b) - fB_vv(c,c)

                                                deltaA = deltaA + LM/D

                                                if( abs(LM/D) > abs(moments(idx_min)) ) then
                                                    triples_list(idx_min, :) = (/2*a-1, 2*b, 2*c, 2*i-1, 2*j, 2*k/)
                                                    moments(idx_min) = LM/D
                                                    idx_min = minloc(abs(moments), dim=1)
                                                end if

                                            end do
                                        end do
                                    end do
                                end do
                            end do
                        end do
              end subroutine ccsdptC_p_with_selection

              subroutine ccsdptD_p_with_selection(deltaA,moments,triples_list,&
                                                  pspace,&
                                                  t1b,t2c,&
                                                  vC_vooo,I2C_vvov,vC_oovv,fB_ov,vC_vovv,vC_ooov,fB_oo,fB_vv,&
                                                  nob,nub,num_add)

                        real(kind=8), intent(out) :: deltaA
                        integer, intent(in) :: nob, nub, num_add
                        logical(kind=1), intent(in) :: pspace(nub,nub,nub,nob,nob,nob)
                        real(kind=8), intent(in) :: fB_oo(1:nob,1:nob),fB_vv(1:nub,1:nub),&
                        vC_vooo(nub,nob,nob,nob),I2C_vvov(nub,nub,nob,nub),t2c(nub,nub,nob,nob),&
                        t1b(nub,nob),vC_oovv(nob,nob,nub,nub),&
                        fB_ov(nob,nub),vC_vovv(nub,nob,nub,nub),vC_ooov(nob,nob,nob,nub)

                        real(kind=8), intent(inout) :: moments(num_add)
                        !f2py intent(in,out) moments(0:num_add-1)
                        integer, intent(inout) :: triples_list(num_add, 6)
                        !f2py intent(in,out) triples_list(0:num_add-1, 0:5)

                        integer :: i, j, k, a, b, c, nub2, idx_min
                        real(kind=8) :: D, temp1, temp2, temp3, LM, X3D(nub,nub,nub), L3D(nub,nub,nub)

                        ! reordered arrays for DGEMMs
                        real(kind=8) :: I2C_vvov_1243(nub,nub,nub,nob), vC_vovv_4312(nub,nub,nub,nob), vC_ooov_4312(nub,nob,nob,nob)

                        call reorder_stripe(4, shape(I2C_vvov), size(I2C_vvov), '1243', I2C_vvov, I2C_vvov_1243)
                        call reorder_stripe(4, shape(vC_vovv), size(vC_vovv), '4312', vC_vovv, vC_vovv_4312)
                        call reorder_stripe(4, shape(vC_ooov), size(vC_ooov), '4312', vC_ooov, vC_ooov_4312)

                        deltaA = 0.0d0
                        idx_min = minloc(abs(moments), dim=1)

                        nub2 = nub*nub
                        do i = 1 , nob
                            do j = i+1, nob
                                do k = j+1, nob

                                   X3D = 0.0d0
                                   L3D = 0.0d0
                                   !!!!! MM(2,3)D -> V*T2 !!!!!
                                   ! Diagram 1: -A(k/ij)A(a/bc) H2C_vooo(a,m,i,j)*t2c(b,c,m,k)
                                   call dgemm('n','t',nub,nub2,nob,-0.5d0,vC_vooo(:,:,i,j),nub,t2c(:,:,:,k),nub2,1.0d0,X3D,nub)
                                   call dgemm('n','t',nub,nub2,nob,0.5d0,vC_vooo(:,:,k,j),nub,t2c(:,:,:,i),nub2,1.0d0,X3D,nub)
                                   call dgemm('n','t',nub,nub2,nob,0.5d0,vC_vooo(:,:,i,k),nub,t2c(:,:,:,j),nub2,1.0d0,X3D,nub)
                                   ! Diagram 2: A(i/jk)A(c/ab) I2C_vvov(a,b,i,e)*t2c(e,c,j,k)
                                   call dgemm('n','n',nub2,nub,nub,0.5d0,I2C_vvov_1243(:,:,:,i),nub2,t2c(:,:,j,k),nub,1.0d0,X3D,nub2)
                                   call dgemm('n','n',nub2,nub,nub,-0.5d0,I2C_vvov_1243(:,:,:,j),nub2,t2c(:,:,i,k),nub,1.0d0,X3D,nub2)
                                   call dgemm('n','n',nub2,nub,nub,-0.5d0,I2C_vvov_1243(:,:,:,k),nub2,t2c(:,:,j,i),nub,1.0d0,X3D,nub2)
                                   !!!!! L3A -? (T3[2])^+ !!!!!
                                   ! Diagram 1: A(i/jk)A(c/ab) H2C_vovv(e,i,b,a)*l2c(e,c,j,k)
                                   !call dgemm('n','n',nub2,nub,nub,0.5d0,vC_vovv_4312(:,:,:,i),nub2,t2c(:,:,j,k),nub,1.0d0,L3D,nub2)
                                   !call dgemm('n','n',nub2,nub,nub,-0.5d0,vC_vovv_4312(:,:,:,j),nub2,t2c(:,:,i,k),nub,1.0d0,L3D,nub2)
                                   !call dgemm('n','n',nub2,nub,nub,-0.5d0,vC_vovv_4312(:,:,:,k),nub2,t2c(:,:,j,i),nub,1.0d0,L3D,nub2)
                                   ! Diagram 2: -A(k/ij)A(a/bc) H2A_ooov(j,i,m,a)*l2a(b,c,m,k)-> a,m,j,i * (b,c,m,k)'
                                   !call dgemm('n','t',nub,nub2,nob,-0.5d0,vC_ooov_4312(:,:,j,i),nub,t2c(:,:,:,k),nub2,1.0d0,L3D,nub)
                                   !call dgemm('n','t',nub,nub2,nob,0.5d0,vC_ooov_4312(:,:,k,i),nub,t2c(:,:,:,j),nub2,1.0d0,L3D,nub)
                                   !call dgemm('n','t',nub,nub2,nob,0.5d0,vC_ooov_4312(:,:,j,k),nub,t2c(:,:,:,i),nub2,1.0d0,L3D,nub)

                                    do a = 1, nub
                                        do b = a+1, nub
                                            do c = b+1, nub

                                                if (pspace(a,b,c,i,j,k)) cycle

                                                temp1 = X3D(a,b,c) + X3D(b,c,a) + X3D(c,a,b)&
                                                - X3D(a,c,b) - X3D(b,a,c) - X3D(c,b,a)

                                                !temp2 = L3D(a,b,c) + L3D(b,c,a) + L3D(c,a,b)&
                                                !- L3D(a,c,b) - L3D(b,a,c) - L3D(c,b,a)

                                                temp3 =&
                                                t1b(c,k)*vC_oovv(i,j,a,b)&
                                                -t1b(a,k)*vC_oovv(i,j,c,b)&
                                                -t1b(b,k)*vC_oovv(i,j,a,c)&
                                                -t1b(c,i)*vC_oovv(k,j,a,b)&
                                                -t1b(c,j)*vC_oovv(i,k,a,b)&
                                                +t1b(a,i)*vC_oovv(k,j,c,b)&
                                                +t1b(b,i)*vC_oovv(k,j,a,c)&
                                                +t1b(a,j)*vC_oovv(i,k,c,b)&
                                                +t1b(b,j)*vC_oovv(i,k,a,c)
                                                temp3 = temp3 &
                                                +fB_ov(k,c)*t2c(a,b,i,j)&
                                                -fB_ov(k,a)*t2c(c,b,i,j)&
                                                -fB_ov(k,b)*t2c(a,c,i,j)&
                                                -fB_ov(i,c)*t2c(a,b,k,j)&
                                                -fB_ov(j,c)*t2c(a,b,i,k)&
                                                +fB_ov(i,a)*t2c(c,b,k,j)&
                                                +fB_ov(i,b)*t2c(a,c,k,j)&
                                                +fB_ov(j,a)*t2c(c,b,i,k)&
                                                +fB_ov(j,b)*t2c(a,c,i,k)

                                                LM = temp1*(temp1+temp3)
                                                if (abs(LM) == 0.0d0) cycle

                                                D = fB_oo(i,i) + fB_oo(j,j) + fB_oo(k,k)&
                                                - fB_vv(a,a) - fB_vv(b,b) - fB_vv(c,c)

                                                deltaA = deltaA + LM/D

                                                if( abs(LM/D) > abs(moments(idx_min)) ) then
                                                    triples_list(idx_min, :) = (/2*a, 2*b, 2*c, 2*i, 2*j, 2*k/)
                                                    moments(idx_min) = LM/D
                                                    idx_min = minloc(abs(moments), dim=1)
                                                end if
                                            end do
                                        end do
                                    end do
                                end do
                            end do
                        end do

              end subroutine ccsdptD_p_with_selection

              !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ADAPTIVE FULL STORAGE MOMENT SELECTIONS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
              subroutine ccsdptA_p_full_moment(deltaA,moments,&
                                                  pspace,&
                                                  t1a,t2a,&
                                                  vA_vooo,I2A_vvov,vA_oovv,fA_ov,vA_vovv,vA_ooov,fA_oo,fA_vv,&
                                                  noa,nua)

                        integer, intent(in) :: noa, nua
                        logical(kind=1), intent(in) :: pspace(nua,nua,nua,noa,noa,noa)
                        real(kind=8), intent(in) :: fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),&
                        vA_vooo(nua,noa,noa,noa),I2A_vvov(nua,nua,noa,nua),t1a(nua,noa),t2a(nua,nua,noa,noa),&
                        vA_oovv(noa,noa,nua,nua),&
                        fA_ov(noa,nua),vA_vovv(nua,noa,nua,nua),vA_ooov(noa,noa,noa,nua)

                        real(kind=8), intent(out) :: deltaA
                        real(kind=8), intent(out) :: moments(nua,nua,nua,noa,noa,noa)

                        integer :: i, j, k, a, b, c, nua2
                        real(kind=8) :: D, temp1, temp2, temp3, LM, X3A(nua,nua,nua), L3A(nua,nua,nua)

                        ! reordered arrays for DGEMMs
                        real(kind=8) :: I2A_vvov_1243(nua,nua,nua,noa), vA_vovv_4312(nua,nua,nua,noa), vA_ooov_4312(nua,noa,noa,noa)

                        call reorder_stripe(4, shape(I2A_vvov), size(I2A_vvov), '1243', I2A_vvov, I2A_vvov_1243)
                        call reorder_stripe(4, shape(vA_vovv), size(vA_vovv), '4312', vA_vovv, vA_vovv_4312)
                        call reorder_stripe(4, shape(vA_ooov), size(vA_ooov), '4312', vA_ooov, vA_ooov_4312)

                        deltaA = 0.0d0
                        moments = 0.0d0

                        nua2 = nua*nua
                        do i = 1 , noa
                            do j = i+1, noa
                                do k = j+1, noa

                                   X3A = 0.0d0
                                   L3A = 0.0d0
                                   !!!!! MM(2,3)A -> V*T2 !!!!!
                                   ! Diagram 1: -A(k/ij)A(a/bc) I2A_vooo(a,m,i,j)*t2a(b,c,m,k)
                                   call dgemm('n','t',nua,nua2,noa,-0.5d0,vA_vooo(:,:,i,j),nua,t2a(:,:,:,k),nua2,1.0d0,X3A,nua)
                                   call dgemm('n','t',nua,nua2,noa,0.5d0,vA_vooo(:,:,k,j),nua,t2a(:,:,:,i),nua2,1.0d0,X3A,nua)
                                   call dgemm('n','t',nua,nua2,noa,0.5d0,vA_vooo(:,:,i,k),nua,t2a(:,:,:,j),nua2,1.0d0,X3A,nua)
                                   ! Diagram 2: A(i/jk)A(c/ab) I2A_vvov(a,b,i,e)*t2a(e,c,j,k)
                                   call dgemm('n','n',nua2,nua,nua,0.5d0,I2A_vvov_1243(:,:,:,i),nua2,t2a(:,:,j,k),nua,1.0d0,X3A,nua2)
                                   call dgemm('n','n',nua2,nua,nua,-0.5d0,I2A_vvov_1243(:,:,:,j),nua2,t2a(:,:,i,k),nua,1.0d0,X3A,nua2)
                                   call dgemm('n','n',nua2,nua,nua,-0.5d0,I2A_vvov_1243(:,:,:,k),nua2,t2a(:,:,j,i),nua,1.0d0,X3A,nua2)
                                   !!!!! L3A -> (T3[2])^+ !!!!!
                                   ! Diagram 1: A(i/jk)A(c/ab) H2A_vovv(e,i,b,a)*l2a(e,c,j,k)
                                   !call dgemm('n','n',nua2,nua,nua,0.5d0,vA_vovv_4312(:,:,:,i),nua2,t2a(:,:,j,k),nua,1.0d0,L3A,nua2)
                                   !call dgemm('n','n',nua2,nua,nua,-0.5d0,vA_vovv_4312(:,:,:,j),nua2,t2a(:,:,i,k),nua,1.0d0,L3A,nua2)
                                   !call dgemm('n','n',nua2,nua,nua,-0.5d0,vA_vovv_4312(:,:,:,k),nua2,t2a(:,:,j,i),nua,1.0d0,L3A,nua2)
                                   ! Diagram 2: -A(k/ij)A(a/bc) H2A_ooov(j,i,m,a)*l2a(b,c,m,k)-> a,m,j,i * (b,c,m,k)'
                                   !call dgemm('n','t',nua,nua2,noa,-0.5d0,vA_ooov_4312(:,:,j,i),nua,t2a(:,:,:,k),nua2,1.0d0,L3A,nua)
                                   !call dgemm('n','t',nua,nua2,noa,0.5d0,vA_ooov_4312(:,:,k,i),nua,t2a(:,:,:,j),nua2,1.0d0,L3A,nua)
                                   !call dgemm('n','t',nua,nua2,noa,0.5d0,vA_ooov_4312(:,:,j,k),nua,t2a(:,:,:,i),nua2,1.0d0,L3A,nua)

                                    do a = 1, nua
                                        do b = a+1, nua
                                            do c = b+1, nua

                                                if (pspace(a,b,c,i,j,k)) cycle

                                                temp1 = X3A(a,b,c) + X3A(b,c,a) + X3A(c,a,b)&
                                                - X3A(a,c,b) - X3A(b,a,c) - X3A(c,b,a)

                                                !temp2 = L3A(a,b,c) + L3A(b,c,a) + L3A(c,a,b)&
                                                !- L3A(a,c,b) - L3A(b,a,c) - L3A(c,b,a)

                                                temp3 =&
                                                t1a(c,k)*vA_oovv(i,j,a,b)&
                                                -t1a(a,k)*vA_oovv(i,j,c,b)&
                                                -t1a(b,k)*vA_oovv(i,j,a,c)&
                                                -t1a(c,i)*vA_oovv(k,j,a,b)&
                                                -t1a(c,j)*vA_oovv(i,k,a,b)&
                                                +t1a(a,i)*vA_oovv(k,j,c,b)&
                                                +t1a(b,i)*vA_oovv(k,j,a,c)&
                                                +t1a(a,j)*vA_oovv(i,k,c,b)&
                                                +t1a(b,j)*vA_oovv(i,k,a,c)
                                                temp3 = temp3 &
                                                +fA_ov(k,c)*t2a(a,b,i,j)&
                                                -fA_ov(k,a)*t2a(c,b,i,j)&
                                                -fA_ov(k,b)*t2a(a,c,i,j)&
                                                -fA_ov(i,c)*t2a(a,b,k,j)&
                                                -fA_ov(j,c)*t2a(a,b,i,k)&
                                                +fA_ov(i,a)*t2a(c,b,k,j)&
                                                +fA_ov(i,b)*t2a(a,c,k,j)&
                                                +fA_ov(j,a)*t2a(c,b,i,k)&
                                                +fA_ov(j,b)*t2a(a,c,i,k)

                                                LM = temp1*(temp1+temp3)

                                                D = fA_oo(i,i) + fA_oo(j,j) + fA_oo(k,k)&
                                                - fA_vv(a,a) - fA_vv(b,b) - fA_vv(c,c)

                                                deltaA = deltaA + LM/D

                                                moments(a,b,c,i,j,k) = LM/D
                                            end do
                                        end do
                                    end do
                                end do
                            end do
                        end do
              end subroutine ccsdptA_p_full_moment

              subroutine ccsdptB_p_full_moment(deltaA,moments,&
                                                  pspace,&
                                                  t1a,t1b,t2a,t2b,&
                                                  I2B_ovoo,I2B_vooo,I2A_vooo,&
                                                  vB_vvvo,vB_vvov,vA_vvov,&
                                                  vB_vovv,vB_ovvv,vA_vovv,&
                                                  vB_ooov,vB_oovo,vA_ooov,&
                                                  fA_ov,fB_ov,&
                                                  vA_oovv,vB_oovv,&
                                                  fA_oo,fA_vv,fB_oo,fB_vv,&
                                                  noa,nua,nob,nub)

                        real(kind=8), intent(out) :: deltaA
                        integer, intent(in) :: noa, nua, nob, nub
                        logical(kind=1), intent(in) :: pspace(nua,nua,nub,noa,noa,nob)
                        real(kind=8), intent(in) :: t2a(nua,nua,noa,noa),t2b(nua,nub,noa,nob),&
                        t1a(nua,noa),t1b(nub,nob),&
                        I2B_ovoo(noa,nub,noa,nob),I2B_vooo(nua,nob,noa,nob),&
                        I2A_vooo(nua,noa,noa,noa),vB_vvvo(nua,nub,nua,nob),&
                        vB_vvov(nua,nub,noa,nub),vA_vvov(nua,nua,noa,nua),&
                        vB_vovv(nua,nob,nua,nub),vB_ovvv(noa,nub,nua,nub),&
                        vA_vovv(nua,noa,nua,nua),vB_ooov(noa,nob,noa,nub),&
                        vB_oovo(noa,nob,nua,nob),vA_ooov(noa,noa,noa,nua),&
                        fA_ov(noa,nua),fB_ov(nob,nub),&
                        vA_oovv(noa,noa,nua,nua),vB_oovv(noa,nob,nua,nub),&
                        fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),&
                        fB_oo(1:nob,1:nob),fB_vv(1:nub,1:nub)

                        real(kind=8), intent(out) :: moments(nua,nua,nub,noa,noa,nob)

                        integer :: i, j, k, a, b, c, nuanub, nua2
                        real(kind=8) :: D, temp1, temp2, temp3, LM, X3B(nua,nua,nub), L3B(nua,nua,nub)

                        ! arrays for reordering
                        real(kind=8) :: t2a_1243(nua,nua,noa,noa), vB_vvov_1243(nua,nub,nub,noa),&
                            t2b_1243(nua,nub,nob,noa), vA_vvov_1243(nua,nua,nua,noa),&
                            vB_vovv_1342(nua,nua,nub,nob), vA_vovv_4312(nua,nua,nua,noa),&
                            vB_ovvv_2341(nub,nua,nub,noa), vB_ooov_3412(noa,nub,noa,nob),&
                            vA_ooov_4312(nua,noa,noa,noa), vB_oovo_3412(nua,nob,noa,nob)

                        call reorder_stripe(4, shape(t2a), size(t2a), '1243', t2a, t2a_1243)
                        call reorder_stripe(4, shape(vB_vvov), size(vB_vvov), '1243', vB_vvov, vB_vvov_1243)
                        call reorder_stripe(4, shape(t2b), size(t2b), '1243', t2b, t2b_1243)
                        call reorder_stripe(4, shape(vA_vvov), size(vA_vvov), '1243', vA_vvov, vA_vvov_1243)
                        call reorder_stripe(4, shape(vB_vovv), size(vB_vovv), '1342', vB_vovv, vB_vovv_1342)
                        call reorder_stripe(4, shape(vA_vovv), size(vA_vovv), '4312', vA_vovv, vA_vovv_4312)
                        call reorder_stripe(4, shape(vB_ovvv), size(vB_ovvv), '2341', vB_ovvv, vB_ovvv_2341)
                        call reorder_stripe(4, shape(vB_ooov), size(vB_ooov), '3412', vB_ooov, vB_ooov_3412)
                        call reorder_stripe(4, shape(vA_ooov), size(vA_ooov), '4312', vA_ooov, vA_ooov_4312)
                        call reorder_stripe(4, shape(vB_oovo), size(vB_oovo), '3412', vB_oovo, vB_oovo_3412)

                        deltaA = 0.0d0
                        moments = 0.0d0

                        nuanub = nua*nub
                        nua2 = nua*nua
                        do i = 1, noa
                            do j = i+1, noa
                                do k = 1, nob

                                    X3B = 0.0d0
                                    L3B = 0.0d0
                                    !!!!! MM(2,3)B -> V*T2 !!!!!
                                    ! Diagram 1: A(ab) H2B(bcek)*t2a(aeij)
                                    call dgemm('n','t',nua,nuanub,nua,1.0d0,t2a(:,:,i,j),nua,vB_vvvo(:,:,:,k),nuanub,1.0d0,X3B,nua)
                                    ! Diagram 2: -A(ij) I2B(mcjk)*t2a(abim)
                                    call dgemm('n','n',nua2,nub,noa,0.5d0,t2a(:,:,:,i),nua2,I2B_ovoo(:,:,j,k),noa,1.0d0,X3B,nua2)
                                    call dgemm('n','n',nua2,nub,noa,-0.5d0,t2a(:,:,:,j),nua2,I2B_ovoo(:,:,i,k),noa,1.0d0,X3B,nua2)
                                    ! Diagram 3: A(ab)A(ij) H2B(acie)*t2b(bejk) -> A(ab)A(ij) t2b(aeik)*H2B(bcje)
                                    call dgemm('n','t',nua,nuanub,nub,1.0d0,t2b(:,:,i,k),nua,vB_vvov_1243(:,:,:,j),nuanub,1.0d0,X3B,nua)
                                    call dgemm('n','t',nua,nuanub,nub,-1.0d0,t2b(:,:,j,k),nua,vB_vvov_1243(:,:,:,i),nuanub,1.0d0,X3B,nua)
                                    ! Diagram 4: -A(ab)A(ij) I2B(amik)*t2b(bcjm)
                                    call dgemm('n','t',nua,nuanub,nob,-1.0d0,I2B_vooo(:,:,i,k),nua,t2b_1243(:,:,:,j),nuanub,1.0d0,X3B,nua)
                                    call dgemm('n','t',nua,nuanub,nob,1.0d0,I2B_vooo(:,:,j,k),nua,t2b_1243(:,:,:,i),nuanub,1.0d0,X3B,nua)
                                    ! Diagram 5: A(ij) H2A(abie)*t2b(ecjk)
                                    call dgemm('n','n',nua2,nub,nua,0.5d0,vA_vvov_1243(:,:,:,i),nua2,t2b(:,:,j,k),nua,1.0d0,X3B,nua2)
                                    call dgemm('n','n',nua2,nub,nua,-0.5d0,vA_vvov_1243(:,:,:,j),nua2,t2b(:,:,i,k),nua,1.0d0,X3B,nua2)
                                    ! Diagram 6: -A(ab) I2A(amij)*t2b(bcmk)
                                    call dgemm('n','t',nua,nuanub,noa,-1.0d0,I2A_vooo(:,:,i,j),nua,t2b(:,:,:,k),nuanub,1.0d0,X3B,nua)
                                    !!!!! L3B -> (T3[2])^+ !!!!!
                                    ! Diagram 1: A(ab) H2B(ekbc)*l2a(aeij)
                                    !call dgemm('n','n',nua,nuanub,nua,1.0d0,t2a(:,:,i,j),nua,vB_vovv_1342(:,:,:,k),nua,1.0d0,L3B,nua)
                                    ! Diagram 2: A(ij) H2A(eiba)*l2b(ecjk)
                                    !call dgemm('n','n',nua2,nub,nua,0.5d0,vA_vovv_4312(:,:,:,i),nua2,t2b(:,:,j,k),nua,1.0d0,L3B,nua2)
                                    !call dgemm('n','n',nua2,nub,nua,-0.5d0,vA_vovv_4312(:,:,:,j),nua2,t2b(:,:,i,k),nua,1.0d0,L3B,nua2)
                                    ! Diagram 3: A(ij)A(ab) H2B(ieac)*l2b(bejk) -> l2b(aeik)*H2B(jebc)
                                    !call dgemm('n','n',nua,nuanub,nub,1.0d0,t2b(:,:,i,k),nua,vB_ovvv_2341(:,:,:,j),nub,1.0d0,L3B,nua)
                                    !call dgemm('n','n',nua,nuanub,nub,-1.0d0,t2b(:,:,j,k),nua,vB_ovvv_2341(:,:,:,i),nub,1.0d0,L3B,nua)
                                    ! Diagram 4: -A(ij) H2B(jkmc)*l2a(abim) -> +A(ij) H2B(jkmc)*l2a(abmi)
                                    !call dgemm('n','n',nua2,nub,noa,0.5d0,t2a(:,:,:,i),nua2,vB_ooov_3412(:,:,j,k),noa,1.0d0,L3B,nua2)
                                    !call dgemm('n','n',nua2,nub,noa,-0.5d0,t2a(:,:,:,j),nua2,vB_ooov_3412(:,:,i,k),noa,1.0d0,L3B,nua2)
                                    ! Diagram 5: -A(ab) H2A(jima)*l2b(bcmk)
                                    !call dgemm('n','t',nua,nuanub,noa,-1.0d0,vA_ooov_4312(:,:,j,i),nua,t2b(:,:,:,k),nuanub,1.0d0,L3B,nua)
                                    ! Diagram 6: -A(ij)A(ab) H2B(ikam)*l2b(bcjm)
                                    !call dgemm('n','t',nua,nuanub,nob,-1.0d0,vB_oovo_3412(:,:,i,k),nua,t2b_1243(:,:,:,j),nuanub,1.0d0,L3B,nua)
                                    !call dgemm('n','t',nua,nuanub,nob,1.0d0,vB_oovo_3412(:,:,j,k),nua,t2b_1243(:,:,:,i),nuanub,1.0d0,L3B,nua)

                                    do a = 1, nua
                                        do b = a+1, nua
                                            do c = 1, nub

                                                if (pspace(a,b,c,i,j,k)) cycle

                                                temp1 = X3B(a,b,c) - X3B(b,a,c)
                                                !temp2 = L3B(a,b,c) - L3B(b,a,c)
                                                temp3 = t1a(a,i)*vB_oovv(j,k,b,c)&
                                                       -t1a(a,j)*vB_oovv(i,k,b,c)&
                                                       -t1a(b,i)*vB_oovv(j,k,a,c)&
                                                       +t1a(b,j)*vB_oovv(i,k,a,c)&
                                                       +t1b(c,k)*vA_oovv(i,j,a,b)
                                               temp3 = temp3 &
                                                       +t2b(b,c,j,k)*fA_ov(i,a)&
                                                       -t2b(b,c,i,k)*fA_ov(j,a)&
                                                       -t2b(a,c,j,k)*fA_ov(i,b)&
                                                       +t2b(a,c,i,k)*fA_ov(j,b)&
                                                       +t2a(a,b,i,j)*fB_ov(k,c)

                                                LM = temp1*(temp1+temp3)

                                                D = fA_oo(i,i) + fA_oo(j,j) + fB_oo(k,k)&
                                                - fA_vv(a,a) - fA_vv(b,b) - fB_vv(c,c)

                                                deltaA = deltaA + LM/D

                                                moments(a,b,c,i,j,k) = LM/D
                                            end do
                                        end do
                                    end do
                                end do
                            end do
                        end do
              end subroutine ccsdptB_p_full_moment

              subroutine ccsdptC_p_full_moment(deltaA,moments,&
                                                  pspace,&
                                                  t1a,t1b,t2b,t2c,&
                                                  I2B_vooo,I2C_vooo,I2B_ovoo,&
                                                  vB_vvov,vC_vvov,vB_vvvo,&
                                                  vB_ovvv,vB_vovv,vC_vovv,&
                                                  vB_oovo,vB_ooov,vC_ooov,&
                                                  fA_ov,fB_ov,&
                                                  vB_oovv,vC_oovv,&
                                                  fA_oo,fA_vv,fB_oo,fB_vv,&
                                                  noa,nua,nob,nub)

                        real(kind=8), intent(out) :: deltaA
                        integer, intent(in) :: noa, nua, nob, nub
                        logical(kind=1), intent(in) :: pspace(nua,nub,nub,noa,nob,nob)
                        real(kind=8), intent(in) :: t2b(nua,nub,noa,nob),&
                        t2c(nub,nub,nob,nob),t1a(nua,noa),t1b(nub,nob),&
                        I2B_vooo(nua,nob,noa,nob),I2C_vooo(nub,nob,nob,nob),&
                        I2B_ovoo(noa,nub,noa,nob),vB_vvov(nua,nub,noa,nub),&
                        vC_vvov(nub,nub,nob,nub),vB_vvvo(nua,nub,nua,nob),&
                        vB_ovvv(noa,nub,nua,nub),vB_vovv(nua,nob,nua,nub),&
                        vC_vovv(nub,nob,nub,nub),vB_oovo(noa,nob,nua,nob),&
                        vB_ooov(noa,nob,noa,nub),vC_ooov(nob,nob,nob,nub),&
                        fA_ov(noa,nua),fB_ov(nob,nub),&
                        vB_oovv(noa,nob,nua,nub),vC_oovv(nob,nob,nub,nub),&
                        fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),&
                        fB_oo(1:nob,1:nob),fB_vv(1:nub,1:nub)

                        real(kind=8), intent(out) :: moments(nua,nub,nub,noa,nob,nob)

                        integer :: i, j, k, a, b, c, nuanub, nub2
                        real(kind=8) :: D, LM, temp1, temp2, temp3, X3C(nua,nub,nub), L3C(nua,nub,nub)

                        ! arrays for reordering
                        real(kind=8) :: vB_vvov_1243(nua,nub,nub,noa),&
                                vC_vvov_4213(nub,nub,nub,noa),&
                                t2b_1243(nua,nub,nob,noa),&
                                I2C_vooo_2134(nob,nub,nob,nob),&
                                vB_ovvv_3421(nua,nub,nub,noa),&
                                vC_vovv_1342(nub,nub,nub,nob),&
                                vB_vovv_3412(nua,nub,nua,nob),&
                                vB_oovo_3412(nua,nob,noa,nob),&
                                vC_ooov_3412(nob,nub,nob,nob),&
                                vB_ooov_3412(noa,nub,noa,nob)

                        call reorder_stripe(4, shape(vB_vvov), size(vB_vvov), '1243', vB_vvov, vB_vvov_1243)
                        call reorder_stripe(4, shape(vC_vvov), size(vC_vvov), '4213', vC_vvov, vC_vvov_4213)
                        call reorder_stripe(4, shape(t2b), size(t2b), '1243', t2b, t2b_1243)
                        call reorder_stripe(4, shape(I2C_vooo), size(I2C_vooo), '2134', I2C_vooo, I2C_vooo_2134)
                        call reorder_stripe(4, shape(vB_ovvv), size(vB_ovvv), '3421', vB_ovvv, vB_ovvv_3421)
                        call reorder_stripe(4, shape(vC_vovv), size(vC_vovv), '1342', vC_vovv, vC_vovv_1342)
                        call reorder_stripe(4, shape(vB_vovv), size(vB_vovv), '3412', vB_vovv, vB_vovv_3412)
                        call reorder_stripe(4, shape(vB_oovo), size(vB_oovo), '3412', vB_oovo, vB_oovo_3412)
                        call reorder_stripe(4, shape(vC_ooov), size(vC_ooov), '3412', vC_ooov, vC_ooov_3412)
                        call reorder_stripe(4, shape(vB_ooov), size(vB_ooov), '3412', vB_ooov, vB_ooov_3412)

                        deltaA = 0.0d0
                        moments = 0.0d0

                        nuanub = nua*nub
                        nub2 = nub*nub
                        do i = 1 , noa
                            do j = 1, nob
                                do k = j+1, nob
                                    X3C = 0.0d0
                                    L3C = 0.0d0

                                    !!!!! MM(2,3)C -> V*T2 !!!!!
                                    ! Diagram 1: A(bc) H2B_vvov(a,b,i,e)*t2c(e,c,j,k)
                                    call dgemm('n','n',nuanub,nub,nub,1.0d0,vB_vvov_1243(:,:,:,i),nuanub,t2c(:,:,j,k),nub,1.0d0,X3C,nuanub)
                                    ! Diagram 2: -A(jk) I2B_vooo(a,m,i,j)*t2c(b,c,m,k)
                                    call dgemm('n','t',nua,nub2,nob,-0.5d0,I2B_vooo(:,:,i,j),nua,t2c(:,:,:,k),nub2,1.0d0,X3C,nua)
                                    call dgemm('n','t',nua,nub2,nob,0.5d0,I2B_vooo(:,:,i,k),nua,t2c(:,:,:,j),nub2,1.0d0,X3C,nua)
                                    ! Diagram 3: A(jk) H2C_vvov(c,b,k,e)*t2b(a,e,i,j)
                                    call dgemm('n','n',nua,nub2,nub,0.5d0,t2b(:,:,i,j),nua,vC_vvov_4213(:,:,:,k),nub,1.0d0,X3C,nua)
                                    call dgemm('n','n',nua,nub2,nub,-0.5d0,t2b(:,:,i,k),nua,vC_vvov_4213(:,:,:,j),nub,1.0d0,X3C,nua)
                                    ! Diagram 4: -A(bc) I2C_vooo(c,m,k,j)*t2b(a,b,i,m)
                                    call dgemm('n','n',nuanub,nub,nob,-1.0d0,t2b_1243(:,:,:,i),nuanub,I2C_vooo_2134(:,:,k,j),nob,1.0d0,X3C,nuanub)
                                    ! Diagram 5: A(jk)A(bc) H2B_vvvo(a,b,e,j)*t2b(e,c,i,k)
                                    call dgemm('n','n',nuanub,nub,nua,1.0d0,vB_vvvo(:,:,:,j),nuanub,t2b(:,:,i,k),nua,1.0d0,X3C,nuanub)
                                    call dgemm('n','n',nuanub,nub,nua,-1.0d0,vB_vvvo(:,:,:,k),nuanub,t2b(:,:,i,j),nua,1.0d0,X3C,nuanub)
                                    ! Diagram 6: -A(jk)A(bc) I2B_ovoo(m,b,i,j)*t2b(a,c,m,k) -> -A(jk)A(bc) I2B_ovoo(m,c,i,k)*t2b(a,b,m,j)
                                    call dgemm('n','n',nuanub,nub,noa,-1.0d0,t2b(:,:,:,j),nuanub,I2B_ovoo(:,:,i,k),noa,1.0d0,X3C,nuanub)
                                    call dgemm('n','n',nuanub,nub,noa,1.0d0,t2b(:,:,:,k),nuanub,I2B_ovoo(:,:,i,j),noa,1.0d0,X3C,nuanub)

                                    !!!!! L3C -> (T3[2])^+ !!!!!
                                    ! Diagram 1: A(bc) H2B_ovvv(i,e,a,b)*l2c(e,c,j,k)
                                    !call dgemm('n','n',nuanub,nub,nub,1.0d0,vB_ovvv_3421(:,:,:,i),nuanub,t2c(:,:,j,k),nub,1.0d0,L3C,nuanub)
                                    ! Diagram 2: A(jk) H2C_vovv(e,k,b,c)*l2b(a,e,i,j)
                                    !call dgemm('n','n',nua,nub2,nub,0.5d0,t2b(:,:,i,j),nua,vC_vovv_1342(:,:,:,k),nub,1.0d0,L3C,nua)
                                    !call dgemm('n','n',nua,nub2,nub,-0.5d0,t2b(:,:,i,k),nua,vC_vovv_1342(:,:,:,j),nub,1.0d0,L3C,nua)
                                    ! Diagram 3: A(jk)A(bc) H2B_vovv(e,j,a,b)*l2b(e,c,i,k)
                                    !call dgemm('n','n',nuanub,nub,nua,1.0d0,vB_vovv_3412(:,:,:,j),nuanub,t2b(:,:,i,k),nua,1.0d0,L3C,nuanub)
                                    !call dgemm('n','n',nuanub,nub,nua,-1.0d0,vB_vovv_3412(:,:,:,k),nuanub,t2b(:,:,i,j),nua,1.0d0,L3C,nuanub)
                                    ! Diagram 4: -A(jk) H2B_oovo(i,j,a,m)*l2c(b,c,m,k)
                                    !call dgemm('n','t',nua,nub2,nob,-0.5d0,vB_oovo_3412(:,:,i,j),nua,t2c(:,:,:,k),nub2,1.0d0,L3C,nua)
                                    !call dgemm('n','t',nua,nub2,nob,0.5d0,vB_oovo_3412(:,:,i,k),nua,t2c(:,:,:,j),nub2,1.0d0,L3C,nua)
                                    ! Diagram 5: -A(bc) H2C_ooov(j,k,m,c)*l2b(a,b,i,m)
                                    !call dgemm('n','n',nuanub,nub,nob,-1.0d0,t2b_1243(:,:,:,i),nuanub,vC_ooov_3412(:,:,j,k),nob,1.0d0,L3C,nuanub)
                                    ! Diagram 6: -A(jk)A(bc) H2B_ooov(i,j,m,b)*l2b(a,c,m,k) -> -A(jk)A(bc) H2B_ooov(i,k,m,c)*l2b(a,b,m,j)
                                    !call dgemm('n','n',nuanub,nub,noa,-1.0d0,t2b(:,:,:,j),nuanub,vB_ooov_3412(:,:,i,k),noa,1.0d0,L3C,nuanub)
                                    !call dgemm('n','n',nuanub,nub,noa,1.0d0,t2b(:,:,:,k),nuanub,vB_ooov_3412(:,:,i,j),noa,1.0d0,L3C,nuanub)

                                    do a = 1, nua
                                        do b = 1, nub
                                            do c = b+1, nub

                                                if (pspace(a,b,c,i,j,k)) cycle

                                                temp1 = X3C(a,b,c) - X3C(a,c,b)
                                                !temp2 = L3C(a,b,c) - L3C(a,c,b)
                                                temp3 = t1b(c,k)*vB_oovv(i,j,a,b)&
                                                        -t1b(b,k)*vB_oovv(i,j,a,c)&
                                                        -t1b(c,j)*vB_oovv(i,k,a,b)&
                                                        +t1b(b,j)*vB_oovv(i,k,a,c)&
                                                        +t1a(a,i)*vC_oovv(j,k,b,c)
                                                temp3 = temp3 &
                                                        +fB_ov(k,c)*t2b(a,b,i,j)&
                                                        -fB_ov(k,b)*t2b(a,c,i,j)&
                                                        -fB_ov(j,c)*t2b(a,b,i,k)&
                                                        +fB_ov(j,b)*t2b(a,c,i,k)&
                                                        +fA_ov(i,a)*t2c(b,c,j,k)

                                                LM = temp1*(temp1+temp3)

                                                D = fA_oo(i,i) + fB_oo(j,j) + fB_oo(k,k)&
                                                - fA_vv(a,a) - fB_vv(b,b) - fB_vv(c,c)

                                                deltaA = deltaA + LM/D

                                                moments(a,b,c,i,j,k) = LM/D
                                            end do
                                        end do
                                    end do
                                end do
                            end do
                        end do
              end subroutine ccsdptC_p_full_moment

              subroutine ccsdptD_p_full_moment(deltaA,moments,&
                                                  pspace,&
                                                  t1b,t2c,&
                                                  vC_vooo,I2C_vvov,vC_oovv,fB_ov,vC_vovv,vC_ooov,fB_oo,fB_vv,&
                                                  nob,nub)

                        real(kind=8), intent(out) :: deltaA
                        integer, intent(in) :: nob, nub
                        logical(kind=1), intent(in) :: pspace(nub,nub,nub,nob,nob,nob)
                        real(kind=8), intent(in) :: fB_oo(1:nob,1:nob),fB_vv(1:nub,1:nub),&
                        vC_vooo(nub,nob,nob,nob),I2C_vvov(nub,nub,nob,nub),t2c(nub,nub,nob,nob),&
                        t1b(nub,nob),vC_oovv(nob,nob,nub,nub),&
                        fB_ov(nob,nub),vC_vovv(nub,nob,nub,nub),vC_ooov(nob,nob,nob,nub)

                        real(kind=8), intent(out) :: moments(nub,nub,nub,nob,nob,nob)

                        integer :: i, j, k, a, b, c, nub2
                        real(kind=8) :: D, temp1, temp2, temp3, LM, X3D(nub,nub,nub), L3D(nub,nub,nub)

                        ! reordered arrays for DGEMMs
                        real(kind=8) :: I2C_vvov_1243(nub,nub,nub,nob), vC_vovv_4312(nub,nub,nub,nob), vC_ooov_4312(nub,nob,nob,nob)

                        call reorder_stripe(4, shape(I2C_vvov), size(I2C_vvov), '1243', I2C_vvov, I2C_vvov_1243)
                        call reorder_stripe(4, shape(vC_vovv), size(vC_vovv), '4312', vC_vovv, vC_vovv_4312)
                        call reorder_stripe(4, shape(vC_ooov), size(vC_ooov), '4312', vC_ooov, vC_ooov_4312)

                        deltaA = 0.0d0
                        moments = 0.0d0

                        nub2 = nub*nub
                        do i = 1 , nob
                            do j = i+1, nob
                                do k = j+1, nob

                                   X3D = 0.0d0
                                   L3D = 0.0d0
                                   !!!!! MM(2,3)D -> V*T2 !!!!!
                                   ! Diagram 1: -A(k/ij)A(a/bc) H2C_vooo(a,m,i,j)*t2c(b,c,m,k)
                                   call dgemm('n','t',nub,nub2,nob,-0.5d0,vC_vooo(:,:,i,j),nub,t2c(:,:,:,k),nub2,1.0d0,X3D,nub)
                                   call dgemm('n','t',nub,nub2,nob,0.5d0,vC_vooo(:,:,k,j),nub,t2c(:,:,:,i),nub2,1.0d0,X3D,nub)
                                   call dgemm('n','t',nub,nub2,nob,0.5d0,vC_vooo(:,:,i,k),nub,t2c(:,:,:,j),nub2,1.0d0,X3D,nub)
                                   ! Diagram 2: A(i/jk)A(c/ab) I2C_vvov(a,b,i,e)*t2c(e,c,j,k)
                                   call dgemm('n','n',nub2,nub,nub,0.5d0,I2C_vvov_1243(:,:,:,i),nub2,t2c(:,:,j,k),nub,1.0d0,X3D,nub2)
                                   call dgemm('n','n',nub2,nub,nub,-0.5d0,I2C_vvov_1243(:,:,:,j),nub2,t2c(:,:,i,k),nub,1.0d0,X3D,nub2)
                                   call dgemm('n','n',nub2,nub,nub,-0.5d0,I2C_vvov_1243(:,:,:,k),nub2,t2c(:,:,j,i),nub,1.0d0,X3D,nub2)
                                   !!!!! L3A -? (T3[2])^+ !!!!!
                                   ! Diagram 1: A(i/jk)A(c/ab) H2C_vovv(e,i,b,a)*l2c(e,c,j,k)
                                   !call dgemm('n','n',nub2,nub,nub,0.5d0,vC_vovv_4312(:,:,:,i),nub2,t2c(:,:,j,k),nub,1.0d0,L3D,nub2)
                                   !call dgemm('n','n',nub2,nub,nub,-0.5d0,vC_vovv_4312(:,:,:,j),nub2,t2c(:,:,i,k),nub,1.0d0,L3D,nub2)
                                   !call dgemm('n','n',nub2,nub,nub,-0.5d0,vC_vovv_4312(:,:,:,k),nub2,t2c(:,:,j,i),nub,1.0d0,L3D,nub2)
                                   ! Diagram 2: -A(k/ij)A(a/bc) H2A_ooov(j,i,m,a)*l2a(b,c,m,k)-> a,m,j,i * (b,c,m,k)'
                                   !call dgemm('n','t',nub,nub2,nob,-0.5d0,vC_ooov_4312(:,:,j,i),nub,t2c(:,:,:,k),nub2,1.0d0,L3D,nub)
                                   !call dgemm('n','t',nub,nub2,nob,0.5d0,vC_ooov_4312(:,:,k,i),nub,t2c(:,:,:,j),nub2,1.0d0,L3D,nub)
                                   !call dgemm('n','t',nub,nub2,nob,0.5d0,vC_ooov_4312(:,:,j,k),nub,t2c(:,:,:,i),nub2,1.0d0,L3D,nub)

                                    do a = 1, nub
                                        do b = a+1, nub
                                            do c = b+1, nub

                                                if (pspace(a,b,c,i,j,k)) cycle

                                                temp1 = X3D(a,b,c) + X3D(b,c,a) + X3D(c,a,b)&
                                                - X3D(a,c,b) - X3D(b,a,c) - X3D(c,b,a)

                                                !temp2 = L3D(a,b,c) + L3D(b,c,a) + L3D(c,a,b)&
                                                !- L3D(a,c,b) - L3D(b,a,c) - L3D(c,b,a)

                                                temp3 =&
                                                t1b(c,k)*vC_oovv(i,j,a,b)&
                                                -t1b(a,k)*vC_oovv(i,j,c,b)&
                                                -t1b(b,k)*vC_oovv(i,j,a,c)&
                                                -t1b(c,i)*vC_oovv(k,j,a,b)&
                                                -t1b(c,j)*vC_oovv(i,k,a,b)&
                                                +t1b(a,i)*vC_oovv(k,j,c,b)&
                                                +t1b(b,i)*vC_oovv(k,j,a,c)&
                                                +t1b(a,j)*vC_oovv(i,k,c,b)&
                                                +t1b(b,j)*vC_oovv(i,k,a,c)
                                                temp3 = temp3 &
                                                +fB_ov(k,c)*t2c(a,b,i,j)&
                                                -fB_ov(k,a)*t2c(c,b,i,j)&
                                                -fB_ov(k,b)*t2c(a,c,i,j)&
                                                -fB_ov(i,c)*t2c(a,b,k,j)&
                                                -fB_ov(j,c)*t2c(a,b,i,k)&
                                                +fB_ov(i,a)*t2c(c,b,k,j)&
                                                +fB_ov(i,b)*t2c(a,c,k,j)&
                                                +fB_ov(j,a)*t2c(c,b,i,k)&
                                                +fB_ov(j,b)*t2c(a,c,i,k)

                                                LM = temp1*(temp1+temp3)

                                                D = fB_oo(i,i) + fB_oo(j,j) + fB_oo(k,k)&
                                                - fB_vv(a,a) - fB_vv(b,b) - fB_vv(c,c)

                                                deltaA = deltaA + LM/D

                                                moments(a,b,c,i,j,k) = LM/D
                                            end do
                                        end do
                                    end do
                                end do
                            end do
                        end do

              end subroutine ccsdptD_p_full_moment

end module ccsdpt_loops
