module crcc_loops

  use constants, only: p
  use reorder, only: reorder_stripe

  implicit none

  contains

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! OPTIMIZED CR-CC(2,3) ROUTINES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine crcc23A_opt(deltaA,deltaB,deltaC,deltaD,&
    t2a,l1a,l2a,&
    H2A_vooo,I2A_vvov,vA_oovv,H1A_ov,H2A_vovv,H2A_ooov,fA_oo,fA_vv,&
    H1A_oo,H1A_vv,&
    H2A_voov,H2A_oooo,H2A_vvvv,&
    D3A_O,D3A_v,noa,nua)

    real(p), intent(out) :: deltaA, deltaB, deltaC, deltaD
    integer, intent(in) :: noa, nua
    real(p), intent(in) :: fA_oo(1:noa,1:noa)
    real(p), intent(in) :: fA_vv(1:nua,1:nua)
    real(p), intent(in) :: H1A_oo(1:noa,1:noa)
    real(p), intent(in) :: H1A_vv(1:nua,1:nua)
    real(p), intent(in) :: H2A_voov(1:nua,1:noa,1:noa,1:nua)
    real(p), intent(in) :: H2A_oooo(1:noa,1:noa,1:noa,1:noa)
    !real(p), intent(in) :: H2A_vvvv(1:nua,1:nua,1:nua,1:nua)
    real(p), intent(in) :: H2A_vvvv(1:nua,1:nua)
    real(p), intent(in) :: D3A_O(1:nua,1:noa,1:noa)
    real(p), intent(in) :: D3A_V(1:nua,1:noa,1:nua)
    real(p), intent(in) :: H2A_vooo(nua,noa,noa,noa)
    real(p), intent(in) :: I2A_vvov(nua,nua,noa,nua)
    real(p), intent(in) :: t2a(nua,nua,noa,noa)
    real(p), intent(in) :: l1a(nua,noa)
    real(p), intent(in) :: l2a(nua,nua,noa,noa)
    real(p), intent(in) :: vA_oovv(noa,noa,nua,nua)
    real(p), intent(in) :: H1A_ov(noa,nua)
    real(p), intent(in) :: H2A_vovv(nua,noa,nua,nua)
    real(p), intent(in) :: H2A_ooov(noa,noa,noa,nua)
    integer :: i, j, k, a, b, c, nua2
    real(p) :: D, temp1, temp2, temp3, LM, X3A(nua,nua,nua), L3A(nua,nua,nua)

    ! reordered arrays for DGEMMs
    real(p) :: I2A_vvov_1243(nua,nua,nua,noa), H2A_vovv_4312(nua,nua,nua,noa), H2A_ooov_4312(nua,noa,noa,noa)

    call reorder_stripe(4, shape(I2A_vvov), size(I2A_vvov), '1243', I2A_vvov, I2A_vvov_1243)
    call reorder_stripe(4, shape(H2A_vovv), size(H2A_vovv), '4312', H2A_vovv, H2A_vovv_4312)
    call reorder_stripe(4, shape(H2A_ooov), size(H2A_ooov), '4312', H2A_ooov, H2A_ooov_4312)

    deltaA = 0.0_p
    deltaB = 0.0_p
    deltaC = 0.0_p
    deltaD = 0.0_p

    nua2 = nua*nua
    do i = 1 , noa
      do j = i+1, noa
        do k = j+1, noa

          X3A = 0.0_p
          L3A = 0.0_p
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
                -H2A_vvvv(b,a) - H2A_vvvv(c,a) - H2A_vvvv(c,b)
                !-H2A_vvvv(b,a,b,a) - H2A_vvvv(c,a,c,a) - H2A_vvvv(c,b,c,b)

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

    real(p), intent(out) :: deltaA, deltaB, deltaC, deltaD
    integer, intent(in) :: noa, nua, nob, nub
    real(p), intent(in) :: H1A_ov(noa,nua)
    real(p), intent(in) :: H1B_ov(nob,nub)
    real(p), intent(in) :: vA_oovv(noa,noa,nua,nua)
    real(p), intent(in) :: vB_oovv(noa,nob,nua,nub)
    real(p), intent(in) :: fA_oo(1:noa,1:noa)
    real(p), intent(in) :: fA_vv(1:nua,1:nua)
    real(p), intent(in) :: fB_oo(1:nob,1:nob)
    real(p), intent(in) :: fB_vv(1:nub,1:nub)
    real(p), intent(in) :: H1A_oo(1:noa,1:noa)
    real(p), intent(in) :: H1A_vv(1:nua,1:nua)
    real(p), intent(in) :: H1B_oo(1:nob,1:nob)
    real(p), intent(in) :: H1B_vv(1:nub,1:nub)
    real(p), intent(in) :: t2a(nua,nua,noa,noa)
    real(p), intent(in) :: t2b(nua,nub,noa,nob)
    real(p), intent(in) :: l1a(nua,noa)
    real(p), intent(in) :: l1b(nub,nob)
    real(p), intent(in) :: l2a(nua,nua,noa,noa)
    real(p), intent(in) :: l2b(nua,nub,noa,nob)
    real(p), intent(in) :: I2B_ovoo(noa,nub,noa,nob)
    real(p), intent(in) :: I2B_vooo(nua,nob,noa,nob)
    real(p), intent(in) :: I2A_vooo(nua,noa,noa,noa)
    real(p), intent(in) :: H2B_vvvo(nua,nub,nua,nob)
    real(p), intent(in) :: H2B_vvov(nua,nub,noa,nub)
    real(p), intent(in) :: H2A_vvov(nua,nua,noa,nua)
    real(p), intent(in) :: H2B_vovv(nua,nob,nua,nub)
    real(p), intent(in) :: H2B_ovvv(noa,nub,nua,nub)
    real(p), intent(in) :: H2A_vovv(nua,noa,nua,nua)
    real(p), intent(in) :: H2B_ooov(noa,nob,noa,nub)
    real(p), intent(in) :: H2B_oovo(noa,nob,nua,nob)
    real(p), intent(in) :: H2A_ooov(noa,noa,noa,nua)
    real(p), intent(in) :: H2A_voov(1:nua,1:noa,1:noa,1:nua)
    real(p), intent(in) :: H2A_oooo(1:noa,1:noa,1:noa,1:noa)
    !real(p), intent(in) :: H2A_vvvv(1:nua,1:nua,1:nua,1:nua)
    real(p), intent(in) :: H2A_vvvv(1:nua,1:nua)
    real(p), intent(in) :: H2B_ovov(1:noa,1:nub,1:noa,1:nub)
    real(p), intent(in) :: H2B_vovo(1:nua,1:nob,1:nua,1:nob)
    real(p), intent(in) :: H2B_oooo(1:noa,1:nob,1:noa,1:nob)
    !real(p), intent(in) :: H2B_vvvv(1:nua,1:nub,1:nua,1:nub)
    real(p), intent(in) :: H2B_vvvv(1:nua,1:nub)
    real(p), intent(in) :: H2C_voov(1:nub,1:nob,1:nob,1:nub)
    real(p), intent(in) :: D3A_O(1:nua,1:noa,1:noa)
    real(p), intent(in) :: D3A_V(1:nua,1:noa,1:nua)
    real(p), intent(in) :: D3B_O(1:nua,1:noa,1:nob)
    real(p), intent(in) :: D3B_V(1:nua,1:noa,1:nub)
    real(p), intent(in) :: D3C_O(1:nub,1:noa,1:nob)
    real(p), intent(in) :: D3C_V(1:nua,1:nob,1:nub)
    integer :: i, j, k, a, b, c, nuanub, nua2
    real(p) :: D, temp1, temp2, temp3, LM, X3B(nua,nua,nub), L3B(nua,nua,nub)

    ! arrays for reordering 
    real(p) :: t2a_1243(nua,nua,noa,noa), H2B_vvov_1243(nua,nub,nub,noa),&
    t2b_1243(nua,nub,nob,noa), H2A_vvov_1243(nua,nua,nua,noa),&
    H2B_vovv_1342(nua,nua,nub,nob), H2A_vovv_4312(nua,nua,nua,noa),&
    H2B_ovvv_2341(nub,nua,nub,noa), H2B_ooov_3412(noa,nub,noa,nob),&
    H2A_ooov_4312(nua,noa,noa,noa), H2B_oovo_3412(nua,nob,noa,nob),&
    l2b_1243(nua,nub,nob,noa)

    call reorder_stripe(4, shape(t2a), size(t2a), '1243', t2a, t2a_1243)
    call reorder_stripe(4, shape(H2B_vvov), size(H2B_vvov), '1243', H2B_vvov, H2B_vvov_1243)
    call reorder_stripe(4, shape(t2b), size(t2b), '1243', t2b, t2b_1243)
    call reorder_stripe(4, shape(H2A_vvov), size(H2A_vvov), '1243', H2A_vvov, H2A_vvov_1243)
    call reorder_stripe(4, shape(H2B_vovv), size(H2B_vovv), '1342', H2B_vovv, H2B_vovv_1342)
    call reorder_stripe(4, shape(H2A_vovv), size(H2A_vovv), '4312', H2A_vovv, H2A_vovv_4312)
    call reorder_stripe(4, shape(H2B_ovvv), size(H2B_ovvv), '2341', H2B_ovvv, H2B_ovvv_2341)
    call reorder_stripe(4, shape(H2B_ooov), size(H2B_ooov), '3412', H2B_ooov, H2B_ooov_3412)
    call reorder_stripe(4, shape(H2A_ooov), size(H2A_ooov), '4312', H2A_ooov, H2A_ooov_4312)
    call reorder_stripe(4, shape(H2B_oovo), size(H2B_oovo), '3412', H2B_oovo, H2B_oovo_3412)
    call reorder_stripe(4, shape(l2b), size(l2b), '1243', l2b, l2b_1243)

    deltaA = 0.0_p
    deltaB = 0.0_p
    deltaC = 0.0_p
    deltaD = 0.0_p

    nuanub = nua*nub
    nua2 = nua*nua
    do i = 1, noa
      do j = i+1, noa
        do k = 1, nob

          X3B = 0.0_p
          L3B = 0.0_p
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
                -H2A_vvvv(b,a) - H2B_vvvv(a,c) - H2B_vvvv(b,c)
                !-H2A_vvvv(b,a,b,a)-H2B_vvvv(a,c,a,c)-H2B_vvvv(b,c,b,c)

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

    real(p), intent(out) :: deltaA, deltaB, deltaC, deltaD
    integer, intent(in) :: noa, nua, nob, nub
    real(p), intent(in) :: t2b(nua,nub,noa,nob),&
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
    !H2B_vvvv(1:nua,1:nub,1:nua,1:nub),&
    H2B_vvvv(1:nua,1:nub),&
    H2C_voov(1:nub,1:nob,1:nob,1:nub),&
    H2C_oooo(1:nob,1:nob,1:nob,1:nob),&
    !H2C_vvvv(1:nub,1:nub,1:nub,1:nub),&
    H2C_vvvv(1:nub,1:nub),&
    D3B_O(1:nua,1:noa,1:nob),&
    D3B_V(1:nua,1:noa,1:nub),&
    D3C_O(1:nub,1:noa,1:nob),&
    D3C_V(1:nua,1:nob,1:nub),&
    D3D_O(1:nub,1:nob,1:nob),&
    D3D_V(1:nub,1:nob,1:nub)
    integer :: i, j, k, a, b, c, nuanub, nub2
    real(p) :: D, LM, temp1, temp2, temp3, X3C(nua,nub,nub), L3C(nua,nub,nub)

    ! arrays for reordering
    real(p) :: H2B_vvov_1243(nua,nub,nub,noa),&
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

    deltaA = 0.0_p
    deltaB = 0.0_p
    deltaC = 0.0_p
    deltaD = 0.0_p

    nuanub = nua*nub
    nub2 = nub*nub

    call reorder_stripe(4, shape(H2B_vvov), size(H2B_vvov), '1243', H2B_vvov, H2B_vvov_1243)
    call reorder_stripe(4, shape(H2C_vvov), size(H2C_vvov), '4213', H2C_vvov, H2C_vvov_4213)
    call reorder_stripe(4, shape(t2b), size(t2b), '1243', t2b, t2b_1243)
    call reorder_stripe(4, shape(I2C_vooo), size(I2C_vooo), '2134', I2C_vooo, I2C_vooo_2134)
    call reorder_stripe(4, shape(H2B_ovvv), size(H2B_ovvv), '3421', H2B_ovvv, H2B_ovvv_3421)
    call reorder_stripe(4, shape(H2C_vovv), size(H2C_vovv), '1342', H2C_vovv, H2C_vovv_1342)
    call reorder_stripe(4, shape(H2B_vovv), size(H2B_vovv), '3412', H2B_vovv, H2B_vovv_3412)
    call reorder_stripe(4, shape(H2B_oovo), size(H2B_oovo), '3412', H2B_oovo, H2B_oovo_3412)
    call reorder_stripe(4, shape(H2C_ooov), size(H2C_ooov), '3412', H2C_ooov, H2C_ooov_3412)
    call reorder_stripe(4, shape(l2b), size(l2b), '1243', l2b, l2b_1243)
    call reorder_stripe(4, shape(H2B_ooov), size(H2B_ooov), '3412', H2B_ooov, H2B_ooov_3412)

    do i = 1 , noa
      do j = 1, nob
        do k = j+1, nob
          X3C = 0.0_p
          L3C = 0.0_p

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
                -H2B_vvvv(a,b)-H2B_vvvv(a,c)-H2C_vvvv(c,b)
                !-H2B_vvvv(a,b,a,b)-H2B_vvvv(a,c,a,c)-H2C_vvvv(c,b,c,b)

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

    real(p), intent(out) :: deltaA, deltaB, deltaC, deltaD
    integer, intent(in) :: nob, nub
    real(p), intent(in) :: fB_oo(1:nob,1:nob),fB_vv(1:nub,1:nub),&
    H1B_oo(1:nob,1:nob),H1B_vv(1:nub,1:nub),&
    H2C_voov(1:nub,1:nob,1:nob,1:nub),&
    H2C_oooo(1:nob,1:nob,1:nob,1:nob),&
    !H2C_vvvv(1:nub,1:nub,1:nub,1:nub),&
    H2C_vvvv(1:nub,1:nub),&
    D3D_O(1:nub,1:nob,1:nob),&
    D3D_V(1:nub,1:nob,1:nub),&
    H2C_vooo(nub,nob,nob,nob),I2C_vvov(nub,nub,nob,nub),t2c(nub,nub,nob,nob),&
    l1b(nub,nob),l2c(nub,nub,nob,nob),vC_oovv(nob,nob,nub,nub),&
    H1B_ov(nob,nub),H2C_vovv(nub,nob,nub,nub),H2C_ooov(nob,nob,nob,nub)
    integer :: i, j, k, a, b, c, nub2
    real(p) :: D, temp1, temp2, temp3, LM, X3D(nub,nub,nub), L3D(nub,nub,nub)

    ! reordered arrays for DGEMMs
    real(p) :: I2C_vvov_1243(nub,nub,nub,nob), H2C_vovv_4312(nub,nub,nub,nob), H2C_ooov_4312(nub,nob,nob,nob)

    call reorder_stripe(4, shape(I2C_vvov), size(I2C_vvov), '1243', I2C_vvov, I2C_vvov_1243)
    call reorder_stripe(4, shape(H2C_vovv), size(H2C_vovv), '4312', H2C_vovv, H2C_vovv_4312)
    call reorder_stripe(4, shape(H2C_ooov), size(H2C_ooov), '4312', H2C_ooov, H2C_ooov_4312)

    deltaA = 0.0_p
    deltaB = 0.0_p
    deltaC = 0.0_p
    deltaD = 0.0_p

    nub2 = nub*nub
    do i = 1 , nob
      do j = i+1, nob
        do k = j+1, nob

          X3D = 0.0_p
          L3D = 0.0_p
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
                -H2C_vvvv(b,a) - H2C_vvvv(c,a) - H2C_vvvv(c,b)
                !-H2C_vvvv(b,a,b,a) - H2C_vvvv(c,a,c,a) - H2C_vvvv(c,b,c,b)

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
    ddeltaA,ddeltaB,ddeltaC,ddeltaD,&
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

    real(p), intent(out) :: deltaA, deltaB, deltaC, deltaD
    real(p), intent(out) :: ddeltaA, ddeltaB, ddeltaC, ddeltaD
    integer, intent(in) :: noa, nua
    real(p), intent(in) :: fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),&
    H1A_oo(1:noa,1:noa),H1A_vv(1:nua,1:nua),&
    H2A_voov(1:nua,1:noa,1:noa,1:nua),&
    H2A_oooo(1:noa,1:noa,1:noa,1:noa),&
    !H2A_vvvv(1:nua,1:nua,1:nua,1:nua),&
    H2A_vvvv(1:nua,1:nua),&
    D3A_O(1:nua,1:noa,1:noa),&
    D3A_V(1:nua,1:noa,1:nua),&
    H2A_vooo(nua,noa,noa,noa),I2A_vvov(nua,nua,noa,nua),t2a(nua,nua,noa,noa),&
    l1a(nua,noa),l2a(nua,nua,noa,noa),vA_oovv(noa,noa,nua,nua),&
    H1A_ov(noa,nua),H2A_vovv(nua,noa,nua,nua),H2A_ooov(noa,noa,noa,nua),&
    H2A_vvov(nua,nua,noa,nua),r2a(nua,nua,noa,noa),&
    chi2A_vvvo(nua,nua,nua,noa),chi2A_ovoo(noa,nua,noa,noa)
    real(p), intent(in) :: r0, omega

    integer :: i, j, k, a, b, c, nua2
    real(p) :: D, temp1, temp2, temp3, temp4, LM, LM1,&
    Y3A(nua,nua,nua), X3A(nua,nua,nua), L3A(nua,nua,nua)

    ! reordered arrays for DGEMMs
    real(p) :: I2A_vvov_1243(nua,nua,nua,noa), H2A_vovv_4312(nua,nua,nua,noa),&
    H2A_ooov_4312(nua,noa,noa,noa), H2A_vvov_2143(nua,nua,nua,noa),&
    H2A_vooo_2143(noa,nua,noa,noa)

    call reorder_stripe(4, shape(I2A_vvov), size(I2A_vvov), '1243', I2A_vvov, I2A_vvov_1243)
    call reorder_stripe(4, shape(H2A_vovv), size(H2A_vovv), '4312', H2A_vovv, H2A_vovv_4312)
    call reorder_stripe(4, shape(H2A_ooov), size(H2A_ooov), '4312', H2A_ooov, H2A_ooov_4312)
    call reorder_stripe(4, shape(H2A_vvov), size(H2A_vvov), '2143', H2A_vvov, H2A_vvov_2143)
    call reorder_stripe(4, shape(H2A_vooo), size(H2A_vooo), '2143', H2A_vooo, H2A_vooo_2143)

    deltaA = 0.0_p
    deltaB = 0.0_p
    deltaC = 0.0_p
    deltaD = 0.0_p

    ddeltaA = 0.0_p
    ddeltaB = 0.0_p
    ddeltaC = 0.0_p
    ddeltaD = 0.0_p

    nua2 = nua*nua
    do i = 1 , noa
      do j = i+1, noa
        do k = j+1, noa

          X3A = 0.0_p
          Y3A = 0.0_p
          L3A = 0.0_p
          !!!!! EOMMM(2,3)A !!!!!
          ! Diagram 1a: A(j/ik)A(c/ab) chi2A_vvvo(a,b,e,j)*t2a(e,c,i,k)
          call dgemm('n','n',nua2,nua,nua,0.5d0,chi2A_vvvo(:,:,:,j),nua2,t2a(:,:,i,k),nua,1.0d0,Y3A,nua2)
          call dgemm('n','n',nua2,nua,nua,-0.5d0,chi2A_vvvo(:,:,:,k),nua2,t2a(:,:,i,j),nua,1.0d0,Y3A,nua2)
          call dgemm('n','n',nua2,nua,nua,-0.5d0,chi2A_vvvo(:,:,:,i),nua2,t2a(:,:,j,k),nua,1.0d0,Y3A,nua2)
          ! Diagram 1b: A(j/ik)A(c/ab) H2A_vvov(b,a,j,e)*r2a(e,c,i,k)
          call dgemm('n','n',nua2,nua,nua,0.5d0,H2A_vvov_2143(:,:,:,j),nua2,r2a(:,:,i,k),nua,1.0d0,Y3A,nua2)
          call dgemm('n','n',nua2,nua,nua,-0.5d0,H2A_vvov_2143(:,:,:,k),nua2,r2a(:,:,i,j),nua,1.0d0,Y3A,nua2)
          call dgemm('n','n',nua2,nua,nua,-0.5d0,H2A_vvov_2143(:,:,:,i),nua2,r2a(:,:,j,k),nua,1.0d0,Y3A,nua2)
          ! Diagram 2a: -A(k/ij)A(b/ac) chi2A_ovoo(m,b,i,j)*t2a(a,c,m,k)
          call dgemm('n','n',nua2,nua,noa,-0.5d0,t2a(:,:,:,j),nua2,chi2A_ovoo(:,:,i,k),noa,1.0d0,Y3A,nua2)
          call dgemm('n','n',nua2,nua,noa,0.5d0,t2a(:,:,:,i),nua2,chi2A_ovoo(:,:,j,k),noa,1.0d0,Y3A,nua2)
          call dgemm('n','n',nua2,nua,noa,0.5d0,t2a(:,:,:,k),nua2,chi2A_ovoo(:,:,i,j),noa,1.0d0,Y3A,nua2)
          ! Diagram 2b: -A(k/ij)A(b/ac) H2A_vooo(b,m,j,i)*r2a(a,c,m,k)
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
                LM1 = temp4*(temp2+temp3)

                D = fA_oo(i,i) + fA_oo(j,j) + fA_oo(k,k)&
                - fA_vv(a,a) - fA_vv(b,b) - fA_vv(c,c)

                deltaA = deltaA + LM/(omega+D)
                ddeltaA = ddeltaA + LM1/(omega+D)

                D = H1A_oo(i,i) + H1A_oo(j,j) + H1A_oo(k,k)&
                - H1A_vv(a,a) - H1A_vv(b,b) - H1A_vv(c,c)

                deltaB = deltaB + LM/(omega+D)
                ddeltaB = ddeltaB + LM1/(omega+D)

                D = D &
                -H2A_voov(a,i,i,a) - H2A_voov(b,i,i,b) - H2A_voov(c,i,i,c)&
                -H2A_voov(a,j,j,a) - H2A_voov(b,j,j,b) - H2A_voov(c,j,j,c)&
                -H2A_voov(a,k,k,a) - H2A_voov(b,k,k,b) - H2A_voov(c,k,k,c)&
                -H2A_oooo(j,i,j,i) - H2A_oooo(k,i,k,i) - H2A_oooo(k,j,k,j)&
                -H2A_vvvv(b,a) - H2A_vvvv(c,a) - H2A_vvvv(c,b)
                !-H2A_vvvv(b,a,b,a) - H2A_vvvv(c,a,c,a) - H2A_vvvv(c,b,c,b)

                deltaC = deltaC + LM/(omega+D)
                ddeltaC = ddeltaC + LM1/(omega+D)

                D = D &
                +D3A_O(a,i,j)+D3A_O(a,i,k)+D3A_O(a,j,k)&
                +D3A_O(b,i,j)+D3A_O(b,i,k)+D3A_O(b,j,k)&
                +D3A_O(c,i,j)+D3A_O(c,i,k)+D3A_O(c,j,k)&
                -D3A_V(a,i,b)-D3A_V(a,i,c)-D3A_V(b,i,c)&
                -D3A_V(a,j,b)-D3A_V(a,j,c)-D3A_V(b,j,c)&
                -D3A_V(a,k,b)-D3A_V(a,k,c)-D3A_V(b,k,c)

                deltaD = deltaD + LM/(omega+D)
                ddeltaD = ddeltaD + LM1/(omega+D)
              end do
            end do 
          end do 
        end do 
      end do 
    end do
  end subroutine creomcc23A_opt

  subroutine creomcc23B_opt(deltaA,deltaB,deltaC,deltaD,&
    ddeltaA,ddeltaB,ddeltaC,ddeltaD,&
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

    real(p), intent(out) :: deltaA, deltaB, deltaC, deltaD
    real(p), intent(out) :: ddeltaA, ddeltaB, ddeltaC, ddeltaD
    integer, intent(in) :: noa, nua, nob, nub
    real(p), intent(in) :: t2a(nua,nua,noa,noa),t2b(nua,nub,noa,nob),&
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
    !H2A_vvvv(1:nua,1:nua,1:nua,1:nua),&
    H2A_vvvv(1:nua,1:nua),&
    H2B_ovov(1:noa,1:nub,1:noa,1:nub),&
    H2B_vovo(1:nua,1:nob,1:nua,1:nob),&
    H2B_oooo(1:noa,1:nob,1:noa,1:nob),&
    !H2B_vvvv(1:nua,1:nub,1:nua,1:nub),&
    H2B_vvvv(1:nua,1:nub),&
    H2C_voov(1:nub,1:nob,1:nob,1:nub),&
    D3A_O(1:nua,1:noa,1:noa),&
    D3A_V(1:nua,1:noa,1:nua),&
    D3B_O(1:nua,1:noa,1:nob),&
    D3B_V(1:nua,1:noa,1:nub),&
    D3C_O(1:nub,1:noa,1:nob),&
    D3C_V(1:nua,1:nob,1:nub)
    real(p), intent(in) :: omega, r0

    integer :: i, j, k, a, b, c, nuanub, nua2
    real(p) :: D, temp1, temp2, temp3, temp4, LM, LM1,&
    X3B(nua,nua,nub), L3B(nua,nua,nub), Y3B(nua,nua,nub)

    ! arrays for reordering 
    real(p) :: t2a_1243(nua,nua,noa,noa), H2B_vvov_1243(nua,nub,nub,noa),&
    t2b_1243(nua,nub,nob,noa), H2A_vvov_1243(nua,nua,nua,noa),&
    H2B_vovv_1342(nua,nua,nub,nob), H2A_vovv_4312(nua,nua,nua,noa),&
    H2B_ovvv_2341(nub,nua,nub,noa), H2B_ooov_3412(noa,nub,noa,nob),&
    H2A_ooov_4312(nua,noa,noa,noa), H2B_oovo_3412(nua,nob,noa,nob),&
    l2b_1243(nua,nub,nob,noa), H2A_vvov_2143(nua,nua,nua,noa),&
    chi2B_vvov_4123(nub,nua,nub,noa),H2B_vvov_4123(nub,nua,nub,noa),&
    r2b_1243(nua,nub,nob,noa)

    call reorder_stripe(4, shape(t2a), size(t2a), '1243', t2a, t2a_1243)
    call reorder_stripe(4, shape(H2B_vvov), size(H2B_vvov), '1243', H2B_vvov, H2B_vvov_1243)
    call reorder_stripe(4, shape(t2b), size(t2b), '1243', t2b, t2b_1243)
    call reorder_stripe(4, shape(H2A_vvov), size(H2A_vvov), '1243', H2A_vvov, H2A_vvov_1243)
    call reorder_stripe(4, shape(H2B_vovv), size(H2B_vovv), '1342', H2B_vovv, H2B_vovv_1342)
    call reorder_stripe(4, shape(H2A_vovv), size(H2A_vovv), '4312', H2A_vovv, H2A_vovv_4312)
    call reorder_stripe(4, shape(H2B_ovvv), size(H2B_ovvv), '2341', H2B_ovvv, H2B_ovvv_2341)
    call reorder_stripe(4, shape(H2B_ooov), size(H2B_ooov), '3412', H2B_ooov, H2B_ooov_3412)
    call reorder_stripe(4, shape(H2A_ooov), size(H2A_ooov), '4312', H2A_ooov, H2A_ooov_4312)
    call reorder_stripe(4, shape(H2B_oovo), size(H2B_oovo), '3412', H2B_oovo, H2B_oovo_3412)
    call reorder_stripe(4, shape(l2b), size(l2b), '1243', l2b, l2b_1243)
    call reorder_stripe(4, shape(H2A_vvov), size(H2A_vvov), '2143', H2A_vvov, H2A_vvov_2143)
    call reorder_stripe(4, shape(chi2B_vvov), size(chi2B_vvov), '4123', chi2B_vvov, chi2B_vvov_4123)
    call reorder_stripe(4, shape(H2B_vvov), size(H2B_vvov), '4123', H2B_vvov, H2B_vvov_4123)
    call reorder_stripe(4, shape(r2b), size(r2b), '1243', r2b, r2b_1243)

    deltaA = 0.0_p
    deltaB = 0.0_p
    deltaC = 0.0_p
    deltaD = 0.0_p

    ddeltaA = 0.0_p
    ddeltaB = 0.0_p
    ddeltaC = 0.0_p
    ddeltaD = 0.0_p

    nuanub = nua*nub
    nua2 = nua*nua
    do i = 1, noa
      do j = i+1, noa
        do k = 1, nob

          X3B = 0.0_p
          L3B = 0.0_p
          Y3B = 0.0_p
          !!!!! EOMMM(2,3)B !!!!!
          ! Diagram 1a: A(ab) chi2B_vvvo(b,c,e,k)*t2a(a,e,i,j)
          call dgemm('n','t',nua,nuanub,nua,1.0d0,t2a(:,:,i,j),nua,chi2B_vvvo(:,:,:,k),nuanub,1.0d0,Y3B,nua)
          ! Diagram 1b: A(ab) H2B_vvvo(b,c,e,k)*r2a(a,e,i,j)
          call dgemm('n','t',nua,nuanub,nua,1.0d0,r2a(:,:,i,j),nua,H2B_vvvo(:,:,:,k),nuanub,1.0d0,Y3B,nua)
          ! Diagram 2a: -A(ij) chi2B_ovoo(n,c,j,k)*t2a(a,b,i,n)
          call dgemm('n','n',nua2,nub,noa,0.5d0,t2a(:,:,:,i),nua2,chi2B_ovoo(:,:,j,k),noa,1.0d0,Y3B,nua2)
          call dgemm('n','n',nua2,nub,noa,-0.5d0,t2a(:,:,:,j),nua2,chi2B_ovoo(:,:,i,k),noa,1.0d0,Y3B,nua2)
          ! Diagram 2b: -A(ij) H2B_ovoo(n,c,j,k)*r2a(a,b,i,n)
          call dgemm('n','n',nua2,nub,noa,0.5d0,r2a(:,:,:,i),nua2,H2B_ovoo(:,:,j,k),noa,1.0d0,Y3B,nua2)
          call dgemm('n','n',nua2,nub,noa,-0.5d0,r2a(:,:,:,j),nua2,H2B_ovoo(:,:,i,k),noa,1.0d0,Y3B,nua2)
          ! Diagram 3a: A(ij) chi2A_vvvo(a,b,e,j)*t2b(e,c,i,k)
          call dgemm('n','n',nua2,nub,nua,0.5d0,chi2A_vvvo(:,:,:,j),nua2,t2b(:,:,i,k),nua,1.0d0,Y3B,nua2)
          call dgemm('n','n',nua2,nub,nua,-0.5d0,chi2A_vvvo(:,:,:,i),nua2,t2b(:,:,j,k),nua,1.0d0,Y3B,nua2)
          ! Diagram 3b: A(ij) H2A_vvov(b,a,j,e)*r2b(e,c,i,k)
          call dgemm('n','n',nua2,nub,nua,0.5d0,H2A_vvov_2143(:,:,:,j),nua2,r2b(:,:,i,k),nua,1.0d0,Y3B,nua2)
          call dgemm('n','n',nua2,nub,nua,-0.5d0,H2A_vvov_2143(:,:,:,i),nua2,r2b(:,:,j,k),nua,1.0d0,Y3B,nua2)
          ! Diagram 4a: -A(ab) chi2A_vooo(b,n,j,i)*t2b(a,c,n,k) -> -A(ab) chi2A_vooo(a,n,i,j)*t2b(b,c,n,k)
          call dgemm('n','t',nua,nuanub,noa,-1.0d0,chi2A_vooo(:,:,i,j),nua,t2b(:,:,:,k),nuanub,1.0d0,Y3B,nua)
          ! Diagram 4b: -A(ab) H2A_vooo(b,n,j,i)*r2b(a,c,n,k)
          call dgemm('n','t',nua,nuanub,noa,-1.0d0,H2A_vooo(:,:,i,j),nua,r2b(:,:,:,k),nuanub,1.0d0,Y3B,nua)
          ! Diagram 5a: A(ij)A(ab) chi2B_vvov(b,c,j,e)*t2b(a,e,i,k)
          call dgemm('n','n',nua,nuanub,nub,1.0d0,t2b(:,:,i,k),nua,chi2B_vvov_4123(:,:,:,j),nub,1.0d0,Y3B,nua)
          call dgemm('n','n',nua,nuanub,nub,-1.0d0,t2b(:,:,j,k),nua,chi2B_vvov_4123(:,:,:,i),nub,1.0d0,Y3B,nua)
          ! Diagram 5b: A(ij)A(ab) H2B_vvov(b,c,j,e)*r2b(a,e,i,k)
          call dgemm('n','n',nua,nuanub,nub,1.0d0,r2b(:,:,i,k),nua,H2B_vvov_4123(:,:,:,j),nub,1.0d0,Y3B,nua)
          call dgemm('n','n',nua,nuanub,nub,-1.0d0,r2b(:,:,j,k),nua,H2B_vvov_4123(:,:,:,i),nub,1.0d0,Y3B,nua)
          ! Diagram 6a: -A(ij)A(ab) chi2B_vooo(b,n,j,k)*t2b(a,c,i,n) -> -A(ij)A(ab) chi2B_vooo(a,n,i,k)*t2b(b,c,j,n)
          call dgemm('n','t',nua,nuanub,nob,-1.0d0,chi2B_vooo(:,:,i,k),nua,t2b_1243(:,:,:,j),nuanub,1.0d0,Y3B,nua)
          call dgemm('n','t',nua,nuanub,nob,1.0d0,chi2B_vooo(:,:,j,k),nua,t2b_1243(:,:,:,i),nuanub,1.0d0,Y3B,nua)
          ! Diagram 6b: -A(ij)A(ab) H2B_vooo(b,n,j,k)*r2b(a,c,i,n)
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
                LM1 = temp4*(temp2+temp3)

                D = fA_oo(i,i) + fA_oo(j,j) + fB_oo(k,k)&
                - fA_vv(a,a) - fA_vv(b,b) - fB_vv(c,c)

                deltaA = deltaA + LM/(omega+D)
                ddeltaA = ddeltaA + LM1/(omega+D)

                D = H1A_oo(i,i) + H1A_oo(j,j) + H1B_oo(k,k)&
                - H1A_vv(a,a) - H1A_vv(b,b) - H1B_vv(c,c)

                deltaB = deltaB + LM/(omega+D)
                ddeltaB = ddeltaB + LM1/(omega+D)

                D = D &
                -H2A_voov(a,i,i,a)-H2A_voov(b,i,i,b)+H2B_ovov(i,c,i,c)&
                -H2A_voov(a,j,j,a)-H2A_voov(b,j,j,b)+H2B_ovov(j,c,j,c)&
                +H2B_vovo(a,k,a,k)+H2B_vovo(b,k,b,k)-H2C_voov(c,k,k,c)&
                -H2A_oooo(j,i,j,i)-H2B_oooo(i,k,i,k)-H2B_oooo(j,k,j,k)&
                -H2A_vvvv(b,a) - H2B_vvvv(a,c) - H2B_vvvv(b,c)
                !-H2A_vvvv(b,a,b,a)-H2B_vvvv(a,c,a,c)-H2B_vvvv(b,c,b,c)

                deltaC = deltaC + LM/(omega+D)
                ddeltaC = ddeltaC + LM1/(omega+D)

                D = D &
                +D3A_O(a,i,j)+D3B_O(a,i,k)+D3B_O(a,j,k)&
                +D3A_O(b,i,j)+D3B_O(b,i,k)+D3B_O(b,j,k)&
                +D3C_O(c,i,k)+D3C_O(c,j,k)&
                -D3A_V(a,i,b)-D3B_V(a,i,c)-D3B_V(b,i,c)&
                -D3A_V(a,j,b)-D3B_V(a,j,c)-D3B_V(b,j,c)&
                -D3C_V(a,k,c)-D3C_V(b,k,c)

                deltaD = deltaD + LM/(omega+D)
                ddeltaD = ddeltaD + LM1/(omega+D)
              end do
            end do 
          end do 
        end do 
      end do 
    end do
  end subroutine creomcc23B_opt

  subroutine creomcc23C_opt(deltaA,deltaB,deltaC,deltaD,&
    ddeltaA,ddeltaB,ddeltaC,ddeltaD,&
    omega,r0,&
    t2b,t2c,r2b,r2c,l1a,l1b,l2b,l2c,&
    I2B_vooo,I2C_vooo,I2B_ovoo,&
    H2B_vvov,H2C_vvov,H2B_vvvo,&
    H2B_ovvv,H2B_vovv,H2C_vovv,&
    H2B_oovo,H2B_ooov,H2C_ooov,&
    chi2B_vvov,chi2B_vooo,chi2C_vvvo,&
    chi2C_vooo,chi2B_vvvo,chi2B_ovoo,&
    H2B_vooo,H2C_vooo,H2B_ovoo,&
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

    real(p), intent(out) :: deltaA, deltaB, deltaC, deltaD
    real(p), intent(out) :: ddeltaA, ddeltaB, ddeltaC, ddeltaD
    integer, intent(in) :: noa, nua, nob, nub
    real(p), intent(in) :: t2b(nua,nub,noa,nob),&
    t2c(nub,nub,nob,nob),r2b(nua,nub,noa,nob),r2c(nub,nub,nob,nob),&
    l1a(nua,noa),l1b(nub,nob),&
    l2b(nua,nub,noa,nob),l2c(nub,nub,nob,nob),&
    I2B_vooo(nua,nob,noa,nob),I2C_vooo(nub,nob,nob,nob),&
    I2B_ovoo(noa,nub,noa,nob),H2B_vvov(nua,nub,noa,nub),&
    H2C_vvov(nub,nub,nob,nub),H2B_vvvo(nua,nub,nua,nob),&
    H2B_ovvv(noa,nub,nua,nub),H2B_vovv(nua,nob,nua,nub),&
    H2C_vovv(nub,nob,nub,nub),H2B_oovo(noa,nob,nua,nob),&
    H2B_ooov(noa,nob,noa,nub),H2C_ooov(nob,nob,nob,nub),&
    chi2B_vvov(nua,nub,noa,nub),chi2B_vooo(nua,nob,noa,nob),&
    chi2C_vvvo(nub,nub,nub,nob),chi2C_vooo(nub,nob,nob,nob),&
    chi2B_vvvo(nua,nub,nua,nob),chi2B_ovoo(noa,nub,noa,nob),&
    H2B_vooo(nua,nob,noa,nob),H2C_vooo(nub,nob,nob,nob),H2B_ovoo(noa,nub,noa,nob),&
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
    !H2B_vvvv(1:nua,1:nub,1:nua,1:nub),&
    H2B_vvvv(1:nua,1:nub),&
    H2C_voov(1:nub,1:nob,1:nob,1:nub),&
    H2C_oooo(1:nob,1:nob,1:nob,1:nob),&
    !H2C_vvvv(1:nub,1:nub,1:nub,1:nub),&
    H2C_vvvv(1:nub,1:nub),&
    D3B_O(1:nua,1:noa,1:nob),&
    D3B_V(1:nua,1:noa,1:nub),&
    D3C_O(1:nub,1:noa,1:nob),&
    D3C_V(1:nua,1:nob,1:nub),&
    D3D_O(1:nub,1:nob,1:nob),&
    D3D_V(1:nub,1:nob,1:nub)
    real(p), intent(in) :: omega, r0

    integer :: i, j, k, a, b, c, nuanub, nub2
    real(p) :: D, LM, LM1, temp1, temp2, temp3, temp4,&
    X3C(nua,nub,nub), L3C(nua,nub,nub), Y3C(nua,nub,nub)

    ! arrays for reordering
    real(p) :: H2B_vvov_1243(nua,nub,nub,noa),&
    H2C_vvov_4213(nub,nub,nub,noa),&
    t2b_1243(nua,nub,nob,noa),&
    I2C_vooo_2134(nob,nub,nob,nob),&
    H2B_ovvv_3421(nua,nub,nub,noa),&
    H2C_vovv_1342(nub,nub,nub,nob),&
    H2B_vovv_3412(nua,nub,nua,nob),&
    H2B_oovo_3412(nua,nob,noa,nob),&
    H2C_ooov_3412(nob,nub,nob,nob),&
    l2b_1243(nua,nub,nob,noa),&
    H2B_ooov_3412(noa,nub,noa,nob),&
    chi2B_vvov_1243(nua,nub,nub,noa),&
    chi2C_vvvo_3214(nub,nub,nub,nob),&
    H2C_vvov_4123(nub,nub,nub,nob),&
    chi2C_vooo_2134(nob,nub,nob,nob),&
    r2b_1243(nua,nub,nob,noa),&
    H2C_vooo_2134(nob,nub,nob,nob)

    deltaA = 0.0_p
    deltaB = 0.0_p
    deltaC = 0.0_p
    deltaD = 0.0_p

    ddeltaA = 0.0_p
    ddeltaB = 0.0_p
    ddeltaC = 0.0_p
    ddeltaD = 0.0_p

    nuanub = nua*nub
    nub2 = nub*nub

    call reorder_stripe(4, shape(H2B_vvov), size(H2B_vvov), '1243', H2B_vvov, H2B_vvov_1243)
    call reorder_stripe(4, shape(H2C_vvov), size(H2C_vvov), '4213', H2C_vvov, H2C_vvov_4213)
    call reorder_stripe(4, shape(t2b), size(t2b), '1243', t2b, t2b_1243)
    call reorder_stripe(4, shape(I2C_vooo), size(I2C_vooo), '2134', I2C_vooo, I2C_vooo_2134)
    call reorder_stripe(4, shape(H2B_ovvv), size(H2B_ovvv), '3421', H2B_ovvv, H2B_ovvv_3421)
    call reorder_stripe(4, shape(H2C_vovv), size(H2C_vovv), '1342', H2C_vovv, H2C_vovv_1342)
    call reorder_stripe(4, shape(H2B_vovv), size(H2B_vovv), '3412', H2B_vovv, H2B_vovv_3412)
    call reorder_stripe(4, shape(H2B_oovo), size(H2B_oovo), '3412', H2B_oovo, H2B_oovo_3412)
    call reorder_stripe(4, shape(H2C_ooov), size(H2C_ooov), '3412', H2C_ooov, H2C_ooov_3412)
    call reorder_stripe(4, shape(l2b), size(l2b), '1243', l2b, l2b_1243)
    call reorder_stripe(4, shape(H2B_ooov), size(H2B_ooov), '3412', H2B_ooov, H2B_ooov_3412)
    call reorder_stripe(4, shape(chi2B_vvov), size(chi2B_vvov), '1243', chi2B_vvov, chi2B_vvov_1243)
    call reorder_stripe(4, shape(chi2C_vvvo), size(chi2C_vvvo), '3214', chi2C_vvvo, chi2C_vvvo_3214)
    call reorder_stripe(4, shape(H2C_vvov), size(H2C_vvov), '4123', H2C_vvov, H2C_vvov_4123)
    call reorder_stripe(4, shape(chi2C_vooo), size(chi2C_vooo), '2134', chi2C_vooo, chi2C_vooo_2134)
    call reorder_stripe(4, shape(r2b), size(r2b), '1243', r2b, r2b_1243)
    call reorder_stripe(4, shape(H2C_vooo), size(H2C_vooo), '2134', H2C_vooo, H2C_vooo_2134)

    do i = 1 , noa
      do j = 1, nob
        do k = j+1, nob
          X3C = 0.0_p
          L3C = 0.0_p
          Y3C = 0.0_p

          !!!!! EOMMM(2,3)C !!!!!
          ! Diagram 1a: A(bc) chi2B_vvov(a,b,i,e)*t2c(e,c,j,k) 
          call dgemm('n','n',nuanub,nub,nub,1.0d0,chi2B_vvov_1243(:,:,:,i),nuanub,t2c(:,:,j,k),nub,1.0d0,Y3C,nuanub)
          ! Diagram 1b: A(bc) H2B_vvov(a,b,i,e)*r2c(e,c,j,k)  
          call dgemm('n','n',nuanub,nub,nub,1.0d0,H2B_vvov_1243(:,:,:,i),nuanub,r2c(:,:,j,k),nub,1.0d0,Y3C,nuanub)
          ! Diagram 2a: -A(jk) chi2B_vooo(a,n,i,j)*t2c(b,c,n,k)
          call dgemm('n','t',nua,nub2,nob,-0.5d0,chi2B_vooo(:,:,i,j),nua,t2c(:,:,:,k),nub2,1.0d0,Y3C,nua) 
          call dgemm('n','t',nua,nub2,nob,0.5d0,chi2B_vooo(:,:,i,k),nua,t2c(:,:,:,j),nub2,1.0d0,Y3C,nua) 
          ! Diagram 2b: -A(jk) H2B_vooo(a,n,i,j)*r2c(b,c,n,k) 
          call dgemm('n','t',nua,nub2,nob,-0.5d0,H2B_vooo(:,:,i,j),nua,r2c(:,:,:,k),nub2,1.0d0,Y3C,nua) 
          call dgemm('n','t',nua,nub2,nob,0.5d0,H2B_vooo(:,:,i,k),nua,r2c(:,:,:,j),nub2,1.0d0,Y3C,nua) 
          ! Diagram 3a: A(jk) chi2C_vvvo(c,b,e,j)*t2b(a,e,i,k)
          call dgemm('n','n',nua,nub2,nub,0.5d0,t2b(:,:,i,k),nua,chi2C_vvvo_3214(:,:,:,j),nub,1.0d0,Y3C,nua)
          call dgemm('n','n',nua,nub2,nub,-0.5d0,t2b(:,:,i,j),nua,chi2C_vvvo_3214(:,:,:,k),nub,1.0d0,Y3C,nua)
          ! Diagram 3b: A(jk) H2C_vvov(b,c,j,e)*r2b(a,e,i,k) 
          call dgemm('n','n',nua,nub2,nub,0.5d0,r2b(:,:,i,k),nua,H2C_vvov_4123(:,:,:,j),nub,1.0d0,Y3C,nua)
          call dgemm('n','n',nua,nub2,nub,-0.5d0,r2b(:,:,i,j),nua,H2C_vvov_4123(:,:,:,k),nub,1.0d0,Y3C,nua)
          ! Diagram 4a: -A(bc) chi2C_vooo(b,n,j,k)*t2b(a,c,i,n) -> -A(bc) t2b(a,b,i,n)*chi2C_vooo(c,n,k,j)
          call dgemm('n','n',nuanub,nub,nob,-1.0d0,t2b_1243(:,:,:,i),nuanub,chi2C_vooo_2134(:,:,k,j),nob,1.0d0,Y3C,nuanub)
          ! Diagram 4b: -A(bc) H2C_vooo(b,n,j,k)*r2b(a,c,i,n) -> -A(bc) r2b(a,b,i,n)*H2C_vooo(c,n,k,j)
          call dgemm('n','n',nuanub,nub,nob,-1.0d0,r2b_1243(:,:,:,i),nuanub,H2C_vooo_2134(:,:,k,j),nob,1.0d0,Y3C,nuanub)
          ! Diagram 5a: A(jk)A(bc) chi2B_vvvo(a,b,e,j)*t2b(e,c,i,k)
          call dgemm('n','n',nuanub,nub,nua,1.0d0,chi2B_vvvo(:,:,:,j),nuanub,t2b(:,:,i,k),nua,1.0d0,Y3C,nuanub) 
          call dgemm('n','n',nuanub,nub,nua,-1.0d0,chi2B_vvvo(:,:,:,k),nuanub,t2b(:,:,i,j),nua,1.0d0,Y3C,nuanub) 
          ! Diagram 5b: A(jk)A(bc) H2B_vvvo(a,b,e,j)*r2b(e,c,i,k) 
          call dgemm('n','n',nuanub,nub,nua,1.0d0,H2B_vvvo(:,:,:,j),nuanub,r2b(:,:,i,k),nua,1.0d0,Y3C,nuanub) 
          call dgemm('n','n',nuanub,nub,nua,-1.0d0,H2B_vvvo(:,:,:,k),nuanub,r2b(:,:,i,j),nua,1.0d0,Y3C,nuanub) 
          ! Diagram 6a: -A(jk)A(bc) chi2B_ovoo(n,b,i,j)*t2b(a,c,n,k) -> -A(jk)A(bc) t2b(a,b,n,j)*chi2B_ovoo(n,c,i,k)
          call dgemm('n','n',nuanub,nub,noa,-1.0d0,t2b(:,:,:,j),nuanub,chi2B_ovoo(:,:,i,k),noa,1.0d0,Y3C,nuanub)
          call dgemm('n','n',nuanub,nub,noa,1.0d0,t2b(:,:,:,k),nuanub,chi2B_ovoo(:,:,i,j),noa,1.0d0,Y3C,nuanub)
          ! Diagram 6b: -A(jk)A(bc) H2B_ovoo(n,b,i,j)*r2b(a,c,n,k) -> -A(jk)A(bc) r2b(a,b,n,j)*H2B_ovoo(n,c,i,k)
          call dgemm('n','n',nuanub,nub,noa,-1.0d0,r2b(:,:,:,j),nuanub,H2B_ovoo(:,:,i,k),noa,1.0d0,Y3C,nuanub)
          call dgemm('n','n',nuanub,nub,noa,1.0d0,r2b(:,:,:,k),nuanub,H2B_ovoo(:,:,i,j),noa,1.0d0,Y3C,nuanub)

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
                temp4 = Y3C(a,b,c) - Y3C(a,c,b)

                LM = (r0*temp1+temp4)*(temp2+temp3)
                LM1 = temp4*(temp2+temp3)

                D = fA_oo(i,i) + fB_oo(j,j) + fB_oo(k,k)&
                - fA_vv(a,a) - fB_vv(b,b) - fB_vv(c,c)

                deltaA = deltaA + LM/(omega+D)
                ddeltaA = ddeltaA + LM1/(omega+D)

                D = H1A_oo(i,i) + H1B_oo(j,j) + H1B_oo(k,k)&
                - H1A_vv(a,a) - H1B_vv(b,b) - H1B_vv(c,c)

                deltaB = deltaB + LM/(omega+D)
                ddeltaB = ddeltaB + LM1/(omega+D)

                D = D &
                -H2A_voov(a,i,i,a)+H2B_ovov(i,b,i,b)+H2B_ovov(i,c,i,c)&
                +H2B_vovo(a,j,a,j)-H2C_voov(b,j,j,b)-H2C_voov(c,j,j,c)&
                +H2B_vovo(a,k,a,k)-H2C_voov(b,k,k,b)-H2C_voov(c,k,k,c)&
                -H2B_oooo(i,j,i,j)-H2B_oooo(i,k,i,k)-H2C_oooo(k,j,k,j)&
                -H2B_vvvv(a,b) - H2B_vvvv(a,c) - H2C_vvvv(c,b)
                !-H2B_vvvv(a,b,a,b)-H2B_vvvv(a,c,a,c)-H2C_vvvv(c,b,c,b)

                deltaC = deltaC + LM/(omega+D)
                ddeltaC = ddeltaC + LM1/(omega+D)

                D = D &
                +D3B_O(a,i,j)+D3B_O(a,i,k)&
                +D3C_O(b,i,j)+D3C_O(b,i,k)+D3D_O(b,j,k)&
                +D3C_O(c,i,j)+D3C_O(c,i,k)+D3D_O(c,j,k)&
                -D3B_V(a,i,b)-D3B_V(a,i,c)&
                -D3C_V(a,j,b)-D3C_V(a,j,c)-D3D_V(b,j,c)&
                -D3C_V(a,k,b)-D3C_V(a,k,c)-D3D_V(b,k,c)

                deltaD = deltaD + LM/(omega+D)
                ddeltaD = ddeltaD + LM1/(omega+D)
              end do
            end do 
          end do 
        end do 
      end do 
    end do
  end subroutine creomcc23C_opt

  subroutine creomcc23D_opt(deltaA,deltaB,deltaC,deltaD,&
    ddeltaA,ddeltaB,ddeltaC,ddeltaD,&
    omega,r0,&
    t2c,r2c,l1b,l2c,&
    H2C_vooo,I2C_vvov,H2C_vvov,&
    chi2C_vvvo,chi2C_ovoo,&
    vC_oovv,H1B_ov,H2C_vovv,H2C_ooov,fB_oo,fB_vv,&
    H1B_oo,H1B_vv,&
    H2C_voov,H2C_oooo,H2C_vvvv,&
    D3D_O,D3D_V,nob,nub)

    real(p), intent(out) :: deltaA, deltaB, deltaC, deltaD
    real(p), intent(out) :: ddeltaA, ddeltaB, ddeltaC, ddeltaD
    integer, intent(in) :: nob, nub
    real(p), intent(in) :: fB_oo(1:nob,1:nob),fB_vv(1:nub,1:nub),&
    H1B_oo(1:nob,1:nob),H1B_vv(1:nub,1:nub),&
    H2C_voov(1:nub,1:nob,1:nob,1:nub),&
    H2C_oooo(1:nob,1:nob,1:nob,1:nob),&
    !H2C_vvvv(1:nub,1:nub,1:nub,1:nub),&
    H2C_vvvv(1:nub,1:nub),&
    D3D_O(1:nub,1:nob,1:nob),&
    D3D_V(1:nub,1:nob,1:nub),&
    H2C_vooo(nub,nob,nob,nob),I2C_vvov(nub,nub,nob,nub),t2c(nub,nub,nob,nob),&
    l1b(nub,nob),l2c(nub,nub,nob,nob),vC_oovv(nob,nob,nub,nub),&
    H1B_ov(nob,nub),H2C_vovv(nub,nob,nub,nub),H2C_ooov(nob,nob,nob,nub),&
    r2c(nub,nub,nob,nob),H2C_vvov(nub,nub,nob,nub),chi2C_vvvo(nub,nub,nub,nob),&
    chi2C_ovoo(nob,nub,nob,nob)
    real(p), intent(in) :: omega, r0

    integer :: i, j, k, a, b, c, nub2
    real(p) :: D, temp1, temp2, temp3, temp4, LM, LM1,&
    X3D(nub,nub,nub), L3D(nub,nub,nub), Y3D(nub,nub,nub)

    ! reordered arrays for DGEMMs
    real(p) :: I2C_vvov_1243(nub,nub,nub,nob), H2C_vovv_4312(nub,nub,nub,nob),&
    H2C_ooov_4312(nub,nob,nob,nob), H2C_vvov_2143(nub,nub,nub,nob),&
    H2C_vooo_2143(nob,nub,nob,nob)

    call reorder_stripe(4, shape(I2C_vvov), size(I2C_vvov), '1243', I2C_vvov, I2C_vvov_1243)
    call reorder_stripe(4, shape(H2C_vovv), size(H2C_vovv), '4312', H2C_vovv, H2C_vovv_4312)
    call reorder_stripe(4, shape(H2C_ooov), size(H2C_ooov), '4312', H2C_ooov, H2C_ooov_4312)
    call reorder_stripe(4, shape(H2C_vvov), size(H2C_vvov), '2143', H2C_vvov, H2C_vvov_2143)
    call reorder_stripe(4, shape(H2C_vooo), size(H2C_vooo), '2143', H2C_vooo, H2C_vooo_2143)

    deltaA = 0.0_p
    deltaB = 0.0_p
    deltaC = 0.0_p
    deltaD = 0.0_p

    ddeltaA = 0.0_p
    ddeltaB = 0.0_p
    ddeltaC = 0.0_p
    ddeltaD = 0.0_p

    nub2 = nub*nub
    do i = 1 , nob
      do j = i+1, nob
        do k = j+1, nob

          Y3D = 0.0_p
          X3D = 0.0_p
          L3D = 0.0_p
          !!!!! EOMMM(2,3)D !!!!!
          ! Diagram 1a: A(j/ik)A(c/ab) chi2C_vvvo(a,b,e,j)*t2c(e,c,i,k)
          call dgemm('n','n',nub2,nub,nub,0.5d0,chi2C_vvvo(:,:,:,j),nub2,t2c(:,:,i,k),nub,1.0d0,Y3D,nub2)
          call dgemm('n','n',nub2,nub,nub,-0.5d0,chi2C_vvvo(:,:,:,k),nub2,t2c(:,:,i,j),nub,1.0d0,Y3D,nub2)
          call dgemm('n','n',nub2,nub,nub,-0.5d0,chi2C_vvvo(:,:,:,i),nub2,t2c(:,:,j,k),nub,1.0d0,Y3D,nub2)
          ! Diagram 1b: A(j/ik)A(c/ab) H2C_vvov(b,a,j,e)*r2c(e,c,i,k)
          call dgemm('n','n',nub2,nub,nub,0.5d0,H2C_vvov_2143(:,:,:,j),nub2,r2c(:,:,i,k),nub,1.0d0,Y3D,nub2)
          call dgemm('n','n',nub2,nub,nub,-0.5d0,H2C_vvov_2143(:,:,:,k),nub2,r2c(:,:,i,j),nub,1.0d0,Y3D,nub2)
          call dgemm('n','n',nub2,nub,nub,-0.5d0,H2C_vvov_2143(:,:,:,i),nub2,r2c(:,:,j,k),nub,1.0d0,Y3D,nub2)
          ! Diagram 2a: -A(k/ij)A(b/ac) chi2C_ovoo(m,b,i,j)*t2c(a,c,m,k)
          call dgemm('n','n',nub2,nub,nob,-0.5d0,t2c(:,:,:,j),nub2,chi2C_ovoo(:,:,i,k),nob,1.0d0,Y3D,nub2)
          call dgemm('n','n',nub2,nub,nob,0.5d0,t2c(:,:,:,i),nub2,chi2C_ovoo(:,:,j,k),nob,1.0d0,Y3D,nub2)
          call dgemm('n','n',nub2,nub,nob,0.5d0,t2c(:,:,:,k),nub2,chi2C_ovoo(:,:,i,j),nob,1.0d0,Y3D,nub2)
          ! Diagram 2b: -A(k/ij)A(b/ac) H2C_vooo(b,m,j,i)*r2c(a,c,m,k)
          call dgemm('n','n',nub2,nub,nob,-0.5d0,r2c(:,:,:,j),nub2,H2C_vooo_2143(:,:,i,k),nob,1.0d0,Y3D,nub2)
          call dgemm('n','n',nub2,nub,nob,0.5d0,r2c(:,:,:,i),nub2,H2C_vooo_2143(:,:,j,k),nob,1.0d0,Y3D,nub2)
          call dgemm('n','n',nub2,nub,nob,0.5d0,r2c(:,:,:,k),nub2,H2C_vooo_2143(:,:,i,j),nob,1.0d0,Y3D,nub2)
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

                temp4 = Y3D(a,b,c) + Y3D(b,c,a) + Y3D(c,a,b)&
                - Y3D(a,c,b) - Y3D(b,a,c) - Y3D(c,b,a)

                LM = (r0*temp1+temp4)*(temp2+temp3)
                LM1 = temp4*(temp2+temp3)

                D = fB_oo(i,i) + fB_oo(j,j) + fB_oo(k,k)&
                - fB_vv(a,a) - fB_vv(b,b) - fB_vv(c,c)

                deltaA = deltaA + LM/(omega+D)
                ddeltaA = ddeltaA + LM1/(omega+D)

                D = H1B_oo(i,i) + H1B_oo(j,j) + H1B_oo(k,k)&
                - H1B_vv(a,a) - H1B_vv(b,b) - H1B_vv(c,c)

                deltaB = deltaB + LM/(omega+D)
                ddeltaB = ddeltaB + LM1/(omega+D)

                D = D &
                -H2C_voov(a,i,i,a) - H2C_voov(b,i,i,b) - H2C_voov(c,i,i,c)&
                -H2C_voov(a,j,j,a) - H2C_voov(b,j,j,b) - H2C_voov(c,j,j,c)&
                -H2C_voov(a,k,k,a) - H2C_voov(b,k,k,b) - H2C_voov(c,k,k,c)&
                -H2C_oooo(j,i,j,i) - H2C_oooo(k,i,k,i) - H2C_oooo(k,j,k,j)&
                -H2C_vvvv(b,a) - H2C_vvvv(c,a) - H2C_vvvv(c,b)
                !-H2C_vvvv(b,a,b,a) - H2C_vvvv(c,a,c,a) - H2C_vvvv(c,b,c,b)

                deltaC = deltaC + LM/(omega+D)
                ddeltaC = ddeltaC + LM1/(omega+D)

                D = D &
                +D3D_O(a,i,j)+D3D_O(a,i,k)+D3D_O(a,j,k)&
                +D3D_O(b,i,j)+D3D_O(b,i,k)+D3D_O(b,j,k)&
                +D3D_O(c,i,j)+D3D_O(c,i,k)+D3D_O(c,j,k)&
                -D3D_V(a,i,b)-D3D_V(a,i,c)-D3D_V(b,i,c)&
                -D3D_V(a,j,b)-D3D_V(a,j,c)-D3D_V(b,j,c)&
                -D3D_V(a,k,b)-D3D_V(a,k,c)-D3D_V(b,k,c)

                deltaD = deltaD + LM/(omega+D)
                ddeltaD = ddeltaD + LM1/(omega+D)
              end do
            end do 
          end do
        end do 
      end do 
    end do

  end subroutine creomcc23D_opt

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! OLD CR-CC(2,3) ROUTINES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine crcc23A(deltaA,deltaB,deltaC,deltaD,&
    MM23A,L3A,omega,&
    fA_oo,fA_vv,H1A_oo,H1A_vv,&
    H2A_voov,H2A_oooo,H2A_vvvv,&
    D3A_O,D3A_V,noa,nua)

    real(p), intent(out) :: deltaA, deltaB, deltaC, deltaD
    integer, intent(in) :: noa, nua
    real(p), intent(in) :: MM23A(1:nua,1:nua,1:nua,1:noa,1:noa,1:noa),&
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
    real(p) :: D, temp

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

    real(p), intent(out) :: deltaA, deltaB, deltaC, deltaD
    integer, intent(in) :: noa, nua, nob, nub
    real(p), intent(in) :: MM23B(1:nua,1:nua,1:nub,1:noa,1:noa,1:nob),&
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
    real(p) :: D, temp

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

    real(p), intent(out) :: deltaA, deltaB, deltaC, deltaD
    integer, intent(in) :: noa, nua, nob, nub
    real(p), intent(in) :: MM23C(1:nua,1:nub,1:nub,1:noa,1:nob,1:nob),&
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
    real(p) :: D, temp

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

    real(p), intent(out) :: deltaA, deltaB, deltaC, deltaD
    integer, intent(in) :: nob, nub
    real(p), intent(in) :: MM23D(1:nub,1:nub,1:nub,1:nob,1:nob,1:nob),&
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
    real(p) :: D, temp

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

end module crcc_loops
