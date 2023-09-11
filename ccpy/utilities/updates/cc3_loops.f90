module cc3_loops
 
    use omp_lib
 
    implicit none
    
    contains
               subroutine update_t(t1a,t1b,t2a,t2b,t2c,&
                                   resid_a,resid_b,resid_aa,resid_ab,resid_bb,&
                                   X1A,X1B,X2A,X2B,X2C,&
                                   fA_oo,fA_vv,fB_oo,fB_vv,&
                                   h1a_ov,h1b_ov,&
                                   vA_oovv,vB_oovv,vC_oovv,&
                                   h2a_ooov,h2a_vovv,&
                                   h2b_ooov,h2b_oovo,h2b_vovv,h2b_ovvv,&
                                   h2c_ooov,h2c_vovv,&
                                   h2a_vooo,h2a_vvov,&
                                   h2b_vooo,h2b_ovoo,h2b_vvov,h2b_vvvo,&
                                   h2c_vooo,h2c_vvov,&
                                   shift,&
                                   noa,nua,nob,nub)

                      integer, intent(in) :: noa, nua, nob, nub
                      real(kind=8), intent(in) :: fA_oo(noa,noa), fA_vv(nua,nua),&
                                                  fB_oo(nob,nob), fB_vv(nub,nub),&
                                                  h1a_ov(noa,nua),h1b_ov(nob,nub),&
                                                  vA_oovv(noa,noa,nua,nua),&
                                                  vB_oovv(noa,nob,nua,nub),&
                                                  vC_oovv(nob,nob,nub,nub),&
                                                  h2a_ooov(noa,noa,noa,nua),&
                                                  h2a_vovv(nua,noa,nua,nua),&
                                                  h2b_ooov(noa,nob,noa,nub),&
                                                  h2b_oovo(noa,nob,nua,nob),&
                                                  h2b_vovv(nua,nob,nua,nub),&
                                                  h2b_ovvv(noa,nub,nua,nub),&
                                                  h2c_ooov(nob,nob,nob,nub),&
                                                  h2c_vovv(nub,nob,nub,nub),&
                                                  h2a_vooo(nua,noa,noa,noa),&
                                                  h2a_vvov(nua,nua,noa,nua),&
                                                  h2b_vooo(nua,nob,noa,nob),&
                                                  h2b_ovoo(noa,nub,noa,nob),&
                                                  h2b_vvov(nua,nub,noa,nub),&
                                                  h2b_vvvo(nua,nub,nua,nob),&
                                                  h2c_vooo(nub,nob,nob,nob),&
                                                  h2c_vvov(nub,nub,nob,nub)
                      real(kind=8), intent(in) :: X1A(1:nua,1:noa)
                      real(kind=8), intent(in) :: X1B(1:nub,1:nob)
                      real(kind=8), intent(in) :: X2A(1:nua,1:nua,1:noa,1:noa)
                      real(kind=8), intent(in) :: X2B(1:nua,1:nub,1:noa,1:nob)
                      real(kind=8), intent(in) :: X2C(1:nub,1:nub,1:nob,1:nob)
                      real(kind=8), intent(in) :: shift
                      
                      real(kind=8), intent(inout) :: t1a(1:nua,1:noa)
                      !f2py intent(in,out) :: t1a(0:nua-1,0:noa-1)
                      real(kind=8), intent(inout) :: t1b(1:nub,1:nob)
                      !f2py intent(in,out) :: t1b(0:nub-1,0:nob-1)
                      real(kind=8), intent(inout) :: t2a(1:nua,1:nua,1:noa,1:noa)
                      !f2py intent(in,out) :: t2a(0:nua-1,0:nua-1,0:noa-1,0:noa-1)
                      real(kind=8), intent(inout) :: t2b(1:nua,1:nub,1:noa,1:nob)
                      !f2py intent(in,out) :: t2b(0:nua-1,0:nub-1,0:noa-1,0:nob-1)
                      real(kind=8), intent(inout) :: t2c(1:nub,1:nub,1:nob,1:nob)
                      !f2py intent(in,out) :: t2c(0:nub-1,0:nub-1,0:nob-1,0:nob-1)
                      real(kind=8), intent(out) :: resid_a(1:nua,1:noa)
                      real(kind=8), intent(out) :: resid_b(1:nub,1:nob)
                      real(kind=8), intent(out) :: resid_aa(1:nua,1:nua,1:noa,1:noa)
                      real(kind=8), intent(out) :: resid_ab(1:nua,1:nub,1:noa,1:nob)
                      real(kind=8), intent(out) :: resid_bb(1:nub,1:nub,1:nob,1:nob)

                      integer :: i, j, k, a, b, c
                      real(kind=8) :: denom, val
                      real(kind=8) :: t3a_o, t3a_v, t3b_o, t3b_v, t3c_o, t3c_v, t3d_o, t3d_v
                      real(kind=8) :: t3a, t3b, t3c, t3d
                      real(kind=8) :: t3_denom
                      
                      ! allocatable array to hold t3(abc) for a given (i,j,k) block
                      real(kind=8), allocatable :: temp(:,:,:)
                      ! reordered arrays for the DGEMM operations
                      real(kind=8) :: H2A_vvov_1243(nua,nua,nua,noa)
                      real(kind=8) :: H2B_vvov_1243(nua,nub,nub,noa), t2b_1243(nua,nub,nob,noa)
                      real(kind=8) :: H2C_vvov_4213(nub,nub,nub,noa), H2C_vooo_2134(nob,nub,nob,nob)
                      real(kind=8) :: H2C_vvov_1243(nub,nub,nub,nob)
                      
                      ! Allocate residual containers (these will hold contractions with T3)
                      resid_a = 0.0d0
                      resid_b = 0.0d0
                      resid_aa = 0.0d0
                      resid_ab = 0.0d0
                      resid_bb = 0.0d0
                      
                      ! Call reordering routines for arrays entering DGEMM
                      call reorder1243(H2A_vvov,H2A_vvov_1243)
                      call reorder1243(H2B_vvov,H2B_vvov_1243)
                      call reorder1243(t2b,t2b_1243)
                      call reorder4213(H2C_vvov,H2C_vvov_4213)
                      call reorder2134(H2C_vooo,H2C_vooo_2134)
                      call reorder1243(H2C_vvov,H2C_vvov_1243)

                      ! contribution from t3a
                      allocate(temp(nua,nua,nua))
                      do i = 1,noa
                        do j = i+1,noa
                           do k = j+1,noa
                              temp = 0.0d0
                              ! Diagram 1: -A(k/ij)A(a/bc) I2A_vooo(a,m,i,j)*t2a(b,c,m,k)
                              call dgemm('n','t',nua,nua**2,noa,-0.5d0,H2A_vooo(:,:,i,j),nua,t2a(:,:,:,k),nua**2,1.0d0,temp,nua)
                              call dgemm('n','t',nua,nua**2,noa,0.5d0,H2A_vooo(:,:,k,j),nua,t2a(:,:,:,i),nua**2,1.0d0,temp,nua)
                              call dgemm('n','t',nua,nua**2,noa,0.5d0,H2A_vooo(:,:,i,k),nua,t2a(:,:,:,j),nua**2,1.0d0,temp,nua)
                              ! Diagram 2: A(i/jk)A(c/ab) I2A_vvov(a,b,i,e)*t2a(e,c,j,k)
                              call dgemm('n','n',nua**2,nua,nua,0.5d0,H2A_vvov_1243(:,:,:,i),nua**2,t2a(:,:,j,k),nua,1.0d0,temp,nua**2)
                              call dgemm('n','n',nua**2,nua,nua,-0.5d0,H2A_vvov_1243(:,:,:,j),nua**2,t2a(:,:,i,k),nua,1.0d0,temp,nua**2)
                              call dgemm('n','n',nua**2,nua,nua,-0.5d0,H2A_vvov_1243(:,:,:,k),nua**2,t2a(:,:,j,i),nua,1.0d0,temp,nua**2)
                              !$omp parallel shared(temp,vA_oovv,h1a_ov,h2a_ooov,h2a_vovv,i,j,k),&
                              !$omp private(t3a,t3_denom,a,b,c)
                              !$omp do schedule(static) reduction(+:resid_a,resid_aa)
                              do a = 1,nua
                                 do b = a+1,nua
                                    do c = b+1,nua
                                       t3_denom = fA_oo(i,i)+fA_oo(j,j)+fA_oo(k,k)-fA_vv(a,a)-fA_vv(b,b)-fA_vv(c,c)
                                       t3a = temp(a,b,c) + temp(b,c,a) + temp(c,a,b) - temp(a,c,b) - temp(b,a,c) - temp(c,b,a)
                                       t3a = t3a / t3_denom
                                       ! A(a/bc)A(i/jk) vA(jkbc)*t3a(abcijk)
                                       resid_a(a,i) = resid_a(a,i) + vA_oovv(j,k,b,c) * t3a ! (1)
                                       resid_a(b,i) = resid_a(b,i) - vA_oovv(j,k,a,c) * t3a ! (ae)
                                       resid_a(c,i) = resid_a(c,i) - vA_oovv(j,k,b,a) * t3a ! (af)
                                       resid_a(a,j) = resid_a(a,j) - vA_oovv(i,k,b,c) * t3a ! (im)
                                       resid_a(b,j) = resid_a(b,j) + vA_oovv(i,k,a,c) * t3a ! (ae)(im)
                                       resid_a(c,j) = resid_a(c,j) + vA_oovv(i,k,b,a) * t3a ! (af)(im)
                                       resid_a(a,k) = resid_a(a,k) - vA_oovv(j,i,b,c) * t3a ! (in)
                                       resid_a(b,k) = resid_a(b,k) + vA_oovv(j,i,a,c) * t3a ! (ae)(in)
                                       resid_a(c,k) = resid_a(c,k) + vA_oovv(j,i,b,a) * t3a ! (af)(in)
                                       ! A(ij)A(ab) [A(m/ij)A(e/ab) h1a(me) * t3a(abeijm)]
                                       resid_aa(a,b,i,j) = resid_aa(a,b,i,j) + H1A_ov(k,c) * t3a ! (1)
                                       resid_aa(a,b,k,j) = resid_aa(a,b,k,j) - H1A_ov(i,c) * t3a ! (im)
                                       resid_aa(a,b,i,k) = resid_aa(a,b,i,k) - H1A_ov(j,c) * t3a ! (jm)
                                       resid_aa(c,b,i,j) = resid_aa(c,b,i,j) - H1A_ov(k,a) * t3a ! (ae)
                                       resid_aa(c,b,k,j) = resid_aa(c,b,k,j) + H1A_ov(i,a) * t3a ! (im)(ae)
                                       resid_aa(c,b,i,k) = resid_aa(c,b,i,k) + H1A_ov(j,a) * t3a ! (jm)(ae)
                                       resid_aa(a,c,i,j) = resid_aa(a,c,i,j) - H1A_ov(k,b) * t3a ! (be)
                                       resid_aa(a,c,k,j) = resid_aa(a,c,k,j) + H1A_ov(i,b) * t3a ! (im)(be)
                                       resid_aa(a,c,i,k) = resid_aa(a,c,i,k) + H1A_ov(j,b) * t3a ! (jm)(be)
                                       ! A(ij)A(ab) [A(j/mn)A(f/ab) -h2a(mnif) * t3a(abfmjn)]
                                       resid_aa(a,b,:,j) = resid_aa(a,b,:,j) - H2A_ooov(i,k,:,c) * t3a ! (1)
                                       resid_aa(a,b,:,i) = resid_aa(a,b,:,i) + H2A_ooov(j,k,:,c) * t3a ! (jm)
                                       resid_aa(a,b,:,k) = resid_aa(a,b,:,k) + H2A_ooov(i,j,:,c) * t3a ! (jn)
                                       resid_aa(c,b,:,j) = resid_aa(c,b,:,j) + H2A_ooov(i,k,:,a) * t3a ! (af)
                                       resid_aa(c,b,:,i) = resid_aa(c,b,:,i) - H2A_ooov(j,k,:,a) * t3a ! (jm)(af)
                                       resid_aa(c,b,:,k) = resid_aa(c,b,:,k) - H2A_ooov(i,j,:,a) * t3a ! (jn)(af)
                                       resid_aa(a,c,:,j) = resid_aa(a,c,:,j) + H2A_ooov(i,k,:,b) * t3a ! (bf)
                                       resid_aa(a,c,:,i) = resid_aa(a,c,:,i) - H2A_ooov(j,k,:,b) * t3a ! (jm)(bf)
                                       resid_aa(a,c,:,k) = resid_aa(a,c,:,k) - H2A_ooov(i,j,:,b) * t3a ! (jn)(bf)
                                       ! A(ij)A(ab) [A(n/ij)A(b/ef) h2a(anef) * t3a(ebfijn)]
                                       resid_aa(:,b,i,j) = resid_aa(:,b,i,j) + H2A_vovv(:,k,a,c) * t3a ! (1)
                                       resid_aa(:,b,k,j) = resid_aa(:,b,k,j) - H2A_vovv(:,i,a,c) * t3a ! (in)
                                       resid_aa(:,b,i,k) = resid_aa(:,b,i,k) - H2A_vovv(:,j,a,c) * t3a ! (jn)
                                       resid_aa(:,a,i,j) = resid_aa(:,a,i,j) - H2A_vovv(:,k,b,c) * t3a ! (be)
                                       resid_aa(:,a,k,j) = resid_aa(:,a,k,j) + H2A_vovv(:,i,b,c) * t3a ! (in)(be)
                                       resid_aa(:,a,i,k) = resid_aa(:,a,i,k) + H2A_vovv(:,j,b,c) * t3a ! (jn)(be)
                                       resid_aa(:,c,i,j) = resid_aa(:,c,i,j) - H2A_vovv(:,k,a,b) * t3a ! (bf)
                                       resid_aa(:,c,k,j) = resid_aa(:,c,k,j) + H2A_vovv(:,i,a,b) * t3a ! (in)(bf)
                                       resid_aa(:,c,i,k) = resid_aa(:,c,i,k) + H2A_vovv(:,j,a,b) * t3a ! (jn)(bf)
                                    end do
                                 end do
                              end do
                              !$omp end do
                              !$omp end parallel
                           end do
                        end do
                      end do
                      deallocate(temp)
                      ! contribution from t3b
                      allocate(temp(nua,nua,nub))
                      do i = 1,noa
                         do j = i+1,noa
                           do k = 1,nob
                              temp = 0.0d0
                              ! Diagram 1: A(ab) H2B(bcek)*t2a(aeij)
                              call dgemm('n','t',nua,nua*nub,nua,1.0d0,t2a(:,:,i,j),nua,H2B_vvvo(:,:,:,k),nua*nub,1.0d0,temp,nua)
                              ! Diagram 2: -A(ij) I2B(mcjk)*t2a(abim)
                              call dgemm('n','n',nua**2,nub,noa,0.5d0,t2a(:,:,:,i),nua**2,H2B_ovoo(:,:,j,k),noa,1.0d0,temp,nua**2)
                              call dgemm('n','n',nua**2,nub,noa,-0.5d0,t2a(:,:,:,j),nua**2,H2B_ovoo(:,:,i,k),noa,1.0d0,temp,nua**2)
                              ! Diagram 3: A(ab)A(ij) H2B(acie)*t2b(bejk) -> A(ab)A(ij) t2b(aeik)*H2B(bcje)
                              call dgemm('n','t',nua,nua*nub,nub,1.0d0,t2b(:,:,i,k),nua,H2B_vvov_1243(:,:,:,j),nua*nub,1.0d0,temp,nua)
                              call dgemm('n','t',nua,nua*nub,nub,-1.0d0,t2b(:,:,j,k),nua,H2B_vvov_1243(:,:,:,i),nua*nub,1.0d0,temp,nua)
                              ! Diagram 4: -A(ab)A(ij) I2B(amik)*t2b(bcjm)
                              call dgemm('n','t',nua,nua*nub,nob,-1.0d0,H2B_vooo(:,:,i,k),nua,t2b_1243(:,:,:,j),nua*nub,1.0d0,temp,nua)
                              call dgemm('n','t',nua,nua*nub,nob,1.0d0,H2B_vooo(:,:,j,k),nua,t2b_1243(:,:,:,i),nua*nub,1.0d0,temp,nua)
                              ! Diagram 5: A(ij) H2A(abie)*t2b(ecjk)
                              call dgemm('n','n',nua**2,nub,nua,0.5d0,H2A_vvov_1243(:,:,:,i),nua**2,t2b(:,:,j,k),nua,1.0d0,temp,nua**2)
                              call dgemm('n','n',nua**2,nub,nua,-0.5d0,H2A_vvov_1243(:,:,:,j),nua**2,t2b(:,:,i,k),nua,1.0d0,temp,nua**2)
                              ! Diagram 6: -A(ab) I2A(amij)*t2b(bcmk)
                              call dgemm('n','t',nua,nua*nub,noa,-1.0d0,H2A_vooo(:,:,i,j),nua,t2b(:,:,:,k),nua*nub,1.0d0,temp,nua)
                              !$omp parallel shared(temp,vA_oovv,vB_oovv,vC_oovv,&
                              !$omp h1a_ov,h1b_ov,&
                              !$omp h2a_ooov,h2a_vovv,&
                              !$omp h2b_ooov,h2b_oovo,h2b_vvov,h2b_vvvo,&
                              !$omp i,j,k),&
                              !$omp private(t3b,t3_denom,a,b,c)
                              !$omp do schedule(static) reduction(+:resid_a,resid_b,resid_aa,resid_ab)
                              do a = 1,nua
                                 do b = a+1,nua
                                    do c = 1,nub
                                       t3_denom = fA_oo(i,i)+fA_oo(j,j)+fB_oo(k,k)-fA_vv(a,a)-fA_vv(b,b)-fB_vv(c,c)
                                       t3b = temp(a,b,c) - temp(b,a,c)
                                       t3b = t3b / t3_denom
                                       ! A(ij)A(ab) vB(jkbc) * t3b(abcijk)
                                       resid_a(a,i) = resid_a(a,i) + vB_oovv(j,k,b,c) * t3b ! (1)
                                       resid_a(b,i) = resid_a(b,i) - vB_oovv(j,k,a,c) * t3b ! (ae)
                                       resid_a(a,j) = resid_a(a,j) - vB_oovv(i,k,b,c) * t3b ! (im)
                                       resid_a(b,j) = resid_a(b,j) + vB_oovv(i,k,a,c) * t3b ! (ae)(im)
                                       ! vA(ijab) * t3b(abcijk)
                                       resid_b(c,k) = resid_b(c,k) + vA_oovv(i,j,a,b) * t3b ! (1)
                                       ! A(ij)A(ab) [h1b(me) * t3b(abeijm)]
                                       resid_aa(a,b,i,j) = resid_aa(a,b,i,j) + H1B_ov(k,c) * t3b ! (1)
                                       ! A(ij)A(ab) [A(jm) -h2b(mnif) * t3b(abfmjn)]
                                       resid_aa(a,b,:,j) = resid_aa(a,b,:,j) - H2B_ooov(i,k,:,c) * t3b ! (1)
                                       resid_aa(a,b,:,i) = resid_aa(a,b,:,i) + H2B_ooov(j,k,:,c) * t3b ! (jm)
                                       ! A(ij)A(ab) [A(be) h2b(anef) * t3b(ebfijn)] (!!! expensive; ~3s)
                                       resid_aa(:,b,i,j) = resid_aa(:,b,i,j) + H2B_vovv(:,k,a,c) * t3b ! (1)
                                       resid_aa(:,a,i,j) = resid_aa(:,a,i,j) - H2B_vovv(:,k,b,c) * t3b ! (be)
                                       ! A(af) -h2a(mnif) * t3b(afbmnj)
                                       resid_ab(a,c,:,k) = resid_ab(a,c,:,k) - H2A_ooov(i,j,:,b) * t3b ! (1)
                                       resid_ab(b,c,:,k) = resid_ab(b,c,:,k) + H2A_ooov(i,j,:,a) * t3b ! (af)
                                       ! A(af)A(in) -h2b(nmfj) * t3b(afbinm)
                                       resid_ab(a,c,i,:) = resid_ab(a,c,i,:) - H2B_oovo(j,k,b,:) * t3b ! (1)
                                       resid_ab(b,c,i,:) = resid_ab(b,c,i,:) + H2B_oovo(j,k,a,:) * t3b ! (af)
                                       resid_ab(a,c,j,:) = resid_ab(a,c,j,:) + H2B_oovo(i,k,b,:) * t3b ! (in)
                                       resid_ab(b,c,j,:) = resid_ab(b,c,j,:) - H2B_oovo(i,k,a,:) * t3b ! (af)(in)
                                       ! A(in) h2a(anef) * t3b(efbinj) (!!! expensive; effect is not much, ~1-2s)
                                       resid_ab(:,c,i,k) = resid_ab(:,c,i,k) + H2A_vovv(:,j,a,b) * t3b ! (1)
                                       resid_ab(:,c,j,k) = resid_ab(:,c,j,k) - H2A_vovv(:,i,a,b) * t3b ! (in)
                                       ! A(af)A(in) h2b(nbfe) * t3b(afeinj) (!!! expensive; LARGE effect ~8-10s)
                                       resid_ab(a,:,i,k) = resid_ab(a,:,i,k) + H2B_ovvv(j,:,b,c) * t3b ! (1)
                                       resid_ab(b,:,i,k) = resid_ab(b,:,i,k) - H2B_ovvv(j,:,a,c) * t3b ! (af)
                                       resid_ab(a,:,j,k) = resid_ab(a,:,j,k) - H2B_ovvv(i,:,b,c) * t3b ! (in)
                                       resid_ab(b,:,j,k) = resid_ab(b,:,j,k) + H2B_ovvv(i,:,a,c) * t3b ! (af)(in)
                                       ! A(ae)A(im) h1a(me) * t3b(aebimj)
                                       resid_ab(a,c,i,k) = resid_ab(a,c,i,k) + H1A_ov(j,b) * t3b ! (1)
                                       resid_ab(a,c,j,k) = resid_ab(a,c,j,k) - H1A_ov(i,b) * t3b ! (im)
                                       resid_ab(b,c,i,k) = resid_ab(b,c,i,k) - H1A_ov(j,a) * t3b ! (ae)
                                       resid_ab(b,c,j,k) = resid_ab(b,c,j,k) + H1A_ov(i,a) * t3b ! (im)(ae)
                                    end do
                                 end do
                              end do
                              !$omp end do
                              !$omp end parallel
                           end do
                         end do
                      end do
                      deallocate(temp)
                      ! contribution from t3c
                      allocate(temp(nua,nub,nub))
                      do i = 1,noa
                         do j = 1,nob
                           do k = j+1,nob
                              temp = 0.0d0
                              ! Diagram 1: A(bc) H2B_vvov(a,b,i,e)*t2c(e,c,j,k)
                              call dgemm('n','n',nua*nub,nub,nub,1.0d0,H2B_vvov_1243(:,:,:,i),nua*nub,t2c(:,:,j,k),nub,1.0d0,temp,nua*nub)
                              ! Diagram 2: -A(jk) I2B_vooo(a,m,i,j)*t2c(b,c,m,k)
                              call dgemm('n','t',nua,nub**2,nob,-0.5d0,H2B_vooo(:,:,i,j),nua,t2c(:,:,:,k),nub**2,1.0d0,temp,nua)
                              call dgemm('n','t',nua,nub**2,nob,0.5d0,H2B_vooo(:,:,i,k),nua,t2c(:,:,:,j),nub**2,1.0d0,temp,nua)
                              ! Diagram 3: A(jk) H2C_vvov(c,b,k,e)*t2b(a,e,i,j)
                              call dgemm('n','n',nua,nub**2,nub,0.5d0,t2b(:,:,i,j),nua,H2C_vvov_4213(:,:,:,k),nub,1.0d0,temp,nua)
                              call dgemm('n','n',nua,nub**2,nub,-0.5d0,t2b(:,:,i,k),nua,H2C_vvov_4213(:,:,:,j),nub,1.0d0,temp,nua)
                              ! Diagram 4: -A(bc) I2C_vooo(c,m,k,j)*t2b(a,b,i,m)
                              call dgemm('n','n',nua*nub,nub,nob,-1.0d0,t2b_1243(:,:,:,i),nua*nub,H2C_vooo_2134(:,:,k,j),nob,1.0d0,temp,nua*nub)
                              ! Diagram 5: A(jk)A(bc) H2B_vvvo(a,b,e,j)*t2b(e,c,i,k)
                              call dgemm('n','n',nua*nub,nub,nua,1.0d0,H2B_vvvo(:,:,:,j),nua*nub,t2b(:,:,i,k),nua,1.0d0,temp,nua*nub)
                              call dgemm('n','n',nua*nub,nub,nua,-1.0d0,H2B_vvvo(:,:,:,k),nua*nub,t2b(:,:,i,j),nua,1.0d0,temp,nua*nub)
                              ! Diagram 6: -A(jk)A(bc) I2B_ovoo(m,b,i,j)*t2b(a,c,m,k) -> -A(jk)A(bc) I2B_ovoo(m,c,i,k)*t2b(a,b,m,j)
                              call dgemm('n','n',nua*nub,nub,noa,-1.0d0,t2b(:,:,:,j),nua*nub,H2B_ovoo(:,:,i,k),noa,1.0d0,temp,nua*nub)
                              call dgemm('n','n',nua*nub,nub,noa,1.0d0,t2b(:,:,:,k),nua*nub,H2B_ovoo(:,:,i,j),noa,1.0d0,temp,nua*nub)
                              !$omp parallel shared(temp,vA_oovv,vB_oovv,vC_oovv,&
                              !$omp h1a_ov,h1b_ov,&
                              !$omp h2c_ooov,h2c_vovv,&
                              !$omp h2b_ooov,h2b_oovo,h2b_vvov,h2b_vvvo,&
                              !$omp i,j,k),&
                              !$omp private(t3c,t3_denom,a,b,c)
                              !$omp do schedule(static) reduction(+:resid_a,resid_b,resid_ab,resid_bb)
                              do a = 1,nua
                                 do b = 1,nub
                                    do c = b+1,nub
                                       t3_denom = fA_oo(i,i)+fB_oo(j,j)+fB_oo(k,k)-fA_vv(a,a)-fB_vv(b,b)-fB_vv(c,c)
                                       t3c = temp(a,b,c) - temp(a,c,b)
                                       t3c = t3c / t3_denom
                                       ! vC(jkbc) * t3c(abcijk)
                                       resid_a(a,i) = resid_a(a,i) + vC_oovv(j,k,b,c) * t3c ! (1)
                                       ! A(bc)A(jk) vB(ijab) * t3c(abcijk)
                                       resid_b(c,k) = resid_b(c,k) + vB_oovv(i,j,a,b) * t3c ! (1)
                                       resid_b(b,k) = resid_b(b,k) - vB_oovv(i,j,a,c) * t3c ! (bc)
                                       resid_b(c,j) = resid_b(c,j) - vB_oovv(i,k,a,b) * t3c ! (jk)
                                       resid_b(b,j) = resid_b(b,j) + vB_oovv(i,k,a,c) * t3c ! (bc)(jk)
                                       ! A(bf) -h2c(mnjf) * t3c(afbinm)
                                       resid_ab(a,c,i,:) = resid_ab(a,c,i,:) - H2C_ooov(k,j,:,b) * t3c ! (1)
                                       resid_ab(a,b,i,:) = resid_ab(a,b,i,:) + H2C_ooov(k,j,:,c) * t3c ! (bf)
                                       ! A(bf)A(jn) -h2b(mnif) * t3c(afbmnj)
                                       resid_ab(a,c,:,k) = resid_ab(a,c,:,k) - H2B_ooov(i,j,:,b) * t3c ! (1)
                                       resid_ab(a,b,:,k) = resid_ab(a,b,:,k) + H2B_ooov(i,j,:,c) * t3c ! (bf)
                                       resid_ab(a,c,:,j) = resid_ab(a,c,:,j) + H2B_ooov(i,k,:,b) * t3c ! (jn)
                                       resid_ab(a,b,:,j) = resid_ab(a,b,:,j) - H2B_ooov(i,k,:,c) * t3c ! (bf)(jn)
                                       ! A(jn) h2c(bnef) * t3c(afeinj) (!!! expensive)
                                       resid_ab(a,:,i,k) = resid_ab(a,:,i,k) + H2C_vovv(:,j,c,b) * t3c ! (1)
                                       resid_ab(a,:,i,j) = resid_ab(a,:,i,j) - H2C_vovv(:,k,c,b) * t3c ! (jn)
                                       ! A(bf)A(jn) h2b(anef) * t3c(efbinj) (!!! expensive; LARGE effect)
                                       resid_ab(:,c,i,k) = resid_ab(:,c,i,k) + H2B_vovv(:,j,a,b) * t3c ! (1)
                                       resid_ab(:,b,i,k) = resid_ab(:,b,i,k) - H2B_vovv(:,j,a,c) * t3c ! (bf)
                                       resid_ab(:,c,i,j) = resid_ab(:,c,i,j) - H2B_vovv(:,k,a,b) * t3c ! (jn)
                                       resid_ab(:,b,i,j) = resid_ab(:,b,i,j) + H2B_vovv(:,k,a,c) * t3c ! (bf)(jn)
                                       ! [A(be)A(mj) h1b(me) * t3c(aebimj)]
                                       resid_ab(a,c,i,k) = resid_ab(a,c,i,k) + H1B_ov(j,b) * t3c ! (1)
                                       resid_ab(a,c,i,j) = resid_ab(a,c,i,j) - H1B_ov(k,b) * t3c ! (jm)
                                       resid_ab(a,b,i,k) = resid_ab(a,b,i,k) - H1B_ov(j,c) * t3c ! (be)
                                       resid_ab(a,b,i,j) = resid_ab(a,b,i,j) + H1B_ov(k,c) * t3c ! (jm)(be)
                                       ! A(ij)A(ab) [h1a(me) * t3c(eabmij)]
                                       resid_bb(b,c,j,k) = resid_bb(b,c,j,k) + H1A_ov(i,a) * t3c ! (1)
                                       ! A(ij)A(ab) [A(be) h2b(nafe) * t3c(febnij)] (!!! expensive)
                                       resid_bb(:,c,j,k) = resid_bb(:,c,j,k) + H2B_ovvv(i,:,a,b) * t3c ! (1)
                                       resid_bb(:,b,j,k) = resid_bb(:,b,j,k) - H2B_ovvv(i,:,a,c) * t3c ! (be)
                                       ! A(ij)A(ab) [A(jm) -h2b(nmfi) * t3c(fabnmj)]
                                       resid_bb(b,c,:,k) = resid_bb(b,c,:,k) - H2B_oovo(i,j,a,:) * t3c ! (1)
                                       resid_bb(b,c,:,j) = resid_bb(b,c,:,j) + H2B_oovo(i,k,a,:) * t3c ! (jm)
                                    end do
                                 end do
                              end do
                              !$omp end do
                              !$omp end parallel
                           end do
                         end do
                      end do
                      deallocate(temp)
                      ! contribution from t3d
                      allocate(temp(nub,nub,nub))
                      do i = 1,nob
                        do j = i+1,nob
                           do k = j+1,nob
                              temp = 0.0d0
                              ! Diagram 1: -A(k/ij)A(a/bc) H2C_vooo(a,m,i,j)*t2c(b,c,m,k)
                              call dgemm('n','t',nub,nub**2,nob,-0.5d0,H2C_vooo(:,:,i,j),nub,t2c(:,:,:,k),nub**2,1.0d0,temp,nub)
                              call dgemm('n','t',nub,nub**2,nob,0.5d0,H2C_vooo(:,:,k,j),nub,t2c(:,:,:,i),nub**2,1.0d0,temp,nub)
                              call dgemm('n','t',nub,nub**2,nob,0.5d0,H2C_vooo(:,:,i,k),nub,t2c(:,:,:,j),nub**2,1.0d0,temp,nub)
                              ! Diagram 2: A(i/jk)A(c/ab) I2C_vvov(a,b,i,e)*t2c(e,c,j,k)
                              call dgemm('n','n',nub**2,nub,nub,0.5d0,H2C_vvov_1243(:,:,:,i),nub**2,t2c(:,:,j,k),nub,1.0d0,temp,nub**2)
                              call dgemm('n','n',nub**2,nub,nub,-0.5d0,H2C_vvov_1243(:,:,:,j),nub**2,t2c(:,:,i,k),nub,1.0d0,temp,nub**2)
                              call dgemm('n','n',nub**2,nub,nub,-0.5d0,H2C_vvov_1243(:,:,:,k),nub**2,t2c(:,:,j,i),nub,1.0d0,temp,nub**2)
                              !$omp parallel shared(temp,vC_oovv,h1b_ov,h2c_ooov,h2c_vovv,i,j,k),&
                              !$omp private(t3d,t3_denom,a,b,c)
                              !$omp do schedule(static) reduction(+:resid_b,resid_bb)
                              do a = 1,nub
                                 do b = a+1,nub
                                    do c = b+1,nub
                                       t3_denom = fB_oo(i,i)+fB_oo(j,j)+fB_oo(k,k)-fB_vv(a,a)-fB_vv(b,b)-fB_vv(c,c)
                                       t3d = temp(a,b,c) + temp(b,c,a) + temp(c,a,b) - temp(a,c,b) - temp(b,a,c) - temp(c,b,a)
                                       t3d = t3d / t3_denom
                                       ! A(a/bc)A(i/jk) vC(jkbc)*t3d(abcijk)
                                       resid_b(a,i) = resid_b(a,i) + vC_oovv(j,k,b,c) * t3d ! (1)
                                       resid_b(b,i) = resid_b(b,i) - vC_oovv(j,k,a,c) * t3d ! (ae)
                                       resid_b(c,i) = resid_b(c,i) - vC_oovv(j,k,b,a) * t3d ! (af)
                                       resid_b(a,j) = resid_b(a,j) - vC_oovv(i,k,b,c) * t3d ! (im)
                                       resid_b(b,j) = resid_b(b,j) + vC_oovv(i,k,a,c) * t3d ! (ae)(im)
                                       resid_b(c,j) = resid_b(c,j) + vC_oovv(i,k,b,a) * t3d ! (af)(im)
                                       resid_b(a,k) = resid_b(a,k) - vC_oovv(j,i,b,c) * t3d ! (in)
                                       resid_b(b,k) = resid_b(b,k) + vC_oovv(j,i,a,c) * t3d ! (ae)(in)
                                       resid_b(c,k) = resid_b(c,k) + vC_oovv(j,i,b,a) * t3d ! (af)(in)
                                       ! A(ij)A(ab) [A(m/ij)A(e/ab) h1b(me) * t3d(abeijm)]
                                       resid_bb(a,b,i,j) = resid_bb(a,b,i,j) + H1B_ov(k,c) * t3d ! (1)
                                       resid_bb(a,b,k,j) = resid_bb(a,b,k,j) - H1B_ov(i,c) * t3d ! (im)
                                       resid_bb(a,b,i,k) = resid_bb(a,b,i,k) - H1B_ov(j,c) * t3d ! (jm)
                                       resid_bb(c,b,i,j) = resid_bb(c,b,i,j) - H1B_ov(k,a) * t3d ! (ae)
                                       resid_bb(c,b,k,j) = resid_bb(c,b,k,j) + H1B_ov(i,a) * t3d ! (im)(ae)
                                       resid_bb(c,b,i,k) = resid_bb(c,b,i,k) + H1B_ov(j,a) * t3d ! (jm)(ae)
                                       resid_bb(a,c,i,j) = resid_bb(a,c,i,j) - H1B_ov(k,b) * t3d ! (be)
                                       resid_bb(a,c,k,j) = resid_bb(a,c,k,j) + H1B_ov(i,b) * t3d ! (im)(be)
                                       resid_bb(a,c,i,k) = resid_bb(a,c,i,k) + H1B_ov(j,b) * t3d ! (jm)(be)
                                       ! A(ij)A(ab) [A(j/mn)A(f/ab) -h2c(mnif) * t3d(abfmjn)]
                                       resid_bb(a,b,:,j) = resid_bb(a,b,:,j) - H2C_ooov(i,k,:,c) * t3d ! (1)
                                       resid_bb(a,b,:,i) = resid_bb(a,b,:,i) + H2C_ooov(j,k,:,c) * t3d ! (jm)
                                       resid_bb(a,b,:,k) = resid_bb(a,b,:,k) + H2C_ooov(i,j,:,c) * t3d ! (jn)
                                       resid_bb(c,b,:,j) = resid_bb(c,b,:,j) + H2C_ooov(i,k,:,a) * t3d ! (af)
                                       resid_bb(c,b,:,i) = resid_bb(c,b,:,i) - H2C_ooov(j,k,:,a) * t3d ! (jm)(af)
                                       resid_bb(c,b,:,k) = resid_bb(c,b,:,k) - H2C_ooov(i,j,:,a) * t3d ! (jn)(af)
                                       resid_bb(a,c,:,j) = resid_bb(a,c,:,j) + H2C_ooov(i,k,:,b) * t3d ! (bf)
                                       resid_bb(a,c,:,i) = resid_bb(a,c,:,i) - H2C_ooov(j,k,:,b) * t3d ! (jm)(bf)
                                       resid_bb(a,c,:,k) = resid_bb(a,c,:,k) - H2C_ooov(i,j,:,b) * t3d ! (jn)(bf)
                                       ! A(ij)A(ab) [A(n/ij)A(b/ef) h2c(anef) * t3d(ebfijn)]
                                       resid_bb(:,b,i,j) = resid_bb(:,b,i,j) + H2C_vovv(:,k,a,c) * t3d ! (1)
                                       resid_bb(:,b,k,j) = resid_bb(:,b,k,j) - H2C_vovv(:,i,a,c) * t3d ! (in)
                                       resid_bb(:,b,i,k) = resid_bb(:,b,i,k) - H2C_vovv(:,j,a,c) * t3d ! (jn)
                                       resid_bb(:,a,i,j) = resid_bb(:,a,i,j) - H2C_vovv(:,k,b,c) * t3d ! (be)
                                       resid_bb(:,a,k,j) = resid_bb(:,a,k,j) + H2C_vovv(:,i,b,c) * t3d ! (in)(be)
                                       resid_bb(:,a,i,k) = resid_bb(:,a,i,k) + H2C_vovv(:,j,b,c) * t3d ! (jn)(be)
                                       resid_bb(:,c,i,j) = resid_bb(:,c,i,j) - H2C_vovv(:,k,a,b) * t3d ! (bf)
                                       resid_bb(:,c,k,j) = resid_bb(:,c,k,j) + H2C_vovv(:,i,a,b) * t3d ! (in)(bf)
                                       resid_bb(:,c,i,k) = resid_bb(:,c,i,k) + H2C_vovv(:,j,a,b) * t3d ! (jn)(bf)
                                    end do
                                 end do
                              end do
                              !$omp end do
                              !$omp end parallel
                           end do
                        end do
                      end do
                      deallocate(temp)
                      
                      ! update t1a
                      do i = 1,noa
                         do a = 1,nua
                            denom = fA_oo(i,i) - fA_vv(a,a)
                            resid_a(a,i) = (resid_a(a,i) + X1A(a,i))/(denom - shift)
                            t1a(a,i) = t1a(a,i) + resid_a(a,i)
                         end do
                      end do
                      ! update t1b
                      do i = 1,nob
                         do a = 1,nub
                            denom = fB_oo(i,i) - fB_vv(a,a)
                            resid_b(a,i) = (resid_b(a,i) + X1B(a,i))/(denom - shift)
                            t1b(a,i) = t1b(a,i) + resid_b(a,i)
                         end do
                      end do
                      ! update t2a
                      do i = 1,noa
                         do j = i+1,noa
                            do a = 1,nua
                               do b = a+1,nua
                                  denom = fA_oo(i,i) + fA_oo(j,j) - fA_vv(a,a) - fA_vv(b,b)
                                  
                                  resid_aa(a,b,i,j) = resid_aa(a,b,i,j) - resid_aa(b,a,i,j) - resid_aa(a,b,j,i) + resid_aa(b,a,j,i)
                                  val = X2A(a,b,i,j) - X2A(b,a,i,j) - X2A(a,b,j,i) + X2A(b,a,j,i)
                                  
                                  resid_aa(a,b,i,j) = (resid_aa(a,b,i,j) + val)/(denom - shift)
                                  resid_aa(b,a,i,j) = -resid_aa(a,b,i,j)
                                  resid_aa(a,b,j,i) = -resid_aa(a,b,i,j)
                                  resid_aa(b,a,j,i) = resid_aa(a,b,i,j)
                                  
                                  t2a(a,b,i,j) = t2a(a,b,i,j) + resid_aa(a,b,i,j)
                                  t2a(b,a,i,j) = -t2a(a,b,i,j)
                                  t2a(a,b,j,i) = -t2a(a,b,i,j)
                                  t2a(b,a,j,i) = t2a(a,b,i,j)
                               end do
                            end do
                         end do
                      end do
                      ! update t2b
                      do i = 1,noa
                         do j = 1,nob
                            do a = 1,nua
                               do b = 1,nub
                                  denom = fA_oo(i,i) + fB_oo(j,j) - fA_vv(a,a) - fB_vv(b,b)
                                  resid_ab(a,b,i,j) = (resid_ab(a,b,i,j) + X2B(a,b,i,j))/(denom - shift)
                                  t2b(a,b,i,j) = t2b(a,b,i,j) + resid_ab(a,b,i,j)
                               end do
                            end do
                         end do
                      end do
                      ! update t2c
                      do i = 1,nob
                         do j = i+1,nob
                            do a = 1,nub
                               do b = a+1,nub
                                  denom = fB_oo(i,i) + fB_oo(j,j) - fB_vv(a,a) - fB_vv(b,b)
                                  
                                  resid_bb(a,b,i,j) = resid_bb(a,b,i,j) - resid_bb(b,a,i,j) - resid_bb(a,b,j,i) + resid_bb(b,a,j,i)
                                  val = X2C(a,b,i,j) - X2C(b,a,i,j) - X2C(a,b,j,i) + X2C(b,a,j,i)
                                  
                                  resid_bb(a,b,i,j) = (resid_bb(a,b,i,j) + val)/(denom - shift)
                                  resid_bb(b,a,i,j) = -resid_bb(a,b,i,j)
                                  resid_bb(a,b,j,i) = -resid_bb(a,b,i,j)
                                  resid_bb(b,a,j,i) = resid_bb(a,b,i,j)
                                  
                                  t2c(a,b,i,j) = t2c(a,b,i,j) + resid_bb(a,b,i,j)
                                  t2c(b,a,i,j) = -t2c(a,b,i,j)
                                  t2c(a,b,j,i) = -t2c(a,b,i,j)
                                  t2c(b,a,j,i) = t2c(a,b,i,j)
                               end do
                            end do
                         end do
                      end do
                      ! manually set diagonally broadcasted blocks in aa and bb residuals to 0
                      do a = 1,nua
                         resid_aa(a,a,:,:) = 0.0d0
                      end do
                      do i = 1,noa
                         resid_aa(:,:,i,i) = 0.0d0
                      end do
                      do a = 1,nub
                         resid_bb(a,a,:,:) = 0.0d0
                      end do
                      do i = 1,nob
                         resid_bb(:,:,i,i) = 0.0d0
                      end do
               end subroutine update_t
	
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
      
              !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! REORDER ROUTINES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         
!               subroutine reorder4(y, x, iorder)
!
!                   integer, intent(in) :: iorder(4)
!                   real(kind=8), intent(in) :: x(:,:,:,:)
!
!                   real(kind=8), intent(out) :: y(:,:,:,:)
!
!                   integer :: i, j, k, l
!                   integer :: vec(4)
!
!                   y = 0.0d0
!                   do i = 1, size(x,1)
!                      do j = 1, size(x,2)
!                         do k = 1, size(x,3)
!                            do l = 1, size(x,4)
!                               vec = (/i,j,k,l/)
!                               y(vec(iorder(1)),vec(iorder(2)),vec(iorder(3)),vec(iorder(4))) = x(i,j,k,l)
!                            end do
!                         end do
!                      end do
!                   end do
!
!               end subroutine reorder4

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

             subroutine reorder3214(x_in,x_out)

                      real(kind=8), intent(in) :: x_in(:,:,:,:)
                      real(kind=8), intent(out) :: x_out(:,:,:,:)

                      integer :: i1, i2, i3, i4

                      do i1 = 1,size(x_in,1)
                         do i2 = 1,size(x_in,2)
                            do i3 = 1,size(x_in,3)
                               do i4= 1,size(x_in,4)
                                  x_out(i3,i2,i1,i4) = x_in(i1,i2,i3,i4)
                               end do
                            end do
                         end do
                      end do

             end subroutine reorder3214
end module cc3_loops
