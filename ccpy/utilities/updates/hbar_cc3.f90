module hbar_cc3
	
	implicit none
	
	contains
			subroutine build_hbar(h2a_vooo,h2a_vvov,&
                                     h2b_vooo,h2b_ovoo,h2b_vvov,h2b_vvvo,&
                                     h2c_vooo,h2c_vvov,&
                                     t2a,t2b,t2c,&
                                     x2a_vooo,x2a_vvov,&
                                     x2b_vooo,x2b_ovoo,x2b_vvov,x2b_vvvo,&
                                     x2c_vooo,x2c_vvov,&
                                     fA_oo,fA_vv,fB_oo,fB_vv,&
                                     h2a_oovv,h2b_oovv,h2c_oovv,&
                                     noa,nua,nob,nub)

                      integer, intent(in) :: noa, nua, nob, nub
                      real(kind=8), intent(in) :: fA_oo(noa,noa),fA_vv(nua,nua),&
                                                  fB_oo(nob,nob),fB_vv(nub,nub),&
                                                  h2a_oovv(noa,noa,nua,nua),&
                                                  h2b_oovv(noa,nob,nua,nub),&
                                                  h2c_oovv(nob,nob,nub,nub),&
                                                  x2a_vooo(nua,noa,noa,noa),&
                                                  x2a_vvov(nua,nua,noa,nua),&
                                                  x2b_vooo(nua,nob,noa,nob),&
                                                  x2b_ovoo(noa,nub,noa,nob),&
                                                  x2b_vvov(nua,nub,noa,nub),&
                                                  x2b_vvvo(nua,nub,nua,nob),&
                                                  x2c_vooo(nub,nob,nob,nob),&
                                                  x2c_vvov(nub,nub,nob,nub)
                      real(kind=8), intent(in) :: t2a(1:nua,1:nua,1:noa,1:noa)
                      real(kind=8), intent(in) :: t2b(1:nua,1:nub,1:noa,1:nob)
                      real(kind=8), intent(in) :: t2c(1:nub,1:nub,1:nob,1:nob)
                      
                      real(kind=8), intent(inout) :: h2a_vooo(nua,noa,noa,noa)
                      !f2py intent(in,out) :: h2a_vooo(0:nua-1,0:noa-1,0:noa-1,0:noa-1)
                      real(kind=8), intent(inout) :: h2a_vvov(nua,nua,noa,nua)
                      !f2py intent(in,out) :: h2a_vvov(0:nua-1,0:nua-1,0:noa-1,0:nua-1)
                      real(kind=8), intent(inout) :: h2b_vooo(nua,nob,noa,nob)
                      !f2py intent(in,out) :: h2b_vooo(0:nua-1,0:nob-1,0:noa-1,0:nob-1)
                      real(kind=8), intent(inout) :: h2b_ovoo(noa,nub,noa,nob)
                      !f2py intent(in,out) :: h2b_ovoo(0:noa-1,0:nub-1,0:noa-1,0:nob-1)
                      real(kind=8), intent(inout) :: h2b_vvov(nua,nub,noa,nub)
                      !f2py intent(in,out) :: h2b_vvov(0:nua-1,0:nub-1,0:noa-1,0:nub-1)
                      real(kind=8), intent(inout) :: h2b_vvvo(nua,nub,nua,nob)
                      !f2py intent(in,out) :: h2b_vvvo(0:nua-1,0:nub-1,0:nua-1,0:nob-1)
                      real(kind=8), intent(inout) :: h2c_vooo(nub,nob,nob,nob)
                      !f2py intent(in,out) :: h2c_vooo(0:nub-1,0:nob-1,0:nob-1,0:nob-1)
                      real(kind=8), intent(inout) :: h2c_vvov(nub,nub,nob,nub)
                      !f2py intent(in,out) :: h2c_vvov(0:nub-1,0:nub-1,0:nob-1,0:nub-1)

                      integer :: i, j, k, a, b, c, m, e
                      real(kind=8) :: denom, val
                      real(kind=8) :: t3a_o, t3a_v, t3b_o, t3b_v, t3c_o, t3c_v, t3d_o, t3d_v
                      real(kind=8) :: t3a, t3b, t3c, t3d
                      real(kind=8) :: t3_denom
                      
                      ! allocatable array to hold t3(abc) for a given (i,j,k) block
                      real(kind=8), allocatable :: temp(:,:,:)
                      ! reordered arrays for the DGEMM operations
                      real(kind=8) :: X2A_vvov_1243(nua,nua,nua,noa)
                      real(kind=8) :: X2B_vvov_1243(nua,nub,nub,noa), t2b_1243(nua,nub,nob,noa)
                      real(kind=8) :: X2C_vvov_4213(nub,nub,nub,noa), X2C_vooo_2134(nob,nub,nob,nob)
                      real(kind=8) :: X2C_vvov_1243(nub,nub,nub,nob)
                      
                      ! Call reordering routines for arrays entering DGEMM
                      call reorder4(x2a_vvov_1243, x2a_vvov, (/1,2,4,3/))
                      call reorder4(x2b_vvov_1243, x2b_vvov, (/1,2,4,3/))
                      call reorder4(t2b_1243, t2b, (/1,2,4,3/))
                      call reorder4(x2c_vvov_4213, x2c_vvov, (/4,2,1,3/))
                      call reorder4(x2c_vooo_2134, x2c_vooo, (/2,1,3,4/))
                      call reorder4(x2c_vvov_1243, x2c_vvov, (/1,2,4,3/))
                      
                      ! Scale these terms by 1/2 to account for antisymmetrizer applied at the end
                      h2a_vooo = 0.5d0 * h2a_vooo
                      h2a_vvov = 0.5d0 * h2a_vvov
                      h2c_vooo = 0.5d0 * h2c_vooo
                      h2c_vvov = 0.5d0 * h2c_vvov

                      ! contribution from t3a
                      allocate(temp(nua,nua,nua))
                      do i = 1,noa
                        do j = i+1,noa
                           do k = j+1,noa
                              temp = 0.0d0
                              ! Diagram 1: -A(k/ij)A(a/bc) I2A_vooo(a,m,i,j)*t2a(b,c,m,k)
                              call dgemm('n','t',nua,nua**2,noa,-0.5d0,X2A_vooo(:,:,i,j),nua,t2a(:,:,:,k),nua**2,1.0d0,temp,nua)
                              call dgemm('n','t',nua,nua**2,noa,0.5d0,X2A_vooo(:,:,k,j),nua,t2a(:,:,:,i),nua**2,1.0d0,temp,nua)
                              call dgemm('n','t',nua,nua**2,noa,0.5d0,X2A_vooo(:,:,i,k),nua,t2a(:,:,:,j),nua**2,1.0d0,temp,nua)
                              ! Diagram 2: A(i/jk)A(c/ab) I2A_vvov(a,b,i,e)*t2a(e,c,j,k)
                              call dgemm('n','n',nua**2,nua,nua,0.5d0,X2A_vvov_1243(:,:,:,i),nua**2,t2a(:,:,j,k),nua,1.0d0,temp,nua**2)
                              call dgemm('n','n',nua**2,nua,nua,-0.5d0,X2A_vvov_1243(:,:,:,j),nua**2,t2a(:,:,i,k),nua,1.0d0,temp,nua**2)
                              call dgemm('n','n',nua**2,nua,nua,-0.5d0,X2A_vvov_1243(:,:,:,k),nua**2,t2a(:,:,j,i),nua,1.0d0,temp,nua**2)
                              do a = 1,nua
                                 do b = a+1,nua
                                    do c = b+1,nua
                                       t3_denom = fA_oo(i,i)+fA_oo(j,j)+fA_oo(k,k)-fA_vv(a,a)-fA_vv(b,b)-fA_vv(c,c)
                                       t3a = temp(a,b,c) + temp(b,c,a) + temp(c,a,b) - temp(a,c,b) - temp(b,a,c) - temp(c,b,a)
                                       t3a = t3a / t3_denom
                                       ! I2A(amij) <- A(ij) [A(n/ij)A(a/ef) h2a(mnef) * t3a(aefijn)]
                                       h2a_vooo(a,:,i,j) = h2a_vooo(a,:,i,j) + h2a_oovv(:,k,b,c) * t3a ! (1)
                                       h2a_vooo(a,:,j,k) = h2a_vooo(a,:,j,k) + h2a_oovv(:,i,b,c) * t3a ! (in)
                                       h2a_vooo(a,:,i,k) = h2a_vooo(a,:,i,k) - h2a_oovv(:,j,b,c) * t3a ! (jn)
                                       h2a_vooo(b,:,i,j) = h2a_vooo(b,:,i,j) - h2a_oovv(:,k,a,c) * t3a ! (ae)
                                       h2a_vooo(b,:,j,k) = h2a_vooo(b,:,j,k) - h2a_oovv(:,i,a,c) * t3a ! (in)(ae)
                                       h2a_vooo(b,:,i,k) = h2a_vooo(b,:,i,k) + h2a_oovv(:,j,a,c) * t3a ! (jn)(ae)
                                       h2a_vooo(c,:,i,j) = h2a_vooo(c,:,i,j) - h2a_oovv(:,k,b,a) * t3a ! (af)
                                       h2a_vooo(c,:,j,k) = h2a_vooo(c,:,j,k) - h2a_oovv(:,i,b,a) * t3a ! (in)(af)
                                       h2a_vooo(c,:,i,k) = h2a_vooo(c,:,i,k) + h2a_oovv(:,j,b,a) * t3a ! (jn)(af)
                                       ! I2A(abie) <- A(ab) [A(i/mn)A(f/ab) -h2a(mnef) * t3a(abfimn)]
                                       h2a_vvov(a,b,i,:) = h2a_vvov(a,b,i,:) - h2a_oovv(j,k,:,c) * t3a ! (1)
                                       h2a_vvov(a,b,j,:) = h2a_vvov(a,b,j,:) + h2a_oovv(i,k,:,c) * t3a ! (im)
                                       h2a_vvov(a,b,k,:) = h2a_vvov(a,b,k,:) + h2a_oovv(j,i,:,c) * t3a ! (in)
                                       h2a_vvov(b,c,i,:) = h2a_vvov(b,c,i,:) - h2a_oovv(j,k,:,a) * t3a ! (af)
                                       h2a_vvov(b,c,j,:) = h2a_vvov(b,c,j,:) + h2a_oovv(i,k,:,a) * t3a ! (im)(af)
                                       h2a_vvov(b,c,k,:) = h2a_vvov(b,c,k,:) + h2a_oovv(j,i,:,a) * t3a ! (in)(af)
                                       h2a_vvov(a,c,i,:) = h2a_vvov(a,c,i,:) + h2a_oovv(j,k,:,b) * t3a ! (bf)
                                       h2a_vvov(a,c,j,:) = h2a_vvov(a,c,j,:) - h2a_oovv(i,k,:,b) * t3a ! (im)(bf)
                                       h2a_vvov(a,c,k,:) = h2a_vvov(a,c,k,:) - h2a_oovv(j,i,:,b) * t3a ! (in)(bf)
                                    end do
                                 end do
                              end do
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
                              call dgemm('n','t',nua,nua*nub,nua,1.0d0,t2a(:,:,i,j),nua,X2B_vvvo(:,:,:,k),nua*nub,1.0d0,temp,nua)
                              ! Diagram 2: -A(ij) I2B(mcjk)*t2a(abim)
                              call dgemm('n','n',nua**2,nub,noa,0.5d0,t2a(:,:,:,i),nua**2,X2B_ovoo(:,:,j,k),noa,1.0d0,temp,nua**2)
                              call dgemm('n','n',nua**2,nub,noa,-0.5d0,t2a(:,:,:,j),nua**2,X2B_ovoo(:,:,i,k),noa,1.0d0,temp,nua**2)
                              ! Diagram 3: A(ab)A(ij) H2B(acie)*t2b(bejk) -> A(ab)A(ij) t2b(aeik)*H2B(bcje)
                              call dgemm('n','t',nua,nua*nub,nub,1.0d0,t2b(:,:,i,k),nua,X2B_vvov_1243(:,:,:,j),nua*nub,1.0d0,temp,nua)
                              call dgemm('n','t',nua,nua*nub,nub,-1.0d0,t2b(:,:,j,k),nua,X2B_vvov_1243(:,:,:,i),nua*nub,1.0d0,temp,nua)
                              ! Diagram 4: -A(ab)A(ij) I2B(amik)*t2b(bcjm)
                              call dgemm('n','t',nua,nua*nub,nob,-1.0d0,X2B_vooo(:,:,i,k),nua,t2b_1243(:,:,:,j),nua*nub,1.0d0,temp,nua)
                              call dgemm('n','t',nua,nua*nub,nob,1.0d0,X2B_vooo(:,:,j,k),nua,t2b_1243(:,:,:,i),nua*nub,1.0d0,temp,nua)
                              ! Diagram 5: A(ij) H2A(abie)*t2b(ecjk)
                              call dgemm('n','n',nua**2,nub,nua,0.5d0,X2A_vvov_1243(:,:,:,i),nua**2,t2b(:,:,j,k),nua,1.0d0,temp,nua**2)
                              call dgemm('n','n',nua**2,nub,nua,-0.5d0,X2A_vvov_1243(:,:,:,j),nua**2,t2b(:,:,i,k),nua,1.0d0,temp,nua**2)
                              ! Diagram 6: -A(ab) I2A(amij)*t2b(bcmk)
                              call dgemm('n','t',nua,nua*nub,noa,-1.0d0,X2A_vooo(:,:,i,j),nua,t2b(:,:,:,k),nua*nub,1.0d0,temp,nua)
                              do a = 1,nua
                                 do b = a+1,nua
                                    do c = 1,nub
                                       t3_denom = fA_oo(i,i)+fA_oo(j,j)+fB_oo(k,k)-fA_vv(a,a)-fA_vv(b,b)-fB_vv(c,c)
                                       t3b = temp(a,b,c) - temp(b,a,c)
                                       t3b = t3b / t3_denom
                                       ! I2A(amij) <- A(ij) [A(ae) h2b(mnef) * t3b(aefijn)]
                                       h2a_vooo(a,:,i,j) = h2a_vooo(a,:,i,j) + h2b_oovv(:,k,b,c) * t3b ! (1)
                                       h2a_vooo(b,:,i,j) = h2a_vooo(b,:,i,j) - h2b_oovv(:,k,a,c) * t3b ! (ae)
                                       ! I2A(abie) <- A(ab) [A(im) -h2b(mnef) * t3b(abfimn)]
                                       h2a_vvov(a,b,i,:) = h2a_vvov(a,b,i,:) - h2b_oovv(j,k,:,c) * t3b ! (1)
                                       h2a_vvov(a,b,j,:) = h2a_vvov(a,b,j,:) + h2b_oovv(i,k,:,c) * t3b ! (im)
                                       ! I2B(amij) <- A(af)A(in) h2b(nmfe) * t3b(afeinj)
                                       h2b_vooo(a,:,i,k) = h2b_vooo(a,:,i,k) + h2b_oovv(j,:,b,c) * t3b ! (1)
                                       h2b_vooo(b,:,i,k) = h2b_vooo(b,:,i,k) - h2b_oovv(j,:,a,c) * t3b ! (af)
                                       h2b_vooo(a,:,j,k) = h2b_vooo(a,:,j,k) - h2b_oovv(i,:,b,c) * t3b ! (in)
                                       h2b_vooo(b,:,j,k) = h2b_vooo(b,:,j,k) + h2b_oovv(i,:,a,c) * t3b ! (af)(in)
                                       ! I2B(mbij) <- A(in) h2a(mnef) * t3b(efbinj)
                                       h2b_ovoo(:,c,i,k) = h2b_ovoo(:,c,i,k) + h2a_oovv(:,j,a,b) * t3b ! (1)
                                       h2b_ovoo(:,c,j,k) = h2b_ovoo(:,c,j,k) - h2a_oovv(:,i,a,b) * t3b ! (in)
                                       ! I2B(abie) <- A(af)A(in) -h2b(nmfe) * t3b(afbinm)
                                       h2b_vvov(a,c,i,:) = h2b_vvov(a,c,i,:) - h2b_oovv(j,k,b,:) * t3b ! (1)
                                       h2b_vvov(b,c,i,:) = h2b_vvov(b,c,i,:) + h2b_oovv(j,k,a,:) * t3b ! (af)
                                       h2b_vvov(a,c,j,:) = h2b_vvov(a,c,j,:) + h2b_oovv(i,k,b,:) * t3b ! (in)
                                       h2b_vvov(b,c,j,:) = h2b_vvov(b,c,j,:) - h2b_oovv(i,k,a,:) * t3b ! (af)(in)
                                       ! I2B(abej) <- A(af) -h2a(mnef) * t3b(afbmnj)
                                       h2b_vvvo(a,c,:,k) = h2b_vvvo(a,c,:,k) - h2a_oovv(i,j,:,b) * t3b ! (1)
                                       h2b_vvvo(b,c,:,k) = h2b_vvvo(b,c,:,k) + h2a_oovv(i,j,:,a) * t3b ! (af)
                                    end do
                                 end do
                              end do
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
                              call dgemm('n','n',nua*nub,nub,nub,1.0d0,X2B_vvov_1243(:,:,:,i),nua*nub,t2c(:,:,j,k),nub,1.0d0,temp,nua*nub)
                              ! Diagram 2: -A(jk) I2B_vooo(a,m,i,j)*t2c(b,c,m,k)
                              call dgemm('n','t',nua,nub**2,nob,-0.5d0,X2B_vooo(:,:,i,j),nua,t2c(:,:,:,k),nub**2,1.0d0,temp,nua)
                              call dgemm('n','t',nua,nub**2,nob,0.5d0,X2B_vooo(:,:,i,k),nua,t2c(:,:,:,j),nub**2,1.0d0,temp,nua)
                              ! Diagram 3: A(jk) H2C_vvov(c,b,k,e)*t2b(a,e,i,j)
                              call dgemm('n','n',nua,nub**2,nub,0.5d0,t2b(:,:,i,j),nua,X2C_vvov_4213(:,:,:,k),nub,1.0d0,temp,nua)
                              call dgemm('n','n',nua,nub**2,nub,-0.5d0,t2b(:,:,i,k),nua,X2C_vvov_4213(:,:,:,j),nub,1.0d0,temp,nua)
                              ! Diagram 4: -A(bc) I2C_vooo(c,m,k,j)*t2b(a,b,i,m)
                              call dgemm('n','n',nua*nub,nub,nob,-1.0d0,t2b_1243(:,:,:,i),nua*nub,X2C_vooo_2134(:,:,k,j),nob,1.0d0,temp,nua*nub)
                              ! Diagram 5: A(jk)A(bc) H2B_vvvo(a,b,e,j)*t2b(e,c,i,k)
                              call dgemm('n','n',nua*nub,nub,nua,1.0d0,X2B_vvvo(:,:,:,j),nua*nub,t2b(:,:,i,k),nua,1.0d0,temp,nua*nub)
                              call dgemm('n','n',nua*nub,nub,nua,-1.0d0,X2B_vvvo(:,:,:,k),nua*nub,t2b(:,:,i,j),nua,1.0d0,temp,nua*nub)
                              ! Diagram 6: -A(jk)A(bc) I2B_ovoo(m,b,i,j)*t2b(a,c,m,k) -> -A(jk)A(bc) I2B_ovoo(m,c,i,k)*t2b(a,b,m,j)
                              call dgemm('n','n',nua*nub,nub,noa,-1.0d0,t2b(:,:,:,j),nua*nub,X2B_ovoo(:,:,i,k),noa,1.0d0,temp,nua*nub)
                              call dgemm('n','n',nua*nub,nub,noa,1.0d0,t2b(:,:,:,k),nua*nub,X2B_ovoo(:,:,i,j),noa,1.0d0,temp,nua*nub)
                              do a = 1,nua
                                 do b = 1,nub
                                    do c = b+1,nub
                                       t3_denom = fA_oo(i,i)+fB_oo(j,j)+fB_oo(k,k)-fA_vv(a,a)-fB_vv(b,b)-fB_vv(c,c)
                                       t3c = temp(a,b,c) - temp(a,c,b)
                                       t3c = t3c / t3_denom
                                       ! I2B(amij) <- A(jn) h2c(nmfe) * t3c(afeinj)
                                       h2b_vooo(a,:,i,k) = h2b_vooo(a,:,i,k) + h2c_oovv(j,:,b,c) * t3c ! (1)
                                       h2b_vooo(a,:,i,j) = h2b_vooo(a,:,i,j) - h2c_oovv(k,:,b,c) * t3c ! (jn)
                                       ! I2B(mbij) <- A(bf)A(jn) h2B(mnef) * t3c(efbinj)
                                       h2b_ovoo(:,c,i,k) = h2b_ovoo(:,c,i,k) + h2b_oovv(:,j,a,b) * t3c ! (1)
                                       h2b_ovoo(:,b,i,k) = h2b_ovoo(:,b,i,k) - h2b_oovv(:,j,a,c) * t3c ! (bf)
                                       h2b_ovoo(:,c,i,j) = h2b_ovoo(:,c,i,j) - h2b_oovv(:,k,a,b) * t3c ! (jn)
                                       h2b_ovoo(:,b,i,j) = h2b_ovoo(:,b,i,j) + h2b_oovv(:,k,a,c) * t3c ! (bf)(jn)
                                       ! I2B(abie) <- A(bf) -h2c(nmfe) * t3c(afbinm)
                                       h2b_vvov(a,c,i,:) = h2b_vvov(a,c,i,:) - h2c_oovv(j,k,b,:) * t3c ! (1)
                                       h2b_vvov(a,b,i,:) = h2b_vvov(a,b,i,:) + h2c_oovv(j,k,c,:) * t3c ! (bf)
                                       ! I2B(abej) <- A(bf)A(jn) -h2b(mnef) * t3c(afbmnj)
                                       h2b_vvvo(a,c,:,k) = h2b_vvvo(a,c,:,k) - h2b_oovv(i,j,:,b) * t3c ! (1)
                                       h2b_vvvo(a,b,:,k) = h2b_vvvo(a,b,:,k) + h2b_oovv(i,j,:,c) * t3c ! (bf)
                                       h2b_vvvo(a,c,:,j) = h2b_vvvo(a,c,:,j) + h2b_oovv(i,k,:,b) * t3c ! (jn)
                                       h2b_vvvo(a,b,:,j) = h2b_vvvo(a,b,:,j) - h2b_oovv(i,k,:,c) * t3c ! (bf)(jn)
                                       ! I2C(amij) <- A(ij) [A(ae) h2b(nmfe) * t3c(faenij)]
                                       h2c_vooo(b,:,j,k) = h2c_vooo(b,:,j,k) + h2b_oovv(i,:,a,c) * t3c ! (1)
                                       h2c_vooo(c,:,j,k) = h2c_vooo(c,:,j,k) - h2b_oovv(i,:,a,b) * t3c ! (ae)
                                       ! I2C(abie) <- A(ab) [A(im) -h2b(nmfe) * t3c(fabnim)]
                                       h2c_vvov(b,c,j,:) = h2c_vvov(b,c,j,:) - h2b_oovv(i,k,a,:) * t3c ! (1)
                                       h2c_vvov(b,c,k,:) = h2c_vvov(b,c,k,:) + h2b_oovv(i,j,a,:) * t3c ! (im)
                                    end do
                                 end do
                              end do
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
                              call dgemm('n','t',nub,nub**2,nob,-0.5d0,X2C_vooo(:,:,i,j),nub,t2c(:,:,:,k),nub**2,1.0d0,temp,nub)
                              call dgemm('n','t',nub,nub**2,nob,0.5d0,X2C_vooo(:,:,k,j),nub,t2c(:,:,:,i),nub**2,1.0d0,temp,nub)
                              call dgemm('n','t',nub,nub**2,nob,0.5d0,X2C_vooo(:,:,i,k),nub,t2c(:,:,:,j),nub**2,1.0d0,temp,nub)
                              ! Diagram 2: A(i/jk)A(c/ab) I2C_vvov(a,b,i,e)*t2c(e,c,j,k)
                              call dgemm('n','n',nub**2,nub,nub,0.5d0,X2C_vvov_1243(:,:,:,i),nub**2,t2c(:,:,j,k),nub,1.0d0,temp,nub**2)
                              call dgemm('n','n',nub**2,nub,nub,-0.5d0,X2C_vvov_1243(:,:,:,j),nub**2,t2c(:,:,i,k),nub,1.0d0,temp,nub**2)
                              call dgemm('n','n',nub**2,nub,nub,-0.5d0,X2C_vvov_1243(:,:,:,k),nub**2,t2c(:,:,j,i),nub,1.0d0,temp,nub**2)
                              do a = 1,nub
                                 do b = a+1,nub
                                    do c = b+1,nub
                                       t3_denom = fB_oo(i,i)+fB_oo(j,j)+fB_oo(k,k)-fB_vv(a,a)-fB_vv(b,b)-fB_vv(c,c)
                                       t3d = temp(a,b,c) + temp(b,c,a) + temp(c,a,b) - temp(a,c,b) - temp(b,a,c) - temp(c,b,a)
                                       t3d = t3d / t3_denom
                                       ! I2C(amij) <- A(ij) [A(n/ij)A(a/ef) h2c(mnef) * t3d(aefijn)]
                                       h2c_vooo(a,:,i,j) = h2c_vooo(a,:,i,j) + h2c_oovv(:,k,b,c) * t3d ! (1)
                                       h2c_vooo(a,:,j,k) = h2c_vooo(a,:,j,k) + h2c_oovv(:,i,b,c) * t3d ! (in)
                                       h2c_vooo(a,:,i,k) = h2c_vooo(a,:,i,k) - h2c_oovv(:,j,b,c) * t3d ! (jn)
                                       h2c_vooo(b,:,i,j) = h2c_vooo(b,:,i,j) - h2c_oovv(:,k,a,c) * t3d ! (ae)
                                       h2c_vooo(b,:,j,k) = h2c_vooo(b,:,j,k) - h2c_oovv(:,i,a,c) * t3d ! (in)(ae)
                                       h2c_vooo(b,:,i,k) = h2c_vooo(b,:,i,k) + h2c_oovv(:,j,a,c) * t3d ! (jn)(ae)
                                       h2c_vooo(c,:,i,j) = h2c_vooo(c,:,i,j) - h2c_oovv(:,k,b,a) * t3d ! (af)
                                       h2c_vooo(c,:,j,k) = h2c_vooo(c,:,j,k) - h2c_oovv(:,i,b,a) * t3d ! (in)(af)
                                       h2c_vooo(c,:,i,k) = h2c_vooo(c,:,i,k) + h2c_oovv(:,j,b,a) * t3d ! (jn)(af)
                                       ! I2C(abie) <- A(ab) [A(i/mn)A(f/ab) -h2c(mnef) * t3d(abfimn)]
                                       h2c_vvov(a,b,i,:) = h2c_vvov(a,b,i,:) - h2c_oovv(j,k,:,c) * t3d ! (1)
                                       h2c_vvov(a,b,j,:) = h2c_vvov(a,b,j,:) + h2c_oovv(i,k,:,c) * t3d ! (im)
                                       h2c_vvov(a,b,k,:) = h2c_vvov(a,b,k,:) + h2c_oovv(j,i,:,c) * t3d ! (in)
                                       h2c_vvov(b,c,i,:) = h2c_vvov(b,c,i,:) - h2c_oovv(j,k,:,a) * t3d ! (af)
                                       h2c_vvov(b,c,j,:) = h2c_vvov(b,c,j,:) + h2c_oovv(i,k,:,a) * t3d ! (im)(af)
                                       h2c_vvov(b,c,k,:) = h2c_vvov(b,c,k,:) + h2c_oovv(j,i,:,a) * t3d ! (in)(af)
                                       h2c_vvov(a,c,i,:) = h2c_vvov(a,c,i,:) + h2c_oovv(j,k,:,b) * t3d ! (bf)
                                       h2c_vvov(a,c,j,:) = h2c_vvov(a,c,j,:) - h2c_oovv(i,k,:,b) * t3d ! (im)(bf)
                                       h2c_vvov(a,c,k,:) = h2c_vvov(a,c,k,:) - h2c_oovv(j,i,:,b) * t3d ! (in)(bf)
                                    end do
                                 end do
                              end do
                           end do
                        end do
                      end do
                      deallocate(temp)
                      
                      ! apply the common A(ij) antisymmetrizer
                      do i = 1, noa
                         do j = i+1, noa
                            do m = 1, noa
                               do a = 1, nua
                                  h2a_vooo(a,m,i,j) = h2a_vooo(a,m,i,j) - h2a_vooo(a,m,j,i)
                               end do
                            end do
                         end do
                      end do
                      ! explicitly antisymmetrize
                      do i = 1, noa
                         do j = i+1, noa
                            h2a_vooo(:,:,j,i) = -h2a_vooo(:,:,i,j)
                         end do
                      end do
                      ! apply the common A(ab) antisymmetrizer
                      do e = 1, nua
                         do i = 1, noa
                            do a = 1, nua
                                do b = a+1, nua
                                  h2a_vvov(a,b,i,e) = h2a_vvov(a,b,i,e) - h2a_vvov(b,a,i,e)
                               end do
                             end do
                         end do
                      end do
                      ! explicitly antisymmetrize
                      do a = 1, nua
                         do b = a+1, nua
                            h2a_vvov(b,a,:,:) = -h2a_vvov(a,b,:,:)
                         end do
                      end do
                      ! apply the common A(ij) antisymmetrizer
                      do i = 1, nob
                         do j = i+1, nob
                            do m = 1, nob
                               do a = 1, nub
                                  h2c_vooo(a,m,i,j) = h2c_vooo(a,m,i,j) - h2c_vooo(a,m,j,i)
                               end do
                            end do
                         end do
                      end do
                      ! explicitly antisymmetrize
                      do i = 1, nob
                         do j = i+1, nob
                            h2c_vooo(:,:,j,i) = -h2c_vooo(:,:,i,j)
                         end do
                      end do
                      ! apply the common A(ab) antisymmetrizer
                      do e = 1, nub
                         do i = 1, nob
                            do a = 1, nub
                               do b = a+1, nub
                                  h2c_vvov(a,b,i,e) = h2c_vvov(a,b,i,e) - h2c_vvov(b,a,i,e)
                               end do
                            end do
                         end do
                      end do
                      ! explicitly antisymmetrize
                      do a = 1, nub
                         do b = a+1, nub
                            h2c_vvov(b,a,:,:) = -h2c_vvov(a,b,:,:)
                         end do
                      end do
            end subroutine build_hbar

            subroutine reorder4(y, x, iorder)

                   integer, intent(in) :: iorder(4)
                   real(kind=8), intent(in) :: x(:,:,:,:)

                   real(kind=8), intent(out) :: y(:,:,:,:)

                   integer :: i, j, k, l
                   integer :: vec(4)

                   y = 0.0d0
                   do i = 1, size(x,1)
                      do j = 1, size(x,2)
                         do k = 1, size(x,3)
                            do l = 1, size(x,4)
                               vec = (/i,j,k,l/)
                               y(vec(iorder(1)),vec(iorder(2)),vec(iorder(3)),vec(iorder(4))) = x(i,j,k,l)
                            end do
                         end do
                      end do
                   end do

            end subroutine reorder4

end module hbar_cc3