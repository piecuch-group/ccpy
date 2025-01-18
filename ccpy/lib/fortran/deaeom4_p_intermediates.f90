module deaeom4_p_intermediates

        implicit none
        
        contains

              subroutine add_r4_x3b_vvoo(x3b_vvoo,&
                                         r4b_amps, r4b_excits,&
                                         r4c_amps, r4c_excits,&
                                         h2a_oovv, h2b_oovv,&
                                         n4abaa, n4abab,&
                                         noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n4abaa, n4abab

                  integer, intent(in) :: r4b_excits(n4abaa,6)
                  real(kind=8), intent(in) :: r4b_amps(n4abaa)
                  integer, intent(in) :: r4c_excits(n4abab,6)
                  real(kind=8), intent(in) :: r4c_amps(n4abab)

                  real(kind=8), intent(in) :: h2a_oovv(noa,noa,nua,nua)
                  real(kind=8), intent(in) :: h2b_oovv(noa,nob,nua,nub)

                  real(kind=8), intent(inout) :: x3b_vvoo(nua,nub,noa,noa)
                  !f2py intent(in,out) :: x3b_vvoo(0:nua-1,0:nub-1,0:noa-1,0:noa-1)

                  real(kind=8) :: r_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  do idet=1,n4abaa
                     r_amp = r4b_amps(idet)
                     ! x3b(a,b,m,k) <- A(a/ef)A(kn) h2a(mnef)*r4c(abefkn)
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); e = r4b_excits(idet,3); f = r4b_excits(idet,4);
                     k = r4b_excits(idet,5); n = r4b_excits(idet,6);
                     x3b_vvoo(a,b,:,k) = x3b_vvoo(a,b,:,k) + h2a_oovv(:,n,e,f)*r_amp ! (1)
                     x3b_vvoo(e,b,:,k) = x3b_vvoo(e,b,:,k) - h2a_oovv(:,n,a,f)*r_amp ! (ae)
                     x3b_vvoo(f,b,:,k) = x3b_vvoo(f,b,:,k) - h2a_oovv(:,n,e,a)*r_amp ! (af)
                     x3b_vvoo(a,b,:,n) = x3b_vvoo(a,b,:,n) - h2a_oovv(:,k,e,f)*r_amp ! (kn)
                     x3b_vvoo(e,b,:,n) = x3b_vvoo(e,b,:,n) + h2a_oovv(:,k,a,f)*r_amp ! (ae)(kn)
                     x3b_vvoo(f,b,:,n) = x3b_vvoo(f,b,:,n) + h2a_oovv(:,k,e,a)*r_amp ! (af)(kn)
                  end do
                  do idet=1,n4abab
                     r_amp = r4c_amps(idet)
                     ! x3b(a,b,m,k) <- A(ae)A(bf) h2b(mnef)*r4c(abefkn)
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); e = r4c_excits(idet,3); f = r4c_excits(idet,4);
                     k = r4c_excits(idet,5); n = r4c_excits(idet,6);
                     x3b_vvoo(a,b,:,k) = x3b_vvoo(a,b,:,k) + h2b_oovv(:,n,e,f)*r_amp ! (1)
                     x3b_vvoo(e,b,:,k) = x3b_vvoo(e,b,:,k) - h2b_oovv(:,n,a,f)*r_amp ! (ae)
                     x3b_vvoo(a,f,:,k) = x3b_vvoo(a,f,:,k) - h2b_oovv(:,n,e,b)*r_amp ! (bf)
                     x3b_vvoo(e,f,:,k) = x3b_vvoo(e,f,:,k) + h2b_oovv(:,n,a,b)*r_amp ! (ae)(bf)
                  end do
              end subroutine add_r4_x3b_vvoo

              subroutine add_r4_x3b_vvvv(x3b_vvvv,&
                                         r4b_amps, r4b_excits,&
                                         r4c_amps, r4c_excits,&
                                         h2a_oovv, h2b_oovv,&
                                         n4abaa, n4abab,&
                                         noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n4abaa, n4abab

                  integer, intent(in) :: r4b_excits(n4abaa,6)
                  real(kind=8), intent(in) :: r4b_amps(n4abaa)
                  integer, intent(in) :: r4c_excits(n4abab,6)
                  real(kind=8), intent(in) :: r4c_amps(n4abab)

                  real(kind=8), intent(in) :: h2a_oovv(noa,noa,nua,nua)
                  real(kind=8), intent(in) :: h2b_oovv(noa,nob,nua,nub)

                  real(kind=8), intent(inout) :: x3b_vvvv(nua,nub,nua,nua)
                  !f2py intent(in,out) :: x3b_vvvv(0:nua-1,0:nub-1,0:nua-1,0:nua-1)

                  real(kind=8) :: r_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  ! divide incoming antisymmetrized quantity by half to cancel redundant antisymmetrization
                  ! later on; be sure to only update permutationally unique elements!                  
                  x3b_vvvv = 0.5d0*x3b_vvvv
                  do idet=1,n4abaa
                     r_amp = r4b_amps(idet)
                     ! x3b(a,b,c,e) <- -A(f/ac) h2a(mnef)*r4b(abcfmn)
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); f = r4b_excits(idet,4);
                     m = r4b_excits(idet,5); n = r4b_excits(idet,6);
                     x3b_vvvv(a,b,c,:) = x3b_vvvv(a,b,c,:) - h2a_oovv(m,n,:,f)*r_amp ! (1)
                     x3b_vvvv(c,b,f,:) = x3b_vvvv(c,b,f,:) - h2a_oovv(m,n,:,a)*r_amp ! (af)
                     x3b_vvvv(a,b,f,:) = x3b_vvvv(a,b,f,:) + h2a_oovv(m,n,:,c)*r_amp ! (cf)
                  end do
                  do idet=1,n4abab
                     r_amp = r4c_amps(idet)
                     ! x3b(a,b,c,e) <- A(bf) -h2b(mnef)*r4c(abcfmn)
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); f = r4c_excits(idet,4);
                     m = r4c_excits(idet,5); n = r4c_excits(idet,6);
                     x3b_vvvv(a,b,c,:) = x3b_vvvv(a,b,c,:) - h2b_oovv(m,n,:,f)*r_amp ! (1)
                     x3b_vvvv(a,f,c,:) = x3b_vvvv(a,f,c,:) + h2b_oovv(m,n,:,b)*r_amp ! (bf)
                  end do
                  ! apply the common A(ac) antisymmetrizer
                  do e = 1,nua
                     do b = 1,nub
                        do a = 1,nua
                           do c = a+1,nua
                              x3b_vvvv(a,b,c,e) = x3b_vvvv(a,b,c,e) - x3b_vvvv(c,b,a,e)
                           end do
                        end do
                     end do
                  end do
                  ! explicitly antisymmetrize
                  do a = 1,nua
                     do c = a+1,nua
                        x3b_vvvv(c,:,a,:) = -x3b_vvvv(a,:,c,:)
                     end do
                  end do
              end subroutine add_r4_x3b_vvvv

              subroutine add_r4_x3b_vovo(x3b_vovo,&
                                         r4b_amps, r4b_excits,&
                                         r4c_amps, r4c_excits,&
                                         h2b_oovv, h2c_oovv,&
                                         n4abaa, n4abab,&
                                         noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n4abaa, n4abab

                  integer, intent(in) :: r4b_excits(n4abaa,6)
                  real(kind=8), intent(in) :: r4b_amps(n4abaa)
                  integer, intent(in) :: r4c_excits(n4abab,6)
                  real(kind=8), intent(in) :: r4c_amps(n4abab)

                  real(kind=8), intent(in) :: h2b_oovv(noa,nob,nua,nub)
                  real(kind=8), intent(in) :: h2c_oovv(nob,nob,nub,nub)

                  real(kind=8), intent(inout) :: x3b_vovo(nua,nob,nua,noa)
                  !f2py intent(in,out) :: x3b_vovo(0:nua-1,0:nob-1,0:nua-1,0:noa-1)

                  real(kind=8) :: r_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  ! divide incoming antisymmetrized quantity by half to cancel redundant antisymmetrization
                  ! later on; be sure to only update permutationally unique elements!                  
                  x3b_vovo = 0.5d0*x3b_vovo
                  do idet=1,n4abaa
                     r_amp = r4b_amps(idet)
                     ! x3b(a,m,c,k) <- A(f/ac)A(kn) h2b(nmfe)*r4b(aecfkn)
                     a = r4b_excits(idet,1); e = r4b_excits(idet,2); c = r4b_excits(idet,3); f = r4b_excits(idet,4);
                     k = r4b_excits(idet,5); n = r4b_excits(idet,6);
                     x3b_vovo(a,:,c,k) = x3b_vovo(a,:,c,k) + h2b_oovv(n,:,f,e)*r_amp ! (1)
                     x3b_vovo(c,:,f,k) = x3b_vovo(c,:,f,k) + h2b_oovv(n,:,a,e)*r_amp ! (af)
                     x3b_vovo(a,:,f,k) = x3b_vovo(a,:,f,k) - h2b_oovv(n,:,c,e)*r_amp ! (cf)
                     x3b_vovo(a,:,c,n) = x3b_vovo(a,:,c,n) - h2b_oovv(k,:,f,e)*r_amp ! (kn)
                     x3b_vovo(c,:,f,n) = x3b_vovo(c,:,f,n) - h2b_oovv(k,:,a,e)*r_amp ! (af)(kn)
                     x3b_vovo(a,:,f,n) = x3b_vovo(a,:,f,n) + h2b_oovv(k,:,c,e)*r_amp ! (cf)(kn)
                  end do
                  do idet=1,n4abab
                     r_amp = r4c_amps(idet)
                     ! x3b(a,m,c,k) <- h2c(mnef)*r4c(aecfkn)
                     a = r4c_excits(idet,1); e = r4c_excits(idet,2); c = r4c_excits(idet,3); f = r4c_excits(idet,4);
                     k = r4c_excits(idet,5); n = r4c_excits(idet,6);
                     x3b_vovo(a,:,c,k) = x3b_vovo(a,:,c,k) + h2c_oovv(:,n,e,f)*r_amp ! (1)
                  end do
                  ! apply the common A(ac) antisymmetrizer
                  do m = 1,nob
                     do k = 1,noa
                        do a = 1,nua
                           do c = a+1,nua
                              x3b_vovo(a,m,c,k) = x3b_vovo(a,m,c,k) - x3b_vovo(c,m,a,k)
                           end do
                        end do
                     end do
                  end do
                  ! explicitly antisymmetrize
                  do a = 1,nua
                     do c = a+1,nua
                        x3b_vovo(c,:,a,:) = -x3b_vovo(a,:,c,:)
                     end do
                  end do
              end subroutine add_r4_x3b_vovo

              subroutine add_r4_x3c_vvoo(x3c_vvoo,&
                                         r4c_amps, r4c_excits,&
                                         r4d_amps, r4d_excits,&
                                         h2b_oovv, h2c_oovv,&
                                         n4abab, n4abbb,&
                                         noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n4abab, n4abbb

                  integer, intent(in) :: r4c_excits(n4abab,6)
                  real(kind=8), intent(in) :: r4c_amps(n4abab)
                  integer, intent(in) :: r4d_excits(n4abbb,6)
                  real(kind=8), intent(in) :: r4d_amps(n4abbb)

                  real(kind=8), intent(in) :: h2b_oovv(noa,nob,nua,nub)
                  real(kind=8), intent(in) :: h2c_oovv(nob,nob,nub,nub)

                  real(kind=8), intent(inout) :: x3c_vvoo(nua,nub,nob,nob)
                  !f2py intent(in,out) :: x3c_vvoo(0:nua-1,0:nub-1,0:nob-1,0:nob-1)

                  real(kind=8) :: r_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  do idet=1,n4abbb
                     r_amp = r4d_amps(idet)
                     ! x3c(a,b,m,k) <- A(b/ef)A(kn) h2c(mnef)*r4d(abefkn)
                     a = r4d_excits(idet,1); b = r4d_excits(idet,2); e = r4d_excits(idet,3); f = r4d_excits(idet,4);
                     k = r4d_excits(idet,5); n = r4d_excits(idet,6);
                     x3c_vvoo(a,b,:,k) = x3c_vvoo(a,b,:,k) + h2c_oovv(:,n,e,f)*r_amp ! (1)
                     x3c_vvoo(a,e,:,k) = x3c_vvoo(a,e,:,k) - h2c_oovv(:,n,b,f)*r_amp ! (be)
                     x3c_vvoo(a,f,:,k) = x3c_vvoo(a,f,:,k) - h2c_oovv(:,n,e,b)*r_amp ! (bf)
                     x3c_vvoo(a,b,:,n) = x3c_vvoo(a,b,:,n) - h2c_oovv(:,k,e,f)*r_amp ! (kn)
                     x3c_vvoo(a,e,:,n) = x3c_vvoo(a,e,:,n) + h2c_oovv(:,k,b,f)*r_amp ! (be)(kn)
                     x3c_vvoo(a,f,:,n) = x3c_vvoo(a,f,:,n) + h2c_oovv(:,k,e,b)*r_amp ! (bf)(kn)
                  end do
                  do idet=1,n4abab
                     r_amp = r4c_amps(idet)
                     ! x3c(a,b,m,k) <- A(af)A(be) h2b(nmfe)*r4c(abfenk)
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); f = r4c_excits(idet,3); e = r4c_excits(idet,4);
                     n = r4c_excits(idet,5); k = r4c_excits(idet,6);
                     x3c_vvoo(a,b,:,k) = x3c_vvoo(a,b,:,k) + h2b_oovv(n,:,f,e)*r_amp ! (1)
                     x3c_vvoo(f,b,:,k) = x3c_vvoo(f,b,:,k) - h2b_oovv(n,:,a,e)*r_amp ! (af)
                     x3c_vvoo(a,e,:,k) = x3c_vvoo(a,e,:,k) - h2b_oovv(n,:,f,b)*r_amp ! (be)
                     x3c_vvoo(f,e,:,k) = x3c_vvoo(f,e,:,k) + h2b_oovv(n,:,a,b)*r_amp ! (af)(be)
                  end do
              end subroutine add_r4_x3c_vvoo

              subroutine add_r4_x3c_vvvv(x3c_vvvv,&
                                         r4c_amps, r4c_excits,&
                                         r4d_amps, r4d_excits,&
                                         h2b_oovv, h2c_oovv,&
                                         n4abab, n4abbb,&
                                         noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n4abab, n4abbb

                  integer, intent(in) :: r4c_excits(n4abab,6)
                  real(kind=8), intent(in) :: r4c_amps(n4abab)
                  integer, intent(in) :: r4d_excits(n4abbb,6)
                  real(kind=8), intent(in) :: r4d_amps(n4abbb)

                  real(kind=8), intent(in) :: h2b_oovv(noa,nob,nua,nub)
                  real(kind=8), intent(in) :: h2c_oovv(nob,nob,nub,nub)

                  real(kind=8), intent(inout) :: x3c_vvvv(nua,nub,nub,nub)
                  !f2py intent(in,out) :: x3c_vvvv(0:nua-1,0:nub-1,0:nub-1,0:nub-1)

                  real(kind=8) :: r_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  ! divide incoming antisymmetrized quantity by half to cancel redundant antisymmetrization
                  ! later on; be sure to only update permutationally unique elements!                  
                  x3c_vvvv = 0.5d0*x3c_vvvv
                  do idet=1,n4abbb
                     r_amp = r4d_amps(idet)
                     ! x3c(a,b,c,e) <- -A(f/bc) h2c(mnef)*r4d(abcfmn)
                     a = r4d_excits(idet,1); b = r4d_excits(idet,2); c = r4d_excits(idet,3); f = r4d_excits(idet,4);
                     m = r4d_excits(idet,5); n = r4d_excits(idet,6);
                     x3c_vvvv(a,b,c,:) = x3c_vvvv(a,b,c,:) - h2c_oovv(m,n,:,f)*r_amp ! (1)
                     x3c_vvvv(a,c,f,:) = x3c_vvvv(a,c,f,:) - h2c_oovv(m,n,:,b)*r_amp ! (bf)
                     x3c_vvvv(a,b,f,:) = x3c_vvvv(a,b,f,:) + h2c_oovv(m,n,:,c)*r_amp ! (cf)
                  end do
                  do idet=1,n4abab
                     r_amp = r4c_amps(idet)
                     ! x3c(a,b,c,e) <- A(af) -h2b(nmfe)*r4c(abfcnm)
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); f = r4c_excits(idet,3); c = r4c_excits(idet,4);
                     n = r4c_excits(idet,5); m = r4c_excits(idet,6);
                     x3c_vvvv(a,b,c,:) = x3c_vvvv(a,b,c,:) - h2b_oovv(n,m,f,:)*r_amp ! (1)
                     x3c_vvvv(f,b,c,:) = x3c_vvvv(f,b,c,:) + h2b_oovv(n,m,a,:)*r_amp ! (af)
                  end do
                  ! apply the common A(bc) antisymmetrizer
                  do e = 1,nub
                     do b = 1,nub
                        do a = 1,nua
                           do c = b+1,nub
                              x3c_vvvv(a,b,c,e) = x3c_vvvv(a,b,c,e) - x3c_vvvv(a,c,b,e)
                           end do
                        end do
                     end do
                  end do
                  ! explicitly antisymmetrize
                  do b = 1,nub
                     do c = b+1,nub
                        x3c_vvvv(:,c,b,:) = -x3c_vvvv(:,b,c,:)
                     end do
                  end do
              end subroutine add_r4_x3c_vvvv

              subroutine add_r4_x3c_ovvo(x3c_ovvo,&
                                         r4c_amps, r4c_excits,&
                                         r4d_amps, r4d_excits,&
                                         h2a_oovv, h2b_oovv,&
                                         n4abab, n4abbb,&
                                         noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n4abab, n4abbb

                  integer, intent(in) :: r4c_excits(n4abab,6)
                  real(kind=8), intent(in) :: r4c_amps(n4abab)
                  integer, intent(in) :: r4d_excits(n4abbb,6)
                  real(kind=8), intent(in) :: r4d_amps(n4abbb)

                  real(kind=8), intent(in) :: h2a_oovv(noa,noa,nua,nua)
                  real(kind=8), intent(in) :: h2b_oovv(noa,nob,nua,nub)

                  real(kind=8), intent(inout) :: x3c_ovvo(noa,nub,nub,nob)
                  !f2py intent(in,out) :: x3c_ovvo(0:noa-1,0:nub-1,0:nub-1,0:nob-1)

                  real(kind=8) :: r_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  ! divide incoming antisymmetrized quantity by half to cancel redundant antisymmetrization
                  ! later on; be sure to only update permutationally unique elements!                  
                  x3c_ovvo = 0.5d0*x3c_ovvo
                  do idet=1,n4abbb
                     r_amp = r4d_amps(idet)
                     ! x3c(m,b,d,l) <- A(f/bd)A(nl) h2b(mnef)*r4d(ebfdnl) 
                     e = r4d_excits(idet,1); b = r4d_excits(idet,2); f = r4d_excits(idet,3); d = r4d_excits(idet,4);
                     n = r4d_excits(idet,5); l = r4d_excits(idet,6);
                     x3c_ovvo(:,b,d,l) = x3c_ovvo(:,b,d,l) + h2b_oovv(:,n,e,f)*r_amp ! (1)
                     x3c_ovvo(:,f,d,l) = x3c_ovvo(:,f,d,l) - h2b_oovv(:,n,e,b)*r_amp ! (bf)
                     x3c_ovvo(:,b,f,l) = x3c_ovvo(:,b,f,l) - h2b_oovv(:,n,e,d)*r_amp ! (df)
                     x3c_ovvo(:,b,d,n) = x3c_ovvo(:,b,d,n) - h2b_oovv(:,l,e,f)*r_amp ! (nl)
                     x3c_ovvo(:,f,d,n) = x3c_ovvo(:,f,d,n) + h2b_oovv(:,l,e,b)*r_amp ! (bf)(nl)
                     x3c_ovvo(:,b,f,n) = x3c_ovvo(:,b,f,n) + h2b_oovv(:,l,e,d)*r_amp ! (df)(nl)
                  end do
                  do idet=1,n4abab
                     r_amp = r4c_amps(idet)
                     ! x3c(m,b,d,l) <- h2a(mnef)*r4b(ebfdnl) 
                     e = r4c_excits(idet,1); b = r4c_excits(idet,2); f = r4c_excits(idet,3); d = r4c_excits(idet,4);
                     n = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     x3c_ovvo(:,b,d,l) = x3c_ovvo(:,b,d,l) + h2a_oovv(:,n,e,f)*r_amp ! (1)
                  end do
                  ! apply the common A(bd) antisymmetrizer
                  do m = 1,noa
                     do l = 1,nob
                        do b = 1,nub
                           do d = b+1,nub
                              x3c_ovvo(m,b,d,l) = x3c_ovvo(m,b,d,l) - x3c_ovvo(m,d,b,l)
                           end do
                        end do
                     end do
                  end do
                  ! explicitly antisymmetrize
                  do b = 1,nub
                     do d = b+1,nub
                        x3c_ovvo(:,d,b,:) = -x3c_ovvo(:,b,d,:)
                     end do
                  end do
              end subroutine add_r4_x3c_ovvo

end module deaeom4_p_intermediates
