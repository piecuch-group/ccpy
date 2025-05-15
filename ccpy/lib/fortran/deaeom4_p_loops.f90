module deaeom4_p_loops

      use omp_lib

      implicit none

      contains

              subroutine build_hr_2b(sigma_2b,&
                                     r4b_amps,r4b_excits,&
                                     r4c_amps,r4c_excits,&
                                     r4d_amps,r4d_excits,&
                                     h2a_oovv,h2b_oovv,h2c_oovv,&
                                     n4abaa,n4abab,n4abbb,&
                                     noa,nua,nob,nub)

                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n4abaa, n4abab, n4abbb
                  ! Input R and T arrays
                  integer, intent(in) :: r4b_excits(n4abaa,6)
                  integer, intent(in) :: r4c_excits(n4abab,6)
                  integer, intent(in) :: r4d_excits(n4abbb,6)
                  real(kind=8), intent(in) :: r4b_amps(n4abaa)
                  real(kind=8), intent(in) :: r4c_amps(n4abab)
                  real(kind=8), intent(in) :: r4d_amps(n4abbb)
                  ! Input H  and X arrays
                  real(kind=8), intent(in) :: h2a_oovv(noa,noa,nua,nua)
                  real(kind=8), intent(in) :: h2b_oovv(noa,nob,nua,nub)
                  real(kind=8), intent(in) :: h2c_oovv(nob,nob,nub,nub)
                  ! Output and Inout variables
                  real(kind=8), intent(inout) :: sigma_2b(nua,nub)
                  !f2py intent(in,out) :: sigma_2b(0:nua-1,0:nub-1)
                  ! Local variables
                  real(kind=8) :: r_amp, val
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  do idet = 1,n4abaa
                     r_amp = r4b_amps(idet)
                     ! x2b(a,b) <- A(a/ef) v(mnef)*r4b(ab~efmn)
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); e = r4b_excits(idet,3); f = r4b_excits(idet,4)
                     m = r4b_excits(idet,5); n = r4b_excits(idet,6);
                     sigma_2b(a,b) = sigma_2b(a,b) + h2a_oovv(m,n,e,f)*r_amp ! (1)
                     sigma_2b(e,b) = sigma_2b(e,b) - h2a_oovv(m,n,a,f)*r_amp ! (ae)
                     sigma_2b(f,b) = sigma_2b(f,b) - h2a_oovv(m,n,e,a)*r_amp ! (af)
                  end do
                  do idet = 1,n4abab
                     r_amp = r4c_amps(idet)
                     ! x2b(a,b) <- A(ae)A(bf) v(mn~ef~)*r4c(ab~ef~mn~)
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); e = r4c_excits(idet,3); f = r4c_excits(idet,4)
                     m = r4c_excits(idet,5); n = r4c_excits(idet,6);
                     sigma_2b(a,b) = sigma_2b(a,b) + h2b_oovv(m,n,e,f)*r_amp ! (1)
                     sigma_2b(e,b) = sigma_2b(e,b) - h2b_oovv(m,n,a,f)*r_amp ! (ae)
                     sigma_2b(a,f) = sigma_2b(a,f) - h2b_oovv(m,n,e,b)*r_amp ! (bf)
                     sigma_2b(e,f) = sigma_2b(e,f) + h2b_oovv(m,n,a,b)*r_amp ! (ae)(bf)
                  end do
                  do idet = 1,n4abbb
                     r_amp = r4d_amps(idet)
                     ! x2b(a,b) <- A(b/ef) v(m~n~e~f~)*r4d(ab~e~f~m~n~)
                     a = r4d_excits(idet,1); b = r4d_excits(idet,2); e = r4d_excits(idet,3); f = r4d_excits(idet,4)
                     m = r4d_excits(idet,5); n = r4d_excits(idet,6);
                     sigma_2b(a,b) = sigma_2b(a,b) + h2c_oovv(m,n,e,f)*r_amp ! (1)
                     sigma_2b(a,e) = sigma_2b(a,e) - h2c_oovv(m,n,b,f)*r_amp ! (be)
                     sigma_2b(a,f) = sigma_2b(a,f) - h2c_oovv(m,n,e,b)*r_amp ! (bf)
                  end do

              end subroutine build_hr_2b

              subroutine build_hr_3b(sigma_3b,&
                                     r4b_amps,r4b_excits,&
                                     r4c_amps,r4c_excits,&
                                     h1a_ov,h1b_ov,&
                                     h2a_ooov,h2a_vovv,&
                                     h2b_ooov,h2b_vovv,h2b_ovvv,&
                                     h2c_vovv,&
                                     n4abaa,n4abab,&
                                     noa,nua,nob,nub)

                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n4abaa, n4abab
                  ! Input R and T arrays
                  integer, intent(in) :: r4b_excits(n4abaa,6)
                  integer, intent(in) :: r4c_excits(n4abab,6)
                  real(kind=8), intent(in) :: r4b_amps(n4abaa)
                  real(kind=8), intent(in) :: r4c_amps(n4abab)
                  ! Input H and X arrays
                  real(kind=8), intent(in) :: h1a_ov(noa,nua)
                  real(kind=8), intent(in) :: h1b_ov(nob,nub)
                  real(kind=8), intent(in) :: h2a_ooov(noa,noa,noa,nua)
                  real(kind=8), intent(in) :: h2a_vovv(nua,noa,nua,nua)
                  real(kind=8), intent(in) :: h2b_ooov(noa,nob,noa,nub)
                  real(kind=8), intent(in) :: h2b_vovv(nua,nob,nua,nub)
                  real(kind=8), intent(in) :: h2b_ovvv(noa,nub,nua,nub)
                  real(kind=8), intent(in) :: h2c_vovv(nub,nob,nub,nub)
                  ! Output and Inout variables
                  real(kind=8), intent(inout) :: sigma_3b(nua,nub,nua,noa)
                  !f2py intent(in,out) :: sigma_3b(0:nua-1,0:nub-1,0:nua-1,0:noa-1)
                  ! Local variables
                  real(kind=8) :: r_amp, val
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  do idet=1,n4abaa
                     r_amp = r4b_amps(idet)
                     ! x3b(a,b,c,k) <- A(ac)[ A(e/ac)A(km) h1a(me)*r4b(ab~cekm) ]
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); e = r4b_excits(idet,4);
                     k = r4b_excits(idet,5); m = r4b_excits(idet,6);
                     sigma_3b(a,b,c,k) = sigma_3b(a,b,c,k) + h1a_ov(m,e)*r_amp ! (1)
                     sigma_3b(c,b,e,k) = sigma_3b(c,b,e,k) + h1a_ov(m,a)*r_amp ! (ae)
                     sigma_3b(a,b,e,k) = sigma_3b(a,b,e,k) - h1a_ov(m,c)*r_amp ! (ce)
                     sigma_3b(a,b,c,m) = sigma_3b(a,b,c,m) - h1a_ov(k,e)*r_amp ! (km)
                     sigma_3b(c,b,e,m) = sigma_3b(c,b,e,m) - h1a_ov(k,a)*r_amp ! (ae)(km)
                     sigma_3b(a,b,e,m) = sigma_3b(a,b,e,m) + h1a_ov(k,c)*r_amp ! (ce)(km)
                     ! x3b(a,b,c,k) <- A(ac)[ A(f/ac) -h2a(mnkf)*r4b(ab~cfmn)]
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); f = r4b_excits(idet,4);
                     m = r4b_excits(idet,5); n = r4b_excits(idet,6);
                     sigma_3b(a,b,c,:) = sigma_3b(a,b,c,:) - h2a_ooov(m,n,:,f)*r_amp ! (1)
                     sigma_3b(c,b,f,:) = sigma_3b(c,b,f,:) - h2a_ooov(m,n,:,a)*r_amp ! (af)
                     sigma_3b(a,b,f,:) = sigma_3b(a,b,f,:) + h2a_ooov(m,n,:,c)*r_amp ! (cf)
                     ! x3b(a,b,c,k) <- A(ac)[ A(a/ef)A(kn) h2a(cnef)*r4b(ab~efkn) ]
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); e = r4b_excits(idet,3); f = r4b_excits(idet,4);
                     k = r4b_excits(idet,5); n = r4b_excits(idet,6);
                     sigma_3b(a,b,:,k) = sigma_3b(a,b,:,k) + h2a_vovv(:,n,e,f)*r_amp ! (1)
                     sigma_3b(e,b,:,k) = sigma_3b(e,b,:,k) - h2a_vovv(:,n,a,f)*r_amp ! (ae)
                     sigma_3b(f,b,:,k) = sigma_3b(f,b,:,k) - h2a_vovv(:,n,e,a)*r_amp ! (af)
                     sigma_3b(a,b,:,n) = sigma_3b(a,b,:,n) - h2a_vovv(:,k,e,f)*r_amp ! (kn)
                     sigma_3b(e,b,:,n) = sigma_3b(e,b,:,n) + h2a_vovv(:,k,a,f)*r_amp ! (ae)(kn)
                     sigma_3b(f,b,:,n) = sigma_3b(f,b,:,n) + h2a_vovv(:,k,e,a)*r_amp ! (af)(kn)
                     ! x3b(a,b,c,k) <- A(ac)[ A(f/ac)A(kn) h2b(nbfe)*r4b(ae~cfkn) ]
                     a = r4b_excits(idet,1); e = r4b_excits(idet,2); c = r4b_excits(idet,3); f = r4b_excits(idet,4);
                     k = r4b_excits(idet,5); n = r4b_excits(idet,6);
                     sigma_3b(a,:,c,k) = sigma_3b(a,:,c,k) + h2b_ovvv(n,:,f,e)*r_amp ! (1)
                     sigma_3b(c,:,f,k) = sigma_3b(c,:,f,k) + h2b_ovvv(n,:,a,e)*r_amp ! (af)
                     sigma_3b(a,:,f,k) = sigma_3b(a,:,f,k) - h2b_ovvv(n,:,c,e)*r_amp ! (cf)
                     sigma_3b(a,:,c,n) = sigma_3b(a,:,c,n) - h2b_ovvv(k,:,f,e)*r_amp ! (kn)
                     sigma_3b(c,:,f,n) = sigma_3b(c,:,f,n) - h2b_ovvv(k,:,a,e)*r_amp ! (af)(kn)
                     sigma_3b(a,:,f,n) = sigma_3b(a,:,f,n) + h2b_ovvv(k,:,c,e)*r_amp ! (cf)(kn)
                  end do
                  do idet=1,n4abab
                     r_amp = r4c_amps(idet)
                     ! x3b(a,b,c,k) <- A(ac)[ A(be) h1b(me)*r4c(ab~ce~km~) ] 
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); e = r4c_excits(idet,4);
                     k = r4c_excits(idet,5); m = r4c_excits(idet,6);
                     sigma_3b(a,b,c,k) = sigma_3b(a,b,c,k) + h1b_ov(m,e)*r_amp ! (1)
                     sigma_3b(a,e,c,k) = sigma_3b(a,e,c,k) - h1b_ov(m,b)*r_amp ! (be)
                     ! x3b(a,b,c,k) <- A(ac)[ -A(bf) h2b(mnkf)*r4c(ab~cf~mn~) ]
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); f = r4c_excits(idet,4);
                     m = r4c_excits(idet,5); n = r4c_excits(idet,6);
                     sigma_3b(a,b,c,:) = sigma_3b(a,b,c,:) - h2b_ooov(m,n,:,f)*r_amp ! (1) 
                     sigma_3b(a,f,c,:) = sigma_3b(a,f,c,:) + h2b_ooov(m,n,:,b)*r_amp ! (bf) 
                     ! x3b(a,b,c,k) <- A(ae)A(bf) h2b(cnef)*r4c(ab~ef~kn~)
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); e = r4c_excits(idet,3); f = r4c_excits(idet,4);
                     k = r4c_excits(idet,5); n = r4c_excits(idet,6);
                     sigma_3b(a,b,:,k) = sigma_3b(a,b,:,k) + h2b_vovv(:,n,e,f)*r_amp ! (1)
                     sigma_3b(e,b,:,k) = sigma_3b(e,b,:,k) - h2b_vovv(:,n,a,f)*r_amp ! (ae)
                     sigma_3b(a,f,:,k) = sigma_3b(a,f,:,k) - h2b_vovv(:,n,e,b)*r_amp ! (bf)
                     sigma_3b(e,f,:,k) = sigma_3b(e,f,:,k) + h2b_vovv(:,n,a,b)*r_amp ! (ae)(bf)
                     ! x3b(a,b,c,k) <- A(ac)[ h2c(bnef)*r4c(ae~cf~kn~) ]
                     a = r4c_excits(idet,1); e = r4c_excits(idet,2); c = r4c_excits(idet,3); f = r4c_excits(idet,4);
                     k = r4c_excits(idet,5); n = r4c_excits(idet,6);
                     sigma_3b(a,:,c,k) = sigma_3b(a,:,c,k) + h2c_vovv(:,n,e,f)*r_amp ! (1)
                  end do

                  ! antisymmetrize (this replaces the x2a -= np.transpose(x2a, (...)) stuff in vector update
                  do k = 1, noa
                     do a = 1, nua
                        do b = 1, nub
                           do c = a+1, nua
                              val = sigma_3b(a,b,c,k) - sigma_3b(c,b,a,k)
                              sigma_3b(a,b,c,k) =  val
                              sigma_3b(c,b,a,k) = -val
                           end do
                        end do
                     end do
                  end do
                  ! (H(2) * T3)_C terms are vectorized and generally broadcast to diagonal elements, which should
                  ! be 0. Set them to 0 manually (you need to do this).
                  do a = 1,nua
                     sigma_3b(a,:,a,:) = 0.0d0
                  end do
              end subroutine build_hr_3b

              subroutine build_hr_3c(sigma_3c,&
                                     r4c_amps,r4c_excits,&
                                     r4d_amps,r4d_excits,&
                                     h1a_ov,h1b_ov,&
                                     h2a_vovv,&
                                     h2b_oovo,h2b_vovv,h2b_ovvv,&
                                     h2c_ooov,h2c_vovv,&
                                     n4abab,n4abbb,&
                                     noa,nua,nob,nub)

                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n4abab, n4abbb
                  ! Input R and T arrays
                  integer, intent(in) :: r4c_excits(n4abab,6)
                  integer, intent(in) :: r4d_excits(n4abbb,6)
                  real(kind=8), intent(in) :: r4c_amps(n4abab)
                  real(kind=8), intent(in) :: r4d_amps(n4abbb)
                  ! Input H and X arrays
                  real(kind=8), intent(in) :: h1a_ov(noa,nua)
                  real(kind=8), intent(in) :: h1b_ov(nob,nub)
                  real(kind=8), intent(in) :: h2a_vovv(nua,noa,nua,nua)
                  real(kind=8), intent(in) :: h2b_oovo(noa,nob,nua,nob)
                  real(kind=8), intent(in) :: h2b_vovv(nua,nob,nua,nub)
                  real(kind=8), intent(in) :: h2b_ovvv(noa,nub,nua,nub)
                  real(kind=8), intent(in) :: h2c_ooov(nob,nob,nob,nub)
                  real(kind=8), intent(in) :: h2c_vovv(nub,nob,nub,nub)
                  ! Output and Inout variables
                  real(kind=8), intent(inout) :: sigma_3c(nua,nub,nub,nob)
                  !f2py intent(in,out) :: sigma_3c(0:nua-1,0:nub-1,0:nub-1,0:nob-1)
                  ! Local variables
                  real(kind=8) :: r_amp, val
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  do idet=1,n4abab
                     r_amp = r4c_amps(idet)
                     ! x3c(a,b,c,k) <- A(bc)[ A(ae) h1a(me)*r4c(abecmk)
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); e = r4c_excits(idet,3); c = r4c_excits(idet,4);
                     m = r4c_excits(idet,5); k = r4c_excits(idet,6);
                     sigma_3c(a,b,c,k) = sigma_3c(a,b,c,k) + h1a_ov(m,e)*r_amp ! (1)
                     sigma_3c(e,b,c,k) = sigma_3c(e,b,c,k) - h1a_ov(m,a)*r_amp ! (ae)
                     ! x3c(a,b,c,k) <- A(bc)[ -A(af) h2b(nmfk)*r4c(abfcnm) ]
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); f = r4c_excits(idet,3); c = r4c_excits(idet,4);
                     n = r4c_excits(idet,5); m = r4c_excits(idet,6);
                     sigma_3c(a,b,c,:) = sigma_3c(a,b,c,:) - h2b_oovo(n,m,f,:)*r_amp ! (1)
                     sigma_3c(f,b,c,:) = sigma_3c(f,b,c,:) + h2b_oovo(n,m,a,:)*r_amp ! (af)
                     ! x3c(a,b,c,k) <- A(af)A(be) h2b(ncfe)*r4c(abfenk)
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); f = r4c_excits(idet,3); e = r4c_excits(idet,4);
                     n = r4c_excits(idet,5); k = r4c_excits(idet,6);
                     sigma_3c(a,b,:,k) = sigma_3c(a,b,:,k) + h2b_ovvv(n,:,f,e)*r_amp ! (1)
                     sigma_3c(f,b,:,k) = sigma_3c(f,b,:,k) - h2b_ovvv(n,:,a,e)*r_amp ! (af)
                     sigma_3c(a,e,:,k) = sigma_3c(a,e,:,k) - h2b_ovvv(n,:,f,b)*r_amp ! (be)
                     sigma_3c(f,e,:,k) = sigma_3c(f,e,:,k) + h2b_ovvv(n,:,a,b)*r_amp ! (af)(be)
                     ! x3c(a,b,c,k) <- A(bc)[ h2a(anef)*r4c(ebfcnk) ]
                     e = r4c_excits(idet,1); b = r4c_excits(idet,2); f = r4c_excits(idet,3); c = r4c_excits(idet,4);
                     n = r4c_excits(idet,5); k = r4c_excits(idet,6);
                     sigma_3c(:,b,c,k) = sigma_3c(:,b,c,k) + h2a_vovv(:,n,e,f)*r_amp ! (1)
                  end do
                  do idet=1,n4abbb
                     r_amp = r4d_amps(idet)
                     ! x3c(a,b,c,k) <- A(e/bc)A(mk) h1b(me)*r4d(abecmk)
                     a = r4d_excits(idet,1); b = r4d_excits(idet,2); e = r4d_excits(idet,3); c = r4d_excits(idet,4);
                     m = r4d_excits(idet,5); k = r4d_excits(idet,6);
                     sigma_3c(a,b,c,k) = sigma_3c(a,b,c,k) + h1b_ov(m,e)*r_amp ! (1)
                     sigma_3c(a,e,c,k) = sigma_3c(a,e,c,k) - h1b_ov(m,b)*r_amp ! (be)
                     sigma_3c(a,b,e,k) = sigma_3c(a,b,e,k) - h1b_ov(m,c)*r_amp ! (ce)
                     sigma_3c(a,b,c,m) = sigma_3c(a,b,c,m) - h1b_ov(k,e)*r_amp ! (mk)
                     sigma_3c(a,e,c,m) = sigma_3c(a,e,c,m) + h1b_ov(k,b)*r_amp ! (be)(mk)
                     sigma_3c(a,b,e,m) = sigma_3c(a,b,e,m) + h1b_ov(k,c)*r_amp ! (ce)(mk)
                     ! x3c(a,b,c,k) <- -A(f/bc) h2c(mnkf)*r4d(abcfmn)
                     a = r4d_excits(idet,1); b = r4d_excits(idet,2); c = r4d_excits(idet,3); f = r4d_excits(idet,4);
                     m = r4d_excits(idet,5); n = r4d_excits(idet,6);
                     sigma_3c(a,b,c,:) = sigma_3c(a,b,c,:) - h2c_ooov(m,n,:,f)*r_amp ! (1)
                     sigma_3c(a,c,f,:) = sigma_3c(a,c,f,:) - h2c_ooov(m,n,:,b)*r_amp ! (bf)
                     sigma_3c(a,b,f,:) = sigma_3c(a,b,f,:) + h2c_ooov(m,n,:,c)*r_amp ! (cf)
                     ! x3c(a,b,c,k) <- A(b/ef)A(kn) h2c(cnef)*r4d(abefkn)
                     a = r4d_excits(idet,1); b = r4d_excits(idet,2); e = r4d_excits(idet,3); f = r4d_excits(idet,4);
                     k = r4d_excits(idet,5); n = r4d_excits(idet,6);
                     sigma_3c(a,b,:,k) = sigma_3c(a,b,:,k) + h2c_vovv(:,n,e,f)*r_amp ! (1)
                     sigma_3c(a,e,:,k) = sigma_3c(a,e,:,k) - h2c_vovv(:,n,b,f)*r_amp ! (be)
                     sigma_3c(a,f,:,k) = sigma_3c(a,f,:,k) - h2c_vovv(:,n,e,b)*r_amp ! (bf)
                     sigma_3c(a,b,:,n) = sigma_3c(a,b,:,n) - h2c_vovv(:,k,e,f)*r_amp ! (kn)
                     sigma_3c(a,e,:,n) = sigma_3c(a,e,:,n) + h2c_vovv(:,k,b,f)*r_amp ! (be)(kn)
                     sigma_3c(a,f,:,n) = sigma_3c(a,f,:,n) + h2c_vovv(:,k,e,b)*r_amp ! (bf)(kn)
                     ! x3c(a,b,c,k) <- A(f/bc)A(kn) h2b(anef)*r4d(ebfcnk)
                     e = r4d_excits(idet,1); b = r4d_excits(idet,2); f = r4d_excits(idet,3); c = r4d_excits(idet,4);
                     n = r4d_excits(idet,5); k = r4d_excits(idet,6);
                     sigma_3c(:,b,c,k) = sigma_3c(:,b,c,k) + h2b_vovv(:,n,e,f)*r_amp ! (1)
                     sigma_3c(:,f,c,k) = sigma_3c(:,f,c,k) - h2b_vovv(:,n,e,b)*r_amp ! (bf)
                     sigma_3c(:,b,f,k) = sigma_3c(:,b,f,k) - h2b_vovv(:,n,e,c)*r_amp ! (cf)
                     sigma_3c(:,b,c,n) = sigma_3c(:,b,c,n) - h2b_vovv(:,k,e,f)*r_amp ! (kn)
                     sigma_3c(:,f,c,n) = sigma_3c(:,f,c,n) + h2b_vovv(:,k,e,b)*r_amp ! (bf)(kn)
                     sigma_3c(:,b,f,n) = sigma_3c(:,b,f,n) + h2b_vovv(:,k,e,c)*r_amp ! (cf)(kn)
                  end do

                  ! antisymmetrize (this replaces the x2a -= np.transpose(x2a, (...)) stuff in vector update
                  do k = 1, nob
                     do a = 1, nua
                        do b = 1, nub
                           do c = b+1, nub
                              val = sigma_3c(a,b,c,k) - sigma_3c(a,c,b,k)
                              sigma_3c(a,b,c,k) =  val
                              sigma_3c(a,c,b,k) = -val
                           end do
                        end do
                     end do
                  end do
                  ! (H(2) * T3)_C terms are vectorized and generally broadcast to diagonal elements, which should
                  ! be 0. Set them to 0 manually (you need to do this).
                  do a = 1,nub
                     sigma_3c(:,a,a,:) = 0.0d0
                  end do
              end subroutine build_hr_3c

              subroutine build_hr_4b(resid,&
                                     r3b,&
                                     r4b_amps, r4b_excits,&
                                     r4c_amps, r4c_excits,&
                                     t2a, t2b,&
                                     h1a_oo, h1a_vv, h1b_vv,&
                                     h2a_vvvv, h2a_oooo, h2a_voov, h2a_vooo, h2a_vvov,&
                                     h2b_vvvv, h2b_voov, h2b_ovov, h2b_vvov,&
                                     x3b_vvoo, x3b_vvvv, x3b_vovo,&
                                     x2b_oo,&
                                     n4abaa, n4abab,&
                                     noa, nua, nob, nub)
                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n4abaa, n4abab
                  !
                  real(kind=8), intent(in) :: r3b(nua,nub,nua,noa)
                  real(kind=8), intent(in) :: t2a(nua,nua,noa,noa)
                  real(kind=8), intent(in) :: t2b(nua,nub,noa,nob)
                  real(kind=8), intent(in) :: r4c_amps(n4abab)
                  integer, intent(in) :: r4c_excits(n4abab,6)
                  ! Input H and X arrays
                  real(kind=8), intent(in) :: h1a_oo(noa,noa)
                  real(kind=8), intent(in) :: h1b_vv(nub,nub)
                  real(kind=8), intent(in) :: h1a_vv(nua,nua)
                  real(kind=8), intent(in) :: h2a_vvvv(nua,nua,nua,nua)
                  real(kind=8), intent(in) :: h2a_oooo(noa,noa,noa,noa)
                  real(kind=8), intent(in) :: h2a_voov(nua,noa,noa,nua)
                  real(kind=8), intent(in) :: h2a_vooo(nua,noa,noa,noa)
                  real(kind=8), intent(in) :: h2a_vvov(nua,nua,noa,nua)
                  real(kind=8), intent(in) :: h2b_vvvv(nua,nub,nua,nub)
                  real(kind=8), intent(in) :: h2b_voov(nua,nob,noa,nub)
                  real(kind=8), intent(in) :: h2b_ovov(noa,nub,noa,nub)
                  real(kind=8), intent(in) :: h2b_vvov(nua,nub,noa,nub)
                  real(kind=8), intent(in) :: x3b_vvoo(nua,nub,noa,noa)
                  real(kind=8), intent(in) :: x3b_vvvv(nua,nub,nua,nua)
                  real(kind=8), intent(in) :: x3b_vovo(nua,nob,nua,nob)
                  real(kind=8), intent(in) :: x2b_oo(noa,nob)
                  ! Output and Inout variables
                  real(kind=8), intent(out) :: resid(n4abaa)
                  integer, intent(inout) :: r4b_excits(n4abaa,6)
                  !f2py intent(in,out) :: r4b_excits(0:n4abaa-1,0:5)
                  real(kind=8), intent(inout) :: r4b_amps(n4abaa)
                  !f2py intent(in,out) :: r4b_amps(0:n4abaa-1)
                  ! Local variables
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:)
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)
                  real(kind=8) :: r_amp, hmatel, hmatel1, res_mm23
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc
                  real(kind=8) :: ff
                  
                  ! Zero the container that holds H*R
                  resid = 0.0d0
                  
                  !!!! diagram 1: A(d/ac) h1a(de) r4b(ab~cekl)
                  !!!! diagram 2: 1/2 A(a/cd) h2a(cdef) r4b(ab~efkl)
                  ! NOTE: WITHIN THESE LOOPS, H1A(VV) TERMS ARE DOUBLE-COUNTED SO COMPENSATE BY FACTOR OF 1/2
                  ! allocate new sorting arrays
                  nloc = noa*(noa-1)/2*nua*nub
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(noa,noa,nua,nub))
                  !!! SB: (5,6,1,2) -> KLAB~ LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nua-2/), (/1,nub/), noa, noa, nua, nub)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/5,6,1,2/), noa, noa, nua, nub, nloc, n4abaa, resid)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(k,l,a,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); f = r4b_excits(jdet,4);
                        ! compute < klab~cd | h2a(vvvv) | klab~ef >
                        hmatel = h2a_vvvv(c,d,e,f)
                        ! compute < klab~cd | h1a(vv) | klab~ef > = A(cd)A(ef) h1a(ce) delta(d,f)
                        hmatel1 = 0.0d0
                        if (d==f) hmatel1 = hmatel1 + h1a_vv(c,e) ! (1)
                        if (c==f) hmatel1 = hmatel1 - h1a_vv(d,e) ! (cd)
                        if (d==e) hmatel1 = hmatel1 - h1a_vv(c,f) ! (ef)
                        if (c==e) hmatel1 = hmatel1 + h1a_vv(d,f) ! (cd)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(k,l,c,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); f = r4b_excits(jdet,4);
                        ! compute < klab~cd | h2a(vvvv) | klcb~ef >
                        hmatel = -h2a_vvvv(a,d,e,f)
                        ! compute < klab~cd | h1a(vv) | klcb~ef > = -A(ad)A(ef) h1a(ae) delta(d,f)
                        hmatel1 = 0.0d0
                        if (d==f) hmatel1 = hmatel1 - h1a_vv(a,e) ! (1)
                        if (a==f) hmatel1 = hmatel1 + h1a_vv(d,e) ! (ad)
                        if (d==e) hmatel1 = hmatel1 + h1a_vv(a,f) ! (ef)
                        if (a==e) hmatel1 = hmatel1 - h1a_vv(d,f) ! (ad)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (ad)
                     idx = idx_table(k,l,d,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); f = r4b_excits(jdet,4);
                        ! compute < klab~cd | h2a(vvvv) | kldb~ef >
                        hmatel = -h2a_vvvv(c,a,e,f)
                        ! compute < klab~cd | h1a(vv) | kldb~ef > = A(ac)A(ef) h1a(ce) delta(a,f)
                        hmatel1 = 0.0d0
                        if (a==f) hmatel1 = hmatel1 - h1a_vv(c,e) ! (1)
                        if (c==f) hmatel1 = hmatel1 + h1a_vv(a,e) ! (ac)
                        if (a==e) hmatel1 = hmatel1 + h1a_vv(c,f) ! (ef)
                        if (c==e) hmatel1 = hmatel1 - h1a_vv(a,f) ! (ac)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (5,6,3,2) -> KLCB~ LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/2,nua-1/), (/1,nub/), noa, noa, nua, nub)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/5,6,3,2/), noa, noa, nua, nub, nloc, n4abaa, resid)
                  do idet = 1, n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(k,l,c,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); f = r4b_excits(jdet,4);
                        ! compute < klab~cd | h2a(vvvv) | kleb~cf >
                        hmatel = h2a_vvvv(a,d,e,f)
                        ! compute < klab~cd | h1a(vv) | kleb~cf > = A(ad)A(ef) h1a(ae) delta(d,f)
                        hmatel1 = 0.0d0
                        if (d==f) hmatel1 = hmatel1 + h1a_vv(a,e) ! (1)
                        if (a==f) hmatel1 = hmatel1 - h1a_vv(d,e) ! (ad)
                        if (d==e) hmatel1 = hmatel1 - h1a_vv(a,f) ! (ef)
                        if (a==e) hmatel1 = hmatel1 + h1a_vv(d,f) ! (ad)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(k,l,a,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); f = r4b_excits(jdet,4);
                        ! compute < klab~cd | h2a(vvvv) | kleb~af >
                        hmatel = -h2a_vvvv(c,d,e,f)
                        ! compute < klab~cd | h1a(vv) | kleb~af > = -A(cd)A(ef) h1a(ce) delta(d,f)
                        hmatel1 = 0.0d0
                        if (d==f) hmatel1 = hmatel1 - h1a_vv(c,e) ! (1)
                        if (c==f) hmatel1 = hmatel1 + h1a_vv(d,e) ! (cd)
                        if (d==e) hmatel1 = hmatel1 + h1a_vv(c,f) ! (ef)
                        if (c==e) hmatel1 = hmatel1 - h1a_vv(d,f) ! (cd)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(k,l,d,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); f = r4b_excits(jdet,4);
                        ! compute < klab~cd | h2a(vvvv) | kleb~df >
                        hmatel = -h2a_vvvv(a,c,e,f)
                        ! compute < klab~cd | h1a(vv) | kleb~df > = -A(ac)A(ef) h1a(ae) delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 - h1a_vv(a,e) ! (1)
                        if (a==f) hmatel1 = hmatel1 + h1a_vv(c,e) ! (ac)
                        if (c==e) hmatel1 = hmatel1 + h1a_vv(a,f) ! (ef)
                        if (a==e) hmatel1 = hmatel1 - h1a_vv(c,f) ! (ac)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (5,6,4,2) -> KLDB~ LOOP !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/3,nua/), (/1,nub/), noa, noa, nua, nub)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/5,6,4,2/), noa, noa, nua, nub, nloc, n4abaa, resid)
                  do idet = 1, n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(k,l,d,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); f = r4b_excits(jdet,3);
                        ! compute < klab~cd | h2a(vvvv) | kleb~fd >
                        hmatel = h2a_vvvv(a,c,e,f)
                        ! compute < klab~cd | h1a(vv) | kleb~fd > = A(ac)A(ef) h1a(ae) delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 + h1a_vv(a,e) ! (1)
                        if (a==f) hmatel1 = hmatel1 - h1a_vv(c,e) ! (ac)
                        if (c==e) hmatel1 = hmatel1 - h1a_vv(a,f) ! (ef)
                        if (a==e) hmatel1 = hmatel1 + h1a_vv(c,f) ! (ac)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     ! (ad)
                     idx = idx_table(k,l,a,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); f = r4b_excits(jdet,3);
                        ! compute < klab~cd | h2a(vvvv) | kleb~fa >
                        hmatel = -h2a_vvvv(d,c,e,f)
                        ! compute < klab~cd | h1a(vv) | kleb~fa > = -A(cd)A(ef) h1a(de) delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 - h1a_vv(d,e) ! (1)
                        if (d==f) hmatel1 = hmatel1 + h1a_vv(c,e) ! (ac)
                        if (c==e) hmatel1 = hmatel1 + h1a_vv(d,f) ! (ef)
                        if (d==e) hmatel1 = hmatel1 - h1a_vv(c,f) ! (ac)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(k,l,c,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); f = r4b_excits(jdet,3);
                        ! compute < klab~cd | h2a(vvvv) | kleb~fc >
                        hmatel = -h2a_vvvv(a,d,e,f)
                        ! compute < klab~cd | h1a(vv) | kleb~fc > = -A(ad)A(ef) h1a(ae) delta(d,f)
                        hmatel1 = 0.0d0
                        if (d==f) hmatel1 = hmatel1 - h1a_vv(a,e) ! (1)
                        if (a==f) hmatel1 = hmatel1 + h1a_vv(d,e) ! (ac)
                        if (d==e) hmatel1 = hmatel1 + h1a_vv(a,f) ! (ef)
                        if (a==e) hmatel1 = hmatel1 - h1a_vv(d,f) ! (ac)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 3: h1b(b~e~) * r4b(ae~cdkl)
                  !!!! diagram 4: A(a/cd) h2b(ab~ef~) * r4b(ef~cdkl)
                  ff = 1.0d0 / 3.0d0
                  ! allocate new sorting arrays
                  nloc = noa*(noa-1)/2*(nua-1)*(nua-2)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(noa,noa,nua,nua))
                  !!! SB: (5,6,3,4) !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/2,nua-1/), (/-1,nua/), noa, noa, nua, nua)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/5,6,3,4/), noa, noa, nua, nua, nloc, n4abaa, resid)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(k,l,c,d)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); f = r4b_excits(jdet,2);
                        ! compute < klab~cd | h2b(vvvv) | klef~cd >
                        hmatel = h2b_vvvv(a,b,e,f)
                        ! compute < klab~cd | h1b(vv) | klef~cd > = delta(a,e) * h1b(b~f~)
                        hmatel1 = 0.0d0
                        if (a==e) hmatel1 = hmatel1 + h1b_vv(b,f) 
                        hmatel = hmatel + ff * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(k,l,a,d)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); f = r4b_excits(jdet,2);
                        ! compute < klab~cd | h2b(vvvv) | klef~ad >
                        hmatel = -h2b_vvvv(c,b,e,f)
                        ! compute < klab~cd | h1b(vv) | klef~cd > = delta(c,e) * h1b(b~f~)
                        hmatel1 = 0.0d0
                        if (c==e) hmatel1 = hmatel1 - h1b_vv(b,f) 
                        hmatel = hmatel + ff * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (ad), -
                     idx = idx_table(k,l,a,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); f = r4b_excits(jdet,2);
                        ! compute < klab~cd | h2b(vvvv) | klef~ac >
                        hmatel = h2b_vvvv(d,b,e,f)
                        ! compute < klab~cd | h1b(vv) | klef~cd > = delta(d,e) * h1b(b~f~)
                        hmatel1 = 0.0d0
                        if (d==e) hmatel1 = hmatel1 + h1b_vv(b,f) 
                        hmatel = hmatel + ff * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (5,6,1,4) !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nua-2/), (/-2,nua/), noa, noa, nua, nua)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/5,6,1,4/), noa, noa, nua, nua, nloc, n4abaa, resid)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(k,l,a,d)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); f = r4b_excits(jdet,2);
                        ! compute < klab~cd | h2b(vvvv) | klaf~ed >
                        hmatel = h2b_vvvv(c,b,e,f)
                        ! compute < klab~cd | h1b(vv) | klaf~ed > = delta(c,e) * h1b(b~f~)
                        hmatel1 = 0.0d0
                        if (c==e) hmatel1 = hmatel1 + h1b_vv(b,f) 
                        hmatel = hmatel + ff * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(k,l,c,d)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); f = r4b_excits(jdet,2);
                        ! compute < klab~cd | h2b(vvvv) | klcf~ed >
                        hmatel = -h2b_vvvv(a,b,e,f)
                        ! compute < klab~cd | h1b(vv) | klaf~ed > = delta(a,e) * h1b(b~f~)
                        hmatel1 = 0.0d0
                        if (a==e) hmatel1 = hmatel1 - h1b_vv(b,f) 
                        hmatel = hmatel + ff * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(k,l,a,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); f = r4b_excits(jdet,2);
                        ! compute < klab~cd | h2b(vvvv) | klaf~ec >
                        hmatel = -h2b_vvvv(d,b,e,f)
                        ! compute < klab~cd | h1b(vv) | klaf~ed > = delta(d,e) * h1b(b~f~)
                        hmatel1 = 0.0d0
                        if (d==e) hmatel1 = hmatel1 - h1b_vv(b,f) 
                        hmatel = hmatel + ff * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (5,6,1,3) !!!
                  call get_index_table(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nua-2/), (/-1,nua-1/), noa, noa, nua, nua)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/5,6,1,3/), noa, noa, nua, nua, nloc, n4abaa, resid)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(k,l,a,c)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,4); f = r4b_excits(jdet,2);
                        ! compute < klab~cd | h2b(vvvv) | klaf~ce >
                        hmatel = h2b_vvvv(d,b,e,f)
                        ! compute < klab~cd | h1b(vv) | klaf~ce > = delta(d,e) * h1b(b~f~)
                        hmatel1 = 0.0d0
                        if (d==e) hmatel1 = hmatel1 + h1b_vv(b,f) 
                        hmatel = hmatel + ff * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     ! (ad), -
                     idx = idx_table(k,l,c,d)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,4); f = r4b_excits(jdet,2);
                        ! compute < klab~cd | h2b(vvvv) | klaf~ce >
                        hmatel = h2b_vvvv(a,b,e,f)
                        ! compute < klab~cd | h1b(vv) | klaf~ce > = delta(a,e) * h1b(b~f~)
                        hmatel1 = 0.0d0
                        if (a==e) hmatel1 = hmatel1 + h1b_vv(b,f) 
                        hmatel = hmatel + ff * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(k,l,a,d)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,4); f = r4b_excits(jdet,2);
                        ! compute < klab~cd | h2b(vvvv) | klaf~ce >
                        hmatel = -h2b_vvvv(c,b,e,f)
                        ! compute < klab~cd | h1b(vv) | klaf~ce > = delta(c,e) * h1b(b~f~)
                        hmatel1 = 0.0d0
                        if (c==e) hmatel1 = hmatel1 - h1b_vv(b,f) 
                        hmatel = hmatel + ff * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!! diagram 5: h2a(mnkl) * r4b(ab~cdmn)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)*(nua-2)/6 * nub 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,nua,nub))
                  !!! SB: (1,3,4,2) !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/1,nub/), nua, nua, nua, nub)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/1,3,4,2/), nua, nua, nua, nub, nloc, n4abaa, resid)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,d,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4b_excits(jdet,5); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(oooo) | mnab~cd >
                        hmatel = h2a_oooo(m,n,k,l)
                        ! compute < klab~cd | h1a(oo) | mnab~cd > = A(kl)A(mn) -delta(m,k) * h1a_oo(n,l)
                        hmatel1 = 0.0d0
                        if (m==k) hmatel1 = hmatel1  - h1a_oo(n,l) ! (1) 
                        if (m==l) hmatel1 = hmatel1  + h1a_oo(n,k) ! (kl) 
                        if (n==k) hmatel1 = hmatel1  + h1a_oo(m,l) ! (mn) 
                        if (n==l) hmatel1 = hmatel1  - h1a_oo(m,k) ! (kl)(mn) 
                        hmatel = hmatel + hmatel1
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!! diagram 6: A(d/ac)A(kl) h2a(dmle) * r4b(abcekm)
                  ! allocate new sorting arrays
                  nloc = (nua-1)*(nua-2)/2 * (noa-1) * nub 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,noa,nub))
                  !!! SB: (1,3,5,2) !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/1,noa-1/), (/1,nub/), nua, nua, noa, nub)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/1,3,5,2/), nua, nua, noa, nub, nloc, n4abaa, resid)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,k,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,4); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | knab~cf >
                        hmatel = h2a_voov(d,n,l,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     ! (ad), -
                     idx = idx_table(c,d,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,4); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | kndb~cf >
                        hmatel = h2a_voov(a,n,l,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(a,d,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,4); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | knab~df >
                        hmatel = -h2a_voov(c,n,l,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(a,c,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,4); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | lnab~cf >
                        hmatel = -h2a_voov(d,n,k,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (ad)(kl), -
                     idx = idx_table(c,d,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,4); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | lndb~cf >
                        hmatel = -h2a_voov(a,n,k,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (cd)(kl)
                     idx = idx_table(a,d,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,4); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | lnab~df >
                        hmatel = h2a_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,4,5,2) !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/1,noa-1/), (/1,nub/), nua, nua, noa, nub)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/1,4,5,2/), nua, nua, noa, nub, nloc, n4abaa, resid)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,d,k,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | knab~ed >
                        hmatel = h2a_voov(c,n,l,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(c,d,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | kncb~ed >
                        hmatel = -h2a_voov(a,n,l,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(a,c,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | knab~ec >
                        hmatel = -h2a_voov(d,n,l,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(a,d,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | lnab~ed >
                        hmatel = -h2a_voov(c,n,k,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (ac)(kl)
                     idx = idx_table(c,d,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | lncb~ed >
                        hmatel = h2a_voov(a,n,k,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (cd)(kl)
                     idx = idx_table(a,c,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | lnab~ec >
                        hmatel = h2a_voov(d,n,k,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (3,4,5,2) !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/1,noa-1/), (/1,nub/), nua, nua, noa, nub)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/3,4,5,2/), nua, nua, noa, nub, nloc, n4abaa, resid)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(c,d,k,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | kneb~cd >
                        hmatel = h2a_voov(a,n,l,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(a,d,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | kneb~ad >
                        hmatel = -h2a_voov(c,n,l,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (ad), -
                     idx = idx_table(a,c,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | kneb~ac >
                        hmatel = h2a_voov(d,n,l,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(c,d,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | lneb~cd >
                        hmatel = -h2a_voov(a,n,k,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (ac)(kl)
                     idx = idx_table(a,d,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | lneb~ad >
                        hmatel = h2a_voov(c,n,k,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (ad)(kl), -
                     idx = idx_table(a,c,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); n = r4b_excits(jdet,6);
                        ! compute < klab~cd | h2a(voov) | lneb~ac >
                        hmatel = -h2a_voov(d,n,k,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,3,6,2) !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/2,noa/), (/1,nub/), nua, nua, noa, nub)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/1,3,6,2/), nua, nua, noa, nub, nloc, n4abaa, resid)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,l,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,4); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mlab~cf >
                        hmatel = h2a_voov(d,m,k,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     ! (ad), -
                     idx = idx_table(c,d,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,4); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mldb~cf >
                        hmatel = h2a_voov(a,m,k,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(a,d,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,4); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mlab~df >
                        hmatel = -h2a_voov(c,m,k,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(a,c,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,4); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mkab~cf >
                        hmatel = -h2a_voov(d,m,l,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (ad)(kl), -
                     idx = idx_table(c,d,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,4); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mkdb~cf >
                        hmatel = -h2a_voov(a,m,l,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (cd)(kl)
                     idx = idx_table(a,d,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,4); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mkab~df >
                        hmatel = h2a_voov(c,m,l,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,4,6,2) !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/2,noa/), (/1,nub/), nua, nua, noa, nub)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/1,4,6,2/), nua, nua, noa, nub, nloc, n4abaa, resid)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,d,l,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mlab~ed >
                        hmatel = h2a_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(c,d,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mlcb~ed >
                        hmatel = -h2a_voov(a,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(a,c,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mlab~ec >
                        hmatel = -h2a_voov(d,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(a,d,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mkab~ed >
                        hmatel = -h2a_voov(c,m,l,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (ac)(kl)
                     idx = idx_table(c,d,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mkcb~ed >
                        hmatel = h2a_voov(a,m,l,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (cd)(kl)
                     idx = idx_table(a,c,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,3); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mkab~ec >
                        hmatel = h2a_voov(d,m,l,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (3,4,6,2) !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/2,noa/), (/1,nub/), nua, nua, noa, nub)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/3,4,6,2/), nua, nua, noa, nub, nloc, n4abaa, resid)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(c,d,l,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mleb~cd >
                        hmatel = h2a_voov(a,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(a,d,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mleb~ad >
                        hmatel = -h2a_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (ad), -
                     idx = idx_table(a,c,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mleb~ac >
                        hmatel = h2a_voov(d,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(c,d,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mkeb~cd >
                        hmatel = -h2a_voov(a,m,l,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (ac)(kl)
                     idx = idx_table(a,d,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mkeb~ad >
                        hmatel = h2a_voov(c,m,l,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                     ! (ad)(kl), -
                     idx = idx_table(a,c,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4b_excits(jdet,1); m = r4b_excits(jdet,5);
                        ! compute < klab~cd | h2a(voov) | mkeb~ac >
                        hmatel = -h2a_voov(d,m,l,e)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!! diagram 7: A(d/ac)A(kl) h2b(dm~le~) * r4c(ab~ce~km~)
                  ! copy over excitations
                  allocate(excits_buff(n4abab,6),amps_buff(n4abab))
                  excits_buff(:,:) = r4c_excits(:,:)
                  amps_buff(:) = r4c_amps(:)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2 * (nub-1) * noa 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,nub,noa))
                  !!! SB: (1,3,2,5) !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub-1/), (/1,noa/), nua, nua, nub, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,2,5/), nua, nua, nub, noa, nloc, n4abab)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,4); n = excits_buff(jdet,6);
                        ! compute < klab~cd | h2b(voov) | kn~ab~cf~ >
                        hmatel = h2b_voov(d,n,l,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ad), -
                     idx = idx_table(c,d,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,4); n = excits_buff(jdet,6);
                        ! compute < klab~cd | h2b(voov) | kn~cb~df~ >
                        hmatel = h2b_voov(a,n,l,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(a,d,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,4); n = excits_buff(jdet,6);
                        ! compute < klab~cd | h2b(voov) | kn~ab~df~ >
                        hmatel = -h2b_voov(c,n,l,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(a,c,b,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,4); n = excits_buff(jdet,6);
                        ! compute < klab~cd | h2b(voov) | ln~ab~cf~ >
                        hmatel = -h2b_voov(d,n,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ad)(kl), -
                     idx = idx_table(c,d,b,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,4); n = excits_buff(jdet,6);
                        ! compute < klab~cd | h2b(voov) | ln~cb~df~ >
                        hmatel = -h2b_voov(a,n,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (cd)(kl)
                     idx = idx_table(a,d,b,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,4); n = excits_buff(jdet,6);
                        ! compute < klab~cd | h2b(voov) | ln~ab~df~ >
                        hmatel = h2b_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,3,4,5) !!!
                  !!! THIS SB IS IDENTICAL TO THE ABOVE ON EXCEPT ALL SIGNS ARE REVERSED !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/2,nub/), (/1,noa/), nua, nua, nub, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,4,5/), nua, nua, nub, noa, nloc, n4abab)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        ! compute < klab~cd | h2b(voov) | kn~af~cb~ >
                        hmatel = -h2b_voov(d,n,l,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ad), -
                     idx = idx_table(c,d,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        ! compute < klab~cd | h2b(voov) | kn~cf~db~ >
                        hmatel = -h2b_voov(a,n,l,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(a,d,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        ! compute < klab~cd | h2b(voov) | kn~af~db~ >
                        hmatel = h2b_voov(c,n,l,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(a,c,b,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        ! compute < klab~cd | h2b(voov) | ln~af~cb~ >
                        hmatel = h2b_voov(d,n,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ad)(kl), -
                     idx = idx_table(c,d,b,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        ! compute < klab~cd | h2b(voov) | ln~cf~db~ >
                        hmatel = h2b_voov(a,n,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (cd)(kl)
                     idx = idx_table(a,d,b,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,2); n = excits_buff(jdet,6);
                        ! compute < klab~cd | h2b(voov) | ln~af~db~ >
                        hmatel = -h2b_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate buffer arrays
                  deallocate(excits_buff,amps_buff)

                  !!! diagram 8: A(kl) -h2b(mb~le~) * r4b(ae~cdkm)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)*(nua-2)/6 * (noa-1) 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,nua,noa))
                  !!! SB: (1,3,4,5) !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/1,noa-1/), nua, nua, nua, noa)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/1,3,4,5/), nua, nua, nua, noa, nloc, n4abaa, resid)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,d,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,2); n = r4b_excits(jdet,6);
                        ! compute < ab~cdkl | h2b(ovov) | af~cdkn >
                        hmatel = -h2b_ovov(n,b,l,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     ! (kl)
                     idx = idx_table(a,c,d,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,2); n = r4b_excits(jdet,6);
                        ! compute < ab~cdkl | h2b(ovov) | af~cdln >
                        hmatel = h2b_ovov(n,b,k,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,3,4,6) !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/2,noa/), nua, nua, nua, noa)
                  call sort4(r4b_excits, r4b_amps, loc_arr, idx_table, (/1,3,4,6/), nua, nua, nua, noa, nloc, n4abaa, resid)
                  do idet = 1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,d,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,2); m = r4b_excits(jdet,5);
                        ! compute < ab~cdkl | h2b(ovov) | af~cdml >
                        hmatel = -h2b_ovov(m,b,k,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     ! (kl)
                     idx = idx_table(a,c,d,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4b_excits(jdet,2); m = r4b_excits(jdet,5);
                        ! compute < ab~cdkl | h2b(ovov) | af~cdmk >
                        hmatel = h2b_ovov(m,b,l,f)
                        resid(idet) = resid(idet) + hmatel * r4b_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !
                  ! Moment contributions
                  !
                  do idet=1,n4abaa
                     a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                     k = r4b_excits(idet,5); l = r4b_excits(idet,6);
                     res_mm23 = 0.0d0
                     do e=1,nua
                        ! A(a/cd)A(kl) h2a(cdke)*r3b(ab~el)
                        res_mm23 = res_mm23 + h2a_vvov(c,d,k,e)*r3b(a,b,e,l) ! (1)
                        res_mm23 = res_mm23 - h2a_vvov(a,d,k,e)*r3b(c,b,e,l) ! (ac)
                        res_mm23 = res_mm23 - h2a_vvov(c,a,k,e)*r3b(d,b,e,l) ! (ad)
                        res_mm23 = res_mm23 - h2a_vvov(c,d,l,e)*r3b(a,b,e,k) ! (kl)
                        res_mm23 = res_mm23 + h2a_vvov(a,d,l,e)*r3b(c,b,e,k) ! (ac)(kl)
                        res_mm23 = res_mm23 + h2a_vvov(c,a,l,e)*r3b(d,b,e,k) ! (ad)(kl)
                        ! A(c/ad) x3b(ab~de)*t2a(cekl)
                        res_mm23 = res_mm23 + x3b_vvvv(a,b,d,e)*t2a(c,e,k,l) ! (1)
                        res_mm23 = res_mm23 - x3b_vvvv(c,b,d,e)*t2a(a,e,k,l) ! (ac)
                        res_mm23 = res_mm23 - x3b_vvvv(a,b,c,e)*t2a(d,e,k,l) ! (cd)
                     end do
                     do e=1,nub
                        ! A(c/ad)A(kl) h2b(cbke)*r3b(ae~dl)
                        res_mm23 = res_mm23 + h2b_vvov(c,b,k,e)*r3b(a,e,d,l) ! (1)
                        res_mm23 = res_mm23 - h2b_vvov(a,b,k,e)*r3b(c,e,d,l) ! (ac)
                        res_mm23 = res_mm23 - h2b_vvov(d,b,k,e)*r3b(a,e,c,l) ! (cd)
                        res_mm23 = res_mm23 - h2b_vvov(c,b,l,e)*r3b(a,e,d,k) ! (kl)
                        res_mm23 = res_mm23 + h2b_vvov(a,b,l,e)*r3b(c,e,d,k) ! (ac)(kl)
                        res_mm23 = res_mm23 + h2b_vvov(d,b,l,e)*r3b(a,e,c,k) ! (cd)(kl)
                     end do
                     ! include this here:
                     !X4B += (6.0 / 12.0) * np.einsum("mn,adml,cbkn->abcdkl", X["ab"]["oo"], T.aa, T.ab, optimize=True)
                     do m=1,noa
                        ! -A(c/ad) h2a(cmkl)*r3b(ab~dm)
                        res_mm23 = res_mm23 - h2a_vooo(c,m,k,l)*r3b(a,b,d,m) ! (1)
                        res_mm23 = res_mm23 + h2a_vooo(a,m,k,l)*r3b(c,b,d,m) ! (ac)
                        res_mm23 = res_mm23 + h2a_vooo(d,m,k,l)*r3b(a,b,c,m) ! (cd)
                        ! -A(a/cd)A(kl) x3b(ab~ml)*t2a(cdkm)
                        res_mm23 = res_mm23 - x3b_vvoo(a,b,m,l)*t2a(c,d,k,m) ! (1)
                        res_mm23 = res_mm23 + x3b_vvoo(c,b,m,l)*t2a(a,d,k,m) ! (ac)
                        res_mm23 = res_mm23 + x3b_vvoo(d,b,m,l)*t2a(c,a,k,m) ! (ad)
                        res_mm23 = res_mm23 + x3b_vvoo(a,b,m,k)*t2a(c,d,l,m) ! (kl)
                        res_mm23 = res_mm23 - x3b_vvoo(c,b,m,k)*t2a(a,d,l,m) ! (ac)(kl)
                        res_mm23 = res_mm23 - x3b_vvoo(d,b,m,k)*t2a(c,a,l,m) ! (ad)(kl)
                     end do
                     do m=1,nob
                        ! -A(d/ac)A(kl) x3b(am~ck)*t2b(dblm)
                        res_mm23 = res_mm23 - x3b_vovo(a,m,c,k)*t2b(d,b,l,m) ! (1) 
                        res_mm23 = res_mm23 + x3b_vovo(d,m,c,k)*t2b(a,b,l,m) ! (ad) 
                        res_mm23 = res_mm23 + x3b_vovo(a,m,d,k)*t2b(c,b,l,m) ! (cd) 
                        res_mm23 = res_mm23 + x3b_vovo(a,m,c,l)*t2b(d,b,k,m) ! (kl) 
                        res_mm23 = res_mm23 - x3b_vovo(d,m,c,l)*t2b(a,b,k,m) ! (ad)(kl) 
                        res_mm23 = res_mm23 - x3b_vovo(a,m,d,l)*t2b(c,b,k,m) ! (cd) (kl)
                     end do
                     do m=1,noa
                        do n=1,nob
                           ! A(c/ad)A(kl) x2b(mn~)*t2a(adml)*t2b(cbkn)
                           res_mm23 = res_mm23 + x2b_oo(m,n)*t2a(a,d,m,l)*t2b(c,b,k,n) ! (1)
                           res_mm23 = res_mm23 - x2b_oo(m,n)*t2a(c,d,m,l)*t2b(a,b,k,n) ! (ac)
                           res_mm23 = res_mm23 - x2b_oo(m,n)*t2a(a,c,m,l)*t2b(d,b,k,n) ! (cd)
                           res_mm23 = res_mm23 - x2b_oo(m,n)*t2a(a,d,m,k)*t2b(c,b,l,n) ! (kl)
                           res_mm23 = res_mm23 + x2b_oo(m,n)*t2a(c,d,m,k)*t2b(a,b,l,n) ! (ac)(kl)
                           res_mm23 = res_mm23 + x2b_oo(m,n)*t2a(a,c,m,k)*t2b(d,b,l,n) ! (cd)(kl(
                        end do
                     end do
                     resid(idet) = resid(idet) + res_mm23
                  end do
                  
              end subroutine build_hr_4b

              subroutine build_hr_4c(resid,&
                                     r3b, r3c,&
                                     r4b_amps, r4b_excits,&
                                     r4c_amps, r4c_excits,&
                                     r4d_amps, r4d_excits,&
                                     t2a, t2b, t2c,&
                                     h1a_oo, h1b_oo, h1a_vv, h1b_vv,&
                                     h2a_vvvv, h2a_voov, h2a_vvov,&
                                     h2b_vvvv, h2b_oooo, h2b_voov, h2b_ovvo, h2b_ovov, h2b_vovo,&
                                     h2b_vvov, h2b_vvvo, h2b_vooo, h2b_ovoo,&
                                     h2c_vvvv, h2c_voov, h2c_vvov,&
                                     x3b_vvvv, x3b_vvoo, x3b_vovo, x3c_vvvv, x3c_vvoo, x3c_ovvo,&
                                     x2b_oo,&
                                     n4abaa, n4abab, n4abbb,&
                                     noa, nua, nob, nub)
                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n4abaa, n4abab, n4abbb
                  !
                  real(kind=8), intent(in) :: r3b(nua,nub,nua,noa)
                  real(kind=8), intent(in) :: r3c(nua,nub,nub,nob)
                  real(kind=8), intent(in) :: t2a(nua,nua,noa,noa)
                  real(kind=8), intent(in) :: t2b(nua,nub,noa,nob)
                  real(kind=8), intent(in) :: t2c(nub,nub,nob,nob)
                  real(kind=8), intent(in) :: r4b_amps(n4abaa), r4d_amps(n4abbb)
                  integer, intent(in) :: r4b_excits(n4abaa,6), r4d_excits(n4abbb,6)
                  ! Input H and X arrays
                  real(kind=8), intent(in) :: h1a_oo(noa,noa)
                  real(kind=8), intent(in) :: h1b_oo(nob,nob)
                  real(kind=8), intent(in) :: h1b_vv(nub,nub)
                  real(kind=8), intent(in) :: h1a_vv(nua,nua)
                  real(kind=8), intent(in) :: h2a_vvvv(nua,nua,nua,nua)
                  real(kind=8), intent(in) :: h2a_voov(nua,noa,noa,nua)
                  real(kind=8), intent(in) :: h2a_vvov(nua,nua,noa,nua)
                  real(kind=8), intent(in) :: h2b_vvvv(nua,nub,nua,nub)
                  real(kind=8), intent(in) :: h2b_oooo(noa,nob,noa,nob)
                  real(kind=8), intent(in) :: h2b_voov(nua,nob,noa,nub)
                  real(kind=8), intent(in) :: h2b_ovvo(noa,nub,nua,nob)
                  real(kind=8), intent(in) :: h2b_ovov(noa,nub,noa,nub)
                  real(kind=8), intent(in) :: h2b_vovo(nua,nob,nua,nob)
                  real(kind=8), intent(in) :: h2b_vvov(nua,nub,noa,nub)
                  real(kind=8), intent(in) :: h2b_vvvo(nua,nub,nua,nob)
                  real(kind=8), intent(in) :: h2b_vooo(nua,nob,noa,nob)
                  real(kind=8), intent(in) :: h2b_ovoo(noa,nub,noa,nob)
                  real(kind=8), intent(in) :: h2c_vvvv(nub,nub,nub,nub)
                  real(kind=8), intent(in) :: h2c_voov(nub,nob,nob,nub)
                  real(kind=8), intent(in) :: h2c_vvov(nub,nub,nob,nub)
                  !
                  real(kind=8), intent(in) :: x3b_vvvv(nua,nub,nua,nua)
                  real(kind=8), intent(in) :: x3b_vvoo(nua,nub,noa,noa)
                  real(kind=8), intent(in) :: x3b_vovo(nua,nob,nua,nob)
                  real(kind=8), intent(in) :: x3c_vvvv(nua,nub,nub,nub)
                  real(kind=8), intent(in) :: x3c_vvoo(nua,nub,nob,nob)
                  real(kind=8), intent(in) :: x3c_ovvo(noa,nub,nub,nob)
                  real(kind=8), intent(in) :: x2b_oo(noa,nob)
                  ! Output and Inout variables
                  real(kind=8), intent(out) :: resid(n4abab)
                  integer, intent(inout) :: r4c_excits(n4abab,6)
                  !f2py intent(in,out) :: r4c_excits(0:n4abab-1,0:5)
                  real(kind=8), intent(inout) :: r4c_amps(n4abab)
                  !f2py intent(in,out) :: r4c_amps(0:n4abab-1)
                  ! Local variables
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:)
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)
                  real(kind=8) :: r_amp, hmatel, hmatel1, hmatel2, res_mm23
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc
                  real(kind=8) :: ff
                  
                  ! Zero the container that holds H*R
                  resid = 0.0d0
                  
                  !!! diagram 1: A(ac) h1a(ae) * r4c(eb~cd~kl~)
                  !!! diagram 7: h2a(acef) * r4c(eb~fd~kl~) 
                  ! allocate new sorting arrays
                  nloc = nub*(nub-1)/2 * noa * nob 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,noa,nob))
                  !!! SB: (2,4,5,6) !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,noa/), (/1,nob/), nub, nub, noa, nob)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/2,4,5,6/), nub, nub, noa, nob, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,d,k,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,1); f = r4c_excits(jdet,3);
                        ! compute < ab~cd~kl~ | h2a(vvvv) | eb~fd~kl~ >
                        hmatel = h2a_vvvv(a,c,e,f)
                        ! compute < ab~cd~kl~ | h1a(vv) | eb~fd~kl~ > = A(ac)(ef) h1a(ae) delta(f,c)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 + h1a_vv(a,e) ! (1)
                        if (a==f) hmatel1 = hmatel1 - h1a_vv(c,e) ! (ac)
                        if (c==e) hmatel1 = hmatel1 - h1a_vv(a,f) ! (ef)
                        if (a==e) hmatel1 = hmatel1 + h1a_vv(c,f) ! (ac)(ef)
                        hmatel = hmatel + hmatel1
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!! diagram 2: A(bd) h1b(bf) * r4c(af~cd~kl~)
                  !!! diagram 8: A(bd) h2c(bdef) * r4c(ae~cf~kl~)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2 * noa * nob 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,noa,nob))
                  !!! SB: (1,3,5,6) !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,noa/), (/1,nob/), nua, nua, noa, nob)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/1,3,5,6/), nua, nua, noa, nob, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,k,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,2); f = r4c_excits(jdet,4);
                        ! compute < ab~cd~kl~ | h2c(vvvv) | ae~cf~kl~ >
                        hmatel = h2c_vvvv(b,d,e,f)
                        ! compute < ab~cd~kl~ | h1a(vv) | ae~cf~kl~ > = A(bd)(ef) h1b(be) delta(f,d)
                        hmatel1 = 0.0d0
                        if (d==f) hmatel1 = hmatel1 + h1b_vv(b,e) ! (1)
                        if (b==f) hmatel1 = hmatel1 - h1b_vv(d,e) ! (bd)
                        if (d==e) hmatel1 = hmatel1 - h1b_vv(b,f) ! (ef)
                        if (b==e) hmatel1 = hmatel1 + h1b_vv(d,f) ! (bd)(ef)
                        hmatel = hmatel + hmatel1
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!! diagram 6: A(ac)A(bd) h2b(cd~ef~) * r4c(ab~ef~kl~)
                  ! allocate new sorting arrays
                  nloc = (nua-1)*(nub-1)*noa*nob 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nub,noa,nob))
                  !!! SB: (1,2,5,6) !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/1,nub-1/), (/1,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/1,2,5,6/), nua, nub, noa, nob, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,k,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,3); f = r4c_excits(jdet,4);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | ab~ef~kl~ >
                        hmatel = h2b_vvvv(c,d,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(c,b,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,3); f = r4c_excits(jdet,4);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | cb~ef~kl~ >
                        hmatel = -h2b_vvvv(a,d,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                     ! (bd)
                     idx = idx_table(a,d,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,3); f = r4c_excits(jdet,4);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | ad~ef~kl~ >
                        hmatel = -h2b_vvvv(c,b,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                     ! (ac)(bd)
                     idx = idx_table(c,d,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,3); f = r4c_excits(jdet,4);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | cd~ef~kl~ >
                        hmatel = h2b_vvvv(a,b,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,4,5,6) !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/2,nub/), (/1,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/1,4,5,6/), nua, nub, noa, nob, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,d,k,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,3); f = r4c_excits(jdet,2);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | af~ed~kl~ >
                        hmatel = h2b_vvvv(c,b,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(c,d,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,3); f = r4c_excits(jdet,2);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | cf~ed~kl~ >
                        hmatel = -h2b_vvvv(a,b,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                     ! (bd)
                     idx = idx_table(a,b,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,3); f = r4c_excits(jdet,2);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | af~eb~kl~ >
                        hmatel = -h2b_vvvv(c,d,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                     ! (ac)(bd)
                     idx = idx_table(c,b,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,3); f = r4c_excits(jdet,2);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | cf~eb~kl~ >
                        hmatel = h2b_vvvv(a,d,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (3,2,5,6) !!!
                  call get_index_table(idx_table, (/2,nua/), (/1,nub-1/), (/1,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/3,2,5,6/), nua, nub, noa, nob, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(c,b,k,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,1); f = r4c_excits(jdet,4);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | eb~cf~kl~ >
                        hmatel = h2b_vvvv(a,d,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(a,b,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,1); f = r4c_excits(jdet,4);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | eb~af~kl~ >
                        hmatel = -h2b_vvvv(c,d,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                     ! (bd)
                     idx = idx_table(c,d,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,1); f = r4c_excits(jdet,4);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | ed~cf~kl~ >
                        hmatel = -h2b_vvvv(a,b,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                     ! (ac)(bd)
                     idx = idx_table(a,d,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,1); f = r4c_excits(jdet,4);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | ed~af~kl~ >
                        hmatel = h2b_vvvv(c,b,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (3,4,5,6) !!!
                  call get_index_table(idx_table, (/2,nua/), (/2,nub/), (/1,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/3,4,5,6/), nua, nub, noa, nob, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(c,d,k,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,1); f = r4c_excits(jdet,2);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | ef~cd~kl~ >
                        hmatel = h2b_vvvv(a,b,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(a,d,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,1); f = r4c_excits(jdet,2);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | ef~ad~kl~ >
                        hmatel = -h2b_vvvv(c,b,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                     ! (bd)
                     idx = idx_table(c,b,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,1); f = r4c_excits(jdet,2);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | ef~cb~kl~ >
                        hmatel = -h2b_vvvv(a,d,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                     ! (ac)(bd)
                     idx = idx_table(a,b,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,1); f = r4c_excits(jdet,2);
                        ! compute < ab~cd~kl~ | h2b(vvvv) | ef~ab~kl~ >
                        hmatel = h2b_vvvv(c,d,e,f)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!! diagram 3: -h1a(mk) * r4c(ab~cd~ml~) 
                  !!! diagram 4: -h1b(nl) * r4c(ab~cd~kn~)
                  !!! diagram 5: h2b(mn~kl~) * r4c(ab~cd~mn~)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2 * nub*(nub-1)/2 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,nub,nub))
                  !!! SB: (1,3,2,4) !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub-1/), (/-1,nub/), nua, nua, nub, nub)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/1,3,2,4/), nua, nua, nub, nub, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,b,d)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4c_excits(jdet,5); n = r4c_excits(jdet,6);
                        ! compute < ab~cd~kl~ | h2b(oooo) | ab~cd~mn~ >
                        hmatel = h2b_oooo(m,n,k,l)
                        ! compute < ab~cd~kl~ | h1a(oo) | ab~cd~mn~ > = -h1b(m,k) * delta(l,n)
                        hmatel1 = 0.0d0
                        if (l==n) hmatel1 = hmatel1 - h1a_oo(m,k) ! (1)
                        ! compute < ab~cd~kl~ | h1b(oo) | ab~cd~mn~ > = -h1b(n,l) * delta(k,m)
                        hmatel2 = 0.0d0
                        if (k==m) hmatel2 = hmatel2 - h1b_oo(n,l) ! (1)
                        hmatel = hmatel + hmatel1 + hmatel2
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!! diagram 9: A(ac) h2a(cmke) * r4c(ab~ed~ml~) 
                  ! allocate new sorting arrays
                  nloc = nub*(nub-1)/2 * (nua-1) * nob 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nua,nob))
                  !!! SB: (2,4,1,6) !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nua-1/), (/1,nob/), nub, nub, nua, nob)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/2,4,1,6/), nub, nub, nua, nob, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,d,a,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,3); m = r4c_excits(jdet,5);
                        ! compute < ab~cd~kl~ | h2a(voov) | ab~ed~ml~ >
                        hmatel = h2a_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(b,d,c,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,3); m = r4c_excits(jdet,5);
                        ! compute < ab~cd~kl~ | h2a(voov) | cb~ed~ml~ >
                        hmatel = -h2a_voov(a,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (2,4,3,6) !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/2,nua/), (/1,nob/), nub, nub, nua, nob)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/2,4,3,6/), nub, nub, nua, nob, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,d,c,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,1); m = r4c_excits(jdet,5);
                        ! compute < ab~cd~kl~ | h2a(voov) | eb~cd~ml~ >
                        hmatel = h2a_voov(a,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(b,d,a,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,1); m = r4c_excits(jdet,5);
                        ! compute < ab~cd~kl~ | h2a(voov) | eb~ad~ml~ >
                        hmatel = -h2a_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!! diagram 10: h2b(cm~ke~) * r4d(ab~e~d~m~l~) 
                  ! copy over excitations
                  allocate(excits_buff(n4abbb,6),amps_buff(n4abbb))
                  excits_buff(:,:) = r4d_excits(:,:)
                  amps_buff(:) = r4d_amps(:)
                  ! allocate new sorting arrays
                  nloc = nua * (nub-1)*(nub-2)/2 * (nob-1) 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nua,nob))
                  !!! SB: (2,4,1,6) !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/1,nua/), (/2,nob/), nub, nub, nua, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,4,1,6/), nub, nub, nua, nob, nloc, n4abbb)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,d,a,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        ! compute < ab~cd~kl~ | h2a(voov) | ab~e~d~m~l~ >
                        hmatel = h2b_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,d,c,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        ! compute < ab~cd~kl~ | h2a(voov) | cb~e~d~m~l~ >
                        hmatel = -h2b_voov(a,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (2,4,1,5) !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/1,nua/), (/1,nob-1/), nub, nub, nua, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,4,1,5/), nub, nub, nua, nob, nloc, n4abbb)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,d,a,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                        ! compute < ab~cd~kl~ | h2a(voov) | ab~e~d~l~m~ >
                        hmatel = -h2b_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,d,c,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                        ! compute < ab~cd~kl~ | h2a(voov) | cb~e~d~l~m~ >
                        hmatel = h2b_voov(a,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (2,3,1,6) !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/1,nua/), (/2,nob/), nub, nub, nua, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,1,6/), nub, nub, nua, nob, nloc, n4abbb)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,d,a,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,4); m = excits_buff(jdet,5);
                        ! compute < ab~cd~kl~ | h2a(voov) | ab~d~e~m~l~ >
                        hmatel = -h2b_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,d,c,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,4); m = excits_buff(jdet,5);
                        ! compute < ab~cd~kl~ | h2a(voov) | cb~d~e~m~l~ >
                        hmatel = h2b_voov(a,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (2,3,1,5) !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/1,nua/), (/1,nob-1/), nub, nub, nua, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,3,1,5/), nub, nub, nua, nob, nloc, n4abbb)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,d,a,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,4); m = excits_buff(jdet,6);
                        ! compute < ab~cd~kl~ | h2a(voov) | ab~d~e~m~l~ >
                        hmatel = h2b_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,d,c,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,4); m = excits_buff(jdet,6);
                        ! compute < ab~cd~kl~ | h2a(voov) | cb~d~e~m~l~ >
                        hmatel = -h2b_voov(a,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (3,4,1,6) !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/1,nua/), (/2,nob/), nub, nub, nua, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/3,4,1,6/), nub, nub, nua, nob, nloc, n4abbb)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,d,a,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        ! compute < ab~cd~kl~ | h2a(voov) | ae~b~d~m~l~ >
                        hmatel = -h2b_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,d,c,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,5);
                        ! compute < ab~cd~kl~ | h2a(voov) | ce~b~d~m~l~ >
                        hmatel = h2b_voov(a,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (3,4,1,5) !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/1,nua/), (/1,nob-1/), nub, nub, nua, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/3,4,1,5/), nub, nub, nua, nob, nloc, n4abbb)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,d,a,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,6);
                        ! compute < ab~cd~kl~ | h2a(voov) | ae~b~d~m~l~ >
                        hmatel = h2b_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,d,c,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,6);
                        ! compute < ab~cd~kl~ | h2a(voov) | ce~b~d~m~l~ >
                        hmatel = -h2b_voov(a,m,k,e)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate buffer arrays
                  deallocate(excits_buff,amps_buff)

                  !!! diagram 11: A(bd) h2b(md~el~) * r4b(ab~cekm) 
                  ! copy over excitations
                  allocate(excits_buff(n4abaa,6),amps_buff(n4abaa))
                  excits_buff(:,:) = r4b_excits(:,:)
                  amps_buff(:) = r4b_amps(:)
                  ! allocate new sorting arrays
                  nloc = (nua-1)*(nua-2)/2 * nub * (noa-1) 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,nub,noa))
                  !!! SB: (1,3,2,5) !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/1,nub/), (/1,noa-1/), nua, nua, nub, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,2,5/), nua, nua, nub, noa, nloc, n4abaa)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,4); m = excits_buff(jdet,6);
                        ! compute < ab~cd~kl~ | h2b(ovvo) | ab~cekm >
                        hmatel = h2b_ovvo(m,d,e,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bd)
                     idx = idx_table(a,c,d,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,4); m = excits_buff(jdet,6);
                        ! compute < ab~cd~kl~ | h2b(ovvo) | ad~cekm >
                        hmatel = -h2b_ovvo(m,b,e,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,3,2,6) !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/1,nub/), (/2,noa/), nua, nua, nub, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,3,2,6/), nua, nua, nub, noa, nloc, n4abaa)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,4); m = excits_buff(jdet,5);
                        ! compute < ab~cd~kl~ | h2b(ovvo) | ab~cekm >
                        hmatel = -h2b_ovvo(m,d,e,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bd)
                     idx = idx_table(a,c,d,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,4); m = excits_buff(jdet,5);
                        ! compute < ab~cd~kl~ | h2b(ovvo) | ab~cekm >
                        hmatel = h2b_ovvo(m,b,e,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,4,2,5) !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/1,nub/), (/1,noa-1/), nua, nua, nub, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,4,2,5/), nua, nua, nub, noa, nloc, n4abaa)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                        ! compute < ab~cd~kl~ | h2b(ovvo) | ab~eckm >
                        hmatel = -h2b_ovvo(m,d,e,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bd)
                     idx = idx_table(a,c,d,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,6);
                        ! compute < ab~cd~kl~ | h2b(ovvo) | ad~eckm >
                        hmatel = h2b_ovvo(m,b,e,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,4,2,6) !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-2,nua/), (/1,nub/), (/2,noa/), nua, nua, nub, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,4,2,6/), nua, nua, nub, noa, nloc, n4abaa)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        ! compute < ab~cd~kl~ | h2b(ovvo) | ab~cekm >
                        hmatel = h2b_ovvo(m,d,e,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bd)
                     idx = idx_table(a,c,d,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,3); m = excits_buff(jdet,5);
                        ! compute < ab~cd~kl~ | h2b(ovvo) | ab~cekm >
                        hmatel = -h2b_ovvo(m,b,e,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (3,4,2,5) !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/1,nub/), (/1,noa-1/), nua, nua, nub, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/3,4,2,5/), nua, nua, nub, noa, nloc, n4abaa)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,6);
                        ! compute < ab~cd~kl~ | h2b(ovvo) | eb~ackm >
                        hmatel = h2b_ovvo(m,d,e,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bd)
                     idx = idx_table(a,c,d,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,6);
                        ! compute < ab~cd~kl~ | h2b(ovvo) | ab~cekm >
                        hmatel = -h2b_ovvo(m,b,e,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (3,4,2,6) !!!
                  call get_index_table(idx_table, (/2,nua-1/), (/-1,nua/), (/1,nub/), (/2,noa/), nua, nua, nub, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/3,4,2,6/), nua, nua, nub, noa, nloc, n4abaa)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        ! compute < ab~cd~kl~ | h2b(ovvo) | ab~cekm >
                        hmatel = -h2b_ovvo(m,d,e,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bd)
                     idx = idx_table(a,c,d,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = excits_buff(jdet,1); m = excits_buff(jdet,5);
                        ! compute < ab~cd~kl~ | h2b(ovvo) | ab~cekm >
                        hmatel = h2b_ovvo(m,b,e,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate buffer arrays
                  deallocate(excits_buff,amps_buff)
                  
                  !!! diagram 12: A(bd) h2c(d~m~l~e~) * r4c(ab~ce~km~) 
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2 * (nub-1) * noa 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,nub,noa))
                  !!! SB: (1,3,2,5) !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub-1/), (/1,noa/), nua, nua, nub, noa)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/1,3,2,5/), nua, nua, nub, noa, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,b,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,4); m = r4c_excits(jdet,6);
                        ! compute < ab~cd~kl~ | h2c(voov) | ab~ce~km~ >
                        hmatel = h2c_voov(d,m,l,e)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     ! (bd)
                     idx = idx_table(a,c,d,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,4); m = r4c_excits(jdet,6);
                        ! compute < ab~cd~kl~ | h2c(voov) | ad~ce~km~ >
                        hmatel = -h2c_voov(b,m,l,e)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,3,4,5) !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/2,nub/), (/1,noa/), nua, nua, nub, noa)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/1,3,4,5/), nua, nua, nub, noa, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,d,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,2); m = r4c_excits(jdet,6);
                        ! compute < ab~cd~kl~ | h2c(voov) | ae~cd~km~ >
                        hmatel = h2c_voov(b,m,l,e)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     ! (bd)
                     idx = idx_table(a,c,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,2); m = r4c_excits(jdet,6);
                        ! compute < ab~cd~kl~ | h2c(voov) | ae~cb~km~ >
                        hmatel = -h2c_voov(d,m,l,e)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!! diagram 13: A(bd) -h2b(md~ke~) * r4c(ab~ce~ml~) 
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2 * (nub-1) * nob 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nua,nua,nub,nob))
                  !!! SB: (1,3,2,6) !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub-1/), (/1,nob/), nua, nua, nub, nob)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/1,3,2,6/), nua, nua, nub, nob, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,b,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,4); m = r4c_excits(jdet,5);
                        ! compute < ab~cd~kl~ | h2b(ovov) | ab~ce~ml~ >
                        hmatel = -h2b_ovov(m,d,k,e)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     ! (bd)
                     idx = idx_table(a,c,d,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,4); m = r4c_excits(jdet,5);
                        ! compute < ab~cd~kl~ | h2b(ovov) | ad~ce~ml~ >
                        hmatel = h2b_ovov(m,b,k,e)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (1,3,4,6) !!!
                  call get_index_table(idx_table, (/1,nua-1/), (/-1,nua/), (/2,nub/), (/1,nob/), nua, nua, nub, nob)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/1,3,4,6/), nua, nua, nub, nob, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,d,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,2); m = r4c_excits(jdet,5);
                        ! compute < ab~cd~kl~ | h2b(ovov) | ae~cd~ml~ >
                        hmatel = -h2b_ovov(m,b,k,e)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     ! (bd)
                     idx = idx_table(a,c,b,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,2); m = r4c_excits(jdet,5);
                        ! compute < ab~cd~kl~ | h2b(ovov) | ae~cb~ml~ >
                        hmatel = h2b_ovov(m,d,k,e)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!! diagram 14: A(ac) -h2b(cm~el~) * r4c(ab~ed~km~) 
                  ! allocate new sorting arrays
                  nloc = nub*(nub-1)/2 * (nua-1) * noa 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nua,noa))
                  !!! SB: (2,4,1,5) !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nua-1/), (/1,noa/), nub, nub, nua, noa)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/2,4,1,5/), nub, nub, nua, noa, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,d,a,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,3); m = r4c_excits(jdet,6);
                        ! compute < ab~cd~kl~ | h2b(vovo) | ab~ed~km~ >
                        hmatel = -h2b_vovo(c,m,e,l)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(b,d,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,3); m = r4c_excits(jdet,6);
                        ! compute < ab~cd~kl~ | h2b(vovo) | cb~ed~km~ >
                        hmatel = h2b_vovo(a,m,e,l)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (2,4,3,5) !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/2,nua/), (/1,noa/), nub, nub, nua, noa)
                  call sort4(r4c_excits, r4c_amps, loc_arr, idx_table, (/2,4,3,5/), nub, nub, nua, noa, nloc, n4abab, resid)
                  do idet = 1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,d,c,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,1); m = r4c_excits(jdet,6);
                        ! compute < ab~cd~kl~ | h2b(vovo) | eb~cd~km~ >
                        hmatel = -h2b_vovo(a,m,e,l)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(b,d,a,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4c_excits(jdet,1); m = r4c_excits(jdet,6);
                        ! compute < ab~cd~kl~ | h2b(vovo) | eb~ad~km~ >
                        hmatel = h2b_vovo(c,m,e,l)
                        resid(idet) = resid(idet) + hmatel * r4c_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !
                  ! Moment contributions
                  !
                  do idet=1,n4abab
                     a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                     k = r4c_excits(idet,5); l = r4c_excits(idet,6);

                     res_mm23 = 0.0d0
                     do e=1,nua
                        ! A(ac)A(bd) h2b(cdel)*r3b(ab~ek)
                        res_mm23 = res_mm23 + h2b_vvvo(c,d,e,l)*r3b(a,b,e,k) ! (1)
                        res_mm23 = res_mm23 - h2b_vvvo(a,d,e,l)*r3b(c,b,e,k) ! (ac)
                        res_mm23 = res_mm23 - h2b_vvvo(c,b,e,l)*r3b(a,d,e,k) ! (bd)
                        res_mm23 = res_mm23 + h2b_vvvo(a,b,e,l)*r3b(c,d,e,k) ! (ac)(bd)
                        ! h2a(cake)*r3c(eb~d~l~)
                        res_mm23 = res_mm23 + h2a_vvov(c,a,k,e)*r3c(e,b,d,l) ! (1)
                        ! A(bd) x3b(ab~ce)*t2b(edkl)
                        res_mm23 = res_mm23 + x3b_vvvv(a,b,c,e)*t2b(e,d,k,l) ! (1)
                        res_mm23 = res_mm23 - x3b_vvvv(a,d,c,e)*t2b(e,b,k,l) ! (bd)
                     end do
                     do e=1,nub
                        ! A(ac)A(bd) h2b(cdke)*r3c(ab~e~l~)
                        res_mm23 = res_mm23 + h2b_vvov(c,d,k,e)*r3c(a,b,e,l) ! (1)
                        res_mm23 = res_mm23 - h2b_vvov(a,d,k,e)*r3c(c,b,e,l) ! (ac)
                        res_mm23 = res_mm23 - h2b_vvov(c,b,k,e)*r3c(a,d,e,l) ! (bd)
                        res_mm23 = res_mm23 + h2b_vvov(a,b,k,e)*r3c(c,d,e,l) ! (ac)(bd)
                        ! h2c(dble)*r3b(ae~ck)
                        res_mm23 = res_mm23 + h2c_vvov(d,b,l,e)*r3b(a,e,c,k) ! (1)
                        ! A(ac) x3c(ab~d~e~)*t2b(cekl)
                        res_mm23 = res_mm23 + x3c_vvvv(a,b,d,e)*t2b(c,e,k,l) ! (1)
                        res_mm23 = res_mm23 - x3c_vvvv(c,b,d,e)*t2b(a,e,k,l) ! (ac)
                     end do
                     do m=1,noa
                        ! -A(bd) h2b(mdkl)*r3b(ab~cm)
                        res_mm23 = res_mm23 - h2b_ovoo(m,d,k,l)*r3b(a,b,c,m) ! (1)
                        res_mm23 = res_mm23 + h2b_ovoo(m,b,k,l)*r3b(a,d,c,m) ! (bd)
                        ! -A(ac)A(bd) x3b(ab~mk)*t2b(cdml)
                        res_mm23 = res_mm23 - x3b_vvoo(a,b,m,k)*t2b(c,d,m,l) ! (1)
                        res_mm23 = res_mm23 + x3b_vvoo(c,b,m,k)*t2b(a,d,m,l) ! (ac)
                        res_mm23 = res_mm23 + x3b_vvoo(a,d,m,k)*t2b(c,b,m,l) ! (bd)
                        res_mm23 = res_mm23 - x3b_vvoo(c,d,m,k)*t2b(a,b,m,l) ! (ac)(bd)
                        ! -x3c(mb~d~l~)*t2a(acmk)
                        res_mm23 = res_mm23 - x3c_ovvo(m,b,d,l)*t2a(a,c,m,k) ! (1)
                     end do
                     do m=1,nob
                        ! -A(ac) h2b(cmkl)*r3c(ab~d~m~)
                        res_mm23 = res_mm23 - h2b_vooo(c,m,k,l)*r3c(a,b,d,m) ! (1)
                        res_mm23 = res_mm23 + h2b_vooo(a,m,k,l)*r3c(c,b,d,m) ! (ac)
                        ! -A(ac)A(bd) x3c(ab~m~l~)*t2b(cdkm)
                        res_mm23 = res_mm23 - x3c_vvoo(a,b,m,l)*t2b(c,d,k,m) ! (1)
                        res_mm23 = res_mm23 + x3c_vvoo(c,b,m,l)*t2b(a,d,k,m) ! (ac)
                        res_mm23 = res_mm23 + x3c_vvoo(a,d,m,l)*t2b(c,b,k,m) ! (bd)
                        res_mm23 = res_mm23 - x3c_vvoo(c,d,m,l)*t2b(a,b,k,m) ! (ac)(bd)
                        ! -x3b(am~ck)*t2c(bdml)
                        res_mm23 = res_mm23 - x3b_vovo(a,m,c,k)*t2c(b,d,m,l) ! (1)
                     end do
                     do m=1,noa
                        do n=1,nob
                           ! x2b(mn~)*t2a(acmk)*t2c(b~d~n~l~)
                           res_mm23 = res_mm23 + x2b_oo(m,n)*t2a(a,c,m,k)*t2c(b,d,n,l) ! (1)
                           ! A(ac)A(bd) x2b(mn~)*t2b(ad~ml~)*t2b(cb~kn~)
                           res_mm23 = res_mm23 + x2b_oo(m,n)*t2b(a,d,m,l)*t2b(c,b,k,n) ! (1)
                           res_mm23 = res_mm23 - x2b_oo(m,n)*t2b(c,d,m,l)*t2b(a,b,k,n) ! (ac)
                           res_mm23 = res_mm23 - x2b_oo(m,n)*t2b(a,b,m,l)*t2b(c,d,k,n) ! (bd)
                           res_mm23 = res_mm23 + x2b_oo(m,n)*t2b(c,b,m,l)*t2b(a,d,k,n) ! (ac)(bd)
                        end do
                     end do
                     resid(idet) = resid(idet) + res_mm23
                  end do

              end subroutine build_hr_4c

              subroutine build_hr_4d(resid,&
                                     r3c,&
                                     r4c_amps, r4c_excits,&
                                     r4d_amps, r4d_excits,&
                                     t2b, t2c,&
                                     h1b_oo, h1a_vv, h1b_vv,&
                                     h2b_vvvv, h2b_ovvo, h2b_vovo, h2b_vvvo,&
                                     h2c_vvvv, h2c_oooo, h2c_voov, h2c_vvov, h2c_vooo,&
                                     x3c_vvvv, x3c_ovvo, x3c_vvoo,&
                                     x2b_oo,&
                                     n4abab, n4abbb,&
                                     noa, nua, nob, nub)
                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n4abab, n4abbb
                  !
                  real(kind=8), intent(in) :: r3c(nua,nub,nub,nob)
                  real(kind=8), intent(in) :: t2b(nua,nub,noa,nob)
                  real(kind=8), intent(in) :: t2c(nub,nub,nob,nob)
                  real(kind=8), intent(in) :: r4c_amps(n4abab)
                  integer, intent(in) :: r4c_excits(n4abab,6)
                  ! Input H and X arrays
                  real(kind=8), intent(in) :: h1b_oo(nob,nob)
                  real(kind=8), intent(in) :: h1b_vv(nub,nub)
                  real(kind=8), intent(in) :: h1a_vv(nua,nua)
                  real(kind=8), intent(in) :: h2b_vvvv(nua,nub,nua,nub)
                  real(kind=8), intent(in) :: h2b_ovvo(noa,nub,nua,nob)
                  real(kind=8), intent(in) :: h2b_vovo(nua,nob,nua,nob)
                  real(kind=8), intent(in) :: h2b_vvvo(nua,nub,nua,nob)
                  real(kind=8), intent(in) :: h2c_vvvv(nub,nub,nub,nub)
                  real(kind=8), intent(in) :: h2c_oooo(nob,nob,nob,nob)
                  real(kind=8), intent(in) :: h2c_voov(nub,nob,nob,nub)
                  real(kind=8), intent(in) :: h2c_vvov(nub,nub,nob,nub)
                  real(kind=8), intent(in) :: h2c_vooo(nub,nob,nob,nob)
                  real(kind=8), intent(in) :: x3c_vvvv(nua,nub,nub,nub)
                  real(kind=8), intent(in) :: x3c_ovvo(noa,nub,nub,nob)
                  real(kind=8), intent(in) :: x3c_vvoo(nua,nub,nob,nob)
                  real(kind=8), intent(in) :: x2b_oo(noa,nob)
                  ! Output and Inout variables
                  real(kind=8), intent(out) :: resid(n4abbb)
                  integer, intent(inout) :: r4d_excits(n4abbb,6)
                  !f2py intent(in,out) :: r4d_excits(0:n4abbb-1,0:5)
                  real(kind=8), intent(inout) :: r4d_amps(n4abbb)
                  !f2py intent(in,out) :: r4d_amps(0:n4abbb-1)
                  ! Local variables
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:)
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)
                  real(kind=8) :: r_amp, hmatel, hmatel1, res_mm23
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc
                  real(kind=8) :: ff
                  
                  ! Zero the container that holds H*R
                  resid = 0.0d0

                  !!!! diagram 1: A(d/ac) h1b(d~e~) r4d(ab~c~e~k~l~)
                  !!!! diagram 2: 1/2 A(b/cd) h2c(c~d~e~f~) r4d(ab~e~f~k~l~)
                  ! NOTE: WITHIN THESE LOOPS, H1B(VV) TERMS ARE DOUBLE-COUNTED SO COMPENSATE BY FACTOR OF 1/2
                  ! allocate new sorting arrays
                  nloc = nob*(nob-1)/2*(nub-2)*nua
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nob,nob,nua,nub))
                  !!! SB: (5,6,1,2) !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,nua/), (/1,nub-2/), nob, nob, nua, nub)
                  call sort4(r4d_excits, r4d_amps, loc_arr, idx_table, (/5,6,1,2/), nob, nob, nua, nub, nloc, n4abbb, resid)
                  do idet = 1,n4abbb
                     a = r4d_excits(idet,1); b = r4d_excits(idet,2); c = r4d_excits(idet,3); d = r4d_excits(idet,4)
                     k = r4d_excits(idet,5); l = r4d_excits(idet,6);
                     ! (1)
                     idx = idx_table(k,l,a,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,3); f = r4d_excits(jdet,4);
                        ! compute < ab~c~d~k~l~ | h2c(vvvv) | ab~e~f~k~l~ >
                        hmatel = h2c_vvvv(c,d,e,f)
                        ! compute < klab~cd | h1b(vv) | klab~ef > = A(cd)A(ef) h1b(ce) delta(d,f)
                        hmatel1 = 0.0d0
                        if (d==f) hmatel1 = hmatel1 + h1b_vv(c,e) ! (1)
                        if (c==f) hmatel1 = hmatel1 - h1b_vv(d,e) ! (cd)
                        if (d==e) hmatel1 = hmatel1 - h1b_vv(c,f) ! (ef)
                        if (c==e) hmatel1 = hmatel1 + h1b_vv(d,f) ! (cd)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table(k,l,a,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,3); f = r4d_excits(jdet,4);
                        ! compute < ab~c~d~k~l~ | h2c(vvvv) | ac~e~f~k~l~ >
                        hmatel = -h2c_vvvv(b,d,e,f)
                        ! compute < klab~cd | h1b(vv) | klcb~ef > = -A(ad)A(ef) h1b(ae) delta(d,f)
                        hmatel1 = 0.0d0
                        if (d==f) hmatel1 = hmatel1 - h1b_vv(b,e) ! (1)
                        if (b==f) hmatel1 = hmatel1 + h1b_vv(d,e) ! (ad)
                        if (d==e) hmatel1 = hmatel1 + h1b_vv(b,f) ! (ef)
                        if (b==e) hmatel1 = hmatel1 - h1b_vv(d,f) ! (ad)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (bd)
                     idx = idx_table(k,l,a,d)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,3); f = r4d_excits(jdet,4);
                        ! compute < ab~c~d~k~l~ | h2c(vvvv) | ad~e~f~k~l~ >
                        hmatel = -h2c_vvvv(c,b,e,f)
                        ! compute < klab~cd | h1b(vv) | kldb~ef > = A(ac)A(ef) h1b(ce) delta(a,f)
                        hmatel1 = 0.0d0
                        if (b==f) hmatel1 = hmatel1 - h1b_vv(c,e) ! (1)
                        if (c==f) hmatel1 = hmatel1 + h1b_vv(b,e) ! (cd)
                        if (b==e) hmatel1 = hmatel1 + h1b_vv(c,f) ! (ef)
                        if (c==e) hmatel1 = hmatel1 - h1b_vv(b,f) ! (cd)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (5,6,1,3) !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,nua/), (/2,nub-1/), nob, nob, nua, nub)
                  call sort4(r4d_excits, r4d_amps, loc_arr, idx_table, (/5,6,1,3/), nob, nob, nua, nub, nloc, n4abbb, resid)
                  do idet = 1, n4abbb
                     a = r4d_excits(idet,1); b = r4d_excits(idet,2); c = r4d_excits(idet,3); d = r4d_excits(idet,4)
                     k = r4d_excits(idet,5); l = r4d_excits(idet,6);
                     ! (1)
                     idx = idx_table(k,l,a,c)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,2); f = r4d_excits(jdet,4);
                        ! compute < ab~c~d~k~l~ | h2c(vvvv) | ae~c~f~k~l~ >
                        hmatel = h2c_vvvv(b,d,e,f)
                        hmatel1 = 0.0d0
                        if (b==e) hmatel1 = hmatel1 + h1b_vv(d,f) ! (1)
                        if (b==f) hmatel1 = hmatel1 - h1b_vv(d,e) ! (ad)
                        if (d==e) hmatel1 = hmatel1 - h1b_vv(b,f) ! (ef)
                        if (d==f) hmatel1 = hmatel1 + h1b_vv(b,e) ! (ad)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table(k,l,a,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,2); f = r4d_excits(jdet,4);
                        ! compute < ab~c~d~k~l~ | h2c(vvvv) | ae~c~f~k~l~ >
                        hmatel = -h2c_vvvv(c,d,e,f)
                        hmatel1 = 0.0d0
                        if (c==e) hmatel1 = hmatel1 - h1b_vv(d,f) ! (1)
                        if (c==f) hmatel1 = hmatel1 + h1b_vv(d,e) ! (ad)
                        if (d==e) hmatel1 = hmatel1 + h1b_vv(c,f) ! (ef)
                        if (d==f) hmatel1 = hmatel1 - h1b_vv(c,e) ! (ad)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(k,l,a,d)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,2); f = r4d_excits(jdet,4);
                        ! compute < ab~c~d~k~l~ | h2c(vvvv) | ae~c~f~k~l~ >
                        hmatel = -h2c_vvvv(b,c,e,f)
                        hmatel1 = 0.0d0
                        if (b==e) hmatel1 = hmatel1 - h1b_vv(c,f) ! (1)
                        if (b==f) hmatel1 = hmatel1 + h1b_vv(c,e) ! (ad)
                        if (c==e) hmatel1 = hmatel1 + h1b_vv(b,f) ! (ef)
                        if (c==f) hmatel1 = hmatel1 - h1b_vv(b,e) ! (ad)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (5,6,1,4) !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,nua/), (/3,nub/), nob, nob, nua, nub)
                  call sort4(r4d_excits, r4d_amps, loc_arr, idx_table, (/5,6,1,4/), nob, nob, nua, nub, nloc, n4abbb, resid)
                  do idet = 1, n4abbb
                     a = r4d_excits(idet,1); b = r4d_excits(idet,2); c = r4d_excits(idet,3); d = r4d_excits(idet,4)
                     k = r4d_excits(idet,5); l = r4d_excits(idet,6);
                     ! (1)
                     idx = idx_table(k,l,a,d)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,2); f = r4d_excits(jdet,3);
                        ! compute < klab~cd | h2c(vvvv) | kleb~fd >
                        hmatel = h2c_vvvv(b,c,e,f)
                        hmatel1 = 0.0d0
                        if (b==e) hmatel1 = hmatel1 + h1b_vv(c,f) ! (1)
                        if (b==f) hmatel1 = hmatel1 - h1b_vv(c,e) ! (ac)
                        if (c==e) hmatel1 = hmatel1 - h1b_vv(b,f) ! (ef)
                        if (c==f) hmatel1 = hmatel1 + h1b_vv(b,e) ! (ac)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     ! (bd)
                     idx = idx_table(k,l,a,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,2); f = r4d_excits(jdet,3);
                        ! compute < klab~cd | h2c(vvvv) | kleb~fd >
                        hmatel = -h2c_vvvv(d,c,e,f)
                        hmatel1 = 0.0d0
                        if (d==e) hmatel1 = hmatel1 - h1b_vv(c,f) ! (1)
                        if (d==f) hmatel1 = hmatel1 + h1b_vv(c,e) ! (ac)
                        if (c==e) hmatel1 = hmatel1 + h1b_vv(d,f) ! (ef)
                        if (c==f) hmatel1 = hmatel1 - h1b_vv(d,e) ! (ac)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(k,l,a,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,2); f = r4d_excits(jdet,3);
                        ! compute < klab~cd | h2c(vvvv) | kleb~fd >
                        hmatel = -h2c_vvvv(b,d,e,f)
                        hmatel1 = 0.0d0
                        if (b==e) hmatel1 = hmatel1 - h1b_vv(d,f) ! (1)
                        if (b==f) hmatel1 = hmatel1 + h1b_vv(d,e) ! (ac)
                        if (d==e) hmatel1 = hmatel1 + h1b_vv(b,f) ! (ef)
                        if (d==f) hmatel1 = hmatel1 - h1b_vv(b,e) ! (ac)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  
                  !!!! diagram 3: A(kl) h1b(m~l~) r4d(ab~c~d~k~m~)
                  !!!! diagram 4: 1/2 h2c(m~n~k~l~) r4d(ab~c~d~m~n~)
                  ! allocate new sorting arrays
                  nloc = nub*(nub-1)*(nub-2)/6 * nua
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nub,nua))
                  !!! SB: (2,3,4,1) !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/-1,nub/), (/1,nua/), nub, nub, nub, nua)
                  call sort4(r4d_excits, r4d_amps, loc_arr, idx_table, (/2,3,4,1/), nub, nub, nub, nua, nloc, n4abbb, resid)
                  do idet = 1,n4abbb
                     a = r4d_excits(idet,1); b = r4d_excits(idet,2); c = r4d_excits(idet,3); d = r4d_excits(idet,4)
                     k = r4d_excits(idet,5); l = r4d_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,d,a)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4d_excits(jdet,5); n = r4d_excits(jdet,6);
                        ! compute < klab~cd | h2a(oooo) | mnab~cd >
                        hmatel = h2c_oooo(m,n,k,l)
                        ! compute < klab~cd | h1a(oo) | mnab~cd > = A(kl)A(mn) -delta(m,k) * h1a_oo(n,l)
                        hmatel1 = 0.0d0
                        if (m==k) hmatel1 = hmatel1  - h1b_oo(n,l) ! (1)
                        if (m==l) hmatel1 = hmatel1  + h1b_oo(n,k) ! (kl)
                        if (n==k) hmatel1 = hmatel1  + h1b_oo(m,l) ! (mn)
                        if (n==l) hmatel1 = hmatel1  - h1b_oo(m,k) ! (kl)(mn)
                        hmatel = hmatel + hmatel1
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                 
                  !!!! diagram 3: h1a(ae) * r4d(ab~c~d~k~l~)
                  !!!! diagram 4: A(b/cd) h2b(ab~ef~) * r4d(ef~c~d~k~l~)
                  ff = 1.0d0 / 3.0d0
                  ! allocate new sorting arrays
                  nloc = nob*(nob-1)/2*(nub-1)*(nub-2)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nob,nob,nub,nub))
                  !!! SB: (5,6,3,4) !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/2,nub-1/), (/-1,nub/), nob, nob, nub, nub)
                  call sort4(r4d_excits, r4d_amps, loc_arr, idx_table, (/5,6,3,4/), nob, nob, nub, nub, nloc, n4abbb, resid)
                  do idet = 1,n4abbb
                     a = r4d_excits(idet,1); b = r4d_excits(idet,2); c = r4d_excits(idet,3); d = r4d_excits(idet,4)
                     k = r4d_excits(idet,5); l = r4d_excits(idet,6);
                     ! (1)
                     idx = idx_table(k,l,c,d)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,1); f = r4d_excits(jdet,2);
                        ! compute < k~l~ab~c~d~ | h2b(vvvv) | k~l~ef~c~d~ >
                        hmatel = h2b_vvvv(a,b,e,f)
                        ! compute < klab~cd | h1b(vv) | klef~cd > = delta(a,e) * h1b(b~f~)
                        hmatel1 = 0.0d0
                        if (b==f) hmatel1 = hmatel1 + h1a_vv(a,e)
                        hmatel = hmatel + ff * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table(k,l,b,d)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,1); f = r4d_excits(jdet,2);
                        ! compute < klab~cd | h2b(vvvv) | klef~ad >
                        hmatel = -h2b_vvvv(a,c,e,f)
                        ! compute < klab~cd | h1b(vv) | klef~cd > = delta(c,e) * h1b(b~f~)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 - h1a_vv(a,e)
                        hmatel = hmatel + ff * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (bd), -
                     idx = idx_table(k,l,b,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,1); f = r4d_excits(jdet,2);
                        ! compute < ab~c~d~ | h2b(vvvv) | ef~b~c~ >
                        hmatel = h2b_vvvv(a,d,e,f)
                        ! compute < klab~cd | h1b(vv) | klef~cd > = delta(d,e) * h1b(b~f~)
                        hmatel1 = 0.0d0
                        if (d==f) hmatel1 = hmatel1 + h1a_vv(a,e)
                        hmatel = hmatel + ff * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (5,6,2,3) !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,nub-2/), (/-1,nub-1/), nob, nob, nub, nub)
                  call sort4(r4d_excits, r4d_amps, loc_arr, idx_table, (/5,6,2,3/), nob, nob, nub, nub, nloc, n4abbb, resid)
                  do idet = 1,n4abbb
                     a = r4d_excits(idet,1); b = r4d_excits(idet,2); c = r4d_excits(idet,3); d = r4d_excits(idet,4)
                     k = r4d_excits(idet,5); l = r4d_excits(idet,6);
                     ! (1)
                     idx = idx_table(k,l,b,c)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,1); f = r4d_excits(jdet,4);
                        ! compute < k~l~ab~c~d~ | h2b(vvvv) | k~l~ef~c~d~ >
                        hmatel = h2b_vvvv(a,d,e,f)
                        ! compute < klab~cd | h1b(vv) | klef~cd > = delta(a,e) * h1b(b~f~)
                        hmatel1 = 0.0d0
                        if (d==f) hmatel1 = hmatel1 + h1a_vv(a,e)
                        hmatel = hmatel + ff * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     ! (bd), -
                     idx = idx_table(k,l,c,d)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,1); f = r4d_excits(jdet,4);
                        ! compute < klab~cd | h2b(vvvv) | klef~ad >
                        hmatel = h2b_vvvv(a,b,e,f)
                        ! compute < klab~cd | h1b(vv) | klef~cd > = delta(c,e) * h1b(b~f~)
                        hmatel1 = 0.0d0
                        if (b==f) hmatel1 = hmatel1 + h1a_vv(a,e)
                        hmatel = hmatel + ff * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(k,l,b,d)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,1); f = r4d_excits(jdet,4);
                        ! compute < ab~c~d~ | h2b(vvvv) | ef~b~c~ >
                        hmatel = -h2b_vvvv(a,c,e,f)
                        ! compute < klab~cd | h1b(vv) | klef~cd > = delta(d,e) * h1b(b~f~)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 - h1a_vv(a,e)
                        hmatel = hmatel + ff * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (5,6,2,4) !!!
                  call get_index_table(idx_table, (/1,nob-1/), (/-1,nob/), (/1,nub-2/), (/-2,nub/), nob, nob, nub, nub)
                  call sort4(r4d_excits, r4d_amps, loc_arr, idx_table, (/5,6,2,4/), nob, nob, nub, nub, nloc, n4abbb, resid)
                  do idet = 1,n4abbb
                     a = r4d_excits(idet,1); b = r4d_excits(idet,2); c = r4d_excits(idet,3); d = r4d_excits(idet,4)
                     k = r4d_excits(idet,5); l = r4d_excits(idet,6);
                     ! (1)
                     idx = idx_table(k,l,b,d)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,1); f = r4d_excits(jdet,3);
                        ! compute < k~l~ab~c~d~ | h2b(vvvv) | k~l~ef~c~d~ >
                        hmatel = h2b_vvvv(a,c,e,f)
                        ! compute < klab~cd | h1b(vv) | klef~cd > = delta(a,e) * h1b(b~f~)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 + h1a_vv(a,e)
                        hmatel = hmatel + ff * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     ! (bc)
                     idx = idx_table(k,l,c,d)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,1); f = r4d_excits(jdet,3);
                        ! compute < k~l~ab~c~d~ | h2b(vvvv) | k~l~ef~c~d~ >
                        hmatel = -h2b_vvvv(a,b,e,f)
                        ! compute < klab~cd | h1b(vv) | klef~cd > = delta(a,e) * h1b(b~f~)
                        hmatel1 = 0.0d0
                        if (b==f) hmatel1 = hmatel1 - h1a_vv(a,e)
                        hmatel = hmatel + ff * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(k,l,b,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,1); f = r4d_excits(jdet,3);
                        ! compute < k~l~ab~c~d~ | h2b(vvvv) | k~l~ef~c~d~ >
                        hmatel = -h2b_vvvv(a,d,e,f)
                        ! compute < klab~cd | h1b(vv) | klef~cd > = delta(a,e) * h1b(b~f~)
                        hmatel1 = 0.0d0
                        if (d==f) hmatel1 = hmatel1 - h1a_vv(a,e)
                        hmatel = hmatel + ff * hmatel1
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!! diagram 6: A(d/ac)A(kl) h2c(d~m~l~e~) * r4d(ab~c~e~k~m~)
                  ! allocate new sorting arrays
                  nloc = (nub-1)*(nub-2)/2 * (nob-1) * nua 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nob,nua))
                  !!! SB: (2,3,5,1) !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/1,nob-1/), (/1,nua/), nub, nub, nob, nua)
                  call sort4(r4d_excits, r4d_amps, loc_arr, idx_table, (/2,3,5,1/), nub, nub, nob, nua, nloc, n4abbb, resid)
                  do idet = 1,n4abbb
                     a = r4d_excits(idet,1); b = r4d_excits(idet,2); c = r4d_excits(idet,3); d = r4d_excits(idet,4)
                     k = r4d_excits(idet,5); l = r4d_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,k,a)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4d_excits(jdet,4); n = r4d_excits(jdet,6);
                        ! compute < klab~cd | h2c(voov) | knab~cf >
                        hmatel = h2c_voov(d,n,l,f)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     ! (ad), -
                     idx = idx_table(c,d,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4d_excits(jdet,4); n = r4d_excits(jdet,6);
                        ! compute < klab~cd | h2c(voov) | kndb~cf >
                        hmatel = h2c_voov(b,n,l,f)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(b,d,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4d_excits(jdet,4); n = r4d_excits(jdet,6);
                        ! compute < klab~cd | h2c(voov) | knab~df >
                        hmatel = -h2c_voov(c,n,l,f)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(b,c,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4d_excits(jdet,4); n = r4d_excits(jdet,6);
                        ! compute < klab~cd | h2c(voov) | lnab~cf >
                        hmatel = -h2c_voov(d,n,k,f)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (ad)(kl), -
                     idx = idx_table(c,d,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4d_excits(jdet,4); n = r4d_excits(jdet,6);
                        ! compute < klab~cd | h2c(voov) | lndb~cf >
                        hmatel = -h2c_voov(b,n,k,f)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (cd)(kl)
                     idx = idx_table(b,d,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4d_excits(jdet,4); n = r4d_excits(jdet,6);
                        ! compute < klab~cd | h2c(voov) | lnab~df >
                        hmatel = h2c_voov(c,n,k,f)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (2,4,5,1) !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/1,nob-1/), (/1,nua/), nub, nub, nob, nua)
                  call sort4(r4d_excits, r4d_amps, loc_arr, idx_table, (/2,4,5,1/), nub, nub, nob, nua, nloc, n4abbb, resid)
                  do idet = 1,n4abbb
                     a = r4d_excits(idet,1); b = r4d_excits(idet,2); c = r4d_excits(idet,3); d = r4d_excits(idet,4)
                     k = r4d_excits(idet,5); l = r4d_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,d,k,a)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,3); n = r4d_excits(jdet,6);
                        ! compute < klab~cd | h2c(voov) | knab~ed >
                        hmatel = h2c_voov(c,n,l,e)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(c,d,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,3); n = r4d_excits(jdet,6);
                        ! compute < klab~cd | h2c(voov) | kncb~ed >
                        hmatel = -h2c_voov(b,n,l,e)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(b,c,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,3); n = r4d_excits(jdet,6);
                        ! compute < klab~cd | h2c(voov) | knab~ec >
                        hmatel = -h2c_voov(d,n,l,e)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(b,d,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,3); n = r4d_excits(jdet,6);
                        ! compute < klab~cd | h2c(voov) | lnab~ed >
                        hmatel = -h2c_voov(c,n,k,e)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (ac)(kl)
                     idx = idx_table(c,d,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,3); n = r4d_excits(jdet,6);
                        ! compute < klab~cd | h2c(voov) | lncb~ed >
                        hmatel = h2c_voov(b,n,k,e)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (cd)(kl)
                     idx = idx_table(b,c,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,3); n = r4d_excits(jdet,6);
                        ! compute < klab~cd | h2c(voov) | lnab~ec >
                        hmatel = h2c_voov(d,n,k,e)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (3,4,5,1) !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/1,nob-1/), (/1,nua/), nub, nub, nob, nua)
                  call sort4(r4d_excits, r4d_amps, loc_arr, idx_table, (/3,4,5,1/), nub, nub, nob, nua, nloc, n4abbb, resid)
                  do idet = 1,n4abbb
                     a = r4d_excits(idet,1); b = r4d_excits(idet,2); c = r4d_excits(idet,3); d = r4d_excits(idet,4)
                     k = r4d_excits(idet,5); l = r4d_excits(idet,6);
                     ! (1)
                     idx = idx_table(c,d,k,a)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,2); n = r4d_excits(jdet,6);
                        ! compute < klab~cd | h2c(voov) | kneb~cd >
                        hmatel = h2c_voov(b,n,l,e)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(b,d,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,2); n = r4d_excits(jdet,6);
                        ! compute < klab~cd | h2c(voov) | kneb~ad >
                        hmatel = -h2c_voov(c,n,l,e)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (ad), -
                     idx = idx_table(b,c,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,2); n = r4d_excits(jdet,6);
                        ! compute < klab~cd | h2c(voov) | kneb~ac >
                        hmatel = h2c_voov(d,n,l,e)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(c,d,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,2); n = r4d_excits(jdet,6);
                        ! compute < klab~cd | h2c(voov) | lneb~cd >
                        hmatel = -h2c_voov(b,n,k,e)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (ac)(kl)
                     idx = idx_table(b,d,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,2); n = r4d_excits(jdet,6);
                        ! compute < klab~cd | h2c(voov) | lneb~ad >
                        hmatel = h2c_voov(c,n,k,e)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (ad)(kl), -
                     idx = idx_table(b,c,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,2); n = r4d_excits(jdet,6);
                        ! compute < klab~cd | h2c(voov) | lneb~ac >
                        hmatel = -h2c_voov(d,n,k,e)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (2,3,6,1) !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/2,nob/), (/1,nua/), nub, nub, nob, nua)
                  call sort4(r4d_excits, r4d_amps, loc_arr, idx_table, (/2,3,6,1/), nub, nub, nob, nua, nloc, n4abbb, resid)
                  do idet = 1,n4abbb
                     a = r4d_excits(idet,1); b = r4d_excits(idet,2); c = r4d_excits(idet,3); d = r4d_excits(idet,4)
                     k = r4d_excits(idet,5); l = r4d_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,l,a)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4d_excits(jdet,4); m = r4d_excits(jdet,5);
                        ! compute < klab~cd | h2c(voov) | mlab~cf >
                        hmatel = h2c_voov(d,m,k,f)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     ! (ad), -
                     idx = idx_table(c,d,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4d_excits(jdet,4); m = r4d_excits(jdet,5);
                        ! compute < klab~cd | h2c(voov) | mldb~cf >
                        hmatel = h2c_voov(b,m,k,f)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(b,d,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4d_excits(jdet,4); m = r4d_excits(jdet,5);
                        ! compute < klab~cd | h2c(voov) | mlab~df >
                        hmatel = -h2c_voov(c,m,k,f)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(b,c,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4d_excits(jdet,4); m = r4d_excits(jdet,5);
                        ! compute < klab~cd | h2c(voov) | mkab~cf >
                        hmatel = -h2c_voov(d,m,l,f)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (ad)(kl), -
                     idx = idx_table(c,d,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4d_excits(jdet,4); m = r4d_excits(jdet,5);
                        ! compute < klab~cd | h2c(voov) | mkdb~cf >
                        hmatel = -h2c_voov(b,m,l,f)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (cd)(kl)
                     idx = idx_table(b,d,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4d_excits(jdet,4); m = r4d_excits(jdet,5);
                        ! compute < klab~cd | h2c(voov) | mkab~df >
                        hmatel = h2c_voov(c,m,l,f)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (2,4,6,1) !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-2,nub/), (/2,nob/), (/1,nua/), nub, nub, nob, nua)
                  call sort4(r4d_excits, r4d_amps, loc_arr, idx_table, (/2,4,6,1/), nub, nub, nob, nua, nloc, n4abbb, resid)
                  do idet = 1,n4abbb
                     a = r4d_excits(idet,1); b = r4d_excits(idet,2); c = r4d_excits(idet,3); d = r4d_excits(idet,4)
                     k = r4d_excits(idet,5); l = r4d_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,d,l,a)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,3); m = r4d_excits(jdet,5);
                        ! compute < klab~cd | h2c(voov) | mlab~ed >
                        hmatel = h2c_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(c,d,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,3); m = r4d_excits(jdet,5);
                        ! compute < klab~cd | h2c(voov) | mlcb~ed >
                        hmatel = -h2c_voov(b,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(b,c,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,3); m = r4d_excits(jdet,5);
                        ! compute < klab~cd | h2c(voov) | mlab~ec >
                        hmatel = -h2c_voov(d,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(b,d,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,3); m = r4d_excits(jdet,5);
                        ! compute < klab~cd | h2c(voov) | mkab~ed >
                        hmatel = -h2c_voov(c,m,l,e)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (ac)(kl)
                     idx = idx_table(c,d,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,3); m = r4d_excits(jdet,5);
                        ! compute < klab~cd | h2c(voov) | mkcb~ed >
                        hmatel = h2c_voov(b,m,l,e)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (cd)(kl)
                     idx = idx_table(b,c,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,3); m = r4d_excits(jdet,5);
                        ! compute < klab~cd | h2c(voov) | mkab~ec >
                        hmatel = h2c_voov(d,m,l,e)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (3,4,6,1) !!!
                  call get_index_table(idx_table, (/2,nub-1/), (/-1,nub/), (/2,nob/), (/1,nua/), nub, nub, nob, nua)
                  call sort4(r4d_excits, r4d_amps, loc_arr, idx_table, (/3,4,6,1/), nub, nub, nob, nua, nloc, n4abbb, resid)
                  do idet = 1,n4abbb
                     a = r4d_excits(idet,1); b = r4d_excits(idet,2); c = r4d_excits(idet,3); d = r4d_excits(idet,4)
                     k = r4d_excits(idet,5); l = r4d_excits(idet,6);
                     ! (1)
                     idx = idx_table(c,d,l,a)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,2); m = r4d_excits(jdet,5);
                        ! compute < klab~cd | h2c(voov) | mleb~cd >
                        hmatel = h2c_voov(b,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(b,d,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,2); m = r4d_excits(jdet,5);
                        ! compute < klab~cd | h2c(voov) | mleb~ad >
                        hmatel = -h2c_voov(c,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (ad), -
                     idx = idx_table(b,c,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,2); m = r4d_excits(jdet,5);
                        ! compute < klab~cd | h2c(voov) | mleb~ac >
                        hmatel = h2c_voov(d,m,k,e)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(c,d,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,2); m = r4d_excits(jdet,5);
                        ! compute < klab~cd | h2c(voov) | mkeb~cd >
                        hmatel = -h2c_voov(b,m,l,e)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (ac)(kl)
                     idx = idx_table(b,d,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,2); m = r4d_excits(jdet,5);
                        ! compute < klab~cd | h2c(voov) | mkeb~ad >
                        hmatel = h2c_voov(c,m,l,e)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                     ! (ad)(kl), -
                     idx = idx_table(b,c,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = r4d_excits(jdet,2); m = r4d_excits(jdet,5);
                        ! compute < klab~cd | h2c(voov) | mkeb~ac >
                        hmatel = -h2c_voov(d,m,l,e)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!! diagram 7: A(d/bc)A(kl) h2b(md~el~) * r4c(ab~ec~mk~)
                  ! copy over excitations
                  allocate(excits_buff(n4abab,6),amps_buff(n4abab))
                  excits_buff(:,:) = r4c_excits(:,:)
                  amps_buff(:) = r4c_amps(:)
                  ! allocate new sorting arrays
                  nloc = nub*(nub-1)/2 * (nua-1) * nob 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nua,nob))
                  !!! SB: (2,4,1,6) !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nua-1/), (/1,nob/), nub, nub, nua, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,4,1,6/), nub, nub, nua, nob, nloc, n4abab)
                  do idet = 1,n4abbb
                     a = r4d_excits(idet,1); b = r4d_excits(idet,2); c = r4d_excits(idet,3); d = r4d_excits(idet,4)
                     k = r4d_excits(idet,5); l = r4d_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,a,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                        ! compute < ab~c~d~k~l~ | h2b(voov) | ab~fd~nl~ >
                        hmatel = h2b_ovvo(n,d,f,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bd), -
                     idx = idx_table(c,d,a,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                        ! compute < klab~cd | h2b(voov) | kn~ab~cf~ >
                        hmatel = h2b_ovvo(n,b,f,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(b,d,a,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                        ! compute < klab~cd | h2b(voov) | kn~ab~cf~ >
                        hmatel = -h2b_ovvo(n,c,f,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(b,c,a,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                        ! compute < klab~cd | h2b(voov) | kn~ab~cf~ >
                        hmatel = -h2b_ovvo(n,d,f,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bd)(kl),-
                     idx = idx_table(c,d,a,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                        ! compute < klab~cd | h2b(voov) | kn~ab~cf~ >
                        hmatel = -h2b_ovvo(n,b,f,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (cd)(kl)
                     idx = idx_table(b,d,a,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                        ! compute < klab~cd | h2b(voov) | kn~ab~cf~ >
                        hmatel = h2b_ovvo(n,c,f,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (2,4,3,6) !!!
                  !!! THIS SB IS IDENTICAL TO THE ABOVE ON EXCEPT ALL SIGNS ARE REVERSED !!!
                  call get_index_table(idx_table, (/1,nub-1/), (/-1,nub/), (/2,nua/), (/1,nob/), nub, nub, nua, nob)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/2,4,3,6/), nub, nub, nua, nob, nloc, n4abab)
                  do idet = 1,n4abbb
                     a = r4d_excits(idet,1); b = r4d_excits(idet,2); c = r4d_excits(idet,3); d = r4d_excits(idet,4)
                     k = r4d_excits(idet,5); l = r4d_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,a,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,1); n = excits_buff(jdet,5);
                        ! compute < ab~c~d~k~l~ | h2b(voov) | ab~fd~nl~ >
                        hmatel = -h2b_ovvo(n,d,f,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bd), -
                     idx = idx_table(c,d,a,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,1); n = excits_buff(jdet,5);
                        ! compute < klab~cd | h2b(voov) | kn~ab~cf~ >
                        hmatel = -h2b_ovvo(n,b,f,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (cd)
                     idx = idx_table(b,d,a,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,1); n = excits_buff(jdet,5);
                        ! compute < klab~cd | h2b(voov) | kn~ab~cf~ >
                        hmatel = h2b_ovvo(n,c,f,l)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(b,c,a,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,1); n = excits_buff(jdet,5);
                        ! compute < klab~cd | h2b(voov) | kn~ab~cf~ >
                        hmatel = h2b_ovvo(n,d,f,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bd)(kl),-
                     idx = idx_table(c,d,a,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,1); n = excits_buff(jdet,5);
                        ! compute < klab~cd | h2b(voov) | kn~ab~cf~ >
                        hmatel = h2b_ovvo(n,b,f,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (cd)(kl)
                     idx = idx_table(b,d,a,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = excits_buff(jdet,1); n = excits_buff(jdet,5);
                        ! compute < klab~cd | h2b(voov) | kn~ab~cf~ >
                        hmatel = -h2b_ovvo(n,c,f,k)
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  ! deallocate buffer arrays
                  deallocate(excits_buff,amps_buff)

                  !!! diagram 8: A(kl) -h2b(am~el~) * r4d(eb~c~d~k~m~)
                  ! allocate new sorting arrays
                  nloc = nub*(nub-1)*(nub-2)/6 * (nob-1) 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nub,nub,nub,nob))
                  !!! SB: (2,3,4,5) !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/-1,nub/), (/1,nob-1/), nub, nub, nub, nob)
                  call sort4(r4d_excits, r4d_amps, loc_arr, idx_table, (/2,3,4,5/), nub, nub, nub, nob, nloc, n4abbb, resid)
                  do idet = 1,n4abbb
                     a = r4d_excits(idet,1); b = r4d_excits(idet,2); c = r4d_excits(idet,3); d = r4d_excits(idet,4)
                     k = r4d_excits(idet,5); l = r4d_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,d,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4d_excits(jdet,1); n = r4d_excits(jdet,6);
                        ! compute < ab~cdkl | h2b(vovo) | af~cdkn >
                        hmatel = -h2b_vovo(a,n,f,l)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     ! (kl)
                     idx = idx_table(b,c,d,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4d_excits(jdet,1); n = r4d_excits(jdet,6);
                        ! compute < ab~cdkl | h2b(vovo) | af~cdln >
                        hmatel = h2b_vovo(a,n,f,k)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                  end do
                  !!! SB: (2,3,4,6) !!!
                  call get_index_table(idx_table, (/1,nub-2/), (/-1,nub-1/), (/-1,nub/), (/2,nob/), nub, nub, nub, nob)
                  call sort4(r4d_excits, r4d_amps, loc_arr, idx_table, (/2,3,4,6/), nub, nub, nub, nob, nloc, n4abbb, resid)
                  do idet = 1,n4abbb
                     a = r4d_excits(idet,1); b = r4d_excits(idet,2); c = r4d_excits(idet,3); d = r4d_excits(idet,4)
                     k = r4d_excits(idet,5); l = r4d_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,d,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4d_excits(jdet,1); m = r4d_excits(jdet,5);
                        ! compute < ab~cdkl | h2b(vovo) | af~cdml >
                        hmatel = -h2b_vovo(a,m,f,k)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     ! (kl)
                     idx = idx_table(b,c,d,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = r4d_excits(jdet,1); m = r4d_excits(jdet,5);
                        ! compute < ab~cdkl | h2b(vovo) | af~cdmk >
                        hmatel = h2b_vovo(a,m,f,l)
                        resid(idet) = resid(idet) + hmatel * r4d_amps(jdet)
                     end do
                     end if
                  end do
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !
                  ! Moment contributions
                  !
                  do idet=1,n4abbb
                     a = r4d_excits(idet,1); b = r4d_excits(idet,2); c = r4d_excits(idet,3); d = r4d_excits(idet,4)
                     k = r4d_excits(idet,5); l = r4d_excits(idet,6);
                     res_mm23 = 0.0d0
                     do e=1,nua
                        ! A(c/bd)A(kl) h2b(acek)*r3c(eb~d~l~)
                        res_mm23 = res_mm23 + h2b_vvvo(a,c,e,k)*r3c(e,b,d,l) ! (1)
                        res_mm23 = res_mm23 - h2b_vvvo(a,b,e,k)*r3c(e,c,d,l) ! (bc)
                        res_mm23 = res_mm23 - h2b_vvvo(a,d,e,k)*r3c(e,b,c,l) ! (cd)
                        res_mm23 = res_mm23 - h2b_vvvo(a,c,e,l)*r3c(e,b,d,k) ! (kl)
                        res_mm23 = res_mm23 + h2b_vvvo(a,b,e,l)*r3c(e,c,d,k) ! (bc)(kl)
                        res_mm23 = res_mm23 + h2b_vvvo(a,d,e,l)*r3c(e,b,c,k) ! (cd)(kl)
                     end do
                     do e=1,nub
                        ! A(b/cd)A(kl) h2c(cdke)*r3c(ab~e~l~)
                        res_mm23 = res_mm23 + h2c_vvov(c,d,k,e)*r3c(a,b,e,l) ! (1)
                        res_mm23 = res_mm23 - h2c_vvov(b,d,k,e)*r3c(a,c,e,l) ! (bc)
                        res_mm23 = res_mm23 - h2c_vvov(c,b,k,e)*r3c(a,d,e,l) ! (bd)
                        res_mm23 = res_mm23 - h2c_vvov(c,d,l,e)*r3c(a,b,e,k) ! (kl)
                        res_mm23 = res_mm23 + h2c_vvov(b,d,l,e)*r3c(a,c,e,k) ! (bc)(kl)
                        res_mm23 = res_mm23 + h2c_vvov(c,b,l,e)*r3c(a,d,e,k) ! (bd)(kl)
                        ! A(c/bd) x3c(ab~d~e~)*t2c(cekl)
                        res_mm23 = res_mm23 + x3c_vvvv(a,b,d,e)*t2c(c,e,k,l) ! (1)
                        res_mm23 = res_mm23 - x3c_vvvv(a,c,d,e)*t2c(b,e,k,l) ! (bc)
                        res_mm23 = res_mm23 - x3c_vvvv(a,b,c,e)*t2c(d,e,k,l) ! (cd)
                     end do
                     do m=1,noa
                        ! -A(d/bc)A(kl) x3c(mb~c~k~)*t2b(adml)
                        res_mm23 = res_mm23 - x3c_ovvo(m,b,c,k)*t2b(a,d,m,l) ! (1)
                        res_mm23 = res_mm23 + x3c_ovvo(m,b,d,k)*t2b(a,c,m,l) ! (cd)
                        res_mm23 = res_mm23 + x3c_ovvo(m,d,c,k)*t2b(a,b,m,l) ! (bd)
                        res_mm23 = res_mm23 + x3c_ovvo(m,b,c,l)*t2b(a,d,m,k) ! (kl)
                        res_mm23 = res_mm23 - x3c_ovvo(m,b,d,l)*t2b(a,c,m,k) ! (cd)(kl)
                        res_mm23 = res_mm23 - x3c_ovvo(m,d,c,l)*t2b(a,b,m,k) ! (bd)(kl)
                     end do
                     do m=1,nob
                        ! -A(c/bd) h2c(cmkl)*r3c(ab~d~m~)
                        res_mm23 = res_mm23 - h2c_vooo(c,m,k,l)*r3c(a,b,d,m) ! (1)
                        res_mm23 = res_mm23 + h2c_vooo(b,m,k,l)*r3c(a,c,d,m) ! (bc)
                        res_mm23 = res_mm23 + h2c_vooo(d,m,k,l)*r3c(a,b,c,m) ! (cd)
                        ! -A(b/cd)A(kl) x3c(ab~m~l~)*t2c(cdkm)
                        res_mm23 = res_mm23 - x3c_vvoo(a,b,m,l)*t2c(c,d,k,m) ! (1)
                        res_mm23 = res_mm23 + x3c_vvoo(a,c,m,l)*t2c(b,d,k,m) ! (bc)
                        res_mm23 = res_mm23 + x3c_vvoo(a,d,m,l)*t2c(c,b,k,m) ! (bd)
                        res_mm23 = res_mm23 + x3c_vvoo(a,b,m,k)*t2c(c,d,l,m) ! (kl)
                        res_mm23 = res_mm23 - x3c_vvoo(a,c,m,k)*t2c(b,d,l,m) ! (bc)(kl)
                        res_mm23 = res_mm23 - x3c_vvoo(a,d,m,k)*t2c(c,b,l,m) ! (bd)(kl)
                     end do
                     do m=1,noa
                        do n=1,nob
                           ! A(d/bc)A(kl) x2b(mn~)*t2b(ad~ml~)*t2c(b~c~n~k~)
                           res_mm23 = res_mm23 + x2b_oo(m,n)*t2b(a,d,m,l)*t2c(b,c,n,k) ! (1)
                           res_mm23 = res_mm23 - x2b_oo(m,n)*t2b(a,b,m,l)*t2c(d,c,n,k) ! (bd)
                           res_mm23 = res_mm23 - x2b_oo(m,n)*t2b(a,c,m,l)*t2c(b,d,n,k) ! (cd)
                           res_mm23 = res_mm23 - x2b_oo(m,n)*t2b(a,d,m,k)*t2c(b,c,n,l) ! (kl)
                           res_mm23 = res_mm23 + x2b_oo(m,n)*t2b(a,b,m,k)*t2c(d,c,n,l) ! (bd)(kl)
                           res_mm23 = res_mm23 + x2b_oo(m,n)*t2b(a,c,m,k)*t2c(b,d,n,l) ! (cd)(kl)
                        end do
                     end do
                     resid(idet) = resid(idet) + res_mm23
                  end do
              end subroutine build_hr_4d

      subroutine update_r(r2b,&
                          r3b,r3c,&
                          r4b_amps,r4b_excits,&
                          r4c_amps,r4c_excits,&
                          r4d_amps,r4d_excits,&
                          omega,H1A_oo,H1A_vv,H1B_oo,H1B_vv,&
                          n4abaa,n4abab,n4abbb,noa,nua,nob,nub)

              integer, intent(in) :: noa, nua, nob, nub, n4abaa, n4abab, n4abbb
              real(kind=8), intent(in) :: H1A_oo(1:noa,1:noa), H1A_vv(1:nua,1:nua),&
                                          H1B_oo(1:nob,1:nob), H1B_vv(1:nub,1:nub),&
                                          omega
              integer, intent(in) :: r4b_excits(n4abaa,6)
              integer, intent(in) :: r4c_excits(n4abab,6)
              integer, intent(in) :: r4d_excits(n4abbb,6)
              real(kind=8), intent(inout) :: r2b(1:nua,1:nub)
              !f2py intent(in,out) :: r2b(0:nua-1,0:nub-1)
              real(kind=8), intent(inout) :: r3b(1:nua,1:nub,1:nua,1:noa)
              !f2py intent(in,out) :: r3b(0:nua-1,0:nub-1,0:nua-1,0:noa-1)
              real(kind=8), intent(inout) :: r3c(1:nua,1:nub,1:nub,1:nob)
              !f2py intent(in,out) :: r3c(0:nua-1,0:nub-1,0:nub-1,0:nob-1)
              real(kind=8), intent(inout) :: r4b_amps(1:n4abaa)
              !f2py intent(in,out) :: r4b_amps(0:n4abaa-1)
              real(kind=8), intent(inout) :: r4c_amps(1:n4abab)
              !f2py intent(in,out) :: r4c_amps(0:n4abab-1)
              real(kind=8), intent(inout) :: r4d_amps(1:n4abbb)
              !f2py intent(in,out) :: r4d_amps(0:n4abbb-1)

              integer :: idet, a, b, c, d, k, l
              real(kind=8) :: denom

              do a = 1,nua; do b = 1,nub;
                  denom = H1A_vv(a,a) + H1B_vv(b,b)
                  r2b(a,b) = r2b(a,b)/(omega - denom)
              end do; end do;

              do a = 1,nua; do b = 1,nub; do c = 1,nua; do k = 1,noa;
                  if (a==c) cycle
                  denom = H1A_vv(a,a) + H1B_vv(b,b) + H1A_vv(c,c) - H1A_oo(k,k)
                  r3b(a,b,c,k) = r3b(a,b,c,k)/(omega - denom)
              end do; end do; end do; end do;

              do a = 1,nua; do b = 1,nub; do c = 1,nub; do k = 1,nob;
                  if (b==c) cycle
                  denom = H1A_vv(a,a) + H1B_vv(b,b) + H1B_vv(c,c) - H1B_oo(k,k)
                  r3c(a,b,c,k) = r3c(a,b,c,k)/(omega - denom)
              end do; end do; end do; end do;
              
              do idet=1,n4abaa 
                 a = r4b_excits(idet,1); b = r4b_excits(idet,2); c = r4b_excits(idet,3); d = r4b_excits(idet,4)
                 k = r4b_excits(idet,5); l = r4b_excits(idet,6);

                 denom = H1A_vv(a,a) + H1B_vv(b,b) + H1A_vv(c,c) + H1A_vv(d,d) - H1A_oo(k,k) - H1A_oo(l,l)
                 r4b_amps(idet) = r4b_amps(idet)/(omega - denom)
               end do
           
              do idet=1,n4abab
                 a = r4c_excits(idet,1); b = r4c_excits(idet,2); c = r4c_excits(idet,3); d = r4c_excits(idet,4)
                 k = r4c_excits(idet,5); l = r4c_excits(idet,6);

                 denom = H1A_vv(a,a) + H1B_vv(b,b) + H1A_vv(c,c) + H1B_vv(d,d) - H1A_oo(k,k) - H1B_oo(l,l)
                 r4c_amps(idet) = r4c_amps(idet)/(omega - denom)
              end do
              
              do idet=1,n4abbb
                 a = r4d_excits(idet,1); b = r4d_excits(idet,2); c = r4d_excits(idet,3); d = r4d_excits(idet,4)
                 k = r4d_excits(idet,5); l = r4d_excits(idet,6);

                 denom = H1A_vv(a,a) + H1B_vv(b,b) + H1B_vv(c,c) + H1B_vv(d,d) - H1B_oo(k,k) - H1B_oo(l,l)
                 r4d_amps(idet) = r4d_amps(idet)/(omega - denom)
              end do

      end subroutine update_r

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!! SORTING FUNCTIONS !!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

              subroutine get_index_table(idx_table, rng1, rng2, rng3, rng4, n1, n2, n3, n4)

                    integer, intent(in) :: n1, n2, n3, n4
                    integer, intent(in) :: rng1(2), rng2(2), rng3(2), rng4(2)
      
                    integer, intent(inout) :: idx_table(n1,n2,n3,n4)
      
                    integer :: kout
                    integer :: p, q, r, s
      
                    idx_table = 0
                    ! 5 possible cases. Always organize so that ordered indices appear first.
                    if (rng1(1) < 0 .and. rng2(1) < 0 .and. rng3(1) < 0 .and. rng4(1) < 0) then ! p < q < r < s
                       kout = 1
                       do p = rng1(1), rng1(2)
                          do q = p-rng2(1), rng2(2)
                             do r = q-rng3(1), rng3(2)
                                do s = r-rng4(1), rng4(2)
                                   idx_table(p,q,r,s) = kout
                                   kout = kout + 1
                                end do
                             end do
                          end do
                       end do
                    elseif (rng1(1) > 0 .and. rng2(1) < 0 .and. rng3(1) < 0 .and. rng4(1) > 0) then ! p < q < r, s
                       kout = 1
                       do p = rng1(1), rng1(2)
                          do q = p-rng2(1), rng2(2)
                             do r = q-rng3(1), rng3(2)
                                do s = rng4(1), rng4(2)
                                   idx_table(p,q,r,s) = kout
                                   kout = kout + 1
                                end do
                             end do
                          end do
                       end do
                    elseif (rng1(1) > 0 .and. rng2(1) < 0 .and. rng3(1) > 0 .and. rng4(1) < 0) then ! p < q, r < s
                       kout = 1
                       do p = rng1(1), rng1(2)
                          do q = p-rng2(1), rng2(2)
                             do r = rng3(1), rng3(2)
                                do s = r-rng4(1), rng4(2)
                                   idx_table(p,q,r,s) = kout
                                   kout = kout + 1
                                end do
                             end do
                          end do
                       end do
                    elseif (rng1(1) > 0 .and. rng2(1) < 0 .and. rng3(1) > 0 .and. rng4(1) > 0) then ! p < q, r, s
                       kout = 1
                       do p = rng1(1), rng1(2)
                          do q = p-rng2(1), rng2(2)
                             do r = rng3(1), rng3(2)
                                do s = rng4(1), rng4(2)
                                   idx_table(p,q,r,s) = kout
                                   kout = kout + 1
                                end do
                             end do
                          end do
                       end do
                    else ! p, q, r, s
                       kout = 1
                       do p = rng1(1), rng1(2)
                          do q = rng2(1), rng2(2)
                             do r = rng3(1), rng3(2)
                                do s = rng4(1), rng4(2)
                                   idx_table(p,q,r,s) = kout
                                   kout = kout + 1
                                end do
                             end do
                          end do
                       end do
                    end if

              end subroutine get_index_table

              subroutine sort4(excits, amps, loc_arr, idx_table, idims, n1, n2, n3, n4, nloc, n3p, x1a)

                    integer, intent(in) :: n1, n2, n3, n4, nloc, n3p
                    integer, intent(in) :: idims(4)
                    integer, intent(in) :: idx_table(n1,n2,n3,n4)
      
                    integer, intent(inout) :: loc_arr(2,nloc)
                    integer, intent(inout) :: excits(n3p,6)
                    real(kind=8), intent(inout) :: amps(n3p)
                    real(kind=8), intent(inout), optional :: x1a(n3p)
      
                    integer :: idet
                    integer :: p, q, r, s
                    integer :: p1, q1, r1, s1, p2, q2, r2, s2
                    integer :: pqrs1, pqrs2
                    integer, allocatable :: temp(:), idx(:)
      
                    allocate(temp(n3p),idx(n3p))
                    do idet = 1, n3p
                       p = excits(idet,idims(1)); q = excits(idet,idims(2)); r = excits(idet,idims(3)); s = excits(idet,idims(4))
                       temp(idet) = idx_table(p,q,r,s)
                    end do
                    call argsort(temp, idx)
                    excits = excits(idx,:)
                    amps = amps(idx)
                    if (present(x1a)) x1a = x1a(idx)
                    deallocate(temp,idx)
      
                    loc_arr(1,:) = 1; loc_arr(2,:) = 0;
                    !!! WARNING: THERE IS A MEMORY LEAK HERE! pqrs2 is used below but is not set if n3p <= 1
                    !if (n3p <= 1) print*, "eomccsdt_p_loops >> WARNING: potential memory leakage in sort4 function. pqrs2 set to -1"
                    if (n3p == 1) then
                       if (excits(1,1)==1 .and. excits(1,2)==1 .and. excits(1,3)==1 .and. excits(1,4)==1 .and. excits(1,5)==1 .and. excits(1,6)==1) return
                       p2 = excits(n3p,idims(1)); q2 = excits(n3p,idims(2)); r2 = excits(n3p,idims(3)); s2 = excits(n3p,idims(4))
                       pqrs2 = idx_table(p2,q2,r2,s2)
                    else
                       pqrs2 = -1
                    end if
                    do idet = 1, n3p-1
                       p1 = excits(idet,idims(1));   q1 = excits(idet,idims(2));   r1 = excits(idet,idims(3));   s1 = excits(idet,idims(4))
                       p2 = excits(idet+1,idims(1)); q2 = excits(idet+1,idims(2)); r2 = excits(idet+1,idims(3)); s2 = excits(idet+1,idims(4))
                       pqrs1 = idx_table(p1,q1,r1,s1)
                       pqrs2 = idx_table(p2,q2,r2,s2)
                       if (pqrs1 /= pqrs2) then
                          loc_arr(2,pqrs1) = idet
                          loc_arr(1,pqrs2) = idet+1
                       end if
                    end do
                    !if (n3p > 1) then
                    loc_arr(2,pqrs2) = n3p
                    !end if
              end subroutine sort4

              subroutine argsort(r,d)

                    integer, intent(in), dimension(:) :: r
                    integer, intent(out), dimension(size(r)) :: d
      
                    integer, dimension(size(r)) :: il
      
                    integer :: stepsize
                    integer :: i, j, n, left, k, ksize
      
                    n = size(r)
      
                    do i=1,n
                       d(i)=i
                    end do
      
                    if (n==1) return
      
                    stepsize = 1
                    do while (stepsize < n)
                       do left = 1, n-stepsize,stepsize*2
                          i = left
                          j = left+stepsize
                          ksize = min(stepsize*2,n-left+1)
                          k=1
      
                          do while (i < left+stepsize .and. j < left+ksize)
                             if (r(d(i)) < r(d(j))) then
                                il(k) = d(i)
                                i = i+1
                                k = k+1
                             else
                                il(k) = d(j)
                                j = j+1
                                k = k+1
                             endif
                          enddo
      
                          if (i < left+stepsize) then
                             ! fill up remaining from left
                             il(k:ksize) = d(i:left+stepsize-1)
                          else
                             ! fill up remaining from right
                             il(k:ksize) = d(j:left+ksize-1)
                          endif
                          d(left:left+ksize-1) = il(1:ksize)
                       end do
                       stepsize = stepsize*2
                    end do

              end subroutine argsort

end module deaeom4_p_loops
