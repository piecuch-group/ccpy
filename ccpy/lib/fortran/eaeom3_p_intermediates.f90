module eaeom3_p_intermediates

        implicit none
        
        contains

              subroutine add_r3_x2a_voo(x2a_voo,&
                                        r3a_amps, r3a_excits,&
                                        r3b_amps, r3b_excits,&
                                        h2a_oovv, h2b_oovv,&
                                        n3aaa, n3aab,&
                                        noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa, n3aab

                  integer, intent(in) :: r3a_excits(n3aaa,5)
                  real(kind=8), intent(in) :: r3a_amps(n3aaa)
                  integer, intent(in) :: r3b_excits(n3aab,5)
                  real(kind=8), intent(in) :: r3b_amps(n3aab)

                  real(kind=8), intent(in) :: h2a_oovv(noa,noa,nua,nua)
                  real(kind=8), intent(in) :: h2b_oovv(noa,nob,nua,nub)

                  real(kind=8), intent(inout) :: x2a_voo(nua,noa,noa)
                  !f2py intent(in,out) :: x2a_voo(0:nua-1,0:noa-1,0:noa-1)

                  real(kind=8) :: r_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet
                      
                  do idet = 1,n3aaa
                     r_amp = r3a_amps(idet)
                     ! x2a(amj) <- A(a/ef)A(jn) h2a(mnef)*r3a(aefjn)
                     a = r3a_excits(idet,1); e = r3a_excits(idet,2); f = r3a_excits(idet,3);
                     j = r3a_excits(idet,4); n = r3a_excits(idet,5);
                     x2a_voo(a,:,j) = x2a_voo(a,:,j) + h2a_oovv(:,n,e,f) * r_amp ! (1)
                     x2a_voo(e,:,j) = x2a_voo(e,:,j) - h2a_oovv(:,n,a,f) * r_amp ! (ae)
                     x2a_voo(f,:,j) = x2a_voo(f,:,j) - h2a_oovv(:,n,e,a) * r_amp ! (af)
                     x2a_voo(a,:,n) = x2a_voo(a,:,n) - h2a_oovv(:,j,e,f) * r_amp ! (jn)
                     x2a_voo(e,:,n) = x2a_voo(e,:,n) + h2a_oovv(:,j,a,f) * r_amp ! (ae)(jn)
                     x2a_voo(f,:,n) = x2a_voo(f,:,n) + h2a_oovv(:,j,e,a) * r_amp ! (af)(jn)
                  end do
                      
                  do idet = 1,n3aab
                     r_amp = r3b_amps(idet)
                     ! x2a(amj) <- A(ae) h2b(mnef)*r3b(aefjn)
                     a = r3b_excits(idet,1); e = r3b_excits(idet,2); f = r3b_excits(idet,3);
                     j = r3b_excits(idet,4); n = r3b_excits(idet,5);
                     x2a_voo(a,:,j) = x2a_voo(a,:,j) + h2b_oovv(:,n,e,f) * r_amp ! (1)
                     x2a_voo(e,:,j) = x2a_voo(e,:,j) - h2b_oovv(:,n,a,f) * r_amp ! (ae)
                  end do

              end subroutine add_r3_x2a_voo

              subroutine add_r3_x2a_vvv(x2a_vvv,&
                                        r3a_amps, r3a_excits,&
                                        r3b_amps, r3b_excits,&
                                        h2a_oovv, h2b_oovv,&
                                        n3aaa, n3aab,&
                                        noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa, n3aab

                  integer, intent(in) :: r3a_excits(n3aaa,5)
                  real(kind=8), intent(in) :: r3a_amps(n3aaa)
                  integer, intent(in) :: r3b_excits(n3aab,5)
                  real(kind=8), intent(in) :: r3b_amps(n3aab)

                  real(kind=8), intent(in) :: h2a_oovv(noa,noa,nua,nua)
                  real(kind=8), intent(in) :: h2b_oovv(noa,nob,nua,nub)

                  real(kind=8), intent(inout) :: x2a_vvv(nua,nua,nua)
                  !f2py intent(in,out) :: x2a_vvv(0:nua-1,0:nua-1,0:nua-1)

                  real(kind=8) :: r_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  ! divide incoming antisymmetrized quantity by half to cancel redundant antisymmetrization
                  ! later on; be sure to only update permutationally unique elements!                  
                  x2a_vvv = 0.5d0*x2a_vvv    
                  do idet = 1,n3aaa
                     r_amp = r3a_amps(idet)
                     ! x2a(abe) <- A(ab) [A(f/ab) -h2a(mnef)*r3a(abfmn)]
                     a = r3a_excits(idet,1); b = r3a_excits(idet,2); f = r3a_excits(idet,3);
                     m = r3a_excits(idet,4); n = r3a_excits(idet,5);
                     x2a_vvv(a,b,:) = x2a_vvv(a,b,:) - h2a_oovv(m,n,:,f) * r_amp ! (1)
                     x2a_vvv(f,b,:) = x2a_vvv(f,b,:) + h2a_oovv(m,n,:,a) * r_amp ! (af)
                     x2a_vvv(a,f,:) = x2a_vvv(a,f,:) + h2a_oovv(m,n,:,b) * r_amp ! (bf)
                  end do
                      
                  do idet = 1,n3aab
                     r_amp = r3b_amps(idet)
                     ! x2a(abe) <- A(ab) [-h2b(mnef)*r3b(abfmn)]
                     a = r3b_excits(idet,1); b = r3b_excits(idet,2); f = r3b_excits(idet,3);
                     m = r3b_excits(idet,4); n = r3b_excits(idet,5);
                     x2a_vvv(a,b,:) = x2a_vvv(a,b,:) - h2b_oovv(m,n,:,f) * r_amp ! (1)
                  end do

                  ! apply the common A(ab) antisymmetrizer
                  do e = 1,nua
                     do a = 1,nua
                        do b = a+1,nua
                           x2a_vvv(a,b,e) = x2a_vvv(a,b,e) - x2a_vvv(b,a,e)
                        end do
                     end do
                  end do
                  ! explicitly antisymmetrize
                  do a = 1,nua
                     do b = a+1,nua
                        x2a_vvv(b,a,:) = -x2a_vvv(a,b,:)
                     end do
                  end do

              end subroutine add_r3_x2a_vvv
           
              subroutine add_r3_x2b_ovo(x2b_ovo,&
                                        r3b_amps, r3b_excits,&
                                        r3c_amps, r3c_excits,&
                                        h2a_oovv, h2b_oovv,&
                                        n3aab, n3abb,&
                                        noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab, n3abb

                  integer, intent(in) :: r3b_excits(n3aab,5)
                  real(kind=8), intent(in) :: r3b_amps(n3aab)
                  integer, intent(in) :: r3c_excits(n3abb,5)
                  real(kind=8), intent(in) :: r3c_amps(n3abb)

                  real(kind=8), intent(in) :: h2a_oovv(noa,noa,nua,nua)
                  real(kind=8), intent(in) :: h2b_oovv(noa,nob,nua,nub)

                  real(kind=8), intent(inout) :: x2b_ovo(noa,nub,nob)
                  !f2py intent(in,out) :: x2b_ovo(0:noa-1,0:nub-1,0:nob-1)

                  real(kind=8) :: r_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet
                      
                  do idet = 1,n3aab
                     r_amp = r3b_amps(idet)
                     ! x2b(mb~j~) <- h2a(mnef)*r3b(efb~nj~)
                     e = r3b_excits(idet,1); f = r3b_excits(idet,2); b = r3b_excits(idet,3);
                     n = r3b_excits(idet,4); j = r3b_excits(idet,5);
                     x2b_ovo(:,b,j) = x2b_ovo(:,b,j) + h2a_oovv(:,n,e,f) * r_amp ! (1)
                  end do
                      
                  do idet = 1,n3abb
                     r_amp = r3c_amps(idet)
                     ! x2b(mb~j~) <- A(bf)A(jn) h2b(mnef)*r3c(efbnj)
                     e = r3c_excits(idet,1); f = r3c_excits(idet,2); b = r3c_excits(idet,3);
                     n = r3c_excits(idet,4); j = r3c_excits(idet,5);
                     x2b_ovo(:,b,j) = x2b_ovo(:,b,j) + h2b_oovv(:,n,e,f) * r_amp ! (1)
                     x2b_ovo(:,f,j) = x2b_ovo(:,f,j) - h2b_oovv(:,n,e,b) * r_amp ! (bf)
                     x2b_ovo(:,b,n) = x2b_ovo(:,b,n) - h2b_oovv(:,j,e,f) * r_amp ! (jn)
                     x2b_ovo(:,f,n) = x2b_ovo(:,f,n) + h2b_oovv(:,j,e,b) * r_amp ! (bf)(jn)
                  end do

              end subroutine add_r3_x2b_ovo

              subroutine add_r3_x2b_voo(x2b_voo,&
                                        r3b_amps, r3b_excits,&
                                        r3c_amps, r3c_excits,&
                                        h2b_oovv, h2c_oovv,&
                                        n3aab, n3abb,&
                                        noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab, n3abb

                  integer, intent(in) :: r3b_excits(n3aab,5)
                  real(kind=8), intent(in) :: r3b_amps(n3aab)
                  integer, intent(in) :: r3c_excits(n3abb,5)
                  real(kind=8), intent(in) :: r3c_amps(n3abb)

                  real(kind=8), intent(in) :: h2b_oovv(noa,nob,nua,nub)
                  real(kind=8), intent(in) :: h2c_oovv(nob,nob,nub,nub)

                  real(kind=8), intent(inout) :: x2b_voo(nua,nob,nob)
                  !f2py intent(in,out) :: x2b_voo(0:nua-1,0:nob-1,0:nob-1)

                  real(kind=8) :: r_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet
                      
                  do idet = 1,n3aab
                     r_amp = r3b_amps(idet)
                     ! x2b_voo(am~j~) <- A(af) h2b(nmfe)*r3b(afenj)
                     a = r3b_excits(idet,1); f = r3b_excits(idet,2); e = r3b_excits(idet,3);
                     n = r3b_excits(idet,4); j = r3b_excits(idet,5);
                     x2b_voo(a,:,j) = x2b_voo(a,:,j) + h2b_oovv(n,:,f,e) * r_amp ! (1)
                     x2b_voo(f,:,j) = x2b_voo(f,:,j) - h2b_oovv(n,:,a,e) * r_amp ! (af)
                  end do
                      
                  do idet = 1,n3abb
                     r_amp = r3c_amps(idet)
                     ! x2b_voo(am~j~) <- A(jn) h2c(mnef)*r3c(aefjn)
                     a = r3c_excits(idet,1); e = r3c_excits(idet,2); f = r3c_excits(idet,3);
                     j = r3c_excits(idet,4); n = r3c_excits(idet,5);
                     x2b_voo(a,:,j) = x2b_voo(a,:,j) + h2c_oovv(:,n,e,f) * r_amp ! (1)
                     x2b_voo(a,:,n) = x2b_voo(a,:,n) - h2c_oovv(:,j,e,f) * r_amp ! (jn)
                  end do

              end subroutine add_r3_x2b_voo

              subroutine add_r3_x2b_vvv(x2b_vvv,&
                                        r3b_amps, r3b_excits,&
                                        r3c_amps, r3c_excits,&
                                        h2b_oovv, h2c_oovv,&
                                        n3aab, n3abb,&
                                        noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab, n3abb

                  integer, intent(in) :: r3b_excits(n3aab,5)
                  real(kind=8), intent(in) :: r3b_amps(n3aab)
                  integer, intent(in) :: r3c_excits(n3abb,5)
                  real(kind=8), intent(in) :: r3c_amps(n3abb)

                  real(kind=8), intent(in) :: h2b_oovv(noa,nob,nua,nub)
                  real(kind=8), intent(in) :: h2c_oovv(nob,nob,nub,nub)

                  real(kind=8), intent(inout) :: x2b_vvv(nua,nub,nub)
                  !f2py intent(in,out) :: x2b_vvv(0:nua-1,0:nub-1,0:nub-1)

                  real(kind=8) :: r_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet
                      
                  do idet = 1,n3aab
                     r_amp = r3b_amps(idet)
                     ! x2b_vvv(ab~e~) <- A(af) -h2b(nmfe)*r3b(afbnm)
                     a = r3b_excits(idet,1); f = r3b_excits(idet,2); b = r3b_excits(idet,3);
                     n = r3b_excits(idet,4); m = r3b_excits(idet,5);
                     x2b_vvv(a,b,:) = x2b_vvv(a,b,:) - h2b_oovv(n,m,f,:) * r_amp ! (1)
                     x2b_vvv(f,b,:) = x2b_vvv(f,b,:) + h2b_oovv(n,m,a,:) * r_amp ! (af)
                  end do
                      
                  do idet = 1,n3abb
                     r_amp = r3c_amps(idet)
                     ! x2b_vvv(ab~e~) <- A(bf) -h2c(mnef)*r3c(abfmn)
                     a = r3c_excits(idet,1); b = r3c_excits(idet,2); f = r3c_excits(idet,3);
                     m = r3c_excits(idet,4); n = r3c_excits(idet,5);
                     x2b_vvv(a,b,:) = x2b_vvv(a,b,:) - h2c_oovv(m,n,:,f) * r_amp ! (1)
                     x2b_vvv(a,f,:) = x2b_vvv(a,f,:) + h2c_oovv(m,n,:,b) * r_amp ! (bf)
                  end do

              end subroutine add_r3_x2b_vvv

end module eaeom3_p_intermediates
