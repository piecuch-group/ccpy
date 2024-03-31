module ipeom3_p_intermediates

        implicit none
        
        contains

              subroutine add_r3_x2a_ovv(x2a_ovv,&
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

                  real(kind=8), intent(inout) :: x2a_ovv(noa,nua,nua)
                  !f2py intent(in,out) :: x2a_ovv(0:noa-1,0:nua-1,0:nua-1)

                  real(kind=8) :: r_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet
                      
                  do idet = 1,n3aaa
                     r_amp = r3a_amps(idet)
                     ! x2a(ibe) <- A(i/mn)A(bf) -h2a(mnef)*r3a(bfimn)
                     b = r3a_excits(idet,1); f = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); m = r3a_excits(idet,4); n = r3a_excits(idet,5);
                     x2a_ovv(i,b,:) = x2a_ovv(i,b,:) - h2a_oovv(m,n,:,f) * r_amp ! (1)
                     x2a_ovv(m,b,:) = x2a_ovv(m,b,:) + h2a_oovv(i,n,:,f) * r_amp ! (im)
                     x2a_ovv(n,b,:) = x2a_ovv(n,b,:) + h2a_oovv(m,i,:,f) * r_amp ! (in)
                     x2a_ovv(i,f,:) = x2a_ovv(i,f,:) + h2a_oovv(m,n,:,b) * r_amp ! (bf)
                     x2a_ovv(m,f,:) = x2a_ovv(m,f,:) - h2a_oovv(i,n,:,b) * r_amp ! (im)(bf)
                     x2a_ovv(n,f,:) = x2a_ovv(n,f,:) - h2a_oovv(m,i,:,b) * r_amp ! (in)(bf)
                  end do
                      
                  do idet = 1,n3aab
                     r_amp = r3b_amps(idet)
                     ! x2a(ibe) <- A(im) -h2b(mnef)*r3b(bfimn)
                     b = r3b_excits(idet,1); f = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); m = r3b_excits(idet,4); n = r3b_excits(idet,5);
                     x2a_ovv(i,b,:) = x2a_ovv(i,b,:) - h2b_oovv(m,n,:,f) * r_amp ! (1)
                     x2a_ovv(m,b,:) = x2a_ovv(m,b,:) + h2b_oovv(i,n,:,f) * r_amp ! (im)
                  end do

              end subroutine add_r3_x2a_ovv

              subroutine add_r3_x2a_ooo(x2a_ooo,&
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

                  real(kind=8), intent(inout) :: x2a_ooo(noa,noa,noa)
                  !f2py intent(in,out) :: x2a_ooo(0:noa-1,0:noa-1,0:noa-1)

                  real(kind=8) :: r_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  ! divide incoming antisymmetrized quantity by half to cancel redundant antisymmetrization
                  ! later on; be sure to only update permutationally unique elements!                  
                  x2a_ooo = 0.5d0*x2a_ooo 
                  do idet = 1,n3aaa
                     r_amp = r3a_amps(idet)
                     ! x2a_ooo(imj) <- A(ij) [ A(n/ij) h2a(mnef)*r3a(efijn) ]
                     e = r3a_excits(idet,1); f = r3a_excits(idet,2);
                     i = r3a_excits(idet,3); j = r3a_excits(idet,4); n = r3a_excits(idet,5);
                     x2a_ooo(i,:,j) = x2a_ooo(i,:,j) + h2a_oovv(:,n,e,f) * r_amp ! (1)
                     x2a_ooo(n,:,j) = x2a_ooo(n,:,j) - h2a_oovv(:,i,e,f) * r_amp ! (in)
                     x2a_ooo(i,:,n) = x2a_ooo(i,:,n) - h2a_oovv(:,j,e,f) * r_amp ! (jn)
                  end do
                      
                  do idet = 1,n3aab
                     r_amp = r3b_amps(idet)
                     ! x2a_ooo(imj) <- A(ij) [ h2b(mnef)*r3b(efijn) ]
                     e = r3b_excits(idet,1); f = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); j = r3b_excits(idet,4); n = r3b_excits(idet,5);
                     x2a_ooo(i,:,j) = x2a_ooo(i,:,j) + h2b_oovv(:,n,e,f) * r_amp ! (1)
                  end do

                  ! apply the common A(ij) antisymmetrizer
                  do i = 1,noa
                     do m = 1,noa
                        do j = i+1,noa
                           x2a_ooo(i,m,j) = x2a_ooo(i,m,j) - x2a_ooo(j,m,i)
                        end do
                     end do
                  end do
                  ! explicitly antisymmetrize
                  do i = 1,noa
                     do j = i+1,noa
                        x2a_ooo(j,:,i) = -x2a_ooo(i,:,j)
                     end do
                  end do

              end subroutine add_r3_x2a_ooo
           
              subroutine add_r3_x2b_vvo(x2b_vvo,&
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

                  real(kind=8), intent(inout) :: x2b_vvo(nua,nub,nob)
                  !f2py intent(in,out) :: x2b_vvo(0:nua-1,0:nub-1,0:nob-1)

                  real(kind=8) :: r_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet
                      
                  do idet = 1,n3aab
                     r_amp = r3b_amps(idet)
                     ! x2b_vvo(ebj) <- -h2a(mnef)*r3b(fbmnj)
                     f = r3b_excits(idet,1); b = r3b_excits(idet,2);
                     m = r3b_excits(idet,3); n = r3b_excits(idet,4); j = r3b_excits(idet,5);
                     x2b_vvo(:,b,j) = x2b_vvo(:,b,j) - h2a_oovv(m,n,:,f) * r_amp ! (1)
                  end do
                      
                  do idet = 1,n3abb
                     r_amp = r3c_amps(idet)
                     ! x2b_vvo(ebj) <- A(jn)A(bf) -h2b(mnef)*r3c(bfmjn)
                     b = r3c_excits(idet,1); f = r3c_excits(idet,2);
                     m = r3c_excits(idet,3); j = r3c_excits(idet,4); n = r3c_excits(idet,5);
                     x2b_vvo(:,b,j) = x2b_vvo(:,b,j) - h2b_oovv(m,n,:,f) * r_amp ! (1)
                     x2b_vvo(:,f,j) = x2b_vvo(:,f,j) + h2b_oovv(m,n,:,b) * r_amp ! (bf)
                     x2b_vvo(:,b,n) = x2b_vvo(:,b,n) + h2b_oovv(m,j,:,f) * r_amp ! (jn)
                     x2b_vvo(:,f,n) = x2b_vvo(:,f,n) - h2b_oovv(m,j,:,b) * r_amp ! (bf)(jn)
                  end do

              end subroutine add_r3_x2b_vvo

              subroutine add_r3_x2b_ovv(x2b_ovv,&
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

                  real(kind=8), intent(inout) :: x2b_ovv(noa,nub,nub)
                  !f2py intent(in,out) :: x2b_ovv(0:noa-1,0:nub-1,0:nub-1)

                  real(kind=8) :: r_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet
                      
                  do idet = 1,n3aab
                     r_amp = r3b_amps(idet)
                     ! x2b_ovv(ibe) <- A(in) -h2b(nmfe)*r3b(fbinm)
                     f = r3b_excits(idet,1); b = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); n = r3b_excits(idet,4); m = r3b_excits(idet,5);
                     x2b_ovv(i,b,:) = x2b_ovv(i,b,:) - h2b_oovv(n,m,f,:) * r_amp ! (1)
                     x2b_ovv(n,b,:) = x2b_ovv(n,b,:) + h2b_oovv(i,m,f,:) * r_amp ! (in)
                  end do
                      
                  do idet = 1,n3abb
                     r_amp = r3c_amps(idet)
                     ! x2b_ovv(ibe) <- A(bf) -h2c(mnef)*r3c(bfimn)
                     b = r3c_excits(idet,1); f = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); m = r3c_excits(idet,4); n = r3c_excits(idet,5);
                     x2b_ovv(i,b,:) = x2b_ovv(i,b,:) - h2c_oovv(m,n,:,f) * r_amp ! (1)
                     x2b_ovv(i,f,:) = x2b_ovv(i,f,:) + h2c_oovv(m,n,:,b) * r_amp ! (bf)
                  end do

              end subroutine add_r3_x2b_ovv

              subroutine add_r3_x2b_ooo(x2b_ooo,&
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

                  real(kind=8), intent(inout) :: x2b_ooo(noa,nob,nob)
                  !f2py intent(in,out) :: x2b_ooo(0:noa-1,0:nob-1,0:nob-1)

                  real(kind=8) :: r_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet
                      
                  do idet = 1,n3aab
                     r_amp = r3b_amps(idet)
                     ! x2b_ooo(imj) <- A(in) h2b(nmfe)*r3b(feinj)
                     f = r3b_excits(idet,1); e = r3b_excits(idet,2);
                     i = r3b_excits(idet,3); n = r3b_excits(idet,4); j = r3b_excits(idet,5);
                     x2b_ooo(i,:,j) = x2b_ooo(i,:,j) + h2b_oovv(n,:,f,e) * r_amp ! (1)
                     x2b_ooo(n,:,j) = x2b_ooo(n,:,j) - h2b_oovv(i,:,f,e) * r_amp ! (in)
                  end do
                      
                  do idet = 1,n3abb
                     r_amp = r3c_amps(idet)
                     ! x2b_ooo(imj) <- A(jn) h2c(mnef)*r3c(efijn)
                     e = r3c_excits(idet,1); f = r3c_excits(idet,2);
                     i = r3c_excits(idet,3); j = r3c_excits(idet,4); n = r3c_excits(idet,5);
                     x2b_ooo(i,:,j) = x2b_ooo(i,:,j) + h2c_oovv(:,n,e,f) * r_amp ! (1)
                     x2b_ooo(i,:,n) = x2b_ooo(i,:,n) - h2c_oovv(:,j,e,f) * r_amp ! (jn)
                  end do

              end subroutine add_r3_x2b_ooo

end module ipeom3_p_intermediates
