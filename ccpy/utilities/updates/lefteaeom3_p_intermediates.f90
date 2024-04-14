module lefteaeom3_p_intermediates

        implicit none
        
        contains

              subroutine get_x2a_ovo(x2a_ovo,&
                                     l3a_amps, l3a_excits,&
                                     l3b_amps, l3b_excits,&
                                     t2a, t2b,&
                                     n3aaa, n3aab,&
                                     noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa, n3aab

                  integer, intent(in) :: l3a_excits(n3aaa,5)
                  real(kind=8), intent(in) :: l3a_amps(n3aaa)
                  integer, intent(in) :: l3b_excits(n3aab,5)
                  real(kind=8), intent(in) :: l3b_amps(n3aab)

                  real(kind=8), intent(in) :: t2a(nua,nua,noa,noa)
                  real(kind=8), intent(in) :: t2b(nua,nub,noa,nob)

                  real(kind=8), intent(out) :: x2a_ovo(noa,nua,noa)

                  real(kind=8) :: l_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet
                       
                  x2a_ovo = 0.0d0 
                  do idet = 1,n3aaa
                     l_amp = l3a_amps(idet)
                     ! x2a(ibj) <- A(b/ef)A(jn) l3a(ebfjn)*t2a(efin)
                     e = l3a_excits(idet,1); b = l3a_excits(idet,2); f = l3a_excits(idet,3);
                     j = l3a_excits(idet,4); n = l3a_excits(idet,5);
                     x2a_ovo(:,b,j) = x2a_ovo(:,b,j) + t2a(e,f,:,n)*l_amp ! (1)
                     x2a_ovo(:,e,j) = x2a_ovo(:,e,j) - t2a(b,f,:,n)*l_amp ! (be)
                     x2a_ovo(:,f,j) = x2a_ovo(:,f,j) - t2a(e,b,:,n)*l_amp ! (bf)
                     x2a_ovo(:,b,n) = x2a_ovo(:,b,n) - t2a(e,f,:,j)*l_amp ! (jn)
                     x2a_ovo(:,e,n) = x2a_ovo(:,e,n) + t2a(b,f,:,j)*l_amp ! (be)(jn)
                     x2a_ovo(:,f,n) = x2a_ovo(:,f,n) + t2a(e,b,:,j)*l_amp ! (bf)(jn)
                  end do
                      
                  do idet = 1,n3aab
                     l_amp = l3b_amps(idet)
                     ! x2a(ibj) <- A(be) l3b(ebfjn)*t2b(efin)
                     e = l3b_excits(idet,1); b = l3b_excits(idet,2); f = l3b_excits(idet,3);
                     j = l3b_excits(idet,4); n = l3b_excits(idet,5);
                     x2a_ovo(:,b,j) = x2a_ovo(:,b,j) + t2b(e,f,:,n)*l_amp ! (1)
                     x2a_ovo(:,e,j) = x2a_ovo(:,e,j) - t2b(b,f,:,n)*l_amp ! (be)
                  end do

              end subroutine get_x2a_ovo

              subroutine get_x2a_vvv(x2a_vvv,&
                                     l3a_amps, l3a_excits,&
                                     l3b_amps, l3b_excits,&
                                     t2a, t2b,&
                                     n3aaa, n3aab,&
                                     noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa, n3aab

                  integer, intent(in) :: l3a_excits(n3aaa,5)
                  real(kind=8), intent(in) :: l3a_amps(n3aaa)
                  integer, intent(in) :: l3b_excits(n3aab,5)
                  real(kind=8), intent(in) :: l3b_amps(n3aab)

                  real(kind=8), intent(in) :: t2a(nua,nua,noa,noa)
                  real(kind=8), intent(in) :: t2b(nua,nub,noa,nob)

                  real(kind=8), intent(out) :: x2a_vvv(nua,nua,nua)

                  real(kind=8) :: l_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  x2a_vvv = 0.0d0    
                  do idet = 1,n3aaa
                     l_amp = l3a_amps(idet)
                     ! x2a_vvv(aeb) <- A(f/ab) -l3a(abfmn)*t2a(efmn)
                     a = l3a_excits(idet,1); b = l3a_excits(idet,2); f = l3a_excits(idet,3);
                     m = l3a_excits(idet,4); n = l3a_excits(idet,5);
                     x2a_vvv(a,:,b) = x2a_vvv(a,:,b) - t2a(:,f,m,n)*l_amp ! (1)
                     x2a_vvv(f,:,b) = x2a_vvv(f,:,b) + t2a(:,a,m,n)*l_amp ! (af)
                     x2a_vvv(a,:,f) = x2a_vvv(a,:,f) + t2a(:,b,m,n)*l_amp ! (bf)
                  end do
                      
                  do idet = 1,n3aab
                     l_amp = l3b_amps(idet)
                     ! x2a_vvv(aeb) <- -l3b(abfmn)*t2b(efmn)
                     a = l3b_excits(idet,1); b = l3b_excits(idet,2); f = l3b_excits(idet,3);
                     m = l3b_excits(idet,4); n = l3b_excits(idet,5);
                     x2a_vvv(a,:,b) = x2a_vvv(a,:,b) - t2b(:,f,m,n)*l_amp ! (1)
                  end do

                  ! apply the common A(ab) antisymmetrizer
                  do a = 1,nua
                     do b = a+1,nua
                        do e = 1,nua
                           x2a_vvv(a,e,b) = x2a_vvv(a,e,b) - x2a_vvv(b,e,a)
                        end do
                     end do
                  end do
                  ! explicitly antisymmetrize
                  do a = 1,nua
                     do b = a+1,nua
                        x2a_vvv(b,:,a) = -x2a_vvv(a,:,b)
                     end do
                  end do

              end subroutine get_x2a_vvv
           
              subroutine get_x2b_ovo(x2b_ovo,&
                                     l3b_amps, l3b_excits,&
                                     l3c_amps, l3c_excits,&
                                     t2a, t2b,&
                                     n3aab, n3abb,&
                                     noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab, n3abb

                  integer, intent(in) :: l3b_excits(n3aab,5)
                  real(kind=8), intent(in) :: l3b_amps(n3aab)
                  integer, intent(in) :: l3c_excits(n3abb,5)
                  real(kind=8), intent(in) :: l3c_amps(n3abb)

                  real(kind=8), intent(in) :: t2a(nua,nua,noa,noa)
                  real(kind=8), intent(in) :: t2b(nua,nub,noa,nob)

                  real(kind=8), intent(out) :: x2b_ovo(noa,nub,nob)

                  real(kind=8) :: l_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet
                      
                  x2b_ovo = 0.0d0
                  do idet = 1,n3aab
                     l_amp = l3b_amps(idet)
                     ! x2b_ovo(ibj) <- l3b(efbnj)*t2a(efin)
                     e = l3b_excits(idet,1); f = l3b_excits(idet,2); b = l3b_excits(idet,3);
                     n = l3b_excits(idet,4); j = l3b_excits(idet,5);
                     x2b_ovo(:,b,j) = x2b_ovo(:,b,j) + t2a(e,f,:,n)*l_amp ! (1)
                  end do
                      
                  do idet = 1,n3abb
                     l_amp = l3c_amps(idet)
                     ! x2b_ovo(ibj) <- A(bf)A(jn) l3c(ebfjn)*t2b(efin)
                     e = l3c_excits(idet,1); b = l3c_excits(idet,2); f = l3c_excits(idet,3);
                     j = l3c_excits(idet,4); n = l3c_excits(idet,5);
                     x2b_ovo(:,b,j) = x2b_ovo(:,b,j) + t2b(e,f,:,n)*l_amp ! (1)
                     x2b_ovo(:,f,j) = x2b_ovo(:,f,j) - t2b(e,b,:,n)*l_amp ! (bf)
                     x2b_ovo(:,b,n) = x2b_ovo(:,b,n) - t2b(e,f,:,j)*l_amp ! (jn)
                     x2b_ovo(:,f,n) = x2b_ovo(:,f,n) + t2b(e,b,:,j)*l_amp ! (bf)(jn)
                  end do

              end subroutine get_x2b_ovo

              subroutine get_x2b_voo(x2b_voo,&
                                     l3b_amps, l3b_excits,&
                                     l3c_amps, l3c_excits,&
                                     t2b, t2c,&
                                     n3aab, n3abb,&
                                     noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab, n3abb

                  integer, intent(in) :: l3b_excits(n3aab,5)
                  real(kind=8), intent(in) :: l3b_amps(n3aab)
                  integer, intent(in) :: l3c_excits(n3abb,5)
                  real(kind=8), intent(in) :: l3c_amps(n3abb)

                  real(kind=8), intent(in) :: t2b(nua,nub,noa,nob)
                  real(kind=8), intent(in) :: t2c(nub,nub,nob,nob)

                  real(kind=8), intent(out) :: x2b_voo(nua,nob,nob)

                  real(kind=8) :: l_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet
                      
                  x2b_voo = 0.0d0
                  do idet = 1,n3aab
                     l_amp = l3b_amps(idet)
                     ! x2b_voo(ak~m~) <- A(af) l3b(afenk)*t2b(fenm) 
                     a = l3b_excits(idet,1); f = l3b_excits(idet,2); e = l3b_excits(idet,3);
                     n = l3b_excits(idet,4); k = l3b_excits(idet,5);
                     x2b_voo(a,k,:) = x2b_voo(a,k,:) + t2b(f,e,n,:)*l_amp ! (1)
                     x2b_voo(f,k,:) = x2b_voo(f,k,:) - t2b(a,e,n,:)*l_amp ! (af)
                  end do
                      
                  do idet = 1,n3abb
                     l_amp = l3c_amps(idet)
                     ! x2b_voo(ak~m~) <- A(kn) l3c(afenk)*t2c(fenm)
                     a = l3c_excits(idet,1); f = l3c_excits(idet,2); e = l3c_excits(idet,3);
                     n = l3c_excits(idet,4); k = l3c_excits(idet,5);
                     x2b_voo(a,k,:) = x2b_voo(a,k,:) + t2c(f,e,n,:)*l_amp ! (1)
                     x2b_voo(a,n,:) = x2b_voo(a,n,:) - t2c(f,e,k,:)*l_amp ! (kn)
                  end do

              end subroutine get_x2b_voo

              subroutine get_x2b_vvv(x2b_vvv,&
                                     l3b_amps, l3b_excits,&
                                     l3c_amps, l3c_excits,&
                                     t2b, t2c,&
                                     n3aab, n3abb,&
                                     noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab, n3abb

                  integer, intent(in) :: l3b_excits(n3aab,5)
                  real(kind=8), intent(in) :: l3b_amps(n3aab)
                  integer, intent(in) :: l3c_excits(n3abb,5)
                  real(kind=8), intent(in) :: l3c_amps(n3abb)

                  real(kind=8), intent(in) :: t2b(nua,nub,noa,nob)
                  real(kind=8), intent(in) :: t2c(nub,nub,nob,nob)

                  real(kind=8), intent(out) :: x2b_vvv(nua,nub,nub)

                  real(kind=8) :: l_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet
                  
                  x2b_vvv = 0.0d0    
                  do idet = 1,n3aab
                     l_amp = l3b_amps(idet)
                     ! x2b_vvv(ae~b~) <- A(af) -l3b(afbnm)*t2b(fenm)
                     a = l3b_excits(idet,1); f = l3b_excits(idet,2); b = l3b_excits(idet,3);
                     n = l3b_excits(idet,4); m = l3b_excits(idet,5);
                     x2b_vvv(a,:,b) = x2b_vvv(a,:,b) - t2b(f,:,n,m)*l_amp ! (1)
                     x2b_vvv(f,:,b) = x2b_vvv(f,:,b) + t2b(a,:,n,m)*l_amp ! (af)
                  end do
                      
                  do idet = 1,n3abb
                     l_amp = l3c_amps(idet)
                     ! x2b_vvv(ae~b~) <- A(bf) -l3c(afbnm)*t2c(efmn)
                     a = l3c_excits(idet,1); f = l3c_excits(idet,2); b = l3c_excits(idet,3);
                     n = l3c_excits(idet,4); m = l3c_excits(idet,5);
                     x2b_vvv(a,:,b) = x2b_vvv(a,:,b) - t2c(:,f,m,n)*l_amp ! (1)
                     x2b_vvv(a,:,f) = x2b_vvv(a,:,f) + t2c(:,b,m,n)*l_amp ! (bf)
                  end do

              end subroutine get_x2b_vvv

end module lefteaeom3_p_intermediates
