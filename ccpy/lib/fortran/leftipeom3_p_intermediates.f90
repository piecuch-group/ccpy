module leftipeom3_p_intermediates

        implicit none
        
        contains

              subroutine get_x2a_vvo(x2a_vvo,&
                                     l3a_amps, l3a_excits,&
                                     l3b_amps, l3b_excits,&
                                     t2a, t2b,&
                                     do_aaa, do_aab,&
                                     n3aaa, n3aab,&
                                     noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa, n3aab
                  logical, intent(in) :: do_aaa, do_aab

                  integer, intent(in) :: l3a_excits(n3aaa,5)
                  real(kind=8), intent(in) :: l3a_amps(n3aaa)
                  integer, intent(in) :: l3b_excits(n3aab,5)
                  real(kind=8), intent(in) :: l3b_amps(n3aab)

                  real(kind=8), intent(in) :: t2a(nua,nua,noa,noa)
                  real(kind=8), intent(in) :: t2b(nua,nub,noa,nob)

                  real(kind=8), intent(out) :: x2a_vvo(nua,nua,noa)

                  real(kind=8) :: l_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet
                       
                  x2a_vvo = 0.0d0
                  if (do_aaa) then
                  do idet = 1,n3aaa
                     l_amp = l3a_amps(idet)
                     ! x2a(abj) <- A(bf)A(j/mn) -l3a(bfmjn)*t2a(afmn)
                     b = l3a_excits(idet,1); f = l3a_excits(idet,2);
                     m = l3a_excits(idet,3); j = l3a_excits(idet,4); n = l3a_excits(idet,5);
                     x2a_vvo(:,b,j) = x2a_vvo(:,b,j) - t2a(:,f,m,n)*l_amp ! (1)
                     x2a_vvo(:,b,m) = x2a_vvo(:,b,m) + t2a(:,f,j,n)*l_amp ! (jm)
                     x2a_vvo(:,b,n) = x2a_vvo(:,b,n) + t2a(:,f,m,j)*l_amp ! (jn)
                     x2a_vvo(:,f,j) = x2a_vvo(:,f,j) + t2a(:,b,m,n)*l_amp ! (bf)
                     x2a_vvo(:,f,m) = x2a_vvo(:,f,m) - t2a(:,b,j,n)*l_amp ! (jm)(bf)
                     x2a_vvo(:,f,n) = x2a_vvo(:,f,n) - t2a(:,b,m,j)*l_amp ! (jn)(bf)
                  end do
                  end if
                  
                  if (do_aab) then
                  do idet = 1,n3aab
                     l_amp = l3b_amps(idet)
                     ! x2a(abj) <- A(jm) -l3b(bfmjn)*t2b(afmn)
                     b = l3b_excits(idet,1); f = l3b_excits(idet,2);
                     m = l3b_excits(idet,3); j = l3b_excits(idet,4); n = l3b_excits(idet,5);
                     x2a_vvo(:,b,j) = x2a_vvo(:,b,j) - t2b(:,f,m,n)*l_amp ! (1) 
                     x2a_vvo(:,b,m) = x2a_vvo(:,b,m) + t2b(:,f,j,n)*l_amp ! (1) 
                  end do
                  end if

              end subroutine get_x2a_vvo

              subroutine get_x2a_ooo(x2a_ooo,&
                                     l3a_amps, l3a_excits,&
                                     l3b_amps, l3b_excits,&
                                     t2a, t2b,&
                                     do_aaa, do_aab,&
                                     n3aaa, n3aab,&
                                     noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa, n3aab
                  logical, intent(in) :: do_aaa, do_aab

                  integer, intent(in) :: l3a_excits(n3aaa,5)
                  real(kind=8), intent(in) :: l3a_amps(n3aaa)
                  integer, intent(in) :: l3b_excits(n3aab,5)
                  real(kind=8), intent(in) :: l3b_amps(n3aab)

                  real(kind=8), intent(in) :: t2a(nua,nua,noa,noa)
                  real(kind=8), intent(in) :: t2b(nua,nub,noa,nob)

                  real(kind=8), intent(out) :: x2a_ooo(noa,noa,noa)

                  real(kind=8) :: l_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  x2a_ooo = 0.0d0
                  if (do_aaa) then
                  do idet = 1,n3aaa
                     l_amp = l3a_amps(idet)
                     ! x2a_ooo(ijk) <- A(n/ij) l3a(efijn)*t2a(efkn)
                     e = l3a_excits(idet,1); f = l3a_excits(idet,2);
                     i = l3a_excits(idet,3); j = l3a_excits(idet,4); n = l3a_excits(idet,5);
                     x2a_ooo(i,j,:) = x2a_ooo(i,j,:) + t2a(e,f,:,n)*l_amp ! (1)
                     x2a_ooo(j,n,:) = x2a_ooo(j,n,:) + t2a(e,f,:,i)*l_amp ! (in)
                     x2a_ooo(i,n,:) = x2a_ooo(i,n,:) - t2a(e,f,:,j)*l_amp ! (jn)
                  end do
                  end if
                  
                  if (do_aab) then
                  do idet = 1,n3aab
                     l_amp = l3b_amps(idet)
                     ! x2a_ooo(ijk) <- l3b(efijn)*t2b(efkn)
                     e = l3b_excits(idet,1); f = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); j = l3b_excits(idet,4); n = l3b_excits(idet,5);
                     x2a_ooo(i,j,:) = x2a_ooo(i,j,:) + t2b(e,f,:,n)*l_amp ! (1)
                  end do
                  end if

                  ! apply the common A(ij) antisymmetrizer
                  do i = 1,noa
                     do j = i+1,noa
                        do k = 1,noa
                           x2a_ooo(i,j,k) = x2a_ooo(i,j,k) - x2a_ooo(j,i,k)
                        end do
                     end do
                  end do
                  ! explicitly antisymmetrize
                  do i = 1,noa
                     do j = i+1,noa
                        x2a_ooo(j,i,:) = -x2a_ooo(i,j,:)
                     end do
                  end do

              end subroutine get_x2a_ooo
           
              subroutine get_x2b_vvo(x2b_vvo,&
                                     l3b_amps, l3b_excits,&
                                     l3c_amps, l3c_excits,&
                                     t2a, t2b,&
                                     do_aab, do_abb,&
                                     n3aab, n3abb,&
                                     noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab, n3abb
                  logical, intent(in) :: do_aab, do_abb

                  integer, intent(in) :: l3b_excits(n3aab,5)
                  real(kind=8), intent(in) :: l3b_amps(n3aab)
                  integer, intent(in) :: l3c_excits(n3abb,5)
                  real(kind=8), intent(in) :: l3c_amps(n3abb)

                  real(kind=8), intent(in) :: t2a(nua,nua,noa,noa)
                  real(kind=8), intent(in) :: t2b(nua,nub,noa,nob)

                  real(kind=8), intent(out) :: x2b_vvo(nua,nub,nob)

                  real(kind=8) :: l_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet
                      
                  x2b_vvo = 0.0d0
                  if (do_aab) then
                  do idet = 1,n3aab
                     l_amp = l3b_amps(idet)
                     ! x2b_vvo(abj) <- -l3b(fbmnj)*t2a(afmn)
                     f = l3b_excits(idet,1); b = l3b_excits(idet,2);
                     m = l3b_excits(idet,3); n = l3b_excits(idet,4); j = l3b_excits(idet,5);
                     x2b_vvo(:,b,j) = x2b_vvo(:,b,j) - t2a(:,f,m,n)*l_amp ! (1)
                  end do
                  end if
                  
                  if (do_abb) then
                  do idet = 1,n3abb
                     l_amp = l3c_amps(idet)
                     ! x2b_vvo(abj) <- A(bf)A(jn) -l3c(fbmnj)*t2b(afmn)
                     f = l3c_excits(idet,1); b = l3c_excits(idet,2);
                     m = l3c_excits(idet,3); n = l3c_excits(idet,4); j = l3c_excits(idet,5);
                     x2b_vvo(:,b,j) = x2b_vvo(:,b,j) - t2b(:,f,m,n)*l_amp ! (1)
                     x2b_vvo(:,f,j) = x2b_vvo(:,f,j) + t2b(:,b,m,n)*l_amp ! (bf)
                     x2b_vvo(:,b,n) = x2b_vvo(:,b,n) + t2b(:,f,m,j)*l_amp ! (jn)
                     x2b_vvo(:,f,n) = x2b_vvo(:,f,n) - t2b(:,b,m,j)*l_amp ! (bf)(jn)
                  end do
                  end if

              end subroutine get_x2b_vvo

              subroutine get_x2b_ovv(x2b_ovv,&
                                     l3b_amps, l3b_excits,&
                                     l3c_amps, l3c_excits,&
                                     t2b, t2c,&
                                     do_aab, do_abb,&
                                     n3aab, n3abb,&
                                     noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab, n3abb
                  logical, intent(in) :: do_aab, do_abb

                  integer, intent(in) :: l3b_excits(n3aab,5)
                  real(kind=8), intent(in) :: l3b_amps(n3aab)
                  integer, intent(in) :: l3c_excits(n3abb,5)
                  real(kind=8), intent(in) :: l3c_amps(n3abb)

                  real(kind=8), intent(in) :: t2b(nua,nub,noa,nob)
                  real(kind=8), intent(in) :: t2c(nub,nub,nob,nob)

                  real(kind=8), intent(out) :: x2b_ovv(noa,nub,nub)

                  real(kind=8) :: l_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet
                      
                  x2b_ovv = 0.0d0
                  if (do_aab) then
                  do idet = 1,n3aab
                     l_amp = l3b_amps(idet)
                     ! x2b_ovv(icb) <- A(in) -l3b(fbinm)*t2b(fcnm)
                     f = l3b_excits(idet,1); b = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); n = l3b_excits(idet,4); m = l3b_excits(idet,5);
                     x2b_ovv(i,:,b) = x2b_ovv(i,:,b) - t2b(f,:,n,m)*l_amp ! (1)
                     x2b_ovv(n,:,b) = x2b_ovv(n,:,b) + t2b(f,:,i,m)*l_amp ! (in)
                  end do
                  end if
                  
                  if (do_abb) then
                  do idet = 1,n3abb
                     l_amp = l3c_amps(idet)
                     ! x2b_ovv(icb) <- A(bf) -l3c(fbinm)*t2c(fcnm)
                     f = l3c_excits(idet,1); b = l3c_excits(idet,2);
                     i = l3c_excits(idet,3); n = l3c_excits(idet,4); m = l3c_excits(idet,5);
                     x2b_ovv(i,:,b) = x2b_ovv(i,:,b) - t2c(f,:,n,m)*l_amp ! (1)
                     x2b_ovv(i,:,f) = x2b_ovv(i,:,f) + t2c(b,:,n,m)*l_amp ! (bf)
                  end do
                  end if

              end subroutine get_x2b_ovv

              subroutine get_x2b_ooo(x2b_ooo,&
                                     l3b_amps, l3b_excits,&
                                     l3c_amps, l3c_excits,&
                                     t2b, t2c,&
                                     do_aab, do_abb,&
                                     n3aab, n3abb,&
                                     noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab, n3abb
                  logical, intent(in) :: do_aab, do_abb

                  integer, intent(in) :: l3b_excits(n3aab,5)
                  real(kind=8), intent(in) :: l3b_amps(n3aab)
                  integer, intent(in) :: l3c_excits(n3abb,5)
                  real(kind=8), intent(in) :: l3c_amps(n3abb)

                  real(kind=8), intent(in) :: t2b(nua,nub,noa,nob)
                  real(kind=8), intent(in) :: t2c(nub,nub,nob,nob)

                  real(kind=8), intent(out) :: x2b_ooo(noa,nob,nob)

                  real(kind=8) :: l_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet
                  
                  x2b_ooo = 0.0d0
                  if (do_aab) then
                  do idet = 1,n3aab
                     l_amp = l3b_amps(idet)
                     ! x2b_ooo(ijk) <- A(in) l3b(feinj)*t2b(fenk)
                     f = l3b_excits(idet,1); e = l3b_excits(idet,2);
                     i = l3b_excits(idet,3); n = l3b_excits(idet,4); j = l3b_excits(idet,5);
                     x2b_ooo(i,j,:) = x2b_ooo(i,j,:) + t2b(f,e,n,:)*l_amp ! (1)
                     x2b_ooo(n,j,:) = x2b_ooo(n,j,:) - t2b(f,e,i,:)*l_amp ! (in)
                  end do
                  end if
                  
                  if (do_abb) then
                  do idet = 1,n3abb
                     l_amp = l3c_amps(idet)
                     ! x2b_ooo(ijk) <- A(jn) l3c(efijn)*t2c(efkn)
                     e = l3c_excits(idet,1); f = l3c_excits(idet,2);
                     i = l3c_excits(idet,3); j = l3c_excits(idet,4); n = l3c_excits(idet,5);
                     x2b_ooo(i,j,:) = x2b_ooo(i,j,:) + t2c(e,f,:,n)*l_amp ! (1)
                     x2b_ooo(i,n,:) = x2b_ooo(i,n,:) - t2c(e,f,:,j)*l_amp ! (jn)
                  end do
                  end if

              end subroutine get_x2b_ooo

end module leftipeom3_p_intermediates
