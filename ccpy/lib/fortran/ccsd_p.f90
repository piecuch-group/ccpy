MODULE ccsd_p
        IMPLICIT NONE

        CONTAINS

              subroutine ccsd_intermediates(fock, g,&
                                            t1, t2_amps, t2_excits,&
                                            n2, no, nu, norb)

                  integer, intent(in) :: no, nu, norb, n2
                  real(kind=8), intent(in) :: t1(nu,no)
                  integer, intent(in) :: t2_excits(n2,4)
                  real(kind=8), intent(in) :: t2_amps(n2)

                  real(kind=8), intent(inout) :: fock(norb,norb)
                  !f2py intent(in,out) :: fock(0:norb-1,0:norb-1)
                  real(kind=8), intent(inout) :: g(norb,norb,norb,norb)
                  !f2py intent(in,out) :: g(0:norb-1,0:norb-1,0:norb-1,0:norb-1)

                  real(kind=8) :: t_amp
                  integer :: a, b, i, j, idet
                  integer :: e, f, m, n

                  ! h1(mi) <- h1(mi) + A(in) g(mnef) * t2(efin)
                  do idet = 1,n2
                     t_amp = t2_amps(idet)
                     e = t2_excits(idet,1); f = t2_excits(idet,2);
                     i = t2_excits(idet,3); n = t2_excits(idet,4);
                     do m = 1,no
                        fock(m,i) = fock(m,i) + g(m,n,e,f) * t_amp ! (1)
                        fock(m,n) = fock(m,n) - g(m,i,e,f) * t_amp ! (in)
                     end do
                  end do

                  ! h1(ae) <- h1(ae) - A(af) g(mnef) * t2(afmn)
                  do idet = 1,n2
                     t_amp = t2_amps(idet)
                     a = t2_excits(idet,1); f = t2_excits(idet,2);
                     m = t2_excits(idet,3); n = t2_excits(idet,4);
                     do e = 1,nu
                        fock(a,e+no) = fock(a,e+no) - g(m,n,e+no,f) * t_amp ! (1)
                        fock(f,e+no) = fock(f,e+no) + g(m,n,e+no,a) * t_amp ! (af)
                     end do
                  end do

                  ! h2(mnij) <- h2(mnij) + g(mnef) * t2(efij)
                  do idet = 1,n2
                     t_amp = t2_amps(idet)
                     e = t2_excits(idet,1); f = t2_excits(idet,2);
                     i = t2_excits(idet,3); j = t2_excits(idet,4);
                     do m = 1,no
                        do n = m+1,no
                           g(m,n,i,j) = g(m,n,i,j) + g(m,n,e,f) * t_amp
                        end do
                     end do   
                  end do
                  do m = 1,no
                     do n = m+1,no
                        do i = 1,no
                           do j = i+1,no
                              g(n,m,i,j) = -g(m,n,i,j)
                              g(m,n,j,i) = -g(m,n,i,j)
                              g(n,m,j,i) = g(m,n,i,j)
                           end do
                        end do
                     end do
                  end do

                  ! h2(amie) <- h2(amie) + A(af)A(in) 1/2 g(mnef) * t2(afin)
                  ! Note: here, the factor of 1/2 is not a diagrammatic weight, rather
                  !       it cancels the action of an antisymmetrizer acting on this diagram
                  !       in later contractions. It must be included even in the devectorized
                  !       scheme.
                  do idet = 1,n2
                     t_amp = 0.5d0 * t2_amps(idet)
                     a = t2_excits(idet,1); f = t2_excits(idet,2);
                     i = t2_excits(idet,3); n = t2_excits(idet,4);
                     do e = 1,nu
                        do m = 1,no
                           g(a,m,i,e+no) = g(a,m,i,e+no) + g(m,n,e+no,f) * t_amp ! (1)
                           g(f,m,i,e+no) = g(f,m,i,e+no) - g(m,n,e+no,a) * t_amp ! (af)
                           g(a,m,n,e+no) = g(a,m,n,e+no) - g(m,i,e+no,f) * t_amp ! (in)
                           g(f,m,n,e+no) = g(f,m,n,e+no) + g(m,i,e+no,a) * t_amp ! (af)(in)
                        end do
                     end do
                  end do

                  ! h2(amij) <- h2(amij) + g(amef) * t2(efij)
                  do idet = 1,n2
                     t_amp = t2_amps(idet)
                     e = t2_excits(idet,1); f = t2_excits(idet,2);
                     i = t2_excits(idet,3); j = t2_excits(idet,4);
                     do a = 1,nu
                        do m = 1,no
                           g(a+no,m,i,j) = g(a+no,m,i,j) + g(a+no,m,e,f) * t_amp ! (1)
                        end do
                     end do
                  end do      
                  do a = 1,nu
                     do m = 1,no
                        do i = 1,no
                           do j = i+1,no
                              g(a+no,m,j,i) = -g(a+no,m,i,j)
                           end do
                        end do
                     end do
                  end do

              end subroutine ccsd_intermediates


              subroutine doubles_residual(resid, t2_amps, t2_excits,&
                                     t1,&
                                     fock, g,&
                                     n2, no, nu, norb)

                  integer, intent(in) :: no, nu, norb, n2
                  real(kind=8), intent(in) :: t1(nu,no)
                  real(kind=8), intent(in) :: fock(norb,norb), g(norb,norb,norb,norb)

                  integer, intent(inout) :: t2_excits(n2,4)
                  !f2py intent(in,out) :: t2_excits(0:n2-1,0:3)
                  real(kind=8), intent(inout) :: t2_amps(n2)
                  !f2py intent(in,out) :: t2_amps(0:n2-1)

                  real(kind=8), intent(out) :: resid(n2)

                  integer, allocatable :: idx_table3(:,:,:), idx_table2(:,:)
                  integer, allocatable :: loc_arr(:,:)

                  real(kind=8), allocatable :: amps_buff(:)
                  integer, allocatable :: excits_buff(:,:)

                  real(kind=8) :: t_amp, hmatel
                  integer :: a, b, c, d, i, j, k, l, idet, jdet
                  integer :: e, f, m, n
                  integer :: idx, nloc

                  ! Zero the residual container
                  resid = 0.0d0

                  !!!! diagram 0: h2(abij)
                  do idet = 1,n2
                     a = t2_excits(idet,1); b = t2_excits(idet,2);
                     i = t2_excits(idet,3); j = t2_excits(idet,4);
                     hmatel = g(a,b,i,j)
                     resid(idet) = resid(idet) + hmatel
                  end do

                  !!!! diagram 1: A(ij) h2(abie) * t1(ej)
                  do idet = 1,n2
                     a = t2_excits(idet,1); b = t2_excits(idet,2);
                     i = t2_excits(idet,3); j = t2_excits(idet,4);
                     do e = 1,nu
                        hmatel = g(a,b,i,e+no)
                        resid(idet) = resid(idet) + hmatel * t1(e,j)
                        hmatel = g(a,b,j,e+no)
                        resid(idet) = resid(idet) - hmatel * t1(e,i) 
                     end do
                  end do
                     
                  !!!! diagram 2: -A(ab) h2(amij) * t1(bm)
                  do idet = 1,n2
                     a = t2_excits(idet,1); b = t2_excits(idet,2);
                     i = t2_excits(idet,3); j = t2_excits(idet,4);
                     do m = 1,no
                        hmatel = g(a,m,i,j)
                        resid(idet) = resid(idet) - hmatel * t1(b-no,m)
                        hmatel = g(b,m,i,j)
                        resid(idet) = resid(idet) + hmatel * t1(a-no,m) 
                     end do
                  end do

                  !!!! diagram 3: -A(ij) h1(mi) * t2(abmj)
                  ! allocate new sorting arrays
                  nloc = nu*(nu-1)/2 * no
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nu,nu,no))
                  !!! SB: (1,2,3) !!!
                  call get_index_table3(idx_table3, (/1,nu-1/), (/-1,nu/), (/1,no-1/), nu, nu, no)
                  call sort3(t2_excits, t2_amps, loc_arr, idx_table3, (/1,2,3/), nu, nu, no, nloc, n2, (/no,no,0/), resid)
                  do idet = 1,n2
                     a = t2_excits(idet,1); b = t2_excits(idet,2);
                     i = t2_excits(idet,3); j = t2_excits(idet,4);
                     ! (1)
                     idx = idx_table3(a-no,b-no,i)
                     if( idx/=0 ) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           l = t2_excits(jdet,4)
                           ! <abij|H|abil> = -h_oo(l,j)
                           hmatel = -fock(l,j)
                           resid(idet) = resid(idet) + hmatel * t2_amps(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table3(a-no,b-no,j)
                     if( idx/=0 ) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           l = t2_excits(jdet,4)
                           ! <abij|H|abjl> = h_oo(l,i)
                           hmatel = fock(l,i)
                           resid(idet) = resid(idet) + hmatel * t2_amps(jdet)
                        end do
                     end if
                  end do
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nu-1/), (/-1,nu/), (/2,no/), nu, nu, no)
                  call sort3(t2_excits, t2_amps, loc_arr, idx_table3, (/1,2,4/), nu, nu, no, nloc, n2, (/no,no,0/), resid)
                  do idet = 1,n2
                     a = t2_excits(idet,1); b = t2_excits(idet,2);
                     i = t2_excits(idet,3); j = t2_excits(idet,4);
                     ! (1)
                     idx = idx_table3(a-no,b-no,i)
                     if( idx/=0 ) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           k = t2_excits(jdet,3)
                           ! <abij|H|abki> = h_oo(k,j)
                           hmatel = fock(k,j)
                           resid(idet) = resid(idet) + hmatel * t2_amps(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table3(a-no,b-no,j)
                     if( idx/=0 ) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           k = t2_excits(jdet,3)
                           ! <abij|H|abkj> = -h_oo(k,i)
                           hmatel = -fock(k,i)
                           resid(idet) = resid(idet) + hmatel * t2_amps(jdet)
                        end do
                     end if
                  end do
                  deallocate(idx_table3,loc_arr)

                  !!!! diagram 4: A(ab) h1(ae) * t2(ebmj)
                  ! allocate new sorting arrays
                  nloc = no*(no-1)/2 * nu
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(no,no,nu))
                  !!! SB: (3,4,1) !!!
                  call get_index_table3(idx_table3, (/1,no-1/), (/-1,no/), (/1,nu-1/), no, no, nu)
                  call sort3(t2_excits, t2_amps, loc_arr, idx_table3, (/3,4,1/), no, no, nu, nloc, n2, (/0,0,no/), resid)
                  do idet = 1,n2
                     a = t2_excits(idet,1); b = t2_excits(idet,2);
                     i = t2_excits(idet,3); j = t2_excits(idet,4);
                     ! (1)
                     idx = idx_table3(i,j,a-no)
                     if( idx/=0 ) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = t2_excits(jdet,2)
                           ! <abij|H|adij> = h_vv(b,d)
                           hmatel = fock(b,d)
                           resid(idet) = resid(idet) + hmatel * t2_amps(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table3(i,j,b-no)
                     if( idx/=0 ) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = t2_excits(jdet,2)
                           ! <abij|H|bdij> = -h_vv(a,d)
                           hmatel = -fock(a,d)
                           resid(idet) = resid(idet) + hmatel * t2_amps(jdet)
                        end do
                     end if
                  end do
                  !!! SB: (3,4,2) !!!
                  call get_index_table3(idx_table3, (/1,no-1/), (/-1,no/), (/2,nu/), no, no, nu)
                  call sort3(t2_excits, t2_amps, loc_arr, idx_table3, (/3,4,2/), no, no, nu, nloc, n2, (/0,0,no/), resid)
                  do idet = 1,n2
                     a = t2_excits(idet,1); b = t2_excits(idet,2);
                     i = t2_excits(idet,3); j = t2_excits(idet,4);
                     ! (1)
                     idx = idx_table3(i,j,a-no)
                     if( idx/=0 ) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           c = t2_excits(jdet,1)
                           ! <abij|H|caij> = -h_vv(b,c)
                           hmatel = -fock(b,c)
                           resid(idet) = resid(idet) + hmatel * t2_amps(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table3(i,j,b-no)
                     if( idx/=0 ) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           c = t2_excits(jdet,1)
                           ! <abij|H|cbij> = h_vv(a,c)
                           hmatel = fock(a,c)
                           resid(idet) = resid(idet) + hmatel * t2_amps(jdet)
                        end do
                     end if
                  end do
                  deallocate(idx_table3,loc_arr)

                  !!!! diagram 5: 1/2 h2(mnij) * t2(abmn)
                  ! allocate new sorting arrays
                  nloc = nu*(nu-1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table2(nu,nu))
                  !!! SB: (1,2) !!!
                  call get_index_table2(idx_table2, (/1,nu-1/), (/-1,nu/), nu, nu)
                  call sort2(t2_excits, t2_amps, loc_arr, idx_table2, (/1,2/), nu, nu, nloc, n2, (/no,no/), resid)
                  do idet = 1,n2
                     a = t2_excits(idet,1); b = t2_excits(idet,2);
                     i = t2_excits(idet,3); j = t2_excits(idet,4);
                     ! (1)
                     idx = idx_table2(a-no,b-no)
                     if( idx/=0 ) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           k = t2_excits(jdet,3); l = t2_excits(jdet,4);
                           ! <abij|H|abkl> = h_oooo(k,l,i,j)
                           hmatel = g(k,l,i,j)
                           resid(idet) = resid(idet) + hmatel * t2_amps(jdet)
                        end do
                     end if
                  end do
                  deallocate(idx_table2,loc_arr)

                  !!!! diagram 6: 1/2 h2(abef) * tau(efij)
                  ! allocate new sorting arrays
                  nloc = no*(no-1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table2(no,no))
                  !!! SB: (3,4) !!!
                  call get_index_table2(idx_table2, (/1,no-1/), (/-1,no/), no, no)
                  call sort2(t2_excits, t2_amps, loc_arr, idx_table2, (/3,4/), no, no, nloc, n2, (/0,0/), resid)
                  do idet = 1,n2
                     a = t2_excits(idet,1); b = t2_excits(idet,2);
                     i = t2_excits(idet,3); j = t2_excits(idet,4);
                     ! (1)
                     idx = idx_table2(i,j)
                     if( idx/=0 ) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           c = t2_excits(jdet,1); d = t2_excits(jdet,2);
                           ! <abij|H|cdij> = h_vvvv(a,b,c,d)
                           hmatel = g(a,b,c,d)
                           !t_amp = t2_amps(jdet) + t1(c-no,i)*t1(d-no,j) - t1(c-no,j)*t1(d-no,i)
                           resid(idet) = resid(idet) + hmatel * t2_amps(jdet)
                        end do
                     end if
                  end do
                  deallocate(idx_table2,loc_arr)
                  do idet = 1,n2
                     a = t2_excits(idet,1); b = t2_excits(idet,2);
                     i = t2_excits(idet,3); j = t2_excits(idet,4);
                     do e = 1,nu
                        do f = e+1,nu
                           t_amp = t1(e,i)*t1(f,j) - t1(e,j)*t1(f,i)
                           hmatel = g(a,b,e+no,f+no)
                           resid(idet) = resid(idet) + hmatel * t_amp 
                        end do
                     end do
                  end do

                  !!!! diagram 7: A(ij)A(ab) h2(amie) * t2(ebmj)
                  ! allocate new sorting arrays
                  nloc = no*nu
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table2(nu,no))
                  !!! SB: (1,3) !!!
                  call get_index_table2(idx_table2, (/1,nu-1/), (/1,no-1/), nu, no)
                  call sort2(t2_excits, t2_amps, loc_arr, idx_table2, (/1,3/), nu, no, nloc, n2, (/no,0/), resid)
                  do idet = 1,n2
                     a = t2_excits(idet,1); b = t2_excits(idet,2);
                     i = t2_excits(idet,3); j = t2_excits(idet,4);
                     ! (1)
                     idx = idx_table2(a-no,i)
                     if( idx/=0 ) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = t2_excits(jdet,2); l = t2_excits(jdet,4);
                           ! <abij|H|adil> = h_voov(b,l,j,d)
                           hmatel = g(b,l,j,d)
                           resid(idet) = resid(idet) + hmatel * t2_amps(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table2(b-no,i)
                     if( idx/=0 ) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = t2_excits(jdet,2); l = t2_excits(jdet,4);
                           ! <abij|H|bdil> = -h_voov(a,l,j,d)
                           hmatel = -g(a,l,j,d)
                           resid(idet) = resid(idet) + hmatel * t2_amps(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table2(a-no,j)
                     if( idx/=0 ) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = t2_excits(jdet,2); l = t2_excits(jdet,4);
                           ! <abij|H|adjl> = -h_voov(b,l,i,d)
                           hmatel = -g(b,l,i,d)
                           resid(idet) = resid(idet) + hmatel * t2_amps(jdet)
                        end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table2(b-no,j)
                     if( idx/=0 ) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = t2_excits(jdet,2); l = t2_excits(jdet,4);
                           ! <abij|H|bdjl> = h_voov(a,l,i,d)
                           hmatel = g(a,l,i,d)
                           resid(idet) = resid(idet) + hmatel * t2_amps(jdet)
                        end do
                     end if
                  end do
                  !!! SB: (1,4) !!!
                  call get_index_table2(idx_table2, (/1,nu-1/), (/2,no/), nu, no)
                  call sort2(t2_excits, t2_amps, loc_arr, idx_table2, (/1,4/), nu, no, nloc, n2, (/no,0/), resid)
                  do idet = 1,n2
                     a = t2_excits(idet,1); b = t2_excits(idet,2);
                     i = t2_excits(idet,3); j = t2_excits(idet,4);
                     ! (1)
                     idx = idx_table2(a-no,i)
                     if( idx/=0 ) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = t2_excits(jdet,2); k = t2_excits(jdet,3);
                           ! <abij|H|adki> = -h_voov(b,k,j,d)
                           hmatel = -g(b,k,j,d)
                           resid(idet) = resid(idet) + hmatel * t2_amps(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table2(b-no,i)
                     if( idx/=0 ) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = t2_excits(jdet,2); k = t2_excits(jdet,3);
                           ! <abij|H|bdki> = h_voov(a,k,j,d)
                           hmatel = g(a,k,j,d)
                           resid(idet) = resid(idet) + hmatel * t2_amps(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table2(a-no,j)
                     if( idx/=0 ) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = t2_excits(jdet,2); k = t2_excits(jdet,3);
                           ! <abij|H|adkj> = h_voov(b,k,i,d)
                           hmatel = g(b,k,i,d)
                           resid(idet) = resid(idet) + hmatel * t2_amps(jdet)
                        end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table2(b-no,j)
                     if( idx/=0 ) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = t2_excits(jdet,2); k = t2_excits(jdet,3);
                           ! <abij|H|bdkj> = -h_voov(a,k,i,d)
                           hmatel = -g(a,k,i,d)
                           resid(idet) = resid(idet) + hmatel * t2_amps(jdet)
                        end do
                     end if
                  end do
                  !!! SB: (2,3) !!!
                  call get_index_table2(idx_table2, (/2,nu/), (/1,no-1/), nu, no)
                  call sort2(t2_excits, t2_amps, loc_arr, idx_table2, (/2,3/), nu, no, nloc, n2, (/no,0/), resid)
                  do idet = 1,n2
                     a = t2_excits(idet,1); b = t2_excits(idet,2);
                     i = t2_excits(idet,3); j = t2_excits(idet,4);
                     ! (1)
                     idx = idx_table2(a-no,i)
                     if( idx/=0 ) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           c = t2_excits(jdet,1); l = t2_excits(jdet,4);
                           ! <abij|H|cail> = -h_voov(b,l,j,c)
                           hmatel = -g(b,l,j,c)
                           resid(idet) = resid(idet) + hmatel * t2_amps(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table2(b-no,i)
                     if( idx/=0 ) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           c = t2_excits(jdet,1); l = t2_excits(jdet,4);
                           ! <abij|H|cbil> = h_voov(a,l,j,c)
                           hmatel = g(a,l,j,c)
                           resid(idet) = resid(idet) + hmatel * t2_amps(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table2(a-no,j)
                     if( idx/=0 ) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           c = t2_excits(jdet,1); l = t2_excits(jdet,4);
                           ! <abij|H|cajl> = h_voov(b,l,i,c)
                           hmatel = g(b,l,i,c)
                           resid(idet) = resid(idet) + hmatel * t2_amps(jdet)
                        end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table2(b-no,j)
                     if( idx/=0 ) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           c = t2_excits(jdet,1); l = t2_excits(jdet,4);
                           ! <abij|H|cbjl> = -h_voov(a,l,i,c)
                           hmatel = -g(a,l,i,c)
                           resid(idet) = resid(idet) + hmatel * t2_amps(jdet)
                        end do
                     end if
                  end do
                  !!! SB: (2,4) !!!
                  call get_index_table2(idx_table2, (/2,nu/), (/2,no/), nu, no)
                  call sort2(t2_excits, t2_amps, loc_arr, idx_table2, (/2,4/), nu, no, nloc, n2, (/no,0/), resid)
                  do idet = 1,n2
                     a = t2_excits(idet,1); b = t2_excits(idet,2);
                     i = t2_excits(idet,3); j = t2_excits(idet,4);
                     ! (1)
                     idx = idx_table2(a-no,i)
                     if( idx/=0 ) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           c = t2_excits(jdet,1); k = t2_excits(jdet,3);
                           ! <abij|H|caki> = h_voov(b,k,j,c)
                           hmatel = g(b,k,j,c)
                           resid(idet) = resid(idet) + hmatel * t2_amps(jdet)
                        end do
                     end if
                     ! (ab)
                     idx = idx_table2(b-no,i)
                     if( idx/=0 ) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           c = t2_excits(jdet,1); k = t2_excits(jdet,3);
                           ! <abij|H|cbki> = -h_voov(a,k,j,c)
                           hmatel = -g(a,k,j,c)
                           resid(idet) = resid(idet) + hmatel * t2_amps(jdet)
                        end do
                     end if
                     ! (ij)
                     idx = idx_table2(a-no,j)
                     if( idx/=0 ) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           c = t2_excits(jdet,1); k = t2_excits(jdet,3);
                           ! <abij|H|cakj> = -h_voov(b,k,i,c)
                           hmatel = -g(b,k,i,c)
                           resid(idet) = resid(idet) + hmatel * t2_amps(jdet)
                        end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table2(b-no,j)
                     if( idx/=0 ) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           c = t2_excits(jdet,1); k = t2_excits(jdet,3);
                           ! <abij|H|cbkj> = h_voov(a,k,i,c)
                           hmatel = g(a,k,i,c)
                           resid(idet) = resid(idet) + hmatel * t2_amps(jdet)
                        end do
                     end if
                  end do
                  deallocate(idx_table2,loc_arr)

          end subroutine doubles_residual

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!! SORTING FUNCTIONS !!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine get_index_table2(idx_table, rng1, rng2, n1, n2)

      ! This will need to be changed for the MR case. Instead of defining
      ! continuous ranges of orbitals, you will pass in arrays of occ and unocc.
      ! The index tables will have all dimensions equal to norb.

            integer, intent(in) :: n1, n2
            integer, intent(in) :: rng1(2), rng2(2)

            integer, intent(inout) :: idx_table(n1,n2)

            integer :: kout
            integer :: p, q

            idx_table = 0
            if (rng1(1) > 0 .and. rng2(1) < 0) then ! p < q
               kout = 1
               do p = rng1(1), rng1(2)
                  do q = p-rng2(1), rng2(2)
                     idx_table(p,q) = kout
                     kout = kout + 1
                  end do
               end do
            else ! p, q
               kout = 1
               do p = rng1(1), rng1(2)
                  do q = rng2(1), rng2(2)
                     idx_table(p,q) = kout
                     kout = kout + 1
                  end do
               end do
            end if

      end subroutine get_index_table2

      subroutine sort2(excits, amps, loc_arr, idx_table, idims, n1, n2, nloc, n2p, offset, resid)

            integer, intent(in) :: n1, n2, nloc, n2p
            integer, intent(in) :: idims(2), offset(2)
            integer, intent(in) :: idx_table(n1,n2)

            integer, intent(inout) :: loc_arr(2,nloc)
            integer, intent(inout) :: excits(n2p,4)
            real(kind=8), intent(inout) :: amps(n2p)
            real(kind=8), intent(inout), optional :: resid(n2p)

            integer :: idet
            integer :: p, q
            integer :: p1, q1, p2, q2
            integer :: pq1, pq2
            integer, allocatable :: temp(:), idx(:)

            allocate(temp(n2p),idx(n2p))
            do idet = 1,n2p
               p = excits(idet,idims(1)); q = excits(idet,idims(2));
               temp(idet) = idx_table(p-offset(1),q-offset(2))
            end do
            call argsort(temp,idx)
            excits = excits(idx,:)
            amps = amps(idx)
            if (present(resid)) resid = resid(idx)
            deallocate(temp,idx)

            loc_arr(1,:) = 1; loc_arr(2,:) = 0;
            !!! WARNING: THERE IS A MEMORY LEAK HERE! pq2 is used below but is not set if n2p <= 1
            !if (n2p <= 1) print*, "WARNING: potential memory leakage in sort3 function. pq2 set to 0"
            pq2 = 0
            do idet = 1, n2p-1
               p1 = excits(idet,idims(1));   q1 = excits(idet,idims(2));
               p2 = excits(idet+1,idims(1)); q2 = excits(idet+1,idims(2));
               pq1 = idx_table(p1-offset(1),q1-offset(2))
               pq2 = idx_table(p2-offset(1),q2-offset(2))
               if (pq1 /= pq2) then
                  loc_arr(2,pq1) = idet
                  loc_arr(1,pq2) = idet+1
               end if
            end do
            if (n2p > 1) then
               loc_arr(2,pq2) = n2p
            end if
      end subroutine sort2

      subroutine get_index_table3(idx_table, rng1, rng2, rng3, n1, n2, n3)

            integer, intent(in) :: n1, n2, n3
            integer, intent(in) :: rng1(2), rng2(2), rng3(2)

            integer, intent(inout) :: idx_table(n1,n2,n3)

            integer :: kout
            integer :: p, q, r

            idx_table = 0
            if (rng1(1) > 0 .and. rng2(1) < 0 .and. rng3(1) < 0) then ! p < q < r
               kout = 1
               do p = rng1(1), rng1(2)
                  do q = p-rng2(1), rng2(2)
                     do r = q-rng3(1), rng3(2)
                        idx_table(p,q,r) = kout
                        kout = kout + 1
                     end do
                  end do
               end do
            elseif (rng1(1) > 0 .and. rng2(1) > 0 .and. rng3(1) < 0) then ! p, q < r
               kout = 1
               do p = rng1(1), rng1(2)
                  do q = rng2(1), rng2(2)
                     do r = q-rng3(1), rng3(2)
                        idx_table(p,q,r) = kout
                        kout = kout + 1
                     end do
                  end do
               end do
            elseif (rng1(1) > 0 .and. rng2(1) < 0 .and. rng3(1) > 0) then ! p < q, r
               kout = 1
               do p = rng1(1), rng1(2)
                  do q = p-rng2(1), rng2(2)
                     do r = rng3(1), rng3(2)
                        idx_table(p,q,r) = kout
                        kout = kout + 1
                     end do
                  end do
               end do
            else ! p, q, r
               kout = 1
               do p = rng1(1), rng1(2)
                  do q = rng2(1), rng2(2)
                     do r = rng3(1), rng3(2)
                        idx_table(p,q,r) = kout
                        kout = kout + 1
                     end do
                  end do
               end do
            end if

      end subroutine get_index_table3

      subroutine sort3(excits, amps, loc_arr, idx_table, idims, n1, n2, n3, nloc, n2p, offset, resid)

            integer, intent(in) :: n1, n2, n3, nloc, n2p
            integer, intent(in) :: idims(3), offset(3)
            integer, intent(in) :: idx_table(n1,n2,n3)

            integer, intent(inout) :: loc_arr(2,nloc)
            integer, intent(inout) :: excits(n2p,4)
            real(kind=8), intent(inout) :: amps(n2p)
            real(kind=8), intent(inout), optional :: resid(n2p)

            integer :: idet
            integer :: p, q, r
            integer :: p1, q1, r1, p2, q2, r2
            integer :: pqr1, pqr2
            integer, allocatable :: temp(:), idx(:)

            allocate(temp(n2p),idx(n2p))
            do idet = 1, n2p
               p = excits(idet,idims(1)); q = excits(idet,idims(2)); r = excits(idet,idims(3));
               temp(idet) = idx_table(p-offset(1),q-offset(2),r-offset(3))
            end do
            call argsort(temp, idx)
            excits = excits(idx,:)
            amps = amps(idx)
            if (present(resid)) resid = resid(idx)
            deallocate(temp,idx)

            loc_arr(1,:) = 1; loc_arr(2,:) = 0;
            !!! WARNING: THERE IS A MEMORY LEAK HERE! pqr2 is used below but is not set if n2p <= 1
            !if (n2p <= 1) print*, "WARNING: potential memory leakage in sort3 function. pqr2 set to 0"
            pqr2 = 0
            do idet = 1, n2p-1
               p1 = excits(idet,idims(1));   q1 = excits(idet,idims(2));   r1 = excits(idet,idims(3));
               p2 = excits(idet+1,idims(1)); q2 = excits(idet+1,idims(2)); r2 = excits(idet+1,idims(3));
               pqr1 = idx_table(p1-offset(1),q1-offset(2),r1-offset(3))
               pqr2 = idx_table(p2-offset(1),q2-offset(2),r2-offset(3))
               if (pqr1 /= pqr2) then
                  loc_arr(2,pqr1) = idet
                  loc_arr(1,pqr2) = idet+1
               end if
            end do
            if (n2p > 1) then
               loc_arr(2,pqr2) = n2p
            end if
      end subroutine sort3

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

END MODULE ccsd_p
