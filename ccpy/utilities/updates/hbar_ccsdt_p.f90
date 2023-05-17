module hbar_ccsdt_p

        implicit none
        
        contains

              subroutine add_t3_h2a_vooo(h2a_vooo,&
                                         t3a_amps, t3a_excits,&
                                         t3b_amps, t3b_excits,&
                                         h2a_oovv, h2b_oovv,&
                                         n3aaa, n3aab,&
                                         noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa, n3aab

                  integer, intent(in) :: t3a_excits(6,n3aaa)
                  real(kind=8), intent(in) :: t3a_amps(n3aaa)
                  integer, intent(in) :: t3b_excits(6,n3aab)
                  real(kind=8), intent(in) :: t3b_amps(n3aab)

                  real(kind=8), intent(in) :: h2a_oovv(noa,noa,nua,nua)
                  real(kind=8), intent(in) :: h2b_oovv(noa,nob,nua,nub)

                  real(kind=8), intent(inout) :: h2a_vooo(nua,noa,noa,noa)
                  !f2py intent(in,out) :: h2a_vooo(0:nua-1,0:noa-1,0:noa-1,0:noa-1)

                  real(kind=8) :: t_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  ! premultiply h2a_vooo by 1/2 to account for later antisymmetrizer
                  ! remember to update only permutationally unique elements of array
                  h2a_vooo = 0.5d0 * h2a_vooo
                  do idet = 1, n3aaa
                      t_amp = t3a_amps(idet)
                      ! I2A(amij) <- A(ij) [A(n/ij)A(a/ef) h2a(mnef) * t3a(aefijn)]
                      a = t3a_excits(1,idet); e = t3a_excits(2,idet); f = t3a_excits(3,idet);
                      i = t3a_excits(4,idet); j = t3a_excits(5,idet); n = t3a_excits(6,idet);
                      h2a_vooo(a,:,i,j) = h2a_vooo(a,:,i,j) + h2a_oovv(:,n,e,f) * t_amp ! (1)
                      h2a_vooo(a,:,j,n) = h2a_vooo(a,:,j,n) + h2a_oovv(:,i,e,f) * t_amp ! (in)
                      h2a_vooo(a,:,i,n) = h2a_vooo(a,:,i,n) - h2a_oovv(:,j,e,f) * t_amp ! (jn)
                      h2a_vooo(e,:,i,j) = h2a_vooo(e,:,i,j) - h2a_oovv(:,n,a,f) * t_amp ! (ae)
                      h2a_vooo(e,:,j,n) = h2a_vooo(e,:,j,n) - h2a_oovv(:,i,a,f) * t_amp ! (in)(ae)
                      h2a_vooo(e,:,i,n) = h2a_vooo(e,:,i,n) + h2a_oovv(:,j,a,f) * t_amp ! (jn)(ae)
                      h2a_vooo(f,:,i,j) = h2a_vooo(f,:,i,j) - h2a_oovv(:,n,e,a) * t_amp ! (af)
                      h2a_vooo(f,:,j,n) = h2a_vooo(f,:,j,n) - h2a_oovv(:,i,e,a) * t_amp ! (in)(af)
                      h2a_vooo(f,:,i,n) = h2a_vooo(f,:,i,n) + h2a_oovv(:,j,e,a) * t_amp ! (jn)(af)
                  end do

                  do idet = 1, n3aab
                      t_amp = t3b_amps(idet)
                      ! I2A(amij) <- A(ij) [A(ae) h2b(mnef) * t3b(aefijn)]
                      a = t3b_excits(1,idet); e = t3b_excits(2,idet); f = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); j = t3b_excits(5,idet); n = t3b_excits(6,idet);
                      h2a_vooo(a,:,i,j) = h2a_vooo(a,:,i,j) + h2b_oovv(:,n,e,f) * t_amp ! (1)
                      h2a_vooo(e,:,i,j) = h2a_vooo(e,:,i,j) - h2b_oovv(:,n,a,f) * t_amp ! (ae)
                  end do

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

              end subroutine add_t3_h2a_vooo

              subroutine add_t3_h2a_vvov(h2a_vvov,&
                                         t3a_amps, t3a_excits,&
                                         t3b_amps, t3b_excits,&
                                         h2a_oovv, h2b_oovv,&
                                         n3aaa, n3aab,&
                                         noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa, n3aab

                  integer, intent(in) :: t3a_excits(6,n3aaa)
                  real(kind=8), intent(in) :: t3a_amps(n3aaa)
                  integer, intent(in) :: t3b_excits(6,n3aab)
                  real(kind=8), intent(in) :: t3b_amps(n3aab)

                  real(kind=8), intent(in) :: h2a_oovv(noa,noa,nua,nua)
                  real(kind=8), intent(in) :: h2b_oovv(noa,nob,nua,nub)

                  real(kind=8), intent(inout) :: h2a_vvov(nua,nua,noa,nua)
                  !f2py intent(in,out) :: h2a_vvov(0:nua-1,0:nua-1,0:noa-1,0:nua-1)

                  real(kind=8) :: t_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  ! premultiply h2a_vvov by 1/2 to account for later antisymmetrizer
                  ! remember to update only permutationally unique elements of array
                  h2a_vvov = 0.5d0 * h2a_vvov
                  do idet = 1, n3aaa
                      t_amp = t3a_amps(idet)
                      ! I2A(abie) <- A(ab) [A(i/mn)A(f/ab) -h2a(mnef) * t3a(abfimn)]
                      a = t3a_excits(1,idet); b = t3a_excits(2,idet); f = t3a_excits(3,idet);
                      i = t3a_excits(4,idet); m = t3a_excits(5,idet); n = t3a_excits(6,idet);
                      h2a_vvov(a,b,i,:) = h2a_vvov(a,b,i,:) - h2a_oovv(m,n,:,f) * t_amp ! (1)
                      h2a_vvov(a,b,m,:) = h2a_vvov(a,b,m,:) + h2a_oovv(i,n,:,f) * t_amp ! (im)
                      h2a_vvov(a,b,n,:) = h2a_vvov(a,b,n,:) + h2a_oovv(m,i,:,f) * t_amp ! (in)
                      h2a_vvov(b,f,i,:) = h2a_vvov(b,f,i,:) - h2a_oovv(m,n,:,a) * t_amp ! (af)
                      h2a_vvov(b,f,m,:) = h2a_vvov(b,f,m,:) + h2a_oovv(i,n,:,a) * t_amp ! (im)(af)
                      h2a_vvov(b,f,n,:) = h2a_vvov(b,f,n,:) + h2a_oovv(m,i,:,a) * t_amp ! (in)(af)
                      h2a_vvov(a,f,i,:) = h2a_vvov(a,f,i,:) + h2a_oovv(m,n,:,b) * t_amp ! (bf)
                      h2a_vvov(a,f,m,:) = h2a_vvov(a,f,m,:) - h2a_oovv(i,n,:,b) * t_amp ! (im)(bf)
                      h2a_vvov(a,f,n,:) = h2a_vvov(a,f,n,:) - h2a_oovv(m,i,:,b) * t_amp ! (in)(bf)
                  end do

                  do idet = 1, n3aab
                      t_amp = t3b_amps(idet)
                      ! I2A(abie) <- A(ab) [A(im) -h2b(mnef) * t3b(abfimn)]
                      a = t3b_excits(1,idet); b = t3b_excits(2,idet); f = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); m = t3b_excits(5,idet); n = t3b_excits(6,idet);
                      h2a_vvov(a,b,i,:) = h2a_vvov(a,b,i,:) - h2b_oovv(m,n,:,f) * t_amp ! (1)
                      h2a_vvov(a,b,m,:) = h2a_vvov(a,b,m,:) + h2b_oovv(i,n,:,f) * t_amp ! (im)
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

              end subroutine add_t3_h2a_vvov

              subroutine add_t3_h2b_vooo(h2b_vooo,&
                                         t3b_amps, t3b_excits,&
                                         t3c_amps, t3c_excits,&
                                         h2b_oovv, h2c_oovv,&
                                         n3aab, n3abb,&
                                         noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab, n3abb

                  integer, intent(in) :: t3b_excits(6,n3aab)
                  real(kind=8), intent(in) :: t3b_amps(n3aab)
                  integer, intent(in) :: t3c_excits(6,n3abb)
                  real(kind=8), intent(in) :: t3c_amps(n3abb)

                  real(kind=8), intent(in) :: h2b_oovv(noa,nob,nua,nub)
                  real(kind=8), intent(in) :: h2c_oovv(nob,nob,nub,nub)

                  real(kind=8), intent(inout) :: h2b_vooo(nua,nob,noa,nob)
                  !f2py intent(in,out) :: h2b_vooo(0:nua-1,0:nob-1,0:noa-1,0:nob-1)

                  real(kind=8) :: t_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  do idet = 1, n3aab
                      t_amp = t3b_amps(idet)
                      ! I2B(amij) <- A(af)A(in) h2b(nmfe) * t3b(afeinj)
                      a = t3b_excits(1,idet); f = t3b_excits(2,idet); e = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); n = t3b_excits(5,idet); j = t3b_excits(6,idet);
                      h2b_vooo(a,:,i,j) = h2b_vooo(a,:,i,j) + h2b_oovv(n,:,f,e) * t_amp ! (1)
                      h2b_vooo(f,:,i,j) = h2b_vooo(f,:,i,j) - h2b_oovv(n,:,a,e) * t_amp ! (af)
                      h2b_vooo(a,:,n,j) = h2b_vooo(a,:,n,j) - h2b_oovv(i,:,f,e) * t_amp ! (in)
                      h2b_vooo(f,:,n,j) = h2b_vooo(f,:,n,j) + h2b_oovv(i,:,a,e) * t_amp ! (af)(in)
                  end do
                  do idet = 1, n3abb
                      t_amp = t3c_amps(idet)
                      ! I2B(amij) <- A(jn) h2c(nmfe) * t3c(afeinj)
                      a = t3c_excits(1,idet); f = t3c_excits(2,idet); e = t3c_excits(3,idet);
                      i = t3c_excits(4,idet); n = t3c_excits(5,idet); j = t3c_excits(6,idet);
                      h2b_vooo(a,:,i,j) = h2b_vooo(a,:,i,j) + h2c_oovv(n,:,f,e) * t_amp ! (1)
                      h2b_vooo(a,:,i,n) = h2b_vooo(a,:,i,n) - h2c_oovv(j,:,f,e) * t_amp ! (jn)
                  end do

              end subroutine add_t3_h2b_vooo

              subroutine add_t3_h2b_ovoo(h2b_ovoo,&
                                         t3b_amps, t3b_excits,&
                                         t3c_amps, t3c_excits,&
                                         h2a_oovv, h2b_oovv,&
                                         n3aab, n3abb,&
                                         noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab, n3abb

                  integer, intent(in) :: t3b_excits(6,n3aab)
                  real(kind=8), intent(in) :: t3b_amps(n3aab)
                  integer, intent(in) :: t3c_excits(6,n3abb)
                  real(kind=8), intent(in) :: t3c_amps(n3abb)

                  real(kind=8), intent(in) :: h2a_oovv(noa,noa,nua,nua)
                  real(kind=8), intent(in) :: h2b_oovv(noa,nob,nua,nub)

                  real(kind=8), intent(inout) :: h2b_ovoo(noa,nub,noa,nob)
                  !f2py intent(in,out) :: h2b_ovoo(0:noa-1,0:nub-1,0:noa-1,0:nob-1)

                  real(kind=8) :: t_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  do idet = 1, n3aab
                      t_amp = t3b_amps(idet)
                      ! I2B(mbij) <- A(in) h2a(mnef) * t3b(efbinj)
                      e = t3b_excits(1,idet); f = t3b_excits(2,idet); b = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); n = t3b_excits(5,idet); j = t3b_excits(6,idet);
                      h2b_ovoo(:,b,i,j) = h2b_ovoo(:,b,i,j) + h2a_oovv(:,n,e,f) * t_amp ! (1)
                      h2b_ovoo(:,b,n,j) = h2b_ovoo(:,b,n,j) - h2a_oovv(:,i,e,f) * t_amp ! (in)
                  end do
                  do idet = 1, n3abb
                      t_amp = t3c_amps(idet)
                      ! I2B(mbij) <- A(bf)A(jn) h2B(mnef) * t3c(efbinj)
                      e = t3c_excits(1,idet); f = t3c_excits(2,idet); b = t3c_excits(3,idet);
                      i = t3c_excits(4,idet); n = t3c_excits(5,idet); j = t3c_excits(6,idet);
                      h2b_ovoo(:,b,i,j) = h2b_ovoo(:,b,i,j) + h2b_oovv(:,n,e,f) * t_amp ! (1)
                      h2b_ovoo(:,f,i,j) = h2b_ovoo(:,f,i,j) - h2b_oovv(:,n,e,b) * t_amp ! (bf)
                      h2b_ovoo(:,b,i,n) = h2b_ovoo(:,b,i,n) - h2b_oovv(:,j,e,f) * t_amp ! (jn)
                      h2b_ovoo(:,f,i,n) = h2b_ovoo(:,f,i,n) + h2b_oovv(:,j,e,b) * t_amp ! (bf)(jn)
                  end do

              end subroutine add_t3_h2b_ovoo

              subroutine add_t3_h2b_vvov(h2b_vvov,&
                                         t3b_amps, t3b_excits,&
                                         t3c_amps, t3c_excits,&
                                         h2b_oovv, h2c_oovv,&
                                         n3aab, n3abb,&
                                         noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab, n3abb

                  integer, intent(in) :: t3b_excits(6,n3aab)
                  real(kind=8), intent(in) :: t3b_amps(n3aab)
                  integer, intent(in) :: t3c_excits(6,n3abb)
                  real(kind=8), intent(in) :: t3c_amps(n3abb)

                  real(kind=8), intent(in) :: h2b_oovv(noa,nob,nua,nub)
                  real(kind=8), intent(in) :: h2c_oovv(nob,nob,nub,nub)

                  real(kind=8), intent(inout) :: h2b_vvov(nua,nub,noa,nub)
                  !f2py intent(in,out) :: h2b_vvov(0:nua-1,0:nub-1,0:noa-1,0:nub-1)

                  real(kind=8) :: t_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  do idet = 1, n3aab
                      t_amp = t3b_amps(idet)
                      ! I2B(abie) <- A(af)A(in) -h2b(nmfe) * t3b(afbinm)
                      a = t3b_excits(1,idet); f = t3b_excits(2,idet); b = t3b_excits(3,idet);
                      i = t3b_excits(4,idet); n = t3b_excits(5,idet); m = t3b_excits(6,idet);
                      h2b_vvov(a,b,i,:) = h2b_vvov(a,b,i,:) - h2b_oovv(n,m,f,:) * t_amp ! (1)
                      h2b_vvov(f,b,i,:) = h2b_vvov(f,b,i,:) + h2b_oovv(n,m,a,:) * t_amp ! (af)
                      h2b_vvov(a,b,n,:) = h2b_vvov(a,b,n,:) + h2b_oovv(i,m,f,:) * t_amp ! (in)
                      h2b_vvov(f,b,n,:) = h2b_vvov(f,b,n,:) - h2b_oovv(i,m,a,:) * t_amp ! (af)(in)
                  end do
                  do idet = 1, n3abb
                      t_amp = t3c_amps(idet)
                      ! I2B(abie) <- A(bf) -h2c(nmfe) * t3c(afbinm)
                      a = t3c_excits(1,idet); f = t3c_excits(2,idet); b = t3c_excits(3,idet);
                      i = t3c_excits(4,idet); n = t3c_excits(5,idet); m = t3c_excits(6,idet);
                      h2b_vvov(a,b,i,:) = h2b_vvov(a,b,i,:) - h2c_oovv(n,m,f,:) * t_amp ! (1)
                      h2b_vvov(a,f,i,:) = h2b_vvov(a,f,i,:) + h2c_oovv(n,m,b,:) * t_amp ! (bf)
                  end do

              end subroutine add_t3_h2b_vvov

              subroutine add_t3_h2b_vvvo(h2b_vvvo,&
                                         t3b_amps, t3b_excits,&
                                         t3c_amps, t3c_excits,&
                                         h2a_oovv, h2b_oovv,&
                                         n3aab, n3abb,&
                                         noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab, n3abb

                  integer, intent(in) :: t3b_excits(6,n3aab)
                  real(kind=8), intent(in) :: t3b_amps(n3aab)
                  integer, intent(in) :: t3c_excits(6,n3abb)
                  real(kind=8), intent(in) :: t3c_amps(n3abb)

                  real(kind=8), intent(in) :: h2a_oovv(noa,noa,nua,nua)
                  real(kind=8), intent(in) :: h2b_oovv(noa,nob,nua,nub)

                  real(kind=8), intent(inout) :: h2b_vvvo(nua,nub,nua,nob)
                  !f2py intent(in,out) :: h2b_vvvo(0:nua-1,0:nub-1,0:nua-1,0:nob-1)

                  real(kind=8) :: t_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  do idet = 1, n3aab
                      t_amp = t3b_amps(idet)
                      ! I2B(abej) <- A(af) -h2a(mnef) * t3b(afbmnj)
                      a = t3b_excits(1,idet); f = t3b_excits(2,idet); b = t3b_excits(3,idet);
                      m = t3b_excits(4,idet); n = t3b_excits(5,idet); j = t3b_excits(6,idet);
                      h2b_vvvo(a,b,:,j) = h2b_vvvo(a,b,:,j) - h2a_oovv(m,n,:,f) * t_amp ! (1)
                      h2b_vvvo(f,b,:,j) = h2b_vvvo(f,b,:,j) + h2a_oovv(m,n,:,a) * t_amp ! (af)
                  end do
                  do idet = 1, n3abb
                      t_amp = t3c_amps(idet)
                      ! I2B(abej) <- A(bf)A(jn) -h2b(mnef) * t3c(afbmnj)
                      a = t3c_excits(1,idet); f = t3c_excits(2,idet); b = t3c_excits(3,idet);
                      m = t3c_excits(4,idet); n = t3c_excits(5,idet); j = t3c_excits(6,idet);
                      h2b_vvvo(a,b,:,j) = h2b_vvvo(a,b,:,j) - h2b_oovv(m,n,:,f) * t_amp ! (1)
                      h2b_vvvo(a,f,:,j) = h2b_vvvo(a,f,:,j) + h2b_oovv(m,n,:,b) * t_amp ! (bf)
                      h2b_vvvo(a,b,:,n) = h2b_vvvo(a,b,:,n) + h2b_oovv(m,j,:,f) * t_amp ! (jn)
                      h2b_vvvo(a,f,:,n) = h2b_vvvo(a,f,:,n) - h2b_oovv(m,j,:,b) * t_amp ! (bf)(jn)
                  end do

              end subroutine add_t3_h2b_vvvo

              subroutine add_t3_h2c_vooo(h2c_vooo,&
                                         t3c_amps, t3c_excits,&
                                         t3d_amps, t3d_excits,&
                                         h2b_oovv, h2c_oovv,&
                                         n3abb, n3bbb,&
                                         noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3abb, n3bbb

                  integer, intent(in) :: t3c_excits(6,n3abb)
                  real(kind=8), intent(in) :: t3c_amps(n3abb)
                  integer, intent(in) :: t3d_excits(6,n3bbb)
                  real(kind=8), intent(in) :: t3d_amps(n3bbb)

                  real(kind=8), intent(in) :: h2b_oovv(noa,nob,nua,nub)
                  real(kind=8), intent(in) :: h2c_oovv(nob,nob,nub,nub)

                  real(kind=8), intent(inout) :: h2c_vooo(nub,nob,nob,nob)
                  !f2py intent(in,out) :: h2c_vooo(0:nub-1,0:nob-1,0:nob-1,0:nob-1)

                  real(kind=8) :: t_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  ! premultiply h2c_vooo by 1/2 to account for later antisymmetrizer
                  ! remember to update only permutationally unique elements of array
                  h2c_vooo = 0.5d0 * h2c_vooo
                  do idet = 1, n3bbb
                      t_amp = t3d_amps(idet)
                      ! I2C(amij) <- A(ij) [A(n/ij)A(a/ef) h2c(mnef) * t3d(aefijn)]
                      a = t3d_excits(1,idet); e = t3d_excits(2,idet); f = t3d_excits(3,idet);
                      i = t3d_excits(4,idet); j = t3d_excits(5,idet); n = t3d_excits(6,idet);
                      h2c_vooo(a,:,i,j) = h2c_vooo(a,:,i,j) + h2c_oovv(:,n,e,f) * t_amp ! (1)
                      h2c_vooo(a,:,j,n) = h2c_vooo(a,:,j,n) + h2c_oovv(:,i,e,f) * t_amp ! (in)
                      h2c_vooo(a,:,i,n) = h2c_vooo(a,:,i,n) - h2c_oovv(:,j,e,f) * t_amp ! (jn)
                      h2c_vooo(e,:,i,j) = h2c_vooo(e,:,i,j) - h2c_oovv(:,n,a,f) * t_amp ! (ae)
                      h2c_vooo(e,:,j,n) = h2c_vooo(e,:,j,n) - h2c_oovv(:,i,a,f) * t_amp ! (in)(ae)
                      h2c_vooo(e,:,i,n) = h2c_vooo(e,:,i,n) + h2c_oovv(:,j,a,f) * t_amp ! (jn)(ae)
                      h2c_vooo(f,:,i,j) = h2c_vooo(f,:,i,j) - h2c_oovv(:,n,e,a) * t_amp ! (af)
                      h2c_vooo(f,:,j,n) = h2c_vooo(f,:,j,n) - h2c_oovv(:,i,e,a) * t_amp ! (in)(af)
                      h2c_vooo(f,:,i,n) = h2c_vooo(f,:,i,n) + h2c_oovv(:,j,e,a) * t_amp ! (jn)(af)
                  end do

                  do idet = 1, n3abb
                      t_amp = t3c_amps(idet)
                      ! I2C(amij) <- A(ij) [A(ae) h2b(nmfe) * t3c(faenij)]
                      f = t3c_excits(1,idet); a = t3c_excits(2,idet); e = t3c_excits(3,idet);
                      n = t3c_excits(4,idet); i = t3c_excits(5,idet); j = t3c_excits(6,idet);
                      h2c_vooo(a,:,i,j) = h2c_vooo(a,:,i,j) + h2b_oovv(n,:,f,e) * t_amp ! (1)
                      h2c_vooo(e,:,i,j) = h2c_vooo(e,:,i,j) - h2b_oovv(n,:,f,a) * t_amp ! (ae)
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

              end subroutine add_t3_h2c_vooo

              subroutine add_t3_h2c_vvov(h2c_vvov,&
                                         t3c_amps, t3c_excits,&
                                         t3d_amps, t3d_excits,&
                                         h2b_oovv, h2c_oovv,&
                                         n3abb, n3bbb,&
                                         noa, nua, nob, nub)

                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3abb, n3bbb

                  integer, intent(in) :: t3c_excits(6,n3abb)
                  real(kind=8), intent(in) :: t3c_amps(n3abb)
                  integer, intent(in) :: t3d_excits(6,n3bbb)
                  real(kind=8), intent(in) :: t3d_amps(n3bbb)

                  real(kind=8), intent(in) :: h2b_oovv(noa,nob,nua,nub)
                  real(kind=8), intent(in) :: h2c_oovv(nob,nob,nub,nub)

                  real(kind=8), intent(inout) :: h2c_vvov(nub,nub,nob,nub)
                  !f2py intent(in,out) :: h2c_vvov(0:nub-1,0:nub-1,0:nob-1,0:nub-1)

                  real(kind=8) :: t_amp
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet

                  ! premultiply h2c_vvov by 1/2 to account for later antisymmetrizer
                  ! remember to update only permutationally unique elements of array
                  h2c_vvov = 0.5d0 * h2c_vvov
                  do idet = 1, n3bbb
                      t_amp = t3d_amps(idet)
                      ! I2A(abie) <- A(ab) [A(i/mn)A(f/ab) -h2a(mnef) * t3a(abfimn)]
                      a = t3d_excits(1,idet); b = t3d_excits(2,idet); f = t3d_excits(3,idet);
                      i = t3d_excits(4,idet); m = t3d_excits(5,idet); n = t3d_excits(6,idet);
                      h2c_vvov(a,b,i,:) = h2c_vvov(a,b,i,:) - h2c_oovv(m,n,:,f) * t_amp ! (1)
                      h2c_vvov(a,b,m,:) = h2c_vvov(a,b,m,:) + h2c_oovv(i,n,:,f) * t_amp ! (im)
                      h2c_vvov(a,b,n,:) = h2c_vvov(a,b,n,:) + h2c_oovv(m,i,:,f) * t_amp ! (in)
                      h2c_vvov(b,f,i,:) = h2c_vvov(b,f,i,:) - h2c_oovv(m,n,:,a) * t_amp ! (af)
                      h2c_vvov(b,f,m,:) = h2c_vvov(b,f,m,:) + h2c_oovv(i,n,:,a) * t_amp ! (im)(af)
                      h2c_vvov(b,f,n,:) = h2c_vvov(b,f,n,:) + h2c_oovv(m,i,:,a) * t_amp ! (in)(af)
                      h2c_vvov(a,f,i,:) = h2c_vvov(a,f,i,:) + h2c_oovv(m,n,:,b) * t_amp ! (bf)
                      h2c_vvov(a,f,m,:) = h2c_vvov(a,f,m,:) - h2c_oovv(i,n,:,b) * t_amp ! (im)(bf)
                      h2c_vvov(a,f,n,:) = h2c_vvov(a,f,n,:) - h2c_oovv(m,i,:,b) * t_amp ! (in)(bf)
                  end do

                  do idet = 1, n3abb
                      t_amp = t3c_amps(idet)
                      ! I2C(abie) <- A(ab) [A(im) -h2b(nmfe) * t3c(fabnim)]
                      f = t3c_excits(1,idet); a = t3c_excits(2,idet); b = t3c_excits(3,idet);
                      n = t3c_excits(4,idet); i = t3c_excits(5,idet); m = t3c_excits(6,idet);
                      h2c_vvov(a,b,i,:) = h2c_vvov(a,b,i,:) - h2b_oovv(n,m,f,:) * t_amp ! (1)
                      h2c_vvov(a,b,m,:) = h2c_vvov(a,b,m,:) + h2b_oovv(n,i,f,:) * t_amp ! (im)
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

              end subroutine add_t3_h2c_vvov

end module hbar_ccsdt_p
