module sort_t3

      implicit none

      contains

              subroutine sort_t3a_h(t3a_excits, t3a_amps, ID, XiXjXk_table, noa, nua, n3aaa)

                      integer, intent(in) :: n3aaa, noa, nua
                      integer, intent(inout) :: t3a_excits(6,n3aaa)
                      !f2py intent(in,out) :: t3a_excits(6,n3aaa)
                      real(kind=8), intent(inout) :: t3a_amps(n3aaa)
                      !f2py intent(in,out) :: t3a_amps(n3aaa)

                      integer, intent(out) :: XiXjXk_table(noa,noa,noa)
                      integer, intent(out) :: ID(noa*(noa-1)*(noa-2)/6,2)

                      integer :: i, j, k, a, b, c
                      integer :: i1, j1, k1, a1, b1, c1
                      integer :: i2, j2, k2, a2, b2, c2
                      integer :: kout, ijk, ijk1, ijk2, idet
                      integer, allocatable :: temp(:), idx(:)

                      XiXjXk_table = 0
                      kout = 1
                      do i = 1, noa
                         do j = i+1, noa
                            do k = j+1, noa
                               XiXjXk_table(i,j,k) = kout
                               XiXjXk_table(j,k,i) = kout
                               XiXjXk_table(k,i,j) = kout
                               XiXjXk_table(i,k,j) = -kout
                               XiXjXk_table(j,i,k) = -kout
                               XiXjXk_table(k,j,i) = -kout
                               kout = kout + 1
                            end do
                         end do
                      end do

                      allocate(temp(n3aaa),idx(n3aaa))
                      do idet = 1, n3aaa
                         i = t3a_excits(4,idet); j = t3a_excits(5,idet); k = t3a_excits(6,idet);
                         ijk = XiXjXk_table(i,j,k)
                         temp(idet) = ijk
                      end do
                      call argsort(temp, idx)
                      t3a_excits = t3a_excits(:,idx)
                      t3a_amps = t3a_amps(idx)
                      deallocate(temp,idx)

                      ID = 1
                      do idet = 2, n3aaa
                         i1 = t3a_excits(4,idet-1); j1 = t3a_excits(5,idet-1); k1 = t3a_excits(6,idet-1);
                         i2 = t3a_excits(4,idet);   j2 = t3a_excits(5,idet);   k2 = t3a_excits(6,idet);

                         ijk1 = XiXjXk_table(i1,j1,k1)
                         ijk2 = XiXjXk_table(i2,j2,k2)
                         if (ijk1 /= ijk2) then
                                 ID(ijk1,2) = idet - 1
                                 ID(ijk2,1) = idet
                         end if
                      end do
                      ID(ijk2,2) = n3aaa

              end subroutine sort_t3a_h

              subroutine sort_t3b_h(t3b_excits, t3b_amps, ID, Eck_table, XiXj_table, noa, nua, nob, nub, n3aab)

                      integer, intent(in) :: n3aab, noa, nua, nob, nub
                      integer, intent(inout) :: t3b_excits(6,n3aab)
                      !f2py intent(in,out) :: t3b_excits(6,n3aab)
                      real(kind=8), intent(inout) :: t3b_amps(n3aab)
                      !f2py intent(in,out) :: t3b_amps(n3aab)

                      integer, intent(out) :: Eck_table(nub,nob)
                      integer, intent(out) :: XiXj_table(noa,noa)
                      integer, intent(out) :: ID(nub*nob,noa*(noa-1)/2,2)

                      integer :: i, j, k, a, b, c
                      integer :: i1, i2, j1, j2, c1, c2, k1, k2
                      integer:: ij, ib, ib1, ib2, ij1, ij2, kout, idet, num_ij_ib
                      integer :: beta_locs(nub*nob,2)
                      integer, allocatable :: temp(:), idx(:)

                      Eck_table=0
                      kout = 1
                      do c = 1, nub
                         do k = 1, nob
                            Eck_table(c,k) = kout
                            kout = kout + 1
                         end do
                      end do
                      XiXj_table=0
                      kout = 1
                      do i = 1, noa
                         do j = i+1, noa
                            XiXj_table(i,j) = kout
                            XiXj_table(j,i) = -kout
                            kout = kout + 1
                         end do
                      end do

                      allocate(temp(n3aab),idx(n3aab))
                      do idet = 1, n3aab
                         c = t3b_excits(3,idet); k = t3b_excits(6,idet);
                         ib = Eck_table(c,k)
                         temp(idet) = ib
                      end do
                      call argsort(temp, idx)
                      t3b_excits = t3b_excits(:,idx)
                      t3b_amps = t3b_amps(idx)
                      deallocate(temp,idx)

                      beta_locs = 1
                      do idet = 2, n3aab
                         c1 = t3b_excits(3,idet-1); k1 = t3b_excits(6,idet-1);
                         c2 = t3b_excits(3,idet);   k2 = t3b_excits(6,idet);
                         ib1 = Eck_table(c1,k1)
                         ib2 = Eck_table(c2,k2)
                         if (ib1/=ib2) then
                                 beta_locs(ib1,2) = idet - 1
                                 beta_locs(ib2,1) = idet
                         end if
                      end do
                      beta_locs(ib2,2) = n3aab

                      ID = 0
                      do c = 1,nub
                         do k = 1,nob
                            ib = Eck_table(c,k)
                            if (beta_locs(ib,1) > beta_locs(ib,2)) cycle ! skip if beta block is empty

                            num_ij_ib = beta_locs(ib,2) - beta_locs(ib,1) + 1

                            allocate(temp(num_ij_ib), idx(num_ij_ib))
                            kout = 1
                            do idet = beta_locs(ib,1), beta_locs(ib,2)
                               i = t3b_excits(4,idet); j = t3b_excits(5,idet);
                               ij = XiXj_table(i,j)
                               temp(kout) = ij
                               kout = kout + 1
                            end do
                            call argsort(temp,idx)
                            idx = idx + beta_locs(ib,1) - 1
                            t3b_excits(:,beta_locs(ib,1):beta_locs(ib,2)) = t3b_excits(:,idx)
                            t3b_amps(beta_locs(ib,1):beta_locs(ib,2)) = t3b_amps(idx)
                            deallocate(temp,idx)

                            ID(ib,:,1) = beta_locs(ib,1)
                            do ij = 1, num_ij_ib-1
                               idet = ij + beta_locs(ib,1)
                               i1 = t3b_excits(4,idet-1); j1 = t3b_excits(5,idet-1);
                               i2 = t3b_excits(4,idet);   j2 = t3b_excits(5,idet);
                               ij1 = XiXj_table(i1,j1)
                               ij2 = XiXj_table(i2,j2)
                               
                               if (ij1/=ij2) then
                                       ID(ib,ij1,2) = idet-1
                                       ID(ib,ij2,1) = idet
                               end if
                            end do
                            ID(ib,ij2,2) = beta_locs(ib,1) + num_ij_ib - 1
                         end do
                      end do

              end subroutine sort_t3b_h

              subroutine sort_t3b_p(t3b_excits, t3b_amps, ID, Eck_table, XaXb_table, noa, nua, nob, nub, n3aab)

                      integer, intent(in) :: n3aab, noa, nua, nob, nub
                      integer, intent(inout) :: t3b_excits(6,n3aab)
                      !f2py intent(in,out) :: t3b_excits(6,n3aab)
                      real(kind=8), intent(inout) :: t3b_amps(n3aab)
                      !f2py intent(in,out) :: t3b_amps(n3aab)

                      integer, intent(out) :: Eck_table(nub,nob)
                      integer, intent(out) :: XaXb_table(nua,nua)
                      integer, intent(out) :: ID(nub*nob,nua*(nua-1)/2,2)

                      integer :: i, j, k, a, b, c
                      integer :: a1, a2, b1, b2, c1, c2, k1, k2
                      integer:: ab, ib, ib1, ib2, ab1, ab2, kout, idet, num_ab_ib
                      integer :: beta_locs(nub*nob,2)
                      integer, allocatable :: temp(:), idx(:)

                      Eck_table=0
                      kout = 1
                      do c = 1, nub
                         do k = 1, nob
                            Eck_table(c,k) = kout
                            kout = kout + 1
                         end do
                      end do
                      XaXb_table=0
                      kout = 1
                      do a = 1, nua
                         do b = a+1, nua
                            XaXb_table(a,b) = kout
                            XaXb_table(b,a) = -kout
                            kout = kout + 1
                         end do
                      end do

                      allocate(temp(n3aab),idx(n3aab))
                      do idet = 1, n3aab
                         c = t3b_excits(3,idet); k = t3b_excits(6,idet);
                         ib = Eck_table(c,k)
                         temp(idet) = ib
                      end do
                      call argsort(temp, idx)
                      t3b_excits = t3b_excits(:,idx)
                      t3b_amps = t3b_amps(idx)
                      deallocate(temp,idx)

                      beta_locs = 1
                      do idet = 2, n3aab
                         c1 = t3b_excits(3,idet-1); k1 = t3b_excits(6,idet-1);
                         c2 = t3b_excits(3,idet);   k2 = t3b_excits(6,idet);
                         ib1 = Eck_table(c1,k1)
                         ib2 = Eck_table(c2,k2)
                         if (ib1/=ib2) then
                                 beta_locs(ib1,2) = idet - 1
                                 beta_locs(ib2,1) = idet
                         end if
                      end do
                      beta_locs(ib2,2) = n3aab

                      ID = 0
                      do c = 1,nub
                         do k = 1,nob
                            ib = Eck_table(c,k)
                            if (beta_locs(ib,1) > beta_locs(ib,2)) cycle ! skip if beta block is empty

                            num_ab_ib = beta_locs(ib,2) - beta_locs(ib,1) + 1

                            allocate(temp(num_ab_ib), idx(num_ab_ib))
                            kout = 1
                            do idet = beta_locs(ib,1), beta_locs(ib,2)
                               a = t3b_excits(1,idet); b = t3b_excits(2,idet);
                               ab = XaXb_table(a,b)
                               temp(kout) = ab
                               kout = kout + 1
                            end do
                            call argsort(temp,idx)
                            idx = idx + beta_locs(ib,1) - 1
                            t3b_excits(:,beta_locs(ib,1):beta_locs(ib,2)) = t3b_excits(:,idx)
                            t3b_amps(beta_locs(ib,1):beta_locs(ib,2)) = t3b_amps(idx)
                            deallocate(temp,idx)

                            ID(ib,:,1) = beta_locs(ib,1)
                            do ab = 1, num_ab_ib-1
                               idet = ab + beta_locs(ib,1)
                               a1 = t3b_excits(1,idet-1); b1 = t3b_excits(2,idet-1);
                               a2 = t3b_excits(1,idet);   b2 = t3b_excits(2,idet);
                               ab1 = XaXb_table(a1,b1)
                               ab2 = XaXb_table(a2,b2)
                               
                               if (ab1/=ab2) then
                                       ID(ib,ab1,2) = idet-1
                                       ID(ib,ab2,1) = idet
                               end if
                            end do
                            ID(ib,ab2,2) = beta_locs(ib,1) + num_ab_ib - 1
                         end do
                      end do

              end subroutine sort_t3b_p

              subroutine sort_t3c_h(t3c_excits, t3c_amps, ID, Eai_table, XjXk_table, noa, nua, nob, nub, n3abb)

                      integer, intent(in) :: n3abb, noa, nua, nob, nub
                      integer, intent(inout) :: t3c_excits(6,n3abb)
                      !f2py intent(in,out) :: t3c_excits(6,n3abb)
                      real(kind=8), intent(inout) :: t3c_amps(n3abb)
                      !f2py intent(in,out) :: t3c_amps(n3abb)

                      integer, intent(out) :: Eai_table(nua,noa)
                      integer, intent(out) :: XjXk_table(nob,nob)
                      integer, intent(out) :: ID(nua*noa,nob*(nob-1)/2,2)

                      integer :: i, j, k, a, b, c
                      integer :: j1, j2, k1, k2, a1, a2, i1, i2
                      integer:: jk, ia, ia1, ia2, jk1, jk2, kout, idet, num_jk_ia
                      integer :: alpha_locs(nua*noa,2)
                      integer, allocatable :: temp(:), idx(:)

                      Eai_table=0
                      kout = 1
                      do a = 1, nua
                         do i = 1, noa
                            Eai_table(a,i) = kout
                            kout = kout + 1
                         end do
                      end do
                      XjXk_table=0
                      kout = 1
                      do j = 1, nob
                         do k = j+1, nob
                            XjXk_table(j,k) = kout
                            XjXk_table(k,j) = -kout
                            kout = kout + 1
                         end do
                      end do

                      allocate(temp(n3abb),idx(n3abb))
                      do idet = 1, n3abb
                         a = t3c_excits(1,idet); i = t3c_excits(4,idet);
                         ia = Eai_table(a,i)
                         temp(idet) = ia
                      end do
                      call argsort(temp, idx)
                      t3c_excits = t3c_excits(:,idx)
                      t3c_amps = t3c_amps(idx)
                      deallocate(temp,idx)

                      alpha_locs = 1
                      do idet = 2, n3abb
                         a1 = t3c_excits(1,idet-1); i1 = t3c_excits(4,idet-1);
                         a2 = t3c_excits(1,idet);   i2 = t3c_excits(4,idet);
                         ia1 = Eai_table(a1,i1)
                         ia2 = Eai_table(a2,i2)
                         if (ia1/=ia2) then
                                 alpha_locs(ia1,2) = idet - 1
                                 alpha_locs(ia2,1) = idet
                         end if
                      end do
                      alpha_locs(ia2,2) = n3abb

                      ID = 0
                      do a = 1,nua
                         do i = 1,noa
                            ia = Eai_table(a,i)
                            if (alpha_locs(ia,1) > alpha_locs(ia,2)) cycle ! skip if alpha block is empty

                            num_jk_ia = alpha_locs(ia,2) - alpha_locs(ia,1) + 1

                            allocate(temp(num_jk_ia), idx(num_jk_ia))
                            kout = 1
                            do idet = alpha_locs(ia,1), alpha_locs(ia,2)
                               j = t3c_excits(5,idet); k = t3c_excits(6,idet);
                               jk = XjXk_table(j,k)
                               temp(kout) = jk
                               kout = kout + 1
                            end do
                            call argsort(temp,idx)
                            idx = idx + alpha_locs(ia,1) - 1
                            t3c_excits(:,alpha_locs(ia,1):alpha_locs(ia,2)) = t3c_excits(:,idx)
                            t3c_amps(alpha_locs(ia,1):alpha_locs(ia,2)) = t3c_amps(idx)
                            deallocate(temp,idx)

                            ID(ia,:,1) = alpha_locs(ia,1)
                            do jk = 1, num_jk_ia-1
                               idet = jk + alpha_locs(ia,1)
                               j1 = t3c_excits(5,idet-1); k1 = t3c_excits(6,idet-1);
                               j2 = t3c_excits(5,idet);   k2 = t3c_excits(6,idet);
                               jk1 = XjXk_table(j1,k1)
                               jk2 = XjXk_table(j2,k2)
                               
                               if (jk1/=jk2) then
                                       ID(ia,jk1,2) = idet-1
                                       ID(ia,jk2,1) = idet
                               end if
                            end do
                            ID(ia,jk2,2) = alpha_locs(ia,1) + num_jk_ia - 1
                         end do
                      end do

              end subroutine sort_t3c_h

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
            
                      if ( n==1 ) return
                    
                      stepsize = 1
                      do while (stepsize<n)
                          do left=1,n-stepsize,stepsize*2
                              i = left
                              j = left+stepsize
                              ksize = min(stepsize*2,n-left+1)
                              k=1
                        
                              do while ( i<left+stepsize .and. j<left+ksize )
                                  if ( r(d(i))<r(d(j)) ) then
                                      il(k)=d(i)
                                      i=i+1
                                      k=k+1
                                  else
                                      il(k)=d(j)
                                      j=j+1
                                      k=k+1
                                  endif
                              enddo
                        
                              if ( i<left+stepsize ) then
                                  ! fill up remaining from left
                                  il(k:ksize) = d(i:left+stepsize-1)
                              else
                                  ! fill up remaining from right
                                  il(k:ksize) = d(j:left+ksize-1)
                              endif
                              d(left:left+ksize-1) = il(1:ksize)
                          end do
                          stepsize=stepsize*2
                      end do

          end subroutine argsort

end module sort_t3

