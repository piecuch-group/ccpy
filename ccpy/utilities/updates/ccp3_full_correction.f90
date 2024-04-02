module ccp3_full_correction

      implicit none

      contains

              subroutine ccp3a(deltaA,deltaB,deltaC,deltaD,&
                              t3a_excits,&
                              t2a,l1a,l2a,&
                              H2A_vooo,I2A_vvov,vA_oovv,H1A_ov,H2A_vovv,H2A_ooov,fA_oo,fA_vv,&
                              H1A_oo,H1A_vv,&
                              H2A_voov,H2A_oooo,H2A_vvvv,&
                              D3A_O,D3A_v,&
                              n3aaa,noa,nua)

                        real(kind=8), intent(out) :: deltaA, deltaB, deltaC, deltaD
                        integer, intent(in) :: noa, nua, n3aaa
                        integer, intent(in) :: t3a_excits(6,n3aaa)
                        real(kind=8), intent(in) :: fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),&
                        H1A_oo(1:noa,1:noa),H1A_vv(1:nua,1:nua),&
                        H2A_voov(1:nua,1:noa,1:noa,1:nua),&
                        H2A_oooo(1:noa,1:noa,1:noa,1:noa),&
                        H2A_vvvv(1:nua,1:nua,1:nua,1:nua),&
                        D3A_O(1:nua,1:noa,1:noa),&
                        D3A_V(1:nua,1:noa,1:nua),&
                        H2A_vooo(nua,noa,noa,noa),I2A_vvov(nua,nua,noa,nua),t2a(nua,nua,noa,noa),&
                        l1a(nua,noa),l2a(nua,nua,noa,noa),vA_oovv(noa,noa,nua,nua),&
                        H1A_ov(noa,nua),H2A_vovv(nua,noa,nua,nua),H2A_ooov(noa,noa,noa,nua)
                        integer :: i, j, k, a, b, c, nua2
                        real(kind=8) :: D, temp1, temp2, temp3, LM, X3A(nua,nua,nua), L3A(nua,nua,nua)
                        ! Low-memory looping variables
                        logical(kind=1) :: qspace(nua,nua,nua)
                        integer :: nloc, idet, idx
                        integer, allocatable :: loc_arr(:,:), idx_table(:,:,:)
                        integer :: excits_buff(6,n3aaa), excits_buff2(6,n3aaa)
                        ! reordered arrays for DGEMMs
                        real(kind=8) :: I2A_vvov_1243(nua,nua,nua,noa), H2A_vovv_4312(nua,nua,nua,noa), H2A_ooov_4312(nua,noa,noa,noa)
                        
                        ! reorder t3a into (i,j,k) order
                        excits_buff(:,:) = t3a_excits(:,:)
                        nloc = noa*(noa-1)*(noa-2)/6
                        allocate(loc_arr(2,nloc))
                        allocate(idx_table(noa,noa,noa))
                        call get_index_table(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), noa, noa, noa)
                        call sort3(excits_buff, loc_arr, idx_table, (/4,5,6/), noa, noa, noa, nloc, n3aaa)
                        ! reorder H for contractions
                        call reorder1243(I2A_vvov,I2A_vvov_1243)
                        call reorder4312(H2A_vovv,H2A_vovv_4312)
                        call reorder4312(H2A_ooov,H2A_ooov_4312)

                        deltaA = 0.0d0
                        deltaB = 0.0d0
                        deltaC = 0.0d0
                        deltaD = 0.0d0
                        
                        nua2 = nua*nua
                        do i = 1,noa
                           do j = i+1,noa
                              do k = j+1,noa
                                 ! Construct Q space for block (i,j,k)
                                 qspace = .true.
                                 idx = idx_table(i,j,k)
                                 if (idx/=0) then
                                    do idet = loc_arr(1,idx), loc_arr(2,idx)
                                       a = excits_buff(1,idet); b = excits_buff(2,idet); c = excits_buff(3,idet);
                                       qspace(a,b,c) = .false.
                                    end do
                                 end if
                                 !!!!! L3A !!!!!
                                 L3A = 0.0d0
                                 ! Diagram 1: A(i/jk)A(c/ab) H2A_vovv(e,i,b,a)*l2a(e,c,j,k)
                                 call dgemm('n','n',nua2,nua,nua,0.5d0,H2A_vovv_4312(:,:,:,i),nua2,l2a(:,:,j,k),nua,1.0d0,L3A,nua2)
                                 call dgemm('n','n',nua2,nua,nua,-0.5d0,H2A_vovv_4312(:,:,:,j),nua2,l2a(:,:,i,k),nua,1.0d0,L3A,nua2)
                                 call dgemm('n','n',nua2,nua,nua,-0.5d0,H2A_vovv_4312(:,:,:,k),nua2,l2a(:,:,j,i),nua,1.0d0,L3A,nua2)
                                 ! Diagram 2: -A(k/ij)A(a/bc) H2A_ooov(j,i,m,a)*l2a(b,c,m,k)-> a,m,j,i * (b,c,m,k)'
                                 call dgemm('n','t',nua,nua2,noa,-0.5d0,H2A_ooov_4312(:,:,j,i),nua,l2a(:,:,:,k),nua2,1.0d0,L3A,nua)
                                 call dgemm('n','t',nua,nua2,noa,0.5d0,H2A_ooov_4312(:,:,k,i),nua,l2a(:,:,:,j),nua2,1.0d0,L3A,nua)
                                 call dgemm('n','t',nua,nua2,noa,0.5d0,H2A_ooov_4312(:,:,j,k),nua,l2a(:,:,:,i),nua2,1.0d0,L3A,nua)
                                 do a = 1,nua
                                    do b = a+1,nua
                                       do c = b+1,nua
                                          if (.not. qspace(a,b,c)) cycle

                                          l0 = L3A(a,b,c) + L3A(b,c,a) + L3A(c,a,b)&
                                             - L3A(a,c,b) - L3A(b,a,c) - L3A(c,b,a)

                                          l0 = l0 +&
                                                l1a(c,k)*vA_oovv(i,j,a,b)&
                                               -l1a(a,k)*vA_oovv(i,j,c,b)&
                                               -l1a(b,k)*vA_oovv(i,j,a,c)&
                                               -l1a(c,i)*vA_oovv(k,j,a,b)&
                                               -l1a(c,j)*vA_oovv(i,k,a,b)&
                                               +l1a(a,i)*vA_oovv(k,j,c,b)&
                                               +l1a(b,i)*vA_oovv(k,j,a,c)&
                                               +l1a(a,j)*vA_oovv(i,k,c,b)&
                                               +l1a(b,j)*vA_oovv(i,k,a,c)&
                                               +H1A_ov(k,c)*l2a(a,b,i,j)&
                                               -H1A_ov(k,a)*l2a(c,b,i,j)&
                                               -H1A_ov(k,b)*l2a(a,c,i,j)&
                                               -H1A_ov(i,c)*l2a(a,b,k,j)&
                                               -H1A_ov(j,c)*l2a(a,b,i,k)&
                                               +H1A_ov(i,a)*l2a(c,b,k,j)&
                                               +H1A_ov(i,b)*l2a(a,c,k,j)&
                                               +H1A_ov(j,a)*l2a(c,b,i,k)&
                                               +H1A_ov(j,b)*l2a(a,c,i,k)


                                          D = fA_oo(i,i) + fA_oo(j,j) + fA_oo(k,k)&
                                             -fA_vv(a,a) - fA_vv(b,b) - fA_vv(c,c)
                                          l_A = l0/D

                                          D = H1A_oo(i,i) + H1A_oo(j,j) + H1A_oo(k,k)&
                                             -H1A_vv(a,a) - H1A_vv(b,b) - H1A_vv(c,c)
                                          l_B = l0/D

                                          D = D &
                                             -H2A_voov(a,i,i,a) - H2A_voov(b,i,i,b) - H2A_voov(c,i,i,c)&
                                             -H2A_voov(a,j,j,a) - H2A_voov(b,j,j,b) - H2A_voov(c,j,j,c)&
                                             -H2A_voov(a,k,k,a) - H2A_voov(b,k,k,b) - H2A_voov(c,k,k,c)&
                                             -H2A_oooo(j,i,j,i) - H2A_oooo(k,i,k,i) - H2A_oooo(k,j,k,j)&
                                             -H2A_vvvv(b,a,b,a) - H2A_vvvv(c,a,c,a) - H2A_vvvv(c,b,c,b)
                                          l_C = l0/D

                                          D = D &
                                             +D3A_O(a,i,j)+D3A_O(a,i,k)+D3A_O(a,j,k)&
                                             +D3A_O(b,i,j)+D3A_O(b,i,k)+D3A_O(b,j,k)&
                                             +D3A_O(c,i,j)+D3A_O(c,i,k)+D3A_O(c,j,k)&
                                             -D3A_V(a,i,b)-D3A_V(a,i,c)-D3A_V(b,i,c)&
                                             -D3A_V(a,j,b)-D3A_V(a,j,c)-D3A_V(b,j,c)&
                                             -D3A_V(a,k,b)-D3A_V(a,k,c)-D3A_V(b,k,c)
                                          l_D = l0/D

                                       end do
                                    end do
                                 end do
                              end do
                           end do
                        end do
                        deallocate(loc_arr,idx_table)
                 
              end subroutine ccp3a

         
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

              subroutine get_index_table4(idx_table, rng1, rng2, rng3, rng4, n1, n2, n3, n4)

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

              end subroutine get_index_table4

              subroutine sort4(excits, amps, loc_arr, idx_table, idims, n1, n2, n3, n4, nloc, n3p, resid)
                  
                      integer, intent(in) :: n1, n2, n3, n4, nloc, n3p
                      integer, intent(in) :: idims(4)
                      integer, intent(in) :: idx_table(n1,n2,n3,n4)

                      integer, intent(inout) :: loc_arr(2,nloc)
                      integer, intent(inout) :: excits(n3p,6)
                      real(kind=8), intent(inout) :: amps(n3p)
                      real(kind=8), intent(inout), optional :: resid(n3p)

                      integer :: idet
                      integer :: p, q, r, s
                      integer :: p1, q1, r1, s1, p2, q2, r2, s2
                      integer :: pqrs1, pqrs2
                      integer, allocatable :: temp(:), idx(:)

                      ! obtain the lexcial index for each triple excitation in the P space along the sorting dimensions idims
                      allocate(temp(n3p),idx(n3p))
                      do idet = 1, n3p
                         p = excits(idet,idims(1)); q = excits(idet,idims(2)); r = excits(idet,idims(3)); s = excits(idet,idims(4))
                         temp(idet) = idx_table(p,q,r,s)
                      end do
                      ! get the sorting array
                      call argsort(temp, idx)
                      ! apply sorting array to t3 excitations, amplitudes, and, optionally, residual arrays
                      excits = excits(idx,:)
                      amps = amps(idx)
                      if (present(resid)) resid = resid(idx)
                      deallocate(temp,idx)
                      ! obtain the start- and end-point indices for each lexical index in the sorted t3 excitation and amplitude arrays
                      loc_arr(1,:) = 1; loc_arr(2,:) = 0; ! set default start > end so that empty sets do not trigger loops
                      !!! WARNING: THERE IS A MEMORY LEAK HERE! pqrs2 is used below but is not set if n3p <= 1
                      !if (n3p <= 1) print*, "WARNING: potential memory leakage in sort4 function. pqrs2 set to 0"
                      pqrs2 = 0
                      do idet = 1, n3p-1
                         ! get consecutive lexcial indices
                         p1 = excits(idet,idims(1));   q1 = excits(idet,idims(2));   r1 = excits(idet,idims(3));   s1 = excits(idet,idims(4))
                         p2 = excits(idet+1,idims(1)); q2 = excits(idet+1,idims(2)); r2 = excits(idet+1,idims(3)); s2 = excits(idet+1,idims(4))
                         pqrs1 = idx_table(p1,q1,r1,s1)
                         pqrs2 = idx_table(p2,q2,r2,s2)
                         ! if change occurs between consecutive indices, record these locations in loc_arr as new start/end points
                         if (pqrs1 /= pqrs2) then
                            loc_arr(2,pqrs1) = idet
                            loc_arr(1,pqrs2) = idet+1
                         end if
                      end do
                      if (n3p > 1) then
                         loc_arr(2,pqrs2) = n3p
                      end if

              end subroutine sort4


              subroutine sort3(excits, loc_arr, idx_table, idims, n1, n2, n3, nloc, n3p)

                    integer, intent(in) :: n1, n2, n3, nloc, n3p
                    integer, intent(in) :: idims(3)
                    integer, intent(in) :: idx_table(n1,n2,n3)
      
                    integer, intent(inout) :: loc_arr(2,nloc)
                    integer, intent(inout) :: excits(6,n3p)
      
                    integer :: idet
                    integer :: p, q, r
                    integer :: p1, q1, r1, p2, q2, r2
                    integer :: pqr1, pqr2
                    integer, allocatable :: temp(:), idx(:)
      
                    allocate(temp(n3p),idx(n3p))
                    do idet = 1, n3p
                       p = excits(idims(1),idet); q = excits(idims(2),idet); r = excits(idims(3),idet);
                       temp(idet) = idx_table(p,q,r)
                    end do
                    call argsort(temp, idx)
                    excits = excits(:,idx)
                    deallocate(temp,idx)
      
                    loc_arr(1,:) = 1; loc_arr(2,:) = 0;
                    !!! WARNING: THERE IS A MEMORY LEAK HERE! pqr2 is used below but is not set if n3p <= 1
                    !if (n3p <= 1) print*, "WARNING: potential memory leakage in sort3 function. pqr2 set to 0"
                    pqr2 = 0
                    do idet = 1, n3p-1
                       p1 = excits(idims(1),idet);   q1 = excits(idims(2),idet);   r1 = excits(idims(3),idet);
                       p2 = excits(idims(1),idet+1); q2 = excits(idims(2),idet+1); r2 = excits(idims(3),idet+1);
                       pqr1 = idx_table(p1,q1,r1)
                       pqr2 = idx_table(p2,q2,r2)
                       if (pqr1 /= pqr2) then
                          loc_arr(2,pqr1) = idet
                          loc_arr(1,pqr2) = idet+1
                       end if
                    end do
                    if (n3p > 1) then
                       loc_arr(2,pqr2) = n3p
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
         
              !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! REORDER ROUTINES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
              subroutine reorder3412(x_in,x_out)

                      real(kind=8), intent(in) :: x_in(:,:,:,:)
                      real(kind=8), intent(out) :: x_out(:,:,:,:)

                      integer :: i1, i2, i3, i4

                      do i1 = 1,size(x_in,1)
                         do i2 = 1,size(x_in,2)
                            do i3 = 1,size(x_in,3)
                               do i4= 1,size(x_in,4)
                                  x_out(i3,i4,i1,i2) = x_in(i1,i2,i3,i4)
                               end do
                            end do
                         end do
                      end do

             end subroutine reorder3412

             subroutine reorder1342(x_in,x_out)

                      real(kind=8), intent(in) :: x_in(:,:,:,:)
                      real(kind=8), intent(out) :: x_out(:,:,:,:)

                      integer :: i1, i2, i3, i4

                      do i1 = 1,size(x_in,1)
                         do i2 = 1,size(x_in,2)
                            do i3 = 1,size(x_in,3)
                               do i4= 1,size(x_in,4)
                                  x_out(i1,i3,i4,i2) = x_in(i1,i2,i3,i4)
                               end do
                            end do
                         end do
                      end do

             end subroutine reorder1342

            subroutine reorder3421(x_in,x_out)

                      real(kind=8), intent(in) :: x_in(:,:,:,:)
                      real(kind=8), intent(out) :: x_out(:,:,:,:)

                      integer :: i1, i2, i3, i4

                      do i1 = 1,size(x_in,1)
                         do i2 = 1,size(x_in,2)
                            do i3 = 1,size(x_in,3)
                               do i4= 1,size(x_in,4)
                                  x_out(i3,i4,i2,i1) = x_in(i1,i2,i3,i4)
                               end do
                            end do
                         end do
                      end do

             end subroutine reorder3421

             subroutine reorder2134(x_in,x_out)

                      real(kind=8), intent(in) :: x_in(:,:,:,:)
                      real(kind=8), intent(out) :: x_out(:,:,:,:)

                      integer :: i1, i2, i3, i4

                      do i1 = 1,size(x_in,1)
                         do i2 = 1,size(x_in,2)
                            do i3 = 1,size(x_in,3)
                               do i4= 1,size(x_in,4)
                                  x_out(i2,i1,i3,i4) = x_in(i1,i2,i3,i4)
                               end do
                            end do
                         end do
                      end do

             end subroutine reorder2134

            subroutine reorder1243(x_in,x_out)

                      real(kind=8), intent(in) :: x_in(:,:,:,:)
                      real(kind=8), intent(out) :: x_out(:,:,:,:)

                      integer :: i1, i2, i3, i4

                      do i1 = 1,size(x_in,1)
                         do i2 = 1,size(x_in,2)
                            do i3 = 1,size(x_in,3)
                               do i4= 1,size(x_in,4)
                                  x_out(i1,i2,i4,i3) = x_in(i1,i2,i3,i4)
                               end do
                            end do
                         end do
                      end do

             end subroutine reorder1243

             subroutine reorder4213(x_in,x_out)

                      real(kind=8), intent(in) :: x_in(:,:,:,:)
                      real(kind=8), intent(out) :: x_out(:,:,:,:)

                      integer :: i1, i2, i3, i4

                      do i1 = 1,size(x_in,1)
                         do i2 = 1,size(x_in,2)
                            do i3 = 1,size(x_in,3)
                               do i4= 1,size(x_in,4)
                                  x_out(i4,i2,i1,i3) = x_in(i1,i2,i3,i4)
                               end do
                            end do
                         end do
                      end do

             end subroutine reorder4213

             subroutine reorder4312(x_in,x_out)

                      real(kind=8), intent(in) :: x_in(:,:,:,:)
                      real(kind=8), intent(out) :: x_out(:,:,:,:)

                      integer :: i1, i2, i3, i4

                      do i1 = 1,size(x_in,1)
                         do i2 = 1,size(x_in,2)
                            do i3 = 1,size(x_in,3)
                               do i4= 1,size(x_in,4)
                                  x_out(i4,i3,i1,i2) = x_in(i1,i2,i3,i4)
                               end do
                            end do
                         end do
                      end do

             end subroutine reorder4312

             subroutine reorder2341(x_in,x_out)

                      real(kind=8), intent(in) :: x_in(:,:,:,:)
                      real(kind=8), intent(out) :: x_out(:,:,:,:)

                      integer :: i1, i2, i3, i4

                      do i1 = 1,size(x_in,1)
                         do i2 = 1,size(x_in,2)
                            do i3 = 1,size(x_in,3)
                               do i4= 1,size(x_in,4)
                                  x_out(i2,i3,i4,i1) = x_in(i1,i2,i3,i4)
                               end do
                            end do
                         end do
                      end do

             end subroutine reorder2341

             subroutine reorder2143(x_in,x_out)

                      real(kind=8), intent(in) :: x_in(:,:,:,:)
                      real(kind=8), intent(out) :: x_out(:,:,:,:)

                      integer :: i1, i2, i3, i4

                      do i1 = 1,size(x_in,1)
                         do i2 = 1,size(x_in,2)
                            do i3 = 1,size(x_in,3)
                               do i4= 1,size(x_in,4)
                                  x_out(i2,i1,i4,i3) = x_in(i1,i2,i3,i4)
                               end do
                            end do
                         end do
                      end do

             end subroutine reorder2143

             subroutine reorder4123(x_in,x_out)

                      real(kind=8), intent(in) :: x_in(:,:,:,:)
                      real(kind=8), intent(out) :: x_out(:,:,:,:)

                      integer :: i1, i2, i3, i4

                      do i1 = 1,size(x_in,1)
                         do i2 = 1,size(x_in,2)
                            do i3 = 1,size(x_in,3)
                               do i4= 1,size(x_in,4)
                                  x_out(i4,i1,i2,i3) = x_in(i1,i2,i3,i4)
                               end do
                            end do
                         end do
                      end do

             end subroutine reorder4123

             subroutine reorder3214(x_in,x_out)

                      real(kind=8), intent(in) :: x_in(:,:,:,:)
                      real(kind=8), intent(out) :: x_out(:,:,:,:)

                      integer :: i1, i2, i3, i4

                      do i1 = 1,size(x_in,1)
                         do i2 = 1,size(x_in,2)
                            do i3 = 1,size(x_in,3)
                               do i4= 1,size(x_in,4)
                                  x_out(i3,i2,i1,i4) = x_in(i1,i2,i3,i4)
                               end do
                            end do
                         end do
                      end do

             end subroutine reorder3214
   
end module ccp3_full_correction
