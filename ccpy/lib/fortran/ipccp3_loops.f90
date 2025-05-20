module ipccp3_loops

      implicit none

      contains

              subroutine ipccp3A_full(deltaA,deltaB,deltaC,deltaD,&
                                      M3A,L3A,r3a_excits,omega,&
                                      fA_oo,fA_vv,H1A_oo,H1A_vv,&
                                      H2A_vvvv,H2A_oooo,H2A_voov,&
                                      d3A_o,d3A_v,&
                                      n3aaa,noa,nua)

                        ! input variables
                        integer, intent(in) :: noa, nua, n3aaa
                        integer, intent(in) :: r3a_excits(n3aaa,5)
                        real(kind=8), intent(in) :: M3A(noa,nua,nua,noa,noa)
                        real(kind=8), intent(in) :: L3A(noa,nua,nua,noa,noa)
                        real(kind=8), intent(in) :: omega
                        real(kind=8), intent(in) :: fA_oo(noa,noa), fA_vv(nua,nua)
                        real(kind=8), intent(in) :: h1A_oo(noa,noa), h1A_vv(nua,nua)
                        real(kind=8), intent(in) :: h2a_oooo(noa,noa,noa,noa)
                        real(kind=8), intent(in) :: h2a_vvvv(nua,nua,nua,nua)
                        real(kind=8), intent(in) :: h2a_voov(nua,noa,noa,nua)
                        real(kind=8), intent(in) :: d3A_o(nua,noa,noa)
                        real(kind=8), intent(in) :: d3A_v(nua,noa,nua)
                        ! output variables
                        real(kind=8), intent(out) :: deltaA, deltaB, deltaC, deltaD
                        ! Low-memory looping variables
                        logical(kind=1) :: qspace(nua,nua)
                        integer :: nloc, idet, idx
                        integer, allocatable :: loc_arr(:,:), idx_table(:,:,:)
                        integer :: excits_buff(n3aaa,5)
                        ! Local variables
                        integer :: i, j, k, b, c
                        real(kind=8) :: D, temp

                        deltaA = 0.0d0
                        deltaB = 0.0d0
                        deltaC = 0.0d0
                        deltaD = 0.0d0

                        ! reorder r3a into (i,j,k) order
                        excits_buff(:,:) = r3a_excits(:,:)
                        nloc = noa*(noa-1)*(noa-2)/6
                        allocate(loc_arr(2,nloc))
                        allocate(idx_table(noa,noa,noa))
                        call get_index_table3(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), noa, noa, noa)
                        call sort3(excits_buff, loc_arr, idx_table, (/3,4,5/), noa, noa, noa, nloc, n3aaa)

                        do i = 1,noa
                           do j = i+1,noa
                              do k = j+1,noa
                                 ! Construct Q space for block (i,j,k)
                                 qspace = .true.
                                 idx = idx_table(i,j,k)
                                 if (idx/=0) then
                                    do idet = loc_arr(1,idx), loc_arr(2,idx)
                                       b = excits_buff(idet,1); c = excits_buff(idet,2);
                                       qspace(b,c) = .false.
                                    end do
                                 end if
                                 do b = 1,nua
                                    do c = b+1,nua
                                       if (.not. qspace(b,c)) cycle
                                       temp = M3A(i,b,c,j,k)*L3A(i,b,c,j,k)
                                       ! A correction
                                       D = -fA_vv(b,b) - fA_vv(c,c) + fA_oo(i,i) + fA_oo(j,j) + fA_oo(k,k)
                                       deltaA = deltaA + temp/(omega+D)
                                       ! B correction
                                       D = -h1a_vv(b,b) - h1a_vv(c,c) + h1a_oo(i,i) + h1a_oo(j,j) + h1a_oo(k,k)
                                       deltaB = deltaB + temp/(omega+D)
                                       ! C correction
                                       D = D&
                                       -h2a_oooo(i,j,i,j) - h2a_oooo(i,k,i,k) - h2a_oooo(j,k,j,k)&
                                       -h2a_vvvv(b,c,b,c)&
                                       -h2a_voov(b,i,i,b) - h2a_voov(c,i,i,c)&
                                       -h2a_voov(b,j,j,b) - h2a_voov(c,j,j,c)&
                                       -h2a_voov(b,k,k,b) - h2a_voov(c,k,k,c)
                                       deltaC = deltaC + temp/(omega+D)
                                       ! D correction
                                       D = D&
                                       + D3A_O(b,i,j)+D3A_O(b,i,k)+D3A_O(b,j,k)&
                                       +D3A_O(c,i,j)+D3A_O(c,i,k)+D3A_O(c,j,k)&
                                       -D3A_V(b,i,c)&
                                       -D3A_V(b,j,c)&
                                       -D3A_V(b,k,c)
                                       deltaD = deltaD + temp/(omega+D)
                                    end do
                                 end do
                              end do
                           end do
                        end do
                        deallocate(idx_table,loc_arr)

              end subroutine ipccp3A_full

              subroutine ipccp3B_full(deltaA,deltaB,deltaC,deltaD,&
                                      M3B,L3B,r3b_excits,omega,&
                                      fA_oo,fA_vv,fB_oo,fB_vv,&
                                      H1A_oo,H1A_vv,H1B_oo,H1B_vv,&
                                      H2A_oooo,H2A_voov,&
                                      H2B_vvvv,H2B_oooo,H2B_ovov,H2B_vovo,&
                                      H2C_voov,&
                                      d3A_o,d3A_v,d3B_o,d3B_v,d3C_o,d3C_v,&
                                      n3aab,noa,nua,nob,nub)

                        ! input variables
                        integer, intent(in) :: noa, nua, nob, nub, n3aab
                        integer, intent(in) :: r3b_excits(n3aab,5)
                        real(kind=8), intent(in) :: M3B(noa,nua,nub,noa,nob)
                        real(kind=8), intent(in) :: L3B(noa,nua,nub,noa,nob)
                        real(kind=8), intent(in) :: omega
                        real(kind=8), intent(in) :: fA_oo(noa,noa), fA_vv(nua,nua)
                        real(kind=8), intent(in) :: fB_oo(nob,nob), fB_vv(nub,nub)
                        real(kind=8), intent(in) :: h1A_oo(noa,noa), h1A_vv(nua,nua)
                        real(kind=8), intent(in) :: h1B_oo(nob,nob), h1B_vv(nub,nub)
                        real(kind=8), intent(in) :: h2a_oooo(noa,noa,noa,noa)
                        real(kind=8), intent(in) :: h2a_voov(nua,noa,noa,nua)
                        real(kind=8), intent(in) :: h2b_vvvv(nua,nub,nua,nub)
                        real(kind=8), intent(in) :: h2b_oooo(noa,nob,noa,nob)
                        real(kind=8), intent(in) :: h2b_ovov(noa,nub,noa,nub)
                        real(kind=8), intent(in) :: h2b_vovo(nua,nob,nua,nob)
                        real(kind=8), intent(in) :: h2c_voov(nub,nob,nob,nub)
                        real(kind=8), intent(in) :: d3A_o(nua,noa,noa)
                        real(kind=8), intent(in) :: d3A_v(nua,noa,nua)
                        real(kind=8), intent(in) :: d3B_o(nua,noa,nob)
                        real(kind=8), intent(in) :: d3B_v(nua,noa,nub)
                        real(kind=8), intent(in) :: d3C_o(nub,noa,nob)
                        real(kind=8), intent(in) :: d3C_v(nua,nob,nub)
                        ! output variables
                        real(kind=8), intent(out) :: deltaA, deltaB, deltaC, deltaD
                        ! Low-memory looping variables
                        logical(kind=1) :: qspace(nua,nub)
                        integer :: nloc, idet, idx
                        integer, allocatable :: loc_arr(:,:), idx_table(:,:,:)
                        integer :: excits_buff(n3aab,5)
                        ! Local variables
                        integer :: i, j, k, b, c
                        real(kind=8) :: D, temp

                        deltaA = 0.0d0
                        deltaB = 0.0d0
                        deltaC = 0.0d0
                        deltaD = 0.0d0

                        ! reorder r3b into (i,j,k) order
                        excits_buff(:,:) = r3b_excits(:,:)
                        nloc = noa*(noa-1)/2*nob
                        allocate(loc_arr(2,nloc))
                        allocate(idx_table(noa,noa,nob))
                        call get_index_table3(idx_table, (/1,noa-1/), (/-1,noa/), (/1,nob/), noa, noa, nob)
                        call sort3(excits_buff, loc_arr, idx_table, (/3,4,5/), noa, noa, nob, nloc, n3aab)

                        do i = 1,noa
                           do j = i+1,noa
                              do k = 1,nob
                                 qspace = .true.
                                 idx = idx_table(i,j,k)
                                 if (idx/=0) then
                                    do idet = loc_arr(1,idx), loc_arr(2,idx)
                                       b = excits_buff(idet,1); c = excits_buff(idet,2);
                                       qspace(b,c) = .false.
                                    end do
                                 end if
                                 do b = 1,nua
                                    do c = 1,nub
                                       if (.not. qspace(b,c)) cycle
                                       temp = M3B(i,b,c,j,k)*L3B(i,b,c,j,k)
                                       ! A correction
                                       D = -fA_vv(b,b) - fB_vv(c,c) + fA_oo(i,i) + fA_oo(j,j) + fB_oo(k,k)
                                       deltaA = deltaA + temp/(omega+D)
                                       ! B correction
                                       D = -h1a_vv(b,b) - h1b_vv(c,c) + h1a_oo(i,i) + h1a_oo(j,j) + h1b_oo(k,k)
                                       deltaB = deltaB + temp/(omega+D)
                                       ! C correction
                                       D = D&
                                       -h2a_oooo(i,j,i,j) - h2b_oooo(i,k,i,k) - h2b_oooo(j,k,j,k)&
                                       -h2b_vvvv(b,c,b,c)&
                                       -h2a_voov(b,i,i,b) + h2b_ovov(i,c,i,c)&
                                       -h2a_voov(b,j,j,b) + h2b_ovov(j,c,j,c)&
                                       +h2b_vovo(b,k,b,k) - h2c_voov(c,k,k,c)
                                       deltaC = deltaC + temp/(omega+D)
                                       ! D correction
                                       D = D &
                                       + D3A_O(b,i,j)+D3B_O(b,i,k)+D3B_O(b,j,k)&
                                       +D3C_O(c,i,k)+D3C_O(c,j,k)&
                                       -D3B_V(b,i,c)&
                                       -D3B_V(b,j,c)&
                                       -D3C_V(b,k,c)
                                       deltaD = deltaD + temp/(omega+D)
                                    end do
                                 end do
                              end do
                           end do
                        end do
                        deallocate(idx_table,loc_arr)

              end subroutine ipccp3B_full

              subroutine ipccp3C_full(deltaA,deltaB,deltaC,deltaD,&
                                 M3C,L3C,r3c_excits,omega,&
                                 fA_oo,fA_vv,fB_oo,fB_vv,&
                                 H1A_oo,H1A_vv,H1B_oo,H1B_vv,&
                                 H2B_oooo,H2B_ovov,H2B_vovo,&
                                 H2C_vvvv,H2C_oooo,H2C_voov,&
                                 d3B_o,d3B_v,d3C_o,d3C_v,d3D_o,d3D_v,&
                                 n3abb,noa,nua,nob,nub)

                        ! input variables
                        integer, intent(in) :: noa, nua, nob, nub, n3abb
                        integer, intent(in) :: r3c_excits(n3abb,5)
                        real(kind=8), intent(in) :: M3C(noa,nub,nub,nob,nob)
                        real(kind=8), intent(in) :: L3C(noa,nub,nub,nob,nob)
                        real(kind=8), intent(in) :: omega
                        real(kind=8), intent(in) :: fA_oo(noa,noa), fA_vv(nua,nua)
                        real(kind=8), intent(in) :: fB_oo(nob,nob), fB_vv(nub,nub)
                        real(kind=8), intent(in) :: h1A_oo(noa,noa), h1A_vv(nua,nua)
                        real(kind=8), intent(in) :: h1B_oo(nob,nob), h1B_vv(nub,nub)
                        real(kind=8), intent(in) :: h2b_oooo(noa,nob,noa,nob)
                        real(kind=8), intent(in) :: h2b_ovov(noa,nub,noa,nub)
                        real(kind=8), intent(in) :: h2b_vovo(nua,nob,nua,nob)
                        real(kind=8), intent(in) :: h2c_vvvv(nub,nub,nub,nub)
                        real(kind=8), intent(in) :: h2c_oooo(nob,nob,nob,nob)
                        real(kind=8), intent(in) :: h2c_voov(nub,nob,nob,nub)
                        real(kind=8), intent(in) :: d3B_o(nua,noa,nob)
                        real(kind=8), intent(in) :: d3B_v(nua,noa,nub)
                        real(kind=8), intent(in) :: d3C_o(nub,noa,nob)
                        real(kind=8), intent(in) :: d3C_v(nua,nob,nub)
                        real(kind=8), intent(in) :: d3D_o(nub,nob,nob)
                        real(kind=8), intent(in) :: d3D_v(nub,nob,nub)
                        ! output variables
                        real(kind=8), intent(out) :: deltaA, deltaB, deltaC, deltaD
                        ! Low-memory looping variables
                        logical(kind=1) :: qspace(nub,nub)
                        integer :: nloc, idet, idx
                        integer, allocatable :: loc_arr(:,:), idx_table(:,:,:)
                        integer :: excits_buff(n3abb,5)
                        ! Local variables
                        integer :: i, j, k, b, c
                        real(kind=8) :: D, temp

                        deltaA = 0.0d0
                        deltaB = 0.0d0
                        deltaC = 0.0d0
                        deltaD = 0.0d0

                        ! reorder r3c into (j,k,i) order
                        excits_buff(:,:) = r3c_excits(:,:)
                        nloc = nob*(nob-1)/2*noa
                        allocate(loc_arr(2,nloc))
                        allocate(idx_table(nob,nob,noa))
                        call get_index_table3(idx_table, (/1,nob-1/), (/-1,nob/), (/1,noa/), nob, nob, noa)
                        call sort3(excits_buff, loc_arr, idx_table, (/4,5,3/), nob, nob, noa, nloc, n3abb)

                        do i = 1,noa
                           do j = 1,nob
                              do k = j+1,nob
                                 ! Construct Q space for block (i,j,k)
                                 qspace = .true.
                                 idx = idx_table(j,k,i)
                                 if (idx/=0) then
                                    do idet = loc_arr(1,idx), loc_arr(2,idx)
                                       b = excits_buff(idet,1); c = excits_buff(idet,2);
                                       qspace(b,c) = .false.
                                    end do
                                 end if
                                 do b = 1,nub
                                    do c = b+1,nub
                                       if (.not. qspace(b,c)) cycle
                                       temp = M3C(i,b,c,j,k)*L3C(i,b,c,j,k)
                                       ! A correction
                                       D = -fB_vv(b,b) - fB_vv(c,c) + fA_oo(i,i) + fB_oo(j,j) + fB_oo(k,k)
                                       deltaA = deltaA + temp/(omega+D)
                                       ! B correction
                                       D = -h1b_vv(b,b) - h1b_vv(c,c) + h1a_oo(i,i) + h1b_oo(j,j) + h1b_oo(k,k)
                                       deltaB = deltaB + temp/(omega+D)
                                       ! C correction
                                       D = D&
                                       -h2b_oooo(i,j,i,j) - h2b_oooo(i,k,i,k) - h2c_oooo(j,k,j,k)&
                                       -h2c_vvvv(b,c,b,c)&
                                       +h2b_ovov(i,b,i,b) + h2b_ovov(i,c,i,c)&
                                       -h2c_voov(b,j,j,b) - h2c_voov(c,j,j,c)&
                                       -h2c_voov(b,k,k,b) - h2c_voov(c,k,k,c)
                                       deltaC = deltaC + temp/(omega+D)
                                       ! D correction
                                       D = D&
                                       + D3C_O(b,i,j)+D3C_O(b,i,k)+D3D_O(b,j,k)&
                                       +D3C_O(c,i,j)+D3C_O(c,i,k)+D3D_O(c,j,k)&
                                       -D3D_V(b,j,c)&
                                       -D3D_V(b,k,c)
                                       deltaD = deltaD + temp/(omega+D)
                                    end do
                                 end do
                              end do
                           end do
                        end do
                        deallocate(idx_table,loc_arr)

              end subroutine ipccp3C_full

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!! SORTING FUNCTIONS !!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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

              subroutine sort3(excits, loc_arr, idx_table, idims, n1, n2, n3, nloc, n3p)

                    integer, intent(in) :: n1, n2, n3, nloc, n3p
                    integer, intent(in) :: idims(3)
                    integer, intent(in) :: idx_table(n1,n2,n3)

                    integer, intent(inout) :: loc_arr(2,nloc)
                    integer, intent(inout) :: excits(n3p,5)

                    integer :: idet
                    integer :: p, q, r
                    integer :: p1, q1, r1, p2, q2, r2
                    integer :: pqr1, pqr2
                    integer, allocatable :: temp(:), idx(:)

                    allocate(temp(n3p),idx(n3p))
                    do idet = 1, n3p
                       p = excits(idet,idims(1)); q = excits(idet,idims(2)); r = excits(idet,idims(3));
                       temp(idet) = idx_table(p,q,r)
                    end do
                    call argsort(temp, idx)
                    excits = excits(idx,:)
                    deallocate(temp,idx)

                    loc_arr(1,:) = 1; loc_arr(2,:) = 0;
                    !!! WARNING: THERE IS A MEMORY LEAK HERE! pqrs2 is used below but is not set if n3p <= 1
                    !if (n3p <= 1) print*, "eomccsdt_p_loops >> WARNING: potential memory leakage in sort4 function. pqrs2 set to -1"
                    if (n3p == 1) then
                       if (excits(1,1)==1 .and. excits(1,2)==1 .and. excits(1,3)==1 .and. excits(1,4)==1 .and. excits(1,5)==1) return
                       p2 = excits(n3p,idims(1)); q2 = excits(n3p,idims(2)); r2 = excits(n3p,idims(3));
                       pqr2 = idx_table(p2,q2,r2)
                    else
                       pqr2 = -1
                    end if
                    do idet = 1, n3p-1
                       p1 = excits(idet,idims(1));   q1 = excits(idet,idims(2));   r1 = excits(idet,idims(3));
                       p2 = excits(idet+1,idims(1)); q2 = excits(idet+1,idims(2)); r2 = excits(idet+1,idims(3));
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

end module ipccp3_loops
