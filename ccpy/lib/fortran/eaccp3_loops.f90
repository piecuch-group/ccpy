module eaccp3_loops

      use pspace_sort, only: get_index_table3, sort3 => sort3_excits
      implicit none

      contains

              subroutine eaccp3A_full(deltaA,deltaB,deltaC,deltaD,&
                                      M3A,L3A,r3a_excits,omega,&
                                      !orbsyms,target_sym,refsym,&
                                      fA_oo,fA_vv,H1A_oo,H1A_vv,&
                                      H2A_vvvv,H2A_oooo,H2A_voov,&
                                      d3A_o,d3A_v,&
                                      n3aaa,noa,nua)

                        ! input variables
                        integer, intent(in) :: noa, nua, n3aaa
                        !integer, intent(in) :: norb, target_sym, refsym
                        !integer, intent(in) :: orbsyms(norb)
                        integer, intent(in) :: r3a_excits(n3aaa,5)
                        real(kind=8), intent(in) :: M3A(nua,nua,nua,noa,noa)
                        real(kind=8), intent(in) :: L3A(nua,nua,nua,noa,noa)
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
                        logical(kind=1) :: qspace(noa,noa)
                        integer :: nloc, idet, idx
                        integer, allocatable :: loc_arr(:,:), idx_table(:,:,:)
                        integer :: excits_buff(n3aaa,5)
                        ! Local variables
                        integer :: j, k, a, b, c
                        real(kind=8) :: D, temp
                        !integer :: sym_a, sym_b, sym_c, sym_j, sym_k, sym

                        deltaA = 0.0d0
                        deltaB = 0.0d0
                        deltaC = 0.0d0
                        deltaD = 0.0d0

                        ! reorder r3a into (a,b,c) order
                        excits_buff(:,:) = r3a_excits(:,:)
                        nloc = nua*(nua-1)*(nua-2)/6
                        allocate(loc_arr(2,nloc))
                        allocate(idx_table(nua,nua,nua))
                        call get_index_table3(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), nua, nua, nua)
                        call sort3(excits_buff, loc_arr, idx_table, (/1,2,3/), nua, nua, nua, nloc, n3aaa)

                        do a = 1,nua
                           !sym_a = orbsyms(a+noa)
                           !sym = ieor(sym_a, refsym)
                           do b = a+1,nua
                              !sym_b = orbsyms(b+noa)
                              !sym = ieor(sym_b, sym)
                              do c = b+1,nua
                                 !sym_c = orbsyms(c+noa)
                                 !sym = ieor(sym_c, sym)
                                 ! Construct Q space for block (a,b,c)
                                 qspace = .true.
                                 idx = idx_table(a,b,c)
                                 if (idx/=0) then
                                    do idet = loc_arr(1,idx), loc_arr(2,idx)
                                       j = excits_buff(idet,4); k = excits_buff(idet,5);
                                       qspace(j,k) = .false.
                                    end do
                                 end if
                                 do j = 1,noa
                                    !sym_j = orbsyms(j)
                                    !sym = ieor(sym_j, sym)
                                    do k = j+1,noa
                                       !sym_k = orbsyms(k)
                                       !sym = ieor(sym_k, sym)
                                       !if (sym /= target_sym) cycle
                                       if (.not. qspace(j,k)) cycle
                                       temp = M3A(a,b,c,j,k)*L3A(a,b,c,j,k)
                                       ! A correction
                                       D = -fA_vv(b,b) - fA_vv(c,c) - fA_vv(a,a) + fA_oo(j,j) + fA_oo(k,k)
                                       deltaA = deltaA + temp/(omega+D)
                                       ! B correction
                                       D = -h1a_vv(b,b) - h1a_vv(c,c) - h1a_vv(a,a) + h1a_oo(j,j) + h1a_oo(k,k)
                                       deltaB = deltaB + temp/(omega+D)
                                       ! C correction
                                       D = D&
                                       -h2a_vvvv(a,b,a,b) - h2a_vvvv(a,c,a,c) - h2a_vvvv(b,c,b,c)&
                                       -h2a_oooo(j,k,j,k)&
                                       -h2a_voov(a,j,j,a) - h2a_voov(a,k,k,a)&
                                       -h2a_voov(b,j,j,b) - h2a_voov(b,k,k,b)&
                                       -h2a_voov(c,j,j,c) - h2a_voov(c,k,k,c)
                                       deltaC = deltaC + temp/(omega+D)
                                       ! D correction
                                       D = D&
                                       +d3a_o(a,j,k) + d3a_o(b,j,k) + d3a_o(c,j,k)&
                                       -d3a_v(a,j,b) - d3a_v(a,j,c) - d3a_v(b,j,c)&
                                       -d3a_v(a,k,b) - d3a_v(a,k,c) - d3a_v(b,k,c)
                                       deltaD = deltaD + temp/(omega+D)
                                    end do
                                 end do
                              end do
                           end do
                        end do
                        deallocate(idx_table,loc_arr)

              end subroutine eaccp3A_full

              subroutine eaccp3B_full(deltaA,deltaB,deltaC,deltaD,&
                                      M3B,L3B,r3b_excits,omega,&
                                      !orbsyms,target_sym,refsym,&
                                      fA_oo,fA_vv,fB_oo,fB_vv,&
                                      H1A_oo,H1A_vv,H1B_oo,H1B_vv,&
                                      H2A_vvvv,H2A_voov,&
                                      H2B_vvvv,H2B_oooo,H2B_ovov,H2B_vovo,&
                                      H2C_voov,&
                                      d3A_o,d3A_v,d3B_o,d3B_v,d3C_o,d3C_v,&
                                      n3aab,noa,nua,nob,nub)

                        ! input variables
                        integer, intent(in) :: noa, nua, nob, nub, n3aab
                        !integer, intent(in) :: norb, target_sym, refsym
                        !integer, intent(in) :: orbsyms(norb)
                        integer, intent(in) :: r3b_excits(n3aab,5)
                        real(kind=8), intent(in) :: M3B(nua,nua,nub,noa,nob)
                        real(kind=8), intent(in) :: L3B(nua,nua,nub,noa,nob)
                        real(kind=8), intent(in) :: omega
                        real(kind=8), intent(in) :: fA_oo(noa,noa), fA_vv(nua,nua)
                        real(kind=8), intent(in) :: fB_oo(nob,nob), fB_vv(nub,nub)
                        real(kind=8), intent(in) :: h1A_oo(noa,noa), h1A_vv(nua,nua)
                        real(kind=8), intent(in) :: h1B_oo(nob,nob), h1B_vv(nub,nub)
                        real(kind=8), intent(in) :: h2a_vvvv(nua,nua,nua,nua)
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
                        logical(kind=1) :: qspace(noa,nob)
                        integer :: nloc, idet, idx
                        integer, allocatable :: loc_arr(:,:), idx_table(:,:,:)
                        integer :: excits_buff(n3aab,5)
                        ! Local variables
                        integer :: j, k, a, b, c
                        real(kind=8) :: D, temp
                        !integer :: sym_a, sym_b, sym_c, sym_j, sym_k, sym

                        deltaA = 0.0d0
                        deltaB = 0.0d0
                        deltaC = 0.0d0
                        deltaD = 0.0d0

                        ! reorder r3b into (a,b,c) order
                        excits_buff(:,:) = r3b_excits(:,:)
                        nloc = nua*(nua-1)/2*nub
                        allocate(loc_arr(2,nloc))
                        allocate(idx_table(nua,nua,nub))
                        call get_index_table3(idx_table, (/1,nua-1/), (/-1,nua/), (/1,nub/), nua, nua, nub)
                        call sort3(excits_buff, loc_arr, idx_table, (/1,2,3/), nua, nua, nub, nloc, n3aab)

                        do a = 1,nua
                           !sym_a = orbsyms(a+noa)
                           !sym = ieor(sym_a, refsym)
                           do b = a+1,nua
                              !sym_b = orbsyms(b+noa)
                              !sym = ieor(sym_b, sym)
                              do c = 1,nub
                                 !sym_c = orbsyms(c+nob)
                                 !sym = ieor(sym_c, sym)
                                 ! Construct Q space for block (a,b,c)
                                 qspace = .true.
                                 idx = idx_table(a,b,c)
                                 if (idx/=0) then
                                    do idet = loc_arr(1,idx), loc_arr(2,idx)
                                       j = excits_buff(idet,4); k = excits_buff(idet,5);
                                       qspace(j,k) = .false.
                                    end do
                                 end if
                                 do j = 1,noa
                                    !sym_j = orbsyms(j)
                                    !sym = ieor(sym_j, sym)
                                    do k = 1,nob
                                       !sym_k = orbsyms(k)
                                       !sym = ieor(sym_k, sym)
                                       !if (sym /= target_sym) cycle
                                       if (.not. qspace(j,k)) cycle
                                       temp = M3B(a,b,c,j,k)*L3B(a,b,c,j,k)
                                       ! A correction
                                       D = -fA_vv(b,b) - fB_vv(c,c) - fA_vv(a,a) + fA_oo(j,j) + fB_oo(k,k)
                                       deltaA = deltaA + temp/(omega+D)
                                       ! B correction
                                       D = -h1a_vv(b,b) - h1b_vv(c,c) - h1a_vv(a,a) + h1a_oo(j,j) + h1b_oo(k,k)
                                       deltaB = deltaB + temp/(omega+D)
                                       ! C correction
                                       D = D&
                                       -h2b_oooo(j,k,j,k)&
                                       -h2c_voov(c,k,k,c)&
                                       -h2a_voov(a,j,j,a) - h2a_voov(b,j,j,b)&
                                       +h2b_vovo(b,k,b,k) + h2b_vovo(a,k,a,k)&
                                       +h2b_ovov(j,c,j,c)&
                                       -h2a_vvvv(a,b,a,b)&
                                       -h2b_vvvv(a,c,a,c) - h2b_vvvv(b,c,b,c)
                                       deltaC = deltaC + temp/(omega+D)
                                       ! D correction
                                       D = D&
                                       +D3B_O(a,j,k)&
                                       +D3B_O(b,j,k)&
                                       +D3C_O(c,j,k)&
                                       -D3A_V(a,j,b)-D3B_V(a,j,c)-D3B_V(b,j,c)&
                                       -D3C_V(a,k,c)-D3C_V(b,k,c)
                                       deltaD = deltaD + temp/(omega+D)
                                    end do
                                 end do
                              end do
                           end do
                        end do
                        deallocate(idx_table,loc_arr)

              end subroutine eaccp3B_full

              subroutine eaccp3C_full(deltaA,deltaB,deltaC,deltaD,&
                                 M3C,L3C,r3c_excits,omega,&
                                 !orbsyms,target_sym,refsym,&
                                 fA_oo,fA_vv,fB_oo,fB_vv,&
                                 H1A_oo,H1A_vv,H1B_oo,H1B_vv,&
                                 H2B_vvvv,H2B_ovov,H2B_vovo,&
                                 H2C_vvvv,H2C_oooo,H2C_voov,&
                                 d3B_o,d3B_v,d3C_o,d3C_v,d3D_o,d3D_v,&
                                 n3abb,noa,nua,nob,nub)

                        ! input variables
                        integer, intent(in) :: noa, nua, nob, nub, n3abb
                        !integer, intent(in) :: norb, target_sym, refsym
                        !integer, intent(in) :: orbsyms(norb)
                        integer, intent(in) :: r3c_excits(n3abb,5)
                        real(kind=8), intent(in) :: M3C(nua,nub,nub,nob,nob)
                        real(kind=8), intent(in) :: L3C(nua,nub,nub,nob,nob)
                        real(kind=8), intent(in) :: omega
                        real(kind=8), intent(in) :: fA_oo(noa,noa), fA_vv(nua,nua)
                        real(kind=8), intent(in) :: fB_oo(nob,nob), fB_vv(nub,nub)
                        real(kind=8), intent(in) :: h1A_oo(noa,noa), h1A_vv(nua,nua)
                        real(kind=8), intent(in) :: h1B_oo(nob,nob), h1B_vv(nub,nub)
                        real(kind=8), intent(in) :: h2b_vvvv(nua,nub,nua,nub)
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
                        logical(kind=1) :: qspace(nob,nob)
                        integer :: nloc, idet, idx
                        integer, allocatable :: loc_arr(:,:), idx_table(:,:,:)
                        integer :: excits_buff(n3abb,5)
                        ! Local variables
                        integer :: j, k, a, b, c
                        real(kind=8) :: D, temp
                        !integer :: sym_a, sym_b, sym_c, sym_j, sym_k, sym

                        deltaA = 0.0d0
                        deltaB = 0.0d0
                        deltaC = 0.0d0
                        deltaD = 0.0d0

                        ! reorder r3c into (a,b,c) order
                        excits_buff(:,:) = r3c_excits(:,:)
                        nloc = nub*(nub-1)/2*nua
                        allocate(loc_arr(2,nloc))
                        allocate(idx_table(nub,nub,nua))
                        call get_index_table3(idx_table, (/1,nub-1/), (/-1,nub/), (/1,nua/), nub, nub, nua)
                        call sort3(excits_buff, loc_arr, idx_table, (/2,3,1/), nub, nub, nua, nloc, n3abb)

                        do a = 1,nua
                           !sym_a = orbsyms(a+noa)
                           !sym = ieor(sym_a, refsym)
                           do b = 1,nub
                              !sym_b = orbsyms(b+nob)
                              !sym = ieor(sym_b, sym)
                              do c = b+1,nub
                                 !sym_c = orbsyms(c+nob)
                                 !sym = ieor(sym_c, sym)
                                 ! Construct Q space for block (a,b,c)
                                 qspace = .true.
                                 idx = idx_table(b,c,a)
                                 if (idx/=0) then
                                    do idet = loc_arr(1,idx), loc_arr(2,idx)
                                       j = excits_buff(idet,4); k = excits_buff(idet,5);
                                       qspace(j,k) = .false.
                                    end do
                                 end if
                                 do j = 1,nob
                                    !sym_j = orbsyms(j)
                                    !sym = ieor(sym_j, sym)
                                    do k = j+1,nob
                                       !sym_k = orbsyms(k)
                                       !sym = ieor(sym_k, sym)
                                       !if (sym /= target_sym) cycle
                                       if (.not. qspace(j,k)) cycle
                                       temp = M3C(a,b,c,j,k)*L3C(a,b,c,j,k)
                                       ! A correction
                                       D = -fB_vv(b,b) - fB_vv(c,c) - fA_vv(a,a) + fB_oo(j,j) + fB_oo(k,k)
                                       deltaA = deltaA + temp/(omega+D)
                                       ! B correction
                                       D = -h1b_vv(b,b) - h1b_vv(c,c) - h1a_vv(a,a) + h1b_oo(j,j) + h1b_oo(k,k)
                                       deltaB = deltaB + temp/(omega+D)
                                       ! C correction
                                       D = D&
                                       +H2B_vovo(a,j,a,j)-H2C_voov(b,j,j,b)-H2C_voov(c,j,j,c)&
                                       +H2B_vovo(a,k,a,k)-H2C_voov(b,k,k,b)-H2C_voov(c,k,k,c)&
                                       -H2C_oooo(k,j,k,j)&
                                       -H2B_vvvv(a,b,a,b)-H2B_vvvv(a,c,a,c)-H2C_vvvv(c,b,c,b)
                                       deltaC = deltaC + temp/(omega+D)
                                       ! D correction
                                       D = D&
                                       +D3D_O(b,j,k)&
                                       +D3D_O(c,j,k)&
                                       -D3C_V(a,j,b)-D3C_V(a,j,c)-D3D_V(b,j,c)&
                                       -D3C_V(a,k,b)-D3C_V(a,k,c)-D3D_V(b,k,c)
                                       deltaD = deltaD + temp/(omega+D)
                                    end do
                                 end do
                              end do
                           end do
                        end do
                        deallocate(idx_table,loc_arr)

              end subroutine eaccp3C_full

end module eaccp3_loops
