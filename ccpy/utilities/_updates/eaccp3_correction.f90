module eaccp3_correction

      use omp_lib

      implicit none

      contains
         
              subroutine ccp3a_jk(deltaA,deltaB,deltaC,deltaD,&
                                    j,k,omega,&
                                    M3A,L3A,r3a_excits,&
                                    fA_oo,fA_vv,H1A_oo,H1A_vv,&
                                    H2A_voov,H2A_oooo,H2A_vvvv,&
                                    D3A_O,D3A_V,n3aaa,noa,nua)

                        integer, intent(in) :: noa, nua, n3aaa
                        integer, intent(in) :: j, k
                        integer, intent(in) :: r3a_excits(n3aaa,5)
                        real(kind=8), intent(in) :: M3A(1:nua,1:nua,1:nua),&
                                                    L3A(1:nua,1:nua,1:nua),&
                                                    fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),&
                                                    H1A_oo(1:noa,1:noa),H1A_vv(1:nua,1:nua),&
                                                    H2A_voov(1:nua,1:noa,1:noa,1:nua),&
                                                    H2A_oooo(1:noa,1:noa,1:noa,1:noa),&
                                                    H2A_vvvv(1:nua,1:nua,1:nua,1:nua),&
                                                    D3A_O(1:nua,1:noa,1:noa),&
                                                    D3A_V(1:nua,1:noa,1:nua)
                        real(kind=8), intent(in) :: omega
                        ! output variables
                        real(kind=8), intent(inout) :: deltaA
                        !f2py intent(in,out) :: deltaA
                        real(kind=8), intent(inout) :: deltaB
                        !f2py intent(in,out) :: deltaB
                        real(kind=8), intent(inout) :: deltaC
                        !f2py intent(in,out) :: deltaC
                        real(kind=8), intent(inout) :: deltaD
                        !f2py intent(in,out) :: deltaD
                        
                        ! Low-memory looping variables
                        logical(kind=1) :: qspace(nua,nua,nua)
                        integer :: nloc, idet, idx
                        integer, allocatable :: loc_arr(:,:), idx_table(:,:)
                        integer :: excits_buff(n3aaa,5)
                        real(kind=8) :: amps_buff(n3aaa)
                        ! local variables
                        integer :: a, b, c
                        real(kind=8) :: D, LM

                        ! reorder r3a into (i,j,k) order
                        excits_buff(:,:) = r3a_excits(:,:)
                        amps_buff = 0.0
                        nloc = noa*(noa-1)/2
                        allocate(loc_arr(2,nloc))
                        allocate(idx_table(noa,noa))
                        call get_index_table2(idx_table, (/1,noa-1/), (/-1,noa/), noa, noa)
                        call sort2(excits_buff, amps_buff, loc_arr, idx_table, (/4,5/), noa, noa, nloc, n3aaa)
                        
                        ! Construct Q space for block (j,k)
                        qspace = .true.
                        idx = idx_table(j,k)
                        if (idx/=0) then
                           do idet = loc_arr(1,idx), loc_arr(2,idx)
                              a = excits_buff(idet,1); b = excits_buff(idet,2); c = excits_buff(idet,3);
                              qspace(a,b,c) = .false.
                           end do
                        end if
                        deallocate(idx_table,loc_arr)
                        
                        do a = 1, nua
                            do b = a+1, nua
                                do c = b+1, nua
                                    if (.not. qspace(a,b,c)) cycle
                                    LM = M3A(a,b,c) * L3A(a,b,c)

                                    D = fA_oo(j,j) + fA_oo(k,k)&
                                    - fA_vv(a,a) - fA_vv(b,b) - fA_vv(c,c)

                                    deltaA = deltaA + LM/(omega+D)

                                    D = H1A_oo(j,j) + H1A_oo(k,k)&
                                    - H1A_vv(a,a) - H1A_vv(b,b) - H1A_vv(c,c)
                                    deltaB = deltaB + LM/(omega+D)

                                    D = D &
                                    -H2A_voov(a,j,j,a) - H2A_voov(b,j,j,b) - H2A_voov(c,j,j,c)&
                                    -H2A_voov(a,k,k,a) - H2A_voov(b,k,k,b) - H2A_voov(c,k,k,c)&
                                    -H2A_oooo(k,j,k,j)&
                                    -H2A_vvvv(b,a,b,a) - H2A_vvvv(c,a,c,a) - H2A_vvvv(c,b,c,b)
                                    deltaC = deltaC + LM/(omega+D)

                                    D = D &
                                    +D3A_O(a,j,k)&
                                    +D3A_O(b,j,k)&
                                    +D3A_O(c,j,k)&
                                    -D3A_V(a,j,b)-D3A_V(a,j,c)-D3A_V(b,j,c)&
                                    -D3A_V(a,k,b)-D3A_V(a,k,c)-D3A_V(b,k,c)
                                    deltaD = deltaD + LM/(omega+D)
                                end do
                            end do
                        end do
              end subroutine ccp3a_jk
         
              subroutine ccp3b_jk(deltaA,deltaB,deltaC,deltaD,&
                                    j,k,omega,&
                                    M3B,L3B,r3b_excits,&
                                    fA_oo,fA_vv,fB_oo,fB_vv,&
                                    H1A_oo,H1A_vv,H1B_oo,H1B_vv,&
                                    H2A_voov,H2A_oooo,H2A_vvvv,&
                                    H2B_ovov,H2B_vovo,&
                                    H2B_oooo,H2B_vvvv,&
                                    H2C_voov,&
                                    D3A_O,D3A_V,D3B_O,D3B_V,D3C_O,D3C_V,&
                                    n3aab,noa,nua,nob,nub)

                        integer, intent(in) :: noa, nua, nob, nub, n3aab
                        integer, intent(in) :: j, k
                        integer, intent(in) :: r3b_excits(n3aab,5)
                        real(kind=8), intent(in) :: M3B(1:nua,1:nua,1:nub),&
                                                    L3B(1:nua,1:nua,1:nub),&
                                                    fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),&
                                                    fB_oo(1:nob,1:nob),fB_vv(1:nub,1:nub),&
                                                    H1A_oo(1:noa,1:noa),H1A_vv(1:nua,1:nua),&
                                                    H1B_oo(1:nob,1:nob),H1B_vv(1:nub,1:nub),&
                                                    H2A_voov(1:nua,1:noa,1:noa,1:nua),&
                                                    H2A_oooo(1:noa,1:noa,1:noa,1:noa),&
                                                    H2A_vvvv(1:nua,1:nua,1:nua,1:nua),&
                                                    H2B_ovov(1:noa,1:nub,1:noa,1:nub),&
                                                    H2B_vovo(1:nua,1:nob,1:nua,1:nob),&
                                                    H2B_oooo(1:noa,1:nob,1:noa,1:nob),&
                                                    H2B_vvvv(1:nua,1:nub,1:nua,1:nub),&
                                                    H2C_voov(1:nub,1:nob,1:nob,1:nub),&
                                                    D3A_O(1:nua,1:noa,1:noa),&
                                                    D3A_V(1:nua,1:noa,1:nua),&
                                                    D3B_O(1:nua,1:noa,1:nob),&
                                                    D3B_V(1:nua,1:noa,1:nub),&
                                                    D3C_O(1:nub,1:noa,1:nob),&
                                                    D3C_V(1:nua,1:nob,1:nub)
                        real(kind=8), intent(in) :: omega
                        ! output variables
                        real(kind=8), intent(inout) :: deltaA
                        !f2py intent(in,out) :: deltaA
                        real(kind=8), intent(inout) :: deltaB
                        !f2py intent(in,out) :: deltaB
                        real(kind=8), intent(inout) :: deltaC
                        !f2py intent(in,out) :: deltaC
                        real(kind=8), intent(inout) :: deltaD
                        !f2py intent(in,out) :: deltaD
                        
                        ! Low-memory looping variables
                        logical(kind=1) :: qspace(nua,nua,nub)
                        integer :: nloc, idet, idx
                        integer, allocatable :: loc_arr(:,:), idx_table(:,:)
                        integer :: excits_buff(n3aab,5)
                        real(kind=8) :: amps_buff(n3aab)
                        ! local variables
                        integer :: a, b, c
                        real(kind=8) :: D, LM

                        ! reorder r3b into (j,k) order
                        excits_buff(:,:) = r3b_excits(:,:)
                        amps_buff = 0.0
                        nloc = noa*nob
                        allocate(loc_arr(2,nloc))
                        allocate(idx_table(noa,nob))
                        call get_index_table2(idx_table, (/1,noa/), (/1,nob/), noa, nob)
                        call sort2(excits_buff, amps_buff, loc_arr, idx_table, (/4,5/), noa, nob, nloc, n3aab)
                        
                        ! Construct Q space for block (j,k)
                        qspace = .true.
                        idx = idx_table(j,k)
                        if (idx/=0) then
                           do idet = loc_arr(1,idx), loc_arr(2,idx)
                              a = excits_buff(idet,1); b = excits_buff(idet,2); c = excits_buff(idet,3);
                              qspace(a,b,c) = .false.
                           end do
                        end if
                        deallocate(idx_table,loc_arr)
                        
                        do a = 1,nua
                            do b = a+1,nua
                                do c = 1,nub
                                    if (.not. qspace(a,b,c)) cycle
                                    LM = M3B(a,b,c) * L3B(a,b,c)
                                    
                                    D = fA_oo(j,j) + fB_oo(k,k)&
                                    - fA_vv(a,a) - fA_vv(b,b) - fB_vv(c,c)
                                    deltaA = deltaA + LM/(omega+D)
   
                                    D = H1A_oo(j,j) + H1B_oo(k,k)&
                                    - H1A_vv(a,a) - H1A_vv(b,b) - H1B_vv(c,c)
                                    deltaB = deltaB + LM/(omega+D)
   
                                    D = D &
                                    -H2A_voov(a,j,j,a)-H2A_voov(b,j,j,b)+H2B_ovov(j,c,j,c)&
                                    +H2B_vovo(a,k,a,k)+H2B_vovo(b,k,b,k)-H2C_voov(c,k,k,c)&
                                    -H2B_oooo(j,k,j,k)&
                                    -H2A_vvvv(b,a,b,a)-H2B_vvvv(a,c,a,c)-H2B_vvvv(b,c,b,c)
                                    deltaC = deltaC + LM/(omega+D)
   
                                    D = D &
                                    +D3B_O(a,j,k)&
                                    +D3B_O(b,j,k)&
                                    +D3C_O(c,j,k)&
                                    -D3A_V(a,j,b)-D3B_V(a,j,c)-D3B_V(b,j,c)&
                                    -D3C_V(a,k,c)-D3C_V(b,k,c)
                                    deltaD = deltaD + LM/(omega+D)
                                end do
                            end do
                        end do
              end subroutine ccp3b_jk
         
              subroutine ccp3c_jk(deltaA,deltaB,deltaC,deltaD,&
                                    j,k,omega,&
                                    M3C,L3C,r3c_excits,&
                                    fA_oo,fA_vv,fB_oo,fB_vv,&
                                    H1A_oo,H1A_vv,H1B_oo,H1B_vv,&
                                    H2A_voov,&
                                    H2B_ovov,H2B_vovo,&
                                    H2B_oooo,H2B_vvvv,&
                                    H2C_voov,H2C_oooo,H2C_vvvv,&
                                    D3B_O,D3B_V,D3C_O,D3C_V,D3D_O,D3D_V,&
                                    n3abb,noa,nua,nob,nub)

                        integer, intent(in) :: noa, nua, nob, nub, n3abb
                        integer, intent(in) :: j, k
                        integer, intent(in) :: r3c_excits(n3abb,5)
                        real(kind=8), intent(in) :: M3C(1:nua,1:nub,1:nub),&
                                                    L3C(1:nua,1:nub,1:nub),&
                                                    fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),&
                                                    fB_oo(1:nob,1:nob),fB_vv(1:nub,1:nub),&
                                                    H1A_oo(1:noa,1:noa),H1A_vv(1:nua,1:nua),&
                                                    H1B_oo(1:nob,1:nob),H1B_vv(1:nub,1:nub),&
                                                    H2A_voov(1:nua,1:noa,1:noa,1:nua),&
                                                    H2B_ovov(1:noa,1:nub,1:noa,1:nub),&
                                                    H2B_vovo(1:nua,1:nob,1:nua,1:nob),&
                                                    H2B_oooo(1:noa,1:nob,1:noa,1:nob),&
                                                    H2B_vvvv(1:nua,1:nub,1:nua,1:nub),&
                                                    H2C_voov(1:nub,1:nob,1:nob,1:nub),&
                                                    H2C_oooo(1:nob,1:nob,1:nob,1:nob),&
                                                    H2C_vvvv(1:nub,1:nub,1:nub,1:nub),&
                                                    D3B_O(1:nua,1:noa,1:nob),&
                                                    D3B_V(1:nua,1:noa,1:nub),&
                                                    D3C_O(1:nub,1:noa,1:nob),&
                                                    D3C_V(1:nua,1:nob,1:nub),&
                                                    D3D_O(1:nub,1:nob,1:nob),&
                                                    D3D_V(1:nub,1:nob,1:nub)
                        real(kind=8), intent(in) :: omega
                        ! output variables
                        real(kind=8), intent(inout) :: deltaA
                        !f2py intent(in,out) :: deltaA
                        real(kind=8), intent(inout) :: deltaB
                        !f2py intent(in,out) :: deltaB
                        real(kind=8), intent(inout) :: deltaC
                        !f2py intent(in,out) :: deltaC
                        real(kind=8), intent(inout) :: deltaD
                        !f2py intent(in,out) :: deltaD
                        
                        ! Low-memory looping variables
                        logical(kind=1) :: qspace(nua,nub,nub)
                        integer :: nloc, idet, idx
                        integer, allocatable :: loc_arr(:,:), idx_table(:,:)
                        integer :: excits_buff(n3abb,5)
                        real(kind=8) :: amps_buff(n3abb)
                        ! local variables
                        integer :: a, b, c
                        real(kind=8) :: D, LM

                        ! reorder r3b into (i,j,k) order
                        excits_buff(:,:) = r3c_excits(:,:)
                        amps_buff = 0.0
                        nloc = nob*(nob-1)/2
                        allocate(loc_arr(2,nloc))
                        allocate(idx_table(nob,nob))
                        call get_index_table2(idx_table, (/1,nob-1/), (/-1,nob/), nob, nob)
                        call sort2(excits_buff, amps_buff, loc_arr, idx_table, (/4,5/), nob, nob, nloc, n3abb)
                        
                        ! Construct Q space for block (j,k)
                        qspace = .true.
                        idx = idx_table(j,k)
                        if (idx/=0) then
                           do idet = loc_arr(1,idx), loc_arr(2,idx)
                              a = excits_buff(idet,1); b = excits_buff(idet,2); c = excits_buff(idet,3);
                              qspace(a,b,c) = .false.
                           end do
                        end if
                        deallocate(idx_table,loc_arr)
                        
                        do a = 1,nua
                            do b = 1,nub
                                do c = b+1,nub
                                    if (.not. qspace(a,b,c)) cycle
                                    LM = M3C(a,b,c) * L3C(a,b,c)
   
                                    D = fB_oo(j,j) + fB_oo(k,k)&
                                    - fA_vv(a,a) - fB_vv(b,b) - fB_vv(c,c)
                                    deltaA = deltaA + LM/(omega+D)
   
                                    D = H1B_oo(j,j) + H1B_oo(k,k)&
                                    - H1A_vv(a,a) - H1B_vv(b,b) - H1B_vv(c,c)
                                    deltaB = deltaB + LM/(omega+D)
   
                                    D = D &
                                    +H2B_vovo(a,j,a,j)-H2C_voov(b,j,j,b)-H2C_voov(c,j,j,c)&
                                    +H2B_vovo(a,k,a,k)-H2C_voov(b,k,k,b)-H2C_voov(c,k,k,c)&
                                    -H2C_oooo(k,j,k,j)&
                                    -H2B_vvvv(a,b,a,b)-H2B_vvvv(a,c,a,c)-H2C_vvvv(c,b,c,b)
                                    deltaC = deltaC + LM/(omega+D)
                                    
                                    D = D &
                                    +D3D_O(b,j,k)&
                                    +D3D_O(c,j,k)&
                                    -D3C_V(a,j,b)-D3C_V(a,j,c)-D3D_V(b,j,c)&
                                    -D3C_V(a,k,b)-D3C_V(a,k,c)-D3D_V(b,k,c)
                                    deltaD = deltaD + LM/(omega+D)
                                end do
                            end do
                        end do
              end subroutine ccp3c_jk

              subroutine build_moments3a_jk(resid,j,k,&
                                     r2a,&
                                     r3a_amps,r3a_excits,&
                                     r3b_amps,r3b_excits,&
                                     t2a,&
                                     h1a_oo,h1a_vv,&
                                     h2a_vvvv,h2a_oooo,h2a_voov,h2a_vooo,h2a_vvov,&
                                     h2b_voov,&
                                     x2a_voo,x2a_vvv,&
                                     n3aaa,n3aab,&
                                     noa,nua,nob,nub)

                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa, n3aab
                  ! Occupied orbital block indices
                  integer, intent(in) :: j, k
                  ! Input R and T arrays
                  real(kind=8), intent(in) :: r2a(nua,nua,noa), t2a(nua,nua,noa,noa)
                  integer, intent(in) :: r3a_excits(n3aaa,5), r3b_excits(n3aab,5)
                  real(kind=8), intent(in) :: r3a_amps(n3aaa), r3b_amps(n3aab)
                  ! Input H  and X arrays
                  real(kind=8), intent(in) :: h1a_oo(noa,noa)
                  real(kind=8), intent(in) :: h1a_vv(nua,nua)
                  real(kind=8), intent(in) :: h2a_vvvv(nua,nua,nua,nua)
                  real(kind=8), intent(in) :: h2a_oooo(noa,noa,noa,noa)
                  real(kind=8), intent(in) :: h2a_voov(nua,noa,noa,nua)
                  real(kind=8), intent(in) :: h2a_vooo(nua,noa,noa,noa)
                  real(kind=8), intent(in) :: h2a_vvov(nua,nua,noa,nua)
                  real(kind=8), intent(in) :: h2b_voov(nua,nob,noa,nub)
                  real(kind=8), intent(in) :: x2a_voo(nua,noa,noa)
                  real(kind=8), intent(in) :: x2a_vvv(nua,nua,nua)
                  ! Output and Inout variables
                  real(kind=8), intent(out) :: resid(nua,nua,nua)
                  ! Local variables
                  integer, allocatable :: excits_buff(:,:), r3a_excits_copy(:,:)
                  real(kind=8), allocatable :: amps_buff(:), r3a_amps_copy(:)
                  integer, allocatable :: idx_table4(:,:,:,:)
                  integer, allocatable :: idx_table3(:,:,:)
                  integer, allocatable :: idx_table2(:,:)
                  integer, allocatable :: loc_arr(:,:)
                  real(kind=8) :: r_amp, hmatel, res_mm23
                  integer :: a, b, c, d, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc
                  ! Q space array
                  logical(kind=1) :: qspace(nua,nua,nua)
                  
                  ! Zero the container that holds H*R
                  resid = 0.0d0
                  
                  ! copy over r3a_amps and r3a_excits
                  allocate(r3a_amps_copy(n3aaa),r3a_excits_copy(n3aaa,5))
                  r3a_amps_copy(:) = r3a_amps(:)
                  r3a_excits_copy(:,:) = r3a_excits(:,:)
                  
                  ! reorder r3a into (j,k) order
                  nloc = noa*(noa-1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table2(noa,noa))
                  call get_index_table2(idx_table2, (/1,noa-1/), (/-1,noa/), noa, noa)
                  call sort2(r3a_excits_copy, r3a_amps_copy, loc_arr, idx_table2, (/4,5/), noa, noa, nloc, n3aaa)
                  ! Construct Q space for block (j,k)
                  qspace = .true.
                  idx = idx_table2(j,k)
                  if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        a = r3a_excits_copy(jdet,1); b = r3a_excits_copy(jdet,2); c = r3a_excits_copy(jdet,3);
                        qspace(a,b,c) = .false.
                     end do
                  end if
                  deallocate(loc_arr,idx_table2)

                  !!!! diagram 1: -A(jk) h1a(mj)*r3a(abcmk)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)*(nua-2)/6 * noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nua,nua,noa))
                  !!! SB: (1,2,3,4) !!!
                  call get_index_table4(idx_table4, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/1,noa-1/), nua, nua, nua, noa)
                  call sort4(r3a_excits_copy, r3a_amps_copy, loc_arr, idx_table4, (/1,2,3,4/), nua, nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits_copy,&
                  !$omp r3a_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_oo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a = 1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(a,b,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3a_excits_copy(jdet,5);
                        ! compute < abcjk | h1a(oo) | abcjm >
                        hmatel = -h1a_oo(m,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                     end do
                     ! (jk)
                     idx = idx_table4(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           m = r3a_excits_copy(jdet,5);
                           ! compute < abcjk | h1a(oo) | abckm >
                           hmatel = h1a_oo(m,j)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,3,5) !!!
                  call get_index_table4(idx_table4, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/2,noa/), nua, nua, nua, noa)
                  call sort4(r3a_excits_copy, r3a_amps_copy, loc_arr, idx_table4, (/1,2,3,5/), nua, nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits_copy,&
                  !$omp r3a_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_oo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a = 1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(a,b,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3a_excits_copy(jdet,4);
                        ! compute < abcjk | h1a(oo) | abcmk >
                        hmatel = -h1a_oo(m,j)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                     end do
                     ! (jk)
                     idx = idx_table4(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           m = r3a_excits_copy(jdet,4);
                           ! compute < abcjk | h1a(oo) | abcmj >
                           hmatel = h1a_oo(m,k)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 2: A(b/ac) h1a(be)*r3a(aecjk)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*(nua - 2)/2*noa*(noa - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nua,noa,noa))
                  !!! SB: (1,3,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nua-2/), (/-2,nua/), (/1,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(r3a_excits_copy, r3a_amps_copy, loc_arr, idx_table4, (/1,3,4,5/), nua, nua, noa, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits_copy,&
                  !$omp r3a_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_vv,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a = 1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(a,c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3a_excits_copy(jdet,2);
                        ! compute < abcjk | h1a(vv) | aecjk >
                        hmatel = h1a_vv(b,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits_copy(jdet,2);
                           ! compute < abcjk | h1a(vv) | becjk >
                           hmatel = -h1a_vv(a,e)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table4(a,b,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits_copy(jdet,2);
                           ! compute < abcjk | h1a(vv) | aebjk >
                           hmatel = -h1a_vv(c,e)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nua-2/), (/-1,nua-1/), (/1,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(r3a_excits_copy, r3a_amps_copy, loc_arr, idx_table4, (/1,2,4,5/), nua, nua, noa, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits_copy,&
                  !$omp r3a_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_vv,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a = 1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(a,b,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3a_excits_copy(jdet,3);
                        ! compute < abcjk | h1a(vv) | abfjk >
                        hmatel = h1a_vv(c,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                     end do
                     ! (ac)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = r3a_excits_copy(jdet,3);
                           ! compute < abcjk | h1a(vv) | bcfjk >
                           hmatel = h1a_vv(a,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table4(a,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = r3a_excits_copy(jdet,3);
                           ! compute < abcjk | h1a(vv) | acfjk >
                           hmatel = -h1a_vv(b,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4,5) !!!
                  call get_index_table4(idx_table4, (/2,nua-1/), (/-1,nua/), (/1,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(r3a_excits_copy, r3a_amps_copy, loc_arr, idx_table4, (/2,3,4,5/), nua, nua, noa, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits_copy,&
                  !$omp r3a_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_vv,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a = 1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3a_excits_copy(jdet,1);
                        ! compute < abcjk | h1a(vv) | dbcjk >
                        hmatel = h1a_vv(a,d)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table4(a,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits_copy(jdet,1);
                           ! compute < abcjk | h1a(vv) | dacjk >
                           hmatel = -h1a_vv(b,d)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (ac)
                     idx = idx_table4(a,b,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits_copy(jdet,1);
                           ! compute < abcjk | h1a(vv) | dabjk >
                           hmatel = h1a_vv(c,d)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 3: 1/2 A(c/ab) h2a(abef)*r3a(efcjk)
                  ! allocate new sorting arrays
                  nloc = (nua - 2)*noa*(noa - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,noa,noa))
                  !!! SB: (3,4,5) !!!
                  call get_index_table3(idx_table3, (/3,nua/), (/1,noa-1/), (/-1,noa/), nua, noa, noa)
                  call sort3(r3a_excits_copy, r3a_amps_copy, loc_arr, idx_table3, (/3,4,5/), nua, noa, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits_copy,&
                  !$omp r3a_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_vvvv,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a = 1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3a_excits_copy(jdet,1); e = r3a_excits_copy(jdet,2);
                        ! compute < abcjk | h2a(vvvv) | decjk >
                        !hmatel = h2a_vvvv(a,b,d,e)
                        hmatel = h2a_vvvv(d,e,a,b)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                     end do
                     ! (ac)
                     idx = idx_table3(a,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits_copy(jdet,1); e = r3a_excits_copy(jdet,2);
                           ! compute < abcjk | h2a(vvvv) | deajk >
                           !hmatel = h2a_vvvv(b,c,d,e)
                           hmatel = h2a_vvvv(d,e,b,c)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table3(b,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits_copy(jdet,1); e = r3a_excits_copy(jdet,2);
                           ! compute < abcjk | h2a(vvvv) | debjk >
                           !hmatel = -h2a_vvvv(a,c,d,e)
                           hmatel = -h2a_vvvv(d,e,a,c)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,4,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/1,noa-1/), (/-1,noa/), nua, noa, noa)
                  call sort3(r3a_excits_copy, r3a_amps_copy, loc_arr, idx_table3, (/1,4,5/), nua, noa, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits_copy,&
                  !$omp r3a_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_vvvv,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a = 1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3a_excits_copy(jdet,2); f = r3a_excits_copy(jdet,3);
                        ! compute < abcjk | h2a(vvvv) | aefjk >
                        !hmatel = h2a_vvvv(b,c,e,f)
                        hmatel = h2a_vvvv(e,f,b,c)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(b,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits_copy(jdet,2); f = r3a_excits_copy(jdet,3);
                           ! compute < abcjk | h2a(vvvv) | befjk >
                           !hmatel = -h2a_vvvv(a,c,e,f)
                           hmatel = -h2a_vvvv(e,f,a,c)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (ac)
                     idx = idx_table3(c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits_copy(jdet,2); f = r3a_excits_copy(jdet,3);
                           ! compute < abcjk | h2a(vvvv) | cefjk >
                           !hmatel = h2a_vvvv(a,b,e,f)
                           hmatel = h2a_vvvv(e,f,a,b)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,4,5)
                  call get_index_table3(idx_table3, (/2,nua-1/), (/1,noa-1/), (/-1,noa/), nua, noa, noa)
                  call sort3(r3a_excits_copy, r3a_amps_copy, loc_arr, idx_table3, (/2,4,5/), nua, noa, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits_copy,&
                  !$omp r3a_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_vvvv,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a = 1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(b,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3a_excits_copy(jdet,1); f = r3a_excits_copy(jdet,3);
                        ! compute < abcjk | h2a(vvvv) | dbfjk >
                        !hmatel = h2a_vvvv(a,c,d,f)
                        hmatel = h2a_vvvv(d,f,a,c)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(a,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits_copy(jdet,1); f = r3a_excits_copy(jdet,3);
                           ! compute < abcjk | h2a(vvvv) | dafjk >
                           !hmatel = -h2a_vvvv(b,c,d,f)
                           hmatel = -h2a_vvvv(d,f,b,c)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table3(c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits_copy(jdet,1); f = r3a_excits_copy(jdet,3);
                           ! compute < abcjk | h2a(vvvv) | dcfjk >
                           !hmatel = -h2a_vvvv(a,b,d,f)
                           hmatel = -h2a_vvvv(d,f,a,b)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 4: 1/4 h2a(mnjk)*r3a(abcmn)
                  ! allocate new sorting arrays
                  nloc = (nua - 2)*(nua - 1)*nua/6
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,nua))
                  !!! SB: (1,2,3) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), nua, nua, nua)
                  call sort3(r3a_excits_copy, r3a_amps_copy, loc_arr, idx_table3, (/1,2,3/), nua, nua, nua, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits_copy,&
                  !$omp r3a_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_oooo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a = 1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3a_excits_copy(jdet,4); n = r3a_excits_copy(jdet,5);
                        ! compute < abcjk | h2a(oooo) | abcmn >
                        hmatel = h2a_oooo(m,n,j,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                     end do
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 5: A(c/ab)A(jk) h2a(cmke)*r3a(abejm)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*(nua - 2)/2*(noa - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,noa))
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-1,nua-1/), (/1,noa-1/), nua, nua, noa)
                  call sort3(r3a_excits_copy, r3a_amps_copy, loc_arr, idx_table3, (/1,2,4/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits_copy,&
                  !$omp r3a_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a = 1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3a_excits_copy(jdet,3); n = r3a_excits_copy(jdet,5);
                        ! compute < abcjk | h2a(voov) | abfjn >
                        hmatel = h2a_voov(c,n,k,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                     end do
                     ! (ac), (-1)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = r3a_excits_copy(jdet,3); n = r3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | bcfjn >
                           hmatel = h2a_voov(a,n,k,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = r3a_excits_copy(jdet,3); n = r3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | acfjn >
                           hmatel = -h2a_voov(b,n,k,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = r3a_excits_copy(jdet,3); n = r3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | abfkn >
                           hmatel = -h2a_voov(c,n,j,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (ac)(jk), (-1)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = r3a_excits_copy(jdet,3); n = r3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | bcfkn >
                           hmatel = -h2a_voov(a,n,j,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = r3a_excits_copy(jdet,3); n = r3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | acfkn >
                           hmatel = h2a_voov(b,n,j,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-2,nua/), (/1,noa-1/), nua, nua, noa)
                  call sort3(r3a_excits_copy, r3a_amps_copy, loc_arr, idx_table3, (/1,3,4/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits_copy,&
                  !$omp r3a_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a = 1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3a_excits_copy(jdet,2); n = r3a_excits_copy(jdet,5);
                        ! compute < abcjk | h2a(voov) | aecjn >
                        hmatel = h2a_voov(b,n,k,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits_copy(jdet,2); n = r3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | becjn >
                           hmatel = -h2a_voov(a,n,k,e)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits_copy(jdet,2); n = r3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | aebjn >
                           hmatel = -h2a_voov(c,n,k,e)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits_copy(jdet,2); n = r3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | aeckn >
                           hmatel = -h2a_voov(b,n,j,e)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits_copy(jdet,2); n = r3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | beckn >
                           hmatel = h2a_voov(a,n,j,e)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits_copy(jdet,2); n = r3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | aebkn >
                           hmatel = h2a_voov(c,n,j,e)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4) !!!
                  call get_index_table3(idx_table3, (/2,nua-1/), (/-1,nua/), (/1,noa-1/), nua, nua, noa)
                  call sort3(r3a_excits_copy, r3a_amps_copy, loc_arr, idx_table3, (/2,3,4/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits_copy,&
                  !$omp r3a_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a = 1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(b,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3a_excits_copy(jdet,1); n = r3a_excits_copy(jdet,5);
                        ! compute < abcjk | h2a(voov) | dbcjn >
                        hmatel = h2a_voov(a,n,k,d)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits_copy(jdet,1); n = r3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | dacjn >
                           hmatel = -h2a_voov(b,n,k,d)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (ac), (-1)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits_copy(jdet,1); n = r3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | dabjn >
                           hmatel = h2a_voov(c,n,k,d)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits_copy(jdet,1); n = r3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | dbckn >
                           hmatel = -h2a_voov(a,n,j,d)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits_copy(jdet,1); n = r3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | dackn >
                           hmatel = h2a_voov(b,n,j,d)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (ac)(jk), (-1)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits_copy(jdet,1); n = r3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | dabkn >
                           hmatel = -h2a_voov(c,n,j,d)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-1,nua-1/), (/2,noa/), nua, nua, noa)
                  call sort3(r3a_excits_copy, r3a_amps_copy, loc_arr, idx_table3, (/1,2,5/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits_copy,&
                  !$omp r3a_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a = 1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3a_excits_copy(jdet,3); m = r3a_excits_copy(jdet,4);
                        ! compute < abcjk | h2a(voov) | abfmk >
                        hmatel = h2a_voov(c,m,j,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                     end do
                     ! (ac), (-1)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = r3a_excits_copy(jdet,3); m = r3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | bcfmk >
                           hmatel = h2a_voov(a,m,j,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = r3a_excits_copy(jdet,3); m = r3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | acfmk >
                           hmatel = -h2a_voov(b,m,j,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = r3a_excits_copy(jdet,3); m = r3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | abfmj >
                           hmatel = -h2a_voov(c,m,k,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (ac)(jk), (-1)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = r3a_excits_copy(jdet,3); m = r3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | bcfmj >
                           hmatel = -h2a_voov(a,m,k,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = r3a_excits_copy(jdet,3); m = r3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | acfmj >
                           hmatel = h2a_voov(b,m,k,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-2,nua/), (/2,noa/), nua, nua, noa)
                  call sort3(r3a_excits_copy, r3a_amps_copy, loc_arr, idx_table3, (/1,3,5/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits_copy,&
                  !$omp r3a_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a = 1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3a_excits_copy(jdet,2); m = r3a_excits_copy(jdet,4);
                        ! compute < abcjk | h2a(voov) | aecmk >
                        hmatel = h2a_voov(b,m,j,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits_copy(jdet,2); m = r3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | becmk >
                           hmatel = -h2a_voov(a,m,j,e)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits_copy(jdet,2); m = r3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | aebmk >
                           hmatel = -h2a_voov(c,m,j,e)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits_copy(jdet,2); m = r3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | aecmj >
                           hmatel = -h2a_voov(b,m,k,e)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits_copy(jdet,2); m = r3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | becmj >
                           hmatel = h2a_voov(a,m,k,e)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3a_excits_copy(jdet,2); m = r3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | aebmj >
                           hmatel = h2a_voov(c,m,k,e)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,5) !!!
                  call get_index_table3(idx_table3, (/2,nua-1/), (/-1,nua/), (/2,noa/), nua, nua, noa)
                  call sort3(r3a_excits_copy, r3a_amps_copy, loc_arr, idx_table3, (/2,3,5/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits_copy,&
                  !$omp r3a_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a = 1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(b,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3a_excits_copy(jdet,1); m = r3a_excits_copy(jdet,4);
                        ! compute < abcjk | h2a(voov) | dbcmk >
                        hmatel = h2a_voov(a,m,j,d)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits_copy(jdet,1); m = r3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | dacmk >
                           hmatel = -h2a_voov(b,m,j,d)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (ac), (-1)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits_copy(jdet,1); m = r3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | dabmk >
                           hmatel = h2a_voov(c,m,j,d)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits_copy(jdet,1); m = r3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | dbcmj >
                           hmatel = -h2a_voov(a,m,k,d)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits_copy(jdet,1); m = r3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | dacmj >
                           hmatel = h2a_voov(b,m,k,d)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                     ! (ac)(jk), (-1)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3a_excits_copy(jdet,1); m = r3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | dabmj >
                           hmatel = -h2a_voov(c,m,k,d)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3a_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 6: A(c/ab)A(jk) h2b(cmke)*r3b(abejm)
                  ! allocate and copy over r3b arrays
                  allocate(amps_buff(n3aab),excits_buff(n3aab,5))
                  amps_buff(:) = r3b_amps(:)
                  excits_buff(:,:) = r3b_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = nua*(nua - 1)/2*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,noa))
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/-1,nua/), (/1,noa/), nua, nua, noa)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,2,4/), nua, nua, noa, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits_copy,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a = 1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                        ! compute < abcjk | h2a(voov) | abfjn >
                        hmatel = h2b_voov(c,n,k,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     ! (ac), (-1)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                           ! compute < abcjk | h2a(voov) | bcfjn >
                           hmatel = h2b_voov(a,n,k,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                           ! compute < abcjk | h2a(voov) | acfjn >
                           hmatel = -h2b_voov(b,n,k,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                           ! compute < abcjk | h2a(voov) | abfkn >
                           hmatel = -h2b_voov(c,n,j,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ac)(jk), (-1)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                           ! compute < abcjk | h2a(voov) | bcfkn >
                           hmatel = -h2b_voov(a,n,j,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                           ! compute < abcjk | h2a(voov) | acfkn >
                           hmatel = h2b_voov(b,n,j,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3,excits_buff,amps_buff)

                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits_copy,&
                  !$omp t2a,r2a,&
                  !$omp h2a_vvov,h2a_vooo,&
                  !$omp x2a_voo,x2a_vvv,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a = 1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                      ! zero out value
                      res_mm23 = 0.0d0
                      do m = 1,noa
                         ! -A(c/ab) h2a(cmkj)*r2a(abm)
                         res_mm23 = res_mm23 - h2a_vooo(c,m,k,j) * r2a(a,b,m) ! (1)
                         res_mm23 = res_mm23 + h2a_vooo(a,m,k,j) * r2a(c,b,m) ! (ac)
                         res_mm23 = res_mm23 + h2a_vooo(b,m,k,j) * r2a(a,c,m) ! (bc)
                         ! -A(a/bc)A(jk) x2a_voo(amj)*t2a(bcmk)
                         res_mm23 = res_mm23 - x2a_voo(a,m,j) * t2a(b,c,m,k) ! (1)
                         res_mm23 = res_mm23 + x2a_voo(b,m,j) * t2a(a,c,m,k) ! (ab)
                         res_mm23 = res_mm23 + x2a_voo(c,m,j) * t2a(b,a,m,k) ! (ac)
                         res_mm23 = res_mm23 + x2a_voo(a,m,k) * t2a(b,c,m,j) ! (jk)
                         res_mm23 = res_mm23 - x2a_voo(b,m,k) * t2a(a,c,m,j) ! (ab)(jk)
                         res_mm23 = res_mm23 - x2a_voo(c,m,k) * t2a(b,a,m,j) ! (ac)(jk)
                      end do
                      do e = 1,nua
                         ! A(a/bc)A(jk) h2a(cbke)*r2a(aej)
                         res_mm23 = res_mm23 + h2a_vvov(c,b,k,e) * r2a(a,e,j) ! (1)
                         res_mm23 = res_mm23 - h2a_vvov(c,a,k,e) * r2a(b,e,j) ! (ab)
                         res_mm23 = res_mm23 - h2a_vvov(a,b,k,e) * r2a(c,e,j) ! (ac)
                         res_mm23 = res_mm23 - h2a_vvov(c,b,j,e) * r2a(a,e,k) ! (jk)
                         res_mm23 = res_mm23 + h2a_vvov(c,a,j,e) * r2a(b,e,k) ! (ab)(jk)
                         res_mm23 = res_mm23 + h2a_vvov(a,b,j,e) * r2a(c,e,k) ! (ac)(jk)
                         ! A(c/ab) x2a_vvv(abe)*t2a(ecjk)
                         res_mm23 = res_mm23 + x2a_vvv(a,b,e) * t2a(e,c,j,k) ! (1)
                         res_mm23 = res_mm23 - x2a_vvv(c,b,e) * t2a(e,a,j,k) ! (ac)
                         res_mm23 = res_mm23 - x2a_vvv(a,c,e) * t2a(e,b,j,k) ! (bc)
                      end do
                      resid(a,b,c) = resid(a,b,c) + res_mm23
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate r3a array copies
                  deallocate(r3a_excits_copy,r3a_amps_copy)
                  ! antisymmetrize m(abc) block
                  do a = 1,nua
                     do b = a+1,nua
                        do c = b+1,nua
                           resid(a,c,b) = -resid(a,b,c)
                           resid(b,c,a) = resid(a,b,c)
                           resid(b,a,c) = -resid(a,b,c)
                           resid(c,a,b) = resid(a,b,c)
                           resid(c,b,a) = -resid(a,b,c)
                        end do
                     end do
                  end do
              end subroutine build_moments3a_jk

              subroutine build_moments3b_jk(resid,j,k,&
                                            r2a,r2b,&
                                            r3a_amps,r3a_excits,&
                                            r3b_amps,r3b_excits,&
                                            r3c_amps,r3c_excits,&
                                            t2a,t2b,&
                                            h1a_oo,h1a_vv,h1b_oo,h1b_vv,&
                                            h2a_vvvv,h2a_voov,h2a_vvov,&
                                            h2b_vvvv,h2b_oooo,h2b_voov,h2b_vovo,h2b_ovov,h2b_ovvo,&
                                            h2b_vooo,h2b_ovoo,h2b_vvov,h2b_vvvo,&
                                            h2c_voov,&
                                            x2a_voo,x2a_vvv,&
                                            x2b_voo,x2b_ovo,x2b_vvv,&
                                            n3aaa,n3aab,n3abb,&
                                            noa,nua,nob,nub)

                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa, n3aab, n3abb
                  ! Occupied block indices
                  integer, intent(in) :: j, k
                  ! Input R and T arrays
                  real(kind=8), intent(in) :: r2a(nua,nua,noa), t2a(nua,nua,noa,noa)
                  real(kind=8), intent(in) :: r2b(nua,nub,nob), t2b(nua,nub,noa,nob)
                  integer, intent(in) :: r3a_excits(n3aaa,5), r3b_excits(n3aab,5), r3c_excits(n3abb,5)
                  real(kind=8), intent(in) :: r3a_amps(n3aaa), r3b_amps(n3aab), r3c_amps(n3abb)
                  ! Input H  and X arrays
                  real(kind=8), intent(in) :: h1a_oo(noa,noa)
                  real(kind=8), intent(in) :: h1a_vv(nua,nua)
                  real(kind=8), intent(in) :: h1b_oo(nob,nob)
                  real(kind=8), intent(in) :: h1b_vv(nub,nub)
                  real(kind=8), intent(in) :: h2a_vvvv(nua,nua,nua,nua)
                  real(kind=8), intent(in) :: h2a_voov(nua,noa,noa,nua)
                  real(kind=8), intent(in) :: h2a_vvov(nua,nua,noa,nua)
                  real(kind=8), intent(in) :: h2b_vvvv(nua,nub,nua,nub)
                  real(kind=8), intent(in) :: h2b_oooo(noa,nob,noa,nob)
                  real(kind=8), intent(in) :: h2b_voov(nua,nob,noa,nub)
                  real(kind=8), intent(in) :: h2b_vovo(nua,nob,nua,nob)
                  real(kind=8), intent(in) :: h2b_ovov(noa,nub,noa,nub)
                  real(kind=8), intent(in) :: h2b_ovvo(noa,nub,nua,nob)
                  real(kind=8), intent(in) :: h2b_vooo(nua,nob,noa,nob)
                  real(kind=8), intent(in) :: h2b_ovoo(noa,nub,noa,nob)
                  real(kind=8), intent(in) :: h2b_vvov(nua,nub,noa,nub)
                  real(kind=8), intent(in) :: h2b_vvvo(nua,nub,nua,nob)
                  real(kind=8), intent(in) :: h2c_voov(nub,nob,nob,nub)
                  real(kind=8), intent(in) :: x2a_voo(nua,noa,noa)
                  real(kind=8), intent(in) :: x2a_vvv(nua,nua,nua)
                  real(kind=8), intent(in) :: x2b_voo(nua,nob,nob)
                  real(kind=8), intent(in) :: x2b_ovo(noa,nub,nob)
                  real(kind=8), intent(in) :: x2b_vvv(nua,nub,nub)
                  ! Output and Inout variables
                  real(kind=8), intent(out) :: resid(nua,nua,nub)
                  ! Local variables
                  integer, allocatable :: excits_buff(:,:), r3b_excits_copy(:,:)
                  real(kind=8), allocatable :: amps_buff(:), r3b_amps_copy(:)
                  integer, allocatable :: idx_table4(:,:,:,:)
                  integer, allocatable :: idx_table3(:,:,:)
                  integer, allocatable :: idx_table2(:,:)
                  integer, allocatable :: loc_arr(:,:)
                  real(kind=8) :: r_amp, hmatel, res_mm23
                  integer :: a, b, c, d, i, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc
                  ! Q space array
                  logical(kind=1) :: qspace(nua,nua,nub)
                  
                  ! Zero the container that holds H*R
                  resid = 0.0d0
                  
                  ! copy over r3b_amps and r3b_excits
                  allocate(r3b_amps_copy(n3aab),r3b_excits_copy(n3aab,5))
                  r3b_amps_copy(:) = r3b_amps(:)
                  r3b_excits_copy(:,:) = r3b_excits(:,:)
                  
                  ! reorder r3b into (j,k) order
                  nloc = noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table2(noa,nob))
                  call get_index_table2(idx_table2, (/1,noa/), (/1,nob/), noa, nob)
                  call sort2(r3b_excits_copy, r3b_amps_copy, loc_arr, idx_table2, (/4,5/), noa, nob, nloc, n3aab)
                  ! Construct Q space for block (j,k)
                  qspace = .true.
                  idx = idx_table2(j,k)
                  if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        a = r3b_excits_copy(jdet,1); b = r3b_excits_copy(jdet,2); c = r3b_excits_copy(jdet,3);
                        qspace(a,b,c) = .false.
                     end do
                  end if
                  deallocate(loc_arr,idx_table2)

                  !!! diagram 1: -h1a(mj)*r3b(abcmk)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2*nub * nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nua,nub,nob))
                  !!! SB: (1,2,3,5) !!!
                  call get_index_table4(idx_table4, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/1,nob/), nua, nua, nub, nob)
                  call sort4(r3b_excits_copy, r3b_amps_copy, loc_arr, idx_table4, (/1,2,3,5/), nua, nua, nub, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits_copy,&
                  !$omp r3b_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_oo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(a,b,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3b_excits_copy(jdet,4);
                        ! compute < abcjk | h1a(oo) | abcmk >
                        hmatel = -h1a_oo(m,j)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3b_amps_copy(jdet)
                     end do
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 2: -h1b(mk)*r3b(abcjm)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2*nub * noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nua,nub,noa))
                  !!! SB: (1,2,3,4) !!!
                  call get_index_table4(idx_table4, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/1,noa/), nua, nua, nub, noa)
                  call sort4(r3b_excits_copy, r3b_amps_copy, loc_arr, idx_table4, (/1,2,3,4/), nua, nua, nub, noa, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits_copy,&
                  !$omp r3b_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1B_oo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(a,b,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        n = r3b_excits_copy(jdet,5);
                        ! compute < abcjk | h1b(oo) | abcjn >
                        hmatel = -h1b_oo(n,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3b_amps_copy(jdet)
                     end do
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 3: h1b(ce)*r3b(abejk)
                  ! allocate new sorting arrays
                  nloc = nua*(nua - 1)/2 * noa * nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nua,noa,nob))
                  !!! SB: (1,2,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nua-1/), (/-1,nua/), (/1,noa/), (/1,nob/), nua, nua, noa, nob)
                  call sort4(r3b_excits_copy, r3b_amps_copy, loc_arr, idx_table4, (/1,2,4,5/), nua, nua, noa, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits_copy,&
                  !$omp r3b_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1B_vv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(a,b,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3b_excits_copy(jdet,3);
                        ! compute < abcjk | h1b(vv) | abfjk >
                        hmatel = h1b_vv(c,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3b_amps_copy(jdet)
                     end do
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 4: h1a(be)*r3b(aecjk)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*nub*noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nub,noa,nob))
                  !!! SB: (1,3,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nua-1/), (/1,nub/), (/1,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(r3b_excits_copy, r3b_amps_copy, loc_arr, idx_table4, (/1,3,4,5/), nua, nub, noa, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits_copy,&
                  !$omp r3b_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_vv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(a,c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3b_excits_copy(jdet,2);
                        ! compute < abcjk | h1a(vv) | aecjk >
                        hmatel = h1a_vv(b,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3b_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = r3b_excits_copy(jdet,2);
                           ! compute < abcjk | h1a(vv) | becjk >
                           hmatel = -h1a_vv(a,e)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3b_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4,5) !!!
                  call get_index_table4(idx_table4, (/2,nua/), (/1,nub/), (/1,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(r3b_excits_copy, r3b_amps_copy, loc_arr, idx_table4, (/2,3,4,5/), nua, nub, noa, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits_copy,&
                  !$omp r3b_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_vv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3b_excits_copy(jdet,1);
                        ! compute < abcjk | h1a(vv) | dbcjk >
                        hmatel = h1a_vv(a,d)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3b_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table4(a,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = r3b_excits_copy(jdet,1);
                           ! compute < abcjk | h1a(vv) | dacjk >
                           hmatel = -h1a_vv(b,d)
                           resid(a,b,c) = resid(a,b,c) + hmatel * r3b_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 5: h2b(mnjk)*r3b(abcmn)
                  ! allocate new sorting arrays
                  nloc = nua*(nua - 1)/2*nub
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,nub))
                  !!! SB: (1,2,3) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/-1,nua/), (/1,nub/), nua, nua, nub)
                  call sort3(r3b_excits_copy, r3b_amps_copy, loc_arr, idx_table3, (/1,2,3/), nua, nua, nub, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits_copy,&
                  !$omp r3b_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_oooo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3b_excits_copy(jdet,4); n = r3b_excits_copy(jdet,5);
                        ! compute < abcjk | h2b(oooo) | abcmn >
                        hmatel = h2b_oooo(m,n,j,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3b_amps_copy(jdet)
                     end do
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 6: h2b(mcek)*r3a(abejm)
                  ! allocate and copy over r3a arrays
                  allocate(amps_buff(n3aaa),excits_buff(n3aaa,5))
                  amps_buff(:) = r3a_amps(:)
                  excits_buff(:,:) = r3a_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*(nua - 2)/2*(noa - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,noa))
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-1,nua-1/), (/1,noa-1/), nua, nua, noa)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,2,4/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits_copy,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                        ! compute < abc~jk~ | h2b(ovvo) | abfjn >
                        hmatel = h2b_ovvo(n,c,f,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-1,nua-1/), (/2,noa/), nua, nua, noa)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,2,5/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits_copy,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,3); m = excits_buff(jdet,4);
                        ! compute < abc~jk~ | h2b(ovvo) | abfmj >
                        hmatel = -h2b_ovvo(m,c,f,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-2,nua/), (/1,noa-1/), nua, nua, noa)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,3,4/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits_copy,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); n = excits_buff(jdet,5);
                        ! compute < abc~jk~ | h2b(ovvo) | aebjn >
                        hmatel = -h2b_ovvo(n,c,e,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-2,nua/), (/2,noa/), nua, nua, noa)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,3,5/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits_copy,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < abc~jk~ | h2b(ovvo) | aebmj >
                        hmatel = h2b_ovvo(m,c,e,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4) !!!
                  call get_index_table3(idx_table3, (/2,nua-1/), (/-1,nua/), (/1,noa-1/), nua, nua, noa)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/2,3,4/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits_copy,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = excits_buff(jdet,1); n = excits_buff(jdet,5);
                        ! compute < abc~jk~ | h2b(ovvo) | dabjn >
                        hmatel = h2b_ovvo(n,c,d,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,5) !!!
                  call get_index_table3(idx_table3, (/2,nua-1/), (/-1,nua/), (/2,noa/), nua, nua, noa)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/2,3,5/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits_copy,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < abc~jk~ | h2b(ovvo) | dabmj >
                        hmatel = -h2b_ovvo(m,c,d,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3,excits_buff,amps_buff)

                  !!! diagram 7: h2c(cmke)*r3b(abejm)
                  ! allocate new sorting arrays
                  nloc = nua*(nua - 1)/2*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,noa))
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/-1,nua/), (/1,noa/), nua, nua, noa)
                  call sort3(r3b_excits_copy, r3b_amps_copy, loc_arr, idx_table3, (/1,2,4/), nua, nua, noa, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits_copy,&
                  !$omp r3b_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3b_excits_copy(jdet,3); n = r3b_excits_copy(jdet,5);
                        ! compute < abcjk | h2c(voov) | abfjn >
                        hmatel = h2c_voov(c,n,k,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3b_amps_copy(jdet)
                     end do
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 8: h2a(bmje)*r3b(aecmk)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*nub*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nub,nob))
                  !!! SB: (1,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/1,nub/), (/1,nob/), nua, nub, nob)
                  call sort3(r3b_excits_copy, r3b_amps_copy, loc_arr, idx_table3, (/1,3,5/), nua, nub, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits_copy,&
                  !$omp r3b_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3b_excits_copy(jdet,2); m = r3b_excits_copy(jdet,4);
                        ! compute < abc~jk~ | h2a(voov) | aec~mk~ >
                        hmatel = h2a_voov(b,m,j,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3b_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3b_excits_copy(jdet,2); m = r3b_excits_copy(jdet,4);
                        ! compute < abc~jk~ | h2a(voov) | bec~mk~ >
                        hmatel = -h2a_voov(a,m,j,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3b_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,5) !!!
                  call get_index_table3(idx_table3, (/2,nua/), (/1,nub/), (/1,nob/), nua, nub, nob)
                  call sort3(r3b_excits_copy, r3b_amps_copy, loc_arr, idx_table3, (/2,3,5/), nua, nub, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits_copy,&
                  !$omp r3b_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(b,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3b_excits_copy(jdet,1); m = r3b_excits_copy(jdet,4);
                        ! compute < abc~jk~ | h2a(voov) | dbc~mk~ >
                        hmatel = h2a_voov(a,m,j,d)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3b_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3b_excits_copy(jdet,1); m = r3b_excits_copy(jdet,4);
                        ! compute < abc~jk~ | h2a(voov) | dac~mk~ >
                        hmatel = -h2a_voov(b,m,j,d)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3b_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 9: -A(ab) h2b(bmek)*r3b(aecjm)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*nub*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nub,noa))
                  !!! SB: (1,3,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/1,nub/), (/1,noa/), nua, nub, noa)
                  call sort3(r3b_excits_copy, r3b_amps_copy, loc_arr, idx_table3, (/1,3,4/), nua, nub, noa, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits_copy,&
                  !$omp r3b_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vovo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3b_excits_copy(jdet,2); n = r3b_excits_copy(jdet,5);
                        ! compute < abc~jk~ | h2b(vovo) | aec~jn~ >
                        hmatel = -h2b_vovo(b,n,e,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3b_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3b_excits_copy(jdet,2); n = r3b_excits_copy(jdet,5);
                        ! compute < abc~jk~ | h2b(vovo) | bec~jn~ >
                        hmatel = h2b_vovo(a,n,e,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3b_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4) !!!
                  call get_index_table3(idx_table3, (/2,nua/), (/1,nub/), (/1,noa/), nua, nub, noa)
                  call sort3(r3b_excits_copy, r3b_amps_copy, loc_arr, idx_table3, (/2,3,4/), nua, nub, noa, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits_copy,&
                  !$omp r3b_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vovo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(b,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3b_excits_copy(jdet,1); n = r3b_excits_copy(jdet,5);
                        ! compute < abc~jk~ | h2b(vovo) | bec~jn~ >
                        hmatel = -h2b_vovo(a,n,d,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3b_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3b_excits_copy(jdet,1); n = r3b_excits_copy(jdet,5);
                        ! compute < abc~jk~ | h2b(vovo) | aec~jn~ >
                        hmatel = h2b_vovo(b,n,d,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3b_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 10: -h2b(mcje)*r3b(abemk)
                  ! allocate new sorting arrays
                  nloc = nua*(nua - 1)/2 * nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,nob))
                  !!! SB: (1,2,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/-1,nua/), (/1,nob/), nua, nua, nob)
                  call sort3(r3b_excits_copy, r3b_amps_copy, loc_arr, idx_table3, (/1,2,5/), nua, nua, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits_copy,&
                  !$omp r3b_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3b_excits_copy(jdet,3); m = r3b_excits_copy(jdet,4);
                        ! compute < abc~jk~ | h2b(vovo) | abf~mk~ >
                        hmatel = -h2b_ovov(m,c,j,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3b_amps_copy(jdet)
                     end do
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 11: A(ab) h2b(bmje)*r3c(aecmk)
                  ! allocate and copy over r3c arrays
                  allocate(amps_buff(n3abb),excits_buff(n3abb,5))
                  amps_buff(:) = r3c_amps(:)
                  excits_buff(:,:) = r3c_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = nua*(nub - 1)*(nob - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nub,nob))
                  !!! SB: (1,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/2,nub/), (/2,nob/), nua, nub, nob)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,3,5/), nua, nub, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits_copy,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < abc~jk~ | h2b(voov) | ae~c~m~k~ >
                        hmatel = h2b_voov(b,m,j,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < abc~jk~ | h2b(voov) | be~c~m~k~ >
                        hmatel = -h2b_voov(a,m,j,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,4) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/2,nub/), (/1,nob-1/), nua, nub, nob)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,3,4/), nua, nub, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits_copy,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); n = excits_buff(jdet,5);
                        ! compute < abc~jk~ | h2b(voov) | ae~c~k~n~ >
                        hmatel = -h2b_voov(b,n,j,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); n = excits_buff(jdet,5);
                        ! compute < abc~jk~ | h2b(voov) | be~c~k~n~ >
                        hmatel = h2b_voov(a,n,j,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,5) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/1,nub-1/), (/2,nob/), nua, nub, nob)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,2,5/), nua, nub, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits_copy,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,3); m = excits_buff(jdet,4);
                        ! compute < abc~jk~ | h2b(voov) | ac~f~m~k~ >
                        hmatel = -h2b_voov(b,m,j,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,3); m = excits_buff(jdet,4);
                        ! compute < abc~jk~ | h2b(voov) | bc~f~m~k~ >
                        hmatel = h2b_voov(a,m,j,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/1,nub-1/), (/1,nob-1/), nua, nub, nob)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,2,4/), nua, nub, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits_copy,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                        ! compute < abc~jk~ | h2b(voov) | ac~f~k~n~ >
                        hmatel = h2b_voov(b,n,j,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                        ! compute < abc~jk~ | h2b(voov) | bc~f~k~n~ >
                        hmatel = -h2b_voov(a,n,j,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3,excits_buff,amps_buff)

                  !!! diagram 12: 1/2 h2a(abef)*r3b(efcjk)
                  ! allocate new sorting arrays
                  nloc = nub*noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nub,noa,nob))
                  !!! SB: (3,4,5) !!!
                  call get_index_table3(idx_table3, (/1,nub/), (/1,noa/), (/1,nob/), nub, noa, nob)
                  call sort3(r3b_excits_copy, r3b_amps_copy, loc_arr, idx_table3, (/3,4,5/), nub, noa, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits_copy,&
                  !$omp r3b_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_vvvv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3b_excits_copy(jdet,1); e = r3b_excits_copy(jdet,2);
                        ! compute < abc~jk~ | h2a(vvvv) | dfc~jk~ >
                        !hmatel = h2a_vvvv(a,b,d,e)
                        hmatel = h2a_vvvv(d,e,a,b)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3b_amps_copy(jdet)
                     end do
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 13: A(ab) h2b(bcef)*r3b(aefjk)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,noa,nob))
                  !!! SB: (1,4,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/1,noa/), (/1,nob/), nua, noa, nob)
                  call sort3(r3b_excits_copy, r3b_amps_copy, loc_arr, idx_table3, (/1,4,5/), nua, noa, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits_copy,&
                  !$omp r3b_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vvvv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3b_excits_copy(jdet,2); f = r3b_excits_copy(jdet,3);
                        ! compute < abc~jk~ | h2b(vvvv) | aef~jk~ >
                        !hmatel = h2b_vvvv(b,c,e,f)
                        hmatel = h2b_vvvv(e,f,b,c)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3b_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3b_excits_copy(jdet,2); f = r3b_excits_copy(jdet,3);
                        ! compute < abc~jk~ | h2b(vvvv) | bef~jk~ >
                        !hmatel = -h2b_vvvv(a,c,e,f)
                        hmatel = -h2b_vvvv(e,f,a,c)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3b_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,4,5) !!!
                  call get_index_table3(idx_table3, (/2,nua/), (/1,noa/), (/1,nob/), nua, noa, nob)
                  call sort3(r3b_excits_copy, r3b_amps_copy, loc_arr, idx_table3, (/2,4,5/), nua, noa, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits_copy,&
                  !$omp r3b_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vvvv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(b,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3b_excits_copy(jdet,1); f = r3b_excits_copy(jdet,3);
                        ! compute < abc~jk~ | h2b(vvvv) | dbf~jk~ >
                        !hmatel = h2b_vvvv(a,c,d,f)
                        hmatel = h2b_vvvv(d,f,a,c)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3b_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(a,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3b_excits_copy(jdet,1); f = r3b_excits_copy(jdet,3);
                        ! compute < abc~jk~ | h2b(vvvv) | daf~jk~ >
                        !hmatel = -h2b_vvvv(b,c,d,f)
                        hmatel = -h2b_vvvv(d,f,b,c)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3b_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3b_excits_copy,&
                  !$omp t2a,t2b,r2a,r2b,&
                  !$omp h2a_vvov,h2b_vooo,h2b_ovoo,h2b_vvov,h2b_vvvo,&
                  !$omp x2a_voo,x2a_vvv,x2b_voo,x2b_ovo,x2b_vvv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                      ! zero out value
                      res_mm23 = 0.0d0
                      do m = 1,noa
                         ! -h2b(mcjk)*r2a(abm)
                         res_mm23 = res_mm23 - h2b_ovoo(m,c,j,k) * r2a(a,b,m) ! (1)
                         ! -x2b_ovo(mck)*t2a(abmj)
                         res_mm23 = res_mm23 - x2b_ovo(m,c,k) * t2a(a,b,m,j) ! (1)
                         ! -A(ab) x2a_voo(amj)*t2b(bcmk)
                         res_mm23 = res_mm23 - x2a_voo(a,m,j) * t2b(b,c,m,k) ! (1)
                         res_mm23 = res_mm23 + x2a_voo(b,m,j) * t2b(a,c,m,k) ! (ab)
                      end do
                      do m = 1,nob
                         ! -A(ab) h2b(bmjk)*r2b(acm)
                         res_mm23 = res_mm23 - h2b_vooo(b,m,j,k) * r2b(a,c,m) ! (1)
                         res_mm23 = res_mm23 + h2b_vooo(a,m,j,k) * r2b(b,c,m) ! (ab)
                         ! -A(ab) x2b_voo(amk)*t2b(bcjm)
                         res_mm23 = res_mm23 - x2b_voo(a,m,k) * t2b(b,c,j,m) ! (1)
                         res_mm23 = res_mm23 + x2b_voo(b,m,k) * t2b(a,c,j,m) ! (ab)
                      end do
                      do e = 1,nua
                         ! A(ab) h2b(bcek)*r2a(aej)
                         res_mm23 = res_mm23 + h2b_vvvo(b,c,e,k) * r2a(a,e,j) ! (1)
                         res_mm23 = res_mm23 - h2b_vvvo(a,c,e,k) * r2a(b,e,j) ! (ab)
                         ! h2a(baje)*r2b(eck)
                         res_mm23 = res_mm23 + h2a_vvov(b,a,j,e) * r2b(e,c,k) ! (1)
                         ! x2a_vvv(abe)*t2b(ecjk)
                         res_mm23 = res_mm23 + x2a_vvv(a,b,e) * t2b(e,c,j,k) ! (1)
                      end do
                      do e = 1,nub
                         ! A(ab) h2b(bcje)*r2b(aek)
                         res_mm23 = res_mm23 + h2b_vvov(b,c,j,e) * r2b(a,e,k) ! (1)
                         res_mm23 = res_mm23 - h2b_vvov(a,c,j,e) * r2b(b,e,k) ! (ab)
                         ! A(ab) x2b_vvv(ace)*t2b(bejk)
                         res_mm23 = res_mm23 + x2b_vvv(a,c,e) * t2b(b,e,j,k) ! (1)
                         res_mm23 = res_mm23 - x2b_vvv(b,c,e) * t2b(a,e,j,k) ! (ab)
                      end do
                      resid(a,b,c) = resid(a,b,c) + res_mm23
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate r3b array copies
                  deallocate(r3b_excits_copy,r3b_amps_copy)
                  ! antisymmetrize m(abc)
                  do a=1,nua
                     do b=a+1,nua
                        do c=1,nub
                           resid(b,a,c) = -resid(a,b,c)
                        end do
                     end do
                  end do
              end subroutine build_moments3b_jk

              subroutine build_moments3c_jk(resid,j,k,&
                                            r2b,&
                                            r3b_amps,r3b_excits,&
                                            r3c_amps,r3c_excits,&
                                            t2b,t2c,&
                                            h1a_vv,h1b_oo,h1b_vv,&
                                            h2b_vvvv,h2b_vovo,h2b_ovvo,h2b_vvvo,&
                                            h2c_vvvv,h2c_oooo,h2c_voov,h2c_vooo,h2c_vvov,&
                                            x2b_voo,x2b_ovo,x2b_vvv,&
                                            n3aab,n3abb,&
                                            noa,nua,nob,nub)

                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab, n3abb
                  ! Occupied block indices
                  integer, intent(in) :: j, k
                  ! Input R and T arrays
                  real(kind=8), intent(in) :: t2b(nua,nub,noa,nob), t2c(nub,nub,nob,nob)
                  real(kind=8), intent(in) :: r2b(nua,nub,nob)
                  integer, intent(in) :: r3b_excits(n3aab,5), r3c_excits(n3abb,5)
                  real(kind=8), intent(in) :: r3b_amps(n3aab), r3c_amps(n3abb)
                  ! Input H  and X arrays
                  real(kind=8), intent(in) :: h1a_vv(nua,nua)
                  real(kind=8), intent(in) :: h1b_oo(nob,nob)
                  real(kind=8), intent(in) :: h1b_vv(nub,nub)
                  real(kind=8), intent(in) :: h2b_vvvv(nua,nub,nua,nub)
                  real(kind=8), intent(in) :: h2b_vovo(nua,nob,nua,nob)
                  real(kind=8), intent(in) :: h2b_ovvo(noa,nub,nua,nob)
                  real(kind=8), intent(in) :: h2b_vvvo(nua,nub,nua,nob)
                  real(kind=8), intent(in) :: h2c_vvvv(nub,nub,nub,nub)
                  real(kind=8), intent(in) :: h2c_oooo(nob,nob,nob,nob)
                  real(kind=8), intent(in) :: h2c_voov(nub,nob,nob,nub)
                  real(kind=8), intent(in) :: h2c_vooo(nub,nob,nob,nob)
                  real(kind=8), intent(in) :: h2c_vvov(nub,nub,nob,nub)
                  real(kind=8), intent(in) :: x2b_voo(nua,nob,nob)
                  real(kind=8), intent(in) :: x2b_ovo(noa,nub,nob)
                  real(kind=8), intent(in) :: x2b_vvv(nua,nub,nub)
                  ! Output and Inout variables
                  real(kind=8), intent(out) :: resid(nua,nub,nub)
                  ! Local variables
                  integer, allocatable :: excits_buff(:,:), r3c_excits_copy(:,:)
                  real(kind=8), allocatable :: amps_buff(:), r3c_amps_copy(:)
                  integer, allocatable :: idx_table4(:,:,:,:)
                  integer, allocatable :: idx_table3(:,:,:)
                  integer, allocatable :: idx_table2(:,:)
                  integer, allocatable :: loc_arr(:,:)
                  real(kind=8) :: r_amp, hmatel, res_mm23
                  integer :: a, b, c, d, i, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc
                  ! Q space array
                  logical(kind=1) :: qspace(nua,nub,nub)
                  
                  ! Zero the container that holds H*R
                  resid = 0.0d0
                  
                  ! copy over r3c_amps and r3c_excits
                  allocate(r3c_amps_copy(n3abb),r3c_excits_copy(n3abb,5))
                  r3c_amps_copy(:) = r3c_amps(:)
                  r3c_excits_copy(:,:) = r3c_excits(:,:)
                  
                  ! reorder r3c into (j,k) order
                  nloc = noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table2(nob,nob))
                  call get_index_table2(idx_table2, (/1,nob-1/), (/-1,nob/), nob, nob)
                  call sort2(r3c_excits_copy, r3c_amps_copy, loc_arr, idx_table2, (/4,5/), nob, nob, nloc, n3abb)
                  ! Construct Q space for block (j,k)
                  qspace = .true.
                  idx = idx_table2(j,k)
                  if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        a = r3c_excits_copy(jdet,1); b = r3c_excits_copy(jdet,2); c = r3c_excits_copy(jdet,3);
                        qspace(a,b,c) = .false.
                     end do
                  end if
                  deallocate(loc_arr,idx_table2)

                  !!! diagram 1: -A(jk) h1b(mj)*r3c(abcmk)
                  ! allocate new sorting arrays
                  nloc = nua*nub*(nub - 1)/2*(nob - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nub,nub,nua,nob))
                  !!! SB: (2,3,1,5) !!!
                  call get_index_table4(idx_table4, (/1,nub-1/), (/-1,nub/), (/1,nua/), (/2,nob/), nub, nub, nua, nob)
                  call sort4(r3c_excits_copy, r3c_amps_copy, loc_arr, idx_table4, (/2,3,1,5/), nub, nub, nua, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits_copy,&
                  !$omp r3c_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1B_oo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(b,c,a,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3c_excits_copy(jdet,4);
                        ! compute < ab~c~j~k~ | h1b(oo) | ab~c~m~k~ >
                        hmatel = -h1b_oo(m,j)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     ! (jk)
                     idx = idx_table4(b,c,a,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3c_excits_copy(jdet,4);
                        ! compute < ab~c~j~k~ | h1b(oo) | ab~c~m~j~ >
                        hmatel = h1b_oo(m,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,1,4) !!!
                  call get_index_table4(idx_table4, (/1,nub-1/), (/-1,nub/), (/1,nua/), (/1,nob-1/), nub, nub, nua, nob)
                  call sort4(r3c_excits_copy, r3c_amps_copy, loc_arr, idx_table4, (/2,3,1,4/), nub, nub, nua, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits_copy,&
                  !$omp r3c_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1B_oo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(b,c,a,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        n = r3c_excits_copy(jdet,5);
                        ! compute < ab~c~j~k~ | h1b(oo) | ab~c~j~n~ >
                        hmatel = -h1b_oo(n,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     ! (jk)
                     idx = idx_table4(b,c,a,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        n = r3c_excits_copy(jdet,5);
                        ! compute < ab~c~j~k~ | h1b(oo) | ab~c~k~n~ >
                        hmatel = h1b_oo(n,j)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 2: A(bc) h1b(be)*r3c(aecjk)
                  ! allocate new sorting arrays
                  nloc = nua*(nub - 1)*nob*(nob - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nob,nob,nua,nub))
                  !!! SB: (4,5,1,3) !!!
                  call get_index_table4(idx_table4, (/1,nob-1/), (/-1,nob/), (/1,nua/), (/2,nub/), nob, nob, nua, nub)
                  call sort4(r3c_excits_copy, r3c_amps_copy, loc_arr, idx_table4, (/4,5,1,3/), nob, nob, nua, nub, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits_copy,&
                  !$omp r3c_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1B_vv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(j,k,a,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits_copy(jdet,2);
                        ! compute < ab~c~j~k~ | h1b(vv) | ae~c~j~k~ >
                        hmatel = h1b_vv(b,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     ! (bc)
                     idx = idx_table4(j,k,a,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits_copy(jdet,2);
                        ! compute < ab~c~j~k~ | h1b(vv) | ae~b~j~k~ >
                        hmatel = -h1b_vv(c,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,1,2) !!!
                  call get_index_table4(idx_table4, (/1,nob-1/), (/-1,nob/), (/1,nua/), (/1,nub-1/), nob, nob, nua, nub)
                  call sort4(r3c_excits_copy, r3c_amps_copy, loc_arr, idx_table4, (/4,5,1,2/), nob, nob, nua, nub, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits_copy,&
                  !$omp r3c_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1B_vv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(j,k,a,b)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits_copy(jdet,3);
                        ! compute < ab~c~j~k~ | h1b(vv) | ab~f~j~k~ >
                        hmatel = h1b_vv(c,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     ! (bc)
                     idx = idx_table4(j,k,a,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits_copy(jdet,3);
                        ! compute < ab~c~j~k~ | h1b(vv) | ac~f~j~k~ >
                        hmatel = -h1b_vv(b,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)
                 
                  !!! diagram 3: h1a(ae)*r3c(ebcjk)
                  ! allocate new sorting arrays
                  nloc = nub*(nub - 1)/2*nob*(nob - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nub,nub,nob,nob))
                  !!! SB: (2,3,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nub-1/), (/-1,nub/), (/1,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(r3c_excits_copy, r3c_amps_copy, loc_arr, idx_table4, (/2,3,4,5/), nub, nub, nob, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits_copy,&
                  !$omp r3c_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_vv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3c_excits_copy(jdet,1);
                        ! compute < ab~c~j~k~ | h1a(vv) | db~c~j~k~ >
                        hmatel = h1a_vv(a,d)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 4: h2c(mnjk)*r3c(abcmn)
                  ! allocate new sorting arrays
                  nloc = nua*nub*(nub - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nub,nub,nua))
                  !!! SB: (2,3,1) !!!
                  call get_index_table3(idx_table3, (/1,nub-1/), (/-1,nub/), (/1,nua/), nub, nub, nua)
                  call sort3(r3c_excits_copy, r3c_amps_copy, loc_arr, idx_table3, (/2,3,1/), nub, nub, nua, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits_copy,&
                  !$omp r3c_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2C_oooo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(b,c,a)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = r3c_excits_copy(jdet,4); n = r3c_excits_copy(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(oooo) | ab~c~m~n~ >
                        hmatel = h2c_oooo(m,n,j,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 5: A(bc)A(jk) h2b(mbej)*r3b(aecmk)
                  ! allocate and copy over r3b arrays
                  allocate(amps_buff(n3aab),excits_buff(n3aab,5))
                  amps_buff(:) = r3b_amps(:)
                  excits_buff(:,:) = r3b_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*nub*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nub,nob))
                  !!! SB: (1,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/1,nub/), (/1,nob/), nua, nub, nob)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,3,5/), nua, nub, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits_copy,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(ovvo) | aec~mk~ >
                        hmatel = h2b_ovvo(m,b,e,j)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(ovvo) | aeb~mk~ >
                        hmatel = -h2b_ovvo(m,c,e,j)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(ovvo) | aec~mj~ >
                        hmatel = -h2b_ovvo(m,b,e,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(ovvo) | aeb~mj~ >
                        hmatel = h2b_ovvo(m,c,e,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,5) !!!
                  call get_index_table3(idx_table3, (/2,nua/), (/1,nub/), (/1,nob/), nua, nub, nob)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/2,3,5/), nua, nub, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits_copy,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(ovvo) | dac~mk~ >
                        hmatel = -h2b_ovvo(m,b,d,j)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(ovvo) | dab~mk~ >
                        hmatel = h2b_ovvo(m,c,d,j)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(ovvo) | dac~mj~ >
                        hmatel = h2b_ovvo(m,b,d,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(ovvo) | dab~mj~ >
                        hmatel = -h2b_ovvo(m,c,d,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3,excits_buff,amps_buff)

                  !!! diagram 6: A(bc)A(jk) h2c(bmje) * r3c(aecmk)
                  ! allocate new sorting arrays
                  nloc = nua*(nub - 1)*(nob - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nub,nob))
                  !!! SB: (1,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/2,nub/), (/2,nob/), nua, nub, nob)
                  call sort3(r3c_excits_copy, r3c_amps_copy, loc_arr, idx_table3, (/1,3,5/), nua, nub, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits_copy,&
                  !$omp r3c_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits_copy(jdet,2); m = r3c_excits_copy(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~c~m~k~ >
                        hmatel = h2c_voov(b,m,j,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     ! (bc)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits_copy(jdet,2); m = r3c_excits_copy(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~b~m~k~ >
                        hmatel = -h2c_voov(c,m,j,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits_copy(jdet,2); m = r3c_excits_copy(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~c~m~j~ >
                        hmatel = -h2c_voov(b,m,k,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits_copy(jdet,2); m = r3c_excits_copy(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~b~m~j~ >
                        hmatel = h2c_voov(c,m,k,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,5) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/1,nub-1/), (/2,nob/), nua, nub, nob)
                  call sort3(r3c_excits_copy, r3c_amps_copy, loc_arr, idx_table3, (/1,2,5/), nua, nub, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits_copy,&
                  !$omp r3c_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits_copy(jdet,3); m = r3c_excits_copy(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ab~f~m~k~ >
                        hmatel = h2c_voov(c,m,j,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     ! (bc)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits_copy(jdet,3); m = r3c_excits_copy(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ac~f~m~k~ >
                        hmatel = -h2c_voov(b,m,j,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits_copy(jdet,3); m = r3c_excits_copy(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ab~f~m~j~ >
                        hmatel = -h2c_voov(c,m,k,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits_copy(jdet,3); m = r3c_excits_copy(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ac~f~m~j~ >
                        hmatel = h2c_voov(b,m,k,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,4) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/2,nub/), (/1,nob-1/), nua, nub, nob)
                  call sort3(r3c_excits_copy, r3c_amps_copy, loc_arr, idx_table3, (/1,3,4/), nua, nub, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits_copy,&
                  !$omp r3c_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits_copy(jdet,2); n = r3c_excits_copy(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~c~j~n~ >
                        hmatel = h2c_voov(b,n,k,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     ! (bc)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits_copy(jdet,2); n = r3c_excits_copy(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~b~j~n~ >
                        hmatel = -h2c_voov(c,n,k,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits_copy(jdet,2); n = r3c_excits_copy(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~c~k~n~ >
                        hmatel = -h2c_voov(b,n,j,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits_copy(jdet,2); n = r3c_excits_copy(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~b~k~n~ >
                        hmatel = h2c_voov(c,n,j,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/1,nub-1/), (/1,nob-1/), nua, nub, nob)
                  call sort3(r3c_excits_copy, r3c_amps_copy, loc_arr, idx_table3, (/1,2,4/), nua, nub, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits_copy,&
                  !$omp r3c_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits_copy(jdet,3); n = r3c_excits_copy(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ab~f~j~n~ >
                        hmatel = h2c_voov(c,n,k,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     ! (bc)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits_copy(jdet,3); n = r3c_excits_copy(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ac~f~j~n~ >
                        hmatel = -h2c_voov(b,n,k,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits_copy(jdet,3); n = r3c_excits_copy(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ab~f~k~n~ >
                        hmatel = -h2c_voov(c,n,j,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = r3c_excits_copy(jdet,3); n = r3c_excits_copy(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ac~f~k~n~ >
                        hmatel = h2c_voov(b,n,j,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 7: -A(jk) h2b(amej)*r3c(ebcmk)
                  ! allocate new sorting arrays
                  nloc = nub*(nub - 1)/2*(nob - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nub,nub,nob))
                  !!! SB: (2,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nub-1/), (/-1,nub/), (/2,nob/), nub, nub, nob)
                  call sort3(r3c_excits_copy, r3c_amps_copy, loc_arr, idx_table3, (/2,3,5/), nub, nub, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits_copy,&
                  !$omp r3c_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vovo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(b,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3c_excits_copy(jdet,1); m = r3c_excits_copy(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(vovo) | db~c~m~k~ >
                        hmatel = -h2b_vovo(a,m,d,j)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     ! (jk)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3c_excits_copy(jdet,1); m = r3c_excits_copy(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(vovo) | db~c~m~j~ >
                        hmatel = h2b_vovo(a,m,d,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4) !!!
                  call get_index_table3(idx_table3, (/1,nub-1/), (/-1,nub/), (/1,nob-1/), nub, nub, nob)
                  call sort3(r3c_excits_copy, r3c_amps_copy, loc_arr, idx_table3, (/2,3,4/), nub, nub, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits_copy,&
                  !$omp r3c_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vovo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(b,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3c_excits_copy(jdet,1); n = r3c_excits_copy(jdet,5);
                        ! compute < ab~c~j~k~ | h2b(vovo) | db~c~j~n~ >
                        hmatel = -h2b_vovo(a,n,d,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     ! (jk)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3c_excits_copy(jdet,1); n = r3c_excits_copy(jdet,5);
                        ! compute < ab~c~j~k~ | h2b(vovo) | db~c~k~n~ >
                        hmatel = h2b_vovo(a,n,d,j)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 8: A(bc) h2b(abef)*r3c(efcjk)
                  ! allocate new sorting arrays
                  nloc = nob*(nob - 1)/2*(nub - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nob,nob,nub))
                  !!! SB: (4,5,3) !!!
                  call get_index_table3(idx_table3, (/1,nob-1/), (/-1,nob/), (/2,nub/), nob, nob, nub)
                  call sort3(r3c_excits_copy, r3c_amps_copy, loc_arr, idx_table3, (/4,5,3/), nob, nob, nub, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits_copy,&
                  !$omp r3c_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vvvv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(j,k,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3c_excits_copy(jdet,1); e = r3c_excits_copy(jdet,2);
                        ! compute < ab~c~j~k~ | h2b(vvvv) | de~c~j~k~ >
                        !hmatel = h2b_vvvv(a,b,d,e)
                        hmatel = h2b_vvvv(d,e,a,b)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     ! (bc)
                     idx = idx_table3(j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3c_excits_copy(jdet,1); e = r3c_excits_copy(jdet,2);
                        ! compute < ab~c~j~k~ | h2b(vvvv) | de~b~j~k~ >
                        !hmatel = -h2b_vvvv(a,c,d,e)
                        hmatel = -h2b_vvvv(d,e,a,c)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,2) !!!
                  call get_index_table3(idx_table3, (/1,nob-1/), (/-1,nob/), (/1,nub-1/), nob, nob, nub)
                  call sort3(r3c_excits_copy, r3c_amps_copy, loc_arr, idx_table3, (/4,5,2/), nob, nob, nub, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits_copy,&
                  !$omp r3c_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vvvv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(j,k,b)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3c_excits_copy(jdet,1); f = r3c_excits_copy(jdet,3);
                        ! compute < ab~c~j~k~ | h2b(vvvv) | db~f~j~k~ >
                        !hmatel = h2b_vvvv(a,c,d,f)
                        hmatel = h2b_vvvv(d,f,a,c)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     ! (bc)
                     idx = idx_table3(j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = r3c_excits_copy(jdet,1); f = r3c_excits_copy(jdet,3);
                        ! compute < ab~c~j~k~ | h2b(vvvv) | dc~f~j~k~ >
                        !hmatel = -h2b_vvvv(a,b,d,f)
                        hmatel = -h2b_vvvv(d,f,a,b)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 9: 1/2 h2c(bcef)*r3c(aefjk)
                  ! allocate new sorting arrays
                  nloc = nua*nob*(nob - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nob,nob,nua))
                  !!! SB: (4,5,1) !!!
                  call get_index_table3(idx_table3, (/1,nob-1/), (/-1,nob/), (/1,nua/), nob, nob, nua)
                  call sort3(r3c_excits_copy, r3c_amps_copy, loc_arr, idx_table3, (/4,5,1/), nob, nob, nua, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits_copy,&
                  !$omp r3c_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2C_vvvv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(j,k,a)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = r3c_excits_copy(jdet,2); f = r3c_excits_copy(jdet,3);
                        ! compute < ab~c~j~k~ | h2b(vvvv) | ae~f~j~k~ >
                        !hmatel = h2c_vvvv(b,c,e,f)
                        hmatel = h2c_vvvv(e,f,b,c)
                        resid(a,b,c) = resid(a,b,c) + hmatel * r3c_amps_copy(jdet)
                     end do
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3c_excits_copy,&
                  !$omp t2b,t2c,r2b,&
                  !$omp h2b_vvvo,&
                  !$omp h2c_vooo,h2c_vvov,&
                  !$omp x2b_voo,x2b_ovo,x2b_vvv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                      ! zero out value
                      res_mm23 = 0.0d0
                      do m = 1,noa
                         ! -A(jk)A(bc) x2b_ovo(mck) * t2b(abmj)
                         res_mm23 = res_mm23 - x2b_ovo(m,c,k) * t2b(a,b,m,j) ! (1)
                         res_mm23 = res_mm23 + x2b_ovo(m,b,k) * t2b(a,c,m,j) ! (bc)
                         res_mm23 = res_mm23 + x2b_ovo(m,c,j) * t2b(a,b,m,k) ! (jk)
                         res_mm23 = res_mm23 - x2b_ovo(m,b,j) * t2b(a,c,m,k) ! (bc)(jk)
                      end do
                      do m = 1,nob
                         ! -A(bc) h2c(cmkj) * r2b(abm)
                         res_mm23 = res_mm23 - h2c_vooo(c,m,k,j) * r2b(a,b,m) ! (1)
                         res_mm23 = res_mm23 + h2c_vooo(b,m,k,j) * r2b(a,c,m) ! (bc)
                         ! -A(jk) x2b_voo(amj) * t2c(bcmk)
                         res_mm23 = res_mm23 - x2b_voo(a,m,j) * t2c(b,c,m,k) ! (1)
                         res_mm23 = res_mm23 + x2b_voo(a,m,k) * t2c(b,c,m,j) ! (jk)
                      end do
                      do e = 1,nua
                         ! A(bc)A(jk) h2b(acek) * r2b(ebj)
                         res_mm23 = res_mm23 + h2b_vvvo(a,c,e,k) * r2b(e,b,j) ! (1)
                         res_mm23 = res_mm23 - h2b_vvvo(a,b,e,k) * r2b(e,c,j) ! (bc)
                         res_mm23 = res_mm23 - h2b_vvvo(a,c,e,j) * r2b(e,b,k) ! (jk)
                         res_mm23 = res_mm23 + h2b_vvvo(a,b,e,j) * r2b(e,c,k) ! (bc)(jk)
                      end do
                      do e = 1,nub
                         ! A(jk) h2c(cbke) * r2b(aej)
                         res_mm23 = res_mm23 + h2c_vvov(c,b,k,e) * r2b(a,e,j) ! (1)
                         res_mm23 = res_mm23 - h2c_vvov(c,b,j,e) * r2b(a,e,k) ! (jk)
                         ! A(bc) x2b_vvv(abe) * t2c(ecjk)
                         res_mm23 = res_mm23 + x2b_vvv(a,b,e) * t2c(e,c,j,k) ! (1)
                         res_mm23 = res_mm23 - x2b_vvv(a,c,e) * t2c(e,b,j,k) ! (bc)
                      end do
                      resid(a,b,c) = resid(a,b,c) + res_mm23
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate r3c array copies
                  deallocate(r3c_excits_copy,r3c_amps_copy)
                  ! antisymmetrize m(abc)
                  do a=1,nua
                     do b=1,nub
                        do c=b+1,nub
                           resid(a,c,b) = -resid(a,b,c)
                        end do
                     end do
                  end do
              end subroutine build_moments3c_jk

              subroutine build_leftamps3a_jk(resid,j,k,&
                                             l1a,l2a,&
                                             l3a_amps,l3a_excits,&
                                             l3b_amps,l3b_excits,&
                                             h1a_ov,h1a_oo,h1a_vv,&
                                             h2a_vvvv,h2a_oooo,h2a_voov,h2a_ooov,h2a_vovv,h2a_oovv,&
                                             h2b_voov,&
                                             x2a_ovo,x2a_vvv,&
                                             n3aaa,n3aab,&
                                             noa,nua,nob,nub)

                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa, n3aab
                  ! occupied orbital block indices
                  integer, intent(in) :: j, k
                  ! Input L arrays
                  real(kind=8), intent(in) :: l1a(nua)
                  real(kind=8), intent(in) :: l2a(nua,nua,noa)
                  integer, intent(in) :: l3a_excits(n3aaa,5), l3b_excits(n3aab,5)
                  real(kind=8), intent(in) :: l3a_amps(n3aaa), l3b_amps(n3aab)
                  ! Input H  and X arrays
                  real(kind=8), intent(in) :: h1a_ov(noa,nua)
                  real(kind=8), intent(in) :: h1a_oo(noa,noa)
                  real(kind=8), intent(in) :: h1a_vv(nua,nua)
                  real(kind=8), intent(in) :: h2a_vvvv(nua,nua,nua,nua)
                  real(kind=8), intent(in) :: h2a_oooo(noa,noa,noa,noa)
                  real(kind=8), intent(in) :: h2a_voov(nua,noa,noa,nua)
                  real(kind=8), intent(in) :: h2a_ooov(noa,noa,noa,nua)
                  real(kind=8), intent(in) :: h2a_vovv(nua,noa,nua,nua)
                  real(kind=8), intent(in) :: h2a_oovv(noa,noa,nua,nua)
                  real(kind=8), intent(in) :: h2b_voov(nua,nob,noa,nub)
                  real(kind=8), intent(in) :: x2a_ovo(noa,nua,noa)
                  real(kind=8), intent(in) :: x2a_vvv(nua,nua,nua)
                  ! Output and Inout variables
                  real(kind=8), intent(out) :: resid(nua,nua,nua)
                  ! Local variables
                  integer, allocatable :: excits_buff(:,:), l3a_excits_copy(:,:)
                  real(kind=8), allocatable :: amps_buff(:), l3a_amps_copy(:)
                  integer, allocatable :: idx_table4(:,:,:,:)
                  integer, allocatable :: idx_table3(:,:,:)
                  integer, allocatable :: idx_table2(:,:)
                  integer, allocatable :: loc_arr(:,:)
                  real(kind=8) :: r_amp, hmatel, res_mm23
                  integer :: a, b, c, d, i, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc
                  ! Q space array
                  logical(kind=1) :: qspace(nua,nua,nua)
                  
                  ! Zero the container that holds H*R
                  resid = 0.0d0
                  
                  ! copy over l3a_amps and l3a_excits
                  allocate(l3a_amps_copy(n3aaa),l3a_excits_copy(n3aaa,5))
                  l3a_amps_copy(:) = l3a_amps(:)
                  l3a_excits_copy(:,:) = l3a_excits(:,:)
                  
                  ! reorder l3a into (j,k) order
                  nloc = noa*(noa-1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table2(noa,noa))
                  call get_index_table2(idx_table2, (/1,noa-1/), (/-1,noa/), noa, noa)
                  call sort2(l3a_excits_copy, l3a_amps_copy, loc_arr, idx_table2, (/4,5/), noa, noa, nloc, n3aaa)
                  ! Construct Q space for block (j,k)
                  qspace = .true.
                  idx = idx_table2(j,k)
                  if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        a = l3a_excits_copy(jdet,1); b = l3a_excits_copy(jdet,2); c = l3a_excits_copy(jdet,3);
                        qspace(a,b,c) = .false.
                     end do
                  end if
                  deallocate(loc_arr,idx_table2)

                  !!!! diagram 1: -A(jk) h1a(mj)*l3a(abcmk)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)*(nua-2)/6 * noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nua,nua,noa))
                  !!! SB: (1,2,3,4) !!!
                  call get_index_table4(idx_table4, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/1,noa-1/), nua, nua, nua, noa)
                  call sort4(l3a_excits_copy, l3a_amps_copy, loc_arr, idx_table4, (/1,2,3,4/), nua, nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits_copy,&
                  !$omp l3a_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_oo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(a,b,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3a_excits_copy(jdet,5);
                        ! compute < abcjk | h1a(oo) | abcjm >
                        hmatel = -h1a_oo(k,m)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                     end do
                     ! (jk)
                     idx = idx_table4(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           m = l3a_excits_copy(jdet,5);
                           ! compute < abcjk | h1a(oo) | abckm >
                           hmatel = h1a_oo(j,m)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,3,5) !!!
                  call get_index_table4(idx_table4, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/2,noa/), nua, nua, nua, noa)
                  call sort4(l3a_excits_copy, l3a_amps_copy, loc_arr, idx_table4, (/1,2,3,5/), nua, nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits_copy,&
                  !$omp l3a_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_oo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(a,b,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3a_excits_copy(jdet,4);
                        ! compute < abcjk | h1a(oo) | abcmk >
                        hmatel = -h1a_oo(j,m)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                     end do
                     ! (jk)
                     idx = idx_table4(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           m = l3a_excits_copy(jdet,4);
                           ! compute < abcjk | h1a(oo) | abcmj >
                           hmatel = h1a_oo(k,m)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 2: A(b/ac) h1a(be)*l3a(aecjk)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*(nua - 2)/2*noa*(noa - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nua,noa,noa))
                  !!! SB: (1,3,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nua-2/), (/-2,nua/), (/1,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(l3a_excits_copy, l3a_amps_copy, loc_arr, idx_table4, (/1,3,4,5/), nua, nua, noa, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits_copy,&
                  !$omp l3a_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_vv,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(a,c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3a_excits_copy(jdet,2);
                        ! compute < abcjk | h1a(vv) | aecjk >
                        hmatel = h1a_vv(e,b)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits_copy(jdet,2);
                           ! compute < abcjk | h1a(vv) | becjk >
                           hmatel = -h1a_vv(e,a)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table4(a,b,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits_copy(jdet,2);
                           ! compute < abcjk | h1a(vv) | aebjk >
                           hmatel = -h1a_vv(e,c)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nua-2/), (/-1,nua-1/), (/1,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(l3a_excits_copy, l3a_amps_copy, loc_arr, idx_table4, (/1,2,4,5/), nua, nua, noa, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits_copy,&
                  !$omp l3a_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_vv,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(a,b,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3a_excits_copy(jdet,3);
                        ! compute < abcjk | h1a(vv) | abfjk >
                        hmatel = h1a_vv(f,c)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                     end do
                     ! (ac)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = l3a_excits_copy(jdet,3);
                           ! compute < abcjk | h1a(vv) | bcfjk >
                           hmatel = h1a_vv(f,a)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table4(a,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = l3a_excits_copy(jdet,3);
                           ! compute < abcjk | h1a(vv) | acfjk >
                           hmatel = -h1a_vv(f,b)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4,5) !!!
                  call get_index_table4(idx_table4, (/2,nua-1/), (/-1,nua/), (/1,noa-1/), (/-1,noa/), nua, nua, noa, noa)
                  call sort4(l3a_excits_copy, l3a_amps_copy, loc_arr, idx_table4, (/2,3,4,5/), nua, nua, noa, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits_copy,&
                  !$omp l3a_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_vv,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3a_excits_copy(jdet,1);
                        ! compute < abcjk | h1a(vv) | dbcjk >
                        hmatel = h1a_vv(d,a)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table4(a,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits_copy(jdet,1);
                           ! compute < abcjk | h1a(vv) | dacjk >
                           hmatel = -h1a_vv(d,b)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                     ! (ac)
                     idx = idx_table4(a,b,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits_copy(jdet,1);
                           ! compute < abcjk | h1a(vv) | dabjk >
                           hmatel = h1a_vv(d,c)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 3: 1/2 A(c/ab) h2a(abef)*l3a(efcjk)
                  ! allocate new sorting arrays
                  nloc = (nua - 2)*noa*(noa - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,noa,noa))
                  !!! SB: (3,4,5) !!!
                  call get_index_table3(idx_table3, (/3,nua/), (/1,noa-1/), (/-1,noa/), nua, noa, noa)
                  call sort3(l3a_excits_copy, l3a_amps_copy, loc_arr, idx_table3, (/3,4,5/), nua, noa, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits_copy,&
                  !$omp l3a_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_vvvv,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3a_excits_copy(jdet,1); e = l3a_excits_copy(jdet,2);
                        ! compute < abcjk | h2a(vvvv) | decjk >
                        hmatel = h2a_vvvv(d,e,a,b)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                     end do
                     ! (ac)
                     idx = idx_table3(a,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits_copy(jdet,1); e = l3a_excits_copy(jdet,2);
                           ! compute < abcjk | h2a(vvvv) | deajk >
                           hmatel = h2a_vvvv(d,e,b,c)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table3(b,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits_copy(jdet,1); e = l3a_excits_copy(jdet,2);
                           ! compute < abcjk | h2a(vvvv) | debjk >
                           hmatel = -h2a_vvvv(d,e,a,c)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,4,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/1,noa-1/), (/-1,noa/), nua, noa, noa)
                  call sort3(l3a_excits_copy, l3a_amps_copy, loc_arr, idx_table3, (/1,4,5/), nua, noa, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits_copy,&
                  !$omp l3a_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_vvvv,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3a_excits_copy(jdet,2); f = l3a_excits_copy(jdet,3);
                        ! compute < abcjk | h2a(vvvv) | aefjk >
                        hmatel = h2a_vvvv(e,f,b,c)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(b,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits_copy(jdet,2); f = l3a_excits_copy(jdet,3);
                           ! compute < abcjk | h2a(vvvv) | befjk >
                           hmatel = -h2a_vvvv(e,f,a,c)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                     ! (ac)
                     idx = idx_table3(c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits_copy(jdet,2); f = l3a_excits_copy(jdet,3);
                           ! compute < abcjk | h2a(vvvv) | cefjk >
                           hmatel = h2a_vvvv(e,f,a,b)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,4,5)
                  call get_index_table3(idx_table3, (/2,nua-1/), (/1,noa-1/), (/-1,noa/), nua, noa, noa)
                  call sort3(l3a_excits_copy, l3a_amps_copy, loc_arr, idx_table3, (/2,4,5/), nua, noa, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits_copy,&
                  !$omp l3a_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_vvvv,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(b,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3a_excits_copy(jdet,1); f = l3a_excits_copy(jdet,3);
                        ! compute < abcjk | h2a(vvvv) | dbfjk >
                        hmatel = h2a_vvvv(d,f,a,c)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(a,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits_copy(jdet,1); f = l3a_excits_copy(jdet,3);
                           ! compute < abcjk | h2a(vvvv) | dafjk >
                           hmatel = -h2a_vvvv(d,f,b,c)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table3(c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits_copy(jdet,1); f = l3a_excits_copy(jdet,3);
                           ! compute < abcjk | h2a(vvvv) | dcfjk >
                           hmatel = -h2a_vvvv(d,f,a,b)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 4: 1/4 h2a(mnjk)*l3a(abcmn)
                  ! allocate new sorting arrays
                  nloc = (nua - 2)*(nua - 1)*nua/6
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,nua))
                  !!! SB: (1,2,3) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), nua, nua, nua)
                  call sort3(l3a_excits_copy, l3a_amps_copy, loc_arr, idx_table3, (/1,2,3/), nua, nua, nua, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits_copy,&
                  !$omp l3a_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_oooo,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3a_excits_copy(jdet,4); n = l3a_excits_copy(jdet,5);
                        ! compute < abcjk | h2a(oooo) | abcmn >
                        hmatel = h2a_oooo(j,k,m,n)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                     end do
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 5: A(c/ab)A(jk) h2a(cmke)*l3a(abejm)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*(nua - 2)/2*(noa - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,noa))
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-1,nua-1/), (/1,noa-1/), nua, nua, noa)
                  call sort3(l3a_excits_copy, l3a_amps_copy, loc_arr, idx_table3, (/1,2,4/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits_copy,&
                  !$omp l3a_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3a_excits_copy(jdet,3); n = l3a_excits_copy(jdet,5);
                        ! compute < abcjk | h2a(voov) | abfjn >
                        hmatel = h2a_voov(c,n,k,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                     end do
                     ! (ac), (-1)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = l3a_excits_copy(jdet,3); n = l3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | bcfjn >
                           hmatel = h2a_voov(a,n,k,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = l3a_excits_copy(jdet,3); n = l3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | acfjn >
                           hmatel = -h2a_voov(b,n,k,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = l3a_excits_copy(jdet,3); n = l3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | abfkn >
                           hmatel = -h2a_voov(c,n,j,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                     ! (ac)(jk), (-1)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = l3a_excits_copy(jdet,3); n = l3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | bcfkn >
                           hmatel = -h2a_voov(a,n,j,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = l3a_excits_copy(jdet,3); n = l3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | acfkn >
                           hmatel = h2a_voov(b,n,j,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-2,nua/), (/1,noa-1/), nua, nua, noa)
                  call sort3(l3a_excits_copy, l3a_amps_copy, loc_arr, idx_table3, (/1,3,4/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits_copy,&
                  !$omp l3a_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3a_excits_copy(jdet,2); n = l3a_excits_copy(jdet,5);
                        ! compute < abcjk | h2a(voov) | aecjn >
                        hmatel = h2a_voov(b,n,k,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits_copy(jdet,2); n = l3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | becjn >
                           hmatel = -h2a_voov(a,n,k,e)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits_copy(jdet,2); n = l3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | aebjn >
                           hmatel = -h2a_voov(c,n,k,e)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits_copy(jdet,2); n = l3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | aeckn >
                           hmatel = -h2a_voov(b,n,j,e)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits_copy(jdet,2); n = l3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | beckn >
                           hmatel = h2a_voov(a,n,j,e)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits_copy(jdet,2); n = l3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | aebkn >
                           hmatel = h2a_voov(c,n,j,e)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4) !!!
                  call get_index_table3(idx_table3, (/2,nua-1/), (/-1,nua/), (/1,noa-1/), nua, nua, noa)
                  call sort3(l3a_excits_copy, l3a_amps_copy, loc_arr, idx_table3, (/2,3,4/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits_copy,&
                  !$omp l3a_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(b,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3a_excits_copy(jdet,1); n = l3a_excits_copy(jdet,5);
                        ! compute < abcjk | h2a(voov) | dbcjn >
                        hmatel = h2a_voov(a,n,k,d)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits_copy(jdet,1); n = l3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | dacjn >
                           hmatel = -h2a_voov(b,n,k,d)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if 
                     ! (ac), (-1)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits_copy(jdet,1); n = l3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | dabjn >
                           hmatel = h2a_voov(c,n,k,d)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if 
                     ! (jk)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits_copy(jdet,1); n = l3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | dbckn >
                           hmatel = -h2a_voov(a,n,j,d)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if 
                     ! (ab)(jk)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits_copy(jdet,1); n = l3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | dackn >
                           hmatel = h2a_voov(b,n,j,d)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if 
                     ! (ac)(jk), (-1)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits_copy(jdet,1); n = l3a_excits_copy(jdet,5);
                           ! compute < abcjk | h2a(voov) | dabkn >
                           hmatel = -h2a_voov(c,n,j,d)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if 
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-1,nua-1/), (/2,noa/), nua, nua, noa)
                  call sort3(l3a_excits_copy, l3a_amps_copy, loc_arr, idx_table3, (/1,2,5/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits_copy,&
                  !$omp l3a_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3a_excits_copy(jdet,3); m = l3a_excits_copy(jdet,4);
                        ! compute < abcjk | h2a(voov) | abfmk >
                        hmatel = h2a_voov(c,m,j,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                     end do
                     ! (ac), (-1)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = l3a_excits_copy(jdet,3); m = l3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | bcfmk >
                           hmatel = h2a_voov(a,m,j,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = l3a_excits_copy(jdet,3); m = l3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | acfmk >
                           hmatel = -h2a_voov(b,m,j,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = l3a_excits_copy(jdet,3); m = l3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | abfmj >
                           hmatel = -h2a_voov(c,m,k,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                     ! (ac)(jk), (-1)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = l3a_excits_copy(jdet,3); m = l3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | bcfmj >
                           hmatel = -h2a_voov(a,m,k,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = l3a_excits_copy(jdet,3); m = l3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | acfmj >
                           hmatel = h2a_voov(b,m,k,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-2,nua/), (/2,noa/), nua, nua, noa)
                  call sort3(l3a_excits_copy, l3a_amps_copy, loc_arr, idx_table3, (/1,3,5/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits_copy,&
                  !$omp l3a_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3a_excits_copy(jdet,2); m = l3a_excits_copy(jdet,4);
                        ! compute < abcjk | h2a(voov) | aecmk >
                        hmatel = h2a_voov(b,m,j,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits_copy(jdet,2); m = l3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | becmk >
                           hmatel = -h2a_voov(a,m,j,e)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits_copy(jdet,2); m = l3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | aebmk >
                           hmatel = -h2a_voov(c,m,j,e)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits_copy(jdet,2); m = l3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | aecmj >
                           hmatel = -h2a_voov(b,m,k,e)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits_copy(jdet,2); m = l3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | becmj >
                           hmatel = h2a_voov(a,m,k,e)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3a_excits_copy(jdet,2); m = l3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | aebmj >
                           hmatel = h2a_voov(c,m,k,e)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,5) !!!
                  call get_index_table3(idx_table3, (/2,nua-1/), (/-1,nua/), (/2,noa/), nua, nua, noa)
                  call sort3(l3a_excits_copy, l3a_amps_copy, loc_arr, idx_table3, (/2,3,5/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits_copy,&
                  !$omp l3a_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(b,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3a_excits_copy(jdet,1); m = l3a_excits_copy(jdet,4);
                        ! compute < abcjk | h2a(voov) | dbcmk >
                        hmatel = h2a_voov(a,m,j,d)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits_copy(jdet,1); m = l3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | dacmk >
                           hmatel = -h2a_voov(b,m,j,d)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                     ! (ac), (-1)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits_copy(jdet,1); m = l3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | dabmk >
                           hmatel = h2a_voov(c,m,j,d)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits_copy(jdet,1); m = l3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | dbcmj >
                           hmatel = -h2a_voov(a,m,k,d)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits_copy(jdet,1); m = l3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | dacmj >
                           hmatel = h2a_voov(b,m,k,d)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                     ! (ac)(jk), (-1)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3a_excits_copy(jdet,1); m = l3a_excits_copy(jdet,4);
                           ! compute < abcjk | h2a(voov) | dabmj >
                           hmatel = -h2a_voov(c,m,k,d)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3a_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 6: A(c/ab)A(jk) h2b(cmke)*l3b(abejm)
                  ! allocate and copy over l3b arrays
                  allocate(amps_buff(n3aab),excits_buff(n3aab,5))
                  amps_buff(:) = l3b_amps(:)
                  excits_buff(:,:) = l3b_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = nua*(nua - 1)/2*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,noa))
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/-1,nua/), (/1,noa/), nua, nua, noa)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,2,4/), nua, nua, noa, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits_copy,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=b+1,nua;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                        ! compute < abcjk | h2a(voov) | abfjn >
                        hmatel = h2b_voov(c,n,k,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     ! (ac), (-1)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                           ! compute < abcjk | h2a(voov) | bcfjn >
                           hmatel = h2b_voov(a,n,k,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                           ! compute < abcjk | h2a(voov) | acfjn >
                           hmatel = -h2b_voov(b,n,k,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                           ! compute < abcjk | h2a(voov) | abfkn >
                           hmatel = -h2b_voov(c,n,j,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ac)(jk), (-1)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                           ! compute < abcjk | h2a(voov) | bcfkn >
                           hmatel = -h2b_voov(a,n,j,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                           ! compute < abcjk | h2a(voov) | acfkn >
                           hmatel = h2b_voov(b,n,j,f)
                           resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3,excits_buff,amps_buff)

                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3a_excits_copy,&
                  !$omp l1a,l2a,&
                  !$omp h1a_ov,h2a_oovv,h2a_vovv,h2a_ooov,&
                  !$omp x2a_ovo,x2a_vvv,&
                  !$omp noa,nua,nob,nub,n3aaa),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=b+1,nua;
                      if (.not. qspace(a,b,c)) cycle
                      ! zero out value
                      res_mm23 = 0.0d0
                      ! A(a/bc) l1a(a)*h2a_oovv(j,k,b,c)
                      res_mm23 = res_mm23 + l1a(a)*h2a_oovv(j,k,b,c) ! (1)
                      res_mm23 = res_mm23 - l1a(b)*h2a_oovv(j,k,a,c) ! (ab)
                      res_mm23 = res_mm23 - l1a(c)*h2a_oovv(j,k,b,a) ! (ac)
                      ! A(c/ab)A(jk) l2a(a,b,j)*h1a_ov(k,c)
                      res_mm23 = res_mm23 + l2a(a,b,j)*h1a_ov(k,c) ! (1)
                      res_mm23 = res_mm23 - l2a(a,c,j)*h1a_ov(k,b) ! (bc)
                      res_mm23 = res_mm23 - l2a(c,b,j)*h1a_ov(k,a) ! (ac)
                      res_mm23 = res_mm23 - l2a(a,b,k)*h1a_ov(j,c) ! (jk)
                      res_mm23 = res_mm23 + l2a(a,c,k)*h1a_ov(j,b) ! (bc)(jk)
                      res_mm23 = res_mm23 + l2a(c,b,k)*h1a_ov(j,a) ! (ac)(jk)

                      do m = 1,noa
                         ! -A(c/ab) l2a(abm)*h2a_ooov(jkmc)
                         res_mm23 = res_mm23 - l2a(a,b,m)*h2a_ooov(j,k,m,c) ! (1)
                         res_mm23 = res_mm23 + l2a(c,b,m)*h2a_ooov(j,k,m,a) ! (ac)
                         res_mm23 = res_mm23 + l2a(a,c,m)*h2a_ooov(j,k,m,b) ! (bc)
                         ! -A(jk)A(c/ab) x2a_ovo(mck)*h2a_oovv(mjab)
                         res_mm23 = res_mm23 - x2a_ovo(m,c,k)*h2a_oovv(m,j,a,b) ! (1)
                         res_mm23 = res_mm23 + x2a_ovo(m,a,k)*h2a_oovv(m,j,c,b) ! (ac)
                         res_mm23 = res_mm23 + x2a_ovo(m,b,k)*h2a_oovv(m,j,a,c) ! (ab)
                         res_mm23 = res_mm23 + x2a_ovo(m,c,j)*h2a_oovv(m,k,a,b) ! (jk)
                         res_mm23 = res_mm23 - x2a_ovo(m,a,j)*h2a_oovv(m,k,c,b) ! (ac)(jk)
                         res_mm23 = res_mm23 - x2a_ovo(m,b,j)*h2a_oovv(m,k,a,c) ! (ab)(jk)
                      end do
                      do e = 1,nua
                         ! A(jk)A(c/ab) l2a(eck)*h2a_vovv(ejab)
                         res_mm23 = res_mm23 + l2a(e,c,k)*h2a_vovv(e,j,a,b) ! (1)
                         res_mm23 = res_mm23 - l2a(e,a,k)*h2a_vovv(e,j,c,b) ! (ac)
                         res_mm23 = res_mm23 - l2a(e,b,k)*h2a_vovv(e,j,a,c) ! (bc)
                         res_mm23 = res_mm23 - l2a(e,c,j)*h2a_vovv(e,k,a,b) ! (jk)
                         res_mm23 = res_mm23 + l2a(e,a,j)*h2a_vovv(e,k,c,b) ! (ac)(jk)
                         res_mm23 = res_mm23 + l2a(e,b,j)*h2a_vovv(e,k,a,c) ! (bc)(jk)
                         ! A(c/ab) x2a_vvv(aeb)*h2a_oovv(jkec)
                         res_mm23 = res_mm23 + x2a_vvv(a,e,b)*h2a_oovv(j,k,e,c) ! (1)
                         res_mm23 = res_mm23 - x2a_vvv(c,e,b)*h2a_oovv(j,k,e,a) ! (ac)
                         res_mm23 = res_mm23 - x2a_vvv(a,e,c)*h2a_oovv(j,k,e,b) ! (bc)
                      end do
                      resid(a,b,c) = resid(a,b,c) + res_mm23
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate l3a array copies
                  deallocate(l3a_excits_copy,l3a_amps_copy)
                  ! antisymmetrize m(abc) block
                  do a = 1,nua
                     do b = a+1,nua
                        do c = b+1,nua
                           resid(a,c,b) = -resid(a,b,c)
                           resid(b,c,a) = resid(a,b,c)
                           resid(b,a,c) = -resid(a,b,c)
                           resid(c,a,b) = resid(a,b,c)
                           resid(c,b,a) = -resid(a,b,c)
                        end do
                     end do
                  end do
              end subroutine build_leftamps3a_jk

              subroutine build_leftamps3b_jk(resid,j,k,&
                                             l1a,l2a,l2b,&
                                             l3a_amps,l3a_excits,&
                                             l3b_amps,l3b_excits,&
                                             l3c_amps,l3c_excits,&
                                             h1a_ov,h1b_ov,h1a_oo,h1a_vv,h1b_oo,h1b_vv,&
                                             h2a_vvvv,h2a_voov,h2a_vovv,h2a_oovv,&
                                             h2b_vvvv,h2b_oooo,h2b_voov,h2b_vovo,h2b_ovov,h2b_ovvo,h2b_oovv,&
                                             h2b_ooov,h2b_oovo,h2b_vovv,h2b_ovvv,&
                                             h2c_voov,&
                                             x2a_ovo,x2a_vvv,&
                                             x2b_voo,x2b_ovo,x2b_vvv,&
                                             n3aaa,n3aab,n3abb,&
                                             noa,nua,nob,nub)

                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa, n3aab, n3abb
                  ! Occupied block indices
                  integer, intent(in) :: j, k
                  ! Input R and T arrays
                  real(kind=8), intent(in) :: l1a(nua)
                  real(kind=8), intent(in) :: l2a(nua,nua,noa)
                  real(kind=8), intent(in) :: l2b(nua,nub,nob)
                  integer, intent(in) :: l3a_excits(n3aaa,5), l3b_excits(n3aab,5), l3c_excits(n3abb,5)
                  real(kind=8), intent(in) :: l3a_amps(n3aaa), l3b_amps(n3aab), l3c_amps(n3abb)
                  ! Input H  and X arrays
                  real(kind=8), intent(in) :: h1a_ov(noa,nua)
                  real(kind=8), intent(in) :: h1a_oo(noa,noa)
                  real(kind=8), intent(in) :: h1a_vv(nua,nua)
                  real(kind=8), intent(in) :: h1b_ov(nob,nub)
                  real(kind=8), intent(in) :: h1b_oo(nob,nob)
                  real(kind=8), intent(in) :: h1b_vv(nub,nub)
                  real(kind=8), intent(in) :: h2a_vvvv(nua,nua,nua,nua)
                  real(kind=8), intent(in) :: h2a_voov(nua,noa,noa,nua)
                  real(kind=8), intent(in) :: h2a_vovv(nua,noa,nua,nua)
                  real(kind=8), intent(in) :: h2a_oovv(noa,noa,nua,nua)
                  real(kind=8), intent(in) :: h2b_vvvv(nua,nub,nua,nub)
                  real(kind=8), intent(in) :: h2b_oooo(noa,nob,noa,nob)
                  real(kind=8), intent(in) :: h2b_voov(nua,nob,noa,nub)
                  real(kind=8), intent(in) :: h2b_vovo(nua,nob,nua,nob)
                  real(kind=8), intent(in) :: h2b_ovov(noa,nub,noa,nub)
                  real(kind=8), intent(in) :: h2b_ovvo(noa,nub,nua,nob)
                  real(kind=8), intent(in) :: h2b_oovv(noa,nob,nua,nub)
                  real(kind=8), intent(in) :: h2b_ooov(noa,nob,noa,nub)
                  real(kind=8), intent(in) :: h2b_oovo(noa,nob,nua,nob)
                  real(kind=8), intent(in) :: h2b_vovv(nua,nob,nua,nub)
                  real(kind=8), intent(in) :: h2b_ovvv(noa,nub,nua,nub)
                  real(kind=8), intent(in) :: h2c_voov(nub,nob,nob,nub)
                  real(kind=8), intent(in) :: x2a_ovo(noa,nua,noa)
                  real(kind=8), intent(in) :: x2a_vvv(nua,nua,nua)
                  real(kind=8), intent(in) :: x2b_voo(nua,nob,nob)
                  real(kind=8), intent(in) :: x2b_ovo(noa,nub,nob)
                  real(kind=8), intent(in) :: x2b_vvv(nua,nub,nub)
                  ! Output and Inout variables
                  real(kind=8), intent(out) :: resid(nua,nua,nub)
                  ! Local variables
                  integer, allocatable :: excits_buff(:,:), l3b_excits_copy(:,:)
                  real(kind=8), allocatable :: amps_buff(:), l3b_amps_copy(:)
                  integer, allocatable :: idx_table4(:,:,:,:)
                  integer, allocatable :: idx_table3(:,:,:)
                  integer, allocatable :: idx_table2(:,:)
                  integer, allocatable :: loc_arr(:,:)
                  real(kind=8) :: r_amp, hmatel, res_mm23
                  integer :: a, b, c, d, i, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc
                  ! Q space array
                  logical(kind=1) :: qspace(nua,nua,nub)
                  
                  ! Zero the container that holds H*R
                  resid = 0.0d0
                  
                  ! copy over l3b_amps and l3b_excits
                  allocate(l3b_amps_copy(n3aab),l3b_excits_copy(n3aab,5))
                  l3b_amps_copy(:) = l3b_amps(:)
                  l3b_excits_copy(:,:) = l3b_excits(:,:)
                  
                  ! reorder l3b into (j,k) order
                  nloc = noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table2(noa,nob))
                  call get_index_table2(idx_table2, (/1,noa/), (/1,nob/), noa, nob)
                  call sort2(l3b_excits_copy, l3b_amps_copy, loc_arr, idx_table2, (/4,5/), noa, nob, nloc, n3aab)
                  ! Construct Q space for block (j,k)
                  qspace = .true.
                  idx = idx_table2(j,k)
                  if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        a = l3b_excits_copy(jdet,1); b = l3b_excits_copy(jdet,2); c = l3b_excits_copy(jdet,3);
                        qspace(a,b,c) = .false.
                     end do
                  end if
                  deallocate(loc_arr,idx_table2)

                  !!! diagram 1: -h1a(mj)*l3b(abcmk)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2*nub * nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nua,nub,nob))
                  !!! SB: (1,2,3,5) !!!
                  call get_index_table4(idx_table4, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/1,nob/), nua, nua, nub, nob)
                  call sort4(l3b_excits_copy, l3b_amps_copy, loc_arr, idx_table4, (/1,2,3,5/), nua, nua, nub, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits_copy,&
                  !$omp l3b_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_oo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(a,b,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3b_excits_copy(jdet,4);
                        ! compute < abcjk | h1a(oo) | abcmk >
                        hmatel = -h1a_oo(j,m)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3b_amps_copy(jdet)
                     end do
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 2: -h1b(mk)*l3b(abcjm)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)/2*nub * noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nua,nub,noa))
                  !!! SB: (1,2,3,4) !!!
                  call get_index_table4(idx_table4, (/1,nua-1/), (/-1,nua/), (/1,nub/), (/1,noa/), nua, nua, nub, noa)
                  call sort4(l3b_excits_copy, l3b_amps_copy, loc_arr, idx_table4, (/1,2,3,4/), nua, nua, nub, noa, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits_copy,&
                  !$omp l3b_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1B_oo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(a,b,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        n = l3b_excits_copy(jdet,5);
                        ! compute < abcjk | h1b(oo) | abcjn >
                        hmatel = -h1b_oo(k,n)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3b_amps_copy(jdet)
                     end do
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 3: h1b(ce)*l3b(abejk)
                  ! allocate new sorting arrays
                  nloc = nua*(nua - 1)/2 * noa * nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nua,noa,nob))
                  !!! SB: (1,2,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nua-1/), (/-1,nua/), (/1,noa/), (/1,nob/), nua, nua, noa, nob)
                  call sort4(l3b_excits_copy, l3b_amps_copy, loc_arr, idx_table4, (/1,2,4,5/), nua, nua, noa, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits_copy,&
                  !$omp l3b_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1B_vv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(a,b,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3b_excits_copy(jdet,3);
                        ! compute < abcjk | h1b(vv) | abfjk >
                        hmatel = h1b_vv(f,c)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3b_amps_copy(jdet)
                     end do
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 4: h1a(be)*l3b(aecjk)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*nub*noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nua,nub,noa,nob))
                  !!! SB: (1,3,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nua-1/), (/1,nub/), (/1,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(l3b_excits_copy, l3b_amps_copy, loc_arr, idx_table4, (/1,3,4,5/), nua, nub, noa, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits_copy,&
                  !$omp l3b_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_vv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(a,c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3b_excits_copy(jdet,2);
                        ! compute < abcjk | h1a(vv) | aecjk >
                        hmatel = h1a_vv(e,b)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3b_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table4(b,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           e = l3b_excits_copy(jdet,2);
                           ! compute < abcjk | h1a(vv) | becjk >
                           hmatel = -h1a_vv(e,a)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3b_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4,5) !!!
                  call get_index_table4(idx_table4, (/2,nua/), (/1,nub/), (/1,noa/), (/1,nob/), nua, nub, noa, nob)
                  call sort4(l3b_excits_copy, l3b_amps_copy, loc_arr, idx_table4, (/2,3,4,5/), nua, nub, noa, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits_copy,&
                  !$omp l3b_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_vv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3b_excits_copy(jdet,1);
                        ! compute < abcjk | h1a(vv) | dbcjk >
                        hmatel = h1a_vv(d,a)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3b_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table4(a,c,j,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx),loc_arr(2,idx)
                           d = l3b_excits_copy(jdet,1);
                           ! compute < abcjk | h1a(vv) | dacjk >
                           hmatel = -h1a_vv(d,b)
                           resid(a,b,c) = resid(a,b,c) + hmatel * l3b_amps_copy(jdet)
                        end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 5: h2b(mnjk)*l3b(abcmn)
                  ! allocate new sorting arrays
                  nloc = nua*(nua - 1)/2*nub
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,nub))
                  !!! SB: (1,2,3) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/-1,nua/), (/1,nub/), nua, nua, nub)
                  call sort3(l3b_excits_copy, l3b_amps_copy, loc_arr, idx_table3, (/1,2,3/), nua, nua, nub, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits_copy,&
                  !$omp l3b_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_oooo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3b_excits_copy(jdet,4); n = l3b_excits_copy(jdet,5);
                        ! compute < abcjk | h2b(oooo) | abcmn >
                        hmatel = h2b_oooo(j,k,m,n)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3b_amps_copy(jdet)
                     end do
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 6: h2b(mcek)*r3a(abejm)
                  ! allocate and copy over r3a arrays
                  allocate(amps_buff(n3aaa),excits_buff(n3aaa,5))
                  amps_buff(:) = l3a_amps(:)
                  excits_buff(:,:) = l3a_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*(nua - 2)/2*(noa - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,noa))
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-1,nua-1/), (/1,noa-1/), nua, nua, noa)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,2,4/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits_copy,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                        ! compute < abc~jk~ | h2b(ovvo) | abfjn >
                        hmatel = h2b_ovvo(n,c,f,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-1,nua-1/), (/2,noa/), nua, nua, noa)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,2,5/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits_copy,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,3); m = excits_buff(jdet,4);
                        ! compute < abc~jk~ | h2b(ovvo) | abfmj >
                        hmatel = -h2b_ovvo(m,c,f,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-2,nua/), (/1,noa-1/), nua, nua, noa)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,3,4/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits_copy,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); n = excits_buff(jdet,5);
                        ! compute < abc~jk~ | h2b(ovvo) | aebjn >
                        hmatel = -h2b_ovvo(n,c,e,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-2/), (/-2,nua/), (/2,noa/), nua, nua, noa)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,3,5/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits_copy,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < abc~jk~ | h2b(ovvo) | aebmj >
                        hmatel = h2b_ovvo(m,c,e,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4) !!!
                  call get_index_table3(idx_table3, (/2,nua-1/), (/-1,nua/), (/1,noa-1/), nua, nua, noa)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/2,3,4/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits_copy,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = excits_buff(jdet,1); n = excits_buff(jdet,5);
                        ! compute < abc~jk~ | h2b(ovvo) | dabjn >
                        hmatel = h2b_ovvo(n,c,d,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,5) !!!
                  call get_index_table3(idx_table3, (/2,nua-1/), (/-1,nua/), (/2,noa/), nua, nua, noa)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/2,3,5/), nua, nua, noa, nloc, n3aaa)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits_copy,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < abc~jk~ | h2b(ovvo) | dabmj >
                        hmatel = -h2b_ovvo(m,c,d,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3,excits_buff,amps_buff)

                  !!! diagram 7: h2c(cmke)*l3b(abejm)
                  ! allocate new sorting arrays
                  nloc = nua*(nua - 1)/2*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,noa))
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/-1,nua/), (/1,noa/), nua, nua, noa)
                  call sort3(l3b_excits_copy, l3b_amps_copy, loc_arr, idx_table3, (/1,2,4/), nua, nua, noa, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits_copy,&
                  !$omp l3b_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3b_excits_copy(jdet,3); n = l3b_excits_copy(jdet,5);
                        ! compute < abcjk | h2c(voov) | abfjn >
                        hmatel = h2c_voov(c,n,k,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3b_amps_copy(jdet)
                     end do
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 8: h2a(bmje)*l3b(aecmk)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*nub*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nub,nob))
                  !!! SB: (1,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/1,nub/), (/1,nob/), nua, nub, nob)
                  call sort3(l3b_excits_copy, l3b_amps_copy, loc_arr, idx_table3, (/1,3,5/), nua, nub, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits_copy,&
                  !$omp l3b_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3b_excits_copy(jdet,2); m = l3b_excits_copy(jdet,4);
                        ! compute < abc~jk~ | h2a(voov) | aec~mk~ >
                        hmatel = h2a_voov(b,m,j,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3b_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3b_excits_copy(jdet,2); m = l3b_excits_copy(jdet,4);
                        ! compute < abc~jk~ | h2a(voov) | bec~mk~ >
                        hmatel = -h2a_voov(a,m,j,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3b_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,5) !!!
                  call get_index_table3(idx_table3, (/2,nua/), (/1,nub/), (/1,nob/), nua, nub, nob)
                  call sort3(l3b_excits_copy, l3b_amps_copy, loc_arr, idx_table3, (/2,3,5/), nua, nub, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits_copy,&
                  !$omp l3b_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(b,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3b_excits_copy(jdet,1); m = l3b_excits_copy(jdet,4);
                        ! compute < abc~jk~ | h2a(voov) | dbc~mk~ >
                        hmatel = h2a_voov(a,m,j,d)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3b_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3b_excits_copy(jdet,1); m = l3b_excits_copy(jdet,4);
                        ! compute < abc~jk~ | h2a(voov) | dac~mk~ >
                        hmatel = -h2a_voov(b,m,j,d)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3b_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)
                
                  !!! diagram 9: -A(ab) h2b(bmek)*l3b(aecjm)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*nub*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nub,noa))
                  !!! SB: (1,3,4) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/1,nub/), (/1,noa/), nua, nub, noa)
                  call sort3(l3b_excits_copy, l3b_amps_copy, loc_arr, idx_table3, (/1,3,4/), nua, nub, noa, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits_copy,&
                  !$omp l3b_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vovo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3b_excits_copy(jdet,2); n = l3b_excits_copy(jdet,5);
                        ! compute < abc~jk~ | h2b(vovo) | aec~jn~ >
                        hmatel = -h2b_vovo(b,n,e,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3b_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3b_excits_copy(jdet,2); n = l3b_excits_copy(jdet,5);
                        ! compute < abc~jk~ | h2b(vovo) | bec~jn~ >
                        hmatel = h2b_vovo(a,n,e,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3b_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4) !!!
                  call get_index_table3(idx_table3, (/2,nua/), (/1,nub/), (/1,noa/), nua, nub, noa)
                  call sort3(l3b_excits_copy, l3b_amps_copy, loc_arr, idx_table3, (/2,3,4/), nua, nub, noa, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits_copy,&
                  !$omp l3b_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vovo,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(b,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3b_excits_copy(jdet,1); n = l3b_excits_copy(jdet,5);
                        ! compute < abc~jk~ | h2b(vovo) | bec~jn~ >
                        hmatel = -h2b_vovo(a,n,d,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3b_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3b_excits_copy(jdet,1); n = l3b_excits_copy(jdet,5);
                        ! compute < abc~jk~ | h2b(vovo) | aec~jn~ >
                        hmatel = h2b_vovo(b,n,d,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3b_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 10: -h2b(mcje)*l3b(abemk)
                  ! allocate new sorting arrays
                  nloc = nua*(nua - 1)/2 * nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nua,nob))
                  !!! SB: (1,2,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/-1,nua/), (/1,nob/), nua, nua, nob)
                  call sort3(l3b_excits_copy, l3b_amps_copy, loc_arr, idx_table3, (/1,2,5/), nua, nua, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits_copy,&
                  !$omp l3b_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3b_excits_copy(jdet,3); m = l3b_excits_copy(jdet,4);
                        ! compute < abc~jk~ | h2b(vovo) | abf~mk~ >
                        hmatel = -h2b_ovov(m,c,j,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3b_amps_copy(jdet)
                     end do
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 11: A(ab) h2b(bmje)*l3c(aecmk)
                  ! allocate and copy over l3c arrays
                  allocate(amps_buff(n3abb),excits_buff(n3abb,5))
                  amps_buff(:) = l3c_amps(:)
                  excits_buff(:,:) = l3c_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = nua*(nub - 1)*(nob - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nub,nob))
                  !!! SB: (1,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/2,nub/), (/2,nob/), nua, nub, nob)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,3,5/), nua, nub, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits_copy,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < abc~jk~ | h2b(voov) | ae~c~m~k~ >
                        hmatel = h2b_voov(b,m,j,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < abc~jk~ | h2b(voov) | be~c~m~k~ >
                        hmatel = -h2b_voov(a,m,j,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,4) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/2,nub/), (/1,nob-1/), nua, nub, nob)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,3,4/), nua, nub, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits_copy,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); n = excits_buff(jdet,5);
                        ! compute < abc~jk~ | h2b(voov) | ae~c~k~n~ >
                        hmatel = -h2b_voov(b,n,j,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); n = excits_buff(jdet,5);
                        ! compute < abc~jk~ | h2b(voov) | be~c~k~n~ >
                        hmatel = h2b_voov(a,n,j,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,5) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/1,nub-1/), (/2,nob/), nua, nub, nob)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,2,5/), nua, nub, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits_copy,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,3); m = excits_buff(jdet,4);
                        ! compute < abc~jk~ | h2b(voov) | ac~f~m~k~ >
                        hmatel = -h2b_voov(b,m,j,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,3); m = excits_buff(jdet,4);
                        ! compute < abc~jk~ | h2b(voov) | bc~f~m~k~ >
                        hmatel = h2b_voov(a,m,j,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/1,nub-1/), (/1,nob-1/), nua, nub, nob)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,2,4/), nua, nub, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits_copy,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_voov,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                        ! compute < abc~jk~ | h2b(voov) | ac~f~k~n~ >
                        hmatel = h2b_voov(b,n,j,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = excits_buff(jdet,3); n = excits_buff(jdet,5);
                        ! compute < abc~jk~ | h2b(voov) | bc~f~k~n~ >
                        hmatel = -h2b_voov(a,n,j,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3,excits_buff,amps_buff)

                  !!! diagram 12: 1/2 h2a(abef)*l3b(efcjk)
                  ! allocate new sorting arrays
                  nloc = nub*noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nub,noa,nob))
                  !!! SB: (3,4,5) !!!
                  call get_index_table3(idx_table3, (/1,nub/), (/1,noa/), (/1,nob/), nub, noa, nob)
                  call sort3(l3b_excits_copy, l3b_amps_copy, loc_arr, idx_table3, (/3,4,5/), nub, noa, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits_copy,&
                  !$omp l3b_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2A_vvvv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3b_excits_copy(jdet,1); e = l3b_excits_copy(jdet,2);
                        ! compute < abc~jk~ | h2a(vvvv) | dfc~jk~ >
                        hmatel = h2a_vvvv(d,e,a,b)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3b_amps_copy(jdet)
                     end do
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 13: A(ab) h2b(bcef)*l3b(aefjk)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,noa,nob))
                  !!! SB: (1,4,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/1,noa/), (/1,nob/), nua, noa, nob)
                  call sort3(l3b_excits_copy, l3b_amps_copy, loc_arr, idx_table3, (/1,4,5/), nua, noa, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits_copy,&
                  !$omp l3b_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vvvv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3b_excits_copy(jdet,2); f = l3b_excits_copy(jdet,3);
                        ! compute < abc~jk~ | h2b(vvvv) | aef~jk~ >
                        hmatel = h2b_vvvv(e,f,b,c)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3b_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3b_excits_copy(jdet,2); f = l3b_excits_copy(jdet,3);
                        ! compute < abc~jk~ | h2b(vvvv) | bef~jk~ >
                        hmatel = -h2b_vvvv(e,f,a,c)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3b_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,4,5) !!!
                  call get_index_table3(idx_table3, (/2,nua/), (/1,noa/), (/1,nob/), nua, noa, nob)
                  call sort3(l3b_excits_copy, l3b_amps_copy, loc_arr, idx_table3, (/2,4,5/), nua, noa, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits_copy,&
                  !$omp l3b_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vvvv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(b,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3b_excits_copy(jdet,1); f = l3b_excits_copy(jdet,3);
                        ! compute < abc~jk~ | h2b(vvvv) | dbf~jk~ >
                        hmatel = h2b_vvvv(d,f,a,c)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3b_amps_copy(jdet)
                     end do
                     ! (ab)
                     idx = idx_table3(a,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3b_excits_copy(jdet,1); f = l3b_excits_copy(jdet,3);
                        ! compute < abc~jk~ | h2b(vvvv) | daf~jk~ >
                        hmatel = -h2b_vvvv(d,f,b,c)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3b_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3b_excits_copy,&
                  !$omp l1a,l2a,l2b,&
                  !$omp h2a_vovv,h2b_ooov,h2b_oovo,h2b_vovv,h2b_ovvv,&
                  !$omp h1a_ov,h1b_ov,h2a_oovv,h2b_oovv,&
                  !$omp x2a_ovo,x2a_vvv,x2b_voo,x2b_ovo,x2b_vvv,&
                  !$omp noa,nua,nob,nub,n3aab),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=a+1,nua; do c=1,nub;
                     if (.not. qspace(a,b,c)) cycle
                      res_mm23 = 0.0d0
                      ! A(ab) l1a(a)*h2b_oovv(jkbc)
                      res_mm23 = res_mm23 + l1a(a)*h2b_oovv(j,k,b,c) ! (1)
                      res_mm23 = res_mm23 - l1a(b)*h2b_oovv(j,k,a,c) ! (ab)
                      ! l2a(abj)*h1b_ov(kc)
                      res_mm23 = res_mm23 + l2a(a,b,j)*h1b_ov(k,c) ! (1)
                      ! A(ab) l2b(ack)*h1a_ov(jb)
                      res_mm23 = res_mm23 + l2b(a,c,k)*h1a_ov(j,b) ! (1)
                      res_mm23 = res_mm23 - l2b(b,c,k)*h1a_ov(j,a) ! (ab)
                      do m = 1,noa
                         ! -l2a(abm)*h2b_ooov(jkmc)
                         res_mm23 = res_mm23 - l2a(a,b,m)*h2b_ooov(j,k,m,c)
                         ! -x2b_ovo(mck)*h2a_oovv(mjab)
                         res_mm23 = res_mm23 - x2b_ovo(m,c,k)*h2a_oovv(m,j,a,b)
                         ! -A(ab) x2a_ovo(mbj)*h2b_oovv(mkac)
                         res_mm23 = res_mm23 - x2a_ovo(m,b,j)*h2b_oovv(m,k,a,c) ! (1)
                         res_mm23 = res_mm23 + x2a_ovo(m,a,j)*h2b_oovv(m,k,b,c) ! (ab)
                      end do
                      do m = 1,nob
                         ! A(ab) -l2b(acm)*h2b_oovo(jkbm)
                         res_mm23 = res_mm23 - l2b(a,c,m)*h2b_oovo(j,k,b,m) ! (1)
                         res_mm23 = res_mm23 + l2b(b,c,m)*h2b_oovo(j,k,a,m) ! (ab)
                         ! A(ab) -x2b_voo(akm)*h2b_oovv(jmbc)
                         res_mm23 = res_mm23 - x2b_voo(a,k,m)*h2b_oovv(j,m,b,c) ! (1)
                         res_mm23 = res_mm23 + x2b_voo(b,k,m)*h2b_oovv(j,m,a,c) ! (ab)
                      end do
                      do e = 1,nua
                         ! A(ab) l2a(aej)*h2b_vovv(ekbc)
                         res_mm23 = res_mm23 + l2a(a,e,j)*h2b_vovv(e,k,b,c) ! (1)
                         res_mm23 = res_mm23 - l2a(b,e,j)*h2b_vovv(e,k,a,c) ! (ab)
                         ! l2b(eck)*h2a_vovv(ejab)
                         res_mm23 = res_mm23 + l2b(e,c,k)*h2a_vovv(e,j,a,b) ! (1)
                         ! x2a_vvv(aeb)*h2b_oovv(jkec)
                         res_mm23 = res_mm23 + x2a_vvv(a,e,b)*h2b_oovv(j,k,e,c) ! (1)
                      end do
                      do e = 1,nub
                         ! A(ab) l2b(aek)*h2b_ovvv(jebc)
                         res_mm23 = res_mm23 + l2b(a,e,k)*h2b_ovvv(j,e,b,c) ! (1)
                         res_mm23 = res_mm23 - l2b(b,e,k)*h2b_ovvv(j,e,a,c) ! (ab)
                         ! A(ab) x2b_vvv(aec)*h2b_oovv(jkbe)
                         res_mm23 = res_mm23 + x2b_vvv(a,e,c)*h2b_oovv(j,k,b,e) ! (1)
                         res_mm23 = res_mm23 - x2b_vvv(b,e,c)*h2b_oovv(j,k,a,e) ! (ab)
                      end do
                      resid(a,b,c) = resid(a,b,c) + res_mm23
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate l3b array copies
                  deallocate(l3b_excits_copy,l3b_amps_copy)
                  ! antisymmetrize m(abc)
                  do a=1,nua
                     do b=a+1,nua
                        do c=1,nub
                           resid(b,a,c) = -resid(a,b,c)
                        end do
                     end do
                  end do
              end subroutine build_leftamps3b_jk

              subroutine build_leftamps3c_jk(resid,j,k,&
                                             l1a,l2b,&
                                             l3b_amps,l3b_excits,&
                                             l3c_amps,l3c_excits,&
                                             h1b_ov,h1a_vv,h1b_oo,h1b_vv,&
                                             h2b_vvvv,h2b_vovo,h2b_ovvo,h2b_vovv,h2b_oovv,&
                                             h2c_vvvv,h2c_oooo,h2c_voov,h2c_ooov,h2c_vovv,h2c_oovv,&
                                             x2b_voo,x2b_ovo,x2b_vvv,&
                                             n3aab,n3abb,&
                                             noa,nua,nob,nub)

                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aab, n3abb
                  ! Occupied block indices
                  integer, intent(in) :: j, k
                  ! Input R and T arrays
                  real(kind=8), intent(in) :: l1a(nua)
                  real(kind=8), intent(in) :: l2b(nua,nub,nob) 
                  integer, intent(in) :: l3b_excits(n3aab,5), l3c_excits(n3abb,5)
                  real(kind=8), intent(in) :: l3b_amps(n3aab), l3c_amps(n3abb)
                  ! Input H  and X arrays
                  real(kind=8), intent(in) :: h1b_ov(nob,nub)
                  real(kind=8), intent(in) :: h1a_vv(nua,nua)
                  real(kind=8), intent(in) :: h1b_oo(nob,nob)
                  real(kind=8), intent(in) :: h1b_vv(nub,nub)
                  real(kind=8), intent(in) :: h2b_vvvv(nua,nub,nua,nub)
                  real(kind=8), intent(in) :: h2b_vovo(nua,nob,nua,nob)
                  real(kind=8), intent(in) :: h2b_ovvo(noa,nub,nua,nob)
                  real(kind=8), intent(in) :: h2b_vovv(nua,nob,nua,nub)
                  real(kind=8), intent(in) :: h2b_oovv(noa,nob,nua,nub)
                  real(kind=8), intent(in) :: h2c_vvvv(nub,nub,nub,nub)
                  real(kind=8), intent(in) :: h2c_oooo(nob,nob,nob,nob)
                  real(kind=8), intent(in) :: h2c_voov(nub,nob,nob,nub)
                  real(kind=8), intent(in) :: h2c_ooov(nob,nob,nob,nub)
                  real(kind=8), intent(in) :: h2c_vovv(nub,nob,nub,nub)
                  real(kind=8), intent(in) :: h2c_oovv(nob,nob,nub,nub)
                  real(kind=8), intent(in) :: x2b_voo(nua,nob,nob)
                  real(kind=8), intent(in) :: x2b_ovo(noa,nub,nob)
                  real(kind=8), intent(in) :: x2b_vvv(nua,nub,nub)
                  ! Output and Inout variables
                  real(kind=8), intent(out) :: resid(nua,nub,nub)
                  ! Local variables
                  integer, allocatable :: excits_buff(:,:), l3c_excits_copy(:,:)
                  real(kind=8), allocatable :: amps_buff(:), l3c_amps_copy(:)
                  integer, allocatable :: idx_table4(:,:,:,:)
                  integer, allocatable :: idx_table3(:,:,:)
                  integer, allocatable :: idx_table2(:,:)
                  integer, allocatable :: loc_arr(:,:)
                  real(kind=8) :: r_amp, hmatel, res_mm23
                  integer :: a, b, c, d, i, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc
                  ! Q space array
                  logical(kind=1) :: qspace(nua,nub,nub)
                  
                  ! Zero the container that holds H*R
                  resid = 0.0d0
                  
                  ! copy over l3c_amps and l3c_excits
                  allocate(l3c_amps_copy(n3abb),l3c_excits_copy(n3abb,5))
                  l3c_amps_copy(:) = l3c_amps(:)
                  l3c_excits_copy(:,:) = l3c_excits(:,:)
                  
                  ! reorder l3c into (j,k) order
                  nloc = noa*nob
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table2(nob,nob))
                  call get_index_table2(idx_table2, (/1,nob-1/), (/-1,nob/), nob, nob)
                  call sort2(l3c_excits_copy, l3c_amps_copy, loc_arr, idx_table2, (/4,5/), nob, nob, nloc, n3abb)
                  ! Construct Q space for block (j,k)
                  qspace = .true.
                  idx = idx_table2(j,k)
                  if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        a = l3c_excits_copy(jdet,1); b = l3c_excits_copy(jdet,2); c = l3c_excits_copy(jdet,3);
                        qspace(a,b,c) = .false.
                     end do
                  end if
                  deallocate(loc_arr,idx_table2)

                  !!! diagram 1: -A(jk) h1b(mj)*l3c(abcmk)
                  ! allocate new sorting arrays
                  nloc = nua*nub*(nub - 1)/2*(nob - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nub,nub,nua,nob))
                  !!! SB: (2,3,1,5) !!!
                  call get_index_table4(idx_table4, (/1,nub-1/), (/-1,nub/), (/1,nua/), (/2,nob/), nub, nub, nua, nob)
                  call sort4(l3c_excits_copy, l3c_amps_copy, loc_arr, idx_table4, (/2,3,1,5/), nub, nub, nua, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits_copy,&
                  !$omp l3c_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1B_oo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(b,c,a,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3c_excits_copy(jdet,4);
                        ! compute < ab~c~j~k~ | h1b(oo) | ab~c~m~k~ >
                        hmatel = -h1b_oo(j,m)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     ! (jk)
                     idx = idx_table4(b,c,a,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3c_excits_copy(jdet,4);
                        ! compute < ab~c~j~k~ | h1b(oo) | ab~c~m~j~ >
                        hmatel = h1b_oo(k,m)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,1,4) !!!
                  call get_index_table4(idx_table4, (/1,nub-1/), (/-1,nub/), (/1,nua/), (/1,nob-1/), nub, nub, nua, nob)
                  call sort4(l3c_excits_copy, l3c_amps_copy, loc_arr, idx_table4, (/2,3,1,4/), nub, nub, nua, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits_copy,&
                  !$omp l3c_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1B_oo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(b,c,a,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        n = l3c_excits_copy(jdet,5);
                        ! compute < ab~c~j~k~ | h1b(oo) | ab~c~j~n~ >
                        hmatel = -h1b_oo(k,n)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     ! (jk)
                     idx = idx_table4(b,c,a,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        n = l3c_excits_copy(jdet,5);
                        ! compute < ab~c~j~k~ | h1b(oo) | ab~c~k~n~ >
                        hmatel = h1b_oo(j,n)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 2: A(bc) h1b(be)*l3c(aecjk)
                  ! allocate new sorting arrays
                  nloc = nua*(nub - 1)*nob*(nob - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nob,nob,nua,nub))
                  !!! SB: (4,5,1,3) !!!
                  call get_index_table4(idx_table4, (/1,nob-1/), (/-1,nob/), (/1,nua/), (/2,nub/), nob, nob, nua, nub)
                  call sort4(l3c_excits_copy, l3c_amps_copy, loc_arr, idx_table4, (/4,5,1,3/), nob, nob, nua, nub, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits_copy,&
                  !$omp l3c_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1B_vv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(j,k,a,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits_copy(jdet,2);
                        ! compute < ab~c~j~k~ | h1b(vv) | ae~c~j~k~ >
                        hmatel = h1b_vv(e,b)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     ! (bc)
                     idx = idx_table4(j,k,a,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits_copy(jdet,2);
                        ! compute < ab~c~j~k~ | h1b(vv) | ae~b~j~k~ >
                        hmatel = -h1b_vv(e,c)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,1,2) !!!
                  call get_index_table4(idx_table4, (/1,nob-1/), (/-1,nob/), (/1,nua/), (/1,nub-1/), nob, nob, nua, nub)
                  call sort4(l3c_excits_copy, l3c_amps_copy, loc_arr, idx_table4, (/4,5,1,2/), nob, nob, nua, nub, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits_copy,&
                  !$omp l3c_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1B_vv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(j,k,a,b)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits_copy(jdet,3);
                        ! compute < ab~c~j~k~ | h1b(vv) | ab~f~j~k~ >
                        hmatel = h1b_vv(f,c)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     ! (bc)
                     idx = idx_table4(j,k,a,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits_copy(jdet,3);
                        ! compute < ab~c~j~k~ | h1b(vv) | ac~f~j~k~ >
                        hmatel = -h1b_vv(f,b)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)
                 
                  !!! diagram 3: h1a(ae)*l3c(ebcjk)
                  ! allocate new sorting arrays
                  nloc = nub*(nub - 1)/2*nob*(nob - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table4(nub,nub,nob,nob))
                  !!! SB: (2,3,4,5) !!!
                  call get_index_table4(idx_table4, (/1,nub-1/), (/-1,nub/), (/1,nob-1/), (/-1,nob/), nub, nub, nob, nob)
                  call sort4(l3c_excits_copy, l3c_amps_copy, loc_arr, idx_table4, (/2,3,4,5/), nub, nub, nob, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits_copy,&
                  !$omp l3c_amps_copy,&
                  !$omp loc_arr,idx_table4,&
                  !$omp H1A_vv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table4(b,c,j,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3c_excits_copy(jdet,1);
                        ! compute < ab~c~j~k~ | h1a(vv) | db~c~j~k~ >
                        hmatel = h1a_vv(d,a)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table4)

                  !!! diagram 4: h2c(mnjk)*l3c(abcmn)
                  ! allocate new sorting arrays
                  nloc = nua*nub*(nub - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nub,nub,nua))
                  !!! SB: (2,3,1) !!!
                  call get_index_table3(idx_table3, (/1,nub-1/), (/-1,nub/), (/1,nua/), nub, nub, nua)
                  call sort3(l3c_excits_copy, l3c_amps_copy, loc_arr, idx_table3, (/2,3,1/), nub, nub, nua, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits_copy,&
                  !$omp l3c_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2C_oooo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(b,c,a)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        m = l3c_excits_copy(jdet,4); n = l3c_excits_copy(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(oooo) | ab~c~m~n~ >
                        hmatel = h2c_oooo(j,k,m,n)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 5: A(bc)A(jk) h2b(mbej)*l3b(aecmk)
                  ! allocate and copy over l3b arrays
                  allocate(amps_buff(n3aab),excits_buff(n3aab,5))
                  amps_buff(:) = l3b_amps(:)
                  excits_buff(:,:) = l3b_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = (nua - 1)*nub*noa
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nub,nob))
                  !!! SB: (1,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nua-1/), (/1,nub/), (/1,nob/), nua, nub, nob)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/1,3,5/), nua, nub, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits_copy,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(ovvo) | aec~mk~ >
                        hmatel = h2b_ovvo(m,b,e,j)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(ovvo) | aeb~mk~ >
                        hmatel = -h2b_ovvo(m,c,e,j)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(ovvo) | aec~mj~ >
                        hmatel = -h2b_ovvo(m,b,e,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = excits_buff(jdet,2); m = excits_buff(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(ovvo) | aeb~mj~ >
                        hmatel = h2b_ovvo(m,c,e,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,5) !!!
                  call get_index_table3(idx_table3, (/2,nua/), (/1,nub/), (/1,nob/), nua, nub, nob)
                  call sort3(excits_buff, amps_buff, loc_arr, idx_table3, (/2,3,5/), nua, nub, nob, nloc, n3aab)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits_copy,excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_ovvo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(ovvo) | dac~mk~ >
                        hmatel = -h2b_ovvo(m,b,d,j)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(ovvo) | dab~mk~ >
                        hmatel = h2b_ovvo(m,c,d,j)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(ovvo) | dac~mj~ >
                        hmatel = h2b_ovvo(m,b,d,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = excits_buff(jdet,1); m = excits_buff(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(ovvo) | dab~mj~ >
                        hmatel = -h2b_ovvo(m,c,d,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3,excits_buff,amps_buff)

                  !!! diagram 6: A(bc)A(jk) h2c(bmje) * l3c(aecmk) 
                  ! allocate new sorting arrays
                  nloc = nua*(nub - 1)*(nob - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nua,nub,nob))
                  !!! SB: (1,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/2,nub/), (/2,nob/), nua, nub, nob)
                  call sort3(l3c_excits_copy, l3c_amps_copy, loc_arr, idx_table3, (/1,3,5/), nua, nub, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits_copy,&
                  !$omp l3c_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits_copy(jdet,2); m = l3c_excits_copy(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~c~m~k~ >
                        hmatel = h2c_voov(b,m,j,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     ! (bc)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits_copy(jdet,2); m = l3c_excits_copy(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~b~m~k~ >
                        hmatel = -h2c_voov(c,m,j,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits_copy(jdet,2); m = l3c_excits_copy(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~c~m~j~ >
                        hmatel = -h2c_voov(b,m,k,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits_copy(jdet,2); m = l3c_excits_copy(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~b~m~j~ >
                        hmatel = h2c_voov(c,m,k,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,5) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/1,nub-1/), (/2,nob/), nua, nub, nob)
                  call sort3(l3c_excits_copy, l3c_amps_copy, loc_arr, idx_table3, (/1,2,5/), nua, nub, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits_copy,&
                  !$omp l3c_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits_copy(jdet,3); m = l3c_excits_copy(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ab~f~m~k~ >
                        hmatel = h2c_voov(c,m,j,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     ! (bc)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits_copy(jdet,3); m = l3c_excits_copy(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ac~f~m~k~ >
                        hmatel = -h2c_voov(b,m,j,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits_copy(jdet,3); m = l3c_excits_copy(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ab~f~m~j~ >
                        hmatel = -h2c_voov(c,m,k,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits_copy(jdet,3); m = l3c_excits_copy(jdet,4);
                        ! compute < ab~c~j~k~ | h2c(voov) | ac~f~m~j~ >
                        hmatel = h2c_voov(b,m,k,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,3,4) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/2,nub/), (/1,nob-1/), nua, nub, nob)
                  call sort3(l3c_excits_copy, l3c_amps_copy, loc_arr, idx_table3, (/1,3,4/), nua, nub, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits_copy,&
                  !$omp l3c_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits_copy(jdet,2); n = l3c_excits_copy(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~c~j~n~ >
                        hmatel = h2c_voov(b,n,k,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     ! (bc)
                     idx = idx_table3(a,b,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits_copy(jdet,2); n = l3c_excits_copy(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~b~j~n~ >
                        hmatel = -h2c_voov(c,n,k,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits_copy(jdet,2); n = l3c_excits_copy(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~c~k~n~ >
                        hmatel = -h2c_voov(b,n,j,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits_copy(jdet,2); n = l3c_excits_copy(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ae~b~k~n~ >
                        hmatel = h2c_voov(c,n,j,e)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4) !!!
                  call get_index_table3(idx_table3, (/1,nua/), (/1,nub-1/), (/1,nob-1/), nua, nub, nob)
                  call sort3(l3c_excits_copy, l3c_amps_copy, loc_arr, idx_table3, (/1,2,4/), nua, nub, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits_copy,&
                  !$omp l3c_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2C_voov,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(a,b,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits_copy(jdet,3); n = l3c_excits_copy(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ab~f~j~n~ >
                        hmatel = h2c_voov(c,n,k,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     ! (bc)
                     idx = idx_table3(a,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits_copy(jdet,3); n = l3c_excits_copy(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ac~f~j~n~ >
                        hmatel = -h2c_voov(b,n,k,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table3(a,b,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits_copy(jdet,3); n = l3c_excits_copy(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ab~f~k~n~ >
                        hmatel = -h2c_voov(c,n,j,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table3(a,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        f = l3c_excits_copy(jdet,3); n = l3c_excits_copy(jdet,5);
                        ! compute < ab~c~j~k~ | h2c(voov) | ac~f~k~n~ >
                        hmatel = h2c_voov(b,n,j,f)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 7: -A(jk) h2b(amej)*l3c(ebcmk)
                  ! allocate new sorting arrays
                  nloc = nub*(nub - 1)/2*(nob - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nub,nub,nob))
                  !!! SB: (2,3,5) !!!
                  call get_index_table3(idx_table3, (/1,nub-1/), (/-1,nub/), (/2,nob/), nub, nub, nob)
                  call sort3(l3c_excits_copy, l3c_amps_copy, loc_arr, idx_table3, (/2,3,5/), nub, nub, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits_copy,&
                  !$omp l3c_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vovo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(b,c,k)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3c_excits_copy(jdet,1); m = l3c_excits_copy(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(vovo) | db~c~m~k~ >
                        hmatel = -h2b_vovo(a,m,d,j)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     ! (jk)
                     idx = idx_table3(b,c,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3c_excits_copy(jdet,1); m = l3c_excits_copy(jdet,4);
                        ! compute < ab~c~j~k~ | h2b(vovo) | db~c~m~j~ >
                        hmatel = h2b_vovo(a,m,d,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (2,3,4) !!!
                  call get_index_table3(idx_table3, (/1,nub-1/), (/-1,nub/), (/1,nob-1/), nub, nub, nob)
                  call sort3(l3c_excits_copy, l3c_amps_copy, loc_arr, idx_table3, (/2,3,4/), nub, nub, nob, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits_copy,&
                  !$omp l3c_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vovo,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(b,c,j)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3c_excits_copy(jdet,1); n = l3c_excits_copy(jdet,5);
                        ! compute < ab~c~j~k~ | h2b(vovo) | db~c~j~n~ >
                        hmatel = -h2b_vovo(a,n,d,k)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     ! (jk)
                     idx = idx_table3(b,c,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3c_excits_copy(jdet,1); n = l3c_excits_copy(jdet,5);
                        ! compute < ab~c~j~k~ | h2b(vovo) | db~c~k~n~ >
                        hmatel = h2b_vovo(a,n,d,j)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 8: A(bc) h2b(abef)*l3c(efcjk)
                  ! allocate new sorting arrays
                  nloc = nob*(nob - 1)/2*(nub - 1)
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nob,nob,nub))
                  !!! SB: (4,5,3) !!!
                  call get_index_table3(idx_table3, (/1,nob-1/), (/-1,nob/), (/2,nub/), nob, nob, nub)
                  call sort3(l3c_excits_copy, l3c_amps_copy, loc_arr, idx_table3, (/4,5,3/), nob, nob, nub, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits_copy,&
                  !$omp l3c_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vvvv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(j,k,c)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3c_excits_copy(jdet,1); e = l3c_excits_copy(jdet,2);
                        ! compute < ab~c~j~k~ | h2b(vvvv) | de~c~j~k~ >
                        hmatel = h2b_vvvv(d,e,a,b)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     ! (bc)
                     idx = idx_table3(j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3c_excits_copy(jdet,1); e = l3c_excits_copy(jdet,2);
                        ! compute < ab~c~j~k~ | h2b(vvvv) | de~b~j~k~ >
                        hmatel = -h2b_vvvv(d,e,a,c)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,2) !!!
                  call get_index_table3(idx_table3, (/1,nob-1/), (/-1,nob/), (/1,nub-1/), nob, nob, nub)
                  call sort3(l3c_excits_copy, l3c_amps_copy, loc_arr, idx_table3, (/4,5,2/), nob, nob, nub, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits_copy,&
                  !$omp l3c_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2B_vvvv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(j,k,b)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3c_excits_copy(jdet,1); f = l3c_excits_copy(jdet,3);
                        ! compute < ab~c~j~k~ | h2b(vvvv) | db~f~j~k~ >
                        hmatel = h2b_vvvv(d,f,a,c)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     ! (bc)
                     idx = idx_table3(j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        d = l3c_excits_copy(jdet,1); f = l3c_excits_copy(jdet,3);
                        ! compute < ab~c~j~k~ | h2b(vvvv) | dc~f~j~k~ >
                        hmatel = -h2b_vvvv(d,f,a,b)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                     end if
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!! diagram 9: 1/2 h2c(bcef)*l3c(aefjk)
                  ! allocate new sorting arrays
                  nloc = nua*nob*(nob - 1)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table3(nob,nob,nua))
                  !!! SB: (4,5,1) !!!
                  call get_index_table3(idx_table3, (/1,nob-1/), (/-1,nob/), (/1,nua/), nob, nob, nua)
                  call sort3(l3c_excits_copy, l3c_amps_copy, loc_arr, idx_table3, (/4,5,1/), nob, nob, nua, nloc, n3abb)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits_copy,&
                  !$omp l3c_amps_copy,&
                  !$omp loc_arr,idx_table3,&
                  !$omp H2C_vvvv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                     ! (1)
                     idx = idx_table3(j,k,a)
                     do jdet = loc_arr(1,idx),loc_arr(2,idx)
                        e = l3c_excits_copy(jdet,2); f = l3c_excits_copy(jdet,3);
                        ! compute < ab~c~j~k~ | h2b(vvvv) | ae~f~j~k~ >
                        hmatel = h2c_vvvv(e,f,b,c)
                        resid(a,b,c) = resid(a,b,c) + hmatel * l3c_amps_copy(jdet)
                     end do
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table3)

                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp l3c_excits_copy,&
                  !$omp l1a,l2b,&
                  !$omp h2b_vovv,h2b_oovv,&
                  !$omp h2c_ooov,h2c_vovv,h2c_oovv,&
                  !$omp x2b_voo,x2b_ovo,x2b_vvv,&
                  !$omp noa,nua,nob,nub,n3abb),&
                  !$omp do schedule(static)
                  do a=1,nua; do b=1,nub; do c=b+1,nub;
                     if (.not. qspace(a,b,c)) cycle
                      ! zero out value
                      res_mm23 = 0.0d0
                      ! l1a(a)*h2c_oovv(jkbc)
                      res_mm23 = res_mm23 + l1a(a)*h2c_oovv(j,k,b,c) ! (1)
                      ! A(bc)A(jk) l2b(abj)*h1b_ov(kc)
                      res_mm23 = res_mm23 + l2b(a,b,j)*h1b_ov(k,c) ! (1)
                      res_mm23 = res_mm23 - l2b(a,c,j)*h1b_ov(k,b) ! (bc)
                      res_mm23 = res_mm23 - l2b(a,b,k)*h1b_ov(j,c) ! (jk)
                      res_mm23 = res_mm23 + l2b(a,c,k)*h1b_ov(j,b) ! (bc)(jk)
                      do m = 1,noa
                         ! -A(bc)A(jk) x2b_ovo(mck)*h2b_oovv(mjab) 
                         res_mm23 = res_mm23 - x2b_ovo(m,c,k)*h2b_oovv(m,j,a,b) ! (1)
                         res_mm23 = res_mm23 + x2b_ovo(m,b,k)*h2b_oovv(m,j,a,c) ! (bc)
                         res_mm23 = res_mm23 + x2b_ovo(m,c,j)*h2b_oovv(m,k,a,b) ! (jk)
                         res_mm23 = res_mm23 - x2b_ovo(m,b,j)*h2b_oovv(m,k,a,c) ! (bc)(jk)
                      end do
                      do m = 1,nob
                         ! A(bc) -l2b(abm)*h2c_ooov(jkmc)
                         res_mm23 = res_mm23 - l2b(a,b,m)*h2c_ooov(j,k,m,c) ! (1)
                         res_mm23 = res_mm23 + l2b(a,c,m)*h2c_ooov(j,k,m,b) ! (bc)
                         ! A(jk) -x2b_voo(ajm)*h2c_oovv(mkbc)
                         res_mm23 = res_mm23 - x2b_voo(a,j,m)*h2c_oovv(m,k,b,c) ! (1)
                         res_mm23 = res_mm23 + x2b_voo(a,k,m)*h2c_oovv(m,j,b,c) ! (jk)
                      end do
                      do e = 1,nua
                         ! A(jk)A(bc) l2b(eck)*h2b_vovv(ejab)
                         res_mm23 = res_mm23 + l2b(e,c,k)*h2b_vovv(e,j,a,b) ! (1)
                         res_mm23 = res_mm23 - l2b(e,b,k)*h2b_vovv(e,j,a,c) ! (bc)
                         res_mm23 = res_mm23 - l2b(e,c,j)*h2b_vovv(e,k,a,b) ! (jk)
                         res_mm23 = res_mm23 + l2b(e,b,j)*h2b_vovv(e,k,a,c) ! (bc)(jk)
                      end do
                      do e = 1,nub
                         ! A(jk) l2b(aej)*h2c_vovv(ekbc)
                         res_mm23 = res_mm23 + l2b(a,e,j)*h2c_vovv(e,k,b,c) ! (1)
                         res_mm23 = res_mm23 - l2b(a,e,k)*h2c_vovv(e,j,b,c) ! (jk)
                         ! A(bc) x2b_vvv(aeb)*h2c_oovv(jkec)
                         res_mm23 = res_mm23 + x2b_vvv(a,e,b)*h2c_oovv(j,k,e,c) ! (1)
                         res_mm23 = res_mm23 - x2b_vvv(a,e,c)*h2c_oovv(j,k,e,b) ! (bc)
                      end do
                      resid(a,b,c) = resid(a,b,c) + res_mm23
                  end do; end do; end do;
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate l3c array copies
                  deallocate(l3c_excits_copy,l3c_amps_copy)
                  ! antisymmetrize m(abc)
                  do a=1,nua
                     do b=1,nub
                        do c=b+1,nub
                           resid(a,c,b) = -resid(a,b,c)
                        end do
                     end do
                  end do
              end subroutine build_leftamps3c_jk

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!! SORTING FUNCTIONS !!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

               subroutine get_index_table2(idx_table, rng1, rng2, n1, n2)
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

              subroutine sort2(excits, amps, loc_arr, idx_table, idims, n1, n2, nloc, n3p, x1a)

                    integer, intent(in) :: n1, n2, nloc, n3p
                    integer, intent(in) :: idims(2)
                    integer, intent(in) :: idx_table(n1,n2)
      
                    integer, intent(inout) :: loc_arr(2,nloc)
                    integer, intent(inout) :: excits(n3p,5)
                    real(kind=8), intent(inout) :: amps(n3p)
                    real(kind=8), intent(inout), optional :: x1a(n3p)
      
                    integer :: idet
                    integer :: p, q
                    integer :: p1, q1, p2, q2
                    integer :: pq1, pq2
                    integer, allocatable :: temp(:), idx(:)
      
                    allocate(temp(n3p),idx(n3p))
                    do idet = 1, n3p
                       p = excits(idet,idims(1)); q = excits(idet,idims(2));
                       temp(idet) = idx_table(p,q)
                    end do
                    call argsort(temp, idx)
                    excits = excits(idx,:)
                    amps = amps(idx)
                    if (present(x1a)) x1a = x1a(idx)
                    deallocate(temp,idx)
      
                    loc_arr(1,:) = 1; loc_arr(2,:) = 0;
                    !!! WARNING: THERE IS A MEMORY LEAK HERE! pqrs2 is used below but is not set if n3p <= 1
                    !if (n3p <= 1) print*, "eomccsdt_p_loops >> WARNING: potential memory leakage in sort4 function. pqrs2 set to -1"
                    if (n3p == 1) then
                       if (excits(1,1)==1 .and. excits(1,2)==1 .and. excits(1,3)==1 .and. excits(1,4)==1 .and. excits(1,5)==1) return
                       p2 = excits(n3p,idims(1)); q2 = excits(n3p,idims(2));
                       pq2 = idx_table(p2,q2)
                    else
                       pq2 = -1
                    end if
                    do idet = 1, n3p-1
                       p1 = excits(idet,idims(1));   q1 = excits(idet,idims(2));
                       p2 = excits(idet+1,idims(1)); q2 = excits(idet+1,idims(2));
                       pq1 = idx_table(p1,q1)
                       pq2 = idx_table(p2,q2)
                       if (pq1 /= pq2) then
                          loc_arr(2,pq1) = idet
                          loc_arr(1,pq2) = idet+1
                       end if
                    end do
                    if (n3p > 1) then
                       loc_arr(2,pq2) = n3p
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

              subroutine sort3(excits, amps, loc_arr, idx_table, idims, n1, n2, n3, nloc, n3p, x1a)

                    integer, intent(in) :: n1, n2, n3, nloc, n3p
                    integer, intent(in) :: idims(3)
                    integer, intent(in) :: idx_table(n1,n2,n3)
      
                    integer, intent(inout) :: loc_arr(2,nloc)
                    integer, intent(inout) :: excits(n3p,5)
                    real(kind=8), intent(inout) :: amps(n3p)
                    real(kind=8), intent(inout), optional :: x1a(n3p)
      
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
                    amps = amps(idx)
                    if (present(x1a)) x1a = x1a(idx)
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

              subroutine sort4(excits, amps, loc_arr, idx_table, idims, n1, n2, n3, n4, nloc, n3p, x1a)

                    integer, intent(in) :: n1, n2, n3, n4, nloc, n3p
                    integer, intent(in) :: idims(4)
                    integer, intent(in) :: idx_table(n1,n2,n3,n4)
      
                    integer, intent(inout) :: loc_arr(2,nloc)
                    integer, intent(inout) :: excits(n3p,5)
                    real(kind=8), intent(inout) :: amps(n3p)
                    real(kind=8), intent(inout), optional :: x1a(n3p)
      
                    integer :: idet
                    integer :: p, q, r, s
                    integer :: p1, q1, r1, s1, p2, q2, r2, s2
                    integer :: pqrs1, pqrs2
                    integer, allocatable :: temp(:), idx(:)
      
                    allocate(temp(n3p),idx(n3p))
                    do idet = 1, n3p
                       p = excits(idet,idims(1)); q = excits(idet,idims(2)); r = excits(idet,idims(3)); s = excits(idet,idims(4))
                       temp(idet) = idx_table(p,q,r,s)
                    end do
                    call argsort(temp, idx)
                    excits = excits(idx,:)
                    amps = amps(idx)
                    if (present(x1a)) x1a = x1a(idx)
                    deallocate(temp,idx)
      
                    loc_arr(1,:) = 1; loc_arr(2,:) = 0;
                    !!! WARNING: THERE IS A MEMORY LEAK HERE! pqrs2 is used below but is not set if n3p <= 1
                    !if (n3p <= 1) print*, "eomccsdt_p_loops >> WARNING: potential memory leakage in sort4 function. pqrs2 set to -1"
                    if (n3p == 1) then
                       if (excits(1,1)==1 .and. excits(1,2)==1 .and. excits(1,3)==1 .and. excits(1,4)==1 .and. excits(1,5)==1) return
                       p2 = excits(n3p,idims(1)); q2 = excits(n3p,idims(2)); r2 = excits(n3p,idims(3)); s2 = excits(n3p,idims(4))
                       pqrs2 = idx_table(p2,q2,r2,s2)
                    else
                       pqrs2 = -1
                    end if
                    do idet = 1, n3p-1
                       p1 = excits(idet,idims(1));   q1 = excits(idet,idims(2));   r1 = excits(idet,idims(3));   s1 = excits(idet,idims(4))
                       p2 = excits(idet+1,idims(1)); q2 = excits(idet+1,idims(2)); r2 = excits(idet+1,idims(3)); s2 = excits(idet+1,idims(4))
                       pqrs1 = idx_table(p1,q1,r1,s1)
                       pqrs2 = idx_table(p2,q2,r2,s2)
                       if (pqrs1 /= pqrs2) then
                          loc_arr(2,pqrs1) = idet
                          loc_arr(1,pqrs2) = idet+1
                       end if
                    end do
                    !if (n3p > 1) then
                    loc_arr(2,pqrs2) = n3p
                    !end if
              end subroutine sort4

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
     

end module eaccp3_correction
