module eomccsdt_p_loops

      use omp_lib

      implicit none
   
      ! Checklist for converting (H*R3)_C update into (X*T3)_C
      ! [ ] - idet loop is over R3 quantities; idet = 1, n3_r
      ! [ ] - jdet loop is over T3 quantities; jdet = loc_arr(idx,1), loc_arr(idx,2)
      ! [ ] - replace n3_r parameter in sort4 to n3_t
      ! [ ] - remove resid from sort4 function when sorting T3 excitations

      contains

               subroutine build_hr_1a(x1a,&
                                      r3a_excits, r3b_excits, r3c_excits,&
                                      r3a_amps, r3b_amps, r3c_amps,&
                                      h2a_oovv, h2b_oovv, h2c_oovv,&
                                      n3aaa, n3aab, n3abb,&
                                      noa, nua, nob, nub)

                      integer, intent(in) :: noa, nua, nob, nub, n3aaa, n3aab, n3abb
                      integer, intent(in) :: r3a_excits(6,n3aaa), r3b_excits(6,n3aab), r3c_excits(6,n3abb)
                      real(kind=8), intent(in) :: r3a_amps(n3aaa), r3b_amps(n3aab), r3c_amps(n3abb)
                      real(kind=8), intent(in) :: h2a_oovv(1:noa,1:noa,1:nua,1:nua),&
                                                  h2b_oovv(1:noa,1:nob,1:nua,1:nub),&
                                                  h2c_oovv(1:nob,1:nob,1:nub,1:nub)
                      
                      real(kind=8), intent(inout) :: x1a(1:nua,1:noa)
                      !f2py intent(in,out) :: x1a(0:nua-1,0:noa-1)
                      
                      integer :: i, a, m, n, e, f, idet
                      real(kind=8) :: denom, val, r_amp
                      
                      ! compute < ia | (H(2) * R3)_C | 0 >
                      do idet = 1, n3aaa
                          r_amp = r3a_amps(idet)
                          ! A(a/ef)A(i/mn) h2a(mnef) * r3a(aefimn)
                          a = r3a_excits(1,idet); e = r3a_excits(2,idet); f = r3a_excits(3,idet);
                          i = r3a_excits(4,idet); m = r3a_excits(5,idet); n = r3a_excits(6,idet);
                          x1a(a,i) = x1a(a,i) + h2a_oovv(m,n,e,f) * r_amp ! (1)
                          x1a(e,i) = x1a(e,i) - h2a_oovv(m,n,a,f) * r_amp ! (ae)
                          x1a(f,i) = x1a(f,i) - h2a_oovv(m,n,e,a) * r_amp ! (af)
                          x1a(a,m) = x1a(a,m) - h2a_oovv(i,n,e,f) * r_amp ! (im)
                          x1a(e,m) = x1a(e,m) + h2a_oovv(i,n,a,f) * r_amp ! (ae)(im)
                          x1a(f,m) = x1a(f,m) + h2a_oovv(i,n,e,a) * r_amp ! (af)(im)
                          x1a(a,n) = x1a(a,n) - h2a_oovv(m,i,e,f) * r_amp ! (in)
                          x1a(e,n) = x1a(e,n) + h2a_oovv(m,i,a,f) * r_amp ! (ae)(in)
                          x1a(f,n) = x1a(f,n) + h2a_oovv(m,i,e,a) * r_amp ! (af)(in)
                      end do
                      do idet = 1, n3aab
                          r_amp = r3b_amps(idet)
                          ! A(ae)A(im) h2b(mnef) * r3b(aefimn)
                          a = r3b_excits(1,idet); e = r3b_excits(2,idet); f = r3b_excits(3,idet);
                          i = r3b_excits(4,idet); m = r3b_excits(5,idet); n = r3b_excits(6,idet);
                          x1a(a,i) = x1a(a,i) + h2b_oovv(m,n,e,f) * r_amp ! (1)
                          x1a(e,i) = x1a(e,i) - h2b_oovv(m,n,a,f) * r_amp ! (ae)
                          x1a(a,m) = x1a(a,m) - h2b_oovv(i,n,e,f) * r_amp ! (im)
                          x1a(e,m) = x1a(e,m) + h2b_oovv(i,n,a,f) * r_amp ! (ae)(im)
                      end do
                      do idet = 1, n3abb
                          r_amp = r3c_amps(idet)
                          ! h2c(mnef) * r3c(aefimn)
                          a = r3c_excits(1,idet); e = r3c_excits(2,idet); f = r3c_excits(3,idet);
                          i = r3c_excits(4,idet); m = r3c_excits(5,idet); n = r3c_excits(6,idet);
                          x1a(a,i) = x1a(a,i) + h2c_oovv(m,n,e,f) * r_amp ! (1)
                      end do
              end subroutine build_hr_1a
         
              subroutine build_hr_1b(x1b,&
                                     r3b_excits, r3c_excits, r3d_excits,&
                                     r3b_amps, r3c_amps, r3d_amps,&
                                     h2a_oovv, h2b_oovv, h2c_oovv,&
                                     n3aab, n3abb, n3bbb,&
                                     noa, nua, nob, nub)

                      integer, intent(in) :: noa, nua, nob, nub, n3aab, n3abb, n3bbb
                      integer, intent(in) :: r3b_excits(6,n3aab), r3c_excits(6,n3abb), r3d_excits(6,n3bbb)
                      real(kind=8), intent(in) :: r3b_amps(n3aab), r3c_amps(n3abb), r3d_amps(n3bbb)
                      real(kind=8), intent(in) :: h2a_oovv(1:noa,1:noa,1:nua,1:nua),&
                                                  h2b_oovv(1:noa,1:nob,1:nua,1:nub),&
                                                  h2c_oovv(1:nob,1:nob,1:nub,1:nub)

                      real(kind=8), intent(inout) :: x1b(1:nub,1:nob)
                      !f2py intent(in,out) :: x1b(0:nub-1,0:nob-1)
                      
                      integer :: i, a, m, n, e, f, idet
                      real(kind=8) :: denom, val, r_amp
                      
                      ! compute < i~a~ | (H(2) * R3)_C | 0 >
                      do idet = 1, n3aab
                          r_amp = r3b_amps(idet)
                          ! h2a(mnef) * r3b(efamni)
                          e = r3b_excits(1,idet); f = r3b_excits(2,idet); a = r3b_excits(3,idet);
                          m = r3b_excits(4,idet); n = r3b_excits(5,idet); i = r3b_excits(6,idet);
                          x1b(a,i) = x1b(a,i) + h2a_oovv(m,n,e,f) * r_amp ! (1)
                      end do
                      do idet = 1, n3abb
                          r_amp = r3c_amps(idet)
                          ! A(af)A(in) h2b(mnef) * r3c(efamni)
                          e = r3c_excits(1,idet); f = r3c_excits(2,idet); a = r3c_excits(3,idet);
                          m = r3c_excits(4,idet); n = r3c_excits(5,idet); i = r3c_excits(6,idet);
                          x1b(a,i) = x1b(a,i) + h2b_oovv(m,n,e,f) * r_amp ! (1)
                          x1b(f,i) = x1b(f,i) - h2b_oovv(m,n,e,a) * r_amp ! (af)
                          x1b(a,n) = x1b(a,n) - h2b_oovv(m,i,e,f) * r_amp ! (in)
                          x1b(f,n) = x1b(f,n) + h2b_oovv(m,i,e,a) * r_amp ! (af)(in)
                      end do
                      do idet = 1, n3bbb
                          r_amp = r3d_amps(idet)
                          ! A(a/ef)A(i/mn) h2c(mnef) * r3d(aefimn)
                          a = r3d_excits(1,idet); e = r3d_excits(2,idet); f = r3d_excits(3,idet);
                          i = r3d_excits(4,idet); m = r3d_excits(5,idet); n = r3d_excits(6,idet);
                          x1b(a,i) = x1b(a,i) + h2c_oovv(m,n,e,f) * r_amp ! (1)
                          x1b(e,i) = x1b(e,i) - h2c_oovv(m,n,a,f) * r_amp ! (ae)
                          x1b(f,i) = x1b(f,i) - h2c_oovv(m,n,e,a) * r_amp ! (af)
                          x1b(a,m) = x1b(a,m) - h2c_oovv(i,n,e,f) * r_amp ! (im)
                          x1b(e,m) = x1b(e,m) + h2c_oovv(i,n,a,f) * r_amp ! (ae)(im)
                          x1b(f,m) = x1b(f,m) + h2c_oovv(i,n,e,a) * r_amp ! (af)(im)
                          x1b(a,n) = x1b(a,n) - h2c_oovv(m,i,e,f) * r_amp ! (in)
                          x1b(e,n) = x1b(e,n) + h2c_oovv(m,i,a,f) * r_amp ! (ae)(in)
                          x1b(f,n) = x1b(f,n) + h2c_oovv(m,i,e,a) * r_amp ! (af)(in)
                      end do
              end subroutine build_hr_1b
 
              subroutine build_hr_3a(resid,&
                                     r2a,&
                                     r3a_amps, r3a_excits,&
                                     r3b_amps, r3b_excits,&
                                     t2a,&
                                     t3a_amps, t3a_excits,&
                                     t3b_amps, t3b_excits,&
                                     h1a_oo, h1a_vv,&
                                     h2a_oooo, h2a_vooo, h2a_oovv,&
                                     h2a_voov, h2a_vvov, h2a_vvvv,&
                                     h2b_voov,&
                                     x1a_oo, x1a_vv,&
                                     x2a_oooo, x2a_vooo, x2a_oovv,&
                                     x2a_voov, x2a_vvov, x2a_vvvv,&
                                     x2b_voov,&
                                     n3aaa_r, n3aab_r,&
                                     n3aaa_t, n3aab_t,&
                                     noa, nua, nob, nub)
                  ! Input dimension variables
                  integer, intent(in) :: noa, nua, nob, nub
                  integer, intent(in) :: n3aaa_r, n3aaa_t, n3aab_r, n3aab_t
                  ! Input R and T arrays
                  real(kind=8), intent(in) :: r2a(nua,nua,noa,noa), t2a(nua,nua,noa,noa)
                  integer, intent(in) :: r3b_excits(6,n3aab_r), t3b_excits(6,n3aab_t)
                  integer, intent(in) :: t3a_excits(6,n3aaa_t) 
                  real(kind=8), intent(in) :: r3b_amps(n3aab_r), t3b_amps(n3aab_t)
                  real(kind=8), intent(in) :: t3a_amps(n3aaa_t)
                  ! Input H and X arrays
                  real(kind=8), intent(in) :: h1a_oo(noa,noa)
                  real(kind=8), intent(in) :: h1a_vv(nua,nua)
                  real(kind=8), intent(in) :: h2a_oooo(noa,noa,noa,noa)
                  real(kind=8), intent(in) :: h2a_vooo(nua,noa,noa,noa)
                  real(kind=8), intent(in) :: h2a_oovv(noa,noa,nua,nua)
                  real(kind=8), intent(in) :: h2a_voov(nua,noa,noa,nua)
                  real(kind=8), intent(in) :: h2a_vvov(nua,nua,noa,nua)
                  real(kind=8), intent(in) :: h2a_vvvv(nua,nua,nua,nua)
                  real(kind=8), intent(in) :: h2b_voov(nua,nob,noa,nub)
                  real(kind=8), intent(in) :: x1a_oo(noa,noa)
                  real(kind=8), intent(in) :: x1a_vv(nua,nua)
                  real(kind=8), intent(in) :: x2a_oooo(noa,noa,noa,noa)
                  real(kind=8), intent(in) :: x2a_vooo(nua,noa,noa,noa)
                  real(kind=8), intent(in) :: x2a_oovv(noa,noa,nua,nua)
                  real(kind=8), intent(in) :: x2a_voov(nua,noa,noa,nua)
                  real(kind=8), intent(in) :: x2a_vvov(nua,nua,noa,nua)
                  real(kind=8), intent(in) :: x2a_vvvv(nua,nua,nua,nua)
                  real(kind=8), intent(in) :: x2b_voov(nua,nob,noa,nub)
                  ! Output and Inout variables
                  real(kind=8), intent(out) :: resid(n3aaa_r)
                  integer, intent(inout) :: r3a_excits(6,n3aaa_r)
                  !f2py intent(in,out) :: r3a_excits(6,0:n3aaa_r-1)
                  real(kind=8), intent(inout) :: r3a_amps(n3aaa_r)
                  !f2py intent(in,out) :: r3a_amps(0:n3aaa_r-1)
                  ! Local variables
                  integer, allocatable :: excits_buff(:,:)
                  real(kind=8), allocatable :: amps_buff(:)
                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)
                  real(kind=8) :: l_amp, hmatel, hmatel1, res
                  integer :: a, b, c, d, i, j, k, l, m, n, e, f, idet, jdet
                  integer :: idx, nloc

                  resid = 0.0d0
                  !!!! diagram 1a: -A(i/jk) h1a(mi) * r3a(abcmjk)
                  !!!! diagram 3a: 1/2 A(i/jk) h2a(mnij) * r3a(abcmnk)
                  ! NOTE: WITHIN THESE LOOPS, H1A(OO) TERMS ARE DOUBLE-COUNTED SO COMPENSATE BY FACTOR OF 1/2
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)*(nua-2)/6*noa
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nua,nua,nua,noa))
                  !!! ABCK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/3,noa/), nua, nua, nua, noa)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table, (/1,2,3,6/), nua, nua, nua, noa, nloc, n3aaa_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_oo,H2A_oooo,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        l = r3a_excits(4,jdet); m = r3a_excits(5,jdet);
                        ! compute < ijkabc | h2a(oooo) | lmkabc >
                        hmatel = h2a_oooo(l,m,i,j)
                        ! compute < ijkabc | h1a(oo) | lmkabc > = -A(ij)A(lm) h1a_oo(l,i) * delta(m,j)
                        hmatel1 = 0.0d0
                        if (m==j) hmatel1 = hmatel1 - h1a_oo(l,i) ! (1)      < ijkabc | h1a(oo) | ljkabc >
                        if (m==i) hmatel1 = hmatel1 + h1a_oo(l,j) ! (ij)     < ijkabc | h1a(oo) | likabc >
                        if (l==j) hmatel1 = hmatel1 + h1a_oo(m,i) ! (lm)     < ijkabc | h1a(oo) | jmkabc >
                        if (l==i) hmatel1 = hmatel1 - h1a_oo(m,j) ! (ij)(lm) < ijkabc | h1a(oo) | imkabc >
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ik)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l = r3a_excits(4,jdet); m = r3a_excits(5,jdet);
                           ! compute < ijkabc | h2a(oooo) | lmiabc >
                           hmatel = -h2a_oooo(l,m,k,j)
                           ! compute < ijkabc | h1a(oo) | lmiabc > = A(jk)A(lm) h1a_oo(l,k) * delta(m,j)
                           hmatel1 = 0.0d0
                           if (m==j) hmatel1 = hmatel1 + h1a_oo(l,k) ! (1)      < ijkabc | h1a(oo) | ljiabc >
                           if (m==k) hmatel1 = hmatel1 - h1a_oo(l,j) ! (jk)     < ijkabc | h1a(oo) | lkiabc >
                           if (l==j) hmatel1 = hmatel1 - h1a_oo(m,k) ! (lm)
                           if (l==k) hmatel1 = hmatel1 + h1a_oo(m,j) ! (jk)(lm)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l = r3a_excits(4,jdet); m = r3a_excits(5,jdet);
                           ! compute < ijkabc | h2a(oooo) | lmjabc >
                           hmatel = -h2a_oooo(l,m,i,k)
                           ! compute < ijkabc | h1a(oo) | lmjabc > = A(ik)A(lm) h1a_oo(l,i) * delta(m,k)
                           hmatel1 = 0.0d0
                           if (m==k) hmatel1 = hmatel1 + h1a_oo(l,i) ! (1)      < ijkabc | h1a(oo) | lkjabc >
                           if (m==i) hmatel1 = hmatel1 - h1a_oo(l,k) ! (ik)
                           if (l==k) hmatel1 = hmatel1 - h1a_oo(m,i) ! (lm)
                           if (l==i) hmatel1 = hmatel1 + h1a_oo(m,k) ! (ik)(lm)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABCI LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/1,noa-2/), nua, nua, nua, noa)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table, (/1,2,3,4/), nua, nua, nua, noa, nloc, n3aaa_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_oo,H2A_oooo,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,i)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        m = r3a_excits(5,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(oooo) | imnabc >
                        hmatel = h2a_oooo(m,n,j,k)
                        ! compute < ijkabc | h1a(oo) | imnabc > = -A(jk)A(mn) h1a_oo(m,j) * delta(n,k)
                        hmatel1 = 0.0d0
                        if (n==k) hmatel1 = hmatel1 - h1a_oo(m,j)  ! < ijkabc | h1a(oo) | imkabc >
                        if (n==j) hmatel1 = hmatel1 + h1a_oo(m,k)
                        if (m==k) hmatel1 = hmatel1 + h1a_oo(n,j)
                        if (m==j) hmatel1 = hmatel1 - h1a_oo(n,k)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           m = r3a_excits(5,jdet); n = r3a_excits(6,jdet);
                           ! compute < ijkabc | h2a(oooo) | jmnabc >
                           hmatel = -h2a_oooo(m,n,i,k)
                           ! compute < ijkabc | h1a(oo) | jmnabc > = A(ik)A(mn) h1a_oo(m,i) * delta(n,k)
                           hmatel1 = 0.0d0
                           if (n==k) hmatel1 = hmatel1 + h1a_oo(m,i)
                           if (n==i) hmatel1 = hmatel1 - h1a_oo(m,k)
                           if (m==k) hmatel1 = hmatel1 - h1a_oo(n,i)
                           if (m==i) hmatel1 = hmatel1 + h1a_oo(n,k)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           m = r3a_excits(5,jdet); n = r3a_excits(6,jdet);
                           ! compute < ijkabc | h2a(oooo) | kmnabc >
                           hmatel = -h2a_oooo(m,n,j,i)
                           ! compute < ijkabc | h1a(oo) | kmnabc > = A(ij)A(mn) h1a_oo(m,j) * delta(n,i)
                           hmatel1 = 0.0d0
                           if (n==i) hmatel1 = hmatel1 - h1a_oo(m,j)
                           if (n==j) hmatel1 = hmatel1 + h1a_oo(m,i)
                           if (m==i) hmatel1 = hmatel1 + h1a_oo(n,j)
                           if (m==j) hmatel1 = hmatel1 - h1a_oo(n,i)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABCJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/2,noa-1/), nua, nua, nua, noa)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table, (/1,2,3,5/), nua, nua, nua, noa, nloc, n3aaa_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_oo,H2A_oooo,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,j)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        l = r3a_excits(4,jdet); n = r3a_excits(6,jdet);
                        ! compute < ijkabc | h2a(oooo) | ljnabc >
                        hmatel = h2a_oooo(l,n,i,k)
                        ! compute < ijkabc | h1a(oo) | ljnabc > = -A(ik)A(ln) h1a_oo(l,i) * delta(n,k)
                        hmatel1 = 0.0d0
                        if (n==k) hmatel1 = hmatel1 - h1a_oo(l,i)
                        if (n==i) hmatel1 = hmatel1 + h1a_oo(l,k)
                        if (l==k) hmatel1 = hmatel1 + h1a_oo(n,i)
                        if (l==i) hmatel1 = hmatel1 - h1a_oo(n,k)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l = r3a_excits(4,jdet); n = r3a_excits(6,jdet);
                           ! compute < ijkabc | h2a(oooo) | linabc >
                           hmatel = -h2a_oooo(l,n,j,k)
                           ! compute < ijkabc | h1a(oo) | linabc > = A(jk)A(ln) h1a_oo(l,j) * delta(n,k)
                           hmatel1 = 0.0d0
                           if (n==k) hmatel1 = hmatel1 + h1a_oo(l,j)
                           if (n==j) hmatel1 = hmatel1 - h1a_oo(l,k)
                           if (l==k) hmatel1 = hmatel1 - h1a_oo(n,j)
                           if (l==j) hmatel1 = hmatel1 + h1a_oo(n,k)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l = r3a_excits(4,jdet); n = r3a_excits(6,jdet);
                           ! compute < ijkabc | h2a(oooo) | lknabc >
                           hmatel = -h2a_oooo(l,n,i,j)
                           ! compute < ijkabc | h1a(oo) | lknabc > = A(ij)A(ln) h1a_oo(l,i) * delta(n,j)
                           hmatel1 = 0.0d0
                           if (n==j) hmatel1 = hmatel1 + h1a_oo(l,i)
                           if (n==i) hmatel1 = hmatel1 - h1a_oo(l,j)
                           if (l==j) hmatel1 = hmatel1 - h1a_oo(n,i)
                           if (l==i) hmatel1 = hmatel1 + h1a_oo(n,j)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  
                  !!!! diagram 2a: A(a/bc) h1a(ae) * r3a(ebcijk)
                  !!!! diagram 4a: 1/2 A(c/ab) h2a(abef) * r3a(ebcijk)
                  ! NOTE: WITHIN THESE LOOPS, H1A(VV) TERMS ARE DOUBLE-COUNTED SO COMPENSATE BY FACTOR OF 1/2
                  ! allocate new sorting arrays
                  nloc = noa*(noa-1)*(noa-2)/6*nua
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(noa,noa,noa,nua))
                  !!! IJKA LOOP !!!
                  call get_index_table(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/1,nua-2/), noa, noa, noa, nua)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table, (/4,5,6,1/), noa, noa, noa, nua, nloc, n3aaa_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_vv,H2A_vvvv,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,a)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = r3a_excits(2,jdet); f = r3a_excits(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkaef >
                        hmatel = h2a_vvvv(b,c,e,f)
                        ! compute < ijkabc | h1a(vv) | ijkaef > = A(bc)A(ef) h1a_vv(b,e) * delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 + h1a_vv(b,e) ! (1)
                        if (b==f) hmatel1 = hmatel1 - h1a_vv(c,e) ! (bc)
                        if (c==e) hmatel1 = hmatel1 - h1a_vv(b,f) ! (ef)
                        if (b==e) hmatel1 = hmatel1 + h1a_vv(c,f) ! (bc)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = r3a_excits(2,jdet); f = r3a_excits(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkbef >
                        hmatel = -h2a_vvvv(a,c,e,f)
                        ! compute < ijkabc | h1a(vv) | ijkbef > = -A(ac)A(ef) h1a_vv(a,e) * delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 - h1a_vv(a,e) ! (1)
                        if (a==f) hmatel1 = hmatel1 + h1a_vv(c,e) ! (ac)
                        if (c==e) hmatel1 = hmatel1 + h1a_vv(a,f) ! (ef)
                        if (a==e) hmatel1 = hmatel1 - h1a_vv(c,f) ! (ac)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = r3a_excits(2,jdet); f = r3a_excits(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkcef >
                        hmatel = -h2a_vvvv(b,a,e,f)
                        ! compute < ijkabc | h1a(vv) | ijkcef > = -A(ab)A(ef) h1a_vv(b,e) * delta(a,f)
                        hmatel1 = 0.0d0
                        if (a==f) hmatel1 = hmatel1 - h1a_vv(b,e) ! (1)
                        if (b==f) hmatel1 = hmatel1 + h1a_vv(a,e) ! (ab)
                        if (a==e) hmatel1 = hmatel1 + h1a_vv(b,f) ! (ef)
                        if (b==e) hmatel1 = hmatel1 - h1a_vv(a,f) ! (ab)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! IJKB LOOP !!!
                  call get_index_table(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/2,nua-1/), noa, noa, noa, nua)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table, (/4,5,6,2/), noa, noa, noa, nua, nloc, n3aaa_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_vv,H2A_vvvv,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,b)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = r3a_excits(1,jdet); f = r3a_excits(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdbf >
                        hmatel = h2a_vvvv(a,c,d,f)
                        ! compute < ijkabc | h1a(vv) | ijkdbf > = A(ac)A(df) h1a_vv(a,d) * delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 + h1a_vv(a,d) ! (1)
                        if (a==f) hmatel1 = hmatel1 - h1a_vv(c,d) ! (ac)
                        if (c==d) hmatel1 = hmatel1 - h1a_vv(a,f) ! (df)
                        if (a==d) hmatel1 = hmatel1 + h1a_vv(c,f) ! (ac)(df)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = r3a_excits(1,jdet); f = r3a_excits(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdaf >
                        hmatel = -h2a_vvvv(b,c,d,f)
                        ! compute < ijkabc | h1a(vv) | ijkdaf > = -A(bc)A(df) h1a_vv(b,d) * delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 - h1a_vv(b,d) ! (1)
                        if (b==f) hmatel1 = hmatel1 + h1a_vv(c,d) ! (bc)
                        if (c==d) hmatel1 = hmatel1 + h1a_vv(b,f) ! (df)
                        if (b==d) hmatel1 = hmatel1 - h1a_vv(c,f) ! (bc)(df)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = r3a_excits(1,jdet); f = r3a_excits(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdcf >
                        hmatel = -h2a_vvvv(a,b,d,f)
                        ! compute < ijkabc | h1a(vv) | ijkdcf > = -A(ab)A(df) h1a_vv(a,d) * delta(b,f)
                        hmatel1 = 0.0d0
                        if (b==f) hmatel1 = hmatel1 - h1a_vv(a,d) ! (1)
                        if (a==f) hmatel1 = hmatel1 + h1a_vv(b,d) ! (ab)
                        if (b==d) hmatel1 = hmatel1 + h1a_vv(a,f) ! (df)
                        if (a==d) hmatel1 = hmatel1 - h1a_vv(b,f) ! (ab)(df)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! IJKC LOOP !!!
                  call get_index_table(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/3,nua/), noa, noa, noa, nua)
                  call sort4(r3a_excits, r3a_amps, loc_arr, idx_table, (/4,5,6,3/), noa, noa, noa, nua, nloc, n3aaa_r, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r3a_excits,&
                  !$omp r3a_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp H1A_vv,H2A_vvvv,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,c)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = r3a_excits(1,jdet); e = r3a_excits(2,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdec >
                        hmatel = h2a_vvvv(a,b,d,e)
                        ! compute < ijkabc | h1a(vv) | ijkdec > = A(ab)A(de) h1a_vv(a,d) * delta(b,e)
                        hmatel1 = 0.0d0
                        if (b==e) hmatel1 = hmatel1 + h1a_vv(a,d) ! (1)
                        if (a==e) hmatel1 = hmatel1 - h1a_vv(b,d) ! (ab)
                        if (b==d) hmatel1 = hmatel1 - h1a_vv(a,e) ! (de)
                        if (a==d) hmatel1 = hmatel1 + h1a_vv(b,e) ! (ab)(de)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = r3a_excits(1,jdet); e = r3a_excits(2,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdea >
                        hmatel = -h2a_vvvv(c,b,d,e)
                        ! compute < ijkabc | h1a(vv) | ijkdea > = -A(bc)A(de) h1a_vv(c,d) * delta(b,e)
                        hmatel1 = 0.0d0
                        if (b==e) hmatel1 = hmatel1 - h1a_vv(c,d) ! (1)
                        if (c==e) hmatel1 = hmatel1 + h1a_vv(b,d) ! (bc)
                        if (b==d) hmatel1 = hmatel1 + h1a_vv(c,e) ! (de)
                        if (c==d) hmatel1 = hmatel1 - h1a_vv(b,e) ! (bc)(de)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = r3a_excits(1,jdet); e = r3a_excits(2,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdeb >
                        hmatel = -h2a_vvvv(a,c,d,e)
                        ! compute < ijkabc | h1a(vv) | ijkdeb > = -A(ac)A(de) h1a_vv(a,d) * delta(c,e)
                        hmatel1 = 0.0d0
                        if (c==e) hmatel1 = hmatel1 - h1a_vv(a,d) ! (1)
                        if (a==e) hmatel1 = hmatel1 + h1a_vv(c,d) ! (ac)
                        if (c==d) hmatel1 = hmatel1 + h1a_vv(a,e) ! (de)
                        if (a==d) hmatel1 = hmatel1 - h1a_vv(c,e) ! (ac)(de)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * r3a_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                  
                  !!!! diagram 1b: -A(i/jk) x1a(mi) * t3a(abcmjk)
                  !!!! diagram 3b: 1/2 A(i/jk) x2a(mnij) * t3a(abcmnk)
                  ! NOTE: WITHIN THESE LOOPS, X1A(OO) TERMS ARE DOUBLE-COUNTED SO COMPENSATE BY FACTOR OF 1/2
                  ! allocate and initialize the copy of t3a
                  allocate(amps_buff(n3aaa_t))
                  allocate(excits_buff(6,n3aaa_t))
                  amps_buff(:) = t3a_amps(:)
                  excits_buff(:,:) = t3a_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = nua*(nua-1)*(nua-2)/6*noa
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(nua,nua,nua,noa))
                  !!! ABCK LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/3,noa/), nua, nua, nua, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,3,6/), nua, nua, nua, noa, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp excits_buff,r3a_excits,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X1A_oo,X2A_oooo,&
                  !$omp noa,nua,n3aaa_t),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,k)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        l = excits_buff(4,jdet); m = excits_buff(5,jdet);
                        ! compute < ijkabc | h2a(oooo) | lmkabc >
                        hmatel = x2a_oooo(l,m,i,j)
                        ! compute < ijkabc | h1a(oo) | lmkabc > = -A(ij)A(lm) x1a_oo(l,i) * delta(m,j)
                        hmatel1 = 0.0d0
                        if (m==j) hmatel1 = hmatel1 - x1a_oo(l,i) ! (1)      < ijkabc | h1a(oo) | ljkabc >
                        if (m==i) hmatel1 = hmatel1 + x1a_oo(l,j) ! (ij)     < ijkabc | h1a(oo) | likabc >
                        if (l==j) hmatel1 = hmatel1 + x1a_oo(m,i) ! (lm)     < ijkabc | h1a(oo) | jmkabc >
                        if (l==i) hmatel1 = hmatel1 - x1a_oo(m,j) ! (ij)(lm) < ijkabc | h1a(oo) | imkabc >
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ik)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l = excits_buff(4,jdet); m = excits_buff(5,jdet);
                           ! compute < ijkabc | h2a(oooo) | lmiabc >
                           hmatel = -x2a_oooo(l,m,k,j)
                           ! compute < ijkabc | h1a(oo) | lmiabc > = A(jk)A(lm) x1a_oo(l,k) * delta(m,j)
                           hmatel1 = 0.0d0
                           if (m==j) hmatel1 = hmatel1 + x1a_oo(l,k) ! (1)      < ijkabc | h1a(oo) | ljiabc >
                           if (m==k) hmatel1 = hmatel1 - x1a_oo(l,j) ! (jk)     < ijkabc | h1a(oo) | lkiabc >
                           if (l==j) hmatel1 = hmatel1 - x1a_oo(m,k) ! (lm)
                           if (l==k) hmatel1 = hmatel1 + x1a_oo(m,j) ! (jk)(lm)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l = excits_buff(4,jdet); m = excits_buff(5,jdet);
                           ! compute < ijkabc | h2a(oooo) | lmjabc >
                           hmatel = -x2a_oooo(l,m,i,k)
                           ! compute < ijkabc | h1a(oo) | lmjabc > = A(ik)A(lm) x1a_oo(l,i) * delta(m,k)
                           hmatel1 = 0.0d0
                           if (m==k) hmatel1 = hmatel1 + x1a_oo(l,i) ! (1)      < ijkabc | h1a(oo) | lkjabc >
                           if (m==i) hmatel1 = hmatel1 - x1a_oo(l,k) ! (ik)
                           if (l==k) hmatel1 = hmatel1 - x1a_oo(m,i) ! (lm)
                           if (l==i) hmatel1 = hmatel1 + x1a_oo(m,k) ! (ik)(lm)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABCI LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/1,noa-2/), nua, nua, nua, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,3,4/), nua, nua, nua, noa, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp excits_buff,r3a_excits,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X1A_oo,X2A_oooo,&
                  !$omp noa,nua,n3aaa_t),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,i)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        m = excits_buff(5,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(oooo) | imnabc >
                        hmatel = x2a_oooo(m,n,j,k)
                        ! compute < ijkabc | h1a(oo) | imnabc > = -A(jk)A(mn) x1a_oo(m,j) * delta(n,k)
                        hmatel1 = 0.0d0
                        if (n==k) hmatel1 = hmatel1 - x1a_oo(m,j)  ! < ijkabc | h1a(oo) | imkabc >
                        if (n==j) hmatel1 = hmatel1 + x1a_oo(m,k)
                        if (m==k) hmatel1 = hmatel1 + x1a_oo(n,j)
                        if (m==j) hmatel1 = hmatel1 - x1a_oo(n,k)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           m = excits_buff(5,jdet); n = excits_buff(6,jdet);
                           ! compute < ijkabc | h2a(oooo) | jmnabc >
                           hmatel = -x2a_oooo(m,n,i,k)
                           ! compute < ijkabc | h1a(oo) | jmnabc > = A(ik)A(mn) x1a_oo(m,i) * delta(n,k)
                           hmatel1 = 0.0d0
                           if (n==k) hmatel1 = hmatel1 + x1a_oo(m,i)
                           if (n==i) hmatel1 = hmatel1 - x1a_oo(m,k)
                           if (m==k) hmatel1 = hmatel1 - x1a_oo(n,i)
                           if (m==i) hmatel1 = hmatel1 + x1a_oo(n,k)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           m = excits_buff(5,jdet); n = excits_buff(6,jdet);
                           ! compute < ijkabc | h2a(oooo) | kmnabc >
                           hmatel = -x2a_oooo(m,n,j,i)
                           ! compute < ijkabc | h1a(oo) | kmnabc > = A(ij)A(mn) x1a_oo(m,j) * delta(n,i)
                           hmatel1 = 0.0d0
                           if (n==i) hmatel1 = hmatel1 - x1a_oo(m,j)
                           if (n==j) hmatel1 = hmatel1 + x1a_oo(m,i)
                           if (m==i) hmatel1 = hmatel1 + x1a_oo(n,j)
                           if (m==j) hmatel1 = hmatel1 - x1a_oo(n,i)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABCJ LOOP !!!
                  call get_index_table(idx_table, (/1,nua-2/), (/-1,nua-1/), (/-1,nua/), (/2,noa-1/), nua, nua, nua, noa)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/1,2,3,5/), nua, nua, nua, noa, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp excits_buff,r3a_excits,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X1A_oo,X2A_oooo,&
                  !$omp noa,nua,n3aaa_t),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(a,b,c,j)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        l = excits_buff(4,jdet); n = excits_buff(6,jdet);
                        ! compute < ijkabc | h2a(oooo) | ljnabc >
                        hmatel = x2a_oooo(l,n,i,k)
                        ! compute < ijkabc | h1a(oo) | ljnabc > = -A(ik)A(ln) x1a_oo(l,i) * delta(n,k)
                        hmatel1 = 0.0d0
                        if (n==k) hmatel1 = hmatel1 - x1a_oo(l,i)
                        if (n==i) hmatel1 = hmatel1 + x1a_oo(l,k)
                        if (l==k) hmatel1 = hmatel1 + x1a_oo(n,i)
                        if (l==i) hmatel1 = hmatel1 - x1a_oo(n,k)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l = excits_buff(4,jdet); n = excits_buff(6,jdet);
                           ! compute < ijkabc | h2a(oooo) | linabc >
                           hmatel = -x2a_oooo(l,n,j,k)
                           ! compute < ijkabc | h1a(oo) | linabc > = A(jk)A(ln) x1a_oo(l,j) * delta(n,k)
                           hmatel1 = 0.0d0
                           if (n==k) hmatel1 = hmatel1 + x1a_oo(l,j)
                           if (n==j) hmatel1 = hmatel1 - x1a_oo(l,k)
                           if (l==k) hmatel1 = hmatel1 - x1a_oo(n,j)
                           if (l==j) hmatel1 = hmatel1 + x1a_oo(n,k)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(idx,1), loc_arr(idx,2)
                           l = excits_buff(4,jdet); n = excits_buff(6,jdet);
                           ! compute < ijkabc | h2a(oooo) | lknabc >
                           hmatel = -x2a_oooo(l,n,i,j)
                           ! compute < ijkabc | h1a(oo) | lknabc > = A(ij)A(ln) x1a_oo(l,i) * delta(n,j)
                           hmatel1 = 0.0d0
                           if (n==j) hmatel1 = hmatel1 + x1a_oo(l,i)
                           if (n==i) hmatel1 = hmatel1 - x1a_oo(l,j)
                           if (l==j) hmatel1 = hmatel1 - x1a_oo(n,i)
                           if (l==i) hmatel1 = hmatel1 + x1a_oo(n,j)
                           hmatel = hmatel + 0.5d0 * hmatel1
                           resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate temporary amplitude arrays
                  deallocate(excits_buff,amps_buff)
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 2b: A(a/bc) x1a(ae) * t3a(ebcijk)
                  !!!! diagram 4b: 1/2 A(c/ab) x2a(abef) * t3a(ebcijk)
                  ! NOTE: WITHIN THESE LOOPS, X1A(VV) TERMS ARE DOUBLE-COUNTED SO COMPENSATE BY FACTOR OF 1/2
                  ! allocate and initialize the copy of t3a
                  allocate(amps_buff(n3aaa_t))
                  allocate(excits_buff(6,n3aaa_t))
                  amps_buff(:) = t3a_amps(:)
                  excits_buff(:,:) = t3a_excits(:,:)
                  ! allocate new sorting arrays
                  nloc = noa*(noa-1)*(noa-2)/6*nua
                  allocate(loc_arr(nloc,2))
                  allocate(idx_table(noa,noa,noa,nua))
                  !!! IJKA LOOP !!!
                  call get_index_table(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/1,nua-2/), noa, noa, noa, nua)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,1/), noa, noa, noa, nua, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X1A_vv,X2A_vvvv,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,a)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = excits_buff(2,jdet); f = excits_buff(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkaef >
                        hmatel = x2a_vvvv(b,c,e,f)
                        ! compute < ijkabc | h1a(vv) | ijkaef > = A(bc)A(ef) x1a_vv(b,e) * delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 + x1a_vv(b,e) ! (1)
                        if (b==f) hmatel1 = hmatel1 - x1a_vv(c,e) ! (bc)
                        if (c==e) hmatel1 = hmatel1 - x1a_vv(b,f) ! (ef)
                        if (b==e) hmatel1 = hmatel1 + x1a_vv(c,f) ! (bc)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = excits_buff(2,jdet); f = excits_buff(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkbef >
                        hmatel = -x2a_vvvv(a,c,e,f)
                        ! compute < ijkabc | h1a(vv) | ijkbef > = -A(ac)A(ef) x1a_vv(a,e) * delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 - x1a_vv(a,e) ! (1)
                        if (a==f) hmatel1 = hmatel1 + x1a_vv(c,e) ! (ac)
                        if (c==e) hmatel1 = hmatel1 + x1a_vv(a,f) ! (ef)
                        if (a==e) hmatel1 = hmatel1 - x1a_vv(c,f) ! (ac)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        e = excits_buff(2,jdet); f = excits_buff(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkcef >
                        hmatel = -x2a_vvvv(b,a,e,f)
                        ! compute < ijkabc | h1a(vv) | ijkcef > = -A(ab)A(ef) x1a_vv(b,e) * delta(a,f)
                        hmatel1 = 0.0d0
                        if (a==f) hmatel1 = hmatel1 - x1a_vv(b,e) ! (1)
                        if (b==f) hmatel1 = hmatel1 + x1a_vv(a,e) ! (ab)
                        if (a==e) hmatel1 = hmatel1 + x1a_vv(b,f) ! (ef)
                        if (b==e) hmatel1 = hmatel1 - x1a_vv(a,f) ! (ab)(ef)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! IJKB LOOP !!!
                  call get_index_table(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/2,nua-1/), noa, noa, noa, nua)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,2/), noa, noa, noa, nua, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X1A_vv,X2A_vvvv,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,b)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = excits_buff(1,jdet); f = excits_buff(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdbf >
                        hmatel = x2a_vvvv(a,c,d,f)
                        ! compute < ijkabc | h1a(vv) | ijkdbf > = A(ac)A(df) x1a_vv(a,d) * delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 + x1a_vv(a,d) ! (1)
                        if (a==f) hmatel1 = hmatel1 - x1a_vv(c,d) ! (ac)
                        if (c==d) hmatel1 = hmatel1 - x1a_vv(a,f) ! (df)
                        if (a==d) hmatel1 = hmatel1 + x1a_vv(c,f) ! (ac)(df)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = excits_buff(1,jdet); f = excits_buff(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdaf >
                        hmatel = -x2a_vvvv(b,c,d,f)
                        ! compute < ijkabc | h1a(vv) | ijkdaf > = -A(bc)A(df) x1a_vv(b,d) * delta(c,f)
                        hmatel1 = 0.0d0
                        if (c==f) hmatel1 = hmatel1 - x1a_vv(b,d) ! (1)
                        if (b==f) hmatel1 = hmatel1 + x1a_vv(c,d) ! (bc)
                        if (c==d) hmatel1 = hmatel1 + x1a_vv(b,f) ! (df)
                        if (b==d) hmatel1 = hmatel1 - x1a_vv(c,f) ! (bc)(df)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = excits_buff(1,jdet); f = excits_buff(3,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdcf >
                        hmatel = -x2a_vvvv(a,b,d,f)
                        ! compute < ijkabc | h1a(vv) | ijkdcf > = -A(ab)A(df) x1a_vv(a,d) * delta(b,f)
                        hmatel1 = 0.0d0
                        if (b==f) hmatel1 = hmatel1 - x1a_vv(a,d) ! (1)
                        if (a==f) hmatel1 = hmatel1 + x1a_vv(b,d) ! (ab)
                        if (b==d) hmatel1 = hmatel1 + x1a_vv(a,f) ! (df)
                        if (a==d) hmatel1 = hmatel1 - x1a_vv(b,f) ! (ab)(df)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! IJKC LOOP !!!
                  call get_index_table(idx_table, (/1,noa-2/), (/-1,noa-1/), (/-1,noa/), (/3,nua/), noa, noa, noa, nua)
                  call sort4(excits_buff, amps_buff, loc_arr, idx_table, (/4,5,6,3/), noa, noa, noa, nua, nloc, n3aaa_t)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp excits_buff,&
                  !$omp amps_buff,&
                  !$omp loc_arr,idx_table,&
                  !$omp X1A_vv,X2A_vvvv,&
                  !$omp noa,nua,n3aaa_r),&
                  !$omp private(hmatel,hmatel1,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3aaa_r
                     a = r3a_excits(1,idet); b = r3a_excits(2,idet); c = r3a_excits(3,idet);
                     i = r3a_excits(4,idet); j = r3a_excits(5,idet); k = r3a_excits(6,idet);
                     ! (1)
                     idx = idx_table(i,j,k,c)
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = excits_buff(1,jdet); e = excits_buff(2,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdec >
                        hmatel = x2a_vvvv(a,b,d,e)
                        ! compute < ijkabc | h1a(vv) | ijkdec > = A(ab)A(de) x1a_vv(a,d) * delta(b,e)
                        hmatel1 = 0.0d0
                        if (b==e) hmatel1 = hmatel1 + x1a_vv(a,d) ! (1)
                        if (a==e) hmatel1 = hmatel1 - x1a_vv(b,d) ! (ab)
                        if (b==d) hmatel1 = hmatel1 - x1a_vv(a,e) ! (de)
                        if (a==d) hmatel1 = hmatel1 + x1a_vv(b,e) ! (ab)(de)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = excits_buff(1,jdet); e = excits_buff(2,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdea >
                        hmatel = -x2a_vvvv(c,b,d,e)
                        ! compute < ijkabc | h1a(vv) | ijkdea > = -A(bc)A(de) x1a_vv(c,d) * delta(b,e)
                        hmatel1 = 0.0d0
                        if (b==e) hmatel1 = hmatel1 - x1a_vv(c,d) ! (1)
                        if (c==e) hmatel1 = hmatel1 + x1a_vv(b,d) ! (bc)
                        if (b==d) hmatel1 = hmatel1 + x1a_vv(c,e) ! (de)
                        if (c==d) hmatel1 = hmatel1 - x1a_vv(b,e) ! (bc)(de)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(idx,1), loc_arr(idx,2)
                        d = excits_buff(1,jdet); e = excits_buff(2,jdet);
                        ! compute < ijkabc | h2a(vvvv) | ijkdeb >
                        hmatel = -x2a_vvvv(a,c,d,e)
                        ! compute < ijkabc | h1a(vv) | ijkdeb > = -A(ac)A(de) x1a_vv(a,d) * delta(c,e)
                        hmatel1 = 0.0d0
                        if (c==e) hmatel1 = hmatel1 - x1a_vv(a,d) ! (1)
                        if (a==e) hmatel1 = hmatel1 + x1a_vv(c,d) ! (ac)
                        if (c==d) hmatel1 = hmatel1 + x1a_vv(a,e) ! (de)
                        if (a==d) hmatel1 = hmatel1 - x1a_vv(c,e) ! (ac)(de)
                        hmatel = hmatel + 0.5d0 * hmatel1
                        resid(idet) = resid(idet) + hmatel * amps_buff(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate temporary amplitude arrays
                  deallocate(excits_buff,amps_buff)
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)
                 
                 
                 
              end subroutine build_hr_3a
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!! SORTING FUNCTIONS !!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

              subroutine get_index_table(idx_table, rng1, rng2, rng3, rng4, n1, n2, n3, n4)

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

              end subroutine get_index_table

              subroutine sort4(excits, amps, loc_arr, idx_table, idims, n1, n2, n3, n4, nloc, n3p, x1a)

                    integer, intent(in) :: n1, n2, n3, n4, nloc, n3p
                    integer, intent(in) :: idims(4)
                    integer, intent(in) :: idx_table(n1,n2,n3,n4)
      
                    integer, intent(inout) :: loc_arr(nloc,2)
                    integer, intent(inout) :: excits(6,n3p)
                    real(kind=8), intent(inout) :: amps(n3p)
                    real(kind=8), intent(inout), optional :: x1a(n3p)
      
                    integer :: idet
                    integer :: p, q, r, s
                    integer :: p1, q1, r1, s1, p2, q2, r2, s2
                    integer :: pqrs1, pqrs2
                    integer, allocatable :: temp(:), idx(:)
      
                    allocate(temp(n3p),idx(n3p))
                    do idet = 1, n3p
                       p = excits(idims(1),idet); q = excits(idims(2),idet); r = excits(idims(3),idet); s = excits(idims(4),idet)
                       temp(idet) = idx_table(p,q,r,s)
                    end do
                    call argsort(temp, idx)
                    excits = excits(:,idx)
                    amps = amps(idx)
                    if (present(x1a)) x1a = x1a(idx)
                    deallocate(temp,idx)
      
                    loc_arr(:,1) = 1; loc_arr(:,2) = 0;
                    do idet = 1, n3p-1
                       p1 = excits(idims(1),idet);   q1 = excits(idims(2),idet);   r1 = excits(idims(3),idet);   s1 = excits(idims(4),idet)
                       p2 = excits(idims(1),idet+1); q2 = excits(idims(2),idet+1); r2 = excits(idims(3),idet+1); s2 = excits(idims(4),idet+1)
                       pqrs1 = idx_table(p1,q1,r1,s1)
                       pqrs2 = idx_table(p2,q2,r2,s2)
                       if (pqrs1 /= pqrs2) then
                          loc_arr(pqrs1,2) = idet
                          loc_arr(pqrs2,1) = idet+1
                       end if
                    end do
                    loc_arr(pqrs2,2) = n3p

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

end module eomccsdt_p_loops
 
