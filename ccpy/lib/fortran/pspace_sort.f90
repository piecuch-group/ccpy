module pspace_sort
! Sorting of P-space excitation lists, shared by all P-space modules.
!
! The P-space loops visit, for a given excitation, all excitations that share a subset of
! its orbital indices. The excitation list (and its amplitude and residual vectors) is
! sorted so that excitations with the same indices in the chosen columns are contiguous,
! and the first/last position of each such block is recorded:
!
!   * sort2/sort3/sort4 (and variants) sort by the lexical index of 2, 3, or 4 columns
!     given by an index table (get_index_table2/3/4); loc_arr(:,key) = block bounds.
!   * sort_pspace sorts by any number of columns (1 to 6) with keys computed directly from
!     the indices (no index table), returning only the blocks that occur.
!   * bucket_pspace groups excitations by columns without moving any data.
!   * keep_largest moves the k entries of largest magnitude of a list of moments (and their
!     excitations) to its front, for the selection buffers of adaptive CC(P;Q).
!
! All sorts are stable counting sorts (or LSD radix sorts for very large key ranges) that
! use one module workspace, grown to the largest list sorted and kept until
! release_sort_workspace() is called at the end of an update. The workspace is shared, so
! these routines must not be called from inside OpenMP parallel regions.

      implicit none
      private

      public :: get_index_table, get_index_table2, get_index_table3, get_index_table4
      public :: sort2, sort3, sort3_excits, sort3_excits_cm, sort4, sort4t, sort4_lt
      public :: argsort, argsort_r8_descend
      public :: pspace_nkeys, pspace_key, sort_pspace, pspace_find, bucket_pspace
      public :: release_sort_workspace, keep_largest

      ! largest key range for which dense (counting-sort) tables are used
      integer(kind=8), parameter :: max_dense_keys = 2_8**26

      integer, allocatable :: ws_key(:), ws_perm(:), ws_int(:)
      real(kind=8), allocatable :: ws_real(:)

      contains

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! Workspace
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine ensure_workspace(n)
              integer, intent(in) :: n
              if (allocated(ws_key)) then
                 if (size(ws_key) >= n) return
                 deallocate(ws_key, ws_perm, ws_int, ws_real)
              end if
              allocate(ws_key(max(n,1)), ws_perm(max(n,1)), ws_int(max(n,1)), ws_real(max(n,1)))
      end subroutine ensure_workspace

      subroutine release_sort_workspace()
      ! Free the sorting workspace (call at the end of each update routine).
              if (allocated(ws_key)) deallocate(ws_key, ws_perm, ws_int, ws_real)
      end subroutine release_sort_workspace

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! Core: counting sort of ws_key(1:n) into the permutation ws_perm(1:n)
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine counting_sort(n, nkey, first, last)
      ! Stable counting sort by ws_key(1:n) in 1..nkey. On exit, ws_perm(first(k):last(k))
      ! holds the positions with key k (last(k) < first(k) if there are none). Keys outside
      ! 1..nkey (which should not occur) are placed after all blocks.
              integer, intent(in) :: n, nkey
              integer, intent(inout) :: first(:), last(:)
              integer :: i, k, pos, ninvalid, pos_invalid

              last(1:nkey) = 0
              ninvalid = 0
              do i = 1, n
                 k = ws_key(i)
                 if (k >= 1 .and. k <= nkey) then
                    last(k) = last(k) + 1
                 else
                    ninvalid = ninvalid + 1
                 end if
              end do
              pos = 1
              do k = 1, nkey
                 first(k) = pos
                 pos = pos + last(k)
                 last(k) = first(k) - 1
              end do
              pos_invalid = pos
              do i = 1, n
                 k = ws_key(i)
                 if (k >= 1 .and. k <= nkey) then
                    last(k) = last(k) + 1
                    ws_perm(last(k)) = i
                 else
                    ws_perm(pos_invalid) = i
                    pos_invalid = pos_invalid + 1
                 end if
              end do
      end subroutine counting_sort

      subroutine permute_rows(excits, n)
      ! excits(1:n,:) <- excits(ws_perm(1:n),:)
              integer, intent(inout) :: excits(:,:)
              integer, intent(in) :: n
              integer :: i, col
              do col = 1, size(excits,2)
                 do i = 1, n
                    ws_int(i) = excits(ws_perm(i),col)
                 end do
                 excits(1:n,col) = ws_int(1:n)
              end do
      end subroutine permute_rows

      subroutine permute_columns(excits, n)
      ! excits(:,1:n) <- excits(:,ws_perm(1:n)) (excitations stored by column)
              integer, intent(inout) :: excits(:,:)
              integer, intent(in) :: n
              integer :: i, row
              do row = 1, size(excits,1)
                 do i = 1, n
                    ws_int(i) = excits(row,ws_perm(i))
                 end do
                 excits(row,1:n) = ws_int(1:n)
              end do
      end subroutine permute_columns

      subroutine permute_real(x, n)
              real(kind=8), intent(inout) :: x(:)
              integer, intent(in) :: n
              integer :: i
              do i = 1, n
                 ws_real(i) = x(ws_perm(i))
              end do
              x(1:n) = ws_real(1:n)
      end subroutine permute_real

      logical function empty_list(excits, n3p, by_column)
      ! The P space is empty: a single excitation with all indices equal to 1.
              integer, intent(in) :: excits(:,:)
              integer, intent(in) :: n3p
              logical, intent(in) :: by_column
              empty_list = .false.
              if (n3p <= 0) then
                 empty_list = .true.
              elseif (n3p == 1) then
                 if (by_column) then
                    empty_list = all(excits(:,1) == 1)
                 else
                    empty_list = all(excits(1,:) == 1)
                 end if
              end if
      end function empty_list

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! Index tables: idx_table(p,q,...) = lexical index of the tuple, 0 outside the ranges.
      ! A range rng = (lo, hi) with lo < 0 means "from (previous index) - lo to hi", i.e.,
      ! the index is ordered with respect to the previous one.
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      integer function lower_bound(rng, prev)
              integer, intent(in) :: rng(2), prev
              if (rng(1) < 0) then
                 lower_bound = prev - rng(1)
              else
                 lower_bound = rng(1)
              end if
      end function lower_bound

      subroutine get_index_table(idx_table, rng1, rng2, rng3, rng4, n1, n2, n3, n4)
              integer, intent(in) :: n1, n2, n3, n4
              integer, intent(in) :: rng1(2), rng2(2), rng3(2), rng4(2)
              integer, intent(inout) :: idx_table(n1,n2,n3,n4)
              integer :: kout, p, q, r, s
              idx_table = 0
              kout = 1
              do p = rng1(1), rng1(2)
                 do q = lower_bound(rng2, p), rng2(2)
                    do r = lower_bound(rng3, q), rng3(2)
                       do s = lower_bound(rng4, r), rng4(2)
                          idx_table(p,q,r,s) = kout
                          kout = kout + 1
                       end do
                    end do
                 end do
              end do
      end subroutine get_index_table

      subroutine get_index_table4(idx_table, rng1, rng2, rng3, rng4, n1, n2, n3, n4)
              integer, intent(in) :: n1, n2, n3, n4
              integer, intent(in) :: rng1(2), rng2(2), rng3(2), rng4(2)
              integer, intent(inout) :: idx_table(n1,n2,n3,n4)
              call get_index_table(idx_table, rng1, rng2, rng3, rng4, n1, n2, n3, n4)
      end subroutine get_index_table4

      subroutine get_index_table3(idx_table, rng1, rng2, rng3, n1, n2, n3)
              integer, intent(in) :: n1, n2, n3
              integer, intent(in) :: rng1(2), rng2(2), rng3(2)
              integer, intent(inout) :: idx_table(n1,n2,n3)
              integer :: kout, p, q, r
              idx_table = 0
              kout = 1
              do p = rng1(1), rng1(2)
                 do q = lower_bound(rng2, p), rng2(2)
                    do r = lower_bound(rng3, q), rng3(2)
                       idx_table(p,q,r) = kout
                       kout = kout + 1
                    end do
                 end do
              end do
      end subroutine get_index_table3

      subroutine get_index_table2(idx_table, rng1, rng2, n1, n2)
              integer, intent(in) :: n1, n2
              integer, intent(in) :: rng1(2), rng2(2)
              integer, intent(inout) :: idx_table(n1,n2)
              integer :: kout, p, q
              idx_table = 0
              kout = 1
              do p = rng1(1), rng1(2)
                 do q = lower_bound(rng2, p), rng2(2)
                    idx_table(p,q) = kout
                    kout = kout + 1
                 end do
              end do
      end subroutine get_index_table2

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! Sorts by index tables. In all of them:
      !   excits(n3p,ncol) (or excits(ncol,n3p) for the *_cm variant): excitation list
      !   amps(n3p), resid(n3p) (optional): vectors permuted along with excits
      !   idims: columns of excits that define the key; idx_table: key of those indices
      !   loc_arr(2,nloc) (loc_arr(nloc,2) for sort4_lt): first/last position of each key
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine sort4(excits, amps, loc_arr, idx_table, idims, n1, n2, n3, n4, nloc, n3p, resid)
              integer, intent(in) :: n1, n2, n3, n4, nloc, n3p
              integer, intent(in) :: idims(4)
              integer, intent(in) :: idx_table(n1,n2,n3,n4)
              integer, intent(inout) :: loc_arr(2,nloc)
              integer, intent(inout) :: excits(:,:)
              real(kind=8), intent(inout) :: amps(:)
              real(kind=8), intent(inout), optional :: resid(:)
              integer :: idet
              if (empty_list(excits, n3p, .false.)) then
                 loc_arr(1,:) = 1; loc_arr(2,:) = 0
                 return
              end if
              call ensure_workspace(n3p)
              do idet = 1, n3p
                 ws_key(idet) = idx_table(excits(idet,idims(1)), excits(idet,idims(2)),&
                                          excits(idet,idims(3)), excits(idet,idims(4)))
              end do
              call counting_sort(n3p, nloc, loc_arr(1,:), loc_arr(2,:))
              call permute_rows(excits, n3p)
              call permute_real(amps, n3p)
              if (present(resid)) call permute_real(resid, n3p)
      end subroutine sort4

      subroutine sort4t(excits, amps, loc_arr, idx_table, idims, n1, n2, n3, n4, nloc, n3p, resid)
              integer, intent(in) :: n1, n2, n3, n4, nloc, n3p
              integer, intent(in) :: idims(4)
              integer, intent(in) :: idx_table(n1,n2,n3,n4)
              integer, intent(inout) :: loc_arr(2,nloc)
              integer, intent(inout) :: excits(:,:)
              real(kind=8), intent(inout) :: amps(:)
              real(kind=8), intent(inout), optional :: resid(:)
              call sort4(excits, amps, loc_arr, idx_table, idims, n1, n2, n3, n4, nloc, n3p, resid)
      end subroutine sort4t

      subroutine sort4_lt(excits, amps, loc_arr, idx_table, idims, n1, n2, n3, n4, nloc, n3p, resid)
      ! sort4 with block bounds stored as loc_arr(nloc,2)
              integer, intent(in) :: n1, n2, n3, n4, nloc, n3p
              integer, intent(in) :: idims(4)
              integer, intent(in) :: idx_table(n1,n2,n3,n4)
              integer, intent(inout) :: loc_arr(nloc,2)
              integer, intent(inout) :: excits(:,:)
              real(kind=8), intent(inout) :: amps(:)
              real(kind=8), intent(inout), optional :: resid(:)
              integer :: idet
              if (empty_list(excits, n3p, .false.)) then
                 loc_arr(:,1) = 1; loc_arr(:,2) = 0
                 return
              end if
              call ensure_workspace(n3p)
              do idet = 1, n3p
                 ws_key(idet) = idx_table(excits(idet,idims(1)), excits(idet,idims(2)),&
                                          excits(idet,idims(3)), excits(idet,idims(4)))
              end do
              call counting_sort(n3p, nloc, loc_arr(:,1), loc_arr(:,2))
              call permute_rows(excits, n3p)
              call permute_real(amps, n3p)
              if (present(resid)) call permute_real(resid, n3p)
      end subroutine sort4_lt

      subroutine sort3(excits, amps, loc_arr, idx_table, idims, n1, n2, n3, nloc, n3p, resid)
              integer, intent(in) :: n1, n2, n3, nloc, n3p
              integer, intent(in) :: idims(3)
              integer, intent(in) :: idx_table(n1,n2,n3)
              integer, intent(inout) :: loc_arr(2,nloc)
              integer, intent(inout) :: excits(:,:)
              real(kind=8), intent(inout) :: amps(:)
              real(kind=8), intent(inout), optional :: resid(:)
              integer :: idet
              if (empty_list(excits, n3p, .false.)) then
                 loc_arr(1,:) = 1; loc_arr(2,:) = 0
                 return
              end if
              call ensure_workspace(n3p)
              do idet = 1, n3p
                 ws_key(idet) = idx_table(excits(idet,idims(1)), excits(idet,idims(2)), excits(idet,idims(3)))
              end do
              call counting_sort(n3p, nloc, loc_arr(1,:), loc_arr(2,:))
              call permute_rows(excits, n3p)
              call permute_real(amps, n3p)
              if (present(resid)) call permute_real(resid, n3p)
      end subroutine sort3

      subroutine sort3_excits(excits, loc_arr, idx_table, idims, n1, n2, n3, nloc, n3p)
      ! sort3 of an excitation list without amplitudes
              integer, intent(in) :: n1, n2, n3, nloc, n3p
              integer, intent(in) :: idims(3)
              integer, intent(in) :: idx_table(n1,n2,n3)
              integer, intent(inout) :: loc_arr(2,nloc)
              integer, intent(inout) :: excits(:,:)
              integer :: idet
              if (empty_list(excits, n3p, .false.)) then
                 loc_arr(1,:) = 1; loc_arr(2,:) = 0
                 return
              end if
              call ensure_workspace(n3p)
              do idet = 1, n3p
                 ws_key(idet) = idx_table(excits(idet,idims(1)), excits(idet,idims(2)), excits(idet,idims(3)))
              end do
              call counting_sort(n3p, nloc, loc_arr(1,:), loc_arr(2,:))
              call permute_rows(excits, n3p)
      end subroutine sort3_excits

      subroutine sort3_excits_cm(excits, loc_arr, idx_table, idims, n1, n2, n3, nloc, n3p)
      ! sort3_excits for excitations stored by column, excits(ncol,n3p)
              integer, intent(in) :: n1, n2, n3, nloc, n3p
              integer, intent(in) :: idims(3)
              integer, intent(in) :: idx_table(n1,n2,n3)
              integer, intent(inout) :: loc_arr(2,nloc)
              integer, intent(inout) :: excits(:,:)
              integer :: idet
              if (empty_list(excits, n3p, .true.)) then
                 loc_arr(1,:) = 1; loc_arr(2,:) = 0
                 return
              end if
              call ensure_workspace(n3p)
              do idet = 1, n3p
                 ws_key(idet) = idx_table(excits(idims(1),idet), excits(idims(2),idet), excits(idims(3),idet))
              end do
              call counting_sort(n3p, nloc, loc_arr(1,:), loc_arr(2,:))
              call permute_columns(excits, n3p)
      end subroutine sort3_excits_cm

      subroutine sort2(excits, amps, loc_arr, idx_table, idims, n1, n2, nloc, n3p, resid)
              integer, intent(in) :: n1, n2, nloc, n3p
              integer, intent(in) :: idims(2)
              integer, intent(in) :: idx_table(n1,n2)
              integer, intent(inout) :: loc_arr(2,nloc)
              integer, intent(inout) :: excits(:,:)
              real(kind=8), intent(inout) :: amps(:)
              real(kind=8), intent(inout), optional :: resid(:)
              integer :: idet
              if (empty_list(excits, n3p, .false.)) then
                 loc_arr(1,:) = 1; loc_arr(2,:) = 0
                 return
              end if
              call ensure_workspace(n3p)
              do idet = 1, n3p
                 ws_key(idet) = idx_table(excits(idet,idims(1)), excits(idet,idims(2)))
              end do
              call counting_sort(n3p, nloc, loc_arr(1,:), loc_arr(2,:))
              call permute_rows(excits, n3p)
              call permute_real(amps, n3p)
              if (present(resid)) call permute_real(resid, n3p)
      end subroutine sort2

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! General keys of any number of columns, computed from the indices.
      !
      !   cols(ncol): columns of the excitation list that define the key
      !   dims(ncol): range 1..dims(k) of each of those indices
      !   chain(ncol): chain(k) = .true. if index k is strictly greater than index k-1
      !                (same range); chain(1) is ignored
      !
      ! A run of chained indices p1 < p2 < ... < pg in 1..n is numbered by its combinatorial
      ! (colex) rank sum_t C(p_t - 1, t); runs are combined in mixed radix (first run most
      ! significant). Keys are 1..pspace_nkeys(...), and 0 for invalid tuples.
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      pure integer(kind=8) function binomial(n, k)
              integer, intent(in) :: n, k
              integer :: t
              if (k < 0 .or. k > n) then
                 binomial = 0
                 return
              end if
              binomial = 1
              do t = 1, k
                 binomial = binomial * (n - k + t) / t
              end do
      end function binomial

      pure integer(kind=8) function pspace_nkeys(ncol, dims, chain)
      ! Number of possible keys.
              integer, intent(in) :: ncol, dims(ncol)
              logical, intent(in) :: chain(ncol)
              integer :: k, g
              pspace_nkeys = 1
              k = 1
              do while (k <= ncol)
                 g = 1
                 do while (k + g <= ncol)
                    if (.not. chain(k+g)) exit
                    g = g + 1
                 end do
                 pspace_nkeys = pspace_nkeys * binomial(dims(k), g)
                 k = k + g
              end do
      end function pspace_nkeys

      pure integer(kind=8) function pspace_key(vals, ncol, dims, chain)
      ! Key of the indices vals(1:ncol) (0 if out of range or not ordered as required).
              integer, intent(in) :: ncol, vals(ncol), dims(ncol)
              logical, intent(in) :: chain(ncol)
              integer :: k, g, t
              integer(kind=8) :: rank
              pspace_key = 0
              k = 1
              do while (k <= ncol)
                 g = 1
                 do while (k + g <= ncol)
                    if (.not. chain(k+g)) exit
                    g = g + 1
                 end do
                 rank = 0
                 do t = 1, g
                    if (vals(k+t-1) < 1 .or. vals(k+t-1) > dims(k)) then
                       pspace_key = 0
                       return
                    end if
                    if (t > 1) then
                       if (vals(k+t-1) <= vals(k+t-2)) then
                          pspace_key = 0
                          return
                       end if
                    end if
                    rank = rank + binomial(vals(k+t-1) - 1, t)
                 end do
                 pspace_key = pspace_key * binomial(dims(k), g) + rank
                 k = k + g
              end do
              pspace_key = pspace_key + 1
      end function pspace_key

      subroutine sort_pspace(excits, amps, n3p, ncol, cols, dims, chain, nblock, block_key, block_off, resid)
      ! Sort excits(n3p,:), amps, and (optionally) resid by the key of the columns cols. On
      ! exit, the excitations with key block_key(b) occupy positions
      ! block_off(b):block_off(b+1)-1, for the nblock keys that occur (block_key ascending;
      ! use pspace_find to locate a key).
              integer, intent(in) :: n3p, ncol, cols(ncol), dims(ncol)
              logical, intent(in) :: chain(ncol)
              integer, intent(inout) :: excits(:,:)
              real(kind=8), intent(inout) :: amps(:)
              real(kind=8), intent(inout), optional :: resid(:)
              integer, intent(out) :: nblock
              integer(kind=8), allocatable, intent(out) :: block_key(:)
              integer, allocatable, intent(out) :: block_off(:)
              integer(kind=8), allocatable :: keys(:)
              integer(kind=8) :: nk
              integer :: idet, k, vals(6)
              integer, allocatable :: first(:), last(:)

              if (empty_list(excits, n3p, .false.)) then
                 nblock = 0
                 allocate(block_key(0), block_off(1))
                 block_off(1) = 1
                 return
              end if
              call ensure_workspace(n3p)
              nk = pspace_nkeys(ncol, dims, chain)
              allocate(keys(n3p))
              do idet = 1, n3p
                 do k = 1, ncol
                    vals(k) = excits(idet,cols(k))
                 end do
                 keys(idet) = pspace_key(vals, ncol, dims, chain)
              end do
              if (nk <= max(4_8*n3p, 2_8**20) .and. nk <= max_dense_keys) then
                 ws_key(1:n3p) = int(keys)
                 allocate(first(nk), last(nk))
                 call counting_sort(n3p, int(nk), first, last)
                 deallocate(first, last)
              else
                 call radix_sort(keys, n3p, nk)
              end if
              ! blocks of equal keys in the sorted order
              nblock = 0
              do idet = 1, n3p
                 if (idet == 1) then
                    nblock = 1
                 elseif (keys(ws_perm(idet)) /= keys(ws_perm(idet-1))) then
                    nblock = nblock + 1
                 end if
              end do
              allocate(block_key(nblock), block_off(nblock+1))
              nblock = 0
              do idet = 1, n3p
                 if (idet == 1) then
                    nblock = 1
                 elseif (keys(ws_perm(idet)) /= keys(ws_perm(idet-1))) then
                    nblock = nblock + 1
                 else
                    cycle
                 end if
                 block_key(nblock) = keys(ws_perm(idet))
                 block_off(nblock) = idet
              end do
              block_off(nblock+1) = n3p + 1
              deallocate(keys)
              call permute_rows(excits, n3p)
              call permute_real(amps, n3p)
              if (present(resid)) call permute_real(resid, n3p)
      end subroutine sort_pspace

      subroutine radix_sort(keys, n, nk)
      ! Stable LSD radix sort (16-bit digits) of keys(1:n) in 0..nk into ws_perm(1:n).
              integer, intent(in) :: n
              integer(kind=8), intent(in) :: keys(n), nk
              integer, allocatable :: perm2(:), counts(:)
              integer :: i, d, pos, shift, digit, total
              allocate(perm2(n), counts(0:65535))
              do i = 1, n
                 ws_perm(i) = i
              end do
              shift = 0
              do while (shift < 63 .and. ishft(nk, -shift) > 0)
                 counts = 0
                 do i = 1, n
                    digit = int(iand(ishft(keys(ws_perm(i)), -shift), 65535_8))
                    counts(digit) = counts(digit) + 1
                 end do
                 pos = 1
                 do d = 0, 65535
                    total = counts(d)
                    counts(d) = pos
                    pos = pos + total
                 end do
                 do i = 1, n
                    digit = int(iand(ishft(keys(ws_perm(i)), -shift), 65535_8))
                    perm2(counts(digit)) = ws_perm(i)
                    counts(digit) = counts(digit) + 1
                 end do
                 ws_perm(1:n) = perm2
                 shift = shift + 16
              end do
              deallocate(perm2, counts)
      end subroutine radix_sort

      pure integer function pspace_find(block_key, nblock, key)
      ! Index of key in block_key(1:nblock) (ascending), or 0 if absent.
              integer, intent(in) :: nblock
              integer(kind=8), intent(in) :: block_key(nblock), key
              integer :: lo, hi, mid
              pspace_find = 0
              lo = 1
              hi = nblock
              do while (lo <= hi)
                 mid = (lo + hi) / 2
                 if (block_key(mid) == key) then
                    pspace_find = mid
                    return
                 elseif (block_key(mid) < key) then
                    lo = mid + 1
                 else
                    hi = mid - 1
                 end if
              end do
      end function pspace_find

      subroutine bucket_pspace(excits, n3p, ncol, cols, dims, chain, nk, off, dets)
      ! Group the excitations by the key of the columns cols without moving them: the
      ! excitations with key k are dets(off(k):off(k+1)-1), k = 1..nk (nk = pspace_nkeys).
              integer, intent(in) :: n3p, ncol, cols(ncol), dims(ncol), nk
              logical, intent(in) :: chain(ncol)
              integer, intent(in) :: excits(:,:)
              integer, intent(out) :: off(nk+1), dets(n3p)
              integer :: idet, k, key, vals(6), pos, cnt

              off = 0
              if (empty_list(excits, n3p, .false.)) then
                 off = 1
                 return
              end if
              do idet = 1, n3p
                 do k = 1, ncol
                    vals(k) = excits(idet,cols(k))
                 end do
                 key = int(pspace_key(vals, ncol, dims, chain))
                 if (key >= 1) off(key) = off(key) + 1
              end do
              pos = 1
              do key = 1, nk
                 cnt = off(key)
                 off(key) = pos
                 pos = pos + cnt
              end do
              off(nk+1) = pos
              do idet = 1, n3p
                 do k = 1, ncol
                    vals(k) = excits(idet,cols(k))
                 end do
                 key = int(pspace_key(vals, ncol, dims, chain))
                 if (key >= 1) then
                    dets(off(key)) = idet
                    off(key) = off(key) + 1
                 end if
              end do
              ! off(key) now points one past the block of key: shift back to the starts
              do key = nk, 2, -1
                 off(key) = off(key-1)
              end do
              off(1) = 1
      end subroutine bucket_pspace

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! Comparison sorts (merge sorts, same tie order as the original per-module copies)
      ! returning the sorting permutation
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine argsort(r, d)
      ! d: permutation sorting the integers r in ascending order
              integer, intent(in), dimension(:) :: r
              integer, intent(out), dimension(size(r)) :: d
              integer, allocatable :: il(:)
              integer :: stepsize, i, j, n, left, k, ksize
              n = size(r)
              do i = 1, n
                 d(i) = i
              end do
              if (n <= 1) return
              allocate(il(n))
              stepsize = 1
              do while (stepsize < n)
                 do left = 1, n-stepsize, stepsize*2
                    i = left; j = left+stepsize
                    ksize = min(stepsize*2, n-left+1)
                    k = 1
                    do while (i < left+stepsize .and. j < left+ksize)
                       if (r(d(i)) < r(d(j))) then
                          il(k) = d(i); i = i+1
                       else
                          il(k) = d(j); j = j+1
                       end if
                       k = k+1
                    end do
                    if (i < left+stepsize) then
                       il(k:ksize) = d(i:left+stepsize-1)
                    else
                       il(k:ksize) = d(j:left+ksize-1)
                    end if
                    d(left:left+ksize-1) = il(1:ksize)
                 end do
                 stepsize = stepsize*2
              end do
              deallocate(il)
      end subroutine argsort

      subroutine argsort_r8_descend(r, d)
      ! d: permutation sorting the reals r in descending order
              real(kind=8), intent(in), dimension(:) :: r
              integer, intent(out), dimension(size(r)) :: d
              integer, allocatable :: il(:)
              integer :: stepsize, i, j, n, left, k, ksize
              n = size(r)
              do i = 1, n
                 d(i) = i
              end do
              if (n <= 1) return
              allocate(il(n))
              stepsize = 1
              do while (stepsize < n)
                 do left = 1, n-stepsize, stepsize*2
                    i = left; j = left+stepsize
                    ksize = min(stepsize*2, n-left+1)
                    k = 1
                    do while (i < left+stepsize .and. j < left+ksize)
                       if (r(d(i)) > r(d(j))) then
                          il(k) = d(i); i = i+1
                       else
                          il(k) = d(j); j = j+1
                       end if
                       k = k+1
                    end do
                    if (i < left+stepsize) then
                       il(k:ksize) = d(i:left+stepsize-1)
                    else
                       il(k:ksize) = d(j:left+ksize-1)
                    end if
                    d(left:left+ksize-1) = il(1:ksize)
                 end do
                 stepsize = stepsize*2
              end do
              deallocate(il)
      end subroutine argsort_r8_descend

      subroutine keep_largest(moments, excits, n, k, kth)
      ! Reorder moments(1:n) and the rows of excits(1:n,:) so that the k entries of largest
      ! |moment| come first (in no particular order); kth = smallest |moment| among them.
      ! Quickselect: O(n) on average, no workspace.
              integer, intent(in) :: n, k
              real(kind=8), intent(inout) :: moments(:)
              integer, intent(inout) :: excits(:,:)
              real(kind=8), intent(out) :: kth
              integer :: lo, hi, i, j, mid
              real(kind=8) :: pivot, a, b, c

              kth = 0.0d0
              if (k <= 0 .or. n <= 0) return
              if (k >= n) then
                 kth = minval(abs(moments(1:n)))
                 return
              end if
              lo = 1
              hi = n
              do while (lo < hi)
                 ! median-of-three pivot on |moment|
                 mid = lo + (hi - lo) / 2
                 a = abs(moments(lo)); b = abs(moments(mid)); c = abs(moments(hi))
                 pivot = max(min(a, b), min(max(a, b), c))
                 i = lo
                 j = hi
                 do while (i <= j)
                    do while (abs(moments(i)) > pivot)
                       i = i + 1
                    end do
                    do while (abs(moments(j)) < pivot)
                       j = j - 1
                    end do
                    if (i <= j) then
                       call swap(i, j)
                       i = i + 1
                       j = j - 1
                    end if
                 end do
                 ! now |m(lo:j)| >= pivot >= |m(i:hi)|, and entries j+1:i-1 equal the pivot
                 if (k <= j) then
                    hi = j
                 elseif (k >= i) then
                    lo = i
                 else
                    exit
                 end if
              end do
              kth = minval(abs(moments(1:k)))

              contains

              subroutine swap(p, q)
                      integer, intent(in) :: p, q
                      real(kind=8) :: tm
                      integer :: tr(size(excits,2))
                      tm = moments(p); moments(p) = moments(q); moments(q) = tm
                      tr = excits(p,:); excits(p,:) = excits(q,:); excits(q,:) = tr
              end subroutine swap
      end subroutine keep_largest

end module pspace_sort
