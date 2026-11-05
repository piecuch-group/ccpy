module sumrcc_loops
! Projections of the CC wave function of one reference onto the excitations of another,
!     P(q->p)_K = < Phi_K(p) | exp(T(q)) | Phi_q >,   |Phi_K(p)> = E_K(p) |Phi_p>,
! the single primitive of SU-MRCC (Heff_pq = <Phi_p|H|P(q->p)>, coupling terms from
! sum_q Heff_qp P(q->p)), and the MM-SUCC(2,3) triples corrections built on it.
!
! Algorithm. |D> = E_K(p)|Phi_p> is an excitation (H -> P) of Phi_q; its coefficient in
! exp(T(q))|Phi_q> is the sum over all partitions of (H, P) into cluster blocks of the product
! of their amplitudes, with the sign of the partition (cluster_coefficient; ccpy.mrcc.sumrcc.
! cluster_terms lists these terms explicitly). Blocks are t1, t2, and the internal amplitudes of
! rank >= 3 of Li and Paldus [JCP 119, 5320 (2003)], so any model space and any number of active
! electrons is covered; array sizes follow the longest excitation actually present.
!
! Bookkeeping per element is independent of the number of orbitals: for a pair (p, q) the
! orbitals q has and p lacks (rel_a) and the reverse (rel_b), and prefix counts of both
! determinants, give (H, P) and both signs from a few set operations (relative_excitation,
! string_sign). When reference q has no nonzero all-active amplitudes (always so in a complete
! model space) every block needs a core hole or a virtual particle, and branches that cannot
! be covered that way are cut (prune).
!
! Conventions
!   * Spin orbital s = g for alpha spatial orbital g = 0..norb-1 and s = norb + g for beta;
!     determinants are a+_{s1} a+_{s2} ... |vac> with s1 < s2 < ...
!   * occa_p(k) is the spatial orbital of the k-th alpha occupied orbital of reference p (its
!     local index k), unocca_p(k) that of the k-th alpha unoccupied orbital, etc.
!   * T(q) is stored in the local indices of q in the usual ccpy form: t1a(a,i) multiplies
!     a+_a a_i, t2a(a,b,i,j) multiplies a+_a a+_b a_j a_i, t2b(a,b,i,j) multiplies
!     a+_a(alpha) a+_b(beta) a_j(beta) a_i(alpha). Output arrays use the local indices of p
!     with the same meaning, and are fully antisymmetric (triples: unique entries only).
!   * Internal amplitude e of rank int_k(e) multiplies a+_{P1}..a+_{Pk} a_{Hk}..a_{H1} with
!     holes int_h(1:k,e) and particles int_p(1:k,e) (spin orbitals, ascending); int_k(e) = 0
!     marks an empty entry.
!   * inactive(s) = 1 for core spin orbitals (occupied in every reference) and virtual ones
!     (empty in every reference); prune = 1 enables the inactive-index pruning for T(q).

      implicit none

      ! excitations up to this rank use stack buffers; longer ones allocate (no upper limit)
      integer, parameter :: nbuf = 32
      ! the projection loops run in parallel only for arrays of at least 4096 entries (below
      ! that, starting the threads costs more than the work)

      contains

      ! ------------------------------------------------------------------ bookkeeping
      subroutine orbital_maps(lo, lu, det, occa, unocca, occb, unoccb, noa, nua, nob, nub, norb)
            integer, intent(in) :: noa, nua, nob, nub, norb
            integer, intent(in) :: occa(noa), unocca(nua), occb(nob), unoccb(nub)
            integer, intent(out) :: lo(0:2*norb-1), lu(0:2*norb-1)
            logical, intent(out) :: det(0:2*norb-1)
            integer :: k

            lo = 0; lu = 0; det = .false.
            do k = 1, noa
               det(occa(k)) = .true.
               lo(occa(k)) = k
            end do
            do k = 1, nob
               det(norb + occb(k)) = .true.
               lo(norb + occb(k)) = k
            end do
            do k = 1, nua
               lu(unocca(k)) = k
            end do
            do k = 1, nub
               lu(norb + unoccb(k)) = k
            end do
      end subroutine orbital_maps

      ! cnt(s) = number of occupied spin orbitals below s; rel_a = occupied in q, empty in p;
      ! rel_b = occupied in p, empty in q (both ascending)
      subroutine pair_context(cnt_p, cnt_q, rel_a, na, rel_b, nb, det_p, det_q, nso)
            integer, intent(in) :: nso
            logical, intent(in) :: det_p(0:nso-1), det_q(0:nso-1)
            integer, intent(out) :: cnt_p(0:nso-1), cnt_q(0:nso-1), rel_a(nso), rel_b(nso), na, nb
            integer :: s, cp, cq
            cp = 0; cq = 0; na = 0; nb = 0
            rel_a = 0; rel_b = 0
            do s = 0, nso - 1
               cnt_p(s) = cp; cnt_q(s) = cq
               if (det_p(s)) cp = cp + 1
               if (det_q(s)) cq = cq + 1
               if (det_q(s) .and. .not. det_p(s)) then
                  na = na + 1; rel_a(na) = s
               end if
               if (det_p(s) .and. .not. det_q(s)) then
                  nb = nb + 1; rel_b(nb) = s
               end if
            end do
      end subroutine pair_context

      ! sign of a+_{cre(1)}..a+_{cre(nc)} a_{ann(na)}..a_{ann(1)} on a determinant with prefix
      ! counts cnt (a_{ann(1)} acts first, a+_{cre(1)} last; the string is assumed valid)
      function string_sign(cnt, ann, na, cre, nc, nso) result(sgn)
            integer, intent(in) :: nso, na, nc
            integer, intent(in) :: cnt(0:nso-1), ann(na), cre(nc)
            real(kind=8) :: sgn
            integer :: k, par, s
            par = 0
            do k = 1, na
               s = ann(k)
               par = par + cnt(s) - count(ann(1:k-1) < s)
            end do
            do k = nc, 1, -1
               s = cre(k)
               par = par + cnt(s) - count(ann < s) + count(cre(k+1:nc) < s)
            end do
            sgn = 1.0d0
            if (mod(par, 2) == 1) sgn = -1.0d0
      end function string_sign

      ! holes h and particles pt (ascending) of D = E|Phi_p>, E = a+_{x1}..a+_{xm} a_{ym}..a_{y1},
      ! relative to Phi_q: h = (rel_a \ x) + (y in q), pt = (rel_b \ y) + (x not in q)
      subroutine relative_excitation(h, pt, n, x, y, m, det_q, rel_a, na, rel_b, nb, nso)
            integer, intent(in) :: m, na, nb, nso
            integer, intent(in) :: x(m), y(m), rel_a(na), rel_b(nb)
            logical, intent(in) :: det_q(0:nso-1)
            integer, intent(out) :: h(na+m), pt(nb+m), n
            integer :: k, nh, np_
            nh = 0; np_ = 0
            do k = 1, na
               if (all(x /= rel_a(k))) then
                  nh = nh + 1; h(nh) = rel_a(k)
               end if
            end do
            do k = 1, m
               if (det_q(y(k))) then
                  nh = nh + 1; h(nh) = y(k)
               end if
            end do
            do k = 1, nb
               if (all(y /= rel_b(k))) then
                  np_ = np_ + 1; pt(np_) = rel_b(k)
               end if
            end do
            do k = 1, m
               if (.not. det_q(x(k))) then
                  np_ = np_ + 1; pt(np_) = x(k)
               end if
            end do
            call sort_small(h, nh)
            call sort_small(pt, np_)
            n = nh
            if (np_ /= nh) n = -1
      end subroutine relative_excitation

      subroutine sort_small(v, n)
            integer, intent(in) :: n
            integer, intent(inout) :: v(*)
            integer :: k, l, key
            do k = 2, n
               key = v(k)
               l = k - 1
               do while (l >= 1)
                  if (v(l) <= key) exit
                  v(l+1) = v(l)
                  l = l - 1
               end do
               v(l+1) = key
            end do
      end subroutine sort_small

      ! <D|exp(T(q))|Phi_q> for |D> = E|Phi_p>, E = a+_{x1}..a+_{xm} a_{ym}..a_{y1}
      subroutine excitation_coefficient(c, x, y, m, det_q, cnt_p, cnt_q, rel_a, na, rel_b, nb,&
                                        t1a, t1b, t2a, t2b, t2c, int_h, int_p, int_k, int_amp, mr, nint,&
                                        lo, lu, inactive, prune, noa, nua, nob, nub, norb)
            integer, intent(in) :: m, na, nb, mr, nint, prune, noa, nua, nob, nub, norb
            integer, intent(in) :: x(m), y(m), rel_a(na), rel_b(nb)
            logical, intent(in) :: det_q(0:2*norb-1)
            integer, intent(in) :: cnt_p(0:2*norb-1), cnt_q(0:2*norb-1)
            integer, intent(in) :: int_h(mr,nint), int_p(mr,nint), int_k(nint), inactive(0:2*norb-1)
            real(kind=8), intent(in) :: int_amp(nint)
            integer, intent(in) :: lo(0:2*norb-1), lu(0:2*norb-1)
            real(kind=8), intent(in) :: t1a(nua,noa), t1b(nub,nob)
            real(kind=8), intent(in) :: t2a(nua,nua,noa,noa), t2b(nua,nub,noa,nob), t2c(nub,nub,nob,nob)
            real(kind=8), intent(out) :: c
            integer, target :: hbuf(nbuf), pbuf(nbuf)
            integer, allocatable, target :: halloc(:), palloc(:)
            integer, pointer, contiguous :: h(:), pt(:)
            integer :: n

            c = 0.0d0
            if (max(na, nb) + m <= nbuf) then
               h => hbuf; pt => pbuf
            else
               allocate(halloc(na+m), palloc(nb+m))
               h => halloc; pt => palloc
            end if
            call relative_excitation(h, pt, n, x, y, m, det_q, rel_a, na, rel_b, nb, 2*norb)
            if (n < 0) return
            call cluster_coefficient(c, h(1:n), pt(1:n), n, t1a, t1b, t2a, t2b, t2c, int_h, int_p, int_k, int_amp,&
                                     mr, nint, lo, lu, inactive, prune, noa, nua, nob, nub, norb)
            if (c /= 0.0d0) c = c * string_sign(cnt_p, y, m, x, m, 2*norb) * string_sign(cnt_q, h(1:n), n, pt(1:n), n, 2*norb)
      end subroutine excitation_coefficient

      ! ------------------------------------------------------------------ cluster expansion
      ! Coefficient of a+_{P1}..a+_{Pn} a_{Hn}..a_{H1}|Phi_q> (H, P ascending) in exp(T)|Phi_q>:
      ! sum over partitions of (H, P) into cluster blocks. The block holding H1 is chosen first;
      ! a block with holes at positions k_m and particles at positions j_m has the sign
      ! (-1)^sum_m (k_m - m + j_m - m).
      recursive subroutine cluster_coefficient(c, h, pt, n, t1a, t1b, t2a, t2b, t2c, int_h, int_p, int_k, int_amp,&
                                               mr, nint, lo, lu, inactive, prune, noa, nua, nob, nub, norb)
            integer, intent(in) :: n, mr, nint, prune, noa, nua, nob, nub, norb
            integer, intent(in) :: h(n), pt(n)
            integer, intent(in) :: int_h(mr,nint), int_p(mr,nint), int_k(nint), inactive(0:2*norb-1)
            real(kind=8), intent(in) :: int_amp(nint)
            integer, intent(in) :: lo(0:2*norb-1), lu(0:2*norb-1)
            real(kind=8), intent(in) :: t1a(nua,noa), t1b(nub,nob)
            real(kind=8), intent(in) :: t2a(nua,nua,noa,noa), t2b(nua,nub,noa,nob), t2c(nub,nub,nob,nob)
            real(kind=8), intent(out) :: c

            integer, target :: hb(nbuf), pb(nbuf), hpb(nbuf), ppb(nbuf)
            logical, target :: uhb(nbuf), upb(nbuf)
            integer, allocatable, target :: ha(:), pa(:), hpa(:), ppa(:)
            logical, allocatable, target :: uha(:), upa(:)
            integer, pointer, contiguous :: hr(:), pr(:), hpos(:), ppos(:)
            logical, pointer, contiguous :: used_h(:), used_p(:)
            integer :: j, k, l, m, e, kk, nr, par, sh1, shk, ninact
            logical :: found
            real(kind=8) :: amp, cs

            c = 0.0d0
            if (n == 0) then
               c = 1.0d0
               return
            end if
            if (prune /= 0) then
               ! every block needs a core hole or a virtual particle
               ninact = 0
               do m = 1, n
                  ninact = ninact + inactive(h(m)) + inactive(pt(m))
               end do
               if (2*ninact < n) return
            end if
            if (n <= nbuf) then
               hr => hb; pr => pb; hpos => hpb; ppos => ppb; used_h => uhb; used_p => upb
            else
               allocate(ha(n), pa(n), hpa(n), ppa(n), uha(n), upa(n))
               hr => ha; pr => pa; hpos => hpa; ppos => ppa; used_h => uha; used_p => upa
            end if

            sh1 = h(1) / norb
            ! T1 block (H1 -> Pj), sign (-1)^(j-1)
            do j = 1, n
               if (pt(j) / norb /= sh1) cycle
               if (sh1 == 0) then
                  amp = t1a(lu(pt(j)), lo(h(1)))
               else
                  amp = t1b(lu(pt(j)), lo(h(1)))
               end if
               if (amp == 0.0d0) cycle
               hr(1:n-1) = h(2:n)
               pr(1:j-1) = pt(1:j-1); pr(j:n-1) = pt(j+1:n)
               call cluster_coefficient(cs, hr(1:n-1), pr(1:n-1), n-1, t1a, t1b, t2a, t2b, t2c, int_h, int_p, int_k,&
                                        int_amp, mr, nint, lo, lu, inactive, prune, noa, nua, nob, nub, norb)
               c = c + (1 - 2*mod(j-1, 2)) * amp * cs
            end do
            ! T2 block (H1 Hk -> Pj Pl), sign (-1)^(j+l+k-1)
            do k = 2, n
               shk = h(k) / norb
               do j = 1, n - 1
                  do l = j + 1, n
                     if (sh1 == 0 .and. shk == 0) then
                        if (pt(j) / norb /= 0 .or. pt(l) / norb /= 0) cycle
                        amp = t2a(lu(pt(j)), lu(pt(l)), lo(h(1)), lo(h(k)))
                     elseif (sh1 == 1) then
                        if (pt(j) / norb /= 1) cycle
                        amp = t2c(lu(pt(j)), lu(pt(l)), lo(h(1)), lo(h(k)))
                     else
                        if (pt(j) / norb /= 0 .or. pt(l) / norb /= 1) cycle
                        amp = t2b(lu(pt(j)), lu(pt(l)), lo(h(1)), lo(h(k)))
                     end if
                     if (amp == 0.0d0) cycle
                     hr(1:k-2) = h(2:k-1); hr(k-1:n-2) = h(k+1:n)
                     pr(1:j-1) = pt(1:j-1); pr(j:l-2) = pt(j+1:l-1); pr(l-1:n-2) = pt(l+1:n)
                     call cluster_coefficient(cs, hr(1:n-2), pr(1:n-2), n-2, t1a, t1b, t2a, t2b, t2c, int_h, int_p,&
                                              int_k, int_amp, mr, nint, lo, lu, inactive, prune, noa, nua, nob, nub, norb)
                     c = c + (1 - 2*mod(j+l+k-1, 2)) * amp * cs
                  end do
               end do
            end do
            ! internal blocks of rank >= 3 whose lowest hole is H1 and that lie inside (H, P)
            do e = 1, nint
               kk = int_k(e)
               if (kk < 3 .or. kk > n .or. int_amp(e) == 0.0d0) cycle
               if (int_h(1,e) /= h(1)) cycle
               found = .true.
               do m = 1, kk
                  hpos(m) = findloc(h(1:n), int_h(m,e), dim=1)
                  ppos(m) = findloc(pt(1:n), int_p(m,e), dim=1)
                  if (hpos(m) == 0 .or. ppos(m) == 0) found = .false.
               end do
               if (.not. found) cycle
               used_h(1:n) = .false.; used_p(1:n) = .false.
               par = 0
               do m = 1, kk
                  used_h(hpos(m)) = .true.
                  used_p(ppos(m)) = .true.
                  par = par + hpos(m) - m + ppos(m) - m
               end do
               nr = 0
               do m = 1, n
                  if (.not. used_h(m)) then
                     nr = nr + 1; hr(nr) = h(m)
                  end if
               end do
               nr = 0
               do m = 1, n
                  if (.not. used_p(m)) then
                     nr = nr + 1; pr(nr) = pt(m)
                  end if
               end do
               call cluster_coefficient(cs, hr(1:n-kk), pr(1:n-kk), n-kk, t1a, t1b, t2a, t2b, t2c, int_h, int_p, int_k,&
                                        int_amp, mr, nint, lo, lu, inactive, prune, noa, nua, nob, nub, norb)
               c = c + (1 - 2*mod(par, 2)) * int_amp(e) * cs
            end do
      end subroutine cluster_coefficient

      ! ------------------------------------------------------------------ projections
      ! P(q->p) on the singles and doubles of p (fully antisymmetric arrays)
      subroutine project_cc_wavefunction(p1a, p1b, p2a, p2b, p2c, t1a, t1b, t2a, t2b, t2c,&
                                         int_h, int_p, int_k, int_amp, inactive, prune,&
                                         occa_p, unocca_p, occb_p, unoccb_p, occa_q, unocca_q, occb_q, unoccb_q,&
                                         mr, nint, noa, nua, nob, nub, norb)
            integer, intent(in) :: mr, nint, prune, noa, nua, nob, nub, norb
            integer, intent(in) :: occa_p(noa), unocca_p(nua), occb_p(nob), unoccb_p(nub)
            integer, intent(in) :: occa_q(noa), unocca_q(nua), occb_q(nob), unoccb_q(nub)
            integer, intent(in) :: int_h(mr,nint), int_p(mr,nint), int_k(nint), inactive(0:2*norb-1)
            real(kind=8), intent(in) :: int_amp(nint)
            real(kind=8), intent(in) :: t1a(nua,noa), t1b(nub,nob)
            real(kind=8), intent(in) :: t2a(nua,nua,noa,noa), t2b(nua,nub,noa,nob), t2c(nub,nub,nob,nob)
            real(kind=8), intent(out) :: p1a(nua,noa), p1b(nub,nob)
            real(kind=8), intent(out) :: p2a(nua,nua,noa,noa), p2b(nua,nub,noa,nob), p2c(nub,nub,nob,nob)

            integer :: lo(0:2*norb-1), lu(0:2*norb-1), lo_p(0:2*norb-1), lu_p(0:2*norb-1)
            logical :: det_p(0:2*norb-1), det_q(0:2*norb-1)
            integer :: cnt_p(0:2*norb-1), cnt_q(0:2*norb-1), rel_a(2*norb), rel_b(2*norb), na, nb
            integer :: a, b, i, j
            real(kind=8) :: val

            call orbital_maps(lo, lu, det_q, occa_q, unocca_q, occb_q, unoccb_q, noa, nua, nob, nub, norb)
            call orbital_maps(lo_p, lu_p, det_p, occa_p, unocca_p, occb_p, unoccb_p, noa, nua, nob, nub, norb)
            call pair_context(cnt_p, cnt_q, rel_a, na, rel_b, nb, det_p, det_q, 2*norb)
            p1a = 0.0d0; p1b = 0.0d0; p2a = 0.0d0; p2b = 0.0d0; p2c = 0.0d0

            !$omp parallel do private(i, val) schedule(dynamic) if(nua*nua*noa*noa >= 4096)
            do a = 1, nua
               do i = 1, noa
                  call coef((/unocca_p(a)/), (/occa_p(i)/), 1, val)
                  p1a(a,i) = val
               end do
            end do
            !$omp end parallel do
            !$omp parallel do private(i, val) schedule(dynamic) if(nua*nua*noa*noa >= 4096)
            do a = 1, nub
               do i = 1, nob
                  call coef((/norb + unoccb_p(a)/), (/norb + occb_p(i)/), 1, val)
                  p1b(a,i) = val
               end do
            end do
            !$omp end parallel do
            !$omp parallel do private(b, i, j, val) schedule(dynamic) if(nua*nub*noa*nob >= 4096)
            do a = 1, nua
               do b = a + 1, nua
                  do i = 1, noa
                     do j = i + 1, noa
                        call coef((/unocca_p(a), unocca_p(b)/), (/occa_p(i), occa_p(j)/), 2, val)
                        p2a(a,b,i,j) = val; p2a(b,a,i,j) = -val; p2a(a,b,j,i) = -val; p2a(b,a,j,i) = val
                     end do
                  end do
               end do
            end do
            !$omp end parallel do
            !$omp parallel do private(b, i, j, val) schedule(dynamic) if(nua*nub*noa*nob >= 4096)
            do a = 1, nua
               do b = 1, nub
                  do i = 1, noa
                     do j = 1, nob
                        call coef((/unocca_p(a), norb + unoccb_p(b)/), (/occa_p(i), norb + occb_p(j)/), 2, val)
                        p2b(a,b,i,j) = val
                     end do
                  end do
               end do
            end do
            !$omp end parallel do
            !$omp parallel do private(b, i, j, val) schedule(dynamic) if(nua*nub*noa*nob >= 4096)
            do a = 1, nub
               do b = a + 1, nub
                  do i = 1, nob
                     do j = i + 1, nob
                        call coef((/norb + unoccb_p(a), norb + unoccb_p(b)/), (/norb + occb_p(i), norb + occb_p(j)/), 2, val)
                        p2c(a,b,i,j) = val; p2c(b,a,i,j) = -val; p2c(a,b,j,i) = -val; p2c(b,a,j,i) = val
                     end do
                  end do
               end do
            end do
            !$omp end parallel do

            contains

            subroutine coef(x, y, m, val)
                  integer, intent(in) :: m, x(m), y(m)
                  real(kind=8), intent(out) :: val
                  call excitation_coefficient(val, x, y, m, det_q, cnt_p, cnt_q, rel_a(1:na), na, rel_b(1:nb), nb,&
                                              t1a, t1b, t2a, t2b, t2c, int_h, int_p, int_k, int_amp, mr, nint,&
                                              lo, lu, inactive, prune, noa, nua, nob, nub, norb)
            end subroutine coef

      end subroutine project_cc_wavefunction

      ! P3(q->p) on the unique triples of p (a<b<c, i<j<k within each spin; the rest is zero)
      subroutine project_triples(p3a, p3b, p3c, p3d, t1a, t1b, t2a, t2b, t2c,&
                                 int_h, int_p, int_k, int_amp, inactive, prune,&
                                 occa_p, unocca_p, occb_p, unoccb_p, occa_q, unocca_q, occb_q, unoccb_q,&
                                 mr, nint, noa, nua, nob, nub, norb)
            integer, intent(in) :: mr, nint, prune, noa, nua, nob, nub, norb
            integer, intent(in) :: occa_p(noa), unocca_p(nua), occb_p(nob), unoccb_p(nub)
            integer, intent(in) :: occa_q(noa), unocca_q(nua), occb_q(nob), unoccb_q(nub)
            integer, intent(in) :: int_h(mr,nint), int_p(mr,nint), int_k(nint), inactive(0:2*norb-1)
            real(kind=8), intent(in) :: int_amp(nint)
            real(kind=8), intent(in) :: t1a(nua,noa), t1b(nub,nob)
            real(kind=8), intent(in) :: t2a(nua,nua,noa,noa), t2b(nua,nub,noa,nob), t2c(nub,nub,nob,nob)
            real(kind=8), intent(out) :: p3a(nua,nua,nua,noa,noa,noa), p3b(nua,nua,nub,noa,noa,nob)
            real(kind=8), intent(out) :: p3c(nua,nub,nub,noa,nob,nob), p3d(nub,nub,nub,nob,nob,nob)

            integer :: lo(0:2*norb-1), lu(0:2*norb-1), lo_p(0:2*norb-1), lu_p(0:2*norb-1)
            logical :: det_p(0:2*norb-1), det_q(0:2*norb-1)
            integer :: cnt_p(0:2*norb-1), cnt_q(0:2*norb-1), rel_a(2*norb), rel_b(2*norb), na, nb
            integer :: a, b, c, i, j, k
            real(kind=8) :: val

            call orbital_maps(lo, lu, det_q, occa_q, unocca_q, occb_q, unoccb_q, noa, nua, nob, nub, norb)
            call orbital_maps(lo_p, lu_p, det_p, occa_p, unocca_p, occb_p, unoccb_p, noa, nua, nob, nub, norb)
            call pair_context(cnt_p, cnt_q, rel_a, na, rel_b, nb, det_p, det_q, 2*norb)
            p3a = 0.0d0; p3b = 0.0d0; p3c = 0.0d0; p3d = 0.0d0
            !$omp parallel do private(b, c, i, j, k, val) schedule(dynamic) if(nua*nub*noa*nob >= 4096)
            do a = 1, nua
            do b = a+1, nua; do c = b+1, nua; do i = 1, noa; do j = i+1, noa; do k = j+1, noa
               call coef((/unocca_p(a), unocca_p(b), unocca_p(c)/), (/occa_p(i), occa_p(j), occa_p(k)/), val)
               p3a(a,b,c,i,j,k) = val
            end do; end do; end do; end do; end do
            end do
            !$omp end parallel do
            !$omp parallel do private(b, c, i, j, k, val) schedule(dynamic) if(nua*nub*noa*nob >= 4096)
            do a = 1, nua
            do b = a+1, nua; do c = 1, nub; do i = 1, noa; do j = i+1, noa; do k = 1, nob
               call coef((/unocca_p(a), unocca_p(b), norb + unoccb_p(c)/), (/occa_p(i), occa_p(j), norb + occb_p(k)/), val)
               p3b(a,b,c,i,j,k) = val
            end do; end do; end do; end do; end do
            end do
            !$omp end parallel do
            !$omp parallel do private(b, c, i, j, k, val) schedule(dynamic) if(nua*nub*noa*nob >= 4096)
            do a = 1, nua
            do b = 1, nub; do c = b+1, nub; do i = 1, noa; do j = 1, nob; do k = j+1, nob
               call coef((/unocca_p(a), norb + unoccb_p(b), norb + unoccb_p(c)/), (/occa_p(i), norb + occb_p(j), norb + occb_p(k)/), val)
               p3c(a,b,c,i,j,k) = val
            end do; end do; end do; end do; end do
            end do
            !$omp end parallel do
            !$omp parallel do private(b, c, i, j, k, val) schedule(dynamic) if(nua*nub*noa*nob >= 4096)
            do a = 1, nub
            do b = a+1, nub; do c = b+1, nub; do i = 1, nob; do j = i+1, nob; do k = j+1, nob
               call coef((/norb + unoccb_p(a), norb + unoccb_p(b), norb + unoccb_p(c)/),&
                         (/norb + occb_p(i), norb + occb_p(j), norb + occb_p(k)/), val)
               p3d(a,b,c,i,j,k) = val
            end do; end do; end do; end do; end do
            end do
            !$omp end parallel do

            contains

            subroutine coef(x, y, val)
                  integer, intent(in) :: x(3), y(3)
                  real(kind=8), intent(out) :: val
                  call excitation_coefficient(val, x, y, 3, det_q, cnt_p, cnt_q, rel_a(1:na), na, rel_b(1:nb), nb,&
                                              t1a, t1b, t2a, t2b, t2c, int_h, int_p, int_k, int_amp, mr, nint,&
                                              lo, lu, inactive, prune, noa, nua, nob, nub, norb)
            end subroutine coef

      end subroutine project_triples

      ! C conditions: int_amp(e) = -(all disconnected clusters of excitation e), for entries
      ! sorted by ascending rank, so that <Phi_r|exp(T(q))|Phi_q> = 0 for the reference r = e Phi_q.
      subroutine internal_amplitudes(int_amp, t1a, t1b, t2a, t2b, t2c, int_h, int_p, int_k, inactive, prune,&
                                     occa_q, unocca_q, occb_q, unoccb_q, mr, nint, noa, nua, nob, nub, norb)
            integer, intent(in) :: mr, nint, prune, noa, nua, nob, nub, norb
            integer, intent(in) :: occa_q(noa), unocca_q(nua), occb_q(nob), unoccb_q(nub)
            integer, intent(in) :: int_h(mr,nint), int_p(mr,nint), int_k(nint), inactive(0:2*norb-1)
            real(kind=8), intent(in) :: t1a(nua,noa), t1b(nub,nob)
            real(kind=8), intent(in) :: t2a(nua,nua,noa,noa), t2b(nua,nub,noa,nob), t2c(nub,nub,nob,nob)
            real(kind=8), intent(out) :: int_amp(nint)
            integer :: lo(0:2*norb-1), lu(0:2*norb-1), e, kk
            logical :: det_q(0:2*norb-1)
            real(kind=8) :: c

            call orbital_maps(lo, lu, det_q, occa_q, unocca_q, occb_q, unoccb_q, noa, nua, nob, nub, norb)
            int_amp = 0.0d0
            do e = 1, nint
               kk = int_k(e)
               if (kk == 0) cycle
               ! int_amp(e) is still zero, so c holds the disconnected clusters only
               call cluster_coefficient(c, int_h(1:kk,e), int_p(1:kk,e), kk, t1a, t1b, t2a, t2b, t2c,&
                                        int_h, int_p, int_k, int_amp, mr, nint, lo, lu, inactive, prune,&
                                        noa, nua, nob, nub, norb)
               int_amp(e) = -c
            end do
      end subroutine internal_amplitudes

      ! ------------------------------------------------------------------ MM-SUCC(2,3) (dense reference helpers)
      ! [exp(-T(p)) C]_K on the unique triples K of reference p, for a vector C given by its
      ! singles, doubles and triples (coefficient of Phi_p zero): the sum over sub-excitations L of K
      ! of <K|exp(-T(p))|L> C_L. t are the amplitudes of T(p) (internal t2 included).
      subroutine coupling_triples(x3a, x3b, x3c, x3d, c1a, c1b, c2a, c2b, c2c, c3a, c3b, c3c, c3d,&
                                  t1a, t1b, t2a, t2b, t2c, occa_p, unocca_p, occb_p, unoccb_p,&
                                  noa, nua, nob, nub, norb)
            integer, intent(in) :: noa, nua, nob, nub, norb
            integer, intent(in) :: occa_p(noa), unocca_p(nua), occb_p(nob), unoccb_p(nub)
            real(kind=8), intent(in) :: c1a(nua,noa), c1b(nub,nob)
            real(kind=8), intent(in) :: c2a(nua,nua,noa,noa), c2b(nua,nub,noa,nob), c2c(nub,nub,nob,nob)
            real(kind=8), intent(in) :: c3a(nua,nua,nua,noa,noa,noa), c3b(nua,nua,nub,noa,noa,nob)
            real(kind=8), intent(in) :: c3c(nua,nub,nub,noa,nob,nob), c3d(nub,nub,nub,nob,nob,nob)
            real(kind=8), intent(in) :: t1a(nua,noa), t1b(nub,nob)
            real(kind=8), intent(in) :: t2a(nua,nua,noa,noa), t2b(nua,nub,noa,nob), t2c(nub,nub,nob,nob)
            real(kind=8), intent(out) :: x3a(nua,nua,nua,noa,noa,noa), x3b(nua,nua,nub,noa,noa,nob)
            real(kind=8), intent(out) :: x3c(nua,nub,nub,noa,nob,nob), x3d(nub,nub,nub,nob,nob,nob)

            integer :: lo(0:2*norb-1), lu(0:2*norb-1)
            logical :: det_p(0:2*norb-1)
            real(kind=8) :: m1a(nua,noa), m1b(nub,nob)
            real(kind=8) :: m2a(nua,nua,noa,noa), m2b(nua,nub,noa,nob), m2c(nub,nub,nob,nob)
            integer :: dum_h(1,1), dum_p(1,1), dum_k(1), inactive(0:2*norb-1)
            real(kind=8) :: dum_amp(1)
            integer :: a, b, c, i, j, k

            call orbital_maps(lo, lu, det_p, occa_p, unocca_p, occb_p, unoccb_p, noa, nua, nob, nub, norb)
            m1a = -t1a; m1b = -t1b; m2a = -t2a; m2b = -t2b; m2c = -t2c
            dum_h = 0; dum_p = 0; dum_k = 0; dum_amp = 0.0d0; inactive = 0
            x3a = 0.0d0; x3b = 0.0d0; x3c = 0.0d0; x3d = 0.0d0
            do i = 1, noa; do j = i+1, noa; do k = j+1, noa
            do a = 1, nua; do b = a+1, nua; do c = b+1, nua
               x3a(a,b,c,i,j,k) = element((/occa_p(i), occa_p(j), occa_p(k)/), (/unocca_p(a), unocca_p(b), unocca_p(c)/))
            end do; end do; end do; end do; end do; end do
            do i = 1, noa; do j = i+1, noa; do k = 1, nob
            do a = 1, nua; do b = a+1, nua; do c = 1, nub
               x3b(a,b,c,i,j,k) = element((/occa_p(i), occa_p(j), norb + occb_p(k)/),&
                                          (/unocca_p(a), unocca_p(b), norb + unoccb_p(c)/))
            end do; end do; end do; end do; end do; end do
            do i = 1, noa; do j = 1, nob; do k = j+1, nob
            do a = 1, nua; do b = 1, nub; do c = b+1, nub
               x3c(a,b,c,i,j,k) = element((/occa_p(i), norb + occb_p(j), norb + occb_p(k)/),&
                                          (/unocca_p(a), norb + unoccb_p(b), norb + unoccb_p(c)/))
            end do; end do; end do; end do; end do; end do
            do i = 1, nob; do j = i+1, nob; do k = j+1, nob
            do a = 1, nub; do b = a+1, nub; do c = b+1, nub
               x3d(a,b,c,i,j,k) = element((/norb + occb_p(i), norb + occb_p(j), norb + occb_p(k)/),&
                                          (/norb + unoccb_p(a), norb + unoccb_p(b), norb + unoccb_p(c)/))
            end do; end do; end do; end do; end do; end do

            contains

            ! sum over sub-excitations L = (holes hl, particles pl) of K = (h, pt), canonical order
            function element(h, pt) result(val)
                  integer, intent(in) :: h(3), pt(3)
                  real(kind=8) :: val, cc, cl
                  integer :: mh, mp, r, m, nh, np_, hl(3), pl(3), hc(3), pc(3), par
                  logical :: inh(3), inp(3)
                  val = 0.0d0
                  do mh = 1, 7          ! nonempty hole subsets (bit masks)
                     do mp = 1, 7       ! nonempty particle subsets
                        if (popcnt(mh) /= popcnt(mp)) cycle
                        r = popcnt(mh)
                        nh = 0; np_ = 0; par = 0
                        hc = 0; pc = 0
                        do m = 1, 3
                           inh(m) = btest(mh, m - 1)
                           inp(m) = btest(mp, m - 1)
                        end do
                        ! complement block (from -T(p)): positions not in L; sign as for a block
                        do m = 1, 3
                           if (.not. inh(m)) then
                              nh = nh + 1; hc(nh) = h(m); par = par + m - nh
                           end if
                           if (.not. inp(m)) then
                              np_ = np_ + 1; pc(np_) = pt(m); par = par + m - np_
                           end if
                        end do
                        if (r == 3) then
                           cc = 1.0d0
                        else
                           call cluster_coefficient(cc, hc(1:3-r), pc(1:3-r), 3 - r, m1a, m1b, m2a, m2b, m2c,&
                                                    dum_h, dum_p, dum_k, dum_amp, 1, 1, lo, lu, inactive, 0,&
                                                    noa, nua, nob, nub, norb)
                        end if
                        if (cc == 0.0d0) cycle
                        nh = 0; np_ = 0
                        do m = 1, 3
                           if (inh(m)) then
                              nh = nh + 1; hl(nh) = h(m)
                           end if
                           if (inp(m)) then
                              np_ = np_ + 1; pl(np_) = pt(m)
                           end if
                        end do
                        cl = lookup(hl, pl, r)
                        val = val + (1 - 2*mod(par, 2)) * cc * cl
                     end do
                  end do
            end function element

            ! C_L for the canonical excitation (holes hl, particles pl) of rank r
            function lookup(hl, pl, r) result(cl)
                  integer, intent(in) :: hl(3), pl(3), r
                  real(kind=8) :: cl
                  integer :: s(3), sp(3), m
                  cl = 0.0d0
                  do m = 1, r
                     s(m) = hl(m) / norb
                     sp(m) = pl(m) / norb
                  end do
                  ! spin balance: the alpha holes and particles must be as many
                  if (count(s(1:r) == 0) /= count(sp(1:r) == 0)) return
                  if (r == 1) then
                     if (s(1) == 0) then
                        cl = c1a(lu(pl(1)), lo(hl(1)))
                     else
                        cl = c1b(lu(pl(1)), lo(hl(1)))
                     end if
                  else if (r == 2) then
                     if (s(1) == 0 .and. s(2) == 0) then
                        cl = c2a(lu(pl(1)), lu(pl(2)), lo(hl(1)), lo(hl(2)))
                     else if (s(1) == 1) then
                        cl = c2c(lu(pl(1)), lu(pl(2)), lo(hl(1)), lo(hl(2)))
                     else
                        cl = c2b(lu(pl(1)), lu(pl(2)), lo(hl(1)), lo(hl(2)))
                     end if
                  else
                     select case (count(s(1:3) == 0))
                     case (3)
                        cl = c3a(lu(pl(1)), lu(pl(2)), lu(pl(3)), lo(hl(1)), lo(hl(2)), lo(hl(3)))
                     case (2)
                        cl = c3b(lu(pl(1)), lu(pl(2)), lu(pl(3)), lo(hl(1)), lo(hl(2)), lo(hl(3)))
                     case (1)
                        cl = c3c(lu(pl(1)), lu(pl(2)), lu(pl(3)), lo(hl(1)), lo(hl(2)), lo(hl(3)))
                     case default
                        cl = c3d(lu(pl(1)), lu(pl(2)), lu(pl(3)), lo(hl(1)), lo(hl(2)), lo(hl(3)))
                     end select
                  end if
            end function lookup

      end subroutine coupling_triples

      ! Coefficients of the linear wave function (1 + T1 + T2 + Z3)(q)|Phi_q> on the unique triples
      ! of reference p (z3 in the local indices of q, fully antisymmetric).
      subroutine linear_triples(l3a, l3b, l3c, l3d, t1a, t1b, t2a, t2b, t2c, z3a, z3b, z3c, z3d,&
                                occa_p, unocca_p, occb_p, unoccb_p, occa_q, unocca_q, occb_q, unoccb_q,&
                                noa, nua, nob, nub, norb)
            integer, intent(in) :: noa, nua, nob, nub, norb
            integer, intent(in) :: occa_p(noa), unocca_p(nua), occb_p(nob), unoccb_p(nub)
            integer, intent(in) :: occa_q(noa), unocca_q(nua), occb_q(nob), unoccb_q(nub)
            real(kind=8), intent(in) :: t1a(nua,noa), t1b(nub,nob)
            real(kind=8), intent(in) :: t2a(nua,nua,noa,noa), t2b(nua,nub,noa,nob), t2c(nub,nub,nob,nob)
            real(kind=8), intent(in) :: z3a(nua,nua,nua,noa,noa,noa), z3b(nua,nua,nub,noa,noa,nob)
            real(kind=8), intent(in) :: z3c(nua,nub,nub,noa,nob,nob), z3d(nub,nub,nub,nob,nob,nob)
            real(kind=8), intent(out) :: l3a(nua,nua,nua,noa,noa,noa), l3b(nua,nua,nub,noa,noa,nob)
            real(kind=8), intent(out) :: l3c(nua,nub,nub,noa,nob,nob), l3d(nub,nub,nub,nob,nob,nob)

            integer :: lo(0:2*norb-1), lu(0:2*norb-1), lo_p(0:2*norb-1), lu_p(0:2*norb-1)
            logical :: det_p(0:2*norb-1), det_q(0:2*norb-1)
            integer :: a, b, c, i, j, k

            call orbital_maps(lo, lu, det_q, occa_q, unocca_q, occb_q, unoccb_q, noa, nua, nob, nub, norb)
            call orbital_maps(lo_p, lu_p, det_p, occa_p, unocca_p, occb_p, unoccb_p, noa, nua, nob, nub, norb)
            l3a = 0.0d0; l3b = 0.0d0; l3c = 0.0d0; l3d = 0.0d0
            do i = 1, noa; do j = i+1, noa; do k = j+1, noa
            do a = 1, nua; do b = a+1, nua; do c = b+1, nua
               l3a(a,b,c,i,j,k) = lin((/unocca_p(a), unocca_p(b), unocca_p(c)/), (/occa_p(i), occa_p(j), occa_p(k)/))
            end do; end do; end do; end do; end do; end do
            do i = 1, noa; do j = i+1, noa; do k = 1, nob
            do a = 1, nua; do b = a+1, nua; do c = 1, nub
               l3b(a,b,c,i,j,k) = lin((/unocca_p(a), unocca_p(b), norb + unoccb_p(c)/), (/occa_p(i), occa_p(j), norb + occb_p(k)/))
            end do; end do; end do; end do; end do; end do
            do i = 1, noa; do j = 1, nob; do k = j+1, nob
            do a = 1, nua; do b = 1, nub; do c = b+1, nub
               l3c(a,b,c,i,j,k) = lin((/unocca_p(a), norb + unoccb_p(b), norb + unoccb_p(c)/),&
                                      (/occa_p(i), norb + occb_p(j), norb + occb_p(k)/))
            end do; end do; end do; end do; end do; end do
            do i = 1, nob; do j = i+1, nob; do k = j+1, nob
            do a = 1, nub; do b = a+1, nub; do c = b+1, nub
               l3d(a,b,c,i,j,k) = lin((/norb + unoccb_p(a), norb + unoccb_p(b), norb + unoccb_p(c)/),&
                                      (/norb + occb_p(i), norb + occb_p(j), norb + occb_p(k)/))
            end do; end do; end do; end do; end do; end do

            contains

            function lin(x, y) result(val)
                  integer, intent(in) :: x(3), y(3)
                  real(kind=8) :: val, sgn, amp
                  logical :: d(0:2*norb-1), dd(0:2*norb-1)
                  integer :: h(16), pt(16), n, np_, s, m, na
                  d = det_p
                  sgn = 1.0d0
                  do m = 1, 3
                     call annihilate(d, y(m), sgn, 2*norb)
                  end do
                  do m = 3, 1, -1
                     call create(d, x(m), sgn, 2*norb)
                  end do
                  val = 0.0d0
                  n = 0; np_ = 0; h = 0; pt = 0
                  do s = 0, 2*norb - 1
                     if (det_q(s) .and. .not. d(s)) then
                        n = n + 1
                        if (n > 3) return
                        h(n) = s
                     end if
                     if (d(s) .and. .not. det_q(s)) then
                        np_ = np_ + 1
                        if (np_ > 3) return
                        pt(np_) = s
                     end if
                  end do
                  ! canonical string relative to q: E_canon |Phi_q> = sgn' |D>
                  dd = det_q
                  do m = 1, n
                     call annihilate(dd, h(m), sgn, 2*norb)
                  end do
                  do m = n, 1, -1
                     call create(dd, pt(m), sgn, 2*norb)
                  end do
                  na = count(h(1:n) < norb)
                  if (count(pt(1:n) < norb) /= na) return
                  select case (n)
                  case (0)
                     amp = 1.0d0
                  case (1)
                     if (na == 1) then
                        amp = t1a(lu(pt(1)), lo(h(1)))
                     else
                        amp = t1b(lu(pt(1)), lo(h(1)))
                     end if
                  case (2)
                     if (na == 2) then
                        amp = t2a(lu(pt(1)), lu(pt(2)), lo(h(1)), lo(h(2)))
                     else if (na == 1) then
                        amp = t2b(lu(pt(1)), lu(pt(2)), lo(h(1)), lo(h(2)))
                     else
                        amp = t2c(lu(pt(1)), lu(pt(2)), lo(h(1)), lo(h(2)))
                     end if
                  case default
                     select case (na)
                     case (3)
                        amp = z3a(lu(pt(1)), lu(pt(2)), lu(pt(3)), lo(h(1)), lo(h(2)), lo(h(3)))
                     case (2)
                        amp = z3b(lu(pt(1)), lu(pt(2)), lu(pt(3)), lo(h(1)), lo(h(2)), lo(h(3)))
                     case (1)
                        amp = z3c(lu(pt(1)), lu(pt(2)), lu(pt(3)), lo(h(1)), lo(h(2)), lo(h(3)))
                     case default
                        amp = z3d(lu(pt(1)), lu(pt(2)), lu(pt(3)), lo(h(1)), lo(h(2)), lo(h(3)))
                     end select
                  end select
                  val = sgn * amp
            end function lin

      end subroutine linear_triples


      ! a_s and a+_s on an occupation string, with the sign (-1)^(number occupied below s)
      subroutine annihilate(d, s, sgn, nso)
            integer, intent(in) :: nso
            logical, intent(inout) :: d(0:nso-1)
            integer, intent(in) :: s
            real(kind=8), intent(inout) :: sgn
            if (mod(count(d(0:s-1)), 2) == 1) sgn = -sgn
            d(s) = .false.
      end subroutine annihilate

      subroutine create(d, s, sgn, nso)
            integer, intent(in) :: nso
            logical, intent(inout) :: d(0:nso-1)
            integer, intent(in) :: s
            real(kind=8), intent(inout) :: sgn
            if (mod(count(d(0:s-1)), 2) == 1) sgn = -sgn
            d(s) = .true.
      end subroutine create

      ! MM-SUCC(2,3) contributions of reference p, on the fly over its unique triples K (internal
      ! ones skipped): num(mu) += c_pmu O_mu(K) Gamma_K(p) and s3(q) += Z3(p)_K P3(q->p)_K, with
      !   Gamma_K = M3_K - [e^{-T(p)} Ct(p)]_K,  Ct = sum_{q /= p} Heff_qp P(q->p),
      !   O_mu(K) = c_pmu Z3(p)_K + sum_{q /= p} c_qmu <(1 + T1 + T2 + Z3)(q) Phi_q|K(p)>,
      !   Z3(r) = [(V_N T2(r))_C + (V_N T1(r))_DC] / D3(r),  M3 = A[-I_vooo t2 + I_vvov t2] with Hbar
      ! intermediates (the cores of ccpy.cc.ccsdt.update_t3a-d). Per-reference data are stacked on a
      ! last dimension of size nref. In addition, olist(mu, n) = O_mu for the triples klist(:, n) of
      ! spin case kcase(n) (1 aaa, 2 aab, 3 abb, 4 bbb; 0 = none; any index order, local indices of p).
      subroutine mm23_reference(num, s3, olist, p, coef, heffcol,&
                                occa, unocca, occb, unoccb,&
                                t1a, t1b, t2a, t2b, t2c, int_h, int_p, int_k, int_amp, nint_ref, inactive, prune,&
                                va_vooo, va_vvov, vb_vvvo, vb_ovoo, vb_vvov, vb_vooo, vc_vvov, vc_vooo,&
                                va_vvoo, vb_vvoo, vc_vvoo, foa, fva, fob, fvb,&
                                ha_vooo, ha_vvov, hb_vvvo, hb_ovoo, hb_vvov, hb_vooo, hc_vvov, hc_vooo,&
                                c1a, c1b, c2a, c2b, c2c, klist, kcase,&
                                mr, nref, nroot, nint, nlist, noa, nua, nob, nub, norb)
            integer, intent(in) :: p, mr, nref, nroot, nint, nlist, noa, nua, nob, nub, norb
            real(kind=8), intent(in) :: coef(nref,nroot), heffcol(nref)
            integer, intent(in) :: occa(noa,nref), unocca(nua,nref), occb(nob,nref), unoccb(nub,nref)
            real(kind=8), intent(in) :: t1a(nua,noa,nref), t1b(nub,nob,nref)
            real(kind=8), intent(in) :: t2a(nua,nua,noa,noa,nref), t2b(nua,nub,noa,nob,nref), t2c(nub,nub,nob,nob,nref)
            integer, intent(in) :: int_h(mr,nint,nref), int_p(mr,nint,nref), int_k(nint,nref), nint_ref(nref)
            integer, intent(in) :: inactive(0:2*norb-1), prune(nref)
            real(kind=8), intent(in) :: int_amp(nint,nref)
            real(kind=8), intent(in) :: va_vooo(nua,noa,noa,noa,nref), va_vvov(nua,nua,noa,nua,nref)
            real(kind=8), intent(in) :: vb_vvvo(nua,nub,nua,nob,nref), vb_ovoo(noa,nub,noa,nob,nref)
            real(kind=8), intent(in) :: vb_vvov(nua,nub,noa,nub,nref), vb_vooo(nua,nob,noa,nob,nref)
            real(kind=8), intent(in) :: vc_vvov(nub,nub,nob,nub,nref), vc_vooo(nub,nob,nob,nob,nref)
            real(kind=8), intent(in) :: va_vvoo(nua,nua,noa,noa,nref), vb_vvoo(nua,nub,noa,nob,nref)
            real(kind=8), intent(in) :: vc_vvoo(nub,nub,nob,nob,nref)
            real(kind=8), intent(in) :: foa(noa,nref), fva(nua,nref), fob(nob,nref), fvb(nub,nref)
            real(kind=8), intent(in) :: ha_vooo(nua,noa,noa,noa), ha_vvov(nua,nua,noa,nua)
            real(kind=8), intent(in) :: hb_vvvo(nua,nub,nua,nob), hb_ovoo(noa,nub,noa,nob)
            real(kind=8), intent(in) :: hb_vvov(nua,nub,noa,nub), hb_vooo(nua,nob,noa,nob)
            real(kind=8), intent(in) :: hc_vvov(nub,nub,nob,nub), hc_vooo(nub,nob,nob,nob)
            real(kind=8), intent(in) :: c1a(nua,noa), c1b(nub,nob)
            real(kind=8), intent(in) :: c2a(nua,nua,noa,noa), c2b(nua,nub,noa,nob), c2c(nub,nub,nob,nob)
            integer, intent(in) :: klist(6,nlist), kcase(nlist)
            real(kind=8), intent(out) :: num(nroot), s3(nref), olist(nroot,nlist)

            integer :: lo(0:2*norb-1,nref), lu(0:2*norb-1,nref)
            logical :: dets(0:2*norb-1,nref)
            integer :: cnt(0:2*norb-1,nref), rel_a(2*norb,nref), rel_b(2*norb,nref), na(nref), nb(nref), dummy(0:2*norb-1)
            integer :: r, a, b, c, i, j, k, n
            ! permutations of three indices and their signs
            integer, parameter :: perm3(3,6) = reshape((/1,2,3, 2,1,3, 1,3,2, 3,2,1, 2,3,1, 3,1,2/), (/3,6/))
            real(kind=8), parameter :: sign3(6) = (/1.0d0, -1.0d0, -1.0d0, -1.0d0, 1.0d0, 1.0d0/)
            real(kind=8) :: o(nroot), gam, dnum(nroot), ds3(nref)
            ! -T(p), for the e^{-T(p)} coefficients of the coupling term
            real(kind=8) :: mt1a(nua,noa), mt1b(nub,nob)
            real(kind=8), allocatable :: mt2a(:,:,:,:), mt2b(:,:,:,:), mt2c(:,:,:,:)

            dummy = 0
            mt1a = -t1a(:,:,p); mt1b = -t1b(:,:,p)
            allocate(mt2a(nua,nua,noa,noa), mt2b(nua,nub,noa,nob), mt2c(nub,nub,nob,nob))
            mt2a = -t2a(:,:,:,:,p); mt2b = -t2b(:,:,:,:,p); mt2c = -t2c(:,:,:,:,p)
            do r = 1, nref
               call orbital_maps(lo(:,r), lu(:,r), dets(:,r), occa(:,r), unocca(:,r), occb(:,r), unoccb(:,r),&
                                 noa, nua, nob, nub, norb)
            end do
            do r = 1, nref
               call pair_context(dummy, cnt(:,r), rel_a(:,r), na(r), rel_b(:,r), nb(r), dets(:,p), dets(:,r), 2*norb)
            end do
            num = 0.0d0; s3 = 0.0d0; olist = 0.0d0

            !$omp parallel do schedule(dynamic) private(b, c, i, j, k, o, gam, dnum, ds3) reduction(+:num, s3)
            do a = 1, nua
               do b = a+1, nua; do c = b+1, nua
               do i = 1, noa; do j = i+1, noa; do k = j+1, noa
                  call visit(1, (/a,b,c,i,j,k/), .true., o, gam, dnum, ds3)
                  num = num + dnum; s3 = s3 + ds3
               end do; end do; end do; end do; end do
            end do
            !$omp end parallel do
            !$omp parallel do schedule(dynamic) private(b, c, i, j, k, o, gam, dnum, ds3) reduction(+:num, s3)
            do a = 1, nua
               do b = a+1, nua; do c = 1, nub
               do i = 1, noa; do j = i+1, noa; do k = 1, nob
                  call visit(2, (/a,b,c,i,j,k/), .true., o, gam, dnum, ds3)
                  num = num + dnum; s3 = s3 + ds3
               end do; end do; end do; end do; end do
            end do
            !$omp end parallel do
            !$omp parallel do schedule(dynamic) private(b, c, i, j, k, o, gam, dnum, ds3) reduction(+:num, s3)
            do a = 1, nua
               do b = 1, nub; do c = b+1, nub
               do i = 1, noa; do j = 1, nob; do k = j+1, nob
                  call visit(3, (/a,b,c,i,j,k/), .true., o, gam, dnum, ds3)
                  num = num + dnum; s3 = s3 + ds3
               end do; end do; end do; end do; end do
            end do
            !$omp end parallel do
            !$omp parallel do schedule(dynamic) private(b, c, i, j, k, o, gam, dnum, ds3) reduction(+:num, s3)
            do a = 1, nub
               do b = a+1, nub; do c = b+1, nub
               do i = 1, nob; do j = i+1, nob; do k = j+1, nob
                  call visit(4, (/a,b,c,i,j,k/), .true., o, gam, dnum, ds3)
                  num = num + dnum; s3 = s3 + ds3
               end do; end do; end do; end do; end do
            end do
            !$omp end parallel do
            !$omp parallel do schedule(dynamic) private(o, gam, dnum, ds3)
            do n = 1, nlist
               if (kcase(n) == 0) cycle
               call visit(kcase(n), klist(:,n), .false., o, gam, dnum, ds3)
               olist(:,n) = o
            end do
            !$omp end parallel do

            contains

            ! spin (0 alpha, 1 beta) of the three particle / hole slots of a spin case
            function slot_spin(cs, m) result(s)
                  integer, intent(in) :: cs, m
                  integer :: s
                  s = 0
                  if (m > 4 - cs) s = 1
            end function slot_spin

            ! global spin orbitals of the triple idx (local indices of reference r)
            subroutine spin_orbitals(cs, idx, r, x, y)
                  integer, intent(in) :: cs, idx(6), r
                  integer, intent(out) :: x(3), y(3)
                  integer :: m
                  do m = 1, 3
                     if (slot_spin(cs, m) == 0) then
                        x(m) = unocca(idx(m), r); y(m) = occa(idx(m+3), r)
                     else
                        x(m) = norb + unoccb(idx(m), r); y(m) = norb + occb(idx(m+3), r)
                     end if
                  end do
            end subroutine spin_orbitals

            ! O_mu(K) (and, with full, the accumulation of num and s3) for the triple idx of p
            subroutine visit(cs, idx, full, o, gam, dnum, ds3)
                  integer, intent(in) :: cs, idx(6)
                  logical, intent(in) :: full
                  real(kind=8), intent(out) :: o(nroot), gam, dnum(nroot), ds3(nref)
                  integer :: x(3), y(3), q, m, qidx(6), na_, nh
                  integer, target :: hbuf(nbuf), pbuf(nbuf)
                  integer, allocatable, target :: halloc(:), palloc(:)
                  integer, pointer, contiguous :: hq(:), pq(:)
                  real(kind=8) :: sgn, sq, z3p, p3, lin, ct3, amp

                  o = 0.0d0; gam = 0.0d0; dnum = 0.0d0; ds3 = 0.0d0
                  call spin_orbitals(cs, idx, p, x, y)
                  ! repeated indices give zero
                  if (x(1) == x(2) .or. x(1) == x(3) .or. x(2) == x(3)) return
                  if (y(1) == y(2) .or. y(1) == y(3) .or. y(2) == y(3)) return
                  if (maxval(na) + 3 <= nbuf .and. maxval(nb) + 3 <= nbuf) then
                     hq => hbuf; pq => pbuf
                  else
                     allocate(halloc(maxval(na) + 3), palloc(maxval(nb) + 3))
                     hq => halloc; pq => palloc
                  end if
                  ! D = E_K Phi_p relative to every reference; D = Phi_q makes K internal
                  do q = 1, nref
                     call relative_excitation(hq, pq, nh, x, y, 3, dets(:,q), rel_a(1:na(q),q), na(q),&
                                              rel_b(1:nb(q),q), nb(q), 2*norb)
                     if (nh == 0) return
                  end do
                  sgn = string_sign(cnt(:,p), y, 3, x, 3, 2*norb)
                  z3p = z3_element(cs, idx, p)
                  o = coef(p,:) * z3p
                  ct3 = 0.0d0
                  do q = 1, nref
                     call relative_excitation(hq, pq, nh, x, y, 3, dets(:,q), rel_a(1:na(q),q), na(q),&
                                              rel_b(1:nb(q),q), nb(q), 2*norb)
                     if (nh < 0) cycle
                     sq = sgn * string_sign(cnt(:,q), hq(1:nh), nh, pq(1:nh), nh, 2*norb)
                     if (full) then
                        call cluster_coefficient(p3, hq(1:nh), pq(1:nh), nh, t1a(:,:,q), t1b(:,:,q),&
                                                 t2a(:,:,:,:,q), t2b(:,:,:,:,q), t2c(:,:,:,:,q),&
                                                 int_h(:,1:max(nint_ref(q),1),q), int_p(:,1:max(nint_ref(q),1),q),&
                                                 int_k(1:max(nint_ref(q),1),q), int_amp(1:max(nint_ref(q),1),q),&
                                                 mr, max(nint_ref(q),1), lo(:,q), lu(:,q), inactive, prune(q),&
                                                 noa, nua, nob, nub, norb)
                        p3 = sq * p3
                        ds3(q) = z3p * p3
                        if (q /= p) ct3 = ct3 + heffcol(q) * p3
                     end if
                     if (q == p .or. nh > 3) cycle
                     ! linear overlap of (1 + T1 + T2 + Z3)(q) Phi_q with D
                     na_ = count(hq(1:nh) < norb)
                     if (count(pq(1:nh) < norb) /= na_) cycle
                     select case (nh)
                     case (1)
                        if (na_ == 1) then
                           amp = t1a(lu(pq(1),q), lo(hq(1),q), q)
                        else
                           amp = t1b(lu(pq(1),q), lo(hq(1),q), q)
                        end if
                     case (2)
                        if (na_ == 2) then
                           amp = t2a(lu(pq(1),q), lu(pq(2),q), lo(hq(1),q), lo(hq(2),q), q)
                        else if (na_ == 1) then
                           amp = t2b(lu(pq(1),q), lu(pq(2),q), lo(hq(1),q), lo(hq(2),q), q)
                        else
                           amp = t2c(lu(pq(1),q), lu(pq(2),q), lo(hq(1),q), lo(hq(2),q), q)
                        end if
                     case (3)
                        do m = 1, 3
                           qidx(m) = lu(pq(m),q)
                           qidx(m+3) = lo(hq(m),q)
                        end do
                        amp = z3_element(4 - na_, qidx, q)
                     case default
                        amp = 0.0d0
                     end select
                     lin = sq * amp
                     o = o + coef(q,:) * lin
                  end do
                  if (.not. full) return
                  gam = moment(cs, idx) - coupling_element(x, y, ct3)
                  dnum = coef(p,:) * o * gam
            end subroutine visit

            ! Z3(r) at the triple idx of spin case cs (local indices of r)
            function z3_element(cs, idx, r) result(z)
                  integer, intent(in) :: cs, idx(6), r
                  real(kind=8) :: z, den
                  integer :: m
                  den = 0.0d0
                  do m = 1, 3
                     if (slot_spin(cs, m) == 0) then
                        den = den + foa(idx(m+3), r) - fva(idx(m), r)
                     else
                        den = den + fob(idx(m+3), r) - fvb(idx(m), r)
                     end if
                  end do
                  z = (antisym(cs, idx, r, .false.) + antisym_y(cs, idx, r)) / den
            end function z3_element

            function moment(cs, idx) result(val)
                  integer, intent(in) :: cs, idx(6)
                  real(kind=8) :: val
                  val = antisym(cs, idx, p, .true.)
            end function moment

            ! A[core] of the MM(2,3) terms; hbar: Hbar intermediates of p, else the bare V of r
            function antisym(cs, idx, r, hbar) result(val)
                  integer, intent(in) :: cs, idx(6), r
                  logical, intent(in) :: hbar
                  real(kind=8) :: val
                  integer :: s1, s2, e(6)
                  val = 0.0d0
                  select case (cs)
                  case (1, 4)
                     do s1 = 1, 6
                        do s2 = 1, 6
                           e(1:3) = idx(perm3(:,s1)); e(4:6) = idx(3 + perm3(:,s2))
                           val = val + sign3(s1) * sign3(s2) * core_same(cs, e, r, hbar)
                        end do
                     end do
                  case (2)
                     do s1 = 1, 2
                        do s2 = 1, 2
                           e = idx
                           if (s1 == 2) e(1:2) = idx((/2,1/))
                           if (s2 == 2) e(4:5) = idx((/5,4/))
                           val = val + (3 - 2*s1) * (3 - 2*s2) * core_aab(e, r, hbar)
                        end do
                     end do
                  case (3)
                     do s1 = 1, 2
                        do s2 = 1, 2
                           e = idx
                           if (s1 == 2) e(2:3) = idx((/3,2/))
                           if (s2 == 2) e(5:6) = idx((/6,5/))
                           val = val + (3 - 2*s1) * (3 - 2*s2) * core_abb(e, r, hbar)
                        end do
                     end do
                  end select
            end function antisym

            ! -1/4 I(amij) t2(bcmk) + 1/4 I(abie) t2(ecjk), same spin (aaa: cs = 1, bbb: cs = 4)
            function core_same(cs, e, r, hbar) result(val)
                  integer, intent(in) :: cs, e(6), r
                  logical, intent(in) :: hbar
                  real(kind=8) :: val
                  integer :: a, b, c, i, j, k
                  a = e(1); b = e(2); c = e(3); i = e(4); j = e(5); k = e(6)
                  if (cs == 1) then
                     if (hbar) then
                        val = -0.25d0 * dot_product(ha_vooo(a,:,i,j), t2a(b,c,:,k,p)) &
                              + 0.25d0 * dot_product(ha_vvov(a,b,i,:), t2a(:,c,j,k,p))
                     else
                        val = -0.25d0 * dot_product(va_vooo(a,:,i,j,r), t2a(b,c,:,k,r)) &
                              + 0.25d0 * dot_product(va_vvov(a,b,i,:,r), t2a(:,c,j,k,r))
                     end if
                  else
                     if (hbar) then
                        val = -0.25d0 * dot_product(hc_vooo(a,:,i,j), t2c(b,c,:,k,p)) &
                              + 0.25d0 * dot_product(hc_vvov(a,b,i,:), t2c(:,c,j,k,p))
                     else
                        val = -0.25d0 * dot_product(vc_vooo(a,:,i,j,r), t2c(b,c,:,k,r)) &
                              + 0.25d0 * dot_product(vc_vvov(a,b,i,:,r), t2c(:,c,j,k,r))
                     end if
                  end if
            end function core_same

            ! cores of ccsdt.update_t3b (a, b alpha; c beta)
            function core_aab(e, r, hbar) result(val)
                  integer, intent(in) :: e(6), r
                  logical, intent(in) :: hbar
                  real(kind=8) :: val
                  integer :: a, b, c, i, j, k
                  a = e(1); b = e(2); c = e(3); i = e(4); j = e(5); k = e(6)
                  if (hbar) then
                     val = 0.5d0 * dot_product(hb_vvvo(b,c,:,k), t2a(a,:,i,j,p)) &
                         - 0.5d0 * dot_product(hb_ovoo(:,c,j,k), t2a(a,b,i,:,p)) &
                         + dot_product(hb_vvov(a,c,i,:), t2b(b,:,j,k,p)) &
                         - dot_product(hb_vooo(a,:,i,k), t2b(b,c,j,:,p)) &
                         + 0.5d0 * dot_product(ha_vvov(a,b,i,:), t2b(:,c,j,k,p)) &
                         - 0.5d0 * dot_product(ha_vooo(a,:,i,j), t2b(b,c,:,k,p))
                  else
                     val = 0.5d0 * dot_product(vb_vvvo(b,c,:,k,r), t2a(a,:,i,j,r)) &
                         - 0.5d0 * dot_product(vb_ovoo(:,c,j,k,r), t2a(a,b,i,:,r)) &
                         + dot_product(vb_vvov(a,c,i,:,r), t2b(b,:,j,k,r)) &
                         - dot_product(vb_vooo(a,:,i,k,r), t2b(b,c,j,:,r)) &
                         + 0.5d0 * dot_product(va_vvov(a,b,i,:,r), t2b(:,c,j,k,r)) &
                         - 0.5d0 * dot_product(va_vooo(a,:,i,j,r), t2b(b,c,:,k,r))
                  end if
            end function core_aab

            ! cores of ccsdt.update_t3c (a alpha; b, c beta)
            function core_abb(e, r, hbar) result(val)
                  integer, intent(in) :: e(6), r
                  logical, intent(in) :: hbar
                  real(kind=8) :: val
                  integer :: a, b, c, i, j, k
                  a = e(1); b = e(2); c = e(3); i = e(4); j = e(5); k = e(6)
                  if (hbar) then
                     val = 0.5d0 * dot_product(hb_vvov(a,b,i,:), t2c(:,c,j,k,p)) &
                         - 0.5d0 * dot_product(hb_vooo(a,:,i,j), t2c(b,c,:,k,p)) &
                         + 0.5d0 * dot_product(hc_vvov(c,b,k,:), t2b(a,:,i,j,p)) &
                         - 0.5d0 * dot_product(hc_vooo(c,:,k,j), t2b(a,b,i,:,p)) &
                         + dot_product(hb_vvvo(a,b,:,j), t2b(:,c,i,k,p)) &
                         - dot_product(hb_ovoo(:,b,i,j), t2b(a,c,:,k,p))
                  else
                     val = 0.5d0 * dot_product(vb_vvov(a,b,i,:,r), t2c(:,c,j,k,r)) &
                         - 0.5d0 * dot_product(vb_vooo(a,:,i,j,r), t2c(b,c,:,k,r)) &
                         + 0.5d0 * dot_product(vc_vvov(c,b,k,:,r), t2b(a,:,i,j,r)) &
                         - 0.5d0 * dot_product(vc_vooo(c,:,k,j,r), t2b(a,b,i,:,r)) &
                         + dot_product(vb_vvvo(a,b,:,j,r), t2b(:,c,i,k,r)) &
                         - dot_product(vb_ovoo(:,b,i,j,r), t2b(a,c,:,k,r))
                  end if
            end function core_abb

            ! (V_N T1)_DC of reference r: A[core], cores 1/4 t1 v (same spin) etc.
            function antisym_y(cs, idx, r) result(val)
                  integer, intent(in) :: cs, idx(6), r
                  real(kind=8) :: val
                  integer :: s1, s2, e(6)
                  val = 0.0d0
                  select case (cs)
                  case (1, 4)
                     do s1 = 1, 6
                        do s2 = 1, 6
                           e(1:3) = idx(perm3(:,s1)); e(4:6) = idx(3 + perm3(:,s2))
                           if (cs == 1) then
                              val = val + sign3(s1) * sign3(s2) * 0.25d0 * t1a(e(3),e(6),r) * va_vvoo(e(1),e(2),e(4),e(5),r)
                           else
                              val = val + sign3(s1) * sign3(s2) * 0.25d0 * t1b(e(3),e(6),r) * vc_vvoo(e(1),e(2),e(4),e(5),r)
                           end if
                        end do
                     end do
                  case (2)
                     do s1 = 1, 2
                        do s2 = 1, 2
                           e = idx
                           if (s1 == 2) e(1:2) = idx((/2,1/))
                           if (s2 == 2) e(4:5) = idx((/5,4/))
                           val = val + (3 - 2*s1) * (3 - 2*s2) * (0.25d0 * t1b(e(3),e(6),r) * va_vvoo(e(1),e(2),e(4),e(5),r)&
                                 + t1a(e(1),e(4),r) * vb_vvoo(e(2),e(3),e(5),e(6),r))
                        end do
                     end do
                  case (3)
                     do s1 = 1, 2
                        do s2 = 1, 2
                           e = idx
                           if (s1 == 2) e(2:3) = idx((/3,2/))
                           if (s2 == 2) e(5:6) = idx((/6,5/))
                           val = val + (3 - 2*s1) * (3 - 2*s2) * (0.25d0 * t1a(e(1),e(4),r) * vc_vvoo(e(2),e(3),e(5),e(6),r)&
                                 + t1b(e(2),e(5),r) * vb_vvoo(e(1),e(3),e(4),e(6),r))
                        end do
                     end do
                  end select
            end function antisym_y

            ! [e^{-T(p)} Ct]_K: sum over sub-excitations L of K, Ct_K = ct3
            function coupling_element(xk, yk, ct3) result(val)
                  integer, intent(in) :: xk(3), yk(3)
                  real(kind=8), intent(in) :: ct3
                  real(kind=8) :: val, cc, cl
                  integer :: mh, mp, rr, m, nh, np_, hl(3), pl(3), hc(3), pc(3), par
                  integer :: dum_h(1,1), dum_p(1,1), dum_k(1)
                  real(kind=8) :: dum_amp(1)
                  dum_h = 0; dum_p = 0; dum_k = 0; dum_amp = 0.0d0
                  val = ct3
                  do mh = 1, 7
                     do mp = 1, 7
                        if (popcnt(mh) /= popcnt(mp)) cycle
                        rr = popcnt(mh)
                        if (rr == 3) cycle
                        nh = 0; np_ = 0; par = 0; hc = 0; pc = 0
                        do m = 1, 3
                           if (.not. btest(mh, m - 1)) then
                              nh = nh + 1; hc(nh) = yk(m); par = par + m - nh
                           end if
                           if (.not. btest(mp, m - 1)) then
                              np_ = np_ + 1; pc(np_) = xk(m); par = par + m - np_
                           end if
                        end do
                        call cluster_coefficient(cc, hc(1:3-rr), pc(1:3-rr), 3 - rr, mt1a, mt1b, mt2a, mt2b, mt2c,&
                                                 dum_h, dum_p, dum_k, dum_amp, 1, 1, lo(:,p), lu(:,p), dummy, 0,&
                                                 noa, nua, nob, nub, norb)
                        if (cc == 0.0d0) cycle
                        nh = 0; np_ = 0
                        do m = 1, 3
                           if (btest(mh, m - 1)) then
                              nh = nh + 1; hl(nh) = yk(m)
                           end if
                           if (btest(mp, m - 1)) then
                              np_ = np_ + 1; pl(np_) = xk(m)
                           end if
                        end do
                        cl = lookup12(hl, pl, rr)
                        val = val + (1 - 2*mod(par, 2)) * cc * cl
                     end do
                  end do
            end function coupling_element

            function lookup12(hl, pl, rr) result(cl)
                  integer, intent(in) :: hl(3), pl(3), rr
                  real(kind=8) :: cl
                  integer :: s(3), sp(3), m
                  cl = 0.0d0
                  do m = 1, rr
                     s(m) = hl(m) / norb
                     sp(m) = pl(m) / norb
                  end do
                  if (count(s(1:rr) == 0) /= count(sp(1:rr) == 0)) return
                  if (rr == 1) then
                     if (s(1) == 0) then
                        cl = c1a(lu(pl(1),p), lo(hl(1),p))
                     else
                        cl = c1b(lu(pl(1),p), lo(hl(1),p))
                     end if
                  else
                     if (s(1) == 0 .and. s(2) == 0) then
                        cl = c2a(lu(pl(1),p), lu(pl(2),p), lo(hl(1),p), lo(hl(2),p))
                     else if (s(1) == 1) then
                        cl = c2c(lu(pl(1),p), lu(pl(2),p), lo(hl(1),p), lo(hl(2),p))
                     else
                        cl = c2b(lu(pl(1),p), lu(pl(2),p), lo(hl(1),p), lo(hl(2),p))
                     end if
                  end if
            end function lookup12

      end subroutine mm23_reference

end module sumrcc_loops
