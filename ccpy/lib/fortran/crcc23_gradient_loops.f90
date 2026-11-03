module crcc23_gradient_loops

  ! Triples loops of the CR-CC(2,3)_A and CR-EOMCC(2,3)_A analytic gradients
  ! (ccpy/gradients/crcc23_gradient.py, ccpy/gradients/creomcc23_gradient.py).
  !
  ! For every unique occupied triple, the moments M(abc) = <ijkabc|Hbar|R>, the left
  ! projections N(abc) = <L|Hbar|ijkabc> and the bare-Fock denominators D(abc) are built over
  ! all virtual triples, as in the energy loops of crcc_loops, and immediately contracted:
  !
  !   u = M / D,  v = N / D,  w = u v,
  !   delta += sum v M,  W += sum w,  d_p -= sum w chi_p        (sums over unique triples)
  !
  ! and every two-body (non-triples) array the moments and projections were built from receives
  ! its cotangent (reverse-mode derivative) with respect to these sums:
  !
  !   moments   M = A(abc) A(ijk) sum_g X_g . t_g:  dX_g += v . t_g,  dt_g += v . X_g
  !   left      N = A(abc) A(ijk) sum_n H_n . l_n:  dH_n += u . l_n,  dl_n += u . H_n
  !
  ! with the intermediates X_g of ccpy/gradients/crcc23_gradient.py (built from T2 and the
  ! CCS-transformed Hamiltonian). For an excited state (excited = 1) the moments are those of
  ! R = r0 + R1 + R2, M = A(abc) A(ijk) sum_g (J_g . t_g + X_g . r_g), and the routines also
  ! return dJ_g -> t (v . J_g), dX_g from r (v . r_g), F_r0 = sum v M(ground), and
  ! c'_X = <V3|R|X> for X in singles and doubles.
  !
  ! Unique-triple sums are taken over one occupied triple at a time: with an antisymmetric
  ! cotangent G, sum_{abc unique} G M = sum_perm sign(perm) sum_{abc} G X(perm(ijk)), so every
  ! occupied permutation contributes one DGEMM with the sign of the permutation, and no
  ! antisymmetrizer is applied in the reverse direction. Arrays are passed pre-arranged by the
  ! Python modules (layouts in the argument comments) so that each contraction is one DGEMM.
  ! Cotangent arrays are accumulated (intent(inout)); they must be zero on entry.

  use constants, only: p

  implicit none

  contains

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! aaa (and bbb) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine grad_aaa(delta, wsum, fr0, &
                      x1, x2r, t2, j1, j2r, r1, r2, &
                      hvovv_r, hooov_r, hoovv, hov, l1, l2, &
                      eo, ev, omega, excited, &
                      d_o, d_v, dx1, dx2r, dt2, dxr1, dxr2r, dtj2, cp1, cp2, &
                      dl1, dl2, dhvovv_r, dhooov_r, dhoovv, dhov, no, nu)

    integer, intent(in) :: no, nu, excited
    real(p), intent(in) :: omega
    real(p), intent(out) :: delta, wsum, fr0
    ! moments: x1(a,m,i,j) . t2(b,c,m,k);  x2r(a,b,e,i) . t2(e,c,j,k)
    real(p), intent(in) :: x1(nu,no,no,no), x2r(nu,nu,nu,no), t2(nu,nu,no,no)
    real(p), intent(in) :: j1(nu,no,no,no), j2r(nu,nu,nu,no), r1(nu,no), r2(nu,nu,no,no)
    ! left: hvovv_r(a,b,e,i) = <ei||ba>, hooov_r(a,m,j,i) = <ji||ma>, hoovv(i,j,a,b), hov(k,c)
    real(p), intent(in) :: hvovv_r(nu,nu,nu,no), hooov_r(nu,no,no,no), hoovv(no,no,nu,nu), hov(no,nu)
    real(p), intent(in) :: l1(nu,no), l2(nu,nu,no,no)
    real(p), intent(in) :: eo(no), ev(nu)
    real(p), intent(inout) :: d_o(no), d_v(nu)
    real(p), intent(inout) :: dx1(nu,no,no,no), dx2r(nu,nu,nu,no), dt2(nu,nu,no,no)
    real(p), intent(inout) :: dxr1(nu,no,no,no), dxr2r(nu,nu,nu,no), dtj2(nu,nu,no,no)
    real(p), intent(inout) :: cp1(nu,no), cp2(nu,nu,no,no)
    real(p), intent(inout) :: dl1(nu,no), dl2(nu,nu,no,no)
    real(p), intent(inout) :: dhvovv_r(nu,nu,nu,no), dhooov_r(nu,no,no,no), dhoovv(no,no,nu,nu), dhov(no,nu)

    integer :: i, j, k, a, b, c, n, ii, jj, kk, nu2, occ(3)
    integer, parameter :: perm(3,6) = reshape([1,2,3, 2,3,1, 3,1,2, 2,1,3, 1,3,2, 3,2,1], [3,6])
    real(p), parameter :: sgn(6) = [1.0_p, 1.0_p, 1.0_p, -1.0_p, -1.0_p, -1.0_p]
    real(p) :: s, den, ssum
    real(p) :: xm(nu,nu,nu), xe(nu,nu,nu), xn(nu,nu,nu)
    real(p) :: m3(nu,nu,nu), e3(nu,nu,nu), n3(nu,nu,nu), u3(nu,nu,nu), v3(nu,nu,nu)

    nu2 = nu*nu
    delta = 0.0_p
    wsum = 0.0_p
    fr0 = 0.0_p

    do i = 1, no
      do j = i+1, no
        do k = j+1, no
          occ = [i, j, k]

          ! ---- forward: unsymmetrized batches, all occupied permutations
          xm = 0.0_p
          xe = 0.0_p
          xn = 0.0_p
          do n = 1, 6
            ii = occ(perm(1,n)); jj = occ(perm(2,n)); kk = occ(perm(3,n)); s = sgn(n)
            call dgemm('n','t',nu,nu2,no,s,x1(:,:,ii,jj),nu,t2(:,:,:,kk),nu2,1.0_p,xm,nu)
            call dgemm('n','n',nu2,nu,nu,s,x2r(:,:,:,ii),nu2,t2(:,:,jj,kk),nu,1.0_p,xm,nu2)
            if (excited == 1) then
              call dgemm('n','t',nu,nu2,no,s,j1(:,:,ii,jj),nu,t2(:,:,:,kk),nu2,1.0_p,xe,nu)
              call dgemm('n','t',nu,nu2,no,s,x1(:,:,ii,jj),nu,r2(:,:,:,kk),nu2,1.0_p,xe,nu)
              call dgemm('n','n',nu2,nu,nu,s,j2r(:,:,:,ii),nu2,t2(:,:,jj,kk),nu,1.0_p,xe,nu2)
              call dgemm('n','n',nu2,nu,nu,s,x2r(:,:,:,ii),nu2,r2(:,:,jj,kk),nu,1.0_p,xe,nu2)
            end if
            call dgemm('n','n',nu2,nu,nu,0.25_p*s,hvovv_r(:,:,:,ii),nu2,l2(:,:,jj,kk),nu,1.0_p,xn,nu2)
            call dgemm('n','t',nu,nu2,no,-0.25_p*s,hooov_r(:,:,jj,ii),nu,l2(:,:,:,kk),nu2,1.0_p,xn,nu)
            do c = 1, nu
              do b = 1, nu
                do a = 1, nu
                  xn(a,b,c) = xn(a,b,c) + 0.25_p*s*(l1(c,kk)*hoovv(ii,jj,a,b) + hov(kk,c)*l2(a,b,ii,jj))
                end do
              end do
            end do
          end do
          call antisym_abc(xm, m3, nu)
          call antisym_abc(xn, n3, nu)
          if (excited == 1) then
            call antisym_abc(xe, e3, nu)
          else
            e3 = m3
          end if

          ! ---- divided moments and the scalar/vector sums (1/6: unique virtual triples)
          do c = 1, nu
            do b = 1, nu
              do a = 1, nu
                den = omega + eo(i) + eo(j) + eo(k) - ev(a) - ev(b) - ev(c)
                u3(a,b,c) = e3(a,b,c)/den
                v3(a,b,c) = n3(a,b,c)/den
              end do
            end do
          end do
          delta = delta + sum(v3*e3)/6.0_p
          fr0 = fr0 + sum(v3*m3)/6.0_p
          ssum = sum(u3*v3)/6.0_p
          wsum = wsum + ssum
          d_o(i) = d_o(i) - ssum
          d_o(j) = d_o(j) - ssum
          d_o(k) = d_o(k) - ssum
          do a = 1, nu
            d_v(a) = d_v(a) + 0.5_p*sum(u3(a,:,:)*v3(a,:,:))
          end do

          ! ---- reverse: cotangents, all occupied permutations
          do n = 1, 6
            ii = occ(perm(1,n)); jj = occ(perm(2,n)); kk = occ(perm(3,n)); s = sgn(n)
            ! v . M (ground moments)
            call dgemm('n','n',nu,no,nu2,s,v3,nu,t2(:,:,:,kk),nu2,1.0_p,dx1(:,:,ii,jj),nu)
            call dgemm('t','n',nu2,no,nu,s,v3,nu,x1(:,:,ii,jj),nu,1.0_p,dt2(:,:,:,kk),nu2)
            call dgemm('n','t',nu2,nu,nu,s,v3,nu2,t2(:,:,jj,kk),nu,1.0_p,dx2r(:,:,:,ii),nu2)
            call dgemm('t','n',nu,nu,nu2,s,x2r(:,:,:,ii),nu2,v3,nu2,1.0_p,dt2(:,:,jj,kk),nu)
            if (excited == 1) then
              call dgemm('n','n',nu,no,nu2,s,v3,nu,r2(:,:,:,kk),nu2,1.0_p,dxr1(:,:,ii,jj),nu)
              call dgemm('t','n',nu2,no,nu,s,v3,nu,j1(:,:,ii,jj),nu,1.0_p,dtj2(:,:,:,kk),nu2)
              call dgemm('n','t',nu2,nu,nu,s,v3,nu2,r2(:,:,jj,kk),nu,1.0_p,dxr2r(:,:,:,ii),nu2)
              call dgemm('t','n',nu,nu,nu2,s,j2r(:,:,:,ii),nu2,v3,nu2,1.0_p,dtj2(:,:,jj,kk),nu)
              ! c'(a,i) += 1/4 v(abc) r2(bc,jk);  c'(ab,ij) += v(abc) r1(c,k)
              call dgemv('n',nu,nu2,0.25_p*s,v3,nu,r2(:,:,jj,kk),1,1.0_p,cp1(:,ii),1)
              call dgemv('n',nu2,nu,s,v3,nu2,r1(:,kk),1,1.0_p,cp2(:,:,ii,jj),1)
            end if
            ! u . N (left projections)
            call dgemm('t','n',nu,nu,nu2,0.25_p*s,hvovv_r(:,:,:,ii),nu2,u3,nu2,1.0_p,dl2(:,:,jj,kk),nu)
            call dgemm('n','t',nu2,nu,nu,0.25_p*s,u3,nu2,l2(:,:,jj,kk),nu,1.0_p,dhvovv_r(:,:,:,ii),nu2)
            call dgemm('t','n',nu2,no,nu,-0.25_p*s,u3,nu,hooov_r(:,:,jj,ii),nu,1.0_p,dl2(:,:,:,kk),nu2)
            call dgemm('n','n',nu,no,nu2,-0.25_p*s,u3,nu,l2(:,:,:,kk),nu2,1.0_p,dhooov_r(:,:,jj,ii),nu)
            do c = 1, nu
              do b = 1, nu
                do a = 1, nu
                  dl1(c,kk) = dl1(c,kk) + 0.25_p*s*u3(a,b,c)*hoovv(ii,jj,a,b)
                  dhoovv(ii,jj,a,b) = dhoovv(ii,jj,a,b) + 0.25_p*s*u3(a,b,c)*l1(c,kk)
                  dhov(kk,c) = dhov(kk,c) + 0.25_p*s*u3(a,b,c)*l2(a,b,ii,jj)
                  dl2(a,b,ii,jj) = dl2(a,b,ii,jj) + 0.25_p*s*u3(a,b,c)*hov(kk,c)
                end do
              end do
            end do
          end do

        end do
      end do
    end do

  end subroutine grad_aaa

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! aab !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine grad_aab(delta, wsum, fr0, &
                      x1, x2, x3r, x4, x5r, x6, t2a, t2a_r, t2b, t2b_r, &
                      j1, j2, j3r, j4, j5r, j6, r1a, r1b, r2a, r2a_r, r2b, r2b_r, &
                      h1r, h2r, h3r, h4r, h5r, h6r, hoovv_ab, hoovv_aa, hov_a, hov_b, &
                      l1a, l1b, l2a, l2a_r, l2b, l2b_r, &
                      eoa, eob, eva, evb, omega, excited, &
                      d_oa, d_ob, d_va, d_vb, &
                      dx1, dx2, dx3r, dx4, dx5r, dx6, dt2a, dt2a_r, dt2b, dt2b_r, &
                      dxr1, dxr2, dxr3r, dxr4, dxr5r, dxr6, dtj2a, dtj2a_r, dtj2b, dtj2b_r, &
                      cp1a, cp1b, cp2a, cp2b, &
                      dh1r, dh2r, dh3r, dh4r, dh5r, dh6r, dhoovv_ab, dhoovv_aa, dhov_a, dhov_b, &
                      dl1a, dl1b, dl2a, dl2a_r, dl2b, dl2b_r, &
                      noa, nob, nua, nub)

    ! Batches over (a,b,c), a,b alpha and c beta; occupied (i,j,k), i,j alpha and k beta.
    ! t2b, l2b, r2b: alpha-beta doubles (a~ab block) [a,b,i,j]; *_r: last two indices swapped.
    integer, intent(in) :: noa, nob, nua, nub, excited
    real(p), intent(in) :: omega
    real(p), intent(out) :: delta, wsum, fr0
    ! moment groups (layouts) . trailing amplitudes:
    !   x1(b,c,e,k) . t2a(a,e,i,j)      x2(m,c,j,k) . t2a_r(a,b,m,i)    x3r(a,c,e,i) . t2b(b,e,j,k)
    !   x4(a,m,i,k) . t2b_r(b,c,m,j)    x5r(a,b,e,i) . t2b(e,c,j,k)     x6(a,m,i,j) . t2b(b,c,m,k)
    real(p), intent(in) :: x1(nua,nub,nua,nob), x2(noa,nub,noa,nob), x3r(nua,nub,nub,noa)
    real(p), intent(in) :: x4(nua,nob,noa,nob), x5r(nua,nua,nua,noa), x6(nua,noa,noa,noa)
    real(p), intent(in) :: t2a(nua,nua,noa,noa), t2a_r(nua,nua,noa,noa), t2b(nua,nub,noa,nob), t2b_r(nua,nub,nob,noa)
    real(p), intent(in) :: j1(nua,nub,nua,nob), j2(noa,nub,noa,nob), j3r(nua,nub,nub,noa)
    real(p), intent(in) :: j4(nua,nob,noa,nob), j5r(nua,nua,nua,noa), j6(nua,noa,noa,noa)
    real(p), intent(in) :: r1a(nua,noa), r1b(nub,nob)
    real(p), intent(in) :: r2a(nua,nua,noa,noa), r2a_r(nua,nua,noa,noa), r2b(nua,nub,noa,nob), r2b_r(nua,nub,nob,noa)
    ! left terms (layouts):
    !   h1r(e,b,c,k) = <ek|bc>_ab . l2a(a,e,i,j)        h2r(a,b,e,i) = <ei||ba> . l2b(e,c,j,k)
    !   h3r(a,c,e,i) = <ie|ac>_ab . l2b(b,e,j,k)        h4r(m,c,j,k) = <jk|mc>_ab . l2a_r(a,b,m,i)
    !   h5r(a,m,j,i) = <ji||ma> . l2b(b,c,m,k)          h6r(a,m,i,k) = <ik|am>_ab . l2b_r(b,c,m,j)
    !   l1a(a,i) <jk|bc>_ab,  l1b(c,k) <ij||ab>,  l2b(b,c,j,k) f(i,a),  l2a(a,b,i,j) f(k,c)
    real(p), intent(in) :: h1r(nua,nua,nub,nob), h2r(nua,nua,nua,noa), h3r(nua,nub,nub,noa)
    real(p), intent(in) :: h4r(noa,nub,noa,nob), h5r(nua,noa,noa,noa), h6r(nua,nob,noa,nob)
    real(p), intent(in) :: hoovv_ab(noa,nob,nua,nub), hoovv_aa(noa,noa,nua,nua), hov_a(noa,nua), hov_b(nob,nub)
    real(p), intent(in) :: l1a(nua,noa), l1b(nub,nob), l2a(nua,nua,noa,noa), l2a_r(nua,nua,noa,noa)
    real(p), intent(in) :: l2b(nua,nub,noa,nob), l2b_r(nua,nub,nob,noa)
    real(p), intent(in) :: eoa(noa), eob(nob), eva(nua), evb(nub)
    real(p), intent(inout) :: d_oa(noa), d_ob(nob), d_va(nua), d_vb(nub)
    real(p), intent(inout) :: dx1(nua,nub,nua,nob), dx2(noa,nub,noa,nob), dx3r(nua,nub,nub,noa)
    real(p), intent(inout) :: dx4(nua,nob,noa,nob), dx5r(nua,nua,nua,noa), dx6(nua,noa,noa,noa)
    real(p), intent(inout) :: dt2a(nua,nua,noa,noa), dt2a_r(nua,nua,noa,noa), dt2b(nua,nub,noa,nob), dt2b_r(nua,nub,nob,noa)
    real(p), intent(inout) :: dxr1(nua,nub,nua,nob), dxr2(noa,nub,noa,nob), dxr3r(nua,nub,nub,noa)
    real(p), intent(inout) :: dxr4(nua,nob,noa,nob), dxr5r(nua,nua,nua,noa), dxr6(nua,noa,noa,noa)
    real(p), intent(inout) :: dtj2a(nua,nua,noa,noa), dtj2a_r(nua,nua,noa,noa), dtj2b(nua,nub,noa,nob), dtj2b_r(nua,nub,nob,noa)
    real(p), intent(inout) :: cp1a(nua,noa), cp1b(nub,nob), cp2a(nua,nua,noa,noa), cp2b(nua,nub,noa,nob)
    real(p), intent(inout) :: dh1r(nua,nua,nub,nob), dh2r(nua,nua,nua,noa), dh3r(nua,nub,nub,noa)
    real(p), intent(inout) :: dh4r(noa,nub,noa,nob), dh5r(nua,noa,noa,noa), dh6r(nua,nob,noa,nob)
    real(p), intent(inout) :: dhoovv_ab(noa,nob,nua,nub), dhoovv_aa(noa,noa,nua,nua), dhov_a(noa,nua), dhov_b(nob,nub)
    real(p), intent(inout) :: dl1a(nua,noa), dl1b(nub,nob), dl2a(nua,nua,noa,noa), dl2a_r(nua,nua,noa,noa)
    real(p), intent(inout) :: dl2b(nua,nub,noa,nob), dl2b_r(nua,nub,nob,noa)

    integer :: i, j, k, a, b, c, n, ii, jj, kk, nua2, nuab, occ(3)
    real(p) :: s, den, ssum
    real(p) :: xm(nua,nua,nub), xe(nua,nua,nub), xn(nua,nua,nub)
    real(p) :: bm_bca(nua,nub,nua), bm_acb(nua,nub,nua), be_bca(nua,nub,nua), be_acb(nua,nub,nua), bn_acb(nua,nub,nua)
    real(p) :: m3(nua,nua,nub), e3(nua,nua,nub), n3(nua,nua,nub), u3(nua,nua,nub), v3(nua,nua,nub)
    real(p) :: v_bca(nua,nub,nua), v_acb(nua,nub,nua), u_acb(nua,nub,nua)

    nua2 = nua*nua
    nuab = nua*nub
    delta = 0.0_p
    wsum = 0.0_p
    fr0 = 0.0_p

    do i = 1, noa
      do j = i+1, noa
        do k = 1, nob
          occ = [i, j, k]

          ! ---- forward
          xm = 0.0_p; xe = 0.0_p; xn = 0.0_p
          bm_bca = 0.0_p; bm_acb = 0.0_p; be_bca = 0.0_p; be_acb = 0.0_p; bn_acb = 0.0_p
          do n = 1, 2
            if (n == 1) then
              ii = i; jj = j; s = 1.0_p
            else
              ii = j; jj = i; s = -1.0_p
            end if
            kk = k
            ! moments
            call dgemm('n','t',nuab,nua,nua,s,x1(:,:,:,kk),nuab,t2a(:,:,ii,jj),nua,1.0_p,bm_bca,nuab)
            call dgemm('n','n',nua2,nub,noa,s,t2a_r(:,:,:,ii),nua2,x2(:,:,jj,kk),noa,1.0_p,xm,nua2)
            call dgemm('n','t',nuab,nua,nub,s,x3r(:,:,:,ii),nuab,t2b(:,:,jj,kk),nua,1.0_p,bm_acb,nuab)
            call dgemm('n','t',nua,nuab,nob,s,x4(:,:,ii,kk),nua,t2b_r(:,:,:,jj),nuab,1.0_p,xm,nua)
            call dgemm('n','n',nua2,nub,nua,s,x5r(:,:,:,ii),nua2,t2b(:,:,jj,kk),nua,1.0_p,xm,nua2)
            call dgemm('n','t',nua,nuab,noa,s,x6(:,:,ii,jj),nua,t2b(:,:,:,kk),nuab,1.0_p,xm,nua)
            if (excited == 1) then
              call dgemm('n','t',nuab,nua,nua,s,j1(:,:,:,kk),nuab,t2a(:,:,ii,jj),nua,1.0_p,be_bca,nuab)
              call dgemm('n','t',nuab,nua,nua,s,x1(:,:,:,kk),nuab,r2a(:,:,ii,jj),nua,1.0_p,be_bca,nuab)
              call dgemm('n','n',nua2,nub,noa,s,t2a_r(:,:,:,ii),nua2,j2(:,:,jj,kk),noa,1.0_p,xe,nua2)
              call dgemm('n','n',nua2,nub,noa,s,r2a_r(:,:,:,ii),nua2,x2(:,:,jj,kk),noa,1.0_p,xe,nua2)
              call dgemm('n','t',nuab,nua,nub,s,j3r(:,:,:,ii),nuab,t2b(:,:,jj,kk),nua,1.0_p,be_acb,nuab)
              call dgemm('n','t',nuab,nua,nub,s,x3r(:,:,:,ii),nuab,r2b(:,:,jj,kk),nua,1.0_p,be_acb,nuab)
              call dgemm('n','t',nua,nuab,nob,s,j4(:,:,ii,kk),nua,t2b_r(:,:,:,jj),nuab,1.0_p,xe,nua)
              call dgemm('n','t',nua,nuab,nob,s,x4(:,:,ii,kk),nua,r2b_r(:,:,:,jj),nuab,1.0_p,xe,nua)
              call dgemm('n','n',nua2,nub,nua,s,j5r(:,:,:,ii),nua2,t2b(:,:,jj,kk),nua,1.0_p,xe,nua2)
              call dgemm('n','n',nua2,nub,nua,s,x5r(:,:,:,ii),nua2,r2b(:,:,jj,kk),nua,1.0_p,xe,nua2)
              call dgemm('n','t',nua,nuab,noa,s,j6(:,:,ii,jj),nua,t2b(:,:,:,kk),nuab,1.0_p,xe,nua)
              call dgemm('n','t',nua,nuab,noa,s,x6(:,:,ii,jj),nua,r2b(:,:,:,kk),nuab,1.0_p,xe,nua)
            end if
            ! left projections
            call dgemm('n','n',nua,nuab,nua,0.5_p*s,l2a(:,:,ii,jj),nua,h1r(:,:,:,kk),nua,1.0_p,xn,nua)
            call dgemm('n','n',nua2,nub,nua,0.5_p*s,h2r(:,:,:,ii),nua2,l2b(:,:,jj,kk),nua,1.0_p,xn,nua2)
            call dgemm('n','t',nuab,nua,nub,s,h3r(:,:,:,ii),nuab,l2b(:,:,jj,kk),nua,1.0_p,bn_acb,nuab)
            call dgemm('n','n',nua2,nub,noa,-0.5_p*s,l2a_r(:,:,:,ii),nua2,h4r(:,:,jj,kk),noa,1.0_p,xn,nua2)
            call dgemm('n','t',nua,nuab,noa,-0.5_p*s,h5r(:,:,jj,ii),nua,l2b(:,:,:,kk),nuab,1.0_p,xn,nua)
            call dgemm('n','t',nua,nuab,nob,-s,h6r(:,:,ii,kk),nua,l2b_r(:,:,:,jj),nuab,1.0_p,xn,nua)
            do c = 1, nub
              do b = 1, nua
                do a = 1, nua
                  xn(a,b,c) = xn(a,b,c) + s*(l1a(a,ii)*hoovv_ab(jj,kk,b,c) + 0.25_p*l1b(c,kk)*hoovv_aa(ii,jj,a,b) &
                              + l2b(b,c,jj,kk)*hov_a(ii,a) + 0.25_p*l2a(a,b,ii,jj)*hov_b(kk,c))
                end do
              end do
            end do
          end do
          do c = 1, nub
            do b = 1, nua
              do a = 1, nua
                xm(a,b,c) = xm(a,b,c) + bm_bca(b,c,a) + bm_acb(a,c,b)
                xe(a,b,c) = xe(a,b,c) + be_bca(b,c,a) + be_acb(a,c,b)
                xn(a,b,c) = xn(a,b,c) + bn_acb(a,c,b)
              end do
            end do
          end do
          call antisym_ab(xm, m3, nua, nub)
          call antisym_ab(xn, n3, nua, nub)
          if (excited == 1) then
            call antisym_ab(xe, e3, nua, nub)
          else
            e3 = m3
          end if

          ! ---- divided moments and sums (1/2: unique virtual triples)
          do c = 1, nub
            do b = 1, nua
              do a = 1, nua
                den = omega + eoa(i) + eoa(j) + eob(k) - eva(a) - eva(b) - evb(c)
                u3(a,b,c) = e3(a,b,c)/den
                v3(a,b,c) = n3(a,b,c)/den
              end do
            end do
          end do
          delta = delta + sum(v3*e3)/2.0_p
          fr0 = fr0 + sum(v3*m3)/2.0_p
          ssum = sum(u3*v3)/2.0_p
          wsum = wsum + ssum
          d_oa(i) = d_oa(i) - ssum
          d_oa(j) = d_oa(j) - ssum
          d_ob(k) = d_ob(k) - ssum
          do a = 1, nua
            d_va(a) = d_va(a) + sum(u3(a,:,:)*v3(a,:,:))
          end do
          do c = 1, nub
            d_vb(c) = d_vb(c) + 0.5_p*sum(u3(:,:,c)*v3(:,:,c))
          end do
          do c = 1, nub
            do b = 1, nua
              do a = 1, nua
                v_bca(b,c,a) = v3(a,b,c)
                v_acb(a,c,b) = v3(a,b,c)
                u_acb(a,c,b) = u3(a,b,c)
              end do
            end do
          end do

          ! ---- reverse
          do n = 1, 2
            if (n == 1) then
              ii = i; jj = j; s = 1.0_p
            else
              ii = j; jj = i; s = -1.0_p
            end if
            kk = k
            ! v . M
            call dgemm('n','n',nuab,nua,nua,s,v_bca,nuab,t2a(:,:,ii,jj),nua,1.0_p,dx1(:,:,:,kk),nuab)
            call dgemm('t','n',nua,nua,nuab,s,v_bca,nuab,x1(:,:,:,kk),nuab,1.0_p,dt2a(:,:,ii,jj),nua)
            call dgemm('n','t',nua2,noa,nub,s,v3,nua2,x2(:,:,jj,kk),noa,1.0_p,dt2a_r(:,:,:,ii),nua2)
            call dgemm('t','n',noa,nub,nua2,s,t2a_r(:,:,:,ii),nua2,v3,nua2,1.0_p,dx2(:,:,jj,kk),noa)
            call dgemm('n','n',nuab,nub,nua,s,v_acb,nuab,t2b(:,:,jj,kk),nua,1.0_p,dx3r(:,:,:,ii),nuab)
            call dgemm('t','n',nua,nub,nuab,s,v_acb,nuab,x3r(:,:,:,ii),nuab,1.0_p,dt2b(:,:,jj,kk),nua)
            call dgemm('n','n',nua,nob,nuab,s,v3,nua,t2b_r(:,:,:,jj),nuab,1.0_p,dx4(:,:,ii,kk),nua)
            call dgemm('t','n',nuab,nob,nua,s,v3,nua,x4(:,:,ii,kk),nua,1.0_p,dt2b_r(:,:,:,jj),nuab)
            call dgemm('n','t',nua2,nua,nub,s,v3,nua2,t2b(:,:,jj,kk),nua,1.0_p,dx5r(:,:,:,ii),nua2)
            call dgemm('t','n',nua,nub,nua2,s,x5r(:,:,:,ii),nua2,v3,nua2,1.0_p,dt2b(:,:,jj,kk),nua)
            call dgemm('n','n',nua,noa,nuab,s,v3,nua,t2b(:,:,:,kk),nuab,1.0_p,dx6(:,:,ii,jj),nua)
            call dgemm('t','n',nuab,noa,nua,s,v3,nua,x6(:,:,ii,jj),nua,1.0_p,dt2b(:,:,:,kk),nuab)
            if (excited == 1) then
              call dgemm('n','n',nuab,nua,nua,s,v_bca,nuab,r2a(:,:,ii,jj),nua,1.0_p,dxr1(:,:,:,kk),nuab)
              call dgemm('t','n',nua,nua,nuab,s,v_bca,nuab,j1(:,:,:,kk),nuab,1.0_p,dtj2a(:,:,ii,jj),nua)
              call dgemm('n','t',nua2,noa,nub,s,v3,nua2,j2(:,:,jj,kk),noa,1.0_p,dtj2a_r(:,:,:,ii),nua2)
              call dgemm('t','n',noa,nub,nua2,s,r2a_r(:,:,:,ii),nua2,v3,nua2,1.0_p,dxr2(:,:,jj,kk),noa)
              call dgemm('n','n',nuab,nub,nua,s,v_acb,nuab,r2b(:,:,jj,kk),nua,1.0_p,dxr3r(:,:,:,ii),nuab)
              call dgemm('t','n',nua,nub,nuab,s,v_acb,nuab,j3r(:,:,:,ii),nuab,1.0_p,dtj2b(:,:,jj,kk),nua)
              call dgemm('n','n',nua,nob,nuab,s,v3,nua,r2b_r(:,:,:,jj),nuab,1.0_p,dxr4(:,:,ii,kk),nua)
              call dgemm('t','n',nuab,nob,nua,s,v3,nua,j4(:,:,ii,kk),nua,1.0_p,dtj2b_r(:,:,:,jj),nuab)
              call dgemm('n','t',nua2,nua,nub,s,v3,nua2,r2b(:,:,jj,kk),nua,1.0_p,dxr5r(:,:,:,ii),nua2)
              call dgemm('t','n',nua,nub,nua2,s,j5r(:,:,:,ii),nua2,v3,nua2,1.0_p,dtj2b(:,:,jj,kk),nua)
              call dgemm('n','n',nua,noa,nuab,s,v3,nua,r2b(:,:,:,kk),nuab,1.0_p,dxr6(:,:,ii,jj),nua)
              call dgemm('t','n',nuab,noa,nua,s,v3,nua,j6(:,:,ii,jj),nua,1.0_p,dtj2b(:,:,:,kk),nuab)
              ! c' = <V3|R|X>
              call dgemv('n',nua,nuab,s,v3,nua,r2b(:,:,jj,kk),1,1.0_p,cp1a(:,ii),1)
              call dgemv('t',nua2,nub,0.25_p*s,v3,nua2,r2a(:,:,ii,jj),1,1.0_p,cp1b(:,kk),1)
              call dgemv('n',nua2,nub,s,v3,nua2,r1b(:,kk),1,1.0_p,cp2a(:,:,ii,jj),1)
              call dgemv('n',nuab,nua,s,v_acb,nuab,r1a(:,jj),1,1.0_p,cp2b(:,:,ii,kk),1)
            end if
            ! u . N
            call dgemm('n','t',nua,nua,nuab,0.5_p*s,u3,nua,h1r(:,:,:,kk),nua,1.0_p,dl2a(:,:,ii,jj),nua)
            call dgemm('t','n',nua,nuab,nua,0.5_p*s,l2a(:,:,ii,jj),nua,u3,nua,1.0_p,dh1r(:,:,:,kk),nua)
            call dgemm('n','t',nua2,nua,nub,0.5_p*s,u3,nua2,l2b(:,:,jj,kk),nua,1.0_p,dh2r(:,:,:,ii),nua2)
            call dgemm('t','n',nua,nub,nua2,0.5_p*s,h2r(:,:,:,ii),nua2,u3,nua2,1.0_p,dl2b(:,:,jj,kk),nua)
            call dgemm('n','n',nuab,nub,nua,s,u_acb,nuab,l2b(:,:,jj,kk),nua,1.0_p,dh3r(:,:,:,ii),nuab)
            call dgemm('t','n',nua,nub,nuab,s,u_acb,nuab,h3r(:,:,:,ii),nuab,1.0_p,dl2b(:,:,jj,kk),nua)
            call dgemm('n','t',nua2,noa,nub,-0.5_p*s,u3,nua2,h4r(:,:,jj,kk),noa,1.0_p,dl2a_r(:,:,:,ii),nua2)
            call dgemm('t','n',noa,nub,nua2,-0.5_p*s,l2a_r(:,:,:,ii),nua2,u3,nua2,1.0_p,dh4r(:,:,jj,kk),noa)
            call dgemm('n','n',nua,noa,nuab,-0.5_p*s,u3,nua,l2b(:,:,:,kk),nuab,1.0_p,dh5r(:,:,jj,ii),nua)
            call dgemm('t','n',nuab,noa,nua,-0.5_p*s,u3,nua,h5r(:,:,jj,ii),nua,1.0_p,dl2b(:,:,:,kk),nuab)
            call dgemm('n','n',nua,nob,nuab,-s,u3,nua,l2b_r(:,:,:,jj),nuab,1.0_p,dh6r(:,:,ii,kk),nua)
            call dgemm('t','n',nuab,nob,nua,-s,u3,nua,h6r(:,:,ii,kk),nua,1.0_p,dl2b_r(:,:,:,jj),nuab)
            do c = 1, nub
              do b = 1, nua
                do a = 1, nua
                  dl1a(a,ii) = dl1a(a,ii) + s*u3(a,b,c)*hoovv_ab(jj,kk,b,c)
                  dhoovv_ab(jj,kk,b,c) = dhoovv_ab(jj,kk,b,c) + s*u3(a,b,c)*l1a(a,ii)
                  dl1b(c,kk) = dl1b(c,kk) + 0.25_p*s*u3(a,b,c)*hoovv_aa(ii,jj,a,b)
                  dhoovv_aa(ii,jj,a,b) = dhoovv_aa(ii,jj,a,b) + 0.25_p*s*u3(a,b,c)*l1b(c,kk)
                  dl2b(b,c,jj,kk) = dl2b(b,c,jj,kk) + s*u3(a,b,c)*hov_a(ii,a)
                  dhov_a(ii,a) = dhov_a(ii,a) + s*u3(a,b,c)*l2b(b,c,jj,kk)
                  dl2a(a,b,ii,jj) = dl2a(a,b,ii,jj) + 0.25_p*s*u3(a,b,c)*hov_b(kk,c)
                  dhov_b(kk,c) = dhov_b(kk,c) + 0.25_p*s*u3(a,b,c)*l2a(a,b,ii,jj)
                end do
              end do
            end do
          end do

        end do
      end do
    end do

  end subroutine grad_aab

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! abb !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine grad_abb(delta, wsum, fr0, &
                      x1r, x2, x3, x4, x5r, x6, t2c, t2c_r, t2b, t2b_r, &
                      j1r, j2, j3, j4, j5r, j6, r1a, r1b, r2c, r2c_r, r2b, r2b_r, &
                      h1r, h2r, h3r, h4r, h5r, h6r, hoovv_ab, hoovv_bb, hov_a, hov_b, &
                      l1a, l1b, l2c, l2c_r, l2b, l2b_r, &
                      eoa, eob, eva, evb, omega, excited, &
                      d_oa, d_ob, d_va, d_vb, &
                      dx1r, dx2, dx3, dx4, dx5r, dx6, dt2c, dt2c_r, dt2b, dt2b_r, &
                      dxr1r, dxr2, dxr3, dxr4, dxr5r, dxr6, dtj2c, dtj2c_r, dtj2b, dtj2b_r, &
                      cp1a, cp1b, cp2b, cp2c, &
                      dh1r, dh2r, dh3r, dh4r, dh5r, dh6r, dhoovv_ab, dhoovv_bb, dhov_a, dhov_b, &
                      dl1a, dl1b, dl2c, dl2c_r, dl2b, dl2b_r, &
                      noa, nob, nua, nub)

    ! Batches over (c,a,b), c alpha and a,b beta; occupied (k,i,j), k alpha and i,j beta (the
    ! abb layout of ccpy). t2c, l2c, r2c: beta-beta doubles; t2b, l2b, r2b: alpha-beta doubles.
    ! t2c_r = t2c with the last two indices swapped; t2b_r(c,b,m,k) = t2b(c,b,k,m).
    integer, intent(in) :: noa, nob, nua, nub, excited
    real(p), intent(in) :: omega
    real(p), intent(out) :: delta, wsum, fr0
    ! moment groups:
    !   x1r(c,b,e,k) . t2c(a,e,i,j)     x2(c,m,k,j) . t2c_r(a,b,m,i)    x3(c,a,e,i) . t2b(e,b,k,j)
    !   x4(m,a,k,i) . t2b(c,b,m,j)      x5r(a,b,e,i) . t2b(c,e,k,j)     x6(a,m,i,j) . t2b_r(c,b,m,k)
    real(p), intent(in) :: x1r(nua,nub,nub,noa), x2(nua,nob,noa,nob), x3(nua,nub,nua,nob)
    real(p), intent(in) :: x4(noa,nub,noa,nob), x5r(nub,nub,nub,nob), x6(nub,nob,nob,nob)
    real(p), intent(in) :: t2c(nub,nub,nob,nob), t2c_r(nub,nub,nob,nob), t2b(nua,nub,noa,nob), t2b_r(nua,nub,nob,noa)
    real(p), intent(in) :: j1r(nua,nub,nub,noa), j2(nua,nob,noa,nob), j3(nua,nub,nua,nob)
    real(p), intent(in) :: j4(noa,nub,noa,nob), j5r(nub,nub,nub,nob), j6(nub,nob,nob,nob)
    real(p), intent(in) :: r1a(nua,noa), r1b(nub,nob)
    real(p), intent(in) :: r2c(nub,nub,nob,nob), r2c_r(nub,nub,nob,nob), r2b(nua,nub,noa,nob), r2b_r(nua,nub,nob,noa)
    ! left terms:
    !   h1r(c,b,e,k) = <ke|cb>_ab . l2c(a,e,i,j)        h2r(e,b,a,i) = <ei||ba> . l2b(c,e,k,j)
    !   h3r(c,a,e,i) = <ei|ca>_ab . l2b(e,b,k,j)        h4r(c,m,k,j) = <kj|cm>_ab . l2c_r(a,b,m,i)
    !   h5r(a,m,j,i) = <ji||ma> . l2b_r(c,b,m,k)        h6r(a,m,k,i) = <ki|ma>_ab . l2b(c,b,m,j)
    !   l1b(a,i) <kj|cb>_ab,  l1a(c,k) <ij||ab>,  l2b(c,b,k,j) f(i,a),  l2c(a,b,i,j) f(k,c)
    real(p), intent(in) :: h1r(nua,nub,nub,noa), h2r(nub,nub,nub,nob), h3r(nua,nub,nua,nob)
    real(p), intent(in) :: h4r(nua,nob,noa,nob), h5r(nub,nob,nob,nob), h6r(nub,noa,noa,nob)
    real(p), intent(in) :: hoovv_ab(noa,nob,nua,nub), hoovv_bb(nob,nob,nub,nub), hov_a(noa,nua), hov_b(nob,nub)
    real(p), intent(in) :: l1a(nua,noa), l1b(nub,nob), l2c(nub,nub,nob,nob), l2c_r(nub,nub,nob,nob)
    real(p), intent(in) :: l2b(nua,nub,noa,nob), l2b_r(nua,nub,nob,noa)
    real(p), intent(in) :: eoa(noa), eob(nob), eva(nua), evb(nub)
    real(p), intent(inout) :: d_oa(noa), d_ob(nob), d_va(nua), d_vb(nub)
    real(p), intent(inout) :: dx1r(nua,nub,nub,noa), dx2(nua,nob,noa,nob), dx3(nua,nub,nua,nob)
    real(p), intent(inout) :: dx4(noa,nub,noa,nob), dx5r(nub,nub,nub,nob), dx6(nub,nob,nob,nob)
    real(p), intent(inout) :: dt2c(nub,nub,nob,nob), dt2c_r(nub,nub,nob,nob), dt2b(nua,nub,noa,nob), dt2b_r(nua,nub,nob,noa)
    real(p), intent(inout) :: dxr1r(nua,nub,nub,noa), dxr2(nua,nob,noa,nob), dxr3(nua,nub,nua,nob)
    real(p), intent(inout) :: dxr4(noa,nub,noa,nob), dxr5r(nub,nub,nub,nob), dxr6(nub,nob,nob,nob)
    real(p), intent(inout) :: dtj2c(nub,nub,nob,nob), dtj2c_r(nub,nub,nob,nob), dtj2b(nua,nub,noa,nob), dtj2b_r(nua,nub,nob,noa)
    real(p), intent(inout) :: cp1a(nua,noa), cp1b(nub,nob), cp2b(nua,nub,noa,nob), cp2c(nub,nub,nob,nob)
    real(p), intent(inout) :: dh1r(nua,nub,nub,noa), dh2r(nub,nub,nub,nob), dh3r(nua,nub,nua,nob)
    real(p), intent(inout) :: dh4r(nua,nob,noa,nob), dh5r(nub,nob,nob,nob), dh6r(nub,noa,noa,nob)
    real(p), intent(inout) :: dhoovv_ab(noa,nob,nua,nub), dhoovv_bb(nob,nob,nub,nub), dhov_a(noa,nua), dhov_b(nob,nub)
    real(p), intent(inout) :: dl1a(nua,noa), dl1b(nub,nob), dl2c(nub,nub,nob,nob), dl2c_r(nub,nub,nob,nob)
    real(p), intent(inout) :: dl2b(nua,nub,noa,nob), dl2b_r(nua,nub,nob,noa)

    integer :: i, j, k, a, b, c, n, ii, jj, kk, nub2, nuab
    real(p) :: s, den, ssum
    real(p) :: xm(nua,nub,nub), xe(nua,nub,nub), xn(nua,nub,nub)
    real(p) :: bm_cba(nua,nub,nub), bm_abc(nub,nub,nua), bm_acb(nub,nua,nub)
    real(p) :: be_cba(nua,nub,nub), be_abc(nub,nub,nua), be_acb(nub,nua,nub)
    real(p) :: bn_cba(nua,nub,nub), bn_acb(nub,nua,nub)
    real(p) :: m3(nua,nub,nub), e3(nua,nub,nub), n3(nua,nub,nub), u3(nua,nub,nub), v3(nua,nub,nub)
    real(p) :: v_cba(nua,nub,nub), v_abc(nub,nub,nua), v_acb(nub,nua,nub), u_cba(nua,nub,nub), u_acb(nub,nua,nub)

    nub2 = nub*nub
    nuab = nua*nub
    delta = 0.0_p
    wsum = 0.0_p
    fr0 = 0.0_p

    do k = 1, noa
      do i = 1, nob
        do j = i+1, nob

          ! ---- forward
          xm = 0.0_p; xe = 0.0_p; xn = 0.0_p
          bm_cba = 0.0_p; bm_abc = 0.0_p; bm_acb = 0.0_p
          be_cba = 0.0_p; be_abc = 0.0_p; be_acb = 0.0_p
          bn_cba = 0.0_p; bn_acb = 0.0_p
          do n = 1, 2
            if (n == 1) then
              ii = i; jj = j; s = 1.0_p
            else
              ii = j; jj = i; s = -1.0_p
            end if
            kk = k
            ! moments
            call dgemm('n','t',nuab,nub,nub,s,x1r(:,:,:,kk),nuab,t2c(:,:,ii,jj),nub,1.0_p,bm_cba,nuab)
            call dgemm('n','t',nua,nub2,nob,s,x2(:,:,kk,jj),nua,t2c_r(:,:,:,ii),nub2,1.0_p,xm,nua)
            call dgemm('n','n',nuab,nub,nua,s,x3(:,:,:,ii),nuab,t2b(:,:,kk,jj),nua,1.0_p,xm,nuab)
            call dgemm('n','n',nuab,nub,noa,s,t2b(:,:,:,jj),nuab,x4(:,:,kk,ii),noa,1.0_p,bm_cba,nuab)
            call dgemm('n','t',nub2,nua,nub,s,x5r(:,:,:,ii),nub2,t2b(:,:,kk,jj),nua,1.0_p,bm_abc,nub2)
            call dgemm('n','t',nub,nuab,nob,s,x6(:,:,ii,jj),nub,t2b_r(:,:,:,kk),nuab,1.0_p,bm_acb,nub)
            if (excited == 1) then
              call dgemm('n','t',nuab,nub,nub,s,j1r(:,:,:,kk),nuab,t2c(:,:,ii,jj),nub,1.0_p,be_cba,nuab)
              call dgemm('n','t',nuab,nub,nub,s,x1r(:,:,:,kk),nuab,r2c(:,:,ii,jj),nub,1.0_p,be_cba,nuab)
              call dgemm('n','t',nua,nub2,nob,s,j2(:,:,kk,jj),nua,t2c_r(:,:,:,ii),nub2,1.0_p,xe,nua)
              call dgemm('n','t',nua,nub2,nob,s,x2(:,:,kk,jj),nua,r2c_r(:,:,:,ii),nub2,1.0_p,xe,nua)
              call dgemm('n','n',nuab,nub,nua,s,j3(:,:,:,ii),nuab,t2b(:,:,kk,jj),nua,1.0_p,xe,nuab)
              call dgemm('n','n',nuab,nub,nua,s,x3(:,:,:,ii),nuab,r2b(:,:,kk,jj),nua,1.0_p,xe,nuab)
              call dgemm('n','n',nuab,nub,noa,s,t2b(:,:,:,jj),nuab,j4(:,:,kk,ii),noa,1.0_p,be_cba,nuab)
              call dgemm('n','n',nuab,nub,noa,s,r2b(:,:,:,jj),nuab,x4(:,:,kk,ii),noa,1.0_p,be_cba,nuab)
              call dgemm('n','t',nub2,nua,nub,s,j5r(:,:,:,ii),nub2,t2b(:,:,kk,jj),nua,1.0_p,be_abc,nub2)
              call dgemm('n','t',nub2,nua,nub,s,x5r(:,:,:,ii),nub2,r2b(:,:,kk,jj),nua,1.0_p,be_abc,nub2)
              call dgemm('n','t',nub,nuab,nob,s,j6(:,:,ii,jj),nub,t2b_r(:,:,:,kk),nuab,1.0_p,be_acb,nub)
              call dgemm('n','t',nub,nuab,nob,s,x6(:,:,ii,jj),nub,r2b_r(:,:,:,kk),nuab,1.0_p,be_acb,nub)
            end if
            ! left projections
            call dgemm('n','t',nuab,nub,nub,0.5_p*s,h1r(:,:,:,kk),nuab,l2c(:,:,ii,jj),nub,1.0_p,bn_cba,nuab)
            call dgemm('n','n',nua,nub2,nub,0.5_p*s,l2b(:,:,kk,jj),nua,h2r(:,:,:,ii),nub,1.0_p,bn_cba,nua)
            call dgemm('n','n',nuab,nub,nua,s,h3r(:,:,:,ii),nuab,l2b(:,:,kk,jj),nua,1.0_p,xn,nuab)
            call dgemm('n','t',nua,nub2,nob,-0.5_p*s,h4r(:,:,kk,jj),nua,l2c_r(:,:,:,ii),nub2,1.0_p,xn,nua)
            call dgemm('n','t',nub,nuab,nob,-0.5_p*s,h5r(:,:,jj,ii),nub,l2b_r(:,:,:,kk),nuab,1.0_p,bn_acb,nub)
            call dgemm('n','t',nub,nuab,noa,-s,h6r(:,:,kk,ii),nub,l2b(:,:,:,jj),nuab,1.0_p,bn_acb,nub)
            do b = 1, nub
              do a = 1, nub
                do c = 1, nua
                  xn(c,a,b) = xn(c,a,b) + s*(l1b(a,ii)*hoovv_ab(kk,jj,c,b) + 0.25_p*l1a(c,kk)*hoovv_bb(ii,jj,a,b) &
                              + l2b(c,b,kk,jj)*hov_b(ii,a) + 0.25_p*l2c(a,b,ii,jj)*hov_a(kk,c))
                end do
              end do
            end do
          end do
          do b = 1, nub
            do a = 1, nub
              do c = 1, nua
                xm(c,a,b) = xm(c,a,b) + bm_cba(c,b,a) + bm_abc(a,b,c) + bm_acb(a,c,b)
                xe(c,a,b) = xe(c,a,b) + be_cba(c,b,a) + be_abc(a,b,c) + be_acb(a,c,b)
                xn(c,a,b) = xn(c,a,b) + bn_cba(c,b,a) + bn_acb(a,c,b)
              end do
            end do
          end do
          call antisym_bc(xm, m3, nua, nub)
          call antisym_bc(xn, n3, nua, nub)
          if (excited == 1) then
            call antisym_bc(xe, e3, nua, nub)
          else
            e3 = m3
          end if

          ! ---- divided moments and sums (1/2: unique virtual triples)
          do b = 1, nub
            do a = 1, nub
              do c = 1, nua
                den = omega + eoa(k) + eob(i) + eob(j) - eva(c) - evb(a) - evb(b)
                u3(c,a,b) = e3(c,a,b)/den
                v3(c,a,b) = n3(c,a,b)/den
              end do
            end do
          end do
          delta = delta + sum(v3*e3)/2.0_p
          fr0 = fr0 + sum(v3*m3)/2.0_p
          ssum = sum(u3*v3)/2.0_p
          wsum = wsum + ssum
          d_oa(k) = d_oa(k) - ssum
          d_ob(i) = d_ob(i) - ssum
          d_ob(j) = d_ob(j) - ssum
          do c = 1, nua
            d_va(c) = d_va(c) + 0.5_p*sum(u3(c,:,:)*v3(c,:,:))
          end do
          do a = 1, nub
            d_vb(a) = d_vb(a) + sum(u3(:,a,:)*v3(:,a,:))
          end do
          do b = 1, nub
            do a = 1, nub
              do c = 1, nua
                v_cba(c,b,a) = v3(c,a,b)
                v_abc(a,b,c) = v3(c,a,b)
                v_acb(a,c,b) = v3(c,a,b)
                u_cba(c,b,a) = u3(c,a,b)
                u_acb(a,c,b) = u3(c,a,b)
              end do
            end do
          end do

          ! ---- reverse
          do n = 1, 2
            if (n == 1) then
              ii = i; jj = j; s = 1.0_p
            else
              ii = j; jj = i; s = -1.0_p
            end if
            kk = k
            ! v . M
            call dgemm('n','n',nuab,nub,nub,s,v_cba,nuab,t2c(:,:,ii,jj),nub,1.0_p,dx1r(:,:,:,kk),nuab)
            call dgemm('t','n',nub,nub,nuab,s,v_cba,nuab,x1r(:,:,:,kk),nuab,1.0_p,dt2c(:,:,ii,jj),nub)
            call dgemm('n','n',nua,nob,nub2,s,v3,nua,t2c_r(:,:,:,ii),nub2,1.0_p,dx2(:,:,kk,jj),nua)
            call dgemm('t','n',nub2,nob,nua,s,v3,nua,x2(:,:,kk,jj),nua,1.0_p,dt2c_r(:,:,:,ii),nub2)
            call dgemm('n','t',nuab,nua,nub,s,v3,nuab,t2b(:,:,kk,jj),nua,1.0_p,dx3(:,:,:,ii),nuab)
            call dgemm('t','n',nua,nub,nuab,s,x3(:,:,:,ii),nuab,v3,nuab,1.0_p,dt2b(:,:,kk,jj),nua)
            call dgemm('n','t',nuab,noa,nub,s,v_cba,nuab,x4(:,:,kk,ii),noa,1.0_p,dt2b(:,:,:,jj),nuab)
            call dgemm('t','n',noa,nub,nuab,s,t2b(:,:,:,jj),nuab,v_cba,nuab,1.0_p,dx4(:,:,kk,ii),noa)
            call dgemm('n','n',nub2,nub,nua,s,v_abc,nub2,t2b(:,:,kk,jj),nua,1.0_p,dx5r(:,:,:,ii),nub2)
            call dgemm('t','n',nua,nub,nub2,s,v_abc,nub2,x5r(:,:,:,ii),nub2,1.0_p,dt2b(:,:,kk,jj),nua)
            call dgemm('n','n',nub,nob,nuab,s,v_acb,nub,t2b_r(:,:,:,kk),nuab,1.0_p,dx6(:,:,ii,jj),nub)
            call dgemm('t','n',nuab,nob,nub,s,v_acb,nub,x6(:,:,ii,jj),nub,1.0_p,dt2b_r(:,:,:,kk),nuab)
            if (excited == 1) then
              call dgemm('n','n',nuab,nub,nub,s,v_cba,nuab,r2c(:,:,ii,jj),nub,1.0_p,dxr1r(:,:,:,kk),nuab)
              call dgemm('t','n',nub,nub,nuab,s,v_cba,nuab,j1r(:,:,:,kk),nuab,1.0_p,dtj2c(:,:,ii,jj),nub)
              call dgemm('n','n',nua,nob,nub2,s,v3,nua,r2c_r(:,:,:,ii),nub2,1.0_p,dxr2(:,:,kk,jj),nua)
              call dgemm('t','n',nub2,nob,nua,s,v3,nua,j2(:,:,kk,jj),nua,1.0_p,dtj2c_r(:,:,:,ii),nub2)
              call dgemm('n','t',nuab,nua,nub,s,v3,nuab,r2b(:,:,kk,jj),nua,1.0_p,dxr3(:,:,:,ii),nuab)
              call dgemm('t','n',nua,nub,nuab,s,j3(:,:,:,ii),nuab,v3,nuab,1.0_p,dtj2b(:,:,kk,jj),nua)
              call dgemm('n','t',nuab,noa,nub,s,v_cba,nuab,j4(:,:,kk,ii),noa,1.0_p,dtj2b(:,:,:,jj),nuab)
              call dgemm('t','n',noa,nub,nuab,s,r2b(:,:,:,jj),nuab,v_cba,nuab,1.0_p,dxr4(:,:,kk,ii),noa)
              call dgemm('n','n',nub2,nub,nua,s,v_abc,nub2,r2b(:,:,kk,jj),nua,1.0_p,dxr5r(:,:,:,ii),nub2)
              call dgemm('t','n',nua,nub,nub2,s,v_abc,nub2,j5r(:,:,:,ii),nub2,1.0_p,dtj2b(:,:,kk,jj),nua)
              call dgemm('n','n',nub,nob,nuab,s,v_acb,nub,r2b_r(:,:,:,kk),nuab,1.0_p,dxr6(:,:,ii,jj),nub)
              call dgemm('t','n',nuab,nob,nub,s,v_acb,nub,j6(:,:,ii,jj),nub,1.0_p,dtj2b_r(:,:,:,kk),nuab)
              ! c' = <V3|R|X>
              call dgemv('n',nua,nub2,0.25_p*s,v3,nua,r2c(:,:,ii,jj),1,1.0_p,cp1a(:,kk),1)
              call dgemv('n',nub,nuab,s,v_acb,nub,r2b(:,:,kk,jj),1,1.0_p,cp1b(:,ii),1)
              call dgemv('n',nuab,nub,s,v3,nuab,r1b(:,jj),1,1.0_p,cp2b(:,:,kk,ii),1)
              call dgemv('t',nua,nub2,s,v3,nua,r1a(:,kk),1,1.0_p,cp2c(:,:,ii,jj),1)
            end if
            ! u . N
            call dgemm('n','n',nuab,nub,nub,0.5_p*s,u_cba,nuab,l2c(:,:,ii,jj),nub,1.0_p,dh1r(:,:,:,kk),nuab)
            call dgemm('t','n',nub,nub,nuab,0.5_p*s,u_cba,nuab,h1r(:,:,:,kk),nuab,1.0_p,dl2c(:,:,ii,jj),nub)
            call dgemm('n','t',nua,nub,nub2,0.5_p*s,u_cba,nua,h2r(:,:,:,ii),nub,1.0_p,dl2b(:,:,kk,jj),nua)
            call dgemm('t','n',nub,nub2,nua,0.5_p*s,l2b(:,:,kk,jj),nua,u_cba,nua,1.0_p,dh2r(:,:,:,ii),nub)
            call dgemm('n','t',nuab,nua,nub,s,u3,nuab,l2b(:,:,kk,jj),nua,1.0_p,dh3r(:,:,:,ii),nuab)
            call dgemm('t','n',nua,nub,nuab,s,h3r(:,:,:,ii),nuab,u3,nuab,1.0_p,dl2b(:,:,kk,jj),nua)
            call dgemm('n','n',nua,nob,nub2,-0.5_p*s,u3,nua,l2c_r(:,:,:,ii),nub2,1.0_p,dh4r(:,:,kk,jj),nua)
            call dgemm('t','n',nub2,nob,nua,-0.5_p*s,u3,nua,h4r(:,:,kk,jj),nua,1.0_p,dl2c_r(:,:,:,ii),nub2)
            call dgemm('n','n',nub,nob,nuab,-0.5_p*s,u_acb,nub,l2b_r(:,:,:,kk),nuab,1.0_p,dh5r(:,:,jj,ii),nub)
            call dgemm('t','n',nuab,nob,nub,-0.5_p*s,u_acb,nub,h5r(:,:,jj,ii),nub,1.0_p,dl2b_r(:,:,:,kk),nuab)
            call dgemm('n','n',nub,noa,nuab,-s,u_acb,nub,l2b(:,:,:,jj),nuab,1.0_p,dh6r(:,:,kk,ii),nub)
            call dgemm('t','n',nuab,noa,nub,-s,u_acb,nub,h6r(:,:,kk,ii),nub,1.0_p,dl2b(:,:,:,jj),nuab)
            do b = 1, nub
              do a = 1, nub
                do c = 1, nua
                  dl1b(a,ii) = dl1b(a,ii) + s*u3(c,a,b)*hoovv_ab(kk,jj,c,b)
                  dhoovv_ab(kk,jj,c,b) = dhoovv_ab(kk,jj,c,b) + s*u3(c,a,b)*l1b(a,ii)
                  dl1a(c,kk) = dl1a(c,kk) + 0.25_p*s*u3(c,a,b)*hoovv_bb(ii,jj,a,b)
                  dhoovv_bb(ii,jj,a,b) = dhoovv_bb(ii,jj,a,b) + 0.25_p*s*u3(c,a,b)*l1a(c,kk)
                  dl2b(c,b,kk,jj) = dl2b(c,b,kk,jj) + s*u3(c,a,b)*hov_b(ii,a)
                  dhov_b(ii,a) = dhov_b(ii,a) + s*u3(c,a,b)*l2b(c,b,kk,jj)
                  dl2c(a,b,ii,jj) = dl2c(a,b,ii,jj) + 0.25_p*s*u3(c,a,b)*hov_a(kk,c)
                  dhov_a(kk,c) = dhov_a(kk,c) + 0.25_p*s*u3(c,a,b)*l2c(a,b,ii,jj)
                end do
              end do
            end do
          end do

        end do
      end do
    end do

  end subroutine grad_abb

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! antisymmetrizers !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine antisym_abc(x, y, n)
    ! y(a,b,c) = sum over the six permutations of (a,b,c), with signs, of x
    integer, intent(in) :: n
    real(p), intent(in) :: x(n,n,n)
    real(p), intent(out) :: y(n,n,n)
    integer :: a, b, c
    do c = 1, n
      do b = 1, n
        do a = 1, n
          y(a,b,c) = x(a,b,c) + x(b,c,a) + x(c,a,b) - x(a,c,b) - x(b,a,c) - x(c,b,a)
        end do
      end do
    end do
  end subroutine antisym_abc

  subroutine antisym_ab(x, y, na, nb)
    ! y(a,b,c) = x(a,b,c) - x(b,a,c)
    integer, intent(in) :: na, nb
    real(p), intent(in) :: x(na,na,nb)
    real(p), intent(out) :: y(na,na,nb)
    integer :: a, b, c
    do c = 1, nb
      do b = 1, na
        do a = 1, na
          y(a,b,c) = x(a,b,c) - x(b,a,c)
        end do
      end do
    end do
  end subroutine antisym_ab

  subroutine antisym_bc(x, y, na, nb)
    ! y(c,a,b) = x(c,a,b) - x(c,b,a)
    integer, intent(in) :: na, nb
    real(p), intent(in) :: x(na,nb,nb)
    real(p), intent(out) :: y(na,nb,nb)
    integer :: a, b, c
    do b = 1, nb
      do a = 1, nb
        do c = 1, na
          y(c,a,b) = x(c,a,b) - x(c,b,a)
        end do
      end do
    end do
  end subroutine antisym_bc

end module crcc23_gradient_loops
