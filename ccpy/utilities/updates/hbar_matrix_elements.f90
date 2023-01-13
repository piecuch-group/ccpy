module hbar_matrix_elements

      implicit none

      contains


          pure function aaa_oo_aaa(i, j, k, a, b, c, l, m, n, d, e, f, h, noa) result(hmatel)
                  ! Expression:
                  ! -A(abc)A(jk)A(l/mn)A(i/jk) d(a,d)d(b,e)d(c,f)d(j,m)d(k,n) h(l,i)

                  integer, intent(in) :: noa
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: h(1:noa,1:noa)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  ! (1)
                  if (a==d .and. b==e .and. c==f) then
                     ! (1)
                     if (j==m .and. k==n) hmatel = hmatel - h(l,i) ! (1)
                     if (j==l .and. k==n) hmatel = hmatel + h(m,i) ! (lm)
                     if (j==m .and. k==l) hmatel = hmatel + h(n,i) ! (ln)
                     if (i==m .and. k==n) hmatel = hmatel + h(l,j) ! (ij)
                     if (i==l .and. k==n) hmatel = hmatel - h(m,j) ! (lm)(ij)
                     if (i==m .and. k==l) hmatel = hmatel - h(n,j) ! (ln)(ij)
                     if (j==m .and. i==n) hmatel = hmatel + h(l,k) ! (ik)
                     if (j==l .and. i==n) hmatel = hmatel - h(m,k) ! (lm)(ik)
                     if (j==m .and. i==l) hmatel = hmatel - h(n,k) ! (ln)(ik)
                     ! (jk)
                     if (k==m .and. j==n) hmatel = hmatel + h(l,i) ! (1)
                     if (k==l .and. j==n) hmatel = hmatel - h(m,i) ! (lm)
                     if (k==m .and. j==l) hmatel = hmatel - h(n,i) ! (ln)
                     if (i==m .and. j==n) hmatel = hmatel - h(l,k) ! (ij)
                     if (i==l .and. j==n) hmatel = hmatel + h(m,k) ! (lm)(ij)
                     if (i==m .and. j==l) hmatel = hmatel + h(n,k) ! (ln)(ij)
                     if (k==m .and. i==n) hmatel = hmatel - h(l,j) ! (ik)
                     if (k==l .and. i==n) hmatel = hmatel + h(m,j) ! (lm)(ik)
                     if (k==m .and. i==l) hmatel = hmatel + h(n,j) ! (ln)(ik)
                  end if
          end function aaa_oo_aaa

          pure function aaa_vv_aaa(i, j, k, a, b, c, l, m, n, d, e, f, h, nua) result(hmatel)
                  ! Expression:
                  ! A(ijk)A(bc)A(d/ef)A(a/bc) d(i,l)d(j,m)d(k,n)d(b,e)d(c,f) h(a,d)

                  integer, intent(in) :: nua
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: h(1:nua,1:nua)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  ! (1)
                  if (i==l .and. j==m .and. k==n) then
                      ! (1)
                      if (b==e .and. c==f) hmatel = hmatel + h(a,d) ! (1)
                      if (b==d .and. c==f) hmatel = hmatel - h(a,e) ! (de)
                      if (b==e .and. c==d) hmatel = hmatel - h(a,f) ! (df)
                      if (a==e .and. c==f) hmatel = hmatel - h(b,d) ! (ab)
                      if (a==d .and. c==f) hmatel = hmatel + h(b,e) ! (de)(ab)
                      if (a==e .and. c==d) hmatel = hmatel + h(b,f) ! (df)(ab)
                      if (b==e .and. a==f) hmatel = hmatel - h(c,d) ! (ac)
                      if (b==d .and. a==f) hmatel = hmatel + h(c,e) ! (de)(ac)
                      if (b==e .and. a==d) hmatel = hmatel + h(c,f) ! (df)(ac)
                      ! (bc)
                      if (c==e .and. b==f) hmatel = hmatel - h(a,d) ! (1)
                      if (c==d .and. b==f) hmatel = hmatel + h(a,e) ! (de)
                      if (c==e .and. b==d) hmatel = hmatel + h(a,f) ! (df)
                      if (a==e .and. b==f) hmatel = hmatel + h(c,d) ! (ab)
                      if (a==d .and. b==f) hmatel = hmatel - h(c,e) ! (de)(ab)
                      if (a==e .and. b==d) hmatel = hmatel - h(c,f) ! (df)(ab)
                      if (c==e .and. a==f) hmatel = hmatel + h(b,d) ! (ac)
                      if (c==d .and. a==f) hmatel = hmatel - h(b,e) ! (de)(ac)
                      if (c==e .and. a==d) hmatel = hmatel - h(b,f) ! (df)(ac)
                  end if
          end function aaa_vv_aaa

          pure function aaa_oooo_aaa(i, j, k, a, b, c, l, m, n, d, e, f, h, noa) result(hmatel)
                  ! Expression:
                  ! A(abc)A(k/ij)A(n/lm) d(a,d)d(b,e)d(c,f)d(k,n) h(l,m,i,j)

                  integer, intent(in) :: noa
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: h(1:noa,1:noa,1:noa,1:noa)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  ! (1)
                  if (a==d .and. b==e .and. c==f) then
                      ! (1)
                      if (k==n) hmatel = hmatel + h(l,m,i,j) ! (1)
                      if (k==l) hmatel = hmatel - h(n,m,i,j) ! (ln)
                      if (k==m) hmatel = hmatel - h(l,n,i,j) ! (mn)
                      ! (ik)
                      if (i==n) hmatel = hmatel - h(l,m,k,j) ! (1)
                      if (i==l) hmatel = hmatel + h(n,m,k,j) ! (ln)
                      if (i==m) hmatel = hmatel + h(l,n,k,j) ! (mn)
                      ! (jk)
                      if (j==n) hmatel = hmatel - h(l,m,i,k) ! (1)
                      if (j==l) hmatel = hmatel + h(n,m,i,k) ! (ln)
                      if (j==m) hmatel = hmatel + h(l,n,i,k) ! (mn)
                  end if
          end function aaa_oooo_aaa

          pure function aaa_vvvv_aaa(i, j, k, a, b, c, l, m, n, d, e, f, h, nua) result(hmatel)
                  ! Expression:
                  ! A(ijk)A(c/ab)A(f/de) d(i,l)d(j,m)d(k,n)d(c,f) h(a,b,d,e)

                  integer, intent(in) :: nua
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: h(1:nua,1:nua,1:nua,1:nua)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  ! (1)
                  if (i==l .and. j==m .and. k==n) then
                      ! (1)
                      if (c==f) hmatel = hmatel + h(a,b,d,e) ! (1)
                      if (a==f) hmatel = hmatel - h(c,b,d,e) ! (ac)
                      if (b==f) hmatel = hmatel - h(a,c,d,e) ! (bc)
                      ! (fd)
                      if (c==d) hmatel = hmatel - h(a,b,f,e) ! (1)
                      if (a==d) hmatel = hmatel + h(c,b,f,e) ! (ac)
                      if (b==d) hmatel = hmatel + h(a,c,f,e) ! (bc)
                      ! (fe)
                      if (c==e) hmatel = hmatel - h(a,b,d,f) ! (1)
                      if (a==e) hmatel = hmatel + h(c,b,d,f) ! (ac)
                      if (b==e) hmatel = hmatel + h(a,c,d,f) ! (bc)
                  end if
          end function aaa_vvvv_aaa

          pure function aaa_voov_aaa(i, j, k, a, b, c, l, m, n, d, e, f, h, noa, nua) result(hmatel)
                  ! Expression:
                  !  A(jk)A(bc)A(i/jk)A(a/bc)A(l/mn)A(d/ef) d(j,m)d(k,n)d(b,e)d(c,f) h(a,l,i,d)

                  integer, intent(in) :: nua, noa
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: h(1:nua,1:noa,1:noa,1:nua)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  !!! A(bc)A(i/jk)A(a/bc)A(l/mn)A(d/ef) !!!
                  if (j==m .and. k==n .and. b==e .and. c==f) hmatel = hmatel + h(a,l,i,d) ! (1)
                  if (j==m .and. k==n .and. b==d .and. c==f) hmatel = hmatel - h(a,l,i,e) ! (de)
                  if (j==m .and. k==n .and. b==e .and. c==d) hmatel = hmatel - h(a,l,i,f) ! (df)
                  if (j==l .and. k==n .and. b==e .and. c==f) hmatel = hmatel - h(a,m,i,d) ! (lm)
                  if (j==l .and. k==n .and. b==d .and. c==f) hmatel = hmatel + h(a,m,i,e) ! (de)(lm)
                  if (j==l .and. k==n .and. b==e .and. c==d) hmatel = hmatel + h(a,m,i,f) ! (df)(lm)
                  if (j==m .and. k==l .and. b==e .and. c==f) hmatel = hmatel - h(a,n,i,d) ! (ln)
                  if (j==m .and. k==l .and. b==d .and. c==f) hmatel = hmatel + h(a,n,i,e) ! (de)(ln)
                  if (j==m .and. k==l .and. b==e .and. c==d) hmatel = hmatel + h(a,n,i,f) ! (df)(ln)
                  if (j==m .and. k==n .and. a==e .and. c==f) hmatel = hmatel - h(b,l,i,d) ! (ab)
                  if (j==m .and. k==n .and. a==d .and. c==f) hmatel = hmatel + h(b,l,i,e) ! (de)(ab)
                  if (j==m .and. k==n .and. a==e .and. c==d) hmatel = hmatel + h(b,l,i,f) ! (df)(ab)
                  if (j==l .and. k==n .and. a==e .and. c==f) hmatel = hmatel + h(b,m,i,d) ! (lm)(ab)
                  if (j==l .and. k==n .and. a==d .and. c==f) hmatel = hmatel - h(b,m,i,e) ! (de)(lm)(ab)
                  if (j==l .and. k==n .and. a==e .and. c==d) hmatel = hmatel - h(b,m,i,f) ! (df)(lm)(ab)
                  if (j==m .and. k==l .and. a==e .and. c==f) hmatel = hmatel + h(b,n,i,d) ! (ln)(ab)
                  if (j==m .and. k==l .and. a==d .and. c==f) hmatel = hmatel - h(b,n,i,e) ! (de)(ln)(ab)
                  if (j==m .and. k==l .and. a==e .and. c==d) hmatel = hmatel - h(b,n,i,f) ! (df)(ln)(ab)
                  if (j==m .and. k==n .and. b==e .and. a==f) hmatel = hmatel - h(c,l,i,d) ! (ac)
                  if (j==m .and. k==n .and. b==d .and. a==f) hmatel = hmatel + h(c,l,i,e) ! (de)(ac)
                  if (j==m .and. k==n .and. b==e .and. a==d) hmatel = hmatel + h(c,l,i,f) ! (df)(ac)
                  if (j==l .and. k==n .and. b==e .and. a==f) hmatel = hmatel + h(c,m,i,d) ! (lm)(ac)
                  if (j==l .and. k==n .and. b==d .and. a==f) hmatel = hmatel - h(c,m,i,e) ! (de)(lm)(ac)
                  if (j==l .and. k==n .and. b==e .and. a==d) hmatel = hmatel - h(c,m,i,f) ! (df)(lm)(ac)
                  if (j==m .and. k==l .and. b==e .and. a==f) hmatel = hmatel + h(c,n,i,d) ! (ln)(ac)
                  if (j==m .and. k==l .and. b==d .and. a==f) hmatel = hmatel - h(c,n,i,e) ! (de)(ln)(ac)
                  if (j==m .and. k==l .and. b==e .and. a==d) hmatel = hmatel - h(c,n,i,f) ! (df)(ln)(ac)
                  if (i==m .and. k==n .and. b==e .and. c==f) hmatel = hmatel - h(a,l,j,d) ! (ij)
                  if (i==m .and. k==n .and. b==d .and. c==f) hmatel = hmatel + h(a,l,j,e) ! (de)(ij)
                  if (i==m .and. k==n .and. b==e .and. c==d) hmatel = hmatel + h(a,l,j,f) ! (df)(ij)
                  if (i==l .and. k==n .and. b==e .and. c==f) hmatel = hmatel + h(a,m,j,d) ! (lm)(ij)
                  if (i==l .and. k==n .and. b==d .and. c==f) hmatel = hmatel - h(a,m,j,e) ! (de)(lm)(ij)
                  if (i==l .and. k==n .and. b==e .and. c==d) hmatel = hmatel - h(a,m,j,f) ! (df)(lm)(ij)
                  if (i==m .and. k==l .and. b==e .and. c==f) hmatel = hmatel + h(a,n,j,d) ! (ln)(ij)
                  if (i==m .and. k==l .and. b==d .and. c==f) hmatel = hmatel - h(a,n,j,e) ! (de)(ln)(ij)
                  if (i==m .and. k==l .and. b==e .and. c==d) hmatel = hmatel - h(a,n,j,f) ! (df)(ln)(ij)
                  if (i==m .and. k==n .and. a==e .and. c==f) hmatel = hmatel + h(b,l,j,d) ! (ab)(ij)
                  if (i==m .and. k==n .and. a==d .and. c==f) hmatel = hmatel - h(b,l,j,e) ! (de)(ab)(ij)
                  if (i==m .and. k==n .and. a==e .and. c==d) hmatel = hmatel - h(b,l,j,f) ! (df)(ab)(ij)
                  if (i==l .and. k==n .and. a==e .and. c==f) hmatel = hmatel - h(b,m,j,d) ! (lm)(ab)(ij)
                  if (i==l .and. k==n .and. a==d .and. c==f) hmatel = hmatel + h(b,m,j,e) ! (de)(lm)(ab)(ij)
                  if (i==l .and. k==n .and. a==e .and. c==d) hmatel = hmatel + h(b,m,j,f) ! (df)(lm)(ab)(ij)
                  if (i==m .and. k==l .and. a==e .and. c==f) hmatel = hmatel - h(b,n,j,d) ! (ln)(ab)(ij)
                  if (i==m .and. k==l .and. a==d .and. c==f) hmatel = hmatel + h(b,n,j,e) ! (de)(ln)(ab)(ij)
                  if (i==m .and. k==l .and. a==e .and. c==d) hmatel = hmatel + h(b,n,j,f) ! (df)(ln)(ab)(ij)
                  if (i==m .and. k==n .and. b==e .and. a==f) hmatel = hmatel + h(c,l,j,d) ! (ac)(ij)
                  if (i==m .and. k==n .and. b==d .and. a==f) hmatel = hmatel - h(c,l,j,e) ! (de)(ac)(ij)
                  if (i==m .and. k==n .and. b==e .and. a==d) hmatel = hmatel - h(c,l,j,f) ! (df)(ac)(ij)
                  if (i==l .and. k==n .and. b==e .and. a==f) hmatel = hmatel - h(c,m,j,d) ! (lm)(ac)(ij)
                  if (i==l .and. k==n .and. b==d .and. a==f) hmatel = hmatel + h(c,m,j,e) ! (de)(lm)(ac)(ij)
                  if (i==l .and. k==n .and. b==e .and. a==d) hmatel = hmatel + h(c,m,j,f) ! (df)(lm)(ac)(ij)
                  if (i==m .and. k==l .and. b==e .and. a==f) hmatel = hmatel - h(c,n,j,d) ! (ln)(ac)(ij)
                  if (i==m .and. k==l .and. b==d .and. a==f) hmatel = hmatel + h(c,n,j,e) ! (de)(ln)(ac)(ij)
                  if (i==m .and. k==l .and. b==e .and. a==d) hmatel = hmatel + h(c,n,j,f) ! (df)(ln)(ac)(ij)
                  if (j==m .and. i==n .and. b==e .and. c==f) hmatel = hmatel - h(a,l,k,d) ! (ik)
                  if (j==m .and. i==n .and. b==d .and. c==f) hmatel = hmatel + h(a,l,k,e) ! (de)(ik)
                  if (j==m .and. i==n .and. b==e .and. c==d) hmatel = hmatel + h(a,l,k,f) ! (df)(ik)
                  if (j==l .and. i==n .and. b==e .and. c==f) hmatel = hmatel + h(a,m,k,d) ! (lm)(ik)
                  if (j==l .and. i==n .and. b==d .and. c==f) hmatel = hmatel - h(a,m,k,e) ! (de)(lm)(ik)
                  if (j==l .and. i==n .and. b==e .and. c==d) hmatel = hmatel - h(a,m,k,f) ! (df)(lm)(ik)
                  if (j==m .and. i==l .and. b==e .and. c==f) hmatel = hmatel + h(a,n,k,d) ! (ln)(ik)
                  if (j==m .and. i==l .and. b==d .and. c==f) hmatel = hmatel - h(a,n,k,e) ! (de)(ln)(ik)
                  if (j==m .and. i==l .and. b==e .and. c==d) hmatel = hmatel - h(a,n,k,f) ! (df)(ln)(ik)
                  if (j==m .and. i==n .and. a==e .and. c==f) hmatel = hmatel + h(b,l,k,d) ! (ab)(ik)
                  if (j==m .and. i==n .and. a==d .and. c==f) hmatel = hmatel - h(b,l,k,e) ! (de)(ab)(ik)
                  if (j==m .and. i==n .and. a==e .and. c==d) hmatel = hmatel - h(b,l,k,f) ! (df)(ab)(ik)
                  if (j==l .and. i==n .and. a==e .and. c==f) hmatel = hmatel - h(b,m,k,d) ! (lm)(ab)(ik)
                  if (j==l .and. i==n .and. a==d .and. c==f) hmatel = hmatel + h(b,m,k,e) ! (de)(lm)(ab)(ik)
                  if (j==l .and. i==n .and. a==e .and. c==d) hmatel = hmatel + h(b,m,k,f) ! (df)(lm)(ab)(ik)
                  if (j==m .and. i==l .and. a==e .and. c==f) hmatel = hmatel - h(b,n,k,d) ! (ln)(ab)(ik)
                  if (j==m .and. i==l .and. a==d .and. c==f) hmatel = hmatel + h(b,n,k,e) ! (de)(ln)(ab)(ik)
                  if (j==m .and. i==l .and. a==e .and. c==d) hmatel = hmatel + h(b,n,k,f) ! (df)(ln)(ab)(ik)
                  if (j==m .and. i==n .and. b==e .and. a==f) hmatel = hmatel + h(c,l,k,d) ! (ac)(ik)
                  if (j==m .and. i==n .and. b==d .and. a==f) hmatel = hmatel - h(c,l,k,e) ! (de)(ac)(ik)
                  if (j==m .and. i==n .and. b==e .and. a==d) hmatel = hmatel - h(c,l,k,f) ! (df)(ac)(ik)
                  if (j==l .and. i==n .and. b==e .and. a==f) hmatel = hmatel - h(c,m,k,d) ! (lm)(ac)(ik)
                  if (j==l .and. i==n .and. b==d .and. a==f) hmatel = hmatel + h(c,m,k,e) ! (de)(lm)(ac)(ik)
                  if (j==l .and. i==n .and. b==e .and. a==d) hmatel = hmatel + h(c,m,k,f) ! (df)(lm)(ac)(ik)
                  if (j==m .and. i==l .and. b==e .and. a==f) hmatel = hmatel - h(c,n,k,d) ! (ln)(ac)(ik)
                  if (j==m .and. i==l .and. b==d .and. a==f) hmatel = hmatel + h(c,n,k,e) ! (de)(ln)(ac)(ik)
                  if (j==m .and. i==l .and. b==e .and. a==d) hmatel = hmatel + h(c,n,k,f) ! (df)(ln)(ac)(ik)

                  if (j==m .and. k==n .and. c==e .and. b==f) hmatel = hmatel - h(a,l,i,d) ! (bc)
                  if (j==m .and. k==n .and. c==d .and. b==f) hmatel = hmatel + h(a,l,i,e) ! (de)(bc)
                  if (j==m .and. k==n .and. c==e .and. b==d) hmatel = hmatel + h(a,l,i,f) ! (df)(bc)
                  if (j==l .and. k==n .and. c==e .and. b==f) hmatel = hmatel + h(a,m,i,d) ! (lm)(bc)
                  if (j==l .and. k==n .and. c==d .and. b==f) hmatel = hmatel - h(a,m,i,e) ! (de)(lm)(bc)
                  if (j==l .and. k==n .and. c==e .and. b==d) hmatel = hmatel - h(a,m,i,f) ! (df)(lm)(bc)
                  if (j==m .and. k==l .and. c==e .and. b==f) hmatel = hmatel + h(a,n,i,d) ! (ln)(bc)
                  if (j==m .and. k==l .and. c==d .and. b==f) hmatel = hmatel - h(a,n,i,e) ! (de)(ln)(bc)
                  if (j==m .and. k==l .and. c==e .and. b==d) hmatel = hmatel - h(a,n,i,f) ! (df)(ln)(bc)

                  if (j==m .and. k==n .and. a==e .and. b==f) hmatel = hmatel + h(c,l,i,d) ! (ab)(bc)
                  if (j==m .and. k==n .and. a==d .and. b==f) hmatel = hmatel - h(c,l,i,e) ! (de)(ab)(bc)
                  if (j==m .and. k==n .and. a==e .and. b==d) hmatel = hmatel - h(c,l,i,f) ! (df)(ab)(bc)
                  if (j==l .and. k==n .and. a==e .and. b==f) hmatel = hmatel - h(c,m,i,d) ! (lm)(ab)(bc)
                  if (j==l .and. k==n .and. a==d .and. b==f) hmatel = hmatel + h(c,m,i,e) ! (de)(lm)(ab)(bc)
                  if (j==l .and. k==n .and. a==e .and. b==d) hmatel = hmatel + h(c,m,i,f) ! (df)(lm)(ab)(bc)
                  if (j==m .and. k==l .and. a==e .and. b==f) hmatel = hmatel - h(c,n,i,d) ! (ln)(ab)(bc)
                  if (j==m .and. k==l .and. a==d .and. b==f) hmatel = hmatel + h(c,n,i,e) ! (de)(ln)(ab)(bc)
                  if (j==m .and. k==l .and. a==e .and. b==d) hmatel = hmatel + h(c,n,i,f) ! (df)(ln)(ab)(bc)

                  if (j==m .and. k==n .and. c==e .and. a==f) hmatel = hmatel + h(b,l,i,d) ! (ac)(bc)
                  if (j==m .and. k==n .and. c==d .and. a==f) hmatel = hmatel - h(b,l,i,e) ! (de)(ac)(bc)
                  if (j==m .and. k==n .and. c==e .and. a==d) hmatel = hmatel - h(b,l,i,f) ! (df)(ac)(bc)
                  if (j==l .and. k==n .and. c==e .and. a==f) hmatel = hmatel - h(b,m,i,d) ! (lm)(ac)(bc)
                  if (j==l .and. k==n .and. c==d .and. a==f) hmatel = hmatel + h(b,m,i,e) ! (de)(lm)(ac)(bc)
                  if (j==l .and. k==n .and. c==e .and. a==d) hmatel = hmatel + h(b,m,i,f) ! (df)(lm)(ac)(bc)
                  if (j==m .and. k==l .and. c==e .and. a==f) hmatel = hmatel - h(b,n,i,d) ! (ln)(ac)(bc)
                  if (j==m .and. k==l .and. c==d .and. a==f) hmatel = hmatel + h(b,n,i,e) ! (de)(ln)(ac)(bc)
                  if (j==m .and. k==l .and. c==e .and. a==d) hmatel = hmatel + h(b,n,i,f) ! (df)(ln)(ac)(bc)

                  if (i==m .and. k==n .and. c==e .and. b==f) hmatel = hmatel + h(a,l,j,d) ! (ij)(bc)
                  if (i==m .and. k==n .and. c==d .and. b==f) hmatel = hmatel - h(a,l,j,e) ! (de)(ij)(bc)
                  if (i==m .and. k==n .and. c==e .and. b==d) hmatel = hmatel - h(a,l,j,f) ! (df)(ij)(bc)
                  if (i==l .and. k==n .and. c==e .and. b==f) hmatel = hmatel - h(a,m,j,d) ! (lm)(ij)(bc)
                  if (i==l .and. k==n .and. c==d .and. b==f) hmatel = hmatel + h(a,m,j,e) ! (de)(lm)(ij)(bc)
                  if (i==l .and. k==n .and. c==e .and. b==d) hmatel = hmatel + h(a,m,j,f) ! (df)(lm)(ij)(bc)
                  if (i==m .and. k==l .and. c==e .and. b==f) hmatel = hmatel - h(a,n,j,d) ! (ln)(ij)(bc)
                  if (i==m .and. k==l .and. c==d .and. b==f) hmatel = hmatel + h(a,n,j,e) ! (de)(ln)(ij)(bc)
                  if (i==m .and. k==l .and. c==e .and. b==d) hmatel = hmatel + h(a,n,j,f) ! (df)(ln)(ij)(bc)

                  if (i==m .and. k==n .and. a==e .and. b==f) hmatel = hmatel - h(c,l,j,d) ! (ab)(ij)(bc)
                  if (i==m .and. k==n .and. a==d .and. b==f) hmatel = hmatel + h(c,l,j,e) ! (de)(ab)(ij)(bc)
                  if (i==m .and. k==n .and. a==e .and. b==d) hmatel = hmatel + h(c,l,j,f) ! (df)(ab)(ij)(bc)
                  if (i==l .and. k==n .and. a==e .and. b==f) hmatel = hmatel + h(c,m,j,d) ! (lm)(ab)(ij)(bc)
                  if (i==l .and. k==n .and. a==d .and. b==f) hmatel = hmatel - h(c,m,j,e) ! (de)(lm)(ab)(ij)(bc)
                  if (i==l .and. k==n .and. a==e .and. b==d) hmatel = hmatel - h(c,m,j,f) ! (df)(lm)(ab)(ij)(bc)
                  if (i==m .and. k==l .and. a==e .and. b==f) hmatel = hmatel + h(c,n,j,d) ! (ln)(ab)(ij)(bc)
                  if (i==m .and. k==l .and. a==d .and. b==f) hmatel = hmatel - h(c,n,j,e) ! (de)(ln)(ab)(ij)(bc)
                  if (i==m .and. k==l .and. a==e .and. b==d) hmatel = hmatel - h(c,n,j,f) ! (df)(ln)(ab)(ij)(bc)

                  if (i==m .and. k==n .and. b==e .and. a==f) hmatel = hmatel - h(c,l,j,d) ! (ac)(ij)(bc)
                  if (i==m .and. k==n .and. b==d .and. a==f) hmatel = hmatel + h(c,l,j,e) ! (de)(ac)(ij)(bc)
                  if (i==m .and. k==n .and. b==e .and. a==d) hmatel = hmatel + h(c,l,j,f) ! (df)(ac)(ij)(bc)
                  if (i==l .and. k==n .and. b==e .and. a==f) hmatel = hmatel + h(c,m,j,d) ! (lm)(ac)(ij)(bc)
                  if (i==l .and. k==n .and. b==d .and. a==f) hmatel = hmatel - h(c,m,j,e) ! (de)(lm)(ac)(ij)(bc)
                  if (i==l .and. k==n .and. b==e .and. a==d) hmatel = hmatel - h(c,m,j,f) ! (df)(lm)(ac)(ij)(bc)
                  if (i==m .and. k==l .and. b==e .and. a==f) hmatel = hmatel + h(c,n,j,d) ! (ln)(ac)(ij)(bc)
                  if (i==m .and. k==l .and. b==d .and. a==f) hmatel = hmatel - h(c,n,j,e) ! (de)(ln)(ac)(ij)(bc)
                  if (i==m .and. k==l .and. b==e .and. a==d) hmatel = hmatel - h(c,n,j,f) ! (df)(ln)(ac)(ij)(bc)

                  if (j==m .and. i==n .and. b==e .and. c==f) hmatel = hmatel + h(a,l,k,d) ! (ik)(bc)
                  if (j==m .and. i==n .and. b==d .and. c==f) hmatel = hmatel - h(a,l,k,e) ! (de)(ik)(bc)
                  if (j==m .and. i==n .and. b==e .and. c==d) hmatel = hmatel - h(a,l,k,f) ! (df)(ik)(bc)
                  if (j==l .and. i==n .and. b==e .and. c==f) hmatel = hmatel - h(a,m,k,d) ! (lm)(ik)(bc)
                  if (j==l .and. i==n .and. b==d .and. c==f) hmatel = hmatel + h(a,m,k,e) ! (de)(lm)(ik)(bc)
                  if (j==l .and. i==n .and. b==e .and. c==d) hmatel = hmatel + h(a,m,k,f) ! (df)(lm)(ik)(bc)
                  if (j==m .and. i==l .and. b==e .and. c==f) hmatel = hmatel - h(a,n,k,d) ! (ln)(ik)(bc)
                  if (j==m .and. i==l .and. b==d .and. c==f) hmatel = hmatel + h(a,n,k,e) ! (de)(ln)(ik)(bc)
                  if (j==m .and. i==l .and. b==e .and. c==d) hmatel = hmatel + h(a,n,k,f) ! (df)(ln)(ik)(bc)

                  if (j==m .and. i==n .and. a==e .and. c==f) hmatel = hmatel - h(b,l,k,d) ! (ab)(ik)(bc)
                  if (j==m .and. i==n .and. a==d .and. c==f) hmatel = hmatel + h(b,l,k,e) ! (de)(ab)(ik)(bc)
                  if (j==m .and. i==n .and. a==e .and. c==d) hmatel = hmatel + h(b,l,k,f) ! (df)(ab)(ik)(bc)
                  if (j==l .and. i==n .and. a==e .and. c==f) hmatel = hmatel + h(b,m,k,d) ! (lm)(ab)(ik)(bc)
                  if (j==l .and. i==n .and. a==d .and. c==f) hmatel = hmatel - h(b,m,k,e) ! (de)(lm)(ab)(ik)(bc)
                  if (j==l .and. i==n .and. a==e .and. c==d) hmatel = hmatel - h(b,m,k,f) ! (df)(lm)(ab)(ik)(bc)
                  if (j==m .and. i==l .and. a==e .and. c==f) hmatel = hmatel + h(b,n,k,d) ! (ln)(ab)(ik)(bc)
                  if (j==m .and. i==l .and. a==d .and. c==f) hmatel = hmatel - h(b,n,k,e) ! (de)(ln)(ab)(ik)(bc)
                  if (j==m .and. i==l .and. a==e .and. c==d) hmatel = hmatel - h(b,n,k,f) ! (df)(ln)(ab)(ik)(bc)

                  if (j==m .and. i==n .and. b==e .and. a==f) hmatel = hmatel - h(c,l,k,d) ! (ac)(ik)(bc)
                  if (j==m .and. i==n .and. b==d .and. a==f) hmatel = hmatel + h(c,l,k,e) ! (de)(ac)(ik)(bc)
                  if (j==m .and. i==n .and. b==e .and. a==d) hmatel = hmatel + h(c,l,k,f) ! (df)(ac)(ik)(bc)
                  if (j==l .and. i==n .and. b==e .and. a==f) hmatel = hmatel + h(c,m,k,d) ! (lm)(ac)(ik)(bc)
                  if (j==l .and. i==n .and. b==d .and. a==f) hmatel = hmatel - h(c,m,k,e) ! (de)(lm)(ac)(ik)(bc)
                  if (j==l .and. i==n .and. b==e .and. a==d) hmatel = hmatel - h(c,m,k,f) ! (df)(lm)(ac)(ik)(bc)
                  if (j==m .and. i==l .and. b==e .and. a==f) hmatel = hmatel + h(c,n,k,d) ! (ln)(ac)(ik)(bc)
                  if (j==m .and. i==l .and. b==d .and. a==f) hmatel = hmatel - h(c,n,k,e) ! (de)(ln)(ac)(ik)(bc)
                  if (j==m .and. i==l .and. b==e .and. a==d) hmatel = hmatel - h(c,n,k,f) ! (df)(ln)(ac)(ik)(bc)
                  !!!
          end function aaa_voov_aaa

          pure function aaa_voov_aab(i, j, k, a, b, c, l, m, n, d, e, f, h, noa, nua, nob, nub) result(hmatel)
                  ! Expression:
                  !  A(ij)A(ab)A(k/ij)A(c/ab) d(i,l)d(j,m)d(a,d)d(b,e) h(c,n,k,f)

                  integer, intent(in) :: nua, noa, nub, nob
                  integer, intent(in) :: i, j, k, a, b, c
                  integer, intent(in) :: l, m, n, d, e, f
                  real(kind=8), intent(in) :: h(1:nua,1:nob,1:noa,1:nub)

                  real(kind=8) :: hmatel

                  hmatel = 0.0d0

                  ! (1)
          end function aaa_voov_aab

!               !!! A(d/ef) !!!
!                  if (j==m .and. k==n .and. b==e .and. c==f) hmatel = hmatel + h(a,l,i,d) ! (1)
!                  if (j==m .and. k==n .and. b==d .and. c==f) hmatel = hmatel - h(a,l,i,e) ! (de)
!                  if (j==m .and. k==n .and. b==e .and. c==d) hmatel = hmatel - h(a,l,i,f) ! (df)
!                  !!!
!
!                  !!! A(l/mn)A(d/ef) !!!
!                  if (j==m .and. k==n .and. b==e .and. c==f) hmatel = hmatel + h(a,l,i,d) ! (1)
!                  if (j==m .and. k==n .and. b==d .and. c==f) hmatel = hmatel - h(a,l,i,e) ! (de)
!                  if (j==m .and. k==n .and. b==e .and. c==d) hmatel = hmatel - h(a,l,i,f) ! (df)
!                  if (j==l .and. k==n .and. b==e .and. c==f) hmatel = hmatel - h(a,m,i,d) ! (lm)
!                  if (j==l .and. k==n .and. b==d .and. c==f) hmatel = hmatel + h(a,m,i,e) ! (de)(lm)
!                  if (j==l .and. k==n .and. b==e .and. c==d) hmatel = hmatel + h(a,m,i,f) ! (df)(lm)
!                  if (j==m .and. k==l .and. b==e .and. c==f) hmatel = hmatel - h(a,n,i,d) ! (ln)
!                  if (j==m .and. k==l .and. b==d .and. c==f) hmatel = hmatel + h(a,n,i,e) ! (de)(ln)
!                  if (j==m .and. k==l .and. b==e .and. c==d) hmatel = hmatel + h(a,n,i,f) ! (df)(ln)
!                  !!!
!
!                  !!! A(a/bc)A(l/mn)A(d/ef) !!!
!                  if (j==m .and. k==n .and. b==e .and. c==f) hmatel = hmatel + h(a,l,i,d) ! (1)
!                  if (j==m .and. k==n .and. b==d .and. c==f) hmatel = hmatel - h(a,l,i,e) ! (de)
!                  if (j==m .and. k==n .and. b==e .and. c==d) hmatel = hmatel - h(a,l,i,f) ! (df)
!                  if (j==l .and. k==n .and. b==e .and. c==f) hmatel = hmatel - h(a,m,i,d) ! (lm)
!                  if (j==l .and. k==n .and. b==d .and. c==f) hmatel = hmatel + h(a,m,i,e) ! (de)(lm)
!                  if (j==l .and. k==n .and. b==e .and. c==d) hmatel = hmatel + h(a,m,i,f) ! (df)(lm)
!                  if (j==m .and. k==l .and. b==e .and. c==f) hmatel = hmatel - h(a,n,i,d) ! (ln)
!                  if (j==m .and. k==l .and. b==d .and. c==f) hmatel = hmatel + h(a,n,i,e) ! (de)(ln)
!                  if (j==m .and. k==l .and. b==e .and. c==d) hmatel = hmatel + h(a,n,i,f) ! (df)(ln)
!                  if (j==m .and. k==n .and. a==e .and. c==f) hmatel = hmatel - h(b,l,i,d) ! (ab)
!                  if (j==m .and. k==n .and. a==d .and. c==f) hmatel = hmatel + h(b,l,i,e) ! (de)(ab)
!                  if (j==m .and. k==n .and. a==e .and. c==d) hmatel = hmatel + h(b,l,i,f) ! (df)(ab)
!                  if (j==l .and. k==n .and. a==e .and. c==f) hmatel = hmatel + h(b,m,i,d) ! (lm)(ab)
!                  if (j==l .and. k==n .and. a==d .and. c==f) hmatel = hmatel - h(b,m,i,e) ! (de)(lm)(ab)
!                  if (j==l .and. k==n .and. a==e .and. c==d) hmatel = hmatel - h(b,m,i,f) ! (df)(lm)(ab)
!                  if (j==m .and. k==l .and. a==e .and. c==f) hmatel = hmatel + h(b,n,i,d) ! (ln)(ab)
!                  if (j==m .and. k==l .and. a==d .and. c==f) hmatel = hmatel - h(b,n,i,e) ! (de)(ln)(ab)
!                  if (j==m .and. k==l .and. a==e .and. c==d) hmatel = hmatel - h(b,n,i,f) ! (df)(ln)(ab)
!                  if (j==m .and. k==n .and. b==e .and. a==f) hmatel = hmatel - h(c,l,i,d) ! (ac)
!                  if (j==m .and. k==n .and. b==d .and. a==f) hmatel = hmatel + h(c,l,i,e) ! (de)(ac)
!                  if (j==m .and. k==n .and. b==e .and. a==d) hmatel = hmatel + h(c,l,i,f) ! (df)(ac)
!                  if (j==l .and. k==n .and. b==e .and. a==f) hmatel = hmatel + h(c,m,i,d) ! (lm)(ac)
!                  if (j==l .and. k==n .and. b==d .and. a==f) hmatel = hmatel - h(c,m,i,e) ! (de)(lm)(ac)
!                  if (j==l .and. k==n .and. b==e .and. a==d) hmatel = hmatel - h(c,m,i,f) ! (df)(lm)(ac)
!                  if (j==m .and. k==l .and. b==e .and. a==f) hmatel = hmatel + h(c,n,i,d) ! (ln)(ac)
!                  if (j==m .and. k==l .and. b==d .and. a==f) hmatel = hmatel - h(c,n,i,e) ! (de)(ln)(ac)
!                  if (j==m .and. k==l .and. b==e .and. a==d) hmatel = hmatel - h(c,n,i,f) ! (df)(ln)(ac)
!                  !!!
!
!                  !!! A(i/jk)A(a/bc)A(l/mn)A(d/ef) !!!
!                  if (j==m .and. k==n .and. b==e .and. c==f) hmatel = hmatel + h(a,l,i,d) ! (1)
!                  if (j==m .and. k==n .and. b==d .and. c==f) hmatel = hmatel - h(a,l,i,e) ! (de)
!                  if (j==m .and. k==n .and. b==e .and. c==d) hmatel = hmatel - h(a,l,i,f) ! (df)
!                  if (j==l .and. k==n .and. b==e .and. c==f) hmatel = hmatel - h(a,m,i,d) ! (lm)
!                  if (j==l .and. k==n .and. b==d .and. c==f) hmatel = hmatel + h(a,m,i,e) ! (de)(lm)
!                  if (j==l .and. k==n .and. b==e .and. c==d) hmatel = hmatel + h(a,m,i,f) ! (df)(lm)
!                  if (j==m .and. k==l .and. b==e .and. c==f) hmatel = hmatel - h(a,n,i,d) ! (ln)
!                  if (j==m .and. k==l .and. b==d .and. c==f) hmatel = hmatel + h(a,n,i,e) ! (de)(ln)
!                  if (j==m .and. k==l .and. b==e .and. c==d) hmatel = hmatel + h(a,n,i,f) ! (df)(ln)
!                  if (j==m .and. k==n .and. a==e .and. c==f) hmatel = hmatel - h(b,l,i,d) ! (ab)
!                  if (j==m .and. k==n .and. a==d .and. c==f) hmatel = hmatel + h(b,l,i,e) ! (de)(ab)
!                  if (j==m .and. k==n .and. a==e .and. c==d) hmatel = hmatel + h(b,l,i,f) ! (df)(ab)
!                  if (j==l .and. k==n .and. a==e .and. c==f) hmatel = hmatel + h(b,m,i,d) ! (lm)(ab)
!                  if (j==l .and. k==n .and. a==d .and. c==f) hmatel = hmatel - h(b,m,i,e) ! (de)(lm)(ab)
!                  if (j==l .and. k==n .and. a==e .and. c==d) hmatel = hmatel - h(b,m,i,f) ! (df)(lm)(ab)
!                  if (j==m .and. k==l .and. a==e .and. c==f) hmatel = hmatel + h(b,n,i,d) ! (ln)(ab)
!                  if (j==m .and. k==l .and. a==d .and. c==f) hmatel = hmatel - h(b,n,i,e) ! (de)(ln)(ab)
!                  if (j==m .and. k==l .and. a==e .and. c==d) hmatel = hmatel - h(b,n,i,f) ! (df)(ln)(ab)
!                  if (j==m .and. k==n .and. b==e .and. a==f) hmatel = hmatel - h(c,l,i,d) ! (ac)
!                  if (j==m .and. k==n .and. b==d .and. a==f) hmatel = hmatel + h(c,l,i,e) ! (de)(ac)
!                  if (j==m .and. k==n .and. b==e .and. a==d) hmatel = hmatel + h(c,l,i,f) ! (df)(ac)
!                  if (j==l .and. k==n .and. b==e .and. a==f) hmatel = hmatel + h(c,m,i,d) ! (lm)(ac)
!                  if (j==l .and. k==n .and. b==d .and. a==f) hmatel = hmatel - h(c,m,i,e) ! (de)(lm)(ac)
!                  if (j==l .and. k==n .and. b==e .and. a==d) hmatel = hmatel - h(c,m,i,f) ! (df)(lm)(ac)
!                  if (j==m .and. k==l .and. b==e .and. a==f) hmatel = hmatel + h(c,n,i,d) ! (ln)(ac)
!                  if (j==m .and. k==l .and. b==d .and. a==f) hmatel = hmatel - h(c,n,i,e) ! (de)(ln)(ac)
!                  if (j==m .and. k==l .and. b==e .and. a==d) hmatel = hmatel - h(c,n,i,f) ! (df)(ln)(ac)
!                  if (i==m .and. k==n .and. b==e .and. c==f) hmatel = hmatel - h(a,l,j,d) ! (ij)
!                  if (i==m .and. k==n .and. b==d .and. c==f) hmatel = hmatel + h(a,l,j,e) ! (de)(ij)
!                  if (i==m .and. k==n .and. b==e .and. c==d) hmatel = hmatel + h(a,l,j,f) ! (df)(ij)
!                  if (i==l .and. k==n .and. b==e .and. c==f) hmatel = hmatel + h(a,m,j,d) ! (lm)(ij)
!                  if (i==l .and. k==n .and. b==d .and. c==f) hmatel = hmatel - h(a,m,j,e) ! (de)(lm)(ij)
!                  if (i==l .and. k==n .and. b==e .and. c==d) hmatel = hmatel - h(a,m,j,f) ! (df)(lm)(ij)
!                  if (i==m .and. k==l .and. b==e .and. c==f) hmatel = hmatel + h(a,n,j,d) ! (ln)(ij)
!                  if (i==m .and. k==l .and. b==d .and. c==f) hmatel = hmatel - h(a,n,j,e) ! (de)(ln)(ij)
!                  if (i==m .and. k==l .and. b==e .and. c==d) hmatel = hmatel - h(a,n,j,f) ! (df)(ln)(ij)
!                  if (i==m .and. k==n .and. a==e .and. c==f) hmatel = hmatel + h(b,l,j,d) ! (ab)(ij)
!                  if (i==m .and. k==n .and. a==d .and. c==f) hmatel = hmatel - h(b,l,j,e) ! (de)(ab)(ij)
!                  if (i==m .and. k==n .and. a==e .and. c==d) hmatel = hmatel - h(b,l,j,f) ! (df)(ab)(ij)
!                  if (i==l .and. k==n .and. a==e .and. c==f) hmatel = hmatel - h(b,m,j,d) ! (lm)(ab)(ij)
!                  if (i==l .and. k==n .and. a==d .and. c==f) hmatel = hmatel + h(b,m,j,e) ! (de)(lm)(ab)(ij)
!                  if (i==l .and. k==n .and. a==e .and. c==d) hmatel = hmatel + h(b,m,j,f) ! (df)(lm)(ab)(ij)
!                  if (i==m .and. k==l .and. a==e .and. c==f) hmatel = hmatel - h(b,n,j,d) ! (ln)(ab)(ij)
!                  if (i==m .and. k==l .and. a==d .and. c==f) hmatel = hmatel + h(b,n,j,e) ! (de)(ln)(ab)(ij)
!                  if (i==m .and. k==l .and. a==e .and. c==d) hmatel = hmatel + h(b,n,j,f) ! (df)(ln)(ab)(ij)
!                  if (i==m .and. k==n .and. b==e .and. a==f) hmatel = hmatel + h(c,l,j,d) ! (ac)(ij)
!                  if (i==m .and. k==n .and. b==d .and. a==f) hmatel = hmatel - h(c,l,j,e) ! (de)(ac)(ij)
!                  if (i==m .and. k==n .and. b==e .and. a==d) hmatel = hmatel - h(c,l,j,f) ! (df)(ac)(ij)
!                  if (i==l .and. k==n .and. b==e .and. a==f) hmatel = hmatel - h(c,m,j,d) ! (lm)(ac)(ij)
!                  if (i==l .and. k==n .and. b==d .and. a==f) hmatel = hmatel + h(c,m,j,e) ! (de)(lm)(ac)(ij)
!                  if (i==l .and. k==n .and. b==e .and. a==d) hmatel = hmatel + h(c,m,j,f) ! (df)(lm)(ac)(ij)
!                  if (i==m .and. k==l .and. b==e .and. a==f) hmatel = hmatel - h(c,n,j,d) ! (ln)(ac)(ij)
!                  if (i==m .and. k==l .and. b==d .and. a==f) hmatel = hmatel + h(c,n,j,e) ! (de)(ln)(ac)(ij)
!                  if (i==m .and. k==l .and. b==e .and. a==d) hmatel = hmatel + h(c,n,j,f) ! (df)(ln)(ac)(ij)
!                  if (j==m .and. i==n .and. b==e .and. c==f) hmatel = hmatel - h(a,l,k,d) ! (ik)
!                  if (j==m .and. i==n .and. b==d .and. c==f) hmatel = hmatel + h(a,l,k,e) ! (de)(ik)
!                  if (j==m .and. i==n .and. b==e .and. c==d) hmatel = hmatel + h(a,l,k,f) ! (df)(ik)
!                  if (j==l .and. i==n .and. b==e .and. c==f) hmatel = hmatel + h(a,m,k,d) ! (lm)(ik)
!                  if (j==l .and. i==n .and. b==d .and. c==f) hmatel = hmatel - h(a,m,k,e) ! (de)(lm)(ik)
!                  if (j==l .and. i==n .and. b==e .and. c==d) hmatel = hmatel - h(a,m,k,f) ! (df)(lm)(ik)
!                  if (j==m .and. i==l .and. b==e .and. c==f) hmatel = hmatel + h(a,n,k,d) ! (ln)(ik)
!                  if (j==m .and. i==l .and. b==d .and. c==f) hmatel = hmatel - h(a,n,k,e) ! (de)(ln)(ik)
!                  if (j==m .and. i==l .and. b==e .and. c==d) hmatel = hmatel - h(a,n,k,f) ! (df)(ln)(ik)
!                  if (j==m .and. i==n .and. a==e .and. c==f) hmatel = hmatel + h(b,l,k,d) ! (ab)(ik)
!                  if (j==m .and. i==n .and. a==d .and. c==f) hmatel = hmatel - h(b,l,k,e) ! (de)(ab)(ik)
!                  if (j==m .and. i==n .and. a==e .and. c==d) hmatel = hmatel - h(b,l,k,f) ! (df)(ab)(ik)
!                  if (j==l .and. i==n .and. a==e .and. c==f) hmatel = hmatel - h(b,m,k,d) ! (lm)(ab)(ik)
!                  if (j==l .and. i==n .and. a==d .and. c==f) hmatel = hmatel + h(b,m,k,e) ! (de)(lm)(ab)(ik)
!                  if (j==l .and. i==n .and. a==e .and. c==d) hmatel = hmatel + h(b,m,k,f) ! (df)(lm)(ab)(ik)
!                  if (j==m .and. i==l .and. a==e .and. c==f) hmatel = hmatel - h(b,n,k,d) ! (ln)(ab)(ik)
!                  if (j==m .and. i==l .and. a==d .and. c==f) hmatel = hmatel + h(b,n,k,e) ! (de)(ln)(ab)(ik)
!                  if (j==m .and. i==l .and. a==e .and. c==d) hmatel = hmatel + h(b,n,k,f) ! (df)(ln)(ab)(ik)
!                  if (j==m .and. i==n .and. b==e .and. a==f) hmatel = hmatel + h(c,l,k,d) ! (ac)(ik)
!                  if (j==m .and. i==n .and. b==d .and. a==f) hmatel = hmatel - h(c,l,k,e) ! (de)(ac)(ik)
!                  if (j==m .and. i==n .and. b==e .and. a==d) hmatel = hmatel - h(c,l,k,f) ! (df)(ac)(ik)
!                  if (j==l .and. i==n .and. b==e .and. a==f) hmatel = hmatel - h(c,m,k,d) ! (lm)(ac)(ik)
!                  if (j==l .and. i==n .and. b==d .and. a==f) hmatel = hmatel + h(c,m,k,e) ! (de)(lm)(ac)(ik)
!                  if (j==l .and. i==n .and. b==e .and. a==d) hmatel = hmatel + h(c,m,k,f) ! (df)(lm)(ac)(ik)
!                  if (j==m .and. i==l .and. b==e .and. a==f) hmatel = hmatel - h(c,n,k,d) ! (ln)(ac)(ik)
!                  if (j==m .and. i==l .and. b==d .and. a==f) hmatel = hmatel + h(c,n,k,e) ! (de)(ln)(ac)(ik)
!                  if (j==m .and. i==l .and. b==e .and. a==d) hmatel = hmatel + h(c,n,k,f) ! (df)(ln)(ac)(ik)
!                  !!!


end module hbar_matrix_elements
