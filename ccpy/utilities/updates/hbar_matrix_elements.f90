module hbar_matrix_elements

      implicit none

      contains


              pure function aaa_H_aaa_oo(i, j, k, a, b, c, l, m, n, d, e, f, H, noa) result(hmatel)

                      ! Expression:
                      ! -A(abc)A(jk)A(l/mn)A(i/jk) d(ad)d(be)d(cf)d(jm)d(kn) h(l,i)
                      
                      integer, intent(in) :: noa
                      integer, intent(in) :: i, j, k, a, b, c
                      integer, intent(in) :: l, m, n, d, e, f
                      real(kind=8), intent(in) :: H(1:noa,1:noa)

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

              end function aaa_H_aaa_oo




end module hbar_matrix_elements
