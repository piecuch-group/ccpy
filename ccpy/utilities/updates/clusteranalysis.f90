module clusteranalysis

    implicit none

    contains

        subroutine cluster_analysis_t2(t2a, t2b, t2c,&
                                       c1a, c1b, c2a, c2b, c2c,&
                                       noa, nua, nob, nub)

                    integer, intent(in) :: noa, nua, nob, nub
                    real(kind=8), intent(in) :: c1a(nua, noa), c1b(nub, nob),&
                                                c2a(nua, nua, noa, noa), c2b(nua, nub, noa, nob), c2c(nub, nub, nob, nob)

                    real(kind=8), intent(out) :: t2a(nua, nua, noa, noa), t2b(nua, nub, noa, nob), t2c(nub, nub, nob, nob)

                    integer :: a, b, i, j

                    t2a = 0.0d0
                    do a = 1 , nua
                        do b = a + 1 , nua
                            do i = 1 , noa
                                do j = i + 1, noa
                                    if ( abs(c2a(a, b, i, j)) > 0.0d0 ) then
                                        t2a(a, b, i, j) = c2a(a, b, i, j)&
                                                         -c1a(a, i) * c1a(b, j)&
                                                         +c1a(b, i) * c1a(a, j)&
                                                         +c1a(a, j) * c1a(b, i)&
                                                         -c1a(b, j) * c1a(a, i)
                                    else
                                        t2a(a, b, i, j) = 0.0d0
                                    end if
                                    t2a(b, a, i, j) = -1.0 * t2a(a, b, i, j)
                                    t2a(a, b, j, i) = -1.0 * t2a(a, b, i, j)
                                    t2a(b, a, j, i) = t2a(a, b, i, j)
                                end do
                            end do
                        end do
                    end do

                    t2b = 0.0d0
                    do a = 1 , nua
                        do b = 1 , nub
                            do i = 1 , noa
                                do j = 1, nob
                                    if ( abs(c2b(a, b, i, j)) > 0.0d0 ) then
                                        t2b(a, b, i, j) = c2b(a, b, i, j)&
                                                         -c1a(a, i) * c1b(b, j)&
                                    else
                                        t2b(a, b, i, j) = 0.0d0
                                    end if
                                end do
                            end do
                        end do
                    end do

                    t2c = 0.0d0
                    do a = 1 , nub
                        do b = a + 1 , nub
                            do i = 1 , nob
                                do j = i + 1, nob
                                    if ( abs(c2c(a, b, i, j)) > 0.0d0 ) then
                                        t2c(a, b, i, j) = c2c(a, b, i, j)&
                                                         -c1b(a, i) * c1b(b, j)&
                                                         +c1b(b, i) * c1b(a, j)&
                                                         +c1b(a, j) * c1b(b, i)&
                                                         -c1b(b, j) * c1b(a, i)
                                    else
                                        t2c(a, b, i, j) = 0.0d0
                                    end if
                                    t2c(b, a, i, j) = -1.0 * t2c(a, b, i, j)
                                    t2c(a, b, j, i) = -1.0 * t2c(a, b, i, j)
                                    t2c(b, a, j, i) = t2c(a, b, i, j)
                                end do
                            end do
                        end do
                    end do

        end subroutine cluster_analysis_t2

        subroutine cluster_analysis_t3(t3a, t3b, t3c, t3d,&
                                       c1a, c1b, c2a, c2b, c2c,&
                                       c3a, c3b, c3c, c3d,&
                                       noa, nua, nob, nub)

                integer, intent(in) :: noa, nua, nob, nub
                real(kind=8), intent(in) :: c1a(nua, noa), c1b(nub, nob),&
                                            c2a(nua, nua, noa, noa), c2b(nua, nub, noa, nob), c2c(nub, nub, nob, nob),&
                                            c3a(nua, nua, nua, noa, noa, noa), c3b(nua, nua, nub, noa, noa, nob),&
                                            c3c(nua, nub, nub, noa, nob, nob), c3d(nub, nub, nub, nob, nob, nob)

                real(kind=8), intent(out) :: t3a(nua, nua, nua, noa, noa, noa),&
                                             t3b(nua, nua, nub, noa, noa, nob),&
                                             t3c(nua, nub, nub, noa, nob, nob),&
                                             t3d(nub, nub, nub, nob, nob, nob)

                integer :: a, b, c, i, j, k

                t3a = 0.0d0
                do a = 1 , nua
                    do b = a + 1 , nua
                        do c = b + 1 , nua
                            do i = 1 , noa
                                do j = i + 1, noa
                                    do k = j + 1, noa
                                        if (abs(c3a(a, b, c, i, j, k)) > 0.0d0) then
                                            t3a(a, b, c, i, j, k) = c3a(a, b, c, i, j, k)&
                                                                   -c1a(a, i) * c2a(b, c, j, k)&
                                                                   +c1a(b, i) * c2a(a, c, j, k)&
                                                                   +c1a(c, i) * c2a(b, a, j, k)&
                                                                   -c1a(b, j) * c2a(a, c, i, k)&
                                                                   +c1a(a, j) * c2a(b, c, i, k)&
                                                                   +c1a(c, j) * c2a(b, a, i, k)&
                                                                   -c1a(c, k) * c2a(b, a, j, i)&
                                                                   +c1a(a, k) * c2a(b, c, j, i)&
                                                                   +c1a(b, k) * c2a(c, a, j, i)&
                                                                   +2.0d0 * c1a(a, i) * c1a(b, j) * c1a(c, k)&
                                                                   +2.0d0 * c1a(a, j) * c1a(b, k) * c1a(c, i)&
                                                                   +2.0d0 * c1a(a, k) * c1a(b, i) * c1a(c, j)&
                                                                   -2.0d0 * c1a(a, i) * c1a(b, k) * c1a(c, j)&
                                                                   -2.0d0 * c1a(a, j) * c1a(b, i) * c1a(c, k)&
                                                                   -2.0d0 * c1a(a, k) * c1a(b, j) * c1a(c, i)
                                        else
                                            t3a(a, b, c, i, j, k) = 0.0d0
                                        end if
                                        ! antisymmetrize t3a
                                        t3a(A,B,C,K,I,J) = t3a(A,B,C,I,J,K)
                                        t3a(A,B,C,J,K,I) = t3a(A,B,C,I,J,K)
                                        t3a(A,B,C,I,K,J) = -t3a(A,B,C,I,J,K)
                                        t3a(A,B,C,J,I,K) = -t3a(A,B,C,I,J,K)
                                        t3a(A,B,C,K,J,I) = -t3a(A,B,C,I,J,K)

                                        t3a(B,A,C,I,J,K) = -t3a(A,B,C,I,J,K)
                                        t3a(B,A,C,K,I,J) = -t3a(A,B,C,I,J,K)
                                        t3a(B,A,C,J,K,I) = -t3a(A,B,C,I,J,K)
                                        t3a(B,A,C,I,K,J) = t3a(A,B,C,I,J,K)
                                        t3a(B,A,C,J,I,K) = t3a(A,B,C,I,J,K)
                                        t3a(B,A,C,K,J,I) = t3a(A,B,C,I,J,K)

                                        t3a(A,C,B,I,J,K) = -t3a(A,B,C,I,J,K)
                                        t3a(A,C,B,K,I,J) = -t3a(A,B,C,I,J,K)
                                        t3a(A,C,B,J,K,I) = -t3a(A,B,C,I,J,K)
                                        t3a(A,C,B,I,K,J) = t3a(A,B,C,I,J,K)
                                        t3a(A,C,B,J,I,K) = t3a(A,B,C,I,J,K)
                                        t3a(A,C,B,K,J,I) = t3a(A,B,C,I,J,K)

                                        t3a(C,B,A,I,J,K) = -t3a(A,B,C,I,J,K)
                                        t3a(C,B,A,K,I,J) = -t3a(A,B,C,I,J,K)
                                        t3a(C,B,A,J,K,I) = -t3a(A,B,C,I,J,K)
                                        t3a(C,B,A,I,K,J) = t3a(A,B,C,I,J,K)
                                        t3a(C,B,A,J,I,K) = t3a(A,B,C,I,J,K)
                                        t3a(C,B,A,K,J,I) = t3a(A,B,C,I,J,K)

                                        t3a(B,C,A,I,J,K) = t3a(A,B,C,I,J,K)
                                        t3a(B,C,A,K,I,J) = t3a(A,B,C,I,J,K)
                                        t3a(B,C,A,J,K,I) = t3a(A,B,C,I,J,K)
                                        t3a(B,C,A,I,K,J) = -t3a(A,B,C,I,J,K)
                                        t3a(B,C,A,J,I,K) = -t3a(A,B,C,I,J,K)
                                        t3a(B,C,A,K,J,I) = -t3a(A,B,C,I,J,K)

                                        t3a(C,A,B,I,J,K) = t3a(A,B,C,I,J,K)
                                        t3a(C,A,B,K,I,J) = t3a(A,B,C,I,J,K)
                                        t3a(C,A,B,J,K,I) = t3a(A,B,C,I,J,K)
                                        t3a(C,A,B,I,K,J) = -t3a(A,B,C,I,J,K)
                                        t3a(C,A,B,J,I,K) = -t3a(A,B,C,I,J,K)
                                        t3a(C,A,B,K,J,I) = -t3a(A,B,C,I,J,K)
                                    end do
                                end do
                            end do
                        end do
                    end do
                end do

                t3b = 0.0d0
                do a = 1 , nua
                    do b = a + 1 , nua
                        do c = 1 , nub
                            do i = 1 , noa
                                do j = i + 1, noa
                                    do k = 1, nob
                                        if (abs(c3b(a, b, c, i, j, k)) > 0.0d0) then
                                            t3b(a, b, c, i, j, k) = c3b(a, b, c, i, j, k)&
                                                                   -c1a(a, i) * c2b(b, c, j, k)&
                                                                   +c1a(b, i) * c2b(a, c, j, k)&
                                                                   +c1a(a, j) * c2b(a, c, i, k)&
                                                                   -c1a(b, j) * c2b(a, c, i, k)&
                                                                   -c1b(c, k) * c2a(a, b, i, j)&
                                                                   +2.0d0 * c1a(a, i) * c1a(b, j) * c1b(c, k)&
                                                                   -2.0d0 * c1a(a, j) * c1a(b, i) * c1b(c, k)
                                        else
                                            t3b(a, b, c, i, j, k) = 0.0d0
                                        end if
                                        ! antisymmetrize t3b
                                        t3b(b, a, c, i, j, k) = -1.0 * t3b(a, b, c, i, j, k)
                                        t3b(a, b, c, j, i, k) = -1.0 * t3b(a, b, c, i, j, k)
                                        t3b(b, a, c, j, i, k) = t3b(a, b, c, i, j, k)
                                    end do
                                end do
                            end do
                        end do
                    end do
                end do

                t3c = 0.0d0
                do a = 1 , nua
                    do b = 1 , nub
                        do c = b + 1 , nub
                            do i = 1 , noa
                                do j = 1, nob
                                    do k = j + 1, nob
                                        if (abs(c3c(a, b, c, i, j, k)) > 0.0d0) then
                                            t3c(a, b, c, i, j, k) = c3c(a, b, c, i, j, k)&
                                                                   -c1a(a, i) * c2c(b, c, j, k)&
                                                                   -c1b(b, j) * c2b(a, c, i, k)&
                                                                   +c1b(c, j) * c2b(a, b, i, k)&
                                                                   +c1b(b, k) * c2b(a, c, i, j)&
                                                                   -c1b(c, k) * c2b(a, b, i, j)&
                                                                   +2.0d0 * c1b(c, k) * c1b(b, j) * c1a(a, i)&
                                                                   -2.0d0 * c1b(c, j) * c1b(b, k) * c1a(a, i)
                                        else
                                            t3c(a, b, c, i, j, k) = 0.0d0
                                        end if
                                        ! antisymmetrize t3c
                                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)
                                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)
                                        t3c(a, b, c, i, j, k) = t3c(a, b, c, i, j, k)
                                    end do
                                end do
                            end do
                        end do
                    end do
                end do

                t3d = 0.0d0
                do a = 1 , nub
                    do b = a + 1 , nub
                        do c = b + 1 , nub
                            do i = 1 , nob
                                do j = i + 1, nob
                                    do k = j + 1, nob
                                        if (abs(c3d(a, b, c, i, j, k)) > 0.0d0) then
                                            t3d(a, b, c, i, j, k) = c3d(a, b, c, i, j, k)&
                                                                   -c1b(a, i) * c2c(b, c, j, k)&
                                                                   +c1b(b, i) * c2c(a, c, j, k)&
                                                                   +c1b(c, i) * c2c(b, a, j, k)&
                                                                   -c1b(b, j) * c2c(a, c, i, k)&
                                                                   +c1b(a, j) * c2c(b, c, i, k)&
                                                                   +c1b(c, j) * c2c(b, a, i, k)&
                                                                   -c1b(c, k) * c2c(b, a, j, i)&
                                                                   +c1b(a, k) * c2c(b, c, j, i)&
                                                                   +c1b(b, k) * c2c(c, a, j, i)&
                                                                   +2.0d0 * c1b(a, i) * c1b(b, j) * c1b(c, k)&
                                                                   +2.0d0 * c1b(a, j) * c1b(b, k) * c1b(c, i)&
                                                                   +2.0d0 * c1b(a, k) * c1b(b, i) * c1b(c, j)&
                                                                   -2.0d0 * c1b(a, i) * c1b(b, k) * c1b(c, j)&
                                                                   -2.0d0 * c1b(a, j) * c1b(b, i) * c1b(c, k)&
                                                                   -2.0d0 * c1b(a, k) * c1b(b, j) * c1b(c, i)
                                        else
                                            t3d(a, b, c, i, j, k) = 0.0d0
                                        end if
                                        ! antisymmetrize t3a
                                        t3d(A,B,C,K,I,J) = t3d(A,B,C,I,J,K)
                                        t3d(A,B,C,J,K,I) = t3d(A,B,C,I,J,K)
                                        t3d(A,B,C,I,K,J) = -t3d(A,B,C,I,J,K)
                                        t3d(A,B,C,J,I,K) = -t3d(A,B,C,I,J,K)
                                        t3d(A,B,C,K,J,I) = -t3d(A,B,C,I,J,K)

                                        t3d(B,A,C,I,J,K) = -t3d(A,B,C,I,J,K)
                                        t3d(B,A,C,K,I,J) = -t3d(A,B,C,I,J,K)
                                        t3d(B,A,C,J,K,I) = -t3d(A,B,C,I,J,K)
                                        t3d(B,A,C,I,K,J) = t3d(A,B,C,I,J,K)
                                        t3d(B,A,C,J,I,K) = t3d(A,B,C,I,J,K)
                                        t3d(B,A,C,K,J,I) = t3d(A,B,C,I,J,K)

                                        t3d(A,C,B,I,J,K) = -t3d(A,B,C,I,J,K)
                                        t3d(A,C,B,K,I,J) = -t3d(A,B,C,I,J,K)
                                        t3d(A,C,B,J,K,I) = -t3d(A,B,C,I,J,K)
                                        t3d(A,C,B,I,K,J) = t3d(A,B,C,I,J,K)
                                        t3d(A,C,B,J,I,K) = t3d(A,B,C,I,J,K)
                                        t3d(A,C,B,K,J,I) = t3d(A,B,C,I,J,K)

                                        t3d(C,B,A,I,J,K) = -t3d(A,B,C,I,J,K)
                                        t3d(C,B,A,K,I,J) = -t3d(A,B,C,I,J,K)
                                        t3d(C,B,A,J,K,I) = -t3d(A,B,C,I,J,K)
                                        t3d(C,B,A,I,K,J) = t3d(A,B,C,I,J,K)
                                        t3d(C,B,A,J,I,K) = t3d(A,B,C,I,J,K)
                                        t3d(C,B,A,K,J,I) = t3d(A,B,C,I,J,K)

                                        t3d(B,C,A,I,J,K) = t3d(A,B,C,I,J,K)
                                        t3d(B,C,A,K,I,J) = t3d(A,B,C,I,J,K)
                                        t3d(B,C,A,J,K,I) = t3d(A,B,C,I,J,K)
                                        t3d(B,C,A,I,K,J) = -t3d(A,B,C,I,J,K)
                                        t3d(B,C,A,J,I,K) = -t3d(A,B,C,I,J,K)
                                        t3d(B,C,A,K,J,I) = -t3d(A,B,C,I,J,K)

                                        t3d(C,A,B,I,J,K) = t3d(A,B,C,I,J,K)
                                        t3d(C,A,B,K,I,J) = t3d(A,B,C,I,J,K)
                                        t3d(C,A,B,J,K,I) = t3d(A,B,C,I,J,K)
                                        t3d(C,A,B,I,K,J) = -t3d(A,B,C,I,J,K)
                                        t3d(C,A,B,J,I,K) = -t3d(A,B,C,I,J,K)
                                        t3d(C,A,B,K,J,I) = -t3d(A,B,C,I,J,K)
                                    end do
                                end do
                            end do
                        end do
                    end do
                end do

        end subroutine cluster_analysis_t3

        subroutine cluster_analyis_t4_opt_projection(x2a, x2b, x2c,&
                                                     v_aa, v_ab, v_bb,&
                                                     list_aaaa, list_aaab, list_aabb, list_abbb, list_bbbb,&
                                                     c4a, c4b, c4c, c4d, c4e,&
                                                     c1a, c1b, c2a, c2b, c2c, c3a, c3b, c3c, c3d,&
                                                     noa, nua, nob, nub)

                integer, intent(in) :: noa, nua, nob, nub
                integer, intent(in) :: list_aaaa(:, 8), list_aaab(:, 8), list_aabb(:, 8), list_abbb(:, 8), list_bbbb(:, 8)
                real(kind=8), intent(in) :: v_aa(noa, noa, nua, nua), v_ab(noa, nob, nua, nub), v_bb(nob, nob, nub, nub)
                real(kind=8), intent(in) :: c4a(:), c4b(:), c4c(:), c4d(:), c4e(:)
                real(kind=8), intent(in) :: c1a(nua, noa), c1b(nub, nob),&
                                            c2a(nua, nua, noa, noa), c2b(nua, nub, noa, nob), c2c(nub, nub, nob, nob),&
                                            c3a(nua, nua, nua, noa, noa, noa),&
                                            c3b(nua, nua, nub, noa, noa, nob),&
                                            c3c(nua, nub, nub, noa, nob, nob),&
                                            c3d(nub, nub, nub, nob, nob, nob)

                real(kind=8), intent(out) :: x2a(nua, nua, noa, noa), x2b(nua, nub, noa, nob), x2c(nub, nub, nob, nob)

                integer :: i, j, a, b, k, l, m, n, c, d, e, f, idx
                logical :: is_double
                real(kind=8) :: v_matel, c4, c1c3, c22, c14

                ! Calculate < ijab | (V_N * T4)_C | 0 >
                do i = 1 , noa
                    do j = i + 1, noa
                        do a = 1 , nua
                            do b = a + 1 , nua

                                ! loop over C4_aaaa amplitudes in CIPSI
                                do idx = 1, size(list_aaaa)

                                    k = list_aaaa(idx, 1)
                                    l = list_aaaa(idx, 2)
                                    m = list_aaaa(idx, 3)
                                    n = list_aaaa(idx, 4)
                                    c = list_aaaa(idx, 5)
                                    d = list_aaaa(idx, 6)
                                    e = list_aaaa(idx, 7)
                                    f = list_aaaa(idx, 8)

                                    ! check if double excitation
                                    if (is_double)
                                        ! < ijab | V_N | klmncdef > = A(ij)A(ab)A(mn/ij)A(ef/ab)[δ(i,k)*δ(j,l)*δ(a,c)*δ(b,d)*v_aa(m,n,e,f)]
                                        v_matel = delta(i, k) * delta(j, l) * delta(a, c) * delta(b, d) * v_aa(m, n, e, f)&
                                    end if

                                end do

                            end do
                        end do
                    end do
                end do

        end subroutine cluster_analyis_t4_opt_projection

        function delta(i, j) result(x)

            integer, intent(in) :: i, j
            real(kind=8) :: x

            x = 0.0d0
            if (i == j) then
                x = 1.0d0
            end if

        end function delta

end module clusteranalysis