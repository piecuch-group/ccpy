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
                    do a = 1, nua
                        do b = a + 1, nua
                            do i = 1 , noa
                                do j = i + 1, noa
                                    if ( abs(c2a(a, b, i, j)) > 0.0d0 ) then
                                        t2a(a, b, i, j) = c2a(a, b, i, j)&
                                                         -c1a(a, i) * c1a(b, j)&
                                                         +c1a(a, j) * c1a(b, i)
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
                    do a = 1, nua
                        do b = 1, nub
                            do i = 1, noa
                                do j = 1, nob
                                    if ( abs(c2b(a, b, i, j)) > 0.0d0 ) then
                                        t2b(a, b, i, j) = c2b(a, b, i, j)&
                                                         -c1a(a, i) * c1b(b, j)
                                    else
                                        t2b(a, b, i, j) = 0.0d0
                                    end if
                                end do
                            end do
                        end do
                    end do

                    t2c = 0.0d0
                    do a = 1, nub
                        do b = a + 1, nub
                            do i = 1, nob
                                do j = i + 1, nob
                                    if ( abs(c2c(a, b, i, j)) > 0.0d0 ) then
                                        t2c(a, b, i, j) = c2c(a, b, i, j)&
                                                         -c1b(a, i) * c1b(b, j)&
                                                         +c1b(a, j) * c1b(b, i)
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
                                            t3a(a,b,c,i,j,k)=(c3a(a,b,c,i,j,k) &
                                                -c1a(a,i)*c2a(b,c,j,k) &
                                                +c1a(a,j)*c2a(b,c,i,k) &
                                                +c1a(a,k)*c2a(b,c,j,i) &
                                                +c1a(b,i)*c2a(a,c,j,k) &
                                                +c1a(c,i)*c2a(b,a,j,k) &
                                                -c1a(b,j)*c2a(a,c,i,k) &
                                                -c1a(c,j)*c2a(b,a,i,k) &
                                                -c1a(b,k)*c2a(a,c,j,i) &
                                                -c1a(c,k)*c2a(b,a,j,i) &
                                                +2.0d0*(c1a(a,i)*c1a(b,j)*c1a(c,k) &
                                                -c1a(a,j)*c1a(b,i)*c1a(c,k) &
                                                -c1a(a,k)*c1a(b,j)*c1a(c,i) &
                                                -c1a(a,i)*c1a(b,k)*c1a(c,j) &
                                                +c1a(a,j)*c1a(b,k)*c1a(c,i) &
                                                +c1a(a,k)*c1a(b,i)*c1a(c,j)))
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
                                            t3b(a,b,c,i,j,k)=(c3b(a,b,c,i,j,k) &
                                                -c1a(a,i)*c2b(b,c,j,k) &
                                                +c1a(a,j)*c2b(b,c,i,k) &
                                                +c1a(b,i)*c2b(a,c,j,k) &
                                                -c1a(b,j)*c2b(a,c,i,k) &
                                                -c1b(c,k)*c2a(a,b,i,j) &
                                                +2.0d0*(c1a(a,i)*c1a(b,j)*c1b(c,k) &
                                                -c1a(a,j)*c1a(b,i)*c1b(c,k)))
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
                        do c = b + 1, nub
                            do i = 1, noa
                                do j = 1, nob
                                    do k = j + 1, nob
                                        if (abs(c3c(a, b, c, i, j, k)) > 0.0d0) then
                                            t3c(a,b,c,i,j,k)=(c3c(a,b,c,i,j,k) &
                                                -c1a(a,i)*c2c(b,c,j,k) &
                                                -c1b(c,k)*c2b(a,b,i,j) &
                                                +c1b(c,j)*c2b(a,b,i,k) &
                                                +c1b(b,k)*c2b(a,c,i,j) &
                                                -c1b(b,j)*c2b(a,c,i,k) &
                                                +2.0d0*(c1a(a,i)*c1b(b,j)*c1b(c,k) &
                                                -c1a(a,i)*c1b(b,k)*c1b(c,j)))
                                        else
                                            t3c(a, b, c, i, j, k) = 0.0d0
                                        end if
                                        ! antisymmetrize t3c
                                        t3c(a, c, b, i, j, k) = -1.0 * t3c(a, b, c, i, j, k)
                                        t3c(a, b, c, i, k, j) = -1.0 * t3c(a, b, c, i, j, k)
                                        t3c(a, c, b, i, k, j) = t3c(a, b, c, i, j, k)
                                    end do
                                end do
                            end do
                        end do
                    end do
                end do

                t3d = 0.0d0
                do a = 1, nub
                    do b = a + 1, nub
                        do c = b + 1, nub
                            do i = 1, nob
                                do j = i + 1, nob
                                    do k = j + 1, nob
                                        if (abs(c3d(a, b, c, i, j, k)) > 0.0d0) then
                                            t3d(a,b,c,i,j,k)=(c3d(a,b,c,i,j,k) &
                                                -c1b(a,i)*c2c(b,c,j,k) &
                                                +c1b(a,j)*c2c(b,c,i,k) &
                                                +c1b(a,k)*c2c(b,c,j,i) &
                                                +c1b(b,i)*c2c(a,c,j,k) &
                                                +c1b(c,i)*c2c(b,a,j,k) &
                                                -c1b(b,j)*c2c(a,c,i,k) &
                                                -c1b(c,j)*c2c(b,a,i,k) &
                                                -c1b(b,k)*c2c(a,c,j,i) &
                                                -c1b(c,k)*c2c(b,a,j,i) &
                                                +2.0d0*(c1b(a,i)*c1b(b,j)*c1b(c,k) &
                                                -c1b(a,j)*c1b(b,i)*c1b(c,k) &
                                                -c1b(a,k)*c1b(b,j)*c1b(c,i) &
                                                -c1b(a,i)*c1b(b,k)*c1b(c,j) &
                                                +c1b(a,j)*c1b(b,k)*c1b(c,i) &
                                                +c1b(a,k)*c1b(b,i)*c1b(c,j)))
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


        subroutine contract_vt4_opt(x2a, x2b, x2c,&
                                    v_aa, v_ab, v_bb,&
                                    c1_a, c1_b, c2_aa, c2_ab, c2_bb, c3_aaa, c3_aab, c3_abb, c3_bbb,&
                                    c4_aaaa, c4_aaab, c4_aabb, c4_abbb, c4_bbbb,&
                                    c4_aaaa_amps, c4_aaab_amps, c4_aabb_amps, c4_abbb_amps, c4_bbbb_amps,&
                                    n4a, n4b, n4c, n4d, n4e,&
                                    noa, nua, nob, nub)

            integer, intent(in) :: noa, nua, nob, nub
            integer, intent(in) :: n4a, n4b, n4c, n4d, n4e

            integer, intent(in) :: c4_aaaa(n4a, 8), c4_aaab(n4b, 8), c4_aabb(n4c, 8), c4_abbb(n4d, 8), c4_bbbb(n4e, 8)

            real(kind=8), intent(in) :: c1_a(nua, noa), c1_b(nub, nob),&
                                        c2_aa(nua, nua, noa, noa), c2_ab(nua, nub, noa, nob), c2_bb(nub, nub, nob, nob),&
                                        c3_aaa(nua, nua, nua, noa, noa, noa), c3_aab(nua, nua, nub, noa, noa, nob),&
                                        c3_abb(nua, nub, nub, noa, nob, nob), c3_bbb(nub, nub, nub, nob, nob, nob)

            real(kind=8), intent(in) :: c4_aaaa_amps(n4a), c4_aaab_amps(n4b), c4_aabb_amps(n4c), c4_abbb_amps(n4d), c4_bbbb_amps(n4e)
            real(kind=8), intent(in) :: v_aa(noa, noa, nua, nua), v_ab(noa, nob, nua, nub), v_bb(nob, nob, nub, nub)

            real(kind=8), intent(out) :: x2a(nua, nua, noa, noa), x2b(nua, nub, noa, nob), x2c(nub, nub, nob, nob)

            integer :: a, b, i, j, m, n, e, f, idet
            real(kind=8) :: hmatel, t_amp

            x2a = 0.0d0
            x2b = 0.0d0
            x2c = 0.0d0

            do idet = 1, n4a

                a = c4_aaaa(idet, 1); b = c4_aaaa(idet, 2); e = c4_aaaa(idet, 3); f = c4_aaaa(idet, 4);
                i = c4_aaaa(idet, 5); j = c4_aaaa(idet, 6); m = c4_aaaa(idet, 7); n = c4_aaaa(idet, 8);

                call get_t4aaaa_amp(t_amp, c1_a, c2_aa, c3_aaa, c4_aaaa_amps(idet),&
                                    a, b, e, f, i, j, m, n,&
                                    noa, nua)

                ! x2a(abij) <- A(ij/mn)A(ab/ef) v_aa(mnef) * t_aaaa(abefijmn)
                ! A(ij/mn) = (1) - (im) - (in) - (jm) - (jn) + (im)(jn)
                ! A(ab/ef) = (1) - (ae) - (af) - (be) - (bf) + (ae)(bf)
                hmatel =  v_aa(m,n,e,f) - v_aa(i,n,e,f) - v_aa(m,i,e,f) - v_aa(j,n,e,f) - v_aa(m,j,e,f) + v_aa(i,j,e,f)&
                         -v_aa(m,n,a,f) + v_aa(i,n,a,f) + v_aa(m,i,a,f) + v_aa(j,n,a,f) + v_aa(m,j,a,f) - v_aa(i,j,a,f)&
                         -v_aa(m,n,e,a) + v_aa(i,n,e,a) + v_aa(m,i,e,a) + v_aa(j,n,e,a) + v_aa(m,j,e,a) - v_aa(i,j,e,a)&
                         -v_aa(m,n,b,f) + v_aa(i,n,b,f) + v_aa(m,i,b,f) + v_aa(j,n,b,f) + v_aa(m,j,b,f) - v_aa(i,j,b,f)&
                         -v_aa(m,n,e,b) + v_aa(i,n,e,b) + v_aa(m,i,e,b) + v_aa(j,n,e,b) + v_aa(m,j,e,b) - v_aa(i,j,e,b)&
                         +v_aa(m,n,a,b) - v_aa(i,n,a,b) - v_aa(m,i,a,b) - v_aa(j,n,a,b) - v_aa(m,j,a,b) + v_aa(i,j,a,b)

                x2a(a, b, i, j) = x2a(a, b, i, j) + hmatel * t_amp
                x2a(b, a, i, j) = -1.0 * x2a(a, b, i, j)
                x2a(a, b, j, i) = -1.0 * x2a(a, b, i, j)
                x2a(b, a, j, i) = x2a(a, b, i, j)

            end do

            do idet = 1, n4b

                ! x2a(abij) <- A(m/ij)A(e/ab) v_ab(mnef) * t_aaab(abefijmn)
                a = c4_aaab(idet, 1); b = c4_aaab(idet, 2); e = c4_aaab(idet, 3); f = c4_aaab(idet, 4);
                i = c4_aaab(idet, 5); j = c4_aaab(idet, 6); m = c4_aaab(idet, 7); n = c4_aaab(idet, 8);

                call get_t4aaab_amp(t_amp, c1_a, c1_b, c2_aa, c2_ab, c3_aaa, c3_aab, c4_aaab_amps(idet),&
                                    a, b, e, f, i, j, m, n,&
                                    noa, nua, nob, nub)

                hmatel = v_ab(m, n, e, f) - v_ab(i, n, e, f) - v_ab(j, n, e, f)&
                        -v_ab(m, n, a, f) + v_ab(i, n, a, f) + v_ab(j, n, a, f)&
                        -v_ab(m, n, b, f) + v_ab(i, n, b, f) + v_ab(j, n, b, f)

!                x2a(a, b, i, j) = x2a(a, b, i, j) + hmatel * t_amp
!                x2a(b, a, i, j) = -1.0 * x2a(a, b, i, j)
!                x2a(a, b, j, i) = -1.0 * x2a(a, b, i, j)
!                x2a(b, a, j, i) = x2a(a, b, i, j)

                ! x2b(abij) <- A(i/mn)A(a/ef) v_aa(mnef) * t_aaab(aefbimnj)
                a = c4_aaab(idet, 1); e = c4_aaab(idet, 2); f = c4_aaab(idet, 3); b = c4_aaab(idet, 4);
                i = c4_aaab(idet, 5); m = c4_aaab(idet, 6); n = c4_aaab(idet, 7); j = c4_aaab(idet, 8);

                call get_t4aaab_amp(t_amp, c1_a, c1_b, c2_aa, c2_ab, c3_aaa, c3_aab, c4_aaab_amps(idet),&
                                    a, e, f, b, i, m, n, j,&
                                    noa, nua, nob, nub)

                hmatel = v_aa(m, n, e, f) - v_aa(i, n, e, f) - v_aa(m, i, e, f)&
                        -v_aa(m, n, a, f) + v_aa(i, n, a, f) + v_aa(m, i, a, f)&
                        -v_aa(m, n, e, a) + v_aa(i, n, e, a) + v_aa(m, i, e, a)

                x2b(a, b, i, j) = x2b(a, b, i, j) + hmatel * t_amp
            end do

            do idet = 1, n4c

                ! x2a(abij) <- v_bb(mnef) * t_aabb(abefijmn)
                a = c4_aabb(idet, 1); b = c4_aabb(idet, 2); e = c4_aabb(idet, 3); f = c4_aabb(idet, 4);
                i = c4_aabb(idet, 5); j = c4_aabb(idet, 6); m = c4_aabb(idet, 7); n = c4_aabb(idet, 8);

                call get_t4aabb_amp(t_amp, c1_a, c1_b, c2_aa, c2_ab, c2_bb, c3_aaa, c3_aab, c3_abb, c4_aabb_amps(idet),&
                                  a, b, e, f, i, j, m, n,&
                                  noa, nua, nob, nub)

                hmatel = v_bb(m, n, e, f)

!                x2a(a, b, i, j) = x2a(a, b, i, j) + hmatel * t_amp
!                x2a(b, a, i, j) = -1.0 * x2a(a, b, i, j)
!                x2a(a, b, j, i) = -1.0 * x2a(a, b, i, j)
!                x2a(b, a, j, i) = x2a(a, b, i, j)

                ! x2b(abij) <- A(jn)A(im)A(ae)A(bf) v_ab(mnef) * t_aabb(aefbimnj)
                a = c4_aabb(idet, 1); e = c4_aabb(idet, 2); f = c4_aabb(idet, 3); b = c4_aabb(idet, 4);
                i = c4_aabb(idet, 5); m = c4_aabb(idet, 6); n = c4_aabb(idet, 7); j = c4_aabb(idet, 8);

                call get_t4aabb_amp(t_amp, c1_a, c1_b, c2_aa, c2_ab, c2_bb, c3_aaa, c3_aab, c3_abb, c4_aabb_amps(idet),&
                                    a, e, f, b, i, m, n, j,&
                                    noa, nua, nob, nub)

                hmatel = v_ab(m, n, e, f) - v_ab(m, j, e, f) ! (jn)

                x2b(a, b, i, j) = x2b(a, b, i, j) + hmatel * t_amp

                ! x2c(abij) <- v_aa(mnef) * t_aabb(efabmnij)
                e = c4_aabb(idet, 1); f = c4_aabb(idet, 2); a = c4_aabb(idet, 3); b = c4_aabb(idet, 4);
                m = c4_aabb(idet, 5); n = c4_aabb(idet, 6); i = c4_aabb(idet, 7); j = c4_aabb(idet, 8);
                call get_t4aabb_amp(t_amp, c1_a, c1_b, c2_aa, c2_ab, c2_bb, c3_aaa, c3_aab, c3_abb, c4_aabb_amps(idet),&
                                    e, f, a, b, m, n, i, j,&
                                    noa, nua, nob, nub)

                hmatel = v_aa(m, n, e, f)

                x2c(a, b, i, j) = x2c(a, b, i, j) + hmatel * t_amp
                x2c(b, a, i, j) = -1.0 * x2c(a, b, i, j)
                x2c(a, b, j, i) = -1.0 * x2c(a, b, i, j)
                x2c(b, a, j, i) = x2c(a, b, i, j)
            end do

            do idet = 1, n4d

                ! x2b(abij) <- A(j/mn)A(b/ef) v_bb(mnef) * t_abbb(aefbimnj)
                a = c4_abbb(idet, 1); e = c4_abbb(idet, 2); f = c4_abbb(idet, 3); b = c4_abbb(idet, 4);
                i = c4_abbb(idet, 5); m = c4_abbb(idet, 6); n = c4_abbb(idet, 7); j = c4_abbb(idet, 8);

                call get_t4abbb_amp(t_amp, c1_a, c1_b, c2_ab, c2_bb, c3_abb, c3_bbb, c4_abbb_amps(idet),&
                                    a, e, f, b, i, m, n, j,&
                                    noa, nua, nob, nub)

                ! x2c(abij) <- A(n/ij)A(f/ab) v_ab(mnef) * t_abbb(efabmnij)
                e = c4_abbb(idet, 1); f = c4_abbb(idet, 2); a = c4_abbb(idet, 3); b = c4_abbb(idet, 4);
                m = c4_abbb(idet, 5); n = c4_abbb(idet, 6); i = c4_abbb(idet, 7); j = c4_abbb(idet, 8);
                call get_t4abbb_amp(t_amp, c1_a, c1_b, c2_ab, c2_bb, c3_abb, c3_bbb, c4_abbb_amps(idet),&
                                    e, f, a, b, m, n, i, j,&
                                    noa, nua, nob, nub)
            end do

            do idet = 1, n4e

                ! x2c(abij) <- A(ij/mn)A(ab/ef) v_bb(mnef) * t_bbbb(efabmnij)
                e = c4_bbbb(idet, 1); f = c4_bbbb(idet, 2); a = c4_bbbb(idet, 3); b = c4_bbbb(idet, 4);
                m = c4_bbbb(idet, 5); n = c4_bbbb(idet, 6); i = c4_bbbb(idet, 7); j = c4_bbbb(idet, 8);

                call get_t4bbbb_amp(t_amp, c1_b, c2_bb, c3_bbb, c4_bbbb_amps(idet),&
                                    e, f, a, b, m, n, i, j,&
                                    nob, nub)
            end do


        end subroutine contract_vt4_opt

        subroutine get_t4aaaa_amp(t_amp, c1_a, c2_aa, c3_aaa, c_amp,&
                                  a, b, c, d, i, j, k, l,&
                                  noa, nua)

            integer, intent(in) :: noa, nua
            integer, intent(in) :: a, b, c, d, i, j, k, l

            real(kind=8), intent(in) :: c1_a(nua, noa), c2_aa(nua, nua, noa, noa), c3_aaa(nua, nua, nua, noa, noa, noa)
            real(kind=8), intent(in) :: c_amp

            real(kind=8), intent(out) :: t_amp

            t_amp = (c_amp&
                ! -C_1 * C_3
                -c1_a(a,i)*c3_aaa(b,c,d,j,k,l) &
                +c1_a(a,j)*c3_aaa(b,c,d,i,k,l) &
                +c1_a(a,k)*c3_aaa(b,c,d,j,i,l) &
                +c1_a(a,l)*c3_aaa(b,c,d,j,k,i) &
                +c1_a(b,i)*c3_aaa(a,c,d,j,k,l) &
                +c1_a(c,i)*c3_aaa(b,a,d,j,k,l) &
                +c1_a(d,i)*c3_aaa(b,c,a,j,k,l) &
                -c1_a(b,j)*c3_aaa(a,c,d,i,k,l) &
                -c1_a(c,j)*c3_aaa(b,a,d,i,k,l) &
                -c1_a(d,j)*c3_aaa(b,c,a,i,k,l) &
                -c1_a(b,k)*c3_aaa(a,c,d,j,i,l) &
                -c1_a(c,k)*c3_aaa(b,a,d,j,i,l) &
                -c1_a(d,k)*c3_aaa(b,c,a,j,i,l) &
                -c1_a(b,l)*c3_aaa(a,c,d,j,k,i) &
                -c1_a(c,l)*c3_aaa(b,a,d,j,k,i) &
                -c1_a(d,l)*c3_aaa(b,c,a,j,k,i) &
                ! 2 * 1/2 C_1^2 * C_2
            +2.0d0*(c1_a(a,i)*c1_a(b,j)*c2_aa(c,d,k,l) & !1
                -c1_a(a,k)*c1_a(b,j)*c2_aa(c,d,i,l) & !(ik)
                -c1_a(a,l)*c1_a(b,j)*c2_aa(c,d,k,i) & !(il)
                -c1_a(a,j)*c1_a(b,i)*c2_aa(c,d,k,l) & !(ij)
                -c1_a(a,i)*c1_a(b,k)*c2_aa(c,d,j,l) & !(jk)
                -c1_a(a,i)*c1_a(b,l)*c2_aa(c,d,k,j) & !(jl)
                -c1_a(c,i)*c1_a(b,j)*c2_aa(a,d,k,l) & !(ac)
                -c1_a(d,i)*c1_a(b,j)*c2_aa(c,a,k,l) & !(ad)
                -c1_a(a,i)*c1_a(c,j)*c2_aa(b,d,k,l) & !(bc)
                -c1_a(a,i)*c1_a(d,j)*c2_aa(c,b,k,l) & !(bd)
                +c1_a(c,i)*c1_a(d,j)*c2_aa(a,b,k,l) & !(ac)(bd)
                +c1_a(c,k)*c1_a(b,j)*c2_aa(a,d,i,l) & !(ac)(ik)
                +c1_a(d,k)*c1_a(b,j)*c2_aa(c,a,i,l) & !(ad)(ik)
                +c1_a(a,k)*c1_a(c,j)*c2_aa(b,d,i,l) & !(bc)(ik)
                +c1_a(a,k)*c1_a(d,j)*c2_aa(c,b,i,l) & !(bd)(ik)
                +c1_a(c,l)*c1_a(b,j)*c2_aa(a,d,k,i) & !(ac)(il)
                +c1_a(d,l)*c1_a(b,j)*c2_aa(c,a,k,i) & !(ad)(il)
                +c1_a(a,l)*c1_a(c,j)*c2_aa(b,d,k,i) & !(bc)(il)
                +c1_a(a,l)*c1_a(d,j)*c2_aa(c,b,k,i) & !(bd)(il)
                +c1_a(c,j)*c1_a(b,i)*c2_aa(a,d,k,l) & !(ac)(ij)
                +c1_a(d,j)*c1_a(b,i)*c2_aa(c,a,k,l) & !(ad)(ij)
                +c1_a(a,j)*c1_a(c,i)*c2_aa(b,d,k,l) & !(bc)(ij)
                +c1_a(a,j)*c1_a(d,i)*c2_aa(c,b,k,l) & !(bd)(ij)
                +c1_a(c,i)*c1_a(b,k)*c2_aa(a,d,j,l) & !(ac)(jk)
                +c1_a(d,i)*c1_a(b,k)*c2_aa(c,a,j,l) & !(ad)(jk)
                +c1_a(a,i)*c1_a(c,k)*c2_aa(b,d,j,l) & !(bc)(jk)
                +c1_a(a,i)*c1_a(d,k)*c2_aa(c,b,j,l) & !(bd)(jk)
                +c1_a(c,i)*c1_a(b,l)*c2_aa(a,d,k,j) & !(ac)(jl)
                +c1_a(d,i)*c1_a(b,l)*c2_aa(c,a,k,j) & !(ad)(jl)
                +c1_a(a,i)*c1_a(c,l)*c2_aa(b,d,k,j) & !(bc)(jl)
                +c1_a(a,i)*c1_a(d,l)*c2_aa(c,b,k,j) & !(bd)(jl)
                +c1_a(a,j)*c1_a(b,k)*c2_aa(c,d,i,l) & !(ijk)
                +c1_a(a,k)*c1_a(b,i)*c2_aa(c,d,j,l) & !(ikj)
                +c1_a(a,j)*c1_a(b,l)*c2_aa(c,d,k,i) & !(ijl)
                +c1_a(a,l)*c1_a(b,i)*c2_aa(c,d,k,j) & !(ilj)
                -c1_a(a,l)*c1_a(b,k)*c2_aa(c,d,i,j) & !(kilj)
                -c1_a(c,k)*c1_a(d,j)*c2_aa(a,b,i,l) & !(ac)(bd)(ik)
                -c1_a(c,l)*c1_a(d,j)*c2_aa(a,b,k,i) & !(ac)(bd)(il)
                -c1_a(c,j)*c1_a(d,i)*c2_aa(a,b,k,l) & !(ac)(bd)(ij)
                -c1_a(c,i)*c1_a(d,k)*c2_aa(a,b,j,l) & !(ac)(bd)(jk)
                -c1_a(c,i)*c1_a(d,l)*c2_aa(a,b,k,j) & !(ac)(bd)(jl)
                -c1_a(c,j)*c1_a(b,k)*c2_aa(a,d,i,l) & !(ac)(ijk)
                -c1_a(d,j)*c1_a(b,k)*c2_aa(c,a,i,l) & !(ad)(ijk)
                -c1_a(a,j)*c1_a(c,k)*c2_aa(b,d,i,l) & !(bc)(ijk)
                -c1_a(a,j)*c1_a(d,k)*c2_aa(c,b,i,l) & !(bd)(ijk)
                -c1_a(c,k)*c1_a(b,i)*c2_aa(a,d,j,l) & !(ac)(ikj)
                -c1_a(d,k)*c1_a(b,i)*c2_aa(c,a,j,l) & !(ad)(ikj)
                -c1_a(a,k)*c1_a(c,i)*c2_aa(b,d,j,l) & !(bc)(ikj)
                -c1_a(a,k)*c1_a(d,i)*c2_aa(c,b,j,l) & !(bd)(ikj)
                -c1_a(c,j)*c1_a(b,l)*c2_aa(a,d,k,i) & !(ac)(ijl)
                -c1_a(d,j)*c1_a(b,l)*c2_aa(c,a,k,i) & !(ad)(ijl)
                -c1_a(a,j)*c1_a(c,l)*c2_aa(b,d,k,i) & !(bc)(ijl)
                -c1_a(a,j)*c1_a(d,l)*c2_aa(c,b,k,i) & !(bd)(ijl)
                -c1_a(c,l)*c1_a(b,i)*c2_aa(a,d,k,j) & !(ac)(ilj)
                -c1_a(d,l)*c1_a(b,i)*c2_aa(c,a,k,j) & !(ad)(ilj)
                -c1_a(a,l)*c1_a(c,i)*c2_aa(b,d,k,j) & !(bc)(ilj)
                -c1_a(a,l)*c1_a(d,i)*c2_aa(c,b,k,j) & !(bd)(ilj)
                +c1_a(c,j)*c1_a(d,k)*c2_aa(a,b,i,l) & !(ac)(bd)(ijk)
                +c1_a(c,k)*c1_a(d,i)*c2_aa(a,b,j,l) & !(ac)(bd)(ikj)
                +c1_a(c,j)*c1_a(d,l)*c2_aa(a,b,k,i) & !(ac)(bd)(ijl)
                +c1_a(c,l)*c1_a(d,i)*c2_aa(a,b,k,j) & !(ac)(bd)(ilj)
                +c1_a(c,l)*c1_a(b,k)*c2_aa(a,d,i,j) & !(ac)(kilj)
                +c1_a(d,l)*c1_a(b,k)*c2_aa(c,a,i,j) & !(ad)(kilj)
                +c1_a(a,l)*c1_a(c,k)*c2_aa(b,d,i,j) & !(bc)(kilj)
                +c1_a(a,l)*c1_a(d,k)*c2_aa(c,b,i,j) & !(bd)(kilj)
                -c1_a(c,l)*c1_a(d,k)*c2_aa(a,b,i,j) & !(ac)(bd)(kilj)
                -c1_a(c,k)*c1_a(b,l)*c2_aa(a,d,i,j) & !(ac)(ik)(jl)
                -c1_a(d,k)*c1_a(b,l)*c2_aa(c,a,i,j) & !(ad)(ik)(jl)
                -c1_a(a,k)*c1_a(c,l)*c2_aa(b,d,i,j) & !(bc)(ik)(jl)
                -c1_a(a,k)*c1_a(d,l)*c2_aa(c,b,i,j) & !(bd)(ik)(jl)
                +c1_a(c,k)*c1_a(d,l)*c2_aa(a,b,i,j) & !(ac)(bd)(ik)(jl)
                +c1_a(a,k)*c1_a(b,l)*c2_aa(c,d,i,j)) & !(ik)(jl)
                -c2_aa(a,b,i,j)*c2_aa(c,d,k,l) &
                +c2_aa(a,b,k,j)*c2_aa(c,d,i,l) &
                +c2_aa(a,b,l,j)*c2_aa(c,d,k,i) &
                +c2_aa(a,b,i,k)*c2_aa(c,d,j,l) &
                +c2_aa(a,b,i,l)*c2_aa(c,d,k,j) &
                +c2_aa(c,b,i,j)*c2_aa(a,d,k,l) &
                +c2_aa(d,b,i,j)*c2_aa(c,a,k,l) &
                -c2_aa(a,b,k,l)*c2_aa(c,d,i,j) &
                -c2_aa(c,b,k,j)*c2_aa(a,d,i,l) &
                -c2_aa(c,b,l,j)*c2_aa(a,d,k,i) &
                -c2_aa(c,b,i,k)*c2_aa(a,d,j,l) &
                -c2_aa(c,b,i,l)*c2_aa(a,d,k,j) &
                -c2_aa(d,b,k,j)*c2_aa(c,a,i,l) &
                -c2_aa(d,b,l,j)*c2_aa(c,a,k,i) &
                -c2_aa(d,b,i,k)*c2_aa(c,a,j,l) &
                -c2_aa(d,b,i,l)*c2_aa(c,a,k,j) &
                +c2_aa(c,b,k,l)*c2_aa(a,d,i,j) &
                +c2_aa(d,b,k,l)*c2_aa(c,a,i,j) &
                -6.0d0*(c1_a(a,i)*c1_a(b,j)*c1_a(c,k)*c1_a(d,l) &
                -c1_a(a,j)*c1_a(b,i)*c1_a(c,k)*c1_a(d,l) &
                -c1_a(a,k)*c1_a(b,j)*c1_a(c,i)*c1_a(d,l) &
                -c1_a(a,l)*c1_a(b,j)*c1_a(c,k)*c1_a(d,i) &
                -c1_a(a,i)*c1_a(b,k)*c1_a(c,j)*c1_a(d,l) &
                -c1_a(a,i)*c1_a(b,l)*c1_a(c,k)*c1_a(d,j) &
                -c1_a(a,i)*c1_a(b,j)*c1_a(c,l)*c1_a(d,k) &
                +c1_a(a,j)*c1_a(b,k)*c1_a(c,i)*c1_a(d,l) &
                +c1_a(a,k)*c1_a(b,i)*c1_a(c,j)*c1_a(d,l) &
                +c1_a(a,j)*c1_a(b,l)*c1_a(c,k)*c1_a(d,i) &
                +c1_a(a,l)*c1_a(b,i)*c1_a(c,k)*c1_a(d,j) &
                +c1_a(a,k)*c1_a(b,j)*c1_a(c,l)*c1_a(d,i) &
                +c1_a(a,l)*c1_a(b,j)*c1_a(c,i)*c1_a(d,k) &
                +c1_a(a,i)*c1_a(b,k)*c1_a(c,l)*c1_a(d,j) &
                +c1_a(a,i)*c1_a(b,l)*c1_a(c,j)*c1_a(d,k) &
                +c1_a(a,j)*c1_a(b,i)*c1_a(c,l)*c1_a(d,k) &
                +c1_a(a,k)*c1_a(b,l)*c1_a(c,i)*c1_a(d,j) &
                +c1_a(a,l)*c1_a(b,k)*c1_a(c,j)*c1_a(d,i) &
                -c1_a(a,j)*c1_a(b,k)*c1_a(c,l)*c1_a(d,i) &
                -c1_a(a,j)*c1_a(b,l)*c1_a(c,i)*c1_a(d,k) &
                -c1_a(a,k)*c1_a(b,l)*c1_a(c,j)*c1_a(d,i) &
                -c1_a(a,k)*c1_a(b,i)*c1_a(c,l)*c1_a(d,j) &
                -c1_a(a,l)*c1_a(b,i)*c1_a(c,j)*c1_a(d,k) &
                -c1_a(a,l)*c1_a(b,k)*c1_a(c,i)*c1_a(d,j)))

        end subroutine get_t4aaaa_amp

        subroutine get_t4aaab_amp(t_amp, c1_a, c1_b, c2_aa, c2_ab, c3_aaa, c3_aab, c_amp,&
                                  a, b, c, d, i, j, k, l,&
                                  noa, nua, nob, nub)

            integer, intent(in) :: noa, nua, nob, nub
            integer, intent(in) :: a, b, c, d, i, j, k, l

            real(kind=8), intent(in) :: c1_a(nua, noa), c1_b(nub, nob),&
                                        c2_aa(nua, nua, noa, noa), c2_ab(nua, nub, noa, nob),&
                                        c3_aaa(nua, nua, nua, noa, noa, noa), c3_aab(nua, nua, nub, noa, noa, nob)
            real(kind=8), intent(in) :: c_amp

            real(kind=8), intent(out) :: t_amp

            t_amp = (c_amp &
                    -c1_a(a,i)*c3_aab(b,c,d,j,k,l) &
                    +c1_a(a,j)*c3_aab(b,c,d,i,k,l) &
                    +c1_a(a,k)*c3_aab(b,c,d,j,i,l) &
                    +c1_a(b,i)*c3_aab(a,c,d,j,k,l) &
                    +c1_a(c,i)*c3_aab(b,a,d,j,k,l) &
                    -c1_a(b,j)*c3_aab(a,c,d,i,k,l) &
                    -c1_a(c,j)*c3_aab(b,a,d,i,k,l) &
                    -c1_a(b,k)*c3_aab(a,c,d,j,i,l) &
                    -c1_a(c,k)*c3_aab(b,a,d,j,i,l) &
                    -c1_b(d,l)*c3_aaa(a,b,c,i,j,k) &
                    +2.0d0*(c1_a(a,i)*c1_b(d,l)*c2_aa(b,c,j,k) &
                    -c1_a(a,j)*c1_b(d,l)*c2_aa(b,c,i,k) &
                    -c1_a(a,k)*c1_b(d,l)*c2_aa(b,c,j,i) &
                    -c1_a(b,i)*c1_b(d,l)*c2_aa(a,c,j,k) &
                    -c1_a(c,i)*c1_b(d,l)*c2_aa(b,a,j,k) &
                    +c1_a(b,j)*c1_b(d,l)*c2_aa(a,c,i,k) &
                    +c1_a(c,j)*c1_b(d,l)*c2_aa(b,a,i,k) &
                    +c1_a(b,k)*c1_b(d,l)*c2_aa(a,c,j,i) &
                    +c1_a(c,k)*c1_b(d,l)*c2_aa(b,a,j,i) &
                    +c1_a(a,i)*c1_a(b,j)*c2_ab(c,d,k,l) &
                    -c1_a(a,j)*c1_a(b,i)*c2_ab(c,d,k,l) &
                    -c1_a(a,k)*c1_a(b,j)*c2_ab(c,d,i,l) &
                    -c1_a(a,i)*c1_a(b,k)*c2_ab(c,d,j,l) &
                    -c1_a(c,i)*c1_a(b,j)*c2_ab(a,d,k,l) &
                    -c1_a(a,i)*c1_a(c,j)*c2_ab(b,d,k,l) &
                    +c1_a(a,j)*c1_a(b,k)*c2_ab(c,d,i,l) &
                    +c1_a(a,k)*c1_a(b,i)*c2_ab(c,d,j,l) &
                    +c1_a(c,j)*c1_a(b,i)*c2_ab(a,d,k,l) &
                    +c1_a(a,j)*c1_a(c,i)*c2_ab(b,d,k,l) &
                    +c1_a(c,k)*c1_a(b,j)*c2_ab(a,d,i,l) &
                    +c1_a(a,k)*c1_a(c,j)*c2_ab(b,d,i,l) &
                    +c1_a(c,i)*c1_a(b,k)*c2_ab(a,d,j,l) &
                    +c1_a(a,i)*c1_a(c,k)*c2_ab(b,d,j,l) &
                    -c1_a(c,j)*c1_a(b,k)*c2_ab(a,d,i,l) &
                    -c1_a(a,j)*c1_a(c,k)*c2_ab(b,d,i,l) &
                    -c1_a(c,k)*c1_a(b,i)*c2_ab(a,d,j,l) &
                    -c1_a(a,k)*c1_a(c,i)*c2_ab(b,d,j,l)) &
                    -c2_aa(a,b,i,j)*c2_ab(c,d,k,l) &
                    +c2_aa(a,b,k,j)*c2_ab(c,d,i,l) &
                    +c2_aa(a,b,i,k)*c2_ab(c,d,j,l) &
                    +c2_aa(c,b,i,j)*c2_ab(a,d,k,l) &
                    +c2_aa(a,c,i,j)*c2_ab(b,d,k,l) &
                    -c2_aa(c,b,k,j)*c2_ab(a,d,i,l) &
                    -c2_aa(a,c,k,j)*c2_ab(b,d,i,l) &
                    -c2_aa(c,b,i,k)*c2_ab(a,d,j,l) &
                    -c2_aa(a,c,i,k)*c2_ab(b,d,j,l) &
                    -6.0d0*(c1_a(a,i)*c1_a(b,j)*c1_a(c,k)*c1_b(d,l) &
                    -c1_a(a,j)*c1_a(b,i)*c1_a(c,k)*c1_b(d,l) &
                    -c1_a(a,k)*c1_a(b,j)*c1_a(c,i)*c1_b(d,l) &
                    -c1_a(a,i)*c1_a(b,k)*c1_a(c,j)*c1_b(d,l) &
                    +c1_a(a,j)*c1_a(b,k)*c1_a(c,i)*c1_b(d,l) &
                    +c1_a(a,k)*c1_a(b,i)*c1_a(c,j)*c1_b(d,l)))

        end subroutine get_t4aaab_amp

        subroutine get_t4aabb_amp(t_amp, c1_a, c1_b, c2_aa, c2_ab, c2_bb, c3_aaa, c3_aab, c3_abb, c_amp,&
                                  a, b, c, d, i, j, k, l,&
                                  noa, nua, nob, nub)

            integer, intent(in) :: noa, nua, nob, nub
            integer, intent(in) :: a, b, c, d, i, j, k, l

            real(kind=8), intent(in) :: c1_a(nua, noa), c1_b(nub, nob),&
                                        c2_aa(nua, nua, noa, noa), c2_ab(nua, nub, noa, nob), c2_bb(nub, nub, nob, nob),&
                                        c3_aaa(nua, nua, nua, noa, noa, noa), c3_aab(nua, nua, nub, noa, noa, nob), c3_abb(nua, nub, nub, noa, nob, nob)
            real(kind=8), intent(in) :: c_amp

            real(kind=8), intent(out) :: t_amp

            real(kind=8) :: c1c3, c12c2, c22, c13

            c1c3 = (-c1_a(a,i)*c3_abb(b,c,d,j,k,l) &
                +c1_a(a,j)*c3_abb(b,c,d,i,k,l) &
                +c1_a(b,i)*c3_abb(a,c,d,j,k,l) &
                -c1_a(b,j)*c3_abb(a,c,d,i,k,l) &
                -c1_b(d,l)*c3_aab(a,b,c,i,j,k) &
                +c1_b(d,k)*c3_aab(a,b,c,i,j,l) &
                +c1_b(c,l)*c3_aab(a,b,d,i,j,k) &
                -c1_b(c,k)*c3_aab(a,b,d,i,j,l))

            c12c2 = 2.0d0*(c1_a(a,i)*c1_a(b,j)*c2_bb(c,d,k,l) &
                -c1_a(a,j)*c1_a(b,i)*c2_bb(c,d,k,l) &
                +c1_a(a,i)*c1_b(c,k)*c2_ab(b,d,j,l) &
                -c1_a(a,j)*c1_b(c,k)*c2_ab(b,d,i,l) &
                -c1_a(a,i)*c1_b(c,l)*c2_ab(b,d,j,k) &
                -c1_a(b,i)*c1_b(c,k)*c2_ab(a,d,j,l) &
                -c1_a(a,i)*c1_b(d,k)*c2_ab(b,c,j,l) &
                +c1_a(a,j)*c1_b(c,l)*c2_ab(b,d,i,k) &
                +c1_a(b,j)*c1_b(c,k)*c2_ab(a,d,i,l) &
                +c1_a(a,j)*c1_b(d,k)*c2_ab(b,c,i,l) &
                +c1_a(b,i)*c1_b(c,l)*c2_ab(a,d,j,k) &
                +c1_a(a,i)*c1_b(d,l)*c2_ab(b,c,j,k) &
                +c1_a(b,i)*c1_b(d,k)*c2_ab(a,c,j,l) &
                -c1_a(b,j)*c1_b(c,l)*c2_ab(a,d,i,k) &
                -c1_a(a,j)*c1_b(d,l)*c2_ab(b,c,i,k) &
                -c1_a(b,j)*c1_b(d,k)*c2_ab(a,c,i,l) &
                -c1_a(b,i)*c1_b(d,l)*c2_ab(a,c,j,k) &
                +c1_a(b,j)*c1_b(d,l)*c2_ab(a,c,i,k) &
                +c1_b(c,k)*c1_b(d,l)*c2_aa(a,b,i,j) &
                -c1_b(c,l)*c1_b(d,k)*c2_aa(a,b,i,j))

            c22 = (-c2_ab(a,c,i,k)*c2_ab(b,d,j,l) &
                +c2_ab(a,c,j,k)*c2_ab(b,d,i,l) &
                +c2_ab(a,c,i,l)*c2_ab(b,d,j,k) &
                +c2_ab(b,c,i,k)*c2_ab(a,d,j,l) &
                -c2_ab(a,c,j,l)*c2_ab(b,d,i,k) &
                -c2_ab(b,c,j,k)*c2_ab(a,d,i,l) &
                -c2_ab(b,c,i,l)*c2_ab(a,d,j,k) &
                +c2_ab(b,c,j,l)*c2_ab(a,d,i,k) &
                -c2_aa(a,b,i,j)*c2_bb(c,d,k,l))

            c13 = (-6.0d0*(c1_a(a,i)*c1_a(b,j)*c1_b(c,k)*c1_b(d,l) &
                -c1_a(a,j)*c1_a(b,i)*c1_b(c,k)*c1_b(d,l) &
                -c1_a(a,i)*c1_a(b,j)*c1_b(c,l)*c1_b(d,k) &
                +c1_a(a,j)*c1_a(b,i)*c1_b(c,l)*c1_b(d,k)))

            t_amp = c_amp + c1c3 + c12c2 + c22 + c13

        end subroutine get_t4aabb_amp

        subroutine get_t4abbb_amp(t_amp, c1_a, c1_b, c2_ab, c2_bb, c3_abb, c3_bbb, c_amp,&
                                  a, b, c, d, i, j, k, l,&
                                  noa, nua, nob, nub)

            integer, intent(in) :: noa, nua, nob, nub
            integer, intent(in) :: a, b, c, d, i, j, k, l

            real(kind=8), intent(in) :: c1_a(nua, noa), c1_b(nub, nob),&
                                        c2_ab(nua, nub, noa, nob), c2_bb(nub, nub, nob, nob),&
                                        c3_abb(nua, nub, nub, noa, nob, nob), c3_bbb(nub, nub, nub, nob, nob, nob)
            real(kind=8), intent(in) :: c_amp

            real(kind=8), intent(out) :: t_amp

            t_amp = (c_amp &
                    -c1_a(a,i)*c3_bbb(b,c,d,j,k,l) &
                    -c1_b(d,l)*c3_abb(a,b,c,i,j,k) &
                    +c1_b(d,j)*c3_abb(a,b,c,i,l,k) &
                    +c1_b(d,k)*c3_abb(a,b,c,i,j,l) &
                    +c1_b(b,l)*c3_abb(a,d,c,i,j,k) &
                    +c1_b(c,l)*c3_abb(a,b,d,i,j,k) &
                    -c1_b(b,j)*c3_abb(a,d,c,i,l,k) &
                    -c1_b(c,j)*c3_abb(a,b,d,i,l,k) &
                    -c1_b(b,k)*c3_abb(a,d,c,i,j,l) &
                    -c1_b(c,k)*c3_abb(a,b,d,i,j,l) &
                    +2.0d0*(c1_b(c,k)*c1_b(d,l)*c2_ab(a,b,i,j) &
                    -c1_b(c,j)*c1_b(d,l)*c2_ab(a,b,i,k) &
                    -c1_b(c,k)*c1_b(d,j)*c2_ab(a,b,i,l) &
                    -c1_b(c,l)*c1_b(d,k)*c2_ab(a,b,i,j) &
                    -c1_b(b,k)*c1_b(d,l)*c2_ab(a,c,i,j) &
                    -c1_b(c,k)*c1_b(b,l)*c2_ab(a,d,i,j) &
                    +c1_b(c,l)*c1_b(d,j)*c2_ab(a,b,i,k) &
                    +c1_b(c,j)*c1_b(d,k)*c2_ab(a,b,i,l) &
                    +c1_b(b,j)*c1_b(d,l)*c2_ab(a,c,i,k) &
                    +c1_b(c,j)*c1_b(b,l)*c2_ab(a,d,i,k) &
                    +c1_b(b,k)*c1_b(d,j)*c2_ab(a,c,i,l) &
                    +c1_b(c,k)*c1_b(b,j)*c2_ab(a,d,i,l) &
                    +c1_b(b,l)*c1_b(d,k)*c2_ab(a,c,i,j) &
                    +c1_b(c,l)*c1_b(b,k)*c2_ab(a,d,i,j) &
                    -c1_b(b,l)*c1_b(d,j)*c2_ab(a,c,i,k) &
                    -c1_b(c,l)*c1_b(b,j)*c2_ab(a,d,i,k) &
                    -c1_b(b,j)*c1_b(d,k)*c2_ab(a,c,i,l) &
                    -c1_b(c,j)*c1_b(b,k)*c2_ab(a,d,i,l) &
                    +c1_a(a,i)*c1_b(b,j)*c2_bb(c,d,k,l) &
                    -c1_a(a,i)*c1_b(b,k)*c2_bb(c,d,j,l) &
                    -c1_a(a,i)*c1_b(b,l)*c2_bb(c,d,k,j) &
                    -c1_a(a,i)*c1_b(c,j)*c2_bb(b,d,k,l) &
                    -c1_a(a,i)*c1_b(d,j)*c2_bb(c,b,k,l) &
                    +c1_a(a,i)*c1_b(c,k)*c2_bb(b,d,j,l) &
                    +c1_a(a,i)*c1_b(d,k)*c2_bb(c,b,j,l) &
                    +c1_a(a,i)*c1_b(c,l)*c2_bb(b,d,k,j) &
                    +c1_a(a,i)*c1_b(d,l)*c2_bb(c,b,k,j)) &
                    -c2_ab(a,b,i,j)*c2_bb(c,d,k,l) &
                    +c2_ab(a,b,i,k)*c2_bb(c,d,j,l) &
                    +c2_ab(a,b,i,l)*c2_bb(c,d,k,j) &
                    +c2_ab(a,c,i,j)*c2_bb(b,d,k,l) &
                    +c2_ab(a,d,i,j)*c2_bb(c,b,k,l) &
                    -c2_ab(a,c,i,k)*c2_bb(b,d,j,l) &
                    -c2_ab(a,d,i,k)*c2_bb(c,b,j,l) &
                    -c2_ab(a,c,i,l)*c2_bb(b,d,k,j) &
                    -c2_ab(a,d,i,l)*c2_bb(c,b,k,j) &
                    -6.0d0*(c1_a(a,i)*c1_b(b,j)*c1_b(c,k)*c1_b(d,l) &
                    -c1_a(a,i)*c1_b(b,k)*c1_b(c,j)*c1_b(d,l) &
                    -c1_a(a,i)*c1_b(b,l)*c1_b(c,k)*c1_b(d,j) &
                    -c1_a(a,i)*c1_b(b,j)*c1_b(c,l)*c1_b(d,k) &
                    +c1_a(a,i)*c1_b(b,k)*c1_b(c,l)*c1_b(d,j) &
                    +c1_a(a,i)*c1_b(b,l)*c1_b(c,j)*c1_b(d,k)))

        end subroutine get_t4abbb_amp

        subroutine get_t4bbbb_amp(t_amp, c1_b, c2_bb, c3_bbb, c_amp,&
                                  a, b, c, d, i, j, k, l,&
                                  nob, nub)

            integer, intent(in) :: nob, nub
            integer, intent(in) :: a, b, c, d, i, j, k, l

            real(kind=8), intent(in) :: c1_b(nub, nob), c2_bb(nub, nub, nob, nob), c3_bbb(nub, nub, nub, nob, nob, nob)
            real(kind=8), intent(in) :: c_amp

            real(kind=8), intent(out) :: t_amp

            t_amp = (c_amp &
                    -c1_b(a,i)*c3_bbb(b,c,d,j,k,l) &
                    +c1_b(a,j)*c3_bbb(b,c,d,i,k,l) &
                    +c1_b(a,k)*c3_bbb(b,c,d,j,i,l) &
                    +c1_b(a,l)*c3_bbb(b,c,d,j,k,i) &
                    +c1_b(b,i)*c3_bbb(a,c,d,j,k,l) &
                    +c1_b(c,i)*c3_bbb(b,a,d,j,k,l) &
                    +c1_b(d,i)*c3_bbb(b,c,a,j,k,l) &
                    -c1_b(b,j)*c3_bbb(a,c,d,i,k,l) &
                    -c1_b(c,j)*c3_bbb(b,a,d,i,k,l) &
                    -c1_b(d,j)*c3_bbb(b,c,a,i,k,l) &
                    -c1_b(b,k)*c3_bbb(a,c,d,j,i,l) &
                    -c1_b(c,k)*c3_bbb(b,a,d,j,i,l) &
                    -c1_b(d,k)*c3_bbb(b,c,a,j,i,l) &
                    -c1_b(b,l)*c3_bbb(a,c,d,j,k,i) &
                    -c1_b(c,l)*c3_bbb(b,a,d,j,k,i) &
                    -c1_b(d,l)*c3_bbb(b,c,a,j,k,i) &
                    +2.0d0*(c1_b(a,i)*c1_b(b,j)*c2_bb(c,d,k,l) & !1
                    -c1_b(a,k)*c1_b(b,j)*c2_bb(c,d,i,l) & !(ik)
                    -c1_b(a,l)*c1_b(b,j)*c2_bb(c,d,k,i) & !(il)
                    -c1_b(a,j)*c1_b(b,i)*c2_bb(c,d,k,l) & !(ij)
                    -c1_b(a,i)*c1_b(b,k)*c2_bb(c,d,j,l) & !(jk)
                    -c1_b(a,i)*c1_b(b,l)*c2_bb(c,d,k,j) & !(jl)
                    -c1_b(c,i)*c1_b(b,j)*c2_bb(a,d,k,l) & !(ac)
                    -c1_b(d,i)*c1_b(b,j)*c2_bb(c,a,k,l) & !(ad)
                    -c1_b(a,i)*c1_b(c,j)*c2_bb(b,d,k,l) & !(bc)
                    -c1_b(a,i)*c1_b(d,j)*c2_bb(c,b,k,l) & !(bd)
                    +c1_b(c,i)*c1_b(d,j)*c2_bb(a,b,k,l) & !(ac)(bd)
                    +c1_b(c,k)*c1_b(b,j)*c2_bb(a,d,i,l) & !(ac)(ik)
                    +c1_b(d,k)*c1_b(b,j)*c2_bb(c,a,i,l) & !(ad)(ik)
                    +c1_b(a,k)*c1_b(c,j)*c2_bb(b,d,i,l) & !(bc)(ik)
                    +c1_b(a,k)*c1_b(d,j)*c2_bb(c,b,i,l) & !(bd)(ik)
                    +c1_b(c,l)*c1_b(b,j)*c2_bb(a,d,k,i) & !(ac)(il)
                    +c1_b(d,l)*c1_b(b,j)*c2_bb(c,a,k,i) & !(ad)(il)
                    +c1_b(a,l)*c1_b(c,j)*c2_bb(b,d,k,i) & !(bc)(il)
                    +c1_b(a,l)*c1_b(d,j)*c2_bb(c,b,k,i) & !(bd)(il)
                    +c1_b(c,j)*c1_b(b,i)*c2_bb(a,d,k,l) & !(ac)(ij)
                    +c1_b(d,j)*c1_b(b,i)*c2_bb(c,a,k,l) & !(ad)(ij)
                    +c1_b(a,j)*c1_b(c,i)*c2_bb(b,d,k,l) & !(bc)(ij)
                    +c1_b(a,j)*c1_b(d,i)*c2_bb(c,b,k,l) & !(bd)(ij)
                    +c1_b(c,i)*c1_b(b,k)*c2_bb(a,d,j,l) & !(ac)(jk)
                    +c1_b(d,i)*c1_b(b,k)*c2_bb(c,a,j,l) & !(ad)(jk)
                    +c1_b(a,i)*c1_b(c,k)*c2_bb(b,d,j,l) & !(bc)(jk)
                    +c1_b(a,i)*c1_b(d,k)*c2_bb(c,b,j,l) & !(bd)(jk)
                    +c1_b(c,i)*c1_b(b,l)*c2_bb(a,d,k,j) & !(ac)(jl)
                    +c1_b(d,i)*c1_b(b,l)*c2_bb(c,a,k,j) & !(ad)(jl)
                    +c1_b(a,i)*c1_b(c,l)*c2_bb(b,d,k,j) & !(bc)(jl)
                    +c1_b(a,i)*c1_b(d,l)*c2_bb(c,b,k,j) & !(bd)(jl)
                    +c1_b(a,j)*c1_b(b,k)*c2_bb(c,d,i,l) & !(ijk)
                    +c1_b(a,k)*c1_b(b,i)*c2_bb(c,d,j,l) & !(ikj)
                    +c1_b(a,j)*c1_b(b,l)*c2_bb(c,d,k,i) & !(ijl)
                    +c1_b(a,l)*c1_b(b,i)*c2_bb(c,d,k,j) & !(ilj)
                    -c1_b(a,l)*c1_b(b,k)*c2_bb(c,d,i,j) & !(kilj)
                    -c1_b(c,k)*c1_b(d,j)*c2_bb(a,b,i,l) & !(ac)(bd)(ik)
                    -c1_b(c,l)*c1_b(d,j)*c2_bb(a,b,k,i) & !(ac)(bd)(il)
                    -c1_b(c,j)*c1_b(d,i)*c2_bb(a,b,k,l) & !(ac)(bd)(ij)
                    -c1_b(c,i)*c1_b(d,k)*c2_bb(a,b,j,l) & !(ac)(bd)(jk)
                    -c1_b(c,i)*c1_b(d,l)*c2_bb(a,b,k,j) & !(ac)(bd)(jl)
                    -c1_b(c,j)*c1_b(b,k)*c2_bb(a,d,i,l) & !(ac)(ijk)
                    -c1_b(d,j)*c1_b(b,k)*c2_bb(c,a,i,l) & !(ad)(ijk)
                    -c1_b(a,j)*c1_b(c,k)*c2_bb(b,d,i,l) & !(bc)(ijk)
                    -c1_b(a,j)*c1_b(d,k)*c2_bb(c,b,i,l) & !(bd)(ijk)
                    -c1_b(c,k)*c1_b(b,i)*c2_bb(a,d,j,l) & !(ac)(ikj)
                    -c1_b(d,k)*c1_b(b,i)*c2_bb(c,a,j,l) & !(ad)(ikj)
                    -c1_b(a,k)*c1_b(c,i)*c2_bb(b,d,j,l) & !(bc)(ikj)
                    -c1_b(a,k)*c1_b(d,i)*c2_bb(c,b,j,l) & !(bd)(ikj)
                    -c1_b(c,j)*c1_b(b,l)*c2_bb(a,d,k,i) & !(ac)(ijl)
                    -c1_b(d,j)*c1_b(b,l)*c2_bb(c,a,k,i) & !(ad)(ijl)
                    -c1_b(a,j)*c1_b(c,l)*c2_bb(b,d,k,i) & !(bc)(ijl)
                    -c1_b(a,j)*c1_b(d,l)*c2_bb(c,b,k,i) & !(bd)(ijl)
                    -c1_b(c,l)*c1_b(b,i)*c2_bb(a,d,k,j) & !(ac)(ilj)
                    -c1_b(d,l)*c1_b(b,i)*c2_bb(c,a,k,j) & !(ad)(ilj)
                    -c1_b(a,l)*c1_b(c,i)*c2_bb(b,d,k,j) & !(bc)(ilj)
                    -c1_b(a,l)*c1_b(d,i)*c2_bb(c,b,k,j) & !(bd)(ilj)
                    +c1_b(c,j)*c1_b(d,k)*c2_bb(a,b,i,l) & !(ac)(bd)(ijk)
                    +c1_b(c,k)*c1_b(d,i)*c2_bb(a,b,j,l) & !(ac)(bd)(ikj)
                    +c1_b(c,j)*c1_b(d,l)*c2_bb(a,b,k,i) & !(ac)(bd)(ijl)
                    +c1_b(c,l)*c1_b(d,i)*c2_bb(a,b,k,j) & !(ac)(bd)(ilj)
                    +c1_b(c,l)*c1_b(b,k)*c2_bb(a,d,i,j) & !(ac)(kilj)
                    +c1_b(d,l)*c1_b(b,k)*c2_bb(c,a,i,j) & !(ad)(kilj)
                    +c1_b(a,l)*c1_b(c,k)*c2_bb(b,d,i,j) & !(bc)(kilj)
                    +c1_b(a,l)*c1_b(d,k)*c2_bb(c,b,i,j) & !(bd)(kilj)
                    -c1_b(c,l)*c1_b(d,k)*c2_bb(a,b,i,j) & !(ac)(bd)(kilj)
                    -c1_b(c,k)*c1_b(b,l)*c2_bb(a,d,i,j) & !(ac)(ik)(jl)
                    -c1_b(d,k)*c1_b(b,l)*c2_bb(c,a,i,j) & !(ad)(ik)(jl)
                    -c1_b(a,k)*c1_b(c,l)*c2_bb(b,d,i,j) & !(bc)(ik)(jl)
                    -c1_b(a,k)*c1_b(d,l)*c2_bb(c,b,i,j) & !(bd)(ik)(jl)
                    +c1_b(c,k)*c1_b(d,l)*c2_bb(a,b,i,j) & !(ac)(bd)(ik)(jl)
                    +c1_b(a,k)*c1_b(b,l)*c2_bb(c,d,i,j)) & !(ik)(jl)
                    -c2_bb(a,b,i,j)*c2_bb(c,d,k,l) &
                    +c2_bb(a,b,k,j)*c2_bb(c,d,i,l) &
                    +c2_bb(a,b,l,j)*c2_bb(c,d,k,i) &
                    +c2_bb(a,b,i,k)*c2_bb(c,d,j,l) &
                    +c2_bb(a,b,i,l)*c2_bb(c,d,k,j) &
                    +c2_bb(c,b,i,j)*c2_bb(a,d,k,l) &
                    +c2_bb(d,b,i,j)*c2_bb(c,a,k,l) &
                    -c2_bb(a,b,k,l)*c2_bb(c,d,i,j) &
                    -c2_bb(c,b,k,j)*c2_bb(a,d,i,l) &
                    -c2_bb(c,b,l,j)*c2_bb(a,d,k,i) &
                    -c2_bb(c,b,i,k)*c2_bb(a,d,j,l) &
                    -c2_bb(c,b,i,l)*c2_bb(a,d,k,j) &
                    -c2_bb(d,b,k,j)*c2_bb(c,a,i,l) &
                    -c2_bb(d,b,l,j)*c2_bb(c,a,k,i) &
                    -c2_bb(d,b,i,k)*c2_bb(c,a,j,l) &
                    -c2_bb(d,b,i,l)*c2_bb(c,a,k,j) &
                    +c2_bb(c,b,k,l)*c2_bb(a,d,i,j) &
                    +c2_bb(d,b,k,l)*c2_bb(c,a,i,j) &
                    -6.0d0*(c1_b(a,i)*c1_b(b,j)*c1_b(c,k)*c1_b(d,l) &
                    -c1_b(a,j)*c1_b(b,i)*c1_b(c,k)*c1_b(d,l) &
                    -c1_b(a,k)*c1_b(b,j)*c1_b(c,i)*c1_b(d,l) &
                    -c1_b(a,l)*c1_b(b,j)*c1_b(c,k)*c1_b(d,i) &
                    -c1_b(a,i)*c1_b(b,k)*c1_b(c,j)*c1_b(d,l) &
                    -c1_b(a,i)*c1_b(b,l)*c1_b(c,k)*c1_b(d,j) &
                    -c1_b(a,i)*c1_b(b,j)*c1_b(c,l)*c1_b(d,k) &
                    +c1_b(a,j)*c1_b(b,k)*c1_b(c,i)*c1_b(d,l) &
                    +c1_b(a,k)*c1_b(b,i)*c1_b(c,j)*c1_b(d,l) &
                    +c1_b(a,j)*c1_b(b,l)*c1_b(c,k)*c1_b(d,i) &
                    +c1_b(a,l)*c1_b(b,i)*c1_b(c,k)*c1_b(d,j) &
                    +c1_b(a,k)*c1_b(b,j)*c1_b(c,l)*c1_b(d,i) &
                    +c1_b(a,l)*c1_b(b,j)*c1_b(c,i)*c1_b(d,k) &
                    +c1_b(a,i)*c1_b(b,k)*c1_b(c,l)*c1_b(d,j) &
                    +c1_b(a,i)*c1_b(b,l)*c1_b(c,j)*c1_b(d,k) &
                    +c1_b(a,j)*c1_b(b,i)*c1_b(c,l)*c1_b(d,k) &
                    +c1_b(a,k)*c1_b(b,l)*c1_b(c,i)*c1_b(d,j) &
                    +c1_b(a,l)*c1_b(b,k)*c1_b(c,j)*c1_b(d,i) &
                    -c1_b(a,j)*c1_b(b,k)*c1_b(c,l)*c1_b(d,i) &
                    -c1_b(a,j)*c1_b(b,l)*c1_b(c,i)*c1_b(d,k) &
                    -c1_b(a,k)*c1_b(b,l)*c1_b(c,j)*c1_b(d,i) &
                    -c1_b(a,k)*c1_b(b,i)*c1_b(c,l)*c1_b(d,j) &
                    -c1_b(a,l)*c1_b(b,i)*c1_b(c,j)*c1_b(d,k) &
                    -c1_b(a,l)*c1_b(b,k)*c1_b(c,i)*c1_b(d,j)))

        end subroutine get_t4bbbb_amp


        subroutine cluster_analysis_t4(t4_aaaa, t4_aaab, t4_aabb, t4_abbb, t4_bbbb,&
                                       c1_a, c1_b, c2_aa, c2_ab, c2_bb,&
                                       c3_aaa, c3_aab, c3_abb, c3_bbb,&
                                       c4_aaaa, c4_aaab, c4_aabb, c4_abbb, c4_bbbb,&
                                       c4_aaaa_amps, c4_aaab_amps, c4_aabb_amps, c4_abbb_amps, c4_bbbb_amps,&
                                       n4a, n4b, n4c, n4d, n4e,&
                                       noa, nua, nob, nub)

                integer, intent(in) :: noa, nua, nob, nub, n4a, n4b, n4c, n4d, n4e
                integer, intent(in) :: c4_aaaa(n4a, 8), c4_aaab(n4b, 8), c4_aabb(n4c, 8), c4_abbb(n4d, 8), c4_bbbb(n4e, 8)
                real(kind=8), intent(in) :: c1_a(nua, noa), c1_b(nub, nob),&
                                            c2_aa(nua, nua, noa, noa), c2_ab(nua, nub, noa, nob), c2_bb(nub, nub, nob, nob),&
                                            c3_aaa(nua, nua, nua, noa, noa, noa), c3_aab(nua, nua, nub, noa, noa, nob),&
                                            c3_abb(nua, nub, nub, noa, nob, nob), c3_bbb(nub, nub, nub, nob, nob, nob),&
                                            c4_aaaa_amps(n4a),&
                                            c4_aaab_amps(n4b),&
                                            c4_aabb_amps(n4c),&
                                            c4_abbb_amps(n4d),&
                                            c4_bbbb_amps(n4e)

                real(kind=8), intent(out) :: t4_aaaa(nua, nua, nua, nua, noa, noa, noa, noa),&
                                             t4_aaab(nua, nua, nua, nub, noa, noa, noa, nob),&
                                             t4_aabb(nua, nua, nub, nub, noa, noa, nob, nob),&
                                             t4_abbb(nua, nub, nub, nub, noa, nob, nob, nob),&
                                             t4_bbbb(nub, nub, nub, nub, nob, nob, nob, nob)

                integer :: a, b, c, d, i, j, k, l, idet
                real(kind=8) :: c1c3, c12c2, c22, c13

                c13 = 0.0d0
                c22 = 0.0d0
                c12c2 = 0.0d0
                c1c3 = 0.0d0

                t4_aaaa = 0.0d0
                do idet = 1, n4a

                    a = c4_aaaa(idet, 1); b = c4_aaaa(idet, 2); c = c4_aaaa(idet, 3); d = c4_aaaa(idet, 4);
                    i = c4_aaaa(idet, 5); j = c4_aaaa(idet, 6); k = c4_aaaa(idet, 7); l = c4_aaaa(idet, 8);

                    !if (abs(c4_aaaa_amps(idet)) > 0.0d0) then
                        t4_aaaa(a,b,c,d,i,j,k,l) = &
                            (c4_aaaa_amps(idet) &
                            ! -C_1 * C_3
                            -c1_a(a,i)*c3_aaa(b,c,d,j,k,l) &
                            +c1_a(a,j)*c3_aaa(b,c,d,i,k,l) &
                            +c1_a(a,k)*c3_aaa(b,c,d,j,i,l) &
                            +c1_a(a,l)*c3_aaa(b,c,d,j,k,i) &
                            +c1_a(b,i)*c3_aaa(a,c,d,j,k,l) &
                            +c1_a(c,i)*c3_aaa(b,a,d,j,k,l) &
                            +c1_a(d,i)*c3_aaa(b,c,a,j,k,l) &
                            -c1_a(b,j)*c3_aaa(a,c,d,i,k,l) &
                            -c1_a(c,j)*c3_aaa(b,a,d,i,k,l) &
                            -c1_a(d,j)*c3_aaa(b,c,a,i,k,l) &
                            -c1_a(b,k)*c3_aaa(a,c,d,j,i,l) &
                            -c1_a(c,k)*c3_aaa(b,a,d,j,i,l) &
                            -c1_a(d,k)*c3_aaa(b,c,a,j,i,l) &
                            -c1_a(b,l)*c3_aaa(a,c,d,j,k,i) &
                            -c1_a(c,l)*c3_aaa(b,a,d,j,k,i) &
                            -c1_a(d,l)*c3_aaa(b,c,a,j,k,i) &
                            ! 2 * 1/2 C_1^2 * C_2
                        +2.0d0*(c1_a(a,i)*c1_a(b,j)*c2_aa(c,d,k,l) & !1
                            -c1_a(a,k)*c1_a(b,j)*c2_aa(c,d,i,l) & !(ik)
                            -c1_a(a,l)*c1_a(b,j)*c2_aa(c,d,k,i) & !(il)
                            -c1_a(a,j)*c1_a(b,i)*c2_aa(c,d,k,l) & !(ij)
                            -c1_a(a,i)*c1_a(b,k)*c2_aa(c,d,j,l) & !(jk)
                            -c1_a(a,i)*c1_a(b,l)*c2_aa(c,d,k,j) & !(jl)
                            -c1_a(c,i)*c1_a(b,j)*c2_aa(a,d,k,l) & !(ac)
                            -c1_a(d,i)*c1_a(b,j)*c2_aa(c,a,k,l) & !(ad)
                            -c1_a(a,i)*c1_a(c,j)*c2_aa(b,d,k,l) & !(bc)
                            -c1_a(a,i)*c1_a(d,j)*c2_aa(c,b,k,l) & !(bd)
                            +c1_a(c,i)*c1_a(d,j)*c2_aa(a,b,k,l) & !(ac)(bd)
                            +c1_a(c,k)*c1_a(b,j)*c2_aa(a,d,i,l) & !(ac)(ik)
                            +c1_a(d,k)*c1_a(b,j)*c2_aa(c,a,i,l) & !(ad)(ik)
                            +c1_a(a,k)*c1_a(c,j)*c2_aa(b,d,i,l) & !(bc)(ik)
                            +c1_a(a,k)*c1_a(d,j)*c2_aa(c,b,i,l) & !(bd)(ik)
                            +c1_a(c,l)*c1_a(b,j)*c2_aa(a,d,k,i) & !(ac)(il)
                            +c1_a(d,l)*c1_a(b,j)*c2_aa(c,a,k,i) & !(ad)(il)
                            +c1_a(a,l)*c1_a(c,j)*c2_aa(b,d,k,i) & !(bc)(il)
                            +c1_a(a,l)*c1_a(d,j)*c2_aa(c,b,k,i) & !(bd)(il)
                            +c1_a(c,j)*c1_a(b,i)*c2_aa(a,d,k,l) & !(ac)(ij)
                            +c1_a(d,j)*c1_a(b,i)*c2_aa(c,a,k,l) & !(ad)(ij)
                            +c1_a(a,j)*c1_a(c,i)*c2_aa(b,d,k,l) & !(bc)(ij)
                            +c1_a(a,j)*c1_a(d,i)*c2_aa(c,b,k,l) & !(bd)(ij)
                            +c1_a(c,i)*c1_a(b,k)*c2_aa(a,d,j,l) & !(ac)(jk)
                            +c1_a(d,i)*c1_a(b,k)*c2_aa(c,a,j,l) & !(ad)(jk)
                            +c1_a(a,i)*c1_a(c,k)*c2_aa(b,d,j,l) & !(bc)(jk)
                            +c1_a(a,i)*c1_a(d,k)*c2_aa(c,b,j,l) & !(bd)(jk)
                            +c1_a(c,i)*c1_a(b,l)*c2_aa(a,d,k,j) & !(ac)(jl)
                            +c1_a(d,i)*c1_a(b,l)*c2_aa(c,a,k,j) & !(ad)(jl)
                            +c1_a(a,i)*c1_a(c,l)*c2_aa(b,d,k,j) & !(bc)(jl)
                            +c1_a(a,i)*c1_a(d,l)*c2_aa(c,b,k,j) & !(bd)(jl)
                            +c1_a(a,j)*c1_a(b,k)*c2_aa(c,d,i,l) & !(ijk)
                            +c1_a(a,k)*c1_a(b,i)*c2_aa(c,d,j,l) & !(ikj)
                            +c1_a(a,j)*c1_a(b,l)*c2_aa(c,d,k,i) & !(ijl)
                            +c1_a(a,l)*c1_a(b,i)*c2_aa(c,d,k,j) & !(ilj)
                            -c1_a(a,l)*c1_a(b,k)*c2_aa(c,d,i,j) & !(kilj)
                            -c1_a(c,k)*c1_a(d,j)*c2_aa(a,b,i,l) & !(ac)(bd)(ik)
                            -c1_a(c,l)*c1_a(d,j)*c2_aa(a,b,k,i) & !(ac)(bd)(il)
                            -c1_a(c,j)*c1_a(d,i)*c2_aa(a,b,k,l) & !(ac)(bd)(ij)
                            -c1_a(c,i)*c1_a(d,k)*c2_aa(a,b,j,l) & !(ac)(bd)(jk)
                            -c1_a(c,i)*c1_a(d,l)*c2_aa(a,b,k,j) & !(ac)(bd)(jl)
                            -c1_a(c,j)*c1_a(b,k)*c2_aa(a,d,i,l) & !(ac)(ijk)
                            -c1_a(d,j)*c1_a(b,k)*c2_aa(c,a,i,l) & !(ad)(ijk)
                            -c1_a(a,j)*c1_a(c,k)*c2_aa(b,d,i,l) & !(bc)(ijk)
                            -c1_a(a,j)*c1_a(d,k)*c2_aa(c,b,i,l) & !(bd)(ijk)
                            -c1_a(c,k)*c1_a(b,i)*c2_aa(a,d,j,l) & !(ac)(ikj)
                            -c1_a(d,k)*c1_a(b,i)*c2_aa(c,a,j,l) & !(ad)(ikj)
                            -c1_a(a,k)*c1_a(c,i)*c2_aa(b,d,j,l) & !(bc)(ikj)
                            -c1_a(a,k)*c1_a(d,i)*c2_aa(c,b,j,l) & !(bd)(ikj)
                            -c1_a(c,j)*c1_a(b,l)*c2_aa(a,d,k,i) & !(ac)(ijl)
                            -c1_a(d,j)*c1_a(b,l)*c2_aa(c,a,k,i) & !(ad)(ijl)
                            -c1_a(a,j)*c1_a(c,l)*c2_aa(b,d,k,i) & !(bc)(ijl)
                            -c1_a(a,j)*c1_a(d,l)*c2_aa(c,b,k,i) & !(bd)(ijl)
                            -c1_a(c,l)*c1_a(b,i)*c2_aa(a,d,k,j) & !(ac)(ilj)
                            -c1_a(d,l)*c1_a(b,i)*c2_aa(c,a,k,j) & !(ad)(ilj)
                            -c1_a(a,l)*c1_a(c,i)*c2_aa(b,d,k,j) & !(bc)(ilj)
                            -c1_a(a,l)*c1_a(d,i)*c2_aa(c,b,k,j) & !(bd)(ilj)
                            +c1_a(c,j)*c1_a(d,k)*c2_aa(a,b,i,l) & !(ac)(bd)(ijk)
                            +c1_a(c,k)*c1_a(d,i)*c2_aa(a,b,j,l) & !(ac)(bd)(ikj)
                            +c1_a(c,j)*c1_a(d,l)*c2_aa(a,b,k,i) & !(ac)(bd)(ijl)
                            +c1_a(c,l)*c1_a(d,i)*c2_aa(a,b,k,j) & !(ac)(bd)(ilj)
                            +c1_a(c,l)*c1_a(b,k)*c2_aa(a,d,i,j) & !(ac)(kilj)
                            +c1_a(d,l)*c1_a(b,k)*c2_aa(c,a,i,j) & !(ad)(kilj)
                            +c1_a(a,l)*c1_a(c,k)*c2_aa(b,d,i,j) & !(bc)(kilj)
                            +c1_a(a,l)*c1_a(d,k)*c2_aa(c,b,i,j) & !(bd)(kilj)
                            -c1_a(c,l)*c1_a(d,k)*c2_aa(a,b,i,j) & !(ac)(bd)(kilj)
                            -c1_a(c,k)*c1_a(b,l)*c2_aa(a,d,i,j) & !(ac)(ik)(jl)
                            -c1_a(d,k)*c1_a(b,l)*c2_aa(c,a,i,j) & !(ad)(ik)(jl)
                            -c1_a(a,k)*c1_a(c,l)*c2_aa(b,d,i,j) & !(bc)(ik)(jl)
                            -c1_a(a,k)*c1_a(d,l)*c2_aa(c,b,i,j) & !(bd)(ik)(jl)
                            +c1_a(c,k)*c1_a(d,l)*c2_aa(a,b,i,j) & !(ac)(bd)(ik)(jl)
                            +c1_a(a,k)*c1_a(b,l)*c2_aa(c,d,i,j)) & !(ik)(jl)
                            -c2_aa(a,b,i,j)*c2_aa(c,d,k,l) &
                            +c2_aa(a,b,k,j)*c2_aa(c,d,i,l) &
                            +c2_aa(a,b,l,j)*c2_aa(c,d,k,i) &
                            +c2_aa(a,b,i,k)*c2_aa(c,d,j,l) &
                            +c2_aa(a,b,i,l)*c2_aa(c,d,k,j) &
                            +c2_aa(c,b,i,j)*c2_aa(a,d,k,l) &
                            +c2_aa(d,b,i,j)*c2_aa(c,a,k,l) &
                            -c2_aa(a,b,k,l)*c2_aa(c,d,i,j) &
                            -c2_aa(c,b,k,j)*c2_aa(a,d,i,l) &
                            -c2_aa(c,b,l,j)*c2_aa(a,d,k,i) &
                            -c2_aa(c,b,i,k)*c2_aa(a,d,j,l) &
                            -c2_aa(c,b,i,l)*c2_aa(a,d,k,j) &
                            -c2_aa(d,b,k,j)*c2_aa(c,a,i,l) &
                            -c2_aa(d,b,l,j)*c2_aa(c,a,k,i) &
                            -c2_aa(d,b,i,k)*c2_aa(c,a,j,l) &
                            -c2_aa(d,b,i,l)*c2_aa(c,a,k,j) &
                            +c2_aa(c,b,k,l)*c2_aa(a,d,i,j) &
                            +c2_aa(d,b,k,l)*c2_aa(c,a,i,j) &
                            -6.0d0*(c1_a(a,i)*c1_a(b,j)*c1_a(c,k)*c1_a(d,l) &
                            -c1_a(a,j)*c1_a(b,i)*c1_a(c,k)*c1_a(d,l) &
                            -c1_a(a,k)*c1_a(b,j)*c1_a(c,i)*c1_a(d,l) &
                            -c1_a(a,l)*c1_a(b,j)*c1_a(c,k)*c1_a(d,i) &
                            -c1_a(a,i)*c1_a(b,k)*c1_a(c,j)*c1_a(d,l) &
                            -c1_a(a,i)*c1_a(b,l)*c1_a(c,k)*c1_a(d,j) &
                            -c1_a(a,i)*c1_a(b,j)*c1_a(c,l)*c1_a(d,k) &
                            +c1_a(a,j)*c1_a(b,k)*c1_a(c,i)*c1_a(d,l) &
                            +c1_a(a,k)*c1_a(b,i)*c1_a(c,j)*c1_a(d,l) &
                            +c1_a(a,j)*c1_a(b,l)*c1_a(c,k)*c1_a(d,i) &
                            +c1_a(a,l)*c1_a(b,i)*c1_a(c,k)*c1_a(d,j) &
                            +c1_a(a,k)*c1_a(b,j)*c1_a(c,l)*c1_a(d,i) &
                            +c1_a(a,l)*c1_a(b,j)*c1_a(c,i)*c1_a(d,k) &
                            +c1_a(a,i)*c1_a(b,k)*c1_a(c,l)*c1_a(d,j) &
                            +c1_a(a,i)*c1_a(b,l)*c1_a(c,j)*c1_a(d,k) &
                            +c1_a(a,j)*c1_a(b,i)*c1_a(c,l)*c1_a(d,k) &
                            +c1_a(a,k)*c1_a(b,l)*c1_a(c,i)*c1_a(d,j) &
                            +c1_a(a,l)*c1_a(b,k)*c1_a(c,j)*c1_a(d,i) &
                            -c1_a(a,j)*c1_a(b,k)*c1_a(c,l)*c1_a(d,i) &
                            -c1_a(a,j)*c1_a(b,l)*c1_a(c,i)*c1_a(d,k) &
                            -c1_a(a,k)*c1_a(b,l)*c1_a(c,j)*c1_a(d,i) &
                            -c1_a(a,k)*c1_a(b,i)*c1_a(c,l)*c1_a(d,j) &
                            -c1_a(a,l)*c1_a(b,i)*c1_a(c,j)*c1_a(d,k) &
                            -c1_a(a,l)*c1_a(b,k)*c1_a(c,i)*c1_a(d,j)))
                    !else
                    !    t4_aaaa = 0.0d0
                    !end if
                    t4_aaaa(a, b, c, d, i, j, l, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, c, d, i, k, j, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, c, d, i, k, l, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, c, d, i, l, j, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, c, d, i, l, k, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, c, d, j, i, k, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, c, d, j, i, l, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, c, d, j, k, i, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, c, d, j, k, l, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, c, d, j, l, i, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, c, d, j, l, k, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, c, d, k, i, j, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, c, d, k, i, l, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, c, d, k, j, i, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, c, d, k, j, l, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, c, d, k, l, i, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, c, d, k, l, j, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, c, d, l, i, j, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, c, d, l, i, k, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, c, d, l, j, i, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, c, d, l, j, k, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, c, d, l, k, i, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, c, d, l, k, j, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, d, c, i, j, k, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, d, c, i, j, l, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, d, c, i, k, j, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, d, c, i, k, l, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, d, c, i, l, j, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, d, c, i, l, k, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, d, c, j, i, k, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, d, c, j, i, l, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, d, c, j, k, i, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, d, c, j, k, l, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, d, c, j, l, i, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, d, c, j, l, k, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, d, c, k, i, j, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, d, c, k, i, l, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, d, c, k, j, i, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, d, c, k, j, l, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, d, c, k, l, i, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, d, c, k, l, j, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, d, c, l, i, j, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, d, c, l, i, k, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, d, c, l, j, i, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, d, c, l, j, k, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, d, c, l, k, i, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, b, d, c, l, k, j, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, b, d, i, j, k, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, b, d, i, j, l, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, b, d, i, k, j, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, b, d, i, k, l, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, b, d, i, l, j, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, b, d, i, l, k, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, b, d, j, i, k, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, b, d, j, i, l, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, b, d, j, k, i, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, b, d, j, k, l, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, b, d, j, l, i, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, b, d, j, l, k, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, b, d, k, i, j, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, b, d, k, i, l, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, b, d, k, j, i, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, b, d, k, j, l, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, b, d, k, l, i, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, b, d, k, l, j, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, b, d, l, i, j, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, b, d, l, i, k, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, b, d, l, j, i, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, b, d, l, j, k, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, b, d, l, k, i, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, b, d, l, k, j, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, d, b, i, j, k, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, d, b, i, j, l, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, d, b, i, k, j, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, d, b, i, k, l, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, d, b, i, l, j, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, d, b, i, l, k, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, d, b, j, i, k, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, d, b, j, i, l, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, d, b, j, k, i, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, d, b, j, k, l, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, d, b, j, l, i, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, d, b, j, l, k, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, d, b, k, i, j, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, d, b, k, i, l, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, d, b, k, j, i, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, d, b, k, j, l, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, d, b, k, l, i, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, d, b, k, l, j, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, d, b, l, i, j, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, d, b, l, i, k, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, d, b, l, j, i, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, d, b, l, j, k, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, d, b, l, k, i, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, c, d, b, l, k, j, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, b, c, i, j, k, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, b, c, i, j, l, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, b, c, i, k, j, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, b, c, i, k, l, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, b, c, i, l, j, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, b, c, i, l, k, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, b, c, j, i, k, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, b, c, j, i, l, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, b, c, j, k, i, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, b, c, j, k, l, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, b, c, j, l, i, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, b, c, j, l, k, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, b, c, k, i, j, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, b, c, k, i, l, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, b, c, k, j, i, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, b, c, k, j, l, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, b, c, k, l, i, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, b, c, k, l, j, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, b, c, l, i, j, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, b, c, l, i, k, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, b, c, l, j, i, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, b, c, l, j, k, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, b, c, l, k, i, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, b, c, l, k, j, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, c, b, i, j, k, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, c, b, i, j, l, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, c, b, i, k, j, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, c, b, i, k, l, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, c, b, i, l, j, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, c, b, i, l, k, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, c, b, j, i, k, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, c, b, j, i, l, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, c, b, j, k, i, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, c, b, j, k, l, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, c, b, j, l, i, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, c, b, j, l, k, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, c, b, k, i, j, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, c, b, k, i, l, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, c, b, k, j, i, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, c, b, k, j, l, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, c, b, k, l, i, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, c, b, k, l, j, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, c, b, l, i, j, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, c, b, l, i, k, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, c, b, l, j, i, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, c, b, l, j, k, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, c, b, l, k, i, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(a, d, c, b, l, k, j, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, c, d, i, j, k, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, c, d, i, j, l, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, c, d, i, k, j, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, c, d, i, k, l, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, c, d, i, l, j, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, c, d, i, l, k, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, c, d, j, i, k, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, c, d, j, i, l, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, c, d, j, k, i, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, c, d, j, k, l, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, c, d, j, l, i, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, c, d, j, l, k, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, c, d, k, i, j, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, c, d, k, i, l, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, c, d, k, j, i, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, c, d, k, j, l, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, c, d, k, l, i, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, c, d, k, l, j, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, c, d, l, i, j, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, c, d, l, i, k, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, c, d, l, j, i, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, c, d, l, j, k, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, c, d, l, k, i, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, c, d, l, k, j, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, d, c, i, j, k, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, d, c, i, j, l, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, d, c, i, k, j, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, d, c, i, k, l, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, d, c, i, l, j, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, d, c, i, l, k, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, d, c, j, i, k, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, d, c, j, i, l, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, d, c, j, k, i, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, d, c, j, k, l, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, d, c, j, l, i, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, d, c, j, l, k, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, d, c, k, i, j, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, d, c, k, i, l, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, d, c, k, j, i, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, d, c, k, j, l, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, d, c, k, l, i, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, d, c, k, l, j, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, d, c, l, i, j, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, d, c, l, i, k, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, d, c, l, j, i, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, d, c, l, j, k, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, d, c, l, k, i, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, a, d, c, l, k, j, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, a, d, i, j, k, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, a, d, i, j, l, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, a, d, i, k, j, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, a, d, i, k, l, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, a, d, i, l, j, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, a, d, i, l, k, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, a, d, j, i, k, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, a, d, j, i, l, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, a, d, j, k, i, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, a, d, j, k, l, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, a, d, j, l, i, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, a, d, j, l, k, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, a, d, k, i, j, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, a, d, k, i, l, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, a, d, k, j, i, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, a, d, k, j, l, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, a, d, k, l, i, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, a, d, k, l, j, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, a, d, l, i, j, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, a, d, l, i, k, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, a, d, l, j, i, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, a, d, l, j, k, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, a, d, l, k, i, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, a, d, l, k, j, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, d, a, i, j, k, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, d, a, i, j, l, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, d, a, i, k, j, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, d, a, i, k, l, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, d, a, i, l, j, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, d, a, i, l, k, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, d, a, j, i, k, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, d, a, j, i, l, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, d, a, j, k, i, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, d, a, j, k, l, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, d, a, j, l, i, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, d, a, j, l, k, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, d, a, k, i, j, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, d, a, k, i, l, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, d, a, k, j, i, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, d, a, k, j, l, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, d, a, k, l, i, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, d, a, k, l, j, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, d, a, l, i, j, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, d, a, l, i, k, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, d, a, l, j, i, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, d, a, l, j, k, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, d, a, l, k, i, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, c, d, a, l, k, j, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, a, c, i, j, k, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, a, c, i, j, l, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, a, c, i, k, j, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, a, c, i, k, l, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, a, c, i, l, j, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, a, c, i, l, k, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, a, c, j, i, k, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, a, c, j, i, l, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, a, c, j, k, i, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, a, c, j, k, l, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, a, c, j, l, i, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, a, c, j, l, k, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, a, c, k, i, j, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, a, c, k, i, l, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, a, c, k, j, i, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, a, c, k, j, l, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, a, c, k, l, i, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, a, c, k, l, j, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, a, c, l, i, j, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, a, c, l, i, k, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, a, c, l, j, i, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, a, c, l, j, k, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, a, c, l, k, i, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, a, c, l, k, j, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, c, a, i, j, k, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, c, a, i, j, l, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, c, a, i, k, j, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, c, a, i, k, l, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, c, a, i, l, j, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, c, a, i, l, k, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, c, a, j, i, k, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, c, a, j, i, l, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, c, a, j, k, i, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, c, a, j, k, l, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, c, a, j, l, i, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, c, a, j, l, k, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, c, a, k, i, j, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, c, a, k, i, l, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, c, a, k, j, i, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, c, a, k, j, l, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, c, a, k, l, i, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, c, a, k, l, j, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, c, a, l, i, j, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, c, a, l, i, k, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, c, a, l, j, i, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, c, a, l, j, k, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, c, a, l, k, i, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(b, d, c, a, l, k, j, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, b, d, i, j, k, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, b, d, i, j, l, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, b, d, i, k, j, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, b, d, i, k, l, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, b, d, i, l, j, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, b, d, i, l, k, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, b, d, j, i, k, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, b, d, j, i, l, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, b, d, j, k, i, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, b, d, j, k, l, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, b, d, j, l, i, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, b, d, j, l, k, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, b, d, k, i, j, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, b, d, k, i, l, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, b, d, k, j, i, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, b, d, k, j, l, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, b, d, k, l, i, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, b, d, k, l, j, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, b, d, l, i, j, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, b, d, l, i, k, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, b, d, l, j, i, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, b, d, l, j, k, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, b, d, l, k, i, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, b, d, l, k, j, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, d, b, i, j, k, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, d, b, i, j, l, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, d, b, i, k, j, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, d, b, i, k, l, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, d, b, i, l, j, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, d, b, i, l, k, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, d, b, j, i, k, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, d, b, j, i, l, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, d, b, j, k, i, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, d, b, j, k, l, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, d, b, j, l, i, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, d, b, j, l, k, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, d, b, k, i, j, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, d, b, k, i, l, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, d, b, k, j, i, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, d, b, k, j, l, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, d, b, k, l, i, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, d, b, k, l, j, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, d, b, l, i, j, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, d, b, l, i, k, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, d, b, l, j, i, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, d, b, l, j, k, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, d, b, l, k, i, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, a, d, b, l, k, j, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, a, d, i, j, k, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, a, d, i, j, l, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, a, d, i, k, j, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, a, d, i, k, l, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, a, d, i, l, j, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, a, d, i, l, k, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, a, d, j, i, k, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, a, d, j, i, l, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, a, d, j, k, i, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, a, d, j, k, l, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, a, d, j, l, i, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, a, d, j, l, k, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, a, d, k, i, j, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, a, d, k, i, l, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, a, d, k, j, i, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, a, d, k, j, l, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, a, d, k, l, i, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, a, d, k, l, j, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, a, d, l, i, j, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, a, d, l, i, k, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, a, d, l, j, i, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, a, d, l, j, k, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, a, d, l, k, i, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, a, d, l, k, j, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, d, a, i, j, k, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, d, a, i, j, l, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, d, a, i, k, j, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, d, a, i, k, l, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, d, a, i, l, j, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, d, a, i, l, k, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, d, a, j, i, k, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, d, a, j, i, l, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, d, a, j, k, i, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, d, a, j, k, l, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, d, a, j, l, i, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, d, a, j, l, k, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, d, a, k, i, j, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, d, a, k, i, l, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, d, a, k, j, i, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, d, a, k, j, l, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, d, a, k, l, i, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, d, a, k, l, j, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, d, a, l, i, j, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, d, a, l, i, k, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, d, a, l, j, i, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, d, a, l, j, k, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, d, a, l, k, i, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, b, d, a, l, k, j, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, a, b, i, j, k, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, a, b, i, j, l, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, a, b, i, k, j, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, a, b, i, k, l, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, a, b, i, l, j, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, a, b, i, l, k, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, a, b, j, i, k, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, a, b, j, i, l, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, a, b, j, k, i, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, a, b, j, k, l, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, a, b, j, l, i, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, a, b, j, l, k, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, a, b, k, i, j, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, a, b, k, i, l, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, a, b, k, j, i, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, a, b, k, j, l, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, a, b, k, l, i, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, a, b, k, l, j, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, a, b, l, i, j, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, a, b, l, i, k, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, a, b, l, j, i, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, a, b, l, j, k, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, a, b, l, k, i, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, a, b, l, k, j, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, b, a, i, j, k, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, b, a, i, j, l, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, b, a, i, k, j, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, b, a, i, k, l, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, b, a, i, l, j, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, b, a, i, l, k, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, b, a, j, i, k, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, b, a, j, i, l, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, b, a, j, k, i, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, b, a, j, k, l, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, b, a, j, l, i, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, b, a, j, l, k, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, b, a, k, i, j, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, b, a, k, i, l, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, b, a, k, j, i, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, b, a, k, j, l, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, b, a, k, l, i, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, b, a, k, l, j, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, b, a, l, i, j, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, b, a, l, i, k, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, b, a, l, j, i, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, b, a, l, j, k, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, b, a, l, k, i, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(c, d, b, a, l, k, j, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, b, c, i, j, k, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, b, c, i, j, l, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, b, c, i, k, j, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, b, c, i, k, l, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, b, c, i, l, j, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, b, c, i, l, k, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, b, c, j, i, k, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, b, c, j, i, l, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, b, c, j, k, i, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, b, c, j, k, l, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, b, c, j, l, i, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, b, c, j, l, k, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, b, c, k, i, j, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, b, c, k, i, l, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, b, c, k, j, i, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, b, c, k, j, l, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, b, c, k, l, i, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, b, c, k, l, j, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, b, c, l, i, j, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, b, c, l, i, k, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, b, c, l, j, i, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, b, c, l, j, k, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, b, c, l, k, i, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, b, c, l, k, j, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, c, b, i, j, k, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, c, b, i, j, l, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, c, b, i, k, j, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, c, b, i, k, l, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, c, b, i, l, j, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, c, b, i, l, k, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, c, b, j, i, k, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, c, b, j, i, l, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, c, b, j, k, i, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, c, b, j, k, l, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, c, b, j, l, i, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, c, b, j, l, k, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, c, b, k, i, j, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, c, b, k, i, l, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, c, b, k, j, i, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, c, b, k, j, l, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, c, b, k, l, i, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, c, b, k, l, j, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, c, b, l, i, j, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, c, b, l, i, k, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, c, b, l, j, i, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, c, b, l, j, k, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, c, b, l, k, i, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, a, c, b, l, k, j, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, a, c, i, j, k, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, a, c, i, j, l, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, a, c, i, k, j, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, a, c, i, k, l, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, a, c, i, l, j, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, a, c, i, l, k, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, a, c, j, i, k, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, a, c, j, i, l, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, a, c, j, k, i, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, a, c, j, k, l, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, a, c, j, l, i, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, a, c, j, l, k, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, a, c, k, i, j, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, a, c, k, i, l, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, a, c, k, j, i, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, a, c, k, j, l, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, a, c, k, l, i, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, a, c, k, l, j, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, a, c, l, i, j, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, a, c, l, i, k, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, a, c, l, j, i, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, a, c, l, j, k, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, a, c, l, k, i, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, a, c, l, k, j, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, c, a, i, j, k, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, c, a, i, j, l, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, c, a, i, k, j, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, c, a, i, k, l, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, c, a, i, l, j, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, c, a, i, l, k, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, c, a, j, i, k, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, c, a, j, i, l, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, c, a, j, k, i, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, c, a, j, k, l, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, c, a, j, l, i, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, c, a, j, l, k, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, c, a, k, i, j, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, c, a, k, i, l, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, c, a, k, j, i, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, c, a, k, j, l, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, c, a, k, l, i, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, c, a, k, l, j, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, c, a, l, i, j, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, c, a, l, i, k, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, c, a, l, j, i, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, c, a, l, j, k, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, c, a, l, k, i, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, b, c, a, l, k, j, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, a, b, i, j, k, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, a, b, i, j, l, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, a, b, i, k, j, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, a, b, i, k, l, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, a, b, i, l, j, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, a, b, i, l, k, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, a, b, j, i, k, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, a, b, j, i, l, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, a, b, j, k, i, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, a, b, j, k, l, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, a, b, j, l, i, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, a, b, j, l, k, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, a, b, k, i, j, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, a, b, k, i, l, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, a, b, k, j, i, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, a, b, k, j, l, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, a, b, k, l, i, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, a, b, k, l, j, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, a, b, l, i, j, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, a, b, l, i, k, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, a, b, l, j, i, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, a, b, l, j, k, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, a, b, l, k, i, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, a, b, l, k, j, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, b, a, i, j, k, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, b, a, i, j, l, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, b, a, i, k, j, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, b, a, i, k, l, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, b, a, i, l, j, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, b, a, i, l, k, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, b, a, j, i, k, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, b, a, j, i, l, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, b, a, j, k, i, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, b, a, j, k, l, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, b, a, j, l, i, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, b, a, j, l, k, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, b, a, k, i, j, l) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, b, a, k, i, l, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, b, a, k, j, i, l) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, b, a, k, j, l, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, b, a, k, l, i, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, b, a, k, l, j, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, b, a, l, i, j, k) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, b, a, l, i, k, j) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, b, a, l, j, i, k) = t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, b, a, l, j, k, i) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, b, a, l, k, i, j) = -1.0 * t4_aaaa(a, b, c, d, i, j, k, l)
                    t4_aaaa(d, c, b, a, l, k, j, i) = t4_aaaa(a, b, c, d, i, j, k, l)
                enddo

                t4_aaab = 0.0d0
                do idet = 1, n4b

                    a = c4_aaab(idet, 1); b = c4_aaab(idet, 2); c = c4_aaab(idet, 3); d = c4_aaab(idet, 4);
                    i = c4_aaab(idet, 5); j = c4_aaab(idet, 6); k = c4_aaab(idet, 7); l = c4_aaab(idet, 8);

                    !if (abs(c4_aaab_amps(idet)) > 0.0d0) then
                        t4_aaab(a,b,c,d,i,j,k,l)=(c4_aaab_amps(idet) &
                            -c1_a(a,i)*c3_aab(b,c,d,j,k,l) &
                            +c1_a(a,j)*c3_aab(b,c,d,i,k,l) &
                            +c1_a(a,k)*c3_aab(b,c,d,j,i,l) &
                            +c1_a(b,i)*c3_aab(a,c,d,j,k,l) &
                            +c1_a(c,i)*c3_aab(b,a,d,j,k,l) &
                            -c1_a(b,j)*c3_aab(a,c,d,i,k,l) &
                            -c1_a(c,j)*c3_aab(b,a,d,i,k,l) &
                            -c1_a(b,k)*c3_aab(a,c,d,j,i,l) &
                            -c1_a(c,k)*c3_aab(b,a,d,j,i,l) &
                            -c1_b(d,l)*c3_aaa(a,b,c,i,j,k) &
                            +2.0d0*(c1_a(a,i)*c1_b(d,l)*c2_aa(b,c,j,k) &
                            -c1_a(a,j)*c1_b(d,l)*c2_aa(b,c,i,k) &
                            -c1_a(a,k)*c1_b(d,l)*c2_aa(b,c,j,i) &
                            -c1_a(b,i)*c1_b(d,l)*c2_aa(a,c,j,k) &
                            -c1_a(c,i)*c1_b(d,l)*c2_aa(b,a,j,k) &
                            +c1_a(b,j)*c1_b(d,l)*c2_aa(a,c,i,k) &
                            +c1_a(c,j)*c1_b(d,l)*c2_aa(b,a,i,k) &
                            +c1_a(b,k)*c1_b(d,l)*c2_aa(a,c,j,i) &
                            +c1_a(c,k)*c1_b(d,l)*c2_aa(b,a,j,i) &
                            +c1_a(a,i)*c1_a(b,j)*c2_ab(c,d,k,l) &
                            -c1_a(a,j)*c1_a(b,i)*c2_ab(c,d,k,l) &
                            -c1_a(a,k)*c1_a(b,j)*c2_ab(c,d,i,l) &
                            -c1_a(a,i)*c1_a(b,k)*c2_ab(c,d,j,l) &
                            -c1_a(c,i)*c1_a(b,j)*c2_ab(a,d,k,l) &
                            -c1_a(a,i)*c1_a(c,j)*c2_ab(b,d,k,l) &
                            +c1_a(a,j)*c1_a(b,k)*c2_ab(c,d,i,l) &
                            +c1_a(a,k)*c1_a(b,i)*c2_ab(c,d,j,l) &
                            +c1_a(c,j)*c1_a(b,i)*c2_ab(a,d,k,l) &
                            +c1_a(a,j)*c1_a(c,i)*c2_ab(b,d,k,l) &
                            +c1_a(c,k)*c1_a(b,j)*c2_ab(a,d,i,l) &
                            +c1_a(a,k)*c1_a(c,j)*c2_ab(b,d,i,l) &
                            +c1_a(c,i)*c1_a(b,k)*c2_ab(a,d,j,l) &
                            +c1_a(a,i)*c1_a(c,k)*c2_ab(b,d,j,l) &
                            -c1_a(c,j)*c1_a(b,k)*c2_ab(a,d,i,l) &
                            -c1_a(a,j)*c1_a(c,k)*c2_ab(b,d,i,l) &
                            -c1_a(c,k)*c1_a(b,i)*c2_ab(a,d,j,l) &
                            -c1_a(a,k)*c1_a(c,i)*c2_ab(b,d,j,l)) &
                            -c2_aa(a,b,i,j)*c2_ab(c,d,k,l) &
                            +c2_aa(a,b,k,j)*c2_ab(c,d,i,l) &
                            +c2_aa(a,b,i,k)*c2_ab(c,d,j,l) &
                            +c2_aa(c,b,i,j)*c2_ab(a,d,k,l) &
                            +c2_aa(a,c,i,j)*c2_ab(b,d,k,l) &
                            -c2_aa(c,b,k,j)*c2_ab(a,d,i,l) &
                            -c2_aa(a,c,k,j)*c2_ab(b,d,i,l) &
                            -c2_aa(c,b,i,k)*c2_ab(a,d,j,l) &
                            -c2_aa(a,c,i,k)*c2_ab(b,d,j,l) &
                            -6.0d0*(c1_a(a,i)*c1_a(b,j)*c1_a(c,k)*c1_b(d,l) &
                            -c1_a(a,j)*c1_a(b,i)*c1_a(c,k)*c1_b(d,l) &
                            -c1_a(a,k)*c1_a(b,j)*c1_a(c,i)*c1_b(d,l) &
                            -c1_a(a,i)*c1_a(b,k)*c1_a(c,j)*c1_b(d,l) &
                            +c1_a(a,j)*c1_a(b,k)*c1_a(c,i)*c1_b(d,l) &
                            +c1_a(a,k)*c1_a(b,i)*c1_a(c,j)*c1_b(d,l)))
                    !else
                    !    t4_aaab = 0.0d0
                    !end if
                    t4_aaab(a, b, c, d, i, k, j, l) = -1.0 * t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(a, b, c, d, j, i, k, l) = -1.0 * t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(a, b, c, d, j, k, i, l) = t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(a, b, c, d, k, i, j, l) = t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(a, b, c, d, k, j, i, l) = -1.0 * t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(a, c, b, d, i, j, k, l) = -1.0 * t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(a, c, b, d, i, k, j, l) = t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(a, c, b, d, j, i, k, l) = t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(a, c, b, d, j, k, i, l) = -1.0 * t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(a, c, b, d, k, i, j, l) = -1.0 * t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(a, c, b, d, k, j, i, l) = t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(b, a, c, d, i, j, k, l) = -1.0 * t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(b, a, c, d, i, k, j, l) = t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(b, a, c, d, j, i, k, l) = t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(b, a, c, d, j, k, i, l) = -1.0 * t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(b, a, c, d, k, i, j, l) = -1.0 * t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(b, a, c, d, k, j, i, l) = t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(b, c, a, d, i, j, k, l) = t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(b, c, a, d, i, k, j, l) = -1.0 * t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(b, c, a, d, j, i, k, l) = -1.0 * t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(b, c, a, d, j, k, i, l) = t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(b, c, a, d, k, i, j, l) = t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(b, c, a, d, k, j, i, l) = -1.0 * t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(c, a, b, d, i, j, k, l) = t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(c, a, b, d, i, k, j, l) = -1.0 * t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(c, a, b, d, j, i, k, l) = -1.0 * t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(c, a, b, d, j, k, i, l) = t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(c, a, b, d, k, i, j, l) = t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(c, a, b, d, k, j, i, l) = -1.0 * t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(c, b, a, d, i, j, k, l) = -1.0 * t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(c, b, a, d, i, k, j, l) = t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(c, b, a, d, j, i, k, l) = t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(c, b, a, d, j, k, i, l) = -1.0 * t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(c, b, a, d, k, i, j, l) = -1.0 * t4_aaab(a, b, c, d, i, j, k, l)
                    t4_aaab(c, b, a, d, k, j, i, l) = t4_aaab(a, b, c, d, i, j, k, l)

                enddo

                t4_aabb = 0.0d0
                do idet = 1, n4c

                    a = c4_aabb(idet, 1); b = c4_aabb(idet, 2); c = c4_aabb(idet, 3); d = c4_aabb(idet, 4);
                    i = c4_aabb(idet, 5); j = c4_aabb(idet, 6); k = c4_aabb(idet, 7); l = c4_aabb(idet, 8);

                    !if (abs(c4_aabb_amps(idet)) > 0.0d0) then
                        c1c3 = (-c1_a(a,i)*c3_abb(b,c,d,j,k,l) &
                            +c1_a(a,j)*c3_abb(b,c,d,i,k,l) &
                            +c1_a(b,i)*c3_abb(a,c,d,j,k,l) &
                            -c1_a(b,j)*c3_abb(a,c,d,i,k,l) &
                            -c1_b(d,l)*c3_aab(a,b,c,i,j,k) &
                            +c1_b(d,k)*c3_aab(a,b,c,i,j,l) &
                            +c1_b(c,l)*c3_aab(a,b,d,i,j,k) &
                            -c1_b(c,k)*c3_aab(a,b,d,i,j,l))

                        c12c2 = 2.0d0*(c1_a(a,i)*c1_a(b,j)*c2_bb(c,d,k,l) &
                            -c1_a(a,j)*c1_a(b,i)*c2_bb(c,d,k,l) &
                            +c1_a(a,i)*c1_b(c,k)*c2_ab(b,d,j,l) &
                            -c1_a(a,j)*c1_b(c,k)*c2_ab(b,d,i,l) &
                            -c1_a(a,i)*c1_b(c,l)*c2_ab(b,d,j,k) &
                            -c1_a(b,i)*c1_b(c,k)*c2_ab(a,d,j,l) &
                            -c1_a(a,i)*c1_b(d,k)*c2_ab(b,c,j,l) &
                            +c1_a(a,j)*c1_b(c,l)*c2_ab(b,d,i,k) &
                            +c1_a(b,j)*c1_b(c,k)*c2_ab(a,d,i,l) &
                            +c1_a(a,j)*c1_b(d,k)*c2_ab(b,c,i,l) &
                            +c1_a(b,i)*c1_b(c,l)*c2_ab(a,d,j,k) &
                            +c1_a(a,i)*c1_b(d,l)*c2_ab(b,c,j,k) &
                            +c1_a(b,i)*c1_b(d,k)*c2_ab(a,c,j,l) &
                            -c1_a(b,j)*c1_b(c,l)*c2_ab(a,d,i,k) &
                            -c1_a(a,j)*c1_b(d,l)*c2_ab(b,c,i,k) &
                            -c1_a(b,j)*c1_b(d,k)*c2_ab(a,c,i,l) &
                            -c1_a(b,i)*c1_b(d,l)*c2_ab(a,c,j,k) &
                            +c1_a(b,j)*c1_b(d,l)*c2_ab(a,c,i,k) &
                            +c1_b(c,k)*c1_b(d,l)*c2_aa(a,b,i,j) &
                            -c1_b(c,l)*c1_b(d,k)*c2_aa(a,b,i,j))

                        c22 = (-c2_ab(a,c,i,k)*c2_ab(b,d,j,l) &
                            +c2_ab(a,c,j,k)*c2_ab(b,d,i,l) &
                            +c2_ab(a,c,i,l)*c2_ab(b,d,j,k) &
                            +c2_ab(b,c,i,k)*c2_ab(a,d,j,l) &
                            -c2_ab(a,c,j,l)*c2_ab(b,d,i,k) &
                            -c2_ab(b,c,j,k)*c2_ab(a,d,i,l) &
                            -c2_ab(b,c,i,l)*c2_ab(a,d,j,k) &
                            +c2_ab(b,c,j,l)*c2_ab(a,d,i,k) &
                            -c2_aa(a,b,i,j)*c2_bb(c,d,k,l))

                        c13 = (-6.0d0*(c1_a(a,i)*c1_a(b,j)*c1_b(c,k)*c1_b(d,l) &
                            -c1_a(a,j)*c1_a(b,i)*c1_b(c,k)*c1_b(d,l) &
                            -c1_a(a,i)*c1_a(b,j)*c1_b(c,l)*c1_b(d,k) &
                            +c1_a(a,j)*c1_a(b,i)*c1_b(c,l)*c1_b(d,k)))

                        t4_aabb(a,b,c,d,i,j,k,l)=c4_aabb_amps(idet) + c1c3 + c12c2 + c22 + c13
                    !else
                    !    t4_aabb = 0.0d0
                    !end if
                    t4_aabb(a, b, c, d, i, j, l, k) = -1.0 * t4_aabb(a, b, c, d, i, j, k, l)
                    t4_aabb(a, b, c, d, j, i, k, l) = -1.0 * t4_aabb(a, b, c, d, i, j, k, l)
                    t4_aabb(a, b, c, d, j, i, l, k) = +t4_aabb(a, b, c, d, i, j, k, l)
                    t4_aabb(a, b, d, c, i, j, k, l) = -1.0 * t4_aabb(a, b, c, d, i, j, k, l)
                    t4_aabb(a, b, d, c, i, j, l, k) = +t4_aabb(a, b, c, d, i, j, k, l)
                    t4_aabb(a, b, d, c, j, i, k, l) = +t4_aabb(a, b, c, d, i, j, k, l)
                    t4_aabb(a, b, d, c, j, i, l, k) = -1.0 * t4_aabb(a, b, c, d, i, j, k, l)
                    t4_aabb(b, a, c, d, i, j, k, l) = -1.0 * t4_aabb(a, b, c, d, i, j, k, l)
                    t4_aabb(b, a, c, d, i, j, l, k) = +t4_aabb(a, b, c, d, i, j, k, l)
                    t4_aabb(b, a, c, d, j, i, k, l) = +t4_aabb(a, b, c, d, i, j, k, l)
                    t4_aabb(b, a, c, d, j, i, l, k) = -1.0 * t4_aabb(a, b, c, d, i, j, k, l)
                    t4_aabb(b, a, d, c, i, j, k, l) = +t4_aabb(a, b, c, d, i, j, k, l)
                    t4_aabb(b, a, d, c, i, j, l, k) = -1.0 * t4_aabb(a, b, c, d, i, j, k, l)
                    t4_aabb(b, a, d, c, j, i, k, l) = -1.0 * t4_aabb(a, b, c, d, i, j, k, l)
                    t4_aabb(b, a, d, c, j, i, l, k) = +t4_aabb(a, b, c, d, i, j, k, l)

                enddo

                t4_abbb = 0.0d0
                do idet = 1, n4d

                    a = c4_abbb(idet, 1); b = c4_abbb(idet, 2); c = c4_abbb(idet, 3); d = c4_abbb(idet, 4);
                    i = c4_abbb(idet, 5); j = c4_abbb(idet, 6); k = c4_abbb(idet, 7); l = c4_abbb(idet, 8);

                    !if (abs(c4_abbb_amps(idet)) > 0.0d0) then

                        t4_abbb(a,b,c,d,i,j,k,l)=(c4_abbb_amps(idet) &
                            -c1_a(a,i)*c3_bbb(b,c,d,j,k,l) &
                            -c1_b(d,l)*c3_abb(a,b,c,i,j,k) &
                            +c1_b(d,j)*c3_abb(a,b,c,i,l,k) &
                            +c1_b(d,k)*c3_abb(a,b,c,i,j,l) &
                            +c1_b(b,l)*c3_abb(a,d,c,i,j,k) &
                            +c1_b(c,l)*c3_abb(a,b,d,i,j,k) &
                            -c1_b(b,j)*c3_abb(a,d,c,i,l,k) &
                            -c1_b(c,j)*c3_abb(a,b,d,i,l,k) &
                            -c1_b(b,k)*c3_abb(a,d,c,i,j,l) &
                            -c1_b(c,k)*c3_abb(a,b,d,i,j,l) &
                            +2.0d0*(c1_b(c,k)*c1_b(d,l)*c2_ab(a,b,i,j) &
                            -c1_b(c,j)*c1_b(d,l)*c2_ab(a,b,i,k) &
                            -c1_b(c,k)*c1_b(d,j)*c2_ab(a,b,i,l) &
                            -c1_b(c,l)*c1_b(d,k)*c2_ab(a,b,i,j) &
                            -c1_b(b,k)*c1_b(d,l)*c2_ab(a,c,i,j) &
                            -c1_b(c,k)*c1_b(b,l)*c2_ab(a,d,i,j) &
                            +c1_b(c,l)*c1_b(d,j)*c2_ab(a,b,i,k) &
                            +c1_b(c,j)*c1_b(d,k)*c2_ab(a,b,i,l) &
                            +c1_b(b,j)*c1_b(d,l)*c2_ab(a,c,i,k) &
                            +c1_b(c,j)*c1_b(b,l)*c2_ab(a,d,i,k) &
                            +c1_b(b,k)*c1_b(d,j)*c2_ab(a,c,i,l) &
                            +c1_b(c,k)*c1_b(b,j)*c2_ab(a,d,i,l) &
                            +c1_b(b,l)*c1_b(d,k)*c2_ab(a,c,i,j) &
                            +c1_b(c,l)*c1_b(b,k)*c2_ab(a,d,i,j) &
                            -c1_b(b,l)*c1_b(d,j)*c2_ab(a,c,i,k) &
                            -c1_b(c,l)*c1_b(b,j)*c2_ab(a,d,i,k) &
                            -c1_b(b,j)*c1_b(d,k)*c2_ab(a,c,i,l) &
                            -c1_b(c,j)*c1_b(b,k)*c2_ab(a,d,i,l) &
                            +c1_a(a,i)*c1_b(b,j)*c2_bb(c,d,k,l) &
                            -c1_a(a,i)*c1_b(b,k)*c2_bb(c,d,j,l) &
                            -c1_a(a,i)*c1_b(b,l)*c2_bb(c,d,k,j) &
                            -c1_a(a,i)*c1_b(c,j)*c2_bb(b,d,k,l) &
                            -c1_a(a,i)*c1_b(d,j)*c2_bb(c,b,k,l) &
                            +c1_a(a,i)*c1_b(c,k)*c2_bb(b,d,j,l) &
                            +c1_a(a,i)*c1_b(d,k)*c2_bb(c,b,j,l) &
                            +c1_a(a,i)*c1_b(c,l)*c2_bb(b,d,k,j) &
                            +c1_a(a,i)*c1_b(d,l)*c2_bb(c,b,k,j)) &
                            -c2_ab(a,b,i,j)*c2_bb(c,d,k,l) &
                            +c2_ab(a,b,i,k)*c2_bb(c,d,j,l) &
                            +c2_ab(a,b,i,l)*c2_bb(c,d,k,j) &
                            +c2_ab(a,c,i,j)*c2_bb(b,d,k,l) &
                            +c2_ab(a,d,i,j)*c2_bb(c,b,k,l) &
                            -c2_ab(a,c,i,k)*c2_bb(b,d,j,l) &
                            -c2_ab(a,d,i,k)*c2_bb(c,b,j,l) &
                            -c2_ab(a,c,i,l)*c2_bb(b,d,k,j) &
                            -c2_ab(a,d,i,l)*c2_bb(c,b,k,j) &
                            -6.0d0*(c1_a(a,i)*c1_b(b,j)*c1_b(c,k)*c1_b(d,l) &
                            -c1_a(a,i)*c1_b(b,k)*c1_b(c,j)*c1_b(d,l) &
                            -c1_a(a,i)*c1_b(b,l)*c1_b(c,k)*c1_b(d,j) &
                            -c1_a(a,i)*c1_b(b,j)*c1_b(c,l)*c1_b(d,k) &
                            +c1_a(a,i)*c1_b(b,k)*c1_b(c,l)*c1_b(d,j) &
                            +c1_a(a,i)*c1_b(b,l)*c1_b(c,j)*c1_b(d,k)))
                    !else
                    !    t4_abbb = 0.0d0
                    !end if
                        t4_abbb(a, b, c, d, i, j, l, k) = -1.0 * t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, b, c, d, i, k, j, l) = -1.0 * t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, b, c, d, i, k, l, j) = t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, b, c, d, i, l, j, k) = t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, b, c, d, i, l, k, j) = -1.0 * t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, b, d, c, i, j, k, l) = -1.0 * t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, b, d, c, i, j, l, k) = t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, b, d, c, i, k, j, l) = t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, b, d, c, i, k, l, j) = -1.0 * t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, b, d, c, i, l, j, k) = -1.0 * t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, b, d, c, i, l, k, j) = t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, c, b, d, i, j, k, l) = -1.0 * t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, c, b, d, i, j, l, k) = t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, c, b, d, i, k, j, l) = t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, c, b, d, i, k, l, j) = -1.0 * t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, c, b, d, i, l, j, k) = -1.0 * t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, c, b, d, i, l, k, j) = t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, c, d, b, i, j, k, l) = t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, c, d, b, i, j, l, k) = -1.0 * t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, c, d, b, i, k, j, l) = -1.0 * t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, c, d, b, i, k, l, j) = t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, c, d, b, i, l, j, k) = t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, c, d, b, i, l, k, j) = -1.0 * t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, d, b, c, i, j, k, l) = t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, d, b, c, i, j, l, k) = -1.0 * t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, d, b, c, i, k, j, l) = -1.0 * t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, d, b, c, i, k, l, j) = t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, d, b, c, i, l, j, k) = t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, d, b, c, i, l, k, j) = -1.0 * t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, d, c, b, i, j, k, l) = -1.0 * t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, d, c, b, i, j, l, k) = t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, d, c, b, i, k, j, l) = t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, d, c, b, i, k, l, j) = -1.0 * t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, d, c, b, i, l, j, k) = -1.0 * t4_abbb(a, b, c, d, i, j, k, l)
                        t4_abbb(a, d, c, b, i, l, k, j) = t4_abbb(a, b, c, d, i, j, k, l)
                enddo

                t4_bbbb = 0.0d0
                do idet = 1, n4e
                    a = c4_bbbb(idet, 1); b = c4_bbbb(idet, 2); c = c4_bbbb(idet, 3); d = c4_bbbb(idet, 4);
                    i = c4_bbbb(idet, 5); j = c4_bbbb(idet, 6); k = c4_bbbb(idet, 7); l = c4_bbbb(idet, 8);

                    !if (abs(c4_bbbb_amps(idet)) > 0.0d0) then
                        t4_bbbb(a,b,c,d,i,j,k,l) = &
                            (c4_bbbb_amps(idet) &
                            -c1_b(a,i)*c3_bbb(b,c,d,j,k,l) &
                            +c1_b(a,j)*c3_bbb(b,c,d,i,k,l) &
                            +c1_b(a,k)*c3_bbb(b,c,d,j,i,l) &
                            +c1_b(a,l)*c3_bbb(b,c,d,j,k,i) &
                            +c1_b(b,i)*c3_bbb(a,c,d,j,k,l) &
                            +c1_b(c,i)*c3_bbb(b,a,d,j,k,l) &
                            +c1_b(d,i)*c3_bbb(b,c,a,j,k,l) &
                            -c1_b(b,j)*c3_bbb(a,c,d,i,k,l) &
                            -c1_b(c,j)*c3_bbb(b,a,d,i,k,l) &
                            -c1_b(d,j)*c3_bbb(b,c,a,i,k,l) &
                            -c1_b(b,k)*c3_bbb(a,c,d,j,i,l) &
                            -c1_b(c,k)*c3_bbb(b,a,d,j,i,l) &
                            -c1_b(d,k)*c3_bbb(b,c,a,j,i,l) &
                            -c1_b(b,l)*c3_bbb(a,c,d,j,k,i) &
                            -c1_b(c,l)*c3_bbb(b,a,d,j,k,i) &
                            -c1_b(d,l)*c3_bbb(b,c,a,j,k,i) &
                            +2.0d0*(c1_b(a,i)*c1_b(b,j)*c2_bb(c,d,k,l) & !1
                            -c1_b(a,k)*c1_b(b,j)*c2_bb(c,d,i,l) & !(ik)
                            -c1_b(a,l)*c1_b(b,j)*c2_bb(c,d,k,i) & !(il)
                            -c1_b(a,j)*c1_b(b,i)*c2_bb(c,d,k,l) & !(ij)
                            -c1_b(a,i)*c1_b(b,k)*c2_bb(c,d,j,l) & !(jk)
                            -c1_b(a,i)*c1_b(b,l)*c2_bb(c,d,k,j) & !(jl)
                            -c1_b(c,i)*c1_b(b,j)*c2_bb(a,d,k,l) & !(ac)
                            -c1_b(d,i)*c1_b(b,j)*c2_bb(c,a,k,l) & !(ad)
                            -c1_b(a,i)*c1_b(c,j)*c2_bb(b,d,k,l) & !(bc)
                            -c1_b(a,i)*c1_b(d,j)*c2_bb(c,b,k,l) & !(bd)
                            +c1_b(c,i)*c1_b(d,j)*c2_bb(a,b,k,l) & !(ac)(bd)
                            +c1_b(c,k)*c1_b(b,j)*c2_bb(a,d,i,l) & !(ac)(ik)
                            +c1_b(d,k)*c1_b(b,j)*c2_bb(c,a,i,l) & !(ad)(ik)
                            +c1_b(a,k)*c1_b(c,j)*c2_bb(b,d,i,l) & !(bc)(ik)
                            +c1_b(a,k)*c1_b(d,j)*c2_bb(c,b,i,l) & !(bd)(ik)
                            +c1_b(c,l)*c1_b(b,j)*c2_bb(a,d,k,i) & !(ac)(il)
                            +c1_b(d,l)*c1_b(b,j)*c2_bb(c,a,k,i) & !(ad)(il)
                            +c1_b(a,l)*c1_b(c,j)*c2_bb(b,d,k,i) & !(bc)(il)
                            +c1_b(a,l)*c1_b(d,j)*c2_bb(c,b,k,i) & !(bd)(il)
                            +c1_b(c,j)*c1_b(b,i)*c2_bb(a,d,k,l) & !(ac)(ij)
                            +c1_b(d,j)*c1_b(b,i)*c2_bb(c,a,k,l) & !(ad)(ij)
                            +c1_b(a,j)*c1_b(c,i)*c2_bb(b,d,k,l) & !(bc)(ij)
                            +c1_b(a,j)*c1_b(d,i)*c2_bb(c,b,k,l) & !(bd)(ij)
                            +c1_b(c,i)*c1_b(b,k)*c2_bb(a,d,j,l) & !(ac)(jk)
                            +c1_b(d,i)*c1_b(b,k)*c2_bb(c,a,j,l) & !(ad)(jk)
                            +c1_b(a,i)*c1_b(c,k)*c2_bb(b,d,j,l) & !(bc)(jk)
                            +c1_b(a,i)*c1_b(d,k)*c2_bb(c,b,j,l) & !(bd)(jk)
                            +c1_b(c,i)*c1_b(b,l)*c2_bb(a,d,k,j) & !(ac)(jl)
                            +c1_b(d,i)*c1_b(b,l)*c2_bb(c,a,k,j) & !(ad)(jl)
                            +c1_b(a,i)*c1_b(c,l)*c2_bb(b,d,k,j) & !(bc)(jl)
                            +c1_b(a,i)*c1_b(d,l)*c2_bb(c,b,k,j) & !(bd)(jl)
                            +c1_b(a,j)*c1_b(b,k)*c2_bb(c,d,i,l) & !(ijk)
                            +c1_b(a,k)*c1_b(b,i)*c2_bb(c,d,j,l) & !(ikj)
                            +c1_b(a,j)*c1_b(b,l)*c2_bb(c,d,k,i) & !(ijl)
                            +c1_b(a,l)*c1_b(b,i)*c2_bb(c,d,k,j) & !(ilj)
                            -c1_b(a,l)*c1_b(b,k)*c2_bb(c,d,i,j) & !(kilj)
                            -c1_b(c,k)*c1_b(d,j)*c2_bb(a,b,i,l) & !(ac)(bd)(ik)
                            -c1_b(c,l)*c1_b(d,j)*c2_bb(a,b,k,i) & !(ac)(bd)(il)
                            -c1_b(c,j)*c1_b(d,i)*c2_bb(a,b,k,l) & !(ac)(bd)(ij)
                            -c1_b(c,i)*c1_b(d,k)*c2_bb(a,b,j,l) & !(ac)(bd)(jk)
                            -c1_b(c,i)*c1_b(d,l)*c2_bb(a,b,k,j) & !(ac)(bd)(jl)
                            -c1_b(c,j)*c1_b(b,k)*c2_bb(a,d,i,l) & !(ac)(ijk)
                            -c1_b(d,j)*c1_b(b,k)*c2_bb(c,a,i,l) & !(ad)(ijk)
                            -c1_b(a,j)*c1_b(c,k)*c2_bb(b,d,i,l) & !(bc)(ijk)
                            -c1_b(a,j)*c1_b(d,k)*c2_bb(c,b,i,l) & !(bd)(ijk)
                            -c1_b(c,k)*c1_b(b,i)*c2_bb(a,d,j,l) & !(ac)(ikj)
                            -c1_b(d,k)*c1_b(b,i)*c2_bb(c,a,j,l) & !(ad)(ikj)
                            -c1_b(a,k)*c1_b(c,i)*c2_bb(b,d,j,l) & !(bc)(ikj)
                            -c1_b(a,k)*c1_b(d,i)*c2_bb(c,b,j,l) & !(bd)(ikj)
                            -c1_b(c,j)*c1_b(b,l)*c2_bb(a,d,k,i) & !(ac)(ijl)
                            -c1_b(d,j)*c1_b(b,l)*c2_bb(c,a,k,i) & !(ad)(ijl)
                            -c1_b(a,j)*c1_b(c,l)*c2_bb(b,d,k,i) & !(bc)(ijl)
                            -c1_b(a,j)*c1_b(d,l)*c2_bb(c,b,k,i) & !(bd)(ijl)
                            -c1_b(c,l)*c1_b(b,i)*c2_bb(a,d,k,j) & !(ac)(ilj)
                            -c1_b(d,l)*c1_b(b,i)*c2_bb(c,a,k,j) & !(ad)(ilj)
                            -c1_b(a,l)*c1_b(c,i)*c2_bb(b,d,k,j) & !(bc)(ilj)
                            -c1_b(a,l)*c1_b(d,i)*c2_bb(c,b,k,j) & !(bd)(ilj)
                            +c1_b(c,j)*c1_b(d,k)*c2_bb(a,b,i,l) & !(ac)(bd)(ijk)
                            +c1_b(c,k)*c1_b(d,i)*c2_bb(a,b,j,l) & !(ac)(bd)(ikj)
                            +c1_b(c,j)*c1_b(d,l)*c2_bb(a,b,k,i) & !(ac)(bd)(ijl)
                            +c1_b(c,l)*c1_b(d,i)*c2_bb(a,b,k,j) & !(ac)(bd)(ilj)
                            +c1_b(c,l)*c1_b(b,k)*c2_bb(a,d,i,j) & !(ac)(kilj)
                            +c1_b(d,l)*c1_b(b,k)*c2_bb(c,a,i,j) & !(ad)(kilj)
                            +c1_b(a,l)*c1_b(c,k)*c2_bb(b,d,i,j) & !(bc)(kilj)
                            +c1_b(a,l)*c1_b(d,k)*c2_bb(c,b,i,j) & !(bd)(kilj)
                            -c1_b(c,l)*c1_b(d,k)*c2_bb(a,b,i,j) & !(ac)(bd)(kilj)
                            -c1_b(c,k)*c1_b(b,l)*c2_bb(a,d,i,j) & !(ac)(ik)(jl)
                            -c1_b(d,k)*c1_b(b,l)*c2_bb(c,a,i,j) & !(ad)(ik)(jl)
                            -c1_b(a,k)*c1_b(c,l)*c2_bb(b,d,i,j) & !(bc)(ik)(jl)
                            -c1_b(a,k)*c1_b(d,l)*c2_bb(c,b,i,j) & !(bd)(ik)(jl)
                            +c1_b(c,k)*c1_b(d,l)*c2_bb(a,b,i,j) & !(ac)(bd)(ik)(jl)
                            +c1_b(a,k)*c1_b(b,l)*c2_bb(c,d,i,j)) & !(ik)(jl)
                            -c2_bb(a,b,i,j)*c2_bb(c,d,k,l) &
                            +c2_bb(a,b,k,j)*c2_bb(c,d,i,l) &
                            +c2_bb(a,b,l,j)*c2_bb(c,d,k,i) &
                            +c2_bb(a,b,i,k)*c2_bb(c,d,j,l) &
                            +c2_bb(a,b,i,l)*c2_bb(c,d,k,j) &
                            +c2_bb(c,b,i,j)*c2_bb(a,d,k,l) &
                            +c2_bb(d,b,i,j)*c2_bb(c,a,k,l) &
                            -c2_bb(a,b,k,l)*c2_bb(c,d,i,j) &
                            -c2_bb(c,b,k,j)*c2_bb(a,d,i,l) &
                            -c2_bb(c,b,l,j)*c2_bb(a,d,k,i) &
                            -c2_bb(c,b,i,k)*c2_bb(a,d,j,l) &
                            -c2_bb(c,b,i,l)*c2_bb(a,d,k,j) &
                            -c2_bb(d,b,k,j)*c2_bb(c,a,i,l) &
                            -c2_bb(d,b,l,j)*c2_bb(c,a,k,i) &
                            -c2_bb(d,b,i,k)*c2_bb(c,a,j,l) &
                            -c2_bb(d,b,i,l)*c2_bb(c,a,k,j) &
                            +c2_bb(c,b,k,l)*c2_bb(a,d,i,j) &
                            +c2_bb(d,b,k,l)*c2_bb(c,a,i,j) &
                            -6.0d0*(c1_b(a,i)*c1_b(b,j)*c1_b(c,k)*c1_b(d,l) &
                            -c1_b(a,j)*c1_b(b,i)*c1_b(c,k)*c1_b(d,l) &
                            -c1_b(a,k)*c1_b(b,j)*c1_b(c,i)*c1_b(d,l) &
                            -c1_b(a,l)*c1_b(b,j)*c1_b(c,k)*c1_b(d,i) &
                            -c1_b(a,i)*c1_b(b,k)*c1_b(c,j)*c1_b(d,l) &
                            -c1_b(a,i)*c1_b(b,l)*c1_b(c,k)*c1_b(d,j) &
                            -c1_b(a,i)*c1_b(b,j)*c1_b(c,l)*c1_b(d,k) &
                            +c1_b(a,j)*c1_b(b,k)*c1_b(c,i)*c1_b(d,l) &
                            +c1_b(a,k)*c1_b(b,i)*c1_b(c,j)*c1_b(d,l) &
                            +c1_b(a,j)*c1_b(b,l)*c1_b(c,k)*c1_b(d,i) &
                            +c1_b(a,l)*c1_b(b,i)*c1_b(c,k)*c1_b(d,j) &
                            +c1_b(a,k)*c1_b(b,j)*c1_b(c,l)*c1_b(d,i) &
                            +c1_b(a,l)*c1_b(b,j)*c1_b(c,i)*c1_b(d,k) &
                            +c1_b(a,i)*c1_b(b,k)*c1_b(c,l)*c1_b(d,j) &
                            +c1_b(a,i)*c1_b(b,l)*c1_b(c,j)*c1_b(d,k) &
                            +c1_b(a,j)*c1_b(b,i)*c1_b(c,l)*c1_b(d,k) &
                            +c1_b(a,k)*c1_b(b,l)*c1_b(c,i)*c1_b(d,j) &
                            +c1_b(a,l)*c1_b(b,k)*c1_b(c,j)*c1_b(d,i) &
                            -c1_b(a,j)*c1_b(b,k)*c1_b(c,l)*c1_b(d,i) &
                            -c1_b(a,j)*c1_b(b,l)*c1_b(c,i)*c1_b(d,k) &
                            -c1_b(a,k)*c1_b(b,l)*c1_b(c,j)*c1_b(d,i) &
                            -c1_b(a,k)*c1_b(b,i)*c1_b(c,l)*c1_b(d,j) &
                            -c1_b(a,l)*c1_b(b,i)*c1_b(c,j)*c1_b(d,k) &
                            -c1_b(a,l)*c1_b(b,k)*c1_b(c,i)*c1_b(d,j)))
                    !else
                    !    t4_bbbb = 0.0d0
                    !end if
                    t4_bbbb(a, b, c, d, i, j, l, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, c, d, i, k, j, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, c, d, i, k, l, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, c, d, i, l, j, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, c, d, i, l, k, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, c, d, j, i, k, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, c, d, j, i, l, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, c, d, j, k, i, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, c, d, j, k, l, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, c, d, j, l, i, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, c, d, j, l, k, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, c, d, k, i, j, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, c, d, k, i, l, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, c, d, k, j, i, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, c, d, k, j, l, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, c, d, k, l, i, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, c, d, k, l, j, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, c, d, l, i, j, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, c, d, l, i, k, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, c, d, l, j, i, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, c, d, l, j, k, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, c, d, l, k, i, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, c, d, l, k, j, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, d, c, i, j, k, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, d, c, i, j, l, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, d, c, i, k, j, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, d, c, i, k, l, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, d, c, i, l, j, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, d, c, i, l, k, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, d, c, j, i, k, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, d, c, j, i, l, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, d, c, j, k, i, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, d, c, j, k, l, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, d, c, j, l, i, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, d, c, j, l, k, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, d, c, k, i, j, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, d, c, k, i, l, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, d, c, k, j, i, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, d, c, k, j, l, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, d, c, k, l, i, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, d, c, k, l, j, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, d, c, l, i, j, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, d, c, l, i, k, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, d, c, l, j, i, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, d, c, l, j, k, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, d, c, l, k, i, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, b, d, c, l, k, j, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, b, d, i, j, k, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, b, d, i, j, l, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, b, d, i, k, j, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, b, d, i, k, l, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, b, d, i, l, j, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, b, d, i, l, k, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, b, d, j, i, k, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, b, d, j, i, l, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, b, d, j, k, i, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, b, d, j, k, l, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, b, d, j, l, i, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, b, d, j, l, k, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, b, d, k, i, j, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, b, d, k, i, l, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, b, d, k, j, i, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, b, d, k, j, l, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, b, d, k, l, i, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, b, d, k, l, j, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, b, d, l, i, j, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, b, d, l, i, k, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, b, d, l, j, i, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, b, d, l, j, k, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, b, d, l, k, i, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, b, d, l, k, j, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, d, b, i, j, k, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, d, b, i, j, l, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, d, b, i, k, j, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, d, b, i, k, l, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, d, b, i, l, j, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, d, b, i, l, k, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, d, b, j, i, k, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, d, b, j, i, l, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, d, b, j, k, i, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, d, b, j, k, l, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, d, b, j, l, i, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, d, b, j, l, k, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, d, b, k, i, j, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, d, b, k, i, l, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, d, b, k, j, i, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, d, b, k, j, l, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, d, b, k, l, i, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, d, b, k, l, j, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, d, b, l, i, j, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, d, b, l, i, k, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, d, b, l, j, i, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, d, b, l, j, k, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, d, b, l, k, i, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, c, d, b, l, k, j, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, b, c, i, j, k, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, b, c, i, j, l, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, b, c, i, k, j, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, b, c, i, k, l, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, b, c, i, l, j, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, b, c, i, l, k, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, b, c, j, i, k, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, b, c, j, i, l, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, b, c, j, k, i, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, b, c, j, k, l, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, b, c, j, l, i, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, b, c, j, l, k, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, b, c, k, i, j, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, b, c, k, i, l, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, b, c, k, j, i, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, b, c, k, j, l, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, b, c, k, l, i, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, b, c, k, l, j, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, b, c, l, i, j, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, b, c, l, i, k, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, b, c, l, j, i, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, b, c, l, j, k, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, b, c, l, k, i, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, b, c, l, k, j, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, c, b, i, j, k, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, c, b, i, j, l, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, c, b, i, k, j, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, c, b, i, k, l, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, c, b, i, l, j, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, c, b, i, l, k, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, c, b, j, i, k, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, c, b, j, i, l, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, c, b, j, k, i, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, c, b, j, k, l, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, c, b, j, l, i, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, c, b, j, l, k, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, c, b, k, i, j, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, c, b, k, i, l, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, c, b, k, j, i, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, c, b, k, j, l, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, c, b, k, l, i, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, c, b, k, l, j, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, c, b, l, i, j, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, c, b, l, i, k, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, c, b, l, j, i, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, c, b, l, j, k, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, c, b, l, k, i, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(a, d, c, b, l, k, j, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, c, d, i, j, k, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, c, d, i, j, l, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, c, d, i, k, j, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, c, d, i, k, l, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, c, d, i, l, j, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, c, d, i, l, k, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, c, d, j, i, k, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, c, d, j, i, l, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, c, d, j, k, i, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, c, d, j, k, l, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, c, d, j, l, i, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, c, d, j, l, k, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, c, d, k, i, j, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, c, d, k, i, l, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, c, d, k, j, i, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, c, d, k, j, l, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, c, d, k, l, i, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, c, d, k, l, j, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, c, d, l, i, j, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, c, d, l, i, k, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, c, d, l, j, i, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, c, d, l, j, k, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, c, d, l, k, i, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, c, d, l, k, j, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, d, c, i, j, k, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, d, c, i, j, l, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, d, c, i, k, j, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, d, c, i, k, l, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, d, c, i, l, j, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, d, c, i, l, k, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, d, c, j, i, k, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, d, c, j, i, l, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, d, c, j, k, i, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, d, c, j, k, l, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, d, c, j, l, i, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, d, c, j, l, k, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, d, c, k, i, j, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, d, c, k, i, l, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, d, c, k, j, i, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, d, c, k, j, l, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, d, c, k, l, i, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, d, c, k, l, j, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, d, c, l, i, j, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, d, c, l, i, k, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, d, c, l, j, i, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, d, c, l, j, k, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, d, c, l, k, i, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, a, d, c, l, k, j, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, a, d, i, j, k, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, a, d, i, j, l, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, a, d, i, k, j, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, a, d, i, k, l, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, a, d, i, l, j, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, a, d, i, l, k, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, a, d, j, i, k, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, a, d, j, i, l, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, a, d, j, k, i, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, a, d, j, k, l, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, a, d, j, l, i, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, a, d, j, l, k, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, a, d, k, i, j, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, a, d, k, i, l, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, a, d, k, j, i, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, a, d, k, j, l, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, a, d, k, l, i, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, a, d, k, l, j, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, a, d, l, i, j, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, a, d, l, i, k, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, a, d, l, j, i, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, a, d, l, j, k, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, a, d, l, k, i, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, a, d, l, k, j, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, d, a, i, j, k, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, d, a, i, j, l, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, d, a, i, k, j, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, d, a, i, k, l, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, d, a, i, l, j, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, d, a, i, l, k, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, d, a, j, i, k, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, d, a, j, i, l, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, d, a, j, k, i, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, d, a, j, k, l, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, d, a, j, l, i, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, d, a, j, l, k, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, d, a, k, i, j, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, d, a, k, i, l, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, d, a, k, j, i, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, d, a, k, j, l, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, d, a, k, l, i, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, d, a, k, l, j, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, d, a, l, i, j, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, d, a, l, i, k, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, d, a, l, j, i, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, d, a, l, j, k, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, d, a, l, k, i, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, c, d, a, l, k, j, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, a, c, i, j, k, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, a, c, i, j, l, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, a, c, i, k, j, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, a, c, i, k, l, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, a, c, i, l, j, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, a, c, i, l, k, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, a, c, j, i, k, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, a, c, j, i, l, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, a, c, j, k, i, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, a, c, j, k, l, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, a, c, j, l, i, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, a, c, j, l, k, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, a, c, k, i, j, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, a, c, k, i, l, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, a, c, k, j, i, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, a, c, k, j, l, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, a, c, k, l, i, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, a, c, k, l, j, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, a, c, l, i, j, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, a, c, l, i, k, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, a, c, l, j, i, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, a, c, l, j, k, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, a, c, l, k, i, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, a, c, l, k, j, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, c, a, i, j, k, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, c, a, i, j, l, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, c, a, i, k, j, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, c, a, i, k, l, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, c, a, i, l, j, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, c, a, i, l, k, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, c, a, j, i, k, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, c, a, j, i, l, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, c, a, j, k, i, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, c, a, j, k, l, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, c, a, j, l, i, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, c, a, j, l, k, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, c, a, k, i, j, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, c, a, k, i, l, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, c, a, k, j, i, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, c, a, k, j, l, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, c, a, k, l, i, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, c, a, k, l, j, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, c, a, l, i, j, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, c, a, l, i, k, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, c, a, l, j, i, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, c, a, l, j, k, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, c, a, l, k, i, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(b, d, c, a, l, k, j, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, b, d, i, j, k, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, b, d, i, j, l, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, b, d, i, k, j, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, b, d, i, k, l, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, b, d, i, l, j, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, b, d, i, l, k, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, b, d, j, i, k, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, b, d, j, i, l, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, b, d, j, k, i, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, b, d, j, k, l, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, b, d, j, l, i, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, b, d, j, l, k, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, b, d, k, i, j, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, b, d, k, i, l, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, b, d, k, j, i, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, b, d, k, j, l, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, b, d, k, l, i, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, b, d, k, l, j, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, b, d, l, i, j, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, b, d, l, i, k, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, b, d, l, j, i, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, b, d, l, j, k, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, b, d, l, k, i, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, b, d, l, k, j, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, d, b, i, j, k, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, d, b, i, j, l, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, d, b, i, k, j, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, d, b, i, k, l, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, d, b, i, l, j, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, d, b, i, l, k, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, d, b, j, i, k, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, d, b, j, i, l, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, d, b, j, k, i, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, d, b, j, k, l, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, d, b, j, l, i, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, d, b, j, l, k, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, d, b, k, i, j, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, d, b, k, i, l, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, d, b, k, j, i, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, d, b, k, j, l, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, d, b, k, l, i, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, d, b, k, l, j, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, d, b, l, i, j, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, d, b, l, i, k, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, d, b, l, j, i, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, d, b, l, j, k, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, d, b, l, k, i, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, a, d, b, l, k, j, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, a, d, i, j, k, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, a, d, i, j, l, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, a, d, i, k, j, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, a, d, i, k, l, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, a, d, i, l, j, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, a, d, i, l, k, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, a, d, j, i, k, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, a, d, j, i, l, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, a, d, j, k, i, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, a, d, j, k, l, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, a, d, j, l, i, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, a, d, j, l, k, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, a, d, k, i, j, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, a, d, k, i, l, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, a, d, k, j, i, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, a, d, k, j, l, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, a, d, k, l, i, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, a, d, k, l, j, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, a, d, l, i, j, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, a, d, l, i, k, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, a, d, l, j, i, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, a, d, l, j, k, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, a, d, l, k, i, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, a, d, l, k, j, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, d, a, i, j, k, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, d, a, i, j, l, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, d, a, i, k, j, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, d, a, i, k, l, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, d, a, i, l, j, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, d, a, i, l, k, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, d, a, j, i, k, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, d, a, j, i, l, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, d, a, j, k, i, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, d, a, j, k, l, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, d, a, j, l, i, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, d, a, j, l, k, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, d, a, k, i, j, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, d, a, k, i, l, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, d, a, k, j, i, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, d, a, k, j, l, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, d, a, k, l, i, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, d, a, k, l, j, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, d, a, l, i, j, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, d, a, l, i, k, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, d, a, l, j, i, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, d, a, l, j, k, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, d, a, l, k, i, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, b, d, a, l, k, j, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, a, b, i, j, k, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, a, b, i, j, l, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, a, b, i, k, j, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, a, b, i, k, l, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, a, b, i, l, j, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, a, b, i, l, k, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, a, b, j, i, k, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, a, b, j, i, l, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, a, b, j, k, i, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, a, b, j, k, l, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, a, b, j, l, i, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, a, b, j, l, k, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, a, b, k, i, j, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, a, b, k, i, l, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, a, b, k, j, i, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, a, b, k, j, l, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, a, b, k, l, i, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, a, b, k, l, j, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, a, b, l, i, j, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, a, b, l, i, k, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, a, b, l, j, i, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, a, b, l, j, k, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, a, b, l, k, i, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, a, b, l, k, j, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, b, a, i, j, k, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, b, a, i, j, l, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, b, a, i, k, j, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, b, a, i, k, l, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, b, a, i, l, j, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, b, a, i, l, k, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, b, a, j, i, k, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, b, a, j, i, l, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, b, a, j, k, i, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, b, a, j, k, l, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, b, a, j, l, i, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, b, a, j, l, k, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, b, a, k, i, j, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, b, a, k, i, l, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, b, a, k, j, i, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, b, a, k, j, l, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, b, a, k, l, i, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, b, a, k, l, j, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, b, a, l, i, j, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, b, a, l, i, k, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, b, a, l, j, i, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, b, a, l, j, k, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, b, a, l, k, i, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(c, d, b, a, l, k, j, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, b, c, i, j, k, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, b, c, i, j, l, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, b, c, i, k, j, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, b, c, i, k, l, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, b, c, i, l, j, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, b, c, i, l, k, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, b, c, j, i, k, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, b, c, j, i, l, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, b, c, j, k, i, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, b, c, j, k, l, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, b, c, j, l, i, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, b, c, j, l, k, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, b, c, k, i, j, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, b, c, k, i, l, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, b, c, k, j, i, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, b, c, k, j, l, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, b, c, k, l, i, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, b, c, k, l, j, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, b, c, l, i, j, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, b, c, l, i, k, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, b, c, l, j, i, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, b, c, l, j, k, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, b, c, l, k, i, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, b, c, l, k, j, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, c, b, i, j, k, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, c, b, i, j, l, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, c, b, i, k, j, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, c, b, i, k, l, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, c, b, i, l, j, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, c, b, i, l, k, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, c, b, j, i, k, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, c, b, j, i, l, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, c, b, j, k, i, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, c, b, j, k, l, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, c, b, j, l, i, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, c, b, j, l, k, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, c, b, k, i, j, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, c, b, k, i, l, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, c, b, k, j, i, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, c, b, k, j, l, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, c, b, k, l, i, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, c, b, k, l, j, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, c, b, l, i, j, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, c, b, l, i, k, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, c, b, l, j, i, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, c, b, l, j, k, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, c, b, l, k, i, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, a, c, b, l, k, j, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, a, c, i, j, k, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, a, c, i, j, l, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, a, c, i, k, j, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, a, c, i, k, l, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, a, c, i, l, j, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, a, c, i, l, k, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, a, c, j, i, k, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, a, c, j, i, l, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, a, c, j, k, i, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, a, c, j, k, l, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, a, c, j, l, i, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, a, c, j, l, k, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, a, c, k, i, j, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, a, c, k, i, l, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, a, c, k, j, i, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, a, c, k, j, l, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, a, c, k, l, i, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, a, c, k, l, j, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, a, c, l, i, j, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, a, c, l, i, k, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, a, c, l, j, i, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, a, c, l, j, k, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, a, c, l, k, i, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, a, c, l, k, j, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, c, a, i, j, k, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, c, a, i, j, l, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, c, a, i, k, j, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, c, a, i, k, l, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, c, a, i, l, j, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, c, a, i, l, k, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, c, a, j, i, k, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, c, a, j, i, l, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, c, a, j, k, i, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, c, a, j, k, l, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, c, a, j, l, i, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, c, a, j, l, k, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, c, a, k, i, j, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, c, a, k, i, l, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, c, a, k, j, i, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, c, a, k, j, l, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, c, a, k, l, i, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, c, a, k, l, j, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, c, a, l, i, j, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, c, a, l, i, k, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, c, a, l, j, i, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, c, a, l, j, k, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, c, a, l, k, i, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, b, c, a, l, k, j, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, a, b, i, j, k, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, a, b, i, j, l, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, a, b, i, k, j, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, a, b, i, k, l, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, a, b, i, l, j, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, a, b, i, l, k, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, a, b, j, i, k, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, a, b, j, i, l, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, a, b, j, k, i, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, a, b, j, k, l, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, a, b, j, l, i, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, a, b, j, l, k, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, a, b, k, i, j, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, a, b, k, i, l, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, a, b, k, j, i, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, a, b, k, j, l, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, a, b, k, l, i, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, a, b, k, l, j, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, a, b, l, i, j, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, a, b, l, i, k, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, a, b, l, j, i, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, a, b, l, j, k, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, a, b, l, k, i, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, a, b, l, k, j, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, b, a, i, j, k, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, b, a, i, j, l, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, b, a, i, k, j, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, b, a, i, k, l, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, b, a, i, l, j, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, b, a, i, l, k, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, b, a, j, i, k, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, b, a, j, i, l, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, b, a, j, k, i, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, b, a, j, k, l, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, b, a, j, l, i, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, b, a, j, l, k, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, b, a, k, i, j, l) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, b, a, k, i, l, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, b, a, k, j, i, l) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, b, a, k, j, l, i) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, b, a, k, l, i, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, b, a, k, l, j, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, b, a, l, i, j, k) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, b, a, l, i, k, j) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, b, a, l, j, i, k) = t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, b, a, l, j, k, i) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, b, a, l, k, i, j) = -1.0 * t4_bbbb(a, b, c, d, i, j, k, l)
                    t4_bbbb(d, c, b, a, l, k, j, i) = t4_bbbb(a, b, c, d, i, j, k, l)

                enddo

        end subroutine cluster_analysis_t4

end module clusteranalysis

