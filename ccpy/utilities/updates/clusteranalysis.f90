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
                                    do k = j+1, noa
                                        if (abs(c3a(a, b, c, i, j, k)) > 0.0d0) then
                                            t3a(a, b, c, i, j, k) = c3a(a, b, c, i, j, k)&
                                                                   -c1a(a, i) * c2a(b, c, j, k)&
                                                                   +c1a(b, i) * c2a(a, c, j, k)&
                                                                   +c1a(c, i) * c2a(b, a, j, k)&
                                                                   -c1a(b, j) * c2a(a, c, i, k)&
                                                                   +c1a(a, j) * c2a(b, c, i, k)&
                                                                   +c1a(c, j) * c2a(b, a, i, k)&
                                                                   -c1a(c, k) * c2a(b, a, j, i)&
                                                                   +c2a(a, k) * c2a(b, c, j, i)&
                                                                   +c2a(b, k) * c2a(c, a, j, i)&
                                                                   +(1.0/3.0) * c1a(a, i) * c1a(b, j) * c1a(c, k)&
                                                                   +(1.0/3.0) * c1a(a, j) * c1a(b, k) * c1a(c, i)&
                                                                   +(1.0/3.0) * c1a(a, k) * c1a(b, i) * c1a(c, j)&
                                                                   -(1.0/3.0) * c1a(a, i) * c1a(b, k) * c1a(c, j)&
                                                                   -(1.0/3.0) * c1a(a, j) * c1a(b, i) * c1a(c, k)&
                                                                   -(1.0/3.0) * c1a(a, k) * c1a(b, j) * c1a(c, i)
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




        end subroutine cluster_analysis_t3

end module clusteranalysis