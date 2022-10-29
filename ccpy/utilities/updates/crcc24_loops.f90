module crcc24_loops

    implicit none

    contains

              !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CR-CC(2,4) ROUTINES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

              subroutine crcc24A(deltaA,deltaB,deltaC,deltaD,&
                              t2a,l2a,&
                              fA_oo,fA_vv,H1A_oo,H1A_vv,&
                              H2A_voov,H2A_oooo,H2A_vvvv,H2A_oovv,&
                              D3A_O,D3A_V,&
                              noa,nua)

                        real(kind=8), intent(out) :: deltaA, deltaB, deltaC, deltaD
                        integer, intent(in) :: noa, nua
                        real(kind=8), intent(in) :: t2a(1:nua,1:nua,1:noa,1:noa),l2a(1:nua,1:nua,1:noa,1:noa),&
                        fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),&
                        H1A_oo(1:noa,1:noa),H1A_vv(1:nua,1:nua),&
                        H2A_voov(1:nua,1:noa,1:noa,1:nua),&
                        H2A_oooo(1:noa,1:noa,1:noa,1:noa),&
                        H2A_vvvv(1:nua,1:nua,1:nua,1:nua),&
                        H2A_oovv(1:noa,1:noa,1:nua,1:nua),&
                        D3A_O(1:nua,1:noa,1:noa),&
                        D3A_V(1:nua,1:noa,1:nua)

                        !real(kind=8), intent(in) :: test_array(1:nua,1:nua,1:nua,1:nua,1:noa,1:noa,1:noa,1:noa)

                        integer :: i, j, k, l, a, b, c, d, m, n, e, f
                        real(kind=8) :: denom, temp, mm24, l4

                        !real(kind=8) :: error

                        deltaA = 0.0d0
                        deltaB = 0.0d0
                        deltaC = 0.0d0
                        deltaD = 0.0d0

                        !error = 0.0d0
                        do i = 1, noa
                            do j = i+1, noa
                                do k = j+1, noa
                                    do l = k+1, noa
                                        do a = 1, nua
                                            do b = a+1, nua
                                                do c = b+1, nua
                                                    do d = c+1, nua

                                                        mm24 = 0.0d0
                                                        l4 = 0.0d0

                                                        !!! MM(2,4)A Computation !!!
                                                        ! Diagram 1: -A(jl/i/k)A(bc/a/d) h2a(amie) * t2a(bcmk) * t2a(edjl)
                                                        do m = 1, noa
                                                            do e = 1, nua
                                                                ! (1)
                                                                mm24 =  mm24 &
                                                                        -h2a_voov(a, m, i, e) * t2a(b, c, m, k) * t2a(e, d, j, l)&  ! (1)
                                                                        +h2a_voov(a, m, j, e) * t2a(b, c, m, k) * t2a(e, d, i, l)&  ! (ij)
                                                                        +h2a_voov(a, m, i, e) * t2a(b, c, m, j) * t2a(e, d, k, l)&  ! (jk)
                                                                        +h2a_voov(a, m, l, e) * t2a(b, c, m, k) * t2a(e, d, j, i)&  ! (il)
                                                                        +h2a_voov(a, m, i, e) * t2a(b, c, m, l) * t2a(e, d, j, k)&  ! (kl)
                                                                        -h2a_voov(a, m, j, e) * t2a(b, c, m, l) * t2a(e, d, i, k)&  ! (ij)(kl)
                                                                        +h2a_voov(a, m, k, e) * t2a(b, c, m, i) * t2a(e, d, j, l)&  ! (ik)
                                                                        -h2a_voov(a, m, j, e) * t2a(b, c, m, i) * t2a(e, d, k, l)&  ! (ij)(ik)
                                                                        -h2a_voov(a, m, k, e) * t2a(b, c, m, j) * t2a(e, d, i, l)&  ! (jk)(ik)
                                                                        -h2a_voov(a, m, l, e) * t2a(b, c, m, i) * t2a(e, d, j, k)&  ! (il)(ik)
                                                                        -h2a_voov(a, m, k, e) * t2a(b, c, m, l) * t2a(e, d, j, i)&  ! (kl)(ik)
                                                                        +h2a_voov(a, m, j, e) * t2a(b, c, m, l) * t2a(e, d, k, i)   ! (ij)(kl)(ik)
                                                                ! (ab)
                                                                mm24 =  mm24 + h2a_voov(b, m, i, e) * t2a(a, c, m, k) * t2a(e, d, j, l)&  ! (1)
                                                                        -h2a_voov(b, m, j, e) * t2a(a, c, m, k) * t2a(e, d, i, l)&        ! (ij)
                                                                        -h2a_voov(b, m, i, e) * t2a(a, c, m, j) * t2a(e, d, k, l)&        ! (jk)
                                                                        -h2a_voov(b, m, l, e) * t2a(a, c, m, k) * t2a(e, d, j, i)&        ! (il)
                                                                        -h2a_voov(b, m, i, e) * t2a(a, c, m, l) * t2a(e, d, j, k)&        ! (kl)
                                                                        +h2a_voov(b, m, j, e) * t2a(a, c, m, l) * t2a(e, d, i, k)&        ! (ij)(kl)
                                                                        -h2a_voov(b, m, k, e) * t2a(a, c, m, i) * t2a(e, d, j, l)&        ! (ik)
                                                                        +h2a_voov(b, m, j, e) * t2a(a, c, m, i) * t2a(e, d, k, l)&        ! (ij)(ik)
                                                                        +h2a_voov(b, m, k, e) * t2a(a, c, m, j) * t2a(e, d, i, l)&        ! (jk)(ik)
                                                                        +h2a_voov(b, m, l, e) * t2a(a, c, m, i) * t2a(e, d, j, k)&        ! (il)(ik)
                                                                        +h2a_voov(b, m, k, e) * t2a(a, c, m, l) * t2a(e, d, j, i)&        ! (kl)(ik)
                                                                        -h2a_voov(b, m, j, e) * t2a(a, c, m, l) * t2a(e, d, k, i)         ! (ij)(kl)(ik)
                                                                ! (bd)
                                                                mm24 =  mm24 + h2a_voov(a, m, i, e) * t2a(d, c, m, k) * t2a(e, b, j, l)&  ! (1)
                                                                        -h2a_voov(a, m, j, e) * t2a(d, c, m, k) * t2a(e, b, i, l)&        ! (ij)
                                                                        -h2a_voov(a, m, i, e) * t2a(d, c, m, j) * t2a(e, b, k, l)&        ! (jk)
                                                                        -h2a_voov(a, m, l, e) * t2a(d, c, m, k) * t2a(e, b, j, i)&        ! (il)
                                                                        -h2a_voov(a, m, i, e) * t2a(d, c, m, l) * t2a(e, b, j, k)&        ! (kl)
                                                                        +h2a_voov(a, m, j, e) * t2a(d, c, m, l) * t2a(e, b, i, k)&        ! (ij)(kl)
                                                                        -h2a_voov(a, m, k, e) * t2a(d, c, m, i) * t2a(e, b, j, l)&        ! (ik)
                                                                        +h2a_voov(a, m, j, e) * t2a(d, c, m, i) * t2a(e, b, k, l)&        ! (ij)(ik)
                                                                        +h2a_voov(a, m, k, e) * t2a(d, c, m, j) * t2a(e, b, i, l)&        ! (jk)(ik)
                                                                        +h2a_voov(a, m, l, e) * t2a(d, c, m, i) * t2a(e, b, j, k)&        ! (il)(ik)
                                                                        +h2a_voov(a, m, k, e) * t2a(d, c, m, l) * t2a(e, b, j, i)&        ! (kl)(ik)
                                                                        -h2a_voov(a, m, j, e) * t2a(d, c, m, l) * t2a(e, b, k, i)         ! (ij)(kl)(ik)
                                                                ! (ac)
                                                                mm24 =  mm24  + h2a_voov(c, m, i, e) * t2a(b, a, m, k) * t2a(e, d, j, l)&  ! (1)
                                                                        -h2a_voov(c, m, j, e) * t2a(b, a, m, k) * t2a(e, d, i, l)&         ! (ij)
                                                                        -h2a_voov(c, m, i, e) * t2a(b, a, m, j) * t2a(e, d, k, l)&         ! (jk)
                                                                        -h2a_voov(c, m, l, e) * t2a(b, a, m, k) * t2a(e, d, j, i)&         ! (il)
                                                                        -h2a_voov(c, m, i, e) * t2a(b, a, m, l) * t2a(e, d, j, k)&         ! (kl)
                                                                        +h2a_voov(c, m, j, e) * t2a(b, a, m, l) * t2a(e, d, i, k)&         ! (ij)(kl)
                                                                        -h2a_voov(c, m, k, e) * t2a(b, a, m, i) * t2a(e, d, j, l)&         ! (ik)
                                                                        +h2a_voov(c, m, j, e) * t2a(b, a, m, i) * t2a(e, d, k, l)&         ! (ij)(ik)
                                                                        +h2a_voov(c, m, k, e) * t2a(b, a, m, j) * t2a(e, d, i, l)&         ! (jk)(ik)
                                                                        +h2a_voov(c, m, l, e) * t2a(b, a, m, i) * t2a(e, d, j, k)&         ! (il)(ik)
                                                                        +h2a_voov(c, m, k, e) * t2a(b, a, m, l) * t2a(e, d, j, i)&         ! (kl)(ik)
                                                                        -h2a_voov(c, m, j, e) * t2a(b, a, m, l) * t2a(e, d, k, i)          ! (ij)(kl)(ik)
                                                                ! (cd)
                                                                mm24 =  mm24 + h2a_voov(a, m, i, e) * t2a(b, d, m, k) * t2a(e, c, j, l)&  ! (1)
                                                                        -h2a_voov(a, m, j, e) * t2a(b, d, m, k) * t2a(e, c, i, l)&        ! (ij)
                                                                        -h2a_voov(a, m, i, e) * t2a(b, d, m, j) * t2a(e, c, k, l)&        ! (jk)
                                                                        -h2a_voov(a, m, l, e) * t2a(b, d, m, k) * t2a(e, c, j, i)&        ! (il)
                                                                        -h2a_voov(a, m, i, e) * t2a(b, d, m, l) * t2a(e, c, j, k)&        ! (kl)
                                                                        +h2a_voov(a, m, j, e) * t2a(b, d, m, l) * t2a(e, c, i, k)&        ! (ij)(kl)
                                                                        -h2a_voov(a, m, k, e) * t2a(b, d, m, i) * t2a(e, c, j, l)&        ! (ik)
                                                                        +h2a_voov(a, m, j, e) * t2a(b, d, m, i) * t2a(e, c, k, l)&        ! (ij)(ik)
                                                                        +h2a_voov(a, m, k, e) * t2a(b, d, m, j) * t2a(e, c, i, l)&        ! (jk)(ik)
                                                                        +h2a_voov(a, m, l, e) * t2a(b, d, m, i) * t2a(e, c, j, k)&        ! (il)(ik)
                                                                        +h2a_voov(a, m, k, e) * t2a(b, d, m, l) * t2a(e, c, j, i)&        ! (kl)(ik)
                                                                        -h2a_voov(a, m, j, e) * t2a(b, d, m, l) * t2a(e, c, k, i)         ! (ij)(kl)(ik)
                                                                ! (ab)(cd)
                                                                mm24 =  mm24 - h2a_voov(b, m, i, e) * t2a(a, d, m, k) * t2a(e, c, j, l)&  ! (1)
                                                                        +h2a_voov(b, m, j, e) * t2a(a, d, m, k) * t2a(e, c, i, l)&        ! (ij)
                                                                        +h2a_voov(b, m, i, e) * t2a(a, d, m, j) * t2a(e, c, k, l)&        ! (jk)
                                                                        +h2a_voov(b, m, l, e) * t2a(a, d, m, k) * t2a(e, c, j, i)&        ! (il)
                                                                        +h2a_voov(b, m, i, e) * t2a(a, d, m, l) * t2a(e, c, j, k)&        ! (kl)
                                                                        -h2a_voov(b, m, j, e) * t2a(a, d, m, l) * t2a(e, c, i, k)&        ! (ij)(kl)
                                                                        +h2a_voov(b, m, k, e) * t2a(a, d, m, i) * t2a(e, c, j, l)&        ! (ik)
                                                                        -h2a_voov(b, m, j, e) * t2a(a, d, m, i) * t2a(e, c, k, l)&        ! (ij)(ik)
                                                                        -h2a_voov(b, m, k, e) * t2a(a, d, m, j) * t2a(e, c, i, l)&        ! (jk)(ik)
                                                                        -h2a_voov(b, m, l, e) * t2a(a, d, m, i) * t2a(e, c, j, k)&        ! (il)(ik)
                                                                        -h2a_voov(b, m, k, e) * t2a(a, d, m, l) * t2a(e, c, j, i)&        ! (kl)(ik)
                                                                        +h2a_voov(b, m, j, e) * t2a(a, d, m, l) * t2a(e, c, k, i)         ! (ij)(kl)(ik)
                                                                ! (ad)
                                                                mm24 =  mm24 + h2a_voov(d, m, i, e) * t2a(b, c, m, k) * t2a(e, a, j, l)&  ! (1)
                                                                        -h2a_voov(d, m, j, e) * t2a(b, c, m, k) * t2a(e, a, i, l)&        ! (ij)
                                                                        -h2a_voov(d, m, i, e) * t2a(b, c, m, j) * t2a(e, a, k, l)&        ! (jk)
                                                                        -h2a_voov(d, m, l, e) * t2a(b, c, m, k) * t2a(e, a, j, i)&        ! (il)
                                                                        -h2a_voov(d, m, i, e) * t2a(b, c, m, l) * t2a(e, a, j, k)&        ! (kl)
                                                                        +h2a_voov(d, m, j, e) * t2a(b, c, m, l) * t2a(e, a, i, k)&        ! (ij)(kl)
                                                                        -h2a_voov(d, m, k, e) * t2a(b, c, m, i) * t2a(e, a, j, l)&        ! (ik)
                                                                        +h2a_voov(d, m, j, e) * t2a(b, c, m, i) * t2a(e, a, k, l)&        ! (ij)(ik)
                                                                        +h2a_voov(d, m, k, e) * t2a(b, c, m, j) * t2a(e, a, i, l)&        ! (jk)(ik)
                                                                        +h2a_voov(d, m, l, e) * t2a(b, c, m, i) * t2a(e, a, j, k)&        ! (il)(ik)
                                                                        +h2a_voov(d, m, k, e) * t2a(b, c, m, l) * t2a(e, a, j, i)&        ! (kl)(ik)
                                                                        -h2a_voov(d, m, j, e) * t2a(b, c, m, l) * t2a(e, a, k, i)         ! (ij)(kl)(ik)
                                                                ! (ab)(ad)
                                                                mm24 =  mm24 - h2a_voov(b, m, i, e) * t2a(d, c, m, k) * t2a(e, a, j, l)&  ! (1)
                                                                        +h2a_voov(b, m, j, e) * t2a(d, c, m, k) * t2a(e, a, i, l)&        ! (ij)
                                                                        +h2a_voov(b, m, i, e) * t2a(d, c, m, j) * t2a(e, a, k, l)&        ! (jk)
                                                                        +h2a_voov(b, m, l, e) * t2a(d, c, m, k) * t2a(e, a, j, i)&        ! (il)
                                                                        +h2a_voov(b, m, i, e) * t2a(d, c, m, l) * t2a(e, a, j, k)&        ! (kl)
                                                                        -h2a_voov(b, m, j, e) * t2a(d, c, m, l) * t2a(e, a, i, k)&        ! (ij)(kl)
                                                                        +h2a_voov(b, m, k, e) * t2a(d, c, m, i) * t2a(e, a, j, l)&        ! (ik)
                                                                        -h2a_voov(b, m, j, e) * t2a(d, c, m, i) * t2a(e, a, k, l)&        ! (ij)(ik)
                                                                        -h2a_voov(b, m, k, e) * t2a(d, c, m, j) * t2a(e, a, i, l)&        ! (jk)(ik)
                                                                        -h2a_voov(b, m, l, e) * t2a(d, c, m, i) * t2a(e, a, j, k)&        ! (il)(ik)
                                                                        -h2a_voov(b, m, k, e) * t2a(d, c, m, l) * t2a(e, a, j, i)&        ! (kl)(ik)
                                                                        +h2a_voov(b, m, j, e) * t2a(d, c, m, l) * t2a(e, a, k, i)         ! (ij)(kl)(ik)
                                                                ! (bd)(ad)
                                                                mm24 =  mm24 - h2a_voov(d, m, i, e) * t2a(a, c, m, k) * t2a(e, b, j, l)&  ! (1)
                                                                        +h2a_voov(d, m, j, e) * t2a(a, c, m, k) * t2a(e, b, i, l)&        ! (ij)
                                                                        +h2a_voov(d, m, i, e) * t2a(a, c, m, j) * t2a(e, b, k, l)&        ! (jk)
                                                                        +h2a_voov(d, m, l, e) * t2a(a, c, m, k) * t2a(e, b, j, i)&        ! (il)
                                                                        +h2a_voov(d, m, i, e) * t2a(a, c, m, l) * t2a(e, b, j, k)&        ! (kl)
                                                                        -h2a_voov(d, m, j, e) * t2a(a, c, m, l) * t2a(e, b, i, k)&        ! (ij)(kl)
                                                                        +h2a_voov(d, m, k, e) * t2a(a, c, m, i) * t2a(e, b, j, l)&        ! (ik)
                                                                        -h2a_voov(d, m, j, e) * t2a(a, c, m, i) * t2a(e, b, k, l)&        ! (ij)(ik)
                                                                        -h2a_voov(d, m, k, e) * t2a(a, c, m, j) * t2a(e, b, i, l)&        ! (jk)(ik)
                                                                        -h2a_voov(d, m, l, e) * t2a(a, c, m, i) * t2a(e, b, j, k)&        ! (il)(ik)
                                                                        -h2a_voov(d, m, k, e) * t2a(a, c, m, l) * t2a(e, b, j, i)&        ! (kl)(ik)
                                                                        +h2a_voov(d, m, j, e) * t2a(a, c, m, l) * t2a(e, b, k, i)         ! (ij)(kl)(ik)
                                                                ! (ac)(ad)
                                                                mm24 =  mm24 - h2a_voov(c, m, i, e) * t2a(b, d, m, k) * t2a(e, a, j, l)&   ! (1)
                                                                        +h2a_voov(c, m, j, e) * t2a(b, d, m, k) * t2a(e, a, i, l)&         ! (ij)
                                                                        +h2a_voov(c, m, i, e) * t2a(b, d, m, j) * t2a(e, a, k, l)&         ! (jk)
                                                                        +h2a_voov(c, m, l, e) * t2a(b, d, m, k) * t2a(e, a, j, i)&         ! (il)
                                                                        +h2a_voov(c, m, i, e) * t2a(b, d, m, l) * t2a(e, a, j, k)&         ! (kl)
                                                                        -h2a_voov(c, m, j, e) * t2a(b, d, m, l) * t2a(e, a, i, k)&         ! (ij)(kl)
                                                                        +h2a_voov(c, m, k, e) * t2a(b, d, m, i) * t2a(e, a, j, l)&         ! (ik)
                                                                        -h2a_voov(c, m, j, e) * t2a(b, d, m, i) * t2a(e, a, k, l)&         ! (ij)(ik)
                                                                        -h2a_voov(c, m, k, e) * t2a(b, d, m, j) * t2a(e, a, i, l)&         ! (jk)(ik)
                                                                        -h2a_voov(c, m, l, e) * t2a(b, d, m, i) * t2a(e, a, j, k)&         ! (il)(ik)
                                                                        -h2a_voov(c, m, k, e) * t2a(b, d, m, l) * t2a(e, a, j, i)&         ! (kl)(ik)
                                                                        +h2a_voov(c, m, j, e) * t2a(b, d, m, l) * t2a(e, a, k, i)          ! (ij)(kl)(ik)
                                                                ! (cd)(ad)
                                                                mm24 =  mm24 - h2a_voov(d, m, i, e) * t2a(b, a, m, k) * t2a(e, c, j, l)&  ! (1)
                                                                        +h2a_voov(d, m, j, e) * t2a(b, a, m, k) * t2a(e, c, i, l)&        ! (ij)
                                                                        +h2a_voov(d, m, i, e) * t2a(b, a, m, j) * t2a(e, c, k, l)&        ! (jk)
                                                                        +h2a_voov(d, m, l, e) * t2a(b, a, m, k) * t2a(e, c, j, i)&        ! (il)
                                                                        +h2a_voov(d, m, i, e) * t2a(b, a, m, l) * t2a(e, c, j, k)&        ! (kl)
                                                                        -h2a_voov(d, m, j, e) * t2a(b, a, m, l) * t2a(e, c, i, k)&        ! (ij)(kl)
                                                                        +h2a_voov(d, m, k, e) * t2a(b, a, m, i) * t2a(e, c, j, l)&        ! (ik)
                                                                        -h2a_voov(d, m, j, e) * t2a(b, a, m, i) * t2a(e, c, k, l)&        ! (ij)(ik)
                                                                        -h2a_voov(d, m, k, e) * t2a(b, a, m, j) * t2a(e, c, i, l)&        ! (jk)(ik)
                                                                        -h2a_voov(d, m, l, e) * t2a(b, a, m, i) * t2a(e, c, j, k)&        ! (il)(ik)
                                                                        -h2a_voov(d, m, k, e) * t2a(b, a, m, l) * t2a(e, c, j, i)&        ! (kl)(ik)
                                                                        +h2a_voov(d, m, j, e) * t2a(b, a, m, l) * t2a(e, c, k, i)         ! (ij)(kl)(ik)
                                                                ! (ab)(cd)(ad)
                                                                mm24 =  mm24 + h2a_voov(b, m, i, e) * t2a(d, a, m, k) * t2a(e, c, j, l)&  ! (1)
                                                                        -h2a_voov(b, m, j, e) * t2a(d, a, m, k) * t2a(e, c, i, l)&        ! (ij)
                                                                        -h2a_voov(b, m, i, e) * t2a(d, a, m, j) * t2a(e, c, k, l)&        ! (jk)
                                                                        -h2a_voov(b, m, l, e) * t2a(d, a, m, k) * t2a(e, c, j, i)&        ! (il)
                                                                        -h2a_voov(b, m, i, e) * t2a(d, a, m, l) * t2a(e, c, j, k)&        ! (kl)
                                                                        +h2a_voov(b, m, j, e) * t2a(d, a, m, l) * t2a(e, c, i, k)&        ! (ij)(kl)
                                                                        -h2a_voov(b, m, k, e) * t2a(d, a, m, i) * t2a(e, c, j, l)&        ! (ik)
                                                                        +h2a_voov(b, m, j, e) * t2a(d, a, m, i) * t2a(e, c, k, l)&        ! (ij)(ik)
                                                                        +h2a_voov(b, m, k, e) * t2a(d, a, m, j) * t2a(e, c, i, l)&        ! (jk)(ik)
                                                                        +h2a_voov(b, m, l, e) * t2a(d, a, m, i) * t2a(e, c, j, k)&        ! (il)(ik)
                                                                        +h2a_voov(b, m, k, e) * t2a(d, a, m, l) * t2a(e, c, j, i)&        ! (kl)(ik)
                                                                        -h2a_voov(b, m, j, e) * t2a(d, a, m, l) * t2a(e, c, k, i)         ! (ij)(kl)(ik)

                                                            end do
                                                        end do
                                                        ! Diagram 2: A(ij/kl)A(bc/ad) h2a(mnij) * t2a(adml) * t2a(bcnk)
                                                        do m = 1, noa
                                                            do n = 1, noa
                                                                ! (1)
                                                                mm24 = mm24 + h2a_oooo(m, n, i, j) * t2a(a, d, m, l) * t2a(b, c, n, k)&  ! (1)
                                                                       -h2a_oooo(m, n, k, j) * t2a(a, d, m, l) * t2a(b, c, n, i)&        ! (ik)
                                                                       -h2a_oooo(m, n, l, j) * t2a(a, d, m, i) * t2a(b, c, n, k)&        ! (il)
                                                                       -h2a_oooo(m, n, i, k) * t2a(a, d, m, l) * t2a(b, c, n, j)&        ! (jk)
                                                                       -h2a_oooo(m, n, i, l) * t2a(a, d, m, j) * t2a(b, c, n, k)&        ! (jl)
                                                                       +h2a_oooo(m, n, k, l) * t2a(a, d, m, j) * t2a(b, c, n, i)         ! (ik)(jl)
                                                                ! (ab)
                                                                mm24 = mm24 - h2a_oooo(m, n, i, j) * t2a(b, d, m, l) * t2a(a, c, n, k)&  ! (1)
                                                                       +h2a_oooo(m, n, k, j) * t2a(b, d, m, l) * t2a(a, c, n, i)&        ! (ik)
                                                                       +h2a_oooo(m, n, l, j) * t2a(b, d, m, i) * t2a(a, c, n, k)&        ! (il)
                                                                       +h2a_oooo(m, n, i, k) * t2a(b, d, m, l) * t2a(a, c, n, j)&        ! (jk)
                                                                       +h2a_oooo(m, n, i, l) * t2a(b, d, m, j) * t2a(a, c, n, k)&        ! (jl)
                                                                       -h2a_oooo(m, n, k, l) * t2a(b, d, m, j) * t2a(a, c, n, i)         ! (ik)(jl)
                                                                ! (bd)
                                                                mm24 = mm24 - h2a_oooo(m, n, i, j) * t2a(a, b, m, l) * t2a(d, c, n, k)&  ! (1)
                                                                       +h2a_oooo(m, n, k, j) * t2a(a, b, m, l) * t2a(d, c, n, i)&        ! (ik)
                                                                       +h2a_oooo(m, n, l, j) * t2a(a, b, m, i) * t2a(d, c, n, k)&        ! (il)
                                                                       +h2a_oooo(m, n, i, k) * t2a(a, b, m, l) * t2a(d, c, n, j)&        ! (jk)
                                                                       +h2a_oooo(m, n, i, l) * t2a(a, b, m, j) * t2a(d, c, n, k)&        ! (jl)
                                                                       -h2a_oooo(m, n, k, l) * t2a(a, b, m, j) * t2a(d, c, n, i)         ! (ik)(jl)
                                                                ! (ac)
                                                                mm24 = mm24 - h2a_oooo(m, n, i, j) * t2a(c, d, m, l) * t2a(b, a, n, k)&  ! (1)
                                                                       +h2a_oooo(m, n, k, j) * t2a(c, d, m, l) * t2a(b, a, n, i)&        ! (ik)
                                                                       +h2a_oooo(m, n, l, j) * t2a(c, d, m, i) * t2a(b, a, n, k)&        ! (il)
                                                                       +h2a_oooo(m, n, i, k) * t2a(c, d, m, l) * t2a(b, a, n, j)&        ! (jk)
                                                                       +h2a_oooo(m, n, i, l) * t2a(c, d, m, j) * t2a(b, a, n, k)&        ! (jl)
                                                                       -h2a_oooo(m, n, k, l) * t2a(c, d, m, j) * t2a(b, a, n, i)         ! (ik)(jl)
                                                                ! (cd)
                                                                mm24 = mm24 - h2a_oooo(m, n, i, j) * t2a(a, c, m, l) * t2a(b, d, n, k)&  ! (1)
                                                                       +h2a_oooo(m, n, k, j) * t2a(a, c, m, l) * t2a(b, d, n, i)&        ! (ik)
                                                                       +h2a_oooo(m, n, l, j) * t2a(a, c, m, i) * t2a(b, d, n, k)&        ! (il)
                                                                       +h2a_oooo(m, n, i, k) * t2a(a, c, m, l) * t2a(b, d, n, j)&        ! (jk)
                                                                       +h2a_oooo(m, n, i, l) * t2a(a, c, m, j) * t2a(b, d, n, k)&        ! (jl)
                                                                       -h2a_oooo(m, n, k, l) * t2a(a, c, m, j) * t2a(b, d, n, i)         ! (ik)(jl)
                                                                ! (ab)(cd)
                                                                mm24 = mm24 + h2a_oooo(m, n, i, j) * t2a(b, c, m, l) * t2a(a, d, n, k)&  ! (1)
                                                                       -h2a_oooo(m, n, k, j) * t2a(b, c, m, l) * t2a(a, d, n, i)&        ! (ik)
                                                                       -h2a_oooo(m, n, l, j) * t2a(b, c, m, i) * t2a(a, d, n, k)&        ! (il)
                                                                       -h2a_oooo(m, n, i, k) * t2a(b, c, m, l) * t2a(a, d, n, j)&        ! (jk)
                                                                       -h2a_oooo(m, n, i, l) * t2a(b, c, m, j) * t2a(a, d, n, k)&        ! (jl)
                                                                       +h2a_oooo(m, n, k, l) * t2a(b, c, m, j) * t2a(a, d, n, i)         ! (ik)(jl)
                                                            end do
                                                        end do
                                                        ! Diagram 3: A(jk/il)A(ab/cd) h2a(abef) * t2a(fcjk) * t2a(edil)
                                                        do e = 1, nua
                                                            do f = 1, nua
                                                                ! (1)
                                                                mm24 = mm24 + h2a_vvvv(a, b, e, f) * t2a(f, c, j, k) * t2a(e, d, i, l)&  ! (1)
                                                                      -h2a_vvvv(a, b, e, f) * t2a(f, c, i, k) * t2a(e, d, j, l)&         ! (ij)
                                                                      -h2a_vvvv(a, b, e, f) * t2a(f, c, l, k) * t2a(e, d, i, j)&         ! (jl)
                                                                      -h2a_vvvv(a, b, e, f) * t2a(f, c, j, i) * t2a(e, d, k, l)&         ! (ik)
                                                                      -h2a_vvvv(a, b, e, f) * t2a(f, c, j, l) * t2a(e, d, i, k)&         ! (kl)
                                                                      +h2a_vvvv(a, b, e, f) * t2a(f, c, i, l) * t2a(e, d, j, k)          ! (ij)(kl)
                                                                ! (ac)
                                                                mm24 = mm24 - h2a_vvvv(c, b, e, f) * t2a(f, a, j, k) * t2a(e, d, i, l)&  ! (1)
                                                                      +h2a_vvvv(c, b, e, f) * t2a(f, a, i, k) * t2a(e, d, j, l)&         ! (ij)
                                                                      +h2a_vvvv(c, b, e, f) * t2a(f, a, l, k) * t2a(e, d, i, j)&         ! (jl)
                                                                      +h2a_vvvv(c, b, e, f) * t2a(f, a, j, i) * t2a(e, d, k, l)&         ! (ik)
                                                                      +h2a_vvvv(c, b, e, f) * t2a(f, a, j, l) * t2a(e, d, i, k)&         ! (kl)
                                                                      -h2a_vvvv(c, b, e, f) * t2a(f, a, i, l) * t2a(e, d, j, k)          ! (ij)(kl)
                                                                ! (ad)
                                                                mm24 = mm24 - h2a_vvvv(d, b, e, f) * t2a(f, c, j, k) * t2a(e, a, i, l)&  ! (1)
                                                                      +h2a_vvvv(d, b, e, f) * t2a(f, c, i, k) * t2a(e, a, j, l)&         ! (ij)
                                                                      +h2a_vvvv(d, b, e, f) * t2a(f, c, l, k) * t2a(e, a, i, j)&         ! (jl)
                                                                      +h2a_vvvv(d, b, e, f) * t2a(f, c, j, i) * t2a(e, a, k, l)&         ! (ik)
                                                                      +h2a_vvvv(d, b, e, f) * t2a(f, c, j, l) * t2a(e, a, i, k)&         ! (kl)
                                                                      -h2a_vvvv(d, b, e, f) * t2a(f, c, i, l) * t2a(e, a, j, k)          ! (ij)(kl)
                                                                ! (bc)
                                                                mm24 = mm24 - h2a_vvvv(a, c, e, f) * t2a(f, b, j, k) * t2a(e, d, i, l)&  ! (1)
                                                                      +h2a_vvvv(a, c, e, f) * t2a(f, b, i, k) * t2a(e, d, j, l)&         ! (ij)
                                                                      +h2a_vvvv(a, c, e, f) * t2a(f, b, l, k) * t2a(e, d, i, j)&         ! (jl)
                                                                      +h2a_vvvv(a, c, e, f) * t2a(f, b, j, i) * t2a(e, d, k, l)&         ! (ik)
                                                                      +h2a_vvvv(a, c, e, f) * t2a(f, b, j, l) * t2a(e, d, i, k)&         ! (kl)
                                                                      -h2a_vvvv(a, c, e, f) * t2a(f, b, i, l) * t2a(e, d, j, k)          ! (ij)(kl)
                                                                ! (bd)
                                                                mm24 = mm24 - h2a_vvvv(a, d, e, f) * t2a(f, c, j, k) * t2a(e, b, i, l)&  ! (1)
                                                                      +h2a_vvvv(a, d, e, f) * t2a(f, c, i, k) * t2a(e, b, j, l)&         ! (ij)
                                                                      +h2a_vvvv(a, d, e, f) * t2a(f, c, l, k) * t2a(e, b, i, j)&         ! (jl)
                                                                      +h2a_vvvv(a, d, e, f) * t2a(f, c, j, i) * t2a(e, b, k, l)&         ! (ik)
                                                                      +h2a_vvvv(a, d, e, f) * t2a(f, c, j, l) * t2a(e, b, i, k)&         ! (kl)
                                                                      -h2a_vvvv(a, d, e, f) * t2a(f, c, i, l) * t2a(e, b, j, k)          ! (ij)(kl)
                                                                ! (ac)(bd)
                                                                mm24 = mm24 + h2a_vvvv(c, d, e, f) * t2a(f, a, j, k) * t2a(e, b, i, l)&  ! (1)
                                                                      -h2a_vvvv(c, d, e, f) * t2a(f, a, i, k) * t2a(e, b, j, l)&         ! (ij)
                                                                      -h2a_vvvv(c, d, e, f) * t2a(f, a, l, k) * t2a(e, b, i, j)&         ! (jl)
                                                                      -h2a_vvvv(c, d, e, f) * t2a(f, a, j, i) * t2a(e, b, k, l)&         ! (ik)
                                                                      -h2a_vvvv(c, d, e, f) * t2a(f, a, j, l) * t2a(e, b, i, k)&         ! (kl)
                                                                      +h2a_vvvv(c, d, e, f) * t2a(f, a, i, l) * t2a(e, b, j, k)          ! (ij)(kl)
                                                            end do
                                                        end do

                                                        !!! L4A Computation !!!
                                                        ! Diagram 1: A(ij/kl)A(ab/cd) h2a(klcd) * l2a(abij)
                                                        ! (1)
                                                        l4 = l4 + h2a_oovv(k, l, c, d) * l2a(a, b, i, j)&  ! (1)
                                                             -h2a_oovv(i, l, c, d) * l2a(a, b, k, j)&      ! (ik)
                                                             -h2a_oovv(k, i, c, d) * l2a(a, b, l, j)&      ! (il)
                                                             -h2a_oovv(j, l, c, d) * l2a(a, b, i, k)&      ! (jk)
                                                             -h2a_oovv(k, j, c, d) * l2a(a, b, i, l)&      ! (jl)
                                                             +h2a_oovv(i, j, c, d) * l2a(a, b, k, l)       ! (ik)(jl)
                                                        ! (ac)
                                                        l4 = l4 - h2a_oovv(k, l, a, d) * l2a(c, b, i, j)&  ! (1)
                                                             +h2a_oovv(i, l, a, d) * l2a(c, b, k, j)&      ! (ik)
                                                             +h2a_oovv(k, i, a, d) * l2a(c, b, l, j)&      ! (il)
                                                             +h2a_oovv(j, l, a, d) * l2a(c, b, i, k)&      ! (jk)
                                                             +h2a_oovv(k, j, a, d) * l2a(c, b, i, l)&      ! (jl)
                                                             -h2a_oovv(i, j, a, d) * l2a(c, b, k, l)       ! (ik)(jl)
                                                        ! (ad)
                                                        l4 = l4 - h2a_oovv(k, l, c, a) * l2a(d, b, i, j)&  ! (1)
                                                             +h2a_oovv(i, l, c, a) * l2a(d, b, k, j)&      ! (ik)
                                                             +h2a_oovv(k, i, c, a) * l2a(d, b, l, j)&      ! (il)
                                                             +h2a_oovv(j, l, c, a) * l2a(d, b, i, k)&      ! (jk)
                                                             +h2a_oovv(k, j, c, a) * l2a(d, b, i, l)&      ! (jl)
                                                             -h2a_oovv(i, j, c, a) * l2a(d, b, k, l)       ! (ik)(jl)
                                                        ! (bc)
                                                        l4 = l4 - h2a_oovv(k, l, b, d) * l2a(a, c, i, j)&  ! (1)
                                                             +h2a_oovv(i, l, b, d) * l2a(a, c, k, j)&      ! (ik)
                                                             +h2a_oovv(k, i, b, d) * l2a(a, c, l, j)&      ! (il)
                                                             +h2a_oovv(j, l, b, d) * l2a(a, c, i, k)&      ! (jk)
                                                             +h2a_oovv(k, j, b, d) * l2a(a, c, i, l)&      ! (jl)
                                                             -h2a_oovv(i, j, b, d) * l2a(a, c, k, l)       ! (ik)(jl)
                                                        ! (bd)
                                                        l4 = l4 - h2a_oovv(k, l, c, b) * l2a(a, d, i, j)&  ! (1)
                                                             +h2a_oovv(i, l, c, b) * l2a(a, d, k, j)&      ! (ik)
                                                             +h2a_oovv(k, i, c, b) * l2a(a, d, l, j)&      ! (il)
                                                             +h2a_oovv(j, l, c, b) * l2a(a, d, i, k)&      ! (jk)
                                                             +h2a_oovv(k, j, c, b) * l2a(a, d, i, l)&      ! (jl)
                                                             -h2a_oovv(i, j, c, b) * l2a(a, d, k, l)       ! (ik)(jl)
                                                        ! (ac)(bd)
                                                        l4 = l4 + h2a_oovv(k, l, a, b) * l2a(c, d, i, j)&  ! (1)
                                                             -h2a_oovv(i, l, a, b) * l2a(c, d, k, j)&      ! (ik)
                                                             -h2a_oovv(k, i, a, b) * l2a(c, d, l, j)&      ! (il)
                                                             -h2a_oovv(j, l, a, b) * l2a(c, d, i, k)&      ! (jk)
                                                             -h2a_oovv(k, j, a, b) * l2a(c, d, i, l)&      ! (jl)
                                                             +h2a_oovv(i, j, a, b) * l2a(c, d, k, l)       ! (ik)(jl)

                                                        temp = mm24 * l4

                                                        denom = fA_oo(i,i) + fA_oo(j,j) + fA_oo(k,k) + fA_oo(l,l)&
                                                        - fA_vv(a,a) - fA_vv(b,b) - fA_vv(c,c) - fA_vv(d,d)

                                                        deltaA = deltaA + temp/denom

                                                        denom = H1A_oo(i,i) + H1A_oo(j,j) + H1A_oo(k,k) + H1A_oo(l,l)&
                                                        - H1A_vv(a,a) - H1A_vv(b,b) - H1A_vv(c,c) - H1A_vv(d,d)

                                                        deltaB = deltaB + temp/denom

                                                        denom = denom &
                                                        -H2A_oooo(j,i,j,i)-H2A_oooo(k,i,k,i)-H2A_oooo(l,i,l,i)&
                                                        -H2A_oooo(k,j,k,j)-H2A_oooo(l,j,l,j)-H2A_oooo(l,k,l,k)&
                                                        -H2A_voov(a,i,i,a)-H2A_voov(a,j,j,a)-H2A_voov(a,k,k,a)&
                                                        -H2A_voov(a,l,l,a)-H2A_voov(b,i,i,b)-H2A_voov(b,j,j,b)&
                                                        -H2A_voov(b,k,k,b)-H2A_voov(b,l,l,b)-H2A_voov(c,i,i,c)&
                                                        -H2A_voov(c,j,j,c)-H2A_voov(c,k,k,c)-H2A_voov(c,l,l,c)&
                                                        -H2A_voov(d,i,i,d)-H2A_voov(d,j,j,d)-H2A_voov(d,k,k,d)&
                                                        -H2A_voov(d,l,l,d)-H2A_vvvv(a,b,a,b)-H2A_vvvv(a,c,a,c)&
                                                        -H2A_vvvv(a,d,a,d)-H2A_vvvv(b,c,b,c)-H2A_vvvv(b,d,b,d)&
                                                        -H2A_vvvv(c,d,c,d)

                                                        deltaC = deltaC + temp/denom

                                                        denom = denom &
                                                        +D3A_O(a,i,j)+D3A_O(a,i,k)+D3A_O(a,i,l)+D3A_O(a,j,k)&
                                                        +D3A_O(a,j,l)+D3A_O(a,k,l)+D3A_O(b,i,j)+D3A_O(b,i,k)&
                                                        +D3A_O(b,i,l)+D3A_O(b,j,k)+D3A_O(b,j,l)+D3A_O(b,k,l)&
                                                        +D3A_O(c,i,j)+D3A_O(c,i,k)+D3A_O(c,i,l)+D3A_O(c,j,k)&
                                                        +D3A_O(c,j,l)+D3A_O(c,k,l)+D3A_O(d,i,j)+D3A_O(d,i,k)&
                                                        +D3A_O(d,i,l)+D3A_O(d,j,k)+D3A_O(d,j,l)+D3A_O(d,k,l)&
                                                        -D3A_V(a,i,b)-D3A_V(a,j,b)-D3A_V(a,k,b)-D3A_V(a,l,b)&
                                                        -D3A_V(a,i,c)-D3A_V(a,j,c)-D3A_V(a,k,c)-D3A_V(a,l,c)&
                                                        -D3A_V(a,i,d)-D3A_V(a,j,d)-D3A_V(a,k,d)-D3A_V(a,l,d)&
                                                        -D3A_V(b,i,c)-D3A_V(b,j,c)-D3A_V(b,k,c)-D3A_V(b,l,c)&
                                                        -D3A_V(b,i,d)-D3A_V(b,j,d)-D3A_V(b,k,d)-D3A_V(b,l,d)&
                                                        -D3A_V(c,i,d)-D3A_V(c,j,d)-D3A_V(c,k,d)-D3A_V(c,l,d)

                                                        deltaD = deltaD + temp/denom

                                                    end do
                                                end do
                                            end do
                                        end do
                                    end do
                                end do
                            end do
                        end do

                  !print*, "Error = ", error

              end subroutine crcc24A

              subroutine crcc24B(deltaA,deltaB,deltaC,deltaD,&
                              t2a,t2b,l2a,l2b,&
                              fA_oo,fA_vv,fB_oo,fB_vv,&
                              H1A_oo,H1A_vv,H1B_oo,H1B_vv,&
                              H2A_voov, H2A_oooo, H2A_vvvv,H2A_oovv,&
                              H2B_voov,H2B_ovov,H2B_vovo,H2B_ovvo,H2B_oooo,H2B_vvvv,H2B_oovv,&
                              H2C_voov,&
                              D3A_O,D3A_V,D3B_O,D3B_V,D3C_O,D3C_V,&
                              noa,nua,nob,nub)

                        real(kind=8), intent(out) :: deltaA, deltaB, deltaC, deltaD
                        integer, intent(in) :: noa, nua, nob, nub
                        real(kind=8), intent(in) :: t2a(1:nua,1:nua,1:noa,1:noa),t2b(1:nua,1:nub,1:noa,1:nob),&
                        l2a(1:nua,1:nua,1:noa,1:noa),l2b(1:nua,1:nub,1:noa,1:nob),&
                        fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),fB_oo(1:nob,1:nob),fB_vv(1:nub,1:nub),&
                        H1A_oo(1:noa,1:noa),H1A_vv(1:nua,1:nua),H1B_oo(1:nob,1:nob),H1B_vv(1:nub,1:nub),&
                        H2A_voov(1:nua,1:noa,1:noa,1:nua),&
                        H2A_oooo(1:noa,1:noa,1:noa,1:noa),&
                        H2A_vvvv(1:nua,1:nua,1:nua,1:nua),&
                        H2A_oovv(1:noa,1:noa,1:nua,1:nua),&
                        H2B_voov(1:nua,1:nob,1:noa,1:nub),&
                        H2B_ovov(1:noa,1:nub,1:noa,1:nub),&
                        H2B_vovo(1:nua,1:nob,1:nua,1:nob),&
                        H2B_ovvo(1:noa,1:nub,1:nua,1:nob),&
                        H2B_oooo(1:noa,1:nob,1:noa,1:nob),&
                        H2B_vvvv(1:nua,1:nub,1:nua,1:nub),&
                        H2B_oovv(1:noa,1:nob,1:nua,1:nub),&
                        H2C_voov(1:nub,1:nob,1:nob,1:nub),&
                        D3A_O(1:nua,1:noa,1:noa),&
                        D3A_V(1:nua,1:noa,1:nua),&
                        D3B_O(1:nua,1:noa,1:nob),&
                        D3B_V(1:nua,1:noa,1:nub),&
                        D3C_O(1:nub,1:noa,1:nob),&
                        D3C_V(1:nua,1:nob,1:nub)

                        !real(kind=8), intent(in) :: test_array(1:nua,1:nua,1:nua,1:nua,1:noa,1:noa,1:noa,1:noa)

                        integer :: i, j, k, l, a, b, c, d, m, n, e, f
                        real(kind=8) :: denom, temp, mm24, l4
                        !real(kind=8) :: error

                        deltaA = 0.0d0
                        deltaB = 0.0d0
                        deltaC = 0.0d0
                        deltaD = 0.0d0

                        !error = 0.0d0
                        do i = 1, noa
                            do j = i+1, noa
                                do k = j+1, noa
                                    do l = 1, nob
                                        do a = 1, nua
                                            do b = a+1, nua
                                                do c = b+1, nua
                                                    do d = 1, nub

                                                        mm24 = 0.0d0
                                                        l4 = 0.0d0

                                                        !!! MM(2,4)B Computation !!!
                                                        ! Diagram 1:  -A(i/jk)A(c/ab) h2b(mdel) * t2a(abim) * t2a(ecjk)
                                                        do m = 1, noa
                                                            do e = 1, nua
                                                                ! (1)
                                                                mm24 = mm24 - h2b_ovvo(m, d, e, l) * t2a(a, b, i, m) * t2a(e, c, j, k)&  ! (1)
                                                                            + h2b_ovvo(m, d, e, l) * t2a(a, b, j, m) * t2a(e, c, i, k)&  ! (ij)
                                                                            + h2b_ovvo(m, d, e, l) * t2a(a, b, k, m) * t2a(e, c, j, i)   ! (ik)
                                                                ! (ac)
                                                                mm24 = mm24 + h2b_ovvo(m, d, e, l) * t2a(c, b, i, m) * t2a(e, a, j, k)&  ! (1)
                                                                            - h2b_ovvo(m, d, e, l) * t2a(c, b, j, m) * t2a(e, a, i, k)&  ! (ij)
                                                                            - h2b_ovvo(m, d, e, l) * t2a(c, b, k, m) * t2a(e, a, j, i)   ! (ik)
                                                                ! (bc)
                                                                mm24 = mm24 + h2b_ovvo(m, d, e, l) * t2a(a, c, i, m) * t2a(e, b, j, k)&  ! (1)
                                                                            - h2b_ovvo(m, d, e, l) * t2a(a, c, j, m) * t2a(e, b, i, k)&  ! (ij)
                                                                            - h2b_ovvo(m, d, e, l) * t2a(a, c, k, m) * t2a(e, b, j, i)   ! (ik)

                                                            end do
                                                        end do
                                                        ! Diagram 2:   A(k/ij)A(a/bc) h2a(mnij) * t2a(bcnk) * t2b(adml)
                                                        do m = 1, noa
                                                            do n = 1, noa
                                                                ! (1)
                                                                mm24 = mm24 + h2a_oooo(m, n, i, j) * t2a(b, c, n, k) * t2b(a, d, m, l)&  ! (1)
                                                                            - h2a_oooo(m, n, k, j) * t2a(b, c, n, i) * t2b(a, d, m, l)&  ! (ik)
                                                                            - h2a_oooo(m, n, i, k) * t2a(b, c, n, j) * t2b(a, d, m, l)   ! (jk)
                                                                ! (ab)
                                                                mm24 = mm24 - h2a_oooo(m, n, i, j) * t2a(a, c, n, k) * t2b(b, d, m, l)&  ! (1)
                                                                            + h2a_oooo(m, n, k, j) * t2a(a, c, n, i) * t2b(b, d, m, l)&  ! (ik)
                                                                            + h2a_oooo(m, n, i, k) * t2a(a, c, n, j) * t2b(b, d, m, l)   ! (jk)
                                                                ! (ac)
                                                                mm24 = mm24 - h2a_oooo(m, n, i, j) * t2a(b, a, n, k) * t2b(c, d, m, l)&  ! (1)
                                                                            + h2a_oooo(m, n, k, j) * t2a(b, a, n, i) * t2b(c, d, m, l)&  ! (ik)
                                                                            + h2a_oooo(m, n, i, k) * t2a(b, a, n, j) * t2b(c, d, m, l)   ! (jk)
                                                            end do
                                                        end do
                                                        ! Diagram 3:  -A(ijk)A(c/ab) h2b(mdjf) * t2a(abim) * t2b(cfkl)
                                                        do m = 1, noa
                                                            do f = 1, nub
                                                                ! (1)
                                                                mm24 = mm24 - h2b_ovov(m, d, j, f) * t2a(a, b, i, m) * t2b(c, f, k, l)&  ! (1)
                                                                            + h2b_ovov(m, d, i, f) * t2a(a, b, j, m) * t2b(c, f, k, l)&  ! (ij)
                                                                            + h2b_ovov(m, d, j, f) * t2a(a, b, k, m) * t2b(c, f, i, l)&  ! (ik)
                                                                            + h2b_ovov(m, d, k, f) * t2a(a, b, i, m) * t2b(c, f, j, l)&  ! (jk)
                                                                            - h2b_ovov(m, d, i, f) * t2a(a, b, k, m) * t2b(c, f, j, l)&  ! (ij)(jk)
                                                                            - h2b_ovov(m, d, k, f) * t2a(a, b, j, m) * t2b(c, f, i, l)   ! (ik)(jk)
                                                                ! (ac)
                                                                mm24 = mm24 + h2b_ovov(m, d, j, f) * t2a(c, b, i, m) * t2b(a, f, k, l)&  ! (1)
                                                                            - h2b_ovov(m, d, i, f) * t2a(c, b, j, m) * t2b(a, f, k, l)&  ! (ij)
                                                                            - h2b_ovov(m, d, j, f) * t2a(c, b, k, m) * t2b(a, f, i, l)&  ! (ik)
                                                                            - h2b_ovov(m, d, k, f) * t2a(c, b, i, m) * t2b(a, f, j, l)&  ! (jk)
                                                                            + h2b_ovov(m, d, i, f) * t2a(c, b, k, m) * t2b(a, f, j, l)&  ! (ij)(jk)
                                                                            + h2b_ovov(m, d, k, f) * t2a(c, b, j, m) * t2b(a, f, i, l)   ! (ik)(jk)
                                                                ! (bc)
                                                                mm24 = mm24 + h2b_ovov(m, d, j, f) * t2a(a, c, i, m) * t2b(b, f, k, l)&  ! (1)
                                                                            - h2b_ovov(m, d, i, f) * t2a(a, c, j, m) * t2b(b, f, k, l)&  ! (ij)
                                                                            - h2b_ovov(m, d, j, f) * t2a(a, c, k, m) * t2b(b, f, i, l)&  ! (ik)
                                                                            - h2b_ovov(m, d, k, f) * t2a(a, c, i, m) * t2b(b, f, j, l)&  ! (jk)
                                                                            + h2b_ovov(m, d, i, f) * t2a(a, c, k, m) * t2b(b, f, j, l)&  ! (ij)(jk)
                                                                            + h2b_ovov(m, d, k, f) * t2a(a, c, j, m) * t2b(b, f, i, l)   ! (ik)(jk)
                                                            end do
                                                        end do
                                                        ! Diagram 4:  -A(ijk)A(abc) h2b(amie) * t2b(bejl) * t2b(cdkm)
                                                        do m = 1, nob
                                                            do e = 1, nub
                                                                ! (1)
                                                                mm24 = mm24 - h2b_voov(a, m, i, e) * t2b(b, e, j, l) * t2b(c, d, k, m)&  ! (1)
                                                                            + h2b_voov(a, m, j, e) * t2b(b, e, i, l) * t2b(c, d, k, m)&  ! (ij)
                                                                            + h2b_voov(a, m, k, e) * t2b(b, e, j, l) * t2b(c, d, i, m)&  ! (ik)
                                                                            + h2b_voov(a, m, i, e) * t2b(b, e, k, l) * t2b(c, d, j, m)&  ! (jk)
                                                                            - h2b_voov(a, m, k, e) * t2b(b, e, i, l) * t2b(c, d, j, m)&  ! (ij)(jk)
                                                                            - h2b_voov(a, m, j, e) * t2b(b, e, k, l) * t2b(c, d, i, m)   ! (ik)(jk)
                                                                ! (ab)
                                                                mm24 = mm24 + h2b_voov(b, m, i, e) * t2b(a, e, j, l) * t2b(c, d, k, m)&  ! (1)
                                                                            - h2b_voov(b, m, j, e) * t2b(a, e, i, l) * t2b(c, d, k, m)&  ! (ij)
                                                                            - h2b_voov(b, m, k, e) * t2b(a, e, j, l) * t2b(c, d, i, m)&  ! (ik)
                                                                            - h2b_voov(b, m, i, e) * t2b(a, e, k, l) * t2b(c, d, j, m)&  ! (jk)
                                                                            + h2b_voov(b, m, k, e) * t2b(a, e, i, l) * t2b(c, d, j, m)&  ! (ij)(jk)
                                                                            + h2b_voov(b, m, j, e) * t2b(a, e, k, l) * t2b(c, d, i, m)   ! (ik)(jk)
                                                                ! (ac)
                                                                mm24 = mm24 + h2b_voov(c, m, i, e) * t2b(b, e, j, l) * t2b(a, d, k, m)&  ! (1)
                                                                            - h2b_voov(c, m, j, e) * t2b(b, e, i, l) * t2b(a, d, k, m)&  ! (ij)
                                                                            - h2b_voov(c, m, k, e) * t2b(b, e, j, l) * t2b(a, d, i, m)&  ! (ik)
                                                                            - h2b_voov(c, m, i, e) * t2b(b, e, k, l) * t2b(a, d, j, m)&  ! (jk)
                                                                            + h2b_voov(c, m, k, e) * t2b(b, e, i, l) * t2b(a, d, j, m)&  ! (ij)(jk)
                                                                            + h2b_voov(c, m, j, e) * t2b(b, e, k, l) * t2b(a, d, i, m)   ! (ik)(jk)
                                                                ! (bc)
                                                                mm24 = mm24 + h2b_voov(a, m, i, e) * t2b(c, e, j, l) * t2b(b, d, k, m)&  ! (1)
                                                                            - h2b_voov(a, m, j, e) * t2b(c, e, i, l) * t2b(b, d, k, m)&  ! (ij)
                                                                            - h2b_voov(a, m, k, e) * t2b(c, e, j, l) * t2b(b, d, i, m)&  ! (ik)
                                                                            - h2b_voov(a, m, i, e) * t2b(c, e, k, l) * t2b(b, d, j, m)&  ! (jk)
                                                                            + h2b_voov(a, m, k, e) * t2b(c, e, i, l) * t2b(b, d, j, m)&  ! (ij)(jk)
                                                                            + h2b_voov(a, m, j, e) * t2b(c, e, k, l) * t2b(b, d, i, m)   ! (ik)(jk)
                                                                ! (ab)(bc)
                                                                mm24 = mm24 - h2b_voov(c, m, i, e) * t2b(a, e, j, l) * t2b(b, d, k, m)&  ! (1)
                                                                            + h2b_voov(c, m, j, e) * t2b(a, e, i, l) * t2b(b, d, k, m)&  ! (ij)
                                                                            + h2b_voov(c, m, k, e) * t2b(a, e, j, l) * t2b(b, d, i, m)&  ! (ik)
                                                                            + h2b_voov(c, m, i, e) * t2b(a, e, k, l) * t2b(b, d, j, m)&  ! (jk)
                                                                            - h2b_voov(c, m, k, e) * t2b(a, e, i, l) * t2b(b, d, j, m)&  ! (ij)(jk)
                                                                            - h2b_voov(c, m, j, e) * t2b(a, e, k, l) * t2b(b, d, i, m)   ! (ik)(jk)
                                                                ! (ac)(bc)
                                                                mm24 = mm24 - h2b_voov(b, m, i, e) * t2b(c, e, j, l) * t2b(a, d, k, m)&  ! (1)
                                                                            + h2b_voov(b, m, j, e) * t2b(c, e, i, l) * t2b(a, d, k, m)&  ! (ij)
                                                                            + h2b_voov(b, m, k, e) * t2b(c, e, j, l) * t2b(a, d, i, m)&  ! (ik)
                                                                            + h2b_voov(b, m, i, e) * t2b(c, e, k, l) * t2b(a, d, j, m)&  ! (jk)
                                                                            - h2b_voov(b, m, k, e) * t2b(c, e, i, l) * t2b(a, d, j, m)&  ! (ij)(jk)
                                                                            - h2b_voov(b, m, j, e) * t2b(c, e, k, l) * t2b(a, d, i, m)   ! (ik)(jk)
                                                            end do
                                                        end do
                                                        ! Diagram 5:   A(ijk)A(a/bc) h2b(mnjl) * t2a(bcmk) * t2b(adin)
                                                        do m = 1, noa
                                                            do n = 1, nob
                                                                ! (1)
                                                                mm24 = mm24 + h2b_oooo(m, n, j, l) * t2a(b, c, m, k) * t2b(a, d, i, n)&  ! (1)
                                                                            - h2b_oooo(m, n, i, l) * t2a(b, c, m, k) * t2b(a, d, j, n)&  ! (ij)
                                                                            - h2b_oooo(m, n, j, l) * t2a(b, c, m, i) * t2b(a, d, k, n)&  ! (ik)
                                                                            - h2b_oooo(m, n, k, l) * t2a(b, c, m, j) * t2b(a, d, i, n)&  ! (jk)
                                                                            + h2b_oooo(m, n, i, l) * t2a(b, c, m, j) * t2b(a, d, k, n)&  ! (ij)(jk)
                                                                            + h2b_oooo(m, n, k, l) * t2a(b, c, m, i) * t2b(a, d, j, n)   ! (ik)(jk)
                                                                ! (ab)
                                                                mm24 = mm24 - h2b_oooo(m, n, j, l) * t2a(a, c, m, k) * t2b(b, d, i, n)&  ! (1)
                                                                            + h2b_oooo(m, n, i, l) * t2a(a, c, m, k) * t2b(b, d, j, n)&  ! (ij)
                                                                            + h2b_oooo(m, n, j, l) * t2a(a, c, m, i) * t2b(b, d, k, n)&  ! (ik)
                                                                            + h2b_oooo(m, n, k, l) * t2a(a, c, m, j) * t2b(b, d, i, n)&  ! (jk)
                                                                            - h2b_oooo(m, n, i, l) * t2a(a, c, m, j) * t2b(b, d, k, n)&  ! (ij)(jk)
                                                                            - h2b_oooo(m, n, k, l) * t2a(a, c, m, i) * t2b(b, d, j, n)   ! (ik)(jk)
                                                                ! (ac)
                                                                mm24 = mm24 - h2b_oooo(m, n, j, l) * t2a(b, a, m, k) * t2b(c, d, i, n)&  ! (1)
                                                                            + h2b_oooo(m, n, i, l) * t2a(b, a, m, k) * t2b(c, d, j, n)&  ! (ij)
                                                                            + h2b_oooo(m, n, j, l) * t2a(b, a, m, i) * t2b(c, d, k, n)&  ! (ik)
                                                                            + h2b_oooo(m, n, k, l) * t2a(b, a, m, j) * t2b(c, d, i, n)&  ! (jk)
                                                                            - h2b_oooo(m, n, i, l) * t2a(b, a, m, j) * t2b(c, d, k, n)&  ! (ij)(jk)
                                                                            - h2b_oooo(m, n, k, l) * t2a(b, a, m, i) * t2b(c, d, j, n)   ! (ik)(jk)
                                                            end do
                                                        end do
                                                        ! Diagram 6:  -A(i/jk)A(abc) h2b(bmel) * t2a(ecjk) * t2b(adim)
                                                        do m = 1, nob
                                                            do e = 1, nua
                                                                ! (1)
                                                                mm24 = mm24 - h2b_vovo(b, m, e, l) * t2a(e, c, j, k) * t2b(a, d, i, m)&  ! (1)
                                                                            + h2b_vovo(a, m, e, l) * t2a(e, c, j, k) * t2b(b, d, i, m)&  ! (ab)
                                                                            + h2b_vovo(b, m, e, l) * t2a(e, a, j, k) * t2b(c, d, i, m)&  ! (ac)
                                                                            + h2b_vovo(c, m, e, l) * t2a(e, b, j, k) * t2b(a, d, i, m)&  ! (bc)
                                                                            - h2b_vovo(a, m, e, l) * t2a(e, b, j, k) * t2b(c, d, i, m)&  ! (ab)(bc)
                                                                            - h2b_vovo(c, m, e, l) * t2a(e, a, j, k) * t2b(b, d, i, m)   ! (ac)(bc)
                                                                ! (ij)
                                                                mm24 = mm24 + h2b_vovo(b, m, e, l) * t2a(e, c, i, k) * t2b(a, d, j, m)&  ! (1)
                                                                            - h2b_vovo(a, m, e, l) * t2a(e, c, i, k) * t2b(b, d, j, m)&  ! (ab)
                                                                            - h2b_vovo(b, m, e, l) * t2a(e, a, i, k) * t2b(c, d, j, m)&  ! (ac)
                                                                            - h2b_vovo(c, m, e, l) * t2a(e, b, i, k) * t2b(a, d, j, m)&  ! (bc)
                                                                            + h2b_vovo(a, m, e, l) * t2a(e, b, i, k) * t2b(c, d, j, m)&  ! (ab)(bc)
                                                                            + h2b_vovo(c, m, e, l) * t2a(e, a, i, k) * t2b(b, d, j, m)   ! (ac)(bc)
                                                                ! (ik)
                                                                mm24 = mm24 + h2b_vovo(b, m, e, l) * t2a(e, c, j, i) * t2b(a, d, k, m)&  ! (1)
                                                                            - h2b_vovo(a, m, e, l) * t2a(e, c, j, i) * t2b(b, d, k, m)&  ! (ab)
                                                                            - h2b_vovo(b, m, e, l) * t2a(e, a, j, i) * t2b(c, d, k, m)&  ! (ac)
                                                                            - h2b_vovo(c, m, e, l) * t2a(e, b, j, i) * t2b(a, d, k, m)&  ! (bc)
                                                                            + h2b_vovo(a, m, e, l) * t2a(e, b, j, i) * t2b(c, d, k, m)&  ! (ab)(bc)
                                                                            + h2b_vovo(c, m, e, l) * t2a(e, a, j, i) * t2b(b, d, k, m)   ! (ac)(bc)
                                                            end do
                                                        end do
                                                        ! Diagram 7:  -A(i/jk)A(abc) h2a(amie) * t2a(ecjk) * t2b(bdml)
                                                        do m = 1, noa
                                                            do e = 1, nua
                                                                ! (1)
                                                                mm24 = mm24 - h2a_voov(a, m, i, e) * t2a(e, c, j, k) * t2b(b, d, m, l)&  ! (1)
                                                                            + h2a_voov(b, m, i, e) * t2a(e, c, j, k) * t2b(a, d, m, l)&  ! (ab)
                                                                            + h2a_voov(c, m, i, e) * t2a(e, a, j, k) * t2b(b, d, m, l)&  ! (ac)
                                                                            + h2a_voov(a, m, i, e) * t2a(e, b, j, k) * t2b(c, d, m, l)&  ! (bc)
                                                                            - h2a_voov(c, m, i, e) * t2a(e, b, j, k) * t2b(a, d, m, l)&  ! (ab)(bc)
                                                                            - h2a_voov(b, m, i, e) * t2a(e, a, j, k) * t2b(c, d, m, l)   ! (ac)(bc)
                                                                ! (ij)
                                                                mm24 = mm24 + h2a_voov(a, m, j, e) * t2a(e, c, i, k) * t2b(b, d, m, l)&  ! (1)
                                                                            - h2a_voov(b, m, j, e) * t2a(e, c, i, k) * t2b(a, d, m, l)&  ! (ab)
                                                                            - h2a_voov(c, m, j, e) * t2a(e, a, i, k) * t2b(b, d, m, l)&  ! (ac)
                                                                            - h2a_voov(a, m, j, e) * t2a(e, b, i, k) * t2b(c, d, m, l)&  ! (bc)
                                                                            + h2a_voov(c, m, j, e) * t2a(e, b, i, k) * t2b(a, d, m, l)&  ! (ab)(bc)
                                                                            + h2a_voov(b, m, j, e) * t2a(e, a, i, k) * t2b(c, d, m, l)   ! (ac)(bc)
                                                                ! (ik)
                                                                mm24 = mm24 + h2a_voov(a, m, k, e) * t2a(e, c, j, i) * t2b(b, d, m, l)&  ! (1)
                                                                            - h2a_voov(b, m, k, e) * t2a(e, c, j, i) * t2b(a, d, m, l)&  ! (ab)
                                                                            - h2a_voov(c, m, k, e) * t2a(e, a, j, i) * t2b(b, d, m, l)&  ! (ac)
                                                                            - h2a_voov(a, m, k, e) * t2a(e, b, j, i) * t2b(c, d, m, l)&  ! (bc)
                                                                            + h2a_voov(c, m, k, e) * t2a(e, b, j, i) * t2b(a, d, m, l)&  ! (ab)(bc)
                                                                            + h2a_voov(b, m, k, e) * t2a(e, a, j, i) * t2b(c, d, m, l)   ! (ac)(bc)
                                                            end do
                                                        end do
                                                        ! Diagram 8:   A(i/jk)A(c/ab) h2a(abef) * t2a(fcjk) * t2b(edil)
                                                        do e = 1, nua
                                                            do f = 1, nua
                                                                ! (1)
                                                                mm24 = mm24 + h2a_vvvv(a, b, e, f) * t2a(f, c, j, k) * t2b(e, d, i, l)&  ! (1)
                                                                            - h2a_vvvv(c, b, e, f) * t2a(f, a, j, k) * t2b(e, d, i, l)&  ! (ac)
                                                                            - h2a_vvvv(a, c, e, f) * t2a(f, b, j, k) * t2b(e, d, i, l)   ! (bc)
                                                                ! (ij)
                                                                mm24 = mm24 - h2a_vvvv(a, b, e, f) * t2a(f, c, i, k) * t2b(e, d, j, l)&  ! (1)
                                                                            + h2a_vvvv(c, b, e, f) * t2a(f, a, i, k) * t2b(e, d, j, l)&  ! (ac)
                                                                            + h2a_vvvv(a, c, e, f) * t2a(f, b, i, k) * t2b(e, d, j, l)   ! (bc)
                                                                ! (ik)
                                                                mm24 = mm24 - h2a_vvvv(a, b, e, f) * t2a(f, c, j, i) * t2b(e, d, k, l)&  ! (1)
                                                                            + h2a_vvvv(c, b, e, f) * t2a(f, a, j, i) * t2b(e, d, k, l)&  ! (ac)
                                                                            + h2a_vvvv(a, c, e, f) * t2a(f, b, j, i) * t2b(e, d, k, l)   ! (bc)
                                                            end do
                                                        end do
                                                        ! Diagram 9:  -A(ijk)A(a/bc) h2a(amie) * t2a(bcmk) * t2b(edjl)
                                                        do m = 1, noa
                                                            do e = 1, nua
                                                                ! (1)
                                                                mm24 = mm24 - h2a_voov(a, m, i, e) * t2a(b, c, m, k) * t2b(e, d, j, l)&  ! (1)
                                                                            + h2a_voov(a, m, j, e) * t2a(b, c, m, k) * t2b(e, d, i, l)&  ! (ij)
                                                                            + h2a_voov(a, m, k, e) * t2a(b, c, m, i) * t2b(e, d, j, l)&  ! (ik)
                                                                            + h2a_voov(a, m, i, e) * t2a(b, c, m, j) * t2b(e, d, k, l)&  ! (jk)
                                                                            - h2a_voov(a, m, k, e) * t2a(b, c, m, j) * t2b(e, d, i, l)&  ! (ij)(jk)
                                                                            - h2a_voov(a, m, j, e) * t2a(b, c, m, i) * t2b(e, d, k, l)   ! (ik)(jk)
                                                                ! (ab)
                                                                mm24 = mm24 + h2a_voov(b, m, i, e) * t2a(a, c, m, k) * t2b(e, d, j, l)&  ! (1)
                                                                            - h2a_voov(b, m, j, e) * t2a(a, c, m, k) * t2b(e, d, i, l)&  ! (ij)
                                                                            - h2a_voov(b, m, k, e) * t2a(a, c, m, i) * t2b(e, d, j, l)&  ! (ik)
                                                                            - h2a_voov(b, m, i, e) * t2a(a, c, m, j) * t2b(e, d, k, l)&  ! (jk)
                                                                            + h2a_voov(b, m, k, e) * t2a(a, c, m, j) * t2b(e, d, i, l)&  ! (ij)(jk)
                                                                            + h2a_voov(b, m, j, e) * t2a(a, c, m, i) * t2b(e, d, k, l)   ! (ik)(jk)
                                                                ! (ac)
                                                                mm24 = mm24 + h2a_voov(c, m, i, e) * t2a(b, a, m, k) * t2b(e, d, j, l)&  ! (1)
                                                                            - h2a_voov(c, m, j, e) * t2a(b, a, m, k) * t2b(e, d, i, l)&  ! (ij)
                                                                            - h2a_voov(c, m, k, e) * t2a(b, a, m, i) * t2b(e, d, j, l)&  ! (ik)
                                                                            - h2a_voov(c, m, i, e) * t2a(b, a, m, j) * t2b(e, d, k, l)&  ! (jk)
                                                                            + h2a_voov(c, m, k, e) * t2a(b, a, m, j) * t2b(e, d, i, l)&  ! (ij)(jk)
                                                                            + h2a_voov(c, m, j, e) * t2a(b, a, m, i) * t2b(e, d, k, l)   ! (ik)(jk)
                                                            end do
                                                        end do
                                                        ! Diagram 10:  A(k/ij)A(abc) h2b(adef) * t2a(ebij) * t2b(cfkl)
                                                        do e = 1, nua
                                                            do f = 1, nub
                                                                ! (1)
                                                                mm24 = mm24 + h2b_vvvv(a, d, e, f) * t2a(e, b, i, j) * t2b(c, f, k, l)&  ! (1)
                                                                            - h2b_vvvv(b, d, e, f) * t2a(e, a, i, j) * t2b(c, f, k, l)&  ! (ab)
                                                                            - h2b_vvvv(c, d, e, f) * t2a(e, b, i, j) * t2b(a, f, k, l)&  ! (ac)
                                                                            - h2b_vvvv(a, d, e, f) * t2a(e, c, i, j) * t2b(b, f, k, l)&  ! (bc)
                                                                            + h2b_vvvv(c, d, e, f) * t2a(e, a, i, j) * t2b(b, f, k, l)&  ! (ab)(bc)
                                                                            + h2b_vvvv(b, d, e, f) * t2a(e, c, i, j) * t2b(a, f, k, l)   ! (ac)(bc)
                                                                ! (ik)
                                                                mm24 = mm24 - h2b_vvvv(a, d, e, f) * t2a(e, b, k, j) * t2b(c, f, i, l)&  ! (1)
                                                                            + h2b_vvvv(b, d, e, f) * t2a(e, a, k, j) * t2b(c, f, i, l)&  ! (ab)
                                                                            + h2b_vvvv(c, d, e, f) * t2a(e, b, k, j) * t2b(a, f, i, l)&  ! (ac)
                                                                            + h2b_vvvv(a, d, e, f) * t2a(e, c, k, j) * t2b(b, f, i, l)&  ! (bc)
                                                                            - h2b_vvvv(c, d, e, f) * t2a(e, a, k, j) * t2b(b, f, i, l)&  ! (ab)(bc)
                                                                            - h2b_vvvv(b, d, e, f) * t2a(e, c, k, j) * t2b(a, f, i, l)   ! (ac)(bc)
                                                                ! (jk)
                                                                mm24 = mm24 - h2b_vvvv(a, d, e, f) * t2a(e, b, i, k) * t2b(c, f, j, l)&  ! (1)
                                                                            + h2b_vvvv(b, d, e, f) * t2a(e, a, i, k) * t2b(c, f, j, l)&  ! (ab)
                                                                            + h2b_vvvv(c, d, e, f) * t2a(e, b, i, k) * t2b(a, f, j, l)&  ! (ac)
                                                                            + h2b_vvvv(a, d, e, f) * t2a(e, c, i, k) * t2b(b, f, j, l)&  ! (bc)
                                                                            - h2b_vvvv(c, d, e, f) * t2a(e, a, i, k) * t2b(b, f, j, l)&  ! (ab)(bc)
                                                                            - h2b_vvvv(b, d, e, f) * t2a(e, c, i, k) * t2b(a, f, j, l)   ! (ac)(bc)
                                                            end do
                                                        end do

                                                        !!! L4B Computation !!!
                                                        ! Diagram 1:  A(k/ij)A(c/ab) h2a(ijab) * l2b(cdkl)
                                                        ! (1)
                                                        l4 = l4 + h2a_oovv(i, j, a, b) * l2b(c, d, k, l)&  ! (1)
                                                                - h2a_oovv(i, j, c, b) * l2b(a, d, k, l)&  ! (ac)
                                                                - h2a_oovv(i, j, a, c) * l2b(b, d, k, l)   ! (bc)
                                                        ! (ik)
                                                        l4 = l4 - h2a_oovv(k, j, a, b) * l2b(c, d, i, l)&  ! (1)
                                                                + h2a_oovv(k, j, c, b) * l2b(a, d, i, l)&  ! (ac)
                                                                + h2a_oovv(k, j, a, c) * l2b(b, d, i, l)   ! (bc)
                                                        ! (jk)
                                                        l4 = l4 - h2a_oovv(i, k, a, b) * l2b(c, d, j, l)&  ! (1)
                                                                + h2a_oovv(i, k, c, b) * l2b(a, d, j, l)&  ! (ac)
                                                                + h2a_oovv(i, k, a, c) * l2b(b, d, j, l)   ! (bc)
                                                        ! Diagram 2:  A(k/ij)A(c/ab) h2b(klcd) * l2a(abij)
                                                        ! (1)
                                                        l4 = l4 + h2b_oovv(k, l, c, d) * l2a(a, b, i, j)&  ! (1)
                                                                - h2b_oovv(k, l, a, d) * l2a(c, b, i, j)&  ! (ac)
                                                                - h2b_oovv(k, l, b, d) * l2a(a, c, i, j)   ! (bc)
                                                        ! (ik)
                                                        l4 = l4 - h2b_oovv(i, l, c, d) * l2a(a, b, k, j)&  ! (1)
                                                                + h2b_oovv(i, l, a, d) * l2a(c, b, k, j)&  ! (ac)
                                                                + h2b_oovv(i, l, b, d) * l2a(a, c, k, j)   ! (bc)
                                                        ! (jk)
                                                        l4 = l4 - h2b_oovv(j, l, c, d) * l2a(a, b, i, k)&  ! (1)
                                                                + h2b_oovv(j, l, a, d) * l2a(c, b, i, k)&  ! (ac)
                                                                + h2b_oovv(j, l, b, d) * l2a(a, c, i, k)   ! (bc)

                                                        temp = mm24 * l4

                                                        denom = fA_oo(i,i) + fA_oo(j,j) + fA_oo(k,k) + fB_oo(l,l)&
                                                        - fA_vv(a,a) - fA_vv(b,b) - fA_vv(c,c) - fB_vv(d,d)

                                                        deltaA = deltaA + temp/denom

                                                        denom = H1A_oo(i,i) + H1A_oo(j,j) + H1A_oo(k,k) + H1B_oo(l,l)&
                                                        - H1A_vv(a,a) - H1A_vv(b,b) - H1A_vv(c,c) - H1B_vv(d,d)

                                                        deltaB = deltaB + temp/denom

                                                        denom = denom &
                                                        -H2A_oooo(i,j,i,j)-H2A_oooo(i,k,i,k)-H2A_oooo(j,k,j,k)-H2A_voov(a,i,i,a)&
                                                        -H2A_voov(a,j,j,a)-H2A_voov(a,k,k,a)-H2A_voov(b,i,i,b)-H2A_voov(b,j,j,b)&
                                                        -H2A_voov(b,k,k,b)-H2A_voov(c,i,i,c)-H2A_voov(c,j,j,c)-H2A_voov(c,k,k,c)&
                                                        -H2A_vvvv(a,b,a,b)-H2A_vvvv(a,c,a,c)-H2A_vvvv(b,c,b,c)-H2B_oooo(i,l,i,l)&
                                                        -H2B_oooo(j,l,j,l)-H2B_oooo(k,l,k,l)+H2B_ovov(i,d,i,d)+H2B_ovov(j,d,j,d)&
                                                        +H2B_ovov(k,d,k,d)+H2B_vovo(a,l,a,l)+H2B_vovo(b,l,b,l)+H2B_vovo(c,l,c,l)&
                                                        -H2B_vvvv(a,d,a,d)-H2B_vvvv(b,d,b,d)-H2B_vvvv(c,d,c,d)-H2C_voov(d,l,l,d)

                                                        deltaC = deltaC + temp/denom

                                                        denom = denom &
                                                        +D3A_O(a,i,j)+D3A_O(a,i,k)+D3A_O(a,j,k)+D3A_O(b,i,j)&
                                                        +D3A_O(b,i,k)+D3A_O(b,j,k)+D3A_O(c,i,j)+D3A_O(c,i,k)&
                                                        +D3A_O(c,j,k)-D3A_V(a,i,b)-D3A_V(a,j,b)-D3A_V(a,k,b)&
                                                        -D3A_V(a,i,c)-D3A_V(a,j,c)-D3A_V(a,k,c)-D3A_V(b,i,c)&
                                                        -D3A_V(b,j,c)-D3A_V(b,k,c)+D3B_O(a,i,l)+D3B_O(a,j,l)&
                                                        +D3B_O(a,k,l)+D3B_O(b,i,l)+D3B_O(b,j,l)+D3B_O(b,k,l)&
                                                        +D3B_O(c,i,l)+D3B_O(c,j,l)+D3B_O(c,k,l)-D3B_V(a,i,d)&
                                                        -D3B_V(a,j,d)-D3B_V(a,k,d)-D3B_V(b,i,d)-D3B_V(b,j,d)&
                                                        -D3B_V(b,k,d)-D3B_V(c,i,d)-D3B_V(c,j,d)-D3B_V(c,k,d)&
                                                        +D3C_O(d,i,l)+D3C_O(d,j,l)+D3C_O(d,k,l)-D3C_V(a,l,d)&
                                                        -D3C_V(b,l,d)-D3C_V(c,l,d)

                                                        deltaD = deltaD + temp/denom

                                                    end do
                                                end do
                                            end do
                                        end do
                                    end do
                                end do
                            end do
                        end do
                        !print*, "Error = ", error

              end subroutine crcc24B

              subroutine crcc24C(deltaA,deltaB,deltaC,deltaD,&
                              t2a,t2b,t2c,l2a,l2b,l2c,&
                              fA_oo,fA_vv,fB_oo,fB_vv,&
                              H1A_oo,H1A_vv,H1B_oo,H1B_vv,&
                              H2A_voov,H2A_oooo,H2A_vvvv,H2A_oovv,&
                              H2B_voov,H2B_ovov,H2B_vovo,H2B_ovvo,H2B_oooo,H2B_vvvv,H2B_oovv,&
                              H2C_voov,H2C_oooo,H2C_vvvv,H2C_oovv,&
                              D3A_O,D3A_V,D3B_O,D3B_V,D3C_O,D3C_V,D3D_O,D3D_V,&
                              noa,nua,nob,nub)

                        real(kind=8), intent(out) :: deltaA, deltaB, deltaC, deltaD
                        integer, intent(in) :: noa, nua, nob, nub
                        real(kind=8), intent(in) :: t2a(1:nua,1:nua,1:noa,1:noa),t2b(1:nua,1:nub,1:noa,1:nob),&
                        t2c(1:nub,1:nub,1:nob,1:nob),l2a(1:nua,1:nua,1:noa,1:noa),l2b(1:nua,1:nub,1:noa,1:nob),&
                        l2c(1:nub,1:nub,1:nob,1:nob),&
                        fA_oo(1:noa,1:noa),fA_vv(1:nua,1:nua),fB_oo(1:nob,1:nob),fB_vv(1:nub,1:nub),&
                        H1A_oo(1:noa,1:noa),H1A_vv(1:nua,1:nua),H1B_oo(1:nob,1:nob),H1B_vv(1:nub,1:nub),&
                        H2A_voov(1:nua,1:noa,1:noa,1:nua),&
                        H2A_oooo(1:noa,1:noa,1:noa,1:noa),&
                        H2A_vvvv(1:nua,1:nua,1:nua,1:nua),&
                        H2A_oovv(1:noa,1:noa,1:nua,1:nua),&
                        H2B_voov(1:nua,1:nob,1:noa,1:nub),&
                        H2B_ovov(1:noa,1:nub,1:noa,1:nub),&
                        H2B_vovo(1:nua,1:nob,1:nua,1:nob),&
                        H2B_ovvo(1:noa,1:nub,1:nua,1:nob),&
                        H2B_oooo(1:noa,1:nob,1:noa,1:nob),&
                        H2B_vvvv(1:nua,1:nub,1:nua,1:nub),&
                        H2B_oovv(1:noa,1:nob,1:nua,1:nub),&
                        H2C_voov(1:nub,1:nob,1:nob,1:nub),&
                        H2C_oooo(1:nob,1:nob,1:nob,1:nob),&
                        H2C_vvvv(1:nub,1:nub,1:nub,1:nub),&
                        H2C_oovv(1:nob,1:nob,1:nub,1:nub),&
                        D3A_O(1:nua,1:noa,1:noa),&
                        D3A_V(1:nua,1:noa,1:nua),&
                        D3B_O(1:nua,1:noa,1:nob),&
                        D3B_V(1:nua,1:noa,1:nub),&
                        D3C_O(1:nub,1:noa,1:nob),&
                        D3C_V(1:nua,1:nob,1:nub),&
                        D3D_O(1:nub,1:nob,1:nob),&
                        D3D_V(1:nub,1:nob,1:nub)

                        !real(kind=8), intent(in) :: test_array(1:nua,1:nua,1:nub,1:nub,1:noa,1:noa,1:nob,1:nob)

                        integer :: i, j, k, l, a, b, c, d, m, n, e, f
                        real(kind=8) :: denom, temp, mm24, l4
                        !real(kind=8) :: error

                        deltaA = 0.0d0
                        deltaB = 0.0d0
                        deltaC = 0.0d0
                        deltaD = 0.0d0

                        !error = 0.0d0
                        do i = 1, noa
                            do j = i+1, noa
                                do k = 1, nob
                                    do l = k+1, nob
                                        do a = 1, nua
                                            do b = a+1, nua
                                                do c = 1, nub
                                                    do d = c+1, nub

                                                        mm24 = 0.0d0
                                                        l4 = 0.0d0

                                                        !!! MM(2,4)C Computation !!!
                                                        ! Diagram 1:  -A(ij)A(kl)A(ab)A(cd) h2c(cmke) * t2b(adim) * t2b(bejl)
                                                        do m = 1, nob
                                                            do e = 1, nub
                                                                ! (1)
                                                                mm24 = mm24 - h2c_voov(c, m, k, e) * t2b(a, d, i, m) * t2b(b, e, j, l)&  ! (1)
                                                                            + h2c_voov(c, m, k, e) * t2b(b, d, i, m) * t2b(a, e, j, l)&  ! (ab)
                                                                            + h2c_voov(d, m, k, e) * t2b(a, c, i, m) * t2b(b, e, j, l)&  ! (cd)
                                                                            - h2c_voov(d, m, k, e) * t2b(b, c, i, m) * t2b(a, e, j, l)   ! (ab)(cd)
                                                                ! (ij)
                                                                mm24 = mm24 + h2c_voov(c, m, k, e) * t2b(a, d, j, m) * t2b(b, e, i, l)&  ! (1)
                                                                            - h2c_voov(c, m, k, e) * t2b(b, d, j, m) * t2b(a, e, i, l)&  ! (ab)
                                                                            - h2c_voov(d, m, k, e) * t2b(a, c, j, m) * t2b(b, e, i, l)&  ! (cd)
                                                                            + h2c_voov(d, m, k, e) * t2b(b, c, j, m) * t2b(a, e, i, l)   ! (ab)(cd)
                                                                ! (kl)
                                                                mm24 = mm24 + h2c_voov(c, m, l, e) * t2b(a, d, i, m) * t2b(b, e, j, k)&  ! (1)
                                                                            - h2c_voov(c, m, l, e) * t2b(b, d, i, m) * t2b(a, e, j, k)&  ! (ab)
                                                                            - h2c_voov(d, m, l, e) * t2b(a, c, i, m) * t2b(b, e, j, k)&  ! (cd)
                                                                            + h2c_voov(d, m, l, e) * t2b(b, c, i, m) * t2b(a, e, j, k)   ! (ab)(cd)
                                                                ! (ij)(kl)
                                                                mm24 = mm24 - h2c_voov(c, m, l, e) * t2b(a, d, j, m) * t2b(b, e, i, k)&  ! (1)
                                                                            + h2c_voov(c, m, l, e) * t2b(b, d, j, m) * t2b(a, e, i, k)&  ! (ab)
                                                                            + h2c_voov(d, m, l, e) * t2b(a, c, j, m) * t2b(b, e, i, k)&  ! (cd)
                                                                            - h2c_voov(d, m, l, e) * t2b(b, c, j, m) * t2b(a, e, i, k)   ! (ab)(cd)
                                                            end do
                                                        end do
                                                        ! Diagram 2:  -A(ij)A(kl)A(ab)A(cd) h2a(amie) * t2b(bcmk) * t2b(edjl)
                                                        do m = 1, noa
                                                            do e = 1, nua
                                                                ! (1)
                                                                mm24 = mm24 - h2a_voov(a, m, i, e) * t2b(b, c, m, k) * t2b(e, d, j, l)&  ! (1)
                                                                            + h2a_voov(b, m, i, e) * t2b(a, c, m, k) * t2b(e, d, j, l)&  ! (ab)
                                                                            + h2a_voov(a, m, i, e) * t2b(b, d, m, k) * t2b(e, c, j, l)&  ! (cd)
                                                                            - h2a_voov(b, m, i, e) * t2b(a, d, m, k) * t2b(e, c, j, l)   ! (ab)(cd)
                                                                ! (ij)
                                                                mm24 = mm24 + h2a_voov(a, m, j, e) * t2b(b, c, m, k) * t2b(e, d, i, l)&  ! (1)
                                                                            - h2a_voov(b, m, j, e) * t2b(a, c, m, k) * t2b(e, d, i, l)&  ! (ab)
                                                                            - h2a_voov(a, m, j, e) * t2b(b, d, m, k) * t2b(e, c, i, l)&  ! (cd)
                                                                            + h2a_voov(b, m, j, e) * t2b(a, d, m, k) * t2b(e, c, i, l)   ! (ab)(cd)
                                                                ! (kl)
                                                                mm24 = mm24 + h2a_voov(a, m, i, e) * t2b(b, c, m, l) * t2b(e, d, j, k)&  ! (1)
                                                                            - h2a_voov(b, m, i, e) * t2b(a, c, m, l) * t2b(e, d, j, k)&  ! (ab)
                                                                            - h2a_voov(a, m, i, e) * t2b(b, d, m, l) * t2b(e, c, j, k)&  ! (cd)
                                                                            + h2a_voov(b, m, i, e) * t2b(a, d, m, l) * t2b(e, c, j, k)   ! (ab)(cd)
                                                                ! (ij)(kl)
                                                                mm24 = mm24 - h2a_voov(a, m, j, e) * t2b(b, c, m, l) * t2b(e, d, i, k)&  ! (1)
                                                                            + h2a_voov(b, m, j, e) * t2b(a, c, m, l) * t2b(e, d, i, k)&  ! (ab)
                                                                            + h2a_voov(a, m, j, e) * t2b(b, d, m, l) * t2b(e, c, i, k)&  ! (cd)
                                                                            - h2a_voov(b, m, j, e) * t2b(a, d, m, l) * t2b(e, c, i, k)   ! (ab)(cd)
                                                            end do
                                                        end do
                                                        ! Diagram 3:  -A(kl)A(ab)A(cd) h2b(mcek) * t2a(aeij) * t2b(bdml)
                                                        do m = 1, noa
                                                            do e = 1, nua
                                                                ! (1)
                                                                mm24 = mm24 - h2b_ovvo(m, c, e, k) * t2a(a, e, i, j) * t2b(b, d, m, l)&  ! (1)
                                                                            + h2b_ovvo(m, c, e, k) * t2a(b, e, i, j) * t2b(a, d, m, l)&  ! (ab)
                                                                            + h2b_ovvo(m, d, e, k) * t2a(a, e, i, j) * t2b(b, c, m, l)&  ! (cd)
                                                                            - h2b_ovvo(m, d, e, k) * t2a(b, e, i, j) * t2b(a, c, m, l)   ! (ab)(cd)
                                                                ! (kl)
                                                                mm24 = mm24 + h2b_ovvo(m, c, e, l) * t2a(a, e, i, j) * t2b(b, d, m, k)&  ! (1)
                                                                            - h2b_ovvo(m, c, e, l) * t2a(b, e, i, j) * t2b(a, d, m, k)&  ! (ab)
                                                                            - h2b_ovvo(m, d, e, l) * t2a(a, e, i, j) * t2b(b, c, m, k)&  ! (cd)
                                                                            + h2b_ovvo(m, d, e, l) * t2a(b, e, i, j) * t2b(a, c, m, k)   ! (ab)(cd)
                                                            end do
                                                        end do
                                                        ! Diagram 4:  -A(ij)A(ab)A(cd) h2b(amie) * t2b(bdjm) * t2c(cekl)
                                                        do m = 1, nob
                                                            do e = 1, nub
                                                                ! (1)
                                                                mm24 = mm24 - h2b_voov(a, m, i, e) * t2b(b, d, j, m) * t2c(c, e, k, l)&  ! (1)
                                                                            + h2b_voov(b, m, i, e) * t2b(a, d, j, m) * t2c(c, e, k, l)&  ! (ab)
                                                                            + h2b_voov(a, m, i, e) * t2b(b, c, j, m) * t2c(d, e, k, l)&  ! (cd)
                                                                            - h2b_voov(b, m, i, e) * t2b(a, c, j, m) * t2c(d, e, k, l)   ! (ab)(cd)
                                                                ! (ij)
                                                                mm24 = mm24 + h2b_voov(a, m, j, e) * t2b(b, d, i, m) * t2c(c, e, k, l)&  ! (1)
                                                                            - h2b_voov(b, m, j, e) * t2b(a, d, i, m) * t2c(c, e, k, l)&  ! (ab)
                                                                            - h2b_voov(a, m, j, e) * t2b(b, c, i, m) * t2c(d, e, k, l)&  ! (cd)
                                                                            + h2b_voov(b, m, j, e) * t2b(a, c, i, m) * t2c(d, e, k, l)   ! (ab)(cd)
                                                            end do
                                                        end do
                                                        ! Diagram 5:  -A(ij)A(kl)A(cd) h2b(mcek) * t2a(abim) * t2b(edjl)
                                                        do m = 1, noa
                                                            do e = 1, nua
                                                                ! (1)
                                                                mm24 = mm24 - h2b_ovvo(m, c, e, k) * t2a(a, b, i, m) * t2b(e, d, j, l)&  ! (1)
                                                                            + h2b_ovvo(m, c, e, k) * t2a(a, b, j, m) * t2b(e, d, i, l)&  ! (ij)
                                                                            + h2b_ovvo(m, c, e, l) * t2a(a, b, i, m) * t2b(e, d, j, k)&  ! (kl)
                                                                            - h2b_ovvo(m, c, e, l) * t2a(a, b, j, m) * t2b(e, d, i, k)   ! (ij)(kl)
                                                                ! (cd)
                                                                mm24 = mm24 + h2b_ovvo(m, d, e, k) * t2a(a, b, i, m) * t2b(e, c, j, l)&  ! (1)
                                                                            - h2b_ovvo(m, d, e, k) * t2a(a, b, j, m) * t2b(e, c, i, l)&  ! (ij)
                                                                            - h2b_ovvo(m, d, e, l) * t2a(a, b, i, m) * t2b(e, c, j, k)&  ! (kl)
                                                                            + h2b_ovvo(m, d, e, l) * t2a(a, b, j, m) * t2b(e, c, i, k)   ! (ij)(kl)
                                                            end do
                                                        end do
                                                        ! Diagram 6:  -A(ij)A(kl)A(ab) h2b(amie) * t2c(cdkm) * t2b(bejl)
                                                        do m = 1, nob
                                                            do e = 1, nub
                                                                ! (1)
                                                                mm24 = mm24 - h2b_voov(a, m, i, e) * t2c(c, d, k, m) * t2b(b, e, j, l)&  ! (1)
                                                                            + h2b_voov(a, m, j, e) * t2c(c, d, k, m) * t2b(b, e, i, l)&  ! (ij)
                                                                            + h2b_voov(a, m, i, e) * t2c(c, d, l, m) * t2b(b, e, j, k)&  ! (kl)
                                                                            - h2b_voov(a, m, j, e) * t2c(c, d, l, m) * t2b(b, e, i, k)   ! (ij)(kl)
                                                                ! (ab)
                                                                mm24 = mm24 + h2b_voov(b, m, i, e) * t2c(c, d, k, m) * t2b(a, e, j, l)&  ! (1)
                                                                            - h2b_voov(b, m, j, e) * t2c(c, d, k, m) * t2b(a, e, i, l)&  ! (ij)
                                                                            - h2b_voov(b, m, i, e) * t2c(c, d, l, m) * t2b(a, e, j, k)&  ! (kl)
                                                                            + h2b_voov(b, m, j, e) * t2c(c, d, l, m) * t2b(a, e, i, k)   ! (ij)(kl)
                                                            end do
                                                        end do
                                                        ! Diagram 7:  -A(ij)A(kl)A(ab)A(cd) h2b(bmel) * t2b(adim) * t2b(ecjk)
                                                        do m = 1, nob
                                                            do e = 1, nua
                                                                ! (1)
                                                                mm24 = mm24 - h2b_vovo(b, m, e, l) * t2b(a, d, i, m) * t2b(e, c, j, k)&  ! (1)
                                                                            + h2b_vovo(b, m, e, l) * t2b(a, d, j, m) * t2b(e, c, i, k)&  ! (ij)
                                                                            + h2b_vovo(b, m, e, k) * t2b(a, d, i, m) * t2b(e, c, j, l)&  ! (kl)
                                                                            - h2b_vovo(b, m, e, k) * t2b(a, d, j, m) * t2b(e, c, i, l)   ! (ij)(kl)
                                                                ! (ab)
                                                                mm24 = mm24 + h2b_vovo(a, m, e, l) * t2b(b, d, i, m) * t2b(e, c, j, k)&  ! (1)
                                                                            - h2b_vovo(a, m, e, l) * t2b(b, d, j, m) * t2b(e, c, i, k)&  ! (ij)
                                                                            - h2b_vovo(a, m, e, k) * t2b(b, d, i, m) * t2b(e, c, j, l)&  ! (kl)
                                                                            + h2b_vovo(a, m, e, k) * t2b(b, d, j, m) * t2b(e, c, i, l)   ! (ij)(kl)
                                                                ! (cd)
                                                                mm24 = mm24 + h2b_vovo(b, m, e, l) * t2b(a, c, i, m) * t2b(e, d, j, k)&  ! (1)
                                                                            - h2b_vovo(b, m, e, l) * t2b(a, c, j, m) * t2b(e, d, i, k)&  ! (ij)
                                                                            - h2b_vovo(b, m, e, k) * t2b(a, c, i, m) * t2b(e, d, j, l)&  ! (kl)
                                                                            + h2b_vovo(b, m, e, k) * t2b(a, c, j, m) * t2b(e, d, i, l)   ! (ij)(kl)
                                                                ! (ab)(cd)
                                                                mm24 = mm24 - h2b_vovo(a, m, e, l) * t2b(b, c, i, m) * t2b(e, d, j, k)&  ! (1)
                                                                            + h2b_vovo(a, m, e, l) * t2b(b, c, j, m) * t2b(e, d, i, k)&  ! (ij)
                                                                            + h2b_vovo(a, m, e, k) * t2b(b, c, i, m) * t2b(e, d, j, l)&  ! (kl)
                                                                            - h2b_vovo(a, m, e, k) * t2b(b, c, j, m) * t2b(e, d, i, l)   ! (ij)(kl)

                                                            end do
                                                        end do
                                                        ! Diagram 8:  -A(ij)A(kl)A(ab)A(cd) h2b(mdje) * t2b(bcmk) * t2b(aeil)
                                                        do m = 1, noa
                                                            do e = 1, nub
                                                                ! (1)
                                                                mm24 = mm24 - h2b_ovov(m, d, j, e) * t2b(b, c, m, k) * t2b(a, e, i, l)&  ! (1)
                                                                            + h2b_ovov(m, d, i, e) * t2b(b, c, m, k) * t2b(a, e, j, l)&  ! (ij)
                                                                            + h2b_ovov(m, d, j, e) * t2b(b, c, m, l) * t2b(a, e, i, k)&  ! (kl)
                                                                            - h2b_ovov(m, d, i, e) * t2b(b, c, m, l) * t2b(a, e, j, k)   ! (ij)(kl)
                                                                ! (ab)
                                                                mm24 = mm24 + h2b_ovov(m, d, j, e) * t2b(a, c, m, k) * t2b(b, e, i, l)&  ! (1)
                                                                            - h2b_ovov(m, d, i, e) * t2b(a, c, m, k) * t2b(b, e, j, l)&  ! (ij)
                                                                            - h2b_ovov(m, d, j, e) * t2b(a, c, m, l) * t2b(b, e, i, k)&  ! (kl)
                                                                            + h2b_ovov(m, d, i, e) * t2b(a, c, m, l) * t2b(b, e, j, k)   ! (ij)(kl)
                                                                ! (cd)
                                                                mm24 = mm24 + h2b_ovov(m, c, j, e) * t2b(b, d, m, k) * t2b(a, e, i, l)&  ! (1)
                                                                            - h2b_ovov(m, c, i, e) * t2b(b, d, m, k) * t2b(a, e, j, l)&  ! (ij)
                                                                            - h2b_ovov(m, c, j, e) * t2b(b, d, m, l) * t2b(a, e, i, k)&  ! (kl)
                                                                            + h2b_ovov(m, c, i, e) * t2b(b, d, m, l) * t2b(a, e, j, k)   ! (ij)(kl)
                                                                ! (ab)(cd)
                                                                mm24 = mm24 - h2b_ovov(m, c, j, e) * t2b(a, d, m, k) * t2b(b, e, i, l)&  ! (1)
                                                                            + h2b_ovov(m, c, i, e) * t2b(a, d, m, k) * t2b(b, e, j, l)&  ! (ij)
                                                                            + h2b_ovov(m, c, j, e) * t2b(a, d, m, l) * t2b(b, e, i, k)&  ! (kl)
                                                                            - h2b_ovov(m, c, i, e) * t2b(a, d, m, l) * t2b(b, e, j, k)   ! (ij)(kl)
                                                            end do
                                                        end do
                                                        ! Diagram 9:  -A(ij)A(cd) h2b(mdje) * t2a(abim) * t2c(cekl)
                                                        do m = 1, noa
                                                            do e = 1, nub
                                                                ! (1)
                                                                mm24 = mm24 - h2b_ovov(m, d, j, e) * t2a(a, b, i, m) * t2c(c, e, k, l)&  ! (1)
                                                                            + h2b_ovov(m, d, i, e) * t2a(a, b, j, m) * t2c(c, e, k, l)   ! (ij)
                                                                ! (cd)
                                                                mm24 = mm24 + h2b_ovov(m, c, j, e) * t2a(a, b, i, m) * t2c(d, e, k, l)&  ! (1)
                                                                            - h2b_ovov(m, c, i, e) * t2a(a, b, j, m) * t2c(d, e, k, l)   ! (ij)
                                                            end do
                                                        end do
                                                        ! Diagram 10: -A(kl)A(ab) h2b(bmel) * t2a(aeij) * t2c(cdkm)
                                                        do m = 1, nob
                                                            do e = 1, nua
                                                                ! (1)
                                                                mm24 = mm24 - h2b_vovo(b, m, e, l) * t2a(a, e, i, j) * t2c(c, d, k, m)&  ! (1)
                                                                            + h2b_vovo(b, m, e, k) * t2a(a, e, i, j) * t2c(c, d, l, m)   ! (kl)
                                                                ! (ab)
                                                                mm24 = mm24 + h2b_vovo(a, m, e, l) * t2a(b, e, i, j) * t2c(c, d, k, m)&  ! (1)
                                                                            - h2b_vovo(a, m, e, k) * t2a(b, e, i, j) * t2c(c, d, l, m)   ! (kl)
                                                            end do
                                                        end do
                                                        ! Diagram 11:  A(kl)A(ab) h2a(mnij) * t2b(acmk) * t2b(bdnl)
                                                        do m = 1, noa
                                                            do n = 1, noa
                                                                ! (1)
                                                                mm24 = mm24 + h2a_oooo(m, n, i, j) * t2b(a, c, m, k) * t2b(b, d, n, l)&  ! (1)
                                                                            - h2a_oooo(m, n, i, j) * t2b(a, c, m, l) * t2b(b, d, n, k)   ! (kl)
                                                                ! (ab)
                                                                mm24 = mm24 - h2a_oooo(m, n, i, j) * t2b(b, c, m, k) * t2b(a, d, n, l)&  ! (1)
                                                                            + h2a_oooo(m, n, i, j) * t2b(b, c, m, l) * t2b(a, d, n, k)   ! (kl)
                                                            end do
                                                        end do
                                                        ! Diagram 12:  A(ij)A(kl) h2a(abef) * t2b(ecik) * t2b(fdjl)
                                                        do e = 1, nua
                                                            do f = 1, nua
                                                                ! (1)
                                                                mm24 = mm24 + h2a_vvvv(a, b, e, f) * t2b(e, c, i, k) * t2b(f, d, j, l)&  ! (1)
                                                                            - h2a_vvvv(a, b, e, f) * t2b(e, c, j, k) * t2b(f, d, i, l)&  ! (ij)
                                                                            - h2a_vvvv(a, b, e, f) * t2b(e, c, i, l) * t2b(f, d, j, k)&  ! (kl)
                                                                            + h2a_vvvv(a, b, e, f) * t2b(e, c, j, l) * t2b(f, d, i, k)   ! (ij)(kl)
                                                            end do
                                                        end do
                                                        ! Diagram 13:  A(ij)A(kl) h2b(mnik) * t2a(abmj) * t2c(cdnl)
                                                        do m = 1, noa
                                                            do n = 1, nob
                                                                ! (1)
                                                                mm24 = mm24 + h2b_oooo(m, n, i, k) * t2a(a, b, m, j) * t2c(c, d, n, l)&  ! (1)
                                                                            - h2b_oooo(m, n, j, k) * t2a(a, b, m, i) * t2c(c, d, n, l)&  ! (ij)
                                                                            - h2b_oooo(m, n, i, l) * t2a(a, b, m, j) * t2c(c, d, n, k)&  ! (kl)
                                                                            + h2b_oooo(m, n, j, l) * t2a(a, b, m, i) * t2c(c, d, n, k)   ! (ij)(kl)
                                                            end do
                                                        end do
                                                        ! Diagram 14:  A(ab)A(cd) h2b(acef) * t2a(ebij) * t2c(fdkl)
                                                        do e = 1, nua
                                                            do f = 1, nub
                                                                ! (1)
                                                                mm24 = mm24 + h2b_vvvv(a, c, e, f) * t2a(e, b, i, j) * t2c(f, d, k, l)&  ! (1)
                                                                            - h2b_vvvv(b, c, e, f) * t2a(e, a, i, j) * t2c(f, d, k, l)&  ! (ab)
                                                                            - h2b_vvvv(a, d, e, f) * t2a(e, b, i, j) * t2c(f, c, k, l)&  ! (cd)
                                                                            + h2b_vvvv(b, d, e, f) * t2a(e, a, i, j) * t2c(f, c, k, l)   ! (ab)(cd)
                                                            end do
                                                        end do
                                                        ! Diagram 15:  A(ij)A(kl)A(ab)A(cd) h2b(mnik) * t2b(adml) * t2b(bcjn)
                                                        do m = 1, noa
                                                            do n = 1, nob
                                                                ! (1)
                                                                mm24 = mm24 + h2b_oooo(m, n, i, k) * t2b(a, d, m, l) * t2b(b, c, j, n)&  ! (1)
                                                                            - h2b_oooo(m, n, j, k) * t2b(a, d, m, l) * t2b(b, c, i, n)&  ! (ij)
                                                                            - h2b_oooo(m, n, i, l) * t2b(a, d, m, k) * t2b(b, c, j, n)&  ! (kl)
                                                                            + h2b_oooo(m, n, j, l) * t2b(a, d, m, k) * t2b(b, c, i, n)   ! (ij)(kl)
                                                                ! (ab)
                                                                mm24 = mm24 - h2b_oooo(m, n, i, k) * t2b(b, d, m, l) * t2b(a, c, j, n)&  ! (1)
                                                                            + h2b_oooo(m, n, j, k) * t2b(b, d, m, l) * t2b(a, c, i, n)&  ! (ij)
                                                                            + h2b_oooo(m, n, i, l) * t2b(b, d, m, k) * t2b(a, c, j, n)&  ! (kl)
                                                                            - h2b_oooo(m, n, j, l) * t2b(b, d, m, k) * t2b(a, c, i, n)   ! (ij)(kl)
                                                                ! (cd)
                                                                mm24 = mm24 - h2b_oooo(m, n, i, k) * t2b(a, c, m, l) * t2b(b, d, j, n)&  ! (1)
                                                                            + h2b_oooo(m, n, j, k) * t2b(a, c, m, l) * t2b(b, d, i, n)&  ! (ij)
                                                                            + h2b_oooo(m, n, i, l) * t2b(a, c, m, k) * t2b(b, d, j, n)&  ! (kl)
                                                                            - h2b_oooo(m, n, j, l) * t2b(a, c, m, k) * t2b(b, d, i, n)   ! (ij)(kl)
                                                                ! (ab)(cd)
                                                                mm24 = mm24 + h2b_oooo(m, n, i, k) * t2b(b, c, m, l) * t2b(a, d, j, n)&  ! (1)
                                                                            - h2b_oooo(m, n, j, k) * t2b(b, c, m, l) * t2b(a, d, i, n)&  ! (ij)
                                                                            - h2b_oooo(m, n, i, l) * t2b(b, c, m, k) * t2b(a, d, j, n)&  ! (kl)
                                                                            + h2b_oooo(m, n, j, l) * t2b(b, c, m, k) * t2b(a, d, i, n)   ! (ij)(kl)
                                                            end do
                                                        end do
                                                        ! Diagram 16:  A(ij)A(kl)A(ab)A(cd) h2b(acef) * t2b(edil) * t2b(bfjk)
                                                        do e = 1, nua
                                                            do f = 1, nub
                                                                ! (1)
                                                                mm24 = mm24 + h2b_vvvv(a, c, e, f) * t2b(e, d, i, l) * t2b(b, f, j, k)&  ! (1)
                                                                            - h2b_vvvv(a, c, e, f) * t2b(e, d, j, l) * t2b(b, f, i, k)&  ! (ij)
                                                                            - h2b_vvvv(a, c, e, f) * t2b(e, d, i, k) * t2b(b, f, j, l)&  ! (kl)
                                                                            + h2b_vvvv(a, c, e, f) * t2b(e, d, j, k) * t2b(b, f, i, l)   ! (ij)(kl)
                                                                ! (ab)
                                                                mm24 = mm24 - h2b_vvvv(b, c, e, f) * t2b(e, d, i, l) * t2b(a, f, j, k)&  ! (1)
                                                                            + h2b_vvvv(b, c, e, f) * t2b(e, d, j, l) * t2b(a, f, i, k)&  ! (ij)
                                                                            + h2b_vvvv(b, c, e, f) * t2b(e, d, i, k) * t2b(a, f, j, l)&  ! (kl)
                                                                            - h2b_vvvv(b, c, e, f) * t2b(e, d, j, k) * t2b(a, f, i, l)   ! (ij)(kl)
                                                                ! (cd)
                                                                mm24 = mm24 - h2b_vvvv(a, d, e, f) * t2b(e, c, i, l) * t2b(b, f, j, k)&  ! (1)
                                                                            + h2b_vvvv(a, d, e, f) * t2b(e, c, j, l) * t2b(b, f, i, k)&  ! (ij)
                                                                            + h2b_vvvv(a, d, e, f) * t2b(e, c, i, k) * t2b(b, f, j, l)&  ! (kl)
                                                                            - h2b_vvvv(a, d, e, f) * t2b(e, c, j, k) * t2b(b, f, i, l)   ! (ij)(kl)
                                                                ! (ab)(cd)
                                                                mm24 = mm24 + h2b_vvvv(b, d, e, f) * t2b(e, c, i, l) * t2b(a, f, j, k)&  ! (1)
                                                                            - h2b_vvvv(b, d, e, f) * t2b(e, c, j, l) * t2b(a, f, i, k)&  ! (ij)
                                                                            - h2b_vvvv(b, d, e, f) * t2b(e, c, i, k) * t2b(a, f, j, l)&  ! (kl)
                                                                            + h2b_vvvv(b, d, e, f) * t2b(e, c, j, k) * t2b(a, f, i, l)   ! (ij)(kl)
                                                            end do
                                                        end do
                                                        ! Diagram 17:  A(ij)A(cd) h2c(mnkl) * t2b(adin) * t2b(bcjm)
                                                        do m = 1, nob
                                                            do n = 1, nob
                                                                ! (1)
                                                                mm24 = mm24 + h2c_oooo(m, n, k, l) * t2b(a, d, i, n) * t2b(b, c, j, m)&  ! (1)
                                                                            - h2c_oooo(m, n, k, l) * t2b(a, d, j, n) * t2b(b, c, i, m)   ! (ij)
                                                                ! (cd)
                                                                mm24 = mm24 - h2c_oooo(m, n, k, l) * t2b(a, c, i, n) * t2b(b, d, j, m)&  ! (1)
                                                                            + h2c_oooo(m, n, k, l) * t2b(a, c, j, n) * t2b(b, d, i, m)   ! (ij)
                                                            end do
                                                        end do
                                                        ! Diagram 18:  A(ij)A(kl) h2c(cdef) * t2b(afil) * t2b(bejk)
                                                        do e = 1, nub
                                                            do f = 1, nub
                                                                ! (1)
                                                                mm24 = mm24 + h2c_vvvv(c, d, e, f) * t2b(a, f, i, l) * t2b(b, e, j, k)&  ! (1)
                                                                            - h2c_vvvv(c, d, e, f) * t2b(a, f, j, l) * t2b(b, e, i, k)   ! (ij)
                                                                ! (kl)
                                                                mm24 = mm24 - h2c_vvvv(c, d, e, f) * t2b(a, f, i, k) * t2b(b, e, j, l)&  ! (1)
                                                                            + h2c_vvvv(c, d, e, f) * t2b(a, f, j, k) * t2b(b, e, i, l)   ! (ij)
                                                            end do
                                                        end do

                                                        !!! L4C Computation !!!
                                                        ! Diagram 1: h2a(ijab) * l2c(cdkl)
                                                        l4 = l4 + h2a_oovv(i, j, a, b) * l2c(c, d, k, l)
                                                        ! Diagram 2: h2c(klcd) * l2a(abij)
                                                        l4 = l4 + h2c_oovv(k, l, c, d) * l2a(a, b, i, j)
                                                        ! Diagram 3: A(ij)A(kl)A(ab)A(cd) h2b(ikac) * l2b(bdjl)
                                                        ! (1)
                                                        l4 = l4 + h2b_oovv(i, k, a, c) * l2b(b, d, j, l)&  ! (1)
                                                                - h2b_oovv(j, k, a, c) * l2b(b, d, i, l)&  ! (ij)
                                                                - h2b_oovv(i, l, a, c) * l2b(b, d, j, k)&  ! (kl)
                                                                + h2b_oovv(j, l, a, c) * l2b(b, d, i, k)   ! (ij)(kl)
                                                        ! (ab)
                                                        l4 = l4 - h2b_oovv(i, k, b, c) * l2b(a, d, j, l)&  ! (1)
                                                                + h2b_oovv(j, k, b, c) * l2b(a, d, i, l)&  ! (ij)
                                                                + h2b_oovv(i, l, b, c) * l2b(a, d, j, k)&  ! (kl)
                                                                - h2b_oovv(j, l, b, c) * l2b(a, d, i, k)   ! (ij)(kl)
                                                        ! (cd)
                                                        l4 = l4 - h2b_oovv(i, k, a, d) * l2b(b, c, j, l)&  ! (1)
                                                                + h2b_oovv(j, k, a, d) * l2b(b, c, i, l)&  ! (ij)
                                                                + h2b_oovv(i, l, a, d) * l2b(b, c, j, k)&  ! (kl)
                                                                - h2b_oovv(j, l, a, d) * l2b(b, c, i, k)   ! (ij)(kl)
                                                        ! (ab)(cd)
                                                        l4 = l4 + h2b_oovv(i, k, b, d) * l2b(a, c, j, l)&  ! (1)
                                                                - h2b_oovv(j, k, b, d) * l2b(a, c, i, l)&  ! (ij)
                                                                - h2b_oovv(i, l, b, d) * l2b(a, c, j, k)&  ! (kl)
                                                                + h2b_oovv(j, l, b, d) * l2b(a, c, i, k)   ! (ij)(kl)

                                                        temp = mm24 * l4

                                                        denom = fA_oo(i,i) + fA_oo(j,j) + fB_oo(k,k) + fB_oo(l,l)&
                                                        - fA_vv(a,a) - fA_vv(b,b) - fB_vv(c,c) - fB_vv(d,d)

                                                        deltaA = deltaA + temp/denom

                                                        denom = H1A_oo(i,i) + H1A_oo(j,j) + H1B_oo(k,k) + H1B_oo(l,l)&
                                                        - H1A_vv(a,a) - H1A_vv(b,b) - H1B_vv(c,c) - H1B_vv(d,d)

                                                        deltaB = deltaB + temp/denom

                                                        denom = denom &
                                                        -H2A_oooo(i,j,i,j)-H2A_voov(a,i,i,a)-H2A_voov(a,j,j,a)-H2A_voov(b,i,i,b)&
                                                        -H2A_voov(b,j,j,b)-H2A_vvvv(a,b,a,b)-H2B_oooo(i,k,i,k)-H2B_oooo(j,k,j,k)&
                                                        -H2B_oooo(i,l,i,l)-H2B_oooo(j,l,j,l)+H2B_ovov(i,c,i,c)+H2B_ovov(j,c,j,c)&
                                                        +H2B_ovov(i,d,i,d)+H2B_ovov(j,d,j,d)+H2B_vovo(a,k,a,k)+H2B_vovo(a,l,a,l)&
                                                        +H2B_vovo(b,k,b,k)+H2B_vovo(b,l,b,l)-H2B_vvvv(a,c,a,c)-H2B_vvvv(a,d,a,d)&
                                                        -H2B_vvvv(b,c,b,c)-H2B_vvvv(b,d,b,d)-H2C_oooo(l,k,l,k)-H2C_voov(c,k,k,c)&
                                                        -H2C_voov(c,l,l,c)-H2C_voov(d,k,k,d)-H2C_voov(d,l,l,d)-H2C_vvvv(c,d,c,d)

                                                        deltaC = deltaC + temp/denom

                                                        denom = denom &
                                                        +D3A_O(a,i,j)+D3A_O(b,i,j)-D3A_V(a,i,b)-D3A_V(a,j,b)&
                                                        +D3B_O(a,i,k)+D3B_O(a,i,l)+D3B_O(a,j,k)+D3B_O(a,j,l)&
                                                        +D3B_O(b,i,k)+D3B_O(b,i,l)+D3B_O(b,j,k)+D3B_O(b,j,l)&
                                                        -D3B_V(a,i,c)-D3B_V(a,j,c)-D3B_V(b,i,c)-D3B_V(b,j,c)&
                                                        -D3B_V(a,i,d)-D3B_V(a,j,d)-D3B_V(b,i,d)-D3B_V(b,j,d)&
                                                        +D3C_O(c,i,k)+D3C_O(c,i,l)+D3C_O(c,j,k)+D3C_O(c,j,l)&
                                                        +D3C_O(d,i,k)+D3C_O(d,i,l)+D3C_O(d,j,k)+D3C_O(d,j,l)&
                                                        -D3C_V(a,k,c)-D3C_V(a,l,c)-D3C_V(b,k,c)-D3C_V(b,l,c)&
                                                        -D3C_V(a,k,d)-D3C_V(a,l,d)-D3C_V(b,k,d)-D3C_V(b,l,d)&
                                                        +D3D_O(c,k,l)+D3D_O(d,k,l)-D3D_V(c,k,d)-D3D_V(c,l,d)

                                                        deltaD = deltaD + temp/denom

                                                    end do
                                                end do
                                            end do
                                        end do
                                    end do
                                end do
                            end do
                        end do

                        !print*, "Error = ", error

              end subroutine crcc24C

end module crcc24_loops