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
                        integer :: i, j, k, l, a, b, c, d, m, n, e, f
                        real(kind=8) :: denom, temp, mm24, l4

                        deltaA = 0.0d0
                        deltaB = 0.0d0
                        deltaC = 0.0d0
                        deltaD = 0.0d0

                        do i = 1 , noa
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
                                                                mm24 = -h2a_voov(a, m, i, e) * t2a(b, c, m, k) * t2a(e, d, j, l)&
                                                                       +h2a_voov(a, m, k, e) * t2a(b, c, m, i) * t2a(e, d, j, l) ! (ik)
                                                            end do
                                                        end do
                                                        ! Diagram 2: A(ij/kl)A(bc/ad) h2a(mnij) * t2a(adml) * t2a(bcnk)

                                                        ! Diagram 3: A(jk/il)A(ab/cd) h2a(abef) * t2a(fcjk) * t2a(edil)

                                                        !!! L4A Computation !!!
                                                        ! Diagram 1: A(ij/kl)A(ab/cd) h2a(klcd) * l2a(abij)


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

              end subroutine crcc24A

              subroutine crcc24B(deltaA,deltaB,deltaC,deltaD,&
                              t2a,t2b,l2a,l2b,&
                              fA_oo,fA_vv,fB_oo,fB_vv,&
                              H1A_oo,H1A_vv,H1B_oo,H1B_vv,&
                              H2A_voov, H2A_oooo, H2A_vvvv,H2A_oovv,&
                              H2B_ovov,H2B_vovo,H2B_oooo,H2B_vvvv,H2B_oovv,&
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
                        H2B_ovov(1:noa,1:nub,1:noa,1:nub),&
                        H2B_vovo(1:nua,1:nob,1:nua,1:nob),&
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
                        integer :: i, j, k, l, a, b, c, d, m, n, e, f
                        real(kind=8) :: denom, temp, mm24, l4

                        deltaA = 0.0d0
                        deltaB = 0.0d0
                        deltaC = 0.0d0
                        deltaD = 0.0d0

                        do i = 1 , noa
                            do j = i+1, noa
                                do k = j+1, noa
                                    do l = 1, nob
                                        do a = 1, nua
                                            do b = a+1, nua
                                                do c = b+1, nua
                                                    do d = 1, nub
                                                    !!! MM(2,4)B Computation !!!
                                                    !    dT.aaab = -(9.0 / 36.0) * np.einsum("mdel,abim,ecjk->abcdijkl", H.ab.ovvo, T.aa, T.aa, optimize=True)    # (i/jk)(c/ab) = 9
                                                    !    dT.aaab += (9.0 / 36.0) * np.einsum("mnij,bcnk,adml->abcdijkl", H.aa.oooo, T.aa, T.ab, optimize=True)    # (k/ij)(a/bc) = 9
                                                    !    dT.aaab -= (18.0 / 36.0) * np.einsum("mdjf,abim,cfkl->abcdijkl", H.ab.ovov, T.aa, T.ab, optimize=True)   # (ijk)(c/ab) = (i/jk)(c/ab)(jk) = 18
                                                    !    dT.aaab -= np.einsum("amie,bejl,cdkm->abcdijkl", H.ab.voov, T.ab, T.ab, optimize=True)                   # (ijk)(abc) = (i/jk)(a/bc)(jk)(bc) = 36
                                                    !    dT.aaab += (18.0 / 36.0) * np.einsum("mnjl,bcmk,adin->abcdijkl", H.ab.oooo, T.aa, T.ab, optimize=True)   # (ijk)(a/bc) = (i/jk)(a/bc)(jk) = 18
                                                    !    dT.aaab -= (18.0 / 36.0) * np.einsum("bmel,ecjk,adim->abcdijkl", H.ab.vovo, T.aa, T.ab, optimize=True)   # (i/jk)(abc) = (i/jk)(a/bc)(bc) = 18
                                                    !    dT.aaab -= (18.0 / 36.0) * np.einsum("amie,ecjk,bdml->abcdijkl", H.aa.voov, T.aa, T.ab, optimize=True)   # (i/kj)(abc) = (i/kj)(a/bc)(bc) = 18
                                                    !    dT.aaab += (9.0 / 36.0) * np.einsum("abef,fcjk,edil->abcdijkl", H.aa.vvvv, T.aa, T.ab, optimize=True)    # (i/jk)(c/ab) = (i/jk)(c/ab) = 9
                                                    !    dT.aaab -= (18.0 / 36.0) * np.einsum("amie,bcmk,edjl->abcdijkl", H.aa.voov, T.aa, T.ab, optimize=True)   # (ijk)(a/bc) = (i/jk)(a/bc)(jk) = 18
                                                    !    dT.aaab += (18.0 / 36.0) * np.einsum("adef,ebij,cfkl->abcdijkl", H.ab.vvvv, T.aa, T.ab, optimize=True)   # (k/ij)(abc) = (k/ij)(a/bc)(bc) = 18

                                                        !!! L4B Computation !!!
                                                        ! Diagram 1: A(k/ij)A(c/ab) h2a(ijab) * l2b(cdkl)
                                                        ! Diagram 2: A(k/ij)A(c/ab) h2b(klcd) * l2a(abij)

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

              end subroutine crcc24B

              subroutine crcc24C(deltaA,deltaB,deltaC,deltaD,&
                              t2a,t2b,t2c,l2a,l2b,l2c,&
                              fA_oo,fA_vv,fB_oo,fB_vv,&
                              H1A_oo,H1A_vv,H1B_oo,H1B_vv,&
                              H2A_voov,H2A_oooo,H2A_vvvv,H2A_oovv,&
                              H2B_ovov,H2B_vovo,H2B_oooo,H2B_vvvv,H2B_oovv,&
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
                        H2B_ovov(1:noa,1:nub,1:noa,1:nub),&
                        H2B_vovo(1:nua,1:nob,1:nua,1:nob),&
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
                        integer :: i, j, k, l, a, b, c, d, m, n, e, f
                        real(kind=8) :: denom, temp, mm24, l4

                        deltaA = 0.0d0
                        deltaB = 0.0d0
                        deltaC = 0.0d0
                        deltaD = 0.0d0

                        do i = 1 , noa
                            do j = i+1, noa
                                do k = 1, nob
                                    do l = k+1, nob
                                        do a = 1, nua
                                            do b = a+1, nua
                                                do c = 1, nub
                                                    do d = c+1, nub

                                                        !!! MM(2,4)C Computation !!!
                                                        !    dT.aabb = -np.einsum("cmke,adim,bejl->abcdijkl", H.bb.voov, T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
                                                        !    dT.aabb -= np.einsum("amie,bcmk,edjl->abcdijkl", H.aa.voov, T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
                                                        !    dT.aabb -= 0.5 * np.einsum("mcek,aeij,bdml->abcdijkl", H.ab.ovvo, T.aa, T.ab, optimize=True)    # (kl)(ab)(cd) = 8
                                                        !    dT.aabb -= 0.5 * np.einsum("amie,bdjm,cekl->abcdijkl", H.ab.voov, T.ab, T.bb, optimize=True)    # (ij)(ab)(cd) = 8
                                                        !    dT.aabb -= 0.5 * np.einsum("mcek,abim,edjl->abcdijkl", H.ab.ovvo, T.aa, T.ab, optimize=True)    # (ij)(kl)(cd) = 8
                                                        !    dT.aabb -= 0.5 * np.einsum("amie,cdkm,bejl->abcdijkl", H.ab.voov, T.bb, T.ab, optimize=True)    # (ij)(kl)(ab) = 8
                                                        !    dT.aabb -= np.einsum("bmel,adim,ecjk->abcdijkl", H.ab.vovo, T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
                                                        !    dT.aabb -= np.einsum("mdje,bcmk,aeil->abcdijkl", H.ab.ovov, T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
                                                        !    dT.aabb -= 0.25 * np.einsum("mdje,abim,cekl->abcdijkl", H.ab.ovov, T.aa, T.bb, optimize=True)   # (ij)(cd) = 4
                                                        !    dT.aabb -= 0.25 * np.einsum("bmel,cdkm,aeij->abcdijkl", H.ab.vovo, T.bb, T.aa, optimize=True)   # (kl)(ab) = 4
                                                        !    dT.aabb += 0.25 * np.einsum("mnij,acmk,bdnl->abcdijkl", H.aa.oooo, T.ab, T.ab, optimize=True)   # (kl)(ab) = 4 !!! (tricky asym)
                                                        !    dT.aabb += 0.25 * np.einsum("abef,ecik,fdjl->abcdijkl", H.aa.vvvv, T.ab, T.ab, optimize=True)   # (ij)(kl) = 4 !!! (tricky asym)
                                                        !    dT.aabb += 0.25 * np.einsum("mnik,abmj,cdnl->abcdijkl", H.ab.oooo, T.aa, T.bb, optimize=True)   # (ij)(kl) = 4
                                                        !    dT.aabb += 0.25 * np.einsum("acef,ebij,fdkl->abcdijkl", H.ab.vvvv, T.aa, T.bb, optimize=True)   # (ab)(cd) = 4
                                                        !    dT.aabb += np.einsum("mnik,adml,bcjn->abcdijkl", H.ab.oooo, T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
                                                        !    dT.aabb += np.einsum("acef,edil,bfjk->abcdijkl", H.ab.vvvv, T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
                                                        !    dT.aabb += 0.25 * np.einsum("mnkl,adin,bcjm->abcdijkl", H.bb.oooo, T.ab, T.ab, optimize=True)   # (ij)(cd) = 4 !!! (tricky asym)
                                                        !    dT.aabb += 0.25 * np.einsum("cdef,afil,bejk->abcdijkl", H.bb.vvvv, T.ab, T.ab, optimize=True)   # (ij)(kl) = 4 !!! (tricky asym)

                                                        !!! L4C Computation !!!
                                                        ! Diagram 1: h2a(ijab) * l2c(cdkl)
                                                        ! Diagram 2: h2c(klcd) * l2a(abij)
                                                        ! Diagram 3: A(ij)A(kl)A(ab)A(cd) h2b(ikac) * l2b(jlbd)

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

              end subroutine crcc24C

end module crcc24_loops