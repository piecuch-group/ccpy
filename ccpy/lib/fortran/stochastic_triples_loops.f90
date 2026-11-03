module stochastic_triples_loops

  ! Virtual-triplet-driven CR-CC(2,3)-type triples corrections.
  !
  ! For each requested virtual triplet (a,b,c), these routines compute the contribution
  ! E(abc) = sum_{ijk in Q} M(abcijk) * L(ijkabc) / D(abcijk) to the A, B, C, and D variants
  ! of the CR-CC(2,3) / CC(t;3) / CC(P;3)_2BA corrections, i.e., the transpose of the usual
  ! (i,j,k)-driven loops in crcc_loops. This is the elementary task of the semi-stochastic
  ! algorithm of Damour, Gallo, and Scemama [arXiv:2405.16962], where virtual triplets are
  ! sampled and each E(abc) is computed once and memoized.
  !
  ! All two-body arrays are passed pre-arranged (see ccpy/moments/stochastic_triples.py) so that
  ! fixing the virtual indices leaves a contiguous (contraction index x occupied indices) matrix.
  ! For a fixed triplet, the moment and left vector are assembled from occupied-index arrays
  ! antisymmetrized in the virtual indices by the DGEMMs, and the occupied antisymmetrizer is
  ! applied element-wise in the final loop.
  !
  ! The Q space is the complement of the P space, given per triplet as the occupied triples
  ! pexc(:, pstart(t):pend(t)) of the P-space excitations with virtual part (a,b,c), and of the
  ! active-space triples with at least act(1) active particles and holes (CC(t;3); act(1) = 0
  ! disables it). act = (num_active, nacto_a, nactu_a, nacto_b, nactu_b).

      implicit none

      contains

              subroutine crcc23a_abc(e_abc,&
                              trips,pstart,pend,pexc,act,&
                              H2A_vooo_mijx,t2a_mkyz,I2A_vvov_eixy,t2a_ejkz,&
                              H2A_vovv_eixy,l2a_ejkz,H2A_ooov_mijx,l2a_mkyz,&
                              l1a,l2a,vA_oovv,H1A_ov,&
                              fA_o,fA_v,H1A_o,H1A_v,&
                              dvoA,dooA,H2A_vvvv,&
                              D3A_O,D3A_V,&
                              ntrip,npexc,noa,nua)

                        integer, intent(in) :: ntrip, npexc, noa, nua
                        real(kind=8), intent(out) :: e_abc(4,ntrip)
                        integer, intent(in) :: trips(3,ntrip), pstart(ntrip), pend(ntrip), pexc(3,npexc), act(5)
                        real(kind=8), intent(in) :: H2A_vooo_mijx(noa,noa,noa,nua),t2a_mkyz(noa,noa,nua,nua),&
                        I2A_vvov_eixy(nua,noa,nua,nua),t2a_ejkz(nua,noa,noa,nua),&
                        H2A_vovv_eixy(nua,noa,nua,nua),l2a_ejkz(nua,noa,noa,nua),&
                        H2A_ooov_mijx(noa,noa,noa,nua),l2a_mkyz(noa,noa,nua,nua),&
                        l1a(nua,noa),l2a(nua,nua,noa,noa),vA_oovv(noa,noa,nua,nua),H1A_ov(noa,nua),&
                        fA_o(noa),fA_v(nua),H1A_o(noa),H1A_v(nua),&
                        dvoA(nua,noa),dooA(noa,noa),H2A_vvvv(nua,nua),&
                        D3A_O(nua,noa,noa),D3A_V(nua,noa,nua)

                        integer :: it, i, j, k, a, b, c, idet, noa2, npart, nhole
                        real(kind=8) :: D, temp1, temp2, temp3, LM, dA, dB, dC, dD, Dv, Dvb, Dvc
                        real(kind=8), allocatable :: W1(:,:,:), W2(:,:,:), LW1(:,:,:), LW2(:,:,:)
                        logical(kind=1), allocatable :: qspace(:,:,:)

                        noa2 = noa*noa

                        !$omp parallel default(shared) private(it,i,j,k,a,b,c,idet,npart,nhole,D,temp1,temp2,temp3,LM,&
                        !$omp dA,dB,dC,dD,Dv,Dvb,Dvc,W1,W2,LW1,LW2,qspace)
                        allocate(W1(noa,noa,noa), W2(noa,noa,noa), LW1(noa,noa,noa), LW2(noa,noa,noa), qspace(noa,noa,noa))
                        !$omp do schedule(dynamic)
                        do it = 1, ntrip
                           a = trips(1,it); b = trips(2,it); c = trips(3,it);
                           ! Q space for triplet (a,b,c)
                           qspace = .true.
                           do idet = pstart(it), pend(it)
                              qspace(pexc(1,idet),pexc(2,idet),pexc(3,idet)) = .false.
                           end do
                           npart = 0
                           if (act(1) > 0) npart = count((/a,b,c/) <= act(3))

                           W1 = 0.0d0; W2 = 0.0d0; LW1 = 0.0d0; LW2 = 0.0d0;
                           !!!!! MM(2,3)A !!!!!
                           ! Diagram 1: -A(k/ij)A(a/bc) H2A_vooo(a,m,i,j)*t2a(b,c,m,k)
                           call dgemm('t','n',noa2,noa,noa,1.0d0,H2A_vooo_mijx(:,:,:,a),noa,t2a_mkyz(:,:,b,c),noa,1.0d0,W1,noa2)
                           call dgemm('t','n',noa2,noa,noa,-1.0d0,H2A_vooo_mijx(:,:,:,b),noa,t2a_mkyz(:,:,a,c),noa,1.0d0,W1,noa2)
                           call dgemm('t','n',noa2,noa,noa,-1.0d0,H2A_vooo_mijx(:,:,:,c),noa,t2a_mkyz(:,:,b,a),noa,1.0d0,W1,noa2)
                           ! Diagram 2: A(i/jk)A(c/ab) I2A_vvov(a,b,i,e)*t2a(e,c,j,k)
                           call dgemm('t','n',noa,noa2,nua,1.0d0,I2A_vvov_eixy(:,:,a,b),nua,t2a_ejkz(:,:,:,c),nua,1.0d0,W2,noa)
                           call dgemm('t','n',noa,noa2,nua,-1.0d0,I2A_vvov_eixy(:,:,c,b),nua,t2a_ejkz(:,:,:,a),nua,1.0d0,W2,noa)
                           call dgemm('t','n',noa,noa2,nua,-1.0d0,I2A_vvov_eixy(:,:,a,c),nua,t2a_ejkz(:,:,:,b),nua,1.0d0,W2,noa)
                           !!!!! L3A !!!!!
                           ! Diagram 1: A(i/jk)A(c/ab) H2A_vovv(e,i,b,a)*l2a(e,c,j,k)
                           call dgemm('t','n',noa,noa2,nua,1.0d0,H2A_vovv_eixy(:,:,a,b),nua,l2a_ejkz(:,:,:,c),nua,1.0d0,LW2,noa)
                           call dgemm('t','n',noa,noa2,nua,-1.0d0,H2A_vovv_eixy(:,:,c,b),nua,l2a_ejkz(:,:,:,a),nua,1.0d0,LW2,noa)
                           call dgemm('t','n',noa,noa2,nua,-1.0d0,H2A_vovv_eixy(:,:,a,c),nua,l2a_ejkz(:,:,:,b),nua,1.0d0,LW2,noa)
                           ! Diagram 2: -A(k/ij)A(a/bc) H2A_ooov(j,i,m,a)*l2a(b,c,m,k)
                           call dgemm('t','n',noa2,noa,noa,1.0d0,H2A_ooov_mijx(:,:,:,a),noa,l2a_mkyz(:,:,b,c),noa,1.0d0,LW1,noa2)
                           call dgemm('t','n',noa2,noa,noa,-1.0d0,H2A_ooov_mijx(:,:,:,b),noa,l2a_mkyz(:,:,a,c),noa,1.0d0,LW1,noa2)
                           call dgemm('t','n',noa2,noa,noa,-1.0d0,H2A_ooov_mijx(:,:,:,c),noa,l2a_mkyz(:,:,b,a),noa,1.0d0,LW1,noa2)

                           Dv = fA_v(a) + fA_v(b) + fA_v(c)
                           Dvb = H1A_v(a) + H1A_v(b) + H1A_v(c)
                           Dvc = H2A_vvvv(b,a) + H2A_vvvv(c,a) + H2A_vvvv(c,b)
                           dA = 0.0d0; dB = 0.0d0; dC = 0.0d0; dD = 0.0d0;
                           do i = 1, noa
                              do j = i+1, noa
                                 do k = j+1, noa
                                    if (.not. qspace(i,j,k)) cycle
                                    if (npart >= act(1) .and. act(1) > 0) then
                                       nhole = count((/i,j,k/) > noa - act(2))
                                       if (nhole >= act(1)) cycle
                                    end if

                                    temp1 = -W1(i,j,k) + W1(k,j,i) + W1(i,k,j)&
                                            +W2(i,j,k) - W2(j,i,k) - W2(k,j,i)

                                    temp2 = -LW1(i,j,k) + LW1(k,j,i) + LW1(i,k,j)&
                                            +LW2(i,j,k) - LW2(j,i,k) - LW2(k,j,i)

                                    temp3 =&
                                    l1a(c,k)*vA_oovv(i,j,a,b)&
                                    -l1a(a,k)*vA_oovv(i,j,c,b)&
                                    -l1a(b,k)*vA_oovv(i,j,a,c)&
                                    -l1a(c,i)*vA_oovv(k,j,a,b)&
                                    -l1a(c,j)*vA_oovv(i,k,a,b)&
                                    +l1a(a,i)*vA_oovv(k,j,c,b)&
                                    +l1a(b,i)*vA_oovv(k,j,a,c)&
                                    +l1a(a,j)*vA_oovv(i,k,c,b)&
                                    +l1a(b,j)*vA_oovv(i,k,a,c)&
                                    +H1A_ov(k,c)*l2a(a,b,i,j)&
                                    -H1A_ov(k,a)*l2a(c,b,i,j)&
                                    -H1A_ov(k,b)*l2a(a,c,i,j)&
                                    -H1A_ov(i,c)*l2a(a,b,k,j)&
                                    -H1A_ov(j,c)*l2a(a,b,i,k)&
                                    +H1A_ov(i,a)*l2a(c,b,k,j)&
                                    +H1A_ov(i,b)*l2a(a,c,k,j)&
                                    +H1A_ov(j,a)*l2a(c,b,i,k)&
                                    +H1A_ov(j,b)*l2a(a,c,i,k)

                                    LM = temp1*(temp2+temp3)

                                    D = fA_o(i) + fA_o(j) + fA_o(k) - Dv
                                    dA = dA + LM/D

                                    D = H1A_o(i) + H1A_o(j) + H1A_o(k) - Dvb
                                    dB = dB + LM/D

                                    D = D &
                                    -dvoA(a,i) - dvoA(b,i) - dvoA(c,i)&
                                    -dvoA(a,j) - dvoA(b,j) - dvoA(c,j)&
                                    -dvoA(a,k) - dvoA(b,k) - dvoA(c,k)&
                                    -dooA(i,j) - dooA(i,k) - dooA(j,k)&
                                    -Dvc
                                    dC = dC + LM/D

                                    D = D &
                                    +D3A_O(a,i,j)+D3A_O(a,i,k)+D3A_O(a,j,k)&
                                    +D3A_O(b,i,j)+D3A_O(b,i,k)+D3A_O(b,j,k)&
                                    +D3A_O(c,i,j)+D3A_O(c,i,k)+D3A_O(c,j,k)&
                                    -D3A_V(a,i,b)-D3A_V(a,i,c)-D3A_V(b,i,c)&
                                    -D3A_V(a,j,b)-D3A_V(a,j,c)-D3A_V(b,j,c)&
                                    -D3A_V(a,k,b)-D3A_V(a,k,c)-D3A_V(b,k,c)
                                    dD = dD + LM/D
                                 end do
                              end do
                           end do
                           e_abc(:,it) = (/dA, dB, dC, dD/)
                        end do
                        !$omp end do
                        deallocate(W1, W2, LW1, LW2, qspace)
                        !$omp end parallel

              end subroutine crcc23a_abc

              subroutine crcc23b_abc(e_abc,&
                              trips,pstart,pend,pexc,act,&
                              t2a_eijx,H2B_vvvo_ekyc,I2A_vooo_mijx,t2b_mkyc,&
                              t2a_mixy,I2B_ovoo_mjkc,H2A_vvov_eixy,t2b_ejkc,&
                              t2b_eikx,H2B_vvov_ejyc,I2B_vooo_mikx,t2b_mjyc,&
                              l2a_eijx,H2B_vovv,H2A_ooov_mijx,l2b_mkyc,&
                              H2A_vovv_eixy,l2b_ejkc,l2a_mixy,H2B_ooov_mjkc,&
                              l2b_eikx,H2B_ovvv_ejyc,H2B_oovo_mikx,l2b_mjyc,&
                              l1a,l1b,l2a,l2b,vA_oovv,vB_oovv,H1A_ov,H1B_ov,&
                              fA_o,fA_v,fB_o,fB_v,H1A_o,H1A_v,H1B_o,H1B_v,&
                              dvoA,dooA,dovB,dvoB,dooB,dvoC,H2A_vvvv,H2B_vvvv,&
                              D3A_O,D3A_V,D3B_O,D3B_V,D3C_O,D3C_V,&
                              ntrip,npexc,noa,nua,nob,nub)

                        integer, intent(in) :: ntrip, npexc, noa, nua, nob, nub
                        real(kind=8), intent(out) :: e_abc(4,ntrip)
                        integer, intent(in) :: trips(3,ntrip), pstart(ntrip), pend(ntrip), pexc(3,npexc), act(5)
                        real(kind=8), intent(in) :: t2a_eijx(nua,noa,noa,nua),H2B_vvvo_ekyc(nua,nob,nua,nub),&
                        I2A_vooo_mijx(noa,noa,noa,nua),t2b_mkyc(noa,nob,nua,nub),&
                        t2a_mixy(noa,noa,nua,nua),I2B_ovoo_mjkc(noa,noa,nob,nub),&
                        H2A_vvov_eixy(nua,noa,nua,nua),t2b_ejkc(nua,noa,nob,nub),&
                        t2b_eikx(nub,noa,nob,nua),H2B_vvov_ejyc(nub,noa,nua,nub),&
                        I2B_vooo_mikx(nob,noa,nob,nua),t2b_mjyc(nob,noa,nua,nub),&
                        l2a_eijx(nua,noa,noa,nua),H2B_vovv(nua,nob,nua,nub),&
                        H2A_ooov_mijx(noa,noa,noa,nua),l2b_mkyc(noa,nob,nua,nub),&
                        H2A_vovv_eixy(nua,noa,nua,nua),l2b_ejkc(nua,noa,nob,nub),&
                        l2a_mixy(noa,noa,nua,nua),H2B_ooov_mjkc(noa,noa,nob,nub),&
                        l2b_eikx(nub,noa,nob,nua),H2B_ovvv_ejyc(nub,noa,nua,nub),&
                        H2B_oovo_mikx(nob,noa,nob,nua),l2b_mjyc(nob,noa,nua,nub),&
                        l1a(nua,noa),l1b(nub,nob),l2a(nua,nua,noa,noa),l2b(nua,nub,noa,nob),&
                        vA_oovv(noa,noa,nua,nua),vB_oovv(noa,nob,nua,nub),H1A_ov(noa,nua),H1B_ov(nob,nub),&
                        fA_o(noa),fA_v(nua),fB_o(nob),fB_v(nub),H1A_o(noa),H1A_v(nua),H1B_o(nob),H1B_v(nub),&
                        dvoA(nua,noa),dooA(noa,noa),dovB(noa,nub),dvoB(nua,nob),dooB(noa,nob),dvoC(nub,nob),&
                        H2A_vvvv(nua,nua),H2B_vvvv(nua,nub),&
                        D3A_O(nua,noa,noa),D3A_V(nua,noa,nua),D3B_O(nua,noa,nob),D3B_V(nua,noa,nub),&
                        D3C_O(nub,noa,nob),D3C_V(nua,nob,nub)

                        integer :: it, i, j, k, a, b, c, idet, noa2, noanob, npart, nhole
                        real(kind=8) :: D, temp1, temp2, temp3, LM, dA, dB, dC, dD, Dv, Dvb, Dvc
                        real(kind=8), allocatable :: Wij_k(:,:,:), Wi_jk(:,:,:), Wik_j(:,:,:),&
                                                     Lij_k(:,:,:), Li_jk(:,:,:), Lik_j(:,:,:)
                        logical(kind=1), allocatable :: qspace(:,:,:)

                        noa2 = noa*noa
                        noanob = noa*nob

                        !$omp parallel default(shared) private(it,i,j,k,a,b,c,idet,npart,nhole,D,temp1,temp2,temp3,LM,&
                        !$omp dA,dB,dC,dD,Dv,Dvb,Dvc,Wij_k,Wi_jk,Wik_j,Lij_k,Li_jk,Lik_j,qspace)
                        allocate(Wij_k(noa,noa,nob), Wi_jk(noa,noa,nob), Wik_j(noa,nob,noa),&
                                 Lij_k(noa,noa,nob), Li_jk(noa,noa,nob), Lik_j(noa,nob,noa), qspace(noa,noa,nob))
                        !$omp do schedule(dynamic)
                        do it = 1, ntrip
                           a = trips(1,it); b = trips(2,it); c = trips(3,it);
                           qspace = .true.
                           do idet = pstart(it), pend(it)
                              qspace(pexc(1,idet),pexc(2,idet),pexc(3,idet)) = .false.
                           end do
                           npart = 0
                           if (act(1) > 0) npart = count((/a,b/) <= act(3)) + count((/c/) <= act(5))

                           Wij_k = 0.0d0; Wi_jk = 0.0d0; Wik_j = 0.0d0;
                           Lij_k = 0.0d0; Li_jk = 0.0d0; Lik_j = 0.0d0;
                           !!!!! MM(2,3)B !!!!!
                           ! Diagram 1: A(ab) H2B(bcek)*t2a(aeij)
                           call dgemm('t','n',noa2,nob,nua,1.0d0,t2a_eijx(:,:,:,a),nua,H2B_vvvo_ekyc(:,:,b,c),nua,1.0d0,Wij_k,noa2)
                           call dgemm('t','n',noa2,nob,nua,-1.0d0,t2a_eijx(:,:,:,b),nua,H2B_vvvo_ekyc(:,:,a,c),nua,1.0d0,Wij_k,noa2)
                           ! Diagram 6: -A(ab) I2A(amij)*t2b(bcmk)
                           call dgemm('t','n',noa2,nob,noa,-1.0d0,I2A_vooo_mijx(:,:,:,a),noa,t2b_mkyc(:,:,b,c),noa,1.0d0,Wij_k,noa2)
                           call dgemm('t','n',noa2,nob,noa,1.0d0,I2A_vooo_mijx(:,:,:,b),noa,t2b_mkyc(:,:,a,c),noa,1.0d0,Wij_k,noa2)
                           ! Diagram 2: -A(ij) I2B(mcjk)*t2a(abim)
                           call dgemm('t','n',noa,noanob,noa,1.0d0,t2a_mixy(:,:,a,b),noa,I2B_ovoo_mjkc(:,:,:,c),noa,1.0d0,Wi_jk,noa)
                           ! Diagram 5: A(ij) H2A(abie)*t2b(ecjk)
                           call dgemm('t','n',noa,noanob,nua,1.0d0,H2A_vvov_eixy(:,:,a,b),nua,t2b_ejkc(:,:,:,c),nua,1.0d0,Wi_jk,noa)
                           ! Diagram 3: A(ab)A(ij) H2B(bcje)*t2b(aeik)
                           call dgemm('t','n',noanob,noa,nub,1.0d0,t2b_eikx(:,:,:,a),nub,H2B_vvov_ejyc(:,:,b,c),nub,1.0d0,Wik_j,noanob)
                           call dgemm('t','n',noanob,noa,nub,-1.0d0,t2b_eikx(:,:,:,b),nub,H2B_vvov_ejyc(:,:,a,c),nub,1.0d0,Wik_j,noanob)
                           ! Diagram 4: -A(ab)A(ij) I2B(amik)*t2b(bcjm)
                           call dgemm('t','n',noanob,noa,nob,-1.0d0,I2B_vooo_mikx(:,:,:,a),nob,t2b_mjyc(:,:,b,c),nob,1.0d0,Wik_j,noanob)
                           call dgemm('t','n',noanob,noa,nob,1.0d0,I2B_vooo_mikx(:,:,:,b),nob,t2b_mjyc(:,:,a,c),nob,1.0d0,Wik_j,noanob)
                           !!!!! L3B !!!!!
                           ! Diagram 1: A(ab) H2B(ekbc)*l2a(aeij)
                           call dgemm('t','n',noa2,nob,nua,1.0d0,l2a_eijx(:,:,:,a),nua,H2B_vovv(:,:,b,c),nua,1.0d0,Lij_k,noa2)
                           call dgemm('t','n',noa2,nob,nua,-1.0d0,l2a_eijx(:,:,:,b),nua,H2B_vovv(:,:,a,c),nua,1.0d0,Lij_k,noa2)
                           ! Diagram 5: -A(ab) H2A(jima)*l2b(bcmk)
                           call dgemm('t','n',noa2,nob,noa,-1.0d0,H2A_ooov_mijx(:,:,:,a),noa,l2b_mkyc(:,:,b,c),noa,1.0d0,Lij_k,noa2)
                           call dgemm('t','n',noa2,nob,noa,1.0d0,H2A_ooov_mijx(:,:,:,b),noa,l2b_mkyc(:,:,a,c),noa,1.0d0,Lij_k,noa2)
                           ! Diagram 2: A(ij) H2A(eiba)*l2b(ecjk)
                           call dgemm('t','n',noa,noanob,nua,1.0d0,H2A_vovv_eixy(:,:,a,b),nua,l2b_ejkc(:,:,:,c),nua,1.0d0,Li_jk,noa)
                           ! Diagram 4: -A(ij) H2B(jkmc)*l2a(abim)
                           call dgemm('t','n',noa,noanob,noa,1.0d0,l2a_mixy(:,:,a,b),noa,H2B_ooov_mjkc(:,:,:,c),noa,1.0d0,Li_jk,noa)
                           ! Diagram 3: A(ij)A(ab) H2B(jebc)*l2b(aeik)
                           call dgemm('t','n',noanob,noa,nub,1.0d0,l2b_eikx(:,:,:,a),nub,H2B_ovvv_ejyc(:,:,b,c),nub,1.0d0,Lik_j,noanob)
                           call dgemm('t','n',noanob,noa,nub,-1.0d0,l2b_eikx(:,:,:,b),nub,H2B_ovvv_ejyc(:,:,a,c),nub,1.0d0,Lik_j,noanob)
                           ! Diagram 6: -A(ij)A(ab) H2B(ikam)*l2b(bcjm)
                           call dgemm('t','n',noanob,noa,nob,-1.0d0,H2B_oovo_mikx(:,:,:,a),nob,l2b_mjyc(:,:,b,c),nob,1.0d0,Lik_j,noanob)
                           call dgemm('t','n',noanob,noa,nob,1.0d0,H2B_oovo_mikx(:,:,:,b),nob,l2b_mjyc(:,:,a,c),nob,1.0d0,Lik_j,noanob)

                           Dv = fA_v(a) + fA_v(b) + fB_v(c)
                           Dvb = H1A_v(a) + H1A_v(b) + H1B_v(c)
                           Dvc = H2A_vvvv(b,a) + H2B_vvvv(a,c) + H2B_vvvv(b,c)
                           dA = 0.0d0; dB = 0.0d0; dC = 0.0d0; dD = 0.0d0;
                           do i = 1, noa
                              do j = i+1, noa
                                 do k = 1, nob
                                    if (.not. qspace(i,j,k)) cycle
                                    if (npart >= act(1) .and. act(1) > 0) then
                                       nhole = count((/i,j/) > noa - act(2)) + count((/k/) > nob - act(4))
                                       if (nhole >= act(1)) cycle
                                    end if

                                    temp1 = Wij_k(i,j,k) + Wi_jk(i,j,k) - Wi_jk(j,i,k) + Wik_j(i,k,j) - Wik_j(j,k,i)
                                    temp2 = Lij_k(i,j,k) + Li_jk(i,j,k) - Li_jk(j,i,k) + Lik_j(i,k,j) - Lik_j(j,k,i)
                                    temp3 = l1a(a,i)*vB_oovv(j,k,b,c)&
                                    -l1a(a,j)*vB_oovv(i,k,b,c)&
                                    -l1a(b,i)*vB_oovv(j,k,a,c)&
                                    +l1a(b,j)*vB_oovv(i,k,a,c)&
                                    +l1b(c,k)*vA_oovv(i,j,a,b)&
                                    +l2b(b,c,j,k)*H1A_ov(i,a)&
                                    -l2b(b,c,i,k)*H1A_ov(j,a)&
                                    -l2b(a,c,j,k)*H1A_ov(i,b)&
                                    +l2b(a,c,i,k)*H1A_ov(j,b)&
                                    +l2a(a,b,i,j)*H1B_ov(k,c)

                                    LM = temp1*(temp2+temp3)

                                    D = fA_o(i) + fA_o(j) + fB_o(k) - Dv
                                    dA = dA + LM/D

                                    D = H1A_o(i) + H1A_o(j) + H1B_o(k) - Dvb
                                    dB = dB + LM/D

                                    D = D &
                                    -dvoA(a,i)-dvoA(b,i)+dovB(i,c)&
                                    -dvoA(a,j)-dvoA(b,j)+dovB(j,c)&
                                    +dvoB(a,k)+dvoB(b,k)-dvoC(c,k)&
                                    -dooA(i,j)-dooB(i,k)-dooB(j,k)&
                                    -Dvc
                                    dC = dC + LM/D

                                    D = D &
                                    +D3A_O(a,i,j)+D3B_O(a,i,k)+D3B_O(a,j,k)&
                                    +D3A_O(b,i,j)+D3B_O(b,i,k)+D3B_O(b,j,k)&
                                    +D3C_O(c,i,k)+D3C_O(c,j,k)&
                                    -D3A_V(a,i,b)-D3B_V(a,i,c)-D3B_V(b,i,c)&
                                    -D3A_V(a,j,b)-D3B_V(a,j,c)-D3B_V(b,j,c)&
                                    -D3C_V(a,k,c)-D3C_V(b,k,c)
                                    dD = dD + LM/D
                                 end do
                              end do
                           end do
                           e_abc(:,it) = (/dA, dB, dC, dD/)
                        end do
                        !$omp end do
                        deallocate(Wij_k, Wi_jk, Wik_j, Lij_k, Li_jk, Lik_j, qspace)
                        !$omp end parallel

              end subroutine crcc23b_abc

              subroutine crcc23c_abc(e_abc,&
                              trips,pstart,pend,pexc,act,&
                              H2B_vvov_eiay,t2c_ejkz,t2b_miay,I2C_vooo_mjkz,&
                              I2B_vooo_mijx,t2c_mkyz,t2b_eijx,H2C_vvov_ekzy,&
                              H2B_vvvo_ejay,t2b_eikz,t2b_mjay,I2B_ovoo_mikz,&
                              H2B_ovvv_eiay,l2c_ejkz,l2b_miay,H2C_ooov_mjkz,&
                              l2b_eijx,H2C_vovv,H2B_oovo_mijx,l2c_mkyz,&
                              H2B_vovv,l2b_eikz,l2b_mjay,H2B_ooov_mikz,&
                              l1a,l1b,l2b,l2c,vB_oovv,vC_oovv,H1A_ov,H1B_ov,&
                              fA_o,fA_v,fB_o,fB_v,H1A_o,H1A_v,H1B_o,H1B_v,&
                              dvoA,dovB,dvoB,dooB,dvoC,dooC,H2B_vvvv,H2C_vvvv,&
                              D3B_O,D3B_V,D3C_O,D3C_V,D3D_O,D3D_V,&
                              ntrip,npexc,noa,nua,nob,nub)

                        integer, intent(in) :: ntrip, npexc, noa, nua, nob, nub
                        real(kind=8), intent(out) :: e_abc(4,ntrip)
                        integer, intent(in) :: trips(3,ntrip), pstart(ntrip), pend(ntrip), pexc(3,npexc), act(5)
                        real(kind=8), intent(in) :: H2B_vvov_eiay(nub,noa,nua,nub),t2c_ejkz(nub,nob,nob,nub),&
                        t2b_miay(nob,noa,nua,nub),I2C_vooo_mjkz(nob,nob,nob,nub),&
                        I2B_vooo_mijx(nob,noa,nob,nua),t2c_mkyz(nob,nob,nub,nub),&
                        t2b_eijx(nub,noa,nob,nua),H2C_vvov_ekzy(nub,nob,nub,nub),&
                        H2B_vvvo_ejay(nua,nob,nua,nub),t2b_eikz(nua,noa,nob,nub),&
                        t2b_mjay(noa,nob,nua,nub),I2B_ovoo_mikz(noa,noa,nob,nub),&
                        H2B_ovvv_eiay(nub,noa,nua,nub),l2c_ejkz(nub,nob,nob,nub),&
                        l2b_miay(nob,noa,nua,nub),H2C_ooov_mjkz(nob,nob,nob,nub),&
                        l2b_eijx(nub,noa,nob,nua),H2C_vovv(nub,nob,nub,nub),&
                        H2B_oovo_mijx(nob,noa,nob,nua),l2c_mkyz(nob,nob,nub,nub),&
                        H2B_vovv(nua,nob,nua,nub),l2b_eikz(nua,noa,nob,nub),&
                        l2b_mjay(noa,nob,nua,nub),H2B_ooov_mikz(noa,noa,nob,nub),&
                        l1a(nua,noa),l1b(nub,nob),l2b(nua,nub,noa,nob),l2c(nub,nub,nob,nob),&
                        vB_oovv(noa,nob,nua,nub),vC_oovv(nob,nob,nub,nub),H1A_ov(noa,nua),H1B_ov(nob,nub),&
                        fA_o(noa),fA_v(nua),fB_o(nob),fB_v(nub),H1A_o(noa),H1A_v(nua),H1B_o(nob),H1B_v(nub),&
                        dvoA(nua,noa),dovB(noa,nub),dvoB(nua,nob),dooB(noa,nob),dvoC(nub,nob),dooC(nob,nob),&
                        H2B_vvvv(nua,nub),H2C_vvvv(nub,nub),&
                        D3B_O(nua,noa,nob),D3B_V(nua,noa,nub),D3C_O(nub,noa,nob),D3C_V(nua,nob,nub),&
                        D3D_O(nub,nob,nob),D3D_V(nub,nob,nub)

                        integer :: it, i, j, k, a, b, c, idet, nob2, noanob, npart, nhole
                        real(kind=8) :: D, temp1, temp2, temp3, LM, dA, dB, dC, dD, Dv, Dvb, Dvc
                        real(kind=8), allocatable :: Wi_jk(:,:,:), Wij_k(:,:,:), Wj_ik(:,:,:),&
                                                     Li_jk(:,:,:), Lij_k(:,:,:), Lj_ik(:,:,:)
                        logical(kind=1), allocatable :: qspace(:,:,:)

                        nob2 = nob*nob
                        noanob = noa*nob

                        !$omp parallel default(shared) private(it,i,j,k,a,b,c,idet,npart,nhole,D,temp1,temp2,temp3,LM,&
                        !$omp dA,dB,dC,dD,Dv,Dvb,Dvc,Wi_jk,Wij_k,Wj_ik,Li_jk,Lij_k,Lj_ik,qspace)
                        allocate(Wi_jk(noa,nob,nob), Wij_k(noa,nob,nob), Wj_ik(nob,noa,nob),&
                                 Li_jk(noa,nob,nob), Lij_k(noa,nob,nob), Lj_ik(nob,noa,nob), qspace(noa,nob,nob))
                        !$omp do schedule(dynamic)
                        do it = 1, ntrip
                           a = trips(1,it); b = trips(2,it); c = trips(3,it);
                           qspace = .true.
                           do idet = pstart(it), pend(it)
                              qspace(pexc(1,idet),pexc(2,idet),pexc(3,idet)) = .false.
                           end do
                           npart = 0
                           if (act(1) > 0) npart = count((/a/) <= act(3)) + count((/b,c/) <= act(5))

                           Wi_jk = 0.0d0; Wij_k = 0.0d0; Wj_ik = 0.0d0;
                           Li_jk = 0.0d0; Lij_k = 0.0d0; Lj_ik = 0.0d0;
                           !!!!! MM(2,3)C !!!!!
                           ! Diagram 1: A(bc) H2B_vvov(a,b,i,e)*t2c(e,c,j,k)
                           call dgemm('t','n',noa,nob2,nub,1.0d0,H2B_vvov_eiay(:,:,a,b),nub,t2c_ejkz(:,:,:,c),nub,1.0d0,Wi_jk,noa)
                           call dgemm('t','n',noa,nob2,nub,-1.0d0,H2B_vvov_eiay(:,:,a,c),nub,t2c_ejkz(:,:,:,b),nub,1.0d0,Wi_jk,noa)
                           ! Diagram 4: -A(bc) I2C_vooo(c,m,k,j)*t2b(a,b,i,m)
                           call dgemm('t','n',noa,nob2,nob,-1.0d0,t2b_miay(:,:,a,b),nob,I2C_vooo_mjkz(:,:,:,c),nob,1.0d0,Wi_jk,noa)
                           call dgemm('t','n',noa,nob2,nob,1.0d0,t2b_miay(:,:,a,c),nob,I2C_vooo_mjkz(:,:,:,b),nob,1.0d0,Wi_jk,noa)
                           ! Diagram 2: -A(jk) I2B_vooo(a,m,i,j)*t2c(b,c,m,k)
                           call dgemm('t','n',noanob,nob,nob,-1.0d0,I2B_vooo_mijx(:,:,:,a),nob,t2c_mkyz(:,:,b,c),nob,1.0d0,Wij_k,noanob)
                           ! Diagram 3: A(jk) H2C_vvov(c,b,k,e)*t2b(a,e,i,j)
                           call dgemm('t','n',noanob,nob,nub,1.0d0,t2b_eijx(:,:,:,a),nub,H2C_vvov_ekzy(:,:,c,b),nub,1.0d0,Wij_k,noanob)
                           ! Diagram 5: A(jk)A(bc) H2B_vvvo(a,b,e,j)*t2b(e,c,i,k)
                           call dgemm('t','n',nob,noanob,nua,1.0d0,H2B_vvvo_ejay(:,:,a,b),nua,t2b_eikz(:,:,:,c),nua,1.0d0,Wj_ik,nob)
                           call dgemm('t','n',nob,noanob,nua,-1.0d0,H2B_vvvo_ejay(:,:,a,c),nua,t2b_eikz(:,:,:,b),nua,1.0d0,Wj_ik,nob)
                           ! Diagram 6: -A(jk)A(bc) I2B_ovoo(m,c,i,k)*t2b(a,b,m,j)
                           call dgemm('t','n',nob,noanob,noa,-1.0d0,t2b_mjay(:,:,a,b),noa,I2B_ovoo_mikz(:,:,:,c),noa,1.0d0,Wj_ik,nob)
                           call dgemm('t','n',nob,noanob,noa,1.0d0,t2b_mjay(:,:,a,c),noa,I2B_ovoo_mikz(:,:,:,b),noa,1.0d0,Wj_ik,nob)
                           !!!!! L3C !!!!!
                           ! Diagram 1: A(bc) H2B_ovvv(i,e,a,b)*l2c(e,c,j,k)
                           call dgemm('t','n',noa,nob2,nub,1.0d0,H2B_ovvv_eiay(:,:,a,b),nub,l2c_ejkz(:,:,:,c),nub,1.0d0,Li_jk,noa)
                           call dgemm('t','n',noa,nob2,nub,-1.0d0,H2B_ovvv_eiay(:,:,a,c),nub,l2c_ejkz(:,:,:,b),nub,1.0d0,Li_jk,noa)
                           ! Diagram 5: -A(bc) H2C_ooov(j,k,m,c)*l2b(a,b,i,m)
                           call dgemm('t','n',noa,nob2,nob,-1.0d0,l2b_miay(:,:,a,b),nob,H2C_ooov_mjkz(:,:,:,c),nob,1.0d0,Li_jk,noa)
                           call dgemm('t','n',noa,nob2,nob,1.0d0,l2b_miay(:,:,a,c),nob,H2C_ooov_mjkz(:,:,:,b),nob,1.0d0,Li_jk,noa)
                           ! Diagram 2: A(jk) H2C_vovv(e,k,b,c)*l2b(a,e,i,j)
                           call dgemm('t','n',noanob,nob,nub,1.0d0,l2b_eijx(:,:,:,a),nub,H2C_vovv(:,:,b,c),nub,1.0d0,Lij_k,noanob)
                           ! Diagram 4: -A(jk) H2B_oovo(i,j,a,m)*l2c(b,c,m,k)
                           call dgemm('t','n',noanob,nob,nob,-1.0d0,H2B_oovo_mijx(:,:,:,a),nob,l2c_mkyz(:,:,b,c),nob,1.0d0,Lij_k,noanob)
                           ! Diagram 3: A(jk)A(bc) H2B_vovv(e,j,a,b)*l2b(e,c,i,k)
                           call dgemm('t','n',nob,noanob,nua,1.0d0,H2B_vovv(:,:,a,b),nua,l2b_eikz(:,:,:,c),nua,1.0d0,Lj_ik,nob)
                           call dgemm('t','n',nob,noanob,nua,-1.0d0,H2B_vovv(:,:,a,c),nua,l2b_eikz(:,:,:,b),nua,1.0d0,Lj_ik,nob)
                           ! Diagram 6: -A(jk)A(bc) H2B_ooov(i,k,m,c)*l2b(a,b,m,j)
                           call dgemm('t','n',nob,noanob,noa,-1.0d0,l2b_mjay(:,:,a,b),noa,H2B_ooov_mikz(:,:,:,c),noa,1.0d0,Lj_ik,nob)
                           call dgemm('t','n',nob,noanob,noa,1.0d0,l2b_mjay(:,:,a,c),noa,H2B_ooov_mikz(:,:,:,b),noa,1.0d0,Lj_ik,nob)

                           Dv = fA_v(a) + fB_v(b) + fB_v(c)
                           Dvb = H1A_v(a) + H1B_v(b) + H1B_v(c)
                           Dvc = H2B_vvvv(a,b) + H2B_vvvv(a,c) + H2C_vvvv(c,b)
                           dA = 0.0d0; dB = 0.0d0; dC = 0.0d0; dD = 0.0d0;
                           do i = 1, noa
                              do j = 1, nob
                                 do k = j+1, nob
                                    if (.not. qspace(i,j,k)) cycle
                                    if (npart >= act(1) .and. act(1) > 0) then
                                       nhole = count((/i/) > noa - act(2)) + count((/j,k/) > nob - act(4))
                                       if (nhole >= act(1)) cycle
                                    end if

                                    temp1 = Wi_jk(i,j,k) + Wij_k(i,j,k) - Wij_k(i,k,j) + Wj_ik(j,i,k) - Wj_ik(k,i,j)
                                    temp2 = Li_jk(i,j,k) + Lij_k(i,j,k) - Lij_k(i,k,j) + Lj_ik(j,i,k) - Lj_ik(k,i,j)
                                    temp3 = l1b(c,k)*vB_oovv(i,j,a,b)&
                                    -l1b(b,k)*vB_oovv(i,j,a,c)&
                                    -l1b(c,j)*vB_oovv(i,k,a,b)&
                                    +l1b(b,j)*vB_oovv(i,k,a,c)&
                                    +l1a(a,i)*vC_oovv(j,k,b,c)&
                                    +H1B_ov(k,c)*l2b(a,b,i,j)&
                                    -H1B_ov(k,b)*l2b(a,c,i,j)&
                                    -H1B_ov(j,c)*l2b(a,b,i,k)&
                                    +H1B_ov(j,b)*l2b(a,c,i,k)&
                                    +H1A_ov(i,a)*l2c(b,c,j,k)

                                    LM = temp1*(temp2+temp3)

                                    D = fA_o(i) + fB_o(j) + fB_o(k) - Dv
                                    dA = dA + LM/D

                                    D = H1A_o(i) + H1B_o(j) + H1B_o(k) - Dvb
                                    dB = dB + LM/D

                                    D = D &
                                    -dvoA(a,i)+dovB(i,b)+dovB(i,c)&
                                    +dvoB(a,j)-dvoC(b,j)-dvoC(c,j)&
                                    +dvoB(a,k)-dvoC(b,k)-dvoC(c,k)&
                                    -dooB(i,j)-dooB(i,k)-dooC(j,k)&
                                    -Dvc
                                    dC = dC + LM/D

                                    D = D &
                                    +D3B_O(a,i,j)+D3B_O(a,i,k)&
                                    +D3C_O(b,i,j)+D3C_O(b,i,k)+D3D_O(b,j,k)&
                                    +D3C_O(c,i,j)+D3C_O(c,i,k)+D3D_O(c,j,k)&
                                    -D3B_V(a,i,b)-D3B_V(a,i,c)&
                                    -D3C_V(a,j,b)-D3C_V(a,j,c)-D3D_V(b,j,c)&
                                    -D3C_V(a,k,b)-D3C_V(a,k,c)-D3D_V(b,k,c)
                                    dD = dD + LM/D
                                 end do
                              end do
                           end do
                           e_abc(:,it) = (/dA, dB, dC, dD/)
                        end do
                        !$omp end do
                        deallocate(Wi_jk, Wij_k, Wj_ik, Li_jk, Lij_k, Lj_ik, qspace)
                        !$omp end parallel

              end subroutine crcc23c_abc

end module stochastic_triples_loops
