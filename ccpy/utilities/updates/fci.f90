module fci

        implicit none

        contains

            subroutine hamiltonian_matrix(H, z_a, z_b, v_aa, v_ab, v_bb, e_nuc, onv, dets, nocc, norb, nelectrons, nsorb, ndet)

                integer, intent(in) :: nelectrons, nocc, norb, nsorb, ndet
                real(kind=8), intent(in) :: z_a(norb,norb), z_b(norb,norb)
                real(kind=8), intent(in) :: v_aa(norb,norb,norb,norb), v_ab(norb,norb,norb,norb), v_bb(norb,norb,norb,norb)
                real(kind=8), intent(in) :: e_nuc
                integer, intent(in) :: onv(ndet,nsorb), dets(ndet,nelectrons)

                real(kind=8), intent(out) :: H(ndet,ndet)

                ! Local variables
                integer :: idet, jdet, p, q, m, n, exc_l(2), exc_r(2), l1, l2, isum, idiff, isgn, istep
                integer, allocatable :: bra(:), ket(:), isub(:), itmp(:)

                allocate(bra(1:nsorb), ket(1:nsorb), isub(1:nsorb), itmp(1:nsorb))

                H = 0.0d0
                do idet = 1, ndet

                   do p = 1, nsorb
                      bra(p) = onv(idet,p)
                   end do

                   do jdet = idet, ndet
                      
                      do p = 1, nsorb
                         ket(p) = onv(jdet,p)
                         itmp(p) = ket(p)
                         isub(p) = bra(p) - ket(p)
                      end do

                      idiff = 0
                      do p = 1, nsorb
                         idiff = idiff + abs(isub(p))
                      end do
                      idiff = idiff/2

                      select case (idiff)

                             case (2)
                                     l1 = 1
                                     l2 = 1
                                     do p = 1, nsorb
                                        if (isub(p) == 1) then
                                            exc_l(l1) = p
                                            l1 = l1 + 1
                                        end if
                                        if (isub(p) == -1) then
                                            exc_r(l2) = p
                                            l2 = l2 + 1
                                        end if
                                     end do 
                                     istep = (exc_r(1) - exc_l(1)) / (abs(exc_r(1) - exc_l(1)))
                                     isum = 0
                                     do p = exc_l(1), exc_r(1), istep
                                        isum = isum + ket(p)
                                     end do
                                     isum = isum - 1
                                     isgn = (-1) ** isum
                                     itmp(exc_r(1)) = 0
                                     itmp(exc_l(1)) = 1
                                     istep = (exc_r(2) - exc_l(2)) / (abs(exc_r(2) - exc_l(2)))
                                     isum = 0
                                     do p = exc_l(2), exc_r(2), istep
                                        isum = isum + ket(p)
                                     end do
                                     isum = isum - 1
                                     isgn = isgn * (-1) ** isum

                                     H(idet,jdet) = H(idet,jdet) + get_v(v_aa,v_ab,v_bb,exc_l(1),exc_l(2),exc_r(1),exc_r(2)) * isgn
                                     H(jdet,idet) = H(idet,jdet)

                             case (1)
                                     do p = 1, nsorb
                                        if (isub(p) == 1) exc_l(1) = p
                                        if (isub(p) == -1) exc_r(1) = p
                                     end do
                                     istep = (exc_r(1) - exc_l(1)) / (abs(exc_r(1) - exc_l(1)))
                                     isum = 0
                                     do p = exc_l(1), exc_r(1), istep
                                        isum = isum + ket(p)
                                     end do
                                     isum = isum - 1
                                     isgn = (-1) ** isum
                                     
                                     H(idet,jdet) = H(idet,jdet) + get_z(z_a,z_b,exc_l(1),exc_r(1)) * isgn
                                     do p = 1, nsorb
                                        m = dets(idet,p)
                                        H(idet,jdet) = H(idet,jdet) + get_v(v_aa,v_ab,v_bb,exc_l(1),m,exc_r(1),m) * isgn
                                     end do
                                     H(jdet,idet) = H(idet,jdet)
                             case (0)
                                     do p = 1, nsorb
                                        m = dets(idet,p)
                                        H(idet,jdet) = H(idet,jdet) + get_z(z_a,z_b,m,m)
                                        do q = 1, nsorb
                                           n = dets(idet,q)
                                           H(idet,jdet) = H(idet,jdet) + 0.5d0 * get_v(v_aa,v_ab,v_bb,m,n,m,n)
                                        end do
                                     end do
                                     H(idet,jdet) = H(idet,jdet) + e_nuc
                                     H(jdet,idet) = H(idet,jdet)
                      end select                     
                   end do
                end do
                deallocate(bra,ket,isub,itmp)

            end subroutine hamiltonian_matrix 

                
            function get_z(z_a, z_b, i, a) result(z_int)

                ! Get one-body matrix element <i|z|a>

                ! In:
                !    ints: system's integrals
                !    i: ith spin-orbital
                !    a: ath spin-orbital

                ! Out:
                !    z_int: one-body operator matrix element

                real(kind=8) :: z_int
                real(kind=8), intent(in) :: z_a(:,:), z_b(:,:)
                integer, intent(in) :: i, a

                ! Spatial orbitals
                integer :: i_sp, a_sp

                i_sp = int((i + 1) / 2)
                a_sp = int((a + 1) / 2)

                ! Choose spin case
                if (mod(i, 2) == 0) then
                    z_int = z_b(a_sp, i_sp)
                else
                    z_int = z_a(a_sp, i_sp)
                endif

            end function get_z


            function get_v(v_aa, v_ab, v_bb, i, j, a, b) result(v_int)

                ! Get two-body matrix element <ij|v|ab>

                ! In:
                !    ints: system's integrals
                !    i: ith spin-orbital
                !    j: jth spin-orbital
                !    a: ath spin-orbital
                !    b: bth spin-orbital

                ! Out:
                !    v_int: two-body operator matrix element

                real(kind=8) :: v_int
                real(kind=8), intent(in) :: v_aa(:,:,:,:), v_ab(:,:,:,:), v_bb(:,:,:,:)
                integer, intent(in) :: i, j, a, b
                integer :: dod
                integer :: i_sp, j_sp, a_sp, b_sp

                ! Spatial orbitals
                i_sp = int((i + 1) / 2)
                j_sp = int((j + 1) / 2)
                a_sp = int((a + 1) / 2)
                b_sp = int((b + 1) / 2)

                ! Total spin
                dod = mod(i,2) + mod(j,2) + mod(a,2) + mod(b,2)

                v_int = 0.0

                ! All spins are the same
                if (dod == 4) then
                    v_int = v_aa(a_sp, b_sp, i_sp, j_sp)
                else if (dod == 0) then
                    v_int = v_bb(a_sp, b_sp, i_sp, j_sp)
                else if (mod(i, 2) == mod(a, 2)) then
                    ! Bra and ket indices match spin
                    v_int = v_ab(a_sp, b_sp, i_sp, j_sp)
                else if (mod(i, 2) == mod(b, 2)) then
                    ! Bra and ket indices are flipped
                    v_int = -v_ab(b_sp, a_sp, i_sp, j_sp)
                endif

            end function get_v

end module fci
