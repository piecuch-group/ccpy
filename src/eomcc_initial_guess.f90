module eomcc_initial_guess

        implicit none

        contains

                ! There is something wrong here. I am confused whether we should
                ! enforce that
                ! sym_target = sym(K) * sym(H) * sym(L) 
                ! or
                ! 0 = sym(K) * sym(H) * sym(L)

                subroutine get_active_dimensions(idx1A,idx1B,idx2A,idx2B,idx2C,&
                                syms1A,syms1B,syms2A,syms2B,syms2C,&
                                n1a_act,n1b_act,n2a_act,n2b_act,n2c_act,noact,nuact,&
                                mo_syms,mult_table,&
                                noa,nua,nob,nub,norb,h_group)

                        integer, intent(in) :: noa, nua, nob, nub, noact, nuact, norb, h_group
                        integer, intent(in) :: mo_syms(norb), mult_table(0:h_group-1,0:h_group-1)
                        integer, intent(out) :: idx1A(nua,noa), idx1B(nub,nob),& 
                                                idx2A(nua,nua,noa,noa), idx2B(nua,nub,noa,nob), idx2C(nub,nub,nob,nob),&
                                                syms1A(nua,noa), syms1B(nub,nob),&
                                                syms2A(nua,nua,noa,noa), syms2B(nua,nub,noa,nob), syms2C(nub,nub,nob,nob),&
                                                n1a_act, n1b_act, n2a_act, n2b_act, n2c_act

                        integer :: a, b, i, j, act_rng_oa(2), act_rng_ua(2), act_rng_ob(2), act_rng_ub(2),&
                                   num_act_holes, num_act_particles, sa, sb, si, sj

                        act_rng_oa(1) = max(0, noa-noact)
                        act_rng_oa(2) = noa
                        act_rng_ua(1) = 0
                        act_rng_ua(2) = min(nua, nuact)
                        act_rng_ob(1) = max(0, nob-noact)
                        act_rng_ob(2) = nob
                        act_rng_ub(1) = 0
                        act_rng_ub(2) = min(nub, nuact)

                        idx1A = 0
                        syms1A = -1
                        n1a_act = 0
                        do i = 1,noa
                           do a = 1,nua
                                 sa = mo_syms(a+noa)
                                 si = mo_syms(i)
                                 syms1A(a,i) = mult_table(sa,si)
                                 idx1A(a,i) = 1
                                 n1a_act = n1a_act + 1
                           end do
                        end do

                        idx1B = 0
                        syms1B = -1
                        n1b_act = 0
                        do i = 1,nob
                           do a = 1,nub
                                 idx1B(a,i) = 1
                                 sa = mo_syms(a+nob)
                                 si = mo_syms(i)
                                 syms1B(a,i) = mult_table(sa,si)
                                 n1b_act = n1b_act + 1
                           end do
                        end do

                        idx2A = 0
                        syms2A = -1
                        n2a_act = 0
                        do i = 1, noa
                        do j = i+1, noa
                        do a = 1, nua
                        do b = a+1, nua
                            num_act_holes = 0
                            num_act_particles = 0
                            if (i>act_rng_oa(1) .and. i<=act_rng_oa(2)) then
                                num_act_holes = num_act_holes + 1
                            end if
                            if (j>act_rng_oa(1) .and. j<=act_rng_oa(2)) then
                                num_act_holes = num_act_holes + 1
                            end if
                            if (a>act_rng_ua(1) .and. a<=act_rng_ua(2)) then
                                num_act_particles = num_act_particles + 1
                            end if
                            if (b>act_rng_ua(1) .and. b<=act_rng_ua(2)) then
                                num_act_particles = num_act_particles + 1
                            end if
                            if (num_act_holes >= 1 .and. num_act_particles >= 1) then
                                sa = mo_syms(a+noa)
                                sb = mo_syms(b+noa)
                                si = mo_syms(i)
                                sj = mo_syms(j)
                                syms2A(a,b,i,j) = mult_table(sa,mult_table(sb,mult_table(si,sj)))
                                syms2A(b,a,i,j) = syms2A(a,b,i,j)
                                syms2A(a,b,j,i) = syms2A(a,b,i,j)
                                syms2A(b,a,j,i) = syms2A(a,b,i,j)
                                idx2A(a,b,i,j) = 1
                                idx2A(b,a,i,j) = 1
                                idx2A(a,b,j,i) = 1
                                idx2A(b,a,j,i) = 1
                                n2a_act = n2a_act + 1
                            end if
                        end do
                        end do
                        end do
                        end do

                        idx2B = 0
                        syms2B = -1
                        n2b_act = 0
                        do i = 1, noa
                        do j = 1, nob
                        do a = 1, nua
                        do b = 1, nub
                            num_act_holes = 0
                            num_act_particles = 0
                            if (i>act_rng_oa(1) .and. i<=act_rng_oa(2)) then
                                num_act_holes = num_act_holes + 1
                            end if
                            if (j>act_rng_ob(1) .and. j<=act_rng_ob(2)) then
                                num_act_holes = num_act_holes + 1
                            end if
                            if (a>act_rng_ua(1) .and. a<=act_rng_ua(2)) then
                                num_act_particles = num_act_particles + 1
                            end if
                            if (b>act_rng_ub(1) .and. b<=act_rng_ub(2)) then
                                num_act_particles = num_act_particles + 1
                            end if
                            if (num_act_holes >= 1 .and. num_act_particles >= 1) then
                                sa = mo_syms(a+noa)
                                sb = mo_syms(b+nob)
                                si = mo_syms(i)
                                sj = mo_syms(j)
                                syms2B(a,b,i,j) = mult_table(sa,mult_table(sb,mult_table(si,sj)))
                                idx2B(a,b,i,j) = 1
                                n2b_act = n2b_act + 1
                            end if
                        end do
                        end do
                        end do
                        end do

                        idx2C = 0
                        syms2C = -1
                        n2c_act = 0
                        do i = 1, nob
                        do j = i+1, nob
                        do a = 1, nub
                        do b = a+1, nub
                            num_act_holes = 0
                            num_act_particles = 0
                            if (i>act_rng_ob(1) .and. i<=act_rng_ob(2)) then
                                num_act_holes = num_act_holes + 1
                            end if
                            if (j>act_rng_ob(1) .and. j<=act_rng_ob(2)) then
                                num_act_holes = num_act_holes + 1
                            end if
                            if (a>act_rng_ub(1) .and. a<=act_rng_ub(2)) then
                                num_act_particles = num_act_particles + 1
                            end if
                            if (b>act_rng_ub(1) .and. b<=act_rng_ub(2)) then
                                num_act_particles = num_act_particles + 1
                            end if
                            if (num_act_holes >= 1 .and. num_act_particles >= 1) then
                                sa = mo_syms(a+nob)
                                sb = mo_syms(b+nob)
                                si = mo_syms(i)
                                sj = mo_syms(j)
                                syms2C(a,b,i,j) = mult_table(sa,mult_table(sb,mult_table(si,sj)))
                                syms2C(b,a,i,j) = syms2C(a,b,i,j)
                                syms2C(a,b,j,i) = syms2C(a,b,i,j)
                                syms2C(b,a,j,i) = syms2C(a,b,i,j)
                                idx2C(a,b,i,j) = 1
                                idx2C(b,a,i,j) = 1
                                idx2C(a,b,j,i) = 1
                                idx2C(b,a,j,i) = 1
                                n2c_act = n2c_act + 1
                            end if
                        end do
                        end do
                        end do
                        end do

                end subroutine get_active_dimensions
            
                subroutine unflatten_guess_vector(r1a,r1b,r2a,r2b,r2c,&
                                CIvec,&
                                idx1A,idx1B,idx2A,idx2B,idx2C,&
                                noa,nua,nob,nub,&
                                n1a_act,n1b_act,n2a_act,n2b_act,n2c_act,ndim_act)

                        integer, intent(in) :: noa, nua, nob, nub, n1a_act, n1b_act,&
                                n2a_act, n2b_act, n2c_act, ndim_act,&
                                idx1A(nua,noa), idx1B(nub,nob),&
                                idx2A(nua,nua,noa,noa), idx2B(nua,nub,noa,nob), idx2C(nub,nub,nob,nob)
                        real(kind=8), intent(in) :: CIvec(ndim_act)

                        real(kind=8), intent(out) :: r1a(nua,noa), r1b(nub,nob),&
                        r2a(nua,nua,nob,nob), r2b(nua,nub,noa,nob), r2c(nub,nub,nob,nob)

                        integer :: i, j, a, b, ct

                        ct = 0
                        r1a = 0.0d0
                        do i = 1,noa
                           do a = 1,nua
                              if (idx1A(a,i)==0) cycle
                              ct = ct + 1
                              r1a(a,i) = CIvec(ct)
                           end do
                        end do
                        r1b = 0.0d0
                        do i = 1,nob
                           do a = 1,nub
                              if (idx1B(a,i)==0) cycle
                              ct = ct + 1
                              r1b(a,i) = CIvec(ct)
                           end do
                        end do
                        r2a = 0.0d0
                        do i = 1,noa
                           do j = i+1,noa
                              do a = 1,nua
                                 do b = a+1,nua
                                    if (idx2A(b,a,j,i)==0) cycle
                                    ct = ct + 1
                                    r2a(b,a,j,i) = CIvec(ct)
                                    r2a(a,b,j,i) = -r2a(b,a,j,i)
                                    r2a(b,a,i,j) = -r2a(b,a,j,i)
                                    r2a(a,b,i,j) = r2a(b,a,j,i)
                                 end do
                              end do
                           end do
                        end do 
                        r2b = 0.0d0
                        do j = 1,nob
                           do i = 1,noa
                              do b = 1,nub
                                 do a = 1,nua
                                    if (idx2B(a,b,i,j)==0) cycle
                                    ct = ct + 1
                                    r2b(a,b,i,j) = CIvec(ct)
                                 end do
                              end do
                           end do
                        end do 
                        r2c = 0.0d0
                        do i = 1,nob
                           do j = i+1,nob
                              do a = 1,nub
                                 do b = a+1,nub
                                    if (idx2C(b,a,j,i)==0) cycle
                                    ct = ct + 1
                                    r2c(b,a,j,i) = CIvec(ct)
                                    r2c(a,b,j,i) = -r2c(b,a,j,i)
                                    r2c(b,a,i,j) = -r2c(b,a,j,i)
                                    r2c(a,b,i,j) = r2c(b,a,j,i)
                                 end do
                              end do
                           end do
                        end do

                end subroutine unflatten_guess_vector


                subroutine eomccs_d_matrix(CIvec,omega,Hmat,idx1A,idx1B,idx2A,idx2B,idx2C,&
                                    H1A_oo,H1A_vv,H1A_ov,&
                                    H1B_oo,H1B_vv,H1B_ov,&
                                    H2A_oooo,H2A_vvvv,H2A_voov,H2A_vooo,H2A_vvov,H2A_ooov,H2A_vovv,&
                                    H2B_oooo,H2B_vvvv,H2B_voov,H2B_ovvo,H2B_vovo,H2B_ovov,H2B_vooo,&
                                    H2B_ovoo,H2B_vvov,H2B_vvvo,H2B_ooov,H2B_oovo,H2B_vovv,H2B_ovvv,&
                                    H2C_oooo,H2C_vvvv,H2C_voov,H2C_vooo,H2C_vvov,H2C_ooov,H2C_vovv,&
                                    noa,nua,nob,nub,n1a_act,n1b_act,n2a_act,n2b_act,n2c_act,ndim_act,&
                                    sym_target,sym_ref,syms1A,syms1B,syms2A,syms2B,syms2C,mult_table,h_group)

                        integer, intent(in) :: noa, nua, nob, nub, n1a_act, n1b_act, n2a_act, n2b_act, n2c_act, ndim_act,&
                                               idx1A(nua,noa), idx1B(nub,nob),&
                                               idx2A(nua,nua,noa,noa), idx2B(nua,nub,noa,nob), idx2C(nub,nub,nob,nob),&
                                               syms1A(nua,noa), syms1B(nub,nob),&
                                               syms2A(nua,nua,noa,noa), syms2B(nua,nub,noa,nob), syms2C(nub,nub,nob,nob),&
                                               sym_target, sym_ref, h_group
                        integer, intent(in) :: mult_table(0:h_group-1,0:h_group-1)
                        real(kind=8), intent(in) :: H1A_oo(noa,noa),H1A_vv(nua,nua),H1A_ov(noa,nua),&
                                    H1B_oo(nob,nob),H1B_vv(nub,nub),H1B_ov(nob,nub),&
                                    H2A_oooo(noa,noa,noa,noa),H2A_vvvv(nua,nua,nua,nua),H2A_voov(nua,noa,noa,nua),&
                                    H2A_vooo(nua,noa,noa,noa),H2A_vvov(nua,nua,noa,nua),H2A_ooov(noa,noa,noa,nua),&
                                    H2A_vovv(nua,noa,nua,nua),&
                                    H2B_oooo(noa,nob,noa,nob),H2B_vvvv(nua,nub,nua,nub),H2B_voov(nua,nob,noa,nub),&
                                    H2B_ovvo(noa,nub,nua,nob),H2B_vovo(nua,nob,nua,nob),H2B_ovov(noa,nub,noa,nub),&
                                    H2B_vooo(nua,nob,noa,nob),H2B_ovoo(noa,nub,noa,nob),H2B_vvov(nua,nub,noa,nub),&
                                    H2B_vvvo(nua,nub,nua,nob),H2B_ooov(noa,nob,noa,nub),H2B_oovo(noa,nob,nua,nob),&
                                    H2B_vovv(nua,nob,nua,nub),H2B_ovvv(noa,nub,nua,nub),&
                                    H2C_oooo(nob,nob,nob,nob),H2C_vvvv(nub,nub,nub,nub),H2C_voov(nub,nob,nob,nub),&
                                    H2C_vooo(nub,nob,nob,nob),H2C_vvov(nub,nub,nob,nub),H2C_ooov(nob,nob,nob,nub),&
                                    H2C_vovv(nub,nob,nub,nub)

                        real(kind=8), intent(out) :: CIvec(ndim_act,ndim_act), omega(ndim_act), Hmat(ndim_act,ndim_act)
                        
                        real(kind=8) :: Hmat2(ndim_act,ndim_act)
                        real(kind=8), allocatable :: Htemp(:,:), VL(:,:), wi(:), work(:)
                        integer :: i, j, k, l, a, b, c, d, ct1, ct2, pos(6), info, g1, g2

                        pos(1) = 0
                        pos(2) = n1a_act
                        pos(3) = n1a_act+n1b_act
                        pos(4) = n1a_act+n1b_act+n2a_act
                        pos(5) = n1a_act+n1b_act+n2a_act+n2b_act
                        pos(6) = n1a_act+n1b_act+n2a_act+n2b_act+n2c_act

                        ! < ia | H | jb >
                        allocate(Htemp(n1a_act,n1a_act))
                        Htemp = 0.0d0
                        ct1 = 0
                        do i = 1 , noa
                        do a = 1 , nua
                            g1 = syms1A(a,i)
                            if (idx1A(a,i)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do j = 1 , noa
                            do b = 1 , nua
                                g2 = syms1A(b,j)
                                if (idx1A(b,j)==0) cycle
                                ct2 = ct2 + 1
                                Htemp(ct1,ct2) = &
                                calc_SASA_matel(i,a,j,b,H1A_oo,H1A_vv,H2A_voov)
                            end do
                            end do
                        end do
                        end do
                        Hmat(pos(1)+1:pos(2),pos(1)+1:pos(2)) = Htemp
                        deallocate(Htemp)

                        ! < i~a~ | H | jb >
                        allocate(Htemp(n1b_act,n1a_act))
                        Htemp = 0.0d0
                        ct1 = 0
                        do i = 1 , nob
                        do a = 1 , nub
                            g1 = syms1B(a,i)
                            if (idx1B(a,i)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do j = 1 , noa
                            do b = 1 , nua
                                g2 = syms1A(b,j)
                                if (idx1A(b,j)==0) cycle
                                ct2 = ct2 + 1
                                Htemp(ct1,ct2) =&
                                calc_SBSA_matel(i,a,j,b,H2B_ovvo)
                            end do
                            end do
                        end do
                        end do
                        Hmat(pos(2)+1:pos(3),pos(1)+1:pos(2)) = Htemp
                        deallocate(Htemp)

                        ! < ia | H | j~b~ >
                        allocate(Htemp(n1a_act,n1b_act))
                        Htemp = 0.0d0
                        ct1 = 0
                        do i = 1 , noa
                        do a = 1 , nua
                            g1 = syms1A(a,i)
                            if (idx1A(a,i)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do j = 1 , nob
                            do b = 1 , nub
                                g2 = syms1B(b,j)
                                if (idx1B(b,j)==0) cycle
                                ct2 = ct2 + 1
                                Htemp(ct1,ct2) =&
                                calc_SASB_matel(i,a,j,b,H2B_voov)
                            end do
                            end do
                        end do
                        end do
                        Hmat(pos(1)+1:pos(2),pos(2)+1:pos(3)) = Htemp
                        deallocate(Htemp)

                        ! < i~a~ | H | j~b~ >
                        allocate(Htemp(n1b_act,n1b_act))
                        Htemp = 0.0d0
                        ct1 = 0
                        do i = 1 , nob
                        do a = 1 , nub
                            g1 = syms1B(a,i)
                            if (idx1B(a,i)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do j = 1 , nob
                            do b = 1 , nub
                                g2 = syms1B(b,j)
                                if (idx1B(b,j)==0) cycle
                                ct2 = ct2 + 1
                                Htemp(ct1,ct2) =&
                                calc_SBSB_matel(i,a,j,b,H1B_oo,H1B_vv,H2C_voov)
                            end do
                            end do
                        end do
                        end do
                        Hmat(pos(2)+1:pos(3),pos(2)+1:pos(3)) = Htemp
                        deallocate(Htemp)

                        ! < ia | H | jkbc >
                        allocate(Htemp(n1a_act,n2a_act))
                        Htemp = 0.0d0
                        ct1 = 0
                        do i = 1, noa
                        do a = 1, nua
                            g1 = syms1A(a,i)
                            if (idx1A(a,i)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do j = 1, noa
                            do k = j+1, noa
                            do b = 1, nua
                            do c = b+1, nua
                                g2 = syms2A(b,c,j,k)
                                if (idx2A(b,c,j,k)==0) cycle
                                ct2 = ct2 + 1
                                Htemp(ct1,ct2) =&
                                calc_SADA_matel(i,a,j,k,b,c,H1A_ov,H2A_ooov,H2A_vovv)
                            end do
                            end do
                            end do
                            end do
                        end do
                        end do
                        Hmat(pos(1)+1:pos(2),pos(3)+1:pos(4)) = Htemp
                        deallocate(Htemp)

                        ! < ia | H | jk~bc~ >
                        allocate(Htemp(n1a_act,n2b_act))
                        Htemp = 0.0d0
                        ct1 = 0
                        do i = 1, noa
                        do a = 1, nua
                            g1 = syms1A(a,i)
                            if (idx1A(a,i)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do j = 1, noa
                            do k = 1, nob
                            do b = 1, nua
                            do c = 1, nub
                                g2 = syms2B(b,c,j,k)
                                if (idx2B(b,c,j,k)==0) cycle
                                ct2 = ct2 + 1
                                Htemp(ct1,ct2) =&
                                calc_SADB_matel(i,a,j,k,b,c,H1B_ov,H2B_ooov,H2B_vovv)
                            end do
                            end do
                            end do
                            end do
                        end do
                        end do
                        Hmat(pos(1)+1:pos(2),pos(4)+1:pos(5)) = Htemp
                        deallocate(Htemp)

                        ! < i~a~ | H | jk~bc~ >
                        allocate(Htemp(n1b_act,n2b_act))
                        Htemp = 0.0d0
                        ct1 = 0
                        do i = 1, nob
                        do a = 1, nub
                            g1 = syms1B(a,i)
                            if (idx1B(a,i)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do j = 1, noa
                            do k = 1, nob
                            do b = 1, nua
                            do c = 1, nub
                                g2 = syms2B(b,c,j,k)
                                if (idx2B(b,c,j,k)==0) cycle
                                ct2 = ct2 + 1
                                Htemp(ct1,ct2) =&
                                calc_SBDB_matel(i,a,j,k,b,c,H1A_ov,H2B_oovo,H2B_ovvv)
                            end do
                            end do
                            end do
                            end do
                        end do
                        end do
                        Hmat(pos(2)+1:pos(3),pos(4)+1:pos(5)) = Htemp
                        deallocate(Htemp)

                        ! < i~a~ | H | j~k~b~c~ >
                        allocate(Htemp(n1b_act,n2c_act))
                        Htemp = 0.0d0
                        ct1 = 0
                        do i = 1, nob
                        do a = 1, nub
                            g1 = syms1B(a,i)
                            if (idx1B(a,i)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do j = 1, nob
                            do k = j+1, nob
                            do b = 1, nub
                            do c = b+1, nub
                                g2 = syms2C(b,c,j,k)
                                if (idx2C(b,c,j,k)==0) cycle
                                ct2 = ct2 + 1
                                Htemp(ct1,ct2) =&
                                calc_SBDC_matel(i,a,j,k,b,c,H1B_ov,H2C_ooov,H2C_vovv)
                            end do
                            end do
                            end do
                            end do
                        end do
                        end do
                        Hmat(pos(2)+1:pos(3),pos(5)+1:pos(6)) = Htemp
                        deallocate(Htemp)

                        ! < ijab | H | kc >
                        allocate(Htemp(n2a_act,n1a_act))
                        Htemp = 0.0d0
                        ct1 = 0
                        do i = 1, noa
                        do j = i+1, noa
                        do a = 1, nua
                        do b = a+1, nua
                            g1 = syms2A(a,b,i,j)
                            if (idx2A(a,b,i,j)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, noa
                            do c = 1, nua
                                g2 = syms1A(c,k)
                                if (idx1A(c,k)==0) cycle
                                ct2 = ct2 + 1
                                Htemp(ct1,ct2) =&
                                calc_DASA_matel(i,j,a,b,k,c,H2A_vooo,H2A_vvov)
                            end do
                            end do
                        end do
                        end do
                        end do
                        end do
                        Hmat(pos(3)+1:pos(4),pos(1)+1:pos(2)) = Htemp
                        deallocate(Htemp)

                        ! < ij~ab~ | H | kc >
                        allocate(Htemp(n2b_act,n1a_act))
                        Htemp = 0.0d0
                        ct1 = 0
                        do i = 1, noa
                        do j = 1, nob
                        do a = 1, nua
                        do b = 1, nub
                            g1 = syms2B(a,b,i,j)
                            if (idx2B(a,b,i,j)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, noa
                            do c = 1, nua
                                g2 = syms1A(c,k)
                                if (idx1A(c,k)==0) cycle
                                ct2 = ct2 + 1
                                Htemp(ct1,ct2) =&
                                calc_DBSA_matel(i,j,a,b,k,c,H2B_ovoo,H2B_vvvo)
                            end do
                            end do
                        end do
                        end do
                        end do
                        end do
                        Hmat(pos(4)+1:pos(5),pos(1)+1:pos(2)) = Htemp
                        deallocate(Htemp)

                        ! < ij~ab~ | H | k~c~ >
                        allocate(Htemp(n2b_act,n1b_act))
                        Htemp = 0.0d0
                        ct1 = 0
                        do i = 1, noa
                        do j = 1, nob
                        do a = 1, nua
                        do b = 1, nub
                            g1 = syms2B(a,b,i,j)
                            if (idx2B(a,b,i,j)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, nob
                            do c = 1, nub
                                g2 = syms1B(c,k)
                                if (idx1B(c,k)==0) cycle
                                ct2 = ct2 + 1
                                Htemp(ct1,ct2) =&
                                calc_DBSB_matel(i,j,a,b,k,c,H2B_vooo,H2B_vvov)
                            end do
                            end do
                        end do
                        end do
                        end do
                        end do
                        Hmat(pos(4)+1:pos(5),pos(2)+1:pos(3)) = Htemp
                        deallocate(Htemp)

                        ! < i~j~a~b~ | H | k~c~ >
                        allocate(Htemp(n2c_act,n1b_act))
                        Htemp = 0.0d0
                        ct1 = 0
                        do i = 1, nob
                        do j = i+1, nob
                        do a = 1, nub
                        do b = a+1, nub
                            g1 = syms2C(a,b,i,j)
                            if (idx2C(a,b,i,j)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, nob
                            do c = 1, nub
                                g2 = syms1B(c,k)
                                if (idx1B(c,k)==0) cycle
                                ct2 = ct2 + 1
                                Htemp(ct1,ct2) =&
                                calc_DCSB_matel(i,j,a,b,k,c,H2C_vooo,H2C_vvov)
                            end do
                            end do
                        end do
                        end do
                        end do
                        end do
                        Hmat(pos(5)+1:pos(6),pos(2)+1:pos(3)) = Htemp
                        deallocate(Htemp)

                        ! < ijab | H | klcd >
                        allocate(Htemp(n2a_act,n2a_act))
                        Htemp = 0.0d0
                        ct1 = 0
                        do i = 1, noa
                        do j = i+1, noa
                        do a = 1, nua
                        do b = a+1, nua
                            g1 = syms2A(a,b,i,j)
                            if (idx2A(a,b,i,j)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, noa
                            do l = k+1, noa
                            do c = 1, nua
                            do d = c+1, nua
                                g2 = syms2A(c,d,k,l)
                                if (idx2A(c,d,k,l)==0) cycle
                                ct2 = ct2 + 1
                                Htemp(ct1,ct2) =&
                                calc_DADA_matel(i,j,a,b,k,l,c,d,H1A_oo,H1A_vv,H2A_voov,H2A_oooo,H2A_vvvv)
                            end do
                            end do
                            end do
                            end do
                        end do
                        end do
                        end do
                        end do
                        Hmat(pos(3)+1:pos(4),pos(3)+1:pos(4)) = Htemp
                        deallocate(Htemp)

                        ! < ijab | H | kl~cd~ >
                        allocate(Htemp(n2a_act,n2b_act))
                        Htemp = 0.0d0
                        ct1 = 0
                        do i = 1, noa
                        do j = i+1, noa
                        do a = 1, nua
                        do b = a+1, nua
                            g1 = syms2A(a,b,i,j)
                            if (idx2A(a,b,i,j)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, noa
                            do l = 1, nob
                            do c = 1, nua
                            do d = 1, nub
                                g2 = syms2B(c,d,k,l)
                                if (idx2B(c,d,k,l)==0) cycle
                                ct2 = ct2 + 1
                                Htemp(ct1,ct2) =&
                                calc_DADB_matel(i,j,a,b,k,l,c,d,H2B_voov)
                            end do
                            end do
                            end do
                            end do
                        end do
                        end do
                        end do
                        end do
                        Hmat(pos(3)+1:pos(4),pos(4)+1:pos(5)) = Htemp
                        deallocate(Htemp)

                        ! < ij~ab~ | H | klcd >
                        allocate(Htemp(n2b_act,n2a_act))
                        Htemp = 0.0d0
                        ct1 = 0
                        do i = 1, noa
                        do j = 1, nob
                        do a = 1, nua
                        do b = 1, nub
                            g1 = syms2B(a,b,i,j)
                            if (idx2B(a,b,i,j)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, noa
                            do l = k+1, noa
                            do c = 1, nua
                            do d = c+1, nua
                                g2 = syms2A(c,d,k,l)
                                if (idx2A(c,d,k,l)==0) cycle
                                ct2 = ct2 + 1
                                Htemp(ct1,ct2) =&
                                calc_DBDA_matel(i,j,a,b,k,l,c,d,H2B_ovvo)
                            end do
                            end do
                            end do
                            end do
                        end do
                        end do
                        end do
                        end do
                        Hmat(pos(4)+1:pos(5),pos(3)+1:pos(4)) = Htemp
                        deallocate(Htemp)

                        ! < ij~ab~ | H | kl~cd~ >
                        allocate(Htemp(n2b_act,n2b_act))
                        Htemp = 0.0d0
                        ct1 = 0
                        do i = 1, noa
                        do j = 1, nob
                        do a = 1, nua
                        do b = 1, nub
                            g1 = syms2B(a,b,i,j)
                            if (idx2B(a,b,i,j)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, noa
                            do l = 1, nob
                            do c = 1, nua
                            do d = 1, nub
                                g2 = syms2B(c,d,k,l)
                                if (idx2B(c,d,k,l)==0) cycle
                                ct2 = ct2 + 1
                                Htemp(ct1,ct2) =&
                                calc_DBDB_matel(i,j,a,b,k,l,c,d,&
                                H1A_oo,H1A_vv,H1B_oo,H1B_vv,&
                                H2A_voov,&
                                H2B_oooo,H2B_vvvv,H2B_ovov,H2B_vovo,&
                                H2C_voov)
                            end do
                            end do
                            end do
                            end do
                        end do
                        end do
                        end do
                        end do
                        Hmat(pos(4)+1:pos(5),pos(4)+1:pos(5)) = Htemp
                        deallocate(Htemp)

                        ! < ij~ab~ | H | k~l~c~d~ >
                        allocate(Htemp(n2b_act,n2c_act))
                        Htemp = 0.0d0
                        ct1 = 0
                        do i = 1, noa
                        do j = 1, nob
                        do a = 1, nua
                        do b = 1, nub
                            g1 = syms2B(a,b,i,j)
                            if (idx2B(a,b,i,j)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, nob
                            do l = k+1, nob
                            do c = 1, nub
                            do d = c+1, nub
                                g2 = syms2C(c,d,k,l)
                                if (idx2C(c,d,k,l)==0) cycle
                                ct2 = ct2 + 1
                                Htemp(ct1,ct2) =&
                                calc_DBDC_matel(i,j,a,b,k,l,c,d,H2B_voov)
                            end do
                            end do
                            end do
                            end do
                        end do
                        end do
                        end do
                        end do
                        Hmat(pos(4)+1:pos(5),pos(5)+1:pos(6)) = Htemp
                        deallocate(Htemp)

                        ! < i~j~a~b~ | H | k~lc~d >
                        allocate(Htemp(n2c_act,n2b_act))
                        Htemp = 0.0d0
                        ct1 = 0
                        do i = 1, nob
                        do j = i+1, nob
                        do a = 1, nub
                        do b = a+1, nub
                            g1 = syms2C(a,b,i,j)
                            if (idx2C(a,b,i,j)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, noa
                            do l = 1, nob
                            do c = 1, nua
                            do d = 1, nub
                                g2 = syms2B(c,d,k,l)
                                if (idx2B(c,d,k,l)==0) cycle
                                ct2 = ct2 + 1
                                Htemp(ct1,ct2) =&
                                calc_DCDB_matel(i,j,a,b,k,l,c,d,H2B_ovvo)
                            end do
                            end do
                            end do
                            end do
                        end do
                        end do
                        end do
                        end do
                        Hmat(pos(5)+1:pos(6),pos(4)+1:pos(5)) = Htemp
                        deallocate(Htemp)

                        ! < i~j~a~b~ | H | k~l~c~d~ >
                        allocate(Htemp(n2c_act,n2c_act))
                        Htemp = 0.0d0
                        ct1 = 0
                        do i = 1, nob
                        do j = i+1, nob
                        do a = 1, nub
                        do b = a+1, nub
                            g1 = syms2C(a,b,i,j)
                            if (idx2C(a,b,i,j)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, nob
                            do l = k+1, nob
                            do c = 1, nub
                            do d = c+1, nub
                                g2 = syms2C(c,d,k,l)
                                if (idx2C(c,d,k,l)==0) cycle
                                ct2 = ct2 + 1
                                Htemp(ct1,ct2) =&
                                calc_DCDC_matel(i,j,a,b,k,l,c,d,H1B_oo,H1B_vv,H2C_voov,H2C_oooo,H2C_vvvv)
                            end do
                            end do
                            end do
                            end do
                        end do
                        end do
                        end do
                        end do
                        Hmat(pos(5)+1:pos(6),pos(5)+1:pos(6)) = Htemp
                        deallocate(Htemp)

                        Hmat2 = Hmat
                        allocate(VL(ndim_act,ndim_act),wi(ndim_act),work(4*ndim_act))
                        call dgeev('N','V',ndim_act,Hmat2,ndim_act,omega,wi,VL,ndim_act,CIvec,ndim_act,&
                                work,4*ndim_act,info)
                        if (info /= 0) then
                            print*,'Problem diagonalizing EOMCCSd matrix'
                        end if
                        deallocate(VL,wi,work)


                end subroutine eomccs_d_matrix


                function calc_SASA_matel(i,a,j,b,H1A_oo,H1A_vv,H2A_voov) result(val)

                    integer, intent(in) :: i, j, a, b
                    real(kind=8), intent(in) :: H1A_oo(:,:), H1A_vv(:,:),&
                    H2A_voov(:,:,:,:)

                    real(kind=8) :: val

                    val = H2A_voov(a,j,i,b)
                    if (i==j) then
                        val = val + H1A_vv(a,b)
                    end if
                    if (a==b) then
                        val = val - H1A_oo(j,i)
                    end if
                    
                end function calc_SASA_matel

                function calc_SASB_matel(i,a,j,b,H2B_voov) result(val)

                    integer, intent(in) :: i, j, a, b
                    real(kind=8), intent(in) :: H2B_voov(:,:,:,:)

                    real(kind=8) :: val

                    val = H2B_voov(a,j,i,b)
                    
                end function calc_SASB_matel

                function calc_SBSA_matel(i,a,j,b,H2B_ovvo) result(val)

                    integer, intent(in) :: i, j, a, b
                    real(kind=8), intent(in) :: H2B_ovvo(:,:,:,:)

                    real(kind=8) :: val

                    val = H2B_ovvo(j,a,b,i)
                    
                end function calc_SBSA_matel

                function calc_SBSB_matel(i,a,j,b,H1B_oo,H1B_vv,H2C_voov) result(val)

                    integer, intent(in) :: i, j, a, b
                    real(kind=8), intent(in) :: H1B_oo(:,:), H1B_vv(:,:),&
                    H2C_voov(:,:,:,:)

                    real(kind=8) :: val

                    val = H2C_voov(a,j,i,b)
                    if (i==j) then
                        val = val + H1B_vv(a,b)
                    end if
                    if (a==b) then
                        val = val - H1B_oo(j,i)
                    end if
                    
                end function calc_SBSB_matel

                function calc_SADA_matel(i,a,j,k,b,c,H1A_ov,H2A_ooov,H2A_vovv) result(val)

                    integer, intent(in) :: i, a, j, k, b, c
                    real(kind=8), intent(in) :: H1A_ov(:,:), H2A_ooov(:,:,:,:),&
                    H2A_vovv(:,:,:,:)
                
                    real(kind=8) :: val

                    val = 0.0d0
                    if (i==k .and. a==c) then
                        val = val + H1A_ov(j,b)
                    end if
                    if (a==b .and. i==j) then
                        val = val + H1A_ov(k,c)
                    end if
                    if (i==j .and. a==c) then
                        val = val - H1A_ov(k,b)
                    end if
                    if (i==k .and. a==b) then
                        val = val - H1A_ov(j,c)
                    end if
                    if (a==b) then
                        val = val - H2A_ooov(j,k,i,c)
                    end if
                    if (a==c) then
                        val = val - H2A_ooov(k,j,i,b)
                    end if
                    if (i==j) then
                        val = val + H2A_vovv(a,k,b,c)
                    end if
                    if (i==k) then
                        val = val + H2A_vovv(a,j,c,b)
                    end if

                end function calc_SADA_matel

                function calc_SADB_matel(i,a,j,k,b,c,H1B_ov,H2B_ooov,H2B_vovv) result(val)

                    integer, intent(in) :: i, a, j, k, b, c
                    real(kind=8), intent(in) :: H1B_ov(:,:),H2B_ooov(:,:,:,:),&
                    H2B_vovv(:,:,:,:)

                    real(kind=8) :: val

                    val = 0.0d0
                    if (a==b) then
                        val = val - H2B_ooov(j,k,i,c)
                    end if
                    if (i==j) then
                        val = val + H2B_vovv(a,k,b,c)
                    end if
                    if (i==j .and. a==b) then
                        val = val + H1B_ov(k,c)
                    end if

                end function calc_SADB_matel

                function calc_SBDB_matel(i,a,j,k,b,c,H1A_ov,H2B_oovo,H2B_ovvv) result(val)

                    integer, intent(in) :: i, a, j, k, b, c
                    real(kind=8), intent(in) :: H1A_ov(:,:), H2B_oovo(:,:,:,:),&
                    H2B_ovvv(:,:,:,:)

                    real(kind=8) :: val

                    val = 0.0d0
                    if (a==c) then
                        val = val - H2B_oovo(j,k,b,i)
                    end if
                    if (i==k) then
                        val = val + H2B_ovvv(j,a,b,c)
                    end if
                    if (i==k .and. a==c) then
                        val = val + H1A_ov(j,b)
                    end if

                end function calc_SBDB_matel

                function calc_SBDC_matel(i,a,j,k,b,c,H1B_ov,H2C_ooov,H2C_vovv) result(val)

                    integer, intent(in) :: i, a, j, k, b, c
                    real(kind=8), intent(in) :: H1B_ov(:,:), H2C_ooov(:,:,:,:),&
                    H2C_vovv(:,:,:,:)
                
                    real(kind=8) :: val

                    val = 0.0d0
                    if (i==k .and. a==c) then
                        val = val + H1B_ov(j,b)
                    end if
                    if (a==b .and. i==j) then
                        val = val + H1B_ov(k,c)
                    end if
                    if (i==j .and. a==c) then
                        val = val - H1B_ov(k,b)
                    end if
                    if (i==k .and. a==b) then
                        val = val - H1B_ov(j,c)
                    end if
                    if (a==b) then
                        val = val - H2C_ooov(j,k,i,c)
                    end if
                    if (a==c) then
                        val = val - H2C_ooov(k,j,i,b)
                    end if
                    if (i==j) then
                        val = val + H2C_vovv(a,k,b,c)
                    end if
                    if (i==k) then
                        val = val + H2C_vovv(a,j,c,b)
                    end if

                end function calc_SBDC_matel

                function calc_DASA_matel(i,j,a,b,k,c,H2A_vooo,H2A_vvov) result(val)

                        integer, intent(in) :: i, j, a, b, k, c
                        real(kind=8), intent(in) :: H2A_vooo(:,:,:,:), H2A_vvov(:,:,:,:)

                        real(kind=8) :: val

                        val = 0.0d0
                        if (a==c) then
                            val = val - H2A_vooo(b,k,j,i)
                        end if
                        if (b==c) then
                            val = val - H2A_vooo(a,k,i,j)
                        end if
                        if (i==k) then
                            val = val + H2A_vvov(b,a,j,c)
                        end if
                        if (j==k) then
                            val = val + H2A_vvov(a,b,i,c)
                        end if

                end function calc_DASA_matel

                function calc_DBSA_matel(i,j,a,b,k,c,H2B_ovoo,H2B_vvvo) result(val)

                        integer, intent(in) :: i, j, a, b, k, c
                        real(kind=8), intent(in) :: H2B_ovoo(:,:,:,:), H2B_vvvo(:,:,:,:)

                        real(kind=8) :: val

                        val = 0.0d0
                        if (a==c) then
                            val = val - H2B_ovoo(k,b,i,j)
                        end if
                        if (i==k) then
                            val = val + H2B_vvvo(a,b,c,j)
                        end if

                end function calc_DBSA_matel

                function calc_DBSB_matel(i,j,a,b,k,c,H2B_vooo,H2B_vvov) result(val)

                        integer, intent(in) :: i, j, a, b, k, c
                        real(kind=8), intent(in) :: H2B_vooo(:,:,:,:), H2B_vvov(:,:,:,:)

                        real(kind=8) :: val

                        val = 0.0d0
                        if (b==c) then
                            val = val - H2B_vooo(a,k,i,j)
                        end if
                        if (j==k) then
                            val = val + H2B_vvov(a,b,i,c)
                        end if

                end function calc_DBSB_matel

                function calc_DCSB_matel(i,j,a,b,k,c,H2C_vooo,H2C_vvov) result(val)

                        integer, intent(in) :: i, j, a, b, k, c
                        real(kind=8), intent(in) :: H2C_vooo(:,:,:,:), H2C_vvov(:,:,:,:)

                        real(kind=8) :: val

                        val = 0.0d0
                        if (a==c) then
                            val = val - H2C_vooo(b,k,j,i)
                        end if
                        if (b==c) then
                            val = val - H2C_vooo(a,k,i,j)
                        end if
                        if (i==k) then
                            val = val + H2C_vvov(b,a,j,c)
                        end if
                        if (j==k) then
                            val = val + H2C_vvov(a,b,i,c)
                        end if

                end function calc_DCSB_matel

                function calc_DADA_matel(i,j,a,b,k,l,c,d,H1A_oo,H1A_vv,H2A_voov,&
                                H2A_oooo,H2A_vvvv) result(val)

                        integer, intent(in) :: i, j, a, b, k, l, c, d
                        real(kind=8), intent(in) :: H1A_oo(:,:), H1A_vv(:,:),&
                                H2A_voov(:,:,:,:), H2A_oooo(:,:,:,:),&
                                H2A_vvvv(:,:,:,:)

                        real(kind=8) :: val
                    
                        val = 0.0d0
                        if (a==c .and. b==d) then
                            if (j==l) then
                                val = val - H1A_oo(k,i)
                            end if
                            if (i==l) then
                                val = val + H1A_oo(k,j) ! correct
                                !val = val - H1A_oo(k,j) ! previous
                            end if
                            if (j==k) then
                                val = val + H1A_oo(l,i) ! correct
                                !val = val - H1A_oo(l,i) ! previous
                            end if
                            if (i==k) then
                                val = val - H1A_oo(l,j)
                            end if
                        end if
                        if (j==l .and. i==k) then
                            if (b==d) then
                                val = val + H1A_vv(a,c)
                            end if 
                            if (b==c) then
                                val = val - H1A_vv(a,d) ! previous
                                !val = val + H1A_vv(a,d)
                            end if
                            if (a==c) then
                                val = val + H1A_vv(b,d)
                            end if
                            if (a==d) then
                                val = val - H1A_vv(b,c) ! correct
                                !val = val + H1A_vv(b,c) ! previous
                            end if
                        end if
                        if (i==k) then
                            if (a==c) then
                                val = val + H2A_voov(b,l,j,d)
                            end if
                            if (a==d) then
                                val = val - H2A_voov(b,l,j,c)
                            end if
                            if (b==c) then
                                val = val - H2A_voov(a,l,j,d)
                            end if
                            if (b==d) then
                                val = val + H2A_voov(a,l,j,c)
                            end if
                        end if
                        if (i==l) then
                            if (a==c) then
                                val = val - H2A_voov(b,k,j,d)
                            end if
                            if (a==d) then
                                val = val + H2A_voov(b,k,j,c)
                            end if
                            if (b==c) then
                                val = val + H2A_voov(a,k,j,d)
                            end if
                            if (b==d) then
                                val = val - H2A_voov(a,k,j,c)
                            end if
                        end if
                        if (j==k) then
                            if (a==c) then
                                val = val - H2A_voov(b,l,i,d)
                            end if
                            if (a==d) then
                                val = val + H2A_voov(b,l,i,c)
                            end if
                            if (b==c) then
                                val = val + H2A_voov(a,l,i,d)
                            end if
                            if (b==d) then
                                val = val - H2A_voov(a,l,i,c)
                            end if
                        end if
                        if (j==l) then
                            if (a==c) then
                                val = val + H2A_voov(b,k,i,d)
                            end if
                            if (a==d) then
                                val = val - H2A_voov(b,k,i,c)
                            end if
                            if (b==c) then
                                val = val - H2A_voov(a,k,i,d)
                            end if
                            if (b==d) then
                                val = val + H2A_voov(a,k,i,c)
                            end if
                        end if
                        if (b==d .and. a==c) then
                            val = val + H2A_oooo(k,l,i,j)
                        end if
                        if (i==k .and. j==l) then
                            val = val + H2A_vvvv(a,b,c,d)
                        end if

                end function calc_DADA_matel

                function calc_DADB_matel(i,j,a,b,k,l,c,d,H2B_voov) result(val)

                        integer, intent(in) :: i, j, a, b, k, l, c, d
                        real(kind=8), intent(in) :: H2B_voov(:,:,:,:)

                        real(kind=8) :: val

                        val = 0.0d0
                        if (i==k) then
                            if (a==c) then
                                val = val + H2B_voov(b,l,j,d)
                            end if
                            if (b==c) then
                                val = val - H2B_voov(a,l,j,d)
                            end if
                        end if
                        if (j==k) then
                            if (a==c) then
                                val = val - H2B_voov(b,l,i,d)
                            end if
                            if (b==c) then
                                val = val + H2B_voov(a,l,i,d)
                            end if
                        end if

                end function calc_DADB_matel
                            
                function calc_DBDA_matel(i,j,a,b,k,l,c,d,H2B_ovvo) result(val)
                        
                        integer, intent(in) :: i, j, a, b, k, l, c, d
                        real(kind=8), intent(in) :: H2B_ovvo(:,:,:,:)

                        real(kind=8) :: val

                        val = 0.0d0
                        if (i==k) then
                            if (a==c) then
                                val = val + H2B_ovvo(l,b,d,j)
                            end if
                            if (a==d) then
                                val = val - H2B_ovvo(l,b,c,j)
                            end if
                        end if
                        if (i==l) then
                            if (a==c) then
                                val = val - H2B_ovvo(k,b,d,j)
                            end if
                            if (a==d) then
                                val = val + H2B_ovvo(k,b,c,j)
                            end if
                        end if

                end function calc_DBDA_matel

                function calc_DBDB_matel(i,j,a,b,k,l,c,d,&
                                H1A_oo,H1A_vv,H1B_oo,H1B_vv,&
                                H2A_voov,&
                                H2B_oooo,H2B_vvvv,H2B_ovov,H2B_vovo,&
                                H2C_voov) result(val)

                        integer, intent(in) :: i, j, a, b, k, l, c, d
                        real(kind=8), intent(in) :: H1A_oo(:,:), H1A_vv(:,:),&
                                H1B_oo(:,:), H1B_vv(:,:), H2A_voov(:,:,:,:),&
                                H2B_vvvv(:,:,:,:), H2B_oooo(:,:,:,:),&
                                H2B_ovov(:,:,:,:), H2B_vovo(:,:,:,:),&
                                H2C_voov(:,:,:,:)

                        real(kind=8) :: val

                        val = 0.0d0
                        if (j==l) then
                            if (b==d) then
                                val = val + H2A_voov(a,k,i,c)
                            end if 
                            if (i==k) then
                                val = val + H2B_vvvv(a,b,c,d)
                            end if
                        end if
                        if (a==c) then
                            if (i==k) then
                                val = val + H2C_voov(b,l,j,d)
                            end if
                            if (b==d) then
                                val = val + H2B_oooo(k,l,i,j)
                            end if
                        end if
                        if (j==l .and. a==c) then
                            val = val - H2B_ovov(k,b,i,d)
                        end if
                        if (i==k .and. b==d) then
                            val = val - H2B_vovo(a,l,c,j)
                        end if
                        if (j==l .and. a==c .and. b==d) then
                            val = val - H1A_oo(k,i)
                        end if
                        if (a==c .and. b==d .and. i==k) then
                            val = val - H1B_oo(l,j)
                        end if
                        if (i==k .and. b==d .and. j==l) then
                            val = val + H1A_vv(a,c)
                        end if
                        if (j==l .and. i==k .and. a==c) then
                            val = val + H1B_vv(b,d)
                        end if 

                end function calc_DBDB_matel

                function calc_DBDC_matel(i,j,a,b,k,l,c,d,H2B_voov) result(val)

                        integer, intent(in) :: i, j, a, b, k, l, c, d
                        real(kind=8), intent(in) :: H2B_voov(:,:,:,:)

                        real(kind=8) :: val

                        val = 0.0d0
                        if (j==k) then
                            if (b==c) then
                                val = val + H2B_voov(a,l,i,d)
                            end if
                            if (b==d) then
                                val = val - H2B_voov(a,l,i,c)
                            end if
                        end if
                        if (j==l) then
                            if (b==c) then
                                val = val - H2B_voov(a,k,i,d)
                            end if
                            if (b==d) then
                                val = val + H2B_voov(a,k,i,c)
                            end if
                        end if

                end function calc_DBDC_matel

                function calc_DCDB_matel(i,j,a,b,k,l,c,d,H2B_ovvo) result(val)

                        integer, intent(in) :: i, j, a, b, k, l, c, d
                        real(kind=8), intent(in) :: H2B_ovvo(:,:,:,:)

                        real(kind=8) :: val

                        val = 0.0d0
                        if (i==l) then
                            if (a==d) then
                                val = val + H2B_ovvo(k,b,c,j)
                            end if
                            if (b==d) then
                                val = val - H2B_ovvo(k,a,c,j)
                            end if
                        end if
                        if (j==l) then
                            if (a==d) then
                                val = val - H2B_ovvo(k,b,c,i)
                            end if
                            if (b==d) then
                                val = val + H2B_ovvo(k,a,c,i)
                            end if
                        end if

                end function calc_DCDB_matel

                function calc_DCDC_matel(i,j,a,b,k,l,c,d,H1B_oo,H1B_vv,H2C_voov,&
                                H2C_oooo,H2C_vvvv) result(val)

                        integer, intent(in) :: i, j, a, b, k, l, c, d
                        real(kind=8), intent(in) :: H1B_oo(:,:), H1B_vv(:,:),&
                                H2C_voov(:,:,:,:), H2C_oooo(:,:,:,:),&
                                H2C_vvvv(:,:,:,:)

                        real(kind=8) :: val
                    
                        val = 0.0d0
                        if (a==c .and. b==d) then
                            if (j==l) then
                                val = val - H1B_oo(k,i)
                            end if
                            if (i==l) then
                                val = val + H1B_oo(k,j) ! correct
                                !val = val - H1B_oo(k,j) ! previous
                            end if
                            if (j==k) then
                                val = val + H1B_oo(l,i) ! correct
                                !val = val - H1B_oo(l,i) ! previous
                            end if
                            if (i==k) then
                                val = val - H1B_oo(l,j)
                            end if
                        end if
                        if (j==l .and. i==k) then
                            if (b==d) then
                                val = val + H1B_vv(a,c)
                            end if 
                            if (b==c) then
                                val = val - H1B_vv(a,d) ! correct
                                !val = val + H1B_vv(a,d) ! previous
                            end if
                            if (a==c) then
                                val = val + H1B_vv(b,d)
                            end if
                            if (a==d) then
                                val = val - H1B_vv(b,c) ! correct
                                !val = val + H1B_vv(b,c) ! previous
                            end if
                        end if
                        if (i==k) then
                            if (a==c) then
                                val = val + H2C_voov(b,l,j,d)
                            end if
                            if (a==d) then
                                val = val - H2C_voov(b,l,j,c)
                            end if
                            if (b==c) then
                                val = val - H2C_voov(a,l,j,d)
                            end if
                            if (b==d) then
                                val = val + H2C_voov(a,l,j,c)
                            end if
                        end if
                        if (i==l) then
                            if (a==c) then
                                val = val - H2C_voov(b,k,j,d)
                            end if
                            if (a==d) then
                                val = val + H2C_voov(b,k,j,c)
                            end if
                            if (b==c) then
                                val = val + H2C_voov(a,k,j,d)
                            end if
                            if (b==d) then
                                val = val - H2C_voov(a,k,j,c)
                            end if
                        end if
                        if (j==k) then
                            if (a==c) then
                                val = val - H2C_voov(b,l,i,d)
                            end if
                            if (a==d) then
                                val = val + H2C_voov(b,l,i,c)
                            end if
                            if (b==c) then
                                val = val + H2C_voov(a,l,i,d)
                            end if
                            if (b==d) then
                                val = val - H2C_voov(a,l,i,c)
                            end if
                        end if
                        if (j==l) then
                            if (a==c) then
                                val = val + H2C_voov(b,k,i,d)
                            end if
                            if (a==d) then
                                val = val - H2C_voov(b,k,i,c)
                            end if
                            if (b==c) then
                                val = val - H2C_voov(a,k,i,d)
                            end if
                            if (b==d) then
                                val = val + H2C_voov(a,k,i,c)
                            end if
                        end if
                        if (b==d .and. a==c) then
                            val = val + H2C_oooo(k,l,i,j)
                        end if
                        if (i==k .and. j==l) then
                            val = val + H2C_vvvv(a,b,c,d)
                        end if

                end function calc_DCDC_matel


end module eomcc_initial_guess                         
