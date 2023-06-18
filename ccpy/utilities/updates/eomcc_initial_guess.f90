module eomcc_initial_guess

        implicit none

        contains

                subroutine get_active_dimensions(idx1A,idx1B,idx2A,idx2B,idx2C,&
                                                 n1a_act,n1b_act,n2a_act,n2b_act,n2c_act,noact,nuact,&
                                                 noa,nua,nob,nub)

                        integer, intent(in) :: noa, nua, nob, nub, noact, nuact
                        integer, intent(out) :: idx1A(nua,noa), idx1B(nub,nob),&
                                                idx2A(nua,nua,noa,noa), idx2B(nua,nub,noa,nob), idx2C(nub,nub,nob,nob),&
                                                n1a_act, n1b_act, n2a_act, n2b_act, n2c_act

                        integer :: a, b, i, j, act_rng_oa(2), act_rng_ua(2), act_rng_ob(2), act_rng_ub(2),&
                                   num_act_holes, num_act_particles
                        integer :: num_act

                        !num_act = 1 ! |iJAb>
                        num_act = 2 ! |IJAB>

                        act_rng_oa(1) = max(0, noa-noact)
                        act_rng_oa(2) = noa
                        act_rng_ua(1) = 0
                        act_rng_ua(2) = min(nua, nuact)
                        act_rng_ob(1) = max(0, nob-noact)
                        act_rng_ob(2) = nob
                        act_rng_ub(1) = 0
                        act_rng_ub(2) = min(nub, nuact)

                        idx1A = 0
                        n1a_act = 0
                        do i = 1,noa
                           do a = 1,nua
                                 idx1A(a,i) = 1
                                 n1a_act = n1a_act + 1
                           end do
                        end do

                        idx1B = 0
                        n1b_act = 0
                        do i = 1,nob
                           do a = 1,nub
                                 idx1B(a,i) = 1
                                 n1b_act = n1b_act + 1
                           end do
                        end do

                        idx2A = 0
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
                            if (num_act_holes >= num_act .and. num_act_particles >= num_act) then
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
                            if (num_act_holes >= num_act .and. num_act_particles >= num_act) then
                                idx2B(a,b,i,j) = 1
                                n2b_act = n2b_act + 1
                            end if
                        end do
                        end do
                        end do
                        end do

                        idx2C = 0
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
                            if (num_act_holes >= num_act .and. num_act_particles >= num_act) then
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
                                n1a_act,n1b_act,n2a_act,n2b_act,n2c_act,ndim_act,&
                                noa,nua,nob,nub)

                        integer, intent(in) :: noa, nua, nob, nub, n1a_act, n1b_act,&
                                n2a_act, n2b_act, n2c_act, ndim_act,&
                                idx1A(nua,noa), idx1B(nub,nob),&
                                idx2A(nua,nua,noa,noa), idx2B(nua,nub,noa,nob), idx2C(nub,nub,nob,nob)
                        real(kind=8), intent(in) :: CIvec(ndim_act)

                        real(kind=8), intent(out) :: r1a(nua,noa), r1b(nub,nob),&
                        r2a(nua,nua,noa,noa), r2b(nua,nub,noa,nob), r2c(nub,nub,nob,nob)

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
                                    n1a_act,n1b_act,n2a_act,n2b_act,n2c_act,ndim_act,noa,nua,nob,nub)

                        integer, intent(in) :: noa, nua, nob, nub, n1a_act, n1b_act, n2a_act, n2b_act, n2c_act, ndim_act,&
                                               idx1A(nua,noa), idx1B(nub,nob),&
                                               idx2A(nua,nua,noa,noa), idx2B(nua,nub,noa,nob), idx2C(nub,nub,nob,nob)
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
                        integer :: i, j, k, l, a, b, c, d, ct1, ct2, pos(6), info

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
                            if (idx1A(a,i)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do j = 1 , noa
                            do b = 1 , nua
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
                            if (idx1B(a,i)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do j = 1 , noa
                            do b = 1 , nua
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
                            if (idx1A(a,i)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do j = 1 , nob
                            do b = 1 , nub
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
                            if (idx1B(a,i)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do j = 1 , nob
                            do b = 1 , nub
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
                            if (idx1A(a,i)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do j = 1, noa
                            do k = j+1, noa
                            do b = 1, nua
                            do c = b+1, nua
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
                            if (idx1A(a,i)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do j = 1, noa
                            do k = 1, nob
                            do b = 1, nua
                            do c = 1, nub
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
                            if (idx1B(a,i)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do j = 1, noa
                            do k = 1, nob
                            do b = 1, nua
                            do c = 1, nub
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
                            if (idx1B(a,i)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do j = 1, nob
                            do k = j+1, nob
                            do b = 1, nub
                            do c = b+1, nub
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
                            if (idx2A(a,b,i,j)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, noa
                            do c = 1, nua
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
                            if (idx2B(a,b,i,j)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, noa
                            do c = 1, nua
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
                            if (idx2B(a,b,i,j)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, nob
                            do c = 1, nub
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
                            if (idx2C(a,b,i,j)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, nob
                            do c = 1, nub
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
                            if (idx2A(a,b,i,j)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, noa
                            do l = k+1, noa
                            do c = 1, nua
                            do d = c+1, nua
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
                            if (idx2A(a,b,i,j)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, noa
                            do l = 1, nob
                            do c = 1, nua
                            do d = 1, nub
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
                            if (idx2B(a,b,i,j)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, noa
                            do l = k+1, noa
                            do c = 1, nua
                            do d = c+1, nua
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
                            if (idx2B(a,b,i,j)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, noa
                            do l = 1, nob
                            do c = 1, nua
                            do d = 1, nub
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
                            if (idx2B(a,b,i,j)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, nob
                            do l = k+1, nob
                            do c = 1, nub
                            do d = c+1, nub
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
                            if (idx2C(a,b,i,j)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, noa
                            do l = 1, nob
                            do c = 1, nua
                            do d = 1, nub
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
                            if (idx2C(a,b,i,j)==0) cycle
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, nob
                            do l = k+1, nob
                            do c = 1, nub
                            do d = c+1, nub
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
                    if (i==j) val = val + h1a_vv(a,b)
                    if (a==b) val = val - h1a_oo(j,i)
                    
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
                    if (i==j) val = val + h1b_vv(a,b)
                    if (a==b) val = val - h1b_oo(j,i)
                    
                end function calc_SBSB_matel

                function calc_SADA_matel(i,a,j,k,b,c,H1A_ov,H2A_ooov,H2A_vovv) result(val)

                    integer, intent(in) :: i, a, j, k, b, c
                    real(kind=8), intent(in) :: H1A_ov(:,:), H2A_ooov(:,:,:,:),&
                    H2A_vovv(:,:,:,:)
                
                    real(kind=8) :: val

                    val = 0.0d0
                    !!! A(jk)A(bc) h1a(kc) d(ij) d(ab)
                    if (i==j .and. a==b) val = val + h1a_ov(k,c) ! (1)
                    if (i==k .and. a==b) val = val - h1a_ov(j,c) ! (jk)
                    if (i==j .and. a==c) val = val - h1a_ov(k,b) ! (bc)
                    if (i==k .and. a==c) val = val + h1a_ov(j,b) ! (jk)(bc)
                    !!! A(bc) -h2a(jkic) d(ab)
                    if (a==b) val = val - h2a_ooov(j,k,i,c) ! (1)
                    if (a==c) val = val + h2a_ooov(j,k,i,b) ! (bc)
                    !!! A(jk) h2a(akbc) d(ij)
                    if (i==j) val = val + h2a_vovv(a,k,b,c) ! (1)
                    if (i==k) val = val - h2a_vovv(a,j,b,c) ! (jk)

                end function calc_SADA_matel

                function calc_SADB_matel(i,a,j,k,b,c,H1B_ov,H2B_ooov,H2B_vovv) result(val)

                    integer, intent(in) :: i, a, j, k, b, c
                    real(kind=8), intent(in) :: H1B_ov(:,:),H2B_ooov(:,:,:,:),&
                    H2B_vovv(:,:,:,:)

                    real(kind=8) :: val

                    val = 0.0d0
                    !!! h1b(k,c) d(ij) d(ab)
                    if (i==j .and. a==b) val = val + h1b_ov(k,c)
                    !!! -h2b(jkic) d(ab)
                    if (a==b) val = val - h2b_ooov(j,k,i,c)
                    !!! h2b(akbc) d(ij)
                    if (i==j) val = val + h2b_vovv(a,k,b,c)

                end function calc_SADB_matel

                function calc_SBDB_matel(i,a,j,k,b,c,H1A_ov,H2B_oovo,H2B_ovvv) result(val)

                    integer, intent(in) :: i, a, j, k, b, c
                    real(kind=8), intent(in) :: H1A_ov(:,:), H2B_oovo(:,:,:,:),&
                    H2B_ovvv(:,:,:,:)

                    real(kind=8) :: val

                    val = 0.0d0
                    !!! h1a(j,b) d(ik) d(ac)
                    if (i==k .and. a==c) val = val + h1a_ov(j,b)
                    !!! -h2b(jkbi) d(ac)
                    if (a==c) val = val - h2b_oovo(j,k,b,i)
                    !!! h2b(jabc) d(ik)
                    if (i==k) val = val + h2b_ovvv(j,a,b,c)

                end function calc_SBDB_matel

                function calc_SBDC_matel(i,a,j,k,b,c,H1B_ov,H2C_ooov,H2C_vovv) result(val)

                    integer, intent(in) :: i, a, j, k, b, c
                    real(kind=8), intent(in) :: H1B_ov(:,:), H2C_ooov(:,:,:,:),&
                    H2C_vovv(:,:,:,:)
                
                    real(kind=8) :: val

                    val = 0.0d0
                    !!! A(jk)A(bc) h1a(kc) d(ij) d(ab)
                    if (i==j .and. a==b) val = val + h1b_ov(k,c) ! (1)
                    if (i==k .and. a==b) val = val - h1b_ov(j,c) ! (jk)
                    if (i==j .and. a==c) val = val - h1b_ov(k,b) ! (bc)
                    if (i==k .and. a==c) val = val + h1b_ov(j,b) ! (jk)(bc)
                    !!! A(bc) -h2a(jkic) d(ab)
                    if (a==b) val = val - h2c_ooov(j,k,i,c) ! (1)
                    if (a==c) val = val + h2c_ooov(j,k,i,b) ! (bc)
                    !!! A(jk) h2a(akbc) d(ij)
                    if (i==j) val = val + h2c_vovv(a,k,b,c) ! (1)
                    if (i==k) val = val - h2c_vovv(a,j,b,c) ! (jk)

                end function calc_SBDC_matel

                function calc_DASA_matel(i,j,a,b,k,c,H2A_vooo,H2A_vvov) result(val)

                        integer, intent(in) :: i, j, a, b, k, c
                        real(kind=8), intent(in) :: H2A_vooo(:,:,:,:), H2A_vvov(:,:,:,:)

                        real(kind=8) :: val

                        val = 0.0d0
                        !!! A(ab) -h2a(akij) d(bc)
                        if (b==c) val = val - h2a_vooo(a,k,i,j) ! (1)
                        if (a==c) val = val + h2a_vooo(b,k,i,j) ! (ab)
                        !!! A(ij) h2a(abic) d(kj)
                        if (j==k) val = val + h2a_vvov(a,b,i,c) ! (1)
                        if (i==k) val = val - h2a_vvov(a,b,j,c) ! (ij)

                end function calc_DASA_matel

                function calc_DBSA_matel(i,j,a,b,k,c,H2B_ovoo,H2B_vvvo) result(val)

                        integer, intent(in) :: i, j, a, b, k, c
                        real(kind=8), intent(in) :: H2B_ovoo(:,:,:,:), H2B_vvvo(:,:,:,:)

                        real(kind=8) :: val

                        val = 0.0d0
                        !!! -h2b(kbij) d(ac)
                        if (a==c) val = val - h2b_ovoo(k,b,i,j)
                        !!! h2b(abcj) d(ik)
                        if (i==k) val = val + h2b_vvvo(a,b,c,j)

                end function calc_DBSA_matel

                function calc_DBSB_matel(i,j,a,b,k,c,H2B_vooo,H2B_vvov) result(val)

                        integer, intent(in) :: i, j, a, b, k, c
                        real(kind=8), intent(in) :: H2B_vooo(:,:,:,:), H2B_vvov(:,:,:,:)

                        real(kind=8) :: val

                        val = 0.0d0
                        !!! -h2b(akij) d(bc)
                        if (b==c) val = val - h2b_vooo(a,k,i,j)
                        !!! h2b(abic) d(jk)
                        if (j==k) val = val + h2b_vvov(a,b,i,c)

                end function calc_DBSB_matel

                function calc_DCSB_matel(i,j,a,b,k,c,H2C_vooo,H2C_vvov) result(val)

                        integer, intent(in) :: i, j, a, b, k, c
                        real(kind=8), intent(in) :: H2C_vooo(:,:,:,:), H2C_vvov(:,:,:,:)

                        real(kind=8) :: val

                        val = 0.0d0
                        !!! A(ab) -h2a(akij) d(bc)
                        if (b==c) val = val - h2c_vooo(a,k,i,j) ! (1)
                        if (a==c) val = val + h2c_vooo(b,k,i,j) ! (ab)
                        !!! A(ij) h2a(abic) d(kj)
                        if (j==k) val = val + h2c_vvov(a,b,i,c) ! (1)
                        if (i==k) val = val - h2c_vvov(a,b,j,c) ! (ij)

                end function calc_DCSB_matel

                function calc_DADA_matel(i,j,a,b,k,l,c,d,H1A_oo,H1A_vv,H2A_voov,&
                                H2A_oooo,H2A_vvvv) result(val)

                        integer, intent(in) :: i, j, a, b, k, l, c, d
                        real(kind=8), intent(in) :: H1A_oo(:,:), H1A_vv(:,:),&
                                H2A_voov(:,:,:,:), H2A_oooo(:,:,:,:),&
                                H2A_vvvv(:,:,:,:)

                        real(kind=8) :: val
                    
                        val = 0.0d0
                        !!! A(ij)A(kl) -h1a(ki) d(ac)d(bd)d(jl)
                        if (a==c .and. b==d) then
                            if (j==l) val = val - h1a_oo(k,i) ! (1)
                            if (i==l) val = val + h1a_oo(k,j) ! (ij)
                            if (j==k) val = val + h1a_oo(l,i) ! (kl)
                            if (i==k) val = val - h1a_oo(l,j) ! (ij)(kl)
                        end if
                        !!! A(ab)A(cd) h1a(ac) d(ik)d(jl)d(bd)
                        if (j==l .and. i==k) then
                            if (b==d) val = val + h1a_vv(a,c) ! (1)
                            if (a==d) val = val - h1a_vv(b,c) ! (ab)
                            if (b==c) val = val - h1a_vv(a,d) ! (cd)
                            if (a==c) val = val + h1a_vv(b,d) ! (ab)(cd)
                        end if
                        !!! A(ij)A(kl)A(ab)A(cd) h2a(akic) d(bd) d(jl)
                        ! (1)
                        if (b==d .and. j==l) val = val + h2a_voov(a,k,i,c) ! (1)
                        if (b==d .and. i==l) val = val - h2a_voov(a,k,j,c) ! (ij)
                        if (a==d .and. j==l) val = val - h2a_voov(b,k,i,c) ! (ab)
                        if (a==d .and. i==l) val = val + h2a_voov(b,k,j,c) ! (ij)(ab)
                        ! (kl)
                        if (b==d .and. j==k) val = val - h2a_voov(a,l,i,c) ! (1)
                        if (b==d .and. i==k) val = val + h2a_voov(a,l,j,c) ! (ij)
                        if (a==d .and. j==k) val = val + h2a_voov(b,l,i,c) ! (ab)
                        if (a==d .and. i==k) val = val - h2a_voov(b,l,j,c) ! (ij)(ab)
                        ! (cd)
                        if (b==c .and. j==l) val = val - h2a_voov(a,k,i,d) ! (1)
                        if (b==c .and. i==l) val = val + h2a_voov(a,k,j,d) ! (ij)
                        if (a==c .and. j==l) val = val + h2a_voov(b,k,i,d) ! (ab)
                        if (a==c .and. i==l) val = val - h2a_voov(b,k,j,d) ! (ij)(ab)
                        ! (kl)(cd)
                        if (b==c .and. j==k) val = val + h2a_voov(a,l,i,d) ! (1)
                        if (b==c .and. i==k) val = val - h2a_voov(a,l,j,d) ! (ij)
                        if (a==c .and. j==k) val = val - h2a_voov(b,l,i,d) ! (ab)
                        if (a==c .and. i==k) val = val + h2a_voov(b,l,j,d) ! (ij)(ab)
                        !!! h2a(klij) d(bd) d(ac)
                        if (b==d .and. a==c) val = val + h2a_oooo(k,l,i,j)
                        !!! h2a(abcd) d(ik) d(jl)
                        if (i==k .and. j==l) val = val + h2a_vvvv(a,b,c,d)

                end function calc_DADA_matel

                function calc_DADB_matel(i,j,a,b,k,l,c,d,H2B_voov) result(val)

                        integer, intent(in) :: i, j, a, b, k, l, c, d
                        real(kind=8), intent(in) :: H2B_voov(:,:,:,:)

                        real(kind=8) :: val

                        val = 0.0d0
                        !!! A(ij)A(ab) h2b(alid) d(jk) d(bc)
                        if (j==k .and. b==c) val = val + h2b_voov(a,l,i,d) ! (1)
                        if (i==k .and. b==c) val = val - h2b_voov(a,l,j,d) ! (ij)
                        if (j==k .and. a==c) val = val - h2b_voov(b,l,i,d) ! (ab)
                        if (i==k .and. a==c) val = val + h2b_voov(b,l,j,d) ! (ij)(ab)

                end function calc_DADB_matel
                            
                function calc_DBDA_matel(i,j,a,b,k,l,c,d,H2B_ovvo) result(val)
                        
                        integer, intent(in) :: i, j, a, b, k, l, c, d
                        real(kind=8), intent(in) :: H2B_ovvo(:,:,:,:)

                        real(kind=8) :: val

                        val = 0.0d0
                        !!! A(kl)A(cd) h2b(kbcj) d(il) d(ad)
                        if (i==l .and. a==d) val = val + h2b_ovvo(k,b,c,j) ! (1)
                        if (i==k .and. a==d) val = val - h2b_ovvo(l,b,c,j) ! (kl)
                        if (i==l .and. a==c) val = val - h2b_ovvo(k,b,d,j) ! (cd)
                        if (i==k .and. a==c) val = val + h2b_ovvo(l,b,d,j) ! (kl)(cd)

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
                        !!! -h1a(ki) d(ac) d(bd) d(jl)
                        if (a==c .and. b==d .and. j==l) val = val - h1a_oo(k,i)
                        !!! -h1b(lj) d(ac) d(bd) d(ik)
                        if (a==c .and. b==d .and. i==k) val = val - h1b_oo(l,j)
                        !!! h1a(ac) d(ik) d(jl) d(bd)
                        if (i==k .and. j==l .and. b==d) val = val + h1a_vv(a,c)
                        !!! h1b(bd) d(ik) d(jl) d(ac)
                        if (i==k .and. j==l .and. a==c) val = val + h1b_vv(b,d)
                        !!! h2a(akic) d(jl) d(bd)
                        if (j==l .and. b==d) val = val + h2a_voov(a,k,i,c)
                        !!! h2c(bljd) d(ik) d(ac)
                        if (i==k .and. a==c) val = val + h2c_voov(b,l,j,d)
                        !!! -h2b(kbid) d(jl) d(ac)
                        if (j==l .and. a==c) val = val - h2b_ovov(k,b,i,d)
                        !!! -h2b(alcj) d(ik) d(bd)
                        if (i==k .and. b==d) val = val - h2b_vovo(a,l,c,j)
                        !!! h2b(klij) d(ac) d(bd)
                        if (a==c .and. b==d) val = val + h2b_oooo(k,l,i,j)
                        !!! h2b(abcd) d(ik) d(jl)
                        if (i==k .and. j==l) val = val + h2b_vvvv(a,b,c,d)

                end function calc_DBDB_matel

                function calc_DBDC_matel(i,j,a,b,k,l,c,d,H2B_voov) result(val)

                        integer, intent(in) :: i, j, a, b, k, l, c, d
                        real(kind=8), intent(in) :: H2B_voov(:,:,:,:)

                        real(kind=8) :: val

                        val = 0.0d0
                        !!! A(kl)A(cd) h2b(akic) d(bd) d(jl)
                        if (b==d .and. j==l) val = val + h2b_voov(a,k,i,c) ! (1)
                        if (b==d .and. j==k) val = val - h2b_voov(a,l,i,c) ! (kl)
                        if (b==c .and. j==l) val = val - h2b_voov(a,k,i,d) ! (cd)
                        if (b==c .and. j==k) val = val + h2b_voov(a,l,i,d) ! (kl)(cd)

                end function calc_DBDC_matel

                function calc_DCDB_matel(i,j,a,b,k,l,c,d,H2B_ovvo) result(val)

                        integer, intent(in) :: i, j, a, b, k, l, c, d
                        real(kind=8), intent(in) :: H2B_ovvo(:,:,:,:)

                        real(kind=8) :: val

                        val = 0.0d0
                        !!! A(ij)A(ab) h2b(kaci) d(jl) d(bd)
                        if (j==l .and. b==d) val = val + h2b_ovvo(k,a,c,i) ! (1)
                        if (i==l .and. b==d) val = val - h2b_ovvo(k,a,c,j) ! (ij)
                        if (j==l .and. a==d) val = val - h2b_ovvo(k,b,c,i) ! (ab)
                        if (i==l .and. a==d) val = val + h2b_ovvo(k,b,c,j) ! (ij)(ab)

                end function calc_DCDB_matel

                function calc_DCDC_matel(i,j,a,b,k,l,c,d,H1B_oo,H1B_vv,H2C_voov,&
                                H2C_oooo,H2C_vvvv) result(val)

                        integer, intent(in) :: i, j, a, b, k, l, c, d
                        real(kind=8), intent(in) :: H1B_oo(:,:), H1B_vv(:,:),&
                                H2C_voov(:,:,:,:), H2C_oooo(:,:,:,:),&
                                H2C_vvvv(:,:,:,:)

                        real(kind=8) :: val
                    
                        val = 0.0d0
                        !!! A(ij)A(kl) -h1a(ki) d(ac)d(bd)d(jl)
                        if (a==c .and. b==d) then
                            if (j==l) val = val - h1b_oo(k,i) ! (1)
                            if (i==l) val = val + h1b_oo(k,j) ! (ij)
                            if (j==k) val = val + h1b_oo(l,i) ! (kl)
                            if (i==k) val = val - h1b_oo(l,j) ! (ij)(kl)
                        end if
                        !!! A(ab)A(cd) h1a(ac) d(ik)d(jl)d(bd)
                        if (j==l .and. i==k) then
                            if (b==d) val = val + h1b_vv(a,c) ! (1)
                            if (a==d) val = val - h1b_vv(b,c) ! (ab)
                            if (b==c) val = val - h1b_vv(a,d) ! (cd)
                            if (a==c) val = val + h1b_vv(b,d) ! (ab)(cd)
                        end if
                        !!! A(ij)A(kl)A(ab)A(cd) h2a(akic) d(bd) d(jl)
                        ! (1)
                        if (b==d .and. j==l) val = val + h2c_voov(a,k,i,c) ! (1)
                        if (b==d .and. i==l) val = val - h2c_voov(a,k,j,c) ! (ij)
                        if (a==d .and. j==l) val = val - h2c_voov(b,k,i,c) ! (ab)
                        if (a==d .and. i==l) val = val + h2c_voov(b,k,j,c) ! (ij)(ab)
                        ! (kl)
                        if (b==d .and. j==k) val = val - h2c_voov(a,l,i,c) ! (1)
                        if (b==d .and. i==k) val = val + h2c_voov(a,l,j,c) ! (ij)
                        if (a==d .and. j==k) val = val + h2c_voov(b,l,i,c) ! (ab)
                        if (a==d .and. i==k) val = val - h2c_voov(b,l,j,c) ! (ij)(ab)
                        ! (cd)
                        if (b==c .and. j==l) val = val - h2c_voov(a,k,i,d) ! (1)
                        if (b==c .and. i==l) val = val + h2c_voov(a,k,j,d) ! (ij)
                        if (a==c .and. j==l) val = val + h2c_voov(b,k,i,d) ! (ab)
                        if (a==c .and. i==l) val = val - h2c_voov(b,k,j,d) ! (ij)(ab)
                        ! (kl)(cd)
                        if (b==c .and. j==k) val = val + h2c_voov(a,l,i,d) ! (1)
                        if (b==c .and. i==k) val = val - h2c_voov(a,l,j,d) ! (ij)
                        if (a==c .and. j==k) val = val - h2c_voov(b,l,i,d) ! (ab)
                        if (a==c .and. i==k) val = val + h2c_voov(b,l,j,d) ! (ij)(ab)
                        !!! h2a(klij) d(bd) d(ac)
                        if (b==d .and. a==c) val = val + h2c_oooo(k,l,i,j)
                        !!! h2a(abcd) d(ik) d(jl)
                        if (i==k .and. j==l) val = val + h2c_vvvv(a,b,c,d)

                end function calc_DCDC_matel


end module eomcc_initial_guess                         
