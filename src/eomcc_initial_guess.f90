module eomcc_initial_guess

        implicit none

        contains

                subroutine eomccs_d(nroot,noact,nuact,Rvec,omega,Hmat,&
                                    H1A_oo,H1A_vv,H1A_ov,&
                                    H1B_oo,H1B_vv,H1B_ov,&
                                    H2A_oooo,H2A_vvvv,H2A_voov,H2A_vooo,H2A_vvov,H2A_ooov,H2A_vovv,&
                                    H2B_oooo,H2B_vvvv,H2B_voov,H2B_ovvo,H2B_vovo,H2B_ovov,H2B_vooo,&
                                    H2B_ovoo,H2B_vvov,H2B_vvvo,H2B_ooov,H2B_oovo,H2B_vovv,H2B_ovvv,&
                                    H2C_oooo,H2C_vvvv,H2C_voov,H2C_vooo,H2C_vvov,H2C_ooov,H2C_vovv,&
                                    noa,nua,nob,nub,n1a,n1b,n2a,n2a_unique,n2b,n2c,n2c_unique,ndim_unique,ndim)

                        integer, intent(in) :: nroot, noa, nua, nob, nub, noact, nuact,&
                                               n1a, n1b, n2a, n2a_unique, n2b, n2c, n2c_unique, ndim_unique, ndim
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

                        real(kind=8), intent(out) :: omega(nroot), Rvec(ndim,nroot), Hmat(ndim_unique,ndim_unique)

                        real(kind=8), allocatable :: Htemp(:,:)
                        real(kind=8) :: evecs(ndim_unique,ndim_unique), evals(ndim_unique)
                        integer :: i, j, k, l, a, b, c, d, ct1, ct2, pos(6),&
                                   act_rng_oa(2), act_rng_ua(2), act_rng_ob(2), act_rng_ub(2)

                        pos(1) = 0
                        pos(2) = n1a
                        pos(3) = n1a+n1b
                        pos(4) = n1a+n1b+n2a_unique
                        pos(5) = n1a+n1b+n2a_unique+n2b
                        pos(6) = n1a+n1b+n2a_unique+n2b+n2c_unique

                        act_rng_oa(1) = max(0, noa-noact)
                        act_rng_oa(2) = noa
                        act_rng_ua(1) = 0
                        act_rng_ua(2) = min(nua, nuact)
                        act_rng_ob(1) = max(0, nob-noact)
                        act_rng_ob(2) = nob
                        act_rng_ub(1) = 0
                        act_rng_ub(2) = min(nub, nuact)

                        ! < ia | H | jb >
                        allocate(Htemp(n1a,n1a))
                        ct1 = 0
                        do i = 1 , noa
                        do a = 1 , nua
                            ct1 = ct1 + 1
                            ct2 = 0
                            do j = 1 , noa
                            do b = 1 , nua
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
                        allocate(Htemp(n1b,n1a))
                        ct1 = 0
                        do i = 1 , nob
                        do a = 1 , nub
                            ct1 = ct1 + 1
                            ct2 = 0
                            do j = 1 , noa
                            do b = 1 , nua
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
                        allocate(Htemp(n1a,n1b))
                        ct1 = 0
                        do i = 1 , noa
                        do a = 1 , nua
                            ct1 = ct1 + 1
                            ct2 = 0
                            do j = 1 , nob
                            do b = 1 , nub
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
                        allocate(Htemp(n1b,n1b))
                        ct1 = 0
                        do i = 1 , nob
                        do a = 1 , nub
                            ct1 = ct1 + 1
                            ct2 = 0
                            do j = 1 , nob
                            do b = 1 , nub
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
                        allocate(Htemp(n1a,n2a_unique))
                        ct1 = 0
                        do i = 1, noa
                        do a = 1, nua
                            ct1 = ct1 + 1
                            ct2 = 0
                            do j = 1, noa
                            do k = j+1, noa
                            do b = 1, nua
                            do c = b+1, nua
                                ct2 = ct2 + 1
                                if (.not. is_active(j,k,b,c,act_rng_oa,act_rng_ua)) cycle
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
                        allocate(Htemp(n1a,n2b))
                        ct1 = 0
                        do i = 1, noa
                        do a = 1, nua
                            ct1 = ct1 + 1
                            ct2 = 0
                            do j = 1, noa
                            do k = 1, nob
                            do b = 1, nua
                            do c = 1, nub
                                ct2 = ct2 + 1
                                if (.not. is_active(j,k,b,c,act_rng_occ,act_rng_unocc)) cycle
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
                        allocate(Htemp(n1b,n2b))
                        ct1 = 0
                        do i = 1, nob
                        do a = 1, nub
                            ct1 = ct1 + 1
                            ct2 = 0
                            do j = 1, noa
                            do k = 1, nob
                            do b = 1, nua
                            do c = 1, nub
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
                        allocate(Htemp(n1b,n2c_unique))
                        ct1 = 0
                        do i = 1, nob
                        do a = 1, nub
                            ct1 = ct1 + 1
                            ct2 = 0
                            do j = 1, nob
                            do k = j+1, nob
                            do b = 1, nub
                            do c = b+1, nub
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
                        allocate(Htemp(n2a_unique,n1a))
                        ct1 = 0
                        do i = 1, noa
                        do j = i+1, noa
                        do a = 1, nua
                        do b = a+1, nua
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, noa
                            do c = 1, nua
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
                        allocate(Htemp(n2b,n1a))
                        ct1 = 0
                        do i = 1, noa
                        do j = 1, nob
                        do a = 1, nua
                        do b = 1, nub
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, noa
                            do c = 1, nua
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
                        allocate(Htemp(n2b,n1b))
                        ct1 = 0
                        do i = 1, noa
                        do j = 1, nob
                        do a = 1, nua
                        do b = 1, nub
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, nob
                            do c = 1, nub
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
                        allocate(Htemp(n2c_unique,n1b))
                        ct1 = 0
                        do i = 1, nob
                        do j = i+1, nob
                        do a = 1, nub
                        do b = a+1, nub
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, nob
                            do c = 1, nub
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
                        allocate(Htemp(n2a_unique,n2a_unique))
                        ct1 = 0
                        do i = 1, noa
                        do j = i+1, noa
                        do a = 1, nua
                        do b = a+1, nua
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, noa
                            do l = k+1, noa
                            do c = 1, nua
                            do d = c+1, nua
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
                        allocate(Htemp(n2a_unique,n2b))
                        ct1 = 0
                        do i = 1, noa
                        do j = i+1, noa
                        do a = 1, nua
                        do b = a+1, nua
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, noa
                            do l = 1, nob
                            do c = 1, nua
                            do d = 1, nub
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
                        allocate(Htemp(n2b,n2a_unique))
                        ct1 = 0
                        do i = 1, noa
                        do j = 1, nob
                        do a = 1, nua
                        do b = 1, nub
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, noa
                            do l = k+1, noa
                            do c = 1, nua
                            do d = c+1, nua
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
                        allocate(Htemp(n2b,n2b))
                        ct1 = 0
                        do i = 1, noa
                        do j = 1, nob
                        do a = 1, nua
                        do b = 1, nub
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, noa
                            do l = 1, nob
                            do c = 1, nua
                            do d = 1, nub
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
                        allocate(Htemp(n2b,n2c_unique))
                        ct1 = 0
                        do i = 1, noa
                        do j = 1, nob
                        do a = 1, nua
                        do b = 1, nub
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, nob
                            do l = k+1, nob
                            do c = 1, nub
                            do d = c+1, nub
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
                        allocate(Htemp(n2c_unique,n2b))
                        ct1 = 0
                        do i = 1, nob
                        do j = i+1, nob
                        do a = 1, nub
                        do b = a+1, nub
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, noa
                            do l = 1, nob
                            do c = 1, nua
                            do d = 1, nub
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
                        allocate(Htemp(n2c_unique,n2c_unique))
                        ct1 = 0
                        do i = 1, nob
                        do j = i+1, nob
                        do a = 1, nub
                        do b = a+1, nub
                            ct1 = ct1 + 1
                            ct2 = 0
                            do k = 1, nob
                            do l = k+1, nob
                            do c = 1, nub
                            do d = c+1, nub
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


                end subroutine eomccs_d

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
                                val = val - H1A_oo(k,j)
                            end if
                            if (j==k) then
                                val = val - H1A_oo(l,i)
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
                                val = val + H1A_vv(a,d)
                            end if
                            if (a==c) then
                                val = val + H1A_vv(b,d)
                            end if
                            if (a==d) then
                                val = val + H1A_vv(b,c)
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
                                val = val - H1B_oo(k,j)
                            end if
                            if (j==k) then
                                val = val - H1B_oo(l,i)
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
                                val = val + H1B_vv(a,d)
                            end if
                            if (a==c) then
                                val = val + H1B_vv(b,d)
                            end if
                            if (a==d) then
                                val = val + H1B_vv(b,c)
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

                function is_active(i,j,a,b,act_rng_occ,act_rng_unocc) result(xbool)

                        integer, intent(in) :: i, j, a, b,&
                        act_rng_occ(2), act_rng_unocc(2)

                        logical :: xbool

                        integer :: num_act_holes, num_act_particles
                
                        xbool = .false.
                        num_act_holes = 0
                        num_act_particles = 0

                        if (i > act_rng_occ(1) .and. i <= act_rng_occ(2)) then
                            num_act_holes = num_act_holes + 1
                        end if
                        if (j > act_rng_occ(1) .and. j <= act_rng_occ(2)) then
                            num_act_holes = num_act_holes + 1
                        end if
                        if (a > act_rng_unocc(1) .and. a <= act_rng_unocc(2)) then
                            num_act_particles = num_act_particles + 1
                        end if
                        if (b > act_rng_unocc(1) .and. b <= act_rng_unocc(2)) then
                            num_act_particles = num_act_particles + 1
                        end if

                        if (num_act_holes >= 1 .and. num_act_particles >= 1) then
                            xbool = .true.
                        end if

                end function is_active 



                !subroutine reorder_dets(I1,I2,idx1,idx2,phase)
                    ! reorder bitstring determinants I1 = [I1a, I1b] and I2 =
                    ! [I2a,I2b] into the order of maximum coincidence as
                    ! excitations out of HF, idx1 and idx2, and the resulting
                    ! phase



                !end subroutine reorder_dets

                !function onebody_HBar_slater(det1,det2,H1A_oo,H1A_vv,H1A_ov,H1B_oo,H1B_vv,H1B_ov,noa,nua,nob,nub) result(val)

                 !       integer, intent(in) :: noa, nua, nob, nub, i1, a1, i2, a2
                 !       real(kind=8), intent(in) :: H1A_oo(noa,noa),H1A_vv(nua,nua),H1A_ov(noa,nua),&
                 !                                   H1B_oo(nob,nob),H1B_vv(nub,nub),H1B_ov(nob,nub)

                 !       real(kind=8) :: val


                !end function onebody_HBar_slater

end module eomcc_initial_guess                         
