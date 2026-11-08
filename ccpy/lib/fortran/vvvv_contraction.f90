module vvvv_contraction

        implicit none

        contains

              subroutine vvvv_t2_sym(resid, R, t2, no, nu, naux)
                 ! Only resid(:,:,b,a), a<b, is returned. Pack e<f once, then
                 ! contract batches with DGEMM. Each tile owns disjoint output;
                 ! a reduction would replicate the entire residual per thread.
                 integer, intent(in) :: no, nu, naux
                 real(kind=8), intent(in) :: R(naux,nu,nu), t2(no,no,nu,nu)
                 real(kind=8), intent(out) :: resid(no,no,nu,nu)
                 ! Bound the integral/antisymmetric scratch to 64 MiB per worker
                 ! (apart from the unavoidable single-pair minimum). Cap batches
                 ! at 32 pairs to retain enough independent contraction tiles.
                 integer(kind=8), parameter :: scratch_elements = 8388608_8
                 integer :: nblk, ntiles_b, tile, a, b0, bend, nb, k, e, f, kp, npair
                 real(kind=8), allocatable :: tpack(:,:), y(:,:,:), z(:,:)

                 resid = 0.0d0
                 npair = nu*(nu-1)/2
                 if (npair == 0 .or. no == 0 .or. naux == 0) return
                 nblk = min(32, nu, max(1, int(scratch_elements / (int(nu,8)*nu + npair))))
                 ntiles_b = (nu+nblk-1)/nblk
                 allocate(tpack(no*no,npair))
                 kp = 0
                 do e = 1, nu
                    do f = e+1, nu
                       kp = kp + 1
                       tpack(:,kp) = reshape(t2(:,:,f,e), (/no*no/))
                    end do
                 end do

                 !$omp parallel default(shared) private(tile,a,b0,bend,nb,k,e,f,kp,y,z)
                 allocate(y(nu,nu,nblk), z(npair,nblk))
                 ! Enumerate triangular pair tiles, not unequal-length outer-a
                 ! tasks. At nblk=1 this is unique-pair scheduling; larger tiles
                 ! retain BLAS-3 reuse of tpack without a per-pair GEMV scan.
                 !$omp do schedule(dynamic)
                 do tile = 1, nu*ntiles_b
                    a = (tile-1)/ntiles_b + 1
                    b0 = mod(tile-1,ntiles_b)*nblk + 1
                    bend = min(nu,b0+nblk-1)
                    b0 = max(b0,a+1)
                    if (b0 > bend) cycle
                    nb = bend-b0+1
                    call dgemm('t','n',nu,nu*nb,naux,1.0d0,R(:,:,a),naux,R(:,:,b0:bend),naux,0.0d0,y,nu)
                    do k = 1, nb
                       kp = 0
                       do e = 1, nu
                          do f = e+1, nu
                             kp = kp + 1
                             z(kp,k) = y(e,f,k) - y(f,e,k)
                          end do
                       end do
                    end do
                    call dgemm('n','n',no*no,nb,npair,1.0d0,tpack,no*no,z,npair,0.0d0,resid(:,:,b0:bend,a),no*no)
                 end do
                 !$omp end do
                 deallocate(y, z)
                 !$omp end parallel
                 deallocate(tpack)
              end subroutine vvvv_t2_sym

              subroutine vvvv_t2(resid, Ra, Rb, t2, noa, nua, nob, nub, naux)
                 ! resid(:,:,b,a) = sum_{efx} Ra(x,e,a) Rb(x,f,b) t2(:,:,f,e).
                 ! Fix b and batch a: DGEMM then builds y(f,e,a) in exactly
                 ! t2's (f,e) order. No integral transpose or T2 repacking is
                 ! needed. Only the small occupied-pair result is scattered.
                 integer, intent(in) :: noa, nua, nob, nub, naux
                 real(kind=8), intent(in) :: Ra(naux,nua,nua), Rb(naux,nub,nub)
                 real(kind=8), intent(in) :: t2(nob,noa,nub,nua)
                 real(kind=8), intent(out) :: resid(nob,noa,nub,nua)
                 integer(kind=8), parameter :: scratch_elements = 8388608_8
                 integer :: nblk, ntiles_a, tile, a, b, nb, k
                 real(kind=8), allocatable :: y(:,:,:), work(:,:)

                 if (noa == 0 .or. nob == 0 .or. nua == 0 .or. nub == 0 .or. naux == 0) then
                    resid = 0.0d0
                    return
                 end if
                 nblk = min(32, nua, max(1, int(scratch_elements / (int(nua,8)*nub + int(noa,8)*nob))))
                 ntiles_a = (nua+nblk-1)/nblk
                 !$omp parallel default(shared) private(tile,a,b,nb,k,y,work)
                 allocate(y(nub,nua,nblk), work(nob*noa,nblk))
                 !$omp do schedule(static)
                 do tile = 1, nub*ntiles_a
                    b = (tile-1)/ntiles_a + 1
                    a = mod(tile-1,ntiles_a)*nblk + 1
                    nb = min(nblk,nua-a+1)
                    call dgemm('t','n',nub,nua*nb,naux,1.0d0,Rb(:,:,b),naux,Ra(:,:,a:a+nb-1),naux,0.0d0,y,nub)
                    call dgemm('n','n',nob*noa,nb,nub*nua,1.0d0,t2,nob*noa,y,nub*nua,0.0d0,work,nob*noa)
                    do k = 1, nb
                       resid(:,:,b,a+k-1) = reshape(work(:,k), (/nob,noa/))
                    end do
                 end do
                 !$omp end do
                 deallocate(y, work)
                 !$omp end parallel
              end subroutine vvvv_t2

              subroutine vvvv_t3(resid, ra, rb, tdress, hoovv, &
                                 source, amps, goff, target, poff, same_spin, &
                                 noa, nob, nua, nub, naux, nrec, ngroup, nres)
                 ! Raw selected-T3 contribution, shared across spin cases via
                 ! records. Pair (a,b) has index (a-1)*nub+b; same spin uses a<b.
                 ! source(:,is)=(e,f), target(:,it)=(group,signed determinant).
                 ! goff/poff are one-based offsets, with end-exclusive ranges.
                 integer, intent(in) :: noa,nob,nua,nub,naux,nrec,ngroup,nres
                 logical, intent(in) :: same_spin
                 real(kind=8), intent(in) :: ra(naux,nua,nua),rb(naux,nub,nub)
                 real(kind=8), intent(in) :: tdress(nua,nub,noa,nob)
                 real(kind=8), intent(in) :: hoovv(noa,nob,nua,nub),amps(nrec)
                 integer, intent(in) :: source(2,nrec),goff(ngroup+1),target(2,nrec),poff(nua*nub+1)
                 real(kind=8), intent(out) :: resid(nres)
                 integer(kind=8), parameter :: scratch_elements=8388608_8
                 integer(kind=8) :: pair_elements
                 integer :: nblk,ntiles_a,tile,a,b,nb,k,e,f,i,j,ipair,it,is,g,idet
                 logical :: tiled,has_targets
                 real(kind=8) :: val,factor,coulomb,dressing
                 real(kind=8), allocatable :: y(:,:,:),w(:,:,:),tb(:,:,:),contributions(:)
                 integer, allocatable :: slots(:,:)

                 resid=0.0d0
                 if (nua==0 .or. nub==0 .or. noa==0 .or. nob==0) return
                 if (nrec==0 .or. nres==0) return
                 pair_elements=2_8*nua*nub+int(noa,8)*nob
                 nblk=min(32,nua,max(1,int(scratch_elements/pair_elements)))
                 ntiles_a=(nua+nblk-1)/nblk
                 tiled=pair_elements<=scratch_elements
                 factor=merge(0.5d0,1.0d0,same_spin)
                 allocate(contributions(nrec),slots(3,nres))
                 contributions=0.0d0
                 slots=0
                 ! Each determinant has at most three signed pair contributions.
                 ! Store record ids in input order for a deterministic gather.
                 do it=1,nrec
                    idet=abs(target(2,it))
                    do k=1,3
                       if (slots(k,idet)==0) then
                          slots(k,idet)=it
                          exit
                       end if
                    end do
                 end do

                 !$omp parallel default(shared) &
                 !$omp& private(tile,a,b,nb,k,e,f,i,j,ipair,it,is,g,idet,has_targets,val,coulomb,dressing,y,w,tb)
                 ! A scalar fallback keeps even an oversized single pair within
                 ! the 64 MiB per-thread tile budget. Record storage is shared.
                 if (tiled) allocate(y(nub,nua,nblk),w(nua,nub,nblk),tb(nblk,noa,nob))
                 !$omp do schedule(dynamic,1)
                 do tile=1,nub*ntiles_a
                    b=(tile-1)/ntiles_a+1
                    a=mod(tile-1,ntiles_a)*nblk+1
                    nb=min(nblk,merge(min(nua,b-1),nua,same_spin)-a+1)
                    if (nb<=0) cycle
                    has_targets=.false.
                    do k=1,nb
                       ipair=(a+k-2)*nub+b
                       if (poff(ipair)<poff(ipair+1)) then
                          has_targets=.true.
                          exit
                       end if
                    end do
                    if (.not.has_targets) cycle
                    if (tiled) then
                       if (naux>0) then
                          call dgemm('t','n',nub,nua*nb,naux,1.0d0,rb(:,:,b),naux, &
                                     ra(:,:,a:a+nb-1),naux,0.0d0,y,nub)
                       else
                          ! Still form the dressing when the Coulomb rank is zero.
                          y(:,:,1:nb)=0.0d0
                       end if
                       tb(1:nb,:,:)=tdress(a:a+nb-1,b,:,:)
                       call dgemm('t','t',nua*nub,nb,noa*nob,factor,hoovv,noa*nob,tb,nblk,0.0d0,w,nua*nub)
                    end if
                    do k=1,nb
                       ipair=(a+k-2)*nub+b
                       do it=poff(ipair),poff(ipair+1)-1
                          g=target(1,it)
                          val=0.0d0
                          do is=goff(g),goff(g+1)-1
                             e=source(1,is)
                             f=source(2,is)
                             if (tiled) then
                                coulomb=y(f,e,k)
                                if (same_spin) coulomb=coulomb-y(e,f,k)
                                dressing=w(e,f,k)
                             else
                                coulomb=dot_product(rb(:,f,b),ra(:,e,a+k-1))
                                if (same_spin) coulomb=coulomb-dot_product(rb(:,e,b),ra(:,f,a+k-1))
                                dressing=0.0d0
                                do j=1,nob
                                   do i=1,noa
                                      dressing=dressing+tdress(a+k-1,b,i,j)*hoovv(i,j,e,f)
                                   end do
                                end do
                                dressing=factor*dressing
                             end if
                             val=val+(coulomb+dressing)*amps(is)
                          end do
                          if (target(2,it)<0) val=-val
                          contributions(it)=val
                       end do
                    end do
                 end do
                 !$omp end do
                 if (tiled) deallocate(y,w,tb)
                 ! The tile-loop barrier completes all record writes before this
                 ! disjoint determinant gather; no atomics or residual replicas.
                 !$omp do schedule(static)
                 do idet=1,nres
                    val=0.0d0
                    do k=1,3
                       it=slots(k,idet)
                       if (it/=0) val=val+contributions(it)
                    end do
                    resid(idet)=val
                 end do
                 !$omp end do
                 !$omp end parallel
                 deallocate(contributions,slots)
              end subroutine vvvv_t3

              subroutine vvvv_index(idx,a,b,c,d,nu)
                         
                         integer, intent(in) :: a, b, c, d, nu
                         integer, intent(out) :: idx
                         
                         integer :: ab, cd, n
                 
                         ! linear index of (a,b), a<b
                         ab = shiftr((2*nu - 2 - a) * (a - 1),1) + b - 2
                         ! linear index of (c,d), c<d
                         cd = shiftr((2*nu - 2 - d) * (c - 1),1) + d - 2
                         ! dimension of each (a,b), and (c,d) pairs = nu*(nu-1)/2
                         n = shiftr(nu*(nu - 1),1)
                         ! effective linear index h(idx) = <ab||cd>
                         idx = cd + n*ab + 1
                         
              end subroutine vvvv_index
           
              subroutine contract_vt2_pppp(resid,h2_vvvv,t2,no,nu)

                         integer, intent(in) :: no, nu

                         real(kind=8), intent(in) :: h2_vvvv(nu,nu,nu,nu)
                         real(kind=8), intent(in) :: t2(nu,nu,no,no)

                         real(kind=8), intent(out) :: resid(nu,nu,no,no)

                         integer :: i, j, a, b, e, f
                         integer :: idx

                         resid = 0.0d0
                         do i = 1,no
                            do j = i+1,no
                               do a = 1,nu
                                  do b = a+1,nu
                                     ! 1/2 h2(abef) * t2(efij)
                                     do e = 1,nu
                                        do f = e+1,nu
                                           call vvvv_index(idx,a,b,e,f,nu)
                                           !resid(b,a,j,i) = resid(b,a,j,i) + h2_vvvv(idx)*t2(f,e,j,i)
                                        end do
                                     end do
                                     resid(a,b,j,i) = -resid(b,a,j,i)
                                     resid(b,a,i,j) = -resid(b,a,j,i)
                                     resid(a,b,i,j) = resid(b,a,j,i)
                                  end do
                               end do
                            end do
                         end do

              end subroutine contract_vt2_pppp

   
!              subroutine contract_vt2_aa_cholesky(resid, R_chol, t2a, noa, nua, naux, norb)
!
!                        ! input variables
!                        integer, intent(in) :: noa, nua, norb
!                        real(kind=8), intent(in) :: R_chol(naux,norb,norb)
!                        real(kind=8), intent(in) :: t2a(nua,nua,noa,noa)
!                        ! output variables
!                        real(kind=8), intent(inout) :: resid(nua,nua,noa,noa)
!                        ! local variables
!                        integer :: x, e, f, i, j, a, b, ab, nua2
!                        integer, allocatable :: idx(:,:)
!                        real(kind=8), allocatable :: batch_ints(:,:)
!
!                        !
!                        ! get map of linear index ab -> (a,b) for a<b
!                        !
!                        nua2 = nua*(nua-1)/2
!                        allocate(idx(nua2,2))
!                        ab = 1
!                        do a=1,nua
!                           do b=a+1,nua
!                              idx(ab,1)=a
!                              idx(ab,2)=b
!                              ab = ab + 1
!                           end do
!                        end do
!
!                        !
!                        ! Perform loop over pairs a<b
!                        !
!                        allocate(batch_ints(nua,nua))
!                        do ab=1,nua2
!                           ! indices a<b
!                           a = idx(ab,1); b = idx(ab,2);
!                           !
!                           ! build cholesky integral block
!                           !
!                           batch_ints = 0.0d0
!                           do x=1,naux
!                              do e=1,nua
!                                 do f=e+1,nua
!                                    batch_ints(e,f)=batch_ints(e,f)+R_chol(x,a+noa,e+noa)*R_chol(x,b+noa,f+noa)
!                                 end do
!                              end do
!                           end do
!                           !
!                           ! perform PPL contraction
!                           !
!                           do i=1,noa
!                              do j=i+1,noa
!                                 do e=1,nua
!                                    do f=e+1,nua
!                                       resid(a,b,i,j)=resid(a,b,i,j)+batch_ints(e,f)*t2a(e,f,i,j)
!                                       resid(b,a,i,j)=resid(b,a,i,j)+batch_ints(e,f)*t2a(e,f,i,j)
!                                    end do
!                                 end do
!                              end do
!                           end do
!                        end do
!                        deallocate(idx,batch_ints)
!
!              end subroutine contract_vt2_aa_cholesky


end module vvvv_contraction
