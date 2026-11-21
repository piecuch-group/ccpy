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

              subroutine vvvv_tn(resid, ra, rb, tdress, hoovv, ha, hb, &
                                 source, amps, goff, target, poff, tslot, texc, &
                                 plint, plcoef, ploff, sece, secf, te, tf, &
                                 same_spin, use_h1, &
                                 noa, nob, nua, nub, naux, nrec, ngroup, nres, &
                                 ntype, npl, nse, nsf, nce, ncf)
                 ! Every Hbar(vvvv) contraction onto selected T3/T4 determinants of
                 ! one pair spin, from one slice W_ab(e,f) per target pair (a,b):
                 !  * ladder: records as in ccpy.cc.chol_ppl; source(:,is)=(e,f),
                 !    target(:,it)=(group,signed determinant). Pair (a,b) has index
                 !    (a-1)*nub+b; same spin uses a<b. goff/poff are one-based,
                 !    end-exclusive offsets.
                 !  * use_h1: h1(vv) delta terms folded into the ladder, weighted
                 !    1/(nv-1) because each virtual sits in nv-1 pairs of a determinant
                 !    with nv virtuals (texc rows 1:4, zero padded).
                 !  * T2*T2: placements ploff(tslot(it)):ploff(tslot(it)+1)-1 of target
                 !    record it add plcoef * sum_ef W(e,f) te(e,ce) tf(f,cf). plint rows
                 !    (r,x,y,esec,s,z,w,fsec) name texc rows; column
                 !    c = off + D(r) + d1*(D(x)-1 + d2*(D(y)-1)) with sec(:,isec)=(off,d1,d2).
                 integer, intent(in) :: noa,nob,nua,nub,naux,nrec,ngroup,nres
                 integer, intent(in) :: ntype,npl,nse,nsf,nce,ncf
                 logical, intent(in) :: same_spin,use_h1
                 real(kind=8), intent(in) :: ra(naux,nua,nua),rb(naux,nub,nub)
                 real(kind=8), intent(in) :: tdress(nua,nub,noa,nob)
                 real(kind=8), intent(in) :: hoovv(noa,nob,nua,nub),amps(nrec)
                 real(kind=8), intent(in) :: ha(nua,nua),hb(nub,nub)
                 integer, intent(in) :: source(2,nrec),goff(ngroup+1),target(2,nrec),poff(nua*nub+1)
                 integer, intent(in) :: tslot(nrec),texc(8,nres)
                 integer, intent(in) :: plint(8,npl),ploff(ntype+1),sece(3,nse),secf(3,nsf)
                 real(kind=8), intent(in) :: plcoef(npl),te(nua,nce),tf(nub,ncf)
                 real(kind=8), intent(out) :: resid(nres)
                 integer(kind=8), parameter :: scratch_elements=8388608_8
                 integer :: nblk,ntiles_a,tile,a,b,nb,k,e,f,ipair,it,is,g,idet,nslot,ak
                 integer :: it0,it1,m,mcap,maxpl,p,sl,q,ce,cf
                 logical :: has_targets,do_t2t2
                 real(kind=8) :: val,factor,coulomb,hv,lam,t2t2
                 real(kind=8), allocatable :: y(:,:,:),w(:,:,:),tb(:,:,:),contributions(:)
                 real(kind=8), allocatable :: wk(:,:),u(:,:),tsub(:,:)
                 integer, allocatable :: slots(:,:),nper(:),mark(:),cols(:)
                 real(kind=8), external :: ddot

                 resid=0.0d0
                 if (nua==0 .or. nub==0 .or. noa==0 .or. nob==0) return
                 if (nrec==0 .or. nres==0) return
                 nblk=int(min(32_8,int(nua,8),max(1_8,scratch_elements/(2_8*nua*nub+int(noa,8)*nob))))
                 ntiles_a=(nua+nblk-1)/nblk
                 factor=merge(0.5d0,1.0d0,same_spin)
                 do_t2t2=ploff(ntype+1)>ploff(1)
                 maxpl=0
                 do sl=1,ntype
                    maxpl=max(maxpl,ploff(sl+1)-ploff(sl))
                 end do
                 mcap=max(maxpl,int(min(4096_8,scratch_elements/(int(nua,8)+nub))))
                 ! Record ids per determinant, in input order, for a deterministic gather.
                 allocate(nper(nres))
                 nper=0
                 do it=1,nrec
                    idet=abs(target(2,it))
                    nper(idet)=nper(idet)+1
                 end do
                 nslot=maxval(nper)
                 allocate(contributions(nrec),slots(nslot,nres))
                 contributions=0.0d0
                 slots=0
                 nper=0
                 do it=1,nrec
                    idet=abs(target(2,it))
                    nper(idet)=nper(idet)+1
                    slots(nper(idet),idet)=it
                 end do

                 !$omp parallel default(shared) &
                 !$omp& private(tile,a,b,nb,k,e,f,ipair,it,is,g,idet,has_targets,val,coulomb,hv,lam,ak, &
                 !$omp& it0,it1,m,p,sl,q,ce,cf,t2t2,y,w,tb,wk,u,tsub,mark,cols)
                 allocate(y(nub,nua,nblk),w(nua,nub,nblk),tb(nblk,noa,nob))
                 if (do_t2t2) then
                    allocate(wk(nua,nub),u(nua,mcap),tsub(nub,mcap),mark(ncf),cols(mcap))
                    mark=0
                 end if
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
                    if (naux>0) then
                       call dgemm('t','n',nub,nua*nb,naux,1.0d0,rb(:,:,b),naux, &
                                  ra(:,:,a:a+nb-1),naux,0.0d0,y,nub)
                    else
                       y(:,:,1:nb)=0.0d0
                    end if
                    tb(1:nb,:,:)=tdress(a:a+nb-1,b,:,:)
                    call dgemm('t','t',nua*nub,nb,noa*nob,factor,hoovv,noa*nob,tb,nblk,0.0d0,w,nua*nub)
                    do k=1,nb
                       ak=a+k-1
                       ipair=(ak-1)*nub+b
                       if (poff(ipair)==poff(ipair+1)) cycle
                       ! ladder (+ h1) part, signed by the target record
                       do it=poff(ipair),poff(ipair+1)-1
                          g=target(1,it)
                          idet=abs(target(2,it))
                          lam=0.0d0
                          if (use_h1) lam=1.0d0/dble(count(texc(1:4,idet)>0)-1)
                          val=0.0d0
                          do is=goff(g),goff(g+1)-1
                             e=source(1,is)
                             f=source(2,is)
                             coulomb=y(f,e,k)
                             if (same_spin) coulomb=coulomb-y(e,f,k)
                             coulomb=coulomb+w(e,f,k)
                             if (use_h1) then
                                hv=0.0d0
                                if (f==b) hv=hv+ha(ak,e)
                                if (e==ak) hv=hv+hb(b,f)
                                if (same_spin) then
                                   if (e==b) hv=hv-ha(ak,f)
                                   if (f==ak) hv=hv-hb(b,e)
                                end if
                                coulomb=coulomb+lam*hv
                             end if
                             val=val+coulomb*amps(is)
                          end do
                          if (target(2,it)<0) val=-val
                          contributions(it)=val
                       end do
                       if (.not.do_t2t2) cycle
                       ! T2*T2 part: U = W * tf(:,distinct columns) in chunks, then dots
                       do f=1,nub
                          do e=1,nua
                             wk(e,f)=y(f,e,k)+w(e,f,k)
                             if (same_spin) wk(e,f)=wk(e,f)-y(e,f,k)
                          end do
                       end do
                       it0=poff(ipair)
                       do while (it0<poff(ipair+1))
                          m=0
                          it1=it0
                          do while (it1<poff(ipair+1))
                             sl=tslot(it1)
                             if (m+ploff(sl+1)-ploff(sl)>mcap) exit
                             idet=abs(target(2,it1))
                             do p=ploff(sl),ploff(sl+1)-1
                                cf=secf(1,plint(8,p))+texc(plint(5,p),idet)+secf(2,plint(8,p))*((texc(plint(6,p),idet)-1)&
                                      +secf(3,plint(8,p))*(texc(plint(7,p),idet)-1))
                                if (mark(cf)==0) then
                                   m=m+1
                                   mark(cf)=m
                                   cols(m)=cf
                                end if
                             end do
                             it1=it1+1
                          end do
                          if (m>0) then
                             do q=1,m
                                tsub(:,q)=tf(:,cols(q))
                             end do
                             call dgemm('n','n',nua,m,nub,1.0d0,wk,nua,tsub,nub,0.0d0,u,nua)
                             do it=it0,it1-1
                                sl=tslot(it)
                                idet=abs(target(2,it))
                                t2t2=0.0d0
                                do p=ploff(sl),ploff(sl+1)-1
                                   ce=sece(1,plint(4,p))+texc(plint(1,p),idet)+sece(2,plint(4,p))*((texc(plint(2,p),idet)-1)&
                                      +sece(3,plint(4,p))*(texc(plint(3,p),idet)-1))
                                   cf=secf(1,plint(8,p))+texc(plint(5,p),idet)+secf(2,plint(8,p))*((texc(plint(6,p),idet)-1)&
                                      +secf(3,plint(8,p))*(texc(plint(7,p),idet)-1))
                                   t2t2=t2t2+plcoef(p)*ddot(nua,te(:,ce),1,u(:,mark(cf)),1)
                                end do
                                contributions(it)=contributions(it)+t2t2
                             end do
                             do q=1,m
                                mark(cols(q))=0
                             end do
                          end if
                          it0=it1
                       end do
                    end do
                 end do
                 !$omp end do
                 deallocate(y,w,tb)
                 if (do_t2t2) deallocate(wk,u,tsub,mark,cols)
                 ! The tile-loop barrier completes all record writes before this
                 ! disjoint determinant gather; no atomics or residual replicas.
                 !$omp do schedule(static)
                 do idet=1,nres
                    val=0.0d0
                    do k=1,nslot
                       it=slots(k,idet)
                       if (it/=0) val=val+contributions(it)
                    end do
                    resid(idet)=val
                 end do
                 !$omp end do
                 !$omp end parallel
                 deallocate(contributions,slots,nper)

              end subroutine vvvv_tn

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
