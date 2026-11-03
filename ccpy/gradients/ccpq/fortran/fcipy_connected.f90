! CCpy connector for the externally supplied FCIpy ci module.
! Enumerate connected determinants; reuse FCIpy's Slater matrix elements/phases.
module ccpq_fcipy
use ci, only: get_occupied, get_excitation, slater0, slater1, slater2
implicit none
contains
subroutine connected_sigma(det, keys, order, h, v, x, norb, noa, nfrozen, ndet, sigma)
    integer, intent(in) :: norb, noa, nfrozen, ndet
    integer*8, intent(in) :: det(1,2,ndet), keys(ndet)
    integer, intent(in) :: order(ndet)
    double precision, intent(in) :: h(norb,norb), v(norb,norb,norb,norb), x(ndet)
    double precision, intent(out) :: sigma(ndet)
    integer :: src, s, i, j, a, b, oi, oj, p, q
    integer :: occ(noa,2), exc(0:2,2,2), degree
    integer*8 :: target(1,2), first(1,2)
    double precision :: value, phase
    sigma=0d0
    do src=1,ndet
        if (x(src)==0d0) cycle
        call get_occupied(det(:,:,src),1,noa,occ)
        call slater0(occ,noa,noa,norb,h,v,value)
        sigma(src)=sigma(src)+value*x(src)
        ! Same-spin singles and doubles.
        do s=1,2
            do i=nfrozen+1,noa
                oi=occ(i,s)-1
                do a=nfrozen,norb-1
                    if (btest(det(1,s,src),a)) cycle
                    first=det(:,:,src)
                    first(1,s)=ibset(ibclr(first(1,s),oi),a)
                    call add(first)
                    do j=i+1,noa
                        oj=occ(j,s)-1
                        do b=a+1,norb-1
                            if (btest(det(1,s,src),b)) cycle
                            target=first
                            target(1,s)=ibset(ibclr(target(1,s),oj),b)
                            call add(target)
                        end do
                    end do
                end do
            end do
        end do
        ! Opposite-spin doubles.
        do i=nfrozen+1,noa
            oi=occ(i,1)-1
            do a=nfrozen,norb-1
                if (btest(det(1,1,src),a)) cycle
                first=det(:,:,src)
                first(1,1)=ibset(ibclr(first(1,1),oi),a)
                do j=nfrozen+1,noa
                    oj=occ(j,2)-1
                    do b=nfrozen,norb-1
                        if (btest(det(1,2,src),b)) cycle
                        target=first
                        target(1,2)=ibset(ibclr(target(1,2),oj),b)
                        call add(target)
                    end do
                end do
            end do
        end do
    end do
contains
    subroutine add(d)
        integer*8, intent(in) :: d(1,2)
        integer*8 :: key
        integer :: lo, hi, mid, dst
        key=ior(d(1,1),ishft(d(1,2),norb))
        lo=1; hi=ndet
        do while (lo<=hi)
            mid=(lo+hi)/2
            if (keys(mid)<key) then
                lo=mid+1
            else if (keys(mid)>key) then
                hi=mid-1
            else
                dst=order(mid)
                call get_excitation(det(:,:,src),d,exc,degree,phase,1)
                if (degree==1) then
                    call slater1(occ,noa,noa,norb,h,v,exc,phase,value)
                else if (degree==2) then
                    call slater2(occ,noa,noa,norb,h,v,exc,phase,value)
                else
                    return
                end if
                sigma(dst)=sigma(dst)+value*x(src)
                return
            end if
        end do
    end subroutine add
end subroutine connected_sigma
subroutine connected_elements(det, keys, order, h, v, norb, noa, nfrozen, ndet, nconn, indices, values)
    integer, intent(in) :: norb, noa, nfrozen, ndet, nconn
    integer*8, intent(in) :: det(1,2,ndet), keys(ndet)
    integer, intent(in) :: order(ndet)
    double precision, intent(in) :: h(norb,norb), v(norb,norb,norb,norb)
    integer, intent(out) :: indices(nconn,ndet)
    double precision, intent(out) :: values(nconn,ndet)
    integer :: src, s, i, j, a, b, oi, oj, p, q, slot
    integer :: occ(noa,2), exc(0:2,2,2), degree
    integer*8 :: target(1,2), first(1,2)
    double precision :: value, phase
    indices=0; values=0d0
    do src=1,ndet
        call get_occupied(det(:,:,src),1,noa,occ)
        call slater0(occ,noa,noa,norb,h,v,value)
        slot=1
        indices(slot,src)=src; values(slot,src)=value
        ! Same-spin singles and doubles.
        do s=1,2
            do i=nfrozen+1,noa
                oi=occ(i,s)-1
                do a=nfrozen,norb-1
                    if (btest(det(1,s,src),a)) cycle
                    first=det(:,:,src)
                    first(1,s)=ibset(ibclr(first(1,s),oi),a)
                    call add(first)
                    do j=i+1,noa
                        oj=occ(j,s)-1
                        do b=a+1,norb-1
                            if (btest(det(1,s,src),b)) cycle
                            target=first
                            target(1,s)=ibset(ibclr(target(1,s),oj),b)
                            call add(target)
                        end do
                    end do
                end do
            end do
        end do
        ! Opposite-spin doubles.
        do i=nfrozen+1,noa
            oi=occ(i,1)-1
            do a=nfrozen,norb-1
                if (btest(det(1,1,src),a)) cycle
                first=det(:,:,src)
                first(1,1)=ibset(ibclr(first(1,1),oi),a)
                do j=nfrozen+1,noa
                    oj=occ(j,2)-1
                    do b=nfrozen,norb-1
                        if (btest(det(1,2,src),b)) cycle
                        target=first
                        target(1,2)=ibset(ibclr(target(1,2),oj),b)
                        call add(target)
                    end do
                end do
            end do
        end do
    end do
contains
    subroutine add(d)
        integer*8, intent(in) :: d(1,2)
        integer*8 :: key
        integer :: lo, hi, mid, dst
        key=ior(d(1,1),ishft(d(1,2),norb))
        lo=1; hi=ndet
        do while (lo<=hi)
            mid=(lo+hi)/2
            if (keys(mid)<key) then
                lo=mid+1
            else if (keys(mid)>key) then
                hi=mid-1
            else
                dst=order(mid)
                call get_excitation(det(:,:,src),d,exc,degree,phase,1)
                if (degree==1) then
                    call slater1(occ,noa,noa,norb,h,v,exc,phase,value)
                else if (degree==2) then
                    call slater2(occ,noa,noa,norb,h,v,exc,phase,value)
                else
                    return
                end if
                slot=slot+1
                indices(slot,src)=dst; values(slot,src)=value
                return
            end if
        end do
    end subroutine add
end subroutine connected_elements
! Contract either connected vector pairs or sparse coefficient entries directly.
subroutine graph_density(det, indptr, indices, left, right, norb, noa, ndet, nnz, D, G)
    integer, intent(in) :: norb, noa, ndet, nnz
    integer*8, intent(in) :: det(1,2,ndet)
    integer, intent(in) :: indptr(ndet+1), indices(nnz)
    double precision, intent(in) :: left(ndet), right(ndet)
    double precision, intent(out) :: D(norb,norb), G(norb,norb,norb,norb)
    integer :: i, j, k
    double precision :: w
    D=0d0; G=0d0
    do i=1,ndet
        if (left(i)==0d0) cycle
        do k=indptr(i)+1,indptr(i+1)
            j=indices(k)+1
            w=left(i)*right(j)
            if (w==0d0) cycle
            call accumulate_pair(det(:,:,i),det(:,:,j),w,norb,noa,D,G)
        end do
    end do
end subroutine graph_density

subroutine sparse_density(det, rows, cols, weights, norb, noa, ndet, npair, D, G)
    integer, intent(in) :: norb, noa, ndet, npair
    integer*8, intent(in) :: det(1,2,ndet)
    integer, intent(in) :: rows(npair), cols(npair)
    double precision, intent(in) :: weights(npair)
    double precision, intent(out) :: D(norb,norb), G(norb,norb,norb,norb)
    integer :: k
    D=0d0; G=0d0
    do k=1,npair
        if (weights(k)==0d0) cycle
        call accumulate_pair(det(:,:,rows(k)+1),det(:,:,cols(k)+1),weights(k),norb,noa,D,G)
    end do
end subroutine sparse_density

subroutine accumulate_pair(det1, det2, weight, norb, noa, D, G)
    integer, intent(in) :: norb, noa
    integer*8, intent(in) :: det1(1,2), det2(1,2)
    double precision, intent(in) :: weight
    double precision, intent(inout) :: D(norb,norb), G(norb,norb,norb,norb)
    integer :: occ(noa,2), exc(0:2,2,2), degree
    integer :: s, a, b, i, j, k, l
    double precision :: phase, w
    call get_excitation(det1,det2,exc,degree,phase,1)
    if (degree<0 .or. degree>2) return
    call get_occupied(det1,1,noa,occ)
    if (degree==0) then
        do s=1,2
            do a=1,noa
                i=occ(a,s)
                D(i,i)=D(i,i)+weight
                do b=1,noa
                    k=occ(b,s)
                    G(i,k,i,k)=G(i,k,i,k)+weight
                    G(i,k,k,i)=G(i,k,k,i)-weight
                end do
            end do
        end do
        do a=1,noa
            i=occ(a,1)
            do b=1,noa
                k=occ(b,2)
                G(i,k,i,k)=G(i,k,i,k)+weight
                G(k,i,k,i)=G(k,i,k,i)+weight
            end do
        end do
    else if (degree==1) then
        s=2
        if (exc(0,1,1)==1) s=1
        i=exc(1,1,s); j=exc(1,2,s); w=phase*weight
        D(i,j)=D(i,j)+w
        do a=1,noa
            k=occ(a,s)
            G(i,k,j,k)=G(i,k,j,k)+w
            G(k,i,j,k)=G(k,i,j,k)-w
            G(i,k,k,j)=G(i,k,k,j)-w
            G(k,i,k,j)=G(k,i,k,j)+w
            k=occ(a,3-s)
            G(i,k,j,k)=G(i,k,j,k)+w
            G(k,i,k,j)=G(k,i,k,j)+w
        end do
    else
        w=phase*weight
        if (exc(0,1,1)==2 .or. exc(0,1,2)==2) then
            s=2
            if (exc(0,1,1)==2) s=1
            i=exc(1,1,s); j=exc(1,2,s)
            k=exc(2,1,s); l=exc(2,2,s)
            G(i,k,j,l)=G(i,k,j,l)+w
            G(k,i,j,l)=G(k,i,j,l)-w
            G(i,k,l,j)=G(i,k,l,j)-w
            G(k,i,l,j)=G(k,i,l,j)+w
        else
            i=exc(1,1,1); j=exc(1,2,1)
            k=exc(1,1,2); l=exc(1,2,2)
            G(i,k,j,l)=G(i,k,j,l)+w
            G(k,i,l,j)=G(k,i,l,j)+w
        end if
    end if
end subroutine accumulate_pair

! Refresh only numeric values on a fixed connected graph across geometries.
subroutine update_connected_values(det, indptr, indices, h, v, norb, noa, ndet, nnz, values)
    integer, intent(in) :: norb, noa, ndet, nnz
    integer*8, intent(in) :: det(1,2,ndet)
    integer, intent(in) :: indptr(ndet+1), indices(nnz)
    double precision, intent(in) :: h(norb,norb), v(norb,norb,norb,norb)
    double precision, intent(inout) :: values(nnz)
    integer :: i, j, k, degree, occ(noa,2), exc(0:2,2,2)
    double precision :: diagonal, value, phase
    do i=1,ndet
        call get_occupied(det(:,:,i),1,noa,occ)
        call slater0(occ,noa,noa,norb,h,v,diagonal)
        do k=indptr(i)+1,indptr(i+1)
            j=indices(k)+1
            if (j==i) then
                values(k)=diagonal
                cycle
            end if
            call get_excitation(det(:,:,i),det(:,:,j),exc,degree,phase,1)
            value=0d0
            if (degree==1) then
                call slater1(occ,noa,noa,norb,h,v,exc,phase,value)
            else if (degree==2) then
                call slater2(occ,noa,noa,norb,h,v,exc,phase,value)
            end if
            values(k)=value
        end do
    end do
end subroutine update_connected_values

end module ccpq_fcipy
