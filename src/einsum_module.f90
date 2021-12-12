module einsum_module

    ! Written by Karthik Gururangan 2/11/21

    implicit none

    contains

        subroutine einsum(str,A,B,C)

            character(len=*), intent(in) :: str 
            real, intent(in) :: A(..), B(..)
            real, intent(out) :: C(..) 
            integer :: rankA, rankB, rankC, rank_vec(1:3)

            rankA = size(shape(A))
            rankB = size(shape(B))
            rankC = size(shape(C))

            rank_vec = (/rankA, rankB, rankC/)

            ! F * T1 -> SINGLES
            if (all(rank_vec == (/2,2,2/))) then
               call einsum222(str,A,shape(A),B,shape(B),C,shape(C))
            end if 
            ! V * T1 -> SINGLES
            if (all(rank_vec == (/4,2,2/))) then
               call einsum422(str,A,shape(A),B,shape(B),C,shape(C))
            end if 
            ! F * T2 -> SINGLES
            if (all(rank_vec == (/2,4,2/))) then
               call einsum242(str,A,shape(A),B,shape(B),C,shape(C))
            end if 
            ! V * T3 -> SINGLES
            if (all(rank_vec == (/4,6,2/))) then
               call einsum462(str,A,shape(A),B,shape(B),C,shape(C))
            end if
            ! V * T2 -> SINGLES
            if (all(rank_vec == (/4,4,2/))) then
               call einsum442(str,A,shape(A),B,shape(B),C,shape(C))
            end if 
            ! V * T2 -> DOUBLES
            if (all(rank_vec == (/4,4,4/))) then
               call einsum444(str,A,shape(A),B,shape(B),C,shape(C))
            end if 
            ! V * T1 -> DOUBLES
            if (all(rank_vec == (/4,2,4/))) then
               call einsum424(str,A,shape(A),B,shape(B),C,shape(C))
            end if 
            ! F * T2 -> DOUBLES
            if (all(rank_vec == (/2,4,4/))) then
               call einsum244(str,A,shape(A),B,shape(B),C,shape(C))
            end if 
            ! V * T3 -> DOUBLES
            if (all(rank_vec == (/4,6,4/))) then
               call einsum464(str,A,shape(A),B,shape(B),C,shape(C))
            end if 
            ! F * T3 -> DOUBLES
            if (all(rank_vec == (/2,6,4/))) then
               call einsum264(str,A,shape(A),B,shape(B),C,shape(C))
            end if 
            ! F * T3 -> TRIPLES
            if (all(rank_vec == (/2,6,6/))) then
               call einsum266(str,A,shape(A),B,shape(B),C,shape(C))
            end if 
            ! V * T3 -> TRIPLES
            if (all(rank_vec == (/4,6,6/))) then
               call einsum466(str,A,shape(A),B,shape(B),C,shape(C))
            end if
            ! V * T2 -> TRIPLES
            if (all(rank_vec == (/4,4,6/))) then
               call einsum446(str,A,shape(A),B,shape(B),C,shape(C))
            end if

            ! THINGS FOR PARTIAL CONTRACTIONS 
            if (all(rank_vec == (/2,3,3/))) then
               call einsum233(str,A,shape(A),B,shape(B),C,shape(C))
            end if 
            if (all(rank_vec == (/3,2,3/))) then
               call einsum323(str,A,shape(A),B,shape(B),C,shape(C))
            end if 
            
        end subroutine einsum

        subroutine einsum222(str,A,szA,B,szB,C,szC)

            character, intent(in) :: str(1:9)
            integer, intent(in) :: szA(1:2), szB(1:2), szC(1:2)
            real, intent(in) :: A(szA(1),szA(2)), B(szB(1),szB(2))
            real, intent(out) :: C(szC(1),szC(2))
            real, allocatable :: A2(:,:), B2(:,:), C2(:,:),&
                                 Ap(:,:), Bp(:,:), Cp(:,:)
            character :: s1(1:2), s2(1:2), s3(1:2)
            integer :: i, idx, ct1, ct2, ct3, id, &
                       idxA(1:2), idxB(1:2), idxC(1:2), idxC2(1:2),&
                       idxA2(1:2), idxB2(1:2), idxC3(1:2)
            integer :: shapeA(1:2), shapeB(1:2), shapeC(1:2), n1, n2, n3
            integer :: temp1(1:2), temp2(1:2)

            s1 = str(1:2); s2 = str(4:5); s3 = str(8:);
            shapeA = shape(A); shapeB = shape(B); shapeC = shape(C);
            ct1 = 1; ct2 = 1; ct3 = 1;
            n1 = 1; n2 = 1; n3 = 1;

            do i = 1,2
                idx = findloc(s2,s1(i),1)
                if (idx == 0) then ! i is an output index
                    idx = findloc(s3,s1(i),1)
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                    idxA(ct1) = i 
                    n1 = n1 * shapeA(i)
                    ct1 = ct1 + 1 
                else ! i is contracted
                    idxB(ct2) = idx 
                    ct2 = ct2 + 1
                end if
            end do

            do i = 1,2
                idx = findloc(s3,s2(i),1)
                if (idx /= 0) then ! idx is an output index
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                end if
            end do

            temp1 = (/1,2/)
            ct3 = 1
            do i = 1,2
                if (any(temp1(i) == idxA(1:ct1-1))) then
                    cycle
                else 
                    idxA(ct1+ct3-1) = i 
                    ct3 = ct3 + 1
                    n2 = n2 * shapeA(i)
                end if
            end do

            temp2 = (/1,2/)
            ct3 = 1
            do i = 1,2
                if (any(temp2(i) == idxB(1:ct2-1))) then 
                    cycle 
                else 
                    idxB(ct2+ct3-1) = i 
                    ct3 = ct3 + 1 
                    n3 = n3 * shapeB(i)
                end if 
            end do

            shapeA = shapeA(idxA)
            shapeB = shapeB(idxB)
            shapeC = shapeC(idxC)

            allocate(Ap(shapeA(1),shapeA(2)))
            allocate(Bp(shapeB(1),shapeB(2)))
            allocate(Cp(shapeC(1),shapeC(2)))
            allocate(A2(n1,n2),B2(n2,n3),C2(n1,n3))

            ! find order of A
            do i = 1,2
                id = findloc(idxA,i,1)
                idxA2(i) = id
            end do
            Ap = reshape(A,shape=shapeA,order=idxA2)

            ! find order of B
            do i = 1,2
                id = findloc(idxB,i,1)
                idxB2(i) = id
            end do
            Bp = reshape(B,shape=shapeB,order=idxB2) 

            A2 = reshape(Ap,shape=(/n1,n2/))
            B2 = reshape(Bp,shape=(/n2,n3/))
            call gemm(A2,B2,C2)
            !C2 = gemm(A2,B2)

            ! find order of C
            idxC2 = argsort_int(idxC)
            do i = 1,2
                id = findloc(idxC2,i,1)
                idxC3(i) = id
            end do
            C = reshape(C2,shape=shapeC(idxC2),order=idxC3)
        
        deallocate(Ap,Bp,A2,B2,C2)

        end subroutine einsum222

        subroutine einsum264(str,A,szA,B,szB,C,szC)

            character, intent(in) :: str(1:15)
            integer, intent(in) :: szA(1:2), szB(1:6), szC(1:4)
            real, intent(in) :: A(szA(1),szA(2)), B(szB(1),szB(2),szB(3),szB(4),szB(5),szB(6))
            real, intent(out) :: C(szC(1),szC(2),szC(3),szC(4))
            real, allocatable :: A2(:,:), B2(:,:), C2(:,:),&
                                 Ap(:,:), Bp(:,:,:,:,:,:), Cp(:,:,:,:)
            character :: s1(1:2), s2(1:6), s3(1:4)
            integer :: i, idx, ct1, ct2, ct3, id, &
                       idxA(1:2), idxB(1:6), idxC(1:4), idxC2(1:4),&
                       idxA2(1:2), idxB2(1:6), idxC3(1:4)
            integer :: shapeA(1:2), shapeB(1:6), shapeC(1:4), n1, n2, n3
            integer :: temp1(1:2), temp2(1:6)

            s1 = str(1:2); s2 = str(4:9); s3 = str(12:);
            shapeA = shape(A); shapeB = shape(B); shapeC = shape(C);
            ct1 = 1; ct2 = 1; ct3 = 1;
            n1 = 1; n2 = 1; n3 = 1;

            do i = 1,2
                idx = findloc(s2,s1(i),1)
                if (idx == 0) then ! i is an output index
                    idx = findloc(s3,s1(i),1)
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                    idxA(ct1) = i 
                    n1 = n1 * shapeA(i)
                    ct1 = ct1 + 1 
                else ! i is contracted
                    idxB(ct2) = idx 
                    ct2 = ct2 + 1
                end if
            end do

            do i = 1,6
                idx = findloc(s3,s2(i),1)
                if (idx /= 0) then ! idx is an output index
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                end if
            end do

            temp1 = (/1,2/)
            ct3 = 1
            do i = 1,2
                if (any(temp1(i) == idxA(1:ct1-1))) then
                    cycle
                else 
                    idxA(ct1+ct3-1) = i 
                    ct3 = ct3 + 1
                    n2 = n2 * shapeA(i)
                end if
            end do

            temp2 = (/1,2,3,4,5,6/)
            ct3 = 1
            do i = 1,6
                if (any(temp2(i) == idxB(1:ct2-1))) then 
                    cycle 
                else 
                    idxB(ct2+ct3-1) = i 
                    ct3 = ct3 + 1 
                    n3 = n3 * shapeB(i)
                end if 
            end do

            shapeA = shapeA(idxA)
            shapeB = shapeB(idxB)
            shapeC = shapeC(idxC)

            allocate(Ap(shapeA(1),shapeA(2)))
            allocate(Bp(shapeB(1),shapeB(2),shapeB(3),shapeB(4),shapeB(5),shapeB(6)))
            allocate(Cp(shapeC(1),shapeC(2),shapeC(3),shapeC(4)))
            allocate(A2(n1,n2),B2(n2,n3),C2(n1,n3))

            ! find order of A
            do i = 1,2
                id = findloc(idxA,i,1)
                idxA2(i) = id
            end do
            Ap = reshape(A,shape=shapeA,order=idxA2)

            ! find order of B
            do i = 1,6
                id = findloc(idxB,i,1)
                idxB2(i) = id
            end do
            Bp = reshape(B,shape=shapeB,order=idxB2) 

            A2 = reshape(Ap,shape=(/n1,n2/))
            B2 = reshape(Bp,shape=(/n2,n3/))
            call gemm(A2,B2,C2)

            ! find order of C
            idxC2 = argsort_int(idxC)
            do i = 1,4
                id = findloc(idxC2,i,1)
                idxC3(i) = id
            end do
            C = reshape(C2,shape=shapeC(idxC2),order=idxC3)
        
        deallocate(Ap,Bp,A2,B2,C2)

        end subroutine einsum264

        subroutine einsum266(str,A,szA,B,szB,C,szC)

            character, intent(in) :: str(1:17)
            integer, intent(in) :: szA(1:2), szB(1:6), szC(1:6)
            real, intent(in) :: A(szA(1),szA(2)), B(szB(1),szB(2),szB(3),szB(4),szB(5),szB(6))
            real, intent(out) :: C(szC(1),szC(2),szC(3),szC(4),szC(5),szC(6))
            real, allocatable :: A2(:,:), B2(:,:), C2(:,:),&
                                 Ap(:,:), Bp(:,:,:,:,:,:), Cp(:,:,:,:)
            character :: s1(1:2), s2(1:6), s3(1:6)
            integer :: i, idx, ct1, ct2, ct3, id, &
                       idxA(1:2), idxB(1:6), idxC(1:6), idxC2(1:6),&
                       idxA2(1:2), idxB2(1:6), idxC3(1:6)
            integer :: shapeA(1:2), shapeB(1:6), shapeC(1:6), n1, n2, n3
            integer :: temp1(1:2), temp2(1:6)

            s1 = str(1:2); s2 = str(4:9); s3 = str(12:);
            shapeA = shape(A); shapeB = shape(B); shapeC = shape(C);
            ct1 = 1; ct2 = 1; ct3 = 1;
            n1 = 1; n2 = 1; n3 = 1;

            do i = 1,2
                idx = findloc(s2,s1(i),1)
                if (idx == 0) then ! i is an output index
                    idx = findloc(s3,s1(i),1)
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                    idxA(ct1) = i 
                    n1 = n1 * shapeA(i)
                    ct1 = ct1 + 1 
                else ! i is contracted
                    idxB(ct2) = idx 
                    ct2 = ct2 + 1
                end if
            end do

            do i = 1,6
                idx = findloc(s3,s2(i),1)
                if (idx /= 0) then ! idx is an output index
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                end if
            end do

            temp1 = (/1,2/)
            ct3 = 1
            do i = 1,2
                if (any(temp1(i) == idxA(1:ct1-1))) then
                    cycle
                else 
                    idxA(ct1+ct3-1) = i 
                    ct3 = ct3 + 1
                    n2 = n2 * shapeA(i)
                end if
            end do

            temp2 = (/1,2,3,4,5,6/)
            ct3 = 1
            do i = 1,6
                if (any(temp2(i) == idxB(1:ct2-1))) then 
                    cycle 
                else 
                    idxB(ct2+ct3-1) = i 
                    ct3 = ct3 + 1 
                    n3 = n3 * shapeB(i)
                end if 
            end do

            shapeA = shapeA(idxA)
            shapeB = shapeB(idxB)
            shapeC = shapeC(idxC)

            allocate(Ap(shapeA(1),shapeA(2)))
            allocate(Bp(shapeB(1),shapeB(2),shapeB(3),shapeB(4),shapeB(5),shapeB(6)))
            allocate(Cp(shapeC(1),shapeC(2),shapeC(3),shapeC(4)))
            allocate(A2(n1,n2),B2(n2,n3),C2(n1,n3))

            ! find order of A
            do i = 1,2
                id = findloc(idxA,i,1)
                idxA2(i) = id
            end do
            Ap = reshape(A,shape=shapeA,order=idxA2)

            ! find order of B
            do i = 1,6
                id = findloc(idxB,i,1)
                idxB2(i) = id
            end do
            Bp = reshape(B,shape=shapeB,order=idxB2) 

            A2 = reshape(Ap,shape=(/n1,n2/))
            B2 = reshape(Bp,shape=(/n2,n3/))
            call gemm(A2,B2,C2)

            ! find order of C
            idxC2 = argsort_int(idxC)
            do i = 1,6
                id = findloc(idxC2,i,1)
                idxC3(i) = id
            end do
            C = reshape(C2,shape=shapeC(idxC2),order=idxC3)
        
        deallocate(Ap,Bp,A2,B2,C2)

        end subroutine einsum266

        subroutine einsum422(str,A,szA,B,szB,C,szC)

            character, intent(in) :: str(1:11)
            integer, intent(in) :: szA(1:4), szB(1:2), szC(1:2)
            real, intent(in) :: A(szA(1),szA(2),szA(3),szA(4)), B(szB(1),szB(2))
            real, intent(out) :: C(szC(1),szC(2))
            real, allocatable :: A2(:,:), B2(:,:), C2(:,:),&
                                 Ap(:,:,:,:), Bp(:,:), Cp(:,:)
            character :: s1(1:4), s2(1:2), s3(1:2)
            integer :: i, idx, ct1, ct2, ct3, id, &
                       idxA(1:4), idxB(1:2), idxC(1:2), idxC2(1:2),&
                       idxA2(1:4), idxB2(1:2), idxC3(1:2)
            integer :: shapeA(1:4), shapeB(1:2), shapeC(1:2), n1, n2, n3
            integer :: temp1(1:4), temp2(1:2)

            s1 = str(1:4); s2 = str(6:7); s3 = str(10:);
            shapeA = shape(A); shapeB = shape(B); shapeC = shape(C);
            ct1 = 1; ct2 = 1; ct3 = 1;
            n1 = 1; n2 = 1; n3 = 1;

            do i = 1,4
                idx = findloc(s2,s1(i),1)
                if (idx == 0) then ! i is an output index
                    idx = findloc(s3,s1(i),1)
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                    idxA(ct1) = i 
                    n1 = n1 * shapeA(i)
                    ct1 = ct1 + 1 
                else ! i is contracted
                    idxB(ct2) = idx 
                    ct2 = ct2 + 1
                end if
            end do

            do i = 1,2
                idx = findloc(s3,s2(i),1)
                if (idx /= 0) then ! idx is an output index
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                end if
            end do

            temp1 = (/1,2,3,4/)
            ct3 = 1
            do i = 1,4
                if (any(temp1(i) == idxA(1:ct1-1))) then
                    cycle
                else 
                    idxA(ct1+ct3-1) = i 
                    ct3 = ct3 + 1
                    n2 = n2 * shapeA(i)
                end if
            end do

            temp2 = (/1,2/)
            ct3 = 1
            do i = 1,2
                if (any(temp2(i) == idxB(1:ct2-1))) then 
                    cycle 
                else 
                    idxB(ct2+ct3-1) = i 
                    ct3 = ct3 + 1 
                    n3 = n3 * shapeB(i)
                end if 
            end do

            shapeA = shapeA(idxA)
            shapeB = shapeB(idxB)
            shapeC = shapeC(idxC)

            allocate(Ap(shapeA(1),shapeA(2),shapeA(3),shapeA(4)))
            allocate(Bp(shapeB(1),shapeB(2)))
            allocate(Cp(shapeC(1),shapeC(2)))
            allocate(A2(n1,n2),B2(n2,n3),C2(n1,n3))

            ! find order of A
            do i = 1,4
                id = findloc(idxA,i,1)
                idxA2(i) = id
            end do
            Ap = reshape(A,shape=shapeA,order=idxA2)

            ! find order of B
            do i = 1,2
                id = findloc(idxB,i,1)
                idxB2(i) = id
            end do
            Bp = reshape(B,shape=shapeB,order=idxB2) 

            A2 = reshape(Ap,shape=(/n1,n2/))
            B2 = reshape(Bp,shape=(/n2,n3/))
            call gemm(A2,B2,C2)

            ! find order of C
            idxC2 = argsort_int(idxC)
            do i = 1,2
                id = findloc(idxC2,i,1)
                idxC3(i) = id
            end do
            C = reshape(C2,shape=shapeC(idxC2),order=idxC3)
    
        deallocate(Ap,Bp,A2,B2,C2)

        end subroutine einsum422

        subroutine einsum242(str,A,szA,B,szB,C,szC)

            character, intent(in) :: str(1:11)
            integer, intent(in) :: szA(1:2), szB(1:4), szC(1:2)
            real, intent(in) :: A(szA(1),szA(2)), B(szB(1),szB(2),szB(3),szB(4))
            real, intent(out) :: C(szC(1),szC(2))
            real, allocatable :: A2(:,:), B2(:,:), C2(:,:),&
                                 Ap(:,:), Bp(:,:,:,:), Cp(:,:)
            character :: s1(1:2), s2(1:4), s3(1:2)
            integer :: i, idx, ct1, ct2, ct3, id, &
                       idxA(1:2), idxB(1:4), idxC(1:2), idxC2(1:2),&
                       idxA2(1:2), idxB2(1:4), idxC3(1:2)
            integer :: shapeA(1:2), shapeB(1:4), shapeC(1:2), n1, n2, n3
            integer :: temp1(1:2), temp2(1:4)

            s1 = str(1:2); s2 = str(4:7); s3 = str(10:);
            shapeA = shape(A); shapeB = shape(B); shapeC = shape(C);
            ct1 = 1; ct2 = 1; ct3 = 1;
            n1 = 1; n2 = 1; n3 = 1;

            do i = 1,2
                idx = findloc(s2,s1(i),1)
                if (idx == 0) then ! i is an output index
                    idx = findloc(s3,s1(i),1)
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                    idxA(ct1) = i 
                    n1 = n1 * shapeA(i)
                    ct1 = ct1 + 1 
                else ! i is contracted
                    idxB(ct2) = idx 
                    ct2 = ct2 + 1
                end if
            end do

            do i = 1,4
                idx = findloc(s3,s2(i),1)
                if (idx /= 0) then ! idx is an output index
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                end if
            end do

            temp1 = (/1,2/)
            ct3 = 1
            do i = 1,2
                if (any(temp1(i) == idxA(1:ct1-1))) then
                    cycle
                else 
                    idxA(ct1+ct3-1) = i 
                    ct3 = ct3 + 1
                    n2 = n2 * shapeA(i)
                end if
            end do

            temp2 = (/1,2,3,4/)
            ct3 = 1
            do i = 1,4
                if (any(temp2(i) == idxB(1:ct2-1))) then 
                    cycle 
                else 
                    idxB(ct2+ct3-1) = i 
                    ct3 = ct3 + 1 
                    n3 = n3 * shapeB(i)
                end if 
            end do

            shapeA = shapeA(idxA)
            shapeB = shapeB(idxB)
            shapeC = shapeC(idxC)

            allocate(Ap(shapeA(1),shapeA(2)))
            allocate(Bp(shapeB(1),shapeB(2),shapeB(3),shapeB(4)))
            allocate(Cp(shapeC(1),shapeC(2)))
            allocate(A2(n1,n2),B2(n2,n3),C2(n1,n3))

            ! find order of A
            do i = 1,2
                id = findloc(idxA,i,1)
                idxA2(i) = id
            end do
            Ap = reshape(A,shape=shapeA,order=idxA2)

            ! find order of B
            do i = 1,4
                id = findloc(idxB,i,1)
                idxB2(i) = id
            end do
            Bp = reshape(B,shape=shapeB,order=idxB2) 

            A2 = reshape(Ap,shape=(/n1,n2/))
            B2 = reshape(Bp,shape=(/n2,n3/))
            call gemm(A2,B2,C2)

            ! find order of C
            idxC2 = argsort_int(idxC)
            do i = 1,2
                id = findloc(idxC2,i,1)
                idxC3(i) = id
            end do
            C = reshape(C2,shape=shapeC(idxC2),order=idxC3)
    
        deallocate(Ap,Bp,A2,B2,C2)

        end subroutine einsum242

        subroutine einsum424(str,A,szA,B,szB,C,szC)

            character, intent(in) :: str(1:13)
            integer, intent(in) :: szA(1:4), szB(1:2), szC(1:4)
            real, intent(in) :: A(szA(1),szA(2),szA(3),szA(4)), B(szB(1),szB(2))
            real, intent(out) :: C(szC(1),szC(2),szC(3),szC(4))
            real, allocatable :: A2(:,:), B2(:,:), C2(:,:),&
                                 Ap(:,:,:,:), Bp(:,:), Cp(:,:,:,:)
            character :: s1(1:4), s2(1:2), s3(1:4)
            integer :: i, idx, ct1, ct2, ct3, id, &
                       idxA(1:4), idxB(1:2), idxC(1:4), idxC2(1:4),&
                       idxA2(1:4), idxB2(1:2), idxC3(1:4)
            integer :: shapeA(1:4), shapeB(1:2), shapeC(1:4), n1, n2, n3
            integer :: temp1(1:4), temp2(1:2)

            s1 = str(1:4); s2 = str(6:7); s3 = str(10:);
            shapeA = shape(A); shapeB = shape(B); shapeC = shape(C);
            ct1 = 1; ct2 = 1; ct3 = 1;
            n1 = 1; n2 = 1; n3 = 1;

            do i = 1,4
                idx = findloc(s2,s1(i),1)
                if (idx == 0) then ! i is an output index
                    idx = findloc(s3,s1(i),1)
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                    idxA(ct1) = i 
                    n1 = n1 * shapeA(i)
                    ct1 = ct1 + 1 
                else ! i is contracted
                    idxB(ct2) = idx 
                    ct2 = ct2 + 1
                end if
            end do

            do i = 1,2
                idx = findloc(s3,s2(i),1)
                if (idx /= 0) then ! idx is an output index
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                end if
            end do

            temp1 = (/1,2,3,4/)
            ct3 = 1
            do i = 1,4
                if (any(temp1(i) == idxA(1:ct1-1))) then
                    cycle
                else 
                    idxA(ct1+ct3-1) = i 
                    ct3 = ct3 + 1
                    n2 = n2 * shapeA(i)
                end if
            end do

            temp2 = (/1,2/)
            ct3 = 1
            do i = 1,2
                if (any(temp2(i) == idxB(1:ct2-1))) then 
                    cycle 
                else 
                    idxB(ct2+ct3-1) = i 
                    ct3 = ct3 + 1 
                    n3 = n3 * shapeB(i)
                end if 
            end do

            shapeA = shapeA(idxA)
            shapeB = shapeB(idxB)
            shapeC = shapeC(idxC)

            allocate(Ap(shapeA(1),shapeA(2),shapeA(3),shapeA(4)))
            allocate(Bp(shapeB(1),shapeB(2)))
            allocate(Cp(shapeC(1),shapeC(2),shapeC(3),shapeC(4)))
            allocate(A2(n1,n2),B2(n2,n3),C2(n1,n3))

            ! find order of A
            do i = 1,4
                id = findloc(idxA,i,1)
                idxA2(i) = id
            end do
            Ap = reshape(A,shape=shapeA,order=idxA2)

            ! find order of B
            do i = 1,2
                id = findloc(idxB,i,1)
                idxB2(i) = id
            end do
            Bp = reshape(B,shape=shapeB,order=idxB2) 

            A2 = reshape(Ap,shape=(/n1,n2/))
            B2 = reshape(Bp,shape=(/n2,n3/))
            call gemm(A2,B2,C2)

            ! find order of C
            idxC2 = argsort_int(idxC)
            do i = 1,4
                id = findloc(idxC2,i,1)
                idxC3(i) = id
            end do
            C = reshape(C2,shape=shapeC(idxC2),order=idxC3)

        deallocate(Ap,Bp,A2,B2,C2)

        end subroutine einsum424

        subroutine einsum244(str,A,szA,B,szB,C,szC)

            character, intent(in) :: str(1:13)
            integer, intent(in) :: szA(1:2), szB(1:4), szC(1:4)
            real, intent(in) :: A(szA(1),szA(2)), B(szB(1),szB(2),szB(3),szB(4))
            real, intent(out) :: C(szC(1),szC(2),szC(3),szC(4))
            real, allocatable :: A2(:,:), B2(:,:), C2(:,:),&
                                 Ap(:,:), Bp(:,:,:,:), Cp(:,:,:,:)
            character :: s1(1:2), s2(1:4), s3(1:4)
            integer :: i, idx, ct1, ct2, ct3, id, &
                       idxA(1:2), idxB(1:4), idxC(1:4), idxC2(1:4),&
                       idxA2(1:2), idxB2(1:4), idxC3(1:4)
            integer :: shapeA(1:2), shapeB(1:4), shapeC(1:4), n1, n2, n3
            integer :: temp1(1:2), temp2(1:4)

            s1 = str(1:2); s2 = str(4:7); s3 = str(10:);
            shapeA = shape(A); shapeB = shape(B); shapeC = shape(C);
            ct1 = 1; ct2 = 1; ct3 = 1;
            n1 = 1; n2 = 1; n3 = 1;

            do i = 1,2
                idx = findloc(s2,s1(i),1)
                if (idx == 0) then ! i is an output index
                    idx = findloc(s3,s1(i),1)
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                    idxA(ct1) = i 
                    n1 = n1 * shapeA(i)
                    ct1 = ct1 + 1 
                else ! i is contracted
                    idxB(ct2) = idx 
                    ct2 = ct2 + 1
                end if
            end do

            do i = 1,4
                idx = findloc(s3,s2(i),1)
                if (idx /= 0) then ! idx is an output index
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                end if
            end do

            temp1 = (/1,2/)
            ct3 = 1
            do i = 1,2
                if (any(temp1(i) == idxA(1:ct1-1))) then
                    cycle
                else 
                    idxA(ct1+ct3-1) = i 
                    ct3 = ct3 + 1
                    n2 = n2 * shapeA(i)
                end if
            end do

            temp2 = (/1,2,3,4/)
            ct3 = 1
            do i = 1,4
                if (any(temp2(i) == idxB(1:ct2-1))) then 
                    cycle 
                else 
                    idxB(ct2+ct3-1) = i 
                    ct3 = ct3 + 1 
                    n3 = n3 * shapeB(i)
                end if 
            end do

            shapeA = shapeA(idxA)
            shapeB = shapeB(idxB)
            shapeC = shapeC(idxC)

            allocate(Ap(shapeA(1),shapeA(2)))
            allocate(Bp(shapeB(1),shapeB(2),shapeB(3),shapeB(4)))
            allocate(Cp(shapeC(1),shapeC(2),shapeC(3),shapeC(4)))
            allocate(A2(n1,n2),B2(n2,n3),C2(n1,n3))

            ! find order of A
            do i = 1,2
                id = findloc(idxA,i,1)
                idxA2(i) = id
            end do
            Ap = reshape(A,shape=shapeA,order=idxA2)

            ! find order of B
            do i = 1,4
                id = findloc(idxB,i,1)
                idxB2(i) = id
            end do
            Bp = reshape(B,shape=shapeB,order=idxB2) 

            A2 = reshape(Ap,shape=(/n1,n2/))
            B2 = reshape(Bp,shape=(/n2,n3/))
            call gemm(A2,B2,C2)

            ! find order of C
            idxC2 = argsort_int(idxC)
            do i = 1,4
                id = findloc(idxC2,i,1)
                idxC3(i) = id
            end do
            C = reshape(C2,shape=shapeC(idxC2),order=idxC3)


        deallocate(Ap,Bp,A2,B2,C2)

        end subroutine einsum244

        subroutine einsum444(str,A,szA,B,szB,C,szC)

            character, intent(in) :: str(1:15)
            integer, intent(in) :: szA(1:4), szB(1:4), szC(1:4)
            real, intent(in) :: A(szA(1),szA(2),szA(3),szA(4)), B(szB(1),szB(2),szB(3),szB(4))
            real, intent(out) :: C(szC(1),szC(2),szC(3),szC(4))
            real, allocatable :: A2(:,:), B2(:,:), C2(:,:),&
                                 Ap(:,:,:,:), Bp(:,:,:,:), Cp(:,:,:,:)
            character :: s1(1:4), s2(1:4), s3(1:4)
            integer :: i, idx, ct1, ct2, ct3, id, &
                       idxA(1:4), idxB(1:4), idxC(1:4), idxC2(1:4),&
                       idxA2(1:4), idxB2(1:4), idxC3(1:4)
            integer :: shapeA(1:4), shapeB(1:4), shapeC(1:4), n1, n2, n3
            integer :: temp1(1:4), temp2(1:4)

            s1 = str(1:4); s2 = str(6:9); s3 = str(12:);
            shapeA = shape(A); shapeB = shape(B); shapeC = shape(C);
            ct1 = 1; ct2 = 1; ct3 = 1;
            n1 = 1; n2 = 1; n3 = 1;

            do i = 1,4
                idx = findloc(s2,s1(i),1)
                if (idx == 0) then ! i is an output index
                    idx = findloc(s3,s1(i),1)
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                    idxA(ct1) = i 
                    n1 = n1 * shapeA(i)
                    ct1 = ct1 + 1 
                else ! i is contracted
                    idxB(ct2) = idx 
                    ct2 = ct2 + 1
                end if
            end do

            do i = 1,4
                idx = findloc(s3,s2(i),1)
                if (idx /= 0) then ! idx is an output index
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                end if
            end do

            temp1 = (/1,2,3,4/)
            ct3 = 1
            do i = 1,4
                if (any(temp1(i) == idxA(1:ct1-1))) then
                    cycle
                else 
                    idxA(ct1+ct3-1) = i 
                    ct3 = ct3 + 1
                    n2 = n2 * shapeA(i)
                end if
            end do

            temp2 = (/1,2,3,4/)
            ct3 = 1
            do i = 1,4 
                if (any(temp2(i) == idxB(1:ct2-1))) then 
                    cycle 
                else 
                    idxB(ct2+ct3-1) = i 
                    ct3 = ct3 + 1 
                    n3 = n3 * shapeB(i)
                end if 
            end do

            shapeA = shapeA(idxA)
            shapeB = shapeB(idxB)
            shapeC = shapeC(idxC)

            allocate(Ap(shapeA(1),shapeA(2),shapeA(3),shapeA(4)))
            allocate(Bp(shapeB(1),shapeB(2),shapeB(3),shapeB(4)))
            allocate(Cp(shapeC(1),shapeC(2),shapeC(3),shapeC(4)))
            allocate(A2(n1,n2),B2(n2,n3),C2(n1,n3))

            ! find order of A
            do i = 1,4
                id = findloc(idxA,i,1)
                idxA2(i) = id
            end do
            Ap = reshape(A,shape=shapeA,order=idxA2)

            ! find order of B
            do i = 1,4
                id = findloc(idxB,i,1)
                idxB2(i) = id
            end do
            Bp = reshape(B,shape=shapeB,order=idxB2) 

            A2 = reshape(Ap,shape=(/n1,n2/))
            B2 = reshape(Bp,shape=(/n2,n3/))

            call gemm(A2,B2,C2)

            ! Cp = reshape(C2,shape=shapeC)
            ! idxC2 = argsort_int(idxC)
            ! C = reshape(Cp,shape=shapeC(idxC2),order=idxC2)
            
            ! find order of C
            idxC2 = argsort_int(idxC)
            do i = 1,4
                id = findloc(idxC2,i,1)
                idxC3(i) = id
            end do
            C = reshape(C2,shape=shapeC(idxC2),order=idxC3)
        
        deallocate(Ap,Bp,A2,B2,C2)

        end subroutine einsum444

        subroutine einsum442(str,A,szA,B,szB,C,szC)

            character, intent(in) :: str(1:13)
            integer, intent(in) :: szA(1:4), szB(1:4), szC(1:2)
            real, intent(in) :: A(szA(1),szA(2),szA(3),szA(4)), B(szB(1),szB(2),szB(3),szB(4))
            real, intent(out) :: C(szC(1),szC(2))
            real, allocatable :: A2(:,:), B2(:,:), C2(:,:),&
                                 Ap(:,:,:,:), Bp(:,:,:,:), Cp(:,:)
            character :: s1(1:4), s2(1:4), s3(1:2)
            integer :: i, idx, ct1, ct2, ct3, id, &
                       idxA(1:4), idxB(1:4), idxC(1:2), idxC2(1:2),&
                       idxA2(1:4), idxB2(1:4), idxC3(1:2)
            integer :: shapeA(1:4), shapeB(1:4), shapeC(1:2), n1, n2, n3
            integer :: temp1(1:4), temp2(1:4)

            s1 = str(1:4); s2 = str(6:9); s3 = str(12:);
            shapeA = shape(A); shapeB = shape(B); shapeC = shape(C);
            ct1 = 1; ct2 = 1; ct3 = 1;
            n1 = 1; n2 = 1; n3 = 1;

            do i = 1,4
                idx = findloc(s2,s1(i),1)
                if (idx == 0) then ! i is an output index
                    idx = findloc(s3,s1(i),1)
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                    idxA(ct1) = i 
                    n1 = n1 * shapeA(i)
                    ct1 = ct1 + 1 
                else ! i is contracted
                    idxB(ct2) = idx 
                    ct2 = ct2 + 1
                end if
            end do

            do i = 1,4
                idx = findloc(s3,s2(i),1)
                if (idx /= 0) then ! idx is an output index
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                end if
            end do

            temp1 = (/1,2,3,4/)
            ct3 = 1
            do i = 1,4
                if (any(temp1(i) == idxA(1:ct1-1))) then
                    cycle
                else 
                    idxA(ct1+ct3-1) = i 
                    ct3 = ct3 + 1
                    n2 = n2 * shapeA(i)
                end if
            end do

            temp2 = (/1,2,3,4/)
            ct3 = 1
            do i = 1,4 
                if (any(temp2(i) == idxB(1:ct2-1))) then 
                    cycle 
                else 
                    idxB(ct2+ct3-1) = i 
                    ct3 = ct3 + 1 
                    n3 = n3 * shapeB(i)
                end if 
            end do

            shapeA = shapeA(idxA)
            shapeB = shapeB(idxB)
            shapeC = shapeC(idxC)

            allocate(Ap(shapeA(1),shapeA(2),shapeA(3),shapeA(4)))
            allocate(Bp(shapeB(1),shapeB(2),shapeB(3),shapeB(4)))
            allocate(Cp(shapeC(1),shapeC(2)))
            allocate(A2(n1,n2),B2(n2,n3),C2(n1,n3))

            ! find order of A
            do i = 1,4
                id = findloc(idxA,i,1)
                idxA2(i) = id
            end do
            Ap = reshape(A,shape=shapeA,order=idxA2)

            ! find order of B
            do i = 1,4
                id = findloc(idxB,i,1)
                idxB2(i) = id
            end do
            Bp = reshape(B,shape=shapeB,order=idxB2) 

            A2 = reshape(Ap,shape=(/n1,n2/))
            B2 = reshape(Bp,shape=(/n2,n3/))
            call gemm(A2,B2,C2)

            ! Cp = reshape(C2,shape=shapeC)
            ! idxC2 = argsort_int(idxC)
            ! C = reshape(Cp,shape=shapeC(idxC2),order=idxC2)
            
            ! find order of C
            idxC2 = argsort_int(idxC)
            do i = 1,2
                id = findloc(idxC2,i,1)
                idxC3(i) = id
            end do
            C = reshape(C2,shape=shapeC(idxC2),order=idxC3)
        
        deallocate(Ap,Bp,A2,B2,C2)

        end subroutine einsum442

        !subroutine einsum446(str,A,szA,B,szB,C,szC)

        !    character, intent(in) :: str(1:17)
        !    integer, intent(in) :: szA(1:4), szB(1:4), szC(1:6)
        !    real, intent(in) :: A(szA(1),szA(2),szA(3),szA(4)), B(szB(1),szB(2),szB(3),szB(4))
        !    real, intent(out) :: C(szC(1),szC(2),szC(3),szC(4),szC(5),szC(6))
        !    real, allocatable :: A2(:,:), B2(:,:), C2(:,:),&
        !                         Ap(:,:,:,:), Bp(:,:,:,:), Cp(:,:,:,:,:,:)
        !    character :: s1(1:4), s2(1:4), s3(1:6)
        !    integer :: i, idx, ct1, ct2, ct3, id, &
        !               idxA(1:4), idxB(1:4), idxC(1:6), idxC2(1:6),&
        !               idxA2(1:4), idxB2(1:4), idxC3(1:6)
        !    integer :: shapeA(1:4), shapeB(1:4), shapeC(1:6), n1, n2, n3
        !    integer :: temp1(1:4), temp2(1:4)

        !    s1 = str(1:4); s2 = str(6:9); s3 = str(12:);
        !    shapeA = shape(A); shapeB = shape(B); shapeC = shape(C);
        !    ct1 = 1; ct2 = 1; ct3 = 1;
        !    n1 = 1; n2 = 1; n3 = 1;

        !    do i = 1,4
        !        idx = findloc(s2,s1(i),1)
        !        if (idx == 0) then ! i is an output index
        !            idx = findloc(s3,s1(i),1)
        !            idxC(ct3) = idx
        !            ct3 = ct3 + 1
        !            idxA(ct1) = i 
        !            n1 = n1 * shapeA(i)
        !            ct1 = ct1 + 1 
        !        else ! i is contracted
        !            idxB(ct2) = idx 
        !            ct2 = ct2 + 1
        !        end if
        !    end do

        !    do i = 1,4
        !        idx = findloc(s3,s2(i),1)
        !        if (idx /= 0) then ! idx is an output index
        !            idxC(ct3) = idx
        !            ct3 = ct3 + 1
        !        end if
        !    end do

        !    temp1 = (/1,2,3,4/)
        !    ct3 = 1
        !    do i = 1,4
        !        if (any(temp1(i) == idxA(1:ct1-1))) then
        !            cycle
        !        else 
        !            idxA(ct1+ct3-1) = i 
        !            ct3 = ct3 + 1
        !            n2 = n2 * shapeA(i)
        !        end if
        !    end do

        !    temp2 = (/1,2,3,4/)
        !    ct3 = 1
        !    do i = 1,4 
        !        if (any(temp2(i) == idxB(1:ct2-1))) then 
        !            cycle 
        !        else 
        !            idxB(ct2+ct3-1) = i 
        !            ct3 = ct3 + 1 
        !            n3 = n3 * shapeB(i)
        !        end if 
        !    end do

        !    shapeA = shapeA(idxA)
        !    shapeB = shapeB(idxB)
        !    shapeC = shapeC(idxC)

        !    allocate(Ap(shapeA(1),shapeA(2),shapeA(3),shapeA(4)))
        !    allocate(Bp(shapeB(1),shapeB(2),shapeB(3),shapeB(4)))
        !    allocate(Cp(shapeC(1),shapeC(2),shapeC(3),shapeC(4),shapeC(5),shapeC(6)))
        !    allocate(A2(n1,n2),B2(n2,n3),C2(n1,n3))

            ! find order of A
        !    do i = 1,4
        !        id = findloc(idxA,i,1)
        !        idxA2(i) = id
        !    end do
        !    Ap = reshape(A,shape=shapeA,order=idxA2)

            ! find order of B
        !    do i = 1,4
        !        id = findloc(idxB,i,1)
        !        idxB2(i) = id
        !    end do
        !    Bp = reshape(B,shape=shapeB,order=idxB2) 

        !    A2 = reshape(Ap,shape=(/n1,n2/))
        !    B2 = reshape(Bp,shape=(/n2,n3/))
        !    call gemm(A2,B2,C2)

        !    ! Cp = reshape(C2,shape=shapeC)
        !    ! idxC2 = argsort_int(idxC)
        !    ! C = reshape(Cp,shape=shapeC(idxC2),order=idxC2)
            
            ! find order of C
        !    idxC2 = argsort_int(idxC)
        !    do i = 1,6
        !        id = findloc(idxC2,i,1)
        !        idxC3(i) = id
        !    end do
        !    C = reshape(C2,shape=shapeC(idxC2),order=idxC3)
        
        !deallocate(Ap,Bp,A2,B2,C2)

        !end subroutine einsum446

        subroutine einsum462(str,A,szA,B,szB,C,szC)

            character, intent(in) :: str(1:15)
            integer, intent(in) :: szA(1:4), szB(1:6), szC(1:2)
            real, intent(in) :: A(szA(1),szA(2),szA(3),szA(4)), B(szB(1),szB(2),szB(3),szB(4),szB(5),szB(6))
            real, intent(out) :: C(szC(1),szC(2))
            real, allocatable :: A2(:,:), B2(:,:), C2(:,:),&
                                 Ap(:,:,:,:), Bp(:,:,:,:,:,:), Cp(:,:)
            character :: s1(1:4), s2(1:6), s3(1:2)
            integer :: i, idx, ct1, ct2, ct3, id, &
                       idxA(1:4), idxB(1:6), idxC(1:2), idxC2(1:2),&
                       idxA2(1:4), idxB2(1:6), idxC3(1:2)
            integer :: shapeA(1:4), shapeB(1:6), shapeC(1:2), n1, n2, n3
            integer :: temp1(1:4), temp2(1:6)

            s1 = str(1:4); s2 = str(6:11); s3 = str(14:);
            shapeA = shape(A); shapeB = shape(B); shapeC = shape(C);
            ct1 = 1; ct2 = 1; ct3 = 1;
            n1 = 1; n2 = 1; n3 = 1;

            do i = 1,4
                idx = findloc(s2,s1(i),1)
                if (idx == 0) then ! i is an output index
                    idx = findloc(s3,s1(i),1)
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                    idxA(ct1) = i 
                    n1 = n1 * shapeA(i)
                    ct1 = ct1 + 1 
                else ! i is contracted
                    idxB(ct2) = idx 
                    ct2 = ct2 + 1
                end if
            end do

            do i = 1,6
                idx = findloc(s3,s2(i),1)
                if (idx /= 0) then ! idx is an output index
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                end if
            end do

            temp1 = (/1,2,3,4/)
            ct3 = 1
            do i = 1,4
                if (any(temp1(i) == idxA(1:ct1-1))) then
                    cycle
                else 
                    idxA(ct1+ct3-1) = i 
                    ct3 = ct3 + 1
                    n2 = n2 * shapeA(i)
                end if
            end do

            temp2 = (/1,2,3,4,5,6/)
            ct3 = 1
            do i = 1,6
                if (any(temp2(i) == idxB(1:ct2-1))) then 
                    cycle 
                else 
                    idxB(ct2+ct3-1) = i 
                    ct3 = ct3 + 1 
                    n3 = n3 * shapeB(i)
                end if 
            end do

            shapeA = shapeA(idxA)
            shapeB = shapeB(idxB)
            shapeC = shapeC(idxC)

            allocate(Ap(shapeA(1),shapeA(2),shapeA(3),shapeA(4)))
            allocate(Bp(shapeB(1),shapeB(2),shapeB(3),shapeB(4),shapeB(5),shapeB(6)))
            allocate(Cp(shapeC(1),shapeC(2)))
            allocate(A2(n1,n2),B2(n2,n3),C2(n1,n3))

            ! find order of A
            do i = 1,4
                id = findloc(idxA,i,1)
                idxA2(i) = id
            end do
            Ap = reshape(A,shape=shapeA,order=idxA2)

            ! find order of B
            do i = 1,6
                id = findloc(idxB,i,1)
                idxB2(i) = id
            end do
            Bp = reshape(B,shape=shapeB,order=idxB2) 

            A2 = reshape(Ap,shape=(/n1,n2/))
            B2 = reshape(Bp,shape=(/n2,n3/))
            call gemm(A2,B2,C2)

            ! Cp = reshape(C2,shape=shapeC)
            ! idxC2 = argsort_int(idxC)
            ! C = reshape(Cp,shape=shapeC(idxC2),order=idxC2)
            
            ! find order of C
            idxC2 = argsort_int(idxC)
            do i = 1,2
                id = findloc(idxC2,i,1)
                idxC3(i) = id
            end do
            C = reshape(C2,shape=shapeC(idxC2),order=idxC3)
        
        deallocate(Ap,Bp,A2,B2,C2)

        end subroutine einsum462

        subroutine einsum464(str,A,szA,B,szB,C,szC)

            character, intent(in) :: str(1:17)
            integer, intent(in) :: szA(1:4), szB(1:6), szC(1:4)
            real, intent(in) :: A(szA(1),szA(2),szA(3),szA(4)), B(szB(1),szB(2),szB(3),szB(4),szB(5),szB(6))
            real, intent(out) :: C(szC(1),szC(2),szC(3),szC(4))
            real, allocatable :: A2(:,:), B2(:,:), C2(:,:),&
                                 Ap(:,:,:,:), Bp(:,:,:,:,:,:), Cp(:,:,:,:)
            character :: s1(1:4), s2(1:6), s3(1:4)
            integer :: i, idx, ct1, ct2, ct3, id, &
                       idxA(1:4), idxB(1:6), idxC(1:4), idxC2(1:4),&
                       idxA2(1:4), idxB2(1:6), idxC3(1:4)
            integer :: shapeA(1:4), shapeB(1:6), shapeC(1:4), n1, n2, n3
            integer :: temp1(1:4), temp2(1:6)

            s1 = str(1:4); s2 = str(6:11); s3 = str(14:);
            shapeA = shape(A); shapeB = shape(B); shapeC = shape(C);
            ct1 = 1; ct2 = 1; ct3 = 1;
            n1 = 1; n2 = 1; n3 = 1;

            do i = 1,4
                idx = findloc(s2,s1(i),1)
                if (idx == 0) then ! i is an output index
                    idx = findloc(s3,s1(i),1)
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                    idxA(ct1) = i 
                    n1 = n1 * shapeA(i)
                    ct1 = ct1 + 1 
                else ! i is contracted
                    idxB(ct2) = idx 
                    ct2 = ct2 + 1
                end if
            end do

            do i = 1,6
                idx = findloc(s3,s2(i),1)
                if (idx /= 0) then ! idx is an output index
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                end if
            end do

            temp1 = (/1,2,3,4/)
            ct3 = 1
            do i = 1,4
                if (any(temp1(i) == idxA(1:ct1-1))) then
                    cycle
                else 
                    idxA(ct1+ct3-1) = i 
                    ct3 = ct3 + 1
                    n2 = n2 * shapeA(i)
                end if
            end do

            temp2 = (/1,2,3,4,5,6/)
            ct3 = 1
            do i = 1,6
                if (any(temp2(i) == idxB(1:ct2-1))) then 
                    cycle 
                else 
                    idxB(ct2+ct3-1) = i 
                    ct3 = ct3 + 1 
                    n3 = n3 * shapeB(i)
                end if 
            end do

            shapeA = shapeA(idxA)
            shapeB = shapeB(idxB)
            shapeC = shapeC(idxC)

            allocate(Ap(shapeA(1),shapeA(2),shapeA(3),shapeA(4)))
            allocate(Bp(shapeB(1),shapeB(2),shapeB(3),shapeB(4),shapeB(5),shapeB(6)))
            allocate(Cp(shapeC(1),shapeC(2),shapeC(3),shapeC(4)))
            allocate(A2(n1,n2),B2(n2,n3),C2(n1,n3))

            ! find order of A
            do i = 1,4
                id = findloc(idxA,i,1)
                idxA2(i) = id
            end do
            Ap = reshape(A,shape=shapeA,order=idxA2)

            ! find order of B
            do i = 1,6
                id = findloc(idxB,i,1)
                idxB2(i) = id
            end do
            Bp = reshape(B,shape=shapeB,order=idxB2) 

            A2 = reshape(Ap,shape=(/n1,n2/))
            B2 = reshape(Bp,shape=(/n2,n3/))
            call gemm(A2,B2,C2)

            ! Cp = reshape(C2,shape=shapeC)
            ! idxC2 = argsort_int(idxC)
            ! C = reshape(Cp,shape=shapeC(idxC2),order=idxC2)
            
            ! find order of C
            idxC2 = argsort_int(idxC)
            do i = 1,4
                id = findloc(idxC2,i,1)
                idxC3(i) = id
            end do
            C = reshape(C2,shape=shapeC(idxC2),order=idxC3)
        
        deallocate(Ap,Bp,A2,B2,C2)

        end subroutine einsum464

        subroutine einsum466(str,A,szA,B,szB,C,szC)

            character, intent(in) :: str(1:19)
            integer, intent(in) :: szA(1:4), szB(1:6), szC(1:6)
            real, intent(in) :: A(szA(1),szA(2),szA(3),szA(4)), B(szB(1),szB(2),szB(3),szB(4),szB(5),szB(6))
            real, intent(out) :: C(szC(1),szC(2),szC(3),szC(4),szC(5),szC(6))
            real, allocatable :: A2(:,:), B2(:,:), C2(:,:),&
                                 Ap(:,:,:,:), Bp(:,:,:,:,:,:), Cp(:,:,:,:,:,:)
            character :: s1(1:4), s2(1:6), s3(1:6)
            integer :: i, idx, ct1, ct2, ct3, id, &
                       idxA(1:4), idxB(1:6), idxC(1:6), idxC2(1:6),&
                       idxA2(1:4), idxB2(1:6), idxC3(1:6)
            integer :: shapeA(1:4), shapeB(1:6), shapeC(1:6), n1, n2, n3
            integer :: temp1(1:4), temp2(1:6)

            s1 = str(1:4); s2 = str(6:11); s3 = str(14:);
            shapeA = shape(A); shapeB = shape(B); shapeC = shape(C);
            ct1 = 1; ct2 = 1; ct3 = 1;
            n1 = 1; n2 = 1; n3 = 1;

            do i = 1,4
                idx = findloc(s2,s1(i),1)
                if (idx == 0) then ! i is an output index
                    idx = findloc(s3,s1(i),1)
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                    idxA(ct1) = i 
                    n1 = n1 * shapeA(i)
                    ct1 = ct1 + 1 
                else ! i is contracted
                    idxB(ct2) = idx 
                    ct2 = ct2 + 1
                end if
            end do

            do i = 1,6
                idx = findloc(s3,s2(i),1)
                if (idx /= 0) then ! idx is an output index
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                end if
            end do

            temp1 = (/1,2,3,4/)
            ct3 = 1
            do i = 1,4
                if (any(temp1(i) == idxA(1:ct1-1))) then
                    cycle
                else 
                    idxA(ct1+ct3-1) = i 
                    ct3 = ct3 + 1
                    n2 = n2 * shapeA(i)
                end if
            end do

            temp2 = (/1,2,3,4,5,6/)
            ct3 = 1
            do i = 1,6
                if (any(temp2(i) == idxB(1:ct2-1))) then 
                    cycle 
                else 
                    idxB(ct2+ct3-1) = i 
                    ct3 = ct3 + 1 
                    n3 = n3 * shapeB(i)
                end if 
            end do

            shapeA = shapeA(idxA)
            shapeB = shapeB(idxB)
            shapeC = shapeC(idxC)

            allocate(Ap(shapeA(1),shapeA(2),shapeA(3),shapeA(4)))
            allocate(Bp(shapeB(1),shapeB(2),shapeB(3),shapeB(4),shapeB(5),shapeB(6)))
            allocate(Cp(shapeC(1),shapeC(2),shapeC(3),shapeC(4),shapeC(5),shapeC(6)))
            allocate(A2(n1,n2),B2(n2,n3),C2(n1,n3))

            ! find order of A
            do i = 1,4
                id = findloc(idxA,i,1)
                idxA2(i) = id
            end do
            Ap = reshape(A,shape=shapeA,order=idxA2)

            ! find order of B
            do i = 1,6
                id = findloc(idxB,i,1)
                idxB2(i) = id
            end do
            Bp = reshape(B,shape=shapeB,order=idxB2) 

            A2 = reshape(Ap,shape=(/n1,n2/))
            B2 = reshape(Bp,shape=(/n2,n3/))
            call gemm(A2,B2,C2)

            ! Cp = reshape(C2,shape=shapeC)
            ! idxC2 = argsort_int(idxC)
            ! C = reshape(Cp,shape=shapeC(idxC2),order=idxC2)
            
            ! find order of C
            idxC2 = argsort_int(idxC)
            do i = 1,6
                id = findloc(idxC2,i,1)
                idxC3(i) = id
            end do
            C = reshape(C2,shape=shapeC(idxC2),order=idxC3)
        
        deallocate(Ap,Bp,A2,B2,C2)

        end subroutine einsum466


        subroutine einsum233(str,A,szA,B,szB,C,szC)

            character, intent(in) :: str(1:11)
            integer, intent(in) :: szA(1:2), szB(1:3), szC(1:3)
            real, intent(in) :: A(szA(1),szA(2)), B(szB(1),szB(2),szB(3))
            real, intent(out) :: C(szC(1),szC(2),szC(3))
            real, allocatable :: A2(:,:), B2(:,:), C2(:,:),&
                                 Ap(:,:), Bp(:,:,:), Cp(:,:,:)
            character :: s1(1:2), s2(1:3), s3(1:3)
            integer :: i, idx, ct1, ct2, ct3, id, &
                       idxA(1:2), idxB(1:3), idxC(1:3), idxC2(1:3),&
                       idxA2(1:2), idxB2(1:3), idxC3(1:3)
            integer :: shapeA(1:2), shapeB(1:3), shapeC(1:3), n1, n2, n3
            integer :: temp1(1:2), temp2(1:3)

            s1 = str(1:2); s2 = str(4:6); s3 = str(9:);
            shapeA = shape(A); shapeB = shape(B); shapeC = shape(C);
            ct1 = 1; ct2 = 1; ct3 = 1;
            n1 = 1; n2 = 1; n3 = 1;

            do i = 1,2
                idx = findloc(s2,s1(i),1)
                if (idx == 0) then ! i is an output index
                    idx = findloc(s3,s1(i),1)
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                    idxA(ct1) = i 
                    n1 = n1 * shapeA(i)
                    ct1 = ct1 + 1 
                else ! i is contracted
                    idxB(ct2) = idx 
                    ct2 = ct2 + 1
                end if
            end do

            do i = 1,3
                idx = findloc(s3,s2(i),1)
                if (idx /= 0) then ! idx is an output index
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                end if
            end do

            temp1 = (/1,2/)
            ct3 = 1
            do i = 1,2
                if (any(temp1(i) == idxA(1:ct1-1))) then
                    cycle
                else 
                    idxA(ct1+ct3-1) = i 
                    ct3 = ct3 + 1
                    n2 = n2 * shapeA(i)
                end if
            end do

            temp2 = (/1,2,3/)
            ct3 = 1
            do i = 1,3
                if (any(temp2(i) == idxB(1:ct2-1))) then 
                    cycle 
                else 
                    idxB(ct2+ct3-1) = i 
                    ct3 = ct3 + 1 
                    n3 = n3 * shapeB(i)
                end if 
            end do

            shapeA = shapeA(idxA)
            shapeB = shapeB(idxB)
            shapeC = shapeC(idxC)

            allocate(Ap(shapeA(1),shapeA(2)))
            allocate(Bp(shapeB(1),shapeB(2),shapeB(3)))
            allocate(Cp(shapeC(1),shapeC(2),shapeC(3)))
            allocate(A2(n1,n2),B2(n2,n3),C2(n1,n3))

            ! find order of A
            do i = 1,2
                id = findloc(idxA,i,1)
                idxA2(i) = id
            end do
            Ap = reshape(A,shape=shapeA,order=idxA2)

            ! find order of B
            do i = 1,3
                id = findloc(idxB,i,1)
                idxB2(i) = id
            end do
            Bp = reshape(B,shape=shapeB,order=idxB2) 

            A2 = reshape(Ap,shape=(/n1,n2/))
            B2 = reshape(Bp,shape=(/n2,n3/))
            call gemm(A2,B2,C2)
            !C2 = gemm(A2,B2)

            ! find order of C
            idxC2 = argsort_int(idxC)
            do i = 1,3
                id = findloc(idxC2,i,1)
                idxC3(i) = id
            end do
            C = reshape(C2,shape=shapeC(idxC2),order=idxC3)
        
        deallocate(Ap,Bp,A2,B2,C2)

        end subroutine einsum233

        subroutine einsum323(str,A,szA,B,szB,C,szC)

            character, intent(in) :: str(1:11)
            integer, intent(in) :: szA(1:3), szB(1:2), szC(1:3)
            real, intent(in) :: A(szA(1),szA(2),szA(3)), B(szB(1),szB(2))
            real, intent(out) :: C(szC(1),szC(2),szC(3))
            real, allocatable :: A2(:,:), B2(:,:), C2(:,:),&
                                 Ap(:,:,:), Bp(:,:), Cp(:,:,:)
            character :: s1(1:3), s2(1:2), s3(1:3)
            integer :: i, idx, ct1, ct2, ct3, id, &
                       idxA(1:3), idxB(1:2), idxC(1:3), idxC2(1:3),&
                       idxA2(1:3), idxB2(1:2), idxC3(1:3)
            integer :: shapeA(1:3), shapeB(1:2), shapeC(1:3), n1, n2, n3
            integer :: temp1(1:3), temp2(1:2)

            s1 = str(1:3); s2 = str(5:6); s3 = str(9:);
            shapeA = shape(A); shapeB = shape(B); shapeC = shape(C);
            ct1 = 1; ct2 = 1; ct3 = 1;
            n1 = 1; n2 = 1; n3 = 1;

            do i = 1,3
                idx = findloc(s2,s1(i),1)
                if (idx == 0) then ! i is an output index
                    idx = findloc(s3,s1(i),1)
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                    idxA(ct1) = i 
                    n1 = n1 * shapeA(i)
                    ct1 = ct1 + 1 
                else ! i is contracted
                    idxB(ct2) = idx 
                    ct2 = ct2 + 1
                end if
            end do

            do i = 1,2
                idx = findloc(s3,s2(i),1)
                if (idx /= 0) then ! idx is an output index
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                end if
            end do

            temp1 = (/1,2,3/)
            ct3 = 1
            do i = 1,3
                if (any(temp1(i) == idxA(1:ct1-1))) then
                    cycle
                else 
                    idxA(ct1+ct3-1) = i 
                    ct3 = ct3 + 1
                    n2 = n2 * shapeA(i)
                end if
            end do

            temp2 = (/1,2/)
            ct3 = 1
            do i = 1,2
                if (any(temp2(i) == idxB(1:ct2-1))) then 
                    cycle 
                else 
                    idxB(ct2+ct3-1) = i 
                    ct3 = ct3 + 1 
                    n3 = n3 * shapeB(i)
                end if 
            end do

            shapeA = shapeA(idxA)
            shapeB = shapeB(idxB)
            shapeC = shapeC(idxC)

            allocate(Ap(shapeA(1),shapeA(2),shapeA(3)))
            allocate(Bp(shapeB(1),shapeB(2)))
            allocate(Cp(shapeC(1),shapeC(2),shapeC(3)))
            allocate(A2(n1,n2),B2(n2,n3),C2(n1,n3))

            ! find order of A
            do i = 1,3
                id = findloc(idxA,i,1)
                idxA2(i) = id
            end do
            Ap = reshape(A,shape=shapeA,order=idxA2)

            ! find order of B
            do i = 1,2
                id = findloc(idxB,i,1)
                idxB2(i) = id
            end do
            Bp = reshape(B,shape=shapeB,order=idxB2) 

            A2 = reshape(Ap,shape=(/n1,n2/))
            B2 = reshape(Bp,shape=(/n2,n3/))
            call gemm(A2,B2,C2)
            !C2 = gemm(A2,B2)

            ! find order of C
            idxC2 = argsort_int(idxC)
            do i = 1,3
                id = findloc(idxC2,i,1)
                idxC3(i) = id
            end do
            C = reshape(C2,shape=shapeC(idxC2),order=idxC3)
        
        deallocate(Ap,Bp,A2,B2,C2)

        end subroutine einsum323

        subroutine einsum446(str,A,szA,B,szB,C,szC)

            character, intent(in) :: str(1:17)
            integer, intent(in) :: szA(1:4), szB(1:4), szC(1:6)
            real, intent(in) :: A(szA(1),szA(2),szA(3),szA(4)), B(szB(1),szB(2),szB(3),szB(4))
            real, intent(out) :: C(szC(1),szC(2),szC(3),szC(4),szC(5),szC(6))
            real, allocatable :: A2(:,:), B2(:,:), C2(:,:),&
                                 Ap(:,:,:,:), Bp(:,:,:,:), Cp(:,:,:,:,:,:)
            character :: s1(1:4), s2(1:4), s3(1:6)
            integer :: i, idx, ct1, ct2, ct3, id, &
                       idxA(1:4), idxB(1:4), idxC(1:6), idxC2(1:6),&
                       idxA2(1:4), idxB2(1:4), idxC3(1:6)
            integer :: shapeA(1:4), shapeB(1:4), shapeC(1:6), n1, n2, n3
            integer :: temp1(1:4), temp2(1:4)

            s1 = str(1:4); s2 = str(6:9); s3 = str(12:);
            shapeA = shape(A); shapeB = shape(B); shapeC = shape(C);
            ct1 = 1; ct2 = 1; ct3 = 1;
            n1 = 1; n2 = 1; n3 = 1;

            do i = 1,4
                idx = findloc(s2,s1(i),1)
                if (idx == 0) then ! i is an output index
                    idx = findloc(s3,s1(i),1)
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                    idxA(ct1) = i 
                    n1 = n1 * shapeA(i)
                    ct1 = ct1 + 1 
                else ! i is contracted
                    idxB(ct2) = idx 
                    ct2 = ct2 + 1
                end if
            end do

            do i = 1,4
                idx = findloc(s3,s2(i),1)
                if (idx /= 0) then ! idx is an output index
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                end if
            end do

            temp1 = (/1,2,3,4/)
            ct3 = 1
            do i = 1,4
                if (any(temp1(i) == idxA(1:ct1-1))) then
                    cycle
                else 
                    idxA(ct1+ct3-1) = i 
                    ct3 = ct3 + 1
                    n2 = n2 * shapeA(i)
                end if
            end do

            temp2 = (/1,2,3,4/)
            ct3 = 1
            do i = 1,4
                if (any(temp2(i) == idxB(1:ct2-1))) then 
                    cycle 
                else 
                    idxB(ct2+ct3-1) = i 
                    ct3 = ct3 + 1 
                    n3 = n3 * shapeB(i)
                end if 
            end do

            shapeA = shapeA(idxA)
            shapeB = shapeB(idxB)
            shapeC = shapeC(idxC)

            allocate(Ap(shapeA(1),shapeA(2),shapeA(3),shapeA(4)))
            allocate(Bp(shapeB(1),shapeB(2),shapeB(3),shapeB(4)))
            allocate(Cp(shapeC(1),shapeC(2),shapeC(3),shapeC(4),shapeC(5),shapeC(6)))
            allocate(A2(n1,n2),B2(n2,n3),C2(n1,n3))

            ! find order of A
            do i = 1,4
                id = findloc(idxA,i,1)
                idxA2(i) = id
            end do
            Ap = reshape(A,shape=shapeA,order=idxA2)

            ! find order of B
            do i = 1,4
                id = findloc(idxB,i,1)
                idxB2(i) = id
            end do
            Bp = reshape(B,shape=shapeB,order=idxB2) 

            A2 = reshape(Ap,shape=(/n1,n2/))
            B2 = reshape(Bp,shape=(/n2,n3/))
            call gemm(A2,B2,C2)
            !C2 = gemm(A2,B2)

            ! find order of C
            idxC2 = argsort_int(idxC)
            do i = 1,6
                id = findloc(idxC2,i,1)
                idxC3(i) = id
            end do
            C = reshape(C2,shape=shapeC(idxC2),order=idxC3)
        
        deallocate(Ap,Bp,A2,B2,C2)

        end subroutine einsum446

        ! UTILITIES
        subroutine gemm(A,B,C)

            real(kind=8), intent(in) :: A(:,:), B(:,:)
            real(kind=8), intent(out) :: C(:,:)
            integer :: m, n, k
            real :: alpha = 1.0d0, beta = 0.0d0
            m = ubound(A,1)
            n = ubound(B,2)
            k = ubound(B,1)
            call dgemm('n','n',m,n,k,alpha,A,m,B,k,beta,C,m)

        end subroutine gemm

        function argsort_int(a) result(idx)

            integer, intent(in) :: a(:)
            integer :: i, j, idx(size(a)), temp

            do i = 1,size(a)
                idx(i) = i
            end do

            do j = 0, size(a)-1
               do i = 1, size(a)-j-1
                  if ( a(idx(i)) > a(idx(i+1)) ) then
                          temp = idx(i)
                          idx(i) = idx(i+1)
                          idx(i+1) = temp
                  end if
               end do
            end do

        end function argsort_int(a)




end module einsum_module
