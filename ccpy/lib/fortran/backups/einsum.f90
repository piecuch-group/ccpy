module einsum

    ! Written by Karthik Gururangan 2/11/21

    implicit none

    contains

        subroutine tensordot(str,A,B,C)

            character(len=*), intent(in) :: str 
            real(kind=8), intent(in) :: A(..), B(..)
            real(kind=8), intent(out) :: C(..)
            integer :: rankA, rankB, rankC, rank_vec(1:3)

            rankA = size(shape(A))
            rankB = size(shape(B))
            rankC = size(shape(C))

            rank_vec = (/rankA, rankB, rankC/)

            if (all(rank_vec == (/2,2,2/))) then
               call einsum222(str,A,shape(A),B,shape(B),C,shape(C))
            end if 

            
        end subroutine tensordot

        subroutine einsum222(str,A,szA,B,szB,C,szC)

            character, intent(in) :: str(1:9)
            integer, intent(in) :: szA(1:2), szB(1:2), szC(1:2)
            real(kind=8), intent(in) :: A(szA(1),szA(2)), B(szB(1),szB(2))
            real(kind=8), intent(out) :: C(szC(1),szC(2))
            real(kind=8), allocatable :: A2(:,:), B2(:,:), C2(:,:),&
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


        function argsort_int(a) result(idx)

            integer, intent(in) :: a(:)
            integer :: i, j, idx(size(a)), temp

            do i = 1, size(a)
                idx(i) = i
            end do

            do j = 0, size(a)-1
                do i = 1, size(a)-j-1
                    if (a(idx(i)) > a(idx(i+1))) then
                        temp = idx(i)
                        idx(i) = idx(i+1)
                        idx(i+1) = temp
                    end if
                end do
            end do

        end function argsort_int

        subroutine gemm(A, B, C)

            real(kind=8), intent(in) :: A(:,:), B(:,:)
            real, intent(out) :: C(:,:)
            integer :: m, n, k

            m = ubound(A, 1)
            n = ubound(B, 2)
            k = ubound(B, 1)
            call dgemm('n', 'n', m, n, k, 1.0d0, A, m, B, k, 0.0d0, C, m)

        end subroutine gemm

end module einsum
