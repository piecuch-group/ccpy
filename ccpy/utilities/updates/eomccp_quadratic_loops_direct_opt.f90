module eomccp_quadratic_loops_direct_opt

      use omp_lib

      implicit none

      contains

               subroutine build_hr_1a(x1a,&
                                      r3a_excits, r3b_excits, r3c_excits,&
                                      r3a_amps, r3b_amps, r3c_amps,&
                                      h2a_oovv, h2b_oovv, h2c_oovv,&
                                      n3aaa, n3aab, n3abb,&
                                      noa, nua, nob, nub)

                      integer, intent(in) :: noa, nua, nob, nub, n3aaa, n3aab, n3abb
                      integer, intent(in) :: r3a_excits(6,n3aaa), r3b_excits(6,n3aab), r3c_excits(6,n3abb)
                      real(kind=8), intent(in) :: r3a_amps(n3aaa), r3b_amps(n3aab), r3c_amps(n3abb)
                      real(kind=8), intent(in) :: h2a_oovv(1:noa,1:noa,1:nua,1:nua),&
                                                  h2b_oovv(1:noa,1:nob,1:nua,1:nub),&
                                                  h2c_oovv(1:nob,1:nob,1:nub,1:nub)
                      
                      real(kind=8), intent(inout) :: x1a(1:nua,1:noa)
                      !f2py intent(in,out) :: x1a(0:nua-1,0:noa-1)
                      
                      integer :: i, a, m, n, e, f, idet
                      real(kind=8) :: denom, val, r_amp
                      
                      ! compute < ia | (H(2) * R3)_C | 0 >
                      do idet = 1, n3aaa
                          r_amp = r3a_amps(idet)
                          ! A(a/ef)A(i/mn) h2a(mnef) * r3a(aefimn)
                          a = r3a_excits(1,idet); e = r3a_excits(2,idet); f = r3a_excits(3,idet);
                          i = r3a_excits(4,idet); m = r3a_excits(5,idet); n = r3a_excits(6,idet);
                          x1a(a,i) = x1a(a,i) + h2a_oovv(m,n,e,f) * r_amp ! (1)
                          x1a(e,i) = x1a(e,i) - h2a_oovv(m,n,a,f) * r_amp ! (ae)
                          x1a(f,i) = x1a(f,i) - h2a_oovv(m,n,e,a) * r_amp ! (af)
                          x1a(a,m) = x1a(a,m) - h2a_oovv(i,n,e,f) * r_amp ! (im)
                          x1a(e,m) = x1a(e,m) + h2a_oovv(i,n,a,f) * r_amp ! (ae)(im)
                          x1a(f,m) = x1a(f,m) + h2a_oovv(i,n,e,a) * r_amp ! (af)(im)
                          x1a(a,n) = x1a(a,n) - h2a_oovv(m,i,e,f) * r_amp ! (in)
                          x1a(e,n) = x1a(e,n) + h2a_oovv(m,i,a,f) * r_amp ! (ae)(in)
                          x1a(f,n) = x1a(f,n) + h2a_oovv(m,i,e,a) * r_amp ! (af)(in)
                      end do
                      do idet = 1, n3aab
                          r_amp = r3b_amps(idet)
                          ! A(ae)A(im) h2b(mnef) * r3b(aefimn)
                          a = r3b_excits(1,idet); e = r3b_excits(2,idet); f = r3b_excits(3,idet);
                          i = r3b_excits(4,idet); m = r3b_excits(5,idet); n = r3b_excits(6,idet);
                          x1a(a,i) = x1a(a,i) + h2b_oovv(m,n,e,f) * r_amp ! (1)
                          x1a(e,i) = x1a(e,i) - h2b_oovv(m,n,a,f) * r_amp ! (ae)
                          x1a(a,m) = x1a(a,m) - h2b_oovv(i,n,e,f) * r_amp ! (im)
                          x1a(e,m) = x1a(e,m) + h2b_oovv(i,n,a,f) * r_amp ! (ae)(im)
                      end do
                      do idet = 1, n3abb
                          r_amp = r3c_amps(idet)
                          ! h2c(mnef) * r3c(aefimn)
                          a = r3c_excits(1,idet); e = r3c_excits(2,idet); f = r3c_excits(3,idet);
                          i = r3c_excits(4,idet); m = r3c_excits(5,idet); n = r3c_excits(6,idet);
                          x1a(a,i) = x1a(a,i) + h2c_oovv(m,n,e,f) * r_amp ! (1)
                      end do
              end subroutine build_hr_1a
         
              subroutine build_hr_1b(x1b,&
                                     r3b_excits, r3c_excits, r3d_excits,&
                                     r3b_amps, r3c_amps, r3d_amps,&
                                     h2a_oovv, h2b_oovv, h2c_oovv,&
                                     n3aab, n3abb, n3bbb,&
                                     noa, nua, nob, nub)

                      integer, intent(in) :: noa, nua, nob, nub, n3aab, n3abb, n3bbb
                      integer, intent(in) :: r3b_excits(6,n3aab), r3c_excits(6,n3abb), r3d_excits(6,n3bbb)
                      real(kind=8), intent(in) :: r3b_amps(n3aab), r3c_amps(n3abb), r3d_amps(n3bbb)
                      real(kind=8), intent(in) :: h2a_oovv(1:noa,1:noa,1:nua,1:nua),&
                                                  h2b_oovv(1:noa,1:nob,1:nua,1:nub),&
                                                  h2c_oovv(1:nob,1:nob,1:nub,1:nub)

                      real(kind=8), intent(inout) :: x1b(1:nub,1:nob)
                      !f2py intent(in,out) :: x1b(0:nub-1,0:nob-1)
                      
                      integer :: i, a, m, n, e, f, idet
                      real(kind=8) :: denom, val, r_amp
                      
                      ! compute < i~a~ | (H(2) * R3)_C | 0 >
                      do idet = 1, n3aab
                          r_amp = r3b_amps(idet)
                          ! h2a(mnef) * r3b(efamni)
                          e = r3b_excits(1,idet); f = r3b_excits(2,idet); a = r3b_excits(3,idet);
                          m = r3b_excits(4,idet); n = r3b_excits(5,idet); i = r3b_excits(6,idet);
                          x1b(a,i) = x1b(a,i) + h2a_oovv(m,n,e,f) * r_amp ! (1)
                      end do
                      do idet = 1, n3abb
                          r_amp = r3c_amps(idet)
                          ! A(af)A(in) h2b(mnef) * r3c(efamni)
                          e = r3c_excits(1,idet); f = r3c_excits(2,idet); a = r3c_excits(3,idet);
                          m = r3c_excits(4,idet); n = r3c_excits(5,idet); i = r3c_excits(6,idet);
                          x1b(a,i) = x1b(a,i) + h2b_oovv(m,n,e,f) * r_amp ! (1)
                          x1b(f,i) = x1b(f,i) - h2b_oovv(m,n,e,a) * r_amp ! (af)
                          x1b(a,n) = x1b(a,n) - h2b_oovv(m,i,e,f) * r_amp ! (in)
                          x1b(f,n) = x1b(f,n) + h2b_oovv(m,i,e,a) * r_amp ! (af)(in)
                      end do
                      do idet = 1, n3bbb
                          r_amp = r3d_amps(idet)
                          ! A(a/ef)A(i/mn) h2c(mnef) * r3d(aefimn)
                          a = r3d_excits(1,idet); e = r3d_excits(2,idet); f = r3d_excits(3,idet);
                          i = r3d_excits(4,idet); m = r3d_excits(5,idet); n = r3d_excits(6,idet);
                          x1b(a,i) = x1b(a,i) + h2c_oovv(m,n,e,f) * r_amp ! (1)
                          x1b(e,i) = x1b(e,i) - h2c_oovv(m,n,a,f) * r_amp ! (ae)
                          x1b(f,i) = x1b(f,i) - h2c_oovv(m,n,e,a) * r_amp ! (af)
                          x1b(a,m) = x1b(a,m) - h2c_oovv(i,n,e,f) * r_amp ! (im)
                          x1b(e,m) = x1b(e,m) + h2c_oovv(i,n,a,f) * r_amp ! (ae)(im)
                          x1b(f,m) = x1b(f,m) + h2c_oovv(i,n,e,a) * r_amp ! (af)(im)
                          x1b(a,n) = x1b(a,n) - h2c_oovv(m,i,e,f) * r_amp ! (in)
                          x1b(e,n) = x1b(e,n) + h2c_oovv(m,i,a,f) * r_amp ! (ae)(in)
                          x1b(f,n) = x1b(f,n) + h2c_oovv(m,i,e,a) * r_amp ! (af)(in)
                      end do
              end subroutine build_hr_1b
	      
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!! SORTING FUNCTIONS !!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

              subroutine get_index_table(idx_table, rng1, rng2, rng3, rng4, n1, n2, n3, n4)

                    integer, intent(in) :: n1, n2, n3, n4
                    integer, intent(in) :: rng1(2), rng2(2), rng3(2), rng4(2)
      
                    integer, intent(inout) :: idx_table(n1,n2,n3,n4)
      
                    integer :: kout
                    integer :: p, q, r, s
      
                    idx_table = 0
                    ! 5 possible cases. Always organize so that ordered indices appear first.
                    if (rng1(1) < 0 .and. rng2(1) < 0 .and. rng3(1) < 0 .and. rng4(1) < 0) then ! p < q < r < s
                       kout = 1 
                       do p = rng1(1), rng1(2)
                          do q = p-rng2(1), rng2(2)
                             do r = q-rng3(1), rng3(2)
                                do s = r-rng4(1), rng4(2)
                                   idx_table(p,q,r,s) = kout
                                   kout = kout + 1
                                end do
                             end do
                          end do
                       end do
                    elseif (rng1(1) > 0 .and. rng2(1) < 0 .and. rng3(1) < 0 .and. rng4(1) > 0) then ! p < q < r, s
                       kout = 1 
                       do p = rng1(1), rng1(2)
                          do q = p-rng2(1), rng2(2)
                             do r = q-rng3(1), rng3(2)
                                do s = rng4(1), rng4(2)
                                   idx_table(p,q,r,s) = kout
                                   kout = kout + 1
                                end do
                             end do
                          end do
                       end do
                    elseif (rng1(1) > 0 .and. rng2(1) < 0 .and. rng3(1) > 0 .and. rng4(1) < 0) then ! p < q, r < s
                       kout = 1 
                       do p = rng1(1), rng1(2)
                          do q = p-rng2(1), rng2(2)
                             do r = rng3(1), rng3(2)
                                do s = r-rng4(1), rng4(2)
                                   idx_table(p,q,r,s) = kout
                                   kout = kout + 1
                                end do
                             end do
                          end do
                       end do
                    elseif (rng1(1) > 0 .and. rng2(1) < 0 .and. rng3(1) > 0 .and. rng4(1) > 0) then ! p < q, r, s
                       kout = 1 
                       do p = rng1(1), rng1(2)
                          do q = p-rng2(1), rng2(2)
                             do r = rng3(1), rng3(2)
                                do s = rng4(1), rng4(2)
                                   idx_table(p,q,r,s) = kout
                                   kout = kout + 1
                                end do
                             end do
                          end do
                       end do
                    else ! p, q, r, s
                       kout = 1 
                       do p = rng1(1), rng1(2)
                          do q = rng2(1), rng2(2)
                             do r = rng3(1), rng3(2)
                                do s = rng4(1), rng4(2)
                                   idx_table(p,q,r,s) = kout
                                   kout = kout + 1
                                end do
                             end do
                          end do
                       end do
                    end if

              end subroutine get_index_table

              subroutine sort4(excits, amps, loc_arr, idx_table, idims, n1, n2, n3, n4, nloc, n3p, x1a)

                    integer, intent(in) :: n1, n2, n3, n4, nloc, n3p
                    integer, intent(in) :: idims(4)
                    integer, intent(in) :: idx_table(n1,n2,n3,n4)
      
                    integer, intent(inout) :: loc_arr(nloc,2)
                    integer, intent(inout) :: excits(6,n3p)
                    real(kind=8), intent(inout) :: amps(n3p)
                    real(kind=8), intent(inout), optional :: x1a(n3p)
      
                    integer :: idet
                    integer :: p, q, r, s
                    integer :: p1, q1, r1, s1, p2, q2, r2, s2
                    integer :: pqrs1, pqrs2
                    integer, allocatable :: temp(:), idx(:)
      
                    allocate(temp(n3p),idx(n3p))
                    do idet = 1, n3p
                       p = excits(idims(1),idet); q = excits(idims(2),idet); r = excits(idims(3),idet); s = excits(idims(4),idet)
                       temp(idet) = idx_table(p,q,r,s)
                    end do
                    call argsort(temp, idx)
                    excits = excits(:,idx)
                    amps = amps(idx)
                    if (present(x1a)) x1a = x1a(idx)
                    deallocate(temp,idx)
      
                    loc_arr(:,1) = 1; loc_arr(:,2) = 0;
                    do idet = 1, n3p-1
                       p1 = excits(idims(1),idet);   q1 = excits(idims(2),idet);   r1 = excits(idims(3),idet);   s1 = excits(idims(4),idet)
                       p2 = excits(idims(1),idet+1); q2 = excits(idims(2),idet+1); r2 = excits(idims(3),idet+1); s2 = excits(idims(4),idet+1)
                       pqrs1 = idx_table(p1,q1,r1,s1)
                       pqrs2 = idx_table(p2,q2,r2,s2)
                       if (pqrs1 /= pqrs2) then
                          loc_arr(pqrs1,2) = idet
                          loc_arr(pqrs2,1) = idet+1
                       end if
                    end do
                    loc_arr(pqrs2,2) = n3p

              end subroutine sort4

              subroutine argsort(r,d)

                    integer, intent(in), dimension(:) :: r
                    integer, intent(out), dimension(size(r)) :: d
      
                    integer, dimension(size(r)) :: il
      
                    integer :: stepsize
                    integer :: i, j, n, left, k, ksize
      
                    n = size(r)
      
                    do i=1,n
                       d(i)=i
                    end do
      
                    if (n==1) return
      
                    stepsize = 1
                    do while (stepsize < n)
                       do left = 1, n-stepsize,stepsize*2
                          i = left
                          j = left+stepsize
                          ksize = min(stepsize*2,n-left+1)
                          k=1
      
                          do while (i < left+stepsize .and. j < left+ksize)
                             if (r(d(i)) < r(d(j))) then
                                il(k) = d(i)
                                i = i+1
                                k = k+1
                             else
                                il(k) = d(j)
                                j = j+1
                                k = k+1
                             endif
                          enddo
      
                          if (i < left+stepsize) then
                             ! fill up remaining from left
                             il(k:ksize) = d(i:left+stepsize-1)
                          else
                             ! fill up remaining from right
                             il(k:ksize) = d(j:left+ksize-1)
                          endif
                          d(left:left+ksize-1) = il(1:ksize)
                       end do
                       stepsize = stepsize*2
                    end do

              end subroutine argsort

end module eomccp_quadratic_loops_direct_opt
 