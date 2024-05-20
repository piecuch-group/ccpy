module slater

      implicit none

      integer, parameter :: BIT_KIND_SIZE= 32 
      integer, parameter :: N_INT_MAX = 10 ! bit_kind_size * N_int_max = max. number of MO's

      integer(kind=4), parameter :: ZERO_BIT_KIND = 0
      integer(kind=4), parameter :: ONE_BIT_KIND = 1

      integer, parameter :: MAX_EXCITATION_DIFFERENCE = 3

      real(kind=8), parameter :: PHASE_DBLE(0:1) = (/ 1.0d0, -1.0d0 /)

      contains

              subroutine decode_excitation(io, exc, phase, N_int, i, det1, j, det2)

                      integer, intent(in) :: io
                      integer, intent(in) :: N_int, i, j
                      integer, intent(in) :: exc(1:2, 1:2, 1:2, 1:3), det1(4), det2(4)
                      real(kind=8), intent(in) :: phase

                      print*, 'Decoding <',i,'|H|',j,'>'
                      print*, '---------------------------'
                      print*, '   Determinant 1 = ', det1(1), det1(2), det1(3), det1(4)
                      print*, '      # different hole alpha = ', exc(1,1,1,1),' -> ', exc(1,1,1,2), exc(1,1,1,3)
                      print*, '      # different particle alpha = ', exc(1,2,1,1),' -> ', exc(1,2,1,2), exc(1,2,1,3)
                      print*, '      # different hole beta = ', exc(1,1,2,1),' -> ', exc(1,1,2,2), exc(1,1,2,3)
                      print*, '      # different particle beta = ', exc(1,2,2,1), ' -> ', exc(1,2,2,2), exc(1,2,2,3)
                      print*, '   Determinant 2 = ', det2(1), det2(2), det2(3), det2(4)
                      print*, '      # different hole alpha = ', exc(2,1,1,1),' -> ', exc(2,1,1,2), exc(2,1,1,3)
                      print*, '      # different particle alpha = ', exc(2,2,1,1),' -> ', exc(2,2,1,2), exc(2,2,1,3)
                      print*, '      # different hole beta = ', exc(2,1,2,1),' -> ', exc(2,1,2,2), exc(2,1,2,3)
                      print*, '      # different particle beta = ', exc(2,2,2,1), ' -> ', exc(2,2,2,2), exc(2,2,2,3)
                      print*, '   Phase = ', phase

              end subroutine decode_excitation
                      

              ! not compatible with N_int greater than 1. Need to define
              ! dets1(n1, 4, N_int) and call dets1(idet, k, :) to get the 
              ! set of integers for hole/particle alpha/beta
              subroutine determinant_loop(n1, n2, N_int, dets1, dets2)

                      integer, intent(in) :: n1, n2, N_int
                      integer, intent(in) :: dets1(n1, 4), dets2(n2, 4)

                      integer :: exc(1:2, 1:2, 1:2, 1:3), comms(1:2, 1:2, 0:3)
                      real(kind=8) :: phase
                      logical :: iflag
                      integer :: idet, jdet

                      real(kind=8) :: tic, toc

                      integer, parameter :: io = 22

                      logical :: verbose

                      verbose = .false.

                      !open(unit=io, file='dets.log', status='replace')

                      do idet = 1 , n1
                         call cpu_time(tic)
                         do jdet = 1 , n2

                            call get_difference_slater(exc, comms, phase, iflag,&
                                               N_int,&
                                               dets1(idet,1), dets1(idet,2), dets1(idet,3), dets1(idet,4),&
                                               dets2(jdet,1), dets2(jdet,2), dets2(jdet,3), dets2(jdet,4))

                            if (iflag .and. verbose) then
                                !call decode_excitation(io, exc, phase, N_int, idet, dets1(idet,:), jdet, dets2(jdet,:))
                                      print*, 'Decoding <',idet,'|H|',jdet,'>'
                                      print*, '---------------------------'
                                      print*, '   Determinant 1 = ', dets1(idet,1), dets1(idet,2), dets1(idet,3), dets1(idet,4)
                                      print*, '      # different hole alpha = ', exc(1,1,1,1),' -> ', exc(1,1,1,2), exc(1,1,1,3)
                                      print*, '      # different particle alpha = ', exc(1,2,1,1),' -> ', exc(1,2,1,2), exc(1,2,1,3)
                                      print*, '      # different hole beta = ', exc(1,1,2,1),' -> ', exc(1,1,2,2), exc(1,1,2,3)
                                      print*, '      # different particle beta = ', exc(1,2,2,1), ' -> ', exc(1,2,2,2), exc(1,2,2,3)
                                      print*, '   Determinant 2 = ', dets2(jdet,1), dets2(jdet,2), dets2(jdet,3), dets2(jdet,4)
                                      print*, '      # different hole alpha = ', exc(2,1,1,1),' -> ', exc(2,1,1,2), exc(2,1,1,3)
                                      print*, '      # different particle alpha = ', exc(2,2,1,1),' -> ', exc(2,2,1,2), exc(2,2,1,3)
                                      print*, '      # different hole beta = ', exc(2,1,2,1),' -> ', exc(2,1,2,2), exc(2,1,2,3)
                                      print*, '      # different particle beta = ', exc(2,2,2,1), ' -> ', exc(2,2,2,2), exc(2,2,2,3)
                                      print*, '   Phase = ', phase
                            end if

                         end do
                         call cpu_time(toc)
                         print*, 'Time = ', toc - tic
                      end do

                      !close(unit=io)

              end subroutine determinant_loop
              
              ! use exc array much like Quantum Package
              ! exc(I2/I2, hole/particle, alpha/beta, # and MO index)
              !
              ! exc(1 : 2, 1 : 2, 1 : 2, 0 : 2)
              !
              ! This means:
              !
              ! exc(1 or 2, 1, 1, 0) = # of hole alpha indices for I1 or I2
              ! exc(1 or 2, 1, 2, 0) = # of hole beta indices for I1 or I2
              ! exc(1 or 2, 2, 1, 0) = # of particle alpha indices for I1 or I2
              ! exc(1 or 2, 2, 2, 0) = # of particle beta indices for I1 or I2
              !  
              ! exc(1 or 2, 1, 1, 1) = MO index of 1st different hole alpha index on I1 or I2
              ! exc(1 or 2, 1, 2, 1) = MO index of 1st different hole beta index on I1 or I2
              ! exc(1 or 2, 2, 1, 1) = MO index of 1st different particle alpha index on I1 or I2
              ! exc(1, 2, 2, 1) = MO index of 1st different particle beta index on I1 or I2
              !
              ! exc(1 or 2, 1, 1, 2) = MO index of 2nd different hole alpha index on I1 or I2
              ! exc(1 or 2, 1, 2, 2) = MO index of 2nd different hole beta index on I1 or I2
              ! exc(1 or 2, 2, 1, 2) = MO index of 2nd different particle alpha index on I1 or I2
              ! exc(1 or 2, 2, 2, 2) = MO index of 2nd different particle beta index on I1 or I2
              !
              ! Note:
              ! You can't have a different of 3 hole or particle MO indices of
              ! the same spin on a single determinant. This would require 3
              ! lines that are all incoming or outgoing on the right of HBar,
              ! which is impossible for a two-body Hamiltonian. The total degree
              ! of difference, however, which is given by sum(exc(0, :, :, :))/2,
              ! can easily exceed 2 and can be at most 3.

              subroutine compare_determinants(exc, comms, phase, iflag,&
                                               N_int,&
                                               Ha1, Hb1, Pa1, Pb1,&
                                               Ha2, Hb2, Pa2, Pb2)

                      integer, intent(out) :: exc(1:2, 1:2, 1:2, 1:3), comms(1:2, 1:2, 1:4)
                      real(kind=8), intent(out) :: phase
                      logical, intent(out) :: iflag

                      integer, intent(in) :: N_int
                      integer(kind=4), dimension(N_int), intent(in) :: Ha1, Hb1, Pa1, Pb1,&
                                                                       Ha2, Hb2, Pa2, Pb2

                      integer :: degree, nperm, pos1(2), pos2(2), npos1, npos2, perm,&
                                 commpos(3), ncommpos

                      exc = 0
                      nperm = 0
                      iflag = .false.

                      call get_excitation_degree(degree,&
                                                 N_int,&
                                                 Ha1, Hb1, Pa1, Pb1,&
                                                 Ha2, Hb2, Pa2, Pb2)
                      if (degree .gt. 3) then
                            return
                      end if

                      ! Get difference for hole alpha integers
                      call get_difference(pos1, pos2, npos1, npos2, perm, N_int, Ha1, Ha2)
                      exc(1,1,1,2:3) = pos1
                      exc(2,1,1,2:3) = pos2
                      exc(1,1,1,1) = npos1
                      exc(2,1,1,1) = npos2
                      nperm = nperm + perm
                      ! Get common overlap for hole alpha integers
                      call get_common(commpos, ncommpos, N_int, Ha1, Ha2)
                      comms(1,1,1) = ncommpos
                      comms(1,1,2:4) = commpos

                      ! Get difference for hole beta integers
                      call get_difference(pos1, pos2, npos1, npos2, perm, N_int, Hb1, Hb2)
                      exc(1,1,2,2:3) = pos1
                      exc(2,1,2,2:3) = pos2
                      exc(1,1,2,1) = npos1
                      exc(2,1,2,1) = npos2
                      nperm = nperm + perm
                      ! Get common overlap for hole beta integers
                      call get_common(commpos, ncommpos, N_int, Hb1, Hb2)
                      comms(1,2,1) = ncommpos
                      comms(1,2,2:4) = commpos
            
                      ! Special case for similarity-transformed 2-body
                      ! Hamiltonian; no hhh ladder diagram
                      if (exc(2,1,1,1) + exc(2,1,2,1) .ge. 3) then
                            return
                      end if

                      ! Get difference for particle alpha integers
                      call get_difference(pos1, pos2, npos1, npos2, perm, N_int, Pa1, Pa2)
                      exc(1,2,1,2:3) = pos1
                      exc(2,2,1,2:3) = pos2
                      exc(1,2,1,1) = npos1
                      exc(2,2,1,1) = npos2
                      nperm = nperm + perm
                      ! Get common overlap for particle alpha integers
                      call get_common(commpos, ncommpos, N_int, Pa1, Pa2)
                      comms(2,1,1) = ncommpos
                      comms(2,1,2:4) = commpos

                      ! Get difference for particle beta integers
                      call get_difference(pos1, pos2, npos1, npos2, perm, N_int, Pb1, Pb2)
                      exc(1,2,2,2:3) = pos1
                      exc(2,2,2,2:3) = pos2
                      exc(1,2,2,1) = npos1
                      exc(2,2,2,1) = npos2
                      nperm = nperm + perm
                      ! Get common overlap for particle beta integers
                      call get_common(commpos, ncommpos, N_int, Pb1, Pb2)
                      comms(2,2,1) = ncommpos
                      comms(2,2,2:4) = commpos

                      ! Special case for similarity-transformed 2-body
                      ! Hamiltonian; no ppp ladder diagram
                      if (exc(2,2,1,1) + exc(2,2,2,1) .ge. 3) then
                            return
                      end if

                      ! Special case for similarity-transformed 2-body
                      ! Hamiltonian; max. number of lines extending to
                      ! right is 4
                      if (exc(2,1,1,1)+exc(2,1,2,1)+exc(2,2,1,1)+exc(2,2,2,1) .ge. 4) then
                            return
                      end if

                      iflag = .true.

                      ! Return the phase
                      phase = PHASE_DBLE( iand(nperm, 1) )

              end subroutine compare_determinants



              subroutine get_excitation_degree(degree,&
                                               N_int,&
                                               Ha1, Hb1, Pa1, Pb1,&
                                               Ha2, Hb2, Pa2, Pb2)

                      integer, intent(out) :: degree

                      integer, intent(in) :: N_int
                      integer(kind=4), dimension(N_int), intent(in) :: Ha1, Hb1, Pa1, Pb1,&
                                                                       Ha2, Hb2, Pa2, Pb2
                      integer :: l

                      degree = 0
                      do l = 1 , N_int
                         degree = degree + popcnt( ieor(Ha1(l), Ha2(l)) )&
                                         + popcnt( ieor(Hb1(l), Hb2(l)) )&
                                         + popcnt( ieor(Pa1(l), Pa2(l)) )&
                                         + popcnt( ieor(Pb1(l), Pb2(l)) )
                      end do

                      ! Divide by 2 (in a faster way)
                      degree = shiftr(degree, 1)

              end subroutine get_excitation_degree

              subroutine get_difference(holes, particles, idx_hole, idx_particle, nperm, N_int, I1, I2)

                      integer, intent(out) :: holes(2),&
                                              particles(2),&
                                              idx_hole, idx_particle,&
                                              nperm

                      integer, intent(in) :: N_int
                      integer(kind=4), dimension(N_int), intent(in) :: I1, I2

                      integer :: l, ishift, tz, ishift2
                      integer(kind=4) :: H, P, tmp, tmp2, mask

                      ! MO index positions of different indices
                      holes = 0         ! Describes I1
                      particles = 0     ! Describes I2 
                      ! Total number of different indices
                      idx_hole = 0
                      idx_particle = 0
                      ! Number of transposition counter
                      nperm = 0

                      ! Shift to add 32 (or 64) orbitals to hole/particle index
                      ! whenever you move from one of the N_int integers to
                      ! another (also compensates for bit indexing starting from 0)
                      ishift = 1 - BIT_KIND_SIZE
                      do l = 1 , N_int

                         ! Increment shift to next integer
                         ishift = ishift + BIT_KIND_SIZE

                         ! If the two integers are the same, no difference;
                         ! early exit
                         if ( I1(l) == I2(l) ) then
                                 cycle
                         end if

                         ! Get the xor difference between bitstrings
                         tmp = ieor( I1(l), I2(l) )

                         ! Find the holes (position of differences in 1)
                         H = iand( tmp, I1(l) )
                         ! Re-zero the transposition counter helper
                         ishift2 = 0
                         do while ( H /= ZERO_BIT_KIND ) 
                            ! There exists an occupied bit; increment counter
                            idx_hole = idx_hole + 1
                            ! Track the number of differing orbitals on the same integer
                            ishift2 = ishift2 + 1
                            ! Locate position of occupied bit using trailing zeros
                            tz = trailz(H)
                            ! Store the position
                            holes(idx_hole) = tz + ishift
                            ! Clear the rightmost occupied bit
                            H = iand( H, H - ONE_BIT_KIND )

                            ! Clear all bits to the right of index tz in I1(l)
                            mask = not( shiftl(ONE_BIT_KIND, tz) - ONE_BIT_KIND)
                            ! Count number of occupied bits to the left of position tz
                            ! The additional (ishift2 - 1) accounts for moving past
                            ! those bits that were previously moved to the left
                            ! on the same integer.
                            nperm = nperm + popcnt( iand(I1(l), mask) ) - 1 + (ishift2 - 1)
                         end do

                         ! Find the particles (position of differences in 2)
                         P = iand( tmp, I2(l) )
                         ! Re-zero the transposition counter helper
                         ishift2 = 0
                         do while ( P /= ZERO_BIT_KIND )

                            ! There exists an occupied bit; increment counter
                            idx_particle = idx_particle + 1
                            ! Track the number of differing orbitals on the same integer
                            ishift2 = ishift2 + 1
                            ! Locate position of occupied bit using trailing zeros
                            tz = trailz(P)
                            ! Store the position
                            particles(idx_particle) = tz + ishift
                            ! Clear the rightmost occupied bit
                            P = iand( P, P - ONE_BIT_KIND )

                            ! Clear all bits to the right of index tz in I2(l)
                            mask = not( shiftl(ONE_BIT_KIND, tz) - ONE_BIT_KIND)
                            ! Count number of occupied bits to the left of position tz
                            ! The additional (ishift2 - 1) accounts for moving past
                            ! those bits that were previously moved to the left
                            ! on the same integer.
                            nperm = nperm + popcnt( iand(I2(l), mask) ) - 1 + (ishift2 - 1)
                         end do

                      end do 
                      
              end subroutine get_difference


              subroutine get_common(comms, num_common, N_int, I1, I2)

                      ! Maximum number of common indices is set by highest
                      ! excitation level. E.g., for triples, max # common for
                      ! any category (hole alpha, particle alpha, etc.,) is 3.
                      integer, intent(out) :: comms(3), num_common

                      integer, intent(in) :: N_int
                      integer(kind=4), dimension(N_int), intent(in) :: I1, I2

                      integer :: l, ishift, tz
                      integer(kind=4) :: tmp

                      ! MO index positions of common indices
                      comms = 0
                      ! Total number of common indices
                      num_common = 0

                      ! Shift to add 32 (or 64) orbitals to hole/particle index
                      ! whenever you move from one of the N_int integers to
                      ! another (also compensates for bit indexing starting from 0)
                      ishift = 1 - BIT_KIND_SIZE
                      do l = 1 , N_int

                         ! Increment shift to next integer
                         ishift = ishift + BIT_KIND_SIZE

                         ! Get the and between bitstrings
                         tmp = iand( I1(l), I2(l) )

                         ! Find the positions of ones in the overlap of bitstrings
                         do while ( tmp /= ZERO_BIT_KIND ) 
                            ! There exists an occupied bit; increment counter
                            num_common = num_common + 1
                            ! Locate position of occupied bit using trailing zeros
                            tz = trailz(tmp)
                            ! Store the position
                            comms(num_common) = tz + ishift
                            ! Clear the rightmost occupied bit
                            tmp = iand( tmp, tmp - ONE_BIT_KIND )
                         end do

                      end do 
                      
              end subroutine get_common


end module slater

                        


