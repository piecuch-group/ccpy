module excitations

    implicit none

    contains


        subroutine determine_excitation(iflag, det1, det2, N_int)

            integer, intent(in) :: N_int
            integer(kind=8), intent(in) :: det1(N_int, 2), det2(N_int, 2)

            logical, intent(out) :: iflag

            integer :: n_excit

            call get_excitation(n_excit, det1, det2, N_int)

            iflag = .false.

            if (n_excit <= 3) then
                iflag = .true.
            end if

        end subroutine determine_excitation

        subroutine get_excitation(n_excit, det1, det2, N_int)

            integer, intent(in) :: N_int
            integer(kind=8), intent(in) :: det1(N_int, 2), det2(N_int, 2)

            integer, intent(out) :: n_excit

            integer:: l

            n_excit = popcnt(ieor(det1(1, 1), det2(1, 1))) + popcnt(ieor(det1(1, 2), det2(1, 2)))

            do l = 2, N_int
                n_excit = n_excit + popcnt(ieor(det1(l, 1), det2(l, 1))) + popcnt(ieor(det1(l, 2), det2(l, 2)))
            end do

        end subroutine get_excitation

end module excitations
