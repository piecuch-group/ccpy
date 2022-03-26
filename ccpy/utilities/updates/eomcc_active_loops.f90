module eomcc_active_loops

  implicit none

  contains

      subroutine update_r1a(r1a,omega,H1A_oo,H1A_vv,H1B_oo,H1B_vv,shift,noa,nua,nob,nub)

              integer, intent(in) :: noa, nua, nob, nub
              real(8), intent(in) :: H1A_oo(1:noa,1:noa), H1A_vv(1:nua,1:nua), &
                                  H1B_oo(1:nob,1:nob), H1B_vv(1:nub,1:nub), shift, &
                                  omega

              real(8), intent(inout) :: r1a(1:nua,1:noa)
              !f2py intent(in,out) :: r1a(0:nua-1,0:noa-1)

              integer :: i, j, a, b
              real(8) :: denom

              do i = 1,noa
                do a = 1,nua
                  denom = H1A_vv(a,a) - H1A_oo(i,i)
                  r1a(a,i) = r1a(a,i)/(omega-denom+shift)
                end do
              end do

      end subroutine update_r1a

      subroutine update_r1b(r1b,omega,H1A_oo,H1A_vv,H1B_oo,H1B_vv,shift,noa,nua,nob,nub)

              integer, intent(in) :: noa, nua, nob, nub
              real(8), intent(in) :: H1A_oo(1:noa,1:noa), H1A_vv(1:nua,1:nua), &
                                  H1B_oo(1:nob,1:nob), H1B_vv(1:nub,1:nub), shift, &
                                  omega

              real(8), intent(inout) :: r1b(1:nub,1:nob)
              !f2py intent(in,out) :: r1b(0:nub-1,0:nob-1)

              integer :: i, j, a, b
              real(8) :: denom


              do i = 1,nob
                do a = 1,nub
                  denom = H1B_vv(a,a) - H1B_oo(i,i)
                  r1b(a,i) = r1b(a,i)/(omega-denom+shift)
                end do
              end do

      end subroutine update_r1b

      subroutine update_r2a(r2a,omega,H1A_oo,H1A_vv,H1B_oo,H1B_vv,shift,noa,nua,nob,nub)

              integer, intent(in) :: noa, nua, nob, nub
              real(8), intent(in) :: H1A_oo(1:noa,1:noa), H1A_vv(1:nua,1:nua), &
                                  H1B_oo(1:nob,1:nob), H1B_vv(1:nub,1:nub), shift, &
                                  omega

              real(8), intent(inout) :: r2a(1:nua,1:nua,1:noa,1:noa)
              !f2py intent(in,out) :: r2a(0:nua-1,0:nua-1,0:noa-1,0:noa-1)

              integer :: i, j, a, b
              real(8) :: denom

              do i = 1,noa
                do j = 1,noa
                  do a = 1,nua
                    do b = 1,nua
                      denom = H1A_vv(a,a) + H1A_vv(b,b) - H1A_oo(i,i) - H1A_oo(j,j)
                      r2a(b,a,j,i) = r2a(b,a,j,i)/(omega-denom+shift)
                    end do
                  end do
                end do
              end do

      end subroutine update_r2a

      subroutine update_r2b(r2b,omega,H1A_oo,H1A_vv,H1B_oo,H1B_vv,shift,noa,nua,nob,nub)

              integer, intent(in) :: noa, nua, nob, nub
              real(8), intent(in) :: H1A_oo(1:noa,1:noa), H1A_vv(1:nua,1:nua), &
                                  H1B_oo(1:nob,1:nob), H1B_vv(1:nub,1:nub), shift, &
                                  omega

              real(8), intent(inout) :: r2b(1:nua,1:nub,1:noa,1:nob)
              !f2py intent(in,out) :: r2b(0:nua-1,0:nub-1,0:noa-1,0:nob-1)

              integer :: i, j, a, b
              real(8) :: denom

              do j = 1,nob
                do i = 1,noa
                  do b = 1,nub
                    do a = 1,nua
                      denom = H1A_vv(a,a) + H1B_vv(b,b) - H1A_oo(i,i) - H1B_oo(j,j)
                      r2b(a,b,i,j) = r2b(a,b,i,j)/(omega-denom+shift)
                    end do
                  end do
                end do
              end do

      end subroutine update_r2b

      subroutine update_r2c(r2c,omega,H1A_oo,H1A_vv,H1B_oo,H1B_vv,shift,noa,nua,nob,nub)

              integer, intent(in) :: noa, nua, nob, nub
              real(8), intent(in) :: H1A_oo(1:noa,1:noa), H1A_vv(1:nua,1:nua), &
                                  H1B_oo(1:nob,1:nob), H1B_vv(1:nub,1:nub), shift, &
                                  omega

              real(8), intent(inout) :: r2c(1:nub,1:nub,1:nob,1:nob)
              !f2py intent(in,out) :: r2c(0:nub-1,0:nub-1,0:nob-1,0:nob-1)

              integer :: i, j, a, b
              real(8) :: denom

              do i = 1,nob
                do j = 1,nob
                  do a = 1,nub
                    do b = 1,nub
                      denom = H1B_vv(a,a) + H1B_vv(b,b) - H1B_oo(i,i) - H1B_oo(j,j)
                      r2c(b,a,j,i) = r2c(b,a,j,i)/(omega-denom+shift)
                    end do
                  end do
                end do
              end do

      end subroutine update_r2c

subroutine update_r3a_100001(r3a, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3a(1:nua_act, 1:nua_inact, 1:nua_inact, 1:noa_inact, 1:noa_inact, 1:noa_act)
      !f2py intent(in, out)  :: r3a(0:nua_act-1, 0:nua_inact-1, 0:nua_inact-1, 0:noa_inact-1, 0:noa_inact-1, 0:noa_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , noa_inact
            do k = 1 , noa_act
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nua_inact

                        denom = H1A_oo_inact(i,i) + H1A_oo_inact(j,j) + H1A_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_inact(b,b) - H1A_vv_inact(c,c)


                        r3a(a, b, c, i, j, k) = r3a(a, b, c, i, j, k) /(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3a_100001

subroutine update_r3a_100011(r3a, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3a(1:nua_act, 1:nua_inact, 1:nua_inact, 1:noa_inact, 1:noa_act, 1:noa_act)
      !f2py intent(in, out)  :: r3a(0:nua_act-1, 0:nua_inact-1, 0:nua_inact-1, 0:noa_inact-1, 0:noa_act-1, 0:noa_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = 1 , noa_act
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nua_inact

                        denom = H1A_oo_inact(i,i) + H1A_oo_act(j,j) + H1A_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_inact(b,b) - H1A_vv_inact(c,c)


                        r3a(a, b, c, i, j, k) = r3a(a, b, c, i, j, k) /(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3a_100011

    subroutine update_r3a_100111(r3a, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3a(1:nua_act, 1:nua_inact, 1:nua_inact, 1:noa_act, 1:noa_act, 1:noa_act)
      !f2py intent(in, out)  :: r3a(0:nua_act-1, 0:nua_inact-1, 0:nua_inact-1, 0:noa_act-1, 0:noa_act-1, 0:noa_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_act
         do j = 1 , noa_act
            do k = 1 , noa_act
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nua_inact

                        denom = H1A_oo_act(i,i) + H1A_oo_act(j,j) + H1A_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_inact(b,b) - H1A_vv_inact(c,c)


                        r3a(a, b, c, i, j, k) = r3a(a, b, c, i, j, k) /(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3a_100111

    subroutine update_r3a_110001(r3a, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3a(1:nua_act, 1:nua_act, 1:nua_inact, 1:noa_inact, 1:noa_inact, 1:noa_act)
      !f2py intent(in, out)  :: r3a(0:nua_act-1, 0:nua_act-1, 0:nua_inact-1, 0:noa_inact-1, 0:noa_inact-1, 0:noa_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , noa_inact
            do k = 1 , noa_act
               do a = 1 , nua_act
                  do b = 1 , nua_act
                     do c = 1 , nua_inact

                        denom = H1A_oo_inact(i,i) + H1A_oo_inact(j,j) + H1A_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_act(b,b) - H1A_vv_inact(c,c)


                        r3a(a, b, c, i, j, k) = r3a(a, b, c, i, j, k) /(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3a_110001

    subroutine update_r3a_110011(r3a, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3a(1:nua_act, 1:nua_act, 1:nua_inact, 1:noa_inact, 1:noa_act, 1:noa_act)
      !f2py intent(in, out)  :: r3a(0:nua_act-1, 0:nua_act-1, 0:nua_inact-1, 0:noa_inact-1, 0:noa_act-1, 0:noa_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = 1 , noa_act
               do a = 1 , nua_act
                  do b = 1 , nua_act
                     do c = 1 , nua_inact

                        denom = H1A_oo_inact(i,i) + H1A_oo_act(j,j) + H1A_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_act(b,b) - H1A_vv_inact(c,c)


                        r3a(a, b, c, i, j, k) = r3a(a, b, c, i, j, k) /(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3a_110011

    subroutine update_r3a_110111(r3a, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3a(1:nua_act, 1:nua_act, 1:nua_inact, 1:noa_act, 1:noa_act, 1:noa_act)
      !f2py intent(in, out)  :: r3a(0:nua_act-1, 0:nua_act-1, 0:nua_inact-1, 0:noa_act-1, 0:noa_act-1, 0:noa_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_act
         do j = 1 , noa_act
            do k = 1 , noa_act
               do a = 1 , nua_act
                  do b = 1 , nua_act
                     do c = 1 , nua_inact

                        denom = H1A_oo_act(i,i) + H1A_oo_act(j,j) + H1A_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_act(b,b) - H1A_vv_inact(c,c)


                        r3a(a, b, c, i, j, k) = r3a(a, b, c, i, j, k) /(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3a_110111

    subroutine update_r3a_111001(r3a, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3a(1:nua_act, 1:nua_act, 1:nua_act, 1:noa_inact, 1:noa_inact, 1:noa_act)
      !f2py intent(in, out)  :: r3a(0:nua_act-1, 0:nua_act-1, 0:nua_act-1, 0:noa_inact-1, 0:noa_inact-1, 0:noa_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , noa_inact
            do k = 1 , noa_act
               do a = 1 , nua_act
                  do b = 1 , nua_act
                     do c = 1 , nua_act

                        denom = H1A_oo_inact(i,i) + H1A_oo_inact(j,j) + H1A_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_act(b,b) - H1A_vv_act(c,c)


                        r3a(a, b, c, i, j, k) = r3a(a, b, c, i, j, k) /(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3a_111001

subroutine update_r3a_111011(r3a, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3a(1:nua_act, 1:nua_act, 1:nua_act, 1:noa_inact, 1:noa_act, 1:noa_act)
      !f2py intent(in, out)  :: r3a(0:nua_act-1, 0:nua_act-1, 0:nua_act-1, 0:noa_inact-1, 0:noa_act-1, 0:noa_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = 1 , noa_act
               do a = 1 , nua_act
                  do b = 1 , nua_act
                     do c = 1 , nua_act

                        denom = H1A_oo_inact(i,i) + H1A_oo_act(j,j) + H1A_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_act(b,b) - H1A_vv_act(c,c)


                        r3a(a, b, c, i, j, k) = r3a(a, b, c, i, j, k) /(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3a_111011

subroutine update_r3a_111111(r3a, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3a(1:nua_act, 1:nua_act, 1:nua_act, 1:noa_act, 1:noa_act, 1:noa_act)
      !f2py intent(in, out)  :: r3a(0:nua_act-1, 0:nua_act-1, 0:nua_act-1, 0:noa_act-1, 0:noa_act-1, 0:noa_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_act
         do j = 1 , noa_act
            do k = 1 , noa_act
               do a = 1 , nua_act
                  do b = 1 , nua_act
                     do c = 1 , nua_act

                        denom = H1A_oo_act(i,i) + H1A_oo_act(j,j) + H1A_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_act(b,b) - H1A_vv_act(c,c)


                        r3a(a, b, c, i, j, k) = r3a(a, b, c, i, j, k) /(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3a_111111


subroutine update_r3b_001001(r3b, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3b(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_inact, 1:nob_act)
      !f2py intent(in, out)  :: r3b(0:nua_inact-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_inact-1, 0:noa_inact-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , noa_inact
            do k = 1 , nob_act
               do a = 1 , nua_inact
                  do b = 1 , nua_inact
                     do c = 1 , nub_act

                        denom = H1A_oo_inact(i,i) + H1A_oo_inact(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_inact(a,a) - H1A_vv_inact(b,b) - H1B_vv_act(c,c)


                        r3b(a, b, c, i, j, k) = r3b(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3b_001001

    subroutine update_r3b_001010(r3b, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3b(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_inact)
      !f2py intent(in, out)  :: r3b(0:nua_inact-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_inact-1, 0:noa_act-1, 0:nob_inact-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = 1 , nob_inact
               do a = 1 , nua_inact
                  do b = 1 , nua_inact
                     do c = 1 , nub_act

                        denom = H1A_oo_inact(i,i) + H1A_oo_act(j,j) + H1B_oo_inact(k,k)&
                               -H1A_vv_inact(a,a) - H1A_vv_inact(b,b) - H1B_vv_act(c,c)


                        r3b(a, b, c, i, j, k) = r3b(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3b_001010

subroutine update_r3b_001011(r3b, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3b(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: r3b(0:nua_inact-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_inact-1, 0:noa_act-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_inact
                  do b = 1 , nua_inact
                     do c = 1 , nub_act

                        denom = H1A_oo_inact(i,i) + H1A_oo_act(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_inact(a,a) - H1A_vv_inact(b,b) - H1B_vv_act(c,c)


                        r3b(a, b, c, i, j, k) = r3b(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3b_001011


    subroutine update_r3b_001110(r3b, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3b(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_inact)
      !f2py intent(in, out)  :: r3b(0:nua_inact-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_act-1, 0:noa_act-1, 0:nob_inact-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_act
         do j = 1 , noa_act
            do k = 1 , nob_inact
               do a = 1 , nua_inact
                  do b = 1 , nua_inact
                     do c = 1 , nub_act

                        denom = H1A_oo_act(i,i) + H1A_oo_act(j,j) + H1B_oo_inact(k,k)&
                               -H1A_vv_inact(a,a) - H1A_vv_inact(b,b) - H1B_vv_act(c,c)


                        r3b(a, b, c, i, j, k) = r3b(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3b_001110


    subroutine update_r3b_001111(r3b, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3b(1:nua_inact, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: r3b(0:nua_inact-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_act-1, 0:noa_act-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_act
         do j = 1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_inact
                  do b = 1 , nua_inact
                     do c = 1 , nub_act

                        denom = H1A_oo_act(i,i) + H1A_oo_act(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_inact(a,a) - H1A_vv_inact(b,b) - H1B_vv_act(c,c)


                        r3b(a, b, c, i, j, k) = r3b(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3b_001111


    subroutine update_r3b_100001(r3b, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3b(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_inact, 1:noa_inact, 1:nob_act)
      !f2py intent(in, out)  :: r3b(0:nua_act-1, 0:nua_inact-1, 0:nub_inact-1, 0:noa_inact-1, 0:noa_inact-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , noa_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_inact

                        denom = H1A_oo_inact(i,i) + H1A_oo_inact(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_inact(b,b) - H1B_vv_inact(c,c)


                        r3b(a, b, c, i, j, k) = r3b(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3b_100001


subroutine update_r3b_100010(r3b, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3b(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_inact, 1:noa_act, 1:nob_inact)
      !f2py intent(in, out)  :: r3b(0:nua_act-1, 0:nua_inact-1, 0:nub_inact-1, 0:noa_inact-1, 0:noa_act-1, 0:nob_inact-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = 1 , nob_inact
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_inact

                        denom = H1A_oo_inact(i,i) + H1A_oo_act(j,j) + H1B_oo_inact(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_inact(b,b) - H1B_vv_inact(c,c)


                        r3b(a, b, c, i, j, k) = r3b(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3b_100010


    subroutine update_r3b_100011(r3b, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3b(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_inact, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: r3b(0:nua_act-1, 0:nua_inact-1, 0:nub_inact-1, 0:noa_inact-1, 0:noa_act-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_inact

                        denom = H1A_oo_inact(i,i) + H1A_oo_act(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_inact(b,b) - H1B_vv_inact(c,c)


                        r3b(a, b, c, i, j, k) = r3b(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3b_100011


    subroutine update_r3b_100110(r3b, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3b(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_act, 1:noa_act, 1:nob_inact)
      !f2py intent(in, out)  :: r3b(0:nua_act-1, 0:nua_inact-1, 0:nub_inact-1, 0:noa_act-1, 0:noa_act-1, 0:nob_inact-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_act
         do j = 1 , noa_act
            do k = 1 , nob_inact
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_inact

                        denom = H1A_oo_act(i,i) + H1A_oo_act(j,j) + H1B_oo_inact(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_inact(b,b) - H1B_vv_inact(c,c)


                        r3b(a, b, c, i, j, k) = r3b(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3b_100110


    subroutine update_r3b_100111(r3b, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3b(1:nua_act, 1:nua_inact, 1:nub_inact, 1:noa_act, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: r3b(0:nua_act-1, 0:nua_inact-1, 0:nub_inact-1, 0:noa_act-1, 0:noa_act-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_act
         do j = 1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_inact

                        denom = H1A_oo_act(i,i) + H1A_oo_act(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_inact(b,b) - H1B_vv_inact(c,c)


                        r3b(a, b, c, i, j, k) = r3b(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3b_100111

    subroutine update_r3b_101001(r3b, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3b(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_inact, 1:nob_act)
      !f2py intent(in, out)  :: r3b(0:nua_act-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_inact-1, 0:noa_inact-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , noa_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_act

                        denom = H1A_oo_inact(i,i) + H1A_oo_inact(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_inact(b,b) - H1B_vv_act(c,c)


                        r3b(a, b, c, i, j, k) = r3b(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3b_101001


    subroutine update_r3b_101010(r3b, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3b(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_inact)
      !f2py intent(in, out)  :: r3b(0:nua_act-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_inact-1, 0:noa_act-1, 0:nob_inact-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = 1 , nob_inact
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_act

                        denom = H1A_oo_inact(i,i) + H1A_oo_act(j,j) + H1B_oo_inact(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_inact(b,b) - H1B_vv_act(c,c)


                        r3b(a, b, c, i, j, k) = r3b(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3b_101010


    subroutine update_r3b_101011(r3b, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3b(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: r3b(0:nua_act-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_inact-1, 0:noa_act-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_act

                        denom = H1A_oo_inact(i,i) + H1A_oo_act(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_inact(b,b) - H1B_vv_act(c,c)


                        r3b(a, b, c, i, j, k) = r3b(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3b_101011


    subroutine update_r3b_101110(r3b, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3b(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_inact)
      !f2py intent(in, out)  :: r3b(0:nua_act-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_act-1, 0:noa_act-1, 0:nob_inact-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_act
         do j = 1 , noa_act
            do k = 1 , nob_inact
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_act

                        denom = H1A_oo_act(i,i) + H1A_oo_act(j,j) + H1B_oo_inact(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_inact(b,b) - H1B_vv_act(c,c)


                        r3b(a, b, c, i, j, k) = r3b(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3b_101110


    subroutine update_r3b_101111(r3b, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3b(1:nua_act, 1:nua_inact, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: r3b(0:nua_act-1, 0:nua_inact-1, 0:nub_act-1, 0:noa_act-1, 0:noa_act-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_act
         do j = 1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nua_inact
                     do c = 1 , nub_act

                        denom = H1A_oo_act(i,i) + H1A_oo_act(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_inact(b,b) - H1B_vv_act(c,c)


                        r3b(a, b, c, i, j, k) = r3b(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3b_101111


    subroutine update_r3b_110001(r3b, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3b(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_inact, 1:noa_inact, 1:nob_act)
      !f2py intent(in, out)  :: r3b(0:nua_act-1, 0:nua_act-1, 0:nub_inact-1, 0:noa_inact-1, 0:noa_inact-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , noa_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nua_act
                     do c = 1 , nub_inact

                        denom = H1A_oo_inact(i,i) + H1A_oo_inact(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_act(b,b) - H1B_vv_inact(c,c)


                        r3b(a, b, c, i, j, k) = r3b(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3b_110001


    subroutine update_r3b_110010(r3b, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3b(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_inact, 1:noa_act, 1:nob_inact)
      !f2py intent(in, out)  :: r3b(0:nua_act-1, 0:nua_act-1, 0:nub_inact-1, 0:noa_inact-1, 0:noa_act-1, 0:nob_inact-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = 1 , nob_inact
               do a = 1 , nua_act
                  do b = 1 , nua_act
                     do c = 1 , nub_inact

                        denom = H1A_oo_inact(i,i) + H1A_oo_act(j,j) + H1B_oo_inact(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_act(b,b) - H1B_vv_inact(c,c)


                        r3b(a, b, c, i, j, k) = r3b(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3b_110010


    subroutine update_r3b_110011(r3b, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3b(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_inact, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: r3b(0:nua_act-1, 0:nua_act-1, 0:nub_inact-1, 0:noa_inact-1, 0:noa_act-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nua_act
                     do c = 1 , nub_inact

                        denom = H1A_oo_inact(i,i) + H1A_oo_act(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_act(b,b) - H1B_vv_inact(c,c)


                        r3b(a, b, c, i, j, k) = r3b(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3b_110011


    subroutine update_r3b_110110(r3b, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3b(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_act, 1:noa_act, 1:nob_inact)
      !f2py intent(in, out)  :: r3b(0:nua_act-1, 0:nua_act-1, 0:nub_inact-1, 0:noa_act-1, 0:noa_act-1, 0:nob_inact-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_act
         do j = 1 , noa_act
            do k = 1 , nob_inact
               do a = 1 , nua_act
                  do b = 1 , nua_act
                     do c = 1 , nub_inact

                        denom = H1A_oo_act(i,i) + H1A_oo_act(j,j) + H1B_oo_inact(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_act(b,b) - H1B_vv_inact(c,c)


                        r3b(a, b, c, i, j, k) = r3b(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3b_110110


    subroutine update_r3b_110111(r3b, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3b(1:nua_act, 1:nua_act, 1:nub_inact, 1:noa_act, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: r3b(0:nua_act-1, 0:nua_act-1, 0:nub_inact-1, 0:noa_act-1, 0:noa_act-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_act
         do j = 1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nua_act
                     do c = 1 , nub_inact

                        denom = H1A_oo_act(i,i) + H1A_oo_act(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_act(b,b) - H1B_vv_inact(c,c)


                        r3b(a, b, c, i, j, k) = r3b(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3b_110111


    subroutine update_r3b_111001(r3b, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3b(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_inact, 1:noa_inact, 1:nob_act)
      !f2py intent(in, out)  :: r3b(0:nua_act-1, 0:nua_act-1, 0:nub_act-1, 0:noa_inact-1, 0:noa_inact-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , noa_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nua_act
                     do c = 1 , nub_act

                        denom = H1A_oo_inact(i,i) + H1A_oo_inact(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_act(b,b) - H1B_vv_act(c,c)


                        r3b(a, b, c, i, j, k) = r3b(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3b_111001


    subroutine update_r3b_111010(r3b, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3b(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_inact)
      !f2py intent(in, out)  :: r3b(0:nua_act-1, 0:nua_act-1, 0:nub_act-1, 0:noa_inact-1, 0:noa_act-1, 0:nob_inact-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = 1 , nob_inact
               do a = 1 , nua_act
                  do b = 1 , nua_act
                     do c = 1 , nub_act

                        denom = H1A_oo_inact(i,i) + H1A_oo_act(j,j) + H1B_oo_inact(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_act(b,b) - H1B_vv_act(c,c)


                        r3b(a, b, c, i, j, k) = r3b(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3b_111010


    subroutine update_r3b_111011(r3b, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3b(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_inact, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: r3b(0:nua_act-1, 0:nua_act-1, 0:nub_act-1, 0:noa_inact-1, 0:noa_act-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nua_act
                     do c = 1 , nub_act

                        denom = H1A_oo_inact(i,i) + H1A_oo_act(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_act(b,b) - H1B_vv_act(c,c)


                        r3b(a, b, c, i, j, k) = r3b(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3b_111011


    subroutine update_r3b_111110(r3b, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3b(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_inact)
      !f2py intent(in, out)  :: r3b(0:nua_act-1, 0:nua_act-1, 0:nub_act-1, 0:noa_act-1, 0:noa_act-1, 0:nob_inact-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_act
         do j = 1 , noa_act
            do k = 1 , nob_inact
               do a = 1 , nua_act
                  do b = 1 , nua_act
                     do c = 1 , nub_act

                        denom = H1A_oo_act(i,i) + H1A_oo_act(j,j) + H1B_oo_inact(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_act(b,b) - H1B_vv_act(c,c)


                        r3b(a, b, c, i, j, k) = r3b(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3b_111110


    subroutine update_r3b_111111(r3b, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3b(1:nua_act, 1:nua_act, 1:nub_act, 1:noa_act, 1:noa_act, 1:nob_act)
      !f2py intent(in, out)  :: r3b(0:nua_act-1, 0:nua_act-1, 0:nub_act-1, 0:noa_act-1, 0:noa_act-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_act
         do j = 1 , noa_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nua_act
                     do c = 1 , nub_act

                        denom = H1A_oo_act(i,i) + H1A_oo_act(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1A_vv_act(b,b) - H1B_vv_act(c,c)


                        r3b(a, b, c, i, j, k) = r3b(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3b_111111


    subroutine update_r3c_010001(r3c, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3c(1:nua_inact, 1:nub_act, 1:nub_inact, 1:noa_inact, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: r3c(0:nua_inact-1, 0:nub_act-1, 0:nub_inact-1, 0:noa_inact-1, 0:nob_inact-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_inact
                  do b = 1 , nub_act
                     do c = 1 , nub_inact

                        denom = H1A_oo_inact(i,i) + H1B_oo_inact(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_inact(a,a) - H1B_vv_act(b,b) - H1B_vv_inact(c,c)


                        r3c(a, b, c, i, j, k) = r3c(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3c_010001


    subroutine update_r3c_010011(r3c, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3c(1:nua_inact, 1:nub_act, 1:nub_inact, 1:noa_inact, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: r3c(0:nua_inact-1, 0:nub_act-1, 0:nub_inact-1, 0:noa_inact-1, 0:nob_act-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , nob_act
            do k = 1 , nob_act
               do a = 1 , nua_inact
                  do b = 1 , nub_act
                     do c = 1 , nub_inact

                        denom = H1A_oo_inact(i,i) + H1B_oo_act(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_inact(a,a) - H1B_vv_act(b,b) - H1B_vv_inact(c,c)


                        r3c(a, b, c, i, j, k) = r3c(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3c_010011


    subroutine update_r3c_010100(r3c, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3c(1:nua_inact, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_inact)
      !f2py intent(in, out)  :: r3c(0:nua_inact-1, 0:nub_act-1, 0:nub_inact-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_inact-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = 1 , nob_inact
               do a = 1 , nua_inact
                  do b = 1 , nub_act
                     do c = 1 , nub_inact

                        denom = H1A_oo_act(i,i) + H1B_oo_inact(j,j) + H1B_oo_inact(k,k)&
                               -H1A_vv_inact(a,a) - H1B_vv_act(b,b) - H1B_vv_inact(c,c)


                        r3c(a, b, c, i, j, k) = r3c(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3c_010100


    subroutine update_r3c_010101(r3c, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3c(1:nua_inact, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: r3c(0:nua_inact-1, 0:nub_act-1, 0:nub_inact-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_inact
                  do b = 1 , nub_act
                     do c = 1 , nub_inact

                        denom = H1A_oo_act(i,i) + H1B_oo_inact(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_inact(a,a) - H1B_vv_act(b,b) - H1B_vv_inact(c,c)


                        r3c(a, b, c, i, j, k) = r3c(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3c_010101


    subroutine update_r3c_010111(r3c, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3c(1:nua_inact, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: r3c(0:nua_inact-1, 0:nub_act-1, 0:nub_inact-1, 0:noa_act-1, 0:nob_act-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_act
         do j = 1 , nob_act
            do k = 1 , nob_act
               do a = 1 , nua_inact
                  do b = 1 , nub_act
                     do c = 1 , nub_inact

                        denom = H1A_oo_act(i,i) + H1B_oo_act(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_inact(a,a) - H1B_vv_act(b,b) - H1B_vv_inact(c,c)


                        r3c(a, b, c, i, j, k) = r3c(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3c_010111


    subroutine update_r3c_011001(r3c, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3c(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_inact, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: r3c(0:nua_inact-1, 0:nub_act-1, 0:nub_act-1, 0:noa_inact-1, 0:nob_inact-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_inact
                  do b = 1 , nub_act
                     do c = 1 , nub_act

                        denom = H1A_oo_inact(i,i) + H1B_oo_inact(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_inact(a,a) - H1B_vv_act(b,b) - H1B_vv_act(c,c)


                        r3c(a, b, c, i, j, k) = r3c(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3c_011001

    subroutine update_r3c_011011(r3c, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3c(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_inact, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: r3c(0:nua_inact-1, 0:nub_act-1, 0:nub_act-1, 0:noa_inact-1, 0:nob_act-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , nob_act
            do k = 1 , nob_act
               do a = 1 , nua_inact
                  do b = 1 , nub_act
                     do c = 1 , nub_act

                        denom = H1A_oo_inact(i,i) + H1B_oo_act(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_inact(a,a) - H1B_vv_act(b,b) - H1B_vv_act(c,c)


                        r3c(a, b, c, i, j, k) = r3c(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3c_011011

subroutine update_r3c_011100(r3c, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3c(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_inact)
      !f2py intent(in, out)  :: r3c(0:nua_inact-1, 0:nub_act-1, 0:nub_act-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_inact-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = 1 , nob_inact
               do a = 1 , nua_inact
                  do b = 1 , nub_act
                     do c = 1 , nub_act

                        denom = H1A_oo_act(i,i) + H1B_oo_inact(j,j) + H1B_oo_inact(k,k)&
                               -H1A_vv_inact(a,a) - H1B_vv_act(b,b) - H1B_vv_act(c,c)


                        r3c(a, b, c, i, j, k) = r3c(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3c_011100

    subroutine update_r3c_011101(r3c, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3c(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: r3c(0:nua_inact-1, 0:nub_act-1, 0:nub_act-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_inact
                  do b = 1 , nub_act
                     do c = 1 , nub_act

                        denom = H1A_oo_act(i,i) + H1B_oo_inact(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_inact(a,a) - H1B_vv_act(b,b) - H1B_vv_act(c,c)


                        r3c(a, b, c, i, j, k) = r3c(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3c_011101

subroutine update_r3c_011111(r3c, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3c(1:nua_inact, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: r3c(0:nua_inact-1, 0:nub_act-1, 0:nub_act-1, 0:noa_act-1, 0:nob_act-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_act
         do j = 1 , nob_act
            do k = 1 , nob_act
               do a = 1 , nua_inact
                  do b = 1 , nub_act
                     do c = 1 , nub_act

                        denom = H1A_oo_act(i,i) + H1B_oo_act(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_inact(a,a) - H1B_vv_act(b,b) - H1B_vv_act(c,c)


                        r3c(a, b, c, i, j, k) = r3c(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3c_011111


    subroutine update_r3c_100001(r3c, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3c(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_inact, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: r3c(0:nua_act-1, 0:nub_inact-1, 0:nub_inact-1, 0:noa_inact-1, 0:nob_inact-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_inact
                     do c = 1 , nub_inact

                        denom = H1A_oo_inact(i,i) + H1B_oo_inact(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1B_vv_inact(b,b) - H1B_vv_inact(c,c)


                        r3c(a, b, c, i, j, k) = r3c(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3c_100001


    subroutine update_r3c_100011(r3c, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3c(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_inact, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: r3c(0:nua_act-1, 0:nub_inact-1, 0:nub_inact-1, 0:noa_inact-1, 0:nob_act-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , nob_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_inact
                     do c = 1 , nub_inact

                        denom = H1A_oo_inact(i,i) + H1B_oo_act(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1B_vv_inact(b,b) - H1B_vv_inact(c,c)


                        r3c(a, b, c, i, j, k) = r3c(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3c_100011


    subroutine update_r3c_100100(r3c, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3c(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_inact)
      !f2py intent(in, out)  :: r3c(0:nua_act-1, 0:nub_inact-1, 0:nub_inact-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_inact-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = 1 , nob_inact
               do a = 1 , nua_act
                  do b = 1 , nub_inact
                     do c = 1 , nub_inact

                        denom = H1A_oo_act(i,i) + H1B_oo_inact(j,j) + H1B_oo_inact(k,k)&
                               -H1A_vv_act(a,a) - H1B_vv_inact(b,b) - H1B_vv_inact(c,c)


                        r3c(a, b, c, i, j, k) = r3c(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3c_100100


    subroutine update_r3c_100101(r3c, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3c(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: r3c(0:nua_act-1, 0:nub_inact-1, 0:nub_inact-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_inact
                     do c = 1 , nub_inact

                        denom = H1A_oo_act(i,i) + H1B_oo_inact(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1B_vv_inact(b,b) - H1B_vv_inact(c,c)


                        r3c(a, b, c, i, j, k) = r3c(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3c_100101


    subroutine update_r3c_100111(r3c, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3c(1:nua_act, 1:nub_inact, 1:nub_inact, 1:noa_act, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: r3c(0:nua_act-1, 0:nub_inact-1, 0:nub_inact-1, 0:noa_act-1, 0:nob_act-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_act
         do j = 1 , nob_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_inact
                     do c = 1 , nub_inact

                        denom = H1A_oo_act(i,i) + H1B_oo_act(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1B_vv_inact(b,b) - H1B_vv_inact(c,c)


                        r3c(a, b, c, i, j, k) = r3c(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3c_100111


    subroutine update_r3c_110001(r3c, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3c(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_inact, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: r3c(0:nua_act-1, 0:nub_act-1, 0:nub_inact-1, 0:noa_inact-1, 0:nob_inact-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = 1 , nub_inact

                        denom = H1A_oo_inact(i,i) + H1B_oo_inact(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1B_vv_act(b,b) - H1B_vv_inact(c,c)


                        r3c(a, b, c, i, j, k) = r3c(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3c_110001

subroutine update_r3c_110011(r3c, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3c(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_inact, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: r3c(0:nua_act-1, 0:nub_act-1, 0:nub_inact-1, 0:noa_inact-1, 0:nob_act-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , nob_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = 1 , nub_inact

                        denom = H1A_oo_inact(i,i) + H1B_oo_act(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1B_vv_act(b,b) - H1B_vv_inact(c,c)


                        r3c(a, b, c, i, j, k) = r3c(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3c_110011

    subroutine update_r3c_110110(r3c, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3c(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_act, 1:nob_inact)
      !f2py intent(in, out)  :: r3c(0:nua_act-1, 0:nub_act-1, 0:nub_inact-1, 0:noa_act-1, 0:nob_act-1, 0:nob_inact-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_act
         do j = 1 , nob_act
            do k = 1 , nob_inact
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = 1 , nub_inact

                        denom = H1A_oo_act(i,i) + H1B_oo_act(j,j) + H1B_oo_inact(k,k)&
                               -H1A_vv_act(a,a) - H1B_vv_act(b,b) - H1B_vv_inact(c,c)


                        r3c(a, b, c, i, j, k) = r3c(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3c_110110


    subroutine update_r3c_110101(r3c, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3c(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: r3c(0:nua_act-1, 0:nub_act-1, 0:nub_inact-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = 1 , nub_inact

                        denom = H1A_oo_act(i,i) + H1B_oo_inact(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1B_vv_act(b,b) - H1B_vv_inact(c,c)


                        r3c(a, b, c, i, j, k) = r3c(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3c_110101


    subroutine update_r3c_110111(r3c, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3c(1:nua_act, 1:nub_act, 1:nub_inact, 1:noa_act, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: r3c(0:nua_act-1, 0:nub_act-1, 0:nub_inact-1, 0:noa_act-1, 0:nob_act-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_act
         do j = 1 , nob_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = 1 , nub_inact

                        denom = H1A_oo_act(i,i) + H1B_oo_act(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1B_vv_act(b,b) - H1B_vv_inact(c,c)


                        r3c(a, b, c, i, j, k) = r3c(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3c_110111


    subroutine update_r3c_111001(r3c, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3c(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_inact, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: r3c(0:nua_act-1, 0:nub_act-1, 0:nub_act-1, 0:noa_inact-1, 0:nob_inact-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = 1 , nub_act

                        denom = H1A_oo_inact(i,i) + H1B_oo_inact(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1B_vv_act(b,b) - H1B_vv_act(c,c)


                        r3c(a, b, c, i, j, k) = r3c(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3c_111001


    subroutine update_r3c_111011(r3c, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3c(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_inact, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: r3c(0:nua_act-1, 0:nub_act-1, 0:nub_act-1, 0:noa_inact-1, 0:nob_act-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_inact
         do j = 1 , nob_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = 1 , nub_act

                        denom = H1A_oo_inact(i,i) + H1B_oo_act(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1B_vv_act(b,b) - H1B_vv_act(c,c)


                        r3c(a, b, c, i, j, k) = r3c(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3c_111011


    subroutine update_r3c_111100(r3c, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3c(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_inact)
      !f2py intent(in, out)  :: r3c(0:nua_act-1, 0:nub_act-1, 0:nub_act-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_inact-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = 1 , nob_inact
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = 1 , nub_act

                        denom = H1A_oo_act(i,i) + H1B_oo_inact(j,j) + H1B_oo_inact(k,k)&
                               -H1A_vv_act(a,a) - H1B_vv_act(b,b) - H1B_vv_act(c,c)


                        r3c(a, b, c, i, j, k) = r3c(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3c_111100


    subroutine update_r3c_111101(r3c, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3c(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: r3c(0:nua_act-1, 0:nub_act-1, 0:nub_act-1, 0:noa_act-1, 0:nob_inact-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_act
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = 1 , nub_act

                        denom = H1A_oo_act(i,i) + H1B_oo_inact(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1B_vv_act(b,b) - H1B_vv_act(c,c)


                        r3c(a, b, c, i, j, k) = r3c(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3c_111101


    subroutine update_r3c_111111(r3c, omega, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact), &
                              H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3c(1:nua_act, 1:nub_act, 1:nub_act, 1:noa_act, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: r3c(0:nua_act-1, 0:nub_act-1, 0:nub_act-1, 0:noa_act-1, 0:nob_act-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , noa_act
         do j = 1 , nob_act
            do k = 1 , nob_act
               do a = 1 , nua_act
                  do b = 1 , nub_act
                     do c = 1 , nub_act

                        denom = H1A_oo_act(i,i) + H1B_oo_act(j,j) + H1B_oo_act(k,k)&
                               -H1A_vv_act(a,a) - H1B_vv_act(b,b) - H1B_vv_act(c,c)


                        r3c(a, b, c, i, j, k) = r3c(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3c_111111


    subroutine update_r3d_100001(r3d, omega, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3d(1:nub_act, 1:nub_inact, 1:nub_inact, 1:nob_inact, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: r3d(0:nub_act-1, 0:nub_inact-1, 0:nub_inact-1, 0:nob_inact-1, 0:nob_inact-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , nob_inact
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nub_act
                  do b = 1 , nub_inact
                     do c = 1 , nub_inact

                        denom = H1B_oo_inact(i,i) + H1B_oo_inact(j,j) + H1B_oo_act(k,k)&
                               -H1B_vv_act(a,a) - H1B_vv_inact(b,b) - H1B_vv_inact(c,c)


                        r3d(a, b, c, i, j, k) = r3d(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3d_100001


    subroutine update_r3d_100011(r3d, omega, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3d(1:nub_act, 1:nub_inact, 1:nub_inact, 1:nob_inact, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: r3d(0:nub_act-1, 0:nub_inact-1, 0:nub_inact-1, 0:nob_inact-1, 0:nob_act-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , nob_inact
         do j = 1 , nob_act
            do k = 1 , nob_act
               do a = 1 , nub_act
                  do b = 1 , nub_inact
                     do c = 1 , nub_inact

                        denom = H1B_oo_inact(i,i) + H1B_oo_act(j,j) + H1B_oo_act(k,k)&
                               -H1B_vv_act(a,a) - H1B_vv_inact(b,b) - H1B_vv_inact(c,c)


                        r3d(a, b, c, i, j, k) = r3d(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3d_100011


    subroutine update_r3d_100111(r3d, omega, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3d(1:nub_act, 1:nub_inact, 1:nub_inact, 1:nob_act, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: r3d(0:nub_act-1, 0:nub_inact-1, 0:nub_inact-1, 0:nob_act-1, 0:nob_act-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , nob_act
         do j = 1 , nob_act
            do k = 1 , nob_act
               do a = 1 , nub_act
                  do b = 1 , nub_inact
                     do c = 1 , nub_inact

                        denom = H1B_oo_act(i,i) + H1B_oo_act(j,j) + H1B_oo_act(k,k)&
                               -H1B_vv_act(a,a) - H1B_vv_inact(b,b) - H1B_vv_inact(c,c)


                        r3d(a, b, c, i, j, k) = r3d(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3d_100111


    subroutine update_r3d_110001(r3d, omega, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3d(1:nub_act, 1:nub_act, 1:nub_inact, 1:nob_inact, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: r3d(0:nub_act-1, 0:nub_act-1, 0:nub_inact-1, 0:nob_inact-1, 0:nob_inact-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , nob_inact
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nub_act
                  do b = 1 , nub_act
                     do c = 1 , nub_inact

                        denom = H1B_oo_inact(i,i) + H1B_oo_inact(j,j) + H1B_oo_act(k,k)&
                               -H1B_vv_act(a,a) - H1B_vv_act(b,b) - H1B_vv_inact(c,c)


                        r3d(a, b, c, i, j, k) = r3d(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3d_110001


    subroutine update_r3d_110011(r3d, omega, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3d(1:nub_act, 1:nub_act, 1:nub_inact, 1:nob_inact, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: r3d(0:nub_act-1, 0:nub_act-1, 0:nub_inact-1, 0:nob_inact-1, 0:nob_act-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , nob_inact
         do j = 1 , nob_act
            do k = 1 , nob_act
               do a = 1 , nub_act
                  do b = 1 , nub_act
                     do c = 1 , nub_inact

                        denom = H1B_oo_inact(i,i) + H1B_oo_act(j,j) + H1B_oo_act(k,k)&
                               -H1B_vv_act(a,a) - H1B_vv_act(b,b) - H1B_vv_inact(c,c)


                        r3d(a, b, c, i, j, k) = r3d(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3d_110011


    subroutine update_r3d_110111(r3d, omega, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3d(1:nub_act, 1:nub_act, 1:nub_inact, 1:nob_act, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: r3d(0:nub_act-1, 0:nub_act-1, 0:nub_inact-1, 0:nob_act-1, 0:nob_act-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , nob_act
         do j = 1 , nob_act
            do k = 1 , nob_act
               do a = 1 , nub_act
                  do b = 1 , nub_act
                     do c = 1 , nub_inact

                        denom = H1B_oo_act(i,i) + H1B_oo_act(j,j) + H1B_oo_act(k,k)&
                               -H1B_vv_act(a,a) - H1B_vv_act(b,b) - H1B_vv_inact(c,c)


                        r3d(a, b, c, i, j, k) = r3d(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3d_110111


    subroutine update_r3d_111001(r3d, omega, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3d(1:nub_act, 1:nub_act, 1:nub_act, 1:nob_inact, 1:nob_inact, 1:nob_act)
      !f2py intent(in, out)  :: r3d(0:nub_act-1, 0:nub_act-1, 0:nub_act-1, 0:nob_inact-1, 0:nob_inact-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , nob_inact
         do j = 1 , nob_inact
            do k = 1 , nob_act
               do a = 1 , nub_act
                  do b = 1 , nub_act
                     do c = 1 , nub_act

                        denom = H1B_oo_inact(i,i) + H1B_oo_inact(j,j) + H1B_oo_act(k,k)&
                               -H1B_vv_act(a,a) - H1B_vv_act(b,b) - H1B_vv_act(c,c)


                        r3d(a, b, c, i, j, k) = r3d(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3d_111001



    subroutine update_r3d_111011(r3d, omega, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3d(1:nub_act, 1:nub_act, 1:nub_act, 1:nob_inact, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: r3d(0:nub_act-1, 0:nub_act-1, 0:nub_act-1, 0:nob_inact-1, 0:nob_act-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , nob_inact
         do j = 1 , nob_act
            do k = 1 , nob_act
               do a = 1 , nub_act
                  do b = 1 , nub_act
                     do c = 1 , nub_act

                        denom = H1B_oo_inact(i,i) + H1B_oo_act(j,j) + H1B_oo_act(k,k)&
                               -H1B_vv_act(a,a) - H1B_vv_act(b,b) - H1B_vv_act(c,c)


                        r3d(a, b, c, i, j, k) = r3d(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3d_111011



    subroutine update_r3d_111111(r3d, omega, &
                             H1B_oo_act, H1B_vv_act, H1B_oo_inact, H1B_vv_inact, &
                             shift, &
                             nob_act, nub_act, nob_inact, nub_inact)

      integer, intent(in)  :: nob_act, nub_act, nob_inact, nub_inact
      real(8), intent(in)  :: H1B_oo_act(1:nob_act, 1:nob_act), &
                              H1B_vv_act(1:nub_act, 1:nub_act), &
                              H1B_oo_inact(1:nob_inact, 1:nob_inact), &
                              H1B_vv_inact(1:nub_inact, 1:nub_inact)
      real(8), intent(in)  :: shift, omega

      real(8), intent(inout) :: r3d(1:nub_act, 1:nub_act, 1:nub_act, 1:nob_act, 1:nob_act, 1:nob_act)
      !f2py intent(in, out)  :: r3d(0:nub_act-1, 0:nub_act-1, 0:nub_act-1, 0:nob_act-1, 0:nob_act-1, 0:nob_act-1)

      integer :: i, j, k, a, b, c
      real(8) :: denom

      do i = 1 , nob_act
         do j = 1 , nob_act
            do k = 1 , nob_act
               do a = 1 , nub_act
                  do b = 1 , nub_act
                     do c = 1 , nub_act

                        denom = H1B_oo_act(i,i) + H1B_oo_act(j,j) + H1B_oo_act(k,k)&
                               -H1B_vv_act(a,a) - H1B_vv_act(b,b) - H1B_vv_act(c,c)


                        r3d(a, b, c, i, j, k) = r3d(a, b, c, i, j, k)/(omega - denom + shift)


                     end do
                  end do
               end do
            end do
         end do
      end do

end subroutine update_r3d_111111


end module eomcc_active_loops