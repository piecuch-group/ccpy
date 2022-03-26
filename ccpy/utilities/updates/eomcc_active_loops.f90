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

  subroutine update_r3a_100001(r3a, &
                             H1A_oo_act, H1A_vv_act, H1A_oo_inact, H1A_vv_inact, &
                             shift, &
                             noa_act, nua_act, noa_inact, nua_inact)

      integer, intent(in)  :: noa_act, nua_act, noa_inact, nua_inact
      real(8), intent(in)  :: H1A_oo_act(1:noa_act, 1:noa_act), &
                              H1A_vv_act(1:nua_act, 1:nua_act), &
                              H1A_oo_inact(1:noa_inact, 1:noa_inact), &
                              H1A_vv_inact(1:nua_inact, 1:nua_inact)
      real(8), intent(in)  :: shift

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












end module eomcc_active_loops