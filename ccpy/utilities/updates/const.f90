module const
implicit none

  ! 32-bit (4-byte) integer
  integer, parameter :: int_32 = selected_int_kind(6)
  ! 64-bit (8-byte) integer
  integer, parameter :: int_64 = selected_int_kind(15)

  ! 32-bit (4-byte) real
  integer, parameter :: sp = selected_real_kind(6, 37)
  ! 64-bit (8-byte) real
  integer, parameter :: dp = selected_real_kind(15, 307)
  ! 128-bit (16-byte) real
  integer, parameter :: qp = selected_real_kind(33, 4931)
  ! Selected precision
  integer, parameter :: p = dp


end module const
