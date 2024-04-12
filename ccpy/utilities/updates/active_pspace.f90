module active_pspace

      implicit none

      contains

              subroutine get_active_triples_aaa(excitations_aaa,&
                                                orbsym,&
                                                num_active,&
                                                target_irrep,&
                                                n3aaa,noa,nua,norb)

                    ! input variables
                    integer, intent(in) :: noa, nua, norb, n3aaa
                    integer, intent(in) :: num_active
                    integer, intent(in) :: target_irrep
                    integer, intent(in) :: orbsym(norb)
                    ! output variable
                    integer, intent(out) :: excitations_aaa(n3aaa,6)
                    ! local variables
                    integer :: i, j, k, a, b, c

                    do i=1,noa
                       isym = orbsym(i)
                       do j=i+1,noa 
                          jsym = orbsym(j)
                          do k=j+1,noa
                             ksym = orbsym(k)
                             do a=1,nua 
                                asym = orbsym(a + noa)
                                do b=a+1,nua
                                   bsym = orbsym(b + noa)
                                   do c=b+1,nua
                                      csym = orbsym(c + noa)
                                   end do
                                end do
                             end do
                          end do
                       end do
                    end do


              end subroutine get_active_triples_aaa

              subroutine get_active_triples_aab()

              end subroutine get_active_triples_aab

              subroutine get_active_triples_abb()

              end subroutine get_active_triples_abb

              subroutine get_active_triples_bbb()

              end subroutine get_active_triples_bbb

end module active_space
