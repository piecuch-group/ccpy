module ccp_matrix

      implicit none

      contains

              !subroutine get_t3_vector(t3a,t3b,t3c,t3d,noa,nua,nob,nub)
              !
              !        integer, intent(in) :: noa, nua, nob, nub
              !        real(kind=8), intent(in) :: t3a(nua,nua,nua,noa,noa,noa),&
              !                            t3b(nua,nua,nub,noa,noa,nob),&
              !                            t3c(nua,nub,nub,noa,nob,nob),&
              !                            t3d(nub,nub,nub,nob,nob,nob)
              !
              !        real(kind=8), intent(out) :: t3a_vec(

              subroutine build_HT3(HT3A,HT3B,HT3C,HT3D,&
                                    t3a_p,t3b_p,t3c_p,t3d_p,&
                                    list_of_triples_A,&
                                    list_of_triples_B,&
                                    list_of_triples_C,&
                                    list_of_triples_D,&
                                    fA_oo,fA_vv,fB_oo,fB_vv,&
                                    vA_oovv,vB_oovv,vC_oovv,t2a,t2b,t2c,&
                                    H1A_oo,H1A_vv,H1A_ov,&
                                    H1B_oo,H1B_vv,H1B_ov,&
                                    H2A_oooo,H2A_vvvv,H2A_voov,H2A_vooo,H2A_vvov,H2A_ooov,H2A_vovv,&
                                    H2B_oooo,H2B_vvvv,H2B_voov,H2B_ovvo,H2B_vovo,H2B_ovov,H2B_vooo,&
                                    H2B_ovoo,H2B_vvov,H2B_vvvo,H2B_ooov,H2B_oovo,H2B_vovv,H2B_ovvv,&
                                    H2C_oooo,H2C_vvvv,H2C_voov,H2C_vooo,H2C_vvov,H2C_ooov,H2C_vovv,&
                                    H3A_vooovo,H3A_vovovv,&
                                    noa,nua,nob,nub,n3a_p,n3b_p,n3c_p,n3d_p)

                        integer, intent(in) :: noa, nua, nob, nub, n3a_p, n3b_p, n3c_p, n3d_p
                        integer, intent(in) :: list_of_triples_A(n3a_p,6), list_of_triples_B(n3b_p,6),&
                                               list_of_triples_C(n3c_p,6), list_of_triples_D(n3d_p,6) 
                        real(kind=8), intent(in) ::  vA_oovv(noa,noa,nua,nua), vB_oovv(noa,nob,nua,nub),&
                                    vC_oovv(nob,nob,nub,nub), fA_oo(noa,noa), fA_vv(nua,nua), fB_oo(noa,noa), fB_vv(nub,nub),&
                                    t2a(nua,nua,nob,nob), t2b(nua,nub,noa,nob),t2c(nub,nub,nob,nob),&
                                    H1A_oo(noa,noa),H1A_vv(nua,nua),H1A_ov(noa,nua),&
                                    H1B_oo(nob,nob),H1B_vv(nub,nub),H1B_ov(nob,nub),&
                                    H2A_oooo(noa,noa,noa,noa),H2A_vvvv(nua,nua,nua,nua),H2A_voov(nua,noa,noa,nua),&
                                    H2A_vooo(nua,noa,noa,noa),H2A_vvov(nua,nua,noa,nua),H2A_ooov(noa,noa,noa,nua),&
                                    H2A_vovv(nua,noa,nua,nua),&
                                    H2B_oooo(noa,nob,noa,nob),H2B_vvvv(nua,nub,nua,nub),H2B_voov(nua,nob,noa,nub),&
                                    H2B_ovvo(noa,nub,nua,nob),H2B_vovo(nua,nob,nua,nob),H2B_ovov(noa,nub,noa,nub),&
                                    H2B_vooo(nua,nob,noa,nob),H2B_ovoo(noa,nub,noa,nob),H2B_vvov(nua,nub,noa,nub),&
                                    H2B_vvvo(nua,nub,nua,nob),H2B_ooov(noa,nob,noa,nub),H2B_oovo(noa,nob,nua,nob),&
                                    H2B_vovv(nua,nob,nua,nub),H2B_ovvv(noa,nub,nua,nub),&
                                    H2C_oooo(nob,nob,nob,nob),H2C_vvvv(nub,nub,nub,nub),H2C_voov(nub,nob,nob,nub),&
                                    H2C_vooo(nub,nob,nob,nob),H2C_vvov(nub,nub,nob,nub),H2C_ooov(nob,nob,nob,nub),&
                                    H2C_vovv(nub,nob,nub,nub),&
                                    H3A_vovovv(nua,noa,nua,noa,nua,nua),&
                                    H3A_vooovo(nua,noa,noa,noa,nua,noa),&
                                    t3a_p(n3a_p), t3b_p(n3b_p), t3c_p(n3c_p), t3d_p(n3d_p)
                        !real(kind=8), intent(in) :: shift

                        real(kind=8), intent(out) :: HT3A(n3a_p), HT3B(n3b_p), HT3C(n3c_p), HT3d(n3d_p)

                        real(kind=8) :: onebody, twobody, threebody, denom, val,&
                                dgm1

                        integer :: i1, j1, k1, a1, b1, c1, i2, j2, k2, a2, b2, c2, idet, jdet, pos(5)

                        pos(1) = 0
                        pos(2) = n3a_p
                        pos(3) = n3a_p+n3b_p
                        pos(4) = n3a_p+n3b_p+n3c_p
                        pos(5) = n3a_p+n3b_p+n3c_p+n3d_p


                        HT3A = 0.0d0
                        do idet = 1,n3a_p
                           a1 = list_of_triples_A(idet,1) + 1
                           b1 = list_of_triples_A(idet,2) + 1
                           c1 = list_of_triples_A(idet,3) + 1
                           i1 = list_of_triples_A(idet,4) + 1
                           j1 = list_of_triples_A(idet,5) + 1
                           k1 = list_of_triples_A(idet,6) + 1

                           do jdet = 1,n3a_p
                              a2 = list_of_triples_A(jdet,1) + 1
                              b2 = list_of_triples_A(jdet,2) + 1
                              c2 = list_of_triples_A(jdet,3) + 1
                              i2 = list_of_triples_A(jdet,4) + 1
                              j2 = list_of_triples_A(jdet,5) + 1
                              k2 = list_of_triples_A(jdet,6) + 1

                              ! diagram 1 -A(i/jk)h(mi)t3a(abcmjk)
                              dgm1 = -delta(a1,a2)*delta(b1,b2)*delta(c1,c2)&
                                      *(delta(j1,j2)*delta(k1,k2)*h1A_oo(i2,i1)& ! +(1)
                                       -delta(i1,j2)*delta(k1,k2)*h1A_oo(i2,j1)& ! -(i1,j1)
                                       -delta(j1,j2)*delta(i1,k2)*h1A_oo(i2,k1)& ! -(i1,k1)
                                       -delta(j1,i2)*delta(k1,k2)*h1A_oo(j2,i1)& ! -(i2,j2)
                                       -delta(j1,j2)*delta(k1,i2)*h1A_oo(k2,i1)& ! -(i2,k2)
                                       +delta(i1,i2)*delta(k1,k2)*h1A_oo(j2,j1)& ! +(i1,j1)(i2,j2)
                                       +delta(i1,j2)*delta(k1,i2)*h1A_oo(k2,j1)& ! +(i1,j1)(i2,k2)
                                       +delta(j1,i2)*delta(i1,k2)*h1A_oo(j2,k1)& ! +(i1,k1)(i2,j2)
                                       +delta(j1,j2)*delta(i1,k2)*h1A_oo(k2,k1)& ! +(i1,k1)(i2,k2)
                                       -delta(k1,j2)*delta(j1,k2)*h1A_oo(i2,i1)& ! -(j1,k1)
                                       +delta(i1,j2)*delta(j1,k2)*h1A_oo(i2,k1)& ! +(j1,k1)(i1,j1)
                                       +delta(k1,j2)*delta(i1,k2)*h1A_oo(i2,j1)& ! +(j1,k1)(i1,k1)
                                       +delta(k1,i2)*delta(j1,k2)*h1A_oo(j2,i1)& ! +(j1,k1)(i2,j2)
                                       +delta(k1,j2)*delta(j1,i2)*h1A_oo(k2,i1)& ! +(j1,k1)(i2,k2)
                                       -delta(i1,i2)*delta(j1,k2)*h1A_oo(j2,k1)& ! -(j1,k1)(i1,j1)(i2,j2)
                                       -delta(i1,j2)*delta(j1,i2)*h1A_oo(k2,k1)& ! -(j1,k1)(i1,j1)(i2,k2)
                                       -delta(k1,i2)*delta(i1,k2)*h1A_oo(j2,j1)& ! -(j1,k1)(i1,k1)(i2,j2)
                                       -delta(k1,j2)*delta(i1,k2)*h1A_oo(k2,j1)) ! -(j1,k1)(i1,k1)(i2,k2)

                              HT3A(idet) = HT3A(idet) + (dgm1)*t3a_p(jdet)
                           end do
                                        
                        end do   
              end subroutine build_HT3


              function delta(p,q) result(val)

                      integer, intent(in) :: p, q
                      real(kind=8) :: val

                      val = 0.0d0
                      if (p == q) then
                         val = 1.0d0
                      end if

              end function delta

end module ccp_matrix

