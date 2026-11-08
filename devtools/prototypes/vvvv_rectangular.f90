! Experimental mixed-spin T2 only. No production linkage.
! phase values are sums of worker seconds, not elapsed phase durations.
subroutine rectangular(resid,ra,rb,t2,oa,ob,va,vb,q,ba,bb,phase)
 use omp_lib
 implicit none
 integer,intent(in)::oa,ob,va,vb,q,ba,bb
 real(8),intent(in)::ra(q,va,va),rb(q,vb,vb),t2(ob*oa,vb*va)
 real(8),intent(out)::resid(ob*oa,vb,va),phase(3)
 real(8),allocatable::y(:,:,:,:),z(:,:),work(:,:)
 real(8)::start,p1,p2,p3
 integer::tile,na,nb,a,b,ka,kb,e,nta,ntb,col
 nta=(va+ba-1)/ba
 ntb=(vb+bb-1)/bb
 p1=0; p2=0; p3=0
 !$omp parallel private(y,z,work,tile,na,nb,a,b,ka,kb,e,col,start) reduction(+:p1,p2,p3)
 allocate(y(vb,bb,va,ba),z(vb*va,ba*bb),work(ob*oa,ba*bb))
 !$omp do schedule(static)
 do tile=0,nta*ntb-1
   a=mod(tile,nta)*ba+1
   b=(tile/nta)*bb+1
   na=min(ba,va-a+1); nb=min(bb,vb-b+1)
   start=omp_get_wtime()
   ! y's leading dimension includes padded bb; actual b count may be smaller.
   call dgemm('t','n',vb*nb,va*na,q,1d0,rb(:,:,b:b+nb-1),q,ra(:,:,a:a+na-1),q,0d0,y,vb*bb)
   p1=p1+omp_get_wtime()-start
   start=omp_get_wtime()
   do ka=1,na
     do kb=1,nb
       col=kb+(ka-1)*nb
       do e=1,va
         z((e-1)*vb+1:e*vb,col)=y(:,kb,e,ka)
       end do
     end do
   end do
   p2=p2+omp_get_wtime()-start
   start=omp_get_wtime()
   call dgemm('n','n',oa*ob,na*nb,va*vb,1d0,t2,oa*ob,z,va*vb,0d0,work,oa*ob)
   do ka=1,na
     resid(:,b:b+nb-1,a+ka-1)=work(:,(ka-1)*nb+1:ka*nb)
   end do
   p3=p3+omp_get_wtime()-start
 end do
 !$omp end do
 deallocate(y,z,work)
 !$omp end parallel
 phase=[p1,p2,p3]
end subroutine rectangular
