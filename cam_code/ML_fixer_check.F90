module ML_fixer_check


contains
subroutine ML_fixer_calc(mgncol,dt,qc,nc,qr,nr,qctend,nctend,qrtend,nrtend,fixer,qc_fixer, nc_fixer, qr_fixer, nr_fixer)

use shr_kind_mod,   only: r8=>shr_kind_r8
use micro_mg_utils, only: pi, rhow

integer, intent(in) :: mgncol
real(r8), intent(in) :: dt
real(r8), intent(in) :: qc(mgncol)
real(r8), intent(in) :: nc(mgncol)
real(r8), intent(in) :: qr(mgncol)
real(r8), intent(in) :: nr(mgncol)
real(r8), intent(inout) :: qctend(mgncol)
real(r8), intent(inout) :: nctend(mgncol)
real(r8), intent(inout) :: qrtend(mgncol)
real(r8), intent(inout) :: nrtend(mgncol)

real(r8), intent(out) :: qc_fixer(mgncol)
real(r8), intent(out) :: nc_fixer(mgncol)
real(r8), intent(out) :: qr_fixer(mgncol)
real(r8), intent(out) :: nr_fixer(mgncol)

real(r8), intent(out) :: fixer(mgncol)

real(r8) :: qc_tmp, nc_tmp, qr_tmp, nr_tmp
integer :: i

fixer = 0._r8

qc_fixer = 0._r8
qr_fixer = 0._r8
nc_fixer = 0._r8
nr_fixer = 0._r8

do i = 1,mgncol
   qc_tmp = qc(i)+qctend(i)*dt
   nc_tmp = nc(i)+nctend(i)*dt
   qr_tmp = qr(i)+qrtend(i)*dt
   nr_tmp = nr(i)+nrtend(i)*dt

   if( qc_tmp.lt.0._r8 ) then
      fixer(i) = 1._r8
      qctend(i) = -qc(i)/dt
      qrtend(i) = qc(i)/dt
      nctend(i) = -nc(i)/dt   
   end if
   if( qr_tmp.lt.0._r8 ) then
      fixer(i) = 1._r8
      qrtend(i) = -qr(i)/dt
      qctend(i) = qr(i)/dt
      nrtend(i) = -nr(i)/dt   
   end if
   if( nc_tmp.lt.0._r8 ) then
      fixer(i) = 1._r8
      if( qc_tmp.gt.0._r8 ) then
         nc_tmp = qc_tmp/(4._r8/3._r8*pi*(5.e-5_r8)**3._r8*rhow)
         nctend(i) = (nc_tmp-nc(i))/dt
      else
         nctend(i) = -nc(i)/dt
      end if
   end if
   if( nr_tmp.lt.0._r8 ) then
      fixer(i) = 1._r8
      if(qr_tmp.gt.0._r8) then
         nr_tmp = qr_tmp/(4._r8/3._r8*pi*(5.e-5_r8)**3._r8*rhow)
         nrtend(i) = (nr_tmp-nr(i))/dt
      else
         nrtend(i) = -nr(i)/dt
      end if
   end if

   qc_fixer(i) = qc(i)+qctend(i)*dt-qc_tmp
   qr_fixer(i) = qr(i)+qrtend(i)*dt-qr_tmp
   nc_fixer(i) = nc(i)+nctend(i)*dt-nc_tmp
   nr_fixer(i) = nr(i)+nrtend(i)*dt-nr_tmp
end do

end subroutine ML_fixer_calc

end module ML_fixer_check
