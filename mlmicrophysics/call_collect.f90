module call_collect
implicit none
contains
subroutine collect_step(deltatin, t, rho, qcin, ncin, qrin, nrin, lcldm, precip_frac, &
                        qctend_TAU, nctend_TAU, qrtend_TAU, nrtend_TAU, mgncol)
    use stochastic_collect_tau_cam
    integer, parameter :: r8 = selected_real_kind(12)
    integer, parameter  :: n_bins = 35
    integer, intent(inout) :: mgncol
    real(kind=r8), intent(in) :: deltatin
    real(kind=r8), intent(in) :: t(mgncol)
    real(kind=r8), intent(in) :: rho(mgncol)
    real(kind=r8), intent(in) :: qcin(mgncol)
    real(kind=r8), intent(in) :: ncin(mgncol)
    real(kind=r8), intent(in) :: qrin(mgncol)
    real(kind=r8), intent(in) :: nrin(mgncol)
    real(kind=r8), intent(in) :: lcldm(mgncol)
    real(kind=r8), intent(in) :: precip_frac(mgncol)

    real(kind=r8), intent(out) :: qctend_TAU(mgncol)
    real(kind=r8), intent(out) :: nctend_TAU(mgncol)
    real(kind=r8), intent(out) :: qrtend_TAU(mgncol)
    real(kind=r8), intent(out) :: nrtend_TAU(mgncol)
    

    real(kind=r8) :: scale_qc(mgncol)
    real(kind=r8) :: scale_qr(mgncol)
    real(kind=r8) :: scale_nc(mgncol)
    real(kind=r8) :: scale_nr(mgncol)
    real(kind=r8) :: amk_c(mgncol,n_bins)
    real(kind=r8) :: ank_c(mgncol,n_bins)
    real(kind=r8) :: amk_r(mgncol,n_bins)
    real(kind=r8) :: ank_r(mgncol,n_bins)
    real(kind=r8) :: amk(mgncol,n_bins)
    real(kind=r8) :: ank(mgncol,n_bins)
    real(kind=r8) :: amk_out(mgncol,n_bins)
    real(kind=r8) :: ank_out(mgncol,n_bins)
    real(kind=r8) :: qcin_new(mgncol)
    real(kind=r8) :: ncin_new(mgncol)
    real(kind=r8) :: qrin_new(mgncol)
    real(kind=r8) :: nrin_new(mgncol)
    real(kind=r8) :: gmnnn_lmnnn_TAU(mgncol)
    call stochastic_collect_tau_tend(deltatin, t, rho, qcin, qrin, qcin, &
                                     ncin, qrin, nrin, lcldm, precip_frac, &
                                     qcin_new, ncin_new, qrin_new, nrin_new, &
                                     qctend_TAU, nctend_TAU, qrtend_TAU, &
                                     nrtend_TAU, &
                                     scale_qc, scale_nc, scale_qr, scale_nr, &
                                     amk_c, ank_c, amk_r, ank_r, amk, ank, &
                                     amk_out, ank_out, gmnnn_lmnnn_TAU, mgncol)
end subroutine
end module call_collect
