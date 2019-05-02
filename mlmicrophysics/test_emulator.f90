program test_emulator
use tau_neural_net
implicit none
integer, parameter :: mgncol=5
real(r8), dimension(mgncol) :: qc, nc, qr, nr, rho, qc_tend, qr_tend, nc_tend, &
    nr_tend
real(r8) :: qsmall
integer :: i
print *, "load emulators"
call initialize_tau_emulators
print *, "loaded emulators"
qc = (/ 1e-10_r8, 1e-5_r8, 1e-3_r8, 2e-3_r8, 1e-9_r8 /)
qr = (/ 1e-10_r8, 1e-8_r8, 1e-2_r8, 1e-4_r8, 2e-5_r8 /)
nc = (/ 10.0_r8, 100.0_r8, 500.0_r8, 50000.0_r8, 10.0_r8 /)
nr = (/ 10.0_r8, 1.0_r8, 1000.0_r8, 1e6_r8, 10000.0_r8 /)
rho = (/ 0.9_r8, 0.8_r8, 0.6_r8, 0.9_r8, 0.5_r8 /)
qsmall = 1.0e-18_r8
print *, qc
call tau_emulate_cloud_rain_interactions(qc, nc, qr, nr, rho, qsmall, mgncol, &
                                         qc_tend, qr_tend, nc_tend, nr_tend)
do i=1,mgncol
    print *, qr_tend(i), nc_tend(i), nr_tend(i)
end do
end program test_emulator
