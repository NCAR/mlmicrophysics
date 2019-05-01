program test_emulator
use tau_neural_net
implicit none
integer, parameter :: mgncol=3
real(r8), dimension(mgncol) :: qc, nc, qr, nr, rho, qc_tend, qr_tend, nc_tend, &
    nr_tend
real(r8) :: qsmall
integer :: i
print *, "load emulators"
call initialize_tau_emulators
print *, "loaded emulators"
qc = (/ 1e-10, 1e-5, 1e-3 /)
qr = (/ 1e-10, 1e-8, 1e-2 /)
nc = (/ 10.0, 100.0, 500.0 /)
nr = (/ 10.0, 1.0, 1000.0 /)
rho = (/ 0.9, 0.8, 0.6 /)
qsmall = 1.0e-18
print *, qc
call tau_emulate_cloud_rain_interactions(qc, nc, qr, nr, rho, qsmall, mgncol, &
                                         qc_tend, qr_tend, nc_tend, nr_tend)
do i=1,mgncol
    print *, qc_tend(i), qr_tend(i), nc_tend(i), nr_tend(i)
end do
end program test_emulator
