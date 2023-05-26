program test_quantile_emulator
use tau_neural_net_quantile
implicit none

integer, parameter :: mgncol=5
real(r8), dimension(mgncol) :: qc, nc, qr, nr, rho, qc_tend, qr_tend, nc_tend, &
    nr_tend, lamc, lamr, lcldm, n0r, pgam, precip_frac
real(r8) :: qsmall, t_start, t_end
integer :: i, num_loops, t
print *, "load emulators"
call initialize_tau_emulators
print *, "loaded emulators"
qc = (/ 5e-10_r8, 1e-5_r8, 1e-3_r8, 2e-3_r8, 5.2e-4_r8 /)
qr = (/ 1e-10_r8, 1e-8_r8, 1e-2_r8, 1e-4_r8, 2e-3_r8 /)
nc = (/ 10.0_r8, 100.0_r8, 500.0_r8, 50000.0_r8, 1.0_r8 /)
nr = (/ 10.0_r8, 1.0_r8, 1000.0_r8, 1e6_r8, 10000.0_r8 /)
rho = (/ 0.9_r8, 0.8_r8, 0.6_r8, 0.9_r8, 0.5_r8 /)
lamc = (/ 10000.0_r8, 20000.0_r8, 30000.0_r8, 40000.0_r8, 450000.0_r8 /)
lamr = (/ 5000.0_r8, 10000.0_r8, 15000.0_r8, 20000.0_r8, 25000.0_r8 /)
lcldm = (/ 0.5_r8, 0.4_r8, 0.3_r8, 0.6_r8, 0.25_r8 /)
n0r = (/ 0.5e5_r8, 1.0e6_r8, 1.1e7_r8, 1.12e3_r8, 2.0e4_r8 /)
pgam = (/ 10.0_r8, 50.0_r8, 25.0_r8, 19.0_r8, 100.0_r8 /)
precip_frac = (/ 0.3_r8, 0.4_r8, 0.5_r8, 0.6_r8, 0.7_r8 /)
qsmall = 1.0e-18_r8
num_loops = 1000
print *, qc
call cpu_time(t_start)
do t=1,num_loops
call tau_emulate_cloud_rain_interactions(qc, nc, qr, nr, rho, lcldm, &
                                         precip_frac, mgncol, qsmall, &
                                         qc_tend, qr_tend, nc_tend, nr_tend)
end do
call cpu_time(t_end)
print *, "Timing: ", t_end - t_start
do i=1,mgncol
    print *, qc_tend(i), qr_tend(i), nc_tend(i), nr_tend(i)
end do
end program test_quantile_emulator
