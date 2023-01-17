module tau_neural_net_quantile
    use module_neural_net
    implicit none
    integer, parameter, public :: i8 = selected_int_kind(18)
    character(len = *), parameter :: neural_net_path = "/glade/work/dgagne/cam_mp_run6_quantile_nn/"
    integer, parameter :: num_inputs = 7
    integer, parameter :: num_outputs = 3
    integer, parameter :: batch_size = 1

    ! Neural networks and scale values saved within the scope of the module.
    ! Need to call initialize_tau_emulators to load weights and tables from disk.
    type(Dense), allocatable, save :: q_all(:)
    real(r8), dimension(:, :), allocatable, save :: input_scale_values
    real(r8), dimension(:, :), allocatable, save :: output_scale_values
contains


    subroutine initialize_tau_emulators
        ! Load neural network netCDF files and scaling values. Values are placed in to emulators,
        ! input_scale_values, and output_scale_values.
        ! Args:
        !   neural_net_path: Path to neural networks
        !
        !character(len=*), intent(in) :: neural_net_path
        ! Load each neural network from the neural net directory
        print*, "Begin loading neural nets"
        call init_neural_net(neural_net_path // "quantile_neural_net_fortran.nc", batch_size, q_all)
        print*, "End loading neural nets"
        ! Load the scale values from a csv file.
        call load_quantile_scale_values(neural_net_path // "input_quantile_scaler.nc", input_scale_values)
        call load_quantile_scale_values(neural_net_path // "output_quantile_scaler.nc", output_scale_values)
        print*, "Loaded neural nets scaling values"
    end subroutine initialize_tau_emulators


    subroutine tau_emulate_cloud_rain_interactions(qc, nc, qr, nr, rho, lcldm, &
            precip_frac, mgncol, q_small, qc_tend, qr_tend, nc_tend, nr_tend)
        ! Calculates emulated tau microphysics tendencies from neural networks.
        !
        ! Input args:
        !   qc: cloud water mixing ratio in kg kg-1
        !   nc: cloud water number concentration in particles m-3
        !   qr: rain water mixing ratio in kg kg-1
        !   nr: rain water number concentration in particles m-3
        !   rho: density of air in kg m-3
        !   q_small: minimum cloud water mixing ratio value for running the microphysics
        !   mgncol: MG number of grid cells in vertical column
        ! Output args:
        !    qc_tend: qc tendency
        !    qr_tend: qr tendency
        !    nc_tend: nc tendency
        !    nr_tend: nr tendency
        !
        integer, intent(in) :: mgncol
        real(r8), dimension(mgncol), intent(in) :: qc, qr, nc, nr, rho, lcldm, precip_frac
        real(r8), intent(in) :: q_small
        real(r8), dimension(mgncol), intent(out) :: qc_tend, qr_tend, nc_tend, nr_tend
        integer(i8) :: i, j
        real(r8), dimension(batch_size, num_inputs) :: nn_inputs, nn_quantile_inputs
        real(r8), dimension(batch_size, num_outputs) :: nn_quantile_outputs, nn_outputs
        real(r8), parameter :: dt = 1800.0
        do i = 1, mgncol
            if (qc(i) >= q_small) then
                nn_inputs(1, 1) = qc(i)
                nn_inputs(1, 2) = qr(i)
                nn_inputs(1, 3) = nc(i)
                nn_inputs(1, 4) = nr(i)
                nn_inputs(1, 5) = rho(i)
                nn_inputs(1, 6) = precip_frac(i)
                nn_inputs(1, 7) = lcldm(i)
                call quantile_transform(nn_inputs, input_scale_values, nn_quantile_inputs)
                call neural_net_predict(nn_quantile_inputs, q_all, nn_quantile_outputs)
                call quantile_inv_transform(nn_quantile_outputs, output_scale_values, nn_outputs)
                qr_tend(i) = (nn_outputs(1, 1) - qr(i)) / dt
                qc_tend(i) = -qr_tend(i)
                nc_tend(i) = (nn_outputs(1, 2) - nc(i)) / dt
                nr_tend(i) = (nn_outputs(1, 3) - nr(i)) / dt
            else
                qc_tend(i) = 0._r8
                qr_tend(i) = 0._r8
                nc_tend(i) = 0._r8
                nr_tend(i) = 0._r8
            end if
        end do
    end subroutine tau_emulate_cloud_rain_interactions

end module tau_neural_net_quantile
