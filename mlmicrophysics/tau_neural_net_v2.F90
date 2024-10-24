module tau_neural_net_quantile

    use shr_kind_mod,   only: r8=>shr_kind_r8

    use module_neural_net, only : Dense, init_neural_net, load_quantile_scale_values
    use module_neural_net, only : quantile_transform, quantile_inv_transform, neural_net_predict
    use tester, only : write_test_values

    implicit none
    integer, parameter, public :: i8 = selected_int_kind(18)
    integer, parameter :: num_inputs = 9
    integer, parameter :: num_outputs = 3
    integer, parameter :: batch_size = 1

    ! Neural networks and scale values saved within the scope of the module.
    ! Need to call initialize_tau_emulators to load weights and tables from disk.
    type(Dense), allocatable, save :: q_all(:)
    real(r8), dimension(:, :), allocatable, save :: input_scale_values
    real(r8), dimension(:, :), allocatable, save :: output_scale_values
contains


    subroutine initialize_tau_emulators( stochastic_emulated_filename_quantile, stochastic_emulated_filename_input_scale, &
                                         stochastic_emulated_filename_output_scale, iulog, errstring)

    ! Load neural network netCDF files and scaling values. Values are placed in to emulators,
    ! input_scale_values, and output_scale_values.
    character(len=*), intent(in) ::  stochastic_emulated_filename_quantile, stochastic_emulated_filename_input_scale, &
                                     stochastic_emulated_filename_output_scale
    integer,          intent(in)  :: iulog
    character(128),   intent(out) :: errstring  ! output status (non-blank for error return)

        errstring = ''

        write(iulog,*) "Begin loading neural nets"
        write(iulog,*) "emulated filepath is ", stochastic_emulated_filename_quantile
        write(iulog,*) "input filepath is ", stochastic_emulated_filename_input_scale
        write(iulog,*) "output filepath is ", stochastic_emulated_filename_output_scale
        call init_neural_net(trim(stochastic_emulated_filename_quantile), batch_size, q_all, iulog, errstring)
        if (trim(errstring) /= '') return
        write(iulog,*) "End loading neural nets"
        ! Load the scale values from a csv file.
        call load_quantile_scale_values(trim(stochastic_emulated_filename_input_scale), input_scale_values, iulog, errstring)
        call load_quantile_scale_values(trim(stochastic_emulated_filename_output_scale), output_scale_values, iulog, errstring)
        write(iulog,*) "Loaded neural nets scaling values"

    end subroutine initialize_tau_emulators


    subroutine tau_emulated_cloud_rain_interactions(qc, nc, qr, nr, pgam, lamc, lamr, n0r, rho, lcldm, &
            precip_frac, mgncol, q_small, qc_tend, qr_tend, nc_tend, nr_tend, iulog)
        ! Calculates emulated tau microphysics tendencies from neural networks.
        !
        ! Input args:
        !   qc: cloud water mixing ratio in kg kg-1
        !   nc: cloud water number concentration in particles m-3
        !   qr: rain water mixing ratio in kg kg-1
        !   nr: rain water number concentration in particles m-3
        !   pgam: cloud liquid droplet size parameter
        !   lamc: cloud liquid size distribution parameter (slope)
        !   n0r: rain size parameter (intercept)
        !   lamr: rain size parameter (slope)
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
        real(r8), dimension(mgncol), intent(in) :: qc, qr, nc, nr, pgam, lamc, n0r, lamr, rho, lcldm, precip_frac
        real(r8), intent(in) :: q_small
        real(r8), dimension(mgncol), intent(out) :: qc_tend, qr_tend, nc_tend, nr_tend
        integer(i8) :: i, j
        real(r8), dimension(batch_size, num_inputs) :: nn_inputs, nn_quantile_inputs
        real(r8), dimension(batch_size, num_outputs) :: nn_quantile_outputs, nn_outputs
        real(r8), parameter :: dt = 1800.0
        integer,  intent(in) :: iulog
        character(len=128) :: filename

        do i = 1, mgncol
            if (qc(i) >= q_small) then
                nn_inputs(1, 1) = qc(i)
                nn_inputs(1, 2) = qr(i)
                nn_inputs(1, 3) = nc(i)
                nn_inputs(1, 4) = nr(i)
                nn_inputs(1, 5) = pgam(i)
                nn_inputs(1, 6) = lamc(i)
                nn_inputs(1, 7) = lamr(i)
                nn_inputs(1, 8) = n0r(i)
                nn_inputs(1, 9) = rho(i)
                ! nn_inputs(1, 6) = precip_frac(i)
                ! nn_inputs(1, 7) = lcldm(i)
                
                call quantile_transform(nn_inputs, input_scale_values, nn_quantile_inputs)

                call neural_net_predict(nn_quantile_inputs, q_all, nn_quantile_outputs, iulog)

                call quantile_inv_transform(nn_quantile_outputs, output_scale_values, nn_outputs)
                
                qc_tend(i) = nn_outputs(1, 1)
                qr_tend(i) = -qc_tend(i)
                nc_tend(i) = nn_outputs(1, 2)
                nr_tend(i) = nn_outputs(1, 3)
            else
                qc_tend(i) = 0._r8
                qr_tend(i) = 0._r8
                nc_tend(i) = 0._r8
                nr_tend(i) = 0._r8
            end if
        end do

    end subroutine tau_emulated_cloud_rain_interactions
end module tau_neural_net_quantile
