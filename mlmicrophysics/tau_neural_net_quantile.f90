module tau_neural_net_batch
    use module_neural_net
    implicit none
    integer, parameter, public :: i8 = selected_int_kind(18)
    character(len=*), parameter :: neural_net_path = "/glade/p/cisl/aiml/dgagne/cam_run5_models_20190726/"
    integer, parameter :: num_inputs = 11
    integer, parameter :: num_outputs = 4
    integer, parameter :: batch_size = 1
    type tau_emulators
        type(Dense), allocatable :: q_all(:)

    end type tau_emulators

    ! Neural networks and scale values saved within the scope of the module.
    ! Need to call initialize_tau_emulators to load weights and tables from disk.
    type(tau_emulators), save :: emulators
    real(r8), dimension(num_inputs, 2), save :: input_scale_values
    real(r8), dimension(num_outputs, 2), save :: output_scale_values
    contains
        subroutine load_quantile_scale_values
            ! Reads csv files containing means and standard deviations for the inputs and outputs
            ! of each neural network
            ! neural_net_path: Path to directory containing neural net netCDF files and scaling csv files.
            ! character(len=*), intent(in) :: neural_net_path
            integer :: i
            call load_scale_values(neural_net_path // "input_scale_values.csv", num_inputs, input_scale_values)
            call load_scale_values(neural_net_path // "output_scale_values.csv", num_outputs, output_scale_values)
        end subroutine load_mp_scale_values
        
        subroutine linear_interp(x_in, xs, ys, y_in)
            real(kind=8), dimension(:), intent(in) :: x_in
            real(kind=8), dimension(:), intent(in) :: xs
            real(kind=8), dimension(:), intent(in) :: ys
            real(kind=8), dimension(size(x_in, 1)), intent(out) :: y_in
            integer :: i, j, x_in_size, xs_size, x_pos
            x_in_size = size(x_in, 1)
            xs_size = size(xs, 1)
            do i = 1, x_in_size
                if (x_in(i) <= xs(1)) then
                    y_in(i) = ys(1)
                else if (x_in(i) >= xs(xs_size)) then
                    y_in(i) = ys(xs_size)
                else
                    j = 1
                    do while (xs(j) < x_in(i))
                       j = j + 1 
                    end do
                    y_in(i) = (ys(j-1) * (xs(j) - x_in(i)) + ys(j) * (x_in(i) - xs(j - 1))) / (xs(j) - xs(j - 1))
                end if
            end do
        end subroutine linear_interp
        
        subroutine quantile_transform(x_inputs, scale_values, x_transformed)
            real(kind=8), dimension(:, :), intent(in) :: x_inputs
            real(kind=8), dimension(:, size(x_inputs, 2) + 1), intent(in) :: scale_values
            real(kind=8), dimension(size(x_inputs, 1), size(x_inputs, 2)), intent(out) :: x_transformed
            integer :: j, x_size
            x_size = size(x_inputs, 1)
            scale_size = size(scale_values, 1)
            do j=1, size(x_inputs, 2)
                call linear_interp(x_inputs(1:x_size, j), scale_values(1:scale_size, j + 1), scale_values(1:scale_size, 1), x_transformed(1:x_size, j))
            end do   
        end subroutine quantile_transform
        
        subroutine quantile_inv_transform(x_inputs, scale_values, x_transformed)
            real(kind=8), dimension(:, :), intent(in) :: x_inputs
            real(kind=8), dimension(:, size(x_inputs, 2) + 1), intent(in) :: scale_values
            real(kind=8), dimension(size(x_inputs, 1), size(x_inputs, 2)), intent(out) :: x_transformed
            integer :: j, x_size
            x_size = size(x_inputs, 1)
            scale_size = size(scale_values, 1)
            do j=1, size(x_inputs, 2)
                call linear_interp(x_inputs(1:x_size, j), scale_values(1:scale_size, 1), scale_values(1:scale_size, j + 1), x_transformed(1:x_size, j))
            end do   
        end subroutine quantile_inv_transform

        subroutine initialize_tau_emulators
            ! Load neural network netCDF files and scaling values. Values are placed in to emulators,
            ! input_scale_values, and output_scale_values.
            ! Args:
            !   neural_net_path: Path to neural networks
            !
            !character(len=*), intent(in) :: neural_net_path
            ! Load each neural network from the neural net directory
            print*, "Begin loading neural nets"
            call init_neural_net(neural_net_path // "dnn_tau_all.nc", batch_size, emulators%q_all)
            print*, "End loading neural nets"
            ! Load the scale values from a csv file.
            call load_mp_scale_values
            print*, "Loaded neural nets scaling values"
        end subroutine initialize_tau_emulators


        subroutine tau_emulate_cloud_rain_interactions(qc, nc, qr, nr, rho, lcldm, n0r, pgam, &
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
            real(r8), dimension(1, num_inputs) :: nn_inputs, nn_quantile_inputs
            integer, dimension(1, 4) :: nn_quantile_outputs, nn_outputs
            real(r8) :: dt
            dt = 1800.0           
            do i=1, mgncol
                if ((qc(i) >= q_small) then
                    nn_inputs(1, 1) = qc(i)
                    nn_inputs(1, 2) = qr(i)
                    nn_inputs(1, 3) = nc(i)
                    nn_inputs(1, 4) = nr(i)
                    nn_inputs(1, 5) = rho(i)
                    nn_inputs(1, 6) = precip_frac(i)
                    nn_inputs(1, 7) = lcldm(i)
                    call quantile_transform(nn_inputs, input_scale_values, nn_quantile_inputs)
                    call neural_net_predict(nn_inputs_quantile(i:i+1), emulators%q_all, nn_quantile_outputs)
                    call quantile_transform(nn_quantile_outputs, output_scale_values, nn_outputs)
                    qc_tend(i) = (nn_outputs(1, 1) - qc(i)) / dt
                    qr_tend(i) = (nn_outputs(1, 2) - qr(i)) / dt
                    nc_tend(i) = (nn_outputs(1, 3) - nc(i)) / dt
                    nr_tend(i) = (nn_outputs(1, 4) - nr(i)) / dt
                else
                    qc_tend(i) = 0._r8
                    qr_tend(i) = 0._r8
                    nc_tend(i) = 0._r8
                    nr_tend(i) = 0._r8
                end if
            end do
        end subroutine tau_emulate_cloud_rain_interactions

end module tau_neural_net_batch
