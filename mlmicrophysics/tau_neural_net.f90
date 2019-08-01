module tau_neural_net
    use neuralnet
    implicit none
    integer, parameter, public :: r8 = selected_real_kind(12)
    integer, parameter, public :: i8 = selected_int_kind(18)
    character(len=*), parameter :: neural_net_path = "/glade/p/cisl/aiml/dgagne/cam_run5_models_20190524/"
    type tau_emulators
        type(Dense), allocatable :: qr_classifier(:)
        type(Dense), allocatable :: qr_regressor(:)
        type(Dense), allocatable :: nr_classifier(:)
        type(Dense), allocatable :: nr_neg_regressor(:)
        type(Dense), allocatable :: nr_pos_regressor(:)
        type(Dense), allocatable :: nc_classifier(:)
        type(Dense), allocatable :: nc_regressor(:)
    end type tau_emulators

    ! Neural networks and scale values saved within the scope of the module.
    ! Need to call initialize_tau_emulators to load weights and tables from disk.
    type(tau_emulators), save :: emulators
    real(r8), dimension(5, 2), save :: input_scale_values
    real(r8), dimension(4, 2), save :: output_scale_values
    contains
        subroutine load_scale_values
            ! Reads csv files containing means and standard deviations for the inputs and outputs
            ! of each neural network
            ! neural_net_path: Path to directory containing neural net netCDF files and scaling csv files.
            ! character(len=*), intent(in) :: neural_net_path
            integer, parameter :: num_inputs=5, num_outputs = 4
            integer :: isu, osu, i
            character(len=13) :: row_name
            isu = 20
            osu = 25
            open(isu, file=neural_net_path // "input_scale_values.csv", access="sequential", form="formatted")
            read(isu, "(A)")
            do i=1, num_inputs
                read(isu, *) row_name, input_scale_values(i, 1), input_scale_values(i, 2)
            end do
            close(isu)
            open(osu, file=neural_net_path // "output_scale_values.csv", access="sequential", form="formatted")
            read(osu, "(A)")
            do i=1, num_outputs
                read(osu, *)  row_name, output_scale_values(i, 1), output_scale_values(i, 2)
            end do
            close(osu)
            print *, "Input Scale Values"
            do i=1, num_inputs
                print *, input_scale_values(i, 1), input_scale_values(i, 2)
            end do
            print *, "Output Scale Values"
            do i=1, num_outputs
                print *, output_scale_values(i, 1), output_scale_values(i, 2)
            end do
        end subroutine load_scale_values

        subroutine initialize_tau_emulators
            ! Load neural network netCDF files and scaling values. Values are placed in to emulators,
            ! input_scale_values, and output_scale_values.
            ! Args:
            !   neural_net_path: Path to neural networks
            !
            !character(len=*), intent(in) :: neural_net_path
            ! Load each neural network from the neural net directory
            call init_neuralnet(neural_net_path // "dnn_qr_class_fortran.nc", emulators%qr_classifier)
            call init_neuralnet(neural_net_path // "dnn_qr_pos_fortran.nc", emulators%qr_regressor)
            call init_neuralnet(neural_net_path // "dnn_nr_class_fortran.nc", emulators%nr_classifier)
            call init_neuralnet(neural_net_path // "dnn_nr_neg_fortran.nc", emulators%nr_neg_regressor)
            call init_neuralnet(neural_net_path // "dnn_nr_pos_fortran.nc", emulators%nr_pos_regressor)
            call init_neuralnet(neural_net_path // "dnn_nc_class_fortran.nc", emulators%nc_classifier)
            call init_neuralnet(neural_net_path // "dnn_nc_pos_fortran.nc", emulators%nc_regressor)
            ! Load the scale values from a csv file.
            call load_scale_values
        end subroutine initialize_tau_emulators


        subroutine tau_emulate_cloud_rain_interactions(qc, nc, qr, nr, rho, lamc, lamr, lcldm, n0r, pgam, &
                                                       precip_frac, q_small, mgncol, qc_tend, qr_tend, nc_tend, nr_tend)
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
            real(r8), dimension(mgncol), intent(in) :: qc, qr, nc, nr, rho, lamc, lamr, lcldm, n0r, pgam, precip_frac
            real(r8), intent(in) :: q_small
            real(r8), dimension(mgncol), intent(out) :: qc_tend, qr_tend, nc_tend, nr_tend
            integer(i8) :: i, j, qr_class, nc_class, nr_class
            integer, parameter :: num_inputs = 11
            real(r8), dimension(1, num_inputs) :: nn_inputs, nn_inputs_log_norm
            integer, dimension(num_inputs) :: log_inputs
            real(r8), dimension(:, :), allocatable :: nz_qr_prob, nz_nr_prob, nz_nc_prob
            real(r8), dimension(:, :), allocatable :: qr_tend_log_norm, nc_tend_log_norm, nr_tend_log_norm
            do i=1, mgncol
                if ((qc(i) >= q_small) .or. (qr(i) >= q_small)) then
                    nn_inputs = reshape((/ qc(i), nc(i), qr(i), nr(i), rho(i), &
                            lamc(i), lamr(i), lcldm(i), n0r(i), pgam(i), precip_frac(i) /), (/ 1, num_inputs /))
                    log_inputs = (/ 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0 /)
                    do j=1, num_inputs
                        if (log_inputs(j) == 1) then
                            nn_inputs_log_norm(1, j) = (log10(max(nn_inputs(1, j), 1e-40)) - input_scale_values(j, 1)) / &
                            input_scale_values(j, 2)
                        else
                            nn_inputs_log_norm(1, j) = (nn_inputs(1, j) - input_scale_values(j, 1)) / &
                            input_scale_values(j, 2)
                        end if
                    end do
                    ! calculate the qr and qc tendencies
                    call neuralnet_predict(emulators%qr_classifier, nn_inputs_log_norm, nz_qr_prob)
                    qr_class = maxloc(pack(nz_qr_prob, .true.), 1)
                    print*, "qr_prob", nz_qr_prob, qr_class
                    if (qr_class == 1) then
                        qr_tend(i) = 0._r8
                        qc_tend(i) = 0._r8
                    else
                        call neuralnet_predict(emulators%qr_regressor, nn_inputs_log_norm, qr_tend_log_norm)
                        qr_tend(i) = 10 ** (qr_tend_log_norm(1, 1) * output_scale_values(1, 2) + output_scale_values(1, 1))
                        qc_tend(i) = -qr_tend(i)
                    end if
                    ! calculate the nc tendency
                    call neuralnet_predict(emulators%qr_classifier, nn_inputs_log_norm, nz_nc_prob)
                    nc_class = maxloc(pack(nz_nc_prob, .true.), 1)
                    if (nc_class == 1) then
                        nc_tend(i) = 0._r8
                    else
                        call neuralnet_predict(emulators%nc_regressor, nn_inputs_log_norm, nc_tend_log_norm)
                        nc_tend(i) = -10 ** (nc_tend_log_norm(1, 1) * output_scale_values(2, 2) + &
                                output_scale_values(2, 1))
                    end if
                    ! calculate the nr tendency
                    call neuralnet_predict(emulators%nr_classifier, nn_inputs_log_norm, nz_nr_prob)
                    nr_class = maxloc(pack(nz_nr_prob, .true.), 1)
                    print*, "nr_prob", nz_nr_prob, nr_class
                    ! print *, "Classes", qr_class, nc_class, nr_class
                    if (nr_class == 2) then
                        nr_tend(i) = 0._r8
                    elseif (nr_class == 1) then
                        call neuralnet_predict(emulators%nr_neg_regressor, nn_inputs_log_norm, nr_tend_log_norm)
                        nr_tend(i) = -10 ** (nr_tend_log_norm(1, 1) * output_scale_values(3, 2) + &
                                output_scale_values(3, 1))
                    else
                        call neuralnet_predict(emulators%nr_pos_regressor, nn_inputs_log_norm, nr_tend_log_norm)
                        nr_tend(i) = 10 ** (nr_tend_log_norm(1, 1) * output_scale_values(4, 2) + &
                                output_scale_values(4, 1))
                    end if
                else
                    qc_tend(i) = 0._r8
                    qr_tend(i) = 0._r8
                    nc_tend(i) = 0._r8
                    nr_tend(i) = 0._r8
                end if
            end do
        end subroutine tau_emulate_cloud_rain_interactions

end module tau_neural_net
