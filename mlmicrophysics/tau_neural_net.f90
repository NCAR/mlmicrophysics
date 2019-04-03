module tau_neural_net
    use neuralnet
    implicit none
    integer, parameter, public :: r8 = selected_real_kind(12)
    integer, parameter, public :: i8 = selected_int_kind(18)

    type tau_emulators
        type(Dense), allocatable :: qr_classifier(:)
        type(Dense), allocatable :: qr_regressor(:)
        type(Dense), allocatable :: nr_classifier(:)
        type(Dense), allocatable :: nr_neg_regressor(:)
        type(Dense), allocatable :: nr_pos_regressor(:)
        type(Dense), allocatable :: nc_regressor(:)
    end type tau_emulators

    contains

    subroutine intialize_tau_emulators(neural_net_path, emulators, scale_values)
        character(len=*), intent(in) :: neural_net_path
        type(tau_emulators), intent(out) :: emulators
        real(r8), dimension(9, 2), intent(out) :: scale_values
        ! Load each neural network from the neural net directory
        call init_neuralnet(neural_net_path // "qr_classifier_dnn.nc", emulators%qr_classifier)
        call init_neuralnet(neural_net_path // "qr_regressor_dnn.nc", emulators%qr_regressor)
        call init_neuralnet(neural_net_path // "nr_classifier_dnn.nc", emulators%nr_classifier)
        call init_neuralnet(neural_net_path // "nr_neg_regressor_dnn.nc", emulators%nr_neg_regressor)
        call init_neuralnet(neural_net_path // "nr_pos_regressor_dnn.nc", emulators%nr_pos_regressor)
        call init_neuralnet(neural_net_path // "nc_regressor_dnn.nc", emulators%nc_regressor)
        ! Load the scale values from a csv file.
    end subroutine intialize_tau_emulators

    subroutine tau_emulate_cloud_rain_interactions(qc, nc, qr, nr, rho, q_small, emulators, scale_values, &
                                      qc_tend, qr_tend, nc_tend, nr_tend, mgncol)
        integer(i8), intent(in) :: mgncol
        real(r8), dimension(mgncol), intent(in) :: qc, qr, nc, nr, rho
        real(r8), intent(in) :: q_small
        type(tau_emulators), intent(in) :: emulators
        real(r8), dimension(9, 2), intent(in) :: scale_values

        real(r8), dimension(mgncol), intent(out) :: qc_tend, qr_tend, nc_tend, nr_tend
        integer(i8) :: i, j, qr_class, nr_class
        integer :: num_inputs = 5
        real(r8), dimension(1, 5) :: nn_inputs, nn_inputs_log_norm
        real(r8), dimension(:, :), allocatable :: nz_qr_prob, nz_nr_prob
        real(r8), dimension(:, :), allocatable :: qr_tend_log_norm, nc_tend_log_norm, nr_tend_log_norm
        do i=1, mgncol
            if (qc(i) >= q_small) then
                nn_inputs = reshape((/ qc(i), qr(i), nc(i), nr(i), rho(i) /), (/ 1, 5 /))
                do j=1, num_inputs
                    nn_inputs_log_norm(1, j) = (log10(max(nn_inputs(1, j), q_small)) - scale_values(j, 1)) / scale_values(j, 2)
                end do
                ! calculate the qr and qc tendencies
                call neuralnet_predict(emulators%qr_classifier, nn_inputs_log_norm, nz_qr_prob)
                qr_class = maxloc(pack(nz_qr_prob, .true.), 1)
                if (qr_class == 1) then
                    qr_tend(i) = 0._r8
                    qc_tend(i) = 0._r8
                else
                    call neuralnet_predict(emulators%qr_regressor, nn_inputs_log_norm, qr_tend_log_norm)
                    qr_tend(i) = 10 ** (qr_tend_log_norm(1, 1) * scale_values(num_inputs + 1, 2) + scale_values(num_inputs + 1, 1))
                    qc_tend(i) = -qr_tend(i)
                end if
                ! calculate the nc tendency
                call neuralnet_predict(emulators%nc_regressor, nn_inputs_log_norm, nc_tend_log_norm)
                nc_tend(i) = 10 ** (nc_tend_log_norm(1, 1) * scale_values(num_inputs + 2, 2) + scale_values(num_inputs + 2, 1))
                ! calculate the nr tendency
                call neuralnet_predict(emulators%nr_classifier, nn_inputs_log_norm, nz_nr_prob)
                nr_class = maxloc(pack(nz_nr_prob, .true.), 1)
                if (nr_class == 2) then
                    nr_tend(i) = 0._r8
                elseif (nr_class == 1) then
                    call neuralnet_predict(emulators%nr_neg_regressor, nn_inputs_log_norm, nr_tend_log_norm)
                    nr_tend(i) = -10 ** (nr_tend_log_norm(1, 1) * scale_values(num_inputs + 3, 2) + &
                            scale_values(num_inputs + 3, 1))
                else
                    call neuralnet_predict(emulators%nr_pos_regressor, nn_inputs_log_norm, nr_tend_log_norm)
                    nr_tend(i) = 10 ** (nr_tend_log_norm(1, 1) * scale_values(num_inputs + 4, 2) + &
                            scale_values(num_inputs + 4, 1))
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