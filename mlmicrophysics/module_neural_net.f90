module module_neural_net
    use netcdf
    implicit none
    type Dense
        integer :: input_size
        integer :: output_size
        integer :: batch_size
        integer :: activation
        real(kind=8), allocatable :: weights(:, :)
        real(kind=8), allocatable :: bias(:)
        real(kind=8), allocatable :: input(:, :)
        real(kind=8), allocatable :: output(:, :)
    end type Dense

contains

    subroutine apply_dense(input, layer, output)
        ! Description: Pass a set of input data through a single dense layer and nonlinear activation function
        !
        ! Inputs:
        ! layer (input): a single Dense object
        ! input (input): a 2D array where the rows are different examples and
        !   the columns are different model inputs
        !
        ! Output:
        ! output: output of the dense layer as a 2D array with shape (number of inputs, number of neurons)
        real(kind=8), dimension(:, :), intent(in) :: input
        type(Dense), intent(in) :: layer
        real(kind=8), dimension(size(input, 1), layer%output_size), intent(out) :: output
        real(kind=8), dimension(size(input, 1), layer%output_size) :: dense_output
        integer :: i, j, num_examples
        real(kind=8) :: alpha, beta
        !real(kind=8) :: time_start, time_end
        alpha = 1
        beta = 1
        dense_output = 0
        output = 0
        num_examples = size(input, 1)
        !call cpu_time(time_start)
        call dgemm('n', 'n', num_examples, layer%output_size, layer%input_size, &
            alpha, input, num_examples, layer%weights, layer%input_size, beta, dense_output, num_examples)
        !call cpu_time(time_end)
        !print *, num_examples, layer%output_size, layer%input_size
        !print *, "After dgemm ", dense_output(1, 1), time_end - time_start
        !call cpu_time(time_start)
        !dense_output = matmul(input, layer%weights)
        !call cpu_time(time_end)
        !print *, "After matmul", dense_output(1, 1), time_end - time_start
        do i=1, num_examples
            do j=1, layer%output_size
                dense_output(i, j) = dense_output(i, j) + layer%bias(j)
            end do
        end do
        call apply_activation(dense_output, layer%activation, output)
        return
    end subroutine apply_dense

    subroutine apply_activation(input, activation_type, output)
        ! Description: Apply a nonlinear activation function to a given array of input values.
        !
        ! Inputs:
        ! input: A 2D array
        ! activation_type: string describing which activation is being applied. If the activation
        !       type does not match any of the available options, the linear activation is applied.
        !       Currently supported activations are:
        !           relu
        !           elu
        !           selu
        !           sigmoid
        !           tanh
        !           softmax
        !           linear
        ! Output:
        ! output: Array of the same dimensions as input with the nonlinear activation applied.
        real(kind=8), dimension(:, :), intent(in) :: input
        integer, intent(in) :: activation_type
        real(kind=8), dimension(size(input, 1), size(input, 2)), intent(out) :: output

        real(kind=8), dimension(size(input, 1)) :: softmax_sum
        real(kind=8), parameter :: selu_alpha = 1.6732
        real(kind=8), parameter :: selu_lambda = 1.0507
        integer :: i, j
        select case (activation_type)
            case (1)
                where(input < 0)
                    output = 0
                elsewhere
                    output = input
                endwhere
            case (2)
                where(input < 0)
                    output = exp(input) - 1
                elsewhere
                    output = input
                end where
            case (3)
                where(input < 0)
                    output = selu_lambda * ( selu_alpha * exp(input) - selu_alpha)
                elsewhere
                    output = selu_lambda * input
                end where
            case (4)
                output = 1.0 / (1.0 + exp(-input))
            case (5)
                output = tanh(input)
            case (6)
                softmax_sum = sum(exp(input), dim=2) 
                do i=1, size(input, 1)
                    do j=1, size(input, 2)
                        output(i, j) = exp(input(i, j)) / softmax_sum(i)
                    end do
                end do
            case (7)
                output = input
            case default
                output = input
        end select
        return
    end subroutine apply_activation

    subroutine init_neural_net(filename, batch_size, neural_net_model)
        ! init_neuralnet
        ! Description: Loads dense neural network weights from a netCDF file and builds an array of
        ! Dense types from the weights and activations.
        !
        ! Input:
        ! filename: Full path to the netCDF file
        ! batch_size: number of items in single batch. Used to set intermediate array sizes.
        !
        ! Output:
        ! neural_net_model (output): array of Dense layers composing a densely connected neural network
        !
        character(len=*), intent(in) :: filename
        integer, intent(in) :: batch_size
        type(Dense), allocatable, intent(out) :: neural_net_model(:)
        integer :: ncid, num_layers_id, num_layers
        integer :: layer_names_var_id, i, layer_in_dimid, layer_out_dimid
        integer :: layer_in_dim, layer_out_dim
        integer :: layer_weight_var_id
        integer :: layer_bias_var_id

        character (len=8), allocatable :: layer_names(:)
        character (len=10) :: num_layers_dim_name = "num_layers"
        character (len=11) :: layer_name_var = "layer_names"
        character (len=11) :: layer_in_dim_name
        character (len=12) :: layer_out_dim_name
        character (len=10) :: activation_name
        real (kind=8), allocatable :: temp_weights(:, :)
        ! Open netCDF file
        call check(nf90_open(filename, nf90_nowrite, ncid))
        ! Get the number of layers in the neural network
        call check(nf90_inq_dimid(ncid, num_layers_dim_name, num_layers_id))
        call check(nf90_inquire_dimension(ncid, num_layers_id, &
                                          num_layers_dim_name, num_layers))
        call check(nf90_inq_varid(ncid, layer_name_var, layer_names_var_id))
        allocate(layer_names(num_layers))
        call check(nf90_get_var(ncid, layer_names_var_id, layer_names))
        print *, "load neural network " // filename
        allocate(neural_net_model(1:num_layers))
        ! Loop through each layer and load the weights, bias term, and activation function
        do i=1, num_layers
            layer_in_dim_name = trim(layer_names(i)) // "_in"
            layer_out_dim_name = trim(layer_names(i)) // "_out"
            layer_in_dimid = -1
            ! Get layer input and output dimensions
            call check(nf90_inq_dimid(ncid, trim(layer_in_dim_name), layer_in_dimid))
            call check(nf90_inquire_dimension(ncid, layer_in_dimid, layer_in_dim_name, layer_in_dim))
            call check(nf90_inq_dimid(ncid, trim(layer_out_dim_name), layer_out_dimid))
            call check(nf90_inquire_dimension(ncid, layer_out_dimid, layer_out_dim_name, layer_out_dim))
            call check(nf90_inq_varid(ncid, trim(layer_names(i)) // "_weights", &
                                      layer_weight_var_id))
            call check(nf90_inq_varid(ncid, trim(layer_names(i)) // "_bias", &
                                      layer_bias_var_id))
            neural_net_model(i)%input_size = layer_in_dim
            neural_net_model(i)%output_size = layer_out_dim
            neural_net_model(i)%batch_size = batch_size
            ! Fortran loads 2D arrays in the opposite order from Python/C, so I
            ! first load the data into a temporary array and then apply the
            ! transpose operation to copy the weights into the Dense layer
            allocate(neural_net_model(i)%weights(layer_in_dim, layer_out_dim))
            allocate(temp_weights(layer_out_dim, layer_in_dim))

            call check(nf90_get_var(ncid, layer_weight_var_id, &
                                    temp_weights))
            neural_net_model(i)%weights = transpose(temp_weights)
            allocate(neural_net_model(i)%input(batch_size, layer_in_dim))
            allocate(neural_net_model(i)%output(batch_size, layer_out_dim))
            deallocate(temp_weights)
            ! Load the bias weights
            allocate(neural_net_model(i)%bias(layer_out_dim))
            call check(nf90_get_var(ncid, layer_bias_var_id, &
                                    neural_net_model(i)%bias))
            ! Get the name of the activation function, which is stored as an attribute of the weights variable
            call check(nf90_get_att(ncid, layer_weight_var_id, "activation", &
                                    activation_name))
            select case (trim(activation_name))
                case ("relu")
                    neural_net_model(i)%activation = 1
                case ("elu")
                    neural_net_model(i)%activation = 2
                case ("selu")
                    neural_net_model(i)%activation = 3
                case ("sigmoid")
                    neural_net_model(i)%activation = 4
                case ("tanh")
                    neural_net_model(i)%activation = 5
                case ("softmax")
                    neural_net_model(i)%activation = 6
                case ("linear")
                    neural_net_model(i)%activation = 7
                case default
                    neural_net_model(i)%activation = 7
            end select
        end do
        print *, "finished loading neural network " // filename
        call check(nf90_close(ncid))

    end subroutine init_neural_net

    subroutine neural_net_predict(input, neural_net_model, prediction)
        ! neural_net_predict
        ! Description: generate prediction from neural network model for an arbitrary set of input values
        !
        ! Args:
        ! input (input): 2D array of input values. Each row is a separate instance and each column is a model input.
        ! neural_net_model (input): Array of type(Dense) objects
        ! prediction (output): The prediction of the neural network as a 2D array of dimension (examples, outputs)
        real(kind=8), intent(in) :: input(:, :)
        type(Dense), intent(inout) :: neural_net_model(:)
        real(kind=8), intent(out) :: prediction(size(input, 1), neural_net_model(size(neural_net_model))%output_size)
        integer :: bi, i
        integer :: batch_size
        integer :: input_size
        integer :: batch_index_size
        integer, allocatable :: batch_indices(:)
        input_size = size(input, 1)
        batch_size = neural_net_model(1)%batch_size
        batch_index_size = input_size / batch_size
        allocate(batch_indices(batch_index_size))
        i = 1
        do bi=batch_size, input_size, batch_size
            batch_indices(i) = bi
            i = i + 1
        end do
        batch_indices(batch_index_size) = input_size
        do bi=1, batch_index_size
            neural_net_model(1)%input = input(batch_indices(bi)-batch_size+1:batch_indices(bi), :)
            do i=1, size(neural_net_model) - 1
                call apply_dense(neural_net_model(i)%input, neural_net_model(i), neural_net_model(i)%output)
                neural_net_model(i + 1)%input = neural_net_model(i)%output
            end do
                call apply_dense(neural_net_model(i)%input, neural_net_model(i), neural_net_model(i)%output)
            prediction(batch_indices(bi)-batch_size + 1:batch_indices(bi), :) = neural_net_model(size(neural_net_model))%output
        end do
        deallocate(batch_indices)
    end subroutine neural_net_predict

    subroutine standard_scaler_transform(input_data, scale_values, transformed_data)
        ! Perform z-score normalization of input_data table. Equivalent to scikit-learn StandardScaler.
        !
        ! Inputs:
        !   input_data: 2D array where rows are examples and columns are variables
        !   scale_values: 2D array where rows are the input variables and columns are mean and standard deviation
        ! Output:
        !   transformed_data: 2D array with the same shape as input_data containing the transformed values.
        real(8), intent(in) :: input_data(:, :)
        real(8), intent(in) :: scale_values(:, :)
        real(8), intent(out) :: transformed_data(size(input_data, 1), size(input_data, 2))
        integer :: i
        if (size(input_data, 2) /= size(scale_values, 1)) then
            print *, "Size mismatch between input data and scale values", size(input_data, 2), size(scale_values, 1)
            stop 2
        end if
        do i=1, size(input_data, 2)
            transformed_data(:, i) = (input_data(:, i) - scale_values(i, 1)) / scale_values(i, 2)
        end do
    end subroutine standard_scaler_transform
    
    subroutine load_scale_values(filename, num_inputs, scale_values)
        character(len=*), intent(in) :: filename
        integer, intent(in) :: num_inputs
        real(8), intent(out) :: scale_values(num_inputs, 2)
        character(len=40) :: row_name
        integer :: isu, i
        isu = 2
        open(isu, file=filename, access="sequential", form="formatted")
        read(isu, "(A)")
        do i=1, num_inputs
            read(isu, *) row_name, scale_values(i, 1), scale_values(i, 2)
        end do
        close(isu)
    end subroutine load_scale_values

    
    subroutine standard_scaler_inverse_transform(input_data, scale_values, transformed_data)
        ! Perform inverse z-score normalization of input_data table. Equivalent to scikit-learn StandardScaler.
        !
        ! Inputs:
        !   input_data: 2D array where rows are examples and columns are variables
        !   scale_values: 2D array where rows are the input variables and columns are mean and standard deviation
        ! Output:
        !   transformed_data: 2D array with the same shape as input_data containing the transformed values.
        real(8), intent(in) :: input_data(:, :)
        real(8), intent(in) :: scale_values(:, :)
        real(8), intent(out) :: transformed_data(size(input_data, 1), size(input_data, 2))
        integer :: i
        if (size(input_data, 2) /= size(scale_values, 1)) then
            print *, "Size mismatch between input data and scale values", size(input_data, 2), size(scale_values, 1)
            stop 2
        end if
        do i=1, size(input_data, 2)
            transformed_data(:, i) = input_data(:, i) * scale_values(i, 2) + scale_values(i, 1)
        end do
    end subroutine standard_scaler_inverse_transform

    subroutine minmax_scaler_transform(input_data, scale_values, transformed_data)
        ! Perform min-max scaling of input_data table. Equivalent to scikit-learn MinMaxScaler.
        !
        ! Inputs:
        !   input_data: 2D array where rows are examples and columns are variables
        !   scale_values: 2D array where rows are the input variables and columns are min and max.
        ! Output:
        !   transformed_data: 2D array with the same shape as input_data containing the transformed values.
        real(8), intent(in) :: input_data(:, :)
        real(8), intent(in) :: scale_values(:, :)
        real(8), intent(out) :: transformed_data(size(input_data, 1), size(input_data, 2))
        integer :: i
        if (size(input_data, 2) /= size(scale_values, 1)) then
            print *, "Size mismatch between input data and scale values", size(input_data, 2), size(scale_values, 1)
            stop 2
        end if
        do i=1, size(input_data, 2)
            transformed_data(:, i) = (input_data(:, i) - scale_values(i, 1)) / (scale_values(i, 2) - scale_values(i ,1))
        end do
    end subroutine minmax_scaler_transform

    subroutine minmax_scaler_inverse_transform(input_data, scale_values, transformed_data)
        ! Perform inverse min-max scaling of input_data table. Equivalent to scikit-learn MinMaxScaler.
        !
        ! Inputs:
        !   input_data: 2D array where rows are examples and columns are variables
        !   scale_values: 2D array where rows are the input variables and columns are min and max.
        ! Output:
        !   transformed_data: 2D array with the same shape as input_data containing the transformed values.
        real(8), intent(in) :: input_data(:, :)
        real(8), intent(in) :: scale_values(:, :)
        real(8), intent(out) :: transformed_data(size(input_data, 1), size(input_data, 2))
        integer :: i
        if (size(input_data, 2) /= size(scale_values, 1)) then
            print *, "Size mismatch between input data and scale values", size(input_data, 2), size(scale_values, 1)
            stop 2
        end if
        do i=1, size(input_data, 2)
            transformed_data(:, i) = input_data(:, i) * (scale_values(i, 2) - scale_values(i ,1)) + scale_values(i, 1)
        end do
    end subroutine minmax_scaler_inverse_transform

    subroutine check(status)
        ! Check for netCDF errors
        integer, intent ( in) :: status
        if(status /= nf90_noerr) then
          print *, trim(nf90_strerror(status))
          stop 2
        end if
    end subroutine check

    subroutine print_2d_array(input_array)
        ! Print 2D array in pretty format
        real(kind=8), intent(in) :: input_array(:, :)
        integer :: i, j
        do i=1, size(input_array, 1)
            do j=1, size(input_array, 2)
                write(*, fmt="(1x,a,f6.3)", advance="no") "", input_array(i, j)
            end do
            write(*, *)
        end do
    end subroutine print_2d_array
end module module_neural_net


