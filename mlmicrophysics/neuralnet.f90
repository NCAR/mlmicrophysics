module neuralnet
    use netcdf
    implicit none
    type Dense
        integer :: input_size
        integer :: output_size
        real(kind=8), allocatable :: weights(:, :)
        real(kind=8), allocatable :: bias(:)
        character(len=10) :: activation
    end type Dense

contains

    function apply_dense(layer, input)
        ! Description: Pass a set of input data through a single dense layer and nonlinear activation function
        !
        ! Args:
        ! layer (input): a single Dense object
        ! input (input): a 2D array where the rows are different examples and
        !   the columns are different model inputs
        !
        ! Returns:
        ! The output of the dense layer as a 2D array with shape (number of inputs, number of neurons)
        type(Dense), intent(in) :: layer
        real(kind=8), dimension(:, :), intent(in) :: input
        real(kind=8), dimension(size(input, 1), layer%output_size) :: apply_dense
        integer :: i, j
        apply_dense = matmul(input, layer%weights)
        do i=1, size(input, 1)
            do j=1, size(layer%bias)
                apply_dense(i, j) = apply_dense(i, j) + layer%bias(j)
            end do
        end do
        apply_dense = apply_activation(apply_dense, layer%activation)
        return
    end function apply_dense

    function apply_activation(input, activation_type)
        ! Description: Apply a nonlinear activation function to a given array of input values.
        !
        ! Args:
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
        ! Returns:
        ! activation_type: Array of the same dimensions as input with the nonlinear activation applied.
        real(kind=8), dimension(:, :), intent(in) :: input
        character(len=10), intent(in) :: activation_type
        real(kind=8), dimension(size(input, 1)) :: softmax_sum
        real(kind=8), dimension(size(input, 1), size(input, 2)) :: apply_activation
        real(kind=8), parameter :: selu_alpha = 1.6732
        real(kind=8), parameter :: selu_lambda = 1.0507
        integer :: i, j
        select case (trim(activation_type))
            case ("relu")
                where(input < 0)
                    apply_activation = 0
                elsewhere
                    apply_activation = input
                endwhere
            case ("elu")
                where(input < 0)
                    apply_activation = exp(input) - 1
                elsewhere
                    apply_activation = input
                end where
            case ("selu")
                where(input < 0)
                    apply_activation = selu_lambda * ( selu_alpha * exp(input) - selu_alpha)
                elsewhere
                    apply_activation = selu_lambda * input
                end where
            case ("sigmoid")
                apply_activation = 1.0 / (1.0 + exp(-input))
            case ("tanh")
                apply_activation = tanh(input)
            case ("softmax")
                softmax_sum = sum(exp(input), dim=2) 
                do i=1, size(input, 1)
                    do j=1, size(input, 2)
                        apply_activation(i, j) = exp(input(i, j)) / softmax_sum(i)
                    end do
                end do
            case ("linear")
                apply_activation = input
            case default
                apply_activation = input
        end select
        return
    end function apply_activation

    subroutine init_neuralnet(filename, neural_net_model)
        ! init_neuralnet
        ! Description: Loads dense neural network weights from a netCDF file and builds an array of
        ! Dense types from the weights and activations.
        !
        ! Args:
        ! filename (input): Full path to the netCDF file
        ! neural_net_model (output): array of Dense layers composing a densely connected neural network
        !
        character(len=*), intent(in) :: filename
        type(Dense), allocatable, intent(out) :: neural_net_model(:)
        integer :: ncid, num_layers_id, num_layers
        integer :: layer_names_var_id, i, layer_in_dimid, layer_out_dimid
        integer :: layer_in_dim, layer_out_dim, s
        integer :: layer_weight_var_id
        integer :: layer_bias_var_id
        character(len=8), allocatable :: layer_names(:)
        character(len=10) :: num_layers_dim_name = "num_layers"
        character(len=11) :: layer_name_var = "layer_names"
        character(len=10) :: layer_name
        character (len=11) :: layer_in_dim_name
        character (len=12) :: layer_out_dim_name
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
        allocate(neural_net_model(num_layers))
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
            ! Fortran loads 2D arrays in the opposite order from Python/C, so I
            ! first load the data into a temporary array and then apply the
            ! transpose operation to copy the weights into the Dense layer
            allocate(neural_net_model(i)%weights(layer_in_dim, layer_out_dim))
            allocate(temp_weights(layer_out_dim, layer_in_dim))
            call check(nf90_get_var(ncid, layer_weight_var_id, &
                                    temp_weights))
            neural_net_model(i)%weights = transpose(temp_weights)
            deallocate(temp_weights)
            ! Load the bias weights
            allocate(neural_net_model(i)%bias(layer_out_dim))
            call check(nf90_get_var(ncid, layer_bias_var_id, &
                                    neural_net_model(i)%bias))
            ! Get the name of the activation function, which is stored as an attribute of the weights variable
            call check(nf90_get_att(ncid, layer_weight_var_id, "activation", &
                                    neural_net_model(i)%activation))
        end do
        print *, "finished loading neural network " // filename
        call check(nf90_close(ncid))

    end subroutine init_neuralnet

    subroutine neuralnet_predict(neuralnet_model, input, prediction)
        ! neuralnet_predict
        ! Description: generate prediction from neural network model for an arbitrary set of input values
        !
        ! Args:
        ! neuralnet_model (input): Array of type(Dense) objects
        ! input (input): 2D array of input values. Each row is a separate instance and each column is a model input.
        ! prediction (output): The prediction of the neural network as a 2D array of dimension (examples, outputs)

        type(Dense), intent(in) :: neuralnet_model(:)
        real(kind=8), intent(in) :: input(:, :)
        real(kind=8), allocatable, intent(out) :: prediction(:, :)
        real(kind=8), allocatable :: hidden_in(:, :)
        real(kind=8), allocatable :: hidden_out(:, :)
        integer :: i, j, k
        allocate(hidden_in(size(input, 1), size(input, 2)))
        hidden_in = input
        do i=1, size(neuralnet_model, 1)
            allocate(hidden_out(size(input, 1), &
                                neuralnet_model(i)%output_size))
            hidden_out = apply_dense(neuralnet_model(i), hidden_in)
            deallocate(hidden_in)
            allocate(hidden_in(size(hidden_out, 1), size(hidden_out, 2)))
            hidden_in = hidden_out
            deallocate(hidden_out)
        end do
        allocate(prediction(size(hidden_in, 1), size(hidden_in, 2)))
        prediction = hidden_in
        deallocate(hidden_in)
    end subroutine neuralnet_predict

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
end module neuralnet


