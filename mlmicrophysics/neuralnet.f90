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
        type(Dense), intent(in) :: layer
        real(kind=8), dimension(:, :), intent(in) :: input
        real(kind=8), dimension(size(input, 1), layer%output_size) :: apply_dense
        integer :: i
        integer :: j
        apply_dense = matmul(input, transpose(layer%weights))
        do i=1, size(input, 1)
            do j=1, size(layer%bias)
                apply_dense(i, j) = apply_dense(i, j) + layer%bias(j)
            end do
        end do
        apply_dense = apply_activation(apply_dense, layer%activation)
        return
    end function apply_dense

    function apply_activation(input, activation_type)
        real(kind=8), dimension(:, :), intent(in) :: input
        character(len=10), intent(in) :: activation_type
        real(kind=8), dimension(size(input, 1), size(input,2)) :: apply_activation
        select case (trim(activation_type))
        case ("relu")
            where(input < 0)
                apply_activation = 0
            elsewhere
                apply_activation = input
            endwhere
        case ("sigmoid")
            apply_activation = 1.0 / (1.0 + exp(-input))
        case ("tanh")
            apply_activation = tanh(input)
        case ("linear")
            apply_activation = input
        end select
        return
    end function apply_activation

    subroutine init_neuralnet(filename, neural_net_model)
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
        print *, "open file"
        call check(nf90_open(filename, nf90_nowrite, ncid))
        print *, "get num layers"
        call check(nf90_inq_dimid(ncid, num_layers_dim_name, num_layers_id))
        call check(nf90_inquire_dimension(ncid, num_layers_id, &
                                          num_layers_dim_name, num_layers))
        print *, "get layer names var id"
        call check(nf90_inq_varid(ncid, layer_name_var, layer_names_var_id))
        print *, "allocate layer names ", num_layers
        print *, "get layer names ", num_layers
        allocate(layer_names(num_layers))
        call check(nf90_get_var(ncid, layer_names_var_id, layer_names))
        print *, "load neural network", num_layers
        allocate(neural_net_model(num_layers))
        do i=1, num_layers
            layer_in_dim_name = trim(layer_names(i)) // "_in"
            layer_out_dim_name = trim(layer_names(i)) // "_out"
            print *, layer_in_dim_name
            print *, layer_out_dim_name
            layer_in_dimid = -1
            call check(nf90_inq_dimid(ncid, trim(layer_in_dim_name), layer_in_dimid))
            print *, layer_in_dim_name, layer_in_dimid
            call check(nf90_inquire_dimension(ncid, layer_in_dimid, layer_in_dim_name, layer_in_dim))
            call check(nf90_inq_dimid(ncid, trim(layer_out_dim_name), layer_out_dimid))
            call check(nf90_inquire_dimension(ncid, layer_out_dimid, layer_out_dim_name, layer_out_dim))
            print *, trim(layer_names(i)) // "_weights"
            call check(nf90_inq_varid(ncid, trim(layer_names(i)) // "_weights", &
                                      layer_weight_var_id))
            print *, trim(layer_names(i)) // "_bias"
            call check(nf90_inq_varid(ncid, trim(layer_names(i)) // "_bias", &
                                      layer_bias_var_id))
            print *, layer_in_dim, layer_out_dim
            neural_net_model(i)%input_size = layer_in_dim
            neural_net_model(i)%output_size = layer_out_dim
            allocate(neural_net_model(i)%weights(layer_out_dim, layer_in_dim))
            allocate(neural_net_model(i)%bias(layer_out_dim))
            print *, "get weights"
            call check(nf90_get_var(ncid, layer_weight_var_id, &
                                    neural_net_model(i)%weights))
            print *, "get bias"
            call check(nf90_get_var(ncid, layer_bias_var_id, &
                                    neural_net_model(i)%bias))
            print *, "get activation"
            call check(nf90_get_att(ncid, layer_weight_var_id, "activation", &
                                    neural_net_model(i)%activation))
            print *, neural_net_model(i)%activation
            print *, "done loop"
        end do
        print *, "loaded"
        call check(nf90_close(ncid))

    end subroutine init_neuralnet

    subroutine neuralnet_predict(neuralnet_model, input, prediction)
        type(Dense), intent(in) :: neuralnet_model(:)
        real(kind=8), intent(in) :: input(:, :)
        real(kind=8), allocatable, intent(out) :: prediction(:, :)
        real(kind=8), allocatable :: hidden_in(:, :)
        real(kind=8), allocatable :: hidden_out(:, :)
        integer :: i, j, k
        allocate(hidden_in(size(input, 1), size(input, 2)))
        do i=1, size(neuralnet_model, 1)
            allocate(hidden_out(size(input, 1), &
                                neuralnet_model(i)%output_size))
            hidden_out = apply_dense(neuralnet_model(i), hidden_in)
            deallocate(hidden_in)
            allocate(hidden_in(size(hidden_out, 1), size(hidden_out, 2)))
            do j=1, size(hidden_out, 1)
                do k=1, size(hidden_out, 2)
                    hidden_in(j, k) = hidden_out(j, k)
                end do
            end do
            deallocate(hidden_out)
        end do
        allocate(prediction(size(hidden_in, 1), size(hidden_in, 2)))
        do j=1, size(hidden_in, 1)
            do k=1, size(hidden_in, 2)
                prediction(j, k) = hidden_in(j, k)
            end do
        end do
    end subroutine neuralnet_predict

    subroutine check(status)
        integer, intent ( in) :: status

        if(status /= nf90_noerr) then
          print *, trim(nf90_strerror(status))
          stop 2
        end if
    end subroutine check
end module neuralnet


