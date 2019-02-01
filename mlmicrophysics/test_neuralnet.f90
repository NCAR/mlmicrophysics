program test_neural_net
    use neuralnet
    character(len=100) :: filename
    integer :: i, j, k
    real(kind=8) :: input(3, 2), activated_input(3, 2)
    real(kind=8), allocatable :: prediction(:, :)
    type(Dense), allocatable :: model(:)
    filename = "out_model.nc"
    call init_neuralnet(filename, model)
    do i=1,size(model, 1)
        print *, "layer", i
        call print_2d_array(model(i)%weights)
    end do
    input = reshape((/ 0.0, 1.0, -1.0, 1.0, 0.5, 0.5 /), (/3, 2/), order=(/ 2, 1/))
    print *, "Input"
    call print_2d_array(input)
    call neuralnet_predict(model, input, prediction)
    print *, "Prediction"
    call print_2d_array(prediction)
end program test_neural_net
