program test_neural_net
    use neuralnet
    character(len=100) :: filename
    integer :: i
    real(kind=8) :: input(3, 2)
    real(kind=8), allocatable :: prediction(:, :)
    type(Dense), allocatable :: model(:)
    filename = "out_model.nc"
    call init_neuralnet(filename, model)
    do i=1,size(model, 1)
        print *, model(i)%weights
    end do
    input = reshape((/ 0.0, 1.0, -1.0, 1.0, 0.5, 0.5 /), (/3, 2/), order=(/ 2, 1/))
    call neuralnet_predict(model, input, prediction)
    print *, "Prediction", prediction
end program test_neural_net
