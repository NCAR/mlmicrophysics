from mlmicrophysics.models import DenseNeuralNetwork
import numpy as np

dnn = DenseNeuralNetwork(hidden_layers=2, hidden_neurons=10, inputs=2, epochs=20, batch_size=1)
dnn.build_neural_network()
x = np.array([[0, 1], [-1, 1], [0.5, 0.5]])
y = 0.2 * x[:, 0] ** 2 - 3 * x[:, 1] ** 3
print(x)
print(y)
dnn.fit(x, y)
dnn.save_fortran_model("out_model.nc")