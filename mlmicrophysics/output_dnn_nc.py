from mlmicrophysics.models import DenseNeuralNetwork
import keras.backend as K
import numpy as np
seed = 21243
np.random.seed(seed)
sess = K.tf.Session()
K.set_session(sess)
K.tf.set_random_seed(seed)
dnn = DenseNeuralNetwork(hidden_layers=2, hidden_neurons=10, inputs=2, epochs=20, batch_size=1)
dnn.build_neural_network()
x = np.array([[0, 1], [-1, 1], [0.5, 0.5]])
y = 0.2 * x[:, 0] ** 2 - 3 * x[:, 1] ** 3
print(x)
print(y)
dnn.fit(x, y)
print(dnn.model.summary())
for layer in dnn.model.layers:
    if "dense" in layer.name:
        weights = layer.get_weights()[0]
        print("Shape", weights.shape)
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                print("{0:+0.3f} ".format(weights[i, j]), end="", flush=True)
            print("")
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        print("{0:+0.2f} ".format(x[i, j]), end="", flush=True)
    print("")
preds = dnn.predict(x)
print(preds)
dnn.save_fortran_model("out_model.nc")
