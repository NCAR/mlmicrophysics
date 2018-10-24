from mlmicrophysics.models import DenseGAN, DenseNeuralNetwork
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

x = np.random.uniform(-3, 3, size=(16384, 1))
#y = 3 * x[:, 0:1] ** 2 - 2 * x[:, 0:1] + 5
y = np.exp(-x[:, 0:1])
y += np.random.normal(scale=0.1, size=y.shape)
y_norm = (y - y.mean()) / y.std()
print(y_norm.mean(), y_norm.std())
dg = DenseGAN(inputs=1, hidden_layers=4, hidden_neurons=32, activation="selu",
              batch_size=32, epochs=60, verbose=1,
              use_noise=True, noise_sd=0.01, report_frequency=10,
              batch_norm_output=True, optimizer=Adam(lr=0.001, beta_1=0.8, beta_2=0.9))
loss_history = dg.fit(x, y_norm)
plt.figure(figsize=(6, 4))
plt.plot(loss_history["gen_loss"])
plt.plot(loss_history["disc_loss"])
plt.show()
x_range = np.arange(-3, 3.1, 0.1)
num_samples = 1000
x_samples = np.ones((x_range.shape[0], num_samples)) * x_range.reshape(-1, 1)
y_samples = dg.gen_predict_func([x_samples.reshape(x_samples.size, 1), 1])[0].reshape(x_samples.shape)
y_pred = np.median(y_samples, axis=1)
dnn = DenseNeuralNetwork(hidden_neurons=32,
                         hidden_layers=4, activation="selu", use_noise=True, noise_sd=0.1, l2_weight=0,
                         batch_size=32, epochs=30, verbose=1)
dnn.fit(x, y_norm)
y_dnn = dnn.predict(x_range)
plt.figure(figsize=(6, 4))
plt.scatter(x[:, 0], y_norm, 2, 'k')
plt.boxplot(y_samples.T, widths=0.05, positions=x_range, whis=[5, 95], sym='',
            manage_xticks=False, medianprops={"color": "red"})
plt.plot(x_range, y_dnn, 'bo-')
plt.show()

