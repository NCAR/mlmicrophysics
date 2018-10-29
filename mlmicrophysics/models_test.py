import unittest
from .models import DenseNeuralNetwork, DenseGAN
from keras.layers import Dense
import numpy as np


class TestModels(unittest.TestCase):
    def setUp(self):
        return

    def test_dense_neural_network(self):
        net_default = DenseNeuralNetwork()
        self.assertEqual(net_default.inputs, net_default.model.input.shape[1].value,
                         "DenseNeuralNetwork model input does not match specified size.")
        self.assertEqual(net_default.outputs, net_default.model.output.shape[1].value,
                         "DenseNeuralNetwork model output doe not match specified size.")
        dense_count = sum([type(layer) == Dense for layer in net_default.model.layers])
        self.assertEqual(dense_count - 1, net_default.hidden_layers,
                         "Number of dense hidden layers does not match the specified number of hidden layers.")
        random_x = np.random.normal(size=(2048, net_default.inputs))
        random_y = random_x ** 2
        net_default.fit(random_x, random_y)
        net_default.predict(random_x)

    def test_gan(self):
        num_inputs = 1
        random_x = np.random.normal(size=(2048, num_inputs))
        gan_default = DenseGAN(inputs=num_inputs)
        random_y = random_x ** 2
        gan_default.fit(random_x, random_y)
        predictions = gan_default.predict(random_x)
        self.assertEqual(predictions.shape[0], random_y.shape[0],
                         "Length of GAN predictions does not match number of examples")

